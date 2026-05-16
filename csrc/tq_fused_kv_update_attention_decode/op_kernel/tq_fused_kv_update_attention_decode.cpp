/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 * Decode-only M4 TurboQuant fused KV update + attention:
 *   writes the current K/V sidecar cache for future steps while attention reads
 *   compressed historical K/V plus the dense current token directly.
 */

#include "kernel_operator.h"

using namespace AscendC;

struct TqFusedKvUpdateAttentionDecodeTilingData {
    uint32_t batch;
    uint32_t numHeads;
    uint32_t numKvHeads;
    uint32_t qPerKv;
    uint32_t blockSize;
    uint32_t maxBlocksPerSeq;
    uint32_t maxSeqLen;
    uint32_t scoreTileLen;
    uint32_t groupedQ;
    uint32_t skipCacheUpdate;
    uint32_t debugMode;
    uint32_t pretransformedQuery;
    uint32_t historyPartitions;
    uint32_t historyPartitionPhase;
    uint32_t transformMode;
    uint32_t headDim;
    uint32_t kPackedCols;
    uint32_t kQjlCols;
    uint32_t vPackedCols;
    uint32_t kStage1Bits;
    uint32_t vBits;
    uint32_t numCore;
    float scale;
    float correction;
};

template <typename QueryT, typename KeyT, typename ValueT>
class KernelTqFusedKvUpdateAttentionDecode {
public:
    __aicore__ inline KernelTqFusedKvUpdateAttentionDecode() {}

    __aicore__ inline void Init(
        GM_ADDR query,
        GM_ADDR key,
        GM_ADDR value,
        GM_ADDR slotMapping,
        GM_ADDR kPackedIdx,
        GM_ADDR kPackedQjl,
        GM_ADDR kGamma,
        GM_ADDR kNorm,
        GM_ADDR vPackedIdx,
        GM_ADDR vNorm,
        GM_ADDR blockTable,
        GM_ADDR oldSeqLens,
        GM_ADDR kRotation,
        GM_ADDR kQjlQueryMatrix,
        GM_ADDR kQjlProjT,
        GM_ADDR kBoundary,
        GM_ADDR vRotation,
        GM_ADDR vRotationT,
        GM_ADDR vBoundary,
        GM_ADDR kCodebook,
        GM_ADDR vCodebook,
        GM_ADDR scratch,
        GM_ADDR out,
        const TqFusedKvUpdateAttentionDecodeTilingData* tiling)
    {
        queryGm_.SetGlobalBuffer((__gm__ QueryT*)query);
        keyGm_.SetGlobalBuffer((__gm__ KeyT*)key);
        valueGm_.SetGlobalBuffer((__gm__ ValueT*)value);
        slotMappingGm_.SetGlobalBuffer((__gm__ int64_t*)slotMapping);
        kPackedIdxGm_.SetGlobalBuffer((__gm__ uint8_t*)kPackedIdx);
        kPackedQjlGm_.SetGlobalBuffer((__gm__ uint8_t*)kPackedQjl);
        kGammaGm_.SetGlobalBuffer((__gm__ float*)kGamma);
        kNormGm_.SetGlobalBuffer((__gm__ float*)kNorm);
        vPackedIdxGm_.SetGlobalBuffer((__gm__ uint8_t*)vPackedIdx);
        vNormGm_.SetGlobalBuffer((__gm__ float*)vNorm);
        blockTableGm_.SetGlobalBuffer((__gm__ int32_t*)blockTable);
        oldSeqLensGm_.SetGlobalBuffer((__gm__ int32_t*)oldSeqLens);
        kRotationGm_.SetGlobalBuffer((__gm__ float*)kRotation);
        kQjlQueryMatrixGm_.SetGlobalBuffer((__gm__ float*)kQjlQueryMatrix);
        kQjlProjTGm_.SetGlobalBuffer((__gm__ float*)kQjlProjT);
        kBoundaryGm_.SetGlobalBuffer((__gm__ float*)kBoundary);
        vRotationGm_.SetGlobalBuffer((__gm__ float*)vRotation);
        vRotationTGm_.SetGlobalBuffer((__gm__ float*)vRotationT);
        vBoundaryGm_.SetGlobalBuffer((__gm__ float*)vBoundary);
        kCodebookGm_.SetGlobalBuffer((__gm__ float*)kCodebook);
        vCodebookGm_.SetGlobalBuffer((__gm__ float*)vCodebook);
        scratchGm_.SetGlobalBuffer((__gm__ float*)scratch);
        outGm_.SetGlobalBuffer((__gm__ float*)out);

        batch_ = tiling->batch;
        numHeads_ = tiling->numHeads;
        numKvHeads_ = tiling->numKvHeads;
        qPerKv_ = tiling->qPerKv;
        blockSize_ = tiling->blockSize;
        maxBlocksPerSeq_ = tiling->maxBlocksPerSeq;
        maxSeqLen_ = tiling->maxSeqLen;
        scoreTileLen_ = tiling->scoreTileLen;
        groupedQ_ = tiling->groupedQ;
        skipCacheUpdate_ = tiling->skipCacheUpdate;
        debugMode_ = tiling->debugMode;
        pretransformedQuery_ = tiling->pretransformedQuery;
        historyPartitions_ = tiling->historyPartitions;
        historyPartitionPhase_ = tiling->historyPartitionPhase;
        transformMode_ = tiling->transformMode;
        headDim_ = tiling->headDim;
        kPackedCols_ = tiling->kPackedCols;
        kQjlCols_ = tiling->kQjlCols;
        vPackedCols_ = tiling->vPackedCols;
        kStage1Bits_ = tiling->kStage1Bits;
        vBits_ = tiling->vBits;
        numCore_ = tiling->numCore;
        scale_ = tiling->scale;
        correction_ = tiling->correction;

        pipe_.InitBuffer(expBuf_, 64U * sizeof(float));
        LoadSmallParams();
    }

    template <typename InT>
    __aicore__ inline float InputToFloat(InT value)
    {
        if constexpr (IsSameType<InT, bfloat16_t>::value) {
            return ToFloat(value);
        } else {
            return static_cast<float>(value);
        }
    }

    __aicore__ inline void SToVSync()
    {
        event_t eventId = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);
    }

    __aicore__ inline void VToSSync()
    {
        event_t eventId = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
    }

    __aicore__ inline float ExpScalar(float value)
    {
        LocalTensor<float> expLocal = expBuf_.Get<float>();
        expLocal.SetValue(0, value);
        SToVSync();
        Exp(expLocal, expLocal, 1U);
        VToSSync();
        return expLocal.GetValue(0);
    }

    __aicore__ inline float SqrtScalar(float value)
    {
        LocalTensor<float> sqrtLocal = expBuf_.Get<float>();
        sqrtLocal.SetValue(0, value);
        SToVSync();
        Sqrt(sqrtLocal, sqrtLocal, 1U);
        VToSSync();
        return sqrtLocal.GetValue(0);
    }

    __aicore__ inline uint32_t ExtractBits(
        GlobalTensor<uint8_t>& packed,
        uint64_t packedBase,
        uint32_t d,
        uint32_t bits)
    {
        uint32_t bitPos = d * bits;
        uint32_t byteId = bitPos >> 3;
        uint32_t bitOff = bitPos & 7U;
        uint16_t value = static_cast<uint16_t>(packed.GetValue(packedBase + byteId));
        if (bitOff + bits > 8U) {
            uint16_t high = static_cast<uint16_t>(
                packed.GetValue(packedBase + byteId + 1U));
            value = static_cast<uint16_t>(value | (high << 8));
        }
        uint32_t mask = (1U << bits) - 1U;
        return (value >> bitOff) & mask;
    }

    template <uint32_t Bits>
    __aicore__ inline uint32_t ExtractBitsConst(
        GlobalTensor<uint8_t>& packed,
        uint64_t packedBase,
        uint32_t d)
    {
        uint32_t bitPos = d * Bits;
        uint32_t byteId = bitPos >> 3;
        uint32_t bitOff = bitPos & 7U;
        uint16_t value = static_cast<uint16_t>(packed.GetValue(packedBase + byteId));
        if constexpr (Bits == 3U) {
            if (bitOff > 5U) {
                uint16_t high = static_cast<uint16_t>(
                    packed.GetValue(packedBase + byteId + 1U));
                value = static_cast<uint16_t>(value | (high << 8));
            }
            return (value >> bitOff) & 7U;
        } else {
            return (value >> bitOff) & ((1U << Bits) - 1U);
        }
    }

    template <typename InT>
    __aicore__ inline void LoadCurrentVector(
        GlobalTensor<InT>& tensor,
        uint32_t b,
        uint32_t kvHead)
    {
        uint64_t base =
            (static_cast<uint64_t>(b) * numKvHeads_ + kvHead) * headDim_;
        for (uint32_t d = 0; d < headDim_; ++d) {
            currentVec_[d] = InputToFloat(tensor.GetValue(base + d));
        }
    }

    __aicore__ inline float CalcCurrentNormFromBuffer()
    {
        float sum = 0.0F;
        for (uint32_t d = 0; d < headDim_; ++d) {
            sum += currentVec_[d] * currentVec_[d];
        }
        if (sum < 1.0e-12F) {
            sum = 1.0e-12F;
        }
        return SqrtScalar(sum);
    }

    __aicore__ inline float CalcCurrentRotFromBuffer(
        GlobalTensor<float>& rotation,
        uint32_t outDim,
        float invNorm)
    {
        float sum = 0.0F;
        for (uint32_t inDim = 0; inDim < headDim_; ++inDim) {
            float x = currentVec_[inDim] * invNorm;
            float r = rotation.GetValue(
                static_cast<uint64_t>(inDim) * headDim_ + outDim);
            sum += x * r;
        }
        return sum;
    }

    __aicore__ inline bool UseHadamardTransform()
    {
        return transformMode_ == 1U && headDim_ > 0U && headDim_ <= 256U
            && ((headDim_ & (headDim_ - 1U)) == 0U);
    }

    __aicore__ inline float MatrixSign(
        GlobalTensor<float>& matrix,
        uint32_t row)
    {
        return matrix.GetValue(static_cast<uint64_t>(row) * headDim_) >= 0.0F
            ? 1.0F
            : -1.0F;
    }

    __aicore__ inline void FwhtCurrentRot()
    {
        for (uint32_t len = 1U; len < headDim_; len <<= 1U) {
            uint32_t step = len << 1U;
            for (uint32_t base = 0U; base < headDim_; base += step) {
                for (uint32_t off = 0U; off < len; ++off) {
                    uint32_t i = base + off;
                    uint32_t j = i + len;
                    float a = kResidual_[i];
                    float b = kResidual_[j];
                    kResidual_[i] = a + b;
                    kResidual_[j] = a - b;
                }
            }
        }
    }

    __aicore__ inline void FwhtCurrentVec()
    {
        for (uint32_t len = 1U; len < headDim_; len <<= 1U) {
            uint32_t step = len << 1U;
            for (uint32_t base = 0U; base < headDim_; base += step) {
                for (uint32_t off = 0U; off < len; ++off) {
                    uint32_t i = base + off;
                    uint32_t j = i + len;
                    float a = currentVec_[i];
                    float b = currentVec_[j];
                    currentVec_[i] = a + b;
                    currentVec_[j] = a - b;
                }
            }
        }
    }

    __aicore__ inline void FwhtQueryRot()
    {
        for (uint32_t len = 1U; len < headDim_; len <<= 1U) {
            uint32_t step = len << 1U;
            for (uint32_t base = 0U; base < headDim_; base += step) {
                for (uint32_t off = 0U; off < len; ++off) {
                    uint32_t i = base + off;
                    uint32_t j = i + len;
                    float a = qRot_[i];
                    float b = qRot_[j];
                    qRot_[i] = a + b;
                    qRot_[j] = a - b;
                }
            }
        }
    }

    __aicore__ inline void FwhtQueryRotGroup(uint32_t groupBase)
    {
        for (uint32_t len = 1U; len < headDim_; len <<= 1U) {
            uint32_t step = len << 1U;
            for (uint32_t base = 0U; base < headDim_; base += step) {
                for (uint32_t off = 0U; off < len; ++off) {
                    uint32_t i = groupBase + base + off;
                    uint32_t j = i + len;
                    float a = qRotGroup_[i];
                    float b = qRotGroup_[j];
                    qRotGroup_[i] = a + b;
                    qRotGroup_[j] = a - b;
                }
            }
        }
    }

    __aicore__ inline float HadamardScale()
    {
        if (headDim_ == 1U) {
            return 1.0F;
        }
        if (headDim_ == 2U) {
            return 0.7071067811865475F;
        }
        if (headDim_ == 4U) {
            return 0.5F;
        }
        if (headDim_ == 8U) {
            return 0.3535533905932738F;
        }
        if (headDim_ == 16U) {
            return 0.25F;
        }
        if (headDim_ == 32U) {
            return 0.1767766952966369F;
        }
        if (headDim_ == 64U) {
            return 0.125F;
        }
        if (headDim_ == 128U) {
            return 0.08838834764831845F;
        }
        return 0.0625F;
    }

    __aicore__ inline void PrepareCurrentRot(
        GlobalTensor<float>& rotation,
        float invNorm)
    {
        if (UseHadamardTransform()) {
            for (uint32_t d = 0; d < headDim_; ++d) {
                kResidual_[d] = currentVec_[d] * invNorm * MatrixSign(rotation, d);
            }
            FwhtCurrentRot();
            float scale = HadamardScale();
            for (uint32_t d = 0; d < headDim_; ++d) {
                kResidual_[d] *= scale;
            }
            return;
        }

        for (uint32_t outDim = 0; outDim < headDim_; ++outDim) {
            kResidual_[outDim] = CalcCurrentRotFromBuffer(rotation, outDim, invNorm);
        }
    }

    __aicore__ inline void PrepareCurrentRotRange(
        GlobalTensor<float>& rotation,
        float invNorm,
        uint32_t dimStart,
        uint32_t dimEnd)
    {
        if (UseHadamardTransform()) {
            PrepareCurrentRot(rotation, invNorm);
            return;
        }
        for (uint32_t outDim = dimStart; outDim < dimEnd; ++outDim) {
            kResidual_[outDim] = CalcCurrentRotFromBuffer(rotation, outDim, invNorm);
        }
    }

    __aicore__ inline void LoadSmallParams()
    {
        uint32_t kLevels = kStage1Bits_ < 4U ? (1U << kStage1Bits_) : 16U;
        uint32_t vLevels = vBits_ < 4U ? (1U << vBits_) : 16U;
        for (uint32_t i = 0; i < kLevels; ++i) {
            kCodebook_[i] = kCodebookGm_.GetValue(i);
        }
        for (uint32_t i = 0; i + 1U < kLevels; ++i) {
            kBoundary_[i] = kBoundaryGm_.GetValue(i);
        }
        for (uint32_t i = 0; i < vLevels; ++i) {
            vCodebook_[i] = vCodebookGm_.GetValue(i);
        }
        for (uint32_t i = 0; i + 1U < vLevels; ++i) {
            vBoundary_[i] = vBoundaryGm_.GetValue(i);
        }
    }

    __aicore__ inline uint32_t KBoundaryIndex(float x)
    {
        uint32_t levels = (1U << kStage1Bits_) - 1U;
        uint32_t idx = 0;
        for (uint32_t i = 0; i < levels; ++i) {
            idx += x > kBoundary_[i] ? 1U : 0U;
        }
        return idx;
    }

    __aicore__ inline uint32_t VBoundaryIndex(float x)
    {
        uint32_t levels = (1U << vBits_) - 1U;
        uint32_t idx = 0;
        for (uint32_t i = 0; i < levels; ++i) {
            idx += x > vBoundary_[i] ? 1U : 0U;
        }
        return idx;
    }

    __aicore__ inline void PackIndex(
        uint8_t* packed,
        uint32_t packedCols,
        uint32_t bits,
        uint32_t d,
        uint32_t idx)
    {
        uint32_t bitPos = d * bits;
        uint32_t byteId = bitPos >> 3;
        uint32_t bitOff = bitPos & 7U;
        uint32_t value = idx << bitOff;
        packed[byteId] = static_cast<uint8_t>(
            static_cast<uint32_t>(packed[byteId]) | (value & 0xFFU));
        if (bitOff + bits > 8U && byteId + 1U < packedCols) {
            packed[byteId + 1U] = static_cast<uint8_t>(
                static_cast<uint32_t>(packed[byteId + 1U]) | (value >> 8U));
        }
    }

    __aicore__ inline void EncodeCurrentK(
        uint32_t b,
        uint32_t kvHead,
        uint64_t slotHead)
    {
        uint8_t idxPacked[128];
        uint8_t qjlPacked[64];

        for (uint32_t col = 0; col < kPackedCols_; ++col) {
            idxPacked[col] = 0;
        }
        for (uint32_t col = 0; col < kQjlCols_; ++col) {
            qjlPacked[col] = 0;
        }

        LoadCurrentVector(keyGm_, b, kvHead);
        float norm = CalcCurrentNormFromBuffer();
        float invNorm = 1.0F / norm;
        float gammaSq = 0.0F;
        PrepareCurrentRot(kRotationGm_, invNorm);

        for (uint32_t d = 0; d < headDim_; ++d) {
            float xRot = kResidual_[d];
            uint32_t idx = KBoundaryIndex(xRot);
            PackIndex(idxPacked, kPackedCols_, kStage1Bits_, d, idx);
            float xHat = kCodebook_[idx];
            kResidual_[d] = xRot - xHat;
            gammaSq += kResidual_[d] * kResidual_[d];
        }

        float gamma = gammaSq > 0.0F ? SqrtScalar(gammaSq) : 0.0F;
        for (uint32_t j = 0; j < headDim_; ++j) {
            float sum = 0.0F;
            for (uint32_t d = 0; d < headDim_; ++d) {
                float p = kQjlProjTGm_.GetValue(
                    static_cast<uint64_t>(d) * headDim_ + j);
                sum += kResidual_[d] * p;
            }
            PackIndex(qjlPacked, kQjlCols_, 1U, j, sum >= 0.0F ? 1U : 0U);
        }

        uint64_t idxBase = slotHead * kPackedCols_;
        uint64_t qjlBase = slotHead * kQjlCols_;
        for (uint32_t col = 0; col < kPackedCols_; ++col) {
            kPackedIdxGm_.SetValue(idxBase + col, idxPacked[col]);
        }
        for (uint32_t col = 0; col < kQjlCols_; ++col) {
            kPackedQjlGm_.SetValue(qjlBase + col, qjlPacked[col]);
        }
        kGammaGm_.SetValue(slotHead, gamma);
        kNormGm_.SetValue(slotHead, norm);
    }

    __aicore__ inline bool CanPartitionCurrentKQjlEncode()
    {
        return qPerKv_ == 4U && kStage1Bits_ == 2U
            && (kQjlCols_ % qPerKv_) == 0U;
    }

    __aicore__ inline void PrepareCurrentKResidual(
        uint32_t b,
        uint32_t kvHead,
        uint64_t slotHead,
        bool writeStage1)
    {
        uint8_t idxPacked[128];
        if (writeStage1) {
            for (uint32_t col = 0; col < kPackedCols_; ++col) {
                idxPacked[col] = 0;
            }
        }

        LoadCurrentVector(keyGm_, b, kvHead);
        float norm = CalcCurrentNormFromBuffer();
        float invNorm = 1.0F / norm;
        float gammaSq = 0.0F;
        PrepareCurrentRot(kRotationGm_, invNorm);
        for (uint32_t d = 0; d < headDim_; ++d) {
            float xRot = kResidual_[d];
            uint32_t idx = KBoundaryIndex(xRot);
            if (writeStage1) {
                PackIndex(idxPacked, kPackedCols_, kStage1Bits_, d, idx);
            }
            float xHat = kCodebook_[idx];
            kResidual_[d] = xRot - xHat;
            if (writeStage1) {
                gammaSq += kResidual_[d] * kResidual_[d];
            }
        }

        if (writeStage1) {
            uint64_t idxBase = slotHead * kPackedCols_;
            for (uint32_t col = 0; col < kPackedCols_; ++col) {
                kPackedIdxGm_.SetValue(idxBase + col, idxPacked[col]);
            }
            kGammaGm_.SetValue(
                slotHead, gammaSq > 0.0F ? SqrtScalar(gammaSq) : 0.0F);
            kNormGm_.SetValue(slotHead, norm);
        }
    }

    __aicore__ inline void EncodeCurrentKQjlRange(
        uint64_t slotHead,
        uint32_t groupHead)
    {
        uint32_t colStart = (kQjlCols_ * groupHead) / qPerKv_;
        uint32_t colEnd = (kQjlCols_ * (groupHead + 1U)) / qPerKv_;
        if (colStart >= colEnd || colStart >= kQjlCols_) {
            return;
        }
        if (colEnd > kQjlCols_) {
            colEnd = kQjlCols_;
        }

        uint8_t qjlPacked[64];
        for (uint32_t col = colStart; col < colEnd; ++col) {
            qjlPacked[col] = 0;
        }
        uint32_t dimStart = colStart * 8U;
        uint32_t dimEnd = colEnd * 8U;
        if (dimEnd > headDim_) {
            dimEnd = headDim_;
        }
        for (uint32_t j = dimStart; j < dimEnd; ++j) {
            float sum = 0.0F;
            for (uint32_t d = 0; d < headDim_; ++d) {
                float p = kQjlProjTGm_.GetValue(
                    static_cast<uint64_t>(d) * headDim_ + j);
                sum += kResidual_[d] * p;
            }
            PackIndex(qjlPacked, kQjlCols_, 1U, j, sum >= 0.0F ? 1U : 0U);
        }

        uint64_t qjlBase = slotHead * kQjlCols_;
        for (uint32_t col = colStart; col < colEnd; ++col) {
            kPackedQjlGm_.SetValue(qjlBase + col, qjlPacked[col]);
        }
    }

    __aicore__ inline void EncodeCurrentV(
        uint32_t b,
        uint32_t kvHead,
        uint64_t slotHead)
    {
        uint8_t packed[128];
        for (uint32_t col = 0; col < vPackedCols_; ++col) {
            packed[col] = 0;
        }

        LoadCurrentVector(valueGm_, b, kvHead);
        float norm = CalcCurrentNormFromBuffer();
        float invNorm = 1.0F / norm;
        PrepareCurrentRot(vRotationGm_, invNorm);
        for (uint32_t d = 0; d < headDim_; ++d) {
            PackIndex(
                packed,
                vPackedCols_,
                vBits_,
                d,
                VBoundaryIndex(kResidual_[d]));
        }

        uint64_t packedBase = slotHead * vPackedCols_;
        for (uint32_t col = 0; col < vPackedCols_; ++col) {
            vPackedIdxGm_.SetValue(packedBase + col, packed[col]);
        }
        vNormGm_.SetValue(slotHead, norm);
    }

    __aicore__ inline bool CanPartitionCurrentVEncode()
    {
        return qPerKv_ == 4U && (vBits_ == 1U || vBits_ == 2U || vBits_ == 4U);
    }

    __aicore__ inline void EncodeCurrentVRange(
        uint32_t b,
        uint32_t kvHead,
        uint64_t slotHead,
        uint32_t groupHead)
    {
        uint32_t colStart = (vPackedCols_ * groupHead) / qPerKv_;
        uint32_t colEnd = (vPackedCols_ * (groupHead + 1U)) / qPerKv_;
        if (colStart >= colEnd || colStart >= vPackedCols_) {
            return;
        }
        if (colEnd > vPackedCols_) {
            colEnd = vPackedCols_;
        }

        uint8_t packed[128];
        for (uint32_t col = colStart; col < colEnd; ++col) {
            packed[col] = 0;
        }

        LoadCurrentVector(valueGm_, b, kvHead);
        float norm = CalcCurrentNormFromBuffer();
        float invNorm = 1.0F / norm;
        uint32_t dimsPerByte = 8U / vBits_;
        uint32_t dimStart = colStart * dimsPerByte;
        uint32_t dimEnd = colEnd * dimsPerByte;
        if (dimEnd > headDim_) {
            dimEnd = headDim_;
        }
        PrepareCurrentRotRange(vRotationGm_, invNorm, dimStart, dimEnd);
        for (uint32_t d = dimStart; d < dimEnd; ++d) {
            PackIndex(
                packed,
                vPackedCols_,
                vBits_,
                d,
                VBoundaryIndex(kResidual_[d]));
        }

        uint64_t packedBase = slotHead * vPackedCols_;
        for (uint32_t col = colStart; col < colEnd; ++col) {
            vPackedIdxGm_.SetValue(packedBase + col, packed[col]);
        }
        if (groupHead == 0U) {
            vNormGm_.SetValue(slotHead, norm);
        }
    }

    __aicore__ inline void BuildQueryTransforms(uint32_t b, uint32_t head)
    {
        uint64_t qBase =
            (static_cast<uint64_t>(b) * numHeads_ + head) * headDim_;
        if (pretransformedQuery_ != 0U) {
            for (uint32_t d = 0; d < headDim_; ++d) {
                qRot_[d] = kRotationGm_.GetValue(qBase + d);
                qQjl_[d] = kQjlQueryMatrixGm_.GetValue(qBase + d);
            }
            return;
        }
        if (UseHadamardTransform()) {
            for (uint32_t d = 0; d < headDim_; ++d) {
                float q = InputToFloat(queryGm_.GetValue(qBase + d));
                qRot_[d] = q * MatrixSign(kRotationGm_, d);
                qQjl_[d] = 0.0F;
            }
            for (uint32_t inDim = 0; inDim < headDim_; ++inDim) {
                float q = InputToFloat(queryGm_.GetValue(qBase + inDim));
                uint64_t matrixBase = static_cast<uint64_t>(inDim) * headDim_;
                for (uint32_t outDim = 0; outDim < headDim_; ++outDim) {
                    qQjl_[outDim] +=
                        q * kQjlQueryMatrixGm_.GetValue(matrixBase + outDim);
                }
            }
            FwhtQueryRot();
            float rotScale = HadamardScale();
            for (uint32_t d = 0; d < headDim_; ++d) {
                qRot_[d] *= rotScale;
            }
            return;
        }
        for (uint32_t d = 0; d < headDim_; ++d) {
            qRot_[d] = 0.0F;
            qQjl_[d] = 0.0F;
        }
        for (uint32_t inDim = 0; inDim < headDim_; ++inDim) {
            float q = InputToFloat(queryGm_.GetValue(qBase + inDim));
            uint64_t matrixBase = static_cast<uint64_t>(inDim) * headDim_;
            for (uint32_t outDim = 0; outDim < headDim_; ++outDim) {
                qRot_[outDim] += q * kRotationGm_.GetValue(matrixBase + outDim);
                qQjl_[outDim] +=
                    q * kQjlQueryMatrixGm_.GetValue(matrixBase + outDim);
            }
        }
    }

    __aicore__ inline void BuildGroupedQueryTransforms(
        uint32_t b,
        uint32_t kvHead)
    {
        uint32_t qHeadBase = kvHead * qPerKv_;
        uint64_t qBase =
            (static_cast<uint64_t>(b) * numHeads_ + qHeadBase) * headDim_;
        uint32_t q1Base = headDim_;
        uint32_t q2Base = headDim_ << 1;
        uint32_t q3Base = 3U * headDim_;
        if (pretransformedQuery_ != 0U) {
            for (uint32_t d = 0; d < 4U * headDim_; ++d) {
                qRotGroup_[d] = kRotationGm_.GetValue(qBase + d);
                qQjlGroup_[d] = kQjlQueryMatrixGm_.GetValue(qBase + d);
            }
            return;
        }
        if (UseHadamardTransform()) {
            for (uint32_t d = 0; d < headDim_; ++d) {
                float sign = MatrixSign(kRotationGm_, d);
                float query0 = InputToFloat(queryGm_.GetValue(qBase + d));
                float query1 = InputToFloat(
                    queryGm_.GetValue(qBase + q1Base + d));
                float query2 = InputToFloat(
                    queryGm_.GetValue(qBase + q2Base + d));
                float query3 = InputToFloat(
                    queryGm_.GetValue(qBase + q3Base + d));
                qRotGroup_[d] = query0 * sign;
                qRotGroup_[q1Base + d] = query1 * sign;
                qRotGroup_[q2Base + d] = query2 * sign;
                qRotGroup_[q3Base + d] = query3 * sign;
                qQjlGroup_[d] = 0.0F;
                qQjlGroup_[q1Base + d] = 0.0F;
                qQjlGroup_[q2Base + d] = 0.0F;
                qQjlGroup_[q3Base + d] = 0.0F;
            }
            for (uint32_t inDim = 0; inDim < headDim_; ++inDim) {
                float query0 = InputToFloat(queryGm_.GetValue(qBase + inDim));
                float query1 = InputToFloat(
                    queryGm_.GetValue(qBase + q1Base + inDim));
                float query2 = InputToFloat(
                    queryGm_.GetValue(qBase + q2Base + inDim));
                float query3 = InputToFloat(
                    queryGm_.GetValue(qBase + q3Base + inDim));
                uint64_t matrixBase = static_cast<uint64_t>(inDim) * headDim_;
                for (uint32_t outDim = 0; outDim < headDim_; ++outDim) {
                    float qjl = kQjlQueryMatrixGm_.GetValue(matrixBase + outDim);
                    qQjlGroup_[outDim] += query0 * qjl;
                    qQjlGroup_[q1Base + outDim] += query1 * qjl;
                    qQjlGroup_[q2Base + outDim] += query2 * qjl;
                    qQjlGroup_[q3Base + outDim] += query3 * qjl;
                }
            }
            FwhtQueryRotGroup(0U);
            FwhtQueryRotGroup(q1Base);
            FwhtQueryRotGroup(q2Base);
            FwhtQueryRotGroup(q3Base);
            float rotScale = HadamardScale();
            for (uint32_t d = 0; d < 4U * headDim_; ++d) {
                qRotGroup_[d] *= rotScale;
            }
            return;
        }
        for (uint32_t d = 0; d < 4U * headDim_; ++d) {
            qRotGroup_[d] = 0.0F;
            qQjlGroup_[d] = 0.0F;
        }
        for (uint32_t inDim = 0; inDim < headDim_; ++inDim) {
            float query0 = InputToFloat(queryGm_.GetValue(qBase + inDim));
            float query1 = InputToFloat(
                queryGm_.GetValue(qBase + q1Base + inDim));
            float query2 = InputToFloat(
                queryGm_.GetValue(qBase + q2Base + inDim));
            float query3 = InputToFloat(
                queryGm_.GetValue(qBase + q3Base + inDim));
            uint64_t matrixBase = static_cast<uint64_t>(inDim) * headDim_;
            for (uint32_t outDim = 0; outDim < headDim_; ++outDim) {
                float rot = kRotationGm_.GetValue(matrixBase + outDim);
                float qjl = kQjlQueryMatrixGm_.GetValue(matrixBase + outDim);
                qRotGroup_[outDim] += query0 * rot;
                qRotGroup_[q1Base + outDim] += query1 * rot;
                qRotGroup_[q2Base + outDim] += query2 * rot;
                qRotGroup_[q3Base + outDim] += query3 * rot;
                qQjlGroup_[outDim] += query0 * qjl;
                qQjlGroup_[q1Base + outDim] += query1 * qjl;
                qQjlGroup_[q2Base + outDim] += query2 * qjl;
                qQjlGroup_[q3Base + outDim] += query3 * qjl;
            }
        }
    }

    __aicore__ inline void ZeroGroupedScratch()
    {
        for (uint32_t d = 0; d < 4U * headDim_; ++d) {
            qRotGroup_[d] = 0.0F;
        }
    }

    template <uint32_t KBits>
    __aicore__ inline float HistoryScoreByCacheIndexConst(uint64_t cacheIndex)
    {
        uint64_t idxBase = cacheIndex * kPackedCols_;
        uint64_t qjlBase = cacheIndex * kQjlCols_;
        float gamma = kGammaGm_.GetValue(cacheIndex);
        float norm = kNormGm_.GetValue(cacheIndex);
        float mseAcc = 0.0F;
        float qjlAcc = 0.0F;
        for (uint32_t d = 0; d < headDim_; ++d) {
            uint32_t idx = ExtractBitsConst<KBits>(kPackedIdxGm_, idxBase, d);
            uint32_t qjl = ExtractBitsConst<1U>(kPackedQjlGm_, qjlBase, d);
            float sign = qjl != 0U ? 1.0F : -1.0F;
            mseAcc += qRot_[d] * kCodebook_[idx];
            qjlAcc += qQjl_[d] * sign;
        }
        return (mseAcc + correction_ * gamma * qjlAcc) * norm * scale_;
    }

    __aicore__ inline float HistoryScoreByCacheIndexGeneric(uint64_t cacheIndex)
    {
        uint64_t idxBase = cacheIndex * kPackedCols_;
        uint64_t qjlBase = cacheIndex * kQjlCols_;
        float gamma = kGammaGm_.GetValue(cacheIndex);
        float norm = kNormGm_.GetValue(cacheIndex);
        float mseAcc = 0.0F;
        float qjlAcc = 0.0F;
        for (uint32_t d = 0; d < headDim_; ++d) {
            uint32_t idx = ExtractBits(kPackedIdxGm_, idxBase, d, kStage1Bits_);
            uint32_t qjl = ExtractBitsConst<1U>(kPackedQjlGm_, qjlBase, d);
            float sign = qjl != 0U ? 1.0F : -1.0F;
            mseAcc += qRot_[d] * kCodebook_[idx];
            qjlAcc += qQjl_[d] * sign;
        }
        return (mseAcc + correction_ * gamma * qjlAcc) * norm * scale_;
    }

    __aicore__ inline float HistoryScoreByCacheIndex(uint64_t cacheIndex)
    {
        if (kStage1Bits_ == 3U) {
            return HistoryScoreByCacheIndexConst<3U>(cacheIndex);
        }
        if (kStage1Bits_ == 4U) {
            return HistoryScoreByCacheIndexConst<4U>(cacheIndex);
        }
        if (kStage1Bits_ == 2U) {
            return HistoryScoreByCacheIndexConst<2U>(cacheIndex);
        }
        if (kStage1Bits_ == 1U) {
            return HistoryScoreByCacheIndexConst<1U>(cacheIndex);
        }
        return HistoryScoreByCacheIndexGeneric(cacheIndex);
    }

    template <uint32_t KBits>
    __aicore__ inline void HistoryScoresGroupByCacheIndexConst(uint64_t cacheIndex)
    {
        uint64_t idxBase = cacheIndex * kPackedCols_;
        uint64_t qjlBase = cacheIndex * kQjlCols_;
        float gamma = kGammaGm_.GetValue(cacheIndex);
        float norm = kNormGm_.GetValue(cacheIndex);
        float mseAcc0 = 0.0F;
        float mseAcc1 = 0.0F;
        float mseAcc2 = 0.0F;
        float mseAcc3 = 0.0F;
        float qjlAcc0 = 0.0F;
        float qjlAcc1 = 0.0F;
        float qjlAcc2 = 0.0F;
        float qjlAcc3 = 0.0F;
        uint32_t q1Base = headDim_;
        uint32_t q2Base = headDim_ << 1;
        uint32_t q3Base = 3U * headDim_;
        for (uint32_t d = 0; d < headDim_; ++d) {
            uint32_t idx = ExtractBitsConst<KBits>(kPackedIdxGm_, idxBase, d);
            uint32_t qjl = ExtractBitsConst<1U>(kPackedQjlGm_, qjlBase, d);
            float code = kCodebook_[idx];
            float sign = qjl != 0U ? 1.0F : -1.0F;
            mseAcc0 += qRotGroup_[d] * code;
            mseAcc1 += qRotGroup_[q1Base + d] * code;
            mseAcc2 += qRotGroup_[q2Base + d] * code;
            mseAcc3 += qRotGroup_[q3Base + d] * code;
            qjlAcc0 += qQjlGroup_[d] * sign;
            qjlAcc1 += qQjlGroup_[q1Base + d] * sign;
            qjlAcc2 += qQjlGroup_[q2Base + d] * sign;
            qjlAcc3 += qQjlGroup_[q3Base + d] * sign;
        }
        float qjlScale = correction_ * gamma;
        float outScale = norm * scale_;
        scoreGroup_[0] = (mseAcc0 + qjlScale * qjlAcc0) * outScale;
        scoreGroup_[1] = (mseAcc1 + qjlScale * qjlAcc1) * outScale;
        scoreGroup_[2] = (mseAcc2 + qjlScale * qjlAcc2) * outScale;
        scoreGroup_[3] = (mseAcc3 + qjlScale * qjlAcc3) * outScale;
    }

    __aicore__ inline void HistoryScoresGroupByCacheIndexGeneric(uint64_t cacheIndex)
    {
        uint64_t idxBase = cacheIndex * kPackedCols_;
        uint64_t qjlBase = cacheIndex * kQjlCols_;
        float gamma = kGammaGm_.GetValue(cacheIndex);
        float norm = kNormGm_.GetValue(cacheIndex);
        float mseAcc0 = 0.0F;
        float mseAcc1 = 0.0F;
        float mseAcc2 = 0.0F;
        float mseAcc3 = 0.0F;
        float qjlAcc0 = 0.0F;
        float qjlAcc1 = 0.0F;
        float qjlAcc2 = 0.0F;
        float qjlAcc3 = 0.0F;
        uint32_t q1Base = headDim_;
        uint32_t q2Base = headDim_ << 1;
        uint32_t q3Base = 3U * headDim_;
        for (uint32_t d = 0; d < headDim_; ++d) {
            uint32_t idx = ExtractBits(kPackedIdxGm_, idxBase, d, kStage1Bits_);
            uint32_t qjl = ExtractBitsConst<1U>(kPackedQjlGm_, qjlBase, d);
            float code = kCodebook_[idx];
            float sign = qjl != 0U ? 1.0F : -1.0F;
            mseAcc0 += qRotGroup_[d] * code;
            mseAcc1 += qRotGroup_[q1Base + d] * code;
            mseAcc2 += qRotGroup_[q2Base + d] * code;
            mseAcc3 += qRotGroup_[q3Base + d] * code;
            qjlAcc0 += qQjlGroup_[d] * sign;
            qjlAcc1 += qQjlGroup_[q1Base + d] * sign;
            qjlAcc2 += qQjlGroup_[q2Base + d] * sign;
            qjlAcc3 += qQjlGroup_[q3Base + d] * sign;
        }
        float qjlScale = correction_ * gamma;
        float outScale = norm * scale_;
        scoreGroup_[0] = (mseAcc0 + qjlScale * qjlAcc0) * outScale;
        scoreGroup_[1] = (mseAcc1 + qjlScale * qjlAcc1) * outScale;
        scoreGroup_[2] = (mseAcc2 + qjlScale * qjlAcc2) * outScale;
        scoreGroup_[3] = (mseAcc3 + qjlScale * qjlAcc3) * outScale;
    }

    __aicore__ inline void HistoryScoresGroupByCacheIndex(uint64_t cacheIndex)
    {
        if (kStage1Bits_ == 3U) {
            HistoryScoresGroupByCacheIndexConst<3U>(cacheIndex);
            return;
        }
        if (kStage1Bits_ == 4U) {
            HistoryScoresGroupByCacheIndexConst<4U>(cacheIndex);
            return;
        }
        if (kStage1Bits_ == 2U) {
            HistoryScoresGroupByCacheIndexConst<2U>(cacheIndex);
            return;
        }
        if (kStage1Bits_ == 1U) {
            HistoryScoresGroupByCacheIndexConst<1U>(cacheIndex);
            return;
        }
        HistoryScoresGroupByCacheIndexGeneric(cacheIndex);
    }

    template <uint32_t VBits>
    __aicore__ inline void AccumulateHistoryVByCacheIndexConst(
        uint64_t cacheIndex,
        float weight)
    {
        uint64_t vBase = cacheIndex * vPackedCols_;
        float scaledNorm = weight * vNormGm_.GetValue(cacheIndex);
        for (uint32_t d = 0; d < headDim_; ++d) {
            uint32_t idx = ExtractBitsConst<VBits>(vPackedIdxGm_, vBase, d);
            accRot_[d] += scaledNorm * vCodebook_[idx];
        }
    }

    __aicore__ inline void AccumulateHistoryVByCacheIndexGeneric(
        uint64_t cacheIndex,
        float weight)
    {
        uint64_t vBase = cacheIndex * vPackedCols_;
        float scaledNorm = weight * vNormGm_.GetValue(cacheIndex);
        for (uint32_t d = 0; d < headDim_; ++d) {
            uint32_t idx = ExtractBits(vPackedIdxGm_, vBase, d, vBits_);
            accRot_[d] += scaledNorm * vCodebook_[idx];
        }
    }

    __aicore__ inline void AccumulateHistoryVByCacheIndex(
        uint64_t cacheIndex,
        float weight)
    {
        if (vBits_ == 4U) {
            AccumulateHistoryVByCacheIndexConst<4U>(cacheIndex, weight);
            return;
        }
        if (vBits_ == 3U) {
            AccumulateHistoryVByCacheIndexConst<3U>(cacheIndex, weight);
            return;
        }
        if (vBits_ == 2U) {
            AccumulateHistoryVByCacheIndexConst<2U>(cacheIndex, weight);
            return;
        }
        if (vBits_ == 1U) {
            AccumulateHistoryVByCacheIndexConst<1U>(cacheIndex, weight);
            return;
        }
        AccumulateHistoryVByCacheIndexGeneric(cacheIndex, weight);
    }

    template <uint32_t VBits>
    __aicore__ inline void AccumulateHistoryVGroupByCacheIndexConst(
        uint64_t cacheIndex)
    {
        uint64_t vBase = cacheIndex * vPackedCols_;
        float norm = vNormGm_.GetValue(cacheIndex);
        uint32_t q1Base = headDim_;
        uint32_t q2Base = headDim_ << 1;
        uint32_t q3Base = 3U * headDim_;
        for (uint32_t d = 0; d < headDim_; ++d) {
            uint32_t idx = ExtractBitsConst<VBits>(vPackedIdxGm_, vBase, d);
            float v = vCodebook_[idx] * norm;
            accRotGroup_[d] += weightGroup_[0] * v;
            accRotGroup_[q1Base + d] += weightGroup_[1] * v;
            accRotGroup_[q2Base + d] += weightGroup_[2] * v;
            accRotGroup_[q3Base + d] += weightGroup_[3] * v;
        }
    }

    __aicore__ inline void AccumulateHistoryVGroupByCacheIndexGeneric(
        uint64_t cacheIndex)
    {
        uint64_t vBase = cacheIndex * vPackedCols_;
        float norm = vNormGm_.GetValue(cacheIndex);
        uint32_t q1Base = headDim_;
        uint32_t q2Base = headDim_ << 1;
        uint32_t q3Base = 3U * headDim_;
        for (uint32_t d = 0; d < headDim_; ++d) {
            uint32_t idx = ExtractBits(vPackedIdxGm_, vBase, d, vBits_);
            float v = vCodebook_[idx] * norm;
            accRotGroup_[d] += weightGroup_[0] * v;
            accRotGroup_[q1Base + d] += weightGroup_[1] * v;
            accRotGroup_[q2Base + d] += weightGroup_[2] * v;
            accRotGroup_[q3Base + d] += weightGroup_[3] * v;
        }
    }

    __aicore__ inline void AccumulateHistoryVGroupByCacheIndex(
        uint64_t cacheIndex)
    {
        if (vBits_ == 4U) {
            AccumulateHistoryVGroupByCacheIndexConst<4U>(cacheIndex);
            return;
        }
        if (vBits_ == 3U) {
            AccumulateHistoryVGroupByCacheIndexConst<3U>(cacheIndex);
            return;
        }
        if (vBits_ == 2U) {
            AccumulateHistoryVGroupByCacheIndexConst<2U>(cacheIndex);
            return;
        }
        if (vBits_ == 1U) {
            AccumulateHistoryVGroupByCacheIndexConst<1U>(cacheIndex);
            return;
        }
        AccumulateHistoryVGroupByCacheIndexGeneric(cacheIndex);
    }

    __aicore__ inline float CurrentScore(uint32_t b, uint32_t head, uint32_t kvHead)
    {
        uint64_t qBase =
            (static_cast<uint64_t>(b) * numHeads_ + head) * headDim_;
        uint64_t kBase =
            (static_cast<uint64_t>(b) * numKvHeads_ + kvHead) * headDim_;
        float acc = 0.0F;
        for (uint32_t d = 0; d < headDim_; ++d) {
            acc += InputToFloat(queryGm_.GetValue(qBase + d))
                * InputToFloat(keyGm_.GetValue(kBase + d));
        }
        return acc * scale_;
    }

    __aicore__ inline float OnlineStep(float score)
    {
        float newMax = initialized_ && maxScore_ > score ? maxScore_ : score;
        float oldScale = initialized_ ? ExpScalar(maxScore_ - newMax) : 0.0F;
        float weight = ExpScalar(score - newMax);
        for (uint32_t d = 0; d < headDim_; ++d) {
            accRot_[d] *= oldScale;
        }
        currentWeight_ *= oldScale;
        sum_ = sum_ * oldScale + weight;
        maxScore_ = newMax;
        initialized_ = true;
        return weight;
    }

    __aicore__ inline void OnlineAccumulateHistoryByCacheIndex(
        float score,
        uint64_t cacheIndex)
    {
        float weight = OnlineStep(score);
        AccumulateHistoryVByCacheIndex(cacheIndex, weight);
    }

    __aicore__ inline void OnlineAccumulateCurrent(float score)
    {
        currentWeight_ += OnlineStep(score);
    }

    __aicore__ inline float OnlineStepNoAccumulator(float score)
    {
        float newMax = initialized_ && maxScore_ > score ? maxScore_ : score;
        float oldScale = initialized_ ? ExpScalar(maxScore_ - newMax) : 0.0F;
        float weight = ExpScalar(score - newMax);
        currentWeight_ *= oldScale;
        sum_ = sum_ * oldScale + weight;
        maxScore_ = newMax;
        initialized_ = true;
        return weight;
    }

    __aicore__ inline void ScaleGlobalAccumulator(float scale)
    {
        for (uint32_t d = 0; d < headDim_; ++d) {
            accRot_[d] *= scale;
        }
        currentWeight_ *= scale;
    }

    __aicore__ inline void ScaleGroupedAccumulatorRow(
        uint32_t q,
        float scale)
    {
        uint32_t base = q * headDim_;
        for (uint32_t d = 0; d < headDim_; ++d) {
            accRotGroup_[base + d] *= scale;
        }
        currentWeightGroup_[q] *= scale;
    }

    __aicore__ inline void OnlineWeightsGroup()
    {
        for (uint32_t q = 0; q < 4U; ++q) {
            float score = scoreGroup_[q];
            if (!initializedGroup_[q]) {
                maxScoreGroup_[q] = score;
                sumGroup_[q] = 1.0F;
                weightGroup_[q] = 1.0F;
                initializedGroup_[q] = true;
                continue;
            }

            if (score > maxScoreGroup_[q]) {
                float oldScale = ExpScalar(maxScoreGroup_[q] - score);
                ScaleGroupedAccumulatorRow(q, oldScale);
                sumGroup_[q] = sumGroup_[q] * oldScale + 1.0F;
                maxScoreGroup_[q] = score;
                weightGroup_[q] = 1.0F;
            } else {
                float weight = ExpScalar(score - maxScoreGroup_[q]);
                sumGroup_[q] += weight;
                weightGroup_[q] = weight;
            }
        }
    }

    __aicore__ inline void OnlineAccumulateHistoryGroup(uint64_t cacheIndex)
    {
        OnlineWeightsGroup();
        AccumulateHistoryVGroupByCacheIndex(cacheIndex);
    }

    __aicore__ inline void OnlineAccumulateCurrentGroup()
    {
        OnlineWeightsGroup();
        for (uint32_t q = 0; q < 4U; ++q) {
            currentWeightGroup_[q] += weightGroup_[q];
        }
    }

    __aicore__ inline void AccumulateHistoryTile(
        uint64_t cacheBase,
        uint32_t tokenOffset,
        uint32_t tileLen)
    {
        float tileMax = -3.4028234663852886e38F;
        for (uint32_t i = 0; i < tileLen; ++i) {
            uint64_t cacheIndex =
                cacheBase + static_cast<uint64_t>(tokenOffset + i) * numKvHeads_;
            float score = HistoryScoreByCacheIndex(cacheIndex);
            scoreTile_[i] = score;
            if (score > tileMax) {
                tileMax = score;
            }
        }

        bool updateMax = initialized_ && tileMax > maxScore_;
        float newMax = updateMax || !initialized_ ? tileMax : maxScore_;
        float oldScale = initialized_ ? 1.0F : 0.0F;
        if (updateMax) {
            oldScale = ExpScalar(maxScore_ - newMax);
            ScaleGlobalAccumulator(oldScale);
        }

        float tileSum = 0.0F;
        LocalTensor<float> weightLocal = expBuf_.Get<float>();
        for (uint32_t i = 0; i < tileLen; ++i) {
            weightLocal.SetValue(i, scoreTile_[i] - newMax);
        }
        SToVSync();
        Exp(weightLocal, weightLocal, tileLen);
        VToSSync();
        for (uint32_t i = 0; i < tileLen; ++i) {
            float weight = weightLocal.GetValue(i);
            tileSum += weight;
            uint64_t cacheIndex =
                cacheBase + static_cast<uint64_t>(tokenOffset + i) * numKvHeads_;
            AccumulateHistoryVByCacheIndex(cacheIndex, weight);
        }

        sum_ = sum_ * oldScale + tileSum;
        maxScore_ = newMax;
        initialized_ = true;
    }

    __aicore__ inline void ProcessHistoryScalar(
        uint32_t b,
        uint32_t kvHead,
        uint32_t oldLen)
    {
        uint32_t pos = 0;
        uint32_t blockOffset = 0;
        while (pos < oldLen) {
            int32_t blockId = blockTableGm_.GetValue(
                static_cast<uint64_t>(b) * maxBlocksPerSeq_ + blockOffset);
            uint32_t tokensInBlock = blockSize_;
            uint32_t remaining = oldLen - pos;
            if (tokensInBlock > remaining) {
                tokensInBlock = remaining;
            }
            uint64_t cacheBase =
                static_cast<uint64_t>(blockId) * blockSize_ * numKvHeads_
                + kvHead;
            for (uint32_t tokenOffset = 0; tokenOffset < tokensInBlock;
                 ++tokenOffset) {
                uint64_t cacheIndex =
                    cacheBase + static_cast<uint64_t>(tokenOffset) * numKvHeads_;
                float score = HistoryScoreByCacheIndex(cacheIndex);
                OnlineAccumulateHistoryByCacheIndex(score, cacheIndex);
            }
            pos += tokensInBlock;
            ++blockOffset;
        }
    }

    __aicore__ inline float ProcessHistoryScoreOnly(
        uint32_t b,
        uint32_t kvHead,
        uint32_t oldLen)
    {
        float sink = 0.0F;
        uint32_t pos = 0;
        uint32_t blockOffset = 0;
        while (pos < oldLen) {
            int32_t blockId = blockTableGm_.GetValue(
                static_cast<uint64_t>(b) * maxBlocksPerSeq_ + blockOffset);
            uint32_t tokensInBlock = blockSize_;
            uint32_t remaining = oldLen - pos;
            if (tokensInBlock > remaining) {
                tokensInBlock = remaining;
            }
            uint64_t cacheBase =
                static_cast<uint64_t>(blockId) * blockSize_ * numKvHeads_
                + kvHead;
            for (uint32_t tokenOffset = 0; tokenOffset < tokensInBlock;
                 ++tokenOffset) {
                uint64_t cacheIndex =
                    cacheBase + static_cast<uint64_t>(tokenOffset) * numKvHeads_;
                sink += HistoryScoreByCacheIndex(cacheIndex) * 1.0e-6F;
            }
            pos += tokensInBlock;
            ++blockOffset;
        }
        return sink;
    }

    __aicore__ inline float ProcessHistoryScoreOnlineOnly(
        uint32_t b,
        uint32_t kvHead,
        uint32_t oldLen)
    {
        float sink = 0.0F;
        uint32_t pos = 0;
        uint32_t blockOffset = 0;
        while (pos < oldLen) {
            int32_t blockId = blockTableGm_.GetValue(
                static_cast<uint64_t>(b) * maxBlocksPerSeq_ + blockOffset);
            uint32_t tokensInBlock = blockSize_;
            uint32_t remaining = oldLen - pos;
            if (tokensInBlock > remaining) {
                tokensInBlock = remaining;
            }
            uint64_t cacheBase =
                static_cast<uint64_t>(blockId) * blockSize_ * numKvHeads_
                + kvHead;
            for (uint32_t tokenOffset = 0; tokenOffset < tokensInBlock;
                 ++tokenOffset) {
                uint64_t cacheIndex =
                    cacheBase + static_cast<uint64_t>(tokenOffset) * numKvHeads_;
                float score = HistoryScoreByCacheIndex(cacheIndex);
                sink += OnlineStepNoAccumulator(score) * 1.0e-6F;
            }
            pos += tokensInBlock;
            ++blockOffset;
        }
        return sink + sum_ * 1.0e-6F;
    }

    __aicore__ inline void ProcessHistoryTiled(
        uint32_t b,
        uint32_t kvHead,
        uint32_t oldLen)
    {
        uint32_t tileLenLimit = scoreTileLen_;
        if (tileLenLimit > 64U) {
            tileLenLimit = 64U;
        }
        if (tileLenLimit <= 1U) {
            ProcessHistoryScalar(b, kvHead, oldLen);
            return;
        }

        uint32_t pos = 0;
        uint32_t blockOffset = 0;
        while (pos < oldLen) {
            int32_t blockId = blockTableGm_.GetValue(
                static_cast<uint64_t>(b) * maxBlocksPerSeq_ + blockOffset);
            uint32_t tokensInBlock = blockSize_;
            uint32_t remaining = oldLen - pos;
            if (tokensInBlock > remaining) {
                tokensInBlock = remaining;
            }
            uint64_t cacheBase =
                static_cast<uint64_t>(blockId) * blockSize_ * numKvHeads_
                + kvHead;
            uint32_t tokenOffset = 0;
            while (tokenOffset < tokensInBlock) {
                uint32_t tileLen = tokensInBlock - tokenOffset;
                if (tileLen > tileLenLimit) {
                    tileLen = tileLenLimit;
                }
                AccumulateHistoryTile(cacheBase, tokenOffset, tileLen);
                tokenOffset += tileLen;
            }
            pos += tokensInBlock;
            ++blockOffset;
        }
    }

    __aicore__ inline void ProcessHistoryTiledRange(
        uint32_t b,
        uint32_t kvHead,
        uint32_t startPos,
        uint32_t endPos)
    {
        if (startPos >= endPos) {
            return;
        }
        uint32_t tileLenLimit = scoreTileLen_;
        if (tileLenLimit > 64U) {
            tileLenLimit = 64U;
        }
        if (tileLenLimit == 0U) {
            tileLenLimit = 1U;
        }

        uint32_t pos = startPos;
        while (pos < endPos) {
            uint32_t blockOffset = pos / blockSize_;
            uint32_t tokenOffset = pos - blockOffset * blockSize_;
            int32_t blockId = blockTableGm_.GetValue(
                static_cast<uint64_t>(b) * maxBlocksPerSeq_ + blockOffset);
            uint32_t tokensInBlock = blockSize_ - tokenOffset;
            uint32_t remaining = endPos - pos;
            if (tokensInBlock > remaining) {
                tokensInBlock = remaining;
            }
            uint64_t cacheBase =
                static_cast<uint64_t>(blockId) * blockSize_ * numKvHeads_
                + kvHead;
            uint32_t consumed = 0;
            while (consumed < tokensInBlock) {
                uint32_t tileLen = tokensInBlock - consumed;
                if (tileLen > tileLenLimit) {
                    tileLen = tileLenLimit;
                }
                AccumulateHistoryTile(cacheBase, tokenOffset + consumed, tileLen);
                consumed += tileLen;
            }
            pos += tokensInBlock;
        }
    }

    __aicore__ inline uint64_t HistoryPartialStride()
    {
        uint32_t accStride = (headDim_ + 7U) & ~7U;
        return static_cast<uint64_t>(accStride + 8U);
    }

    __aicore__ inline uint64_t HistoryPartialBase(
        uint32_t b,
        uint32_t head,
        uint32_t partition)
    {
        return ((static_cast<uint64_t>(b) * numHeads_ + head)
                * historyPartitions_
                + partition)
            * HistoryPartialStride();
    }

    __aicore__ inline void StoreHistoryPartial(
        uint32_t b,
        uint32_t head,
        uint32_t partition)
    {
        uint64_t base = HistoryPartialBase(b, head, partition);
        LocalTensor<float> scratchLocal = expBuf_.Get<float>();
        scratchLocal.SetValue(
            0,
            initialized_ ? maxScore_ : -3.4028234663852886e38F);
        scratchLocal.SetValue(1, initialized_ ? sum_ : 0.0F);
        for (uint32_t i = 2U; i < 8U; ++i) {
            scratchLocal.SetValue(i, 0.0F);
        }
        pipe_barrier(PIPE_ALL);
        DataCopy(scratchGm_[base], scratchLocal, 8U);
        pipe_barrier(PIPE_ALL);

        uint32_t accStride = (headDim_ + 7U) & ~7U;
        for (uint32_t d = 0; d < accStride; d += 8U) {
            for (uint32_t i = 0; i < 8U; ++i) {
                uint32_t dim = d + i;
                scratchLocal.SetValue(
                    i,
                    initialized_ && dim < headDim_ ? accRot_[dim] : 0.0F);
            }
            pipe_barrier(PIPE_ALL);
            DataCopy(scratchGm_[base + 8U + d], scratchLocal, 8U);
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void ProcessHistoryPartialTask(uint64_t task)
    {
        uint32_t partition = static_cast<uint32_t>(task % historyPartitions_);
        uint64_t headTask = task / historyPartitions_;
        uint32_t b = static_cast<uint32_t>(headTask / numHeads_);
        uint32_t head = static_cast<uint32_t>(headTask % numHeads_);
        uint32_t kvHead = head / qPerKv_;
        if (b >= batch_ || head >= numHeads_ || kvHead >= numKvHeads_) {
            return;
        }

        if (partition == 0U && skipCacheUpdate_ == 0U) {
            uint32_t groupHead = head - kvHead * qPerKv_;
            int64_t slot = slotMappingGm_.GetValue(b);
            if (slot >= 0) {
                uint64_t slotHead =
                    static_cast<uint64_t>(slot) * numKvHeads_ + kvHead;
                if (CanPartitionCurrentKQjlEncode()) {
                    PrepareCurrentKResidual(
                        b, kvHead, slotHead, groupHead == 0U);
                    EncodeCurrentKQjlRange(slotHead, groupHead);
                } else if (groupHead == 0U) {
                    EncodeCurrentK(b, kvHead, slotHead);
                }
                if (CanPartitionCurrentVEncode()) {
                    EncodeCurrentVRange(b, kvHead, slotHead, groupHead);
                } else if (groupHead == 0U) {
                    EncodeCurrentV(b, kvHead, slotHead);
                }
            }
        }

        int32_t oldLenRaw = oldSeqLensGm_.GetValue(b);
        uint32_t oldLen = oldLenRaw > 0 ? static_cast<uint32_t>(oldLenRaw) : 0U;
        if (oldLen > maxSeqLen_) {
            oldLen = maxSeqLen_;
        }
        uint32_t chunk =
            (oldLen + historyPartitions_ - 1U) / historyPartitions_;
        uint32_t startPos = partition * chunk;
        uint32_t endPos = startPos + chunk;
        if (startPos > oldLen) {
            startPos = oldLen;
        }
        if (endPos > oldLen) {
            endPos = oldLen;
        }

        for (uint32_t d = 0; d < headDim_; ++d) {
            accRot_[d] = 0.0F;
        }
        maxScore_ = -3.4028234663852886e38F;
        sum_ = 0.0F;
        currentWeight_ = 0.0F;
        initialized_ = false;

        if (startPos < endPos) {
            BuildQueryTransforms(b, head);
            ProcessHistoryTiledRange(b, kvHead, startPos, endPos);
        }
        StoreHistoryPartial(b, head, partition);
    }

    __aicore__ inline void ReduceHistoryPartialsAndStore(
        uint32_t b,
        uint32_t head)
    {
        uint32_t kvHead = head / qPerKv_;
        if (kvHead >= numKvHeads_) {
            return;
        }

        float globalMax = -3.4028234663852886e38F;
        for (uint32_t partition = 0; partition < historyPartitions_; ++partition) {
            uint64_t base = HistoryPartialBase(b, head, partition);
            float partialSum = scratchGm_.GetValue(base + 1U);
            if (partialSum > 0.0F) {
                float partialMax = scratchGm_.GetValue(base);
                if (partialMax > globalMax) {
                    globalMax = partialMax;
                }
            }
        }
        float curScore = CurrentScore(b, head, kvHead);
        if (curScore > globalMax) {
            globalMax = curScore;
        }

        for (uint32_t d = 0; d < headDim_; ++d) {
            accRot_[d] = 0.0F;
        }
        sum_ = 0.0F;
        maxScore_ = globalMax;
        currentWeight_ = 0.0F;
        initialized_ = true;

        for (uint32_t partition = 0; partition < historyPartitions_; ++partition) {
            uint64_t base = HistoryPartialBase(b, head, partition);
            float partialSum = scratchGm_.GetValue(base + 1U);
            if (partialSum <= 0.0F) {
                continue;
            }
            float partialMax = scratchGm_.GetValue(base);
            float partialScale = ExpScalar(partialMax - globalMax);
            sum_ += partialSum * partialScale;
            for (uint32_t d = 0; d < headDim_; ++d) {
                accRot_[d] += scratchGm_.GetValue(base + 8U + d) * partialScale;
            }
        }

        currentWeight_ = ExpScalar(curScore - globalMax);
        sum_ += currentWeight_;
        StoreOutput(b, head, kvHead);
    }

    __aicore__ inline void ProcessHistoryPartialPhase()
    {
        uint64_t coreCount = static_cast<uint64_t>(numCore_ == 0 ? 1 : numCore_);
        uint64_t partialTasks = static_cast<uint64_t>(batch_)
            * static_cast<uint64_t>(numHeads_)
            * static_cast<uint64_t>(historyPartitions_);
        uint64_t partialTasksPerCore =
            (partialTasks + coreCount - 1U) / coreCount;
        uint64_t partialStart =
            static_cast<uint64_t>(GetBlockIdx()) * partialTasksPerCore;
        uint64_t partialEnd = partialStart + partialTasksPerCore;
        if (partialEnd > partialTasks) {
            partialEnd = partialTasks;
        }
        for (uint64_t task = partialStart; task < partialEnd; ++task) {
            ProcessHistoryPartialTask(task);
        }
    }

    __aicore__ inline void ProcessHistoryReducePhase()
    {
        uint64_t coreCount = static_cast<uint64_t>(numCore_ == 0 ? 1 : numCore_);
        uint64_t reduceTasks = static_cast<uint64_t>(batch_)
            * static_cast<uint64_t>(numHeads_);
        uint64_t reduceTasksPerCore =
            (reduceTasks + coreCount - 1U) / coreCount;
        uint64_t reduceStart =
            static_cast<uint64_t>(GetBlockIdx()) * reduceTasksPerCore;
        uint64_t reduceEnd = reduceStart + reduceTasksPerCore;
        if (reduceEnd > reduceTasks) {
            reduceEnd = reduceTasks;
        }
        for (uint64_t task = reduceStart; task < reduceEnd; ++task) {
            ReduceHistoryPartialsAndStore(
                static_cast<uint32_t>(task / numHeads_),
                static_cast<uint32_t>(task % numHeads_));
        }
    }

    __aicore__ inline void ProcessHistoryGroup(
        uint32_t b,
        uint32_t kvHead,
        uint32_t oldLen)
    {
        uint32_t pos = 0;
        uint32_t blockOffset = 0;
        while (pos < oldLen) {
            int32_t blockId = blockTableGm_.GetValue(
                static_cast<uint64_t>(b) * maxBlocksPerSeq_ + blockOffset);
            uint32_t tokensInBlock = blockSize_;
            uint32_t remaining = oldLen - pos;
            if (tokensInBlock > remaining) {
                tokensInBlock = remaining;
            }
            uint64_t cacheBase =
                static_cast<uint64_t>(blockId) * blockSize_ * numKvHeads_
                + kvHead;
            for (uint32_t tokenOffset = 0; tokenOffset < tokensInBlock;
                 ++tokenOffset) {
                uint64_t cacheIndex =
                    cacheBase + static_cast<uint64_t>(tokenOffset) * numKvHeads_;
                HistoryScoresGroupByCacheIndex(cacheIndex);
                OnlineAccumulateHistoryGroup(cacheIndex);
            }
            pos += tokensInBlock;
            ++blockOffset;
        }
    }

    __aicore__ inline float ProcessHistoryScoreOnlyGroup(
        uint32_t b,
        uint32_t kvHead,
        uint32_t oldLen)
    {
        float sink = 0.0F;
        uint32_t pos = 0;
        uint32_t blockOffset = 0;
        while (pos < oldLen) {
            int32_t blockId = blockTableGm_.GetValue(
                static_cast<uint64_t>(b) * maxBlocksPerSeq_ + blockOffset);
            uint32_t tokensInBlock = blockSize_;
            uint32_t remaining = oldLen - pos;
            if (tokensInBlock > remaining) {
                tokensInBlock = remaining;
            }
            uint64_t cacheBase =
                static_cast<uint64_t>(blockId) * blockSize_ * numKvHeads_
                + kvHead;
            for (uint32_t tokenOffset = 0; tokenOffset < tokensInBlock;
                 ++tokenOffset) {
                uint64_t cacheIndex =
                    cacheBase + static_cast<uint64_t>(tokenOffset) * numKvHeads_;
                HistoryScoresGroupByCacheIndex(cacheIndex);
                sink +=
                    (scoreGroup_[0] + scoreGroup_[1] + scoreGroup_[2]
                     + scoreGroup_[3])
                    * 1.0e-6F;
            }
            pos += tokensInBlock;
            ++blockOffset;
        }
        return sink;
    }

    __aicore__ inline float ProcessHistoryScoreOnlineOnlyGroup(
        uint32_t b,
        uint32_t kvHead,
        uint32_t oldLen)
    {
        float sink = 0.0F;
        uint32_t pos = 0;
        uint32_t blockOffset = 0;
        while (pos < oldLen) {
            int32_t blockId = blockTableGm_.GetValue(
                static_cast<uint64_t>(b) * maxBlocksPerSeq_ + blockOffset);
            uint32_t tokensInBlock = blockSize_;
            uint32_t remaining = oldLen - pos;
            if (tokensInBlock > remaining) {
                tokensInBlock = remaining;
            }
            uint64_t cacheBase =
                static_cast<uint64_t>(blockId) * blockSize_ * numKvHeads_
                + kvHead;
            for (uint32_t tokenOffset = 0; tokenOffset < tokensInBlock;
                 ++tokenOffset) {
                uint64_t cacheIndex =
                    cacheBase + static_cast<uint64_t>(tokenOffset) * numKvHeads_;
                HistoryScoresGroupByCacheIndex(cacheIndex);
                OnlineWeightsGroup();
                sink +=
                    (weightGroup_[0] + weightGroup_[1] + weightGroup_[2]
                     + weightGroup_[3])
                    * 1.0e-6F;
            }
            pos += tokensInBlock;
            ++blockOffset;
        }
        return sink
            + (sumGroup_[0] + sumGroup_[1] + sumGroup_[2] + sumGroup_[3])
            * 1.0e-6F;
    }

    __aicore__ inline void CurrentScoresGroup(uint32_t b, uint32_t kvHead)
    {
        uint32_t qHeadBase = kvHead * qPerKv_;
        uint64_t kBase =
            (static_cast<uint64_t>(b) * numKvHeads_ + kvHead) * headDim_;
        for (uint32_t q = 0; q < 4U; ++q) {
            uint64_t qBase =
                (static_cast<uint64_t>(b) * numHeads_ + qHeadBase + q)
                * headDim_;
            float acc = 0.0F;
            for (uint32_t d = 0; d < headDim_; ++d) {
                acc += InputToFloat(queryGm_.GetValue(qBase + d))
                    * InputToFloat(keyGm_.GetValue(kBase + d));
            }
            scoreGroup_[q] = acc * scale_;
        }
    }

    __aicore__ inline void StoreOutput(uint32_t b, uint32_t head, uint32_t kvHead)
    {
        uint64_t outBase =
            (static_cast<uint64_t>(b) * numHeads_ + head) * headDim_;
        uint64_t valueBase =
            (static_cast<uint64_t>(b) * numKvHeads_ + kvHead) * headDim_;
        float invSum = sum_ > 0.0F ? 1.0F / sum_ : 0.0F;
        if (UseHadamardTransform()) {
            for (uint32_t d = 0; d < headDim_; ++d) {
                currentVec_[d] = accRot_[d] * invSum;
            }
            FwhtCurrentVec();
            float rotScale = HadamardScale();
            for (uint32_t d = 0; d < headDim_; ++d) {
                currentVec_[d] *= rotScale * MatrixSign(vRotationGm_, d);
            }
        } else {
            for (uint32_t d = 0; d < headDim_; ++d) {
                currentVec_[d] = 0.0F;
            }
            for (uint32_t inDim = 0; inDim < headDim_; ++inDim) {
                float acc = accRot_[inDim] * invSum;
                uint64_t matrixBase = static_cast<uint64_t>(inDim) * headDim_;
                for (uint32_t outDim = 0; outDim < headDim_; ++outDim) {
                    currentVec_[outDim] +=
                        acc * vRotationTGm_.GetValue(matrixBase + outDim);
                }
            }
        }
        for (uint32_t outDim = 0; outDim < headDim_; ++outDim) {
            // Current dense V is combined in original space; only compressed
            // history needs inverse rotation.
            currentVec_[outDim] +=
                currentWeight_ * invSum
                * InputToFloat(valueGm_.GetValue(valueBase + outDim));
            outGm_.SetValue(outBase + outDim, currentVec_[outDim]);
        }
    }

    __aicore__ inline void StoreDebugOutput(uint32_t b, uint32_t head, float value)
    {
        uint64_t outBase =
            (static_cast<uint64_t>(b) * numHeads_ + head) * headDim_;
        for (uint32_t d = 0; d < headDim_; ++d) {
            outGm_.SetValue(outBase + d, value);
        }
    }

    __aicore__ inline void StoreDebugOutputGroup(
        uint32_t b,
        uint32_t kvHead,
        float value)
    {
        uint32_t qHeadBase = kvHead * qPerKv_;
        for (uint32_t q = 0; q < 4U; ++q) {
            StoreDebugOutput(b, qHeadBase + q, value);
        }
    }

    __aicore__ inline void StoreOutputGroup(uint32_t b, uint32_t kvHead)
    {
        uint32_t qHeadBase = kvHead * qPerKv_;
        uint64_t outBase =
            (static_cast<uint64_t>(b) * numHeads_ + qHeadBase) * headDim_;
        uint64_t valueBase =
            (static_cast<uint64_t>(b) * numKvHeads_ + kvHead) * headDim_;
        uint32_t q1Base = headDim_;
        uint32_t q2Base = headDim_ << 1;
        uint32_t q3Base = 3U * headDim_;
        float invSum0 = sumGroup_[0] > 0.0F ? 1.0F / sumGroup_[0] : 0.0F;
        float invSum1 = sumGroup_[1] > 0.0F ? 1.0F / sumGroup_[1] : 0.0F;
        float invSum2 = sumGroup_[2] > 0.0F ? 1.0F / sumGroup_[2] : 0.0F;
        float invSum3 = sumGroup_[3] > 0.0F ? 1.0F / sumGroup_[3] : 0.0F;

        ZeroGroupedScratch();
        for (uint32_t inDim = 0; inDim < headDim_; ++inDim) {
            float acc0 = accRotGroup_[inDim] * invSum0;
            float acc1 = accRotGroup_[q1Base + inDim] * invSum1;
            float acc2 = accRotGroup_[q2Base + inDim] * invSum2;
            float acc3 = accRotGroup_[q3Base + inDim] * invSum3;
            uint64_t matrixBase = static_cast<uint64_t>(inDim) * headDim_;
            for (uint32_t outDim = 0; outDim < headDim_; ++outDim) {
                float rot = vRotationTGm_.GetValue(matrixBase + outDim);
                qRotGroup_[outDim] += acc0 * rot;
                qRotGroup_[q1Base + outDim] += acc1 * rot;
                qRotGroup_[q2Base + outDim] += acc2 * rot;
                qRotGroup_[q3Base + outDim] += acc3 * rot;
            }
        }
        for (uint32_t outDim = 0; outDim < headDim_; ++outDim) {
            float value = InputToFloat(valueGm_.GetValue(valueBase + outDim));
            qRotGroup_[outDim] += currentWeightGroup_[0] * invSum0 * value;
            qRotGroup_[q1Base + outDim] +=
                currentWeightGroup_[1] * invSum1 * value;
            qRotGroup_[q2Base + outDim] +=
                currentWeightGroup_[2] * invSum2 * value;
            qRotGroup_[q3Base + outDim] +=
                currentWeightGroup_[3] * invSum3 * value;
            outGm_.SetValue(outBase + outDim, qRotGroup_[outDim]);
            outGm_.SetValue(
                outBase + q1Base + outDim, qRotGroup_[q1Base + outDim]);
            outGm_.SetValue(
                outBase + q2Base + outDim, qRotGroup_[q2Base + outDim]);
            outGm_.SetValue(
                outBase + q3Base + outDim, qRotGroup_[q3Base + outDim]);
        }
    }

    __aicore__ inline void ResetGroupedAttentionState()
    {
        for (uint32_t q = 0; q < 4U; ++q) {
            uint32_t base = q * headDim_;
            for (uint32_t d = 0; d < headDim_; ++d) {
                accRotGroup_[base + d] = 0.0F;
            }
            maxScoreGroup_[q] = -3.4028234663852886e38F;
            sumGroup_[q] = 0.0F;
            currentWeightGroup_[q] = 0.0F;
            initializedGroup_[q] = false;
            weightGroup_[q] = 0.0F;
            scoreGroup_[q] = 0.0F;
        }
    }

    __aicore__ inline void ProcessOne(uint32_t b, uint32_t head)
    {
        if (headDim_ == 0U || headDim_ > 256U || numKvHeads_ == 0U
            || qPerKv_ == 0U || kStage1Bits_ == 0U
            || kStage1Bits_ > 4U || vBits_ == 0U
            || vBits_ > 4U || blockSize_ == 0U
            || maxBlocksPerSeq_ == 0U) {
            return;
        }

        uint32_t kvHead = head / qPerKv_;
        if (kvHead >= numKvHeads_) {
            return;
        }
        // Current dense K/V is used directly by this attention step; the cache
        // write is for future steps, so no cross-core read-after-write barrier
        // is required here.
        uint32_t groupHead = head - kvHead * qPerKv_;
        int64_t slot = slotMappingGm_.GetValue(b);
        if (slot >= 0 && skipCacheUpdate_ == 0U) {
            uint64_t slotHead =
                static_cast<uint64_t>(slot) * numKvHeads_ + kvHead;
            if (CanPartitionCurrentKQjlEncode()) {
                PrepareCurrentKResidual(b, kvHead, slotHead, groupHead == 0U);
                EncodeCurrentKQjlRange(slotHead, groupHead);
            } else if (groupHead == 0U) {
                EncodeCurrentK(b, kvHead, slotHead);
            }
            if (CanPartitionCurrentVEncode()) {
                EncodeCurrentVRange(b, kvHead, slotHead, groupHead);
            } else if (groupHead == 0U) {
                EncodeCurrentV(b, kvHead, slotHead);
            }
        }

        int32_t oldLenRaw = oldSeqLensGm_.GetValue(b);
        uint32_t oldLen = oldLenRaw > 0 ? static_cast<uint32_t>(oldLenRaw) : 0U;
        if (oldLen > maxSeqLen_) {
            oldLen = maxSeqLen_;
        }

        if (debugMode_ == 6U) {
            StoreDebugOutput(b, head, 0.0F);
            return;
        }
        if (debugMode_ == 8U || debugMode_ == 9U) {
            for (uint32_t d = 0; d < headDim_; ++d) {
                accRot_[d] = 0.0F;
            }
            maxScore_ = -3.4028234663852886e38F;
            sum_ = 0.0F;
            currentWeight_ = 0.0F;
            initialized_ = false;
            if (debugMode_ == 9U) {
                float curScore = CurrentScore(b, head, kvHead);
                OnlineAccumulateCurrent(curScore);
            }
            StoreOutput(b, head, kvHead);
            return;
        }

        BuildQueryTransforms(b, head);
        for (uint32_t d = 0; d < headDim_; ++d) {
            accRot_[d] = 0.0F;
        }
        maxScore_ = -3.4028234663852886e38F;
        sum_ = 0.0F;
        currentWeight_ = 0.0F;
        initialized_ = false;

        if (debugMode_ == 7U) {
            StoreDebugOutput(b, head, (qRot_[0] + qQjl_[0]) * 1.0e-6F);
            return;
        }
        if (debugMode_ == 1U) {
            float curScore = CurrentScore(b, head, kvHead);
            OnlineAccumulateCurrent(curScore);
            StoreOutput(b, head, kvHead);
            return;
        }
        if (debugMode_ == 2U) {
            float sink = ProcessHistoryScoreOnly(b, kvHead, oldLen);
            StoreDebugOutput(b, head, sink);
            return;
        }
        if (debugMode_ == 3U) {
            float sink = ProcessHistoryScoreOnlineOnly(b, kvHead, oldLen);
            StoreDebugOutput(b, head, sink);
            return;
        }

        ProcessHistoryTiled(b, kvHead, oldLen);

        if (debugMode_ == 4U) {
            StoreDebugOutput(b, head, sum_ * 1.0e-6F + accRot_[0] * 1.0e-6F);
            return;
        }
        float curScore = CurrentScore(b, head, kvHead);
        OnlineAccumulateCurrent(curScore);
        if (debugMode_ == 5U) {
            StoreDebugOutput(
                b, head, sum_ * 1.0e-6F + currentWeight_ * 1.0e-6F);
            return;
        }
        StoreOutput(b, head, kvHead);
    }

    __aicore__ inline void ProcessGroupedKvHead(uint32_t b, uint32_t kvHead)
    {
        if (headDim_ == 0U || headDim_ > 256U || numKvHeads_ == 0U
            || qPerKv_ != 4U || kStage1Bits_ == 0U
            || kStage1Bits_ > 4U || vBits_ == 0U
            || vBits_ > 4U || blockSize_ == 0U
            || maxBlocksPerSeq_ == 0U || kvHead >= numKvHeads_
            || kvHead * qPerKv_ + 3U >= numHeads_) {
            return;
        }

        int64_t slot = slotMappingGm_.GetValue(b);
        if (slot >= 0 && skipCacheUpdate_ == 0U) {
            uint64_t slotHead =
                static_cast<uint64_t>(slot) * numKvHeads_ + kvHead;
            EncodeCurrentK(b, kvHead, slotHead);
            EncodeCurrentV(b, kvHead, slotHead);
        }

        int32_t oldLenRaw = oldSeqLensGm_.GetValue(b);
        uint32_t oldLen = oldLenRaw > 0 ? static_cast<uint32_t>(oldLenRaw) : 0U;
        if (oldLen > maxSeqLen_) {
            oldLen = maxSeqLen_;
        }

        if (debugMode_ == 6U) {
            StoreDebugOutputGroup(b, kvHead, 0.0F);
            return;
        }
        if (debugMode_ == 8U || debugMode_ == 9U) {
            ResetGroupedAttentionState();
            if (debugMode_ == 9U) {
                CurrentScoresGroup(b, kvHead);
                OnlineAccumulateCurrentGroup();
            }
            StoreOutputGroup(b, kvHead);
            return;
        }
        BuildGroupedQueryTransforms(b, kvHead);
        ResetGroupedAttentionState();

        if (debugMode_ == 7U) {
            StoreDebugOutputGroup(
                b, kvHead, (qRotGroup_[0] + qQjlGroup_[0]) * 1.0e-6F);
            return;
        }
        if (debugMode_ == 1U) {
            CurrentScoresGroup(b, kvHead);
            OnlineAccumulateCurrentGroup();
            StoreOutputGroup(b, kvHead);
            return;
        }
        if (debugMode_ == 2U) {
            float sink = ProcessHistoryScoreOnlyGroup(b, kvHead, oldLen);
            StoreDebugOutputGroup(b, kvHead, sink);
            return;
        }
        if (debugMode_ == 3U) {
            float sink = ProcessHistoryScoreOnlineOnlyGroup(b, kvHead, oldLen);
            StoreDebugOutputGroup(b, kvHead, sink);
            return;
        }

        ProcessHistoryGroup(b, kvHead, oldLen);

        if (debugMode_ == 4U) {
            StoreDebugOutputGroup(
                b,
                kvHead,
                (sumGroup_[0] + sumGroup_[1] + sumGroup_[2] + sumGroup_[3])
                    * 1.0e-6F
                    + accRotGroup_[0] * 1.0e-6F);
            return;
        }
        CurrentScoresGroup(b, kvHead);
        OnlineAccumulateCurrentGroup();
        if (debugMode_ == 5U) {
            StoreDebugOutputGroup(
                b,
                kvHead,
                (sumGroup_[0] + sumGroup_[1] + sumGroup_[2] + sumGroup_[3])
                    * 1.0e-6F
                    + (currentWeightGroup_[0] + currentWeightGroup_[1]
                       + currentWeightGroup_[2] + currentWeightGroup_[3])
                        * 1.0e-6F);
            return;
        }
        StoreOutputGroup(b, kvHead);
    }

    __aicore__ inline void Process()
    {
        bool useGrouped = groupedQ_ != 0U && qPerKv_ == 4U;
        bool useHistoryParallel = historyPartitions_ > 1U
            && !useGrouped
            && debugMode_ == 0U
            && headDim_ > 0U
            && headDim_ <= 256U
            && numKvHeads_ > 0U
            && qPerKv_ > 0U
            && kStage1Bits_ > 0U
            && kStage1Bits_ <= 4U
            && vBits_ > 0U
            && vBits_ <= 4U
            && blockSize_ > 0U
            && maxBlocksPerSeq_ > 0U;
        if (useHistoryParallel) {
            if (historyPartitionPhase_ == 1U) {
                ProcessHistoryPartialPhase();
                return;
            }
            if (historyPartitionPhase_ == 2U) {
                ProcessHistoryReducePhase();
                return;
            }
        }
        uint64_t totalTasks = static_cast<uint64_t>(batch_)
            * static_cast<uint64_t>(useGrouped ? numKvHeads_ : numHeads_);
        if (totalTasks == 0U) {
            return;
        }
        uint64_t coreCount = static_cast<uint64_t>(numCore_ == 0 ? 1 : numCore_);
        uint64_t tasksPerCore = (totalTasks + coreCount - 1U) / coreCount;
        uint64_t start = static_cast<uint64_t>(GetBlockIdx()) * tasksPerCore;
        uint64_t end = start + tasksPerCore;
        if (end > totalTasks) {
            end = totalTasks;
        }
        for (uint64_t task = start; task < end; ++task) {
            if (useGrouped) {
                ProcessGroupedKvHead(
                    static_cast<uint32_t>(task / numKvHeads_),
                    static_cast<uint32_t>(task % numKvHeads_));
            } else {
                ProcessOne(
                    static_cast<uint32_t>(task / numHeads_),
                    static_cast<uint32_t>(task % numHeads_));
            }
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> expBuf_;

    GlobalTensor<QueryT> queryGm_;
    GlobalTensor<KeyT> keyGm_;
    GlobalTensor<ValueT> valueGm_;
    GlobalTensor<int64_t> slotMappingGm_;
    GlobalTensor<uint8_t> kPackedIdxGm_;
    GlobalTensor<uint8_t> kPackedQjlGm_;
    GlobalTensor<float> kGammaGm_;
    GlobalTensor<float> kNormGm_;
    GlobalTensor<uint8_t> vPackedIdxGm_;
    GlobalTensor<float> vNormGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> oldSeqLensGm_;
    GlobalTensor<float> kRotationGm_;
    GlobalTensor<float> kQjlQueryMatrixGm_;
    GlobalTensor<float> kQjlProjTGm_;
    GlobalTensor<float> kBoundaryGm_;
    GlobalTensor<float> vRotationGm_;
    GlobalTensor<float> vRotationTGm_;
    GlobalTensor<float> vBoundaryGm_;
    GlobalTensor<float> kCodebookGm_;
    GlobalTensor<float> vCodebookGm_;
    GlobalTensor<float> scratchGm_;
    GlobalTensor<float> outGm_;

    uint32_t batch_{0};
    uint32_t numHeads_{0};
    uint32_t numKvHeads_{0};
    uint32_t qPerKv_{1};
    uint32_t blockSize_{0};
    uint32_t maxBlocksPerSeq_{0};
    uint32_t maxSeqLen_{0};
    uint32_t scoreTileLen_{0};
    uint32_t groupedQ_{0};
    uint32_t skipCacheUpdate_{0};
    uint32_t debugMode_{0};
    uint32_t pretransformedQuery_{0};
    uint32_t historyPartitions_{1};
    uint32_t historyPartitionPhase_{0};
    uint32_t transformMode_{0};
    uint32_t headDim_{0};
    uint32_t kPackedCols_{0};
    uint32_t kQjlCols_{0};
    uint32_t vPackedCols_{0};
    uint32_t kStage1Bits_{0};
    uint32_t vBits_{0};
    uint32_t numCore_{0};
    float scale_{1.0F};
    float correction_{0.0F};

    float qRot_[256];
    float qQjl_[256];
    float accRot_[256];
    float scoreTile_[64];
    float qRotGroup_[1024];
    float qQjlGroup_[1024];
    float accRotGroup_[1024];
    float scoreGroup_[4];
    float weightGroup_[4];
    float maxScoreGroup_[4];
    float sumGroup_[4];
    float currentWeightGroup_[4];
    float currentVec_[256];
    float kResidual_[256];
    float kCodebook_[16];
    float vCodebook_[16];
    float kBoundary_[16];
    float vBoundary_[16];
    float maxScore_{0.0F};
    float sum_{0.0F};
    float currentWeight_{0.0F};
    bool initialized_{false};
    bool initializedGroup_[4];
};

extern "C" __global__ __aicore__ void tq_fused_kv_update_attention_decode(
    GM_ADDR query,
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR slotMapping,
    GM_ADDR kPackedIdx,
    GM_ADDR kPackedQjl,
    GM_ADDR kGamma,
    GM_ADDR kNorm,
    GM_ADDR vPackedIdx,
    GM_ADDR vNorm,
    GM_ADDR blockTable,
    GM_ADDR oldSeqLens,
    GM_ADDR kRotation,
    GM_ADDR kQjlQueryMatrix,
    GM_ADDR kQjlProjT,
    GM_ADDR kBoundary,
    GM_ADDR vRotation,
    GM_ADDR vRotationT,
    GM_ADDR vBoundary,
    GM_ADDR kCodebook,
    GM_ADDR vCodebook,
    GM_ADDR scratch,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TqFusedKvUpdateAttentionDecodeTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        TqFusedKvUpdateAttentionDecodeTilingData, tilingData, tiling);

    KernelTqFusedKvUpdateAttentionDecode<DTYPE_QUERY, DTYPE_KEY, DTYPE_VALUE> op;
    op.Init(
        query,
        key,
        value,
        slotMapping,
        kPackedIdx,
        kPackedQjl,
        kGamma,
        kNorm,
        vPackedIdx,
        vNorm,
        blockTable,
        oldSeqLens,
        kRotation,
        kQjlQueryMatrix,
        kQjlProjT,
        kBoundary,
        vRotation,
        vRotationT,
        vBoundary,
        kCodebook,
        vCodebook,
        scratch,
        out,
        &tilingData);
    op.Process();
}
