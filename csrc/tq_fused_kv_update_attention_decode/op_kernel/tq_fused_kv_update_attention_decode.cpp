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
        GM_ADDR out,
        const TqFusedKvUpdateAttentionDecodeTilingData* tiling)
    {
        queryGm_.SetGlobalBuffer((__gm__ float*)query);
        keyGm_.SetGlobalBuffer((__gm__ float*)key);
        valueGm_.SetGlobalBuffer((__gm__ float*)value);
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
        outGm_.SetGlobalBuffer((__gm__ float*)out);

        batch_ = tiling->batch;
        numHeads_ = tiling->numHeads;
        numKvHeads_ = tiling->numKvHeads;
        qPerKv_ = tiling->qPerKv;
        blockSize_ = tiling->blockSize;
        maxBlocksPerSeq_ = tiling->maxBlocksPerSeq;
        maxSeqLen_ = tiling->maxSeqLen;
        headDim_ = tiling->headDim;
        kPackedCols_ = tiling->kPackedCols;
        kQjlCols_ = tiling->kQjlCols;
        vPackedCols_ = tiling->vPackedCols;
        kStage1Bits_ = tiling->kStage1Bits;
        vBits_ = tiling->vBits;
        numCore_ = tiling->numCore;
        scale_ = tiling->scale;
        correction_ = tiling->correction;

        pipe_.InitBuffer(expBuf_, 8U * sizeof(float));
        LoadSmallParams();
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

    __aicore__ inline uint64_t CacheIndex(uint32_t b, uint32_t kvHead, uint32_t pos)
    {
        uint32_t blockOffset = pos / blockSize_;
        uint32_t tokenOffset = pos - blockOffset * blockSize_;
        int32_t blockId = blockTableGm_.GetValue(
            static_cast<uint64_t>(b) * maxBlocksPerSeq_ + blockOffset);
        return ((static_cast<uint64_t>(blockId) * blockSize_ + tokenOffset)
            * numKvHeads_ + kvHead);
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

    __aicore__ inline void LoadCurrentVector(
        GlobalTensor<float>& tensor,
        uint32_t b,
        uint32_t kvHead)
    {
        uint64_t base =
            (static_cast<uint64_t>(b) * numKvHeads_ + kvHead) * headDim_;
        for (uint32_t d = 0; d < headDim_; ++d) {
            currentVec_[d] = tensor.GetValue(base + d);
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
        float residual[256];

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

        for (uint32_t d = 0; d < headDim_; ++d) {
            float xRot = CalcCurrentRotFromBuffer(kRotationGm_, d, invNorm);
            uint32_t idx = KBoundaryIndex(xRot);
            PackIndex(idxPacked, kPackedCols_, kStage1Bits_, d, idx);
            float xHat = kCodebook_[idx];
            residual[d] = xRot - xHat;
            gammaSq += residual[d] * residual[d];
        }

        float gamma = gammaSq > 0.0F ? SqrtScalar(gammaSq) : 0.0F;
        for (uint32_t j = 0; j < headDim_; ++j) {
            float sum = 0.0F;
            for (uint32_t d = 0; d < headDim_; ++d) {
                float p = kQjlProjTGm_.GetValue(
                    static_cast<uint64_t>(d) * headDim_ + j);
                sum += residual[d] * p;
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
        for (uint32_t d = 0; d < headDim_; ++d) {
            float xRot = CalcCurrentRotFromBuffer(kRotationGm_, d, invNorm);
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
        for (uint32_t d = 0; d < headDim_; ++d) {
            float xRot = CalcCurrentRotFromBuffer(vRotationGm_, d, invNorm);
            PackIndex(
                packed,
                vPackedCols_,
                vBits_,
                d,
                VBoundaryIndex(xRot));
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
        for (uint32_t d = dimStart; d < dimEnd; ++d) {
            float xRot = CalcCurrentRotFromBuffer(vRotationGm_, d, invNorm);
            PackIndex(
                packed,
                vPackedCols_,
                vBits_,
                d,
                VBoundaryIndex(xRot));
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
        for (uint32_t outDim = 0; outDim < headDim_; ++outDim) {
            float qRot = 0.0F;
            float qQjl = 0.0F;
            for (uint32_t inDim = 0; inDim < headDim_; ++inDim) {
                float q = queryGm_.GetValue(qBase + inDim);
                uint64_t matrixIndex =
                    static_cast<uint64_t>(inDim) * headDim_ + outDim;
                qRot += q * kRotationGm_.GetValue(matrixIndex);
                qQjl += q * kQjlQueryMatrixGm_.GetValue(matrixIndex);
            }
            qRot_[outDim] = qRot;
            qQjl_[outDim] = qQjl;
        }
    }

    __aicore__ inline float HistoryScore(uint32_t b, uint32_t kvHead, uint32_t pos)
    {
        uint64_t cacheIndex = CacheIndex(b, kvHead, pos);
        uint64_t idxBase = cacheIndex * kPackedCols_;
        uint64_t qjlBase = cacheIndex * kQjlCols_;
        float gamma = kGammaGm_.GetValue(cacheIndex);
        float norm = kNormGm_.GetValue(cacheIndex);
        float mseAcc = 0.0F;
        float qjlAcc = 0.0F;
        for (uint32_t d = 0; d < headDim_; ++d) {
            uint32_t idx = ExtractBits(kPackedIdxGm_, idxBase, d, kStage1Bits_);
            uint32_t qjl = ExtractBits(kPackedQjlGm_, qjlBase, d, 1U);
            float sign = qjl != 0U ? 1.0F : -1.0F;
            mseAcc += qRot_[d] * kCodebook_[idx];
            qjlAcc += qQjl_[d] * sign;
        }
        return (mseAcc + correction_ * gamma * qjlAcc) * norm * scale_;
    }

    __aicore__ inline void LoadHistoryVRot(
        uint32_t b,
        uint32_t kvHead,
        uint32_t pos)
    {
        uint64_t cacheIndex = CacheIndex(b, kvHead, pos);
        uint64_t vBase = cacheIndex * vPackedCols_;
        float norm = vNormGm_.GetValue(cacheIndex);
        for (uint32_t d = 0; d < headDim_; ++d) {
            uint32_t idx = ExtractBits(vPackedIdxGm_, vBase, d, vBits_);
            vTmp_[d] = vCodebook_[idx] * norm;
        }
    }

    __aicore__ inline float CurrentScore(uint32_t b, uint32_t head, uint32_t kvHead)
    {
        uint64_t qBase =
            (static_cast<uint64_t>(b) * numHeads_ + head) * headDim_;
        uint64_t kBase =
            (static_cast<uint64_t>(b) * numKvHeads_ + kvHead) * headDim_;
        float acc = 0.0F;
        for (uint32_t d = 0; d < headDim_; ++d) {
            acc += queryGm_.GetValue(qBase + d) * keyGm_.GetValue(kBase + d);
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

    __aicore__ inline void OnlineAccumulateHistory(float score)
    {
        float weight = OnlineStep(score);
        for (uint32_t d = 0; d < headDim_; ++d) {
            accRot_[d] += weight * vTmp_[d];
        }
    }

    __aicore__ inline void OnlineAccumulateCurrent(float score)
    {
        currentWeight_ += OnlineStep(score);
    }

    __aicore__ inline void StoreOutput(uint32_t b, uint32_t head, uint32_t kvHead)
    {
        uint64_t outBase =
            (static_cast<uint64_t>(b) * numHeads_ + head) * headDim_;
        uint64_t valueBase =
            (static_cast<uint64_t>(b) * numKvHeads_ + kvHead) * headDim_;
        float invSum = sum_ > 0.0F ? 1.0F / sum_ : 0.0F;
        for (uint32_t outDim = 0; outDim < headDim_; ++outDim) {
            float acc = 0.0F;
            for (uint32_t inDim = 0; inDim < headDim_; ++inDim) {
                acc += (accRot_[inDim] * invSum) * vRotationTGm_.GetValue(
                    static_cast<uint64_t>(inDim) * headDim_ + outDim);
            }
            // Current dense V is combined in original space; only compressed
            // history needs inverse rotation.
            acc += currentWeight_ * invSum * valueGm_.GetValue(valueBase + outDim);
            outGm_.SetValue(outBase + outDim, acc);
        }
    }

    __aicore__ inline void ProcessOne(uint32_t b, uint32_t head)
    {
        if (headDim_ == 0U || headDim_ > 256U || numKvHeads_ == 0U
            || kStage1Bits_ == 0U || kStage1Bits_ > 4U || vBits_ == 0U
            || vBits_ > 4U) {
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
        if (slot >= 0) {
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

        BuildQueryTransforms(b, head);
        for (uint32_t d = 0; d < headDim_; ++d) {
            accRot_[d] = 0.0F;
        }
        maxScore_ = -3.4028234663852886e38F;
        sum_ = 0.0F;
        currentWeight_ = 0.0F;
        initialized_ = false;

        for (uint32_t pos = 0; pos < oldLen; ++pos) {
            float score = HistoryScore(b, kvHead, pos);
            LoadHistoryVRot(b, kvHead, pos);
            OnlineAccumulateHistory(score);
        }

        float curScore = CurrentScore(b, head, kvHead);
        OnlineAccumulateCurrent(curScore);
        StoreOutput(b, head, kvHead);
    }

    __aicore__ inline void Process()
    {
        uint64_t totalTasks =
            static_cast<uint64_t>(batch_) * static_cast<uint64_t>(numHeads_);
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
            ProcessOne(
                static_cast<uint32_t>(task / numHeads_),
                static_cast<uint32_t>(task % numHeads_));
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> expBuf_;

    GlobalTensor<float> queryGm_;
    GlobalTensor<float> keyGm_;
    GlobalTensor<float> valueGm_;
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
    GlobalTensor<float> outGm_;

    uint32_t batch_{0};
    uint32_t numHeads_{0};
    uint32_t numKvHeads_{0};
    uint32_t qPerKv_{1};
    uint32_t blockSize_{0};
    uint32_t maxBlocksPerSeq_{0};
    uint32_t maxSeqLen_{0};
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
    float vTmp_[256];
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
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TqFusedKvUpdateAttentionDecodeTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        TqFusedKvUpdateAttentionDecodeTilingData, tilingData, tiling);

    KernelTqFusedKvUpdateAttentionDecode op;
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
        out,
        &tilingData);
    op.Process();
}
