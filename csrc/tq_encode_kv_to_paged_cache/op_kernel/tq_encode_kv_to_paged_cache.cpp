/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 * Ascend C kernel: TurboQuant prod-K + MSE-V encode, pack, and paged cache write.
 */

#include <cmath>
#include "kernel_operator.h"

using namespace AscendC;

struct TqEncodeKvToPagedCacheTilingData {
    uint32_t totalTokens;
    uint32_t numKvHeads;
    uint32_t blockSize;
    uint32_t kPackedCols;
    uint32_t kQjlCols;
    uint32_t vPackedCols;
    uint32_t totalBits;
    uint32_t stage1Bits;
    uint32_t vBits;
    uint32_t headDim;
    uint32_t debugMode;
    uint32_t vPartitionCount;
    uint32_t numCore;
};

template <typename KeyT, typename ValueT>
class KernelTqEncodeKvToPagedCache {
public:
    __aicore__ inline KernelTqEncodeKvToPagedCache() {}

    __aicore__ inline void Init(
        GM_ADDR key,
        GM_ADDR value,
        GM_ADDR slotMapping,
        GM_ADDR kIdxCache,
        GM_ADDR kQjlCache,
        GM_ADDR kGammaCache,
        GM_ADDR kNormCache,
        GM_ADDR vIdxCache,
        GM_ADDR vNormCache,
        GM_ADDR kRotation,
        GM_ADDR kBoundary,
        GM_ADDR kCodebook,
        GM_ADDR kQjlProjT,
        GM_ADDR vRotation,
        GM_ADDR vBoundary,
        const TqEncodeKvToPagedCacheTilingData* tiling)
    {
        keyGm_.SetGlobalBuffer((__gm__ KeyT*)key);
        valueGm_.SetGlobalBuffer((__gm__ ValueT*)value);
        slotMappingGm_.SetGlobalBuffer((__gm__ int64_t*)slotMapping);
        kIdxCacheGm_.SetGlobalBuffer((__gm__ uint8_t*)kIdxCache);
        kQjlCacheGm_.SetGlobalBuffer((__gm__ uint8_t*)kQjlCache);
        kGammaCacheGm_.SetGlobalBuffer((__gm__ float*)kGammaCache);
        kNormCacheGm_.SetGlobalBuffer((__gm__ float*)kNormCache);
        vIdxCacheGm_.SetGlobalBuffer((__gm__ uint8_t*)vIdxCache);
        vNormCacheGm_.SetGlobalBuffer((__gm__ float*)vNormCache);
        kRotationGm_.SetGlobalBuffer((__gm__ float*)kRotation);
        kBoundaryGm_.SetGlobalBuffer((__gm__ float*)kBoundary);
        kCodebookGm_.SetGlobalBuffer((__gm__ float*)kCodebook);
        kQjlProjTGm_.SetGlobalBuffer((__gm__ float*)kQjlProjT);
        vRotationGm_.SetGlobalBuffer((__gm__ float*)vRotation);
        vBoundaryGm_.SetGlobalBuffer((__gm__ float*)vBoundary);

        totalTokens_ = tiling->totalTokens;
        numKvHeads_ = tiling->numKvHeads;
        blockSize_ = tiling->blockSize;
        kPackedCols_ = tiling->kPackedCols;
        kQjlCols_ = tiling->kQjlCols;
        vPackedCols_ = tiling->vPackedCols;
        totalBits_ = tiling->totalBits;
        stage1Bits_ = tiling->stage1Bits;
        vBits_ = tiling->vBits;
        headDim_ = tiling->headDim;
        debugMode_ = tiling->debugMode;
        vPartitionCount_ = tiling->vPartitionCount;
        numCore_ = tiling->numCore;
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

    template <typename InT>
    __aicore__ inline float ReadX(
        GlobalTensor<InT>& tensor,
        uint32_t token,
        uint32_t kvHead,
        uint32_t d)
    {
        uint64_t index =
            (static_cast<uint64_t>(token) * numKvHeads_ + kvHead) * headDim_ + d;
        return InputToFloat(tensor.GetValue(index));
    }

    template <typename InT>
    __aicore__ inline void LoadCurrentVector(
        GlobalTensor<InT>& tensor,
        uint32_t token,
        uint32_t kvHead)
    {
        for (uint32_t d = 0; d < headDim_; ++d) {
            currentVec_[d] = ReadX(tensor, token, kvHead, d);
        }
    }

    __aicore__ inline void LoadSmallParams()
    {
        uint32_t kLevels = stage1Bits_ < 4U ? (1U << stage1Bits_) : 16U;
        uint32_t vLevels = vBits_ < 4U ? (1U << vBits_) : 16U;
        for (uint32_t i = 0; i < kLevels; ++i) {
            kCodebook_[i] = kCodebookGm_.GetValue(i);
        }
        for (uint32_t i = 0; i + 1U < kLevels; ++i) {
            kBoundary_[i] = kBoundaryGm_.GetValue(i);
        }
        for (uint32_t i = 0; i + 1U < vLevels; ++i) {
            vBoundary_[i] = vBoundaryGm_.GetValue(i);
        }
    }

    __aicore__ inline float CalcNorm()
    {
        float sum = 0.0F;
        for (uint32_t d = 0; d < headDim_; ++d) {
            float v = currentVec_[d];
            sum += v * v;
        }
        if (sum < 1.0e-12F) {
            sum = 1.0e-12F;
        }
        return sqrt(sum);
    }

    __aicore__ inline void CalcRotRange(
        GlobalTensor<float>& rotation,
        float invNorm,
        uint32_t dimStart,
        uint32_t dimEnd)
    {
        for (uint32_t d = dimStart; d < dimEnd; ++d) {
            kResidual_[d] = 0.0F;
        }
        for (uint32_t inDim = 0; inDim < headDim_; ++inDim) {
            float x = currentVec_[inDim] * invNorm;
            uint64_t matrixBase = static_cast<uint64_t>(inDim) * headDim_;
            for (uint32_t outDim = dimStart; outDim < dimEnd; ++outDim) {
                kResidual_[outDim] +=
                    x * rotation.GetValue(matrixBase + outDim);
            }
        }
    }

    __aicore__ inline uint32_t KBoundaryIndex(float x)
    {
        uint32_t levels = (1U << stage1Bits_) - 1U;
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

    __aicore__ inline void EncodeK(
        uint32_t token,
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

        LoadCurrentVector(keyGm_, token, kvHead);
        float norm = CalcNorm();
        float invNorm = 1.0F / norm;
        float gammaSq = 0.0F;
        CalcRotRange(kRotationGm_, invNorm, 0U, headDim_);
        for (uint32_t d = 0; d < headDim_; ++d) {
            float xRot = kResidual_[d];
            uint32_t idx = KBoundaryIndex(xRot);
            PackIndex(idxPacked, kPackedCols_, stage1Bits_, d, idx);
            float residual = xRot - kCodebook_[idx];
            kResidual_[d] = residual;
            gammaSq += residual * residual;
        }

        float gamma = gammaSq > 0.0F ? sqrt(gammaSq) : 0.0F;
        for (uint32_t j = 0; j < headDim_; ++j) {
            currentVec_[j] = 0.0F;
        }
        for (uint32_t d = 0; d < headDim_; ++d) {
            float residual = kResidual_[d];
            uint64_t matrixBase = static_cast<uint64_t>(d) * headDim_;
            for (uint32_t j = 0; j < headDim_; ++j) {
                currentVec_[j] +=
                    residual * kQjlProjTGm_.GetValue(matrixBase + j);
            }
        }
        for (uint32_t j = 0; j < headDim_; ++j) {
            PackIndex(
                qjlPacked,
                kQjlCols_,
                1U,
                j,
                currentVec_[j] >= 0.0F ? 1U : 0U);
        }

        uint64_t idxBase = slotHead * kPackedCols_;
        uint64_t qjlBase = slotHead * kQjlCols_;
        for (uint32_t col = 0; col < kPackedCols_; ++col) {
            kIdxCacheGm_.SetValue(idxBase + col, idxPacked[col]);
        }
        for (uint32_t col = 0; col < kQjlCols_; ++col) {
            kQjlCacheGm_.SetValue(qjlBase + col, qjlPacked[col]);
        }
        kGammaCacheGm_.SetValue(slotHead, gamma);
        kNormCacheGm_.SetValue(slotHead, norm);
    }

    __aicore__ inline void EncodeKRotateOnly(
        uint32_t token,
        uint32_t kvHead,
        uint64_t slotHead)
    {
        LoadCurrentVector(keyGm_, token, kvHead);
        float norm = CalcNorm();
        float invNorm = 1.0F / norm;
        CalcRotRange(kRotationGm_, invNorm, 0U, headDim_);
        kNormCacheGm_.SetValue(slotHead, norm + kResidual_[0] * 1.0e-6F);
    }

    __aicore__ inline void EncodeKLoadNormOnly(
        uint32_t token,
        uint32_t kvHead,
        uint64_t slotHead)
    {
        LoadCurrentVector(keyGm_, token, kvHead);
        float norm = CalcNorm();
        kNormCacheGm_.SetValue(slotHead, norm + currentVec_[0] * 1.0e-6F);
    }

    __aicore__ inline void EncodeKStage1Only(
        uint32_t token,
        uint32_t kvHead,
        uint64_t slotHead,
        bool writeCache)
    {
        uint8_t idxPacked[128];
        for (uint32_t col = 0; col < kPackedCols_; ++col) {
            idxPacked[col] = 0;
        }

        LoadCurrentVector(keyGm_, token, kvHead);
        float norm = CalcNorm();
        float invNorm = 1.0F / norm;
        float gammaSq = 0.0F;
        CalcRotRange(kRotationGm_, invNorm, 0U, headDim_);
        for (uint32_t d = 0; d < headDim_; ++d) {
            float xRot = kResidual_[d];
            uint32_t idx = KBoundaryIndex(xRot);
            PackIndex(idxPacked, kPackedCols_, stage1Bits_, d, idx);
            float residual = xRot - kCodebook_[idx];
            gammaSq += residual * residual;
        }

        float gamma = gammaSq > 0.0F ? sqrt(gammaSq) : 0.0F;
        if (!writeCache) {
            float packedGuard = idxPacked[0] == 0xFFU ? 1.0e-9F : 0.0F;
            kNormCacheGm_.SetValue(
                slotHead,
                norm + gamma * 1.0e-6F + packedGuard);
            return;
        }

        uint64_t idxBase = slotHead * kPackedCols_;
        for (uint32_t col = 0; col < kPackedCols_; ++col) {
            kIdxCacheGm_.SetValue(idxBase + col, idxPacked[col]);
        }
        kGammaCacheGm_.SetValue(slotHead, gamma);
        kNormCacheGm_.SetValue(slotHead, norm);
    }

    __aicore__ inline void EncodeKStage1QjlDebug(
        uint32_t token,
        uint32_t kvHead,
        uint64_t slotHead,
        bool packQjl)
    {
        uint8_t idxPacked[128];
        uint8_t qjlPacked[64];
        for (uint32_t col = 0; col < kPackedCols_; ++col) {
            idxPacked[col] = 0;
        }
        qjlPacked[0] = 0;
        if (packQjl) {
            for (uint32_t col = 0; col < kQjlCols_; ++col) {
                qjlPacked[col] = 0;
            }
        }

        LoadCurrentVector(keyGm_, token, kvHead);
        float norm = CalcNorm();
        float invNorm = 1.0F / norm;
        float gammaSq = 0.0F;
        CalcRotRange(kRotationGm_, invNorm, 0U, headDim_);
        for (uint32_t d = 0; d < headDim_; ++d) {
            float xRot = kResidual_[d];
            uint32_t idx = KBoundaryIndex(xRot);
            PackIndex(idxPacked, kPackedCols_, stage1Bits_, d, idx);
            float residual = xRot - kCodebook_[idx];
            kResidual_[d] = residual;
            gammaSq += residual * residual;
        }

        for (uint32_t j = 0; j < headDim_; ++j) {
            currentVec_[j] = 0.0F;
        }
        for (uint32_t d = 0; d < headDim_; ++d) {
            float residual = kResidual_[d];
            uint64_t matrixBase = static_cast<uint64_t>(d) * headDim_;
            for (uint32_t j = 0; j < headDim_; ++j) {
                currentVec_[j] +=
                    residual * kQjlProjTGm_.GetValue(matrixBase + j);
            }
        }
        if (packQjl) {
            for (uint32_t j = 0; j < headDim_; ++j) {
                PackIndex(
                    qjlPacked,
                    kQjlCols_,
                    1U,
                    j,
                    currentVec_[j] >= 0.0F ? 1U : 0U);
            }
        }

        uint64_t idxBase = slotHead * kPackedCols_;
        for (uint32_t col = 0; col < kPackedCols_; ++col) {
            kIdxCacheGm_.SetValue(idxBase + col, idxPacked[col]);
        }
        float gamma = gammaSq > 0.0F ? sqrt(gammaSq) : 0.0F;
        float packedGuard = packQjl && qjlPacked[0] == 0xFFU ? 1.0e-9F : 0.0F;
        kGammaCacheGm_.SetValue(slotHead, gamma);
        kNormCacheGm_.SetValue(
            slotHead,
            norm + currentVec_[0] * 1.0e-6F + packedGuard);
    }

    __aicore__ inline void EncodeV(
        uint32_t token,
        uint32_t kvHead,
        uint64_t slotHead)
    {
        uint8_t packed[128];
        for (uint32_t col = 0; col < vPackedCols_; ++col) {
            packed[col] = 0;
        }

        LoadCurrentVector(valueGm_, token, kvHead);
        float norm = CalcNorm();
        float invNorm = 1.0F / norm;
        CalcRotRange(vRotationGm_, invNorm, 0U, headDim_);
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
            vIdxCacheGm_.SetValue(packedBase + col, packed[col]);
        }
        vNormCacheGm_.SetValue(slotHead, norm);
    }

    __aicore__ inline uint32_t VPartitionCount()
    {
        return vPartitionCount_ > 1U ? vPartitionCount_ : 1U;
    }

    __aicore__ inline bool CanPartitionEncode()
    {
        return VPartitionCount() > 1U
            && (vBits_ == 1U || vBits_ == 2U || vBits_ == 4U);
    }

    __aicore__ inline void EncodeVPartition(
        uint32_t token,
        uint32_t kvHead,
        uint64_t slotHead,
        uint32_t partition)
    {
        uint32_t partitionCount = VPartitionCount();
        uint32_t colStart = (vPackedCols_ * partition) / partitionCount;
        uint32_t colEnd = (vPackedCols_ * (partition + 1U)) / partitionCount;
        uint8_t packed[128];
        for (uint32_t col = colStart; col < colEnd; ++col) {
            packed[col] = 0;
        }

        LoadCurrentVector(valueGm_, token, kvHead);
        float norm = CalcNorm();
        float invNorm = 1.0F / norm;
        uint32_t dimsPerByte = 8U / vBits_;
        uint32_t dimStart = colStart * dimsPerByte;
        uint32_t dimEnd = colEnd * dimsPerByte;
        if (dimEnd > headDim_) {
            dimEnd = headDim_;
        }
        CalcRotRange(vRotationGm_, invNorm, dimStart, dimEnd);
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
            vIdxCacheGm_.SetValue(packedBase + col, packed[col]);
        }
        if (partition == 0U) {
            vNormCacheGm_.SetValue(slotHead, norm);
        }
    }

    __aicore__ inline void ProcessOne(
        uint32_t token,
        uint32_t kvHead,
        uint32_t partition)
    {
        int64_t slot = slotMappingGm_.GetValue(token);
        if (slot < 0) {
            return;
        }
        if (headDim_ > 256U || stage1Bits_ == 0U || stage1Bits_ > 4U
            || vBits_ == 0U || vBits_ > 4U) {
            return;
        }

        uint64_t slotHead = static_cast<uint64_t>(slot) * numKvHeads_ + kvHead;
        if (debugMode_ == 6U) {
            return;
        }
        if (debugMode_ == 1U) {
            if (partition == 0U) {
                EncodeK(token, kvHead, slotHead);
            }
            return;
        }
        if (debugMode_ == 2U) {
            if (CanPartitionEncode()) {
                EncodeVPartition(token, kvHead, slotHead, partition);
            } else if (partition == 0U) {
                EncodeV(token, kvHead, slotHead);
            }
            return;
        }
        if (debugMode_ == 3U) {
            if (partition == 0U) {
                EncodeKStage1Only(token, kvHead, slotHead, true);
            }
            return;
        }
        if (debugMode_ == 4U) {
            if (partition == 0U) {
                EncodeKRotateOnly(token, kvHead, slotHead);
            }
            return;
        }
        if (debugMode_ == 5U) {
            if (partition == 0U) {
                EncodeKStage1Only(token, kvHead, slotHead, false);
            }
            return;
        }
        if (debugMode_ == 7U) {
            if (partition == 0U) {
                EncodeKLoadNormOnly(token, kvHead, slotHead);
            }
            return;
        }
        if (debugMode_ == 8U) {
            if (partition == 0U) {
                EncodeKStage1QjlDebug(token, kvHead, slotHead, false);
            }
            return;
        }
        if (debugMode_ == 9U) {
            if (partition == 0U) {
                EncodeKStage1QjlDebug(token, kvHead, slotHead, true);
            }
            return;
        }
        if (CanPartitionEncode()) {
            if (partition == 0U) {
                EncodeK(token, kvHead, slotHead);
            }
            EncodeVPartition(token, kvHead, slotHead, partition);
        } else if (partition == 0U) {
            EncodeK(token, kvHead, slotHead);
            EncodeV(token, kvHead, slotHead);
        }
    }

    __aicore__ inline void Process()
    {
        uint64_t totalPairs =
            static_cast<uint64_t>(totalTokens_) * static_cast<uint64_t>(numKvHeads_);
        if (totalPairs == 0) {
            return;
        }
        uint64_t partitionCount = CanPartitionEncode() ? VPartitionCount() : 1U;
        uint64_t totalTasks = totalPairs * partitionCount;
        uint64_t coreCount = static_cast<uint64_t>(numCore_ == 0 ? 1 : numCore_);
        uint64_t tasksPerCore = (totalTasks + coreCount - 1) / coreCount;
        uint64_t startTask = static_cast<uint64_t>(GetBlockIdx()) * tasksPerCore;
        uint64_t endTask = startTask + tasksPerCore;
        if (endTask > totalTasks) {
            endTask = totalTasks;
        }
        for (uint64_t task = startTask; task < endTask; ++task) {
            uint64_t pair = task / partitionCount;
            ProcessOne(
                static_cast<uint32_t>(pair / numKvHeads_),
                static_cast<uint32_t>(pair % numKvHeads_),
                static_cast<uint32_t>(task - pair * partitionCount));
        }
    }

private:
    GlobalTensor<KeyT> keyGm_;
    GlobalTensor<ValueT> valueGm_;
    GlobalTensor<int64_t> slotMappingGm_;
    GlobalTensor<uint8_t> kIdxCacheGm_;
    GlobalTensor<uint8_t> kQjlCacheGm_;
    GlobalTensor<float> kGammaCacheGm_;
    GlobalTensor<float> kNormCacheGm_;
    GlobalTensor<uint8_t> vIdxCacheGm_;
    GlobalTensor<float> vNormCacheGm_;
    GlobalTensor<float> kRotationGm_;
    GlobalTensor<float> kBoundaryGm_;
    GlobalTensor<float> kCodebookGm_;
    GlobalTensor<float> kQjlProjTGm_;
    GlobalTensor<float> vRotationGm_;
    GlobalTensor<float> vBoundaryGm_;

    uint32_t totalTokens_{0};
    uint32_t numKvHeads_{0};
    uint32_t blockSize_{0};
    uint32_t kPackedCols_{0};
    uint32_t kQjlCols_{0};
    uint32_t vPackedCols_{0};
    uint32_t totalBits_{0};
    uint32_t stage1Bits_{0};
    uint32_t vBits_{0};
    uint32_t headDim_{0};
    uint32_t debugMode_{0};
    uint32_t vPartitionCount_{1};
    uint32_t numCore_{0};
    float currentVec_[256];
    float kResidual_[256];
    float kCodebook_[16];
    float kBoundary_[16];
    float vBoundary_[16];
};

extern "C" __global__ __aicore__ void tq_encode_kv_to_paged_cache(
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR slotMapping,
    GM_ADDR kIdxCache,
    GM_ADDR kQjlCache,
    GM_ADDR kGammaCache,
    GM_ADDR kNormCache,
    GM_ADDR vIdxCache,
    GM_ADDR vNormCache,
    GM_ADDR kRotation,
    GM_ADDR kBoundary,
    GM_ADDR kCodebook,
    GM_ADDR kQjlProjT,
    GM_ADDR vRotation,
    GM_ADDR vBoundary,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TqEncodeKvToPagedCacheTilingData);
    GET_TILING_DATA_WITH_STRUCT(TqEncodeKvToPagedCacheTilingData, tilingData, tiling);

    KernelTqEncodeKvToPagedCache<DTYPE_KEY, DTYPE_VALUE> op;
    op.Init(
        key,
        value,
        slotMapping,
        kIdxCache,
        kQjlCache,
        kGammaCache,
        kNormCache,
        vIdxCache,
        vNormCache,
        kRotation,
        kBoundary,
        kCodebook,
        kQjlProjT,
        vRotation,
        vBoundary,
        &tilingData);
    op.Process();
}
