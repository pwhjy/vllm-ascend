/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 * Ascend C kernel: TurboQuant MSE encode + pack + paged cache write.
 */

#include <cmath>
#include "kernel_operator.h"

using namespace AscendC;

struct TqEncodeMsePagedCacheTilingData {
    uint32_t totalTokens;
    uint32_t numKvHeads;
    uint32_t blockSize;
    uint32_t packedCols;
    uint32_t bits;
    uint32_t headDim;
    uint32_t numCore;
};

template <typename InT>
class KernelTqEncodeMsePagedCache {
public:
    __aicore__ inline KernelTqEncodeMsePagedCache() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR slotMapping,
        GM_ADDR idxCache,
        GM_ADDR normCache,
        GM_ADDR rotation,
        GM_ADDR boundary,
        const TqEncodeMsePagedCacheTilingData* tiling)
    {
        xGm_.SetGlobalBuffer((__gm__ InT*)x);
        slotMappingGm_.SetGlobalBuffer((__gm__ int64_t*)slotMapping);
        idxCacheGm_.SetGlobalBuffer((__gm__ uint8_t*)idxCache);
        normCacheGm_.SetGlobalBuffer((__gm__ float*)normCache);
        rotationGm_.SetGlobalBuffer((__gm__ float*)rotation);
        boundaryGm_.SetGlobalBuffer((__gm__ float*)boundary);

        totalTokens_ = tiling->totalTokens;
        numKvHeads_ = tiling->numKvHeads;
        blockSize_ = tiling->blockSize;
        packedCols_ = tiling->packedCols;
        bits_ = tiling->bits;
        headDim_ = tiling->headDim;
        numCore_ = tiling->numCore;
        pipe_.InitBuffer(sqrtBuf_, 8U * sizeof(float));
        LoadSmallParams();
    }

    __aicore__ inline float ReadX(uint32_t token, uint32_t kvHead, uint32_t d)
    {
        uint64_t index =
            (static_cast<uint64_t>(token) * numKvHeads_ + kvHead) * headDim_ + d;
        return static_cast<float>(xGm_.GetValue(index));
    }

    __aicore__ inline void LoadCurrentVector(uint32_t token, uint32_t kvHead)
    {
        for (uint32_t d = 0; d < headDim_; ++d) {
            currentVec_[d] = ReadX(token, kvHead, d);
        }
    }

    __aicore__ inline void LoadSmallParams()
    {
        uint32_t levels = bits_ < 4U ? (1U << bits_) : 16U;
        for (uint32_t i = 0; i + 1U < levels; ++i) {
            boundary_[i] = boundaryGm_.GetValue(i);
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

    __aicore__ inline float SqrtScalar(float value)
    {
        LocalTensor<float> sqrtLocal = sqrtBuf_.Get<float>();
        sqrtLocal.SetValue(0, value);
        SToVSync();
        Sqrt(sqrtLocal, sqrtLocal, 1U);
        VToSSync();
        return sqrtLocal.GetValue(0);
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
        return SqrtScalar(sum);
    }

    __aicore__ inline void CalcRot(float invNorm)
    {
        for (uint32_t d = 0; d < headDim_; ++d) {
            rotVec_[d] = 0.0F;
        }
        for (uint32_t inDim = 0; inDim < headDim_; ++inDim) {
            float x = currentVec_[inDim] * invNorm;
            uint64_t matrixBase = static_cast<uint64_t>(inDim) * headDim_;
            for (uint32_t outDim = 0; outDim < headDim_; ++outDim) {
                rotVec_[outDim] += x * rotationGm_.GetValue(matrixBase + outDim);
            }
        }
    }

    __aicore__ inline uint32_t BoundaryIndex(float x)
    {
        uint32_t levels = (1U << bits_) - 1U;
        uint32_t idx = 0;
        for (uint32_t i = 0; i < levels; ++i) {
            idx += x > boundary_[i] ? 1U : 0U;
        }
        return idx;
    }

    __aicore__ inline void PackIndex(uint8_t* packed, uint32_t d, uint32_t idx)
    {
        uint32_t bitPos = d * bits_;
        uint32_t byteId = bitPos >> 3;
        uint32_t bitOff = bitPos & 7U;
        uint32_t value = idx << bitOff;
        packed[byteId] = static_cast<uint8_t>(
            static_cast<uint32_t>(packed[byteId]) | (value & 0xFFU));
        if (bitOff + bits_ > 8U && byteId + 1U < packedCols_) {
            packed[byteId + 1U] = static_cast<uint8_t>(
                static_cast<uint32_t>(packed[byteId + 1U]) | (value >> 8U));
        }
    }

    __aicore__ inline void ProcessOne(uint32_t token, uint32_t kvHead)
    {
        int64_t slot = slotMappingGm_.GetValue(token);
        if (slot < 0) {
            return;
        }

        if (headDim_ > 256U || bits_ == 0U || bits_ > 4U) {
            return;
        }

        uint8_t packed[128];
        for (uint32_t col = 0; col < packedCols_; ++col) {
            packed[col] = 0;
        }

        LoadCurrentVector(token, kvHead);
        float norm = CalcNorm();
        float invNorm = 1.0F / norm;

        CalcRot(invNorm);
        for (uint32_t d = 0; d < headDim_; ++d) {
            PackIndex(packed, d, BoundaryIndex(rotVec_[d]));
        }

        uint64_t slotHead = static_cast<uint64_t>(slot) * numKvHeads_ + kvHead;
        uint64_t packedBase = slotHead * packedCols_;
        for (uint32_t col = 0; col < packedCols_; ++col) {
            idxCacheGm_.SetValue(packedBase + col, packed[col]);
        }
        normCacheGm_.SetValue(slotHead, norm);
    }

    __aicore__ inline void Process()
    {
        uint64_t totalTasks = static_cast<uint64_t>(totalTokens_);
        if (totalTasks == 0) {
            return;
        }
        uint64_t coreCount = static_cast<uint64_t>(numCore_ == 0 ? 1 : numCore_);
        uint64_t tasksPerCore = (totalTasks + coreCount - 1) / coreCount;
        uint64_t startToken = static_cast<uint64_t>(GetBlockIdx()) * tasksPerCore;
        uint64_t endToken = startToken + tasksPerCore;
        if (endToken > totalTasks) {
            endToken = totalTasks;
        }
        for (uint64_t token = startToken; token < endToken; ++token) {
            for (uint32_t kvHead = 0; kvHead < numKvHeads_; ++kvHead) {
                ProcessOne(static_cast<uint32_t>(token), kvHead);
            }
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> sqrtBuf_;

    GlobalTensor<InT> xGm_;
    GlobalTensor<int64_t> slotMappingGm_;
    GlobalTensor<uint8_t> idxCacheGm_;
    GlobalTensor<float> normCacheGm_;
    GlobalTensor<float> rotationGm_;
    GlobalTensor<float> boundaryGm_;

    uint32_t totalTokens_{0};
    uint32_t numKvHeads_{0};
    uint32_t blockSize_{0};
    uint32_t packedCols_{0};
    uint32_t bits_{0};
    uint32_t headDim_{0};
    uint32_t numCore_{0};
    float currentVec_[256];
    float rotVec_[256];
    float boundary_[16];
};

extern "C" __global__ __aicore__ void tq_encode_mse_paged_cache(
    GM_ADDR x,
    GM_ADDR slotMapping,
    GM_ADDR rotation,
    GM_ADDR boundary,
    GM_ADDR idxCache,
    GM_ADDR normCache,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TqEncodeMsePagedCacheTilingData);
    GET_TILING_DATA_WITH_STRUCT(TqEncodeMsePagedCacheTilingData, tilingData, tiling);

    KernelTqEncodeMsePagedCache<DTYPE_X> op;
    op.Init(x, slotMapping, idxCache, normCache, rotation, boundary, &tilingData);
    op.Process();
}
