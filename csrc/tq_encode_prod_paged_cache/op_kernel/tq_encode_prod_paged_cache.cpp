/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License").
 *
 * Ascend C kernel: TurboQuant prod encode + pack + paged cache write.
 */

#include <cmath>
#include "kernel_operator.h"

using namespace AscendC;

struct TqEncodeProdPagedCacheTilingData {
    uint32_t totalTokens;
    uint32_t numKvHeads;
    uint32_t blockSize;
    uint32_t idxPackedCols;
    uint32_t qjlPackedCols;
    uint32_t totalBits;
    uint32_t stage1Bits;
    uint32_t headDim;
    uint32_t numCore;
};

template <typename InT>
class KernelTqEncodeProdPagedCache {
public:
    __aicore__ inline KernelTqEncodeProdPagedCache() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR slotMapping,
        GM_ADDR idxCache,
        GM_ADDR qjlCache,
        GM_ADDR gammaCache,
        GM_ADDR normCache,
        GM_ADDR rotation,
        GM_ADDR boundary,
        GM_ADDR codebook,
        GM_ADDR qjlProjT,
        const TqEncodeProdPagedCacheTilingData* tiling)
    {
        xGm_.SetGlobalBuffer((__gm__ InT*)x);
        slotMappingGm_.SetGlobalBuffer((__gm__ int64_t*)slotMapping);
        idxCacheGm_.SetGlobalBuffer((__gm__ uint8_t*)idxCache);
        qjlCacheGm_.SetGlobalBuffer((__gm__ uint8_t*)qjlCache);
        gammaCacheGm_.SetGlobalBuffer((__gm__ float*)gammaCache);
        normCacheGm_.SetGlobalBuffer((__gm__ float*)normCache);
        rotationGm_.SetGlobalBuffer((__gm__ float*)rotation);
        boundaryGm_.SetGlobalBuffer((__gm__ float*)boundary);
        codebookGm_.SetGlobalBuffer((__gm__ float*)codebook);
        qjlProjTGm_.SetGlobalBuffer((__gm__ float*)qjlProjT);

        totalTokens_ = tiling->totalTokens;
        numKvHeads_ = tiling->numKvHeads;
        blockSize_ = tiling->blockSize;
        idxPackedCols_ = tiling->idxPackedCols;
        qjlPackedCols_ = tiling->qjlPackedCols;
        totalBits_ = tiling->totalBits;
        stage1Bits_ = tiling->stage1Bits;
        headDim_ = tiling->headDim;
        numCore_ = tiling->numCore;
    }

    __aicore__ inline float ReadX(uint32_t token, uint32_t kvHead, uint32_t d)
    {
        uint64_t index =
            (static_cast<uint64_t>(token) * numKvHeads_ + kvHead) * headDim_ + d;
        return static_cast<float>(xGm_.GetValue(index));
    }

    __aicore__ inline float CalcNorm(uint32_t token, uint32_t kvHead)
    {
        float sum = 0.0F;
        for (uint32_t d = 0; d < headDim_; ++d) {
            float v = ReadX(token, kvHead, d);
            sum += v * v;
        }
        if (sum < 1.0e-12F) {
            sum = 1.0e-12F;
        }
        return sqrt(sum);
    }

    __aicore__ inline float CalcRot(
        uint32_t token,
        uint32_t kvHead,
        uint32_t outDim,
        float invNorm)
    {
        float sum = 0.0F;
        for (uint32_t inDim = 0; inDim < headDim_; ++inDim) {
            float x = ReadX(token, kvHead, inDim) * invNorm;
            float r = rotationGm_.GetValue(
                static_cast<uint64_t>(inDim) * headDim_ + outDim);
            sum += x * r;
        }
        return sum;
    }

    __aicore__ inline uint32_t BoundaryIndex(float x)
    {
        uint32_t levels = (1U << stage1Bits_) - 1U;
        uint32_t idx = 0;
        for (uint32_t i = 0; i < levels; ++i) {
            idx += x > boundaryGm_.GetValue(i) ? 1U : 0U;
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

    __aicore__ inline void ProcessOne(uint32_t token, uint32_t kvHead)
    {
        int64_t slot = slotMappingGm_.GetValue(token);
        if (slot < 0) {
            return;
        }

        uint8_t idxPacked[128];
        uint8_t qjlPacked[64];
        float residual[256];

        for (uint32_t col = 0; col < idxPackedCols_; ++col) {
            idxPacked[col] = 0;
        }
        for (uint32_t col = 0; col < qjlPackedCols_; ++col) {
            qjlPacked[col] = 0;
        }

        float norm = CalcNorm(token, kvHead);
        float invNorm = 1.0F / norm;
        float gammaSq = 0.0F;

        for (uint32_t d = 0; d < headDim_; ++d) {
            float xRot = CalcRot(token, kvHead, d, invNorm);
            uint32_t idx = BoundaryIndex(xRot);
            PackIndex(idxPacked, idxPackedCols_, stage1Bits_, d, idx);
            float xHat = codebookGm_.GetValue(idx);
            residual[d] = xRot - xHat;
            gammaSq += residual[d] * residual[d];
        }

        float gamma = gammaSq > 0.0F ? sqrt(gammaSq) : 0.0F;

        for (uint32_t j = 0; j < headDim_; ++j) {
            float sum = 0.0F;
            for (uint32_t d = 0; d < headDim_; ++d) {
                float p = qjlProjTGm_.GetValue(
                    static_cast<uint64_t>(d) * headDim_ + j);
                sum += residual[d] * p;
            }
            PackIndex(qjlPacked, qjlPackedCols_, 1U, j, sum >= 0.0F ? 1U : 0U);
        }

        uint64_t slotHead = static_cast<uint64_t>(slot) * numKvHeads_ + kvHead;
        uint64_t idxBase = slotHead * idxPackedCols_;
        uint64_t qjlBase = slotHead * qjlPackedCols_;

        for (uint32_t col = 0; col < idxPackedCols_; ++col) {
            idxCacheGm_.SetValue(idxBase + col, idxPacked[col]);
        }
        for (uint32_t col = 0; col < qjlPackedCols_; ++col) {
            qjlCacheGm_.SetValue(qjlBase + col, qjlPacked[col]);
        }
        gammaCacheGm_.SetValue(slotHead, gamma);
        normCacheGm_.SetValue(slotHead, norm);
    }

    __aicore__ inline void Process()
    {
        uint64_t totalPairs =
            static_cast<uint64_t>(totalTokens_) * static_cast<uint64_t>(numKvHeads_);
        if (totalPairs == 0) {
            return;
        }
        uint64_t coreCount = static_cast<uint64_t>(numCore_ == 0 ? 1 : numCore_);
        uint64_t pairsPerCore = (totalPairs + coreCount - 1) / coreCount;
        uint64_t startPair = static_cast<uint64_t>(GetBlockIdx()) * pairsPerCore;
        uint64_t endPair = startPair + pairsPerCore;
        if (endPair > totalPairs) {
            endPair = totalPairs;
        }
        for (uint64_t pair = startPair; pair < endPair; ++pair) {
            ProcessOne(
                static_cast<uint32_t>(pair / numKvHeads_),
                static_cast<uint32_t>(pair % numKvHeads_));
        }
    }

private:
    GlobalTensor<InT> xGm_;
    GlobalTensor<int64_t> slotMappingGm_;
    GlobalTensor<uint8_t> idxCacheGm_;
    GlobalTensor<uint8_t> qjlCacheGm_;
    GlobalTensor<float> gammaCacheGm_;
    GlobalTensor<float> normCacheGm_;
    GlobalTensor<float> rotationGm_;
    GlobalTensor<float> boundaryGm_;
    GlobalTensor<float> codebookGm_;
    GlobalTensor<float> qjlProjTGm_;

    uint32_t totalTokens_{0};
    uint32_t numKvHeads_{0};
    uint32_t blockSize_{0};
    uint32_t idxPackedCols_{0};
    uint32_t qjlPackedCols_{0};
    uint32_t totalBits_{0};
    uint32_t stage1Bits_{0};
    uint32_t headDim_{0};
    uint32_t numCore_{0};
};

extern "C" __global__ __aicore__ void tq_encode_prod_paged_cache(
    GM_ADDR x,
    GM_ADDR slotMapping,
    GM_ADDR idxCache,
    GM_ADDR qjlCache,
    GM_ADDR gammaCache,
    GM_ADDR normCache,
    GM_ADDR rotation,
    GM_ADDR boundary,
    GM_ADDR codebook,
    GM_ADDR qjlProjT,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TqEncodeProdPagedCacheTilingData);
    GET_TILING_DATA_WITH_STRUCT(TqEncodeProdPagedCacheTilingData, tilingData, tiling);

    KernelTqEncodeProdPagedCache<DTYPE_X> op;
    op.Init(
        x,
        slotMapping,
        idxCache,
        qjlCache,
        gammaCache,
        normCache,
        rotation,
        boundary,
        codebook,
        qjlProjT,
        &tilingData);
    op.Process();
}
