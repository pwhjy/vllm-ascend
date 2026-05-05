/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * Ascend C kernel: TurboQuant prod paged dequant (token-head scalar GM).
 */

#include "kernel_operator.h"

using namespace AscendC;

struct TqDequantProdPagedTilingData {
    uint32_t totalTokens;
    uint32_t blockSize;
    uint32_t numKvHeads;
    uint32_t headDim;
    uint32_t packedIdxCols;
    uint32_t qjlCols;
    uint32_t totalBits;
    uint32_t stage1Bits;
    float qjlCorrection;
    uint32_t numCore;
};

class KernelTqDequantProdPaged {
public:
    __aicore__ inline KernelTqDequantProdPaged() {}

    __aicore__ inline void Init(
        GM_ADDR packedIdx,
        GM_ADDR packedQjl,
        GM_ADDR gamma,
        GM_ADDR norm,
        GM_ADDR tokenBlockIds,
        GM_ADDR tokenOffsets,
        GM_ADDR codebook,
        GM_ADDR qjlProj,
        GM_ADDR denseRot,
        const TqDequantProdPagedTilingData* tiling
    ) {
        packedIdxGm_.SetGlobalBuffer((__gm__ uint8_t*)packedIdx);
        packedQjlGm_.SetGlobalBuffer((__gm__ uint8_t*)packedQjl);
        gammaGm_.SetGlobalBuffer((__gm__ float*)gamma);
        normGm_.SetGlobalBuffer((__gm__ float*)norm);
        tokenBlockIdsGm_.SetGlobalBuffer((__gm__ int32_t*)tokenBlockIds);
        tokenOffsetsGm_.SetGlobalBuffer((__gm__ int32_t*)tokenOffsets);
        codebookGm_.SetGlobalBuffer((__gm__ float*)codebook);
        qjlProjGm_.SetGlobalBuffer((__gm__ float*)qjlProj);
        denseRotGm_.SetGlobalBuffer((__gm__ float*)denseRot);

        totalTokens_ = tiling->totalTokens;
        blockSize_ = tiling->blockSize;
        numKvHeads_ = tiling->numKvHeads;
        headDim_ = tiling->headDim;
        packedIdxCols_ = tiling->packedIdxCols;
        qjlCols_ = tiling->qjlCols;
        totalBits_ = tiling->totalBits;
        stage1Bits_ = tiling->stage1Bits;
        qjlCorrection_ = tiling->qjlCorrection;
        numCore_ = tiling->numCore;
    }

    __aicore__ inline uint32_t ExtractIndex(uint64_t packedBase, uint32_t d)
    {
        uint32_t bitPos = d * stage1Bits_;
        uint32_t byteId = bitPos >> 3;
        uint32_t bitOff = bitPos & 7;

        uint16_t v = static_cast<uint16_t>(packedIdxGm_.GetValue(packedBase + byteId));
        if (bitOff + stage1Bits_ > 8) {
            uint16_t high = static_cast<uint16_t>(
                packedIdxGm_.GetValue(packedBase + byteId + 1));
            v = static_cast<uint16_t>(v | (high << 8));
        }

        uint32_t mask = (1U << stage1Bits_) - 1U;
        return (v >> bitOff) & mask;
    }

    __aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline void LoadCodebook()
    {
        cb0_ = codebookGm_.GetValue(0);
        cb1_ = codebookGm_.GetValue(1);
        if (stage1Bits_ >= 2) {
            cb2_ = codebookGm_.GetValue(2);
            cb3_ = codebookGm_.GetValue(3);
        }
        if (stage1Bits_ >= 3) {
            cb4_ = codebookGm_.GetValue(4);
            cb5_ = codebookGm_.GetValue(5);
            cb6_ = codebookGm_.GetValue(6);
            cb7_ = codebookGm_.GetValue(7);
        }
        if (stage1Bits_ >= 4) {
            cb8_ = codebookGm_.GetValue(8);
            cb9_ = codebookGm_.GetValue(9);
            cb10_ = codebookGm_.GetValue(10);
            cb11_ = codebookGm_.GetValue(11);
            cb12_ = codebookGm_.GetValue(12);
            cb13_ = codebookGm_.GetValue(13);
            cb14_ = codebookGm_.GetValue(14);
            cb15_ = codebookGm_.GetValue(15);
        }
    }

    __aicore__ inline float LookupCodebook(uint32_t idx)
    {
        switch (idx) {
            case 0U:
                return cb0_;
            case 1U:
                return cb1_;
            case 2U:
                return cb2_;
            case 3U:
                return cb3_;
            case 4U:
                return cb4_;
            case 5U:
                return cb5_;
            case 6U:
                return cb6_;
            case 7U:
                return cb7_;
            case 8U:
                return cb8_;
            case 9U:
                return cb9_;
            case 10U:
                return cb10_;
            case 11U:
                return cb11_;
            case 12U:
                return cb12_;
            case 13U:
                return cb13_;
            case 14U:
                return cb14_;
            default:
                return cb15_;
        }
    }

    __aicore__ inline float QjlProjectColumn(uint64_t qjlBase, uint32_t outDim)
    {
        float acc = 0.0F;
        uint32_t srcDim = 0;
        for (uint32_t byteCol = 0; byteCol < qjlCols_; ++byteCol) {
            uint8_t byteValue = packedQjlGm_.GetValue(qjlBase + byteCol);
            uint32_t count = MinU32(8U, headDim_ - srcDim);
            for (uint32_t lane = 0; lane < count; ++lane) {
                float sign = ((byteValue >> lane) & 0x1U) != 0U ? 1.0F : -1.0F;
                uint32_t projIndex = (srcDim + lane) * headDim_ + outDim;
                acc += sign * qjlProjGm_.GetValue(projIndex);
            }
            srcDim += count;
            if (srcDim >= headDim_) {
                break;
            }
        }
        return acc;
    }

    __aicore__ inline void ProcessTokenHead(
        uint32_t token,
        uint32_t kvHead,
        int32_t blockId,
        int32_t offset
    ) {
        uint64_t cacheTokenHead =
            (static_cast<uint64_t>(blockId) * blockSize_ +
             static_cast<uint64_t>(offset)) *
                numKvHeads_ + kvHead;
        uint64_t packedIdxBase = cacheTokenHead * packedIdxCols_;
        uint64_t qjlBase = cacheTokenHead * qjlCols_;
        uint64_t outBase =
            (static_cast<uint64_t>(token) * numKvHeads_ + kvHead) *
            headDim_;

        float gamma = gammaGm_.GetValue(cacheTokenHead);
        float norm = normGm_.GetValue(cacheTokenHead);
        float qjlScale = qjlCorrection_ * gamma;

        for (uint32_t d = 0; d < headDim_; ++d) {
            uint32_t idx = ExtractIndex(packedIdxBase, d);
            float mse = LookupCodebook(idx);
            float qjl = qjlScale * QjlProjectColumn(qjlBase, d);
            denseRotGm_.SetValue(outBase + d, (mse + qjl) * norm);
        }
    }

    __aicore__ inline void Process()
    {
        if (totalTokens_ == 0 || numKvHeads_ == 0 || headDim_ == 0) {
            return;
        }

        uint32_t coreId = GetBlockIdx();
        uint64_t totalPairs =
            static_cast<uint64_t>(totalTokens_) *
            static_cast<uint64_t>(numKvHeads_);
        uint64_t coreCount = static_cast<uint64_t>(numCore_ == 0 ? 1 : numCore_);
        uint64_t pairsPerCore = (totalPairs + coreCount - 1) / coreCount;
        uint64_t startPair = static_cast<uint64_t>(coreId) * pairsPerCore;
        uint64_t endPair = startPair + pairsPerCore;
        if (endPair > totalPairs) {
            endPair = totalPairs;
        }

        LoadCodebook();

        for (uint64_t pair = startPair; pair < endPair; ++pair) {
            uint32_t kvHead = pair % numKvHeads_;
            uint32_t token = pair / numKvHeads_;
            int32_t blockId = tokenBlockIdsGm_.GetValue(token);
            int32_t offset = tokenOffsetsGm_.GetValue(token);
            ProcessTokenHead(token, kvHead, blockId, offset);
        }
    }

private:
    GlobalTensor<uint8_t> packedIdxGm_;
    GlobalTensor<uint8_t> packedQjlGm_;
    GlobalTensor<float> gammaGm_;
    GlobalTensor<float> normGm_;
    GlobalTensor<int32_t> tokenBlockIdsGm_;
    GlobalTensor<int32_t> tokenOffsetsGm_;
    GlobalTensor<float> codebookGm_;
    GlobalTensor<float> qjlProjGm_;
    GlobalTensor<float> denseRotGm_;

    uint32_t totalTokens_{0};
    uint32_t blockSize_{0};
    uint32_t numKvHeads_{0};
    uint32_t headDim_{0};
    uint32_t packedIdxCols_{0};
    uint32_t qjlCols_{0};
    uint32_t totalBits_{0};
    uint32_t stage1Bits_{0};
    float qjlCorrection_{0.0F};
    uint32_t numCore_{0};

    float cb0_{0.0F};
    float cb1_{0.0F};
    float cb2_{0.0F};
    float cb3_{0.0F};
    float cb4_{0.0F};
    float cb5_{0.0F};
    float cb6_{0.0F};
    float cb7_{0.0F};
    float cb8_{0.0F};
    float cb9_{0.0F};
    float cb10_{0.0F};
    float cb11_{0.0F};
    float cb12_{0.0F};
    float cb13_{0.0F};
    float cb14_{0.0F};
    float cb15_{0.0F};
};

extern "C" __global__ __aicore__ void tq_dequant_prod_paged(
    GM_ADDR packedIdx,
    GM_ADDR packedQjl,
    GM_ADDR gamma,
    GM_ADDR norm,
    GM_ADDR tokenBlockIds,
    GM_ADDR tokenOffsets,
    GM_ADDR codebook,
    GM_ADDR qjlProj,
    GM_ADDR denseRot,
    GM_ADDR workspace,
    GM_ADDR tiling
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TqDequantProdPagedTilingData);
    GET_TILING_DATA_WITH_STRUCT(TqDequantProdPagedTilingData, tilingData, tiling);

    KernelTqDequantProdPaged op;
    op.Init(
        packedIdx,
        packedQjl,
        gamma,
        norm,
        tokenBlockIds,
        tokenOffsets,
        codebook,
        qjlProj,
        denseRot,
        &tilingData
    );
    op.Process();
}
