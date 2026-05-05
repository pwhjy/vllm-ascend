/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * Ascend C kernel: TurboQuant MSE paged dequant (token-head scalar GM).
 */

#include "kernel_operator.h"

using namespace AscendC;

// Device-side tiling data — mirrors op_host/tq_dequant_mse_paged_tiling.h
// but with public fields (device-side convention).
struct TqDequantMsePagedTilingData {
    uint32_t totalTokens;
    uint32_t blockSize;
    uint32_t numKvHeads;
    uint32_t headDim;
    uint32_t packedCols;
    uint32_t bits;
    uint32_t numCore;
};

class KernelTqDequantMsePaged {
public:
    __aicore__ inline KernelTqDequantMsePaged() {}

    __aicore__ inline void Init(
        GM_ADDR packedIdx,
        GM_ADDR norm,
        GM_ADDR tokenBlockIds,
        GM_ADDR tokenOffsets,
        GM_ADDR codebook,
        GM_ADDR denseRot,
        const TqDequantMsePagedTilingData* tiling
    ) {
        packedIdxGm_.SetGlobalBuffer((__gm__ uint8_t*)packedIdx);
        normGm_.SetGlobalBuffer((__gm__ float*)norm);
        tokenBlockIdsGm_.SetGlobalBuffer((__gm__ int32_t*)tokenBlockIds);
        tokenOffsetsGm_.SetGlobalBuffer((__gm__ int32_t*)tokenOffsets);
        codebookGm_.SetGlobalBuffer((__gm__ float*)codebook);
        denseRotGm_.SetGlobalBuffer((__gm__ float*)denseRot);

        totalTokens_ = tiling->totalTokens;
        blockSize_ = tiling->blockSize;
        numKvHeads_ = tiling->numKvHeads;
        headDim_ = tiling->headDim;
        packedCols_ = tiling->packedCols;
        bits_ = tiling->bits;
        numCore_ = tiling->numCore;
    }

    __aicore__ inline uint32_t ExtractIndex(uint64_t packedBase, uint32_t d)
    {
        uint32_t bitPos = d * bits_;
        uint32_t byteId = bitPos >> 3;
        uint32_t bitOff = bitPos & 7;

        uint16_t v = static_cast<uint16_t>(packedIdxGm_.GetValue(packedBase + byteId));
        if (bitOff + bits_ > 8) {
            uint16_t high = static_cast<uint16_t>(
                packedIdxGm_.GetValue(packedBase + byteId + 1));
            v = static_cast<uint16_t>(v | (high << 8));
        }

        uint32_t mask = (1U << bits_) - 1U;
        return (v >> bitOff) & mask;
    }

    __aicore__ inline uint32_t ExtractIndexBit1(uint64_t packedBase, uint32_t d)
    {
        uint8_t v = packedIdxGm_.GetValue(packedBase + (d >> 3));
        return (static_cast<uint32_t>(v) >> (d & 7U)) & 0x1U;
    }

    __aicore__ inline uint32_t ExtractIndexBit2(uint64_t packedBase, uint32_t d)
    {
        uint8_t v = packedIdxGm_.GetValue(packedBase + (d >> 2));
        return (static_cast<uint32_t>(v) >> ((d & 3U) << 1)) & 0x3U;
    }

    __aicore__ inline uint32_t ExtractIndexBit4(uint64_t packedBase, uint32_t d)
    {
        uint8_t v = packedIdxGm_.GetValue(packedBase + (d >> 1));
        return (static_cast<uint32_t>(v) >> ((d & 1U) << 2)) & 0xFU;
    }

    __aicore__ inline void LoadCodebook()
    {
        cb0_ = codebookGm_.GetValue(0);
        cb1_ = codebookGm_.GetValue(1);
        if (bits_ >= 2) {
            cb2_ = codebookGm_.GetValue(2);
            cb3_ = codebookGm_.GetValue(3);
        }
        if (bits_ >= 3) {
            cb4_ = codebookGm_.GetValue(4);
            cb5_ = codebookGm_.GetValue(5);
            cb6_ = codebookGm_.GetValue(6);
            cb7_ = codebookGm_.GetValue(7);
        }
        if (bits_ >= 4) {
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

    __aicore__ inline void StoreDequantizedValue(
        uint64_t outBase,
        uint32_t d,
        float scale,
        uint32_t idx
    ) {
        float cb = LookupCodebook(idx);
        denseRotGm_.SetValue(outBase + d, cb * scale);
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();

        uint64_t totalPairs =
            static_cast<uint64_t>(totalTokens_) *
            static_cast<uint64_t>(numKvHeads_);
        if (totalPairs == 0) {
            return;
        }

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

            uint64_t packedBase =
                (((static_cast<uint64_t>(blockId) * blockSize_ +
                   static_cast<uint64_t>(offset))
                    * numKvHeads_ + kvHead)
                    * packedCols_);

            uint64_t normIndex =
                ((static_cast<uint64_t>(blockId) * blockSize_ +
                  static_cast<uint64_t>(offset))
                    * numKvHeads_ + kvHead);
            float scale = normGm_.GetValue(normIndex);

            uint64_t outBase =
                ((static_cast<uint64_t>(token) * numKvHeads_ + kvHead)
                    * headDim_);

            if (bits_ == 1U) {
                for (uint32_t d = 0; d < headDim_; ++d) {
                    StoreDequantizedValue(
                        outBase, d, scale,
                        ExtractIndexBit1(packedBase, d));
                }
            } else if (bits_ == 2U) {
                for (uint32_t d = 0; d < headDim_; ++d) {
                    StoreDequantizedValue(
                        outBase, d, scale,
                        ExtractIndexBit2(packedBase, d));
                }
            } else if (bits_ == 4U) {
                for (uint32_t d = 0; d < headDim_; ++d) {
                    StoreDequantizedValue(
                        outBase, d, scale,
                        ExtractIndexBit4(packedBase, d));
                }
            } else {
                for (uint32_t d = 0; d < headDim_; ++d) {
                    StoreDequantizedValue(
                        outBase, d, scale,
                        ExtractIndex(packedBase, d));
                }
            }
        }
    }

private:
    GlobalTensor<uint8_t> packedIdxGm_;
    GlobalTensor<float> normGm_;
    GlobalTensor<int32_t> tokenBlockIdsGm_;
    GlobalTensor<int32_t> tokenOffsetsGm_;
    GlobalTensor<float> codebookGm_;
    GlobalTensor<float> denseRotGm_;

    uint32_t totalTokens_{0};
    uint32_t blockSize_{0};
    uint32_t numKvHeads_{0};
    uint32_t headDim_{0};
    uint32_t packedCols_{0};
    uint32_t bits_{0};
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

extern "C" __global__ __aicore__ void tq_dequant_mse_paged(
    GM_ADDR packedIdx,
    GM_ADDR norm,
    GM_ADDR tokenBlockIds,
    GM_ADDR tokenOffsets,
    GM_ADDR codebook,
    GM_ADDR denseRot,
    GM_ADDR workspace,
    GM_ADDR tiling
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TqDequantMsePagedTilingData);
    GET_TILING_DATA_WITH_STRUCT(TqDequantMsePagedTilingData, tilingData, tiling);

    KernelTqDequantMsePaged op;
    op.Init(
        packedIdx,
        norm,
        tokenBlockIds,
        tokenOffsets,
        codebook,
        denseRot,
        &tilingData
    );
    op.Process();
}
