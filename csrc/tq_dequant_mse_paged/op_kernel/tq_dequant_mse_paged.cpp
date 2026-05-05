/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * Ascend C kernel: TurboQuant MSE paged dequant (scalar GM, correctness-first).
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

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();

        uint64_t totalElems =
            static_cast<uint64_t>(totalTokens_) *
            static_cast<uint64_t>(numKvHeads_) *
            static_cast<uint64_t>(headDim_);

        uint64_t elemsPerCore = (totalElems + numCore_ - 1) / numCore_;
        uint64_t start = static_cast<uint64_t>(coreId) * elemsPerCore;
        uint64_t end = start + elemsPerCore;
        if (end > totalElems) {
            end = totalElems;
        }

        for (uint64_t linear = start; linear < end; ++linear) {
            uint32_t d = linear % headDim_;
            uint64_t tmp = linear / headDim_;
            uint32_t kvHead = tmp % numKvHeads_;
            uint32_t token = tmp / numKvHeads_;

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

            uint32_t idx = ExtractIndex(packedBase, d);

            float cb = codebookGm_.GetValue(idx);
            float scale = normGm_.GetValue(normIndex);
            float y = cb * scale;

            uint64_t outIndex =
                ((static_cast<uint64_t>(token) * numKvHeads_ + kvHead)
                    * headDim_ + d);

            denseRotGm_.SetValue(outIndex, y);
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
