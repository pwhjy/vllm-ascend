/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * Ascend C kernel: TurboQuant MSE paged dequant with two paged scales and UB row tiling.
 */

#include "kernel_operator.h"

using namespace AscendC;

struct TqDequantMsePagedScaledTilingData {
    uint32_t totalTokens;
    uint32_t blockSize;
    uint32_t numKvHeads;
    uint32_t headDim;
    uint32_t packedCols;
    uint32_t bits;
    uint32_t numCore;
    uint32_t signedBits1;
    float scaleMultiplier;
};

template <typename OutT>
class KernelTqDequantMsePagedScaled {
public:
    __aicore__ inline KernelTqDequantMsePagedScaled() {}

    __aicore__ inline void Init(
        GM_ADDR packedIdx,
        GM_ADDR norm,
        GM_ADDR extraScale,
        GM_ADDR tokenBlockIds,
        GM_ADDR tokenOffsets,
        GM_ADDR codebook,
        GM_ADDR out,
        const TqDequantMsePagedScaledTilingData* tiling
    ) {
        packedIdxGm_.SetGlobalBuffer((__gm__ uint8_t*)packedIdx);
        normGm_.SetGlobalBuffer((__gm__ float*)norm);
        extraScaleGm_.SetGlobalBuffer((__gm__ float*)extraScale);
        tokenBlockIdsGm_.SetGlobalBuffer((__gm__ int32_t*)tokenBlockIds);
        tokenOffsetsGm_.SetGlobalBuffer((__gm__ int32_t*)tokenOffsets);
        codebookGm_.SetGlobalBuffer((__gm__ float*)codebook);
        outGm_.SetGlobalBuffer((__gm__ OutT*)out);

        totalTokens_ = tiling->totalTokens;
        blockSize_ = tiling->blockSize;
        numKvHeads_ = tiling->numKvHeads;
        headDim_ = tiling->headDim;
        packedCols_ = tiling->packedCols;
        bits_ = tiling->bits;
        numCore_ = tiling->numCore;
        signedBits1_ = tiling->signedBits1;
        scaleMultiplier_ = tiling->scaleMultiplier;

        uint32_t outBlockElems = 32U / sizeof(OutT);
        alignedHeadDim_ = AlignUpU32(headDim_, outBlockElems);
        alignedPackedCols_ = AlignUpU32(packedCols_, 32U);
        outputRowAligned_ = (alignedHeadDim_ == headDim_);
        packedRowAligned_ = (alignedPackedCols_ == packedCols_);
        outputRowsPerTile_ = headDim_ <= 256U ? 16U : 4U;

        pipe_.InitBuffer(outBuf_, outputRowsPerTile_ * alignedHeadDim_ * sizeof(OutT));
        pipe_.InitBuffer(packedBuf_, alignedPackedCols_);
    }

    __aicore__ inline uint32_t ExtractIndex(
        const LocalTensor<uint8_t>& packedLocal,
        uint32_t d
    )
    {
        uint32_t bitPos = d * bits_;
        uint32_t byteId = bitPos >> 3;
        uint32_t bitOff = bitPos & 7;

        uint16_t v = static_cast<uint16_t>(packedLocal.GetValue(byteId));
        if (bitOff + bits_ > 8) {
            uint16_t high = static_cast<uint16_t>(
                packedLocal.GetValue(byteId + 1));
            v = static_cast<uint16_t>(v | (high << 8));
        }

        uint32_t mask = (1U << bits_) - 1U;
        return (v >> bitOff) & mask;
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
        const LocalTensor<OutT>& outLocal,
        uint32_t d,
        float scale,
        uint32_t idx
    ) {
        float cb = LookupCodebook(idx);
        outLocal.SetValue(d, static_cast<OutT>(cb * scale));
    }

    __aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline uint32_t AlignUpU32(uint32_t value, uint32_t align)
    {
        return align == 0U ? value : ((value + align - 1U) / align) * align;
    }

    __aicore__ inline void SToMTE3Sync()
    {
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    }

    __aicore__ inline void MTE3ToSSync()
    {
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }

    __aicore__ inline void MTE2ToSSync()
    {
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }

    __aicore__ inline void LoadPackedRow(
        uint64_t packedBase,
        const LocalTensor<uint8_t>& packedLocal
    ) {
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
        DataCopyParams copyParams;
        copyParams.blockLen = packedCols_;
        copyParams.blockCount = 1;
        DataCopyPadParams padParams;
        DataCopyPad(packedLocal, packedIdxGm_[packedBase], copyParams, padParams);
        MTE2ToSSync();
#else
        if (packedRowAligned_) {
            DataCopy(packedLocal, packedIdxGm_[packedBase], packedCols_);
            MTE2ToSSync();
        } else {
            for (uint32_t byteCol = 0; byteCol < packedCols_; ++byteCol) {
                packedLocal.SetValue(byteCol, packedIdxGm_.GetValue(packedBase + byteCol));
            }
        }
#endif
    }

    __aicore__ inline void StoreOutputRows(
        uint64_t outBase,
        const LocalTensor<OutT>& outLocal,
        uint32_t rowCount
    ) {
        if (rowCount == 0U) {
            return;
        }
        if (outputRowAligned_) {
            SToMTE3Sync();
            DataCopy(outGm_[outBase], outLocal, rowCount * headDim_);
            MTE3ToSSync();
        } else {
            for (uint32_t row = 0; row < rowCount; ++row) {
                uint64_t rowBase = outBase + static_cast<uint64_t>(row) * headDim_;
                LocalTensor<OutT> rowLocal = outLocal[row * alignedHeadDim_];
                for (uint32_t d = 0; d < headDim_; ++d) {
                    outGm_.SetValue(rowBase + d, rowLocal.GetValue(d));
                }
            }
        }
    }

    __aicore__ inline void ProcessBits1(
        const LocalTensor<uint8_t>& packedLocal,
        const LocalTensor<OutT>& outLocal,
        float scale
    )
    {
        for (uint32_t byteCol = 0; byteCol < packedCols_; ++byteCol) {
            uint32_t dBase = byteCol << 3;
            if (dBase >= headDim_) {
                return;
            }
            uint32_t count = MinU32(8U, headDim_ - dBase);
            uint8_t byteValue = packedLocal.GetValue(byteCol);
            uint32_t value = static_cast<uint32_t>(byteValue);
            for (uint32_t lane = 0; lane < count; ++lane) {
                StoreDequantizedValue(
                    outLocal, dBase + lane, scale,
                    (value >> lane) & 0x1U);
            }
        }
    }

    __aicore__ inline void ProcessSignedBits1(
        const LocalTensor<uint8_t>& packedLocal,
        const LocalTensor<OutT>& outLocal,
        float scale
    )
    {
        for (uint32_t byteCol = 0; byteCol < packedCols_; ++byteCol) {
            uint32_t dBase = byteCol << 3;
            if (dBase >= headDim_) {
                return;
            }
            uint32_t count = MinU32(8U, headDim_ - dBase);
            uint8_t byteValue = packedLocal.GetValue(byteCol);
            uint32_t value = static_cast<uint32_t>(byteValue);
            for (uint32_t lane = 0; lane < count; ++lane) {
                float signedScale = ((value >> lane) & 0x1U) != 0U ? scale : -scale;
                outLocal.SetValue(dBase + lane, static_cast<OutT>(signedScale));
            }
        }
    }

    __aicore__ inline void ProcessBits2(
        const LocalTensor<uint8_t>& packedLocal,
        const LocalTensor<OutT>& outLocal,
        float scale
    )
    {
        for (uint32_t byteCol = 0; byteCol < packedCols_; ++byteCol) {
            uint32_t dBase = byteCol << 2;
            if (dBase >= headDim_) {
                return;
            }
            uint32_t count = MinU32(4U, headDim_ - dBase);
            uint8_t byteValue = packedLocal.GetValue(byteCol);
            uint32_t value = static_cast<uint32_t>(byteValue);
            for (uint32_t lane = 0; lane < count; ++lane) {
                StoreDequantizedValue(
                    outLocal, dBase + lane, scale,
                    (value >> (lane << 1)) & 0x3U);
            }
        }
    }

    __aicore__ inline void ProcessBits3(
        const LocalTensor<uint8_t>& packedLocal,
        const LocalTensor<OutT>& outLocal,
        float scale
    )
    {
        uint32_t d = 0;
        uint32_t byteCol = 0;
        for (; d + 7U < headDim_ && byteCol + 2U < packedCols_; d += 8U, byteCol += 3U) {
            uint32_t value = static_cast<uint32_t>(packedLocal.GetValue(byteCol));
            value |= static_cast<uint32_t>(packedLocal.GetValue(byteCol + 1U)) << 8;
            value |= static_cast<uint32_t>(packedLocal.GetValue(byteCol + 2U)) << 16;
            for (uint32_t lane = 0; lane < 8U; ++lane) {
                StoreDequantizedValue(
                    outLocal, d + lane, scale,
                    (value >> (lane * 3U)) & 0x7U);
            }
        }
        for (; d < headDim_; ++d) {
            StoreDequantizedValue(
                outLocal, d, scale,
                ExtractIndex(packedLocal, d));
        }
    }

    __aicore__ inline void ProcessBits4(
        const LocalTensor<uint8_t>& packedLocal,
        const LocalTensor<OutT>& outLocal,
        float scale
    )
    {
        for (uint32_t byteCol = 0; byteCol < packedCols_; ++byteCol) {
            uint32_t dBase = byteCol << 1;
            if (dBase >= headDim_) {
                return;
            }
            uint32_t count = MinU32(2U, headDim_ - dBase);
            uint8_t byteValue = packedLocal.GetValue(byteCol);
            uint32_t value = static_cast<uint32_t>(byteValue);
            for (uint32_t lane = 0; lane < count; ++lane) {
                StoreDequantizedValue(
                    outLocal, dBase + lane, scale,
                    (value >> (lane << 2)) & 0xFU);
            }
        }
    }

    __aicore__ inline void ProcessGeneric(
        const LocalTensor<uint8_t>& packedLocal,
        const LocalTensor<OutT>& outLocal,
        float scale
    )
    {
        for (uint32_t d = 0; d < headDim_; ++d) {
            StoreDequantizedValue(
                outLocal, d, scale,
                ExtractIndex(packedLocal, d));
        }
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

        LocalTensor<uint8_t> packedLocal = packedBuf_.Get<uint8_t>();
        LocalTensor<OutT> outTileLocal = outBuf_.Get<OutT>();
        uint64_t tileStartPair = startPair;
        uint32_t rowInTile = 0;

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

            uint64_t scaleIndex =
                ((static_cast<uint64_t>(blockId) * blockSize_ +
                  static_cast<uint64_t>(offset))
                    * numKvHeads_ + kvHead);
            float scale = (
                normGm_.GetValue(scaleIndex) *
                extraScaleGm_.GetValue(scaleIndex) *
                scaleMultiplier_);

            LocalTensor<OutT> outLocal = outTileLocal[rowInTile * alignedHeadDim_];
            LoadPackedRow(packedBase, packedLocal);

            if (bits_ == 1U) {
                if (signedBits1_ != 0U) {
                    ProcessSignedBits1(packedLocal, outLocal, scale);
                } else {
                    ProcessBits1(packedLocal, outLocal, scale);
                }
            } else if (bits_ == 2U) {
                ProcessBits2(packedLocal, outLocal, scale);
            } else if (bits_ == 3U) {
                ProcessBits3(packedLocal, outLocal, scale);
            } else if (bits_ == 4U) {
                ProcessBits4(packedLocal, outLocal, scale);
            } else {
                ProcessGeneric(packedLocal, outLocal, scale);
            }

            ++rowInTile;
            if (rowInTile == outputRowsPerTile_ || pair + 1U == endPair) {
                StoreOutputRows(tileStartPair * headDim_, outTileLocal, rowInTile);
                tileStartPair = pair + 1U;
                rowInTile = 0U;
            }
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> outBuf_;
    TBuf<TPosition::VECCALC> packedBuf_;

    GlobalTensor<uint8_t> packedIdxGm_;
    GlobalTensor<float> normGm_;
    GlobalTensor<float> extraScaleGm_;
    GlobalTensor<int32_t> tokenBlockIdsGm_;
    GlobalTensor<int32_t> tokenOffsetsGm_;
    GlobalTensor<float> codebookGm_;
    GlobalTensor<OutT> outGm_;

    uint32_t totalTokens_{0};
    uint32_t blockSize_{0};
    uint32_t numKvHeads_{0};
    uint32_t headDim_{0};
    uint32_t packedCols_{0};
    uint32_t bits_{0};
    uint32_t numCore_{0};
    uint32_t signedBits1_{0};
    float scaleMultiplier_{1.0F};
    uint32_t alignedHeadDim_{0};
    uint32_t alignedPackedCols_{0};
    uint32_t outputRowsPerTile_{1};
    bool outputRowAligned_{false};
    bool packedRowAligned_{false};

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

template <typename OutT>
__aicore__ inline void RunTqDequantMsePagedScaled(
    GM_ADDR packedIdx,
    GM_ADDR norm,
    GM_ADDR extraScale,
    GM_ADDR tokenBlockIds,
    GM_ADDR tokenOffsets,
    GM_ADDR codebook,
    GM_ADDR out,
    const TqDequantMsePagedScaledTilingData* tilingData
) {
    KernelTqDequantMsePagedScaled<OutT> op;
    op.Init(
        packedIdx,
        norm,
        extraScale,
        tokenBlockIds,
        tokenOffsets,
        codebook,
        out,
        tilingData
    );
    op.Process();
}

extern "C" __global__ __aicore__ void tq_dequant_mse_paged_scaled(
    GM_ADDR packedIdx,
    GM_ADDR norm,
    GM_ADDR extraScale,
    GM_ADDR tokenBlockIds,
    GM_ADDR tokenOffsets,
    GM_ADDR codebook,
    GM_ADDR out,
    GM_ADDR workspace,
    GM_ADDR tiling
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TqDequantMsePagedScaledTilingData);
    GET_TILING_DATA_WITH_STRUCT(TqDequantMsePagedScaledTilingData, tilingData, tiling);

    RunTqDequantMsePagedScaled<DTYPE_OUT>(
        packedIdx,
        norm,
        extraScale,
        tokenBlockIds,
        tokenOffsets,
        codebook,
        out,
        &tilingData
    );
}
