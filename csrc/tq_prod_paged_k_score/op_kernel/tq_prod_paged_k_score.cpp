/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * Ascend C kernel: TurboQuant prod paged compressed K-score.
 */

#include "kernel_operator.h"

using namespace AscendC;

struct TqProdPagedKScoreTilingData {
    uint32_t batch;
    uint32_t numHeads;
    uint32_t numKvHeads;
    uint32_t qPerKv;
    uint32_t blockSize;
    uint32_t maxBlocksPerSeq;
    uint32_t maxSeqLen;
    uint32_t headDim;
    uint32_t packedCols;
    uint32_t qjlCols;
    uint32_t stage1Bits;
    uint32_t numCore;
    float scale;
    float correction;
};

class KernelTqProdPagedKScore {
public:
    __aicore__ inline KernelTqProdPagedKScore() {}

    __aicore__ inline void Init(
        GM_ADDR qRot,
        GM_ADDR qQjl,
        GM_ADDR packedIdx,
        GM_ADDR packedQjl,
        GM_ADDR gamma,
        GM_ADDR norm,
        GM_ADDR blockTable,
        GM_ADDR seqLens,
        GM_ADDR codebook,
        GM_ADDR scores,
        const TqProdPagedKScoreTilingData* tiling
    ) {
        qRotGm_.SetGlobalBuffer((__gm__ float*)qRot);
        qQjlGm_.SetGlobalBuffer((__gm__ float*)qQjl);
        packedIdxGm_.SetGlobalBuffer((__gm__ uint8_t*)packedIdx);
        packedQjlGm_.SetGlobalBuffer((__gm__ uint8_t*)packedQjl);
        gammaGm_.SetGlobalBuffer((__gm__ float*)gamma);
        normGm_.SetGlobalBuffer((__gm__ float*)norm);
        blockTableGm_.SetGlobalBuffer((__gm__ int32_t*)blockTable);
        seqLensGm_.SetGlobalBuffer((__gm__ int32_t*)seqLens);
        codebookGm_.SetGlobalBuffer((__gm__ float*)codebook);
        scoresGm_.SetGlobalBuffer((__gm__ float*)scores);

        batch_ = tiling->batch;
        numHeads_ = tiling->numHeads;
        numKvHeads_ = tiling->numKvHeads;
        qPerKv_ = tiling->qPerKv;
        blockSize_ = tiling->blockSize;
        maxBlocksPerSeq_ = tiling->maxBlocksPerSeq;
        maxSeqLen_ = tiling->maxSeqLen;
        headDim_ = tiling->headDim;
        packedCols_ = tiling->packedCols;
        qjlCols_ = tiling->qjlCols;
        stage1Bits_ = tiling->stage1Bits;
        numCore_ = tiling->numCore;
        scale_ = tiling->scale;
        correction_ = tiling->correction;
    }

    __aicore__ inline uint32_t ExtractIndex(uint64_t packedBase, uint32_t d)
    {
        uint32_t bitPos = d * stage1Bits_;
        uint32_t byteId = bitPos >> 3;
        uint32_t bitOff = bitPos & 7;

        uint16_t v = static_cast<uint16_t>(packedIdxGm_.GetValue(packedBase + byteId));
        if (bitOff + stage1Bits_ > 8) {
            uint16_t high = static_cast<uint16_t>(
                packedIdxGm_.GetValue(packedBase + byteId + 1U));
            v = static_cast<uint16_t>(v | (high << 8));
        }

        uint32_t mask = (1U << stage1Bits_) - 1U;
        return (v >> bitOff) & mask;
    }

    __aicore__ inline float ExtractQjlSign(uint64_t qjlBase, uint32_t d)
    {
        uint32_t byteId = d >> 3;
        uint32_t bitOff = d & 7;
        uint8_t byteValue = packedQjlGm_.GetValue(qjlBase + byteId);
        return ((static_cast<uint32_t>(byteValue) >> bitOff) & 0x1U) != 0U
            ? 1.0F : -1.0F;
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

    __aicore__ inline void StoreInvalid(uint32_t b, uint32_t qHead, uint32_t pos)
    {
        uint64_t outOffset =
            ((static_cast<uint64_t>(b) * numHeads_ + qHead) * maxSeqLen_) + pos;
        scoresGm_.SetValue(outOffset, -3.4028234663852886e38F);
    }

    __aicore__ inline void StoreScore(
        uint32_t b,
        uint32_t qHead,
        uint32_t pos,
        float mseAcc,
        float qjlAcc,
        float gamma,
        float norm
    ) {
        float score = (mseAcc + correction_ * gamma * qjlAcc) * norm * scale_;
        uint64_t outOffset =
            ((static_cast<uint64_t>(b) * numHeads_ + qHead) * maxSeqLen_) + pos;
        scoresGm_.SetValue(outOffset, score);
    }

    __aicore__ inline void ProcessScore(uint32_t b, uint32_t qHead, uint32_t pos)
    {
        int32_t seqLen = seqLensGm_.GetValue(b);
        if (pos >= static_cast<uint32_t>(seqLen)) {
            StoreInvalid(b, qHead, pos);
            return;
        }

        uint32_t kvHead = qHead / qPerKv_;
        uint32_t blockOffset = pos / blockSize_;
        uint32_t tokenOffset = pos - blockOffset * blockSize_;
        int32_t blockId = blockTableGm_.GetValue(
            static_cast<uint64_t>(b) * maxBlocksPerSeq_ + blockOffset);

        uint64_t cacheIndex =
            ((static_cast<uint64_t>(blockId) * blockSize_ + tokenOffset)
                * numKvHeads_ + kvHead);
        uint64_t packedBase = cacheIndex * packedCols_;
        uint64_t qjlBase = cacheIndex * qjlCols_;
        float gamma = gammaGm_.GetValue(cacheIndex);
        float norm = normGm_.GetValue(cacheIndex);

        uint64_t queryBase =
            (static_cast<uint64_t>(b) * numHeads_ + qHead) * headDim_;
        float mseAcc = 0.0F;
        float qjlAcc = 0.0F;
        for (uint32_t d = 0; d < headDim_; ++d) {
            uint32_t idx = ExtractIndex(packedBase, d);
            float qRot = qRotGm_.GetValue(queryBase + d);
            float qQjl = qQjlGm_.GetValue(queryBase + d);
            mseAcc += qRot * LookupCodebook(idx);
            qjlAcc += qQjl * ExtractQjlSign(qjlBase, d);
        }

        StoreScore(b, qHead, pos, mseAcc, qjlAcc, gamma, norm);
    }

    __aicore__ inline void ProcessKvTokenScores(uint32_t b, uint32_t kvHead, uint32_t pos)
    {
        uint32_t qHeadBase = kvHead * qPerKv_;
        if (qPerKv_ > 8U) {
            for (uint32_t q = 0; q < qPerKv_; ++q) {
                ProcessScore(b, qHeadBase + q, pos);
            }
            return;
        }

        int32_t seqLen = seqLensGm_.GetValue(b);
        if (pos >= static_cast<uint32_t>(seqLen)) {
            for (uint32_t q = 0; q < qPerKv_; ++q) {
                StoreInvalid(b, qHeadBase + q, pos);
            }
            return;
        }

        uint32_t blockOffset = pos / blockSize_;
        uint32_t tokenOffset = pos - blockOffset * blockSize_;
        int32_t blockId = blockTableGm_.GetValue(
            static_cast<uint64_t>(b) * maxBlocksPerSeq_ + blockOffset);

        uint64_t cacheIndex =
            ((static_cast<uint64_t>(blockId) * blockSize_ + tokenOffset)
                * numKvHeads_ + kvHead);
        uint64_t packedBase = cacheIndex * packedCols_;
        uint64_t qjlBase = cacheIndex * qjlCols_;
        float gamma = gammaGm_.GetValue(cacheIndex);
        float norm = normGm_.GetValue(cacheIndex);

        uint64_t queryBase =
            (static_cast<uint64_t>(b) * numHeads_ + qHeadBase) * headDim_;
        float mseAcc0 = 0.0F;
        float mseAcc1 = 0.0F;
        float mseAcc2 = 0.0F;
        float mseAcc3 = 0.0F;
        float mseAcc4 = 0.0F;
        float mseAcc5 = 0.0F;
        float mseAcc6 = 0.0F;
        float mseAcc7 = 0.0F;
        float qjlAcc0 = 0.0F;
        float qjlAcc1 = 0.0F;
        float qjlAcc2 = 0.0F;
        float qjlAcc3 = 0.0F;
        float qjlAcc4 = 0.0F;
        float qjlAcc5 = 0.0F;
        float qjlAcc6 = 0.0F;
        float qjlAcc7 = 0.0F;

        for (uint32_t d = 0; d < headDim_; ++d) {
            uint32_t idx = ExtractIndex(packedBase, d);
            float cb = LookupCodebook(idx);
            float qjlSign = ExtractQjlSign(qjlBase, d);
            if (qPerKv_ >= 1U) {
                float qRot = qRotGm_.GetValue(queryBase + d);
                float qQjl = qQjlGm_.GetValue(queryBase + d);
                mseAcc0 += qRot * cb;
                qjlAcc0 += qQjl * qjlSign;
            }
            if (qPerKv_ >= 2U) {
                uint64_t qBase = queryBase + headDim_;
                float qRot = qRotGm_.GetValue(qBase + d);
                float qQjl = qQjlGm_.GetValue(qBase + d);
                mseAcc1 += qRot * cb;
                qjlAcc1 += qQjl * qjlSign;
            }
            if (qPerKv_ >= 3U) {
                uint64_t qBase = queryBase + static_cast<uint64_t>(2U) * headDim_;
                float qRot = qRotGm_.GetValue(qBase + d);
                float qQjl = qQjlGm_.GetValue(qBase + d);
                mseAcc2 += qRot * cb;
                qjlAcc2 += qQjl * qjlSign;
            }
            if (qPerKv_ >= 4U) {
                uint64_t qBase = queryBase + static_cast<uint64_t>(3U) * headDim_;
                float qRot = qRotGm_.GetValue(qBase + d);
                float qQjl = qQjlGm_.GetValue(qBase + d);
                mseAcc3 += qRot * cb;
                qjlAcc3 += qQjl * qjlSign;
            }
            if (qPerKv_ >= 5U) {
                uint64_t qBase = queryBase + static_cast<uint64_t>(4U) * headDim_;
                float qRot = qRotGm_.GetValue(qBase + d);
                float qQjl = qQjlGm_.GetValue(qBase + d);
                mseAcc4 += qRot * cb;
                qjlAcc4 += qQjl * qjlSign;
            }
            if (qPerKv_ >= 6U) {
                uint64_t qBase = queryBase + static_cast<uint64_t>(5U) * headDim_;
                float qRot = qRotGm_.GetValue(qBase + d);
                float qQjl = qQjlGm_.GetValue(qBase + d);
                mseAcc5 += qRot * cb;
                qjlAcc5 += qQjl * qjlSign;
            }
            if (qPerKv_ >= 7U) {
                uint64_t qBase = queryBase + static_cast<uint64_t>(6U) * headDim_;
                float qRot = qRotGm_.GetValue(qBase + d);
                float qQjl = qQjlGm_.GetValue(qBase + d);
                mseAcc6 += qRot * cb;
                qjlAcc6 += qQjl * qjlSign;
            }
            if (qPerKv_ >= 8U) {
                uint64_t qBase = queryBase + static_cast<uint64_t>(7U) * headDim_;
                float qRot = qRotGm_.GetValue(qBase + d);
                float qQjl = qQjlGm_.GetValue(qBase + d);
                mseAcc7 += qRot * cb;
                qjlAcc7 += qQjl * qjlSign;
            }
        }

        if (qPerKv_ >= 1U) {
            StoreScore(b, qHeadBase, pos, mseAcc0, qjlAcc0, gamma, norm);
        }
        if (qPerKv_ >= 2U) {
            StoreScore(b, qHeadBase + 1U, pos, mseAcc1, qjlAcc1, gamma, norm);
        }
        if (qPerKv_ >= 3U) {
            StoreScore(b, qHeadBase + 2U, pos, mseAcc2, qjlAcc2, gamma, norm);
        }
        if (qPerKv_ >= 4U) {
            StoreScore(b, qHeadBase + 3U, pos, mseAcc3, qjlAcc3, gamma, norm);
        }
        if (qPerKv_ >= 5U) {
            StoreScore(b, qHeadBase + 4U, pos, mseAcc4, qjlAcc4, gamma, norm);
        }
        if (qPerKv_ >= 6U) {
            StoreScore(b, qHeadBase + 5U, pos, mseAcc5, qjlAcc5, gamma, norm);
        }
        if (qPerKv_ >= 7U) {
            StoreScore(b, qHeadBase + 6U, pos, mseAcc6, qjlAcc6, gamma, norm);
        }
        if (qPerKv_ >= 8U) {
            StoreScore(b, qHeadBase + 7U, pos, mseAcc7, qjlAcc7, gamma, norm);
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();
        uint64_t totalTasks =
            static_cast<uint64_t>(batch_) *
            static_cast<uint64_t>(numKvHeads_) *
            static_cast<uint64_t>(maxSeqLen_);
        if (totalTasks == 0) {
            return;
        }

        uint64_t coreCount = static_cast<uint64_t>(numCore_ == 0 ? 1 : numCore_);
        uint64_t tasksPerCore = (totalTasks + coreCount - 1) / coreCount;
        uint64_t start = static_cast<uint64_t>(coreId) * tasksPerCore;
        uint64_t end = start + tasksPerCore;
        if (end > totalTasks) {
            end = totalTasks;
        }

        LoadCodebook();
        for (uint64_t linear = start; linear < end; ++linear) {
            uint32_t pos = static_cast<uint32_t>(linear % maxSeqLen_);
            uint64_t kvLinear = linear / maxSeqLen_;
            uint32_t kvHead = static_cast<uint32_t>(kvLinear % numKvHeads_);
            uint32_t b = static_cast<uint32_t>(kvLinear / numKvHeads_);
            ProcessKvTokenScores(b, kvHead, pos);
        }
    }

private:
    GlobalTensor<float> qRotGm_;
    GlobalTensor<float> qQjlGm_;
    GlobalTensor<uint8_t> packedIdxGm_;
    GlobalTensor<uint8_t> packedQjlGm_;
    GlobalTensor<float> gammaGm_;
    GlobalTensor<float> normGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> seqLensGm_;
    GlobalTensor<float> codebookGm_;
    GlobalTensor<float> scoresGm_;

    uint32_t batch_{0};
    uint32_t numHeads_{0};
    uint32_t numKvHeads_{0};
    uint32_t qPerKv_{1};
    uint32_t blockSize_{0};
    uint32_t maxBlocksPerSeq_{0};
    uint32_t maxSeqLen_{0};
    uint32_t headDim_{0};
    uint32_t packedCols_{0};
    uint32_t qjlCols_{0};
    uint32_t stage1Bits_{0};
    uint32_t numCore_{0};
    float scale_{1.0F};
    float correction_{0.0F};

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

extern "C" __global__ __aicore__ void tq_prod_paged_k_score(
    GM_ADDR qRot,
    GM_ADDR qQjl,
    GM_ADDR packedIdx,
    GM_ADDR packedQjl,
    GM_ADDR gamma,
    GM_ADDR norm,
    GM_ADDR blockTable,
    GM_ADDR seqLens,
    GM_ADDR codebook,
    GM_ADDR scores,
    GM_ADDR workspace,
    GM_ADDR tiling
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TqProdPagedKScoreTilingData);
    GET_TILING_DATA_WITH_STRUCT(TqProdPagedKScoreTilingData, tilingData, tiling);

    KernelTqProdPagedKScore op;
    op.Init(
        qRot,
        qQjl,
        packedIdx,
        packedQjl,
        gamma,
        norm,
        blockTable,
        seqLens,
        codebook,
        scores,
        &tilingData
    );
    op.Process();
}
