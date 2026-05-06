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
    uint32_t scoreTileLen;
    uint32_t maxSeqTiles;
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
        scoreTileLen_ = tiling->scoreTileLen;
        maxSeqTiles_ = tiling->maxSeqTiles;
        numCore_ = tiling->numCore;
        scale_ = tiling->scale;
        correction_ = tiling->correction;

        pipe_.InitBuffer(qRotBuf_, 8U * headDim_ * sizeof(float));
        pipe_.InitBuffer(qQjlBuf_, 8U * headDim_ * sizeof(float));
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

    __aicore__ inline void MTE2ToSSync()
    {
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
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

    __aicore__ inline float LookupCodebook2(uint32_t idx)
    {
        switch (idx) {
            case 0U:
                return cb0_;
            case 1U:
                return cb1_;
            case 2U:
                return cb2_;
            default:
                return cb3_;
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

    __aicore__ inline void ProcessKvTokenScoresQ4Local(
        uint32_t b,
        uint32_t kvHead,
        uint32_t pos,
        const LocalTensor<float>& qRotLocal,
        const LocalTensor<float>& qQjlLocal
    ) {
        uint32_t qHeadBase = kvHead * qPerKv_;
        int32_t seqLen = seqLensGm_.GetValue(b);
        if (pos >= maxSeqLen_ || pos >= static_cast<uint32_t>(seqLen)) {
            StoreInvalid(b, qHeadBase, pos);
            StoreInvalid(b, qHeadBase + 1U, pos);
            StoreInvalid(b, qHeadBase + 2U, pos);
            StoreInvalid(b, qHeadBase + 3U, pos);
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

        float mseAcc0 = 0.0F;
        float mseAcc1 = 0.0F;
        float mseAcc2 = 0.0F;
        float mseAcc3 = 0.0F;
        float qjlAcc0 = 0.0F;
        float qjlAcc1 = 0.0F;
        float qjlAcc2 = 0.0F;
        float qjlAcc3 = 0.0F;

        for (uint32_t d = 0; d < headDim_; ++d) {
            uint32_t idx = ExtractIndex(packedBase, d);
            float cb = LookupCodebook(idx);
            float qjlSign = ExtractQjlSign(qjlBase, d);

            float qRot0 = qRotLocal.GetValue(d);
            float qQjl0 = qQjlLocal.GetValue(d);
            float qRot1 = qRotLocal.GetValue(headDim_ + d);
            float qQjl1 = qQjlLocal.GetValue(headDim_ + d);
            float qRot2 = qRotLocal.GetValue((headDim_ << 1) + d);
            float qQjl2 = qQjlLocal.GetValue((headDim_ << 1) + d);
            float qRot3 = qRotLocal.GetValue(3U * headDim_ + d);
            float qQjl3 = qQjlLocal.GetValue(3U * headDim_ + d);

            mseAcc0 += qRot0 * cb;
            mseAcc1 += qRot1 * cb;
            mseAcc2 += qRot2 * cb;
            mseAcc3 += qRot3 * cb;
            qjlAcc0 += qQjl0 * qjlSign;
            qjlAcc1 += qQjl1 * qjlSign;
            qjlAcc2 += qQjl2 * qjlSign;
            qjlAcc3 += qQjl3 * qjlSign;
        }

        StoreScore(b, qHeadBase, pos, mseAcc0, qjlAcc0, gamma, norm);
        StoreScore(b, qHeadBase + 1U, pos, mseAcc1, qjlAcc1, gamma, norm);
        StoreScore(b, qHeadBase + 2U, pos, mseAcc2, qjlAcc2, gamma, norm);
        StoreScore(b, qHeadBase + 3U, pos, mseAcc3, qjlAcc3, gamma, norm);
    }

    __aicore__ inline void ProcessKvTokenScoresQ4LocalBits2Cache(
        uint32_t b,
        uint32_t kvHead,
        uint32_t pos,
        uint64_t cacheIndex,
        const LocalTensor<float>& qRotLocal,
        const LocalTensor<float>& qQjlLocal
    ) {
        uint32_t qHeadBase = kvHead * qPerKv_;
        uint64_t packedBase = cacheIndex * packedCols_;
        uint64_t qjlBase = cacheIndex * qjlCols_;
        float gamma = gammaGm_.GetValue(cacheIndex);
        float norm = normGm_.GetValue(cacheIndex);

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
        uint32_t groups = headDim_ >> 2;
        for (uint32_t g = 0; g < groups; ++g) {
            uint32_t idxBits = static_cast<uint32_t>(
                packedIdxGm_.GetValue(packedBase + g));
            uint32_t qjlBits = static_cast<uint32_t>(
                packedQjlGm_.GetValue(qjlBase + (g >> 1)));
            uint32_t qjlShift = (g & 1U) << 2;

            float cb0 = LookupCodebook2(idxBits & 0x3U);
            float cb1 = LookupCodebook2((idxBits >> 2) & 0x3U);
            float cb2 = LookupCodebook2((idxBits >> 4) & 0x3U);
            float cb3 = LookupCodebook2((idxBits >> 6) & 0x3U);
            float sign0 = (((qjlBits >> qjlShift) & 0x1U) != 0U)
                ? 1.0F : -1.0F;
            float sign1 = (((qjlBits >> (qjlShift + 1U)) & 0x1U) != 0U)
                ? 1.0F : -1.0F;
            float sign2 = (((qjlBits >> (qjlShift + 2U)) & 0x1U) != 0U)
                ? 1.0F : -1.0F;
            float sign3 = (((qjlBits >> (qjlShift + 3U)) & 0x1U) != 0U)
                ? 1.0F : -1.0F;

            uint32_t d0 = g << 2;
            uint32_t d1 = d0 + 1U;
            uint32_t d2 = d0 + 2U;
            uint32_t d3 = d0 + 3U;

            mseAcc0 += qRotLocal.GetValue(d0) * cb0
                + qRotLocal.GetValue(d1) * cb1
                + qRotLocal.GetValue(d2) * cb2
                + qRotLocal.GetValue(d3) * cb3;
            mseAcc1 += qRotLocal.GetValue(q1Base + d0) * cb0
                + qRotLocal.GetValue(q1Base + d1) * cb1
                + qRotLocal.GetValue(q1Base + d2) * cb2
                + qRotLocal.GetValue(q1Base + d3) * cb3;
            mseAcc2 += qRotLocal.GetValue(q2Base + d0) * cb0
                + qRotLocal.GetValue(q2Base + d1) * cb1
                + qRotLocal.GetValue(q2Base + d2) * cb2
                + qRotLocal.GetValue(q2Base + d3) * cb3;
            mseAcc3 += qRotLocal.GetValue(q3Base + d0) * cb0
                + qRotLocal.GetValue(q3Base + d1) * cb1
                + qRotLocal.GetValue(q3Base + d2) * cb2
                + qRotLocal.GetValue(q3Base + d3) * cb3;

            qjlAcc0 += qQjlLocal.GetValue(d0) * sign0
                + qQjlLocal.GetValue(d1) * sign1
                + qQjlLocal.GetValue(d2) * sign2
                + qQjlLocal.GetValue(d3) * sign3;
            qjlAcc1 += qQjlLocal.GetValue(q1Base + d0) * sign0
                + qQjlLocal.GetValue(q1Base + d1) * sign1
                + qQjlLocal.GetValue(q1Base + d2) * sign2
                + qQjlLocal.GetValue(q1Base + d3) * sign3;
            qjlAcc2 += qQjlLocal.GetValue(q2Base + d0) * sign0
                + qQjlLocal.GetValue(q2Base + d1) * sign1
                + qQjlLocal.GetValue(q2Base + d2) * sign2
                + qQjlLocal.GetValue(q2Base + d3) * sign3;
            qjlAcc3 += qQjlLocal.GetValue(q3Base + d0) * sign0
                + qQjlLocal.GetValue(q3Base + d1) * sign1
                + qQjlLocal.GetValue(q3Base + d2) * sign2
                + qQjlLocal.GetValue(q3Base + d3) * sign3;
        }

        StoreScore(b, qHeadBase, pos, mseAcc0, qjlAcc0, gamma, norm);
        StoreScore(b, qHeadBase + 1U, pos, mseAcc1, qjlAcc1, gamma, norm);
        StoreScore(b, qHeadBase + 2U, pos, mseAcc2, qjlAcc2, gamma, norm);
        StoreScore(b, qHeadBase + 3U, pos, mseAcc3, qjlAcc3, gamma, norm);
    }

    __aicore__ inline void ProcessKvTokenScoresQ4LocalBits2Valid(
        uint32_t b,
        uint32_t kvHead,
        uint32_t pos,
        const LocalTensor<float>& qRotLocal,
        const LocalTensor<float>& qQjlLocal
    ) {
        uint32_t blockOffset = pos / blockSize_;
        uint32_t tokenOffset = pos - blockOffset * blockSize_;
        int32_t blockId = blockTableGm_.GetValue(
            static_cast<uint64_t>(b) * maxBlocksPerSeq_ + blockOffset);
        uint64_t cacheIndex =
            ((static_cast<uint64_t>(blockId) * blockSize_ + tokenOffset)
                * numKvHeads_ + kvHead);
        ProcessKvTokenScoresQ4LocalBits2Cache(
            b, kvHead, pos, cacheIndex, qRotLocal, qQjlLocal);
    }

    __aicore__ inline void ProcessKvTokenScoresQ4LocalBits2(
        uint32_t b,
        uint32_t kvHead,
        uint32_t pos,
        const LocalTensor<float>& qRotLocal,
        const LocalTensor<float>& qQjlLocal
    ) {
        uint32_t qHeadBase = kvHead * qPerKv_;
        int32_t seqLen = seqLensGm_.GetValue(b);
        if (pos >= maxSeqLen_ || pos >= static_cast<uint32_t>(seqLen)) {
            StoreInvalid(b, qHeadBase, pos);
            StoreInvalid(b, qHeadBase + 1U, pos);
            StoreInvalid(b, qHeadBase + 2U, pos);
            StoreInvalid(b, qHeadBase + 3U, pos);
            return;
        }

        ProcessKvTokenScoresQ4LocalBits2Valid(
            b, kvHead, pos, qRotLocal, qQjlLocal);
    }

    __aicore__ inline void ProcessKvTokenTileQ4Local(
        uint32_t b,
        uint32_t kvHead,
        uint32_t tile
    ) {
        uint32_t qHeadBase = kvHead * qPerKv_;
        uint32_t posStart = tile * scoreTileLen_;
        uint64_t queryBase =
            (static_cast<uint64_t>(b) * numHeads_ + qHeadBase) * headDim_;
        uint32_t queryElems = 4U * headDim_;

        LocalTensor<float> qRotLocal = qRotBuf_.Get<float>();
        LocalTensor<float> qQjlLocal = qQjlBuf_.Get<float>();
        DataCopy(qRotLocal, qRotGm_[queryBase], queryElems);
        MTE2ToSSync();
        DataCopy(qQjlLocal, qQjlGm_[queryBase], queryElems);
        MTE2ToSSync();

        if (stage1Bits_ == 2U && (headDim_ & 3U) == 0U) {
            int32_t seqLen = seqLensGm_.GetValue(b);
            uint32_t validSeqLen = seqLen > 0
                ? static_cast<uint32_t>(seqLen) : 0U;
            uint32_t tileEnd = posStart + scoreTileLen_;
            if (tileEnd <= maxSeqLen_ && tileEnd <= validSeqLen) {
                uint32_t blockOffset = posStart / blockSize_;
                uint32_t tokenOffset = posStart - blockOffset * blockSize_;
                if (tokenOffset + scoreTileLen_ <= blockSize_) {
                    int32_t blockId = blockTableGm_.GetValue(
                        static_cast<uint64_t>(b) * maxBlocksPerSeq_
                            + blockOffset);
                    uint64_t cacheIndex =
                        ((static_cast<uint64_t>(blockId) * blockSize_
                            + tokenOffset) * numKvHeads_ + kvHead);
                    for (uint32_t i = 0; i < scoreTileLen_; ++i) {
                        ProcessKvTokenScoresQ4LocalBits2Cache(
                            b, kvHead, posStart + i,
                            cacheIndex + static_cast<uint64_t>(i) * numKvHeads_,
                            qRotLocal, qQjlLocal);
                    }
                } else {
                    for (uint32_t i = 0; i < scoreTileLen_; ++i) {
                        ProcessKvTokenScoresQ4LocalBits2Valid(
                            b, kvHead, posStart + i, qRotLocal, qQjlLocal);
                    }
                }
                return;
            }
            for (uint32_t i = 0; i < scoreTileLen_; ++i) {
                ProcessKvTokenScoresQ4LocalBits2(
                    b, kvHead, posStart + i, qRotLocal, qQjlLocal);
            }
            return;
        }

        for (uint32_t i = 0; i < scoreTileLen_; ++i) {
            ProcessKvTokenScoresQ4Local(
                b, kvHead, posStart + i, qRotLocal, qQjlLocal);
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();
        uint64_t totalTasks =
            static_cast<uint64_t>(batch_) *
            static_cast<uint64_t>(numKvHeads_) *
            static_cast<uint64_t>(maxSeqTiles_);
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
            uint32_t tile = static_cast<uint32_t>(linear % maxSeqTiles_);
            uint64_t kvLinear = linear / maxSeqTiles_;
            uint32_t kvHead = static_cast<uint32_t>(kvLinear % numKvHeads_);
            uint32_t b = static_cast<uint32_t>(kvLinear / numKvHeads_);
            if (qPerKv_ == 4U && scoreTileLen_ > 1U) {
                ProcessKvTokenTileQ4Local(b, kvHead, tile);
            } else {
                ProcessKvTokenScores(b, kvHead, tile);
            }
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> qRotBuf_;
    TBuf<TPosition::VECCALC> qQjlBuf_;

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
    uint32_t scoreTileLen_{1};
    uint32_t maxSeqTiles_{0};
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
