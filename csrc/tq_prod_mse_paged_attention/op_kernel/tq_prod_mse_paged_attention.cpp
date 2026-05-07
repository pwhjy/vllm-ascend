/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License");
 *
 * Ascend C kernel: TurboQuant prod-K + MSE-V paged decode attention.
 */

#include "kernel_operator.h"

using namespace AscendC;

struct TqProdMsePagedAttentionTilingData {
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
    uint32_t scoreTileLen;
    uint32_t numCore;
    float scale;
    float correction;
};

class KernelTqProdMsePagedAttention {
public:
    __aicore__ inline KernelTqProdMsePagedAttention() {}

    __aicore__ inline void Init(
        GM_ADDR qRot,
        GM_ADDR qQjl,
        GM_ADDR kPackedIdx,
        GM_ADDR kPackedQjl,
        GM_ADDR kGamma,
        GM_ADDR kNorm,
        GM_ADDR vPackedIdx,
        GM_ADDR vNorm,
        GM_ADDR blockTable,
        GM_ADDR seqLens,
        GM_ADDR kCodebook,
        GM_ADDR vCodebook,
        GM_ADDR outRot,
        const TqProdMsePagedAttentionTilingData* tiling
    ) {
        qRotGm_.SetGlobalBuffer((__gm__ float*)qRot);
        qQjlGm_.SetGlobalBuffer((__gm__ float*)qQjl);
        kPackedIdxGm_.SetGlobalBuffer((__gm__ uint8_t*)kPackedIdx);
        kPackedQjlGm_.SetGlobalBuffer((__gm__ uint8_t*)kPackedQjl);
        kGammaGm_.SetGlobalBuffer((__gm__ float*)kGamma);
        kNormGm_.SetGlobalBuffer((__gm__ float*)kNorm);
        vPackedIdxGm_.SetGlobalBuffer((__gm__ uint8_t*)vPackedIdx);
        vNormGm_.SetGlobalBuffer((__gm__ float*)vNorm);
        blockTableGm_.SetGlobalBuffer((__gm__ int32_t*)blockTable);
        seqLensGm_.SetGlobalBuffer((__gm__ int32_t*)seqLens);
        kCodebookGm_.SetGlobalBuffer((__gm__ float*)kCodebook);
        vCodebookGm_.SetGlobalBuffer((__gm__ float*)vCodebook);
        outRotGm_.SetGlobalBuffer((__gm__ float*)outRot);

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
        scoreTileLen_ = tiling->scoreTileLen;
        numCore_ = tiling->numCore;
        scale_ = tiling->scale;
        correction_ = tiling->correction;

        uint32_t safeSeqLen = maxSeqLen_ == 0U ? 1U : maxSeqLen_;
        if (scoreTileLen_ == 0U) {
            scoreTileLen_ = 64U;
        }
        if (scoreTileLen_ > 256U) {
            scoreTileLen_ = 256U;
        }
        if (scoreTileLen_ > safeSeqLen) {
            scoreTileLen_ = safeSeqLen;
        }
        pipe_.InitBuffer(qRotBuf_, 4U * headDim_ * sizeof(float));
        pipe_.InitBuffer(qQjlBuf_, 4U * headDim_ * sizeof(float));
        pipe_.InitBuffer(scoreBuf_, 4U * scoreTileLen_ * sizeof(float));
        pipe_.InitBuffer(accBuf_, 4U * headDim_ * sizeof(float));
        pipe_.InitBuffer(vBuf_, headDim_ * sizeof(float));
        pipe_.InitBuffer(tmpBuf_, 4U * headDim_ * sizeof(float));
        pipe_.InitBuffer(reduceBuf_, 64U * sizeof(float));
    }

    __aicore__ inline void MTE2ToSSync()
    {
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }

    __aicore__ inline void VToSSync()
    {
        event_t eventId = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
    }

    __aicore__ inline void VToMTE3Sync()
    {
        event_t eventId = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void SToVSync()
    {
        event_t eventId = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);
    }

    __aicore__ inline void LoadCodebooks()
    {
        kCb0_ = kCodebookGm_.GetValue(0);
        kCb1_ = kCodebookGm_.GetValue(1);
        kCb2_ = kCodebookGm_.GetValue(2);
        kCb3_ = kCodebookGm_.GetValue(3);
        vCb0_ = vCodebookGm_.GetValue(0);
        vCb1_ = vCodebookGm_.GetValue(1);
        if (vBits_ >= 2U) {
            vCb2_ = vCodebookGm_.GetValue(2);
            vCb3_ = vCodebookGm_.GetValue(3);
        }
        if (vBits_ >= 3U) {
            vCb4_ = vCodebookGm_.GetValue(4);
            vCb5_ = vCodebookGm_.GetValue(5);
            vCb6_ = vCodebookGm_.GetValue(6);
            vCb7_ = vCodebookGm_.GetValue(7);
        }
    }

    __aicore__ inline float LookupKCodebook2(uint32_t idx)
    {
        switch (idx) {
            case 0U:
                return kCb0_;
            case 1U:
                return kCb1_;
            case 2U:
                return kCb2_;
            default:
                return kCb3_;
        }
    }

    __aicore__ inline float LookupVCodebook(uint32_t idx)
    {
        switch (idx) {
            case 0U:
                return vCb0_;
            case 1U:
                return vCb1_;
            case 2U:
                return vCb2_;
            case 3U:
                return vCb3_;
            case 4U:
                return vCb4_;
            case 5U:
                return vCb5_;
            case 6U:
                return vCb6_;
            default:
                return vCb7_;
        }
    }

    __aicore__ inline uint32_t ExtractVIndex(uint64_t packedBase, uint32_t d)
    {
        uint32_t bitPos = d * vBits_;
        uint32_t byteId = bitPos >> 3;
        uint32_t bitOff = bitPos & 7U;
        uint16_t value = static_cast<uint16_t>(
            vPackedIdxGm_.GetValue(packedBase + byteId));
        if (bitOff + vBits_ > 8U) {
            uint16_t high = static_cast<uint16_t>(
                vPackedIdxGm_.GetValue(packedBase + byteId + 1U));
            value = static_cast<uint16_t>(value | (high << 8));
        }
        uint32_t mask = (1U << vBits_) - 1U;
        return (value >> bitOff) & mask;
    }

    __aicore__ inline uint64_t CacheIndex(
        uint32_t b,
        uint32_t kvHead,
        uint32_t pos
    ) {
        uint32_t blockOffset = pos / blockSize_;
        uint32_t tokenOffset = pos - blockOffset * blockSize_;
        int32_t blockId = blockTableGm_.GetValue(
            static_cast<uint64_t>(b) * maxBlocksPerSeq_ + blockOffset);
        return ((static_cast<uint64_t>(blockId) * blockSize_ + tokenOffset)
            * numKvHeads_ + kvHead);
    }

    __aicore__ inline void StoreZero(uint32_t b, uint32_t kvHead)
    {
        uint32_t qHeadBase = kvHead * qPerKv_;
        uint64_t outBase =
            (static_cast<uint64_t>(b) * numHeads_ + qHeadBase) * headDim_;
        for (uint32_t q = 0; q < 4U; ++q) {
            for (uint32_t d = 0; d < headDim_; ++d) {
                outRotGm_.SetValue(outBase + q * headDim_ + d, 0.0F);
            }
        }
    }

    __aicore__ inline void ComputeTokenScores(
        uint32_t b,
        uint32_t kvHead,
        uint32_t pos,
        uint32_t tileOffset,
        const LocalTensor<float>& qRotLocal,
        const LocalTensor<float>& qQjlLocal,
        const LocalTensor<float>& scoreLocal
    ) {
        uint64_t cacheIndex = CacheIndex(b, kvHead, pos);
        uint64_t packedBase = cacheIndex * kPackedCols_;
        uint64_t qjlBase = cacheIndex * kQjlCols_;
        float gamma = kGammaGm_.GetValue(cacheIndex);
        float norm = kNormGm_.GetValue(cacheIndex);

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
                kPackedIdxGm_.GetValue(packedBase + g));
            uint32_t qjlBits = static_cast<uint32_t>(
                kPackedQjlGm_.GetValue(qjlBase + (g >> 1)));
            uint32_t qjlShift = (g & 1U) << 2;

            float cb0 = LookupKCodebook2(idxBits & 0x3U);
            float cb1 = LookupKCodebook2((idxBits >> 2) & 0x3U);
            float cb2 = LookupKCodebook2((idxBits >> 4) & 0x3U);
            float cb3 = LookupKCodebook2((idxBits >> 6) & 0x3U);
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

        float qjlScale = correction_ * gamma;
        float outScale = norm * scale_;
        scoreLocal.SetValue(
            tileOffset, (mseAcc0 + qjlScale * qjlAcc0) * outScale);
        scoreLocal.SetValue(
            scoreTileLen_ + tileOffset,
            (mseAcc1 + qjlScale * qjlAcc1) * outScale);
        scoreLocal.SetValue(
            (scoreTileLen_ << 1) + tileOffset,
            (mseAcc2 + qjlScale * qjlAcc2) * outScale);
        scoreLocal.SetValue(
            3U * scoreTileLen_ + tileOffset,
            (mseAcc3 + qjlScale * qjlAcc3) * outScale);
    }

    __aicore__ inline float RowMax(
        const LocalTensor<float>& scoreLocal,
        uint32_t rowOffset,
        uint32_t tileLen
    ) {
        float maxScore = -3.4028234663852886e38F;
        for (uint32_t i = 0; i < tileLen; ++i) {
            float value = scoreLocal.GetValue(rowOffset + i);
            if (value > maxScore) {
                maxScore = value;
            }
        }
        return maxScore;
    }

    __aicore__ inline float ExpScalar(
        float value,
        const LocalTensor<float>& reduceLocal
    ) {
        reduceLocal.SetValue(0, value);
        SToVSync();
        Exp(reduceLocal, reduceLocal, 1U);
        VToSSync();
        return reduceLocal.GetValue(0);
    }

    __aicore__ inline float ExpAndReduceRow(
        const LocalTensor<float>& scoreLocal,
        const LocalTensor<float>& reduceLocal,
        uint32_t rowOffset,
        uint32_t tileLen,
        float maxScore
    ) {
        SToVSync();
        Adds(scoreLocal[rowOffset], scoreLocal[rowOffset], -maxScore, tileLen);
        PipeBarrier<PIPE_V>();
        Exp(scoreLocal[rowOffset], scoreLocal[rowOffset], tileLen);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduceLocal, scoreLocal[rowOffset], reduceLocal, tileLen);
        VToSSync();
        return reduceLocal.GetValue(0);
    }

    __aicore__ inline void ScaleAccumulatorRows(
        const LocalTensor<float>& accLocal,
        float scale0,
        float scale1,
        float scale2,
        float scale3
    ) {
        uint32_t q1Base = headDim_;
        uint32_t q2Base = headDim_ << 1;
        uint32_t q3Base = 3U * headDim_;
        SToVSync();
        Muls(accLocal, accLocal, scale0, headDim_);
        PipeBarrier<PIPE_V>();
        Muls(accLocal[q1Base], accLocal[q1Base], scale1, headDim_);
        PipeBarrier<PIPE_V>();
        Muls(accLocal[q2Base], accLocal[q2Base], scale2, headDim_);
        PipeBarrier<PIPE_V>();
        Muls(accLocal[q3Base], accLocal[q3Base], scale3, headDim_);
        PipeBarrier<PIPE_V>();
        VToSSync();
    }

    __aicore__ inline void ScaleAccumulatorRowsIfNeeded(
        const LocalTensor<float>& accLocal,
        float scale0,
        float scale1,
        float scale2,
        float scale3,
        bool scaleRow0,
        bool scaleRow1,
        bool scaleRow2,
        bool scaleRow3
    ) {
        if (!scaleRow0 && !scaleRow1 && !scaleRow2 && !scaleRow3) {
            return;
        }

        uint32_t q1Base = headDim_;
        uint32_t q2Base = headDim_ << 1;
        uint32_t q3Base = 3U * headDim_;
        SToVSync();
        if (scaleRow0) {
            Muls(accLocal, accLocal, scale0, headDim_);
            PipeBarrier<PIPE_V>();
        }
        if (scaleRow1) {
            Muls(accLocal[q1Base], accLocal[q1Base], scale1, headDim_);
            PipeBarrier<PIPE_V>();
        }
        if (scaleRow2) {
            Muls(accLocal[q2Base], accLocal[q2Base], scale2, headDim_);
            PipeBarrier<PIPE_V>();
        }
        if (scaleRow3) {
            Muls(accLocal[q3Base], accLocal[q3Base], scale3, headDim_);
            PipeBarrier<PIPE_V>();
        }
        VToSSync();
    }

    __aicore__ inline void AccumulateTileV(
        uint32_t b,
        uint32_t kvHead,
        uint32_t tileStart,
        uint32_t tileLen,
        const LocalTensor<float>& scoreLocal,
        const LocalTensor<float>& accLocal
    ) {
        LocalTensor<float> vLocal = vBuf_.Get<float>();
        LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();

        uint32_t q1Base = headDim_;
        uint32_t q2Base = headDim_ << 1;
        uint32_t q3Base = 3U * headDim_;
        for (uint32_t i = 0; i < tileLen; ++i) {
            uint32_t pos = tileStart + i;
            uint64_t cacheIndex = CacheIndex(b, kvHead, pos);
            uint64_t vPackedBase = cacheIndex * vPackedCols_;
            float vNorm = vNormGm_.GetValue(cacheIndex);
            float w0 = scoreLocal.GetValue(i);
            float w1 = scoreLocal.GetValue(scoreTileLen_ + i);
            float w2 = scoreLocal.GetValue((scoreTileLen_ << 1) + i);
            float w3 = scoreLocal.GetValue(3U * scoreTileLen_ + i);
            for (uint32_t d = 0; d < headDim_; ++d) {
                float v = LookupVCodebook(ExtractVIndex(vPackedBase, d)) * vNorm;
                vLocal.SetValue(d, v);
            }
            SToVSync();
            if ((headDim_ & 7U) == 0U) {
                Duplicate(tmpLocal, w0, headDim_);
                Duplicate(tmpLocal[q1Base], w1, headDim_);
                Duplicate(tmpLocal[q2Base], w2, headDim_);
                Duplicate(tmpLocal[q3Base], w3, headDim_);
                PipeBarrier<PIPE_V>();

                uint8_t repeatStride = static_cast<uint8_t>(headDim_ >> 3);
                BinaryRepeatParams repeatParams;
                repeatParams.dstBlkStride = 1;
                repeatParams.src0BlkStride = 1;
                repeatParams.src1BlkStride = 1;
                repeatParams.dstRepStride = repeatStride;
                repeatParams.src0RepStride = repeatStride;
                repeatParams.src1RepStride = 0;
                uint32_t offset = 0U;
                while (offset < headDim_) {
                    uint64_t mask = headDim_ - offset;
                    if (mask > 64U) {
                        mask = 64U;
                    }
                    MulAddDst(
                        accLocal[offset],
                        tmpLocal[offset],
                        vLocal[offset],
                        mask,
                        4U,
                        repeatParams);
                    offset += 64U;
                }
                PipeBarrier<PIPE_V>();
            } else {
                Duplicate(tmpLocal, w0, headDim_);
                PipeBarrier<PIPE_V>();
                MulAddDst(accLocal, vLocal, tmpLocal, headDim_);
                PipeBarrier<PIPE_V>();
                Duplicate(tmpLocal, w1, headDim_);
                PipeBarrier<PIPE_V>();
                MulAddDst(accLocal[q1Base], vLocal, tmpLocal, headDim_);
                PipeBarrier<PIPE_V>();
                Duplicate(tmpLocal, w2, headDim_);
                PipeBarrier<PIPE_V>();
                MulAddDst(accLocal[q2Base], vLocal, tmpLocal, headDim_);
                PipeBarrier<PIPE_V>();
                Duplicate(tmpLocal, w3, headDim_);
                PipeBarrier<PIPE_V>();
                MulAddDst(accLocal[q3Base], vLocal, tmpLocal, headDim_);
                PipeBarrier<PIPE_V>();
            }
            VToSSync();
        }
    }

    __aicore__ inline void StoreNormalizedAccumulator(
        uint32_t b,
        uint32_t kvHead,
        const LocalTensor<float>& accLocal,
        float sum0,
        float sum1,
        float sum2,
        float sum3
    ) {
        ScaleAccumulatorRows(
            accLocal,
            1.0F / sum0,
            1.0F / sum1,
            1.0F / sum2,
            1.0F / sum3);

        uint32_t qHeadBase = kvHead * qPerKv_;
        uint64_t outBase =
            (static_cast<uint64_t>(b) * numHeads_ + qHeadBase) * headDim_;
        VToMTE3Sync();
        DataCopy(outRotGm_[outBase], accLocal, 4U * headDim_);
    }

    __aicore__ inline void ProcessKvHead(uint32_t b, uint32_t kvHead)
    {
        int32_t seqLenRaw = seqLensGm_.GetValue(b);
        uint32_t seqLen = seqLenRaw > 0 ? static_cast<uint32_t>(seqLenRaw) : 0U;
        if (seqLen > maxSeqLen_) {
            seqLen = maxSeqLen_;
        }
        if (seqLen == 0U || qPerKv_ != 4U || kStage1Bits_ != 2U
            || (headDim_ & 3U) != 0U) {
            StoreZero(b, kvHead);
            return;
        }

        uint32_t qHeadBase = kvHead * qPerKv_;
        uint64_t queryBase =
            (static_cast<uint64_t>(b) * numHeads_ + qHeadBase) * headDim_;
        uint32_t queryElems = 4U * headDim_;

        LocalTensor<float> qRotLocal = qRotBuf_.Get<float>();
        LocalTensor<float> qQjlLocal = qQjlBuf_.Get<float>();
        LocalTensor<float> scoreLocal = scoreBuf_.Get<float>();
        LocalTensor<float> accLocal = accBuf_.Get<float>();
        LocalTensor<float> reduceLocal = reduceBuf_.Get<float>();

        DataCopy(qRotLocal, qRotGm_[queryBase], queryElems);
        DataCopy(qQjlLocal, qQjlGm_[queryBase], queryElems);
        MTE2ToSSync();

        Duplicate(accLocal, 0.0F, 4U * headDim_);
        PipeBarrier<PIPE_V>();

        float max0 = -3.4028234663852886e38F;
        float max1 = -3.4028234663852886e38F;
        float max2 = -3.4028234663852886e38F;
        float max3 = -3.4028234663852886e38F;
        float sum0 = 0.0F;
        float sum1 = 0.0F;
        float sum2 = 0.0F;
        float sum3 = 0.0F;
        bool initialized = false;

        for (uint32_t tileStart = 0; tileStart < seqLen;
             tileStart += scoreTileLen_) {
            uint32_t tileLen = seqLen - tileStart;
            if (tileLen > scoreTileLen_) {
                tileLen = scoreTileLen_;
            }
            for (uint32_t i = 0; i < tileLen; ++i) {
                ComputeTokenScores(
                    b,
                    kvHead,
                    tileStart + i,
                    i,
                    qRotLocal,
                    qQjlLocal,
                    scoreLocal);
            }

            uint32_t row1 = scoreTileLen_;
            uint32_t row2 = scoreTileLen_ << 1;
            uint32_t row3 = 3U * scoreTileLen_;
            float tileMax0 = RowMax(scoreLocal, 0U, tileLen);
            float tileMax1 = RowMax(scoreLocal, row1, tileLen);
            float tileMax2 = RowMax(scoreLocal, row2, tileLen);
            float tileMax3 = RowMax(scoreLocal, row3, tileLen);
            float newMax0 = max0 > tileMax0 ? max0 : tileMax0;
            float newMax1 = max1 > tileMax1 ? max1 : tileMax1;
            float newMax2 = max2 > tileMax2 ? max2 : tileMax2;
            float newMax3 = max3 > tileMax3 ? max3 : tileMax3;
            bool updateMax0 = initialized && tileMax0 > max0;
            bool updateMax1 = initialized && tileMax1 > max1;
            bool updateMax2 = initialized && tileMax2 > max2;
            bool updateMax3 = initialized && tileMax3 > max3;
            float alpha0 = initialized ? 1.0F : 0.0F;
            float alpha1 = initialized ? 1.0F : 0.0F;
            float alpha2 = initialized ? 1.0F : 0.0F;
            float alpha3 = initialized ? 1.0F : 0.0F;
            if (updateMax0) {
                alpha0 = ExpScalar(max0 - newMax0, reduceLocal);
            }
            if (updateMax1) {
                alpha1 = ExpScalar(max1 - newMax1, reduceLocal);
            }
            if (updateMax2) {
                alpha2 = ExpScalar(max2 - newMax2, reduceLocal);
            }
            if (updateMax3) {
                alpha3 = ExpScalar(max3 - newMax3, reduceLocal);
            }
            float tileSum0 = ExpAndReduceRow(
                scoreLocal, reduceLocal, 0U, tileLen, newMax0);
            float tileSum1 = ExpAndReduceRow(
                scoreLocal, reduceLocal, row1, tileLen, newMax1);
            float tileSum2 = ExpAndReduceRow(
                scoreLocal, reduceLocal, row2, tileLen, newMax2);
            float tileSum3 = ExpAndReduceRow(
                scoreLocal, reduceLocal, row3, tileLen, newMax3);

            ScaleAccumulatorRowsIfNeeded(
                accLocal,
                alpha0,
                alpha1,
                alpha2,
                alpha3,
                updateMax0,
                updateMax1,
                updateMax2,
                updateMax3);
            AccumulateTileV(
                b, kvHead, tileStart, tileLen, scoreLocal, accLocal);

            sum0 = sum0 * alpha0 + tileSum0;
            sum1 = sum1 * alpha1 + tileSum1;
            sum2 = sum2 * alpha2 + tileSum2;
            sum3 = sum3 * alpha3 + tileSum3;
            max0 = newMax0;
            max1 = newMax1;
            max2 = newMax2;
            max3 = newMax3;
            initialized = true;
        }

        StoreNormalizedAccumulator(
            b, kvHead, accLocal, sum0, sum1, sum2, sum3);
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();
        uint64_t totalTasks =
            static_cast<uint64_t>(batch_) * static_cast<uint64_t>(numKvHeads_);
        if (totalTasks == 0) {
            return;
        }

        LoadCodebooks();
        uint64_t coreCount = static_cast<uint64_t>(numCore_ == 0 ? 1 : numCore_);
        uint64_t tasksPerCore = (totalTasks + coreCount - 1U) / coreCount;
        uint64_t start = static_cast<uint64_t>(coreId) * tasksPerCore;
        uint64_t end = start + tasksPerCore;
        if (end > totalTasks) {
            end = totalTasks;
        }

        for (uint64_t linear = start; linear < end; ++linear) {
            uint32_t kvHead = static_cast<uint32_t>(linear % numKvHeads_);
            uint32_t b = static_cast<uint32_t>(linear / numKvHeads_);
            ProcessKvHead(b, kvHead);
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> qRotBuf_;
    TBuf<TPosition::VECCALC> qQjlBuf_;
    TBuf<TPosition::VECCALC> scoreBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> vBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> reduceBuf_;

    GlobalTensor<float> qRotGm_;
    GlobalTensor<float> qQjlGm_;
    GlobalTensor<uint8_t> kPackedIdxGm_;
    GlobalTensor<uint8_t> kPackedQjlGm_;
    GlobalTensor<float> kGammaGm_;
    GlobalTensor<float> kNormGm_;
    GlobalTensor<uint8_t> vPackedIdxGm_;
    GlobalTensor<float> vNormGm_;
    GlobalTensor<int32_t> blockTableGm_;
    GlobalTensor<int32_t> seqLensGm_;
    GlobalTensor<float> kCodebookGm_;
    GlobalTensor<float> vCodebookGm_;
    GlobalTensor<float> outRotGm_;

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
    uint32_t scoreTileLen_{1};
    uint32_t numCore_{0};
    float scale_{1.0F};
    float correction_{0.0F};

    float kCb0_{0.0F};
    float kCb1_{0.0F};
    float kCb2_{0.0F};
    float kCb3_{0.0F};
    float vCb0_{0.0F};
    float vCb1_{0.0F};
    float vCb2_{0.0F};
    float vCb3_{0.0F};
    float vCb4_{0.0F};
    float vCb5_{0.0F};
    float vCb6_{0.0F};
    float vCb7_{0.0F};
};

extern "C" __global__ __aicore__ void tq_prod_mse_paged_attention(
    GM_ADDR qRot,
    GM_ADDR qQjl,
    GM_ADDR kPackedIdx,
    GM_ADDR kPackedQjl,
    GM_ADDR kGamma,
    GM_ADDR kNorm,
    GM_ADDR vPackedIdx,
    GM_ADDR vNorm,
    GM_ADDR blockTable,
    GM_ADDR seqLens,
    GM_ADDR kCodebook,
    GM_ADDR vCodebook,
    GM_ADDR outRot,
    GM_ADDR workspace,
    GM_ADDR tiling
) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TqProdMsePagedAttentionTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        TqProdMsePagedAttentionTilingData, tilingData, tiling);

    KernelTqProdMsePagedAttention op;
    op.Init(
        qRot,
        qQjl,
        kPackedIdx,
        kPackedQjl,
        kGamma,
        kNorm,
        vPackedIdx,
        vNorm,
        blockTable,
        seqLens,
        kCodebook,
        vCodebook,
        outRot,
        &tilingData
    );
    op.Process();
}
