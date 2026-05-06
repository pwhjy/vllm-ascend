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
        numCore_ = tiling->numCore;
        scale_ = tiling->scale;
        correction_ = tiling->correction;

        uint32_t safeSeqLen = maxSeqLen_ == 0U ? 1U : maxSeqLen_;
        pipe_.InitBuffer(qRotBuf_, 4U * headDim_ * sizeof(float));
        pipe_.InitBuffer(qQjlBuf_, 4U * headDim_ * sizeof(float));
        pipe_.InitBuffer(scoreBuf_, 4U * safeSeqLen * sizeof(float));
        pipe_.InitBuffer(accBuf_, 4U * headDim_ * sizeof(float));
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
            pos, (mseAcc0 + qjlScale * qjlAcc0) * outScale);
        scoreLocal.SetValue(
            maxSeqLen_ + pos, (mseAcc1 + qjlScale * qjlAcc1) * outScale);
        scoreLocal.SetValue(
            (maxSeqLen_ << 1) + pos,
            (mseAcc2 + qjlScale * qjlAcc2) * outScale);
        scoreLocal.SetValue(
            3U * maxSeqLen_ + pos,
            (mseAcc3 + qjlScale * qjlAcc3) * outScale);
    }

    __aicore__ inline void SoftmaxRow(
        const LocalTensor<float>& scoreLocal,
        const LocalTensor<float>& reduceLocal,
        uint32_t rowOffset,
        uint32_t seqLen
    ) {
        float maxScore = -3.4028234663852886e38F;
        for (uint32_t i = 0; i < seqLen; ++i) {
            float value = scoreLocal.GetValue(rowOffset + i);
            if (value > maxScore) {
                maxScore = value;
            }
        }
        for (uint32_t i = 0; i < seqLen; ++i) {
            scoreLocal.SetValue(rowOffset + i,
                scoreLocal.GetValue(rowOffset + i) - maxScore);
        }

        SToVSync();
        Exp(scoreLocal[rowOffset], scoreLocal[rowOffset], seqLen);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduceLocal, scoreLocal[rowOffset], reduceLocal, seqLen);
        VToSSync();
        float sumValue = reduceLocal.GetValue(0);
        float invSum = 1.0F / sumValue;
        SToVSync();
        Muls(scoreLocal[rowOffset], scoreLocal[rowOffset], invSum, seqLen);
        PipeBarrier<PIPE_V>();
        VToSSync();
    }

    __aicore__ inline void AccumulateVAndStore(
        uint32_t b,
        uint32_t kvHead,
        uint32_t seqLen,
        const LocalTensor<float>& scoreLocal,
        const LocalTensor<float>& accLocal
    ) {
        for (uint32_t i = 0; i < 4U * headDim_; ++i) {
            accLocal.SetValue(i, 0.0F);
        }

        uint32_t q1Base = headDim_;
        uint32_t q2Base = headDim_ << 1;
        uint32_t q3Base = 3U * headDim_;
        for (uint32_t pos = 0; pos < seqLen; ++pos) {
            uint64_t cacheIndex = CacheIndex(b, kvHead, pos);
            uint64_t vPackedBase = cacheIndex * vPackedCols_;
            float vNorm = vNormGm_.GetValue(cacheIndex);
            float w0 = scoreLocal.GetValue(pos);
            float w1 = scoreLocal.GetValue(maxSeqLen_ + pos);
            float w2 = scoreLocal.GetValue((maxSeqLen_ << 1) + pos);
            float w3 = scoreLocal.GetValue(3U * maxSeqLen_ + pos);
            for (uint32_t d = 0; d < headDim_; ++d) {
                float v = LookupVCodebook(ExtractVIndex(vPackedBase, d)) * vNorm;
                accLocal.SetValue(d, accLocal.GetValue(d) + w0 * v);
                accLocal.SetValue(q1Base + d,
                    accLocal.GetValue(q1Base + d) + w1 * v);
                accLocal.SetValue(q2Base + d,
                    accLocal.GetValue(q2Base + d) + w2 * v);
                accLocal.SetValue(q3Base + d,
                    accLocal.GetValue(q3Base + d) + w3 * v);
            }
        }

        uint32_t qHeadBase = kvHead * qPerKv_;
        uint64_t outBase =
            (static_cast<uint64_t>(b) * numHeads_ + qHeadBase) * headDim_;
        for (uint32_t i = 0; i < 4U * headDim_; ++i) {
            outRotGm_.SetValue(outBase + i, accLocal.GetValue(i));
        }
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

        for (uint32_t pos = 0; pos < seqLen; ++pos) {
            ComputeTokenScores(
                b, kvHead, pos, qRotLocal, qQjlLocal, scoreLocal);
        }

        SoftmaxRow(scoreLocal, reduceLocal, 0U, seqLen);
        SoftmaxRow(scoreLocal, reduceLocal, maxSeqLen_, seqLen);
        SoftmaxRow(scoreLocal, reduceLocal, maxSeqLen_ << 1, seqLen);
        SoftmaxRow(scoreLocal, reduceLocal, 3U * maxSeqLen_, seqLen);
        AccumulateVAndStore(b, kvHead, seqLen, scoreLocal, accLocal);
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
