/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "tq_prod_mse_paged_attention_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdint>

namespace optiling {

static uint32_t NormalizeScoreTileLen(int64_t scoreTileLenAttr)
{
    // This is a UB row stride, so keep it 32-byte aligned even for short seqs.
    uint32_t scoreTileLen = scoreTileLenAttr > 0
        ? static_cast<uint32_t>(scoreTileLenAttr)
        : 64U;
    if (scoreTileLen > 256U) {
        scoreTileLen = 256U;
    }
    if (scoreTileLen < 8U) {
        scoreTileLen = 8U;
    }
    return ((scoreTileLen + 7U) / 8U) * 8U;
}

static ge::graphStatus TqProdMsePagedAttentionTilingFunc(
    gert::TilingContext* context)
{
    auto qRotShape = context->GetInputShape(0)->GetStorageShape();
    auto kPackedIdxShape = context->GetInputShape(2)->GetStorageShape();
    auto kPackedQjlShape = context->GetInputShape(3)->GetStorageShape();
    auto vPackedIdxShape = context->GetInputShape(6)->GetStorageShape();
    auto blockTableShape = context->GetInputShape(8)->GetStorageShape();

    uint32_t batch = static_cast<uint32_t>(qRotShape.GetDim(0));
    uint32_t numHeads = static_cast<uint32_t>(qRotShape.GetDim(1));
    uint32_t blockSize = static_cast<uint32_t>(kPackedIdxShape.GetDim(1));
    uint32_t numKvHeads = static_cast<uint32_t>(kPackedIdxShape.GetDim(2));
    uint32_t kPackedCols = static_cast<uint32_t>(kPackedIdxShape.GetDim(3));
    uint32_t kQjlCols = static_cast<uint32_t>(kPackedQjlShape.GetDim(3));
    uint32_t vPackedCols = static_cast<uint32_t>(vPackedIdxShape.GetDim(3));
    uint32_t maxBlocksPerSeq = static_cast<uint32_t>(blockTableShape.GetDim(1));

    auto attr = context->GetAttrs();
    const int64_t* kTotalBitsPtr = attr->GetAttrPointer<int64_t>(0);
    const int64_t* vBitsPtr = attr->GetAttrPointer<int64_t>(1);
    const int64_t* headDimPtr = attr->GetAttrPointer<int64_t>(2);
    const float* scalePtr = attr->GetAttrPointer<float>(3);
    const int64_t* maxSeqLenPtr = attr->GetAttrPointer<int64_t>(4);
    const int64_t* scoreTileLenPtr = attr->GetAttrPointer<int64_t>(5);

    uint32_t kTotalBits = static_cast<uint32_t>(
        kTotalBitsPtr != nullptr ? *kTotalBitsPtr : 0);
    uint32_t vBits = static_cast<uint32_t>(
        vBitsPtr != nullptr ? *vBitsPtr : 0);
    uint32_t headDim = static_cast<uint32_t>(
        headDimPtr != nullptr ? *headDimPtr : 0);
    float scale = scalePtr != nullptr ? *scalePtr : 1.0F;
    uint32_t maxSeqLen = static_cast<uint32_t>(
        maxSeqLenPtr != nullptr ? *maxSeqLenPtr : 0);
    int64_t scoreTileLenAttr =
        scoreTileLenPtr != nullptr ? *scoreTileLenPtr : 64;
    uint32_t scoreTileLen = NormalizeScoreTileLen(scoreTileLenAttr);

    uint32_t qPerKv = numKvHeads == 0 ? 1U : numHeads / numKvHeads;
    if (qPerKv == 0U) {
        qPerKv = 1U;
    }
    uint32_t kStage1Bits = kTotalBits > 0 ? kTotalBits - 1U : 0U;
    float correction = headDim == 0U ? 0.0F : 1.2533141373155001F / headDim;

    auto platformInfo = platform_ascendc::PlatformAscendC(
        context->GetPlatformInfo());
    uint32_t coreNum = platformInfo.GetCoreNumAiv();
    uint64_t usefulCore =
        static_cast<uint64_t>(batch) * static_cast<uint64_t>(numKvHeads);
    if (usefulCore > 0U && usefulCore < coreNum) {
        coreNum = static_cast<uint32_t>(usefulCore);
    }
    if (coreNum == 0U) {
        coreNum = 1U;
    }

    TqProdMsePagedAttentionTilingData tiling;
    tiling.set_batch(batch);
    tiling.set_numHeads(numHeads);
    tiling.set_numKvHeads(numKvHeads);
    tiling.set_qPerKv(qPerKv);
    tiling.set_blockSize(blockSize);
    tiling.set_maxBlocksPerSeq(maxBlocksPerSeq);
    tiling.set_maxSeqLen(maxSeqLen);
    tiling.set_headDim(headDim);
    tiling.set_kPackedCols(kPackedCols);
    tiling.set_kQjlCols(kQjlCols);
    tiling.set_vPackedCols(vPackedCols);
    tiling.set_kStage1Bits(kStage1Bits);
    tiling.set_vBits(vBits);
    tiling.set_scoreTileLen(scoreTileLen);
    tiling.set_numCore(coreNum);
    tiling.set_scale(scale);
    tiling.set_correction(correction);

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseForTqProdMsePagedAttention(
    gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TqProdMsePagedAttention)
    .Tiling(TqProdMsePagedAttentionTilingFunc)
    .TilingParse<TqProdMsePagedAttentionCompileInfo>(
        TilingParseForTqProdMsePagedAttention);

}  // namespace optiling
