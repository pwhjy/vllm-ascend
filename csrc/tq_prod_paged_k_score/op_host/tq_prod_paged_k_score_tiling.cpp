/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "tq_prod_paged_k_score_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdint>

namespace optiling {

static ge::graphStatus TqProdPagedKScoreTilingFunc(gert::TilingContext* context)
{
    auto qRotShape = context->GetInputShape(0)->GetStorageShape();
    auto packedIdxShape = context->GetInputShape(2)->GetStorageShape();
    auto packedQjlShape = context->GetInputShape(3)->GetStorageShape();
    auto blockTableShape = context->GetInputShape(6)->GetStorageShape();

    uint32_t batch = static_cast<uint32_t>(qRotShape.GetDim(0));
    uint32_t numHeads = static_cast<uint32_t>(qRotShape.GetDim(1));
    uint32_t blockSize = static_cast<uint32_t>(packedIdxShape.GetDim(1));
    uint32_t numKvHeads = static_cast<uint32_t>(packedIdxShape.GetDim(2));
    uint32_t packedCols = static_cast<uint32_t>(packedIdxShape.GetDim(3));
    uint32_t qjlCols = static_cast<uint32_t>(packedQjlShape.GetDim(3));
    uint32_t maxBlocksPerSeq = static_cast<uint32_t>(blockTableShape.GetDim(1));

    auto attr = context->GetAttrs();
    const int64_t* totalBitsPtr = attr->GetAttrPointer<int64_t>(0);
    const int64_t* headDimPtr = attr->GetAttrPointer<int64_t>(1);
    const float* scalePtr = attr->GetAttrPointer<float>(2);
    const int64_t* maxSeqLenPtr = attr->GetAttrPointer<int64_t>(3);

    uint32_t totalBits = static_cast<uint32_t>(
        totalBitsPtr != nullptr ? *totalBitsPtr : 0);
    uint32_t headDim = static_cast<uint32_t>(
        headDimPtr != nullptr ? *headDimPtr : 0);
    float scale = scalePtr != nullptr ? *scalePtr : 1.0F;
    uint32_t maxSeqLen = static_cast<uint32_t>(
        maxSeqLenPtr != nullptr ? *maxSeqLenPtr : 0);
    uint32_t stage1Bits = totalBits > 0 ? totalBits - 1U : 0U;
    float correction = headDim == 0U ? 0.0F : 1.2533141373155001F / headDim;
    uint32_t qPerKv = numKvHeads == 0 ? 1U : numHeads / numKvHeads;
    if (qPerKv == 0) {
        qPerKv = 1;
    }
    uint32_t scoreTileLen = qPerKv == 4U ? 128U : 1U;
    uint32_t maxSeqTiles =
        scoreTileLen == 0U ? maxSeqLen :
        (maxSeqLen + scoreTileLen - 1U) / scoreTileLen;

    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platformInfo.GetCoreNumAiv();
    uint64_t usefulCore =
        static_cast<uint64_t>(batch) *
        static_cast<uint64_t>(numKvHeads) *
        static_cast<uint64_t>(maxSeqTiles);
    if (usefulCore > 0 && usefulCore < coreNum) {
        coreNum = static_cast<uint32_t>(usefulCore);
    }
    if (coreNum == 0) {
        coreNum = 1;
    }

    TqProdPagedKScoreTilingData tiling;
    tiling.set_batch(batch);
    tiling.set_numHeads(numHeads);
    tiling.set_numKvHeads(numKvHeads);
    tiling.set_qPerKv(qPerKv);
    tiling.set_blockSize(blockSize);
    tiling.set_maxBlocksPerSeq(maxBlocksPerSeq);
    tiling.set_maxSeqLen(maxSeqLen);
    tiling.set_headDim(headDim);
    tiling.set_packedCols(packedCols);
    tiling.set_qjlCols(qjlCols);
    tiling.set_stage1Bits(stage1Bits);
    tiling.set_scoreTileLen(scoreTileLen);
    tiling.set_maxSeqTiles(maxSeqTiles);
    tiling.set_numCore(coreNum);
    tiling.set_scale(scale);
    tiling.set_correction(correction);

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseForTqProdPagedKScore(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TqProdPagedKScore)
    .Tiling(TqProdPagedKScoreTilingFunc)
    .TilingParse<TqProdPagedKScoreCompileInfo>(
        TilingParseForTqProdPagedKScore);

}  // namespace optiling
