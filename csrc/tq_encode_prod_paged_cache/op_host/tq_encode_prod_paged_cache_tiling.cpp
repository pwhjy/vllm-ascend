/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

#include "tq_encode_prod_paged_cache_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TqEncodeProdPagedCacheTilingFunc(gert::TilingContext* context)
{
    auto xShape = context->GetInputShape(0)->GetStorageShape();
    auto idxCacheShape = context->GetInputShape(2)->GetStorageShape();
    auto qjlCacheShape = context->GetInputShape(3)->GetStorageShape();

    uint32_t totalTokens = static_cast<uint32_t>(xShape.GetDim(0));
    uint32_t numKvHeads = static_cast<uint32_t>(xShape.GetDim(1));
    uint32_t blockSize = static_cast<uint32_t>(idxCacheShape.GetDim(1));
    uint32_t idxPackedCols = static_cast<uint32_t>(idxCacheShape.GetDim(3));
    uint32_t qjlPackedCols = static_cast<uint32_t>(qjlCacheShape.GetDim(3));

    auto attr = context->GetAttrs();
    const int64_t* totalBitsPtr = attr->GetAttrPointer<int64_t>(0);
    const int64_t* stage1BitsPtr = attr->GetAttrPointer<int64_t>(1);
    const int64_t* headDimPtr = attr->GetAttrPointer<int64_t>(2);

    uint32_t totalBits = static_cast<uint32_t>(
        totalBitsPtr != nullptr ? *totalBitsPtr : 0);
    uint32_t stage1Bits = static_cast<uint32_t>(
        stage1BitsPtr != nullptr ? *stage1BitsPtr : 0);
    uint32_t headDim = static_cast<uint32_t>(
        headDimPtr != nullptr ? *headDimPtr : 0);

    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platformInfo.GetCoreNumAiv();
    uint32_t usefulCore = totalTokens * numKvHeads;
    if (usefulCore > 0 && usefulCore < coreNum) {
        coreNum = usefulCore;
    }
    if (coreNum == 0) {
        coreNum = 1;
    }

    TqEncodeProdPagedCacheTilingData tiling;
    tiling.set_totalTokens(totalTokens);
    tiling.set_numKvHeads(numKvHeads);
    tiling.set_blockSize(blockSize);
    tiling.set_idxPackedCols(idxPackedCols);
    tiling.set_qjlPackedCols(qjlPackedCols);
    tiling.set_totalBits(totalBits);
    tiling.set_stage1Bits(stage1Bits);
    tiling.set_headDim(headDim);
    tiling.set_numCore(coreNum);

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

struct TqEncodeProdPagedCacheCompileInfo {};

ge::graphStatus TilingParseForTqEncodeProdPagedCache(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TqEncodeProdPagedCache)
    .Tiling(TqEncodeProdPagedCacheTilingFunc)
    .TilingParse<TqEncodeProdPagedCacheCompileInfo>(
        TilingParseForTqEncodeProdPagedCache);

}  // namespace optiling
