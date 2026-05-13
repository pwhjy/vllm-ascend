/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

#include "tq_encode_mse_paged_cache_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TqEncodeMsePagedCacheTilingFunc(gert::TilingContext* context)
{
    auto xShape = context->GetInputShape(0)->GetStorageShape();
    auto cacheShape = context->GetInputShape(4)->GetStorageShape();

    uint32_t totalTokens = static_cast<uint32_t>(xShape.GetDim(0));
    uint32_t numKvHeads = static_cast<uint32_t>(xShape.GetDim(1));
    uint32_t blockSize = static_cast<uint32_t>(cacheShape.GetDim(1));
    uint32_t packedCols = static_cast<uint32_t>(cacheShape.GetDim(3));

    auto attr = context->GetAttrs();
    const int64_t* bitsPtr = attr->GetAttrPointer<int64_t>(0);
    const int64_t* headDimPtr = attr->GetAttrPointer<int64_t>(1);

    uint32_t bits = static_cast<uint32_t>(bitsPtr != nullptr ? *bitsPtr : 0);
    uint32_t headDim = static_cast<uint32_t>(headDimPtr != nullptr ? *headDimPtr : 0);

    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platformInfo.GetCoreNumAiv();
    uint32_t usefulCore = totalTokens * numKvHeads;
    if (usefulCore > 0 && usefulCore < coreNum) {
        coreNum = usefulCore;
    }
    if (coreNum == 0) {
        coreNum = 1;
    }

    TqEncodeMsePagedCacheTilingData tiling;
    tiling.set_totalTokens(totalTokens);
    tiling.set_numKvHeads(numKvHeads);
    tiling.set_blockSize(blockSize);
    tiling.set_packedCols(packedCols);
    tiling.set_bits(bits);
    tiling.set_headDim(headDim);
    tiling.set_numCore(coreNum);

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

struct TqEncodeMsePagedCacheCompileInfo {};

ge::graphStatus TilingParseForTqEncodeMsePagedCache(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TqEncodeMsePagedCache)
    .Tiling(TqEncodeMsePagedCacheTilingFunc)
    .TilingParse<TqEncodeMsePagedCacheCompileInfo>(
        TilingParseForTqEncodeMsePagedCache);

}  // namespace optiling
