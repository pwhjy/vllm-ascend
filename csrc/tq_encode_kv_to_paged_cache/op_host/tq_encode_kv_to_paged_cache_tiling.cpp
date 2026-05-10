/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

#include "tq_encode_kv_to_paged_cache_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus TqEncodeKvToPagedCacheTilingFunc(gert::TilingContext* context)
{
    auto keyShape = context->GetInputShape(0)->GetStorageShape();
    auto kIdxCacheShape = context->GetInputShape(3)->GetStorageShape();
    auto kQjlCacheShape = context->GetInputShape(4)->GetStorageShape();
    auto vIdxCacheShape = context->GetInputShape(7)->GetStorageShape();

    uint32_t totalTokens = static_cast<uint32_t>(keyShape.GetDim(0));
    uint32_t numKvHeads = static_cast<uint32_t>(keyShape.GetDim(1));
    uint32_t blockSize = static_cast<uint32_t>(kIdxCacheShape.GetDim(1));
    uint32_t kPackedCols = static_cast<uint32_t>(kIdxCacheShape.GetDim(3));
    uint32_t kQjlCols = static_cast<uint32_t>(kQjlCacheShape.GetDim(3));
    uint32_t vPackedCols = static_cast<uint32_t>(vIdxCacheShape.GetDim(3));

    auto attr = context->GetAttrs();
    const int64_t* totalBitsPtr = attr->GetAttrPointer<int64_t>(0);
    const int64_t* stage1BitsPtr = attr->GetAttrPointer<int64_t>(1);
    const int64_t* vBitsPtr = attr->GetAttrPointer<int64_t>(2);
    const int64_t* headDimPtr = attr->GetAttrPointer<int64_t>(3);
    const int64_t* debugModePtr = attr->GetAttrPointer<int64_t>(4);

    uint32_t totalBits = static_cast<uint32_t>(
        totalBitsPtr != nullptr ? *totalBitsPtr : 0);
    uint32_t stage1Bits = static_cast<uint32_t>(
        stage1BitsPtr != nullptr ? *stage1BitsPtr : 0);
    uint32_t vBits = static_cast<uint32_t>(
        vBitsPtr != nullptr ? *vBitsPtr : 0);
    uint32_t headDim = static_cast<uint32_t>(
        headDimPtr != nullptr ? *headDimPtr : 0);
    int64_t debugModeAttr = debugModePtr != nullptr ? *debugModePtr : 0;
    uint32_t debugMode =
        debugModeAttr > 0 ? static_cast<uint32_t>(debugModeAttr) : 0U;
    if (debugMode > 6U) {
        debugMode = 6U;
    }

    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platformInfo.GetCoreNumAiv();
    uint32_t partitionCount =
        (vPackedCols % 4U) == 0U
        && (vBits == 1U || vBits == 2U || vBits == 4U) ? 4U : 1U;
    uint64_t usefulCore =
        static_cast<uint64_t>(totalTokens) * numKvHeads * partitionCount;
    if (usefulCore > 0 && usefulCore < coreNum) {
        coreNum = static_cast<uint32_t>(usefulCore);
    }
    if (coreNum == 0) {
        coreNum = 1;
    }

    TqEncodeKvToPagedCacheTilingData tiling;
    tiling.set_totalTokens(totalTokens);
    tiling.set_numKvHeads(numKvHeads);
    tiling.set_blockSize(blockSize);
    tiling.set_kPackedCols(kPackedCols);
    tiling.set_kQjlCols(kQjlCols);
    tiling.set_vPackedCols(vPackedCols);
    tiling.set_totalBits(totalBits);
    tiling.set_stage1Bits(stage1Bits);
    tiling.set_vBits(vBits);
    tiling.set_headDim(headDim);
    tiling.set_debugMode(debugMode);
    tiling.set_numCore(coreNum);

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseForTqEncodeKvToPagedCache(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TqEncodeKvToPagedCache)
    .Tiling(TqEncodeKvToPagedCacheTilingFunc)
    .TilingParse<TqEncodeKvToPagedCacheCompileInfo>(
        TilingParseForTqEncodeKvToPagedCache);

}  // namespace optiling
