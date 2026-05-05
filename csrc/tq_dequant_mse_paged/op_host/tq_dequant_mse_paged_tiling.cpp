/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "tq_dequant_mse_paged_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdint>

namespace optiling {

static ge::graphStatus TqDequantMsePagedTilingFunc(gert::TilingContext* context)
{
    auto packedIdxShape = context->GetInputShape(0)->GetStorageShape();
    auto tokenBlockIdsShape = context->GetInputShape(2)->GetStorageShape();

    uint32_t totalTokens = static_cast<uint32_t>(tokenBlockIdsShape.GetDim(0));
    uint32_t blockSize = static_cast<uint32_t>(packedIdxShape.GetDim(1));
    uint32_t numKvHeads = static_cast<uint32_t>(packedIdxShape.GetDim(2));
    uint32_t packedCols = static_cast<uint32_t>(packedIdxShape.GetDim(3));

    int64_t bitsAttr = 0;
    int64_t headDimAttr = 0;
    context->GetAttrs()->GetAttrPointer<int64_t>(0, bitsAttr);
    context->GetAttrs()->GetAttrPointer<int64_t>(1, headDimAttr);

    uint32_t bits = static_cast<uint32_t>(bitsAttr);
    uint32_t headDim = static_cast<uint32_t>(headDimAttr);

    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platformInfo.GetCoreNumAiv();
    uint32_t usefulCore = totalTokens * numKvHeads;
    if (usefulCore > 0 && usefulCore < coreNum) {
        coreNum = usefulCore;
    }
    if (coreNum == 0) {
        coreNum = 1;
    }

    TqDequantMsePagedTilingData tiling;
    tiling.totalTokens = totalTokens;
    tiling.blockSize = blockSize;
    tiling.numKvHeads = numKvHeads;
    tiling.headDim = headDim;
    tiling.packedCols = packedCols;
    tiling.bits = bits;
    tiling.numCore = coreNum;

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

struct TqDequantMsePagedCompileInfo {};

ge::graphStatus TilingParseForTqDequantMsePaged(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TqDequantMsePaged)
    .Tiling(TqDequantMsePagedTilingFunc)
    .TilingParse<TqDequantMsePagedCompileInfo>(TilingParseForTqDequantMsePaged);

}  // namespace optiling
