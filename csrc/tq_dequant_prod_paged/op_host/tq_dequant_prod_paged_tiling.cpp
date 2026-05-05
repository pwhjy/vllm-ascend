/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "tq_dequant_prod_paged_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdint>

namespace optiling {

static ge::graphStatus TqDequantProdPagedTilingFunc(gert::TilingContext* context)
{
    auto packedIdxShape = context->GetInputShape(0)->GetStorageShape();
    auto packedQjlShape = context->GetInputShape(1)->GetStorageShape();
    auto tokenBlockIdsShape = context->GetInputShape(4)->GetStorageShape();

    uint32_t totalTokens = static_cast<uint32_t>(tokenBlockIdsShape.GetDim(0));
    uint32_t blockSize = static_cast<uint32_t>(packedIdxShape.GetDim(1));
    uint32_t numKvHeads = static_cast<uint32_t>(packedIdxShape.GetDim(2));
    uint32_t packedIdxCols = static_cast<uint32_t>(packedIdxShape.GetDim(3));
    uint32_t qjlCols = static_cast<uint32_t>(packedQjlShape.GetDim(3));

    auto attr = context->GetAttrs();
    const int64_t* totalBitsPtr = attr->GetAttrPointer<int64_t>(0);
    const int64_t* headDimPtr = attr->GetAttrPointer<int64_t>(1);

    uint32_t totalBits = static_cast<uint32_t>(totalBitsPtr != nullptr ? *totalBitsPtr : 0);
    uint32_t stage1Bits = totalBits > 0 ? totalBits - 1 : 0;
    uint32_t headDim = static_cast<uint32_t>(headDimPtr != nullptr ? *headDimPtr : 0);
    float qjlCorrection = headDim > 0 ?
        1.2533141373155003F / static_cast<float>(headDim) : 0.0F;

    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platformInfo.GetCoreNumAiv();
    uint32_t usefulCore = totalTokens * numKvHeads;
    if (usefulCore > 0 && usefulCore < coreNum) {
        coreNum = usefulCore;
    }
    if (coreNum == 0) {
        coreNum = 1;
    }

    TqDequantProdPagedTilingData tiling;
    tiling.set_totalTokens(totalTokens);
    tiling.set_blockSize(blockSize);
    tiling.set_numKvHeads(numKvHeads);
    tiling.set_headDim(headDim);
    tiling.set_packedIdxCols(packedIdxCols);
    tiling.set_qjlCols(qjlCols);
    tiling.set_totalBits(totalBits);
    tiling.set_stage1Bits(stage1Bits);
    tiling.set_qjlCorrection(qjlCorrection);
    tiling.set_numCore(coreNum);

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

struct TqDequantProdPagedCompileInfo {};

ge::graphStatus TilingParseForTqDequantProdPaged(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TqDequantProdPaged)
    .Tiling(TqDequantProdPagedTilingFunc)
    .TilingParse<TqDequantProdPagedCompileInfo>(TilingParseForTqDequantProdPaged);

}  // namespace optiling
