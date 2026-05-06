/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "tq_dequant_mse_paged_scaled_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstdint>

namespace optiling {

static ge::graphStatus TqDequantMsePagedScaledTilingFunc(gert::TilingContext* context)
{
    auto packedIdxShape = context->GetInputShape(0)->GetStorageShape();
    auto tokenBlockIdsShape = context->GetInputShape(3)->GetStorageShape();

    uint32_t totalTokens = static_cast<uint32_t>(tokenBlockIdsShape.GetDim(0));
    uint32_t blockSize = static_cast<uint32_t>(packedIdxShape.GetDim(1));
    uint32_t numKvHeads = static_cast<uint32_t>(packedIdxShape.GetDim(2));
    uint32_t packedCols = static_cast<uint32_t>(packedIdxShape.GetDim(3));

    auto attr = context->GetAttrs();
    const int64_t* bitsPtr = attr->GetAttrPointer<int64_t>(0);
    const int64_t* headDimPtr = attr->GetAttrPointer<int64_t>(1);
    const float* scaleMultiplierPtr = attr->GetAttrPointer<float>(2);
    const int64_t* signedBits1Ptr = attr->GetAttrPointer<int64_t>(3);

    uint32_t bits = static_cast<uint32_t>(bitsPtr != nullptr ? *bitsPtr : 0);
    uint32_t headDim = static_cast<uint32_t>(headDimPtr != nullptr ? *headDimPtr : 0);
    float scaleMultiplier =
        scaleMultiplierPtr != nullptr ? *scaleMultiplierPtr : 1.0F;
    uint32_t signedBits1 = static_cast<uint32_t>(
        signedBits1Ptr != nullptr ? *signedBits1Ptr : 0);

    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platformInfo.GetCoreNumAiv();
    uint32_t usefulCore = totalTokens * numKvHeads;
    if (usefulCore > 0 && usefulCore < coreNum) {
        coreNum = usefulCore;
    }
    if (coreNum == 0) {
        coreNum = 1;
    }

    TqDequantMsePagedScaledTilingData tiling;
    tiling.set_totalTokens(totalTokens);
    tiling.set_blockSize(blockSize);
    tiling.set_numKvHeads(numKvHeads);
    tiling.set_headDim(headDim);
    tiling.set_packedCols(packedCols);
    tiling.set_bits(bits);
    tiling.set_numCore(coreNum);
    tiling.set_signedBits1(signedBits1);
    tiling.set_scaleMultiplier(scaleMultiplier);

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingParseForTqDequantMsePagedScaled(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TqDequantMsePagedScaled)
    .Tiling(TqDequantMsePagedScaledTilingFunc)
    .TilingParse<TqDequantMsePagedScaledCompileInfo>(
        TilingParseForTqDequantMsePagedScaled);

}  // namespace optiling
