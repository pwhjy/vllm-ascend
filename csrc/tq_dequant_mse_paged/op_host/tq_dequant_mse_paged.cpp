/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Host-side OpDef and tiling logic for the TurboQuant MSE paged dequant op.
 */

#include <cstdint>
#include "register/op_def_registry.h"
#include "tq_dequant_mse_paged_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace ops {

class TqDequantMsePaged : public OpDef {
public:
    explicit TqDequantMsePaged(const char* name) : OpDef(name)
    {
        this->Input("packedIdx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("norm")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("tokenBlockIds")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("tokenOffsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("codebook")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("denseRot")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Attr("bits").Int();
        this->Attr("headDim").Int();

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");

        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};

OP_ADD(TqDequantMsePaged);

}  // namespace ops


// ---------------------------------------------------------------------------
// Tiling function
// ---------------------------------------------------------------------------

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
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

}  // namespace optiling
