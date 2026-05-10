/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

#include "register/op_def_registry.h"

namespace ops {

class TqFusedKvUpdateAttentionDecode : public OpDef {
public:
    explicit TqFusedKvUpdateAttentionDecode(const char* name) : OpDef(name)
    {
        this->Input("query").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("key").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("value").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("slotMapping").ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("kPackedIdx").ParamType(REQUIRED).DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("kPackedQjl").ParamType(REQUIRED).DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("kGamma").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("kNorm").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("vPackedIdx").ParamType(REQUIRED).DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("vNorm").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("blockTable").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("oldSeqLens").ParamType(REQUIRED).DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("kRotation").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("kQjlQueryMatrix").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("kQjlProjT").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("kBoundary").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("vRotation").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("vRotationT").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("vBoundary").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("kCodebook").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("vCodebook").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Attr("kTotalBits").AttrType(REQUIRED).Int(0);
        this->Attr("vBits").AttrType(REQUIRED).Int(0);
        this->Attr("headDim").AttrType(REQUIRED).Int(0);
        this->Attr("scale").AttrType(REQUIRED).Float(1.0);
        this->Attr("maxSeqLen").AttrType(REQUIRED).Int(0);
        this->Attr("scoreTileLen").AttrType(REQUIRED).Int(0);
        this->Attr("groupedQ").AttrType(REQUIRED).Int(1);
        this->Attr("skipCacheUpdate").AttrType(REQUIRED).Int(0);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
    }
};

OP_ADD(TqFusedKvUpdateAttentionDecode);

}  // namespace ops
