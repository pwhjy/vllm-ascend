/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

#include "register/op_def_registry.h"

namespace ops {

class TqEncodeKvToPagedCache : public OpDef {
public:
    explicit TqEncodeKvToPagedCache(const char* name) : OpDef(name)
    {
        this->Input("key").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("value").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("slotMapping").ParamType(REQUIRED).DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("kIdxCache").ParamType(REQUIRED).DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND});
        this->Input("kQjlCache").ParamType(REQUIRED).DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND});
        this->Input("kGammaCache").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("kNormCache").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("vIdxCache").ParamType(REQUIRED).DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_ND});
        this->Input("vNormCache").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("kRotation").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("kBoundary").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("kCodebook").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("kQjlProjT").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("vRotation").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Input("vBoundary").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).AutoContiguous();
        this->Attr("totalBits").AttrType(REQUIRED).Int(0);
        this->Attr("stage1Bits").AttrType(REQUIRED).Int(0);
        this->Attr("vBits").AttrType(REQUIRED).Int(0);
        this->Attr("headDim").AttrType(REQUIRED).Int(0);
        this->Attr("debugMode").AttrType(REQUIRED).Int(0);
        this->Attr("vPartitionCount").AttrType(REQUIRED).Int(4);

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

OP_ADD(TqEncodeKvToPagedCache);

}  // namespace ops
