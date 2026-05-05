/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include "register/op_def_registry.h"

namespace ops {

class TqDequantMsePaged : public OpDef {
public:
    explicit TqDequantMsePaged(const char* name) : OpDef(name)
    {
        this->Input("packedIdx")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("norm")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("tokenBlockIds")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("tokenOffsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("codebook")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("denseRot")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Attr("bits").AttrType(REQUIRED).Int(0);
        this->Attr("headDim").AttrType(REQUIRED).Int(0);

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
