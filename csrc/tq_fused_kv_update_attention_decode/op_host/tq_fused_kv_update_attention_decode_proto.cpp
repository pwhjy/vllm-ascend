/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

namespace ops {

static ge::graphStatus InferShapeTqFusedKvUpdateAttentionDecode(
    gert::InferShapeContext* context)
{
    const gert::Shape* queryShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    *outShape = *queryShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeTqFusedKvUpdateAttentionDecode(
    gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TqFusedKvUpdateAttentionDecode)
    .InferShape(InferShapeTqFusedKvUpdateAttentionDecode)
    .InferDataType(InferDataTypeTqFusedKvUpdateAttentionDecode);

}  // namespace ops
