/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

namespace ops {

static ge::graphStatus InferShapeTqEncodeMsePagedCache(gert::InferShapeContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeTqEncodeMsePagedCache(gert::InferDataTypeContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TqEncodeMsePagedCache)
    .InferShape(InferShapeTqEncodeMsePagedCache)
    .InferDataType(InferDataTypeTqEncodeMsePagedCache);

}  // namespace ops
