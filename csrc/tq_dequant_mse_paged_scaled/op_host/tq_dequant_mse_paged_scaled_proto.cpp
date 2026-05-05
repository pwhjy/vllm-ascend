/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeTqDequantMsePagedScaled(gert::InferShapeContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeTqDequantMsePagedScaled(gert::InferDataTypeContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TqDequantMsePagedScaled)
    .InferShape(InferShapeTqDequantMsePagedScaled)
    .InferDataType(InferDataTypeTqDequantMsePagedScaled);

}  // namespace ops
