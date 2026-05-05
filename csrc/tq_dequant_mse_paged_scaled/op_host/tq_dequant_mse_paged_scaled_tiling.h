/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#pragma once
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TqDequantMsePagedScaledTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalTokens);
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);
    TILING_DATA_FIELD_DEF(uint32_t, numKvHeads);
    TILING_DATA_FIELD_DEF(uint32_t, headDim);
    TILING_DATA_FIELD_DEF(uint32_t, packedCols);
    TILING_DATA_FIELD_DEF(uint32_t, bits);
    TILING_DATA_FIELD_DEF(uint32_t, numCore);
    TILING_DATA_FIELD_DEF(float, scaleMultiplier);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TqDequantMsePagedScaled, TqDequantMsePagedScaledTilingData)

}  // namespace optiling

struct TqDequantMsePagedScaledCompileInfo {};
