/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License").
 */

#pragma once

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TqEncodeKvToPagedCacheTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalTokens);
    TILING_DATA_FIELD_DEF(uint32_t, numKvHeads);
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);
    TILING_DATA_FIELD_DEF(uint32_t, kPackedCols);
    TILING_DATA_FIELD_DEF(uint32_t, kQjlCols);
    TILING_DATA_FIELD_DEF(uint32_t, vPackedCols);
    TILING_DATA_FIELD_DEF(uint32_t, totalBits);
    TILING_DATA_FIELD_DEF(uint32_t, stage1Bits);
    TILING_DATA_FIELD_DEF(uint32_t, vBits);
    TILING_DATA_FIELD_DEF(uint32_t, headDim);
    TILING_DATA_FIELD_DEF(uint32_t, debugMode);
    TILING_DATA_FIELD_DEF(uint32_t, vPartitionCount);
    TILING_DATA_FIELD_DEF(uint32_t, numCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TqEncodeKvToPagedCache, TqEncodeKvToPagedCacheTilingData)

}  // namespace optiling

struct TqEncodeKvToPagedCacheCompileInfo {};
