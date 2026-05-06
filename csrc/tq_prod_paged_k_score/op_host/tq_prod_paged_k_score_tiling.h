/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#pragma once
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TqProdPagedKScoreTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, numHeads);
    TILING_DATA_FIELD_DEF(uint32_t, numKvHeads);
    TILING_DATA_FIELD_DEF(uint32_t, qPerKv);
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);
    TILING_DATA_FIELD_DEF(uint32_t, maxBlocksPerSeq);
    TILING_DATA_FIELD_DEF(uint32_t, maxSeqLen);
    TILING_DATA_FIELD_DEF(uint32_t, headDim);
    TILING_DATA_FIELD_DEF(uint32_t, packedCols);
    TILING_DATA_FIELD_DEF(uint32_t, qjlCols);
    TILING_DATA_FIELD_DEF(uint32_t, stage1Bits);
    TILING_DATA_FIELD_DEF(uint32_t, numCore);
    TILING_DATA_FIELD_DEF(float, scale);
    TILING_DATA_FIELD_DEF(float, correction);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TqProdPagedKScore, TqProdPagedKScoreTilingData)

}  // namespace optiling

struct TqProdPagedKScoreCompileInfo {};
