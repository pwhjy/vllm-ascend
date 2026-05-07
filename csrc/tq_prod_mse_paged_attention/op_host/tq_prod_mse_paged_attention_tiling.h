/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#pragma once
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TqProdMsePagedAttentionTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, numHeads);
    TILING_DATA_FIELD_DEF(uint32_t, numKvHeads);
    TILING_DATA_FIELD_DEF(uint32_t, qPerKv);
    TILING_DATA_FIELD_DEF(uint32_t, blockSize);
    TILING_DATA_FIELD_DEF(uint32_t, maxBlocksPerSeq);
    TILING_DATA_FIELD_DEF(uint32_t, maxSeqLen);
    TILING_DATA_FIELD_DEF(uint32_t, headDim);
    TILING_DATA_FIELD_DEF(uint32_t, kPackedCols);
    TILING_DATA_FIELD_DEF(uint32_t, kQjlCols);
    TILING_DATA_FIELD_DEF(uint32_t, vPackedCols);
    TILING_DATA_FIELD_DEF(uint32_t, kStage1Bits);
    TILING_DATA_FIELD_DEF(uint32_t, vBits);
    TILING_DATA_FIELD_DEF(uint32_t, scoreTileLen);
    TILING_DATA_FIELD_DEF(uint32_t, numCore);
    TILING_DATA_FIELD_DEF(float, scale);
    TILING_DATA_FIELD_DEF(float, correction);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(
    TqProdMsePagedAttention, TqProdMsePagedAttentionTilingData)

}  // namespace optiling

struct TqProdMsePagedAttentionCompileInfo {};
