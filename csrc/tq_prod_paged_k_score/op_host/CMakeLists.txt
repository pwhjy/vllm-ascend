# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the vllm-ascend project.
# Licensed under the Apache License, Version 2.0 (the "License").
# ========================================================================

add_ops_compile_options(
        OP_NAME TqProdPagedKScore
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
)

target_sources(op_host_aclnn PRIVATE
        tq_prod_paged_k_score_def.cpp
)

target_sources(optiling PRIVATE
        tq_prod_paged_k_score_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
            tq_prod_paged_k_score_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_sources(opsproto PRIVATE
        tq_prod_paged_k_score_proto.cpp
)
