# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Build configuration for the TurboQuant Ascend C dequant op.
#
# Quick-start:
#   cd csrc/turboquant
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#   cp build/vllm_ascend_tq_ops*.so /path/to/vllm-ascend/
#
# The CANN toolkit must be sourced before building:
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh

cmake_minimum_required(VERSION 3.20)
project(vllm_ascend_tq_ops LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -------------------------------------------------------------------------
# CANN path — override with -DASCEND_CANN_PACKAGE_PATH=... if needed.
# -------------------------------------------------------------------------
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
    else()
        set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/ascend-toolkit/latest)
    endif()
endif()

message(STATUS "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")

# -------------------------------------------------------------------------
# Python / torch / pybind11 / torch_npu
# -------------------------------------------------------------------------
find_package(Python REQUIRED COMPONENTS Interpreter Development)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "TORCH_NPU_ROOT=${TORCH_NPU_ROOT}")

# -------------------------------------------------------------------------
# Include directories
# -------------------------------------------------------------------------
include_directories(
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/tikcfw
    ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/tikcfw/impl
    ${TORCH_NPU_ROOT}/include
)

# -------------------------------------------------------------------------
# Target
# -------------------------------------------------------------------------
pybind11_add_module(
    vllm_ascend_tq_ops
    tq_dequant_mse_paged_binding.cpp
    tq_dequant_mse_paged_kernel.cpp
)

target_link_libraries(
    vllm_ascend_tq_ops
    PRIVATE
    ${TORCH_LIBRARIES}
)

target_compile_options(vllm_ascend_tq_ops PRIVATE -O3)
