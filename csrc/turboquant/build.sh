#!/usr/bin/env bash
#
# Build the TurboQuant Ascend C dequant op for CANN 8.5.1 / 910B (dav-m300).
#
# This script compiles the Ascend C kernel (device code) and the pybind11
# binding (host code) separately, then links them into a Python-importable
# shared library.
#
# Prerequisites:
#   source /path/to/Ascend/cann-8.5.1/set_env.sh  (or ascend-toolkit/set_env.sh)
#   python -c "import torch, torch_npu, pybind11"
#
# Usage:
#   cd csrc/turboquant
#   bash build.sh
#   cp build/libvllm_ascend_tq_ops.so ../../  # to repo root
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p "${BUILD_DIR}"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log()  { echo -e "${GREEN}[build]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# -----------------------------------------------------------------------
# 1. Locate CANN paths and bisheng compiler
# -----------------------------------------------------------------------
CANN_BASE="${ASCEND_HOME_PATH:-${ASCEND_CANN_PACKAGE_PATH:-}}"
if [ -z "${CANN_BASE}" ]; then
    # Try to auto-detect from the bisheng in PATH
    BISHENG_IN_PATH=$(command -v bisheng 2>/dev/null || true)
    if [ -n "${BISHENG_IN_PATH}" ]; then
        CANN_BASE=$(dirname "$(dirname "${BISHENG_IN_PATH}")")
    fi
fi
[ -z "${CANN_BASE}" ] && err "Cannot find CANN installation. source set_env.sh first."

log "CANN base: ${CANN_BASE}"

# bisheng: try bin/bisheng first (wrapper), then ccec_compiler/bin/bisheng
BISHENG="${CANN_BASE}/bin/bisheng"
if [ ! -x "${BISHENG}" ]; then
    ARCH_DIR="${CANN_BASE}/aarch64-linux"
    BISHENG="${ARCH_DIR}/ccec_compiler/bin/bisheng"
fi
[ -x "${BISHENG}" ] || err "bisheng not found at ${BISHENG}"
log "bisheng: ${BISHENG}"

# Ascend C devkit (arch-specific)
if [ -d "${CANN_BASE}/aarch64-linux" ]; then
    ASCENDC_DEVKIT="${CANN_BASE}/aarch64-linux"
elif [ -d "${CANN_BASE}/x86_64-linux" ]; then
    ASCENDC_DEVKIT="${CANN_BASE}/x86_64-linux"
else
    ASCENDC_DEVKIT="${CANN_BASE}/tools"
fi
log "AscendC devkit: ${ASCENDC_DEVKIT}"

# -----------------------------------------------------------------------
# 2. Python / torch / pybind11 / torch_npu include & lib paths
# -----------------------------------------------------------------------
PYTHON_EXE="${PYTHON_EXE:-python}"

TORCH_INCLUDES=$(${PYTHON_EXE} -c "
import torch; from torch.utils.cpp_extension import include_paths
print(';'.join(include_paths()))
")
PYBIND11_INCLUDE=$(${PYTHON_EXE} -c "import pybind11; print(pybind11.get_include())")
PYTHON_INCLUDE=$(${PYTHON_EXE} -c "import sysconfig; print(sysconfig.get_path('include'))")
TORCH_NPU_ROOT=$(${PYTHON_EXE} -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))")
TORCH_NPU_INCLUDE="${TORCH_NPU_ROOT}/include"

TORCH_LIB_DIR=$(${PYTHON_EXE} -c "import torch; print(torch.utils.cmake_prefix_path + '/lib')")
TORCH_NPU_LIB_DIR="${TORCH_NPU_ROOT}/lib"

log "torch_npu root: ${TORCH_NPU_ROOT}"

# -----------------------------------------------------------------------
# 3. Include paths
# -----------------------------------------------------------------------
INCLUDES=(
    # Ascend C device headers (from bisheng_intf.cmake)
    "${ASCENDC_DEVKIT}/asc/include"
    "${ASCENDC_DEVKIT}/asc/include/adv_api"
    "${ASCENDC_DEVKIT}/asc/include/basic_api"
    "${ASCENDC_DEVKIT}/asc/include/utils"
    "${ASCENDC_DEVKIT}/asc/impl/adv_api"
    "${ASCENDC_DEVKIT}/asc/impl/basic_api"
    "${ASCENDC_DEVKIT}/asc/impl/utils"
    "${ASCENDC_DEVKIT}/tikcpp/tikcfw"
    "${ASCENDC_DEVKIT}/tikcpp/tikcfw/interface"
    "${ASCENDC_DEVKIT}/tikcpp/tikcfw/impl"
    # CANN top-level
    "${CANN_BASE}/include"
    # torch / python / pybind11
    ${PYBIND11_INCLUDE}
    ${PYTHON_INCLUDE}
    ${TORCH_NPU_INCLUDE}
)

INCLUDE_FLAGS=""
for inc in "${INCLUDES[@]}"; do
    if [ -d "${inc}" ]; then
        INCLUDE_FLAGS="${INCLUDE_FLAGS} -I${inc}"
    fi
done

# torch includes (semicolon-separated)
IFS=';' read -ra TORCH_INC_ARRAY <<< "${TORCH_INCLUDES}"
for inc in "${TORCH_INC_ARRAY[@]}"; do
    if [ -d "${inc}" ]; then
        INCLUDE_FLAGS="${INCLUDE_FLAGS} -I${inc}"
    fi
done

# -----------------------------------------------------------------------
# 4. Compile Ascend C kernel (device code)
#    Architecture: dav-m300 (910B). Flags from m300_intf_pub in
#    bisheng_intf.cmake.
# -----------------------------------------------------------------------
log "Compiling Ascend C kernel (device)..."

KERNEL_SRC="${SCRIPT_DIR}/tq_dequant_mse_paged_kernel.cpp"
KERNEL_OBJ="${BUILD_DIR}/tq_dequant_mse_paged_kernel.o"

DEVICE_FLAGS=(
    -std=c++17
    -O3
    -fPIC
    --cce-aicore-arch=dav-m300
    --cce-aicore-lang
    --cce-aicore-only
    --cce-auto-sync
    --cce-mask-opt
    --cce-disable-kernel-global-attr-check
    -mllvm -cce-aicore-function-stack-size=16000
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-or-combine=false
    -mllvm -instcombine-code-sinking=false
    -mllvm -cce-aicore-jump-expand=false
    -mllvm -cce-aicore-mask-opt=false
)

"${BISHENG}" "${DEVICE_FLAGS[@]}" \
    ${INCLUDE_FLAGS} \
    -c "${KERNEL_SRC}" \
    -o "${KERNEL_OBJ}"

log "Kernel .o: ${KERNEL_OBJ}"

# -----------------------------------------------------------------------
# 5. Compile pybind11 binding (host code)
#    Flags from host_project/CMakeLists.txt + bisheng_intf.cmake.
# -----------------------------------------------------------------------
log "Compiling host binding..."

BINDING_SRC="${SCRIPT_DIR}/tq_dequant_mse_paged_binding.cpp"
BINDING_OBJ="${BUILD_DIR}/tq_dequant_mse_paged_binding.o"

HOST_FLAGS=(
    -std=c++17
    -O3
    -fPIC
    --cce-host-only
    -fcce-kernel-launch-custom
    -D__ASC_NPU_HOST__
)

# The <<<>>> launch syntax needs a helper header from CANN
TRIPLE_CHEVRON_H="${ASCENDC_DEVKIT}/../include/aclrtlaunch_triple_chevrons_func.h"
if [ -f "${TRIPLE_CHEVRON_H}" ]; then
    HOST_FLAGS+=("-include" "${TRIPLE_CHEVRON_H}")
fi

"${BISHENG}" "${HOST_FLAGS[@]}" \
    ${INCLUDE_FLAGS} \
    -c "${BINDING_SRC}" \
    -o "${BINDING_OBJ}"

log "Binding .o: ${BINDING_OBJ}"

# -----------------------------------------------------------------------
# 6. Link shared library
# -----------------------------------------------------------------------
log "Linking..."

OUTPUT_SO="${BUILD_DIR}/libvllm_ascend_tq_ops.so"

"${BISHENG}" \
    -shared \
    -fPIC \
    -o "${OUTPUT_SO}" \
    "${KERNEL_OBJ}" "${BINDING_OBJ}" \
    -L"${TORCH_LIB_DIR}" -ltorch -ltorch_cpu -lc10 \
    -L"${TORCH_NPU_LIB_DIR}" -ltorch_npu \
    -Wl,-rpath,"${TORCH_LIB_DIR}" \
    -Wl,-rpath,"${TORCH_NPU_LIB_DIR}"

echo ""
log "Build complete: ${OUTPUT_SO}"
echo ""
echo "  Copy to repo root and test:"
echo "    cp ${OUTPUT_SO} ${SCRIPT_DIR}/../../"
echo "    export VLLM_ASCEND_TQ_USE_CUSTOM_DEQUANT=1"
echo "    export VLLM_ASCEND_TQ_DEBUG_COMPARE=1"
echo "    python -c 'import vllm_ascend_tq_ops; print(dir(vllm_ascend_tq_ops))'"
