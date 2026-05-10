#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run fused-only TurboQuant M4 diagnostic modes and print profile breakdowns.

Required:
  TQ_MODEL=/path/to/turboquant_model

Optional:
  OUT_ROOT=/tmp/tq_m4_debug_sweep_YYYYmmdd_HHMMSS
  MODES="0 1 2 3 4 5"
  MAX_MODEL_LEN=32768
  MAX_TOKENS=8
  TP_SIZE=1
  GPU_MEMORY_UTILIZATION=0.80
  DTYPE=auto
  TQ_QUANTIZATION=ascend
  M4_SPLIT_CACHE_UPDATE=1
  PROFILE_SYNC=1

Debug modes:
  0 full path
  1 current-only + StoreOutput
  2 history score only
  3 history score + online softmax only
  4 history score + softmax + V accumulate, no StoreOutput/current
  5 full history + current, no StoreOutput
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${TQ_MODEL:-}" ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUT_ROOT="${OUT_ROOT:-/tmp/tq_m4_debug_sweep_$(date +%Y%m%d_%H%M%S)}"
MODES="${MODES:-0 1 2 3 4 5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-8}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
DTYPE="${DTYPE:-auto}"
TQ_QUANTIZATION="${TQ_QUANTIZATION:-ascend}"
M4_SPLIT_CACHE_UPDATE="${M4_SPLIT_CACHE_UPDATE:-1}"
PROFILE_SYNC="${PROFILE_SYNC:-1}"

unset VLLM_ASCEND_TQ_USE_FUSED_KV_UPDATE_ATTENTION
unset VLLM_ASCEND_TQ_USE_CUSTOM_FUSED_KV_UPDATE_ATTENTION
unset VLLM_ASCEND_TQ_USE_CUSTOM_ENCODE_CACHE_UPDATE
unset VLLM_ASCEND_TQ_USE_FUSED_DECODE_DENSE_FIA
unset VLLM_ASCEND_TQ_USE_FUSED_DECODE_ATTENTION_M4
unset VLLM_ASCEND_TQ_M4_GROUPED_Q
unset VLLM_ASCEND_TQ_M4_SCORE_TILE_LEN
unset VLLM_ASCEND_TQ_M4_SPLIT_CACHE_UPDATE
unset VLLM_ASCEND_TQ_M4_DEBUG_MODE
unset VLLM_ASCEND_TQ_PROFILE_M4_STAGES
unset VLLM_ASCEND_TQ_PROFILE_M4_SHADOW
unset VLLM_ASCEND_TQ_USE_COMPRESSED_DECODE_CURRENT
unset VLLM_ASCEND_TQ_USE_COMPRESSED_DECODE_CUSTOM_K_SCORE
unset VLLM_ASCEND_TQ_USE_DECODE_COMPRESSED_FULL_CACHE

mkdir -p "${OUT_ROOT}"
echo "Repo: ${REPO_ROOT}"
echo "Git:  $(git rev-parse --short HEAD)"
echo "Out:  ${OUT_ROOT}"

for mode in ${MODES}; do
  out="${OUT_ROOT}/mode_${mode}"
  echo
  echo "=== Running M4 debug mode ${mode} -> ${out} ==="
  cmd=(
    python benchmarks/scripts/check_turboquant_llama_correctness.py
    --fused-only
    --model "${TQ_MODEL}"
    --quantization "${TQ_QUANTIZATION}"
    --dtype "${DTYPE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-tokens "${MAX_TOKENS}"
    --tensor-parallel-size "${TP_SIZE}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --baseline-mode custom
    --profile-turboquant
    --output-dir "${out}"
    --env-fused VLLM_ASCEND_TQ_USE_FUSED_KV_UPDATE_ATTENTION=1
    --env-fused VLLM_ASCEND_TQ_USE_CUSTOM_FUSED_KV_UPDATE_ATTENTION=0
    --env-fused VLLM_ASCEND_TQ_USE_CUSTOM_ENCODE_CACHE_UPDATE=1
    --env-fused VLLM_ASCEND_TQ_USE_FUSED_DECODE_DENSE_FIA=1
    --env-fused VLLM_ASCEND_TQ_USE_FUSED_DECODE_ATTENTION_M4=1
    --env-fused VLLM_ASCEND_TQ_M4_GROUPED_Q=0
    --env-fused VLLM_ASCEND_TQ_M4_SCORE_TILE_LEN=0
    --env-fused VLLM_ASCEND_TQ_M4_SPLIT_CACHE_UPDATE="${M4_SPLIT_CACHE_UPDATE}"
    --env-fused VLLM_ASCEND_TQ_M4_DEBUG_MODE="${mode}"
    --env-fused VLLM_ASCEND_TQ_PROFILE_M4_STAGES=1
    --env-fused VLLM_ASCEND_TQ_PROFILE_M4_SHADOW=0
    --env-fused VLLM_ASCEND_TQ_USE_COMPRESSED_DECODE_CURRENT=0
    --env-fused VLLM_ASCEND_TQ_USE_COMPRESSED_DECODE_CUSTOM_K_SCORE=0
    --env-fused VLLM_ASCEND_TQ_USE_DECODE_COMPRESSED_FULL_CACHE=0
  )
  if [[ "${PROFILE_SYNC}" == "1" ]]; then
    cmd+=(--profile-turboquant-sync)
  fi
  cmd+=("$@")
  printf 'Cmd: '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
  python benchmarks/scripts/analyze_turboquant_profile.py "${out}"
done

echo
echo "Sweep results: ${OUT_ROOT}"
