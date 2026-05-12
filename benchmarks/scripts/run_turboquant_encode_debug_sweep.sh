#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run fused-only TurboQuant encode-cache diagnostic modes.

Required:
  TQ_MODEL=/path/to/turboquant_model

Optional:
  OUT_ROOT=/tmp/tq_encode_debug_sweep_YYYYmmdd_HHMMSS
  ENCODE_MODES="0 1 2 3 4 5 6 7 8 9"
  MAX_MODEL_LEN=32768
  MAX_TOKENS=8
  TP_SIZE=1
  GPU_MEMORY_UTILIZATION=0.80
  DTYPE=auto
  TQ_QUANTIZATION=ascend
  STRUCTURED_TRANSFORM=dense
  M4_GROUPED_Q=1
  M4_FORCE_FP32_INPUT=1
  ENCODE_V_PARTITIONS=1
  ENCODE_FORCE_FP32_INPUT=0
  ENCODE_STRUCTURED_FAST=0
  PROFILE_SYNC=1

Encode debug modes:
  0 full K+V encode
  1 K encode only
  2 V encode only
  3 K stage1 encode + cache write only
  4 K rotation/norm only
  5 K stage1 encode without cache write
  6 minimal slot/task overhead
  7 K load + norm only
  8 K stage1 + QJL dense project without qjl pack/store
  9 K stage1 + QJL dense project + pack without qjl store
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

OUT_ROOT="${OUT_ROOT:-/tmp/tq_encode_debug_sweep_$(date +%Y%m%d_%H%M%S)}"
ENCODE_MODES="${ENCODE_MODES:-0 1 2 3 4 5 6 7 8 9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-8}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
DTYPE="${DTYPE:-auto}"
TQ_QUANTIZATION="${TQ_QUANTIZATION:-ascend}"
STRUCTURED_TRANSFORM="${STRUCTURED_TRANSFORM:-dense}"
M4_GROUPED_Q="${M4_GROUPED_Q:-1}"
M4_FORCE_FP32_INPUT="${M4_FORCE_FP32_INPUT:-1}"
ENCODE_V_PARTITIONS="${ENCODE_V_PARTITIONS:-1}"
ENCODE_FORCE_FP32_INPUT="${ENCODE_FORCE_FP32_INPUT:-0}"
ENCODE_STRUCTURED_FAST="${ENCODE_STRUCTURED_FAST:-0}"
PROFILE_SYNC="${PROFILE_SYNC:-1}"

unset VLLM_ASCEND_TQ_USE_FUSED_KV_UPDATE_ATTENTION
unset VLLM_ASCEND_TQ_USE_CUSTOM_FUSED_KV_UPDATE_ATTENTION
unset VLLM_ASCEND_TQ_USE_CUSTOM_ENCODE_CACHE_UPDATE
unset VLLM_ASCEND_TQ_USE_FUSED_DECODE_DENSE_FIA
unset VLLM_ASCEND_TQ_USE_FUSED_DECODE_ATTENTION_M4
unset VLLM_ASCEND_TQ_USE_FUSED_PREFILL_DENSE_FIA
unset VLLM_ASCEND_TQ_ENCODE_DEBUG_MODE
unset VLLM_ASCEND_TQ_ENCODE_V_PARTITIONS
unset VLLM_ASCEND_TQ_ENCODE_FORCE_FP32_INPUT
unset VLLM_ASCEND_TQ_ENCODE_STRUCTURED_FAST
unset VLLM_ASCEND_TQ_M4_GROUPED_Q
unset VLLM_ASCEND_TQ_M4_SCORE_TILE_LEN
unset VLLM_ASCEND_TQ_M4_SPLIT_CACHE_UPDATE
unset VLLM_ASCEND_TQ_M4_PRETRANSFORM_QUERY
unset VLLM_ASCEND_TQ_M4_FORCE_FP32_INPUT
unset VLLM_ASCEND_TQ_M4_DEBUG_MODE
unset VLLM_ASCEND_TQ_PROFILE_M4_STAGES
unset VLLM_ASCEND_TQ_PROFILE_ENCODE_STAGES
unset VLLM_ASCEND_TQ_PROFILE_M4_SHADOW
unset VLLM_ASCEND_TQ_STRUCTURED_TRANSFORM
unset VLLM_ASCEND_TQ_USE_COMPRESSED_DECODE_CURRENT
unset VLLM_ASCEND_TQ_USE_COMPRESSED_DECODE_CUSTOM_K_SCORE
unset VLLM_ASCEND_TQ_USE_DECODE_COMPRESSED_FULL_CACHE

mkdir -p "${OUT_ROOT}"
echo "Repo: ${REPO_ROOT}"
echo "Git:  $(git rev-parse --short HEAD)"
echo "Out:  ${OUT_ROOT}"

for mode in ${ENCODE_MODES}; do
  out="${OUT_ROOT}/encode_mode_${mode}"
  echo
  echo "=== Running encode debug mode ${mode} -> ${out} ==="
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
    --env-fused VLLM_ASCEND_TQ_USE_FUSED_PREFILL_DENSE_FIA=1
    --env-fused VLLM_ASCEND_TQ_STRUCTURED_TRANSFORM="${STRUCTURED_TRANSFORM}"
    --env-fused VLLM_ASCEND_TQ_ENCODE_DEBUG_MODE="${mode}"
    --env-fused VLLM_ASCEND_TQ_ENCODE_V_PARTITIONS="${ENCODE_V_PARTITIONS}"
    --env-fused VLLM_ASCEND_TQ_ENCODE_FORCE_FP32_INPUT="${ENCODE_FORCE_FP32_INPUT}"
    --env-fused VLLM_ASCEND_TQ_ENCODE_STRUCTURED_FAST="${ENCODE_STRUCTURED_FAST}"
    --env-fused VLLM_ASCEND_TQ_M4_GROUPED_Q="${M4_GROUPED_Q}"
    --env-fused VLLM_ASCEND_TQ_M4_SCORE_TILE_LEN=0
    --env-fused VLLM_ASCEND_TQ_M4_SPLIT_CACHE_UPDATE=1
    --env-fused VLLM_ASCEND_TQ_M4_PRETRANSFORM_QUERY=0
    --env-fused VLLM_ASCEND_TQ_M4_FORCE_FP32_INPUT="${M4_FORCE_FP32_INPUT}"
    --env-fused VLLM_ASCEND_TQ_M4_DEBUG_MODE=6
    --env-fused VLLM_ASCEND_TQ_PROFILE_M4_STAGES=1
    --env-fused VLLM_ASCEND_TQ_PROFILE_ENCODE_STAGES=1
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
python benchmarks/scripts/analyze_turboquant_profile.py "${OUT_ROOT}" --top 20
