#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the current best TurboQuant path against a non-TurboQuant/plain model.

Required:
  TQ_MODEL=/path/to/turboquant_model
  PLAIN_MODEL=/path/to/plain_dense_model

Optional:
  TOKENIZER=/path/to/tokenizer
  PLAIN_TOKENIZER=/path/to/plain_tokenizer
  OUT=/tmp/tq_vs_plain_best_YYYYmmdd_HHMMSS
  MAX_MODEL_LEN=32768
  MAX_TOKENS=32
  TP_SIZE=1
  GPU_MEMORY_UTILIZATION=0.80
  DTYPE=auto
  TQ_QUANTIZATION=ascend
  PLAIN_QUANTIZATION=none
  PROFILE_TQ=1
  PROFILE_SYNC=1
  PROFILE_M4_STAGES=1
  ALLOW_MISMATCH=1

Extra arguments are forwarded to check_turboquant_llama_correctness.py.
Example:
  TQ_MODEL=/models/llama-tq PLAIN_MODEL=/models/llama-dense \
    bash benchmarks/scripts/run_turboquant_vs_plain_best.sh \
    --prompt "Explain KV cache in one paragraph."
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "${TQ_MODEL:-}" || -z "${PLAIN_MODEL:-}" ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUT="${OUT:-/tmp/tq_vs_plain_best_$(date +%Y%m%d_%H%M%S)}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_TOKENS="${MAX_TOKENS:-32}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
DTYPE="${DTYPE:-auto}"
TQ_QUANTIZATION="${TQ_QUANTIZATION:-ascend}"
PLAIN_QUANTIZATION="${PLAIN_QUANTIZATION:-none}"
PROFILE_TQ="${PROFILE_TQ:-1}"
PROFILE_SYNC="${PROFILE_SYNC:-1}"
PROFILE_M4_STAGES="${PROFILE_M4_STAGES:-1}"
ALLOW_MISMATCH="${ALLOW_MISMATCH:-1}"

cmd=(
  python benchmarks/scripts/check_turboquant_llama_correctness.py
  --model "${TQ_MODEL}"
  --quantization "${TQ_QUANTIZATION}"
  --plain-model "${PLAIN_MODEL}"
  --plain-quantization "${PLAIN_QUANTIZATION}"
  --include-plain-baseline
  --dtype "${DTYPE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-tokens "${MAX_TOKENS}"
  --tensor-parallel-size "${TP_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --baseline-mode custom
  --output-dir "${OUT}"
  --env-fused VLLM_ASCEND_TQ_USE_FUSED_KV_UPDATE_ATTENTION=1
  --env-fused VLLM_ASCEND_TQ_USE_CUSTOM_FUSED_KV_UPDATE_ATTENTION=0
  --env-fused VLLM_ASCEND_TQ_USE_CUSTOM_ENCODE_CACHE_UPDATE=1
  --env-fused VLLM_ASCEND_TQ_USE_FUSED_DECODE_DENSE_FIA=1
  --env-fused VLLM_ASCEND_TQ_USE_FUSED_DECODE_ATTENTION_M4=1
  --env-fused VLLM_ASCEND_TQ_M4_GROUPED_Q=0
  --env-fused VLLM_ASCEND_TQ_M4_SCORE_TILE_LEN=0
  --env-fused VLLM_ASCEND_TQ_PROFILE_M4_STAGES="${PROFILE_M4_STAGES}"
  --env-fused VLLM_ASCEND_TQ_PROFILE_M4_SHADOW=0
  --env-fused VLLM_ASCEND_TQ_USE_COMPRESSED_DECODE_CURRENT=0
  --env-fused VLLM_ASCEND_TQ_USE_COMPRESSED_DECODE_CUSTOM_K_SCORE=0
  --env-fused VLLM_ASCEND_TQ_USE_DECODE_COMPRESSED_FULL_CACHE=0
)

if [[ -n "${TOKENIZER:-}" ]]; then
  cmd+=(--tokenizer "${TOKENIZER}")
fi
plain_tokenizer="${PLAIN_TOKENIZER:-${TOKENIZER:-}}"
if [[ -n "${plain_tokenizer}" ]]; then
  cmd+=(--plain-tokenizer "${plain_tokenizer}")
fi
if [[ "${PROFILE_TQ}" == "1" ]]; then
  cmd+=(--profile-turboquant)
  if [[ "${PROFILE_SYNC}" == "1" ]]; then
    cmd+=(--profile-turboquant-sync)
  fi
fi
cmd+=("$@")

echo "Repo: ${REPO_ROOT}"
echo "Git:  $(git rev-parse --short HEAD)"
echo "Out:  ${OUT}"
printf 'Cmd: '
printf '%q ' "${cmd[@]}"
printf '\n\n'

set +e
"${cmd[@]}"
driver_rc=$?
set -e
if [[ "${driver_rc}" -ne 0 ]]; then
  echo
  echo "Benchmark driver exited with code ${driver_rc}."
  if [[ "${ALLOW_MISMATCH}" == "1" ]]; then
    echo "ALLOW_MISMATCH=1, continuing to print performance summary."
  else
    exit "${driver_rc}"
  fi
fi

OUT_DIR="${OUT}" python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUT_DIR"])
run_stats = json.loads((root / "run_stats.json").read_text())
workers = run_stats.get("workers", {})
children = run_stats.get("child_processes", {})

print("\n=== TurboQuant vs plain summary ===")
for name in ("plain", "baseline", "fused"):
    stats = workers.get(name, {})
    timing = stats.get("timing", {})
    throughput = stats.get("throughput", {})
    child = children.get(name, {})
    log_stats = child.get("log_stats", {})
    kv_cache = stats.get("kv_cache", {})
    kv_tokens = (
        log_stats.get("kv_cache_capacity_tokens")
        or kv_cache.get("capacity_tokens")
        or kv_cache.get("capacity_tokens_estimate")
    )
    print(
        f"{name:8s} "
        f"init={timing.get('init_seconds', 0.0):8.3f}s "
        f"generate={timing.get('generate_seconds', 0.0):8.3f}s "
        f"out_tok/s={throughput.get('output_tokens_per_second', 0.0):8.3f} "
        f"kv_tokens={kv_tokens}"
    )

plain_tps = workers.get("plain", {}).get("throughput", {}).get(
    "output_tokens_per_second"
)
fused_tps = workers.get("fused", {}).get("throughput", {}).get(
    "output_tokens_per_second"
)
if plain_tps and fused_tps:
    print(f"fused/plain output tok/s ratio: {fused_tps / plain_tps:.4f}x")

profile = children.get("fused", {}).get("turboquant_profile", {})
stats = profile.get("stats", {})
interesting = (
    "turboquant_encode_cache_update.total",
    "turboquant_fused_kv_update_attention.prefill_no_cache.run_dense_fia",
    "turboquant_fused_kv_update_attention.decode.m4_attention.prepare_inputs",
    "turboquant_fused_kv_update_attention.decode.m4_attention.custom_op",
    "turboquant_fused_kv_update_attention.decode.m4_attention.output_cast",
    "turboquant_fused_kv_update_attention.decode.m4_attention.total",
    "turboquant_fused_kv_update_attention.forward",
)
if stats:
    print("\n=== Fused TurboQuant profile ===")
    for key in interesting:
        if key not in stats:
            continue
        item = stats[key]
        calls = int(item.get("calls", 0))
        total = float(item.get("total_ms", 0.0))
        avg = total / calls if calls else 0.0
        print(
            f"{key}: calls={calls} total_ms={total:.3f} "
            f"avg_ms={avg:.3f} min_ms={item.get('min_ms')} "
            f"max_ms={item.get('max_ms')}"
        )

print(f"\nResults: {root}")
print(f"Plain comparison: {root / 'plain_comparison.json'}")
print(f"Run stats: {root / 'run_stats.json'}")
PY
