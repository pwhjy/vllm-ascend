#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from argparse import BooleanOptionalAction
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS = [
    "The capital of France is",
    "Explain why the sky looks blue in one short paragraph.",
    "Write a Python function that returns the factorial of n.",
    "Summarize the role of KV cache in transformer decoding.",
]

TURBOQUANT_ENV_KEYS = (
    "VLLM_ASCEND_TQ_USE_CUSTOM_DEQUANT",
    "VLLM_ASCEND_TQ_USE_CUSTOM_PROD_DEQUANT",
    "VLLM_ASCEND_TQ_USE_CUSTOM_K_SCORE",
    "VLLM_ASCEND_TQ_USE_CUSTOM_ATTENTION",
    "VLLM_ASCEND_TQ_ATTENTION_SCORE_TILE_LEN",
    "VLLM_ASCEND_TQ_DEBUG_COMPARE",
    "VLLM_ASCEND_TQ_DEBUG_COMPARE_ATTENTION",
    "VLLM_ASCEND_TQ_DEBUG_COMPARE_DEQUANT",
    "VLLM_ASCEND_TQ_DEBUG_COMPARE_K_SCORE",
    "VLLM_ASCEND_TQ_DEBUG_COMPARE_PROD_DEQUANT",
    "VLLM_ASCEND_TQ_CUSTOM_STRICT",
    "VLLM_ASCEND_TQ_PROFILE",
    "VLLM_ASCEND_TQ_PROFILE_DIR",
    "VLLM_ASCEND_TQ_PROFILE_FLUSH_EVERY",
    "VLLM_ASCEND_TQ_PROFILE_SYNC",
    "VLLM_ASCEND_TQ_STRUCTURED_TRANSFORM",
    "VLLM_ASCEND_TQ_USE_FUSED_KV_UPDATE_ATTENTION",
    "VLLM_ASCEND_TQ_USE_CUSTOM_FUSED_KV_UPDATE_ATTENTION",
    "VLLM_ASCEND_TQ_USE_CUSTOM_ENCODE_CACHE_UPDATE",
    "VLLM_ASCEND_TQ_USE_FUSED_DECODE_DENSE_FIA",
    "VLLM_ASCEND_TQ_USE_FUSED_DECODE_ATTENTION_M4",
    "VLLM_ASCEND_TQ_USE_FUSED_PREFILL_DENSE_FIA",
    "VLLM_ASCEND_TQ_ENCODE_DEBUG_MODE",
    "VLLM_ASCEND_TQ_ENCODE_V_PARTITIONS",
    "VLLM_ASCEND_TQ_ENCODE_FORCE_FP32_INPUT",
    "VLLM_ASCEND_TQ_M4_GROUPED_Q",
    "VLLM_ASCEND_TQ_M4_SCORE_TILE_LEN",
    "VLLM_ASCEND_TQ_M4_SPLIT_CACHE_UPDATE",
    "VLLM_ASCEND_TQ_M4_PRETRANSFORM_QUERY",
    "VLLM_ASCEND_TQ_M4_FORCE_FP32_INPUT",
    "VLLM_ASCEND_TQ_M4_DEBUG_MODE",
    "VLLM_ASCEND_TQ_PROFILE_M4_STAGES",
    "VLLM_ASCEND_TQ_PROFILE_M4_SHADOW",
    "VLLM_ASCEND_TQ_USE_COMPRESSED_DECODE_CURRENT",
    "VLLM_ASCEND_TQ_USE_COMPRESSED_DECODE_CUSTOM_K_SCORE",
    "VLLM_ASCEND_TQ_USE_DECODE_COMPRESSED_FULL_CACHE",
)

DEBUG_COMPARE_ENV_KEYS = (
    "VLLM_ASCEND_TQ_DEBUG_COMPARE",
    "VLLM_ASCEND_TQ_DEBUG_COMPARE_ATTENTION",
    "VLLM_ASCEND_TQ_DEBUG_COMPARE_DEQUANT",
    "VLLM_ASCEND_TQ_DEBUG_COMPARE_K_SCORE",
    "VLLM_ASCEND_TQ_DEBUG_COMPARE_PROD_DEQUANT",
)

KV_MEMORY_RE = re.compile(r"Available KV cache memory:\s*([0-9.]+)\s*GiB")
KV_TOKENS_RE = re.compile(r"GPU KV cache size:\s*([0-9,]+)\s*tokens")
KV_CONCURRENCY_RE = re.compile(
    r"Maximum concurrency for\s*([0-9,]+)\s*tokens per request:\s*([0-9.]+)x"
)
ENGINE_INIT_RE = re.compile(
    r"init engine .* took\s*([0-9.]+)\s*seconds"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if args.prompt:
        prompts.extend(args.prompt)
    if args.prompts_file:
        with Path(args.prompts_file).open("r", encoding="utf-8") as f:
            prompts.extend(line.rstrip("\n") for line in f if line.strip())
    if args.prompts_jsonl:
        with Path(args.prompts_jsonl).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompts.append(str(row["prompt"]))
    return prompts or list(DEFAULT_PROMPTS)


def _parse_extra_env(values: list[str] | None) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in values or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--env expects KEY=VALUE, got: {item!r}")
        env[key] = value
    return env


def _variant_extra_env(args: argparse.Namespace, variant: str) -> dict[str, str]:
    if variant == "plain":
        return _parse_extra_env(args.env_plain)
    if variant == "baseline":
        return _parse_extra_env(args.env_baseline)
    if variant == "fused":
        return _parse_extra_env(args.env_fused)
    return {}


def _debug_compare_requested(
    args: argparse.Namespace,
    extra_env: dict[str, str],
) -> bool:
    return (
        args.debug_compare_all
        or args.debug_compare_attention
        or args.debug_compare_dequant
        or args.debug_compare_k_score
        or args.debug_compare_prod_dequant
        or any(key in extra_env for key in DEBUG_COMPARE_ENV_KEYS)
    )


def _set_turboquant_env(
    variant: str,
    score_tile_len: int,
    baseline_mode: str,
    debug_compare_all: bool,
    debug_compare_attention: bool,
    debug_compare_dequant: bool,
    debug_compare_k_score: bool,
    debug_compare_prod_dequant: bool,
) -> None:
    if variant == "plain":
        for key in TURBOQUANT_ENV_KEYS:
            os.environ.pop(key, None)
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        return

    use_reference_baseline = (
        variant == "baseline" and baseline_mode == "reference"
    )
    os.environ["VLLM_ASCEND_TQ_USE_CUSTOM_DEQUANT"] = (
        "0" if use_reference_baseline else "1"
    )
    os.environ["VLLM_ASCEND_TQ_USE_CUSTOM_PROD_DEQUANT"] = "0"
    os.environ["VLLM_ASCEND_TQ_USE_CUSTOM_K_SCORE"] = (
        "0" if use_reference_baseline else "1"
    )
    os.environ["VLLM_ASCEND_TQ_USE_CUSTOM_ATTENTION"] = (
        "1" if variant == "fused" else "0"
    )
    os.environ["VLLM_ASCEND_TQ_ATTENTION_SCORE_TILE_LEN"] = str(score_tile_len)
    os.environ["VLLM_ASCEND_TQ_DEBUG_COMPARE"] = (
        "1"
        if debug_compare_all
        or os.getenv("VLLM_ASCEND_TQ_DEBUG_COMPARE", "0") == "1"
        else "0"
    )
    os.environ["VLLM_ASCEND_TQ_DEBUG_COMPARE_ATTENTION"] = (
        "1"
        if (variant == "fused" and debug_compare_attention)
        or os.getenv("VLLM_ASCEND_TQ_DEBUG_COMPARE_ATTENTION", "0") == "1"
        else "0"
    )
    os.environ["VLLM_ASCEND_TQ_DEBUG_COMPARE_DEQUANT"] = (
        "1"
        if debug_compare_dequant
        or os.getenv("VLLM_ASCEND_TQ_DEBUG_COMPARE_DEQUANT", "0") == "1"
        else "0"
    )
    os.environ["VLLM_ASCEND_TQ_DEBUG_COMPARE_K_SCORE"] = (
        "1"
        if debug_compare_k_score
        or os.getenv("VLLM_ASCEND_TQ_DEBUG_COMPARE_K_SCORE", "0") == "1"
        else "0"
    )
    os.environ["VLLM_ASCEND_TQ_DEBUG_COMPARE_PROD_DEQUANT"] = (
        "1"
        if debug_compare_prod_dequant
        or os.getenv("VLLM_ASCEND_TQ_DEBUG_COMPARE_PROD_DEQUANT", "0") == "1"
        else "0"
    )
    os.environ["VLLM_ASCEND_TQ_CUSTOM_STRICT"] = "1"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
        "enable_chunked_prefill": not args.disable_chunked_prefill,
        "seed": args.seed,
    }
    if args.tokenizer:
        kwargs["tokenizer"] = args.tokenizer
    if args.quantization not in (None, "", "none", "None"):
        kwargs["quantization"] = args.quantization
    if args.swap_space is not None:
        kwargs["swap_space"] = args.swap_space
    if args.block_size is not None:
        kwargs["block_size"] = args.block_size
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs
    return kwargs


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _collect_kv_cache_stats(llm: Any, args: argparse.Namespace) -> dict[str, Any]:
    vllm_config = getattr(llm.llm_engine, "vllm_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)

    num_gpu_blocks = _safe_int(getattr(cache_config, "num_gpu_blocks", None))
    num_cpu_blocks = _safe_int(getattr(cache_config, "num_cpu_blocks", None))
    block_size = _safe_int(getattr(cache_config, "block_size", args.block_size))
    max_model_len = _safe_int(getattr(model_config, "max_model_len", args.max_model_len))
    dcp_size = _safe_int(
        getattr(parallel_config, "decode_context_parallel_size", None)
    ) or 1
    pcp_size = _safe_int(
        getattr(parallel_config, "prefill_context_parallel_size", None)
    ) or 1

    capacity_tokens = None
    max_concurrency = None
    if num_gpu_blocks is not None and block_size is not None:
        capacity_tokens = num_gpu_blocks * block_size * dcp_size * pcp_size
        if max_model_len:
            max_concurrency = capacity_tokens / max_model_len

    stats: dict[str, Any] = {
        "num_gpu_blocks": num_gpu_blocks,
        "num_cpu_blocks": num_cpu_blocks,
        "block_size": block_size,
        "capacity_tokens_estimate": capacity_tokens,
        "max_model_len": max_model_len,
        "max_concurrency_estimate": max_concurrency,
        "gpu_memory_utilization": _safe_float(
            getattr(cache_config, "gpu_memory_utilization", args.gpu_memory_utilization)
        ),
        "kv_cache_dtype": str(getattr(cache_config, "cache_dtype", None)),
        "decode_context_parallel_size": dcp_size,
        "prefill_context_parallel_size": pcp_size,
        "source": "vllm_config.cache_config",
    }

    model_executor = getattr(llm.llm_engine, "model_executor", None)
    model_runner = getattr(model_executor, "model_runner", None)
    kv_cache_config = getattr(model_runner, "kv_cache_config", None)
    if kv_cache_config is not None:
        kv_groups = getattr(kv_cache_config, "kv_cache_groups", []) or []
        group_block_sizes = []
        for group in kv_groups:
            spec = getattr(group, "kv_cache_spec", None)
            group_block_sizes.append(_safe_int(getattr(spec, "block_size", None)))
        group_block_sizes = [size for size in group_block_sizes if size is not None]
        group_count = len(kv_groups)
        exact_num_blocks = _safe_int(getattr(kv_cache_config, "num_blocks", None))
        if exact_num_blocks is not None and group_count and group_block_sizes:
            exact_tokens = (
                exact_num_blocks
                // group_count
                * min(group_block_sizes)
                * dcp_size
                * pcp_size
            )
            stats.update(
                {
                    "num_gpu_blocks": exact_num_blocks,
                    "kv_cache_group_count": group_count,
                    "group_block_sizes": group_block_sizes,
                    "capacity_tokens": exact_tokens,
                    "max_concurrency": (
                        exact_tokens / max_model_len if max_model_len else None
                    ),
                    "source": "model_runner.kv_cache_config",
                }
            )

    return stats


def _build_throughput_stats(
    outputs: list[dict[str, Any]],
    init_seconds: float,
    generate_seconds: float,
) -> dict[str, Any]:
    prompt_tokens = sum(len(row["prompt_token_ids"]) for row in outputs)
    generated_tokens = sum(len(row["generated_token_ids"]) for row in outputs)
    total_tokens = prompt_tokens + generated_tokens
    seconds = max(generate_seconds, 1e-12)
    total_seconds = max(init_seconds + generate_seconds, 1e-12)
    return {
        "num_requests": len(outputs),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": total_tokens,
        "output_tokens_per_second": generated_tokens / seconds,
        "total_tokens_per_second": total_tokens / seconds,
        "requests_per_second": len(outputs) / seconds,
        "end_to_end_output_tokens_per_second": generated_tokens / total_seconds,
        "end_to_end_total_tokens_per_second": total_tokens / total_seconds,
    }


def _run_worker(args: argparse.Namespace) -> int:
    assert args._worker_variant is not None
    assert args._output_json is not None

    _set_turboquant_env(
        args._worker_variant,
        args.score_tile_len,
        args.baseline_mode,
        args.debug_compare_all,
        args.debug_compare_attention,
        args.debug_compare_dequant,
        args.debug_compare_k_score,
        args.debug_compare_prod_dequant,
    )
    if args.modelscope:
        os.environ["VLLM_USE_MODELSCOPE"] = "True"

    from vllm import LLM, SamplingParams

    try:
        from vllm_ascend.utils import enable_custom_op

        enable_custom_op()
    except Exception:
        pass

    prompts = _load_prompts(args)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    init_start = time.perf_counter()
    llm = LLM(**_build_llm_kwargs(args))
    init_seconds = time.perf_counter() - init_start
    kv_cache_stats = _collect_kv_cache_stats(llm, args)
    generate_start = time.perf_counter()
    request_outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=not args.no_tqdm,
    )
    generate_seconds = time.perf_counter() - generate_start

    outputs: list[dict[str, Any]] = []
    for index, request_output in enumerate(request_outputs):
        sample = request_output.outputs[0]
        outputs.append(
            {
                "index": index,
                "prompt": request_output.prompt,
                "prompt_token_ids": list(request_output.prompt_token_ids),
                "generated_text": sample.text,
                "generated_token_ids": list(sample.token_ids),
                "finish_reason": sample.finish_reason,
            }
        )

    throughput = _build_throughput_stats(
        outputs,
        init_seconds,
        generate_seconds,
    )
    env_snapshot = {key: os.getenv(key) for key in TURBOQUANT_ENV_KEYS}
    _write_json(
        Path(args._output_json),
        {
            "variant": args._worker_variant,
            "model": args.model,
            "quantization": args.quantization,
            "sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
            },
            "env": env_snapshot,
            "timing": {
                "init_seconds": init_seconds,
                "generate_seconds": generate_seconds,
                "end_to_end_seconds": init_seconds + generate_seconds,
            },
            "throughput": throughput,
            "kv_cache": kv_cache_stats,
            "outputs": outputs,
        },
    )
    print(f"{args._worker_variant}: wrote {args._output_json}")
    return 0


def _first_diff(left: list[int], right: list[int]) -> int | None:
    for idx, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _compare_outputs(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_label: str = "baseline",
    right_label: str = "fused",
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    left_outputs = left["outputs"]
    right_outputs = right["outputs"]
    if len(left_outputs) != len(right_outputs):
        mismatches.append(
            {
                "kind": "output_count",
                f"{left_label}_count": len(left_outputs),
                f"{right_label}_count": len(right_outputs),
            }
        )

    for left_row, right_row in zip(left_outputs, right_outputs):
        left_ids = list(left_row["generated_token_ids"])
        right_ids = list(right_row["generated_token_ids"])
        token_diff = _first_diff(left_ids, right_ids)
        prompt_match = (
            left_row["prompt_token_ids"] == right_row["prompt_token_ids"]
        )
        text_match = left_row["generated_text"] == right_row["generated_text"]
        if token_diff is None and prompt_match and text_match:
            continue
        mismatch: dict[str, Any] = {
            "index": left_row["index"],
            "prompt": left_row["prompt"],
            "prompt_token_ids_match": prompt_match,
            "generated_text_match": text_match,
            "generated_token_ids_match": token_diff is None,
            f"{left_label}_token_count": len(left_ids),
            f"{right_label}_token_count": len(right_ids),
            "first_token_diff": token_diff,
            f"{left_label}_text": left_row["generated_text"],
            f"{right_label}_text": right_row["generated_text"],
        }
        if token_diff is not None:
            mismatch[f"{left_label}_token_at_diff"] = (
                left_ids[token_diff] if token_diff < len(left_ids) else None
            )
            mismatch[f"{right_label}_token_at_diff"] = (
                right_ids[token_diff] if token_diff < len(right_ids) else None
            )
        mismatches.append(mismatch)

    return {
        "passed": not mismatches,
        "num_prompts": min(len(left_outputs), len(right_outputs)),
        "mismatches": mismatches,
    }


def _parse_child_log(log_text: str) -> dict[str, Any]:
    kv_memory_matches = KV_MEMORY_RE.findall(log_text)
    kv_token_matches = KV_TOKENS_RE.findall(log_text)
    kv_concurrency_matches = KV_CONCURRENCY_RE.findall(log_text)
    engine_init_matches = ENGINE_INIT_RE.findall(log_text)

    stats: dict[str, Any] = {}
    if kv_memory_matches:
        stats["available_kv_cache_gib"] = float(kv_memory_matches[-1])
        stats["available_kv_cache_bytes_estimate"] = int(
            stats["available_kv_cache_gib"] * (1024**3)
        )
    if kv_token_matches:
        stats["kv_cache_capacity_tokens"] = int(
            kv_token_matches[-1].replace(",", "")
        )
    if kv_concurrency_matches:
        max_model_len, concurrency = kv_concurrency_matches[-1]
        stats["max_model_len_for_concurrency"] = int(
            max_model_len.replace(",", "")
        )
        stats["max_concurrency"] = float(concurrency)
    if engine_init_matches:
        stats["engine_init_seconds_from_log"] = float(engine_init_matches[-1])
    return stats


def _warn_if_plain_model_looks_quantized(model: str) -> None:
    plain_model = Path(model)
    if (
        plain_model.is_dir()
        and (plain_model / "quant_model_description.json").exists()
    ):
        print(
            "WARNING: --plain-model points to a directory with "
            "quant_model_description.json; vllm-ascend may auto-detect "
            "Ascend quantization. Use a model directory without that file "
            "for a true non-TurboQuant baseline."
        )


def _aggregate_profile_stat(
    target: dict[str, Any],
    source: dict[str, Any],
) -> None:
    target["calls"] = int(target.get("calls", 0)) + int(source.get("calls", 0))
    target["total_ms"] = float(target.get("total_ms", 0.0)) + float(
        source.get("total_ms", 0.0)
    )
    source_min = source.get("min_ms")
    if source_min is not None:
        target["min_ms"] = (
            float(source_min)
            if target.get("min_ms") is None
            else min(float(target["min_ms"]), float(source_min))
        )
    target["max_ms"] = max(
        float(target.get("max_ms", 0.0)),
        float(source.get("max_ms", 0.0)),
    )
    for key in ("vectors", "elements", "bytes_out"):
        target[key] = int(target.get(key, 0)) + int(source.get(key, 0))


def _collect_turboquant_profile(profile_dir: Path) -> dict[str, Any]:
    files = sorted(profile_dir.glob("turboquant_profile_*.json"))
    stats: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for path in files:
        try:
            payload = _read_json(path)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        for name, stat in payload.get("stats", {}).items():
            aggregate = stats.setdefault(
                name,
                {
                    "calls": 0,
                    "total_ms": 0.0,
                    "min_ms": None,
                    "max_ms": 0.0,
                    "vectors": 0,
                    "elements": 0,
                    "bytes_out": 0,
                },
            )
            _aggregate_profile_stat(aggregate, stat)

    for stat in stats.values():
        calls = int(stat.get("calls", 0))
        stat["avg_ms"] = (
            float(stat.get("total_ms", 0.0)) / calls if calls else None
        )

    key_stats = {
        key: stats[key]
        for key in (
            "turboquant_decode.fused_attention",
            "turboquant_decode.run_dense_fia",
            "turboquant_decode.total",
            "turboquant_decode.custom_mse.total",
            "turboquant_decode.hybrid_prod.k",
            "turboquant_decode.custom_mse.v",
            "turboquant_chunked_prefill.run_dense_fia",
            "turboquant_chunked_prefill.total",
            "turboquant_chunked_prefill.custom_mse.total",
            "turboquant_prefill_cache_hit.run_dense_fia",
            "turboquant_prefill_cache_hit.total",
        )
        if key in stats
    }
    return {
        "profile_dir": str(profile_dir),
        "files": [str(path) for path in files],
        "num_files": len(files),
        "stats": stats,
        "key_stats": key_stats,
        "errors": errors,
    }


def _summarize_worker_stats(worker_json: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": worker_json.get("variant"),
        "model": worker_json.get("model"),
        "quantization": worker_json.get("quantization"),
        "timing": worker_json.get("timing", {}),
        "throughput": worker_json.get("throughput", {}),
        "kv_cache": worker_json.get("kv_cache", {}),
    }


def _print_run_stats_summary(run_stats: dict[str, Any]) -> None:
    print("\nRun stats:")
    workers = run_stats.get("workers", {})
    child_processes = run_stats.get("child_processes", {})
    for variant, stats in workers.items():
        timing = stats.get("timing", {})
        throughput = stats.get("throughput", {})
        kv_cache = stats.get("kv_cache", {})
        log_stats = child_processes.get(variant, {}).get("log_stats", {})
        kv_tokens = (
            log_stats.get("kv_cache_capacity_tokens")
            or kv_cache.get("capacity_tokens")
            or kv_cache.get("capacity_tokens_estimate")
        )
        kv_gib = log_stats.get("available_kv_cache_gib")
        generated_tps = throughput.get("output_tokens_per_second")
        print(
            f"  {variant}: "
            f"init={timing.get('init_seconds', 0.0):.3f}s "
            f"generate={timing.get('generate_seconds', 0.0):.3f}s "
            f"out_tok/s={generated_tps or 0.0:.2f} "
            f"kv_tokens={kv_tokens}"
            + (f" kv_gib={kv_gib:.2f}" if kv_gib is not None else "")
        )
        profile = child_processes.get(variant, {}).get("turboquant_profile", {})
        key_stats = profile.get("key_stats", {})
        if key_stats:
            fused_calls = key_stats.get(
                "turboquant_decode.fused_attention", {}
            ).get("calls", 0)
            dense_calls = key_stats.get(
                "turboquant_decode.run_dense_fia", {}
            ).get("calls", 0)
            print(
                f"    tq_profile: fused_attention_calls={fused_calls} "
                f"dense_fia_calls={dense_calls} files={profile.get('num_files', 0)}"
            )


def _worker_command(
    args: argparse.Namespace,
    variant: str,
    output_json: Path,
) -> list[str]:
    model = args.model
    tokenizer = args.tokenizer
    quantization = args.quantization
    if variant == "plain":
        model = args.plain_model or args.model
        tokenizer = args.plain_tokenizer or args.tokenizer
        quantization = args.plain_quantization

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker-variant",
        variant,
        "--_output-json",
        str(output_json),
        "--model",
        model,
        "--dtype",
        args.dtype,
        "--quantization",
        quantization,
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--seed",
        str(args.seed),
        "--score-tile-len",
        str(args.score_tile_len),
        "--baseline-mode",
        args.baseline_mode,
    ]
    if tokenizer:
        cmd.extend(["--tokenizer", tokenizer])
    if args.swap_space is not None:
        cmd.extend(["--swap-space", str(args.swap_space)])
    if args.block_size is not None:
        cmd.extend(["--block-size", str(args.block_size)])
    if args.max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(args.max_num_seqs)])
    if args.disable_chunked_prefill:
        cmd.append("--disable-chunked-prefill")
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    else:
        cmd.append("--no-trust-remote-code")
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.modelscope:
        cmd.append("--modelscope")
    if args.no_tqdm:
        cmd.append("--no-tqdm")
    if args.debug_compare_all:
        cmd.append("--debug-compare-all")
    if args.debug_compare_attention:
        cmd.append("--debug-compare-attention")
    if args.debug_compare_dequant:
        cmd.append("--debug-compare-dequant")
    if args.debug_compare_k_score:
        cmd.append("--debug-compare-k-score")
    if args.debug_compare_prod_dequant:
        cmd.append("--debug-compare-prod-dequant")
    if args.profile_turboquant:
        cmd.append("--profile-turboquant")
    if args.profile_turboquant_sync:
        cmd.append("--profile-turboquant-sync")
    for prompt in args.prompt or []:
        cmd.extend(["--prompt", prompt])
    if args.prompts_file:
        cmd.extend(["--prompts-file", args.prompts_file])
    if args.prompts_jsonl:
        cmd.extend(["--prompts-jsonl", args.prompts_jsonl])
    return cmd


def _run_child(
    args: argparse.Namespace,
    variant: str,
    output_json: Path,
    env: dict[str, str],
    log_path: Path,
) -> dict[str, Any]:
    cmd = _worker_command(args, variant, output_json)
    print(f"\n=== Running {variant} ===")
    print(" ".join(cmd))
    timeout = None if args.timeout_sec <= 0 else args.timeout_sec
    start = time.perf_counter()
    log_lines: list[str] = []
    child_env = env.copy()
    child_env.update(_variant_extra_env(args, variant))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    profile_dir: Path | None = None
    if args.profile_turboquant:
        profile_dir = log_path.parent / "turboquant_profiles" / variant
        if profile_dir.exists():
            shutil.rmtree(profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        child_env["VLLM_ASCEND_TQ_PROFILE"] = "1"
        child_env["VLLM_ASCEND_TQ_PROFILE_DIR"] = str(profile_dir)
        child_env["VLLM_ASCEND_TQ_PROFILE_FLUSH_EVERY"] = "100"
        child_env["VLLM_ASCEND_TQ_PROFILE_SYNC"] = (
            "1" if args.profile_turboquant_sync else "0"
        )
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=str(_repo_root()),
            env=child_env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )

        def _tee_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()
                log_lines.append(line)

        reader = threading.Thread(target=_tee_output, daemon=True)
        reader.start()
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            reader.join(timeout=5)
            raise SystemExit(
                f"{variant} run timed out after {args.timeout_sec} seconds"
            ) from None
        reader.join()

    elapsed = time.perf_counter() - start
    if returncode != 0:
        raise SystemExit(f"{variant} run failed with exit code {returncode}")

    log_text = "".join(log_lines)
    result = {
        "variant": variant,
        "returncode": returncode,
        "process_seconds": elapsed,
        "log_path": str(log_path),
        "log_stats": _parse_child_log(log_text),
    }
    if profile_dir is not None:
        result["turboquant_profile"] = _collect_turboquant_profile(profile_dir)
    return result


def _run_compare(args: argparse.Namespace) -> int:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _repo_root() / "benchmarks" / "results" / f"turboquant_llama_correctness_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plain_json = output_dir / "plain.json"
    baseline_json = output_dir / "baseline.json"
    fused_json = output_dir / "fused.json"
    comparison_json = output_dir / "comparison.json"
    plain_comparison_json = output_dir / "plain_comparison.json"
    run_stats_json = output_dir / "run_stats.json"

    extra_env = _parse_extra_env(args.env)
    env = os.environ.copy()
    if not _debug_compare_requested(args, extra_env):
        ignored_debug_env = [
            key for key in DEBUG_COMPARE_ENV_KEYS if key in env
        ]
        for key in ignored_debug_env:
            env.pop(key, None)
        if ignored_debug_env:
            print(
                "NOTE: ignoring inherited TurboQuant debug compare env: "
                + ", ".join(ignored_debug_env)
            )
            print(
                "      Use --debug-compare-all, "
                "--debug-compare-attention, or --env KEY=VALUE to enable it."
            )
    env.update(extra_env)

    child_processes: dict[str, dict[str, Any]] = {}
    if args.plain_only:
        _warn_if_plain_model_looks_quantized(args.plain_model or args.model)
        child_processes["plain"] = _run_child(
            args, "plain", plain_json, env, output_dir / "plain.log"
        )
        plain = _read_json(plain_json)
        worker_stats = {"plain": _summarize_worker_stats(plain)}
        run_stats = {
            "profile_turboquant": args.profile_turboquant,
            "profile_turboquant_sync": args.profile_turboquant_sync,
            "workers": worker_stats,
            "child_processes": child_processes,
        }
        _write_json(run_stats_json, run_stats)
        print(f"Results: {output_dir}")
        _print_run_stats_summary(run_stats)
        return 0

    if args.fused_only:
        child_processes["fused"] = _run_child(
            args, "fused", fused_json, env, output_dir / "fused.log"
        )
        fused = _read_json(fused_json)
        worker_stats = {"fused": _summarize_worker_stats(fused)}
        run_stats = {
            "profile_turboquant": args.profile_turboquant,
            "profile_turboquant_sync": args.profile_turboquant_sync,
            "workers": worker_stats,
            "child_processes": child_processes,
        }
        _write_json(run_stats_json, run_stats)
        _write_json(
            comparison_json,
            {
                "passed": True,
                "num_prompts": len(fused.get("outputs", [])),
                "mismatches": [],
                "fused_json": str(fused_json),
                "run_stats_json": str(run_stats_json),
                "run_stats": run_stats,
            },
        )
        print(f"\nPASS: fused-only run completed.")
        print(f"Results: {output_dir}")
        _print_run_stats_summary(run_stats)
        return 0

    if args.include_plain_baseline:
        _warn_if_plain_model_looks_quantized(args.plain_model or args.model)
        child_processes["plain"] = _run_child(
            args, "plain", plain_json, env, output_dir / "plain.log"
        )
    child_processes["baseline"] = _run_child(
        args, "baseline", baseline_json, env, output_dir / "baseline.log"
    )
    child_processes["fused"] = _run_child(
        args, "fused", fused_json, env, output_dir / "fused.log"
    )

    baseline = _read_json(baseline_json)
    fused = _read_json(fused_json)
    worker_stats = {
        "baseline": _summarize_worker_stats(baseline),
        "fused": _summarize_worker_stats(fused),
    }
    comparison = _compare_outputs(baseline, fused)
    comparison.update(
        {
            "baseline_json": str(baseline_json),
            "fused_json": str(fused_json),
            "baseline_mode": args.baseline_mode,
            "run_stats_json": str(run_stats_json),
        }
    )
    if args.include_plain_baseline:
        plain = _read_json(plain_json)
        worker_stats["plain"] = _summarize_worker_stats(plain)
        plain_comparison = _compare_outputs(
            plain,
            baseline,
            left_label="plain",
            right_label="turboquant_baseline",
        )
        plain_comparison.update(
            {
                "plain_json": str(plain_json),
                "turboquant_baseline_json": str(baseline_json),
                "note": (
                    "This compares normal KV-cache generation with the "
                    "TurboQuant reference baseline. Exact token equality is "
                    "not required for PR5 fused-attention correctness because "
                    "TurboQuant itself is lossy."
                ),
            }
        )
        plain_comparison["run_stats"] = {
            "plain": worker_stats["plain"],
            "turboquant_baseline": worker_stats["baseline"],
        }
        plain_comparison["child_processes"] = {
            "plain": child_processes["plain"],
            "turboquant_baseline": child_processes["baseline"],
        }
        _write_json(plain_comparison_json, plain_comparison)
        comparison["plain_comparison_json"] = str(plain_comparison_json)
    run_stats = {
        "profile_turboquant": args.profile_turboquant,
        "profile_turboquant_sync": args.profile_turboquant_sync,
        "workers": worker_stats,
        "child_processes": child_processes,
    }
    _write_json(run_stats_json, run_stats)
    comparison["run_stats"] = run_stats
    _write_json(comparison_json, comparison)

    if comparison["passed"]:
        print(
            f"\nPASS: fused PR5 attention matched baseline for "
            f"{comparison['num_prompts']} prompt(s)."
        )
    else:
        print(
            f"\nFAIL: found {len(comparison['mismatches'])} mismatch(es). "
            f"See {comparison_json}"
        )
        for mismatch in comparison["mismatches"][:3]:
            print(json.dumps(mismatch, ensure_ascii=False, indent=2))
    print(f"Results: {output_dir}")
    if args.include_plain_baseline:
        print(f"Plain comparison: {plain_comparison_json}")
    _print_run_stats_summary(run_stats)
    return 0 if comparison["passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare TurboQuant LLaMA decode outputs with PR5 fused attention "
            "disabled and enabled."
        )
    )
    parser.add_argument("--model", required=True, help="Model name or local model path.")
    parser.add_argument("--tokenizer", help="Optional tokenizer name or path.")
    parser.add_argument(
        "--quantization",
        default="ascend",
        help="Use 'ascend' for a TurboQuant ModelSlim model; use 'none' to omit.",
    )
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--swap-space", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument(
        "--trust-remote-code",
        action=BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--modelscope", action="store_true")

    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--prompt",
        action="append",
        help="Prompt to test. May be passed multiple times.",
    )
    parser.add_argument(
        "--prompts-file",
        help="Plain text file with one prompt per non-empty line.",
    )
    parser.add_argument(
        "--prompts-jsonl",
        help="JSONL file with a 'prompt' field on each row.",
    )

    parser.add_argument("--score-tile-len", type=int, default=64)
    parser.add_argument(
        "--baseline-mode",
        choices=("reference", "custom"),
        default="reference",
        help=(
            "'reference' disables TurboQuant custom dequant for the baseline; "
            "'custom' keeps the faster custom MSE dequant baseline."
        ),
    )
    parser.add_argument(
        "--debug-compare-all",
        action="store_true",
        help=(
            "Enable all TurboQuant custom-op debug comparisons inside workers. "
            "This is equivalent to VLLM_ASCEND_TQ_DEBUG_COMPARE=1."
        ),
    )
    parser.add_argument(
        "--debug-compare-attention",
        action="store_true",
        help=(
            "Enable only the PR5 fused attention debug comparison in the "
            "fused worker."
        ),
    )
    parser.add_argument(
        "--debug-compare-dequant",
        action="store_true",
        help="Enable TurboQuant MSE dequant debug comparisons.",
    )
    parser.add_argument(
        "--debug-compare-k-score",
        action="store_true",
        help="Enable TurboQuant compressed K-score debug comparisons.",
    )
    parser.add_argument(
        "--debug-compare-prod-dequant",
        action="store_true",
        help="Enable TurboQuant prod dequant debug comparisons.",
    )
    parser.add_argument(
        "--profile-turboquant",
        action="store_true",
        help=(
            "Collect TurboQuant runtime profile counters for each worker and "
            "embed them in run_stats.json."
        ),
    )
    parser.add_argument(
        "--profile-turboquant-sync",
        action="store_true",
        help=(
            "Synchronize NPU profile points for more accurate per-op timing. "
            "By default --profile-turboquant collects hit counts with async "
            "timings to reduce measurement overhead."
        ),
    )
    parser.add_argument(
        "--include-plain-baseline",
        action="store_true",
        help="Also run a non-TurboQuant/plain model baseline for context.",
    )
    parser.add_argument(
        "--plain-only",
        action="store_true",
        help=(
            "Run only the non-TurboQuant/plain worker and write plain.json "
            "plus run_stats.json."
        ),
    )
    parser.add_argument(
        "--fused-only",
        action="store_true",
        help=(
            "Run only the fused TurboQuant worker and write fused.json plus "
            "run_stats.json. Useful for diagnostic profile sweeps."
        ),
    )
    parser.add_argument(
        "--plain-model",
        help=(
            "Model path for --include-plain-baseline or --plain-only. "
            "Defaults to --model."
        ),
    )
    parser.add_argument(
        "--plain-tokenizer",
        help=(
            "Tokenizer path for --include-plain-baseline or --plain-only. "
            "Defaults to --tokenizer."
        ),
    )
    parser.add_argument(
        "--plain-quantization",
        default="none",
        help="Quantization argument for the plain baseline; default omits it.",
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--env",
        action="append",
        help="Extra child-process environment variable, in KEY=VALUE form.",
    )
    parser.add_argument(
        "--env-plain",
        action="append",
        help="Extra environment variable for only the plain worker.",
    )
    parser.add_argument(
        "--env-baseline",
        action="append",
        help="Extra environment variable for only the TurboQuant baseline worker.",
    )
    parser.add_argument(
        "--env-fused",
        action="append",
        help="Extra environment variable for only the fused worker.",
    )
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument("--no-tqdm", action="store_true", default=True)

    parser.add_argument("--_worker-variant", choices=("plain", "baseline", "fused"), help=argparse.SUPPRESS)
    parser.add_argument("--_output-json", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.score_tile_len <= 0 or args.score_tile_len > 256:
        raise ValueError("--score-tile-len must be in [1, 256]")
    if args.profile_turboquant_sync and not args.profile_turboquant:
        args.profile_turboquant = True
    if args._worker_variant:
        return _run_worker(args)
    return _run_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
