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
import subprocess
import sys
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
    "VLLM_ASCEND_TQ_CUSTOM_STRICT",
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


def _set_turboquant_env(variant: str, score_tile_len: int) -> None:
    os.environ["VLLM_ASCEND_TQ_USE_CUSTOM_DEQUANT"] = "1"
    os.environ["VLLM_ASCEND_TQ_USE_CUSTOM_PROD_DEQUANT"] = "0"
    os.environ["VLLM_ASCEND_TQ_USE_CUSTOM_K_SCORE"] = "1"
    os.environ["VLLM_ASCEND_TQ_USE_CUSTOM_ATTENTION"] = (
        "1" if variant == "fused" else "0"
    )
    os.environ["VLLM_ASCEND_TQ_ATTENTION_SCORE_TILE_LEN"] = str(score_tile_len)
    os.environ["VLLM_ASCEND_TQ_DEBUG_COMPARE"] = "0"
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


def _run_worker(args: argparse.Namespace) -> int:
    assert args._worker_variant is not None
    assert args._output_json is not None

    _set_turboquant_env(args._worker_variant, args.score_tile_len)
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
    llm = LLM(**_build_llm_kwargs(args))
    request_outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=not args.no_tqdm,
    )

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


def _compare_outputs(baseline: dict[str, Any], fused: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    baseline_outputs = baseline["outputs"]
    fused_outputs = fused["outputs"]
    if len(baseline_outputs) != len(fused_outputs):
        mismatches.append(
            {
                "kind": "output_count",
                "baseline_count": len(baseline_outputs),
                "fused_count": len(fused_outputs),
            }
        )

    for base_row, fused_row in zip(baseline_outputs, fused_outputs):
        base_ids = list(base_row["generated_token_ids"])
        fused_ids = list(fused_row["generated_token_ids"])
        token_diff = _first_diff(base_ids, fused_ids)
        prompt_match = base_row["prompt_token_ids"] == fused_row["prompt_token_ids"]
        text_match = base_row["generated_text"] == fused_row["generated_text"]
        if token_diff is None and prompt_match and text_match:
            continue
        mismatch: dict[str, Any] = {
            "index": base_row["index"],
            "prompt": base_row["prompt"],
            "prompt_token_ids_match": prompt_match,
            "generated_text_match": text_match,
            "generated_token_ids_match": token_diff is None,
            "baseline_token_count": len(base_ids),
            "fused_token_count": len(fused_ids),
            "first_token_diff": token_diff,
            "baseline_text": base_row["generated_text"],
            "fused_text": fused_row["generated_text"],
        }
        if token_diff is not None:
            mismatch["baseline_token_at_diff"] = (
                base_ids[token_diff] if token_diff < len(base_ids) else None
            )
            mismatch["fused_token_at_diff"] = (
                fused_ids[token_diff] if token_diff < len(fused_ids) else None
            )
        mismatches.append(mismatch)

    return {
        "passed": not mismatches,
        "num_prompts": min(len(baseline_outputs), len(fused_outputs)),
        "mismatches": mismatches,
    }


def _worker_command(
    args: argparse.Namespace,
    variant: str,
    output_json: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker-variant",
        variant,
        "--_output-json",
        str(output_json),
        "--model",
        args.model,
        "--dtype",
        args.dtype,
        "--quantization",
        args.quantization,
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
    ]
    if args.tokenizer:
        cmd.extend(["--tokenizer", args.tokenizer])
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
) -> None:
    cmd = _worker_command(args, variant, output_json)
    print(f"\n=== Running {variant} ===")
    print(" ".join(cmd))
    timeout = None if args.timeout_sec <= 0 else args.timeout_sec
    result = subprocess.run(
        cmd,
        cwd=str(_repo_root()),
        env=env,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"{variant} run failed with exit code {result.returncode}")


def _run_compare(args: argparse.Namespace) -> int:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _repo_root() / "benchmarks" / "results" / f"turboquant_llama_correctness_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_json = output_dir / "baseline.json"
    fused_json = output_dir / "fused.json"
    comparison_json = output_dir / "comparison.json"

    env = os.environ.copy()
    env.update(_parse_extra_env(args.env))

    _run_child(args, "baseline", baseline_json, env)
    _run_child(args, "fused", fused_json, env)

    baseline = _read_json(baseline_json)
    fused = _read_json(fused_json)
    comparison = _compare_outputs(baseline, fused)
    comparison.update(
        {
            "baseline_json": str(baseline_json),
            "fused_json": str(fused_json),
        }
    )
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
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--env",
        action="append",
        help="Extra child-process environment variable, in KEY=VALUE form.",
    )
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument("--no-tqdm", action="store_true", default=True)

    parser.add_argument("--_worker-variant", choices=("baseline", "fused"), help=argparse.SUPPRESS)
    parser.add_argument("--_output-json", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.score_tile_len <= 0 or args.score_tile_len > 256:
        raise ValueError("--score-tile-len must be in [1, 256]")
    if args._worker_variant:
        return _run_worker(args)
    return _run_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
