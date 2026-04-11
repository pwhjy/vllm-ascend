import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (
    DEFAULT_SEED,
    REPO_ROOT,
    add_common_llm_args,
    build_repeated_token_sequence,
    ensure_dir,
    make_llm,
    make_prompt_from_token_ids,
    write_json,
)


MARKER = "P0_MEMORY_PROBE_JSON="
KV_GIB_RE = re.compile(r"Available KV cache memory:\s*([0-9.]+)\s*GiB")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe P0 memory baseline and max stable context length.")
    add_common_llm_args(parser)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-max-model-len", type=int, default=32768)
    parser.add_argument("--max-model-len-limit", type=int, default=65536)
    parser.add_argument("--step", type=int, default=4096)
    parser.add_argument("--target-context-ratio", type=float, default=0.90)
    parser.add_argument("--generate-tokens", type=int, default=1)
    parser.add_argument("--probe-filler-text", default="This is a baseline memory probe sentence. ")
    parser.add_argument("--probe-subcommand", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe-log-file", help=argparse.SUPPRESS)
    return parser


def parse_probe_marker(text: str) -> dict[str, Any] | None:
    for line in text.splitlines():
        if line.startswith(MARKER):
            return json.loads(line[len(MARKER) :].strip())
    return None


def parse_kv_gib(text: str) -> float | None:
    match = KV_GIB_RE.search(text)
    if not match:
        return None
    return float(match.group(1))


def run_probe_subprocess(args: argparse.Namespace, max_model_len: int, log_file: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--probe-subcommand",
        "--model",
        args.model,
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--swap-space",
        str(args.swap_space),
        "--block-size",
        str(args.block_size),
        "--seed",
        str(args.seed if args.seed is not None else DEFAULT_SEED),
        "--generate-tokens",
        str(args.generate_tokens),
        "--target-context-ratio",
        str(args.target_context_ratio),
        "--probe-filler-text",
        args.probe_filler_text,
        "--output-dir",
        str(args.output_dir),
        "--probe-log-file",
        str(log_file),
    ]
    if args.tokenizer:
        command.extend(["--tokenizer", args.tokenizer])
    if args.quantization not in (None, "", "none", "None"):
        command.extend(["--quantization", args.quantization])
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.disable_chunked_prefill:
        command.append("--disable-chunked-prefill")
    if args.disable_log_stats:
        command.append("--disable-log-stats")
    if args.trust_remote_code:
        command.append("--trust-remote-code")

    env = dict(os.environ)
    env.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    combined_text = result.stdout + "\n" + result.stderr
    log_file.write_text(combined_text, encoding="utf-8")
    payload = parse_probe_marker(combined_text) or {}
    payload["returncode"] = result.returncode
    payload["available_kv_cache_gib"] = parse_kv_gib(combined_text)
    payload["log_file"] = str(log_file)
    if result.returncode != 0 and "error" not in payload:
        payload["error"] = combined_text[-4000:]
    return payload


def run_probe(args: argparse.Namespace) -> None:
    from vllm import SamplingParams

    llm = make_llm(args)
    tokenizer = llm.get_tokenizer()

    target_prompt_tokens = max(
        32,
        int((args.max_model_len - args.generate_tokens - 8) * args.target_context_ratio),
    )
    prompt_token_ids = build_repeated_token_sequence(
        tokenizer,
        target_token_count=target_prompt_tokens,
        unit_text=args.probe_filler_text,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_tokens=args.generate_tokens,
        seed=args.seed if args.seed is not None else DEFAULT_SEED,
    )
    outputs = llm.generate([make_prompt_from_token_ids(prompt_token_ids)], sampling_params=sampling_params, use_tqdm=False)
    output = outputs[0].outputs[0]
    payload = {
        "success": True,
        "max_model_len": args.max_model_len,
        "target_prompt_tokens": target_prompt_tokens,
        "actual_prompt_tokens": len(outputs[0].prompt_token_ids),
        "generated_tokens": len(output.token_ids),
        "finish_reason": output.finish_reason,
    }
    print(f"{MARKER}{json.dumps(payload, ensure_ascii=False)}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.probe_subcommand:
        run_probe(args)
        return

    output_dir = ensure_dir(args.output_dir)
    logs_dir = ensure_dir(output_dir / "logs")

    trials: list[dict[str, Any]] = []
    first_success_log: Path | None = None
    last_success: dict[str, Any] | None = None

    for max_model_len in range(args.start_max_model_len, args.max_model_len_limit + 1, args.step):
        log_file = logs_dir / f"probe_max_model_len_{max_model_len}.log"
        trial = run_probe_subprocess(args, max_model_len, log_file)
        trial["max_model_len"] = max_model_len
        trial["success"] = bool(trial.get("success")) and trial.get("returncode", 1) == 0
        if not first_success_log and trial["success"]:
            first_success_log = log_file
        if trial["success"]:
            last_success = trial
        trials.append(trial)
        if not trial["success"]:
            break

    if first_success_log is not None:
        (output_dir / "profile_run_log.txt").write_text(first_success_log.read_text(encoding="utf-8"), encoding="utf-8")

    available_kv_cache_gib = None if last_success is None else last_success.get("available_kv_cache_gib")
    available_kv_cache_bytes = None
    if available_kv_cache_gib is not None:
        available_kv_cache_bytes = int(float(available_kv_cache_gib) * (1024**3))

    summary = {
        "model": args.model,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "start_max_model_len": args.start_max_model_len,
        "max_model_len_limit": args.max_model_len_limit,
        "step": args.step,
        "target_context_ratio": args.target_context_ratio,
        "available_kv_cache_gib": available_kv_cache_gib,
        "available_kv_cache_bytes_estimate": available_kv_cache_bytes,
        "max_stable_context_length": None if last_success is None else last_success["max_model_len"],
        "tokens_per_gib_estimate": (
            None
            if last_success is None or not available_kv_cache_gib
            else last_success["max_model_len"] / float(available_kv_cache_gib)
        ),
    }

    write_json(output_dir / "max_model_len_trials.json", trials)
    write_json(output_dir / "kv_cache_capacity_summary.json", summary)


if __name__ == "__main__":
    main()
