import argparse
import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import ensure_dir, parse_request_rates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TurboQuant P0 baseline suite.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--swap-space", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--disable-log-stats", action="store_true", default=True)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--disable-chunked-prefill", action="store_true", default=False)
    parser.add_argument("--result-root", required=True)
    parser.add_argument(
        "--stages",
        default="env,greedy,ppl,niah,memory,throughput,aggregate",
        help="Comma-separated stages: env,greedy,ppl,niah,longbench,memory,throughput,serve,aggregate",
    )
    parser.add_argument("--greedy-prompts-jsonl", default=str(THIS_DIR / "data" / "greedy_prompts.jsonl"))
    parser.add_argument("--greedy-max-tokens", type=int, default=256)
    parser.add_argument("--greedy-logprobs", type=int)
    parser.add_argument("--ppl-text-file")
    parser.add_argument("--ppl-jsonl-file")
    parser.add_argument("--ppl-text-key", default="text")
    parser.add_argument("--ppl-hf-dataset")
    parser.add_argument("--ppl-hf-split", default="test")
    parser.add_argument("--ppl-max-samples", type=int)
    parser.add_argument("--ppl-chunk-len", type=int, default=1024)
    parser.add_argument("--ppl-stride", type=int, default=1024)
    parser.add_argument("--ppl-batch-size", type=int, default=8)
    parser.add_argument("--niah-lengths", default="8192,16384,32768")
    parser.add_argument("--niah-depths", default="0.1,0.5,0.9")
    parser.add_argument("--niah-max-tokens", type=int, default=16)
    parser.add_argument("--longbench-jsonl")
    parser.add_argument(
        "--longbench-default-metric",
        default="exact_match",
        choices=["exact_match", "contains", "token_f1", "rouge_l_f1"],
    )
    parser.add_argument("--longbench-max-tokens", type=int, default=256)
    parser.add_argument("--memory-start-max-model-len", type=int, default=32768)
    parser.add_argument("--memory-max-model-len-limit", type=int, default=65536)
    parser.add_argument("--memory-step", type=int, default=4096)
    parser.add_argument("--memory-target-context-ratio", type=float, default=0.90)
    parser.add_argument("--memory-generate-tokens", type=int, default=1)
    parser.add_argument("--bench-dataset-name", default="random")
    parser.add_argument("--bench-dataset-path")
    parser.add_argument("--bench-input-len", type=int, default=1024)
    parser.add_argument("--bench-output-len", type=int, default=256)
    parser.add_argument("--bench-num-prompts", type=int, default=100)
    parser.add_argument("--serve-request-rates", default="1,4,16,inf")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--startup-timeout-sec", type=int, default=300)
    return parser


def run_step(command: list[str], env: dict[str, str]) -> None:
    result = subprocess.run(command, cwd=str(THIS_DIR.parent.parent), env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}: {' '.join(command)}")


def append_common_llm_flags(args: argparse.Namespace) -> list[str]:
    flags = [
        "--model",
        args.model,
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--swap-space",
        str(args.swap_space),
        "--block-size",
        str(args.block_size),
        "--seed",
        str(args.seed),
    ]
    if args.tokenizer:
        flags.extend(["--tokenizer", args.tokenizer])
    if args.quantization not in (None, "", "none", "None"):
        flags.extend(["--quantization", args.quantization])
    if args.disable_log_stats:
        flags.append("--disable-log-stats")
    if args.trust_remote_code:
        flags.append("--trust-remote-code")
    if args.enforce_eager:
        flags.append("--enforce-eager")
    if args.disable_chunked_prefill:
        flags.append("--disable-chunked-prefill")
    return flags


def main() -> None:
    args = build_parser().parse_args()
    stages = {item.strip() for item in args.stages.split(",") if item.strip()}

    result_root = ensure_dir(args.result_root)
    env_dir = ensure_dir(result_root / "env")
    quality_dir = ensure_dir(result_root / "quality")
    memory_dir = ensure_dir(result_root / "memory")
    performance_dir = ensure_dir(result_root / "performance")

    env = dict(os.environ)
    env.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    python_exec = sys.executable
    common_flags = append_common_llm_flags(args)

    if "env" in stages:
        run_step(
            [
                python_exec,
                str(THIS_DIR / "collect_env.py"),
                "--output-dir",
                str(env_dir),
            ],
            env,
        )

    if "greedy" in stages:
        run_step(
            [
                python_exec,
                str(THIS_DIR / "run_greedy_decode.py"),
                *common_flags,
                "--prompts-jsonl",
                args.greedy_prompts_jsonl,
                "--max-tokens",
                str(args.greedy_max_tokens),
                "--output-jsonl",
                str(quality_dir / "greedy_decode_outputs.jsonl"),
                "--summary-json",
                str(quality_dir / "greedy_decode_summary.json"),
            ]
            + ([] if args.greedy_logprobs is None else ["--logprobs", str(args.greedy_logprobs)]),
            env,
        )

    if "ppl" in stages:
        if not any([args.ppl_text_file, args.ppl_jsonl_file, args.ppl_hf_dataset]):
            raise ValueError(
                "One of --ppl-text-file, --ppl-jsonl-file, or --ppl-hf-dataset is required when stage 'ppl' is enabled"
            )
        ppl_command = [
            python_exec,
            str(THIS_DIR / "run_ppl.py"),
            *common_flags,
        ]
        if args.ppl_text_file:
            ppl_command.extend(["--text-file", args.ppl_text_file])
        if args.ppl_jsonl_file:
            ppl_command.extend(["--jsonl-file", args.ppl_jsonl_file])
        if args.ppl_hf_dataset:
            ppl_command.extend(["--hf-dataset", args.ppl_hf_dataset, "--hf-split", args.ppl_hf_split])
        if args.ppl_max_samples is not None:
            ppl_command.extend(["--max-samples", str(args.ppl_max_samples)])
        ppl_command.extend(
            [
                "--text-key",
                args.ppl_text_key,
                "--chunk-len",
                str(args.ppl_chunk_len),
                "--stride",
                str(args.ppl_stride),
                "--batch-size",
                str(args.ppl_batch_size),
                "--output-json",
                str(quality_dir / "ppl_wikitext2.json"),
            ]
        )
        run_step(ppl_command, env)

    if "niah" in stages:
        run_step(
            [
                python_exec,
                str(THIS_DIR / "run_niah.py"),
                *common_flags,
                "--lengths",
                args.niah_lengths,
                "--depths",
                args.niah_depths,
                "--max-tokens",
                str(args.niah_max_tokens),
                "--output-json",
                str(quality_dir / "niah_results.json"),
            ],
            env,
        )

    if "longbench" in stages:
        if not args.longbench_jsonl:
            raise ValueError("--longbench-jsonl is required when stage 'longbench' is enabled")
        run_step(
            [
                python_exec,
                str(THIS_DIR / "run_prompt_dataset_eval.py"),
                *common_flags,
                "--dataset-jsonl",
                args.longbench_jsonl,
                "--default-metric",
                args.longbench_default_metric,
                "--max-tokens",
                str(args.longbench_max_tokens),
                "--per-sample-jsonl",
                str(quality_dir / "longbench_e_per_sample.jsonl"),
                "--summary-json",
                str(quality_dir / "longbench_e_results.json"),
            ],
            env,
        )

    if "memory" in stages:
        run_step(
            [
                python_exec,
                str(THIS_DIR / "run_memory_baseline.py"),
                *common_flags,
                "--output-dir",
                str(memory_dir),
                "--start-max-model-len",
                str(args.memory_start_max_model_len),
                "--max-model-len-limit",
                str(args.memory_max_model_len_limit),
                "--step",
                str(args.memory_step),
                "--target-context-ratio",
                str(args.memory_target_context_ratio),
                "--generate-tokens",
                str(args.memory_generate_tokens),
            ],
            env,
        )

    if "throughput" in stages or "serve" in stages:
        bench_mode = "all" if {"throughput", "serve"} <= stages else ("serve" if "serve" in stages else "throughput")
        bench_command = [
            python_exec,
            str(THIS_DIR / "run_benchmarks.py"),
            "--model",
            args.model,
            "--dtype",
            args.dtype,
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--output-dir",
            str(performance_dir),
            "--mode",
            bench_mode,
            "--dataset-name",
            args.bench_dataset_name,
            "--input-len",
            str(args.bench_input_len),
            "--output-len",
            str(args.bench_output_len),
            "--num-prompts",
            str(args.bench_num_prompts),
            "--request-rates",
            ",".join(parse_request_rates(args.serve_request_rates)),
            "--server-host",
            args.server_host,
            "--server-port",
            str(args.server_port),
            "--startup-timeout-sec",
            str(args.startup_timeout_sec),
        ]
        if args.tokenizer:
            bench_command.extend(["--tokenizer", args.tokenizer])
        if args.quantization not in (None, "", "none", "None"):
            bench_command.extend(["--quantization", args.quantization])
        if args.bench_dataset_path:
            bench_command.extend(["--dataset-path", args.bench_dataset_path])
        run_step(bench_command, env)

    if "aggregate" in stages:
        run_step(
            [
                python_exec,
                str(THIS_DIR / "aggregate_summary.py"),
                "--result-root",
                str(result_root),
                "--summary-json",
                str(result_root / "summary.json"),
                "--summary-md",
                str(result_root / "summary.md"),
            ],
            env,
        )


if __name__ == "__main__":
    main()
