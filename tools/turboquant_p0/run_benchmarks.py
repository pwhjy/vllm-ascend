import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import ensure_dir, parse_request_rates, wait_for_http_ready


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P0 throughput and serving benchmarks.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["throughput", "serve", "all"], default="all")
    parser.add_argument("--dataset-name", default="random")
    parser.add_argument("--dataset-path")
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--request-rates", default="1,4,16,inf")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--startup-timeout-sec", type=int, default=300)
    return parser


def run_and_log(command: list[str], log_file: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, capture_output=True, env=env, check=False)
    lines = [f"$ {' '.join(command)}", ""]
    if result.stdout:
        lines.extend(["[stdout]", result.stdout.rstrip(), ""])
    if result.stderr:
        lines.extend(["[stderr]", result.stderr.rstrip(), ""])
    lines.append(f"[returncode] {result.returncode}")
    log_file.write_text("\n".join(lines), encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")
    return result


def build_dataset_args(args: argparse.Namespace) -> list[str]:
    dataset_args = ["--dataset-name", args.dataset_name]
    if args.dataset_path:
        dataset_args.extend(["--dataset-path", args.dataset_path])
    elif args.dataset_name == "random":
        dataset_args.extend(["--input-len", str(args.input_len), "--output-len", str(args.output_len)])
    return dataset_args


def main() -> None:
    args = build_parser().parse_args()
    output_dir = ensure_dir(args.output_dir)
    logs_dir = ensure_dir(output_dir / "logs")

    env = dict(os.environ)
    env.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")

    bench_base = [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench"]
    dataset_args = build_dataset_args(args)

    if args.mode in ("throughput", "all"):
        throughput_json = output_dir / "throughput.json"
        throughput_log = logs_dir / "throughput.log"
        command = bench_base + [
            "throughput",
            "--model",
            args.model,
            "--dtype",
            args.dtype,
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--max-model-len",
            str(args.max_model_len),
            "--output-json",
            str(throughput_json),
        ] + dataset_args + ["--num-prompts", str(args.num_prompts)]
        if args.tokenizer:
            command.extend(["--tokenizer", args.tokenizer])
        if args.quantization not in (None, "", "none", "None"):
            command.extend(["--quantization", args.quantization])
        run_and_log(command, throughput_log, env)

    if args.mode in ("serve", "all"):
        server_command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            args.model,
            "--dtype",
            args.dtype,
            "--tensor-parallel-size",
            str(args.tensor_parallel_size),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--disable-log-requests",
            "--port",
            str(args.server_port),
            "--host",
            args.server_host,
        ]
        if args.tokenizer:
            server_command.extend(["--tokenizer", args.tokenizer])
        if args.quantization not in (None, "", "none", "None"):
            server_command.extend(["--quantization", args.quantization])
        server_log = logs_dir / "server.log"
        server_proc = subprocess.Popen(
            server_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        try:
            wait_for_http_ready(args.server_host, args.server_port, timeout_sec=args.startup_timeout_sec)
            for request_rate in parse_request_rates(args.request_rates):
                bench_json = output_dir / f"serve_qps_{request_rate}.json"
                bench_log = logs_dir / f"serve_qps_{request_rate}.log"
                command = bench_base + [
                    "serve",
                    "--backend",
                    "vllm",
                    "--model",
                    args.model,
                    "--served-model-name",
                    args.model,
                    "--host",
                    args.server_host,
                    "--port",
                    str(args.server_port),
                    "--endpoint",
                    "/v1/completions",
                    "--num-prompts",
                    str(args.num_prompts),
                    "--request-rate",
                    request_rate,
                    "--save-result",
                    "--result-filename",
                    str(bench_json),
                ] + dataset_args
                if args.tokenizer:
                    command.extend(["--tokenizer", args.tokenizer])
                if args.quantization not in (None, "", "none", "None"):
                    command.extend(["--quantization", args.quantization])
                run_and_log(command, bench_log, env)
        finally:
            if server_proc.stdout is not None:
                output = server_proc.stdout.read()
                server_log.write_text(output, encoding="utf-8")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait(timeout=30)
            time.sleep(1)


if __name__ == "__main__":
    main()
