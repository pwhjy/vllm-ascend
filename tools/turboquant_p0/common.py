import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.error import URLError
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED = 1234


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, data: Any) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def collapse_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def normalize_text(text: str) -> str:
    return collapse_whitespace(text).lower()


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for token_a in a:
        prev = 0
        for idx, token_b in enumerate(b, start=1):
            cur = dp[idx]
            if token_a == token_b:
                dp[idx] = prev + 1
            else:
                dp[idx] = max(dp[idx], dp[idx - 1])
            prev = cur
    return dp[-1]


def exact_match_score(prediction: str, references: list[str]) -> float:
    pred = normalize_text(prediction)
    return float(any(pred == normalize_text(ref) for ref in references))


def contains_score(prediction: str, references: list[str]) -> float:
    pred = normalize_text(prediction)
    return float(any(normalize_text(ref) in pred for ref in references))


def token_f1_score(prediction: str, references: list[str]) -> float:
    pred_tokens = normalize_text(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for ref in references:
        ref_tokens = normalize_text(ref).split()
        if not ref_tokens:
            continue
        common: dict[str, int] = {}
        for token in pred_tokens:
            common[token] = common.get(token, 0) + 1
        overlap = 0
        for token in ref_tokens:
            count = common.get(token, 0)
            if count > 0:
                overlap += 1
                common[token] = count - 1
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def rouge_l_f1_score(prediction: str, references: list[str]) -> float:
    pred_tokens = normalize_text(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for ref in references:
        ref_tokens = normalize_text(ref).split()
        if not ref_tokens:
            continue
        lcs = lcs_length(pred_tokens, ref_tokens)
        if lcs == 0:
            continue
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


METRIC_REGISTRY = {
    "exact_match": exact_match_score,
    "contains": contains_score,
    "token_f1": token_f1_score,
    "rouge_l_f1": rouge_l_f1_score,
}


def score_prediction(prediction: str, references: list[str], metric: str) -> float:
    if metric not in METRIC_REGISTRY:
        raise ValueError(f"Unsupported metric: {metric}")
    return float(METRIC_REGISTRY[metric](prediction, references))


def add_common_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Model name or local model path.")
    parser.add_argument("--tokenizer", help="Optional tokenizer path.")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--swap-space", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--disable-log-stats", action="store_true", default=True)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--disable-chunked-prefill", action="store_true", default=False)


def build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "swap_space": args.swap_space,
        "block_size": args.block_size,
        "disable_log_stats": args.disable_log_stats,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
        "enable_chunked_prefill": not args.disable_chunked_prefill,
    }
    if args.tokenizer:
        kwargs["tokenizer"] = args.tokenizer
    if args.quantization not in (None, "", "none", "None"):
        kwargs["quantization"] = args.quantization
    return kwargs


def make_llm(args: argparse.Namespace):
    from vllm import LLM

    return LLM(**build_llm_kwargs(args))


def serialize_logprob_dicts(logprob_entries: Any) -> Any:
    if logprob_entries is None:
        return None
    serialized: list[Any] = []
    for entry in logprob_entries:
        if entry is None:
            serialized.append(None)
            continue
        row: dict[str, Any] = {}
        for token_id, token_logprob in entry.items():
            row[str(token_id)] = {
                "logprob": float(token_logprob.logprob),
                "rank": None if getattr(token_logprob, "rank", None) is None else int(token_logprob.rank),
                "decoded_token": getattr(token_logprob, "decoded_token", None),
            }
        serialized.append(row)
    return serialized


def make_prompt_from_token_ids(token_ids: list[int]) -> dict[str, Any]:
    return {"prompt_token_ids": token_ids}


def batch_iter(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def run_subprocess(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout_sec: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )


def wait_for_http_ready(
    host: str,
    port: int,
    timeout_sec: int = 300,
    candidate_paths: tuple[str, ...] = ("/health", "/v1/models"),
) -> None:
    deadline = time.time() + timeout_sec
    last_error = "service did not become ready"
    while time.time() < deadline:
        for path in candidate_paths:
            url = f"http://{host}:{port}{path}"
            try:
                with urlopen(url, timeout=5) as response:
                    if response.status < 500:
                        return
            except URLError as exc:
                last_error = str(exc)
            except Exception as exc:  # pragma: no cover
                last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for server readiness: {last_error}")


def parse_request_rates(value: str) -> list[str]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        raise ValueError("request rates cannot be empty")
    return parts


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")


def maybe_run_command_to_file(
    command: list[str],
    output_file: str | Path,
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
    timeout_sec: int | None = None,
) -> subprocess.CompletedProcess[str]:
    result = run_subprocess(command, cwd=cwd, env=env, timeout_sec=timeout_sec)
    text = []
    text.append(f"$ {' '.join(command)}")
    text.append("")
    if result.stdout:
        text.append("[stdout]")
        text.append(result.stdout.rstrip())
        text.append("")
    if result.stderr:
        text.append("[stderr]")
        text.append(result.stderr.rstrip())
        text.append("")
    text.append(f"[returncode] {result.returncode}")
    ensure_parent(output_file).write_text("\n".join(text), encoding="utf-8")
    return result


def load_text_corpus(
    *,
    text_file: str | None,
    jsonl_file: str | None,
    text_key: str,
    hf_dataset: str | None,
    hf_split: str,
    max_samples: int | None,
    separator: str,
) -> tuple[str, dict[str, Any]]:
    if text_file:
        corpus = Path(text_file).read_text(encoding="utf-8")
        return corpus, {"source": "text_file", "path": text_file}

    if jsonl_file:
        rows = load_jsonl(jsonl_file)
        if max_samples is not None:
            rows = rows[:max_samples]
        texts = [str(row[text_key]) for row in rows if text_key in row]
        return separator.join(texts), {
            "source": "jsonl_file",
            "path": jsonl_file,
            "text_key": text_key,
            "num_records": len(texts),
        }

    if hf_dataset:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("datasets is required when using --hf-dataset") from exc
        dataset = load_dataset(hf_dataset, split=hf_split)
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        texts = [str(item[text_key]) for item in dataset if text_key in item]
        return separator.join(texts), {
            "source": "hf_dataset",
            "dataset": hf_dataset,
            "split": hf_split,
            "text_key": text_key,
            "num_records": len(texts),
        }

    raise ValueError("One of --text-file, --jsonl-file, or --hf-dataset must be provided.")


def build_repeated_token_sequence(
    tokenizer: Any,
    *,
    target_token_count: int,
    unit_text: str,
) -> list[int]:
    unit_tokens = tokenizer.encode(unit_text, add_special_tokens=False)
    if not unit_tokens:
        raise ValueError("unit_text produced no tokens")
    repeats = math.ceil(target_token_count / len(unit_tokens)) + 1
    return (unit_tokens * repeats)[:target_token_count]


def append_summary_line(summary_lines: list[str], title: str, value: Any) -> None:
    summary_lines.append(f"- {title}: {value}")


def detect_version(package_name: str) -> str | None:
    try:
        if sys.version_info >= (3, 8):
            from importlib import metadata
        else:  # pragma: no cover
            import importlib_metadata as metadata  # type: ignore
        return metadata.version(package_name)
    except Exception:
        return None


def which(executable: str) -> str | None:
    return shutil.which(executable)
