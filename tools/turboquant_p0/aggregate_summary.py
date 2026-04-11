import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import append_summary_line, write_json


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate P0 outputs into one summary.")
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--summary-md", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.result_root)

    env_summary = load_json(root / "env" / "env_summary.json")
    greedy_summary = load_json(root / "quality" / "greedy_decode_summary.json")
    ppl_summary = load_json(root / "quality" / "ppl_wikitext2.json")
    niah_summary = load_json(root / "quality" / "niah_results.json")
    longbench_summary = load_json(root / "quality" / "longbench_e_results.json")
    memory_summary = load_json(root / "memory" / "kv_cache_capacity_summary.json")
    throughput_summary = load_json(root / "performance" / "throughput.json")

    serving_files = sorted((root / "performance").glob("serve_qps_*.json"))
    serving = {file.stem: load_json(file) for file in serving_files}

    merged = {
        "env": env_summary,
        "quality": {
            "greedy_decode": greedy_summary,
            "perplexity": ppl_summary,
            "niah": niah_summary,
            "longbench_e": longbench_summary,
        },
        "memory": memory_summary,
        "performance": {
            "throughput": throughput_summary,
            "serving": serving,
        },
    }
    write_json(args.summary_json, merged)

    lines = ["# TurboQuant P0 Summary", ""]
    if env_summary:
        lines.append("## Environment")
        append_summary_line(lines, "torch", env_summary.get("versions", {}).get("torch"))
        append_summary_line(lines, "torch-npu", env_summary.get("versions", {}).get("torch-npu"))
        append_summary_line(lines, "vllm", env_summary.get("versions", {}).get("vllm"))
        append_summary_line(lines, "vllm-ascend", env_summary.get("versions", {}).get("vllm-ascend"))
        lines.append("")

    if greedy_summary:
        lines.append("## Quality")
        append_summary_line(lines, "greedy_decode.num_samples", greedy_summary.get("num_samples"))
        append_summary_line(lines, "greedy_decode.avg_prompt_tokens", greedy_summary.get("avg_prompt_tokens"))
        append_summary_line(lines, "greedy_decode.avg_generated_tokens", greedy_summary.get("avg_generated_tokens"))
    if ppl_summary:
        append_summary_line(lines, "perplexity", ppl_summary.get("perplexity"))
    if niah_summary:
        append_summary_line(lines, "niah.accuracy", niah_summary.get("accuracy"))
    if longbench_summary:
        append_summary_line(lines, "longbench_e.overall_average_score", longbench_summary.get("overall_average_score"))
    lines.append("")

    if memory_summary:
        lines.append("## Memory")
        append_summary_line(lines, "available_kv_cache_gib", memory_summary.get("available_kv_cache_gib"))
        append_summary_line(lines, "max_stable_context_length", memory_summary.get("max_stable_context_length"))
        append_summary_line(lines, "tokens_per_gib_estimate", memory_summary.get("tokens_per_gib_estimate"))
        lines.append("")

    lines.append("## Performance")
    if throughput_summary:
        append_summary_line(lines, "throughput.requests_per_second", throughput_summary.get("request_throughput"))
        append_summary_line(lines, "throughput.total_tokens_per_second", throughput_summary.get("total_token_throughput"))
        append_summary_line(lines, "throughput.output_tokens_per_second", throughput_summary.get("output_throughput"))
    for name, summary in serving.items():
        if summary is None:
            continue
        append_summary_line(lines, f"{name}.request_throughput", summary.get("request_throughput"))
        append_summary_line(lines, f"{name}.mean_ttft_ms", summary.get("mean_ttft_ms"))
        append_summary_line(lines, f"{name}.mean_itl_ms", summary.get("mean_itl_ms"))
    Path(args.summary_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
