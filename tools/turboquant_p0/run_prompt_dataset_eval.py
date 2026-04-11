import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (
    DEFAULT_SEED,
    add_common_llm_args,
    build_llm_kwargs,
    load_jsonl,
    score_prediction,
    write_json,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a prompt dataset and score generated outputs.")
    add_common_llm_args(parser)
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument(
        "--default-metric",
        default="exact_match",
        choices=["exact_match", "contains", "token_f1", "rouge_l_f1"],
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--per-sample-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from vllm import LLM, SamplingParams

    dataset = load_jsonl(args.dataset_jsonl)
    llm = LLM(**build_llm_kwargs(args))

    per_sample: list[dict[str, Any]] = []
    per_metric_scores: dict[str, list[float]] = defaultdict(list)
    per_task_scores: dict[str, list[float]] = defaultdict(list)

    for index, row in enumerate(dataset):
        prompt = str(row["prompt"])
        references = row.get("references")
        if references is None:
            reference = row.get("reference")
            if reference is None:
                references = []
            elif isinstance(reference, list):
                references = [str(item) for item in reference]
            else:
                references = [str(reference)]
        else:
            references = [str(item) for item in references]

        metric = str(row.get("metric", args.default_metric))
        task = str(row.get("task", "default"))
        sample_seed = int(row.get("seed", args.seed if args.seed is not None else DEFAULT_SEED))
        sample_max_tokens = int(row.get("max_tokens", args.max_tokens))

        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=sample_max_tokens,
            seed=sample_seed,
        )
        outputs = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
        sample = outputs[0].outputs[0]
        score = score_prediction(sample.text, references, metric) if references else None

        if score is not None:
            per_metric_scores[metric].append(score)
            per_task_scores[task].append(score)

        per_sample.append(
            {
                "index": index,
                "id": row.get("id", f"sample_{index:04d}"),
                "task": task,
                "metric": metric,
                "prompt": prompt,
                "references": references,
                "generated_text": sample.text,
                "generated_token_ids": list(sample.token_ids),
                "generated_token_count": len(sample.token_ids),
                "finish_reason": sample.finish_reason,
                "score": score,
                "metadata": row.get("metadata"),
            }
        )

    write_jsonl(args.per_sample_jsonl, per_sample)
    write_json(
        args.summary_json,
        {
            "model": args.model,
            "dataset_jsonl": args.dataset_jsonl,
            "num_samples": len(per_sample),
            "num_scored_samples": sum(1 for item in per_sample if item["score"] is not None),
            "overall_average_score": (
                statistics.mean([item["score"] for item in per_sample if item["score"] is not None])
                if any(item["score"] is not None for item in per_sample)
                else None
            ),
            "by_metric": {
                metric: {
                    "num_samples": len(scores),
                    "average_score": statistics.mean(scores) if scores else None,
                }
                for metric, scores in per_metric_scores.items()
            },
            "by_task": {
                task: {
                    "num_samples": len(scores),
                    "average_score": statistics.mean(scores) if scores else None,
                }
                for task, scores in per_task_scores.items()
            },
        },
    )


if __name__ == "__main__":
    main()
