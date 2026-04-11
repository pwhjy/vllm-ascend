import argparse
import statistics
import sys
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
    serialize_logprob_dicts,
    sha256_text,
    write_json,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P0 greedy decode baseline and save outputs.")
    add_common_llm_args(parser)
    parser.add_argument(
        "--prompts-jsonl",
        default=str(THIS_DIR / "data" / "greedy_prompts.jsonl"),
        help="JSONL file with fields: id, prompt, optional max_tokens, optional seed, optional metadata.",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--logprobs", type=int, default=None)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from vllm import LLM, SamplingParams

    rows = load_jsonl(args.prompts_jsonl)
    llm = LLM(**build_llm_kwargs(args))

    outputs: list[dict[str, Any]] = []
    output_hashes: list[str] = []
    prompt_token_counts: list[int] = []
    generated_token_counts: list[int] = []

    for index, row in enumerate(rows):
        prompt = str(row["prompt"])
        sample_seed = int(row.get("seed", args.seed if args.seed is not None else DEFAULT_SEED))
        sample_max_tokens = int(row.get("max_tokens", args.max_tokens))
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            max_tokens=sample_max_tokens,
            seed=sample_seed,
            logprobs=args.logprobs,
        )
        request_outputs = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
        request_output = request_outputs[0]
        sample = request_output.outputs[0]

        prompt_token_count = len(request_output.prompt_token_ids)
        generated_token_count = len(sample.token_ids)
        prompt_token_counts.append(prompt_token_count)
        generated_token_counts.append(generated_token_count)
        output_hash = sha256_text(sample.text)
        output_hashes.append(output_hash)

        avg_logprob = None
        if sample.logprobs:
            chosen_logprobs = []
            for token_id, token_data in zip(sample.token_ids, sample.logprobs):
                if token_data is None or token_id not in token_data:
                    continue
                chosen_logprobs.append(float(token_data[token_id].logprob))
            if chosen_logprobs:
                avg_logprob = sum(chosen_logprobs) / len(chosen_logprobs)

        outputs.append(
            {
                "index": index,
                "id": row.get("id", f"sample_{index:04d}"),
                "prompt": prompt,
                "prompt_token_count": prompt_token_count,
                "generated_text": sample.text,
                "generated_token_ids": list(sample.token_ids),
                "generated_token_count": generated_token_count,
                "finish_reason": sample.finish_reason,
                "seed": sample_seed,
                "max_tokens": sample_max_tokens,
                "avg_output_logprob": avg_logprob,
                "output_sha256": output_hash,
                "metadata": row.get("metadata"),
                "output_logprobs": serialize_logprob_dicts(sample.logprobs),
            }
        )

    write_jsonl(args.output_jsonl, outputs)
    write_json(
        args.summary_json,
        {
            "model": args.model,
            "num_samples": len(outputs),
            "prompts_jsonl": args.prompts_jsonl,
            "avg_prompt_tokens": statistics.mean(prompt_token_counts) if prompt_token_counts else 0.0,
            "avg_generated_tokens": statistics.mean(generated_token_counts) if generated_token_counts else 0.0,
            "min_generated_tokens": min(generated_token_counts) if generated_token_counts else 0,
            "max_generated_tokens": max(generated_token_counts) if generated_token_counts else 0,
            "combined_output_sha256": sha256_text("".join(output_hashes)),
        },
    )


if __name__ == "__main__":
    main()
