import argparse
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
    build_repeated_token_sequence,
    make_prompt_from_token_ids,
    normalize_text,
    parse_request_rates,
    write_json,
)


def parse_float_list(value: str) -> list[float]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if not parts:
        raise ValueError("value cannot be empty")
    return [float(item) for item in parts]


def build_case_prompt(tokenizer: Any, *, target_tokens: int, depth: float, answer: str) -> list[int]:
    prefix = (
        "You will read a long context. A fact appears exactly once in the context.\n"
        "Read carefully and answer the question at the end.\n\nContext:\n"
    )
    suffix = (
        "\n\nQuestion: What is the secret code mentioned in the context?\n"
        "Instruction: Reply with only the secret code.\nAnswer:"
    )
    needle = f"\nThe secret code is {answer}.\n"
    filler = (
        "This is background text for a long context retrieval test. "
        "It contains no useful answer and only serves as filler. "
    )

    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    needle_tokens = tokenizer.encode(needle, add_special_tokens=False)

    body_target = target_tokens - len(prefix_tokens) - len(suffix_tokens)
    if body_target <= len(needle_tokens) + 8:
        raise ValueError(
            f"target_tokens={target_tokens} is too small for the NIAH template; "
            f"minimum needed is {len(prefix_tokens) + len(suffix_tokens) + len(needle_tokens) + 8}"
        )

    filler_tokens = build_repeated_token_sequence(
        tokenizer,
        target_token_count=body_target - len(needle_tokens),
        unit_text=filler,
    )
    insert_at = int((body_target - len(needle_tokens)) * depth)
    body_tokens = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    body_tokens = body_tokens[:body_target]
    return prefix_tokens + body_tokens + suffix_tokens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a lightweight Needle-in-a-Haystack baseline.")
    add_common_llm_args(parser)
    parser.add_argument("--lengths", default="8192,16384,32768")
    parser.add_argument("--depths", default="0.1,0.5,0.9")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from vllm import LLM, SamplingParams

    lengths = [int(item) for item in parse_request_rates(args.lengths)]
    depths = parse_float_list(args.depths)
    for depth in depths:
        if not 0.0 <= depth <= 1.0:
            raise ValueError(f"Depth must be within [0, 1], got {depth}")

    llm = LLM(**build_llm_kwargs(args))
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        max_tokens=args.max_tokens,
        seed=args.seed if args.seed is not None else DEFAULT_SEED,
    )

    cases: list[dict[str, Any]] = []
    aggregate_by_length: dict[str, dict[str, Any]] = {}
    total_pass = 0

    for length in lengths:
        length_key = str(length)
        aggregate_by_length[length_key] = {"num_cases": 0, "num_pass": 0, "accuracy": 0.0}
        for depth in depths:
            answer = f"TQ-{length}-{int(depth * 1000):03d}"
            prompt_token_ids = build_case_prompt(tokenizer, target_tokens=length, depth=depth, answer=answer)
            outputs = llm.generate(
                [make_prompt_from_token_ids(prompt_token_ids)],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            output = outputs[0].outputs[0]
            prediction = output.text.strip()
            passed = normalize_text(prediction) == normalize_text(answer)
            total_pass += int(passed)
            aggregate_by_length[length_key]["num_cases"] += 1
            aggregate_by_length[length_key]["num_pass"] += int(passed)
            cases.append(
                {
                    "context_length": length,
                    "depth": depth,
                    "answer": answer,
                    "prediction": prediction,
                    "passed": passed,
                    "prompt_token_count": len(prompt_token_ids),
                    "generated_token_count": len(output.token_ids),
                    "finish_reason": output.finish_reason,
                }
            )

    for stats in aggregate_by_length.values():
        if stats["num_cases"]:
            stats["accuracy"] = stats["num_pass"] / stats["num_cases"]

    write_json(
        args.output_json,
        {
            "model": args.model,
            "lengths": lengths,
            "depths": depths,
            "num_cases": len(cases),
            "num_pass": total_pass,
            "accuracy": total_pass / len(cases) if cases else 0.0,
            "by_length": aggregate_by_length,
            "cases": cases,
        },
    )


if __name__ == "__main__":
    main()
