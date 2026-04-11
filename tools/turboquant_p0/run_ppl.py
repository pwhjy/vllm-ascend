import argparse
import math
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (
    add_common_llm_args,
    batch_iter,
    build_llm_kwargs,
    load_text_corpus,
    make_prompt_from_token_ids,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run P0 perplexity baseline using vLLM prompt_logprobs.")
    add_common_llm_args(parser)
    parser.add_argument("--text-file")
    parser.add_argument("--jsonl-file")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--hf-dataset")
    parser.add_argument("--hf-split", default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--separator", default="\n\n")
    parser.add_argument("--chunk-len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    from vllm import LLM, SamplingParams

    corpus, corpus_meta = load_text_corpus(
        text_file=args.text_file,
        jsonl_file=args.jsonl_file,
        text_key=args.text_key,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        max_samples=args.max_samples,
        separator=args.separator,
    )

    llm = LLM(**build_llm_kwargs(args))
    tokenizer = llm.get_tokenizer()
    model_max_len = llm.llm_engine.model_config.max_model_len
    chunk_len = min(args.chunk_len, model_max_len - 1)
    stride = min(args.stride, chunk_len)

    token_ids = tokenizer.encode(corpus)
    chunks = []
    for begin_loc in range(0, len(token_ids), stride):
        end_loc = min(begin_loc + chunk_len, len(token_ids))
        chunk = token_ids[begin_loc:end_loc]
        if len(chunk) >= 2:
            chunks.append(chunk)
        if end_loc >= len(token_ids):
            break

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=0,
        detokenize=False,
    )

    nll_sum = 0.0
    n_tokens_scored = 0
    for chunk_batch in batch_iter(chunks, args.batch_size):
        prompts = [make_prompt_from_token_ids(chunk) for chunk in chunk_batch]
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        for output in outputs:
            prompt_logprobs = output.prompt_logprobs
            if prompt_logprobs is None:
                raise RuntimeError("prompt_logprobs was not returned; cannot compute perplexity")
            if not prompt_logprobs or prompt_logprobs[0] is not None:
                raise RuntimeError("Unexpected prompt_logprobs layout")
            for token_data in prompt_logprobs[1:]:
                if token_data is None or len(token_data) != 1:
                    raise RuntimeError("Expected exactly one selected prompt token logprob")
                selected = next(iter(token_data.values()))
                nll_sum += -float(selected.logprob)
                n_tokens_scored += 1

    if n_tokens_scored == 0:
        raise RuntimeError("No prompt tokens were scored; perplexity is undefined")

    avg_nll = nll_sum / n_tokens_scored
    ppl = math.exp(avg_nll)
    write_json(
        args.output_json,
        {
            "model": args.model,
            "dtype": args.dtype,
            "corpus": corpus_meta,
            "chunk_len": chunk_len,
            "stride": stride,
            "total_corpus_tokens": len(token_ids),
            "num_chunks": len(chunks),
            "n_tokens_scored": n_tokens_scored,
            "nll_sum": nll_sum,
            "avg_nll": avg_nll,
            "perplexity": ppl,
        },
    )


if __name__ == "__main__":
    main()
