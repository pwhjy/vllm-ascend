# TurboQuant P0 Scripts

This directory contains the P0 baseline scripts for later TurboQuant comparison.

## Outputs

The scripts are designed to produce a stable result tree like this:

```text
<result-root>/
  env/
  quality/
  memory/
  performance/
  summary.json
  summary.md
```

Key comparison files:

- `env/env_summary.json`
- `quality/greedy_decode_summary.json`
- `quality/ppl_wikitext2.json`
- `quality/niah_results.json`
- `quality/longbench_e_results.json`
- `memory/kv_cache_capacity_summary.json`
- `performance/throughput.json`
- `performance/serve_qps_*.json`
- `summary.json`
- `summary.md`

## One-shot run

```powershell
py -3 tools/turboquant_p0/run_p0_suite.py `
  --model meta-llama/Llama-3.1-8B-Instruct `
  --result-root benchmarks/results/turboquant_p0/llama31_8b_bf16 `
  --ppl-hf-dataset wikitext `
  --ppl-hf-split test `
  --ppl-text-key text `
  --stages env,greedy,ppl,niah,memory,throughput,aggregate
```

Add `longbench` to `--stages` and provide `--longbench-jsonl` if you want dataset scoring.

## Per-script purpose

- `collect_env.py`: freeze environment and version info
- `run_greedy_decode.py`: deterministic output baseline
- `run_ppl.py`: perplexity baseline using prompt logprobs
- `run_niah.py`: long-context retrieval baseline
- `run_prompt_dataset_eval.py`: generic prompt dataset scoring
- `run_memory_baseline.py`: max stable context and KV capacity estimate
- `run_benchmarks.py`: throughput / serving benchmarks
- `aggregate_summary.py`: merge all outputs into one summary
- `run_p0_suite.py`: orchestrate the full P0 run
