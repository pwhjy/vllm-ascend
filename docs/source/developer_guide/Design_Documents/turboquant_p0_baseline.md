# TurboQuant P0 Baseline

This document defines the `P0` baseline that must be completed before any TurboQuant code is integrated into `vllm-ascend`.

The purpose of `P0` is simple:

- Freeze a reproducible Ascend baseline environment.
- Run one stable BF16 Llama baseline end to end.
- Produce baseline numbers for quality, KV cache memory, and performance.
- Ensure later TurboQuant results can be compared against the same model, same workload, and same runtime configuration.

`P0` does not include any TurboQuant code changes.

## 1. Scope

The `P0` baseline is intentionally narrow.

- Model family: Llama
- Recommended target model: `meta-llama/Llama-3.1-8B-Instruct`
- Execution path: `v1` model runner only
- Precision: BF16 weights + BF16 KV cache
- Deployment mode: single node, single card, `TP=1`
- Attention type: dense decoder attention only
- Non-goals:
  - no TurboQuant
  - no weight quantization
  - no MoE
  - no MLA
  - no multi-card
  - no speculative decoding
  - no prefix caching

If `meta-llama/Llama-3.1-8B-Instruct` is not available on the target machine, select one replacement Llama model and keep it fixed for the entire project. Do not switch models between `P0` and TurboQuant validation.

## 2. Output Artifacts

At the end of `P0`, the following artifacts must exist.

- Environment snapshot
- Baseline startup command
- Baseline evaluation commands
- Raw benchmark outputs
- Final baseline summary

Recommended output directory:

```text
C:\Code\vllm-workspace\vllm-ascend\benchmarks\results\turboquant_p0\
```

Recommended file layout:

```text
benchmarks/results/turboquant_p0/
|-- env/
|   |-- collect_env.txt
|   |-- npu_smi.txt
|   |-- pip_freeze.txt
|   |-- vllm_git_head.txt
|   `-- vllm_ascend_git_head.txt
|-- launch/
|   |-- offline_llm_args.txt
|   `-- serve_args.txt
|-- quality/
|   |-- greedy_decode_samples.jsonl
|   |-- ppl_wikitext2.json
|   |-- niah_results.json
|   `-- longbench_e_results.json
|-- memory/
|   |-- profile_run_log.txt
|   |-- max_model_len_trials.json
|   `-- kv_cache_capacity_summary.json
|-- performance/
|   |-- throughput.json
|   |-- serve_qps_inf.json
|   |-- serve_qps_1.json
|   |-- serve_qps_4.json
|   `-- serve_qps_16.json
`-- summary.md
```

## 3. Environment Freeze

Before any baseline run, freeze the runtime environment.

Run the following in `C:\Code\vllm-workspace\vllm-ascend` and save outputs into `benchmarks/results/turboquant_p0/env/`.

```powershell
python collect_env.py *> benchmarks/results/turboquant_p0/env/collect_env.txt
npu-smi info *> benchmarks/results/turboquant_p0/env/npu_smi.txt
pip freeze *> benchmarks/results/turboquant_p0/env/pip_freeze.txt
git -C C:\Code\vllm-workspace\vllm rev-parse HEAD *> benchmarks/results/turboquant_p0/env/vllm_git_head.txt
git -C C:\Code\vllm-workspace\vllm-ascend rev-parse HEAD *> benchmarks/results/turboquant_p0/env/vllm_ascend_git_head.txt
```

Additionally record these values in `summary.md`.

- Device model, for example `Atlas 800I A2` or `Atlas A3`
- Number of visible NPUs
- CANN version
- `torch`, `torch-npu`, `vllm`, `vllm-ascend` versions
- Driver version
- OS image

## 4. Runtime Freeze

All later TurboQuant comparisons must reuse the same runtime settings.

Freeze the following parameters now.

- `tensor_parallel_size = 1`
- `pipeline_parallel_size = 1`
- `dtype = bfloat16`
- `kv_cache_dtype = auto`
- `max_model_len = 32768` as the starting point
- `gpu_memory_utilization = 0.90`
- `enforce_eager = False`
- `enable_prefix_caching = False`
- `VLLM_USE_V2_MODEL_RUNNER = 0`

Recommended shell setup:

```powershell
$env:VLLM_USE_V2_MODEL_RUNNER="0"
$env:VLLM_ATTENTION_BACKEND=""
$env:VLLM_USE_MODELSCOPE="True"
```

Do not enable any optimization that is unlikely to survive the TurboQuant branch. `P0` must be a fair control group, not an over-tuned special case.

## 5. Baseline Launch Commands

Prepare one offline command and one serving command, then save the exact commands into:

- `benchmarks/results/turboquant_p0/launch/offline_llm_args.txt`
- `benchmarks/results/turboquant_p0/launch/serve_args.txt`

Recommended offline smoke command:

```powershell
python -c "from vllm import LLM; llm = LLM(
    model='meta-llama/Llama-3.1-8B-Instruct',
    tensor_parallel_size=1,
    dtype='bfloat16',
    max_model_len=32768,
    gpu_memory_utilization=0.90,
    quantization=None
); print(llm.generate(['Explain KV cache in one paragraph.'])[0].outputs[0].text)"
```

Recommended serving command:

```powershell
vllm serve meta-llama/Llama-3.1-8B-Instruct `
  --tensor-parallel-size 1 `
  --dtype bfloat16 `
  --max-model-len 32768 `
  --gpu-memory-utilization 0.90 `
  --disable-log-requests `
  --port 8000
```

`P0` is considered invalid if these commands are not preserved exactly.

## 6. Baseline Sanity Checks

These checks must pass before collecting any metrics.

### 6.1 Startup

- Model loads successfully.
- No fallback to `v2` model runner.
- No quantization-related warnings.
- No KV cache allocation failure.

### 6.2 Functional Smoke

Use 3 prompt classes.

- Short prompt: 32 to 64 tokens
- Medium prompt: 512 to 1024 tokens
- Long prompt: 8k to 16k tokens

For each class, verify:

- non-empty output
- no repeated-token collapse
- no runtime exception
- stable generation across two repeated runs with fixed seed

### 6.3 Long Context Smoke

At least once, run a single request close to the target context length and verify:

- prompt is accepted
- decode starts successfully
- no KV cache OOM

## 7. Quality Baseline

The goal is not to publish a leaderboard result. The goal is to capture a control group that later TurboQuant runs will be compared against.

### 7.1 Greedy Decode Regression Set

Create a small fixed prompt set of 50 to 100 prompts, stored as JSONL. The set should include:

- instruction following
- summarization
- code generation
- factual QA
- long-context retrieval prompts

For each sample, store:

- prompt text
- seed
- generated text
- generated token ids if available
- average logprob if available

Generation policy:

- `temperature = 0`
- `top_p = 1`
- `max_tokens = 256`

Output file:

- `benchmarks/results/turboquant_p0/quality/greedy_decode_samples.jsonl`

This file is the primary later comparison target for token-level and text-level diff.

### 7.2 Perplexity Baseline

Use a small stable subset of `WikiText-2` or `C4`.

Requirements:

- fixed sample count
- fixed tokenizer version
- fixed truncation length

Store:

- dataset name
- sample count
- total tokens
- average NLL
- perplexity

Output file:

- `benchmarks/results/turboquant_p0/quality/ppl_wikitext2.json`

### 7.3 Needle-in-a-Haystack Baseline

Run a small but meaningful NIAH baseline, focused on long-context retrieval.

Recommended lengths:

- 8k
- 16k
- 32k

Recommended depth settings:

- shallow
- middle
- deep

Store:

- context length
- needle depth
- exact match / pass-fail
- response text

Output file:

- `benchmarks/results/turboquant_p0/quality/niah_results.json`

This baseline is mandatory because TurboQuant mainly changes KV cache behavior, and long-context retrieval is one of the most sensitive downstream checks.

### 7.4 LongBench-E Baseline

Run a small fixed subset of `LongBench-E`, not the entire benchmark.

Recommended subset:

- one summarization task
- one retrieval task
- one multi-document QA task

Store:

- task name
- prompt count
- score
- evaluation script version

Output file:

- `benchmarks/results/turboquant_p0/quality/longbench_e_results.json`

## 8. Memory Baseline

The memory baseline is required because later TurboQuant claims must be compared to actual KV cache capacity, not nominal bit width.

### 8.1 Profile Run Snapshot

Save the model profiling log used to determine available KV cache memory.

This information is generated by `NPUWorker.determine_available_memory()` and `NPUModelRunner.profile_run()`.

Required fields:

- total requested memory
- non-KV memory
- available KV cache memory

Output file:

- `benchmarks/results/turboquant_p0/memory/profile_run_log.txt`

### 8.2 Max Context Capacity Trial

Measure the real maximum usable context length under the frozen runtime settings.

Recommended procedure:

1. Start from `max_model_len = 32768`
2. Increase in steps of `4096`
3. Stop at first stable failure
4. Repeat the last success once more

Store for each trial:

- `max_model_len`
- startup success or failure
- first-token generation success or failure
- failure type

Output file:

- `benchmarks/results/turboquant_p0/memory/max_model_len_trials.json`

### 8.3 KV Cache Capacity Summary

Summarize the usable BF16 KV capacity in practical terms.

Required summary fields:

- model name
- block size
- KV cache dtype
- available KV cache memory bytes
- max stable context length
- estimated tokens per GiB of KV cache

Output file:

- `benchmarks/results/turboquant_p0/memory/kv_cache_capacity_summary.json`

## 9. Performance Baseline

Use the existing `vllm bench` scripts so later TurboQuant numbers are comparable.

Install benchmark dependencies if needed:

```powershell
pip install -r benchmarks/requirements-bench.txt
```

### 9.1 Offline Throughput

Run one offline throughput benchmark.

Recommended command:

```powershell
vllm bench throughput `
  --model meta-llama/Llama-3.1-8B-Instruct `
  --backend vllm `
  --dataset-name random `
  --input-len 1024 `
  --output-len 256 `
  --tensor-parallel-size 1 `
  --dtype bfloat16 `
  --max-model-len 32768 `
  --output-json benchmarks/results/turboquant_p0/performance/throughput.json
```

Store:

- requests/s
- total tokens/s
- output tokens/s

### 9.2 Online Serving

Use the same server process for all serving benchmarks.

Recommended request-rate set:

- `1`
- `4`
- `16`
- `inf`

Recommended dataset:

- `sharegpt` if available
- otherwise `random`

Example:

```powershell
vllm bench serve `
  --backend vllm `
  --model meta-llama/Llama-3.1-8B-Instruct `
  --endpoint /v1/completions `
  --dataset-name random `
  --input-len 1024 `
  --output-len 256 `
  --num-prompts 100 `
  --request-rate 4 `
  --save-result `
  --result-filename benchmarks/results/turboquant_p0/performance/serve_qps_4.json
```

For each QPS, store:

- successful requests
- failed requests
- request throughput
- output token throughput
- total token throughput
- mean / median / p99 TTFT
- mean / median / p99 ITL

## 10. Acceptance Criteria

`P0` is complete only if all items below are satisfied.

- Environment is frozen and archived.
- One BF16 Llama model runs stably on Ascend with `v1`.
- A fixed greedy decode regression set exists.
- A fixed perplexity baseline exists.
- A fixed NIAH baseline exists.
- A fixed LongBench-E subset baseline exists.
- Available KV cache memory has been measured.
- Maximum stable context length has been measured.
- Offline throughput baseline exists.
- Online serving baseline exists for at least 4 request rates.
- All commands and outputs are stored under `benchmarks/results/turboquant_p0/`.

## 11. What Not To Change After P0

After `P0` is complete, the following must remain unchanged when comparing TurboQuant against the baseline.

- model
- tokenizer
- prompt set
- runtime environment
- `vllm` commit
- `vllm-ascend` commit, unless the TurboQuant branch is rebased from this exact baseline
- serving arguments
- benchmark dataset and sample count
- evaluation scripts

If any of these changes, the old baseline is invalid and `P0` must be rerun.

## 12. Recommended Next Step

Once `P0` is complete, the next milestone is `P1`.

`P1` should introduce only the TurboQuant reference path for KV cache, with no fused kernel optimization. The goal of `P1` is to answer:

- Does quality remain nearly unchanged?
- Is KV cache memory actually reduced in the real allocator?
- What is the overhead of the reference decode path?
