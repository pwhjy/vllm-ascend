# vllm-Ascend TurboQuant 实现总结

版本：2026-05-04

## 1. 本次实现的范围

本次实现按 `vllm_ascend_turboquant_dev_doc.md` 的 MVP 路线落地，目标是先把 `v1` 上的 **TurboQuant KV cache write/read/dequant** 路径接通，优先保证数值正确，不追求第一版性能最优。

实现覆盖：

- `TurboQuant` 量化 scheme 入口
- `modelslim_config.py` 中的 TurboQuant special-case
- `attention_v1.py` 中的 TurboQuant backend
- `model_runner_v1.py` 中的 TurboQuant sidecar KV cache 分配/reshape
- 最小的纯 Python runtime 和单元测试

没有覆盖：

- 压缩域直接 attention
- 自定义 CANN 融合算子
- 完整的 2.5-bit / 3.5-bit outlier 分裂策略
- 上游 vLLM 的 `KVQuantMode` / `KVCacheSpec` 正式扩展

## 2. 新增与修改的文件

### 2.1 新增文件

- `vllm_ascend/quantization/methods/turboquant_layout.py`
- `vllm_ascend/quantization/methods/turboquant_runtime.py`
- `vllm_ascend/quantization/methods/kv_turboquant.py`
- `tests/ut/quantization/test_turboquant_runtime.py`
- `tests/ut/quantization/test_kv_turboquant.py`

### 2.2 修改文件

- `vllm_ascend/quantization/modelslim_config.py`
- `vllm_ascend/attention/attention_v1.py`
- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/quantization/methods/__init__.py`

## 3. 实现方式

### 3.1 TurboQuant 的入口

`AscendModelSlimConfig.get_quant_method()` 新增了 `kv_cache_type == "TurboQuant"` 的 special-case。

当 quant config 命中 TurboQuant attention 层时，会返回：

```python
AscendKVCacheMethod(AscendTurboQuantKVCacheAttentionMethod(...))
```

这和现有 C8 路径保持一致。

### 3.2 Layer 初始化阶段做了什么

`AscendTurboQuantKVCacheAttentionMethod.create_weights()` 会：

- 把 `layer.impl` 切到 `AscendTurboQuantAttentionBackendImpl`
- 把 `layer.kv_cache_torch_dtype` 设成 `torch.int8`
- 在 layer 上注册：
  - `k_codebook`
  - `v_codebook`
  - `k_boundary`
  - `v_boundary`
  - `k_rot`
  - `v_rot`
  - `k_qjl_proj`
- 在 layer 上挂 TurboQuant 运行配置：
  - `tq_k_variant`
  - `tq_v_variant`
  - `tq_k_bits`
  - `tq_v_bits`
  - `tq_rotation_scope`
  - `tq_outlier_channels`

当前默认实现是：

- `K = prod`
- `V = mse`

但 runtime 本身也兼容 `K = mse`

### 3.3 Reference runtime 做了什么

`turboquant_runtime.py` 里实现了第一版 reference 算法：

- `build_turboquant_codebook()`
- `build_rotation_matrix()`
- `build_qjl_projection()`
- `pack_bits() / unpack_bits()`
- `turboquant_encode_mse()`
- `turboquant_encode_prod()`
- `turboquant_decode_mse()`
- `turboquant_decode_prod()`

这里的实现是偏工程化参考版本：

- codebook 已实现基于 Beta 坐标分布的 Lloyd-Max 标量量化（`_beta_coordinate_pdf()` + `build_beta_lloyd_max_codebook()`）
- rotation / qjl projection 使用随机正交矩阵（QR 分解）/ 随机高斯投影矩阵
- 所有 encode/decode 都是 PyTorch 版本
- 仍未实现：artifact loader、outlier split、fused Ascend op、compressed-domain attention

### 3.4 KV cache 布局怎么落的

`turboquant_layout.py` 新增了 `TurboQuantAttentionSpec`，在 `page_size_bytes` 里显式预算 sidecar 布局。

当前 sidecar cache 形态是：

- `k_idx`
- `k_qjl`（仅 `K=prod`）
- `k_gamma`（仅 `K=prod`）
- `k_norm`
- `v_idx`
- `v_norm`

`model_runner_v1.py` 会：

1. 在 `get_kv_cache_spec()` 里把普通 `AttentionSpec` 包装成 `TurboQuantAttentionSpec`
2. 在 `_allocate_kv_cache_tensors()` 里按 TurboQuant page budget 分配多块 raw tensor
3. 在 `_reshape_kv_cache_tensors()` 里把这些 raw tensor reshape 成 dict 结构

最终 attention backend 看到的 `kv_cache` 是：

```python
{
  "k_idx": ...,
  "k_qjl": ...,
  "k_gamma": ...,
  "k_norm": ...,
  "v_idx": ...,
  "v_norm": ...,
}
```

### 3.5 Attention backend 怎么跑

`AscendTurboQuantAttentionBackendImpl` 的路径是：

1. `forward()`
2. `_quantize_kv_to_turboquant()`
3. `reshape_and_cache()`
4. `_dequant_paged_kv_to_dense()`
5. `_run_dense_fia()`

具体来说：

- 写 cache 时：
  - 新 K/V 先 encode 成 TurboQuant sidecar 表示
  - 然后按 `slot_mapping` 写进各自 cache tensor

- 读 cache 时：
  - 按 `block_table` gather 当前 batch 需要的 block
  - 解包并 dequant 到 dense BF16/FP16
  - 再调用现有 `torch_npu.npu_fused_infer_attention_score()`

这条路径符合设计文档里的 `dense_then_attention`。

## 4. 当前实现的限制

### 4.1 这是 correctness-first 版本

当前实现没有做：

- 压缩态 paged attention
- fused unpack + dequant + attention
- 自定义 CANN kernel

所以它更适合先验证：

- 链路是否正确
- 精度是否可接受
- sidecar KV cache 布局是否工作正常

### 4.2 Codebook 已升级为 Beta Lloyd-Max，但仍缺部分组件

当前 codebook 已实现基于 Beta 坐标分布的 Lloyd-Max 标量量化（`turboquant_runtime.py` 中的 `_beta_coordinate_pdf()` 和 `build_beta_lloyd_max_codebook()`），不再使用均匀标量 codebook。

仍未实现的 TurboQuant 组件：

- artifact loader（矩阵和 codebook 从外部文件加载）
- outlier channel 分裂策略
- fused Ascend C 自定义算子
- compressed-domain attention

当前版本的核心贡献是”把 TurboQuant 结构接进 vllm-ascend”，在正确性优先的前提下完成了 KV cache sidecar 布局和 reference runtime。

### 4.3 当前没有做完整的 TurboQuant artifact loader

目前 `kv_turboquant.py` 里的旋转矩阵、QJL projection、codebook 都是运行时根据配置种子生成的。

后续如果你要严格对齐论文或离线标定版本，建议把它升级为：

- 从独立 artifact 文件加载
- 支持 per-layer / per-group 预生成矩阵和 codebook

### 4.4 当前 prefix cache / swap / copy 只做了基础兼容

`AscendAttentionBackend.swap_blocks()` 和 `copy_blocks()` 已经补了 dict-based cache 的基础处理。

但更复杂的：

- prefix caching 深场景
- offload
- multi-group cache

还需要在 Ascend 机器上做真实运行验证。

## 5. 建议的下一步

### 5.1 先在 Ascend 机器上做功能验证

建议顺序：

1. 先做最小模型初始化
2. 验证 TurboQuant KV cache 是否能成功分配
3. 验证短 prompt decode 是否能跑通
4. 验证长上下文 decode / chunked prefill

### 5.2 再做数值评测

重点测：

- greedy decode 对比 BF16
- PPL
- NIAH
- LongBench 子集

### 5.3 最后再做性能版

性能版建议下一阶段重点做：

- page-local gather
- 局部 block 解压
- fused unpack + dequant
- 自定义 Ascend op
