# vllm-Ascend 集成 TurboQuant 的开发文档

版本：2026-04-11  
目标读者：熟悉 Python / PyTorch / vLLM 架构的工程师  
适用范围：`vllm-project/vllm-ascend` 当前 main 分支风格的代码基线

---

## 1. 目标与非目标

### 1.1 目标

在 `vllm_ascend` 中新增一条 **KV Cache TurboQuant** 路径，使其能在推理时对新生成的 K/V 向量执行在线量化，并在后续 attention 中从压缩态恢复或使用压缩态数据进行计算。

第一阶段要达成的目标：

1. 支持 **KV cache write-time quantization**：新 K/V 写入 cache 前执行 TurboQuant。
2. 支持 **decode-time read/dequant**：从 paged KV cache 中按 block 读取并恢复 K/V，再走现有 attention kernel。
3. 支持最小可用的配置入口和单元测试。
4. 先以“**数值正确**”为第一优先级，性能优化第二。

### 1.2 非目标

第一阶段不要求：

1. 一开始就做到“压缩域直接 attention”。
2. 一开始就实现自定义 CANN 融合算子。
3. 一开始就完整复现论文中的 2.5-bit / 3.5-bit outlier 通道分裂策略。
4. 一开始就与上游 vLLM 的 KV cache dtype / spec 抽象完全对齐。

---

## 2. 当前代码基线的关键事实

在当前 `vllm-ascend` 里，**最接近 TurboQuant 的现成模板是 C8(INT8 KV cache)**。

### 2.1 当前 C8 的接法

`AscendC8KVCacheAttentionMethod` 的模式是：

- 在 `create_weights(layer)` 里把 `layer.kv_cache_torch_dtype` 改成 `torch.int8`
- 把 `layer.impl` 替换成 `AscendC8AttentionBackendImpl`
- 给 layer 挂上 `k_cache_scale / k_cache_offset / v_cache_scale / v_cache_offset`
- `apply()` 不负责真正量化逻辑，真正的量化 / 反量化发生在 attention backend 中

这说明 **KV cache 量化在 Ascend 里不是普通 linear quant path，而是 attention backend path**。

### 2.2 当前 attention backend 的关键路径

`AscendAttentionBackendImpl` 的典型路径是：

1. `forward()`
2. `reshape_and_cache()`
3. `forward_impl()`
4. `forward_paged_attention()` 或 `forward_fused_infer_attention()`

其中：

- `reshape_and_cache()` 是 **写 cache** 的核心位置
- `forward_paged_attention()` / `forward_fused_infer_attention()` 是 **读 cache / 用 cache 做 attention** 的核心位置

C8 的做法是在 `AscendC8AttentionBackendImpl.forward()` 中先把 K/V 量化成 INT8，然后再进入 `reshape_and_cache()`。后续 decode/pre-fill 阶段，再用 C8 专用路径读取或反量化。

### 2.3 当前 KV cache 分配逻辑

`model_runner_v1.py` 里的 `_allocate_kv_cache_tensors()` / `_reshape_kv_cache_tensors()` 目前默认假设普通 attention 层是 **K tensor + V tensor** 两块 raw buffer。之后会根据 `KVCacheSpec.page_size_bytes` 和 backend 的 `get_kv_cache_shape()` 重新 view 成真正的 cache tensor。

这意味着：

- 如果 TurboQuant 只需要两个载体张量（例如把所有 side metadata 都 pack 进 K/V 两个 byte buffer 中），改动较小。
- 如果 TurboQuant 需要显式 sidecar（例如 `idx / qjl / gamma / norm` 分开存），就必须修改 KV cache 分配和 reshape 逻辑。

### 2.4 当前抽象的限制

上游 vLLM 当前 `KVQuantMode` 只显式支持：

- `NONE`
- `FP8_PER_TENSOR`
- `INT8_PER_TOKEN_HEAD`
- `FP8_PER_TOKEN_HEAD`

并且 `AttentionSpec.page_size_bytes` 的预算逻辑只对常规 K/V 页面和 per-token-head scale 做了额外预算。

因此，**如果 TurboQuant 引入新的 side metadata 布局，做得干净的话，最终应该扩展上游 vLLM 的 KV cache spec / quant mode 抽象**。

---

## 3. 建议的集成策略

## 3.1 第一阶段的推荐落地方式

采用 **“C8 模式的 TurboQuant 版本”**：

- 入口仍然挂在 quant method 上
- 真实逻辑落在 attention backend
- 第一版不追求压缩域 attention
- 第一版先做：
  - 写 cache 时 TurboQuant 编码
  - 读 cache 时按 block 解码到 dense BF16/FP16
  - 再调用现有 attention 路径

这是最稳的路线。

## 3.2 第一阶段的算法选择

建议的默认策略：

- **K cache：TurboQuant_prod**
- **V cache：TurboQuant_mse**

原因：

- K 直接参与 `q·k`，更需要 inner product 性质
- V 更接近“被聚合的内容向量”，MSE 版通常更自然

如果你想进一步降低第一版复杂度，也可以先做：

- K cache：TurboQuant_mse
- V cache：TurboQuant_mse

跑通后再升级到 `K=prod, V=mse`。

## 3.3 第一阶段的数据布局建议

推荐使用 **显式 sidecar 布局**，而不是把所有东西硬塞进现有两个 tensor 的 head_size 维度里。

### 对 K（prod）

建议存：

- `k_idx_cache`：MSE 子量化的 codebook index bit-pack 结果
- `k_qjl_cache`：残差 QJL 的 sign bit-pack 结果
- `k_gamma_cache`：残差范数 `gamma`
- `k_norm_cache`：原始向量 L2 norm（如果输入未单位化）

### 对 V（mse）

建议存：

- `v_idx_cache`：codebook index bit-pack 结果
- `v_norm_cache`：原始向量 L2 norm（如果输入未单位化）

### 对静态参数

建议按 layer 挂载：

- `k_rot` / `v_rot`：旋转矩阵 Π
- `k_codebook` / `v_codebook`
- `k_boundary` / `v_boundary`：用于 codebook lookup 的中点边界
- `k_qjl_proj`：QJL 随机投影矩阵 S（仅 prod 路径）
- `outlier_channel_mask` / `outlier_channel_index`（如果要做 2.5 / 3.5 bit 策略）

---

## 4. 必须修改的文件清单

## 4.1 新增：`vllm_ascend/quantization/methods/kv_turboquant.py`

### 作用

定义 TurboQuant 对 attention layer 的 scheme，作用类似当前 `kv_c8.py` 里的 `AscendC8KVCacheAttentionMethod`。

### 需要新增的类

```python
class AscendTurboQuantKVCacheAttentionMethod(AscendAttentionScheme):
    def __init__(self, quant_description: dict, prefix: str):
        ...

    def create_weights(self, layer: torch.nn.Module) -> None:
        ...

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        ...

    def apply(...):
        raise RuntimeError("TurboQuant KV cache is handled by attention backend")
```

### `create_weights(layer)` 中必须做的事

1. 把 `layer.impl` 替换为 `AscendTurboQuantAttentionBackendImpl`
2. 设置 TurboQuant 需要的静态参数/缓存：
   - `k_codebook`
   - `v_codebook`
   - `k_boundary`
   - `v_boundary`
   - `k_rot`
   - `v_rot`
   - `k_qjl_proj`（如果 K 用 prod）
3. 为后续运行期准备必要的配置字段：
   - `layer.tq_k_variant = "prod" or "mse"`
   - `layer.tq_v_variant = "mse"`
   - `layer.tq_k_bits`
   - `layer.tq_v_bits`
   - `layer.tq_rotation_scope`
   - `layer.tq_outlier_channels`
4. 如果第一版打算沿用 `torch.int8` 作为底层 carrier，可以把 `layer.kv_cache_torch_dtype` 设为 `torch.int8`。
   - 注意：这里的 `int8` 只是 **raw byte carrier**，不意味着 TurboQuant 本身是 INT8 算法。

### `process_weights_after_loading(layer)` 中建议做的事

1. 把 codebook / boundary / rotation / qjl projection 转成最终运行 dtype
2. 做 TP shard 或 per-head reshape
3. 预计算便于 kernel / 向量化处理的布局：
   - `boundary` contiguous 化
   - `rot_t` / `proj_t` contiguous 化
4. 如果采用固定随机种子生成矩阵，也可在这里直接 materialize 到设备上

### 建议增加的辅助函数

```python
def build_turboquant_codebook(head_dim: int, bits: int, device, dtype): ...
def build_rotation_matrix(head_dim: int, seed: int, device, dtype): ...
def build_qjl_projection(head_dim: int, seed: int, device, dtype): ...
def pack_bits(indices: torch.Tensor, bits: int) -> torch.Tensor: ...
def unpack_bits(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor: ...
```

---

## 4.2 修改：`vllm_ascend/quantization/modelslim_config.py`

### 当前现状

当前 `get_quant_method()` 已经对 `kv_cache_type == "C8"` 做了 attention 层 special-case。

### 需要做的修改

在 `get_quant_method()` 中新增 TurboQuant special-case，形式上直接仿照 C8：

```python
elif isinstance(layer, AttentionLayerBase) and self.quant_description.get("kv_cache_type") == "TurboQuant":
    from .methods.kv_turboquant import AscendTurboQuantKVCacheAttentionMethod
    return AscendKVCacheMethod(
        AscendTurboQuantKVCacheAttentionMethod(self.quant_description, prefix)
    )
```

### 同时修改 `_add_kvcache_quant_metadata()`

当前这个函数只会识别：

- `fa_quant_type`
- `indexer_quant_type`
- 对应 layer id

TurboQuant 需要新增自己的 metadata 解析，例如：

```python
self.enable_turboquant = self.quant_description.get("kv_cache_type") == "TurboQuant"
self.turboquant_layers = self.quant_description.get("turboquant_layers", [])
```

如果你希望支持“只给部分层启用 TurboQuant”，这里必须解析 layer id。

### 推荐的 quant_description 格式

```json
{
  "kv_cache_type": "TurboQuant",
  "turboquant_layers": [0, 1, 2, 3, 4, 5],
  "turboquant": {
    "k_variant": "prod",
    "v_variant": "mse",
    "k_bits": 3,
    "v_bits": 2,
    "rotation_scope": "per_layer",
    "rotation_seed_base": 1234,
    "outlier_channels": 32,
    "outlier_bits": 3,
    "regular_bits": 2,
    "dequant_mode": "dense_then_attention"
  }
}
```

第一版建议：

- 不做太多动态开关
- 先支持固定的 `k=prod, v=mse`
- 先支持全层统一配置

---

## 4.3 新增或修改：`vllm_ascend/attention/attention_v1.py`

这是 **TurboQuant 的主战场**。

### 需要新增的类

```python
class AscendTurboQuantAttentionBackendImpl(AscendAttentionBackendImpl):
    def forward(...): ...
    def _prepare_turboquant_runtime(...): ...
    def _quantize_kv_to_turboquant(...): ...
    def _dequant_paged_kv_to_dense(...): ...
    def _forward_turboquant_decode(...): ...
    def _forward_turboquant_chunked_prefill(...): ...
    def _forward_turboquant_fused_infer_attention(...): ...
```

### `forward()` 的建议逻辑

参考 C8 路径，但替换成 TurboQuant：

```python
if key is not None and value is not None:
    k_pack, v_pack = self._quantize_kv_to_turboquant(key, value, layer, num_actual_tokens)
    query, k_pack, v_pack, _ = self.reshape_and_cache(query, k_pack, v_pack, kv_cache, attn_metadata, output)

self._prepare_turboquant_runtime(layer, query.device)

if attn_metadata.attn_state == DecodeOnly:
    return self._forward_turboquant_decode(...)
elif attn_metadata.attn_state == ChunkedPrefill:
    return self._forward_turboquant_chunked_prefill(...)
else:
    return self._forward_turboquant_fused_infer_attention(...)
```

### `reshape_and_cache()` 的处理方式

有两种做法：

#### 做法 A：沿用基类 `reshape_and_cache()`

前提：`kv_cache` 的 tensor 结构仍然是简单 tuple，而且你的 TurboQuant pack 结果能伪装成 “key/value tensor”。

优点：改动少。  
缺点：不适合 sidecar 明确分离的布局。

#### 做法 B：在 `AscendTurboQuantAttentionBackendImpl` 里 override `reshape_and_cache()`

推荐第一版就这样做。

你需要：

- 明确 `kv_cache` 的 tuple / dict 结构
- 手动根据 `slot_mapping` 把 `idx / qjl / gamma / norm` 写入各自 cache tensor

这会更清晰，也更利于后续调试。

### `_quantize_kv_to_turboquant()` 应做的事

#### 对 K（prod）

1. 取出每个 token / head 的 `k_vec`
2. 如有需要，计算 `norm = ||k_vec||`
3. 归一化到 unit sphere
4. 执行 `TurboQuant_mse(bits=b-1)`：
   - `y = Π @ x`
   - 做 codebook lookup，得到 `idx`
5. 残差 `r = x - dequant_mse(idx)`
6. `qjl = sign(S @ r)`
7. `gamma = ||r||`
8. 把 `idx / qjl / gamma / norm` pack 成待写 cache 的格式

#### 对 V（mse）

1. 计算 `norm`
2. 单位化
3. `y = Π @ x`
4. codebook lookup -> `idx`
5. 存 `idx / norm`

### `_dequant_paged_kv_to_dense()` 应做的事

它应该模仿当前 C8 的 `_dequant_paged_kv_to_dense()` 的“按 block gather -> dense -> dequant”思路，但解码规则换成 TurboQuant。

建议流程：

1. 用 `block_table` 找到当前 batch 需要的 KV block
2. gather 对应的 packed tensors
3. 根据 `seq_lens` 过滤无效 token
4. 对 K：
   - unpack `idx`
   - MSE dequant -> `x_mse_hat`
   - unpack `qjl`
   - `x_qjl_hat = sqrt(pi/2)/d * gamma * S^T @ qjl`
   - `x_hat = (x_mse_hat + x_qjl_hat) * norm`
5. 对 V：
   - unpack `idx`
   - MSE dequant -> `x_hat`
   - `x_hat = x_hat * norm`
6. 返回 dense BF16/FP16 `key/value`

### `_forward_turboquant_decode()` 的第一版建议

第一版不要直接做压缩域 attention。建议：

1. 从 paged TurboQuant cache gather 当前 decode 需要的 block
2. `dequant_paged_kv_to_dense()`
3. 调用现有 `torch_npu.npu_fused_infer_attention_score(...)`

也就是说，第一版 decode path 是：

```text
paged compressed KV
  -> gather needed blocks
  -> dequant to dense bf16
  -> existing FIA
```

### `_forward_turboquant_chunked_prefill()` 的第一版建议

和 C8 chunked prefill 类似分两段：

- decode 段：从 paged compressed KV dequant
- prefill 段：
  - 如果全是新 token，可直接用 float new KV
  - 如果是 continuing prefill，就从 cache gather + dequant

---

## 4.4 修改：`vllm_ascend/worker/model_runner_v1.py`

这是 **KV cache 内存布局必须改的地方**。

### 当前问题

当前普通 attention 层默认分配：

```python
(k_tensor, v_tensor)
```

然后 `_reshape_kv_cache_tensors()` 默认也按这个结构做 view。

TurboQuant 如果采用 sidecar 布局，至少需要扩成：

- `k_idx_tensor`
- `k_qjl_tensor`（可选）
- `k_gamma_tensor`（可选）
- `k_norm_tensor`
- `v_idx_tensor`
- `v_norm_tensor`

### 推荐修改方案

#### 第一步：扩展 raw tensor 分配

在 `_allocate_kv_cache_tensors()` 中，为 TurboQuant attention 层分配自定义 tuple：

```python
kv_cache_raw_tensors[layer_name] = (
    k_idx_tensor,
    k_qjl_tensor,
    k_gamma_tensor,
    k_norm_tensor,
    v_idx_tensor,
    v_norm_tensor,
)
```

如果暂时不做 `prod`，可以先省掉 `k_qjl_tensor` / `k_gamma_tensor`。

#### 第二步：扩展 reshape

在 `_reshape_kv_cache_tensors()` 中识别 TurboQuant 层，并把 raw tensor reshape 成你自己的 cache layout。

例如：

```python
kv_caches[layer_name] = {
    "k_idx": k_idx_cache,
    "k_qjl": k_qjl_cache,
    "k_gamma": k_gamma_cache,
    "k_norm": k_norm_cache,
    "v_idx": v_idx_cache,
    "v_norm": v_norm_cache,
}
```

### 第一版推荐的 cache object 形态

推荐直接用 dict：

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

而不是硬编码 tuple 下标。这样更利于维护和 debug。

### 注意事项

1. 当前代码里 `bind_kv_cache()` 和 attention backend 可能默认假设 KV cache 是 tuple/list。必要时你需要同时修改绑定和消费逻辑。
2. 当前 `page_size_bytes` 是基于标准 K/V cache 计算出来的。TurboQuant 的 sidecar 版本需要重新定义页面大小预算。
3. 第一版如果你只想尽快跑通，可以在 `model_runner_v1.py` 里做 plugin-local page budget 计算；但长期看最好上游化。

---

## 4.5 可选修改：`vllm_ascend/quantization/methods/registry.py`

如果你打算让 TurboQuant 走注册机制，可以新增 attention scheme 注册，例如：

```python
@register_scheme("TurboQuant", "attention")
class AscendTurboQuantKVCacheAttentionMethod(...):
    ...
```

不过因为当前 C8 走的是 `modelslim_config.py` 里的 special-case，所以第一版完全可以不改 registry，直接沿用 C8 风格。

建议：

- 第一版：**不改 registry，直接 special-case**
- 第二版：再做统一注册化

---

## 4.6 建议新增：辅助模块

为了避免 `attention_v1.py` 过胖，建议新增两个辅助模块。

### `vllm_ascend/quantization/methods/turboquant_runtime.py`

负责：

- codebook 构造/加载
- rotation / qjl matrix 生成
- quant / dequant reference 实现
- bit pack / unpack

### `vllm_ascend/quantization/methods/turboquant_layout.py`

负责：

- 每种 bit-width 的 bytes-per-vector 计算
- side metadata 大小计算
- page size / split factor 计算

这样 `attention_v1.py` 只负责：

- gather
- reshape
- cache 写入
- decode 路径调用

---

## 5. 推荐的最小实现顺序

## 阶段 A：Reference 实现（框架外）

先不要改 vLLM。

完成：

1. `TurboQuantMSE`
2. `TurboQuantProd`
3. bit pack / unpack
4. synthetic 测试：
   - MSE 曲线
   - inner product unbiased 性能

## 阶段 B：vllm-Ascend MVP

建议范围：

1. `kv_turboquant.py` 新增
2. `modelslim_config.py` 增加 TurboQuant 入口
3. `attention_v1.py` 增加 `AscendTurboQuantAttentionBackendImpl`
4. `model_runner_v1.py` 增加 TurboQuant cache 布局
5. decode 路径先做 **gather + dequant to dense + existing attention**

## 阶段 C：质量版本

1. K 用 `prod`
2. V 用 `mse`
3. 增加 outlier / non-outlier channel split
4. 支持 2.5-bit / 3.5-bit effective bits

## 阶段 D：性能版本

1. 自定义 Ascend op：
   - rotate + codebook lookup + pack
   - unpack + dequant
   - residual qjl correction
2. 减少 Python 侧循环
3. 尽量避免整页解压
4. 只对当前 block_table 需要的块做局部恢复

---

## 6. 推荐的函数接口

### 6.1 写 cache 侧接口

```python
def turboquant_encode_k(
    k: torch.Tensor,          # [T, H_kv, D]
    layer: torch.nn.Module,
    num_actual_tokens: int,
) -> dict[str, torch.Tensor]:
    ...


def turboquant_encode_v(
    v: torch.Tensor,          # [T, H_kv, D]
    layer: torch.nn.Module,
    num_actual_tokens: int,
) -> dict[str, torch.Tensor]:
    ...
```

返回建议：

```python
{
  "idx": packed_idx,
  "qjl": packed_qjl,      # 仅 K-prod
  "gamma": gamma,         # 仅 K-prod
  "norm": norm,
}
```

### 6.2 读 cache 侧接口

```python
def turboquant_decode_k_from_paged_cache(
    k_cache: dict[str, torch.Tensor],
    block_table: torch.Tensor,
    seq_lens: list[int],
    layer: torch.nn.Module,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    ...


def turboquant_decode_v_from_paged_cache(...):
    ...
```

---

## 7. 测试计划

## 7.1 单元测试

### 算法正确性

1. `test_turboquant_mse_roundtrip.py`
2. `test_turboquant_prod_unbiased_ip.py`
3. `test_turboquant_pack_unpack.py`

### cache 布局正确性

1. `test_turboquant_page_size.py`
2. `test_turboquant_cache_write_read.py`
3. `test_turboquant_block_table_gather.py`

## 7.2 集成测试

1. 单层 attention smoke test
2. 小模型（例如 1B 以内）短上下文生成测试
3. 长上下文 decode 测试
4. KV cache 命中 / prefix cache / chunked prefill 场景测试

## 7.3 精度测试

至少做：

1. logits MSE / cosine
2. perplexity on short text
3. Needle-in-a-Haystack
4. LongBench 子集

## 7.4 性能测试

至少测：

1. cache 写入延迟
2. decode 吞吐 tokens/s
3. GPU/NPU HBM 占用
4. page gather + dequant 时间占比

---

## 8. 风险点与规避策略

## 8.1 风险：旋转矩阵开销过大

### 现象

- 量化前 `Π @ x`
- 反量化时 `Π^T @ y_hat`
- `prod` 还要 `S @ r` / `S^T @ qjl`

### 对策

- 第一版允许 dense matmul
- 默认 `rotation_scope = per_layer` 或 `per_head_group`
- 不要全模型共享一套矩阵
- 后续再考虑结构化旋转或自定义算子

## 8.2 风险：codebook lookup 变成分支地狱

### 对策

- 不要在 kernel 里写二分查找
- 用 boundary + vectorized compare / LUT
- 先在 Python/PyTorch reference 里验证，再做算子化

## 8.3 风险：读放大

### 现象

- 虽然 cache 省了
- 但 decode 时把整页或整条 request 全部解码成 dense BF16
- 带宽收益被吃回去

### 对策

- 只对 `block_table` 命中的 block 做 gather
- 只对当前 batch 需要的 token 做 dequant
- 第一版允许 dense 化，但范围必须局部化

## 8.4 风险：page size / split factor 算错

### 对策

- 单独写 `layout.py`
- 所有 bytes-per-vector 和 bytes-per-page 都走统一函数
- 单测覆盖不同 bit、不同 head size、不同 outlier 配置

---

## 9. 我建议你现在就按这个顺序开工

### 第 1 步：新建 reference 目录

先做：

- `TurboQuantMSE`
- `TurboQuantProd`
- `pack/unpack`

### 第 2 步：在 `kv_turboquant.py` 里实现量化 scheme

先把 layer 的静态参数挂上去，并把 backend 替换掉。

### 第 3 步：在 `attention_v1.py` 里实现 `AscendTurboQuantAttentionBackendImpl`

先做最朴素路径：

- 写 cache 前编码
- 读 cache 时解码到 dense
- 复用现有 attention kernel

### 第 4 步：在 `model_runner_v1.py` 中实现 sidecar cache 分配

这一阶段不要偷懒把 side metadata 硬塞进现有 head_size 维度。先做清晰版。

### 第 5 步：补测试

至少补：

- roundtrip
- unbiased IP
- cache read/write
- decode smoke test

### 第 6 步：再做性能优化

等数值正确以后，再考虑：

- 自定义 aclnn op
- 局部块恢复
- 更细的 outlier 分裂策略

---

## 10. 结论

如果只问一句：**在 vllm-ascend 里复现 TurboQuant，最应该照着谁改？**

答案是：

**照着当前 C8(INT8 KV cache) 的接法改，但把 INT8 的 scale/offset 标量量化，替换成 TurboQuant 的向量量化编码/解码。**

从代码层面，第一阶段真正必须改的核心文件就是这四个：

1. `vllm_ascend/quantization/methods/kv_turboquant.py`（新增）
2. `vllm_ascend/quantization/modelslim_config.py`（入口）
3. `vllm_ascend/attention/attention_v1.py`（主逻辑）
4. `vllm_ascend/worker/model_runner_v1.py`（cache 布局）

如果要做成长期可维护的正式方案，最终还应同步考虑上游 vLLM 的：

- `KVQuantMode`
- `KVCacheSpec.page_size_bytes`
- `CacheDType / kv_cache_dtype`

但作为第一版可跑 MVP，上面四个文件已经足够开始。
