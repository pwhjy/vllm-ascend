#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Paged dequant API for TurboQuant.

Provides a unified dispatch layer between the attention backend and the
dequant implementation. Currently all paths fall back to the PyTorch
reference runtime. When Ascend C custom ops become available (gated by
the env var ``VLLM_ASCEND_TQ_USE_CUSTOM_DEQUANT``), the dispatch switches
to the accelerated path automatically.

Output convention (P2):
    Dequant functions ending in ``_rot`` return vectors in **rotated
    space** (no inverse rotation). The caller applies the inverse rotation
    separately (``dense = dense_rot @ rotation_t``) and eventually fuses
    it into the attention kernel (P3–P4).
"""

from __future__ import annotations

from collections import OrderedDict
import math
import os
import threading
import weakref

import torch

from vllm_ascend.quantization.methods.turboquant_layout import (
    get_stage1_bits,
)
from vllm_ascend.quantization.methods.turboquant_runtime import (
    apply_rotation,
    unpack_bits,
)


# ---------------------------------------------------------------------------
# Environment gates
# ---------------------------------------------------------------------------

def custom_dequant_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_TQ_USE_CUSTOM_DEQUANT", "0") == "1"


def prod_custom_dequant_enabled() -> bool:
    return (
        custom_dequant_enabled()
        and os.getenv("VLLM_ASCEND_TQ_USE_CUSTOM_PROD_DEQUANT", "0") == "1"
    )


def k_score_custom_enabled() -> bool:
    return (
        custom_dequant_enabled()
        and os.getenv("VLLM_ASCEND_TQ_USE_CUSTOM_K_SCORE", "0") == "1"
    )


def fused_attention_custom_enabled() -> bool:
    return (
        custom_dequant_enabled()
        and os.getenv("VLLM_ASCEND_TQ_USE_CUSTOM_ATTENTION", "0") == "1"
    )


def custom_strict_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_TQ_CUSTOM_STRICT", "0") == "1"


def debug_compare_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_TQ_DEBUG_COMPARE", "0") == "1"


def attention_debug_compare_enabled() -> bool:
    return (
        debug_compare_enabled()
        or os.getenv("VLLM_ASCEND_TQ_DEBUG_COMPARE_ATTENTION", "0") == "1"
    )


def dequant_debug_compare_enabled() -> bool:
    return (
        debug_compare_enabled()
        or os.getenv("VLLM_ASCEND_TQ_DEBUG_COMPARE_DEQUANT", "0") == "1"
    )


def k_score_debug_compare_enabled() -> bool:
    return (
        debug_compare_enabled()
        or os.getenv("VLLM_ASCEND_TQ_DEBUG_COMPARE_K_SCORE", "0") == "1"
    )


def prod_dequant_debug_compare_enabled() -> bool:
    return (
        debug_compare_enabled()
        or os.getenv("VLLM_ASCEND_TQ_DEBUG_COMPARE_PROD_DEQUANT", "0") == "1"
    )


def token_map_cache_size() -> int:
    return max(0, int(os.getenv("VLLM_ASCEND_TQ_TOKEN_MAP_CACHE_SIZE", "32")))


def fused_attention_score_tile_len() -> int:
    try:
        tile_len = int(
            os.getenv("VLLM_ASCEND_TQ_ATTENTION_SCORE_TILE_LEN", "64")
        )
    except ValueError:
        tile_len = 64
    tile_len = min(256, max(8, tile_len))
    return ((tile_len + 7) // 8) * 8


_TQ_CUSTOM_OUT_DTYPE_CODES = {
    torch.float32: 0,
    torch.float16: 1,
}


def _custom_out_dtype_code(dtype: torch.dtype) -> int | None:
    return _TQ_CUSTOM_OUT_DTYPE_CODES.get(dtype)


# ---------------------------------------------------------------------------
# Custom op availability (aclnn, via torch.ops._C_ascend)
# ---------------------------------------------------------------------------

def _is_npu_tensor(tensor: torch.Tensor) -> bool:
    return bool(getattr(tensor, "is_npu", False)) or tensor.device.type in {
        "npu",
        "privateuseone",
    }


def _custom_op_available(op_name: str, *tensors: torch.Tensor) -> bool:
    if not custom_dequant_enabled():
        return False
    if tensors and not all(_is_npu_tensor(tensor) for tensor in tensors):
        return False
    try:
        from vllm_ascend.utils import enable_custom_op
        if not enable_custom_op():
            return False
    except Exception:
        return False
    return (
        hasattr(torch.ops, "_C_ascend")
        and hasattr(torch.ops._C_ascend, op_name)
    )


# ---------------------------------------------------------------------------
# Block-table → token-map conversion
# ---------------------------------------------------------------------------

def build_token_map_from_block_table(
    block_table: torch.Tensor,
    seq_lens: list[int] | torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand vLLM paged block_table + seq_lens into per-token indices.

    Args:
        block_table: ``int32 [batch, max_blocks_per_seq]``
        seq_lens: KV length for each request (list or 1-D int tensor)
        block_size: number of token positions per block (e.g. 128)

    Returns:
        token_block_ids: ``int32 [total_tokens]`` — physical block id
        token_offsets:   ``int32 [total_tokens]`` — offset within the block
    """
    device = block_table.device

    if isinstance(seq_lens, list):
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    else:
        seq_lens_t = seq_lens.to(device=device, dtype=torch.int32)

    batch = int(seq_lens_t.numel())
    if batch == 0:
        empty = torch.empty((0,), dtype=torch.int32, device=device)
        return empty, empty

    max_len = int(seq_lens_t.max().item())
    if max_len == 0:
        empty = torch.empty((0,), dtype=torch.int32, device=device)
        return empty, empty

    positions = torch.arange(max_len, dtype=torch.int32, device=device)
    valid = positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)

    block_indices = positions // block_size
    offsets = positions % block_size

    max_block_idx = int(block_indices.max().item())
    physical_blocks = torch.gather(
        block_table[:, : max_block_idx + 1].to(torch.int32),
        dim=1,
        index=block_indices.unsqueeze(0).expand(batch, max_len).to(torch.long),
    )

    token_block_ids = physical_blocks[valid].contiguous().to(torch.int32)
    token_offsets = offsets.unsqueeze(0).expand(batch, max_len)[valid].contiguous().to(torch.int32)
    return token_block_ids, token_offsets


_TOKEN_MAP_CACHE_LOCK = threading.Lock()
_TOKEN_MAP_CACHE: OrderedDict[
    tuple,
    tuple[weakref.ReferenceType[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
] = OrderedDict()


def _seq_lens_cache_key(seq_lens: list[int] | torch.Tensor) -> tuple[int, ...]:
    if isinstance(seq_lens, list):
        return tuple(int(seq_len) for seq_len in seq_lens)
    return tuple(int(seq_len) for seq_len in seq_lens.detach().cpu().tolist())


def _token_map_cache_key(
    block_table: torch.Tensor,
    seq_lens: list[int] | torch.Tensor,
    block_size: int,
) -> tuple:
    return (
        str(block_table.device),
        int(block_table.data_ptr()),
        tuple(int(dim) for dim in block_table.shape),
        tuple(int(stride) for stride in block_table.stride()),
        int(block_table.storage_offset()),
        int(getattr(block_table, "_version", 0)),
        str(block_table.dtype),
        int(block_size),
        _seq_lens_cache_key(seq_lens),
    )


def cached_token_map_from_block_table(
    block_table: torch.Tensor,
    seq_lens: list[int] | torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_size = token_map_cache_size()
    if cache_size == 0:
        return build_token_map_from_block_table(block_table, seq_lens, block_size)

    key = _token_map_cache_key(block_table, seq_lens, block_size)
    with _TOKEN_MAP_CACHE_LOCK:
        cached = _TOKEN_MAP_CACHE.get(key)
        if cached is not None:
            cached_block_table, cached_token_map = cached
            if cached_block_table() is block_table:
                _TOKEN_MAP_CACHE.move_to_end(key)
                return cached_token_map
            del _TOKEN_MAP_CACHE[key]

    token_map = build_token_map_from_block_table(block_table, seq_lens, block_size)
    with _TOKEN_MAP_CACHE_LOCK:
        _TOKEN_MAP_CACHE[key] = (weakref.ref(block_table), token_map)
        _TOKEN_MAP_CACHE.move_to_end(key)
        while len(_TOKEN_MAP_CACHE) > cache_size:
            _TOKEN_MAP_CACHE.popitem(last=False)
    return token_map


# ---------------------------------------------------------------------------
# Reference implementation (rotated-space output)
# ---------------------------------------------------------------------------

def tq_dequant_mse_paged_reference_rot(
    packed_idx: torch.Tensor,
    norm: torch.Tensor,
    token_block_ids: torch.Tensor,
    token_offsets: torch.Tensor,
    codebook: torch.Tensor,
    bits: int,
    head_dim: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference: paged gather + unpack + codebook lookup + norm rescale.

    Output is in **rotated space** (no inverse rotation applied).

    Shapes:
        packed_idx:  [num_blocks, block_size, num_kv_heads, packed_cols]
        norm:        [num_blocks, block_size, num_kv_heads, 1]
        output:      [total_tokens, num_kv_heads, head_dim]
    """
    total_tokens = int(token_block_ids.numel())
    num_kv_heads = int(packed_idx.shape[2])

    if total_tokens == 0:
        return torch.empty(
            (0, num_kv_heads, head_dim),
            dtype=target_dtype,
            device=packed_idx.device,
        )

    gathered_idx = packed_idx[token_block_ids.long(), token_offsets.long()]
    gathered_norm = norm[token_block_ids.long(), token_offsets.long()]

    idx = unpack_bits(gathered_idx, bits=bits, dim=head_dim)
    dense_rot = codebook[idx.long()].to(target_dtype) * gathered_norm.to(target_dtype)

    return dense_rot.contiguous()


def tq_dequant_prod_paged_reference_rot(
    packed_idx: torch.Tensor,
    packed_qjl: torch.Tensor,
    gamma: torch.Tensor,
    norm: torch.Tensor,
    token_block_ids: torch.Tensor,
    token_offsets: torch.Tensor,
    codebook: torch.Tensor,
    qjl_proj: torch.Tensor,
    total_bits: int,
    head_dim: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference: paged K=prod dequant into rotated space.

    This returns ``(stage1_mse + qjl_correction) * norm`` before the final
    inverse rotation. The caller applies ``rotation_t`` separately.
    """
    total_tokens = int(token_block_ids.numel())
    num_kv_heads = int(packed_idx.shape[2])

    if total_tokens == 0:
        return torch.empty(
            (0, num_kv_heads, head_dim),
            dtype=target_dtype,
            device=packed_idx.device,
        )

    stage1_bits = get_stage1_bits(total_bits, "prod")
    gathered_idx = packed_idx[token_block_ids.long(), token_offsets.long()]
    gathered_qjl = packed_qjl[token_block_ids.long(), token_offsets.long()]
    gathered_gamma = gamma[token_block_ids.long(), token_offsets.long()]
    gathered_norm = norm[token_block_ids.long(), token_offsets.long()]

    idx = unpack_bits(gathered_idx, bits=stage1_bits, dim=head_dim)
    qjl = unpack_bits(gathered_qjl, bits=1, dim=head_dim).to(target_dtype)
    qjl = qjl * 2.0 - 1.0

    x_mse_hat_rot = codebook[idx.long()].to(target_dtype)
    correction = math.sqrt(math.pi / 2.0) / head_dim
    x_qjl_hat_rot = (
        correction
        * gathered_gamma.to(target_dtype)
        * apply_rotation(qjl, qjl_proj.to(target_dtype))
    )
    dense_rot = (
        x_mse_hat_rot + x_qjl_hat_rot
    ) * gathered_norm.to(target_dtype)

    return dense_rot.contiguous()


# ---------------------------------------------------------------------------
# Reference implementation (compressed-domain K score)
# ---------------------------------------------------------------------------

def _seq_lens_to_list(seq_lens: list[int] | torch.Tensor) -> list[int]:
    if isinstance(seq_lens, list):
        return [int(seq_len) for seq_len in seq_lens]
    return [int(seq_len) for seq_len in seq_lens.detach().cpu().tolist()]


def tq_dequant_prod_paged_k_score_reference(
    query: torch.Tensor,
    packed_idx: torch.Tensor,
    packed_qjl: torch.Tensor,
    gamma: torch.Tensor,
    norm: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int] | torch.Tensor,
    codebook: torch.Tensor,
    rotation: torch.Tensor,
    qjl_proj: torch.Tensor,
    total_bits: int,
    head_dim: int,
    *,
    scale: float = 1.0,
    max_seq_len: int | None = None,
    score_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference compressed K-score for prod-compressed paged K cache.

    Computes ``query @ K.T`` directly from TurboQuant compressed K without
    materializing dense K. The output layout is ``[batch, num_heads,
    max_seq_len]`` and positions beyond each request's sequence length are
    filled with ``-inf`` so the tensor can be fed to a softmax-like stage.
    """
    batch = int(query.shape[0])
    num_heads = int(query.shape[1])
    num_kv_heads = int(packed_idx.shape[2])
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if int(query.shape[-1]) != head_dim:
        raise ValueError("query last dimension must match head_dim")
    if int(qjl_proj.shape[0]) != head_dim or int(qjl_proj.shape[1]) != head_dim:
        raise ValueError("qjl_proj must be [head_dim, head_dim]")
    if int(rotation.shape[0]) != head_dim or int(rotation.shape[1]) != head_dim:
        raise ValueError("rotation must be [head_dim, head_dim]")

    seq_lens_list = _seq_lens_to_list(seq_lens)
    if len(seq_lens_list) != batch:
        raise ValueError("seq_lens length must match query batch")
    if max_seq_len is None:
        max_seq_len = max(seq_lens_list, default=0)
    max_seq_len = int(max_seq_len)

    scores = torch.full(
        (batch, num_heads, max_seq_len),
        float("-inf"),
        dtype=score_dtype,
        device=query.device,
    )
    if batch == 0 or max_seq_len == 0:
        return scores

    stage1_bits = get_stage1_bits(total_bits, "prod")
    q_per_kv = num_heads // num_kv_heads
    correction = math.sqrt(math.pi / 2.0) / head_dim

    q_rot = apply_rotation(
        query.to(score_dtype),
        rotation.to(device=query.device, dtype=score_dtype),
    )
    q_qjl = apply_rotation(
        q_rot,
        qjl_proj.transpose(0, 1).contiguous().to(
            device=query.device, dtype=score_dtype,
        ),
    )
    codebook = codebook.to(device=query.device, dtype=score_dtype)

    for batch_idx, seq_len in enumerate(seq_lens_list):
        if seq_len <= 0:
            continue
        if seq_len > max_seq_len:
            raise ValueError("seq_len cannot exceed max_seq_len")

        positions = torch.arange(seq_len, dtype=torch.long, device=query.device)
        block_indices = positions // int(packed_idx.shape[1])
        token_offsets = positions % int(packed_idx.shape[1])
        block_ids = block_table[batch_idx, block_indices].long()

        gathered_idx = packed_idx[block_ids, token_offsets]
        gathered_qjl = packed_qjl[block_ids, token_offsets]
        gathered_gamma = gamma[block_ids, token_offsets].squeeze(-1).to(score_dtype)
        gathered_norm = norm[block_ids, token_offsets].squeeze(-1).to(score_dtype)

        idx = unpack_bits(gathered_idx, bits=stage1_bits, dim=head_dim)
        mse_k = codebook[idx.long()]
        qjl_sign = unpack_bits(gathered_qjl, bits=1, dim=head_dim).to(score_dtype)
        qjl_sign = qjl_sign * 2.0 - 1.0

        for kv_head in range(num_kv_heads):
            h_start = kv_head * q_per_kv
            h_end = h_start + q_per_kv
            mse_score = q_rot[batch_idx, h_start:h_end] @ mse_k[:, kv_head].T
            qjl_score = q_qjl[batch_idx, h_start:h_end] @ qjl_sign[:, kv_head].T
            token_score = (
                mse_score
                + correction * gathered_gamma[:, kv_head].unsqueeze(0) * qjl_score
            )
            token_score = (
                token_score
                * gathered_norm[:, kv_head].unsqueeze(0)
                * float(scale)
            )
            scores[batch_idx, h_start:h_end, :seq_len] = token_score

    return scores.contiguous()


def tq_dequant_prod_paged_k_score(
    query: torch.Tensor,
    packed_idx: torch.Tensor,
    packed_qjl: torch.Tensor,
    gamma: torch.Tensor,
    norm: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int] | torch.Tensor,
    codebook: torch.Tensor,
    rotation: torch.Tensor,
    qjl_proj: torch.Tensor,
    total_bits: int,
    head_dim: int,
    *,
    scale: float = 1.0,
    max_seq_len: int | None = None,
    score_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dispatch compressed K-score for prod-compressed paged K cache."""
    seq_lens_list = _seq_lens_to_list(seq_lens)
    if max_seq_len is None:
        max_seq_len = max(seq_lens_list, default=0)
    max_seq_len = int(max_seq_len)

    def _reference() -> torch.Tensor:
        return tq_dequant_prod_paged_k_score_reference(
            query,
            packed_idx,
            packed_qjl,
            gamma,
            norm,
            block_table,
            seq_lens,
            codebook,
            rotation,
            qjl_proj,
            total_bits,
            head_dim,
            scale=scale,
            max_seq_len=max_seq_len,
            score_dtype=score_dtype,
        )

    if (
        score_dtype != torch.float32
        or head_dim < 16
        or not k_score_custom_enabled()
        or not _custom_op_available(
            "tq_prod_paged_k_score",
            query,
            packed_idx,
            packed_qjl,
            gamma,
            norm,
            block_table,
            codebook,
        )
    ):
        return _reference()

    seq_lens_t = (
        torch.tensor(seq_lens_list, dtype=torch.int32, device=query.device)
        if isinstance(seq_lens, list)
        else seq_lens.to(device=query.device, dtype=torch.int32).contiguous()
    )
    q_rot = apply_rotation(
        query.to(torch.float32),
        rotation.to(device=query.device, dtype=torch.float32),
    ).contiguous()
    q_qjl = apply_rotation(
        q_rot,
        qjl_proj.transpose(0, 1).contiguous().to(
            device=query.device, dtype=torch.float32,
        ),
    ).contiguous()

    out = torch.ops._C_ascend.tq_prod_paged_k_score(
        q_rot,
        q_qjl,
        packed_idx.contiguous(),
        packed_qjl.contiguous(),
        gamma.contiguous(),
        norm.contiguous(),
        block_table.contiguous(),
        seq_lens_t,
        codebook.contiguous(),
        int(total_bits),
        int(head_dim),
        float(scale),
        int(max_seq_len),
    )

    if k_score_debug_compare_enabled():
        ref = _reference()
        torch.npu.synchronize()
        valid = torch.isfinite(ref)
        max_diff = (out[valid] - ref[valid]).abs().max().item() if valid.any() else 0.0
        if max_diff > 1e-3:
            raise RuntimeError(
                f"TurboQuant compressed K-score mismatch: max_diff={max_diff}"
            )

    return out.contiguous()


# ---------------------------------------------------------------------------
# Reference/dispatch implementation (prod-K + MSE-V fused decode attention)
# ---------------------------------------------------------------------------

def tq_prod_mse_paged_attention_reference(
    query: torch.Tensor,
    k_packed_idx: torch.Tensor,
    k_packed_qjl: torch.Tensor,
    k_gamma: torch.Tensor,
    k_norm: torch.Tensor,
    v_packed_idx: torch.Tensor,
    v_norm: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int] | torch.Tensor,
    k_codebook: torch.Tensor,
    v_codebook: torch.Tensor,
    k_rotation: torch.Tensor,
    k_qjl_proj: torch.Tensor,
    v_rotation_t: torch.Tensor,
    k_total_bits: int,
    v_bits: int,
    head_dim: int,
    *,
    scale: float = 1.0,
    max_seq_len: int | None = None,
    score_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference fused decode attention for prod-compressed K and MSE V.

    The kernel prototype fuses compressed-domain K-score, softmax, and
    V weighted sum. V is accumulated in rotated space, then this reference
    applies the normal V inverse rotation so callers get dense attention
    output in the same layout as FIA: ``[batch, num_heads, head_dim]``.
    """
    batch = int(query.shape[0])
    num_heads = int(query.shape[1])
    num_kv_heads = int(k_packed_idx.shape[2])
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    seq_lens_list = _seq_lens_to_list(seq_lens)
    if max_seq_len is None:
        max_seq_len = max(seq_lens_list, default=0)
    max_seq_len = int(max_seq_len)

    scores = tq_dequant_prod_paged_k_score_reference(
        query,
        k_packed_idx,
        k_packed_qjl,
        k_gamma,
        k_norm,
        block_table,
        seq_lens,
        k_codebook,
        k_rotation,
        k_qjl_proj,
        k_total_bits,
        head_dim,
        scale=scale,
        max_seq_len=max_seq_len,
        score_dtype=score_dtype,
    )
    probs = torch.softmax(scores, dim=-1)

    token_block_ids, token_offsets = build_token_map_from_block_table(
        block_table,
        seq_lens,
        int(v_packed_idx.shape[1]),
    )
    dense_v_rot = tq_dequant_mse_paged_reference_rot(
        v_packed_idx,
        v_norm,
        token_block_ids,
        token_offsets,
        v_codebook,
        v_bits,
        head_dim,
        score_dtype,
    )

    q_per_kv = num_heads // num_kv_heads
    out_rot = torch.zeros(
        (batch, num_heads, head_dim),
        dtype=score_dtype,
        device=query.device,
    )
    token_start = 0
    for batch_idx, seq_len in enumerate(seq_lens_list):
        if seq_len <= 0:
            continue
        batch_v = dense_v_rot[token_start:token_start + seq_len].to(score_dtype)
        for kv_head in range(num_kv_heads):
            h_start = kv_head * q_per_kv
            h_end = h_start + q_per_kv
            out_rot[batch_idx, h_start:h_end] = (
                probs[batch_idx, h_start:h_end, :seq_len] @ batch_v[:, kv_head]
            )
        token_start += seq_len

    return apply_rotation(
        out_rot,
        v_rotation_t.to(device=query.device, dtype=score_dtype),
    ).contiguous()


def tq_prod_mse_paged_attention(
    query: torch.Tensor,
    k_packed_idx: torch.Tensor,
    k_packed_qjl: torch.Tensor,
    k_gamma: torch.Tensor,
    k_norm: torch.Tensor,
    v_packed_idx: torch.Tensor,
    v_norm: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int] | torch.Tensor,
    k_codebook: torch.Tensor,
    v_codebook: torch.Tensor,
    k_rotation: torch.Tensor,
    k_qjl_proj: torch.Tensor,
    v_rotation_t: torch.Tensor,
    k_total_bits: int,
    v_bits: int,
    head_dim: int,
    *,
    scale: float = 1.0,
    max_seq_len: int | None = None,
    score_tile_len: int | None = None,
    score_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dispatch fused prod-K/MSE-V decode attention prototype."""
    seq_lens_list = _seq_lens_to_list(seq_lens)
    if max_seq_len is None:
        max_seq_len = max(seq_lens_list, default=0)
    max_seq_len = int(max_seq_len)
    num_heads = int(query.shape[1])
    num_kv_heads = int(k_packed_idx.shape[2])
    q_per_kv = num_heads // num_kv_heads if num_kv_heads else 0
    k_stage1_bits = get_stage1_bits(k_total_bits, "prod")
    score_tile_len = (
        fused_attention_score_tile_len()
        if score_tile_len is None
        else int(score_tile_len)
    )
    if score_tile_len <= 0 or score_tile_len > 256:
        raise ValueError("score_tile_len must be in [1, 256]")

    def _reference() -> torch.Tensor:
        return tq_prod_mse_paged_attention_reference(
            query,
            k_packed_idx,
            k_packed_qjl,
            k_gamma,
            k_norm,
            v_packed_idx,
            v_norm,
            block_table,
            seq_lens,
            k_codebook,
            v_codebook,
            k_rotation,
            k_qjl_proj,
            v_rotation_t,
            k_total_bits,
            v_bits,
            head_dim,
            scale=scale,
            max_seq_len=max_seq_len,
            score_dtype=score_dtype,
        )

    fallback_reasons: list[str] = []
    if score_dtype != torch.float32:
        fallback_reasons.append(f"score_dtype={score_dtype}")
    if head_dim < 16:
        fallback_reasons.append(f"head_dim={head_dim} < 16")
    if (head_dim & 3) != 0:
        fallback_reasons.append(f"head_dim={head_dim} is not divisible by 4")
    if q_per_kv != 4:
        fallback_reasons.append(f"q_per_kv={q_per_kv} != 4")
    if k_stage1_bits != 2:
        fallback_reasons.append(f"k_stage1_bits={k_stage1_bits} != 2")
    if v_bits > 3:
        fallback_reasons.append(f"v_bits={v_bits} > 3")
    if max_seq_len > 1024:
        fallback_reasons.append(f"max_seq_len={max_seq_len} > 1024")

    custom_enabled = fused_attention_custom_enabled()
    if not custom_enabled:
        fallback_reasons.append("VLLM_ASCEND_TQ_USE_CUSTOM_ATTENTION is not enabled")
    elif not _custom_op_available(
            "tq_prod_mse_paged_attention",
            query,
            k_packed_idx,
            k_packed_qjl,
            k_gamma,
            k_norm,
            v_packed_idx,
            v_norm,
            block_table,
            k_codebook,
            v_codebook,
        ):
        fallback_reasons.append(
            "torch.ops._C_ascend.tq_prod_mse_paged_attention is unavailable"
        )

    if fallback_reasons:
        if custom_enabled and custom_strict_enabled():
            raise RuntimeError(
                "TurboQuant fused attention custom op fallback: "
                + "; ".join(fallback_reasons)
            )
        return _reference()

    seq_lens_t = (
        torch.tensor(seq_lens_list, dtype=torch.int32, device=query.device)
        if isinstance(seq_lens, list)
        else seq_lens.to(device=query.device, dtype=torch.int32).contiguous()
    )
    q_rot = apply_rotation(
        query.to(torch.float32),
        k_rotation.to(device=query.device, dtype=torch.float32),
    ).contiguous()
    q_qjl = apply_rotation(
        q_rot,
        k_qjl_proj.transpose(0, 1).contiguous().to(
            device=query.device, dtype=torch.float32,
        ),
    ).contiguous()

    out_rot = torch.ops._C_ascend.tq_prod_mse_paged_attention(
        q_rot,
        q_qjl,
        k_packed_idx.contiguous(),
        k_packed_qjl.contiguous(),
        k_gamma.contiguous(),
        k_norm.contiguous(),
        v_packed_idx.contiguous(),
        v_norm.contiguous(),
        block_table.contiguous(),
        seq_lens_t,
        k_codebook.contiguous(),
        v_codebook.contiguous(),
        int(k_total_bits),
        int(v_bits),
        int(head_dim),
        float(scale),
        int(max_seq_len),
        int(score_tile_len),
    )
    out = apply_rotation(
        out_rot,
        v_rotation_t.to(device=query.device, dtype=torch.float32),
    ).contiguous()

    if attention_debug_compare_enabled():
        ref = _reference()
        torch.npu.synchronize()
        max_diff = (out - ref).abs().max().item() if out.numel() else 0.0
        if max_diff > 1e-3:
            raise RuntimeError(
                f"TurboQuant fused attention mismatch: max_diff={max_diff}"
            )

    return out


# ---------------------------------------------------------------------------
# Unified dispatch entry points
# ---------------------------------------------------------------------------

def tq_dequant_mse_paged_rot(
    packed_idx: torch.Tensor,
    norm: torch.Tensor,
    token_block_ids: torch.Tensor,
    token_offsets: torch.Tensor,
    codebook: torch.Tensor,
    bits: int,
    head_dim: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequant MSE-compressed K or V from paged cache → rotated space.

    Dispatches to ``torch.ops._C_ascend.tq_dequant_mse_paged`` when the
    aclnn custom op is built and enabled, otherwise falls back to the
    PyTorch reference.
    """
    # The P2 Ascend C scalar kernel is validated for real model head dims
    # (64/80/96/128). Tiny test-only dims can expose backend edge cases, so
    # keep them on the reference path.
    if head_dim < 16:
        return tq_dequant_mse_paged_reference_rot(
            packed_idx, norm, token_block_ids, token_offsets,
            codebook, bits, head_dim, target_dtype,
        )

    if not _custom_op_available(
        "tq_dequant_mse_paged",
        packed_idx,
        norm,
        token_block_ids,
        token_offsets,
        codebook,
    ):
        return tq_dequant_mse_paged_reference_rot(
            packed_idx, norm, token_block_ids, token_offsets,
            codebook, bits, head_dim, target_dtype,
        )

    out_dtype_code = _custom_out_dtype_code(target_dtype)
    if (
        out_dtype_code is not None
        and _custom_op_available(
            "tq_dequant_mse_paged_out",
            packed_idx,
            norm,
            token_block_ids,
            token_offsets,
            codebook,
        )
    ):
        out = torch.ops._C_ascend.tq_dequant_mse_paged_out(
            packed_idx.contiguous(),
            norm.contiguous(),
            token_block_ids.contiguous(),
            token_offsets.contiguous(),
            codebook.contiguous(),
            int(bits),
            int(head_dim),
            int(out_dtype_code),
        )
    else:
        out = torch.ops._C_ascend.tq_dequant_mse_paged(
            packed_idx.contiguous(),
            norm.contiguous(),
            token_block_ids.contiguous(),
            token_offsets.contiguous(),
            codebook.contiguous(),
            int(bits),
            int(head_dim),
        ).to(target_dtype)

    if dequant_debug_compare_enabled():
        ref = tq_dequant_mse_paged_reference_rot(
            packed_idx, norm, token_block_ids, token_offsets,
            codebook, bits, head_dim, target_dtype,
        )
        torch.npu.synchronize()
        max_diff = (out - ref).abs().max().item() if out.numel() else 0.0
        if max_diff > 1e-3:
            raise RuntimeError(
                f"TurboQuant custom dequant mismatch: max_diff={max_diff}"
            )

    return out.contiguous()


def tq_dequant_mse_paged_scaled_rot(
    packed_idx: torch.Tensor,
    norm: torch.Tensor,
    extra_scale: torch.Tensor,
    token_block_ids: torch.Tensor,
    token_offsets: torch.Tensor,
    codebook: torch.Tensor,
    bits: int,
    head_dim: int,
    target_dtype: torch.dtype,
    scale_multiplier: float,
    signed_bits1: bool = False,
) -> torch.Tensor:
    """Dequant MSE data with two paged scale tensors.

    This is used for the prod-QJL correction path where the scale is
    ``norm * gamma * correction``. Keeping both scale tensors as inputs avoids
    materializing a full-cache temporary scale tensor every decode step.
    """
    out_dtype_code = _custom_out_dtype_code(target_dtype)
    if (
        head_dim >= 16
        and out_dtype_code is not None
        and _custom_op_available(
            "tq_dequant_mse_paged_scaled_out",
            packed_idx,
            norm,
            extra_scale,
            token_block_ids,
            token_offsets,
            codebook,
        )
    ):
        out = torch.ops._C_ascend.tq_dequant_mse_paged_scaled_out(
            packed_idx.contiguous(),
            norm.contiguous(),
            extra_scale.contiguous(),
            token_block_ids.contiguous(),
            token_offsets.contiguous(),
            codebook.contiguous(),
            int(bits),
            int(head_dim),
            int(out_dtype_code),
            float(scale_multiplier),
            1 if signed_bits1 else 0,
        )
    else:
        combined_scale = (norm * extra_scale * float(scale_multiplier)).contiguous()
        out = tq_dequant_mse_paged_rot(
            packed_idx=packed_idx,
            norm=combined_scale,
            token_block_ids=token_block_ids,
            token_offsets=token_offsets,
            codebook=codebook,
            bits=bits,
            head_dim=head_dim,
            target_dtype=target_dtype,
        )

    if dequant_debug_compare_enabled():
        combined_scale = (norm * extra_scale * float(scale_multiplier)).contiguous()
        ref = tq_dequant_mse_paged_reference_rot(
            packed_idx, combined_scale, token_block_ids, token_offsets,
            codebook, bits, head_dim, target_dtype,
        )
        torch.npu.synchronize()
        max_diff = (out - ref).abs().max().item() if out.numel() else 0.0
        if max_diff > 1e-3:
            raise RuntimeError(
                f"TurboQuant custom scaled dequant mismatch: max_diff={max_diff}"
            )

    return out.contiguous()


def tq_dequant_prod_paged_rot(
    packed_idx: torch.Tensor,
    packed_qjl: torch.Tensor,
    gamma: torch.Tensor,
    norm: torch.Tensor,
    token_block_ids: torch.Tensor,
    token_offsets: torch.Tensor,
    codebook: torch.Tensor,
    qjl_proj: torch.Tensor,
    total_bits: int,
    head_dim: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequant prod-compressed K from paged cache to rotated space.

    The first custom op version is guarded by
    ``VLLM_ASCEND_TQ_USE_CUSTOM_PROD_DEQUANT`` so the established MSE path
    stays unchanged while this heavier QJL kernel is benchmarked.
    """
    if head_dim < 16 or not prod_custom_dequant_enabled():
        return tq_dequant_prod_paged_reference_rot(
            packed_idx, packed_qjl, gamma, norm,
            token_block_ids, token_offsets,
            codebook, qjl_proj,
            total_bits, head_dim, target_dtype,
        )

    if not _custom_op_available(
        "tq_dequant_prod_paged",
        packed_idx,
        packed_qjl,
        gamma,
        norm,
        token_block_ids,
        token_offsets,
        codebook,
        qjl_proj,
    ):
        return tq_dequant_prod_paged_reference_rot(
            packed_idx, packed_qjl, gamma, norm,
            token_block_ids, token_offsets,
            codebook, qjl_proj,
            total_bits, head_dim, target_dtype,
        )

    out = torch.ops._C_ascend.tq_dequant_prod_paged(
        packed_idx.contiguous(),
        packed_qjl.contiguous(),
        gamma.contiguous(),
        norm.contiguous(),
        token_block_ids.contiguous(),
        token_offsets.contiguous(),
        codebook.contiguous(),
        qjl_proj.contiguous(),
        int(total_bits),
        int(head_dim),
    )

    out = out.to(target_dtype)

    if prod_dequant_debug_compare_enabled():
        ref = tq_dequant_prod_paged_reference_rot(
            packed_idx, packed_qjl, gamma, norm,
            token_block_ids, token_offsets,
            codebook, qjl_proj,
            total_bits, head_dim, target_dtype,
        )
        torch.npu.synchronize()
        max_diff = (out - ref).abs().max().item() if out.numel() else 0.0
        if max_diff > 1e-3:
            raise RuntimeError(
                f"TurboQuant custom prod dequant mismatch: max_diff={max_diff}"
            )

    return out.contiguous()
