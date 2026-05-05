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


def debug_compare_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_TQ_DEBUG_COMPARE", "0") == "1"


def token_map_cache_size() -> int:
    return max(0, int(os.getenv("VLLM_ASCEND_TQ_TOKEN_MAP_CACHE_SIZE", "32")))


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

    if debug_compare_enabled():
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

    if debug_compare_enabled():
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
