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

import importlib
import os

import torch

from vllm_ascend.quantization.methods.turboquant_runtime import (
    unpack_bits,
)


# ---------------------------------------------------------------------------
# Environment gates
# ---------------------------------------------------------------------------

def custom_dequant_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_TQ_USE_CUSTOM_DEQUANT", "0") == "1"


def debug_compare_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_TQ_DEBUG_COMPARE", "0") == "1"


# ---------------------------------------------------------------------------
# Extension loading
# ---------------------------------------------------------------------------

_TQ_EXT = None
_TQ_EXT_LOAD_TRIED = False


def _load_ext():
    """Load the compiled pybind extension ``vllm_ascend_tq_ops``.

    Returns ``None`` when the extension is not available.
    """
    global _TQ_EXT, _TQ_EXT_LOAD_TRIED

    if _TQ_EXT_LOAD_TRIED:
        return _TQ_EXT

    _TQ_EXT_LOAD_TRIED = True
    try:
        _TQ_EXT = importlib.import_module("vllm_ascend_tq_ops")
    except Exception:
        _TQ_EXT = None
    return _TQ_EXT


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

    Dispatches to the Ascend C custom op when available, otherwise falls
    back to the PyTorch reference.
    """
    if not custom_dequant_enabled():
        return tq_dequant_mse_paged_reference_rot(
            packed_idx, norm, token_block_ids, token_offsets,
            codebook, bits, head_dim, target_dtype,
        )

    ext = _load_ext()
    if ext is None or not getattr(packed_idx, "is_npu", False):
        return tq_dequant_mse_paged_reference_rot(
            packed_idx, norm, token_block_ids, token_offsets,
            codebook, bits, head_dim, target_dtype,
        )

    out = ext.tq_dequant_mse_paged(
        packed_idx, norm, token_block_ids, token_offsets,
        codebook, int(bits), int(head_dim),
    )

    out = out.to(target_dtype)

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
