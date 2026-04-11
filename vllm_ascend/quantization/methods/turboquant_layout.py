#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import FullAttentionSpec


def packed_bytes_per_vector(dim: int, bits: int) -> int:
    if dim <= 0:
        raise ValueError(f"dim must be > 0, got {dim}")
    if bits <= 0 or bits > 8:
        raise ValueError(f"bits must be within [1, 8], got {bits}")
    return (dim * bits + 7) // 8


def qjl_bytes_per_vector(dim: int) -> int:
    return packed_bytes_per_vector(dim, 1)


def scalar_bytes_per_vector(dtype: torch.dtype) -> int:
    return get_dtype_size(dtype)


@dataclass(frozen=True, kw_only=True)
class TurboQuantAttentionSpec(FullAttentionSpec):
    k_total_bits: int
    v_total_bits: int
    k_variant: str
    v_variant: str
    scalar_dtype: torch.dtype = torch.float32
    use_k_qjl: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.k_stage1_bits <= 0:
            raise ValueError(f"Invalid K stage1 bits derived from total_bits={self.k_total_bits}")
        if self.v_stage1_bits <= 0:
            raise ValueError(f"Invalid V stage1 bits derived from total_bits={self.v_total_bits}")

    @property
    def k_stage1_bits(self) -> int:
        return self.k_total_bits - 1 if self.k_variant == "prod" else self.k_total_bits

    @property
    def v_stage1_bits(self) -> int:
        return self.v_total_bits - 1 if self.v_variant == "prod" else self.v_total_bits

    @property
    def k_idx_bytes_per_vector(self) -> int:
        return packed_bytes_per_vector(self.head_size, self.k_stage1_bits)

    @property
    def v_idx_bytes_per_vector(self) -> int:
        return packed_bytes_per_vector(self.head_size_v, self.v_stage1_bits)

    @property
    def k_qjl_bytes_per_vector(self) -> int:
        return qjl_bytes_per_vector(self.head_size) if self.use_k_qjl else 0

    @property
    def scalar_bytes(self) -> int:
        return scalar_bytes_per_vector(self.scalar_dtype)

    @property
    def real_page_size_bytes(self) -> int:
        per_token_per_head = (
            self.k_idx_bytes_per_vector
            + self.k_qjl_bytes_per_vector
            + self.scalar_bytes  # k_gamma
            + self.scalar_bytes  # k_norm
            + self.v_idx_bytes_per_vector
            + self.scalar_bytes  # v_norm
        )
        return self.block_size * self.num_kv_heads * per_token_per_head

    @classmethod
    def merge(cls, specs: list["TurboQuantAttentionSpec"]) -> "TurboQuantAttentionSpec":
        assert all(isinstance(spec, TurboQuantAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be TurboQuantAttentionSpec."
        )
        first = specs[0]
        for spec in specs[1:]:
            assert spec.block_size == first.block_size
            assert spec.num_kv_heads == first.num_kv_heads
            assert spec.head_size == first.head_size
            assert spec.head_size_v == first.head_size_v
            assert spec.dtype == first.dtype
            assert spec.k_total_bits == first.k_total_bits
            assert spec.v_total_bits == first.v_total_bits
            assert spec.k_variant == first.k_variant
            assert spec.v_variant == first.v_variant
            assert spec.scalar_dtype == first.scalar_dtype
            assert spec.use_k_qjl == first.use_k_qjl
        return cls(
            block_size=first.block_size,
            num_kv_heads=first.num_kv_heads,
            head_size=first.head_size,
            head_size_v=first.head_size_v,
            dtype=first.dtype,
            kv_quant_mode=first.kv_quant_mode,
            page_size_padded=first.page_size_padded,
            sliding_window=first.sliding_window,
            attention_chunk_size=first.attention_chunk_size,
            k_total_bits=first.k_total_bits,
            v_total_bits=first.v_total_bits,
            k_variant=first.k_variant,
            v_variant=first.v_variant,
            scalar_dtype=first.scalar_dtype,
            use_k_qjl=first.use_k_qjl,
        )


def turboquant_cache_shapes(
    spec: TurboQuantAttentionSpec, num_blocks: int
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    base = (num_blocks, spec.block_size, spec.num_kv_heads)
    shapes: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        "k_idx": ((*base, spec.k_idx_bytes_per_vector), torch.uint8),
        "k_norm": ((*base, 1), spec.scalar_dtype),
        "v_idx": ((*base, spec.v_idx_bytes_per_vector), torch.uint8),
        "v_norm": ((*base, 1), spec.scalar_dtype),
    }
    if spec.use_k_qjl:
        shapes["k_qjl"] = ((*base, spec.k_qjl_bytes_per_vector), torch.uint8)
        shapes["k_gamma"] = ((*base, 1), spec.scalar_dtype)
    return shapes
