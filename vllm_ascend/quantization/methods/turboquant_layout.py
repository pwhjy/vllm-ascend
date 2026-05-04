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


def get_stage1_bits(total_bits: int, variant: str) -> int:
    if total_bits <= 0:
        raise ValueError(f"total_bits must be > 0, got {total_bits}")
    if variant == "prod":
        if total_bits < 2:
            raise ValueError(f"TurboQuant prod requires total_bits >= 2, got {total_bits}")
        return total_bits - 1
    if variant == "mse":
        return total_bits
    raise ValueError(f"Unsupported TurboQuant variant: {variant}")


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
        return get_stage1_bits(self.k_total_bits, self.k_variant)

    @property
    def v_stage1_bits(self) -> int:
        return get_stage1_bits(self.v_total_bits, self.v_variant)

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

    @property
    def k_logical_bits(self) -> int:
        """Configured K bit width (total_bits)."""
        return self.k_total_bits

    @property
    def v_logical_bits(self) -> int:
        """Configured V bit width (total_bits)."""
        return self.v_total_bits

    @property
    def k_real_bits_per_channel(self) -> float:
        """Effective K bits/channel including scalar sidecar overhead."""
        idx_bits = self.k_idx_bytes_per_vector * 8
        qjl_bits = self.k_qjl_bytes_per_vector * 8
        scalar_bits = self.scalar_bytes * 8  # k_gamma + k_norm
        return (idx_bits + qjl_bits + scalar_bits) / self.head_size

    @property
    def v_real_bits_per_channel(self) -> float:
        """Effective V bits/channel including scalar sidecar overhead."""
        idx_bits = self.v_idx_bytes_per_vector * 8
        scalar_bits = self.scalar_bytes * 8  # v_norm
        return (idx_bits + scalar_bits) / self.head_size_v

    @property
    def real_allocated_bytes(self) -> int:
        """Total bytes allocated per page (real_page_size_bytes)."""
        return self.real_page_size_bytes

    @property
    def bf16_baseline_bytes(self) -> int:
        """Bytes per page if KV were stored as BF16."""
        return (
            self.block_size
            * self.num_kv_heads
            * (self.head_size + self.head_size_v)
            * 2
        )

    @property
    def effective_compression_ratio(self) -> float:
        """Compression ratio vs BF16 baseline.

        Returns a value in (0, 1] where lower means more compression.
        E.g. 0.25 means 4x compression vs BF16.
        """
        baseline = self.bf16_baseline_bytes
        if baseline == 0:
            return 1.0
        return self.real_page_size_bytes / baseline

    def get_memory_stats(self) -> dict:
        """Return a dictionary of memory statistics suitable for logging."""
        return {
            "k_logical_bits": self.k_logical_bits,
            "v_logical_bits": self.v_logical_bits,
            "k_real_bits_per_channel": round(self.k_real_bits_per_channel, 2),
            "v_real_bits_per_channel": round(self.v_real_bits_per_channel, 2),
            "scalar_dtype": str(self.scalar_dtype),
            "real_allocated_bytes_per_page": self.real_allocated_bytes,
            "bf16_baseline_bytes_per_page": self.bf16_baseline_bytes,
            "effective_compression_ratio": round(self.effective_compression_ratio, 4),
        }

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


def register_turboquant_spec_in_manager() -> None:
    import sys

    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

    stm = sys.modules.get("vllm.v1.core.single_type_kv_cache_manager")
    if stm is not None and TurboQuantAttentionSpec not in stm.spec_manager_map:
        stm.spec_manager_map[TurboQuantAttentionSpec] = FullAttentionManager


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
