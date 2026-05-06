#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Benchmark TurboQuant compressed-domain K-score formula.

This script validates the algebra used by the next compressed attention step:
compute ``Q @ K.T`` directly from prod-compressed paged K, instead of
materializing dense K first.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch

from vllm_ascend.ops.turboquant.dequant import (
    build_token_map_from_block_table,
    tq_dequant_mse_paged_rot,
    tq_dequant_mse_paged_scaled_rot,
    tq_dequant_prod_paged_k_score,
    tq_dequant_prod_paged_k_score_reference,
)
from vllm_ascend.quantization.methods.turboquant_layout import get_stage1_bits
from vllm_ascend.quantization.methods.turboquant_runtime import (
    apply_rotation,
    build_qjl_projection,
    build_rotation_matrix,
    build_turboquant_codebook,
    pack_bits,
)


def _sync() -> None:
    torch.npu.synchronize()


def _bench(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1000.0 / iters


def _dtype_from_name(name: str) -> torch.dtype:
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in dtypes:
        raise ValueError(f"Unsupported dtype: {name}")
    return dtypes[name]


def _random_packed_cache(
    shape_prefix: tuple[int, int, int],
    head_dim: int,
    bits: int,
    device: str,
) -> torch.Tensor:
    unpacked = torch.randint(
        0,
        1 << bits,
        (*shape_prefix, head_dim),
        dtype=torch.uint8,
        device=device,
    )
    return pack_bits(unpacked, bits=bits).contiguous()


def _make_block_table(args, device: str) -> tuple[torch.Tensor, list[int]]:
    blocks_per_seq = (args.seq_len + args.block_size - 1) // args.block_size
    required_blocks = args.batch_size * blocks_per_seq
    if required_blocks > args.num_blocks:
        raise ValueError(
            f"num_blocks={args.num_blocks} is too small; need "
            f"{required_blocks} blocks for this case."
        )
    block_ids = torch.randperm(args.num_blocks, device=device)[:required_blocks]
    block_table = block_ids.to(torch.int32).view(args.batch_size, blocks_per_seq)
    return block_table.contiguous(), [args.seq_len] * args.batch_size


def _make_inputs(args):
    device = "npu"
    torch.manual_seed(args.seed)
    stage1_bits = get_stage1_bits(args.k_bits, "prod")
    cache_shape = (args.num_blocks, args.block_size, args.num_kv_heads)

    packed_idx = _random_packed_cache(
        cache_shape, args.head_dim, stage1_bits, device,
    )
    packed_qjl = _random_packed_cache(cache_shape, args.head_dim, 1, device)
    gamma = (
        0.1 + torch.rand((*cache_shape, 1), dtype=torch.float32, device=device)
    ).contiguous()
    norm = (
        0.1 + torch.rand((*cache_shape, 1), dtype=torch.float32, device=device)
    ).contiguous()

    codebook, _ = build_turboquant_codebook(
        args.head_dim, stage1_bits, device, torch.float32,
    )
    qjl_codebook = torch.tensor([-1.0, 1.0], dtype=torch.float32, device=device)
    rotation = build_rotation_matrix(
        args.head_dim, args.seed + 11, device, torch.float32,
    )
    rotation_t = rotation.transpose(0, 1).contiguous()
    qjl_proj = build_qjl_projection(
        args.head_dim, args.seed + 12, device, torch.float32,
    )
    block_table, seq_lens = _make_block_table(args, device)
    query = torch.randn(
        args.batch_size,
        args.num_heads,
        args.head_dim,
        dtype=_dtype_from_name(args.query_dtype),
        device=device,
    )
    return (
        packed_idx,
        packed_qjl,
        gamma,
        norm,
        codebook,
        qjl_codebook,
        rotation,
        rotation_t,
        qjl_proj,
        block_table,
        seq_lens,
        query,
    )


def _dense_score_from_flat_k(
    query: torch.Tensor,
    dense_k: torch.Tensor,
    seq_lens: list[int],
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    batch = int(query.shape[0])
    num_heads = int(query.shape[1])
    max_seq_len = max(seq_lens, default=0)
    q_per_kv = num_heads // num_kv_heads
    scores = torch.full(
        (batch, num_heads, max_seq_len),
        float("-inf"),
        dtype=torch.float32,
        device=query.device,
    )
    token_start = 0
    for batch_idx, seq_len in enumerate(seq_lens):
        batch_k = dense_k[token_start:token_start + seq_len].to(torch.float32)
        batch_q = query[batch_idx].to(torch.float32)
        for kv_head in range(num_kv_heads):
            h_start = kv_head * q_per_kv
            h_end = h_start + q_per_kv
            scores[batch_idx, h_start:h_end, :seq_len] = (
                batch_q[h_start:h_end] @ batch_k[:, kv_head].T
            ) * scale
        token_start += seq_len
    return scores.contiguous()


def _run_case(args) -> None:
    (
        packed_idx,
        packed_qjl,
        gamma,
        norm,
        codebook,
        qjl_codebook,
        rotation,
        rotation_t,
        qjl_proj,
        block_table,
        seq_lens,
        query,
    ) = _make_inputs(args)

    stage1_bits = get_stage1_bits(args.k_bits, "prod")
    token_block_ids, token_offsets = build_token_map_from_block_table(
        block_table, seq_lens, args.block_size,
    )
    scale = args.head_dim ** -0.5

    def dense_k_score_fn():
        stage1_rot = tq_dequant_mse_paged_rot(
            packed_idx,
            norm,
            token_block_ids,
            token_offsets,
            codebook,
            stage1_bits,
            args.head_dim,
            torch.float32,
        )
        correction = math.sqrt(math.pi / 2.0) / args.head_dim
        qjl_scaled = tq_dequant_mse_paged_scaled_rot(
            packed_qjl,
            norm,
            gamma,
            token_block_ids,
            token_offsets,
            qjl_codebook,
            1,
            args.head_dim,
            torch.float32,
            correction,
            signed_bits1=True,
        )
        dense_rot = stage1_rot + apply_rotation(qjl_scaled, qjl_proj)
        dense_k = apply_rotation(dense_rot, rotation_t).contiguous()
        return _dense_score_from_flat_k(
            query, dense_k, seq_lens, args.num_kv_heads, scale,
        )

    def compressed_score_fn():
        return tq_dequant_prod_paged_k_score(
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
            args.k_bits,
            args.head_dim,
            scale=scale,
            max_seq_len=args.seq_len,
        )

    def compressed_reference_fn():
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
            args.k_bits,
            args.head_dim,
            scale=scale,
            max_seq_len=args.seq_len,
        )

    dense_scores = dense_k_score_fn()
    compressed_scores = compressed_score_fn()
    compressed_ref = compressed_reference_fn()
    _sync()

    dense_ms = _bench(dense_k_score_fn, warmup=args.warmup, iters=args.iters)
    compressed_ms = _bench(
        compressed_score_fn, warmup=args.warmup, iters=args.iters,
    )
    compressed_ref_ms = _bench(
        compressed_reference_fn, warmup=args.warmup, iters=args.iters,
    )
    valid = torch.isfinite(dense_scores)
    max_diff = (
        (compressed_scores[valid] - dense_scores[valid]).abs().max().item()
        if valid.any() else 0.0
    )
    ref_diff = (
        (compressed_ref[valid] - dense_scores[valid]).abs().max().item()
        if valid.any() else 0.0
    )
    print(
        f"k_bits={args.k_bits} batch={args.batch_size} seq_len={args.seq_len} "
        f"tokens={args.batch_size * args.seq_len} "
        f"heads={args.num_heads}/{args.num_kv_heads} head_dim={args.head_dim} "
        f"query_dtype={args.query_dtype}"
    )
    print(
        f"dense_k_score={dense_ms:.3f} ms "
        f"compressed_dispatch={compressed_ms:.3f} ms "
        f"compressed_reference={compressed_ref_ms:.3f} ms "
        f"speedup={dense_ms / compressed_ms:.2f}x "
        f"max_diff={max_diff:.3g} ref_diff={ref_diff:.3g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-bits", type=int, default=3)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--query-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--include-custom-op",
        action="store_true",
        help="Use the experimental tq_prod_paged_k_score custom op if present.",
    )
    args = parser.parse_args()

    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available")

    os.environ["VLLM_ASCEND_TQ_USE_CUSTOM_DEQUANT"] = "1"
    os.environ["VLLM_ASCEND_TQ_USE_CUSTOM_PROD_DEQUANT"] = "0"
    os.environ["VLLM_ASCEND_TQ_USE_CUSTOM_K_SCORE"] = (
        "1" if args.include_custom_op else "0"
    )
    os.environ["VLLM_ASCEND_TQ_DEBUG_COMPARE"] = "0"
    os.environ["VLLM_ASCEND_TQ_CUSTOM_STRICT"] = "1"
    try:
        from vllm_ascend.utils import enable_custom_op

        enable_custom_op()
    except Exception:
        pass

    has_mse = hasattr(torch.ops, "_C_ascend") and hasattr(
        torch.ops._C_ascend, "tq_dequant_mse_paged",
    )
    has_mse_scaled = hasattr(torch.ops, "_C_ascend") and hasattr(
        torch.ops._C_ascend, "tq_dequant_mse_paged_scaled_out",
    )
    has_k_score = hasattr(torch.ops, "_C_ascend") and hasattr(
        torch.ops._C_ascend, "tq_prod_paged_k_score",
    )
    print(
        f"custom ops: mse={has_mse} mse_scaled={has_mse_scaled} "
        f"k_score={has_k_score}"
    )
    _run_case(args)


if __name__ == "__main__":
    main()
