#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Benchmark TurboQuant fused compressed decode attention.

This PR5 benchmark compares:
  1. dense_attention: dequant K/V then PyTorch softmax-weighted sum
  2. fused_dispatch: prod-K/MSE-V custom op (K-score + softmax + V sum)
  3. fused_reference: Python reference for correctness
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
    tq_prod_mse_paged_attention,
    tq_prod_mse_paged_attention_reference,
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
    k_stage1_bits = get_stage1_bits(args.k_bits, "prod")
    cache_shape = (args.num_blocks, args.block_size, args.num_kv_heads)

    k_idx = _random_packed_cache(
        cache_shape, args.head_dim, k_stage1_bits, device,
    )
    k_qjl = _random_packed_cache(cache_shape, args.head_dim, 1, device)
    k_gamma = (
        0.1 + torch.rand((*cache_shape, 1), dtype=torch.float32, device=device)
    ).contiguous()
    k_norm = (
        0.1 + torch.rand((*cache_shape, 1), dtype=torch.float32, device=device)
    ).contiguous()
    v_idx = _random_packed_cache(cache_shape, args.head_dim, args.v_bits, device)
    v_norm = (
        0.1 + torch.rand((*cache_shape, 1), dtype=torch.float32, device=device)
    ).contiguous()

    k_codebook, _ = build_turboquant_codebook(
        args.head_dim, k_stage1_bits, device, torch.float32,
    )
    v_codebook, _ = build_turboquant_codebook(
        args.head_dim, args.v_bits, device, torch.float32,
    )
    qjl_codebook = torch.tensor([-1.0, 1.0], dtype=torch.float32, device=device)
    k_rotation = build_rotation_matrix(
        args.head_dim, args.seed + 11, device, torch.float32,
    )
    k_rotation_t = k_rotation.transpose(0, 1).contiguous()
    v_rotation = build_rotation_matrix(
        args.head_dim, args.seed + 13, device, torch.float32,
    )
    v_rotation_t = v_rotation.transpose(0, 1).contiguous()
    k_qjl_proj = build_qjl_projection(
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
        k_idx,
        k_qjl,
        k_gamma,
        k_norm,
        v_idx,
        v_norm,
        k_codebook,
        v_codebook,
        qjl_codebook,
        k_rotation,
        k_rotation_t,
        v_rotation_t,
        k_qjl_proj,
        block_table,
        seq_lens,
        query,
    )


def _dense_attention_from_flat_kv(
    query: torch.Tensor,
    dense_k: torch.Tensor,
    dense_v: torch.Tensor,
    seq_lens: list[int],
    num_kv_heads: int,
    scale: float,
) -> torch.Tensor:
    batch = int(query.shape[0])
    num_heads = int(query.shape[1])
    head_dim = int(query.shape[2])
    q_per_kv = num_heads // num_kv_heads
    out = torch.empty(
        (batch, num_heads, head_dim),
        dtype=torch.float32,
        device=query.device,
    )
    token_start = 0
    for batch_idx, seq_len in enumerate(seq_lens):
        batch_q = query[batch_idx].to(torch.float32)
        batch_k = dense_k[token_start:token_start + seq_len].to(torch.float32)
        batch_v = dense_v[token_start:token_start + seq_len].to(torch.float32)
        for kv_head in range(num_kv_heads):
            h_start = kv_head * q_per_kv
            h_end = h_start + q_per_kv
            scores = (batch_q[h_start:h_end] @ batch_k[:, kv_head].T) * scale
            probs = torch.softmax(scores, dim=-1)
            out[batch_idx, h_start:h_end] = probs @ batch_v[:, kv_head]
        token_start += seq_len
    return out.contiguous()


def _run_case(args) -> None:
    (
        k_idx,
        k_qjl,
        k_gamma,
        k_norm,
        v_idx,
        v_norm,
        k_codebook,
        v_codebook,
        qjl_codebook,
        k_rotation,
        k_rotation_t,
        v_rotation_t,
        k_qjl_proj,
        block_table,
        seq_lens,
        query,
    ) = _make_inputs(args)

    k_stage1_bits = get_stage1_bits(args.k_bits, "prod")
    token_block_ids, token_offsets = build_token_map_from_block_table(
        block_table, seq_lens, args.block_size,
    )
    scale = args.head_dim ** -0.5

    def dense_attention_fn():
        k_stage1_rot = tq_dequant_mse_paged_rot(
            k_idx,
            k_norm,
            token_block_ids,
            token_offsets,
            k_codebook,
            k_stage1_bits,
            args.head_dim,
            torch.float32,
        )
        correction = math.sqrt(math.pi / 2.0) / args.head_dim
        k_qjl_scaled = tq_dequant_mse_paged_scaled_rot(
            k_qjl,
            k_norm,
            k_gamma,
            token_block_ids,
            token_offsets,
            qjl_codebook,
            1,
            args.head_dim,
            torch.float32,
            correction,
            signed_bits1=True,
        )
        dense_k_rot = k_stage1_rot + apply_rotation(k_qjl_scaled, k_qjl_proj)
        dense_k = apply_rotation(dense_k_rot, k_rotation_t).contiguous()
        dense_v_rot = tq_dequant_mse_paged_rot(
            v_idx,
            v_norm,
            token_block_ids,
            token_offsets,
            v_codebook,
            args.v_bits,
            args.head_dim,
            torch.float32,
        )
        dense_v = apply_rotation(dense_v_rot, v_rotation_t).contiguous()
        return _dense_attention_from_flat_kv(
            query, dense_k, dense_v, seq_lens, args.num_kv_heads, scale,
        )

    def fused_attention_fn():
        return tq_prod_mse_paged_attention(
            query,
            k_idx,
            k_qjl,
            k_gamma,
            k_norm,
            v_idx,
            v_norm,
            block_table,
            seq_lens,
            k_codebook,
            v_codebook,
            k_rotation,
            k_qjl_proj,
            v_rotation_t,
            args.k_bits,
            args.v_bits,
            args.head_dim,
            scale=scale,
            max_seq_len=args.seq_len,
            score_tile_len=args.score_tile_len,
        )

    def fused_reference_fn():
        return tq_prod_mse_paged_attention_reference(
            query,
            k_idx,
            k_qjl,
            k_gamma,
            k_norm,
            v_idx,
            v_norm,
            block_table,
            seq_lens,
            k_codebook,
            v_codebook,
            k_rotation,
            k_qjl_proj,
            v_rotation_t,
            args.k_bits,
            args.v_bits,
            args.head_dim,
            scale=scale,
            max_seq_len=args.seq_len,
        )

    dense_out = dense_attention_fn()
    fused_out = fused_attention_fn()
    fused_ref = fused_reference_fn()
    _sync()

    dense_ms = _bench(dense_attention_fn, warmup=args.warmup, iters=args.iters)
    fused_ms = _bench(fused_attention_fn, warmup=args.warmup, iters=args.iters)
    fused_ref_ms = _bench(
        fused_reference_fn, warmup=args.warmup, iters=args.iters,
    )
    max_diff = (fused_out - dense_out).abs().max().item()
    ref_diff = (fused_ref - dense_out).abs().max().item()
    print(
        f"k_bits={args.k_bits} v_bits={args.v_bits} "
        f"batch={args.batch_size} seq_len={args.seq_len} "
        f"tokens={args.batch_size * args.seq_len} "
        f"heads={args.num_heads}/{args.num_kv_heads} "
        f"head_dim={args.head_dim} query_dtype={args.query_dtype} "
        f"score_tile_len={args.score_tile_len}"
    )
    print(
        f"dense_attention={dense_ms:.3f} ms "
        f"fused_dispatch={fused_ms:.3f} ms "
        f"fused_reference={fused_ref_ms:.3f} ms "
        f"speedup={dense_ms / fused_ms:.2f}x "
        f"max_diff={max_diff:.3g} ref_diff={ref_diff:.3g}"
    )
    if args.profile_components:
        _profile_fused_components(
            args,
            query,
            k_idx,
            k_qjl,
            k_gamma,
            k_norm,
            v_idx,
            v_norm,
            block_table,
            seq_lens,
            k_codebook,
            v_codebook,
            k_rotation,
            k_qjl_proj,
            v_rotation_t,
            scale,
        )


def _profile_fused_components(
    args,
    query: torch.Tensor,
    k_idx: torch.Tensor,
    k_qjl: torch.Tensor,
    k_gamma: torch.Tensor,
    k_norm: torch.Tensor,
    v_idx: torch.Tensor,
    v_norm: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: list[int],
    k_codebook: torch.Tensor,
    v_codebook: torch.Tensor,
    k_rotation: torch.Tensor,
    k_qjl_proj: torch.Tensor,
    v_rotation_t: torch.Tensor,
    scale: float,
) -> None:
    if not (
        hasattr(torch.ops, "_C_ascend")
        and hasattr(torch.ops._C_ascend, "tq_prod_mse_paged_attention")
    ):
        print("component_profile: skipped (custom op unavailable)")
        return

    query_f32 = query.to(torch.float32)
    k_rotation_f32 = k_rotation.to(device=query.device, dtype=torch.float32)
    k_qjl_proj_t = k_qjl_proj.transpose(0, 1).contiguous().to(
        device=query.device, dtype=torch.float32,
    )
    v_rotation_t_f32 = v_rotation_t.to(device=query.device, dtype=torch.float32)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=query.device)
    q_rot = apply_rotation(query_f32, k_rotation_f32).contiguous()
    q_qjl = apply_rotation(q_rot, k_qjl_proj_t).contiguous()

    def op_fn():
        return torch.ops._C_ascend.tq_prod_mse_paged_attention(
            q_rot,
            q_qjl,
            k_idx.contiguous(),
            k_qjl.contiguous(),
            k_gamma.contiguous(),
            k_norm.contiguous(),
            v_idx.contiguous(),
            v_norm.contiguous(),
            block_table.contiguous(),
            seq_lens_t,
            k_codebook.contiguous(),
            v_codebook.contiguous(),
            int(args.k_bits),
            int(args.v_bits),
            int(args.head_dim),
            float(scale),
            int(args.seq_len),
            int(args.score_tile_len),
        )

    out_rot = op_fn()
    _sync()
    q_rot_ms = _bench(
        lambda: apply_rotation(query_f32, k_rotation_f32).contiguous(),
        warmup=args.warmup,
        iters=args.iters,
    )
    q_qjl_ms = _bench(
        lambda: apply_rotation(q_rot, k_qjl_proj_t).contiguous(),
        warmup=args.warmup,
        iters=args.iters,
    )
    op_ms = _bench(op_fn, warmup=args.warmup, iters=args.iters)
    v_rot_ms = _bench(
        lambda: apply_rotation(out_rot, v_rotation_t_f32).contiguous(),
        warmup=args.warmup,
        iters=args.iters,
    )
    print(
        f"component_profile: q_rotate={q_rot_ms:.3f} ms "
        f"q_qjl_rotate={q_qjl_ms:.3f} ms "
        f"custom_op={op_ms:.3f} ms "
        f"v_rotate={v_rot_ms:.3f} ms "
        f"sum={q_rot_ms + q_qjl_ms + op_ms + v_rot_ms:.3f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-bits", type=int, default=3)
    parser.add_argument("--v-bits", type=int, default=2)
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
    parser.add_argument("--score-tile-len", type=int, default=64)
    parser.add_argument("--profile-components", action="store_true")
    parser.add_argument(
        "--include-custom-op",
        action="store_true",
        help="Use tq_prod_mse_paged_attention custom op if present.",
    )
    args = parser.parse_args()

    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if args.score_tile_len <= 0 or args.score_tile_len > 256:
        raise ValueError("score_tile_len must be in [1, 256]")
    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available")

    os.environ["VLLM_ASCEND_TQ_USE_CUSTOM_DEQUANT"] = "1"
    os.environ["VLLM_ASCEND_TQ_USE_CUSTOM_K_SCORE"] = "1"
    os.environ["VLLM_ASCEND_TQ_USE_CUSTOM_ATTENTION"] = (
        "1" if args.include_custom_op else "0"
    )
    os.environ["VLLM_ASCEND_TQ_ATTENTION_SCORE_TILE_LEN"] = str(
        args.score_tile_len
    )
    os.environ["VLLM_ASCEND_TQ_DEBUG_COMPARE"] = "0"
    os.environ["VLLM_ASCEND_TQ_CUSTOM_STRICT"] = "1"
    try:
        from vllm_ascend.utils import enable_custom_op

        enable_custom_op()
    except Exception:
        pass

    has_attention = hasattr(torch.ops, "_C_ascend") and hasattr(
        torch.ops._C_ascend, "tq_prod_mse_paged_attention",
    )
    print(f"custom ops: fused_attention={has_attention}")
    _run_case(args)


if __name__ == "__main__":
    main()
