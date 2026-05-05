#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Microbenchmark TurboQuant K=prod paged dequant paths on NPU.

Compares:
  1. full PyTorch reference decode_prod
  2. hybrid path: custom MSE stage1 + PyTorch QJL correction
  3. experimental custom prod op: MSE stage1 + QJL correction in Ascend C
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch

from vllm_ascend.ops.turboquant.dequant import (
    tq_dequant_mse_paged_rot,
    tq_dequant_prod_paged_rot,
)
from vllm_ascend.quantization.methods.turboquant_layout import get_stage1_bits
from vllm_ascend.quantization.methods.turboquant_runtime import (
    apply_rotation,
    build_qjl_projection,
    build_rotation_matrix,
    build_turboquant_codebook,
    pack_bits,
    turboquant_decode_prod,
    unpack_bits,
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


def _make_inputs(args, total_bits: int):
    device = "npu"
    stage1_bits = get_stage1_bits(total_bits, "prod")
    levels = 1 << stage1_bits

    idx = torch.randint(
        0,
        levels,
        (
            args.num_blocks,
            args.block_size,
            args.num_kv_heads,
            args.head_dim,
        ),
        dtype=torch.uint8,
        device=device,
    )
    packed_idx = pack_bits(idx, bits=stage1_bits).contiguous()

    qjl = torch.randint(
        0,
        2,
        (
            args.num_blocks,
            args.block_size,
            args.num_kv_heads,
            args.head_dim,
        ),
        dtype=torch.uint8,
        device=device,
    )
    packed_qjl = pack_bits(qjl, bits=1).contiguous()

    gamma = torch.rand(
        (args.num_blocks, args.block_size, args.num_kv_heads, 1),
        dtype=torch.float32,
        device=device,
    )
    norm = torch.rand(
        (args.num_blocks, args.block_size, args.num_kv_heads, 1),
        dtype=torch.float32,
        device=device,
    )
    token_block_ids = torch.randint(
        0,
        args.num_blocks,
        (args.total_tokens,),
        dtype=torch.int32,
        device=device,
    )
    token_offsets = torch.randint(
        0,
        args.block_size,
        (args.total_tokens,),
        dtype=torch.int32,
        device=device,
    )

    codebook, _ = build_turboquant_codebook(
        args.head_dim, stage1_bits, device, torch.float32,
    )
    qjl_codebook = torch.tensor([-1.0, 1.0], dtype=torch.float32, device=device)
    qjl_proj = build_qjl_projection(args.head_dim, args.seed + 1, device, torch.float32)
    rotation = build_rotation_matrix(args.head_dim, args.seed + 2, device, torch.float32)
    rotation_t = rotation.transpose(0, 1).contiguous()
    return (
        packed_idx,
        packed_qjl,
        gamma,
        norm,
        token_block_ids,
        token_offsets,
        codebook,
        qjl_codebook,
        qjl_proj,
        rotation_t,
    )


def _gather(cache, token_block_ids, token_offsets):
    return cache[token_block_ids.long(), token_offsets.long()]


def _run_case(args, total_bits: int) -> None:
    (
        packed_idx,
        packed_qjl,
        gamma,
        norm,
        token_block_ids,
        token_offsets,
        codebook,
        qjl_codebook,
        qjl_proj,
        rotation_t,
    ) = _make_inputs(args, total_bits)

    stage1_bits = get_stage1_bits(total_bits, "prod")

    def ref_fn():
        return turboquant_decode_prod(
            _gather(packed_idx, token_block_ids, token_offsets),
            _gather(packed_qjl, token_block_ids, token_offsets),
            _gather(gamma, token_block_ids, token_offsets),
            _gather(norm, token_block_ids, token_offsets),
            rotation_t,
            codebook,
            qjl_proj,
            total_bits,
            args.head_dim,
            torch.float32,
        )

    def torch_hybrid_fn():
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
        gathered_qjl = _gather(packed_qjl, token_block_ids, token_offsets)
        gathered_gamma = _gather(gamma, token_block_ids, token_offsets)
        gathered_norm = _gather(norm, token_block_ids, token_offsets)
        qjl = unpack_bits(gathered_qjl, bits=1, dim=args.head_dim).to(torch.float32)
        qjl = qjl * 2.0 - 1.0
        correction = math.sqrt(math.pi / 2.0) / args.head_dim
        qjl_rot = correction * gathered_gamma * gathered_norm * apply_rotation(qjl, qjl_proj)
        return apply_rotation(stage1_rot + qjl_rot, rotation_t).contiguous()

    def hybrid_fn():
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
        qjl_scale_cache = (correction * gamma * norm).contiguous()
        qjl_scaled = tq_dequant_mse_paged_rot(
            packed_qjl,
            qjl_scale_cache,
            token_block_ids,
            token_offsets,
            qjl_codebook,
            1,
            args.head_dim,
            torch.float32,
        )
        qjl_rot = apply_rotation(qjl_scaled, qjl_proj)
        return apply_rotation(stage1_rot + qjl_rot, rotation_t).contiguous()

    def custom_prod_fn():
        dense_rot = tq_dequant_prod_paged_rot(
            packed_idx,
            packed_qjl,
            gamma,
            norm,
            token_block_ids,
            token_offsets,
            codebook,
            qjl_proj,
            total_bits,
            args.head_dim,
            torch.float32,
        )
        return apply_rotation(dense_rot, rotation_t).contiguous()

    ref = ref_fn()
    torch_hybrid = torch_hybrid_fn()
    hybrid = hybrid_fn()
    custom_prod = custom_prod_fn()
    _sync()

    ref_ms = _bench(ref_fn, warmup=args.warmup, iters=args.iters)
    torch_hybrid_ms = _bench(torch_hybrid_fn, warmup=args.warmup, iters=args.iters)
    hybrid_ms = _bench(hybrid_fn, warmup=args.warmup, iters=args.iters)
    custom_ms = _bench(custom_prod_fn, warmup=args.warmup, iters=args.iters)

    torch_hybrid_diff = (torch_hybrid - ref).abs().max().item() if ref.numel() else 0.0
    hybrid_diff = (hybrid - ref).abs().max().item() if ref.numel() else 0.0
    custom_diff = (custom_prod - ref).abs().max().item() if ref.numel() else 0.0

    print(
        f"bits={total_bits} "
        f"ref={ref_ms:.3f} ms "
        f"torch_hybrid={torch_hybrid_ms:.3f} ms "
        f"hybrid={hybrid_ms:.3f} ms "
        f"prod_custom={custom_ms:.3f} ms "
        f"torch_hybrid_speedup={ref_ms / torch_hybrid_ms:.2f}x "
        f"hybrid_speedup={ref_ms / hybrid_ms:.2f}x "
        f"hybrid_vs_torch={torch_hybrid_ms / hybrid_ms:.2f}x "
        f"prod_speedup={ref_ms / custom_ms:.2f}x "
        f"custom_vs_hybrid={hybrid_ms / custom_ms:.2f}x "
        f"torch_hybrid_diff={torch_hybrid_diff:.3g} "
        f"hybrid_diff={hybrid_diff:.3g} "
        f"prod_diff={custom_diff:.3g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--total-tokens", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available")

    os.environ["VLLM_ASCEND_TQ_USE_CUSTOM_DEQUANT"] = "1"
    os.environ["VLLM_ASCEND_TQ_USE_CUSTOM_PROD_DEQUANT"] = "1"
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
    has_prod = hasattr(torch.ops, "_C_ascend") and hasattr(
        torch.ops._C_ascend, "tq_dequant_prod_paged",
    )
    print(
        f"custom ops: mse={has_mse} prod={has_prod} "
        f"tokens={args.total_tokens} heads={args.num_kv_heads} "
        f"head_dim={args.head_dim}"
    )

    torch.manual_seed(args.seed)
    for total_bits in args.bits:
        _run_case(args, total_bits)


if __name__ == "__main__":
    main()
