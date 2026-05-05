#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""Benchmark TurboQuant decode dequant paths through the attention backend.

This script builds a synthetic TurboQuant sidecar KV cache and exercises the
same backend methods used by decode:

  1. reference path: paged gather + PyTorch TurboQuant decode
  2. hybrid path: paged custom MSE dequant for V and K=prod stage1/QJL unpack

By default it times the dequant portion only. Pass ``--include-fia`` to also
run the dense FIA step after dequant, which is closer to end-to-end decode but
depends on the local torch_npu FIA dtype support.
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from types import SimpleNamespace

import torch

from vllm_ascend.attention.attention_v1 import AscendTurboQuantAttentionBackendImpl
from vllm_ascend.quantization.methods import turboquant_runtime as tq_runtime
from vllm_ascend.quantization.methods.turboquant_layout import get_stage1_bits
from vllm_ascend.quantization.methods.turboquant_runtime import (
    build_qjl_projection,
    build_rotation_matrix,
    build_turboquant_codebook,
    pack_bits,
)


def _sync() -> None:
    torch.npu.synchronize()


def _set_custom_mode(enabled: bool) -> None:
    os.environ["VLLM_ASCEND_TQ_USE_CUSTOM_DEQUANT"] = "1" if enabled else "0"
    os.environ["VLLM_ASCEND_TQ_USE_CUSTOM_PROD_DEQUANT"] = "0"
    os.environ["VLLM_ASCEND_TQ_DEBUG_COMPARE"] = "0"
    os.environ["VLLM_ASCEND_TQ_CUSTOM_STRICT"] = "1" if enabled else "0"


def _set_profile_mode(enabled: bool, profile_dir: str) -> None:
    os.environ["VLLM_ASCEND_TQ_PROFILE"] = "1" if enabled else "0"
    os.environ["VLLM_ASCEND_TQ_PROFILE_DIR"] = profile_dir
    os.environ["VLLM_ASCEND_TQ_PROFILE_FLUSH_EVERY"] = "1000000"


def _reset_profile_stats() -> None:
    with tq_runtime._TQ_PROFILE_LOCK:
        tq_runtime._TQ_PROFILE_STATS.clear()
        tq_runtime._TQ_PROFILE_UPDATE_COUNT = 0


def _profile_snapshot() -> dict[str, dict]:
    with tq_runtime._TQ_PROFILE_LOCK:
        return copy.deepcopy(tq_runtime._TQ_PROFILE_STATS)


def _dtype_from_name(name: str) -> torch.dtype:
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in dtypes:
        raise ValueError(f"Unsupported dtype: {name}")
    return dtypes[name]


def _make_impl(args) -> AscendTurboQuantAttentionBackendImpl:
    impl = object.__new__(AscendTurboQuantAttentionBackendImpl)
    impl.num_heads = args.num_heads
    impl.num_kv_heads = args.num_kv_heads
    impl.num_queries_per_kv = args.num_heads // args.num_kv_heads
    impl.head_size = args.head_dim
    impl.hidden_size = args.num_heads * args.head_dim
    impl.scale = args.head_dim ** -0.5
    impl.key_cache = None
    impl.value_cache = None
    impl.attn_type = ""
    impl.sliding_window = None
    impl.sinks = None
    return impl


def _make_layer(args, device: str) -> SimpleNamespace:
    scalar_dtype = torch.float32
    k_stage1_bits = get_stage1_bits(args.k_bits, args.k_variant)
    v_stage1_bits = get_stage1_bits(args.v_bits, "mse")
    k_codebook, k_boundary = build_turboquant_codebook(
        args.head_dim, k_stage1_bits, device, scalar_dtype,
    )
    v_codebook, v_boundary = build_turboquant_codebook(
        args.head_dim, v_stage1_bits, device, scalar_dtype,
    )
    k_rot = build_rotation_matrix(args.head_dim, args.seed + 11, device, scalar_dtype)
    v_rot = build_rotation_matrix(args.head_dim, args.seed + 12, device, scalar_dtype)
    k_qjl_proj = build_qjl_projection(args.head_dim, args.seed + 13, device, scalar_dtype)
    return SimpleNamespace(
        tq_runtime_prepared=False,
        tq_scalar_dtype=scalar_dtype,
        tq_k_variant=args.k_variant,
        tq_v_variant="mse",
        tq_k_total_bits=args.k_bits,
        tq_v_total_bits=args.v_bits,
        tq_k_stage1_bits=k_stage1_bits,
        tq_v_stage1_bits=v_stage1_bits,
        tq_head_size_v=args.head_dim,
        k_codebook=k_codebook,
        k_boundary=k_boundary,
        v_codebook=v_codebook,
        v_boundary=v_boundary,
        k_rot=k_rot,
        v_rot=v_rot,
        k_qjl_proj=k_qjl_proj,
    )


def _random_packed_cache(
    shape_prefix: tuple[int, int, int],
    head_dim: int,
    bits: int,
    device: str,
) -> torch.Tensor:
    levels = 1 << bits
    unpacked = torch.randint(
        0,
        levels,
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
            f"num_blocks={args.num_blocks} is too small for "
            f"batch_size={args.batch_size}, seq_len={args.seq_len}, "
            f"block_size={args.block_size}; need at least {required_blocks}."
        )
    block_ids = torch.randperm(args.num_blocks, device=device)[:required_blocks]
    block_table = block_ids.to(torch.int32).view(args.batch_size, blocks_per_seq)
    seq_lens = [args.seq_len] * args.batch_size
    return block_table.contiguous(), seq_lens


def _make_inputs(args, device: str):
    torch.manual_seed(args.seed)
    cache_shape = (args.num_blocks, args.block_size, args.num_kv_heads)
    k_stage1_bits = get_stage1_bits(args.k_bits, args.k_variant)
    v_stage1_bits = get_stage1_bits(args.v_bits, "mse")

    kv_cache = {
        "k_idx": _random_packed_cache(cache_shape, args.head_dim, k_stage1_bits, device),
        "k_norm": (0.1 + torch.rand((*cache_shape, 1), dtype=torch.float32, device=device)).contiguous(),
        "v_idx": _random_packed_cache(cache_shape, args.head_dim, v_stage1_bits, device),
        "v_norm": (0.1 + torch.rand((*cache_shape, 1), dtype=torch.float32, device=device)).contiguous(),
    }
    if args.k_variant == "prod":
        kv_cache["k_qjl"] = _random_packed_cache(cache_shape, args.head_dim, 1, device)
        kv_cache["k_gamma"] = (
            0.1 + torch.rand((*cache_shape, 1), dtype=torch.float32, device=device)
        ).contiguous()

    block_table, seq_lens = _make_block_table(args, device)
    query = torch.randn(
        args.batch_size,
        args.num_heads,
        args.head_dim,
        dtype=_dtype_from_name(args.query_dtype),
        device=device,
    )
    return kv_cache, block_table, seq_lens, query


def _run_decode_case(
    impl: AscendTurboQuantAttentionBackendImpl,
    kv_cache: dict[str, torch.Tensor],
    block_table: torch.Tensor,
    seq_lens: list[int],
    query: torch.Tensor,
    layer: SimpleNamespace,
    *,
    label: str,
    include_fia: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    tq_runtime._maybe_sync_for_profile(query, block_table, kv_cache)
    t0 = time.perf_counter()
    dense_k, dense_v = impl._dequant_paged_kv_to_dense(
        kv_cache,
        block_table,
        seq_lens,
        torch.float32,
        layer,
        profile_label=f"{label}.turboquant_decode",
    )
    if not include_fia:
        tq_runtime._maybe_sync_for_profile(dense_k, dense_v)
        tq_runtime._record_tq_profile(
            f"{label}.turboquant_decode.e2e_total",
            (time.perf_counter() - t0) * 1000.0,
            vectors=len(seq_lens),
        )
        return dense_k, dense_v

    impl.key_cache = kv_cache
    output = torch.empty_like(query)
    metadata = SimpleNamespace(attn_mask=None)
    actual_seq_lengths_q = torch.ones(
        len(seq_lens), dtype=torch.int32, device=query.device,
    ).cumsum(0).tolist()
    actual_seq_lengths_kv = torch.tensor(
        seq_lens, dtype=torch.int32, device=query.device,
    ).cumsum(0).tolist()
    out = impl._run_dense_fia(
        query,
        dense_k.to(query.dtype),
        dense_v.to(query.dtype),
        actual_seq_lengths_q,
        actual_seq_lengths_kv,
        metadata,
        output,
        profile_name=f"{label}.turboquant_decode.run_dense_fia",
    )
    tq_runtime._maybe_sync_for_profile(out)
    tq_runtime._record_tq_profile(
        f"{label}.turboquant_decode.e2e_total",
        (time.perf_counter() - t0) * 1000.0,
        vectors=len(seq_lens),
    )
    return out


def _bench(fn, *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync()
    return (time.perf_counter() - t0) * 1000.0 / iters


def _max_diff(lhs, rhs) -> float:
    if isinstance(lhs, tuple):
        return max(_max_diff(l, r) for l, r in zip(lhs, rhs))
    return (lhs - rhs).abs().max().item() if lhs.numel() else 0.0


def _run_mode(
    mode: str,
    custom_enabled: bool,
    impl: AscendTurboQuantAttentionBackendImpl,
    kv_cache: dict[str, torch.Tensor],
    block_table: torch.Tensor,
    seq_lens: list[int],
    query: torch.Tensor,
    layer: SimpleNamespace,
    args,
    *,
    enable_profile: bool,
):
    _set_custom_mode(custom_enabled)
    profile_dir = os.path.join(args.profile_dir, mode) if enable_profile else args.profile_dir
    _set_profile_mode(enable_profile, profile_dir)
    if enable_profile:
        _reset_profile_stats()

    def fn():
        return _run_decode_case(
            impl,
            kv_cache,
            block_table,
            seq_lens,
            query,
            layer,
            label=mode,
            include_fia=args.include_fia,
        )

    if enable_profile:
        for _ in range(args.warmup):
            fn()
        _sync()
        _reset_profile_stats()
        t0 = time.perf_counter()
        out = None
        for _ in range(args.iters):
            out = fn()
        _sync()
        ms = (time.perf_counter() - t0) * 1000.0 / args.iters
        stats = _profile_snapshot()
        tq_runtime._dump_tq_profile()
        return ms, out, stats

    ms = _bench(fn, warmup=args.warmup, iters=args.iters)
    out = fn()
    _sync()
    stats = {}
    return ms, out, stats


def _stat_avg_ms(stat: dict) -> float:
    calls = max(int(stat.get("calls", 0)), 1)
    return float(stat.get("total_ms", 0.0)) / calls


def _print_profile_rows(prefix: str, stats: dict[str, dict], rows: list[str]) -> None:
    print(f"{prefix} profile:")
    for name in rows:
        stat = stats.get(name)
        if stat is None:
            continue
        print(
            f"  {name}: avg={_stat_avg_ms(stat):.3f} ms "
            f"calls={stat.get('calls', 0)} "
            f"min={stat.get('min_ms', 0.0):.3f} "
            f"max={stat.get('max_ms', 0.0):.3f}"
        )


def _run_case(args) -> None:
    device = "npu"
    impl = _make_impl(args)
    layer = _make_layer(args, device)
    kv_cache, block_table, seq_lens, query = _make_inputs(args, device)

    _set_profile_mode(False, args.profile_dir)
    _set_custom_mode(False)
    ref_out = _run_decode_case(
        impl, kv_cache, block_table, seq_lens, query, layer,
        label="warm_reference", include_fia=args.include_fia,
    )
    _set_custom_mode(True)
    custom_out = _run_decode_case(
        impl, kv_cache, block_table, seq_lens, query, layer,
        label="warm_hybrid", include_fia=args.include_fia,
    )
    _sync()
    diff = _max_diff(custom_out, ref_out)

    _set_profile_mode(False, args.profile_dir)
    ref_ms, _, _ = _run_mode(
        "reference", False, impl, kv_cache, block_table, seq_lens, query, layer,
        args, enable_profile=False,
    )
    custom_ms, _, _ = _run_mode(
        "hybrid", True, impl, kv_cache, block_table, seq_lens, query, layer,
        args, enable_profile=False,
    )

    profile_ref_ms, _, ref_stats = _run_mode(
        "reference", False, impl, kv_cache, block_table, seq_lens, query, layer,
        args, enable_profile=True,
    )
    profile_custom_ms, _, custom_stats = _run_mode(
        "hybrid", True, impl, kv_cache, block_table, seq_lens, query, layer,
        args, enable_profile=True,
    )

    total_tokens = args.batch_size * args.seq_len
    mode = "decode+fia" if args.include_fia else "dequant"
    stream_overlap = args.stream_overlap or os.getenv("VLLM_ASCEND_TQ_STREAM_OVERLAP", "0") == "1"
    print(
        f"mode={mode} k_bits={args.k_bits} v_bits={args.v_bits} "
        f"k_variant={args.k_variant} batch={args.batch_size} "
        f"seq_len={args.seq_len} tokens={total_tokens} "
        f"heads={args.num_heads}/{args.num_kv_heads} head_dim={args.head_dim} "
        f"stream_overlap={stream_overlap}"
    )
    print(
        f"timing_no_profile: reference={ref_ms:.3f} ms "
        f"hybrid={custom_ms:.3f} ms speedup={ref_ms / custom_ms:.2f}x "
        f"max_diff={diff:.3g}"
    )
    print(
        f"timing_with_profile: reference={profile_ref_ms:.3f} ms "
        f"hybrid={profile_custom_ms:.3f} ms"
    )

    ref_rows = [
        "reference.turboquant_decode.gather.k_idx",
        "reference.turboquant_decode.gather.k_qjl",
        "reference.turboquant_decode.gather.k_gamma",
        "reference.turboquant_decode.gather.k_norm",
        "reference.turboquant_decode.gather.v_idx",
        "reference.turboquant_decode.gather.v_norm",
        "turboquant_decode_prod.total",
        "turboquant_decode_mse.total",
        "reference.turboquant_decode.run_dense_fia",
        "reference.turboquant_decode.total",
        "reference.turboquant_decode.e2e_total",
    ]
    hybrid_rows = [
        "hybrid.turboquant_decode.token_map",
        "hybrid.turboquant_decode.hybrid_prod.k",
        "hybrid.turboquant_decode.hybrid_prod.k.stage1_mse",
        "hybrid.turboquant_decode.hybrid_prod.k.qjl_scale_cache",
        "hybrid.turboquant_decode.hybrid_prod.k.qjl_unpack_scale",
        "hybrid.turboquant_decode.hybrid_prod.k.qjl_project",
        "hybrid.turboquant_decode.hybrid_prod.k.combine",
        "hybrid.turboquant_decode.hybrid_prod.k.inverse_rotate",
        "hybrid.turboquant_decode.custom_mse.k",
        "hybrid.turboquant_decode.custom_mse.k.paged_rot",
        "hybrid.turboquant_decode.custom_mse.k.inverse_rotate",
        "hybrid.turboquant_decode.custom_mse.v",
        "hybrid.turboquant_decode.custom_mse.v.paged_rot",
        "hybrid.turboquant_decode.custom_mse.v.inverse_rotate",
        "hybrid.turboquant_decode.custom_mse.total",
        "hybrid.turboquant_decode.run_dense_fia",
        "hybrid.turboquant_decode.e2e_total",
    ]
    _print_profile_rows("reference", ref_stats, ref_rows)
    _print_profile_rows("hybrid", custom_stats, hybrid_rows)
    print(f"profile_dir={args.profile_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-bits", type=int, default=3)
    parser.add_argument("--v-bits", type=int, default=2)
    parser.add_argument("--k-variant", choices=["prod", "mse"], default="prod")
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--query-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--profile-dir", default="/tmp/turboquant_decode_profile")
    parser.add_argument(
        "--stream-overlap",
        action="store_true",
        help="Enable experimental K/V dequant stream overlap for no-profile timing.",
    )
    parser.add_argument(
        "--include-fia",
        action="store_true",
        help="Also run dense FIA after dequant for a fuller decode path.",
    )
    args = parser.parse_args()

    if args.num_heads % args.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available")

    if args.stream_overlap:
        os.environ["VLLM_ASCEND_TQ_STREAM_OVERLAP"] = "1"

    try:
        from vllm_ascend.utils import enable_custom_op

        enable_custom_op()
    except Exception:
        pass

    has_mse = hasattr(torch.ops, "_C_ascend") and hasattr(
        torch.ops._C_ascend, "tq_dequant_mse_paged",
    )
    print(f"custom op mse={has_mse}")
    _run_case(args)


if __name__ == "__main__":
    main()
