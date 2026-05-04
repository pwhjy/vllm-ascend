#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

from __future__ import annotations

import atexit
import json
import math
import os
import socket
import threading
import time

import torch

_CODEBOOK_CACHE: dict[tuple[int, int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

# =========================
# TurboQuant profiling
# =========================
_TQ_PROFILE_LOCK = threading.Lock()
_TQ_PROFILE_STATS: dict[str, dict] = {}
_TQ_PROFILE_FLUSH_EVERY = max(1, int(os.getenv("VLLM_ASCEND_TQ_PROFILE_FLUSH_EVERY", "100")))
_TQ_PROFILE_UPDATE_COUNT = 0


def _tq_profile_enabled() -> bool:
    return os.getenv("VLLM_ASCEND_TQ_PROFILE", "0") == "1"


def _tq_profile_dir() -> str:
    return os.getenv("VLLM_ASCEND_TQ_PROFILE_DIR", "/tmp/turboquant_profile")


def _is_npu_tensor(x) -> bool:
    return isinstance(x, torch.Tensor) and hasattr(x, "is_npu") and x.is_npu


def _maybe_sync_for_profile(*objs):
    if not _tq_profile_enabled():
        return

    def _walk(obj):
        if _is_npu_tensor(obj):
            return [obj]
        if isinstance(obj, (list, tuple)):
            out = []
            for item in obj:
                out.extend(_walk(item))
            return out
        if isinstance(obj, dict):
            out = []
            for item in obj.values():
                out.extend(_walk(item))
            return out
        return []

    tensors = []
    for obj in objs:
        tensors.extend(_walk(obj))

    for tensor in tensors:
        if _is_npu_tensor(tensor):
            torch.npu.synchronize(tensor.device)
            break


def _record_tq_profile(name: str, elapsed_ms: float, **meta):
    global _TQ_PROFILE_UPDATE_COUNT

    if not _tq_profile_enabled():
        return

    with _TQ_PROFILE_LOCK:
        stat = _TQ_PROFILE_STATS.setdefault(
            name,
            {
                "calls": 0,
                "total_ms": 0.0,
                "min_ms": None,
                "max_ms": 0.0,
                "vectors": 0,
                "elements": 0,
                "bytes_out": 0,
            },
        )
        stat["calls"] += 1
        stat["total_ms"] += float(elapsed_ms)
        stat["max_ms"] = max(stat["max_ms"], float(elapsed_ms))
        stat["min_ms"] = float(elapsed_ms) if stat["min_ms"] is None else min(stat["min_ms"], float(elapsed_ms))

        for key in ("vectors", "elements", "bytes_out"):
            if key in meta and meta[key] is not None:
                stat[key] += int(meta[key])

        _TQ_PROFILE_UPDATE_COUNT += 1
        if _TQ_PROFILE_UPDATE_COUNT % _TQ_PROFILE_FLUSH_EVERY == 0:
            _dump_tq_profile_locked()


def _dump_tq_profile():
    if not _tq_profile_enabled():
        return

    with _TQ_PROFILE_LOCK:
        _dump_tq_profile_locked()


def _dump_tq_profile_locked():
    if not _TQ_PROFILE_STATS:
        return

    profile_dir = _tq_profile_dir()
    os.makedirs(profile_dir, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "stats": _TQ_PROFILE_STATS,
    }
    out_path = os.path.join(profile_dir, f"turboquant_profile_{os.getpid()}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


atexit.register(_dump_tq_profile)


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


def _interp_from_cdf(grid: torch.Tensor, cdf: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    indices = torch.searchsorted(cdf, quantiles)
    indices = torch.clamp(indices, 1, grid.numel() - 1)
    left_idx = indices - 1
    right_idx = indices
    left_cdf = cdf[left_idx]
    right_cdf = cdf[right_idx]
    left_grid = grid[left_idx]
    right_grid = grid[right_idx]
    denom = torch.clamp(right_cdf - left_cdf, min=1e-18)
    ratio = (quantiles - left_cdf) / denom
    return left_grid + ratio * (right_grid - left_grid)


def _beta_coordinate_pdf(head_dim: int, grid: torch.Tensor) -> torch.Tensor:
    if head_dim <= 1:
        raise ValueError(f"head_dim must be > 1 for TurboQuant codebook, got {head_dim}")
    eps = torch.finfo(grid.dtype).tiny
    support = torch.clamp(1.0 - grid.square(), min=eps)
    exponent = 0.5 * (head_dim - 3)
    log_const = (
        math.lgamma(head_dim / 2.0)
        - 0.5 * math.log(math.pi)
        - math.lgamma((head_dim - 1) / 2.0)
    )
    pdf = torch.exp(log_const + exponent * torch.log(support))
    pdf = torch.where(grid.abs() >= 1.0, torch.zeros_like(pdf), pdf)
    return pdf


def build_beta_lloyd_max_codebook(
    head_dim: int,
    bits: int,
    *,
    num_grid_points: int = 65537,
    max_iters: int = 96,
    tol: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if bits <= 0:
        raise ValueError(f"bits must be > 0, got {bits}")
    if num_grid_points < 1025:
        raise ValueError(f"num_grid_points must be >= 1025, got {num_grid_points}")

    levels = 2**bits
    grid = torch.linspace(-1.0, 1.0, steps=num_grid_points, dtype=torch.float64, device="cpu")
    pdf = _beta_coordinate_pdf(head_dim, grid)
    cdf = torch.cumsum(pdf, dim=0)
    cdf = cdf / torch.clamp(cdf[-1], min=1e-18)
    quantiles = (torch.arange(levels, dtype=torch.float64, device=grid.device) + 0.5) / levels
    codebook = _interp_from_cdf(grid, cdf, quantiles)

    for _ in range(max_iters):
        boundary = 0.5 * (codebook[:-1] + codebook[1:])
        assignments = torch.bucketize(grid, boundary)
        updated = torch.empty_like(codebook)
        for bucket in range(levels):
            mask = assignments == bucket
            bucket_pdf = pdf[mask]
            if bucket_pdf.numel() == 0:
                updated[bucket] = codebook[bucket]
                continue
            weight_sum = torch.sum(bucket_pdf)
            if torch.abs(weight_sum) <= 1e-18:
                updated[bucket] = codebook[bucket]
                continue
            updated[bucket] = torch.sum(bucket_pdf * grid[mask]) / weight_sum
        updated = torch.sort(torch.clamp(updated, -1.0, 1.0)).values
        if torch.max(torch.abs(updated - codebook)).item() < tol:
            codebook = updated
            break
        codebook = updated

    boundary = 0.5 * (codebook[:-1] + codebook[1:])
    return codebook.contiguous(), boundary.contiguous()


def get_cached_codebook(
    head_dim: int,
    bits: int,
    *,
    num_grid_points: int = 65537,
    max_iters: int = 96,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_key = (head_dim, bits, num_grid_points, max_iters)
    cached = _CODEBOOK_CACHE.get(cache_key)
    if cached is None:
        cached = build_beta_lloyd_max_codebook(
            head_dim,
            bits,
            num_grid_points=num_grid_points,
            max_iters=max_iters,
        )
        _CODEBOOK_CACHE[cache_key] = cached
    return cached


def build_turboquant_codebook(head_dim: int, bits: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    codebook, boundary = get_cached_codebook(head_dim, bits)
    return (
        codebook.to(device=device, dtype=dtype).contiguous(),
        boundary.to(device=device, dtype=dtype).contiguous(),
    )


def build_rotation_matrix(head_dim: int, seed: int, device, dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_matrix = torch.randn(head_dim, head_dim, generator=generator, dtype=torch.float32, device="cpu")
    q, _ = torch.linalg.qr(random_matrix, mode="reduced")
    return q.to(device=device, dtype=dtype).contiguous()


def build_qjl_projection(head_dim: int, seed: int, device, dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_matrix = torch.randn(head_dim, head_dim, generator=generator, dtype=torch.float32, device="cpu")
    return random_matrix.to(device=device, dtype=dtype).contiguous()


def _pack_bits_fast(flat: torch.Tensor, bits: int) -> torch.Tensor | None:
    if bits == 1:
        values = flat.to(torch.uint8)
        pad = (-values.shape[-1]) % 8
        if pad:
            padded = torch.zeros((values.shape[0], values.shape[-1] + pad), dtype=torch.uint8, device=values.device)
            padded[:, : values.shape[-1]] = values
            values = padded
        chunks = (values & 0x1).view(values.shape[0], -1, 8)
        packed = (
            chunks[..., 0]
            | (chunks[..., 1] << 1)
            | (chunks[..., 2] << 2)
            | (chunks[..., 3] << 3)
            | (chunks[..., 4] << 4)
            | (chunks[..., 5] << 5)
            | (chunks[..., 6] << 6)
            | (chunks[..., 7] << 7)
        )
        return packed.contiguous()

    if bits == 2:
        values = flat.to(torch.uint8) & 0x3
        pad = (-values.shape[-1]) % 4
        if pad:
            padded = torch.zeros((values.shape[0], values.shape[-1] + pad), dtype=torch.uint8, device=values.device)
            padded[:, : values.shape[-1]] = values
            values = padded
        chunks = values.view(values.shape[0], -1, 4)
        packed = (
            chunks[..., 0]
            | (chunks[..., 1] << 2)
            | (chunks[..., 2] << 4)
            | (chunks[..., 3] << 6)
        )
        return packed.contiguous()

    if bits == 3:
        values = flat.to(torch.uint8) & 0x7
        pad = (-values.shape[-1]) % 8
        if pad:
            padded = torch.zeros((values.shape[0], values.shape[-1] + pad), dtype=torch.uint8, device=values.device)
            padded[:, : values.shape[-1]] = values
            values = padded
        chunks = values.view(values.shape[0], -1, 8)
        byte0 = chunks[..., 0] | (chunks[..., 1] << 3) | (chunks[..., 2] << 6)
        byte1 = (chunks[..., 2] >> 2) | (chunks[..., 3] << 1) | (chunks[..., 4] << 4) | (chunks[..., 5] << 7)
        byte2 = (chunks[..., 5] >> 1) | (chunks[..., 6] << 2) | (chunks[..., 7] << 5)
        packed = torch.stack([byte0, byte1, byte2], dim=-1).view(values.shape[0], -1)
        return packed.contiguous()

    if bits == 4:
        values = flat.to(torch.uint8) & 0xF
        pad = (-values.shape[-1]) % 2
        if pad:
            padded = torch.zeros((values.shape[0], values.shape[-1] + pad), dtype=torch.uint8, device=values.device)
            padded[:, : values.shape[-1]] = values
            values = padded
        chunks = values.view(values.shape[0], -1, 2)
        packed = chunks[..., 0] | (chunks[..., 1] << 4)
        return packed.contiguous()

    return None


def pack_bits(indices: torch.Tensor, bits: int) -> torch.Tensor:
    _maybe_sync_for_profile(indices)
    t0 = time.perf_counter()

    if indices.numel() == 0:
        packed = indices.to(torch.uint8)
        _maybe_sync_for_profile(packed)
        _record_tq_profile("pack_bits", (time.perf_counter() - t0) * 1000.0, elements=0, bytes_out=0)
        return packed

    original_shape = indices.shape[:-1]
    flat_u8 = indices.to(torch.uint8).reshape(-1, indices.shape[-1])
    packed_cols = (flat_u8.shape[-1] * bits + 7) // 8
    packed_fast = _pack_bits_fast(flat_u8, bits)
    if packed_fast is not None:
        packed = packed_fast[:, :packed_cols]
    else:
        if indices.dtype not in (torch.uint8, torch.int16, torch.int32, torch.int64):
            indices = indices.to(torch.int64)
        values = indices.to(torch.int64)
        flat = values.reshape(-1, values.shape[-1])
        packed = torch.zeros((flat.shape[0], packed_cols), dtype=torch.uint8, device=flat.device)
        bit_cursor = 0
        mask = (1 << bits) - 1
        for col in range(flat.shape[-1]):
            value = flat[:, col] & mask
            byte_index = bit_cursor // 8
            bit_offset = bit_cursor % 8
            packed[:, byte_index] |= (value << bit_offset).to(torch.uint8)
            if bit_offset + bits > 8:
                packed[:, byte_index + 1] |= (value >> (8 - bit_offset)).to(torch.uint8)
            bit_cursor += bits
    packed = packed.reshape(*original_shape, packed_cols)

    _maybe_sync_for_profile(packed)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _record_tq_profile(
        "pack_bits",
        elapsed_ms,
        elements=flat_u8.numel(),
        bytes_out=packed.numel(),
    )
    return packed


def _unpack_bits_fast(flat: torch.Tensor, bits: int, dim: int) -> torch.Tensor | None:
    values = flat.to(torch.uint8)

    if bits == 1:
        unpacked = torch.empty((values.shape[0], values.shape[1] * 8), dtype=torch.uint8, device=values.device)
        unpacked[:, 0::8] = values & 0x1
        unpacked[:, 1::8] = (values >> 1) & 0x1
        unpacked[:, 2::8] = (values >> 2) & 0x1
        unpacked[:, 3::8] = (values >> 3) & 0x1
        unpacked[:, 4::8] = (values >> 4) & 0x1
        unpacked[:, 5::8] = (values >> 5) & 0x1
        unpacked[:, 6::8] = (values >> 6) & 0x1
        unpacked[:, 7::8] = (values >> 7) & 0x1
        return unpacked[:, :dim].contiguous()

    if bits == 2:
        unpacked = torch.empty((values.shape[0], values.shape[1] * 4), dtype=torch.uint8, device=values.device)
        unpacked[:, 0::4] = values & 0x3
        unpacked[:, 1::4] = (values >> 2) & 0x3
        unpacked[:, 2::4] = (values >> 4) & 0x3
        unpacked[:, 3::4] = (values >> 6) & 0x3
        return unpacked[:, :dim].contiguous()

    if bits == 3:
        n_groups = values.shape[1] // 3
        unpacked = torch.empty((values.shape[0], n_groups * 8), dtype=torch.uint8, device=values.device)
        chunks = values.view(values.shape[0], n_groups, 3)
        byte0 = chunks[..., 0]
        byte1 = chunks[..., 1]
        byte2 = chunks[..., 2]
        unpacked[:, 0::8] = byte0 & 0x7
        unpacked[:, 1::8] = (byte0 >> 3) & 0x7
        unpacked[:, 2::8] = ((byte0 >> 6) | (byte1 << 2)) & 0x7
        unpacked[:, 3::8] = (byte1 >> 1) & 0x7
        unpacked[:, 4::8] = (byte1 >> 4) & 0x7
        unpacked[:, 5::8] = ((byte1 >> 7) | (byte2 << 1)) & 0x7
        unpacked[:, 6::8] = (byte2 >> 2) & 0x7
        unpacked[:, 7::8] = (byte2 >> 5) & 0x7
        return unpacked[:, :dim].contiguous()

    if bits == 4:
        unpacked = torch.empty((values.shape[0], values.shape[1] * 2), dtype=torch.uint8, device=values.device)
        unpacked[:, 0::2] = values & 0xF
        unpacked[:, 1::2] = (values >> 4) & 0xF
        return unpacked[:, :dim].contiguous()

    return None


def unpack_bits(packed: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    _maybe_sync_for_profile(packed)
    t0 = time.perf_counter()

    if packed.numel() == 0:
        unpacked = packed.to(torch.uint8)
        _maybe_sync_for_profile(unpacked)
        _record_tq_profile("unpack_bits", (time.perf_counter() - t0) * 1000.0, elements=0, bytes_out=0)
        return unpacked

    original_shape = packed.shape[:-1]
    flat_u8 = packed.to(torch.uint8).reshape(-1, packed.shape[-1])
    unpacked_fast = _unpack_bits_fast(flat_u8, bits, dim)
    if unpacked_fast is not None:
        unpacked = unpacked_fast
    else:
        values = packed.to(torch.int64)
        flat = values.reshape(-1, values.shape[-1])
        unpacked = torch.zeros((flat.shape[0], dim), dtype=torch.int64, device=flat.device)
        bit_cursor = 0
        mask = (1 << bits) - 1
        for col in range(dim):
            byte_index = bit_cursor // 8
            bit_offset = bit_cursor % 8
            value = (flat[:, byte_index] >> bit_offset) & mask
            if bit_offset + bits > 8:
                spill_bits = bit_offset + bits - 8
                spill_mask = (1 << spill_bits) - 1
                value |= (flat[:, byte_index + 1] & spill_mask) << (8 - bit_offset)
            unpacked[:, col] = value
            bit_cursor += bits
        unpacked = unpacked.to(torch.uint8)
    unpacked = unpacked.reshape(*original_shape, dim).to(torch.uint8)

    _maybe_sync_for_profile(unpacked)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _record_tq_profile(
        "unpack_bits",
        elapsed_ms,
        elements=unpacked.numel(),
        bytes_out=unpacked.numel(),
    )
    return unpacked


def apply_rotation(x: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    original_shape = x.shape
    flat = x.reshape(-1, original_shape[-1])
    rotated = flat @ rotation
    return rotated.reshape(original_shape)


def _encode_scalar_with_boundary(x: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
    return torch.sum(x.unsqueeze(-1) > boundary, dim=-1).to(torch.uint8)


def _dequant_scalar(idx: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    return codebook[idx.long()]


def turboquant_encode_mse(
    x: torch.Tensor,
    rotation: torch.Tensor,
    codebook: torch.Tensor,
    boundary: torch.Tensor,
    bits: int,
) -> dict[str, torch.Tensor]:
    num_vectors = int(x.numel() // x.shape[-1])

    _maybe_sync_for_profile(x, rotation, codebook, boundary)
    t_total_0 = time.perf_counter()

    t0 = time.perf_counter()
    norm = torch.clamp(x.norm(dim=-1, keepdim=True), min=1e-6)
    x_unit = x / norm
    _maybe_sync_for_profile(norm, x_unit)
    _record_tq_profile(
        "turboquant_encode_mse.norm_unit",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    t0 = time.perf_counter()
    x_rot = apply_rotation(x_unit, rotation)
    _maybe_sync_for_profile(x_rot)
    _record_tq_profile(
        "turboquant_encode_mse.rotate",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    t0 = time.perf_counter()
    idx = _encode_scalar_with_boundary(x_rot, boundary)
    _maybe_sync_for_profile(idx)
    _record_tq_profile(
        "turboquant_encode_mse.boundary_encode",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    t0 = time.perf_counter()
    x_hat_rot = _dequant_scalar(idx, codebook).to(x.dtype)
    _maybe_sync_for_profile(x_hat_rot)
    _record_tq_profile(
        "turboquant_encode_mse.stage1_dequant",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    t0 = time.perf_counter()
    packed_idx = pack_bits(idx, bits)
    _maybe_sync_for_profile(packed_idx)
    _record_tq_profile(
        "turboquant_encode_mse.pack_idx_only",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    _maybe_sync_for_profile(packed_idx, norm, x_hat_rot, x_rot)
    total_ms = (time.perf_counter() - t_total_0) * 1000.0
    _record_tq_profile(
        "turboquant_encode_mse.total",
        total_ms,
        vectors=num_vectors,
    )
    return {
        "idx": packed_idx,
        "norm": norm.to(torch.float32),
        "x_hat_rot": x_hat_rot,
        "x_rot": x_rot,
    }


def turboquant_encode_prod(
    x: torch.Tensor,
    rotation: torch.Tensor,
    codebook: torch.Tensor,
    boundary: torch.Tensor,
    qjl_proj: torch.Tensor,
    total_bits: int,
) -> dict[str, torch.Tensor]:
    num_vectors = int(x.numel() // x.shape[-1])

    _maybe_sync_for_profile(x, rotation, codebook, boundary, qjl_proj)
    t_total_0 = time.perf_counter()

    stage1_bits = get_stage1_bits(total_bits, "prod")

    t0 = time.perf_counter()
    mse = turboquant_encode_mse(x, rotation, codebook, boundary, stage1_bits)
    _maybe_sync_for_profile(mse)
    _record_tq_profile(
        "turboquant_encode_prod.stage1_mse",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    t0 = time.perf_counter()
    residual = mse["x_rot"] - mse["x_hat_rot"]
    gamma = residual.norm(dim=-1, keepdim=True).to(torch.float32)
    _maybe_sync_for_profile(residual, gamma)
    _record_tq_profile(
        "turboquant_encode_prod.residual_gamma",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    t0 = time.perf_counter()
    qjl_sign = (apply_rotation(residual, qjl_proj.transpose(0, 1)) >= 0).to(torch.uint8)
    _maybe_sync_for_profile(qjl_sign)
    _record_tq_profile(
        "turboquant_encode_prod.qjl_rotate_sign",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    t0 = time.perf_counter()
    packed_qjl = pack_bits(qjl_sign, 1)
    _maybe_sync_for_profile(packed_qjl)
    _record_tq_profile(
        "turboquant_encode_prod.pack_qjl_only",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    _maybe_sync_for_profile(packed_qjl, gamma, mse["idx"], mse["norm"])
    total_ms = (time.perf_counter() - t_total_0) * 1000.0
    _record_tq_profile(
        "turboquant_encode_prod.total",
        total_ms,
        vectors=num_vectors,
    )
    return {
        "idx": mse["idx"],
        "qjl": packed_qjl,
        "gamma": gamma,
        "norm": mse["norm"],
    }


def turboquant_decode_mse(
    packed_idx: torch.Tensor,
    norm: torch.Tensor,
    rotation_t: torch.Tensor,
    codebook: torch.Tensor,
    bits: int,
    dim: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    num_vectors = int(norm.numel())

    _maybe_sync_for_profile(packed_idx, norm, rotation_t, codebook)
    t_total_0 = time.perf_counter()

    t0 = time.perf_counter()
    idx = unpack_bits(packed_idx, bits, dim)
    _maybe_sync_for_profile(idx)
    _record_tq_profile(
        "turboquant_decode_mse.unpack_idx",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    t0 = time.perf_counter()
    x_hat_rot = _dequant_scalar(idx, codebook).to(target_dtype)
    _maybe_sync_for_profile(x_hat_rot)
    _record_tq_profile(
        "turboquant_decode_mse.stage1_dequant",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    t0 = time.perf_counter()
    x_hat = apply_rotation(x_hat_rot, rotation_t)
    _maybe_sync_for_profile(x_hat)
    _record_tq_profile(
        "turboquant_decode_mse.inverse_rotate",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    t0 = time.perf_counter()
    decoded = x_hat * norm.to(target_dtype)
    _maybe_sync_for_profile(decoded)
    _record_tq_profile(
        "turboquant_decode_mse.rescale_norm",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    _record_tq_profile(
        "turboquant_decode_mse.total",
        (time.perf_counter() - t_total_0) * 1000.0,
        vectors=num_vectors,
    )
    return decoded


def turboquant_decode_prod(
    packed_idx: torch.Tensor,
    packed_qjl: torch.Tensor,
    gamma: torch.Tensor,
    norm: torch.Tensor,
    rotation_t: torch.Tensor,
    codebook: torch.Tensor,
    qjl_proj: torch.Tensor,
    total_bits: int,
    dim: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    num_vectors = int(norm.numel())

    _maybe_sync_for_profile(packed_idx, packed_qjl, gamma, norm, rotation_t, codebook, qjl_proj)
    t_total_0 = time.perf_counter()

    stage1_bits = get_stage1_bits(total_bits, "prod")

    t0 = time.perf_counter()
    idx = unpack_bits(packed_idx, stage1_bits, dim)
    _maybe_sync_for_profile(idx)
    _record_tq_profile(
        "turboquant_decode_prod.unpack_idx",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    t0 = time.perf_counter()
    qjl = unpack_bits(packed_qjl, 1, dim).to(target_dtype)
    _maybe_sync_for_profile(qjl)
    _record_tq_profile(
        "turboquant_decode_prod.unpack_qjl",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    t0 = time.perf_counter()
    qjl = qjl * 2.0 - 1.0
    _maybe_sync_for_profile(qjl)
    _record_tq_profile(
        "turboquant_decode_prod.qjl_sign_to_pm1",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    t0 = time.perf_counter()
    x_mse_hat_rot = _dequant_scalar(idx, codebook).to(target_dtype)
    _maybe_sync_for_profile(x_mse_hat_rot)
    _record_tq_profile(
        "turboquant_decode_prod.stage1_dequant",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    correction = math.sqrt(math.pi / 2.0) / dim

    t0 = time.perf_counter()
    x_qjl_hat_rot = correction * gamma.to(target_dtype) * apply_rotation(qjl, qjl_proj)
    _maybe_sync_for_profile(x_qjl_hat_rot)
    _record_tq_profile(
        "turboquant_decode_prod.qjl_correction_rotate",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    t0 = time.perf_counter()
    x_hat = apply_rotation(x_mse_hat_rot + x_qjl_hat_rot, rotation_t)
    _maybe_sync_for_profile(x_hat)
    _record_tq_profile(
        "turboquant_decode_prod.inverse_rotate",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    t0 = time.perf_counter()
    decoded = x_hat * norm.to(target_dtype)
    _maybe_sync_for_profile(decoded)
    _record_tq_profile(
        "turboquant_decode_prod.rescale_norm",
        (time.perf_counter() - t0) * 1000.0,
        vectors=num_vectors,
    )

    _record_tq_profile(
        "turboquant_decode_prod.total",
        (time.perf_counter() - t_total_0) * 1000.0,
        vectors=num_vectors,
    )
    return decoded


def monte_carlo_bias_eval(
    head_dim: int,
    total_bits: int,
    *,
    num_samples: int = 2048,
    seed: int = 1234,
) -> dict[str, float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn(num_samples, head_dim, generator=generator, dtype=torch.float32, device="cpu")
    y = torch.randn(num_samples, head_dim, generator=generator, dtype=torch.float32, device="cpu")
    codebook, boundary = build_turboquant_codebook(head_dim, get_stage1_bits(total_bits, "prod"), "cpu", torch.float32)
    rotation = build_rotation_matrix(head_dim, seed + 1, "cpu", torch.float32)
    qjl_proj = build_qjl_projection(head_dim, seed + 2, "cpu", torch.float32)
    encoded = turboquant_encode_prod(x, rotation, codebook, boundary, qjl_proj, total_bits)
    decoded = turboquant_decode_prod(
        encoded["idx"],
        encoded["qjl"],
        encoded["gamma"],
        encoded["norm"],
        rotation.transpose(0, 1).contiguous(),
        codebook,
        qjl_proj,
        total_bits,
        head_dim,
        torch.float32,
    )
    original_inner = torch.sum(x * y, dim=-1)
    decoded_inner = torch.sum(decoded * y, dim=-1)
    diff = decoded_inner - original_inner
    return {
        "bias": float(diff.mean().item()),
        "std": float(diff.std(unbiased=False).item()),
    }
