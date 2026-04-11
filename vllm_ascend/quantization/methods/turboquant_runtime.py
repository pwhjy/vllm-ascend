#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

from __future__ import annotations

import math

import torch


def build_turboquant_codebook(head_dim: int, bits: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    del head_dim
    levels = 2**bits
    codebook = torch.linspace(-1.0, 1.0, levels, device=device, dtype=dtype)
    boundary = (codebook[:-1] + codebook[1:]) / 2.0
    return codebook, boundary


def build_rotation_matrix(head_dim: int, seed: int, device, dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_matrix = torch.randn(head_dim, head_dim, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(random_matrix, mode="reduced")
    return q.to(device=device, dtype=dtype).contiguous()


def build_qjl_projection(head_dim: int, seed: int, device, dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random_matrix = torch.randn(head_dim, head_dim, generator=generator, dtype=torch.float32)
    random_matrix = random_matrix / torch.clamp(random_matrix.norm(dim=1, keepdim=True), min=1e-6)
    return random_matrix.to(device=device, dtype=dtype).contiguous()


def pack_bits(indices: torch.Tensor, bits: int) -> torch.Tensor:
    if indices.dtype not in (torch.uint8, torch.int16, torch.int32, torch.int64):
        indices = indices.to(torch.int64)
    values = indices.to(torch.int64)
    if values.numel() == 0:
        return values.to(torch.uint8)
    original_shape = values.shape[:-1]
    flat = values.reshape(-1, values.shape[-1])
    packed_cols = (flat.shape[-1] * bits + 7) // 8
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
    return packed.reshape(*original_shape, packed_cols)


def unpack_bits(packed: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    values = packed.to(torch.int64)
    if values.numel() == 0:
        return values.to(torch.uint8)
    original_shape = values.shape[:-1]
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
    return unpacked.reshape(*original_shape, dim).to(torch.uint8)


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
    norm = torch.clamp(x.norm(dim=-1, keepdim=True), min=1e-6)
    x_unit = x / norm
    x_rot = apply_rotation(x_unit, rotation)
    idx = _encode_scalar_with_boundary(x_rot, boundary)
    x_hat_rot = _dequant_scalar(idx, codebook).to(x.dtype)
    packed_idx = pack_bits(idx, bits)
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
    bits: int,
) -> dict[str, torch.Tensor]:
    mse = turboquant_encode_mse(x, rotation, codebook, boundary, bits)
    residual = mse["x_rot"] - mse["x_hat_rot"]
    gamma = residual.norm(dim=-1, keepdim=True).to(torch.float32)
    qjl_sign = (apply_rotation(residual, qjl_proj.transpose(0, 1)) >= 0).to(torch.uint8)
    packed_qjl = pack_bits(qjl_sign, 1)
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
    idx = unpack_bits(packed_idx, bits, dim)
    x_hat_rot = _dequant_scalar(idx, codebook).to(target_dtype)
    x_hat = apply_rotation(x_hat_rot, rotation_t)
    return x_hat * norm.to(target_dtype)


def turboquant_decode_prod(
    packed_idx: torch.Tensor,
    packed_qjl: torch.Tensor,
    gamma: torch.Tensor,
    norm: torch.Tensor,
    rotation_t: torch.Tensor,
    codebook: torch.Tensor,
    qjl_proj: torch.Tensor,
    bits: int,
    dim: int,
    target_dtype: torch.dtype,
) -> torch.Tensor:
    idx = unpack_bits(packed_idx, bits, dim)
    qjl = unpack_bits(packed_qjl, 1, dim).to(target_dtype)
    qjl = qjl * 2.0 - 1.0
    x_mse_hat_rot = _dequant_scalar(idx, codebook).to(target_dtype)
    correction = math.sqrt(math.pi / 2.0) / dim
    x_qjl_hat_rot = correction * gamma.to(target_dtype) * apply_rotation(qjl, qjl_proj)
    x_hat = apply_rotation(x_mse_hat_rot + x_qjl_hat_rot, rotation_t)
    return x_hat * norm.to(target_dtype)
