#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

"""TurboQuant fused KV-update + attention dispatch.

This module is the Python-level contract for the final serving op:

    dense current Q/K/V
      + compressed historical paged KV cache
      -> attention output
      + in-place compressed cache update for the current K/V chunk

The reference implementation intentionally keeps the final semantics even
though it still materializes dense history.  The custom-op dispatch point uses
the same arguments so the Ascend C implementation can replace the reference
without changing the attention backend again.
"""

from __future__ import annotations

import os
import time

import torch

from vllm_ascend.quantization.methods.turboquant_layout import get_stage1_bits
from vllm_ascend.quantization.methods.turboquant_runtime import (
    _maybe_sync_for_profile,
    _record_tq_profile,
    apply_rotation,
)

from .dequant import (
    cached_token_map_from_block_table,
    custom_strict_enabled,
    tq_dequant_mse_paged_rot,
    tq_dequant_mse_paged_reference_rot,
    tq_dequant_prod_paged_k_score,
    tq_dequant_prod_paged_k_score_reference,
    tq_dequant_prod_paged_reference_rot,
)


def fused_kv_update_attention_enabled() -> bool:
    """Enable the final-architecture Python path in attention_v1."""

    return os.getenv("VLLM_ASCEND_TQ_USE_FUSED_KV_UPDATE_ATTENTION", "0") == "1"


def fused_kv_update_attention_custom_enabled() -> bool:
    """Try the future Ascend C unified op before the reference path."""

    return os.getenv("VLLM_ASCEND_TQ_USE_CUSTOM_FUSED_KV_UPDATE_ATTENTION", "0") == "1"


def encode_cache_update_custom_enabled() -> bool:
    """Try the Phase-A Ascend C cache-update op before the reference path."""

    return os.getenv("VLLM_ASCEND_TQ_USE_CUSTOM_ENCODE_CACHE_UPDATE", "0") == "1"


def encode_cache_update_stage_profile_enabled() -> bool:
    """Record detailed Phase-A Python fallback timings.

    This adds extra sync points and is intended for diagnosis, not normal
    throughput measurement.
    """

    return os.getenv("VLLM_ASCEND_TQ_PROFILE_ENCODE_STAGES", "0") == "1"


def combined_kv_mse_encode_enabled() -> bool:
    """Combine K stage1 MSE and V MSE launches in the NPU fallback path."""

    return os.getenv("VLLM_ASCEND_TQ_COMBINE_KV_MSE_ENCODE", "1") == "1"


def compressed_decode_current_enabled() -> bool:
    """Use compressed-history scores plus dense-current V for decode."""

    return os.getenv("VLLM_ASCEND_TQ_USE_COMPRESSED_DECODE_CURRENT", "0") == "1"


def fused_decode_attention_m4_enabled() -> bool:
    """Use the M4 decode-only custom attention kernel after staged cache update."""

    return os.getenv("VLLM_ASCEND_TQ_USE_FUSED_DECODE_ATTENTION_M4", "0") == "1"


def fused_decode_attention_m4_shadow_profile_enabled() -> bool:
    """Run the decomposed reference path only to profile M4 decode stages."""

    return os.getenv("VLLM_ASCEND_TQ_PROFILE_M4_SHADOW", "0") == "1"


def fused_decode_attention_m4_stage_profile_enabled() -> bool:
    """Record synchronized sub-stage timings around the M4 custom op."""

    return os.getenv("VLLM_ASCEND_TQ_PROFILE_M4_STAGES", "0") == "1"


def fused_decode_attention_m4_score_tile_len() -> int:
    """History-score tile length for the M4 decode custom op.

    Set ``VLLM_ASCEND_TQ_M4_SCORE_TILE_LEN=0`` to use the scalar online path.
    """

    try:
        tile_len = int(os.getenv("VLLM_ASCEND_TQ_M4_SCORE_TILE_LEN", "0"))
    except ValueError:
        tile_len = 0
    if tile_len <= 0:
        return 0
    return min(64, max(1, tile_len))


def fused_decode_attention_m4_grouped_q_enabled() -> bool:
    """Process all Q heads that share a KV head in one M4 decode task."""

    return os.getenv("VLLM_ASCEND_TQ_M4_GROUPED_Q", "1") == "1"


def compressed_decode_current_custom_k_score_enabled() -> bool:
    """Use custom compressed K-score inside the decode-current path."""

    return (
        os.getenv("VLLM_ASCEND_TQ_USE_COMPRESSED_DECODE_CUSTOM_K_SCORE", "0")
        == "1"
    )


def decode_compressed_full_cache_enabled() -> bool:
    """Use compressed full-cache attention after Phase-A cache update.

    This is a performance A/B path. It is safe as separate kernel launches, but
    differs from the final architecture because the current token is read back
    from compressed cache instead of participating as dense K/V.
    """

    return os.getenv("VLLM_ASCEND_TQ_USE_DECODE_COMPRESSED_FULL_CACHE", "0") == "1"


def _is_npu_tensor(tensor: torch.Tensor) -> bool:
    return bool(getattr(tensor, "is_npu", False)) or tensor.device.type in {
        "npu",
        "privateuseone",
    }


def _custom_op_available(op_name: str, *tensors: torch.Tensor) -> bool:
    if tensors and not all(_is_npu_tensor(tensor) for tensor in tensors):
        return False
    try:
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            return False
    except Exception:
        return False
    return hasattr(torch.ops, "_C_ascend") and hasattr(torch.ops._C_ascend, op_name)


def _as_int32_tensor(x: torch.Tensor | list[int], device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.int32).contiguous()
    return torch.tensor([int(v) for v in x], dtype=torch.int32, device=device)


def _to_int_list(x: torch.Tensor | list[int]) -> list[int]:
    if isinstance(x, list):
        return [int(v) for v in x]
    return [int(v) for v in x.detach().cpu().tolist()]


def current_lengths_from_start_loc(start_loc: torch.Tensor | list[int]) -> list[int]:
    loc = _to_int_list(start_loc)
    if len(loc) < 2:
        return []
    return [max(0, int(loc[i + 1]) - int(loc[i])) for i in range(len(loc) - 1)]


def old_seq_lens_from_total(
    total_seq_lens: torch.Tensor | list[int],
    key_start_loc: torch.Tensor | list[int],
) -> list[int]:
    totals = _to_int_list(total_seq_lens)
    cur_lens = current_lengths_from_start_loc(key_start_loc)
    batch = min(len(totals), len(cur_lens))
    return [max(0, totals[i] - cur_lens[i]) for i in range(batch)]


def _validate_cache(kv_cache: dict[str, torch.Tensor]) -> None:
    required = ("k_idx", "k_norm", "v_idx", "v_norm")
    missing = [name for name in required if name not in kv_cache]
    if missing:
        raise KeyError(f"TurboQuant KV cache is missing sidecar tensors: {missing}")


def _empty_history_tensors(
    *,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    empty = torch.empty(
        (0, int(num_kv_heads), int(head_dim)),
        dtype=dtype,
        device=device,
    )
    return empty, empty


def _pack_bits_no_profile(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack low-bit indices without nested TurboQuant profile sync points."""

    if indices.numel() == 0:
        return indices.to(torch.uint8)

    original_shape = indices.shape[:-1]
    values = indices.to(torch.uint8).reshape(-1, indices.shape[-1])
    packed_cols = (values.shape[-1] * bits + 7) // 8

    def _pad_last_dim(x: torch.Tensor, pad: int) -> torch.Tensor:
        if not pad:
            return x
        padded = torch.zeros(
            (x.shape[0], x.shape[-1] + pad),
            dtype=torch.uint8,
            device=x.device,
        )
        padded[:, : x.shape[-1]] = x
        return padded

    if bits == 1:
        values = _pad_last_dim(values & 0x1, (-values.shape[-1]) % 8)
        chunks = values.view(values.shape[0], -1, 8)
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
    elif bits == 2:
        values = _pad_last_dim(values & 0x3, (-values.shape[-1]) % 4)
        chunks = values.view(values.shape[0], -1, 4)
        packed = (
            chunks[..., 0]
            | (chunks[..., 1] << 2)
            | (chunks[..., 2] << 4)
            | (chunks[..., 3] << 6)
        )
    elif bits == 3:
        values = _pad_last_dim(values & 0x7, (-values.shape[-1]) % 8)
        chunks = values.view(values.shape[0], -1, 8)
        byte0 = chunks[..., 0] | (chunks[..., 1] << 3) | (chunks[..., 2] << 6)
        byte1 = (
            (chunks[..., 2] >> 2)
            | (chunks[..., 3] << 1)
            | (chunks[..., 4] << 4)
            | (chunks[..., 5] << 7)
        )
        byte2 = (chunks[..., 5] >> 1) | (chunks[..., 6] << 2) | (chunks[..., 7] << 5)
        packed = torch.stack([byte0, byte1, byte2], dim=-1).view(values.shape[0], -1)
    elif bits == 4:
        values = _pad_last_dim(values & 0xF, (-values.shape[-1]) % 2)
        chunks = values.view(values.shape[0], -1, 2)
        packed = chunks[..., 0] | (chunks[..., 1] << 4)
    else:
        values = indices.to(torch.int64)
        flat = values.reshape(-1, values.shape[-1])
        packed = torch.zeros(
            (flat.shape[0], packed_cols),
            dtype=torch.uint8,
            device=flat.device,
        )
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

    return packed[:, :packed_cols].reshape(*original_shape, packed_cols).contiguous()


def _encode_scalar_with_boundary_no_profile(
    x: torch.Tensor,
    boundary: torch.Tensor,
) -> torch.Tensor:
    if boundary.numel() == 3:
        return (
            (x > boundary[0]).to(torch.uint8)
            + (x > boundary[1]).to(torch.uint8)
            + (x > boundary[2]).to(torch.uint8)
        )
    return torch.sum(x.unsqueeze(-1) > boundary, dim=-1).to(torch.uint8)


def _encode_mse_cache_no_profile(
    x: torch.Tensor,
    rotation: torch.Tensor,
    codebook: torch.Tensor,
    boundary: torch.Tensor,
    bits: int,
    *,
    return_rot_and_dequant: bool,
) -> dict[str, torch.Tensor]:
    norm = torch.clamp(x.norm(dim=-1, keepdim=True), min=1e-6)
    x_rot = apply_rotation(x / norm, rotation)
    idx = _encode_scalar_with_boundary_no_profile(x_rot, boundary)
    out = {
        "idx": _pack_bits_no_profile(idx, bits),
        "norm": norm.to(torch.float32),
    }
    if return_rot_and_dequant:
        out["x_rot"] = x_rot
        out["x_hat_rot"] = codebook[idx.long()].to(x.dtype)
    return out


def _encode_prod_cache_no_profile(
    x: torch.Tensor,
    rotation: torch.Tensor,
    codebook: torch.Tensor,
    boundary: torch.Tensor,
    qjl_proj: torch.Tensor,
    total_bits: int,
    *,
    qjl_proj_t: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    stage1_bits = get_stage1_bits(total_bits, "prod")
    mse = _encode_mse_cache_no_profile(
        x,
        rotation,
        codebook,
        boundary,
        stage1_bits,
        return_rot_and_dequant=True,
    )
    residual = mse["x_rot"] - mse["x_hat_rot"]
    gamma = residual.norm(dim=-1, keepdim=True).to(torch.float32)
    qjl_rotation = qjl_proj_t if qjl_proj_t is not None else qjl_proj.transpose(0, 1)
    qjl_sign = (apply_rotation(residual, qjl_rotation) >= 0).to(torch.uint8)
    return {
        "idx": mse["idx"],
        "qjl": _pack_bits_no_profile(qjl_sign, 1),
        "gamma": gamma,
        "norm": mse["norm"],
    }


def _can_combine_kv_mse_encode(
    key: torch.Tensor,
    value: torch.Tensor,
    k_boundary: torch.Tensor,
    k_codebook: torch.Tensor,
    v_boundary: torch.Tensor,
    v_codebook: torch.Tensor,
    *,
    k_variant: str,
    k_stage1_bits: int,
    v_bits: int,
    kv_mse_rotation: torch.Tensor | None,
    kv_mse_shared_boundary: bool,
) -> bool:
    return (
        combined_kv_mse_encode_enabled()
        and _is_npu_tensor(key)
        and k_variant == "prod"
        and int(k_stage1_bits) == int(v_bits)
        and kv_mse_rotation is not None
        and bool(kv_mse_shared_boundary)
        and key.shape == value.shape
        and k_boundary.shape == v_boundary.shape
        and k_codebook.shape == v_codebook.shape
    )


def _encode_prod_k_mse_v_cache_no_profile(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_mse_rotation: torch.Tensor,
    codebook: torch.Tensor,
    boundary: torch.Tensor,
    qjl_proj: torch.Tensor,
    total_bits: int,
    *,
    qjl_proj_t: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    stage1_bits = get_stage1_bits(total_bits, "prod")
    kv = torch.stack((key, value), dim=0)
    norm = torch.clamp(kv.norm(dim=-1, keepdim=True), min=1e-6)
    flat = (kv / norm).reshape(2, -1, key.shape[-1])
    kv_rot = torch.bmm(flat, kv_mse_rotation).reshape(2, *key.shape)
    idx = _encode_scalar_with_boundary_no_profile(kv_rot, boundary)
    packed = _pack_bits_no_profile(idx, stage1_bits)

    key_rot = kv_rot[0]
    key_idx = idx[0]
    residual = key_rot - codebook[key_idx.long()].to(key.dtype)
    gamma = residual.norm(dim=-1, keepdim=True).to(torch.float32)
    qjl_rotation = qjl_proj_t if qjl_proj_t is not None else qjl_proj.transpose(0, 1)
    qjl_sign = (apply_rotation(residual, qjl_rotation) >= 0).to(torch.uint8)

    encoded_k = {
        "idx": packed[0].contiguous(),
        "qjl": _pack_bits_no_profile(qjl_sign, 1),
        "gamma": gamma,
        "norm": norm[0].to(torch.float32),
    }
    encoded_v = {
        "idx": packed[1].contiguous(),
        "norm": norm[1].to(torch.float32),
    }
    return encoded_k, encoded_v


def tq_encode_kv_to_paged_cache_reference(
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: dict[str, torch.Tensor],
    k_rotation: torch.Tensor,
    k_codebook: torch.Tensor,
    k_boundary: torch.Tensor,
    k_qjl_proj: torch.Tensor,
    v_rotation: torch.Tensor,
    v_codebook: torch.Tensor,
    v_boundary: torch.Tensor,
    *,
    k_variant: str,
    k_total_bits: int,
    k_stage1_bits: int,
    v_bits: int,
    num_kv_heads: int | None = None,
    assume_valid_slots: bool = False,
    k_qjl_proj_t: torch.Tensor | None = None,
    kv_mse_rotation: torch.Tensor | None = None,
    kv_mse_shared_boundary: bool = False,
) -> None:
    """Reference fused encode + bit-pack + paged sidecar cache write.

    This is the P6 stepping stone from the design doc.  It performs the same
    side effects as the final op's Phase A and deliberately writes directly to
    the sidecar layout using ``slot_mapping``.
    """

    stage_profile = encode_cache_update_stage_profile_enabled()

    def _stage_start(*objs) -> float:
        if stage_profile:
            _maybe_sync_for_profile(*objs)
            return time.perf_counter()
        return 0.0

    def _stage_record(
        name: str,
        start: float,
        *objs,
        vectors: int = 0,
    ) -> None:
        if not stage_profile:
            return
        _maybe_sync_for_profile(*objs)
        _record_tq_profile(
            f"turboquant_encode_cache_update.{name}",
            (time.perf_counter() - start) * 1000.0,
            vectors=vectors,
        )

    _validate_cache(kv_cache)
    _maybe_sync_for_profile(key, value, slot_mapping)
    t0 = time.perf_counter()

    t_stage = _stage_start(key, value, slot_mapping)
    slots = slot_mapping.to(device=key.device, dtype=torch.long).reshape(-1)
    if slots.numel() == 0:
        _record_tq_profile("turboquant_encode_cache_update.total", 0.0, vectors=0)
        return

    num_tokens = int(slots.numel())
    num_kv_heads = int(num_kv_heads if num_kv_heads is not None else key.shape[1])
    if assume_valid_slots:
        valid_slots = slots
        actual_key = key[:num_tokens].to(torch.float32)
        actual_value = value[:num_tokens].to(torch.float32)
    else:
        valid = slots >= 0
        valid_slots = slots[valid]
        if valid_slots.numel() == 0:
            _record_tq_profile("turboquant_encode_cache_update.total", 0.0, vectors=0)
            return
        actual_key = key[:num_tokens][valid].to(torch.float32)
        actual_value = value[:num_tokens][valid].to(torch.float32)

    vector_count = int(valid_slots.numel()) * num_kv_heads
    _stage_record(
        "prepare",
        t_stage,
        actual_key,
        actual_value,
        valid_slots,
        vectors=vector_count,
    )

    use_combined_kv_mse = _can_combine_kv_mse_encode(
        actual_key,
        actual_value,
        k_boundary,
        k_codebook,
        v_boundary,
        v_codebook,
        k_variant=k_variant,
        k_stage1_bits=int(k_stage1_bits),
        v_bits=int(v_bits),
        kv_mse_rotation=kv_mse_rotation,
        kv_mse_shared_boundary=kv_mse_shared_boundary,
    )
    if use_combined_kv_mse:
        assert kv_mse_rotation is not None
        t_stage = _stage_start(actual_key, actual_value)
        encoded_k, encoded_v = _encode_prod_k_mse_v_cache_no_profile(
            actual_key,
            actual_value,
            kv_mse_rotation,
            k_codebook,
            k_boundary,
            k_qjl_proj,
            int(k_total_bits),
            qjl_proj_t=k_qjl_proj_t,
        )
        _stage_record(
            "encode.combined_kv_mse",
            t_stage,
            encoded_k,
            encoded_v,
            vectors=vector_count,
        )
    else:
        t_stage = _stage_start(actual_key)
        if k_variant == "prod":
            encoded_k = _encode_prod_cache_no_profile(
                actual_key,
                k_rotation,
                k_codebook,
                k_boundary,
                k_qjl_proj,
                int(k_total_bits),
                qjl_proj_t=k_qjl_proj_t,
            )
        elif k_variant == "mse":
            encoded_k = _encode_mse_cache_no_profile(
                actual_key,
                k_rotation,
                k_codebook,
                k_boundary,
                int(k_stage1_bits),
                return_rot_and_dequant=False,
            )
        else:
            raise ValueError(f"Unsupported TurboQuant K variant: {k_variant}")
        _stage_record("encode.k", t_stage, encoded_k, vectors=vector_count)

        t_stage = _stage_start(actual_value)
        encoded_v = _encode_mse_cache_no_profile(
            actual_value,
            v_rotation,
            v_codebook,
            v_boundary,
            int(v_bits),
            return_rot_and_dequant=False,
        )
        _stage_record("encode.v", t_stage, encoded_v, vectors=vector_count)

    def _write(cache_name: str, encoded: torch.Tensor) -> None:
        cache = kv_cache[cache_name]
        flat_cache = cache.view(-1, num_kv_heads, cache.shape[-1])
        t_write = _stage_start(encoded, flat_cache)
        update = (
            encoded if encoded.dtype == flat_cache.dtype
            else encoded.to(flat_cache.dtype)
        )
        flat_cache.index_copy_(0, valid_slots, update)
        _stage_record(
            f"write.{cache_name}",
            t_write,
            flat_cache,
            vectors=vector_count,
        )

    t_stage = _stage_start(kv_cache)
    _write("k_idx", encoded_k["idx"])
    _write("k_norm", encoded_k["norm"])
    _write("v_idx", encoded_v["idx"])
    _write("v_norm", encoded_v["norm"])
    if "qjl" in encoded_k and "k_qjl" in kv_cache:
        _write("k_qjl", encoded_k["qjl"])
    if "gamma" in encoded_k and "k_gamma" in kv_cache:
        _write("k_gamma", encoded_k["gamma"])
    _stage_record("write.total", t_stage, kv_cache, vectors=vector_count)

    _maybe_sync_for_profile(kv_cache)
    _record_tq_profile(
        "turboquant_encode_cache_update.total",
        (time.perf_counter() - t0) * 1000.0,
        vectors=vector_count,
    )


def tq_encode_kv_to_paged_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: dict[str, torch.Tensor],
    k_rotation: torch.Tensor,
    k_codebook: torch.Tensor,
    k_boundary: torch.Tensor,
    k_qjl_proj: torch.Tensor,
    v_rotation: torch.Tensor,
    v_codebook: torch.Tensor,
    v_boundary: torch.Tensor,
    *,
    k_variant: str,
    k_total_bits: int,
    k_stage1_bits: int,
    v_bits: int,
    num_kv_heads: int | None = None,
    assume_valid_slots: bool = False,
    k_qjl_proj_t: torch.Tensor | None = None,
    kv_mse_rotation: torch.Tensor | None = None,
    kv_mse_shared_boundary: bool = False,
) -> None:
    """Dispatch Phase A: encode current K/V and update sidecar cache.

    The custom op is the P6 stepping stone from the final-architecture design.
    It is intentionally gated separately from the full fused attention op so
    we can bring up cache update independently, then fold it into the unified
    serving op once the Ascend C kernel is ready.
    """

    custom_enabled = encode_cache_update_custom_enabled()
    if custom_enabled:
        _maybe_sync_for_profile(key, value, slot_mapping)
        custom_t0 = time.perf_counter()
        stage_profile = encode_cache_update_stage_profile_enabled()

        def _custom_stage_record(
            name: str,
            start: float,
            *objs,
        ) -> float:
            if not stage_profile:
                return time.perf_counter()
            _maybe_sync_for_profile(*objs)
            _record_tq_profile(
                f"turboquant_encode_cache_update.{name}",
                (time.perf_counter() - start) * 1000.0,
                vectors=int(slot_mapping.numel()) * int(key.shape[1]),
            )
            return time.perf_counter()

        fallback_reasons: list[str] = []
        if k_variant != "prod":
            fallback_reasons.append(f"unsupported K variant: {k_variant}")
        if not assume_valid_slots:
            fallback_reasons.append("custom cache update requires valid slots")
        tensors = (
            key,
            value,
            slot_mapping,
            kv_cache["k_idx"],
            kv_cache.get("k_qjl", torch.empty(0, device=key.device, dtype=torch.uint8)),
            kv_cache.get("k_gamma", torch.empty(0, device=key.device, dtype=torch.float32)),
            kv_cache["k_norm"],
            kv_cache["v_idx"],
            kv_cache["v_norm"],
            k_rotation,
            k_boundary,
            k_codebook,
            k_qjl_proj_t if k_qjl_proj_t is not None else k_qjl_proj,
            v_rotation,
            v_boundary,
        )
        if not _custom_op_available("tq_encode_kv_to_paged_cache", *tensors):
            fallback_reasons.append("torch.ops._C_ascend.tq_encode_kv_to_paged_cache is unavailable")

        if not fallback_reasons:
            try:
                stage_t0 = time.perf_counter()
                key_f = key.to(torch.float32).contiguous()
                value_f = value.to(torch.float32).contiguous()
                slot_mapping_i64 = slot_mapping.to(torch.long).contiguous()
                k_qjl_cache = kv_cache.get(
                    "k_qjl",
                    torch.empty(0, device=key.device, dtype=torch.uint8),
                )
                k_gamma_cache = kv_cache.get(
                    "k_gamma",
                    torch.empty(0, device=key.device, dtype=torch.float32),
                )
                k_rotation_f = k_rotation.contiguous()
                k_boundary_c = k_boundary.contiguous()
                k_codebook_c = k_codebook.contiguous()
                k_qjl_proj_t_c = (
                    k_qjl_proj_t
                    if k_qjl_proj_t is not None
                    else k_qjl_proj.transpose(0, 1)
                ).contiguous()
                v_rotation_f = v_rotation.contiguous()
                v_boundary_c = v_boundary.contiguous()
                stage_t0 = _custom_stage_record(
                    "custom.prepare_inputs",
                    stage_t0,
                    key_f,
                    value_f,
                    slot_mapping_i64,
                    k_rotation_f,
                    k_boundary_c,
                    k_codebook_c,
                    k_qjl_proj_t_c,
                    v_rotation_f,
                    v_boundary_c,
                )
                torch.ops._C_ascend.tq_encode_kv_to_paged_cache(
                    key_f,
                    value_f,
                    slot_mapping_i64,
                    kv_cache["k_idx"],
                    k_qjl_cache,
                    k_gamma_cache,
                    kv_cache["k_norm"],
                    kv_cache["v_idx"],
                    kv_cache["v_norm"],
                    k_rotation_f,
                    k_boundary_c,
                    k_codebook_c,
                    k_qjl_proj_t_c,
                    v_rotation_f,
                    v_boundary_c,
                    int(k_total_bits),
                    int(k_stage1_bits),
                    int(v_bits),
                    int(key.shape[-1]),
                )
                stage_t0 = _custom_stage_record(
                    "custom.op",
                    stage_t0,
                    kv_cache,
                )
                if not stage_profile:
                    _maybe_sync_for_profile(kv_cache)
                _record_tq_profile(
                    "turboquant_encode_cache_update.total",
                    (time.perf_counter() - custom_t0) * 1000.0,
                    vectors=int(slot_mapping.numel()) * int(key.shape[1]),
                )
                return
            except Exception:
                if custom_strict_enabled():
                    raise
                fallback_reasons.append("custom cache update call failed")

        staged_reasons: list[str] = []
        if k_variant != "prod":
            staged_reasons.append(f"unsupported K variant: {k_variant}")
        if not assume_valid_slots:
            staged_reasons.append("staged cache update requires valid slots")
        if "k_qjl" not in kv_cache or "k_gamma" not in kv_cache:
            staged_reasons.append("prod staged cache update requires k_qjl/k_gamma")
        if (
            kv_cache.get("k_norm", torch.empty(0)).dtype != torch.float32
            or kv_cache.get("k_gamma", torch.empty(0)).dtype != torch.float32
            or kv_cache.get("v_norm", torch.empty(0)).dtype != torch.float32
        ):
            staged_reasons.append("staged cache update requires fp32 scalar caches")
        if int(key.shape[-1]) > 256 or int(value.shape[-1]) > 256:
            staged_reasons.append("staged cache update supports head_dim <= 256")

        staged_tensors = (
            key,
            value,
            slot_mapping,
            kv_cache["k_idx"],
            kv_cache.get("k_qjl", torch.empty(0, device=key.device, dtype=torch.uint8)),
            kv_cache.get("k_gamma", torch.empty(0, device=key.device, dtype=torch.float32)),
            kv_cache["k_norm"],
            kv_cache["v_idx"],
            kv_cache["v_norm"],
            k_rotation,
            k_boundary,
            k_codebook,
            k_qjl_proj_t if k_qjl_proj_t is not None else k_qjl_proj,
            v_rotation,
            v_boundary,
        )
        if not _custom_op_available("tq_encode_prod_paged_cache", *staged_tensors):
            staged_reasons.append("torch.ops._C_ascend.tq_encode_prod_paged_cache is unavailable")
        if not _custom_op_available("tq_encode_mse_paged_cache", *staged_tensors):
            staged_reasons.append("torch.ops._C_ascend.tq_encode_mse_paged_cache is unavailable")

        if not staged_reasons:
            try:
                qjl_proj_t = (
                    k_qjl_proj_t if k_qjl_proj_t is not None
                    else k_qjl_proj.transpose(0, 1)
                )
                slot_mapping_i64 = slot_mapping.to(torch.long).contiguous()
                torch.ops._C_ascend.tq_encode_prod_paged_cache(
                    key.to(torch.float32).contiguous(),
                    slot_mapping_i64,
                    kv_cache["k_idx"],
                    kv_cache["k_qjl"],
                    kv_cache["k_gamma"],
                    kv_cache["k_norm"],
                    k_rotation.contiguous(),
                    k_boundary.contiguous(),
                    k_codebook.contiguous(),
                    qjl_proj_t.contiguous(),
                    int(k_total_bits),
                    int(k_stage1_bits),
                    int(key.shape[-1]),
                )
                torch.ops._C_ascend.tq_encode_mse_paged_cache(
                    value.to(torch.float32).contiguous(),
                    slot_mapping_i64,
                    kv_cache["v_idx"],
                    kv_cache["v_norm"],
                    v_rotation.contiguous(),
                    v_boundary.contiguous(),
                    int(v_bits),
                    int(value.shape[-1]),
                )
                _maybe_sync_for_profile(kv_cache)
                _record_tq_profile(
                    "turboquant_encode_cache_update.total",
                    (time.perf_counter() - custom_t0) * 1000.0,
                    vectors=int(slot_mapping.numel()) * int(key.shape[1]),
                )
                return
            except Exception:
                if custom_strict_enabled():
                    raise
                staged_reasons.append("staged custom cache update call failed")

        fallback_reasons.extend(staged_reasons)

        if custom_strict_enabled():
            raise RuntimeError(
                "TurboQuant encode cache-update custom op fallback: "
                + "; ".join(fallback_reasons)
            )

    return tq_encode_kv_to_paged_cache_reference(
        key,
        value,
        slot_mapping,
        kv_cache,
        k_rotation,
        k_codebook,
        k_boundary,
        k_qjl_proj,
        v_rotation,
        v_codebook,
        v_boundary,
        k_variant=k_variant,
        k_total_bits=k_total_bits,
        k_stage1_bits=k_stage1_bits,
        v_bits=v_bits,
        num_kv_heads=num_kv_heads,
        assume_valid_slots=assume_valid_slots,
        k_qjl_proj_t=k_qjl_proj_t,
        kv_mse_rotation=kv_mse_rotation,
        kv_mse_shared_boundary=kv_mse_shared_boundary,
    )


def _decode_history_to_dense(
    kv_cache: dict[str, torch.Tensor],
    block_table: torch.Tensor,
    old_seq_lens: list[int],
    k_codebook: torch.Tensor,
    v_codebook: torch.Tensor,
    k_rotation_t: torch.Tensor,
    k_qjl_proj: torch.Tensor,
    v_rotation_t: torch.Tensor,
    *,
    k_variant: str,
    k_total_bits: int,
    k_stage1_bits: int,
    v_bits: int,
    head_dim: int,
    target_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not any(int(seq_len) > 0 for seq_len in old_seq_lens):
        return _empty_history_tensors(
            num_kv_heads=int(kv_cache["k_idx"].shape[2]),
            head_dim=int(head_dim),
            dtype=target_dtype,
            device=block_table.device,
        )

    block_size = int(kv_cache["k_idx"].shape[1])
    token_block_ids, token_offsets = cached_token_map_from_block_table(
        block_table,
        old_seq_lens,
        block_size,
    )
    if token_block_ids.numel() == 0:
        return _empty_history_tensors(
            num_kv_heads=int(kv_cache["k_idx"].shape[2]),
            head_dim=int(head_dim),
            dtype=target_dtype,
            device=block_table.device,
        )

    if k_variant == "prod" and "k_qjl" in kv_cache and "k_gamma" in kv_cache:
        k_rot = tq_dequant_prod_paged_reference_rot(
            kv_cache["k_idx"],
            kv_cache["k_qjl"],
            kv_cache["k_gamma"],
            kv_cache["k_norm"],
            token_block_ids,
            token_offsets,
            k_codebook,
            k_qjl_proj,
            int(k_total_bits),
            int(head_dim),
            target_dtype,
        )
    else:
        k_rot = tq_dequant_mse_paged_reference_rot(
            kv_cache["k_idx"],
            kv_cache["k_norm"],
            token_block_ids,
            token_offsets,
            k_codebook,
            int(k_stage1_bits),
            int(head_dim),
            target_dtype,
        )

    v_rot = tq_dequant_mse_paged_rot(
        kv_cache["v_idx"],
        kv_cache["v_norm"],
        token_block_ids,
        token_offsets,
        v_codebook,
        int(v_bits),
        int(head_dim),
        target_dtype,
    )
    dense_k = apply_rotation(k_rot, k_rotation_t.to(device=k_rot.device, dtype=target_dtype))
    dense_v = apply_rotation(v_rot, v_rotation_t.to(device=v_rot.device, dtype=target_dtype))
    return dense_k.contiguous(), dense_v.contiguous()


def tq_decode_history_to_dense(
    kv_cache: dict[str, torch.Tensor],
    block_table: torch.Tensor,
    old_seq_lens: torch.Tensor | list[int],
    k_codebook: torch.Tensor,
    v_codebook: torch.Tensor,
    k_rotation_t: torch.Tensor,
    k_qjl_proj: torch.Tensor,
    v_rotation_t: torch.Tensor,
    *,
    k_variant: str,
    k_total_bits: int,
    k_stage1_bits: int,
    v_bits: int,
    head_dim: int,
    target_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _decode_history_to_dense(
        kv_cache,
        block_table,
        _to_int_list(old_seq_lens),
        k_codebook,
        v_codebook,
        k_rotation_t,
        k_qjl_proj,
        v_rotation_t,
        k_variant=k_variant,
        k_total_bits=k_total_bits,
        k_stage1_bits=k_stage1_bits,
        v_bits=v_bits,
        head_dim=head_dim,
        target_dtype=target_dtype,
    )


def tq_prod_mse_history_current_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: dict[str, torch.Tensor],
    block_table: torch.Tensor,
    old_seq_lens: torch.Tensor | list[int],
    k_codebook: torch.Tensor,
    v_codebook: torch.Tensor,
    k_rotation: torch.Tensor,
    k_qjl_proj: torch.Tensor,
    v_rotation_t: torch.Tensor,
    *,
    k_total_bits: int,
    v_bits: int,
    head_dim: int,
    scale: float,
    score_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
    profile_prefix: str | None = None,
) -> torch.Tensor:
    """Decode-only final-architecture attention without dense historical K.

    Historical K stays compressed and is used only for K-score. Historical V is
    dequantized, then the dense current K/V is appended logically in the
    softmax domain. This is equivalent to dense attention over
    ``compressed history + dense current`` for one current token per request.
    """

    _validate_cache(kv_cache)
    if "k_qjl" not in kv_cache or "k_gamma" not in kv_cache:
        raise ValueError("prod K decode requires k_qjl and k_gamma sidecar cache")

    old_seq_lens_list = _to_int_list(old_seq_lens)
    batch = len(old_seq_lens_list)
    if batch == 0:
        return torch.empty(
            (0, int(query.shape[1]), int(head_dim)),
            dtype=output_dtype or score_dtype,
            device=query.device,
        )
    if (
        int(query.shape[0]) < batch
        or int(key.shape[0]) < batch
        or int(value.shape[0]) < batch
    ):
        raise ValueError("Decode compressed-current path expects one current token per request")

    num_heads = int(query.shape[1])
    num_kv_heads = int(kv_cache["k_idx"].shape[2])
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    q_per_kv = num_heads // num_kv_heads
    max_old_len = max(old_seq_lens_list, default=0)
    query_f = query[:batch].to(score_dtype)
    key_f = key[:batch].to(score_dtype)
    value_f = value[:batch].to(score_dtype)

    if max_old_len > 0:
        if profile_prefix is not None:
            _maybe_sync_for_profile(query_f, block_table)
            t_stage = time.perf_counter()
        k_score_fn = (
            tq_dequant_prod_paged_k_score
            if compressed_decode_current_custom_k_score_enabled()
            else tq_dequant_prod_paged_k_score_reference
        )
        history_scores = k_score_fn(
            query_f,
            kv_cache["k_idx"],
            kv_cache["k_qjl"],
            kv_cache["k_gamma"],
            kv_cache["k_norm"],
            block_table,
            old_seq_lens_list,
            k_codebook,
            k_rotation,
            k_qjl_proj,
            int(k_total_bits),
            int(head_dim),
            scale=float(scale),
            max_seq_len=max_old_len,
            score_dtype=score_dtype,
        )
        if profile_prefix is not None:
            _maybe_sync_for_profile(history_scores)
            _record_tq_profile(
                f"{profile_prefix}.compressed_k_score",
                (time.perf_counter() - t_stage) * 1000.0,
                vectors=sum(old_seq_lens_list) * num_kv_heads,
            )

        block_size = int(kv_cache["v_idx"].shape[1])
        token_block_ids, token_offsets = cached_token_map_from_block_table(
            block_table,
            old_seq_lens_list,
            block_size,
        )
        if profile_prefix is not None:
            _maybe_sync_for_profile(token_block_ids, token_offsets)
            t_stage = time.perf_counter()
        history_v_rot = tq_dequant_mse_paged_rot(
            kv_cache["v_idx"],
            kv_cache["v_norm"],
            token_block_ids,
            token_offsets,
            v_codebook,
            int(v_bits),
            int(head_dim),
            score_dtype,
        )
        history_v = apply_rotation(
            history_v_rot,
            v_rotation_t.to(device=query.device, dtype=score_dtype),
        ).contiguous()
        if profile_prefix is not None:
            _maybe_sync_for_profile(history_v)
            _record_tq_profile(
                f"{profile_prefix}.decode_v",
                (time.perf_counter() - t_stage) * 1000.0,
                vectors=sum(old_seq_lens_list) * num_kv_heads,
            )
    else:
        history_scores = torch.empty(
            (batch, num_heads, 0),
            dtype=score_dtype,
            device=query.device,
        )
        history_v = torch.empty(
            (0, num_kv_heads, int(head_dim)),
            dtype=score_dtype,
            device=query.device,
        )

    if profile_prefix is not None:
        _maybe_sync_for_profile(query_f, key_f, value_f, history_scores, history_v)
        t_stage = time.perf_counter()
    q_group = query_f.reshape(batch, num_kv_heads, q_per_kv, int(head_dim))
    current_scores = torch.einsum("bkqd,bkd->bkq", q_group, key_f)
    current_scores = current_scores.reshape(batch, num_heads, 1) * float(scale)
    scores = torch.cat((history_scores, current_scores), dim=-1)
    probs = torch.softmax(scores, dim=-1)

    out = torch.empty(
        (batch, num_heads, int(head_dim)),
        dtype=score_dtype,
        device=query.device,
    )
    hist_cursor = 0
    for batch_idx, old_len in enumerate(old_seq_lens_list):
        prob_group = probs[batch_idx].reshape(num_kv_heads, q_per_kv, -1)
        cur_out = (
            prob_group[..., -1].unsqueeze(-1)
            * value_f[batch_idx].unsqueeze(1)
        )
        if old_len > 0:
            batch_v = history_v[hist_cursor:hist_cursor + old_len]
            cur_out = cur_out + torch.einsum(
                "kqs,skd->kqd",
                prob_group[..., :old_len],
                batch_v,
            )
            hist_cursor += old_len
        out[batch_idx] = cur_out.reshape(num_heads, int(head_dim))

    if output_dtype is not None:
        out = out.to(output_dtype)
    if profile_prefix is not None:
        _maybe_sync_for_profile(out)
        _record_tq_profile(
            f"{profile_prefix}.combine_current",
            (time.perf_counter() - t_stage) * 1000.0,
            vectors=batch,
        )
    return out.contiguous()


def tq_fused_decode_history_current_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: dict[str, torch.Tensor],
    block_table: torch.Tensor,
    old_seq_lens: torch.Tensor | list[int],
    k_rotation: torch.Tensor,
    k_qjl_query_matrix: torch.Tensor,
    k_qjl_proj_t: torch.Tensor,
    k_boundary: torch.Tensor,
    v_rotation: torch.Tensor,
    v_rotation_t: torch.Tensor,
    v_boundary: torch.Tensor,
    k_codebook: torch.Tensor,
    v_codebook: torch.Tensor,
    *,
    k_total_bits: int,
    v_bits: int,
    head_dim: int,
    scale: float,
    output_dtype: torch.dtype | None = None,
    max_seq_len: int | None = None,
    profile_prefix: str | None = None,
) -> torch.Tensor:
    """M4 decode-only custom cache update + attention."""

    _validate_cache(kv_cache)
    if "k_qjl" not in kv_cache or "k_gamma" not in kv_cache:
        raise ValueError("M4 decode attention requires k_qjl/k_gamma sidecars")
    old_seq_lens_t = _as_int32_tensor(old_seq_lens, query.device)
    max_seq = (
        int(max_seq_len)
        if max_seq_len is not None
        else max(_to_int_list(old_seq_lens_t), default=0)
    )
    tensors = (
        query,
        key,
        value,
        slot_mapping,
        kv_cache["k_idx"],
        kv_cache["k_qjl"],
        kv_cache["k_gamma"],
        kv_cache["k_norm"],
        kv_cache["v_idx"],
        kv_cache["v_norm"],
        block_table,
        old_seq_lens_t,
        k_rotation,
        k_qjl_query_matrix,
        k_qjl_proj_t,
        k_boundary,
        v_rotation,
        v_rotation_t,
        v_boundary,
        k_codebook,
        v_codebook,
    )
    if not _custom_op_available("tq_fused_kv_update_attention_decode", *tensors):
        raise RuntimeError(
            "torch.ops._C_ascend.tq_fused_kv_update_attention_decode is unavailable"
        )

    stage_profile = (
        profile_prefix is not None
        and fused_decode_attention_m4_stage_profile_enabled()
    )
    if profile_prefix is not None:
        _maybe_sync_for_profile(*tensors)
        t_total = time.perf_counter()
        t_stage = t_total
    query_f = query.to(torch.float32).contiguous()
    key_f = key.to(torch.float32).contiguous()
    value_f = value.to(torch.float32).contiguous()
    slot_mapping_i64 = slot_mapping.to(torch.long).contiguous()
    block_table_i32 = block_table.to(torch.int32).contiguous()
    k_rotation_f = k_rotation.to(torch.float32).contiguous()
    k_qjl_query_matrix_f = k_qjl_query_matrix.to(torch.float32).contiguous()
    k_qjl_proj_t_f = k_qjl_proj_t.to(torch.float32).contiguous()
    k_boundary_f = k_boundary.to(torch.float32).contiguous()
    v_rotation_f = v_rotation.to(torch.float32).contiguous()
    v_rotation_t_f = v_rotation_t.to(torch.float32).contiguous()
    v_boundary_f = v_boundary.to(torch.float32).contiguous()
    k_codebook_f = k_codebook.to(torch.float32).contiguous()
    v_codebook_f = v_codebook.to(torch.float32).contiguous()
    if stage_profile:
        _maybe_sync_for_profile(
            query_f,
            key_f,
            value_f,
            slot_mapping_i64,
            block_table_i32,
            k_rotation_f,
            k_qjl_query_matrix_f,
            k_qjl_proj_t_f,
            k_boundary_f,
            v_rotation_f,
            v_rotation_t_f,
            v_boundary_f,
            k_codebook_f,
            v_codebook_f,
        )
        _record_tq_profile(
            f"{profile_prefix}.prepare_inputs",
            (time.perf_counter() - t_stage) * 1000.0,
            vectors=int(query.shape[0]),
        )
        t_stage = time.perf_counter()
    out = torch.ops._C_ascend.tq_fused_kv_update_attention_decode(
        query_f,
        key_f,
        value_f,
        slot_mapping_i64,
        kv_cache["k_idx"],
        kv_cache["k_qjl"],
        kv_cache["k_gamma"],
        kv_cache["k_norm"],
        kv_cache["v_idx"],
        kv_cache["v_norm"],
        block_table_i32,
        old_seq_lens_t,
        k_rotation_f,
        k_qjl_query_matrix_f,
        k_qjl_proj_t_f,
        k_boundary_f,
        v_rotation_f,
        v_rotation_t_f,
        v_boundary_f,
        k_codebook_f,
        v_codebook_f,
        int(k_total_bits),
        int(v_bits),
        int(head_dim),
        float(scale),
        int(max_seq),
        fused_decode_attention_m4_score_tile_len(),
        1 if fused_decode_attention_m4_grouped_q_enabled() else 0,
    )
    if stage_profile:
        _maybe_sync_for_profile(out)
        _record_tq_profile(
            f"{profile_prefix}.custom_op",
            (time.perf_counter() - t_stage) * 1000.0,
            vectors=int(query.shape[0]),
        )
        t_stage = time.perf_counter()
    if output_dtype is not None:
        out = out.to(output_dtype)
        if stage_profile:
            _maybe_sync_for_profile(out)
            _record_tq_profile(
                f"{profile_prefix}.output_cast",
                (time.perf_counter() - t_stage) * 1000.0,
                vectors=int(query.shape[0]),
            )
            t_stage = time.perf_counter()
    if profile_prefix is not None:
        if not stage_profile:
            _maybe_sync_for_profile(out)
        _record_tq_profile(
            f"{profile_prefix}.total",
            (time.perf_counter() - t_total) * 1000.0,
            vectors=int(query.shape[0]),
        )
    if (
        profile_prefix is not None
        and fused_decode_attention_m4_shadow_profile_enabled()
    ):
        shadow_t0 = time.perf_counter()
        shadow_out = tq_prod_mse_history_current_decode_attention(
            query,
            key,
            value,
            kv_cache,
            block_table,
            old_seq_lens_t,
            k_codebook,
            v_codebook,
            k_rotation,
            k_qjl_proj_t.transpose(0, 1).contiguous(),
            v_rotation_t,
            k_total_bits=int(k_total_bits),
            v_bits=int(v_bits),
            head_dim=int(head_dim),
            scale=float(scale),
            score_dtype=torch.float32,
            output_dtype=output_dtype,
            profile_prefix=f"{profile_prefix}.shadow",
        )
        _maybe_sync_for_profile(shadow_out)
        _record_tq_profile(
            f"{profile_prefix}.shadow.total",
            (time.perf_counter() - shadow_t0) * 1000.0,
            vectors=int(query.shape[0]),
        )
    return out.contiguous()


def _dense_history_current_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    history_key: torch.Tensor,
    history_value: torch.Tensor,
    old_seq_lens: list[int],
    query_start_loc: torch.Tensor | list[int],
    key_start_loc: torch.Tensor | list[int],
    *,
    scale: float,
    causal: bool,
    score_dtype: torch.dtype,
) -> torch.Tensor:
    q_locs = _to_int_list(query_start_loc)
    k_locs = _to_int_list(key_start_loc)
    if len(q_locs) != len(k_locs):
        raise ValueError("query_start_loc and key_start_loc must have the same length")

    batch = len(q_locs) - 1
    num_heads = int(query.shape[1])
    num_kv_heads = int(key.shape[1])
    head_dim = int(query.shape[-1])
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")

    q_per_kv = num_heads // num_kv_heads
    q_lens = [max(0, int(q_locs[i + 1]) - int(q_locs[i])) for i in range(batch)]
    cur_lens = [max(0, int(k_locs[i + 1]) - int(k_locs[i])) for i in range(batch)]
    if batch > 0 and all(q_len == 1 and cur_len == 1 for q_len, cur_len in zip(q_lens, cur_lens)):
        return _dense_single_token_history_current_attention(
            query,
            key,
            value,
            history_key,
            history_value,
            old_seq_lens,
            q_locs,
            k_locs,
            scale=scale,
            score_dtype=score_dtype,
        )

    out = torch.empty(
        (int(query.shape[0]), num_heads, head_dim),
        dtype=score_dtype,
        device=query.device,
    )
    hist_cursor = 0

    for batch_idx in range(batch):
        q0, q1 = int(q_locs[batch_idx]), int(q_locs[batch_idx + 1])
        k0, k1 = int(k_locs[batch_idx]), int(k_locs[batch_idx + 1])
        old_len = int(old_seq_lens[batch_idx]) if batch_idx < len(old_seq_lens) else 0
        q_len = max(0, q1 - q0)
        cur_len = max(0, k1 - k0)

        q = query[q0:q1].to(score_dtype)
        hist_k = history_key[hist_cursor:hist_cursor + old_len].to(score_dtype)
        hist_v = history_value[hist_cursor:hist_cursor + old_len].to(score_dtype)
        cur_k = key[k0:k1].to(score_dtype)
        cur_v = value[k0:k1].to(score_dtype)
        hist_cursor += old_len

        if q_len == 0:
            continue
        if old_len + cur_len == 0:
            out[q0:q1].zero_()
            continue

        all_k = torch.cat([hist_k, cur_k], dim=0)
        all_v = torch.cat([hist_v, cur_v], dim=0)
        expanded_k = all_k.repeat_interleave(q_per_kv, dim=1)
        expanded_v = all_v.repeat_interleave(q_per_kv, dim=1)

        scores = torch.einsum("qhd,shd->qhs", q, expanded_k) * float(scale)
        if causal:
            total_len = old_len + cur_len
            allowed = torch.arange(total_len, device=query.device).unsqueeze(0)
            current_positions = old_len + torch.arange(q_len, device=query.device).unsqueeze(1)
            scores = scores.masked_fill(
                (allowed > current_positions).unsqueeze(1),
                float("-inf"),
            )

        probs = torch.softmax(scores, dim=-1)
        out[q0:q1] = torch.einsum("qhs,shd->qhd", probs, expanded_v)

    return out.contiguous()


def _dense_single_token_history_current_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    history_key: torch.Tensor,
    history_value: torch.Tensor,
    old_seq_lens: list[int],
    q_locs: list[int],
    k_locs: list[int],
    *,
    scale: float,
    score_dtype: torch.dtype,
) -> torch.Tensor:
    num_heads = int(query.shape[1])
    num_kv_heads = int(key.shape[1])
    head_dim = int(query.shape[-1])
    q_per_kv = num_heads // num_kv_heads
    out = torch.empty(
        (int(query.shape[0]), num_heads, head_dim),
        dtype=score_dtype,
        device=query.device,
    )
    hist_cursor = 0

    for batch_idx in range(len(q_locs) - 1):
        q0 = int(q_locs[batch_idx])
        k0 = int(k_locs[batch_idx])
        old_len = int(old_seq_lens[batch_idx]) if batch_idx < len(old_seq_lens) else 0

        q = query[q0].to(score_dtype).view(num_kv_heads, q_per_kv, head_dim)
        cur_k = key[k0].to(score_dtype)
        cur_v = value[k0].to(score_dtype)
        hist_k = history_key[hist_cursor:hist_cursor + old_len].to(score_dtype)
        hist_v = history_value[hist_cursor:hist_cursor + old_len].to(score_dtype)
        hist_cursor += old_len

        cur_scores = torch.einsum("kqd,kd->kq", q, cur_k).unsqueeze(-1)
        if old_len > 0:
            hist_scores = torch.einsum("kqd,skd->kqs", q, hist_k)
            scores = torch.cat((hist_scores, cur_scores), dim=-1)
        else:
            scores = cur_scores
        probs = torch.softmax(scores * float(scale), dim=-1)

        cur_out = probs[..., -1].unsqueeze(-1) * cur_v.unsqueeze(1)
        if old_len > 0:
            hist_out = torch.einsum("kqs,skd->kqd", probs[..., :old_len], hist_v)
            cur_out = cur_out + hist_out
        out[q0] = cur_out.reshape(num_heads, head_dim)

    return out.contiguous()


def tq_fused_kv_update_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    old_seq_lens: torch.Tensor | list[int],
    query_start_loc: torch.Tensor | list[int],
    key_start_loc: torch.Tensor | list[int],
    kv_cache: dict[str, torch.Tensor],
    k_rotation: torch.Tensor,
    k_qjl_query_matrix: torch.Tensor,
    k_boundary: torch.Tensor,
    k_codebook: torch.Tensor,
    k_qjl_proj: torch.Tensor,
    v_rotation: torch.Tensor,
    v_rotation_t: torch.Tensor,
    v_boundary: torch.Tensor,
    v_codebook: torch.Tensor,
    *,
    k_variant: str,
    k_total_bits: int,
    k_stage1_bits: int,
    v_bits: int,
    head_dim: int,
    scale: float,
    causal: bool = True,
    score_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
    k_qjl_proj_t: torch.Tensor | None = None,
    k_rotation_t: torch.Tensor | None = None,
    kv_mse_rotation: torch.Tensor | None = None,
    kv_mse_shared_boundary: bool = False,
) -> torch.Tensor:
    """Reference implementation for the final unified TurboQuant op.

    Attention reads only ``old_seq_lens`` tokens from the compressed cache.
    The dense current chunk participates directly, while the same call writes
    the current K/V to the compressed sidecar cache for the next step.
    """

    del k_qjl_query_matrix
    _validate_cache(kv_cache)
    _maybe_sync_for_profile(query, key, value, block_table, slot_mapping)
    t0 = time.perf_counter()

    tq_encode_kv_to_paged_cache(
        key,
        value,
        slot_mapping,
        kv_cache,
        k_rotation,
        k_codebook,
        k_boundary,
        k_qjl_proj,
        v_rotation,
        v_codebook,
        v_boundary,
        k_variant=k_variant,
        k_total_bits=int(k_total_bits),
        k_stage1_bits=int(k_stage1_bits),
        v_bits=int(v_bits),
        num_kv_heads=int(key.shape[1]),
        assume_valid_slots=True,
        k_qjl_proj_t=k_qjl_proj_t,
        kv_mse_rotation=kv_mse_rotation,
        kv_mse_shared_boundary=kv_mse_shared_boundary,
    )

    old_seq_lens_list = _to_int_list(old_seq_lens)
    history_k, history_v = _decode_history_to_dense(
        kv_cache,
        block_table,
        old_seq_lens_list,
        k_codebook,
        v_codebook,
        k_rotation_t if k_rotation_t is not None else k_rotation.transpose(0, 1).contiguous(),
        k_qjl_proj,
        v_rotation_t,
        k_variant=k_variant,
        k_total_bits=int(k_total_bits),
        k_stage1_bits=int(k_stage1_bits),
        v_bits=int(v_bits),
        head_dim=int(head_dim),
        target_dtype=score_dtype,
    )
    out = _dense_history_current_attention(
        query,
        key,
        value,
        history_k,
        history_v,
        old_seq_lens_list,
        query_start_loc,
        key_start_loc,
        scale=float(scale),
        causal=bool(causal),
        score_dtype=score_dtype,
    )
    if output_dtype is not None:
        out = out.to(output_dtype)

    _maybe_sync_for_profile(out)
    _record_tq_profile(
        "turboquant_fused_kv_update_attention.reference.total",
        (time.perf_counter() - t0) * 1000.0,
        vectors=int(query.shape[0]),
    )
    return out


def tq_fused_kv_update_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    old_seq_lens: torch.Tensor | list[int],
    query_start_loc: torch.Tensor | list[int],
    key_start_loc: torch.Tensor | list[int],
    kv_cache: dict[str, torch.Tensor],
    k_rotation: torch.Tensor,
    k_qjl_query_matrix: torch.Tensor,
    k_boundary: torch.Tensor,
    k_codebook: torch.Tensor,
    k_qjl_proj: torch.Tensor,
    v_rotation: torch.Tensor,
    v_rotation_t: torch.Tensor,
    v_boundary: torch.Tensor,
    v_codebook: torch.Tensor,
    *,
    k_variant: str,
    k_total_bits: int,
    k_stage1_bits: int | None = None,
    v_bits: int,
    head_dim: int,
    scale: float,
    causal: bool = True,
    score_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
    k_qjl_proj_t: torch.Tensor | None = None,
    k_rotation_t: torch.Tensor | None = None,
    kv_mse_rotation: torch.Tensor | None = None,
    kv_mse_shared_boundary: bool = False,
    mode: int = 0,
    score_tile_len: int = 64,
) -> torch.Tensor:
    """Dispatch the final TurboQuant fused KV-update attention op.

    The reference path is always available.  The custom path is attempted only
    when ``VLLM_ASCEND_TQ_USE_CUSTOM_FUSED_KV_UPDATE_ATTENTION=1`` and the
    compiled ``torch.ops._C_ascend.tq_fused_kv_update_attention`` exists.
    """

    k_stage1_bits = (
        get_stage1_bits(int(k_total_bits), k_variant)
        if k_stage1_bits is None
        else int(k_stage1_bits)
    )
    old_seq_lens_t = _as_int32_tensor(old_seq_lens, query.device)
    query_start_loc_t = _as_int32_tensor(query_start_loc, query.device)
    key_start_loc_t = _as_int32_tensor(key_start_loc, query.device)

    fallback_reasons: list[str] = []
    custom_enabled = fused_kv_update_attention_custom_enabled()
    if custom_enabled:
        tensors = (
            query,
            key,
            value,
            slot_mapping,
            block_table,
            old_seq_lens_t,
            query_start_loc_t,
            key_start_loc_t,
            kv_cache["k_idx"],
            kv_cache["k_norm"],
            kv_cache["v_idx"],
            kv_cache["v_norm"],
            k_rotation,
            k_qjl_query_matrix,
            k_boundary,
            k_codebook,
            k_qjl_proj,
            v_rotation,
            v_rotation_t,
            v_boundary,
            v_codebook,
        )
        if not _custom_op_available("tq_fused_kv_update_attention", *tensors):
            fallback_reasons.append("torch.ops._C_ascend.tq_fused_kv_update_attention is unavailable")
    else:
        fallback_reasons.append("custom fused KV-update attention is not enabled")

    if custom_enabled and not fallback_reasons:
        return torch.ops._C_ascend.tq_fused_kv_update_attention(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            slot_mapping.contiguous(),
            block_table.contiguous(),
            old_seq_lens_t,
            query_start_loc_t,
            key_start_loc_t,
            kv_cache["k_idx"],
            kv_cache.get("k_qjl", torch.empty(0, device=query.device, dtype=torch.uint8)),
            kv_cache.get("k_gamma", torch.empty(0, device=query.device, dtype=torch.float32)),
            kv_cache["k_norm"],
            kv_cache["v_idx"],
            kv_cache["v_norm"],
            k_rotation.contiguous(),
            k_qjl_query_matrix.contiguous(),
            k_boundary.contiguous(),
            k_codebook.contiguous(),
            k_qjl_proj.contiguous(),
            v_rotation.contiguous(),
            v_rotation_t.contiguous(),
            v_boundary.contiguous(),
            v_codebook.contiguous(),
            int(k_total_bits),
            int(k_stage1_bits),
            int(v_bits),
            int(head_dim),
            float(scale),
            int(mode),
            1 if causal else 0,
            int(score_tile_len),
        )

    if custom_enabled and custom_strict_enabled():
        raise RuntimeError(
            "TurboQuant fused KV-update attention custom op fallback: "
            + "; ".join(fallback_reasons)
        )

    return tq_fused_kv_update_attention_reference(
        query,
        key,
        value,
        slot_mapping,
        block_table,
        old_seq_lens_t,
        query_start_loc_t,
        key_start_loc_t,
        kv_cache,
        k_rotation,
        k_qjl_query_matrix,
        k_boundary,
        k_codebook,
        k_qjl_proj,
        v_rotation,
        v_rotation_t,
        v_boundary,
        v_codebook,
        k_variant=k_variant,
        k_total_bits=int(k_total_bits),
        k_stage1_bits=int(k_stage1_bits),
        v_bits=int(v_bits),
        head_dim=int(head_dim),
        scale=float(scale),
        causal=bool(causal),
        score_dtype=score_dtype,
        output_dtype=output_dtype,
        k_qjl_proj_t=k_qjl_proj_t,
        k_rotation_t=k_rotation_t,
        kv_mse_rotation=kv_mse_rotation,
        kv_mse_shared_boundary=kv_mse_shared_boundary,
    )


__all__ = [
    "current_lengths_from_start_loc",
    "combined_kv_mse_encode_enabled",
    "compressed_decode_current_custom_k_score_enabled",
    "compressed_decode_current_enabled",
    "decode_compressed_full_cache_enabled",
    "encode_cache_update_custom_enabled",
    "encode_cache_update_stage_profile_enabled",
    "fused_decode_attention_m4_enabled",
    "fused_decode_attention_m4_grouped_q_enabled",
    "fused_decode_attention_m4_score_tile_len",
    "fused_decode_attention_m4_stage_profile_enabled",
    "fused_kv_update_attention_custom_enabled",
    "fused_kv_update_attention_enabled",
    "old_seq_lens_from_total",
    "tq_decode_history_to_dense",
    "tq_encode_kv_to_paged_cache",
    "tq_encode_kv_to_paged_cache_reference",
    "tq_fused_kv_update_attention",
    "tq_fused_kv_update_attention_reference",
    "tq_fused_decode_history_current_attention",
    "tq_prod_mse_history_current_decode_attention",
]
