#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#

from __future__ import annotations

import re

import torch

from .base import AscendAttentionScheme
from .turboquant_runtime import (
    build_qjl_projection,
    build_rotation_matrix,
    build_turboquant_codebook,
    get_stage1_bits,
)


def _parse_layer_id(prefix: str) -> int:
    match = re.search(r"\.(\d+)\.", prefix)
    return int(match.group(1)) if match else 0


class AscendTurboQuantKVCacheAttentionMethod(AscendAttentionScheme):
    def __init__(self, quant_description: dict, prefix: str):
        self.quant_description = quant_description
        self.prefix = prefix
        self.layer_id = _parse_layer_id(prefix)
        tq_config = quant_description.get("turboquant", {})
        self.k_variant = tq_config.get("k_variant", "prod")
        self.v_variant = tq_config.get("v_variant", "mse")
        self.k_total_bits = int(tq_config.get("k_total_bits", tq_config.get("k_bits", 3)))
        self.v_total_bits = int(tq_config.get("v_total_bits", tq_config.get("v_bits", 2)))
        self.rotation_scope = tq_config.get("rotation_scope", "per_layer")
        self.rotation_seed_base = int(tq_config.get("rotation_seed_base", 1234))
        self.outlier_channels = int(tq_config.get("outlier_channels", 0))
        self.dequant_mode = tq_config.get("dequant_mode", "dense_then_attention")
        scalar_dtype_str = tq_config.get("scalar_dtype", "float32")
        _valid_scalar_dtypes = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if scalar_dtype_str not in _valid_scalar_dtypes:
            raise ValueError(
                f"Unsupported TurboQuant scalar_dtype={scalar_dtype_str} for {prefix}. "
                f"Supported: {list(_valid_scalar_dtypes.keys())}"
            )
        self.scalar_dtype = _valid_scalar_dtypes[scalar_dtype_str]

        if self.k_variant not in {"mse", "prod"}:
            raise ValueError(f"Unsupported TurboQuant k_variant={self.k_variant} for {prefix}")
        if self.v_variant != "mse":
            raise ValueError(
                f"TurboQuant MVP only supports v_variant='mse', got {self.v_variant} for {prefix}"
            )
        if self.rotation_scope != "per_layer":
            raise ValueError(
                f"TurboQuant MVP only supports rotation_scope='per_layer', got {self.rotation_scope} for {prefix}"
            )
        if self.outlier_channels != 0:
            raise ValueError(
                f"TurboQuant MVP does not implement outlier_channels yet, got {self.outlier_channels} for {prefix}"
            )
        if self.dequant_mode != "dense_then_attention":
            raise ValueError(
                "TurboQuant MVP only supports dequant_mode='dense_then_attention', "
                f"got {self.dequant_mode} for {prefix}"
            )
        self.k_stage1_bits = get_stage1_bits(self.k_total_bits, self.k_variant)
        self.v_stage1_bits = get_stage1_bits(self.v_total_bits, self.v_variant)

    def create_weights(self, layer: torch.nn.Module) -> None:
        from vllm_ascend.attention.attention_v1 import AscendTurboQuantAttentionBackendImpl

        k_dim = int(layer.impl.head_size)
        v_dim = int(getattr(layer.impl, "head_size_v", k_dim))
        params_dtype = torch.float32
        seed = self.rotation_seed_base + self.layer_id

        k_codebook, k_boundary = build_turboquant_codebook(k_dim, self.k_stage1_bits, "cpu", params_dtype)
        v_codebook, v_boundary = build_turboquant_codebook(v_dim, self.v_stage1_bits, "cpu", params_dtype)
        k_rot = build_rotation_matrix(k_dim, seed, "cpu", params_dtype)
        v_rot = build_rotation_matrix(v_dim, seed + 1, "cpu", params_dtype)
        k_qjl_proj = build_qjl_projection(k_dim, seed + 2, "cpu", params_dtype)

        layer.kv_cache_torch_dtype = torch.int8
        layer.impl.__class__ = AscendTurboQuantAttentionBackendImpl
        layer.turboquant_enabled = True
        layer.tq_k_variant = self.k_variant
        layer.tq_v_variant = self.v_variant
        layer.tq_k_total_bits = self.k_total_bits
        layer.tq_v_total_bits = self.v_total_bits
        layer.tq_k_stage1_bits = self.k_stage1_bits
        layer.tq_v_stage1_bits = self.v_stage1_bits
        layer.tq_rotation_scope = self.rotation_scope
        layer.tq_outlier_channels = self.outlier_channels
        layer.tq_dequant_mode = self.dequant_mode
        layer.tq_scalar_dtype = self.scalar_dtype
        layer.tq_layer_id = self.layer_id
        layer.tq_head_size_v = v_dim
        layer.tq_runtime_prepared = False

        layer.register_buffer("k_codebook", k_codebook, persistent=False)
        layer.register_buffer("v_codebook", v_codebook, persistent=False)
        layer.register_buffer("k_boundary", k_boundary, persistent=False)
        layer.register_buffer("v_boundary", v_boundary, persistent=False)
        layer.register_buffer("k_rot", k_rot, persistent=False)
        layer.register_buffer("v_rot", v_rot, persistent=False)
        layer.register_buffer("k_qjl_proj", k_qjl_proj, persistent=False)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.k_codebook.data = layer.k_codebook.data.contiguous()
        layer.v_codebook.data = layer.v_codebook.data.contiguous()
        layer.k_boundary.data = layer.k_boundary.data.contiguous()
        layer.v_boundary.data = layer.v_boundary.data.contiguous()
        layer.k_rot.data = layer.k_rot.data.contiguous()
        layer.v_rot.data = layer.v_rot.data.contiguous()
        layer.k_qjl_proj.data = layer.k_qjl_proj.data.contiguous()
        layer.tq_runtime_prepared = False

    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        raise RuntimeError("TurboQuant KV cache is handled by attention backend")
