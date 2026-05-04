/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * PyTorch / pybind11 binding for the TurboQuant MSE dequant Ascend C kernel.
 *
 * Builds a Python-importable shared library ``vllm_ascend_tq_ops`` that
 * exposes ``tq_dequant_mse_paged()``. The Python wrapper in
 * ``vllm_ascend/ops/turboquant/dequant.py`` loads this extension at
 * runtime when ``VLLM_ASCEND_TQ_USE_CUSTOM_DEQUANT=1``.
 *
 * NOTE: The include path for torch_npu headers and the kernel-launch
 * ABI depend on the installed CANN / torch-npu version. If compilation
 * fails, first locate the correct headers:
 *
 *   python - <<'PY'
 *   import torch_npu, os
 *   print(os.path.dirname(torch_npu.__file__))
 *   PY
 *
 * Common adjustments:
 *   - #include "torch_npu/csrc/core/npu/NPUStream.h"  (may differ by version)
 *   - launch API: direct <<< >>> or OpCommand (see comments below)
 */

#include <torch/extension.h>
#include <ATen/ATen.h>

#include <stdexcept>
#include <string>

// -----------------------------------------------------------------------
// torch_npu header — adjust path to match your installed version.
// Run the Python snippet above to find the correct root.
// -----------------------------------------------------------------------
#include "torch_npu/csrc/core/npu/NPUStream.h"


// -----------------------------------------------------------------------
// Forward declaration of the Ascend C kernel entry point.
// -----------------------------------------------------------------------
extern "C" void tq_dequant_mse_paged_kernel_fp32(
    void* packed_idx,
    void* norm,
    void* token_block_ids,
    void* token_offsets,
    void* codebook,
    void* out,
    uint32_t total_tokens,
    uint32_t block_size,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t packed_cols,
    uint32_t bits,
    uint32_t core_num
);


// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

static void check_npu_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_privateuseone(), name, " must be an NPU tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}


// -----------------------------------------------------------------------
// Main entry point
// -----------------------------------------------------------------------

torch::Tensor tq_dequant_mse_paged(
    torch::Tensor packed_idx,
    torch::Tensor norm,
    torch::Tensor token_block_ids,
    torch::Tensor token_offsets,
    torch::Tensor codebook,
    int64_t bits,
    int64_t head_dim
) {
    check_npu_tensor(packed_idx, "packed_idx");
    check_npu_tensor(norm, "norm");
    check_npu_tensor(token_block_ids, "token_block_ids");
    check_npu_tensor(token_offsets, "token_offsets");
    check_npu_tensor(codebook, "codebook");

    TORCH_CHECK(packed_idx.scalar_type() == at::ScalarType::Byte,
                "packed_idx must be uint8");
    TORCH_CHECK(token_block_ids.scalar_type() == at::ScalarType::Int,
                "token_block_ids must be int32");
    TORCH_CHECK(token_offsets.scalar_type() == at::ScalarType::Int,
                "token_offsets must be int32");

    TORCH_CHECK(bits >= 1 && bits <= 4, "bits must be in [1, 4]");
    TORCH_CHECK(packed_idx.dim() == 4,
                "packed_idx must be [num_blocks, block_size, num_kv_heads, packed_cols]");
    TORCH_CHECK(norm.dim() == 4,
                "norm must be [num_blocks, block_size, num_kv_heads, 1]");
    TORCH_CHECK(norm.size(3) == 1, "norm last dim must be 1");

    int64_t total_tokens = token_block_ids.numel();
    TORCH_CHECK(token_offsets.numel() == total_tokens,
                "token_offsets length must equal token_block_ids length");

    int64_t num_blocks = packed_idx.size(0);
    int64_t block_size = packed_idx.size(1);
    int64_t num_kv_heads = packed_idx.size(2);
    int64_t packed_cols = packed_idx.size(3);

    TORCH_CHECK(norm.size(0) == num_blocks, "norm num_blocks mismatch");
    TORCH_CHECK(norm.size(1) == block_size, "norm block_size mismatch");
    TORCH_CHECK(norm.size(2) == num_kv_heads, "norm num_kv_heads mismatch");
    TORCH_CHECK(head_dim > 0, "head_dim must be positive");
    TORCH_CHECK(packed_cols == (head_dim * bits + 7) / 8,
                "packed_cols mismatch: expected ",
                (head_dim * bits + 7) / 8, " got ", packed_cols);

    // Output fp32 for the first version — easiest to compare with reference.
    auto out = torch::empty(
        {total_tokens, num_kv_heads, head_dim},
        packed_idx.options().dtype(at::ScalarType::Float)
    );

    if (total_tokens == 0) {
        return out;
    }

    // -------------------------------------------------------------------
    // Core count — tune for your device.
    // 910B / 910B2: 32–64 AI Cores. Start with a fixed value and adjust
    // after profiling.
    // -------------------------------------------------------------------
    uint32_t core_num = 32;

    // -------------------------------------------------------------------
    // Direct-launch approach (recommended for research prototypes).
    //
    // If your torch-npu / CANN version supports <<< >>> syntax, use:
    // -------------------------------------------------------------------
    auto stream = c10_npu::getCurrentNPUStream().stream();
    tq_dequant_mse_paged_kernel_fp32<<<core_num, nullptr, stream>>>(
        packed_idx.data_ptr(),
        norm.data_ptr(),
        token_block_ids.data_ptr(),
        token_offsets.data_ptr(),
        codebook.data_ptr(),
        out.data_ptr(),
        static_cast<uint32_t>(total_tokens),
        static_cast<uint32_t>(block_size),
        static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(packed_cols),
        static_cast<uint32_t>(bits),
        core_num
    );

    // -------------------------------------------------------------------
    // OpCommand alternative (if direct launch is not supported):
    //
    // at_npu::native::OpCommand cmd;
    // cmd.Name("TqDequantMsePaged")
    //    .Input(packed_idx)
    //    .Input(norm)
    //    .Input(token_block_ids)
    //    .Input(token_offsets)
    //    .Input(codebook)
    //    .Output(out)
    //    .Attr("bits", bits)
    //    .Attr("head_dim", head_dim)
    //    .Run();
    // -------------------------------------------------------------------

    return out;
}


// -----------------------------------------------------------------------
// pybind11 module definition
// -----------------------------------------------------------------------

PYBIND11_MODULE(vllm_ascend_tq_ops, m) {
    m.def(
        "tq_dequant_mse_paged",
        &tq_dequant_mse_paged,
        "TurboQuant MSE paged dequant Ascend C op"
    );
}
