/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
 * This file is a part of the vllm-ascend project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * TurboQuant MSE paged dequant — Ascend C kernel.
 *
 * This is the first custom kernel in the P2→P4 roadmap. It replaces the
 * heaviest part of the Python reference path:
 *
 *   paged gather + bit unpack + codebook lookup + norm rescale
 *
 * The kernel operates on pre-expanded token maps so it never touches
 * the vLLM block_table / ragged batching logic directly.
 *
 * Output convention: the result is in *rotated space* (no inverse
 * rotation). The Python caller applies ``dense = dense_rot @ rotation_t``.
 */

#include "kernel_operator.h"

using namespace AscendC;

template <typename NormT, typename CodebookT>
class KernelTqDequantMsePaged {
public:
    __aicore__ inline KernelTqDequantMsePaged() {}

    __aicore__ inline void Init(
        GM_ADDR packed_idx,
        GM_ADDR norm,
        GM_ADDR token_block_ids,
        GM_ADDR token_offsets,
        GM_ADDR codebook,
        GM_ADDR out,
        uint32_t total_tokens,
        uint32_t block_size,
        uint32_t num_kv_heads,
        uint32_t head_dim,
        uint32_t packed_cols,
        uint32_t bits,
        uint32_t core_num
    ) {
        packed_idx_.SetGlobalBuffer((__gm__ uint8_t*)packed_idx);
        norm_.SetGlobalBuffer((__gm__ NormT*)norm);
        token_block_ids_.SetGlobalBuffer((__gm__ int32_t*)token_block_ids);
        token_offsets_.SetGlobalBuffer((__gm__ int32_t*)token_offsets);
        codebook_.SetGlobalBuffer((__gm__ CodebookT*)codebook);
        out_.SetGlobalBuffer((__gm__ float*)out);

        total_tokens_ = total_tokens;
        block_size_ = block_size;
        num_kv_heads_ = num_kv_heads;
        head_dim_ = head_dim;
        packed_cols_ = packed_cols;
        bits_ = bits;
        core_num_ = core_num;
    }

    __aicore__ inline uint32_t ExtractIndex(
        uint64_t packed_base,
        uint32_t d
    ) {
        uint32_t bit_pos = d * bits_;
        uint32_t byte_id = bit_pos >> 3;
        uint32_t bit_off = bit_pos & 7;

        uint16_t v = (uint16_t)packed_idx_.GetValue(packed_base + byte_id);

        // Cross-byte read for 3-bit / 4-bit
        if (bit_off + bits_ > 8) {
            uint16_t high = (uint16_t)packed_idx_.GetValue(packed_base + byte_id + 1);
            v = v | (high << 8);
        }

        uint32_t mask = (1u << bits_) - 1u;
        return (v >> bit_off) & mask;
    }

    __aicore__ inline void Process() {
        uint32_t core_id = GetBlockIdx();

        uint64_t total_elements =
            (uint64_t)total_tokens_ * (uint64_t)num_kv_heads_ * (uint64_t)head_dim_;

        uint64_t elems_per_core = (total_elements + core_num_ - 1) / core_num_;
        uint64_t start = (uint64_t)core_id * elems_per_core;
        uint64_t end = start + elems_per_core;
        if (end > total_elements) {
            end = total_elements;
        }

        for (uint64_t linear = start; linear < end; ++linear) {
            uint32_t d = linear % head_dim_;
            uint64_t tmp = linear / head_dim_;

            uint32_t kv_head = tmp % num_kv_heads_;
            uint32_t token = tmp / num_kv_heads_;

            int32_t block_id = token_block_ids_.GetValue(token);
            int32_t offset = token_offsets_.GetValue(token);

            uint64_t packed_base =
                (((uint64_t)block_id * block_size_ + (uint64_t)offset)
                    * num_kv_heads_ + (uint64_t)kv_head)
                    * packed_cols_;

            uint64_t norm_index =
                (((uint64_t)block_id * block_size_ + (uint64_t)offset)
                    * num_kv_heads_ + (uint64_t)kv_head);

            uint32_t idx = ExtractIndex(packed_base, d);

            float cb = (float)codebook_.GetValue(idx);
            float scale = (float)norm_.GetValue(norm_index);

            float y = cb * scale;

            uint64_t out_index =
                ((uint64_t)token * num_kv_heads_ + (uint64_t)kv_head)
                    * head_dim_ + (uint64_t)d;

            out_.SetValue(out_index, y);
        }
    }

private:
    GlobalTensor<uint8_t> packed_idx_;
    GlobalTensor<NormT> norm_;
    GlobalTensor<int32_t> token_block_ids_;
    GlobalTensor<int32_t> token_offsets_;
    GlobalTensor<CodebookT> codebook_;
    GlobalTensor<float> out_;

    uint32_t total_tokens_;
    uint32_t block_size_;
    uint32_t num_kv_heads_;
    uint32_t head_dim_;
    uint32_t packed_cols_;
    uint32_t bits_;
    uint32_t core_num_;
};


extern "C" __global__ __aicore__ void tq_dequant_mse_paged_kernel_fp32(
    GM_ADDR packed_idx,
    GM_ADDR norm,
    GM_ADDR token_block_ids,
    GM_ADDR token_offsets,
    GM_ADDR codebook,
    GM_ADDR out,
    uint32_t total_tokens,
    uint32_t block_size,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t packed_cols,
    uint32_t bits,
    uint32_t core_num
) {
    KernelTqDequantMsePaged<float, float> op;
    op.Init(
        packed_idx,
        norm,
        token_block_ids,
        token_offsets,
        codebook,
        out,
        total_tokens,
        block_size,
        num_kv_heads,
        head_dim,
        packed_cols,
        bits,
        core_num
    );
    op.Process();
}
