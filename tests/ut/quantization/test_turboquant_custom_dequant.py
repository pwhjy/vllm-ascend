"""Standalone parity tests for TurboQuant paged dequant API.

These tests verify that the ``tq_dequant_mse_paged_rot`` wrapper and
``build_token_map_from_block_table`` produce correct results independent
of the vLLM attention backend. They run on CPU so CI can gate correctness
before NPU hardware is available.
"""

import math

import pytest
import torch

from vllm_ascend.ops.turboquant.dequant import (
    build_token_map_from_block_table,
    cached_token_map_from_block_table,
    tq_dequant_mse_paged_reference_rot,
    tq_dequant_mse_paged_rot,
    tq_dequant_mse_paged_scaled_rot,
    tq_dequant_prod_paged_k_score,
    tq_dequant_prod_paged_k_score_reference,
    tq_dequant_prod_paged_reference_rot,
    tq_dequant_prod_paged_rot,
)
from vllm_ascend.quantization.methods.turboquant_layout import (
    get_stage1_bits,
)
from vllm_ascend.quantization.methods.turboquant_runtime import (
    apply_rotation,
    build_qjl_projection,
    build_rotation_matrix,
    build_turboquant_codebook,
    pack_bits,
    turboquant_decode_prod,
    turboquant_encode_prod,
)


def _npu_available() -> bool:
    npu = getattr(torch, "npu", None)
    if npu is None or not hasattr(npu, "is_available"):
        return False
    try:
        return bool(npu.is_available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_random_paged_inputs(
    bits: int,
    head_dim: int,
    num_blocks: int = 4,
    block_size: int = 16,
    num_kv_heads: int = 2,
    total_tokens: int = 20,
    device: str = "cpu",
):
    levels = 1 << bits
    idx = torch.randint(
        0, levels,
        (num_blocks, block_size, num_kv_heads, head_dim),
        dtype=torch.uint8, device=device,
    )
    packed = pack_bits(idx, bits=bits).contiguous()

    norm = torch.rand(
        (num_blocks, block_size, num_kv_heads, 1),
        dtype=torch.float32, device=device,
    )

    codebook, _ = build_turboquant_codebook(head_dim, bits, device, torch.float32)

    token_block_ids = torch.randint(
        0, num_blocks,
        (total_tokens,), dtype=torch.int32, device=device,
    )
    token_offsets = torch.randint(
        0, block_size,
        (total_tokens,), dtype=torch.int32, device=device,
    )

    return packed, norm, codebook, token_block_ids, token_offsets


def _make_random_prod_paged_inputs(
    total_bits: int,
    head_dim: int,
    num_blocks: int = 4,
    block_size: int = 8,
    num_kv_heads: int = 2,
    total_tokens: int = 12,
    device: str = "cpu",
):
    stage1_bits = get_stage1_bits(total_bits, "prod")
    levels = 1 << stage1_bits
    idx = torch.randint(
        0, levels,
        (num_blocks, block_size, num_kv_heads, head_dim),
        dtype=torch.uint8, device=device,
    )
    packed_idx = pack_bits(idx, bits=stage1_bits).contiguous()

    qjl = torch.randint(
        0, 2,
        (num_blocks, block_size, num_kv_heads, head_dim),
        dtype=torch.uint8, device=device,
    )
    packed_qjl = pack_bits(qjl, bits=1).contiguous()

    gamma = torch.rand(
        (num_blocks, block_size, num_kv_heads, 1),
        dtype=torch.float32, device=device,
    )
    norm = torch.rand(
        (num_blocks, block_size, num_kv_heads, 1),
        dtype=torch.float32, device=device,
    )

    codebook, _ = build_turboquant_codebook(
        head_dim, stage1_bits, device, torch.float32,
    )
    qjl_proj = build_qjl_projection(head_dim, 2026, device, torch.float32)

    token_block_ids = torch.randint(
        0, num_blocks,
        (total_tokens,), dtype=torch.int32, device=device,
    )
    token_offsets = torch.randint(
        0, block_size,
        (total_tokens,), dtype=torch.int32, device=device,
    )

    return (
        packed_idx, packed_qjl, gamma, norm,
        codebook, qjl_proj, token_block_ids, token_offsets,
    )


# ---------------------------------------------------------------------------
# dequant parity tests
# ---------------------------------------------------------------------------

class TestTqDequantMsePagedRot:
    """Verify that the dispatch wrapper and reference produce identical output."""

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    @pytest.mark.parametrize("head_dim", [8, 64, 80, 96, 128])
    def test_matches_reference(self, bits, head_dim):
        packed, norm, codebook, token_block_ids, token_offsets = \
            _make_random_paged_inputs(bits, head_dim)

        ref = tq_dequant_mse_paged_reference_rot(
            packed, norm, token_block_ids, token_offsets,
            codebook, bits, head_dim, torch.float32,
        )

        out = tq_dequant_mse_paged_rot(
            packed, norm, token_block_ids, token_offsets,
            codebook, bits, head_dim, torch.float32,
        )

        assert out.shape == ref.shape
        assert torch.equal(out, ref), \
            f"Mismatch for bits={bits} head_dim={head_dim}"

    def test_zero_tokens_returns_empty(self):
        packed, norm, codebook, _, _ = _make_random_paged_inputs(
            bits=3, head_dim=64, total_tokens=0,
        )
        empty_ids = torch.empty((0,), dtype=torch.int32)
        empty_offsets = torch.empty((0,), dtype=torch.int32)

        ref = tq_dequant_mse_paged_reference_rot(
            packed, norm, empty_ids, empty_offsets,
            codebook, 3, 64, torch.float32,
        )
        out = tq_dequant_mse_paged_rot(
            packed, norm, empty_ids, empty_offsets,
            codebook, 3, 64, torch.float32,
        )

        assert out.shape == (0, packed.shape[2], 64)
        assert torch.equal(out, ref)

    def test_different_target_dtypes(self):
        packed, norm, codebook, token_block_ids, token_offsets = \
            _make_random_paged_inputs(bits=3, head_dim=64)

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            out = tq_dequant_mse_paged_rot(
                packed, norm, token_block_ids, token_offsets,
                codebook, 3, 64, dtype,
            )
            assert out.dtype == dtype
            assert out.shape[1:] == (packed.shape[2], 64)

    def test_cpu_tensors_fallback_when_custom_env_enabled(self, monkeypatch):
        monkeypatch.setenv("VLLM_ASCEND_TQ_USE_CUSTOM_DEQUANT", "1")
        packed, norm, codebook, token_block_ids, token_offsets = \
            _make_random_paged_inputs(bits=3, head_dim=8)

        ref = tq_dequant_mse_paged_reference_rot(
            packed, norm, token_block_ids, token_offsets,
            codebook, 3, 8, torch.float32,
        )
        out = tq_dequant_mse_paged_rot(
            packed, norm, token_block_ids, token_offsets,
            codebook, 3, 8, torch.float32,
        )

        assert torch.equal(out, ref)

    @pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    @pytest.mark.parametrize("head_dim", [64, 80, 96, 128])
    def test_custom_op_matches_reference_on_npu(self, monkeypatch, bits, head_dim):
        monkeypatch.setenv("VLLM_ASCEND_TQ_USE_CUSTOM_DEQUANT", "1")
        monkeypatch.setenv("VLLM_ASCEND_TQ_CUSTOM_STRICT", "1")

        packed, norm, codebook, token_block_ids, token_offsets = \
            _make_random_paged_inputs(bits, head_dim, device="npu")

        ref = tq_dequant_mse_paged_reference_rot(
            packed, norm, token_block_ids, token_offsets,
            codebook, bits, head_dim, torch.float32,
        )
        out = tq_dequant_mse_paged_rot(
            packed, norm, token_block_ids, token_offsets,
            codebook, bits, head_dim, torch.float32,
        )

        torch.npu.synchronize()
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-6, rtol=0), \
            f"Mismatch for bits={bits} head_dim={head_dim}"

    @pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_custom_op_writes_target_dtype_on_npu(self, monkeypatch, dtype):
        monkeypatch.setenv("VLLM_ASCEND_TQ_USE_CUSTOM_DEQUANT", "1")
        monkeypatch.setenv("VLLM_ASCEND_TQ_CUSTOM_STRICT", "1")

        bits = 3
        head_dim = 128
        packed, norm, codebook, token_block_ids, token_offsets = \
            _make_random_paged_inputs(bits, head_dim, device="npu")

        ref = tq_dequant_mse_paged_reference_rot(
            packed, norm, token_block_ids, token_offsets,
            codebook, bits, head_dim, dtype,
        )
        out = tq_dequant_mse_paged_rot(
            packed, norm, token_block_ids, token_offsets,
            codebook, bits, head_dim, dtype,
        )

        torch.npu.synchronize()
        assert out.dtype == dtype
        assert out.shape == ref.shape
        assert torch.allclose(out.float(), ref.float(), atol=1e-3, rtol=0)

    @pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_tiny_head_dim_uses_reference_on_npu(self, monkeypatch, bits):
        monkeypatch.setenv("VLLM_ASCEND_TQ_USE_CUSTOM_DEQUANT", "1")
        monkeypatch.setenv("VLLM_ASCEND_TQ_DEBUG_COMPARE", "1")

        head_dim = 8
        packed, norm, codebook, token_block_ids, token_offsets = \
            _make_random_paged_inputs(bits, head_dim, device="npu")

        ref = tq_dequant_mse_paged_reference_rot(
            packed, norm, token_block_ids, token_offsets,
            codebook, bits, head_dim, torch.float32,
        )
        out = tq_dequant_mse_paged_rot(
            packed, norm, token_block_ids, token_offsets,
            codebook, bits, head_dim, torch.float32,
        )

        torch.npu.synchronize()
        assert torch.equal(out, ref)

    @pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
    def test_custom_op_matches_reference_from_npu_block_table(self, monkeypatch):
        monkeypatch.setenv("VLLM_ASCEND_TQ_USE_CUSTOM_DEQUANT", "1")
        monkeypatch.setenv("VLLM_ASCEND_TQ_CUSTOM_STRICT", "1")

        bits = 3
        head_dim = 80
        block_size = 4
        packed, norm, codebook, _, _ = _make_random_paged_inputs(
            bits=bits,
            head_dim=head_dim,
            num_blocks=24,
            block_size=block_size,
            total_tokens=1,
            device="npu",
        )
        block_table = torch.tensor([
            [10, 3, 0],
            [7, 15, 21],
        ], dtype=torch.int32, device="npu")
        seq_lens = [5, 9]
        token_block_ids, token_offsets = build_token_map_from_block_table(
            block_table, seq_lens, block_size,
        )

        ref = tq_dequant_mse_paged_reference_rot(
            packed, norm, token_block_ids, token_offsets,
            codebook, bits, head_dim, torch.float32,
        )
        out = tq_dequant_mse_paged_rot(
            packed, norm, token_block_ids, token_offsets,
            codebook, bits, head_dim, torch.float32,
        )

        torch.npu.synchronize()
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-6, rtol=0)

    def test_prod_hybrid_stage1_mse_plus_qjl_matches_reference(self):
        total_bits = 3
        stage1_bits = total_bits - 1
        head_dim = 16
        num_blocks = 4
        block_size = 2
        num_kv_heads = 2

        torch.manual_seed(0)
        x = torch.randn(
            num_blocks * block_size, num_kv_heads, head_dim,
            dtype=torch.float32,
        )
        rotation = build_rotation_matrix(head_dim, 123, "cpu", torch.float32)
        rotation_t = rotation.transpose(0, 1).contiguous()
        qjl_proj = build_qjl_projection(head_dim, 124, "cpu", torch.float32)
        codebook, boundary = build_turboquant_codebook(
            head_dim, stage1_bits, "cpu", torch.float32,
        )
        encoded = turboquant_encode_prod(
            x, rotation, codebook, boundary, qjl_proj, total_bits,
        )

        def _as_cache(name):
            return encoded[name].view(num_blocks, block_size, num_kv_heads, -1)

        k_idx = _as_cache("idx")
        k_qjl = _as_cache("qjl")
        k_gamma = _as_cache("gamma")
        k_norm = _as_cache("norm")
        block_table = torch.tensor([
            [2, 0],
            [3, 1],
        ], dtype=torch.int32)
        seq_lens = [3, 4]
        token_block_ids, token_offsets = build_token_map_from_block_table(
            block_table, seq_lens, block_size,
        )

        ref = turboquant_decode_prod(
            k_idx[token_block_ids.long(), token_offsets.long()],
            k_qjl[token_block_ids.long(), token_offsets.long()],
            k_gamma[token_block_ids.long(), token_offsets.long()],
            k_norm[token_block_ids.long(), token_offsets.long()],
            rotation_t,
            codebook,
            qjl_proj,
            total_bits,
            head_dim,
            torch.float32,
        )

        stage1_rot = tq_dequant_mse_paged_rot(
            k_idx,
            k_norm,
            token_block_ids,
            token_offsets,
            codebook,
            stage1_bits,
            head_dim,
            torch.float32,
        )
        qjl_codebook = torch.tensor([-1.0, 1.0], dtype=torch.float32)
        qjl_scaled = tq_dequant_mse_paged_scaled_rot(
            k_qjl,
            k_norm,
            k_gamma,
            token_block_ids,
            token_offsets,
            qjl_codebook,
            1,
            head_dim,
            torch.float32,
            math.sqrt(math.pi / 2.0) / head_dim,
            signed_bits1=True,
        )
        qjl_rot = apply_rotation(qjl_scaled, qjl_proj)
        out = apply_rotation(stage1_rot + qjl_rot, rotation_t).contiguous()

        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    def test_scaled_mse_paged_rot_matches_combined_scale_reference(self):
        bits = 1
        head_dim = 64
        packed, norm, _, token_block_ids, token_offsets = \
            _make_random_paged_inputs(bits, head_dim)
        extra_scale = torch.rand_like(norm)
        codebook = torch.tensor([-1.0, 1.0], dtype=torch.float32)
        scale_multiplier = 0.125
        combined_scale = (norm * extra_scale * scale_multiplier).contiguous()

        ref = tq_dequant_mse_paged_reference_rot(
            packed, combined_scale, token_block_ids, token_offsets,
            codebook, bits, head_dim, torch.float32,
        )
        out = tq_dequant_mse_paged_scaled_rot(
            packed,
            norm,
            extra_scale,
            token_block_ids,
            token_offsets,
            codebook,
            bits,
            head_dim,
            torch.float32,
            scale_multiplier,
        )

        assert torch.equal(out, ref)

        out_signed = tq_dequant_mse_paged_scaled_rot(
            packed,
            norm,
            extra_scale,
            token_block_ids,
            token_offsets,
            codebook,
            bits,
            head_dim,
            torch.float32,
            scale_multiplier,
            signed_bits1=True,
        )

        assert torch.equal(out_signed, ref)

    @pytest.mark.parametrize("total_bits", [2, 3, 4])
    @pytest.mark.parametrize("head_dim", [16, 64])
    def test_prod_paged_reference_rot_matches_decode_prod(self, total_bits, head_dim):
        packed_idx, packed_qjl, gamma, norm, codebook, qjl_proj, \
            token_block_ids, token_offsets = _make_random_prod_paged_inputs(
                total_bits, head_dim,
            )
        rotation = build_rotation_matrix(head_dim, 99, "cpu", torch.float32)
        rotation_t = rotation.transpose(0, 1).contiguous()

        def _gather(cache):
            return cache[token_block_ids.long(), token_offsets.long()]

        ref = turboquant_decode_prod(
            _gather(packed_idx),
            _gather(packed_qjl),
            _gather(gamma),
            _gather(norm),
            rotation_t,
            codebook,
            qjl_proj,
            total_bits,
            head_dim,
            torch.float32,
        )
        out_rot = tq_dequant_prod_paged_reference_rot(
            packed_idx, packed_qjl, gamma, norm,
            token_block_ids, token_offsets,
            codebook, qjl_proj,
            total_bits, head_dim, torch.float32,
        )
        out = apply_rotation(out_rot, rotation_t).contiguous()

        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    def test_prod_paged_compressed_k_score_matches_dense_score(self):
        total_bits = 3
        head_dim = 32
        num_blocks = 8
        block_size = 4
        batch = 2
        num_heads = 4
        num_kv_heads = 2
        scale = head_dim ** -0.5

        packed_idx, packed_qjl, gamma, norm, codebook, qjl_proj, \
            _, _ = _make_random_prod_paged_inputs(
                total_bits,
                head_dim,
                num_blocks=num_blocks,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
            )
        rotation = build_rotation_matrix(head_dim, 99, "cpu", torch.float32)
        rotation_t = rotation.transpose(0, 1).contiguous()
        block_table = torch.tensor([
            [2, 1, 0],
            [4, 3, 7],
        ], dtype=torch.int32)
        seq_lens = [7, 9]
        max_seq_len = max(seq_lens)
        query = torch.randn(batch, num_heads, head_dim, dtype=torch.float32)
        token_block_ids, token_offsets = build_token_map_from_block_table(
            block_table, seq_lens, block_size,
        )

        dense_rot = tq_dequant_prod_paged_reference_rot(
            packed_idx,
            packed_qjl,
            gamma,
            norm,
            token_block_ids,
            token_offsets,
            codebook,
            qjl_proj,
            total_bits,
            head_dim,
            torch.float32,
        )
        dense_k = apply_rotation(dense_rot, rotation_t).contiguous()
        dense_scores = torch.full(
            (batch, num_heads, max_seq_len),
            float("-inf"),
            dtype=torch.float32,
        )
        q_per_kv = num_heads // num_kv_heads
        token_start = 0
        for batch_idx, seq_len in enumerate(seq_lens):
            batch_k = dense_k[token_start:token_start + seq_len]
            for kv_head in range(num_kv_heads):
                h_start = kv_head * q_per_kv
                h_end = h_start + q_per_kv
                dense_scores[batch_idx, h_start:h_end, :seq_len] = (
                    query[batch_idx, h_start:h_end] @ batch_k[:, kv_head].T
                ) * scale
            token_start += seq_len

        compressed_scores = tq_dequant_prod_paged_k_score_reference(
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
            total_bits,
            head_dim,
            scale=scale,
            max_seq_len=max_seq_len,
        )

        valid_mask = torch.isfinite(dense_scores)
        assert torch.equal(
            torch.isneginf(compressed_scores),
            torch.isneginf(dense_scores),
        )
        assert torch.allclose(
            compressed_scores[valid_mask],
            dense_scores[valid_mask],
            atol=1e-5,
            rtol=1e-5,
        )

        dispatched_scores = tq_dequant_prod_paged_k_score(
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
            total_bits,
            head_dim,
            scale=scale,
            max_seq_len=max_seq_len,
        )
        assert torch.equal(dispatched_scores, compressed_scores)

    @pytest.mark.parametrize("total_bits", [2, 3, 4])
    def test_prod_paged_rot_matches_reference(self, total_bits):
        head_dim = 64
        packed_idx, packed_qjl, gamma, norm, codebook, qjl_proj, \
            token_block_ids, token_offsets = _make_random_prod_paged_inputs(
                total_bits, head_dim,
            )

        ref = tq_dequant_prod_paged_reference_rot(
            packed_idx, packed_qjl, gamma, norm,
            token_block_ids, token_offsets,
            codebook, qjl_proj,
            total_bits, head_dim, torch.float32,
        )
        out = tq_dequant_prod_paged_rot(
            packed_idx, packed_qjl, gamma, norm,
            token_block_ids, token_offsets,
            codebook, qjl_proj,
            total_bits, head_dim, torch.float32,
        )

        assert torch.allclose(out, ref, atol=1e-6, rtol=0)

    def test_prod_cpu_tensors_fallback_when_custom_env_enabled(self, monkeypatch):
        monkeypatch.setenv("VLLM_ASCEND_TQ_USE_CUSTOM_DEQUANT", "1")
        monkeypatch.setenv("VLLM_ASCEND_TQ_USE_CUSTOM_PROD_DEQUANT", "1")

        total_bits = 3
        head_dim = 64
        packed_idx, packed_qjl, gamma, norm, codebook, qjl_proj, \
            token_block_ids, token_offsets = _make_random_prod_paged_inputs(
                total_bits, head_dim,
            )

        ref = tq_dequant_prod_paged_reference_rot(
            packed_idx, packed_qjl, gamma, norm,
            token_block_ids, token_offsets,
            codebook, qjl_proj,
            total_bits, head_dim, torch.float32,
        )
        out = tq_dequant_prod_paged_rot(
            packed_idx, packed_qjl, gamma, norm,
            token_block_ids, token_offsets,
            codebook, qjl_proj,
            total_bits, head_dim, torch.float32,
        )

        assert torch.allclose(out, ref, atol=1e-6, rtol=0)

    @pytest.mark.skipif(not _npu_available(), reason="NPU is not available")
    @pytest.mark.parametrize("total_bits", [2, 3, 4])
    def test_prod_custom_op_matches_reference_on_npu(self, monkeypatch, total_bits):
        monkeypatch.setenv("VLLM_ASCEND_TQ_USE_CUSTOM_DEQUANT", "1")
        monkeypatch.setenv("VLLM_ASCEND_TQ_USE_CUSTOM_PROD_DEQUANT", "1")
        monkeypatch.setenv("VLLM_ASCEND_TQ_CUSTOM_STRICT", "1")

        head_dim = 64
        packed_idx, packed_qjl, gamma, norm, codebook, qjl_proj, \
            token_block_ids, token_offsets = _make_random_prod_paged_inputs(
                total_bits,
                head_dim,
                num_blocks=4,
                block_size=4,
                total_tokens=8,
                device="npu",
            )

        ref = tq_dequant_prod_paged_reference_rot(
            packed_idx, packed_qjl, gamma, norm,
            token_block_ids, token_offsets,
            codebook, qjl_proj,
            total_bits, head_dim, torch.float32,
        )
        out = tq_dequant_prod_paged_rot(
            packed_idx, packed_qjl, gamma, norm,
            token_block_ids, token_offsets,
            codebook, qjl_proj,
            total_bits, head_dim, torch.float32,
        )

        torch.npu.synchronize()
        assert out.shape == ref.shape
        assert torch.allclose(out, ref, atol=1e-3, rtol=0)


# ---------------------------------------------------------------------------
# build_token_map_from_block_table tests
# ---------------------------------------------------------------------------

class TestBuildTokenMapFromBlockTable:

    def test_single_request_exact_one_block(self):
        block_table = torch.tensor([[3, 0]], dtype=torch.int32)
        seq_lens = [4]
        block_size = 4

        ids, offsets = build_token_map_from_block_table(
            block_table, seq_lens, block_size,
        )

        assert ids.tolist() == [3, 3, 3, 3]
        assert offsets.tolist() == [0, 1, 2, 3]

    def test_single_request_multi_block(self):
        block_table = torch.tensor([[5, 7, 0]], dtype=torch.int32)
        seq_lens = [6]
        block_size = 4

        ids, offsets = build_token_map_from_block_table(
            block_table, seq_lens, block_size,
        )

        assert ids.tolist() == [5, 5, 5, 5, 7, 7]
        assert offsets.tolist() == [0, 1, 2, 3, 0, 1]

    def test_multi_request_ragged(self):
        block_table = torch.tensor([
            [11, 12, 0],
            [21, 22, 23],
        ], dtype=torch.int32)
        seq_lens = [5, 9]
        block_size = 4

        ids, offsets = build_token_map_from_block_table(
            block_table, seq_lens, block_size,
        )

        assert ids.shape[0] == 14  # 5 + 9 tokens
        # first request: blocks 11 (pos 0-3), 12 (pos 4)
        assert ids[:5].tolist() == [11, 11, 11, 11, 12]
        assert offsets[:5].tolist() == [0, 1, 2, 3, 0]
        # second request: blocks 21 (pos 0-3), 22 (pos 4-7), 23 (pos 8)
        assert ids[5:].tolist() == [21, 21, 21, 21, 22, 22, 22, 22, 23]
        assert offsets[5:].tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0]

    def test_empty_batch(self):
        block_table = torch.empty((0, 4), dtype=torch.int32)
        ids, offsets = build_token_map_from_block_table(
            block_table, [], 16,
        )
        assert ids.numel() == 0
        assert offsets.numel() == 0

    def test_zero_len_sequence(self):
        block_table = torch.tensor([[0]], dtype=torch.int32)
        ids, offsets = build_token_map_from_block_table(
            block_table, [0], 16,
        )
        assert ids.numel() == 0
        assert offsets.numel() == 0

    def test_non_contiguous_block_table(self):
        # block table may have gaps between blocks
        block_table = torch.tensor([[10, 99, 2, 55]], dtype=torch.int32)
        seq_lens = [14]
        block_size = 4

        ids, offsets = build_token_map_from_block_table(
            block_table, seq_lens, block_size,
        )

        expected_ids = (
            [10] * 4 + [99] * 4 + [2] * 4 + [55] * 2
        )
        expected_offsets = (
            [0, 1, 2, 3] * 3 + [0, 1]
        )
        assert ids.tolist() == expected_ids
        assert offsets.tolist() == expected_offsets

    def test_cached_token_map_reuses_same_block_table(self, monkeypatch):
        monkeypatch.setenv("VLLM_ASCEND_TQ_TOKEN_MAP_CACHE_SIZE", "32")
        block_table = torch.tensor([[5, 7, 0]], dtype=torch.int32)
        seq_lens = [6]
        block_size = 4

        ids1, offsets1 = cached_token_map_from_block_table(
            block_table, seq_lens, block_size,
        )
        ids2, offsets2 = cached_token_map_from_block_table(
            block_table, seq_lens, block_size,
        )

        assert ids1.data_ptr() == ids2.data_ptr()
        assert offsets1.data_ptr() == offsets2.data_ptr()
        assert ids2.tolist() == [5, 5, 5, 5, 7, 7]
        assert offsets2.tolist() == [0, 1, 2, 3, 0, 1]

    def test_cached_token_map_invalidates_on_block_table_update(self, monkeypatch):
        monkeypatch.setenv("VLLM_ASCEND_TQ_TOKEN_MAP_CACHE_SIZE", "32")
        block_table = torch.tensor([[5, 7, 0]], dtype=torch.int32)
        seq_lens = [6]
        block_size = 4

        ids1, _ = cached_token_map_from_block_table(
            block_table, seq_lens, block_size,
        )
        block_table[0, 0] = 9
        ids2, offsets2 = cached_token_map_from_block_table(
            block_table, seq_lens, block_size,
        )

        assert ids1.data_ptr() != ids2.data_ptr()
        assert ids2.tolist() == [9, 9, 9, 9, 7, 7]
        assert offsets2.tolist() == [0, 1, 2, 3, 0, 1]
