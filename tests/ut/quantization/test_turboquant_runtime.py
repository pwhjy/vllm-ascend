import math
import os

import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.methods.turboquant_layout import get_stage1_bits
from vllm_ascend.quantization.methods.turboquant_runtime import (
    build_qjl_projection,
    build_rotation_matrix,
    build_turboquant_codebook,
    monte_carlo_bias_eval,
    pack_bits,
    turboquant_decode_mse,
    turboquant_decode_prod,
    turboquant_encode_mse,
    turboquant_encode_prod,
    unpack_bits,
)


class TestTurboQuantRuntime(TestBase):

    def test_pack_unpack_roundtrip(self):
        indices = torch.tensor(
            [[[0, 1, 2, 3, 4, 5, 6, 7]]],
            dtype=torch.uint8,
        )
        packed = pack_bits(indices, bits=3)
        unpacked = unpack_bits(packed, bits=3, dim=indices.shape[-1])
        self.assertTrue(torch.equal(unpacked, indices))

    def test_pack_unpack_roundtrip_fast_path_1bit(self):
        indices = torch.tensor(
            [[[0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1]]],
            dtype=torch.uint8,
        )
        packed = pack_bits(indices, bits=1)
        unpacked = unpack_bits(packed, bits=1, dim=indices.shape[-1])
        self.assertTrue(torch.equal(unpacked, indices))

    def test_pack_unpack_roundtrip_fast_path_2bit(self):
        indices = torch.tensor(
            [[[0, 1, 2, 3, 1, 0, 2]]],
            dtype=torch.uint8,
        )
        packed = pack_bits(indices, bits=2)
        unpacked = unpack_bits(packed, bits=2, dim=indices.shape[-1])
        self.assertTrue(torch.equal(unpacked, indices))

    def test_pack_unpack_roundtrip_fast_path_3bit(self):
        indices = torch.tensor(
            [[[0, 1, 2, 3, 4, 5, 6, 7]]],
            dtype=torch.uint8,
        )
        packed = pack_bits(indices, bits=3)
        unpacked = unpack_bits(packed, bits=3, dim=indices.shape[-1])
        self.assertTrue(torch.equal(unpacked, indices))

    def test_pack_unpack_roundtrip_3bit_realistic_dim(self):
        dim = 128
        indices = torch.randint(0, 8, (3, 4, dim), dtype=torch.uint8)
        packed = pack_bits(indices, bits=3)
        unpacked = unpack_bits(packed, bits=3, dim=dim)
        self.assertTrue(torch.equal(unpacked, indices))

    def test_pack_unpack_roundtrip_3bit_odd_dim(self):
        for dim in [7, 9, 15, 17, 31, 33, 65, 127]:
            indices = torch.randint(0, 8, (2, dim), dtype=torch.uint8)
            packed = pack_bits(indices, bits=3)
            unpacked = unpack_bits(packed, bits=3, dim=dim)
            self.assertTrue(torch.equal(unpacked, indices),
                            f"3-bit roundtrip failed for dim={dim}")

    def test_pack_unpack_roundtrip_fast_path_4bit(self):
        indices = torch.tensor(
            [[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]]],
            dtype=torch.uint8,
        )
        packed = pack_bits(indices, bits=4)
        unpacked = unpack_bits(packed, bits=4, dim=indices.shape[-1])
        self.assertTrue(torch.equal(unpacked, indices))

    def test_pack_unpack_roundtrip_4bit_realistic_dim(self):
        dim = 128
        indices = torch.randint(0, 16, (3, 4, dim), dtype=torch.uint8)
        packed = pack_bits(indices, bits=4)
        unpacked = unpack_bits(packed, bits=4, dim=dim)
        self.assertTrue(torch.equal(unpacked, indices))

    def test_pack_unpack_roundtrip_4bit_odd_dim(self):
        for dim in [1, 3, 5, 15, 63, 127]:
            indices = torch.randint(0, 16, (2, dim), dtype=torch.uint8)
            packed = pack_bits(indices, bits=4)
            unpacked = unpack_bits(packed, bits=4, dim=dim)
            self.assertTrue(torch.equal(unpacked, indices),
                            f"4-bit roundtrip failed for dim={dim}")

    def test_turboquant_mse_roundtrip_shape(self):
        x = torch.randn(2, 3, 8, dtype=torch.float32)
        codebook, boundary = build_turboquant_codebook(8, 3, "cpu", torch.float32)
        rot = build_rotation_matrix(8, 1234, "cpu", torch.float32)
        encoded = turboquant_encode_mse(x, rot, codebook, boundary, bits=3)
        decoded = turboquant_decode_mse(
            encoded["idx"],
            encoded["norm"],
            rot.transpose(0, 1).contiguous(),
            codebook,
            bits=3,
            dim=8,
            target_dtype=torch.float32,
        )
        self.assertEqual(decoded.shape, x.shape)
        self.assertTrue(torch.isfinite(decoded).all())

    def test_turboquant_prod_roundtrip_shape(self):
        x = torch.randn(2, 2, 8, dtype=torch.float32)
        codebook, boundary = build_turboquant_codebook(8, get_stage1_bits(3, "prod"), "cpu", torch.float32)
        rot = build_rotation_matrix(8, 1234, "cpu", torch.float32)
        proj = build_qjl_projection(8, 5678, "cpu", torch.float32)
        encoded = turboquant_encode_prod(x, rot, codebook, boundary, proj, total_bits=3)
        decoded = turboquant_decode_prod(
            encoded["idx"],
            encoded["qjl"],
            encoded["gamma"],
            encoded["norm"],
            rot.transpose(0, 1).contiguous(),
            codebook,
            proj,
            total_bits=3,
            dim=8,
            target_dtype=torch.float32,
        )
        self.assertEqual(decoded.shape, x.shape)
        self.assertTrue(torch.isfinite(decoded).all())

    def test_codebook_depends_on_head_dim(self):
        codebook_8, _ = build_turboquant_codebook(8, 3, "cpu", torch.float32)
        codebook_64, _ = build_turboquant_codebook(64, 3, "cpu", torch.float32)
        self.assertFalse(torch.allclose(codebook_8, codebook_64))

    def test_prod_budget_is_total_bits_minus_one_plus_one(self):
        x = torch.randn(1, 1, 8, dtype=torch.float32)
        total_bits = 4
        stage1_bits = get_stage1_bits(total_bits, "prod")
        codebook, boundary = build_turboquant_codebook(8, stage1_bits, "cpu", torch.float32)
        rot = build_rotation_matrix(8, 1234, "cpu", torch.float32)
        proj = build_qjl_projection(8, 5678, "cpu", torch.float32)
        encoded = turboquant_encode_prod(x, rot, codebook, boundary, proj, total_bits=total_bits)
        self.assertEqual(encoded["idx"].shape[-1], (8 * stage1_bits + 7) // 8)
        self.assertEqual(encoded["qjl"].shape[-1], (8 + 7) // 8)

    def test_qjl_projection_distribution_no_row_normalization(self):
        proj = build_qjl_projection(32, 1234, "cpu", torch.float32)
        row_norms = proj.norm(dim=1)
        self.assertGreater(torch.max(torch.abs(row_norms - 1.0)).item(), 0.1)

    def test_structured_hadamard_transform_matrices(self):
        old_mode = os.environ.get("VLLM_ASCEND_TQ_STRUCTURED_TRANSFORM")
        os.environ["VLLM_ASCEND_TQ_STRUCTURED_TRANSFORM"] = "hadamard"
        try:
            rot = build_rotation_matrix(8, 1234, "cpu", torch.float32)
            proj = build_qjl_projection(8, 5678, "cpu", torch.float32)
        finally:
            if old_mode is None:
                os.environ.pop("VLLM_ASCEND_TQ_STRUCTURED_TRANSFORM", None)
            else:
                os.environ["VLLM_ASCEND_TQ_STRUCTURED_TRANSFORM"] = old_mode

        eye = torch.eye(8, dtype=torch.float32)
        self.assertTrue(torch.allclose(rot.T @ rot, eye, atol=1e-6))
        self.assertTrue(torch.equal(proj.abs(), torch.ones_like(proj)))
        self.assertTrue(
            torch.allclose(
                proj.norm(dim=1),
                torch.full((8,), math.sqrt(8), dtype=torch.float32),
            )
        )

    # ========================
    # Numerical correctness tests
    # ========================

    def test_mse_roundtrip_error_reasonable(self):
        head_dim = 64
        n_vec = 512
        bits = 3
        x = torch.randn(n_vec, head_dim, dtype=torch.float32)
        codebook, boundary = build_turboquant_codebook(head_dim, bits, "cpu", torch.float32)
        rot = build_rotation_matrix(head_dim, 1234, "cpu", torch.float32)
        encoded = turboquant_encode_mse(x, rot, codebook, boundary, bits=bits)
        decoded = turboquant_decode_mse(
            encoded["idx"], encoded["norm"],
            rot.transpose(0, 1).contiguous(), codebook,
            bits=bits, dim=head_dim, target_dtype=torch.float32,
        )
        mse = torch.mean((decoded - x) ** 2).item()
        # MSE should be nonzero but bounded (quantization introduces error)
        self.assertGreater(mse, 0.0)
        self.assertLess(mse, 0.5, f"MSE={mse:.4f} too large for {bits}-bit quantization")

    def test_mse_error_decreases_with_more_bits(self):
        head_dim = 64
        n_vec = 512
        x = torch.randn(n_vec, head_dim, dtype=torch.float32)
        mse_vals = {}
        for bits in [1, 2, 3, 4]:
            codebook, boundary = build_turboquant_codebook(head_dim, bits, "cpu", torch.float32)
            rot = build_rotation_matrix(head_dim, 1234, "cpu", torch.float32)
            encoded = turboquant_encode_mse(x, rot, codebook, boundary, bits=bits)
            decoded = turboquant_decode_mse(
                encoded["idx"], encoded["norm"],
                rot.transpose(0, 1).contiguous(), codebook,
                bits=bits, dim=head_dim, target_dtype=torch.float32,
            )
            mse_vals[bits] = torch.mean((decoded - x) ** 2).item()
        self.assertGreater(mse_vals[1], mse_vals[2])
        self.assertGreater(mse_vals[2], mse_vals[3])
        self.assertGreater(mse_vals[3], mse_vals[4])

    def test_mse_inner_product_bias_expected_nonzero(self):
        head_dim = 64
        bits = 3
        x = torch.randn(256, head_dim, dtype=torch.float32)
        y = torch.randn(256, head_dim, dtype=torch.float32)
        codebook, boundary = build_turboquant_codebook(head_dim, bits, "cpu", torch.float32)
        rot = build_rotation_matrix(head_dim, 1234, "cpu", torch.float32)
        encoded = turboquant_encode_mse(x, rot, codebook, boundary, bits=bits)
        decoded = turboquant_decode_mse(
            encoded["idx"], encoded["norm"],
            rot.transpose(0, 1).contiguous(), codebook,
            bits=bits, dim=head_dim, target_dtype=torch.float32,
        )
        original_ip = torch.sum(x * y, dim=-1)
        decoded_ip = torch.sum(decoded * y, dim=-1)
        bias = (decoded_ip - original_ip).mean().item()
        # MSE-only has inner product bias (no QJL correction)
        self.assertNotEqual(bias, 0.0)

    def test_prod_inner_product_bias_near_zero(self):
        stats = monte_carlo_bias_eval(64, 3, num_samples=1024, seed=42)
        normalized_bias = stats["bias"] / (stats["std"] + 1e-6)
        self.assertLess(abs(normalized_bias), 1.0,
                        f"Prod inner product bias too large: bias={stats['bias']:.6f}, std={stats['std']:.6f}")

    def test_prod_inner_product_bias_multiple_bit_configs(self):
        for head_dim, total_bits in [(64, 2), (64, 3), (64, 4), (128, 3), (128, 4)]:
            with self.subTest(head_dim=head_dim, total_bits=total_bits):
                stats = monte_carlo_bias_eval(head_dim, total_bits, num_samples=1024, seed=42)
                normalized_bias = abs(stats["bias"]) / (stats["std"] + 1e-6)
                self.assertLess(normalized_bias, 1.5,
                                f"head_dim={head_dim} total_bits={total_bits}: bias={stats['bias']:.6f}")

    def test_decode_close_to_reference_for_seeded_vectors(self):
        head_dim = 32
        total_bits = 3
        n_vec = 128
        generator = torch.Generator(device="cpu")
        generator.manual_seed(42)
        x = torch.randn(n_vec, head_dim, generator=generator, dtype=torch.float32, device="cpu")
        stage1_bits = get_stage1_bits(total_bits, "prod")
        codebook, boundary = build_turboquant_codebook(head_dim, stage1_bits, "cpu", torch.float32)
        rot = build_rotation_matrix(head_dim, 1234, "cpu", torch.float32)
        proj = build_qjl_projection(head_dim, 5678, "cpu", torch.float32)
        encoded = turboquant_encode_prod(x, rot, codebook, boundary, proj, total_bits=total_bits)
        decoded = turboquant_decode_prod(
            encoded["idx"], encoded["qjl"], encoded["gamma"], encoded["norm"],
            rot.transpose(0, 1).contiguous(), codebook, proj,
            total_bits=total_bits, dim=head_dim, target_dtype=torch.float32,
        )
        mse = torch.mean((decoded - x) ** 2).item()
        self.assertLess(mse, 0.3, f"MSE={mse:.4f} too large for seeded decode test")
