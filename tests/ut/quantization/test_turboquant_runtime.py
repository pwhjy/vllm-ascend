import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.methods.turboquant_runtime import (
    build_qjl_projection,
    build_rotation_matrix,
    build_turboquant_codebook,
    get_stage1_bits,
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
