import torch

from tests.ut.base import TestBase
from vllm_ascend.quantization.methods.turboquant_runtime import (
    build_qjl_projection,
    build_rotation_matrix,
    build_turboquant_codebook,
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
        codebook, boundary = build_turboquant_codebook(8, 3, "cpu", torch.float32)
        rot = build_rotation_matrix(8, 1234, "cpu", torch.float32)
        proj = build_qjl_projection(8, 5678, "cpu", torch.float32)
        encoded = turboquant_encode_prod(x, rot, codebook, boundary, proj, bits=3)
        decoded = turboquant_decode_prod(
            encoded["idx"],
            encoded["qjl"],
            encoded["gamma"],
            encoded["norm"],
            rot.transpose(0, 1).contiguous(),
            codebook,
            proj,
            bits=3,
            dim=8,
            target_dtype=torch.float32,
        )
        self.assertEqual(decoded.shape, x.shape)
        self.assertTrue(torch.isfinite(decoded).all())
