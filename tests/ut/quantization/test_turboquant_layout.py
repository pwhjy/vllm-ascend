import torch

from tests.ut.base import TestBase
from vllm.v1.kv_cache_interface import KVQuantMode

from vllm_ascend.quantization.methods.turboquant_layout import TurboQuantAttentionSpec


class TestTurboQuantLayout(TestBase):

    def test_prod_uses_stage1_bits_for_k_idx_bytes(self):
        spec = TurboQuantAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=8,
            head_size_v=8,
            dtype=torch.int8,
            kv_quant_mode=KVQuantMode.NONE,
            k_total_bits=4,
            v_total_bits=2,
            k_variant="prod",
            v_variant="mse",
            use_k_qjl=True,
        )
        self.assertEqual(spec.k_stage1_bits, 3)
        self.assertEqual(spec.k_idx_bytes_per_vector, (8 * 3 + 7) // 8)
        self.assertEqual(spec.k_qjl_bytes_per_vector, (8 + 7) // 8)

    def test_v_idx_bytes_use_head_size_v(self):
        spec = TurboQuantAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=8,
            head_size_v=4,
            dtype=torch.int8,
            kv_quant_mode=KVQuantMode.NONE,
            k_total_bits=3,
            v_total_bits=2,
            k_variant="prod",
            v_variant="mse",
            use_k_qjl=True,
        )
        self.assertEqual(spec.v_idx_bytes_per_vector, (4 * 2 + 7) // 8)
