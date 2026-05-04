import torch

from tests.ut.base import TestBase
from vllm.v1.kv_cache_interface import KVQuantMode

from vllm_ascend.quantization.methods.turboquant_layout import TurboQuantAttentionSpec


class TestTurboQuantLayout(TestBase):

    def _make_spec(self, **overrides):
        defaults = dict(
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
        defaults.update(overrides)
        return TurboQuantAttentionSpec(**defaults)

    def test_prod_uses_stage1_bits_for_k_idx_bytes(self):
        spec = self._make_spec()
        self.assertEqual(spec.k_stage1_bits, 3)
        self.assertEqual(spec.k_idx_bytes_per_vector, (8 * 3 + 7) // 8)
        self.assertEqual(spec.k_qjl_bytes_per_vector, (8 + 7) // 8)

    def test_v_idx_bytes_use_head_size_v(self):
        spec = self._make_spec(head_size_v=4)
        self.assertEqual(spec.v_idx_bytes_per_vector, (4 * 2 + 7) // 8)

    def test_memory_stats_k_real_bits_includes_sidecar(self):
        spec = self._make_spec(k_total_bits=3, v_total_bits=2, k_variant="prod")
        stats = spec.get_memory_stats()
        # k_real_bits should be > k_logical_bits due to norm+gamma+qjl overhead
        self.assertGreater(stats["k_real_bits_per_channel"], stats["k_logical_bits"])
        self.assertGreater(stats["v_real_bits_per_channel"], stats["v_logical_bits"])

    def test_compression_ratio_less_than_one(self):
        spec = self._make_spec(k_total_bits=3, v_total_bits=2)
        ratio = spec.effective_compression_ratio
        self.assertLess(ratio, 1.0)
        self.assertGreater(ratio, 0.0)

    def test_scalar_dtype_affects_memory_stats(self):
        spec_f32 = self._make_spec(scalar_dtype=torch.float32)
        spec_f16 = self._make_spec(scalar_dtype=torch.float16)
        self.assertGreater(
            spec_f32.real_page_size_bytes,
            spec_f16.real_page_size_bytes,
        )
        self.assertGreater(
            spec_f32.k_real_bits_per_channel,
            spec_f16.k_real_bits_per_channel,
        )

    def test_memory_stats_output_is_json_serializable(self):
        import json

        spec = self._make_spec()
        stats = spec.get_memory_stats()
        json.dumps(stats)  # should not raise
