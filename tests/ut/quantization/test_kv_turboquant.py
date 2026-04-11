from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

from tests.ut.base import TestBase
from vllm_ascend.quantization.modelslim_config import AscendModelSlimConfig


class TestAscendTurboQuantKVCacheAttentionMethod(TestBase):

    def _make_layer(self):
        layer = nn.Module()
        layer.impl = MagicMock()
        layer.impl.head_size = 8
        return layer

    def test_create_weights_sets_runtime_fields(self):
        from vllm_ascend.quantization.methods.kv_turboquant import AscendTurboQuantKVCacheAttentionMethod

        method = AscendTurboQuantKVCacheAttentionMethod(
            {
                "kv_cache_type": "TurboQuant",
                "turboquant": {
                    "k_variant": "prod",
                    "v_variant": "mse",
                    "k_total_bits": 3,
                    "v_total_bits": 2,
                },
            },
            "model.layers.0.self_attn.attn",
        )
        layer = self._make_layer()
        method.create_weights(layer)
        self.assertTrue(layer.turboquant_enabled)
        self.assertEqual(layer.kv_cache_torch_dtype, torch.int8)
        self.assertEqual(layer.tq_k_variant, "prod")
        self.assertEqual(layer.tq_v_variant, "mse")
        self.assertEqual(layer.tq_k_total_bits, 3)
        self.assertEqual(layer.tq_k_stage1_bits, 2)
        self.assertEqual(layer.tq_v_total_bits, 2)
        self.assertEqual(layer.tq_v_stage1_bits, 2)
        self.assertIsInstance(layer.k_codebook, torch.Tensor)
        self.assertIsInstance(layer.k_qjl_proj, torch.Tensor)

    def test_process_weights_after_loading_marks_runtime_unprepared(self):
        from vllm_ascend.quantization.methods.kv_turboquant import AscendTurboQuantKVCacheAttentionMethod

        method = AscendTurboQuantKVCacheAttentionMethod({"kv_cache_type": "TurboQuant"}, "model.layers.0.self_attn.attn")
        layer = self._make_layer()
        method.create_weights(layer)
        method.process_weights_after_loading(layer)
        self.assertFalse(layer.tq_runtime_prepared)

    def test_create_weights_uses_head_size_v_when_available(self):
        from vllm_ascend.quantization.methods.kv_turboquant import AscendTurboQuantKVCacheAttentionMethod

        method = AscendTurboQuantKVCacheAttentionMethod({"kv_cache_type": "TurboQuant"}, "model.layers.0.self_attn.attn")
        layer = self._make_layer()
        layer.impl.head_size_v = 4
        method.create_weights(layer)
        self.assertEqual(layer.tq_head_size_v, 4)
        self.assertEqual(tuple(layer.v_rot.shape), (4, 4))

    def test_rejects_unsupported_v_variant(self):
        from vllm_ascend.quantization.methods.kv_turboquant import AscendTurboQuantKVCacheAttentionMethod

        with self.assertRaises(ValueError):
            AscendTurboQuantKVCacheAttentionMethod(
                {"kv_cache_type": "TurboQuant", "turboquant": {"v_variant": "prod"}},
                "model.layers.0.self_attn.attn",
            )

    def test_rejects_unsupported_rotation_scope(self):
        from vllm_ascend.quantization.methods.kv_turboquant import AscendTurboQuantKVCacheAttentionMethod

        with self.assertRaises(ValueError):
            AscendTurboQuantKVCacheAttentionMethod(
                {"kv_cache_type": "TurboQuant", "turboquant": {"rotation_scope": "per_head"}},
                "model.layers.0.self_attn.attn",
            )

    def test_rejects_outlier_channels_without_artifact_support(self):
        from vllm_ascend.quantization.methods.kv_turboquant import AscendTurboQuantKVCacheAttentionMethod

        with self.assertRaises(ValueError):
            AscendTurboQuantKVCacheAttentionMethod(
                {"kv_cache_type": "TurboQuant", "turboquant": {"outlier_channels": 16}},
                "model.layers.0.self_attn.attn",
            )

    def test_rejects_unsupported_dequant_mode(self):
        from vllm_ascend.quantization.methods.kv_turboquant import AscendTurboQuantKVCacheAttentionMethod

        with self.assertRaises(ValueError):
            AscendTurboQuantKVCacheAttentionMethod(
                {"kv_cache_type": "TurboQuant", "turboquant": {"dequant_mode": "fused_turboquant_attention"}},
                "model.layers.0.self_attn.attn",
            )


class TestModelSlimTurboQuantConfig(TestBase):

    def test_turboquant_metadata_and_layer_filter(self):
        config = AscendModelSlimConfig(
            {
                "kv_cache_type": "TurboQuant",
                "turboquant_layers": [1, 3],
                "turboquant": {"k_variant": "prod", "v_variant": "mse"},
            }
        )
        self.assertTrue(config.enable_turboquant)
        self.assertTrue(config.is_turboquant_layer("model.layers.1.self_attn.attn"))
        self.assertFalse(config.is_turboquant_layer("model.layers.2.self_attn.attn"))

    def test_get_quant_method_for_turboquant_attention(self):
        config = AscendModelSlimConfig(
            {
                "kv_cache_type": "TurboQuant",
                "turboquant_layers": [0],
                "turboquant": {"k_variant": "prod", "v_variant": "mse"},
            }
        )
        attention_layer = MagicMock(spec=AttentionLayerBase)
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.hf_config.model_type = None
        with patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_vllm_config), \
            patch("vllm_ascend.quantization.method_adapters.AscendKVCacheMethod", return_value=MagicMock()) as mock_kvcache:
            method = config.get_quant_method(attention_layer, "model.layers.0.self_attn.attn")
            self.assertIs(method, mock_kvcache.return_value)
            args, _ = mock_kvcache.call_args
            from vllm_ascend.quantization.methods.kv_turboquant import AscendTurboQuantKVCacheAttentionMethod
            self.assertIsInstance(args[0], AscendTurboQuantKVCacheAttentionMethod)

    def test_turboquant_linear_without_weight_quant_desc_falls_back_to_unquantized(self):
        from vllm.model_executor.layers.linear import LinearBase
        from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod

        config = AscendModelSlimConfig(
            {
                "kv_cache_type": "TurboQuant",
                "turboquant": {"k_variant": "prod", "v_variant": "mse"},
            }
        )
        linear_layer = MagicMock(spec=LinearBase)
        mock_vllm_config = MagicMock()
        mock_vllm_config.model_config.hf_config.model_type = None
        with patch("vllm_ascend.quantization.modelslim_config.get_current_vllm_config", return_value=mock_vllm_config):
            method = config.get_quant_method(linear_layer, "model.layers.0.mlp.down_proj")
            self.assertIsInstance(method, AscendUnquantizedLinearMethod)
