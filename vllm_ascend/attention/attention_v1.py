#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from dataclasses import dataclass
from enum import Enum
import math
import os
import time

import torch
import torch_npu
import vllm.envs as envs_vllm
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (  # type: ignore
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
)
from vllm.v1.attention.backends.registry import (  # type: ignore
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import AttentionSpec, CrossAttentionSpec

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
from vllm_ascend.attention.context_parallel.common_cp import AscendMetadataForDecode, AscendMetadataForPrefill
from vllm_ascend.attention.utils import (
    AscendCommonAttentionMetadata,
    enable_cp,
    split_decodes_and_prefills,
    using_paged_attention,
)
from vllm_ascend.compilation.acl_graph import (
    get_draft_graph_params,
    get_graph_params,
    update_draft_graph_params_workspaces,
    update_graph_params_workspaces,
)
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.ops.flashcomm2_oshard_manager import flashcomm2_oshard_manager
from vllm_ascend.ops.turboquant.dequant import (
    attention_debug_compare_enabled,
    cached_token_map_from_block_table,
    custom_dequant_enabled,
    dequant_debug_compare_enabled,
    fused_attention_custom_enabled,
    tq_prod_mse_paged_attention,
    tq_dequant_mse_paged_rot,
    tq_dequant_mse_paged_scaled_rot,
)
from vllm_ascend.ops.turboquant.fused import (
    compressed_decode_current_enabled,
    decode_compressed_full_cache_enabled,
    fused_kv_update_attention_enabled,
    tq_decode_history_to_dense,
    tq_encode_kv_to_paged_cache,
    tq_fused_kv_update_attention,
    tq_prod_mse_history_current_decode_attention,
)
from vllm_ascend.quantization.methods.turboquant_layout import TurboQuantAttentionSpec
from vllm_ascend.quantization.methods.turboquant_runtime import (
    _maybe_sync_for_profile,
    _record_tq_profile,
    apply_rotation,
    turboquant_decode_mse,
    turboquant_decode_prod,
    turboquant_encode_mse,
    turboquant_encode_prod,
)
from vllm_ascend.utils import weak_ref_tensors

# default max value of sliding window size
SWA_INT_MAX = 2147483647


@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        # HACK(Ronald1995): vllm `initialize_kv_cache` method in model runner v2 make
        # attention name assertion, we just set name to FLASH_ATTN to avoid assertion error.
        # rectify this when vllm disable the assertion.
        return "CUSTOM" if not envs_vllm.VLLM_USE_V2_MODEL_RUNNER else "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["AscendAttentionBackendImpl"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPImpl

            return AscendAttentionCPImpl
        return AscendAttentionBackendImpl

    @staticmethod
    def get_builder_cls() -> type["AscendAttentionMetadataBuilder"]:
        if enable_cp():
            from vllm_ascend.attention.context_parallel.attention_cp import AscendAttentionCPMetadataBuilder

            return AscendAttentionCPMetadataBuilder
        return AscendAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_type: str = "",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: list[torch.Tensor],
        dst_kv_cache: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        if isinstance(src_kv_cache, dict) and isinstance(dst_kv_cache, dict):
            src_indices = src_to_dst[:, 0]
            dst_indices = src_to_dst[:, 1]
            for key in src_kv_cache:
                dst_kv_cache[key][dst_indices] = src_kv_cache[key][src_indices].to(dst_kv_cache[key].device)
            return
        src_key_cache, src_value_cache = src_kv_cache[0], src_kv_cache[1]
        dst_key_cache, dst_value_cache = dst_kv_cache[0], dst_kv_cache[1]
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]

        dst_key_cache[dst_indices] = src_key_cache[src_indices].to(dst_key_cache.device)
        dst_value_cache[dst_indices] = src_value_cache[src_indices].to(dst_key_cache.device)

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]

        for kv_cache in kv_caches:
            if isinstance(kv_cache, dict):
                for value in kv_cache.values():
                    value[dst_indices] = value[src_indices]
                continue
            key_caches = kv_cache[0]
            value_caches = kv_cache[1]
            key_caches[dst_indices] = key_caches[src_indices]
            value_caches[dst_indices] = value_caches[src_indices]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]


class AscendAttentionState(Enum):
    PrefillNoCache = 0
    PrefillCacheHit = 1
    DecodeOnly = 2
    ChunkedPrefill = 3
    SpecDecoding = 4


@dataclass
class AscendMetadata:
    """
    Per-layer attention metadata for Ascend FlashAttention backend.

    Contains attention masks, token counts, sequence lengths and KV cache
    related properties for attention computation.
    """

    # **************************** Basic Properties ************************** #
    attn_mask: torch.Tensor | None = None
    # Current state of this attention run.
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    # Number of tokens excluding padding.
    num_actual_tokens_pcp_padded: int = 0
    num_actual_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0
    num_decodes_flatten: int = 0

    # The sequence length per sequence. Sequence length means the computed
    # tokens + new tokens (is None if it is a decoding).
    # (batch_size,)
    # TODO(Angazenn): The following parameters are quite redundant and
    # contains similar information (such as seq_lens seq_lens_list). We
    # should simplified these parameters once attention schema in vLLM-Ascend
    # is unified.
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    seq_lens_list: list[int] = None  # type: ignore
    actual_seq_lengths_q: list[int] = None  # type: ignore

    query_start_loc: torch.Tensor = None
    # Maximum query length in the batch (None for decoding).
    max_query_len: int | None = None

    # ********************** KV Cache Related Properties ********************* #
    # Block addresses per sequence (Seq id -> list of physical block).
    # (batch_size, max_blocks_per_seq)
    block_tables: torch.Tensor = None

    # The indices of the token slots that input tokens will be stored into.
    # E.g., if `slot_mapping` is [35, 2, 17] and the block size is 16, the
    # three tokens are stored in the 3rd slot in block 2, 2nd slot in block 0,
    # and 1st slot in block 1, respectively.
    # (num_tokens,)
    slot_mapping: torch.Tensor = None
    # pcp
    prefill: AscendMetadataForPrefill | None = None
    # dcp
    decode_meta: AscendMetadataForDecode | None = None

    causal: bool = True
    # runner_type in model_config.
    model_runner_type: str = ""
    # prefill reshape_and_cache event
    reshape_cache_event: torch.npu.Event = None

    # sliding window attention mask
    swa_mask: torch.Tensor | None = None


class AscendAttentionMetadataBuilder(AttentionMetadataBuilder[AscendMetadata]):
    """
    Builder for constructing AscendMetadata from CommonAttentionMetadata.

    Handles attention mask generation and metadata preparation for
    Ascend FlashAttention backend.
    """

    # Does this backend/builder reorder the batch?
    # If not, set this to None. Otherwise set it to the query
    # length that will be pulled into the front of the batch.
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.compilation_config = vllm_config.compilation_config
        self.device = device
        self.max_num_blocks_per_req = cdiv(
            self.model_config.max_model_len, AscendAttentionBackend.get_supported_kernel_block_sizes()[0]
        )

        self.speculative_config = vllm_config.speculative_config
        self.decode_threshold = 1
        if self.speculative_config:
            spec_token_num = self.speculative_config.num_speculative_tokens
            self.decode_threshold += spec_token_num
            assert self.decode_threshold <= 16, (
                f"decode_threshold exceeded \
                npu_fused_infer_attention_score TND layout's limit of 16, \
                got {self.decode_threshold}"
            )

        self.reorder_batch_threshold = self.decode_threshold

        scheduler_config = vllm_config.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.enable_chunked_prefill
        self.attn_mask_builder = AttentionMaskBuilder(self.device)

    @classmethod
    def get_cudagraph_support(
        cls: type["AscendAttentionMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        # Explicit override in case the underlying builder specialized this getter.
        # @override omitted only because of mypy limitation due to type variable.
        return AttentionCGSupport.ALWAYS

    def reorder_batch(self, input_batch, scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: AscendCommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AscendMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1]

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata, decode_threshold=self.decode_threshold
        )

        block_table = common_attn_metadata.block_table_tensor
        # Prefer _seq_lens_cpu (always available, updated during draft
        # iterations) over seq_lens_cpu (None in async spec decode mode).
        if common_attn_metadata._seq_lens_cpu is not None:
            seq_lens = common_attn_metadata._seq_lens_cpu[:num_reqs]
        elif common_attn_metadata.seq_lens_cpu is not None:
            seq_lens = common_attn_metadata.seq_lens_cpu[:num_reqs]
        else:
            seq_lens = common_attn_metadata.seq_lens[:num_reqs].to("cpu")

        slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens]
        # this slot_mapping override doesn't work since vllm will override it again. We should fix it vllm.
        # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
        if isinstance(self.kv_cache_spec, CrossAttentionSpec):
            seq_lens = common_attn_metadata.seq_lens
            slot_mapping = common_attn_metadata.slot_mapping.to(torch.int32)
        elif self.speculative_config and self.speculative_config.parallel_drafting:
            seq_lens = common_attn_metadata.seq_lens

        attn_state = common_attn_metadata.attn_state

        # Get attn_mask and swa_mask from singleton AttentionMaskBuilder
        attn_mask = self.attn_mask_builder.get_attention_mask(self.model_config)

        swa_mask = None
        is_swa = hasattr(self.model_config.hf_text_config, "sliding_window")
        if self.model_config is not None and is_swa:
            swa_mask = self.attn_mask_builder.get_swa_mask(
                self.model_config.dtype, self.model_config.hf_text_config.sliding_window
            )

        # TODO: Yet another unnecessary H2D while we already have a query_start_loc on device
        query_start_loc = query_start_loc_cpu.pin_memory().to(self.device, non_blocking=True)

        attn_metadata = AscendMetadata(
            num_actual_tokens=num_actual_tokens,
            num_decode_tokens=num_decode_tokens,
            block_tables=block_table,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            seq_lens_list=seq_lens.tolist(),
            max_query_len=common_attn_metadata.max_query_len,
            actual_seq_lengths_q=query_start_loc_cpu[1:].tolist(),
            slot_mapping=slot_mapping,
            attn_mask=attn_mask,
            swa_mask=swa_mask,
            attn_state=attn_state,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            causal=common_attn_metadata.causal,
            model_runner_type=self.model_config.runner_type,
        )
        return attn_metadata

    def build_for_graph_capture(
        self,
        common_attn_metadata: AscendCommonAttentionMetadata,
        attn_state: AscendAttentionState = AscendAttentionState.DecodeOnly,
    ):
        if attn_state in (
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.ChunkedPrefill,
            AscendAttentionState.SpecDecoding,
        ):
            attn_metadata = self.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise NotImplementedError(
                "Currently we only support building dummy metadata for DecodeOnly and ChunkedPrefill state"
            )

        attn_metadata.attn_state = attn_state
        return attn_metadata


class AscendAttentionBackendImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        sinks: torch.Tensor = None,
        **kwargs,
    ) -> None:
        self.vllm_config = get_current_vllm_config()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.hidden_size = self.num_heads * self.head_size
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32, device="npu")
        self.alibi_slopes = alibi_slopes
        self.attn_type = attn_type

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.key_cache = None
        self.value_cache = None
        self.is_kv_producer = (
            self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.is_kv_producer
        )
        self.sinks = sinks

    @staticmethod
    def update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config=None,
        num_dcp_pcp_tokens=None,
        draft_attn_metadatas=None,
    ):
        if using_paged_attention(num_tokens, vllm_config):
            # Paged Attention update logic
            if _EXTRA_CTX.is_draft_model:
                graph_params = get_draft_graph_params()
            else:
                graph_params = get_graph_params()
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    forward_context.attn_metadata,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    (
                        query,
                        key_cache,
                        value_cache,
                        num_kv_heads,
                        num_heads,
                        scale,
                        block_table,
                        seq_lens,
                        output,
                    ) = param
                    seq_lens = forward_context.attn_metadata[key].seq_lens

                    workspace = torch_npu._npu_paged_attention_get_workspace(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                    )
                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu._npu_paged_attention(
                        query=query,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_kv_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale_value=scale,
                        block_table=block_table,
                        context_lens=seq_lens,
                        out=output,
                        workspace=workspace,
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    event.record(update_stream)
        else:
            # FIA update logic
            if _EXTRA_CTX.is_draft_model:
                graph_params = get_draft_graph_params()
                attn_metadata = draft_attn_metadatas
                attn_keys = list(attn_metadata[0].keys())
            else:
                graph_params = get_graph_params()
                attn_metadata = forward_context.attn_metadata
                attn_keys = list(attn_metadata.keys())
            # For Qwen3-next, since the kv_cache_config has already categorized
            # linear_attn and self_attn, the attn_metadata is first arranged with
            # self_attn followed by linear_attn. Therefore, using zip directly
            # filters out the update operations for linear_attn.
            # TODO: We use a new variable `attn_keys` to ensure the loop count is
            # correct after get by `zip` because of the new structure of the attn_metadata
            # when running with the merged full eagle-graph. Should check it with Qwen3-next.
            num_layers = len(attn_keys)
            if num_layers == 0:
                return
            if _EXTRA_CTX.is_draft_model:
                attn_keys = attn_keys * (len(graph_params.attn_params[num_tokens]) // num_layers)
            attn_count = 0
            with torch.npu.stream(update_stream):
                for key, param, handle, event in zip(
                    attn_keys,
                    graph_params.attn_params[num_tokens],
                    graph_params.handles[num_tokens],
                    graph_params.events[num_tokens],
                ):
                    (
                        query,
                        key_cache,
                        value,
                        block_tables,
                        attn_mask,
                        block_size,
                        seq_lens,
                        query_start_loc,
                        num_kv_heads,
                        num_heads,
                        scale,
                        attn_output,
                        softmax_lse,
                    ) = param

                    if _EXTRA_CTX.is_draft_model:
                        draft_step = attn_count // num_layers
                        seq_lens = attn_metadata[draft_step][key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[draft_step][key].actual_seq_lengths_q
                        block_tables = attn_metadata[draft_step][key].block_tables
                        attn_count = attn_count + 1
                    else:
                        seq_lens = attn_metadata[key].seq_lens_list
                        actual_seq_lengths_q = attn_metadata[key].actual_seq_lengths_q
                        block_tables = attn_metadata[key].block_tables

                    torch.npu.graph_task_update_begin(update_stream, handle)
                    torch_npu.npu_fused_infer_attention_score.out(
                        query=query,
                        key=key_cache,
                        value=value,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_lengths=actual_seq_lengths_q,
                        actual_seq_lengths_kv=seq_lens,
                        num_key_value_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale=scale,
                        sparse_mode=3,
                        workspace=graph_params.workspaces.get(num_tokens),
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)

                    event.record(update_stream)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        if flashcomm2_oshard_manager.flashcomm2_oshard_enable():
            flashcomm2_oshard_manager.post_process_after_loading()

    def full_graph_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)

        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        if _EXTRA_CTX.is_draft_model:
            graph_params = get_draft_graph_params()
        else:
            graph_params = get_graph_params()
        actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q
        # Prepare tensors for attention output
        # TODO: Refactor this to step-level instead of layer-level

        # Get workspace from cache or calculate it if not present.
        workspace = graph_params.workspaces.get(num_tokens)
        softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        if workspace is None:
            workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                sparse_mode=3,
                scale=self.scale,
            )
            if _EXTRA_CTX.is_draft_model:
                update_draft_graph_params_workspaces(num_tokens, workspace)
            else:
                update_graph_params_workspaces(num_tokens, workspace)

        # Handle graph capturing mode
        stream = torch_npu.npu.current_stream()

        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        graph_params.events[num_tokens].append(event)
        graph_params.attn_params[num_tokens].append(
            (
                weak_ref_tensors(query),
                weak_ref_tensors(key),
                weak_ref_tensors(value),
                weak_ref_tensors(block_table),
                weak_ref_tensors(attn_metadata.attn_mask),
                block_size,
                actual_seq_lengths_kv,
                actual_seq_lengths_q,
                self.num_kv_heads,
                self.num_heads,
                self.scale,
                weak_ref_tensors(output),
                weak_ref_tensors(softmax_lse),
            )
        )

        torch.npu.graph_task_group_begin(stream)
        torch_npu.npu_fused_infer_attention_score.out(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
            workspace=workspace,
            out=[output, softmax_lse],
        )

        output = output.view(num_tokens, self.num_heads, self.head_size)

        handle = torch.npu.graph_task_group_end(stream)
        graph_params.handles[num_tokens].append(handle)
        return output, num_tokens

    def full_graph_pa(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ):
        graph_params = get_graph_params()
        num_tokens = query.shape[0]
        if _EXTRA_CTX.capturing:
            # Get workspace from cache or calculate it if not present.
            workspace = graph_params.workspaces.get(num_tokens)
            if workspace is None:
                workspace = torch_npu._npu_paged_attention_get_workspace(
                    query=query,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    num_kv_heads=self.num_kv_heads,
                    num_heads=self.num_heads,
                    scale_value=self.scale,
                    block_table=attn_metadata.block_tables,
                    context_lens=attn_metadata.seq_lens,
                    out=output,
                )
                update_graph_params_workspaces(num_tokens, workspace)

            # Handle graph capturing mode
            stream = torch_npu.npu.current_stream()

            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)
            graph_params.attn_params[num_tokens].append(
                (
                    weak_ref_tensors(query),
                    weak_ref_tensors(self.key_cache),
                    weak_ref_tensors(self.value_cache),
                    self.num_kv_heads,
                    self.num_heads,
                    self.scale,
                    attn_metadata.block_tables,
                    attn_metadata.seq_lens,
                    weak_ref_tensors(output),
                )
            )

            torch.npu.graph_task_group_begin(stream)
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                num_kv_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale_value=self.scale,
                block_table=attn_metadata.block_tables,
                context_lens=attn_metadata.seq_lens,
                out=output,
                workspace=workspace,
            )
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
            return output

    def _get_fia_params(self, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata, kv_cache=None):
        # PrefillNoCache doesn't need key_cache, but other modes do
        # Only initialize/require cache for modes that actually use it
        if attn_metadata.attn_state != AscendAttentionState.PrefillNoCache:
            # Initialize cache from kv_cache if not already set (for DecodeOnly mode)
            if self.key_cache is None and kv_cache is not None:
                if (
                    isinstance(kv_cache, torch.Tensor)
                    and kv_cache.dim() > 0
                    and kv_cache.shape[0] == 2
                    or isinstance(kv_cache, (list, tuple))
                    and len(kv_cache) >= 2
                ):
                    self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

            if self.key_cache is None:
                raise RuntimeError(
                    f"key_cache is None in _get_fia_params for mode {attn_metadata.attn_state}. kv_cache={kv_cache}"
                )

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            block_size = 128
            block_table = None
            actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
            if self.attn_type == AttentionType.ENCODER_DECODER:
                actual_seq_lengths_kv = torch.cumsum(attn_metadata.seq_lens, dim=0).tolist()
        elif attn_metadata.attn_state == AscendAttentionState.PrefillCacheHit:
            batch_size = attn_metadata.seq_lens.shape[0]
            block_table = attn_metadata.block_tables[:batch_size, :]
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        # chunked prefill.
        else:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore
            key = self.key_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            value = self.value_cache.view(  # type: ignore
                num_block, block_size, -1
            )
            block_table = attn_metadata.block_tables
            actual_seq_lengths_kv = attn_metadata.seq_lens_list
        return key, value, block_size, block_table, actual_seq_lengths_kv

    def _forward_fia_slidingwindow(self, query: torch.Tensor, attn_metadata: AscendMetadata, output: torch.Tensor):
        batch_size = attn_metadata.seq_lens.shape[0]
        block_size = 128
        query = query.view(batch_size, 1, self.num_heads * self.head_size)
        key = self.key_cache
        value = self.value_cache
        if self.key_cache is not None and self.value_cache is not None:
            block_size = self.key_cache.shape[1]
            key = self.key_cache.flatten(2, 3).contiguous()
            value = self.value_cache.flatten(2, 3).contiguous()

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query,
            key,
            value,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BSH",
            block_size=block_size,
            pre_tokens=self.sliding_window,
            scale=self.scale,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths=[1] * len(attn_metadata.seq_lens),
            actual_seq_lengths_kv=attn_metadata.seq_lens,
        )

        attn_output = attn_output.view(batch_size, self.num_heads, self.head_size)
        output[:batch_size] = attn_output[:batch_size]
        return output

    def forward_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        kv_cache=None,
    ):
        # we inherit ForwardContext in model runner v2, when enable model
        # runner v2, there is not capturing attribute in forward_context,
        # just use getattr to avoid attribute error.
        if _EXTRA_CTX.capturing:
            attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and self.sliding_window is not None
            and attn_metadata.seq_lens.shape[0] == query.size(0)
            and self.sinks is None
        ):
            return self._forward_fia_slidingwindow(query, attn_metadata, output)
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(
            key, value, attn_metadata, kv_cache
        )
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        query = query[:num_tokens]
        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]
        # Get workspace from cache or calculate it if not present.
        if self.sinks is not None:
            actual_seq_qlen = attn_metadata.actual_seq_lengths_q
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                actual_seq_qlen = torch.tensor([1] * len(attn_metadata.seq_lens_list), dtype=torch.int32).cumsum(dim=0)
            if self.sliding_window is not None:
                atten_mask = attn_metadata.swa_mask
                sparse_mode = 4
            else:
                atten_mask = attn_metadata.attn_mask
                sparse_mode = 3
            attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                query,
                key,
                value,
                num_query_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="TND",
                pre_tokens=self.sliding_window if self.sliding_window is not None else SWA_INT_MAX,
                next_tokens=0,
                atten_mask=atten_mask,
                sparse_mode=sparse_mode,
                softmax_scale=self.scale,
                block_table=block_table,
                block_size=block_size,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_lengths_kv,
                learnable_sink=self.sinks,
            )
        else:
            attn_output, _ = torch_npu.npu_fused_infer_attention_score(
                query=query,
                key=key,
                value=value,
                atten_mask=attn_metadata.attn_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )

            attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _EXTRA_CTX.capturing:
            return self.full_graph_pa(query, attn_metadata, output)
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            block_table=attn_metadata.block_tables,
            context_lens=attn_metadata.seq_lens,
            out=output,
        )
        return output

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        _: torch.Tensor,
    ) -> torch.Tensor:
        # use default sparse_mode 0 in normal scenario, which means no mask works on it
        return torch_npu.npu_fusion_attention(
            query=query,
            key=key,
            value=value,
            head_num=self.num_heads,
            input_layout="TND",
            scale=self.scale,
            actual_seq_qlen=attn_metadata.actual_seq_lengths_q,
            actual_seq_kvlen=attn_metadata.actual_seq_lengths_q,
        )[0]

    def reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if len(kv_cache) > 1:
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event = torch.npu.Event()
            if self.key_cache is None:
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
            slots = attn_metadata.slot_mapping
            encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER
            DeviceOperator.reshape_and_cache(
                key=key[: attn_metadata.num_actual_tokens] if not encoder_decoder else key,
                value=value[: attn_metadata.num_actual_tokens] if not encoder_decoder else value,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                # quick fix to make sure slots is int32 for cross attention case.
                # see: https://github.com/vllm-project/vllm/blob/ce88756b967c2c5006746a424c15dd59a284ed8c/vllm/model_executor/layers/attention/cross_attention.py#L117
                slot_mapping=slots[: attn_metadata.num_actual_tokens] if not encoder_decoder else slots.to(torch.int32),
            )
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()
        return query, key, value, output

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        num_tokens = query.shape[0]
        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and using_paged_attention(num_tokens, self.vllm_config)
            and self.sliding_window is None
        ):
            output = self.forward_paged_attention(query, attn_metadata, output)
        else:
            output = self.forward_fused_infer_attention(query, key, value, attn_metadata, output, kv_cache)

        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendAttentionBackendImpl")

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        # Initialize key_cache and value_cache from kv_cache if not already set.
        # This is needed for DecodeOnly mode where key/value are None but we still
        # need access to the cache for attention computation.
        if self.key_cache is None and kv_cache is not None:
            if (
                isinstance(kv_cache, torch.Tensor)
                and kv_cache.dim() > 0
                and kv_cache.shape[0] == 2
                or isinstance(kv_cache, (list, tuple))
                and len(kv_cache) >= 2
            ):
                self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        output_padded = None
        if key is not None and value is not None:
            output_padded = output
            query, key, value, output_padded = self.reshape_and_cache(
                query, key, value, kv_cache, attn_metadata, output
            )
        # pooling model branch
        if attn_metadata.model_runner_type == "pooling" and not attn_metadata.causal:
            attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output
        if output_padded is not None:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output_padded)
        else:
            attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output)
        output[:num_tokens] = attn_output[:num_tokens]
        return output


class AscendC8AttentionBackendImpl(AscendAttentionBackendImpl):
    """Attention backend implementation for INT8 KV cache (C8/QuaRot) models.

    This subclass handles static per-channel INT8 KV cache quantization.
    It is activated via class surgery in AscendC8KVCacheAttentionMethod.create_weights
    (vllm_ascend/quantization/methods/kv_c8.py)
    so that C8 attention layers automatically use this forward path.
    """

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendC8AttentionBackendImpl")

        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        float_key, float_value = None, None
        if key is not None and value is not None:
            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                float_key, float_value = key, value
            key, value = self._quantize_kv_to_int8(key, value, layer, attn_metadata.num_actual_tokens)
            query, key, value, _ = self.reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)

        if attn_metadata.model_runner_type == "pooling":
            attn_output = self._forward_encoder_attention(query, key, value, attn_metadata, output)
            output[:num_tokens] = attn_output[:num_tokens]
            return output

        self._prepare_c8_scales(layer, query.device)
        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            return self._forward_c8_decode(query, attn_metadata, output, layer)
        elif attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
            return self._forward_c8_chunked_prefill(query, float_key, float_value, attn_metadata, output, layer)
        else:
            return self._forward_c8_fused_infer_attention(
                query,
                float_key if float_key is not None else key,
                float_value if float_value is not None else value,
                attn_metadata,
                output,
                layer,
            )

    def _prepare_c8_scales(self, layer: AttentionLayer, device: torch.device) -> None:
        """Shard per-channel C8 scales/offsets to this TP rank and pre-compute
        BF16 BNSD antiquant tensors for FIA V1 decode fast path.
        """
        if hasattr(layer, "_c8_scales_prepared"):
            return

        def _shard_and_reshape(raw: torch.Tensor) -> torch.Tensor:
            if raw.numel() == 1:
                return raw.to(device=device)
            expected = self.num_kv_heads * self.head_size
            if raw.numel() != expected:
                total_kv_heads = raw.numel() // self.head_size
                tp_rank = get_tensor_model_parallel_rank()
                tp_size = get_tensor_model_parallel_world_size()
                kv_head_start = tp_rank * total_kv_heads // tp_size
                raw = raw.view(total_kv_heads, self.head_size)[
                    kv_head_start : kv_head_start + self.num_kv_heads
                ].contiguous()
            return raw.view(1, self.num_kv_heads, self.head_size).to(device=device)

        layer._c8_k_scale = _shard_and_reshape(layer.k_cache_scale.data)
        layer._c8_k_offset = _shard_and_reshape(layer.k_cache_offset.data)
        layer._c8_v_scale = _shard_and_reshape(layer.v_cache_scale.data)
        layer._c8_v_offset = _shard_and_reshape(layer.v_cache_offset.data)

        bnsd = (1, self.num_kv_heads, 1, self.head_size)
        layer._c8_k_aq_scale = layer._c8_k_scale.to(torch.bfloat16).view(bnsd).contiguous()
        layer._c8_k_aq_offset = layer._c8_k_offset.to(torch.bfloat16).view(bnsd).contiguous()
        layer._c8_v_aq_scale = layer._c8_v_scale.to(torch.bfloat16).view(bnsd).contiguous()
        layer._c8_v_aq_offset = layer._c8_v_offset.to(torch.bfloat16).view(bnsd).contiguous()

        layer._c8_k_inv_scale_bf16 = (1.0 / layer._c8_k_scale).to(torch.bfloat16)
        layer._c8_k_offset_bf16 = layer._c8_k_offset.to(torch.bfloat16)
        layer._c8_v_inv_scale_bf16 = (1.0 / layer._c8_v_scale).to(torch.bfloat16)
        layer._c8_v_offset_bf16 = layer._c8_v_offset.to(torch.bfloat16)

        layer._c8_scales_prepared = True

    def _dequant_paged_kv_to_dense(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: list,
        target_dtype: torch.dtype,
        layer,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather paged INT8 KV blocks and dequantize to target_dtype."""
        batch_size = block_table.shape[0]
        block_size = key.shape[1]
        H = key.shape[2]
        max_blocks_per_seq = block_table.shape[1]
        max_tokens_padded = max_blocks_per_seq * block_size

        flat_ids = block_table.reshape(-1)
        gathered_k = key[flat_ids].view(batch_size, max_tokens_padded, H)
        gathered_v = value[flat_ids].view(batch_size, max_tokens_padded, H)

        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=key.device)
        positions = torch.arange(max_tokens_padded, dtype=torch.long, device=key.device)
        valid_mask = (positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)).view(-1)

        dense_k = gathered_k.view(-1, H)[valid_mask]
        dense_v = gathered_v.view(-1, H)[valid_mask]

        dense_k = dense_k.view(-1, self.num_kv_heads, self.head_size)
        dense_v = dense_v.view(-1, self.num_kv_heads, self.head_size)
        k_scale = layer._c8_k_scale.to(target_dtype)
        k_offset = layer._c8_k_offset.to(target_dtype)
        v_scale = layer._c8_v_scale.to(target_dtype)
        v_offset = layer._c8_v_offset.to(target_dtype)
        dense_k = (dense_k.to(target_dtype) - k_offset) * k_scale
        dense_v = (dense_v.to(target_dtype) - v_offset) * v_scale
        return dense_k, dense_v

    def _quantize_kv_to_int8(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: AttentionLayer,
        num_actual_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize K/V from float to INT8 using static per-channel C8 scales."""
        self._prepare_c8_scales(layer, key.device)

        actual_key = key[:num_actual_tokens]
        actual_value = value[:num_actual_tokens]

        k_int8 = torch.clamp(
            torch.round(actual_key * layer._c8_k_inv_scale_bf16 + layer._c8_k_offset_bf16),
            -128,
            127,
        ).to(torch.int8)
        v_int8 = torch.clamp(
            torch.round(actual_value * layer._c8_v_inv_scale_bf16 + layer._c8_v_offset_bf16),
            -128,
            127,
        ).to(torch.int8)
        return k_int8, v_int8

    def _forward_c8_decode(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 decode via FIA V1 BNSD with native paged INT8 KV + perchannel antiquant."""
        num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
        assert block_size % 32 == 0, f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
        key = self.key_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]
        value = self.value_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]
        batch_size = len(attn_metadata.seq_lens_list)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query[:batch_size].unsqueeze(2),
            key,
            value,
            key_antiquant_scale=layer._c8_k_aq_scale,
            key_antiquant_offset=layer._c8_k_aq_offset,
            value_antiquant_scale=layer._c8_v_aq_scale,
            value_antiquant_offset=layer._c8_v_aq_offset,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD",
            scale=self.scale,
            block_size=block_size,
            key_antiquant_mode=0,
            value_antiquant_mode=0,
            sparse_mode=0,
        )
        attn_output = attn_output.squeeze(2)
        output[:batch_size] = attn_output
        return output

    def _forward_c8_chunked_prefill(
        self,
        query: torch.Tensor,
        float_key: torch.Tensor | None,
        float_value: torch.Tensor | None,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        """C8 ChunkedPrefill: decode via FIA V1 BNSD paged INT8 (zero gather),
        prefill via FIA V1 TND with float KV (new) or gather+dequant (continuing).
        """
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]

        if num_decode_tokens > 0:
            num_block, block_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
            assert block_size % 32 == 0, (
                f"C8 INT8 KV cache requires block_size to be a multiple of 32, got {block_size}"
            )
            kv_k = self.key_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]
            kv_v = self.value_cache.view(num_block, block_size, -1)  # type: ignore[attr-defined]

            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query[:num_decode_tokens].unsqueeze(2),
                kv_k,
                kv_v,
                key_antiquant_scale=layer._c8_k_aq_scale,
                key_antiquant_offset=layer._c8_k_aq_offset,
                value_antiquant_scale=layer._c8_v_aq_scale,
                value_antiquant_offset=layer._c8_v_aq_offset,
                block_table=attn_metadata.block_tables[:num_decodes],
                actual_seq_lengths_kv=attn_metadata.seq_lens_list[:num_decodes],
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD",
                scale=self.scale,
                block_size=block_size,
                key_antiquant_mode=0,
                value_antiquant_mode=0,
                sparse_mode=0,
            )
            output[:num_decode_tokens] = attn_out.squeeze(2)

        if attn_metadata.num_prefills > 0:
            prefill_q = query[num_decode_tokens:num_tokens]

            prefill_seq_qlen = [
                actual_seq_qlen[i] - num_decode_tokens for i in range(num_decodes, len(actual_seq_qlen))
            ]

            all_new_prefill = True
            for i in range(num_decodes, len(attn_metadata.seq_lens_list)):
                q_start = actual_seq_qlen[i - 1] if i > 0 else 0
                qlen_i = actual_seq_qlen[i] - q_start
                if attn_metadata.seq_lens_list[i] > qlen_i:
                    all_new_prefill = False
                    break

            if all_new_prefill and float_key is not None and float_value is not None:
                prefill_k = float_key[num_decode_tokens:num_tokens]
                prefill_v = float_value[num_decode_tokens:num_tokens]
                prefill_seq_kvlen = prefill_seq_qlen
            else:
                num_block, blk_size, _, _ = self.key_cache.shape  # type: ignore[attr-defined]
                paged_k = self.key_cache.view(num_block, blk_size, -1)  # type: ignore[attr-defined]
                paged_v = self.value_cache.view(num_block, blk_size, -1)  # type: ignore[attr-defined]
                prefill_bt = attn_metadata.block_tables[num_decodes:]
                prefill_sl = attn_metadata.seq_lens_list[num_decodes:]
                prefill_k, prefill_v = self._dequant_paged_kv_to_dense(
                    paged_k, paged_v, prefill_bt, prefill_sl, query.dtype, layer
                )
                prefill_seq_kvlen = torch.tensor(prefill_sl, dtype=torch.int32).cumsum(dim=0)

            # block_table is None for prefill; FIA ignores block_size in this case.
            # Use cache block_size for consistency rather than a magic number.
            cache_block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query=prefill_q,
                key=prefill_k,
                value=prefill_v,
                atten_mask=attn_metadata.attn_mask,
                block_table=None,
                input_layout="TND",
                block_size=cache_block_size,
                actual_seq_lengths=prefill_seq_qlen,
                actual_seq_lengths_kv=prefill_seq_kvlen,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
                sparse_mode=3,
            )
            n_prefill = num_tokens - num_decode_tokens
            attn_out = attn_out.view(n_prefill, self.num_heads, self.head_size)
            output[num_decode_tokens:num_tokens] = attn_out[:n_prefill]

        return output

    def _forward_c8_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ):
        """C8 FIA V1 TND for prefill states (PrefillNoCache uses float KV directly,
        PrefillCacheHit gathers + dequants paged INT8 KV).
        """
        self._prepare_c8_scales(layer, query.device)
        key, value, block_size, block_table, actual_seq_lengths_kv = self._get_fia_params(key, value, attn_metadata)

        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])  # type: ignore[index]
        query = query[:num_tokens]

        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]

        if key.dtype == torch.int8:
            if block_table is not None:
                seq_lens = (
                    actual_seq_lengths_kv if isinstance(actual_seq_lengths_kv, list) else actual_seq_lengths_kv.tolist()
                )
                key, value = self._dequant_paged_kv_to_dense(key, value, block_table, seq_lens, query.dtype, layer)
                block_table = None
                # block_table is None after dequant; FIA ignores block_size.
                # Use cache block_size for consistency rather than a magic number.
                block_size = self.key_cache.shape[1]  # type: ignore[attr-defined]
                actual_seq_lengths_kv = torch.tensor(seq_lens, dtype=torch.int32).cumsum(dim=0)
            else:
                qdt = query.dtype
                k_scale = layer._c8_k_scale.to(qdt)
                k_offset = layer._c8_k_offset.to(qdt)
                v_scale = layer._c8_v_scale.to(qdt)
                v_offset = layer._c8_v_offset.to(qdt)
                key = (key.to(qdt) - k_offset) * k_scale
                value = (value.to(qdt) - v_offset) * v_scale

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output
        return output


class AscendTurboQuantAttentionBackendImpl(AscendAttentionBackendImpl):
    """Reference TurboQuant backend for dense attention on Ascend.

    This first-stage implementation intentionally favors correctness over
    throughput. KV is encoded on cache write, gathered back from paged cache,
    dequantized to dense tensors, and then computed by the existing FIA path.
    """

    def _prepare_turboquant_runtime(self, layer: AttentionLayer, device: torch.device) -> None:
        if (
            getattr(layer, "tq_runtime_prepared", False)
            and hasattr(layer, "_tq_k_qjl_proj_t")
            and hasattr(layer, "_tq_kv_mse_rotation")
            and hasattr(layer, "_tq_kv_mse_shared_boundary")
        ):
            return
        target_dtype = getattr(layer, "tq_scalar_dtype", torch.float32)
        layer._tq_k_codebook = layer.k_codebook.data.to(device=device, dtype=target_dtype)
        layer._tq_v_codebook = layer.v_codebook.data.to(device=device, dtype=target_dtype)
        layer._tq_k_boundary = layer.k_boundary.data.to(device=device, dtype=target_dtype)
        layer._tq_v_boundary = layer.v_boundary.data.to(device=device, dtype=target_dtype)
        layer._tq_k_rot = layer.k_rot.data.to(device=device, dtype=target_dtype)
        layer._tq_v_rot = layer.v_rot.data.to(device=device, dtype=target_dtype)
        layer._tq_k_rot_t = layer._tq_k_rot.transpose(0, 1).contiguous()
        layer._tq_v_rot_t = layer._tq_v_rot.transpose(0, 1).contiguous()
        layer._tq_k_qjl_proj = layer.k_qjl_proj.data.to(device=device, dtype=target_dtype)
        layer._tq_k_qjl_proj_t = layer._tq_k_qjl_proj.transpose(0, 1).contiguous()
        layer._tq_k_qjl_query_matrix = (
            layer._tq_k_rot @ layer._tq_k_qjl_proj_t
        ).contiguous()
        if (
            layer._tq_k_rot.shape == layer._tq_v_rot.shape
            and int(layer.tq_k_stage1_bits) == int(layer.tq_v_stage1_bits)
        ):
            layer._tq_kv_mse_shared_boundary = bool(
                layer.k_boundary.shape == layer.v_boundary.shape
                and torch.equal(layer.k_boundary.data, layer.v_boundary.data)
            )
            layer._tq_kv_mse_rotation = torch.stack(
                (layer.k_rot.data, layer.v_rot.data),
                dim=0,
            ).to(device=device, dtype=target_dtype).contiguous()
        else:
            layer._tq_kv_mse_shared_boundary = False
            layer._tq_kv_mse_rotation = None
        layer._tq_qjl_codebook = torch.tensor(
            [-1.0, 1.0],
            dtype=torch.float32,
            device=device,
        ).contiguous()
        layer.tq_runtime_prepared = True

    def _quantize_kv_to_turboquant(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        layer: AttentionLayer,
        num_actual_tokens: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._prepare_turboquant_runtime(layer, key.device)
        actual_key = key[:num_actual_tokens].to(torch.float32)
        actual_value = value[:num_actual_tokens].to(torch.float32)

        if getattr(layer, "tq_k_variant", "prod") == "prod":
            encoded_k = turboquant_encode_prod(
                actual_key,
                layer._tq_k_rot,
                layer._tq_k_codebook,
                layer._tq_k_boundary,
                layer._tq_k_qjl_proj,
                int(layer.tq_k_total_bits),
            )
        else:
            encoded_k = turboquant_encode_mse(
                actual_key,
                layer._tq_k_rot,
                layer._tq_k_codebook,
                layer._tq_k_boundary,
                int(layer.tq_k_stage1_bits),
            )

        if getattr(layer, "tq_v_variant", "mse") != "mse":
            raise RuntimeError(f"Unsupported TurboQuant V variant at runtime: {layer.tq_v_variant}")
        encoded_v = turboquant_encode_mse(
            actual_value,
            layer._tq_v_rot,
            layer._tq_v_codebook,
            layer._tq_v_boundary,
            int(layer.tq_v_stage1_bits),
        )
        return encoded_k, encoded_v

    def _reshape_and_cache_turboquant(
        self,
        query: torch.Tensor,
        key: dict[str, torch.Tensor],
        value: dict[str, torch.Tensor],
        kv_cache,
        attn_metadata: AscendMetadata,
    ):
        if not isinstance(kv_cache, dict):
            raise TypeError("TurboQuant KV cache must be a dict-based sidecar cache")
        self.key_cache = kv_cache  # type: ignore[assignment]
        slots = attn_metadata.slot_mapping[: attn_metadata.num_actual_tokens].to(torch.long)
        if slots.numel() == 0:
            return

        for cache_name, encoded in (
            ("k_idx", key["idx"]),
            ("k_norm", key["norm"]),
            ("v_idx", value["idx"]),
            ("v_norm", value["norm"]),
        ):
            flat_cache = kv_cache[cache_name].view(-1, self.num_kv_heads, kv_cache[cache_name].shape[-1])
            flat_cache[slots] = encoded.to(flat_cache.dtype)
        if "qjl" in key and "k_qjl" in kv_cache:
            flat_cache = kv_cache["k_qjl"].view(-1, self.num_kv_heads, kv_cache["k_qjl"].shape[-1])
            flat_cache[slots] = key["qjl"].to(flat_cache.dtype)
        if "gamma" in key and "k_gamma" in kv_cache:
            flat_cache = kv_cache["k_gamma"].view(-1, self.num_kv_heads, kv_cache["k_gamma"].shape[-1])
            flat_cache[slots] = key["gamma"].to(flat_cache.dtype)

    def _gather_turboquant_cache_tokens(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: list[int],
        *,
        profile_label: str,
    ) -> torch.Tensor:
        _maybe_sync_for_profile(cache, block_table)
        t0 = time.perf_counter()
        batch_size = block_table.shape[0]
        block_size = cache.shape[1]
        max_blocks_per_seq = block_table.shape[1]
        max_tokens_padded = max_blocks_per_seq * block_size
        flat_ids = block_table.reshape(-1)
        gathered = cache[flat_ids].view(batch_size, max_tokens_padded, *cache.shape[2:])
        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=cache.device)
        positions = torch.arange(max_tokens_padded, dtype=torch.long, device=cache.device)
        valid_mask = (positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)).reshape(-1)
        gathered = gathered.reshape(batch_size * max_tokens_padded, *cache.shape[2:])[valid_mask]
        _maybe_sync_for_profile(gathered)
        _record_tq_profile(
            profile_label, (time.perf_counter() - t0) * 1000.0,
            vectors=int(gathered.numel() // max(gathered.shape[-1], 1)),
            bytes_out=gathered.numel(),
        )
        return gathered

    def _dequant_paged_k_reference(
        self,
        kv_cache: dict[str, torch.Tensor],
        block_table: torch.Tensor,
        seq_lens: list[int],
        target_dtype: torch.dtype,
        layer: AttentionLayer,
        *,
        profile_label: str,
    ) -> torch.Tensor:
        if getattr(layer, "tq_k_variant", "prod") == "prod" and "k_qjl" in kv_cache and "k_gamma" in kv_cache:
            return turboquant_decode_prod(
                self._gather_turboquant_cache_tokens(
                    kv_cache["k_idx"], block_table, seq_lens,
                    profile_label=f"{profile_label}.gather.k_idx",
                ),
                self._gather_turboquant_cache_tokens(
                    kv_cache["k_qjl"], block_table, seq_lens,
                    profile_label=f"{profile_label}.gather.k_qjl",
                ),
                self._gather_turboquant_cache_tokens(
                    kv_cache["k_gamma"], block_table, seq_lens,
                    profile_label=f"{profile_label}.gather.k_gamma",
                ),
                self._gather_turboquant_cache_tokens(
                    kv_cache["k_norm"], block_table, seq_lens,
                    profile_label=f"{profile_label}.gather.k_norm",
                ),
                layer._tq_k_rot_t,
                layer._tq_k_codebook,
                layer._tq_k_qjl_proj,
                int(layer.tq_k_total_bits),
                self.head_size,
                target_dtype,
            )

        return turboquant_decode_mse(
            self._gather_turboquant_cache_tokens(
                kv_cache["k_idx"], block_table, seq_lens,
                profile_label=f"{profile_label}.gather.k_idx",
            ),
            self._gather_turboquant_cache_tokens(
                kv_cache["k_norm"], block_table, seq_lens,
                profile_label=f"{profile_label}.gather.k_norm",
            ),
            layer._tq_k_rot_t,
            layer._tq_k_codebook,
            int(layer.tq_k_stage1_bits),
            self.head_size,
            target_dtype,
        )

    def _dequant_paged_v_reference(
        self,
        kv_cache: dict[str, torch.Tensor],
        block_table: torch.Tensor,
        seq_lens: list[int],
        target_dtype: torch.dtype,
        layer: AttentionLayer,
        *,
        profile_label: str,
    ) -> torch.Tensor:
        v_dim = int(getattr(layer, "tq_head_size_v", self.head_size))
        return turboquant_decode_mse(
            self._gather_turboquant_cache_tokens(
                kv_cache["v_idx"], block_table, seq_lens,
                profile_label=f"{profile_label}.gather.v_idx",
            ),
            self._gather_turboquant_cache_tokens(
                kv_cache["v_norm"], block_table, seq_lens,
                profile_label=f"{profile_label}.gather.v_norm",
            ),
            layer._tq_v_rot_t,
            layer._tq_v_codebook,
            int(layer.tq_v_stage1_bits),
            v_dim,
            target_dtype,
        )

    def _dequant_paged_kv_to_dense_reference(
        self,
        kv_cache: dict[str, torch.Tensor],
        block_table: torch.Tensor,
        seq_lens: list[int],
        target_dtype: torch.dtype,
        layer: AttentionLayer,
        *,
        profile_label: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference path: gather each cache tensor then full PyTorch decode."""
        _maybe_sync_for_profile(block_table)
        t_total_0 = time.perf_counter()
        self._prepare_turboquant_runtime(layer, next(iter(kv_cache.values())).device)

        dense_k = self._dequant_paged_k_reference(
            kv_cache, block_table, seq_lens, target_dtype, layer,
            profile_label=profile_label,
        )
        dense_v = self._dequant_paged_v_reference(
            kv_cache, block_table, seq_lens, target_dtype, layer,
            profile_label=profile_label,
        )

        _record_tq_profile(
            f"{profile_label}.total",
            (time.perf_counter() - t_total_0) * 1000.0,
            vectors=len(seq_lens),
        )
        return dense_k, dense_v

    def _dequant_mse_paged_cache_custom(
        self,
        packed_idx: torch.Tensor,
        norm: torch.Tensor,
        token_block_ids: torch.Tensor,
        token_offsets: torch.Tensor,
        codebook: torch.Tensor,
        rotation_t: torch.Tensor,
        bits: int,
        head_dim: int,
        target_dtype: torch.dtype,
        *,
        profile_label: str,
    ) -> torch.Tensor:
        _maybe_sync_for_profile(packed_idx, norm, token_block_ids, token_offsets)
        t0 = time.perf_counter()
        t_stage = time.perf_counter()
        dense_rot = tq_dequant_mse_paged_rot(
            packed_idx=packed_idx,
            norm=norm,
            token_block_ids=token_block_ids,
            token_offsets=token_offsets,
            codebook=codebook,
            bits=bits,
            head_dim=head_dim,
            target_dtype=target_dtype,
        )
        _maybe_sync_for_profile(dense_rot)
        _record_tq_profile(
            f"{profile_label}.paged_rot",
            (time.perf_counter() - t_stage) * 1000.0,
            vectors=int(dense_rot.shape[0]) if dense_rot.dim() else 0,
            bytes_out=dense_rot.numel() * dense_rot.element_size(),
        )
        t_stage = time.perf_counter()
        dense = apply_rotation(dense_rot, rotation_t).contiguous()
        _maybe_sync_for_profile(dense)
        _record_tq_profile(
            f"{profile_label}.inverse_rotate",
            (time.perf_counter() - t_stage) * 1000.0,
            vectors=int(dense.shape[0]) if dense.dim() else 0,
            bytes_out=dense.numel() * dense.element_size(),
        )
        _record_tq_profile(
            profile_label,
            (time.perf_counter() - t0) * 1000.0,
            vectors=int(dense.shape[0]) if dense.dim() else 0,
            bytes_out=dense.numel() * dense.element_size(),
        )
        return dense

    def _dequant_prod_paged_cache_hybrid(
        self,
        kv_cache: dict[str, torch.Tensor],
        token_block_ids: torch.Tensor,
        token_offsets: torch.Tensor,
        target_dtype: torch.dtype,
        layer: AttentionLayer,
        *,
        profile_label: str,
    ) -> torch.Tensor:
        _maybe_sync_for_profile(
            kv_cache["k_idx"],
            kv_cache["k_qjl"],
            kv_cache["k_gamma"],
            kv_cache["k_norm"],
            token_block_ids,
            token_offsets,
        )
        t0 = time.perf_counter()

        t_stage = time.perf_counter()
        k_stage1_rot = tq_dequant_mse_paged_rot(
            packed_idx=kv_cache["k_idx"],
            norm=kv_cache["k_norm"],
            token_block_ids=token_block_ids,
            token_offsets=token_offsets,
            codebook=layer._tq_k_codebook,
            bits=int(layer.tq_k_stage1_bits),
            head_dim=self.head_size,
            target_dtype=target_dtype,
        )
        _maybe_sync_for_profile(k_stage1_rot)
        _record_tq_profile(
            f"{profile_label}.stage1_mse",
            (time.perf_counter() - t_stage) * 1000.0,
            vectors=int(k_stage1_rot.shape[0]) if k_stage1_rot.dim() else 0,
            bytes_out=k_stage1_rot.numel() * k_stage1_rot.element_size(),
        )

        correction = math.sqrt(math.pi / 2.0) / self.head_size
        t_stage = time.perf_counter()
        qjl_scaled = tq_dequant_mse_paged_scaled_rot(
            packed_idx=kv_cache["k_qjl"],
            norm=kv_cache["k_norm"],
            extra_scale=kv_cache["k_gamma"],
            token_block_ids=token_block_ids,
            token_offsets=token_offsets,
            codebook=layer._tq_qjl_codebook,
            bits=1,
            head_dim=self.head_size,
            target_dtype=target_dtype,
            scale_multiplier=correction,
            signed_bits1=True,
        )
        _maybe_sync_for_profile(qjl_scaled)
        _record_tq_profile(
            f"{profile_label}.qjl_unpack_scale",
            (time.perf_counter() - t_stage) * 1000.0,
            vectors=int(qjl_scaled.shape[0]) if qjl_scaled.dim() else 0,
            bytes_out=qjl_scaled.numel() * qjl_scaled.element_size(),
        )
        t_stage = time.perf_counter()
        qjl_rot = apply_rotation(qjl_scaled, layer._tq_k_qjl_proj)
        _maybe_sync_for_profile(qjl_rot)
        _record_tq_profile(
            f"{profile_label}.qjl_project",
            (time.perf_counter() - t_stage) * 1000.0,
            vectors=int(qjl_rot.shape[0]) if qjl_rot.dim() else 0,
            bytes_out=qjl_rot.numel() * qjl_rot.element_size(),
        )

        t_stage = time.perf_counter()
        k_rot = k_stage1_rot + qjl_rot
        _maybe_sync_for_profile(k_rot)
        _record_tq_profile(
            f"{profile_label}.combine",
            (time.perf_counter() - t_stage) * 1000.0,
            vectors=int(k_rot.shape[0]) if k_rot.dim() else 0,
            bytes_out=k_rot.numel() * k_rot.element_size(),
        )
        t_stage = time.perf_counter()
        dense_k = apply_rotation(k_rot, layer._tq_k_rot_t).contiguous()
        _maybe_sync_for_profile(dense_k)
        _record_tq_profile(
            f"{profile_label}.inverse_rotate",
            (time.perf_counter() - t_stage) * 1000.0,
            vectors=int(dense_k.shape[0]) if dense_k.dim() else 0,
            bytes_out=dense_k.numel() * dense_k.element_size(),
        )
        _record_tq_profile(
            profile_label,
            (time.perf_counter() - t0) * 1000.0,
            vectors=int(dense_k.shape[0]) if dense_k.dim() else 0,
            bytes_out=dense_k.numel() * dense_k.element_size(),
        )
        return dense_k

    def _dequant_paged_kv_to_dense_custom_mse(
        self,
        kv_cache: dict[str, torch.Tensor],
        block_table: torch.Tensor,
        seq_lens: list[int],
        target_dtype: torch.dtype,
        layer: AttentionLayer,
        *,
        profile_label: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """P2 path: custom MSE dequant plus PyTorch QJL correction for K=prod."""
        _maybe_sync_for_profile(block_table)
        t_total_0 = time.perf_counter()
        self._prepare_turboquant_runtime(layer, next(iter(kv_cache.values())).device)
        cache_block_size = kv_cache["v_idx"].shape[1]

        t_map = time.perf_counter()
        token_block_ids, token_offsets = cached_token_map_from_block_table(
            block_table=block_table, seq_lens=seq_lens, block_size=cache_block_size,
        )
        _maybe_sync_for_profile(token_block_ids, token_offsets)
        _record_tq_profile(
            f"{profile_label}.token_map",
            (time.perf_counter() - t_map) * 1000.0,
            vectors=int(token_block_ids.numel()),
            bytes_out=(
                token_block_ids.numel() * token_block_ids.element_size()
                + token_offsets.numel() * token_offsets.element_size()
            ),
        )

        # V (always MSE)
        v_dim = int(getattr(layer, "tq_head_size_v", self.head_size))
        dense_v = self._dequant_mse_paged_cache_custom(
            kv_cache["v_idx"],
            kv_cache["v_norm"],
            token_block_ids,
            token_offsets,
            layer._tq_v_codebook,
            layer._tq_v_rot_t,
            int(layer.tq_v_stage1_bits),
            v_dim,
            target_dtype,
            profile_label=f"{profile_label}.custom_mse.v",
        )

        if getattr(layer, "tq_k_variant", "prod") == "mse":
            dense_k = self._dequant_mse_paged_cache_custom(
                kv_cache["k_idx"],
                kv_cache["k_norm"],
                token_block_ids,
                token_offsets,
                layer._tq_k_codebook,
                layer._tq_k_rot_t,
                int(layer.tq_k_stage1_bits),
                self.head_size,
                target_dtype,
                profile_label=f"{profile_label}.custom_mse.k",
            )
        else:
            dense_k = self._dequant_prod_paged_cache_hybrid(
                kv_cache,
                token_block_ids,
                token_offsets,
                target_dtype,
                layer,
                profile_label=f"{profile_label}.hybrid_prod.k",
            )

        _record_tq_profile(
            f"{profile_label}.custom_mse.total",
            (time.perf_counter() - t_total_0) * 1000.0,
            vectors=len(seq_lens),
        )

        if dequant_debug_compare_enabled():
            ref_k, ref_v = self._dequant_paged_kv_to_dense_reference(
                kv_cache, block_table, seq_lens, target_dtype, layer,
                profile_label=f"{profile_label}.debug_reference",
            )
            _maybe_sync_for_profile(dense_k, dense_v, ref_k, ref_v)
            max_k_diff = (
                (dense_k - ref_k).abs().max().item()
                if dense_k.numel() else 0.0
            )
            max_v_diff = (
                (dense_v - ref_v).abs().max().item()
                if dense_v.numel() else 0.0
            )
            if max(max_k_diff, max_v_diff) > 1e-3:
                raise RuntimeError(
                    "TurboQuant custom dequant mismatch: "
                    f"max_k_diff={max_k_diff}, max_v_diff={max_v_diff}"
                )
        return dense_k, dense_v

    def _dequant_paged_kv_to_dense(
        self,
        kv_cache: dict[str, torch.Tensor],
        block_table: torch.Tensor,
        seq_lens: list[int],
        target_dtype: torch.dtype,
        layer: AttentionLayer,
        *,
        profile_label: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if custom_dequant_enabled():
            try:
                return self._dequant_paged_kv_to_dense_custom_mse(
                    kv_cache, block_table, seq_lens, target_dtype, layer,
                    profile_label=profile_label,
                )
            except Exception:
                if (
                    os.getenv("VLLM_ASCEND_TQ_CUSTOM_STRICT", "0") == "1"
                    or dequant_debug_compare_enabled()
                ):
                    raise

        return self._dequant_paged_kv_to_dense_reference(
            kv_cache, block_table, seq_lens, target_dtype, layer,
            profile_label=profile_label,
        )

    def _run_dense_fia(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        actual_seq_lengths_q: list[int],
        actual_seq_lengths_kv: list[int],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        *,
        profile_name: str,
    ) -> torch.Tensor:
        _maybe_sync_for_profile(query, key, value)
        t0 = time.perf_counter()
        attn_mask = getattr(attn_metadata, "attn_mask", None)
        sparse_mode = 3 if attn_mask is not None else 0
        if isinstance(self.key_cache, dict):
            cache_block_size = next(iter(self.key_cache.values())).shape[1]
        elif self.key_cache is not None:
            cache_block_size = self.key_cache.shape[1]
        else:
            cache_block_size = 1
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_mask,
            block_table=None,
            input_layout="TND",
            block_size=cache_block_size,
            actual_seq_lengths=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=sparse_mode,
        )
        num_tokens = actual_seq_lengths_q[-1]
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output[:num_tokens]
        _maybe_sync_for_profile(attn_output, output)
        _record_tq_profile(
            profile_name,
            (time.perf_counter() - t0) * 1000.0,
            vectors=num_tokens,
        )
        return output

    def _run_turboquant_fused_decode_attention(
        self,
        query: torch.Tensor,
        kv_cache: dict[str, torch.Tensor],
        block_table: torch.Tensor,
        seq_lens: list[int],
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        _maybe_sync_for_profile(query, block_table)
        t0 = time.perf_counter()
        self._prepare_turboquant_runtime(layer, next(iter(kv_cache.values())).device)

        batch_size = len(seq_lens)
        fused_out = tq_prod_mse_paged_attention(
            query[:batch_size],
            kv_cache["k_idx"],
            kv_cache["k_qjl"],
            kv_cache["k_gamma"],
            kv_cache["k_norm"],
            kv_cache["v_idx"],
            kv_cache["v_norm"],
            block_table,
            seq_lens,
            layer._tq_k_codebook,
            layer._tq_v_codebook,
            layer._tq_k_rot,
            layer._tq_k_qjl_proj,
            layer._tq_v_rot_t,
            int(layer.tq_k_total_bits),
            int(layer.tq_v_stage1_bits),
            self.head_size,
            scale=self.scale,
            max_seq_len=max(seq_lens, default=0),
        )
        output[:batch_size] = fused_out.to(dtype=output.dtype)
        _maybe_sync_for_profile(output)
        _record_tq_profile(
            "turboquant_decode.fused_attention",
            (time.perf_counter() - t0) * 1000.0,
            vectors=batch_size,
        )
        return output

    def _forward_turboquant_decode(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        _maybe_sync_for_profile(query, attn_metadata.block_tables)
        t_total_0 = time.perf_counter()
        kv_seq_lens = attn_metadata.seq_lens_list
        if (
            fused_attention_custom_enabled()
            and isinstance(self.key_cache, dict)
            and getattr(layer, "tq_k_variant", "prod") == "prod"
            and "k_qjl" in self.key_cache
            and "k_gamma" in self.key_cache
        ):
            try:
                output = self._run_turboquant_fused_decode_attention(
                    query,
                    self.key_cache,
                    attn_metadata.block_tables,
                    kv_seq_lens,
                    output,
                    layer,
                )
                _record_tq_profile(
                    "turboquant_decode.total",
                    (time.perf_counter() - t_total_0) * 1000.0,
                    vectors=len(kv_seq_lens),
                )
                return output
            except Exception:
                if (
                    os.getenv("VLLM_ASCEND_TQ_CUSTOM_STRICT", "0") == "1"
                    or attention_debug_compare_enabled()
                ):
                    raise

        dense_k, dense_v = self._dequant_paged_kv_to_dense(
            self.key_cache,  # type: ignore[arg-type]
            attn_metadata.block_tables,
            kv_seq_lens,
            query.dtype,
            layer,
            profile_label="turboquant_decode",
        )
        batch_size = len(kv_seq_lens)
        actual_seq_lengths_q = torch.ones(batch_size, dtype=torch.int32, device=query.device).cumsum(0).tolist()
        actual_seq_lengths_kv = torch.tensor(kv_seq_lens, dtype=torch.int32, device=query.device).cumsum(0).tolist()
        output = self._run_dense_fia(
            query[:batch_size],
            dense_k,
            dense_v,
            actual_seq_lengths_q,
            actual_seq_lengths_kv,
            attn_metadata,
            output,
            profile_name="turboquant_decode.run_dense_fia",
        )
        _maybe_sync_for_profile(output)
        _record_tq_profile(
            "turboquant_decode.total",
            (time.perf_counter() - t_total_0) * 1000.0,
            vectors=batch_size,
        )
        return output

    def _forward_turboquant_fused_kv_update_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: dict[str, torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        _maybe_sync_for_profile(query, key, value, attn_metadata.block_tables)
        t0 = time.perf_counter()
        self.key_cache = kv_cache  # type: ignore[assignment]
        self._prepare_turboquant_runtime(layer, next(iter(kv_cache.values())).device)

        batch_size = len(attn_metadata.seq_lens_list)
        query_start_loc = attn_metadata.query_start_loc[: batch_size + 1].to(
            device=query.device, dtype=torch.int32,
        ).contiguous()
        key_start_loc = query_start_loc
        current_lens = key_start_loc[1:] - key_start_loc[:-1]
        seq_lens_t = torch.tensor(
            attn_metadata.seq_lens_list,
            dtype=torch.int32,
            device=query.device,
        )
        old_seq_lens = torch.clamp(seq_lens_t - current_lens, min=0).contiguous()
        num_tokens = int(attn_metadata.actual_seq_lengths_q[-1])

        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            tq_encode_kv_to_paged_cache(
                key[:num_tokens],
                value[:num_tokens],
                attn_metadata.slot_mapping[:num_tokens],
                kv_cache,
                layer._tq_k_rot,
                layer._tq_k_codebook,
                layer._tq_k_boundary,
                layer._tq_k_qjl_proj,
                layer._tq_v_rot,
                layer._tq_v_codebook,
                layer._tq_v_boundary,
                k_qjl_proj_t=layer._tq_k_qjl_proj_t,
                k_variant=getattr(layer, "tq_k_variant", "prod"),
                k_total_bits=int(layer.tq_k_total_bits),
                k_stage1_bits=int(layer.tq_k_stage1_bits),
                v_bits=int(layer.tq_v_stage1_bits),
                num_kv_heads=self.num_kv_heads,
                assume_valid_slots=True,
                kv_mse_rotation=getattr(layer, "_tq_kv_mse_rotation", None),
                kv_mse_shared_boundary=getattr(
                    layer, "_tq_kv_mse_shared_boundary", False,
                ),
            )
            output = self._run_dense_fia(
                query[:num_tokens],
                key[:num_tokens],
                value[:num_tokens],
                attn_metadata.actual_seq_lengths_q,
                attn_metadata.actual_seq_lengths_q,
                attn_metadata,
                output,
                profile_name=(
                    "turboquant_fused_kv_update_attention."
                    "prefill_no_cache.run_dense_fia"
                ),
            )
            _maybe_sync_for_profile(output)
            _record_tq_profile(
                "turboquant_fused_kv_update_attention.forward",
                (time.perf_counter() - t0) * 1000.0,
                vectors=num_tokens,
            )
            return output

        if (
            attn_metadata.attn_state == AscendAttentionState.DecodeOnly
            and os.getenv("VLLM_ASCEND_TQ_USE_FUSED_DECODE_DENSE_FIA", "1") == "1"
        ):
            q_ends = [int(v) for v in attn_metadata.actual_seq_lengths_q]
            q_starts = [0] + q_ends[:-1]
            current_lens_list = [
                max(0, q_end - q_start)
                for q_start, q_end in zip(q_starts, q_ends)
            ]
            old_seq_lens_list = [
                max(0, int(total_len) - cur_len)
                for total_len, cur_len in zip(
                    attn_metadata.seq_lens_list,
                    current_lens_list,
                )
            ]

            tq_encode_kv_to_paged_cache(
                key[:num_tokens],
                value[:num_tokens],
                attn_metadata.slot_mapping[:num_tokens],
                kv_cache,
                layer._tq_k_rot,
                layer._tq_k_codebook,
                layer._tq_k_boundary,
                layer._tq_k_qjl_proj,
                layer._tq_v_rot,
                layer._tq_v_codebook,
                layer._tq_v_boundary,
                k_qjl_proj_t=layer._tq_k_qjl_proj_t,
                k_variant=getattr(layer, "tq_k_variant", "prod"),
                k_total_bits=int(layer.tq_k_total_bits),
                k_stage1_bits=int(layer.tq_k_stage1_bits),
                v_bits=int(layer.tq_v_stage1_bits),
                num_kv_heads=self.num_kv_heads,
                assume_valid_slots=True,
                kv_mse_rotation=getattr(layer, "_tq_kv_mse_rotation", None),
                kv_mse_shared_boundary=getattr(
                    layer, "_tq_kv_mse_shared_boundary", False,
                ),
            )

            if (
                decode_compressed_full_cache_enabled()
                and fused_attention_custom_enabled()
                and bool(attn_metadata.causal)
                and getattr(layer, "tq_k_variant", "prod") == "prod"
                and "k_qjl" in kv_cache
                and "k_gamma" in kv_cache
                and all(cur_len == 1 for cur_len in current_lens_list)
            ):
                try:
                    t_compressed_full = time.perf_counter()
                    output = self._run_turboquant_fused_decode_attention(
                        query,
                        kv_cache,
                        attn_metadata.block_tables,
                        attn_metadata.seq_lens_list,
                        output,
                        layer,
                    )
                    _record_tq_profile(
                        (
                            "turboquant_fused_kv_update_attention."
                            "decode.compressed_full_cache.total"
                        ),
                        (time.perf_counter() - t_compressed_full) * 1000.0,
                        vectors=num_tokens,
                    )
                    _record_tq_profile(
                        "turboquant_fused_kv_update_attention.forward",
                        (time.perf_counter() - t0) * 1000.0,
                        vectors=num_tokens,
                    )
                    return output
                except Exception:
                    if (
                        os.getenv("VLLM_ASCEND_TQ_CUSTOM_STRICT", "0") == "1"
                        or attention_debug_compare_enabled()
                    ):
                        raise

            if (
                compressed_decode_current_enabled()
                and bool(attn_metadata.causal)
                and getattr(layer, "tq_k_variant", "prod") == "prod"
                and "k_qjl" in kv_cache
                and "k_gamma" in kv_cache
                and all(cur_len == 1 for cur_len in current_lens_list)
            ):
                try:
                    _maybe_sync_for_profile(query, key, value, kv_cache)
                    t_compressed = time.perf_counter()
                    compressed_out = tq_prod_mse_history_current_decode_attention(
                        query[:num_tokens],
                        key[:num_tokens],
                        value[:num_tokens],
                        kv_cache,
                        attn_metadata.block_tables,
                        old_seq_lens_list,
                        layer._tq_k_codebook,
                        layer._tq_v_codebook,
                        layer._tq_k_rot,
                        layer._tq_k_qjl_proj,
                        layer._tq_v_rot_t,
                        k_total_bits=int(layer.tq_k_total_bits),
                        v_bits=int(layer.tq_v_stage1_bits),
                        head_dim=self.head_size,
                        scale=self.scale,
                        score_dtype=torch.float32,
                        output_dtype=output.dtype,
                        profile_prefix=(
                            "turboquant_fused_kv_update_attention."
                            "decode.compressed"
                        ),
                    )
                    output[:num_tokens] = compressed_out.to(dtype=output.dtype)
                    _maybe_sync_for_profile(output)
                    _record_tq_profile(
                        (
                            "turboquant_fused_kv_update_attention."
                            "decode.compressed.total"
                        ),
                        (time.perf_counter() - t_compressed) * 1000.0,
                        vectors=num_tokens,
                    )
                    _record_tq_profile(
                        "turboquant_fused_kv_update_attention.forward",
                        (time.perf_counter() - t0) * 1000.0,
                        vectors=num_tokens,
                    )
                    return output
                except Exception:
                    if (
                        os.getenv("VLLM_ASCEND_TQ_CUSTOM_STRICT", "0") == "1"
                        or attention_debug_compare_enabled()
                    ):
                        raise

            _maybe_sync_for_profile(kv_cache, attn_metadata.block_tables)
            t_decode = time.perf_counter()
            history_k, history_v = tq_decode_history_to_dense(
                kv_cache,
                attn_metadata.block_tables,
                old_seq_lens_list,
                layer._tq_k_codebook,
                layer._tq_v_codebook,
                layer._tq_k_rot_t,
                layer._tq_k_qjl_proj,
                layer._tq_v_rot_t,
                k_variant=getattr(layer, "tq_k_variant", "prod"),
                k_total_bits=int(layer.tq_k_total_bits),
                k_stage1_bits=int(layer.tq_k_stage1_bits),
                v_bits=int(layer.tq_v_stage1_bits),
                head_dim=self.head_size,
                target_dtype=query.dtype,
            )
            _maybe_sync_for_profile(history_k, history_v)
            _record_tq_profile(
                "turboquant_fused_kv_update_attention.decode.decode_history",
                (time.perf_counter() - t_decode) * 1000.0,
                vectors=sum(old_seq_lens_list) * self.num_kv_heads,
            )

            key_chunks: list[torch.Tensor] = []
            value_chunks: list[torch.Tensor] = []
            hist_cursor = 0
            for q_start, q_end, old_len in zip(
                q_starts,
                q_ends,
                old_seq_lens_list,
            ):
                if old_len > 0:
                    key_chunks.append(history_k[hist_cursor:hist_cursor + old_len])
                    value_chunks.append(history_v[hist_cursor:hist_cursor + old_len])
                    hist_cursor += old_len
                if q_end > q_start:
                    key_chunks.append(key[q_start:q_end].to(query.dtype))
                    value_chunks.append(value[q_start:q_end].to(query.dtype))

            dense_k = torch.cat(key_chunks, dim=0).contiguous()
            dense_v = torch.cat(value_chunks, dim=0).contiguous()
            actual_seq_lengths_kv = []
            running_kv_len = 0
            for old_len, cur_len in zip(old_seq_lens_list, current_lens_list):
                running_kv_len += old_len + cur_len
                actual_seq_lengths_kv.append(running_kv_len)

            output = self._run_dense_fia(
                query[:num_tokens],
                dense_k,
                dense_v,
                attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv,
                attn_metadata,
                output,
                profile_name=(
                    "turboquant_fused_kv_update_attention."
                    "decode.run_dense_fia"
                ),
            )
            _maybe_sync_for_profile(output)
            _record_tq_profile(
                "turboquant_fused_kv_update_attention.forward",
                (time.perf_counter() - t0) * 1000.0,
                vectors=num_tokens,
            )
            return output

        state_to_mode = {
            AscendAttentionState.DecodeOnly: 0,
            AscendAttentionState.ChunkedPrefill: 1,
            AscendAttentionState.PrefillCacheHit: 1,
            AscendAttentionState.PrefillNoCache: 2,
        }
        fused_out = tq_fused_kv_update_attention(
            query[:num_tokens],
            key[:num_tokens],
            value[:num_tokens],
            attn_metadata.slot_mapping[:num_tokens],
            attn_metadata.block_tables,
            old_seq_lens,
            query_start_loc,
            key_start_loc,
            kv_cache,
            layer._tq_k_rot,
            layer._tq_k_qjl_query_matrix,
            layer._tq_k_boundary,
            layer._tq_k_codebook,
            layer._tq_k_qjl_proj,
            layer._tq_v_rot,
            layer._tq_v_rot_t,
            layer._tq_v_boundary,
            layer._tq_v_codebook,
            k_qjl_proj_t=layer._tq_k_qjl_proj_t,
            k_variant=getattr(layer, "tq_k_variant", "prod"),
            k_total_bits=int(layer.tq_k_total_bits),
            k_stage1_bits=int(layer.tq_k_stage1_bits),
            v_bits=int(layer.tq_v_stage1_bits),
            head_dim=self.head_size,
            scale=self.scale,
            causal=bool(attn_metadata.causal),
            output_dtype=output.dtype,
            mode=state_to_mode.get(attn_metadata.attn_state, 1),
            k_rotation_t=layer._tq_k_rot_t,
            kv_mse_rotation=getattr(layer, "_tq_kv_mse_rotation", None),
            kv_mse_shared_boundary=getattr(
                layer, "_tq_kv_mse_shared_boundary", False,
            ),
        )
        output[:num_tokens] = fused_out.to(dtype=output.dtype)
        _maybe_sync_for_profile(output)
        _record_tq_profile(
            "turboquant_fused_kv_update_attention.forward",
            (time.perf_counter() - t0) * 1000.0,
            vectors=num_tokens,
        )
        return output

    def _can_use_turboquant_fused_kv_update_attention(
        self,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        kv_cache,
        attn_metadata: AscendMetadata,
    ) -> bool:
        if not fused_kv_update_attention_enabled():
            return False
        if key is None or value is None or not isinstance(kv_cache, dict):
            return False
        if attn_metadata.model_runner_type == "pooling" and not attn_metadata.causal:
            return False
        if self.attn_type == AttentionType.ENCODER_DECODER:
            return False
        if self.alibi_slopes is not None:
            return False
        if getattr(attn_metadata, "swa_mask", None) is not None:
            return False
        return attn_metadata.attn_state in {
            AscendAttentionState.DecodeOnly,
            AscendAttentionState.ChunkedPrefill,
            AscendAttentionState.PrefillCacheHit,
            AscendAttentionState.PrefillNoCache,
        }

    def _forward_turboquant_chunked_prefill(
        self,
        query: torch.Tensor,
        float_key: torch.Tensor | None,
        float_value: torch.Tensor | None,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        _maybe_sync_for_profile(query, attn_metadata.block_tables)
        t_total_0 = time.perf_counter()
        del float_key, float_value
        dense_k, dense_v = self._dequant_paged_kv_to_dense(
            self.key_cache,  # type: ignore[arg-type]
            attn_metadata.block_tables,
            attn_metadata.seq_lens_list,
            query.dtype,
            layer,
            profile_label="turboquant_chunked_prefill",
        )
        actual_seq_lengths_kv = torch.tensor(
            attn_metadata.seq_lens_list, dtype=torch.int32, device=query.device
        ).cumsum(0).tolist()
        output = self._run_dense_fia(
            query[: attn_metadata.actual_seq_lengths_q[-1]],
            dense_k,
            dense_v,
            attn_metadata.actual_seq_lengths_q,
            actual_seq_lengths_kv,
            attn_metadata,
            output,
            profile_name="turboquant_chunked_prefill.run_dense_fia",
        )
        _maybe_sync_for_profile(output)
        _record_tq_profile(
            "turboquant_chunked_prefill.total",
            (time.perf_counter() - t_total_0) * 1000.0,
            vectors=attn_metadata.actual_seq_lengths_q[-1],
        )
        return output

    def _forward_turboquant_fused_infer_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        layer: AttentionLayer,
    ) -> torch.Tensor:
        _maybe_sync_for_profile(query)
        t_total_0 = time.perf_counter()
        num_tokens = attn_metadata.actual_seq_lengths_q[-1]
        if attn_metadata.attn_state == AscendAttentionState.PrefillNoCache:
            actual_seq_lengths_kv = attn_metadata.actual_seq_lengths_q
            if self.attn_type == AttentionType.ENCODER_DECODER:
                actual_seq_lengths_kv = torch.cumsum(attn_metadata.seq_lens, dim=0).tolist()
            output = self._run_dense_fia(
                query[:num_tokens],
                key[:num_tokens],
                value[:num_tokens],
                attn_metadata.actual_seq_lengths_q,
                actual_seq_lengths_kv,
                attn_metadata,
                output,
                profile_name="turboquant_prefill_no_cache.run_dense_fia",
            )
            _maybe_sync_for_profile(output)
            _record_tq_profile(
                "turboquant_prefill_no_cache.total",
                (time.perf_counter() - t_total_0) * 1000.0,
                vectors=num_tokens,
            )
            return output

        dense_k, dense_v = self._dequant_paged_kv_to_dense(
            self.key_cache,  # type: ignore[arg-type]
            attn_metadata.block_tables,
            attn_metadata.seq_lens_list,
            query.dtype,
            layer,
            profile_label="turboquant_prefill_cache_hit",
        )
        actual_seq_lengths_kv = torch.tensor(
            attn_metadata.seq_lens_list, dtype=torch.int32, device=query.device
        ).cumsum(0).tolist()
        output = self._run_dense_fia(
            query[:num_tokens],
            dense_k,
            dense_v,
            attn_metadata.actual_seq_lengths_q,
            actual_seq_lengths_kv,
            attn_metadata,
            output,
            profile_name="turboquant_prefill_cache_hit.run_dense_fia",
        )
        _maybe_sync_for_profile(output)
        _record_tq_profile(
            "turboquant_prefill_cache_hit.total",
            (time.perf_counter() - t_total_0) * 1000.0,
            vectors=num_tokens,
        )
        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("fused output quantization is not yet supported for AscendTurboQuantAttentionBackendImpl")
        if attn_metadata is None:
            return output.fill_(0)

        float_key, float_value = key, value
        if self._can_use_turboquant_fused_kv_update_attention(
            key, value, kv_cache, attn_metadata,
        ):
            try:
                return self._forward_turboquant_fused_kv_update_attention(
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    layer,
                )
            except Exception:
                if os.getenv("VLLM_ASCEND_TQ_CUSTOM_STRICT", "0") == "1":
                    raise

        if key is not None and value is not None:
            encoded_k, encoded_v = self._quantize_kv_to_turboquant(key, value, layer, attn_metadata.num_actual_tokens)
            self._reshape_and_cache_turboquant(query, encoded_k, encoded_v, kv_cache, attn_metadata)

        if attn_metadata.model_runner_type == "pooling" and not attn_metadata.causal:
            attn_output = self._forward_encoder_attention(query, float_key, float_value, attn_metadata, output)
            output[: query.shape[0]] = attn_output[: query.shape[0]]
            return output

        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            return self._forward_turboquant_decode(query, attn_metadata, output, layer)
        if attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
            return self._forward_turboquant_chunked_prefill(query, float_key, float_value, attn_metadata, output, layer)
        return self._forward_turboquant_fused_infer_attention(query, float_key, float_value, attn_metadata, output, layer)
