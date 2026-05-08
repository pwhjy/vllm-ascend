import torch

from vllm_ascend.ops.turboquant.fused import (
    old_seq_lens_from_total,
    tq_decode_history_to_dense,
    tq_encode_kv_to_paged_cache,
    tq_encode_kv_to_paged_cache_reference,
    tq_fused_kv_update_attention_reference,
    tq_prod_mse_history_current_decode_attention,
)
from vllm_ascend.quantization.methods.turboquant_layout import (
    get_stage1_bits,
    packed_bytes_per_vector,
)
from vllm_ascend.quantization.methods.turboquant_runtime import (
    build_qjl_projection,
    build_rotation_matrix,
    build_turboquant_codebook,
    turboquant_decode_mse,
    turboquant_decode_prod,
    turboquant_encode_mse,
    turboquant_encode_prod,
)


def _make_sidecar_cache(num_blocks, block_size, num_kv_heads, head_dim, k_total_bits, v_bits):
    k_stage1_bits = get_stage1_bits(k_total_bits, "prod")
    return {
        "k_idx": torch.zeros(
            num_blocks,
            block_size,
            num_kv_heads,
            packed_bytes_per_vector(head_dim, k_stage1_bits),
            dtype=torch.uint8,
        ),
        "k_qjl": torch.zeros(
            num_blocks,
            block_size,
            num_kv_heads,
            packed_bytes_per_vector(head_dim, 1),
            dtype=torch.uint8,
        ),
        "k_gamma": torch.zeros(num_blocks, block_size, num_kv_heads, 1),
        "k_norm": torch.zeros(num_blocks, block_size, num_kv_heads, 1),
        "v_idx": torch.zeros(
            num_blocks,
            block_size,
            num_kv_heads,
            packed_bytes_per_vector(head_dim, v_bits),
            dtype=torch.uint8,
        ),
        "v_norm": torch.zeros(num_blocks, block_size, num_kv_heads, 1),
    }


def _make_params(head_dim, k_total_bits, v_bits):
    k_stage1_bits = get_stage1_bits(k_total_bits, "prod")
    k_codebook, k_boundary = build_turboquant_codebook(head_dim, k_stage1_bits, "cpu", torch.float32)
    v_codebook, v_boundary = build_turboquant_codebook(head_dim, v_bits, "cpu", torch.float32)
    k_rotation = build_rotation_matrix(head_dim, 1234, "cpu", torch.float32)
    v_rotation = build_rotation_matrix(head_dim, 5678, "cpu", torch.float32)
    k_qjl_proj = build_qjl_projection(head_dim, 9012, "cpu", torch.float32)
    return {
        "k_stage1_bits": k_stage1_bits,
        "k_codebook": k_codebook,
        "k_boundary": k_boundary,
        "v_codebook": v_codebook,
        "v_boundary": v_boundary,
        "k_rotation": k_rotation,
        "v_rotation": v_rotation,
        "k_qjl_proj": k_qjl_proj,
        "k_qjl_query_matrix": (k_rotation @ k_qjl_proj.transpose(0, 1)).contiguous(),
    }


def _flat(cache_tensor, num_kv_heads):
    return cache_tensor.view(-1, num_kv_heads, cache_tensor.shape[-1])


def test_old_seq_lens_are_explicit_history_lengths():
    assert old_seq_lens_from_total([4, 7], [0, 2, 5]) == [2, 4]


def test_encode_kv_to_paged_cache_matches_runtime_encode():
    torch.manual_seed(0)
    head_dim = 8
    num_kv_heads = 2
    k_total_bits = 3
    v_bits = 2
    params = _make_params(head_dim, k_total_bits, v_bits)
    cache = _make_sidecar_cache(2, 4, num_kv_heads, head_dim, k_total_bits, v_bits)
    key = torch.randn(3, num_kv_heads, head_dim)
    value = torch.randn(3, num_kv_heads, head_dim)
    slots = torch.tensor([0, 3, 5], dtype=torch.int64)

    tq_encode_kv_to_paged_cache_reference(
        key,
        value,
        slots,
        cache,
        params["k_rotation"],
        params["k_codebook"],
        params["k_boundary"],
        params["k_qjl_proj"],
        params["v_rotation"],
        params["v_codebook"],
        params["v_boundary"],
        k_variant="prod",
        k_total_bits=k_total_bits,
        k_stage1_bits=params["k_stage1_bits"],
        v_bits=v_bits,
        num_kv_heads=num_kv_heads,
    )

    encoded_k = turboquant_encode_prod(
        key,
        params["k_rotation"],
        params["k_codebook"],
        params["k_boundary"],
        params["k_qjl_proj"],
        k_total_bits,
    )
    encoded_v = turboquant_encode_mse(
        value,
        params["v_rotation"],
        params["v_codebook"],
        params["v_boundary"],
        v_bits,
    )

    assert torch.equal(_flat(cache["k_idx"], num_kv_heads)[slots], encoded_k["idx"])
    assert torch.equal(_flat(cache["k_qjl"], num_kv_heads)[slots], encoded_k["qjl"])
    assert torch.allclose(_flat(cache["k_gamma"], num_kv_heads)[slots], encoded_k["gamma"])
    assert torch.allclose(_flat(cache["k_norm"], num_kv_heads)[slots], encoded_k["norm"])
    assert torch.equal(_flat(cache["v_idx"], num_kv_heads)[slots], encoded_v["idx"])
    assert torch.allclose(_flat(cache["v_norm"], num_kv_heads)[slots], encoded_v["norm"])


def test_encode_kv_to_paged_cache_ignores_padding_slots():
    torch.manual_seed(3)
    head_dim = 8
    num_kv_heads = 2
    k_total_bits = 3
    v_bits = 2
    params = _make_params(head_dim, k_total_bits, v_bits)
    cache = _make_sidecar_cache(2, 4, num_kv_heads, head_dim, k_total_bits, v_bits)
    key = torch.randn(3, num_kv_heads, head_dim)
    value = torch.randn(3, num_kv_heads, head_dim)
    slots = torch.tensor([0, -1, 5], dtype=torch.int64)

    tq_encode_kv_to_paged_cache_reference(
        key,
        value,
        slots,
        cache,
        params["k_rotation"],
        params["k_codebook"],
        params["k_boundary"],
        params["k_qjl_proj"],
        params["v_rotation"],
        params["v_codebook"],
        params["v_boundary"],
        k_variant="prod",
        k_total_bits=k_total_bits,
        k_stage1_bits=params["k_stage1_bits"],
        v_bits=v_bits,
        num_kv_heads=num_kv_heads,
    )

    valid = slots >= 0
    valid_slots = slots[valid]
    encoded_k = turboquant_encode_prod(
        key[valid],
        params["k_rotation"],
        params["k_codebook"],
        params["k_boundary"],
        params["k_qjl_proj"],
        k_total_bits,
    )
    encoded_v = turboquant_encode_mse(
        value[valid],
        params["v_rotation"],
        params["v_codebook"],
        params["v_boundary"],
        v_bits,
    )

    assert torch.equal(_flat(cache["k_idx"], num_kv_heads)[valid_slots], encoded_k["idx"])
    assert torch.equal(_flat(cache["k_qjl"], num_kv_heads)[valid_slots], encoded_k["qjl"])
    assert torch.allclose(_flat(cache["k_gamma"], num_kv_heads)[valid_slots], encoded_k["gamma"])
    assert torch.allclose(_flat(cache["k_norm"], num_kv_heads)[valid_slots], encoded_k["norm"])
    assert torch.equal(_flat(cache["v_idx"], num_kv_heads)[valid_slots], encoded_v["idx"])
    assert torch.allclose(_flat(cache["v_norm"], num_kv_heads)[valid_slots], encoded_v["norm"])
    assert torch.count_nonzero(_flat(cache["k_idx"], num_kv_heads)[-1]) == 0
    assert torch.count_nonzero(_flat(cache["v_idx"], num_kv_heads)[-1]) == 0


def test_encode_kv_to_paged_cache_dispatch_falls_back_to_reference(monkeypatch):
    monkeypatch.delenv("VLLM_ASCEND_TQ_USE_CUSTOM_ENCODE_CACHE_UPDATE", raising=False)
    torch.manual_seed(4)
    head_dim = 8
    num_kv_heads = 2
    k_total_bits = 3
    v_bits = 2
    params = _make_params(head_dim, k_total_bits, v_bits)
    cache = _make_sidecar_cache(2, 4, num_kv_heads, head_dim, k_total_bits, v_bits)
    ref_cache = _make_sidecar_cache(2, 4, num_kv_heads, head_dim, k_total_bits, v_bits)
    key = torch.randn(3, num_kv_heads, head_dim)
    value = torch.randn(3, num_kv_heads, head_dim)
    slots = torch.tensor([0, 3, 5], dtype=torch.int64)

    tq_encode_kv_to_paged_cache(
        key,
        value,
        slots,
        cache,
        params["k_rotation"],
        params["k_codebook"],
        params["k_boundary"],
        params["k_qjl_proj"],
        params["v_rotation"],
        params["v_codebook"],
        params["v_boundary"],
        k_variant="prod",
        k_total_bits=k_total_bits,
        k_stage1_bits=params["k_stage1_bits"],
        v_bits=v_bits,
        num_kv_heads=num_kv_heads,
    )
    tq_encode_kv_to_paged_cache_reference(
        key,
        value,
        slots,
        ref_cache,
        params["k_rotation"],
        params["k_codebook"],
        params["k_boundary"],
        params["k_qjl_proj"],
        params["v_rotation"],
        params["v_codebook"],
        params["v_boundary"],
        k_variant="prod",
        k_total_bits=k_total_bits,
        k_stage1_bits=params["k_stage1_bits"],
        v_bits=v_bits,
        num_kv_heads=num_kv_heads,
    )

    for name, tensor in cache.items():
        assert torch.equal(tensor, ref_cache[name])


def test_fused_kv_update_attention_uses_dense_current_chunk_and_updates_cache():
    torch.manual_seed(1)
    head_dim = 8
    num_heads = 4
    num_kv_heads = 2
    q_per_kv = num_heads // num_kv_heads
    k_total_bits = 3
    v_bits = 2
    params = _make_params(head_dim, k_total_bits, v_bits)
    cache = _make_sidecar_cache(1, 4, num_kv_heads, head_dim, k_total_bits, v_bits)

    history_key = torch.randn(2, num_kv_heads, head_dim)
    history_value = torch.randn(2, num_kv_heads, head_dim)
    current_key = torch.randn(2, num_kv_heads, head_dim)
    current_value = torch.randn(2, num_kv_heads, head_dim)
    query = torch.randn(2, num_heads, head_dim)

    tq_encode_kv_to_paged_cache_reference(
        history_key,
        history_value,
        torch.tensor([0, 1], dtype=torch.int64),
        cache,
        params["k_rotation"],
        params["k_codebook"],
        params["k_boundary"],
        params["k_qjl_proj"],
        params["v_rotation"],
        params["v_codebook"],
        params["v_boundary"],
        k_variant="prod",
        k_total_bits=k_total_bits,
        k_stage1_bits=params["k_stage1_bits"],
        v_bits=v_bits,
        num_kv_heads=num_kv_heads,
    )

    out = tq_fused_kv_update_attention_reference(
        query,
        current_key,
        current_value,
        torch.tensor([2, 3], dtype=torch.int64),
        torch.tensor([[0]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
        torch.tensor([0, 2], dtype=torch.int32),
        cache,
        params["k_rotation"],
        params["k_qjl_query_matrix"],
        params["k_boundary"],
        params["k_codebook"],
        params["k_qjl_proj"],
        params["v_rotation"],
        params["v_rotation"].transpose(0, 1).contiguous(),
        params["v_boundary"],
        params["v_codebook"],
        k_variant="prod",
        k_total_bits=k_total_bits,
        k_stage1_bits=params["k_stage1_bits"],
        v_bits=v_bits,
        head_dim=head_dim,
        scale=1.0 / (head_dim**0.5),
    )

    flat_k_idx = _flat(cache["k_idx"], num_kv_heads)
    flat_k_qjl = _flat(cache["k_qjl"], num_kv_heads)
    flat_k_gamma = _flat(cache["k_gamma"], num_kv_heads)
    flat_k_norm = _flat(cache["k_norm"], num_kv_heads)
    flat_v_idx = _flat(cache["v_idx"], num_kv_heads)
    flat_v_norm = _flat(cache["v_norm"], num_kv_heads)

    hist_k = turboquant_decode_prod(
        flat_k_idx[:2],
        flat_k_qjl[:2],
        flat_k_gamma[:2],
        flat_k_norm[:2],
        params["k_rotation"].transpose(0, 1).contiguous(),
        params["k_codebook"],
        params["k_qjl_proj"],
        k_total_bits,
        head_dim,
        torch.float32,
    )
    hist_v = turboquant_decode_mse(
        flat_v_idx[:2],
        flat_v_norm[:2],
        params["v_rotation"].transpose(0, 1).contiguous(),
        params["v_codebook"],
        v_bits,
        head_dim,
        torch.float32,
    )

    all_k = torch.cat([hist_k, current_key], dim=0)
    all_v = torch.cat([hist_v, current_value], dim=0)
    expanded_k = all_k.repeat_interleave(q_per_kv, dim=1)
    expanded_v = all_v.repeat_interleave(q_per_kv, dim=1)
    scores = torch.einsum("qhd,shd->qhs", query, expanded_k) / (head_dim**0.5)
    allowed = torch.arange(4).unsqueeze(0)
    current_positions = 2 + torch.arange(2).unsqueeze(1)
    scores = scores.masked_fill((allowed > current_positions).unsqueeze(1), float("-inf"))
    ref = torch.einsum("qhs,shd->qhd", torch.softmax(scores, dim=-1), expanded_v)

    current_encoded = turboquant_encode_prod(
        current_key,
        params["k_rotation"],
        params["k_codebook"],
        params["k_boundary"],
        params["k_qjl_proj"],
        k_total_bits,
    )
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
    assert torch.equal(flat_k_idx[2:4], current_encoded["idx"])


def test_fused_kv_update_attention_single_token_decode_matches_dense_reference():
    torch.manual_seed(5)
    head_dim = 8
    num_heads = 4
    num_kv_heads = 2
    q_per_kv = num_heads // num_kv_heads
    k_total_bits = 3
    v_bits = 2
    params = _make_params(head_dim, k_total_bits, v_bits)
    cache = _make_sidecar_cache(1, 4, num_kv_heads, head_dim, k_total_bits, v_bits)

    history_key = torch.randn(2, num_kv_heads, head_dim)
    history_value = torch.randn(2, num_kv_heads, head_dim)
    current_key = torch.randn(1, num_kv_heads, head_dim)
    current_value = torch.randn(1, num_kv_heads, head_dim)
    query = torch.randn(1, num_heads, head_dim)

    tq_encode_kv_to_paged_cache_reference(
        history_key,
        history_value,
        torch.tensor([0, 1], dtype=torch.int64),
        cache,
        params["k_rotation"],
        params["k_codebook"],
        params["k_boundary"],
        params["k_qjl_proj"],
        params["v_rotation"],
        params["v_codebook"],
        params["v_boundary"],
        k_variant="prod",
        k_total_bits=k_total_bits,
        k_stage1_bits=params["k_stage1_bits"],
        v_bits=v_bits,
        num_kv_heads=num_kv_heads,
    )

    out = tq_fused_kv_update_attention_reference(
        query,
        current_key,
        current_value,
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([[0]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        cache,
        params["k_rotation"],
        params["k_qjl_query_matrix"],
        params["k_boundary"],
        params["k_codebook"],
        params["k_qjl_proj"],
        params["v_rotation"],
        params["v_rotation"].transpose(0, 1).contiguous(),
        params["v_boundary"],
        params["v_codebook"],
        k_variant="prod",
        k_total_bits=k_total_bits,
        k_stage1_bits=params["k_stage1_bits"],
        v_bits=v_bits,
        head_dim=head_dim,
        scale=1.0 / (head_dim**0.5),
    )

    flat_k_idx = _flat(cache["k_idx"], num_kv_heads)
    flat_k_qjl = _flat(cache["k_qjl"], num_kv_heads)
    flat_k_gamma = _flat(cache["k_gamma"], num_kv_heads)
    flat_k_norm = _flat(cache["k_norm"], num_kv_heads)
    flat_v_idx = _flat(cache["v_idx"], num_kv_heads)
    flat_v_norm = _flat(cache["v_norm"], num_kv_heads)

    hist_k = turboquant_decode_prod(
        flat_k_idx[:2],
        flat_k_qjl[:2],
        flat_k_gamma[:2],
        flat_k_norm[:2],
        params["k_rotation"].transpose(0, 1).contiguous(),
        params["k_codebook"],
        params["k_qjl_proj"],
        k_total_bits,
        head_dim,
        torch.float32,
    )
    hist_v = turboquant_decode_mse(
        flat_v_idx[:2],
        flat_v_norm[:2],
        params["v_rotation"].transpose(0, 1).contiguous(),
        params["v_codebook"],
        v_bits,
        head_dim,
        torch.float32,
    )

    all_k = torch.cat([hist_k, current_key], dim=0)
    all_v = torch.cat([hist_v, current_value], dim=0)
    expanded_k = all_k.repeat_interleave(q_per_kv, dim=1)
    expanded_v = all_v.repeat_interleave(q_per_kv, dim=1)
    scores = torch.einsum("qhd,shd->qhs", query, expanded_k) / (head_dim**0.5)
    ref = torch.einsum("qhs,shd->qhd", torch.softmax(scores, dim=-1), expanded_v)

    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_compressed_history_current_decode_attention_matches_dense_reference():
    torch.manual_seed(6)
    head_dim = 16
    num_heads = 4
    num_kv_heads = 2
    q_per_kv = num_heads // num_kv_heads
    k_total_bits = 3
    v_bits = 2
    params = _make_params(head_dim, k_total_bits, v_bits)
    cache = _make_sidecar_cache(2, 4, num_kv_heads, head_dim, k_total_bits, v_bits)

    history_key = torch.randn(3, num_kv_heads, head_dim)
    history_value = torch.randn(3, num_kv_heads, head_dim)
    current_key = torch.randn(2, num_kv_heads, head_dim)
    current_value = torch.randn(2, num_kv_heads, head_dim)
    query = torch.randn(2, num_heads, head_dim)
    block_table = torch.tensor([[0], [1]], dtype=torch.int32)
    old_seq_lens = [2, 1]

    tq_encode_kv_to_paged_cache_reference(
        history_key,
        history_value,
        torch.tensor([0, 1, 4], dtype=torch.int64),
        cache,
        params["k_rotation"],
        params["k_codebook"],
        params["k_boundary"],
        params["k_qjl_proj"],
        params["v_rotation"],
        params["v_codebook"],
        params["v_boundary"],
        k_variant="prod",
        k_total_bits=k_total_bits,
        k_stage1_bits=params["k_stage1_bits"],
        v_bits=v_bits,
        num_kv_heads=num_kv_heads,
    )

    out = tq_prod_mse_history_current_decode_attention(
        query,
        current_key,
        current_value,
        cache,
        block_table,
        old_seq_lens,
        params["k_codebook"],
        params["v_codebook"],
        params["k_rotation"],
        params["k_qjl_proj"],
        params["v_rotation"].transpose(0, 1).contiguous(),
        k_total_bits=k_total_bits,
        v_bits=v_bits,
        head_dim=head_dim,
        scale=1.0 / (head_dim**0.5),
    )

    hist_k, hist_v = tq_decode_history_to_dense(
        cache,
        block_table,
        old_seq_lens,
        params["k_codebook"],
        params["v_codebook"],
        params["k_rotation"].transpose(0, 1).contiguous(),
        params["k_qjl_proj"],
        params["v_rotation"].transpose(0, 1).contiguous(),
        k_variant="prod",
        k_total_bits=k_total_bits,
        k_stage1_bits=params["k_stage1_bits"],
        v_bits=v_bits,
        head_dim=head_dim,
        target_dtype=torch.float32,
    )

    ref = torch.empty_like(out)
    hist_cursor = 0
    for batch_idx, old_len in enumerate(old_seq_lens):
        all_k = torch.cat(
            [
                hist_k[hist_cursor:hist_cursor + old_len],
                current_key[batch_idx:batch_idx + 1],
            ],
            dim=0,
        )
        all_v = torch.cat(
            [
                hist_v[hist_cursor:hist_cursor + old_len],
                current_value[batch_idx:batch_idx + 1],
            ],
            dim=0,
        )
        hist_cursor += old_len
        expanded_k = all_k.repeat_interleave(q_per_kv, dim=1)
        expanded_v = all_v.repeat_interleave(q_per_kv, dim=1)
        scores = (
            torch.einsum("hd,shd->hs", query[batch_idx], expanded_k)
            / (head_dim**0.5)
        )
        ref[batch_idx] = torch.einsum(
            "hs,shd->hd",
            torch.softmax(scores, dim=-1),
            expanded_v,
        )

    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)
