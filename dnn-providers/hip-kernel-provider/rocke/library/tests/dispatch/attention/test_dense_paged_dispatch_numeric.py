# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Paged K/V requests served end to end through dispatch on gfx942 and gfx950
(GPU). The host-only dispatcher contract of the same requests is in
``test_dense_paged_dispatch.py``."""

import math

import pytest

from ..._dense_paged_stand_ins import pinned
from dispatch.attention import dispatch_attention

from .test_dense_paged_dispatch import _BS, _CASES, _D, _HKV, _HQ, _SQ, _request


def _check_paged_varlen_dispatch_on_gpu(
    arch, spec_id, lens, *, one_seq_4d=False, q_lens=None
):
    """Sequences of ``lens`` served end to end through dispatch: packed Q, pages
    permuted in a 2x cache whose unused pages and past-length slots are NaN, a
    table padded with garbage ids and a NaN-filled output, against per-sequence
    fp32 references (sliding window unless bottom-right). ``one_seq_4d``: one
    sequence passed as [1, S, Hq, D] without cu_seqlens_q, which the binding
    then builds.
    ``q_lens``: shorter query lengths, requested as bottom-right causal (no
    window); the reference shifts each sequence by its kv_len - q_len."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != arch:
        pytest.skip(f"needs {arch}")
    bottom_right = q_lens is not None
    q_lens = q_lens if bottom_right else lens
    window, seqlen = (0 if bottom_right else 1024), max(lens)
    torch.manual_seed(0)
    total = sum(q_lens)
    cu = torch.tensor([0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32)
    q = torch.randn(total, _HQ, _D, device="cuda", dtype=torch.float16)
    pages = [-(-n // _BS) for n in lens]
    width = -(-seqlen // _BS)
    num_blocks = 2 * sum(pages)
    phys = torch.randperm(num_blocks, device="cuda")[: sum(pages)]
    k_cache = torch.full(
        (num_blocks, _BS, _HKV, _D), float("nan"), device="cuda", dtype=torch.float16
    )
    v_cache = torch.full_like(k_cache, float("nan"))
    table = torch.full(
        (len(lens), width), 0x7FFF_FFFF, device="cuda", dtype=torch.int32
    )
    ks, vs, first = [], [], 0
    for b, n in enumerate(lens):
        ids = phys[first : first + pages[b]]
        first += pages[b]
        table[b, : pages[b]] = ids.to(torch.int32)
        k = torch.randn(n, _HKV, _D, device="cuda", dtype=torch.float16)
        v = torch.randn(n, _HKV, _D, device="cuda", dtype=torch.float16)
        for p in range(pages[b]):
            rows = slice(p * _BS, min(n, (p + 1) * _BS))
            k_cache[ids[p], : rows.stop - rows.start] = k[rows]
            v_cache[ids[p], : rows.stop - rows.start] = v[rows]
        ks.append(k)
        vs.append(v)
    # Rows past the packed output catch a tail store the kernel's guard missed.
    out_buf = torch.full(
        (total + 256, _HQ, _D), float("nan"), device="cuda", dtype=torch.float16
    )
    out = out_buf[:total]
    tensors = {
        "q": q,
        "k": k_cache,
        "v": v_cache,
        "out": out,
        "block_tables": table,
        "kv_lens": torch.tensor(lens, device="cuda", dtype=torch.int32),
        "cu_seqlens_q": cu.to("cuda"),
    }
    if one_seq_4d:
        tensors["q"], tensors["out"] = q.unsqueeze(0), out.unsqueeze(0)
        del tensors["cu_seqlens_q"]
    req = _request(
        arch,
        batch=len(lens),
        seqlen_q=max(q_lens),
        seqlen_k=seqlen,
        sliding_window=window,
        mask_type=2 if bottom_right else 1,
    )
    result = dispatch_attention(pinned(req, spec_id))
    assert result.spec.kernel_spec.varlen
    assert result.spec.kernel_spec.causal_bottom_right == bottom_right
    result.bind_torch(tensors).launch()
    torch.cuda.synchronize()
    assert torch.isnan(out_buf[total:]).all(), "store past the packed output"

    rep = _HQ // _HKV
    for b, n in enumerate(lens):
        if q_lens[b] == 0:
            continue
        rows = slice(int(cu[b]), int(cu[b + 1]))
        qi = torch.arange(q_lens[b], device="cuda").view(-1, 1) + (n - q_lens[b])
        ki = torch.arange(n, device="cuda").view(1, -1)
        keep = ki <= qi
        if window:
            keep &= ki > qi - window
        ref = torch.nn.functional.scaled_dot_product_attention(
            q[rows].float().transpose(0, 1),
            ks[b].float().transpose(0, 1).repeat_interleave(rep, 0),
            vs[b].float().transpose(0, 1).repeat_interleave(rep, 0),
            attn_mask=keep,
            scale=1.0 / math.sqrt(_D),
        ).transpose(0, 1)
        assert torch.isfinite(out[rows]).all(), f"sequence {b}"
        assert (out[rows].float() - ref).abs().max().item() < 2e-2, f"sequence {b}"


@pytest.mark.gpu
@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_multi_seq_paged_dispatch_matches_the_reference_on_gpu(arch, spec_id):
    _check_paged_varlen_dispatch_on_gpu(arch, spec_id, [_SQ, 1000, 17])


@pytest.mark.gpu
@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_bottom_right_paged_dispatch_matches_the_reference_on_gpu(arch, spec_id):
    """Prefix-cache style sequences: queries shorter than their keys (shift 0, mid
    tile and large), and an empty query over cached keys."""
    _check_paged_varlen_dispatch_on_gpu(
        arch, spec_id, [_SQ, 1000, 300, 17], q_lens=[300, 1000, 0, 17]
    )


@pytest.mark.gpu
@pytest.mark.parametrize("arch,spec_id", _CASES)
def test_one_sequence_bottom_right_dispatch_matches_the_reference_on_gpu(arch, spec_id):
    _check_paged_varlen_dispatch_on_gpu(
        arch, spec_id, [_SQ], q_lens=[512], one_seq_4d=True
    )


@pytest.mark.gpu
@pytest.mark.parametrize("arch,spec_id", _CASES)
@pytest.mark.parametrize(
    "seqlen", [1000, 1088], ids=["partial_kv_tile", "partial_q_tile"]
)
def test_one_sequence_partial_tile_dispatch_matches_the_reference_on_gpu(
    arch, spec_id, seqlen
):
    _check_paged_varlen_dispatch_on_gpu(arch, spec_id, [seqlen], one_seq_4d=True)
