# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-GPU guardrail: the gfx942 4-warp GQA kernel must not read outside a sequence's Q and K/V.

Q: each CTA covers a fixed query tile, so the last partial q-block spans rows beyond the sequence. Those rows are never stored, but their Q loads have to stay in bounds.
An over-read only faults when it leaves Q's allocation and the pages behind it are unmapped.
That is only reliable if Q's allocation is the first one of a fresh process and ends exactly at
its segment's end, so each case runs in its own spawned process (all of them at once).

K/V: the K/V cases surround each sequence with hostile cache data (``_paged_kv_tail``).
"""

from __future__ import annotations

import math
import os

import pytest

from _gpu_gate import requires_gpu
from _paged_kv_tail import own_blocks, poison, result, spawn_results
from test_gfx942_gqa_head_fold_numeric import _load_windowed_harness

requires_gfx942_gpu = requires_gpu("gfx942")

TOL = 4e-2  # repo-wide bf16 attention gate
FOUR_WARP_SPEC_ID = "gfx942_u2d_gfx942_4warp_nw1_mw32_t64_llvm"


def _tail_buffer(numel, dtype):
    """Empty tensor of ``numel`` elements at the end of its own allocation (>= 10 MiB gets its own
    segment, rounded up to 2 MiB). Allocate it before anything else."""
    import torch

    esize = torch.empty((), dtype=dtype).element_size()
    seg = max(12 << 20, (numel * esize + (2 << 20) - 1) // (2 << 20) * (2 << 20))
    buf = torch.empty(seg // esize, dtype=dtype, device="cuda")
    return buf[buf.numel() - numel :]


def _prefill(case):
    """Prefill through default dispatch; the cases' shapes route to the 4-warp kernel."""
    import torch

    H = _load_windowed_harness()
    s = H.Scenario(
        name=case["id"],
        seq_lens=[tuple(x) for x in case["seq_lens"]],
        num_query_heads=case["hq"],
        num_kv_heads=case["hk"],
        head_size=case["d"],
        block_size=case["bs"],
        dtype=torch.bfloat16,
        sliding_window=case["window"],
        num_blocks=256,
    )
    total_q = sum(x[0] for x in s.seq_lens)
    q_tail = _tail_buffer(total_q * s.num_query_heads * s.head_size, s.dtype)
    data = H.make_inputs(s, seed=0)
    kv_lens = [x[1] for x in s.seq_lens]
    kv_poison = case.get("kv_poison")
    if kv_poison:
        data["block_tables"] = own_blocks(kv_lens, s.block_size, s.num_blocks)
    ref = H.run_reference(s, data)
    if kv_poison:
        data["block_tables"] = poison(
            data["key_cache"],
            data["value_cache"],
            data["block_tables"],
            kv_lens,
            s.block_size,
            kv_poison,
        )
    q_tail = q_tail.view(data["query"].shape)
    q_tail.copy_(data["query"])
    data["query"] = q_tail
    out, _ = H.run_unified("rocke", s, data)
    torch.cuda.synchronize()  # an out-of-bounds read surfaces here as a HIP error
    _assert_ran_4warp_kernel(case)
    return H.compare(ref, out)["max_abs"]


def _assert_ran_4warp_kernel(case):
    """Default dispatch ran the 4-warp kernel the case is named after; a routing change would
    otherwise move the case to another kernel and it would still pass. Each case runs in a fresh
    process, so the kernel caches hold only the kernels it launched."""
    from kernels.common import attention_unified as au

    names = [name for _, name in au._ATTN_TILED_CACHE.values()]
    assert (
        names and not au._ATTN_3D_TILED_CACHE
    ), f"2D {names}, 3D {au._ATTN_3D_TILED_CACHE}"
    for name in names:
        assert "_4wgqa" in name, name
        assert name.endswith("_fold") == ("_fold" in case["id"]), name


def _pinned_decode(case):
    """The explicit tuning spec on decode, as an external tuning harness launches it."""
    import torch
    from dispatch.attention import AttentionRequest, dispatch_attention

    B, hq, skv = case["B"], case["hq"], case["skv"]
    hk, d, bs = 2, 128, 16
    dt = torch.bfloat16
    q = _tail_buffer(B * hq * d, dt).view(B, hq, d)
    q.copy_(torch.randn(B, hq, d, dtype=dt, device="cuda") * 0.3)
    bps = (skv + bs - 1) // bs
    k = torch.randn(B * bps + 1, bs, hk, d, dtype=dt, device="cuda") * 0.3
    v = torch.randn_like(k)
    table = own_blocks([skv] * B, bs, k.shape[0])

    def seq(cache):
        return cache[table.long()].reshape(B, bps * bs, hk, d)[:, :skv]

    kc = seq(k).permute(0, 2, 1, 3).float().repeat_interleave(hq // hk, dim=1)
    vc = seq(v).permute(0, 2, 1, 3).float().repeat_interleave(hq // hk, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float().unsqueeze(2), kc, vc
    )
    if case.get("kv_poison"):
        table = poison(k, v, table, [skv] * B, bs, case["kv_poison"])
    out = torch.empty_like(q)
    cu = torch.arange(B + 1, dtype=torch.int32, device="cuda")
    lens = torch.full((B,), skv, dtype=torch.int32, device="cuda")
    req = AttentionRequest(
        batch=B,
        nhead_q=hq,
        nhead_k=hk,
        seqlen_q=1,
        seqlen_k=skv,
        hdim_q=d,
        hdim_v=d,
        arch="gfx942",
        dtype="bf16",
        mask_type=0,
        kv_block_size=bs,
        algorithm="unified_tuning",
        spec_id=FOUR_WARP_SPEC_ID,
    )
    tensors = dict(
        q=q, k=k, v=v, out=out, cu_seqlens_q=cu, seqused_k=lens, block_table=table
    )
    dispatch_attention(req).bind_torch(tensors, softmax_scale=1 / math.sqrt(d)).launch()
    torch.cuda.synchronize()
    return (out.float() - ref.squeeze(2)).abs().max().item()


def _prefill_case(id, seq_lens, d, bs, window, kv_poison=None):
    return dict(
        id=id,
        seq_lens=seq_lens,
        hq=8,
        hk=2,
        d=d,
        bs=bs,
        window=window,
        kv_poison=kv_poison,
    )


# Q-tail cases are chunked prefill (Sq < Skv) with Skv a multiple of the kernel's KV tile widths,
# so only the q-block is partial. K/V-tail cases use 232 keys: not a multiple of the block size
# or of a tile width.
PREFILL = [
    _prefill_case("d256_causal_q_tail", [(240, 512)], 256, 16, None),
    _prefill_case("d128_swa_fold_q_tail", [(240, 512)], 128, 16, 64),
    _prefill_case("d128_swa_fold_ragged_q_tail", [(100, 128), (240, 512)], 128, 16, 64),
    _prefill_case("d128_swa_nonfold_bs64_q_tail", [(240, 512)], 128, 64, 64),
    _prefill_case("d256_causal_kv_nan", [(232, 232)], 256, 16, None, "nan"),
    _prefill_case("d256_causal_kv_far", [(232, 232)], 256, 16, None, "far"),
    _prefill_case("d128_swa_fold_kv_nan", [(232, 232)], 128, 16, 64, "nan"),
    _prefill_case("d128_swa_fold_kv_far", [(232, 232)], 128, 16, 64, "far"),
    _prefill_case(
        "d128_swa_fold_ragged_kv_nan", [(100, 100), (232, 232)], 128, 16, 64, "nan"
    ),
    _prefill_case("d128_swa_nonfold_bs64_kv_nan", [(232, 232)], 128, 64, 64, "nan"),
]

# Decode shapes that faulted under explicit-spec tuning (D128, bs16), then K/V tails.
DECODE = [
    dict(id="b1_hq16_q_tail", B=1, hq=16, skv=512),
    dict(id="b4_hq16_q_tail", B=4, hq=16, skv=512),
    dict(id="b4_hq16_kv_nan", B=4, hq=16, skv=232, kv_poison="nan"),
    dict(id="b4_hq16_kv_far", B=4, hq=16, skv=232, kv_poison="far"),
]


def _without_expandable_segments(conf):
    # The Q cases need Q's allocation to end at its segment's end; expandable segments map more
    # memory behind it, and the over-read would no longer fault.
    keep = [
        o
        for o in (conf or "").split(",")
        if o and not o.startswith("expandable_segments")
    ]
    return ",".join(keep + ["expandable_segments:False"])


@pytest.fixture(scope="module")
def results(request):
    """Start each selected case in its own fresh process up front; the tests collect the results."""
    worker = {
        "test_prefill_tail_reads_in_bounds": _prefill,
        "test_pinned_4warp_decode_tail_reads_in_bounds": _pinned_decode,
    }
    selected = [
        (worker[item.originalname], item.callspec.params["case"])
        for item in request.session.items
        if item.module is request.module and item.originalname in worker
    ]
    if _load_windowed_harness() is None:
        selected = [(fn, case) for fn, case in selected if fn is not _prefill]
    env = {
        name: _without_expandable_segments(os.environ.get(name))
        for name in ("PYTORCH_HIP_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
    }
    yield from spawn_results(
        {(fn.__name__, case["id"]): (fn, case) for fn, case in selected}, env
    )


def _max_abs(results, fn, case):
    key = (fn.__name__, case["id"])
    if key not in results:
        pytest.skip("windowed parity harness not present in this checkout")
    return result(results, key)


@requires_gfx942_gpu
@pytest.mark.parametrize("case", PREFILL, ids=lambda c: c["id"])
def test_prefill_tail_reads_in_bounds(case, results):
    assert _max_abs(results, _prefill, case) <= TOL


@requires_gfx942_gpu
@pytest.mark.parametrize("case", DECODE, ids=lambda c: c["id"])
def test_pinned_4warp_decode_tail_reads_in_bounds(case, results):
    assert _max_abs(results, _pinned_decode, case) <= TOL


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
