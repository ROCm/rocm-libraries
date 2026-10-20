# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-GPU guardrail: the gfx942 4-warp GQA kernel must not read Q past its last token.

Each CTA covers a fixed 128-row (fold: 32-token) query tile, so the last partial q-block spans
rows beyond the sequence. Those rows are never stored, but their Q loads have to stay in bounds.
An over-read only faults when it leaves Q's allocation and the pages behind it are unmapped.
That is only reliable if Q's allocation is the first one of a fresh process and ends exactly at
its segment's end, so each case runs in its own spawned process (all of them at once).
"""

from __future__ import annotations

import concurrent.futures as cf
import math
import multiprocessing as mp

import pytest

from test_gfx942_gqa_head_fold_numeric import (
    _load_windowed_harness,
    requires_gfx942_gpu,
)

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
    """Prefill through default dispatch (D256 causal, D128 sliding window)."""
    import torch

    H = _load_windowed_harness()
    if H is None:
        return "skip"
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
    ref = H.run_reference(s, data)
    q_tail = q_tail.view(data["query"].shape)
    q_tail.copy_(data["query"])
    data["query"] = q_tail
    out, _ = H.run_unified("rocke", s, data)
    torch.cuda.synchronize()  # an out-of-bounds Q read surfaces here as a HIP error
    return H.compare(ref, out)["max_abs"]


def _pinned_decode(case):
    """The explicit tuning spec on decode, as an external tuning harness launches it."""
    import torch
    from dispatch.attention import AttentionRequest, dispatch_attention

    B, hq = case["B"], case["hq"]
    hk, d, bs, skv = 2, 128, 16, 512
    dt = torch.bfloat16
    q = _tail_buffer(B * hq * d, dt).view(B, hq, d)
    q.copy_(torch.randn(B, hq, d, dtype=dt, device="cuda") * 0.3)
    k = torch.randn(B * skv // bs, bs, hk, d, dtype=dt, device="cuda") * 0.3
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    cu = torch.arange(B + 1, dtype=torch.int32, device="cuda")
    lens = torch.full((B,), skv, dtype=torch.int32, device="cuda")
    table = torch.arange(B * skv // bs, dtype=torch.int32, device="cuda").reshape(B, -1)
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
    kc = (
        k.reshape(B, skv, hk, d)
        .permute(0, 2, 1, 3)
        .float()
        .repeat_interleave(hq // hk, dim=1)
    )
    vc = (
        v.reshape(B, skv, hk, d)
        .permute(0, 2, 1, 3)
        .float()
        .repeat_interleave(hq // hk, dim=1)
    )
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float().unsqueeze(2), kc, vc
    )
    return (out.float() - ref.squeeze(2)).abs().max().item()


# Chunked prefill (Sq < Skv), with Skv a multiple of both KV tile widths (32 and 64): the q-block
# is partial and the last KV tile is not.
PREFILL = [
    dict(
        id="d256_causal_partial",
        seq_lens=[(240, 512)],
        hq=8,
        hk=2,
        d=256,
        bs=16,
        window=None,
    ),
    dict(
        id="d128_swa_fold_partial",
        seq_lens=[(240, 512)],
        hq=8,
        hk=2,
        d=128,
        bs=16,
        window=64,
    ),
    dict(
        id="d128_swa_fold_ragged",
        seq_lens=[(100, 128), (240, 512)],
        hq=8,
        hk=2,
        d=128,
        bs=16,
        window=64,
    ),
    dict(
        id="d128_swa_nonfold_bs64",
        seq_lens=[(240, 512)],
        hq=8,
        hk=2,
        d=128,
        bs=64,
        window=64,
    ),
]

# Decode shapes that faulted under explicit-spec tuning (D128, bs16).
DECODE = [dict(id="b1_hq16", B=1, hq=16), dict(id="b4_hq16", B=4, hq=16)]


@pytest.fixture(scope="module")
def results():
    """Start every case in its own fresh process up front; the tests collect the results."""
    ctx = mp.get_context("spawn")  # fork after the parent touched the GPU is unsafe
    pools, futures = [], {}
    for fn, cases in ((_prefill, PREFILL), (_pinned_decode, DECODE)):
        for case in cases:
            pool = cf.ProcessPoolExecutor(max_workers=1, mp_context=ctx)
            pools.append(pool)
            futures[(fn.__name__, case["id"])] = pool.submit(fn, case)
    yield futures
    for pool in pools:
        pool.shutdown(wait=True, cancel_futures=True)


def _max_abs(results, fn, case):
    # A fault in the worker re-raises here; a hard abort raises BrokenProcessPool.
    value = results[(fn.__name__, case["id"])].result()
    if value == "skip":
        pytest.skip("windowed parity harness not present in this checkout")
    return value


@requires_gfx942_gpu
@pytest.mark.parametrize("case", PREFILL, ids=lambda c: c["id"])
def test_prefill_tail_q_in_bounds(case, results):
    assert _max_abs(results, _prefill, case) <= TOL


@requires_gfx942_gpu
@pytest.mark.parametrize("case", DECODE, ids=lambda c: c["id"])
def test_pinned_4warp_decode_tail_q_in_bounds(case, results):
    assert _max_abs(results, _pinned_decode, case) <= TOL


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
