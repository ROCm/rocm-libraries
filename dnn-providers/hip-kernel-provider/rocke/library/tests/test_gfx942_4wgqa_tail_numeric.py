# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-GPU guardrail: the gfx942 4-warp GQA kernel must not read outside a sequence's Q and K/V.

Q: each CTA covers a fixed 128-row (fold: 32-token) query tile, so the last partial q-block spans
rows beyond the sequence. Those rows are never stored, but their Q loads have to stay in bounds.
An over-read only faults when it leaves Q's allocation and the pages behind it are unmapped.
That is only reliable if Q's allocation is the first one of a fresh process and ends exactly at
its segment's end, so each case runs in its own spawned process (all of them at once).

K/V: the last KV tile spans keys past the sequence. They are masked to P = 0, but 0 * NaN is NaN,
so their K/V has to come from the sequence's own blocks. The K/V cases surround each sequence
with hostile data: NaN in the cache slots after its last key, and a block table padded with a
NaN block or with a block id far outside the cache.
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
FAR_BLOCK = 100_000  # outside any cache here, but within i32 element addressing


def _tail_buffer(numel, dtype):
    """Empty tensor of ``numel`` elements at the end of its own allocation (>= 10 MiB gets its own
    segment, rounded up to 2 MiB). Allocate it before anything else."""
    import torch

    esize = torch.empty((), dtype=dtype).element_size()
    seg = max(12 << 20, (numel * esize + (2 << 20) - 1) // (2 << 20) * (2 << 20))
    buf = torch.empty(seg // esize, dtype=dtype, device="cuda")
    return buf[buf.numel() - numel :]


def _own_blocks(kv_lens, bs, num_blocks):
    """Block table giving each sequence its own blocks; the cache's last block stays unused."""
    import torch

    need = [(kl + bs - 1) // bs for kl in kv_lens]
    perm = torch.randperm(num_blocks - 1, device="cuda").int()
    table = torch.zeros(len(kv_lens), max(need), dtype=torch.int32, device="cuda")
    start = 0
    for i, n in enumerate(need):
        table[i, :n] = perm[start : start + n]
        start += n
    return table


def _poison(k, v, table, kv_lens, bs, mode):
    """Return the block table padded with hostile entries, after making the cache around each
    sequence hostile (see the module docstring)."""
    import torch

    spare = k.shape[0] - 1
    if mode == "nan":
        k[spare] = float("nan")
        v[spare] = float("nan")
        for i, kl in enumerate(kv_lens):
            if kl % bs:
                blk = int(table[i, (kl - 1) // bs])
                k[blk, kl % bs :] = float("nan")
                v[blk, kl % bs :] = float("nan")
    pad = spare if mode == "nan" else FAR_BLOCK
    extra = torch.full((table.shape[0], 8), pad, dtype=table.dtype, device=table.device)
    return torch.cat([table, extra], 1).contiguous()


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
    kv_lens = [x[1] for x in s.seq_lens]
    poison = case.get("kv_poison")
    if poison:
        data["block_tables"] = _own_blocks(kv_lens, s.block_size, s.num_blocks)
    ref = H.run_reference(s, data)
    if poison:
        data["block_tables"] = _poison(
            data["key_cache"],
            data["value_cache"],
            data["block_tables"],
            kv_lens,
            s.block_size,
            poison,
        )
    q_tail = q_tail.view(data["query"].shape)
    q_tail.copy_(data["query"])
    data["query"] = q_tail
    out, _ = H.run_unified("rocke", s, data)
    torch.cuda.synchronize()  # an out-of-bounds read surfaces here as a HIP error
    return H.compare(ref, out)["max_abs"]


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
    table = _own_blocks([skv] * B, bs, k.shape[0])

    def seq(cache):
        return cache[table.long()].reshape(B, bps * bs, hk, d)[:, :skv]

    kc = seq(k).permute(0, 2, 1, 3).float().repeat_interleave(hq // hk, dim=1)
    vc = seq(v).permute(0, 2, 1, 3).float().repeat_interleave(hq // hk, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.float().unsqueeze(2), kc, vc
    )
    if case.get("kv_poison"):
        table = _poison(k, v, table, [skv] * B, bs, case["kv_poison"])
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


# Q-tail cases are chunked prefill (Sq < Skv) with Skv a multiple of both KV tile widths (32 and
# 64), so only the q-block is partial. K/V-tail cases use 232 keys: not a multiple of the block
# size or of either tile width.
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
def test_prefill_tail_reads_in_bounds(case, results):
    assert _max_abs(results, _prefill, case) <= TOL


@requires_gfx942_gpu
@pytest.mark.parametrize("case", DECODE, ids=lambda c: c["id"])
def test_pinned_4warp_decode_tail_reads_in_bounds(case, results):
    assert _max_abs(results, _pinned_decode, case) <= TOL


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
