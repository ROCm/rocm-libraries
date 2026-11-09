# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-GPU guard: the gfx942 4-warp GQA bodies must not read outside a sequence's Q rows, K/V rows
or block-table entries.

The lean natural-QK body (head_size 192 and 256 bf16 causal prefill) walks 64-key
tiles and 128-row Q blocks; the D128 body (sliding-window prefill, and any D128
shape the 4-warp tuning spec is pinned to, decode included) walks 32- or 64-key
tiles and 128-row Q blocks (32 tokens when head-folded). When a sequence length is
not a tile multiple, the last tile covers keys and rows past the sequence.

Q: the rows past the sequence are never stored, but their loads must stay inside Q. An over-read
only faults when it leaves Q's allocation and the pages behind it are unmapped. That is only
reliable if Q is the first allocation of a fresh process and ends exactly at its segment's end, so
every case runs in its own spawned process (``_paged_kv_tail.spawn_results``), which also keeps a
fault from poisoning the other cases' HIP context. A one-token decode sequence reads 127 Q rows past
its token without the Q clamp.

K/V: each sequence is surrounded with hostile cache data (``_paged_kv_tail.poison``): its
block-table entries past its own blocks point to a block far outside the cache ("far", the
default: a read faults) or to a NaN block, with NaN in its last block's unused slots ("nan").
Output is checked against an fp32 reference. The structural CPU twins are
``TestLeanBodyBoundsReads`` and ``TestD128BodyBoundsReads`` in
``test_d256_gfx942_fast_routing.py``.
"""

from __future__ import annotations

import math
import os
from types import SimpleNamespace

import pytest
from kernels.common import attention_unified as au

from _gpu_gate import requires_gpu
from _paged_kv_tail import TOL, own_blocks, poison, result, spawn_selected

_4WARP_SPEC = "gfx942_u2d_gfx942_4warp_nw1_mw32_t64_llvm"


requires_gfx942_gpu = requires_gpu("gfx942")


def _tail_buffer(numel, dtype):
    """Empty tensor of ``numel`` elements at the end of its own allocation (>= 10 MiB gets its own
    segment, rounded up to 2 MiB). Allocate it before anything else."""
    import torch

    esize = torch.empty((), dtype=dtype).element_size()
    seg = max(12 << 20, (numel * esize + (2 << 20) - 1) // (2 << 20) * (2 << 20))
    buf = torch.empty(seg // esize, dtype=dtype, device="cuda")
    return buf[buf.numel() - numel :]


def _paged_inputs(c):
    """Random Q (at the end of its allocation) and K/V, each sequence's own blocks in a poisoned
    block table, sequence lengths, and a NaN-filled output."""
    import torch

    tdt = {"bf16": torch.bfloat16, "fp16": torch.float16}[c["dt"]]
    seqs, hq, hkv, d, bs = c["seqs"], c["hq"], c["hkv"], c["d"], c["bs"]
    total_q = sum(s for s, _ in seqs)
    q = _tail_buffer(total_q * hq * d, tdt).view(total_q, hq, d)
    torch.manual_seed(c["seed"])  # also seeds own_blocks' permutation
    q.copy_(torch.randn(total_q, hq, d, device="cuda", dtype=tdt))
    kv_lens = [k for _, k in seqs]
    pages = [-(-k // bs) for k in kv_lens]
    nblocks = sum(pages) + 3
    kc = torch.randn(nblocks, bs, hkv, d, device="cuda", dtype=tdt)
    vc = torch.randn(nblocks, bs, hkv, d, device="cuda", dtype=tdt)
    bt = own_blocks(kv_lens, bs, nblocks)
    bt = poison(kc, vc, bt, kv_lens, bs, c["poison"])
    cu = torch.tensor(
        [0] + [s for s, _ in seqs], dtype=torch.int32, device="cuda"
    ).cumsum(0, dtype=torch.int32)
    kl = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    return SimpleNamespace(
        q=q,
        k=kc,
        v=vc,
        bt=bt,
        cu=cu,
        kl=kl,
        pages=pages,
        nblocks=nblocks,
        out=torch.full_like(q, float("nan")),
    )


def _launch_production(x, seqs, hq, hkv, d, bs, dt, window):
    """Launch through the production route; return the name of the kernel that ran (the case runs
    in a fresh process, so the kernel caches hold only what it launched)."""
    import torch

    problem = au.UnifiedAttentionProblem(
        total_q=sum(s for s, _ in seqs),
        num_seqs=len(seqs),
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=d,
        block_size=bs,
        max_seqlen_q=max(s for s, _ in seqs),
        max_seqlen_k=max(k for _, k in seqs),
        dtype=dt,
        sliding_window=window,
        num_cus=torch.cuda.get_device_properties(0).multi_processor_count,
        num_kv_blocks=x.nblocks,
    )
    assert (
        au._gfx942_4warp_route(problem, "gfx942") is not None
    ), "case must exercise a 4-warp body"
    au.run_unified_attention_torch(
        problem=problem,
        q=x.q,
        k=x.k,
        v=x.v,
        out=x.out,
        cu_seqlens_q=x.cu,
        seqused_k=x.kl,
        softmax_scale=1.0 / math.sqrt(d),
        block_table=x.bt,
        softcap=0.0,
        backend="auto",
    )
    names = {name for _, name in au._ATTN_TILED_CACHE.values()}
    assert (
        len(names) == 1 and not au._ATTN_3D_TILED_CACHE
    ), f"2D {names}, 3D {au._ATTN_3D_TILED_CACHE}"
    return names.pop()


def _launch_pinned(x, seqs, hq, hkv, d, bs, dt, window):
    """Launch the 4-warp tuning spec through the dispatcher (the only way to reach
    a 4-warp body on decode); return the kernel name it builds."""
    from dispatch.attention import AttentionRequest, dispatch_attention

    assert len({s for s, _ in seqs}) == 1, "pinned cases use one query length"
    result = dispatch_attention(
        AttentionRequest(
            batch=len(seqs),
            nhead_q=hq,
            nhead_k=hkv,
            seqlen_q=seqs[0][0],
            seqlen_k=max(k for _, k in seqs),
            hdim_q=d,
            hdim_v=d,
            arch="gfx942",
            mask_type=2,
            sliding_window=window,
            dtype=dt,
            kv_block_size=bs,
            algorithm="unified_tuning",
            spec_id=_4WARP_SPEC,
        )
    )
    tensors = {
        "q": x.q,
        "out": x.out,
        "k": x.k,
        "v": x.v,
        "cu_seqlens_q": x.cu,
        "seqused_k": x.kl,
        "block_table": x.bt,
    }
    result.bind_torch(tensors).launch()
    return result.build().name


def _max_abs_vs_reference(x, seqs, hq, hkv, d, window):
    """max |out - ref| over all sequences against an fp32 SDPA reference with the
    bottom-right causal (and sliding-window) mask; asserts the output is finite."""
    import torch
    import torch.nn.functional as F

    rep = hq // hkv
    st = 0
    worst = 0.0
    for i, (sq, skv) in enumerate(seqs):
        idx = x.bt[i, : x.pages[i]].long()
        k = x.k[idx].reshape(-1, hkv, d)[:skv].float().permute(1, 0, 2)
        v = x.v[idx].reshape(-1, hkv, d)[:skv].float().permute(1, 0, 2)
        k = k.repeat_interleave(rep, 0)
        v = v.repeat_interleave(rep, 0)
        qs = x.q[st : st + sq].float().permute(1, 0, 2)
        ii = torch.arange(sq, device="cuda").view(sq, 1) + (skv - sq)
        jj = torch.arange(skv, device="cuda").view(1, skv)
        keep = jj <= ii
        if window:
            keep &= ii - jj < window
        mask = torch.where(keep, 0.0, float("-inf"))
        ref = F.scaled_dot_product_attention(
            qs, k, v, attn_mask=mask, scale=1.0 / math.sqrt(d)
        ).permute(1, 0, 2)
        o = x.out[st : st + sq].float()
        assert torch.isfinite(o).all(), f"seq {i}: non-finite output"
        worst = max(worst, (o - ref).abs().max().item())
        st += sq
    return worst


def _run(c):
    """Launch one case (in its own process); return (max_abs vs the fp32 reference, kernel name)."""
    import torch

    x = _paged_inputs(c)
    launch = _launch_pinned if c["pinned"] else _launch_production
    args = (c["seqs"], c["hq"], c["hkv"], c["d"])
    name = launch(x, *args, c["bs"], c["dt"], c["window"])
    torch.cuda.synchronize()  # an out-of-bounds read surfaces here as a HIP error
    return _max_abs_vs_reference(x, *args, c["window"]), name


def _case(id, seqs, hq, hkv, d, bs, suffix, **kw):
    c = dict(dt="bf16", window=0, pinned=False, poison="far", seed=0)
    c.update(kw, id=id, seqs=seqs, hq=hq, hkv=hkv, d=d, bs=bs, suffix=suffix)
    return c


def _seqs_id(seqs):
    return "_".join(f"{s}x{k}" for s, k in seqs)


# Q-tail cases are chunked prefill (Sq < Skv) with Skv a multiple of the kernel's KV tile widths,
# so only the q-block is partial. Poisoned K/V cases use 232 keys: not a multiple of the block size
# or of a tile width.
LEAN = [
    _case(f"d{d}_{_seqs_id(seqs)}_h{hq}_{hkv}_bs{bs}", seqs, hq, hkv, d, bs, "_4wgqa")
    for d in (192, 256)
    for seqs, hq, hkv, bs in [
        ([(129, 129)], 16, 2, 16),
        ([(129, 129)], 16, 2, 32),
        ([(2, 2)], 12, 2, 16),
        ([(339, 367)], 40, 8, 16),
        ([(513, 513), (77, 2001), (1023, 1023)], 16, 16, 16),
    ]
] + [
    _case("d256_q_tail", [(240, 512)], 8, 2, 256, 16, "_4wgqa"),
    _case("d256_kv_nan", [(232, 232)], 8, 2, 256, 16, "_4wgqa", poison="nan"),
    _case("d256_kv_far", [(232, 232)], 8, 2, 256, 16, "_4wgqa"),
]

D128_SWA = [
    _case(
        f"{_seqs_id(seqs)}_{dt}_h{hq}_{hkv}_bs{bs}",
        seqs,
        hq,
        hkv,
        128,
        bs,
        suffix,
        dt=dt,
        window=64,
    )
    for seqs in ([(129, 129)], [(339, 367)], [(513, 513), (77, 2001), (1023, 1023)])
    for dt, hq, hkv, bs, suffix in [
        ("bf16", 32, 8, 16, "_4wgqa_fold"),
        ("fp16", 32, 8, 16, "_4wgqa"),
        ("bf16", 64, 8, 32, "_4wgqa"),
        ("bf16", 32, 8, 64, "_4wgqa"),
    ]
] + [
    _case(id, seqs, 8, 2, 128, bs, suffix, window=64, poison=mode)
    for id, seqs, bs, suffix, mode in [
        ("fold_q_tail", [(240, 512)], 16, "_4wgqa_fold", "far"),
        ("fold_ragged_q_tail", [(100, 128), (240, 512)], 16, "_4wgqa_fold", "far"),
        ("nonfold_bs64_q_tail", [(240, 512)], 64, "_4wgqa", "far"),
        ("fold_kv_nan", [(232, 232)], 16, "_4wgqa_fold", "nan"),
        ("fold_kv_far", [(232, 232)], 16, "_4wgqa_fold", "far"),
        ("fold_ragged_kv_nan", [(100, 100), (232, 232)], 16, "_4wgqa_fold", "nan"),
        ("nonfold_bs64_kv_nan", [(232, 232)], 64, "_4wgqa", "nan"),
    ]
]

# KV lengths in (4064, 4096) that are not tile multiples: the gfx942 decode tuning shape that
# faulted (B=64, Hq=64, Hkv=4). Then decode shapes whose Q over-read faulted under the pinned spec.
_TUNING_SEQS = [(1, 4095 - (7 * i) % 31) for i in range(64)]
D128_DECODE_PINNED = [
    _case("h64_4", _TUNING_SEQS, 64, 4, 128, 16, "_4wgqa", pinned=True),
    _case(
        "h32_8_w128",
        _TUNING_SEQS,
        32,
        8,
        128,
        16,
        "_4wgqa_fold",
        window=128,
        pinned=True,
    ),
    _case("b1_q_tail", [(1, 512)], 16, 2, 128, 16, "_4wgqa", pinned=True),
    _case("b4_q_tail", [(1, 512)] * 4, 16, 2, 128, 16, "_4wgqa", pinned=True),
    _case(
        "b4_kv_nan", [(1, 232)] * 4, 16, 2, 128, 16, "_4wgqa", pinned=True, poison="nan"
    ),
    _case("b4_kv_far", [(1, 232)] * 4, 16, 2, 128, 16, "_4wgqa", pinned=True),
]


def _without_expandable_segments(conf):
    # Q's allocation must end at its segment's end; expandable segments map more memory behind it,
    # and the over-read would no longer fault.
    keep = [
        o
        for o in (conf or "").split(",")
        if o and not o.startswith("expandable_segments")
    ]
    return ",".join(keep + ["expandable_segments:False"])


@pytest.fixture(scope="module")
def results(request):
    """Start each selected case in its own fresh process up front; the tests collect the results."""
    env = {
        name: _without_expandable_segments(os.environ.get(name))
        for name in ("PYTORCH_HIP_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
    }
    yield from spawn_selected(request, _run, env)


def _check(results, case):
    worst, name = result(results, case["id"])
    assert name.endswith(case["suffix"]), name
    assert worst <= TOL


@requires_gfx942_gpu
@pytest.mark.parametrize("case", LEAN, ids=lambda c: c["id"])
def test_lean_body_reads_in_bounds(case, results):
    _check(results, case)


@requires_gfx942_gpu
@pytest.mark.parametrize("case", D128_SWA, ids=lambda c: c["id"])
def test_d128_swa_reads_in_bounds(case, results):
    _check(results, case)


@requires_gfx942_gpu
@pytest.mark.parametrize("case", D128_DECODE_PINNED, ids=lambda c: c["id"])
def test_d128_decode_pinned_reads_in_bounds(case, results):
    _check(results, case)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
