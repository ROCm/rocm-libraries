# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-GPU guard for the gfx942 4-warp bodies on odd sequence lengths.

The lean natural-QK body (head_size 192 and 256 bf16 causal prefill) walks 64-key
tiles and 128-row Q blocks; the D128 body (sliding-window prefill, and any D128
shape the 4-warp tuning spec is pinned to, decode included) walks 32- or 64-key
tiles and 128-row Q blocks (32 tokens when head-folded). When a sequence length is
not a tile multiple, the last tile covers keys and rows past the sequence. A body
must not read block-table entries, K/V rows or Q rows for them: an
out-of-sequence block-table entry holds an arbitrary page id, and the K/V load built
from it faults. A one-token decode sequence reads 127 Q rows past its token without
the Q clamp.

Each block-table row here carries extra columns filled with a page id whose K/V
offset lies about 2 GiB past the cache, so any read past a sequence's last page
faults deterministically instead of depending on what the allocator left there.
Output is checked against an FP64 numpy paged reference; the launches run on
``rocke.runtime`` device arrays. The structural CPU twins are
``TestLeanBodyBoundsReads`` and ``TestD128BodyBoundsReads`` in
``test_d256_gfx942_fast_routing.py``.
"""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest
from kernels.common import attention_unified as au
from rocke.runtime.device_array import DeviceArray
from rocke.runtime.hip_module import get_device_num_cus
from rocke.tests.gpu import current_device_arch_in

from ._attention_paged_reference import (
    decode_half,
    encode_half,
    paged_attention_reference,
    round_to_half,
)
from ._attention_tiled_launch import launch_unified

_4WARP_SPEC = "gfx942_u2d_gfx942_4warp_nw1_mw32_t64_llvm"


@pytest.fixture
def gfx942():
    current_device_arch_in(("gfx942",), "the gfx942 4-warp bounds check")


_PAD_COLS = 8


def _paged_inputs(seqs, hq, hkv, d, bs, dt, seed):
    """Random Q/K/V (float32 values exact in ``dt``), a permuted block table
    whose padding columns hold the poison page id, and the sequence lengths."""
    rng = np.random.default_rng(seed)
    pages = [-(-k // bs) for _, k in seqs]
    nblocks = sum(pages) + 3

    def values(*shape):
        return round_to_half(rng.standard_normal(shape, dtype=np.float32), dt)

    q = values(sum(s for s, _ in seqs), hq, d)
    kc = values(nblocks, bs, hkv, d)
    vc = values(nblocks, bs, hkv, d)
    perm = rng.permutation(nblocks).astype(np.int32)
    poison = (1 << 30) // (bs * hkv * d)
    bt = np.full((len(seqs), max(pages) + _PAD_COLS), poison, np.int32)
    for row, n, start in zip(bt, pages, itertools.accumulate([0, *pages])):
        row[:n] = perm[start : start + n]
    return SimpleNamespace(
        q=q, k=kc, v=vc, bt=bt, pages=pages, nblocks=nblocks, dtype=dt
    )


def _problem(x, seqs, hq, hkv, d, bs, window):
    return au.UnifiedAttentionProblem(
        total_q=sum(s for s, _ in seqs),
        num_seqs=len(seqs),
        num_query_heads=hq,
        num_kv_heads=hkv,
        head_size=d,
        block_size=bs,
        max_seqlen_q=max(s for s, _ in seqs),
        max_seqlen_k=max(k for _, k in seqs),
        dtype=x.dtype,
        sliding_window=window,
        num_cus=get_device_num_cus(),
        num_kv_blocks=x.nblocks,
    )


def _launch_production(x, seqs, hq, hkv, d, bs, window):
    """Launch through the production route; return O and the kernel name it
    builds."""
    from kernels.gfx942.attention_tiled_2d import build_gfx942_4warp_gqa

    problem = _problem(x, seqs, hq, hkv, d, bs, window)
    assert (
        au._gfx942_4warp_route(problem, "gfx942") is not None
    ), "case must exercise a 4-warp body"
    spec = au._tiled_spec_from_problem(problem, "gfx942")
    out = launch_unified(
        problem,
        x.q,
        x.k,
        x.v,
        x.bt,
        1.0 / math.sqrt(d),
        q_lens=[s for s, _ in seqs],
        kv_lens=[k for _, k in seqs],
        backend="auto",
    )
    return out, build_gfx942_4warp_gqa(spec, arch="gfx942").name


def _launch_pinned(x, seqs, hq, hkv, d, bs, window):
    """Launch the 4-warp tuning spec through the dispatcher (the only way to reach
    a 4-warp body on decode); return O and the kernel name it builds."""
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
            dtype=x.dtype,
            kv_block_size=bs,
            algorithm="unified_tuning",
            spec_id=_4WARP_SPEC,
        )
    )

    def device(values):
        return DeviceArray(encode_half(values, x.dtype), x.dtype)

    out = DeviceArray(encode_half(np.full(x.q.shape, np.nan), x.dtype), x.dtype)
    tensors = {
        "q": device(x.q),
        "out": out,
        "k": device(x.k),
        "v": device(x.v),
        "cu_seqlens_q": DeviceArray(
            np.array([0, *itertools.accumulate(s for s, _ in seqs)], np.int32)
        ),
        "seqused_k": DeviceArray(np.array([k for _, k in seqs], np.int32)),
        "block_table": DeviceArray(x.bt),
    }
    result.bind_torch(tensors).launch()
    return decode_half(out.values(), x.dtype), result.build().name


def _max_abs_vs_reference(out, x, seqs, d, window):
    """max |out - ref| over all sequences against the FP64 paged reference with
    the bottom-right causal (and sliding-window) mask; asserts the output is
    finite."""
    worst = 0.0
    starts = itertools.accumulate([0, *(s for s, _ in seqs)])
    for i, ((sq, skv), st) in enumerate(zip(seqs, starts)):
        ref = paged_attention_reference(
            x.q[st : st + sq],
            x.k,
            x.v,
            skv,
            x.bt[i, : x.pages[i]],
            1.0 / math.sqrt(d),
            sliding_window=window,
        )
        o = out[st : st + sq]
        assert np.isfinite(o).all(), f"seq {i}: non-finite output"
        worst = max(worst, float(np.abs(o - ref).max()))
    return worst


def _run(seqs, hq, hkv, d, bs, dt="bf16", window=0, pinned=False, seed=0):
    """Launch one case; return (max_abs vs the FP64 reference, kernel name)."""
    x = _paged_inputs(seqs, hq, hkv, d, bs, dt, seed)
    launch = _launch_pinned if pinned else _launch_production
    out, name = launch(x, seqs, hq, hkv, d, bs, window)
    return _max_abs_vs_reference(out, x, seqs, d, window), name


@pytest.mark.parametrize("d", [192, 256])
@pytest.mark.parametrize(
    "seqs,hq,hkv,bs",
    [
        ([(129, 129)], 16, 2, 16),
        ([(129, 129)], 16, 2, 32),
        ([(2, 2)], 12, 2, 16),
        ([(339, 367)], 40, 8, 16),
        ([(513, 513), (77, 2001), (1023, 1023)], 16, 16, 16),
    ],
)
def test_lean_body_odd_lengths(gfx942, d, seqs, hq, hkv, bs):
    assert _run(seqs, hq, hkv, d, bs)[0] <= 5e-2


@pytest.mark.parametrize(
    "seqs",
    [[(129, 129)], [(339, 367)], [(513, 513), (77, 2001), (1023, 1023)]],
)
@pytest.mark.parametrize(
    "dt,hq,hkv,bs,suffix",
    [
        ("bf16", 32, 8, 16, "_4wgqa_fold"),
        ("fp16", 32, 8, 16, "_4wgqa"),
        ("bf16", 64, 8, 32, "_4wgqa"),
        ("bf16", 32, 8, 64, "_4wgqa"),
    ],
)
def test_d128_swa_odd_lengths(gfx942, seqs, dt, hq, hkv, bs, suffix):
    worst, name = _run(seqs, hq, hkv, 128, bs, dt=dt, window=64)
    assert name.endswith(suffix), name
    assert worst <= 5e-2


@pytest.mark.parametrize(
    "hq,hkv,window,suffix", [(64, 4, 0, "_4wgqa"), (32, 8, 128, "_4wgqa_fold")]
)
def test_d128_decode_pinned(gfx942, hq, hkv, window, suffix):
    """One query token per sequence, KV lengths in (4064, 4096) that are not tile
    multiples: the gfx942 decode tuning shape that faulted (B=64, Hq=64, Hkv=4)."""
    seqs = [(1, 4095 - (7 * i) % 31) for i in range(64)]
    worst, name = _run(seqs, hq, hkv, 128, 16, window=window, pinned=True)
    assert name.endswith(suffix), name
    assert worst <= 5e-2
