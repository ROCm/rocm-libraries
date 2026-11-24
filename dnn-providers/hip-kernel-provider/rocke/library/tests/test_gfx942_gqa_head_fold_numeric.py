# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-GPU numeric guardrail for the gfx942 GQA head-fold (D128 sliding-window).

The head-fold packs the ``num_queries_per_kv`` query heads that share a KV head
into one workgroup's 128-row M-tile (row m -> token m//4, head m%4), so the paged
K/V is loaded once per KV head instead of once per query head. It rewrites the
device-side ``(token, head)`` index math, the folded head index ``kv_head*4 + m%4``
and the launch grid. Its structure/selection is covered by the CPU emit test
(``test_gfx942_gqa_head_fold.py``); this file guards the one thing a host test
cannot: that the folded kernel still computes the *right numbers*.

A spec/emit test cannot catch a numeric regression in the emitted kernel -- that
is exactly what let #9198 (the D128 ring slot-reuse bug) ship. The guardrail below
launches the production kernel at unit-variance (randn) magnitude -- NOT the
uniform_(-0.1, 0.1) range whose near-uniform softmax hides such bugs -- and asserts
max_abs against a float64 NumPy windowed paged-attention oracle. If a future edit
flips the (token, head) packing, gets the folded head index or grid wrong, or drops
the window mask, this fails. The test needs no torch.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from kernels import UnifiedAttentionProblem
from kernels.common import attention_unified as au
from rocke.runtime.hip_module import get_device_arch

from . import _attention_numpy as an
from ._attention_device_array import DeviceArray

_ARCH = get_device_arch()
requires_gfx942_gpu = pytest.mark.skipif(
    _ARCH != "gfx942", reason=f"needs a gfx942 GPU, found {_ARCH}"
)

_NUM_BLOCKS = 32768


@pytest.fixture(autouse=True)
def _eager_launches(monkeypatch):
    """Graph replay of short launches needs torch; run them eagerly."""
    monkeypatch.setenv("HIPDNN_GFX942_2D_GRAPH", "0")
    monkeypatch.setenv("HIPDNN_GFX942_3D_GRAPH", "0")


def _fold_problem(dtype="bf16", sq=4096, hq=32, hk=8, d=128, bs=32, window=4096):
    return UnifiedAttentionProblem(
        total_q=sq,
        num_seqs=1,
        num_query_heads=hq,
        num_kv_heads=hk,
        head_size=d,
        block_size=bs,
        max_seqlen_q=sq,
        max_seqlen_k=sq,
        dtype=dtype,
        sliding_window=window,
    )


# ---------------------------------------------------------------------------
# Routing anchor (no GPU): the numeric cohort below must build the fold kernel,
# so a future cohort change that stops routing D128-SWA-GQA-4:1-bf16 to the fold
# turns this file's numeric guardrail into a no-op -- and this assert red first.
# ---------------------------------------------------------------------------


@pytest.fixture
def gfx942():
    old_arch = au._RESOLVED_ATTENTION_ARCH
    au._RESOLVED_ATTENTION_ARCH = "gfx942"
    try:
        yield
    finally:
        au._RESOLVED_ATTENTION_ARCH = old_arch
        au._2D_LAUNCH_META.clear()


@pytest.mark.parametrize("bs", [16, 32])
def test_numeric_cohort_routes_to_fold(gfx942, bs):
    from kernels.common.attention_unified import gfx942_gqa_fold_eligible
    from kernels.gfx942.attention_tiled_2d import build_gfx942_4warp_gqa
    from rocke.core.ir_print import print_ir

    p = _fold_problem(bs=bs)
    assert gfx942_gqa_fold_eligible(
        p.head_size,
        p.num_queries_per_kv,
        p.sliding_window,
        p.dtype,
        p.block_size,
    ), "D128 SWA GQA-4:1 bf16 must be fold-eligible"
    spec = au._tiled_spec_from_problem(p, "gfx942")
    ir = print_ir(build_gfx942_4warp_gqa(spec, arch="gfx942"))
    assert re.search(r"_4wgqa_fold\b", ir), "cohort must build the fold kernel"


# ---------------------------------------------------------------------------
# On-GPU numeric guardrail -- the check a spec/emit test cannot make
# ---------------------------------------------------------------------------


@requires_gfx942_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("bs", [16, 32])
@pytest.mark.parametrize("sq,window", [(2048, 512), (8192, 4096)])
@pytest.mark.parametrize("hq,hk", [(32, 8)])
def test_fold_numeric_vs_fp32_windowed_oracle(sq, hq, hk, window, bs):
    """Launch the production fold kernel at randn magnitude and assert max_abs
    against the float64 windowed paged-attn oracle.

    The window is deliberately < seqlen so keys older than ``window`` are
    actually clipped -- a window >= kv_len degenerates to full causal and would
    pass even with a broken SWA lower bound. These cases exercise the window
    boundary the fold cohort is named for (an off-by-one in the drop-out-of-
    window mask fails here, not just a packing bug).

    ``bs`` matches the block sizes the routing anchor above asserts are
    fold-eligible, so everything the predicate ships is numerically checked.
    The two are NOT redundant: BPT = BN // BS is the number of paged KV blocks
    consumed per 32-key tile, so bs=32 walks one block-table entry per tile and
    bs=16 walks two -- a different gather, and a stride/index error in the
    second entry is invisible at bs=32.
    """
    assert window < sq, "window must be < seqlen to exercise SWA masking"

    d = 128
    rng = np.random.default_rng(0)
    pages_used = sq // bs
    pages = rng.choice(_NUM_BLOCKS, size=pages_used, replace=False).astype(np.int32)
    rt = an.runtime()
    k_cache, v_cache, k_host, v_host = an.scattered_paged_cache(
        rt, rng, _NUM_BLOCKS, bs, hk, d, "bf16", pages
    )
    q = an.randn(rng, (sq, hq, d), "bf16")
    query = an.to_device(rt, q, "bf16")
    out = an.empty_device(rt, q.shape, "bf16")
    block_table = DeviceArray(rt, pages.reshape(1, pages_used))
    scale = d**-0.5
    problem = UnifiedAttentionProblem(
        total_q=sq,
        num_seqs=1,
        num_query_heads=hq,
        num_kv_heads=hk,
        head_size=d,
        block_size=bs,  # fold requires block_size <= 32
        max_seqlen_q=sq,
        max_seqlen_k=sq,
        dtype="bf16",
        sliding_window=window,
        num_cus=120,
        num_kv_blocks=_NUM_BLOCKS,
    )
    au.run_unified_attention_torch(
        problem=problem,
        q=query,
        k=k_cache,
        v=v_cache,
        out=out,
        cu_seqlens_q=DeviceArray(rt, np.array([0, sq], dtype=np.int32)),
        seqused_k=DeviceArray(rt, np.array([sq], dtype=np.int32)),
        softmax_scale=scale,
        block_table=block_table,
        softcap=0.0,
        backend="auto",
    )
    ref = an.paged_attention_reference(
        q,
        k_host,
        v_host,
        np.arange(pages_used, dtype=np.int32).reshape(1, -1),
        [sq],
        [sq],
        scale=scale,
        window=window,
    )
    # 4e-2 is the repo-wide bf16 attention gate (ALGORITHM.md §10, the gfx942
    # attention README, parity_unified_attention.py, both dense numeric tests).
    # The fold repacks which M-tile row holds which (token, head) pair; it does
    # not change the MFMA accumulation order or the fp32 accumulator, so it must
    # not need more headroom than the unfolded kernel.
    #
    # Measured on gfx942 (ROCm 7.13): max_abs = 0.01562 on ALL four
    # cases -- 2.6x inside this gate. That value is exactly 2^-6, i.e. one bf16
    # ulp for outputs in [2, 4), so the residual is the bf16 rounding of the
    # stored output, not accumulation drift. There is no mechanism here that
    # would need the 6e-2 this file originally asserted.
    an.assert_close(
        an.read(out, "bf16"),
        ref,
        "bf16",
        f"fold D128 SWA {hq}/{hk} S{sq} w{window} bs{bs}",
        atol=4e-2,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
