# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression tests for gfx942 bf16 D128 sharing the sliced-K ring geometry.

Pure Python (no GPU / compilation). #9057 excluded D128 bf16 from the ring
because the ring was attached to the non-ring bf16-wide geometry (nw=2, no
cfvst), which the ring cannot use -- it requires the conflict-free-V store and
the wide nw=4 flash geometry. The fix routes D128 bf16 through the fp16-flash
ring geometry (nw=4, tile=64, cfvst) instead -- verified numerically correct
(max_abs 0.00049, GQA + MHA) on both the Python and C++ engines with the
byte-identity gate GREEN.

These lock in:
  1. D128 bf16 prefill now enables the sliced-K ring (no longer excluded).
  2. The resulting spec is the fp16-flash geometry (nw=4, tile=64, ring, cfvst,
     mask-limit) -- i.e. identical to the fp16 D128 spec modulo dtype.
  3. _get_2d_launch_meta computes the grid/block for the ring geometry (nw=4),
     matching the spec the launcher builds (not the non-ring bf16-wide nw=2).
  4. D64 bf16 and fp16 D128 are unchanged (no regression).
"""

from __future__ import annotations

import dataclasses

import pytest

from kernels import UnifiedAttentionProblem
from kernels.common import attention_unified as au


@pytest.fixture
def gfx942(monkeypatch):
    # Set the memoized arch global directly (same pattern as
    # test_gfx1250_attention.py) rather than monkeypatching the resolver: the
    # resolver caches into _RESOLVED_ATTENTION_ARCH, so replacing only the
    # function leaves the cached global stale and leaks into sibling tests.
    old_arch = au._RESOLVED_ATTENTION_ARCH
    au._RESOLVED_ATTENTION_ARCH = "gfx942"
    # Neutralize any inherited env overrides so we test the default policy.
    # HIPDNN_GFX942_FLASH_WIDE and _K_LDSSEQ are included because they change the
    # ring num_warps / LDS-sequence geometry these tests assert on -- a set value
    # in a dev/CI shell would otherwise make the assertions non-deterministic.
    for var in (
        "HIPDNN_GFX942_K_SLICED_RING",
        "HIPDNN_GFX942_BF16_WIDE",
        "HIPDNN_GFX942_D128_SMALLTILE_DK",
        "HIPDNN_GFX942_FLASH_MLIM",
        "HIPDNN_GFX942_FLASH_WIDE",
        "HIPDNN_GFX942_K_LDSSEQ",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        yield
    finally:
        au._RESOLVED_ATTENTION_ARCH = old_arch
        au._2D_LAUNCH_META.clear()


def _problem(dtype, sq=4096, hq=32, hk=8, d=128, bs=64):
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
    )


@pytest.mark.parametrize("hq,hk", [(32, 8), (16, 16), (64, 8), (128, 8), (64, 4)])
def test_d128_bf16_enables_ring_by_default(gfx942, hq, hk):
    p = _problem("bf16", hq=hq, hk=hk)
    assert au._enable_gfx942_bf16_flash(p), "bf16 D128 prefill should be flash-eligible"
    assert au._enable_gfx942_flash_k_sliced_ring(
        p
    ), "D128 bf16 should now share the sliced-K ring (exclusion removed)"


@pytest.mark.parametrize("hq,hk", [(32, 8), (16, 16)])
def test_d128_bf16_spec_matches_fp16_modulo_dtype(gfx942, hq, hk):
    sb = au._tiled_spec_from_problem(_problem("bf16", hq=hq, hk=hk))
    sf = au._tiled_spec_from_problem(_problem("fp16", hq=hq, hk=hk))
    fb = {f.name: getattr(sb, f.name) for f in dataclasses.fields(sb)}
    ff = {f.name: getattr(sf, f.name) for f in dataclasses.fields(sf)}
    differing = {k for k in fb if fb[k] != ff[k]}
    assert differing == {"dtype"}, f"unexpected spec diffs: {differing}"
    # Sanity: it really is the ring geometry.
    assert sb.num_warps == 4
    assert sb.tile_size == 64
    assert sb.use_k_sliced_ring
    assert sb.use_conflict_free_v_store
    assert sb.block_m_per_warp == 32


@pytest.mark.parametrize("hq,hk", [(32, 8), (16, 16), (64, 8), (128, 8), (64, 4)])
def test_d128_bf16_launch_meta_matches_ring_spec(gfx942, hq, hk):
    au._2D_LAUNCH_META.clear()
    p = _problem("bf16", hq=hq, hk=hk)
    spec = au._tiled_spec_from_problem(p)
    meta = au._get_2d_launch_meta(p, au._tiled_cache_key(p))
    block_m = spec.num_warps * spec.block_m_per_warp
    nqk = hq // hk
    block_q = block_m // nqk if nqk <= block_m else 1
    expected_blocks = p.total_q // block_q + p.num_seqs
    assert meta.grid[1] == expected_blocks
    assert meta.block[0] == 64 * spec.num_warps  # 64*4 == 256, not the wide 64*2


def test_d64_bf16_unchanged(gfx942):
    # D64 already used the ring pre-fix; must stay nw=4 ring.
    s = au._tiled_spec_from_problem(_problem("bf16", d=64, bs=16))
    assert s.use_k_sliced_ring
    assert s.num_warps == 4


def test_fp16_d128_unchanged(gfx942):
    s = au._tiled_spec_from_problem(_problem("fp16"))
    assert s.num_warps == 4
    assert s.tile_size == 64
    assert s.use_k_sliced_ring


def test_short_context_bf16_still_narrow(gfx942):
    # small_q_narrow (q<=768) must NOT take the ring (per _enable_gfx942_bf16_flash).
    p = _problem("bf16", sq=512)
    assert au._enable_gfx942_small_q_narrow(p)
    assert not au._enable_gfx942_flash_k_sliced_ring(p)
