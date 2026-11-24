# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression tests for the gfx942 D128 sliced-K ring routing.

The sliced-K ring stages K in ``k_slice_hd``-wide slices (k_groups = HD/k_slice_hd)
across ``ring_depth`` LDS slots with a kg%depth map. At the 32-wide slice, D128
(k_groups=4) had slice 3 reuse slot 0, and the reusing DMA was unfenced ->
numerically WRONG at realistic magnitude (max_abs ~0.5-1.3, both dtypes; #9198
masked it with a uniform_(-0.1,0.1) oracle whose near-uniform softmax never
exercised the rescale across the stale slot). D64 was k_groups=2 then and never
reused a slot; it now routes to width 16 (k_groups=4) and does.

Two independent fixes are in the ring schedule now: a drain-on-reuse fence (always
applied) and a depth-2 pipeline (ring_depth=2, k%2 -> no reuse). Current routing
(study s34_gfx942_d128_ring_bug_and_perf_recovery):
  * bf16 D128 -> NON-ring T=64. At the production block_size=64 the correct rings
    are all slower than the non-ring flash path; bf16 keeps non-ring.
  * fp16 D128 -> depth-2 ring: correct at magnitude and the fastest correct fp16
    D128 prefill path (~0.72-1.05x AOTriton flash).
  * D64 (both dtypes) -> depth-3 ring, unchanged.

Two layers:
  * Pure-Python spec assertions (always run): per-dtype D128 routing, D64 ring.
  * On-GPU numeric guardrail (skipped without a gfx942 device; needs no torch):
    launch the production kernel at randn magnitude and assert max_abs vs the
    float64 NumPy oracle
    for BOTH dtypes -- so both the bf16 non-ring path and the fp16 depth-2 ring are
    gated on numerics. A spec-only test cannot catch a numerics regression in the
    emitted kernel -- that is exactly what let #9198 through.
"""

from __future__ import annotations

import numpy as np
import pytest
from dispatch.attention.common import AttentionTuningSpec
from kernels import UnifiedAttentionProblem
from kernels.common import attention_unified as au
from rocke.runtime.hip_module import get_device_arch

from . import _attention_numpy as an
from ._attention_device_array import DeviceArray


@pytest.fixture
def gfx942(monkeypatch):
    # Set the memoized arch global directly (same pattern as
    # test_gfx1250_attention.py) rather than monkeypatching the resolver: the
    # resolver caches into _RESOLVED_ATTENTION_ARCH, so replacing only the
    # function leaves the cached global stale and leaks into sibling tests.
    old_arch = au._RESOLVED_ATTENTION_ARCH
    au._RESOLVED_ATTENTION_ARCH = "gfx942"
    # Neutralize any inherited env overrides so we test the default policy.
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


# ---------------------------------------------------------------------------
# Pure-Python spec assertions (no GPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hq,hk", [(32, 8), (16, 16), (64, 8), (128, 8), (64, 4)])
def test_d128_bf16_excluded_from_ring(gfx942, hq, hk):
    # bf16 D128 stays OFF the ring: at the production block_size=64 the correct
    # rings are all slower than the non-ring T=64 flash path, so bf16 keeps
    # non-ring (the faster T=32 ring needs block_size=32, which prod never sends).
    p = _problem("bf16", hq=hq, hk=hk)
    assert not au._enable_gfx942_flash_k_sliced_ring(
        p, "gfx942"
    ), "bf16 D128 must stay on the non-ring path"
    spec = au._tiled_spec_from_problem(p, "gfx942")
    assert not spec.use_k_sliced_ring
    assert spec.tile_size == 64


@pytest.mark.parametrize("hq,hk", [(32, 8), (16, 16), (64, 8), (128, 8), (64, 4)])
def test_d128_fp16_uses_depth2_ring(gfx942, hq, hk):
    # fp16 D128 takes the depth-2 ring: correct at magnitude (the fence + k%2 slot
    # map remove the k_groups=4 reuse hazard) and the fastest correct fp16 D128
    # prefill path (~0.72-1.05x AOTriton flash).
    p = _problem("fp16", hq=hq, hk=hk)
    assert au._enable_gfx942_flash_k_sliced_ring(
        p, "gfx942"
    ), "fp16 D128 should use the (correct) depth-2 ring"
    spec = au._tiled_spec_from_problem(p, "gfx942")
    assert spec.use_k_sliced_ring
    assert spec.ring_depth == 2, "fp16 D128 ring must be depth-2 (k%2, no slot reuse)"
    assert spec.tile_size == 64
    assert spec.num_warps == 4


def test_d128_bf16_ring_force_on_still_off(gfx942, monkeypatch):
    # Even the explicit force-on env must NOT re-enable the bf16 D128 ring: the
    # bf16 head_size==128 guard short-circuits before the env is consulted.
    monkeypatch.setenv("HIPDNN_GFX942_K_SLICED_RING", "1")
    assert not au._enable_gfx942_flash_k_sliced_ring(_problem("bf16"), "gfx942")


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("hq,hk", [(32, 8), (16, 16), (64, 8), (128, 8), (64, 4)])
def test_d128_launch_meta_matches_spec(gfx942, dtype, hq, hk):
    # The launch grid/block must match the geometry the launcher actually builds
    # (ring or non-ring) -- a mismatch silently corrupts output.
    au._2D_LAUNCH_META.clear()
    p = _problem(dtype, hq=hq, hk=hk)
    spec = au._tiled_spec_from_problem(p, "gfx942")
    meta = au._get_2d_launch_meta(p, au._tiled_cache_key(p, "gfx942"), arch="gfx942")
    assert meta.block[0] == 64 * spec.num_warps


def test_d64_ring_depth_unchanged(gfx942):
    # D64 keeps the depth-3 ring and nw4. The slice WIDTH moved to 16 (k_groups
    # 2 -> 4); this deliberately does not pin the width -- test_attn_k_slice_hd.py
    # owns it, and asserting it here too would give two places to update.
    for dtype in ("bf16", "fp16"):
        s = au._tiled_spec_from_problem(_problem(dtype, d=64, bs=16), "gfx942")
        assert s.use_k_sliced_ring, f"D64 {dtype} must keep the ring"
        assert s.ring_depth == 3, f"D64 {dtype} ring stays depth-3"
        assert s.num_warps == 4


# ---------------------------------------------------------------------------
# On-GPU numeric guardrail -- the check #9198 lacked
# ---------------------------------------------------------------------------


_ARCH = get_device_arch()
requires_gfx942_gpu = pytest.mark.skipif(
    _ARCH != "gfx942", reason=f"needs a gfx942 (MI300X) GPU, found {_ARCH}"
)

_BLOCK_SIZE = 64


@pytest.fixture(autouse=True)
def _eager_launches(monkeypatch):
    """Graph replay of short launches needs torch; run them eagerly."""
    monkeypatch.setenv("HIPDNN_GFX942_2D_GRAPH", "0")
    monkeypatch.setenv("HIPDNN_GFX942_3D_GRAPH", "0")


@requires_gfx942_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("dtype,tol", [("bf16", 5e-2), ("fp16", 5e-2)])
@pytest.mark.parametrize("hq,hk", [(32, 8), (16, 16)])
@pytest.mark.parametrize("sq", [512, 1024, 4096])
def test_d128_numeric_vs_fp32_oracle_at_magnitude(monkeypatch, dtype, tol, hq, hk, sq):
    """Launch the production D128 kernel at realistic (unit-variance) magnitude
    and assert max_abs vs the float64 paged-attn oracle.

    This is the guardrail #9198 lacked: with the default uniform_(-0.1, 0.1)
    inputs the ring bug is invisible (softmax ~uniform); randn inputs expose it
    immediately. If a future change re-enables the broken D128 ring, this fails.
    """
    # Force the arch resolver so the production selector picks the gfx942 path.
    old_arch = au._RESOLVED_ATTENTION_ARCH
    au._RESOLVED_ATTENTION_ARCH = "gfx942"
    # monkeypatch.delenv, not os.environ.pop: a bare pop leaks the cleared
    # override into every later gfx942 routing test in the same process. This
    # test is GPU-gated, so the leak is invisible on a CPU host and only shows up
    # in the ordering runs the regression guard is meant to cover.
    for var in (
        "HIPDNN_GFX942_K_SLICED_RING",
        "HIPDNN_GFX942_FLASH_WIDE",
        "HIPDNN_GFX942_FLASH_MLIM",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        from builders.common.attention_spec_builder import _tiled_spec_from_problem

        # randn (unit-variance) inputs -- NOT the uniform_(-0.1,0.1) default that
        # masked the ring bug.
        rng = np.random.default_rng(0)
        rt = an.runtime()
        max_blk = -(-sq // _BLOCK_SIZE)
        q = an.randn(rng, (sq, hq, 128), dtype)
        k_cache = an.randn(rng, (max_blk, _BLOCK_SIZE, hk, 128), dtype)
        v_cache = an.randn(rng, (max_blk, _BLOCK_SIZE, hk, 128), dtype)
        block_tables = np.arange(max_blk, dtype=np.int32).reshape(1, max_blk)
        scale = 128**-0.5
        ref = an.paged_attention_reference(
            q, k_cache, v_cache, block_tables, [sq], [sq], scale=scale
        )

        p = _problem(dtype, sq=sq, hq=hq, hk=hk)
        spec = _tiled_spec_from_problem(p, "gfx942")
        # Routing sanity: bf16 D128 is non-ring; fp16 D128 uses the depth-2 ring
        # once past the small-q narrow path (q<=768 -> narrow, no ring). The
        # numeric assert below gates every path on correctness regardless.
        if dtype == "bf16":
            assert not spec.use_k_sliced_ring, "bf16 D128 must be non-ring"
        elif spec.use_k_sliced_ring:
            assert spec.ring_depth == 2, "fp16 D128 ring must be depth-2"

        # Launch exactly the spec asserted on above, not a re-selected one.
        tuned = AttentionTuningSpec(
            path="2d",
            arch="gfx942",
            builder_kind="tiled",
            compile_backend="llvm",
            candidate_name="d128_ring_guard",
            tuning_id="d128_ring_guard",
            kernel_spec=spec,
        )
        launched = tuned.with_num_kv_blocks(max_blk)
        assert launched.kernel_spec == spec, "cache size must not retarget the spec"
        key = launched.cache_key()
        au._2D_LAUNCHERS.pop(key, None)
        au._ATTN_TILED_CACHE.pop(key, None)

        out = an.empty_device(rt, q.shape, dtype)
        au.run_unified_attention_torch(
            problem=p,
            q=an.to_device(rt, q, dtype),
            k=an.to_device(rt, k_cache, dtype),
            v=an.to_device(rt, v_cache, dtype),
            out=out,
            cu_seqlens_q=DeviceArray(rt, np.array([0, sq], dtype=np.int32)),
            seqused_k=DeviceArray(rt, np.array([sq], dtype=np.int32)),
            softmax_scale=scale,
            block_table=DeviceArray(rt, block_tables),
            softcap=0.0,
            backend="tiled",
            tuning_spec=tuned,
        )
        # The launcher cached under the asserted spec's identity is the one that ran.
        assert key in au._2D_LAUNCHERS, "the asserted spec was not the one launched"
        an.assert_close(
            an.read(out, dtype),
            ref,
            dtype,
            f"D128 {dtype} {hq}/{hk} S{sq} (ring regression?)",
            atol=tol,
        )
    finally:
        au._RESOLVED_ATTENTION_ARCH = old_arch
