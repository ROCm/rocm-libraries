# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression guard for the CShuffle LDS-reuse barrier (PR #8844).

The common-LDS packer aliases the CShuffle C staging tile onto the A/B staging
bytes (they are non-interfering in program order).  The double-buffered and
prefetched mainloops end with a *tail* MFMA that reads A/B from LDS after their
last drain barrier and emit no trailing barrier, so once C aliases A/B the first
C ``ds_write`` in the epilogue can clobber A/B bytes a slow wave is still reading
for its tail MFMA -- a cross-wave WAR hazard on the shared pool.

The fix adds a workgroup barrier at the very start of the CShuffle epilogue,
before the first C store, in both epilogue emitters:
  * ``gemm_universal._emit_epilogue_cshuffle`` (inline), and
  * ``helpers.epilogues.CShuffleEpilogue.store`` (used by conv).

These tests assert, at the KernelDef IR level (CPU-only, no compile), that a
barrier separates the C staging allocation from the first store into it.
"""

from __future__ import annotations

import pytest

_BARRIERS = ("tile.sync", "tile.sync_lds_only", "tile.s_barrier_bare")


def _flatten(region, out):
    for op in region.ops:
        out.append(op)
        for r in op.regions:
            _flatten(r, out)
    return out


def _assert_barrier_before_first_c_store(kernel) -> None:
    """The first smem store after the C_smem allocation must be preceded by a
    workgroup barrier that itself follows the allocation (i.e. no C store may
    race the just-completed A/B reads on the aliased pool)."""
    ops = _flatten(kernel.body, [])

    c_allocs = [
        i
        for i, o in enumerate(ops)
        if o.name == "tile.smem_alloc" and "C_smem" in o.results[0].name
    ]
    assert c_allocs, "expected a C_smem allocation in a cshuffle kernel"
    c_idx = c_allocs[0]

    later_stores = [
        i
        for i, o in enumerate(ops)
        if o.name.startswith("tile.smem_store") and i > c_idx
    ]
    assert later_stores, "expected at least one store into the C staging tile"
    first_store = later_stores[0]

    barriers = [i for i in range(c_idx + 1, first_store) if ops[i].name in _BARRIERS]
    assert barriers, (
        "no workgroup barrier between the C_smem allocation and the first C "
        "store: a fast wave's C write could clobber A/B bytes a slow wave is "
        "still reading for its tail MFMA (cross-wave WAR on the aliased pool)"
    )
    # And no C store may sneak in before that barrier.
    assert not any(
        ops[i].name.startswith("tile.smem_store") for i in range(c_idx + 1, barriers[0])
    ), "a C store precedes the reuse barrier"


@pytest.mark.parametrize("pipeline", ["mem", "compv3", "compv4"])
def test_gemm_cshuffle_has_reuse_barrier(pipeline):
    """Every cshuffle GEMM pipeline emits the reuse barrier before the first C
    store (covers ``_emit_epilogue_cshuffle`` for single-buffer, double-buffer
    and prefetched mainloops)."""
    from rocke.instances.common.gemm_universal import (
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
        build_universal_gemm,
    )

    spec = UniversalGemmSpec(
        name="vg",
        tile=TileSpec(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        ),
        trait=TraitSpec(pipeline=pipeline, scheduler="intrawave", epilogue="cshuffle"),
    )
    _assert_barrier_before_first_c_store(build_universal_gemm(spec))


def test_conv_cshuffle_has_reuse_barrier():
    """The conv implicit-GEMM cshuffle path (double-buffered compv4, via the
    shared ``CShuffleEpilogue`` helper) emits the reuse barrier before the first
    C store."""
    from rocke.instances.common.conv_implicit_gemm import (
        ConvProblem,
        ImplicitGemmConvSpec,
        build_implicit_gemm_conv,
    )

    spec = ImplicitGemmConvSpec(
        problem=ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, pH=1, pW=1),
        name="c",
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
        wave_size=64,
        pipeline="compv4",
        epilogue="cshuffle",
    )
    _assert_barrier_before_first_c_store(build_implicit_gemm_conv(spec, arch="gfx950"))


def test_gemm_cshuffle_no_alias_elides_barrier_and_marks_exclusive():
    """With ``cshuffle_no_alias=True`` the C tile gets its own exclusive LDS
    bytes, so the step-0 reuse barrier is elided: there is NO barrier between the
    C_smem allocation and the first C store (the small-tile latency win)."""
    from rocke.instances.common.gemm_universal import (
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
        build_universal_gemm,
    )

    spec = UniversalGemmSpec(
        name="vg",
        tile=TileSpec(
            tile_m=128,
            tile_n=128,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
        ),
        trait=TraitSpec(
            pipeline="compv4",
            scheduler="intrawave",
            epilogue="cshuffle",
            cshuffle_no_alias=True,
        ),
    )
    ops = _flatten(build_universal_gemm(spec).body, [])
    c_allocs = [
        i
        for i, o in enumerate(ops)
        if o.name == "tile.smem_alloc" and "C_smem" in o.results[0].name
    ]
    assert c_allocs, "expected a C_smem allocation"
    c_idx = c_allocs[0]
    assert (
        ops[c_idx].results[0].type.exclusive
    ), "no_alias C tile must be an exclusive smem allocation"
    later_stores = [
        i
        for i, o in enumerate(ops)
        if o.name.startswith("tile.smem_store") and i > c_idx
    ]
    assert later_stores, "expected at least one store into the C staging tile"
    barriers = [
        i for i in range(c_idx + 1, later_stores[0]) if ops[i].name in _BARRIERS
    ]
    assert not barriers, (
        "cshuffle_no_alias must elide the step-0 reuse barrier (C has its own "
        "LDS bytes, so there is no A/B WAR hazard to guard)"
    )


def test_conv_cshuffle_no_alias_lds_budget_is_per_case():
    """is_valid_spec must size the LDS budget per ``cshuffle_no_alias``: aliased
    uses max(ab, c); no-alias uses ab + c. A spec whose max(ab, c) fits gfx950's
    LDS but whose ab + c overflows it must be accepted aliased and rejected
    no-alias (regression for yraparti's PR #8844 review comment)."""
    from rocke.instances.common.conv_implicit_gemm import (
        ConvProblem,
        ImplicitGemmConvSpec,
        is_valid_spec,
    )

    def _mk(no_alias):
        return ImplicitGemmConvSpec(
            problem=ConvProblem(N=4, Hi=28, Wi=28, C=64, K=64, Y=3, X=3),
            tile_m=256,
            tile_n=256,
            tile_k=32,
            warp_m=2,
            warp_n=2,
            warp_tile_m=32,
            warp_tile_n=32,
            warp_tile_k=16,
            pipeline="mem",
            epilogue="cshuffle",
            wave_size=64,
            cshuffle_no_alias=no_alias,
        )

    ok_alias, _ = is_valid_spec(_mk(False), "gfx950")
    ok_noalias, why = is_valid_spec(_mk(True), "gfx950")
    assert ok_alias, "aliased cshuffle (pool = max(ab, c)) should fit gfx950 LDS"
    assert not ok_noalias, (
        "no-alias cshuffle (pool = ab + c) should exceed gfx950 LDS and be "
        "rejected; sizing it as max(ab, c) would silently accept an over-budget "
        "kernel"
    )
    assert "LDS budget" in why


@pytest.mark.parametrize("cshuffle_no_alias", [False, True])
def test_wavelet_cshuffle_both_branches_have_equal_barrier_count(cshuffle_no_alias):
    """Both branches of the wavelet scf.if_else must emit the same number of
    barriers, or the workgroup hangs at a barrier one role issues but the other
    does not.

    The math branch ends with CShuffleEpilogue (N_epi barriers inside it).
    The load branch ends with a bare stub that emits N_epi raw s_barrier calls.
    This test builds the conv IR and counts barriers in each region to confirm
    both sides agree — a regression guard so future edits to either the epilogue
    or the stub cannot silently diverge.
    """
    from rocke.instances.common.conv_implicit_gemm import (
        ConvProblem,
        ImplicitGemmConvSpec,
        build_implicit_gemm_conv,
    )

    spec = ImplicitGemmConvSpec(
        problem=ConvProblem(N=2, Hi=14, Wi=14, C=32, K=32, Y=3, X=3, pH=1, pW=1),
        name="wv_barrier_check",
        tile_m=32,
        tile_n=32,
        tile_k=32,
        warp_m=1,
        warp_n=1,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32,
        wave_size=32,
        pipeline="wavelet",
        epilogue="cshuffle",
        num_load_waves=1,
        cshuffle_no_alias=cshuffle_no_alias,
    )
    kernel = build_implicit_gemm_conv(spec, arch="gfx1250")

    ops = _flatten(kernel.body, [])

    # Locate the scf.if_else that splits math vs load waves.
    if_else_ops = [o for o in ops if o.name == "scf.if_else"]
    assert if_else_ops, "expected an scf.if_else in the wavelet kernel"
    ife = if_else_ops[0]

    def _count_barriers_in_region(region):
        buf = []
        _flatten(region, buf)
        return sum(1 for o in buf if o.name in _BARRIERS)

    then_count = _count_barriers_in_region(ife.regions[0])
    else_count = _count_barriers_in_region(ife.regions[1])

    assert then_count == else_count, (
        f"barrier count mismatch (cshuffle_no_alias={cshuffle_no_alias}): "
        f"math branch has {then_count}, load branch has {else_count}. "
        "A workgroup hang will occur at the barrier one role skips."
    )
    assert then_count > 0, "expected at least one barrier in each wavelet branch"


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
