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


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
