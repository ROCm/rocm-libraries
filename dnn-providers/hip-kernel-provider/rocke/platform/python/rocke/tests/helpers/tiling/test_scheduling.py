# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for the tiling scheduling helpers -- ``InstrClass`` masks and ``derive_sched_group_counts``.

Build-only (no GPU): these exercise pure arithmetic + a target-resolved ``TileMma`` plan. The
key gate is that ``derive_sched_group_counts`` reproduces the EXACT per-class count math the demo kernels used
inline before it was centralized -- that equality is what makes the demo migration byte-identical.
"""

from __future__ import annotations

import pytest

pytest.importorskip("rocke.core.ir", reason="rocke IR substrate required (build-only; no GPU)")

from rocke.helpers.tiling import (  # noqa: E402
    InstrClass,
    TileMma,
    Tiling,
    derive_sched_group_counts,
)

_ATOM = 16
_VW = 8  # dwordx4 f16 per wide memory op


def test_instrclass_masks_are_the_hardware_bits():
    # The member value IS the AMDGPU sched_group_barrier class mask, so it passes straight to
    # b.sched_group_barrier(mask, ...) with no indirection.
    assert (InstrClass.VALU, InstrClass.MFMA, InstrClass.VMEM_READ,
            InstrClass.VMEM_WRITE, InstrClass.DS_READ, InstrClass.DS_WRITE) \
        == (0x002, 0x008, 0x020, 0x040, 0x100, 0x200)
    assert int(InstrClass.MFMA) == 0x008  # IntEnum -> usable as the raw int mask


def _tile_mma(warp_m, warp_n, tile_k):
    return TileMma((warp_m, warp_n, tile_k), a="f16", b="f16", c="f32", target="gfx90a",
                   tiling=Tiling(atom_shape=(_ATOM, _ATOM, _ATOM)))


def _hand_counts(*, warp_m, warp_n, tile_m, tile_n, tile_k, wave_size, n_waves):
    """The exact inline count math the demos carried before centralizing it (the reference)."""
    warp_vol = warp_m * tile_k + warp_n * tile_k
    macro_vol = tile_m * tile_k + tile_n * tile_k
    return {
        InstrClass.MFMA: (warp_m // _ATOM) * (warp_n // _ATOM) * (tile_k // _ATOM),
        InstrClass.DS_READ: warp_vol // wave_size // _VW,
        InstrClass.VMEM_READ: macro_vol // (wave_size * n_waves) // _VW,
        InstrClass.DS_WRITE: macro_vol // (wave_size * n_waves) // _VW,
    }


@pytest.mark.parametrize("warp_m,warp_n,tile_k,tile_m,tile_n,n_waves", [
    (64, 64, 16, 128, 128, 4),   # 2x2 wave grid, macro 128 -- the interwave/pipelined default shape
    (64, 64, 16, 64, 64, 1),     # single wave: warp tile == macro tile
    (32, 64, 16, 64, 128, 2),    # rectangular warp + macro
])
def test_derive_sched_group_counts_matches_the_inline_hand_formula(warp_m, warp_n, tile_k, tile_m, tile_n, n_waves):
    mma = _tile_mma(warp_m, warp_n, tile_k)
    got = derive_sched_group_counts(mma.plan, tile_m=tile_m, tile_n=tile_n, n_waves=n_waves, vw=_VW)
    want = _hand_counts(warp_m=warp_m, warp_n=warp_n, tile_m=tile_m, tile_n=tile_n,
                        tile_k=tile_k, wave_size=mma.plan.wave_size, n_waves=n_waves)
    assert got == want


def test_divisor_is_asymmetric():
    # DS_READ (per-wave warp read) divides by wave_size ONLY; VMEM/DS_WRITE (cooperative macro
    # load/store) divide by wave_size * n_waves. With n_waves > 1 the two must differ for a square tile.
    mma = _tile_mma(64, 64, 16)
    c = derive_sched_group_counts(mma.plan, tile_m=128, tile_n=128, n_waves=4, vw=_VW)
    assert c[InstrClass.VMEM_READ] == c[InstrClass.DS_WRITE]         # store volume == load volume
    assert c[InstrClass.DS_READ] != c[InstrClass.VMEM_READ]          # asymmetric divisor


def test_mfma_count_is_the_plan_subtile_product():
    mma = _tile_mma(64, 64, 16)
    m, n, k = mma.plan.subtiles
    assert mma.plan.mfma_count == m * n * k
    assert derive_sched_group_counts(mma.plan, tile_m=64, tile_n=64, n_waves=1, vw=_VW)[InstrClass.MFMA] == m * n * k
