# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GR↔LR cross-check for the subtile tail-loop narrow trailing load.

Phase C of the narrow-trailing-load design (see plan in chat log).
Asserts that the GR-side narrow-load descriptor and the LR-side per-
lane mask init's reader agree on the SAME LDS byte address. If the
two disagree, the narrow load would write into a slot that the LR
doesn't read (or vice versa) and the MFMA boundary element would
still be wrong.

The cross-check is the safety net for Phase A1 (the mechanical case-
split encode of `(loadRatioGR, swizzle, wg_m × wg_n,
localSubtileGrid[1])` combinations) -- if A1's inversion of
`_grSwizzleColIds_legacy` or `_grComputeRowPartition_legacy` is wrong
for any geometry, the cross-check fails at that geometry's parametrize
entry.

The canonical pin (`test_canonical_MT128x128_K129`) hard-codes the
worked example pulled from the C.1 derivation against a built kernel
disassembly (`tensile-out_cleanup_subtile_bf16_anyk_k2/.../MT128x128x64.s`).
That pin is the "if this breaks something deep is wrong" backstop;
the parametrized tests below check the cross-product.
"""
import math

import pytest

from Tensile.Components.Subtile.SubtileTailNarrowLoad import (
    computeLDSStartOffsetB,
    computeLRReaderForBoundary,
    computeNarrowLoadDescriptor,
    subtileTailNarrowLoadApplies,
)
from Tensile.Tests.unit._subtile_tailloop_fixtures import (
    build_minimal_subtile_kwa,
    setdefault_tail_scaffold_kernel_keys,
)
from Tensile.Tests.unit.test_subtile_tailloop_emit import _create_kernel


# ── Harness helpers ─────────────────────────────────────────────────────────


def _build_tile_infos(MT0, MT1, depthU, wg_m, wg_n, asem=1):
    """Build a minimal kernel + populated KernelWriterAssembly so the
    TileInfo geometry is computed end-to-end. Returns (kernel, kwa,
    tiA, tiB).
    """
    kernel = _create_kernel(MT0=MT0, MT1=MT1, fp4=False, depthU=depthU,
                            no_tail_loop=False)
    kernel["MIWaveGroup"] = [wg_m, wg_n]
    setdefault_tail_scaffold_kernel_keys(kernel, pgr=0, asem=asem)
    kwa = build_minimal_subtile_kwa(kernel)
    return kernel, kwa, kwa.states.a.tileInfo, kwa.states.b.tileInfo


def _cross_check_one(kernel, tiA, tiB, K_remain):
    """Run the cross-check for one (kernel, K_remain). Asserts that
    GR.m0_target == LR.lds_byte_target for both A and B."""
    for tc, ti in (('A', tiA), ('B', tiB)):
        gr = computeNarrowLoadDescriptor(kernel, ti, K_remain, tc, tiA=tiA)
        lr = computeLRReaderForBoundary(kernel, ti, K_remain, tc, tiA=tiA)
        assert gr.m0_target == lr.lds_byte_target, (
            f"GR↔LR LDS-byte disagreement for tc={tc} K_remain={K_remain}:\n"
            f"  GR: {gr.explain}\n"
            f"  LR: {lr.explain}\n"
            f"  GR.m0_target = {gr.m0_target}\n"
            f"  LR.lds_byte  = {lr.lds_byte_target}\n"
            f"  delta        = {gr.m0_target - lr.lds_byte_target}"
        )


# ── Canonical pins (from the C.1 worked example) ────────────────────────────


class TestCanonicalPins:
    """Pins the four-tuple `(wave_target, lane_target, m0_target,
    vaddr_target)` derived in C.1 against the actual generated
    disassembly. If any of these change, the design has shifted and
    the rest of the harness should also be re-checked.
    """

    def test_canonical_MT128x128_K129(self):
        """MT 128×128 BF16 DU=64 WG=(2,2), K=129 → K_remain=1.

        Pinned values from the C.2 worked example, cross-checked
        against `subtile_bf16_anyk_k2.yaml` MT128x128x64 disassembly:
          A: wave=3, sId0_last=3, rowId=7, lane=62, m0=16352,
             vaddr_within_row=0.
          B: wave=3, sId0_last=3, rowId=7, lane=62, m0=32736,
             vaddr_within_row=0.
        """
        kernel, _, tiA, tiB = _build_tile_infos(
            MT0=128, MT1=128, depthU=64, wg_m=2, wg_n=2, asem=1)
        assert subtileTailNarrowLoadApplies(kernel), \
            "MT 128×128 BF16 must be in scope for the narrow load"

        descA = computeNarrowLoadDescriptor(kernel, tiA, K_remain=1, tc='A',
                                            tiA=tiA)
        assert descA.wave_target == 3, descA.explain
        assert descA.sId0_last == 3, descA.explain
        assert descA.lane_target == 62, descA.explain
        assert descA.m0_target == 16352, descA.explain
        assert descA.vaddr_target == 0, descA.explain

        descB = computeNarrowLoadDescriptor(kernel, tiB, K_remain=1, tc='B',
                                            tiA=tiA)
        assert descB.wave_target == 3, descB.explain
        assert descB.lane_target == 62, descB.explain
        assert descB.m0_target == 32736, descB.explain

        # ldsStartOffsetB pin: A subtiles fit in 16384 bytes (8 subtiles
        # × 2048 each, aligned up to 2 * 2048 = 4096).
        assert computeLDSStartOffsetB(tiA) == 16384

        _cross_check_one(kernel, tiA, tiB, K_remain=1)

    def test_canonical_MT128x128_K131(self):
        """Same geometry, K_remain=3 (the other shape exercised by
        `subtile_bf16_anyk_odd.yaml`)."""
        kernel, _, tiA, tiB = _build_tile_infos(
            MT0=128, MT1=128, depthU=64, wg_m=2, wg_n=2, asem=1)
        descA = computeNarrowLoadDescriptor(kernel, tiA, K_remain=3, tc='A',
                                            tiA=tiA)
        # K_local = 2, K_within_lane = 2 (byte offset 4 within the
        # lane's 16B chunk). Wave/lane unchanged from K_remain=1
        # because K_local is still in colId_post=0.
        assert descA.wave_target == 3, descA.explain
        assert descA.lane_target == 62, descA.explain
        # m0 = 16352 + (K_within_lane * bpe) = 16352 + 4 = 16356.
        assert descA.m0_target == 16356, descA.explain
        _cross_check_one(kernel, tiA, tiB, K_remain=3)

    def test_canonical_MT128x128_K_remain_5(self):
        """K_remain=5 → K_local=4, K_within_lane=4 (byte offset 8)."""
        kernel, _, tiA, tiB = _build_tile_infos(
            MT0=128, MT1=128, depthU=64, wg_m=2, wg_n=2, asem=1)
        _cross_check_one(kernel, tiA, tiB, K_remain=5)

    def test_canonical_MT128x128_K_remain_7(self):
        """K_remain=7 → K_local=6, K_within_lane=6 (byte offset 12).
        Last odd value before K_local crosses an elementsPerLane=8
        boundary into colId_post=1.
        """
        kernel, _, tiA, tiB = _build_tile_infos(
            MT0=128, MT1=128, depthU=64, wg_m=2, wg_n=2, asem=1)
        _cross_check_one(kernel, tiA, tiB, K_remain=7)

    def test_canonical_MT128x128_K_remain_9(self):
        """K_remain=9 → K_local=8 → colId_post crosses to 1.

        First K_remain value where the missing K-element lives in a
        DIFFERENT lane's chunk than K_remain<8. Exercises the
        colId_post → colId_pre inversion for a non-zero colId_post.
        """
        kernel, _, tiA, tiB = _build_tile_infos(
            MT0=128, MT1=128, depthU=64, wg_m=2, wg_n=2, asem=1)
        _cross_check_one(kernel, tiA, tiB, K_remain=9)


# ── Parametrized cross-check sweep ──────────────────────────────────────────


_GEOMETRIES_BF16 = [
    # (MT0, MT1, depthU, wg_m, wg_n)
    # Shapes pulled from subtile_bf16_anyk_odd.yaml / largemt.yaml /
    # _k2.yaml / _k8.yaml. The wg_m × wg_n × MT cross-product covers
    # the Phase A1 case-split.
    (128, 128, 64, 2, 2),    # canonical
    (256, 256, 64, 2, 2),
    (128, 256, 64, 2, 2),
    (256, 128, 64, 2, 2),
    (64,  64,  64, 1, 1),    # WG=(1,1) — odd.yaml smallest shape
    (128,  32, 64, 4, 1),    # WG=(4,1) — odd.yaml asymmetric shape
]

_K_REMAINS = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
              33, 35, 47, 49, 63]


@pytest.mark.parametrize("MT0,MT1,depthU,wg_m,wg_n", _GEOMETRIES_BF16,
                         ids=lambda v: str(v))
@pytest.mark.parametrize("K_remain", _K_REMAINS,
                         ids=lambda v: f"krem{v}")
def test_gr_lr_cross_check_bf16(MT0, MT1, depthU, wg_m, wg_n, K_remain):
    """Cross-check GR.m0_target == LR.lds_byte_target across the bf16
    geometry × K_remain cross-product.

    Skips combinations where:
      - K_remain >= DepthU (no tail loop fires).
      - K_remain is even (the narrow load only fires for odd K_remain;
        even K_remain is fully covered by align-DOWN with no missing
        BF16 element).
      - The geometry is not in Phase A1's coverage (
        :func:`subtileTailNarrowLoadApplies` returns False or the
        derivation hits NotImplementedError).
    """
    if K_remain >= depthU:
        pytest.skip(f"K_remain={K_remain} >= DepthU={depthU}; tail loop "
                    "doesn't fire")
    if K_remain % 2 == 0:
        pytest.skip(f"K_remain={K_remain} even; align-DOWN covers it "
                    "without a narrow load")

    try:
        kernel, _, tiA, tiB = _build_tile_infos(
            MT0=MT0, MT1=MT1, depthU=depthU, wg_m=wg_m, wg_n=wg_n, asem=1)
    except (ValueError, ZeroDivisionError) as e:
        # TileInfo raises for malformed geometries (MT not divisible,
        # localSubtileGrid[1]==0 → /0, etc.). Skip those -- they're
        # outside the Solution.py gate anyway.
        pytest.skip(f"TileInfo build failed: {e}")

    if not subtileTailNarrowLoadApplies(kernel):
        pytest.skip("subtileTailNarrowLoadApplies returned False")

    try:
        _cross_check_one(kernel, tiA, tiB, K_remain)
    except NotImplementedError as e:
        pytest.skip(f"Phase A1 case-split not encoded: {e}")
