# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pins for the VGPR-first accumulator split estimate (FP4 subtile).

These exercise TileInfo.estimateVgprAccumulatorSplit / estimateSubtileMainLoopVgprs
directly (no register allocation) so a scheduler or reserve change that shifts the
VGPR/AGPR accumulator boundary is caught here rather than as a generated-assembly
regression.

The scaffolding (_make_kernel / _build_tiles / _make_writer / _StubPool) is shared
with test_SubtileMainLoopReserveNoFit.

A separate contiguity pin (test_arch_order_contiguous_valuCSourceMap) performs a
real allocation and asserts the VGPR-resident D accumulators form arch-order
contiguous, width-aligned groups so the epilogue paired dwordx4 store resolves
directly (Option B: contiguous store-order VGPR-first accumulators).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from types import SimpleNamespace

import pytest

from test_SubtileBasedLogicalScheduler import (
    create_kernel,
    makeTileInfo,
    make_writer_and_tileinfos,
)


# ---------------------------------------------------------------------------
# Shared scaffolding
# ---------------------------------------------------------------------------

class _StubPool:
    """Minimal RegisterPool stand-in.

    estimateVgprAccumulatorSplit / estimateSubtileMainLoopVgprs are pure with
    respect to the pools: they only read the current high-water size(). A stub
    lets the no-fit / split estimate run without rocisa register allocation.
    """

    def __init__(self, size=0):
        self._size = int(size)

    def size(self):
        return self._size


def _make_kernel(MT0, MT1, miWaveGroup, streamK, pgr):
    """FP4 subtile kernel dict wired for the VGPR-first split estimate."""
    kernel = create_kernel(MT0, MT1, fp4=True, miWaveGroup=list(miWaveGroup))
    kernel["UseSubtileImpl"] = True
    kernel["StreamK"] = streamK
    kernel["PrefetchGlobalRead"] = pgr
    # The split estimate reads DataType for the store min-element floor; FP4
    # subtile is never complex, so pin isComplex() False (the shared _mock_dtype
    # leaves it as a truthy MagicMock otherwise).
    dt = kernel["ProblemType"]["DataTypeA"]
    dt.isComplex.return_value = False
    kernel["ProblemType"]["DataType"] = dt
    return kernel


def _build_tiles(kernel):
    """Geometry-only TileInfos (no allocation) for A/B/MXSA/MXSB/D."""
    tiA = makeTileInfo("A", kernel)
    tiB = makeTileInfo("B", kernel)
    scaleTiA = makeTileInfo("MXSA", kernel)
    scaleTiB = makeTileInfo("MXSB", kernel)
    tiD = makeTileInfo("D", kernel)
    return tiA, tiB, scaleTiA, scaleTiB, tiD


def _make_writer(tiA, tiB, scaleTiA, scaleTiB, max_vgpr=256, physical_max_vgpr=512):
    """Stub writer carrying just what the split estimate consults."""
    writer = SimpleNamespace()
    writer.vgprPool = _StubPool(0)
    writer.agprPool = _StubPool(0)
    writer.states = SimpleNamespace(
        regCaps={"MaxSgpr": 106, "MaxVgpr": max_vgpr,
                 "PhysicalMaxVgpr": physical_max_vgpr},
        a=SimpleNamespace(tileInfo=tiA),
        b=SimpleNamespace(tileInfo=tiB),
        mxsa=SimpleNamespace(tileInfo=scaleTiA),
        mxsb=SimpleNamespace(tileInfo=scaleTiB),
        c=SimpleNamespace(numVgprValu=0),
    )
    return writer


# ---------------------------------------------------------------------------
# Split-estimate pins
# ---------------------------------------------------------------------------

def test_fp4_subtile_prefers_vgpr_with_free_pool():
    """With an empty pool, the FP4 subtile D tile keeps accumulators in VGPR."""
    kernel = _make_kernel(128, 128, [2, 2], 3, 2)
    tiA, tiB, scaleTiA, scaleTiB, tiD = _build_tiles(kernel)
    writer = _make_writer(tiA, tiB, scaleTiA, scaleTiB)

    est = tiD.estimateVgprAccumulatorSplit(writer, kernel)
    assert est.preferVgpr is True
    # vgprAccLimit must leave room below the VGPR file (never past v255-ish).
    assert 0 < est.vgprAccLimit <= writer.states.regCaps["MaxVgpr"]


def test_fully_non_bypassable_falls_back_to_agpr_first():
    """A FP4 subtile kernel whose every store section is non-bypassable (UseScaleCD
    routes the scaleD epilogue through raw "ValuC+N") must NOT pay VGPR-first's cost:
    the allocator falls back to AGPR-first (preferVgpr False)."""
    kernel = _make_kernel(128, 128, [2, 2], 3, 2)
    # UseScaleCD's scaleDModule emits raw "ValuC+%d", so _can_bypass_valu_c is
    # False for every (beta, factorDim) section -> no section can bypass.
    kernel["ProblemType"]["UseScaleCD"] = True
    tiA, tiB, scaleTiA, scaleTiB, tiD = _build_tiles(kernel)
    writer = _make_writer(tiA, tiB, scaleTiA, scaleTiB)

    est = tiD.estimateVgprAccumulatorSplit(writer, kernel)
    assert est.preferVgpr is False


def test_mixed_bypassability_stays_vgpr_first():
    """A FP4 subtile kernel where only some sections bypass must stay VGPR-first.
    With UseBeta + UseScaleAlphaVec, the beta=True section cannot bypass but the
    beta=False section still can, so at least one section benefits."""
    kernel = _make_kernel(128, 128, [2, 2], 3, 2)
    kernel["ProblemType"]["UseBeta"] = True
    kernel["ProblemType"]["UseScaleAlphaVec"] = 1
    tiA, tiB, scaleTiA, scaleTiB, tiD = _build_tiles(kernel)
    writer = _make_writer(tiA, tiB, scaleTiA, scaleTiB)

    est = tiD.estimateVgprAccumulatorSplit(writer, kernel)
    assert est.preferVgpr is True


def test_main_loop_reserve_is_positive():
    """The main-loop working-set reserve is a positive VGPR count."""
    kernel = _make_kernel(128, 128, [2, 2], 3, 2)
    tiA, tiB, scaleTiA, scaleTiB, tiD = _build_tiles(kernel)
    writer = _make_writer(tiA, tiB, scaleTiA, scaleTiB)

    reserve = tiD.estimateSubtileMainLoopVgprs(writer, kernel)
    assert reserve > 0


# ---------------------------------------------------------------------------
# Contiguity pin: arch-order contiguous VGPR D accumulators (Option B)
# ---------------------------------------------------------------------------

def _arch_phys_map(kernel, dTile):
    """Return itemPhys[archOffset] = (isVgpr, physReg) for the D-tile allocation.

    Mirrors mapAcctoArchRegsFromAccRegMap: composes the allocation map
    (accRegMapFromTileInfo, in MMA/acc order) with the MMA->arch permutation
    (accToArchMapper) so the result is indexed in the order the epilogue store
    reads ValuC.
    """
    from Tensile.Components.Subtile.Kernel import accRegMapFromTileInfo
    from Tensile.KernelWriterModules import accToArchMapper

    accRegMap = accRegMapFromTileInfo(kernel, dTile)
    acc2arch, _ = accToArchMapper(kernel)
    miRegPerOut = kernel["MIRegPerOut"]

    itemPhys = [None] * (len(acc2arch) * miRegPerOut)
    for i in range(len(acc2arch)):
        for r in range(miRegPerOut):
            itemPhys[acc2arch[i] * miRegPerOut + r] = accRegMap[i * miRegPerOut + r]
    return itemPhys


def test_arch_order_contiguous_valuCSourceMap():
    """All-VGPR FP4 subtile D tile: every width-4 arch-order store group maps to
    4 contiguous, 4-aligned physical VGPRs (so _directValuCDTileGroup resolves
    and the epilogue emits buffer_store_dwordx4 with zero acc gather)."""
    kernel = _make_kernel(128, 128, [2, 2], 3, 2)
    # accToArchMapper inputs for an FP4 16x16 single-block MFMA with VW=1.
    kernel["MatrixInstBM"] = 1
    kernel["MatrixInstBN"] = 1
    kernel["VectorWidthA"] = 1
    kernel["VectorWidthB"] = 1
    kernel["MIRegPerOut"] = kernel.get("MIRegPerOut", 1)

    writer, tiA, tiB, scaleTiA, scaleTiB, dTile = make_writer_and_tileinfos(
        kernel, fp4=True)

    # Sanity: this config keeps the whole D tile in VGPR (no AGPR spill), which is
    # the case Option B's contiguous arch-order layout targets.
    assert all(vt.regList.is_vgpr for vt in dTile.vgprTiles), \
        "expected an all-VGPR D tile for the contiguity pin"

    itemPhys = _arch_phys_map(kernel, dTile)
    width = 4
    assert len(itemPhys) % width == 0
    for g in range(len(itemPhys) // width):
        regs = itemPhys[g * width:(g + 1) * width]
        assert all(rp is not None for rp in regs)
        assert all(isVgpr for isVgpr, _ in regs), "arch group must be all-VGPR"
        idxs = [reg for _, reg in regs]
        assert idxs[0] % width == 0, \
            "arch group base %u is not %u-aligned" % (idxs[0], width)
        assert idxs == [idxs[0] + j for j in range(width)], \
            "arch group %u is non-contiguous: %s" % (g, idxs)
