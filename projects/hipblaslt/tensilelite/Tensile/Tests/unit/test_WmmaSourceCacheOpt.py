# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for zigzag WMMA tile ordering in emit_mfma."""

import re
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from Tensile.Components.Subtile.LogicalScheduler import (
    MFMATileRange, ReadGranularity, SchedulerConfig, MFMAPlacement,
)
from Tensile.Components.Subtile.InstructionEmitter import _zigzag_order
from Tensile.Components.Subtile.InstructionEmitter import InstructionEmitter
from Tensile.Components.Subtile.Kernel import TileInfo, AB_B16, CD_F32


# ── Helpers ─────────────────────────────────────────────────────

_COMMENT_RE = re.compile(r"MFMA C\[(\d+),(\d+)\]")


def _extract_ab_pairs(instructions):
    """Return [(a, b), ...] from MFMA instruction comments."""
    pairs = []
    for inst in instructions:
        m = _COMMENT_RE.search(str(inst))
        if m:
            pairs.append((int(m.group(1)), int(m.group(2))))
    return pairs


def _mock_dtype(num_bytes=2):
    mock = MagicMock()
    mock.numBytes.return_value = num_bytes
    mock.numRegisters.return_value = num_bytes / 4
    mock.isBFloat16.return_value = (num_bytes == 2)
    mock.is8bitFloat.return_value = (num_bytes == 1)
    return mock


def _make_kernel(numM, numN):
    """Minimal kernel dict for emit_mfma."""
    dtype = _mock_dtype(2)
    return {
        "MacroTile0": numM * 16,
        "MacroTile1": numN * 16,
        "MacroTileA": numM * 16,
        "MacroTileB": numN * 16,
        "DepthU": 64,
        "_DepthUA": 64,
        "_DepthUB": 64,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": 32,
        "MatrixInstB": 1,
        "MIInputPerThreadA": 8,
        "MIInputPerThreadB": 8,
        "MIWaveGroup": [1, 1],
        "WavefrontSize": 32,
        "SourceSwap": False,
        "MIArchVgpr": True,
        "ProblemType": {
            "DataTypeA": dtype,
            "DataTypeB": dtype,
            "ComputeDataType": _mock_dtype(4),
            "MXBlockA": 0,
            "MXBlockB": 0,
        },
    }


def _make_emitter(numM, numN):
    """Build a minimal InstructionEmitter that can call emit_mfma."""
    from rocisa.register import RegisterPool
    from rocisa.enum import RegisterType

    kernel = _make_kernel(numM, numN)

    writer = SimpleNamespace()
    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr,
                                    defaultPreventOverflow=False, printRP=False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr,
                                    defaultPreventOverflow=False, printRP=False)
    writer.agprPool = RegisterPool(0, RegisterType.Accvgpr,
                                    defaultPreventOverflow=False, printRP=False)
    writer.states = SimpleNamespace(
        archCaps={"LDSBankCount": 64, "LDSBankWidth": 4},
        asmCaps={"HasWMMA_V3": True},
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256, "PhysicalMaxVgpr": 512},
        subtileLdsSwizzle=False,
    )

    tileInfoA = TileInfo(AB_B16, 'A', writer, kernel)
    tileInfoB = TileInfo(AB_B16, 'B', writer, kernel)
    dtileInfo = TileInfo(CD_F32, 'D', writer, kernel)

    # Allocate VGPR tiles for A, B, D
    tileInfoA.allocVgprTileRegisters_legacy(writer, kernel)
    tileInfoB.allocVgprTileRegisters_legacy(writer, kernel)
    dtileInfo.allocVgprTileRegisters_legacy(writer, kernel)

    config = SchedulerConfig(
        numMFMATilesM=numM,
        numMFMATilesN=numN,
        numSubIterK=2,
        lrA=ReadGranularity(mn=1, k=1),
        lrB=ReadGranularity(mn=1, k=1),
        grA=ReadGranularity(mn=1, k=2),
        grB=ReadGranularity(mn=1, k=2),
    )

    emitter = InstructionEmitter(
        writer, kernel, config,
        tileInfoA, tileInfoB, dtileInfo,
        tileInfoA.vgprTiles, tileInfoB.vgprTiles,
    )
    return emitter


def _make_placement(numM, numN, subIterK=0):
    """Create an MFMAPlacement spanning all M and N tiles at one subIterK."""
    placement = MFMAPlacement(
        subIterK=subIterK,
        tileA=MFMATileRange(subIterK, subIterK + 1, 0, numM),
        tileB=MFMATileRange(subIterK, subIterK + 1, 0, numN),
    )
    # Identity tile map: tileId -> vgprTileId (one unroll iter)
    placement.vgpr_tile_maps = {
        'A': [{i: i for i in range(numM)}],
        'B': [{i: i for i in range(numN)}],
    }
    return placement


@pytest.fixture(scope="module", autouse=True)
def _init_rocisa():
    from gpu_test_helpers import init_rocisa
    init_rocisa(target="gfx1250", wavesize=32)


# ── Tests ───────────────────────────────────────────────────────

class TestZigzagOrder:
    """Verify the _zigzag_order helper produces correct zigzag-matrix traversal."""

    def test_1x1(self):
        assert _zigzag_order(1, 1) == [(0, 0)]

    def test_2x2(self):
        assert _zigzag_order(2, 2) == [(0, 0), (0, 1), (1, 1), (1, 0)]

    def test_3x3(self):
        expected = [
            (0, 0), (0, 1), (0, 2),
            (1, 2), (1, 1), (1, 0),
            (2, 0), (2, 1), (2, 2),
        ]
        assert _zigzag_order(3, 3) == expected

    def test_4x4(self):
        result = _zigzag_order(4, 4)
        assert len(result) == 16
        assert len(set(result)) == 16  # all unique
        # every consecutive pair shares exactly one coordinate
        for i in range(1, len(result)):
            r0, c0 = result[i - 1]
            r1, c1 = result[i]
            assert (r0 == r1) != (c0 == c1), (
                f"Step {i}: ({r0},{c0})->({r1},{c1}) must share exactly one coord"
            )

    def test_2x4_rectangular(self):
        result = _zigzag_order(2, 4)
        assert len(result) == 8
        assert len(set(result)) == 8
        # first row left-to-right, then down, then bottom row right-to-left, then up
        assert result == [
            (0, 0), (0, 1), (0, 2), (0, 3),
            (1, 3), (1, 2), (1, 1), (1, 0),
        ]

    def test_4x2_rectangular(self):
        result = _zigzag_order(4, 2)
        assert len(result) == 8
        assert len(set(result)) == 8
        assert result == [
            (0, 0), (0, 1),
            (1, 1), (1, 0),
            (2, 0), (2, 1),
            (3, 1), (3, 0),
        ]

    def test_1xN_single_row(self):
        assert _zigzag_order(1, 5) == [(0, c) for c in range(5)]

    def test_Nx1_single_col(self):
        assert _zigzag_order(5, 1) == [(r, 0) for r in range(5)]


class TestZigzagEmitMfma:
    """Verify emit_mfma produces zigzag-ordered (a,b) pairs when enabled."""

    def test_longer_a_sweeps_along_a(self):
        """When A > B, the contiguous zig sweeps along A (longer dim)."""
        numM, numN = 4, 3
        emitter = _make_emitter(numM, numN)
        placement = _make_placement(numM, numN)
        insts = emitter.emit_mfma(placement, unroll_iter=0)
        pairs = _extract_ab_pairs(insts)

        # Sweep along A: outer=B(3), inner=A(4)
        expected = [(c, r) for r, c in _zigzag_order(numN, numM)]
        assert pairs == expected

    def test_longer_b_sweeps_along_b(self):
        """When B > A, the contiguous zig sweeps along B (longer dim)."""
        numM, numN = 3, 4
        emitter = _make_emitter(numM, numN)
        placement = _make_placement(numM, numN)
        insts = emitter.emit_mfma(placement, unroll_iter=0)
        pairs = _extract_ab_pairs(insts)

        # Sweep along B: outer=A(3), inner=B(4)
        expected = [(r, c) for r, c in _zigzag_order(numM, numN)]
        assert pairs == expected

    def test_every_pair_shares_one_operand(self):
        """Every consecutive WMMA pair must share exactly one operand (A or B)."""
        numM, numN = 4, 4
        emitter = _make_emitter(numM, numN)
        placement = _make_placement(numM, numN)
        insts = emitter.emit_mfma(placement, unroll_iter=0)
        pairs = _extract_ab_pairs(insts)

        for i in range(1, len(pairs)):
            prev_a, prev_b = pairs[i - 1]
            cur_a, cur_b = pairs[i]
            shared_a = (prev_a == cur_a)
            shared_b = (prev_b == cur_b)
            assert shared_a != shared_b, (
                f"Step {i}: ({prev_a},{prev_b})->({cur_a},{cur_b}) "
                f"must share exactly one operand"
            )

    def test_single_b_tile_unaffected(self):
        """With only 1 B tile, zigzag falls back to raster."""
        numM, numN = 4, 1
        emitter = _make_emitter(numM, numN)
        placement = _make_placement(numM, numN)
        insts = emitter.emit_mfma(placement, unroll_iter=0)
        pairs = _extract_ab_pairs(insts)

        expected = [(a, 0) for a in range(numM)]
        assert pairs == expected

    def test_single_a_tile_unaffected(self):
        """With only 1 A tile, zigzag falls back to raster."""
        numM, numN = 1, 4
        emitter = _make_emitter(numM, numN)
        placement = _make_placement(numM, numN)
        insts = emitter.emit_mfma(placement, unroll_iter=0)
        pairs = _extract_ab_pairs(insts)

        expected = [(0, b) for b in range(numN)]
        assert pairs == expected

    def test_2x2_zigzag(self):
        """Minimal 2x2 case: zigzag snake (A==B, sweeps along A)."""
        numM, numN = 2, 2
        emitter = _make_emitter(numM, numN)
        placement = _make_placement(numM, numN)
        insts = emitter.emit_mfma(placement, unroll_iter=0)
        pairs = _extract_ab_pairs(insts)

        # A>=B so sweep along A: [(0,0),(1,0),(1,1),(0,1)]
        assert pairs == [(0, 0), (1, 0), (1, 1), (0, 1)]
