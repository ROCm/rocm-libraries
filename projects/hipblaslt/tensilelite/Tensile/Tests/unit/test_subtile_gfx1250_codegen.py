#!/usr/bin/env python3
################################################################################
# Codegen smoke test for gfx1250 (wave32) subtile paths.
#
# Exercises TileInfo construction and GR offset allocation with wave32
# kernel dicts to verify the code path doesn't crash.
#
# Usage:
#   pytest test_subtile_gfx1250_codegen.py -v
################################################################################

import os
import sys
import shutil

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)


GFX1250_ISA = (12, 5, 0)
WAVESIZE_32 = 32


def _init_rocisa_gfx1250():
    """Initialize rocIsa singleton for gfx1250."""
    from rocisa import rocIsa
    from Tensile.Common.Architectures import gfxToIsa
    ri = rocIsa.getInstance()
    isa = gfxToIsa("gfx1250")
    asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
    ri.init(isa, asmpath)
    ri.setKernel(isa, WAVESIZE_32)


def _mock_dtype(num_bytes=2):
    mock = MagicMock()
    mock.numBytes.return_value = num_bytes
    return mock


def _create_gfx1250_kernel(mt_a, mt_b, mi_wave_group=None):
    """Create a minimal gfx1250 wave32 kernel dict for subtile codegen."""
    dtype = _mock_dtype(2)
    if mi_wave_group is None:
        mi_wave_group = [1, 1]
    return {
        "DepthU": 64,
        "_DepthU": 64,
        "_DepthUA": 64,
        "_DepthUB": 64,
        "MacroTileA": mt_a,
        "MacroTileB": mt_b,
        "MacroTile0": mt_a,
        "MacroTile1": mt_b,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": 32,
        "MIWaveGroup": mi_wave_group,
        "WavefrontSize": WAVESIZE_32,
        "UseSubtileImpl": True,
        "ISA": GFX1250_ISA,
        "MIArchVgpr": True,
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "enableTDMA": True,
        "enableTDMB": True,
        "ProblemType": {
            "DataTypeA": dtype,
            "DataTypeB": dtype,
            "ComputeDataType": _mock_dtype(4),
        },
    }


CONFIGS = [
    (32, 32, [1, 1]),
    (64, 64, [2, 2]),
    (128, 32, [4, 1]),
    (32, 128, [1, 4]),
]


class TestGfx1250SubtileCodegen:
    """Smoke test: gfx1250 subtile TileInfo + GR alloc doesn't crash."""

    @pytest.mark.parametrize("mt_a,mt_b,wg", CONFIGS,
                             ids=[f"{a}x{b}_wg{wg[0]}x{wg[1]}" for a, b, wg in CONFIGS])
    def test_tdm_skips_gr_offset_alloc(self, mt_a, mt_b, wg):
        """With TDM enabled, GR offset registers should not be allocated."""
        _init_rocisa_gfx1250()
        from rocisa.register import RegisterPool
        from rocisa.enum import RegisterType
        from Tensile.Components.Subtile.Kernel import TileInfo, AB_B16_W32

        kernel = _create_gfx1250_kernel(mt_a, mt_b, mi_wave_group=wg)
        writer = SimpleNamespace()
        writer.vgprPool = RegisterPool(0, RegisterType.Vgpr,
                                       defaultPreventOverflow=False, printRP=False)
        writer.sgprPool = RegisterPool(0, RegisterType.Sgpr,
                                       defaultPreventOverflow=False, printRP=False)
        writer.agprPool = RegisterPool(0, RegisterType.Accvgpr,
                                       defaultPreventOverflow=False, printRP=False)
        writer.sgprs = {}
        writer.vgprPool.checkOut(1)

        tiA = TileInfo(AB_B16_W32, 'A', writer, kernel)
        tiA.allocOffsetRegisters(writer, kernel)

        assert tiA.sharedVgprGROffset == [], \
            f"TDM kernel should have empty GR offsets, got {tiA.sharedVgprGROffset}"
