# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for sharedVgprGROffsetSwap NoSwizzle-only allocation.

NoSwizzle double-buffers GR ds_store via a VGPR swap mask.
HostPreSwizzle / InMemorySwizzle swap LocalWriteBaseAddr in SGPR space and
must not take sharedVgprGROffsetSwap, or later VGPR numbering drifts from
pre-NoSwizzle assembly.
"""

from types import SimpleNamespace

import pytest

from Tensile.Components.Subtile.Kernel import MXSA_B4, TileInfo

pytestmark = pytest.mark.unit


class _MockPool:
    def __init__(self, start=0):
        self._counter = start
        self.checkouts = []

    def size(self):
        return self._counter

    def checkOut(self, n, align=None, tag=None, preventOverflow=True, **_kw):
        r = self._counter
        self._counter += n
        self.checkouts.append(tag)
        return r

    def checkIn(self, _v):
        pass

    def checkOutAligned(self, n, align, name=None, tag=None, preventOverflow=True):
        return self.checkOut(n, tag=tag)


def _make_writer():
    w = SimpleNamespace()
    w.states = SimpleNamespace(
        regCaps={"PhysicalMaxVgpr": 512, "MaxVgpr": 256, "MaxSgpr": 256},
    )
    w.vgprPool = _MockPool()
    w.sgprPool = _MockPool()
    w.agprPool = _MockPool()
    return w


def _mx_scale_kernel(mx_scale_format):
    return {
        "MacroTileA": 128,
        "MacroTileB": 128,
        "DepthU": 256,
        "_DepthUA": 256,
        "_DepthUB": 256,
        "_DepthUMXSA": 8,
        "_DepthUMXSB": 8,
        "MIWaveGroup": [2, 2],
        "WavefrontSize": 64,
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "MXScaleFormat": mx_scale_format,
    }


def _alloc_mxsa(fmt):
    writer = _make_writer()
    kernel = _mx_scale_kernel(fmt)
    ti = TileInfo(MXSA_B4, "MXSA", writer, kernel)
    ti.allocOffsetRegisters(writer, kernel)
    return ti, writer


def test_shared_vgpr_gr_offset_swap_allocated_for_noswizzle():
    ti, writer = _alloc_mxsa("NoSwizzle")
    assert ti.sharedVgprGROffsetSwap, (
        "NoSwizzle must allocate sharedVgprGROffsetSwap for GR ds_store swap"
    )
    assert (
        "allocOffsetRegisters_sharedVgprGROffsetSwap" in writer.vgprPool.checkouts
    )


@pytest.mark.parametrize("fmt", ["HostPreSwizzle", "InMemorySwizzle"])
def test_shared_vgpr_gr_offset_swap_not_allocated_for_swizzled(fmt):
    ti, writer = _alloc_mxsa(fmt)
    assert ti.sharedVgprGROffsetSwap == [], (
        f"{fmt} must not allocate sharedVgprGROffsetSwap "
        "(SGPR LocalWriteBaseAddr swap only)"
    )
    assert (
        "allocOffsetRegisters_sharedVgprGROffsetSwap"
        not in writer.vgprPool.checkouts
    )
