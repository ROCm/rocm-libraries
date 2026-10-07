# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TileInfo MXScaleFormat gates: isSwizzled and NoSwizzle-only GROffsetSwap.

HostPreSwizzle / InMemorySwizzle use SGPR LocalWriteBaseAddr swap and must not
allocate sharedVgprGROffsetSwap (VGPR numbering would drift from pre-NoSwizzle
asm). NoSwizzle needs the VGPR swap mask for ds_store double-buffering.
"""

import pytest

from Tensile.Tests.unit.test_SubtileBasedLogicalScheduler import (
    create_kernel,
    makeTileInfo,
)

pytestmark = pytest.mark.unit


class _Pool:
    def __init__(self, base=0):
        self._next = base
        self.outstanding = set()

    def checkOut(self, n, tag=None, preventOverflow=None, align=None):
        idx = self._next
        self._next += n
        self.outstanding.add(idx)
        return idx

    def checkOutAligned(self, n, align, tag=None, preventOverflow=None):
        return self.checkOut(n, tag=tag)

    def checkIn(self, v):
        self.outstanding.discard(v)

    def size(self):
        return 0


class _Writer:
    def __init__(self):
        self.vgprPool = _Pool()
        self.sgprPool = _Pool(100)
        self.agprPool = _Pool()
        self.states = type("S", (), {"regCaps": {
            "PhysicalMaxVgpr": 512, "MaxVgpr": 256,
            "MaxSgpr": 102, "PhysicalMaxSgpr": 102,
        }})()


def _mx_kernel(fmt):
    kernel = create_kernel(64, 64, fp4=True, depthU=256, miWaveGroup=[2, 2])
    kernel["MXScaleFormat"] = fmt
    return kernel


@pytest.mark.parametrize(
    "fmt, expected",
    [
        ("NoSwizzle", False),
        ("HostPreSwizzle", True),
        ("InMemorySwizzle", True),
    ],
)
def test_tileinfo_is_swizzled_follows_mx_scale_format(fmt, expected):
    ti = makeTileInfo("MXSA", _mx_kernel(fmt))
    assert ti.isSwizzled is expected


def test_tileinfo_is_swizzled_defaults_false_for_omitted_format():
    kernel = create_kernel(64, 64, fp4=True, depthU=256, miWaveGroup=[2, 2])
    assert "MXScaleFormat" not in kernel
    ti = makeTileInfo("MXSA", kernel)
    assert ti.isSwizzled is False


def test_alloc_groffset_swap_only_for_noswizzle():
    writer = _Writer()
    kernel = _mx_kernel("NoSwizzle")
    ti = makeTileInfo("MXSA", kernel)
    ti.allocOffsetRegisters(writer, kernel)
    assert len(ti.sharedVgprGROffsetSwap) == 1
    assert len(ti.sharedVgprGROffset) == 1
    assert len(ti.sharedVgprLROffsetSwap) == 1


@pytest.mark.parametrize("fmt", ["HostPreSwizzle", "InMemorySwizzle"])
def test_alloc_skips_groffset_swap_for_swizzled_formats(fmt):
    writer = _Writer()
    kernel = _mx_kernel(fmt)
    ti = makeTileInfo("MXSA", kernel)
    ti.allocOffsetRegisters(writer, kernel)
    assert ti.sharedVgprGROffsetSwap == []
    assert not hasattr(ti, "_sharedVgprGROffsetSwap")

def test_dealloc_returns_noswizzle_groffset_swap():
    writer = _Writer()
    kernel = _mx_kernel("NoSwizzle")
    ti = makeTileInfo("MXSA", kernel)
    ti.allocOffsetRegisters(writer, kernel)
    allocated = set(writer.vgprPool.outstanding)
    assert allocated  # GR, GRSwap, LR, LRSwap
    ti.deallocOffsetRegisters(writer, kernel)
    assert writer.vgprPool.outstanding == set()
