# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Emit test for the StreamK K-start advance of the dual-period MX scale SRDs.

computeLoadSrd sizes a dual-period scale window as (numLine+1)*span measured
from the tile start, then StreamK folds StreamKLocalStart*depthU*stride into the
SRD base. A work item that begins mid-tile therefore ends up with a window that
hangs over the end of the scale allocation by exactly that offset, and because
the MX scale tensors are small enough to end on a mapping boundary the first
prefetch reads unmapped memory. num_records has to shrink by the same amount the
base advanced, the way the per-epoch advance in SubtileScaleEmit already does.

A and B are deliberately left out: their Srd+2 is a tile-boundary window with
its own slack, and shrinking it clips legitimate top-group reads.
"""

import pytest

from rocisa.code import Module
from rocisa.container import sgpr
from rocisa.instruction import SAddU32, SSubU32

from tensilelite.Components.StreamK import StreamKTwoTileDPFirst

pytestmark = pytest.mark.unit

_STMP = 50
_TILE_START = _STMP + 2


class _Writer:
    """Stub with only what computeLoadSrdCommon reads."""

    def strideRef(self, tc, dim):
        return sgpr("Stride%s+0" % tc)

    def s_mul_u64_u32(self, dst0, dst1, src0, src1, comment=""):
        # The real 64-bit multiply leaves the scaled offset in place; nothing
        # here asserts on its expansion.
        return Module("s_mul_u64_u32 stub")


def _kernel(dualPeriod=True):
    kernel = {
        "DepthU": 128,
        "UseSubtileImpl": True,
        "MXScaleFormat": "HostPreSwizzle",
        # 8 * 32 = one 256-K scale epoch, reused by two data iterations.
        "_DepthUMXSA": 8,
        "_DepthUMXSB": 8,
        "_ScaleReuseA": 2,
        "_ScaleReuseB": 2,
        "ProblemType": {"Sparse": 0, "IndicesSummation": [3]},
    }
    if dualPeriod:
        kernel["_DualPeriodMXFP8"] = True
    return kernel


def _emit(tc, dualPeriod=True):
    tP = {"tensorChar": tc, "bpe": 1, "bpeGR": 1}
    module = StreamKTwoTileDPFirst().computeLoadSrd(
        _Writer(), _kernel(dualPeriod), tP, _STMP)
    return list(module.flatitems())


def _only(items, kind, dst):
    found = [i for i in items if isinstance(i, kind) and str(i.dst) == dst]
    assert len(found) <= 1, "expected at most one %s on %s, got %d" % (
        kind.__name__, dst, len(found))
    return found[0] if found else None


@pytest.mark.parametrize("tc", ["MXSA", "MXSB"])
def test_scale_window_shrinks_by_the_base_advance(tc):
    """num_records drops by the same register that advanced the base."""
    items = _emit(tc)

    baseAdvance = _only(items, SAddU32, str(sgpr(_TILE_START)))
    assert baseAdvance is not None, "StreamK K-start offset no longer reaches tileStart"
    shrink = _only(items, SSubU32, str(sgpr("Srd%s+2" % tc)))
    assert shrink is not None, "%s num_records is not shrunk by the K-start advance" % tc

    # Same offset register on both, so the window still ends where the
    # allocation does however the offset was computed.
    offset = [str(s) for s in baseAdvance.srcs if str(s) != str(baseAdvance.dst)]
    assert [str(s) for s in shrink.srcs if str(s) != str(shrink.dst)] == offset

    # And it has to come after the add, which is what consumes the live offset.
    assert items.index(shrink) > items.index(baseAdvance)


@pytest.mark.parametrize("tc", ["A", "B"])
def test_data_operand_window_is_left_alone(tc):
    assert not [i for i in _emit(tc) if isinstance(i, SSubU32)]


@pytest.mark.parametrize("tc", ["MXSA", "MXSB", "A", "B"])
def test_single_period_path_is_untouched(tc):
    """Scoped to dual-period MXFP8; every other kernel emits what it did before."""
    assert not [i for i in _emit(tc, dualPeriod=False) if isinstance(i, SSubU32)]
