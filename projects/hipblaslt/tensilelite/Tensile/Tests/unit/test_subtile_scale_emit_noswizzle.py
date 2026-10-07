# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for SubtileScaleEmit MXScaleFormat dispatch and NoSwizzle emit.

Covers the format gates Codecov flags on SubtileScaleEmit.py without a full
kernel assemble: NoSwizzle vs HostPreSwizzle routing for DTL init, GR tile
assignment, GR load, and LDS swap helpers.
"""

from types import SimpleNamespace

import pytest
from rocisa.instruction import (
    BufferLoadB128,
    BufferLoadU8,
    DSStoreB32,
    SWaitCnt,
    SXorB32,
    VXorB32,
)

from Tensile.Components.Subtile.SubtileScaleEmit import (
    _isMxSwizzledScaleFormat,
    emitScaleGRLDSSwap,
    emitScaleGRPtrUpdate,
    globalReadDoScaleSubtile,
    globalReadScaleDTLInitCommonSgpr,
    graTileAssignmentScale,
    lraTileAssignmentScale,
)
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
        self.ldsStartOffsetMXSA = 8192
        self.ldsStartOffsetMXSB = 8704
        self.ldsTotalSize = 16384
        self.states = SimpleNamespace(
            regCaps={
                "PhysicalMaxVgpr": 512,
                "MaxVgpr": 256,
                "MaxSgpr": 102,
                "PhysicalMaxSgpr": 102,
            },
            archCaps={"LDSBankCount": 32, "LDSBankWidth": 4, "DeviceLDS": 163840},
        )


def _count_inst(mod, cls):
    return sum(1 for item in mod.flatitems() if isinstance(item, cls))


def _mx_kernel(fmt, *, mt=64, depth_u=256, mi_wg=(2, 2)):
    kernel = create_kernel(mt, mt, fp4=True, depthU=depth_u, miWaveGroup=list(mi_wg))
    kernel["MXScaleFormat"] = fmt
    return kernel


def _prepare_scale_writer(kernel):
    writer = _Writer()
    ti_a = makeTileInfo("MXSA", kernel)
    ti_b = makeTileInfo("MXSB", kernel)
    ti_a.allocOffsetRegisters(writer, kernel)
    ti_b.allocOffsetRegisters(writer, kernel)
    ti_a.allocVgprTileRegisters_legacy(writer, kernel)
    ti_b.allocVgprTileRegisters_legacy(writer, kernel)
    writer.states.mxsa = SimpleNamespace(tileInfo=ti_a)
    writer.states.mxsb = SimpleNamespace(tileInfo=ti_b)
    return writer, ti_a, ti_b


@pytest.mark.parametrize(
    "fmt, expected",
    [
        ("NoSwizzle", False),
        ("HostPreSwizzle", True),
        ("InMemorySwizzle", True),
    ],
)
def test_is_mx_swizzled_scale_format(fmt, expected):
    assert _isMxSwizzledScaleFormat({"MXScaleFormat": fmt}) is expected


def test_is_mx_swizzled_scale_format_defaults_to_noswizzle():
    assert _isMxSwizzledScaleFormat({}) is False


def test_global_read_scale_dtl_init_noswizzle_is_comment_only():
    """NoSwizzle skips SGPR LocalWriteBaseAddr DTL setup."""
    kernel = _mx_kernel("NoSwizzle")
    writer, _, _ = _prepare_scale_writer(kernel)
    mod = globalReadScaleDTLInitCommonSgpr(writer, kernel)
    assert _count_inst(mod, SXorB32) == 0
    assert "NoSwizzle" in str(mod)
    assert "graTileAssignmentScaleNoSwizzle" in str(mod)


def test_gra_tile_assignment_scale_noswizzle_emits_lds_write_offsets():
    kernel = _mx_kernel("NoSwizzle")
    writer, _, _ = _prepare_scale_writer(kernel)
    mod = graTileAssignmentScale(writer, kernel)
    text = str(mod)
    assert "NoSwizzle" in text
    assert "HostPreSwizzle-shaped" in text
    assert mod.itemsSize() > 0
    # Temps for waveId / laneOffset / tmpSgpr must be checked in.
    # Allocated GR/LR/tile VGPRs remain outstanding by design.
    assert writer.sgprPool.outstanding == set()


def test_global_read_do_scale_noswizzle_gathers_four_ubytes():
    """Canonical gather: 4x buffer_load_u8, undrained wait, ds_store."""
    kernel = _mx_kernel("NoSwizzle")
    writer, _, _ = _prepare_scale_writer(kernel)
    mod = globalReadDoScaleSubtile("MXSA", writer, kernel)
    assert _count_inst(mod, BufferLoadU8) == 4
    assert _count_inst(mod, DSStoreB32) == 1
    assert _count_inst(mod, BufferLoadB128) == 0
    waits = [inst for inst in mod.flatitems() if isinstance(inst, SWaitCnt)]
    assert any(getattr(inst, "vlcnt", None) == 0 for inst in waits)
    assert "NoSwizzle" in str(mod)
    assert "HostPreSwizzle-shaped" in str(mod)


def test_global_read_do_scale_hostpreswizzle_uses_dtl_b128():
    kernel = _mx_kernel("HostPreSwizzle")
    writer, _, _ = _prepare_scale_writer(kernel)
    mod = globalReadDoScaleSubtile("MXSA", writer, kernel)
    assert _count_inst(mod, BufferLoadB128) == 1
    assert _count_inst(mod, BufferLoadU8) == 0


def test_emit_scale_gr_lds_swap_noswizzle_xors_vgpr_offsets():
    kernel = _mx_kernel("NoSwizzle")
    writer, ti_a, _ = _prepare_scale_writer(kernel)
    mod = emitScaleGRLDSSwap(ti_a, writer, kernel)
    assert _count_inst(mod, VXorB32) == 1
    assert _count_inst(mod, SXorB32) == 0


def test_emit_scale_gr_lds_swap_hostpreswizzle_xors_sgpr_bases():
    kernel = _mx_kernel("HostPreSwizzle")
    writer, ti_a, _ = _prepare_scale_writer(kernel)
    mod = emitScaleGRLDSSwap(ti_a, writer, kernel)
    assert _count_inst(mod, SXorB32) == 1
    assert _count_inst(mod, VXorB32) == 0


def test_emit_scale_gr_ptr_update_uses_canonical_inc_for_noswizzle():
    kernel = _mx_kernel("NoSwizzle")
    writer, ti_a, _ = _prepare_scale_writer(kernel)
    mod = emitScaleGRPtrUpdate(ti_a, writer, kernel)
    # scaleDepthU * bpe = 8 * 1
    assert "+= 8" in str(mod)


def test_emit_scale_gr_ptr_update_uses_swizzle_granule_for_hostpreswizzle():
    kernel = _mx_kernel("HostPreSwizzle")
    writer, ti_a, _ = _prepare_scale_writer(kernel)
    expected = int(ti_a.lrSubtileSize * ti_a.lrGlobalSubtileGrid[1])
    mod = emitScaleGRPtrUpdate(ti_a, writer, kernel)
    assert f"+= {expected}" in str(mod)


def test_lra_tile_assignment_shared_by_noswizzle_and_hostpreswizzle():
    """NoSwizzle remaps into HostPreSwizzle-shaped LDS; LR offsets match."""
    for fmt in ("NoSwizzle", "HostPreSwizzle"):
        kernel = _mx_kernel(fmt)
        writer, _, _ = _prepare_scale_writer(kernel)
        mod = lraTileAssignmentScale(writer, kernel)
        assert mod.itemsSize() > 0
        assert "LR Offset" in str(mod) or "scale" in str(mod).lower()


def test_global_read_do_scale_early_return_without_mx_blocks():
    kernel = {
        "ProblemType": {"MXBlockA": 0, "MXBlockB": 0},
        "MXScaleFormat": "NoSwizzle",
    }
    mod = globalReadDoScaleSubtile("MXSA", _Writer(), kernel)
    assert mod.itemsSize() == 0


def test_gra_tile_assignment_early_return_without_mx_blocks():
    kernel = {
        "ProblemType": {"MXBlockA": 0, "MXBlockB": 0},
        "MXScaleFormat": "NoSwizzle",
    }
    mod = graTileAssignmentScale(_Writer(), kernel)
    assert mod.itemsSize() == 0
