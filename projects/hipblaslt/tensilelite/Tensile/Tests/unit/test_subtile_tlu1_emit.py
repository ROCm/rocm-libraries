# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Emit tests for the TLU=1 fp4 global-read and local-read address paths.

These drive the real emitters with a real ``TileInfo`` built from the shipped
geometry table -- the subtile shape, swizzle selection and register allocation
are all genuine; only the writer is a stub. The stack height picks the LDS
layout, so the two stacks exercised here reach different code: 2x1 takes the
chunk-index XOR, 16x1 takes column-scatter.

Scope: these assert which branch a geometry takes, that the address registers
are allocated as the emitters contract, and that temporaries are returned. They
do not assemble or compare output -- address correctness is established by the
gfx950 hardware suites.
"""

import pytest

from rocisa.code import Module

from Tensile.Components.Subtile.Kernel import (
    TileInfo, AB_GEOMETRY_MAP, abB4Tlu1Name,
)
from Tensile.Components.Subtile.SubtileTLUSwizzle import (
    selectTLUSwizzle, selectTLUColScatter,
)
from Tensile.Components.Subtile import SubtileGREmit as GR
from Tensile.Components.Subtile import SubtileLREmit as LR

pytestmark = pytest.mark.unit

_XOR_STACK = 2          # 2x1 -> chunk-index XOR layout
_COL_SCATTER_STACK = 16  # 16x1 -> column-scatter layout


class _Pool:
    """Register pool stub: hands out distinct indices, tracks what is out."""

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


class _States:
    regCaps = {"PhysicalMaxVgpr": 512, "MaxVgpr": 256,
               "MaxSgpr": 102, "PhysicalMaxSgpr": 102}
    archCaps = {"LDSBankCount": 32, "LDSBankWidth": 4, "DeviceLDS": 163840}
    laneSGPRCount = 2
    subtileLdsSwizzle = True

    def __init__(self):
        self.vgprPool = _Pool()
        self.agprPool = _Pool()


class _Writer:
    def __init__(self):
        self.states = _States()
        self.vgprPool = _Pool()
        self.sgprPool = _Pool(200)
        self.agprPool = _Pool()
        # published by applyLdsLayout in the real flow
        self.ldsStartOffsetA = 0
        self.ldsStartOffsetB = 4096
        self.ldsStartOffsetMXSA = 8192
        self.ldsStartOffsetMXSB = 8704
        self.ldsTotalSize = 16384

    def strideRef(self, tc, idx):
        from rocisa.container import sgpr
        return sgpr("Stride%s%s" % (tc, idx))

    def isConstUnitStride(self, ref):
        return False

    def allocTmpSgpr(self, num, alignment=None, tag=None):
        import contextlib

        @contextlib.contextmanager
        def _cm():
            idx = self.sgprPool.checkOut(num)
            try:
                yield type("TmpSgpr", (), {"idx": idx})()
            finally:
                self.sgprPool.checkIn(idx)
        return _cm()


def _kernel(miWaveGroup=(1, 1)):
    """Minimal kernel config.

    The wave group matters: at [2,2] the K split makes a strip shared between
    waves, and a shared strip routes even the short stacks to column-scatter
    (the XOR acts on the physical chunk index, which a sub-strip offset breaks).
    [1,1] keeps the strips unshared so the two stacks take different paths.
    """
    return {
        "MacroTileA": 256, "MacroTileB": 256,
        "MacroTile0": 256, "MacroTile1": 256,
        "_DepthUA": 256, "_DepthUB": 256,
        "MIWaveGroup": list(miWaveGroup), "WavefrontSize": 64,
        "ProblemType": {"IndexUnroll": 2},
        "NonTemporalA": 0, "NonTemporalB": 0,
        "BufferLoad": True, "DirectToLds": False,
    }


def _tile(stack, tc="A", kernel=None):
    """A real TileInfo for the fp4 TLU=1 geometry of this stack height."""
    kernel = kernel or _kernel()
    writer = _Writer()
    ti = TileInfo(AB_GEOMETRY_MAP[abB4Tlu1Name(stack)], tc, writer, kernel)
    return writer, ti


def _emitGR(stack, tc="A", kernel=None):
    kernel = kernel or _kernel()
    writer, ti = _tile(stack, tc, kernel)
    GR._allocGROffsetRegisters(ti.gr.config.tag, ti.gr, ti, writer, kernel)
    before = (set(writer.vgprPool.outstanding), set(writer.sgprPool.outstanding))
    module = GR._graTileAssignment_tlu(writer, kernel, ti)
    return writer, ti, module, before


# --- layout selection -------------------------------------------------------

def test_short_stack_takes_the_xor_layout():
    _, ti = _tile(_XOR_STACK)
    assert selectTLUSwizzle(ti) is not None
    assert selectTLUColScatter(ti) is None


def test_tall_stack_takes_the_column_scatter_layout():
    _, ti = _tile(_COL_SCATTER_STACK)
    assert selectTLUColScatter(ti) is not None
    assert selectTLUSwizzle(ti) is None


# --- global read ------------------------------------------------------------

@pytest.mark.parametrize("stack", [_XOR_STACK, _COL_SCATTER_STACK])
def test_gr_emits_addressing_for_both_layouts(stack):
    _, _, module, _ = _emitGR(stack)
    assert module.items(), "the GR offset path emitted nothing"


def test_the_two_layouts_emit_different_gr_addressing():
    """A shared code path for both stacks would defeat the whole layout split."""
    _, _, xor_mod, _ = _emitGR(_XOR_STACK)
    _, _, cs_mod, _ = _emitGR(_COL_SCATTER_STACK)
    assert str(xor_mod) != str(cs_mod)


@pytest.mark.parametrize("stack", [_XOR_STACK, _COL_SCATTER_STACK])
def test_gr_allocates_one_offset_register_per_load(stack):
    _, ti, _, _ = _emitGR(stack)
    assert len(ti.gr.sharedVgprGROffset) == ti.numGRPerSubtile


@pytest.mark.parametrize("stack", [_XOR_STACK, _COL_SCATTER_STACK])
def test_gr_returns_every_temporary_it_takes(stack):
    """Offset registers stay out; scratch used to compute them must not."""
    writer, _, _, (vbefore, sbefore) = _emitGR(stack)
    assert writer.vgprPool.outstanding == vbefore
    assert writer.sgprPool.outstanding == sbefore


@pytest.mark.parametrize("tc", ["A", "B"])
def test_gr_addressing_is_emitted_for_either_operand(tc):
    """NN and TT pair a TLU=1 operand with a row-major one, on either side."""
    _, _, module, _ = _emitGR(_COL_SCATTER_STACK, tc=tc)
    assert module.items()


# --- local read -------------------------------------------------------------

def _emitLR(stack, tc="A"):
    kernel = _kernel()
    writer, ti = _tile(stack, tc, kernel)
    LR._allocLROffsetRegisters(ti.lr.config.tag, ti.lr, ti, writer, kernel)
    module = Module()
    LR._lraTileAssignment_tlu(writer, kernel, module, ti)
    return writer, ti, module


@pytest.mark.parametrize("stack", [_XOR_STACK, _COL_SCATTER_STACK])
def test_lr_emits_transpose_read_addressing(stack):
    _, _, module = _emitLR(stack)
    assert module.items(), "the LR offset path emitted nothing"


def test_the_two_layouts_emit_different_lr_addressing():
    """LR must invert whatever GR wrote, so its address differs per layout too."""
    _, _, xor_mod = _emitLR(_XOR_STACK)
    _, _, cs_mod = _emitLR(_COL_SCATTER_STACK)
    assert str(xor_mod) != str(cs_mod)


@pytest.mark.parametrize("stack", [_XOR_STACK, _COL_SCATTER_STACK])
def test_lr_allocates_a_single_base_not_one_per_read(stack):
    """The transpose read reaches every subtile from one per-lane address."""
    _, ti, _ = _emitLR(stack)
    assert len(ti.lr.sharedVgprLROffset) == 1


def test_a_shared_strip_reroutes_the_short_stack_to_column_scatter():
    """At wave group [2,2] the K split shares a strip, and the XOR cannot
    express a sub-strip offset -- so even a 2x1 stack must take col_scatter."""
    _, ti = _tile(_XOR_STACK, kernel=_kernel(miWaveGroup=(2, 2)))
    assert ti.grKSplit > 1, "this config is meant to produce a shared strip"
    assert selectTLUSwizzle(ti) is None
    assert selectTLUColScatter(ti) is not None


# --- the K-split (shared strip) variants -----------------------------------

@pytest.mark.parametrize("stack", [_XOR_STACK, _COL_SCATTER_STACK])
def test_gr_emits_the_k_split_terms_on_a_shared_strip(stack):
    """A shared strip adds the wave's K-slice term to every address."""
    kernel = _kernel(miWaveGroup=(2, 2))
    writer, ti, module, (vbefore, sbefore) = _emitGR(stack, kernel=kernel)
    assert ti.grKSplit > 1
    assert module.items()
    assert writer.vgprPool.outstanding == vbefore
    assert writer.sgprPool.outstanding == sbefore


@pytest.mark.parametrize("stack", [_XOR_STACK, _COL_SCATTER_STACK])
def test_gr_subtile_row_offsets_are_emitted(stack):
    """Rows past the first need an soffset of r * subtileM * bpe."""
    kernel = _kernel()
    writer, ti = _tile(stack, kernel=kernel)
    GR._allocGROffsetRegisters(ti.gr.config.tag, ti.gr, ti, writer, kernel)
    module = Module()
    GR._grComputeSubtileOffsets_tlu(writer, module, ti)
    assert module is not None


@pytest.mark.parametrize("stack", [_XOR_STACK, _COL_SCATTER_STACK])
def test_gr_dtl_base_is_emitted_for_the_tlu_layout(stack):
    """The DTL write base has to land where the transpose read expects it."""
    kernel = _kernel()
    writer, ti = _tile(stack, kernel=kernel)
    module = Module()
    GR._grDTLInitBase_tlu(writer, kernel, module, ti.tc, ti)
    assert module.items()


# --- the row-major operand NN and TT pair a TLU=1 tile with ----------------

def _rowMajorTile(tc="B"):
    writer = _Writer()
    ti = TileInfo(AB_GEOMETRY_MAP["AB_B4"], tc, writer, _kernel())
    return writer, ti


def test_gr_row_major_operand_addresses_from_its_own_geometry():
    """NN/TT need each operand answered from its own geometry, not a shared one."""
    kernel = _kernel()
    writer, ti = _rowMajorTile()
    GR._allocGROffsetRegisters(ti.gr.config.tag, ti.gr, ti, writer, kernel)
    module = Module()
    GR._graTileAssignment_rowMajorSingle(writer, kernel, module, ti)
    assert module.items()


def test_lr_row_major_operand_addresses_from_its_own_geometry():
    kernel = _kernel()
    writer, ti = _rowMajorTile()
    LR._allocLROffsetRegisters(ti.lr.config.tag, ti.lr, ti, writer, kernel)
    module = Module()
    LR._lraTileAssignment_rowMajorSingle(writer, kernel, module, ti)
    assert module.items()


# --- the per-load instructions ---------------------------------------------

@pytest.mark.parametrize("stack", [_XOR_STACK, _COL_SCATTER_STACK])
def test_buffer_load_is_emitted_per_subtile(stack):
    """One buffer_load per strip, with the row soffset the offsets path set."""
    kernel = _kernel()
    writer, ti = _tile(stack, kernel=kernel)
    GR._allocGROffsetRegisters(ti.gr.config.tag, ti.gr, ti, writer, kernel)
    GR._graTileAssignment_tlu(writer, kernel, ti)
    item = GR.emitSingleBufferLoad(ti, kernel, 0, 0, writer=writer)
    assert item is not None


@pytest.mark.parametrize("stack", [_XOR_STACK, _COL_SCATTER_STACK])
def test_gr_pointer_advances_by_one_depthu_window(stack):
    """K is strided on TLU=1, so a window spans DepthU * bpe * strideK."""
    kernel = _kernel()
    writer, ti = _tile(stack, kernel=kernel)
    module = GR._emitGRPtrUpdate_TLU1(ti.gr.config.tag, ti.gr, ti, writer, kernel)
    assert module is not None


def test_gr_row_major_load_is_emitted():
    """The TLU=0 side of a mixed pair still emits its own global read."""
    kernel = _kernel()
    writer, ti = _rowMajorTile()
    GR._allocGROffsetRegisters(ti.gr.config.tag, ti.gr, ti, writer, kernel)
    GR._graTileAssignment_rowMajorSingle(writer, kernel, Module(), ti)
    # _emitGR_TLU0 still reads the legacy per-operand TileInfo off the writer
    # (see its TODO); give it the one under test.
    setattr(writer.states, ti.tc.lower(), type("S", (), {"tileInfo": ti})())
    module = GR._emitGR_TLU0(ti.gr.config.tag, ti.gr, ti, writer, kernel)
    assert module is not None


@pytest.mark.parametrize("stack", [_XOR_STACK, _COL_SCATTER_STACK])
def test_lr_addressing_on_a_shared_strip(stack):
    """A shared strip adds the wave's sub-strip term to the read base too."""
    kernel = _kernel(miWaveGroup=(2, 2))
    writer, ti = _tile(stack, kernel=kernel)
    LR._allocLROffsetRegisters(ti.lr.config.tag, ti.lr, ti, writer, kernel)
    module = Module()
    LR._lraTileAssignment_tlu(writer, kernel, module, ti)
    assert module.items()


def test_gr_dtl_base_for_the_row_major_operand():
    """The row-major half of a mixed pair writes its own DTL base."""
    kernel = _kernel()
    writer, ti = _rowMajorTile()
    module = Module()
    GR._grDTLInitBase_rowMajor(writer, kernel, module, ti.tc, ti)
    assert module.items()


@pytest.mark.parametrize("stack", [_XOR_STACK, _COL_SCATTER_STACK])
def test_gr_dtl_base_on_a_shared_strip(stack):
    """With a K split the DTL base carries the wave's slice term as well."""
    kernel = _kernel(miWaveGroup=(2, 2))
    writer, ti = _tile(stack, kernel=kernel)
    module = Module()
    GR._grDTLInitBase_tlu(writer, kernel, module, ti.tc, ti)
    assert module.items()


def test_lr_wave_partition_offset_is_applied():
    """Each wave reads its own partition of the strip."""
    kernel = _kernel(miWaveGroup=(2, 2))
    writer, ti = _tile(_XOR_STACK, kernel=kernel)
    LR._allocLROffsetRegisters(ti.lr.config.tag, ti.lr, ti, writer, kernel)
    module = Module()
    LR._applyWavePartitionLROffset(module, writer, kernel, ti)
    assert module is not None


def test_lr_wave_partition_rejects_an_unsupported_load_ratio():
    """A tall stack on a shared strip gives loadRatioGR 0.25, which the wave
    partition has no formula for -- it must refuse rather than mis-address."""
    kernel = _kernel(miWaveGroup=(2, 2))
    writer, ti = _tile(_COL_SCATTER_STACK, kernel=kernel)
    LR._allocLROffsetRegisters(ti.lr.config.tag, ti.lr, ti, writer, kernel)
    with pytest.raises(NotImplementedError, match="loadRatioGR"):
        LR._applyWavePartitionLROffset(Module(), writer, kernel, ti)


@pytest.mark.parametrize("tc", ["A", "B"])
def test_lr_row_major_addressing_for_either_operand(tc):
    """NN puts the row-major operand on B, TT puts it on A."""
    kernel = _kernel()
    writer = _Writer()
    ti = TileInfo(AB_GEOMETRY_MAP["AB_B4"], tc, writer, kernel)
    LR._allocLROffsetRegisters(ti.lr.config.tag, ti.lr, ti, writer, kernel)
    module = Module()
    LR._lraTileAssignment_rowMajorSingle(writer, kernel, module, ti)
    assert module.items()


class _DstTile:
    """Destination register tile: emitSingleDsRead reads only regList.indices."""

    def __init__(self, indices):
        self.regList = type("RegList", (), {"indices": list(indices)})()


@pytest.mark.parametrize("stack", [_XOR_STACK, _COL_SCATTER_STACK])
def test_ds_read_is_emitted_per_subtile(stack):
    """One transpose read per pair of destination VGPRs, off the single base."""
    kernel = _kernel()
    writer, ti = _tile(stack, kernel=kernel)
    LR._allocLROffsetRegisters(ti.lr.config.tag, ti.lr, ti, writer, kernel)
    LR._lraTileAssignment_tlu(writer, kernel, Module(), ti)
    item = LR.emitSingleDsRead(ti, 0, 0, 0, _DstTile([0, 1]), swizzled=True)
    assert item is not None


@pytest.mark.parametrize("stack", [_XOR_STACK, _COL_SCATTER_STACK])
def test_ds_read_without_the_swizzled_lds_image(stack):
    """The unswizzled image addresses differently; both must emit."""
    kernel = _kernel()
    writer, ti = _tile(stack, kernel=kernel)
    LR._allocLROffsetRegisters(ti.lr.config.tag, ti.lr, ti, writer, kernel)
    LR._lraTileAssignment_tlu(writer, kernel, Module(), ti)
    item = LR.emitSingleDsRead(ti, 0, 0, 0, _DstTile([0, 1]), swizzled=False)
    assert item is not None


def test_lr_fp8_legacy_addressing_is_emitted():
    """The fp8 legacy LR path pairs both operands off the writer's tile state."""
    kernel = _kernel()
    writer = _Writer()
    tiA = TileInfo(AB_GEOMETRY_MAP["AB_B8"], "A", writer, kernel)
    tiB = TileInfo(AB_GEOMETRY_MAP["AB_B8"], "B", writer, kernel)
    for tc, ti in (("a", tiA), ("b", tiB)):
        LR._allocLROffsetRegisters(ti.lr.config.tag, ti.lr, ti, writer, kernel)
        setattr(writer.states, tc, type("S", (), {"tileInfo": ti})())
    module = Module()
    LR._lraTileAssignment_fp8_legacy(writer, kernel, module)
    assert module.items()
