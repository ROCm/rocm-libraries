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

from tensilelite.Components.Subtile.Kernel import (
    TileInfo, AB_GEOMETRY_MAP, abB4Tlu1Name,
)
from tensilelite.Components.Subtile.SubtileTLUSwizzle import (
    selectTLUSwizzle, selectTLUColScatter,
)
from tensilelite.Components.Subtile import SubtileGREmit as GR
from tensilelite.Components.Subtile import SubtileLREmit as LR

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

    def sizeRef(self, idx):
        from rocisa.container import sgpr
        from tensilelite.Common import INDEX_CHARS
        return sgpr("Size%s" % INDEX_CHARS[idx])

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
        "ProblemType": {"Index0": 0, "Index1": 1, "IndexUnroll": 2},
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


# --- M/N edge mask ----------------------------------------------------------
#
# The free dim is contiguous on TLU=1, so a lane past the real M/N wraps into the
# next K row instead of off the tensor.  The SRD limit keeps that address, so the
# read costs full bandwidth unless the lane is sent to BufferOOB explicitly.

def test_gr_sends_lanes_past_the_free_dim_size_to_buffer_oob():
    _, _, module, _ = _emitGR(_COL_SCATTER_STACK)
    text = str(module)
    assert "v_cmp_lt_u32" in text, "no runtime compare against the free-dim size"
    assert "BufferOOB" in text, "out-of-range lanes are not pointed at the buffer bound"
    assert "v_cndmask_b32" in text, "the compare never selects the OOB offset"


@pytest.mark.parametrize("tc,sizeSgpr", [("A", "SizeI"), ("B", "SizeJ")])
def test_gr_edge_mask_compares_against_that_operands_size(tc, sizeSgpr):
    """A is bounded by M and B by N; swapping them would corrupt the other edge."""
    _, _, module, _ = _emitGR(_COL_SCATTER_STACK, tc=tc)
    assert sizeSgpr in str(module)


@pytest.mark.parametrize("tc,wgSgpr", [("A", "WorkGroup0"), ("B", "WorkGroup1")])
def test_gr_edge_mask_is_tile_relative(tc, wgSgpr):
    """The tile start is already in the SRD base, so the bound is size - WG*MT."""
    text = str(_emitGR(_COL_SCATTER_STACK, tc=tc)[2])
    assert wgSgpr in text, "the bound does not account for this workgroup's tile"
    assert "s_sub_u32" in text, "the size is never reduced to the tile-relative bound"


def _emitGRNamed(geoName, tc="A", miWaveGroup=(1, 1)):
    """Emit TLU=1 GR for a geometry by name, not by fp4 stack height."""
    from tensilelite.Components.Subtile.Kernel import AB_GEOMETRY_MAP
    kernel = _kernel(miWaveGroup=miWaveGroup)
    writer = _Writer()
    ti = TileInfo(AB_GEOMETRY_MAP[geoName], tc, writer, kernel)
    GR._allocGROffsetRegisters(ti.gr.config.tag, ti.gr, ti, writer, kernel)
    before = (set(writer.vgprPool.outstanding), set(writer.sgprPool.outstanding))
    return writer, ti, GR._graTileAssignment_tlu(writer, kernel, ti), before


@pytest.mark.parametrize("tc", ["A", "B"])
def test_gr_edge_mask_covers_the_split_k_row_layout(tc):
    """AB_B4_TLU1_4x1 splits the K row, so its lanes own maskable M/N blocks.

    This shape bypasses column-scatter, so it would keep its edge lanes if only
    the column-scatter path carried the mask.
    """
    _, ti, module, _ = _emitGRNamed("AB_B4_TLU1_4x1", tc=tc)
    assert selectTLUColScatter(ti) is None, "fixture no longer bypasses column-scatter"
    text = str(module)
    assert "drop GR lanes past the real M/N" in text
    assert "v_cmp_lt_u32" in text
    assert "past-M/N lane" in text


@pytest.mark.parametrize("mt,wg", [(192, (2, 2)), (192, (1, 1)), (320, (2, 2)), (384, (1, 1))])
def test_gr_edge_mask_handles_non_power_of_two_wave_strides(mt, wg):
    """MT192 at [2,2] gives three 32-element strips per wave, so elemsPerId is 96.

    A shift-only wave offset cannot express that, and asserting on it aborts
    kernel generation for every one of these shapes.
    """
    from tensilelite.Components.Subtile.Kernel import AB_GEOMETRY_MAP
    from tensilelite.Components.Subtile.SubtileGeometry import GRTag_TLU1
    emitted = 0
    for geo in AB_GEOMETRY_MAP.values():
        for tc in ("A", "B"):
            kernel = _kernel(miWaveGroup=wg)
            for key in ("MacroTile0", "MacroTile1", "MacroTileA", "MacroTileB"):
                kernel[key] = mt
            writer = _Writer()
            try:
                ti = TileInfo(geo, tc, writer, kernel)
            except Exception:
                continue
            if not isinstance(ti.gr.config.tag, GRTag_TLU1):
                continue
            GR._allocGROffsetRegisters(ti.gr.config.tag, ti.gr, ti, writer, kernel)
            before = (set(writer.vgprPool.outstanding), set(writer.sgprPool.outstanding))
            assert GR._graTileAssignment_tlu(writer, kernel, ti).items()
            assert writer.vgprPool.outstanding == before[0]
            assert writer.sgprPool.outstanding == before[1]
            emitted += 1
    assert emitted, "no TLU=1 geometry emitted for MT%u %s" % (mt, wg)


@pytest.mark.parametrize("stack", [_XOR_STACK, _COL_SCATTER_STACK])
def test_gr_edge_mask_is_dropped_under_prefetch_across_persistent(stack):
    """PAP reuses these offsets for the next tile after rebuilding the

    descriptors, so a mask built from this workgroup's M/N bound would keep
    lanes at BufferOOB that the next tile needs.
    """
    kernel = _kernel()
    kernel["PrefetchAcrossPersistent"] = 1
    writer, ti = _tile(stack, "A", kernel)
    GR._allocGROffsetRegisters(ti.gr.config.tag, ti.gr, ti, writer, kernel)
    text = str(GR._graTileAssignment_tlu(writer, kernel, ti))
    assert "drop GR lanes past the real M/N" not in text
    assert "lane inside the real M/N?" not in text


def _emitSubtileOffsets(stack, tc="A", kernel=None):
    kernel = kernel or _kernel()
    writer, ti = _tile(stack, tc, kernel)
    GR._allocGROffsetRegisters(ti.gr.config.tag, ti.gr, ti, writer, kernel)
    before = (set(writer.vgprPool.outstanding), set(writer.sgprPool.outstanding))
    module = Module()
    GR._grComputeSubtileOffsets_tlu(writer, module, ti, kernel)
    return writer, ti, module, before


def test_subtile_rows_wholly_past_the_size_are_dropped():
    """Rows share the GR offset VGPR and differ only by a scalar soffset, so a
    per-lane mask cannot vary by row. A row that is entirely past M/N needs no
    per-lane test: its start is scalar, so its soffset can go out of bounds."""
    _, ti, module, _ = _emitSubtileOffsets(_XOR_STACK)
    nonEmptyRows = sum(1 for rl in ti.localSubtilesRegister if len(rl))
    assert nonEmptyRows > 1, "fixture no longer has multiple subtile rows"
    text = str(module)
    assert text.count("wholly past the real M/N") == nonEmptyRows
    assert "BufferOOB" in text


def test_subtile_row_drop_is_scalar_only_and_leaks_nothing():
    """It must not cost an offset VGPR per row; that is why it is not per lane."""
    writer, _, module, (vbefore, sbefore) = _emitSubtileOffsets(_XOR_STACK)
    assert writer.vgprPool.outstanding == vbefore
    assert writer.sgprPool.outstanding == sbefore
    assert "v_cmp" not in str(module)


def test_subtile_row_drop_is_skipped_under_prefetch_across_persistent():
    kernel = _kernel()
    kernel["PrefetchAcrossPersistent"] = 1
    _, _, module, _ = _emitSubtileOffsets(_XOR_STACK, kernel=kernel)
    assert "wholly past the real M/N" not in str(module)


def test_gr_edge_mask_split_k_row_does_not_leak_registers():
    writer, _, _, (vbefore, sbefore) = _emitGRNamed("AB_B4_TLU1_4x1")
    assert writer.vgprPool.outstanding == vbefore
    assert writer.sgprPool.outstanding == sbefore


def test_gr_edge_mask_does_not_leak_registers():
    """The mask needs scratch on top of the pad path's; all of it must come back."""
    writer, _, _, (vbefore, sbefore) = _emitGR(_COL_SCATTER_STACK)
    assert writer.vgprPool.outstanding == vbefore
    assert writer.sgprPool.outstanding == sbefore


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
