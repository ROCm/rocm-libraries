# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the C++-backed subtile geometry facade.

``Tensile.Components.Subtile.SubtileGeometry`` is now a thin Python dataclass
facade over the compiled ``tensile_writer.subtile.geometry`` nanobind extension:
every ported value/query method forwards to C++ unconditionally (there is no
pure-Python geometry fallback). These tests therefore:

  * pin absolute gfx950 regression values through the facade, locking
    correctness of the C++ formulas as exercised by real callers, and
  * confirm the facade is a faithful pass-through by comparing each facade
    query against an independently constructed ``tensile_writer`` C++ twin
    across the existing AB / MX-scale / C-D gfx950 cases.

The extension is a hard dependency of the geometry layer, so the tests skip
only when the (build-time) extension or the rocisa ISA layer is unavailable.

PR creation for this slice is human-only: a ``human:pr`` task is filed for
Bryant Nelson only after review says merge-ready. Agents never open PRs.
"""

import pytest

# Both the ISA layer (rocisa) and the geometry extension must be present.
pytest.importorskip("rocisa")
cppgeo = pytest.importorskip("tensile_writer.subtile.geometry")

from Tensile.Components.Subtile import SubtileGeometry as sg
from Tensile.Components.Subtile.Kernel import (
    AB_B16,
    AB_B8,
    AB_B4,
    AB_B4_2x2,
    AB_B16_2x2,
    AB_B16_TLU1,
    AB_B16_TLU1_16x1,
    CD_F32,
    MXSA_B4,
    MXSB_B4,
    MXSA_B8,
    MXSB_B8,
)


# A representative AB_B16-style gfx950 kernel configuration.
KERNEL = {
    "MIWaveGroup": [4, 1],
    "WavefrontSize": 64,
    "MacroTileA": 256,
    "MacroTileB": 128,
    "_DepthUA": 128,
    "_DepthUB": 128,
}

AB_PAIRS = {
    "AB_B16": AB_B16,
    "AB_B8": AB_B8,
    "AB_B4": AB_B4,
    "AB_B4_2x2": AB_B4_2x2,
    "AB_B16_2x2": AB_B16_2x2,
    "AB_B16_TLU1": AB_B16_TLU1,
    "AB_B16_TLU1_16x1": AB_B16_TLU1_16x1,
}

MX_PAIRS = {
    "MXSA_B4": MXSA_B4,
    "MXSB_B4": MXSB_B4,
    "MXSA_B8": MXSA_B8,
    "MXSB_B8": MXSB_B8,
}

# Macro tile / depthU sweep that stays divisible by every layout's MMA tile.
MT_DU = [(256, 128), (128, 64), (512, 256), (64, 32)]


# ---------------------------------------------------------------------------
# Helpers: build an independent tensile_writer C++ twin from a facade object's
# public fields. This mirrors (but does not import) the facade's private
# ``_cpp_twin`` construction so the comparison is a genuine pass-through check.
# ---------------------------------------------------------------------------
def _assert_same(facade, cpp, ctx=""):
    """Recursively assert facade and C++ results match (float-tolerant)."""
    if isinstance(facade, (tuple, list)):
        assert isinstance(cpp, (tuple, list)), f"{ctx}: type mismatch {facade!r} vs {cpp!r}"
        assert len(facade) == len(cpp), f"{ctx}: length {len(facade)} vs {len(cpp)}"
        for i, (a, b) in enumerate(zip(facade, cpp)):
            _assert_same(a, b, f"{ctx}[{i}]")
    elif isinstance(facade, float) or isinstance(cpp, float):
        assert facade == pytest.approx(cpp), f"{ctx}: {facade!r} != {cpp!r}"
    else:
        assert facade == cpp, f"{ctx}: {facade!r} != {cpp!r}"


def _mma_twin(layout):
    return cppgeo.MMALayout(layout.instM, layout.blocks, layout.vgprs, layout.waveSize)


def _scale_twin(layout):
    return cppgeo.MMAScaleLayout(layout.instM, layout.blocks, layout.vgprs,
                                 layout.mxBlock, layout.waveSize)


def _gr_twin(gr):
    return cppgeo.ABGRGeometry(_mma_twin(gr.mmaLayout), gr.instK, float(gr.bpe),
                               cppgeo.LoadShape(gr.loadShape.m, gr.loadShape.k),
                               tuple(gr.subtileShape), gr.subtileCount,
                               gr.subtileStride, gr.tlu, gr.loadWidth)


def _lr_twin(lr):
    return cppgeo.ABLRGeometry(_mma_twin(lr.mmaLayout), lr.instK, float(lr.bpe),
                               cppgeo.LoadShape(lr.loadShape.m, lr.loadShape.k),
                               tuple(lr.subtileShape), lr.tlu, lr.loadWidth)


def _mx_gr_twin(gr):
    shape = tuple(gr.subtileShape) if gr.subtileShape is not None else None
    return cppgeo.MXScaleGRGeometry(_scale_twin(gr.scaleLayout), gr.instK,
                                    float(gr.bpe), gr.loadWidth, shape)


def _mx_lr_twin(lr):
    return cppgeo.MXScaleLRGeometry(_scale_twin(lr.scaleLayout), lr.instK,
                                    float(lr.bpe), lr.loadWidth,
                                    tuple(lr.subtileShape))


def _cd_twin(cd):
    return cppgeo.CDTileGeometry(_mma_twin(cd.mmaLayout), float(cd.bpe),
                                 cppgeo.LoadShape(cd.storeShape.m, cd.storeShape.k))


# ---------------------------------------------------------------------------
# MMALayout / MMAScaleLayout constants
# ---------------------------------------------------------------------------
class TestLayoutConstants:
    @pytest.mark.parametrize(
        "name",
        ["MFMA_16x16_1B_4K_4V", "MFMA_16x16_1B_4K_8V", "MFMA_16x16_1B_4N_4V"],
    )
    def test_mma_layout_facade_matches_ext(self, name):
        facade = getattr(sg, name)
        cpp = getattr(cppgeo, name)
        assert (facade.instM, facade.blocks, facade.vgprs, facade.waveSize) == (
            cpp.instM, cpp.blocks, cpp.vgprs, cpp.waveSize)
        assert facade.contiguousLanes == cpp.contiguousLanes
        assert facade.kGroups == cpp.kGroups
        assert facade.elementsPerLaneNonK == cpp.elementsPerLaneNonK
        assert facade.inputBytesPerLane() == cpp.inputBytesPerLane()
        for instK, eb in [(32, 2.0), (128, 0.5), (128, 1.0), (16, 16.0)]:
            assert facade.tileSizeBytes(instK, eb) == cpp.tileSizeBytes(instK, eb)
            assert facade.regsPerTile(instK, eb) == pytest.approx(
                cpp.regsPerTile(instK, eb))

    def test_mma_layout_known_values(self):
        layout = sg.MFMA_16x16_1B_4K_4V
        assert (layout.instM, layout.blocks, layout.vgprs, layout.waveSize) == (16, 1, 4, 64)
        assert layout.contiguousLanes == 16
        assert layout.kGroups == 4
        assert layout.elementsPerLaneNonK == 4
        assert layout.inputBytesPerLane() == 16
        assert layout.tileSizeBytes(32, 2.0) == 1024
        assert layout.regsPerTile(32, 2.0) == pytest.approx(4.0)

    def test_scale_layout_facade_matches_ext(self):
        facade = sg.MFMA_SCALE_16x16_1B_MX32_8V
        cpp = cppgeo.MFMA_SCALE_16x16_1B_MX32_8V
        assert (facade.instM, facade.blocks, facade.mxBlock, facade.waveSize) == (
            cpp.instM, cpp.blocks, cpp.mxBlock, cpp.waveSize)
        assert facade.vgprs == pytest.approx(cpp.vgprs)
        assert facade.contiguousLanes == cpp.contiguousLanes
        # Absolute pins for the gfx950 mxfp4 scale layout.
        assert (facade.instM, facade.blocks, facade.mxBlock, facade.waveSize) == (16, 1, 32, 64)
        assert facade.vgprs == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# A/B GR + LR geometry: facade forwards faithfully to the C++ twin.
# ---------------------------------------------------------------------------
class TestABGeometryFacade:
    @pytest.mark.parametrize("name", list(AB_PAIRS))
    @pytest.mark.parametrize("mt,du", MT_DU)
    def test_gr_queries(self, name, mt, du):
        gr = AB_PAIRS[name].gr
        twin = _gr_twin(gr)
        _assert_same(gr.globalMMATileGrid(mt, du),
                     tuple(twin.globalMMATileGrid(mt, du)), f"{name}.gr.globalMMATileGrid")
        _assert_same(gr.localMMATileGrid(mt, du, 4),
                     tuple(twin.localMMATileGrid(mt, du, 4)), f"{name}.gr.localMMATileGrid")
        _assert_same(gr.globalSubtileGrid(mt, du),
                     tuple(twin.globalSubtileGrid(mt, du)), f"{name}.gr.globalSubtileGrid")
        _assert_same(gr.subtileSizeBytes(), twin.subtileSizeBytes(),
                     f"{name}.gr.subtileSizeBytes")
        _assert_same(gr.bytesPerLoad(4), twin.bytesPerLoad(4), f"{name}.gr.bytesPerLoad")
        _assert_same(gr.loadsPerStrip(4), twin.loadsPerStrip(4), f"{name}.gr.loadsPerStrip")
        _assert_same(gr.localGRGranularity(4),
                     tuple(twin.localGRGranularity(4)), f"{name}.gr.localGRGranularity")

    @pytest.mark.parametrize("name", list(AB_PAIRS))
    @pytest.mark.parametrize("mt,du", MT_DU)
    def test_lr_queries(self, name, mt, du):
        lr = AB_PAIRS[name].lr
        twin = _lr_twin(lr)
        _assert_same(lr.globalMMATileGrid(mt, du),
                     tuple(twin.globalMMATileGrid(mt, du)), f"{name}.lr.globalMMATileGrid")
        _assert_same(lr.localMMATileGrid(mt, du, 4),
                     tuple(twin.localMMATileGrid(mt, du, 4)), f"{name}.lr.localMMATileGrid")
        _assert_same(lr.globalSubtileGrid(mt, du),
                     tuple(twin.globalSubtileGrid(mt, du)), f"{name}.lr.globalSubtileGrid")
        _assert_same(lr.subtileSizeBytes(), twin.subtileSizeBytes(),
                     f"{name}.lr.subtileSizeBytes")

    @pytest.mark.parametrize("name", list(AB_PAIRS))
    @pytest.mark.parametrize("tc", ["A", "B"])
    def test_for_kernel(self, name, tc):
        gr = AB_PAIRS[name].gr
        fk = gr.for_kernel(KERNEL, tc)
        twin_fk = _gr_twin(gr).for_kernel(KERNEL, tc)
        assert (fk.subtileCount, fk.subtileStride) == (
            twin_fk.subtileCount, twin_fk.subtileStride), f"{name}.for_kernel({tc})"
        # The facade must still return a Python dataclass carrying the tag.
        assert type(fk) is type(gr)
        assert fk.tag is gr.tag

    @pytest.mark.parametrize("name", list(AB_PAIRS))
    def test_subtile_for_mma_tile(self, name):
        gr = AB_PAIRS[name].gr.for_kernel(KERNEL, "A")
        twin = _gr_twin(gr)
        rows, cols = gr.globalMMATileGrid(KERNEL["MacroTileA"], KERNEL["_DepthUA"])
        for r in range(int(rows)):
            for c in range(int(cols)):
                facade_res = gr.subtileForMmaTile(r, c)
                tsid, tbshape, ttiles = twin.subtileForMmaTile(r, c)
                cpp_res = (tuple(tsid), tuple(tbshape), [tuple(t) for t in ttiles])
                _assert_same(facade_res, cpp_res, f"{name}.subtileForMmaTile({r},{c})")

    def test_subtile_for_mma_tile_requires_for_kernel(self):
        """The facade keeps the materialization precondition before delegating."""
        gr = AB_B16.gr  # template: subtileCount/subtileStride not yet derived
        with pytest.raises(RuntimeError):
            gr.subtileForMmaTile(0, 0)


# ---------------------------------------------------------------------------
# MX scale geometry facade
# ---------------------------------------------------------------------------
class TestMXScaleFacade:
    @pytest.mark.parametrize("name", list(MX_PAIRS))
    @pytest.mark.parametrize("mt,du", MT_DU)
    def test_gr_and_lr_queries(self, name, mt, du):
        pair = MX_PAIRS[name]
        gr_twin = _mx_gr_twin(pair.gr)
        lr_twin = _mx_lr_twin(pair.lr)
        _assert_same(pair.gr.globalMMATileGrid(mt, du),
                     tuple(gr_twin.globalMMATileGrid(mt, du)), f"{name}.gr.globalMMATileGrid")
        _assert_same(pair.lr.globalMMATileGrid(mt, du),
                     tuple(lr_twin.globalMMATileGrid(mt, du)), f"{name}.lr.globalMMATileGrid")
        _assert_same(pair.lr.globalSubtileGrid(mt, du),
                     tuple(lr_twin.globalSubtileGrid(mt, du)), f"{name}.lr.globalSubtileGrid")
        _assert_same(pair.lr.subtileSizeBytes(), lr_twin.subtileSizeBytes(),
                     f"{name}.lr.subtileSizeBytes")

    @pytest.mark.parametrize("name", list(MX_PAIRS))
    @pytest.mark.parametrize("tc", ["A", "B"])
    def test_gr_for_kernel(self, name, tc):
        gr = MX_PAIRS[name].gr
        fk = gr.for_kernel(KERNEL, tc)
        twin_fk = _mx_gr_twin(gr).for_kernel(KERNEL, tc)
        assert tuple(fk.subtileShape) == tuple(twin_fk.subtileShape), f"{name}.for_kernel({tc})"
        # The facade must still return a Python dataclass.
        assert type(fk) is type(gr)


# ---------------------------------------------------------------------------
# C/D output geometry facade
# ---------------------------------------------------------------------------
class TestCDGeometryFacade:
    @pytest.mark.parametrize("mt0,mt1", [(256, 128), (128, 128), (64, 256)])
    def test_cd_queries(self, mt0, mt1):
        cd = CD_F32
        twin = _cd_twin(cd)
        wg = (2, 2)
        ss = (1.0, 1.0)
        _assert_same(cd.globalMMATileGrid(mt0, mt1),
                     tuple(twin.globalMMATileGrid(mt0, mt1)), "CD.globalMMATileGrid")
        _assert_same(cd.localMMATileGrid(mt0, mt1, wg),
                     tuple(twin.localMMATileGrid(mt0, mt1, wg)), "CD.localMMATileGrid")
        _assert_same(cd.globalSubtileGrid(mt0, mt1, ss),
                     tuple(twin.globalSubtileGrid(mt0, mt1, ss)), "CD.globalSubtileGrid")
        _assert_same(cd.localSubtileGrid(mt0, mt1, wg, ss),
                     tuple(twin.localSubtileGrid(mt0, mt1, wg, ss)), "CD.localSubtileGrid")


# ---------------------------------------------------------------------------
# Absolute-value pins for an AB_B16 gfx950 case (lock correctness of the C++
# formulas as exercised through the public Python facade).
# ---------------------------------------------------------------------------
class TestAbsoluteValues:
    def test_ab_b16_known_values(self):
        gr = AB_B16.gr
        assert gr.mmaTileShape == (16, 32)
        assert gr.mmaTileSize == 1024
        assert tuple(gr.globalMMATileGrid(256, 128)) == (16, 4)
        assert tuple(gr.globalSubtileGrid(256, 128)) == (16.0, 2.0)
        assert gr.bytesPerLoad(4) == 4096
        fk = gr.for_kernel(KERNEL, "A")
        assert (fk.subtileCount, fk.subtileStride) == (4, 4)
        sid, bshape, tiles = fk.subtileForMmaTile(5, 3)
        assert tuple(sid) == (1, 1)
        assert tuple(bshape) == (1, 2)
        assert [tuple(t) for t in tiles] == [
            (1, 2), (1, 3), (5, 2), (5, 3),
            (9, 2), (9, 3), (13, 2), (13, 3)]
