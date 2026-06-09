# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Parity tests for the optional C++ (nanobind) subtile geometry path.

These tests compare the pure-Python geometry math in
``Tensile.Components.Subtile.SubtileGeometry`` against the compiled
``tensile_writer.subtile.geometry`` extension. They run only when the
extension is importable; otherwise they skip, so the default (Python-only)
TensileLite build is unaffected.

The comparison flips the module-level ``_USE_CPP`` delegation switch at call
time, so the *same* pre-defined geometry instances are exercised through both
the Python and the C++ code paths and asserted to produce identical results.

PR creation for this slice is human-only: a ``human:pr`` task is filed for
Bryant Nelson only after review says merge-ready. Agents never open PRs.
"""

import contextlib

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


@contextlib.contextmanager
def cpp_delegation():
    """Temporarily enable C++ delegation in SubtileGeometry."""
    saved_use, saved_cpp = sg._USE_CPP, sg._CPP
    sg._CPP = cppgeo
    sg._USE_CPP = True
    try:
        yield
    finally:
        sg._USE_CPP = saved_use
        sg._CPP = saved_cpp


def _assert_same(py, cpp, ctx=""):
    """Recursively assert Python and C++ results match (float-tolerant)."""
    if isinstance(py, (tuple, list)):
        assert isinstance(cpp, (tuple, list)), f"{ctx}: type mismatch {py!r} vs {cpp!r}"
        assert len(py) == len(cpp), f"{ctx}: length {len(py)} vs {len(cpp)}"
        for i, (a, b) in enumerate(zip(py, cpp)):
            _assert_same(a, b, f"{ctx}[{i}]")
    elif isinstance(py, float) or isinstance(cpp, float):
        assert py == pytest.approx(cpp), f"{ctx}: {py!r} != {cpp!r}"
    else:
        assert py == cpp, f"{ctx}: {py!r} != {cpp!r}"


def _both(call):
    """Run ``call()`` with the Python path and the C++ path; return (py, cpp)."""
    py = call()
    with cpp_delegation():
        cpp = call()
    return py, cpp


def _parity(call, ctx):
    py, cpp = _both(call)
    _assert_same(py, cpp, ctx)
    return py


# Macro tile / depthU sweep that stays divisible by every layout's MMA tile.
MT_DU = [(256, 128), (128, 64), (512, 256), (64, 32)]


# ---------------------------------------------------------------------------
# MMALayout / MMAScaleLayout constants
# ---------------------------------------------------------------------------
class TestLayoutConstants:
    @pytest.mark.parametrize(
        "name",
        ["MFMA_16x16_1B_4K_4V", "MFMA_16x16_1B_4K_8V", "MFMA_16x16_1B_4N_4V"],
    )
    def test_mma_layout_derived_and_methods(self, name):
        py = getattr(sg, name)
        cpp = getattr(cppgeo, name)
        assert (py.instM, py.blocks, py.vgprs, py.waveSize) == (
            cpp.instM, cpp.blocks, cpp.vgprs, cpp.waveSize)
        assert py.contiguousLanes == cpp.contiguousLanes
        assert py.kGroups == cpp.kGroups
        assert py.elementsPerLaneNonK == cpp.elementsPerLaneNonK
        assert py.inputBytesPerLane() == cpp.inputBytesPerLane()
        for instK, eb in [(32, 2.0), (128, 0.5), (128, 1.0), (16, 16.0)]:
            assert py.tileSizeBytes(instK, eb) == cpp.tileSizeBytes(instK, eb)
            assert py.regsPerTile(instK, eb) == pytest.approx(
                cpp.regsPerTile(instK, eb))

    def test_mma_layout_method_delegation_parity(self):
        layout = sg.MFMA_16x16_1B_4K_4V
        _parity(lambda: layout.inputBytesPerLane(), "inputBytesPerLane")
        _parity(lambda: layout.tileSizeBytes(32, 2.0), "tileSizeBytes")
        _parity(lambda: layout.regsPerTile(32, 2.0), "regsPerTile")

    def test_scale_layout_derived(self):
        py = sg.MFMA_SCALE_16x16_1B_MX32_8V
        cpp = cppgeo.MFMA_SCALE_16x16_1B_MX32_8V
        assert (py.instM, py.blocks, py.mxBlock, py.waveSize) == (
            cpp.instM, cpp.blocks, cpp.mxBlock, cpp.waveSize)
        assert py.vgprs == pytest.approx(cpp.vgprs)
        assert py.contiguousLanes == cpp.contiguousLanes


# ---------------------------------------------------------------------------
# A/B GR + LR geometry parity
# ---------------------------------------------------------------------------
class TestABGeometryParity:
    @pytest.mark.parametrize("name", list(AB_PAIRS))
    @pytest.mark.parametrize("mt,du", MT_DU)
    def test_gr_queries(self, name, mt, du):
        gr = AB_PAIRS[name].gr
        _parity(lambda: gr.globalMMATileGrid(mt, du), f"{name}.gr.globalMMATileGrid")
        _parity(lambda: gr.localMMATileGrid(mt, du, 4), f"{name}.gr.localMMATileGrid")
        _parity(lambda: gr.globalSubtileGrid(mt, du), f"{name}.gr.globalSubtileGrid")
        _parity(lambda: gr.subtileSizeBytes(), f"{name}.gr.subtileSizeBytes")
        _parity(lambda: gr.bytesPerLoad(4), f"{name}.gr.bytesPerLoad")
        _parity(lambda: gr.loadsPerStrip(4), f"{name}.gr.loadsPerStrip")
        _parity(lambda: gr.localGRGranularity(4), f"{name}.gr.localGRGranularity")

    @pytest.mark.parametrize("name", list(AB_PAIRS))
    @pytest.mark.parametrize("mt,du", MT_DU)
    def test_lr_queries(self, name, mt, du):
        lr = AB_PAIRS[name].lr
        _parity(lambda: lr.globalMMATileGrid(mt, du), f"{name}.lr.globalMMATileGrid")
        _parity(lambda: lr.localMMATileGrid(mt, du, 4), f"{name}.lr.localMMATileGrid")
        _parity(lambda: lr.globalSubtileGrid(mt, du), f"{name}.lr.globalSubtileGrid")
        _parity(lambda: lr.subtileSizeBytes(), f"{name}.lr.subtileSizeBytes")

    @pytest.mark.parametrize("name", list(AB_PAIRS))
    @pytest.mark.parametrize("tc", ["A", "B"])
    def test_for_kernel(self, name, tc):
        gr = AB_PAIRS[name].gr
        py, cpp = _both(lambda: gr.for_kernel(KERNEL, tc))
        assert (py.subtileCount, py.subtileStride) == (
            cpp.subtileCount, cpp.subtileStride), f"{name}.for_kernel({tc})"
        # The returned object must still be a Python dataclass carrying the tag.
        assert type(cpp) is type(gr)
        assert cpp.tag is gr.tag

    @pytest.mark.parametrize("name", list(AB_PAIRS))
    def test_subtile_for_mma_tile(self, name):
        gr = AB_PAIRS[name].gr.for_kernel(KERNEL, "A")
        rows, cols = gr.globalMMATileGrid(KERNEL["MacroTileA"], KERNEL["_DepthUA"])
        for r in range(int(rows)):
            for c in range(int(cols)):
                _parity(lambda r=r, c=c: gr.subtileForMmaTile(r, c),
                        f"{name}.subtileForMmaTile({r},{c})")


# ---------------------------------------------------------------------------
# MX scale geometry parity
# ---------------------------------------------------------------------------
class TestMXScaleParity:
    @pytest.mark.parametrize("name", list(MX_PAIRS))
    @pytest.mark.parametrize("mt,du", MT_DU)
    def test_gr_and_lr_queries(self, name, mt, du):
        pair = MX_PAIRS[name]
        _parity(lambda: pair.gr.globalMMATileGrid(mt, du), f"{name}.gr.globalMMATileGrid")
        _parity(lambda: pair.lr.globalMMATileGrid(mt, du), f"{name}.lr.globalMMATileGrid")
        _parity(lambda: pair.lr.globalSubtileGrid(mt, du), f"{name}.lr.globalSubtileGrid")
        _parity(lambda: pair.lr.subtileSizeBytes(), f"{name}.lr.subtileSizeBytes")

    @pytest.mark.parametrize("name", list(MX_PAIRS))
    @pytest.mark.parametrize("tc", ["A", "B"])
    def test_gr_for_kernel(self, name, tc):
        gr = MX_PAIRS[name].gr
        py, cpp = _both(lambda: gr.for_kernel(KERNEL, tc))
        assert tuple(py.subtileShape) == tuple(cpp.subtileShape), f"{name}.for_kernel({tc})"


# ---------------------------------------------------------------------------
# C/D output geometry parity
# ---------------------------------------------------------------------------
class TestCDGeometryParity:
    @pytest.mark.parametrize("mt0,mt1", [(256, 128), (128, 128), (64, 256)])
    def test_cd_queries(self, mt0, mt1):
        cd = CD_F32
        wg = (2, 2)
        ss = (1.0, 1.0)
        _parity(lambda: cd.globalMMATileGrid(mt0, mt1), "CD.globalMMATileGrid")
        _parity(lambda: cd.localMMATileGrid(mt0, mt1, wg), "CD.localMMATileGrid")
        _parity(lambda: cd.globalSubtileGrid(mt0, mt1, ss), "CD.globalSubtileGrid")
        _parity(lambda: cd.localSubtileGrid(mt0, mt1, wg, ss), "CD.localSubtileGrid")


# ---------------------------------------------------------------------------
# Absolute-value pins for an AB_B16 gfx950 case (lock correctness, not just
# self-consistency between the two implementations).
# ---------------------------------------------------------------------------
class TestAbsoluteValues:
    def test_ab_b16_known_values(self):
        gr = AB_B16.gr
        assert gr.mmaTileShape == (16, 32)
        assert gr.mmaTileSize == 1024
        assert gr.globalMMATileGrid(256, 128) == (16, 4)
        assert gr.globalSubtileGrid(256, 128) == (16.0, 2.0)
        assert gr.bytesPerLoad(4) == 4096
        fk = gr.for_kernel(KERNEL, "A")
        assert (fk.subtileCount, fk.subtileStride) == (4, 4)
        sid, bshape, tiles = fk.subtileForMmaTile(5, 3)
        assert sid == (1, 1)
        assert bshape == (1, 2)
        assert tiles == [(1, 2), (1, 3), (5, 2), (5, 3),
                         (9, 2), (9, 3), (13, 2), (13, 3)]

    def test_ab_b16_values_match_under_cpp(self):
        with cpp_delegation():
            gr = AB_B16.gr
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


def test_default_path_is_python_only():
    """With the env flag unset, delegation must be disabled by default."""
    import os
    if os.environ.get("TENSILE_WRITER_CPP", "").strip().lower() not in (
            "", "0", "false", "no", "off"):
        pytest.skip("TENSILE_WRITER_CPP is set; default-off behavior not under test")
    assert sg._USE_CPP is False
    assert sg._CPP is None
