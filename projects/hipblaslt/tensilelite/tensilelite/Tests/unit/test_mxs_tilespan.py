#!/usr/bin/env python3
################################################################################
# Unit tests for the MXS TileSpan scale-select gating logic.
#
# TileSpan lays out each MX scale tile span so a single ds_load holds two scale
# blocks (lower half-wave = block 2g, upper half-wave = partner block 2g+1). The
# consuming gfx1250 WMMA then reads the partner block directly via
# matrix_{a,b}_scale:N, collapsing N scale ds_loads to N/2.
#
# The activation (ds_load halving) is gated by LocalReadMFMA.getMxsTileSpanInfo
# and is independent of the wave count. It MUST agree exactly with
# LraTileAssignment.tileSpan. LraTileAssignment.tileSpanWaveSplit is a second,
# narrower gate (tileSpan AND MIWaveGroup>1) that only selects between the two
# load layouts: MIWaveGroup==1 non-split (nIdx = wtid) vs MIWaveGroup>1 wave-split
# (nIdx = wtid % MI + hi offset). When getMxsTileSpanInfo and tileSpan disagree,
# the WMMA reads a scale block the load never placed in the half-wave and produces
# wrong results. These tests pin the gate contract and the LRA/LocalRead agreement.
#
# Pure Python logic -- no GPU hardware required.
#
# Usage:
#   pytest test_mxs_tilespan.py -v
################################################################################

import os
import sys
import types

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

try:
    from Tensile.Components.LocalRead import LocalReadMFMA
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    import Tensile.Component as _Comp
    _IMPORT_ERR = None
except Exception as exc:  # pragma: no cover - environment guard
    LocalReadMFMA = None
    KernelWriterAssembly = None
    _Comp = None
    _IMPORT_ERR = exc

# The `unit` marker is applied automatically by Tensile/Tests/conftest.py
# (pytest_collection_modifyitems tags every test by its top-level dir), so only
# the rocisa-availability skip guard is declared here.
pytestmark = pytest.mark.skipif(
    LocalReadMFMA is None,
    reason=f"cannot import LocalReadMFMA (rocisa unavailable?): {_IMPORT_ERR}",
)

# gfx1250 is wave32; the half-wave scale-select requires the tile-axis matrix
# instruction to land on the wave midpoint, i.e. MatrixInst == WavefrontSize/2.
WAVESIZE_32 = 32


def _make_kernel(
    tc="MXSA",
    tile01=0,
    vector_width=1,
    mi_wave_tile=(4, 4),
    matrix_inst_m=16,
    matrix_inst_n=16,
    mi_wave_group=(2, 2),
    wavefront_size=WAVESIZE_32,
):
    """Minimal kernel dict exercising getMxsTileSpanInfo.

    Defaults describe a TileSpan-eligible MXSA config (tile01=0):
      ratio = MIWaveTile[0]//VectorWidthMXSA = 4//1 = 4 (>=2, even),
      MatrixInstM = 16 = WavefrontSize/2, MIWaveGroup[0] = 2 (>1).
    """
    return {
        "VectorWidth%s" % tc: vector_width,
        "MIWaveTile": list(mi_wave_tile),
        "MatrixInstM": matrix_inst_m,
        "MatrixInstN": matrix_inst_n,
        "MIWaveGroup": list(mi_wave_group),
        "WavefrontSize": wavefront_size,
        "MXScaleFormat": "InMemorySwizzle",
        "ISA": (12, 5, 0),
    }, tc, tile01


# Default caps: the whole TileSpan feature only exists on gfx1250 WMMA_V3, so the
# eligible-path tests run with HasWMMA_V3 on.
_CAPS_V3 = {"HasWMMA_V3": True}


def _info(kernel_args, asmCaps=_CAPS_V3):
    kernel, tc, tile01 = kernel_args
    return LocalReadMFMA.getMxsTileSpanInfo(kernel, tc, tile01, asmCaps)


class TestGetMxsTileSpanInfoGate:
    """getMxsTileSpanInfo activation contract (positive + every negative branch)."""

    def test_eligible_returns_layout(self):
        """A fully eligible MXSA config returns the tile-span layout."""
        info = _info(_make_kernel())
        assert info == {"vectorWidth": 1, "numGroups": 2}

    def test_non_mxs_tensor_returns_none(self):
        """Non-MX-scale tensors (tc without 'MXS') never tile-span."""
        # A regular data tensor: getMxsTileSpanInfo must bail before touching
        # any MXS-only kernel keys.
        assert LocalReadMFMA.getMxsTileSpanInfo({}, "A", 0, _CAPS_V3) is None
        assert LocalReadMFMA.getMxsTileSpanInfo({}, "B", 1, _CAPS_V3) is None

    def test_no_wmma_v3_returns_none(self):
        """The whole feature is gfx1250 WMMA_V3 only: no WMMA_V3 -> None even when the
        geometry is otherwise eligible (e.g. gfx1151, where MatrixInst 16 == WavefrontSize/2
        would pass the geometry checks). Keeps the load side aligned with the consumer gate."""
        assert _info(_make_kernel(), asmCaps={"HasWMMA_V3": False}) is None
        assert _info(_make_kernel(), asmCaps={}) is None

    def test_wrong_scale_format_returns_none(self):
        """MXScaleFormat != InMemorySwizzle -> None (matches mxsUsesScaleSel)."""
        kernel, tc, tile01 = _make_kernel()
        kernel["MXScaleFormat"] = "Separate"
        assert LocalReadMFMA.getMxsTileSpanInfo(kernel, tc, tile01, _CAPS_V3) is None

    def test_non_gfx1250_returns_none(self):
        """TileSpan is only supported on gfx1250 (ISA (12,5,0)); other ISAs -> None
        even with an otherwise-eligible geometry and HasWMMA_V3 asserted."""
        kernel, tc, tile01 = _make_kernel()
        kernel["ISA"] = (11, 5, 1)  # gfx1151
        assert LocalReadMFMA.getMxsTileSpanInfo(kernel, tc, tile01, _CAPS_V3) is None
        kernel["ISA"] = (9, 5, 0)  # gfx950
        assert LocalReadMFMA.getMxsTileSpanInfo(kernel, tc, tile01, _CAPS_V3) is None

    def test_ratio_below_two_returns_none(self):
        """ratio (MIWaveTile//VectorWidth) < 2 -> no 2-block group to split."""
        # MIWaveTile[0]=1, VW=1 -> ratio 1.
        assert _info(_make_kernel(mi_wave_tile=(1, 4))) is None

    def test_odd_ratio_returns_none(self):
        """Odd ratio cannot be paired into 2-block groups."""
        # MIWaveTile[0]=3, VW=1 -> ratio 3 (>=2 but odd).
        assert _info(_make_kernel(mi_wave_tile=(3, 4))) is None

    def test_ratio_from_vector_width(self):
        """ratio uses VectorWidth: MIWaveTile=4, VW=2 -> ratio 2 -> 1 group."""
        info = _info(_make_kernel(vector_width=2, mi_wave_tile=(4, 4)))
        assert info == {"vectorWidth": 2, "numGroups": 1}

    def test_matrix_inst_not_wave_midpoint_returns_none(self):
        """MatrixInstM != WavefrontSize/2 breaks the half-wave scale-select."""
        # MatrixInstM=32 with wave32 -> midpoint is 16, so 32 != 16.
        assert _info(_make_kernel(matrix_inst_m=32)) is None

    def test_single_wave_group_still_tile_spans(self):
        """MIWaveGroup[tile]==1 still halves ds_loads: it uses the non-split layout
        (nIdx = wtid) rather than the wave-split layout, but the gate is wave-count
        independent so getMxsTileSpanInfo still returns a layout."""
        info = _info(_make_kernel(mi_wave_group=(1, 2)))
        assert info == {"vectorWidth": 1, "numGroups": 2}

    def test_num_groups_scales_with_tile(self):
        """numGroups = (MIWaveTile//VectorWidth)//2."""
        # ratio 8 -> 4 groups.
        info = _info(_make_kernel(mi_wave_tile=(8, 4)))
        assert info == {"vectorWidth": 1, "numGroups": 4}


class TestGetMxsTileSpanInfoAxisNeutral:
    """The gate is axis-neutral: tile01 selects axis, MatrixInst, and wave group."""

    def test_mxsb_axis_uses_matrix_inst_n(self):
        """tile01=1 (MXSB) reads MatrixInstN and MIWaveGroup[1]."""
        info = _info(_make_kernel(
            tc="MXSB", tile01=1,
            mi_wave_tile=(4, 6), vector_width=1,
            matrix_inst_n=16, matrix_inst_m=32,  # M is irrelevant on the N axis
            mi_wave_group=(1, 2),                 # group[0] irrelevant on the N axis
        ))
        assert info == {"vectorWidth": 1, "numGroups": 3}

    def test_mxsb_axis_gates_on_matrix_inst_n(self):
        """tile01=1 must reject when MatrixInstN != WavefrontSize/2."""
        assert _info(_make_kernel(
            tc="MXSB", tile01=1,
            mi_wave_tile=(4, 4), matrix_inst_n=32, mi_wave_group=(2, 2),
        )) is None


def _lra_tile_span(kernel, tc, tile01):
    """Replicate LraTileAssignment.tileSpan gating verbatim (the ds_load-halving gate,
    wave-count independent).

    Kept independent of getMxsTileSpanInfo so the test detects any future drift
    between the two gates.
    """
    if "MXS" not in tc:
        return False
    ratio = kernel["MIWaveTile"][tile01] // kernel["VectorWidth%s" % tc]
    matrix_inst_t = kernel["MatrixInstM"] if tile01 == 0 else kernel["MatrixInstN"]
    return (ratio >= 2
            and ratio % 2 == 0
            and matrix_inst_t == kernel["WavefrontSize"] // 2)


def _lra_wave_split(kernel, tc, tile01):
    """Replicate LraTileAssignment.tileSpanWaveSplit: tileSpan AND MIWaveGroup>1."""
    return _lra_tile_span(kernel, tc, tile01) and kernel["MIWaveGroup"][tile01] > 1


class TestGateMatchesLraTileSpan:
    """getMxsTileSpanInfo activation must equal LraTileAssignment.tileSpan (the ds_load
    halving gate), and tileSpanWaveSplit must be exactly tileSpan AND MIWaveGroup>1."""

    @pytest.mark.parametrize("tc,tile01", [("MXSA", 0), ("MXSB", 1)])
    @pytest.mark.parametrize("vector_width", [1, 2])
    @pytest.mark.parametrize("mi_wave_tile", [(1, 1), (2, 2), (3, 3), (4, 4), (8, 8)])
    @pytest.mark.parametrize("matrix_inst", [16, 32])
    @pytest.mark.parametrize("wave_group", [(1, 1), (2, 2), (1, 2), (2, 1)])
    def test_gates_agree(self, tc, tile01, vector_width, mi_wave_tile, matrix_inst, wave_group):
        kernel, _, _ = _make_kernel(
            tc=tc, tile01=tile01,
            vector_width=vector_width, mi_wave_tile=mi_wave_tile,
            matrix_inst_m=matrix_inst, matrix_inst_n=matrix_inst,
            mi_wave_group=wave_group,
        )
        info = LocalReadMFMA.getMxsTileSpanInfo(kernel, tc, tile01, _CAPS_V3)
        # The activation gate is wave-count independent and matches tileSpan.
        assert (info is not None) == _lra_tile_span(kernel, tc, tile01), (
            f"gate mismatch for tc={tc} tile01={tile01} vw={vector_width} "
            f"mt={mi_wave_tile} mi={matrix_inst} wg={wave_group}: "
            f"getMxsTileSpanInfo={info!r}, lraTileSpan={_lra_tile_span(kernel, tc, tile01)}"
        )
        # The wave-split layout is the strict subset that also needs >1 wave.
        assert _lra_wave_split(kernel, tc, tile01) == (
            (info is not None) and wave_group[tile01] > 1
        )


def _make_scalesel_kernel(vector_width=1, mi_wave_tile=(4, 4), mxscale_format="InMemorySwizzle"):
    return {
        "MXScaleFormat": mxscale_format,
        "VectorWidthMXSA": vector_width,
        "VectorWidthMXSB": vector_width,
        "MIWaveTile": list(mi_wave_tile),
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MIWaveGroup": [2, 2],
        "WavefrontSize": WAVESIZE_32,
        "ISA": (12, 5, 0),
    }


@pytest.fixture
def scalesel_writer(monkeypatch):
    """Build a minimal object that can drive KernelWriterAssembly.mxsTileSpanScaleSel.

    mxsTileSpanScaleSel pulls in self.mxsUsesScaleSel (asmCaps + MXScaleFormat gate)
    and Component.LocalRead.find(self).getMxsTileSpanInfo. We bind the real
    mxsUsesScaleSel onto a stub self and point the component lookup at the real
    LocalReadMFMA (which owns getMxsTileSpanInfo), so the production mapping code
    runs unchanged.
    """
    def _factory(has_wmma_v3=True):
        stub = types.SimpleNamespace()
        stub.states = types.SimpleNamespace(asmCaps={"HasWMMA_V3": has_wmma_v3})
        stub.mxsUsesScaleSel = types.MethodType(KernelWriterAssembly.mxsUsesScaleSel, stub)
        monkeypatch.setattr(_Comp.Component.LocalRead, "find",
                            classmethod(lambda cls, writer: LocalReadMFMA))
        return stub
    return _factory


def _scale_sel(stub, kernel, tc, tile01, idxAB):
    tP = {"tensorChar": tc, "tile01Idx": tile01}
    return KernelWriterAssembly.mxsTileSpanScaleSel(stub, kernel, tP, idxAB)


class TestScaleSelectEffect:
    """When active, the WMMA scale-select maps 2 blocks per register (matrix_*_scale:N)."""

    def test_partner_blocks_share_register_via_scale_select(self, scalesel_writer):
        """VW=1: logical blocks 0..3 -> regs 0,0,1,1 with scaleSel 0,1,0,1.

        This is the observable effect of TileSpan: each group's two scale blocks
        occupy one register; the partner block (scaleSel==1) is read from that
        same register via matrix_*_scale:1. Proves the loaded scale layout halved.
        """
        stub = scalesel_writer(has_wmma_v3=True)
        kernel = _make_scalesel_kernel(vector_width=1, mi_wave_tile=(4, 4))
        mapping = [_scale_sel(stub, kernel, "MXSA", 0, idx) for idx in range(4)]
        assert mapping == [(0, 0), (0, 1), (1, 0), (1, 1)]
        # 4 logical blocks collapse onto 2 distinct registers.
        assert len({m[0] for m in mapping}) == 2

    def test_vector_width_two_layout(self, scalesel_writer):
        """VW=2: within a 2*VW group, the first VW regs are scaleSel 0, next VW scaleSel 1
        and reuse the same registers; next group starts at the compacted offset."""
        stub = scalesel_writer(has_wmma_v3=True)
        kernel = _make_scalesel_kernel(vector_width=2, mi_wave_tile=(8, 8))
        mapping = [_scale_sel(stub, kernel, "MXSA", 0, idx) for idx in range(5)]
        assert mapping == [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]

    def test_mxsb_axis_scale_select(self, scalesel_writer):
        """The effect is axis-neutral: MXSB (tile01=1) collapses the same way."""
        stub = scalesel_writer(has_wmma_v3=True)
        kernel = _make_scalesel_kernel(vector_width=1, mi_wave_tile=(4, 4))
        mapping = [_scale_sel(stub, kernel, "MXSB", 1, idx) for idx in range(4)]
        assert mapping == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_single_wave_group_scale_select(self, scalesel_writer):
        """MIWaveGroup==1 (non-split layout) still collapses 2 blocks per register:
        the ds_load halving / scale-select is wave-count independent."""
        stub = scalesel_writer(has_wmma_v3=True)
        kernel = _make_scalesel_kernel(vector_width=1, mi_wave_tile=(4, 4))
        kernel["MIWaveGroup"] = [1, 1]
        mapping = [_scale_sel(stub, kernel, "MXSA", 0, idx) for idx in range(4)]
        assert mapping == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_no_wmma_v3_disables_scale_select(self, scalesel_writer):
        """Without HasWMMA_V3 the outer gate is off: identity mapping, no scale-select."""
        stub = scalesel_writer(has_wmma_v3=False)
        kernel = _make_scalesel_kernel(vector_width=1, mi_wave_tile=(4, 4))
        for idx in range(4):
            assert _scale_sel(stub, kernel, "MXSA", 0, idx) == (idx, 0)

    def test_wrong_scale_format_disables_scale_select(self, scalesel_writer):
        """MXScaleFormat != InMemorySwizzle: outer gate off, identity mapping."""
        stub = scalesel_writer(has_wmma_v3=True)
        kernel = _make_scalesel_kernel(vector_width=1, mi_wave_tile=(4, 4),
                                       mxscale_format="Separate")
        for idx in range(4):
            assert _scale_sel(stub, kernel, "MXSA", 0, idx) == (idx, 0)

    def test_ineligible_tile_span_disables_scale_select(self, scalesel_writer):
        """Outer gate on but getMxsTileSpanInfo ineligible (ratio<2): identity mapping."""
        stub = scalesel_writer(has_wmma_v3=True)
        kernel = _make_scalesel_kernel(vector_width=1, mi_wave_tile=(1, 1))
        for idx in range(4):
            assert _scale_sel(stub, kernel, "MXSA", 0, idx) == (idx, 0)

    def test_non_mxs_tensor_passthrough(self, scalesel_writer):
        """A non-MX-scale tensor is never scale-selected (identity mapping)."""
        stub = scalesel_writer(has_wmma_v3=True)
        kernel = _make_scalesel_kernel(vector_width=1, mi_wave_tile=(4, 4))
        for idx in range(4):
            assert _scale_sel(stub, kernel, "A", 0, idx) == (idx, 0)
