# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU-only codegen sweep for gfx1250 swizzled-TDM (SwizzleTensorA/B with TDM).

Each case derives a real Solution and generates its kernel source (no assembly,
no GPU). Codegen itself asserts the swizzled-TDM invariants (local-read address
register in range, non-empty TDM tiles, whole 16-byte ds_load K blocks); this
test adds structural checks on the emitted source:

* every LocalReadAddr{A,B}+i the kernel uses is a declared address register
  (an LSI read window mapped wrongly read through a VGPR outside the range);
* no TDM descriptor moves an empty tile (a one-nO macro-tile split over two
  wave components left the operand unloaded);
* the swizzled tail K-step advances every local-read address register (only +0
  was advanced, so LSI component 1 re-read the first tail K block);
* with LDSSegmentInterleave and B not spanning its LDS component, the vIdx group
  of each swizzled B read selects the component, and component h is read through
  LocalReadAddrB+h (the reader once kept every vIdx group in component 0).

Cases whose configuration the Solution rejects are skipped (the gating policy
is covered by test_swizzled_tdm_gating.py).
"""

import itertools
import re

import pytest

import swizzled_tdm_helpers as H
from test_halfplr_streamk_rejects import _gp_gfx1250  # noqa: F401

pytestmark = pytest.mark.unit


def _case(name, side, dtype, wt=(2, 2), wg=(2, 2), **fork):
    return pytest.param(dict(side=side, dtype=dtype, wt=wt, wg=wg, fork=fork), id=name)


# Configurations that produced wrong results before their fix.
REGRESSIONS = [
    # One nO row per macro-tile over two wave components: tile1 was 0 (operand never loaded).
    _case("n16_B_mt64x16_wg41", "B", "f8", wt=(1, 1), wg=(4, 1)),
    _case("n16_B_mt128x16_wg41", "B", "f8", wt=(2, 1), wg=(4, 1)),
    _case("m16_A_mt16x64_wg14", "A", "f8", wt=(1, 1), wg=(1, 4)),
    _case("n16_AB_wg41", "AB", "f8", wt=(1, 1), wg=(4, 1)),
    # LSI component stride >= 128K: reads must go through the window registers.
    _case("lsi_window_bf16_du512", "B", "bf16", DepthU=512, VectorWidthA=2,
          LDSSegmentInterleave=1, TDMIterateMode=1),
    _case("lsi_window_bf16_du512_brot", "B", "bf16", DepthU=512, VectorWidthA=2,
          LDSSegmentInterleave=1, LDSSegmentBRotate=1, TDMIterateMode=1),
    # LSI with VWB < MIWaveTile[1]: vIdx group h reads component h; tail must step every register.
    _case("lsi_vidx_f8", "B", "f8", wt=(2, 4), VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    _case("lsi_vidx_f8_brot", "B", "f8", wt=(2, 4), VectorWidthA=2, VectorWidthB=2,
          LDSSegmentInterleave=1, LDSSegmentBRotate=1),
    _case("lsi_vidx_mxf8_brot", "B", "mxf8", wt=(2, 4), VectorWidthA=2, VectorWidthB=2,
          LDSSegmentInterleave=1, LDSSegmentBRotate=1),
    _case("lsi_vidx_f8_wt8", "B", "f8", wt=(2, 8), VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    # LSI on swizzle-A (coarse A) and A+B.
    _case("lsi_A_f8", "A", "f8", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    _case("lsi_A_f8_wt4", "A", "f8", wt=(4, 2), VectorWidthA=4, VectorWidthB=2, LDSSegmentInterleave=1),
    _case("lsi_AB_f8", "AB", "f8", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    _case("lsi_A_bf16_du512", "A", "bf16", DepthU=512, VectorWidthA=2, LDSSegmentInterleave=1),
    # LSI on F4 / MXF4 swizzled operands (validated on real HW).
    _case("lsi_A_f4", "A", "f4", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    _case("lsi_A_mxf4_wt4", "A", "mxf4", wt=(4, 2), VectorWidthA=4, VectorWidthB=2, LDSSegmentInterleave=1),
    _case("lsi_B_f4", "B", "f4", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    _case("lsi_B_mxf4_vidx", "B", "mxf4", wt=(2, 4), VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    _case("lsi_B_mxf4_vidx_brot", "B", "mxf4", wt=(2, 4), VectorWidthA=2, VectorWidthB=2,
          LDSSegmentInterleave=1, LDSSegmentBRotate=1),
    _case("lsi_AB_mxf4", "AB", "mxf4", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    # ... and on the stacked 32-row operand of the 32x16x128 WMMA (B under SourceSwap, A without).
    _case("lsi_B_mxf4_32x16", "B", "mxf4", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1, miMN=(32, 16)),
    _case("lsi_B_mxf4_32x16_vidx", "B", "mxf4", wt=(2, 4), VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1,
          miMN=(32, 16)),
    _case("lsi_A_mxf4_32x16_nss", "A", "mxf4", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1,
          miMN=(32, 16), SourceSwap=False),
    _case("lsi_AB_mxf4_32x16", "AB", "mxf4", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1, miMN=(32, 16)),
    # F4 is packed bytes: innerK / pad derive from a 0.5-byte element.
    # One wave: TDM is not wave-separated, so the single-descriptor start address / descriptor init run.
    _case("one_wave_B", "B", "f8", wg=(1, 1)),
    _case("one_wave_A", "A", "f8", wg=(1, 1)),
    _case("one_wave_AB", "AB", "f8", wg=(1, 1)),
    _case("one_wave_B_gsu4", "B", "f8", wg=(1, 1), GlobalSplitU=4),
    _case("one_wave_A_gsu4", "A", "f8", wg=(1, 1), GlobalSplitU=4),
    # ... with the VW>1 pad deposited by iterate-mode: its init takes the unclamped row count.
    _case("one_wave_B_bf16_vwb2_iterate", "B", "bf16", wg=(1, 1), VectorWidthA=2, VectorWidthB=2, DepthU=512,
          TDMIterateMode=1),  # A is the plain side: its own pad block needs iterate-mode too
    _case("one_wave_A_bf16_vwa2_iterate", "A", "bf16", wg=(1, 1), VectorWidthA=2, VectorWidthB=2, DepthU=512,
          TDMIterateMode=2),
    _case("f4_B", "B", "f4"),
    _case("f4_A_vwa2_pad", "A", "f4", VectorWidthA=2),
    # nO tile <= 1024 B: the VW>1 pad comes from the descriptor's pad_interval (no iterate-mode)
    _case("bf16_B_vwb2_small_tile", "B", "bf16", VectorWidthB=2, DepthU=32),
    _case("bf16_A_vwa2_small_tile", "A", "bf16", VectorWidthA=2, DepthU=32),
    _case("f4_B_vwb2_small_tile", "B", "f4", VectorWidthB=2, DepthU=128),
    _case("f4_A_vwa2_small_tile", "A", "f4", VectorWidthA=2, DepthU=128),
    _case("f4_B_vwb4_pad", "B", "f4", wt=(4, 4), VectorWidthB=4),
    _case("mxf4_AB", "AB", "mxf4"),
    # 32x16x128 WMMA: the 32-row operand is B under SourceSwap, A without
    _case("mxf4_B_32x16", "B", "mxf4", miMN=(32, 16)),
    _case("mxf4_AB_32x16", "AB", "mxf4", miMN=(32, 16)),
    _case("mxf4_A_32x16_nss", "A", "mxf4", miMN=(32, 16), SourceSwap=False),
    _case("mxf4_AB_32x16_nss", "AB", "mxf4", miMN=(32, 16), SourceSwap=False),
    _case("mxf4_B_32x16_vwb2", "B", "mxf4", VectorWidthB=2, miMN=(32, 16)),
    _case("mxf4_A_32x16_vwa2_nss", "A", "mxf4", VectorWidthA=2, miMN=(32, 16), SourceSwap=False),
    _case("mxf4_B_32x16_gsu4", "B", "mxf4", miMN=(32, 16), GlobalSplitU=4),
]

# Broad sweep: side x data type x wave layout x vector width.
_LAYOUTS = {"wg22": dict(wt=(2, 2), wg=(2, 2)), "wg41": dict(wt=(1, 1), wg=(4, 1)),
            "wg14": dict(wt=(1, 1), wg=(1, 4)), "wt44": dict(wt=(4, 4), wg=(2, 2))}
SWEEP = []
for side, dtype, lay, vw in itertools.product(("A", "B", "AB"), ("f8", "bf16", "f4", "mxf8", "i8"),
                                              _LAYOUTS, (1, 2)):
    if vw > _LAYOUTS[lay]["wt"][0] or vw > _LAYOUTS[lay]["wt"][1]:
        continue
    SWEEP.append(_case("%s_%s_%s_vw%d" % (side, dtype, lay, vw), side, dtype,
                       VectorWidthA=vw, VectorWidthB=vw, **_LAYOUTS[lay]))
# Pipeline / K-loop variants on the default swizzle-B layout.
for name, fork in (("du128", dict(DepthU=128)), ("du512", dict(DepthU=512)), ("pgr1", dict(PrefetchGlobalRead=1)),
                   ("plr0", dict(PrefetchLocalRead=0)), ("hplr3", dict(HalfPLR=3)),
                   ("gsu_mbsk", dict(GlobalSplitU=4, GlobalSplitUAlgorithm="MultipleBufferSingleKernel")),
                   ("pgl2", dict(PrefetchGL2=2, ClusterDim=[2, 2]))):
    SWEEP.append(_case("B_f8_" + name, "B", "f8", **fork))
    SWEEP.append(_case("A_f8_" + name, "A", "f8", **fork))
# Persistent StreamK (the K start of a workgroup is StreamKLocalStart * the MI-scaled K increment).
for side in ("A", "B", "AB"):
    SWEEP.append(_case("%s_f8_streamk" % side, side, "f8", GlobalSplitU=0, StreamK=3))


_LRA = re.compile(r"vgprLocalReadAddr([AB])\+(\d+)")
_TILE = re.compile(r"set tile([01]) to (\d+)")
_TAIL_STEP = re.compile(r"lr([AB])\+(\d+) \+= \d+ \(swizzledTDM tail K-step")
_SWZ_READ = re.compile(r"ds_load_b128 [^\n]*vgprLocalReadAddrB\+(\d+)\][^\n]*swizzledTDM L->Reg nt=(\d+) r=\d+")


def _generate(cfg, gfx1250_iim, assembler):
    params = H.swizzled_params(gfx1250_iim, cfg["side"], cfg["dtype"], wt=cfg["wt"], wg=cfg["wg"], **cfg["fork"])
    sol = H.derive(params, assembler, gfx1250_iim)
    if not sol["Valid"]:
        pytest.skip("configuration rejected by the Solution")
    err, src, kwa = H.codegen(sol, assembler)
    assert err == 0, "codegen failed (err=%s)" % err
    return sol, src, kwa


def _numLra(kwa, tc):
    return (kwa.states.a if tc == "A" else kwa.states.b).numVgprLocalReadAddr


def check_source(sol, src, kwa):
    """Structural invariants of a swizzled-TDM kernel source."""
    # 1. Only declared local-read address registers are referenced.
    for tc, idx in {(m.group(1), int(m.group(2))) for m in _LRA.finditer(src)}:
        assert idx < _numLra(kwa, tc), \
            "LocalReadAddr%s+%d used, but only %d address registers are declared" % (tc, idx, _numLra(kwa, tc))
    # 2. No TDM descriptor moves an empty tile.
    empty = [m.group(0) for m in _TILE.finditer(src) if int(m.group(2)) == 0]
    assert not empty, "TDM descriptor with an empty tile: %s" % empty[:3]
    # 3. The swizzled tail K-step advances every local-read address register of the operand.
    steps = {}
    for m in _TAIL_STEP.finditer(src):
        steps.setdefault(m.group(1), set()).add(int(m.group(2)))
    for tc in "AB":
        if not sol["ProblemType"]["SwizzleTensor" + tc]:
            continue
        assert tc in steps, "swizzled %s has no tail K-step" % tc
        assert steps[tc] == set(range(_numLra(kwa, tc))), \
            "swizzled %s tail K-step advances LocalReadAddr%s+%s, expected all %d" \
            % (tc, tc, sorted(steps[tc]), _numLra(kwa, tc))
    # 4. LSI, B not spanning its component: vIdx group -> component -> LocalReadAddrB+component.
    if sol["LDSSegmentInterleave"] == 1 and sol["ProblemType"]["SwizzleTensorB"]:
        vw, wgN = sol["VectorWidthB"], sol["MIWaveGroup"][1]
        numComp = sol["NumWaves"] // 2
        compNO = sol["MacroTile1"] // sol["MatrixInstN"] // numComp
        if sol["MIWaveTile"][1] // vw > 1 and compNO % (vw * wgN) == 0:
            reads = [(int(m.group(2)), int(m.group(1))) for m in _SWZ_READ.finditer(src)]
            assert reads, "no swizzled B reads found"
            for nt, reg in reads:
                comp = (nt // vw) * vw * wgN // compNO
                assert reg == comp, "swizzled B nt=%d (component %d) reads through LocalReadAddrB+%d" % (nt, comp, reg)


@pytest.mark.parametrize("cfg", REGRESSIONS)
def test_swizzled_tdm_regressions(_gp_gfx1250, gfx1250_iim, assembler, cfg):
    sol, src, kwa = _generate(cfg, gfx1250_iim, assembler)
    check_source(sol, src, kwa)


@pytest.mark.parametrize("cfg", SWEEP)
def test_swizzled_tdm_sweep(_gp_gfx1250, gfx1250_iim, assembler, cfg):
    sol, src, kwa = _generate(cfg, gfx1250_iim, assembler)
    check_source(sol, src, kwa)


@pytest.mark.parametrize("side,fork", [("B", {}), ("A", dict(SourceSwap=False)), ("AB", dict(SourceSwap=False))])
def test_stacked_wmma_descriptor_counts_16_row_blocks(_gp_gfx1250, gfx1250_iim, assembler, capsys, side, fork):
    """A 32x16x128 WMMA reads two 16-row nO tiles, but the host buffer and the TDM descriptor count 16-row
    blocks: dim1 (rows left) in 32-row units loads only half of the macro-tile (the rest stays zero)."""
    params = H.swizzled_params(gfx1250_iim, side, "mxf4", miMN=(32, 16), **fork)
    sol = H.derive(params, assembler, gfx1250_iim)
    assert sol["Valid"], capsys.readouterr().out[-1500:]
    err, src, _ = H.codegen(sol, assembler)
    assert err == 0
    assert "swizzled dim1: SizeI + MI(16)-1" in src or "swizzled dim1: SizeJ + MI(16)-1" in src
    assert "MI(32)" not in src
