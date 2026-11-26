# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Gating policy of gfx1250 swizzled-TDM (SwizzleTensorA/B with TDM).

Combinations that generate a kernel are only safe once they have been validated:
LocalSplitU>1 and Sparse used to be accepted and silently produced wrong results.
This test pins the policy one axis at a time from a validated base. Every case
lists whether the Solution must accept it, and for a reject the expected reason.
Accepting a new combination therefore needs a validation run and an update of
ALLOWED below (with its evidence) -- the test fails until then.

Evidence for the allowed set: FFM validation of the swizzle-A / swizzle-B /
swizzle-A+B suites (Tests/common/gemm/gfx12/swizzle_*_tdm_gfx1250.yaml) plus
probe sweeps; real-HW runs of the suites where noted in the suite comments.
"""

import pytest

import swizzled_tdm_helpers as H
from test_halfplr_streamk_rejects import _gp_gfx1250  # noqa: F401

pytestmark = pytest.mark.unit


def _c(name, side="B", dtype="f8", wt=(2, 2), wg=(2, 2), pt=None, **fork):
    return dict(name=name, side=side, dtype=dtype, wt=wt, wg=wg, pt=pt or {}, fork=fork)


# Validated combinations: the Solution must accept them.
ALLOWED = [
    _c("base_B"), _c("base_A", side="A"), _c("base_AB", side="AB"),
    # data types (1-/2-byte, and F4 as packed bytes)
    *[_c("dt_%s_%s" % (dt, side), side=side, dtype=dt)
      for dt in ("f8", "bf16", "f16", "i8", "f4", "mxf8", "mxf4") for side in ("A", "B", "AB")],
    # vector width (swizzled side, VW>1 pads the LDS tile via TDM iterate-mode)
    # VW=1: an automatic (-1) LDS pad on the swizzled side resolves to none (see test_swizzled_tdm_auto_ldspad)
    _c("auto_ldspad_B", LdsPadB=-1, LdsBlockSizePerPadB=-1),
    _c("auto_ldspad_A", side="A", LdsPadA=-1, LdsBlockSizePerPadA=-1),
    _c("auto_ldspad_AB", side="AB", LdsPadA=-1, LdsBlockSizePerPadA=-1, LdsPadB=-1, LdsBlockSizePerPadB=-1),
    # 32x16x128 WMMA = two stacked 16x16x128: its 32-row operand (B under SourceSwap, else A) reads two nO tiles
    *[_c("mxf4_32x16_%s" % side, side=side, dtype="mxf4", miMN=(32, 16)) for side in ("A", "B", "AB")],
    _c("mxf4_32x16_vwb2", dtype="mxf4", VectorWidthB=2, miMN=(32, 16)),
    _c("mxf4_32x16_vwa2_nss", side="A", dtype="mxf4", VectorWidthA=2, miMN=(32, 16), SourceSwap=False),
    *[_c("mxf4_32x16_nss_%s" % side, side=side, dtype="mxf4", miMN=(32, 16), SourceSwap=False) for side in ("A", "B", "AB")],
    _c("vwb2", VectorWidthB=2), _c("vwa2_A", side="A", VectorWidthA=2),
    _c("vwb4", wt=(4, 4), VectorWidthB=4), _c("f4_vwb2", dtype="f4", VectorWidthB=2),
    # wave layouts, including a one-nO free dim over two wave components
    _c("wg41", wt=(1, 1), wg=(4, 1)), _c("wg14_A", side="A", wt=(1, 1), wg=(1, 4)),
    _c("wt44", wt=(4, 4)),
    # K loop / pipeline
    _c("du128", DepthU=128), _c("du512", DepthU=512), _c("pgr1", PrefetchGlobalRead=1),
    _c("plr0", PrefetchLocalRead=0), _c("hplr3", HalfPLR=3),
    _c("gsu4_mb", GlobalSplitU=4),
    _c("gsu4_mbsk", GlobalSplitU=4, GlobalSplitUAlgorithm="MultipleBufferSingleKernel"),
    _c("pgl2", PrefetchGL2=2, ClusterDim=[2, 2]),
    # persistent data-parallel / StreamK (swizzled K start = StreamKLocalStart * MI-scaled GlobalReadIncs).
    # Evidence: the persistent suites added with the StreamK support (swizzle_{a,b,ab}_tdm_persistent_gfx1250).
    _c("streamk_B", GlobalSplitU=0, StreamK=3), _c("streamk_A", side="A", GlobalSplitU=0, StreamK=3),
    _c("streamk_AB", side="AB", GlobalSplitU=0, StreamK=3),
    # LDSSegmentInterleave / B-rotate on swizzle-B
    _c("lsi_span", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    _c("lsi_vidx", wt=(2, 4), VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    _c("lsi_vidx_brot", wt=(2, 4), VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1, LDSSegmentBRotate=1),
    # LSI on swizzle-A (A must be coarse: VWA == MIWaveTile[0], so VWA > 1 with the iterate-mode pad) and A+B
    _c("lsi_A", side="A", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    # ... with F4 data (validated end to end on real HW: F4 and MXF4, VWA 2 and 4)
    *[_c("lsi_A_%s" % dt, side="A", dtype=dt, VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1) for dt in ("f4", "mxf4")],
    # swizzle-B / A+B with F4 data (real HW: F4, MXF4, B-rotate), and with a stacked 32x16x128 WMMA (real HW: SS1 B, SS0 A)
    _c("lsi_B_f4", dtype="f4", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    _c("lsi_B_mxf4", dtype="mxf4", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    _c("lsi_AB_mxf4", side="AB", dtype="mxf4", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    _c("lsi_B_mxf4_brot", dtype="mxf4", wt=(2, 4), VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1,
       LDSSegmentBRotate=1),
    _c("lsi_B_mxf4_32x16", dtype="mxf4", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1, miMN=(32, 16)),
    _c("lsi_B_mxf4_32x16_vidx", dtype="mxf4", wt=(2, 4), VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1,
       miMN=(32, 16)),
    _c("lsi_A_mxf4_32x16", side="A", dtype="mxf4", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1,
       miMN=(32, 16), SourceSwap=False),
    _c("lsi_AB_mxf4_32x16", side="AB", dtype="mxf4", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1,
       miMN=(32, 16)),
    _c("lsi_A_wt4", side="A", wt=(4, 2), VectorWidthA=4, VectorWidthB=2, LDSSegmentInterleave=1),
    _c("lsi_AB", side="AB", VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
    _c("lsi_A_bf16_du512", side="A", dtype="bf16", DepthU=512, VectorWidthA=2, LDSSegmentInterleave=1),
]

# Combinations that must be rejected, with the expected reason (substring of the printed reject).
REJECTED = [
    (_c("dt_f6", pt={"DataType": "F6", "MXBlockA": 32, "MXBlockB": 32}), "supports F4 and 1-/2-byte data types only"),
    (_c("dt_f32", dtype="f32"), "supports F4 and 1-/2-byte data types only"),
    (_c("lsu2_B", wg=(2, 1), WorkGroup=[16, 16, 2]), "does not support LocalSplitU>1"),
    (_c("lsu2_A", side="A", wg=(2, 1), WorkGroup=[16, 16, 2]), "does not support LocalSplitU>1"),
    (_c("sparse_B", pt={"Sparse": 1}), "does not support Sparse"),
    (_c("layout_NN_ldstr", pt={"TransposeA": False}, LDSTrInst=True), "supports TN only"),
    (_c("layout_TT_A_ldstr", side="A", pt={"TransposeB": True}, LDSTrInst=True), "supports TN only"),
    (_c("ldspad_B", LdsPadB=16, LdsBlockSizePerPadB=-1), "requires LdsPadB==0"),
    (_c("ldspad_A", side="A", LdsPadA=16, LdsBlockSizePerPadA=-1), "requires LdsPadA==0"),
    (_c("ldspad_B_blk_only", LdsPadB=-1, LdsBlockSizePerPadB=1024), "requires LdsPadB==0"),
    (_c("ldspad_A_pad_only", side="A", LdsPadA=16, LdsBlockSizePerPadA=0), "requires LdsPadA==0"),
    (_c("iterate_B", TDMIterateMode=2), "incompatible with TDMIterateMode"),
    (_c("iterate_A", side="A", TDMIterateMode=1), "incompatible with TDMIterateMode"),
    (_c("tdm0_B", TDMInst=0), "requires DirectToVgprB"),
    (_c("tdm_onlyA", TDMInst=1), "TDMA and TDMB must be enabled simultaneously"),
    (_c("tdmsplit", TDMSplit=True), "TDMSplit is currently disabled"),
    (_c("1lds", **{"1LDSBuffer": 1}), "1LDSBuffer must be 0 for TDM"),
    # the subtile swizzle is validated on gfx950 only (the client pads a swizzled K by UseSubtileImpl)
    *[(_c("subtile_%s" % side, side=side, UseSubtileImpl=True),
       "SwizzleTensor with UseSubtileImpl is validated on gfx950 only") for side in ("A", "B", "AB")],
    (_c("lsi_B_wg41", wt=(1, 2), wg=(4, 1), VectorWidthB=2, LDSSegmentInterleave=1),
     "swizzle-B LSI: only MIWaveGroup=[2,2]"),
    # LSI, B not spanning its LDS component, with the nO rows of a vIdx group not dividing evenly over
    # the components: a group would straddle two components.
    (_c("lsi_B_straddle", wt=(2, 6), VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1),
     "swizzle-B LSI: a vIdx group straddles an LDS component"),
    (_c("brot_A", side="A", wt=(2, 4), VectorWidthA=2, VectorWidthB=2, LDSSegmentInterleave=1,
        LDSSegmentBRotate=1), "SwizzleTensorA not supported"),
    (_c("lsi_A_vwa1", side="A", LDSSegmentInterleave=1), "A: VWA must be WaveTileA"),
]


def _derive(case, gfx1250_iim, assembler, capsys):
    fork = dict(case["fork"])
    workgroup = tuple(fork.pop("WorkGroup", (16, 16, 1)))   # WorkGroup[2] is LocalSplitU
    params = H.swizzled_params(gfx1250_iim, case["side"], case["dtype"], wt=case["wt"], wg=case["wg"],
                               pt=case["pt"], workgroup=workgroup, **fork)
    capsys.readouterr()
    sol = H.derive(params, assembler, gfx1250_iim)
    return sol, capsys.readouterr().out


@pytest.mark.parametrize("case", ALLOWED, ids=[c["name"] for c in ALLOWED])
def test_swizzled_tdm_allowed(_gp_gfx1250, gfx1250_iim, assembler, capsys, case):
    sol, out = _derive(case, gfx1250_iim, assembler, capsys)
    assert sol["Valid"], "validated swizzled-TDM combination rejected:\n" + out[-2000:]


@pytest.mark.parametrize("case,reason", REJECTED, ids=[c["name"] for c, _ in REJECTED])
def test_swizzled_tdm_rejected(_gp_gfx1250, gfx1250_iim, assembler, capsys, case, reason):
    sol, out = _derive(case, gfx1250_iim, assembler, capsys)
    assert not sol["Valid"], ("swizzled-TDM combination '%s' is accepted but has not been validated; "
                              "validate it and move it to ALLOWED" % case["name"])
    assert reason in out, "rejected for another reason (expected %r):\n%s" % (reason, out[-2000:])


@pytest.mark.parametrize("side", ["A", "B"])
@pytest.mark.parametrize("pad,block", [(-1, -1), (-1, 0), (0, -1), (0, 0)])
def test_swizzled_tdm_auto_ldspad(_gp_gfx1250, gfx1250_iim, assembler, capsys, side, pad, block):
    """VW=1: automatic (-1) LdsPad / LdsBlockSizePerPad on the swizzled side resolve to 0."""
    case = _c("auto", side=side, **{"LdsPad" + side: pad, "LdsBlockSizePerPad" + side: block})
    sol, out = _derive(case, gfx1250_iim, assembler, capsys)
    assert sol["Valid"], "automatic LDS pad on a VW=1 swizzled tensor must be accepted:\n" + out[-2000:]
    assert sol["LdsPad" + side] == 0 and sol["LdsBlockSizePerPad" + side] == 0


def test_swizzled_tdm_auto_ldspad_leaves_the_other_tensor_automatic(_gp_gfx1250, gfx1250_iim, assembler, capsys):
    """Only the swizzled side is forced to 0; the plain tensor keeps its own automatic resolution."""
    case = _c("auto_other", side="B", LdsPadA=-1, LdsPadB=-1, LdsBlockSizePerPadA=-1, LdsBlockSizePerPadB=-1)
    sol, out = _derive(case, gfx1250_iim, assembler, capsys)
    assert sol["Valid"], out[-2000:]
    assert sol["LdsPadB"] == 0 and sol["LdsBlockSizePerPadB"] == 0
    assert sol["LdsPadA"] != -1 and sol["LdsBlockSizePerPadA"] != -1, "plain tensor A must still resolve its auto pad"
