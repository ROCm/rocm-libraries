import pytest
from Tensile.SolutionStructs.segment_interleave import evaluate, aligned_budget_ok, SEG

pytestmark = pytest.mark.unit

class _FakeDataType:
    # Mirrors the real DataType API the oracle uses, without importing rocisa.
    def __init__(self, bf16=True, half=False, f8=False, nbytes=2.0):
        self._bf16 = bf16
        self._half = half
        self._f8 = f8
        self._nbytes = nbytes
    def isBFloat16(self):
        return self._bf16
    def isHalf(self):
        return self._half
    def is8bitFloat(self):
        return self._f8
    def numBytes(self):
        return self._nbytes

def _vw8_state(**ovr):
    # TLUA/TLUB are TOP-LEVEL state keys (not under ProblemType); DataType is an object.
    s = dict(NumWaves=4, WavefrontSize=32, MacroTile0=256, MacroTile1=256, DepthU=128,
             ISA=(12, 5, 0),
             LdsOffsetA=0, LdsBlockSizePerPadA=2048, LdsBlockSizePerPadB=2048,
             LdsPadA=8, LdsPadB=8, VectorWidthA=8, VectorWidthB=8,
             MatrixInstM=16, MatrixInstN=16, MIWaveGroup=[2, 2], TDMSplit=0, enableTDMA=1, enableTDMB=1,
             UnrollMajorLDSA=1, UnrollMajorLDSB=1,
             ProblemType=dict(Sparse=0, DataType=_FakeDataType(), MXBlockA=0, MXBlockB=0))
    s["ProblemType"] = {**s["ProblemType"], **ovr.pop("ProblemType", {})}
    s.update(ovr); return s

def test_vw8_applies_with_handedit_values():
    # Footprint-packed tight: writeStrideBytes = fA+fB (post-pad), readWaveStride = that//bpe.
    r = evaluate(_vw8_state())
    assert r["applicable"] is True
    assert r["offsets"] == {"ldsBaseB": 33024, "writeStrideBytes": 66048,
                            "readWaveStride": 33024, "footprintPacked": True}

def test_vw4_skips_fine_vw():
    r = evaluate(_vw8_state(VectorWidthA=4))
    assert r["applicable"] is False and "fine VW" in r["reason"]

def test_non_gfx1250_skips():
    # SEG=64KiB layout is gfx1250-specific; other ISAs must not apply the interleave.
    for isa in [(9, 4, 2), (9, 5, 0), (12, 0, 0), (11, 0, 0)]:
        r = evaluate(_vw8_state(ISA=isa))
        assert r["applicable"] is False and "gfx1250" in r["reason"]

def test_vwb_fine_aligned_applies():
    # Fine VWB is OK when each VW-group (vIdx) stays within one component, i.e. the
    # component column span is a whole multiple of one vIdx's column advance. For
    # MT256x256 (compCols=128), VWB in {1,2,4} all divide cleanly -> apply. LocalRead
    # (calcGfx1250LdsOffset) adds the per-vIdx component jump. GPU-validated 8/4/2/1.
    for vwb in (4, 2, 1):
        r = evaluate(_vw8_state(VectorWidthB=vwb))
        assert r["applicable"] is True, f"VWB={vwb} should apply"

def test_vwb_fine_unaligned_skips():
    # When a single vIdx would straddle a component (component span not a multiple of the
    # vIdx column advance), the per-vIdx fix can't place it -> must skip. MT*x224 VWB1:
    # compCols=112, vIdxCols=32, 112 % 32 != 0. (GPU-confirmed MT128x224 was wrong.)
    r = evaluate(_vw8_state(MacroTile1=224, VectorWidthB=1))
    assert r["applicable"] is False and "fine VW" in r["reason"]

def test_vwa_fine_always_skips():
    # A is the de-conflicted operand; its own read must never cross a component -> coarseA
    # stays mandatory regardless of B.
    r = evaluate(_vw8_state(VectorWidthA=4))
    assert r["applicable"] is False and "fine VW" in r["reason"]

def test_small_mt_skips_without_pgr2():
    # Small MT is the aligned candidate, but it requires PGR2 double-buffer; with no
    # PrefetchGlobalRead set it skips to baseline (buffering gate).
    r = evaluate(_vw8_state(MacroTile0=128, MacroTile1=128))
    assert r["applicable"] is False and "PGR" in r["reason"]

def test_non_square_small_skips_without_pgr2():
    # MT0=256, MT1=128: coarse VW ok (16*8=128 >= 256//2), but fA+fB < SEG -> small MT;
    # no PGR2 -> buffering gate skip.
    r = evaluate(_vw8_state(MacroTile1=128))
    assert r["applicable"] is False and "PGR" in r["reason"]

def _aligned_tiles_disjoint(r, fA, fB):
    # Aligned footprint packing: A_i = i*pre, B_i = ldsBaseB + i*pre (base=0, ldsBaseB=fA).
    o = r["offsets"]; pre = o["writeStrideBytes"]; assert o["ldsBaseB"] == fA
    tiles = [(0, fA), (fA, fA + fB), (pre, pre + fA), (fA + pre, fA + pre + fB)]  # A0,B0,A1,B1
    tiles.sort()
    return all(tiles[i][1] <= tiles[i + 1][0] for i in range(3))

def test_aligned_applies_small_mt():
    # MT128 bf16 symmetric: fA=fB=16512, sum<SEG -> aligned. Footprint-packed, segment stride.
    r = evaluate(_vw8_state(MacroTile0=128, MacroTile1=128, PrefetchGlobalRead=2, LDSSegmentInterleave=1))
    assert r["applicable"] is True and r["aligned"] is True
    assert r["offsets"] == {"ldsBaseB": 16512, "writeStrideBytes": SEG,
                            "readWaveStride": SEG // 2, "footprintPacked": True}
    assert r["blockSpan"] == SEG + 16512 + 16512    # base(0) + pre(SEG) + fA + fB
    assert _aligned_tiles_disjoint(r, 16512, 16512)
    assert "ALIGNED" in r["segmentMap"]

def test_aligned_fine_vwb_applies():
    # Fine-VWB is supported on the aligned branch too (per-vIdx component jump + enough
    # LocalReadAddr +64K registers, see KernelWriter numVgprLocalReadAddr). MT128x128 VWB2.
    r = evaluate(_vw8_state(MacroTile0=128, MacroTile1=128, PrefetchGlobalRead=2,
                            LDSSegmentInterleave=1, VectorWidthB=2))
    assert r["applicable"] is True and r["aligned"] is True

def test_aligned_unequal_pad_no_overlap():
    # Asymmetric pads (padA=16 != padB=8): footprint packing places each tile at its own post-pad
    # size, so ldsBaseB=base+fA (no compensation) and the segment stride keeps all 4 tiles disjoint.
    # fA=16384+(16384//2048)*(16*2)=16640 ; fB=16384+(16384//2048)*(8*2)=16512
    r = evaluate(_vw8_state(MacroTile0=128, MacroTile1=128, PrefetchGlobalRead=2,
                            LdsPadA=16, LdsPadB=8, LDSSegmentInterleave=1))
    assert r["applicable"] is True and r["aligned"] is True and r["offsets"]["footprintPacked"]
    assert r["offsets"]["ldsBaseB"] == 16640         # base + fA, no compensation
    assert _aligned_tiles_disjoint(r, 16640, 16512)

def test_aligned_skips_pgr_not_2():
    r = evaluate(_vw8_state(MacroTile0=128, MacroTile1=128, PrefetchGlobalRead=1))
    assert r["applicable"] is False and "PGR" in r["reason"]

def test_skips_1ldsbuffer():
    # 1LDSBuffer breaks the double-buffered layout the offsets assume. The oracle runs before
    # Solution.py resolves 1LDSBuffer==-1, so both 1 and unresolved -1 must be rejected.
    for v in (1, -1):
        r = evaluate(_vw8_state(**{"1LDSBuffer": v}))
        assert r["applicable"] is False and "1LDSBuffer" in r["reason"]

def test_subtile_skips():
    # Subtile is a separate codegen body; interleave never applies (tight or aligned).
    assert evaluate(_vw8_state(UseSubtileImpl=1))["applicable"] is False
    r = evaluate(_vw8_state(MacroTile0=128, MacroTile1=128, PrefetchGlobalRead=2, UseSubtileImpl=1))
    assert r["applicable"] is False and "subtile" in r["reason"]

def test_aligned_budget_ok_fits():
    # blockSpan 98560 -> max -> roundup 131072; 131072*2=262144 <= 327680 (gfx1250 MaxLDS).
    ok, blk = aligned_budget_ok(98560, 2, 0, 327680)
    assert ok is True and blk == 131072

def test_aligned_budget_too_big():
    # MaxLDS=163840 (gfx950): 131072*2=262144 > 163840 -> reject (would force StoreSwapAddr).
    ok, blk = aligned_budget_ok(98560, 2, 0, 163840)
    assert ok is False and blk is None

def test_aligned_budget_numldsblk_not_2():
    ok, blk = aligned_budget_ok(98560, 3, 0, 327680)
    assert ok is False and blk is None

def test_parameter_off_skips():
    # LDSSegmentInterleave=0 forces baseline (per-solution tuning knob).
    r = evaluate(_vw8_state(LDSSegmentInterleave=0))
    assert r["applicable"] is False and "parameter off" in r["reason"]
    assert evaluate(_vw8_state(LDSSegmentInterleave=1))["applicable"] is True

def _tight_tiles_disjoint(r):
    # Footprint packing: A_i = i*stride, B_i = ldsBaseB + i*stride; A spans fA=ldsBaseB, B spans
    # fB=stride-fA. Assert [A0][B0][A1][B1] are pairwise non-overlapping (base=0).
    o = r["offsets"]; s = o["writeStrideBytes"]; fA = o["ldsBaseB"]; fB = s - fA
    tiles = [(0, fA), (fA, fA + fB), (s, s + fA), (s + fA, s + fB + fA)]  # A0,B0,A1,B1
    tiles.sort()
    return all(tiles[i][1] <= tiles[i + 1][0] for i in range(len(tiles) - 1))

def test_tight_asymmetric_pad_applies_no_overlap():
    # Footprint packing places each tile at its own post-pad size, so asymmetric A/B pads pack
    # exactly (no B0/A1 overlap) instead of skipping. Every mismatch must apply AND be disjoint.
    for ovr in (dict(LdsPadB=16), dict(LdsPadA=16), dict(LdsPadA=4, LdsPadB=8),
                dict(LdsBlockSizePerPadB=1024), dict(LdsBlockSizePerPadA=4096)):
        r = evaluate(_vw8_state(**ovr))
        assert r["applicable"] is True and r["offsets"]["footprintPacked"], ovr
        assert _tight_tiles_disjoint(r), ovr

def test_tight_symmetric_pad_disjoint():
    assert evaluate(_vw8_state(LdsPadA=16, LdsPadB=16))["applicable"] is True
    assert _tight_tiles_disjoint(evaluate(_vw8_state()))

def test_auto_enables_tight():
    # -1 (auto, the default) takes the free tight branch: big MT, no LDS growth.
    r = evaluate(_vw8_state(LDSSegmentInterleave=-1))
    assert r["applicable"] is True and r["aligned"] is False

def test_auto_skips_aligned():
    # -1 (auto) declines the LDS-growing aligned branch even when it is applicable; needs 1.
    r = evaluate(_vw8_state(MacroTile0=128, MacroTile1=128, PrefetchGlobalRead=2, LDSSegmentInterleave=-1))
    assert r["applicable"] is False and "auto" in r["reason"]
    r1 = evaluate(_vw8_state(MacroTile0=128, MacroTile1=128, PrefetchGlobalRead=2, LDSSegmentInterleave=1))
    assert r1["applicable"] is True and r1["aligned"] is True

def test_miwavegroup_not_2x2_skips():
    # footprint packing assumes 2 waves per MFMA dim; NumWaves=4 with MIWG [4,1]/[1,4] passes
    # Solution.py's prod>1 / pow2 gates but would lose/OOB the component jump. Must skip.
    for miwg in ([4, 1], [1, 4]):
        r = evaluate(_vw8_state(MIWaveGroup=miwg))
        assert r["applicable"] is False and "MIWaveGroup" in r["reason"], miwg

def test_tdmsplit_skips():
    assert evaluate(_vw8_state(TDMSplit=1))["applicable"] is False

def test_tile_major_skips():
    r = evaluate(_vw8_state(UnrollMajorLDSA=0))  # not unrollMajor -> deferred
    assert r["applicable"] is False and ("unrollMajor" in r["reason"] or "tile-major" in r["reason"])

def test_fp16_applies_same_as_bf16():
    # fp16 has the same bpe (2) and the same write/read paths -> identical offsets.
    r = evaluate(_vw8_state(ProblemType={"DataType": _FakeDataType(bf16=False, half=True)}))
    assert r["applicable"] is True
    assert r["offsets"] == {"ldsBaseB": 33024, "writeStrideBytes": 66048,
                            "readWaveStride": 33024, "footprintPacked": True}

def test_fp8_tight_applies():
    # fp8 (bpe=1) is now supported. DepthU=256 -> fA+fB >= SEG -> tight branch.
    r = evaluate(_vw8_state(DepthU=256,
                            ProblemType={"DataType": _FakeDataType(bf16=False, f8=True, nbytes=1)}))
    fA = 32768 + (32768 // 2048) * 8   # data + pad (LdsPadA=8, block=2048, bpe=1)
    assert r["applicable"] is True and r["reason"] == "tight"
    assert r["offsets"]["writeStrideBytes"] == 2 * fA and r["offsets"]["readWaveStride"] == 2 * fA
    # No MX scales -> no relocation keys emitted.
    assert "ldsBaseMXSA" not in r["offsets"] and "ldsBaseMXSB" not in r["offsets"]

def test_mxf8_tight_relocates_scales():
    # mxf8: fp8 + MXBlock scales. Tight A/B interleave unchanged; scale block relocated to
    # base + 2*(fA+fB), sized from LdsNumElementsAlignedMXS{A,B}.
    r = evaluate(_vw8_state(DepthU=256,
                            LdsNumElementsAlignedMXSA=2304, LdsNumElementsAlignedMXSB=2304,
                            DirectToVgprMXSA=0, DirectToVgprMXSB=0,
                            ProblemType={"DataType": _FakeDataType(bf16=False, f8=True, nbytes=1),
                                         "MXBlockA": 32, "MXBlockB": 32}))
    fA = 32768 + (32768 // 2048) * 8
    assert r["applicable"] is True and r["reason"] == "tight"
    assert r["offsets"]["ldsBaseMXSA"] == 2 * (2 * fA)          # after [A0][B0][A1][B1]
    assert r["offsets"]["ldsBaseMXSB"] == 2 * (2 * fA) + 2304   # MXSB right after MXSA

def test_fp32_skips():
    r = evaluate(_vw8_state(ProblemType={"DataType": _FakeDataType(bf16=False, half=False, nbytes=4)}))
    assert r["applicable"] is False and ("bf16" in r["reason"] or "fp16" in r["reason"])