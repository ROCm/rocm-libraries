import pytest
from Tensile.SolutionStructs.segment_interleave import evaluate, aligned_budget_ok, SEG

pytestmark = pytest.mark.unit

class _FakeDataType:
    # Mirrors the real DataType API the oracle uses, without importing rocisa.
    def __init__(self, bf16=True, half=False, f8=False, f4=False, nbytes=2.0):
        self._bf16 = bf16
        self._half = half
        self._f8 = f8
        self._f4 = f4
        self._nbytes = nbytes
    def isBFloat16(self):
        return self._bf16
    def isHalf(self):
        return self._half
    def is8bitFloat(self):
        return self._f8
    def isFloat4(self):
        return self._f4
    def numBytes(self):
        return self._nbytes

def _vw8_state(**ovr):
    # DataType is an object (mirrors the real API). MacroTile0/1 are derived below, not set
    # here, so a test only overrides the MI primitives it cares about and the tile follows.
    s = dict(NumWaves=4, WavefrontSize=32, DepthU=128,
             ISA=(12, 5, 0),
             LdsOffsetA=0, LdsBlockSizePerPadA=2048, LdsBlockSizePerPadB=2048,
             LdsPadA=8, LdsPadB=8, VectorWidthA=8, VectorWidthB=8,
             MatrixInstM=16, MatrixInstN=16, MIWaveGroup=[2, 2], MIWaveTile=[8, 8], TDMSplit=0, enableTDMA=1, enableTDMB=1,
             UnrollMajorLDSA=1, UnrollMajorLDSB=1,
             ProblemType=dict(Sparse=0, DataType=_FakeDataType(), MXBlockA=0, MXBlockB=0))
    s["ProblemType"] = {**s["ProblemType"], **ovr.pop("ProblemType", {})}
    s.update(ovr)
    # Derive MacroTile from the MI geometry unless a test overrides it explicitly, so every
    # state is MI-consistent by construction: MacroTile = MatrixInst * MIWaveTile * MIWaveGroup.
    s.setdefault("MacroTile0", s["MatrixInstM"] * s["MIWaveTile"][0] * s["MIWaveGroup"][0])
    s.setdefault("MacroTile1", s["MatrixInstN"] * s["MIWaveTile"][1] * s["MIWaveGroup"][1])
    return s

def test_vw8_applies_with_handedit_values():
    # Footprint-packed tight: writeStrideBytes = fA+fB (post-pad).
    r = evaluate(_vw8_state())
    assert r["applicable"] is True
    assert r["offsets"] == {"ldsBaseB": 33024, "writeStrideBytes": 66048,
                            "footprintPacked": True}

def test_vw4_fine_uses_port_split():
    # VWA=4 == WaveTileA/2 + TDMSplit -> port-split; same tight footprint as coarse.
    r = evaluate(_vw8_state(VectorWidthA=4, TDMSplit=1))
    assert r["applicable"] is True
    assert r["offsets"].get("portSplitA") is True
    assert r["offsets"]["writeStrideBytes"] == 66048 and r["offsets"]["ldsBaseB"] == 33024

def test_port_split_needs_tdmsplit():
    # Port-split needs TDMSplit's store split; VWA==WaveTileA/2 without TDMSplit must reject.
    r = evaluate(_vw8_state(VectorWidthA=4, TDMSplit=0))
    assert r["applicable"] is False and "WaveTileA/2" in r["reason"]

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

def test_vwb_fine_unaligned_uses_bcontig():
    # When a single vIdx would straddle a component (component span not a multiple of the vIdx
    # column advance), B cannot be split -> fall back to the bcontig layout [A0][B0][B1][A1]:
    # B stays whole (baseline reads, WaveTileB unrestricted), only A moves to a separate segment. MT*x224
    # VWB1: compCols=112, vIdxCols=32, 112 % 32 != 0. Span-neutral (blockSpan 0), B relocated.
    r = evaluate(_vw8_state(MIWaveTile=[8, 7], VectorWidthB=1))  # WaveTileB=7 -> MT1=224
    assert r["applicable"] is True and r["reason"] == "bcontig"
    o = r["offsets"]
    # fA=33024, fB=28896 -> ldsBaseB=fA, A stride=fA+2fB, B not interleaved.
    assert o["ldsBaseB"] == 33024 and o["writeStrideBytes"] == 33024 + 2 * 28896
    assert o["bBaseline"] is True and r["blockSpan"] == 0
    assert "BCONTIG" in r["segmentMap"]

def test_bcontig_small_mt_aligned_applies():
    # Small footprint where fA+2fB stays in one segment: bcontig pushes A1 to the next segment
    # boundary (bcontig-aligned, grows LDS). Needs PGR2 + forced (LDSSI=1); B stays whole/baseline.
    # DepthU=64 shrinks the footprint; MIWaveTile[1]=7 -> MT1=224, the unreadable odd-B tile.
    r = evaluate(_vw8_state(MIWaveTile=[8, 7], DepthU=64, VectorWidthB=1,
                            PrefetchGlobalRead=2, LDSSegmentInterleave=1))
    assert r["applicable"] is True and r["reason"] == "bcontig-aligned" and r["aligned"] is True
    o = r["offsets"]
    # fA=16512; A stride pushed to the segment boundary; B (fB=14448) stays contiguous/baseline.
    assert o["ldsBaseB"] == 16512 and o["writeStrideBytes"] == SEG and o["bBaseline"] is True
    assert r["blockSpan"] == SEG + 16512               # A1 ends at base+pre+fA
    assert "BCONTIG-ALIGNED" in r["segmentMap"]

def test_bcontig_small_mt_auto_skips():
    # Auto (-1) refuses the LDS-growing bcontig-aligned branch (like split aligned).
    r = evaluate(_vw8_state(MIWaveTile=[8, 7], DepthU=64, VectorWidthB=1, PrefetchGlobalRead=2))
    assert r["applicable"] is False and "auto: skip aligned" in r["reason"]

def test_bcontig_small_mt_needs_pgr2():
    r = evaluate(_vw8_state(MIWaveTile=[8, 7], DepthU=64, VectorWidthB=1,
                            PrefetchGlobalRead=1, LDSSegmentInterleave=1))
    assert r["applicable"] is False and "PGR" in r["reason"]

def test_vwa_finer_than_half_skips():
    # VWA=2 (numVec=4) is finer than WaveTileA/2 -> TDMSplit's 2-way split can't place it -> reject.
    r = evaluate(_vw8_state(VectorWidthA=2, TDMSplit=1))
    assert r["applicable"] is False and "WaveTileA/2" in r["reason"]

def test_small_mt_skips_without_pgr2():
    # Small footprint is the aligned candidate, but it requires PGR2 double-buffer; with no
    # PrefetchGlobalRead set it skips to baseline (buffering gate).
    r = evaluate(_vw8_state(DepthU=64))
    assert r["applicable"] is False and "PGR" in r["reason"]

def test_non_square_small_skips_without_pgr2():
    # Non-square: MIWaveTile[1]=4 -> MT1=128 (< MT0=256), VWB=4 <= WaveTileB=4. Coarse VW ok
    # (16*8=128 >= 256//2), fA+fB < SEG -> small MT; no PGR2 -> buffering gate skip.
    r = evaluate(_vw8_state(MIWaveTile=[8, 4], VectorWidthB=4))
    assert r["applicable"] is False and "PGR" in r["reason"]

def _aligned_tiles_disjoint(r, fA, fB):
    # Aligned footprint packing: A_i = i*pre, B_i = ldsBaseB + i*pre (base=0, ldsBaseB=fA).
    o = r["offsets"]; pre = o["writeStrideBytes"]; assert o["ldsBaseB"] == fA
    tiles = [(0, fA), (fA, fA + fB), (pre, pre + fA), (fA + pre, fA + pre + fB)]  # A0,B0,A1,B1
    tiles.sort()
    return all(tiles[i][1] <= tiles[i + 1][0] for i in range(3))

def test_aligned_applies_small_mt():
    # Shallow DepthU=64 (MT stays 256, MI-consistent): fA=fB=16512, sum<SEG -> aligned.
    # Footprint-packed, segment stride.
    r = evaluate(_vw8_state(DepthU=64, PrefetchGlobalRead=2, LDSSegmentInterleave=1))
    assert r["applicable"] is True and r["aligned"] is True
    assert r["offsets"] == {"ldsBaseB": 16512, "writeStrideBytes": SEG,
                            "footprintPacked": True}
    assert r["blockSpan"] == SEG + 16512 + 16512    # base(0) + pre(SEG) + fA + fB
    assert _aligned_tiles_disjoint(r, 16512, 16512)
    assert "ALIGNED" in r["segmentMap"]

def test_aligned_fine_vwb_applies():
    # Fine-VWB is supported on the aligned branch too (per-vIdx component jump + enough
    # LocalReadAddr +64K registers, see KernelWriter numVgprLocalReadAddr). DepthU=64 VWB2.
    r = evaluate(_vw8_state(DepthU=64, PrefetchGlobalRead=2,
                            LDSSegmentInterleave=1, VectorWidthB=2))
    assert r["applicable"] is True and r["aligned"] is True

def test_aligned_unequal_pad_no_overlap():
    # Asymmetric pads (padA=16 != padB=8): footprint packing places each tile at its own post-pad
    # size, so ldsBaseB=base+fA (no compensation) and the segment stride keeps all 4 tiles disjoint.
    # fA=16384+(16384//2048)*(16*2)=16640 ; fB=16384+(16384//2048)*(8*2)=16512
    r = evaluate(_vw8_state(DepthU=64, PrefetchGlobalRead=2,
                            LdsPadA=16, LdsPadB=8, LDSSegmentInterleave=1))
    assert r["applicable"] is True and r["aligned"] is True and r["offsets"]["footprintPacked"]
    assert r["offsets"]["ldsBaseB"] == 16640         # base + fA, no compensation
    assert _aligned_tiles_disjoint(r, 16640, 16512)

def test_aligned_skips_pgr_not_2():
    r = evaluate(_vw8_state(DepthU=64, PrefetchGlobalRead=1))
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
    r = evaluate(_vw8_state(DepthU=64, PrefetchGlobalRead=2, UseSubtileImpl=1))
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
    r = evaluate(_vw8_state(DepthU=64, PrefetchGlobalRead=2, LDSSegmentInterleave=-1))
    assert r["applicable"] is False and "auto" in r["reason"]
    r1 = evaluate(_vw8_state(DepthU=64, PrefetchGlobalRead=2, LDSSegmentInterleave=1))
    assert r1["applicable"] is True and r1["aligned"] is True

def test_miwavegroup_not_2x2_skips():
    # footprint packing assumes 2 waves per MFMA dim; NumWaves=4 with MIWG [4,1]/[1,4] passes
    # Solution.py's prod>1 / pow2 gates but would lose/OOB the component jump. Must skip.
    # MacroTile follows MIWG (derived in the helper), so each state stays MI-consistent.
    for miwg in ([4, 1], [1, 4]):
        r = evaluate(_vw8_state(MIWaveGroup=miwg))
        assert r["applicable"] is False and "MIWaveGroup" in r["reason"], miwg

def test_tdmsplit_composes():
    # TDMSplit composes: same offsets as the non-split path.
    r = evaluate(_vw8_state(TDMSplit=1))
    assert r["applicable"] is True
    assert r["offsets"] == evaluate(_vw8_state(TDMSplit=0))["offsets"]

def test_tile_major_skips():
    r = evaluate(_vw8_state(UnrollMajorLDSA=0))  # not unrollMajor -> deferred
    assert r["applicable"] is False and ("unrollMajor" in r["reason"] or "tile-major" in r["reason"])

def test_fp16_applies_same_as_bf16():
    # fp16 has the same bpe (2) and the same write/read paths -> identical offsets.
    r = evaluate(_vw8_state(ProblemType={"DataType": _FakeDataType(bf16=False, half=True)}))
    assert r["applicable"] is True
    assert r["offsets"] == {"ldsBaseB": 33024, "writeStrideBytes": 66048,
                            "footprintPacked": True}

def test_fp8_tight_applies():
    # fp8 (bpe=1) is now supported. DepthU=256 -> fA+fB >= SEG -> tight branch.
    r = evaluate(_vw8_state(DepthU=256,
                            ProblemType={"DataType": _FakeDataType(bf16=False, f8=True, nbytes=1)}))
    fA = 32768 + (32768 // 2048) * 8   # data + pad (LdsPadA=8, block=2048, bpe=1)
    assert r["applicable"] is True and r["reason"] == "tight"
    assert r["offsets"]["writeStrideBytes"] == 2 * fA
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

def test_fp4_tight_applies():
    # fp4 (bpe=0.5): fractional bpe needs DepthU>=512 for MT256; byte footprints stay integral.
    r = evaluate(_vw8_state(DepthU=512,
                            ProblemType={"DataType": _FakeDataType(bf16=False, f4=True, nbytes=0.5)}))
    fA = int(128 * 512 * 0.5) + (int(128 * 512 * 0.5) // 2048) * int(8 * 0.5)
    assert r["applicable"] is True and r["reason"] == "tight"
    assert r["offsets"]["writeStrideBytes"] == 2 * fA
    assert "ldsBaseMXSA" not in r["offsets"] and "ldsBaseMXSB" not in r["offsets"]

def test_mxf4_tight_relocates_scales():
    # mxf4: scales are 1 B/elem (bpe-independent), relocated to base+2*(fA+fB) as mxf8.
    r = evaluate(_vw8_state(DepthU=512,
                            LdsNumElementsAlignedMXSA=2304, LdsNumElementsAlignedMXSB=2304,
                            DirectToVgprMXSA=0, DirectToVgprMXSB=0,
                            ProblemType={"DataType": _FakeDataType(bf16=False, f4=True, nbytes=0.5),
                                         "MXBlockA": 32, "MXBlockB": 32}))
    fA = int(128 * 512 * 0.5) + (int(128 * 512 * 0.5) // 2048) * int(8 * 0.5)
    assert r["applicable"] is True and r["reason"] == "tight"
    assert r["offsets"]["ldsBaseMXSA"] == 2 * (2 * fA)
    assert r["offsets"]["ldsBaseMXSB"] == 2 * (2 * fA) + 2304

def test_fp32_skips():
    r = evaluate(_vw8_state(ProblemType={"DataType": _FakeDataType(bf16=False, half=False, nbytes=4)}))
    assert r["applicable"] is False and ("bf16" in r["reason"] or "fp16" in r["reason"])

def test_not_wave_separated_tdm_skips():
    # enableTDMA/B off (or NumWaves<=1) means the kernel is not wave-separated TDM -> no interleave.
    r = evaluate(_vw8_state(enableTDMA=0))
    assert r["applicable"] is False and "not wave-separated TDM" in r["reason"]

def test_localsplitu_skips():
    # LocalSplitU>1 changes the LDS reduction layout the offsets assume -> reject.
    r = evaluate(_vw8_state(LocalSplitU=2))
    assert r["applicable"] is False and "LocalSplitU" in r["reason"]

def test_numcomp_not_2_skips():
    # NumWaves//2 != 2 (NumWaves=6 -> 3 components); write/read both assume exactly 2 comps.
    # MIWaveGroup=[3,2] keeps prod=6=NumWaves (MT0=384 follows via the helper).
    r = evaluate(_vw8_state(NumWaves=6, MIWaveGroup=[3, 2]))
    assert r["applicable"] is False and "numComp!=2" in r["reason"]

def test_sparse_skips():
    # Sparse metadata adds its own LDS region; the interleave layout does not account for it.
    r = evaluate(_vw8_state(ProblemType={"Sparse": 1}))
    assert r["applicable"] is False and "sparse" in r["reason"]

def test_zero_pad_footprint_is_unpadded():
    # LdsPad*=0 (or block=0) -> _pad returns 0, so each footprint is the raw data size with no pad.
    r = evaluate(_vw8_state(LdsPadA=0))
    assert r["applicable"] is True
    # fA = MacroTile0//numComp * DepthU * bpe = 128 * 128 * 2 = 32768, no pad.
    assert r["offsets"]["ldsBaseB"] == 32768

def test_aligned_budget_nonpositive_block():
    # offsetBlk = max(naturalOffsetBlk, blockSpan) <= 0 -> nothing to place -> reject.
    ok, blk = aligned_budget_ok(0, 2, 0, 327680)
    assert ok is False and blk is None

def test_aligned_port_split():
    # port-split A (VWA=WaveTileA/2 + TDMSplit) composes with the aligned branch. VWA=4 stays
    # non-coarse at MT0=256 (16*4=64 < 128); DepthU=64 shrinks fA+fB below one segment -> aligned.
    r = evaluate(_vw8_state(DepthU=64, VectorWidthA=4, TDMSplit=1,
                            PrefetchGlobalRead=2, LDSSegmentInterleave=1))
    assert r["applicable"] is True and r["reason"] == "aligned"
    assert r["offsets"].get("portSplitA") is True

def test_bcontig_port_split():
    # port-split A composes with the bcontig fallback (unreadable B, whole-B gap pushes A1 over).
    r = evaluate(_vw8_state(MIWaveTile=[8, 7], VectorWidthB=1,
                            VectorWidthA=4, TDMSplit=1))  # WaveTileB=7 -> MT1=224
    assert r["applicable"] is True and r["reason"] == "bcontig"
    assert r["offsets"].get("portSplitA") is True

def test_bcontig_aligned_port_split():
    # port-split A composes with the bcontig-aligned branch (small tile, A1 padded to next segment).
    # VWA=4 keeps A non-coarse (port-split); MIWaveTile[1]=7 -> MT1=224 (unreadable odd-B);
    # DepthU=64 shrinks fA+2fB below one segment so A1 must be padded across the boundary.
    r = evaluate(_vw8_state(MIWaveTile=[8, 7], DepthU=64, VectorWidthB=1,
                            VectorWidthA=4, TDMSplit=1, PrefetchGlobalRead=2,
                            LDSSegmentInterleave=1))
    assert r["applicable"] is True and r["reason"] == "bcontig-aligned"
    assert r["offsets"].get("portSplitA") is True

def test_mixed_type_per_tensor_bpe():
    # Mixed types: DataType=fp8 passes the gate, but MacDataTypeA=fp4 (0.5B) / MacDataTypeB=fp8 (1B).
    # Footprints must use the per-tensor MacDataType, so fA != fB.
    r = evaluate(_vw8_state(
        DepthU=512,
        ProblemType={"DataType": _FakeDataType(bf16=False, f8=True, nbytes=1),
                     "MacDataTypeA": _FakeDataType(bf16=False, f4=True, nbytes=0.5),
                     "MacDataTypeB": _FakeDataType(bf16=False, f8=True, nbytes=1)}))
    fA = int(128 * 512 * 0.5) + (int(128 * 512 * 0.5) // 2048) * int(8 * 0.5)
    fB = int(128 * 512 * 1.0) + (int(128 * 512 * 1.0) // 2048) * int(8 * 1.0)
    assert fA != fB, "test setup must exercise asymmetric footprints"
    assert r["applicable"] is True and r["reason"] == "tight"
    assert r["offsets"]["ldsBaseB"] == fA
    assert r["offsets"]["writeStrideBytes"] == fA + fB
