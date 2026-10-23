import pytest
from tensilelite.SolutionStructs.segment_interleave import evaluate, aligned_budget_ok, SEG

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

def test_vw4_halfvw_uses_port_split():
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

def test_vwb_halfvw_aligned_applies():
    # VWB=WaveTileB/2 is OK when each VW-group (vIdx) stays within one component, i.e. the
    # component column span is a whole multiple of one vIdx's column advance. For
    # MT256x256 (compCols=128), VWB in {1,2,4} all divide cleanly -> apply. LocalRead
    # (calcGfx1250LdsOffset) adds the per-vIdx component jump. GPU-validated 8/4/2/1.
    for vwb in (4, 2, 1):
        r = evaluate(_vw8_state(VectorWidthB=vwb))
        assert r["applicable"] is True, f"VWB={vwb} should apply"

def test_vwb_halfvw_unaligned_uses_bcontig():
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
    # boundary (bcontig-aligned, grows LDS). Needs prefetch + forced (LDSSI=1); B stays whole/baseline.
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

def test_bcontig_small_mt_allows_pgr1():
    r = evaluate(_vw8_state(MIWaveTile=[8, 7], DepthU=64, VectorWidthB=1,
                            PrefetchGlobalRead=1, LDSSegmentInterleave=1))
    assert r["applicable"] is True and r["aligned"] is True

def test_vwa_smaller_than_half_skips():
    # VWA=2 (numVec=4) is smaller than WaveTileA/2 -> TDMSplit's 2-way split can't place it -> reject.
    r = evaluate(_vw8_state(VectorWidthA=2, TDMSplit=1))
    assert r["applicable"] is False and "WaveTileA/2" in r["reason"]

def test_small_mt_skips_without_prefetch():
    # Small footprint is the aligned candidate, but it requires double-buffered prefetch; with no
    # PrefetchGlobalRead set it skips to baseline (buffering gate).
    r = evaluate(_vw8_state(DepthU=64))
    assert r["applicable"] is False and "PGR" in r["reason"]

def test_non_square_small_skips_without_prefetch():
    # Non-square: MIWaveTile[1]=4 -> MT1=128 (< MT0=256), VWB=4 <= WaveTileB=4. Coarse VW ok
    # (16*8=128 >= 256//2), fA+fB < SEG -> small MT; no prefetch -> buffering gate skip.
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

def test_aligned_halfvw_vwb_applies():
    # VWB=WaveTileB/2 is supported on the aligned branch too (per-vIdx component jump + enough
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

def test_aligned_allows_pgr1():
    r = evaluate(_vw8_state(DepthU=64, PrefetchGlobalRead=1, LDSSegmentInterleave=1))
    assert r["applicable"] is True and r["aligned"] is True

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

def test_miwg_4x1_large_tile_baseline_sufficient():
    # [4,1] large tile: active data fits one segment (fActData <= SEG) AND baseline already lands
    # A0/A1 in different segments -> no interleave needed (only pad tail spills, negligible).
    r = evaluate(_vw8_state(MIWaveGroup=[4, 1]))            # MIWaveTile[8,8] -> fActData=SEG
    assert r["applicable"] is False
    assert "baseline" in r["reason"], r["reason"]

def test_miwg_1x4_large_tile_interleaves_not_baseline():
    # [1,4]: B active, but baseline lays out [A][MX][B] so B sits at a non-aligned offset -> its
    # comps span/overlap a segment even at fB==SEG. No baseline shortcut here; interleave realigns
    # B to offset 0 (bcontig) -> aBaseline. (Contrast [4,1] where A@0 stays baseline.)
    r = evaluate(_vw8_state(MIWaveGroup=[1, 4]))
    assert r["applicable"] is True
    assert r["offsets"]["aBaseline"] is True

def test_asym_componentsplit_tdmsplit():
    # TDMSplit + asymmetric MIWaveGroup ([4,1]/[1,4]) with a VW==WaveTile/2 active tensor
    # component axis: the two components go to different segments, each wave's two load-halves stay
    # together -- halves conflict at the same load count. Marked componentSplit (not portSplitA/B).
    r = evaluate(_vw8_state(MIWaveGroup=[4, 1], TDMSplit=1, VectorWidthA=4))
    assert r["applicable"] is True and r["reason"] == "compaxis-asym"
    assert r["offsets"].get("componentSplit") is True
    assert r["offsets"].get("activeTC") == "A" and r["offsets"].get("bBaseline") is True
    assert "portSplitA" not in r["offsets"]
    r = evaluate(_vw8_state(MIWaveGroup=[1, 4], TDMSplit=1, VectorWidthB=4))
    assert r["applicable"] is True and r["reason"] == "compaxis-asym"
    assert r["offsets"].get("componentSplit") is True
    assert r["offsets"].get("activeTC") == "B" and r["offsets"].get("aBaseline") is True
    assert "portSplitB" not in r["offsets"]

def test_asym_componentsplit_aligned_labels_compaxis():
    # componentSplit that falls into the aligned branch (small DepthU -> both components fit one
    # segment) must report compaxis, not portaxis, in reason/segmentMap and in the PGR reject.
    r = evaluate(_vw8_state(MIWaveGroup=[4, 1], TDMSplit=1, VectorWidthA=4, DepthU=64,
                            PrefetchGlobalRead=2, LDSSegmentInterleave=1))
    assert r["applicable"] is True and r["aligned"] is True
    assert r["reason"] == "compaxis-asym-aligned" and "COMPAXIS-ALIGNED" in r["segmentMap"]
    assert r["offsets"].get("componentSplit") is True
    r = evaluate(_vw8_state(MIWaveGroup=[4, 1], TDMSplit=1, VectorWidthA=4, DepthU=64,
                            PrefetchGlobalRead=1, LDSSegmentInterleave=1))
    assert r["applicable"] is False and r["reason"].startswith("compaxis-asym")

def test_asym_4x1_coarse_tdmsplit_port_axis():
    # Coarse A ([4,1], default VWA=8 == WaveTileA=8) + TDMSplit -> portSplit layout:
    # [active.port0][shared][active.port1], shared (B) stays whole/baseline.
    st = _vw8_state(MIWaveGroup=[4, 1], TDMSplit=1)
    r = evaluate(st)
    assert r["applicable"] and r["offsets"]["portSplitA"] is True
    assert r["offsets"].get("bBaseline") is True
    assert "portSplitB" not in r["offsets"]
    # fAct=footprint(A)=66048, fSh=footprint(B)=16512 -> shared @ base+fAct, stride=fAct+2*fSh.
    assert r["offsets"]["ldsBaseB"] == 66048
    assert r["offsets"]["writeStrideBytes"] == 66048 + 2 * 16512
    assert r["blockSpan"] == 0

def test_asym_1x4_coarse_tdmsplit_port_axis():
    # Mirror: coarse B ([1,4], default VWB=8 == WaveTileB=8) + TDMSplit -> portSplit, A shared.
    st = _vw8_state(MIWaveGroup=[1, 4], TDMSplit=1)
    r = evaluate(st)
    assert r["applicable"] and r["offsets"]["portSplitB"] is True
    assert r["offsets"].get("aBaseline") is True
    assert "portSplitA" not in r["offsets"]
    assert r["offsets"]["ldsBaseA"] == 66048
    assert r["offsets"]["writeStrideBytes"] == 66048 + 2 * 16512

def test_asym_tdmsplit_tight_reason():
    # Coarse [4,1]+TDMSplit at DepthU=128: the shared B block already pushes port1 into the next
    # segment (base+strideAct crosses base+fActData's segment) -> tight portSplit, no extra LDS.
    r = evaluate(_vw8_state(MIWaveGroup=[4, 1], TDMSplit=1))
    assert r["applicable"] is True and r["reason"] == "portaxis-asym"
    assert r["aligned"] is False and r["blockSpan"] == 0
    assert r["offsets"].get("portSplitA") is True

def test_asym_tdmsplit_nontight_uses_aligned():
    # Coarse [4,1]+TDMSplit, DepthU=64: fAct+2*fSh stays inside comp0's segment, so port1 does NOT
    # cross on its own -> aligned fallback pads the portSplit stride to the next segment (grows LDS).
    r = evaluate(_vw8_state(MIWaveGroup=[4, 1], TDMSplit=1, DepthU=64,
                            PrefetchGlobalRead=2, LDSSegmentInterleave=1))
    assert r["applicable"] is True and r["reason"] == "portaxis-asym-aligned"
    assert r["aligned"] is True
    o = r["offsets"]
    # fAct=33024, fSh=8256, strideAct=49536 (<SEG) -> pre padded to SEG.
    assert o.get("portSplitA") is True and o.get("bBaseline") is True
    assert o["ldsBaseB"] == 33024 and o["writeStrideBytes"] == SEG
    assert r["blockSpan"] == SEG + 33024               # base + pre + fAct

def test_asym_tdmsplit_nontight_auto_skips():
    # Auto (-1) refuses the LDS-growing aligned fallback for the portSplit asym branch.
    r = evaluate(_vw8_state(MIWaveGroup=[4, 1], TDMSplit=1, DepthU=64, PrefetchGlobalRead=2))
    assert r["applicable"] is False and "auto" in r["reason"]

def test_asym_tdmsplit_nontight_needs_pgr2():
    # The portSplit aligned fallback double-buffers -> requires PGR2.
    r = evaluate(_vw8_state(MIWaveGroup=[4, 1], TDMSplit=1, DepthU=64,
                            PrefetchGlobalRead=1, LDSSegmentInterleave=1))
    assert r["applicable"] is False and "PGR" in r["reason"]

def test_asym_tdmsplit_nontight_mirror_1x4_aligned():
    # [1,4] mirror: B active, non-tight -> aligned fallback, aBaseline + portSplitB.
    r = evaluate(_vw8_state(MIWaveGroup=[1, 4], TDMSplit=1, DepthU=64,
                            PrefetchGlobalRead=2, LDSSegmentInterleave=1))
    assert r["applicable"] is True and r["reason"] == "portaxis-asym-aligned"
    o = r["offsets"]
    assert o.get("portSplitB") is True and o.get("aBaseline") is True
    assert o["ldsBaseA"] == 33024 and o["writeStrideBytes"] == SEG

def test_asym_smaller_than_half_tdmsplit_rejected():
    # VWA=2 with WaveTileA=8 -> numVec=4 (smaller than WaveTile/2): a port/comp would need >2 pieces,
    # but TDMSplit gives only a 2-way split. Only VW==WaveTile/2 (componentSplit) is supported.
    st = _vw8_state(MIWaveGroup=[4, 1], TDMSplit=1, VectorWidthA=2)
    r = evaluate(st)
    assert r["applicable"] is False and "TDMSplit" in r["reason"]

def test_directtovgpr_operand_rejected():
    # DirectToVgpr operands bypass LDS -> operand LDS placement is inapplicable; must reject.
    for kw in (dict(DirectToVgprA=True), dict(DirectToVgprB=True)):
        r = evaluate(_vw8_state(**kw))
        assert r["applicable"] is False and "DirectToVgpr" in r["reason"], kw

def test_miwg_4x1_nonzero_base_no_false_baseline_shortcut():
    # fp8/fp4 LDS-transpose kernels use a half-wave shift (LdsOffsetA=4). With fActData==SEG the
    # active comp0 data straddles into the next segment where comp1 also lands, so the [4,1] baseline
    # shortcut must NOT fire (it would wrongly report the conflict as already resolved). base=0 is
    # genuinely clean and still takes the shortcut.
    f8 = _FakeDataType(bf16=False, f8=True, nbytes=1.0)
    kw = dict(MIWaveGroup=[4, 1], MIWaveTile=[8, 4], DepthU=256, VectorWidthA=8, VectorWidthB=4,
              ProblemType={"DataType": f8})   # MT0=512 -> fActData = 256*256*1 = SEG
    r4 = evaluate(_vw8_state(LdsOffsetA=4, **kw))
    assert not (r4["applicable"] is False and "baseline already separates" in r4["reason"]), \
        "base=4, fActData==SEG must not take the (false) baseline shortcut"
    r0 = evaluate(_vw8_state(LdsOffsetA=0, **kw))
    assert r0["applicable"] is False and "baseline already separates" in r0["reason"], \
        "base=0, fActData==SEG is genuinely clean -> baseline shortcut"

def test_miwg_4x1_small_tile_bcontig():
    # [4,1] small tile: baseline packs A0/A1 in one segment -> interleave separates them with the
    # whole shared B as the gap ([A0][B_whole][A1]) -> bBaseline.
    r = evaluate(_vw8_state(MIWaveGroup=[4, 1], MIWaveTile=[4, 8], VectorWidthA=4))
    assert r["applicable"] is True
    assert r["offsets"]["bBaseline"] is True

def test_miwg_1x4_small_tile_bcontig_mirror():
    # Mirror small [1,4] -> aBaseline.
    r = evaluate(_vw8_state(MIWaveGroup=[1, 4], MIWaveTile=[8, 4], VectorWidthB=4))
    assert r["applicable"] is True
    assert r["offsets"]["aBaseline"] is True

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


def _tdmfuse_state(tdmFuse, **ovr):
    # TDMFuse 1-3 are accepted only with TDM on both tensors and MX scales.
    pt = dict(DataType=_FakeDataType(bf16=False, f8=True, nbytes=1.0), MXBlockA=32, MXBlockB=32)
    pt.update(ovr.pop("ProblemType", {}))
    kw = dict(TDMFuse=tdmFuse, TDMInst=3, DepthU=256)
    kw.update(ovr)
    return _vw8_state(ProblemType=pt, **kw)

# [2,2] tight, [2,2] with B kept whole, [4,1] with A active and its [1,4] mirror.
_TDMFUSE_LAYOUTS = {
    "tight": dict(),
    "bcontig": dict(MIWaveTile=[8, 7], VectorWidthB=4),
    "asym_4x1": dict(MIWaveGroup=[4, 1], MIWaveTile=[4, 8], VectorWidthA=4),
    "asym_1x4": dict(MIWaveGroup=[1, 4], MIWaveTile=[8, 4], VectorWidthB=4),
    "tight_f8f4": dict(DepthU=512, ProblemType=dict(MacDataTypeA=_FakeDataType(bf16=False, f8=True, nbytes=1.0),
                                        MacDataTypeB=_FakeDataType(bf16=False, f4=True, nbytes=0.5))),
}

def _tc_bpe(state, tc):
    pt = state["ProblemType"]
    return pt.get("MacDataType%s" % tc, pt["DataType"]).numBytes()

def _tc_mt(state, tc):
    return state["MacroTile0"] if tc == "A" else state["MacroTile1"]

def _piece_lds_addrs(state, tc):
    """LDS address of every TDM piece of `tc`, run through the writer's SGPR arithmetic."""
    from tensilelite.Components.TDMFuse import tdmWavePartition
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly

    class _Writer:
        tdmWaveLdsAddr = KernelWriterAssembly.tdmWaveLdsAddr

    numComp = tdmWavePartition(state, tc)[0]
    bpe = _tc_bpe(state, tc)
    mod = _Writer().tdmWaveLdsAddr(state, tc, numComp,
                                   int(_tc_mt(state, tc) // numComp * state["DepthU"] * bpe),
                                   state.get("LdsOffset%s" % tc, 0),
                                   state["LdsBlockSizePerPad%s" % tc],
                                   int(state["LdsPad%s" % tc] * bpe), 1, 2)
    ops = {"s_and_b32": lambda x, y: x & y, "s_mul_i32": lambda x, y: x * y,
           "s_lshr_b32": lambda x, y: x >> y, "s_add_u32": lambda x, y: x + y}
    addrs = []
    for piece in range(numComp):
        regs = {"s1": piece}
        for item in mod.flatitems():
            op, dst, *srcs = str(item).split("//")[0].replace(",", " ").split()
            regs[dst] = ops[op](*[regs[s] if s in regs else int(s, 0) for s in srcs])
        addrs.append(regs["s1"])
    return addrs

def _read_addr(state, offsets, tc, row):
    """Where the local reads expect `row` of `tc`: component jump plus the padded offset in it."""
    bpe = _tc_bpe(state, tc)
    whole = offsets.get("aBaseline" if tc == "A" else "bBaseline", False)
    compRows = _tc_mt(state, tc) // (1 if whole else 2)
    jump = 0 if whole else row // compRows * offsets["writeStrideBytes"]
    x = int(row % compRows * state["DepthU"] * bpe)
    pad = x // state["LdsBlockSizePerPad%s" % tc] * int(state["LdsPad%s" % tc] * bpe)
    return offsets.get("ldsBase%s" % tc, state.get("LdsOffset%s" % tc, 0)) + jump + x + pad

@pytest.mark.parametrize("layout", sorted(_TDMFUSE_LAYOUTS))
@pytest.mark.parametrize("mode", [-1, 1])
@pytest.mark.parametrize("tdmFuse", [1, 2, 3])
def test_tdmfuse_rows_keep_the_two_component_layout(tdmFuse, mode, layout):
    r = evaluate(_tdmfuse_state(tdmFuse, LDSSegmentInterleave=mode, **_TDMFUSE_LAYOUTS[layout]))
    assert r["applicable"] is True, r["reason"]
    assert r == evaluate(_tdmfuse_state(0, LDSSegmentInterleave=mode, **_TDMFUSE_LAYOUTS[layout]))

@pytest.mark.parametrize("layout", sorted(_TDMFUSE_LAYOUTS))
@pytest.mark.parametrize("tdmFuse", [0, 1, 2, 3])
def test_tdm_pieces_land_where_the_local_reads_expect_them(tdmFuse, layout):
    from tensilelite.Components.TDMFuse import tdmWavePartition

    state = _tdmfuse_state(tdmFuse, LDSSegmentInterleave=1, **_TDMFUSE_LAYOUTS[layout])
    r = evaluate(state)
    assert r["applicable"] is True, r["reason"]
    state["LDSSegInterleaveOffsets"] = r["offsets"]
    for tc in ("A", "B"):
        numComp = tdmWavePartition(state, tc)[0]
        rows = _tc_mt(state, tc) // numComp
        expected = [_read_addr(state, r["offsets"], tc, p * rows) for p in range(numComp)]
        assert _piece_lds_addrs(state, tc) == expected, (tc, numComp)


# ---- LDSSegmentBRotate gate (b_rotate_reason) ----
from tensilelite.SolutionStructs.segment_interleave import b_rotate_reason, paired_buffer_offsets

def _brot_state(**ovr):
    # F8 MT128x128x256 [2,2] aligned layout: A0,B0 fit seg0, so the oracle pads comp 1 to 64K.
    s = _vw8_state(DepthU=256, MatrixInstM=16, MatrixInstN=16, MIWaveTile=[4, 4],
                   VectorWidthA=4, VectorWidthB=1, LdsBlockSizePerPadA=256, LdsBlockSizePerPadB=256,
                   LdsPadA=16, LdsPadB=16, PrefetchGlobalRead=2, LDSSegmentInterleave=1, MaxLDS=5*SEG,
                   ProblemType=dict(DataType=_FakeDataType(bf16=False, f8=True, nbytes=1.0)))
    r = evaluate(s)
    assert r["applicable"] and r["aligned"], r["reason"]
    s.update({"LDSSegInterleaveOffsets": r["offsets"], "NumLdsBlk": 2, "1LDSBuffer": 0,
              "LdsOffsetA_Blk": 131072, "StoreSwapAddr": False, "ExpandPointerSwap": 0,
              "LDSSegmentBRotate": 1, "HalfPLR": 0, "LocalSplitU": 1})
    s.update(ovr)
    return s

def test_brotate_applies_on_aligned_22():
    assert b_rotate_reason(_brot_state()) is None
    assert b_rotate_reason(_brot_state(VectorWidthB=2)) is None
    # -1 = auto/unresolved tri-state knobs must not block.
    assert b_rotate_reason(_brot_state(UseCustomMainLoopSchedule=-1)) is None

def test_brotate_needs_even_vidx_count():
    assert "even" in b_rotate_reason(_brot_state(VectorWidthB=4))

def test_brotate_needs_lsi_aligned_layout():
    assert "LDSSegmentInterleave" in b_rotate_reason(_brot_state(LDSSegmentInterleave=0))
    s = _brot_state()
    s["LDSSegInterleaveOffsets"] = {**s["LDSSegInterleaveOffsets"], "writeStrideBytes": 140000}
    assert "[64K, 128K)" in b_rotate_reason(s)

def test_brotate_rejects_unsupported():
    assert "gfx1250" in b_rotate_reason(_brot_state(ISA=(9, 5, 0)))
    assert "TDMSplit" in b_rotate_reason(_brot_state(TDMSplit=1))
    assert "persistent" in b_rotate_reason(_brot_state(), persistent=True)

def _brot_tight_state(**ovr):
    # F8 MT256x256x256 [2,2]: A0+B0 exceed 64K, so the oracle packs tight [A0][B0][A1][B1]
    # with component stride S = fA+fB (64K < S < 128K); LDS buffers swap through StoreSwapAddr.
    s = _vw8_state(DepthU=256, MatrixInstM=16, MatrixInstN=16, MIWaveTile=[8, 8],
                   VectorWidthA=8, VectorWidthB=4, LdsBlockSizePerPadA=256, LdsBlockSizePerPadB=256,
                   LdsPadA=16, LdsPadB=16, PrefetchGlobalRead=2, LDSSegmentInterleave=1, MaxLDS=5*SEG,
                   ProblemType=dict(DataType=_FakeDataType(bf16=False, f8=True, nbytes=1.0)))
    r = evaluate(s)
    assert r["applicable"] and r["reason"] == "tight", r["reason"]
    s.update({"LDSSegInterleaveOffsets": r["offsets"], "NumLdsBlk": 2, "1LDSBuffer": 0,
              "LdsOffsetA_Blk": 2 * r["offsets"]["writeStrideBytes"], "StoreSwapAddr": True,
              "ExpandPointerSwap": 0, "LDSSegmentBRotate": 1, "HalfPLR": 3, "LocalSplitU": 1})
    s.update(ovr)
    return s

def test_brotate_applies_on_tight_22_with_halfplr():
    s = _brot_tight_state()
    assert SEG < s["LDSSegInterleaveOffsets"]["writeStrideBytes"] < 2 * SEG
    assert b_rotate_reason(s) is None
    for vwb in (1, 2):
        assert b_rotate_reason(_brot_tight_state(VectorWidthB=vwb)) is None

def _brot_mx_state(**ovr):
    # MXF8 on the tight MT256 layout: VWB4 on MIWaveTile[1]=8 gives nV=2, so the MXSB TileSpan
    # layout holds vIdx 0/1 scales in the lower/upper half-wave of one ds_load.
    s = _brot_tight_state(MXScaleFormat="InMemorySwizzle", WavefrontSize=32)
    s["ProblemType"] = {**s["ProblemType"], "MXBlockA": 32, "MXBlockB": 32}
    s.update(ovr)
    return s

def test_brotate_mx_applies_on_tilespan_single_group():
    assert b_rotate_reason(_brot_mx_state()) is None

def test_brotate_mx_rejects_unsupported_scale_layouts():
    assert "MIWaveTile[1]/VectorWidthB == 2" in b_rotate_reason(_brot_mx_state(VectorWidthB=1))
    assert "MIWaveTile[1]/VectorWidthB == 2" in b_rotate_reason(_brot_mx_state(VectorWidthB=2))
    assert "DirectToVgprMXSB" in b_rotate_reason(_brot_mx_state(DirectToVgprMXSB=True))
    assert "InMemorySwizzle" in b_rotate_reason(_brot_mx_state(MXScaleFormat="NoSwizzle"))
    assert "(12, 5, 0)" in b_rotate_reason(_brot_mx_state(ISA=(12, 5, 1)))

def _brot_rejects(state, text):
    reason = b_rotate_reason(state)
    assert reason is not None and text in reason, reason

def test_brotate_rejects_unsupported_lsi_layouts():
    for key in ("aBaseline", "bBaseline", "portSplitA", "portSplitB", "componentSplit"):
        s = _brot_state()
        s["LDSSegInterleaveOffsets"] = {**s["LDSSegInterleaveOffsets"], key: True}
        _brot_rejects(s, "LSI layout '%s'" % key)

def test_brotate_rejects_b_reach_and_base():
    s = _brot_state()
    o = s["LDSSegInterleaveOffsets"]
    s["LDSSegInterleaveOffsets"] = {**o, "writeStrideBytes": 100000, "ldsBaseB": 0}
    _brot_rejects(s, "ldsBaseB below")
    s["LDSSegInterleaveOffsets"] = {**o, "writeStrideBytes": 131000, "ldsBaseB": 131000}
    _brot_rejects(s, "reach exceeds")

def test_brotate_rejects_buffering_and_schedule_modes():
    _brot_rejects(_brot_state(NumLdsBlk=1), "2 LDS buffers")
    _brot_rejects(_brot_state(**{"1LDSBuffer": 1}), "2 LDS buffers")
    assert b_rotate_reason(_brot_state(PrefetchGlobalRead=1)) is None
    _brot_rejects(_brot_state(PrefetchGlobalRead=0), "PrefetchGlobalRead=1 or 2")
    _brot_rejects(_brot_state(ExpandPointerSwap=1), "ExpandPointerSwap=0")

def test_brotate_rejects_wave_config():
    _brot_rejects(_brot_state(MIWaveGroup=[4, 1]), "[2,2]")
    _brot_rejects(_brot_state(NumWaves=8), "[2,2]")
    _brot_rejects(_brot_state(WavefrontSize=64), "wave32")
    _brot_rejects(_brot_state(VectorWidthB=0), "even")

def test_brotate_rejects_mx_matrix_inst_and_sparse():
    _brot_rejects(_brot_mx_state(MatrixInstN=32), "MatrixInstN")
    s = _brot_state()
    s["ProblemType"] = {**s["ProblemType"], "Sparse": 1}
    _brot_rejects(s, "sparse")

def test_brotate_rejects_each_unsupported_knob():
    for key in ("UseSubtileImpl", "DirectToVgprA", "DirectToVgprB", "DirectToLdsB", "enableLDSTrA",
                "enableLDSTrB", "UseDotInstruction", "UseCustomMainLoopSchedule",
                "PrefetchAcrossPersistent", "TDMFuse"):
        _brot_rejects(_brot_state(**{key: 1}), key)
    _brot_rejects(_brot_state(LocalSplitU=2), "LocalSplitU")
    _brot_rejects(_brot_state(NumWaveSplitK=2), "NumWaveSplitK")
def test_brotate_uses_per_component_buffer_masks():
    # Tight layouts use paired StoreSwapAddr masks; aligned layouts retain
    # whole-buffer swapping. The BRotate gate accepts either layout.
    assert b_rotate_reason(_brot_tight_state(StoreSwapAddr=False)) is None
    assert b_rotate_reason(_brot_state(StoreSwapAddr=False)) is None


def test_brotate_paired_buffers_matches_validated_fp8_layout():
    s = _brot_tight_state()
    s.update(LdsBlockSizePerPadA=2048, LdsBlockSizePerPadB=1024)
    s["LDSSegInterleaveOffsets"] = evaluate(s)["offsets"]
    assert b_rotate_reason(s) is None
    o = paired_buffer_offsets(s)
    assert o["writeStrideBytes"] == 132608
    assert o["readStrideBytesB"] == 66304
    assert o["ldsBaseB"] == 66048
    assert o["bufferStrideA"] == 33024
    assert o["bufferStrideB"] == 33280
    # Re-resolving a serialized solution must not double the component stride again.
    s["LDSSegInterleaveOffsets"] = o
    assert b_rotate_reason(s) is None
    assert paired_buffer_offsets(s) == o


@pytest.mark.parametrize("make_state", [_brot_state, _brot_tight_state])
def test_brotate_paired_buffers_read_write_and_swap_agree(make_state):
    s = make_state()
    original = dict(s["LDSSegInterleaveOffsets"])
    o = paired_buffer_offsets(s)
    assert s["LDSSegInterleaveOffsets"] == original
    # All eight padded footprints are disjoint and fit the allocation. Segment
    # gaps may be present between component pairs, just as with rotation off.
    regions = []
    for tc in ("A", "B"):
        size = o["bufferStride" + tc]
        base = s["LdsOffsetA"] if tc == "A" else o["ldsBaseB"]
        for component in range(2):
            for buf in range(2):
                start = base + component * o["writeStrideBytes"] + buf * size
                regions.append((start, start + size))
    regions.sort()
    assert all(a[1] <= b[0] for a, b in zip(regions, regions[1:]))
    assert regions[-1][1] == o["pairedUsedBytes"]
    assert regions[-1][1] <= 2 * s["LdsOffsetA_Blk"]
    # For every byte of a B component, both rotations and both buffer states must
    # address exactly the data placed there by TDM. Exercise padding and 64K carries.
    physical = o["writeStrideBytes"]
    logical = o["readStrideBytesB"]
    for pair in range(2):
        root = o["ldsBaseB"] + pair * physical
        delta = physical - logical % SEG - pair * 2 * physical
        for within in range(o["bufferStrideB"]):
            ping = root + within
            pong = ping + o["bufferStrideB"]
            mask = ping ^ pong
            assert pong ^ mask == ping
            for buf, addr in enumerate((ping, pong)):
                for half in range(2):
                    read = addr + half * delta + half * (logical % SEG)
                    written = (o["ldsBaseB"] + (half ^ pair) * physical
                               + buf * o["bufferStrideB"] + within)
                    assert read == written



@pytest.mark.parametrize("make_state", [_brot_state, _brot_tight_state])
@pytest.mark.parametrize("depth", [128, 256])
@pytest.mark.parametrize("padding", [0, 16])
@pytest.mark.parametrize("mx", [False, True])
def test_brotate_reuses_common_physical_layout(make_state, depth, padding, mx):
    s = make_state(DepthU=depth, LdsPadA=padding, LdsPadB=padding,
                   VectorWidthB=2 if make_state is _brot_state else 4,
                   MXScaleFormat="InMemorySwizzle",
                   LdsNumElementsAlignedMXSA=1024, LdsNumElementsAlignedMXSB=2048)
    s["ProblemType"] = {**s["ProblemType"], "MXBlockA": 32 if mx else 0, "MXBlockB": 32 if mx else 0}
    s["LDSSegInterleaveOffsets"] = evaluate(s)["offsets"]
    assert b_rotate_reason(s) is None
    ordinary = paired_buffer_offsets({**s, "LDSSegmentBRotate": 0})
    rotated = paired_buffer_offsets(s)
    # Only the read-address representation changes. Bases, component/buffer
    # strides, MX scale placement and allocation extent must be identical.
    readKeys = {"readStrideBytesB", "readAddressWindowsB"}
    physical = lambda offsets: {k: v for k, v in offsets.items() if k not in readKeys}
    assert physical(rotated) == physical(ordinary)
    assert "readAddressWindowsB" not in rotated
    assert rotated["readStrideBytesB"] == s["LDSSegInterleaveOffsets"]["writeStrideBytes"]
    for tc in ("A", "B"):
        size = rotated["bufferStride" + tc]
        for buffer in (0, 1):
            first = rotated["ldsBase" + tc] + buffer * size
            second = first + rotated["writeStrideBytes"]
            firstSegments = set(range(first//SEG, (first+size-1)//SEG+1))
            secondSegments = set(range(second//SEG, (second+size-1)//SEG+1))
            assert firstSegments.isdisjoint(secondSegments)
    s["LDSSegInterleaveOffsets"] = rotated
    assert paired_buffer_offsets(s) == rotated


def test_brotate_aligned_pong_tail_separation_and_budget():
    s = _brot_state()
    o = paired_buffer_offsets(s)
    # The previous 68 KiB stride put B0 pong [51,68) and B1 pong [119,136)
    # in S1. The common 77 KiB stride puts B1 pong at exactly 128 KiB.
    assert o["writeStrideBytes"] == 77*1024
    assert o["ldsBaseB"] + o["writeStrideBytes"] + o["bufferStrideB"] == 2*SEG
    s["MaxLDS"] = o["pairedUsedBytes"]
    assert paired_buffer_offsets(s) == o
    s["MaxLDS"] -= 16
    assert paired_buffer_offsets(s) is None


def test_brotate_paired_buffers_rejects_incompatible_tdm_layouts():
    assert "both operands" in b_rotate_reason(_brot_tight_state(enableTDMB=False))
    assert "decoupled" in b_rotate_reason(_brot_tight_state(PrefetchGlobalReadA=2))
    assert "A component" in b_rotate_reason(_brot_tight_state(LdsPadA=512))


@pytest.mark.parametrize("make_state", [_brot_state, _brot_tight_state])
@pytest.mark.parametrize("direct_a", [False, True])
def test_brotate_paired_mx_copies_do_not_overlap_data(make_state, direct_a):
    s = make_state(VectorWidthB=2 if make_state is _brot_state else 4,
                   MXScaleFormat="InMemorySwizzle", DirectToVgprMXSA=direct_a,
                   LdsNumElementsAlignedMXSA=2048, LdsNumElementsAlignedMXSB=4096)
    s["ProblemType"] = {**s["ProblemType"], "MXBlockA": 32, "MXBlockB": 32}
    original = evaluate(s)
    s["LDSSegInterleaveOffsets"] = original["offsets"]
    assert b_rotate_reason(s) is None
    o = paired_buffer_offsets(s)
    data_end = o["ldsBaseB"] + o["writeStrideBytes"] + 2 * o["bufferStrideB"]
    regions = [(s["LdsOffsetA"], data_end)]
    for tc in ("MXSA", "MXSB"):
        if tc == "MXSA" and direct_a:
            assert "ldsBaseMXSA" not in o
            continue
        size = s["LdsNumElementsAligned" + tc]
        assert o["bufferStride" + tc] == size
        for buf in range(2):
            start = o["ldsBase" + tc] + buf * size
            regions.append((start, start + size))
            # Per-wave TDM and per-lane local-read masks must both round-trip.
            for within in (0, size // 2, size - 1):
                ping = o["ldsBase" + tc] + within
                mask = ping ^ (ping + size)
                assert (ping ^ mask) == ping + size
                assert ((ping + size) ^ mask) == ping
    regions.sort()
    assert all(a[1] == b[0] for a, b in zip(regions, regions[1:]))
    one_scale_copy = (0 if direct_a else 2048) + 4096
    assert regions[-1][1] <= 2 * (s["LdsOffsetA_Blk"] + one_scale_copy)
    s["LDSSegInterleaveOffsets"] = o
    assert paired_buffer_offsets(s) == o


# Paired placement preserves logical groups and validates the complete two-buffer footprint.
@pytest.mark.parametrize("wg,wt,vwa,vwb,tdm", [
    ([2,2],[8,8],8,4,0), ([2,2],[8,8],4,4,1),
    ([2,2],[8,7],8,1,0), ([2,2],[8,7],4,1,1),
    ([4,1],[4,8],4,8,0), ([1,4],[8,4],8,4,0),
    ([4,1],[4,8],4,8,1), ([1,4],[8,4],8,4,1),
    ([4,1],[4,8],2,8,1), ([1,4],[8,4],8,2,1),
])
@pytest.mark.parametrize("du", [64,128])
@pytest.mark.parametrize("isolate", [False,True])
def test_paired_groups_preserve_membership_and_segment_separation(wg,wt,vwa,vwb,tdm,du,isolate):
    from copy import deepcopy
    from tensilelite.SolutionStructs.segment_interleave import paired_buffer_offsets, _footprint, _tensor_footprint
    s=_vw8_state(MIWaveGroup=wg, MIWaveTile=wt, VectorWidthA=vwa, VectorWidthB=vwb,
                 TDMSplit=tdm, DepthU=du, PrefetchGlobalRead=2, LDSSegmentInterleave=1,
                 MaxLDS=5*SEG)
    original=evaluate(s);assert original["applicable"]
    s["LDSSegInterleaveOffsets"]=original["offsets"]
    before=deepcopy(original["offsets"])
    o=paired_buffer_offsets(s,isolate)
    assert original["offsets"] == before
    if o is None:
        assert isolate and du==128  # complete isolated pairs exceed the five-segment budget
        return
    for key in ("aBaseline","bBaseline","portSplitA","portSplitB","componentSplit","activeTC"):
        assert o.get(key)==before.get(key)
    ranges=[];groups=[]
    for tc in ("A","B"):
        shared=o.get("aBaseline" if tc=="A" else "bBaseline",False)
        size=_tensor_footprint(s,tc) if shared else _footprint(s,tc)
        bufferStride=o["bufferStride"+tc]
        assert bufferStride % 256 == 0
        assert size <= bufferStride < size + 256
        segments=[]
        for c in range(1 if shared else 2):
            start=o["ldsBase"+tc]+c*o["writeStrideBytes"]
            assert start % 256 == 0
            pair_segments=set(range(start//SEG,(start+bufferStride+size-1)//SEG+1))
            segments.append(pair_segments);groups.append(pair_segments)
            for b in (0,1):ranges.append((start+b*bufferStride,start+b*bufferStride+size))
        if not shared:
            for buffer in (0, 1):
                start = o["ldsBase"+tc] + buffer*bufferStride
                end = start + size
                nextStart = start + o["writeStrideBytes"]
                occupied = set(range(start//SEG, (end-1)//SEG+1))
                nextOccupied = set(range(nextStart//SEG, (nextStart+size-1)//SEG+1))
                assert occupied.isdisjoint(nextOccupied)
            if isolate:
                assert segments[0].isdisjoint(segments[1])
    ranges.sort()
    assert all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:]))
    assert ranges[-1][1]==o["pairedUsedBytes"]<=s["MaxLDS"]
    if isolate:
        assert all(a.isdisjoint(b) for i,a in enumerate(groups) for b in groups[i+1:])


def test_paired_small_tile_retains_separation_and_read_window_gap():
    from tensilelite.SolutionStructs.segment_interleave import paired_buffer_offsets
    s=_vw8_state(DepthU=32,VectorWidthB=1,PrefetchGlobalRead=2,LDSSegmentInterleave=1,
                 MaxLDS=5*SEG,LdsPadA=0,LdsPadB=0)
    s["LDSSegInterleaveOffsets"]=evaluate(s)["offsets"]
    o=paired_buffer_offsets(s)
    assert o["writeStrideBytes"]==SEG  # tight pairing would put all four groups in segment 0
    assert o["readAddressWindowsB"]==[0,1]
    isolated=paired_buffer_offsets(s,True)
    assert isolated["writeStrideBytes"]==2*SEG
    assert isolated["readAddressWindowsB"]==[0,2]  # no unused window-1 register


def test_paired_scales_follow_both_data_buffers_and_budget_includes_them():
    from tensilelite.SolutionStructs.segment_interleave import paired_buffer_offsets
    s=_vw8_state(DepthU=64,VectorWidthB=1,PrefetchGlobalRead=2,LDSSegmentInterleave=1,
                 MaxLDS=5*SEG,ProblemType=dict(MXBlockA=32,MXBlockB=32),
                 LdsNumElementsAlignedMXSA=1024,LdsNumElementsAlignedMXSB=2048)
    s["LDSSegInterleaveOffsets"]=evaluate(s)["offsets"]
    o=paired_buffer_offsets(s)
    end=o["ldsBaseB"]+o["writeStrideBytes"]+2*o["bufferStrideB"]
    assert o["ldsBaseMXSA"]==end
    assert o["ldsBaseMXSB"]==end+2048
    assert o["pairedUsedBytes"]==end+6144
    s["MaxLDS"]=o["pairedUsedBytes"]-1
    assert paired_buffer_offsets(s) is None


@pytest.mark.parametrize("dtype", [
    _FakeDataType(), _FakeDataType(bf16=False, f8=True, nbytes=1),
    _FakeDataType(bf16=False, f4=True, nbytes=0.5),
])
def test_paired_alignment_adds_external_gaps_without_changing_tensor_padding(dtype):
    from copy import deepcopy
    from tensilelite.SolutionStructs.segment_interleave import _footprint, paired_buffer_layout_reason
    s = _vw8_state(DepthU=64, VectorWidthB=1, PrefetchGlobalRead=2,
                   LDSSegmentInterleave=1, MaxLDS=5*SEG, LdsOffsetA=16,
                   ProblemType=dict(DataType=dtype))
    s["LDSSegInterleaveOffsets"] = evaluate(s)["offsets"]
    before = deepcopy(s)
    o = paired_buffer_offsets(s)
    assert paired_buffer_layout_reason(s, o) is None
    assert o["ldsBaseA"] == 256
    for tc in ("A", "B"):
        size = _footprint(s, tc)
        assert size % 256 != 0
        assert o["bufferStride"+tc] > size
        assert s["LdsPad"+tc] == before["LdsPad"+tc]
        assert s["LdsBlockSizePerPad"+tc] == before["LdsBlockSizePerPad"+tc]
    assert s["LDSSegInterleaveOffsets"] == before["LDSSegInterleaveOffsets"]
    s["MaxLDS"] = o["pairedUsedBytes"]
    assert paired_buffer_offsets(s) == o
    s["MaxLDS"] -= 1
    assert paired_buffer_offsets(s) is None


def test_paired_mx_alignment_counts_gaps_but_not_unused_final_tail():
    from tensilelite.SolutionStructs.segment_interleave import _footprint, paired_buffer_layout_reason
    s = _tdmfuse_state(0, DepthU=128, VectorWidthB=1,
                       LDSSegmentInterleave=1, PrefetchGlobalRead=2, MaxLDS=5*SEG,
                       LdsNumElementsAlignedMXSA=1040, LdsNumElementsAlignedMXSB=2064)
    s["LDSSegInterleaveOffsets"] = evaluate(s)["offsets"]
    o = paired_buffer_offsets(s)
    assert paired_buffer_layout_reason(s, o) is None
    dataEnd = o["ldsBaseB"] + o["writeStrideBytes"] + o["bufferStrideB"] + _footprint(s, "B")
    assert 0 <= o["ldsBaseMXSA"] - dataEnd < 256
    assert o["bufferStrideMXSA"] == 1280
    assert o["bufferStrideMXSB"] == 2304
    scaleAEnd = o["ldsBaseMXSA"] + 1280 + 1040
    assert 0 <= o["ldsBaseMXSB"] - scaleAEnd < 256
    assert o["pairedUsedBytes"] == o["ldsBaseMXSB"] + 2304 + 2064
    s["MaxLDS"] = o["pairedUsedBytes"] - 1
    assert paired_buffer_offsets(s) is None


def test_paired_emitted_read_swap_preserves_lane_bank_phase():
    from types import SimpleNamespace
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    s = _vw8_state(DepthU=64, VectorWidthB=1, PrefetchGlobalRead=2,
                   LDSSegmentInterleave=1, MaxLDS=5*SEG,
                   StoreSwapAddr=True, LdsOffsetA_Blk=2*SEG)
    s["LDSSegInterleaveOffsets"] = evaluate(s)["offsets"]
    o = paired_buffer_offsets(s)
    s["LDSSegInterleaveOffsets"] = o
    writer = SimpleNamespace(_dcpDivergent=lambda kernel: False)
    for tc in ("A", "B"):
        mod = KernelWriterAssembly.lraSwapAddressesForDTLPad(writer, s, {"tensorChar": tc})
        for component in (0, 1):
            base = o["ldsBase"+tc] + component*o["writeStrideBytes"]
            for laneOffset in range(256):
                ping = base + laneOffset
                regs = {"v[vgprLocalReadAddr%s]" % tc: ping}
                for item in mod.flatitems():
                    op, dst, *srcs = str(item).split("//")[0].replace(",", " ").split()
                    values = [regs[src] if src in regs else int(src, 0)
                              for src in srcs if src != "vcc_lo"]
                    regs[dst] = (values[0] + values[1] if op == "v_add_u32"
                                 else values[0] ^ values[1])
                mask = regs["v[vgprLocalReadSwapAddr%s]" % tc]
                pong = ping ^ mask
                assert pong == ping + o["bufferStride"+tc]
                assert pong ^ mask == ping
                assert pong % 256 == ping % 256


def test_aligned_brotate_emits_the_original_tdm_buffer_swap(monkeypatch):
    from types import SimpleNamespace
    from tensilelite.Components.TensorDataMover import TensorDataMoverLoad
    from tensilelite.KernelWriterAssembly import KernelWriterAssembly
    state = _brot_state(StoreSwapAddr=False, LdsOffsetA_Blk=2*SEG)
    component = SimpleNamespace(getLdsAddrSgprName=lambda name: name)
    monkeypatch.setattr(TensorDataMoverLoad, "find", lambda writer: component)
    writer = SimpleNamespace(
        tdmDescriptorSetOwner=lambda kernel, tc: tc,
        _dcpDivergent=lambda kernel: False,
        states=SimpleNamespace(IncLdsBufSwitch=False),
    )
    for tc in ("A", "B"):
        module = KernelWriterAssembly.tdmSwapLdsOffset(writer, state, {"tensorChar": tc})
        text = str(module)
        assert "s_xor_b32" in text
        assert hex(2*SEG) in text
        assert "PairedTDMSwap" not in text


@pytest.mark.parametrize("override", [
    {"NumLdsBlk":3},{"1LDSBuffer":1},{"PrefetchGlobalRead":0},
    {"PrefetchGlobalReadA":2},{"TDMSplit":1},{"ExpandPointerSwap":True},
    {"DirectToLdsB":True},{"NumWaveSplitK":2},{"WavefrontSize":64},
])
def test_paired_emit_gate_excludes_unsupported_buffer_paths(override):
    from tensilelite.SolutionStructs.segment_interleave import paired_buffer_reason
    s=_vw8_state(LDSSegmentInterleave=1,NumLdsBlk=2,PrefetchGlobalRead=2,**{"1LDSBuffer":0})
    assert paired_buffer_reason(s) is None
    s.update(override)
    assert paired_buffer_reason(s) is not None


@pytest.mark.parametrize("tc", ["A", "B"])
def test_paired_gate_accepts_mixed_fp4_operands(tc):
    from tensilelite.SolutionStructs.segment_interleave import paired_buffer_reason
    s=_vw8_state(LDSSegmentInterleave=1,NumLdsBlk=2,PrefetchGlobalRead=2,**{"1LDSBuffer":0})
    s["ProblemType"]["MacDataType"+tc]=_FakeDataType(bf16=False,f4=True,nbytes=0.5)
    assert paired_buffer_reason(s) is None


def test_paired_split_aligned_allows_only_cross_phase_tail_sharing():
    from tensilelite.SolutionStructs.segment_interleave import paired_buffer_offsets, paired_buffer_layout_reason
    s = _vw8_state(MIWaveTile=[4,4], VectorWidthA=4, VectorWidthB=1, DepthU=256,
                   ProblemType=dict(DataType=_FakeDataType(bf16=False, f8=True, nbytes=1)),
                   LdsPadA=16, LdsPadB=16, LdsBlockSizePerPadA=1024, LdsBlockSizePerPadB=256,
                   PrefetchGlobalRead=2, LDSSegmentInterleave=1, MaxLDS=5*SEG)
    s["LDSSegInterleaveOffsets"] = evaluate(s)["offsets"]
    o = paired_buffer_offsets(s)
    assert paired_buffer_layout_reason(s, o) is None
    # B0 pong and B1 ping may both touch S1. B1 pong must clear B0 pong
    # completely, so same-phase footprints stay separate without isolating pairs.
    assert o["bufferStrideA"] == 16640
    assert o["bufferStrideB"] == 17408
    assert o["writeStrideBytes"] == 80384
    assert o["pairedUsedBytes"] == 148480
    assert o["readAddressWindowsB"] == [0,1]
    assert (o["ldsBaseB"] + 2*17408 - 1)//SEG == (o["ldsBaseB"] + o["writeStrideBytes"])//SEG == 1
    assert (o["ldsBaseB"] + o["writeStrideBytes"] + 17408)//SEG == 2


@pytest.mark.parametrize("wg,wt,vwa,vwb,active", [
    ([4,1],[4,4],4,4,"A"), ([1,4],[4,4],4,4,"B"),
])
def test_paired_shared_operand_uses_same_range_policy_for_a_and_b(wg,wt,vwa,vwb,active):
    from tensilelite.SolutionStructs.segment_interleave import paired_buffer_offsets, paired_buffer_layout_reason
    s = _vw8_state(MIWaveGroup=wg, MIWaveTile=wt, VectorWidthA=vwa, VectorWidthB=vwb,
                   DepthU=256, LdsPadA=16, LdsPadB=16,
                   LdsBlockSizePerPadA=1024, LdsBlockSizePerPadB=1024,
                   ProblemType=dict(DataType=_FakeDataType(bf16=False, f8=True, nbytes=1)),
                   PrefetchGlobalRead=2, LDSSegmentInterleave=1, MaxLDS=5*SEG)
    s["LDSSegInterleaveOffsets"] = evaluate(s)["offsets"]
    o = paired_buffer_offsets(s)
    assert paired_buffer_layout_reason(s, o) is None
    assert o["bufferStride"+active] == 33280
    assert o["writeStrideBytes"] == 99840
    # Active pair 0 spans S0/S1, pair 1 spans S1/S2. Each matching
    # ping/pong footprint is disjoint; complete pair ranges may share a segment.
    assert o["pairedUsedBytes"] == 166400
    assert o["ldsBase"+active] == 0


@pytest.mark.parametrize("wt,vwa,expected_stride", [([10,2],10,80*1024), ([2,10],2,72*1024)])
def test_paired_checks_footprint_end_and_uses_minimum_gap(wt,vwa,expected_stride):
    from tensilelite.SolutionStructs.segment_interleave import paired_buffer_offsets, paired_buffer_layout_reason
    s = _vw8_state(MIWaveTile=wt, VectorWidthA=vwa, VectorWidthB=1, DepthU=64,
                   LdsOffsetA=48*1024, LdsPadA=0, LdsPadB=0,
                   PrefetchGlobalRead=2, LDSSegmentInterleave=1, MaxLDS=5*SEG)
    s["LDSSegInterleaveOffsets"] = evaluate(s)["offsets"]
    o = paired_buffer_offsets(s)
    assert paired_buffer_layout_reason(s, o) is None
    # Natural stride is 48 KiB. A/B ping footprints respectively cross a
    # segment boundary and require more space; do not round stride gratuitously.
    assert o["writeStrideBytes"] == expected_stride
    assert any((o["ldsBase"+tc]+(buffer+1)*o["bufferStride"+tc]-1)//SEG ==
               (o["ldsBase"+tc]+buffer*o["bufferStride"+tc]+expected_stride-16)//SEG
               for tc in ("A","B") for buffer in (0,1))



@pytest.mark.parametrize("padding, expected_stride", [(0, 80*1024), (16, 95*1024 + 512)])
@pytest.mark.parametrize("mx", [False, True])
def test_paired_pong_footprint_clears_last_segment_and_checks_budget(padding, expected_stride, mx):
    from tensilelite.SolutionStructs.segment_interleave import paired_buffer_offsets, paired_buffer_layout_reason
    s = _vw8_state(MIWaveTile=[8,2], VectorWidthA=8, VectorWidthB=1, DepthU=256,
                   ProblemType=dict(DataType=_FakeDataType(bf16=False, f8=True, nbytes=1),
                                    MXBlockA=32 if mx else 0, MXBlockB=32 if mx else 0),
                   LdsPadA=padding, LdsPadB=padding,
                   LdsBlockSizePerPadA=1024, LdsBlockSizePerPadB=256,
                   LdsNumElementsAlignedMXSA=1024, LdsNumElementsAlignedMXSB=2048,
                   PrefetchGlobalRead=2, LDSSegmentInterleave=1, MaxLDS=5*SEG)
    s["LDSSegInterleaveOffsets"] = evaluate(s)["offsets"]
    o = paired_buffer_offsets(s)
    assert paired_buffer_layout_reason(s, o) is None
    assert o["writeStrideBytes"] == expected_stride
    size = o["bufferStrideA"]
    if padding:
        # A0 pong occupies [32.5, 65) KiB. Start-only separation used an 82 KiB
        # stride and put A1 pong at 114.5 KiB in the same segment as A0's tail.
        assert size == 33280
        assert (2*size-1)//SEG == (size+82*1024)//SEG == 1
        assert size + expected_stride == 2*SEG
    else:
        # A0 pong ends exactly at 64 KiB; that boundary remains reusable.
        assert 2*size == SEG
        assert size + expected_stride < 2*SEG
    for tc in ("A", "B"):
        size = o["bufferStride"+tc]
        for buffer in (0, 1):
            first = o["ldsBase"+tc] + buffer*size
            second = first + o["writeStrideBytes"]
            firstSegments = set(range(first//SEG, (first+size-1)//SEG+1))
            secondSegments = set(range(second//SEG, (second+size-1)//SEG+1))
            assert firstSegments.isdisjoint(secondSegments)
    dataEnd = o["ldsBaseB"] + o["writeStrideBytes"] + 2*o["bufferStrideB"]
    if mx:
        assert o["ldsBaseMXSA"] == dataEnd
        assert o["ldsBaseMXSB"] == dataEnd + 2048
        assert o["pairedUsedBytes"] == dataEnd + 6144
    s["MaxLDS"] = o["pairedUsedBytes"]
    assert paired_buffer_offsets(s) == o
    s["MaxLDS"] -= 16
    assert paired_buffer_offsets(s) is None


def _paired_validation_state():
    from tensilelite.SolutionStructs.segment_interleave import paired_buffer_offsets
    s = _vw8_state(DepthU=64, VectorWidthB=1, PrefetchGlobalRead=2,
                   LDSSegmentInterleave=1, MaxLDS=5*SEG,
                   ProblemType=dict(MXBlockA=32,MXBlockB=32),
                   LdsNumElementsAlignedMXSA=1024,LdsNumElementsAlignedMXSB=2048)
    s["LDSSegInterleaveOffsets"] = evaluate(s)["offsets"]
    return s, paired_buffer_offsets(s)


@pytest.mark.parametrize("key", ["ldsBaseA", "ldsBaseB", "writeStrideBytes",
    "bufferStrideA", "bufferStrideB", "ldsBaseMXSA", "ldsBaseMXSB",
    "bufferStrideMXSA", "bufferStrideMXSB"])
@pytest.mark.parametrize("delta", [8, 16])
def test_paired_validator_checks_all_copy_starts_for_256_byte_alignment(key, delta):
    from tensilelite.SolutionStructs.segment_interleave import paired_buffer_layout_reason
    s, o = _paired_validation_state()
    assert paired_buffer_layout_reason(s, o) is None
    o[key] += delta
    assert "256-byte-aligned" in paired_buffer_layout_reason(s, o)


def test_paired_validator_rejects_overlap_and_budget_overflow():
    from tensilelite.SolutionStructs.segment_interleave import paired_buffer_layout_reason
    s, o = _paired_validation_state()
    for bad in ({**o, "ldsBaseB": o["ldsBaseA"]},
                {**o, "ldsBaseMXSA": o["ldsBaseB"]},
                {**o, "bufferStrideA": o["bufferStrideA"]-256}):
        assert "overlaps" in paired_buffer_layout_reason(s, bad)
    s["MaxLDS"] = o["pairedUsedBytes"]-1
    assert "MaxLDS" in paired_buffer_layout_reason(s, o)


def test_paired_validator_rejects_a_footprint_beyond_the_declared_allocation():
    from tensilelite.SolutionStructs.segment_interleave import paired_buffer_layout_reason
    s, o = _paired_validation_state()
    o["pairedUsedBytes"] -= 16
    assert "pairedUsedBytes" in paired_buffer_layout_reason(s, o)


def test_brotate_paired_layout_has_the_same_alignment_and_overlap_checks():
    from tensilelite.SolutionStructs.segment_interleave import paired_buffer_layout_reason
    s = _brot_state(MaxLDS=5*SEG)
    o = paired_buffer_offsets(s)
    assert paired_buffer_layout_reason(s, o) is None
    o["bufferStrideB"] += 8
    assert "256-byte-aligned" in paired_buffer_layout_reason(s, o)


@pytest.mark.parametrize("layout", sorted(_TDMFUSE_LAYOUTS))
@pytest.mark.parametrize("base", [0, 16])
def test_paired_tdm_piece_bases_match_relocated_local_reads(layout, base):
    from tensilelite.SolutionStructs.segment_interleave import paired_buffer_offsets, paired_buffer_layout_reason
    from tensilelite.Components.TDMFuse import tdmWavePartition
    s = _tdmfuse_state(0, LDSSegmentInterleave=1, PrefetchGlobalRead=2,
                       MaxLDS=5*SEG, LdsOffsetA=base,
                       LdsNumElementsAlignedMXSA=1024, LdsNumElementsAlignedMXSB=1024,
                       **_TDMFUSE_LAYOUTS[layout])
    if layout == "tight_f8f4":
        s["DepthU"] = 256  # Keep both data copies and scales within MaxLDS.
    s["LDSSegInterleaveOffsets"] = evaluate(s)["offsets"]
    o = paired_buffer_offsets(s)
    assert paired_buffer_layout_reason(s, o) is None
    s["LDSSegInterleaveOffsets"] = o
    for tc in ("A", "B"):
        components = tdmWavePartition(s, tc)[0]
        rows = _tc_mt(s, tc) // components
        # Execute actual emitted TDM scalar arithmetic, then compare with the
        # independent padded logical-row mapping used by the read-side oracle.
        assert _piece_lds_addrs(s, tc) == [_read_addr(s, o, tc, c*rows) for c in range(components)]
