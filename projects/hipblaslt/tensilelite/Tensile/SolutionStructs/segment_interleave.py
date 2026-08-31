# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx1250 LDS segment-conflict interleave oracle.

Pure function of `state`: decides how to place an operand's components across LDS segments, and
returns the byte offsets the code generator uses to lay the data into LDS and read it back.

Background: LDS is partitioned into 5 x 64KiB segments, read through two ports (one per SIMD pair).
Two ports reading the same segment in a cycle is a segment conflict.
Each operand is loaded into LDS in two parts called components; putting the two components in
different segments keeps the ports apart and avoids the conflict. Baseline instead
packs each operand's two components adjacently ([A0][A1][B0][B1]); this oracle repacks them, and how
depends on MIWaveGroup.

[2,2] (both operands span two waves): separate A across the two segments so its two read ports read
different segments; B is repacked to match.
  Without TDMSplit (VWA == WaveTileA only): A's two components go to different segments
      ([A0][B0][A1][B1]).
  With TDMSplit, the vector width picks the placement:
    VWA == WaveTileA:   the same layout, A's two components in different segments.
    VWA == WaveTileA/2: portSplitA; each component is two halves, split by half instead (half 0 of
                        both components in one segment, half 1 in the other).
  For an odd WaveTileB a split B read would cross a component, so B is kept whole ([A0][B0][B1][A1],
  bcontig).

[4,1]/[1,4] (only one operand spans the four waves, the "active" one; the other is "shared"): put
only the active operand's two components in different segments; the shared operand keeps its baseline
layout.
  Without TDMSplit (VW == WaveTile only): the two components are split by wave-pair, which halves the
      conflict.
  With TDMSplit, the vector width picks the placement:
    VW == WaveTile:   portSplitA/portSplitB puts even waves in one segment and odd in the other, so
                      the two read ports never share a segment (conflict removed).
    VW == WaveTile/2: componentSplit puts wave-pair {W0,W1} in one segment and {W2,W3} in the other;
                      each segment still holds one even and one odd wave, so the conflict is only
                      halved, but with no extra loads (splitting by port here would cost a second
                      load).

Every layout is either tight (no extra LDS) or aligned (padded to a segment boundary, which uses more
LDS and needs PrefetchGlobalRead=2).
"""

# gfx1250 LDS segment size (5 x 64 KiB segments).
SEG = 65536

def _bpe(state, tc):
    # Float, not int -- fp4 is 0.5 B/elem; callers int() the byte counts.
    pt = state["ProblemType"]
    return pt.get("MacDataType%s" % tc, pt["DataType"]).numBytes()

def _pad(x, blk, padElems, bpe):
    if blk == 0 or padElems == 0:
        return 0
    return int((x // blk) * padElems * bpe)

def _data_bytes(state, tc):
    numComp = state["NumWaves"] // 2
    mt = state["MacroTile0"] if tc == "A" else state["MacroTile1"]
    return int((mt // numComp) * state["DepthU"] * _bpe(state, tc))

def _footprint(state, tc):
    d = _data_bytes(state, tc)
    blk = state["LdsBlockSizePerPad%s" % tc]
    padElems = state["LdsPad%s" % tc]
    return d + _pad(d, blk, padElems, _bpe(state, tc))

def _mx_scale_bases(state, mxsaStart):
    """MX scale-block LDS bases, placed after the interleaved A/B region. Returns
    (ldsBaseMXSA, ldsBaseMXSB, end); a base is None when that scale is not LDS-resident."""
    pt = state["ProblemType"]
    hasA = bool(pt.get("MXBlockA")) and not state.get("DirectToVgprMXSA")
    hasB = bool(pt.get("MXBlockB")) and not state.get("DirectToVgprMXSB")
    if not (hasA or hasB):
        return None, None, mxsaStart
    baseA = mxsaStart
    szA = int(state.get("LdsNumElementsAlignedMXSA", 0)) if hasA else 0
    baseB = baseA + szA
    szB = int(state.get("LdsNumElementsAlignedMXSB", 0)) if hasB else 0
    return (baseA if hasA else None), (baseB if hasB else None), baseB + szB

def _wactive(state, tc):
    return state["MIWaveGroup"][0] if tc == "A" else state["MIWaveGroup"][1]

def _coarse(state, tc):
    # coarse = VW == WaveTile (one read fits one segment). W = waves on this tensor's dim
    # (MIWaveGroup[dim]); for [2,2] W == 2.
    W = _wactive(state, tc)
    mt = state["MacroTile0"] if tc == "A" else state["MacroTile1"]
    vw = state["VectorWidthA"] if tc == "A" else state["VectorWidthB"]
    mi_threads = min(state["MatrixInstM"], state["MatrixInstN"])
    return W > 0 and mi_threads * vw >= mt // W

def _port_split(state, tc):
    # VW==WaveTile/2 (2 vIdx per port) with TDMSplit. A smaller VW would need a >2-way split.
    if _coarse(state, tc) or not state.get("TDMSplit"):
        return False
    idx = 0 if tc == "A" else 1
    vw = state["VectorWidthA"] if tc == "A" else state["VectorWidthB"]
    wt = state["MIWaveTile"][idx]
    return vw > 0 and wt % vw == 0 and wt // vw == 2

def _port_split_a(state):
    return _port_split(state, "A")

def _b_readable(state):
    # True if B can be split across segments and still read correctly: either B covers a full
    # component (coarse), or its per-vIdx column span (vIdxColsB) divides compColsB evenly so no
    # single ds_load straddles the component boundary. WaveTileB=7 -> 112 % 32 != 0 -> False.
    numComp = state["NumWaves"] // 2
    mi_threads = min(state["MatrixInstM"], state["MatrixInstN"])
    if mi_threads * state["VectorWidthB"] >= state["MacroTile1"] // numComp:
        return True
    compColsB = state["MacroTile1"] // numComp
    vIdxColsB = state["MatrixInstN"] * state.get("MatrixInstBN", 1) * state["MIWaveGroup"][1] * state["VectorWidthB"]
    return vIdxColsB > 0 and compColsB % vIdxColsB == 0

def _no(reason):
    return {"applicable": False, "aligned": False, "offsets": None,
            "blockSpan": 0, "reason": reason, "segmentMap": ""}

def _ceil_seg(x):
    return ((x + SEG - 1) // SEG) * SEG

def aligned_budget_ok(blockSpan, numLdsBlk, naturalOffsetBlk, maxLDS):
    """Return (ok, per-buffer block) for the aligned branch: the block is the next power
    of two >= max(naturalOffsetBlk, blockSpan), valid only if double-buffering it fits MaxLDS."""
    if numLdsBlk != 2:
        return (False, None)
    offsetBlk = max(naturalOffsetBlk, blockSpan)
    if offsetBlk <= 0:
        return (False, None)
    roundup = 1 << (offsetBlk - 1).bit_length()   # next power of two
    if roundup * 2 > maxLDS:
        return (False, None)                      # total = roundup + blockSpan <= roundup*2
    return (True, roundup)

def _evaluate_asymmetric(state):
    # [4,1]/[1,4]: active tensor = the dim>1 one (A for [4,1], B for [1,4]); the other is shared.
    # split the active comps across segments; leave the shared one baseline (aBaseline/bBaseline).
    # numComp stays 2, so the load count is unchanged.
    activeTC = "A" if state["MIWaveGroup"][0] > 1 else "B"
    sharedTC = "B" if activeTC == "A" else "A"

    fAct     = _footprint(state, activeTC)
    fActData = _data_bytes(state, activeTC)
    fSh      = _footprint(state, sharedTC)
    base     = state["LdsOffsetA"]
    baselineKey = "bBaseline" if activeTC == "A" else "aBaseline"
    sharedBaseKey = "ldsBaseB" if activeTC == "A" else "ldsBaseA"

    def _build_offsets(stride):
        # active comp0 @ base, shared @ base+fAct; emit sites read ldsBase<tc>.
        o = {sharedBaseKey: base + fAct, "ldsBase%s" % activeTC: base,
             "writeStrideBytes": stride, "footprintPacked": True,
             baselineKey: True, "activeTC": activeTC}
        bMXSA, bMXSB, _ = _mx_scale_bases(state, base + 2 * fAct + 2 * fSh)
        if bMXSA is not None: o["ldsBaseMXSA"] = bMXSA
        if bMXSB is not None: o["ldsBaseMXSB"] = bMXSB
        return o

    if state.get("TDMSplit"):
        if not (_coarse(state, activeTC) or _port_split(state, activeTC)):
            return _no("TDMSplit asymmetric: active VW must be WaveTile or WaveTile/2")
        # VW==WaveTile splits by read port; VW==WaveTile/2 can't (needs >2-way) so it splits by component.
        useCompAxis = _port_split(state, activeTC) and not _coarse(state, activeTC)
        splitKey = "componentSplit" if useCompAxis else ("portSplitA" if activeTC == "A" else "portSplitB")
        strideAct = fAct + 2 * fSh
        c0end = (base + fActData - 1) // SEG
        c1 = (base + strideAct) // SEG
        if c1 > c0end:
            # shared block already separates the two ports into different segments; free.
            o = _build_offsets(strideAct)
            o[splitKey] = True
            return {"applicable": True, "aligned": False, "offsets": o,
                    "blockSpan": 0, "reason": "compaxis-asym" if useCompAxis else "portaxis-asym",
                    "segmentMap": "%s active=%s seg%d/seg%d" % ("COMPAXIS" if useCompAxis else "PORTAXIS",
                                  activeTC, base // SEG, (base + strideAct) // SEG)}
        # port1 still shares port0's segment: pad it to the next segment boundary. Grows LDS.
        if state.get("PrefetchGlobalRead") != 2:        return _no(("compaxis-asym" if useCompAxis else "portaxis-asym") + ": both components fit one segment, aligned layout (LDS grows) needs PGR=2")
        if state.get("LDSSegmentInterleave", -1) == -1: return _no("auto: skip aligned (LDS growth)")
        pre = _ceil_seg(base + strideAct) - base
        o = _build_offsets(pre)
        o[splitKey] = True
        blockSpan = base + pre + fAct
        bMXSA, bMXSB, mxEnd = _mx_scale_bases(state, blockSpan)
        if bMXSA is not None: o["ldsBaseMXSA"] = bMXSA
        if bMXSB is not None: o["ldsBaseMXSB"] = bMXSB
        blockSpan = max(blockSpan, mxEnd)
        return {"applicable": True, "aligned": True, "offsets": o,
                "blockSpan": blockSpan,
                "reason": "compaxis-asym-aligned" if useCompAxis else "portaxis-asym-aligned",
                "segmentMap": "%s-ALIGNED active=%s seg%d/seg%d"
                              % ("COMPAXIS" if useCompAxis else "PORTAXIS", activeTC, base // SEG, (base + pre) // SEG)}

    # active must be coarse (VW == WaveTile).
    if not _coarse(state, activeTC):
        return _no("%s active: VW must be WaveTile (coarse)" % activeTC)

    # A active only (baseline puts A at base, B after A+scales -> [1,4] can't shortcut): baseline
    # already splits A0/A1 when comp0 data fits one segment and comp1 lands in the next.
    if activeTC == "A" and (base % SEG) + fActData <= SEG and (base + fAct) // SEG != base // SEG:
        return _no("baseline already separates A0/A1 into different segments "
                   "(A active; comp0 data within one segment)")

    # put the shared tensor between the two active comps. bcontig if that stride already crosses a
    # segment; else aligned (pad to the boundary).
    #   A active: [A0][B][A1] bBaseline    B active: [B0][A][B1] aBaseline
    strideAct = fAct + 2 * fSh              # comp0 -> comp1 gap

    c0 = base // SEG
    # comp0 can span 2 segments (unaligned base); comp1 must start past its last one.
    c0end = (base + fActData - 1) // SEG
    c1 = (base + strideAct) // SEG
    if c1 > c0end:
        # shared block already pushed comp1 to a new segment; free.
        return {"applicable": True, "aligned": False, "offsets": _build_offsets(strideAct),
                "blockSpan": 0, "reason": "bcontig-asym",
                "segmentMap": "BCONTIG-ASYM active=%s seg%d={c0,shared} seg%d={c1}" % (activeTC, c0, c1)}

    # smaller: pad comp1 to the next segment. grows LDS -> PGR2 + force only.
    if state.get("PrefetchGlobalRead") != 2:   return _no("bcontig-asym: both components fit one segment, aligned layout (LDS grows) needs PGR=2")
    if state.get("LDSSegmentInterleave", -1) == -1: return _no("auto: skip aligned (LDS growth)")
    pre = _ceil_seg(base + strideAct) - base
    offsets = _build_offsets(pre)
    blockSpan = base + pre + fAct
    bMXSA, bMXSB, mxEnd = _mx_scale_bases(state, blockSpan)
    if bMXSA is not None: offsets["ldsBaseMXSA"] = bMXSA
    if bMXSB is not None: offsets["ldsBaseMXSB"] = bMXSB
    blockSpan = max(blockSpan, mxEnd)
    return {"applicable": True, "aligned": True, "offsets": offsets,
            "blockSpan": blockSpan, "reason": "bcontig-asym-aligned",
            "segmentMap": "BCONTIG-ASYM-ALIGNED active=%s seg%d/seg%d"
                          % (activeTC, base // SEG, (base + pre) // SEG)}


def evaluate(state):
    pt = state["ProblemType"]
    # Tri-state knob: -1 = auto (default), 0 = force baseline, 1 = force on where applicable.
    # Auto takes only the no-trade-off tight branch; the LDS-growing aligned branch needs 1.
    mode = state.get("LDSSegmentInterleave", -1)
    if mode == 0:                                              return _no("parameter off")
    if tuple(state.get("ISA", ()))[:2] != (12, 5):             return _no("not gfx1250")
    if not (state.get("enableTDMA") and state.get("enableTDMB") and state["NumWaves"] > 1):
        return _no("not wave-separated TDM")
    if state.get("LocalSplitU", 1) > 1:
        return _no("LocalSplitU>1")
    if not state.get("UnrollMajorLDSA") or not state.get("UnrollMajorLDSB"):
        return _no("not unrollMajor")
    # DirectToVgpr operands are not in LDS, so segment placement does not apply.
    if state.get("DirectToVgprA") or state.get("DirectToVgprB"):
        return _no("DirectToVgpr operand not LDS-resident")
    # numComp==2 (NumWaves==4) restricts MIWaveGroup to {[2,2],[4,1],[1,4]}, all with even waves
    # per active dim. [2,2] interleaves both tensors; [4,1]/[1,4] have one active + one shared
    # tensor and are handled by _evaluate_asymmetric below.
    if state["NumWaves"] // 2 != 2:                             return _no("numComp!=2")
    if pt.get("Sparse"):
        return _no("sparse")
    # Subtile uses a separate codegen body; the emit path these offsets target runs only for
    # non-subtile kernels.
    if state.get("UseSubtileImpl"):                            return _no("subtile")
    # Needs double-buffering; 1LDSBuffer==1 breaks the assumed layout. Unresolved -1 is rejected too
    # (Solution.py resolves it later, then re-evaluates).
    if state.get("1LDSBuffer", 0) != 0:                         return _no("needs 1LDSBuffer==0")
    _dt = pt["DataType"]
    # fp8/fp4 cover mxf8/mxf4; MX scales are relocated as a trailing block (see _mx_scale_bases).
    if not (_dt.isBFloat16() or _dt.isHalf() or _dt.is8bitFloat() or _dt.isFloat4()):
        return _no("bf16/fp16/fp8/fp4 only")

    # [4,1]/[1,4]: exactly one MIWaveGroup dim is 1 -> one active + one shared tensor.
    wgM, wgN = state["MIWaveGroup"][0], state["MIWaveGroup"][1]
    if (wgM == 1) ^ (wgN == 1):
        return _evaluate_asymmetric(state)
    if [wgM, wgN] != [2, 2]:
        return _no("MIWaveGroup unsupported")

    # [2,2]: A must be coarse (VWA==WaveTileA) or port-split (VWA==WaveTileA/2, needs TDMSplit).
    _portSplit = _port_split_a(state)
    if not (_coarse(state, "A") or _portSplit):               return _no("A: VWA must be WaveTileA, or WaveTileA/2 with TDMSplit")

    fA, fB = _footprint(state, "A"), _footprint(state, "B")
    base = state["LdsOffsetA"]

    # bcontig fallback [A0][B0][B1][A1] (auto-only, not user-forceable): when B can't be split
    # (odd WaveTileB), keep B whole and use it as the gap that pushes A1 into the next segment.
    if not _b_readable(state):
        strideA = fA + 2 * fB                       # distance A0 -> A1: skip A0 and the whole B block
        a0 = base // SEG
        a1 = (base + strideA) // SEG
        if a1 != a0:
            # The B block already pushes A1 into the next segment, so this uses no extra LDS.
            offsets = {
                "ldsBaseB":         base + fA,      # B starts right after A0
                "writeStrideBytes": strideA,        # A0 -> A1 distance (pad already included)
                "footprintPacked":  True,
                "bBaseline":        True,           # B uses its normal (non-interleaved) addressing
            }
            if _portSplit:
                offsets["portSplitA"] = True
            # mxf8: put the scale block after A1 (bf16/fp16 have no scales).
            bMXSA, bMXSB, _ = _mx_scale_bases(state, base + 2 * fA + 2 * fB)
            if bMXSA is not None: offsets["ldsBaseMXSA"] = bMXSA
            if bMXSB is not None: offsets["ldsBaseMXSB"] = bMXSB
            return {"applicable": True, "aligned": False, "offsets": offsets,
                    "blockSpan": 0, "reason": "bcontig",
                    "segmentMap": "BCONTIG seg%d={A0,B0,B1} seg%d={A1}" % (a0, a1)}

        # Small tile: A0+B0+B1 all fit in one segment, so A1 would stay with A0. Pad the A0 -> A1
        # distance up to the next segment boundary so A1 lands in a different segment. Uses more LDS
        # (checked in Solution.py) and needs PGR=2 -- same idea as the split branch below.
        if state.get("PrefetchGlobalRead") != 2:   return _no("bcontig: A0+B0+B1 fit one segment, aligned layout (LDS grows) needs PGR=2")
        if mode == -1:                             return _no("auto: skip aligned (LDS growth)")
        pre = _ceil_seg(base + strideA) - base      # round A0 -> A1 distance up to a segment boundary
        offsets = {
            "ldsBaseB":         base + fA,          # B starts right after A0
            "writeStrideBytes": pre,                # A0 -> A1 distance (rounded to a segment)
            "footprintPacked":  True,
            "bBaseline":        True,
        }
        if _portSplit:
            offsets["portSplitA"] = True
        blockSpan = base + pre + fA                 # A1 ends here (past the B block, with a gap)
        bMXSA, bMXSB, mxEnd = _mx_scale_bases(state, blockSpan)
        if bMXSA is not None: offsets["ldsBaseMXSA"] = bMXSA
        if bMXSB is not None: offsets["ldsBaseMXSB"] = bMXSB
        blockSpan = max(blockSpan, mxEnd)
        return {"applicable": True, "aligned": True, "offsets": offsets,
                "blockSpan": blockSpan, "reason": "bcontig-aligned",
                "segmentMap": "BCONTIG-ALIGNED seg%d={A0,B0,B1} seg%d={A1}"
                              % (base // SEG, (base + pre) // SEG)}

    if (base % SEG) + fA + fB < SEG:
        # Small MacroTile: A0,B0 fit one segment, so push component 1 to the next segment boundary
        # with a segment-aligned stride. Grows LDS (Solution.py budget-checks); PGR2 double-buffer only.
        if state.get("PrefetchGlobalRead") != 2:        return _no("aligned: A0+B0 fit one segment, aligned layout (LDS grows) needs PGR=2")
        if mode == -1:                                  return _no("auto: skip aligned (LDS growth)")
        pre = _ceil_seg(base + fA + fB) - base          # segment-aligned stride (== SEG for base<SEG)
        offsets = {
            "ldsBaseB":         base + fA,              # B0 right after A0 in seg0
            "writeStrideBytes": pre,                    # segment stride; no re-pad on the jump
            "footprintPacked":  True,
        }
        if _portSplit:
            offsets["portSplitA"] = True
        # Per-buffer span: B1 ends at base + pre(=A1) + fA + fB.
        blockSpan = base + pre + fA + fB
        # mxf8: put the scale block after B1. Needs extra LDS, so extend the size below.
        bMXSA, bMXSB, mxEnd = _mx_scale_bases(state, blockSpan)
        if bMXSA is not None: offsets["ldsBaseMXSA"] = bMXSA
        if bMXSB is not None: offsets["ldsBaseMXSB"] = bMXSB
        blockSpan = max(blockSpan, mxEnd)
        return {"applicable": True, "aligned": True, "offsets": offsets,
                "blockSpan": blockSpan, "reason": "aligned",
                "segmentMap": "ALIGNED seg%d={A0,B0} seg%d={A1,B1}"
                              % (base // SEG, (base + pre) // SEG)}

    # Tight: pack [A0][B0][A1][B1] with component stride fA+fB (each footprint already includes its
    # pad, so the jump is not re-padded -> A1/B1 land exactly at the previous tile's end). No LDS growth.
    offsets = {
        "ldsBaseB":         base + fA,          # B0 right after A0
        "writeStrideBytes": fA + fB,            # footprint stride (post-pad), no re-pad on the jump
        "footprintPacked":  True,
    }
    if _portSplit:
        offsets["portSplitA"] = True
    # mxf8: put the scale block after B1. Uses no more LDS than the non-interleaved layout.
    bMXSA, bMXSB, _ = _mx_scale_bases(state, base + 2 * (fA + fB))
    if bMXSA is not None: offsets["ldsBaseMXSA"] = bMXSA
    if bMXSB is not None: offsets["ldsBaseMXSB"] = bMXSB
    a0 = base // SEG
    a1 = (base + fA + fB) // SEG                 # tight branch guarantees a1 > a0
    seg_map = "TIGHT seg%d={A0,B0} seg%d={A1,B1}" % (a0, a1)
    return {"applicable": True, "aligned": False, "offsets": offsets,
            "blockSpan": 0, "reason": "tight", "segmentMap": seg_map}
