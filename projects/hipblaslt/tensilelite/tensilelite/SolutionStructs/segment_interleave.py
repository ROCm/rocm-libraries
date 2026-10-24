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

The diagrams below describe evaluate()'s logical grouping for one buffer. A0/A1 and B0/B1 are
data components, not ping/pong buffer numbers.

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
LDS and needs double-buffered PGR1/PGR2; aligned TDMSplit needs PGR2).

For non-TDMSplit tight layouts, paired_buffer_offsets() places each group's ping/pong copies next to each other:
  split A/B:    [A0 pair][B0 pair][gap if needed][A1 pair][B1 pair]
  contiguous B: [A0 pair][B pair] [gap if needed][A1 pair]
  contiguous A: [B0 pair][A pair] [gap if needed][B1 pair]
Each pair is [ping][pong]. Corresponding component footprints, including padding, occupy disjoint
segments within each buffer phase: A0 ping versus A1 ping, A0 pong versus A1 pong, likewise for split B.
Different operands or buffer phases may share segments. LDS-resident MX scale pairs follow the data.
Aligned layouts retain their one-buffer grouping and separate ping/pong blocks.
Solution.py requires a valid paired layout for non-TDMSplit tight LDSSegmentInterleave=1. Automatic (-1)
uses paired only when supported and it needs no more LDS than LDSSegmentInterleave=0; otherwise it
uses ordinary placement. BRotate supports aligned and paired tight layouts and changes B's read order only.
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

def _tensor_footprint(state, tc):
    """Full baseline span, whose padding phase does not reset per component."""
    mt = state["MacroTile0"] if tc == "A" else state["MacroTile1"]
    d = int(mt * state["DepthU"] * _bpe(state, tc))
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

def _align_paired(x):
    # Preserve the 256-byte LDS bank phase when moving between copies.
    return ((x + 255) // 256) * 256

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
    sharedSpan = _tensor_footprint(state, sharedTC)
    base     = state["LdsOffsetA"]
    baselineKey = "bBaseline" if activeTC == "A" else "aBaseline"
    sharedBaseKey = "ldsBaseB" if activeTC == "A" else "ldsBaseA"

    def _build_offsets(stride):
        # active comp0 @ base, shared @ base+fAct; emit sites read ldsBase<tc>.
        o = {sharedBaseKey: base + fAct, "ldsBase%s" % activeTC: base,
             "writeStrideBytes": stride, "footprintPacked": True,
             baselineKey: True, "activeTC": activeTC}
        bMXSA, bMXSB, _ = _mx_scale_bases(state, base + 2 * fAct + sharedSpan)
        if bMXSA is not None: o["ldsBaseMXSA"] = bMXSA
        if bMXSB is not None: o["ldsBaseMXSB"] = bMXSB
        return o

    if state.get("TDMSplit"):
        if not (_coarse(state, activeTC) or _port_split(state, activeTC)):
            return _no("TDMSplit asymmetric: active VW must be WaveTile or WaveTile/2")
        # VW==WaveTile splits by read port; VW==WaveTile/2 can't (needs >2-way) so it splits by component.
        useCompAxis = _port_split(state, activeTC) and not _coarse(state, activeTC)
        splitKey = "componentSplit" if useCompAxis else ("portSplitA" if activeTC == "A" else "portSplitB")
        strideAct = fAct + sharedSpan
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
    strideAct = fAct + sharedSpan            # comp0 -> comp1 gap

    c0 = base // SEG
    # comp0 can span 2 segments (unaligned base); comp1 must start past its last one.
    c0end = (base + fActData - 1) // SEG
    c1 = (base + strideAct) // SEG
    if c1 > c0end:
        # shared block already pushed comp1 to a new segment; free.
        return {"applicable": True, "aligned": False, "offsets": _build_offsets(strideAct),
                "blockSpan": 0, "reason": "bcontig-asym",
                "segmentMap": "BCONTIG-ASYM active=%s seg%d={c0,shared} seg%d={c1}" % (activeTC, c0, c1)}

    # smaller: pad comp1 to the next segment. grows LDS -> PGR1/PGR2 + force only.
    if state.get("PrefetchGlobalRead") not in (1, 2):
        return _no("bcontig-asym: aligned layout needs PGR=1 or 2")
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
    bBaselineSpan = _tensor_footprint(state, "B")
    base = state["LdsOffsetA"]

    # bcontig fallback [A0][B0][B1][A1] (auto-only, not user-forceable): when B can't be split
    # (odd WaveTileB), keep B whole and use it as the gap that pushes A1 into the next segment.
    if not _b_readable(state):
        strideA = fA + bBaselineSpan                 # distance A0 -> A1: skip A0 and the whole B block
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
            bMXSA, bMXSB, _ = _mx_scale_bases(state, base + 2 * fA + bBaselineSpan)
            if bMXSA is not None: offsets["ldsBaseMXSA"] = bMXSA
            if bMXSB is not None: offsets["ldsBaseMXSB"] = bMXSB
            return {"applicable": True, "aligned": False, "offsets": offsets,
                    "blockSpan": 0, "reason": "bcontig",
                    "segmentMap": "BCONTIG seg%d={A0,B0,B1} seg%d={A1}" % (a0, a1)}

        # Small tile: A0+B0+B1 all fit in one segment, so A1 would stay with A0. Pad the A0 -> A1
        # distance up to the next segment boundary so A1 lands in a different segment. Uses more LDS
        # (checked in Solution.py). This requires PGR1/PGR2, or PGR2 for TDMSplit.
        if (state.get("PrefetchGlobalRead") != 2
                and (state.get("TDMSplit") or state.get("PrefetchGlobalRead") != 1)):
            return _no("bcontig: aligned layout needs PGR=2 with TDMSplit, otherwise PGR=1 or 2")
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
        # with a segment-aligned stride. Grows LDS (Solution.py budget-checks).
        if (state.get("PrefetchGlobalRead") != 2
                and (state.get("TDMSplit") or state.get("PrefetchGlobalRead") != 1)):
            return _no("aligned: needs PGR=2 with TDMSplit, otherwise PGR=1 or 2")
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


def b_rotate_reason(state, persistent=False):
    """LDSSegmentBRotate gate. Returns None when B-rotate can be applied, else the reject reason.

    Validate the logical [2,2] split layout used by aligned and paired tight kernels.
    B's logical component stride stays in [64K, 128K), so LocalReadAddrB+h still selects
    half h. The physical component stride comes from the selected layout; a per-wave
    register delta bridges that distance and rotates odd waves' B reads. The accumulator halves are
    swapped back before the epilogue. Aligned kernels retain separate ping/pong blocks.
    """
    pt = state["ProblemType"]
    if tuple(state.get("ISA", ()))[:2] != (12, 5):            return "not gfx1250"
    if state.get("LDSSegmentInterleave") != 1:                return "needs LDSSegmentInterleave=1"
    o = state.get("LDSSegInterleaveOffsets") or {}
    for key in ("aBaseline", "bBaseline", "portSplitA", "portSplitB", "componentSplit"):
        if o.get(key):                                        return "LSI layout '%s' not supported" % key
    S = o.get("readStrideBytesB", o.get("writeStrideBytes", 0))
    if not (SEG <= S < 2 * SEG):                              return "component stride must be in [64K, 128K)"
    fB = _footprint(state, "B")
    baseB = o.get("ldsBaseB", 0)
    # LocalRead picks LocalReadAddrB+h by offset // 64K: half 0 must stay below 64K, half 1 below 128K.
    if fB > SEG or S + fB > 2 * SEG:                          return "B component reach exceeds the two 64K local-read registers"
    # pair 1's +1 = root - (S - 64K) must stay non-negative.
    if baseB < S - SEG:                                       return "ldsBaseB below S - 64K"
    if state.get("NumLdsBlk") != 2 or state.get("1LDSBuffer") != 0:
        return "needs 2 LDS buffers"
    if list(state["MIWaveGroup"]) != [2, 2] or state["NumWaves"] != 4 or state["WavefrontSize"] != 32:
        return "needs MIWaveGroup [2,2], 4 waves, wave32"
    fA = _footprint(state, "A")
    if fA > SEG:
        return "A component exceeds 64K"
    if not (state.get("enableTDMA") and state.get("enableTDMB")):
        return "needs wave-separated TDM for both operands"
    if state.get("PrefetchGlobalReadA", -1) != -1 or state.get("PrefetchGlobalReadB", -1) != -1:
        return "decoupled PrefetchGlobalReadA/B not supported"
    if state.get("PrefetchGlobalRead") not in (1, 2):          return "needs PrefetchGlobalRead=1 or 2"
    if state.get("ExpandPointerSwap"):                        return "needs ExpandPointerSwap=0"
    vwb, wtb = state["VectorWidthB"], state["MIWaveTile"][1]
    if vwb <= 0 or wtb % vwb != 0 or (wtb // vwb) % 2 != 0:
        return "MIWaveTile[1]/VectorWidthB must be even"
    if pt.get("MXBlockB"):
        # MXSB must follow the rotation. Supported only for the TileSpan wave-split layout with
        # one group (nV == 2): vIdx 0/1 scales sit in the lower/upper half-wave of one ds_load,
        # so pair 1 flips the half-wave select in LRA. MXSA needs nothing (A is not rotated).
        # TileSpan (LocalRead.getMxsTileSpanInfo) is gated on exactly gfx1250.
        if tuple(state.get("ISA", ())) != (12, 5, 0):         return "MX: needs ISA (12, 5, 0)"
        if state.get("DirectToVgprMXSB"):                     return "MX: DirectToVgprMXSB not supported"
        if state.get("MXScaleFormat") != "InMemorySwizzle":   return "MX: needs MXScaleFormat InMemorySwizzle"
        if state["MatrixInstN"] != state["WavefrontSize"] // 2:
            return "MX: needs MatrixInstN == WavefrontSize/2"
        if wtb // vwb != 2:                                   return "MX: needs MIWaveTile[1]/VectorWidthB == 2"
    if pt.get("Sparse"):                                      return "sparse not supported"
    for key in ("TDMSplit", "UseSubtileImpl", "DirectToVgprA", "DirectToVgprB", "DirectToLdsB",
                "enableLDSTrA", "enableLDSTrB", "UseDotInstruction", "UseCustomMainLoopSchedule",
                "PrefetchAcrossPersistent", "TDMFuse"):
        v = state.get(key)
        # -1 (auto) means unresolved/off for the tri-state knobs; only a positive value enables.
        if v is True or (isinstance(v, int) and v > 0):      return "%s not supported" % key
    if state.get("LocalSplitU", 1) > 1:                       return "LocalSplitU>1 not supported"
    if state.get("NumWaveSplitK", 1) > 1:                     return "NumWaveSplitK>1 not supported"
    if persistent:                                            return "persistent / StreamK not supported"
    return None


def paired_buffer_reason(state, persistent=False):
    """Limit general pairing to the existing two-buffer TDM emit path."""
    if state.get("LDSSegmentInterleave") != 1:
        return "segment interleave is not active"
    if state.get("NumLdsBlk") != 2 or state.get("1LDSBuffer") != 0:
        return "needs two LDS buffers"
    if state.get("PrefetchGlobalRead") not in (1, 2):
        return "needs PrefetchGlobalRead=1 or 2"
    if state.get("WavefrontSize") != 32:
        return "needs wave32"
    if state.get("PrefetchGlobalReadA", -1) != -1 or state.get("PrefetchGlobalReadB", -1) != -1:
        return "decoupled prefetch is not supported"
    if persistent:
        return "persistent / StreamK is not supported"
    for key in ("ExpandPointerSwap", "TDMSplit", "UseSubtileImpl", "DirectToVgprA",
                "DirectToVgprB", "DirectToLdsA", "DirectToLdsB", "enableLDSTrA",
                "enableLDSTrB", "UseDotInstruction", "UseCustomMainLoopSchedule",
                "PrefetchAcrossPersistent", "TDMFuse"):
        v = state.get(key)
        if v is True or (isinstance(v, int) and v > 0):
            return "%s is not supported" % key
    if state.get("LocalSplitU", 1) != 1 or state.get("NumWaveSplitK", 1) > 1:
        return "split-K LDS layout is not supported"
    return None


def paired_buffer_offsets(state, isolate_operands=False):
    """Pair existing read groups without changing their membership or read order.

    Separate corresponding component footprints in each buffer phase, equally for
    A and B. Cross-phase and cross-operand accesses may still share segments.
    A contiguous operand retains its original padding phase. Optional operand isolation
    adds segment gaps between every pair; reject it if the complete layout does not
    fit. The caller includes pairedUsedBytes in the normal LDS budget calculation.
    bufferStride<tensor> is the ping-to-pong distance within a pair;
    writeStrideBytes is the component-0-to-component-1 distance. Both distances
    and group bases are 256-byte aligned without changing internal padding.
    """
    offsets = dict(state["LDSSegInterleaveOffsets"])
    base = _align_paired(state["LdsOffsetA"])
    shared = "A" if offsets.get("aBaseline") else ("B" if offsets.get("bBaseline") else None)
    active = "B" if shared == "A" else "A"
    fActive = _footprint(state, active)
    fOther = _tensor_footprint(state, shared) if shared else _footprint(state, "B")
    dActive = _align_paired(fActive)
    dOther = _align_paired(fOther)
    other = shared or "B"
    activeEnd = base + dActive + fActive
    otherBase = _ceil_seg(activeEnd) if isolate_operands else _align_paired(activeEnd)
    stride = _align_paired(otherBase - base + dOther + fOther)
    if isolate_operands:
        stride = _ceil_seg(base + stride) - base
    else:
        # Use one component stride for both operands. Each component-1 ping/pong
        # copy must start past the last segment occupied by the matching
        # component-0 copy, including padding. The natural stride already keeps
        # the intervening pair out of the way; only add a gap if needed.
        splitPairs = [(base, fActive, dActive)]
        if not shared:
            splitPairs.append((otherBase, fOther, dOther))
        for pairBase, size, bufferStride in splitPairs:
            for buffer in (0, 1):
                start = pairBase + buffer * bufferStride
                end = start + size  # Exclusive: an exact segment boundary is reusable.
                nextSegmentStart = _ceil_seg(end)
                stride = max(stride, nextSegmentStart - start)
    stride = _align_paired(stride)
    offsets["ldsBase" + active] = base
    offsets["ldsBase" + other] = otherBase
    offsets["bufferStride" + active] = dActive
    offsets["bufferStride" + other] = dOther
    if state.get("LDSSegmentBRotate"):
        # Preserve logical DS half selection while VGPR bases bridge the common
        # physical stride. Rotation changes read order, never data placement.
        offsets["readStrideBytesB"] = offsets.get("readStrideBytesB", offsets["writeStrideBytes"])
    else:
        offsets.pop("readStrideBytesB", None)
    offsets["writeStrideBytes"] = stride
    end = base + stride + dActive + fActive if shared else otherBase + stride + dOther + fOther
    for tc in ("MXSA", "MXSB"):
        if "ldsBase" + tc in offsets:
            size = int(state["LdsNumElementsAligned" + tc])
            scaleBase = _align_paired(end)
            offsets["ldsBase" + tc] = scaleBase
            offsets["bufferStride" + tc] = _align_paired(size)
            end = scaleBase + offsets["bufferStride" + tc] + size
    if end > state["MaxLDS"]:
        return None
    offsets["pairedBuffers"] = True
    offsets["pairedUsedBytes"] = end
    offsets["pairedOperandIsolation"] = isolate_operands
    # Narrow [2,2] B reads reach both components through DS immediates. Reserve
    # only reachable 64 KiB windows, rather than allocating registers for gaps.
    if state.get("LDSSegmentBRotate"):
        # B rotation uses two logical halves, not the ordinary physical windows.
        offsets.pop("readAddressWindowsB", None)
    elif not shared and state["MIWaveGroup"] == [2, 2] and not _coarse(state, "B"):
        windows = set(range((fOther - 1)//SEG + 1))
        windows.update(range(stride//SEG, (stride + fOther - 1)//SEG + 1))
        offsets["readAddressWindowsB"] = sorted(windows)
    return offsets


def paired_buffer_layout_reason(state, offsets):
    """Validate physical placement, independently of the read-conflict policy.

    Footprints retain their existing internal padding. This checks every ping/pong
    start (including LDS-resident MX scales), allocation overlap and the LDS budget;
    it does not replace the TDM padding-phase or per-instruction read checks.
    """
    regions = []
    for tc in ("A", "B", "MXSA", "MXSB"):
        if tc in ("A", "B"):
            shared = offsets.get("aBaseline" if tc == "A" else "bBaseline", False)
            size = _tensor_footprint(state, tc) if shared else _footprint(state, tc)
            components = 1 if shared else 2
        else:
            if "ldsBase" + tc not in offsets:
                continue
            size = int(state["LdsNumElementsAligned" + tc])
            components = 1
        base = offsets.get("ldsBase" + tc, state["LdsOffsetA"])
        bufferStride = offsets["bufferStride" + tc]
        if size <= 0:
            return "%s has an empty paired footprint" % tc
        for component in range(components):
            for buffer in (0, 1):
                start = base + component * offsets["writeStrideBytes"] + buffer * bufferStride
                name = "%s%d %s" % (tc, component, "pong" if buffer else "ping")
                if start < 0 or start % 256:
                    return "%s start %d is not a non-negative 256-byte-aligned address" % (name, start)
                regions.append((start, start + size, name))
    regions.sort()
    for left, right in zip(regions, regions[1:]):
        if left[1] > right[0]:
            return "%s overlaps %s" % (left[2], right[2])
    if regions[-1][1] > state["MaxLDS"]:
        return "paired layout exceeds MaxLDS"
    if regions[-1][1] > offsets["pairedUsedBytes"]:
        return "paired footprint exceeds pairedUsedBytes"
    return None
