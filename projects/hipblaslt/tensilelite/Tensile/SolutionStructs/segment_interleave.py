# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx1250 LDS segment-conflict interleave oracle.

Pure function of `state` that decides whether a wave-separated TDM kernel should put
operand A's two halves in different LDS segments (so A's two MFMA read ports stop
conflicting), and returns the byte offsets the emit sites consume.

Only A is separated this way (B is not). Baseline packs each operand's halves adjacently
([A0][A1][B0][B1]); this picks one of two layouts by whether B can be split cleanly:
  split   [A0][B0][A1][B1]: B's halves move too, but its ports can still hit one B segment.
  bcontig [A0][B0][B1][A1]: for odd WaveTileB (a split B read would cross a component),
          B is kept whole with baseline addressing and only A moves.
Each layout is tight (no extra LDS) or aligned (padded to a segment boundary; more LDS, PGR2).
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

def _coarse_a(state):
    # A must cover a full component (never crosses one) so each A read lands entirely within one
    # segment. Equivalent to VWA == WaveTileA.
    numComp = state["NumWaves"] // 2
    mi_threads = min(state["MatrixInstM"], state["MatrixInstN"])
    return mi_threads * state["VectorWidthA"] >= state["MacroTile0"] // numComp

def _port_split_a(state):
    # Fine A (VWA==WaveTileA/2, not coarse): split along the port axis instead of the component axis.
    # Only VWA==WaveTileA/2 (2 vIdx per port) works, and only with TDMSplit; finer VWA can't.
    if _coarse_a(state) or not state.get("TDMSplit"):
        return False
    vwa = state["VectorWidthA"]
    return vwa > 0 and state["MIWaveTile"][0] % vwa == 0 and state["MIWaveTile"][0] // vwa == 2

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
    if state["NumWaves"] // 2 != 2:                             return _no("numComp!=2")
    # Both write (WaveIdx//2 -> 2 comps) and read (wtid0*stride, num1DWaves=MIWaveGroup dim)
    # assume exactly 2 waves per MFMA dim. MIWaveGroup!=[2,2] (e.g. [4,1]) loses the component
    # jump on the dim==1 tensor and reads OOB on the dim==4 one.
    if list(state.get("MIWaveGroup", [])) != [2, 2]:           return _no("MIWaveGroup!=[2,2]")
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
    # A must be coarse (VWA==WaveTileA) or port-split (VWA==WaveTileA/2, needs TDMSplit).
    _portSplit = _port_split_a(state)
    if not (_coarse_a(state) or _portSplit):                  return _no("A: VWA must be WaveTileA, or WaveTileA/2 with TDMSplit")

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
        # (checked in Solution.py) and needs PrefetchGlobalRead=2 -- same idea as the split branch below.
        if state.get("PrefetchGlobalRead") != 2:   return _no("small MT: PGR!=2")
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
        if state.get("PrefetchGlobalRead") != 2:        return _no("small MT: PGR!=2")
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
