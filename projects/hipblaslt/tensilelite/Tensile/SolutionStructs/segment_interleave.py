# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx1250 LDS segment-conflict interleave oracle.

Pure function of `state` that decides whether a wave-separated TDM kernel should
split its A/B halves across LDS segments (so the two MFMA read ports hit different
segments), and returns the offsets the emit sites consume.
"""

# gfx1250 LDS segment size (5 x 64 KiB segments).
SEG = 65536

def _bpe(state):
    # DataType is a DataType object, not a string; numBytes() is 2.0 for bf16.
    return int(state["ProblemType"]["DataType"].numBytes())

def _pad(x, blk, padElems, bpe):
    if blk == 0 or padElems == 0:
        return 0
    return (x // blk) * (padElems * bpe)

def _data_bytes(state, tc):
    numComp = state["NumWaves"] // 2
    mt = state["MacroTile0"] if tc == "A" else state["MacroTile1"]
    return (mt // numComp) * state["DepthU"] * _bpe(state)

def _footprint(state, tc):
    d = _data_bytes(state, tc)
    blk = state["LdsBlockSizePerPad%s" % tc]
    padElems = state["LdsPad%s" % tc]
    return d + _pad(d, blk, padElems, _bpe(state))

def _coarse_vw(state):
    # Each read port must cover one contiguous group per tensor; check A and B, since a
    # fine VW (e.g. VWB=1 from an odd WaveTile) reads stripes spanning both wave-halves.
    numComp = state["NumWaves"] // 2
    mi_threads = min(state["MatrixInstM"], state["MatrixInstN"])
    coarseA = mi_threads * state["VectorWidthA"] >= state["MacroTile0"] // numComp
    coarseB = mi_threads * state["VectorWidthB"] >= state["MacroTile1"] // numComp
    return coarseA and coarseB

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
    if state.get("TDMSplit") or pt.get("MXBlockA") or pt.get("MXBlockB") or pt.get("Sparse"):
        return _no("split/mxs/sparse")
    # Subtile is a separate codegen body (kernelBodySubtile); the wave-separated TDM emit path
    # the offsets rely on runs only for non-subtile kernels, so interleave never applies here.
    if state.get("UseSubtileImpl"):                            return _no("subtile")
    # Needs double-buffering: 1LDSBuffer==1 breaks the layout the offsets assume. It is not yet
    # resolved here (Solution.py resolves -1 later), so reject both 1 and the unresolved -1.
    if state.get("1LDSBuffer", 0) != 0:                         return _no("needs 1LDSBuffer==0")
    _dt = pt["DataType"]
    if not (_dt.isBFloat16() or _dt.isHalf()):
        return _no("bf16/fp16 only")
    if not _coarse_vw(state):                                   return _no("fine VW")

    fA, fB = _footprint(state, "A"), _footprint(state, "B")
    base = state["LdsOffsetA"]
    bpe = _bpe(state)

    if (base % SEG) + fA + fB < SEG:
        # Small MacroTile: A0,B0 fit one segment, so push component 1 to the next segment boundary
        # via a segment-aligned component stride. Footprint-packed (no re-pad on the component jump,
        # like tight), so asymmetric A/B pads/blocks are handled by construction. Grows LDS ->
        # Solution.py budget-checks; supports simple double-buffer only:
        if state.get("PrefetchGlobalRead") != 2:        return _no("small MT: PGR!=2")
        if mode == -1:                                  return _no("auto: skip aligned (LDS growth)")
        pre = _ceil_seg(base + fA + fB) - base          # segment-aligned stride (== SEG for base<SEG)
        offsets = {
            "ldsBaseB":         base + fA,              # B0 right after A0 in seg0
            "writeStrideBytes": pre,                    # segment stride; no re-pad on the jump
            "readWaveStride":   pre // bpe,
            "footprintPacked":  True,
        }
        # Per-buffer span: B1 ends at base + pre(=A1) + fA + fB.
        blockSpan = base + pre + fA + fB
        return {"applicable": True, "aligned": True, "offsets": offsets,
                "blockSpan": blockSpan, "reason": "aligned",
                "segmentMap": "ALIGNED seg%d={A0,B0} seg%d={A1,B1}"
                              % (base // SEG, (base + pre) // SEG)}

    # Tight: pack [A0][B0][A1][B1] by FOOTPRINT (each tile's own post-pad size fX). The component
    # stride is fA+fB and is applied WITHOUT the per-tensor re-pad (fX already includes each tile's
    # pad), so A1 lands exactly at B0's end and B1 at A1's end regardless of padA vs padB. The
    # emit sites honour footprintPacked by skipping the component-offset re-pad on write and read.
    offsets = {
        "ldsBaseB":         base + fA,          # B0 right after A0
        "writeStrideBytes": fA + fB,            # footprint stride (post-pad), no re-pad on the jump
        "readWaveStride":   (fA + fB) // bpe,   # same, in elements
        "footprintPacked":  True,
    }
    a0 = base // SEG
    a1 = (base + fA + fB) // SEG                 # tight branch guarantees a1 > a0
    seg_map = "CLEAN seg%d={A0,B0} seg%d={A1,B1}" % (a0, a1)
    return {"applicable": True, "aligned": False, "offsets": offsets,
            "blockSpan": 0, "reason": "tight", "segmentMap": seg_map}
