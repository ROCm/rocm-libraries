# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Leaf emitters -- the L3 per-tile realization of a GIR Move/Mma.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import ceil

from rocisa.code import Module
from rocisa.container import vgpr, DSModifiers
from rocisa.instruction import MFMAInstruction, MXMFMAInstruction
from rocisa.enum import InstType

from ..Component import Component
from ..Common.MatrixInstructionNaming import (dataTypeToMfmaInstTypePair,
                                              dataTypeNameAbbrevToInstType)
from .lds_geometry import (tile_row, region_bytes, region_row_elems, component_fold,
                           addr_coord_on_split_axis, read_fragments, BPR)
from ..Components.TDMSplit import split_of, seg_interleave_row_shape, AXIS_MT
from .gir_tag import gir_tag

#: Per-tensor Solution and LDSSegInterleaveOffsets keys: (vector width, port split, baseline).
_SEG_INTERLEAVE_KEYS = {"A": ("VectorWidthA", "portSplitA", "aBaseline"),
                        "B": ("VectorWidthB", "portSplitB", "bBaseline")}
_MACRO_TILE_KEYS = ("MacroTile0", "MacroTile1")


@dataclass
class MfmaTileContext:
    """The loop-order-invariant constants a per-tile WMMA emit needs.  Built ONCE per kiter,
    then reused for every (idx0, idx1) tile."""
    miInInstType: object
    miOutInstType: object
    variant: list
    accs_per_wave: int
    vgprPerInputA: int
    vgprPerInputB: int
    mfma_1k: bool
    neg_flag: bool
    mi_wavetile: list
    tile01B: int
    mxBlock: int = 0
    vgprPerInputMXSA: int = 0
    vgprPerInputMXSB: int = 0
    mxScaleATypeInst: object = None
    mxScaleBTypeInst: object = None


@dataclass
class LdsReadTileContext:
    """Loop-order-invariant LDS-read constants for one operand (transpose OR general path)."""
    tc: str
    tile01: int
    LocalReadX: object
    blockWidth: object
    bpeDS: object
    LdsPad: int
    ldsBlockSizePerPad: int
    padBytes: int
    UnrollStride: int
    MIWaveGroupShape: list
    localReadOffset: int
    wtRegStride: int
    substepStrideBytes: int
    maxLDSConstOffset: int
    miWaveTileAxis: int
    # The operand's free-axis extent, which the tile-major region fold divides.
    macroTile: int = 0
    unitRegStride: int = 1
    enableLDSTr: bool = True
    tileStrideElems: int = 0
    # VECTOR WIDTH for this operand, and the ROW JUMP between vector groups.  A wave's tiles
    # are handed out in groups of `vectorWidth` ADJACENT rows, then the next group starts
    vectorWidth: int = 1
    # Which layout `tileStrideElems` is expressed in.  True (unroll-major): it is the ROW stride
    # (DU + pad) and the tile index must be turned into a row index first.  False (transpose path).
    unrollMajor: bool = True
    # BYTES BETWEEN CONSECUTIVE STORAGE REGIONS of this operand's tile, 0 when unsplit.
    splitBoundaryBytes: int = 0
    # TDMSplit region count for this operand (1 = unsplit), and WHICH axis it cuts.  The two
    # always travel together: every consequence -- how many wave tiles share a region, how short
    nsplit: int = 1
    splitAxis: object = AXIS_MT
    fragments: tuple = ((0, 0),)
    tilePerRead: int = 1
    # MX TileSpan: the VectorWidth when the half-wave scale layout is ON, else 0.  Under it LRA
    # packs `2*VW` tiles' scale blocks so ONE ds_load holds two half-waves' worth; only the LOWER
    tileSpanVW: int = 0
    # LDSSegmentInterleave.  `segRowShape` replaces the wave-group row stride when the segment
    # jump rides on the wave stride instead (port/component split); `segCompCols` and
    # `segWriteStrideBytes` are the component fold.  All zero = baseline layout.
    segRowShape: int = 0
    segCompCols: int = 0
    segWriteStrideBytes: int = 0


def _abbrevToInstType(abbrev: str):
    """Minimal dtype-abbrev -> InstType for the ACCUMULATOR type (ComputeDataType; f32 here).

    The INPUT side does not come through here -- it reads the scaffold's shared
    `dataTypeToMfmaInstTypePair`, which is the 40-way table including the mixed fp8/bf8 pairs."""
    table = {
        "f32": InstType.INST_F32,
        "f64": InstType.INST_F64,
        "f16": InstType.INST_F16,
        "bf16": InstType.INST_BF16,
        "i32": InstType.INST_I32,
    }
    if abbrev not in table:
        raise NotImplementedError(f"LeafEmitters: output abbrev {abbrev!r} unsupported")
    return table[abbrev]


class LeafEmitters:
    """Per-coordinate leaf emitters for the UseLoopModel L3.  Constructed with the active
    KernelWriterAssembly (`writer`) and the GIR register ring widths (`reg_depth`).

    `reg_depth` is `{operand: W}` -- PER OPERAND, because they genuinely differ: a scale operand's
    ring is derived from its own fragment, so an MX kernel at PLR0 has A at 2 and MXSA at 1.  An
    int is still accepted and means "this width for every operand" (hand-built contexts, tests)."""

    def __init__(self, writer, reg_depth=1):
        self.writer = writer
        self.reg_depth = reg_depth or 1

    def _ringWidth(self, tc) -> int:
        """The rotation width for operand `tc`, for the `m = u % W` fallback only.  Every
        GIR-planned act supplies its own slot, so this is reached only by a caller that passed
        none."""
        if isinstance(self.reg_depth, dict):
            return max(1, int(self.reg_depth.get(tc, 1)))
        return max(1, int(self.reg_depth or 1))

    def _valuStart(self, tc) -> int:
        attr = {"A": "a", "B": "b", "MXSA": "mxsa", "MXSB": "mxsb"}[tc]
        return int(getattr(self.writer.states, attr).startVgprValu)

    # ------------------------------------------------------------------ WMMA compute leaf
    def buildMfmaContext(self, kernel, tPA, tPB) -> MfmaTileContext:
        """Loop-order-invariant WMMA constants for this kernel (validated to reject anything other
        than dense gfx1250 WMMA at a supported input width)."""
        w = self.writer
        pt = kernel["ProblemType"]

        if not w.states.asmCaps.get("HasWMMA_V3", False) and \
           not w.states.asmCaps.get("HasWMMA_V2", False) and \
           not w.states.asmCaps.get("HasWMMA_V1", False):
            raise NotImplementedError("LeafEmitters: non-WMMA target (needs gfx1250)")
        if w.states.asmCaps.get("HasMFMA", False):
            raise NotImplementedError("LeafEmitters: MFMA (CDNA) path not supported")
        if pt["Sparse"]:
            raise NotImplementedError("LeafEmitters: sparse not supported (phase 3a)")
        if kernel["UseF32XEmulation"] or kernel.get("EnableF32XdlMathOp", False):
            raise NotImplementedError("LeafEmitters: F32 emulation not supported (phase 3a)")

        miInputTypeA = pt["MacDataTypeA"]
        miInputTypeB = pt["MacDataTypeB"]
        # SUPPORTED INPUT WIDTHS.  bf16 (2 bytes) and 8-bit float; both are dense WMMA whose whole
        # dtype dependence is `numRegisters()` (below) plus the opcode (`dataTypeToMfmaInstTypePair`).
        for _t in (miInputTypeA, miInputTypeB):
            if not (_t.isBFloat16() or _t.is8bitFloat()):
                raise NotImplementedError(
                    f"LeafEmitters: only bf16 and 8-bit float inputs supported (got "
                    f"{miInputTypeA.toChar()}/{miInputTypeB.toChar()})")

        numRegistersInA = miInputTypeA.numRegisters()
        numRegistersInB = miInputTypeB.numRegisters()
        numRegistersOut = kernel["MIRegPerOut"]
        accs_per_wave = kernel["MatrixInstM"] * kernel["MatrixInstN"] * kernel["MatrixInstB"] \
                        // kernel["WavefrontSize"] * numRegistersOut

        numMIInputA = kernel["MIInputPerThreadA"]
        numMIInputB = kernel["MIInputPerThreadB"]
        vgprPerInputA = int(numMIInputA * numRegistersInA)
        vgprPerInputB = int(numMIInputB * numRegistersInB)

        miInInstType, _ = dataTypeToMfmaInstTypePair(miInputTypeA, miInputTypeB,
                                                     kernel["SourceSwap"])
        miOutInstType = _abbrevToInstType(pt["ComputeDataType"].toNameAbbrev())

        variant = [kernel["MIBlock"][0], kernel["MIBlock"][1],
                   kernel["MatrixInstK"], kernel["MatrixInstB"]]

        # MICROSCALING CONSTANTS, derived exactly as `mfmaIter` derives them.  `block` is the MAX
        # of the two block sizes because the instruction takes ONE block modifier for both scale
        mxA = int(pt["MXBlockA"] or 0)
        mxB = int(pt["MXBlockB"] or 0)
        if mxA and mxB and mxA != mxB:
            raise NotImplementedError(
                f"LeafEmitters: MXBlockA={mxA} != MXBlockB={mxB}; the WMMA carries ONE block "
                f"modifier for both scale operands, so an asymmetric pair has no faithful emit")
        vgprPerInputMXSA = int(ceil(kernel["MIInputPerThreadMXSA"]
                                    * pt["DataTypeMXSA"].numRegisters())) if mxA else 0
        vgprPerInputMXSB = int(ceil(kernel["MIInputPerThreadMXSB"]
                                    * pt["DataTypeMXSB"].numRegisters())) if mxB else 0

        return MfmaTileContext(
            miInInstType=miInInstType, miOutInstType=miOutInstType, variant=variant,
            accs_per_wave=accs_per_wave, vgprPerInputA=vgprPerInputA,
            vgprPerInputB=vgprPerInputB, mfma_1k=bool(kernel["MFMA_BF16_1K"]),
            neg_flag=False, mi_wavetile=list(kernel["MIWaveTile"]),
            tile01B=tPB["tile01Idx"],
            mxBlock=max(mxA, mxB),
            vgprPerInputMXSA=vgprPerInputMXSA, vgprPerInputMXSB=vgprPerInputMXSB,
            mxScaleATypeInst=dataTypeNameAbbrevToInstType(pt["DataTypeMXSA"].toNameAbbrev()),
            mxScaleBTypeInst=dataTypeNameAbbrevToInstType(pt["DataTypeMXSB"].toNameAbbrev()))

    def emitWmmaTile(self, kernel, tPA, tPB, ctx: MfmaTileContext,
                     idx0: int, idx1: int, u: int, iui: int = 0,
                     vregSetIdx: int = 0, unrollLoopIdx: int = 0,
                     bufA: int = None, bufB: int = None,
                     bufMXA: int = None, bufMXB: int = None,
                     unitA: int = None, unitB: int = None,
                     unitMXA: int = None, unitMXB: int = None,
                     baseA: int = None, baseB: int = None,
                     baseMXA: int = None, baseMXB: int = None) -> Module:
        """Emit the SINGLE WMMA instruction for tile (idx0, idx1) at kiter substep `u`."""
        w = self.writer
        # register generation m = u % W, W = the GIR register ring width (reg_depth).  PLR0 has
        # W=1 -> m=0 for every substep (single register generation); PLR1 W=2 splits X0/X1.
        def _fb(tc):
            return u % self._ringWidth(tc)
        mA = bufA if bufA is not None else _fb(tPA["tensorChar"])
        mB = bufB if bufB is not None else _fb(tPB["tensorChar"])
        mMXA = bufMXA if bufMXA is not None else (
            _fb(tPA["MX"]["tensorChar"]) if tPA.get("MX") is not None else 0)
        mMXB = bufMXB if bufMXB is not None else (
            _fb(tPB["MX"]["tensorChar"]) if tPB.get("MX") is not None else 0)
        # the wmma reads the SAME Valu ring the ds_read wrote, so it needs the same bound check --
        # a consume-side slot past the `.set` table is the identical undefined-symbol failure.
        self._checkRegBuffer(tPA["tensorChar"], mA, "wmma source A")
        self._checkRegBuffer(tPB["tensorChar"], mB, "wmma source B")

        accIdx = idx1 * ctx.mi_wavetile[0] + idx0
        accStart = accIdx * ctx.accs_per_wave
        accEnd = accStart + ctx.accs_per_wave - 1

        idxA = idx0 if ctx.tile01B else idx1
        idxB = idx1 if ctx.tile01B else idx0

        if baseA is not None and unitA is not None:
            aStr = vgpr(self._valuStart(tPA["tensorChar"]) + baseA
                        + unitA * ctx.vgprPerInputA, ctx.vgprPerInputA)
        else:
            aStr_base = w.generateSrcStrForMFMA(
                kernel, tPA, kernel["InnerUnroll"], vregSetIdx, ctx.vgprPerInputA,
                mA, u, iui, idxA, unrollLoopIdx)
            aStr = vgpr(aStr_base, ctx.vgprPerInputA)
        if baseB is not None and unitB is not None:
            bStr = vgpr(self._valuStart(tPB["tensorChar"]) + baseB
                        + unitB * ctx.vgprPerInputB, ctx.vgprPerInputB)
        else:
            bStr_base = w.generateSrcStrForMFMA(
                kernel, tPB, kernel["InnerUnroll"], vregSetIdx, ctx.vgprPerInputB,
                mB, u, iui, idxB, unrollLoopIdx)
            bStr = vgpr(bStr_base, ctx.vgprPerInputB)

        Str0 = aStr if ctx.tile01B else bStr
        Str1 = bStr if ctx.tile01B else aStr
        src0, src1 = (Str1, Str0) if kernel["SourceSwap"] else (Str0, Str1)

        acc = w.accVgprReadWriteIndex(kernel, accStart, (accEnd - accStart + 1))
        imod = Module("<LoopIR: wmma tile (%u,%u) u=%u>" % (idx0, idx1, u))
        _loopir = "<LoopIR: C[m=%u,n=%u] += A[m,k=%u]*B[n,k=%u]>" % (idx0, idx1, u, u)
        _wmma = "wmma regA=X%u regB=X%u" % (mA, mB)
        if not ctx.mxBlock:
            imod.add(MFMAInstruction(
                instType=ctx.miInInstType, accType=ctx.miOutInstType, variant=ctx.variant,
                mfma1k=ctx.mfma_1k, acc=acc, a=src0, b=src1, acc2=acc, neg=ctx.neg_flag,
                comment="%s  %s" % (_loopir, gir_tag(_wmma))))
            return imod

        # ---- MICROSCALED: the same opcode, plus the two scale registers and a block modifier ---
        #
        mxaStr = mxbStr = None
        selA = selB = 0
        if tPA.get("MX") is not None:
            mA_idx, selA = w.mxsTileSpanScaleSel(kernel, tPA["MX"], idxA)
            self._checkRegBuffer(tPA["MX"]["tensorChar"], mMXA, "wmma scale source A")
            if baseMXA is not None and unitMXA is not None:
                mxaStr = vgpr(self._valuStart(tPA["MX"]["tensorChar"]) + baseMXA
                              + unitMXA * ctx.vgprPerInputMXSA, ctx.vgprPerInputMXSA)
            else:
                mxaStr = vgpr(w.generateSrcStrForMFMA(
                    kernel, tPA["MX"], kernel["InnerUnroll"], vregSetIdx,
                    ctx.vgprPerInputMXSA, mMXA, u, iui, mA_idx), ctx.vgprPerInputMXSA)
        if tPB.get("MX") is not None:
            mB_idx, selB = w.mxsTileSpanScaleSel(kernel, tPB["MX"], idxB)
            self._checkRegBuffer(tPB["MX"]["tensorChar"], mMXB, "wmma scale source B")
            if baseMXB is not None and unitMXB is not None:
                mxbStr = vgpr(self._valuStart(tPB["MX"]["tensorChar"]) + baseMXB
                              + unitMXB * ctx.vgprPerInputMXSB, ctx.vgprPerInputMXSB)
            else:
                mxbStr = vgpr(w.generateSrcStrForMFMA(
                    kernel, tPB["MX"], kernel["InnerUnroll"], vregSetIdx,
                    ctx.vgprPerInputMXSB, mMXB, u, iui, mB_idx), ctx.vgprPerInputMXSB)
        if mxaStr is None or mxbStr is None:
            raise NotImplementedError(
                "LeafEmitters: one-sided MX (only one of A/B carries scales) is not supported; "
                "the absent side needs the scaffold's ValuMXSDummy, whose width follows the "
                "present side's block size")

        # The scale operands follow the SAME tile01/SourceSwap swap as the data operands, because
        # `matrix_a_scale`/`matrix_b_scale` name the operand POSITIONS, not A and B.
        strMX0 = mxaStr if ctx.tile01B else mxbStr
        strMX1 = mxbStr if ctx.tile01B else mxaStr
        selMX0 = selA if ctx.tile01B else selB
        selMX1 = selB if ctx.tile01B else selA
        typeMX0 = ctx.mxScaleATypeInst if ctx.tile01B else ctx.mxScaleBTypeInst
        typeMX1 = ctx.mxScaleBTypeInst if ctx.tile01B else ctx.mxScaleATypeInst
        if kernel["SourceSwap"]:
            strMX0, strMX1 = strMX1, strMX0
            selMX0, selMX1 = selMX1, selMX0
            typeMX0, typeMX1 = typeMX1, typeMX0
        imod.add(MXMFMAInstruction(
            instType=ctx.miInInstType, accType=ctx.miOutInstType, variant=ctx.variant,
            mxScaleAType=typeMX0, mxScaleBType=typeMX1,
            acc=acc, a=src0, b=src1, acc2=acc,
            mxsa=strMX0, mxsb=strMX1, block=ctx.mxBlock,
            mxScaleASel=selMX0, mxScaleBSel=selMX1,
            comment="%s  %s" % (_loopir,
                                gir_tag(_wmma + " mxA=X%u mxB=X%u" % (mMXA, mMXB)))))
        return imod

    # ------------------------------------------------------------- MX scale LDS read leaf
    def buildMxScaleReadContext(self, kernel, tP) -> LdsReadTileContext:
        """Loop-order-invariant LDS-read constants for a MICROSCALING SCALE tensor (`MXSA`/`MXSB`)."""
        w = self.writer
        tc = tP["tensorChar"]
        if "MXS" not in tc:
            raise NotImplementedError(f"LeafEmitters MX read: {tc} is not a scale tensor")
        if not w.states.asmCaps.get("HasWMMA_V3", False):
            raise NotImplementedError("LeafEmitters MX read: needs HasWMMA_V3 (gfx1250)")

        tile01 = tP["tile01Idx"]
        instruction = tP["localReadInstruction"]
        blockWidth = instruction.blockWidth
        # `mxTc` is the PARENT's letter: the scale tensor named `MXSA` scales `A`, so its block
        # size is `MXBlockA`.  Taken off the tensor char rather than passed in, so a caller cannot
        mxTc = tc[3]
        mxBlock = int(kernel["ProblemType"]["MXBlock%s" % mxTc])
        if mxBlock <= 0:
            raise NotImplementedError(f"LeafEmitters MX read: {tc} present but MXBlock{mxTc}=0")
        mxUnit = int(kernel["MatrixInstK"]) // mxBlock
        stridePerRead = int(blockWidth * BPR)          # bytes one ds_read moves
        tilePerRead = stridePerRead // mxUnit if mxUnit else 0
        if tilePerRead < 1 or stridePerRead % mxUnit:
            raise NotImplementedError(
                f"LeafEmitters MX read ({tc}): one ds_read moves {stridePerRead} bytes, which is "
                f"not a whole number of tiles' scales (mxUnit {mxUnit}, blockWidth {blockWidth}); "
                f"a partial tile has no group-leader to attach it to")
        if kernel["MIWaveTile"][tile01] % tilePerRead:
            raise NotImplementedError(
                f"LeafEmitters MX read ({tc}): {tilePerRead} tiles per ds_read does not divide "
                f"MIWaveTile {kernel['MIWaveTile'][tile01]}, so the last group would be partial")

        MIWaveGroupShape = [
            kernel["MatrixInstM"] * kernel["MatrixInstBM"] * kernel["MIWaveGroup"][0] * kernel["VectorWidthA"],
            kernel["MatrixInstN"] * kernel["MatrixInstBN"] * kernel["MIWaveGroup"][1] * kernel["VectorWidthB"]]

        span = Component.LocalRead.find(w).getMxsTileSpanInfo(kernel, tc, tile01, w.states.asmCaps)
        tileSpanVW = int(span["vectorWidth"]) if span else 0

        return LdsReadTileContext(
            tc=tc, tile01=tile01, LocalReadX=instruction.getInst(), blockWidth=blockWidth,
            bpeDS=tP["bpeDS"], LdsPad=kernel["LdsPad%s" % tc],
            ldsBlockSizePerPad=kernel["LdsBlockSizePerPad%s" % tc],
            padBytes=kernel["LdsPad%s" % tc],
            # The per-SUBSTEP and per-TILE strides above.  `UnrollStride` is 1 because the fragment
            # displacement (there is one fragment, at 0) is already in elements.
            UnrollStride=1, tileStrideElems=mxUnit,
            substepStrideBytes=int(kernel["MacroTile%s" % mxTc] * mxUnit * tP["bpeDS"]),
            MIWaveGroupShape=MIWaveGroupShape, localReadOffset=tP["localReadOffset"],
            wtRegStride=int(ceil(blockWidth)), fragments=((0, 0),), tilePerRead=tilePerRead,
            unitRegStride=int(ceil(kernel["MIInputPerThread%s" % tc]
                                   * kernel["ProblemType"]["DataType%s" % tc].numRegisters())),
            tileSpanVW=tileSpanVW,
            maxLDSConstOffset=w.states.regCaps["maxLDSConstOffset"],
            miWaveTileAxis=kernel["MIWaveTile"][tile01],
            enableLDSTr=False, unrollMajor=True,
            vectorWidth=max(1, kernel["VectorWidth%s" % tc]),
            # A scale tensor is NEVER region-split -- `TDMSplit.split_of` answers `(1, None)` for
            # any `MXS*` char -- so every region term below is inert by construction.
            splitBoundaryBytes=0, nsplit=1, splitAxis=AXIS_MT)

    @staticmethod
    def _segInterleaveTerms(kernel, tc, tile01, waveGroupShape):
        """`(segRowShape, segCompCols, segWriteStrideBytes)` for LDSSegmentInterleave.

        Mirrors the three arms `Components/LocalRead.py` takes over `vCols`: a port split drops
        the wave-group factor from the row stride, a component split drops the per-component
        share, and a plain interleave folds the row onto the component it lands in.
        """
        if kernel.get("LDSSegmentInterleave") != 1:
            return (0, 0, 0)
        off = kernel["LDSSegInterleaveOffsets"] or {}
        stride = int(off.get("writeStrideBytes", 0))
        baseKey = _SEG_INTERLEAVE_KEYS[tc][2]
        # ONE derivation, shared with `TDMSplit.wave_region_span`: the row stride decides both the
        # read address and whether a wave's share crosses a TDMSplit region.
        rowShape = seg_interleave_row_shape(kernel, tc, tile01, waveGroupShape)
        if rowShape:
            return (rowShape, 0, 0)
        if off.get(baseKey, False) or not off.get("footprintPacked", False):
            return (0, 0, 0)
        compCols = kernel[_MACRO_TILE_KEYS[tile01]] // (kernel["NumWaves"] // 2)
        return (0, compCols if compCols > 0 and stride else 0, stride)

    # ------------------------------------------------------------------ LDS read leaf
    def buildLdsReadContext(self, kernel, tP) -> LdsReadTileContext:
        """Loop-order-invariant LDS-read constants for one operand (gfx1250 transpose OR
        general UnrollMajorLDS path)."""
        w = self.writer
        tc = tP["tensorChar"]

        enableLDSTr = tP.get("enableLDSTr", False)
        unrollMajor = bool(kernel["UnrollMajorLDS%s" % tc])
        if tc not in ("A", "B"):
            raise NotImplementedError(f"LeafEmitters read: tc {tc} not supported (phase 3a)")
        if not (enableLDSTr or unrollMajor):
            raise NotImplementedError("LeafEmitters read: only the LDS-transpose (enableLDSTr) "
                                      "or general UnrollMajorLDS read paths are supported")
        if not w.states.asmCaps.get("HasWMMA_V3", False):
            raise NotImplementedError("LeafEmitters read: needs HasWMMA_V3 (gfx1250)")
        # ONE BYTE OR TWO.  `bpeDS` below 1 (fp6 = 0.75, fp4 = 0.5) is a DIFFERENT read shape, not
        # a smaller one: the scaffold gives each its own branch with a padded register stride
        if tP["bpeDS"] not in (1, 2):
            raise NotImplementedError(f"LeafEmitters read: only bpeDS 1 (8-bit) and 2 (bf16) "
                                      f"supported (got {tP['bpeDS']})")
        if kernel["HalfPLR%s" % tc] or kernel["ProblemType"]["Sparse"] or kernel["numSubTiles"] > 1:
            raise NotImplementedError("LeafEmitters read: HalfPLR/Sparse/SubTiles not "
                                      "supported (phase 3a)")

        tile01 = tP["tile01Idx"]
        instruction = tP["localReadInstruction"]
        bpr = 4
        blockWidth = instruction.blockWidth
        LocalReadX = instruction.getInst(0)

        # THE ONE READER of `TDMSplitA`/`TDMSplitB`, so the read side cannot disagree with the
        # descriptor about which axis this operand is cut on.  It also absorbs the MX/sparse/
        nsplit, splitAxis = split_of(kernel, tc)

        LdsPad = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] == 0 else 0
        unrollMajorLDS = bool(kernel["UnrollMajorLDS%s" % tc])
        UnrollStride = region_row_elems(kernel["MacroTile%s" % tc], LdsPad, unrollMajorLDS,
                                        nsplit, splitAxis)
        if unrollMajorLDS:
            UnrollStride = 1

        MIWaveGroupShape = [
            kernel["MatrixInstM"] * kernel["MatrixInstBM"] * kernel["MIWaveGroup"][0] * kernel["VectorWidthA"],
            kernel["MatrixInstN"] * kernel["MatrixInstBN"] * kernel["MIWaveGroup"][1] * kernel["VectorWidthB"]]

        matrixInstT = kernel["MatrixInstM"] if (tile01 == 0) else kernel["MatrixInstN"]
        matrixInstTO = min(kernel["MatrixInstM"], kernel["MatrixInstN"])
        numTilePerInst = matrixInstT // matrixInstTO
        if numTilePerInst != 1:
            # A NON-SQUARE MATRIX INSTRUCTION puts several instruction tiles inside one wave tile,
            # and the scaffold walks them in an extra `ti` loop whose per-tile displacement
            raise NotImplementedError(
                f"LeafEmitters read: MatrixInstM != MatrixInstN (numTilePerInst="
                f"{numTilePerInst}) is not supported")
        MIInputPerThUnroll = kernel["MIInputPerThread%s" % tc] // numTilePerInst
        lrvw = kernel["LocalReadVectorWidth%s" % tc]
        wtRegStride = int(MIInputPerThUnroll * tP["bpeDS"] // bpr)
        maxLDSConstOffset = w.states.regCaps["maxLDSConstOffset"]

        try:
            fragments = read_fragments(blockWidth, tP["bpeDS"], lrvw, MIInputPerThUnroll,
                                       kernel["MatrixInstK"])
        except NotImplementedError as e:
            raise NotImplementedError(f"LeafEmitters read ({tc}): {e}")

        if unrollMajor:
            # A PACKED REGION SHORTENS THE UNROLL ROW HERE, exactly as it shortens the free row in
            # `UnrollStride` on the tile-major path -- same rule, different axis, because "the row"
            tileStrideElems = region_row_elems(kernel["_DepthU%s" % tc], LdsPad,
                                               unrollMajorLDS, nsplit, splitAxis)
        else:
            # TILE-MAJOR: the free axis is the image's INNER one, so it has stride 1 and
            # `tile_row` already returns the element position folded into its region.  The old
            # pair (row = tile index, stride = the group jump) composes to the same address only
            # while VectorWidth is 1, which is the only value LDSTrInst admits here.
            tileStrideElems = 1

        splitBoundaryBytes = 0
        if nsplit > 1:
            splitBoundaryBytes = int(w.tdmSplitLdsBoundary(kernel, tP))

        segRowShape, segCompCols, segWriteStrideBytes = self._segInterleaveTerms(
            kernel, tc, tile01, MIWaveGroupShape[tile01])

        return LdsReadTileContext(
            segRowShape=segRowShape, segCompCols=segCompCols,
            segWriteStrideBytes=segWriteStrideBytes,
            tc=tc, tile01=tile01, LocalReadX=LocalReadX, blockWidth=blockWidth,
            bpeDS=tP["bpeDS"], LdsPad=LdsPad, ldsBlockSizePerPad=kernel["LdsBlockSizePerPad%s" % tc],
            padBytes=kernel["LdsPad%s" % tc], UnrollStride=UnrollStride,
            MIWaveGroupShape=MIWaveGroupShape, localReadOffset=tP["localReadOffset"],
            wtRegStride=wtRegStride, fragments=fragments,
            substepStrideBytes=int(kernel["MatrixInstK"] * UnrollStride * tP["bpeDS"]),
            maxLDSConstOffset=maxLDSConstOffset, miWaveTileAxis=kernel["MIWaveTile"][tile01],
            macroTile=kernel["MacroTile%u" % tile01],
            enableLDSTr=enableLDSTr, tileStrideElems=tileStrideElems,
            vectorWidth=max(1, kernel["VectorWidth%s" % tc]), unrollMajor=bool(unrollMajor),
            splitBoundaryBytes=splitBoundaryBytes, nsplit=nsplit, splitAxis=splitAxis)

    def _checkRegBuffer(self, tc, bufferIdx, site):
        """INVARIANT GUARD -- not a diagnostic anything is expected to trip."""
        widths = getattr(self.writer.states, "loopModelRegBufferWidths", {}) or {}
        n = int(widths.get(tc, getattr(self.writer.states, "numVgprBuffer", 0)))
        if bufferIdx >= n:
            raise RuntimeError(
                f"LoopModel {site}: GIR named Valu{tc}_X{bufferIdx} but only X0..X{n - 1} are "
                f"allocated.  theta rotation widths are {widths or '<unset>'}.  "
                f"This is an INVARIANT break, not a tuning limit: the vgpr buffers must be sized "
                f"from theta (KernelWriter.loopModelRegBuffers). Something re-sized them from "
                f"another source -- PrefetchLocalRead+1 and ClusterLocalRead/LoopIters are the two "
                f"that previously did.")

    def emitLdsReadTile(self, kernel, tP, ctx: LdsReadTileContext, tileIdx: int,
                        kIdx: int, bufferIdx: int, iui: int = 0, memToken=None,
                        region: int = 0, regTileIdx: int = None, regUnitIdx: int = None,
                        regBase: int = None, kFlat: int = None,
                        quantum=None, gen: int = None) -> Module:
        """Emit the ds_read(s) for ONE operand M/N-tile `tileIdx` at K-substep `kIdx`.
 `memToken` -- the LDS memory-token id(s) this read CONSUMES, from GIR. Passed down
 verbatim; `None` leaves the scaffold's own derivation in place.
 `gen` -- the LDS generation GIR says this read sources, tagged for comparison with the copy's.
 """
        w = self.writer
        comp = Component.LocalRead.find(w)
        tc = ctx.tc
        self._checkRegBuffer(tc, bufferIdx, "ds_read dest")
        imod = Module("<LoopIR: ldsread %s tile %u k %u buf %u>" % (tc, tileIdx, kIdx, bufferIdx))

        if quantum is not None:
            if not quantum.emit:
                return imod
            regSlot = quantum.reg_slot
        else:
            # MANY ACTS, ONE INSTRUCTION -- the group-leader rule.  When one ds_read covers
            # `tilePerRead` tiles (the MX scales of adjacent tiles are contiguous in LDS), the act
            regSlot = None
            if ctx.tileSpanVW:
                vw = ctx.tileSpanVW
                if (tileIdx % (2 * vw)) >= vw:
                    return imod                   # upper half-wave: never loaded
                perGroup = max(1, vw // max(1, ctx.tilePerRead))
                regSlot = ((tileIdx // (2 * vw)) * perGroup
                           + (tileIdx % vw) // max(1, ctx.tilePerRead))

        def _pad(off):
            if ctx.ldsBlockSizePerPad != 0 and ctx.padBytes != 0:
                off += int((off // ctx.ldsBlockSizePerPad) * ctx.padBytes * ctx.bpeDS)
            return off

        # WHETHER THE READ OWES A REGION TERM IS DECIDED BY THE LAYOUT, and the rule lives in
        # `lds_geometry.region_bytes` -- one statement, read by both the base and the row index.
        if regTileIdx is None:
            regTileIdx = tileIdx
        # ROW INDEX, not `stride * t`: the tiles of one wave are a vector group of `VW` adjacent
        # rows and then a jump of `MIWaveGroupShape` (see `tile_row`).
        kAddr = addr_coord_on_split_axis(kIdx, kIdx if kFlat is None else kFlat, ctx)
        # The fold takes the GROUP-strided part only; the within-vector row rides outside it,
        # as `LocalRead` folds `vCols` and adds `eIdx` after.
        subRow = regTileIdx % max(1, ctx.vectorWidth)
        segRow, segOff = component_fold(ctx, tile_row(ctx, regTileIdx) - subRow)
        tileBase = int((ctx.tileStrideElems * (segRow + subRow)) * ctx.bpeDS)
        base = tileBase + kAddr * ctx.substepStrideBytes
        regionOff = region_bytes(ctx, region) + segOff
        # APPEND the GIR act to the existing path comment ("LDS Transpose" / "LDS general" says
        # WHICH READ PATH was taken and must survive); the tile/k/buffer says WHICH ACT it realizes.
        _cmt = ("<LoopIR: LDS Transpose>" if ctx.enableLDSTr else "<LoopIR: LDS general>") + \
               "  " + gir_tag("read %s[tile=%u,k=%u%s%s]->X%u" % (
                   tc, regTileIdx, kAddr,
                   "" if not ctx.splitBoundaryBytes else ",r%u" % region,
                   "" if gen is None else ",gen=%u" % gen, bufferIdx))
        # ONE ds_read PER FRAGMENT.  The list is dtype-derived (`buildLdsReadContext`), so this
        # loop is the same for bf16's two reads and fp8's four; `unrollElems` is a displacement
        for unrollElems, fragReg in ctx.fragments:
            raw = base + int(unrollElems * ctx.UnrollStride * ctx.bpeDS)
            off_split, srcAddr = comp.cal_offset_srcAddr(ctx.maxLDSConstOffset, tc,
                                                         _pad(raw) + regionOff)
            ds = DSModifiers(na=1, offset=off_split)
            # THE REGISTER BASE IS THE GROUP'S, not the tile's, when a read covers several tiles.
            # MXS unit indices name scalar lanes; its write starts at that lane. Other reads index
            # instruction-width groups and therefore scale by `wtRegStride`.
            if regUnitIdx is not None:
                regSlot = regUnitIdx
            scalar_unit = quantum is not None and tc.startswith("MXS") and regUnitIdx is not None
            regOff = ((regSlot * ctx.unitRegStride if scalar_unit else
                       ctx.wtRegStride * (regSlot if regSlot is not None
                                         else regTileIdx // ctx.tilePerRead))
                      + fragReg)
            dst = (vgpr(self._valuStart(tc) + regBase + int(regOff), ctx.blockWidth)
                   if regBase is not None else
                   vgpr("Valu%s_X%u_I%u+%u" % (tc, bufferIdx, iui, int(regOff)),
                        ctx.blockWidth))
            comp._emitLdsRead(w, kernel, tP, ctx.LocalReadX, dst=dst, src=srcAddr, ds=ds,
                              module=imod, comment=_cmt, memToken=memToken)
        return imod

    def emitCopyTile(self, kernel, tP, bufIdx: int = 0, tPM=None, memToken=None,
                     regions: int = 1, region: int = 0) -> Module:
        """Emit the global->shared copy (tensor_load / buffer_load) for ONE operand, into staging
        buffer `bufIdx`.
        """
        if regions > 1:
            # `region` matters beyond the descriptor position at NumWaves>1, where each region has
            # its own dim1 extent -- see `tdmLoadRegionGir`.
            return self.writer.tdmLoadRegionGir(kernel, tP, memToken, region=region)
        return self.writer.globalReadDo(kernel, 0, tP, g2lBufIdx=bufIdx, tPM=tPM,
                                        girMemToken=memToken)
