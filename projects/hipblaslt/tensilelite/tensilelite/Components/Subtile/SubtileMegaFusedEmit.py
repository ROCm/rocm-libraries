# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""MegaFusedEpilogue: single-pass fused ResidualAdd + PartialRMS + MXFP8Quant.

Ordering invariant enforced by the fused element loop for each (m, n, k):
  H            = acc + residual          -- residual add before gamma
  rmsPartials += H * H                   -- square pre-gamma H for RMS
  ResidualOut <- bf16(H)                 -- store pre-gamma value
  acc          = H * gamma               -- scale after squaring and store
  blkAmax[j]  = max(blkAmax, |H*gamma|) -- inline MXFP8 amax fold

Wide bf16 gamma is broadcast through LDS via DTL: contiguous waveN-zero lanes
[0, tilesPerBlockM*mfma_m//2) issue buffer_load_b32 lds (no VGPR destination) once
per qi, producing a dense lane-contiguous LDS layout; all lanes broadcast-read and
convert bf16 to f32 via ds_read_b64; ping-ponged across qi.
The scalar-gamma path retains per-lane global loads.  Residual elements are loaded
wide (one BufferLoadB64 per 4-element chunk) when useWideResidual is set; software
OOB masking corrects elements that straddle N_hidden boundaries.

All helpers (residual addressing, gamma-load, rmsSum reduction, partial-buf write,
and the MXFP8 streaming context / per-group quantisation) live on the single
SubtileMegaFusedEmitter class below.
"""

import math
import struct

from rocisa.code import Label, Module
from rocisa.container import (
    ContinuousRegister,
    DSModifiers,
    EXEC,
    MUBUFModifiers,
    accvgpr,
    mgpr,
    sgpr,
    vgpr,
)
from rocisa.enum import HighBitSel
from rocisa.functions import vectorStaticDivide
from rocisa.instruction import (
    VSubU32,
    VMed3I32,
    VCmpEQF32,
    BufferStoreB8,
    BufferLoadB32,
    BufferLoadB64,
    BufferLoadB128,
    BufferLoadD16B16,
    BufferLoadD16U8,
    BufferStoreB16,
    BufferStoreB32,
    BufferStoreB64,
    BufferStoreB128,
    DSBPermuteB32,
    DSLoadB32,
    DSLoadB64,
    DSStoreB32,
    ECvtPkBF8toF32,
    ECvtPkFP8toF32,
    SAddU32,
    SAndB64,
    SAndN2B32,
    SAndSaveExecB64,
    SBranch,
    SCBranchSCC0,
    SCmpLtU32,
    SLShiftLeftB32,
    SLShiftRightB32,
    SMovB32,
    SMovB64,
    SMulHIU32,
    SMulI32,
    SNop,
    SWaitCnt,
    VAccvgprReadB32,
    VAccvgprWriteB32,
    VAdd3U32,
    VAddF32,
    VAddPKF32,
    VAddU32,
    VAndB32,
    VCmpEQU32,
    VCmpLtU32,
    VCndMaskB32,
    VCvtBF16toFP32,
    VCvtBF8toF32,
    VCvtF16toF32,
    VCvtFP8toF32,
    VCvtPkF32toBF16,
    VLShiftLeftB32,
    VLShiftRightB32,
    VMacF32,
    VMaxF32,
    VMovB32,
    VMulF32,
    VMulLOU32,
    VMulPKF32,
    VOrB32,
    VReadfirstlaneB32,
    VXorB32,
)
from Tensile.Common.DataType import DataType


# Maximum inline-literal integer for VOP encodings; larger immediates must be
# materialized in a VGPR before a v_add.
_INLINE_CONST_MAX = 64

# Upper sanity bound on prefetch depth across the flattened (qi, colBatch) unit
# sequence.  Prefetch depth is derived (maximized within the VGPR budget), not fixed:
# deeper prefetch hides more residual-load latency, but a single HBM round-trip is
# fully covered by only a few units of compute lookahead, so beyond this depth extra
# prefetch only trades resBank VGPRs for no additional latency coverage while raising
# spill risk on heavier tiles.  Profiling found depth 4 drains the residual-load
# `s_waitcnt` stall for the reference kernel; cap here so lightweight tiles with large
# VGPR headroom (small S) do not derive an absurdly deep, register-wasteful ring.
_PREFETCH_DEPTH_CAP = 4
# Total VGPR budget for gfx950 kernels (hardware limit).
_VGPR_BUDGET = 256
# Rough estimate of VGPRs consumed by the kernel outside the epilogue emitter's own
# residual ring and acc bank.  Reverse-engineered from MT192x256 MXFP8 (§6 of
# SUBTILE_MEGAFUSED_BATCHED_REDESIGN.md): actual fixed ≈204–205 (including the new
# vPermAddr VGPR added by the dwordx4 interior path).  Raised from 192 to 212 so
# EBC derivation keeps heavy tiles (MT384x256, MT320x320) safely below 256 VGPRs.
# Lower this constant only if the allocator reports spills.
_VGPR_FIXED_ESTIMATE = 212

# Maximum representable magnitudes of the FP8 quantization output types, used
# to scale amax so K2 can requantize. e5m2/bf8 has a far larger range than e4m3.
_FP8_E4M3_MAX = 448.0        # OCP FP8 e4m3.
_FP8_E4M3_FNUZ_MAX = 240.0   # FNUZ FP8 e4m3.
_BF8_E5M2_MAX = 57344.0      # FP8 e5m2 (bf8).
_fp8E4m3Max = 448.0          # OCP FP8 e4m3 (name used by the MXFP8 quant helpers).


# Returns (magic, postShift) for unsigned floor-division by constant d via SMulHIU32.
# floor(x/d) = mulhi(x, magic) >> postShift; for ceil-div pre-add (d-1). Valid for d >= 2.
def _ceilDivMagic(d: int):
    p = (d - 1).bit_length()          # smallest p such that 2^p >= d.
    magic = -(-( 1 << (32 + p - 1)) // d)  # ceil(2^(32+p-1) / d), using integer ceiling.
    return magic & 0xFFFFFFFF, p - 1  # postShift = p-1 (mulhi already shifts by 32).

class SubtileMegaFusedEmitter:
    """Emit the MegaFusedEpilogue for the Subtile gfx950 kernel."""

    def __init__(self, writer, kernel):
        self.writer = writer
        self.kernel = kernel
        # MXFP8 quant is derived: RMSEpilogue active and D output is F8 (OCP e4m3).
        self.useMxfp8 = (bool(kernel.get("RMSEpilogue", False))
                         and kernel["ProblemType"]["DestDataType"].isFloat8())

        # ---- Residual/RMS shared geometry (snake_case) ----
        self.mfma_m = kernel["MatrixInstM"]
        self.mfma_n = kernel["MatrixInstN"]
        self.waveSize = kernel["WavefrontSize"]
        assert self.waveSize == 64, "megaFused epilogue requires wavefrontSize == 64"
        self.rows_per_lane = (self.mfma_m * self.mfma_n) // self.waveSize
        wg = kernel["MIWaveGroup"]
        self.wg_m = wg[0]
        self.wg_n = wg[1]
        self.mma_m = (kernel["MacroTile0"] // self.mfma_m) // self.wg_m
        self.mma_n = (kernel["MacroTile1"] // self.mfma_n) // self.wg_n
        self.macro_tile0 = kernel["MacroTile0"]
        self.macro_tile1 = kernel["MacroTile1"]
        self.lane_sgpr_count = writer.states.laneSGPRCount
        self.numPartials = self.mma_n

        dt = kernel["ProblemType"]["DataType"]
        # elemBytes/log2ElemBytes encode the GEMM input element size used in colByte.
        self.elemBytes = 1 if (dt.isAnyFloat8() or dt.isAnyBFloat8()) else 2
        self.log2ElemBytes = 0 if self.elemBytes == 1 else 1

        # Residual side input. RMSEpilogue is the single public knob and always fuses
        # residual-add plus the bf16 ResidualOut store (both forced on by
        # _expandRMSEpilogue), so they are unconditional here, not optional.
        self.residualType = DataType(kernel.get("RMSEpilogueResidualType") or "b")
        self.residualBytes, self.residualLog2Bytes = self._sideBytes(self.residualType)
        # Wide residual load: fp8/bf8 packs 4 per dword, bf16 packs 4 per dwordx2.
        self.useWideResidual = ((self.rows_per_lane % 4 == 0)
                                and (self.residualBytes == 1
                                     or (self.residualBytes == 2
                                         and not self.residualType.isHalf())))

        # Gamma side input.
        self.gammaType = DataType(kernel.get("RMSEpilogueGammaType") or "b")
        self.gammaBytes, self.gammaLog2Bytes = self._sideBytes(self.gammaType)
        # Wide gamma load: bf16 packs 4 per dwordx2; fp8/bf8 packs 4 per dword.
        self.useWideGamma = (self.rows_per_lane % 4 == 0
                             and (self.gammaBytes == 1
                                  or (self.gammaBytes == 2 and not self.gammaType.isHalf())))

        # ---- MXFP8 dynamic-quant geometry (camelBack); only when useMxfp8 ----
        if self.useMxfp8:
            self.tagPrefix = "mx"
            self.mfmaM = kernel["MatrixInstM"]
            self.mfmaN = kernel["MatrixInstN"]
            self.rowsPerLane = (self.mfmaM * self.mfmaN) // self.waveSize
            self.wgM = wg[0]
            self.wgN = wg[1]
            self.mmaM = (kernel["MacroTile0"] // self.mfmaM) // self.wgM
            self.mmaN = (kernel["MacroTile1"] // self.mfmaN) // self.wgN
            self.macroTile1 = kernel["MacroTile1"]
            self.q0 = 32  # MXFP8 block shape is always 32x1.
            self.q1 = 1
            self.laneSgprCount = writer.states.laneSGPRCount
            self.nQTilesM = (self.mmaM * self.mfmaM) // self.q0
            # subCol quant (q1 < mfmaN) is the only MXFP8 mode megaFused emits; subRow
            # (q0 < mfmaM) is excluded by the emit() assertion.
            self.subColQuant = self.q1 < self.mfmaN and not (self.q0 < self.mfmaM)
            if self.subColQuant:
                self.streamGroup = 4
                self.tilesPerBlockM = self.q0 // self.mfmaM

        self._initGeometry(kernel)
        # epilogueBatchCols (EBC) and prefetchDepth (PFD) are derived from the VGPR
        # budget formula after geometry is known.  prefetchDepth is set as a side-effect
        # of _deriveEpilogueBatchCols.
        self.epilogueBatchCols = self._deriveEpilogueBatchCols()

        # ---- Gamma DTL-to-LDS broadcast geometry ----
        # One buffer covers all waveM slots; a second buffer enables ping-pong across qi.
        self.numRowGroups = self.waveSize // self.mfma_n
        self.gammaLdsWaveStride = self.tilesPerBlockM * self.mfma_m * self.gammaBytes
        self.gammaLdsBufBytes = self.wg_m * self.gammaLdsWaveStride
        self.gammaBuffers = 2 if self.nQTilesM > 1 else 1
        # DTL path requires rpl==4 (consumer reads rpl bf16 = 8 bytes via ds_read_b64),
        # bf16 gamma, and the contiguous-lane constraint: numStageLanes must be even
        # (each b32 covers 2 rows) and fit within one wave.
        self.gammaLdsStaging = (self.useWideGamma
                                and self.rows_per_lane == 4
                                and self.gammaBytes == 2
                                and self.numRowGroups % 2 == 0
                                and (self.tilesPerBlockM * self.mfma_m) % 2 == 0
                                and (self.tilesPerBlockM * self.mfma_m) // 2 <= self.waveSize)

        # ---- Shared registers allocated across the emission (init None) ----
        self.laneId = None
        self.colByte = None
        self.col = None
        self.rowGroup = None
        self.rowGroupOff = None
        self.wgRowBase = None
        self.nhBase = None
        self.partials = None
        self.waveIdV = None
        self.resSrd = None
        self.residualOutSrd = None
        self.gammaSrd = None
        self.savedExec = None
        self.laneMaskSgpr = None
        self.mxSrd = None
        self.resTokenBase = None
        self.resRowByteBase = None
        self.resAddr = None
        self.resOobV = None
        self.resOobMask = None
        self.roRowBase = None
        self.roColByteBase = None
        # dwordx4 interior path: partner-lane byte address (VGPR) and exec masks (SGPR pairs).
        self.vPermAddr = None
        self.pairLowerLaneMask = None
        self.pairUpperLaneMask = None
        # Shared per-tile nhInRange predicate (tail-wide path only); None elsewhere.
        self.nhInRangeMask = None
        # Gamma DTL broadcast: two VGPRs and two SGPRs; allocated only when gammaLdsStaging.
        self.gammaDtlVaddr = None
        self.gammaLdsReadAddr = None
        self.gammaSoffsetSgpr = None
        self.gammaM0Base = None

    def _initGeometry(self, kernel):
        """Derive MegaFused tile geometry (camelBack) shared by orchestration."""
        mfmaM = kernel["MatrixInstM"]
        mfmaN = kernel["MatrixInstN"]
        waveSize = kernel["WavefrontSize"]
        wg = kernel["MIWaveGroup"]
        wgM, wgN = wg[0], wg[1]
        self.mfmaM = mfmaM
        self.mfmaN = mfmaN
        self.rowsPerLane = (mfmaM * mfmaN) // waveSize
        self.mmaN = (kernel["MacroTile1"] // mfmaN) // wgN
        self.wgM = wgM
        self.streamGroup = 4
        if self.useMxfp8:
            self.tilesPerBlockM = self.tilesPerBlockM if self.subColQuant else 2
            self.streamGroup = self.streamGroup if self.subColQuant else 4
            return
        mmaM = (kernel["MacroTile0"] // mfmaM) // wgM
        self.tilesPerBlockM = 2 if mmaM % 2 == 0 else 1
        self.nQTilesM = mmaM // self.tilesPerBlockM

    def _deriveEpilogueBatchCols(self) -> int:
        """Maximize prefetchDepth within the §6 VGPR budget; derive epilogueBatchCols.

        S = tilesPerBlockM * rowsPerLane is the bank size per column in the ring.
        EBC_max(PFD) = clamp(floor((VGPR_BUDGET - VGPRFixed) / (S * (1 + PFD))), 1, mmaN).
        Prefetch depth is the primary latency-hiding lever (profiling P1), so grow PFD
        upward from 1 while EBC_max(PFD) >= 1 still fits, capped at _PREFETCH_DEPTH_CAP,
        rather than fixing PFD and only shrinking EBC: once EBC is at its floor, leftover
        budget funds deeper prefetch instead of going unused.  EBC follows as the largest
        column batch fitting at the chosen depth.  If even PFD=1 yields EBC=0 the tile
        shape is infeasible and the solution is rejected.
        """
        s = self.tilesPerBlockM * self.rowsPerLane
        budget = _VGPR_BUDGET - _VGPR_FIXED_ESTIMATE
        bestPfd = 0
        bestEbc = 0
        # EBC_max is monotonically non-increasing in PFD, so the first infeasible depth
        # ends the search; the deepest feasible PFD (up to the cap) wins, with EBC as the
        # largest batch still fitting there.  Ring cost bestEbc*S*(1+bestPfd) <= budget by
        # construction, so peak VGPRs never exceed the budget envelope.
        for pfd in range(1, _PREFETCH_DEPTH_CAP + 1):
            ebc = budget // (s * (1 + pfd))
            if ebc < 1:
                break
            bestPfd = pfd
            bestEbc = min(ebc, self.mmaN)
        if bestPfd == 0:
            # EBC=0 even at PFD=1 means S*(1+1) > budget; this tile shape does not fit.
            raise RuntimeError(
                f"megaFused epilogue infeasible: EBC=0 at PFD=1 for this tile shape "
                f"(S={s}, mmaN={self.mmaN}, VGPRFixed est={_VGPR_FIXED_ESTIMATE})"
            )
        # Never prefetch past the last unit; always keep at least depth 1.
        numUnits = self.nQTilesM * math.ceil(self.mmaN / bestEbc)
        self.prefetchDepth = max(1, min(bestPfd, numUnits - 1))
        return bestEbc

    @staticmethod
    def _isPackPair(a, b):
        """True when a,b are a consecutive even-aligned VGPR pair for packed VALU."""
        return (a % 2 == 0) and (b == a + 1)


    def _useDwordx4Interior(self) -> bool:
        """Return True when the interior arm may use dwordx4 residual load/store.

        Only bf16 residual qualifies; fp8 residual stays dwordx2 (design §4).
        Excluded for MXFP8 kernels which are SGPR-constrained (the 4 extra pair-mask
        SGPRs push them over the gfx950 limit of 104 SGPRs).
        Excluded for partial-accumulation modes where this epilogue does not run
        on final values (resolves design §8.5).
        """
        if self.residualBytes != 2 or self.useMxfp8:
            return False
        partialModes = ("MultipleBuffer", "MultipleBufferSingleKernel")
        return self.kernel.get("_GlobalAccumulation") not in partialModes


    def _allocSharedRegs(self) -> None:
        """Check out shared VGPRs that stay live for the whole emission.

        SGPRs for SRDs, exec, and lane mask are allocated in _buildAndFreeSrds and
        freed in _freeSharedRegs so they remain live across the fused element loop.
        """
        vgprPool = self.writer.vgprPool
        self.laneId      = vgprPool.checkOut(1, tag="mf_laneId")
        self.colByte     = vgprPool.checkOut(1, tag="mf_colByte")
        self.col         = vgprPool.checkOut(1, tag="mf_col")
        self.rowGroup    = vgprPool.checkOut(1, tag="mf_rowGroup")
        self.rowGroupOff = vgprPool.checkOut(1, tag="mf_rowGroupOff")
        self.wgRowBase   = vgprPool.checkOut(1, tag="mf_wgRowBase")
        self.nhBase      = vgprPool.checkOut(1, tag="mf_nhBase")
        self.partials      = vgprPool.checkOut(self.mmaN, tag="mf_rmsSum")
        if self.wgM > 1:
            self.waveIdV = vgprPool.checkOut(1, tag="mf_waveIdV")
        # vPermAddr is allocated only for the bf16 dwordx4 interior path.
        if self._useDwordx4Interior():
            self.vPermAddr = vgprPool.checkOut(1, tag="mf_vPermAddr")
        # DTL gamma broadcast: producer vaddr and consumer read base (loop-invariant).
        if self.gammaLdsStaging:
            self.gammaDtlVaddr    = vgprPool.checkOut(1, tag="mf_gammaDtlVaddr")
            self.gammaLdsReadAddr = vgprPool.checkOut(1, tag="mf_gammaLdsReadAddr")


    def _freeSharedRegs(self) -> None:
        """Return shared VGPRs and SGPRs to their pools in reverse allocation order."""
        sgprPool = self.writer.sgprPool
        vgprPool = self.writer.vgprPool
        # Free SGPRs kept live across the fused loop (reverse allocation order).
        if self.gammaM0Base is not None:
            sgprPool.checkIn(self.gammaM0Base)
        if self.gammaSoffsetSgpr is not None:
            sgprPool.checkIn(self.gammaSoffsetSgpr)
        if self.pairUpperLaneMask is not None:
            sgprPool.checkIn(self.pairUpperLaneMask)
        if self.pairLowerLaneMask is not None:
            sgprPool.checkIn(self.pairLowerLaneMask)
        if self.mxSrd is not None:
            sgprPool.checkIn(self.mxSrd)
        sgprPool.checkIn(self.laneMaskSgpr)
        sgprPool.checkIn(self.savedExec)
        sgprPool.checkIn(self.gammaSrd)
        if self.resSrd is not None:
            sgprPool.checkIn(self.resSrd)
        # Free VGPRs in reverse allocation order.
        if self.gammaLdsReadAddr is not None:
            vgprPool.checkIn(self.gammaLdsReadAddr)
        if self.gammaDtlVaddr is not None:
            vgprPool.checkIn(self.gammaDtlVaddr)
        if self.vPermAddr is not None:
            vgprPool.checkIn(self.vPermAddr)
        if self.waveIdV is not None:
            vgprPool.checkIn(self.waveIdV)
        vgprPool.checkIn(self.partials)
        vgprPool.checkIn(self.nhBase)
        vgprPool.checkIn(self.wgRowBase)
        vgprPool.checkIn(self.rowGroupOff)
        vgprPool.checkIn(self.rowGroup)
        vgprPool.checkIn(self.col)
        vgprPool.checkIn(self.colByte)
        vgprPool.checkIn(self.laneId)


    def _setupShared(self) -> Module:
        """Emit shared setup: drain waits, lane arithmetic, colByte, col, rowGroup, SRDs.

        SGPRs for SRDs, exec, and lane mask are allocated here and remain live
        until _freeSharedRegs to avoid re-building them per tile row.
        """
        module = Module("MegaFused shared setup")
        module.addComment0("MF begin _setupShared: drain waits, lane arithmetic, colByte, col, rowGroup, SRDs.")
        module.add(SWaitCnt(kmcnt=0, comment="drain kernarg s_loads before reading kernel args."))
        module.add(SWaitCnt(vlcnt=0, comment="drain GEMM vector-memory before AGPR reuse."))
        mfmaN, waveSize = self.mfma_n, self.waveSize
        log2N = int(math.log2(mfmaN))
        module.add(VAndB32(dst=vgpr(self.laneId), src0=vgpr("Serial"), src1=waveSize - 1,
                           comment="laneId = Serial & (waveSize-1)."))
        if self.wgM > 1:
            waveIdTmp = self.writer.vgprPool.checkOutAligned(2, 2, tag="mf_waveIdDiv")
            module.add(vectorStaticDivide(self.waveIdV, "Serial", waveSize,
                                         ContinuousRegister(waveIdTmp, 2),
                                         comment="waveId = Serial / waveSize."))
            self.writer.vgprPool.checkIn(waveIdTmp)
        # col is the raw free1 column index used by the MXScale path.
        module.add(VAndB32(dst=vgpr(self.col), src0=vgpr(self.laneId), src1=mfmaN - 1,
                           comment="col = laneId & (mfmaN-1)."))
        # colByte encodes the token index as col * elemBytes; wave/wg offsets added below.
        module.add(VLShiftLeftB32(dst=vgpr(self.colByte), shiftHex=hex(self.log2ElemBytes),
                                  src=vgpr(self.col), comment="colByte = col * elemBytes."))
        module.add(VLShiftRightB32(dst=vgpr(self.rowGroup), shiftHex=hex(log2N),
                                   src=vgpr(self.laneId), comment="rowGroup = laneId >> log2(mfmaN)."))
        self._addWaveNColByte(module, self.colByte)
        with self.writer.allocTmpSgpr(1, tag="mf_wg1ColByte") as wg1S:
            wg1Bytes = self.macro_tile1 * self.elemBytes
            module.add(SMulI32(dst=sgpr(wg1S.idx), src0=sgpr("WorkGroup1"), src1=wg1Bytes,
                               comment=f"wg1ColByte = WorkGroup1 * MT1*elemBytes ({wg1Bytes})."))
            module.add(VAddU32(dst=vgpr(self.colByte), src0=vgpr(self.colByte), src1=sgpr(wg1S.idx),
                               comment="colByte += WorkGroup1 * MT1 * elemBytes."))
        self._buildAndFreeSrds(module)
        module.addComment0("MF end _setupShared.")
        return module


    def _buildAndFreeSrds(self, module) -> None:
        """Allocate shared SRD SGPRs and emit build instructions.

        SRDs are kept live across the entire fused element loop so helpers such as
        _subColStoreGroup can reference them without re-building per tile.
        """
        module.addComment1("MF begin _buildAndFreeSrds: allocate and build shared SRD SGPRs.")
        sgprPool = self.writer.sgprPool
        lsc = self.writer.states.laneSGPRCount
        self.resSrd = sgprPool.checkOutAligned(4, 4, tag="mf_resSrd", preventOverflow=False)
        self._buildResidualSrd(module, self.resSrd)
        # ResidualOut aliases the (beta=0 unused) SrdC named SGPR, so its descriptor
        # must be built here; otherwise bf16(H) stores target the stale C buffer.
        self.residualOutSrd = self.writer.sgprs["SrdResidualOut"]
        self._buildResidualOutSrd(module, self.residualOutSrd)
        self.gammaSrd  = sgprPool.checkOutAligned(4, 4, tag="mf_gammaSrd", preventOverflow=False)
        self.savedExec = sgprPool.checkOutAligned(lsc, lsc, tag="mf_savedExec", preventOverflow=False)
        self.laneMaskSgpr  = sgprPool.checkOutAligned(lsc, lsc, tag="mf_laneMask", preventOverflow=False)
        self._buildBufferSrd(module, self.gammaSrd, "RMSNormGamma", "gamma")
        # MXScale SRD is only needed when MXFP8 dynamic quant is active.
        if self.useMxfp8:
            self.mxSrd = sgprPool.checkOutAligned(4, 4, tag="mf_mxSrd", preventOverflow=False)
            self._buildBufferSrd(module, self.mxSrd, "MXScale", "mxScale")
        # dwordx4 pair masks: pairLower = lanes {0-15, 32-47}, pairUpper = the complement.
        if self._useDwordx4Interior():
            self.pairLowerLaneMask = sgprPool.checkOutAligned(
                lsc, lsc, tag="mf_pairLower", preventOverflow=False)
            self.pairUpperLaneMask = sgprPool.checkOutAligned(
                lsc, lsc, tag="mf_pairUpper", preventOverflow=False)
        # DTL gamma broadcast: per-wave soffset (wgRowBase*gammaBytes) and M0 wave base.
        if self.gammaLdsStaging:
            self.gammaSoffsetSgpr = sgprPool.checkOutAligned(
                1, 1, tag="mf_gammaSoffset", preventOverflow=False)
            self.gammaM0Base = sgprPool.checkOutAligned(
                1, 1, tag="mf_gammaM0Base", preventOverflow=False)
        module.addComment1("MF end _buildAndFreeSrds.")


    def _initRmsSum(self) -> Module:
        """Zero-initialise the rmsSum bank once before the fused sweep.

        Zeroing upfront avoids a first-element VMulF32 / is-first-element branch
        in the inner loop: VMacF32 then works uniformly across all iterations.
        """
        module = Module("MegaFused initRmsSum")
        module.addComment0("MF begin _initRmsSum: zero-initialise rmsSum bank before the fused sweep.")
        for n in range(self.mmaN):
            module.add(VMovB32(dst=vgpr(self.partials + n), src=0,
                               comment=f"rmsSum[{n}] = 0.0f."))
        module.addComment0("MF end _initRmsSum.")
        return module


    def _loadGammaBlockWide(self, module, gammaBank, qi, gammaByteV, mBaseV) -> None:
        """Issue wide (dwordx2) gamma loads for all tilesPerBlockM rows of quant-tile qi.

        gammaBank is 2-aligned so gammaBank + mi*rpl is always 2-aligned for dwordx2.
        """
        module.addComment1(f"MF begin _loadGammaBlockWide: wide dwordx2 gamma loads (qi={qi}).")
        rpl = self.rowsPerLane
        tpb = self.tilesPerBlockM
        for mi in range(tpb):
            m = qi * tpb + mi
            self._free0RowPos(module, self.nhBase, self.wgRowBase, self.rowGroupOff, m, 0, mBaseV)
            module.add(VLShiftLeftB32(dst=vgpr(gammaByteV), shiftHex=hex(self.gammaLog2Bytes),
                                      src=vgpr(self.nhBase),
                                      comment=f"gammaByte = nhBase * gammaBytes (mi={mi})."))
            module.add(BufferLoadB64(vgpr(gammaBank + mi * rpl, 2), vgpr(gammaByteV),
                                     sgpr(self.gammaSrd, 4), 0, MUBUFModifiers(offen=True),
                                     comment=f"gamma[nhBase..+3] dwordx2 (m={m})."))
        module.add(SWaitCnt(vlcnt=0, comment="wait wide gamma loads."))
        for mi in range(tpb):
            self._convertGammaChunkBf16(module, gammaBank + mi * rpl)
        module.addComment1("MF end _loadGammaBlockWide.")


    def _loadGammaBlockScalar(self, module, gammaBank, qi, gammaByteV, mBaseV) -> None:
        """Issue one scalar buffer_load per gamma element for quant-tile qi."""
        module.addComment1(f"MF begin _loadGammaBlockScalar: scalar gamma loads (qi={qi}).")
        rpl = self.rowsPerLane
        tpb = self.tilesPerBlockM
        for mi in range(tpb):
            m = qi * tpb + mi
            self._free0RowPos(module, self.nhBase, self.wgRowBase, self.rowGroupOff, m, 0, mBaseV)
            for k in range(rpl):
                r = self._addImmU32(module, gammaByteV, self.nhBase, k, mBaseV,
                                   f"gammaIdx = nhBase + {k} (mi={mi},k={k}).")
                module.add(VLShiftLeftB32(dst=vgpr(gammaByteV), shiftHex=hex(self.gammaLog2Bytes),
                                          src=vgpr(r),
                                          comment="gammaByte = gammaIdx * gammaBytes."))
                self._issueSideLoad(module, gammaBank + mi * rpl + k, gammaByteV, self.gammaSrd,
                                   f"gamma[m={m},k={k}].", dtype=self.gammaType)
        module.add(SWaitCnt(vlcnt=0, comment="wait gamma loads."))
        for mi in range(tpb):
            for k in range(rpl):
                self._convertSideElem(module, gammaBank + mi * rpl + k,
                                     f"gamma->fp32 (mi={mi},k={k}).", dtype=self.gammaType)
        module.addComment1("MF end _loadGammaBlockScalar.")


    def _loadGammaBlock(self, gammaBank, qi) -> Module:
        """Load and convert gamma for the tilesPerBlockM rows of quant-tile qi.

        Gamma is per free0 row and independent of the free1 (N) sweep, so it is
        loaded once per qi and reused across all N-groups.
        """
        module = Module(f"MegaFused loadGammaBlock qi={qi}")
        module.addComment0(f"MF begin _loadGammaBlock: load and convert gamma for qi={qi}.")
        gammaByteV = self.writer.vgprPool.checkOut(1, tag="mf_gammaByte")
        mBaseV     = self.writer.vgprPool.checkOut(1, tag="mf_gammaM")
        if self.useWideGamma:
            self._loadGammaBlockWide(module, gammaBank, qi, gammaByteV, mBaseV)
        else:
            self._loadGammaBlockScalar(module, gammaBank, qi, gammaByteV, mBaseV)
        self.writer.vgprPool.checkIn(mBaseV)
        self.writer.vgprPool.checkIn(gammaByteV)
        module.addComment0("MF end _loadGammaBlock.")
        return module


    def _beginResidualScratch(self, module) -> None:
        """Allocate per-element residual scratch VGPRs and compute invariants.

        Scratch registers are kept live across the entire N-sweep and freed in
        _endResidualScratch. resOobV and resTokenBase are needed for both the
        residual load path and the inline bf16 store path.
        """
        module.addComment1("MF begin _beginResidualScratch: allocate residual scratch and compute invariants.")
        writer = self.writer
        self.resTokenBase   = writer.vgprPool.checkOut(1, tag="mf_resTokenBase")
        self.resRowByteBase = writer.vgprPool.checkOut(1, tag="mf_resRowByteBase")
        self.resAddr        = writer.vgprPool.checkOut(1, tag="mf_resAddr")
        self.resOobV        = writer.vgprPool.checkOut(1, tag="mf_resOobV")
        self.resOobMask     = writer.sgprPool.checkOutAligned(
            self.lane_sgpr_count, self.lane_sgpr_count, tag="mf_resOobMask", preventOverflow=False)
        # resTokenBase = colByte >> log2ElemBytes; used by residual loads and bf16 store.
        module.add(VLShiftRightB32(dst=vgpr(self.resTokenBase),
                                   shiftHex=hex(self.log2ElemBytes),
                                   src=vgpr(self.colByte),
                                   comment="resTokenBase = colByte >> log2ElemBytes."))
        module.add(VMovB32(dst=vgpr(self.resOobV), src="BufferOOB",
                           comment="resOobV = BufferOOB (OOB loads return 0 / stores dropped)."))
        module.addComment1("MF end _beginResidualScratch.")


    def _endResidualScratch(self, module) -> None:
        """Wait for pending ResidualOut stores and free residual scratch registers."""
        module.addComment1("MF begin _endResidualScratch: drain ResidualOut stores, free scratch registers.")
        writer = self.writer
        module.add(SWaitCnt(vscnt=0, comment="wait ResidualOut bf16 stores."))
        writer.sgprPool.checkIn(self.resOobMask)
        writer.vgprPool.checkIn(self.resOobV)
        writer.vgprPool.checkIn(self.resAddr)
        writer.vgprPool.checkIn(self.resRowByteBase)
        writer.vgprPool.checkIn(self.resTokenBase)
        module.addComment1("MF end _endResidualScratch.")


    def _computeBf16Addr(self, module, n, k, addrV, valV, nhByteV, nhMaskIdx) -> None:
        """Compute clamped byte address for ResidualOut[token_n, nhPos] at column (n, k).

        Token-OOB lanes are dropped by the ResidualOut SRD bounds, so no explicit
        token mask is applied here; only nhPos-straddle elements are clamped to
        BufferOOB to avoid aliasing the next token's row.
        When nhInRangeMask is set the per-k predicate was already computed by
        _maskWideResidualOOB and is reused directly; otherwise it is computed here.
        """
        module.addComment1(f"MF begin _computeBf16Addr: clamped byte address for ResidualOut (n={n},k={k}).")
        lsc = self.lane_sgpr_count
        module.add(VLShiftLeftB32(dst=vgpr(addrV), shiftHex=hex(1), src=vgpr(self.roRowBase),
                                  comment="base0 = roRowBase * 2 (bf16); token_n*N_hidden reused."))
        nh = self._addImmU32(module, nhByteV, self.nhBase, k, valV,
                            f"nhPos = nhBase + {k} (k={k}).")
        if self.nhInRangeMask is not None:
            maskReg = self.nhInRangeMask + k * lsc
        else:
            module.add(VCmpLtU32(dst=sgpr(nhMaskIdx, lsc), src0=vgpr(nh),
                                 src1=sgpr("SizesFree+0"),
                                 comment="nhInRange = nhPos < N_hidden."))
            maskReg = nhMaskIdx
        module.add(VLShiftLeftB32(dst=vgpr(nhByteV), shiftHex=hex(1), src=vgpr(nh),
                                  comment="nhByte = nhPos * 2 (bf16)."))
        module.add(VAddU32(vgpr(addrV), vgpr(addrV), vgpr(nhByteV),
                           comment="byteAddr = base0 + nhByte."))
        module.add(VCndMaskB32(dst=vgpr(addrV), src0=vgpr(self.resOobV), src1=vgpr(addrV),
                               src2=sgpr(maskReg, lsc),
                               comment="clamp OOB when nhPos >= N_hidden."))
        module.addComment1("MF end _computeBf16Addr.")


    def _computeResidualOutRowBaseAndMask(self, module, n, tokMaskSgpr) -> None:
        """Per-N-column setup: token(N) OOB mask and roRowBase = token_n * N_hidden.

        token_n is the free1 index owned by each lane and is constant across all m and k
        within the N-group, so both the mask and the row base are computed once per n and
        reused. Mirrors beta*C's GWB store, which computes its address base once rather than
        multiplying per tile. self.roRowBase and self.roColByteBase must be checked out by the caller.
        """
        module.addComment1(f"MF begin _computeResidualOutRowBaseAndMask: token OOB mask and roRowBase (n={n}).")
        lsc = self.lane_sgpr_count
        nOff = n * self.mfma_n
        tokV = self.writer.vgprPool.checkOut(1, tag="mf_roTokV")
        scratch = self.writer.vgprPool.checkOut(1, tag="mf_roTokScratch")
        r = self._addImmU32(module, tokV, self.resTokenBase, nOff, scratch,
                           f"token_n = resTokenBase + {nOff} (n={n}).")
        module.add(VCmpLtU32(dst=sgpr(tokMaskSgpr, lsc), src0=vgpr(r),
                             src1=sgpr("SizesFree+1"),
                             comment="tokenInRange = token_n < M_tokens (ResidualOut N mask)."))
        module.add(VMulLOU32(dst=vgpr(self.roRowBase), src0=sgpr("SizesFree+0"), src1=vgpr(r),
                             comment=f"roRowBase = token_n * N_hidden (n={n}); reused across m,k."))
        # Fold the per-lane invariant row origin into a byte base so the per-tile
        # store only needs a compile-time offset12 (no per-tile address VALU).
        module.add(VAddU32(dst=vgpr(self.roColByteBase), src0=vgpr(self.roRowBase),
                           src1=vgpr(self.wgRowBase),
                           comment="roColBase = token_n*N_hidden + wgRowBase."))
        module.add(VAddU32(dst=vgpr(self.roColByteBase), src0=vgpr(self.roColByteBase),
                           src1=vgpr(self.rowGroupOff),
                           comment="roColBase += rowGroupOff (per-lane row origin)."))
        module.add(VLShiftLeftB32(dst=vgpr(self.roColByteBase), shiftHex=hex(1),
                                  src=vgpr(self.roColByteBase),
                                  comment="roColByteBase = roColBase * 2 (bf16)."))
        self.writer.vgprPool.checkIn(scratch)
        self.writer.vgprPool.checkIn(tokV)
        module.addComment1("MF end _computeResidualOutRowBaseAndMask.")


    def _packResidualOutRow(self, module, srcRegs, packBank) -> None:
        """Pack rpl bf16(H) values into packBank (rpl/2 dwords, 2-aligned) for a dwordx2 store."""
        module.addComment1("MF begin _packResidualOutRow: pack rpl bf16(H) values into dwordx2 store bank.")
        rpl = self.rowsPerLane
        for p in range(rpl // 2):
            module.add(VCvtPkF32toBF16(dst=vgpr(packBank + p),
                                       src0=vgpr(srcRegs[2 * p]), src1=vgpr(srcRegs[2 * p + 1]),
                                       comment=f"pack H[{2 * p}] lo16, H[{2 * p + 1}] hi16 -> bf16x2."))
        module.addComment1("MF end _packResidualOutRow.")


    def _storeResidualOutRow(self, module, srcRegs, tokMaskSgpr, m, n) -> None:
        """Store rpl bf16(H) to ResidualOut as one dwordx2 for interior lanes, under FULL exec.

        Interior (b64Safe) lanes use BufferStoreB64 with the real address; straddling lanes
        get BufferOOB so the SRD drops their wide store.  The per-element fallback clamps
        non-straddle lanes' address to BufferOOB so only straddle lanes store.  A scalar SCC
        branch still skips the fallback when no lane straddles.  Exec stays full throughout
        and does not need to be restored before returning.
        """
        module.addComment1(f"MF begin _storeResidualOutRow: pack and store rpl bf16(H) to ResidualOut (m={m},n={n}).")
        rpl = self.rowsPerLane
        assert rpl % 2 == 0, "rpl must be even for dwordx2 bf16 packing"
        lsc = self.lane_sgpr_count
        # gfx950 is wave64-only for this path; HasWave32 excludes gfx9,
        # _validateSubtileEpiloguePrereqs rejects non-gfx950.
        assert lsc == 2, "storeResidualOutRow hardcodes wave64 b64 exec ops"
        vgprPool = self.writer.vgprPool
        sgprPool = self.writer.sgprPool
        packBank = vgprPool.checkOutAligned(rpl // 2, 2, tag="mf_roPack")
        nhTopV   = vgprPool.checkOut(1, tag="mf_roNhTop")
        self._packResidualOutRow(module, srcRegs, packBank)
        safe      = sgprPool.checkOutAligned(lsc, lsc, tag="mf_roSafe", preventOverflow=False)
        wideAddrV = vgprPool.checkOut(1, tag="mf_roWideAddr")
        self._issueResidualOutWide(module, m, n, packBank, nhTopV, safe, wideAddrV)
        self._issueResidualOutStraddle(module, tokMaskSgpr, m, n, srcRegs, safe)
        vgprPool.checkIn(wideAddrV)
        sgprPool.checkIn(safe)
        vgprPool.checkIn(nhTopV)
        vgprPool.checkIn(packBank)
        module.addComment1("MF end _storeResidualOutRow.")


    def _issueResidualOutWide(self, module, m, n, packBank,
                               nhTopV, safeIdx, wideAddrV) -> None:
        """Full-exec dwordx2 store with BufferOOB clamp for straddling lanes.

        safeIdx receives b64Safe (nhTop < N_hidden) and is consumed unchanged by
        _issueResidualOutStraddle to compute the straddle subset.  Straddling lanes
        get BufferOOB so the SRD drops their wide store; token-OOB lanes are also
        silently dropped by the SRD bounds.  wideAddrV holds the per-lane address for
        this store; it must not be roColByteBase.  The per-tile row offset m*mfma_m*2
        is a compile-time constant folded into offset12.
        """
        module.addComment1(f"MF begin _issueResidualOutWide: full-exec dwordx2 store with BufferOOB clamp (m={m},n={n}).")
        rpl = self.rowsPerLane
        lsc = self.lane_sgpr_count
        nhTop = self._addImmU32(module, nhTopV, self.nhBase, rpl - 1, nhTopV,
                               f"nhTop = nhBase + {rpl - 1}.")
        # b64Safe = interior lanes whose whole rpl-group is < N_hidden (no straddle).
        # Token-OOB lanes are dropped by the ResidualOut SRD bounds, so no token mask
        # is folded into the safe mask here (relies on SRD OOB clamping).
        module.add(VCmpLtU32(dst=sgpr(safeIdx, lsc), src0=vgpr(nhTop),
                             src1=sgpr("SizesFree+0"),
                             comment="b64Safe = nhBase+rpl-1 < N_hidden (no straddle)."))
        # GlobalWriteBatch-style: clamp straddling lanes' wide-store address to
        # BufferOOB and store under FULL exec (the SRD drops OOB), instead of
        # narrowing exec.  b64Safe lanes keep the real address; token-OOB lanes are
        # dropped by the SRD regardless.  BufferOOB + the compile-time offset12 stays OOB.
        module.add(VCndMaskB32(dst=vgpr(wideAddrV), src0=vgpr(self.resOobV),
                               src1=vgpr(self.roColByteBase), src2=sgpr(safeIdx, lsc),
                               comment="wideAddr = b64Safe ? roColByteBase : BufferOOB."))
        rowOff = m * self.mfma_m * 2
        assert rowOff < 4096, f"residualOut row offset {rowOff} exceeds MUBUF offset12 range"
        module.add(BufferStoreB64(src=vgpr(packBank, 2), vaddr=vgpr(wideAddrV),
                                  saddr=sgpr(self.residualOutSrd, 4), soffset=0,
                                  mubuf=MUBUFModifiers(offen=True, offset12=rowOff),
                                  comment=f"ResidualOut dwordx2 (m={m},n={n}) off={rowOff} (full exec, straddle OOB)."))
        module.addComment1("MF end _issueResidualOutWide.")


    def _issueResidualOutStraddle(self, module, tokMaskSgpr, m, n, srcRegs,
                                   safeIdx) -> None:
        """Run the per-element fallback under FULL exec for straddle lanes only.

        safeIdx on entry holds b64Safe from _issueResidualOutWide and is overwritten with
        the straddle mask (tokMask AND NOT b64Safe); SCC is set if any straddle lane
        exists.  The scalar SCC branch skips the fallback when no lane straddles.  Each
        per-element store clamps non-straddle lanes' address to BufferOOB so the SRD
        drops them; exec stays full throughout and is not restored on return.
        """
        module.addComment1(f"MF begin _issueResidualOutStraddle: full-exec per-element fallback for straddle lanes (m={m},n={n}).")
        lsc = self.lane_sgpr_count
        # Straddle mask: tokMask AND NOT b64Safe.
        # Two SAndN2B32 reuse the safe register pair; SAndB64 sets SCC for the branch.
        module.add(SAndN2B32(dst=sgpr(safeIdx), src0=sgpr(tokMaskSgpr), src1=sgpr(safeIdx),
                             comment="straddle_lo = tokMask_lo & ~b64Safe_lo."))
        module.add(SAndN2B32(dst=sgpr(safeIdx + 1), src0=sgpr(tokMaskSgpr + 1),
                             src1=sgpr(safeIdx + 1),
                             comment="straddle_hi = tokMask_hi & ~b64Safe_hi."))
        module.add(SAndB64(dst=sgpr(safeIdx, lsc), src0=sgpr(safeIdx, lsc),
                           src1=sgpr(safeIdx, lsc),
                           comment="SCC = (straddle != 0); safe still holds straddle mask."))
        skipLabel = Label(self.writer.labels.getNameInc(f"mf_roStraddleEnd_m{m}n{n}"), "")
        module.add(SCBranchSCC0(labelName=skipLabel.getLabelName(),
                                comment="no straddle lanes -> skip per-element fallback."))
        # GlobalWriteBatch-style: per-element fallback runs under FULL exec; each element
        # clamps non-straddle lanes' address to BufferOOB so only straddle lanes store.
        for k in range(self.rowsPerLane):
            self._storeBf16ElemInline(module, srcRegs[k], m, n, k, safeIdx)
        module.add(skipLabel)
        module.addComment1("MF end _issueResidualOutStraddle.")


    def _storeBf16ElemInline(self, module, accReg, m, n, k, straddleMaskSgpr) -> None:
        """Store bf16(accReg) to ResidualOut[token_n, nhidden_pos] under full exec.

        Stores only straddle lanes: _computeBf16Addr clamps nhPos-OOB lanes to BufferOOB,
        then an additional clamp drops non-straddle lanes so only straddle lanes write.
        """
        module.addComment1(f"MF begin _storeBf16ElemInline: inline masked bf16(H) store (m={m},n={n},k={k}).")
        lsc = self.lane_sgpr_count
        addrV   = self.writer.vgprPool.checkOut(1, tag="mf_bf16Addr")
        valV    = self.writer.vgprPool.checkOut(1, tag="mf_bf16Val")
        nhByteV = self.writer.vgprPool.checkOut(1, tag="mf_nhByte")
        with self.writer.allocTmpSgpr(lsc, tag="mf_nhMask") as nhMask:
            self._computeBf16Addr(module, n, k, addrV, valV, nhByteV, nhMask.idx)
            # Store only straddle lanes; non-straddle lanes (already wide-stored, or
            # token-OOB) -> BufferOOB so the store is a no-op under full exec.
            module.add(VCndMaskB32(dst=vgpr(addrV), src0=vgpr(self.resOobV),
                                   src1=vgpr(addrV), src2=sgpr(straddleMaskSgpr, lsc),
                                   comment="store only straddle lanes; others -> BufferOOB."))
            module.add(VCvtPkF32toBF16(dst=vgpr(valV), src0=vgpr(accReg),
                                        src1=vgpr(accReg),
                                        comment="H -> bf16 (low 16 bits)."))
            module.add(BufferStoreB16(src=vgpr(valV), vaddr=vgpr(addrV),
                                      saddr=sgpr(self.residualOutSrd, 4), soffset=0,
                                      mubuf=MUBUFModifiers(offen=True),
                                      comment=f"ResidualOut bf16(H) (m={m},n={n},k={k})."))
        self.writer.vgprPool.checkIn(nhByteV)
        self.writer.vgprPool.checkIn(valV)
        self.writer.vgprPool.checkIn(addrV)
        module.addComment1("MF end _storeBf16ElemInline.")


    def _issueResidualWide(self, module, m, n, burstBase) -> None:
        """Issue wide residual load(s) for tile (m, n); caller must SWaitCnt(vlcnt=0) after.

        self.resRowByteBase holds (token_n*N_hidden + wgRowBase + rowGroupOff)*residualBytes
        for this n (row origin already folded by _residualRowByteBase when useWideResidual).
        Each chunk is addressed via a compile-time offset12 = m*mfma_m*bytes + chunkBytes*c,
        matching the ResidualOut store's base+offset12 model. No nhBase needed.
        Issues one BufferLoadB64 (bf16) or BufferLoadB32 (fp8) per 4-element chunk.
        """
        module.addComment1(f"MF begin _issueResidualWide: wide residual load (m={m},n={n}).")
        isBf16 = self.residualBytes == 2
        loadCls = BufferLoadB64 if isBf16 else BufferLoadB32
        chunkBytes = 4 << self.residualLog2Bytes
        rowOff = m * self.mfma_m * self.residualBytes
        for c in range(self.rows_per_lane // 4):
            off = rowOff + chunkBytes * c
            assert off < 4096, f"residual wide load offset {off} exceeds MUBUF offset12 range"
            dstBase = burstBase + 4 * c
            dst = vgpr(dstBase, 2) if isBf16 else vgpr(dstBase)
            module.add(loadCls(dst, vgpr(self.resRowByteBase), sgpr(self.resSrd, 4), 0,
                               MUBUFModifiers(offen=True, offset12=off),
                               comment=f"R wide [4 residual] (m={m},n={n},c={c}) off={off}."))
        module.addComment1("MF end _issueResidualWide.")


    def _issueResidualDwordx4(self, module, m, n, burstBase, _mBaseV) -> int:
        """Issue one full-exec BufferLoadB128 for tile (m, n); returns 1 (one load issued).

        self.resRowByteBase holds (token_n*N_hidden + wgRowBase + rowGroupOff)*2 for this n
        (row origin already folded by _residualRowByteBase).  The tile is addressed via a
        compile-time offset12 = m*mfma_m*2 (bf16), matching the ResidualOut store model.
        The upper lane's base is clamped to BufferOOB via a fresh scratch VGPR so the
        B128 load is dropped by the SRD; _redistributeResidualLoad scatters the lower
        lane's upper half to the upper lane.  _mBaseV is unused here but kept in the
        signature for caller compatibility.
        """
        module.addComment1(f"MF begin _issueResidualDwordx4: exec-masked B128 load for pair (m={m},n={n}).")
        lsc = self.lane_sgpr_count
        rowOff = m * self.mfma_m * 2
        assert rowOff < 4096, f"residual dwordx4 load offset {rowOff} exceeds MUBUF offset12 range"
        loadAddr = self.writer.vgprPool.checkOut(1, tag="mf_dx4LoadAddr")
        # GWB base+offset parity: address via resRowByteBase (folded row origin) + a
        # compile-time offset12; clamp the upper lane's base to BufferOOB so its B128
        # load is dropped by the SRD (the upper lane's data is re-gathered by
        # _redistributeResidualLoad anyway).
        module.add(VCndMaskB32(dst=vgpr(loadAddr), src0=vgpr(self.resRowByteBase),
                               src1=vgpr(self.resOobV), src2=sgpr(self.pairUpperLaneMask, lsc),
                               comment="upper lane -> BufferOOB (load dropped); lower lane keeps base."))
        module.add(BufferLoadB128(vgpr(burstBase, 4), vgpr(loadAddr),
                                  sgpr(self.resSrd, 4), 0,
                                  MUBUFModifiers(offen=True, offset12=rowOff),
                                  comment=f"R dwordx4 (m={m},n={n}) off={rowOff}: 8 bf16 for pair (full exec)."))
        self.writer.vgprPool.checkIn(loadAddr)
        module.addComment1("MF end _issueResidualDwordx4.")
        return 1


    def _redistributeResidualLoad(self, module, burstBase) -> None:
        """Scatter partner rows from lower lane (LG0/LG2) to upper lane (LG1/LG3).

        After the exec-masked B128 load, the lower lane holds 8 bf16 in burstBase[0..3]:
        dwords [0,1] are its own 4 rows, dwords [2,3] are the upper lane's 4 rows.
        The ds_bpermute must run under full exec: it only routes source data from
        lanes whose exec bit is set, and the data lives in the LOWER (source) lane,
        so masking to the upper lane before the permute would make every upper lane
        read zero from its disabled partner.  Gather into the dead burstBase[2,3]
        scratch under full exec, then adopt it into burstBase[0,1] using branchless
        VCndMaskB32 keyed by pairUpperLaneMask; this avoids save/set/restore-exec
        churn while preserving the full-exec correctness of the ds_bpermute.
        After this, both lanes have their own 4 bf16 in burstBase[0,1], ready for
        _convertResidualChunkBf16 to expand to f32 at burstBase[0..3].
        """
        module.addComment1("MF begin _redistributeResidualLoad: full-exec ds_bpermute gather then branchless cndmask adoption.")
        lsc = self.lane_sgpr_count
        # Full-exec gather: every lane pulls its partner's bank[2]/bank[3].  In-place
        # dst==src is safe because ds_bpermute snapshots all source lanes before writing.
        module.add(DSBPermuteB32(vgpr(burstBase + 2), vgpr(self.vPermAddr),
                                 vgpr(burstBase + 2),
                                 comment="gather partner bank[2] (rows 4,5); full exec."))
        module.add(DSBPermuteB32(vgpr(burstBase + 3), vgpr(self.vPermAddr),
                                 vgpr(burstBase + 3),
                                 comment="gather partner bank[3] (rows 6,7); full exec."))
        module.add(SWaitCnt(dscnt=0, comment="wait ds_bpermute."))
        # Adopt the gathered partner rows on the upper lane only; the lower lane keeps
        # its own loaded bank[0,1].  Branchless v_cndmask under full exec replaces an
        # exec-masked v_mov, dropping the save/set/restore-exec churn; the permutes
        # above still run under full exec, preserving the 099b29d correctness fix.
        module.add(VCndMaskB32(dst=vgpr(burstBase), src0=vgpr(burstBase),
                               src1=vgpr(burstBase + 2),
                               src2=sgpr(self.pairUpperLaneMask, lsc),
                               comment="bank[0] = upperLane ? partner rows 4,5 : own rows 0,1."))
        module.add(VCndMaskB32(dst=vgpr(burstBase + 1), src0=vgpr(burstBase + 1),
                               src1=vgpr(burstBase + 3),
                               src2=sgpr(self.pairUpperLaneMask, lsc),
                               comment="bank[1] = upperLane ? partner rows 6,7 : own rows 2,3."))
        module.addComment1("MF end _redistributeResidualLoad.")


    def _storeResidualOutRowDwordx4(self, module, srcRegs, m, n) -> None:
        """Pack H, gather partner's packed dwords, and store 8 bf16 via one B128 store.

        Both lower (LG0/LG2) and upper (LG1/LG3) lanes pack their H[0..3] into
        vPack[0,1].  Under full exec, the lower lane gathers the upper lane's
        vPack[0,1] into vPack[2,3] via ds_bpermute, then issues one BufferStoreB128
        covering 8 contiguous free0 rows.  Token-OOB lanes are suppressed by the
        ResidualOut SRD bounds; no explicit token mask is needed (interior path).
        """
        module.addComment1(f"MF begin _storeResidualOutRowDwordx4: gather partner packed dwords, B128 store (m={m},n={n}).")
        rpl = self.rowsPerLane
        assert rpl % 2 == 0, "rpl must be even for bf16 packing"
        lsc = self.lane_sgpr_count
        vgprPool = self.writer.vgprPool
        # vPack must be 4-aligned for BufferStoreB128.
        vPack = vgprPool.checkOutAligned(4, 4, tag="mf_dx4Pack")
        # Pack H[0..rpl-1] into vPack[0..(rpl//2)-1]; for rpl=4: two bf16x2 dwords.
        self._packResidualOutRow(module, srcRegs, vPack)
        # Gather the upper lane's packed dwords into vPack[2,3] under FULL exec: the
        # source is the UPPER lane, and ds_bpermute only routes data from lanes whose
        # exec bit is set, so masking to the lower lane before the permute would read
        # zero.  The upper lane's own vPack[2,3] become garbage but it never stores.
        module.add(DSBPermuteB32(vgpr(vPack + 2), vgpr(self.vPermAddr), vgpr(vPack),
                                 comment="lower lane: vPack[2] <- upper lane's vPack[0]; full exec."))
        module.add(DSBPermuteB32(vgpr(vPack + 3), vgpr(self.vPermAddr), vgpr(vPack + 1),
                                 comment="lower lane: vPack[3] <- upper lane's vPack[1]; full exec."))
        module.add(SWaitCnt(dscnt=0, comment="wait ds_bpermute."))
        # GlobalWriteBatch-style lane selection: clamp the upper lane's store address
        # to BufferOOB (branchless v_cndmask) so its B128 store is dropped by the SRD
        # bounds, then store under full exec.  Keeps exec full for the following
        # gamma/amax VALU and removes the exec save/set/restore around the store.
        dx4Addr = vgprPool.checkOut(1, tag="mf_dx4StoreAddr")
        module.add(VCndMaskB32(dst=vgpr(dx4Addr), src0=vgpr(self.roColByteBase),
                               src1=vgpr(self.resOobV),
                               src2=sgpr(self.pairUpperLaneMask, lsc),
                               comment="upper lane -> BufferOOB (store dropped); lower lane keeps addr."))
        rowOff = m * self.mfma_m * 2
        assert rowOff < 4096, f"residualOut row offset {rowOff} exceeds MUBUF offset12 range"
        module.add(BufferStoreB128(src=vgpr(vPack, 4), vaddr=vgpr(dx4Addr),
                                   saddr=sgpr(self.residualOutSrd, 4), soffset=0,
                                   mubuf=MUBUFModifiers(offen=True, offset12=rowOff),
                                   comment=f"ResidualOut dwordx4 (m={m},n={n}) off={rowOff} (full exec, upper OOB)."))
        vgprPool.checkIn(dx4Addr)
        vgprPool.checkIn(vPack)
        module.addComment1("MF end _storeResidualOutRowDwordx4.")


    def _maskWideResidualOOB(self, module, burstBase) -> None:
        """Software-mask wide residual elements where nhBase+k >= N_hidden.

        Wide loads read rows_per_lane contiguous nhidden positions from nhBase.
        Elements straddling the N_hidden boundary alias the next token's row in
        memory instead of returning buffer-OOB zero; software masking corrects this.
        self.resAddr and self.resRowByteBase are reused as scratch (load is done).
        When nhInRangeMask is set (bf16 tail-wide path), writes each per-k predicate
        into the shared slot so _computeBf16Addr can reuse it without recomputing.
        When nhInRangeMask is None (MXFP8 path), falls back to the reused resOobMask.
        """
        module.addComment1("MF begin _maskWideResidualOOB: software-mask wide residual OOB elements.")
        lsc = self.lane_sgpr_count
        for k in range(self.rows_per_lane):
            maskK = (self.nhInRangeMask + k * lsc) if self.nhInRangeMask is not None \
                    else self.resOobMask
            nhR = self._addImmU32(module, self.resAddr, self.nhBase, k, self.resRowByteBase,
                                 f"nhPos = nhBase + {k}.")
            module.add(VCmpLtU32(dst=sgpr(maskK, lsc), src0=vgpr(nhR),
                                 src1=sgpr("SizesFree+0"),
                                 comment=f"nhInRange[{k}] = nhPos < N_hidden (k={k})."))
            module.add(VCndMaskB32(dst=vgpr(burstBase + k), src0=0,
                                   src1=vgpr(burstBase + k),
                                   src2=sgpr(maskK, lsc),
                                   comment=f"residual = nhInRange ? residual : 0 (k={k})."))
        module.addComment1("MF end _maskWideResidualOOB.")


    def _issueResidualTile(self, module, m, n, burstBase, mBaseV,
                           pathInterior: bool = False) -> int:
        """Issue residual loads for tile (m, n) into burstBase; return #loads issued.

        No wait/convert here: loads are drained and converted later in the compute
        pass so the whole N-group's loads stay in flight together (GWB-style).
        Interior bf16 path: one exec-masked BufferLoadB128 for the lane-group pair.
        Wide path: one BufferLoad per 4-element chunk. Scalar path: one per element.
        """
        module.addComment1(f"MF begin _issueResidualTile: issue residual loads for tile (m={m},n={n}).")
        rpl = self.rows_per_lane
        if pathInterior and self._useDwordx4Interior():
            result = self._issueResidualDwordx4(module, m, n, burstBase, mBaseV)
            module.addComment1("MF end _issueResidualTile.")
            return result
        if self.useWideResidual:
            self._issueResidualWide(module, m, n, burstBase)
            module.addComment1("MF end _issueResidualTile.")
            return rpl // 4
        for k in range(rpl):
            self._residualElemAddr(module, self.resAddr, self.resRowByteBase,
                                  self.wgRowBase, self.rowGroupOff,
                                  self.resOobV, self.resOobMask, mBaseV, m, k)
            self._issueSideLoad(module, burstBase + k, self.resAddr, self.resSrd,
                               f"R[m={m},n={n},k={k}].", dtype=self.residualType)
        module.addComment1("MF end _issueResidualTile.")
        return rpl


    def _finishResidualTile(self, module, burstBase, pathInterior: bool = False) -> None:
        """Convert (and, for wide loads, software-mask) an already-loaded residual tile.

        Interior dwordx4 path: scatter bank[2,3] to the upper lane via exec-masked
        ds_bpermute, then convert bank[0,1] for all lanes.  No OOB masking needed.
        Wide interior path: convert only, no OOB masking — every row is < N_hidden
        (wgMaxRow < N_hidden is the interior arm's entry condition), so no element
        straddles the boundary and the per-element mask is a no-op.
        Wide tail path: convert and software-mask; the tail arm (Edge) may have rows
        that straddle the N_hidden boundary and requires explicit masking.
        self.nhBase must hold this tile's row position (set by _free0RowPos in the
        compute pass) before calling, because the wide OOB mask reads nhBase.
        """
        module.addComment1("MF begin _finishResidualTile: convert and mask already-loaded residual tile.")
        rpl = self.rows_per_lane
        if pathInterior and self._useDwordx4Interior():
            # Scatter partner rows from lower lane to upper lane, then convert.
            self._redistributeResidualLoad(module, burstBase)
            self._convertResidualChunkBf16(module, burstBase)
            # Interior lanes are guaranteed to be in-range; no OOB masking needed.
            module.addComment1("MF end _finishResidualTile.")
            return
        if self.useWideResidual:
            if self.residualBytes == 2:
                self._convertResidualChunkBf16(module, burstBase)
            else:
                self._convertResidualChunkFp8(module, burstBase)
            # GlobalWriteBatch NonEdge parity: the interior arm guarantees every row is
            # < N_hidden (wgMaxRow < N_hidden), so no element straddles the boundary and
            # the per-element OOB mask is a no-op.  Skip it in the interior arm, matching
            # the dwordx4 interior path; only the tail arm (Edge) needs software masking.
            if not pathInterior:
                self._maskWideResidualOOB(module, burstBase)
            module.addComment1("MF end _finishResidualTile.")
            return
        for k in range(rpl):
            self._convertSideElem(module, burstBase + k,
                                 f"residual->fp32 (k={k}).", dtype=self.residualType)
        module.addComment1("MF end _finishResidualTile.")


    def _pass1AccResRms(self, module, srcRegs, burstBase, m, n, rpl) -> None:
        """Fuse residual add and rmsSum accumulation: H = acc + R, rmsSum[n] += H²."""
        module.addComment1(f"MF begin _pass1AccResRms: H = acc + R, rmsSum[n] += H^2 (m={m},n={n}).")
        # Use packed add for aligned acc pairs; adds come before both squares so
        # the FMAs read the post-add values, matching the original per-element order.
        for k in range(0, rpl - rpl % 2, 2):
            sk0, sk1 = srcRegs[k], srcRegs[k + 1]
            if self._isPackPair(sk0, sk1):
                module.add(VAddPKF32(dst=vgpr(sk0, 2), src0=vgpr(sk0, 2),
                                     src1=vgpr(burstBase + k, 2),
                                     comment=f"H = acc + residual (packed k={k},{k+1})."))
            else:
                module.add(VAddF32(dst=vgpr(sk0), src0=vgpr(sk0),
                                   src1=vgpr(burstBase + k),
                                   comment=f"H = acc + residual (m={m},n={n},k={k})."))
                module.add(VAddF32(dst=vgpr(sk1), src0=vgpr(sk1),
                                   src1=vgpr(burstBase + k + 1),
                                   comment=f"H = acc + residual (m={m},n={n},k={k+1})."))
            module.add(VMacF32(dst=vgpr(self.partials + n), src0=vgpr(sk0),
                               src1=vgpr(sk0),
                               comment=f"rmsSum[{n}] += H² (m={m},n={n},k={k})."))
            module.add(VMacF32(dst=vgpr(self.partials + n), src0=vgpr(sk1),
                               src1=vgpr(sk1),
                               comment=f"rmsSum[{n}] += H² (m={m},n={n},k={k+1})."))
        # Defensive odd tail (rpl is even in practice).
        if rpl % 2 == 1:
            k = rpl - 1
            module.add(VAddF32(dst=vgpr(srcRegs[k]), src0=vgpr(srcRegs[k]),
                               src1=vgpr(burstBase + k),
                               comment=f"H = acc + residual (m={m},n={n},k={k})."))
            module.add(VMacF32(dst=vgpr(self.partials + n), src0=vgpr(srcRegs[k]),
                               src1=vgpr(srcRegs[k]),
                               comment=f"rmsSum[{n}] += H² (m={m},n={n},k={k})."))
        module.addComment1("MF end _pass1AccResRms.")


    def _amaxAndWriteAcc(self, module, sk, vgprTiles, blkAmaxJ, m, n, ki) -> None:
        """Fold |H*gamma| into blkAmax (MXFP8 only) and write sk back to the accumulator.

        Must be called once per k element after the multiply so the amax fold and
        accumulator writeback remain scalar (per-k) even when the multiply was packed.
        """
        module.addComment1(f"MF begin _amaxAndWriteAcc: fold |H*gamma| into blkAmax and write acc (m={m},n={n},k={ki}).")
        if self.useMxfp8:
            module.add(VAndB32(dst=vgpr(self._scAccTmp), src0=vgpr(sk),
                               src1=vgpr(self._scAbsMask),
                               comment=f"|H*gamma| (m={m},n={n},k={ki})."))
            module.add(VMaxF32(dst=vgpr(blkAmaxJ), src0=vgpr(blkAmaxJ),
                               src1=vgpr(self._scAccTmp),
                               comment="blkAmax = max(blkAmax, |H*gamma|)."))
        self._writeAccFrom(module, sk, vgprTiles, m, n, ki,
                              f"write H*gamma back to acc (m={m},n={n},k={ki}).")
        module.addComment1("MF end _amaxAndWriteAcc.")


    def _pass3GammaAmax(self, module, srcRegs, vgprTiles, gammaBank, blkAmaxJ,
                        mi, m, n, rpl) -> None:
        """Apply gamma, fold |H*gamma| into blkAmax for MXFP8, and write the result back to acc.

        The gamma-scaled value is written to the accumulator so the deferred MXFP8
        tail can re-read it. The amax fold and writeAccFrom MUST stay scalar per-k;
        only the multiply is packed.
        """
        module.addComment1(f"MF begin _pass3GammaAmax: apply gamma, fold amax, write acc (m={m},n={n}).")
        for k in range(0, rpl - rpl % 2, 2):
            sk0, sk1 = srcRegs[k], srcRegs[k + 1]
            # gammaBank is 2-aligned; when rpl % 2 == 0 (production gfx950 config),
            # mi*rpl+k is even so gk is 2-aligned by construction.
            gk = gammaBank + mi * rpl + k
            if self._isPackPair(sk0, sk1):
                module.add(VMulPKF32(dst=vgpr(sk0, 2), src0=vgpr(sk0, 2),
                                     src1=vgpr(gk, 2),
                                     comment=f"acc = H * gamma (packed k={k},{k+1})."))
            else:
                module.add(VMulF32(dst=vgpr(sk0), src0=vgpr(sk0), src1=vgpr(gk),
                                   comment=f"acc = H * gamma (m={m},n={n},k={k})."))
                module.add(VMulF32(dst=vgpr(sk1), src0=vgpr(sk1), src1=vgpr(gk + 1),
                                   comment=f"acc = H * gamma (m={m},n={n},k={k+1})."))
            self._amaxAndWriteAcc(module, sk0, vgprTiles, blkAmaxJ, m, n, k)
            self._amaxAndWriteAcc(module, sk1, vgprTiles, blkAmaxJ, m, n, k + 1)
        # Defensive odd tail (rpl is even in practice).
        if rpl % 2 == 1:
            k = rpl - 1
            acc = srcRegs[k]
            gammaReg = gammaBank + mi * rpl + k
            module.add(VMulF32(dst=vgpr(acc), src0=vgpr(acc), src1=vgpr(gammaReg),
                               comment=f"acc = H * gamma (m={m},n={n},k={k})."))
            self._amaxAndWriteAcc(module, acc, vgprTiles, blkAmaxJ, m, n, k)
        module.addComment1("MF end _pass3GammaAmax.")


    def _prologResidualLoads(self, module, resBank, mBaseV, qi, nBase, g,
                             pathInterior: bool = False):
        """Issue every residual load for the N-group into resBank so loads overlap.

        Returns (loadsCumulative, totalIssued): loadsCumulative[t] is the number
        of residual loads issued up to and including tile t (issue order equals
        compute order), which drives the per-tile decreasing vlcnt in the compute
        pass.
        """
        module.addComment1(f"MF begin _prologResidualLoads: issue all residual loads for N-group (qi={qi},nBase={nBase},g={g}).")
        rpl = self.rowsPerLane
        tpb = self.tilesPerBlockM
        loadsCumulative = []
        issued = 0
        for j in range(g):
            n = nBase + j
            self._residualRowByteBase(module, self.resRowByteBase,
                                     self.resTokenBase, n, self.resAddr)
            for mi in range(tpb):
                m = qi * tpb + mi
                burstBase = resBank + (j * tpb + mi) * rpl
                issued += self._issueResidualTile(module, m, n, burstBase, mBaseV,
                                                  pathInterior)
                loadsCumulative.append(issued)
        module.addComment1("MF end _prologResidualLoads.")
        return loadsCumulative, issued


    def _computePassTile(self, module, vgprTiles, accBank, resBank, gammaBank,
                         blkAmax, loadsCumulative, totalIssued, mBaseV, tokMaskSgpr,
                         qi, nBase, j, mi, t, pathInterior: bool = False) -> int:
        """Emit instructions for one (mi, j) tile in the compute pass; returns updated t."""
        module.addComment1(f"MF begin _computePassTile: compute pass for tile (qi={qi},j={j},mi={mi}).")
        rpl = self.rowsPerLane
        tpb = self.tilesPerBlockM
        lsc = self.lane_sgpr_count
        m = qi * tpb + mi
        n = nBase + j
        bankBase = (j * tpb + mi) * rpl
        burstBase = resBank + bankBase
        coords = [(m, n, k) for k in range(rpl)]
        srcRegs = self._readAccBurst(module, accBank + bankBase, vgprTiles,
                                    coords, f"acc m={m},n={n}.")
        self._free0RowPos(module, self.nhBase, self.wgRowBase,
                         self.rowGroupOff, m, 0, mBaseV)
        # Wait only for THIS tile's residual load; later tiles' loads
        # stay in flight (GWB decreasing-vlcnt schedule).
        remaining = totalIssued - loadsCumulative[t]
        module.add(SWaitCnt(vlcnt=remaining,
                            comment=f"wait residual tile {t}: vlcnt={totalIssued}-{loadsCumulative[t]}."))
        useDx4 = pathInterior and self._useDwordx4Interior()
        # Tail-wide path (bf16 only): check out a shared nhInRange mask block before
        # _finishResidualTile so the per-k predicates computed by _maskWideResidualOOB
        # can be reused by _computeBf16Addr without recomputing.  Skipped for MXFP8
        # kernels to avoid SGPR pressure (they recompute the predicate in _computeBf16Addr).
        tailWide = self.useWideResidual and not pathInterior and not self.useMxfp8
        if tailWide:
            self.nhInRangeMask = self.writer.sgprPool.checkOutAligned(
                rpl * lsc, lsc, tag="mf_nhInRange", preventOverflow=False)
        self._finishResidualTile(module, burstBase, pathInterior)
        self._pass1AccResRms(module, srcRegs, burstBase, m, n, rpl)
        if useDx4:
            self._storeResidualOutRowDwordx4(module, srcRegs, m, n)
        else:
            self._storeResidualOutRow(module, srcRegs, tokMaskSgpr, m, n)
        if tailWide:
            self.writer.sgprPool.checkIn(self.nhInRangeMask)
            self.nhInRangeMask = None
        blkAmaxJ = (blkAmax + n) if self.useMxfp8 else None
        self._pass3GammaAmax(module, srcRegs, vgprTiles, gammaBank, blkAmaxJ,
                             mi, m, n, rpl)
        module.addComment1("MF end _computePassTile.")
        return t + 1


    def _computePass(self, module, vgprTiles, accBank, resBank, gammaBank,
                     blkAmax, loadsCumulative, totalIssued, mBaseV, qi, nBase, g,
                     pathInterior: bool = False) -> None:
        """Drain residual loads per tile, then run residual add, bf16 store, rmsSum, gamma/amax.

        Uses a GWB decreasing-vlcnt schedule: each tile waits only for its own
        residual load so later tiles' loads stay in flight.
        """
        module.addComment1(f"MF begin _computePass: compute pass over N-group (qi={qi},nBase={nBase},g={g}).")
        lsc = self.lane_sgpr_count
        tpb = self.tilesPerBlockM
        t = 0
        for j in range(g):
            n = nBase + j
            # Compute the token(N) OOB mask once per column n; reused across all tpb tiles.
            tokMaskSgpr = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mf_roTokMask",
                                                               preventOverflow=False)
            self.roRowBase = self.writer.vgprPool.checkOut(1, tag="mf_roRowBase")
            self.roColByteBase = self.writer.vgprPool.checkOut(1, tag="mf_roColByteBase")
            self._computeResidualOutRowBaseAndMask(module, n, tokMaskSgpr)
            for mi in range(tpb):
                t = self._computePassTile(module, vgprTiles, accBank, resBank, gammaBank,
                                          blkAmax, loadsCumulative, totalIssued, mBaseV,
                                          tokMaskSgpr, qi, nBase, j, mi, t, pathInterior)
            self.writer.vgprPool.checkIn(self.roColByteBase)
            self.roColByteBase = None
            self.writer.vgprPool.checkIn(self.roRowBase)
            self.roRowBase = None
            self.writer.sgprPool.checkIn(tokMaskSgpr)
        module.addComment1("MF end _computePass.")


    def _fusedElementLoop(self, module, vgprTiles, accBank, resBank, gammaBank,
                          blkAmax, qi, nBase, g) -> None:
        """Emit the fused loop as a GWB-style split: a load prolog then a compute pass.

        The prolog issues every residual load for the N-group into resBank so the
        loads overlap; the compute pass drains them per tile and runs residual add,
        bf16 store, rmsSum, and gamma/amax.
        """
        module.addComment1(f"MF begin _fusedElementLoop: GWB prolog + compute pass (qi={qi},nBase={nBase},g={g}).")
        mBaseV = self.writer.vgprPool.checkOut(1, tag="mf_mBase")
        loadsCumulative, totalIssued = self._prologResidualLoads(
            module, resBank, mBaseV, qi, nBase, g)
        self._computePass(module, vgprTiles, accBank, resBank, gammaBank,
                          blkAmax, loadsCumulative, totalIssued, mBaseV, qi, nBase, g)
        self.writer.vgprPool.checkIn(mBaseV)
        module.addComment1("MF end _fusedElementLoop.")


    def _initBlkAmax(self, blkAmax) -> Module:
        """Zero the per-qi persistent blkAmax bank (one f32 per absolute N column)."""
        module = Module("MegaFused initBlkAmax")
        module.addComment0("MF begin _initBlkAmax: zero per-qi blkAmax bank.")
        for n in range(self.mmaN):
            module.add(VMovB32(dst=vgpr(blkAmax + n), src=0, comment=f"blkAmax[{n}] = 0."))
        module.addComment0("MF end _initBlkAmax.")
        return module


    def _fusedFrontHalf(self, vgprTiles, gammaBank, blkAmax, qi, nBase, g) -> Module:
        """Emit one N-group's element loop (residual add, bf16 store, rmsSum, gamma).

        For MXFP8 the gamma-scaled result is written back to the accumulator and
        |H*gamma| is folded into the persistent blkAmax bank; the group's MXFP8 tail
        is deferred and emitted later by _mxDeferredTail.
        """
        module = Module(f"MegaFused frontHalf qi={qi} nBase={nBase}")
        module.addComment0(f"MF begin _fusedFrontHalf: fused element loop for N-group (qi={qi},nBase={nBase}).")
        vgprPool = self.writer.vgprPool
        bankSize = g * self.tilesPerBlockM * self.rowsPerLane
        # 2-aligned so AGPR-staged acc pairs are packed-VALU eligible.
        accBank = vgprPool.checkOutAligned(bankSize, 2, tag="mf_accBank")
        # Whole-N-group residual bank so all residual loads overlap (2-aligned for
        # the wide BufferLoadB64 path).
        resBank = vgprPool.checkOutAligned(bankSize, 2, tag="mf_resBank")
        self._fusedElementLoop(module, vgprTiles, accBank, resBank, gammaBank,
                               blkAmax, qi, nBase, g)
        vgprPool.checkIn(resBank)
        vgprPool.checkIn(accBank)
        module.addComment0("MF end _fusedFrontHalf.")
        return module


    def _mxDeferredTail(self, vgprTiles, blkAmax, qi, nBase, g) -> Module:
        """Deferred MXFP8 tail for one N-group: butterfly-reduce blkAmax, compute e8m0
        scales, re-read the accumulator to apply alpha*quantMult, and store MXScale bytes.

        blkAmax is the persistent mmaN bank; this group owns the slice [nBase, nBase+g).
        """
        module = Module(f"MegaFused mxDeferredTail qi={qi} nBase={nBase}")
        module.addComment0(f"MF begin _mxDeferredTail: butterfly-reduce blkAmax, compute e8m0 scales, apply and store (qi={qi},nBase={nBase}).")
        vgprPool = self.writer.vgprPool
        amaxSlice = blkAmax + nBase
        addrBf = vgprPool.checkOut(1, tag="mf_addrBf")
        tmpBf = vgprPool.checkOut(g, tag="mf_tmpBf")
        for r in range(2):
            self._butterflyRound(module, addrBf, tmpBf, amaxSlice, g, self.laneId,
                                   self.mfmaN << r)
        vgprPool.checkIn(tmpBf)
        vgprPool.checkIn(addrBf)
        # Alpha fold: blkAmax[j] = |alpha * blkAmax[j]|, matching _streamSubColGroup.
        for j in range(g):
            module.add(VMulF32(dst=vgpr(amaxSlice + j), src0=vgpr(amaxSlice + j),
                               src1=sgpr("Alpha"), comment=f"blkAmax[{nBase + j}] *= alpha."))
            module.add(VAndB32(dst=vgpr(amaxSlice + j), src0=vgpr(amaxSlice + j),
                               src1=vgpr(self._scAbsMask),
                               comment=f"blkAmax[{nBase + j}] = |alpha*blkAmax|."))
        # _computeSubColScales overwrites the slice with alpha*quantMult (the apply multiplier).
        scaleByteBank = self._computeSubColScales(module, amaxSlice, qi, nBase, g)
        applyScratch = vgprPool.checkOut(self.rowsPerLane, tag="mf_applyScratch")
        mStart = qi * self.tilesPerBlockM
        mEnd = (qi + 1) * self.tilesPerBlockM
        self._subColApplyFromAcc(module, vgprTiles, amaxSlice, applyScratch,
                                   mStart, mEnd, nBase, g)
        vgprPool.checkIn(applyScratch)
        self._subColStoreGroup(module, self.mxSrd, scaleByteBank, self.col, self.rowGroup,
                                 self.savedExec, self.laneMaskSgpr, qi, nBase, g)
        vgprPool.checkIn(scaleByteBank)
        module.addComment0("MF end _mxDeferredTail.")
        return module


    def _reduceAndWriteRms(self) -> Module:
        """Finalise rmsSum: reduce across row groups and waves, then write to partialBuf.

        PartialBuf SRD is built here (deferred from setup to reduce SGPR pressure
        during the fused element loop) and freed before returning.
        """
        module = Module("MegaFused reduceAndWriteRms")
        module.addComment0("MF begin _reduceAndWriteRms: reduce rmsSum across row groups and waves, write to partialBuf.")
        sgprPool = self.writer.sgprPool
        partialSrd = sgprPool.checkOutAligned(4, 4, tag="mf_partialSrd", preventOverflow=False)
        self._buildBufferSrd(module, partialSrd, "PartialBuf", "partialBuf")
        module.add(self._reduceFree0())
        globalAddr = self.writer.vgprPool.checkOut(1, tag="mf_globalAddr")
        module.add(self._writePartialsFree0(
            self.partials, partialSrd, self.laneId, self.savedExec, self.laneMaskSgpr,
            globalAddr, self.colByte))
        self.writer.vgprPool.checkIn(globalAddr)
        sgprPool.checkIn(partialSrd)
        module.addComment0("MF end _reduceAndWriteRms.")
        return module


    def _issueUnitLoads(self, resBank, mBaseV, qi, nBase, g,
                        pathInterior: bool = False) -> Module:
        """Emit residual loads for one unit into resBank (prolog or prefetch phase).

        Wraps _prologResidualLoads; the static load-count tracking is done by the
        caller (pre-computed once), so the return value is intentionally discarded.
        """
        module = Module(f"MegaFused issueUnitLoads qi={qi} nBase={nBase}")
        module.addComment0(f"MF begin _issueUnitLoads: residual loads for one unit (qi={qi},nBase={nBase}).")
        self._prologResidualLoads(module, resBank, mBaseV, qi, nBase, g, pathInterior)
        module.addComment0("MF end _issueUnitLoads.")
        return module


    def _computeUnitFromBank(self, vgprTiles, accBank, resBank, gammaBank, blkAmax,
                              issuedWatermark, localLoadsCum, mBaseV,
                              qi, nBase, g, pathInterior: bool = False) -> Module:
        """Run the compute pass for one unit using pre-issued residual loads in resBank.

        issuedWatermark replaces the old per-unit totalIssued: it is the global count
        of loads issued up to and including the last in-flight prefetch unit, so the
        per-tile decreasing-vlcnt schedule spans the entire PFD-deep pipeline.
        """
        module = Module(f"MegaFused computeUnit qi={qi} nBase={nBase}")
        module.addComment0(f"MF begin _computeUnitFromBank: compute pass for one ring unit (qi={qi},nBase={nBase}).")
        self._computePass(module, vgprTiles, accBank, resBank, gammaBank, blkAmax,
                          localLoadsCum, issuedWatermark, mBaseV, qi, nBase, g, pathInterior)
        module.addComment0("MF end _computeUnitFromBank.")
        return module


    def _emitSharedSetup(self, module) -> int:
        """Emit row geometry, rmsSum init, gamma bank allocation, residual scratch, stream context.

        Returns the gamma bank VGPR base; caller must free it after the main sweep.
        """
        module.addComment1("MF begin _emitSharedSetup: row geometry, rmsSum init, gamma bank, residual scratch.")
        self._computeRowGroupOff(module, self.rowGroupOff)
        self._computeFree0RowBase(module, self.wgRowBase)
        module.add(self._initRmsSum())
        # Gamma stays VGPR-resident for the whole sweep; 2-aligned for dwordx2 loads.
        gammaBank = self.writer.vgprPool.checkOutAligned(
            self.tilesPerBlockM * self.rowsPerLane, 2, tag="mf_gamma")
        self._beginResidualScratch(module)
        if self.useMxfp8:
            self._beginStreamContext(module)
        # dwordx4 setup must happen BEFORE the interior/tail branch so that vPermAddr
        # and the pair masks are available in both arms (only interior uses them).
        if self._useDwordx4Interior():
            self._emitDwordx4Setup(module)
        if self.gammaLdsStaging:
            self._emitGammaLdsSetup(module)
        module.addComment1("MF end _emitSharedSetup.")
        return gammaBank


    def _emitDwordx4Setup(self, module) -> None:
        """Precompute vPermAddr and pair exec masks for the dwordx4 interior path.

        vPermAddr = (laneId XOR 16) * 4: the ds_bpermute partner-lane byte address.
        pairLowerLaneMask: lanes {0-15, 32-47} (rowGroup even — lower of each LG pair).
        pairUpperLaneMask: lanes {16-31, 48-63} (rowGroup odd  — upper of each LG pair).
        All three are loop-invariant and live for the entire epilogue.
        """
        module.addComment1("MF begin _emitDwordx4Setup: precompute vPermAddr and pair exec masks.")
        module.add(VXorB32(dst=vgpr(self.vPermAddr), src0=vgpr(self.laneId), src1=16,
                           comment="partner = laneId XOR 16 (pair within 32-lane half)."))
        module.add(VLShiftLeftB32(dst=vgpr(self.vPermAddr), shiftHex=hex(2),
                                  src=vgpr(self.vPermAddr),
                                  comment="vPermAddr = partner * 4 (ds_bpermute byte addr)."))
        lsc = self.lane_sgpr_count
        # Wave64: lower lanes are {0-15, 32-47} = 0x0000FFFF in each 32-bit half.
        module.add(SMovB32(dst=sgpr(self.pairLowerLaneMask), src=hex(0x0000FFFF),
                           comment="pairLower lo: lanes 0-15."))
        module.add(SMovB32(dst=sgpr(self.pairLowerLaneMask + 1), src=hex(0x0000FFFF),
                           comment="pairLower hi: lanes 32-47."))
        module.add(SMovB32(dst=sgpr(self.pairUpperLaneMask), src=hex(0xFFFF0000),
                           comment="pairUpper lo: lanes 16-31."))
        module.add(SMovB32(dst=sgpr(self.pairUpperLaneMask + 1), src=hex(0xFFFF0000),
                           comment="pairUpper hi: lanes 48-63."))
        module.addComment1("MF end _emitDwordx4Setup.")


    def _emitGammaLdsSetup(self, module) -> None:
        """Precompute loop-invariant gamma DTL addresses and M0 base (per-wave, per-lane).

        gammaDtlVaddr (VGPR): producer vaddr = laneId * 4 (contiguous b32 DTL lane offset).
        gammaLdsReadAddr (VGPR): consumer base = waveM*ldsWaveStride + rowGroupOff*gammaBytes.
        gammaSoffsetSgpr (SGPR): soffset = wgRowBase * gammaBytes (uniform within a wave).
        gammaM0Base (SGPR): waveM * ldsWaveStride (wave-uniform M0 base; bufIdx offset added per-call).
        """
        module.addComment1("MF begin _emitGammaLdsSetup: precompute gamma DTL addresses and M0 base.")
        # gammaDtlVaddr = laneId * 4: the DTL hardware writes M0+laneId*loadWidth, so
        # using a contiguous lane offset makes the LDS write layout lane-position-dense.
        module.add(VLShiftLeftB32(dst=vgpr(self.gammaDtlVaddr), shiftHex=hex(2),
                                  src=vgpr(self.laneId),
                                  comment="gammaDtlVaddr = laneId * 4 (contiguous b32 DTL offset)."))
        # gammaSoffsetSgpr = wgRowBase * gammaBytes; wgRowBase is wave-uniform in a VGPR.
        module.add(SNop(waitState=0,
                        comment="conservative nop: wgRowBase was written well before this point, so the VALU hazard window is already closed."))
        with self.writer.allocTmpSgpr(1, tag="mf_gammaRfl") as t:
            module.add(VReadfirstlaneB32(dst=sgpr(t.idx), src=vgpr(self.wgRowBase),
                                         comment="extract uniform wgRowBase from VGPR."))
            module.add(SLShiftLeftB32(dst=sgpr(self.gammaSoffsetSgpr), src=sgpr(t.idx),
                                      shiftHex=hex(self.gammaLog2Bytes),
                                      comment="gammaSoffsetSgpr = wgRowBase * gammaBytes."))
        assert (self.gammaLdsWaveStride & (self.gammaLdsWaveStride - 1)) == 0, \
            "gamma LDS wave stride must be a power of two for the log2 shift"
        log2LdsWaveStride = self.gammaLdsWaveStride.bit_length() - 1
        if self.wg_m > 1:
            waveMV = self.writer.vgprPool.checkOut(1, tag="mf_gammaWaveMV")
            module.add(VAndB32(dst=vgpr(waveMV), src0=vgpr(self.waveIdV), src1=self.wg_m - 1,
                               comment=f"waveM = waveId % {self.wg_m}."))
            module.add(SNop(waitState=0,
                            comment="wait for VGPR before readfirstlane (VALU write hazard)."))
            module.add(VReadfirstlaneB32(dst=sgpr(self.gammaM0Base), src=vgpr(waveMV),
                                         comment="waveM_scalar for M0 base."))
            self.writer.vgprPool.checkIn(waveMV)
            module.add(SLShiftLeftB32(dst=sgpr(self.gammaM0Base), src=sgpr(self.gammaM0Base),
                                      shiftHex=hex(log2LdsWaveStride),
                                      comment=f"gammaM0Base = waveM * ldsWaveStride({self.gammaLdsWaveStride})."))
        else:
            module.add(SMovB32(dst=sgpr(self.gammaM0Base), src=0,
                               comment="gammaM0Base = 0 (wg_m == 1, waveM always 0)."))
        # gammaLdsReadAddr = waveM*ldsWaveStride + rowGroupOff*gammaBytes.
        # rowGroupOff*gammaBytes is computed into a scratch VGPR (not gammaDtlVaddr,
        # which now holds the producer offset laneId*4).
        rowGroupOffBytes = self.writer.vgprPool.checkOut(1, tag="mf_gammaRGOff")
        module.add(VLShiftLeftB32(dst=vgpr(rowGroupOffBytes), shiftHex=hex(self.gammaLog2Bytes),
                                  src=vgpr(self.rowGroupOff),
                                  comment="rowGroupOff * gammaBytes for consumer LDS read base."))
        module.add(VMovB32(dst=vgpr(self.gammaLdsReadAddr), src=sgpr(self.gammaM0Base),
                           comment="gammaLdsReadAddr = waveM * ldsWaveStride (SGPR -> VGPR)."))
        module.add(VAddU32(dst=vgpr(self.gammaLdsReadAddr),
                           src0=vgpr(self.gammaLdsReadAddr),
                           src1=vgpr(rowGroupOffBytes),
                           comment="gammaLdsReadAddr += rowGroupOff * gammaBytes."))
        self.writer.vgprPool.checkIn(rowGroupOffBytes)
        module.addComment1("MF end _emitGammaLdsSetup.")


    def _stageGammaToLds(self, module, qi, bufIdx) -> None:
        """Stage qi's gamma from global memory to LDS via one contiguous-lane b32 DTL.

        Lanes [0, numStageLanes) in waveN==0 each issue one buffer_load_b32 lds.
        DTL writes M0+laneId*4 in LDS, producing a dense lane-contiguous layout that
        matches the byte offsets the consumer reads with ds_read_b64.
        """
        module.addComment1(f"MF begin _stageGammaToLds: contiguous-lane DTL stage gamma to LDS (qi={qi},buf={bufIdx}).")
        sgprPool = self.writer.sgprPool
        lsc = self.lane_sgpr_count
        numStageLanes = self.tilesPerBlockM * self.mfma_m // 2
        module.add(self.writer._syncThreads(self.kernel,
                                            "gamma DTL stage: WAR barrier before reusing LDS region."))
        savedExec = sgprPool.checkOutAligned(lsc, lsc, tag="mf_gammaStageExec", preventOverflow=False)
        module.add(SMovB64(dst=sgpr(savedExec, lsc), src=EXEC(),
                           comment="save exec around contiguous-lane DTL gamma load."))
        # Narrow exec to laneId < numStageLanes (contiguous lanes); AND with waveN==0.
        repMask = sgprPool.checkOutAligned(lsc, lsc, tag="mf_gammaRep", preventOverflow=False)
        module.add(VCmpLtU32(dst=sgpr(repMask, lsc), src0=vgpr(self.laneId), src1=numStageLanes,
                             comment=f"stageLane = laneId < {numStageLanes} (contiguous b32 DTL)."))
        if self.wg_n > 1:
            wtmp = sgprPool.checkOutAligned(lsc, lsc, tag="mf_gammaRepWN", preventOverflow=False)
            waveThreshold = self.wg_m * self.waveSize
            with self.writer.allocTmpSgpr(1, tag="mf_gammaWNThr") as thr:
                module.add(SMovB32(dst=sgpr(thr.idx), src=hex(waveThreshold),
                                   comment=f"wg_m*waveSize = {waveThreshold} (waveN==0 bound)."))
                module.add(VCmpLtU32(dst=sgpr(wtmp, lsc), src0=vgpr("Serial"),
                                     src1=sgpr(thr.idx),
                                     comment="waveN0 = Serial < wg_m*waveSize."))
            module.add(SAndB64(dst=sgpr(repMask, lsc), src0=sgpr(repMask, lsc),
                               src1=sgpr(wtmp, lsc), comment="repLane &= (waveN == 0)."))
            sgprPool.checkIn(wtmp)
        module.add(SMovB64(dst=EXEC(), src=sgpr(repMask, lsc),
                           comment="exec = contiguous stage lanes for DTL gamma load."))
        sgprPool.checkIn(repMask)
        # M0 = waveM * ldsWaveStride + bufIdx * ldsBufBytes.
        if bufIdx == 0:
            module.add(SMovB32(dst=mgpr(0), src=sgpr(self.gammaM0Base),
                               comment="M0 = gamma LDS wave base (buf0)."))
        else:
            with self.writer.allocTmpSgpr(1, tag="mf_gammaM0") as t:
                module.add(SAddU32(dst=sgpr(t.idx), src0=sgpr(self.gammaM0Base),
                                   src1=bufIdx * self.gammaLdsBufBytes,
                                   comment=f"M0 base + buf{bufIdx} offset."))
                module.add(SMovB32(dst=mgpr(0), src=sgpr(t.idx),
                                   comment=f"M0 = gamma LDS wave base (buf{bufIdx})."))
        # soffset = wgRowBase*gammaBytes + qi*tilesPerBlockM*mfma_m*gammaBytes.
        qiSoffsetAdj = qi * self.tilesPerBlockM * self.mfma_m * self.gammaBytes
        if qiSoffsetAdj > 0:
            qiSoffSgpr = sgprPool.checkOutAligned(1, 1, tag="mf_gammaQiSoff", preventOverflow=False)
            module.add(SAddU32(dst=sgpr(qiSoffSgpr), src0=sgpr(self.gammaSoffsetSgpr),
                               src1=qiSoffsetAdj,
                               comment=f"soffset += qi*tpb*mfma_m*gammaBytes for qi={qi}."))
        else:
            qiSoffSgpr = self.gammaSoffsetSgpr
        # One b32 load covers the entire qi block: lane i reads gamma[2i] and gamma[2i+1]
        # and DTL writes them contiguously at LDS M0+i*4, matching the consumer layout.
        module.add(BufferLoadB32(
            dst=None, vaddr=vgpr(self.gammaDtlVaddr), saddr=sgpr(self.gammaSrd, 4),
            soffset=sgpr(qiSoffSgpr),
            mubuf=MUBUFModifiers(offen=True, offset12=0, lds=True),
            comment=f"gamma DTL b32 -> LDS (qi={qi},buf={bufIdx}); {numStageLanes} lanes x 2 bf16."))
        if qiSoffsetAdj > 0:
            sgprPool.checkIn(qiSoffSgpr)
        # DTL is tracked by vmcnt; wait before restoring exec and releasing the exec guard.
        module.add(SWaitCnt(vlcnt=0, comment="wait gamma DTL loads land in LDS (vmcnt tracks DTL)."))
        module.add(SMovB64(dst=EXEC(), src=sgpr(savedExec, lsc),
                           comment="restore full exec after gamma DTL stage."))
        sgprPool.checkIn(savedExec)
        module.add(self.writer._syncThreads(self.kernel,
                                            "gamma DTL stage: LDS writes visible before broadcast read."))
        module.addComment1("MF end _stageGammaToLds.")


    def _ldsReadGammaBlock(self, module, gammaBank, qi, bufIdx) -> None:
        """Broadcast-read staged gamma from LDS; convert bf16 to f32 in place.

        All lanes read rpl bf16 (one DSLoadB64 per mi) from their LDS slot, broadcasting
        data written by the representative lane that owns the matching rowGroup-pair.
        """
        module.addComment1(f"MF begin _ldsReadGammaBlock: broadcast-read gamma from LDS (qi={qi},buf={bufIdx}).")
        tpb = self.tilesPerBlockM
        for mi in range(tpb):
            off = bufIdx * self.gammaLdsBufBytes + mi * self.mfma_m * self.gammaBytes
            module.add(DSLoadB64(
                dst=vgpr(gammaBank + mi * self.rows_per_lane, 2),
                src=vgpr(self.gammaLdsReadAddr),
                ds=DSModifiers(offset=off),
                comment=f"broadcast-read gamma bf16 (qi={qi},mi={mi},buf={bufIdx})."))
        module.add(SWaitCnt(dscnt=0, comment="wait gamma LDS broadcast reads."))
        for mi in range(tpb):
            self._convertGammaChunkBf16(module, gammaBank + mi * self.rows_per_lane)
        module.addComment1("MF end _ldsReadGammaBlock.")


    def _buildUnitSequence(self, ebc):
        """Build the flat unit list and precompute global per-tile cumulative load counts.

        Returns (units, globalLoadsCum, issuedThroughUnit, tileStarts).  A unit is
        (qi, nBase, g); globalLoadsCum[t] is the count of loads issued through tile t
        in issue order; tileStarts[i] is the first tile index in globalLoadsCum for unit i.
        """
        tpb = self.tilesPerBlockM
        units = [
            (qi, nBase, min(ebc, self.mmaN - nBase))
            for qi in range(self.nQTilesM)
            for nBase in range(0, self.mmaN, ebc)
        ]
        loadsPerTile = self.rowsPerLane // 4 if self.useWideResidual else self.rowsPerLane
        globalLoadsCum = []
        issuedThroughUnit = []
        running = 0
        for _qi, _nBase, g in units:
            for _j in range(g):
                for _mi in range(tpb):
                    running += loadsPerTile
                    globalLoadsCum.append(running)
            issuedThroughUnit.append(running)
        tileStarts = []
        cumTiles = 0
        for _qi, _nBase, g in units:
            tileStarts.append(cumTiles)
            cumTiles += g * tpb
        return units, globalLoadsCum, issuedThroughUnit, tileStarts


    def _allocRing(self, ebc, pfd):
        """Allocate PFD residual-bank ring, acc bank, blkAmax, and mBase scratch.

        Returns (resRing, accBank, blkAmax, mBaseV); blkAmax is None when not useMxfp8.
        blkAmax is mmaN-wide: each unit's g-slice is zeroed before use and consumed by
        mxDeferredTail before the next unit begins, so no cross-unit aliasing occurs.
        """
        tpb = self.tilesPerBlockM
        bankSize = ebc * tpb * self.rowsPerLane
        resRing = [
            self.writer.vgprPool.checkOutAligned(bankSize, 2, tag=f"mf_resBank{sl}")
            for sl in range(pfd)
        ]
        accBank = self.writer.vgprPool.checkOutAligned(bankSize, 2, tag="mf_accBank")
        blkAmax = None
        if self.useMxfp8:
            blkAmax = self.writer.vgprPool.checkOut(self.mmaN, tag="mf_blkAmax")
        mBaseV = self.writer.vgprPool.checkOut(1, tag="mf_mBase")
        return resRing, accBank, blkAmax, mBaseV


    def _freeRing(self, resRing, accBank, blkAmax, mBaseV, pfd) -> None:
        """Free the ring, acc bank, blkAmax, and mBase in reverse allocation order.

        Reverse order matters: the VGPR pool is a stack allocator, so freeing in
        reverse allocation order keeps the high-water mark low and avoids fragmenting
        the pool between the ring slots.
        """
        self.writer.vgprPool.checkIn(mBaseV)
        if self.useMxfp8:
            self.writer.vgprPool.checkIn(blkAmax)
        self.writer.vgprPool.checkIn(accBank)
        for sl in range(pfd - 1, -1, -1):
            self.writer.vgprPool.checkIn(resRing[sl])


    def _emitProlog(self, module, units, pfd, resRing, mBaseV, pathInterior: bool) -> None:
        """Issue residual loads for the first PFD units before any compute begins."""
        module.addComment1("MF begin _emitProlog: issue residual loads for the first PFD units.")
        numUnits = len(units)
        for u in range(min(pfd, numUnits)):
            pqi, pnBase, pg = units[u]
            module.add(self._issueUnitLoads(resRing[u % pfd], mBaseV, pqi, pnBase, pg,
                                            pathInterior))
        module.addComment1("MF end _emitProlog.")


    def _emitZeroBlkAmaxSlice(self, module, blkAmax, nBase, g, qi) -> None:
        """Zero the blkAmax slice [nBase, nBase+g) for one unit."""
        module.addComment1(f"MF begin _emitZeroBlkAmaxSlice: zero blkAmax[nBase..nBase+g) (qi={qi},nBase={nBase}).")
        zeroMod = Module(f"MegaFused zeroBlkAmaxSlice qi={qi} nBase={nBase}")
        for j in range(g):
            zeroMod.add(VMovB32(dst=vgpr(blkAmax + nBase + j), src=0,
                                comment=f"blkAmax[{nBase + j}] = 0."))
        module.add(zeroMod)
        module.addComment1("MF end _emitZeroBlkAmaxSlice.")


    def _emitUnitIteration(self, module, vgprTiles, units, resRing, accBank, gammaBank,
                            blkAmax, mBaseV, globalLoadsCum, issuedThroughUnit,
                            tileStarts, pfd, pathInterior: bool) -> None:
        """Compute each unit, issue its mxDeferredTail when active, then prefetch the next unit.

        Uses a PFD-deep ring: unit i computes while unit i+PFD's loads are in flight.
        """
        module.addComment1("MF begin _emitUnitIteration: PFD-pipeline iteration over all units.")
        tpb = self.tilesPerBlockM
        numUnits = len(units)
        for i, (qi, nBase, g) in enumerate(units):
            if nBase == 0 and not self.gammaLdsStaging:
                module.add(self._loadGammaBlock(gammaBank, qi))
            if self.useMxfp8:
                self._emitZeroBlkAmaxSlice(module, blkAmax, nBase, g, qi)
            # Watermark: global load count through the last in-flight prefetch unit.
            watermarkIdx = min(i + pfd - 1, numUnits - 1)
            issuedWatermark = issuedThroughUnit[watermarkIdx]
            numTilesInUnit = g * tpb
            localLoadsCum = [globalLoadsCum[tileStarts[i] + t] for t in range(numTilesInUnit)]
            module.add(self._computeUnitFromBank(
                vgprTiles, accBank, resRing[i % pfd], gammaBank, blkAmax,
                issuedWatermark, localLoadsCum, mBaseV, qi, nBase, g, pathInterior))
            if self.useMxfp8:
                module.add(self._mxDeferredTail(vgprTiles, blkAmax, qi, nBase, g))
            # Prefetch the next unit AFTER compute+tail so scratch regs do not collide.
            prefetchIdx = i + pfd
            if prefetchIdx < numUnits:
                pqi, pnBase, pg = units[prefetchIdx]
                module.add(self._issueUnitLoads(resRing[prefetchIdx % pfd], mBaseV,
                                                pqi, pnBase, pg, pathInterior))
            # Ping-pong: stage next qi's gamma then read it immediately from LDS so the
            # DS read lands directly after the visibility barrier with no intervening vmem.
            if (self.gammaLdsStaging and self.gammaBuffers > 1
                    and nBase + g == self.mmaN and qi + 1 < self.nQTilesM):
                nextBufIdx = (qi + 1) % self.gammaBuffers
                self._stageGammaToLds(module, qi + 1, nextBufIdx)
                self._ldsReadGammaBlock(module, gammaBank, qi + 1, nextBufIdx)
        module.addComment1("MF end _emitUnitIteration.")


    def _emitFusedBody(self, module, vgprTiles, units, resRing, accBank, gammaBank,
                       blkAmax, mBaseV, globalLoadsCum, issuedThroughUnit,
                       tileStarts, pfd, pathInterior: bool = False) -> None:
        """Issue the prolog loads then iterate all units through the PFD pipeline.

        pathInterior selects the dwordx4 bf16 load/store path (interior arm) vs
        the scalar/wide masked path (tail arm), per design §5.
        """
        module.addComment1("MF begin _emitFusedBody: prolog loads then PFD pipeline iteration.")
        if self.gammaLdsStaging:
            self._stageGammaToLds(module, 0, 0)
            # Read qi=0 gamma from LDS immediately after the visibility barrier so the
            # DS read lands with no intervening vmem instructions before it.
            self._ldsReadGammaBlock(module, gammaBank, 0, 0)
        self._emitProlog(module, units, pfd, resRing, mBaseV, pathInterior)
        self._emitUnitIteration(module, vgprTiles, units, resRing, accBank, gammaBank,
                                blkAmax, mBaseV, globalLoadsCum, issuedThroughUnit,
                                tileStarts, pfd, pathInterior)
        module.addComment1("MF end _emitFusedBody.")


    def _emitWgMaxRowCmp(self, module) -> None:
        """Compute wgMaxRow = WorkGroup0*MT0 + (MT0-1) and set SCC = (wgMaxRow < N_hidden).

        MT0 is not always a power of two (valid values include 384, 320, 192),
        so SMulI32 is used for the general case.  Leaves SCC = 1 when the entire
        tile is interior (no row straddles the boundary).
        """
        module.addComment1("MF begin _emitWgMaxRowCmp: compute wgMaxRow and set SCC for interior/tail branch.")
        mt0 = self.macro_tile0
        with self.writer.allocTmpSgpr(1, tag="mf_wgMaxRow") as wgMaxRowS:
            dst = sgpr(wgMaxRowS.idx)
            module.add(SMulI32(dst=dst, src0=sgpr("WorkGroup0"), src1=mt0,
                               comment=f"wgMaxRow_lo = WorkGroup0 * MT0={mt0}."))
            module.add(SAddU32(dst=dst, src0=dst, src1=mt0 - 1,
                               comment=f"wgMaxRow = WorkGroup0*MT0 + (MT0-1)."))
            module.add(SCmpLtU32(src0=dst, src1=sgpr("SizesFree+0"),
                                 comment="SCC = (wgMaxRow < N_hidden): interior path safe."))
        module.addComment1("MF end _emitWgMaxRowCmp.")


    def _emitInteriorTailBranch(self, module, emitBodyFn) -> None:
        """Emit a workgroup-uniform branch selecting the interior or tail body.

        The predicate wgMaxRow = WorkGroup0*MT0 + (MT0-1) is the largest free0 row
        index any lane in this workgroup can reach.  When wgMaxRow < N_hidden the
        whole tile is interior (no row straddles the boundary), so the interior arm
        is taken; otherwise the tail arm handles the partial edge tile.

        Both arms call emitBodyFn with identical arguments in Increment 2a — the
        intentional body duplication is the branch skeleton for the dwordx4 interior
        arm that lands in Increment 2b (design §5).  The branch is scalar and
        workgroup-uniform: exec is untouched, both arms run with full exec, and only
        one arm executes at runtime per workgroup.
        """
        module.addComment1("MF begin _emitInteriorTailBranch: workgroup-uniform interior/tail branch.")
        tailLabel = Label(self.writer.labels.getNameInc("mf_interiorTail_tail"),
                          "tail arm entry (wgMaxRow >= N_hidden).")
        endLabel  = Label(self.writer.labels.getNameInc("mf_interiorTail_end"),
                          "interior/tail merge point.")

        # Compute wgMaxRow and set SCC; SCC = 1 means interior-safe.
        self._emitWgMaxRowCmp(module)

        # Branch to tail when SCC == 0 (wgMaxRow >= N_hidden).
        module.add(SCBranchSCC0(labelName=tailLabel.getLabelName(),
                                comment="branch to tail when wgMaxRow >= N_hidden."))

        # Interior arm — falls through from the branch.
        module.addComment0("Interior arm (wgMaxRow < N_hidden): all rows fit, no boundary straddle.")
        emitBodyFn(module, pathInterior=True)
        module.add(SBranch(labelName=endLabel.getLabelName(),
                           comment="interior done; skip tail arm."))

        # Tail arm — workgroups whose last row straddles or exceeds N_hidden.
        module.add(tailLabel)
        module.addComment0("Tail arm (wgMaxRow >= N_hidden): boundary tile, scalar masked path.")
        emitBodyFn(module, pathInterior=False)

        module.add(endLabel)
        module.addComment1("MF end _emitInteriorTailBranch.")


    def _emitTeardown(self, module, gammaBank) -> None:
        """End stream context, drain stores, free residual scratch and gamma bank."""
        module.addComment1("MF begin _emitTeardown: drain stores, free residual scratch and gamma bank.")
        if self.useMxfp8:
            self._endStreamContext()
        # One vscnt=0 drains both MXScale stores (from _subColStoreGroup) and ResidualOut stores.
        module.add(SWaitCnt(vscnt=0, comment="drain MXScale and ResidualOut stores."))
        self._endResidualScratch(module)
        self.writer.vgprPool.checkIn(gammaBank)
        module.addComment1("MF end _emitTeardown.")


    def emit(self, vgprTiles):
        assert not self.useMxfp8 or self.subColQuant, \
            "megaFused MXFP8 requires subColQuant (q1 < mfmaN)"
        ebc = self.epilogueBatchCols
        pfd = self.prefetchDepth
        s = self.tilesPerBlockM * self.rowsPerLane
        module = Module("SubtileMegaFusedEpilogue")
        module.addComment0(
            f"MegaFused PFD ring: EBC={ebc} PFD={pfd} S={s} "
            f"(nQTilesM={self.nQTilesM} mmaN={self.mmaN}).")
        module.addComment0("MF begin emit: top-level MegaFused epilogue emission.")
        self._allocSharedRegs()
        module.add(self._setupShared())
        gammaBank = self._emitSharedSetup(module)
        units, globalLoadsCum, issuedThroughUnit, tileStarts = self._buildUnitSequence(ebc)
        resRing, accBank, blkAmax, mBaseV = self._allocRing(ebc, pfd)
        # Banks are allocated once and shared by both branch arms; only one arm
        # executes at runtime (the branch is workgroup-uniform), so VGPR usage
        # is unchanged relative to a single-path emit.
        emitBodyFn = lambda mod, pathInterior: self._emitFusedBody(
            mod, vgprTiles, units, resRing, accBank, gammaBank, blkAmax,
            mBaseV, globalLoadsCum, issuedThroughUnit, tileStarts, pfd, pathInterior)
        self._emitInteriorTailBranch(module, emitBodyFn)
        self._freeRing(resRing, accBank, blkAmax, mBaseV, pfd)
        self._emitTeardown(module, gammaBank)
        module.add(self._reduceAndWriteRms())
        self._freeSharedRegs()
        module.addComment0("MF end emit.")
        return module


    def _addWaveNColByte(self, module, colByte: int) -> None:
        if self.wg_n <= 1:
            return
        module.addComment1("MF begin _addWaveNColByte: add waveN column byte offset to colByte.")
        waveN = self.writer.vgprPool.checkOut(1, tag="rAdd_setupWaveN")
        tmpVgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="rAdd_setupTmp")
        tmpRes = ContinuousRegister(tmpVgpr, 2)
        module.add(vectorStaticDivide(waveN, "Serial", self.waveSize * self.wg_m, tmpRes,
                                      comment=f"waveN = Serial / {self.waveSize * self.wg_m}"))
        colBaseBytes = self.mma_n * self.mfma_n * self.elemBytes
        with self.writer.allocTmpSgpr(1, tag="rAdd_setupColBase") as tmpSgprInfo:
            module.add(SMovB32(dst=sgpr(tmpSgprInfo.idx), src=hex(colBaseBytes),
                               comment=f"col base bytes per wave ({colBaseBytes})"))
            module.add(VMulLOU32(dst=vgpr(waveN), src0=sgpr(tmpSgprInfo.idx), src1=vgpr(waveN),
                                 comment="waveN * mma_n * mfma_n * elemBytes"))
        module.add(VAddU32(vgpr(colByte), vgpr(colByte), vgpr(waveN),
                           comment="colByte += wave column base"))
        self.writer.vgprPool.checkIn(tmpVgpr)
        self.writer.vgprPool.checkIn(waveN)
        module.addComment1("MF end _addWaveNColByte.")


    def _buildResidualOutSrd(self, module, srd: int) -> None:
        # ResidualOut SRD bounds = M_tokens*N_hidden*2 (bf16) so OOB lanes store to /dev/null.
        module.addComment1("MF begin _buildResidualOutSrd: build ResidualOut SRD with OOB bounds.")
        with self.writer.allocTmpSgpr(1, tag="rAdd_roSrdNumRec") as tmpSgpr:
            module.add(SMovB64(dst=sgpr(srd, 2), src=sgpr("AddressResidualOut", 2),
                               comment="ResidualOut SRD base."))
            module.add(SMulI32(dst=sgpr(tmpSgpr.idx), src0=sgpr("SizesFree+0"),
                               src1=sgpr("SizesFree+1"), comment="numRecords = N_hidden * M_tokens"))
            module.add(SLShiftLeftB32(dst=sgpr(srd + 2), src=sgpr(tmpSgpr.idx),
                                      shiftHex=hex(1), comment="numRecords *= 2 (bf16)."))
        module.add(SMovB32(dst=sgpr(srd + 3), src="Srd127_96",
                           comment="ResidualOut SRD flags."))
        module.addComment1("MF end _buildResidualOutSrd.")


    def _buildResidualSrd(self, module, resSrd: int) -> None:
        # Residual SRD bounds = M_tokens*N_hidden*elemBytes so tail-WG OOB lanes read 0.
        module.addComment1("MF begin _buildResidualSrd: build residual SRD with OOB bounds.")
        with self.writer.allocTmpSgpr(1, tag="rAdd_resSrdNumRec") as tmpSgpr:
            module.add(SMovB64(dst=sgpr(resSrd, 2), src=sgpr("ResidualBuf", 2),
                               comment="residual SRD base"))
            module.add(SMulI32(dst=sgpr(tmpSgpr.idx), src0=sgpr("SizesFree+0"),
                               src1=sgpr("SizesFree+1"), comment="numRecords = N_hidden * M_tokens"))
            module.add(SLShiftLeftB32(dst=sgpr(resSrd + 2), src=sgpr(tmpSgpr.idx),
                                      shiftHex=hex(self.residualLog2Bytes),
                                      comment="numRecords *= residualBytes."))
        module.add(SMovB32(dst=sgpr(resSrd + 3), src="Srd127_96", comment="residual SRD flags"))
        module.addComment1("MF end _buildResidualSrd.")


    def _convertResidualChunkBf16(self, module, base: int) -> None:
        # 4 bf16 in dwords (base, base+1) -> 4 f32; high indices first so each source
        # dword is fully read before it is overwritten.
        # sel 0 = low16 (WORD_0), sel 1 = high16 (WORD_1).
        module.addComment1("MF begin _convertResidualChunkBf16: expand 4 packed bf16 dwords to 4 f32.")
        module.add(VCvtBF16toFP32(vgpr(base + 3), vgpr(base + 1), None, 1,
                                  comment="residual k=3 bf16(hi) -> f32."))
        module.add(VCvtBF16toFP32(vgpr(base + 2), vgpr(base + 1), None, 0,
                                  comment="residual k=2 bf16(lo) -> f32."))
        module.add(VCvtBF16toFP32(vgpr(base + 1), vgpr(base + 0), None, 1,
                                  comment="residual k=1 bf16(hi) -> f32."))
        module.add(VCvtBF16toFP32(vgpr(base + 0), vgpr(base + 0), None, 0,
                                  comment="residual k=0 bf16(lo) -> f32."))
        module.addComment1("MF end _convertResidualChunkBf16.")


    def _convertResidualChunkFp8(self, module, base: int) -> None:
        cvt = ECvtPkFP8toF32 if self.residualType.isAnyFloat8() else ECvtPkBF8toF32
        # HIGH before LOW: HIGH reads the packed dword at base; LOW then overwrites base.
        module.addComment1("MF begin _convertResidualChunkFp8: expand packed fp8 dword to 4 f32.")
        module.add(cvt(dst=vgpr(base + 2, 2), src=vgpr(base), sel=HighBitSel.HIGH,
                       comment="residual pair (k=2,3) fp8 -> f32."))
        module.add(cvt(dst=vgpr(base, 2), src=vgpr(base), sel=HighBitSel.LOW,
                       comment="residual pair (k=0,1) fp8 -> f32."))
        module.addComment1("MF end _convertResidualChunkFp8.")


    def _convertSideElem(self, module, dstVgpr: int, comment: str, dtype) -> None:
        """Convert an already-loaded side element to fp32 in place (no-op for f32)."""
        if dtype.isSingle():
            return
        module.addComment1("MF begin _convertSideElem: in-place conversion of side element to f32.")
        if dtype.isHalf():
            module.add(VCvtF16toF32(vgpr(dstVgpr), vgpr(dstVgpr), comment=comment))
        elif dtype.isAnyFloat8():
            module.add(VCvtFP8toF32(dst=vgpr(dstVgpr), src=vgpr(dstVgpr), comment=comment))
        elif dtype.isAnyBFloat8():
            module.add(VCvtBF8toF32(dst=vgpr(dstVgpr), src=vgpr(dstVgpr), comment=comment))
        else:
            module.add(VCvtBF16toFP32(vgpr(dstVgpr), vgpr(dstVgpr), None, 0, comment=comment))
        module.addComment1("MF end _convertSideElem.")


    def _issueSideLoad(self, module, dstVgpr: int, addrVgpr: int, srd: int,
                       comment: str, dtype) -> None:
        """Issue one side-input buffer_load without waiting (burst-friendly)."""
        module.addComment1("MF begin _issueSideLoad: issue one side-input buffer_load.")
        loadCls = self._sideLoadClass(dtype)
        module.add(loadCls(vgpr(dstVgpr), vgpr(addrVgpr), sgpr(srd, 4), 0,
                           MUBUFModifiers(offen=True), comment=comment))
        module.addComment1("MF end _issueSideLoad.")


    def _residualElemAddr(self, module, resAddr: int, rowByteBase: int, nhiddenBase: int,
                          rowGroupOff: int, oobV: int, oobMask: int, scratch: int,
                          m: int, k: int) -> None:
        module.addComment1(f"MF begin _residualElemAddr: compute clamped residual byte address (m={m},k={k}).")
        self._free0RowPos(module, resAddr, nhiddenBase, rowGroupOff, m, k, scratch)
        module.add(VCmpLtU32(dst=sgpr(oobMask, self.lane_sgpr_count), src0=vgpr(resAddr),
                             src1=sgpr("SizesFree+0"), comment="inRange = nhidden_pos < N_hidden"))
        module.add(VLShiftLeftB32(dst=vgpr(resAddr), shiftHex=hex(self.residualLog2Bytes),
                                  src=vgpr(resAddr),
                                  comment="nhiddenByte = nhidden_pos * residualBytes."))
        module.add(VAddU32(vgpr(resAddr), vgpr(resAddr), vgpr(rowByteBase),
                           comment="byteAddr = rowByteBase + nhiddenByte"))
        module.add(VCndMaskB32(dst=vgpr(resAddr), src0=vgpr(oobV), src1=vgpr(resAddr),
                               src2=sgpr(oobMask, self.lane_sgpr_count),
                               comment="clamp OOB when nhidden_pos >= N_hidden"))
        module.addComment1("MF end _residualElemAddr.")


    def _residualRowByteBase(self, module, dst: int, tokenBase: int, n: int, scratch: int) -> None:
        """Compute the residual row byte base for token group n into dst.

        For wide/dwordx4 configs (useWideResidual), the per-lane invariant row origin
        (wgRowBase + rowGroupOff) is folded into the base before the byte shift, matching
        the ResidualOut store's roColByteBase model.  Per-tile addressing then uses a
        compile-time offset12 = m*mfma_m*residualBytes rather than a per-tile VALU add.
        The scalar path (not useWideResidual) omits the fold and uses the original
        per-element address helper.
        """
        module.addComment1(f"MF begin _residualRowByteBase: compute row byte base for residual (n={n}).")
        nOff = n * self.mfma_n
        r = self._addImmU32(module, dst, tokenBase, nOff, scratch, f"token_n = tokenBase + {nOff} (n={n}).")
        # token_n * SizesFree0 uses 32-bit VMulLOU32; valid while the element index
        # token_n * N_hidden stays below 2^32 (all currently supported tensor sizes).
        module.add(VMulLOU32(dst=vgpr(dst), src0=sgpr("SizesFree+0"), src1=vgpr(r),
                             comment="token_n * SizesFree0."))
        if self.useWideResidual:
            # GWB base+offset parity (mirrors the ResidualOut store's roColByteBase):
            # fold the per-lane row origin into the per-n base so wide/dwordx4 tiles
            # address via a compile-time offset12 = m*mfma_m*bytes.
            module.add(VAddU32(vgpr(dst), vgpr(dst), vgpr(self.wgRowBase),
                               comment="+ wgRowBase (fold row origin)."))
            module.add(VAddU32(vgpr(dst), vgpr(dst), vgpr(self.rowGroupOff),
                               comment="+ rowGroupOff (fold row origin)."))
        module.add(VLShiftLeftB32(dst=vgpr(dst), shiftHex=hex(self.residualLog2Bytes), src=vgpr(dst),
                                  comment="rowByteBase * residualBytes (row origin folded when wide)."))
        module.addComment1("MF end _residualRowByteBase.")


    @staticmethod
    def _sideBytes(dtype):
        """Return (bytes, log2Bytes) for a side-input element type."""
        if dtype.isSingle():
            return 4, 2
        if dtype.isAnyFloat8() or dtype.isAnyBFloat8():
            return 1, 0
        return 2, 1  # bf16 / f16.


    def _sideLoadClass(self, dtype):
        if dtype.isSingle():
            return BufferLoadB32
        if dtype.isAnyFloat8() or dtype.isAnyBFloat8():
            return BufferLoadD16U8
        return BufferLoadD16B16  # bf16 / f16.


    def _addImmU32(self, module, dst: int, src: int, imm: int, scratch: int, comment: str) -> int:
        """Compute src + imm; return the register holding the result.

        When imm == 0 nothing is emitted and src is returned, so the caller reads
        src directly instead of a redundant copy in dst.
        """
        if imm == 0:
            return src
        module.addComment1("MF begin _addImmU32: compute src + imm into dst.")
        # Materialize the immediate in a VGPR when it exceeds the inline-literal range.
        if imm > _INLINE_CONST_MAX:
            module.add(VMovB32(dst=vgpr(scratch), src=imm, comment=f"imm={imm}"))
            module.add(VAddU32(vgpr(dst), vgpr(src), vgpr(scratch), comment=comment))
            module.addComment1("MF end _addImmU32.")
            return dst
        module.add(VAddU32(vgpr(dst), vgpr(src), imm, comment=comment))
        module.addComment1("MF end _addImmU32.")
        return dst


    def _free0RowPos(self, module, dst: int, rowBase: int, rowGroupOff: int,
                     m: int, k: int, scratch: int) -> None:
        # free0 row = rowBase + rowGroupOff + (m*mfma_m + k). One v_add3_u32 folds
        # the row-base, the m/k immediate, and rowGroupOff into a single VALU op.
        module.addComment1(f"MF begin _free0RowPos: compute free0 row position (m={m},k={k}).")
        mBase = m * self.mfma_m + k
        if mBase == 0:
            module.add(VAddU32(vgpr(dst), vgpr(rowBase), vgpr(rowGroupOff),
                               comment=f"row = base + rowGroupOff (m={m},k={k})"))
            module.addComment1("MF end _free0RowPos.")
            return
        if mBase <= _INLINE_CONST_MAX:
            module.add(VAdd3U32(dst=vgpr(dst), src0=vgpr(rowBase), src1=vgpr(rowGroupOff),
                                src2=mBase,
                                comment=f"row = base + {mBase} + rowGroupOff (m={m},k={k})"))
            module.addComment1("MF end _free0RowPos.")
            return
        # mBase exceeds the inline-constant range: materialize it, then fold in one add3.
        module.add(VMovB32(dst=vgpr(scratch), src=mBase, comment=f"imm={mBase}"))
        module.add(VAdd3U32(dst=vgpr(dst), src0=vgpr(rowBase), src1=vgpr(rowGroupOff),
                            src2=vgpr(scratch),
                            comment=f"row = base + {mBase} + rowGroupOff (m={m},k={k})"))
        module.addComment1("MF end _free0RowPos.")


    def _buildBufferSrd(self, module, srd: int, ptrName: str, name: str) -> None:
        module.addComment1(f"MF begin _buildBufferSrd: build generic buffer SRD for {name}.")
        module.add(SMovB64(dst=sgpr(srd, 2), src=sgpr(ptrName, 2), comment=f"{name} SRD base."))
        module.add(SMovB32(dst=sgpr(srd + 2), src="BufferOOB", comment=f"{name} SRD limit."))
        module.add(SMovB32(dst=sgpr(srd + 3), src="Srd127_96", comment=f"{name} SRD flags."))
        module.addComment1("MF end _buildBufferSrd.")


    def _buildWriteMask(self, module, laneMaskSgpr: int, laneId: int) -> None:
        # Active iff rowGroup==0 AND waveM==0 (every lane already holds the all-reduced value).
        module.addComment1("MF begin _buildWriteMask: compute lane mask for rowGroup==0 and waveM==0.")
        log2MfmaN = int(math.log2(self.mfma_n))
        rgV = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0RowGroup")
        module.add(VLShiftRightB32(dst=vgpr(rgV), shiftHex=hex(log2MfmaN), src=vgpr(laneId),
                                   comment=f"rowGroup = laneId >> {log2MfmaN}"))
        if self.wg_m > 1:
            waveMv = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0WaveM")
            self._computeWaveM(module, waveMv)
            module.add(VOrB32(dst=vgpr(rgV), src0=vgpr(rgV), src1=vgpr(waveMv),
                              comment="selV = rowGroup | waveM (zero iff both zero)"))
            self.writer.vgprPool.checkIn(waveMv)
        module.add(VCmpEQU32(dst=sgpr(laneMaskSgpr, self.lane_sgpr_count), src0=0, src1=vgpr(rgV),
                             comment="laneMask: rowGroup==0 && waveM==0"))
        self.writer.vgprPool.checkIn(rgV)
        module.addComment1("MF end _buildWriteMask.")


    def _computeFree0RowBase(self, module, dst: int) -> None:
        # rowBase = WorkGroup0 * MT0 (+ waveM * mma_m*mfma_m when wg_m > 1).
        module.addComment1("MF begin _computeFree0RowBase: compute free0 row base for this wave.")
        mt0Vgpr = self.writer.vgprPool.checkOut(1, tag="pRMS_rbMT0")
        module.add(VMovB32(dst=vgpr(mt0Vgpr), src=self.macro_tile0, comment=f"MT0={self.macro_tile0}"))
        module.add(VMulLOU32(dst=vgpr(dst), src0=vgpr(mt0Vgpr), src1=sgpr("WorkGroup0"),
                             comment="rowBase = WorkGroup0 * MT0"))
        self.writer.vgprPool.checkIn(mt0Vgpr)
        if self.wg_m <= 1:
            module.addComment1("MF end _computeFree0RowBase.")
            return
        waveM = self.writer.vgprPool.checkOut(1, tag="pRMS_rbWaveM")
        self._computeWaveM(module, waveM)
        waveStride = self.mma_m * self.mfma_m
        strideV = self.writer.vgprPool.checkOut(1, tag="pRMS_rbStride")
        module.add(VMovB32(dst=vgpr(strideV), src=waveStride,
                           comment=f"waveStride = mma_m * mfma_m = {waveStride}"))
        module.add(VMulLOU32(dst=vgpr(waveM), src0=vgpr(strideV), src1=vgpr(waveM),
                             comment="waveMOff = waveM * waveStride"))
        module.add(VAddU32(vgpr(dst), vgpr(dst), vgpr(waveM), comment="rowBase += waveMOff"))
        self.writer.vgprPool.checkIn(strideV)
        self.writer.vgprPool.checkIn(waveM)
        module.addComment1("MF end _computeFree0RowBase.")


    def _computeNTiles(self, module, dst: int) -> None:
        # n_d = ceil(SizesFree0 / MT0).
        module.addComment1("MF begin _computeNTiles: compute n_d = ceil(SizesFree0 / MT0).")
        with self.writer.allocTmpSgpr(1, tag="pRMS_wF0NTilesS") as ntilesS:
            module.add(SAddU32(dst=sgpr(ntilesS.idx), src0=sgpr("SizesFree+0"),
                               src1=self.macro_tile0 - 1,
                               comment=f"N_hidden + MT0-1 (MT0={self.macro_tile0})"))
            mt0 = self.macro_tile0
            if mt0 & (mt0 - 1) == 0:
                module.add(SLShiftRightB32(dst=sgpr(ntilesS.idx), shiftHex=hex(mt0.bit_length() - 1),
                                           src=sgpr(ntilesS.idx),
                                           comment=f"n_d = ceil(SizesFree0 / MT0={mt0})"))
            else:
                magic, postShift = _ceilDivMagic(mt0)
                module.add(SMulHIU32(dst=sgpr(ntilesS.idx), src0=sgpr(ntilesS.idx), src1=hex(magic),
                                     comment=f"n_d magic mul (divisor={mt0})"))
                if postShift:
                    module.add(SLShiftRightB32(dst=sgpr(ntilesS.idx), shiftHex=hex(postShift),
                                               src=sgpr(ntilesS.idx),
                                               comment=f"n_d >> {postShift} (magic post-shift)"))
            module.add(VMovB32(dst=vgpr(dst), src=sgpr(ntilesS.idx), comment="ntilesV = n_d"))
        module.addComment1("MF end _computeNTiles.")


    def _computeRowGroupOff(self, module, dst: int) -> None:
        # Reuse the cached self.laneId instead of recomputing Serial & (waveSize-1).
        module.addComment1("MF begin _computeRowGroupOff: compute rowGroupOff = rowGroup * rows_per_lane.")
        log2MfmaN = int(math.log2(self.mfma_n))
        module.add(VLShiftRightB32(dst=vgpr(dst), shiftHex=hex(log2MfmaN), src=vgpr(self.laneId),
                                   comment=f"rowGroup = laneId >> {log2MfmaN}"))
        module.add(VMulLOU32(dst=vgpr(dst), src0=self.rows_per_lane, src1=vgpr(dst),
                             comment=f"rowGroupOff = rowGroup * {self.rows_per_lane}"))
        module.addComment1("MF end _computeRowGroupOff.")


    def _computeWaveM(self, module, dst: int) -> None:
        # waveId is cached once in _setup (self.waveIdV); only callers with wg_m > 1
        # reach here, so self.waveIdV is always valid.
        module.addComment1("MF begin _computeWaveM: compute waveM = waveId mod wg_m.")
        module.add(VAndB32(dst=vgpr(dst), src0=vgpr(self.waveIdV), src1=self.wg_m - 1,
                           comment=f"waveM = waveId % {self.wg_m}"))
        module.addComment1("MF end _computeWaveM.")


    def _convertGammaChunkBf16(self, module, base: int) -> None:
        """Unpack 4 bf16 from dwords (base, base+1) into 4 f32 at base+0..3.

        Mirrors _convertResidualChunkBf16 in SubtileResidualAddEmit.py.
        High indices first so each source dword is fully read before overwritten.
        """
        module.addComment1("MF begin _convertGammaChunkBf16: expand 4 packed bf16 gamma dwords to 4 f32.")
        module.add(VCvtBF16toFP32(vgpr(base + 3), vgpr(base + 1), None, 1,
                                   comment="gamma k=3 bf16(hi) -> f32."))
        module.add(VCvtBF16toFP32(vgpr(base + 2), vgpr(base + 1), None, 0,
                                   comment="gamma k=2 bf16(lo) -> f32."))
        module.add(VCvtBF16toFP32(vgpr(base + 1), vgpr(base + 0), None, 1,
                                   comment="gamma k=1 bf16(hi) -> f32."))
        module.add(VCvtBF16toFP32(vgpr(base + 0), vgpr(base + 0), None, 0,
                                   comment="gamma k=0 bf16(lo) -> f32."))
        module.addComment1("MF end _convertGammaChunkBf16.")


    def _crossWaveAccum(self, module, readTmp: int, arrays, j: int) -> None:
        # j==0 loads go directly into base+i (see _crossWaveLoadReduce); only j>0 reaches here.
        module.addComment1(f"MF begin _crossWaveAccum: accumulate wave[{j}] partials into base.")
        for a, (base, op, verb) in enumerate(arrays):
            for i in range(self.numPartials):
                src = readTmp + a * self.numPartials + i
                module.add(op(dst=vgpr(base + i), src0=vgpr(base + i), src1=vgpr(src),
                              comment=f"arr[{a}] partial[{i}] {verb} wave[{j}]."))
        module.addComment1("MF end _crossWaveAccum.")


    def _crossWaveComputeAddrs(self, module, writeAddr: int, readAddr: int,
                               numArrays: int = 1) -> None:
        # Only reached when wg_m > 1, so self.waveIdV is valid. Reuse the cached
        # waveId and laneId instead of recomputing them here.
        module.addComment1("MF begin _crossWaveComputeAddrs: compute LDS write/read addresses for cross-wave reduction.")
        laneSlotBytes = numArrays * self.numPartials * 4
        strideW = self.waveSize * laneSlotBytes
        waveM = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0WaveM")
        readBaseWave = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0ReadBase")
        laneLoc = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0Lane")
        # laneLoc is mutated in place below, so copy the cached laneId into it.
        module.add(VMovB32(dst=vgpr(laneLoc), src=vgpr(self.laneId),
                           comment="laneId for LDS addressing (cached)"))
        module.add(VAndB32(dst=vgpr(waveM), src0=vgpr(self.waveIdV), src1=self.wg_m - 1,
                           comment=f"waveM = waveId % {self.wg_m}"))
        # readBaseWave = waveId XOR waveM = waveN * wg_m.
        module.add(VXorB32(dst=vgpr(readBaseWave), src0=vgpr(self.waveIdV), src1=vgpr(waveM),
                           comment="readBaseWave = waveN * wg_m"))
        with self.writer.allocTmpSgpr(1, tag="pRMS_xwF0AddrSetup") as tmpSgprInfo:
            tmpSgpr = tmpSgprInfo.idx
            module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(strideW), comment=f"strideW={strideW}"))
            module.add(VMulLOU32(dst=vgpr(writeAddr), src0=sgpr(tmpSgpr), src1=vgpr(self.waveIdV),
                                 comment="writeAddr = waveId * strideW"))
            module.add(VMulLOU32(dst=vgpr(readAddr), src0=sgpr(tmpSgpr), src1=vgpr(readBaseWave),
                                 comment="readAddr = readBaseWave * strideW"))
            module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(laneSlotBytes),
                               comment=f"laneSlotBytes={laneSlotBytes}"))
            module.add(VMulLOU32(dst=vgpr(laneLoc), src0=sgpr(tmpSgpr), src1=vgpr(laneLoc),
                                 comment="lane * laneSlotBytes"))
            module.add(VAddU32(vgpr(writeAddr), vgpr(writeAddr), vgpr(laneLoc),
                               comment="writeAddr += lane*laneSlotBytes"))
            module.add(VAddU32(vgpr(readAddr), vgpr(readAddr), vgpr(laneLoc),
                               comment="readAddr += lane*laneSlotBytes"))
        self.writer.vgprPool.checkIn(laneLoc)
        self.writer.vgprPool.checkIn(readBaseWave)
        self.writer.vgprPool.checkIn(waveM)
        module.addComment1("MF end _crossWaveComputeAddrs.")


    def _crossWaveLoadReduce(self, module, readAddr: int, readTmp: int, arrays, strideW: int) -> None:
        # TODO(perf): prefetch wave[j+1]'s LDS loads while accumulating wave[j] to
        # overlap load and compute. Deferred: needs a second readTmp buffer.
        module.addComment1("MF begin _crossWaveLoadReduce: LDS-staged cross-wave load and reduce.")
        numArrays = len(arrays)
        for j in range(self.wg_m):
            for a, (base, _op, _verb) in enumerate(arrays):
                for i in range(self.numPartials):
                    off = (a * self.numPartials + i) * 4
                    # For j==0 load directly into the accumulator base to avoid a
                    # redundant VMovB32 copy; for j>0 use the temp buffer.
                    dst = (base + i) if j == 0 else (readTmp + a * self.numPartials + i)
                    module.add(DSLoadB32(dst=vgpr(dst), src=vgpr(readAddr), ds=DSModifiers(offset=off),
                                         comment=f"LDS load wave[{j}] arr[{a}] partial[{i}]."))
            module.add(SWaitCnt(dscnt=0, comment="wait LDS reads."))
            if j > 0:
                self._crossWaveAccum(module, readTmp, arrays, j)
            if j < self.wg_m - 1:
                with self.writer.allocTmpSgpr(1, tag="pRMS_xwF0Advance") as tmpSgprInfo:
                    module.add(SMovB32(dst=sgpr(tmpSgprInfo.idx), src=hex(strideW),
                                       comment=f"strideW={strideW}."))
                    module.add(VAddU32(vgpr(readAddr), vgpr(readAddr), sgpr(tmpSgprInfo.idx),
                                       comment="advance readAddr to next sibling wave."))
        module.addComment1("MF end _crossWaveLoadReduce.")


    def _crossWaveReduceFree0(self, arrays) -> Module:
        # Step 3 (free0): reduce every array in `arrays` across wg_m sibling waves
        # in a single LDS pass so the three barriers are shared, not paid per array.
        # arrays: list of (baseVgpr, op, verb); array a occupies lane-slot dwords
        # [a*numPartials, (a+1)*numPartials).
        numArrays = len(arrays)
        laneSlotBytes = numArrays * self.numPartials * 4
        strideW = self.waveSize * laneSlotBytes
        module = Module("PartialRMS crossWaveReduceFree0")
        module.addComment0("MF begin _crossWaveReduceFree0: fused cross-wave LDS reduction pass.")
        module.addComment1(
            f"PartialRMS step 3 (free0): cross-wave LDS reduction over wg_m={self.wg_m}, "
            f"arrays={numArrays}."
        )
        module.add(self.writer._syncThreads(
            self.kernel,
            "partialRMS free0 cross-wave: ensure siblings done reading LDS before scratch write."))
        writeAddr = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0WriteAddr")
        readAddr = self.writer.vgprPool.checkOut(1, tag="pRMS_xwF0ReadAddr")
        readTmp = self.writer.vgprPool.checkOut(numArrays * self.numPartials, tag="pRMS_xwF0ReadTmp")
        self._crossWaveComputeAddrs(module, writeAddr, readAddr, numArrays)
        self._crossWaveStore(module, writeAddr, arrays)
        module.add(SWaitCnt(dscnt=0, comment="wait LDS writes."))
        module.add(self.writer._syncThreads(self.kernel, "partialRMS free0 cross-wave write."))
        self._crossWaveLoadReduce(module, readAddr, readTmp, arrays, strideW)
        # No further LDS use after this point, so the post-read WAR barrier is
        # unnecessary; the next persistent-loop tile's main loop re-barriers
        # before its own first LDS write.
        self.writer.vgprPool.checkIn(readTmp)
        self.writer.vgprPool.checkIn(readAddr)
        self.writer.vgprPool.checkIn(writeAddr)
        module.addComment0("MF end _crossWaveReduceFree0.")
        return module


    def _crossWaveStore(self, module, writeAddr: int, arrays) -> None:
        module.addComment1("MF begin _crossWaveStore: LDS store each array partial for cross-wave reduction.")
        for a, (base, _op, _verb) in enumerate(arrays):
            for i in range(self.numPartials):
                off = (a * self.numPartials + i) * 4
                module.add(DSStoreB32(dstAddr=vgpr(writeAddr), src=vgpr(base + i),
                                      ds=DSModifiers(offset=off),
                                      comment=f"LDS store arr[{a}] partial[{i}]."))
        module.addComment1("MF end _crossWaveStore.")


    def _readAccBurst(self, module, dstBase: int, vgprTiles, coords, comment: str):
        """Read accumulator elements, staging AGPR tiles and returning VGPR tiles in place.

        coords is a list of (m, n, k) tile coordinates. AGPR elements are read into
        dstBase+0, dstBase+1, ... (packed); VGPR-resident elements are returned as
        their own register indices without emitting any copy. The returned srcRegs list
        has one entry per coord, in order.
        """
        module.addComment1("MF begin _readAccBurst: read accumulator burst into VGPR staging buffer.")
        srcRegs = []
        slot = 0
        for i, (m, n, k) in enumerate(coords):
            tile = vgprTiles[n * self.mma_m + m]
            reg = tile.regList.indices[k]
            if tile.regList.pool == self.writer.vgprPool:
                srcRegs.append(reg)
                continue
            module.add(VAccvgprReadB32(vgpr(dstBase + slot), accvgpr(reg),
                                       comment=f"{comment} [{i}]"))
            srcRegs.append(dstBase + slot)
            slot += 1
        # gfx950 requires one wait state between v_accvgpr_read_b32 and a dependent
        # VALU consumer. slot >= 2 fills that gap automatically via the subsequent read;
        # slot == 1 needs an explicit s_nop.
        if 0 < slot < 2:
            module.add(SNop(waitState=1, comment="fill the mandatory v_accvgpr_read->VALU wait state (gfx950)."))
        module.addComment1("MF end _readAccBurst.")
        return srcRegs


    def _reduceFree0(self) -> Module:
        # Row-group butterfly per array, then one fused cross-wave pass so the amax
        # and Σx² reductions share barriers instead of paying them per array.
        module = Module("PartialRMS reduceFree0")
        module.addComment0("MF begin _reduceFree0: row-group butterfly then optional cross-wave reduction.")
        # TODO(perf): fuse the Σx² and amax row-group butterflies to share the
        # partner-address computation and dscnt wait. Deferred for simplicity.
        module.add(self._rowGroupReduceFree0(self.partials))
        if self.wg_m <= 1:
            module.addComment0("MF end _reduceFree0.")
            return module
        reduceArrays = [(self.partials, VAddF32, "+")]
        module.add(self._crossWaveReduceFree0(reduceArrays))
        module.addComment0("MF end _reduceFree0.")
        return module


    def _rowGroupReduceFree0(self, partials: int, op=VAddF32, verb="+") -> Module:
        # Step 2 (free0): all-reduce partial[n] across row groups via ds_bpermute XOR butterfly.
        numRounds = int(math.log2(self.waveSize // self.mfma_n))
        module = Module("PartialRMS rowGroupReduceFree0")
        module.addComment0("MF begin _rowGroupReduceFree0: XOR butterfly row-group reduction.")
        module.addComment1(
            f"PartialRMS step 2 (free0): XOR butterfly over {self.waveSize // self.mfma_n} row groups"
        )
        if numRounds == 0:
            module.addComment0("MF end _rowGroupReduceFree0.")
            return module

        addrV = self.writer.vgprPool.checkOut(1, tag="pRMS_rgrAddr")
        tmpV = self.writer.vgprPool.checkOut(self.numPartials, tag="pRMS_rgrTmp")

        for i in range(numRounds):
            xorVal = self.mfma_n << i
            module.add(
                VXorB32(dst=vgpr(addrV), src0=vgpr(self.laneId), src1=xorVal,
                        comment=f"partnerLane = laneId ^ {xorVal}")
            )
            module.add(
                VLShiftLeftB32(dst=vgpr(addrV), shiftHex=hex(2), src=vgpr(addrV),
                               comment="byteAddr = partnerLane * 4")
            )
            for n in range(self.numPartials):
                module.add(
                    DSBPermuteB32(vgpr(tmpV + n), vgpr(addrV), vgpr(partials + n),
                                  comment=f"fetch partner partial[{n}]")
                )
            module.add(SWaitCnt(dscnt=0, comment="wait ds_bpermute"))
            for n in range(self.numPartials):
                module.add(
                    op(dst=vgpr(partials + n), src0=vgpr(partials + n),
                       src1=vgpr(tmpV + n), comment=f"partial[{n}] {verb} partner")
                )

        self.writer.vgprPool.checkIn(tmpV)
        self.writer.vgprPool.checkIn(addrV)
        module.addComment0("MF end _rowGroupReduceFree0.")
        return module


    def _writeAccFrom(self, module, src: int, vgprTiles, m: int, n: int, k: int, comment: str) -> None:
        """Write VGPR src back into accumulator element (m, n, k), selecting the right register file."""
        module.addComment1("MF begin _writeAccFrom: write VGPR back to accumulator register file.")
        tile = vgprTiles[n * self.mma_m + m]
        reg = tile.regList.indices[k]
        if tile.regList.pool == self.writer.vgprPool:
            if src != reg:
                module.add(VMovB32(dst=vgpr(reg), src=vgpr(src), comment=comment))
            module.addComment1("MF end _writeAccFrom.")
            return
        module.add(VAccvgprWriteB32(accvgpr(reg), vgpr(src), comment=comment))
        module.addComment1("MF end _writeAccFrom.")


    def _writePartialsFree0(self, partials: int, partialSrd: int, laneId: int, savedExec: int,
                            laneMaskSgpr: int, globalAddr: int, colByte: int,
                            label: str = "Σx²") -> Module:
        module = Module("PartialRMS writePartialsFree0")
        module.addComment0("MF begin _writePartialsFree0: predicated write of Σx² partials to partialBuf.")
        module.addComment1(
            "PartialRMS step 4 (free0): predicated write of Σx² to partialBuf[token, WG0]")
        lsc = self.lane_sgpr_count
        self._buildWriteMask(module, laneMaskSgpr, laneId)
        ntilesV = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0NTiles")
        self._computeNTiles(module, ntilesV)
        tokenBase = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0TokenBase")
        module.add(VLShiftRightB32(dst=vgpr(tokenBase), shiftHex=hex(self.log2ElemBytes),
                                   src=vgpr(colByte),
                                   comment="tokenBase = colByte >> log2ElemBytes."))
        module.add(SAndSaveExecB64(dst=sgpr(savedExec, lsc), src=sgpr(laneMaskSgpr, lsc),
                                   comment="save exec; set exec = writing-lane mask"))
        # Strength-reduce token*n_d across the n loop: token advances by mfma_n each
        # step, so token*n_d advances by the loop-invariant stride mfma_n*n_d. This
        # replaces the per-n multiply with a single add.
        accumV = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0Accum")
        # token*n_d uses 32-bit VMulLOU32; assumes token*n_d < 2^32.
        module.add(VMulLOU32(dst=vgpr(accumV), src0=vgpr(ntilesV), src1=vgpr(tokenBase),
                             comment="accum = tokenBase * n_d"))
        strideV = None
        if self.mma_n > 1:
            strideV = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0Stride")
            module.add(VMulLOU32(dst=vgpr(strideV), src0=self.mfma_n, src1=vgpr(ntilesV),
                                 comment=f"stride = mfma_n({self.mfma_n}) * n_d"))
        # Pre-compute byteAddr = (token*n_d + WG0) * 4 once, then stride by stride4 per n.
        module.add(VAddU32(vgpr(globalAddr), vgpr(accumV), sgpr("WorkGroup0"),
                           comment="token*n_d + WorkGroup0 (n=0)"))
        module.add(VLShiftLeftB32(dst=vgpr(globalAddr), shiftHex=hex(2), src=vgpr(globalAddr),
                                  comment="byteAddr = (token*n_d + WG0) * 4"))
        if strideV is not None:
            module.add(VLShiftLeftB32(dst=vgpr(strideV), shiftHex=hex(2), src=vgpr(strideV),
                                      comment="stride4 = stride * 4"))
        for n in range(self.mma_n):
            module.add(BufferStoreB32(src=vgpr(partials + n), vaddr=vgpr(globalAddr),
                                      saddr=sgpr(partialSrd, 4), soffset=0,
                                      mubuf=MUBUFModifiers(offen=True),
                                      comment=f"partialBuf[token+n*{self.mfma_n}, WG0] = {label} (n={n})"))
            if n < self.mma_n - 1:
                module.add(VAddU32(vgpr(globalAddr), vgpr(globalAddr), vgpr(strideV),
                                   comment=f"byteAddr += stride4 (advance to n={n + 1})"))
        module.add(SWaitCnt(vscnt=0, comment="wait partialBuf stores"))
        module.add(SMovB64(dst=EXEC(), src=sgpr(savedExec, lsc), comment="restore exec mask"))
        if strideV is not None:
            self.writer.vgprPool.checkIn(strideV)
        self.writer.vgprPool.checkIn(accumV)
        self.writer.vgprPool.checkIn(tokenBase)
        self.writer.vgprPool.checkIn(ntilesV)
        module.addComment0("MF end _writePartialsFree0.")
        return module


    def _beginStreamContext(self, module) -> None:
        """Check out shared streaming context and expose it via self._sc* aliases."""
        module.addComment1("MF begin _beginStreamContext: allocate and init shared streaming context registers.")
        lsc = self.laneSgprCount
        invFp8Bits = struct.unpack('<I', struct.pack('<f', 1.0 / _fp8E4m3Max))[0]
        invFp8V = self.writer.vgprPool.checkOut(1, tag="mx_scInvFp8")
        module.add(VMovB32(dst=vgpr(invFp8V), src=hex(invFp8Bits),
                           comment=f"1/fp8_max = 1/{_fp8E4m3Max}."))
        c254V = self.writer.vgprPool.checkOut(1, tag="mx_scC254")
        module.add(VMovB32(dst=vgpr(c254V), src=254, comment="constant 254."))
        zeroMask = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mx_scZeroMask",
                                                         preventOverflow=False)
        absMask = self.writer.vgprPool.checkOut(1, tag="mx_scAbsMask")
        module.add(VMovB32(dst=vgpr(absMask), src=hex(0x7FFFFFFF), comment="abs mask."))
        accTmp = self.writer.vgprPool.checkOut(1, tag="mx_scAccTmp")
        waveM, waveN = self._computeWaveIndices(module)
        totalFree = self.writer.vgprPool.checkOut(1, tag="mx_scTotalFree")
        self._computeTotalQTilesN(module, totalFree)
        totalKBlocks = self.writer.vgprPool.checkOut(1, tag="mx_scTotalKBlks")
        self._computeTotalQTilesM(module, totalKBlocks)
        freeBaseV = self.writer.vgprPool.checkOut(1, tag="mx_scFreeBase")
        self._computeFreeBase(module, freeBaseV, waveN)
        kblkBaseV = self.writer.vgprPool.checkOut(1, tag="mx_scKblkBase")
        self._computeKblkBase(module, kblkBaseV, waveM)
        strideV = self.writer.vgprPool.checkOut(1, tag="mx_scStride")
        self._computeSwizzleStride(module, strideV, totalKBlocks)
        self._scInvFp8V, self._scC254V, self._scZeroMask = invFp8V, c254V, zeroMask
        self._scAbsMask, self._scAccTmp = absMask, accTmp
        self._scWaveM, self._scWaveN = waveM, waveN
        self._scTotalFree, self._scTotalKBlocks = totalFree, totalKBlocks
        self._scFreeBase, self._scKblkBase, self._scStrideV = freeBaseV, kblkBaseV, strideV
        module.addComment1("MF end _beginStreamContext.")


    def _buildSubColGroupMask(self, module, rowGroup: int, kblkV: int) -> int:
        """Group sub-mask rowGroup==0 AND kblkV<totalKBlocks, shared by all g stores.

        Returns a checked-out lane-mask SGPR; caller must checkIn it.
        """
        module.addComment1("MF begin _buildSubColGroupMask: compute rowGroup==0 AND kblk-in-range lane mask.")
        lsc = self.laneSgprCount
        groupMask = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mx_scGroupMask",
                                                         preventOverflow=False)
        rgCond = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mx_scRgCond",
                                                      preventOverflow=False)
        module.add(VCmpEQU32(dst=sgpr(rgCond, lsc), src0=0, src1=vgpr(rowGroup),
                             comment="rowGroup == 0?."))
        kblkCond = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mx_scKblkIR",
                                                        preventOverflow=False)
        module.add(VCmpLtU32(dst=sgpr(kblkCond, lsc), src0=vgpr(kblkV),
                             src1=vgpr(self._scTotalKBlocks), comment="kblkV < totalKBlocks?."))
        module.add(SAndB64(dst=sgpr(groupMask, lsc), src0=sgpr(rgCond, lsc),
                           src1=sgpr(kblkCond, lsc),
                           comment="group sub-mask = rowGroup==0 AND kblk in range."))
        self.writer.sgprPool.checkIn(kblkCond)
        self.writer.sgprPool.checkIn(rgCond)
        module.addComment1("MF end _buildSubColGroupMask.")
        return groupMask


    def _butterflyRound(self, module, addrV: int, tmpV: int,
                         amaxVgprs: int, totalTiles: int, laneId: int,
                         xorVal: int) -> None:
        """Emit one XOR-butterfly round: fetch partner amax and fold via VMaxF32."""
        module.addComment1(f"MF begin _butterflyRound: one XOR-butterfly amax reduction round (xorVal={xorVal}).")
        module.add(VXorB32(dst=vgpr(addrV), src0=vgpr(laneId), src1=xorVal,
                           comment=f"partnerLane = laneId ^ {xorVal}."))
        module.add(VLShiftLeftB32(dst=vgpr(addrV), shiftHex=hex(2), src=vgpr(addrV),
                                  comment="byteAddr = partnerLane * 4."))
        for t in range(totalTiles):
            module.add(DSBPermuteB32(vgpr(tmpV + t), vgpr(addrV), vgpr(amaxVgprs + t),
                                      comment=f"fetch partner amax[tile={t}]."))
        module.add(SWaitCnt(dscnt=0, comment="wait ds_bpermute."))
        for t in range(totalTiles):
            module.add(VMaxF32(dst=vgpr(amaxVgprs + t), src0=vgpr(amaxVgprs + t),
                               src1=vgpr(tmpV + t),
                               comment=f"amax[tile={t}] = max(amax, partner)."))
        module.addComment1("MF end _butterflyRound.")


    def _computeCeilAdj(self, module, scaleFV: int, adjV: int) -> None:
        """Compute ceil adjustment (0 or 1) from scaleFV mantissa into adjV.

        Internally allocates and frees a temporary mantissa VGPR and mask SGPR.
        """
        module.addComment1("MF begin _computeCeilAdj: compute ceiling adjustment from mantissa.")
        # mantV = scaleFV << 9: discards sign and exponent, non-zero iff mantissa != 0.
        mantV = self.writer.vgprPool.checkOut(1, tag="mx_mant")
        module.add(VLShiftLeftB32(dst=vgpr(mantV), shiftHex=hex(9),
                                  src=vgpr(scaleFV),
                                  comment="mantV = scaleF << 9 (mant != 0 iff mantV != 0)."))
        lsc = self.laneSgprCount
        zmc = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mx_zmc", preventOverflow=False)
        module.add(VCmpEQU32(dst=sgpr(zmc, lsc), src0=0, src1=vgpr(mantV),
                             comment="mant == 0?."))
        # When zmc TRUE (mant==0): dst=src1=0; FALSE (mant!=0): dst=src0=1.
        module.add(VCndMaskB32(dst=vgpr(adjV), src0=1, src1=0, src2=sgpr(zmc, lsc),
                               comment="adj = (mant!=0) ? 1 : 0."))
        self.writer.sgprPool.checkIn(zmc)
        self.writer.vgprPool.checkIn(mantV)
        module.addComment1("MF end _computeCeilAdj.")


    def _computeFreeBase(self, module, freeBaseV: int, waveN) -> None:
        """freeBase = WG1*MT1 + waveN*waveSpanN (loop-invariant part of freeV)."""
        module.addComment1("MF begin _computeFreeBase: compute freeBase = WG1*MT1 + waveN*waveSpanN.")
        self._mulVgprBySgprConst(module, freeBaseV, "WorkGroup1", self.macroTile1,
                                  "freeBase = WG1 * MT1.")
        if waveN is None:
            module.addComment1("MF end _computeFreeBase.")
            return
        waveSpanN = self.mmaN * self.mfmaN
        tmp = self.writer.vgprPool.checkOut(1, tag="mx_scFreeBaseTmp")
        self._shiftOrMulVgprConst(module, tmp, waveN, waveSpanN,
                                  f"waveN * waveSpanN={waveSpanN}.")
        module.add(VAddU32(vgpr(freeBaseV), vgpr(freeBaseV), vgpr(tmp),
                           comment="+ waveN * waveSpanN."))
        self.writer.vgprPool.checkIn(tmp)
        module.addComment1("MF end _computeFreeBase.")


    def _computeKblkBase(self, module, kblkBaseV: int, waveM) -> None:
        """kblkBase = WG0*(nQTilesM*wgM) + waveM*nQTilesM (loop-invariant part of kblkV)."""
        module.addComment1("MF begin _computeKblkBase: compute kblkBase = WG0*(nQTilesM*wgM) + waveM*nQTilesM.")
        nQTilesMPerWG = self.nQTilesM * self.wgM
        self._mulVgprBySgprConst(module, kblkBaseV, "WorkGroup0", nQTilesMPerWG,
                                  f"kblkBase = WG0 * {nQTilesMPerWG}.")
        if waveM is None:
            module.addComment1("MF end _computeKblkBase.")
            return
        tmp = self.writer.vgprPool.checkOut(1, tag="mx_scKblkBaseTmp")
        self._shiftOrMulVgprConst(module, tmp, waveM, self.nQTilesM,
                                  f"waveM * nQTilesM={self.nQTilesM}.")
        module.add(VAddU32(vgpr(kblkBaseV), vgpr(kblkBaseV), vgpr(tmp),
                           comment="+ waveM * nQTilesM."))
        self.writer.vgprPool.checkIn(tmp)
        module.addComment1("MF end _computeKblkBase.")


    def _computeOneMXScale(self, module, slot: int, amaxVgpr: int,
                            quantMultVgpr: int, invFp8V: int,
                            c254V: int, zeroMask: int, scaleByteVgpr: int = None) -> None:
        """Emit e8m0 quantMult for slot; quantMultVgpr reused as temp, c254V/zeroMask shared."""
        module.addComment1(f"MF begin _computeOneMXScale: compute e8m0 quantMult for slot {slot}.")
        lsc = self.laneSgprCount
        # scaleF = amax * (1/448) -> into quantMultVgpr (temp for scaleByte).
        module.add(VMulF32(dst=vgpr(quantMultVgpr), src0=vgpr(amaxVgpr),
                           src1=vgpr(invFp8V),
                           comment=f"scaleF[{slot}] = amax * (1/448)."))
        adjV = self.writer.vgprPool.checkOut(1, tag="mx_adj")
        self._computeCeilAdj(module, quantMultVgpr, adjV)
        # expByte = scaleF >> 23; & 0xFF not needed since scaleF >= 0 (sign bit = 0).
        module.add(VLShiftRightB32(dst=vgpr(quantMultVgpr), shiftHex=hex(23),
                                   src=vgpr(quantMultVgpr),
                                   comment="expByte = scaleF >> 23."))
        # scaleByte = expByte + adj -> into quantMultVgpr.
        module.add(VAddU32(vgpr(quantMultVgpr), vgpr(quantMultVgpr), vgpr(adjV),
                           comment=f"scaleByte[{slot}] = expByte + ceilAdj."))
        self.writer.vgprPool.checkIn(adjV)
        # clamp(scaleByte, 0, 254); VMed3I32 requires src2 Container.
        module.add(VMed3I32(dst=vgpr(quantMultVgpr), src0=0,
                            src1=vgpr(quantMultVgpr), src2=vgpr(c254V),
                            comment=f"scaleByte[{slot}] = clamp(scaleByte, 0, 254)."))
        if scaleByteVgpr is not None:
            # the clamped scaleByte is naturally 0 when amax==0, so no zero-guard is needed.
            module.add(VMovB32(dst=vgpr(scaleByteVgpr), src=vgpr(quantMultVgpr),
                               comment=f"bank scaleByte[{slot}] for the store path."))
        # qExpField = 254 - scaleByte -> into quantMultVgpr.
        module.add(VSubU32(vgpr(quantMultVgpr), vgpr(c254V), vgpr(quantMultVgpr),
                           comment=f"qExpField[{slot}] = 254 - scaleByte."))
        # clamp(qExpField, 1, 254).
        module.add(VMed3I32(dst=vgpr(quantMultVgpr), src0=1,
                            src1=vgpr(quantMultVgpr), src2=vgpr(c254V),
                            comment=f"qExpField[{slot}] = clamp(qExpField, 1, 254)."))
        # quantMult = bitcast<float>(qExpField << 23).
        module.add(VLShiftLeftB32(dst=vgpr(quantMultVgpr), shiftHex=hex(23),
                                  src=vgpr(quantMultVgpr),
                                  comment=f"quantMult[{slot}] = qExpField << 23."))
        # amax==0 override: quantMult = 0 when amax==0 (scaleByte is naturally 0).
        module.add(VCmpEQF32(dst=sgpr(zeroMask, lsc), src0=0, src1=vgpr(amaxVgpr),
                             comment=f"amax[{slot}] == 0?."))
        module.add(VCndMaskB32(dst=vgpr(quantMultVgpr), src0=vgpr(quantMultVgpr),
                               src1=0, src2=sgpr(zeroMask, lsc),
                               comment=f"quantMult[{slot}] = 0 if amax==0."))
        module.addComment1("MF end _computeOneMXScale.")


    def _computeSubColScales(self, module, amaxBase: int, qi: int, nBase: int, g: int) -> int:
        """Compute e8m0 quantMult per j, bank scaleByte, and fold alpha into amaxBase (applyMult).

        Returns the scaleByte bank; amaxBase is overwritten with the alpha-folded apply
        multiplier and qmulBase is freed here since only the store needs the banked scaleByte.
        """
        module.addComment1(f"MF begin _computeSubColScales: compute e8m0 scales and alpha-fold for group (qi={qi},nBase={nBase},g={g}).")
        qmulBase = self.writer.vgprPool.checkOut(g, tag=f"mx_scQmul_qi{qi}_n{nBase}")
        scaleByteBank = self.writer.vgprPool.checkOut(g, tag=f"mx_scByteBank_qi{qi}_n{nBase}")
        for j in range(g):
            self._computeOneMXScale(module, j, amaxBase + j, qmulBase + j,
                                    self._scInvFp8V, self._scC254V, self._scZeroMask,
                                    scaleByteVgpr=scaleByteBank + j)
        for j in range(g):
            module.add(VMulF32(dst=vgpr(amaxBase + j), src0=vgpr(qmulBase + j),
                               src1=sgpr("Alpha"),
                               comment=f"applyMult[j={j}] = alpha*quantMult."))
        self.writer.vgprPool.checkIn(qmulBase)
        module.addComment1("MF end _computeSubColScales.")
        return scaleByteBank


    def _computeSwizzleStride(self, module, strideV: int, nTilesV: int) -> None:
        """strideV = ceil(nTiles/8) * 256 (the swizzle d0 stride)."""
        module.addComment1("MF begin _computeSwizzleStride: compute d0 stride = ceil(nTiles/8) * 256.")
        module.add(VAddU32(vgpr(strideV), vgpr(nTilesV), 7, comment="nTiles + 7."))
        module.add(VLShiftRightB32(dst=vgpr(strideV), shiftHex=hex(3), src=vgpr(strideV),
                                   comment="colBlocks = ceil(nTiles/8)."))
        module.add(VLShiftLeftB32(dst=vgpr(strideV), shiftHex=hex(8), src=vgpr(strideV),
                                  comment="d0 stride = colBlocks * 256."))
        module.addComment1("MF end _computeSwizzleStride.")


    def _computeTotalQTilesM(self, module, dst: int) -> None:
        """Compute ceil(nHidden / Q0) into VGPR dst (scale buffer row count for sub-row mode)."""
        module.addComment1("MF begin _computeTotalQTilesM: compute totalQTilesM = ceil(nHidden/Q0).")
        with self.writer.allocTmpSgpr(1, tag=f"{self.tagPrefix}_nQTMsS") as s:
            module.add(SAddU32(dst=sgpr(s.idx), src0=sgpr("SizesFree+0"),
                               src1=self.q0 - 1,
                               comment=f"nHidden + Q0-1 (Q0={self.q0})."))
            log2q0 = int(math.log2(self.q0)) if self.q0 > 1 else 0
            module.add(SLShiftRightB32(dst=sgpr(s.idx), shiftHex=hex(log2q0),
                                       src=sgpr(s.idx),
                                       comment=f"totalQTilesM = ceil(nHidden/Q0={self.q0})."))
            module.add(VMovB32(dst=vgpr(dst), src=sgpr(s.idx),
                               comment="totalQTilesM into VGPR."))
        module.addComment1("MF end _computeTotalQTilesM.")


    def _computeTotalQTilesN(self, module, dst: int) -> None:
        """Compute ceil(N / Q1) into VGPR dst at runtime from SizesFree+1."""
        module.addComment1("MF begin _computeTotalQTilesN: compute totalQTilesN = ceil(N/Q1).")
        with self.writer.allocTmpSgpr(1, tag=f"{self.tagPrefix}_nQTNsS") as s:
            module.add(SAddU32(dst=sgpr(s.idx), src0=sgpr("SizesFree+1"),
                               src1=self.q1 - 1,
                               comment=f"N + Q1-1 (Q1={self.q1})."))
            log2q1 = int(math.log2(self.q1))
            module.add(SLShiftRightB32(dst=sgpr(s.idx), shiftHex=hex(log2q1),
                                       src=sgpr(s.idx),
                                       comment=f"totalQTilesN = ceil(N/Q1={self.q1})."))
            module.add(VMovB32(dst=vgpr(dst), src=sgpr(s.idx),
                               comment="totalQTilesN into VGPR."))
        module.addComment1("MF end _computeTotalQTilesN.")


    def _computeWaveIndices(self, module) -> tuple:
        """Compute waveM = waveIdx % wg_m and waveN = (waveIdx // wg_m) % wg_n."""
        if self.wgM <= 1 and self.wgN <= 1:
            return None, None
        module.addComment1("MF begin _computeWaveIndices: compute waveM and waveN from waveIdx.")
        log2Wave = int(math.log2(self.waveSize))
        waveIdx = self.writer.vgprPool.checkOut(1, tag=f"{self.tagPrefix}_waveIdx")
        module.add(VLShiftRightB32(dst=vgpr(waveIdx), shiftHex=hex(log2Wave),
                                   src=vgpr("Serial"),
                                   comment=f"waveIdx = Serial >> {log2Wave}."))
        waveM = None
        if self.wgM > 1:
            waveM = self.writer.vgprPool.checkOut(1, tag=f"{self.tagPrefix}_waveM")
            module.add(VAndB32(dst=vgpr(waveM), src0=vgpr(waveIdx), src1=self.wgM - 1,
                               comment=f"waveM = waveIdx & ({self.wgM}-1)."))
        waveN = None
        if self.wgN > 1:
            log2WgM = int(math.log2(self.wgM))
            waveN = self.writer.vgprPool.checkOut(1, tag=f"{self.tagPrefix}_waveN")
            module.add(VLShiftRightB32(dst=vgpr(waveN), shiftHex=hex(log2WgM),
                                       src=vgpr(waveIdx),
                                       comment=f"waveIdx >> {log2WgM}."))
            module.add(VAndB32(dst=vgpr(waveN), src0=vgpr(waveN), src1=self.wgN - 1,
                               comment=f"waveN = (waveIdx >> {log2WgM}) & ({self.wgN}-1)."))
        self.writer.vgprPool.checkIn(waveIdx)
        module.addComment1("MF end _computeWaveIndices.")
        return waveM, waveN


    def _endStreamContext(self) -> None:
        """Return the shared streaming-context registers to their pools."""
        self.writer.vgprPool.checkIn(self._scStrideV)
        self.writer.vgprPool.checkIn(self._scKblkBase)
        self.writer.vgprPool.checkIn(self._scFreeBase)
        self.writer.vgprPool.checkIn(self._scTotalKBlocks)
        self.writer.vgprPool.checkIn(self._scTotalFree)
        if self._scWaveN is not None:
            self.writer.vgprPool.checkIn(self._scWaveN)
        if self._scWaveM is not None:
            self.writer.vgprPool.checkIn(self._scWaveM)
        self.writer.vgprPool.checkIn(self._scAccTmp)
        self.writer.vgprPool.checkIn(self._scAbsMask)
        self.writer.sgprPool.checkIn(self._scZeroMask)
        self.writer.vgprPool.checkIn(self._scC254V)
        self.writer.vgprPool.checkIn(self._scInvFp8V)


    def _mulVgprBySgprConst(self, module, dstVgpr: int, sgprName: str,
                             const: int, comment: str) -> None:
        """Emit dstVgpr = sgpr(sgprName) * const via a full-rate shift for pow2, else a literal mul."""
        module.addComment1("MF begin _mulVgprBySgprConst: dstVgpr = sgpr * const (shift or mul).")
        if const > 0 and (const & (const - 1)) == 0:
            module.add(VLShiftLeftB32(dst=vgpr(dstVgpr), shiftHex=hex(int(math.log2(const))),
                                      src=sgpr(sgprName), comment=comment))
            module.addComment1("MF end _mulVgprBySgprConst.")
            return
        # v_mul_lo_u32 is VOP3 and rejects literal operands in any source position.
        # Materialize the constant in a temporary SGPR, then move the input SGPR to
        # the destination VGPR and multiply using the SGPR constant.
        sTmp = self.writer.sgprPool.checkOut(1, tag="mulBySgprConst")
        module.add(SMovB32(dst=sgpr(sTmp), src=const, comment=f"load {const} into SGPR."))
        module.add(VMovB32(dst=vgpr(dstVgpr), src=sgpr(sgprName), comment=comment))
        module.add(VMulLOU32(dst=vgpr(dstVgpr), src0=vgpr(dstVgpr), src1=sgpr(sTmp), comment=comment))
        self.writer.sgprPool.checkIn(sTmp)
        module.addComment1("MF end _mulVgprBySgprConst.")


    def _readAccBurstStaged(self, module, dstBase: int, vgprTiles, coords, comment: str) -> None:
        """Read a burst of accumulator elements into consecutive VGPRs [dstBase, dstBase+len).

        Reads issue back-to-back so the mandatory post-v_accvgpr_read wait state
        (gfx950, MIArchVgpr=False) is hidden by later reads and by the compute that
        consumes earlier entries; a trailing s_nop is only needed for a burst of one.
        """
        module.addComment1("MF begin _readAccBurstStaged: staged read of accumulator burst into consecutive VGPRs.")
        usedAcc = False
        for i, (m, n, k) in enumerate(coords):
            tile = vgprTiles[n * self.mmaM + m]
            reg = tile.regList.indices[k]
            if tile.regList.pool == self.writer.vgprPool:
                module.add(VMovB32(dst=vgpr(dstBase + i), src=vgpr(reg), comment=f"{comment} [{i}]."))
                continue
            module.add(VAccvgprReadB32(vgpr(dstBase + i), accvgpr(reg), comment=f"{comment} [{i}]."))
            usedAcc = True
        if usedAcc and len(coords) < 2:
            module.add(SNop(waitState=1, comment="s_nop after v_accvgpr_read before VALU (gfx950)."))
        module.addComment1("MF end _readAccBurstStaged.")


    def _shiftOrMulVgprConst(self, module, dst: int, srcVgpr: int,
                              const: int, comment: str) -> None:
        """Emit dst = srcVgpr * const via a full-rate shift for pow2, else a literal mul."""
        module.addComment1("MF begin _shiftOrMulVgprConst: dst = srcVgpr * const (shift or mul).")
        if const > 0 and (const & (const - 1)) == 0:
            module.add(VLShiftLeftB32(dst=vgpr(dst), shiftHex=hex(int(math.log2(const))),
                                      src=vgpr(srcVgpr), comment=comment))
            module.addComment1("MF end _shiftOrMulVgprConst.")
            return
        # v_mul_lo_u32 is VOP3 and rejects literal operands in any source position.
        # Materialize the constant in a temporary SGPR and multiply.
        sTmp = self.writer.sgprPool.checkOut(1, tag="mulByVgprConst")
        module.add(SMovB32(dst=sgpr(sTmp), src=const, comment=f"load {const} into SGPR."))
        module.add(VMulLOU32(dst=vgpr(dst), src0=vgpr(srcVgpr), src1=sgpr(sTmp), comment=comment))
        self.writer.sgprPool.checkIn(sTmp)
        module.addComment1("MF end _shiftOrMulVgprConst.")


    def _subColApplyFromAcc(self, module, vgprTiles, applyMult: int, accScratch: int,
                            mStart: int, mEnd: int, nBase: int, g: int) -> None:
        """Re-read each group tile from the accumulator, scale by applyMult[j], write back.

        Used by the fused epilogue's deferred tail: the gamma-scaled value already
        lives in the accumulator, so no per-group staging bank is retained. accScratch
        is a rowsPerLane-sized read buffer, reused per tile column.
        """
        module.addComment1(f"MF begin _subColApplyFromAcc: re-read acc, scale by applyMult, write back (nBase={nBase},g={g}).")
        module.add(SNop(waitState=1, comment="hazard guard: accvgpr_write in element loop -> accvgpr_read here (gfx950)."))
        for j in range(g):
            n = nBase + j
            for m in range(mStart, mEnd):
                coords = [(m, n, k) for k in range(self.rowsPerLane)]
                self._readAccBurstStaged(module, accScratch, vgprTiles, coords,
                                   f"reread acc[m={m},n={n}].")
                for k in range(self.rowsPerLane):
                    module.add(VMulF32(dst=vgpr(accScratch + k), src0=vgpr(accScratch + k),
                                       src1=vgpr(applyMult + j),
                                       comment=f"acc *= alpha*quantMult[j={j}] (m={m},n={n},k={k})."))
                    self._writeAccFromMx(module, accScratch + k, vgprTiles, m, n, k,
                                       f"write acc[m={m},n={n},k={k}].")
        module.addComment1("MF end _subColApplyFromAcc.")


    def _subColFreeV(self, module, n: int, col: int) -> int:
        """Per-lane freeV = freeBase + n*mfmaN + col (freeBase hoisted into context)."""
        module.addComment1(f"MF begin _subColFreeV: compute per-lane freeV = freeBase + n*mfmaN + col (n={n}).")
        freeV = self.writer.vgprPool.checkOut(1, tag="mx_scFreeV")
        module.add(VAddU32(vgpr(freeV), vgpr(self._scFreeBase), vgpr(col),
                           comment="freeBase + col (per-lane)."))
        nOff = n * self.mfmaN
        if 0 < nOff <= 64:
            module.add(VAddU32(vgpr(freeV), vgpr(freeV), nOff, comment=f"+ n*mfmaN={nOff}."))
        elif nOff > 64:
            tmpN = self.writer.vgprPool.checkOut(1, tag="mx_scNOff")
            module.add(VMovB32(dst=vgpr(tmpN), src=nOff, comment=f"n*mfmaN={nOff}."))
            module.add(VAddU32(vgpr(freeV), vgpr(freeV), vgpr(tmpN), comment="+ n*mfmaN."))
            self.writer.vgprPool.checkIn(tmpN)
        module.addComment1("MF end _subColFreeV.")
        return freeV


    def _subColStoreGroup(self, module, mxSrd: int, scaleByteBank: int, col: int, rowGroup: int,
                          savedExec: int, laneMask: int,
                          qi: int, nBase: int, g: int) -> None:
        """Store g e8m0 scale bytes for the group from a precomputed scaleByte bank."""
        module.addComment1(f"MF begin _subColStoreGroup: store g e8m0 scale bytes for group (qi={qi},nBase={nBase},g={g}).")
        lsc = self.laneSgprCount
        kblkV = self.writer.vgprPool.checkOut(1, tag="mx_scKblkV")
        if qi:
            module.add(VAddU32(vgpr(kblkV), vgpr(self._scKblkBase), qi,
                               comment=f"kblkV = kblkBase + qi={qi}."))
        else:
            module.add(VMovB32(dst=vgpr(kblkV), src=vgpr(self._scKblkBase),
                               comment="kblkV = kblkBase."))
        groupMask = self._buildSubColGroupMask(module, rowGroup, kblkV)
        # colV=kblkV is invariant across the group's g stores; compute its swizzle
        # bits once here instead of inside _swizzleTileByteOffset per store.
        colLowV = self.writer.vgprPool.checkOut(1, tag="mx_scColLow")
        colLowTmp = self.writer.vgprPool.checkOut(1, tag="mx_scColLowTmp")
        module.add(VMovB32(dst=vgpr(colLowV), src=0, comment="init colLow=0."))
        self._swizzleColBits(module, kblkV, colLowV, colLowTmp)
        self.writer.vgprPool.checkIn(colLowTmp)
        for j in range(g):
            n = nBase + j
            module.addComment0(f"  SubCol store qi={qi}, n={n}.")
            freeV = self._subColFreeV(module, n, col)
            freeCond = self.writer.sgprPool.checkOutAligned(lsc, lsc, tag="mx_scFreeIR",
                                                            preventOverflow=False)
            module.add(VCmpLtU32(dst=sgpr(freeCond, lsc), src0=vgpr(freeV),
                                 src1=vgpr(self._scTotalFree), comment="freeV < totalFree?."))
            module.add(SAndB64(dst=sgpr(laneMask, lsc), src0=sgpr(groupMask, lsc),
                               src1=sgpr(freeCond, lsc),
                               comment="mask = groupMask AND freeV in range."))
            self.writer.sgprPool.checkIn(freeCond)
            module.add(SAndSaveExecB64(dst=sgpr(savedExec, lsc), src=sgpr(laneMask, lsc),
                                       comment="save exec; set exec = write-lane mask."))
            self._swizzleTileByteOffset(module, freeV, kblkV, self._scTotalKBlocks,
                                        strideV=self._scStrideV, colLowV=colLowV)
            module.add(BufferStoreB8(
                src=vgpr(scaleByteBank + j), vaddr=vgpr(freeV),
                saddr=sgpr(mxSrd, 4), soffset=0,
                mubuf=MUBUFModifiers(offen=True),
                comment=f"MXScale[freeV, kblkV] byte (qi={qi}, n={n})."))
            module.add(SMovB64(dst=EXEC(), src=sgpr(savedExec, lsc),
                               comment="restore exec mask."))
            self.writer.vgprPool.checkIn(freeV)
        self.writer.vgprPool.checkIn(colLowV)
        self.writer.sgprPool.checkIn(groupMask)
        self.writer.vgprPool.checkIn(kblkV)
        module.addComment1("MF end _subColStoreGroup.")


    def _swizzleColBits(self, module, colV: int, lowV: int, tmpV: int) -> None:
        """OR d5<<6 | d4<<1 | d3<<8 into lowV using colV; clobbers tmpV."""
        module.addComment1("MF begin _swizzleColBits: OR swizzle column bits d3/d4/d5 into lowV.")
        module.add(VAndB32(dst=vgpr(tmpV), src0=vgpr(colV), src1=3, comment="d5 = col & 3."))
        module.add(VLShiftLeftB32(dst=vgpr(tmpV), shiftHex=hex(6), src=vgpr(tmpV),
                                  comment="d5 << 6."))
        module.add(VOrB32(dst=vgpr(lowV), src0=vgpr(lowV), src1=vgpr(tmpV), comment="lowV |= d5<<6."))
        module.add(VLShiftRightB32(dst=vgpr(tmpV), shiftHex=hex(2), src=vgpr(colV),
                                   comment="col >> 2."))
        module.add(VAndB32(dst=vgpr(tmpV), src0=vgpr(tmpV), src1=1, comment="d4 = (col>>2)&1."))
        module.add(VLShiftLeftB32(dst=vgpr(tmpV), shiftHex=hex(1), src=vgpr(tmpV),
                                  comment="d4 << 1."))
        module.add(VOrB32(dst=vgpr(lowV), src0=vgpr(lowV), src1=vgpr(tmpV), comment="lowV |= d4<<1."))
        module.add(VLShiftRightB32(dst=vgpr(tmpV), shiftHex=hex(3), src=vgpr(colV),
                                   comment="d3 = col >> 3."))
        module.add(VLShiftLeftB32(dst=vgpr(tmpV), shiftHex=hex(8), src=vgpr(tmpV),
                                  comment="d3 << 8."))
        module.add(VOrB32(dst=vgpr(lowV), src0=vgpr(lowV), src1=vgpr(tmpV), comment="lowV |= d3<<8."))
        module.addComment1("MF end _swizzleColBits.")


    def _swizzleRowBits(self, module, rowV: int, lowV: int, tmpV: int) -> None:
        """Write d2<<2 | d1 into lowV using rowV; clobbers tmpV. rowV is not modified."""
        module.addComment1("MF begin _swizzleRowBits: write swizzle row bits d1/d2 into lowV.")
        module.add(VAndB32(dst=vgpr(lowV), src0=vgpr(rowV), src1=0xF, comment="d2 = row & 0xF."))
        module.add(VLShiftLeftB32(dst=vgpr(lowV), shiftHex=hex(2), src=vgpr(lowV),
                                  comment="d2 << 2."))
        module.add(VLShiftRightB32(dst=vgpr(tmpV), shiftHex=hex(4), src=vgpr(rowV),
                                   comment="row >> 4."))
        module.add(VAndB32(dst=vgpr(tmpV), src0=vgpr(tmpV), src1=1, comment="d1 = (row>>4)&1."))
        module.add(VOrB32(dst=vgpr(lowV), src0=vgpr(lowV), src1=vgpr(tmpV), comment="lowV |= d1."))
        module.addComment1("MF end _swizzleRowBits.")


    def _swizzleTileByteOffset(self, module, addrV: int, colV: int, nColTiles: int,
                                strideV: int = None, colLowV: int = None) -> None:
        """Overwrite addrV with the GFX950 pre-swizzled MXScale byte offset.

        addrV holds the free/token tile index (swizzle row bits d0/d1/d2) and colV holds
        the kblock tile index (swizzle col bits d3/d4/d5). When strideV is given it is a
        persistent register holding the loop-invariant d0 stride and must not be clobbered.
        byteOff = d0*(colBlocks*256) + d3*256 + d5*64 + d2*4 + d4*2 + d1.
        When colLowV is given it holds the precomputed col-swizzle bits (invariant
        across a store group) and colV is ignored.
        """
        module.addComment1("MF begin _swizzleTileByteOffset: compute GFX950 pre-swizzled MXScale byte offset.")
        sLow = self.writer.vgprPool.checkOut(1, tag="mx_swzLow")
        sTmp = self.writer.vgprPool.checkOut(1, tag="mx_swzTmp")
        ownStride = strideV is None
        if ownStride:
            strideV = self.writer.vgprPool.checkOut(1, tag="mx_swzStride")
            self._computeSwizzleStride(module, strideV, nColTiles)
        # d0*stride survives the swizzle bit-mixing, so it needs a register the mixing does not touch.
        d0Prod = strideV if ownStride else self.writer.vgprPool.checkOut(1, tag="mx_swzD0")
        module.add(VLShiftRightB32(dst=vgpr(sTmp), shiftHex=hex(5), src=vgpr(addrV),
                                   comment="d0 = qTileRow >> 5."))
        module.add(VMulLOU32(dst=vgpr(d0Prod), src0=vgpr(sTmp), src1=vgpr(strideV),
                             comment="d0 * stride."))
        self._swizzleRowBits(module, addrV, sLow, sTmp)
        if colLowV is None:
            self._swizzleColBits(module, colV, sLow, sTmp)
        else:
            module.add(VOrB32(dst=vgpr(sLow), src0=vgpr(sLow), src1=vgpr(colLowV),
                              comment="lowV |= precomputed col bits (hoisted)."))
        module.add(VAddU32(vgpr(addrV), vgpr(d0Prod), vgpr(sLow), comment="swizzled byteOff."))
        if ownStride:
            self.writer.vgprPool.checkIn(strideV)
        else:
            self.writer.vgprPool.checkIn(d0Prod)
        self.writer.vgprPool.checkIn(sTmp)
        self.writer.vgprPool.checkIn(sLow)
        module.addComment1("MF end _swizzleTileByteOffset.")


    def _writeAccFromMx(self, module, src: int, vgprTiles, m: int, n: int, k: int,
                      comment: str) -> None:
        """Write VGPR src back into accumulator element (m, n, k)."""
        module.addComment1("MF begin _writeAccFromMx: write VGPR back to MX accumulator register file.")
        tile = vgprTiles[n * self.mmaM + m]
        reg  = tile.regList.indices[k]
        if tile.regList.pool == self.writer.vgprPool:
            module.add(VMovB32(dst=vgpr(reg), src=vgpr(src), comment=comment))
            module.addComment1("MF end _writeAccFromMx.")
            return
        module.add(VAccvgprWriteB32(accvgpr(reg), vgpr(src), comment=comment))
        module.addComment1("MF end _writeAccFromMx.")

