# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""PartialRMS fused epilogue emitter for the Subtile kernel (gfx950, bf16/f16/fp8/bfp8 A/B input).

Reduces the fp32 GEMM accumulator (AGPRs) over free0 (the N_hidden dimension),
producing one fp32 Σx² per output token (free1 row) per free0 tile (WorkGroup0).
Also applies the gamma weight to the accumulator in-place.

H is read from the accumulator registers as written by the preceding ResidualAdd
epilogue; this emitter does no residual loading, residual adding, or bf16 storing.
Because it holds only the gamma burst and partials in scratch VGPRs (released after
the pass), the VGPR peak is lower than the old fused single-pass version.

This is Phase 1 (K1) of a two-kernel RMSNorm pipeline operating on row-major output:
  - K1 (this kernel): free0=N_hidden, free1=M_tokens. Each WG writes
    partialBuf[token, WorkGroup0] = Σ_{i in tile} h1[token, i]².
  - K2 (row_div): reads all partialBuf tiles per token, reduces, computes
    rstd = rsqrt(Σx²/N_hidden + eps), and divides D in-place.

partialBuf layout contract (2D, row-major):
  - Logical shape [M_padded, n_d] for Σx² (n_d = ceil(SizesFree0 / MT0)).
    When PartialRMSQuant is set, a second region of the same shape is
    stacked below it, so the real buffer is [2*M_padded, n_d]: rows
    [0, M_padded) hold Σx² and rows [M_padded, 2*M_padded) hold
    amax(|D|)/fp8_max, written by _amaxEpilogueFree0 with rowOffset =
    M_padded.
  - partialBuf[token, t] = Σ_{i in WG t's free0 columns} h1[token, i]².
  - Byte offset for (token, t) = (token * n_d + t) * 4.
  - Token index = WorkGroup1*MT1 + intra-tile token offset.
  - Tile column t = WorkGroup0 (the WG's index along free0).
  - n_d is computed on device from SizesFree0; it is not a kernarg.

Reduction stages (free0 axis):
  1. Sum acc²  over all free0 MMA-tiles (mma_m) and k-offsets (rows_per_lane)
     within each wave, yielding one partial[n] per free1 lane-column (mfma_n lanes).
  2. XOR butterfly via ds_bpermute over waveSize/mfma_n row groups so every
     lane holds the full free0 sum for its n column.
  3. (When wg_m > 1) LDS cross-wave reduction over wg_m sibling waves.
  4. Lanes with rowGroup==0 and waveM==0 write partialBuf[token, WG0].

Gamma application:
  Loads gamma for each free0 position (1 byte for fp8/bfp8, 2 bytes for bf16/f16),
  converts to fp32, and multiplies each accumulator element in-place.

MFMA layout (gfx950, waveSize=64, 16x16 MFMA):
  - lane % mfma_n = free1 column within MMA tile (token lane)
  - rows_per_lane = (mfma_m * mfma_n) // waveSize

Acc VGPR ordering (N-outer, M-inner):
  acc_idx(base, m, n, k) = base + (n*mma_m + m)*rows_per_lane + k

Alpha=1, beta=0 must be passed by the host.
"""

import math
import struct

from rocisa.code import Module
from rocisa.container import (
    ContinuousRegister,
    DSModifiers,
    EXEC,
    MUBUFModifiers,
    accvgpr,
    sgpr,
    vgpr,
)
from rocisa.functions import vectorStaticDivide
from rocisa.instruction import (
    BufferLoadB32,
    BufferLoadD16B16,
    BufferLoadD16U8,
    BufferStoreB32,
    DSBPermuteB32,
    DSLoadB32,
    DSStoreB32,
    SAddU32,
    SAndSaveExecB64,
    SLShiftRightB32,
    SMovB32,
    SMovB64,
    SMulHIU32,
    SMulI32,
    SNop,
    SWaitCnt,
    VAccvgprReadB32,
    VAccvgprWriteB32,
    VAddF32,
    VAddU32,
    VAndB32,
    VCmpEQU32,
    VCvtBF16toFP32,
    VCvtBF8toF32,
    VCvtF16toF32,
    VCvtFP8toF32,
    VFmaF32,
    VLShiftLeftB32,
    VLShiftRightB32,
    VMaxF32,
    VMulF32,
    VMulLOU32,
    VMovB32,
    VOrB32,
    VXorB32,
)
from Tensile.Common.DataType import DataType


# Maximum inline-literal integer for VOP encodings; larger immediates must be
# materialized in a VGPR before a v_add.
_INLINE_CONST_MAX = 64

# Maximum representable magnitudes of the FP8 quantization output types, used
# to scale amax so K2 can requantize. e5m2/bf8 has a far larger range than e4m3.
_FP8_E4M3_MAX = 448.0        # OCP FP8 e4m3.
_FP8_E4M3_FNUZ_MAX = 240.0   # FNUZ FP8 e4m3.
_BF8_E5M2_MAX = 57344.0      # FP8 e5m2 (bf8).


# Returns (magic, postShift) for unsigned floor-division by constant d via SMulHIU32.
# floor(x/d) = mulhi(x, magic) >> postShift; for ceil-div pre-add (d-1). Valid for d >= 2.
def _ceilDivMagic(d: int):
    p = (d - 1).bit_length()          # smallest p such that 2^p >= d.
    magic = -(-( 1 << (32 + p - 1)) // d)  # ceil(2^(32+p-1) / d), using integer ceiling.
    return magic & 0xFFFFFFFF, p - 1  # postShift = p-1 (mulhi already shifts by 32).


class SubtilePartialRMSEmitter:
    """Emit the PartialRMS epilogue for the Subtile gfx950 kernel.

    Reads H directly from accumulators written by the preceding ResidualAdd
    epilogue. Computes per-row Σx² from fp32 AGPRs, writes to partialBuf (fp32),
    and applies gamma in-place to the accumulator. Gamma element type is
    configurable via PartialRMSGammaType.
    """

    def __init__(self, writer, kernel):
        self.writer = writer
        self.kernel = kernel
        self.archCaps = writer.states.archCaps

        # Derive all geometry from kernel params; no module-level constants.
        self.mfma_m = kernel["MatrixInstM"]
        self.mfma_n = kernel["MatrixInstN"]
        self.waveSize = kernel["WavefrontSize"]
        # This epilogue uses 64-bit EXEC ops (SAndSaveExecB64 / SMovB64(EXEC()))
        # and a 2-dword lane mask, so it only supports wave64.
        assert self.waveSize == 64, "partialRMS epilogue requires wavefrontSize == 64"
        self.rows_per_lane = (self.mfma_m * self.mfma_n) // self.waveSize

        wg = kernel["MIWaveGroup"]
        self.wg_m = wg[0]
        self.wg_n = wg[1]

        self.mma_m = (kernel["MacroTile0"] // self.mfma_m) // self.wg_m
        self.mma_n = (kernel["MacroTile1"] // self.mfma_n) // self.wg_n
        self.macro_tile0 = kernel["MacroTile0"]
        self.macro_tile1 = kernel["MacroTile1"]
        self.numPartials = self.mma_n

        # laneSGPRCount: 1 for wave32, 2 for wave64.
        self.lane_sgpr_count = writer.states.laneSGPRCount
        self.quant = bool(kernel.get("PartialRMSQuant", False))

        dt = kernel["ProblemType"]["DataType"]
        # Quant output type selects the fp8 max used to scale amax (see _quantOutMax).
        self.destType = kernel["ProblemType"]["DestDataType"]
        # elemBytes/log2ElemBytes are the GEMM input element size, used only as a
        # reversible scale for the token-index encoding in colByte (encoded in
        # _setup, decoded in _writePartialsFree0); gamma carries its own byte sizes.
        self.elemBytes   = 1 if (dt.isAnyFloat8() or dt.isAnyBFloat8()) else 2
        self.log2ElemBytes = 0 if self.elemBytes == 1 else 1
        # Side-input types resolve to a concrete char before codegen ('b' bf16
        # default, 's' f32).
        self.gammaType    = DataType(kernel.get("PartialRMSGammaType") or "b")
        self.gammaBytes,    self.gammaLog2Bytes    = self._sideBytes(self.gammaType)

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

    def _quantOutMax(self) -> float:
        """Return the max magnitude of the quant output type used to scale amax.

        The two-kernel RMSNorm pipeline quantizes to OCP FP8 e4m3 (448.0) by
        default; a fnuz e4m3 or bf8/e5m2 dest selects its own range. Mirrors the
        DestDataType-driven selection in GlobalWriteBatch.py so the amax scale
        matches whatever K2 requantizes to.
        """
        if self.destType.isFloat8_fnuz():
            return _FP8_E4M3_FNUZ_MAX
        if self.destType.isAnyBFloat8():
            return _BF8_E5M2_MAX
        return _FP8_E4M3_MAX

    def _issueSideLoad(self, module, dstVgpr: int, addrVgpr: int, srd: int,
                       comment: str, dtype) -> None:
        """Issue one gamma buffer_load without waiting (burst-friendly)."""
        loadCls = self._sideLoadClass(dtype)
        module.add(loadCls(vgpr(dstVgpr), vgpr(addrVgpr), sgpr(srd, 4), 0,
                           MUBUFModifiers(offen=True), comment=comment))

    def _convertSideElem(self, module, dstVgpr: int, comment: str, dtype) -> None:
        """Convert an already-loaded side element to fp32 in place (no-op for f32)."""
        if dtype.isSingle():
            return
        if dtype.isHalf():
            module.add(VCvtF16toF32(vgpr(dstVgpr), vgpr(dstVgpr), comment=comment))
        elif dtype.isAnyFloat8():
            module.add(VCvtFP8toF32(dst=vgpr(dstVgpr), src=vgpr(dstVgpr), comment=comment))
        elif dtype.isAnyBFloat8():
            module.add(VCvtBF8toF32(dst=vgpr(dstVgpr), src=vgpr(dstVgpr), comment=comment))
        else:
            module.add(VCvtBF16toFP32(vgpr(dstVgpr), vgpr(dstVgpr), None, 0, comment=comment))

    def _readAccBurst(self, module, dstBase: int, vgprTiles, coords, comment: str):
        """Read accumulator elements, staging AGPR tiles and returning VGPR tiles in place.

        coords is a list of (m, n, k) tile coordinates. AGPR elements are read into
        dstBase+0, dstBase+1, ... (packed); VGPR-resident elements are returned as
        their own register indices without emitting any copy. The returned srcRegs list
        has one entry per coord, in order.
        """
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
        return srcRegs

    def _writeAccFrom(self, module, src: int, vgprTiles, m: int, n: int, k: int, comment: str) -> None:
        """Write VGPR src back into accumulator element (m, n, k), selecting the right register file."""
        tile = vgprTiles[n * self.mma_m + m]
        reg = tile.regList.indices[k]
        if tile.regList.pool == self.writer.vgprPool:
            if src != reg:
                module.add(VMovB32(dst=vgpr(reg), src=vgpr(src), comment=comment))
            return
        module.add(VAccvgprWriteB32(accvgpr(reg), vgpr(src), comment=comment))

    def _buildBufferSrd(self, module, srd: int, ptrName: str, name: str) -> None:
        module.add(SMovB64(dst=sgpr(srd, 2), src=sgpr(ptrName, 2), comment=f"{name} SRD base"))
        module.add(SMovB32(dst=sgpr(srd + 2), src="BufferOOB", comment=f"{name} SRD limit"))
        module.add(SMovB32(dst=sgpr(srd + 3), src="Srd127_96", comment=f"{name} SRD flags"))

    def _addImmU32(self, module, dst: int, src: int, imm: int, scratch: int, comment: str) -> None:
        # imm == 0 is a no-op add; emit a copy only when the value must move registers.
        if imm == 0:
            if dst != src:
                module.add(VMovB32(dst=vgpr(dst), src=vgpr(src), comment=comment))
            return
        # Materialize the immediate in a VGPR when it exceeds the inline-literal range.
        if imm > _INLINE_CONST_MAX:
            module.add(VMovB32(dst=vgpr(scratch), src=imm, comment=f"imm={imm}"))
            module.add(VAddU32(vgpr(dst), vgpr(src), vgpr(scratch), comment=comment))
            return
        module.add(VAddU32(vgpr(dst), vgpr(src), imm, comment=comment))

    def _computeWaveM(self, module, dst: int) -> None:
        # waveId is cached once in _setup (self.waveIdV); only callers with wg_m > 1
        # reach here, so self.waveIdV is always valid.
        module.add(VAndB32(dst=vgpr(dst), src0=vgpr(self.waveIdV), src1=self.wg_m - 1,
                           comment=f"waveM = waveId % {self.wg_m}"))

    def _computeRowGroupOff(self, module, dst: int) -> None:
        # Reuse the cached self.laneId instead of recomputing Serial & (waveSize-1).
        log2MfmaN = int(math.log2(self.mfma_n))
        module.add(VLShiftRightB32(dst=vgpr(dst), shiftHex=hex(log2MfmaN), src=vgpr(self.laneId),
                                   comment=f"rowGroup = laneId >> {log2MfmaN}"))
        module.add(VMulLOU32(dst=vgpr(dst), src0=self.rows_per_lane, src1=vgpr(dst),
                             comment=f"rowGroupOff = rowGroup * {self.rows_per_lane}"))

    def _computeFree0RowBase(self, module, dst: int) -> None:
        # rowBase = WorkGroup0 * MT0 (+ waveM * mma_m*mfma_m when wg_m > 1).
        mt0Vgpr = self.writer.vgprPool.checkOut(1, tag="pRMS_rbMT0")
        module.add(VMovB32(dst=vgpr(mt0Vgpr), src=self.macro_tile0, comment=f"MT0={self.macro_tile0}"))
        module.add(VMulLOU32(dst=vgpr(dst), src0=vgpr(mt0Vgpr), src1=sgpr("WorkGroup0"),
                             comment="rowBase = WorkGroup0 * MT0"))
        self.writer.vgprPool.checkIn(mt0Vgpr)
        if self.wg_m <= 1:
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

    def _free0RowPos(self, module, dst: int, rowBase: int, rowGroupOff: int,
                     m: int, k: int, scratch: int) -> None:
        # free0 row = rowBase + rowGroupOff + (m*mfma_m + k).
        mBase = m * self.mfma_m + k
        self._addImmU32(module, dst, rowBase, mBase, scratch, f"row = base + {mBase} (m={m},k={k})")
        module.add(VAddU32(vgpr(dst), vgpr(dst), vgpr(rowGroupOff), comment="row += rowGroupOff"))

    def emit(self, vgprTiles) -> Module:
        # vgprTiles is dtileInfo.vgprTiles, the per-tile allocator records for the D accumulator.
        numAccVgpr = self.mma_m * self.mma_n * self.rows_per_lane
        accVgprBase = vgprTiles[0].regList.indices[0] if vgprTiles else 0
        module = Module("PartialRMS epilogue")
        module.addComment1("PartialRMS: fused partial sum-of-squares + gamma epilogue")
        module.addComment0(
            f"  Acc AGPRs [{accVgprBase}, {accVgprBase + numAccVgpr}), "
            f"mma_m={self.mma_m}, mma_n={self.mma_n}, "
            f"MT0={self.macro_tile0}, MT1={self.macro_tile1}"
        )
        module.addComment0(
            "  partialBuf: raw Σx² per row (K2 divides by N_hidden, not this kernel)"
        )

        self._allocEpilogueRegs()

        # gfx950 has no MFMA waitcnt and the MFMA->v_accvgpr_read RAW is hardware
        # interlocked, so accumulators need no explicit wait here. _setup begins
        # with an lgkmcnt (kmcnt=0) wait that drains the scalar/LDS tail, so only
        # the vector-memory tail must be drained now to make VGPR reuse safe.
        # ResidualAdd ran first so H is already in the accumulator registers.
        module.add(SWaitCnt(vlcnt=0, comment="drain GEMM vector-memory before VGPR reuse."))

        module.add(self._setup(self.gammaSrd, self.partialSrd, self.laneId, self.colByte))
        absMask = None
        if self.quant:
            absMask = self.writer.vgprPool.checkOut(1, tag="pRMS_absMask")
            module.add(VMovB32(dst=vgpr(absMask), src=hex(0x7FFFFFFF),
                               comment="abs mask = 0x7FFFFFFF"))
        module.add(self._fusedAccPassFree0(
            vgprTiles, self.gammaSrd, self.partials,
            self.amaxPartials if self.quant else None, absMask, self.globalAddr))
        if absMask is not None:
            self.writer.vgprPool.checkIn(absMask)

        # Σx² (and amax when quant) row-group + cross-wave reduction; see _reduceFree0.
        module.add(self._reduceFree0())
        module.add(
            self._writePartialsFree0(
                self.partials, self.partialSrd, self.laneId, self.savedExec, self.laneMaskSgpr,
                self.globalAddr, self.colByte
            )
        )
        if self.quant:
            module.add(self._amaxEpilogueFree0(
                self.amaxPartials, self.partialSrd, self.laneId,
                self.savedExec, self.laneMaskSgpr, self.globalAddr, self.colByte,
                self.mPaddedV, self.amaxScaleV))

        self._freeEpilogueRegs()

        return module

    def _allocEpilogueRegs(self) -> None:
        # Allocate all VGPRs for temporaries.
        self.partials = self.writer.vgprPool.checkOut(self.numPartials, tag="pRMS_partials")
        self.laneId = self.writer.vgprPool.checkOut(1, tag="pRMS_laneId")
        # waveId = Serial / WavefrontSize is computed once in _setup and reused by
        # _computeWaveM / _buildWriteMask / _crossWaveComputeAddrs (wg_m > 1 only).
        self.waveIdV = self.writer.vgprPool.checkOut(1, tag="pRMS_waveId") if self.wg_m > 1 else None
        # colByte is computed and consumed outside the EXEC-narrowed window in _writePartials.
        self.colByte = self.writer.vgprPool.checkOut(1, tag="pRMS_colByte")
        self.globalAddr = self.writer.vgprPool.checkOut(1, tag="pRMS_globalAddr")
        if self.quant:
            self.amaxPartials = self.writer.vgprPool.checkOut(self.numPartials, tag="pRMS_amaxPartials")
            self.mPaddedV     = self.writer.vgprPool.checkOut(1, tag="pRMS_mPaddedV")
            self.amaxScaleV   = self.writer.vgprPool.checkOut(1, tag="pRMS_amaxScale")

        # Allocate SGPRs: gamma SRD, partialBuf SRD, saved exec.
        # savedExec and laneMaskSgpr must be 2-aligned for 64-bit EXEC operations.
        # rowBase = WorkGroup0 * MT0 is computed into globalAddr on demand (no SGPR).
        # tileCol = sgpr("WorkGroup1"), live named SGPR, no allocation needed.
        # preventOverflow=False: the epilogue SGPRs are temporary (live only during
        # the epilogue) and hardware SGPR budget has been verified to accommodate them.
        self.gammaSrd = self.writer.sgprPool.checkOutAligned(4, 4, tag="pRMS_gammaSrd",
                                                             preventOverflow=False)
        self.partialSrd = self.writer.sgprPool.checkOutAligned(4, 4, tag="pRMS_partialSrd",
                                                               preventOverflow=False)
        self.savedExec = self.writer.sgprPool.checkOutAligned(
            self.lane_sgpr_count, self.lane_sgpr_count, tag="pRMS_savedExec",
            preventOverflow=False,
        )
        self.laneMaskSgpr = self.writer.sgprPool.checkOutAligned(
            self.lane_sgpr_count, self.lane_sgpr_count, tag="pRMS_laneMask",
            preventOverflow=False,
        )

    def _freeEpilogueRegs(self) -> None:
        self.writer.sgprPool.checkIn(self.laneMaskSgpr)
        self.writer.sgprPool.checkIn(self.savedExec)
        self.writer.sgprPool.checkIn(self.partialSrd)
        self.writer.sgprPool.checkIn(self.gammaSrd)
        self.writer.vgprPool.checkIn(self.globalAddr)
        self.writer.vgprPool.checkIn(self.colByte)
        self.writer.vgprPool.checkIn(self.laneId)
        if self.wg_m > 1:
            self.writer.vgprPool.checkIn(self.waveIdV)
        self.writer.vgprPool.checkIn(self.partials)
        if self.quant:
            self.writer.vgprPool.checkIn(self.amaxScaleV)
            self.writer.vgprPool.checkIn(self.mPaddedV)
            self.writer.vgprPool.checkIn(self.amaxPartials)

    def _setup(self, gammaSrd: int, partialSrd: int, laneId: int, colByte: int) -> Module:
        module = Module("PartialRMS setup")
        # Both passes must drain kernarg s_loads before reading kernel arguments.
        module.add(SWaitCnt(kmcnt=0, comment="wait for PartialRMS kernarg s_load"))
        self._buildBufferSrd(module, gammaSrd, "RMSNormGamma", "gamma")
        self._buildBufferSrd(module, partialSrd, "PartialBuf", "partialBuf")
        module.add(VAndB32(dst=vgpr(laneId), src0=vgpr("Serial"), src1=self.waveSize - 1,
                           comment="laneId = Serial & (waveSize-1)"))
        if self.wg_m > 1:
            waveIdTmp = self.writer.vgprPool.checkOutAligned(2, 2, tag="pRMS_setupWaveIdDiv")
            waveIdRes = ContinuousRegister(waveIdTmp, 2)
            module.add(vectorStaticDivide(self.waveIdV, "Serial", self.waveSize, waveIdRes,
                                          comment="waveId = Serial / WavefrontSize (cached once)"))
            self.writer.vgprPool.checkIn(waveIdTmp)
        module.add(VAndB32(dst=vgpr(colByte), src0=vgpr(laneId), src1=self.mfma_n - 1,
                           comment=f"colInMma = laneId % {self.mfma_n}"))
        module.add(VLShiftLeftB32(dst=vgpr(colByte), shiftHex=hex(self.log2ElemBytes), src=vgpr(colByte),
                                  comment="colByte = colInMma * elemBytes."))
        self._addWaveNColByte(module, colByte)
        with self.writer.allocTmpSgpr(1, tag="pRMS_setupWG1") as wg1S:
            module.add(SMulI32(dst=sgpr(wg1S.idx), src0=sgpr("WorkGroup1"),
                               src1=self.macro_tile1 * self.elemBytes,
                               comment=f"wg1ColByte = WorkGroup1 * MT1*elemBytes (MT1={self.macro_tile1})"))
            module.add(VAddU32(vgpr(colByte), vgpr(colByte), sgpr(wg1S.idx),
                               comment="colByte += WorkGroup1 * MT1 * elemBytes"))
        return module

    def _addWaveNColByte(self, module, colByte: int) -> None:
        if self.wg_n <= 1:
            return
        waveN = self.writer.vgprPool.checkOut(1, tag="pRMS_setupWaveN")
        tmpVgpr = self.writer.vgprPool.checkOutAligned(2, 2, tag="pRMS_setupTmp")
        tmpRes = ContinuousRegister(tmpVgpr, 2)
        module.add(vectorStaticDivide(waveN, "Serial", self.waveSize * self.wg_m, tmpRes,
                                      comment=f"waveN = Serial / {self.waveSize * self.wg_m}"))
        colBaseBytes = self.mma_n * self.mfma_n * self.elemBytes
        with self.writer.allocTmpSgpr(1, tag="pRMS_setupColBase") as tmpSgprInfo:
            module.add(SMovB32(dst=sgpr(tmpSgprInfo.idx), src=hex(colBaseBytes),
                               comment=f"col base bytes per wave ({colBaseBytes})"))
            module.add(VMulLOU32(dst=vgpr(waveN), src0=sgpr(tmpSgprInfo.idx), src1=vgpr(waveN),
                                 comment="waveN * mma_n * mfma_n * elemBytes"))
        module.add(VAddU32(vgpr(colByte), vgpr(colByte), vgpr(waveN),
                           comment="colByte += wave column base"))
        self.writer.vgprPool.checkIn(tmpVgpr)
        self.writer.vgprPool.checkIn(waveN)

    def _reduceFree0(self) -> Module:
        # Row-group butterfly per array, then one fused cross-wave pass so the amax
        # and Σx² reductions share barriers instead of paying them per array.
        module = Module("PartialRMS reduceFree0")
        # TODO(perf): fuse the Σx² and amax row-group butterflies to share the
        # partner-address computation and dscnt wait. Deferred for simplicity.
        module.add(self._rowGroupReduceFree0(self.partials))
        if self.quant:
            module.add(self._rowGroupReduceFree0(self.amaxPartials, op=VMaxF32, verb="max"))
        if self.wg_m <= 1:
            return module
        reduceArrays = [(self.partials, VAddF32, "+")]
        if self.quant:
            reduceArrays.append((self.amaxPartials, VMaxF32, "max"))
        module.add(self._crossWaveReduceFree0(reduceArrays))
        return module

    def _rowGroupReduceFree0(self, partials: int, op=VAddF32, verb="+") -> Module:
        # Step 2 (free0): all-reduce partial[n] across row groups via ds_bpermute XOR butterfly.
        numRounds = int(math.log2(self.waveSize // self.mfma_n))
        module = Module("PartialRMS rowGroupReduceFree0")
        module.addComment1(
            f"PartialRMS step 2 (free0): XOR butterfly over {self.waveSize // self.mfma_n} row groups"
        )
        if numRounds == 0:
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
        return module

    def _crossWaveComputeAddrs(self, module, writeAddr: int, readAddr: int,
                               numArrays: int = 1) -> None:
        # Only reached when wg_m > 1, so self.waveIdV is valid. Reuse the cached
        # waveId and laneId instead of recomputing them here.
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

    def _crossWaveReduceFree0(self, arrays) -> Module:
        # Step 3 (free0): reduce every array in `arrays` across wg_m sibling waves
        # in a single LDS pass so the three barriers are shared, not paid per array.
        # arrays: list of (baseVgpr, op, verb); array a occupies lane-slot dwords
        # [a*numPartials, (a+1)*numPartials).
        numArrays = len(arrays)
        laneSlotBytes = numArrays * self.numPartials * 4
        strideW = self.waveSize * laneSlotBytes
        module = Module("PartialRMS crossWaveReduceFree0")
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
        # Non-quant epilogue has no further LDS use after this point, so the
        # post-read WAR barrier is unnecessary; the next persistent-loop tile's
        # main loop re-barriers before its own first LDS write.  The quant path
        # retains it defensively because the amax epilogue could acquire LDS use.
        if self.quant:
            module.add(self.writer._syncThreads(self.kernel, "partialRMS free0 cross-wave done."))
        self.writer.vgprPool.checkIn(readTmp)
        self.writer.vgprPool.checkIn(readAddr)
        self.writer.vgprPool.checkIn(writeAddr)
        return module

    def _crossWaveStore(self, module, writeAddr: int, arrays) -> None:
        for a, (base, _op, _verb) in enumerate(arrays):
            for i in range(self.numPartials):
                off = (a * self.numPartials + i) * 4
                module.add(DSStoreB32(dstAddr=vgpr(writeAddr), src=vgpr(base + i),
                                      ds=DSModifiers(offset=off),
                                      comment=f"LDS store arr[{a}] partial[{i}]."))

    def _crossWaveLoadReduce(self, module, readAddr: int, readTmp: int, arrays, strideW: int) -> None:
        # TODO(perf): prefetch wave[j+1]'s LDS loads while accumulating wave[j] to
        # overlap load and compute. Deferred: needs a second readTmp buffer.
        numArrays = len(arrays)
        for j in range(self.wg_m):
            for a in range(numArrays):
                for i in range(self.numPartials):
                    off = (a * self.numPartials + i) * 4
                    module.add(DSLoadB32(dst=vgpr(readTmp + a * self.numPartials + i),
                                         src=vgpr(readAddr), ds=DSModifiers(offset=off),
                                         comment=f"LDS load wave[{j}] arr[{a}] partial[{i}]."))
            module.add(SWaitCnt(dscnt=0, comment="wait LDS reads."))
            self._crossWaveAccum(module, readTmp, arrays, j)
            if j < self.wg_m - 1:
                with self.writer.allocTmpSgpr(1, tag="pRMS_xwF0Advance") as tmpSgprInfo:
                    module.add(SMovB32(dst=sgpr(tmpSgprInfo.idx), src=hex(strideW),
                                       comment=f"strideW={strideW}."))
                    module.add(VAddU32(vgpr(readAddr), vgpr(readAddr), sgpr(tmpSgprInfo.idx),
                                       comment="advance readAddr to next sibling wave."))

    def _crossWaveAccum(self, module, readTmp: int, arrays, j: int) -> None:
        for a, (base, op, verb) in enumerate(arrays):
            for i in range(self.numPartials):
                src = readTmp + a * self.numPartials + i
                if j == 0:
                    module.add(VMovB32(dst=vgpr(base + i), src=vgpr(src),
                                       comment=f"arr[{a}] partial[{i}] = wave[0]."))
                    continue
                module.add(op(dst=vgpr(base + i), src0=vgpr(base + i), src1=vgpr(src),
                              comment=f"arr[{a}] partial[{i}] {verb} wave[{j}]."))

    def _buildWriteMask(self, module, laneMaskSgpr: int, laneId: int) -> None:
        # Active iff rowGroup==0 AND waveM==0 (every lane already holds the all-reduced value).
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

    def _computeNTiles(self, module, dst: int) -> None:
        # n_d = ceil(SizesFree0 / MT0).
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

    def _computeMPadded(self, module, dst: int) -> None:
        mt1 = self.macro_tile1
        with self.writer.allocTmpSgpr(1, tag="pRMS_mPaddedS") as s:
            module.add(SAddU32(dst=sgpr(s.idx), src0=sgpr("SizesFree+1"), src1=mt1 - 1,
                               comment=f"M + MT1-1 (MT1={mt1})"))
            if mt1 & (mt1 - 1) == 0:
                module.add(SLShiftRightB32(dst=sgpr(s.idx), shiftHex=hex(mt1.bit_length() - 1),
                                           src=sgpr(s.idx), comment=f"mTiles = ceil(M/MT1={mt1})"))
            else:
                magic, postShift = _ceilDivMagic(mt1)
                module.add(SMulHIU32(dst=sgpr(s.idx), src0=sgpr(s.idx), src1=hex(magic),
                                     comment=f"mTiles magic mul (divisor={mt1})"))
                if postShift:
                    module.add(SLShiftRightB32(dst=sgpr(s.idx), shiftHex=hex(postShift),
                                               src=sgpr(s.idx), comment=f"mTiles >> {postShift}"))
            module.add(SMulI32(dst=sgpr(s.idx), src0=sgpr(s.idx), src1=mt1,
                               comment=f"M_padded = mTiles * MT1({mt1})"))
            module.add(VMovB32(dst=vgpr(dst), src=sgpr(s.idx), comment="M_paddedV = M_padded"))

    def _amaxEpilogueFree0(self, amaxPartials: int, partialSrd: int,
                           laneId: int, savedExec: int, laneMaskSgpr: int, globalAddr: int,
                           colByte: int, mPaddedV: int, scaleV: int) -> Module:
        module = Module("PartialRMSQuant amaxEpilogueFree0")
        module.addComment1("PartialRMSQuant: partial amax(|D|)/fp8_max into partialBuf second half")
        # amax is already row-group- and cross-wave-reduced in emit(); only scale+write here.
        self._computeMPadded(module, mPaddedV)
        fp8Max = self._quantOutMax()
        bits = struct.unpack('<I', struct.pack('<f', 1.0 / fp8Max))[0]
        module.add(VMovB32(dst=vgpr(scaleV), src=hex(bits),
                           comment=f"1/fp8_max ({fp8Max})"))
        for n in range(self.mma_n):
            module.add(VMulF32(dst=vgpr(amaxPartials + n), src0=vgpr(amaxPartials + n),
                               src1=vgpr(scaleV), comment=f"amax[n={n}] /= fp8_max"))
        module.add(self._writePartialsFree0(
            amaxPartials, partialSrd, laneId, savedExec, laneMaskSgpr,
            globalAddr, colByte, rowOffset=mPaddedV, label="amax/fp8_max"))
        return module

    def _writePartialsFree0(self, partials: int, partialSrd: int, laneId: int, savedExec: int,
                            laneMaskSgpr: int, globalAddr: int, colByte: int,
                            rowOffset: int = None, label: str = "Σx²") -> Module:
        module = Module("PartialRMS writePartialsFree0")
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
        if rowOffset is not None:
            module.add(VAddU32(vgpr(accumV), vgpr(tokenBase), vgpr(rowOffset),
                               comment="token0 = tokenBase + M_padded (amax second half)"))
            # token*n_d uses 32-bit VMulLOU32; assumes token*n_d < 2^32.
            module.add(VMulLOU32(dst=vgpr(accumV), src0=vgpr(ntilesV), src1=vgpr(accumV),
                                 comment="accum = token0 * n_d"))
        else:
            module.add(VMulLOU32(dst=vgpr(accumV), src0=vgpr(ntilesV), src1=vgpr(tokenBase),
                                 comment="accum = tokenBase * n_d"))
        strideV = None
        if self.mma_n > 1:
            strideV = self.writer.vgprPool.checkOut(1, tag="pRMS_wF0Stride")
            module.add(VMulLOU32(dst=vgpr(strideV), src0=self.mfma_n, src1=vgpr(ntilesV),
                                 comment=f"stride = mfma_n({self.mfma_n}) * n_d"))
        for n in range(self.mma_n):
            module.add(VAddU32(vgpr(globalAddr), vgpr(accumV), sgpr("WorkGroup0"),
                               comment=f"token*n_d + WorkGroup0 (n={n})"))
            module.add(VLShiftLeftB32(dst=vgpr(globalAddr), shiftHex=hex(2), src=vgpr(globalAddr),
                                      comment="byteOff = (token*n_d + WG0) * 4"))
            module.add(BufferStoreB32(src=vgpr(partials + n), vaddr=vgpr(globalAddr),
                                      saddr=sgpr(partialSrd, 4), soffset=0,
                                      mubuf=MUBUFModifiers(offen=True),
                                      comment=f"partialBuf[token+n*{self.mfma_n}, WG0] = {label} (n={n})"))
            if n < self.mma_n - 1:
                module.add(VAddU32(vgpr(accumV), vgpr(accumV), vgpr(strideV),
                                   comment=f"accum += stride (advance to n={n + 1})"))
        module.add(SWaitCnt(vscnt=0, comment="wait partialBuf stores"))
        module.add(SMovB64(dst=EXEC(), src=sgpr(savedExec, lsc), comment="restore exec mask"))
        if strideV is not None:
            self.writer.vgprPool.checkIn(strideV)
        self.writer.vgprPool.checkIn(accumV)
        self.writer.vgprPool.checkIn(tokenBase)
        self.writer.vgprPool.checkIn(ntilesV)
        return module

    def _issueGammaLoads(self, module, gammaSrd: int, gammaBurst: int, gammaByteVgpr: int,
                         m: int) -> None:
        """Issue all rows_per_lane gamma loads for tile row m (no wait; batched)."""
        for k in range(self.rows_per_lane):
            self._addImmU32(module, gammaByteVgpr, self._nhBaseV, k, gammaByteVgpr,
                            f"nhidden = nhBase + {k} (m={m},k={k}).")
            module.add(VLShiftLeftB32(dst=vgpr(gammaByteVgpr), shiftHex=hex(self.gammaLog2Bytes),
                                      src=vgpr(gammaByteVgpr),
                                      comment="gammaByte = nhidden * gammaBytes."))
            self._issueSideLoad(module, gammaBurst + k, gammaByteVgpr, gammaSrd,
                                f"gamma[nhidden] (m={m},k={k}).", dtype=self.gammaType)

    def _convertGammaRow(self, module, gammaBurst: int) -> None:
        for k in range(self.rows_per_lane):
            self._convertSideElem(module, gammaBurst + k, f"gamma -> fp32 (k={k}).", dtype=self.gammaType)

    def _fusedAccElement(self, module, vgprTiles, accReg: int, gammaReg,
                         partials, amaxPartials, absMask, m: int, n: int, k: int, first: bool) -> None:
        # accReg already holds H as written by the preceding ResidualAdd epilogue.
        pidx = partials + n
        if first:
            module.add(VMulF32(dst=vgpr(pidx), src0=vgpr(accReg), src1=vgpr(accReg),
                               comment=f"partial[n={n}] = H^2."))
        else:
            module.add(VFmaF32(dst=vgpr(pidx), src0=vgpr(accReg), src1=vgpr(accReg),
                               src2=vgpr(pidx), comment=f"partial[n={n}] += H^2."))
        module.add(VMulF32(dst=vgpr(accReg), src0=vgpr(accReg), src1=vgpr(gammaReg),
                           comment="acc *= gamma."))
        self._writeAccFrom(module, accReg, vgprTiles, m, n, k, f"write acc[m={m},n={n},k={k}].")
        if amaxPartials is None:
            return
        # amax uses the post-gamma value; accReg is already written back so it is free to clobber.
        aidx = amaxPartials + n
        module.add(VAndB32(dst=vgpr(accReg), src0=vgpr(accReg), src1=vgpr(absMask),
                           comment="|D| (post-gamma)."))
        if first:
            module.add(VMovB32(dst=vgpr(aidx), src=vgpr(accReg), comment=f"amax[n={n}] = |D|."))
        else:
            module.add(VMaxF32(dst=vgpr(aidx), src0=vgpr(aidx), src1=vgpr(accReg),
                               comment=f"amax[n={n}] = max(amax, |D|)."))

    def _fusedAccRow(self, module, vgprTiles, accBurst: int, gammaBurst,
                     partials, amaxPartials, absMask, m: int) -> None:
        for k in range(self.rows_per_lane):
            coords = [(m, n, k) for n in range(self.mma_n)]
            srcRegs = self._readAccBurst(module, accBurst, vgprTiles, coords, f"read acc m={m},k={k}.")
            first = m == 0 and k == 0
            for n in range(self.mma_n):
                gammaReg = None if gammaBurst is None else gammaBurst + k
                self._fusedAccElement(module, vgprTiles, srcRegs[n], gammaReg,
                                      partials, amaxPartials, absMask, m, n, k, first)

    def _fusedAccBlock(self, module, vgprTiles, accBurst: int, gammaBurst: int,
                       partials, amaxPartials, absMask, m: int) -> None:
        """Convert one loaded gamma block to fp32 and fold it into acc row m."""
        self._convertGammaRow(module, gammaBurst)
        self._fusedAccRow(module, vgprTiles, accBurst, gammaBurst, partials,
                          amaxPartials, absMask, m)

    def _emitBlockPair(self, module, vgprTiles, accBurst: int, gammaBurst: int,
                       gammaSrd: int, gammaByteVgpr: int, mBaseVgpr: int,
                       wgRowBase: int, rowGroupOff: int, partials, amaxPartials,
                       absMask, m: int) -> None:
        """Pipeline blocks m and m+1: overlap block m+1's gamma loads with block m compute."""
        rowsPerLane = self.rows_per_lane
        paired = (m + 1) < self.mma_m
        self._free0RowPos(module, self._nhBaseV, wgRowBase, rowGroupOff, m, 0, mBaseVgpr)
        self._issueGammaLoads(module, gammaSrd, gammaBurst, gammaByteVgpr, m)
        if paired:
            self._addImmU32(module, self._nhBaseV, self._nhBaseV, self.mfma_m, mBaseVgpr,
                            f"nhBase += mfma_m (advance to block {m + 1}).")
            self._issueGammaLoads(module, gammaSrd, gammaBurst + rowsPerLane, gammaByteVgpr, m + 1)
            module.add(SWaitCnt(vlcnt=rowsPerLane,
                                comment="wait block m gamma; block m+1 still in flight."))
        else:
            module.add(SWaitCnt(vlcnt=0, comment="wait gamma row burst."))
        self._fusedAccBlock(module, vgprTiles, accBurst, gammaBurst, partials,
                            amaxPartials, absMask, m)
        if not paired:
            return
        module.add(SWaitCnt(vlcnt=0, comment="wait block m+1 gamma."))
        self._fusedAccBlock(module, vgprTiles, accBurst, gammaBurst + rowsPerLane, partials,
                            amaxPartials, absMask, m + 1)

    def _fusedAccPassFree0(self, vgprTiles, gammaSrd: int, partials,
                           amaxPartials, absMask, scratchV: int) -> Module:
        module = Module("PartialRMS fusedAccPassFree0")
        module.addComment1("PartialRMS fused pass (free0): Σx², gamma, and amax in one sweep")
        rowGroupOff = self.writer.vgprPool.checkOut(1, tag="pRMS_fusRGOff")
        wgRowBase = self.writer.vgprPool.checkOut(1, tag="pRMS_fusWgRowBase")
        self._computeRowGroupOff(module, rowGroupOff)
        self._computeFree0RowBase(module, wgRowBase)
        # Shared nhidden base (k=0) per tile row; computed once and reused by
        # gamma instead of being recomputed in each load.
        self._nhBaseV = self.writer.vgprPool.checkOut(1, tag="pRMS_fusNhBase")
        # Reuse globalAddr (scratchV) as the gamma-byte scratch; free until _writePartialsFree0.
        gammaByteVgpr = scratchV
        mBaseVgpr = self.writer.vgprPool.checkOut(1, tag="pRMS_fusMBase")
        accBurst = self.writer.vgprPool.checkOut(self.mma_n, tag="pRMS_fusAccBurst")
        gammaBurst = self.writer.vgprPool.checkOut(2 * self.rows_per_lane, tag="pRMS_fusGammaBurst")
        for m in range(0, self.mma_m, 2):
            self._emitBlockPair(module, vgprTiles, accBurst, gammaBurst, gammaSrd,
                                gammaByteVgpr, mBaseVgpr, wgRowBase, rowGroupOff,
                                partials, amaxPartials, absMask, m)
        self.writer.vgprPool.checkIn(gammaBurst)
        self.writer.vgprPool.checkIn(accBurst)
        self.writer.vgprPool.checkIn(mBaseVgpr)
        self.writer.vgprPool.checkIn(self._nhBaseV)
        self.writer.vgprPool.checkIn(wgRowBase)
        self.writer.vgprPool.checkIn(rowGroupOff)
        return module
