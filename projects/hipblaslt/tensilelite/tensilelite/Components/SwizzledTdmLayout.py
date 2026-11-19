# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Geometry of a gfx1250 swizzled-TDM operand (SwizzleTensorA/B staged through TDM).

The host pre-swizzles the operand into WMMA order, per batch:

    [nO = free // MI, kO = K // swzK, kM = wave // MI, nI = free % MI, kI = innerK]

with the free dimension padded to MI rows and K padded to PAD_K_BYTES (paddedK). One
innerK block is one 16-byte ds_load_b128; F4 (0.5 byte) is handled as packed
bytes, so its byte layout equals F8's. TDM copies one DepthU K-slice of every nO
row of the macro-tile into LDS (nO-major, optionally padded per nO tile), and the
WMMA operand is read back with plain ds_load_b128.

This module is the single source of the constants and offsets the kernel writer
emits for that path; Tests/unit/test_swizzled_tdm_layout.py checks it against an
independent model of the host layout, TDM and the WMMA lane mapping. Offsets are
in elements unless named *Bytes; descriptor extents are in descriptor units
(elements, or bytes for F4, see unitShift).
"""

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

MAX_LDS_CONST_OFFSET = 65536   # ds_load immediate range; a larger offset selects LocalReadAddr+i

# The host pads K to this many bytes, on every architecture. Flattening (kO, kM) gives the index of
# the 16-byte K block, so for a given padded K the swizzled bytes are identical whatever the kernel's
# swzK is; padding to the gfx942 granule (64 B, hipblasLtOrder_t COL16_4R*) keeps one host layout and
# one hipBLASLt user contract across architectures.
PAD_K_BYTES = 64


@dataclass(frozen=True)
class SwizzledTdmGeometry:
    bpe: float          # element bytes (0.5 for F4)
    mi: int             # MI extent along the swizzled free dim (MatrixInstM for A, MatrixInstN for B)
    wave: int           # wavefront size
    miInputPerThread: int  # elements of one WMMA operand per lane (MIInputPerThread{A,B})
    depthU: int         # _DepthU{A,B}
    macroTile: int      # MacroTile along the free dim
    waveTile: int       # MIWaveTile along the free dim
    numWaves: int       # MIWaveGroup along the free dim
    vw: int             # VectorWidth{A,B}
    padElems: int = 0   # LdsPad{A,B} (VW>1 per-nO-tile pad), elements

    # ---- construction -------------------------------------------------------------------
    @classmethod
    def fromKernel(cls, kernel: Mapping, tc: str) -> "SwizzledTdmGeometry":
        tile01 = 0 if tc == "A" else 1
        padElems = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] else 0
        return cls(bpe=kernel["ProblemType"]["DataType%s" % tc].numBytes(),
                   mi=kernel["MatrixInstM"] if tile01 == 0 else kernel["MatrixInstN"],
                   wave=kernel["WavefrontSize"],
                   miInputPerThread=kernel["MIInputPerThread%s" % tc],
                   depthU=kernel["_DepthU%s" % tc],
                   macroTile=kernel["MacroTile%u" % tile01],
                   waveTile=kernel["MIWaveTile"][tile01],
                   numWaves=kernel["MIWaveGroup"][tile01],
                   vw=kernel["VectorWidth%s" % tc],
                   padElems=int(padElems))

    # ---- swizzle block ------------------------------------------------------------------
    @property
    def innerK(self) -> int:
        """K elements in one 16-byte ds_load_b128 (kI extent)."""
        innerK = int(16 / self.bpe)
        assert innerK > 0 and innerK * self.bpe == 16, "innerK x bpe must be 16 bytes"
        return innerK

    @property
    def kM(self) -> int:
        """K blocks interleaved across the lane groups of one wave (wave // MI)."""
        return self.wave // self.mi

    @property
    def swzK(self) -> int:
        """Host swizzle K granule (elements): K is padded to a multiple of it."""
        return self.innerK * self.kM

    @property
    def padK(self) -> int:
        """K elements the host pads to (PAD_K_BYTES); a multiple of swzK."""
        padK = int(PAD_K_BYTES / self.bpe)
        assert padK % self.swzK == 0, "host K padding must be a whole number of swizzle K granules"
        return padK

    @property
    def unitShift(self) -> int:
        """Right shift from element counts to TDM descriptor units (F4 is described in bytes)."""
        return 1 if self.bpe == 0.5 else 0

    @property
    def blockOff(self) -> int:
        """Stride of one [nI, kI] block (the kM stride)."""
        return self.mi * self.innerK

    @property
    def kOStride(self) -> int:
        """Stride of one swzK K block (kO) inside an nO tile."""
        return self.kM * self.blockOff

    @property
    def numKChunk(self) -> int:
        """ds_load_b128 per lane per WMMA operand (kO blocks per MatrixInstK)."""
        return self.miInputPerThread // self.innerK

    @property
    def kStepPerMI(self) -> int:
        """LDS element advance per WMMA K step (numKChunk kO blocks)."""
        return self.numKChunk * self.kOStride

    @property
    def nORows(self) -> int:
        """nO rows (MI row blocks) in one macro-tile."""
        assert self.macroTile % self.mi == 0
        return self.macroTile // self.mi

    @property
    def nOTile(self) -> int:
        """Elements of one nO row's DepthU slice: the TDM tile0 and the unpadded LDS nO stride."""
        return self.mi * self.depthU

    @property
    def nOStridePadded(self) -> int:
        """LDS nO stride including the VW>1 pad."""
        return self.nOTile + self.padElems

    @property
    def nOTileBytes(self) -> int:
        return int(self.nOTile * self.bpe)

    @property
    def padBytes(self) -> int:
        return int(round(self.padElems * self.bpe))

    def paddedK(self, sizeL: int) -> int:
        """K extent of one nO row in the host buffer (its row stride is MI * paddedK)."""
        return -(-sizeL // self.padK) * self.padK

    def validK(self, sizeL: int) -> int:
        """K extent the kernel reads for the tail: whole swzK granules covering sizeL (the rest of
        paddedK is zero padding that is never loaded)."""
        return -(-sizeL // self.swzK) * self.swzK

    # ---- TDM descriptor -----------------------------------------------------------------
    def compRows(self, numComp: int) -> Tuple[int, bool]:
        """(nO rows per wave component, replicate).

        The swizzled layout cannot be split below one nO row. When the rows do not
        divide over the components, every component loads the whole tile from the same
        global address into the same LDS address (identical bytes)."""
        nO = self.nORows
        if numComp <= 1 or nO % numComp == 0:
            return nO // max(numComp, 1), False
        return nO, True

    def descTile0(self) -> int:
        return self.nOTile >> self.unitShift

    def descStride0(self, sizeL: int) -> int:
        """Global nO row stride (MI * paddedK), descriptor units."""
        return (self.mi * self.paddedK(sizeL)) >> self.unitShift

    def descTailDim0(self, sizeL: int) -> int:
        """Tail-loop dim0: the valid K prefix of an nO row, MI * alignUp(K % DepthU, swzK)."""
        return (self.mi * self.validK(sizeL % self.depthU)) >> self.unitShift

    def remainingRows(self, sizeFree: int, wg: int, comp: int = 0, rowsPerComp: int = 0) -> int:
        """dim1: nO rows left in the (MI-padded) buffer from this load's first row."""
        rows = -(-sizeFree // self.mi) - wg * self.nORows - comp * rowsPerComp
        return max(0, rows)

    def iterLdsIncBytes(self) -> int:
        """TDM iterate-mode LDS step per nO tile (VW>1 pad)."""
        return self.nOTileBytes + self.padBytes

    # ---- global offsets (bytes) ---------------------------------------------------------
    def tileStrideBytes(self, sizeL: int) -> int:
        """Bytes between consecutive macro-tiles: MT * paddedK (the host ignores the ld)."""
        return int(self.paddedK(sizeL) * self.macroTile * self.bpe)

    def compGlobalOffsetBytes(self, comp: int, sizeL: int, numComp: int) -> int:
        rowsPerComp, replicate = self.compRows(numComp)
        if replicate:
            return 0
        return int(comp * rowsPerComp * self.mi * self.paddedK(sizeL) * self.bpe)

    def kIterBytes(self) -> int:
        """Global advance per DepthU iteration (and per GSU iteration offset unit)."""
        return int(self.mi * self.depthU * self.bpe)

    def compLdsOffsetBytes(self, comp: int, numComp: int) -> int:
        """LDS offset of a wave component's slice (non-LSI layout), pad included."""
        rowsPerComp, replicate = self.compRows(numComp)
        if replicate:
            return 0
        return comp * rowsPerComp * (self.nOTileBytes + self.padBytes)

    # ---- local read -----------------------------------------------------------------------
    def laneBase(self, lane: int) -> int:
        """Per-lane LDS element offset within a wave's tile (lraTileAssignmentSwizzledTDM)."""
        kMl, nIdx = divmod(lane % self.wave, self.mi)
        nVW = self.vw * nIdx
        nOl, nIl = divmod(nVW, self.mi)
        return kMl * self.blockOff + nOl * self.nOTile + nIl * self.innerK

    def waveBase(self, wtid0: int) -> int:
        """Per-wave LDS element offset of its nO band (when it does not REPLACE by an LSI jump)."""
        return wtid0 * self.vw * self.nOTile

    def padBaseBytes(self, baseBytes: int) -> int:
        """Lane/wave base with the per-nO-tile pad applied (lraFinalOffset)."""
        if not self.padElems:
            return baseBytes
        return baseBytes + (baseBytes // self.nOTileBytes) * self.padBytes

    def readOffsetElems(self, nt: int, r: int, compNO: Optional[int] = None) -> Tuple[int, int]:
        """(component, element immediate) of ds_load (nt, r).

        nt = vIdx*vw + eIdx. compNO (nO rows per LDS component) is given for
        LDSSegmentInterleave when B does not span its component: the vIdx group then
        selects the component. Otherwise the component is 0."""
        vIdx, eIdx = divmod(nt, self.vw)
        vNO = vIdx * self.vw * self.numWaves
        comp = 0
        if compNO is not None:
            assert compNO % (self.vw * self.numWaves) == 0, "a vIdx group must not straddle a component"
            comp, vNO = divmod(vNO, compNO)
        return comp, vNO * self.nOStridePadded + eIdx * self.innerK + r * self.kOStride

    def readOffsetBytes(self, nt: int, r: int, extraElems: int = 0, compNO: Optional[int] = None,
                        readStride: int = 0) -> Tuple[int, int]:
        """(component, byte offset from LocalReadAddr+0) of ds_load (nt, r).

        extraElems is the K-sub-iteration progression (localReadOffset); the component jump
        (readStride, post-pad) is added when compNO is given (see readOffsetElems)."""
        comp, imm = self.readOffsetElems(nt, r, compNO)
        return comp, comp * readStride + int(round((imm + extraElems) * self.bpe))

    @staticmethod
    def selectReadRegister(offBytes: int, windows: Optional[Sequence[int]] = None,
                           maxOff: int = MAX_LDS_CONST_OFFSET) -> Tuple[int, int]:
        """(LocalReadAddr index, ds_load immediate) for a byte offset from LocalReadAddr+0.

        LocalReadAddr+i holds +0 plus window i (LDSSegmentInterleave read windows) or i*64K."""
        window = offBytes // maxOff
        regIdx = list(windows).index(window) if windows is not None else window
        return regIdx, offBytes - window * maxOff
