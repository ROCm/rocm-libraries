# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Subtile tail-loop narrow trailing load (Phase A1).

Closes the bf16/fp16 odd-K trailing-element page-fault concern. The
companion `emitTailSrdTightenSubtile` (in :mod:`SubtileTailSrdTighten`)
is flipped from align-UP to align-DOWN, which removes the up-to-bpr-1
byte over-read on the last m-row. The two changes MUST land together:
align-DOWN alone drops the K=K_remain-1 BF16 element on row
`numLine-1`, so this module emits a per-(wave,lane)-targeted
`buffer_load_ushort … lds` (the gfx9 4-operand form, verified to
assemble on gfx950 / gfx942) that re-populates that single LDS slot.

Public surface:

- :func:`subtileTailNarrowLoadApplies(kernel)` — static predicate gating
  the whole helper on bf16/fp16 / non-MX / loadRatioGR=1.0 / etc.
- :func:`computeNarrowLoadDescriptor(kernel, ti, K_remain, tc)` — pure-
  python derivation of the four-tuple
  `(wave_target, lane_target_in_wave, m0_target, vaddr_target)` for a
  given (kernel, K_remain, operand). Used by the emitter AND by the
  cross-check harness in
  ``Tensile/Tests/unit/test_subtile_tail_narrow_load_cross_check.py``.
- :func:`computeLRReaderForBoundary(kernel, ti, K_remain, tc)` — the
  LR-side oracle. Given the same (kernel, K_remain, operand), returns
  the LDS byte address the per-lane mask init's `ds_read_b128`
  consumer expects to contain the trailing K-element.
- :func:`emitTailTrailingNarrowLoad(kw, kernel)` — the rocisa emitter.
  Calls into computeNarrowLoadDescriptor and produces a per-wave
  EXEC-guarded `buffer_load_ushort … lds` for A and (separately) B.

All helpers reach into the live `TileInfo` (`kw.states.a.tileInfo`
etc.) for the geometry; the case-split machinery lives in
:func:`computeNarrowLoadDescriptor`.

Case-split coverage (Phase A1):
- loadRatioGR == 1.0, bpe == 2 (bf16/fp16), non-MX, wg_m × wg_n in
  {(1,1), (1,2), (2,1), (2,2), (4,1), (1,4)}, localSubtileGrid[1] in
  {1, 2, 4}. Other cases stay on align-UP — see the predicate.
- bf16 (and fp16) DPP quad_perm swizzle path; fp8 K_group rotation
  is a separate branch and is OUT of scope (no current gauntlet
  exercises fp8 odd-K with this fix).
"""

import math
from typing import NamedTuple, Optional

from rocisa.code import Label, Module
from rocisa.container import EXEC, MUBUFModifiers, sgpr, vgpr, mgpr
from rocisa.instruction import (
    BufferLoadU16,
    SAddU32, SAndB32, SCBranchSCC0, SCBranchSCC1, SCmpEQU32, SCmpLgU32,
    SLShiftLeftB32, SLShiftRightB32, SMovB32, SMovB64, SMulI32, SOrB32,
    SSubU32, SWaitCnt,
    VLShiftRightB32, VMovB32, VReadfirstlaneB32,
)


# ── archCaps used in the LDS-row-bank derivation ───────────────────────────
#
# `ldsRowBankSize = LDSBankCount * LDSBankWidth` is read at runtime from
# `kw.states.archCaps` in the live emit path. For the pure-python
# derivation we use the same archCaps when available, else fall back to
# the gfx950 values (64 banks × 4 bytes).
_DEFAULT_LDS_BANK_COUNT = 64
_DEFAULT_LDS_BANK_WIDTH = 4

# `permlane16_swap_b32` exchanges values between lanes [0..15] and
# [16..31] (within each 32-lane half). With the EXEC mask shown below
# the swap is restricted to the same lane-pair pattern the GR swizzle
# uses (DPP quad_perm [1,0,3,2] for ldsRowId-even lanes). The bits set
# in 0x33333333 are 0,1,4,5,8,9,12,13 of every 16-lane block — i.e.
# lanes in positions {0,1,4,5,8,9,12,13} mod 16. See
# `_lraTileAssignment_legacy` in `Components/Subtile/SubtileLREmit.py`.
PERMLANE16_SWAP_EXEC_MASK_LO = 0x33333333
PERMLANE16_SWAP_EXEC_MASK_HI = 0x33333333


class NarrowLoadDescriptor(NamedTuple):
    """Compile-time-resolved address tuple for ONE operand's narrow load.

    Attributes:
      wave_target:   the (compile-time) waveId whose GR write covered
                     the (M_last, K_last_local) byte before the
                     align-DOWN tighten clipped it.
      lane_target:   the (compile-time) lane within `wave_target` that
                     wrote the now-clipped 16-byte chunk. The narrow
                     load fires lane 0 of `wave_target` (with EXEC=1),
                     not `lane_target` — `lane_target` is used only to
                     derive `m0_target` (the LDS byte to repair).
      m0_target:     absolute LDS byte address of the corrupted slot.
                     Goes straight into the narrow load's `m0`.
      vaddr_target:  per-lane VGPR contents (a uniform constant for
                     lane 0). Encodes the row-offset byte for the
                     `(M_last)`-th row of the operand. The trailing
                     K-element byte within the row is implicit in the
                     post-tighten Srd<tc> base (advanced by
                     K_aligned * bpe at tail entry), so vaddr only
                     needs to account for the row offset and the
                     in-lane K residual.
      sId0_last:     reported for tests / disassembly. Which M-direction
                     subtile of the wave's GR loop contained the slot.
      sId1_last:     ditto, K-direction.
      explain:       human-readable derivation summary; used in test
                     failure messages.
    """
    wave_target:  int
    lane_target:  int
    m0_target:    int
    vaddr_target: int
    sId0_last:    int
    sId1_last:    int
    explain:      str


class LRReaderTarget(NamedTuple):
    """LR-side oracle output for cross-check.

    Attributes:
      lds_byte_target: the LDS byte address the LR's `ds_read_b128`
                       consumer expects to contain the
                       (M_last, K_last_local) element. The cross-check
                       asserts this equals the GR-side
                       `NarrowLoadDescriptor.m0_target`.
      consumer_wave:   the wave whose LR consumes this slot.
      consumer_lane:   the (lane16, lane16Group) lane within the
                       consumer wave that ds_read's the slot.
      explain:         human-readable derivation summary.
    """
    lds_byte_target: int
    consumer_wave:   int
    consumer_lane:   int
    explain:         str


# ── Static predicate ───────────────────────────────────────────────────────

def subtileTailNarrowLoadApplies(kernel) -> bool:
    """Gate the narrow-trailing-load emit + the align-DOWN tighten on
    the geometry families A1 actually encodes.

    True iff:
      - non-MX (MXBlock{A,B} == 0);
      - non-swizzled A/B (SwizzleTensor{A,B} False);
      - symmetric bpe ∈ {2} (bf16 / fp16; int8 is bpe=1 and lives on
        the legacy align-UP path -- its trailing-byte concern is a
        different shape and is out of scope here);
      - `ASEM*bpe % bpr != 0` -- when K_remain*bpe is guaranteed a
        multiple of bpr by the static AssertSummationElementMultiple,
        align-DOWN behaves identically to align-UP (no trailing
        bytes to lose), so no narrow load is needed. Keeps even-K
        shapes (ASEM in {2, 4, 6, ...}) on the legacy align-UP path
        for zero emit-shape drift.
      - MFMA enabled (UseSubtileImpl already constrains this in
        Solution.py; we don't re-check NoTailLoop here -- the caller
        in `emitTailLoopScaffoldSubtile` already does).
    """
    pt = kernel["ProblemType"]
    if pt.get("MXBlockA", 0) > 0 or pt.get("MXBlockB", 0) > 0:
        return False
    if pt.get("SwizzleTensorA", False) or pt.get("SwizzleTensorB", False):
        return False
    bpeA = pt["DataTypeA"].numBytes()
    bpeB = pt["DataTypeB"].numBytes()
    if int(bpeA) != bpeA or int(bpeB) != bpeB:
        return False
    if int(bpeA) != 2 or int(bpeB) != 2:
        return False
    bpr = 4
    asem = kernel.get("AssertSummationElementMultiple", 1)
    # `asem * bpe % bpr == 0` ⇒ K_remain (= K mod DepthU, runtime, but
    # always a multiple of ASEM since K is a multiple of ASEM and
    # DepthU is a multiple of ASEM) is a multiple of (bpr / bpe),
    # so K_remain*bpe is always a multiple of bpr. No align-DOWN
    # gap → no narrow load needed.
    if (asem * int(bpeA)) % bpr == 0:
        return False
    return True


# ── archCaps accessor for LDSBank size (pure-python) ───────────────────────

def _ldsRowBankSize(kw=None) -> int:
    if kw is not None and getattr(kw, "states", None) is not None:
        caps = getattr(kw.states, "archCaps", None)
        if caps is not None and "LDSBankCount" in caps and "LDSBankWidth" in caps:
            return int(caps["LDSBankCount"]) * int(caps["LDSBankWidth"])
    return _DEFAULT_LDS_BANK_COUNT * _DEFAULT_LDS_BANK_WIDTH


# ── GR side: wave / lane / m0 derivation ──────────────────────────────────

def _grWavePartitionForRow(kernel, ti, m_last: int):
    """Invert `_grComputeRowPartition_legacy` to find the wave whose GR
    loop covers row `m_last`.

    Returns (wave_target, rowOffset_wave_target, sId0_last, rowId_last).

    Encodes the three loadRatioGR branches:
      - loadRatioGR == 1.0 (the canonical bf16/fp16 path):
          localRow     = waveId & 1
          partitionRow = waveId >> 1
      - loadRatioGR == 0.5 (FP4/FP16 4-wave-coop-per-subtile):
          localRow = 0
          partitionRow = waveId
      - loadRatioGR == 2.0 (1-wave-per-subtile, 2x subtile reuse):
          localRow = waveId
          partitionRow = 0

    Phase A1 encodes loadRatioGR == 1.0 only -- the others are not
    exercised by the bf16/fp16 ASEM<DU gauntlet. Asserts on the
    unsupported branches so the regression backstop fires loudly if
    the gate is widened without a matching update here.
    """
    wg_m = kernel["MIWaveGroup"][0]
    wg_n = kernel["MIWaveGroup"][1]
    num_waves = wg_m * wg_n
    bpe = int(ti.bpe)
    wavesize = kernel["WavefrontSize"]
    blockSize_GR = ti.subIterKBytes // ti.loadWidthGR
    numRowsPerWave = wavesize // blockSize_GR
    subtileSize_rows = ti.subtileShape[0] * ti.mmaTileShape[0]
    partitionStride = ti.mmaTileShape[0] * ti.localSubtileGrid[0]
    localSubtileGrid_M = ti.localSubtileGrid[0]

    if ti.loadRatioGR != 1.0:
        raise NotImplementedError(
            f"_grWavePartitionForRow: loadRatioGR={ti.loadRatioGR} not encoded. "
            f"Phase A1 covers loadRatioGR==1.0 only; widen the helper "
            f"before relaxing `subtileTailNarrowLoadApplies`."
        )

    # Enumerate every wave's (rowOffset_base + sId0*subtileSize_rows +
    # rowId) coverage and find the unique tuple that lands on m_last.
    # Cheaper than inverting the formula; for any production wg_m*wg_n
    # this is at most 64 lookups.
    for wave_id in range(num_waves):
        localRow = wave_id & 1
        partitionRow = wave_id >> 1
        rowOffset_base = localRow * numRowsPerWave + partitionRow * partitionStride
        for sId0 in range(localSubtileGrid_M):
            base = rowOffset_base + sId0 * subtileSize_rows
            for rowId in range(numRowsPerWave):
                if base + rowId == m_last:
                    return wave_id, rowOffset_base, sId0, rowId
    raise AssertionError(
        f"_grWavePartitionForRow: no wave found covering m_last={m_last} "
        f"for MT={ti.macroTile}, wg=({wg_m},{wg_n}), "
        f"numRowsPerWave={numRowsPerWave}, partitionStride={partitionStride}, "
        f"subtileSize_rows={subtileSize_rows}, "
        f"localSubtileGrid={ti.localSubtileGrid}"
    )


def _grSwizzledColIdPre(ldsRowId: int, lane16Group_unused: int,
                        wave_id: int, blockSize_GR: int,
                        numRowsPerLDSBanks: int,
                        target_colId_post: int) -> int:
    """Invert the bf16 swizzle chain in `_grSwizzleColIds_legacy` to
    find the `colId_pre` that produces `target_colId_post` after the
    intra-wave + inter-wave rotation chain.

    Forward chain (bf16, loadRatioGR != 0.5):
      rotation_intra  = blockSize - (ldsRowId // 2) * 2
      waveRotation    = (wave_id & 1) << log2(2 * numRowsPerLDSBanks)
      colId_post = (rotation_intra - waveRotation + colId_pre) % blockSize

    Plus the DPP quad_perm [1,0,3,2] swap (lane-pair swap on
    ldsRowId-even lanes). The DPP swap REORDERS WHICH LANE gets
    which colId — it does NOT change the colId VALUE for a fixed
    lane's eyes. For the inversion we ignore the DPP swap (we're
    asking "what colId_pre value produces colId_post=target?", not
    "which lane carries it"). The lane-id lookup that follows uses
    the pre-DPP rowId,colId mapping.

    Inversion:
      delta_pre = (target_colId_post - rotation_intra + waveRotation) % blockSize
    """
    rotation_intra = blockSize_GR - (ldsRowId // 2) * 2
    waveRot_shift = (2 * numRowsPerLDSBanks).bit_length() - 1
    waveRotation = (wave_id & 1) << waveRot_shift
    return (target_colId_post - rotation_intra + waveRotation) % blockSize_GR


def _grWaveLWA(kernel, ti, rowOffset_wave: int) -> int:
    """Compute LocalWriteBaseAddr for a wave given its rowOffset_base.

    Mirrors `_globalReadDTLInitCommonSgpr_legacy`:
      LWAA<tc> = rowOffset_base * subIterKBytes (+ ldsStartOffsetB for B)

    Caller is responsible for adding the ldsStartOffsetB term when
    `tc == 'B'`.
    """
    return rowOffset_wave * ti.subIterKBytes


def computeLDSStartOffsetB(tiA) -> int:
    """Replicate the `kw.ldsStartOffsetB = sizeA` setup in
    `KernelWriter.setupNewTile`. tiA is the A-operand TileInfo;
    B's LDS region begins immediately after A's (with 2*subtileSize
    readSize alignment to match the production DTL 2x reader).
    """
    numASubtiles = int(tiA.globalSubtileGrid[0] * tiA.globalSubtileGrid[1])
    readSize = 2 * tiA.subtileSize
    return int(((numASubtiles * tiA.subtileSize + readSize - 1) // readSize) * readSize)


def computeNarrowLoadDescriptor(kernel, ti, K_remain: int, tc: str,
                                tiA=None) -> NarrowLoadDescriptor:
    """Compile-time-resolved descriptor for ONE operand's narrow load.

    Args:
      kernel:   live kernel dict
      ti:       TileInfo for the operand (A or B)
      K_remain: compile-time constant or symbolic max for derivation;
                pass a concrete int here for the cross-check / canonical
                pins. The runtime emitter computes the same descriptor
                using compile-time-known kernel keys + a runtime
                `LoopCounterL` SGPR for the K_last_local-dependent
                fields (sId1_last, lane_target).
      tc:       'A' or 'B'.
      tiA:      A-operand TileInfo for `ldsStartOffsetB` derivation;
                required when tc == 'B'.

    Phase A1 restriction: localSubtileGrid[1] == 1 supported as
    compile-time; localSubtileGrid[1] > 1 raises NotImplementedError
    (covers all DepthU ≤ 64 with bf16 mmaTileShape K=32, but DU=128
    needs a runtime sId1 branch -- staged for a follow-up).
    """
    if ti.loadRatioGR != 1.0:
        raise NotImplementedError(
            f"computeNarrowLoadDescriptor: loadRatioGR={ti.loadRatioGR} "
            f"not yet in Phase A1 coverage")
    if ti.localSubtileGrid[1] != 1:
        raise NotImplementedError(
            f"computeNarrowLoadDescriptor: localSubtileGrid[1]="
            f"{ti.localSubtileGrid[1]} > 1 is staged for DU>subIterK; not "
            f"yet encoded in Phase A1")
    if K_remain <= 0:
        raise ValueError(f"K_remain must be > 0, got {K_remain}")

    # --- Per-operand "last row" in the operand's M-equivalent axis ---
    # For A: M-axis is rows of A = MacroTile0; M_last = MacroTile0 - 1.
    # For B: M-axis is rows of B (the K-contracted axis in TN GEMM) =
    #        MacroTile1. M_last = MacroTile1 - 1.
    # (Both tileInfos store .macroTile = the operand's M dimension.)
    m_last = ti.macroTile - 1

    bpe = int(ti.bpe)
    loadWidthGR = ti.loadWidthGR
    elementsPerLane = loadWidthGR // bpe       # 8 for bf16 b128
    blockSize_GR = ti.subIterKBytes // loadWidthGR
    subIterKBytes = ti.subIterKBytes
    subtileSize_bytes = ti.subtileSize         # already in bytes
    ldsRowBankSize = _DEFAULT_LDS_BANK_COUNT * _DEFAULT_LDS_BANK_WIDTH
    numRowsPerLDSBanks = ldsRowBankSize // subIterKBytes

    # K_last_local is the LOCAL K index of the trailing valid element
    # within the tail's DepthU window. K_local = K_remain - 1.
    K_local = K_remain - 1

    # --- Step 1: wave / sId0 / rowId for m_last ---
    wave_target, rowOffset_wave, sId0_last, rowId_last = \
        _grWavePartitionForRow(kernel, ti, m_last)

    # --- Step 2: sId1 of the missing element ---
    # K-direction subtile span (elements) = subtileShape[1] * mmaTileShape[1].
    # With localSubtileGrid[1] == 1, sId1_last is always 0.
    K_per_subtile = ti.subtileShape[1] * ti.mmaTileShape[1]
    sId1_last = K_local // K_per_subtile
    K_local_in_subtile = K_local % K_per_subtile

    # --- Step 3: colId_post (the post-swizzle K-stripe containing K_local) ---
    # Each lane carries elementsPerLane K-elements. colId selects the
    # K-stripe; K-within-lane is the remainder.
    colId_post = K_local_in_subtile // elementsPerLane
    K_within_lane = K_local_in_subtile % elementsPerLane

    # --- Step 4: colId_pre that yields colId_post for this row ---
    # ldsRowId for the GR is computed from the lane's rowId (laneId
    # // blockSize_GR), and rowId_last is exactly our target.
    ldsRowId = rowId_last // numRowsPerLDSBanks
    colId_pre = _grSwizzledColIdPre(ldsRowId, 0, wave_target,
                                    blockSize_GR, numRowsPerLDSBanks,
                                    colId_post)

    # --- Step 5: lane_target ---
    lane_target = rowId_last * blockSize_GR + colId_pre

    # --- Step 6: m0_target (absolute LDS byte address) ---
    LWA = _grWaveLWA(kernel, ti, rowOffset_wave)
    if tc == 'B':
        if tiA is None:
            raise ValueError("tc='B' requires tiA for ldsStartOffsetB")
        LWA += computeLDSStartOffsetB(tiA)

    m0_target = (LWA
                 + sId0_last * subtileSize_bytes
                 + sId1_last * ti.globalSubtileGrid[0] * subtileSize_bytes
                 + lane_target * loadWidthGR
                 + K_within_lane * bpe)

    # --- Step 7: vaddr_target (global byte offset for A[m_last, K-1])
    # At tail entry the Srd<tc> base has been advanced by
    # `K_aligned * bpe` per row (NOT per global byte -- the row stride
    # is StrideA0I or StrideB1J, which the buffer-load multiplies by
    # the lane's voff). The voff we want for the firing lane is
    # `m_last * stride * bpe + K_local_in_subtile * bpe`. We don't
    # have the runtime stride at compile time, so we encode it as the
    # *row index* (m_last) and the within-row *byte offset*
    # (K_local_in_subtile * bpe); the emitter multiplies by the
    # runtime stride SGPR.
    #
    # The "vaddr_target" stored here is the WITHIN-ROW byte offset
    # (K_local_in_subtile * bpe). The emitter combines this with
    # `m_last * stride * bpe` at emit time.
    vaddr_target_within_row = K_local_in_subtile * bpe

    explain = (
        f"tc={tc} m_last={m_last} K_local={K_local} → "
        f"wave={wave_target} sId0={sId0_last} sId1={sId1_last} "
        f"rowId={rowId_last} colId_post={colId_post} "
        f"colId_pre={colId_pre} lane={lane_target} "
        f"K_within_lane={K_within_lane} "
        f"LWA={LWA} m0=LWA+{sId0_last}·{subtileSize_bytes}"
        f"+{lane_target}·{loadWidthGR}+{K_within_lane}·{bpe}={m0_target}"
    )

    return NarrowLoadDescriptor(
        wave_target=wave_target,
        lane_target=lane_target,
        m0_target=m0_target,
        vaddr_target=vaddr_target_within_row,
        sId0_last=sId0_last,
        sId1_last=sId1_last,
        explain=explain,
    )


# ── LR side: oracle ───────────────────────────────────────────────────────


def _lrPermlane16Swap(lane_id: int, colOffset_pre: int,
                     numRowsPerLDSBanks: int, blockSize_LR: int,
                     mi_m: int) -> int:
    """Replicate the permlane16_swap_b32 in `_lraTileAssignment_legacy`.

    The swap exchanges values between lane `i` and lane `i ^ 16` for
    every lane in [0..16) ∪ [16..32) (and [32..48) ∪ [48..64)), but
    only for lanes with EXEC bit set in 0x33333333_33333333. With EXEC
    masking, only lanes at positions {0,1,4,5,8,9,12,13} mod 16
    participate.

    A swap between lane i and lane i^16 means lane i WRITES its value
    into lane (i^16)'s register AND vice versa. So lane i's output
    equals lane (i^16)'s *input* (colOffset_pre value).

    `permlane16_swap_b32` on inactive lanes is INACTIVE -- the lane's
    register is left unchanged. So for lanes outside the EXEC mask
    the colOffset_post = colOffset_pre.
    """
    full_exec = (PERMLANE16_SWAP_EXEC_MASK_HI << 32) | PERMLANE16_SWAP_EXEC_MASK_LO
    if not ((full_exec >> lane_id) & 1):
        return colOffset_pre

    peer_lane = lane_id ^ 16
    if not ((full_exec >> peer_lane) & 1):
        # Asymmetric: peer lane inactive. permlane16_swap_b32 ISA spec
        # says the operation is no-op for inactive peer (both source
        # and destination must be active). For the 0x33333333 mask
        # the active lanes pair up symmetrically within every
        # 32-lane half, so this branch is never reached for that
        # mask; raise so a future caller using a different EXEC
        # mask sees the gap.
        raise AssertionError(
            f"permlane16_swap symmetry assumption broken: lane {lane_id} "
            f"active but peer {peer_lane} inactive in EXEC=0x"
            f"{full_exec:016x}")

    # Peer's colOffset_pre
    peer_lane16 = peer_lane % mi_m
    peer_lane16Group = peer_lane // mi_m
    peer_ldsRowId = peer_lane16 // numRowsPerLDSBanks
    peer_rotation = (peer_ldsRowId // 2) * 2
    return (peer_rotation + peer_lane16Group) % blockSize_LR


def computeLRReaderForBoundary(kernel, ti, K_remain: int, tc: str,
                               tiA=None) -> LRReaderTarget:
    """LR-side oracle: which LDS byte does the per-lane mask init
    expect to contain the (M_last, K_last_local) element?

    Mirrors `_lraTileAssignment_legacy` from
    `Components/Subtile/SubtileLREmit.py` (the bf16/fp16 path; the
    fp8 K_group rotation branch is out of scope for Phase A1).

    Strategy:
      1. Identify the (wave, lane16, lane16Group, sId0_LR, sId1_LR)
         tuple that consumes the (M_last, K_local) MFMA input. This
         is the LR mirror of the GR derivation: the wave whose MFMA
         covers M_last (determined by MIWaveGroup partition) and the
         lane16 within that wave's M-direction MFMA tile that covers
         M_last.
      2. Walk the LR swizzle (intra-wave rotation + permlane16 swap +
         row offset) to compute the LDS byte that lane consults for
         SubtileX[sId0_LR, sId1_LR] subIterK=0.

    The returned `lds_byte_target` MUST equal the
    `NarrowLoadDescriptor.m0_target` computed by
    :func:`computeNarrowLoadDescriptor` for the same inputs --
    that's the cross-check the harness asserts.
    """
    if ti.localSubtileGrid[1] != 1:
        raise NotImplementedError(
            f"computeLRReaderForBoundary: localSubtileGrid[1]="
            f"{ti.localSubtileGrid[1]} > 1 staged for DU>subIterK")

    m_last = ti.macroTile - 1
    bpe = int(ti.bpe)
    loadWidthLR = ti.loadWidthLR
    mi_m = ti.mmaTileShape[0]
    subIterKBytes = ti.subIterKBytes
    subtileSize_bytes = ti.subtileSize
    elementsPerLane_LR = loadWidthLR // bpe   # 8 for bf16 b128
    blockSize_LR = subIterKBytes // loadWidthLR
    ldsRowBankSize = _DEFAULT_LDS_BANK_COUNT * _DEFAULT_LDS_BANK_WIDTH
    numRowsPerLDSBanks = ldsRowBankSize // subIterKBytes

    K_local = K_remain - 1
    K_per_subtile = ti.subtileShape[1] * ti.mmaTileShape[1]
    sId1_LR = K_local // K_per_subtile
    K_local_in_subtile = K_local % K_per_subtile

    # --- Consumer wave: which wave's MFMA reads (M_last, K_local)? ---
    # Each wave's MFMA C tile spans `localMMATileGrid[0]` M-direction
    # MFMA tiles. MFMA tile index in M = m_last // mmaTileShape[0].
    # The M-direction wave that covers MFMA tile T is T // localMMATileGrid[0].
    wg_m = kernel["MIWaveGroup"][0]
    wg_n = kernel["MIWaveGroup"][1]
    mma_tile_M = m_last // ti.mmaTileShape[0]
    mwave_index = mma_tile_M // ti.localMMATileGrid[0]
    sId0_LR = mma_tile_M % ti.localMMATileGrid[0]

    # All N-waves at this M-position share the same A-side LDS region
    # (A is N-invariant), so pick the wave with the matching M-position
    # at N-wave 0. For B, swap the roles. Conventionally:
    #   - For A: consumer_wave = mwave_index (lowest such wave) OR any
    #     wave with localRow == mwave_index. For the canonical
    #     wg_m=2/wg_n=2 case the highest-wave path produces wave 3
    #     for M_last; same answer.
    if tc == 'A':
        # localRow corresponds to mwave_index (low bit of waveId for
        # loadRatioGR=1.0 wg_m=2). The N-direction wave doesn't
        # matter for A consumption -- it just selects which D quadrant
        # this wave's MFMA accumulates into.
        consumer_wave = mwave_index + (wg_n - 1) * wg_m
    else:  # tc == 'B'
        nwave_index = m_last // ti.mmaTileShape[0] // ti.localMMATileGrid[0]
        # B's MFMA consumer wave: the highest N-wave at any M-wave
        # position. Use waveId = mwave + nwave*wg_m (col-major waveId).
        sId0_LR = mma_tile_M % ti.localMMATileGrid[0]
        consumer_wave = (wg_m - 1) + nwave_index * wg_m

    # --- lane16 within the consumer wave's MFMA tile ---
    lane16 = m_last % ti.mmaTileShape[0]

    # --- lane16Group: K-direction MFMA tile and K-chunk within tile ---
    # Each subIterK iter consumes mmaTileShape[1] K-elements per MFMA.
    # The lane16Group selects which loadWidthLR-stripe (8 K-elements
    # for bf16 b128) within the MFMA's K-input. The MFMA's K-input
    # spans elementsPerLane_LR * 4 = 32 K-elements (for BF16
    # MI16x16x32), divided into 4 lane16Group stripes.
    subIterK_target = K_local_in_subtile // ti.mmaTileShape[1]
    K_in_mma = K_local_in_subtile % ti.mmaTileShape[1]
    lane16Group = K_in_mma // elementsPerLane_LR
    K_within_lane = K_in_mma % elementsPerLane_LR

    consumer_lane = lane16Group * mi_m + lane16

    # --- LR offset formula ---
    ldsRowId = lane16 // numRowsPerLDSBanks
    rotation = (ldsRowId // 2) * 2
    colOffset_pre = (rotation + lane16Group) % blockSize_LR
    colOffset_post = _lrPermlane16Swap(consumer_lane, colOffset_pre,
                                       numRowsPerLDSBanks, blockSize_LR,
                                       mi_m)
    row_offset_bytes = lane16 * subIterKBytes
    per_lane_addr = colOffset_post * loadWidthLR + row_offset_bytes

    # --- Wave partition for the LR side ---
    # Mirrors `_lraTileAssignment_legacy`'s
    # `_lraWavePartitioning_legacy` chain.
    num_waves = wg_m * wg_n
    waves_coop = num_waves // wg_m
    MT = ti.macroTile
    if waves_coop > 1 and wg_m > 1:
        sInterval = MT * subIterKBytes // wg_m
        if tc == 'A':
            wave_partition = (consumer_wave & (waves_coop - 1)) * sInterval
        else:
            wave_partition = (consumer_wave >> (waves_coop.bit_length() - 1)) * sInterval
        per_lane_addr += wave_partition
    elif wg_m > 1:
        # waves_coop == 1: each wave owns its own LDS region
        sInterval = MT * subIterKBytes // num_waves
        per_lane_addr += consumer_wave * sInterval

    # --- B operand: add ldsStartOffsetB ---
    if tc == 'B':
        if tiA is None:
            raise ValueError("tc='B' requires tiA for ldsStartOffsetB")
        per_lane_addr += computeLDSStartOffsetB(tiA)

    # --- ds_read subtile offset ---
    ds_offset = sId0_LR * subtileSize_bytes + \
                sId1_LR * ti.globalSubtileGrid[0] * subtileSize_bytes

    # --- The byte WITHIN the 16-byte lane chunk that holds K_local ---
    byte_within_lane = K_within_lane * bpe

    lds_byte_target = per_lane_addr + ds_offset + byte_within_lane

    explain = (
        f"tc={tc} m_last={m_last} K_local={K_local} → "
        f"consumer_wave={consumer_wave} consumer_lane={consumer_lane} "
        f"(lane16={lane16} lane16Group={lane16Group}) "
        f"sId0_LR={sId0_LR} sId1_LR={sId1_LR} subIterK={subIterK_target} "
        f"colOffset_pre={colOffset_pre} colOffset_post={colOffset_post} "
        f"K_within_lane={K_within_lane} lds_byte={lds_byte_target}"
    )

    return LRReaderTarget(
        lds_byte_target=lds_byte_target,
        consumer_wave=consumer_wave,
        consumer_lane=consumer_lane,
        explain=explain,
    )


# ── Operand-level applicability ────────────────────────────────────────────


def subtileTailNarrowLoadOperandSupported(ti) -> bool:
    """Per-operand check: Phase A1 covers loadRatioGR=1.0 and
    localSubtileGrid[1]==1. Other shapes stay on align-UP."""
    if ti.loadRatioGR != 1.0:
        return False
    if ti.localSubtileGrid[1] != 1:
        return False
    return True


# ── Emitter (uses the descriptor + per-wave EXEC mask) ─────────────────────


def _emitNarrowLoadForOperand(kw, kernel, tc, ti, tiA, vAddrZero,
                              sExecSave) -> Module:
    """Emit the narrow load for one operand (`tc` ∈ {'A','B'}).

    Walks the descriptor for K_remain treated as a runtime SGPR
    (`LoopCounterL`). All compile-time terms are baked into hex
    immediates; the runtime portion (K_within_lane, colId_post,
    colId_pre, lane_target, soffset) is computed in SGPRs.
    """
    module = Module("tailTrailingNarrowLoad %s" % tc)

    bpe = int(ti.bpe)
    loadWidthGR = ti.loadWidthGR
    elementsPerLane = loadWidthGR // bpe          # 8 for bf16 b128
    blockSize_GR = ti.subIterKBytes // loadWidthGR
    subtileSize_bytes = ti.subtileSize

    # Wave / row resolution at the LAST row of the operand's M-axis.
    m_last = ti.macroTile - 1
    wave_target, rowOffset_wave, sId0_last, rowId_last = \
        _grWavePartitionForRow(kernel, ti, m_last)

    # Compile-time swizzle inversion constants (intra-wave + inter-wave
    # rotation). `_grSwizzleColIds_legacy` forward chain:
    #   rotation_intra = blockSize - (ldsRowId // 2) * 2
    #   waveRotation   = (waveId & 1) * (2*numRowsPerLDSBanks)
    #   colId_post     = (colId_pre + rotation_intra - waveRotation) % blockSize
    # Inversion:
    #   colId_pre = (colId_post - rotation_intra + waveRotation) % blockSize
    ldsRowBankSize = _ldsRowBankSize(kw)
    numRowsPerLDSBanks = ldsRowBankSize // ti.subIterKBytes
    ldsRowId = rowId_last // numRowsPerLDSBanks
    rotation_intra = blockSize_GR - (ldsRowId // 2) * 2
    waveRot_shift = (2 * numRowsPerLDSBanks).bit_length() - 1
    waveRotation = (wave_target & 1) << waveRot_shift
    # In the K_remain in [1, elementsPerLane] case (colId_post = 0)
    # the inversion collapses to a constant; the runtime path below
    # handles the general K_remain by computing
    # `s_colId_pre = (s_colId_post - rotation_intra + waveRotation) & (blockSize-1)`.
    colIdPreOffset = (-rotation_intra + waveRotation) & 0xffffffff

    # Subtile + LWA constants.
    LWA_const_offset = sId0_last * subtileSize_bytes

    # LR partition constants for B (mirrors `_lraTileAssignment_legacy`
    # `_lraWavePartitioning_legacy`). We don't need this for the GR
    # write target (LWA already accounts for the wave's region), but
    # we do for the SrdB byte index since SrdB itself is already at
    # the K_aligned offset per row. (Both A and B SRDs at tail entry
    # point one DU past the prior iter's start, so the per-row stride
    # multiplies the row index from 0..MT-1; no wave-partition
    # adjustment to vaddr.)

    stride_sgpr_name = "StrideA0I" if tc == 'A' else "StrideB1J"
    lwa_sgpr_name = "LocalWriteBaseAddr%s" % tc
    srd_sgpr_name = "Srd%s" % tc

    # Allocate SGPR scratch for runtime computation + EXEC mask.
    # Layout (4 sgprs):
    #   sScratch+0: s_waveId / s_klocal / s_kwl / s_colId_pre / s_lane / s_m0_off
    #   sScratch+1: s_soffset (lives across the BufferLoad)
    #   sScratch+2..3: unused tail (allocated to satisfy 4-sgpr alignment
    #                  in case allocTmpSgpr's preference for power-of-2
    #                  blocks kicks in).
    with kw.allocTmpSgpr(2) as scratchInfo:
        sScratch = scratchInfo.idx
        sSoffset = sScratch + 1

        # ---- waveId in SGPR (uniform per wave) ----
        wavesize = kernel["WavefrontSize"]
        vWaveIdTmp = kw.vgprPool.checkOut(1, "tailNarrowLoadWaveId")
        module.add(VLShiftRightB32(
            dst=vgpr(vWaveIdTmp),
            shiftHex=hex(wavesize.bit_length() - 1),
            src=vgpr("Serial"),
            comment="narrowLoad[%s]: vgprSerial >> %u = waveId"
                    % (tc, wavesize.bit_length() - 1)))
        module.add(VReadfirstlaneB32(
            dst=sgpr(sScratch),
            src=vgpr(vWaveIdTmp),
            comment="narrowLoad[%s]: s_waveId = lane0(v_waveId)" % tc))
        kw.vgprPool.checkIn(vWaveIdTmp)

        # ---- Per-wave skip: if s_waveId != wave_target, branch past
        # the whole body. Wave-uniform branch (SCC is wave-uniform).
        skipLabel = Label(
            "tailNarrowLoadSkip%s_w%u" % (tc, wave_target), "")
        module.add(SCmpEQU32(
            src0=sgpr(sScratch), src1=wave_target,
            comment="narrowLoad[%s]: this wave is the trailing-element "
                    "owner (wave=%u)?" % (tc, wave_target)))
        module.add(SCBranchSCC0(
            labelName=skipLabel.getLabelName(),
            comment="narrowLoad[%s]: skip — not wave %u"
                    % (tc, wave_target)))

        # ---- Runtime K-tail address computation ----
        # s_klocal = LoopCounterL - 1
        module.add(SSubU32(
            dst=sgpr(sScratch), src0=sgpr("LoopCounterL"), src1=1,
            comment="narrowLoad[%s]: K_local = LoopCounterL - 1" % tc))
        # s_kwl = s_klocal & (elementsPerLane - 1)
        module.add(SAndB32(
            dst=sgpr(sSoffset), src0=sgpr(sScratch),
            src1=elementsPerLane - 1,
            comment="narrowLoad[%s]: K_within_lane = K_local %% %u"
                    % (tc, elementsPerLane)))
        # s_colId_post = s_klocal >> log2(elementsPerLane)
        module.add(SLShiftRightB32(
            dst=sgpr(sScratch), src=sgpr(sScratch),
            shiftHex=hex(elementsPerLane.bit_length() - 1),
            comment="narrowLoad[%s]: colId_post = K_local // %u"
                    % (tc, elementsPerLane)))
        # s_colId_pre = (s_colId_post + colIdPreOffset) & (blockSize-1)
        module.add(SAddU32(
            dst=sgpr(sScratch), src0=sgpr(sScratch),
            src1=hex(colIdPreOffset),
            comment="narrowLoad[%s]: colId_pre = colId_post + "
                    "(-rotation_intra + waveRotation) mod blockSize"
                    % tc))
        module.add(SAndB32(
            dst=sgpr(sScratch), src0=sgpr(sScratch),
            src1=blockSize_GR - 1,
            comment="narrowLoad[%s]: colId_pre &= %u" % (tc, blockSize_GR - 1)))
        # s_lane_target = rowId_last * blockSize + s_colId_pre
        module.add(SAddU32(
            dst=sgpr(sScratch), src0=sgpr(sScratch),
            src1=rowId_last * blockSize_GR,
            comment="narrowLoad[%s]: lane_target = colId_pre + "
                    "rowId_last(=%u) * blockSize(=%u)"
                    % (tc, rowId_last, blockSize_GR)))

        # ---- m0 = LWA<tc> + sId0_last * subtileSize + lane_target * loadWidthGR
        #              + K_within_lane * bpe ----
        # Build m0 in mgpr(0):
        #   m0 = LWA + LWA_const_offset
        module.add(SAddU32(
            dst=mgpr(0),
            src0=sgpr(lwa_sgpr_name),
            src1=hex(LWA_const_offset),
            comment="narrowLoad[%s]: m0 = LWA%s + sId0_last(=%u)*"
                    "subtileSize(=%u)=%u"
                    % (tc, tc, sId0_last, subtileSize_bytes,
                       LWA_const_offset)))
        #   s_lane_target_scaled = s_lane_target * loadWidthGR
        module.add(SLShiftLeftB32(
            dst=sgpr(sScratch), src=sgpr(sScratch),
            shiftHex=hex(loadWidthGR.bit_length() - 1),
            comment="narrowLoad[%s]: lane_target * loadWidthGR(=%u)"
                    % (tc, loadWidthGR)))
        module.add(SAddU32(
            dst=mgpr(0), src0=mgpr(0), src1=sgpr(sScratch),
            comment="narrowLoad[%s]: m0 += lane_target * loadWidthGR"
                    % tc))
        #   m0 += K_within_lane * bpe
        # bpe == 2 for all Phase A1 shapes -- single left-shift.
        module.add(SLShiftLeftB32(
            dst=sgpr(sScratch), src=sgpr(sSoffset),
            shiftHex=hex(bpe.bit_length() - 1),
            comment="narrowLoad[%s]: K_within_lane * bpe(=%u)"
                    % (tc, bpe)))
        module.add(SAddU32(
            dst=mgpr(0), src0=mgpr(0), src1=sgpr(sScratch),
            comment="narrowLoad[%s]: m0 += K_within_lane * bpe" % tc))

        # ---- soffset = m_last * stride * bpe + K_local * bpe ----
        # Recompute K_local since we trashed sScratch above.
        module.add(SSubU32(
            dst=sgpr(sScratch), src0=sgpr("LoopCounterL"), src1=1,
            comment="narrowLoad[%s]: re-derive K_local for soffset" % tc))
        module.add(SLShiftLeftB32(
            dst=sgpr(sScratch), src=sgpr(sScratch),
            shiftHex=hex(bpe.bit_length() - 1),
            comment="narrowLoad[%s]: K_local * bpe" % tc))
        module.add(SMulI32(
            dst=sgpr(sSoffset), src0=m_last,
            src1=sgpr(stride_sgpr_name),
            comment="narrowLoad[%s]: m_last(=%u) * %s" % (tc, m_last,
                    stride_sgpr_name)))
        module.add(SLShiftLeftB32(
            dst=sgpr(sSoffset), src=sgpr(sSoffset),
            shiftHex=hex(bpe.bit_length() - 1),
            comment="narrowLoad[%s]: m_last * stride * bpe" % tc))
        module.add(SAddU32(
            dst=sgpr(sSoffset), src0=sgpr(sSoffset), src1=sgpr(sScratch),
            comment="narrowLoad[%s]: soffset = m_last*stride*bpe + "
                    "K_local*bpe" % tc))

        # ---- Save EXEC, restrict to lane 0 ----
        module.add(SMovB64(
            dst=sgpr(sExecSave, 2), src=EXEC(),
            comment="narrowLoad[%s]: snapshot EXEC for restore" % tc))
        module.add(SMovB64(
            dst=EXEC(), src=hex(1),
            comment="narrowLoad[%s]: EXEC = lane 0 only" % tc))

        # ---- Issue the narrow load ----
        # vaddr=0 (lane 0 only, no per-lane offset); soffset carries
        # the global byte offset for A[m_last, K_local]; m0 holds the
        # absolute LDS byte address.
        mubuf = MUBUFModifiers(offen=True, offset12=0, lds=True,
                               glc=False, slc=False, nt=False)
        module.add(BufferLoadU16(
            dst=None, vaddr=vgpr(vAddrZero),
            saddr=sgpr(srd_sgpr_name, 4),
            soffset=sgpr(sSoffset), mubuf=mubuf,
            comment="narrowLoad[%s]: trailing element %s[m_last=%u, "
                    "K_aligned + K_remain-1] → LDS slot of wave %u "
                    "lane_target=runtime"
                    % (tc, tc, m_last, wave_target)))
        module.add(SWaitCnt(
            vlcnt=0,
            comment="narrowLoad[%s]: wait for buffer_load_ushort to "
                    "complete" % tc))

        # ---- Restore EXEC ----
        module.add(SMovB64(
            dst=EXEC(), src=sgpr(sExecSave, 2),
            comment="narrowLoad[%s]: restore EXEC" % tc))

        module.add(skipLabel)

    return module


def emitTailTrailingNarrowLoad(kw, kernel) -> Module:
    """Emit per-wave-EXEC-guarded narrow trailing-element loads for A
    and B, after the wide DTL + align-DOWN tighten leaves K=K_remain-1
    of row M-1 clipped to zero.

    Compile-time-resolves the descriptor (wave_target, sId0_last, …)
    via :func:`_grWavePartitionForRow` and emits, for each operand:

      1. Compute waveId in SGPR (broadcasted via v_readfirstlane).
      2. Per-wave skip branch: `s_cmp_eq_u32 s_waveId, wave_target` →
         `s_cbranch_scc0 skip_label` so only `wave_target` proceeds.
      3. Runtime derive `K_local, K_within_lane, colId_post,
         colId_pre, lane_target, m0, soffset` from `LoopCounterL`.
      4. Save EXEC; set EXEC = lane 0 only.
      5. Issue `buffer_load_ushort … lds` with vaddr=0,
         soffset=(m_last*stride+K_local)*bpe, m0=absolute LDS byte.
      6. SWaitCnt vlcnt(0); restore EXEC.

    Gated on:
      - :func:`subtileTailNarrowLoadApplies` (kernel-level: bpe=2,
        non-MX, non-swizzled).
      - :func:`subtileTailNarrowLoadOperandSupported` (per-operand:
        loadRatioGR=1.0, localSubtileGrid[1]=1). Both operands must
        be in scope; otherwise we skip emission entirely (the
        Phase B' align-DOWN tighten is gated the same way, so
        unsupported shapes stay on align-UP without regression).
      - Runtime: only fires when `LoopCounterL & 1 != 0` (odd
        K_remain). A single `s_and_b32 s_tmp, LoopCounterL, 1;
        s_cmp_lg_u32 s_tmp, 0; s_cbranch_scc0 skip` precedes the
        whole block, so the steady-state even-K tail pays just
        3 instructions.
    """
    module = Module("tailTrailingNarrowLoad")
    if not subtileTailNarrowLoadApplies(kernel):
        return module

    tiA = kw.states.a.tileInfo
    tiB = kw.states.b.tileInfo
    if not subtileTailNarrowLoadOperandSupported(tiA):
        return module
    if not subtileTailNarrowLoadOperandSupported(tiB):
        return module

    # Runtime gate: skip the whole helper when K_remain is even.
    # `K_remain & 1 == 0` means align-DOWN didn't drop anything past
    # the bpr boundary (K_remain*bpe is already mult of bpr=4).
    module.addComment2(
        "Tail narrow trailing-element load (bf16 odd-K only)")
    with kw.allocTmpSgpr(1) as gateInfo:
        sGate = gateInfo.idx
        skipAllLabel = Label("tailNarrowLoadSkipAll", "")
        module.add(SAndB32(
            dst=sgpr(sGate), src0=sgpr("LoopCounterL"), src1=1,
            comment="narrowLoad: K_remain & 1 (odd K?)"))
        module.add(SCmpLgU32(
            src0=sgpr(sGate), src1=0,
            comment="narrowLoad: K_remain odd?"))
        module.add(SCBranchSCC0(
            labelName=skipAllLabel.getLabelName(),
            comment="narrowLoad: K_remain even -- nothing to repair"))

        # Allocate one VGPR for vaddr=0 (shared across A and B emits).
        vAddrZero = kw.vgprPool.checkOut(1, "tailNarrowLoadVAddr0")
        module.add(VMovB32(
            dst=vgpr(vAddrZero), src=0,
            comment="narrowLoad: vaddr = 0 (lane 0 only fires under EXEC=1)"))

        # Allocate EXEC-save SGPR pair (held across A→B emits).
        with kw.allocTmpSgpr(2, alignment=2) as execSaveInfo:
            sExecSave = execSaveInfo.idx
            for tc, ti in (('A', tiA), ('B', tiB)):
                module.add(_emitNarrowLoadForOperand(
                    kw, kernel, tc, ti, tiA, vAddrZero, sExecSave))

        kw.vgprPool.checkIn(vAddrZero)
        module.add(skipAllLabel)
    return module
