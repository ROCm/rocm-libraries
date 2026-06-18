"""Golden tests for the InstructionScheduler PLACEMENT OUTPUT.

The dep-path golden tests in test_SubtileBasedSchedulerRef.py lock the
LogicalScheduler topology (the before-dep chains rendered by
print_emit_dep_order). They do NOT exercise instructionSchedule(), so they
cannot see how the slot placer actually orders instructions between MFMAs.

These tests close that gap: they run the full pipeline
(emit -> allocVgprTiles -> populate_instructions -> instructionSchedule) and
lock the per-slot ordered placement that the instruction scheduler emits, for
representative multi-DU MXFP8 and non-multi-DU configs.

Each placed instruction is rendered as a compact, register-agnostic token tied
to the MFMA interval it lands in ("[iNN] kind"), where NN counts the MFMAs
emitted so far in that subIterK. This captures exactly what the multi-DU
scheduling rules affect:
  - wait_gr packing position (forward in multi-DU today vs late in non-multi-DU,
    via reverse = (not multiDU) and hasWaitGR),
  - wait_gr+barrier (sync) contiguity,
  - post-barrier GR (m0 / buffer_load) deferral past the post-barrier ds_reads,
  - the vmcnt the post-pass assigns to a wait_gr (a function of how many
    buffer_loads were placed ahead of it).

The wait_gr vmcnt is included in the token so a change in buffer-load packing
ahead of the wait is visible.
"""
import pytest

from Tensile.Tests.unit.test_SubtileBasedLogicalScheduler import (
    create_kernel,
    make_cfg_256x256_fp4,
    make_cfg_bf16,
    make_writer_and_tileinfos,
)
from Tensile.Components.Subtile.LogicalScheduler import LogicalScheduler
from Tensile.Components.Subtile.InstructionScheduler import instructionSchedule
from rocisa.instruction import (
    SWaitCnt, SBarrier, MFMAInstruction, MXMFMAInstruction,
    LocalReadInstruction, GlobalReadInstruction, CommonInstruction, SAddU64,
)


def _is_output_kind(inst):
    """Classify a placed instruction into a stable, register-agnostic token.

    The interesting kinds (lr / gr / m0 / wait_gr / wait_lr / sync) are exactly
    the instruction classes the scheduling rules act on; everything else falls
    back to its class name so nothing in the placement is hidden.
    """
    if isinstance(inst, (MFMAInstruction, MXMFMAInstruction)):
        return 'MFMA'
    if isinstance(inst, SWaitCnt):
        # wait_gr is the only vmcnt-bearing wait on this path (vlcnt != -1);
        # wait_lr uses vlcnt == -1. Mirrors InstructionScheduler._isWaitGr.
        return f'wait_gr(vmcnt={inst.vlcnt})' if inst.vlcnt != -1 else 'wait_lr'
    if isinstance(inst, SBarrier):
        return 'sync'
    if isinstance(inst, LocalReadInstruction):
        return 'lr'
    if isinstance(inst, GlobalReadInstruction):
        return 'gr'
    if (isinstance(inst, CommonInstruction) and hasattr(inst, 'dst')
            and hasattr(inst.dst, 'regType') and inst.dst.regType == 'm'):
        return 'm0'
    if isinstance(inst, SAddU64):
        return 'gr_inc'
    return type(inst).__name__


def _render_instruction_schedule_output(make_cfg, kernel, fp4):
    """Full-pipeline render of the instruction-scheduler placement output.

    Returns one block per (partition, subIterK) that contains MFMAs, listing
    each placed non-MFMA instruction as "[iNN] kind" where NN is the count of
    MFMAs emitted so far in that subIterK (i.e. the MFMA interval).
    """
    writer, tiA, tiB, scaleTiA, scaleTiB, dTileInfo = make_writer_and_tileinfos(
        kernel, fp4=fp4)
    cfg = make_cfg()
    sched = LogicalScheduler(cfg)
    sched.emit()
    sched.allocVgprTiles(writer, tiA, tiB,
                         scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB)
    try:
        sched.populate_instructions(
            writer, kernel,
            tileInfoA=tiA, tileInfoB=tiB, dtileInfo=dTileInfo,
            scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB)
        multiDU = sched._is_multi_du()
        out = [f"multiDU={multiDU}"]
        for pi, partition_emitted in enumerate(sched._emitted_per_unroll[0]):
            for k, em_list in enumerate(partition_emitted):
                if not any(em.opType == 'mfma' for em in em_list):
                    continue
                scheduled = instructionSchedule(em_list, multiDU=multiDU)
                out.append(f"P{pi} subIterK={k}:")
                mfma = 0
                for inst in scheduled.flatitems():
                    kind = _is_output_kind(inst)
                    if kind == 'MFMA':
                        mfma += 1
                        continue
                    out.append(f"  [i{mfma:02d}] {kind}")
        return "\n".join(out) + "\n"
    finally:
        sched.deallocVgprTiles(writer)


IS_OUTPUT_MXFP8_MULTI_DU_1x1 = """\
multiDU=True
P0 subIterK=0:
  [i00] wait_lr
  [i01] lr
  [i01] m0
  [i02] lr
  [i02] gr
  [i03] lr
  [i03] m0
  [i04] lr
  [i05] lr
  [i06] lr
  [i07] lr
  [i08] lr
  [i09] lr
  [i10] lr
  [i11] lr
  [i12] lr
  [i13] lr
  [i14] lr
  [i14] gr
  [i15] lr
  [i15] m0
  [i16] lr
  [i17] lr
  [i18] lr
  [i19] lr
  [i20] lr
  [i26] gr
  [i27] m0
  [i38] gr
  [i39] m0
  [i50] gr
P0 subIterK=1:
  [i00] wait_lr
  [i32] wait_gr(vmcnt=0)
  [i33] sync
  [i33] TextBlock
  [i34] VXorB32
  [i34] VXorB32
  [i35] TextBlock
  [i35] VXorB32
  [i36] VXorB32
  [i36] lr
  [i37] lr
  [i38] lr
  [i39] lr
  [i40] lr
  [i41] lr
  [i42] lr
  [i43] lr
  [i44] lr
  [i45] lr
  [i46] lr
  [i47] lr
  [i48] lr
  [i49] lr
  [i50] lr
  [i51] lr
  [i52] lr
  [i53] lr
  [i54] lr
  [i55] lr
  [i55] m0
  [i56] gr
  [i57] m0
  [i58] gr
  [i59] m0
  [i60] gr
  [i60] SAddU32
  [i61] SAddCU32
  [i61] TextBlock
  [i62] SXorB32
  [i62] m0
  [i63] gr
P0 subIterK=2:
  [i00] wait_lr
  [i24] wait_gr(vmcnt=0)
  [i25] sync
  [i25] lr
  [i26] lr
  [i27] lr
  [i28] lr
  [i29] lr
  [i30] lr
  [i31] lr
  [i32] lr
  [i33] lr
  [i34] lr
  [i35] lr
  [i36] lr
  [i37] lr
  [i38] lr
  [i39] lr
  [i40] lr
  [i40] m0
  [i41] gr
  [i42] m0
  [i43] gr
  [i44] m0
  [i45] gr
  [i46] m0
  [i47] gr
  [i48] m0
  [i49] gr
  [i50] m0
  [i51] gr
  [i52] m0
  [i53] gr
  [i54] SAddU32
  [i54] SAddCU32
  [i55] TextBlock
  [i55] SXorB32
  [i56] TextBlock
  [i56] m0
  [i57] gr
  [i57] TextBlock
  [i58] SAddU32
  [i58] SAddCU32
  [i59] TextBlock
  [i59] SXorB32
  [i60] TextBlock
  [i60] m0
  [i61] gr
  [i61] TextBlock
  [i62] SAddU32
  [i62] SAddCU32
  [i63] TextBlock
  [i63] SXorB32
P0 subIterK=3:
  [i00] wait_lr
  [i01] wait_gr(vmcnt=0)
  [i01] sync
  [i01] TextBlock
  [i02] VXorB32
  [i02] VXorB32
  [i03] TextBlock
  [i03] VXorB32
  [i04] VXorB32
  [i04] TextBlock
  [i05] VXorB32
  [i05] TextBlock
  [i06] VXorB32
  [i06] lr
  [i07] lr
  [i08] lr
  [i09] lr
  [i10] lr
  [i11] lr
  [i12] lr
  [i13] lr
  [i14] lr
  [i15] lr
  [i16] lr
  [i17] lr
  [i18] lr
  [i19] lr
  [i20] lr
  [i21] lr
  [i22] lr
  [i23] lr
  [i24] lr
  [i25] lr
  [i26] lr
  [i27] lr
  [i28] lr
  [i29] lr
  [i29] m0
  [i30] gr
  [i31] m0
  [i32] gr
  [i33] m0
  [i34] gr
  [i35] m0
  [i36] gr
  [i37] m0
  [i38] gr
  [i39] m0
  [i40] gr
  [i41] m0
  [i42] gr
  [i43] m0
  [i44] gr
  [i44] SAddU32
  [i45] SAddCU32
  [i45] TextBlock
  [i46] SXorB32
  [i46] m0
  [i47] gr
  [i48] m0
  [i49] gr
  [i50] m0
  [i51] gr
  [i52] m0
  [i53] gr
  [i54] m0
  [i55] gr
  [i56] m0
  [i57] gr
  [i58] m0
  [i59] gr
  [i60] m0
  [i61] gr
  [i62] SAddU32
  [i62] SAddCU32
  [i63] TextBlock
  [i63] SXorB32
"""


def test_mxfp8_multi_du_1x1():
    """Multi-DU MXFP8 (numUnroll=2), 1x1 partition."""
    actual = _render_instruction_schedule_output(
        lambda: make_cfg_256x256_fp4(grSA_k_gran=2, grSB_k_gran=2, pgr=1), create_kernel(256, 256, fp4=True), fp4=True)
    assert actual == IS_OUTPUT_MXFP8_MULTI_DU_1x1, (
        "InstructionScheduler placement output mismatch.\n"
        "--- Expected ---\n" + IS_OUTPUT_MXFP8_MULTI_DU_1x1 + "\n--- Actual ---\n" + actual)


IS_OUTPUT_MXFP8_MULTI_DU_PARTN = """\
multiDU=True
P0 subIterK=0:
  [i00] wait_lr
  [i01] lr
  [i01] m0
  [i02] lr
  [i02] gr
  [i03] lr
  [i03] m0
  [i04] lr
  [i05] lr
  [i06] lr
  [i07] lr
  [i08] lr
  [i09] lr
  [i10] lr
  [i11] lr
  [i12] lr
  [i13] lr
  [i13] gr
  [i14] lr
  [i14] m0
  [i15] lr
  [i16] lr
  [i17] lr
  [i18] lr
  [i24] gr
  [i25] m0
  [i35] gr
P0 subIterK=1:
  [i00] wait_lr
  [i01] m0
  [i01] TextBlock
  [i01] VXorB32
  [i02] gr
  [i02] VXorB32
  [i02] TextBlock
  [i03] m0
  [i03] VXorB32
  [i03] VXorB32
  [i04] lr
  [i05] lr
  [i06] lr
  [i07] lr
  [i08] lr
  [i09] lr
  [i10] lr
  [i11] lr
  [i12] lr
  [i13] lr
  [i14] lr
  [i15] lr
  [i16] lr
  [i17] gr
  [i17] lr
  [i18] m0
  [i18] lr
  [i19] lr
  [i20] lr
  [i32] gr
P0 subIterK=2:
  [i00] wait_lr
  [i01] lr
  [i01] m0
  [i02] lr
  [i02] gr
  [i02] SAddU32
  [i03] lr
  [i03] SAddCU32
  [i03] TextBlock
  [i04] lr
  [i04] SXorB32
  [i04] m0
  [i05] lr
  [i06] lr
  [i07] lr
  [i08] lr
  [i09] lr
  [i10] lr
  [i11] lr
  [i12] lr
  [i13] lr
  [i13] gr
  [i14] lr
  [i14] m0
  [i15] lr
  [i24] gr
  [i25] m0
  [i35] gr
P0 subIterK=3:
  [i00] wait_lr
  [i01] m0
  [i01] TextBlock
  [i01] VXorB32
  [i02] gr
  [i02] VXorB32
  [i02] lr
  [i03] m0
  [i03] lr
  [i17] gr
  [i18] m0
  [i32] gr
P1 subIterK=0:
  [i00] wait_lr
  [i09] wait_gr(vmcnt=0)
  [i10] sync
  [i10] lr
  [i11] lr
  [i12] TextBlock
  [i12] m0
  [i13] gr
  [i13] TextBlock
  [i14] SAddU32
  [i14] SAddCU32
  [i15] TextBlock
  [i15] SXorB32
P1 subIterK=1:
  [i00] wait_lr
  [i07] wait_gr(vmcnt=0)
  [i07] sync
  [i08] TextBlock
  [i08] VXorB32
  [i09] VXorB32
  [i09] lr
  [i10] lr
  [i11] lr
  [i12] TextBlock
  [i12] m0
  [i13] gr
  [i13] TextBlock
  [i14] SAddU32
  [i14] SAddCU32
  [i15] TextBlock
  [i15] SXorB32
P1 subIterK=2:
  [i00] wait_lr
  [i08] wait_gr(vmcnt=0)
  [i09] sync
  [i09] lr
  [i10] lr
  [i10] m0
  [i11] gr
  [i12] m0
  [i13] gr
  [i14] SAddU32
  [i14] SAddCU32
  [i15] TextBlock
  [i15] SXorB32
P1 subIterK=3:
  [i00] wait_lr
  [i01] wait_gr(vmcnt=0)
  [i01] sync
  [i01] TextBlock
  [i01] VXorB32
  [i01] VXorB32
  [i01] TextBlock
  [i01] VXorB32
  [i01] VXorB32
  [i01] TextBlock
  [i01] VXorB32
  [i01] TextBlock
  [i01] VXorB32
  [i01] lr
  [i01] lr
  [i01] lr
  [i01] lr
  [i01] lr
  [i01] lr
  [i01] lr
  [i01] lr
  [i01] lr
  [i01] lr
  [i01] lr
  [i01] lr
  [i01] lr
  [i01] lr
  [i01] lr
  [i01] lr
  [i01] lr
  [i01] lr
  [i01] lr
  [i01] lr
  [i01] lr
  [i01] m0
  [i01] gr
  [i01] m0
  [i01] gr
  [i01] m0
  [i01] gr
  [i01] m0
  [i01] gr
  [i01] m0
  [i01] gr
  [i01] m0
  [i01] gr
  [i01] m0
  [i01] gr
  [i01] m0
  [i01] gr
  [i01] SAddU32
  [i01] SAddCU32
  [i01] TextBlock
  [i01] SXorB32
  [i01] m0
  [i01] gr
  [i01] m0
  [i01] gr
  [i02] m0
  [i03] gr
  [i04] m0
  [i05] gr
  [i06] m0
  [i07] gr
  [i08] m0
  [i09] gr
  [i10] m0
  [i11] gr
  [i12] m0
  [i13] gr
  [i14] SAddU32
  [i14] SAddCU32
  [i15] TextBlock
  [i15] SXorB32
"""


def test_mxfp8_multi_du_partition_remainder():
    """Multi-DU MXFP8 with an uneven N partition (remainder-last split)."""
    actual = _render_instruction_schedule_output(
        lambda: make_cfg_256x256_fp4(grSA_k_gran=2, grSB_k_gran=2, pgr=1, partSizeN=6), create_kernel(256, 256, fp4=True), fp4=True)
    assert actual == IS_OUTPUT_MXFP8_MULTI_DU_PARTN, (
        "InstructionScheduler placement output mismatch.\n"
        "--- Expected ---\n" + IS_OUTPUT_MXFP8_MULTI_DU_PARTN + "\n--- Actual ---\n" + actual)


IS_OUTPUT_MXFP8_SINGLE_DU_1x1 = """\
multiDU=False
P0 subIterK=0:
  [i00] wait_lr
  [i01] lr
  [i01] SAddU32
  [i01] SAddCU32
  [i02] lr
  [i02] TextBlock
  [i02] SXorB32
  [i03] lr
  [i03] m0
  [i04] lr
  [i04] gr
  [i05] lr
  [i05] m0
  [i06] lr
  [i07] lr
  [i07] gr
  [i08] lr
  [i08] m0
  [i09] lr
  [i10] lr
  [i10] gr
  [i11] lr
  [i11] m0
  [i12] lr
  [i13] lr
  [i13] gr
  [i14] lr
  [i14] m0
  [i15] lr
  [i16] lr
  [i16] gr
  [i17] m0
  [i19] gr
  [i20] m0
  [i22] gr
  [i23] m0
  [i25] gr
  [i25] SAddU32
  [i26] SAddCU32
  [i26] TextBlock
  [i27] SXorB32
  [i27] m0
  [i28] gr
  [i29] m0
  [i31] gr
  [i32] m0
  [i34] gr
  [i35] m0
  [i37] gr
  [i38] m0
  [i40] gr
  [i41] m0
  [i43] gr
  [i44] m0
  [i46] gr
  [i47] m0
  [i49] gr
  [i49] TextBlock
  [i50] SAddU32
  [i50] SAddCU32
  [i51] TextBlock
  [i51] SXorB32
  [i52] TextBlock
  [i52] m0
  [i53] gr
  [i53] TextBlock
  [i54] SAddU32
  [i54] SAddCU32
  [i55] TextBlock
  [i55] SXorB32
  [i56] TextBlock
  [i56] m0
  [i57] gr
P0 subIterK=1:
  [i00] wait_lr
  [i31] wait_gr(vmcnt=0)
  [i32] sync
  [i32] TextBlock
  [i33] VXorB32
  [i33] VXorB32
  [i34] TextBlock
  [i34] VXorB32
  [i35] VXorB32
  [i35] TextBlock
  [i36] VXorB32
  [i36] TextBlock
  [i37] VXorB32
  [i37] lr
  [i38] lr
  [i39] lr
  [i40] lr
  [i41] lr
  [i42] lr
  [i43] lr
  [i44] lr
  [i45] lr
  [i46] lr
  [i47] lr
  [i48] lr
  [i49] lr
  [i50] lr
  [i51] lr
  [i52] lr
  [i53] lr
  [i54] lr
  [i55] lr
  [i56] lr
  [i57] lr
  [i58] lr
  [i59] lr
  [i60] lr
"""


def test_mxfp8_single_du_1x1():
    """Single-DU MXFP8 (numUnroll=1) — a non-multi-DU path."""
    actual = _render_instruction_schedule_output(
        lambda: make_cfg_256x256_fp4(pgr=1), create_kernel(256, 256, fp4=True), fp4=True)
    assert actual == IS_OUTPUT_MXFP8_SINGLE_DU_1x1, (
        "InstructionScheduler placement output mismatch.\n"
        "--- Expected ---\n" + IS_OUTPUT_MXFP8_SINGLE_DU_1x1 + "\n--- Actual ---\n" + actual)


IS_OUTPUT_BF16_256x256_1x1 = """\
multiDU=False
P0 subIterK=0:
  [i00] wait_lr
  [i01] lr
  [i02] lr
  [i03] lr
  [i04] lr
  [i05] lr
  [i06] lr
  [i07] lr
  [i08] lr
  [i09] lr
  [i10] lr
  [i11] lr
  [i12] lr
  [i13] lr
  [i14] lr
  [i15] lr
  [i16] lr
  [i20] wait_lr
  [i20] sync
  [i21] SAddU32
  [i21] SAddCU32
  [i22] TextBlock
  [i22] SXorB32
  [i23] m0
  [i24] gr
  [i25] m0
  [i28] gr
  [i29] m0
  [i33] gr
  [i34] m0
  [i37] gr
  [i38] m0
  [i42] gr
  [i43] m0
  [i46] gr
  [i47] m0
  [i51] gr
  [i52] m0
  [i55] gr
P0 subIterK=1:
  [i00] wait_lr
  [i01] SAddU32
  [i01] SAddCU32
  [i02] TextBlock
  [i02] SXorB32
  [i03] m0
  [i04] gr
  [i05] m0
  [i11] gr
  [i12] m0
  [i18] gr
  [i19] m0
  [i25] gr
  [i26] m0
  [i32] gr
  [i33] m0
  [i39] gr
  [i40] m0
  [i41] wait_gr(vmcnt=14)
  [i42] sync
  [i42] TextBlock
  [i43] VXorB32
  [i43] VXorB32
  [i44] TextBlock
  [i44] VXorB32
  [i45] VXorB32
  [i45] lr
  [i46] gr
  [i46] lr
  [i47] m0
  [i47] lr
  [i48] lr
  [i49] lr
  [i50] lr
  [i51] lr
  [i52] lr
  [i53] gr
  [i53] lr
  [i54] lr
  [i55] lr
  [i56] lr
  [i57] lr
  [i58] lr
  [i59] lr
  [i60] lr
"""


def test_bf16_256x256_1x1():
    """BF16 256x256, 1x1 partition — a non-multi-DU path."""
    actual = _render_instruction_schedule_output(
        lambda: make_cfg_bf16(256, 256), create_kernel(256, 256, fp4=False), fp4=False)
    assert actual == IS_OUTPUT_BF16_256x256_1x1, (
        "InstructionScheduler placement output mismatch.\n"
        "--- Expected ---\n" + IS_OUTPUT_BF16_256x256_1x1 + "\n--- Actual ---\n" + actual)


