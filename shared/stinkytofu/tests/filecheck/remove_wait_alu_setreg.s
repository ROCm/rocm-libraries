# RUN: %stinkytofu-opt --arch gfx1250 %s --RemoveWaitAluPass --emit-asm --from-label test_start --to-label test_end
#
# RemoveWaitAluPass — setreg(SCHED_MODE) stripping path, exercised via raw
# assembly because the stir parser does not accept hwreg() destinations.
#
# Strip only s_setreg* targeting the DEP_MODE field of SCHED_MODE:
#   hwreg(26, offset=0, size=2)  — DEP_MODE {0,1,2} = expert scheduling mode
# Other SCHED_MODE bit-fields must NOT be stripped:
#   hwreg(26, offset=4, size=1)  — DISABLE_XDL_ARB_STALL (back-to-back WMMA hint)
# Setregs targeting a different hwreg id are preserved unchanged.
#
# All s_wait_alu instructions in this region carry only GPR hazard fields,
# so every one is deleted by the pass.
#
# CHECK-LABEL: test_start:
# DEP_MODE setregs (hwreg(26,0,2)) and all s_wait_alu are stripped.
# CHECK-NOT: hwreg(26,0,2)
# CHECK-NOT: s_wait_alu
# Source ordering of the VALUs plus the kept setregs is preserved.
# CHECK: v_add_f32
# CHECK-NEXT: s_setreg_IMM32_b32 hwreg(26,4,1)
# CHECK-NEXT: v_mul_f32
# CHECK-NEXT: s_setreg_IMM32_b32 hwreg(1,0,2)
# CHECK-NEXT: v_sub_f32
# CHECK-LABEL: test_end:

test_start:
    s_setreg_IMM32_b32 hwreg(26,0,2), 2          // DEP_MODE = mode2 enable - strip
    s_wait_alu depctr_va_vdst(0)                 // GPR field only - strip
    v_add_f32 v0, v1, v2
    s_setreg_IMM32_b32 hwreg(26,4,1), 1          // DISABLE_XDL_ARB_STALL - KEEP (WMMA perf hint)
    s_wait_alu depctr_vm_vsrc(0)                 // GPR field only - strip
    v_mul_f32 v3, v0, v4
    s_setreg_IMM32_b32 hwreg(1,0,2), 3           // MODE register (id=1), unrelated - keep
    s_setreg_IMM32_b32 hwreg(26,0,2), 1          // DEP_MODE = mode1 enable - strip
    s_wait_alu depctr_va_sdst(0) | depctr_va_vdst(0)  // GPR fields - strip
    v_sub_f32 v5, v3, v6
    s_setreg_IMM32_b32 hwreg(26,0,2), 0          // DEP_MODE = mode0 disable - strip
test_end:
