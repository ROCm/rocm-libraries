# RUN: %stinkytofu-opt --arch gfx1250 %s --CFGBuilderPass --RemoveWaitAluPass --InsertWaitAluPass --from-label label_LoopBeginL --to-label label_LoopEndL
#
# A region that starts at a loop head is not the kernel entry: the kernel's own
# prologue already set SCHED_MODE. InsertWaitAlu must not put a mode write at
# the region's first label (it would run on every iteration with no drain), and
# RemoveWaitAlu must keep a mode switch the input owns there, since nothing in a
# mid-kernel region would put it back.

# CHECK-LABEL: label_LoopBeginL:
# CHECK-NOT:   {{s_setreg_(IMM32|imm32)_b32 hwreg\(26, ?0, ?2\), 2}}
# CHECK:       s_wait_alu depctr_va_vdst(0)
# CHECK-NEXT:  buffer_store_b32 v1, v2, s[4:7], s3 offen offset:16
# CHECK:       {{s_setreg_(IMM32|imm32)_b32 hwreg\(26, ?0, ?2\), 0}}
# CHECK-LABEL: label_LoopEndL:

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
kernel:
s_setreg_imm32_b32 hwreg(26, 0, 2), 2
v_mov_b32 v9, 0
label_LoopBeginL:
v_add_f32 v1, v2, v3
buffer_store_b32 v1, v2, s[4:7], s3 offen offset:16
s_sub_u32 s10, s10, 1
s_cmp_eq_u32 s10, 0
s_cbranch_scc0 label_LoopBeginL
s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)
s_setreg_imm32_b32 hwreg(26, 0, 2), 0
label_LoopEndL:
s_endpgm
