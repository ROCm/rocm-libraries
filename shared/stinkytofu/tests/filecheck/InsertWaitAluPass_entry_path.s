# RUN: %stinkytofu-opt --arch gfx1250 %s --InsertWaitAluPass --whole-function; echo "=== mid"; %stinkytofu-opt --arch gfx1250 %s --InsertWaitAluPass --from-label label_LoopBeginL
#
# Whole function, no canonical entry label: the SCHED_MODE write goes before
# the first instruction, after the kernel symbol (nothing branches to it), not
# at the end of the leading directive block.
#
# Region starting at the loop head: no mode write, and one drain on the entry
# path, in front of the loop label. The code before the region may leave work
# in flight, while the analysis starts with nothing in flight; the back-branch
# skips the drain.

# CHECK-LABEL: kernel:
# CHECK-NEXT:  s_setreg_IMM32_b32 hwreg(26,0,2), 2
# CHECK-NEXT:  v_mov_b32 v9, 0
# CHECK-NEXT:  label_LoopBeginL:
# CHECK-NEXT:  s_wait_alu depctr_va_vdst(0)
# CHECK-NEXT:  buffer_store_b32 v1, v2, s[4:7], s3 offen offset:16
# CHECK-LABEL: === mid
# CHECK-NOT:   s_setreg
# CHECK:       v_mov_b32 v9, 0
# CHECK-NEXT:  s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)
# CHECK-NEXT:  label_LoopBeginL:
# CHECK-NEXT:  s_wait_alu depctr_va_vdst(0)
# CHECK-NEXT:  buffer_store_b32 v1, v2, s[4:7], s3 offen offset:16
# CHECK-NOT:   s_setreg

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.globl kernel
.p2align 8
.type kernel,@function
kernel:
v_mov_b32 v9, 0
label_LoopBeginL:
buffer_store_b32 v1, v2, s[4:7], s3 offen offset:16
s_sub_u32 s10, s10, 1
s_cmp_eq_u32 s10, 0
v_add_f32 v1, v2, v3
s_cbranch_scc0 label_LoopBeginL
s_endpgm
