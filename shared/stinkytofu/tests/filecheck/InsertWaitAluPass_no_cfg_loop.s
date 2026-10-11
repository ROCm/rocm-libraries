# RUN: %stinkytofu-opt --arch gfx1250 %s --InsertWaitAluPass --from-label label_LoopBeginL --to-label label_LoopEndL
#
# No CFGBuilderPass before InsertWaitAlu: the pass builds the CFG itself. In one
# flat block the back-branch never fed the loop head, so the store at the top
# of the loop read v1 before the previous iteration's v_add_f32 retired, and
# the SCHED_MODE write landed after the loop label (run every iteration).

# CHECK-LABEL: label_LoopBeginL:
# CHECK-NEXT:  s_wait_alu depctr_va_vdst(0)
# CHECK-NEXT:  buffer_store_b32 v1, v2, s[4:7], s3 offen offset:16
# CHECK:       s_wait_alu depctr_vm_vsrc(0)
# CHECK-NEXT:  v_add_f32 v1, v2, v3
# CHECK-NEXT:  s_cbranch_scc0 label_LoopBeginL

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
label_LoopBeginL:
buffer_store_b32 v1, v2, s[4:7], s3 offen offset:16
s_sub_u32 s10, s10, 1
s_cmp_eq_u32 s10, 0
v_add_f32 v1, v2, v3
s_cbranch_scc0 label_LoopBeginL
label_LoopEndL:
s_endpgm
