# RUN: %stinkytofu-opt --arch gfx1250 %s --CFGBuilderPass --InsertWaitAluPass --from-label region_begin --to-label region_end
#
# Assembler-local branch targets (`.L_loop`) lex as '.' + identifier. The branch
# used to be stored as text, so the CFG had no back edge and InsertWaitAlu never
# saw that the store at the end of one iteration still reads v1 when the next
# iteration's v_add overwrites it (VM_VSRC write-after-read).

# CHECK-LABEL: .L_loop:
# CHECK-NEXT:  s_wait_alu depctr_vm_vsrc(0)
# CHECK-NEXT:  v_add_f32 v1, v2, v3
# CHECK-NEXT:  s_wait_alu depctr_va_vdst(0)
# CHECK-NEXT:  buffer_store_b32 v1, v2, s[4:7], s3 offen offset:16
# CHECK:       s_cbranch_scc0 .L_loop

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
s_mov_b32 s10, 4
.L_loop:
v_add_f32 v1, v2, v3
buffer_store_b32 v1, v2, s[4:7], s3 offen offset:16
s_sub_u32 s10, s10, 1
s_cmp_eq_u32 s10, 0
s_cbranch_scc0 .L_loop
region_end:
s_endpgm
