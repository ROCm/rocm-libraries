# RUN: %stinkytofu-opt --arch gfx1250 %s --CFGBuilderPass --InsertWaitAluPass --from-label region_begin --to-label region_end --emit-asm
#
# InsertWaitAlu analyses address-taken blocks the CFG cannot reach. The
# indirect jump below is not a recognised long branch (the carry adds s65), so
# label_Target has no predecessor, and the pass, which walked only RPO blocks,
# left the v1 store there without a wait. The block is entered with unknown
# DEPCTR state, so it now starts with a drain; label_Dead is not referenced and
# never runs.

# CHECK:      s_setpc_b64 s[62:63]
# CHECK-NEXT: label_Dead:
# CHECK-NEXT: v_mov_b32 v3, v1
# CHECK-NEXT: label_Target:
# CHECK-NEXT: s_wait_alu depctr_va_vdst(0) depctr_vm_vsrc(0)
# CHECK-NEXT: buffer_store_b32 v1
# CHECK-NEXT: v_add_f32 v4, v2, v3
# CHECK-NEXT: label_Next:
# CHECK-NEXT: v_mov_b32 v5, v4

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
v_add_f32 v1, v2, v3
s_getpc_b64 s[62:63]
s_add_i32 s64, label_Target, 4
s_mov_b32 s65, 0
s_add_u32 s62, s62, s64
s_addc_u32 s63, s63, s65
s_setpc_b64 s[62:63]
label_Dead:
v_mov_b32 v3, v1
label_Target:
buffer_store_b32 v1, v2, s[4:7], s3 offen offset:16
v_add_f32 v4, v2, v3
label_Next:
v_mov_b32 v5, v4
region_end:
s_endpgm
