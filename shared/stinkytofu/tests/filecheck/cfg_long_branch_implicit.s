# RUN: %stinkytofu-opt --arch gfx1250 %s --CFGBuilderPass --InsertWaitAluPass --from-label region_begin --to-label region_end --emit-asm; echo "=== elim"; %stinkytofu-opt --arch gfx1250 %s --CFGBuilderPass --StinkyUnreachableBlockElimPass --from-label region_begin --to-label region_end --emit-asm
#
# A raw .s long branch gets its CFG edge from CFGBuilderPass alone. Only
# LongBranchLoweringPass used to recognise the idiom, and no pipeline runs it,
# so label_Target had no predecessor: InsertWaitAlu left the v1 store without
# a wait, and unreachable-block elimination deleted label_Target with it.

# CHECK:      s_setpc_b64 s[62:63]
# CHECK:      label_Target:
# CHECK-NEXT: s_wait_alu depctr_va_vdst(0)
# CHECK-NEXT: buffer_store_b32 v1
# CHECK:      === elim
# CHECK:      s_setpc_b64 s[62:63]
# CHECK-NOT:  label_Dead
# CHECK:      label_Target:
# CHECK-NEXT: buffer_store_b32 v1
# CHECK-NEXT: region_end:

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
v_add_f32 v1, v2, v3
s_getpc_b64 s[62:63]
s_add_i32 s64, label_Target, 4
s_add_u32 s62, s62, s64
s_addc_u32 s63, s63, 0
s_setpc_b64 s[62:63]
label_Dead:
s_nop 0
label_Target:
buffer_store_b32 v1, v2, s[4:7], s3 offen offset:16
region_end:
s_endpgm
