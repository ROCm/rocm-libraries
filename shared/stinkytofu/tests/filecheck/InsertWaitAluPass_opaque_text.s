# RUN: %stinkytofu-opt --arch gfx1250 %s --InsertWaitAluPass --allow-opaque-text --from-label region_begin --to-label region_end

# The text line may read v1 (written by the v_add_f32 before it) and may write
# any VGPR read after it. InsertWaitAlu cannot see either, so it drains VALU
# results and VMEM source reads on both sides of the text.

# CHECK-LABEL: region_begin:
# CHECK:      v_add_f32 v1, v2, v3
# CHECK-NEXT: s_wait_alu {{.*}}va_vdst(0){{.*}}vm_vsrc(0)
# CHECK-NEXT: future_global_load_b32 v4, v1, off offset:16
# CHECK-NEXT: s_wait_alu {{.*}}va_vdst(0){{.*}}vm_vsrc(0)
# CHECK-NEXT: v_add_f32 v6, v5, v5

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
v_add_f32 v1, v2, v3
future_global_load_b32 v4, v1, off offset:16
v_add_f32 v6, v5, v5
region_end:
s_endpgm
