# RUN: %stinkytofu-opt --arch gfx1250 %s --CFGBuilderPass --StinkyBuildImplicitDependencyPass --StinkyDAGSchedulerPass --from-label region_begin --to-label region_end
#
# Bare special registers (exec_lo, vcc_lo, ...) are registers, not strings. As
# strings, EXEC grouping never saw the narrow EXEC write, so the scheduler
# hoisted the masked v_add_f32 and the buffer_load above `s_mov_b32 exec_lo, s21`
# (inactive lanes written, load issued under the wrong mask).

# CHECK-LABEL: region_begin:
# CHECK-NOT:   buffer_load_b32
# CHECK:       s_mov_b32 exec_lo, s21
# CHECK-NEXT:  v_add_f32 v4, v5, v6
# CHECK-NEXT:  s_mov_b32 exec_lo, -1
# CHECK-LABEL: region_end:

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
v_cmp_gt_u32_e64 vcc_lo, v0, v9
v_mov_b32 v20, v21
v_cndmask_b32 v1, v2, v3, vcc_lo
s_mov_b32 exec_lo, s21
v_add_f32 v4, v5, v6
s_mov_b32 exec_lo, -1
buffer_load_b32 v7, v8, s[4:7], s3 offen
v_mul_f32 v10, v11, v12
region_end:
s_endpgm
