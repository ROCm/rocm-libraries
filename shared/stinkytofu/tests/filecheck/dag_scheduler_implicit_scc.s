# RUN: %stinkytofu-opt --arch gfx1250 %s --StinkyDAGSchedulerPass --from-label region_begin --to-label region_end 2>/dev/null
#
# The scheduler builds its DAG from operands, so it materialises the implicit
# SCC/VCC/EXEC operands itself instead of relying on
# StinkyBuildImplicitDependencyPass having run first. Without them it
# hoisted the s_cselect above the s_cmp whose SCC it reads.

# CHECK:      s_cmp_eq_u32 s0, 0
# CHECK-NEXT: s_cselect_b32 s4, s5, s6
# CHECK-NEXT: s_add_u32 s1, s2, s3
# CHECK-NEXT: s_addc_u32 s7, s8, 0

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
s_cmp_eq_u32 s0, 0
s_cselect_b32 s4, s5, s6
s_add_u32 s1, s2, s3
s_addc_u32 s7, s8, 0
global_load_b128 v[12:15], v21, s[4:5]
ds_load_b128 v[8:11], v20
region_end:
s_endpgm
