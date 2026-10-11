# RUN: %stinkytofu-opt --arch gfx1250 %s --CFGBuilderPass --InsertVgprMsbPass --from-label region_begin --to-label region_end
#
# Shape InsertClusterBarrierPass leaves before InsertVgprMsbPass: wave 0 falls through
# into the signal block, the others branch to label_skipCBPreSignal_*, and both meet there.
# Each join sets the MSB again before its first VGPR use, since the two paths may arrive
# with different state, and the fall-through block encodes its own high VGPRs.

# CHECK-LABEL: region_begin:
# CHECK:       s_set_vgpr_msb 69
# CHECK-NEXT:  v_add_f32 v[257-256], v[258-256], v[259-256]
# CHECK:       s_set_vgpr_msb 17664
# CHECK-NEXT:  s_cbranch_scc0 label_skipCBPreSignal_0
# CHECK-NEXT:  s_barrier_signal -3
# CHECK-NEXT:  label_skipCBPreSignal_0:
# CHECK:       s_set_vgpr_msb 0
# CHECK-NEXT:  v_add_f32 v1, v2, v3
# CHECK-NEXT:  s_set_vgpr_msb 69
# CHECK-NEXT:  v_add_f32 v[257-256], v[258-256], v[259-256]
# CHECK:       s_set_vgpr_msb 17664
# CHECK-NEXT:  s_cbranch_scc0 label_skipCBPreSignal_1
# CHECK-NEXT:  s_set_vgpr_msb
# CHECK-NEXT:  v_mov_b32 v[300-256], v[301-256]
# CHECK-NEXT:  s_barrier_signal -3
# CHECK-NEXT:  s_set_vgpr_msb 16640
# CHECK-NEXT:  label_skipCBPreSignal_1:
# CHECK:       s_set_vgpr_msb 69
# CHECK-NEXT:  v_add_f32 v[260-256], v[261-256], v[262-256]
# CHECK-NEXT:  s_set_vgpr_msb 17664
# CHECK-NEXT:  region_end:

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
region_begin:
v_add_f32 v257, v258, v259
s_cmp_eq_u32 s10, 0
s_cbranch_scc0 label_skipCBPreSignal_0
s_barrier_signal -3
label_skipCBPreSignal_0:
v_add_f32 v1, v2, v3
v_add_f32 v257, v258, v259
s_cmp_eq_u32 s10, 0
s_cbranch_scc0 label_skipCBPreSignal_1
v_mov_b32 v300, v301
s_barrier_signal -3
label_skipCBPreSignal_1:
v_add_f32 v260, v261, v262
region_end:
s_endpgm
