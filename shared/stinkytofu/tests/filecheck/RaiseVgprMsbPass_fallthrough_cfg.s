# RUN: %stinkytofu-opt --arch gfx1250 %s --CFGBuilderPass --RaiseVgprMsbPass --from-label label_start --to-label label_end
#
# CFGBuilder splits after s_cbranch_scc0 without a label. The not-taken path
# falls through with the hardware MSB state unchanged (0x55), so the v_add_f32
# after the branch addresses v260..v262. Only a real label resets the state.

# CHECK-LABEL: label_start:
# CHECK:       v_add_f32 v257, v258, v259
# CHECK-NEXT:  v_add_f32 v257, v258, v259
# CHECK-NEXT:  s_cbranch_scc0 label_x
# CHECK-NEXT:  v_add_f32 v260, v261, v262
# CHECK-LABEL: label_x:
# CHECK-NEXT:  v_mov_b32 v775, v776

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
label_start:
s_set_vgpr_msb 0x55
v_add_f32 v1, v2, v3
s_set_vgpr_msb 85
v_add_f32 v1, v2, v3
s_cbranch_scc0 label_x
v_add_f32 v4, v5, v6
label_x:
s_set_vgpr_msb 0xc3
v_mov_b32 v7, v8
label_end:
s_endpgm
