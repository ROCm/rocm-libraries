# RUN: %stinkytofu-opt --arch gfx1250 %s --vgpr-msb-mode=msb16 --RaiseVgprMsbPass --InsertVgprMsbPass --from-label label_LoopBeginL --to-after-branch-to label_LoopBeginL
#
# --to-after-branch-to ends the region right after the first branch back to the
# given label, so a loop whose exit has no label can still be cut out. Only the
# loop body is rewritten: the setter before the region and the one after it are
# left exactly as written.

# CHECK:       s_set_vgpr_msb 0x41
# CHECK-NEXT:  v_mov_b32 v9, 0
# CHECK-LABEL: label_LoopBeginL:
# CHECK:       v_add_f32 v[257-256], v[258-256], v3
# CHECK:       s_cbranch_scc0 label_LoopBeginL
# CHECK-NEXT:  s_set_vgpr_msb 0x4100
# CHECK-NEXT:  v_mov_b32 v8, 0

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
kernel:
s_set_vgpr_msb 0x41
v_mov_b32 v9, 0
label_LoopBeginL:
s_set_vgpr_msb 0x4141
v_add_f32 v1, v2, v3
s_sub_u32 s10, s10, 1
s_cmp_eq_u32 s10, 0
s_cbranch_scc0 label_LoopBeginL
s_set_vgpr_msb 0x4100
v_mov_b32 v8, 0
s_endpgm
