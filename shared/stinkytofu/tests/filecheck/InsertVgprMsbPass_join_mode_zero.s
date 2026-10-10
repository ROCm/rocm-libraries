# RUN: %stinkytofu-opt --arch gfx1250 %s --whole-function --RaiseVgprMsbPass --InsertVgprMsbPass --emit-asm 2>/dev/null
#
# Every edge into a label carries VGPR MSB mode 0. gfx1250's s_set_vgpr_msb
# stores the previous mode in its high byte, so a setter after a join whose
# edges arrive in different modes has no correct encoding (lower_vgpr_msb.py
# rejects the code object). A known non-zero mode goes back to 0 before each
# branch and before falling into a label; the setter after the label then has
# previous mode 0 on every edge. Before: the loop head was entered in mode 0
# (fall-through) and 0x41 (back edge), .L_join in 0x41 and 0x45. 16640 = 0x4100
# (mode 0, previous 0x41), 17664 = 0x4500.

# CHECK-LABEL: .L_loop:
# CHECK:       s_set_vgpr_msb 65{{$}}
# CHECK-NEXT:  v_add_f32 v[257-256], v[258-256], v3
# CHECK-NEXT:  s_cmp_lt_u32 s0, s1
# CHECK-NEXT:  s_set_vgpr_msb 16640{{$}}
# CHECK-NEXT:  s_cbranch_scc1 .L_loop
# CHECK-NEXT:  s_cbranch_scc0 .L_else
# CHECK-NEXT:  s_set_vgpr_msb 65{{$}}
# CHECK-NEXT:  v_add_f32 v[260-256], v[258-256], v3
# CHECK-NEXT:  s_set_vgpr_msb 16640{{$}}
# CHECK-NEXT:  s_branch .L_join
# CHECK-NEXT:  .L_else:
# CHECK:       s_set_vgpr_msb 69{{$}}
# CHECK-NEXT:  v_add_f32 v[261-256], v[262-256], v[263-256]
# CHECK-NEXT:  s_set_vgpr_msb 17664{{$}}
# CHECK-NEXT:  .L_join:
# CHECK-NEXT:  s_endpgm

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
k:
v_mov_b32 v0, 0
.L_loop:
s_set_vgpr_msb 0x41
v_add_f32 v1, v2, v3
s_set_vgpr_msb 0x4100
s_cmp_lt_u32 s0, s1
s_cbranch_scc1 .L_loop
s_cbranch_scc0 .L_else
s_set_vgpr_msb 0x41
v_add_f32 v4, v2, v3
s_branch .L_join
.L_else:
s_set_vgpr_msb 0x45
v_add_f32 v5, v6, v7
.L_join:
s_endpgm
