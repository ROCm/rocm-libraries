# RUN: %stinkytofu-opt --arch gfx1250 %s --InsertClusterBarrierPass=pgr=1,lead=0 --from-label region_begin --to-label region_end --emit-asm --preserve-symbolic-regs
#
# ClusterBarrierRule3Mode=0 with lead 0 and SCC live at the wait: the signal block, whose
# gate s_cmp_eq_u32 s[sgprWaveIdx], 0 writes SCC, must not land between s_cmp_eq_u32 s80, 0
# and its reader s_cselect_b32, so it climbs above the def instead of co-locating.
#
# CHECK-LABEL: label_LoopBeginL:
# CHECK: s_barrier_signal -3
# CHECK: s_cmp_eq_u32 s80, 0
# CHECK-NOT: s_cmp_eq_u32 s[sgprWaveIdx]
# CHECK: s_cselect_b32 s81, s82, 0

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
.set sgprWaveIdx, 2
.set sgprLoopCounterL, 3
region_begin:
label_GSU_1:
v_add_f32 v1, v2, v3
tensor_load_to_lds s[20:23], s[24:31]
v_add_f32 v4, v5, v6
label_PreLoopJoin:
v_add_f32 v7, v8, v9
label_LoopBeginL:
v_add_f32 v10, v11, v12
s_cmp_eq_u32 s80, 0
s_wait_tensorcnt 0
s_barrier_signal -1
s_barrier_wait -1
s_cselect_b32 s81, s82, 0
tensor_load_to_lds s[20:23], s[24:31]
v_add_f32 v13, v14, v15
s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1
s_cmp_eq_u32 s[sgprLoopCounterL], 0
s_cbranch_scc1 label_LoopEndL
s_branch label_LoopBeginL
label_LoopEndL:
s_nop 0
region_end:
s_endpgm
