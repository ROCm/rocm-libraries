# RUN: %stinkytofu-opt --arch gfx1250 %s --InsertClusterBarrierPass=pgr=1,lead=0 --from-label region_begin --to-label region_end --emit-asm --preserve-symbolic-regs
#
# ClusterBarrierRule3Mode=0 with lead 0 and SCC dead at the wait: the signal block is
# co-located with its wait, right above the drains of the protect s_barrier_signal -1.
#
# CHECK-LABEL: label_LoopBeginL:
# CHECK-NEXT: v_add_f32 v10, v11, v12
# CHECK-NEXT: s_cmp_eq_u32 s[sgprWaveIdx], 0
# CHECK-NEXT: s_cbranch_scc0 label_skipCBPreSignal_{{[A-Za-z0-9]+}}
# CHECK-NEXT: s_barrier_signal -3
# CHECK-NEXT: label_skipCBPreSignal_{{[A-Za-z0-9]+}}:
# CHECK-NEXT: s_barrier_wait -3
# CHECK-NEXT: s_wait_tensorcnt 0
# CHECK-NEXT: s_barrier_signal -1
# CHECK-NEXT: s_barrier_wait -1
# CHECK-NEXT: tensor_load_to_lds s[20:23], s[24:31]

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
s_wait_tensorcnt 0
s_barrier_signal -1
s_barrier_wait -1
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
