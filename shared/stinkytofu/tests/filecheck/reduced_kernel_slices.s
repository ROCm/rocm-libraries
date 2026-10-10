# RUN: %stinkytofu-opt --arch gfx1250 %s --whole-function --preserve-comments --RaiseVgprMsbPass --RemoveWaitAluPass --InsertVgprMsbPass --InsertWaitAluPass --emit-asm 2>/dev/null
#
# Representative slices reduced from full GEMM kernels. They pin the loop head,
# cluster gate, MSB join, named-barrier M0 dependency, and IU8 signedness modifier.

# CHECK-LABEL: reduced_kernel_slices:
# Loop-carried VALU destination read at the next loop head.
# CHECK-LABEL: .L_loop:
# CHECK:       s_wait_alu depctr_va_vdst(0)
# CHECK:       buffer_store_b32 v1
# CHECK:       s_cbranch_scc0 .L_loop
# Cluster gate and its converged low-VGPR path survive the full ord pipeline.
# CHECK:       s_cbranch_scc0 .L_cluster_join
# CHECK-NEXT:  s_barrier_signal -3
# CHECK-NEXT:  s_barrier_wait -3{{.*}}asm_verify: cluster-round-v1 participants=0
# CHECK-NEXT:  .L_cluster_join:
# CHECK:       v_add_f32 v4, v5, v6
# A high-VGPR path is lowered and reset before the low-VGPR join.
# CHECK:       s_set_vgpr_msb {{(65|134)}}
# CHECK:       v_add_f32
# CHECK:       s_set_vgpr_msb {{(16640|34304)}}
# CHECK-LABEL: .L_msb_join:
# Named-barrier reads of M0 retain the required expert-mode SALU fence.
# CHECK:       s_mov_b32 m0, s26
# CHECK-NEXT:  s_wait_alu depctr_sa_sdst(0)
# CHECK-NEXT:  s_barrier_join m0
# IU8 signedness is a structured modifier and must not be dropped.
# CHECK:       v_wmma_i32_16x16x64_iu8 v[0:7], v[8:11], v[12:15], v[0:7] neg_lo:[1,1,0]

.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
.text
reduced_kernel_slices:
.L_loop:
buffer_store_b32 v1, v2, s[4:7], s3 offen offset:16
s_sub_u32 s10, s10, 1
s_cmp_eq_u32 s10, 0
v_add_f32 v1, v2, v3
s_cbranch_scc0 .L_loop
s_cmp_eq_u32 s11, 0
s_cbranch_scc0 .L_cluster_join
s_barrier_signal -3
s_barrier_wait -3 // asm_verify: cluster-round-v1 participants=0
.L_cluster_join:
v_add_f32 v4, v5, v6
s_cbranch_scc0 .L_msb_join
s_set_vgpr_msb 0x41
v_add_f32 v257, v258, v259
s_set_vgpr_msb 0x4100
.L_msb_join:
s_mov_b32 m0, s26
s_barrier_join m0
v_wmma_i32_16x16x64_iu8 v[0:7], v[8:11], v[12:15], v[0:7] neg_lo:[1,1,0]
s_endpgm
