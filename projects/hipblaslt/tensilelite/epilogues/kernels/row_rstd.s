  ; Module: row_rstd_mod
  ; Decomposed-flow producer reduction (Kernel 2, "reduce-and-return"). Reduces the K1
  ; per-tile partial sums-of-squares for each row and writes the finalized per-row rstd
  ;   rstd[m] = rsqrt( (sum_t partialBuf[m,t]) * inv_d + eps )
  ; to the handoff buffer, WITHOUT touching D (unlike row_div, which applies rstd to D).
  ;
  ; Kernarg layout (matches buildRowDivArgs / launchRowRstd):
  ;   0  rstdOut ptr (8)   8  partialBuf ptr (8)   16 pad(i32)   20 n (nHidden, i32)
  ;   24 n_c (i32)         28 n_d (nD, i32)         32 inv_d(f32) 36 eps(f32)
  ; grid = (M, 1, 1), block = (64, 1, 1); one workgroup per row m = workgroup_id_x.
  .amdgcn_target "amdgcn-amd-amdhsa--gfx950"
  .text
  .globl row_rstd
  .p2align 8
  .type row_rstd,@function
row_rstd:
  s_load_dword s15, s[0:1], 28
  s_load_dword s14, s[0:1], 32
  s_load_dword s13, s[0:1], 36
  s_load_dwordx2 s[8:9], s[0:1], 8
  s_load_dwordx2 s[10:11], s[0:1], 0
  s_waitcnt lgkmcnt(0)
  ; partialBuf SRD s[4:7]: base = partialBuf + row*nD*4, row = s2 (workgroup_id_x).
  s_lshl_b32 s6, s15, 2
  s_mul_i32 s0, s2, s6
  s_mov_b32 s1, s0
  s_ashr_i32 s0, s0, 31
  s_add_u32 s4, s8, s1
  s_addc_u32 s5, s9, s0
  s_mov_b32 s7, 163840
  ; lane byte offset into the row (each lane t reads partialBuf[row, lane + 64*i]).
  v_lshlrev_b32 v3, 2, v0
  s_cmp_gt_i32 s15, 0
  s_mov_b32 s0, 0
  v_mov_b32 v2, 0
  v_mov_b32 v1, 0
  s_cbranch_scc0 .RSTD_BB_1
.RSTD_BB_2:
  s_lshl_b32 s1, s0, 2
  v_add_u32 v1, s1, v3
  buffer_load_dword v1, v1, s[4:7], 0 offen
  s_waitcnt vmcnt(0)
  v_add_f32 v1, v2, v1
  s_add_u32 s0, s0, 64
  s_cmp_lt_i32 s0, s15
  s_cbranch_scc0 .RSTD_BB_1
.RSTD_BB_3:
  v_mov_b32 v2, v1
  s_branch .RSTD_BB_2
.RSTD_BB_1:
  v_nop
  v_nop
  v_add_f32_dpp v1, v1, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1
  v_nop
  v_nop
  v_add_f32_dpp v1, v1, v1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf bound_ctrl:1
  v_nop
  v_nop
  v_add_f32_dpp v1, v1, v1 row_half_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
  v_nop
  v_nop
  v_add_f32_dpp v1, v1, v1 row_mirror row_mask:0xf bank_mask:0xf bound_ctrl:1
  v_nop
  v_nop
  v_add_f32_dpp v1, v1, v1 row_bcast:15 row_mask:0xa bank_mask:0xf
  v_nop
  v_nop
  v_add_f32_dpp v1, v1, v1 row_bcast:31 row_mask:0xf bank_mask:0xf bound_ctrl:1
  v_nop
  v_readlane_b32 s0, v1, 63
  v_mov_b32 v1, s0
  v_mul_f32 v1, s14, v1
  v_add_f32 v1, s13, v1
  v_rsq_f32 v1, v1
  ; rstdOut[m] = v1 ; m = s2, rstdOut base = s[10:11].
  s_lshl_b32 s0, s2, 2
  s_add_u32 s0, s10, s0
  s_addc_u32 s1, s11, 0
  v_cmp_eq_u32 vcc, 0, v0
  s_and_saveexec_b64 s[4:5], vcc
  v_mov_b32 v2, s0
  v_mov_b32 v3, s1
  global_store_dword v[2:3], v1, off
  s_mov_b64 exec, s[4:5]
  s_endpgm
  .section .rodata,"a",@progbits
  .p2align 6, 0x0
  .amdhsa_kernel row_rstd
    .amdhsa_kernarg_size 64
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_next_free_vgpr 16
    .amdhsa_next_free_sgpr 24
    .amdhsa_accum_offset 16
  .end_amdhsa_kernel
  .text
.Lfunc_end0:
  .size row_rstd, .Lfunc_end0-row_rstd

  .amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count: 0
    .args:
      - .offset: 0
        .size: 8
        .value_kind: by_value
      - .offset: 8
        .size: 8
        .value_kind: by_value
      - .offset: 16
        .size: 4
        .value_kind: by_value
      - .offset: 20
        .size: 4
        .value_kind: by_value
      - .offset: 24
        .size: 4
        .value_kind: by_value
      - .offset: 28
        .size: 4
        .value_kind: by_value
      - .offset: 32
        .size: 4
        .value_kind: by_value
      - .offset: 36
        .size: 4
        .value_kind: by_value
      - .offset: 40
        .size: 4
        .value_kind: hidden_group_size_x
      - .offset: 44
        .size: 4
        .value_kind: hidden_group_size_y
      - .offset: 48
        .size: 4
        .value_kind: hidden_group_size_z
      - .offset: 52
        .size: 4
        .value_kind: hidden_block_count_x
      - .offset: 56
        .size: 4
        .value_kind: hidden_block_count_y
      - .offset: 60
        .size: 4
        .value_kind: hidden_block_count_z
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 64
    .language: Assembler
    .max_flat_workgroup_size: 1024
    .name: row_rstd
    .private_segment_fixed_size: 0
    .sgpr_count: 18
    .sgpr_spill_count: 0
    .symbol: row_rstd.kd
    .vgpr_count: 12
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdgcn_target: amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
---

  .end_amdgpu_metadata
