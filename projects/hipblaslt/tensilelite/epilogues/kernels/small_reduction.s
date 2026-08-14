  ; Module: module
  .amdgcn_target "amdgcn-amd-amdhsa--gfx950"
  .text
  .globl rsqrt_row
  .p2align 8
  .type rsqrt_row,@function
rsqrt_row:
  s_load_dword s12, s[0:1], 28
  s_load_dword s13, s[0:1], 24
  s_load_dword s14, s[0:1], 20
  s_load_dword s15, s[0:1], 16
  s_load_dwordx2 s[6:7], s[0:1], 8
  s_load_dwordx2 s[4:5], s[0:1], 0
  s_waitcnt lgkmcnt(0)
  s_lshl_b32 s3, s14, 2
  s_mul_i32 s1, s2, s3
  s_mov_b32 s0, s1
  s_ashr_i32 s1, s1, 31
  s_add_u32 s8, s6, s0
  s_addc_u32 s9, s7, s1
  s_mov_b32 s10, s3
  s_mov_b32 s11, 163840
  v_ashrrev_i32 v2, 6, v0
  v_mov_b32 v1, -256
  v_mul_lo_u32 v2, v1, v2
  v_lshlrev_b32 v1, 2, v0
  v_add_u32 v2, v2, v1
  s_cmp_gt_i32 s14, 0
  s_mov_b32 s3, 0
  v_mov_b32 v1, 0
  s_cbranch_scc0 .AMDGCN_BB_1
.AMDGCN_BB_2:
  s_mov_b32 s0, s3
  s_lshl_b32 s3, s0, 2
  v_add_u32 v3, s3, v2
  buffer_load_dword v3, v3, s[8:11], 0 offen
  s_waitcnt vmcnt(0)
  v_add_f32 v1, v1, v3
  s_add_u32 s3, s0, 64
  s_cmp_lt_i32 s3, s14
  s_cbranch_scc1 .AMDGCN_BB_2
.AMDGCN_BB_1:
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
  v_mul_f32 v2, s13, v1
  v_add_f32 v2, s12, v2
  v_rsq_f32 v1, v2
  s_lshl_b32 s3, s15, 2
  s_mov_b32 s6, s3
  s_mov_b32 s7, 163840
  s_lshl_b32 s3, s2, 2
  v_cmp_eq_i32 vcc, 0, v0
  s_and_saveexec_b64 s[0:1], vcc
  s_cbranch_vccz .AMDGCN_BB_3
.AMDGCN_BB_4:
  v_mov_b32 v0, s3
  buffer_store_dword v1, v0, s[4:7], 0 offen
  s_branch .AMDGCN_BB_3
.AMDGCN_BB_3:
  s_mov_b64 exec, s[0:1]
  s_endpgm
  .section .rodata,"a",@progbits
  .p2align 6, 0x0
  .amdhsa_kernel rsqrt_row
    .amdhsa_kernarg_size 56
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 8
    .amdhsa_next_free_sgpr 16
    .amdhsa_accum_offset 8
  .end_amdhsa_kernel
  .text
.Lfunc_end0:
  .size rsqrt_row, .Lfunc_end0-rsqrt_row

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
        .value_kind: hidden_group_size_x
      - .offset: 36
        .size: 4
        .value_kind: hidden_group_size_y
      - .offset: 40
        .size: 4
        .value_kind: hidden_group_size_z
      - .offset: 44
        .size: 4
        .value_kind: hidden_block_count_x
      - .offset: 48
        .size: 4
        .value_kind: hidden_block_count_y
      - .offset: 52
        .size: 4
        .value_kind: hidden_block_count_z
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 56
    .language: Assembler
    .max_flat_workgroup_size: 1024
    .name: rsqrt_row
    .private_segment_fixed_size: 0
    .sgpr_count: 16
    .sgpr_spill_count: 0
    .symbol: rsqrt_row.kd
    .vgpr_count: 4
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdgcn_target: amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
---

  .end_amdgpu_metadata
