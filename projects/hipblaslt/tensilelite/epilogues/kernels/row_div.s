  ; Module: row_div_mod
  .amdgcn_target "amdgcn-amd-amdhsa--gfx950"
  .text
  .globl row_div
  .p2align 8
  .type row_div,@function
row_div:
  s_load_dword s4, s[0:1], 40
  s_waitcnt lgkmcnt(0)
  s_and_b32 s12, s4, 65535
  s_load_dword s13, s[0:1], 36
  s_load_dword s14, s[0:1], 32
  s_load_dword s15, s[0:1], 28
  s_load_dword s16, s[0:1], 24
  s_load_dword s17, s[0:1], 20
  s_load_dwordx2 s[8:9], s[0:1], 8
  s_load_dwordx2 s[10:11], s[0:1], 0
  s_waitcnt lgkmcnt(0)
  s_lshl_b32 s6, s15, 2
  s_mul_i32 s0, s2, s6
  s_mov_b32 s1, s0
  s_ashr_i32 s0, s0, 31
  s_add_u32 s4, s8, s1
  s_addc_u32 s5, s9, s0
  s_mov_b32 s7, 163840
  v_ashrrev_i32 v1, 6, v0
  v_mov_b32 v2, -256
  v_mul_lo_u32 v1, v2, v1
  v_lshlrev_b32 v2, 2, v0
  v_add_u32 v3, v1, v2
  s_cmp_gt_i32 s15, 0
  s_mov_b32 s0, 0
  v_mov_b32 v2, 0
  v_mov_b32 v1, 0
  s_cbranch_scc0 .AMDGCN_BB_1
.AMDGCN_BB_2:
  s_lshl_b32 s1, s0, 2
  v_add_u32 v1, s1, v3
  buffer_load_dword v1, v1, s[4:7], 0 offen
  s_waitcnt vmcnt(0)
  v_add_f32 v1, v2, v1
  s_add_u32 s0, s0, 64
  s_cmp_lt_i32 s0, s15
  s_cbranch_scc0 .AMDGCN_BB_1
.AMDGCN_BB_3:
  v_mov_b32 v2, v1
  s_branch .AMDGCN_BB_2
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
  v_mul_f32 v1, s14, v1
  v_add_f32 v1, s13, v1
  v_rsq_f32 v1, v1
  s_mul_i32 s0, s2, s17
  s_lshl_b32 s0, s0, 1
  s_mul_i32 s1, s16, s3
  s_lshl_b32 s1, s1, 1
  s_add_u32 s0, s0, s1
  s_mov_b32 s1, s0
  s_ashr_i32 s2, s0, 31
  s_add_u32 s0, s10, s1
  s_addc_u32 s1, s11, s2
  s_lshl_b32 s8, s16, 1
  s_or_b32 s1, s1, 65536
  s_mov_b32 s2, s8
  s_mov_b32 s3, 163840
  s_lshl_b32 s5, s12, 4
  s_cmp_lt_i32 s16, 0
  s_mov_b32 s6, scc
  s_sub_u32 s4, -1, s16
  s_cmp_eq_u32 s6, 1
  s_cselect_b32 s4, s4, s16
  s_ashr_i32 s4, s4, 1
  s_sub_u32 s6, -1, s4
  s_cselect_b32 s4, s6, s4
  s_lshl_b32 s6, s4, 2
  v_lshlrev_b32 v2, 4, v0
  s_cmp_gt_i32 s6, 0
  s_mov_b32 s4, 0
  s_cbranch_scc0 .AMDGCN_BB_4
.AMDGCN_BB_5:
  v_add_u32 v3, s4, v2
  buffer_load_dwordx4 v[4:7], v3, s[0:3], 0 offen
  s_waitcnt vmcnt(0)
  v_and_b32 v8, 65535, v4
  v_and_b32 v9, 65535, v5
  v_and_b32 v10, 65535, v6
  v_and_b32 v11, 65535, v7
  v_lshlrev_b32 v8, 16, v8
  v_and_b32 v4, -65536, v4
  v_lshlrev_b32 v9, 16, v9
  v_and_b32 v5, -65536, v5
  v_lshlrev_b32 v10, 16, v10
  v_and_b32 v6, -65536, v6
  v_lshlrev_b32 v11, 16, v11
  v_and_b32 v7, -65536, v7
  v_mul_f32 v8, v8, v1
  v_mul_f32 v4, v4, v1
  v_mul_f32 v9, v9, v1
  v_mul_f32 v5, v5, v1
  v_mul_f32 v10, v10, v1
  v_mul_f32 v6, v6, v1
  v_mul_f32 v11, v11, v1
  v_mul_f32 v7, v7, v1
  v_lshrrev_b32 v8, 16, v8
  v_lshrrev_b32 v4, 16, v4
  v_lshrrev_b32 v9, 16, v9
  v_lshrrev_b32 v5, 16, v5
  v_lshrrev_b32 v10, 16, v10
  v_lshrrev_b32 v6, 16, v6
  v_lshrrev_b32 v11, 16, v11
  v_lshrrev_b32 v7, 16, v7
  v_pack_b32_f16 v4, v8, v4
  v_pack_b32_f16 v5, v9, v5
  v_pack_b32_f16 v6, v10, v6
  v_pack_b32_f16 v7, v11, v7
  buffer_store_dwordx4 v[4:7], v3, s[0:3], 0 offen
  s_add_u32 s4, s4, s5
  s_cmp_lt_i32 s4, s6
  s_cbranch_scc1 .AMDGCN_BB_5
.AMDGCN_BB_4:
  s_and_b32 s4, s16, 1
  s_cmp_lg_i32 s4, 0
  v_cmp_eq_i32 vcc, 0, v0
  s_mov_b64 s[4:5], vcc
  s_cselect_b64 vcc, -1, 0
  s_mov_b64 s[6:7], vcc
  s_and_b64 vcc, s[6:7], s[4:5]
  s_and_saveexec_b64 s[4:5], vcc
  s_cbranch_vccz .AMDGCN_BB_6
.AMDGCN_BB_7:
  s_add_u32 s6, s8, -2
  v_mov_b32 v0, s6
  buffer_load_ushort v0, v0, s[0:3], 0 offen
  s_waitcnt vmcnt(0)
  v_lshlrev_b32 v0, 16, v0
  v_mul_f32 v0, v0, v1
  v_lshrrev_b32 v0, 16, v0
  v_mov_b32 v1, s6
  buffer_store_short v0, v1, s[0:3], 0 offen
  s_branch .AMDGCN_BB_6
.AMDGCN_BB_6:
  s_mov_b64 exec, s[4:5]
  s_endpgm
  .section .rodata,"a",@progbits
  .p2align 6, 0x0
  .amdhsa_kernel row_div
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
  .size row_div, .Lfunc_end0-row_div

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
    .name: row_div
    .private_segment_fixed_size: 0
    .sgpr_count: 18
    .sgpr_spill_count: 0
    .symbol: row_div.kd
    .vgpr_count: 12
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdgcn_target: amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
---

  .end_amdgpu_metadata
