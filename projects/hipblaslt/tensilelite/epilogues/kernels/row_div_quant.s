  ; Module: module
  .amdgcn_target "amdgcn-amd-amdhsa--gfx950"
  .text
  .globl row_div_quant
  .p2align 8
  .type row_div_quant,@function
row_div_quant:
  s_load_dword s4, s[0:1], 60
  s_waitcnt lgkmcnt(0)
  s_and_b32 s15, s4, 65535
  s_load_dword s16, s[0:1], 56
  s_load_dword s17, s[0:1], 52
  s_load_dword s18, s[0:1], 48
  s_load_dword s19, s[0:1], 44
  s_load_dword s20, s[0:1], 40
  s_load_dword s21, s[0:1], 32
  s_load_dwordx2 s[10:11], s[0:1], 24
  s_load_dwordx2 s[8:9], s[0:1], 16
  s_load_dwordx2 s[6:7], s[0:1], 8
  s_load_dwordx2 s[12:13], s[0:1], 0
  s_waitcnt lgkmcnt(0)
  s_lshl_b32 s14, s18, 2
  s_mul_i32 s1, s2, s14
  s_mov_b32 s0, s1
  s_ashr_i32 s1, s1, 31
  s_add_u32 s4, s6, s0
  s_addc_u32 s5, s7, s1
  s_mov_b32 s6, s14
  s_mov_b32 s7, 163840
  v_ashrrev_i32 v8, 6, v0
  v_mov_b32 v1, -256
  v_mul_lo_u32 v8, v1, v8
  v_lshlrev_b32 v1, 2, v0
  v_add_u32 v8, v8, v1
  s_cmp_gt_i32 s18, 0
  s_mov_b32 s14, 0
  v_mov_b32 v1, 0
  s_cbranch_scc0 .AMDGCN_BB_1
.AMDGCN_BB_2:
  s_mov_b32 s0, s14
  s_branch .AMDGCN_BB_3
.AMDGCN_BB_3:
  s_lshl_b32 s1, s0, 2
  v_add_u32 v2, s1, v8
  buffer_load_dword v2, v2, s[4:7], 0 offen
  s_waitcnt vmcnt(0)
  v_add_f32 v1, v1, v2
  s_add_u32 s0, s0, 64
  s_cmp_lt_i32 s0, s18
  s_cbranch_scc1 .AMDGCN_BB_3
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
  v_mul_f32 v8, s17, v1
  v_add_f32 v8, s16, v8
  v_rsq_f32 v1, v8
  v_nop
  v_div_scale_f32 v2, vcc, s21, s21, v1
  v_rcp_f32 v3, v2
  s_mov_b64 s[0:1], vcc
  v_div_scale_f32 v4, vcc, v1, s21, v1
  v_mov_b32 v5, 0
  v_sub_f32 v2, v5, v2
  v_mov_b32 v5, 1065353216
  v_fma_f32 v5, v2, v3, v5
  v_fma_f32 v3, v5, v3, v3
  v_mul_f32 v5, v4, v3
  v_fma_f32 v6, v2, v5, v4
  v_fma_f32 v5, v6, v3, v5
  v_fma_f32 v2, v2, v5, v4
  v_div_fmas_f32 v2, v2, v3, v5
  v_div_fixup_f32 v8, v2, s21, v1
  s_mul_i32 s1, s2, s20
  s_lshl_b32 s0, s1, 1
  s_mul_i32 s2, s19, s3
  s_lshl_b32 s3, s2, 1
  s_add_u32 s3, s0, s3
  s_mov_b32 s0, s3
  s_ashr_i32 s4, s3, 31
  s_add_u32 s0, s12, s0
  s_addc_u32 s5, s13, s4
  s_add_u32 s2, s1, s2
  s_mov_b32 s1, s2
  s_ashr_i32 s2, s2, 31
  s_add_u32 s4, s8, s1
  s_addc_u32 s6, s9, s2
  s_mov_b32 s1, s3
  s_ashr_i32 s2, s3, 31
  s_add_u32 s8, s10, s1
  s_addc_u32 s9, s11, s2
  s_lshl_b32 s10, s19, 1
  s_or_b32 s1, s5, 65536
  s_mov_b32 s2, s10
  s_mov_b32 s3, 163840
  s_or_b32 s5, s6, 65536
  s_mov_b32 s6, s19
  s_mov_b32 s7, 163840
  s_or_b32 s9, s9, 65536
  s_mov_b32 s11, 163840
  s_lshl_b32 s12, s15, 4
  s_cmp_lt_i32 s19, 0
  s_mov_b32 s15, scc
  s_sub_u32 s13, -1, s19
  s_cmp_eq_u32 s15, 1
  s_cselect_b32 s13, s13, s19
  s_ashr_i32 s15, s13, 1
  s_sub_u32 s16, -1, s15
  s_cselect_b32 s15, s16, s15
  s_lshl_b32 s15, s15, 2
  v_lshlrev_b32 v13, 4, v0
  s_cmp_gt_i32 s15, 0
  s_cbranch_scc0 .AMDGCN_BB_4
.AMDGCN_BB_5:
  v_add_u32 v14, s14, v13
  buffer_load_dwordx4 v[4:7], v14, s[0:3], 0 offen
  s_waitcnt vmcnt(0)
  v_and_b32 v2, 65535, v4
  v_and_b32 v3, 65535, v5
  v_and_b32 v9, 65535, v6
  v_and_b32 v10, 65535, v7
  v_lshlrev_b32 v2, 16, v2
  v_and_b32 v4, -65536, v4
  v_lshlrev_b32 v3, 16, v3
  v_and_b32 v5, -65536, v5
  v_lshlrev_b32 v9, 16, v9
  v_and_b32 v6, -65536, v6
  v_lshlrev_b32 v10, 16, v10
  v_and_b32 v7, -65536, v7
  v_mul_f32 v11, v2, v8
  v_mul_f32 v12, v4, v8
  v_mul_f32 v15, v3, v8
  v_mul_f32 v16, v5, v8
  v_mul_f32 v17, v9, v8
  v_mul_f32 v18, v6, v8
  v_mul_f32 v19, v10, v8
  v_mul_f32 v20, v7, v8
  v_mul_f32 v2, v2, v1
  v_mul_f32 v4, v4, v1
  v_mul_f32 v3, v3, v1
  v_mul_f32 v5, v5, v1
  v_mul_f32 v9, v9, v1
  v_mul_f32 v6, v6, v1
  v_mul_f32 v10, v10, v1
  v_mul_f32 v7, v7, v1
  v_lshrrev_b32 v2, 16, v2
  v_lshrrev_b32 v4, 16, v4
  v_lshrrev_b32 v3, 16, v3
  v_lshrrev_b32 v5, 16, v5
  v_lshrrev_b32 v9, 16, v9
  v_lshrrev_b32 v6, 16, v6
  v_lshrrev_b32 v10, 16, v10
  v_lshrrev_b32 v7, 16, v7
  v_pack_b32_f16 v4, v2, v4
  v_pack_b32_f16 v5, v3, v5
  v_pack_b32_f16 v6, v9, v6
  v_pack_b32_f16 v7, v10, v7
  v_mov_b32 v2, 1138753536
  v_mov_b32 v3, -1008730112
  v_med3_f32 v9, v11, v3, v2
  v_med3_f32 v2, v12, v3, v2
  v_cvt_pk_fp8_f32 v2, v9, v2
  v_mov_b32 v3, 1138753536
  v_mov_b32 v9, -1008730112
  v_med3_f32 v10, v15, v9, v3
  v_med3_f32 v3, v16, v9, v3
  v_cvt_pk_fp8_f32 v3, v10, v3
  v_mov_b32 v9, 1138753536
  v_mov_b32 v10, -1008730112
  v_med3_f32 v11, v17, v10, v9
  v_med3_f32 v9, v18, v10, v9
  v_cvt_pk_fp8_f32 v9, v11, v9
  v_mov_b32 v10, 1138753536
  v_mov_b32 v11, -1008730112
  v_med3_f32 v12, v19, v11, v10
  v_med3_f32 v10, v20, v11, v10
  v_cvt_pk_fp8_f32 v10, v12, v10
  buffer_store_dwordx4 v[4:7], v14, s[8:11], 0 offen
  v_and_b32 v2, 65535, v2
  v_lshlrev_b32 v3, 16, v3
  v_or_b32 v2, v2, v3
  v_and_b32 v3, 65535, v9
  v_lshlrev_b32 v4, 16, v10
  v_or_b32 v3, v3, v4
  v_lshrrev_b32 v4, 1, v14
  buffer_store_dwordx2 v[2:3], v4, s[4:7], 0 offen
  s_add_u32 s14, s14, s12
  s_cmp_lt_i32 s14, s15
  s_cbranch_scc1 .AMDGCN_BB_5
.AMDGCN_BB_4:
  s_ashr_i32 s14, s13, 2
  s_sub_u32 s12, -1, s14
  s_cselect_b32 s14, s12, s14
  s_lshl_b32 s14, s14, 2
  v_cmp_eq_i32 vcc, 0, v0
  s_and_saveexec_b64 s[12:13], vcc
  s_cbranch_vccz .AMDGCN_BB_6
.AMDGCN_BB_7:
  s_cmp_lt_i32 s14, s19
  s_cbranch_scc0 .AMDGCN_BB_6
.AMDGCN_BB_8:
  s_mov_b32 s15, s14
  s_lshl_b32 s14, s15, 1
  v_mov_b32 v0, s14
  buffer_load_ushort v0, v0, s[0:3], 0 offen
  s_waitcnt vmcnt(0)
  v_lshlrev_b32 v2, 16, v0
  v_mul_f32 v0, v2, v1
  v_lshrrev_b32 v3, 16, v0
  v_mov_b32 v0, s14
  buffer_store_short v3, v0, s[8:11], 0 offen
  v_mul_f32 v4, v2, v8
  v_mov_b32 v0, 1138753536
  v_mov_b32 v2, -1008730112
  v_med3_f32 v3, v4, v2, v0
  v_med3_f32 v0, v4, v2, v0
  v_cvt_pk_fp8_f32 v0, v3, v0
  v_mov_b32 v2, 255
  v_and_b32 v2, v0, v2
  v_mov_b32 v0, s15
  buffer_store_byte v2, v0, s[4:7], 0 offen
  s_add_u32 s14, s15, 1
  s_cmp_lt_i32 s14, s19
  s_cbranch_scc1 .AMDGCN_BB_8
.AMDGCN_BB_6:
  s_mov_b64 exec, s[12:13]
  s_endpgm
  .section .rodata,"a",@progbits
  .p2align 6, 0x0
  .amdhsa_kernel row_div_quant
    .amdhsa_kernarg_size 84
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_next_free_vgpr 24
    .amdhsa_next_free_sgpr 24
    .amdhsa_accum_offset 24
  .end_amdhsa_kernel
  .text
.Lfunc_end0:
  .size row_div_quant, .Lfunc_end0-row_div_quant

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
        .size: 8
        .value_kind: by_value
      - .offset: 24
        .size: 8
        .value_kind: by_value
      - .offset: 32
        .size: 4
        .value_kind: by_value
      - .offset: 36
        .size: 4
        .value_kind: by_value
      - .offset: 40
        .size: 4
        .value_kind: by_value
      - .offset: 44
        .size: 4
        .value_kind: by_value
      - .offset: 48
        .size: 4
        .value_kind: by_value
      - .offset: 52
        .size: 4
        .value_kind: by_value
      - .offset: 56
        .size: 4
        .value_kind: by_value
      - .offset: 60
        .size: 4
        .value_kind: hidden_group_size_x
      - .offset: 64
        .size: 4
        .value_kind: hidden_group_size_y
      - .offset: 68
        .size: 4
        .value_kind: hidden_group_size_z
      - .offset: 72
        .size: 4
        .value_kind: hidden_block_count_x
      - .offset: 76
        .size: 4
        .value_kind: hidden_block_count_y
      - .offset: 80
        .size: 4
        .value_kind: hidden_block_count_z
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 84
    .language: Assembler
    .max_flat_workgroup_size: 1024
    .name: row_div_quant
    .private_segment_fixed_size: 0
    .sgpr_count: 22
    .sgpr_spill_count: 0
    .symbol: row_div_quant.kd
    .vgpr_count: 21
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdgcn_target: amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
---

  .end_amdgpu_metadata
