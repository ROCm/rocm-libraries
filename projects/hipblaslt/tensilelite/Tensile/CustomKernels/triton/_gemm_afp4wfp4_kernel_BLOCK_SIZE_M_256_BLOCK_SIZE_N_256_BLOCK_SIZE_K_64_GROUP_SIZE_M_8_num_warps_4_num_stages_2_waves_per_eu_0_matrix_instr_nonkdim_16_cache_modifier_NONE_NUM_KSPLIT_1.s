	.text
	.globl	_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1 ; -- Begin function _gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1
	.p2align	8
	.type	_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1,@function
_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1: ; @_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.659:
	.file	1 "/home/jincheye" "compile_native_aiter_afp4_recovered.py"
	.loc	1 89 0 prologue_end             ; compile_native_aiter_afp4_recovered.py:89:0
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.loc	1 0 0 is_stmt 0                 ; :0:0
.Ltmp0:
	.p2align	8
; %bb.660:
.LBB0_0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
                                        ; implicit-def: $vgpr255 : SGPR spill to VGPR lane
	s_mov_b32 s30, s13
	v_writelane_b32 v255, s6, 0
	s_mov_b32 s88, s12
	s_nop 0
	v_writelane_b32 v255, s7, 1
.Ltmp1:
	.file	2 "/home/jincheye/triton/python/triton/language" "standard.py"
	.loc	2 43 13 is_stmt 1               ; standard.py:43:13 @[ compile_native_aiter_afp4_recovered.py:104:15 ]
	s_add_i32 s6, s12, 0xff
	.loc	2 43 12 is_stmt 0               ; standard.py:43:12 @[ compile_native_aiter_afp4_recovered.py:104:15 ]
	s_ashr_i32 s7, s6, 31
	s_lshr_b32 s7, s7, 24
	s_add_i32 s6, s6, s7
	s_ashr_i32 s13, s6, 8
.Ltmp2:
	.loc	2 43 13                         ; standard.py:43:13 @[ compile_native_aiter_afp4_recovered.py:104:42 ]
	s_add_i32 s6, s30, 0xff
	.loc	2 43 12                         ; standard.py:43:12 @[ compile_native_aiter_afp4_recovered.py:104:42 ]
	s_ashr_i32 s7, s6, 31
	s_lshr_b32 s7, s7, 24
	s_add_i32 s6, s6, s7
	s_ashr_i32 s17, s6, 8
.Ltmp3:
	.loc	1 104 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:104:15
	s_mul_i32 s6, s17, s13
.Ltmp4:
	.loc	1 67 21                         ; compile_native_aiter_afp4_recovered.py:67:21 @[ compile_native_aiter_afp4_recovered.py:105:11 ]
	s_add_i32 s7, s6, 7
	.loc	1 67 20 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:67:20 @[ compile_native_aiter_afp4_recovered.py:105:11 ]
	s_ashr_i32 s12, s7, 31
	s_lshr_b32 s12, s12, 29
	s_add_i32 s7, s7, s12
	s_ashr_i32 s18, s7, 3
	.loc	1 68 17 is_stmt 1               ; compile_native_aiter_afp4_recovered.py:68:17 @[ compile_native_aiter_afp4_recovered.py:105:11 ]
	s_ashr_i32 s7, s6, 31
	s_lshr_b32 s7, s7, 29
	s_add_i32 s7, s6, s7
	s_and_b32 s7, s7, -8
	s_sub_i32 s6, s6, s7
	.loc	1 69 17                         ; compile_native_aiter_afp4_recovered.py:69:17 @[ compile_native_aiter_afp4_recovered.py:105:11 ]
	s_cmp_lg_u32 s6, 0
	s_cselect_b32 s6, s6, 8
	.loc	1 71 17                         ; compile_native_aiter_afp4_recovered.py:71:17 @[ compile_native_aiter_afp4_recovered.py:105:11 ]
	s_ashr_i32 s7, s16, 31
	s_lshr_b32 s7, s7, 29
	s_add_i32 s20, s16, s7
	.loc	1 70 11                         ; compile_native_aiter_afp4_recovered.py:70:11 @[ compile_native_aiter_afp4_recovered.py:105:11 ]
	s_and_b32 s7, s20, -8
	s_sub_i32 s16, s16, s7
	.loc	1 72 8                          ; compile_native_aiter_afp4_recovered.py:72:8 @[ compile_native_aiter_afp4_recovered.py:105:11 ]
	s_cmp_ge_i32 s16, s6
.Ltmp5:
	.loc	1 127 20                        ; compile_native_aiter_afp4_recovered.py:127:20
	v_readfirstlane_b32 s12, v0
.Ltmp6:
	.loc	1 72 8                          ; compile_native_aiter_afp4_recovered.py:72:8 @[ compile_native_aiter_afp4_recovered.py:105:11 ]
	s_cbranch_scc0 .LBB0_2
; %bb.1:
	.loc	1 74 12                         ; compile_native_aiter_afp4_recovered.py:74:12 @[ compile_native_aiter_afp4_recovered.py:105:11 ]
	s_mul_i32 s7, s6, s18
	.loc	1 74 40 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:74:40 @[ compile_native_aiter_afp4_recovered.py:105:11 ]
	s_sub_i32 s6, s16, s6
	.loc	1 74 60                         ; compile_native_aiter_afp4_recovered.py:74:60 @[ compile_native_aiter_afp4_recovered.py:105:11 ]
	s_add_i32 s19, s18, -1
	.loc	1 74 39                         ; compile_native_aiter_afp4_recovered.py:74:39 @[ compile_native_aiter_afp4_recovered.py:105:11 ]
	s_mul_i32 s6, s6, s19
	.loc	1 74 12                         ; compile_native_aiter_afp4_recovered.py:74:12 @[ compile_native_aiter_afp4_recovered.py:105:11 ]
	s_add_i32 s19, s6, s7
	s_ashr_i32 s6, s20, 3
	s_cbranch_execz .LBB0_3
	s_branch .LBB0_4
.LBB0_2:
                                        ; implicit-def: $sgpr19
	.loc	1 0 12                          ; compile_native_aiter_afp4_recovered.py:0:12
	s_ashr_i32 s6, s20, 3
.LBB0_3:
	.loc	1 73 16 is_stmt 1               ; compile_native_aiter_afp4_recovered.py:73:16 @[ compile_native_aiter_afp4_recovered.py:105:11 ]
	s_mul_i32 s19, s18, s16
.Ltmp7:
.LBB0_4:
	.loc	1 79 24                         ; compile_native_aiter_afp4_recovered.py:79:24 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	s_lshl_b32 s7, s17, 3
	.loc	1 80 16                         ; compile_native_aiter_afp4_recovered.py:80:16 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	s_abs_i32 s16, s7
	v_cvt_f32_u32_e32 v1, s16
.Ltmp8:
	.loc	1 0 0 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0 @[ compile_native_aiter_afp4_recovered.py:105:11 ]
	s_add_i32 s6, s19, s6
.Ltmp9:
	.loc	1 80 16                         ; compile_native_aiter_afp4_recovered.py:80:16 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	s_sub_i32 s19, 0, s16
	.loc	1 83 28 is_stmt 1               ; compile_native_aiter_afp4_recovered.py:83:28 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	s_abs_i32 s18, s6
	.loc	1 80 16                         ; compile_native_aiter_afp4_recovered.py:80:16 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	v_rcp_f32_e32 v1, v1
	s_xor_b32 s17, s6, s7
	s_ashr_i32 s17, s17, 31
.Ltmp10:
	.loc	1 111 39                        ; compile_native_aiter_afp4_recovered.py:111:39
	v_and_b32_e32 v15, 31, v0
.Ltmp11:
	.loc	1 80 16                         ; compile_native_aiter_afp4_recovered.py:80:16 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
.Ltmp12:
	.loc	1 111 39                        ; compile_native_aiter_afp4_recovered.py:111:39
	s_and_b32 s31, s12, 0x80
.Ltmp13:
	.loc	1 80 16                         ; compile_native_aiter_afp4_recovered.py:80:16 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	s_mov_b32 s85, 0
.Ltmp14:
	.loc	1 111 39                        ; compile_native_aiter_afp4_recovered.py:111:39
	v_and_b32_e32 v16, 32, v0
.Ltmp15:
	.loc	1 80 16                         ; compile_native_aiter_afp4_recovered.py:80:16 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	v_mul_lo_u32 v2, s19, v1
	v_mul_hi_u32 v2, v1, v2
	v_add_u32_e32 v1, v1, v2
	v_mul_hi_u32 v1, s18, v1
	v_mul_lo_u32 v2, v1, s16
	v_sub_u32_e32 v2, s18, v2
	v_add_u32_e32 v3, 1, v1
	v_subrev_u32_e32 v4, s16, v2
	v_cmp_le_u32_e32 vcc, s16, v2
	v_accvgpr_write_b32 a67, v15
	s_nop 0
	v_cndmask_b32_e32 v1, v1, v3, vcc
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_add_u32_e32 v3, 1, v1
	v_cmp_le_u32_e32 vcc, s16, v2
	s_nop 1
	v_cndmask_b32_e32 v1, v1, v3, vcc
	v_xor_b32_e32 v1, s17, v1
	v_subrev_u32_e32 v1, s17, v1
	.loc	1 81 19                         ; compile_native_aiter_afp4_recovered.py:81:19 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	v_lshlrev_b32_e32 v2, 3, v1
	.loc	1 82 24                         ; compile_native_aiter_afp4_recovered.py:82:24 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	v_sub_u32_e32 v3, s13, v2
	.loc	1 82 20 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:82:20 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	v_min_i32_e32 v3, 8, v3
	.loc	1 83 28 is_stmt 1               ; compile_native_aiter_afp4_recovered.py:83:28 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	v_sub_u32_e32 v4, 0, v3
	v_max_i32_e32 v4, v3, v4
	v_cvt_f32_u32_e32 v5, v4
	v_sub_u32_e32 v6, 0, v4
	s_ashr_i32 s13, s6, 31
	.loc	1 84 14                         ; compile_native_aiter_afp4_recovered.py:84:14 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	v_mul_lo_u32 v1, v1, s7
	.loc	1 83 28                         ; compile_native_aiter_afp4_recovered.py:83:28 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	v_rcp_f32_e32 v5, v5
	.loc	1 84 14                         ; compile_native_aiter_afp4_recovered.py:84:14 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	v_sub_u32_e32 v1, s6, v1
	.loc	1 84 13 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:84:13 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	v_xor_b32_e32 v3, v1, v3
	v_ashrrev_i32_e32 v3, 31, v3
	.loc	1 83 28 is_stmt 1               ; compile_native_aiter_afp4_recovered.py:83:28 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	v_mul_f32_e32 v5, 0x4f7ffffe, v5
	v_cvt_u32_f32_e32 v5, v5
.Ltmp16:
	.loc	1 111 39                        ; compile_native_aiter_afp4_recovered.py:111:39
	s_and_b32 s7, s12, 64
	s_lshr_b32 s33, s7, 1
	.loc	1 122 11                        ; compile_native_aiter_afp4_recovered.py:122:11
	s_cmp_lt_i32 s14, 1
.Ltmp17:
	.loc	1 83 28                         ; compile_native_aiter_afp4_recovered.py:83:28 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	v_mul_lo_u32 v6, v6, v5
	v_mul_hi_u32 v6, v5, v6
	v_add_u32_e32 v5, v5, v6
	v_mul_hi_u32 v6, s18, v5
	v_mul_lo_u32 v6, v6, v4
	v_sub_u32_e32 v6, s18, v6
	v_sub_u32_e32 v7, v6, v4
	v_cmp_ge_u32_e32 vcc, v6, v4
	s_nop 1
	v_cndmask_b32_e32 v6, v6, v7, vcc
	v_sub_u32_e32 v7, v6, v4
	v_cmp_ge_u32_e32 vcc, v6, v4
	s_nop 1
	v_cndmask_b32_e32 v6, v6, v7, vcc
	v_xor_b32_e32 v6, s13, v6
	v_subrev_u32_e32 v6, s13, v6
	.loc	1 83 13 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:83:13 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	v_add_u32_e32 v2, v2, v6
	.loc	1 84 13 is_stmt 1               ; compile_native_aiter_afp4_recovered.py:84:13 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	v_sub_u32_e32 v6, 0, v1
	v_max_i32_e32 v1, v1, v6
	v_mul_hi_u32 v5, v1, v5
	v_mul_lo_u32 v6, v5, v4
	v_sub_u32_e32 v1, v1, v6
	v_add_u32_e32 v6, 1, v5
	v_sub_u32_e32 v7, v1, v4
	v_cmp_ge_u32_e32 vcc, v1, v4
.Ltmp18:
	.loc	1 111 16                        ; compile_native_aiter_afp4_recovered.py:111:16
	v_lshlrev_b32_e32 v12, 8, v2
	v_ashrrev_i32_e32 v11, 31, v12
.Ltmp19:
	.loc	1 84 13                         ; compile_native_aiter_afp4_recovered.py:84:13 @[ compile_native_aiter_afp4_recovered.py:108:20 ]
	v_cndmask_b32_e32 v5, v5, v6, vcc
	v_cndmask_b32_e32 v1, v1, v7, vcc
	v_add_u32_e32 v6, 1, v5
	v_cmp_ge_u32_e32 vcc, v1, v4
	v_accvgpr_write_b32 a63, v11
	v_accvgpr_write_b32 a64, v12
	v_cndmask_b32_e32 v1, v5, v6, vcc
	v_xor_b32_e32 v1, v1, v3
	v_sub_u32_e32 v1, v1, v3
.Ltmp20:
	.loc	1 112 16                        ; compile_native_aiter_afp4_recovered.py:112:16
	v_lshlrev_b32_e32 v14, 8, v1
	v_ashrrev_i32_e32 v13, 31, v14
	v_accvgpr_write_b32 a65, v13
	.loc	1 122 11                        ; compile_native_aiter_afp4_recovered.py:122:11
	s_cbranch_scc1 .LBB0_145
; %bb.5:                                ; %.lr.ph
	.loc	1 112 15                        ; compile_native_aiter_afp4_recovered.py:112:15
	s_abs_i32 s24, s30
	v_cvt_f32_u32_e32 v2, s24
	.loc	1 124 13                        ; compile_native_aiter_afp4_recovered.py:124:13
	s_bfe_i32 s13, s14, 0x1001e
	s_lshl_b32 s6, s14, 1
	s_lshr_b32 s13, s13, 27
	.loc	1 112 15                        ; compile_native_aiter_afp4_recovered.py:112:15
	v_rcp_f32_e32 v2, v2
	.loc	1 124 13                        ; compile_native_aiter_afp4_recovered.py:124:13
	s_add_i32 s6, s6, s13
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_b32 s13, s12, 0xc0
	.loc	1 111 39                        ; compile_native_aiter_afp4_recovered.py:111:39
	v_and_b32_e32 v4, 63, v0
	.loc	1 112 15                        ; compile_native_aiter_afp4_recovered.py:112:15
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v2, v2
	.loc	1 111 39                        ; compile_native_aiter_afp4_recovered.py:111:39
	v_or_b32_e32 v1, s13, v4
	v_lshrrev_b32_e32 v8, 1, v1
	v_or_b32_e32 v5, 0x80, v8
	.loc	1 112 15                        ; compile_native_aiter_afp4_recovered.py:112:15
	s_sub_i32 s21, 0, s24
	.loc	1 112 16 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:112:16
	v_or_b32_e32 v3, v14, v5
	.loc	1 112 15                        ; compile_native_aiter_afp4_recovered.py:112:15
	v_mul_lo_u32 v6, s21, v2
	v_add_u32_e32 v3, v3, v13
	v_mul_hi_u32 v6, v2, v6
	v_xor_b32_e32 v3, v3, v13
	v_add_u32_e32 v9, v2, v6
	v_mul_hi_u32 v2, v3, v9
	v_mul_lo_u32 v2, v2, s24
	v_sub_u32_e32 v2, v3, v2
	v_subrev_u32_e32 v3, s24, v2
	v_cmp_le_u32_e32 vcc, s24, v2
	s_load_dwordx2 s[82:83], s[0:1], 0x38
	s_load_dword s23, s[0:1], 0x40
	s_load_dwordx4 s[16:19], s[0:1], 0x50
	v_cndmask_b32_e32 v2, v2, v3, vcc
	v_subrev_u32_e32 v3, s24, v2
	v_cmp_le_u32_e32 vcc, s24, v2
	.loc	1 117 67 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:117:67
	v_and_b32_e32 v22, 1, v0
	.loc	1 118 67                        ; compile_native_aiter_afp4_recovered.py:118:67
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v6, s19, v22
	.loc	1 112 15                        ; compile_native_aiter_afp4_recovered.py:112:15
	v_cndmask_b32_e32 v2, v2, v3, vcc
	v_xor_b32_e32 v2, v2, v13
	v_sub_u32_e32 v2, v2, v13
	.loc	1 118 35                        ; compile_native_aiter_afp4_recovered.py:118:35
	v_mul_lo_u32 v2, v2, s18
	.loc	1 118 20 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:118:20
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[10:11], 0, v[2:3]
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[6:7]
	v_accvgpr_write_b32 a71, v3
	v_accvgpr_write_b32 a70, v2
	.loc	1 112 16 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:112:16
	v_or_b32_e32 v2, v14, v8
	.loc	1 112 15 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:112:15
	v_add_u32_e32 v2, v2, v13
	v_xor_b32_e32 v2, v2, v13
	v_mul_hi_u32 v3, v2, v9
	v_mul_lo_u32 v3, v3, s24
	v_sub_u32_e32 v2, v2, v3
	v_subrev_u32_e32 v3, s24, v2
	v_cmp_le_u32_e32 vcc, s24, v2
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	s_abs_i32 s21, s88
	v_cvt_f32_u32_e32 v10, s21
	.loc	1 112 15                        ; compile_native_aiter_afp4_recovered.py:112:15
	v_cndmask_b32_e32 v2, v2, v3, vcc
	v_subrev_u32_e32 v3, s24, v2
	v_cmp_le_u32_e32 vcc, s24, v2
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_rcp_f32_e32 v10, v10
	.loc	1 124 13                        ; compile_native_aiter_afp4_recovered.py:124:13
	s_ashr_i32 s84, s6, 5
	.loc	1 112 15                        ; compile_native_aiter_afp4_recovered.py:112:15
	v_cndmask_b32_e32 v2, v2, v3, vcc
	v_xor_b32_e32 v2, v2, v13
	v_sub_u32_e32 v2, v2, v13
	.loc	1 118 35                        ; compile_native_aiter_afp4_recovered.py:118:35
	v_mul_lo_u32 v2, v2, s18
	.loc	1 118 20 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:118:20
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[10:11], 0, v[2:3]
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[6:7]
	v_accvgpr_write_b32 a79, v3
	v_accvgpr_write_b32 a78, v2
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_mul_f32_e32 v3, 0x4f7ffffe, v10
	v_cvt_u32_f32_e32 v3, v3
	s_sub_i32 s10, 0, s21
	.loc	1 111 16 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:111:16
	v_or_b32_e32 v2, v12, v5
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v2, v2, v11
	v_mul_lo_u32 v5, s10, v3
	v_mul_hi_u32 v5, v3, v5
	v_xor_b32_e32 v2, v2, v11
	v_add_u32_e32 v5, v3, v5
	v_mul_hi_u32 v3, v2, v5
	v_mul_lo_u32 v3, v3, s21
	v_sub_u32_e32 v2, v2, v3
	v_subrev_u32_e32 v3, s21, v2
	v_cmp_le_u32_e32 vcc, s21, v2
	.loc	1 117 67 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:117:67
	v_mul_lo_u32 v6, s17, v22
	.loc	1 117 20 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:117:20
	v_ashrrev_i32_e32 v7, 31, v6
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_cndmask_b32_e32 v2, v2, v3, vcc
	v_subrev_u32_e32 v3, s21, v2
	v_cmp_le_u32_e32 vcc, s21, v2
	.loc	1 139 25                        ; compile_native_aiter_afp4_recovered.py:139:25
	s_lshl_b32 s6, s19, 1
	.loc	1 138 25                        ; compile_native_aiter_afp4_recovered.py:138:25
	s_lshl_b32 s12, s17, 1
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_cndmask_b32_e32 v2, v2, v3, vcc
	v_xor_b32_e32 v2, v2, v11
	v_sub_u32_e32 v2, v2, v11
	.loc	1 117 35                        ; compile_native_aiter_afp4_recovered.py:117:35
	v_mul_lo_u32 v2, v2, s16
	.loc	1 117 20 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:117:20
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[8:9], 0, v[2:3]
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[6:7]
	v_accvgpr_write_b32 a73, v3
	v_accvgpr_write_b32 a72, v2
	.loc	1 111 16 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:16
	v_or_b32_e32 v2, v12, v8
	.loc	1 111 15 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v2, v2, v11
	v_xor_b32_e32 v2, v2, v11
	v_mul_hi_u32 v3, v2, v5
	v_mul_lo_u32 v3, v3, s21
	v_sub_u32_e32 v2, v2, v3
	v_subrev_u32_e32 v3, s21, v2
	v_cmp_le_u32_e32 vcc, s21, v2
	.loc	1 137 19 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:137:19
	s_lshl_b32 s20, s83, 5
	.loc	1 136 19                        ; compile_native_aiter_afp4_recovered.py:136:19
	s_lshl_b32 s22, s82, 5
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_cndmask_b32_e32 v2, v2, v3, vcc
	v_subrev_u32_e32 v3, s21, v2
	v_cmp_le_u32_e32 vcc, s21, v2
	v_lshlrev_b32_e32 v20, 3, v0
	v_lshlrev_b32_e32 v19, 7, v0
	v_cndmask_b32_e32 v2, v2, v3, vcc
	v_xor_b32_e32 v2, v2, v11
	v_sub_u32_e32 v2, v2, v11
	.loc	1 117 35                        ; compile_native_aiter_afp4_recovered.py:117:35
	v_mul_lo_u32 v2, v2, s16
	.loc	1 117 20 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:117:20
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[2:3], s[8:9], 0, v[2:3]
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[6:7]
	v_accvgpr_write_b32 a75, v3
	v_accvgpr_write_b32 a74, v2
	.loc	1 112 16 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:112:16
	v_or_b32_e32 v2, v14, v1
	.loc	1 112 15 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:112:15
	v_add_u32_e32 v2, v2, v13
	v_xor_b32_e32 v2, v2, v13
	v_mul_hi_u32 v3, v2, v9
	v_mul_lo_u32 v3, v3, s24
	v_sub_u32_e32 v2, v2, v3
	v_subrev_u32_e32 v3, s24, v2
	v_cmp_le_u32_e32 vcc, s24, v2
	.loc	1 116 22 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s8, s83, 31
	.loc	1 116 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s9, s8, 31
	.loc	1 112 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:112:15
	v_cndmask_b32_e32 v2, v2, v3, vcc
	v_subrev_u32_e32 v3, s24, v2
	v_cmp_le_u32_e32 vcc, s24, v2
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_add_u32 s10, s4, s8
	s_addc_u32 s11, s5, s9
	.loc	1 112 15                        ; compile_native_aiter_afp4_recovered.py:112:15
	v_cndmask_b32_e32 v2, v2, v3, vcc
	v_xor_b32_e32 v2, v2, v13
	v_sub_u32_e32 v2, v2, v13
	.loc	1 116 52                        ; compile_native_aiter_afp4_recovered.py:116:52
	v_mul_lo_u32 v2, v2, s23
	.loc	1 116 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:116:14
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[242:243], s[10:11], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s10, s83, 30
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s11, s10, 31
	s_add_u32 s16, s4, s10
	s_addc_u32 s17, s5, s11
	v_lshl_add_u64 v[244:245], s[16:17], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s16, s83, 29
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s17, s16, 31
	s_add_u32 s18, s4, s16
	s_addc_u32 s19, s5, s17
	v_lshl_add_u64 v[6:7], s[18:19], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s18, s83, 28
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s19, s18, 31
	s_add_u32 s24, s4, s18
	v_accvgpr_write_b32 a61, v7
	s_addc_u32 s25, s5, s19
	v_accvgpr_write_b32 a60, v6
	v_lshl_add_u64 v[6:7], s[24:25], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s24, s83, 27
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s25, s24, 31
	s_add_u32 s26, s4, s24
	v_accvgpr_write_b32 a119, v7
	s_addc_u32 s27, s5, s25
	v_accvgpr_write_b32 a118, v6
	v_lshl_add_u64 v[6:7], s[26:27], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s26, s83, 26
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s27, s26, 31
	s_add_u32 s28, s4, s26
	v_accvgpr_write_b32 a59, v7
	s_addc_u32 s29, s5, s27
	v_accvgpr_write_b32 a58, v6
	v_lshl_add_u64 v[6:7], s[28:29], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s28, s83, 25
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s29, s28, 31
	s_add_u32 s34, s4, s28
	v_accvgpr_write_b32 a121, v7
	s_addc_u32 s35, s5, s29
	v_accvgpr_write_b32 a120, v6
	v_lshl_add_u64 v[6:7], s[34:35], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s34, s83, 24
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s35, s34, 31
	s_add_u32 s36, s4, s34
	v_accvgpr_write_b32 a57, v7
	s_addc_u32 s37, s5, s35
	v_accvgpr_write_b32 a56, v6
	v_lshl_add_u64 v[6:7], s[36:37], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s36, s83, 23
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s37, s36, 31
	s_add_u32 s38, s4, s36
	v_accvgpr_write_b32 a123, v7
	s_addc_u32 s39, s5, s37
	v_accvgpr_write_b32 a122, v6
	v_lshl_add_u64 v[6:7], s[38:39], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s38, s83, 22
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s39, s38, 31
	s_add_u32 s40, s4, s38
	v_accvgpr_write_b32 a55, v7
	s_addc_u32 s41, s5, s39
	v_accvgpr_write_b32 a54, v6
	v_lshl_add_u64 v[6:7], s[40:41], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s40, s83, 21
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s41, s40, 31
	s_add_u32 s42, s4, s40
	v_accvgpr_write_b32 a125, v7
	s_addc_u32 s43, s5, s41
	v_accvgpr_write_b32 a124, v6
	v_lshl_add_u64 v[6:7], s[42:43], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s42, s83, 20
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s43, s42, 31
	s_add_u32 s44, s4, s42
	v_accvgpr_write_b32 a53, v7
	s_addc_u32 s45, s5, s43
	v_accvgpr_write_b32 a52, v6
	v_lshl_add_u64 v[6:7], s[44:45], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s44, s83, 19
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s45, s44, 31
	s_add_u32 s46, s4, s44
	v_accvgpr_write_b32 a127, v7
	s_addc_u32 s47, s5, s45
	v_accvgpr_write_b32 a126, v6
	v_lshl_add_u64 v[6:7], s[46:47], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s46, s83, 18
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s47, s46, 31
	s_add_u32 s48, s4, s46
	v_accvgpr_write_b32 a51, v7
	s_addc_u32 s49, s5, s47
	v_accvgpr_write_b32 a50, v6
	v_lshl_add_u64 v[6:7], s[48:49], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s48, s83, 17
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s49, s48, 31
	s_add_u32 s50, s4, s48
	v_accvgpr_write_b32 a129, v7
	s_addc_u32 s51, s5, s49
	v_accvgpr_write_b32 a128, v6
	v_lshl_add_u64 v[6:7], s[50:51], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_lshl_b32 s50, s83, 4
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s51, s50, 31
	s_add_u32 s52, s4, s50
	v_accvgpr_write_b32 a49, v7
	s_addc_u32 s53, s5, s51
	v_accvgpr_write_b32 a48, v6
	v_lshl_add_u64 v[6:7], s[52:53], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s52, s83, 15
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s53, s52, 31
	s_add_u32 s54, s4, s52
	v_accvgpr_write_b32 a191, v7
	s_addc_u32 s55, s5, s53
	v_accvgpr_write_b32 a190, v6
	v_lshl_add_u64 v[6:7], s[54:55], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s54, s83, 14
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s55, s54, 31
	s_add_u32 s56, s4, s54
	v_accvgpr_write_b32 a47, v7
	s_addc_u32 s57, s5, s55
	v_accvgpr_write_b32 a46, v6
	v_lshl_add_u64 v[6:7], s[56:57], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s56, s83, 13
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s57, s56, 31
	s_add_u32 s58, s4, s56
	v_accvgpr_write_b32 a193, v7
	s_addc_u32 s59, s5, s57
	v_accvgpr_write_b32 a192, v6
	v_lshl_add_u64 v[6:7], s[58:59], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s58, s83, 12
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s59, s58, 31
	s_add_u32 s60, s4, s58
	v_accvgpr_write_b32 a45, v7
	s_addc_u32 s61, s5, s59
	v_accvgpr_write_b32 a44, v6
	v_lshl_add_u64 v[6:7], s[60:61], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s60, s83, 11
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s61, s60, 31
	s_add_u32 s62, s4, s60
	v_accvgpr_write_b32 a195, v7
	s_addc_u32 s63, s5, s61
	v_accvgpr_write_b32 a194, v6
	v_lshl_add_u64 v[6:7], s[62:63], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s62, s83, 10
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s63, s62, 31
	s_add_u32 s64, s4, s62
	v_accvgpr_write_b32 a43, v7
	s_addc_u32 s65, s5, s63
	v_accvgpr_write_b32 a42, v6
	v_lshl_add_u64 v[6:7], s[64:65], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s64, s83, 9
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s65, s64, 31
	s_add_u32 s66, s4, s64
	v_accvgpr_write_b32 a197, v7
	s_addc_u32 s67, s5, s65
	v_accvgpr_write_b32 a196, v6
	v_lshl_add_u64 v[6:7], s[66:67], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_lshl_b32 s66, s83, 3
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s67, s66, 31
	s_add_u32 s68, s4, s66
	v_accvgpr_write_b32 a41, v7
	s_addc_u32 s69, s5, s67
	v_accvgpr_write_b32 a40, v6
	v_lshl_add_u64 v[6:7], s[68:69], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s68, s83, 7
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s69, s68, 31
	s_add_u32 s70, s4, s68
	v_accvgpr_write_b32 a199, v7
	s_addc_u32 s71, s5, s69
	v_accvgpr_write_b32 a198, v6
	v_lshl_add_u64 v[6:7], s[70:71], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s70, s83, 6
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s71, s70, 31
	s_add_u32 s72, s4, s70
	v_accvgpr_write_b32 a39, v7
	s_addc_u32 s73, s5, s71
	v_accvgpr_write_b32 a38, v6
	v_lshl_add_u64 v[6:7], s[72:73], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s72, s83, 5
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s73, s72, 31
	s_add_u32 s74, s4, s72
	v_accvgpr_write_b32 a201, v7
	s_addc_u32 s75, s5, s73
	v_accvgpr_write_b32 a200, v6
	v_lshl_add_u64 v[6:7], s[74:75], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_lshl_b32 s74, s83, 2
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s75, s74, 31
	s_add_u32 s76, s4, s74
	v_accvgpr_write_b32 a37, v7
	s_addc_u32 s77, s5, s75
	v_accvgpr_write_b32 a36, v6
	v_lshl_add_u64 v[6:7], s[76:77], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_mul_i32 s76, s83, 3
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s77, s76, 31
	s_add_u32 s78, s4, s76
	v_accvgpr_write_b32 a203, v7
	s_addc_u32 s79, s5, s77
	v_accvgpr_write_b32 a202, v6
	v_lshl_add_u64 v[6:7], s[78:79], 0, v[2:3]
	.loc	1 116 22                        ; compile_native_aiter_afp4_recovered.py:116:22
	s_lshl_b32 s78, s83, 1
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_ashr_i32 s79, s78, 31
	s_add_u32 s80, s4, s78
	v_accvgpr_write_b32 a35, v7
	s_addc_u32 s81, s5, s79
	v_accvgpr_write_b32 a34, v6
	v_lshl_add_u64 v[6:7], s[80:81], 0, v[2:3]
	s_ashr_i32 s81, s83, 31
	s_add_u32 s86, s4, s83
	v_accvgpr_write_b32 a205, v7
	s_addc_u32 s87, s5, s81
	v_accvgpr_write_b32 a204, v6
	v_lshl_add_u64 v[6:7], s[86:87], 0, v[2:3]
	v_lshl_add_u64 v[252:253], s[4:5], 0, v[2:3]
	.loc	1 111 39 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:39
	v_lshrrev_b32_e32 v2, 5, v1
	v_accvgpr_write_b32 a33, v7
	v_or_b32_e32 v2, v12, v2
	v_accvgpr_write_b32 a32, v6
	.loc	1 111 15 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v6, v2, v11
	v_add_u32_e32 v2, 0xf8, v6
	v_xor_b32_e32 v2, v2, v11
	v_mul_hi_u32 v3, v2, v5
	v_mul_lo_u32 v3, v3, s21
	v_sub_u32_e32 v2, v2, v3
	v_subrev_u32_e32 v3, s21, v2
	v_cmp_le_u32_e32 vcc, s21, v2
	v_add_u32_e32 v7, 0xf0, v6
	v_xor_b32_e32 v7, v7, v11
	v_cndmask_b32_e32 v2, v2, v3, vcc
	v_subrev_u32_e32 v3, s21, v2
	v_cmp_le_u32_e32 vcc, s21, v2
	v_and_b32_e32 v20, 0x78, v20
	v_and_b32_e32 v21, 16, v0
	v_cndmask_b32_e32 v2, v2, v3, vcc
	v_xor_b32_e32 v2, v2, v11
	v_sub_u32_e32 v2, v2, v11
	.loc	1 115 22 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v2, v2, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[2:3]
	.loc	1 115 53                        ; compile_native_aiter_afp4_recovered.py:115:53
	v_mul_lo_u32 v2, s82, v15
	.loc	1 115 14                        ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v3, 31, v2
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	v_accvgpr_write_b32 a77, v9
	v_accvgpr_write_b32 a76, v8
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_add_u32_e32 v254, 0, v1
	v_mov_b32_e32 v10, 0x770
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_bitop3_b32 v10, s13, v10, v4 bitop3:0x36
	v_mov_b32_e32 v12, 0x990
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0xe8, v6
	v_accvgpr_write_b32 a81, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a80, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_bitop3_b32 v12, s13, v12, v4 bitop3:0x36
	v_mov_b32_e32 v13, 0xaa0
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_accvgpr_write_b32 a66, v14
	v_bitop3_b32 v13, s13, v13, v4 bitop3:0x36
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0xe0, v6
	v_accvgpr_write_b32 a83, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a82, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v14, 0xbb0
	v_bitop3_b32 v14, s13, v14, v4 bitop3:0x36
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v15, 0xcc0
	v_accvgpr_write_b32 a68, v16
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0xd8, v6
	.loc	1 115 14                        ; compile_native_aiter_afp4_recovered.py:115:14
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_xor_b32_e32 v7, v7, v11
	.loc	1 115 14                        ; compile_native_aiter_afp4_recovered.py:115:14
	v_lshl_add_u64 v[192:193], v[8:9], 0, v[2:3]
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_bitop3_b32 v15, s13, v15, v4 bitop3:0x36
	v_mov_b32_e32 v17, 0xee0
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_bitop3_b32 v17, s13, v17, v4 bitop3:0x36
	v_mov_b32_e32 v18, 0xff0
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0xd0, v6
	.loc	1 115 14                        ; compile_native_aiter_afp4_recovered.py:115:14
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_xor_b32_e32 v7, v7, v11
	.loc	1 115 14                        ; compile_native_aiter_afp4_recovered.py:115:14
	v_lshl_add_u64 v[194:195], v[8:9], 0, v[2:3]
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_bitop3_b32 v18, s13, v18, v4 bitop3:0x36
	s_lshr_b32 s5, s31, 6
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v208, 0
	.loc	1 116 14                        ; compile_native_aiter_afp4_recovered.py:116:14
	s_mov_b32 s80, s83
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0xc8, v6
	.loc	1 115 14                        ; compile_native_aiter_afp4_recovered.py:115:14
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_xor_b32_e32 v7, v7, v11
	.loc	1 115 14                        ; compile_native_aiter_afp4_recovered.py:115:14
	v_lshl_add_u64 v[196:197], v[8:9], 0, v[2:3]
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_accvgpr_write_b32 a69, v22
	s_ashr_i32 s23, s22, 31
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	s_movk_i32 s4, 0xff
	v_mov_b32_e32 v209, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0xc0, v6
	.loc	1 115 14                        ; compile_native_aiter_afp4_recovered.py:115:14
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_xor_b32_e32 v7, v7, v11
	.loc	1 115 14                        ; compile_native_aiter_afp4_recovered.py:115:14
	v_lshl_add_u64 v[198:199], v[8:9], 0, v[2:3]
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v210, v208
	v_mov_b32_e32 v211, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v212, v208
	v_mov_b32_e32 v213, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0xb8, v6
	.loc	1 115 14                        ; compile_native_aiter_afp4_recovered.py:115:14
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_xor_b32_e32 v7, v7, v11
	.loc	1 115 14                        ; compile_native_aiter_afp4_recovered.py:115:14
	v_lshl_add_u64 v[200:201], v[8:9], 0, v[2:3]
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v214, v208
	v_mov_b32_e32 v215, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v216, v208
	v_mov_b32_e32 v217, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0xb0, v6
	.loc	1 115 14                        ; compile_native_aiter_afp4_recovered.py:115:14
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_xor_b32_e32 v7, v7, v11
	.loc	1 115 14                        ; compile_native_aiter_afp4_recovered.py:115:14
	v_lshl_add_u64 v[202:203], v[8:9], 0, v[2:3]
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v218, v208
	v_mov_b32_e32 v219, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v220, v208
	v_mov_b32_e32 v221, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0xa8, v6
	.loc	1 115 14                        ; compile_native_aiter_afp4_recovered.py:115:14
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_xor_b32_e32 v7, v7, v11
	.loc	1 115 14                        ; compile_native_aiter_afp4_recovered.py:115:14
	v_lshl_add_u64 v[204:205], v[8:9], 0, v[2:3]
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v222, v208
	v_mov_b32_e32 v223, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v23, v208
	v_mov_b32_e32 v24, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0xa0, v6
	.loc	1 115 14                        ; compile_native_aiter_afp4_recovered.py:115:14
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_xor_b32_e32 v7, v7, v11
	.loc	1 115 14                        ; compile_native_aiter_afp4_recovered.py:115:14
	v_lshl_add_u64 v[206:207], v[8:9], 0, v[2:3]
	.loc	1 111 15                        ; compile_native_aiter_afp4_recovered.py:111:15
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v25, v208
	v_mov_b32_e32 v26, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v27, v208
	v_mov_b32_e32 v28, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0x98, v6
	v_accvgpr_write_b32 a131, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a130, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v29, v208
	v_mov_b32_e32 v30, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v31, v208
	v_mov_b32_e32 v224, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0x90, v6
	v_accvgpr_write_b32 a133, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a132, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v225, v208
	v_mov_b32_e32 v226, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v227, v208
	v_mov_b32_e32 v228, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0x88, v6
	v_accvgpr_write_b32 a135, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a134, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v229, v208
	v_mov_b32_e32 v230, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v231, v208
	v_mov_b32_e32 v232, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0x80, v6
	v_accvgpr_write_b32 a137, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a136, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v233, v208
	v_mov_b32_e32 v234, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v235, v208
	v_mov_b32_e32 v236, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0x78, v6
	v_accvgpr_write_b32 a139, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a138, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v237, v208
	v_mov_b32_e32 v238, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v239, v208
	v_mov_b32_e32 v176, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0x70, v6
	v_accvgpr_write_b32 a141, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a140, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v177, v208
	v_mov_b32_e32 v178, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v179, v208
	v_mov_b32_e32 v180, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0x68, v6
	v_accvgpr_write_b32 a143, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a142, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v181, v208
	v_mov_b32_e32 v182, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v183, v208
	v_mov_b32_e32 v184, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0x60, v6
	v_accvgpr_write_b32 a145, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a144, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v185, v208
	v_mov_b32_e32 v186, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v187, v208
	v_mov_b32_e32 v188, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0x58, v6
	v_accvgpr_write_b32 a147, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a146, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v189, v208
	v_mov_b32_e32 v190, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v191, v208
	v_mov_b32_e32 v160, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0x50, v6
	v_accvgpr_write_b32 a149, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a148, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v161, v208
	v_mov_b32_e32 v162, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v163, v208
	v_mov_b32_e32 v164, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 0x48, v6
	v_accvgpr_write_b32 a151, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a150, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v165, v208
	v_mov_b32_e32 v166, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v167, v208
	v_mov_b32_e32 v168, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 64, v6
	v_accvgpr_write_b32 a153, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a152, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v169, v208
	v_mov_b32_e32 v170, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v171, v208
	v_mov_b32_e32 v172, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 56, v6
	v_accvgpr_write_b32 a155, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a154, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v173, v208
	v_mov_b32_e32 v174, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v175, v208
	v_mov_b32_e32 v144, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 48, v6
	v_accvgpr_write_b32 a157, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a156, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v145, v208
	v_mov_b32_e32 v146, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v147, v208
	v_mov_b32_e32 v148, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 40, v6
	v_accvgpr_write_b32 a159, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a158, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v149, v208
	v_mov_b32_e32 v150, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v151, v208
	v_mov_b32_e32 v152, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 32, v6
	v_accvgpr_write_b32 a161, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a160, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v153, v208
	v_mov_b32_e32 v154, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v155, v208
	v_mov_b32_e32 v156, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 24, v6
	v_accvgpr_write_b32 a163, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a162, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v157, v208
	v_mov_b32_e32 v158, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v159, v208
	v_mov_b32_e32 v128, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 16, v6
	v_accvgpr_write_b32 a167, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a166, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v129, v208
	v_mov_b32_e32 v130, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mov_b32_e32 v131, v208
	v_mov_b32_e32 v132, v208
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[2:3]
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_add_u32_e32 v7, 8, v6
	v_accvgpr_write_b32 a169, v9
	v_xor_b32_e32 v7, v7, v11
	v_accvgpr_write_b32 a168, v8
	v_mul_hi_u32 v8, v7, v5
	v_mul_lo_u32 v8, v8, s21
	v_sub_u32_e32 v7, v7, v8
	v_xor_b32_e32 v6, v6, v11
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_mul_hi_u32 v5, v6, v5
	v_mul_lo_u32 v5, v5, s21
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_subrev_u32_e32 v8, s21, v7
	v_cmp_le_u32_e32 vcc, s21, v7
	v_sub_u32_e32 v5, v6, v5
	v_subrev_u32_e32 v6, s21, v5
	v_cndmask_b32_e32 v7, v7, v8, vcc
	v_cmp_le_u32_e32 vcc, s21, v5
	v_xor_b32_e32 v7, v7, v11
	v_sub_u32_e32 v7, v7, v11
	v_cndmask_b32_e32 v5, v5, v6, vcc
	v_subrev_u32_e32 v6, s21, v5
	v_cmp_le_u32_e32 vcc, s21, v5
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v8, v7, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v9, 31, v8
	.loc	1 111 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:111:15
	v_cndmask_b32_e32 v5, v5, v6, vcc
	v_xor_b32_e32 v5, v5, v11
	v_sub_u32_e32 v5, v5, v11
	.loc	1 115 22                        ; compile_native_aiter_afp4_recovered.py:115:22
	v_mul_lo_u32 v6, v5, s15
	.loc	1 115 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:115:14
	v_ashrrev_i32_e32 v7, 31, v6
	v_lshl_add_u64 v[8:9], s[2:3], 0, v[8:9]
	v_lshl_add_u64 v[6:7], s[2:3], 0, v[6:7]
	v_lshl_add_u64 v[248:249], v[8:9], 0, v[2:3]
	v_lshl_add_u64 v[250:251], v[6:7], 0, v[2:3]
	v_lshlrev_b32_e32 v3, 5, v0
	v_bfe_i32 v5, v0, 3, 1
	s_movk_i32 s3, 0x1700
	v_lshlrev_b32_e32 v0, 1, v0
	v_and_b32_e32 v3, 0x2e0, v3
	v_and_b32_e32 v5, 0x110, v5
	v_lshrrev_b32_e32 v6, 1, v16
	v_and_or_b32 v19, v19, s3, v20
	v_and_b32_e32 v0, 0x7c, v0
	v_bitop3_b32 v3, v5, v6, v3 bitop3:0x36
	s_lshl3_add_u32 s2, s31, 0
	v_mov_b32_e32 v5, 0x220
	v_bitop3_b32 v19, s33, v19, v21 bitop3:0x36
	v_lshlrev_b32_e32 v21, 7, v22
	s_lshl_b32 s3, s7, 2
	v_add_u32_e32 v0, 0, v0
	v_bitop3_b32 v5, s13, v5, v4 bitop3:0x36
	v_mov_b32_e32 v6, 0x330
	v_add3_u32 v0, v0, v21, s3
	v_lshlrev_b32_e32 v21, 2, v1
	v_add_u32_e32 v1, s2, v3
	v_bitop3_b32 v6, s13, v6, v4 bitop3:0x36
	v_mov_b32_e32 v7, 0x440
	v_accvgpr_write_b32 a170, v1
	v_add_u32_e32 v1, 0, v5
	v_bitop3_b32 v7, s13, v7, v4 bitop3:0x36
	v_mov_b32_e32 v8, 0x550
	v_accvgpr_write_b32 a171, v1
	v_add_u32_e32 v1, 0, v6
	v_bitop3_b32 v8, s13, v8, v4 bitop3:0x36
	v_mov_b32_e32 v9, 0x660
	v_accvgpr_write_b32 a172, v1
	v_add_u32_e32 v1, 0, v7
	v_bitop3_b32 v9, s13, v9, v4 bitop3:0x36
	v_accvgpr_write_b32 a173, v1
	v_add_u32_e32 v1, 0, v8
	v_mov_b32_e32 v11, 0x880
	v_accvgpr_write_b32 a174, v1
	v_add_u32_e32 v1, 0, v9
	v_bitop3_b32 v11, s13, v11, v4 bitop3:0x36
	v_accvgpr_write_b32 a175, v1
	v_add_u32_e32 v1, 0, v10
	v_accvgpr_write_b32 a176, v1
	v_add_u32_e32 v1, 0, v11
	v_accvgpr_write_b32 a177, v1
	v_add_u32_e32 v1, 0, v12
	v_accvgpr_write_b32 a178, v1
	v_add_u32_e32 v1, 0, v13
	v_mov_b32_e32 v16, 0xdd0
	v_accvgpr_write_b32 a179, v1
	v_add_u32_e32 v1, 0, v14
	v_bitop3_b32 v16, s13, v16, v4 bitop3:0x36
	v_accvgpr_write_b32 a180, v1
	v_add_u32_e32 v1, 0, v15
	v_accvgpr_write_b32 a181, v1
	v_add_u32_e32 v1, 0, v16
	v_mov_b32_e32 v2, 0x110
	v_accvgpr_write_b32 a182, v1
	v_add_u32_e32 v1, 0, v17
	v_bitop3_b32 v2, s13, v2, v4 bitop3:0x36
	s_lshl_b32 s3, s31, 1
	v_lshl_add_u32 v4, v4, 2, 0
	v_accvgpr_write_b32 a183, v1
	v_add_u32_e32 v1, 0, v18
	v_add_u32_e32 v0, s5, v0
	v_xor_b32_e32 v20, 64, v19
	v_and_b32_e32 v21, 0x1fc, v21
	v_accvgpr_write_b32 a184, v1
	v_add_u32_e32 v1, 0, v19
	v_accvgpr_write_b32 a187, v0
	v_add_u32_e32 v0, s3, v4
	v_accvgpr_write_b32 a185, v1
	v_add_u32_e32 v1, 0, v20
	v_accvgpr_write_b32 a188, v0
	v_add_u32_e32 v0, 0, v21
	v_mov_b32_e32 v15, v208
	v_add_u32_e32 v240, 0, v2
	v_accvgpr_write_b32 a186, v1
	v_accvgpr_write_b32 a189, v0
	v_mov_b32_e32 v0, v208
	v_mov_b32_e32 v1, v208
	v_mov_b32_e32 v2, v208
	v_mov_b32_e32 v3, v208
	v_mov_b32_e32 v4, v208
	v_mov_b32_e32 v5, v208
	v_mov_b32_e32 v6, v208
	v_mov_b32_e32 v7, v208
	v_mov_b32_e32 v8, v208
	v_mov_b32_e32 v9, v208
	v_mov_b32_e32 v10, v208
	v_mov_b32_e32 v11, v208
	v_mov_b32_e32 v12, v208
	v_mov_b32_e32 v13, v208
	v_mov_b32_e32 v14, v208
	v_accvgpr_write_b32 a117, v15
	v_accvgpr_write_b32 a99, v15
	s_ashr_i32 s21, s20, 31
	s_ashr_i32 s13, s12, 31
	s_ashr_i32 s7, s6, 31
	s_mov_b32 s5, 0x7050604
	s_mov_b32 s15, 0
	v_mov_b32_e32 v16, v208
	v_mov_b32_e32 v17, v208
	v_mov_b32_e32 v18, v208
	v_mov_b32_e32 v19, v208
	v_mov_b32_e32 v20, v208
	v_mov_b32_e32 v21, v208
	v_mov_b32_e32 v22, v208
	v_accvgpr_write_b32 a116, v14
	v_accvgpr_write_b32 a115, v13
	v_accvgpr_write_b32 a114, v12
	v_accvgpr_write_b32 a113, v11
	v_accvgpr_write_b32 a112, v10
	v_accvgpr_write_b32 a111, v9
	v_accvgpr_write_b32 a110, v8
	v_accvgpr_write_b32 a109, v7
	v_accvgpr_write_b32 a108, v6
	v_accvgpr_write_b32 a107, v5
	v_accvgpr_write_b32 a106, v4
	v_accvgpr_write_b32 a105, v3
	v_accvgpr_write_b32 a104, v2
	v_accvgpr_write_b32 a103, v1
	v_accvgpr_write_b32 a102, v0
	v_accvgpr_write_b32 a98, v14
	v_accvgpr_write_b32 a97, v13
	v_accvgpr_write_b32 a96, v12
	v_accvgpr_write_b32 a95, v11
	v_accvgpr_write_b32 a94, v10
	v_accvgpr_write_b32 a93, v9
	v_accvgpr_write_b32 a92, v8
	v_accvgpr_write_b32 a91, v7
	v_accvgpr_write_b32 a90, v6
	v_accvgpr_write_b32 a89, v5
	v_accvgpr_write_b32 a88, v4
	v_accvgpr_write_b32 a87, v3
	v_accvgpr_write_b32 a86, v2
	v_accvgpr_write_b32 a85, v1
	v_accvgpr_write_b32 a84, v0
	v_mov_b32_e32 v133, v208
	v_mov_b32_e32 v134, v208
	v_mov_b32_e32 v135, v208
	v_mov_b32_e32 v136, v208
	v_mov_b32_e32 v137, v208
	v_mov_b32_e32 v138, v208
	v_mov_b32_e32 v139, v208
	v_mov_b32_e32 v140, v208
	v_mov_b32_e32 v141, v208
	v_mov_b32_e32 v142, v208
	v_mov_b32_e32 v143, v208
	v_mov_b32_e32 v112, v208
	v_mov_b32_e32 v113, v208
	v_mov_b32_e32 v114, v208
	v_mov_b32_e32 v115, v208
	v_mov_b32_e32 v116, v208
	v_mov_b32_e32 v117, v208
	v_mov_b32_e32 v118, v208
	v_mov_b32_e32 v119, v208
	v_mov_b32_e32 v120, v208
	v_mov_b32_e32 v121, v208
	v_mov_b32_e32 v122, v208
	v_mov_b32_e32 v123, v208
	v_mov_b32_e32 v124, v208
	v_mov_b32_e32 v125, v208
	v_mov_b32_e32 v126, v208
	v_mov_b32_e32 v127, v208
	v_mov_b32_e32 v96, v208
	v_mov_b32_e32 v97, v208
	v_mov_b32_e32 v98, v208
	v_mov_b32_e32 v99, v208
	v_mov_b32_e32 v100, v208
	v_mov_b32_e32 v101, v208
	v_mov_b32_e32 v102, v208
	v_mov_b32_e32 v103, v208
	v_mov_b32_e32 v104, v208
	v_mov_b32_e32 v105, v208
	v_mov_b32_e32 v106, v208
	v_mov_b32_e32 v107, v208
	v_mov_b32_e32 v108, v208
	v_mov_b32_e32 v109, v208
	v_mov_b32_e32 v110, v208
	v_mov_b32_e32 v111, v208
	v_mov_b32_e32 v80, v208
	v_mov_b32_e32 v81, v208
	v_mov_b32_e32 v82, v208
	v_mov_b32_e32 v83, v208
	v_mov_b32_e32 v84, v208
	v_mov_b32_e32 v85, v208
	v_mov_b32_e32 v86, v208
	v_mov_b32_e32 v87, v208
	v_mov_b32_e32 v88, v208
	v_mov_b32_e32 v89, v208
	v_mov_b32_e32 v90, v208
	v_mov_b32_e32 v91, v208
	v_mov_b32_e32 v92, v208
	v_mov_b32_e32 v93, v208
	v_mov_b32_e32 v94, v208
	v_mov_b32_e32 v95, v208
	v_mov_b32_e32 v64, v208
	v_mov_b32_e32 v65, v208
	v_mov_b32_e32 v66, v208
	v_mov_b32_e32 v67, v208
	v_mov_b32_e32 v68, v208
	v_mov_b32_e32 v69, v208
	v_mov_b32_e32 v70, v208
	v_mov_b32_e32 v71, v208
	v_mov_b32_e32 v72, v208
	v_mov_b32_e32 v73, v208
	v_mov_b32_e32 v74, v208
	v_mov_b32_e32 v75, v208
	v_mov_b32_e32 v76, v208
	v_mov_b32_e32 v77, v208
	v_mov_b32_e32 v78, v208
	v_mov_b32_e32 v79, v208
	v_mov_b32_e32 v48, v208
	v_mov_b32_e32 v49, v208
	v_mov_b32_e32 v50, v208
	v_mov_b32_e32 v51, v208
	v_mov_b32_e32 v52, v208
	v_mov_b32_e32 v53, v208
	v_mov_b32_e32 v54, v208
	v_mov_b32_e32 v55, v208
	v_mov_b32_e32 v56, v208
	v_mov_b32_e32 v57, v208
	v_mov_b32_e32 v58, v208
	v_mov_b32_e32 v59, v208
	v_mov_b32_e32 v60, v208
	v_mov_b32_e32 v61, v208
	v_mov_b32_e32 v62, v208
	v_mov_b32_e32 v63, v208
	v_mov_b32_e32 v32, v208
	v_mov_b32_e32 v33, v208
	v_mov_b32_e32 v34, v208
	v_mov_b32_e32 v35, v208
	v_mov_b32_e32 v36, v208
	v_mov_b32_e32 v37, v208
	v_mov_b32_e32 v38, v208
	v_mov_b32_e32 v39, v208
	v_mov_b32_e32 v40, v208
	v_mov_b32_e32 v41, v208
	v_mov_b32_e32 v42, v208
	v_mov_b32_e32 v43, v208
	v_mov_b32_e32 v44, v208
	v_mov_b32_e32 v45, v208
	v_mov_b32_e32 v46, v208
	v_mov_b32_e32 v47, v208
	s_branch .LBB0_8
	.loc	1 0 14                          ; :0:14
.Ltmp21:
	.p2align	5, , 4
.LBB0_6:                                ;   in Loop: Header=BB0_8 Depth=1
	v_accvgpr_read_b32 v193, a165
	v_accvgpr_read_b32 v192, a164
	.loc	1 129 13 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a62, v[192:193], off
.LBB0_7:                                ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v192, a170
	.loc	1 128 13 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:128:13
	s_waitcnt vmcnt(63) expcnt(7) lgkmcnt(15)
	s_barrier
	s_waitcnt vmcnt(0)
	ds_write_b8 v254, a0
	ds_write_b8 v254, a2 offset:512
	ds_write_b8 v254, a4 offset:1024
	ds_write_b8 v254, a6 offset:1536
	ds_write_b8 v254, a8 offset:2048
	ds_write_b8 v254, a10 offset:2560
	ds_write_b8 v254, a12 offset:3072
	ds_write_b8 v254, a14 offset:3584
	ds_write_b8 v254, a16 offset:4096
	ds_write_b8 v254, a18 offset:4608
	ds_write_b8 v254, a20 offset:5120
	ds_write_b8 v254, a22 offset:5632
	ds_write_b8 v254, a24 offset:6144
	ds_write_b8 v254, a26 offset:6656
	ds_write_b8 v254, a28 offset:7168
	ds_write_b8 v254, a30 offset:7680
	ds_write_b8 v240, v243
	ds_write_b8 v240, a1 offset:512
	ds_write_b8 v240, a3 offset:1024
	ds_write_b8 v240, a5 offset:1536
	ds_write_b8 v240, a7 offset:2048
	ds_write_b8 v240, a9 offset:2560
	ds_write_b8 v240, a11 offset:3072
	ds_write_b8 v240, a13 offset:3584
	ds_write_b8 v240, a15 offset:4096
	ds_write_b8 v240, a17 offset:4608
	ds_write_b8 v240, a19 offset:5120
	ds_write_b8 v240, a21 offset:5632
	ds_write_b8 v240, a23 offset:6144
	ds_write_b8 v240, a25 offset:6656
	ds_write_b8 v240, a27 offset:7168
	ds_write_b8 v240, a29 offset:7680
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b128 a[16:19], v192
	ds_read_b128 a[20:23], v192 offset:2048
	ds_read_b128 a[24:27], v192 offset:4096
	ds_read_b128 a[0:3], v192 offset:6144
	v_accvgpr_read_b32 v192, a171
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b8 v192, a33
	ds_write_b8 v192, a49 offset:4096
	v_accvgpr_read_b32 v192, a172
	ds_write_b8 v192, a34
	ds_write_b8 v192, a50 offset:4096
	v_accvgpr_read_b32 v192, a173
	ds_write_b8 v192, a35
	ds_write_b8 v192, a51 offset:4096
	v_accvgpr_read_b32 v192, a174
	ds_write_b8 v192, a36
	ds_write_b8 v192, a52 offset:4096
	v_accvgpr_read_b32 v192, a175
	ds_write_b8 v192, a37
	ds_write_b8 v192, a53 offset:4096
	v_accvgpr_read_b32 v192, a176
	ds_write_b8 v192, a38
	ds_write_b8 v192, a54 offset:4096
	v_accvgpr_read_b32 v192, a177
	ds_write_b8 v192, a39
	ds_write_b8 v192, a55 offset:4096
	v_accvgpr_read_b32 v192, a178
	ds_write_b8 v192, a40
	ds_write_b8 v192, a56 offset:4096
	v_accvgpr_read_b32 v192, a179
	ds_write_b8 v192, a41
	ds_write_b8 v192, a57 offset:4096
	v_accvgpr_read_b32 v192, a180
	ds_write_b8 v192, a42
	ds_write_b8 v192, a58 offset:4096
	v_accvgpr_read_b32 v192, a181
	ds_write_b8 v192, a43
	ds_write_b8 v192, a59 offset:4096
	v_accvgpr_read_b32 v192, a182
	ds_write_b8 v192, a44
	ds_write_b8 v192, a60 offset:4096
	v_accvgpr_read_b32 v192, a183
	ds_write_b8 v192, a45
	ds_write_b8 v192, a61 offset:4096
	v_accvgpr_read_b32 v192, a184
	ds_write_b8 v192, a46
	ds_write_b8 v192, a62 offset:4096
	v_accvgpr_read_b32 v192, a185
	.loc	1 126 20                        ; compile_native_aiter_afp4_recovered.py:126:20
	v_lshlrev_b16_e32 v245, 8, v245
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	ds_write_b8 v254, a31
	ds_write_b8 v254, a47 offset:4096
	ds_write_b8 v240, a32
	ds_write_b8 v240, a48 offset:4096
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b64_tr_b8 a[28:29], v192
	ds_read_b64_tr_b8 a[30:31], v192 offset:2176
	ds_read_b64_tr_b8 a[8:9], v192 offset:128
	ds_read_b64_tr_b8 a[10:11], v192 offset:2048
	v_accvgpr_read_b32 v192, a186
	.loc	1 126 20                        ; compile_native_aiter_afp4_recovered.py:126:20
	v_bitop3_b16 v242, v242, v245, s4 bitop3:0xec
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	ds_read_b64_tr_b8 a[12:13], v192
	ds_read_b64_tr_b8 a[14:15], v192 offset:2176
	ds_read_b64_tr_b8 a[4:5], v192 offset:128
	ds_read_b64_tr_b8 a[6:7], v192 offset:2048
	v_accvgpr_read_b32 v192, a187
	v_accvgpr_read_b32 v193, a188
	.loc	1 126 20                        ; compile_native_aiter_afp4_recovered.py:126:20
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b16 v192, v242
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_read_b32 v242, v193
	.loc	1 127 20                        ; compile_native_aiter_afp4_recovered.py:127:20
	v_lshlrev_b16_e32 v241, 8, v241
	v_bitop3_b16 v241, v244, v241, s4 bitop3:0xec
	s_waitcnt lgkmcnt(0)
	s_barrier
	ds_write_b16 v192, v241
	v_accvgpr_read_b32 v192, a189
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 135 23                        ; compile_native_aiter_afp4_recovered.py:135:23
	v_perm_b32 v241, v242, v242, s5
	.loc	1 127 20                        ; compile_native_aiter_afp4_recovered.py:127:20
	ds_read_b32 v242, v192
	v_accvgpr_read_b32 v193, a169
	v_accvgpr_read_b32 v192, a168
	.loc	1 136 9                         ; compile_native_aiter_afp4_recovered.py:136:9
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	v_accvgpr_write_b32 a168, v192
	.loc	1 135 23                        ; compile_native_aiter_afp4_recovered.py:135:23
	s_waitcnt lgkmcnt(0)
	v_perm_b32 v242, v242, v242, s5
	v_accvgpr_write_b32 a169, v193
	v_accvgpr_read_b32 v193, a167
	v_mfma_scale_f32_32x32x64_f8f6f4 v[208:223], a[28:31], a[16:19], v[208:223], v242, v241 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v192, a166
	.loc	1 136 9                         ; compile_native_aiter_afp4_recovered.py:136:9
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	v_accvgpr_write_b32 a166, v192
	v_accvgpr_write_b32 a167, v193
	v_accvgpr_read_b32 v193, a163
	v_accvgpr_read_b32 v192, a162
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	.loc	1 135 23                        ; compile_native_aiter_afp4_recovered.py:135:23
	v_mfma_scale_f32_32x32x64_f8f6f4 v[16:31], a[12:15], a[16:19], v[16:31], v242, v241 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_accvgpr_write_b32 a162, v192
	v_accvgpr_write_b32 a163, v193
	v_accvgpr_read_b32 v193, a161
	v_accvgpr_read_b32 v192, a160
	.loc	1 136 9                         ; compile_native_aiter_afp4_recovered.py:136:9
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	v_accvgpr_write_b32 a160, v192
	v_accvgpr_write_b32 a161, v193
	.loc	1 135 23                        ; compile_native_aiter_afp4_recovered.py:135:23
	v_mfma_scale_f32_32x32x64_f8f6f4 v[224:239], a[8:11], a[16:19], v[224:239], v242, v241 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v193, a159
	v_accvgpr_read_b32 v192, a158
	.loc	1 136 9                         ; compile_native_aiter_afp4_recovered.py:136:9
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	.loc	1 137 9                         ; compile_native_aiter_afp4_recovered.py:137:9
	v_lshl_add_u64 v[252:253], v[252:253], 0, s[20:21]
	v_accvgpr_write_b32 a158, v192
	v_accvgpr_write_b32 a159, v193
	v_accvgpr_read_b32 v193, a157
	.loc	1 135 23                        ; compile_native_aiter_afp4_recovered.py:135:23
	v_mfma_scale_f32_32x32x64_f8f6f4 a[102:117], a[4:7], a[16:19], a[102:117], v242, v241 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v192, a156
	.loc	1 136 9                         ; compile_native_aiter_afp4_recovered.py:136:9
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	v_accvgpr_write_b32 a156, v192
	v_accvgpr_write_b32 a157, v193
	v_accvgpr_read_b32 v193, a155
	v_accvgpr_read_b32 v192, a154
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	.loc	1 135 23                        ; compile_native_aiter_afp4_recovered.py:135:23
	v_mfma_scale_f32_32x32x64_f8f6f4 a[84:99], a[28:31], a[20:23], a[84:99], v242, v241 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_accvgpr_write_b32 a154, v192
	v_accvgpr_write_b32 a155, v193
	v_accvgpr_read_b32 v193, a153
	v_accvgpr_read_b32 v192, a152
	.loc	1 136 9                         ; compile_native_aiter_afp4_recovered.py:136:9
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	v_accvgpr_write_b32 a152, v192
	v_accvgpr_write_b32 a153, v193
	.loc	1 135 23                        ; compile_native_aiter_afp4_recovered.py:135:23
	v_mfma_scale_f32_32x32x64_f8f6f4 v[176:191], a[12:15], a[20:23], v[176:191], v242, v241 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v193, a151
	v_accvgpr_read_b32 v192, a150
	.loc	1 136 9                         ; compile_native_aiter_afp4_recovered.py:136:9
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	v_accvgpr_write_b32 a150, v192
	v_accvgpr_write_b32 a151, v193
	v_accvgpr_read_b32 v193, a149
	v_accvgpr_read_b32 v192, a148
	.loc	1 135 23                        ; compile_native_aiter_afp4_recovered.py:135:23
	v_mfma_scale_f32_32x32x64_f8f6f4 v[160:175], a[8:11], a[20:23], v[160:175], v242, v241 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	.loc	1 136 9                         ; compile_native_aiter_afp4_recovered.py:136:9
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	v_accvgpr_write_b32 a148, v192
	v_accvgpr_write_b32 a149, v193
	v_accvgpr_read_b32 v193, a147
	v_accvgpr_read_b32 v192, a146
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	v_accvgpr_write_b32 a146, v192
	.loc	1 135 23                        ; compile_native_aiter_afp4_recovered.py:135:23
	v_mfma_scale_f32_32x32x64_f8f6f4 v[144:159], a[4:7], a[20:23], v[144:159], v242, v241 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_accvgpr_write_b32 a147, v193
	v_accvgpr_read_b32 v193, a145
	v_accvgpr_read_b32 v192, a144
	.loc	1 136 9                         ; compile_native_aiter_afp4_recovered.py:136:9
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	v_accvgpr_write_b32 a144, v192
	v_accvgpr_write_b32 a145, v193
	v_accvgpr_read_b32 v193, a143
	.loc	1 135 23                        ; compile_native_aiter_afp4_recovered.py:135:23
	v_mfma_scale_f32_32x32x64_f8f6f4 v[128:143], a[28:31], a[24:27], v[128:143], v242, v241 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v192, a142
	.loc	1 136 9                         ; compile_native_aiter_afp4_recovered.py:136:9
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	v_accvgpr_write_b32 a142, v192
	v_accvgpr_write_b32 a143, v193
	v_accvgpr_read_b32 v193, a141
	v_accvgpr_read_b32 v192, a140
	v_accvgpr_read_b32 v207, a129
	.loc	1 135 23                        ; compile_native_aiter_afp4_recovered.py:135:23
	v_mfma_scale_f32_32x32x64_f8f6f4 v[112:127], a[12:15], a[24:27], v[112:127], v242, v241 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	.loc	1 136 9                         ; compile_native_aiter_afp4_recovered.py:136:9
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	v_accvgpr_read_b32 v206, a128
	v_accvgpr_write_b32 a140, v192
	v_accvgpr_write_b32 a141, v193
	v_accvgpr_read_b32 v193, a139
	v_accvgpr_read_b32 v192, a138
	v_accvgpr_read_b32 v205, a127
	.loc	1 135 23                        ; compile_native_aiter_afp4_recovered.py:135:23
	v_mfma_scale_f32_32x32x64_f8f6f4 v[96:111], a[8:11], a[24:27], v[96:111], v242, v241 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	.loc	1 136 9                         ; compile_native_aiter_afp4_recovered.py:136:9
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	v_accvgpr_read_b32 v204, a126
	v_accvgpr_write_b32 a138, v192
	v_accvgpr_write_b32 a139, v193
	v_accvgpr_read_b32 v193, a137
	v_accvgpr_read_b32 v192, a136
	v_accvgpr_read_b32 v203, a125
	.loc	1 135 23                        ; compile_native_aiter_afp4_recovered.py:135:23
	v_mfma_scale_f32_32x32x64_f8f6f4 v[80:95], a[4:7], a[24:27], v[80:95], v242, v241 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	.loc	1 136 9                         ; compile_native_aiter_afp4_recovered.py:136:9
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	v_accvgpr_read_b32 v202, a124
	v_accvgpr_write_b32 a136, v192
	v_accvgpr_write_b32 a137, v193
	v_accvgpr_read_b32 v193, a135
	v_accvgpr_read_b32 v192, a134
	v_accvgpr_read_b32 v201, a123
	.loc	1 135 23                        ; compile_native_aiter_afp4_recovered.py:135:23
	v_mfma_scale_f32_32x32x64_f8f6f4 v[64:79], a[28:31], a[0:3], v[64:79], v242, v241 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	.loc	1 136 9                         ; compile_native_aiter_afp4_recovered.py:136:9
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	v_accvgpr_read_b32 v200, a122
	v_accvgpr_write_b32 a134, v192
	v_accvgpr_write_b32 a135, v193
	v_accvgpr_read_b32 v193, a133
	v_accvgpr_read_b32 v192, a132
	v_accvgpr_read_b32 v199, a121
	.loc	1 135 23                        ; compile_native_aiter_afp4_recovered.py:135:23
	v_mfma_scale_f32_32x32x64_f8f6f4 v[48:63], a[12:15], a[0:3], v[48:63], v242, v241 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	.loc	1 136 9                         ; compile_native_aiter_afp4_recovered.py:136:9
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	v_accvgpr_read_b32 v198, a120
	v_accvgpr_write_b32 a132, v192
	v_accvgpr_write_b32 a133, v193
	v_accvgpr_read_b32 v193, a131
	v_accvgpr_read_b32 v192, a130
	v_accvgpr_read_b32 v197, a119
	.loc	1 135 23                        ; compile_native_aiter_afp4_recovered.py:135:23
	v_mfma_scale_f32_32x32x64_f8f6f4 v[32:47], a[8:11], a[0:3], v[32:47], v242, v241 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	.loc	1 136 9                         ; compile_native_aiter_afp4_recovered.py:136:9
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	v_accvgpr_read_b32 v196, a118
	v_accvgpr_write_b32 a130, v192
	v_accvgpr_read_b32 v195, a101
	.loc	1 140 9                         ; compile_native_aiter_afp4_recovered.py:140:9
	s_add_i32 s15, s15, 1
	v_accvgpr_write_b32 a131, v193
	v_accvgpr_read_b32 v194, a100
	.loc	1 135 23                        ; compile_native_aiter_afp4_recovered.py:135:23
	v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], a[4:7], a[0:3], v[0:15], v242, v241 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v243, a83
	v_accvgpr_read_b32 v242, a82
	.loc	1 136 9                         ; compile_native_aiter_afp4_recovered.py:136:9
	v_lshl_add_u64 v[242:243], v[242:243], 0, s[22:23]
	v_accvgpr_write_b32 a82, v242
	v_accvgpr_write_b32 a83, v243
	v_accvgpr_read_b32 v243, a81
	v_accvgpr_read_b32 v242, a80
	v_lshl_add_u64 v[242:243], v[242:243], 0, s[22:23]
	v_accvgpr_write_b32 a80, v242
	v_accvgpr_write_b32 a81, v243
	v_accvgpr_read_b32 v243, a77
	v_accvgpr_read_b32 v242, a76
	v_lshl_add_u64 v[242:243], v[242:243], 0, s[22:23]
	v_accvgpr_write_b32 a76, v242
	v_accvgpr_write_b32 a77, v243
	v_accvgpr_read_b32 v243, a75
	v_accvgpr_read_b32 v242, a74
	.loc	1 138 9                         ; compile_native_aiter_afp4_recovered.py:138:9
	v_lshl_add_u64 v[242:243], v[242:243], 0, s[12:13]
	v_accvgpr_write_b32 a74, v242
	v_accvgpr_write_b32 a75, v243
	v_accvgpr_read_b32 v243, a73
	v_accvgpr_read_b32 v242, a72
	v_lshl_add_u64 v[242:243], v[242:243], 0, s[12:13]
	v_accvgpr_write_b32 a72, v242
	v_accvgpr_write_b32 a73, v243
	v_accvgpr_read_b32 v243, a79
	v_accvgpr_read_b32 v242, a78
	.loc	1 139 9                         ; compile_native_aiter_afp4_recovered.py:139:9
	v_lshl_add_u64 v[242:243], v[242:243], 0, s[6:7]
	v_accvgpr_write_b32 a78, v242
	v_accvgpr_write_b32 a79, v243
	v_accvgpr_read_b32 v243, a71
	v_accvgpr_read_b32 v242, a70
	v_lshl_add_u64 v[242:243], v[242:243], 0, s[6:7]
	v_accvgpr_write_b32 a70, v242
	v_accvgpr_write_b32 a71, v243
	.loc	1 137 9                         ; compile_native_aiter_afp4_recovered.py:137:9
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[80:81]
	v_accvgpr_write_b32 a32, v242
	v_accvgpr_write_b32 a33, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[78:79]
	v_accvgpr_write_b32 a204, v242
	v_accvgpr_write_b32 a205, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[76:77]
	v_accvgpr_write_b32 a34, v242
	v_accvgpr_write_b32 a35, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[74:75]
	v_accvgpr_write_b32 a202, v242
	v_accvgpr_write_b32 a203, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[72:73]
	v_accvgpr_write_b32 a36, v242
	v_accvgpr_write_b32 a37, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[70:71]
	v_accvgpr_write_b32 a200, v242
	v_accvgpr_write_b32 a201, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[68:69]
	v_accvgpr_write_b32 a38, v242
	v_accvgpr_write_b32 a39, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[66:67]
	v_accvgpr_write_b32 a198, v242
	v_accvgpr_write_b32 a199, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[64:65]
	v_accvgpr_write_b32 a40, v242
	v_accvgpr_write_b32 a41, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[62:63]
	v_accvgpr_write_b32 a196, v242
	v_accvgpr_write_b32 a197, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[60:61]
	v_accvgpr_write_b32 a42, v242
	v_accvgpr_write_b32 a43, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[58:59]
	v_accvgpr_write_b32 a194, v242
	v_accvgpr_write_b32 a195, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[56:57]
	v_accvgpr_write_b32 a44, v242
	v_accvgpr_write_b32 a45, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[54:55]
	v_accvgpr_write_b32 a192, v242
	v_accvgpr_write_b32 a193, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[52:53]
	v_accvgpr_write_b32 a46, v242
	v_accvgpr_write_b32 a47, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[50:51]
	v_accvgpr_write_b32 a190, v242
	v_accvgpr_write_b32 a191, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[48:49]
	v_accvgpr_write_b32 a48, v242
	v_accvgpr_write_b32 a49, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[46:47]
	v_accvgpr_write_b32 a128, v242
	v_accvgpr_write_b32 a129, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[44:45]
	v_accvgpr_write_b32 a50, v242
	v_accvgpr_write_b32 a51, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[42:43]
	v_accvgpr_write_b32 a126, v242
	v_accvgpr_write_b32 a127, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[40:41]
	v_accvgpr_write_b32 a52, v242
	v_accvgpr_write_b32 a53, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[38:39]
	v_accvgpr_write_b32 a124, v242
	v_accvgpr_write_b32 a125, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[36:37]
	v_accvgpr_write_b32 a54, v242
	v_accvgpr_write_b32 a55, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[34:35]
	v_accvgpr_write_b32 a122, v242
	v_accvgpr_write_b32 a123, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[28:29]
	v_accvgpr_write_b32 a56, v242
	v_accvgpr_write_b32 a57, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[26:27]
	v_accvgpr_write_b32 a120, v242
	v_accvgpr_write_b32 a121, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[24:25]
	v_accvgpr_write_b32 a58, v242
	v_accvgpr_write_b32 a59, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[18:19]
	v_accvgpr_write_b32 a118, v242
	v_accvgpr_write_b32 a119, v243
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[16:17]
	v_mov_b64_e32 v[192:193], v[246:247]
	.loc	1 122 11                        ; compile_native_aiter_afp4_recovered.py:122:11
	s_lshl_b32 s85, s15, 5
	v_accvgpr_write_b32 a60, v242
	.loc	1 136 9                         ; compile_native_aiter_afp4_recovered.py:136:9
	v_lshl_add_u64 v[250:251], v[250:251], 0, s[22:23]
	v_lshl_add_u64 v[248:249], v[248:249], 0, s[22:23]
	v_lshl_add_u64 v[206:207], v[206:207], 0, s[22:23]
	v_lshl_add_u64 v[204:205], v[204:205], 0, s[22:23]
	v_lshl_add_u64 v[202:203], v[202:203], 0, s[22:23]
	v_lshl_add_u64 v[200:201], v[200:201], 0, s[22:23]
	v_lshl_add_u64 v[198:199], v[198:199], 0, s[22:23]
	v_lshl_add_u64 v[196:197], v[196:197], 0, s[22:23]
	v_lshl_add_u64 v[194:195], v[194:195], 0, s[22:23]
	v_lshl_add_u64 v[192:193], v[192:193], 0, s[22:23]
	.loc	1 122 11                        ; compile_native_aiter_afp4_recovered.py:122:11
	s_cmp_lt_i32 s85, s14
	v_accvgpr_write_b32 a61, v243
	.loc	1 137 9                         ; compile_native_aiter_afp4_recovered.py:137:9
	v_lshl_add_u64 v[244:245], v[252:253], 0, s[10:11]
	v_lshl_add_u64 v[242:243], v[252:253], 0, s[8:9]
	.loc	1 122 11                        ; compile_native_aiter_afp4_recovered.py:122:11
	s_cbranch_scc0 .LBB0_146
.LBB0_8:                                ; =>This Inner Loop Header: Depth=1
	.loc	1 0 11 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:11
	v_accvgpr_write_b32 a100, v244
	.loc	1 124 41 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:124:41
	s_lshl_b32 s2, s15, 1
	v_accvgpr_write_b32 a101, v245
	v_accvgpr_write_b32 a164, v242
	.loc	1 124 13 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:124:13
	s_sub_i32 s2, s84, s2
	v_accvgpr_read_b32 v241, a69
	.loc	1 126 20 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:126:20
	v_mov_b32_e32 v245, 0x7f
	v_accvgpr_write_b32 a165, v243
	.loc	1 123 22                        ; compile_native_aiter_afp4_recovered.py:123:22
	v_cmp_gt_i32_e32 vcc, s2, v241
	v_mov_b32_e32 v242, v245
	.loc	1 126 20                        ; compile_native_aiter_afp4_recovered.py:126:20
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_10
; %bb.9:                                ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 20 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:20
	v_accvgpr_read_b32 v243, a75
	v_accvgpr_read_b32 v242, a74
	.loc	1 126 20                        ; compile_native_aiter_afp4_recovered.py:126:20
	global_load_ubyte v242, v[242:243], off
.LBB0_10:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 20                          ; compile_native_aiter_afp4_recovered.py:0:20
	s_or_b64 exec, exec, s[2:3]
	.loc	1 126 20                        ; compile_native_aiter_afp4_recovered.py:126:20
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_12
; %bb.11:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 20                          ; compile_native_aiter_afp4_recovered.py:0:20
	v_accvgpr_read_b32 v245, a73
	v_accvgpr_read_b32 v244, a72
	.loc	1 126 20                        ; compile_native_aiter_afp4_recovered.py:126:20
	global_load_ubyte v245, v[244:245], off
.LBB0_12:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 20                          ; compile_native_aiter_afp4_recovered.py:0:20
	s_or_b64 exec, exec, s[2:3]
	.loc	1 127 20 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:127:20
	v_mov_b32_e32 v241, 0x7f
	v_mov_b32_e32 v244, v241
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_14
; %bb.13:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 20 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:20
	v_accvgpr_read_b32 v247, a79
	v_accvgpr_read_b32 v246, a78
	.loc	1 127 20                        ; compile_native_aiter_afp4_recovered.py:127:20
	global_load_ubyte v244, v[246:247], off
.LBB0_14:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 20                          ; compile_native_aiter_afp4_recovered.py:0:20
	s_or_b64 exec, exec, s[2:3]
	.loc	1 127 20                        ; compile_native_aiter_afp4_recovered.py:127:20
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_16
; %bb.15:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 20                          ; compile_native_aiter_afp4_recovered.py:0:20
	v_accvgpr_read_b32 v247, a71
	v_accvgpr_read_b32 v246, a70
	.loc	1 127 20                        ; compile_native_aiter_afp4_recovered.py:127:20
	global_load_ubyte v241, v[246:247], off
.LBB0_16:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 20                          ; compile_native_aiter_afp4_recovered.py:0:20
	s_or_b64 exec, exec, s[2:3]
	.loc	1 128 52 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:128:52
	s_sub_i32 s82, s14, s85
	v_accvgpr_read_b32 v243, a67
	.loc	1 128 34 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:128:34
	v_cmp_gt_i32_e32 vcc, s82, v243
	v_mov_b32_e32 v243, 0
	v_accvgpr_write_b32 a0, 0
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_18
; %bb.17:                               ;   in Loop: Header=BB0_8 Depth=1
	global_load_ubyte a0, v[250:251], off
.LBB0_18:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_20
; %bb.19:                               ;   in Loop: Header=BB0_8 Depth=1
	global_load_ubyte v243, v[248:249], off
.LBB0_20:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a1, 0
	v_accvgpr_write_b32 a2, 0
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_22
; %bb.21:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a169
	v_accvgpr_read_b32 v246, a168
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a2, v[246:247], off
.LBB0_22:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_24
; %bb.23:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a167
	v_accvgpr_read_b32 v246, a166
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a1, v[246:247], off
.LBB0_24:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a3, 0
	v_accvgpr_write_b32 a4, 0
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_26
; %bb.25:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a163
	v_accvgpr_read_b32 v246, a162
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a4, v[246:247], off
.LBB0_26:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_28
; %bb.27:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a161
	v_accvgpr_read_b32 v246, a160
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a3, v[246:247], off
.LBB0_28:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a5, 0
	v_accvgpr_write_b32 a6, 0
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_30
; %bb.29:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a159
	v_accvgpr_read_b32 v246, a158
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a6, v[246:247], off
.LBB0_30:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_32
; %bb.31:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a157
	v_accvgpr_read_b32 v246, a156
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a5, v[246:247], off
.LBB0_32:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a7, 0
	v_accvgpr_write_b32 a8, 0
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_34
; %bb.33:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a155
	v_accvgpr_read_b32 v246, a154
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a8, v[246:247], off
.LBB0_34:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_36
; %bb.35:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a153
	v_accvgpr_read_b32 v246, a152
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a7, v[246:247], off
.LBB0_36:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a9, 0
	v_accvgpr_write_b32 a10, 0
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_38
; %bb.37:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a151
	v_accvgpr_read_b32 v246, a150
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a10, v[246:247], off
.LBB0_38:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_40
; %bb.39:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a149
	v_accvgpr_read_b32 v246, a148
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a9, v[246:247], off
.LBB0_40:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a11, 0
	v_accvgpr_write_b32 a12, 0
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_42
; %bb.41:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a147
	v_accvgpr_read_b32 v246, a146
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a12, v[246:247], off
.LBB0_42:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_44
; %bb.43:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a145
	v_accvgpr_read_b32 v246, a144
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a11, v[246:247], off
.LBB0_44:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a13, 0
	v_accvgpr_write_b32 a14, 0
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_46
; %bb.45:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a143
	v_accvgpr_read_b32 v246, a142
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a14, v[246:247], off
.LBB0_46:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_48
; %bb.47:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a141
	v_accvgpr_read_b32 v246, a140
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a13, v[246:247], off
.LBB0_48:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a15, 0
	v_accvgpr_write_b32 a16, 0
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_50
; %bb.49:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a139
	v_accvgpr_read_b32 v246, a138
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a16, v[246:247], off
.LBB0_50:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_52
; %bb.51:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a137
	v_accvgpr_read_b32 v246, a136
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a15, v[246:247], off
.LBB0_52:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a17, 0
	v_accvgpr_write_b32 a18, 0
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_54
; %bb.53:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a135
	v_accvgpr_read_b32 v246, a134
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a18, v[246:247], off
.LBB0_54:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a133
	v_accvgpr_read_b32 v246, a132
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a17, v[246:247], off
.LBB0_56:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a19, 0
	v_accvgpr_write_b32 a20, 0
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_58
; %bb.57:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a131
	v_accvgpr_read_b32 v246, a130
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a20, v[246:247], off
.LBB0_58:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_60
; %bb.59:                               ;   in Loop: Header=BB0_8 Depth=1
	global_load_ubyte a19, v[206:207], off
.LBB0_60:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a21, 0
	v_accvgpr_write_b32 a22, 0
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_62
; %bb.61:                               ;   in Loop: Header=BB0_8 Depth=1
	global_load_ubyte a22, v[204:205], off
.LBB0_62:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_64
; %bb.63:                               ;   in Loop: Header=BB0_8 Depth=1
	global_load_ubyte a21, v[202:203], off
.LBB0_64:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a23, 0
	v_accvgpr_write_b32 a24, 0
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_66
; %bb.65:                               ;   in Loop: Header=BB0_8 Depth=1
	global_load_ubyte a24, v[200:201], off
.LBB0_66:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_68
; %bb.67:                               ;   in Loop: Header=BB0_8 Depth=1
	global_load_ubyte a23, v[198:199], off
.LBB0_68:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a25, 0
	v_accvgpr_write_b32 a26, 0
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_70
; %bb.69:                               ;   in Loop: Header=BB0_8 Depth=1
	global_load_ubyte a26, v[196:197], off
.LBB0_70:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_72
; %bb.71:                               ;   in Loop: Header=BB0_8 Depth=1
	global_load_ubyte a25, v[194:195], off
.LBB0_72:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a27, 0
	v_accvgpr_write_b32 a28, 0
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_74
; %bb.73:                               ;   in Loop: Header=BB0_8 Depth=1
	global_load_ubyte a28, v[192:193], off
.LBB0_74:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_76
; %bb.75:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a83
	v_accvgpr_read_b32 v246, a82
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a27, v[246:247], off
.LBB0_76:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	v_accvgpr_write_b32 a29, 0
	v_accvgpr_write_b32 a30, 0
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execnz .LBB0_110
; %bb.77:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execnz .LBB0_111
.LBB0_78:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	s_or_b64 exec, exec, s[2:3]
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 1
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_112
.LBB0_79:                               ;   in Loop: Header=BB0_8 Depth=1
	global_load_ubyte a31, v[252:253], off
	.loc	1 131 18                        ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 2
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_113
.LBB0_80:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a33
	v_accvgpr_read_b32 v246, a32
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a32, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 3
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_114
.LBB0_81:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a33, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 4
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_115
.LBB0_82:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a34, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 5
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_116
.LBB0_83:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a35, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 6
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_117
.LBB0_84:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a36, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 7
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_118
.LBB0_85:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a37, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 8
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_119
.LBB0_86:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a38, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 9
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_120
.LBB0_87:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a39, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 10
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_121
.LBB0_88:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a40, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 11
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_122
.LBB0_89:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a41, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 12
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_123
.LBB0_90:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a42, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 13
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_124
.LBB0_91:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a43, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 14
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_125
.LBB0_92:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a44, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 15
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_126
.LBB0_93:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a45, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 16
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_127
.LBB0_94:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a46, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 17
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_128
.LBB0_95:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a47, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 18
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_129
.LBB0_96:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a48, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 19
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_130
.LBB0_97:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a49, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 20
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_131
.LBB0_98:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a50, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 21
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_132
.LBB0_99:                               ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a51, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 22
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_133
.LBB0_100:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a52, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 23
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_134
.LBB0_101:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a53, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 24
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_135
.LBB0_102:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a54, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 25
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_136
.LBB0_103:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a55, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 26
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_137
.LBB0_104:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a56, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 27
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_138
.LBB0_105:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a57, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 28
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_139
.LBB0_106:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a58, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 29
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_140
.LBB0_107:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a59, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 30
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_141
.LBB0_108:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a60, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 31
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_142
.LBB0_109:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a61, 0
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_branch .LBB0_143
	.loc	1 0 13                          ; :0:13
.Ltmp22:
	.p2align	5, , 4
.LBB0_110:                              ;   in Loop: Header=BB0_8 Depth=1
	v_accvgpr_read_b32 v247, a81
	v_accvgpr_read_b32 v246, a80
	.loc	1 128 13 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a30, v[246:247], off
	s_or_b64 exec, exec, s[2:3]
	s_and_saveexec_b64 s[2:3], vcc
	s_cbranch_execz .LBB0_78
.LBB0_111:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a77
	v_accvgpr_read_b32 v246, a76
	.loc	1 128 13                        ; compile_native_aiter_afp4_recovered.py:128:13
	global_load_ubyte a29, v[246:247], off
	s_or_b64 exec, exec, s[2:3]
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 1
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_79
.LBB0_112:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a31, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 2
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_80
.LBB0_113:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a32, 0
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 3
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_81
.LBB0_114:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a205
	v_accvgpr_read_b32 v246, a204
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a33, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 4
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_82
.LBB0_115:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a35
	v_accvgpr_read_b32 v246, a34
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a34, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 5
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_83
.LBB0_116:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a203
	v_accvgpr_read_b32 v246, a202
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a35, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 6
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_84
.LBB0_117:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a37
	v_accvgpr_read_b32 v246, a36
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a36, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 7
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_85
.LBB0_118:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a201
	v_accvgpr_read_b32 v246, a200
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a37, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 8
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_86
.LBB0_119:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a39
	v_accvgpr_read_b32 v246, a38
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a38, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 9
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_87
.LBB0_120:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a199
	v_accvgpr_read_b32 v246, a198
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a39, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 10
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_88
.LBB0_121:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a41
	v_accvgpr_read_b32 v246, a40
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a40, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 11
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_89
.LBB0_122:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a197
	v_accvgpr_read_b32 v246, a196
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a41, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 12
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_90
.LBB0_123:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a43
	v_accvgpr_read_b32 v246, a42
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a42, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 13
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_91
.LBB0_124:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a195
	v_accvgpr_read_b32 v246, a194
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a43, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 14
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_92
.LBB0_125:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a45
	v_accvgpr_read_b32 v246, a44
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a44, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 15
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_93
.LBB0_126:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a193
	v_accvgpr_read_b32 v246, a192
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a45, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 16
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_94
.LBB0_127:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a47
	v_accvgpr_read_b32 v246, a46
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a46, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 17
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_95
.LBB0_128:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a191
	v_accvgpr_read_b32 v246, a190
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a47, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 18
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_96
.LBB0_129:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a49
	v_accvgpr_read_b32 v246, a48
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a48, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 19
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_97
.LBB0_130:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a129
	v_accvgpr_read_b32 v246, a128
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a49, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 20
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_98
.LBB0_131:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a51
	v_accvgpr_read_b32 v246, a50
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a50, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 21
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_99
.LBB0_132:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a127
	v_accvgpr_read_b32 v246, a126
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a51, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 22
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_100
.LBB0_133:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a53
	v_accvgpr_read_b32 v246, a52
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a52, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 23
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_101
.LBB0_134:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a125
	v_accvgpr_read_b32 v246, a124
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a53, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 24
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_102
.LBB0_135:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a55
	v_accvgpr_read_b32 v246, a54
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a54, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 25
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_103
.LBB0_136:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a123
	v_accvgpr_read_b32 v246, a122
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a55, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 26
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_104
.LBB0_137:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a57
	v_accvgpr_read_b32 v246, a56
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a56, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 27
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_105
.LBB0_138:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a121
	v_accvgpr_read_b32 v246, a120
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a57, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 28
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_106
.LBB0_139:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a59
	v_accvgpr_read_b32 v246, a58
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a58, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 29
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_107
.LBB0_140:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a119
	v_accvgpr_read_b32 v246, a118
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a59, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 30
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_108
.LBB0_141:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a61
	v_accvgpr_read_b32 v246, a60
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a60, v[246:247], off
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 31
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc1 .LBB0_109
.LBB0_142:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_read_b32 v247, a101
	v_accvgpr_read_b32 v246, a100
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	global_load_ubyte a61, v[246:247], off
.LBB0_143:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a128, v206
	v_accvgpr_write_b32 a126, v204
	v_accvgpr_write_b32 a124, v202
	v_accvgpr_write_b32 a122, v200
	v_accvgpr_write_b32 a120, v198
	v_accvgpr_write_b32 a118, v196
	v_accvgpr_write_b32 a100, v194
	v_accvgpr_write_b32 a129, v207
	v_accvgpr_write_b32 a127, v205
	v_accvgpr_write_b32 a125, v203
	v_accvgpr_write_b32 a123, v201
	v_accvgpr_write_b32 a121, v199
	v_accvgpr_write_b32 a119, v197
	v_accvgpr_write_b32 a101, v195
	v_mov_b64_e32 v[246:247], v[192:193]
	.loc	1 131 18 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:131:18
	s_cmp_lt_i32 s82, 32
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_cbranch_scc0 .LBB0_6
; %bb.144:                              ;   in Loop: Header=BB0_8 Depth=1
	.loc	1 0 13 is_stmt 0                ; compile_native_aiter_afp4_recovered.py:0:13
	v_accvgpr_write_b32 a62, 0
	.loc	1 129 13                        ; compile_native_aiter_afp4_recovered.py:129:13
	s_branch .LBB0_7
.LBB0_145:
	.loc	1 0 13                          ; compile_native_aiter_afp4_recovered.py:0:13
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v31, v15
	v_accvgpr_write_b32 a68, v16
	v_mov_b32_e32 v30, v15
	v_mov_b32_e32 v29, v15
	v_mov_b32_e32 v28, v15
	v_mov_b32_e32 v27, v15
	v_mov_b32_e32 v26, v15
	v_mov_b32_e32 v25, v15
	v_mov_b32_e32 v24, v15
	v_mov_b32_e32 v23, v15
	v_mov_b32_e32 v22, v15
	v_mov_b32_e32 v21, v15
	v_mov_b32_e32 v20, v15
	v_mov_b32_e32 v19, v15
	v_mov_b32_e32 v18, v15
	v_mov_b32_e32 v17, v15
	v_mov_b32_e32 v16, v15
	v_accvgpr_write_b32 a99, v31
	v_accvgpr_write_b32 a117, v31
	v_accvgpr_write_b32 a66, v14
	v_mov_b32_e32 v14, v15
	v_mov_b32_e32 v13, v15
	v_mov_b32_e32 v12, v15
	v_mov_b32_e32 v11, v15
	v_mov_b32_e32 v10, v15
	v_mov_b32_e32 v9, v15
	v_mov_b32_e32 v8, v15
	v_mov_b32_e32 v7, v15
	v_mov_b32_e32 v6, v15
	v_mov_b32_e32 v5, v15
	v_mov_b32_e32 v4, v15
	v_mov_b32_e32 v3, v15
	v_mov_b32_e32 v2, v15
	v_mov_b32_e32 v1, v15
	v_mov_b32_e32 v0, v15
	v_mov_b32_e32 v47, v15
	v_mov_b32_e32 v46, v15
	v_mov_b32_e32 v45, v15
	v_mov_b32_e32 v44, v15
	v_mov_b32_e32 v43, v15
	v_mov_b32_e32 v42, v15
	v_mov_b32_e32 v41, v15
	v_mov_b32_e32 v40, v15
	v_mov_b32_e32 v39, v15
	v_mov_b32_e32 v38, v15
	v_mov_b32_e32 v37, v15
	v_mov_b32_e32 v36, v15
	v_mov_b32_e32 v35, v15
	v_mov_b32_e32 v34, v15
	v_mov_b32_e32 v33, v15
	v_mov_b32_e32 v32, v15
	v_mov_b32_e32 v63, v15
	v_mov_b32_e32 v62, v15
	v_mov_b32_e32 v61, v15
	v_mov_b32_e32 v60, v15
	v_mov_b32_e32 v59, v15
	v_mov_b32_e32 v58, v15
	v_mov_b32_e32 v57, v15
	v_mov_b32_e32 v56, v15
	v_mov_b32_e32 v55, v15
	v_mov_b32_e32 v54, v15
	v_mov_b32_e32 v53, v15
	v_mov_b32_e32 v52, v15
	v_mov_b32_e32 v51, v15
	v_mov_b32_e32 v50, v15
	v_mov_b32_e32 v49, v15
	v_mov_b32_e32 v48, v15
	v_mov_b32_e32 v79, v15
	v_mov_b32_e32 v78, v15
	v_mov_b32_e32 v77, v15
	v_mov_b32_e32 v76, v15
	v_mov_b32_e32 v75, v15
	v_mov_b32_e32 v74, v15
	v_mov_b32_e32 v73, v15
	v_mov_b32_e32 v72, v15
	v_mov_b32_e32 v71, v15
	v_mov_b32_e32 v70, v15
	v_mov_b32_e32 v69, v15
	v_mov_b32_e32 v68, v15
	v_mov_b32_e32 v67, v15
	v_mov_b32_e32 v66, v15
	v_mov_b32_e32 v65, v15
	v_mov_b32_e32 v64, v15
	v_mov_b32_e32 v95, v15
	v_mov_b32_e32 v94, v15
	v_mov_b32_e32 v93, v15
	v_mov_b32_e32 v92, v15
	v_mov_b32_e32 v91, v15
	v_mov_b32_e32 v90, v15
	v_mov_b32_e32 v89, v15
	v_mov_b32_e32 v88, v15
	v_mov_b32_e32 v87, v15
	v_mov_b32_e32 v86, v15
	v_mov_b32_e32 v85, v15
	v_mov_b32_e32 v84, v15
	v_mov_b32_e32 v83, v15
	v_mov_b32_e32 v82, v15
	v_mov_b32_e32 v81, v15
	v_mov_b32_e32 v80, v15
	v_mov_b32_e32 v111, v15
	v_mov_b32_e32 v110, v15
	v_mov_b32_e32 v109, v15
	v_mov_b32_e32 v108, v15
	v_mov_b32_e32 v107, v15
	v_mov_b32_e32 v106, v15
	v_mov_b32_e32 v105, v15
	v_mov_b32_e32 v104, v15
	v_mov_b32_e32 v103, v15
	v_mov_b32_e32 v102, v15
	v_mov_b32_e32 v101, v15
	v_mov_b32_e32 v100, v15
	v_mov_b32_e32 v99, v15
	v_mov_b32_e32 v98, v15
	v_mov_b32_e32 v97, v15
	v_mov_b32_e32 v96, v15
	v_mov_b32_e32 v127, v15
	v_mov_b32_e32 v126, v15
	v_mov_b32_e32 v125, v15
	v_mov_b32_e32 v124, v15
	v_mov_b32_e32 v123, v15
	v_mov_b32_e32 v122, v15
	v_mov_b32_e32 v121, v15
	v_mov_b32_e32 v120, v15
	v_mov_b32_e32 v119, v15
	v_mov_b32_e32 v118, v15
	v_mov_b32_e32 v117, v15
	v_mov_b32_e32 v116, v15
	v_mov_b32_e32 v115, v15
	v_mov_b32_e32 v114, v15
	v_mov_b32_e32 v113, v15
	v_mov_b32_e32 v112, v15
	v_mov_b32_e32 v143, v15
	v_mov_b32_e32 v142, v15
	v_mov_b32_e32 v141, v15
	v_mov_b32_e32 v140, v15
	v_mov_b32_e32 v139, v15
	v_mov_b32_e32 v138, v15
	v_mov_b32_e32 v137, v15
	v_mov_b32_e32 v136, v15
	v_mov_b32_e32 v135, v15
	v_mov_b32_e32 v134, v15
	v_mov_b32_e32 v133, v15
	v_mov_b32_e32 v132, v15
	v_mov_b32_e32 v131, v15
	v_mov_b32_e32 v130, v15
	v_mov_b32_e32 v129, v15
	v_mov_b32_e32 v128, v15
	v_mov_b32_e32 v159, v15
	v_mov_b32_e32 v158, v15
	v_mov_b32_e32 v157, v15
	v_mov_b32_e32 v156, v15
	v_mov_b32_e32 v155, v15
	v_mov_b32_e32 v154, v15
	v_mov_b32_e32 v153, v15
	v_mov_b32_e32 v152, v15
	v_mov_b32_e32 v151, v15
	v_mov_b32_e32 v150, v15
	v_mov_b32_e32 v149, v15
	v_mov_b32_e32 v148, v15
	v_mov_b32_e32 v147, v15
	v_mov_b32_e32 v146, v15
	v_mov_b32_e32 v145, v15
	v_mov_b32_e32 v144, v15
	v_mov_b32_e32 v175, v15
	v_mov_b32_e32 v174, v15
	v_mov_b32_e32 v173, v15
	v_mov_b32_e32 v172, v15
	v_mov_b32_e32 v171, v15
	v_mov_b32_e32 v170, v15
	v_mov_b32_e32 v169, v15
	v_mov_b32_e32 v168, v15
	v_mov_b32_e32 v167, v15
	v_mov_b32_e32 v166, v15
	v_mov_b32_e32 v165, v15
	v_mov_b32_e32 v164, v15
	v_mov_b32_e32 v163, v15
	v_mov_b32_e32 v162, v15
	v_mov_b32_e32 v161, v15
	v_mov_b32_e32 v160, v15
	v_mov_b32_e32 v191, v15
	v_mov_b32_e32 v190, v15
	v_mov_b32_e32 v189, v15
	v_mov_b32_e32 v188, v15
	v_mov_b32_e32 v187, v15
	v_mov_b32_e32 v186, v15
	v_mov_b32_e32 v185, v15
	v_mov_b32_e32 v184, v15
	v_mov_b32_e32 v183, v15
	v_mov_b32_e32 v182, v15
	v_mov_b32_e32 v181, v15
	v_mov_b32_e32 v180, v15
	v_mov_b32_e32 v179, v15
	v_mov_b32_e32 v178, v15
	v_mov_b32_e32 v177, v15
	v_mov_b32_e32 v176, v15
	v_accvgpr_write_b32 a98, v30
	v_accvgpr_write_b32 a97, v29
	v_accvgpr_write_b32 a96, v28
	v_accvgpr_write_b32 a95, v27
	v_accvgpr_write_b32 a94, v26
	v_accvgpr_write_b32 a93, v25
	v_accvgpr_write_b32 a92, v24
	v_accvgpr_write_b32 a91, v23
	v_accvgpr_write_b32 a90, v22
	v_accvgpr_write_b32 a89, v21
	v_accvgpr_write_b32 a88, v20
	v_accvgpr_write_b32 a87, v19
	v_accvgpr_write_b32 a86, v18
	v_accvgpr_write_b32 a85, v17
	v_accvgpr_write_b32 a84, v16
	v_accvgpr_write_b32 a116, v30
	v_accvgpr_write_b32 a115, v29
	v_accvgpr_write_b32 a114, v28
	v_accvgpr_write_b32 a113, v27
	v_accvgpr_write_b32 a112, v26
	v_accvgpr_write_b32 a111, v25
	v_accvgpr_write_b32 a110, v24
	v_accvgpr_write_b32 a109, v23
	v_accvgpr_write_b32 a108, v22
	v_accvgpr_write_b32 a107, v21
	v_accvgpr_write_b32 a106, v20
	v_accvgpr_write_b32 a105, v19
	v_accvgpr_write_b32 a104, v18
	v_accvgpr_write_b32 a103, v17
	v_accvgpr_write_b32 a102, v16
	v_mov_b32_e32 v239, v15
	v_mov_b32_e32 v238, v15
	v_mov_b32_e32 v237, v15
	v_mov_b32_e32 v236, v15
	v_mov_b32_e32 v235, v15
	v_mov_b32_e32 v234, v15
	v_mov_b32_e32 v233, v15
	v_mov_b32_e32 v232, v15
	v_mov_b32_e32 v231, v15
	v_mov_b32_e32 v230, v15
	v_mov_b32_e32 v229, v15
	v_mov_b32_e32 v228, v15
	v_mov_b32_e32 v227, v15
	v_mov_b32_e32 v226, v15
	v_mov_b32_e32 v225, v15
	v_mov_b32_e32 v224, v15
	v_mov_b32_e32 v223, v15
	v_mov_b32_e32 v222, v15
	v_mov_b32_e32 v221, v15
	v_mov_b32_e32 v220, v15
	v_mov_b32_e32 v219, v15
	v_mov_b32_e32 v218, v15
	v_mov_b32_e32 v217, v15
	v_mov_b32_e32 v216, v15
	v_mov_b32_e32 v215, v15
	v_mov_b32_e32 v214, v15
	v_mov_b32_e32 v213, v15
	v_mov_b32_e32 v212, v15
	v_mov_b32_e32 v211, v15
	v_mov_b32_e32 v210, v15
	v_mov_b32_e32 v209, v15
	v_mov_b32_e32 v208, v15
.LBB0_146:                              ; %Flow525
	s_load_dwordx2 s[34:35], s[0:1], 0x48
	.loc	1 145 22 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:145:22
	v_cvt_pk_bf16_f32 v254, v16, v17
	.loc	1 111 39                        ; compile_native_aiter_afp4_recovered.py:111:39
	s_lshr_b32 s0, s31, 2
	v_accvgpr_read_b32 v16, a67
	v_or_b32_e32 v194, s0, v16
	v_accvgpr_read_b32 v16, a68
	v_lshrrev_b32_e32 v16, 2, v16
	.loc	1 145 22                        ; compile_native_aiter_afp4_recovered.py:145:22
	v_cvt_pk_bf16_f32 v253, v18, v19
	.loc	1 111 39                        ; compile_native_aiter_afp4_recovered.py:111:39
	v_or_b32_e32 v19, s33, v16
	v_accvgpr_read_b32 v16, a64
	.loc	1 142 15                        ; compile_native_aiter_afp4_recovered.py:142:15
	v_or_b32_e32 v16, v16, v194
	.loc	1 145 22                        ; compile_native_aiter_afp4_recovered.py:145:22
	v_cvt_pk_bf16_f32 v252, v20, v21
	.loc	1 144 22                        ; compile_native_aiter_afp4_recovered.py:144:22
	s_waitcnt lgkmcnt(0)
	v_mad_i64_i32 v[20:21], s[0:1], v16, s34, 0
	.loc	1 145 22                        ; compile_native_aiter_afp4_recovered.py:145:22
	v_cvt_pk_bf16_f32 v241, v210, v211
	v_cvt_pk_bf16_f32 v240, v212, v213
	v_cvt_pk_bf16_f32 v213, v216, v217
	v_cvt_pk_bf16_f32 v212, v218, v219
	v_cvt_pk_bf16_f32 v211, v220, v221
	v_cvt_pk_bf16_f32 v210, v222, v223
	v_cvt_pk_bf16_f32 v249, v26, v27
	v_cvt_pk_bf16_f32 v246, v224, v225
	v_cvt_pk_bf16_f32 v245, v226, v227
	v_cvt_pk_bf16_f32 v244, v228, v229
	v_cvt_pk_bf16_f32 v243, v230, v231
	v_accvgpr_read_b32 v27, a66
	v_accvgpr_read_b32 v231, a117
	.loc	1 144 14                        ; compile_native_aiter_afp4_recovered.py:144:14
	v_readlane_b32 s0, v255, 0
	.loc	1 145 22                        ; compile_native_aiter_afp4_recovered.py:145:22
	v_cvt_pk_bf16_f32 v242, v208, v209
	.loc	1 143 15                        ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v208, v27, v19
	v_accvgpr_read_b32 v230, a116
	v_accvgpr_read_b32 v229, a115
	v_accvgpr_read_b32 v228, a114
	v_accvgpr_read_b32 v227, a113
	v_accvgpr_read_b32 v226, a112
	v_accvgpr_read_b32 v225, a111
	v_accvgpr_read_b32 v224, a110
	v_accvgpr_read_b32 v223, a109
	v_accvgpr_read_b32 v222, a108
	v_accvgpr_read_b32 v221, a107
	v_accvgpr_read_b32 v220, a106
	v_accvgpr_read_b32 v219, a105
	v_accvgpr_read_b32 v218, a104
	v_accvgpr_read_b32 v217, a103
	v_accvgpr_read_b32 v216, a102
	.loc	1 144 14                        ; compile_native_aiter_afp4_recovered.py:144:14
	v_readlane_b32 s1, v255, 1
	.loc	1 145 67                        ; compile_native_aiter_afp4_recovered.py:145:67
	s_ashr_i32 s89, s88, 31
	.loc	1 145 22 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:22
	v_cvt_pk_bf16_f32 v250, v24, v25
	v_cvt_pk_bf16_f32 v248, v28, v29
	v_cvt_pk_bf16_f32 v247, v30, v31
	v_cvt_pk_bf16_f32 v26, v232, v233
	v_cvt_pk_bf16_f32 v25, v234, v235
	v_cvt_pk_bf16_f32 v24, v236, v237
	v_cvt_pk_bf16_f32 v236, v238, v239
	v_accvgpr_read_b32 v209, a65
	v_cvt_pk_bf16_f32 v239, v216, v217
	v_cvt_pk_bf16_f32 v238, v218, v219
	v_cvt_pk_bf16_f32 v237, v220, v221
	v_cvt_pk_bf16_f32 v235, v222, v223
	v_cvt_pk_bf16_f32 v234, v224, v225
	v_cvt_pk_bf16_f32 v233, v226, v227
	v_cvt_pk_bf16_f32 v232, v228, v229
	v_cvt_pk_bf16_f32 v199, v230, v231
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[192:193], v[20:21], 1, s[0:1]
	.loc	1 144 53 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[20:21], s[0:1], v208, s35, 0
	v_writelane_b32 v255, s88, 2
	.loc	1 145 92 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:145:92
	s_ashr_i32 s31, s30, 31
	v_accvgpr_read_b32 v231, a99
	v_mov_b64_e32 v[28:29], v[32:33]
	.loc	1 145 22 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:22
	v_cvt_pk_bf16_f32 v214, v214, v215
	v_cvt_pk_bf16_f32 v251, v22, v23
	v_accvgpr_read_b32 v17, a63
	v_accvgpr_write_b32 a2, v20
	v_writelane_b32 v255, s89, 3
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[0:1], s[30:31], v[208:209]
	v_accvgpr_read_b32 v230, a98
	v_accvgpr_read_b32 v229, a97
	v_accvgpr_read_b32 v228, a96
	v_accvgpr_read_b32 v227, a95
	v_accvgpr_read_b32 v226, a94
	v_accvgpr_read_b32 v225, a93
	v_accvgpr_read_b32 v224, a92
	v_accvgpr_read_b32 v223, a91
	v_accvgpr_read_b32 v222, a90
	v_accvgpr_read_b32 v221, a89
	v_accvgpr_read_b32 v220, a88
	v_accvgpr_read_b32 v219, a87
	v_accvgpr_read_b32 v218, a86
	v_accvgpr_read_b32 v217, a85
	v_accvgpr_read_b32 v216, a84
	.loc	1 145 22                        ; compile_native_aiter_afp4_recovered.py:145:22
	v_cvt_pk_bf16_f32 v215, v188, v189
	v_cvt_pk_bf16_f32 v23, v190, v191
	v_cvt_pk_bf16_f32 v191, v134, v135
	v_cvt_pk_bf16_f32 v189, v138, v139
	v_mov_b64_e32 v[30:31], v[34:35]
	v_mov_b64_e32 v[32:33], v[36:37]
	v_mov_b64_e32 v[34:35], v[38:39]
	v_mov_b64_e32 v[36:37], v[40:41]
	v_mov_b64_e32 v[38:39], v[42:43]
	v_mov_b64_e32 v[40:41], v[44:45]
	v_mov_b64_e32 v[42:43], v[46:47]
	v_cvt_pk_bf16_f32 v139, v0, v1
	v_cvt_pk_bf16_f32 v135, v8, v9
	v_cvt_pk_bf16_f32 v134, v10, v11
	v_cvt_pk_bf16_f32 v1, v12, v13
	v_cvt_pk_bf16_f32 v0, v14, v15
	v_accvgpr_write_b32 a3, v21
	.loc	1 145 67                        ; compile_native_aiter_afp4_recovered.py:145:67
	v_cmp_gt_i64_e32 vcc, s[88:89], v[16:17]
	.loc	1 145 22                        ; compile_native_aiter_afp4_recovered.py:145:22
	v_cvt_pk_bf16_f32 v198, v216, v217
	v_cvt_pk_bf16_f32 v197, v218, v219
	v_cvt_pk_bf16_f32 v196, v220, v221
	v_cvt_pk_bf16_f32 v195, v222, v223
	v_cvt_pk_bf16_f32 v17, v224, v225
	v_cvt_pk_bf16_f32 v225, v226, v227
	v_cvt_pk_bf16_f32 v223, v228, v229
	v_cvt_pk_bf16_f32 v222, v230, v231
	v_cvt_pk_bf16_f32 v221, v176, v177
	v_cvt_pk_bf16_f32 v220, v178, v179
	v_cvt_pk_bf16_f32 v219, v180, v181
	v_cvt_pk_bf16_f32 v218, v182, v183
	v_cvt_pk_bf16_f32 v217, v184, v185
	v_cvt_pk_bf16_f32 v216, v186, v187
	v_cvt_pk_bf16_f32 v22, v160, v161
	v_cvt_pk_bf16_f32 v21, v162, v163
	v_cvt_pk_bf16_f32 v20, v164, v165
	v_cvt_pk_bf16_f32 v18, v166, v167
	v_cvt_pk_bf16_f32 v209, v168, v169
	v_cvt_pk_bf16_f32 v207, v170, v171
	v_cvt_pk_bf16_f32 v206, v172, v173
	v_cvt_pk_bf16_f32 v205, v174, v175
	v_cvt_pk_bf16_f32 v204, v144, v145
	v_cvt_pk_bf16_f32 v203, v146, v147
	v_cvt_pk_bf16_f32 v202, v148, v149
	v_cvt_pk_bf16_f32 v201, v150, v151
	v_cvt_pk_bf16_f32 v200, v152, v153
	v_cvt_pk_bf16_f32 v231, v154, v155
	v_cvt_pk_bf16_f32 v230, v156, v157
	v_cvt_pk_bf16_f32 v229, v158, v159
	v_cvt_pk_bf16_f32 v228, v128, v129
	v_cvt_pk_bf16_f32 v227, v130, v131
	v_cvt_pk_bf16_f32 v224, v132, v133
	v_cvt_pk_bf16_f32 v190, v136, v137
	v_cvt_pk_bf16_f32 v188, v140, v141
	v_cvt_pk_bf16_f32 v187, v142, v143
	v_cvt_pk_bf16_f32 v186, v112, v113
	v_cvt_pk_bf16_f32 v185, v114, v115
	v_cvt_pk_bf16_f32 v184, v116, v117
	v_cvt_pk_bf16_f32 v183, v118, v119
	v_cvt_pk_bf16_f32 v182, v120, v121
	v_cvt_pk_bf16_f32 v181, v122, v123
	v_cvt_pk_bf16_f32 v180, v124, v125
	v_cvt_pk_bf16_f32 v179, v126, v127
	v_cvt_pk_bf16_f32 v178, v96, v97
	v_cvt_pk_bf16_f32 v177, v98, v99
	v_cvt_pk_bf16_f32 v176, v100, v101
	v_cvt_pk_bf16_f32 v175, v102, v103
	v_cvt_pk_bf16_f32 v174, v104, v105
	v_cvt_pk_bf16_f32 v173, v106, v107
	v_cvt_pk_bf16_f32 v172, v108, v109
	v_cvt_pk_bf16_f32 v171, v110, v111
	v_cvt_pk_bf16_f32 v170, v80, v81
	v_cvt_pk_bf16_f32 v169, v82, v83
	v_cvt_pk_bf16_f32 v168, v84, v85
	v_cvt_pk_bf16_f32 v167, v86, v87
	v_cvt_pk_bf16_f32 v166, v88, v89
	v_cvt_pk_bf16_f32 v165, v90, v91
	v_cvt_pk_bf16_f32 v164, v92, v93
	v_cvt_pk_bf16_f32 v226, v94, v95
	v_cvt_pk_bf16_f32 v163, v64, v65
	v_cvt_pk_bf16_f32 v162, v66, v67
	v_cvt_pk_bf16_f32 v161, v68, v69
	v_cvt_pk_bf16_f32 v160, v70, v71
	v_cvt_pk_bf16_f32 v159, v72, v73
	v_cvt_pk_bf16_f32 v158, v74, v75
	v_cvt_pk_bf16_f32 v157, v76, v77
	v_cvt_pk_bf16_f32 v156, v78, v79
	v_cvt_pk_bf16_f32 v155, v48, v49
	v_cvt_pk_bf16_f32 v154, v50, v51
	v_cvt_pk_bf16_f32 v153, v52, v53
	v_cvt_pk_bf16_f32 v152, v54, v55
	v_cvt_pk_bf16_f32 v151, v56, v57
	v_cvt_pk_bf16_f32 v150, v58, v59
	v_cvt_pk_bf16_f32 v149, v60, v61
	v_cvt_pk_bf16_f32 v148, v62, v63
	v_cvt_pk_bf16_f32 v147, v28, v29
	v_cvt_pk_bf16_f32 v146, v30, v31
	v_cvt_pk_bf16_f32 v145, v32, v33
	v_cvt_pk_bf16_f32 v144, v34, v35
	v_cvt_pk_bf16_f32 v143, v36, v37
	v_cvt_pk_bf16_f32 v142, v38, v39
	v_cvt_pk_bf16_f32 v141, v40, v41
	v_cvt_pk_bf16_f32 v140, v42, v43
	v_cvt_pk_bf16_f32 v138, v2, v3
	v_cvt_pk_bf16_f32 v137, v4, v5
	v_cvt_pk_bf16_f32 v136, v6, v7
	v_writelane_b32 v255, s0, 4
	v_permlane32_swap_b32_e32 v135, v1
	v_permlane32_swap_b32_e32 v134, v0
	v_permlane32_swap_b32_e32 v242, v240
	v_permlane32_swap_b32_e32 v241, v214
	v_permlane32_swap_b32_e32 v213, v211
	v_writelane_b32 v255, s1, 5
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[0:1], vcc, s[0:1]
	.loc	1 145 22                        ; compile_native_aiter_afp4_recovered.py:145:22
	v_permlane32_swap_b32_e32 v212, v210
	v_accvgpr_write_b32 a1, v1
	v_accvgpr_write_b32 a0, v0
	v_permlane32_swap_b32_e32 v254, v252
	v_permlane32_swap_b32_e32 v253, v251
	v_permlane32_swap_b32_e32 v250, v248
	v_permlane32_swap_b32_e32 v249, v247
	v_permlane32_swap_b32_e32 v246, v244
	v_permlane32_swap_b32_e32 v245, v243
	v_permlane32_swap_b32_e32 v26, v24
	v_permlane32_swap_b32_e32 v25, v236
	v_permlane32_swap_b32_e32 v239, v237
	v_permlane32_swap_b32_e32 v238, v235
	v_permlane32_swap_b32_e32 v234, v232
	v_permlane32_swap_b32_e32 v233, v199
	v_permlane32_swap_b32_e32 v198, v196
	v_permlane32_swap_b32_e32 v197, v195
	v_permlane32_swap_b32_e32 v17, v223
	v_permlane32_swap_b32_e32 v225, v222
	v_permlane32_swap_b32_e32 v221, v219
	v_permlane32_swap_b32_e32 v220, v218
	v_permlane32_swap_b32_e32 v217, v215
	v_permlane32_swap_b32_e32 v216, v23
	v_permlane32_swap_b32_e32 v22, v20
	v_permlane32_swap_b32_e32 v21, v18
	v_permlane32_swap_b32_e32 v209, v206
	v_permlane32_swap_b32_e32 v207, v205
	v_permlane32_swap_b32_e32 v204, v202
	v_permlane32_swap_b32_e32 v203, v201
	v_permlane32_swap_b32_e32 v200, v230
	v_permlane32_swap_b32_e32 v231, v229
	v_permlane32_swap_b32_e32 v228, v224
	v_permlane32_swap_b32_e32 v227, v191
	v_permlane32_swap_b32_e32 v190, v188
	v_permlane32_swap_b32_e32 v189, v187
	v_permlane32_swap_b32_e32 v186, v184
	v_permlane32_swap_b32_e32 v185, v183
	v_permlane32_swap_b32_e32 v182, v180
	v_permlane32_swap_b32_e32 v181, v179
	v_permlane32_swap_b32_e32 v178, v176
	v_permlane32_swap_b32_e32 v177, v175
	v_permlane32_swap_b32_e32 v174, v172
	v_permlane32_swap_b32_e32 v173, v171
	v_permlane32_swap_b32_e32 v170, v168
	v_permlane32_swap_b32_e32 v169, v167
	v_permlane32_swap_b32_e32 v166, v164
	v_permlane32_swap_b32_e32 v165, v226
	v_permlane32_swap_b32_e32 v163, v161
	v_permlane32_swap_b32_e32 v162, v160
	v_permlane32_swap_b32_e32 v159, v157
	v_permlane32_swap_b32_e32 v158, v156
	v_permlane32_swap_b32_e32 v155, v153
	v_permlane32_swap_b32_e32 v154, v152
	v_permlane32_swap_b32_e32 v151, v149
	v_permlane32_swap_b32_e32 v150, v148
	v_permlane32_swap_b32_e32 v147, v145
	v_permlane32_swap_b32_e32 v146, v144
	v_permlane32_swap_b32_e32 v143, v141
	v_permlane32_swap_b32_e32 v142, v140
	v_permlane32_swap_b32_e32 v139, v137
	v_permlane32_swap_b32_e32 v138, v136
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_148
; %bb.147:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	v_accvgpr_read_b32 v0, a2
	v_accvgpr_read_b32 v1, a3
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v242, off
.LBB0_148:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[2:3]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or3_b32 v130, v19, v27, 1
	v_accvgpr_read_b32 v131, a65
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[0:1], s[0:1], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[0:1], s[30:31], v[130:131]
	v_accvgpr_write_b32 a5, v1
	v_accvgpr_write_b32 a4, v0
	v_writelane_b32 v255, s0, 6
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[0:1]
	s_nop 0
	v_writelane_b32 v255, s1, 7
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_150
; %bb.149:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	v_accvgpr_read_b32 v0, a4
	v_accvgpr_read_b32 v1, a5
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v242, off
.LBB0_150:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or3_b32 v130, v19, v27, 2
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[0:1], s[0:1], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[0:1], s[30:31], v[130:131]
	v_accvgpr_write_b32 a7, v1
	v_accvgpr_write_b32 a6, v0
	v_writelane_b32 v255, s0, 8
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[0:1]
	s_nop 0
	v_writelane_b32 v255, s1, 9
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_152
; %bb.151:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	v_accvgpr_read_b32 v0, a6
	v_accvgpr_read_b32 v1, a7
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v241, off
.LBB0_152:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or3_b32 v130, v19, v27, 3
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[4:5], s[0:1], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[0:1], s[30:31], v[130:131]
	s_nop 1
	v_writelane_b32 v255, s0, 10
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[0:1]
	s_nop 0
	v_writelane_b32 v255, s1, 11
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_154
; %bb.153:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[4:5], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v241, off
.LBB0_154:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or3_b32 v130, v19, v27, 4
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[6:7], s[0:1], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[0:1], s[30:31], v[130:131]
	s_nop 1
	v_writelane_b32 v255, s0, 12
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[0:1]
	s_nop 0
	v_writelane_b32 v255, s1, 13
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_156
; %bb.155:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[6:7], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v240, off
.LBB0_156:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or3_b32 v130, v19, v27, 5
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[8:9], s[0:1], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[0:1], s[30:31], v[130:131]
	s_nop 1
	v_writelane_b32 v255, s0, 14
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[0:1]
	s_nop 0
	v_writelane_b32 v255, s1, 15
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_158
; %bb.157:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[8:9], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v240, off
.LBB0_158:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or3_b32 v130, v19, v27, 6
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[10:11], s[0:1], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[0:1], s[30:31], v[130:131]
	s_nop 1
	v_writelane_b32 v255, s0, 16
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[0:1]
	s_nop 0
	v_writelane_b32 v255, s1, 17
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_160
; %bb.159:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[10:11], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v214, off
.LBB0_160:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or3_b32 v130, v19, v27, 7
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[12:13], s[0:1], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[0:1], s[30:31], v[130:131]
	s_nop 1
	v_writelane_b32 v255, s0, 18
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[0:1]
	s_nop 0
	v_writelane_b32 v255, s1, 19
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_162
; %bb.161:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[12:13], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v214, off
.LBB0_162:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or3_b32 v130, v19, v27, 16
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[14:15], s[0:1], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[0:1], s[30:31], v[130:131]
	s_nop 1
	v_writelane_b32 v255, s0, 20
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[0:1]
	s_nop 0
	v_writelane_b32 v255, s1, 21
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_164
; %bb.163:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[14:15], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v213, off
.LBB0_164:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or3_b32 v130, v19, v27, 17
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[0:1], s[0:1], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[0:1], s[30:31], v[130:131]
	s_nop 1
	v_writelane_b32 v255, s0, 22
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[0:1]
	s_nop 0
	v_writelane_b32 v255, s1, 23
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_166
; %bb.165:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[2:3], v[0:1], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[2:3], v213, off
.LBB0_166:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or3_b32 v130, v19, v27, 18
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[2:3], s[0:1], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[0:1], s[30:31], v[130:131]
	s_nop 1
	v_writelane_b32 v255, s0, 24
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[0:1]
	s_nop 0
	v_writelane_b32 v255, s1, 25
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_168
; %bb.167:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[28:29], v[2:3], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[28:29], v212, off
.LBB0_168:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or3_b32 v130, v19, v27, 19
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[132:133], s[0:1], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[0:1], s[30:31], v[130:131]
	s_nop 1
	v_writelane_b32 v255, s0, 26
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[0:1]
	s_nop 0
	v_writelane_b32 v255, s1, 27
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_170
; %bb.169:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[28:29], v[132:133], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[28:29], v212, off
.LBB0_170:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or3_b32 v130, v19, v27, 20
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[212:213], s[0:1], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[0:1], s[30:31], v[130:131]
	s_nop 1
	v_writelane_b32 v255, s0, 28
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[0:1]
	s_nop 0
	v_writelane_b32 v255, s1, 29
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_172
; %bb.171:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[28:29], v[212:213], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[28:29], v211, off
.LBB0_172:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or3_b32 v130, v19, v27, 21
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[28:29], s[0:1], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[0:1], s[30:31], v[130:131]
	s_nop 1
	v_writelane_b32 v255, s0, 30
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[0:1]
	s_nop 0
	v_writelane_b32 v255, s1, 31
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_174
; %bb.173:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[30:31], v[28:29], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[30:31], v211, off
.LBB0_174:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or3_b32 v130, v19, v27, 22
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[30:31], s[0:1], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[0:1], s[30:31], v[130:131]
	s_nop 1
	v_writelane_b32 v255, s0, 32
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[0:1]
	s_nop 0
	v_writelane_b32 v255, s1, 33
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_176
; %bb.175:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[32:33], v[30:31], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[32:33], v210, off
.LBB0_176:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or3_b32 v130, v19, v27, 23
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[32:33], s[0:1], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[0:1], s[30:31], v[130:131]
	s_nop 1
	v_writelane_b32 v255, s0, 34
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[0:1]
	s_nop 0
	v_writelane_b32 v255, s1, 35
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_178
; %bb.177:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[34:35], v[32:33], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[34:35], v210, off
.LBB0_178:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or3_b32 v130, v19, v27, 64
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[34:35], s[0:1], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[0:1], s[30:31], v[130:131]
	s_nop 1
	v_writelane_b32 v255, s0, 36
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[0:1]
	s_nop 0
	v_writelane_b32 v255, s1, 37
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_180
; %bb.179:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[36:37], v[34:35], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[36:37], v254, off
.LBB0_180:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x41, v208
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[36:37], s[0:1], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[0:1], s[30:31], v[130:131]
	s_nop 1
	v_writelane_b32 v255, s0, 38
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[0:1]
	s_nop 0
	v_writelane_b32 v255, s1, 39
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_182
; %bb.181:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[38:39], v[36:37], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[38:39], v254, off
.LBB0_182:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x42, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[40:41], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[38:39], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[40:41]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_184
; %bb.183:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[40:41], v[38:39], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[40:41], v253, off
.LBB0_184:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x43, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[42:43], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[40:41], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[42:43]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_186
; %bb.185:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[42:43], v[40:41], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[42:43], v253, off
.LBB0_186:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x44, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[44:45], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[42:43], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[44:45]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_188
; %bb.187:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[44:45], v[42:43], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[44:45], v252, off
.LBB0_188:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x45, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[46:47], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[44:45], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[46:47]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_190
; %bb.189:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[46:47], v[44:45], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[46:47], v252, off
.LBB0_190:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x46, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[48:49], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[46:47], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[48:49]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_192
; %bb.191:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[48:49], v[46:47], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[48:49], v251, off
.LBB0_192:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x47, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[50:51], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[48:49], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[50:51]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_194
; %bb.193:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[50:51], v[48:49], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[50:51], v251, off
.LBB0_194:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x50, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[52:53], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[50:51], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[52:53]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_196
; %bb.195:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[52:53], v[50:51], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[52:53], v250, off
.LBB0_196:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x51, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[54:55], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[52:53], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[54:55]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_198
; %bb.197:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[54:55], v[52:53], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[54:55], v250, off
.LBB0_198:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x52, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[56:57], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[54:55], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[56:57]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_200
; %bb.199:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[56:57], v[54:55], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[56:57], v249, off
.LBB0_200:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x53, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[58:59], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[56:57], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[58:59]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_202
; %bb.201:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[58:59], v[56:57], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[58:59], v249, off
.LBB0_202:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x54, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[60:61], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[58:59], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[60:61]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_204
; %bb.203:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[60:61], v[58:59], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[60:61], v248, off
.LBB0_204:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x55, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[62:63], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[60:61], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[62:63]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_206
; %bb.205:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[62:63], v[60:61], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[62:63], v248, off
.LBB0_206:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x56, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[64:65], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[62:63], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[64:65]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_208
; %bb.207:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[64:65], v[62:63], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[64:65], v247, off
.LBB0_208:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x57, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[66:67], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[64:65], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[66:67]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_210
; %bb.209:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[66:67], v[64:65], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[66:67], v247, off
.LBB0_210:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x80, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[68:69], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[66:67], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[68:69]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_212
; %bb.211:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[68:69], v[66:67], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[68:69], v246, off
.LBB0_212:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x81, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[70:71], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[68:69], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[70:71]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_214
; %bb.213:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[70:71], v[68:69], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[70:71], v246, off
.LBB0_214:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x82, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[72:73], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[70:71], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[72:73]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_216
; %bb.215:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[72:73], v[70:71], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[72:73], v245, off
.LBB0_216:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x83, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[74:75], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[72:73], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[74:75]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_218
; %bb.217:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[74:75], v[72:73], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[74:75], v245, off
.LBB0_218:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x84, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[76:77], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[74:75], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[76:77]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_220
; %bb.219:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[76:77], v[74:75], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[76:77], v244, off
.LBB0_220:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x85, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[78:79], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[76:77], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[78:79]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_222
; %bb.221:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[78:79], v[76:77], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[78:79], v244, off
.LBB0_222:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x86, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[80:81], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[78:79], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[80:81]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_224
; %bb.223:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[80:81], v[78:79], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[80:81], v243, off
.LBB0_224:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x87, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[82:83], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[80:81], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[82:83]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_226
; %bb.225:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[82:83], v[80:81], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[82:83], v243, off
.LBB0_226:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x90, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[84:85], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[82:83], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[84:85]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_228
; %bb.227:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[84:85], v[82:83], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[84:85], v26, off
.LBB0_228:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x91, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[86:87], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[84:85], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[86:87]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_230
; %bb.229:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[86:87], v[84:85], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[86:87], v26, off
.LBB0_230:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x92, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[88:89], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[86:87], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[88:89]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_232
; %bb.231:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[26:27], v[86:87], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[26:27], v25, off
.LBB0_232:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x93, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[90:91], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[88:89], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[90:91]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_234
; %bb.233:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[26:27], v[88:89], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[26:27], v25, off
.LBB0_234:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x94, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[92:93], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[90:91], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[92:93]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_236
; %bb.235:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[26:27], v[90:91], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[26:27], v24, off
.LBB0_236:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x95, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[94:95], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[92:93], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[94:95]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_238
; %bb.237:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[26:27], v[92:93], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[26:27], v24, off
.LBB0_238:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x96, v208
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[94:95], s[0:1], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[96:97], s[30:31], v[130:131]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[96:97]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_mov_b64 s[0:1], exec
	s_and_b64 s[2:3], s[0:1], s[2:3]
	v_accvgpr_read_b32 v27, a63
	s_mov_b64 exec, s[2:3]
	s_cbranch_execz .LBB0_240
; %bb.239:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[94:95], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v236, off
.LBB0_240:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0x97, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[98:99], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[96:97], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[98:99]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_242
; %bb.241:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[96:97], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v236, off
.LBB0_242:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0xc0, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[6:7], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[98:99], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[6:7]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_244
; %bb.243:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[98:99], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v239, off
.LBB0_244:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0xc1, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[12:13], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[100:101], s[0:1], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[12:13]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_246
; %bb.245:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[100:101], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v239, off
.LBB0_246:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0xc2, v208
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[102:103], s[0:1], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[0:1], s[30:31], v[130:131]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[4:5], vcc, s[0:1]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[2:3], s[4:5]
	s_cbranch_execz .LBB0_248
; %bb.247:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[102:103], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v238, off
.LBB0_248:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[2:3]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0xc3, v208
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[104:105], s[2:3], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[2:3], s[30:31], v[130:131]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[8:9], vcc, s[2:3]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[4:5], s[8:9]
	s_cbranch_execz .LBB0_250
; %bb.249:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[104:105], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v238, off
.LBB0_250:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[4:5]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0xc4, v208
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[106:107], s[4:5], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[4:5], s[30:31], v[130:131]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[10:11], vcc, s[4:5]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[8:9], s[10:11]
	s_cbranch_execz .LBB0_252
; %bb.251:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[106:107], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v237, off
.LBB0_252:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[8:9]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0xc5, v208
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[108:109], s[8:9], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[8:9], s[30:31], v[130:131]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[14:15], vcc, s[8:9]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[10:11], s[14:15]
	s_cbranch_execz .LBB0_254
; %bb.253:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[108:109], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v237, off
.LBB0_254:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[10:11]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0xc6, v208
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[110:111], s[10:11], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[10:11], s[30:31], v[130:131]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[16:17], vcc, s[10:11]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[14:15], s[16:17]
	s_cbranch_execz .LBB0_256
; %bb.255:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[110:111], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v235, off
.LBB0_256:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[14:15]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0xc7, v208
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[112:113], s[14:15], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[14:15], s[30:31], v[130:131]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[18:19], vcc, s[14:15]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[16:17], s[18:19]
	s_cbranch_execz .LBB0_258
; %bb.257:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[112:113], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v235, off
.LBB0_258:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[16:17]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0xd0, v208
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[114:115], s[16:17], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[16:17], s[30:31], v[130:131]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[20:21], vcc, s[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[18:19], s[20:21]
	s_cbranch_execz .LBB0_260
; %bb.259:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[114:115], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v234, off
.LBB0_260:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[18:19]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0xd1, v208
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[116:117], s[18:19], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[18:19], s[30:31], v[130:131]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[22:23], vcc, s[18:19]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[20:21], s[22:23]
	s_cbranch_execz .LBB0_262
; %bb.261:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[116:117], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v234, off
.LBB0_262:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[20:21]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0xd2, v208
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[118:119], s[20:21], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[20:21], s[30:31], v[130:131]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[24:25], vcc, s[20:21]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[22:23], s[24:25]
	s_cbranch_execz .LBB0_264
; %bb.263:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[118:119], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v233, off
.LBB0_264:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[22:23]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0xd3, v208
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[120:121], s[22:23], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[22:23], s[30:31], v[130:131]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[26:27], vcc, s[22:23]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[24:25], s[26:27]
	s_cbranch_execz .LBB0_266
; %bb.265:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[120:121], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v233, off
.LBB0_266:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[24:25]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0xd4, v208
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[122:123], s[24:25], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[24:25], s[30:31], v[130:131]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[28:29], vcc, s[24:25]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[26:27], s[28:29]
	s_cbranch_execz .LBB0_268
; %bb.267:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[122:123], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v232, off
.LBB0_268:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[26:27]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0xd5, v208
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[124:125], s[26:27], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[26:27], s[30:31], v[130:131]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[26:27]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[28:29], s[36:37]
	s_cbranch_execz .LBB0_270
; %bb.269:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[124:125], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v232, off
.LBB0_270:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[28:29]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0xd6, v208
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[126:127], s[28:29], v130, s35, 0
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[28:29], s[30:31], v[130:131]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[28:29]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_272
; %bb.271:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[126:127], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v199, off
.LBB0_272:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 143 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:143:15
	v_or_b32_e32 v130, 0xd7, v208
	.loc	1 145 92                        ; compile_native_aiter_afp4_recovered.py:145:92
	v_cmp_gt_i64_e64 s[30:31], s[30:31], v[130:131]
	.loc	1 144 53                        ; compile_native_aiter_afp4_recovered.py:144:53
	v_mad_i64_i32 v[128:129], s[36:37], v130, s35, 0
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[30:31]
	.loc	1 145 5 is_stmt 0               ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_274
; %bb.273:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[128:129], 1, v[192:193]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v199, off
.LBB0_274:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	v_accvgpr_read_b32 v19, a64
	.loc	1 142 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:142:15
	v_or3_b32 v26, v194, v19, 64
	.loc	1 144 22                        ; compile_native_aiter_afp4_recovered.py:144:22
	v_mad_i64_i32 v[24:25], s[36:37], v26, s34, 0
	.loc	1 144 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:144:14
	v_readlane_b32 s36, v255, 0
	v_readlane_b32 s37, v255, 1
	s_nop 1
	v_lshl_add_u64 v[130:131], v[24:25], 1, s[36:37]
	.loc	1 145 67 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:145:67
	v_readlane_b32 s36, v255, 2
	v_readlane_b32 s37, v255, 3
	s_nop 1
	v_cmp_gt_i64_e32 vcc, s[36:37], v[26:27]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 4
	v_readlane_b32 s37, v255, 5
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_276
; %bb.275:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	v_accvgpr_read_b32 v25, a3
	v_accvgpr_read_b32 v24, a2
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[24:25], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v198, off
.LBB0_276:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 6
	v_readlane_b32 s37, v255, 7
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_278
; %bb.277:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	v_accvgpr_read_b32 v25, a5
	v_accvgpr_read_b32 v24, a4
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[24:25], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v198, off
.LBB0_278:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 8
	v_readlane_b32 s37, v255, 9
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_280
; %bb.279:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	v_accvgpr_read_b32 v25, a7
	v_accvgpr_read_b32 v24, a6
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[24:25], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v197, off
.LBB0_280:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 10
	v_readlane_b32 s37, v255, 11
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_282
; %bb.281:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[4:5], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v197, off
.LBB0_282:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 12
	v_readlane_b32 s37, v255, 13
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_284
; %bb.283:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[6:7], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v196, off
.LBB0_284:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 14
	v_readlane_b32 s37, v255, 15
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_286
; %bb.285:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[8:9], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v196, off
.LBB0_286:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 16
	v_readlane_b32 s37, v255, 17
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_288
; %bb.287:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[10:11], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v195, off
.LBB0_288:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 18
	v_readlane_b32 s37, v255, 19
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_290
; %bb.289:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[12:13], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v195, off
.LBB0_290:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 20
	v_readlane_b32 s37, v255, 21
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_292
; %bb.291:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[14:15], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v17, off
.LBB0_292:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 22
	v_readlane_b32 s37, v255, 23
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_294
; %bb.293:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[0:1], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v17, off
.LBB0_294:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 24
	v_readlane_b32 s37, v255, 25
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_296
; %bb.295:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[2:3], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v225, off
.LBB0_296:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 26
	v_readlane_b32 s37, v255, 27
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_298
; %bb.297:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[132:133], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v225, off
.LBB0_298:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 28
	v_readlane_b32 s37, v255, 29
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_300
; %bb.299:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[212:213], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v223, off
.LBB0_300:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 30
	v_readlane_b32 s37, v255, 31
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_302
; %bb.301:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[28:29], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v223, off
.LBB0_302:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 32
	v_readlane_b32 s37, v255, 33
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_304
; %bb.303:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[30:31], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v222, off
.LBB0_304:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 34
	v_readlane_b32 s37, v255, 35
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_306
; %bb.305:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[32:33], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v222, off
.LBB0_306:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 36
	v_readlane_b32 s37, v255, 37
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_308
; %bb.307:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[34:35], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v221, off
.LBB0_308:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 38
	v_readlane_b32 s37, v255, 39
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_520
; %bb.309:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[40:41]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_521
.LBB0_310:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[42:43]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_522
.LBB0_311:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[44:45]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_523
.LBB0_312:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[46:47]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_524
.LBB0_313:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[48:49]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_525
.LBB0_314:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[50:51]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_526
.LBB0_315:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[52:53]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_527
.LBB0_316:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[54:55]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_528
.LBB0_317:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[56:57]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_529
.LBB0_318:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[58:59]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_530
.LBB0_319:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[60:61]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_531
.LBB0_320:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[62:63]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_532
.LBB0_321:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[64:65]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_533
.LBB0_322:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[66:67]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_534
.LBB0_323:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[68:69]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_535
.LBB0_324:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[70:71]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_536
.LBB0_325:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[72:73]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_537
.LBB0_326:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[74:75]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_538
.LBB0_327:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[76:77]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_539
.LBB0_328:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[78:79]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_540
.LBB0_329:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[80:81]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_541
.LBB0_330:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[82:83]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_542
.LBB0_331:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[84:85]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_543
.LBB0_332:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[86:87]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_544
.LBB0_333:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[88:89]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_545
.LBB0_334:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[90:91]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_546
.LBB0_335:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[92:93]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_547
.LBB0_336:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[94:95]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_548
.LBB0_337:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[96:97]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_549
.LBB0_338:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[98:99]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_550
.LBB0_339:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[6:7]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_551
.LBB0_340:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[12:13]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_552
.LBB0_341:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[0:1]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_553
.LBB0_342:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[2:3]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_554
.LBB0_343:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[4:5]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_555
.LBB0_344:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[8:9]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_556
.LBB0_345:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[10:11]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_557
.LBB0_346:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[14:15]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_558
.LBB0_347:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_559
.LBB0_348:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[18:19]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_560
.LBB0_349:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[20:21]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_561
.LBB0_350:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[22:23]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_562
.LBB0_351:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[24:25]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_563
.LBB0_352:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[26:27]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_564
.LBB0_353:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[28:29]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_565
.LBB0_354:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[30:31]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_356
.LBB0_355:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[128:129], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v229, off
.LBB0_356:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 142 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:142:15
	v_or_b32_e32 v26, 0x80, v16
	.loc	1 144 22                        ; compile_native_aiter_afp4_recovered.py:144:22
	v_mad_i64_i32 v[18:19], s[36:37], v26, s34, 0
	.loc	1 144 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:144:14
	v_readlane_b32 s36, v255, 0
	v_readlane_b32 s37, v255, 1
	s_nop 1
	v_lshl_add_u64 v[130:131], v[18:19], 1, s[36:37]
	.loc	1 145 67 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:145:67
	v_readlane_b32 s36, v255, 2
	v_readlane_b32 s37, v255, 3
	s_nop 1
	v_cmp_gt_i64_e32 vcc, s[36:37], v[26:27]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 4
	v_readlane_b32 s37, v255, 5
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_358
; %bb.357:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	v_accvgpr_read_b32 v19, a3
	v_accvgpr_read_b32 v18, a2
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v228, off
.LBB0_358:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 6
	v_readlane_b32 s37, v255, 7
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_360
; %bb.359:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	v_accvgpr_read_b32 v19, a5
	v_accvgpr_read_b32 v18, a4
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v228, off
.LBB0_360:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 8
	v_readlane_b32 s37, v255, 9
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_362
; %bb.361:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	v_accvgpr_read_b32 v19, a7
	v_accvgpr_read_b32 v18, a6
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v227, off
.LBB0_362:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 10
	v_readlane_b32 s37, v255, 11
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_364
; %bb.363:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[4:5], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v227, off
.LBB0_364:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 12
	v_readlane_b32 s37, v255, 13
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_366
; %bb.365:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[6:7], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v224, off
.LBB0_366:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 14
	v_readlane_b32 s37, v255, 15
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_368
; %bb.367:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[8:9], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v224, off
.LBB0_368:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 16
	v_readlane_b32 s37, v255, 17
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_370
; %bb.369:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[10:11], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v191, off
.LBB0_370:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 18
	v_readlane_b32 s37, v255, 19
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_372
; %bb.371:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[12:13], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v191, off
.LBB0_372:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 20
	v_readlane_b32 s37, v255, 21
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_374
; %bb.373:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[14:15], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v190, off
.LBB0_374:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 22
	v_readlane_b32 s37, v255, 23
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_376
; %bb.375:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[0:1], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v190, off
.LBB0_376:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 24
	v_readlane_b32 s37, v255, 25
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_378
; %bb.377:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[2:3], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v189, off
.LBB0_378:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 26
	v_readlane_b32 s37, v255, 27
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_380
; %bb.379:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[132:133], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v189, off
.LBB0_380:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 28
	v_readlane_b32 s37, v255, 29
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_382
; %bb.381:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[212:213], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v188, off
.LBB0_382:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 30
	v_readlane_b32 s37, v255, 31
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_384
; %bb.383:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[28:29], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v188, off
.LBB0_384:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 32
	v_readlane_b32 s37, v255, 33
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_386
; %bb.385:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[30:31], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v187, off
.LBB0_386:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 34
	v_readlane_b32 s37, v255, 35
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_388
; %bb.387:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[32:33], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v187, off
.LBB0_388:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 36
	v_readlane_b32 s37, v255, 37
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_390
; %bb.389:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[34:35], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v186, off
.LBB0_390:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s36, v255, 38
	v_readlane_b32 s37, v255, 39
	s_and_b64 s[38:39], vcc, s[36:37]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_566
; %bb.391:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[40:41]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_567
.LBB0_392:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[42:43]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_568
.LBB0_393:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[44:45]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_569
.LBB0_394:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[46:47]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_570
.LBB0_395:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[48:49]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_571
.LBB0_396:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[50:51]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_572
.LBB0_397:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[52:53]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_573
.LBB0_398:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[54:55]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_574
.LBB0_399:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[56:57]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_575
.LBB0_400:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[58:59]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_576
.LBB0_401:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[60:61]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_577
.LBB0_402:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[62:63]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_578
.LBB0_403:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[64:65]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_579
.LBB0_404:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[66:67]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_580
.LBB0_405:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[68:69]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_581
.LBB0_406:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[70:71]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_582
.LBB0_407:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[72:73]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_583
.LBB0_408:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[74:75]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_584
.LBB0_409:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[76:77]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_585
.LBB0_410:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[78:79]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_586
.LBB0_411:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[80:81]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_587
.LBB0_412:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[82:83]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_588
.LBB0_413:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[84:85]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_589
.LBB0_414:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[86:87]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_590
.LBB0_415:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[88:89]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_591
.LBB0_416:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[90:91]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_592
.LBB0_417:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[92:93]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_593
.LBB0_418:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[94:95]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_594
.LBB0_419:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[96:97]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_595
.LBB0_420:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[98:99]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_596
.LBB0_421:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[6:7]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_597
.LBB0_422:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[12:13]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_598
.LBB0_423:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[0:1]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_599
.LBB0_424:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[2:3]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_600
.LBB0_425:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[4:5]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_601
.LBB0_426:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[8:9]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_602
.LBB0_427:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[10:11]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_603
.LBB0_428:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[14:15]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_604
.LBB0_429:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_605
.LBB0_430:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[18:19]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_606
.LBB0_431:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[20:21]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_607
.LBB0_432:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[22:23]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_608
.LBB0_433:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[24:25]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_609
.LBB0_434:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[26:27]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_610
.LBB0_435:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[28:29]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_611
.LBB0_436:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[30:31]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_438
.LBB0_437:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[128:129], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v226, off
.LBB0_438:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[36:37]
	.loc	1 142 15 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:142:15
	v_or_b32_e32 v26, 0xc0, v16
	.loc	1 144 22                        ; compile_native_aiter_afp4_recovered.py:144:22
	v_mad_i64_i32 v[16:17], s[34:35], v26, s34, 0
	.loc	1 144 14 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:144:14
	v_readlane_b32 s34, v255, 0
	v_readlane_b32 s35, v255, 1
	s_nop 1
	v_lshl_add_u64 v[16:17], v[16:17], 1, s[34:35]
	.loc	1 145 67 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:145:67
	v_readlane_b32 s34, v255, 2
	v_readlane_b32 s35, v255, 3
	s_nop 1
	v_cmp_gt_i64_e32 vcc, s[34:35], v[26:27]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s34, v255, 4
	v_readlane_b32 s35, v255, 5
	s_and_b64 s[36:37], vcc, s[34:35]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_440
; %bb.439:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	v_accvgpr_read_b32 v19, a3
	v_accvgpr_read_b32 v18, a2
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v163, off
.LBB0_440:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s34, v255, 6
	v_readlane_b32 s35, v255, 7
	s_and_b64 s[36:37], vcc, s[34:35]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_442
; %bb.441:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	v_accvgpr_read_b32 v19, a5
	v_accvgpr_read_b32 v18, a4
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v163, off
.LBB0_442:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s34, v255, 8
	v_readlane_b32 s35, v255, 9
	s_and_b64 s[36:37], vcc, s[34:35]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_444
; %bb.443:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	v_accvgpr_read_b32 v19, a7
	v_accvgpr_read_b32 v18, a6
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[18:19], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v162, off
.LBB0_444:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s34, v255, 10
	v_readlane_b32 s35, v255, 11
	s_and_b64 s[36:37], vcc, s[34:35]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_446
; %bb.445:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[4:5], v[4:5], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[4:5], v162, off
.LBB0_446:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s34, v255, 12
	v_readlane_b32 s35, v255, 13
	s_and_b64 s[36:37], vcc, s[34:35]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_448
; %bb.447:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[4:5], v[6:7], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[4:5], v161, off
.LBB0_448:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s34, v255, 14
	v_readlane_b32 s35, v255, 15
	s_and_b64 s[36:37], vcc, s[34:35]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_450
; %bb.449:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[4:5], v[8:9], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[4:5], v161, off
.LBB0_450:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s34, v255, 16
	v_readlane_b32 s35, v255, 17
	s_and_b64 s[36:37], vcc, s[34:35]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_452
; %bb.451:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[4:5], v[10:11], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[4:5], v160, off
.LBB0_452:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s34, v255, 18
	v_readlane_b32 s35, v255, 19
	s_and_b64 s[36:37], vcc, s[34:35]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_454
; %bb.453:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[4:5], v[12:13], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[4:5], v160, off
.LBB0_454:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s34, v255, 20
	v_readlane_b32 s35, v255, 21
	s_and_b64 s[36:37], vcc, s[34:35]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_456
; %bb.455:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[4:5], v[14:15], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[4:5], v159, off
.LBB0_456:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s34, v255, 22
	v_readlane_b32 s35, v255, 23
	s_and_b64 s[36:37], vcc, s[34:35]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_458
; %bb.457:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[0:1], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v159, off
.LBB0_458:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s34, v255, 24
	v_readlane_b32 s35, v255, 25
	s_and_b64 s[36:37], vcc, s[34:35]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_460
; %bb.459:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[2:3], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v158, off
.LBB0_460:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s34, v255, 26
	v_readlane_b32 s35, v255, 27
	s_and_b64 s[36:37], vcc, s[34:35]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_462
; %bb.461:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[132:133], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v158, off
.LBB0_462:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s34, v255, 28
	v_readlane_b32 s35, v255, 29
	s_and_b64 s[36:37], vcc, s[34:35]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_464
; %bb.463:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[212:213], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v157, off
.LBB0_464:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s34, v255, 30
	v_readlane_b32 s35, v255, 31
	s_and_b64 s[36:37], vcc, s[34:35]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_466
; %bb.465:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[28:29], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v157, off
.LBB0_466:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s34, v255, 32
	v_readlane_b32 s35, v255, 33
	s_and_b64 s[36:37], vcc, s[34:35]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_468
; %bb.467:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[30:31], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v156, off
.LBB0_468:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s34, v255, 34
	v_readlane_b32 s35, v255, 35
	s_and_b64 s[36:37], vcc, s[34:35]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_470
; %bb.469:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[32:33], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v156, off
.LBB0_470:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s34, v255, 36
	v_readlane_b32 s35, v255, 37
	s_and_b64 s[36:37], vcc, s[34:35]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_472
; %bb.471:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[34:35], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v155, off
.LBB0_472:
	.loc	1 0 5 is_stmt 0                 ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	v_readlane_b32 s34, v255, 38
	v_readlane_b32 s35, v255, 39
	s_and_b64 s[36:37], vcc, s[34:35]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_612
; %bb.473:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[40:41]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_613
.LBB0_474:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[42:43]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_614
.LBB0_475:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[44:45]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_615
.LBB0_476:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[46:47]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_616
.LBB0_477:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[48:49]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_617
.LBB0_478:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[50:51]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_618
.LBB0_479:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[52:53]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_619
.LBB0_480:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[54:55]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_620
.LBB0_481:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[56:57]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_621
.LBB0_482:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[58:59]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_622
.LBB0_483:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[60:61]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_623
.LBB0_484:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[62:63]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_624
.LBB0_485:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[64:65]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_625
.LBB0_486:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[66:67]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_626
.LBB0_487:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[68:69]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_627
.LBB0_488:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[70:71]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_628
.LBB0_489:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[72:73]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_629
.LBB0_490:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[74:75]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_630
.LBB0_491:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[76:77]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_631
.LBB0_492:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[78:79]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_632
.LBB0_493:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[80:81]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_633
.LBB0_494:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[82:83]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_634
.LBB0_495:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[84:85]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_635
.LBB0_496:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[86:87]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_636
.LBB0_497:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[88:89]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_637
.LBB0_498:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[90:91]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_638
.LBB0_499:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[92:93]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_639
.LBB0_500:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[94:95]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_640
.LBB0_501:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[96:97]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_641
.LBB0_502:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[98:99]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execnz .LBB0_642
.LBB0_503:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[34:35], vcc, s[6:7]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[6:7], s[34:35]
	s_cbranch_execnz .LBB0_643
.LBB0_504:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[6:7]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[12:13], vcc, s[12:13]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[6:7], s[12:13]
	s_cbranch_execnz .LBB0_644
.LBB0_505:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[6:7]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[6:7], vcc, s[0:1]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[6:7]
	s_cbranch_execnz .LBB0_645
.LBB0_506:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_646
.LBB0_507:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[4:5]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_647
.LBB0_508:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[8:9]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_648
.LBB0_509:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[10:11]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_649
.LBB0_510:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[14:15]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_650
.LBB0_511:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_651
.LBB0_512:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[18:19]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_652
.LBB0_513:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[20:21]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_653
.LBB0_514:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[22:23]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_654
.LBB0_515:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[24:25]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_655
.LBB0_516:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[26:27]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_656
.LBB0_517:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[28:29]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execnz .LBB0_657
.LBB0_518:
	.loc	1 0 5                           ; compile_native_aiter_afp4_recovered.py:0:5
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66                        ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[0:1], vcc, s[30:31]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execnz .LBB0_658
.LBB0_519:
	.loc	1 89 1 is_stmt 1                ; compile_native_aiter_afp4_recovered.py:89:1
	s_endpgm
.LBB0_520:
	.loc	1 144 14                        ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[36:37], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v221, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[40:41]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_310
.LBB0_521:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[38:39], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v220, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[42:43]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_311
.LBB0_522:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[40:41], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v220, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[44:45]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_312
.LBB0_523:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[42:43], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v219, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[46:47]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_313
.LBB0_524:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[44:45], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v219, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[48:49]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_314
.LBB0_525:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[46:47], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v218, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[50:51]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_315
.LBB0_526:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[48:49], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v218, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[52:53]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_316
.LBB0_527:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[50:51], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v217, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[54:55]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_317
.LBB0_528:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[52:53], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v217, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[56:57]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_318
.LBB0_529:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[54:55], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v216, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[58:59]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_319
.LBB0_530:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[56:57], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v216, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[60:61]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_320
.LBB0_531:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[58:59], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v215, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[62:63]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_321
.LBB0_532:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[60:61], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v215, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[64:65]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_322
.LBB0_533:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[62:63], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v23, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[66:67]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_323
.LBB0_534:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[64:65], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v23, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[68:69]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_324
.LBB0_535:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[66:67], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[24:25], v22, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[70:71]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_325
.LBB0_536:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[24:25], v[68:69], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[24:25], v22, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[72:73]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_326
.LBB0_537:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[22:23], v[70:71], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[22:23], v21, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[74:75]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_327
.LBB0_538:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[22:23], v[72:73], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[22:23], v21, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[76:77]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_328
.LBB0_539:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[22:23], v[74:75], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[22:23], v20, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[78:79]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_329
.LBB0_540:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[22:23], v[76:77], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[22:23], v20, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[80:81]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_330
.LBB0_541:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[20:21], v[78:79], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[20:21], v18, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[82:83]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_331
.LBB0_542:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[20:21], v[80:81], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[20:21], v18, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[84:85]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_332
.LBB0_543:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[82:83], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v209, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[86:87]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_333
.LBB0_544:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[84:85], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v209, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[88:89]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_334
.LBB0_545:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[86:87], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v207, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[90:91]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_335
.LBB0_546:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[88:89], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v207, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[92:93]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_336
.LBB0_547:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[90:91], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v206, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[94:95]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_337
.LBB0_548:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[92:93], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v206, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[96:97]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_338
.LBB0_549:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[94:95], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v205, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[98:99]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_339
.LBB0_550:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[96:97], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v205, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[6:7]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_340
.LBB0_551:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[98:99], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v204, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[12:13]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_341
.LBB0_552:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[100:101], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v204, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[0:1]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_342
.LBB0_553:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[102:103], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v203, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[2:3]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_343
.LBB0_554:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[104:105], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v203, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[4:5]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_344
.LBB0_555:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[106:107], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v202, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[8:9]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_345
.LBB0_556:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[108:109], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v202, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[10:11]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_346
.LBB0_557:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[110:111], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v201, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[14:15]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_347
.LBB0_558:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[112:113], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v201, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_348
.LBB0_559:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[114:115], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v200, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[18:19]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_349
.LBB0_560:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[116:117], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v200, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[20:21]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_350
.LBB0_561:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[118:119], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v231, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[22:23]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_351
.LBB0_562:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[120:121], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v231, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[24:25]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_352
.LBB0_563:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[122:123], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v230, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[26:27]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_353
.LBB0_564:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[124:125], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v230, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[28:29]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_354
.LBB0_565:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[126:127], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v229, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[30:31]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_355
	s_branch .LBB0_356
.LBB0_566:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[36:37], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v186, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[40:41]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_392
.LBB0_567:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[38:39], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v185, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[42:43]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_393
.LBB0_568:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[40:41], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v185, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[44:45]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_394
.LBB0_569:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[42:43], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v184, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[46:47]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_395
.LBB0_570:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[44:45], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v184, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[48:49]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_396
.LBB0_571:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[46:47], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v183, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[50:51]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_397
.LBB0_572:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[48:49], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v183, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[52:53]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_398
.LBB0_573:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[50:51], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v182, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[54:55]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_399
.LBB0_574:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[52:53], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v182, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[56:57]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_400
.LBB0_575:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[54:55], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v181, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[58:59]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_401
.LBB0_576:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[56:57], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v181, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[60:61]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_402
.LBB0_577:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[58:59], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v180, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[62:63]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_403
.LBB0_578:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[60:61], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v180, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[64:65]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_404
.LBB0_579:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[62:63], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v179, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[66:67]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_405
.LBB0_580:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[64:65], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v179, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[68:69]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_406
.LBB0_581:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[66:67], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v178, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[70:71]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_407
.LBB0_582:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[68:69], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v178, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[72:73]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_408
.LBB0_583:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[70:71], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v177, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[74:75]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_409
.LBB0_584:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[72:73], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v177, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[76:77]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_410
.LBB0_585:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[74:75], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v176, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[78:79]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_411
.LBB0_586:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[76:77], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v176, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[80:81]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_412
.LBB0_587:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[78:79], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v175, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[82:83]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_413
.LBB0_588:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[80:81], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v175, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[84:85]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_414
.LBB0_589:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[82:83], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v174, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[86:87]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_415
.LBB0_590:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[84:85], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v174, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[88:89]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_416
.LBB0_591:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[86:87], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v173, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[90:91]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_417
.LBB0_592:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[88:89], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v173, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[92:93]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_418
.LBB0_593:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[90:91], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v172, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[94:95]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_419
.LBB0_594:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[92:93], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v172, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[96:97]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_420
.LBB0_595:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[94:95], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v171, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[98:99]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_421
.LBB0_596:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[96:97], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v171, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[6:7]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_422
.LBB0_597:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[98:99], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v170, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[12:13]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_423
.LBB0_598:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[100:101], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v170, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[0:1]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_424
.LBB0_599:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[102:103], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v169, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[2:3]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_425
.LBB0_600:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[104:105], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v169, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[4:5]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_426
.LBB0_601:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[106:107], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v168, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[8:9]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_427
.LBB0_602:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[108:109], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v168, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[10:11]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_428
.LBB0_603:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[110:111], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v167, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[14:15]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_429
.LBB0_604:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[112:113], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v167, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_430
.LBB0_605:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[114:115], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v166, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[18:19]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_431
.LBB0_606:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[116:117], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v166, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[20:21]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_432
.LBB0_607:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[118:119], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v165, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[22:23]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_433
.LBB0_608:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[120:121], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v165, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[24:25]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_434
.LBB0_609:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[122:123], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v164, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[26:27]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_435
.LBB0_610:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[124:125], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[18:19], v164, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[28:29]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execz .LBB0_436
.LBB0_611:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[18:19], v[126:127], 1, v[130:131]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[18:19], v226, off
	s_or_b64 exec, exec, s[36:37]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[38:39], vcc, s[30:31]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[36:37], s[38:39]
	s_cbranch_execnz .LBB0_437
	s_branch .LBB0_438
.LBB0_612:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[36:37], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v155, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[40:41]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_474
.LBB0_613:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[38:39], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v154, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[42:43]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_475
.LBB0_614:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[40:41], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v154, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[44:45]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_476
.LBB0_615:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[42:43], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v153, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[46:47]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_477
.LBB0_616:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[44:45], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v153, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[48:49]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_478
.LBB0_617:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[46:47], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v152, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[50:51]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_479
.LBB0_618:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[48:49], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v152, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[52:53]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_480
.LBB0_619:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[50:51], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v151, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[54:55]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_481
.LBB0_620:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[52:53], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v151, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[56:57]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_482
.LBB0_621:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[54:55], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v150, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[58:59]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_483
.LBB0_622:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[56:57], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v150, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[60:61]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_484
.LBB0_623:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[58:59], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v149, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[62:63]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_485
.LBB0_624:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[60:61], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v149, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[64:65]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_486
.LBB0_625:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[62:63], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v148, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[66:67]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_487
.LBB0_626:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[64:65], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v148, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[68:69]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_488
.LBB0_627:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[66:67], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v147, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[70:71]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_489
.LBB0_628:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[68:69], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v147, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[72:73]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_490
.LBB0_629:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[70:71], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v146, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[74:75]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_491
.LBB0_630:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[72:73], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v146, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[76:77]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_492
.LBB0_631:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[74:75], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v145, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[78:79]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_493
.LBB0_632:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[76:77], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v145, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[80:81]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_494
.LBB0_633:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[78:79], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v144, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[82:83]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_495
.LBB0_634:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[80:81], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v144, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[84:85]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_496
.LBB0_635:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[82:83], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v143, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[86:87]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_497
.LBB0_636:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[84:85], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v143, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[88:89]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_498
.LBB0_637:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[86:87], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v142, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[90:91]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_499
.LBB0_638:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[88:89], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v142, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[92:93]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_500
.LBB0_639:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[90:91], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v141, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[94:95]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_501
.LBB0_640:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[92:93], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v141, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[96:97]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_502
.LBB0_641:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[94:95], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v140, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[36:37], vcc, s[98:99]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[34:35], s[36:37]
	s_cbranch_execz .LBB0_503
.LBB0_642:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[96:97], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v140, off
	s_or_b64 exec, exec, s[34:35]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[34:35], vcc, s[6:7]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[6:7], s[34:35]
	s_cbranch_execz .LBB0_504
.LBB0_643:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[98:99], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v139, off
	s_or_b64 exec, exec, s[6:7]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[12:13], vcc, s[12:13]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[6:7], s[12:13]
	s_cbranch_execz .LBB0_505
.LBB0_644:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[100:101], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v139, off
	s_or_b64 exec, exec, s[6:7]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[6:7], vcc, s[0:1]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[6:7]
	s_cbranch_execz .LBB0_506
.LBB0_645:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[102:103], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v138, off
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[2:3]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_507
.LBB0_646:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[104:105], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v138, off
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[4:5]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_508
.LBB0_647:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[106:107], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v137, off
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[8:9]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_509
.LBB0_648:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[108:109], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v137, off
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[10:11]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_510
.LBB0_649:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[110:111], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v136, off
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[14:15]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_511
.LBB0_650:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[112:113], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v136, off
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_512
.LBB0_651:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[114:115], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v135, off
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[18:19]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_513
.LBB0_652:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[116:117], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v135, off
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[20:21]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_514
.LBB0_653:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[118:119], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], v134, off
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[22:23]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_515
.LBB0_654:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[120:121], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], v134, off
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[24:25]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_516
.LBB0_655:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[122:123], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], a1, off
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[26:27]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_517
.LBB0_656:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[124:125], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], a1, off
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[2:3], vcc, s[28:29]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[0:1], s[2:3]
	s_cbranch_execz .LBB0_518
.LBB0_657:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[126:127], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short v[0:1], a0, off
	s_or_b64 exec, exec, s[0:1]
	.loc	1 145 66 is_stmt 0              ; compile_native_aiter_afp4_recovered.py:145:66
	s_and_b64 s[0:1], vcc, s[30:31]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	s_and_saveexec_b64 s[2:3], s[0:1]
	s_cbranch_execz .LBB0_519
.LBB0_658:
	.loc	1 144 14 is_stmt 1              ; compile_native_aiter_afp4_recovered.py:144:14
	v_lshl_add_u64 v[0:1], v[128:129], 1, v[16:17]
	.loc	1 145 5                         ; compile_native_aiter_afp4_recovered.py:145:5
	global_store_short_d16_hi v[0:1], a0, off
	.loc	1 89 1                          ; compile_native_aiter_afp4_recovered.py:89:1
	s_endpgm
.Ltmp23:
.Lfunc_end0:
	.size	_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1, .Lfunc_end0-_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1
		.amdhsa_group_segment_fixed_size 8192
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 112
		.amdhsa_user_sgpr_count 16
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 14
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 462
		.amdhsa_next_free_sgpr 100
		.amdhsa_accum_offset 256
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .L_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1.num_vgpr, 256
	.set .L_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1.num_agpr, 206
	.set .L_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1.numbered_sgpr, 100
	.set .L_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1.num_named_barrier, 0
	.set .L_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1.private_seg_size, 0
	.set .L_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1.uses_vcc, 1
	.set .L_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1.uses_flat_scratch, 0
	.set .L_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1.has_dyn_sized_stack, 0
	.set .L_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1.has_recursion, 0
	.set .L_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 30840
; TotalNumSgprs: 106
; NumVgprs: 256
; NumAgprs: 206
; TotalNumVgprs: 462
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 13
; VGPRBlocks: 57
; NumSGPRsForWavesPerEU: 106
; NumVGPRsForWavesPerEU: 462
; AccumOffset: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 16
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 63
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	1                               ; DW_CHILDREN_yes
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	14                              ; DW_FORM_strp
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	2                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	0                               ; DW_CHILDREN_no
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	32                              ; DW_AT_inline
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	3                               ; Abbreviation Code
	.byte	46                              ; DW_TAG_subprogram
	.byte	1                               ; DW_CHILDREN_yes
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	4                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	5                               ; Abbreviation Code
	.byte	29                              ; DW_TAG_inlined_subroutine
	.byte	0                               ; DW_CHILDREN_no
	.byte	49                              ; DW_AT_abstract_origin
	.byte	19                              ; DW_FORM_ref4
	.byte	85                              ; DW_AT_ranges
	.byte	23                              ; DW_FORM_sec_offset
	.byte	88                              ; DW_AT_call_file
	.byte	11                              ; DW_FORM_data1
	.byte	89                              ; DW_AT_call_line
	.byte	11                              ; DW_FORM_data1
	.byte	87                              ; DW_AT_call_column
	.byte	11                              ; DW_FORM_data1
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x78 DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.byte	2                               ; Abbrev [2] 0x2a:0x6 DW_TAG_subprogram
	.long	.Linfo_string3                  ; DW_AT_name
	.byte	1                               ; DW_AT_inline
	.byte	3                               ; Abbrev [3] 0x30:0x52 DW_TAG_subprogram
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
	.long	42                              ; DW_AT_abstract_origin
	.byte	4                               ; Abbrev [4] 0x41:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp1                          ; DW_AT_low_pc
	.long	.Ltmp2-.Ltmp1                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	104                             ; DW_AT_call_line
	.byte	15                              ; DW_AT_call_column
	.byte	4                               ; Abbrev [4] 0x55:0x14 DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.quad	.Ltmp2                          ; DW_AT_low_pc
	.long	.Ltmp3-.Ltmp2                   ; DW_AT_high_pc
	.byte	1                               ; DW_AT_call_file
	.byte	104                             ; DW_AT_call_line
	.byte	42                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x69:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges0                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	105                             ; DW_AT_call_line
	.byte	11                              ; DW_AT_call_column
	.byte	5                               ; Abbrev [5] 0x75:0xc DW_TAG_inlined_subroutine
	.long	42                              ; DW_AT_abstract_origin
	.long	.Ldebug_ranges1                 ; DW_AT_ranges
	.byte	1                               ; DW_AT_call_file
	.byte	108                             ; DW_AT_call_line
	.byte	20                              ; DW_AT_call_column
	.byte	0                               ; End Of Children Mark
	.byte	0                               ; End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Ltmp4-.Lfunc_begin0
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
	.quad	.Ltmp7-.Lfunc_begin0
	.quad	.Ltmp8-.Lfunc_begin0
	.quad	.Ltmp9-.Lfunc_begin0
	.quad	.Ltmp10-.Lfunc_begin0
	.quad	.Ltmp11-.Lfunc_begin0
	.quad	.Ltmp12-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.quad	.Ltmp14-.Lfunc_begin0
	.quad	.Ltmp15-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"triton"                        ; string offset=0 ; triton
.Linfo_string1:
	.asciz	"compile_native_aiter_afp4_recovered.py" ; string offset=7 ; compile_native_aiter_afp4_recovered.py
.Linfo_string2:
	.asciz	"/home/jincheye"                ; string offset=46 ; /home/jincheye
.Linfo_string3:
	.asciz	"_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1" ; string offset=61 ; _gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
custom.config:
  Source:
    Origin: triton
  Version: 1.0.0
  Features:
    SupportsUserArgs: false
    SupportsBias: false
    SupportsActivation: false
    SupportsScaleAlpha: false
    SupportsGSU: false
  InternalSupportParams:
    KernArgsVersion: 0
  ProblemType:
    OperationType: GEMM
    DataType: F4
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: 1
    TransposeB: 0
    UseBeta: False
    Batched: True
    UseBias: 0
    Activation: False
    UseScaleAlphaVec: 0
    SwizzleTensorA: False
    SwizzleTensorB: False
    MXBlockA: 32
    MXBlockB: 32
  CustomKernel:
    args: [ { type: address, semantic: AddressA },
            { type: address, semantic: AddressB },
            { type: address, semantic: AddressD },
            { type: address, semantic: AddressMXScaleA },
            { type: address, semantic: AddressMXScaleB },
            { type: uint32, semantic: SizeFree0 },
            { type: uint32, semantic: SizeFree1 },
            { type: uint32, semantic: SizeSumDiv2 },
            { type: uint32, semantic: StrideA0Bytes },
            { type: uint32, semantic: ConstantOne },
            { type: uint32, semantic: ConstantOne },
            { type: uint32, semantic: StrideB0Bytes },
            { type: uint32, semantic: StrideCK },
            { type: uint32, semantic: ConstantOne },
            { type: uint32, semantic: StrideD0 },
            { type: uint32, semantic: StrideScaleA0 },
            { type: uint32, semantic: ConstantOne },
            { type: uint32, semantic: StrideScaleB0 },
            { type: uint32, semantic: ConstantOne, padding: 16 } ]
    macrotile: [256, 256, 64]
    threads: [256, 1, 1]
    grid: [TilesXY, One, One]
  MatrixInstruction: [16, 16, 128, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [8, 8]
  AssertSummationElementMultiple: 32
  WavefrontSize: 64
amdhsa.kernels:
  - .agpr_count:     206
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         32
        .size:           8
        .value_kind:     global_buffer
      - .offset:         40
        .size:           4
        .value_kind:     by_value
      - .offset:         44
        .size:           4
        .value_kind:     by_value
      - .offset:         48
        .size:           4
        .value_kind:     by_value
      - .offset:         52
        .size:           4
        .value_kind:     by_value
      - .offset:         56
        .size:           4
        .value_kind:     by_value
      - .offset:         60
        .size:           4
        .value_kind:     by_value
      - .offset:         64
        .size:           4
        .value_kind:     by_value
      - .offset:         68
        .size:           4
        .value_kind:     by_value
      - .offset:         72
        .size:           4
        .value_kind:     by_value
      - .offset:         76
        .size:           4
        .value_kind:     by_value
      - .offset:         80
        .size:           4
        .value_kind:     by_value
      - .offset:         84
        .size:           4
        .value_kind:     by_value
      - .offset:         88
        .size:           4
        .value_kind:     by_value
      - .offset:         92
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         96
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         104
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 8192
    .kernarg_segment_align: 8
    .kernarg_segment_size: 112
    .max_flat_workgroup_size: 256
    .name:           _gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1
    .private_segment_fixed_size: 0
    .sgpr_count:     106
    .sgpr_spill_count: 40
    .symbol:         _gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_NUM_KSPLIT_1.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     462
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
