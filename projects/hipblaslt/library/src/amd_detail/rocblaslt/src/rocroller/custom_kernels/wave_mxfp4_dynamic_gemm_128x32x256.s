; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_dynamic_gemm_128x32x256
	.p2align	8
	.type	wave_mxfp4_dynamic_gemm_128x32x256,@function
wave_mxfp4_dynamic_gemm_128x32x256:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	s_load_dwordx2 s[52:53], s[0:1], 0x38
	s_load_dwordx2 s[54:55], s[0:1], 0x48
	s_load_dwordx2 s[56:57], s[0:1], 0x58
	s_mov_b64 s[40:41], s[2:3]
	s_mov_b64 s[36:37], s[6:7]
	s_waitcnt lgkmcnt(0)
	s_sub_u32 s20, 0, s52
	s_subb_u32 s21, 0, s53
	s_add_u32 s22, s52, -1
	v_cmp_lt_i64_e64 s[2:3], s[52:53], 1
	s_addc_u32 s23, s53, -1
	s_and_b64 s[6:7], s[2:3], exec
	s_cselect_b32 s7, s21, s23
	s_cselect_b32 s6, s20, s22
	s_ashr_i32 s20, s7, 31
	s_lshr_b32 s20, s20, 24
	s_add_u32 s6, s6, s20
	s_addc_u32 s7, s7, 0
	s_ashr_i64 s[6:7], s[6:7], 8
	s_sub_u32 s20, 0, s6
	s_subb_u32 s21, 0, s7
	v_and_b32_e32 v43, 0x3ff, v0
	v_bfe_u32 v40, v0, 10, 10
	s_add_u32 s6, s6, 1
	s_addc_u32 s7, s7, 0
	v_lshrrev_b32_e32 v0, 6, v43
	v_lshlrev_b32_e32 v55, 5, v40
	s_and_b64 s[2:3], s[2:3], exec
	v_lshl_or_b32 v1, v0, 3, v55
	v_lshlrev_b32_e32 v3, 3, v40
	s_mov_b32 s30, s17
	s_mov_b32 s17, 0
	s_cselect_b32 s62, s20, s6
	v_readfirstlane_b32 s85, v1
	v_lshl_or_b32 v1, v0, 1, v3
	v_bfe_u32 v13, v43, 4, 2
	s_mov_b64 s[80:81], s[10:11]
	s_mov_b32 s31, s17
	s_mov_b64 s[18:19], -1
	s_cselect_b32 s63, s21, s7
	v_readfirstlane_b32 s84, v1
	s_cmp_gt_i32 s62, 1
	v_readfirstlane_b32 s33, v0
	v_lshrrev_b32_e32 v46, 3, v43
	v_lshrrev_b32_e32 v44, 5, v43
	v_lshrrev_b32_e32 v42, 4, v43
	v_and_b32_e32 v1, 15, v43
	v_lshlrev_b32_e32 v41, 6, v13
	v_lshlrev_b32_e32 v45, 2, v43
	s_cbranch_scc1 .LBB0_2
	s_ashr_i32 s2, s63, 31
	s_mov_b32 s3, s2
	s_xor_b64 s[6:7], s[2:3], s[62:63]
	s_lshr_b32 s18, s7, 31
	s_add_u32 s6, s6, s18
	s_addc_u32 s7, s7, 0
	s_lshr_b64 s[6:7], s[6:7], 1
	s_xor_b64 s[2:3], s[6:7], s[2:3]
	s_lshl_b64 s[66:67], s[2:3], 1
	s_ashr_i32 s2, s53, 31
	s_mov_b32 s3, s2
	s_xor_b64 s[6:7], s[52:53], s[2:3]
	s_lshr_b32 s18, s7, 31
	s_add_u32 s6, s6, s18
	s_addc_u32 s7, s7, 0
	s_ashr_i64 s[6:7], s[6:7], 1
	v_or_b32_e32 v0, v46, v55
	s_lshl_b64 s[34:35], s[16:17], 7
	s_xor_b64 s[58:59], s[6:7], s[2:3]
	v_or_b32_e32 v18, s34, v0
	v_mad_u64_u32 v[14:15], s[2:3], s58, v18, 0
	s_and_b32 s2, s58, 0x3fff
	v_or_b32_e32 v20, 64, v18
	s_lshr_b32 s19, s16, 25
	s_bitset1_b32 s2, 14
	v_mad_u64_u32 v[16:17], s[6:7], s58, v20, 0
	s_mul_i32 s19, s54, s19
	s_mul_hi_u32 s20, s54, s34
	s_lshl_b32 s18, s2, 16
	s_and_b32 s2, s41, 0xffff
	s_and_b32 s6, s37, 0xffff
	s_add_i32 s19, s20, s19
	s_mul_i32 s20, s55, s34
	s_or_b32 s3, s2, s18
	s_or_b32 s7, s6, s18
	v_or_b32_e32 v0, v44, v3
	s_lshl_b64 s[60:61], s[30:31], 5
	s_lshl_b32 s18, s33, 5
	s_add_i32 s65, s19, s20
	s_and_b32 s19, s54, 0x3fff
	v_or_b32_e32 v22, s60, v0
	v_or_b32_e32 v0, s18, v1
	s_or_b32 s73, s19, 0x4000
	s_lshr_b32 s19, s30, 27
	v_or_b32_e32 v26, s34, v0
	v_lshlrev_b32_e32 v0, 6, v42
	s_mul_i32 s19, s56, s19
	s_mul_hi_u32 s20, s56, s60
	v_sub_u32_e32 v17, v45, v0
	s_add_i32 s19, s20, s19
	s_mul_i32 s20, s57, s60
	s_mov_b32 s43, 0x27000
	s_mov_b32 s42, 0x7ffffffe
	v_add_u32_e32 v15, v17, v41
	v_lshl_or_b32 v12, v40, 4, v1
	s_add_i32 s69, s19, s20
	s_and_b32 s19, s56, 0x3fff
	v_mov_b32_e32 v19, s35
	s_mov_b32 s2, s40
	v_mov_b32_e32 v21, s35
	s_mov_b32 s6, s36
	v_mov_b32_e32 v23, s61
	v_or_b32_e32 v24, 16, v22
	v_mov_b32_e32 v25, s61
	v_mov_b32_e32 v27, s35
	s_mul_i32 s64, s54, s34
	v_or_b32_e32 v54, 2, v15
	v_or_b32_e32 v28, 16, v26
	v_mov_b32_e32 v29, s35
	v_or_b32_e32 v30, s60, v12
	v_mov_b32_e32 v31, s61
	s_mul_i32 s68, s56, s60
	s_or_b32 s89, s19, 0x4000
	v_mov_b32_e32 v32, s18
	s_mov_b64 s[38:39], s[42:43]
	s_mov_b64 s[18:19], 0
	s_branch .LBB0_3
.LBB0_2:
.LBB0_3:
	v_bfe_u32 v47, v43, 2, 3
	s_or_b32 s10, s85, 64
	s_or_b32 s86, s84, 16
	s_mov_b32 s20, 0
	s_andn2_b64 vcc, exec, s[18:19]
	v_lshrrev_b32_e32 v49, 2, v43
	v_lshlrev_b32_e32 v52, 7, v40
	v_and_b32_e32 v50, 31, v43
	v_lshlrev_b32_e32 v51, 2, v47
	v_lshlrev_b32_e32 v48, 3, v43
	v_lshlrev_b32_e32 v53, 1, v43
	s_cbranch_vccnz .LBB0_12
	s_ashr_i32 s2, s53, 31
	s_mov_b32 s3, s2
	s_xor_b64 s[6:7], s[52:53], s[2:3]
	s_lshr_b32 s18, s7, 31
	s_add_u32 s6, s6, s18
	s_addc_u32 s7, s7, 0
	s_ashr_i64 s[6:7], s[6:7], 1
	v_or_b32_e32 v0, v46, v55
	s_lshl_b64 s[34:35], s[16:17], 7
	s_xor_b64 s[58:59], s[6:7], s[2:3]
	v_or_b32_e32 v18, s34, v0
	v_mad_u64_u32 v[14:15], s[2:3], s58, v18, 0
	s_and_b32 s2, s58, 0x3fff
	v_bitop3_b32 v0, v46, 7, v43 bitop3:0x48
	s_bitset1_b32 s2, 14
	v_mov_b32_e32 v19, s35
	s_lshl_b32 s17, s2, 16
	s_and_b32 s2, s41, 0xffff
	v_lshlrev_b32_e32 v4, 5, v0
	v_mov_b32_e32 v5, 0
	v_lshlrev_b32_e32 v66, 4, v0
	s_or_b32 s41, s2, s17
	v_cmp_gt_i64_e64 s[2:3], s[12:13], v[18:19]
	v_cmp_gt_i64_e32 vcc, s[52:53], v[4:5]
	v_add_u32_e32 v0, v66, v14
	v_bfrev_b32_e32 v9, -2
	s_and_b64 s[6:7], s[2:3], vcc
	v_or_b32_e32 v20, 64, v18
	v_cndmask_b32_e64 v0, v9, v0, s[6:7]
	s_lshl_b32 s88, s85, 7
	v_mov_b32_e32 v21, s35
	v_mad_u64_u32 v[16:17], s[6:7], s58, v20, 0
	s_mov_b32 s43, 0x27000
	s_mov_b32 s42, 0x7ffffffe
	s_mov_b32 m0, s88
	v_cmp_gt_i64_e64 s[6:7], s[12:13], v[20:21]
	buffer_load_dwordx4 v0, s[40:43], 0 offen lds
	s_and_b64 vcc, s[6:7], vcc
	s_add_i32 m0, s88, 0x2000
	s_ashr_i32 s70, s59, 31
	s_add_u32 s18, s58, s70
	s_mov_b32 s71, s70
	s_addc_u32 s19, s59, s70
	s_xor_b64 s[22:23], s[18:19], s[70:71]
	v_cvt_f32_u32_e32 v12, s22
	v_cvt_f32_u32_e32 v15, s23
	s_sub_u32 s18, 0, s22
	s_subb_u32 s19, 0, s23
	v_bitop3_b32 v11, v49, v44, 7 bitop3:0x6c
	v_fmamk_f32 v12, v15, 0x4f800000, v12
	v_rcp_f32_e32 v12, v12
	v_sub_u32_e32 v6, v50, v51
	v_lshl_add_u32 v6, v11, 2, v6
	v_ashrrev_i32_e32 v7, 31, v6
	v_mul_f32_e32 v12, 0x5f7ffffc, v12
	v_mul_f32_e32 v15, 0x2f800000, v12
	v_trunc_f32_e32 v15, v15
	v_fmamk_f32 v12, v15, 0xcf800000, v12
	v_cvt_u32_f32_e32 v12, v12
	v_cvt_u32_f32_e32 v15, v15
	v_xor_b32_e32 v6, v7, v6
	v_ashrrev_i32_e32 v8, 31, v6
	v_readfirstlane_b32 s21, v12
	v_readfirstlane_b32 s26, v15
	s_mul_hi_u32 s25, s18, s21
	s_mul_i32 s27, s18, s26
	s_mul_i32 s24, s19, s21
	s_add_i32 s25, s25, s27
	s_add_i32 s25, s25, s24
	s_mul_i32 s28, s18, s21
	s_mul_i32 s27, s21, s25
	s_mul_hi_u32 s29, s21, s28
	s_mul_hi_u32 s24, s21, s25
	s_add_u32 s27, s29, s27
	s_addc_u32 s24, 0, s24
	s_mul_hi_u32 s38, s26, s28
	s_mul_i32 s28, s26, s28
	s_add_u32 s27, s27, s28
	s_mul_hi_u32 s29, s26, s25
	s_addc_u32 s24, s24, s38
	s_addc_u32 s27, s29, 0
	s_mul_i32 s25, s26, s25
	s_add_u32 s24, s24, s25
	s_addc_u32 s25, 0, s27
	s_add_u32 s21, s21, s24
	v_lshrrev_b32_e32 v8, 29, v8
	s_addc_u32 s24, s26, s25
	v_add_u32_e32 v6, v6, v8
	s_mul_i32 s25, s18, s24
	s_mul_hi_u32 s26, s18, s21
	v_lshrrev_b32_e32 v6, 3, v6
	s_add_i32 s25, s26, s25
	s_mul_i32 s19, s19, s21
	v_add_u32_e32 v0, v66, v16
	v_xor_b32_e32 v6, v6, v7
	s_add_i32 s25, s25, s19
	s_mul_i32 s18, s18, s21
	v_cndmask_b32_e32 v0, v9, v0, vcc
	v_lshlrev_b32_e32 v8, 9, v6
	v_lshrrev_b32_e32 v6, 1, v11
	s_mul_hi_u32 s26, s24, s18
	s_mul_i32 s27, s24, s18
	s_mul_i32 s29, s21, s25
	s_mul_hi_u32 s18, s21, s18
	buffer_load_dwordx4 v0, s[40:43], 0 offen lds
	v_lshlrev_b32_e32 v0, 8, v11
	v_lshlrev_b32_e32 v2, 4, v44
	v_mul_i32_i24_e32 v10, 0xfffffe00, v6
	v_mul_i32_i24_e32 v32, -16, v49
	s_mul_hi_u32 s28, s21, s25
	s_add_u32 s18, s18, s29
	v_add3_u32 v0, v52, v45, v0
	v_add3_u32 v6, v32, v2, v10
	s_addc_u32 s28, 0, s28
	v_add3_u32 v6, v6, v0, v8
	s_add_u32 s18, s18, s27
	v_not_b32_e32 v17, v6
	s_mul_hi_u32 s19, s24, s25
	s_addc_u32 s18, s28, s26
	v_ashrrev_i32_e32 v7, 31, v6
	v_ashrrev_i32_e32 v22, 31, v17
	v_cmp_gt_i32_e32 vcc, 0, v6
	s_addc_u32 s19, s19, 0
	s_mul_i32 s25, s24, s25
	v_cndmask_b32_e32 v23, v7, v22, vcc
	s_add_u32 s18, s18, s25
	v_cndmask_b32_e32 v22, v6, v17, vcc
	s_addc_u32 s19, 0, s19
	v_mov_b32_e32 v24, v23
	v_mov_b32_e32 v25, v23
	s_add_u32 s21, s21, s18
	v_lshl_add_u64 v[24:25], v[22:23], 0, v[24:25]
	s_addc_u32 s24, s24, s19
	v_xor_b32_e32 v15, v24, v23
	v_xor_b32_e32 v12, v25, v23
	v_mad_u64_u32 v[24:25], s[18:19], v15, s24, 0
	v_mul_hi_u32 v26, v15, s21
	v_mov_b32_e32 v27, v5
	v_lshl_add_u64 v[24:25], v[26:27], 0, v[24:25]
	v_mad_u64_u32 v[28:29], s[18:19], v12, s21, 0
	v_add_co_u32_e32 v17, vcc, v24, v28
	v_mad_u64_u32 v[26:27], s[18:19], v12, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v25, v29, vcc
	v_mov_b32_e32 v25, s20
	s_nop 0
	v_addc_co_u32_e32 v27, vcc, 0, v27, vcc
	v_lshl_add_u64 v[24:25], v[24:25], 0, v[26:27]
	v_mul_lo_u32 v17, s23, v24
	v_mul_lo_u32 v22, s22, v25
	v_mad_u64_u32 v[26:27], s[18:19], s22, v24, 0
	v_add3_u32 v17, v27, v22, v17
	v_sub_u32_e32 v22, v12, v17
	v_mov_b32_e32 v28, s23
	v_sub_co_u32_e32 v15, vcc, v15, v26
	s_lshl_b64 s[60:61], s[30:31], 5
	s_nop 0
	v_subb_co_u32_e64 v22, s[18:19], v22, v28, vcc
	v_subrev_co_u32_e64 v25, s[18:19], s22, v15
	v_subb_co_u32_e32 v12, vcc, v12, v17, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v22, s[18:19], 0, v22, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s23, v22
	v_cmp_le_u32_e32 vcc, s23, v12
	v_or_b32_e32 v3, v44, v3
	v_cndmask_b32_e64 v26, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s22, v25
	v_cndmask_b32_e64 v17, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s22, v15
	v_cndmask_b32_e64 v25, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s23, v22
	v_cndmask_b32_e64 v15, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s23, v12
	v_cndmask_b32_e64 v22, v26, v25, s[18:19]
	v_add_u32_e32 v25, 2, v24
	v_add_u32_e32 v26, 1, v24
	v_cmp_ne_u32_e64 s[18:19], 0, v22
	v_cndmask_b32_e32 v12, v17, v15, vcc
	v_cmp_ne_u32_e32 vcc, 0, v12
	v_cndmask_b32_e64 v22, v26, v25, s[18:19]
	v_xor_b32_e32 v15, s70, v23
	v_cndmask_b32_e32 v12, v24, v22, vcc
	v_mov_b32_e32 v22, v7
	v_mov_b32_e32 v23, v7
	v_xor_b32_e32 v12, v12, v15
	v_lshl_add_u64 v[22:23], v[6:7], 0, v[22:23]
	v_sub_u32_e32 v12, v12, v15
	v_xor_b32_e32 v15, v22, v7
	v_xor_b32_e32 v6, v23, v7
	v_mad_u64_u32 v[22:23], s[18:19], v15, s24, 0
	v_mul_hi_u32 v24, v15, s21
	v_mov_b32_e32 v25, v5
	v_lshl_add_u64 v[22:23], v[24:25], 0, v[22:23]
	v_mad_u64_u32 v[26:27], s[18:19], v6, s21, 0
	v_add_co_u32_e32 v17, vcc, v22, v26
	v_mad_u64_u32 v[24:25], s[18:19], v6, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v22, vcc, v23, v27, vcc
	v_mov_b32_e32 v23, s20
	s_nop 0
	v_addc_co_u32_e32 v25, vcc, 0, v25, vcc
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[24:25]
	v_mul_lo_u32 v17, s23, v22
	v_mul_lo_u32 v24, s22, v23
	v_mad_u64_u32 v[22:23], s[18:19], s22, v22, 0
	v_add3_u32 v17, v23, v24, v17
	v_sub_u32_e32 v23, v6, v17
	v_sub_co_u32_e32 v15, vcc, v15, v22
	v_xad_u32 v12, v12, v7, s60
	s_nop 0
	v_subb_co_u32_e64 v22, s[18:19], v23, v28, vcc
	v_subrev_co_u32_e64 v23, s[18:19], s22, v15
	v_subb_co_u32_e32 v6, vcc, v6, v17, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v24, s[20:21], 0, v22, s[18:19]
	v_cmp_le_u32_e64 s[20:21], s23, v24
	v_subb_co_u32_e64 v22, s[18:19], v22, v28, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v25, 0, -1, s[20:21]
	v_cmp_le_u32_e64 s[20:21], s22, v23
	v_cmp_le_u32_e32 vcc, s23, v6
	s_lshl_b32 s91, s84, 7
	v_cndmask_b32_e64 v26, 0, -1, s[20:21]
	v_cmp_eq_u32_e64 s[20:21], s23, v24
	v_cndmask_b32_e64 v17, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s22, v15
	v_cndmask_b32_e64 v25, v25, v26, s[20:21]
	v_subrev_co_u32_e64 v26, s[18:19], s22, v23
	s_mov_b32 s38, s42
	s_nop 0
	v_subbrev_co_u32_e64 v22, s[18:19], 0, v22, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v25
	v_mov_b32_e32 v25, s61
	s_mov_b32 s39, s43
	v_cndmask_b32_e64 v22, v24, v22, s[18:19]
	v_cndmask_b32_e64 v24, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s23, v6
	s_add_i32 m0, s91, 0xc000
	s_lshl_b64 s[72:73], s[58:59], 4
	v_cndmask_b32_e32 v17, v17, v24, vcc
	v_cmp_ne_u32_e32 vcc, 0, v17
	v_cndmask_b32_e64 v17, v23, v26, s[18:19]
	s_and_b32 s18, s37, 0xffff
	v_cndmask_b32_e32 v15, v15, v17, vcc
	v_cndmask_b32_e32 v6, v6, v22, vcc
	v_xor_b32_e32 v15, v15, v7
	v_xor_b32_e32 v6, v6, v7
	v_sub_co_u32_e32 v15, vcc, v15, v7
	v_or_b32_e32 v22, s60, v3
	s_nop 0
	v_subb_co_u32_e32 v6, vcc, v6, v7, vcc
	v_ashrrev_i32_e32 v6, 31, v6
	v_and_b32_e32 v6, s58, v6
	v_mul_lo_u32 v7, v12, s58
	v_add3_u32 v6, v6, v15, v7
	v_lshlrev_b32_e32 v7, 8, v44
	v_sub_u32_e32 v3, v11, v47
	v_sub_u32_e32 v7, v48, v7
	v_lshl_add_u32 v36, v3, 5, v7
	v_mov_b32_e32 v23, s61
	v_ashrrev_i32_e32 v37, 31, v36
	s_or_b32 s37, s18, s17
	v_cmp_gt_i64_e64 s[18:19], s[14:15], v[22:23]
	v_cmp_gt_i64_e32 vcc, s[52:53], v[36:37]
	s_and_b64 s[20:21], s[18:19], vcc
	v_or_b32_e32 v24, 16, v22
	v_cndmask_b32_e64 v3, v9, v6, s[20:21]
	v_cmp_gt_i64_e64 s[20:21], s[14:15], v[24:25]
	buffer_load_dword v3, s[36:39], 0 offen lds
	v_add_u32_e32 v3, s72, v6
	s_and_b64 vcc, s[20:21], vcc
	s_lshl_b32 s92, s86, 7
	v_lshlrev_b32_e32 v34, 6, v42
	v_cndmask_b32_e32 v3, v9, v3, vcc
	s_add_i32 m0, s92, 0xc000
	s_lshl_b32 s90, s33, 5
	v_sub_u32_e32 v17, v45, v34
	buffer_load_dword v3, s[36:39], 0 offen lds
	v_or_b32_e32 v3, s90, v1
	v_add_u32_e32 v15, v17, v41
	s_mul_i32 s17, s54, s90
	v_lshlrev_b32_e32 v12, 5, v42
	v_lshlrev_b32_e32 v6, 5, v13
	v_or_b32_e32 v26, s34, v3
	v_add_u32_e32 v3, s17, v15
	s_lshr_b32 s17, s16, 25
	v_sub_u32_e32 v12, v53, v12
	v_mov_b32_e32 v7, v5
	v_mov_b32_e32 v27, s35
	s_mul_i32 s17, s54, s17
	s_mul_hi_u32 s26, s54, s34
	v_add_u32_e32 v12, v12, v6
	v_cmp_gt_i64_e32 vcc, s[52:53], v[6:7]
	v_cmp_gt_i64_e64 s[22:23], s[12:13], v[26:27]
	s_add_i32 s17, s26, s17
	s_mul_i32 s26, s55, s34
	v_ashrrev_i32_e32 v30, 31, v12
	s_and_b64 s[24:25], s[22:23], vcc
	s_add_i32 s65, s17, s26
	s_mul_i32 s64, s54, s34
	v_xor_b32_e32 v12, v30, v12
	s_add_u32 s44, s4, s64
	v_ashrrev_i32_e32 v31, 31, v12
	s_addc_u32 s17, s5, s65
	s_and_b32 s26, s54, 0x3fff
	v_lshrrev_b32_e32 v31, 25, v31
	s_or_b32 s73, s26, 0x4000
	v_add_u32_e32 v12, v12, v31
	s_and_b32 s17, s17, 0xffff
	s_lshl_b32 s26, s73, 16
	v_or_b32_e32 v28, 0x80, v6
	v_mov_b32_e32 v29, v5
	v_ashrrev_i32_e32 v12, 7, v12
	s_or_b32 s45, s17, s26
	v_cmp_gt_i64_e64 s[26:27], s[52:53], v[28:29]
	v_xor_b32_e32 v12, v12, v30
	v_cndmask_b32_e64 v11, v9, v3, s[24:25]
	v_or_b32_e32 v3, 2, v3
	s_and_b64 s[24:25], s[22:23], s[26:27]
	v_or_b32_e32 v28, 16, v26
	v_mov_b32_e32 v29, s35
	v_add_u32_e32 v30, s90, v12
	v_lshlrev_b32_e32 v35, 8, v12
	v_cndmask_b32_e64 v3, v9, v3, s[24:25]
	v_cmp_gt_i64_e64 s[24:25], s[12:13], v[28:29]
	v_sub_u32_e32 v12, v15, v35
	v_mul_lo_u32 v38, s54, v30
	v_add3_u32 v12, v38, v12, 1
	s_and_b64 s[28:29], s[24:25], vcc
	v_cndmask_b32_e64 v12, v9, v12, s[28:29]
	v_or_b32_e32 v30, 3, v15
	v_xor_b32_e32 v31, -4, v15
	v_cmp_gt_i32_e64 s[28:29], 0, v15
	v_ashrrev_i32_e32 v33, 31, v15
	s_mov_b32 s46, s42
	v_cndmask_b32_e64 v30, v30, v31, s[28:29]
	v_ashrrev_i32_e32 v31, 31, v30
	v_add_u32_sdwa v30, v30, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_ashrrev_i32_e32 v30, 8, v30
	v_xor_b32_e32 v30, v30, v33
	v_add_u32_e32 v31, s90, v30
	v_lshlrev_b32_e32 v39, 8, v30
	v_sub_u32_e32 v30, v15, v39
	v_mul_lo_u32 v69, s54, v31
	s_mov_b32 s47, s43
	v_add3_u32 v30, v69, v30, 3
	s_and_b64 s[28:29], s[24:25], s[26:27]
	v_cndmask_b32_e64 v30, v9, v30, s[28:29]
	buffer_load_ubyte v59, v11, s[44:47], 0 offen
	buffer_load_ubyte v57, v3, s[44:47], 0 offen
	buffer_load_ubyte v58, v12, s[44:47], 0 offen
	buffer_load_ubyte v56, v30, s[44:47], 0 offen
	v_lshl_or_b32 v12, v40, 4, v1
	v_bitop3_b32 v1, v33, v15, v40 bitop3:0x1e
	v_ashrrev_i32_e32 v3, 31, v1
	v_add_u32_sdwa v1, v1, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_or_b32_e32 v30, s60, v12
	v_mov_b32_e32 v31, s61
	v_ashrrev_i32_e32 v1, 8, v1
	s_movk_i32 s17, 0xfd
	v_cmp_gt_i64_e64 s[28:29], s[14:15], v[30:31]
	v_xor_b32_e32 v67, v1, v33
	v_bitop3_b32 v60, v15, s17, v40 bitop3:0xc8
	v_mad_u64_u32 v[60:61], s[38:39], v67, s56, v[60:61]
	s_and_b64 vcc, s[28:29], vcc
	v_or_b32_e32 v1, v15, v40
	s_lshr_b32 s17, s30, 27
	v_cndmask_b32_e32 v3, v9, v60, vcc
	v_or_b32_e32 v11, 2, v1
	v_bitop3_b32 v33, v15, -3, v40 bitop3:0x36
	v_cmp_gt_i32_e32 vcc, -2, v1
	s_mul_i32 s17, s56, s17
	s_mul_hi_u32 s30, s56, s60
	v_cndmask_b32_e32 v1, v11, v33, vcc
	s_add_i32 s17, s30, s17
	s_mul_i32 s30, s57, s60
	v_ashrrev_i32_e32 v11, 31, v1
	s_add_i32 s69, s17, s30
	s_mul_i32 s68, s56, s60
	v_add_u32_sdwa v1, v1, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	s_add_u32 s48, s8, s68
	v_ashrrev_i32_e32 v1, 8, v1
	v_cndmask_b32_e64 v11, 0, -1, vcc
	v_or_b32_e32 v54, 2, v15
	s_addc_u32 s17, s9, s69
	s_and_b32 s30, s56, 0x3fff
	v_xor_b32_e32 v70, v1, v11
	s_or_b32 s89, s30, 0x4000
	v_or_b32_e32 v1, v54, v40
	v_lshlrev_b32_e32 v68, 8, v70
	s_and_b32 s17, s17, 0xffff
	s_lshl_b32 s30, s89, 16
	v_sub_u32_e32 v60, v1, v68
	s_or_b32 s49, s17, s30
	s_mov_b32 s50, s42
	s_mov_b32 s51, s43
	v_mad_u64_u32 v[60:61], s[30:31], v70, s56, v[60:61]
	s_and_b64 vcc, s[28:29], s[26:27]
	v_cndmask_b32_e32 v1, v9, v60, vcc
	buffer_load_ubyte v65, v3, s[48:51], 0 offen
	buffer_load_ubyte v64, v1, s[48:51], 0 offen
	v_cmp_eq_u32_e64 s[26:27], 0, v40
	s_mov_b32 s55, 0xc000
	s_and_b64 vcc, exec, s[26:27]
	s_barrier
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	s_ashr_i32 s30, s63, 31
	s_mov_b32 s31, s30
	s_xor_b64 s[38:39], s[30:31], s[62:63]
	s_lshr_b32 s17, s39, 31
	s_add_u32 s38, s38, s17
	s_addc_u32 s39, s39, 0
	v_and_b32_e32 v1, 7, v43
	s_ashr_i64 s[38:39], s[38:39], 1
	v_lshlrev_b32_e32 v9, 7, v43
	v_lshlrev_b32_e32 v11, 11, v42
	s_xor_b64 s[30:31], s[38:39], s[30:31]
	v_bitop3_b32 v3, v13, v43, 7 bitop3:0x78
	v_sub_u32_e32 v9, v9, v11
	v_bitop3_b32 v1, v13, v1, 4 bitop3:0x36
	s_lshl_b64 s[66:67], s[30:31], 1
	v_lshl_add_u32 v11, s33, 12, v9
	v_lshlrev_b32_e32 v3, 4, v3
	v_lshl_add_u32 v9, v40, 11, v9
	v_lshlrev_b32_e32 v1, 4, v1
	v_cmp_lt_i64_e64 s[30:31], s[30:31], 1
	s_mov_b32 s57, 0
	v_or_b32_e32 v60, v11, v3
	v_or_b32_e32 v63, v9, v3
	v_or_b32_e32 v61, v1, v11
	v_or_b32_e32 v62, v1, v9
	s_and_b64 vcc, exec, s[30:31]
	s_cbranch_vccnz .LBB0_17
	v_lshlrev_b16_e32 v71, 2, v43
	v_or_b32_e32 v71, v40, v71
	s_lshl_b32 s93, s10, 7
	v_mov_b32_e32 v3, v5
	v_ashrrev_i32_e32 v33, 31, v32
	v_mul_lo_u32 v67, s56, v67
	v_and_b32_e32 v71, 0xfd, v71
	v_ashrrev_i32_e32 v11, 31, v10
	s_add_u32 s74, s66, -1
	v_add3_u32 v67, v67, s56, v71
	v_mul_lo_u32 v70, s56, v70
	v_add_u32_e32 v71, s56, v40
	v_add3_u32 v38, v38, s54, v41
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[32:33]
	v_ashrrev_i32_e32 v9, 31, v8
	s_addc_u32 s75, s67, -1
	v_add3_u32 v70, v71, v70, v41
	v_sub_u32_e32 v38, v38, v34
	s_add_i32 s17, s90, 1
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[10:11]
	v_mov_b32_e32 v1, v5
	v_sub_u32_e32 v70, v70, v34
	v_sub_u32_e32 v35, v38, v35
	s_mul_i32 s17, s54, s17
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[8:9]
	v_sub_u32_e32 v68, v70, v68
	v_add3_u32 v69, v69, s54, v41
	v_add_u32_e32 v70, 1, v35
	v_add_u32_e32 v35, s17, v41
	v_lshl_add_u64 v[0:1], v[2:3], 0, v[0:1]
	s_movk_i32 s17, 0xf7ff
	s_mov_b64 s[76:77], 0x800
	s_lshl_b32 s16, s16, 7
	v_sub_u32_e32 v69, v69, v34
	v_sub_u32_e32 v71, v35, v34
	v_sub_co_u32_e32 v32, vcc, s17, v0
	v_lshl_add_u64 v[34:35], v[0:1], 0, s[76:77]
	v_or_b32_e32 v0, s16, v55
	v_add_u32_e32 v0, v0, v46
	v_subb_co_u32_e32 v33, vcc, -1, v1, vcc
	v_add_u32_e32 v1, 64, v0
	v_mul_lo_u32 v1, v1, s58
	s_movk_i32 s16, 0x80
	v_mul_lo_u32 v0, v0, s58
	v_writelane_b32 v92, s80, 0
	v_sub_u32_e32 v39, v69, v39
	s_mov_b64 s[78:79], 0x100
	v_add3_u32 v55, v1, v66, s16
	v_add3_u32 v66, v0, v66, s16
	v_mov_b32_e32 v0, 0
	v_writelane_b32 v92, s81, 1
	v_add_u32_e32 v68, 2, v68
	v_add_u32_e32 v69, 3, v39
	v_lshl_add_u64 v[36:37], v[36:37], 0, s[78:79]
	v_or_b32_e32 v38, 0x100, v4
	v_mov_b32_e32 v39, v5
	s_mov_b64 s[80:81], 0
	s_mov_b32 s94, 0
	s_movk_i32 s17, 0x4000
	s_mov_b32 s16, 0xd000
	v_bfrev_b32_e32 v72, -2
	s_mov_b32 s71, s70
	s_mov_b32 s38, s42
	s_mov_b32 s39, s43
	s_mov_b32 s46, s42
	s_mov_b32 s47, s43
	s_mov_b32 s50, s42
	s_mov_b32 s51, s43
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v8, v0
	v_mov_b32_e32 v9, v0
	v_mov_b32_e32 v10, v0
	v_mov_b32_e32 v11, v0
.LBB0_8:
	s_mov_b32 s95, s57
	s_mov_b32 s57, s17
	s_mov_b32 s96, s55
	s_mov_b32 s55, s16
	v_lshl_add_u64 v[74:75], v[38:39], 0, s[80:81]
	v_cmp_gt_i64_e32 vcc, s[52:53], v[74:75]
	s_and_b64 s[16:17], s[2:3], vcc
	v_cndmask_b32_e64 v4, v72, v66, s[16:17]
	s_add_i32 m0, s57, s88
	s_waitcnt vmcnt(0)
	s_barrier
	s_barrier
	buffer_load_dwordx4 v4, s[40:43], 0 offen lds
	s_and_b64 vcc, s[6:7], vcc
	s_add_i32 m0, s57, s93
	s_add_u32 s16, s58, s70
	s_addc_u32 s17, s59, s70
	s_xor_b64 s[82:83], s[16:17], s[70:71]
	v_cvt_f32_u32_e32 v4, s82
	v_cvt_f32_u32_e32 v73, s83
	s_sub_u32 s16, 0, s82
	s_subb_u32 s17, 0, s83
	v_cndmask_b32_e32 v74, v72, v55, vcc
	v_fmac_f32_e32 v4, 0x4f800000, v73
	v_rcp_f32_e32 v4, v4
	v_cmp_gt_i32_e32 vcc, 0, v35
	buffer_load_dwordx4 v74, s[40:43], 0 offen lds
	v_mov_b32_e32 v82, s83
	v_mul_f32_e32 v4, 0x5f7ffffc, v4
	v_mul_f32_e32 v73, 0x2f800000, v4
	v_trunc_f32_e32 v73, v73
	v_fmac_f32_e32 v4, 0xcf800000, v73
	v_cvt_u32_f32_e32 v4, v4
	v_cvt_u32_f32_e32 v73, v73
	v_cndmask_b32_e32 v75, v35, v33, vcc
	v_cndmask_b32_e32 v74, v34, v32, vcc
	v_readfirstlane_b32 s30, v4
	v_readfirstlane_b32 s98, v73
	s_mul_hi_u32 s97, s16, s30
	s_mul_i32 s99, s16, s98
	s_mul_i32 s31, s17, s30
	s_add_i32 s97, s97, s99
	s_add_i32 s97, s97, s31
	s_mul_i32 vcc_lo, s16, s30
	s_mul_i32 s99, s30, s97
	s_mul_hi_u32 vcc_hi, s30, vcc_lo
	s_mul_hi_u32 s31, s30, s97
	s_add_u32 s99, vcc_hi, s99
	s_addc_u32 s31, 0, s31
	s_mul_i32 s11, s98, vcc_lo
	s_mul_hi_u32 s87, s98, vcc_lo
	s_add_u32 s11, s99, s11
	s_mul_hi_u32 vcc_hi, s98, s97
	s_addc_u32 s11, s31, s87
	s_addc_u32 s31, vcc_hi, 0
	s_mul_i32 s87, s98, s97
	s_add_u32 s11, s11, s87
	s_addc_u32 s31, 0, s31
	s_add_u32 s11, s30, s11
	s_addc_u32 s30, s98, s31
	s_mul_i32 s31, s16, s30
	s_mul_hi_u32 s87, s16, s11
	s_add_i32 s31, s87, s31
	s_mul_i32 s17, s17, s11
	s_add_i32 s31, s31, s17
	s_mul_i32 s16, s16, s11
	s_mul_hi_u32 s87, s30, s16
	s_mul_i32 s97, s30, s16
	s_mul_i32 s99, s11, s31
	s_mul_hi_u32 s16, s11, s16
	s_mul_hi_u32 s98, s11, s31
	s_add_u32 s16, s16, s99
	s_addc_u32 s98, 0, s98
	s_add_u32 s16, s16, s97
	s_mul_hi_u32 s17, s30, s31
	s_addc_u32 s16, s98, s87
	s_addc_u32 s17, s17, 0
	s_mul_i32 s31, s30, s31
	s_add_u32 s16, s16, s31
	v_ashrrev_i32_e32 v76, 31, v75
	s_addc_u32 s17, 0, s17
	v_mov_b32_e32 v77, v76
	s_add_u32 s11, s11, s16
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[76:77]
	s_addc_u32 s30, s30, s17
	v_xor_b32_e32 v77, v74, v76
	v_xor_b32_e32 v73, v75, v76
	v_mad_u64_u32 v[74:75], s[16:17], v77, s30, 0
	v_mul_hi_u32 v4, v77, s11
	v_lshl_add_u64 v[74:75], v[4:5], 0, v[74:75]
	v_mad_u64_u32 v[80:81], s[16:17], v73, s11, 0
	v_add_co_u32_e32 v4, vcc, v74, v80
	v_mad_u64_u32 v[78:79], s[16:17], v73, s30, 0
	s_nop 0
	v_addc_co_u32_e32 v74, vcc, v75, v81, vcc
	v_mov_b32_e32 v75, s94
	s_nop 0
	v_addc_co_u32_e32 v79, vcc, 0, v79, vcc
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[78:79]
	v_mul_lo_u32 v4, s83, v74
	v_mul_lo_u32 v75, s82, v75
	v_mad_u64_u32 v[78:79], s[16:17], s82, v74, 0
	v_add3_u32 v4, v79, v75, v4
	v_sub_u32_e32 v75, v73, v4
	v_sub_co_u32_e32 v77, vcc, v77, v78
	s_add_i32 m0, s55, s91
	s_nop 0
	v_subb_co_u32_e64 v75, s[16:17], v75, v82, vcc
	v_subrev_co_u32_e64 v78, s[16:17], s82, v77
	v_subb_co_u32_e32 v4, vcc, v73, v4, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v75, s[16:17], 0, v75, s[16:17]
	v_cmp_le_u32_e64 s[16:17], s83, v75
	v_cmp_le_u32_e32 vcc, s83, v4
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[16:17]
	v_cmp_le_u32_e64 s[16:17], s82, v78
	v_cndmask_b32_e64 v73, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s82, v77
	v_cndmask_b32_e64 v78, 0, -1, s[16:17]
	v_cmp_eq_u32_e64 s[16:17], s83, v75
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s83, v4
	v_cndmask_b32_e64 v75, v79, v78, s[16:17]
	v_add_u32_e32 v78, 2, v74
	v_add_u32_e32 v79, 1, v74
	v_cmp_ne_u32_e64 s[16:17], 0, v75
	v_cndmask_b32_e32 v4, v73, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v4
	v_cndmask_b32_e64 v75, v79, v78, s[16:17]
	v_xor_b32_e32 v73, s70, v76
	v_cndmask_b32_e32 v4, v74, v75, vcc
	v_ashrrev_i32_e32 v74, 31, v35
	v_mov_b32_e32 v75, v74
	v_xor_b32_e32 v4, v4, v73
	v_lshl_add_u64 v[76:77], v[34:35], 0, v[74:75]
	v_sub_u32_e32 v4, v4, v73
	v_xor_b32_e32 v83, v76, v74
	v_xad_u32 v73, v4, v74, s60
	v_xor_b32_e32 v75, v77, v74
	v_mad_u64_u32 v[76:77], s[16:17], v83, s30, 0
	v_mul_hi_u32 v4, v83, s11
	v_lshl_add_u64 v[76:77], v[4:5], 0, v[76:77]
	v_mad_u64_u32 v[80:81], s[16:17], v75, s11, 0
	v_add_co_u32_e32 v4, vcc, v76, v80
	v_mad_u64_u32 v[78:79], s[16:17], v75, s30, 0
	s_nop 0
	v_addc_co_u32_e32 v76, vcc, v77, v81, vcc
	v_mov_b32_e32 v77, s94
	s_nop 0
	v_addc_co_u32_e32 v79, vcc, 0, v79, vcc
	v_lshl_add_u64 v[76:77], v[76:77], 0, v[78:79]
	v_mul_lo_u32 v4, s83, v76
	v_mul_lo_u32 v78, s82, v77
	v_mad_u64_u32 v[76:77], s[16:17], s82, v76, 0
	v_add3_u32 v4, v77, v78, v4
	v_sub_u32_e32 v77, v75, v4
	v_sub_co_u32_e32 v76, vcc, v83, v76
	v_mul_lo_u32 v73, v73, s58
	s_nop 0
	v_subb_co_u32_e64 v77, s[16:17], v77, v82, vcc
	v_subrev_co_u32_e64 v78, s[16:17], s82, v76
	v_subb_co_u32_e32 v4, vcc, v75, v4, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v79, s[30:31], 0, v77, s[16:17]
	v_cmp_le_u32_e64 s[30:31], s83, v79
	v_subb_co_u32_e64 v77, s[16:17], v77, v82, s[16:17]
	s_nop 0
	v_cndmask_b32_e64 v80, 0, -1, s[30:31]
	v_cmp_le_u32_e64 s[30:31], s82, v78
	v_cmp_le_u32_e32 vcc, s83, v4
	s_nop 0
	v_cndmask_b32_e64 v81, 0, -1, s[30:31]
	v_cmp_eq_u32_e64 s[30:31], s83, v79
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s82, v76
	v_cndmask_b32_e64 v80, v80, v81, s[30:31]
	v_subrev_co_u32_e64 v81, s[16:17], s82, v78
	s_nop 1
	v_subbrev_co_u32_e64 v77, s[16:17], 0, v77, s[16:17]
	v_cmp_ne_u32_e64 s[16:17], 0, v80
	s_nop 1
	v_cndmask_b32_e64 v77, v79, v77, s[16:17]
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s83, v4
	s_nop 1
	v_cndmask_b32_e32 v75, v75, v79, vcc
	v_cmp_ne_u32_e32 vcc, 0, v75
	v_cndmask_b32_e64 v75, v78, v81, s[16:17]
	s_nop 0
	v_cndmask_b32_e32 v75, v76, v75, vcc
	v_cndmask_b32_e32 v4, v4, v77, vcc
	v_xor_b32_e32 v75, v75, v74
	v_xor_b32_e32 v4, v4, v74
	v_sub_co_u32_e32 v75, vcc, v75, v74
	s_nop 1
	v_subb_co_u32_e32 v4, vcc, v4, v74, vcc
	v_ashrrev_i32_e32 v4, 31, v4
	v_and_b32_e32 v4, s58, v4
	v_add3_u32 v4, v4, v75, v73
	v_lshl_add_u64 v[74:75], v[36:37], 0, s[80:81]
	v_cmp_gt_i64_e32 vcc, s[52:53], v[74:75]
	s_and_b64 s[16:17], s[18:19], vcc
	v_cndmask_b32_e64 v73, v72, v4, s[16:17]
	v_add_u32_e32 v4, s72, v4
	s_and_b64 vcc, s[20:21], vcc
	buffer_load_dword v73, s[36:39], 0 offen lds
	v_cndmask_b32_e32 v4, v72, v4, vcc
	s_add_i32 m0, s55, s92
	s_nop 0
	buffer_load_dword v4, s[36:39], 0 offen lds
	v_lshl_add_u64 v[74:75], v[6:7], 0, s[80:81]
	v_lshl_add_u64 v[76:77], v[74:75], 0, s[78:79]
	v_cmp_gt_i64_e32 vcc, s[52:53], v[76:77]
	v_add_u32_e32 v4, v45, v71
	s_and_b64 s[16:17], s[22:23], vcc
	v_cndmask_b32_e64 v73, v72, v4, s[16:17]
	s_mov_b64 s[16:17], 0x180
	v_lshl_add_u64 v[74:75], v[74:75], 0, s[16:17]
	v_cmp_gt_i64_e64 s[16:17], s[52:53], v[74:75]
	v_add_u32_e32 v4, 2, v4
	s_and_b64 s[30:31], s[22:23], s[16:17]
	v_cndmask_b32_e64 v74, v72, v4, s[30:31]
	v_add_u32_e32 v4, v45, v70
	s_and_b64 s[30:31], s[24:25], vcc
	v_cndmask_b32_e64 v75, v72, v4, s[30:31]
	v_add_u32_e32 v4, v45, v69
	s_and_b64 s[30:31], s[24:25], s[16:17]
	v_cndmask_b32_e64 v76, v72, v4, s[30:31]
	s_and_b64 vcc, s[28:29], vcc
	buffer_load_ubyte v4, v73, s[44:47], 0 offen
	s_nop 0
	buffer_load_ubyte v73, v74, s[44:47], 0 offen
	s_nop 0
	buffer_load_ubyte v74, v75, s[44:47], 0 offen
	s_nop 0
	buffer_load_ubyte v75, v76, s[44:47], 0 offen
	v_cndmask_b32_e32 v76, v72, v67, vcc
	v_add_u32_e32 v77, v45, v68
	s_and_b64 vcc, s[28:29], s[16:17]
	v_cndmask_b32_e32 v77, v72, v77, vcc
	buffer_load_ubyte v76, v76, s[48:51], 0 offen
	s_nop 0
	buffer_load_ubyte v77, v77, s[48:51], 0 offen
	v_add_u32_e32 v78, s95, v60
	ds_read_b128 v[80:83], v78
	ds_read_b128 v[84:87], v78 offset:2048
	v_add_u32_e32 v78, s96, v63
	ds_read_b128 v[88:91], v78
	s_barrier
	s_setprio 1
	v_and_b32_e32 v59, 0xff, v59
	v_and_b32_e32 v65, 0xff, v65
	v_and_b32_e32 v58, 0xff, v58
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[80:83], v[88:91], v[0:3], v59, v65 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[84:87], v[88:91], v[8:11], v58, v65 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v58, s95, v61
	ds_read_b128 v[78:81], v58
	ds_read_b128 v[82:85], v58 offset:2048
	v_add_u32_e32 v58, s96, v62
	ds_read_b128 v[86:89], v58
	s_barrier
	s_setprio 1
	v_and_b32_e32 v57, 0xff, v57
	v_and_b32_e32 v58, 0xff, v64
	v_and_b32_e32 v56, 0xff, v56
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[78:81], v[86:89], v[0:3], v57, v58 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[82:85], v[86:89], v[8:11], v56, v58 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_add_u32 s74, s74, -1
	s_addc_u32 s75, s75, -1
	s_movk_i32 s16, 0xf800
	s_add_u32 s80, s80, 0x100
	s_mov_b32 s17, -1
	s_addc_u32 s81, s81, 0
	v_add_u32_e32 v67, s56, v67
	v_add_u32_e32 v68, s56, v68
	v_add_u32_e32 v69, s54, v69
	v_add_u32_e32 v70, s54, v70
	v_add_u32_e32 v71, s54, v71
	v_lshl_add_u64 v[32:33], v[32:33], 0, s[16:17]
	v_lshl_add_u64 v[34:35], v[34:35], 0, s[76:77]
	v_add_u32_e32 v55, 0x80, v55
	v_add_u32_e32 v66, 0x80, v66
	s_cmp_lg_u64 s[74:75], 0
	s_mov_b32 s16, s96
	s_mov_b32 s17, s95
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v64, v77
	v_mov_b32_e32 v65, v76
	v_mov_b32_e32 v56, v75
	v_mov_b32_e32 v58, v74
	v_mov_b32_e32 v57, v73
	v_mov_b32_e32 v59, v4
	s_cbranch_scc1 .LBB0_8
	v_readlane_b32 s80, v92, 0
	v_mov_b32_e32 v64, v77
	v_mov_b32_e32 v65, v76
	v_mov_b32_e32 v56, v75
	v_mov_b32_e32 v58, v74
	v_mov_b32_e32 v57, v73
	v_mov_b32_e32 v59, v4
	v_readlane_b32 s81, v92, 1
	s_andn2_b64 vcc, exec, s[26:27]
	s_cbranch_vccnz .LBB0_11
.LBB0_10:
	s_barrier
.LBB0_11:
	v_add_u32_e32 v5, s57, v60
	s_waitcnt vmcnt(6)
	s_barrier
	ds_read_b128 v[34:37], v5
	v_add_u32_e32 v7, s57, v61
	v_add_u32_e32 v4, s55, v63
	v_add_u32_e32 v6, s55, v62
	ds_read_b128 v[60:63], v7
	ds_read_b128 v[66:69], v4
	ds_read_b128 v[70:73], v6
	ds_read_b128 v[74:77], v5 offset:2048
	s_waitcnt vmcnt(5)
	v_and_b32_e32 v4, 0xff, v59
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v32, 0xff, v65
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v33, 0xff, v64
	s_mov_b64 s[38:39], s[42:43]
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[34:37], v[66:69], v[0:3], v4, v32 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[34:37], v7 offset:2048
	v_and_b32_e32 v4, 0xff, v57
	s_waitcnt lgkmcnt(2)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[60:63], v[70:73], v[0:3], v4, v33 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_and_b32_e32 v0, 0xff, v58
	s_waitcnt lgkmcnt(1)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[74:77], v[66:69], v[8:11], v0, v32 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mov_b32_e32 v32, s90
	s_nop 1
	v_and_b32_e32 v8, 0xff, v56
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[34:37], v[70:73], v[0:3], v8, v33 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_branch .LBB0_13
.LBB0_12:
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v1, v0
	v_mov_b64_e32 v[6:7], v[2:3]
	s_mov_b64 s[40:41], s[2:3]
	s_mov_b64 s[36:37], s[6:7]
	v_mov_b64_e32 v[4:5], v[0:1]
.LBB0_13:
	s_load_dwordx2 s[44:45], s[0:1], 0x60
	v_cmp_gt_i64_e64 s[0:1], s[62:63], 1
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s47, s67, 0
	s_cselect_b32 s46, s66, 0
	v_mov_b64_e32 v[8:9], s[62:63]
	v_cmp_ge_i64_e32 vcc, s[46:47], v[8:9]
	s_mov_b32 s55, 0
	v_cmp_gt_i64_e64 s[0:1], s[14:15], v[30:31]
	s_cbranch_vccnz .LBB0_16
	v_or_b32_e32 v8, v40, v17
	v_cmp_gt_i64_e64 s[16:17], s[14:15], v[22:23]
	v_add_u32_e32 v23, v8, v41
	v_sub_u32_e32 v8, -4, v15
	v_or_b32_e32 v9, 3, v15
	v_cmp_gt_i32_e32 vcc, 0, v15
	v_cmp_gt_i64_e64 s[18:19], s[12:13], v[26:27]
	s_add_u32 s24, s4, s64
	v_cndmask_b32_e32 v8, v9, v8, vcc
	v_ashrrev_i32_e32 v9, 31, v8
	v_add_u32_sdwa v8, v8, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_ashrrev_i32_e32 v8, 8, v8
	v_ashrrev_i32_e32 v9, 31, v15
	v_xor_b32_e32 v27, v8, v9
	v_lshlrev_b32_e32 v8, 5, v42
	v_sub_u32_e32 v9, v53, v8
	v_lshlrev_b32_e32 v8, 5, v13
	v_add_u32_e32 v9, v9, v8
	v_ashrrev_i32_e32 v10, 31, v9
	v_xor_b32_e32 v9, v10, v9
	v_ashrrev_i32_e32 v11, 31, v9
	v_lshrrev_b32_e32 v11, 25, v11
	v_add_u32_e32 v9, v9, v11
	s_addc_u32 s4, s5, s65
	v_ashrrev_i32_e32 v9, 7, v9
	s_and_b32 s4, s4, 0xffff
	s_lshl_b32 s5, s73, 16
	v_xor_b32_e32 v33, v9, v10
	v_sub_u32_e32 v9, -3, v23
	v_or_b32_e32 v10, 2, v23
	v_cmp_gt_i32_e32 vcc, -2, v23
	s_or_b32 s25, s4, s5
	v_cmp_gt_i64_e64 s[6:7], s[12:13], v[20:21]
	v_cndmask_b32_e32 v9, v10, v9, vcc
	s_add_u32 s28, s8, s68
	v_ashrrev_i32_e32 v10, 31, v9
	v_bitop3_b32 v20, v49, v44, 7 bitop3:0x6c
	v_sub_u32_e32 v30, v50, v51
	s_addc_u32 s8, s9, s69
	v_add_u32_sdwa v9, v9, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_lshl_add_u32 v30, v20, 2, v30
	s_and_b32 s8, s8, 0xffff
	s_lshl_b32 s9, s89, 16
	v_lshrrev_b32_e32 v9, 8, v9
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_ashrrev_i32_e32 v31, 31, v30
	v_cmp_gt_i64_e64 s[4:5], s[12:13], v[28:29]
	s_or_b32 s29, s8, s9
	v_xor_b32_e32 v9, v9, v10
	s_movk_i32 s8, 0xf800
	v_lshlrev_b32_e32 v29, 11, v40
	v_xor_b32_e32 v30, v31, v30
	v_lshlrev_b32_e32 v26, 8, v9
	v_lshlrev_b32_e32 v9, 7, v43
	v_lshlrev_b32_e32 v21, 8, v20
	v_ashrrev_i32_e32 v34, 31, v30
	v_mad_i32_i24 v29, v42, s8, v29
	s_mov_b32 s8, 0xe000
	v_cmp_gt_i64_e64 s[2:3], s[12:13], v[18:19]
	v_mul_i32_i24_e32 v28, 0xfffff800, v42
	v_add3_u32 v18, v52, v45, v21
	v_lshrrev_b32_e32 v34, 29, v34
	v_and_b32_e32 v21, 0xfffffe00, v21
	v_add3_u32 v42, v29, v9, s8
	v_lshl_add_u32 v9, s33, 12, v9
	s_mov_b32 s8, 0x8000
	v_cmp_gt_i64_e64 s[14:15], s[14:15], v[24:25]
	v_lshlrev_b32_e32 v24, 4, v44
	v_add_u32_e32 v30, v30, v34
	v_sub_u32_e32 v34, 0, v21
	v_lshlrev_b32_e32 v21, 8, v44
	v_add3_u32 v44, v9, v28, s8
	v_ashrrev_i32_e32 v28, 31, v23
	v_and_b32_e32 v22, 0xfd, v23
	v_and_b32_e32 v10, 7, v43
	v_xor_b32_e32 v23, v28, v23
	v_bitop3_b32 v10, v13, v10, 4 bitop3:0x36
	v_ashrrev_i32_e32 v29, 31, v23
	v_lshlrev_b32_e32 v38, 4, v10
	v_bitop3_b32 v10, v13, v43, 7 bitop3:0x78
	v_add_u32_sdwa v23, v23, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_or_b32_e32 v29, v40, v15
	v_lshlrev_b32_e32 v39, 4, v10
	v_bitop3_b32 v10, v46, 7, v43 bitop3:0x48
	v_or_b32_e32 v45, 2, v29
	v_sub_u32_e32 v46, -3, v29
	v_cmp_gt_i32_e32 vcc, -2, v29
	v_ashrrev_i32_e32 v23, 8, v23
	v_xor_b32_e32 v23, v23, v28
	v_cndmask_b32_e32 v29, v45, v46, vcc
	v_ashrrev_i32_e32 v45, 31, v29
	v_add_u32_sdwa v29, v29, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_ashrrev_i32_e32 v29, 8, v29
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_and_b32_e32 v28, 32, v12
	v_xor_b32_e32 v29, v29, v45
	v_add3_u32 v29, v29, v28, s46
	v_mul_lo_u32 v29, s56, v29
	v_add3_u32 v23, v23, v28, s46
	v_add3_u32 v29, v40, v15, v29
	v_mad_u64_u32 v[22:23], s[8:9], s56, v23, v[22:23]
	v_add_u32_e32 v40, s46, v32
	v_add_u32_e32 v28, v41, v17
	s_movk_i32 s8, 0xff00
	v_add_u32_e32 v23, v40, v27
	v_mad_i32_i24 v17, v27, s8, v28
	v_mul_lo_u32 v23, s54, v23
	v_add3_u32 v23, v17, v23, 3
	v_add_u32_e32 v17, v40, v33
	v_mov_b32_e32 v19, 0
	v_sub_u32_e32 v26, v29, v26
	v_mad_u64_u32 v[28:29], s[8:9], s54, v17, v[28:29]
	v_mov_b32_e32 v25, v19
	v_lshrrev_b32_e32 v30, 3, v30
	v_mul_i32_i24_e32 v36, -16, v49
	s_lshl_b64 s[8:9], s[46:47], 11
	v_xor_b32_e32 v30, v30, v31
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshlrev_b32_e32 v17, 8, v33
	v_lshl_add_u64 v[24:25], s[8:9], 0, v[24:25]
	v_lshlrev_b32_e32 v30, 9, v30
	v_ashrrev_i32_e32 v35, 31, v34
	v_sub_u32_e32 v17, v28, v17
	v_lshl_add_u64 v[24:25], v[24:25], 0, v[36:37]
	v_lshlrev_b32_e32 v43, 4, v10
	v_ashrrev_i32_e32 v31, 31, v30
	v_sub_u32_e32 v20, v20, v47
	v_sub_u32_e32 v21, v48, v21
	v_add_u32_e32 v27, 1, v17
	v_mul_lo_u32 v17, s54, v40
	v_lshl_add_u64 v[24:25], v[24:25], 0, v[34:35]
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	s_lshl_b32 s11, s85, 7
	s_lshl_b32 s10, s10, 7
	v_lshl_add_u32 v20, v20, 5, v21
	s_lshl_b32 s57, s84, 7
	s_lshl_b32 s70, s86, 7
	s_lshl_b64 s[48:49], s[58:59], 4
	v_add_u32_e32 v29, v15, v17
	v_lshl_add_u64 v[24:25], v[24:25], 0, v[30:31]
	v_lshl_or_b32 v15, s46, 7, v43
	s_ashr_i32 s64, s59, 31
	v_lshlrev_b32_e32 v10, 5, v10
	v_mov_b32_e32 v11, v19
	v_ashrrev_i32_e32 v21, 31, v20
	v_mov_b32_e32 v9, v19
	v_add_u32_e32 v26, 2, v26
	v_add_u32_e32 v28, v54, v17
	v_lshl_add_u64 v[24:25], v[24:25], 0, v[18:19]
	s_lshl_b64 s[50:51], s[46:47], 8
	v_add_u32_e32 v30, v15, v16
	v_add_u32_e32 v31, v15, v14
	v_bfrev_b32_e32 v33, -2
	s_add_i32 s33, s11, 0x8000
	s_add_i32 s49, s10, 0x8000
	s_mov_b32 s65, s64
	s_add_i32 s57, s57, 0xe000
	s_add_i32 s70, s70, 0xe000
	v_add_u32_e32 v34, v42, v39
	v_add_u32_e32 v35, v42, v38
	v_add_u32_e32 v36, v44, v39
	v_add_u32_e32 v37, v44, v38
	s_mov_b64 s[66:67], 0x80
	s_mov_b64 s[68:69], 0x800
	v_mov_b64_e32 v[14:15], s[62:63]
	s_mov_b32 s30, s26
	s_mov_b32 s31, s27
	s_mov_b32 s71, 0
	s_mov_b32 s72, 0
.LBB0_15:
	v_lshl_add_u64 v[38:39], v[10:11], 0, s[50:51]
	v_ashrrev_i32_e32 v16, 31, v25
	v_cmp_gt_i64_e32 vcc, s[52:53], v[38:39]
	v_lshl_add_u64 v[44:45], v[20:21], 0, s[50:51]
	v_lshl_add_u64 v[46:47], v[8:9], 0, s[50:51]
	v_xor_b32_e32 v39, v16, v25
	s_and_b64 s[22:23], s[2:3], vcc
	s_and_b64 vcc, s[6:7], vcc
	v_cmp_gt_i64_e64 s[8:9], s[52:53], v[44:45]
	v_cmp_gt_i64_e64 s[20:21], s[52:53], v[46:47]
	v_lshl_add_u64 v[44:45], v[46:47], 0, s[66:67]
	v_cndmask_b32_e64 v18, v33, v31, s[22:23]
	v_ashrrev_i32_e32 v46, 31, v39
	s_add_u32 s22, s58, s64
	v_xor_b32_e32 v38, v16, v24
	v_mov_b32_e32 v47, v46
	s_addc_u32 s23, s59, s64
	s_mov_b32 m0, s33
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[46:47]
	s_xor_b64 s[62:63], s[22:23], s[64:65]
	buffer_load_dwordx4 v18, s[40:43], 0 offen lds
	v_xor_b32_e32 v57, v38, v46
	v_cvt_f32_u32_e32 v18, s62
	v_cvt_f32_u32_e32 v38, s63
	s_sub_u32 s10, 0, s62
	s_subb_u32 s11, 0, s63
	v_mov_b32_e32 v17, v16
	v_fmac_f32_e32 v18, 0x4f800000, v38
	v_rcp_f32_e32 v18, v18
	v_cndmask_b32_e32 v54, v33, v30, vcc
	v_lshl_add_u64 v[48:49], v[24:25], 0, v[16:17]
	s_mov_b32 m0, s49
	v_mul_f32_e32 v18, 0x5f7ffffc, v18
	v_mul_f32_e32 v38, 0x2f800000, v18
	v_trunc_f32_e32 v38, v38
	v_fmac_f32_e32 v18, 0xcf800000, v38
	v_cvt_u32_f32_e32 v38, v38
	v_cvt_u32_f32_e32 v18, v18
	v_xor_b32_e32 v55, v49, v16
	buffer_load_dwordx4 v54, s[40:43], 0 offen lds
	v_readfirstlane_b32 s22, v38
	v_readfirstlane_b32 s23, v18
	s_mul_i32 s73, s10, s22
	s_mul_hi_u32 s75, s10, s23
	s_mul_i32 s74, s11, s23
	s_add_i32 s73, s75, s73
	s_mul_i32 s76, s10, s23
	s_add_i32 s73, s73, s74
	s_mul_hi_u32 s75, s23, s76
	s_mul_i32 s78, s23, s73
	s_mul_hi_u32 s74, s23, s73
	s_add_u32 s75, s75, s78
	s_mul_hi_u32 s77, s22, s76
	s_mul_i32 s76, s22, s76
	s_addc_u32 s74, 0, s74
	s_add_u32 s75, s75, s76
	s_mul_hi_u32 s79, s22, s73
	s_addc_u32 s74, s74, s77
	s_mul_i32 s73, s22, s73
	s_addc_u32 s75, s79, 0
	s_add_u32 s73, s74, s73
	s_addc_u32 s74, 0, s75
	s_add_u32 s23, s23, s73
	s_addc_u32 s22, s22, s74
	s_mul_hi_u32 s73, s10, s23
	s_mul_i32 s74, s10, s23
	s_mul_i32 s10, s10, s22
	s_mul_i32 s11, s11, s23
	s_add_i32 s10, s73, s10
	s_add_i32 s10, s10, s11
	s_mul_hi_u32 s75, s22, s74
	s_mul_i32 s76, s22, s74
	s_mul_hi_u32 s74, s23, s74
	s_mul_i32 s77, s23, s10
	s_mul_hi_u32 s73, s23, s10
	s_add_u32 s74, s74, s77
	s_addc_u32 s73, 0, s73
	s_add_u32 s74, s74, s76
	s_mul_hi_u32 s11, s22, s10
	s_addc_u32 s73, s73, s75
	s_mul_i32 s10, s22, s10
	s_addc_u32 s11, s11, 0
	s_add_u32 s10, s73, s10
	s_addc_u32 s11, 0, s11
	v_xor_b32_e32 v54, v39, v46
	s_add_u32 s10, s23, s10
	v_cmp_gt_i64_e32 vcc, s[52:53], v[44:45]
	s_addc_u32 s11, s22, s11
	v_mad_u64_u32 v[38:39], s[22:23], v54, s10, 0
	v_mad_u64_u32 v[44:45], s[22:23], v55, s10, 0
	v_add_u32_e32 v40, s71, v29
	s_and_b64 s[22:23], s[18:19], s[20:21]
	v_add_u32_e32 v42, s71, v28
	v_cndmask_b32_e64 v40, v33, v40, s[22:23]
	s_and_b64 s[22:23], s[18:19], vcc
	v_add_u32_e32 v52, s72, v22
	v_cndmask_b32_e64 v42, v33, v42, s[22:23]
	s_and_b64 s[22:23], s[4:5], s[20:21]
	s_and_b64 s[20:21], s[0:1], s[20:21]
	v_add_u32_e32 v50, s71, v27
	v_xor_b32_e32 v17, s64, v46
	v_mul_hi_u32 v18, v57, s10
	v_cndmask_b32_e64 v61, v33, v52, s[20:21]
	v_mad_u64_u32 v[46:47], s[20:21], v57, s11, 0
	v_add_u32_e32 v51, s71, v23
	v_add_u32_e32 v53, s72, v26
	v_cndmask_b32_e64 v59, v33, v50, s[22:23]
	s_and_b64 s[22:23], s[4:5], vcc
	s_and_b64 vcc, s[0:1], vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_xor_b32_e32 v56, v48, v16
	v_cndmask_b32_e64 v60, v33, v51, s[22:23]
	v_cndmask_b32_e32 v62, v33, v53, vcc
	v_add_co_u32_e32 v38, vcc, v46, v38
	v_mad_u64_u32 v[48:49], s[20:21], v54, s11, 0
	v_mad_u64_u32 v[50:51], s[20:21], v56, s11, 0
	buffer_load_ubyte v61, v61, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v62, v62, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v63, v40, s[24:27], 0 offen
	buffer_load_ubyte v64, v42, s[24:27], 0 offen
	s_nop 0
	buffer_load_ubyte v59, v59, s[24:27], 0 offen
	s_nop 0
	buffer_load_ubyte v60, v60, s[24:27], 0 offen
	v_mul_hi_u32 v18, v56, s10
	v_addc_co_u32_e32 v40, vcc, v47, v39, vcc
	v_mov_b32_e32 v41, s55
	v_lshl_add_u64 v[50:51], v[18:19], 0, v[50:51]
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[38:39], v[40:41], 0, v[48:49]
	v_add_co_u32_e32 v18, vcc, v50, v44
	v_mad_u64_u32 v[52:53], s[20:21], v55, s11, 0
	v_mul_lo_u32 v18, s63, v38
	v_mul_lo_u32 v39, s62, v39
	v_mad_u64_u32 v[40:41], s[20:21], s62, v38, 0
	v_addc_co_u32_e32 v42, vcc, v51, v45, vcc
	v_add3_u32 v18, v41, v39, v18
	s_nop 0
	v_addc_co_u32_e32 v53, vcc, 0, v53, vcc
	v_mov_b32_e32 v58, s63
	v_sub_u32_e32 v39, v54, v18
	v_sub_co_u32_e32 v45, vcc, v57, v40
	v_mov_b32_e32 v43, s55
	s_nop 0
	v_subb_co_u32_e64 v39, s[20:21], v39, v58, vcc
	v_subb_co_u32_e32 v18, vcc, v54, v18, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[52:53]
	v_cmp_le_u32_e32 vcc, s62, v45
	v_mul_lo_u32 v48, s63, v42
	v_mad_u64_u32 v[40:41], s[20:21], s62, v42, 0
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_subrev_co_u32_e32 v42, vcc, s62, v45
	v_mul_lo_u32 v43, s62, v43
	s_nop 0
	v_subbrev_co_u32_e32 v39, vcc, 0, v39, vcc
	v_cmp_le_u32_e32 vcc, s62, v42
	v_cmp_le_u32_e64 s[20:21], s63, v39
	v_add_u32_e32 v44, 2, v38
	v_cndmask_b32_e64 v42, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s63, v18
	v_add_u32_e32 v46, 1, v38
	s_mov_b32 m0, s57
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v18
	v_add3_u32 v18, v41, v43, v48
	v_cndmask_b32_e64 v41, 0, -1, s[20:21]
	v_cndmask_b32_e32 v43, v45, v47, vcc
	v_sub_u32_e32 v45, v55, v18
	v_sub_co_u32_e32 v40, vcc, v56, v40
	v_cmp_eq_u32_e64 s[20:21], s63, v39
	v_cmp_ne_u32_e64 s[22:23], 0, v43
	v_lshl_add_u64 v[24:25], v[24:25], 0, s[68:69]
	v_cndmask_b32_e64 v39, v41, v42, s[20:21]
	v_subb_co_u32_e64 v41, s[20:21], v45, v58, vcc
	v_subb_co_u32_e32 v18, vcc, v55, v18, vcc
	v_cmp_le_u32_e32 vcc, s62, v40
	v_subrev_co_u32_e64 v42, s[20:21], s62, v40
	s_nop 0
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_ne_u32_e32 vcc, 0, v39
	v_add_u32_e32 v30, 0x80, v30
	v_add_u32_e32 v31, 0x80, v31
	v_cndmask_b32_e32 v39, v46, v44, vcc
	v_subbrev_co_u32_e64 v44, vcc, 0, v41, s[20:21]
	v_cmp_le_u32_e32 vcc, s62, v42
	v_cndmask_b32_e64 v38, v38, v39, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s63, v44
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_subb_co_u32_e64 v41, vcc, v41, v58, s[20:21]
	v_subrev_co_u32_e32 v47, vcc, s62, v42
	v_cmp_le_u32_e64 s[20:21], s63, v18
	v_cndmask_b32_e64 v39, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s63, v44
	v_cndmask_b32_e64 v48, 0, -1, s[20:21]
	v_cmp_eq_u32_e64 s[20:21], s63, v18
	v_subbrev_co_u32_e32 v41, vcc, 0, v41, vcc
	v_cndmask_b32_e64 v39, v39, v46, s[22:23]
	v_cndmask_b32_e64 v43, v48, v45, s[20:21]
	v_xor_b32_e32 v38, v38, v17
	v_cmp_ne_u32_e32 vcc, 0, v39
	v_sub_u32_e32 v17, v38, v17
	v_cmp_ne_u32_e64 s[20:21], 0, v43
	v_cndmask_b32_e32 v38, v44, v41, vcc
	v_xad_u32 v17, v17, v16, s60
	v_cndmask_b32_e64 v18, v18, v38, s[20:21]
	v_cndmask_b32_e32 v38, v42, v47, vcc
	v_cndmask_b32_e64 v38, v40, v38, s[20:21]
	v_xor_b32_e32 v38, v38, v16
	v_xor_b32_e32 v18, v18, v16
	v_sub_co_u32_e32 v38, vcc, v38, v16
	v_mul_lo_u32 v17, v17, s58
	s_nop 0
	v_subb_co_u32_e32 v16, vcc, v18, v16, vcc
	v_ashrrev_i32_e32 v16, 31, v16
	v_and_b32_e32 v16, s58, v16
	v_add3_u32 v16, v16, v38, v17
	s_and_b64 vcc, s[16:17], s[8:9]
	v_cndmask_b32_e32 v17, v33, v16, vcc
	v_add_u32_e32 v16, s48, v16
	s_and_b64 vcc, s[14:15], s[8:9]
	buffer_load_dword v17, s[36:39], 0 offen lds
	v_cndmask_b32_e32 v16, v33, v16, vcc
	s_mov_b32 m0, s70
	s_add_u32 s46, s46, 1
	buffer_load_dword v16, s[36:39], 0 offen lds
	s_waitcnt vmcnt(8)
	ds_read_b128 v[38:41], v36
	s_waitcnt vmcnt(0)
	ds_read_b128 v[42:45], v34
	ds_read_b128 v[46:49], v36 offset:2048
	ds_read_b128 v[50:53], v35
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[38:41], v[42:45], v[4:7], v63, v61 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[38:41], v37
	ds_read_b128 v[54:57], v37 offset:2048
	s_addc_u32 s47, s47, 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[46:49], v[42:45], v[0:3], v59, v61 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_i32 s72, s72, s56
	s_add_i32 s71, s71, s54
	s_add_u32 s50, s50, 0x100
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[38:41], v[50:53], v[4:7], v64, v62 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_lt_i64_e32 vcc, s[46:47], v[14:15]
	s_addc_u32 s51, s51, 0
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[54:57], v[50:53], v[0:3], v60, v62 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cbranch_vccnz .LBB0_15
.LBB0_16:
	s_nop 2
	v_bfe_u32 v8, v7, 16, 1
	s_movk_i32 s2, 0x7fff
	v_bfe_u32 v9, v6, 16, 1
	v_add3_u32 v8, v7, v8, s2
	v_bfe_u32 v10, v5, 16, 1
	v_add3_u32 v9, v6, v9, s2
	v_lshrrev_b32_e32 v8, 16, v8
	v_mov_b32_e32 v14, 0x7fc0
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v11, v4, 16, 1
	v_add3_u32 v10, v5, v10, s2
	v_lshrrev_b32_e32 v9, 16, v9
	v_cndmask_b32_e32 v7, v14, v8, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v11, v4, v11, s2
	v_lshrrev_b32_e32 v10, 16, v10
	v_cndmask_b32_e32 v6, v14, v9, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_bfe_u32 v5, v3, 16, 1
	v_lshrrev_b32_e32 v11, 16, v11
	v_cndmask_b32_e32 v8, v14, v10, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v9, v2, 16, 1
	v_add3_u32 v5, v3, v5, s2
	v_cndmask_b32_e32 v4, v14, v11, vcc
	v_bfe_u32 v10, v1, 16, 1
	v_add3_u32 v9, v2, v9, s2
	v_lshrrev_b32_e32 v5, 16, v5
	v_cmp_o_f32_e32 vcc, v3, v3
	v_add3_u32 v10, v1, v10, s2
	v_lshrrev_b32_e32 v9, 16, v9
	v_cndmask_b32_e32 v15, v14, v5, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	v_bfe_u32 v11, v0, 16, 1
	v_lshrrev_b32_e32 v10, 16, v10
	v_cndmask_b32_e32 v9, v14, v9, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	v_lshl_or_b32 v2, v13, 2, v32
	v_add3_u32 v11, v0, v11, s2
	v_cndmask_b32_e32 v10, v14, v10, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	v_or_b32_e32 v0, s34, v2
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[2:3], s[2:3], s44, v2, v[12:13]
	s_mul_i32 s2, s44, s35
	s_mul_hi_u32 s3, s44, s34
	v_lshrrev_b32_e32 v11, 16, v11
	v_mov_b32_e32 v1, s35
	s_add_i32 s2, s3, s2
	s_mul_i32 s3, s45, s34
	v_cndmask_b32_e32 v11, v14, v11, vcc
	v_cmp_gt_i64_e32 vcc, s[12:13], v[0:1]
	s_add_i32 s3, s2, s3
	s_mul_i32 s2, s44, s34
	s_and_b64 vcc, vcc, s[0:1]
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s4, s80, s2
	s_addc_u32 s5, s81, s3
	s_lshl_b64 s[2:3], s[60:61], 1
	s_add_u32 s4, s4, s2
	s_addc_u32 s2, s5, s3
	s_and_b32 s3, s44, 0x3fff
	v_not_b32_e32 v3, -2.0
	s_lshl_b32 s3, s3, 16
	s_and_b32 s2, s2, 0xffff
	s_or_b32 s2, s3, s2
	v_cndmask_b32_e32 v1, v3, v2, vcc
	s_or_b32 s5, s2, 2.0
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffd
	v_lshlrev_b32_e32 v1, 1, v1
	buffer_store_short v4, v1, s[4:7], 0 offen
	v_or_b32_e32 v4, 1, v0
	v_mov_b32_e32 v5, s35
	v_cmp_gt_i64_e32 vcc, s[12:13], v[4:5]
	v_add_u32_e32 v1, s44, v2
	s_and_b64 vcc, vcc, s[0:1]
	v_cndmask_b32_e32 v1, v3, v1, vcc
	v_or_b32_e32 v4, 2, v0
	v_lshlrev_b32_e32 v1, 1, v1
	v_cmp_gt_i64_e32 vcc, s[12:13], v[4:5]
	buffer_store_short v8, v1, s[4:7], 0 offen
	v_lshl_add_u32 v1, s44, 1, v2
	s_and_b64 vcc, vcc, s[0:1]
	v_or_b32_e32 v4, 3, v0
	v_cndmask_b32_e32 v1, v3, v1, vcc
	v_cmp_gt_i64_e32 vcc, s[12:13], v[4:5]
	v_lshlrev_b32_e32 v1, 1, v1
	v_mad_u64_u32 v[4:5], s[2:3], s44, 3, v[2:3]
	s_and_b64 vcc, vcc, s[0:1]
	buffer_store_short v6, v1, s[4:7], 0 offen
	v_cndmask_b32_e32 v1, v3, v4, vcc
	v_or_b32_e32 v4, 16, v0
	v_mov_b32_e32 v5, s35
	v_lshlrev_b32_e32 v1, 1, v1
	v_cmp_gt_i64_e32 vcc, s[12:13], v[4:5]
	buffer_store_short v7, v1, s[4:7], 0 offen
	v_lshl_add_u32 v1, s44, 4, v2
	s_and_b64 vcc, vcc, s[0:1]
	v_or_b32_e32 v4, 17, v0
	v_cndmask_b32_e32 v1, v3, v1, vcc
	v_cmp_gt_i64_e32 vcc, s[12:13], v[4:5]
	v_lshlrev_b32_e32 v1, 1, v1
	v_mad_u64_u32 v[4:5], s[2:3], s44, 17, v[2:3]
	s_and_b64 vcc, vcc, s[0:1]
	buffer_store_short v11, v1, s[4:7], 0 offen
	v_cndmask_b32_e32 v1, v3, v4, vcc
	v_or_b32_e32 v4, 18, v0
	v_mov_b32_e32 v5, s35
	v_cmp_gt_i64_e32 vcc, s[12:13], v[4:5]
	v_lshlrev_b32_e32 v1, 1, v1
	v_mad_u64_u32 v[4:5], s[2:3], s44, 18, v[2:3]
	s_and_b64 vcc, vcc, s[0:1]
	buffer_store_short v10, v1, s[4:7], 0 offen
	v_cndmask_b32_e32 v1, v3, v4, vcc
	v_lshlrev_b32_e32 v1, 1, v1
	buffer_store_short v9, v1, s[4:7], 0 offen
	v_or_b32_e32 v0, 19, v0
	v_mov_b32_e32 v1, s35
	v_cmp_gt_i64_e32 vcc, s[12:13], v[0:1]
	v_mad_u64_u32 v[0:1], s[2:3], s44, 19, v[2:3]
	s_and_b64 vcc, vcc, s[0:1]
	v_cndmask_b32_e32 v0, v3, v0, vcc
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v15, v0, s[4:7], 0 offen
	s_endpgm
.LBB0_17:
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, v3
	v_mov_b32_e32 v1, v3
	v_mov_b32_e32 v0, v3
	v_mov_b32_e32 v11, v3
	v_mov_b32_e32 v10, v3
	v_mov_b32_e32 v9, v3
	v_mov_b32_e32 v8, v3
	s_andn2_b64 vcc, exec, s[26:27]
	s_cbranch_vccnz .LBB0_11
	s_branch .LBB0_10
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_dynamic_gemm_128x32x256
		.amdhsa_group_segment_fixed_size 61440
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 104
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
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 97
		.amdhsa_next_free_sgpr 100
		.amdhsa_accum_offset 96
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
.Lfunc_end0:
	.size	wave_mxfp4_dynamic_gemm_128x32x256, .Lfunc_end0-wave_mxfp4_dynamic_gemm_128x32x256

	.set wave_mxfp4_dynamic_gemm_128x32x256.num_vgpr, 93
	.set wave_mxfp4_dynamic_gemm_128x32x256.num_agpr, 0
	.set wave_mxfp4_dynamic_gemm_128x32x256.numbered_sgpr, 100
	.set wave_mxfp4_dynamic_gemm_128x32x256.num_named_barrier, 0
	.set wave_mxfp4_dynamic_gemm_128x32x256.private_seg_size, 0
	.set wave_mxfp4_dynamic_gemm_128x32x256.uses_vcc, 1
	.set wave_mxfp4_dynamic_gemm_128x32x256.uses_flat_scratch, 0
	.set wave_mxfp4_dynamic_gemm_128x32x256.has_dyn_sized_stack, 0
	.set wave_mxfp4_dynamic_gemm_128x32x256.has_recursion, 0
	.set wave_mxfp4_dynamic_gemm_128x32x256.has_indirect_call, 0
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .actual_access:  read_only
        .address_space:  generic
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  generic
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  generic
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  generic
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  generic
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
      - .offset:         96
        .size:           4
        .value_kind:     by_value
      - .offset:         100
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 61440
    .kernarg_segment_align: 8
    .kernarg_segment_size: 104
    .max_flat_workgroup_size: 512
    .name:           wave_mxfp4_dynamic_gemm_128x32x256
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     106
    .sgpr_spill_count: 2
    .symbol:         wave_mxfp4_dynamic_gemm_128x32x256.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     93
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
