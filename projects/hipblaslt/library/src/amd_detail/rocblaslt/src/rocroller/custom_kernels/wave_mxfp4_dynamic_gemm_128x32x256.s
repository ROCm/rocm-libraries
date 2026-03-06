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
	s_load_dwordx2 s[54:55], s[0:1], 0x38
	s_load_dword s33, s[0:1], 0x50
	s_load_dword s66, s[0:1], 0x58
	v_writelane_b32 v144, s16, 0
	v_writelane_b32 v144, s10, 1
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s30, s54
	s_mov_b32 s31, 0
	v_writelane_b32 v144, s11, 2
	v_writelane_b32 v144, s30, 3
	s_ashr_i32 s78, s55, 31
	s_mov_b32 s79, s78
	v_writelane_b32 v144, s31, 4
	s_mov_b32 s31, s55
	s_xor_b64 s[76:77], s[30:31], s[78:79]
	s_mov_b64 s[56:57], s[0:1]
	s_lshr_b32 s0, s77, 31
	s_add_u32 s0, s76, s0
	s_addc_u32 s1, s77, 0
	s_ashr_i32 s82, s77, 31
	s_mov_b64 s[40:41], s[2:3]
	s_ashr_i64 s[2:3], s[0:1], 1
	s_lshr_b32 s0, s82, 27
	s_add_u32 s0, s76, s0
	s_addc_u32 s1, s77, 0
	s_ashr_i64 s[18:19], s[0:1], 5
	s_sub_u32 s10, 0, s54
	s_subb_u32 s11, 0, s55
	s_add_u32 s16, s54, -1
	v_cmp_lt_i64_e64 s[0:1], s[30:31], 1
	s_addc_u32 s22, s55, -1
	s_mov_b64 s[36:37], s[6:7]
	s_and_b64 s[6:7], s[0:1], exec
	s_cselect_b32 s7, s11, s22
	s_cselect_b32 s6, s10, s16
	s_ashr_i32 s10, s7, 31
	s_lshr_b32 s10, s10, 24
	s_add_u32 s6, s6, s10
	s_addc_u32 s7, s7, 0
	s_ashr_i64 s[6:7], s[6:7], 8
	s_sub_u32 s10, 0, s6
	v_and_b32_e32 v96, 0x3ff, v0
	s_subb_u32 s11, 0, s7
	v_bfe_u32 v0, v0, 10, 10
	s_add_u32 s6, s6, 1
	v_lshrrev_b32_e32 v1, 6, v96
	v_lshlrev_b32_e32 v97, 4, v0
	s_addc_u32 s7, s7, 0
	v_lshl_or_b32 v2, v1, 3, v97
	s_and_b64 s[0:1], s[0:1], exec
	v_readfirstlane_b32 s54, v2
	v_lshlrev_b32_e32 v2, 1, v0
	s_cselect_b32 s34, s10, s6
	v_or_b32_e32 v0, v1, v2
	s_mov_b64 s[20:21], -1
	s_cselect_b32 s35, s11, s7
	v_readfirstlane_b32 s0, v0
	s_cmp_gt_i32 s34, 5
	v_readfirstlane_b32 s86, v1
	v_mov_b32_e32 v27, 0
	v_readfirstlane_b32 s1, v2
	v_bfe_u32 v98, v96, 4, 2
	v_lshrrev_b32_e32 v99, 3, v96
	v_lshrrev_b32_e32 v45, 4, v96
	v_writelane_b32 v144, s1, 5
	s_cbranch_scc1 .LBB0_2
	s_ashr_i32 s6, s35, 31
	s_mov_b32 s7, s6
	s_xor_b64 s[10:11], s[6:7], s[34:35]
	s_lshr_b32 s1, s11, 31
	s_add_u32 s10, s10, s1
	s_addc_u32 s11, s11, 0
	s_lshr_b64 s[10:11], s[10:11], 1
	s_xor_b64 s[6:7], s[10:11], s[6:7]
	s_lshl_b64 s[52:53], s[6:7], 1
	s_sub_u32 s6, 0, s14
	s_subb_u32 s7, 0, s15
	v_writelane_b32 v144, s6, 6
	s_mov_b32 s43, 0x27000
	s_mov_b32 s42, 0x7ffffffe
	v_writelane_b32 v144, s7, 7
	s_add_u32 s6, s14, -1
	s_addc_u32 s7, s15, -1
	s_sub_u32 s58, 0, s12
	s_subb_u32 s59, 0, s13
	v_writelane_b32 v144, s6, 8
	s_add_u32 s60, s12, -1
	s_addc_u32 s61, s13, -1
	v_writelane_b32 v144, s7, 9
	s_ashr_i32 s6, s15, 31
	s_mov_b32 s7, s6
	s_xor_b64 s[10:11], s[14:15], s[6:7]
	s_ashr_i32 s1, s11, 31
	s_and_b32 s23, s41, 0xffff
	s_and_b32 s25, s5, 0xffff
	s_lshr_b32 s16, s1, 27
	s_add_u32 s26, s10, s16
	s_addc_u32 s27, s11, 0
	s_ashr_i64 s[26:27], s[26:27], 5
	s_xor_b64 s[26:27], s[26:27], s[6:7]
	s_lshr_b32 s1, s1, 22
	s_add_u32 s10, s10, s1
	s_addc_u32 s11, s11, 0
	s_ashr_i64 s[10:11], s[10:11], 10
	s_xor_b64 s[70:71], s[10:11], s[6:7]
	s_lshl_b64 s[48:49], s[70:71], 5
	s_sub_u32 s6, s26, s48
	s_subb_u32 s7, s27, s49
	v_cmp_gt_i64_e64 s[10:11], s[6:7], 1
	s_and_b64 s[10:11], s[10:11], exec
	s_cselect_b32 s51, s7, 0
	s_cselect_b32 s50, s6, 1
	s_and_b32 s1, s14, 0x3ff
	s_mov_b64 s[20:21], 0
	v_or_b32_e32 v26, v99, v97
	s_mov_b32 s22, s40
	s_mov_b32 s24, s4
	s_and_b32 s27, s37, 0xffff
	s_mov_b32 s26, s36
	s_and_b32 s29, s9, 0xffff
	s_mov_b32 s28, s8
	v_mov_b32_e32 v16, s1
	s_mov_b64 s[6:7], s[42:43]
	s_mov_b64 s[38:39], s[42:43]
	s_mov_b64 s[10:11], s[42:43]
	s_branch .LBB0_3
.LBB0_2:
.LBB0_3:
	s_or_b32 s1, s54, 32
	v_writelane_b32 v144, s1, 10
	s_or_b32 s1, s54, 64
	v_writelane_b32 v144, s1, 11
	s_or_b32 s1, s54, 0x60
	s_xor_b64 s[46:47], s[18:19], s[78:79]
	v_writelane_b32 v144, s1, 12
	v_cmp_lt_i64_e64 s[18:19], s[14:15], 1
	s_xor_b64 s[44:45], s[2:3], s[78:79]
	s_lshl_b32 s67, s0, 8
	v_writelane_b32 v144, s18, 13
	s_andn2_b64 vcc, exec, s[20:21]
	v_cmp_lt_i64_e64 s[72:73], s[12:13], 1
	v_lshlrev_b32_e32 v24, 7, v96
	v_writelane_b32 v144, s19, 14
	v_writelane_b32 v144, s54, 15
	s_cbranch_vccnz .LBB0_10
	s_sub_u32 s2, 0, s14
	s_subb_u32 s3, 0, s15
	s_add_u32 s6, s14, -1
	s_addc_u32 s7, s15, -1
	s_and_b64 s[0:1], s[18:19], exec
	s_cselect_b32 s0, s2, s6
	v_writelane_b32 v144, s2, 6
	v_or_b32_e32 v26, v99, v97
	v_mov_b32_e32 v27, 0
	v_writelane_b32 v144, s3, 7
	v_writelane_b32 v144, s6, 8
	s_cselect_b32 s1, s3, s7
	s_ashr_i32 s2, s1, 31
	s_lshr_b32 s2, s2, 27
	s_add_u32 s0, s0, s2
	s_addc_u32 s1, s1, 0
	s_ashr_i64 s[0:1], s[0:1], 5
	s_sub_u32 s2, 0, s0
	s_subb_u32 s3, 0, s1
	v_writelane_b32 v144, s7, 9
	s_add_u32 s6, s0, 1
	s_addc_u32 s7, s1, 0
	s_and_b64 s[0:1], s[18:19], exec
	s_cselect_b32 s0, s2, s6
	s_cselect_b32 s1, s3, s7
	s_and_b32 s2, s0, 31
	s_max_u32 s16, s2, 1
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[2:3], -1, 0
	s_sub_u32 s58, 0, s12
	s_subb_u32 s59, 0, s13
	s_add_u32 s64, s12, -1
	s_addc_u32 s65, s13, -1
	s_and_b64 s[6:7], s[72:73], exec
	s_cselect_b32 s7, s59, s65
	s_cselect_b32 s6, s58, s64
	s_ashr_i32 s10, s7, 31
	s_lshr_b32 s10, s10, 25
	s_add_u32 s6, s6, s10
	s_addc_u32 s7, s7, 0
	s_ashr_i64 s[6:7], s[6:7], 7
	s_sub_u32 s10, 0, s6
	s_subb_u32 s11, 0, s7
	s_add_u32 s18, s6, 1
	s_addc_u32 s19, s7, 0
	s_and_b64 s[6:7], s[72:73], exec
	s_cselect_b32 s39, s11, s19
	s_cselect_b32 s38, s10, s18
	s_mul_i32 s6, s39, s17
	s_mul_hi_u32 s7, s38, s17
	s_add_i32 s10, s7, s6
	s_mul_i32 s11, s38, s17
	v_readlane_b32 s19, v144, 0
	s_add_u32 s22, s11, s19
	s_addc_u32 s23, s10, 0
	s_ashr_i32 s6, s1, 31
	s_mov_b32 s7, s6
	s_xor_b64 s[0:1], s[6:7], s[0:1]
	s_lshr_b32 s18, s1, 27
	s_add_u32 s0, s0, s18
	s_addc_u32 s1, s1, 0
	s_lshr_b64 s[0:1], s[0:1], 5
	s_xor_b64 s[0:1], s[0:1], s[6:7]
	s_mul_i32 s1, s38, s1
	s_mul_hi_u32 s6, s38, s0
	s_add_i32 s1, s6, s1
	s_mul_i32 s6, s39, s0
	s_add_i32 s1, s1, s6
	s_mul_i32 s0, s38, s0
	v_cvt_f32_u32_e32 v1, s16
	s_lshl_b64 s[0:1], s[0:1], 5
	v_mov_b64_e32 v[2:3], s[0:1]
	v_cmp_ge_i64_e32 vcc, s[22:23], v[2:3]
	v_mov_b32_e32 v2, 0x4f800000
	v_fmac_f32_e32 v1, 0, v2
	v_rcp_f32_e32 v1, v1
	s_and_b64 s[2:3], s[2:3], vcc
	s_sub_u32 s0, s19, s0
	s_subb_u32 s1, 0, s1
	v_mul_f32_e32 v1, 0x5f7ffffc, v1
	v_mul_f32_e32 v2, 0x2f800000, v1
	v_trunc_f32_e32 v2, v2
	s_add_u32 s0, s0, s11
	v_fmac_f32_e32 v1, 0xcf800000, v2
	s_addc_u32 s1, s1, s10
	v_cvt_u32_f32_e32 v3, v2
	v_cvt_u32_f32_e32 v1, v1
	s_ashr_i32 s6, s1, 31
	s_mov_b32 s7, s6
	s_xor_b64 s[0:1], s[6:7], s[0:1]
	s_sub_u32 s10, 0, s16
	v_readfirstlane_b32 s18, v3
	v_readfirstlane_b32 s20, v1
	s_subb_u32 s11, 0, 0
	s_mul_i32 s19, s10, s18
	s_mul_hi_u32 s21, s10, s20
	s_add_i32 s19, s21, s19
	s_mul_i32 s21, s11, s20
	s_add_i32 s19, s19, s21
	s_mul_i32 s25, s10, s20
	s_mul_i32 s24, s20, s19
	s_mul_hi_u32 s26, s20, s25
	s_mul_hi_u32 s21, s20, s19
	s_add_u32 s24, s26, s24
	s_addc_u32 s21, 0, s21
	s_mul_hi_u32 s27, s18, s25
	s_mul_i32 s25, s18, s25
	s_add_u32 s24, s24, s25
	s_mul_hi_u32 s26, s18, s19
	s_addc_u32 s21, s21, s27
	s_addc_u32 s24, s26, 0
	s_mul_i32 s19, s18, s19
	s_add_u32 s19, s21, s19
	s_addc_u32 s21, 0, s24
	s_add_u32 s19, s20, s19
	s_addc_u32 s18, s18, s21
	s_mul_i32 s20, s10, s18
	s_mul_hi_u32 s21, s10, s19
	s_add_i32 s20, s21, s20
	s_mul_i32 s11, s11, s19
	s_add_i32 s20, s20, s11
	s_mul_i32 s10, s10, s19
	s_mul_hi_u32 s21, s18, s10
	s_mul_i32 s24, s18, s10
	s_mul_i32 s26, s19, s20
	s_mul_hi_u32 s10, s19, s10
	s_mul_hi_u32 s25, s19, s20
	s_add_u32 s10, s10, s26
	s_addc_u32 s25, 0, s25
	s_add_u32 s10, s10, s24
	s_mul_hi_u32 s11, s18, s20
	s_addc_u32 s10, s25, s21
	s_addc_u32 s11, s11, 0
	s_mul_i32 s20, s18, s20
	s_add_u32 s10, s10, s20
	s_addc_u32 s11, 0, s11
	s_add_u32 s19, s19, s10
	s_addc_u32 s18, s18, s11
	s_ashr_i32 s10, s1, 31
	s_add_u32 s0, s0, s10
	s_mov_b32 s11, s10
	s_addc_u32 s1, s1, s10
	s_xor_b64 s[0:1], s[0:1], s[10:11]
	s_mul_i32 s21, s0, s18
	s_mul_hi_u32 s24, s0, s19
	s_mul_hi_u32 s20, s0, s18
	s_add_u32 s21, s24, s21
	s_addc_u32 s20, 0, s20
	s_mul_hi_u32 s25, s1, s19
	s_mul_i32 s19, s1, s19
	s_add_u32 s19, s21, s19
	s_mul_hi_u32 s24, s1, s18
	s_addc_u32 s19, s20, s25
	s_addc_u32 s20, s24, 0
	s_mul_i32 s18, s1, s18
	s_add_u32 s18, s19, s18
	s_addc_u32 s19, 0, s20
	s_add_u32 s20, s18, 1
	s_addc_u32 s21, s19, 0
	s_add_u32 s24, s18, 2
	s_mul_i32 s26, s16, s19
	s_mul_hi_u32 s27, s16, s18
	s_addc_u32 s25, s19, 0
	s_add_i32 s27, s27, s26
	s_mul_i32 s26, s16, s18
	s_sub_u32 s0, s0, s26
	s_subb_u32 s1, s1, s27
	s_sub_u32 s26, s0, s16
	s_subb_u32 s27, s1, 0
	s_cmp_ge_u32 s26, s16
	s_cselect_b32 s26, -1, 0
	s_cmp_eq_u32 s27, 0
	s_cselect_b32 s26, s26, -1
	s_cmp_lg_u32 s26, 0
	s_cselect_b32 s20, s24, s20
	s_cselect_b32 s21, s25, s21
	s_cmp_ge_u32 s0, s16
	s_cselect_b32 s0, -1, 0
	s_cmp_eq_u32 s1, 0
	s_cselect_b32 s0, s0, -1
	s_cmp_lg_u32 s0, 0
	s_cselect_b32 s1, s21, s19
	s_cselect_b32 s0, s20, s18
	s_xor_b64 s[0:1], s[0:1], s[10:11]
	s_sub_u32 s0, s0, s10
	s_subb_u32 s1, s1, s10
	s_xor_b64 s[6:7], s[0:1], s[6:7]
	s_ashr_i32 s0, s23, 31
	s_mov_b32 s1, s0
	s_xor_b64 s[10:11], s[0:1], s[22:23]
	s_ashr_i32 s16, s11, 31
	s_lshr_b32 s16, s16, 27
	s_add_u32 s10, s10, s16
	s_addc_u32 s11, s11, 0
	s_ashr_i64 s[10:11], s[10:11], 5
	s_xor_b64 s[62:63], s[10:11], s[0:1]
	s_ashr_i32 s0, s39, 31
	s_add_u32 s10, s38, s0
	s_mov_b32 s1, s0
	s_addc_u32 s11, s39, s0
	s_xor_b64 s[10:11], s[10:11], s[0:1]
	v_cvt_f32_u32_e32 v1, s10
	v_cvt_f32_u32_e32 v2, s11
	s_sub_u32 s0, 0, s10
	s_subb_u32 s1, 0, s11
	v_mov_b32_e32 v33, v27
	v_fmac_f32_e32 v1, 0x4f800000, v2
	v_rcp_f32_e32 v1, v1
	v_bfrev_b32_e32 v6, -2
	s_mov_b32 s43, 0x27000
	s_mov_b32 s42, 0x7ffffffe
	v_mul_f32_e32 v1, 0x5f7ffffc, v1
	v_mul_f32_e32 v2, 0x2f800000, v1
	v_trunc_f32_e32 v2, v2
	v_fmac_f32_e32 v1, 0xcf800000, v2
	v_cvt_u32_f32_e32 v1, v1
	v_cvt_u32_f32_e32 v2, v2
	v_mov_b32_e32 v25, v27
	v_readfirstlane_b32 s16, v1
	v_readfirstlane_b32 s20, v2
	s_mul_hi_u32 s19, s0, s16
	s_mul_i32 s21, s0, s20
	s_mul_i32 s18, s1, s16
	s_add_i32 s19, s19, s21
	s_add_i32 s19, s19, s18
	s_mul_i32 s24, s0, s16
	s_mul_i32 s21, s16, s19
	s_mul_hi_u32 s25, s16, s24
	s_mul_hi_u32 s18, s16, s19
	s_add_u32 s21, s25, s21
	s_addc_u32 s18, 0, s18
	s_mul_hi_u32 s26, s20, s24
	s_mul_i32 s24, s20, s24
	s_add_u32 s21, s21, s24
	s_mul_hi_u32 s25, s20, s19
	s_addc_u32 s18, s18, s26
	s_addc_u32 s21, s25, 0
	s_mul_i32 s19, s20, s19
	s_add_u32 s18, s18, s19
	s_addc_u32 s19, 0, s21
	s_add_u32 s16, s16, s18
	s_addc_u32 s18, s20, s19
	s_mul_i32 s19, s0, s18
	s_mul_hi_u32 s20, s0, s16
	s_add_i32 s19, s20, s19
	s_mul_i32 s1, s1, s16
	s_add_i32 s19, s19, s1
	s_mul_i32 s0, s0, s16
	s_mul_hi_u32 s20, s18, s0
	s_mul_i32 s21, s18, s0
	s_mul_i32 s25, s16, s19
	s_mul_hi_u32 s0, s16, s0
	s_mul_hi_u32 s24, s16, s19
	s_add_u32 s0, s0, s25
	s_addc_u32 s24, 0, s24
	s_add_u32 s0, s0, s21
	s_mul_hi_u32 s1, s18, s19
	s_addc_u32 s0, s24, s20
	s_addc_u32 s1, s1, 0
	s_mul_i32 s19, s18, s19
	s_add_u32 s0, s0, s19
	s_addc_u32 s1, 0, s1
	s_add_u32 s16, s16, s0
	s_addc_u32 s18, s18, s1
	s_ashr_i32 s68, s63, 31
	s_add_u32 s0, s62, s68
	s_mov_b32 s69, s68
	s_addc_u32 s1, s63, s68
	s_xor_b64 s[0:1], s[0:1], s[68:69]
	s_mul_i32 s20, s0, s18
	s_mul_hi_u32 s21, s0, s16
	s_mul_hi_u32 s19, s0, s18
	s_add_u32 s20, s21, s20
	s_addc_u32 s19, 0, s19
	s_mul_hi_u32 s24, s1, s16
	s_mul_i32 s16, s1, s16
	s_add_u32 s16, s20, s16
	s_mul_hi_u32 s21, s1, s18
	s_addc_u32 s16, s19, s24
	s_addc_u32 s19, s21, 0
	s_mul_i32 s18, s1, s18
	s_add_u32 s16, s16, s18
	s_addc_u32 s18, 0, s19
	s_mul_i32 s18, s10, s18
	s_mul_hi_u32 s19, s10, s16
	s_add_i32 s18, s19, s18
	s_mul_i32 s19, s11, s16
	s_add_i32 s24, s18, s19
	s_sub_i32 s20, s1, s24
	s_mul_i32 s16, s10, s16
	s_sub_u32 s0, s0, s16
	s_cselect_b64 s[18:19], -1, 0
	s_subb_u32 s16, s20, s11
	s_sub_u32 s25, s0, s10
	s_cselect_b64 s[20:21], -1, 0
	s_subb_u32 s26, s16, 0
	s_cmp_ge_u32 s26, s11
	s_cselect_b32 s27, -1, 0
	s_cmp_ge_u32 s25, s10
	s_cselect_b32 s28, -1, 0
	s_cmp_eq_u32 s26, s11
	s_cselect_b32 s27, s28, s27
	s_cmp_lg_u64 s[20:21], 0
	s_subb_u32 s16, s16, s11
	s_sub_u32 s20, s25, s10
	s_subb_u32 s16, s16, 0
	s_cmp_lg_u32 s27, 0
	s_cselect_b32 s20, s20, s25
	s_cselect_b32 s16, s16, s26
	s_cmp_lg_u64 s[18:19], 0
	s_subb_u32 s1, s1, s24
	s_cmp_ge_u32 s1, s11
	s_cselect_b32 s18, -1, 0
	s_cmp_ge_u32 s0, s10
	s_cselect_b32 s10, -1, 0
	s_cmp_eq_u32 s1, s11
	s_cselect_b32 s10, s10, s18
	s_cmp_lg_u32 s10, 0
	s_cselect_b32 s1, s16, s1
	s_cselect_b32 s0, s20, s0
	s_xor_b64 s[0:1], s[0:1], s[68:69]
	s_sub_u32 s10, s0, s68
	s_subb_u32 s11, s1, s68
	s_ashr_i32 s0, s11, 31
	s_mov_b32 s1, s0
	s_and_b64 s[0:1], s[0:1], s[38:39]
	s_add_u32 s10, s0, s10
	s_addc_u32 s11, s1, s11
	s_and_b64 s[0:1], s[2:3], exec
	s_cselect_b32 s1, s7, s11
	s_cselect_b32 s0, s6, s10
	s_lshl_b64 s[10:11], s[0:1], 7
	v_or_b32_e32 v2, s10, v26
	v_bitop3_b32 v1, v99, 7, v96 bitop3:0x48
	v_mov_b32_e32 v3, s11
	v_mad_u64_u32 v[30:31], s[2:3], v2, s44, 0
	v_lshlrev_b32_e32 v32, 5, v1
	v_lshlrev_b32_e32 v28, 4, v1
	v_cmp_gt_i64_e32 vcc, s[30:31], v[32:33]
	v_cmp_gt_i64_e64 s[2:3], s[12:13], v[2:3]
	v_add_u32_e32 v1, v28, v30
	s_and_b64 s[6:7], vcc, s[2:3]
	s_lshl_b32 s54, s54, 7
	v_or_b32_e32 v4, 32, v2
	v_mov_b32_e32 v5, s11
	s_and_b32 s41, s41, 0xffff
	v_cndmask_b32_e64 v1, v6, v1, s[6:7]
	s_mov_b32 m0, s54
	v_mad_u64_u32 v[34:35], s[6:7], v4, s44, 0
	v_cmp_gt_i64_e64 s[26:27], s[12:13], v[4:5]
	buffer_load_dwordx4 v1, s[40:43], 0 offen lds
	v_add_u32_e32 v1, v28, v34
	s_and_b64 s[6:7], vcc, s[26:27]
	s_add_i32 s16, s54, 0x1000
	v_or_b32_e32 v4, 64, v2
	v_cndmask_b32_e64 v1, v6, v1, s[6:7]
	s_mov_b32 m0, s16
	v_mad_u64_u32 v[36:37], s[6:7], v4, s44, 0
	v_cmp_gt_i64_e64 s[18:19], s[12:13], v[4:5]
	buffer_load_dwordx4 v1, s[40:43], 0 offen lds
	v_add_u32_e32 v1, v28, v36
	s_and_b64 s[6:7], vcc, s[18:19]
	s_add_i32 s60, s54, 0x2000
	v_or_b32_e32 v2, 0x60, v2
	v_cndmask_b32_e64 v1, v6, v1, s[6:7]
	s_mov_b32 m0, s60
	v_mad_u64_u32 v[38:39], s[6:7], v2, s44, 0
	v_cmp_gt_i64_e64 s[20:21], s[12:13], v[2:3]
	buffer_load_dwordx4 v1, s[40:43], 0 offen lds
	v_add_u32_e32 v1, v28, v38
	s_and_b64 vcc, vcc, s[20:21]
	s_add_i32 s61, s54, 0x3000
	v_cndmask_b32_e32 v1, v6, v1, vcc
	s_mov_b32 m0, s61
	s_lshl_b32 s74, s86, 13
	buffer_load_dwordx4 v1, s[40:43], 0 offen lds
	s_sub_i32 s6, 0, s74
	s_lshl_b64 s[0:1], s[0:1], 2
	s_lshl_b32 s10, s30, 5
	s_ashr_i32 s7, s6, 31
	v_or_b32_e32 v2, s0, v0
	v_lshl_add_u64 v[0:1], s[6:7], 0, v[24:25]
	s_mul_i32 s6, s10, s1
	s_lshr_b64 s[0:1], s[30:31], 27
	v_mul_lo_u32 v3, s0, v2
	v_add_u32_e32 v3, s6, v3
	v_mad_u64_u32 v[40:41], s[0:1], s10, v2, v[0:1]
	v_add_u32_e32 v41, v3, v41
	v_or_b32_e32 v0, s31, v41
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_ashrrev_i32_e32 v0, 31, v41
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[24:25], exec, s[0:1]
	s_cbranch_execz .LBB0_6
	s_ashr_i32 s0, s31, 31
	s_add_u32 s6, s30, s0
	s_mov_b32 s1, s0
	s_addc_u32 s7, s31, s0
	s_xor_b64 s[28:29], s[6:7], s[0:1]
	v_cvt_f32_u32_e32 v1, s28
	v_cvt_f32_u32_e32 v2, s29
	s_sub_u32 s0, 0, s28
	s_subb_u32 s1, 0, s29
	v_mov_b32_e32 v5, v27
	v_fmac_f32_e32 v1, 0x4f800000, v2
	v_rcp_f32_e32 v1, v1
	s_mov_b32 s51, 0
	v_mul_f32_e32 v1, 0x5f7ffffc, v1
	v_mul_f32_e32 v2, 0x2f800000, v1
	v_trunc_f32_e32 v2, v2
	v_fmac_f32_e32 v1, 0xcf800000, v2
	v_cvt_u32_f32_e32 v2, v2
	v_cvt_u32_f32_e32 v1, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v1
	s_mul_i32 s10, s0, s6
	s_mul_hi_u32 s48, s0, s7
	s_mul_i32 s11, s1, s7
	s_add_i32 s10, s48, s10
	s_add_i32 s10, s10, s11
	s_mul_i32 s49, s0, s7
	s_mul_i32 s48, s7, s10
	s_mul_hi_u32 s50, s7, s49
	s_mul_hi_u32 s11, s7, s10
	s_add_u32 s48, s50, s48
	s_addc_u32 s11, 0, s11
	s_mul_hi_u32 s52, s6, s49
	s_mul_i32 s49, s6, s49
	s_add_u32 s48, s48, s49
	s_mul_hi_u32 s50, s6, s10
	s_addc_u32 s11, s11, s52
	s_addc_u32 s48, s50, 0
	s_mul_i32 s10, s6, s10
	s_add_u32 s10, s11, s10
	s_addc_u32 s11, 0, s48
	s_add_u32 s7, s7, s10
	s_addc_u32 s6, s6, s11
	s_mul_i32 s10, s0, s6
	s_mul_hi_u32 s11, s0, s7
	s_add_i32 s10, s11, s10
	s_mul_i32 s1, s1, s7
	s_add_i32 s10, s10, s1
	s_mul_i32 s0, s0, s7
	s_mul_hi_u32 s11, s6, s0
	s_mul_i32 s48, s6, s0
	s_mul_i32 s50, s7, s10
	s_mul_hi_u32 s0, s7, s0
	s_mul_hi_u32 s49, s7, s10
	s_add_u32 s0, s0, s50
	s_addc_u32 s49, 0, s49
	s_add_u32 s0, s0, s48
	s_mul_hi_u32 s1, s6, s10
	s_addc_u32 s0, s49, s11
	s_addc_u32 s1, s1, 0
	s_mul_i32 s10, s6, s10
	s_add_u32 s0, s0, s10
	s_addc_u32 s1, 0, s1
	v_mov_b32_e32 v1, v0
	s_add_u32 s7, s7, s0
	v_lshl_add_u64 v[2:3], v[40:41], 0, v[0:1]
	s_addc_u32 s6, s6, s1
	v_xor_b32_e32 v8, v2, v0
	v_xor_b32_e32 v1, v3, v0
	v_mad_u64_u32 v[2:3], s[0:1], v8, s6, 0
	v_mul_hi_u32 v4, v8, s7
	v_lshl_add_u64 v[2:3], v[4:5], 0, v[2:3]
	v_mad_u64_u32 v[6:7], s[0:1], v1, s7, 0
	v_add_co_u32_e32 v2, vcc, v2, v6
	v_mad_u64_u32 v[4:5], s[0:1], v1, s6, 0
	s_nop 0
	v_addc_co_u32_e32 v2, vcc, v3, v7, vcc
	v_mov_b32_e32 v3, s51
	s_nop 0
	v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	v_mul_lo_u32 v4, s29, v2
	v_mul_lo_u32 v5, s28, v3
	v_mad_u64_u32 v[2:3], s[0:1], s28, v2, 0
	v_add3_u32 v3, v3, v5, v4
	v_sub_u32_e32 v4, v1, v3
	v_mov_b32_e32 v5, s29
	v_sub_co_u32_e32 v2, vcc, v8, v2
	s_nop 1
	v_subb_co_u32_e64 v4, s[6:7], v4, v5, vcc
	v_subrev_co_u32_e64 v6, s[6:7], s28, v2
	v_subb_co_u32_e32 v1, vcc, v1, v3, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v7, s[10:11], 0, v4, s[6:7]
	v_cmp_le_u32_e64 s[10:11], s29, v7
	v_subb_co_u32_e64 v4, s[6:7], v4, v5, s[6:7]
	s_nop 0
	v_cndmask_b32_e64 v8, 0, -1, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s28, v6
	v_subrev_co_u32_e64 v5, s[6:7], s28, v6
	s_nop 0
	v_cndmask_b32_e64 v9, 0, -1, s[10:11]
	v_cmp_eq_u32_e64 s[10:11], s29, v7
	v_subbrev_co_u32_e64 v4, s[6:7], 0, v4, s[6:7]
	s_nop 0
	v_cndmask_b32_e64 v8, v8, v9, s[10:11]
	v_cmp_le_u32_e32 vcc, s29, v1
	v_cmp_ne_u32_e64 s[6:7], 0, v8
	s_nop 0
	v_cndmask_b32_e64 v3, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v2
	v_cndmask_b32_e64 v4, v7, v4, s[6:7]
	s_nop 0
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v1
	s_nop 1
	v_cndmask_b32_e32 v3, v3, v7, vcc
	v_cmp_ne_u32_e32 vcc, 0, v3
	v_cndmask_b32_e64 v3, v6, v5, s[6:7]
	s_nop 0
	v_cndmask_b32_e32 v2, v2, v3, vcc
	v_cndmask_b32_e32 v1, v1, v4, vcc
	v_xor_b32_e32 v2, v2, v0
	v_xor_b32_e32 v1, v1, v0
	v_sub_co_u32_e32 v2, vcc, v2, v0
	s_nop 1
	v_subb_co_u32_e32 v3, vcc, v1, v0, vcc
.LBB0_6:
	s_andn2_saveexec_b64 s[6:7], s[24:25]
	s_cbranch_execz .LBB0_8
	v_cvt_f32_u32_e32 v1, s30
	s_sub_i32 s0, 0, s30
	v_mov_b32_e32 v3, 0
	v_rcp_iflag_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_mul_lo_u32 v2, s0, v1
	v_mul_hi_u32 v2, v1, v2
	v_add_u32_e32 v1, v1, v2
	v_mul_hi_u32 v1, v40, v1
	v_mul_lo_u32 v1, v1, s30
	v_sub_u32_e32 v1, v40, v1
	v_subrev_u32_e32 v2, s30, v1
	v_cmp_le_u32_e32 vcc, s30, v1
	s_nop 1
	v_cndmask_b32_e32 v1, v1, v2, vcc
	v_subrev_u32_e32 v2, s30, v1
	v_cmp_le_u32_e32 vcc, s30, v1
	s_nop 1
	v_cndmask_b32_e32 v2, v1, v2, vcc
.LBB0_8:
	v_writelane_b32 v144, s78, 16
	s_nop 1
	v_writelane_b32 v144, s79, 17
	v_writelane_b32 v144, s72, 18
	s_nop 1
	v_writelane_b32 v144, s73, 19
	v_writelane_b32 v144, s64, 20
	s_nop 1
	v_writelane_b32 v144, s65, 21
	v_writelane_b32 v144, s58, 22
	s_nop 1
	v_writelane_b32 v144, s59, 23
	v_writelane_b32 v144, s56, 24
	s_nop 1
	v_writelane_b32 v144, s57, 25
	s_or_b64 exec, exec, s[6:7]
	s_ashr_i32 s80, s31, 31
	s_add_u32 s0, s30, s80
	s_mov_b32 s81, s80
	s_addc_u32 s1, s31, s80
	s_xor_b64 s[10:11], s[0:1], s[80:81]
	v_cvt_f32_u32_e32 v1, s10
	v_cvt_f32_u32_e32 v4, s11
	s_sub_u32 s0, 0, s10
	s_subb_u32 s1, 0, s11
	v_xor_b32_e32 v5, v0, v41
	v_fmac_f32_e32 v1, 0x4f800000, v4
	v_rcp_f32_e32 v1, v1
	v_xor_b32_e32 v4, v0, v40
	v_mov_b32_e32 v9, 0
	s_mov_b32 s49, 0
	v_mul_f32_e32 v1, 0x5f7ffffc, v1
	v_mul_f32_e32 v6, 0x2f800000, v1
	v_trunc_f32_e32 v6, v6
	v_fmac_f32_e32 v1, 0xcf800000, v6
	v_cvt_u32_f32_e32 v6, v6
	v_cvt_u32_f32_e32 v1, v1
	v_readfirstlane_b32 s6, v6
	v_readfirstlane_b32 s7, v1
	s_mul_hi_u32 s25, s0, s7
	s_mul_i32 s28, s0, s6
	s_mul_i32 s24, s1, s7
	s_add_i32 s25, s25, s28
	s_add_i32 s25, s25, s24
	s_mul_i32 s29, s0, s7
	s_mul_i32 s28, s7, s25
	s_mul_hi_u32 s48, s7, s29
	s_mul_hi_u32 s24, s7, s25
	s_add_u32 s28, s48, s28
	s_addc_u32 s24, 0, s24
	s_mul_hi_u32 s50, s6, s29
	s_mul_i32 s29, s6, s29
	s_add_u32 s28, s28, s29
	s_mul_hi_u32 s48, s6, s25
	s_addc_u32 s24, s24, s50
	s_addc_u32 s28, s48, 0
	s_mul_i32 s25, s6, s25
	s_add_u32 s24, s24, s25
	s_addc_u32 s25, 0, s28
	s_add_u32 s7, s7, s24
	s_addc_u32 s6, s6, s25
	s_mul_i32 s24, s0, s6
	s_mul_hi_u32 s25, s0, s7
	s_add_i32 s24, s25, s24
	s_mul_i32 s1, s1, s7
	s_add_i32 s24, s24, s1
	s_mul_i32 s0, s0, s7
	s_mul_hi_u32 s25, s6, s0
	s_mul_i32 s28, s6, s0
	s_mul_i32 s48, s7, s24
	s_mul_hi_u32 s0, s7, s0
	s_mul_hi_u32 s29, s7, s24
	s_add_u32 s0, s0, s48
	s_addc_u32 s29, 0, s29
	s_add_u32 s0, s0, s28
	s_mul_hi_u32 s1, s6, s24
	s_addc_u32 s0, s29, s25
	s_addc_u32 s1, s1, 0
	s_mul_i32 s24, s6, s24
	s_add_u32 s0, s0, s24
	v_ashrrev_i32_e32 v6, 31, v5
	s_addc_u32 s1, 0, s1
	v_mov_b32_e32 v7, v6
	s_add_u32 s7, s7, s0
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[6:7]
	s_addc_u32 s6, s6, s1
	v_xor_b32_e32 v7, v4, v6
	v_xor_b32_e32 v1, v5, v6
	v_mad_u64_u32 v[4:5], s[0:1], v7, s6, 0
	v_mul_hi_u32 v8, v7, s7
	v_lshl_add_u64 v[4:5], v[8:9], 0, v[4:5]
	v_mad_u64_u32 v[10:11], s[0:1], v1, s7, 0
	v_add_co_u32_e32 v4, vcc, v4, v10
	v_mad_u64_u32 v[8:9], s[0:1], v1, s6, 0
	s_nop 0
	v_addc_co_u32_e32 v4, vcc, v5, v11, vcc
	v_mov_b32_e32 v5, s49
	s_nop 0
	v_addc_co_u32_e32 v9, vcc, 0, v9, vcc
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[8:9]
	v_mul_lo_u32 v10, s11, v4
	v_mul_lo_u32 v5, s10, v5
	v_mad_u64_u32 v[8:9], s[0:1], s10, v4, 0
	v_add3_u32 v5, v9, v5, v10
	v_sub_u32_e32 v9, v1, v5
	v_mov_b32_e32 v10, s11
	v_sub_co_u32_e32 v7, vcc, v7, v8
	s_add_i32 s64, s67, 0xc000
	s_nop 0
	v_subb_co_u32_e64 v8, s[6:7], v9, v10, vcc
	v_subrev_co_u32_e64 v9, s[6:7], s10, v7
	v_subb_co_u32_e32 v1, vcc, v1, v5, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v8, s[6:7], 0, v8, s[6:7]
	v_cmp_le_u32_e64 s[6:7], s11, v8
	v_cmp_le_u32_e32 vcc, s11, v1
	s_and_b32 s5, s5, 0xffff
	v_cndmask_b32_e64 v10, 0, -1, s[6:7]
	v_cmp_le_u32_e64 s[6:7], s10, v9
	v_cndmask_b32_e64 v5, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s10, v7
	v_cndmask_b32_e64 v9, 0, -1, s[6:7]
	v_cmp_eq_u32_e64 s[6:7], s11, v8
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s11, v1
	v_cndmask_b32_e64 v8, v10, v9, s[6:7]
	v_add_u32_e32 v9, 2, v4
	v_add_u32_e32 v10, 1, v4
	v_cmp_ne_u32_e64 s[6:7], 0, v8
	v_cndmask_b32_e32 v1, v5, v7, vcc
	v_cmp_ne_u32_e32 vcc, 0, v1
	v_cndmask_b32_e64 v8, v10, v9, s[6:7]
	s_mov_b32 s7, 0x27000
	v_cndmask_b32_e32 v1, v4, v8, vcc
	v_xor_b32_e32 v4, s80, v6
	v_xor_b32_e32 v1, v1, v4
	v_sub_u32_e32 v1, v1, v4
	v_xor_b32_e32 v4, v0, v1
	v_ashrrev_i32_e32 v0, 31, v3
	v_and_b32_e32 v1, s31, v0
	v_and_b32_e32 v0, s30, v0
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[2:3]
	v_ashrrev_i32_e32 v5, 31, v1
	v_xor_b32_e32 v1, v5, v1
	v_ashrrev_i32_e32 v2, 31, v1
	v_xor_b32_e32 v0, v5, v0
	v_lshrrev_b32_e32 v2, 27, v2
	v_mov_b32_e32 v3, s49
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[2:3]
	v_alignbit_b32 v0, v1, v0, 5
	v_mul_lo_u32 v1, v4, s46
	v_xad_u32 v0, v0, v5, v1
	s_mov_b32 s6, 0x7ffffffe
	s_mov_b32 m0, s64
	s_ashr_i32 s0, s15, 31
	buffer_load_dword v0, s[4:7], 0 offen lds
	s_mov_b32 s1, s0
	s_xor_b64 s[10:11], s[14:15], s[0:1]
	s_ashr_i32 s15, s11, 31
	s_lshr_b32 s24, s15, 27
	s_add_u32 s24, s10, s24
	s_addc_u32 s25, s11, 0
	s_ashr_i64 s[24:25], s[24:25], 5
	s_xor_b64 s[24:25], s[24:25], s[0:1]
	s_lshr_b32 s15, s15, 22
	s_add_u32 s10, s10, s15
	s_addc_u32 s11, s11, 0
	s_ashr_i64 s[10:11], s[10:11], 10
	s_xor_b64 s[70:71], s[10:11], s[0:1]
	s_lshl_b64 s[48:49], s[70:71], 5
	s_sub_u32 s0, s24, s48
	s_subb_u32 s1, s25, s49
	v_cmp_gt_i64_e64 s[10:11], s[0:1], 1
	s_and_b64 s[10:11], s[10:11], exec
	s_cselect_b32 s51, s1, 0
	s_cselect_b32 s50, s0, 1
	s_mul_i32 s0, s70, s39
	s_mul_hi_u32 s1, s70, s38
	s_add_i32 s0, s1, s0
	s_mul_i32 s1, s71, s38
	s_add_i32 s29, s0, s1
	s_mul_i32 s28, s70, s38
	s_lshl_b64 s[0:1], s[28:29], 5
	s_sub_u32 s24, 0, s0
	s_subb_u32 s25, 0, s1
	s_sub_u32 s10, s22, s0
	s_subb_u32 s11, s23, s1
	s_or_b64 s[0:1], s[10:11], s[50:51]
	s_cmp_lg_u32 s1, 0
	s_cbranch_scc0 .LBB0_11
	s_ashr_i32 s0, s51, 31
	s_add_u32 s52, s50, s0
	s_mov_b32 s1, s0
	s_addc_u32 s53, s51, s0
	s_xor_b64 s[72:73], s[52:53], s[0:1]
	v_cvt_f32_u32_e32 v0, s72
	v_cvt_f32_u32_e32 v1, s73
	s_sub_u32 s0, 0, s72
	s_subb_u32 s1, 0, s73
	s_mov_b64 s[52:53], 0
	v_fmac_f32_e32 v0, 0x4f800000, v1
	v_rcp_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x5f7ffffc, v0
	v_mul_f32_e32 v1, 0x2f800000, v0
	v_trunc_f32_e32 v1, v1
	v_fmac_f32_e32 v0, 0xcf800000, v1
	v_cvt_u32_f32_e32 v1, v1
	v_cvt_u32_f32_e32 v0, v0
	v_readfirstlane_b32 s15, v1
	v_readfirstlane_b32 s56, v0
	s_mul_i32 s57, s0, s15
	s_mul_hi_u32 s59, s0, s56
	s_mul_i32 s58, s1, s56
	s_add_i32 s57, s59, s57
	s_add_i32 s57, s57, s58
	s_mul_i32 s65, s0, s56
	s_mul_i32 s59, s56, s57
	s_mul_hi_u32 s69, s56, s65
	s_mul_hi_u32 s58, s56, s57
	s_add_u32 s59, s69, s59
	s_addc_u32 s58, 0, s58
	s_mul_hi_u32 s75, s15, s65
	s_mul_i32 s65, s15, s65
	s_add_u32 s59, s59, s65
	s_mul_hi_u32 s69, s15, s57
	s_addc_u32 s58, s58, s75
	s_addc_u32 s59, s69, 0
	s_mul_i32 s57, s15, s57
	s_add_u32 s57, s58, s57
	s_addc_u32 s58, 0, s59
	s_add_u32 s56, s56, s57
	s_addc_u32 s15, s15, s58
	s_mul_i32 s57, s0, s15
	s_mul_hi_u32 s58, s0, s56
	s_add_i32 s57, s58, s57
	s_mul_i32 s1, s1, s56
	s_add_i32 s57, s57, s1
	s_mul_i32 s0, s0, s56
	s_mul_hi_u32 s58, s15, s0
	s_mul_i32 s59, s15, s0
	s_mul_i32 s69, s56, s57
	s_mul_hi_u32 s0, s56, s0
	s_mul_hi_u32 s65, s56, s57
	s_add_u32 s0, s0, s69
	s_addc_u32 s65, 0, s65
	s_add_u32 s0, s0, s59
	s_mul_hi_u32 s1, s15, s57
	s_addc_u32 s0, s65, s58
	s_addc_u32 s1, s1, 0
	s_mul_i32 s57, s15, s57
	s_add_u32 s0, s0, s57
	s_addc_u32 s1, 0, s1
	s_add_u32 s58, s56, s0
	s_addc_u32 s15, s15, s1
	s_ashr_i32 s0, s11, 31
	s_add_u32 s56, s10, s0
	s_mov_b32 s1, s0
	s_addc_u32 s57, s11, s0
	s_xor_b64 s[56:57], s[56:57], s[0:1]
	s_mul_i32 s59, s56, s15
	s_mul_hi_u32 s65, s56, s58
	s_mul_hi_u32 s11, s56, s15
	s_add_u32 s59, s65, s59
	s_addc_u32 s11, 0, s11
	s_mul_hi_u32 s69, s57, s58
	s_mul_i32 s58, s57, s58
	s_add_u32 s58, s59, s58
	s_mul_hi_u32 s65, s57, s15
	s_addc_u32 s11, s11, s69
	s_addc_u32 s58, s65, 0
	s_mul_i32 s15, s57, s15
	s_add_u32 s11, s11, s15
	s_addc_u32 s15, 0, s58
	s_mul_i32 s15, s72, s15
	s_mul_hi_u32 s58, s72, s11
	s_add_i32 s15, s58, s15
	s_mul_i32 s58, s73, s11
	s_add_i32 s15, s15, s58
	s_sub_i32 s65, s57, s15
	s_mul_i32 s11, s72, s11
	s_sub_u32 s11, s56, s11
	s_cselect_b64 s[58:59], -1, 0
	s_subb_u32 s56, s65, s73
	s_sub_u32 s65, s11, s72
	s_cselect_b64 s[78:79], -1, 0
	s_subb_u32 s69, s56, 0
	s_cmp_ge_u32 s69, s73
	s_cselect_b32 s75, -1, 0
	s_cmp_ge_u32 s65, s72
	s_cselect_b32 s81, -1, 0
	s_cmp_eq_u32 s69, s73
	s_cselect_b32 s75, s81, s75
	s_cmp_lg_u64 s[78:79], 0
	s_subb_u32 s56, s56, s73
	s_sub_u32 s78, s65, s72
	s_subb_u32 s56, s56, 0
	s_cmp_lg_u32 s75, 0
	s_cselect_b32 s65, s78, s65
	s_cselect_b32 s56, s56, s69
	s_cmp_lg_u64 s[58:59], 0
	s_subb_u32 s15, s57, s15
	s_cmp_ge_u32 s15, s73
	s_cselect_b32 s57, -1, 0
	s_cmp_ge_u32 s11, s72
	s_cselect_b32 s58, -1, 0
	s_cmp_eq_u32 s15, s73
	s_cselect_b32 s57, s58, s57
	s_cmp_lg_u32 s57, 0
	s_cselect_b32 s57, s56, s15
	s_cselect_b32 s56, s65, s11
	s_xor_b64 s[56:57], s[56:57], s[0:1]
	s_sub_u32 s56, s56, s0
	s_subb_u32 s57, s57, s0
	s_branch .LBB0_12
.LBB0_10:
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v1, v0
	v_mov_b64_e32 v[6:7], v[2:3]
	v_mov_b64_e32 v[10:11], v[2:3]
	v_mov_b64_e32 v[14:15], v[2:3]
	s_mov_b64 s[40:41], s[22:23]
	s_mov_b64 s[4:5], s[24:25]
	s_mov_b64 s[36:37], s[26:27]
	s_mov_b64 s[8:9], s[28:29]
	v_mov_b64_e32 v[4:5], v[0:1]
	v_mov_b64_e32 v[8:9], v[0:1]
	v_mov_b64_e32 v[12:13], v[0:1]
	s_branch .LBB0_106
.LBB0_11:
	s_mov_b64 s[52:53], -1
.LBB0_12:
	s_andn2_b64 vcc, exec, s[52:53]
	v_writelane_b32 v144, s82, 26
	s_cbranch_vccnz .LBB0_14
	v_cvt_f32_u32_e32 v0, s50
	s_sub_i32 s0, 0, s50
	s_mov_b32 s57, 0
	v_rcp_iflag_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_nop 0
	v_readfirstlane_b32 s1, v0
	s_mul_i32 s0, s0, s1
	s_mul_hi_u32 s0, s1, s0
	s_add_i32 s1, s1, s0
	s_mul_hi_u32 s0, s10, s1
	s_mul_i32 s0, s0, s50
	s_sub_i32 s0, s10, s0
	s_sub_i32 s1, s0, s50
	s_cmp_ge_u32 s0, s50
	s_cselect_b32 s0, s1, s0
	s_sub_i32 s1, s0, s50
	s_cmp_ge_u32 s0, s50
	s_cselect_b32 s56, s1, s0
.LBB0_14:
	s_and_b32 s0, s14, 0x3ff
	v_writelane_b32 v144, s0, 27
	s_mul_i32 s0, s48, s39
	s_mul_hi_u32 s1, s48, s38
	s_cselect_b64 s[52:53], -1, 0
	s_add_i32 s0, s1, s0
	s_mul_i32 s1, s49, s38
	s_add_i32 s0, s0, s1
	s_mul_i32 s1, s48, s38
	v_mov_b32_e32 v0, s1
	v_mov_b32_e32 v1, s0
	s_ashr_i32 s0, s57, 31
	v_cmp_ge_i64_e32 vcc, s[22:23], v[0:1]
	s_mov_b32 s1, s0
	s_and_b64 s[10:11], s[52:53], vcc
	s_and_b64 s[0:1], s[0:1], s[50:51]
	s_add_u32 s14, s56, s48
	s_addc_u32 s15, s57, s49
	s_add_u32 s65, s14, s0
	s_mov_b32 s69, s68
	s_addc_u32 s23, s15, s1
	s_xor_b64 s[0:1], s[68:69], s[62:63]
	s_ashr_i32 s14, s39, 31
	s_add_u32 s38, s38, s14
	s_mov_b32 s15, s14
	s_addc_u32 s39, s39, s14
	s_xor_b64 s[38:39], s[38:39], s[14:15]
	v_cvt_f32_u32_e32 v0, s38
	v_cvt_f32_u32_e32 v1, s39
	s_sub_u32 s56, 0, s38
	s_subb_u32 s57, 0, s39
	s_mov_b32 s63, 0
	v_fmac_f32_e32 v0, 0x4f800000, v1
	v_rcp_f32_e32 v0, v0
	v_lshlrev_b32_e32 v5, 4, v96
	v_lshl_add_u32 v44, v98, 8, v5
	v_mul_f32_e32 v0, 0x5f7ffffc, v0
	v_mul_f32_e32 v1, 0x2f800000, v0
	v_trunc_f32_e32 v1, v1
	v_fmac_f32_e32 v0, 0xcf800000, v1
	v_cvt_u32_f32_e32 v0, v0
	v_cvt_u32_f32_e32 v1, v1
	v_readfirstlane_b32 s58, v0
	v_readfirstlane_b32 s72, v1
	s_mul_hi_u32 s62, s56, s58
	s_mul_i32 s73, s56, s72
	s_mul_i32 s59, s57, s58
	s_add_i32 s62, s62, s73
	s_add_i32 s62, s62, s59
	s_mul_i32 s75, s56, s58
	s_mul_i32 s73, s58, s62
	s_mul_hi_u32 s78, s58, s75
	s_mul_hi_u32 s59, s58, s62
	s_add_u32 s73, s78, s73
	s_addc_u32 s59, 0, s59
	s_mul_hi_u32 s79, s72, s75
	s_mul_i32 s75, s72, s75
	s_add_u32 s73, s73, s75
	s_mul_hi_u32 s78, s72, s62
	s_addc_u32 s59, s59, s79
	s_addc_u32 s73, s78, 0
	s_mul_i32 s62, s72, s62
	s_add_u32 s59, s59, s62
	s_addc_u32 s62, 0, s73
	s_add_u32 s58, s58, s59
	s_addc_u32 s59, s72, s62
	s_mul_i32 s62, s56, s59
	s_mul_hi_u32 s72, s56, s58
	s_add_i32 s62, s72, s62
	s_mul_i32 s57, s57, s58
	s_add_i32 s62, s62, s57
	s_mul_i32 s56, s56, s58
	s_mul_hi_u32 s72, s59, s56
	s_mul_i32 s73, s59, s56
	s_mul_i32 s78, s58, s62
	s_mul_hi_u32 s56, s58, s56
	s_mul_hi_u32 s75, s58, s62
	s_add_u32 s56, s56, s78
	s_addc_u32 s75, 0, s75
	s_add_u32 s56, s56, s73
	s_mul_hi_u32 s57, s59, s62
	s_addc_u32 s56, s75, s72
	s_addc_u32 s57, s57, 0
	s_mul_i32 s62, s59, s62
	s_add_u32 s56, s56, s62
	s_addc_u32 s57, 0, s57
	s_add_u32 s58, s58, s56
	s_addc_u32 s59, s59, s57
	s_ashr_i32 s56, s1, 31
	s_add_u32 s0, s0, s56
	s_mov_b32 s57, s56
	s_addc_u32 s1, s1, s56
	s_xor_b64 s[0:1], s[0:1], s[56:57]
	s_mul_i32 s72, s0, s59
	s_mul_hi_u32 s73, s0, s58
	s_mul_hi_u32 s62, s0, s59
	s_add_u32 s72, s73, s72
	s_addc_u32 s62, 0, s62
	s_mul_hi_u32 s75, s1, s58
	s_mul_i32 s58, s1, s58
	s_add_u32 s58, s72, s58
	s_mul_hi_u32 s73, s1, s59
	s_addc_u32 s58, s62, s75
	s_addc_u32 s62, s73, 0
	s_mul_i32 s59, s1, s59
	s_add_u32 s72, s58, s59
	s_addc_u32 s62, 0, s62
	s_mul_i32 s58, s38, s62
	s_mul_hi_u32 s59, s38, s72
	s_add_i32 s58, s59, s58
	s_mul_i32 s59, s39, s72
	s_add_i32 s73, s58, s59
	s_sub_i32 s75, s1, s73
	s_mul_i32 s58, s38, s72
	s_sub_u32 s0, s0, s58
	s_cselect_b64 s[58:59], -1, 0
	s_subb_u32 s75, s75, s39
	s_sub_u32 s78, s0, s38
	s_subb_u32 s75, s75, 0
	s_cmp_ge_u32 s75, s39
	s_cselect_b32 s79, -1, 0
	s_cmp_ge_u32 s78, s38
	s_cselect_b32 s78, -1, 0
	s_cmp_eq_u32 s75, s39
	s_cselect_b32 s75, s78, s79
	s_add_u32 s78, s72, 1
	s_addc_u32 s79, s62, 0
	s_add_u32 s81, s72, 2
	s_addc_u32 s82, s62, 0
	s_cmp_lg_u32 s75, 0
	s_cselect_b32 s75, s81, s78
	s_cselect_b32 s78, s82, s79
	s_cmp_lg_u64 s[58:59], 0
	s_subb_u32 s1, s1, s73
	s_cmp_ge_u32 s1, s39
	s_cselect_b32 s58, -1, 0
	s_cmp_ge_u32 s0, s38
	s_cselect_b32 s0, -1, 0
	s_cmp_eq_u32 s1, s39
	s_cselect_b32 s0, s0, s58
	s_cmp_lg_u32 s0, 0
	s_cselect_b32 s1, s78, s62
	s_cselect_b32 s0, s75, s72
	s_xor_b64 s[14:15], s[56:57], s[14:15]
	s_xor_b64 s[0:1], s[0:1], s[14:15]
	s_sub_u32 s0, s0, s14
	s_subb_u32 s1, s1, s15
	s_xor_b64 s[0:1], s[0:1], s[68:69]
	s_lshl_b64 s[0:1], s[0:1], 5
	s_and_b32 s62, s22, 31
	s_or_b64 s[0:1], s[0:1], s[62:63]
	s_and_b64 s[10:11], s[10:11], exec
	s_cselect_b32 s23, s23, s1
	s_cselect_b32 s22, s65, s0
	s_movk_i32 s0, 0xff00
	s_ashr_i32 s72, s45, 31
	v_mad_i32_i24 v2, v45, s0, v44
	s_add_u32 s0, s44, s72
	s_mov_b32 s73, s72
	s_addc_u32 s1, s45, s72
	s_xor_b64 s[78:79], s[0:1], s[72:73]
	v_cvt_f32_u32_e32 v4, s78
	v_cvt_f32_u32_e32 v8, s79
	v_not_b32_e32 v0, v2
	v_ashrrev_i32_e32 v3, 31, v2
	v_ashrrev_i32_e32 v1, 31, v0
	v_fmac_f32_e32 v4, 0x4f800000, v8
	v_cmp_gt_i32_e64 s[10:11], 0, v2
	v_rcp_f32_e32 v4, v4
	s_sub_u32 s0, 0, s78
	v_cndmask_b32_e64 v7, v3, v1, s[10:11]
	v_cndmask_b32_e64 v6, v2, v0, s[10:11]
	v_mov_b32_e32 v0, v7
	v_mov_b32_e32 v1, v7
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[0:1]
	v_xor_b32_e32 v14, v0, v7
	v_mul_f32_e32 v0, 0x5f7ffffc, v4
	v_xor_b32_e32 v6, v1, v7
	v_mul_f32_e32 v1, 0x2f800000, v0
	v_trunc_f32_e32 v1, v1
	v_fmac_f32_e32 v0, 0xcf800000, v1
	v_cvt_u32_f32_e32 v0, v0
	v_cvt_u32_f32_e32 v1, v1
	s_subb_u32 s1, 0, s79
	v_readfirstlane_b32 s14, v0
	v_readfirstlane_b32 s39, v1
	s_mul_hi_u32 s38, s0, s14
	s_mul_i32 s56, s0, s39
	s_mul_i32 s15, s1, s14
	s_add_i32 s38, s38, s56
	s_add_i32 s38, s38, s15
	s_mul_i32 s57, s0, s14
	s_mul_i32 s56, s14, s38
	s_mul_hi_u32 s58, s14, s57
	s_mul_hi_u32 s15, s14, s38
	s_add_u32 s56, s58, s56
	s_addc_u32 s15, 0, s15
	s_mul_hi_u32 s59, s39, s57
	s_mul_i32 s57, s39, s57
	s_add_u32 s56, s56, s57
	s_mul_hi_u32 s58, s39, s38
	s_addc_u32 s15, s15, s59
	s_addc_u32 s56, s58, 0
	s_mul_i32 s38, s39, s38
	s_add_u32 s15, s15, s38
	s_addc_u32 s38, 0, s56
	s_add_u32 s14, s14, s15
	s_addc_u32 s15, s39, s38
	s_mul_i32 s38, s0, s15
	s_mul_hi_u32 s39, s0, s14
	s_add_i32 s38, s39, s38
	s_mul_i32 s1, s1, s14
	s_add_i32 s38, s38, s1
	s_mul_i32 s0, s0, s14
	s_mul_hi_u32 s39, s15, s0
	s_mul_i32 s56, s15, s0
	s_mul_i32 s58, s14, s38
	s_mul_hi_u32 s0, s14, s0
	s_mul_hi_u32 s57, s14, s38
	s_add_u32 s0, s0, s58
	s_addc_u32 s57, 0, s57
	s_add_u32 s0, s0, s56
	s_mul_hi_u32 s1, s15, s38
	s_addc_u32 s0, s57, s39
	s_addc_u32 s1, s1, 0
	s_mul_i32 s38, s15, s38
	s_add_u32 s0, s0, s38
	s_addc_u32 s1, 0, s1
	s_add_u32 s0, s14, s0
	s_addc_u32 s1, s15, s1
	v_mad_u64_u32 v[8:9], s[14:15], v14, s1, 0
	v_mul_hi_u32 v0, v14, s0
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[8:9], v[0:1], 0, v[8:9]
	v_mad_u64_u32 v[12:13], s[14:15], v6, s0, 0
	v_add_co_u32_e32 v0, vcc, v8, v12
	v_mad_u64_u32 v[10:11], s[14:15], v6, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v8, vcc, v9, v13, vcc
	v_mov_b32_e32 v9, s63
	s_nop 0
	v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[10:11]
	v_mul_lo_u32 v0, s79, v8
	v_mul_lo_u32 v4, s78, v9
	v_mad_u64_u32 v[10:11], s[14:15], s78, v8, 0
	v_add3_u32 v0, v11, v4, v0
	v_sub_u32_e32 v9, v6, v0
	v_mov_b32_e32 v4, s79
	v_sub_co_u32_e32 v10, vcc, v14, v10
	s_lshl_b64 s[68:69], s[22:23], 5
	s_nop 0
	v_subb_co_u32_e64 v9, s[14:15], v9, v4, vcc
	v_subrev_co_u32_e64 v11, s[14:15], s78, v10
	v_subb_co_u32_e32 v0, vcc, v6, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v9, s[14:15], 0, v9, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v9
	v_cmp_le_u32_e32 vcc, s79, v0
	v_or_b32_e32 v37, s68, v97
	v_cndmask_b32_e64 v12, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v11
	v_cndmask_b32_e64 v6, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v10
	v_cndmask_b32_e64 v11, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v9
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v9, v12, v11, s[14:15]
	v_add_u32_e32 v11, 2, v8
	v_add_u32_e32 v12, 1, v8
	v_cmp_ne_u32_e64 s[14:15], 0, v9
	v_cndmask_b32_e32 v0, v6, v10, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v9, v12, v11, s[14:15]
	v_xor_b32_e32 v6, s72, v7
	v_cndmask_b32_e32 v0, v8, v9, vcc
	v_xor_b32_e32 v0, v0, v6
	v_sub_u32_e32 v0, v0, v6
	v_mov_b32_e32 v6, v3
	v_mov_b32_e32 v7, v3
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[6:7]
	v_xor_b32_e32 v14, v6, v3
	v_xad_u32 v12, v0, v3, v37
	v_xor_b32_e32 v13, v7, v3
	v_mad_u64_u32 v[6:7], s[14:15], v14, s1, 0
	v_mul_hi_u32 v0, v14, s0
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[6:7]
	v_mad_u64_u32 v[10:11], s[14:15], v13, s0, 0
	v_add_co_u32_e32 v0, vcc, v6, v10
	v_mad_u64_u32 v[8:9], s[14:15], v13, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v6, vcc, v7, v11, vcc
	v_mov_b32_e32 v7, s63
	s_nop 0
	v_addc_co_u32_e32 v9, vcc, 0, v9, vcc
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[8:9]
	v_mul_lo_u32 v0, s79, v6
	v_mul_lo_u32 v8, s78, v7
	v_mad_u64_u32 v[6:7], s[14:15], s78, v6, 0
	v_add3_u32 v0, v7, v8, v0
	v_sub_u32_e32 v7, v13, v0
	v_sub_co_u32_e32 v6, vcc, v14, v6
	s_mov_b32 s39, 0x27000
	s_nop 0
	v_subb_co_u32_e64 v7, s[14:15], v7, v4, vcc
	v_subrev_co_u32_e64 v8, s[14:15], s78, v6
	v_subb_co_u32_e32 v0, vcc, v13, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v9, s[22:23], 0, v7, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v9
	v_subb_co_u32_e64 v7, s[14:15], v7, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v10, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v8
	v_cmp_le_u32_e32 vcc, s79, v0
	s_mov_b32 s38, 0x7ffffffe
	v_cndmask_b32_e64 v11, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v9
	s_and_b32 s37, s37, 0xffff
	s_nop 0
	v_cndmask_b32_e64 v10, v10, v11, s[22:23]
	v_subrev_co_u32_e64 v11, s[14:15], s78, v8
	s_nop 1
	v_subbrev_co_u32_e64 v7, s[14:15], 0, v7, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v10
	s_nop 1
	v_cndmask_b32_e64 v7, v9, v7, s[14:15]
	v_cndmask_b32_e64 v9, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v6
	s_nop 1
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v9, v9, v10, vcc
	v_cmp_ne_u32_e32 vcc, 0, v9
	s_nop 1
	v_cndmask_b32_e32 v0, v0, v7, vcc
	v_cndmask_b32_e64 v7, v8, v11, s[14:15]
	v_cndmask_b32_e32 v6, v6, v7, vcc
	v_xor_b32_e32 v6, v6, v3
	v_xor_b32_e32 v0, v0, v3
	v_sub_co_u32_e32 v6, vcc, v6, v3
	v_mul_lo_u32 v7, v12, s33
	s_nop 0
	v_subb_co_u32_e32 v0, vcc, v0, v3, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_add3_u32 v6, v0, v6, v7
	v_or_b32_e32 v8, 1, v2
	v_sub_u32_e32 v0, -2, v2
	v_ashrrev_i32_e32 v9, 31, v8
	v_ashrrev_i32_e32 v7, 31, v0
	v_cndmask_b32_e64 v11, v9, v7, s[10:11]
	v_cndmask_b32_e64 v10, v8, v0, s[10:11]
	v_mov_b32_e32 v12, v11
	v_mov_b32_e32 v13, v11
	v_lshl_add_u64 v[12:13], v[10:11], 0, v[12:13]
	v_xor_b32_e32 v10, v12, v11
	v_xor_b32_e32 v7, v13, v11
	v_mad_u64_u32 v[12:13], s[14:15], v10, s1, 0
	v_mul_hi_u32 v0, v10, s0
	v_lshl_add_u64 v[12:13], v[0:1], 0, v[12:13]
	v_mad_u64_u32 v[16:17], s[14:15], v7, s0, 0
	v_add_co_u32_e32 v0, vcc, v12, v16
	v_mad_u64_u32 v[14:15], s[14:15], v7, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v13, v17, vcc
	v_mov_b32_e32 v13, s63
	s_nop 0
	v_addc_co_u32_e32 v15, vcc, 0, v15, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[14:15]
	v_mul_lo_u32 v0, s79, v12
	v_mul_lo_u32 v13, s78, v13
	v_mad_u64_u32 v[14:15], s[14:15], s78, v12, 0
	v_add3_u32 v0, v15, v13, v0
	v_sub_u32_e32 v13, v7, v0
	v_sub_co_u32_e32 v10, vcc, v10, v14
	s_nop 1
	v_subb_co_u32_e64 v13, s[14:15], v13, v4, vcc
	v_subrev_co_u32_e64 v14, s[14:15], s78, v10
	v_subb_co_u32_e32 v0, vcc, v7, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v13, s[14:15], 0, v13, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v13
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v15, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v14
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v10
	v_cndmask_b32_e64 v14, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v13
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v13, v15, v14, s[14:15]
	v_add_u32_e32 v14, 2, v12
	v_add_u32_e32 v15, 1, v12
	v_cmp_ne_u32_e64 s[14:15], 0, v13
	v_cndmask_b32_e32 v0, v7, v10, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v13, v15, v14, s[14:15]
	v_xor_b32_e32 v7, s72, v11
	v_cndmask_b32_e32 v0, v12, v13, vcc
	v_mov_b32_e32 v10, v9
	v_mov_b32_e32 v11, v9
	v_xor_b32_e32 v0, v0, v7
	v_lshl_add_u64 v[10:11], v[8:9], 0, v[10:11]
	v_sub_u32_e32 v0, v0, v7
	v_xor_b32_e32 v16, v10, v9
	v_xad_u32 v7, v0, v3, v37
	v_xor_b32_e32 v8, v11, v9
	v_mad_u64_u32 v[10:11], s[14:15], v16, s1, 0
	v_mul_hi_u32 v0, v16, s0
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[10:11]
	v_mad_u64_u32 v[14:15], s[14:15], v8, s0, 0
	v_add_co_u32_e32 v0, vcc, v10, v14
	v_mad_u64_u32 v[12:13], s[14:15], v8, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v15, vcc
	v_mov_b32_e32 v11, s63
	s_nop 0
	v_addc_co_u32_e32 v13, vcc, 0, v13, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[12:13]
	v_mul_lo_u32 v0, s79, v10
	v_mul_lo_u32 v12, s78, v11
	v_mad_u64_u32 v[10:11], s[14:15], s78, v10, 0
	v_add3_u32 v0, v11, v12, v0
	v_sub_u32_e32 v11, v8, v0
	v_sub_co_u32_e32 v10, vcc, v16, v10
	v_mul_lo_u32 v7, v7, s33
	s_nop 0
	v_subb_co_u32_e64 v11, s[14:15], v11, v4, vcc
	v_subrev_co_u32_e64 v12, s[14:15], s78, v10
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v13, s[22:23], 0, v11, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v13
	v_subb_co_u32_e64 v11, s[14:15], v11, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v14, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v12
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v15, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v13
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v10
	v_cndmask_b32_e64 v14, v14, v15, s[22:23]
	v_subrev_co_u32_e64 v15, s[14:15], s78, v12
	s_nop 1
	v_subbrev_co_u32_e64 v11, s[14:15], 0, v11, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v14
	s_nop 1
	v_cndmask_b32_e64 v11, v13, v11, s[14:15]
	v_cndmask_b32_e64 v13, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v8, v8, v13, vcc
	v_cmp_ne_u32_e32 vcc, 0, v8
	v_cndmask_b32_e64 v8, v12, v15, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v8, v10, v8, vcc
	v_cndmask_b32_e32 v0, v0, v11, vcc
	v_xor_b32_e32 v8, v8, v9
	v_xor_b32_e32 v0, v0, v9
	v_sub_co_u32_e32 v8, vcc, v8, v9
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v9, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_add3_u32 v7, v0, v8, v7
	v_or_b32_e32 v8, 2, v2
	v_sub_u32_e32 v0, -3, v2
	v_ashrrev_i32_e32 v9, 31, v8
	v_ashrrev_i32_e32 v10, 31, v0
	v_cndmask_b32_e64 v11, v9, v10, s[10:11]
	v_cndmask_b32_e64 v10, v8, v0, s[10:11]
	v_mov_b32_e32 v12, v11
	v_mov_b32_e32 v13, v11
	v_lshl_add_u64 v[12:13], v[10:11], 0, v[12:13]
	v_xor_b32_e32 v18, v12, v11
	v_xor_b32_e32 v10, v13, v11
	v_mad_u64_u32 v[12:13], s[14:15], v18, s1, 0
	v_mul_hi_u32 v0, v18, s0
	v_lshl_add_u64 v[12:13], v[0:1], 0, v[12:13]
	v_mad_u64_u32 v[16:17], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v12, v16
	v_mad_u64_u32 v[14:15], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v13, v17, vcc
	v_mov_b32_e32 v13, s63
	s_nop 0
	v_addc_co_u32_e32 v15, vcc, 0, v15, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[14:15]
	v_mul_lo_u32 v0, s79, v12
	v_mul_lo_u32 v13, s78, v13
	v_mad_u64_u32 v[14:15], s[14:15], s78, v12, 0
	v_add3_u32 v0, v15, v13, v0
	v_sub_u32_e32 v13, v10, v0
	v_sub_co_u32_e32 v14, vcc, v18, v14
	s_nop 1
	v_subb_co_u32_e64 v13, s[14:15], v13, v4, vcc
	v_subrev_co_u32_e64 v15, s[14:15], s78, v14
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v13, s[14:15], 0, v13, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v13
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v16, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v15
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v14
	v_cndmask_b32_e64 v15, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v13
	v_cndmask_b32_e64 v14, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v13, v16, v15, s[14:15]
	v_add_u32_e32 v15, 2, v12
	v_add_u32_e32 v16, 1, v12
	v_cmp_ne_u32_e64 s[14:15], 0, v13
	v_cndmask_b32_e32 v0, v10, v14, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v13, v16, v15, s[14:15]
	v_xor_b32_e32 v10, s72, v11
	v_cndmask_b32_e32 v0, v12, v13, vcc
	v_xor_b32_e32 v0, v0, v10
	v_sub_u32_e32 v0, v0, v10
	v_mov_b32_e32 v10, v9
	v_mov_b32_e32 v11, v9
	v_lshl_add_u64 v[10:11], v[8:9], 0, v[10:11]
	v_xor_b32_e32 v17, v10, v9
	v_xad_u32 v16, v0, v3, v37
	v_xor_b32_e32 v8, v11, v9
	v_mad_u64_u32 v[10:11], s[14:15], v17, s1, 0
	v_mul_hi_u32 v0, v17, s0
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[10:11]
	v_mad_u64_u32 v[14:15], s[14:15], v8, s0, 0
	v_add_co_u32_e32 v0, vcc, v10, v14
	v_mad_u64_u32 v[12:13], s[14:15], v8, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v15, vcc
	v_mov_b32_e32 v11, s63
	s_nop 0
	v_addc_co_u32_e32 v13, vcc, 0, v13, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[12:13]
	v_mul_lo_u32 v0, s79, v10
	v_mul_lo_u32 v12, s78, v11
	v_mad_u64_u32 v[10:11], s[14:15], s78, v10, 0
	v_add3_u32 v0, v11, v12, v0
	v_sub_u32_e32 v11, v8, v0
	v_sub_co_u32_e32 v10, vcc, v17, v10
	s_nop 1
	v_subb_co_u32_e64 v11, s[14:15], v11, v4, vcc
	v_subrev_co_u32_e64 v12, s[14:15], s78, v10
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v13, s[22:23], 0, v11, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v13
	v_subb_co_u32_e64 v11, s[14:15], v11, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v14, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v12
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v15, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v13
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v10
	v_cndmask_b32_e64 v14, v14, v15, s[22:23]
	v_subrev_co_u32_e64 v15, s[14:15], s78, v12
	s_nop 1
	v_subbrev_co_u32_e64 v11, s[14:15], 0, v11, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v14
	s_nop 1
	v_cndmask_b32_e64 v11, v13, v11, s[14:15]
	v_cndmask_b32_e64 v13, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v8, v8, v13, vcc
	v_cmp_ne_u32_e32 vcc, 0, v8
	v_cndmask_b32_e64 v8, v12, v15, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v8, v10, v8, vcc
	v_cndmask_b32_e32 v0, v0, v11, vcc
	v_xor_b32_e32 v8, v8, v9
	v_xor_b32_e32 v0, v0, v9
	v_sub_co_u32_e32 v8, vcc, v8, v9
	v_or_b32_e32 v10, 3, v2
	s_nop 0
	v_subb_co_u32_e32 v0, vcc, v0, v9, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v9, v16, s33
	v_add3_u32 v8, v0, v8, v9
	v_sub_u32_e32 v0, -4, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v9, 31, v0
	v_cndmask_b32_e64 v13, v11, v9, s[10:11]
	v_cndmask_b32_e64 v12, v10, v0, s[10:11]
	v_mov_b32_e32 v14, v13
	v_mov_b32_e32 v15, v13
	v_lshl_add_u64 v[14:15], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v12, v14, v13
	v_xor_b32_e32 v9, v15, v13
	v_mad_u64_u32 v[14:15], s[14:15], v12, s1, 0
	v_mul_hi_u32 v0, v12, s0
	v_lshl_add_u64 v[14:15], v[0:1], 0, v[14:15]
	v_mad_u64_u32 v[18:19], s[14:15], v9, s0, 0
	v_add_co_u32_e32 v0, vcc, v14, v18
	v_mad_u64_u32 v[16:17], s[14:15], v9, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v14, vcc, v15, v19, vcc
	v_mov_b32_e32 v15, s63
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[16:17]
	v_mul_lo_u32 v0, s79, v14
	v_mul_lo_u32 v15, s78, v15
	v_mad_u64_u32 v[16:17], s[14:15], s78, v14, 0
	v_add3_u32 v0, v17, v15, v0
	v_sub_u32_e32 v15, v9, v0
	v_sub_co_u32_e32 v12, vcc, v12, v16
	s_nop 1
	v_subb_co_u32_e64 v15, s[14:15], v15, v4, vcc
	v_subrev_co_u32_e64 v16, s[14:15], s78, v12
	v_subb_co_u32_e32 v0, vcc, v9, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v15, s[14:15], 0, v15, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v15
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v16
	v_cndmask_b32_e64 v9, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v12
	v_cndmask_b32_e64 v16, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v15
	v_cndmask_b32_e64 v12, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v15, v17, v16, s[14:15]
	v_add_u32_e32 v16, 2, v14
	v_add_u32_e32 v17, 1, v14
	v_cmp_ne_u32_e64 s[14:15], 0, v15
	v_cndmask_b32_e32 v0, v9, v12, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v15, v17, v16, s[14:15]
	v_xor_b32_e32 v9, s72, v13
	v_cndmask_b32_e32 v0, v14, v15, vcc
	v_mov_b32_e32 v12, v11
	v_mov_b32_e32 v13, v11
	v_xor_b32_e32 v0, v0, v9
	v_lshl_add_u64 v[12:13], v[10:11], 0, v[12:13]
	v_sub_u32_e32 v0, v0, v9
	v_xor_b32_e32 v18, v12, v11
	v_xad_u32 v9, v0, v3, v37
	v_xor_b32_e32 v10, v13, v11
	v_mad_u64_u32 v[12:13], s[14:15], v18, s1, 0
	v_mul_hi_u32 v0, v18, s0
	v_lshl_add_u64 v[12:13], v[0:1], 0, v[12:13]
	v_mad_u64_u32 v[16:17], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v12, v16
	v_mad_u64_u32 v[14:15], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v13, v17, vcc
	v_mov_b32_e32 v13, s63
	s_nop 0
	v_addc_co_u32_e32 v15, vcc, 0, v15, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[14:15]
	v_mul_lo_u32 v0, s79, v12
	v_mul_lo_u32 v14, s78, v13
	v_mad_u64_u32 v[12:13], s[14:15], s78, v12, 0
	v_add3_u32 v0, v13, v14, v0
	v_sub_u32_e32 v13, v10, v0
	v_sub_co_u32_e32 v12, vcc, v18, v12
	v_mul_lo_u32 v9, v9, s33
	s_nop 0
	v_subb_co_u32_e64 v13, s[14:15], v13, v4, vcc
	v_subrev_co_u32_e64 v14, s[14:15], s78, v12
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v15, s[22:23], 0, v13, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v15
	v_subb_co_u32_e64 v13, s[14:15], v13, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v16, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v14
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v15
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v12
	v_cndmask_b32_e64 v16, v16, v17, s[22:23]
	v_subrev_co_u32_e64 v17, s[14:15], s78, v14
	s_nop 1
	v_subbrev_co_u32_e64 v13, s[14:15], 0, v13, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v16
	s_nop 1
	v_cndmask_b32_e64 v13, v15, v13, s[14:15]
	v_cndmask_b32_e64 v15, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v15, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v14, v17, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v10, v12, v10, vcc
	v_cndmask_b32_e32 v0, v0, v13, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_add3_u32 v9, v0, v10, v9
	v_or_b32_e32 v10, 4, v2
	v_sub_u32_e32 v0, -5, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v12, 31, v0
	v_cndmask_b32_e64 v13, v11, v12, s[10:11]
	v_cndmask_b32_e64 v12, v10, v0, s[10:11]
	v_mov_b32_e32 v14, v13
	v_mov_b32_e32 v15, v13
	v_lshl_add_u64 v[14:15], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v20, v14, v13
	v_xor_b32_e32 v12, v15, v13
	v_mad_u64_u32 v[14:15], s[14:15], v20, s1, 0
	v_mul_hi_u32 v0, v20, s0
	v_lshl_add_u64 v[14:15], v[0:1], 0, v[14:15]
	v_mad_u64_u32 v[18:19], s[14:15], v12, s0, 0
	v_add_co_u32_e32 v0, vcc, v14, v18
	v_mad_u64_u32 v[16:17], s[14:15], v12, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v14, vcc, v15, v19, vcc
	v_mov_b32_e32 v15, s63
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[16:17]
	v_mul_lo_u32 v0, s79, v14
	v_mul_lo_u32 v15, s78, v15
	v_mad_u64_u32 v[16:17], s[14:15], s78, v14, 0
	v_add3_u32 v0, v17, v15, v0
	v_sub_u32_e32 v15, v12, v0
	v_sub_co_u32_e32 v16, vcc, v20, v16
	s_nop 1
	v_subb_co_u32_e64 v15, s[14:15], v15, v4, vcc
	v_subrev_co_u32_e64 v17, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v12, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v15, s[14:15], 0, v15, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v15
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v18, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v17
	v_cndmask_b32_e64 v12, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v17, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v15
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v15, v18, v17, s[14:15]
	v_add_u32_e32 v17, 2, v14
	v_add_u32_e32 v18, 1, v14
	v_cmp_ne_u32_e64 s[14:15], 0, v15
	v_cndmask_b32_e32 v0, v12, v16, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v15, v18, v17, s[14:15]
	v_xor_b32_e32 v12, s72, v13
	v_cndmask_b32_e32 v0, v14, v15, vcc
	v_xor_b32_e32 v0, v0, v12
	v_sub_u32_e32 v0, v0, v12
	v_mov_b32_e32 v12, v11
	v_mov_b32_e32 v13, v11
	v_lshl_add_u64 v[12:13], v[10:11], 0, v[12:13]
	v_xor_b32_e32 v19, v12, v11
	v_xad_u32 v18, v0, v3, v37
	v_xor_b32_e32 v10, v13, v11
	v_mad_u64_u32 v[12:13], s[14:15], v19, s1, 0
	v_mul_hi_u32 v0, v19, s0
	v_lshl_add_u64 v[12:13], v[0:1], 0, v[12:13]
	v_mad_u64_u32 v[16:17], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v12, v16
	v_mad_u64_u32 v[14:15], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v13, v17, vcc
	v_mov_b32_e32 v13, s63
	s_nop 0
	v_addc_co_u32_e32 v15, vcc, 0, v15, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[14:15]
	v_mul_lo_u32 v0, s79, v12
	v_mul_lo_u32 v14, s78, v13
	v_mad_u64_u32 v[12:13], s[14:15], s78, v12, 0
	v_add3_u32 v0, v13, v14, v0
	v_sub_u32_e32 v13, v10, v0
	v_sub_co_u32_e32 v12, vcc, v19, v12
	s_nop 1
	v_subb_co_u32_e64 v13, s[14:15], v13, v4, vcc
	v_subrev_co_u32_e64 v14, s[14:15], s78, v12
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v15, s[22:23], 0, v13, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v15
	v_subb_co_u32_e64 v13, s[14:15], v13, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v16, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v14
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v15
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v12
	v_cndmask_b32_e64 v16, v16, v17, s[22:23]
	v_subrev_co_u32_e64 v17, s[14:15], s78, v14
	s_nop 1
	v_subbrev_co_u32_e64 v13, s[14:15], 0, v13, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v16
	s_nop 1
	v_cndmask_b32_e64 v13, v15, v13, s[14:15]
	v_cndmask_b32_e64 v15, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v15, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v14, v17, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v10, v12, v10, vcc
	v_cndmask_b32_e32 v0, v0, v13, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v11, v18, s33
	v_add3_u32 v20, v0, v10, v11
	v_or_b32_e32 v10, 5, v2
	v_sub_u32_e32 v0, -6, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v12, 31, v0
	v_cndmask_b32_e64 v13, v11, v12, s[10:11]
	v_cndmask_b32_e64 v12, v10, v0, s[10:11]
	v_mov_b32_e32 v14, v13
	v_mov_b32_e32 v15, v13
	v_lshl_add_u64 v[14:15], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v21, v14, v13
	v_xor_b32_e32 v12, v15, v13
	v_mad_u64_u32 v[14:15], s[14:15], v21, s1, 0
	v_mul_hi_u32 v0, v21, s0
	v_lshl_add_u64 v[14:15], v[0:1], 0, v[14:15]
	v_mad_u64_u32 v[18:19], s[14:15], v12, s0, 0
	v_add_co_u32_e32 v0, vcc, v14, v18
	v_mad_u64_u32 v[16:17], s[14:15], v12, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v14, vcc, v15, v19, vcc
	v_mov_b32_e32 v15, s63
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[16:17]
	v_mul_lo_u32 v0, s79, v14
	v_mul_lo_u32 v15, s78, v15
	v_mad_u64_u32 v[16:17], s[14:15], s78, v14, 0
	v_add3_u32 v0, v17, v15, v0
	v_sub_u32_e32 v15, v12, v0
	v_sub_co_u32_e32 v16, vcc, v21, v16
	s_nop 1
	v_subb_co_u32_e64 v15, s[14:15], v15, v4, vcc
	v_subrev_co_u32_e64 v17, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v12, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v15, s[14:15], 0, v15, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v15
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v18, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v17
	v_cndmask_b32_e64 v12, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v17, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v15
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v15, v18, v17, s[14:15]
	v_add_u32_e32 v17, 2, v14
	v_add_u32_e32 v18, 1, v14
	v_cmp_ne_u32_e64 s[14:15], 0, v15
	v_cndmask_b32_e32 v0, v12, v16, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v15, v18, v17, s[14:15]
	v_xor_b32_e32 v12, s72, v13
	v_cndmask_b32_e32 v0, v14, v15, vcc
	v_xor_b32_e32 v0, v0, v12
	v_sub_u32_e32 v0, v0, v12
	v_mov_b32_e32 v12, v11
	v_mov_b32_e32 v13, v11
	v_lshl_add_u64 v[12:13], v[10:11], 0, v[12:13]
	v_xor_b32_e32 v19, v12, v11
	v_xad_u32 v18, v0, v3, v37
	v_xor_b32_e32 v10, v13, v11
	v_mad_u64_u32 v[12:13], s[14:15], v19, s1, 0
	v_mul_hi_u32 v0, v19, s0
	v_lshl_add_u64 v[12:13], v[0:1], 0, v[12:13]
	v_mad_u64_u32 v[16:17], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v12, v16
	v_mad_u64_u32 v[14:15], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v13, v17, vcc
	v_mov_b32_e32 v13, s63
	s_nop 0
	v_addc_co_u32_e32 v15, vcc, 0, v15, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[14:15]
	v_mul_lo_u32 v0, s79, v12
	v_mul_lo_u32 v14, s78, v13
	v_mad_u64_u32 v[12:13], s[14:15], s78, v12, 0
	v_add3_u32 v0, v13, v14, v0
	v_sub_u32_e32 v13, v10, v0
	v_sub_co_u32_e32 v12, vcc, v19, v12
	s_nop 1
	v_subb_co_u32_e64 v13, s[14:15], v13, v4, vcc
	v_subrev_co_u32_e64 v14, s[14:15], s78, v12
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v15, s[22:23], 0, v13, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v15
	v_subb_co_u32_e64 v13, s[14:15], v13, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v16, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v14
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v15
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v12
	v_cndmask_b32_e64 v16, v16, v17, s[22:23]
	v_subrev_co_u32_e64 v17, s[14:15], s78, v14
	s_nop 1
	v_subbrev_co_u32_e64 v13, s[14:15], 0, v13, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v16
	s_nop 1
	v_cndmask_b32_e64 v13, v15, v13, s[14:15]
	v_cndmask_b32_e64 v15, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v15, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v14, v17, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v10, v12, v10, vcc
	v_cndmask_b32_e32 v0, v0, v13, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v11, v18, s33
	v_add3_u32 v21, v0, v10, v11
	v_or_b32_e32 v10, 6, v2
	v_sub_u32_e32 v0, -7, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v12, 31, v0
	v_cndmask_b32_e64 v13, v11, v12, s[10:11]
	v_cndmask_b32_e64 v12, v10, v0, s[10:11]
	v_mov_b32_e32 v14, v13
	v_mov_b32_e32 v15, v13
	v_lshl_add_u64 v[14:15], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v22, v14, v13
	v_xor_b32_e32 v12, v15, v13
	v_mad_u64_u32 v[14:15], s[14:15], v22, s1, 0
	v_mul_hi_u32 v0, v22, s0
	v_lshl_add_u64 v[14:15], v[0:1], 0, v[14:15]
	v_mad_u64_u32 v[18:19], s[14:15], v12, s0, 0
	v_add_co_u32_e32 v0, vcc, v14, v18
	v_mad_u64_u32 v[16:17], s[14:15], v12, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v14, vcc, v15, v19, vcc
	v_mov_b32_e32 v15, s63
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[16:17]
	v_mul_lo_u32 v0, s79, v14
	v_mul_lo_u32 v15, s78, v15
	v_mad_u64_u32 v[16:17], s[14:15], s78, v14, 0
	v_add3_u32 v0, v17, v15, v0
	v_sub_u32_e32 v15, v12, v0
	v_sub_co_u32_e32 v16, vcc, v22, v16
	s_nop 1
	v_subb_co_u32_e64 v15, s[14:15], v15, v4, vcc
	v_subrev_co_u32_e64 v17, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v12, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v15, s[14:15], 0, v15, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v15
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v18, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v17
	v_cndmask_b32_e64 v12, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v17, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v15
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v15, v18, v17, s[14:15]
	v_add_u32_e32 v17, 2, v14
	v_add_u32_e32 v18, 1, v14
	v_cmp_ne_u32_e64 s[14:15], 0, v15
	v_cndmask_b32_e32 v0, v12, v16, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v15, v18, v17, s[14:15]
	v_xor_b32_e32 v12, s72, v13
	v_cndmask_b32_e32 v0, v14, v15, vcc
	v_xor_b32_e32 v0, v0, v12
	v_sub_u32_e32 v0, v0, v12
	v_mov_b32_e32 v12, v11
	v_mov_b32_e32 v13, v11
	v_lshl_add_u64 v[12:13], v[10:11], 0, v[12:13]
	v_xor_b32_e32 v19, v12, v11
	v_xad_u32 v18, v0, v3, v37
	v_xor_b32_e32 v10, v13, v11
	v_mad_u64_u32 v[12:13], s[14:15], v19, s1, 0
	v_mul_hi_u32 v0, v19, s0
	v_lshl_add_u64 v[12:13], v[0:1], 0, v[12:13]
	v_mad_u64_u32 v[16:17], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v12, v16
	v_mad_u64_u32 v[14:15], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v13, v17, vcc
	v_mov_b32_e32 v13, s63
	s_nop 0
	v_addc_co_u32_e32 v15, vcc, 0, v15, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[14:15]
	v_mul_lo_u32 v0, s79, v12
	v_mul_lo_u32 v14, s78, v13
	v_mad_u64_u32 v[12:13], s[14:15], s78, v12, 0
	v_add3_u32 v0, v13, v14, v0
	v_sub_u32_e32 v13, v10, v0
	v_sub_co_u32_e32 v12, vcc, v19, v12
	s_nop 1
	v_subb_co_u32_e64 v13, s[14:15], v13, v4, vcc
	v_subrev_co_u32_e64 v14, s[14:15], s78, v12
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v15, s[22:23], 0, v13, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v15
	v_subb_co_u32_e64 v13, s[14:15], v13, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v16, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v14
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v15
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v12
	v_cndmask_b32_e64 v16, v16, v17, s[22:23]
	v_subrev_co_u32_e64 v17, s[14:15], s78, v14
	s_nop 1
	v_subbrev_co_u32_e64 v13, s[14:15], 0, v13, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v16
	s_nop 1
	v_cndmask_b32_e64 v13, v15, v13, s[14:15]
	v_cndmask_b32_e64 v15, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v15, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v14, v17, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v10, v12, v10, vcc
	v_cndmask_b32_e32 v0, v0, v13, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v11, v18, s33
	v_add3_u32 v22, v0, v10, v11
	v_or_b32_e32 v10, 7, v2
	v_sub_u32_e32 v0, -8, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v12, 31, v0
	v_cndmask_b32_e64 v13, v11, v12, s[10:11]
	v_cndmask_b32_e64 v12, v10, v0, s[10:11]
	v_mov_b32_e32 v14, v13
	v_mov_b32_e32 v15, v13
	v_lshl_add_u64 v[14:15], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v23, v14, v13
	v_xor_b32_e32 v12, v15, v13
	v_mad_u64_u32 v[14:15], s[14:15], v23, s1, 0
	v_mul_hi_u32 v0, v23, s0
	v_lshl_add_u64 v[14:15], v[0:1], 0, v[14:15]
	v_mad_u64_u32 v[18:19], s[14:15], v12, s0, 0
	v_add_co_u32_e32 v0, vcc, v14, v18
	v_mad_u64_u32 v[16:17], s[14:15], v12, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v14, vcc, v15, v19, vcc
	v_mov_b32_e32 v15, s63
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[16:17]
	v_mul_lo_u32 v0, s79, v14
	v_mul_lo_u32 v15, s78, v15
	v_mad_u64_u32 v[16:17], s[14:15], s78, v14, 0
	v_add3_u32 v0, v17, v15, v0
	v_sub_u32_e32 v15, v12, v0
	v_sub_co_u32_e32 v16, vcc, v23, v16
	s_nop 1
	v_subb_co_u32_e64 v15, s[14:15], v15, v4, vcc
	v_subrev_co_u32_e64 v17, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v12, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v15, s[14:15], 0, v15, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v15
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v18, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v17
	v_cndmask_b32_e64 v12, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v17, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v15
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v15, v18, v17, s[14:15]
	v_add_u32_e32 v17, 2, v14
	v_add_u32_e32 v18, 1, v14
	v_cmp_ne_u32_e64 s[14:15], 0, v15
	v_cndmask_b32_e32 v0, v12, v16, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v15, v18, v17, s[14:15]
	v_xor_b32_e32 v12, s72, v13
	v_cndmask_b32_e32 v0, v14, v15, vcc
	v_xor_b32_e32 v0, v0, v12
	v_sub_u32_e32 v0, v0, v12
	v_mov_b32_e32 v12, v11
	v_mov_b32_e32 v13, v11
	v_lshl_add_u64 v[12:13], v[10:11], 0, v[12:13]
	v_xor_b32_e32 v19, v12, v11
	v_xad_u32 v18, v0, v3, v37
	v_xor_b32_e32 v10, v13, v11
	v_mad_u64_u32 v[12:13], s[14:15], v19, s1, 0
	v_mul_hi_u32 v0, v19, s0
	v_lshl_add_u64 v[12:13], v[0:1], 0, v[12:13]
	v_mad_u64_u32 v[16:17], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v12, v16
	v_mad_u64_u32 v[14:15], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v13, v17, vcc
	v_mov_b32_e32 v13, s63
	s_nop 0
	v_addc_co_u32_e32 v15, vcc, 0, v15, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[14:15]
	v_mul_lo_u32 v0, s79, v12
	v_mul_lo_u32 v14, s78, v13
	v_mad_u64_u32 v[12:13], s[14:15], s78, v12, 0
	v_add3_u32 v0, v13, v14, v0
	v_sub_u32_e32 v13, v10, v0
	v_sub_co_u32_e32 v12, vcc, v19, v12
	s_nop 1
	v_subb_co_u32_e64 v13, s[14:15], v13, v4, vcc
	v_subrev_co_u32_e64 v14, s[14:15], s78, v12
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v15, s[22:23], 0, v13, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v15
	v_subb_co_u32_e64 v13, s[14:15], v13, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v16, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v14
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v15
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v12
	v_cndmask_b32_e64 v16, v16, v17, s[22:23]
	v_subrev_co_u32_e64 v17, s[14:15], s78, v14
	s_nop 1
	v_subbrev_co_u32_e64 v13, s[14:15], 0, v13, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v16
	s_nop 1
	v_cndmask_b32_e64 v13, v15, v13, s[14:15]
	v_cndmask_b32_e64 v15, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v15, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v14, v17, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v10, v12, v10, vcc
	v_cndmask_b32_e32 v0, v0, v13, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v11, v18, s33
	v_add3_u32 v0, v0, v10, v11
	buffer_load_ubyte v12, v6, s[36:39], 0 offen
	buffer_load_ubyte v14, v7, s[36:39], 0 offen
	buffer_load_ubyte v13, v8, s[36:39], 0 offen
	buffer_load_ubyte v15, v9, s[36:39], 0 offen
	buffer_load_ubyte v18, v20, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v20, v21, s[36:39], 0 offen
	buffer_load_ubyte v19, v22, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v21, v0, s[36:39], 0 offen
	v_or_b32_e32 v6, 8, v2
	v_sub_u32_e32 v0, -9, v2
	v_ashrrev_i32_e32 v7, 31, v6
	v_ashrrev_i32_e32 v8, 31, v0
	v_cndmask_b32_e64 v9, v7, v8, s[10:11]
	v_cndmask_b32_e64 v8, v6, v0, s[10:11]
	v_mov_b32_e32 v10, v9
	v_mov_b32_e32 v11, v9
	v_lshl_add_u64 v[10:11], v[8:9], 0, v[10:11]
	v_xor_b32_e32 v29, v10, v9
	v_xor_b32_e32 v8, v11, v9
	v_mad_u64_u32 v[10:11], s[14:15], v29, s1, 0
	v_mul_hi_u32 v0, v29, s0
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[10:11]
	v_mad_u64_u32 v[22:23], s[14:15], v8, s0, 0
	v_add_co_u32_e32 v0, vcc, v10, v22
	v_mad_u64_u32 v[16:17], s[14:15], v8, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v23, vcc
	v_mov_b32_e32 v11, s63
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[16:17]
	v_mul_lo_u32 v0, s79, v10
	v_mul_lo_u32 v11, s78, v11
	v_mad_u64_u32 v[16:17], s[14:15], s78, v10, 0
	v_add3_u32 v0, v17, v11, v0
	v_sub_u32_e32 v11, v8, v0
	v_sub_co_u32_e32 v16, vcc, v29, v16
	s_nop 1
	v_subb_co_u32_e64 v11, s[14:15], v11, v4, vcc
	v_subrev_co_u32_e64 v17, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v11, s[14:15], 0, v11, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v11
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v22, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v17
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v17, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v11
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v11, v22, v17, s[14:15]
	v_add_u32_e32 v17, 2, v10
	v_add_u32_e32 v22, 1, v10
	v_cmp_ne_u32_e64 s[14:15], 0, v11
	v_cndmask_b32_e32 v0, v8, v16, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v11, v22, v17, s[14:15]
	v_xor_b32_e32 v8, s72, v9
	v_cndmask_b32_e32 v0, v10, v11, vcc
	v_xor_b32_e32 v0, v0, v8
	v_sub_u32_e32 v0, v0, v8
	v_mov_b32_e32 v8, v7
	v_mov_b32_e32 v9, v7
	v_lshl_add_u64 v[8:9], v[6:7], 0, v[8:9]
	v_xor_b32_e32 v23, v8, v7
	v_xad_u32 v22, v0, v3, v37
	v_xor_b32_e32 v6, v9, v7
	v_mad_u64_u32 v[8:9], s[14:15], v23, s1, 0
	v_mul_hi_u32 v0, v23, s0
	v_lshl_add_u64 v[8:9], v[0:1], 0, v[8:9]
	v_mad_u64_u32 v[16:17], s[14:15], v6, s0, 0
	v_add_co_u32_e32 v0, vcc, v8, v16
	v_mad_u64_u32 v[10:11], s[14:15], v6, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v8, vcc, v9, v17, vcc
	v_mov_b32_e32 v9, s63
	s_nop 0
	v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[10:11]
	v_mul_lo_u32 v0, s79, v8
	v_mul_lo_u32 v10, s78, v9
	v_mad_u64_u32 v[8:9], s[14:15], s78, v8, 0
	v_add3_u32 v0, v9, v10, v0
	v_sub_u32_e32 v9, v6, v0
	v_sub_co_u32_e32 v8, vcc, v23, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[14:15], v9, v4, vcc
	v_subrev_co_u32_e64 v10, s[14:15], s78, v8
	v_subb_co_u32_e32 v0, vcc, v6, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v11, s[22:23], 0, v9, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v11
	v_subb_co_u32_e64 v9, s[14:15], v9, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v16, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v10
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v11
	v_cndmask_b32_e64 v6, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v8
	v_cndmask_b32_e64 v16, v16, v17, s[22:23]
	v_subrev_co_u32_e64 v17, s[14:15], s78, v10
	s_nop 1
	v_subbrev_co_u32_e64 v9, s[14:15], 0, v9, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v16
	s_nop 1
	v_cndmask_b32_e64 v9, v11, v9, s[14:15]
	v_cndmask_b32_e64 v11, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v6, v6, v11, vcc
	v_cmp_ne_u32_e32 vcc, 0, v6
	v_cndmask_b32_e64 v6, v10, v17, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v6, v8, v6, vcc
	v_cndmask_b32_e32 v0, v0, v9, vcc
	v_xor_b32_e32 v6, v6, v7
	v_xor_b32_e32 v0, v0, v7
	v_sub_co_u32_e32 v6, vcc, v6, v7
	v_or_b32_e32 v8, 9, v2
	s_nop 0
	v_subb_co_u32_e32 v0, vcc, v0, v7, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v7, v22, s33
	v_add3_u32 v6, v0, v6, v7
	v_sub_u32_e32 v0, -10, v2
	v_ashrrev_i32_e32 v9, 31, v8
	v_ashrrev_i32_e32 v7, 31, v0
	v_cndmask_b32_e64 v11, v9, v7, s[10:11]
	v_cndmask_b32_e64 v10, v8, v0, s[10:11]
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_xor_b32_e32 v10, v16, v11
	v_xor_b32_e32 v7, v17, v11
	v_mad_u64_u32 v[16:17], s[14:15], v10, s1, 0
	v_mul_hi_u32 v0, v10, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[42:43], s[14:15], v7, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v42
	v_mad_u64_u32 v[22:23], s[14:15], v7, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v43, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v23, vcc, 0, v23, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[22:23]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v17, s78, v17
	v_mad_u64_u32 v[22:23], s[14:15], s78, v16, 0
	v_add3_u32 v0, v23, v17, v0
	v_sub_u32_e32 v17, v7, v0
	v_sub_co_u32_e32 v10, vcc, v10, v22
	s_nop 1
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, vcc
	v_subrev_co_u32_e64 v22, s[14:15], s78, v10
	v_subb_co_u32_e32 v0, vcc, v7, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v17, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v17
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v23, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v22
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v10
	v_cndmask_b32_e64 v22, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v17
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v17, v23, v22, s[14:15]
	v_add_u32_e32 v22, 2, v16
	v_add_u32_e32 v23, 1, v16
	v_cmp_ne_u32_e64 s[14:15], 0, v17
	v_cndmask_b32_e32 v0, v7, v10, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v17, v23, v22, s[14:15]
	v_xor_b32_e32 v7, s72, v11
	v_cndmask_b32_e32 v0, v16, v17, vcc
	v_mov_b32_e32 v10, v9
	v_mov_b32_e32 v11, v9
	v_xor_b32_e32 v0, v0, v7
	v_lshl_add_u64 v[10:11], v[8:9], 0, v[10:11]
	v_sub_u32_e32 v0, v0, v7
	v_xor_b32_e32 v29, v10, v9
	v_xad_u32 v7, v0, v3, v37
	v_xor_b32_e32 v8, v11, v9
	v_mad_u64_u32 v[10:11], s[14:15], v29, s1, 0
	v_mul_hi_u32 v0, v29, s0
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[10:11]
	v_mad_u64_u32 v[22:23], s[14:15], v8, s0, 0
	v_add_co_u32_e32 v0, vcc, v10, v22
	v_mad_u64_u32 v[16:17], s[14:15], v8, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v23, vcc
	v_mov_b32_e32 v11, s63
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[16:17]
	v_mul_lo_u32 v0, s79, v10
	v_mul_lo_u32 v16, s78, v11
	v_mad_u64_u32 v[10:11], s[14:15], s78, v10, 0
	v_add3_u32 v0, v11, v16, v0
	v_sub_u32_e32 v11, v8, v0
	v_sub_co_u32_e32 v10, vcc, v29, v10
	v_mul_lo_u32 v7, v7, s33
	s_nop 0
	v_subb_co_u32_e64 v11, s[14:15], v11, v4, vcc
	v_subrev_co_u32_e64 v16, s[14:15], s78, v10
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v17, s[22:23], 0, v11, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v17
	v_subb_co_u32_e64 v11, s[14:15], v11, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v22, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v16
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v23, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v17
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v10
	v_cndmask_b32_e64 v22, v22, v23, s[22:23]
	v_subrev_co_u32_e64 v23, s[14:15], s78, v16
	s_nop 1
	v_subbrev_co_u32_e64 v11, s[14:15], 0, v11, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v22
	s_nop 1
	v_cndmask_b32_e64 v11, v17, v11, s[14:15]
	v_cndmask_b32_e64 v17, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v8, v8, v17, vcc
	v_cmp_ne_u32_e32 vcc, 0, v8
	v_cndmask_b32_e64 v8, v16, v23, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v8, v10, v8, vcc
	v_cndmask_b32_e32 v0, v0, v11, vcc
	v_xor_b32_e32 v8, v8, v9
	v_xor_b32_e32 v0, v0, v9
	v_sub_co_u32_e32 v8, vcc, v8, v9
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v9, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_add3_u32 v7, v0, v8, v7
	v_or_b32_e32 v8, 10, v2
	v_sub_u32_e32 v0, -11, v2
	v_ashrrev_i32_e32 v9, 31, v8
	v_ashrrev_i32_e32 v10, 31, v0
	v_cndmask_b32_e64 v11, v9, v10, s[10:11]
	v_cndmask_b32_e64 v10, v8, v0, s[10:11]
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_xor_b32_e32 v29, v16, v11
	v_xor_b32_e32 v10, v17, v11
	v_mad_u64_u32 v[16:17], s[14:15], v29, s1, 0
	v_mul_hi_u32 v0, v29, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[42:43], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v42
	v_mad_u64_u32 v[22:23], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v43, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v23, vcc, 0, v23, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[22:23]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v17, s78, v17
	v_mad_u64_u32 v[22:23], s[14:15], s78, v16, 0
	v_add3_u32 v0, v23, v17, v0
	v_sub_u32_e32 v17, v10, v0
	v_sub_co_u32_e32 v22, vcc, v29, v22
	s_nop 1
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, vcc
	v_subrev_co_u32_e64 v23, s[14:15], s78, v22
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v17, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v17
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v29, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v23
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v22
	v_cndmask_b32_e64 v23, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v17
	v_cndmask_b32_e64 v22, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v17, v29, v23, s[14:15]
	v_add_u32_e32 v23, 2, v16
	v_add_u32_e32 v29, 1, v16
	v_cmp_ne_u32_e64 s[14:15], 0, v17
	v_cndmask_b32_e32 v0, v10, v22, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v17, v29, v23, s[14:15]
	v_xor_b32_e32 v10, s72, v11
	v_cndmask_b32_e32 v0, v16, v17, vcc
	v_xor_b32_e32 v0, v0, v10
	v_sub_u32_e32 v0, v0, v10
	v_mov_b32_e32 v10, v9
	v_mov_b32_e32 v11, v9
	v_lshl_add_u64 v[10:11], v[8:9], 0, v[10:11]
	v_xor_b32_e32 v31, v10, v9
	v_xad_u32 v29, v0, v3, v37
	v_xor_b32_e32 v8, v11, v9
	v_mad_u64_u32 v[10:11], s[14:15], v31, s1, 0
	v_mul_hi_u32 v0, v31, s0
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[10:11]
	v_mad_u64_u32 v[22:23], s[14:15], v8, s0, 0
	v_add_co_u32_e32 v0, vcc, v10, v22
	v_mad_u64_u32 v[16:17], s[14:15], v8, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v23, vcc
	v_mov_b32_e32 v11, s63
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[16:17]
	v_mul_lo_u32 v0, s79, v10
	v_mul_lo_u32 v16, s78, v11
	v_mad_u64_u32 v[10:11], s[14:15], s78, v10, 0
	v_add3_u32 v0, v11, v16, v0
	v_sub_u32_e32 v11, v8, v0
	v_sub_co_u32_e32 v10, vcc, v31, v10
	s_nop 1
	v_subb_co_u32_e64 v11, s[14:15], v11, v4, vcc
	v_subrev_co_u32_e64 v16, s[14:15], s78, v10
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v17, s[22:23], 0, v11, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v17
	v_subb_co_u32_e64 v11, s[14:15], v11, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v22, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v16
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v23, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v17
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v10
	v_cndmask_b32_e64 v22, v22, v23, s[22:23]
	v_subrev_co_u32_e64 v23, s[14:15], s78, v16
	s_nop 1
	v_subbrev_co_u32_e64 v11, s[14:15], 0, v11, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v22
	s_nop 1
	v_cndmask_b32_e64 v11, v17, v11, s[14:15]
	v_cndmask_b32_e64 v17, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v8, v8, v17, vcc
	v_cmp_ne_u32_e32 vcc, 0, v8
	v_cndmask_b32_e64 v8, v16, v23, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v8, v10, v8, vcc
	v_cndmask_b32_e32 v0, v0, v11, vcc
	v_xor_b32_e32 v8, v8, v9
	v_xor_b32_e32 v0, v0, v9
	v_sub_co_u32_e32 v8, vcc, v8, v9
	v_or_b32_e32 v10, 11, v2
	s_nop 0
	v_subb_co_u32_e32 v0, vcc, v0, v9, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v9, v29, s33
	v_add3_u32 v8, v0, v8, v9
	v_sub_u32_e32 v0, -12, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v9, 31, v0
	v_cndmask_b32_e64 v17, v11, v9, s[10:11]
	v_cndmask_b32_e64 v16, v10, v0, s[10:11]
	v_mov_b32_e32 v22, v17
	v_mov_b32_e32 v23, v17
	v_lshl_add_u64 v[22:23], v[16:17], 0, v[22:23]
	v_xor_b32_e32 v16, v22, v17
	v_xor_b32_e32 v9, v23, v17
	v_mad_u64_u32 v[22:23], s[14:15], v16, s1, 0
	v_mul_hi_u32 v0, v16, s0
	v_lshl_add_u64 v[22:23], v[0:1], 0, v[22:23]
	v_mad_u64_u32 v[46:47], s[14:15], v9, s0, 0
	v_add_co_u32_e32 v0, vcc, v22, v46
	v_mad_u64_u32 v[42:43], s[14:15], v9, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v22, vcc, v23, v47, vcc
	v_mov_b32_e32 v23, s63
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[42:43]
	v_mul_lo_u32 v0, s79, v22
	v_mul_lo_u32 v23, s78, v23
	v_mad_u64_u32 v[42:43], s[14:15], s78, v22, 0
	v_add3_u32 v0, v43, v23, v0
	v_sub_u32_e32 v23, v9, v0
	v_sub_co_u32_e32 v16, vcc, v16, v42
	s_nop 1
	v_subb_co_u32_e64 v23, s[14:15], v23, v4, vcc
	v_subrev_co_u32_e64 v29, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v9, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v23, s[14:15], 0, v23, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v23
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v31, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v29
	v_cndmask_b32_e64 v9, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v29, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v23
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v23, v31, v29, s[14:15]
	v_add_u32_e32 v29, 2, v22
	v_add_u32_e32 v31, 1, v22
	v_cmp_ne_u32_e64 s[14:15], 0, v23
	v_cndmask_b32_e32 v0, v9, v16, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v23, v31, v29, s[14:15]
	v_xor_b32_e32 v9, s72, v17
	v_cndmask_b32_e32 v0, v22, v23, vcc
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_xor_b32_e32 v0, v0, v9
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_sub_u32_e32 v0, v0, v9
	v_xor_b32_e32 v29, v16, v11
	v_xad_u32 v9, v0, v3, v37
	v_xor_b32_e32 v10, v17, v11
	v_mad_u64_u32 v[16:17], s[14:15], v29, s1, 0
	v_mul_hi_u32 v0, v29, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[42:43], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v42
	v_mad_u64_u32 v[22:23], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v43, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v23, vcc, 0, v23, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[22:23]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v22, s78, v17
	v_mad_u64_u32 v[16:17], s[14:15], s78, v16, 0
	v_add3_u32 v0, v17, v22, v0
	v_sub_u32_e32 v17, v10, v0
	v_sub_co_u32_e32 v16, vcc, v29, v16
	v_mul_lo_u32 v9, v9, s33
	s_nop 0
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, vcc
	v_subrev_co_u32_e64 v22, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v23, s[22:23], 0, v17, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v23
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v29, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v22
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v31, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v23
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v29, v29, v31, s[22:23]
	v_subrev_co_u32_e64 v31, s[14:15], s78, v22
	s_nop 1
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v17, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v29
	s_nop 1
	v_cndmask_b32_e64 v17, v23, v17, s[14:15]
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v23, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v22, v31, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v10, v16, v10, vcc
	v_cndmask_b32_e32 v0, v0, v17, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_add3_u32 v9, v0, v10, v9
	v_or_b32_e32 v10, 12, v2
	v_sub_u32_e32 v0, -13, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v16, 31, v0
	v_cndmask_b32_e64 v17, v11, v16, s[10:11]
	v_cndmask_b32_e64 v16, v10, v0, s[10:11]
	v_mov_b32_e32 v22, v17
	v_mov_b32_e32 v23, v17
	v_lshl_add_u64 v[22:23], v[16:17], 0, v[22:23]
	v_xor_b32_e32 v29, v22, v17
	v_xor_b32_e32 v16, v23, v17
	v_mad_u64_u32 v[22:23], s[14:15], v29, s1, 0
	v_mul_hi_u32 v0, v29, s0
	v_lshl_add_u64 v[22:23], v[0:1], 0, v[22:23]
	v_mad_u64_u32 v[46:47], s[14:15], v16, s0, 0
	v_add_co_u32_e32 v0, vcc, v22, v46
	v_mad_u64_u32 v[42:43], s[14:15], v16, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v22, vcc, v23, v47, vcc
	v_mov_b32_e32 v23, s63
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[42:43]
	v_mul_lo_u32 v0, s79, v22
	v_mul_lo_u32 v23, s78, v23
	v_mad_u64_u32 v[42:43], s[14:15], s78, v22, 0
	v_add3_u32 v0, v43, v23, v0
	v_sub_u32_e32 v23, v16, v0
	v_sub_co_u32_e32 v29, vcc, v29, v42
	s_nop 1
	v_subb_co_u32_e64 v23, s[14:15], v23, v4, vcc
	v_subrev_co_u32_e64 v31, s[14:15], s78, v29
	v_subb_co_u32_e32 v0, vcc, v16, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v23, s[14:15], 0, v23, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v23
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v33, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v31
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v29
	v_cndmask_b32_e64 v31, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v23
	v_cndmask_b32_e64 v29, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v23, v33, v31, s[14:15]
	v_add_u32_e32 v31, 2, v22
	v_add_u32_e32 v33, 1, v22
	v_cmp_ne_u32_e64 s[14:15], 0, v23
	v_cndmask_b32_e32 v0, v16, v29, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v23, v33, v31, s[14:15]
	v_xor_b32_e32 v16, s72, v17
	v_cndmask_b32_e32 v0, v22, v23, vcc
	v_xor_b32_e32 v0, v0, v16
	v_sub_u32_e32 v0, v0, v16
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_xor_b32_e32 v31, v16, v11
	v_xad_u32 v29, v0, v3, v37
	v_xor_b32_e32 v10, v17, v11
	v_mad_u64_u32 v[16:17], s[14:15], v31, s1, 0
	v_mul_hi_u32 v0, v31, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[42:43], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v42
	v_mad_u64_u32 v[22:23], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v43, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v23, vcc, 0, v23, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[22:23]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v22, s78, v17
	v_mad_u64_u32 v[16:17], s[14:15], s78, v16, 0
	v_add3_u32 v0, v17, v22, v0
	v_sub_u32_e32 v17, v10, v0
	v_sub_co_u32_e32 v16, vcc, v31, v16
	s_nop 1
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, vcc
	v_subrev_co_u32_e64 v22, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v23, s[22:23], 0, v17, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v23
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v31, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v22
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v33, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v23
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v31, v31, v33, s[22:23]
	v_subrev_co_u32_e64 v33, s[14:15], s78, v22
	s_nop 1
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v17, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v31
	s_nop 1
	v_cndmask_b32_e64 v17, v23, v17, s[14:15]
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v23, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v22, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v10, v16, v10, vcc
	v_cndmask_b32_e32 v0, v0, v17, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v11, v29, s33
	v_add3_u32 v31, v0, v10, v11
	v_or_b32_e32 v10, 13, v2
	v_sub_u32_e32 v0, -14, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v16, 31, v0
	v_cndmask_b32_e64 v17, v11, v16, s[10:11]
	v_cndmask_b32_e64 v16, v10, v0, s[10:11]
	v_mov_b32_e32 v22, v17
	v_mov_b32_e32 v23, v17
	v_lshl_add_u64 v[22:23], v[16:17], 0, v[22:23]
	v_xor_b32_e32 v29, v22, v17
	v_xor_b32_e32 v16, v23, v17
	v_mad_u64_u32 v[22:23], s[14:15], v29, s1, 0
	v_mul_hi_u32 v0, v29, s0
	v_lshl_add_u64 v[22:23], v[0:1], 0, v[22:23]
	v_mad_u64_u32 v[46:47], s[14:15], v16, s0, 0
	v_add_co_u32_e32 v0, vcc, v22, v46
	v_mad_u64_u32 v[42:43], s[14:15], v16, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v22, vcc, v23, v47, vcc
	v_mov_b32_e32 v23, s63
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[42:43]
	v_mul_lo_u32 v0, s79, v22
	v_mul_lo_u32 v23, s78, v23
	v_mad_u64_u32 v[42:43], s[14:15], s78, v22, 0
	v_add3_u32 v0, v43, v23, v0
	v_sub_u32_e32 v23, v16, v0
	v_sub_co_u32_e32 v29, vcc, v29, v42
	s_nop 1
	v_subb_co_u32_e64 v23, s[14:15], v23, v4, vcc
	v_subrev_co_u32_e64 v33, s[14:15], s78, v29
	v_subb_co_u32_e32 v0, vcc, v16, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v23, s[14:15], 0, v23, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v23
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v35, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v33
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v29
	v_cndmask_b32_e64 v33, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v23
	v_cndmask_b32_e64 v29, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v23, v35, v33, s[14:15]
	v_add_u32_e32 v33, 2, v22
	v_add_u32_e32 v35, 1, v22
	v_cmp_ne_u32_e64 s[14:15], 0, v23
	v_cndmask_b32_e32 v0, v16, v29, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v23, v35, v33, s[14:15]
	v_xor_b32_e32 v16, s72, v17
	v_cndmask_b32_e32 v0, v22, v23, vcc
	v_xor_b32_e32 v0, v0, v16
	v_sub_u32_e32 v0, v0, v16
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_xor_b32_e32 v33, v16, v11
	v_xad_u32 v29, v0, v3, v37
	v_xor_b32_e32 v10, v17, v11
	v_mad_u64_u32 v[16:17], s[14:15], v33, s1, 0
	v_mul_hi_u32 v0, v33, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[42:43], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v42
	v_mad_u64_u32 v[22:23], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v43, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v23, vcc, 0, v23, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[22:23]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v22, s78, v17
	v_mad_u64_u32 v[16:17], s[14:15], s78, v16, 0
	v_add3_u32 v0, v17, v22, v0
	v_sub_u32_e32 v17, v10, v0
	v_sub_co_u32_e32 v16, vcc, v33, v16
	s_nop 1
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, vcc
	v_subrev_co_u32_e64 v22, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v23, s[22:23], 0, v17, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v23
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v33, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v22
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v35, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v23
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v33, v33, v35, s[22:23]
	v_subrev_co_u32_e64 v35, s[14:15], s78, v22
	s_nop 1
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v17, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v33
	s_nop 1
	v_cndmask_b32_e64 v17, v23, v17, s[14:15]
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v23, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v22, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v10, v16, v10, vcc
	v_cndmask_b32_e32 v0, v0, v17, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v11, v29, s33
	v_add3_u32 v33, v0, v10, v11
	v_or_b32_e32 v10, 14, v2
	v_sub_u32_e32 v0, -15, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v16, 31, v0
	v_cndmask_b32_e64 v17, v11, v16, s[10:11]
	v_cndmask_b32_e64 v16, v10, v0, s[10:11]
	v_mov_b32_e32 v22, v17
	v_mov_b32_e32 v23, v17
	v_lshl_add_u64 v[22:23], v[16:17], 0, v[22:23]
	v_xor_b32_e32 v29, v22, v17
	v_xor_b32_e32 v16, v23, v17
	v_mad_u64_u32 v[22:23], s[14:15], v29, s1, 0
	v_mul_hi_u32 v0, v29, s0
	v_lshl_add_u64 v[22:23], v[0:1], 0, v[22:23]
	v_mad_u64_u32 v[46:47], s[14:15], v16, s0, 0
	v_add_co_u32_e32 v0, vcc, v22, v46
	v_mad_u64_u32 v[42:43], s[14:15], v16, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v22, vcc, v23, v47, vcc
	v_mov_b32_e32 v23, s63
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[42:43]
	v_mul_lo_u32 v0, s79, v22
	v_mul_lo_u32 v23, s78, v23
	v_mad_u64_u32 v[42:43], s[14:15], s78, v22, 0
	v_add3_u32 v0, v43, v23, v0
	v_sub_u32_e32 v23, v16, v0
	v_sub_co_u32_e32 v29, vcc, v29, v42
	s_nop 1
	v_subb_co_u32_e64 v23, s[14:15], v23, v4, vcc
	v_subrev_co_u32_e64 v35, s[14:15], s78, v29
	v_subb_co_u32_e32 v0, vcc, v16, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v23, s[14:15], 0, v23, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v23
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v39, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v35
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v29
	v_cndmask_b32_e64 v35, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v23
	v_cndmask_b32_e64 v29, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v23, v39, v35, s[14:15]
	v_add_u32_e32 v35, 2, v22
	v_add_u32_e32 v39, 1, v22
	v_cmp_ne_u32_e64 s[14:15], 0, v23
	v_cndmask_b32_e32 v0, v16, v29, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v23, v39, v35, s[14:15]
	v_xor_b32_e32 v16, s72, v17
	v_cndmask_b32_e32 v0, v22, v23, vcc
	v_xor_b32_e32 v0, v0, v16
	v_sub_u32_e32 v0, v0, v16
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_xor_b32_e32 v35, v16, v11
	v_xad_u32 v29, v0, v3, v37
	v_xor_b32_e32 v10, v17, v11
	v_mad_u64_u32 v[16:17], s[14:15], v35, s1, 0
	v_mul_hi_u32 v0, v35, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[42:43], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v42
	v_mad_u64_u32 v[22:23], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v43, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v23, vcc, 0, v23, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[22:23]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v22, s78, v17
	v_mad_u64_u32 v[16:17], s[14:15], s78, v16, 0
	v_add3_u32 v0, v17, v22, v0
	v_sub_u32_e32 v17, v10, v0
	v_sub_co_u32_e32 v16, vcc, v35, v16
	s_nop 1
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, vcc
	v_subrev_co_u32_e64 v22, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v23, s[22:23], 0, v17, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v23
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v35, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v22
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v39, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v23
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v35, v35, v39, s[22:23]
	v_subrev_co_u32_e64 v39, s[14:15], s78, v22
	s_nop 1
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v17, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v35
	s_nop 1
	v_cndmask_b32_e64 v17, v23, v17, s[14:15]
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v23, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v22, v39, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v10, v16, v10, vcc
	v_cndmask_b32_e32 v0, v0, v17, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v11, v29, s33
	v_add3_u32 v39, v0, v10, v11
	v_or_b32_e32 v10, 15, v2
	v_sub_u32_e32 v0, -16, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v16, 31, v0
	v_cndmask_b32_e64 v17, v11, v16, s[10:11]
	v_cndmask_b32_e64 v16, v10, v0, s[10:11]
	v_mov_b32_e32 v22, v17
	v_mov_b32_e32 v23, v17
	v_lshl_add_u64 v[22:23], v[16:17], 0, v[22:23]
	v_xor_b32_e32 v29, v22, v17
	v_xor_b32_e32 v16, v23, v17
	v_mad_u64_u32 v[22:23], s[10:11], v29, s1, 0
	v_mul_hi_u32 v0, v29, s0
	v_lshl_add_u64 v[22:23], v[0:1], 0, v[22:23]
	v_mad_u64_u32 v[46:47], s[10:11], v16, s0, 0
	v_add_co_u32_e32 v0, vcc, v22, v46
	v_mad_u64_u32 v[42:43], s[10:11], v16, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v22, vcc, v23, v47, vcc
	v_mov_b32_e32 v23, s63
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[42:43]
	v_mul_lo_u32 v0, s79, v22
	v_mul_lo_u32 v23, s78, v23
	v_mad_u64_u32 v[42:43], s[10:11], s78, v22, 0
	v_add3_u32 v0, v43, v23, v0
	v_sub_u32_e32 v23, v16, v0
	v_sub_co_u32_e32 v29, vcc, v29, v42
	s_nop 1
	v_subb_co_u32_e64 v23, s[10:11], v23, v4, vcc
	v_subrev_co_u32_e64 v35, s[10:11], s78, v29
	v_subb_co_u32_e32 v0, vcc, v16, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v23, s[10:11], 0, v23, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s79, v23
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v42, 0, -1, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s78, v35
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v29
	v_cndmask_b32_e64 v35, 0, -1, s[10:11]
	v_cmp_eq_u32_e64 s[10:11], s79, v23
	v_cndmask_b32_e64 v29, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v23, v42, v35, s[10:11]
	v_add_u32_e32 v35, 2, v22
	v_add_u32_e32 v42, 1, v22
	v_cmp_ne_u32_e64 s[10:11], 0, v23
	v_cndmask_b32_e32 v0, v16, v29, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v23, v42, v35, s[10:11]
	v_xor_b32_e32 v16, s72, v17
	v_cndmask_b32_e32 v0, v22, v23, vcc
	v_xor_b32_e32 v0, v0, v16
	v_sub_u32_e32 v0, v0, v16
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_xor_b32_e32 v29, v16, v11
	v_xad_u32 v3, v0, v3, v37
	v_xor_b32_e32 v10, v17, v11
	v_mad_u64_u32 v[16:17], s[10:11], v29, s1, 0
	v_mul_hi_u32 v0, v29, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[42:43], s[10:11], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v42
	v_mad_u64_u32 v[22:23], s[10:11], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v43, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v23, vcc, 0, v23, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[22:23]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v22, s78, v17
	v_mad_u64_u32 v[16:17], s[10:11], s78, v16, 0
	v_add3_u32 v0, v17, v22, v0
	v_sub_u32_e32 v17, v10, v0
	v_sub_co_u32_e32 v16, vcc, v29, v16
	v_mul_lo_u32 v3, v3, s33
	s_nop 0
	v_subb_co_u32_e64 v17, s[10:11], v17, v4, vcc
	v_subrev_co_u32_e64 v22, s[10:11], s78, v16
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v23, s[14:15], 0, v17, s[10:11]
	v_cmp_le_u32_e64 s[14:15], s79, v23
	v_subb_co_u32_e64 v17, s[10:11], v17, v4, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v29, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v22
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v35, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v23
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v29, v29, v35, s[14:15]
	v_subrev_co_u32_e64 v35, s[10:11], s78, v22
	s_nop 1
	v_subbrev_co_u32_e64 v17, s[10:11], 0, v17, s[10:11]
	v_cmp_ne_u32_e64 s[10:11], 0, v29
	s_nop 1
	v_cndmask_b32_e64 v17, v23, v17, s[10:11]
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v23, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v22, v35, s[10:11]
	s_movk_i32 s10, 0xfc00
	v_cndmask_b32_e32 v10, v16, v10, vcc
	v_cndmask_b32_e32 v0, v0, v17, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	v_cmp_gt_i32_e64 s[10:11], s10, v2
	s_nop 0
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_add3_u32 v0, v0, v10, v3
	buffer_load_ubyte v3, v6, s[36:39], 0 offen
	buffer_load_ubyte v23, v7, s[36:39], 0 offen
	buffer_load_ubyte v22, v8, s[36:39], 0 offen
	buffer_load_ubyte v29, v9, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v31, v31, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v35, v33, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v33, v39, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v39, v0, s[36:39], 0 offen
	v_add_u32_e32 v8, 0x400, v2
	v_sub_u32_e32 v0, 0xfffffbff, v2
	v_ashrrev_i32_e32 v9, 31, v8
	v_ashrrev_i32_e32 v6, 31, v0
	v_cndmask_b32_e64 v7, v9, v6, s[10:11]
	v_cndmask_b32_e64 v6, v8, v0, s[10:11]
	v_mov_b32_e32 v10, v7
	v_mov_b32_e32 v11, v7
	v_lshl_add_u64 v[10:11], v[6:7], 0, v[10:11]
	v_xor_b32_e32 v6, v10, v7
	v_xor_b32_e32 v2, v11, v7
	v_mad_u64_u32 v[10:11], s[14:15], v6, s1, 0
	v_mul_hi_u32 v0, v6, s0
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[10:11]
	v_mad_u64_u32 v[42:43], s[14:15], v2, s0, 0
	v_add_co_u32_e32 v0, vcc, v10, v42
	v_mad_u64_u32 v[16:17], s[14:15], v2, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v43, vcc
	v_mov_b32_e32 v11, s63
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[16:17]
	v_mul_lo_u32 v0, s79, v10
	v_mul_lo_u32 v11, s78, v11
	v_mad_u64_u32 v[16:17], s[14:15], s78, v10, 0
	v_add3_u32 v0, v17, v11, v0
	v_sub_u32_e32 v11, v2, v0
	v_sub_co_u32_e32 v6, vcc, v6, v16
	s_nop 1
	v_subb_co_u32_e64 v11, s[14:15], v11, v4, vcc
	v_subrev_co_u32_e64 v16, s[14:15], s78, v6
	v_subb_co_u32_e32 v0, vcc, v2, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v11, s[14:15], 0, v11, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v11
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v16
	v_cndmask_b32_e64 v2, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v6
	v_cndmask_b32_e64 v16, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v11
	v_cndmask_b32_e64 v6, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v11, v17, v16, s[14:15]
	v_add_u32_e32 v16, 2, v10
	v_add_u32_e32 v17, 1, v10
	v_cmp_ne_u32_e64 s[14:15], 0, v11
	v_cndmask_b32_e32 v0, v2, v6, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v11, v17, v16, s[14:15]
	v_xor_b32_e32 v2, s72, v7
	v_cndmask_b32_e32 v0, v10, v11, vcc
	v_mov_b32_e32 v6, v9
	v_mov_b32_e32 v7, v9
	v_xor_b32_e32 v0, v0, v2
	v_lshl_add_u64 v[6:7], v[8:9], 0, v[6:7]
	v_sub_u32_e32 v0, v0, v2
	v_cndmask_b32_e64 v2, 0, -1, s[10:11]
	v_xor_b32_e32 v43, v6, v9
	v_xad_u32 v2, v0, v2, v37
	v_xor_b32_e32 v42, v7, v9
	v_mad_u64_u32 v[6:7], s[10:11], v43, s1, 0
	v_mul_hi_u32 v0, v43, s0
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[6:7]
	v_mad_u64_u32 v[16:17], s[10:11], v42, s0, 0
	v_add_co_u32_e32 v0, vcc, v6, v16
	v_mad_u64_u32 v[10:11], s[10:11], v42, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v6, vcc, v7, v17, vcc
	v_mov_b32_e32 v7, s63
	s_nop 0
	v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[10:11]
	v_mul_lo_u32 v0, s79, v6
	v_mul_lo_u32 v10, s78, v7
	v_mad_u64_u32 v[6:7], s[10:11], s78, v6, 0
	v_add3_u32 v0, v7, v10, v0
	v_sub_u32_e32 v7, v42, v0
	v_sub_co_u32_e32 v6, vcc, v43, v6
	v_mul_lo_u32 v2, v2, s33
	s_nop 0
	v_subb_co_u32_e64 v7, s[10:11], v7, v4, vcc
	v_subrev_co_u32_e64 v10, s[10:11], s78, v6
	v_subb_co_u32_e32 v0, vcc, v42, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v11, s[14:15], 0, v7, s[10:11]
	v_cmp_le_u32_e64 s[14:15], s79, v11
	v_subb_co_u32_e64 v7, s[10:11], v7, v4, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v16, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v10
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v11
	s_nop 1
	v_cndmask_b32_e64 v16, v16, v17, s[14:15]
	v_subrev_co_u32_e64 v17, s[10:11], s78, v10
	s_nop 1
	v_subbrev_co_u32_e64 v7, s[10:11], 0, v7, s[10:11]
	v_cmp_ne_u32_e64 s[10:11], 0, v16
	s_nop 1
	v_cndmask_b32_e64 v7, v11, v7, s[10:11]
	v_cndmask_b32_e64 v11, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v6
	s_nop 1
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v11, v11, v16, vcc
	v_cmp_ne_u32_e32 vcc, 0, v11
	s_nop 1
	v_cndmask_b32_e32 v0, v0, v7, vcc
	v_cndmask_b32_e64 v7, v10, v17, s[10:11]
	v_cndmask_b32_e32 v6, v6, v7, vcc
	v_xor_b32_e32 v6, v6, v9
	v_xor_b32_e32 v0, v0, v9
	v_sub_co_u32_e32 v6, vcc, v6, v9
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v9, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_add3_u32 v6, v0, v6, v2
	v_bfe_u32 v2, v45, 1, 1
	v_and_b32_e32 v0, 0x200, v5
	v_mul_i32_i24_e32 v66, 0xfffffe00, v2
	v_add3_u32 v2, v8, v0, v66
	v_or_b32_e32 v8, 1, v2
	v_sub_u32_e32 v0, -2, v2
	v_ashrrev_i32_e32 v9, 31, v8
	v_ashrrev_i32_e32 v5, 31, v0
	v_cmp_gt_i32_e64 s[10:11], 0, v2
	s_nop 1
	v_cndmask_b32_e64 v11, v9, v5, s[10:11]
	v_cndmask_b32_e64 v10, v8, v0, s[10:11]
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_xor_b32_e32 v7, v16, v11
	v_xor_b32_e32 v5, v17, v11
	v_mad_u64_u32 v[16:17], s[14:15], v7, s1, 0
	v_mul_hi_u32 v0, v7, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[46:47], s[14:15], v5, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v46
	v_mad_u64_u32 v[42:43], s[14:15], v5, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v47, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[42:43]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v10, s78, v17
	v_mad_u64_u32 v[42:43], s[14:15], s78, v16, 0
	v_add3_u32 v0, v43, v10, v0
	v_sub_u32_e32 v10, v5, v0
	v_sub_co_u32_e32 v7, vcc, v7, v42
	s_nop 1
	v_subb_co_u32_e64 v10, s[14:15], v10, v4, vcc
	v_subrev_co_u32_e64 v17, s[14:15], s78, v7
	v_subb_co_u32_e32 v0, vcc, v5, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v10, s[14:15], 0, v10, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v10
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v42, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v17
	v_cndmask_b32_e64 v5, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v7
	v_cndmask_b32_e64 v17, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v10
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v10, v42, v17, s[14:15]
	v_add_u32_e32 v17, 2, v16
	v_add_u32_e32 v42, 1, v16
	v_cmp_ne_u32_e64 s[14:15], 0, v10
	v_cndmask_b32_e32 v0, v5, v7, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v10, v42, v17, s[14:15]
	v_xor_b32_e32 v5, s72, v11
	v_cndmask_b32_e32 v0, v16, v10, vcc
	v_mov_b32_e32 v10, v9
	v_mov_b32_e32 v11, v9
	v_xor_b32_e32 v0, v0, v5
	v_lshl_add_u64 v[10:11], v[8:9], 0, v[10:11]
	v_sub_u32_e32 v0, v0, v5
	v_ashrrev_i32_e32 v5, 31, v2
	v_xor_b32_e32 v46, v10, v9
	v_xad_u32 v7, v0, v5, v37
	v_xor_b32_e32 v8, v11, v9
	v_mad_u64_u32 v[10:11], s[14:15], v46, s1, 0
	v_mul_hi_u32 v0, v46, s0
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[10:11]
	v_mad_u64_u32 v[42:43], s[14:15], v8, s0, 0
	v_add_co_u32_e32 v0, vcc, v10, v42
	v_mad_u64_u32 v[16:17], s[14:15], v8, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v43, vcc
	v_mov_b32_e32 v11, s63
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[16:17]
	v_mul_lo_u32 v0, s79, v10
	v_mul_lo_u32 v16, s78, v11
	v_mad_u64_u32 v[10:11], s[14:15], s78, v10, 0
	v_add3_u32 v0, v11, v16, v0
	v_sub_u32_e32 v11, v8, v0
	v_sub_co_u32_e32 v10, vcc, v46, v10
	v_mul_lo_u32 v7, v7, s33
	s_nop 0
	v_subb_co_u32_e64 v11, s[14:15], v11, v4, vcc
	v_subrev_co_u32_e64 v16, s[14:15], s78, v10
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v17, s[22:23], 0, v11, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v17
	v_subb_co_u32_e64 v11, s[14:15], v11, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v42, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v16
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v43, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v17
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v10
	v_cndmask_b32_e64 v42, v42, v43, s[22:23]
	v_subrev_co_u32_e64 v43, s[14:15], s78, v16
	s_nop 1
	v_subbrev_co_u32_e64 v11, s[14:15], 0, v11, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v42
	s_nop 1
	v_cndmask_b32_e64 v11, v17, v11, s[14:15]
	v_cndmask_b32_e64 v17, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v8, v8, v17, vcc
	v_cmp_ne_u32_e32 vcc, 0, v8
	v_cndmask_b32_e64 v8, v16, v43, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v8, v10, v8, vcc
	v_cndmask_b32_e32 v0, v0, v11, vcc
	v_xor_b32_e32 v8, v8, v9
	v_xor_b32_e32 v0, v0, v9
	v_sub_co_u32_e32 v8, vcc, v8, v9
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v9, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_add3_u32 v7, v0, v8, v7
	v_or_b32_e32 v8, 2, v2
	v_sub_u32_e32 v0, -3, v2
	v_ashrrev_i32_e32 v9, 31, v8
	v_ashrrev_i32_e32 v10, 31, v0
	v_cndmask_b32_e64 v11, v9, v10, s[10:11]
	v_cndmask_b32_e64 v10, v8, v0, s[10:11]
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_xor_b32_e32 v48, v16, v11
	v_xor_b32_e32 v10, v17, v11
	v_mad_u64_u32 v[16:17], s[14:15], v48, s1, 0
	v_mul_hi_u32 v0, v48, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[46:47], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v46
	v_mad_u64_u32 v[42:43], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v47, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[42:43]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v17, s78, v17
	v_mad_u64_u32 v[42:43], s[14:15], s78, v16, 0
	v_add3_u32 v0, v43, v17, v0
	v_sub_u32_e32 v17, v10, v0
	v_sub_co_u32_e32 v42, vcc, v48, v42
	s_nop 1
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, vcc
	v_subrev_co_u32_e64 v43, s[14:15], s78, v42
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v17, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v17
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v46, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v43
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v42
	v_cndmask_b32_e64 v43, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v17
	v_cndmask_b32_e64 v42, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v17, v46, v43, s[14:15]
	v_add_u32_e32 v43, 2, v16
	v_add_u32_e32 v46, 1, v16
	v_cmp_ne_u32_e64 s[14:15], 0, v17
	v_cndmask_b32_e32 v0, v10, v42, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v17, v46, v43, s[14:15]
	v_xor_b32_e32 v10, s72, v11
	v_cndmask_b32_e32 v0, v16, v17, vcc
	v_xor_b32_e32 v0, v0, v10
	v_sub_u32_e32 v0, v0, v10
	v_mov_b32_e32 v10, v9
	v_mov_b32_e32 v11, v9
	v_lshl_add_u64 v[10:11], v[8:9], 0, v[10:11]
	v_xor_b32_e32 v47, v10, v9
	v_xad_u32 v46, v0, v5, v37
	v_xor_b32_e32 v8, v11, v9
	v_mad_u64_u32 v[10:11], s[14:15], v47, s1, 0
	v_mul_hi_u32 v0, v47, s0
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[10:11]
	v_mad_u64_u32 v[42:43], s[14:15], v8, s0, 0
	v_add_co_u32_e32 v0, vcc, v10, v42
	v_mad_u64_u32 v[16:17], s[14:15], v8, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v43, vcc
	v_mov_b32_e32 v11, s63
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[16:17]
	v_mul_lo_u32 v0, s79, v10
	v_mul_lo_u32 v16, s78, v11
	v_mad_u64_u32 v[10:11], s[14:15], s78, v10, 0
	v_add3_u32 v0, v11, v16, v0
	v_sub_u32_e32 v11, v8, v0
	v_sub_co_u32_e32 v10, vcc, v47, v10
	s_nop 1
	v_subb_co_u32_e64 v11, s[14:15], v11, v4, vcc
	v_subrev_co_u32_e64 v16, s[14:15], s78, v10
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v17, s[22:23], 0, v11, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v17
	v_subb_co_u32_e64 v11, s[14:15], v11, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v42, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v16
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v43, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v17
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v10
	v_cndmask_b32_e64 v42, v42, v43, s[22:23]
	v_subrev_co_u32_e64 v43, s[14:15], s78, v16
	s_nop 1
	v_subbrev_co_u32_e64 v11, s[14:15], 0, v11, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v42
	s_nop 1
	v_cndmask_b32_e64 v11, v17, v11, s[14:15]
	v_cndmask_b32_e64 v17, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v8, v8, v17, vcc
	v_cmp_ne_u32_e32 vcc, 0, v8
	v_cndmask_b32_e64 v8, v16, v43, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v8, v10, v8, vcc
	v_cndmask_b32_e32 v0, v0, v11, vcc
	v_xor_b32_e32 v8, v8, v9
	v_xor_b32_e32 v0, v0, v9
	v_sub_co_u32_e32 v8, vcc, v8, v9
	v_or_b32_e32 v10, 3, v2
	s_nop 0
	v_subb_co_u32_e32 v0, vcc, v0, v9, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v9, v46, s33
	v_add3_u32 v8, v0, v8, v9
	v_sub_u32_e32 v0, -4, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v9, 31, v0
	v_cndmask_b32_e64 v17, v11, v9, s[10:11]
	v_cndmask_b32_e64 v16, v10, v0, s[10:11]
	v_mov_b32_e32 v42, v17
	v_mov_b32_e32 v43, v17
	v_lshl_add_u64 v[42:43], v[16:17], 0, v[42:43]
	v_xor_b32_e32 v16, v42, v17
	v_xor_b32_e32 v9, v43, v17
	v_mad_u64_u32 v[42:43], s[14:15], v16, s1, 0
	v_mul_hi_u32 v0, v16, s0
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[42:43]
	v_mad_u64_u32 v[48:49], s[14:15], v9, s0, 0
	v_add_co_u32_e32 v0, vcc, v42, v48
	v_mad_u64_u32 v[46:47], s[14:15], v9, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v49, vcc
	v_mov_b32_e32 v43, s63
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	v_mul_lo_u32 v0, s79, v42
	v_mul_lo_u32 v43, s78, v43
	v_mad_u64_u32 v[46:47], s[14:15], s78, v42, 0
	v_add3_u32 v0, v47, v43, v0
	v_sub_u32_e32 v43, v9, v0
	v_sub_co_u32_e32 v16, vcc, v16, v46
	s_nop 1
	v_subb_co_u32_e64 v43, s[14:15], v43, v4, vcc
	v_subrev_co_u32_e64 v46, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v9, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[14:15], 0, v43, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v43
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v46
	v_cndmask_b32_e64 v9, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v46, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v43
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v43, v47, v46, s[14:15]
	v_add_u32_e32 v46, 2, v42
	v_add_u32_e32 v47, 1, v42
	v_cmp_ne_u32_e64 s[14:15], 0, v43
	v_cndmask_b32_e32 v0, v9, v16, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v43, v47, v46, s[14:15]
	v_xor_b32_e32 v9, s72, v17
	v_cndmask_b32_e32 v0, v42, v43, vcc
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_xor_b32_e32 v0, v0, v9
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_sub_u32_e32 v0, v0, v9
	v_xor_b32_e32 v48, v16, v11
	v_xad_u32 v9, v0, v5, v37
	v_xor_b32_e32 v10, v17, v11
	v_mad_u64_u32 v[16:17], s[14:15], v48, s1, 0
	v_mul_hi_u32 v0, v48, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[46:47], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v46
	v_mad_u64_u32 v[42:43], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v47, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[42:43]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v42, s78, v17
	v_mad_u64_u32 v[16:17], s[14:15], s78, v16, 0
	v_add3_u32 v0, v17, v42, v0
	v_sub_u32_e32 v17, v10, v0
	v_sub_co_u32_e32 v16, vcc, v48, v16
	v_mul_lo_u32 v9, v9, s33
	s_nop 0
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, vcc
	v_subrev_co_u32_e64 v42, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[22:23], 0, v17, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v43
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v46, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v42
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v43
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v46, v46, v47, s[22:23]
	v_subrev_co_u32_e64 v47, s[14:15], s78, v42
	s_nop 1
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v17, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v46
	s_nop 1
	v_cndmask_b32_e64 v17, v43, v17, s[14:15]
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v42, v47, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v10, v16, v10, vcc
	v_cndmask_b32_e32 v0, v0, v17, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_add3_u32 v9, v0, v10, v9
	v_or_b32_e32 v10, 4, v2
	v_sub_u32_e32 v0, -5, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v16, 31, v0
	v_cndmask_b32_e64 v17, v11, v16, s[10:11]
	v_cndmask_b32_e64 v16, v10, v0, s[10:11]
	v_mov_b32_e32 v42, v17
	v_mov_b32_e32 v43, v17
	v_lshl_add_u64 v[42:43], v[16:17], 0, v[42:43]
	v_xor_b32_e32 v50, v42, v17
	v_xor_b32_e32 v16, v43, v17
	v_mad_u64_u32 v[42:43], s[14:15], v50, s1, 0
	v_mul_hi_u32 v0, v50, s0
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[42:43]
	v_mad_u64_u32 v[48:49], s[14:15], v16, s0, 0
	v_add_co_u32_e32 v0, vcc, v42, v48
	v_mad_u64_u32 v[46:47], s[14:15], v16, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v49, vcc
	v_mov_b32_e32 v43, s63
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	v_mul_lo_u32 v0, s79, v42
	v_mul_lo_u32 v43, s78, v43
	v_mad_u64_u32 v[46:47], s[14:15], s78, v42, 0
	v_add3_u32 v0, v47, v43, v0
	v_sub_u32_e32 v43, v16, v0
	v_sub_co_u32_e32 v46, vcc, v50, v46
	s_nop 1
	v_subb_co_u32_e64 v43, s[14:15], v43, v4, vcc
	v_subrev_co_u32_e64 v47, s[14:15], s78, v46
	v_subb_co_u32_e32 v0, vcc, v16, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[14:15], 0, v43, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v43
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v48, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v47
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v46
	v_cndmask_b32_e64 v47, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v43
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v43, v48, v47, s[14:15]
	v_add_u32_e32 v47, 2, v42
	v_add_u32_e32 v48, 1, v42
	v_cmp_ne_u32_e64 s[14:15], 0, v43
	v_cndmask_b32_e32 v0, v16, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v43, v48, v47, s[14:15]
	v_xor_b32_e32 v16, s72, v17
	v_cndmask_b32_e32 v0, v42, v43, vcc
	v_xor_b32_e32 v0, v0, v16
	v_sub_u32_e32 v0, v0, v16
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_xor_b32_e32 v49, v16, v11
	v_xad_u32 v48, v0, v5, v37
	v_xor_b32_e32 v10, v17, v11
	v_mad_u64_u32 v[16:17], s[14:15], v49, s1, 0
	v_mul_hi_u32 v0, v49, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[46:47], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v46
	v_mad_u64_u32 v[42:43], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v47, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[42:43]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v42, s78, v17
	v_mad_u64_u32 v[16:17], s[14:15], s78, v16, 0
	v_add3_u32 v0, v17, v42, v0
	v_sub_u32_e32 v17, v10, v0
	v_sub_co_u32_e32 v16, vcc, v49, v16
	s_nop 1
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, vcc
	v_subrev_co_u32_e64 v42, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[22:23], 0, v17, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v43
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v46, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v42
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v43
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v46, v46, v47, s[22:23]
	v_subrev_co_u32_e64 v47, s[14:15], s78, v42
	s_nop 1
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v17, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v46
	s_nop 1
	v_cndmask_b32_e64 v17, v43, v17, s[14:15]
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v42, v47, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v10, v16, v10, vcc
	v_cndmask_b32_e32 v0, v0, v17, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v11, v48, s33
	v_add3_u32 v54, v0, v10, v11
	v_or_b32_e32 v10, 5, v2
	v_sub_u32_e32 v0, -6, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v16, 31, v0
	v_cndmask_b32_e64 v17, v11, v16, s[10:11]
	v_cndmask_b32_e64 v16, v10, v0, s[10:11]
	v_mov_b32_e32 v42, v17
	v_mov_b32_e32 v43, v17
	v_lshl_add_u64 v[42:43], v[16:17], 0, v[42:43]
	v_xor_b32_e32 v50, v42, v17
	v_xor_b32_e32 v16, v43, v17
	v_mad_u64_u32 v[42:43], s[14:15], v50, s1, 0
	v_mul_hi_u32 v0, v50, s0
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[42:43]
	v_mad_u64_u32 v[48:49], s[14:15], v16, s0, 0
	v_add_co_u32_e32 v0, vcc, v42, v48
	v_mad_u64_u32 v[46:47], s[14:15], v16, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v49, vcc
	v_mov_b32_e32 v43, s63
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	v_mul_lo_u32 v0, s79, v42
	v_mul_lo_u32 v43, s78, v43
	v_mad_u64_u32 v[46:47], s[14:15], s78, v42, 0
	v_add3_u32 v0, v47, v43, v0
	v_sub_u32_e32 v43, v16, v0
	v_sub_co_u32_e32 v46, vcc, v50, v46
	s_nop 1
	v_subb_co_u32_e64 v43, s[14:15], v43, v4, vcc
	v_subrev_co_u32_e64 v47, s[14:15], s78, v46
	v_subb_co_u32_e32 v0, vcc, v16, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[14:15], 0, v43, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v43
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v48, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v47
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v46
	v_cndmask_b32_e64 v47, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v43
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v43, v48, v47, s[14:15]
	v_add_u32_e32 v47, 2, v42
	v_add_u32_e32 v48, 1, v42
	v_cmp_ne_u32_e64 s[14:15], 0, v43
	v_cndmask_b32_e32 v0, v16, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v43, v48, v47, s[14:15]
	v_xor_b32_e32 v16, s72, v17
	v_cndmask_b32_e32 v0, v42, v43, vcc
	v_xor_b32_e32 v0, v0, v16
	v_sub_u32_e32 v0, v0, v16
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_xor_b32_e32 v49, v16, v11
	v_xad_u32 v48, v0, v5, v37
	v_xor_b32_e32 v10, v17, v11
	v_mad_u64_u32 v[16:17], s[14:15], v49, s1, 0
	v_mul_hi_u32 v0, v49, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[46:47], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v46
	v_mad_u64_u32 v[42:43], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v47, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[42:43]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v42, s78, v17
	v_mad_u64_u32 v[16:17], s[14:15], s78, v16, 0
	v_add3_u32 v0, v17, v42, v0
	v_sub_u32_e32 v17, v10, v0
	v_sub_co_u32_e32 v16, vcc, v49, v16
	s_nop 1
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, vcc
	v_subrev_co_u32_e64 v42, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[22:23], 0, v17, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v43
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v46, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v42
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v43
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v46, v46, v47, s[22:23]
	v_subrev_co_u32_e64 v47, s[14:15], s78, v42
	s_nop 1
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v17, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v46
	s_nop 1
	v_cndmask_b32_e64 v17, v43, v17, s[14:15]
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v42, v47, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v10, v16, v10, vcc
	v_cndmask_b32_e32 v0, v0, v17, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v11, v48, s33
	v_add3_u32 v55, v0, v10, v11
	v_or_b32_e32 v10, 6, v2
	v_sub_u32_e32 v0, -7, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v16, 31, v0
	v_cndmask_b32_e64 v17, v11, v16, s[10:11]
	v_cndmask_b32_e64 v16, v10, v0, s[10:11]
	v_mov_b32_e32 v42, v17
	v_mov_b32_e32 v43, v17
	v_lshl_add_u64 v[42:43], v[16:17], 0, v[42:43]
	v_xor_b32_e32 v50, v42, v17
	v_xor_b32_e32 v16, v43, v17
	v_mad_u64_u32 v[42:43], s[14:15], v50, s1, 0
	v_mul_hi_u32 v0, v50, s0
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[42:43]
	v_mad_u64_u32 v[48:49], s[14:15], v16, s0, 0
	v_add_co_u32_e32 v0, vcc, v42, v48
	v_mad_u64_u32 v[46:47], s[14:15], v16, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v49, vcc
	v_mov_b32_e32 v43, s63
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	v_mul_lo_u32 v0, s79, v42
	v_mul_lo_u32 v43, s78, v43
	v_mad_u64_u32 v[46:47], s[14:15], s78, v42, 0
	v_add3_u32 v0, v47, v43, v0
	v_sub_u32_e32 v43, v16, v0
	v_sub_co_u32_e32 v46, vcc, v50, v46
	s_nop 1
	v_subb_co_u32_e64 v43, s[14:15], v43, v4, vcc
	v_subrev_co_u32_e64 v47, s[14:15], s78, v46
	v_subb_co_u32_e32 v0, vcc, v16, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[14:15], 0, v43, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v43
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v48, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v47
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v46
	v_cndmask_b32_e64 v47, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v43
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v43, v48, v47, s[14:15]
	v_add_u32_e32 v47, 2, v42
	v_add_u32_e32 v48, 1, v42
	v_cmp_ne_u32_e64 s[14:15], 0, v43
	v_cndmask_b32_e32 v0, v16, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v43, v48, v47, s[14:15]
	v_xor_b32_e32 v16, s72, v17
	v_cndmask_b32_e32 v0, v42, v43, vcc
	v_xor_b32_e32 v0, v0, v16
	v_sub_u32_e32 v0, v0, v16
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_xor_b32_e32 v49, v16, v11
	v_xad_u32 v48, v0, v5, v37
	v_xor_b32_e32 v10, v17, v11
	v_mad_u64_u32 v[16:17], s[14:15], v49, s1, 0
	v_mul_hi_u32 v0, v49, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[46:47], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v46
	v_mad_u64_u32 v[42:43], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v47, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[42:43]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v42, s78, v17
	v_mad_u64_u32 v[16:17], s[14:15], s78, v16, 0
	v_add3_u32 v0, v17, v42, v0
	v_sub_u32_e32 v17, v10, v0
	v_sub_co_u32_e32 v16, vcc, v49, v16
	s_nop 1
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, vcc
	v_subrev_co_u32_e64 v42, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[22:23], 0, v17, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v43
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v46, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v42
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v43
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v46, v46, v47, s[22:23]
	v_subrev_co_u32_e64 v47, s[14:15], s78, v42
	s_nop 1
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v17, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v46
	s_nop 1
	v_cndmask_b32_e64 v17, v43, v17, s[14:15]
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v42, v47, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v10, v16, v10, vcc
	v_cndmask_b32_e32 v0, v0, v17, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v11, v48, s33
	v_add3_u32 v57, v0, v10, v11
	v_or_b32_e32 v10, 7, v2
	v_sub_u32_e32 v0, -8, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v16, 31, v0
	v_cndmask_b32_e64 v17, v11, v16, s[10:11]
	v_cndmask_b32_e64 v16, v10, v0, s[10:11]
	v_mov_b32_e32 v42, v17
	v_mov_b32_e32 v43, v17
	v_lshl_add_u64 v[42:43], v[16:17], 0, v[42:43]
	v_xor_b32_e32 v50, v42, v17
	v_xor_b32_e32 v16, v43, v17
	v_mad_u64_u32 v[42:43], s[14:15], v50, s1, 0
	v_mul_hi_u32 v0, v50, s0
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[42:43]
	v_mad_u64_u32 v[48:49], s[14:15], v16, s0, 0
	v_add_co_u32_e32 v0, vcc, v42, v48
	v_mad_u64_u32 v[46:47], s[14:15], v16, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v49, vcc
	v_mov_b32_e32 v43, s63
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	v_mul_lo_u32 v0, s79, v42
	v_mul_lo_u32 v43, s78, v43
	v_mad_u64_u32 v[46:47], s[14:15], s78, v42, 0
	v_add3_u32 v0, v47, v43, v0
	v_sub_u32_e32 v43, v16, v0
	v_sub_co_u32_e32 v46, vcc, v50, v46
	s_nop 1
	v_subb_co_u32_e64 v43, s[14:15], v43, v4, vcc
	v_subrev_co_u32_e64 v47, s[14:15], s78, v46
	v_subb_co_u32_e32 v0, vcc, v16, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[14:15], 0, v43, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v43
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v48, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v47
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v46
	v_cndmask_b32_e64 v47, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v43
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v43, v48, v47, s[14:15]
	v_add_u32_e32 v47, 2, v42
	v_add_u32_e32 v48, 1, v42
	v_cmp_ne_u32_e64 s[14:15], 0, v43
	v_cndmask_b32_e32 v0, v16, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v43, v48, v47, s[14:15]
	v_xor_b32_e32 v16, s72, v17
	v_cndmask_b32_e32 v0, v42, v43, vcc
	v_xor_b32_e32 v0, v0, v16
	v_sub_u32_e32 v0, v0, v16
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_xor_b32_e32 v49, v16, v11
	v_xad_u32 v48, v0, v5, v37
	v_xor_b32_e32 v10, v17, v11
	v_mad_u64_u32 v[16:17], s[14:15], v49, s1, 0
	v_mul_hi_u32 v0, v49, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[46:47], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v46
	v_mad_u64_u32 v[42:43], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v47, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[42:43]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v42, s78, v17
	v_mad_u64_u32 v[16:17], s[14:15], s78, v16, 0
	v_add3_u32 v0, v17, v42, v0
	v_sub_u32_e32 v17, v10, v0
	v_sub_co_u32_e32 v16, vcc, v49, v16
	s_nop 1
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, vcc
	v_subrev_co_u32_e64 v42, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[22:23], 0, v17, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v43
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v46, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v42
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v43
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v46, v46, v47, s[22:23]
	v_subrev_co_u32_e64 v47, s[14:15], s78, v42
	s_nop 1
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v17, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v46
	s_nop 1
	v_cndmask_b32_e64 v17, v43, v17, s[14:15]
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v42, v47, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v10, v16, v10, vcc
	v_cndmask_b32_e32 v0, v0, v17, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v11, v48, s33
	v_add3_u32 v0, v0, v10, v11
	buffer_load_ubyte v50, v6, s[36:39], 0 offen
	buffer_load_ubyte v52, v7, s[36:39], 0 offen
	buffer_load_ubyte v51, v8, s[36:39], 0 offen
	buffer_load_ubyte v53, v9, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v54, v54, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v56, v55, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v55, v57, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v57, v0, s[36:39], 0 offen
	v_or_b32_e32 v6, 8, v2
	v_sub_u32_e32 v0, -9, v2
	v_ashrrev_i32_e32 v7, 31, v6
	v_ashrrev_i32_e32 v8, 31, v0
	v_cndmask_b32_e64 v9, v7, v8, s[10:11]
	v_cndmask_b32_e64 v8, v6, v0, s[10:11]
	v_mov_b32_e32 v10, v9
	v_mov_b32_e32 v11, v9
	v_lshl_add_u64 v[10:11], v[8:9], 0, v[10:11]
	v_xor_b32_e32 v46, v10, v9
	v_xor_b32_e32 v8, v11, v9
	v_mad_u64_u32 v[10:11], s[14:15], v46, s1, 0
	v_mul_hi_u32 v0, v46, s0
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[10:11]
	v_mad_u64_u32 v[42:43], s[14:15], v8, s0, 0
	v_add_co_u32_e32 v0, vcc, v10, v42
	v_mad_u64_u32 v[16:17], s[14:15], v8, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v43, vcc
	v_mov_b32_e32 v11, s63
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[16:17]
	v_mul_lo_u32 v0, s79, v10
	v_mul_lo_u32 v11, s78, v11
	v_mad_u64_u32 v[16:17], s[14:15], s78, v10, 0
	v_add3_u32 v0, v17, v11, v0
	v_sub_u32_e32 v11, v8, v0
	v_sub_co_u32_e32 v16, vcc, v46, v16
	s_nop 1
	v_subb_co_u32_e64 v11, s[14:15], v11, v4, vcc
	v_subrev_co_u32_e64 v17, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v11, s[14:15], 0, v11, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v11
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v42, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v17
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v17, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v11
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v11, v42, v17, s[14:15]
	v_add_u32_e32 v17, 2, v10
	v_add_u32_e32 v42, 1, v10
	v_cmp_ne_u32_e64 s[14:15], 0, v11
	v_cndmask_b32_e32 v0, v8, v16, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v11, v42, v17, s[14:15]
	v_xor_b32_e32 v8, s72, v9
	v_cndmask_b32_e32 v0, v10, v11, vcc
	v_xor_b32_e32 v0, v0, v8
	v_sub_u32_e32 v0, v0, v8
	v_mov_b32_e32 v8, v7
	v_mov_b32_e32 v9, v7
	v_lshl_add_u64 v[8:9], v[6:7], 0, v[8:9]
	v_xor_b32_e32 v43, v8, v7
	v_xad_u32 v42, v0, v5, v37
	v_xor_b32_e32 v6, v9, v7
	v_mad_u64_u32 v[8:9], s[14:15], v43, s1, 0
	v_mul_hi_u32 v0, v43, s0
	v_lshl_add_u64 v[8:9], v[0:1], 0, v[8:9]
	v_mad_u64_u32 v[16:17], s[14:15], v6, s0, 0
	v_add_co_u32_e32 v0, vcc, v8, v16
	v_mad_u64_u32 v[10:11], s[14:15], v6, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v8, vcc, v9, v17, vcc
	v_mov_b32_e32 v9, s63
	s_nop 0
	v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[10:11]
	v_mul_lo_u32 v0, s79, v8
	v_mul_lo_u32 v10, s78, v9
	v_mad_u64_u32 v[8:9], s[14:15], s78, v8, 0
	v_add3_u32 v0, v9, v10, v0
	v_sub_u32_e32 v9, v6, v0
	v_sub_co_u32_e32 v8, vcc, v43, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[14:15], v9, v4, vcc
	v_subrev_co_u32_e64 v10, s[14:15], s78, v8
	v_subb_co_u32_e32 v0, vcc, v6, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v11, s[22:23], 0, v9, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v11
	v_subb_co_u32_e64 v9, s[14:15], v9, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v16, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v10
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v11
	v_cndmask_b32_e64 v6, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v8
	v_cndmask_b32_e64 v16, v16, v17, s[22:23]
	v_subrev_co_u32_e64 v17, s[14:15], s78, v10
	s_nop 1
	v_subbrev_co_u32_e64 v9, s[14:15], 0, v9, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v16
	s_nop 1
	v_cndmask_b32_e64 v9, v11, v9, s[14:15]
	v_cndmask_b32_e64 v11, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v6, v6, v11, vcc
	v_cmp_ne_u32_e32 vcc, 0, v6
	v_cndmask_b32_e64 v6, v10, v17, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v6, v8, v6, vcc
	v_cndmask_b32_e32 v0, v0, v9, vcc
	v_xor_b32_e32 v6, v6, v7
	v_xor_b32_e32 v0, v0, v7
	v_sub_co_u32_e32 v6, vcc, v6, v7
	v_or_b32_e32 v8, 9, v2
	s_nop 0
	v_subb_co_u32_e32 v0, vcc, v0, v7, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v7, v42, s33
	v_add3_u32 v6, v0, v6, v7
	v_sub_u32_e32 v0, -10, v2
	v_ashrrev_i32_e32 v9, 31, v8
	v_ashrrev_i32_e32 v7, 31, v0
	v_cndmask_b32_e64 v11, v9, v7, s[10:11]
	v_cndmask_b32_e64 v10, v8, v0, s[10:11]
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_xor_b32_e32 v10, v16, v11
	v_xor_b32_e32 v7, v17, v11
	v_mad_u64_u32 v[16:17], s[14:15], v10, s1, 0
	v_mul_hi_u32 v0, v10, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[46:47], s[14:15], v7, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v46
	v_mad_u64_u32 v[42:43], s[14:15], v7, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v47, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[42:43]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v17, s78, v17
	v_mad_u64_u32 v[42:43], s[14:15], s78, v16, 0
	v_add3_u32 v0, v43, v17, v0
	v_sub_u32_e32 v17, v7, v0
	v_sub_co_u32_e32 v10, vcc, v10, v42
	s_nop 1
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, vcc
	v_subrev_co_u32_e64 v42, s[14:15], s78, v10
	v_subb_co_u32_e32 v0, vcc, v7, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v17, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v17
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v43, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v42
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v10
	v_cndmask_b32_e64 v42, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v17
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v17, v43, v42, s[14:15]
	v_add_u32_e32 v42, 2, v16
	v_add_u32_e32 v43, 1, v16
	v_cmp_ne_u32_e64 s[14:15], 0, v17
	v_cndmask_b32_e32 v0, v7, v10, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v17, v43, v42, s[14:15]
	v_xor_b32_e32 v7, s72, v11
	v_cndmask_b32_e32 v0, v16, v17, vcc
	v_mov_b32_e32 v10, v9
	v_mov_b32_e32 v11, v9
	v_xor_b32_e32 v0, v0, v7
	v_lshl_add_u64 v[10:11], v[8:9], 0, v[10:11]
	v_sub_u32_e32 v0, v0, v7
	v_xor_b32_e32 v46, v10, v9
	v_xad_u32 v7, v0, v5, v37
	v_xor_b32_e32 v8, v11, v9
	v_mad_u64_u32 v[10:11], s[14:15], v46, s1, 0
	v_mul_hi_u32 v0, v46, s0
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[10:11]
	v_mad_u64_u32 v[42:43], s[14:15], v8, s0, 0
	v_add_co_u32_e32 v0, vcc, v10, v42
	v_mad_u64_u32 v[16:17], s[14:15], v8, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v43, vcc
	v_mov_b32_e32 v11, s63
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[16:17]
	v_mul_lo_u32 v0, s79, v10
	v_mul_lo_u32 v16, s78, v11
	v_mad_u64_u32 v[10:11], s[14:15], s78, v10, 0
	v_add3_u32 v0, v11, v16, v0
	v_sub_u32_e32 v11, v8, v0
	v_sub_co_u32_e32 v10, vcc, v46, v10
	v_mul_lo_u32 v7, v7, s33
	s_nop 0
	v_subb_co_u32_e64 v11, s[14:15], v11, v4, vcc
	v_subrev_co_u32_e64 v16, s[14:15], s78, v10
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v17, s[22:23], 0, v11, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v17
	v_subb_co_u32_e64 v11, s[14:15], v11, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v42, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v16
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v43, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v17
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v10
	v_cndmask_b32_e64 v42, v42, v43, s[22:23]
	v_subrev_co_u32_e64 v43, s[14:15], s78, v16
	s_nop 1
	v_subbrev_co_u32_e64 v11, s[14:15], 0, v11, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v42
	s_nop 1
	v_cndmask_b32_e64 v11, v17, v11, s[14:15]
	v_cndmask_b32_e64 v17, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v8, v8, v17, vcc
	v_cmp_ne_u32_e32 vcc, 0, v8
	v_cndmask_b32_e64 v8, v16, v43, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v8, v10, v8, vcc
	v_cndmask_b32_e32 v0, v0, v11, vcc
	v_xor_b32_e32 v8, v8, v9
	v_xor_b32_e32 v0, v0, v9
	v_sub_co_u32_e32 v8, vcc, v8, v9
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v9, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_add3_u32 v7, v0, v8, v7
	v_or_b32_e32 v8, 10, v2
	v_sub_u32_e32 v0, -11, v2
	v_ashrrev_i32_e32 v9, 31, v8
	v_ashrrev_i32_e32 v10, 31, v0
	v_cndmask_b32_e64 v11, v9, v10, s[10:11]
	v_cndmask_b32_e64 v10, v8, v0, s[10:11]
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_xor_b32_e32 v48, v16, v11
	v_xor_b32_e32 v10, v17, v11
	v_mad_u64_u32 v[16:17], s[14:15], v48, s1, 0
	v_mul_hi_u32 v0, v48, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[46:47], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v46
	v_mad_u64_u32 v[42:43], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v47, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[42:43]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v17, s78, v17
	v_mad_u64_u32 v[42:43], s[14:15], s78, v16, 0
	v_add3_u32 v0, v43, v17, v0
	v_sub_u32_e32 v17, v10, v0
	v_sub_co_u32_e32 v42, vcc, v48, v42
	s_nop 1
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, vcc
	v_subrev_co_u32_e64 v43, s[14:15], s78, v42
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v17, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v17
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v46, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v43
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v42
	v_cndmask_b32_e64 v43, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v17
	v_cndmask_b32_e64 v42, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v17, v46, v43, s[14:15]
	v_add_u32_e32 v43, 2, v16
	v_add_u32_e32 v46, 1, v16
	v_cmp_ne_u32_e64 s[14:15], 0, v17
	v_cndmask_b32_e32 v0, v10, v42, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v17, v46, v43, s[14:15]
	v_xor_b32_e32 v10, s72, v11
	v_cndmask_b32_e32 v0, v16, v17, vcc
	v_xor_b32_e32 v0, v0, v10
	v_sub_u32_e32 v0, v0, v10
	v_mov_b32_e32 v10, v9
	v_mov_b32_e32 v11, v9
	v_lshl_add_u64 v[10:11], v[8:9], 0, v[10:11]
	v_xor_b32_e32 v47, v10, v9
	v_xad_u32 v46, v0, v5, v37
	v_xor_b32_e32 v8, v11, v9
	v_mad_u64_u32 v[10:11], s[14:15], v47, s1, 0
	v_mul_hi_u32 v0, v47, s0
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[10:11]
	v_mad_u64_u32 v[42:43], s[14:15], v8, s0, 0
	v_add_co_u32_e32 v0, vcc, v10, v42
	v_mad_u64_u32 v[16:17], s[14:15], v8, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v43, vcc
	v_mov_b32_e32 v11, s63
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[16:17]
	v_mul_lo_u32 v0, s79, v10
	v_mul_lo_u32 v16, s78, v11
	v_mad_u64_u32 v[10:11], s[14:15], s78, v10, 0
	v_add3_u32 v0, v11, v16, v0
	v_sub_u32_e32 v11, v8, v0
	v_sub_co_u32_e32 v10, vcc, v47, v10
	s_nop 1
	v_subb_co_u32_e64 v11, s[14:15], v11, v4, vcc
	v_subrev_co_u32_e64 v16, s[14:15], s78, v10
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v17, s[22:23], 0, v11, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v17
	v_subb_co_u32_e64 v11, s[14:15], v11, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v42, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v16
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v43, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v17
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v10
	v_cndmask_b32_e64 v42, v42, v43, s[22:23]
	v_subrev_co_u32_e64 v43, s[14:15], s78, v16
	s_nop 1
	v_subbrev_co_u32_e64 v11, s[14:15], 0, v11, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v42
	s_nop 1
	v_cndmask_b32_e64 v11, v17, v11, s[14:15]
	v_cndmask_b32_e64 v17, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v8, v8, v17, vcc
	v_cmp_ne_u32_e32 vcc, 0, v8
	v_cndmask_b32_e64 v8, v16, v43, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v8, v10, v8, vcc
	v_cndmask_b32_e32 v0, v0, v11, vcc
	v_xor_b32_e32 v8, v8, v9
	v_xor_b32_e32 v0, v0, v9
	v_sub_co_u32_e32 v8, vcc, v8, v9
	v_or_b32_e32 v10, 11, v2
	s_nop 0
	v_subb_co_u32_e32 v0, vcc, v0, v9, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v9, v46, s33
	v_add3_u32 v8, v0, v8, v9
	v_sub_u32_e32 v0, -12, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v9, 31, v0
	v_cndmask_b32_e64 v17, v11, v9, s[10:11]
	v_cndmask_b32_e64 v16, v10, v0, s[10:11]
	v_mov_b32_e32 v42, v17
	v_mov_b32_e32 v43, v17
	v_lshl_add_u64 v[42:43], v[16:17], 0, v[42:43]
	v_xor_b32_e32 v16, v42, v17
	v_xor_b32_e32 v9, v43, v17
	v_mad_u64_u32 v[42:43], s[14:15], v16, s1, 0
	v_mul_hi_u32 v0, v16, s0
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[42:43]
	v_mad_u64_u32 v[48:49], s[14:15], v9, s0, 0
	v_add_co_u32_e32 v0, vcc, v42, v48
	v_mad_u64_u32 v[46:47], s[14:15], v9, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v49, vcc
	v_mov_b32_e32 v43, s63
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	v_mul_lo_u32 v0, s79, v42
	v_mul_lo_u32 v43, s78, v43
	v_mad_u64_u32 v[46:47], s[14:15], s78, v42, 0
	v_add3_u32 v0, v47, v43, v0
	v_sub_u32_e32 v43, v9, v0
	v_sub_co_u32_e32 v16, vcc, v16, v46
	s_nop 1
	v_subb_co_u32_e64 v43, s[14:15], v43, v4, vcc
	v_subrev_co_u32_e64 v46, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v9, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[14:15], 0, v43, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v43
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v46
	v_cndmask_b32_e64 v9, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v46, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v43
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v43, v47, v46, s[14:15]
	v_add_u32_e32 v46, 2, v42
	v_add_u32_e32 v47, 1, v42
	v_cmp_ne_u32_e64 s[14:15], 0, v43
	v_cndmask_b32_e32 v0, v9, v16, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v43, v47, v46, s[14:15]
	v_xor_b32_e32 v9, s72, v17
	v_cndmask_b32_e32 v0, v42, v43, vcc
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_xor_b32_e32 v0, v0, v9
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_sub_u32_e32 v0, v0, v9
	v_xor_b32_e32 v48, v16, v11
	v_xad_u32 v9, v0, v5, v37
	v_xor_b32_e32 v10, v17, v11
	v_mad_u64_u32 v[16:17], s[14:15], v48, s1, 0
	v_mul_hi_u32 v0, v48, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[46:47], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v46
	v_mad_u64_u32 v[42:43], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v47, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[42:43]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v42, s78, v17
	v_mad_u64_u32 v[16:17], s[14:15], s78, v16, 0
	v_add3_u32 v0, v17, v42, v0
	v_sub_u32_e32 v17, v10, v0
	v_sub_co_u32_e32 v16, vcc, v48, v16
	v_mul_lo_u32 v9, v9, s33
	s_nop 0
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, vcc
	v_subrev_co_u32_e64 v42, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[22:23], 0, v17, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v43
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v46, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v42
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v43
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v46, v46, v47, s[22:23]
	v_subrev_co_u32_e64 v47, s[14:15], s78, v42
	s_nop 1
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v17, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v46
	s_nop 1
	v_cndmask_b32_e64 v17, v43, v17, s[14:15]
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v42, v47, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v10, v16, v10, vcc
	v_cndmask_b32_e32 v0, v0, v17, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_add3_u32 v9, v0, v10, v9
	v_or_b32_e32 v10, 12, v2
	v_sub_u32_e32 v0, -13, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v16, 31, v0
	v_cndmask_b32_e64 v17, v11, v16, s[10:11]
	v_cndmask_b32_e64 v16, v10, v0, s[10:11]
	v_mov_b32_e32 v42, v17
	v_mov_b32_e32 v43, v17
	v_lshl_add_u64 v[42:43], v[16:17], 0, v[42:43]
	v_xor_b32_e32 v58, v42, v17
	v_xor_b32_e32 v16, v43, v17
	v_mad_u64_u32 v[42:43], s[14:15], v58, s1, 0
	v_mul_hi_u32 v0, v58, s0
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[42:43]
	v_mad_u64_u32 v[48:49], s[14:15], v16, s0, 0
	v_add_co_u32_e32 v0, vcc, v42, v48
	v_mad_u64_u32 v[46:47], s[14:15], v16, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v49, vcc
	v_mov_b32_e32 v43, s63
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	v_mul_lo_u32 v0, s79, v42
	v_mul_lo_u32 v43, s78, v43
	v_mad_u64_u32 v[46:47], s[14:15], s78, v42, 0
	v_add3_u32 v0, v47, v43, v0
	v_sub_u32_e32 v43, v16, v0
	v_sub_co_u32_e32 v46, vcc, v58, v46
	s_nop 1
	v_subb_co_u32_e64 v43, s[14:15], v43, v4, vcc
	v_subrev_co_u32_e64 v47, s[14:15], s78, v46
	v_subb_co_u32_e32 v0, vcc, v16, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[14:15], 0, v43, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v43
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v48, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v47
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v46
	v_cndmask_b32_e64 v47, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v43
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v43, v48, v47, s[14:15]
	v_add_u32_e32 v47, 2, v42
	v_add_u32_e32 v48, 1, v42
	v_cmp_ne_u32_e64 s[14:15], 0, v43
	v_cndmask_b32_e32 v0, v16, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v43, v48, v47, s[14:15]
	v_xor_b32_e32 v16, s72, v17
	v_cndmask_b32_e32 v0, v42, v43, vcc
	v_xor_b32_e32 v0, v0, v16
	v_sub_u32_e32 v0, v0, v16
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_xor_b32_e32 v49, v16, v11
	v_xad_u32 v48, v0, v5, v37
	v_xor_b32_e32 v10, v17, v11
	v_mad_u64_u32 v[16:17], s[14:15], v49, s1, 0
	v_mul_hi_u32 v0, v49, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[46:47], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v46
	v_mad_u64_u32 v[42:43], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v47, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[42:43]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v42, s78, v17
	v_mad_u64_u32 v[16:17], s[14:15], s78, v16, 0
	v_add3_u32 v0, v17, v42, v0
	v_sub_u32_e32 v17, v10, v0
	v_sub_co_u32_e32 v16, vcc, v49, v16
	s_nop 1
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, vcc
	v_subrev_co_u32_e64 v42, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[22:23], 0, v17, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v43
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v46, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v42
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v43
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v46, v46, v47, s[22:23]
	v_subrev_co_u32_e64 v47, s[14:15], s78, v42
	s_nop 1
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v17, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v46
	s_nop 1
	v_cndmask_b32_e64 v17, v43, v17, s[14:15]
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v42, v47, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v10, v16, v10, vcc
	v_cndmask_b32_e32 v0, v0, v17, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v11, v48, s33
	v_add3_u32 v62, v0, v10, v11
	v_or_b32_e32 v10, 13, v2
	v_sub_u32_e32 v0, -14, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v16, 31, v0
	v_cndmask_b32_e64 v17, v11, v16, s[10:11]
	v_cndmask_b32_e64 v16, v10, v0, s[10:11]
	v_mov_b32_e32 v42, v17
	v_mov_b32_e32 v43, v17
	v_lshl_add_u64 v[42:43], v[16:17], 0, v[42:43]
	v_xor_b32_e32 v58, v42, v17
	v_xor_b32_e32 v16, v43, v17
	v_mad_u64_u32 v[42:43], s[14:15], v58, s1, 0
	v_mul_hi_u32 v0, v58, s0
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[42:43]
	v_mad_u64_u32 v[48:49], s[14:15], v16, s0, 0
	v_add_co_u32_e32 v0, vcc, v42, v48
	v_mad_u64_u32 v[46:47], s[14:15], v16, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v49, vcc
	v_mov_b32_e32 v43, s63
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	v_mul_lo_u32 v0, s79, v42
	v_mul_lo_u32 v43, s78, v43
	v_mad_u64_u32 v[46:47], s[14:15], s78, v42, 0
	v_add3_u32 v0, v47, v43, v0
	v_sub_u32_e32 v43, v16, v0
	v_sub_co_u32_e32 v46, vcc, v58, v46
	s_nop 1
	v_subb_co_u32_e64 v43, s[14:15], v43, v4, vcc
	v_subrev_co_u32_e64 v47, s[14:15], s78, v46
	v_subb_co_u32_e32 v0, vcc, v16, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[14:15], 0, v43, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v43
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v48, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v47
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v46
	v_cndmask_b32_e64 v47, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v43
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v43, v48, v47, s[14:15]
	v_add_u32_e32 v47, 2, v42
	v_add_u32_e32 v48, 1, v42
	v_cmp_ne_u32_e64 s[14:15], 0, v43
	v_cndmask_b32_e32 v0, v16, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v43, v48, v47, s[14:15]
	v_xor_b32_e32 v16, s72, v17
	v_cndmask_b32_e32 v0, v42, v43, vcc
	v_xor_b32_e32 v0, v0, v16
	v_sub_u32_e32 v0, v0, v16
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_xor_b32_e32 v49, v16, v11
	v_xad_u32 v48, v0, v5, v37
	v_xor_b32_e32 v10, v17, v11
	v_mad_u64_u32 v[16:17], s[14:15], v49, s1, 0
	v_mul_hi_u32 v0, v49, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[46:47], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v46
	v_mad_u64_u32 v[42:43], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v47, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[42:43]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v42, s78, v17
	v_mad_u64_u32 v[16:17], s[14:15], s78, v16, 0
	v_add3_u32 v0, v17, v42, v0
	v_sub_u32_e32 v17, v10, v0
	v_sub_co_u32_e32 v16, vcc, v49, v16
	s_nop 1
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, vcc
	v_subrev_co_u32_e64 v42, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[22:23], 0, v17, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v43
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v46, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v42
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v43
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v46, v46, v47, s[22:23]
	v_subrev_co_u32_e64 v47, s[14:15], s78, v42
	s_nop 1
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v17, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v46
	s_nop 1
	v_cndmask_b32_e64 v17, v43, v17, s[14:15]
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v42, v47, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v10, v16, v10, vcc
	v_cndmask_b32_e32 v0, v0, v17, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v11, v48, s33
	v_add3_u32 v63, v0, v10, v11
	v_or_b32_e32 v10, 14, v2
	v_sub_u32_e32 v0, -15, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v16, 31, v0
	v_cndmask_b32_e64 v17, v11, v16, s[10:11]
	v_cndmask_b32_e64 v16, v10, v0, s[10:11]
	v_mov_b32_e32 v42, v17
	v_mov_b32_e32 v43, v17
	v_lshl_add_u64 v[42:43], v[16:17], 0, v[42:43]
	v_xor_b32_e32 v58, v42, v17
	v_xor_b32_e32 v16, v43, v17
	v_mad_u64_u32 v[42:43], s[14:15], v58, s1, 0
	v_mul_hi_u32 v0, v58, s0
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[42:43]
	v_mad_u64_u32 v[48:49], s[14:15], v16, s0, 0
	v_add_co_u32_e32 v0, vcc, v42, v48
	v_mad_u64_u32 v[46:47], s[14:15], v16, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v49, vcc
	v_mov_b32_e32 v43, s63
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	v_mul_lo_u32 v0, s79, v42
	v_mul_lo_u32 v43, s78, v43
	v_mad_u64_u32 v[46:47], s[14:15], s78, v42, 0
	v_add3_u32 v0, v47, v43, v0
	v_sub_u32_e32 v43, v16, v0
	v_sub_co_u32_e32 v46, vcc, v58, v46
	s_nop 1
	v_subb_co_u32_e64 v43, s[14:15], v43, v4, vcc
	v_subrev_co_u32_e64 v47, s[14:15], s78, v46
	v_subb_co_u32_e32 v0, vcc, v16, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[14:15], 0, v43, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s79, v43
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v48, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v47
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v46
	v_cndmask_b32_e64 v47, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v43
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v43, v48, v47, s[14:15]
	v_add_u32_e32 v47, 2, v42
	v_add_u32_e32 v48, 1, v42
	v_cmp_ne_u32_e64 s[14:15], 0, v43
	v_cndmask_b32_e32 v0, v16, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v43, v48, v47, s[14:15]
	v_xor_b32_e32 v16, s72, v17
	v_cndmask_b32_e32 v0, v42, v43, vcc
	v_xor_b32_e32 v0, v0, v16
	v_sub_u32_e32 v0, v0, v16
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_xor_b32_e32 v49, v16, v11
	v_xad_u32 v48, v0, v5, v37
	v_xor_b32_e32 v10, v17, v11
	v_mad_u64_u32 v[16:17], s[14:15], v49, s1, 0
	v_mul_hi_u32 v0, v49, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[46:47], s[14:15], v10, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v46
	v_mad_u64_u32 v[42:43], s[14:15], v10, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v47, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[42:43]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v42, s78, v17
	v_mad_u64_u32 v[16:17], s[14:15], s78, v16, 0
	v_add3_u32 v0, v17, v42, v0
	v_sub_u32_e32 v17, v10, v0
	v_sub_co_u32_e32 v16, vcc, v49, v16
	s_nop 1
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, vcc
	v_subrev_co_u32_e64 v42, s[14:15], s78, v16
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[22:23], 0, v17, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s79, v43
	v_subb_co_u32_e64 v17, s[14:15], v17, v4, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v46, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s78, v42
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s79, v43
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v46, v46, v47, s[22:23]
	v_subrev_co_u32_e64 v47, s[14:15], s78, v42
	s_nop 1
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v17, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v46
	s_nop 1
	v_cndmask_b32_e64 v17, v43, v17, s[14:15]
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v42, v47, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v10, v16, v10, vcc
	v_cndmask_b32_e32 v0, v0, v17, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_mul_lo_u32 v11, v48, s33
	v_add3_u32 v65, v0, v10, v11
	v_or_b32_e32 v10, 15, v2
	v_sub_u32_e32 v0, -16, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v2, 31, v0
	v_cndmask_b32_e64 v17, v11, v2, s[10:11]
	v_cndmask_b32_e64 v16, v10, v0, s[10:11]
	v_mov_b32_e32 v42, v17
	v_mov_b32_e32 v43, v17
	v_lshl_add_u64 v[42:43], v[16:17], 0, v[42:43]
	v_xor_b32_e32 v16, v42, v17
	v_xor_b32_e32 v2, v43, v17
	v_mad_u64_u32 v[42:43], s[10:11], v16, s1, 0
	v_mul_hi_u32 v0, v16, s0
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[42:43]
	v_mad_u64_u32 v[48:49], s[10:11], v2, s0, 0
	v_add_co_u32_e32 v0, vcc, v42, v48
	v_mad_u64_u32 v[46:47], s[10:11], v2, s1, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v49, vcc
	v_mov_b32_e32 v43, s63
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	v_mul_lo_u32 v0, s79, v42
	v_mul_lo_u32 v43, s78, v43
	v_mad_u64_u32 v[46:47], s[10:11], s78, v42, 0
	v_add3_u32 v0, v47, v43, v0
	v_sub_u32_e32 v43, v2, v0
	v_sub_co_u32_e32 v16, vcc, v16, v46
	s_nop 1
	v_subb_co_u32_e64 v43, s[10:11], v43, v4, vcc
	v_subrev_co_u32_e64 v46, s[10:11], s78, v16
	v_subb_co_u32_e32 v0, vcc, v2, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[10:11], 0, v43, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s79, v43
	v_cmp_le_u32_e32 vcc, s79, v0
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s78, v46
	v_cndmask_b32_e64 v2, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v16
	v_cndmask_b32_e64 v46, 0, -1, s[10:11]
	v_cmp_eq_u32_e64 s[10:11], s79, v43
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	v_cndmask_b32_e64 v43, v47, v46, s[10:11]
	v_add_u32_e32 v46, 2, v42
	v_add_u32_e32 v47, 1, v42
	v_cmp_ne_u32_e64 s[10:11], 0, v43
	v_cndmask_b32_e32 v0, v2, v16, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v43, v47, v46, s[10:11]
	v_xor_b32_e32 v2, s72, v17
	v_cndmask_b32_e32 v0, v42, v43, vcc
	v_mov_b32_e32 v16, v11
	v_mov_b32_e32 v17, v11
	v_xor_b32_e32 v0, v0, v2
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_sub_u32_e32 v0, v0, v2
	v_xor_b32_e32 v10, v16, v11
	v_xad_u32 v2, v0, v5, v37
	v_xor_b32_e32 v5, v17, v11
	v_mad_u64_u32 v[16:17], s[10:11], v10, s1, 0
	v_mul_hi_u32 v0, v10, s0
	v_lshl_add_u64 v[16:17], v[0:1], 0, v[16:17]
	v_mad_u64_u32 v[42:43], s[10:11], v5, s1, 0
	v_mad_u64_u32 v[46:47], s[0:1], v5, s0, 0
	v_add_co_u32_e32 v0, vcc, v16, v46
	v_mul_lo_u32 v2, v2, s33
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v47, vcc
	v_mov_b32_e32 v17, s63
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[42:43]
	v_mul_lo_u32 v0, s79, v16
	v_mul_lo_u32 v42, s78, v17
	v_mad_u64_u32 v[16:17], s[0:1], s78, v16, 0
	v_add3_u32 v0, v17, v42, v0
	v_sub_u32_e32 v17, v5, v0
	v_sub_co_u32_e32 v10, vcc, v10, v16
	s_nop 1
	v_subb_co_u32_e64 v16, s[10:11], v17, v4, vcc
	v_subrev_co_u32_e64 v17, s[10:11], s78, v10
	v_subb_co_u32_e32 v0, vcc, v5, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v42, s[14:15], 0, v16, s[10:11]
	v_cmp_le_u32_e64 s[14:15], s79, v42
	v_subb_co_u32_e64 v4, s[10:11], v16, v4, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v43, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s78, v17
	v_subrev_co_u32_e64 v16, s[10:11], s78, v17
	s_nop 0
	v_cndmask_b32_e64 v46, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s79, v42
	v_subbrev_co_u32_e64 v4, s[10:11], 0, v4, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v43, v43, v46, s[14:15]
	v_cmp_le_u32_e32 vcc, s79, v0
	v_cmp_ne_u32_e64 s[10:11], 0, v43
	s_nop 0
	v_cndmask_b32_e64 v5, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s78, v10
	v_cndmask_b32_e64 v4, v42, v4, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v42, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s79, v0
	s_nop 1
	v_cndmask_b32_e32 v5, v5, v42, vcc
	v_cmp_ne_u32_e32 vcc, 0, v5
	s_nop 1
	v_cndmask_b32_e32 v0, v0, v4, vcc
	v_cndmask_b32_e64 v4, v17, v16, s[10:11]
	v_cndmask_b32_e32 v4, v10, v4, vcc
	v_xor_b32_e32 v4, v4, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v4, vcc, v4, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s44, v0
	v_add3_u32 v0, v0, v4, v2
	buffer_load_ubyte v58, v6, s[36:39], 0 offen
	buffer_load_ubyte v60, v7, s[36:39], 0 offen
	buffer_load_ubyte v59, v8, s[36:39], 0 offen
	buffer_load_ubyte v61, v9, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v62, v62, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v64, v63, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v63, v65, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v65, v0, s[36:39], 0 offen
	s_add_u32 s10, s46, 7
	s_addc_u32 s11, s47, 0
	v_lshlrev_b32_e32 v70, 2, v96
	s_movk_i32 s0, 0xffc0
	v_mul_i32_i24_e32 v4, -16, v45
	s_sub_u32 s14, -8, s46
	v_mad_i32_i24 v8, v45, s0, v70
	v_ashrrev_i32_e32 v5, 31, v4
	v_add_u32_e32 v0, v97, v96
	s_subb_u32 s15, -1, s47
	v_cmp_lt_i64_e64 s[0:1], s[46:47], -7
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[4:5]
	v_bfe_u32 v2, v6, 4, 1
	v_cndmask_b32_e64 v0, 0, -1, s[0:1]
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s1, s15, s11
	s_cselect_b32 s0, s14, s10
	s_ashr_i32 s10, s1, 31
	s_lshr_b32 s10, s10, 29
	s_add_u32 s0, s0, s10
	s_addc_u32 s1, s1, 0
	v_readfirstlane_b32 s10, v0
	v_or_b32_e32 v42, v8, v2
	s_ashr_i64 s[0:1], s[0:1], 3
	s_mov_b32 s11, s10
	v_lshlrev_b32_e32 v46, 6, v98
	s_xor_b64 s[22:23], s[0:1], s[10:11]
	v_add_u32_e32 v4, v42, v46
	s_lshl_b64 s[78:79], s[22:23], 3
	v_ashrrev_i32_e32 v5, 31, v4
	v_or_b32_e32 v0, s79, v5
	v_cmp_ne_u32_e32 vcc, 0, v0
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[82:83], exec, s[0:1]
	s_cbranch_execz .LBB0_16
	s_ashr_i32 s0, s79, 31
	s_add_u32 s10, s78, s0
	s_mov_b32 s1, s0
	s_addc_u32 s11, s79, s0
	s_xor_b64 s[84:85], s[10:11], s[0:1]
	v_cvt_f32_u32_e32 v0, s84
	v_cvt_f32_u32_e32 v8, s85
	s_sub_u32 s0, 0, s84
	s_subb_u32 s1, 0, s85
	v_ashrrev_i32_e32 v10, 31, v5
	v_fmac_f32_e32 v0, 0x4f800000, v8
	v_rcp_f32_e32 v0, v0
	v_mov_b32_e32 v11, v10
	v_mul_f32_e32 v0, 0x5f7ffffc, v0
	v_mul_f32_e32 v8, 0x2f800000, v0
	v_trunc_f32_e32 v8, v8
	v_fmac_f32_e32 v0, 0xcf800000, v8
	v_cvt_u32_f32_e32 v8, v8
	v_cvt_u32_f32_e32 v0, v0
	v_readfirstlane_b32 s10, v8
	v_readfirstlane_b32 s11, v0
	s_mul_i32 s14, s0, s10
	s_mul_hi_u32 s56, s0, s11
	s_mul_i32 s15, s1, s11
	s_add_i32 s14, s56, s14
	s_add_i32 s14, s14, s15
	s_mul_i32 s57, s0, s11
	s_mul_i32 s56, s11, s14
	s_mul_hi_u32 s58, s11, s57
	s_mul_hi_u32 s15, s11, s14
	s_add_u32 s56, s58, s56
	s_addc_u32 s15, 0, s15
	s_mul_hi_u32 s59, s10, s57
	s_mul_i32 s57, s10, s57
	s_add_u32 s56, s56, s57
	s_mul_hi_u32 s58, s10, s14
	s_addc_u32 s15, s15, s59
	s_addc_u32 s56, s58, 0
	s_mul_i32 s14, s10, s14
	s_add_u32 s14, s15, s14
	s_addc_u32 s15, 0, s56
	s_add_u32 s11, s11, s14
	s_addc_u32 s10, s10, s15
	s_mul_i32 s14, s0, s10
	s_mul_hi_u32 s15, s0, s11
	s_add_i32 s14, s15, s14
	s_mul_i32 s1, s1, s11
	s_add_i32 s14, s14, s1
	s_mul_i32 s0, s0, s11
	s_mul_hi_u32 s15, s10, s0
	s_mul_i32 s56, s10, s0
	s_mul_i32 s58, s11, s14
	s_mul_hi_u32 s0, s11, s0
	s_mul_hi_u32 s57, s11, s14
	s_add_u32 s0, s0, s58
	s_addc_u32 s57, 0, s57
	s_add_u32 s0, s0, s56
	s_mul_hi_u32 s1, s10, s14
	s_addc_u32 s0, s57, s15
	s_addc_u32 s1, s1, 0
	s_mul_i32 s14, s10, s14
	s_add_u32 s0, s0, s14
	s_addc_u32 s1, 0, s1
	s_add_u32 s11, s11, s0
	v_lshl_add_u64 v[8:9], v[4:5], 0, v[10:11]
	s_addc_u32 s10, s10, s1
	v_xor_b32_e32 v11, v8, v10
	v_xor_b32_e32 v5, v9, v10
	v_mad_u64_u32 v[8:9], s[0:1], v11, s10, 0
	v_mul_hi_u32 v0, v11, s11
	v_lshl_add_u64 v[8:9], v[0:1], 0, v[8:9]
	v_mad_u64_u32 v[48:49], s[0:1], v5, s11, 0
	v_add_co_u32_e32 v0, vcc, v8, v48
	v_mad_u64_u32 v[16:17], s[0:1], v5, s10, 0
	s_nop 0
	v_addc_co_u32_e32 v8, vcc, v9, v49, vcc
	v_mov_b32_e32 v9, s63
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[16:17]
	v_mul_lo_u32 v0, s85, v8
	v_mul_lo_u32 v16, s84, v9
	v_mad_u64_u32 v[8:9], s[0:1], s84, v8, 0
	v_add3_u32 v0, v9, v16, v0
	v_sub_u32_e32 v9, v5, v0
	v_mov_b32_e32 v16, s85
	v_sub_co_u32_e32 v8, vcc, v11, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[10:11], v9, v16, vcc
	v_subrev_co_u32_e64 v11, s[10:11], s84, v8
	v_subb_co_u32_e32 v0, vcc, v5, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v9, s[10:11]
	v_cmp_le_u32_e64 s[14:15], s85, v17
	v_subb_co_u32_e64 v9, s[10:11], v9, v16, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v43, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s84, v11
	v_subrev_co_u32_e64 v16, s[10:11], s84, v11
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s85, v17
	v_subbrev_co_u32_e64 v9, s[10:11], 0, v9, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v43, v43, v47, s[14:15]
	v_cmp_le_u32_e32 vcc, s85, v0
	v_cmp_ne_u32_e64 s[10:11], 0, v43
	s_nop 0
	v_cndmask_b32_e64 v5, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s84, v8
	v_cndmask_b32_e64 v9, v17, v9, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s85, v0
	s_nop 1
	v_cndmask_b32_e32 v5, v5, v17, vcc
	v_cmp_ne_u32_e32 vcc, 0, v5
	v_cndmask_b32_e64 v5, v11, v16, s[10:11]
	s_nop 0
	v_cndmask_b32_e32 v5, v8, v5, vcc
	v_cndmask_b32_e32 v0, v0, v9, vcc
	v_xor_b32_e32 v5, v5, v10
	v_xor_b32_e32 v0, v0, v10
	v_sub_co_u32_e32 v8, vcc, v5, v10
	s_nop 1
	v_subb_co_u32_e32 v9, vcc, v0, v10, vcc
.LBB0_16:
	s_or_saveexec_b64 s[10:11], s[82:83]
	v_cvt_f32_u32_e32 v100, s78
	s_xor_b64 exec, exec, s[10:11]
	s_cbranch_execz .LBB0_18
	v_rcp_iflag_f32_e32 v0, v100
	s_sub_i32 s0, 0, s78
	v_mov_b32_e32 v9, 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	v_mul_lo_u32 v5, s0, v0
	v_mul_hi_u32 v5, v0, v5
	v_add_u32_e32 v0, v0, v5
	v_mul_hi_u32 v0, v4, v0
	v_mul_lo_u32 v0, v0, s78
	v_sub_u32_e32 v0, v4, v0
	v_subrev_u32_e32 v5, s78, v0
	v_cmp_le_u32_e32 vcc, s78, v0
	s_nop 1
	v_cndmask_b32_e32 v0, v0, v5, vcc
	v_subrev_u32_e32 v5, s78, v0
	v_cmp_le_u32_e32 vcc, s78, v0
	s_nop 1
	v_cndmask_b32_e32 v8, v0, v5, vcc
.LBB0_18:
	s_or_b64 exec, exec, s[10:11]
	v_lshl_add_u64 v[6:7], v[6:7], 0, s[68:69]
	v_ashrrev_i32_e32 v0, 31, v7
	v_xor_b32_e32 v7, v0, v7
	v_ashrrev_i32_e32 v5, 31, v7
	v_mov_b32_e32 v47, 0
	v_xor_b32_e32 v6, v0, v6
	v_lshrrev_b32_e32 v10, 27, v5
	v_mov_b32_e32 v11, v47
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[10:11]
	v_ashrrev_i64 v[6:7], 5, v[6:7]
	v_xor_b32_e32 v72, v6, v0
	s_lshl_b32 s73, s22, 8
	s_lshr_b64 s[0:1], s[22:23], 24
	s_ashr_i32 s82, s79, 31
	v_mul_lo_u32 v74, s0, v72
	s_add_u32 s0, s78, s82
	s_mov_b32 s83, s82
	s_addc_u32 s1, s79, s82
	v_xor_b32_e32 v5, v7, v0
	s_xor_b64 s[14:15], s[0:1], s[82:83]
	v_mul_lo_u32 v75, s73, v5
	v_cvt_f32_u32_e32 v0, s14
	v_cvt_f32_u32_e32 v5, s15
	v_ashrrev_i32_e32 v43, 31, v42
	v_mad_u64_u32 v[48:49], s[0:1], s73, v72, v[42:43]
	v_fmac_f32_e32 v0, 0x4f800000, v5
	v_rcp_f32_e32 v5, v0
	s_sub_u32 s0, 0, s14
	s_subb_u32 s1, 0, s15
	v_add3_u32 v49, v74, v49, v75
	v_mul_f32_e32 v5, 0x5f7ffffc, v5
	v_mul_f32_e32 v16, 0x2f800000, v5
	v_trunc_f32_e32 v16, v16
	v_fmac_f32_e32 v5, 0xcf800000, v16
	v_cvt_u32_f32_e32 v5, v5
	v_cvt_u32_f32_e32 v16, v16
	v_lshl_add_u64 v[6:7], v[48:49], 0, v[46:47]
	v_ashrrev_i32_e32 v0, 31, v7
	v_readfirstlane_b32 s10, v5
	v_readfirstlane_b32 s23, v16
	s_mul_hi_u32 s22, s0, s10
	s_mul_i32 s56, s0, s23
	s_mul_i32 s11, s1, s10
	s_add_i32 s22, s22, s56
	s_add_i32 s22, s22, s11
	s_mul_i32 s57, s0, s10
	s_mul_i32 s56, s10, s22
	s_mul_hi_u32 s58, s10, s57
	s_mul_hi_u32 s11, s10, s22
	s_add_u32 s56, s58, s56
	s_addc_u32 s11, 0, s11
	s_mul_hi_u32 s59, s23, s57
	s_mul_i32 s57, s23, s57
	s_add_u32 s56, s56, s57
	s_mul_hi_u32 s58, s23, s22
	s_addc_u32 s11, s11, s59
	s_addc_u32 s56, s58, 0
	s_mul_i32 s22, s23, s22
	s_add_u32 s11, s11, s22
	s_addc_u32 s22, 0, s56
	s_add_u32 s10, s10, s11
	s_addc_u32 s11, s23, s22
	s_mul_i32 s22, s0, s11
	s_mul_hi_u32 s23, s0, s10
	s_add_i32 s22, s23, s22
	s_mul_i32 s1, s1, s10
	s_add_i32 s22, s22, s1
	s_mul_i32 s0, s0, s10
	s_mul_hi_u32 s23, s11, s0
	s_mul_i32 s56, s11, s0
	s_mul_i32 s58, s10, s22
	s_mul_hi_u32 s0, s10, s0
	s_mul_hi_u32 s57, s10, s22
	s_add_u32 s0, s0, s58
	s_addc_u32 s57, 0, s57
	s_add_u32 s0, s0, s56
	s_mul_hi_u32 s1, s11, s22
	s_addc_u32 s0, s57, s23
	v_xor_b32_e32 v11, v0, v7
	s_addc_u32 s1, s1, 0
	s_mul_i32 s22, s11, s22
	s_add_u32 s0, s0, s22
	v_ashrrev_i32_e32 v16, 31, v11
	v_xor_b32_e32 v10, v0, v6
	s_addc_u32 s1, 0, s1
	v_mov_b32_e32 v17, v16
	s_add_u32 s10, s10, s0
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[16:17]
	s_addc_u32 s11, s11, s1
	v_xor_b32_e32 v17, v10, v16
	v_xor_b32_e32 v5, v11, v16
	v_mad_u64_u32 v[10:11], s[0:1], v17, s11, 0
	v_mul_hi_u32 v68, v17, s10
	v_mov_b32_e32 v69, v47
	v_lshl_add_u64 v[10:11], v[68:69], 0, v[10:11]
	v_mad_u64_u32 v[76:77], s[0:1], v5, s10, 0
	v_add_co_u32_e32 v10, vcc, v10, v76
	v_mad_u64_u32 v[68:69], s[0:1], v5, s11, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v77, vcc
	v_mov_b32_e32 v11, v47
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[68:69]
	v_mul_lo_u32 v67, s15, v10
	v_mul_lo_u32 v11, s14, v11
	v_mad_u64_u32 v[68:69], s[0:1], s14, v10, 0
	v_add3_u32 v11, v69, v11, v67
	v_sub_u32_e32 v67, v5, v11
	v_mov_b32_e32 v69, s15
	v_sub_co_u32_e32 v17, vcc, v17, v68
	v_ashrrev_i32_e32 v9, 31, v9
	s_nop 0
	v_subb_co_u32_e64 v67, s[10:11], v67, v69, vcc
	v_subrev_co_u32_e64 v68, s[10:11], s14, v17
	v_subb_co_u32_e32 v5, vcc, v5, v11, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[10:11], 0, v67, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s15, v67
	v_cmp_le_u32_e32 vcc, s15, v5
	v_and_b32_e32 v9, s78, v9
	v_cndmask_b32_e64 v69, 0, -1, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s14, v68
	v_cndmask_b32_e64 v11, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s14, v17
	v_cndmask_b32_e64 v68, 0, -1, s[10:11]
	v_cmp_eq_u32_e64 s[10:11], s15, v67
	v_cndmask_b32_e64 v17, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s15, v5
	v_cndmask_b32_e64 v67, v69, v68, s[10:11]
	v_add_u32_e32 v68, 2, v10
	v_add_u32_e32 v69, 1, v10
	v_cmp_ne_u32_e64 s[10:11], 0, v67
	v_cndmask_b32_e32 v5, v11, v17, vcc
	v_cmp_ne_u32_e32 vcc, 0, v5
	v_cndmask_b32_e64 v67, v69, v68, s[10:11]
	s_mov_b32 s11, 0x27000
	v_cndmask_b32_e32 v5, v10, v67, vcc
	v_xor_b32_e32 v10, s82, v16
	v_xor_b32_e32 v5, v5, v10
	v_sub_u32_e32 v5, v5, v10
	v_xor_b32_e32 v5, v5, v0
	v_mul_lo_u32 v5, v5, s66
	v_add3_u32 v5, v5, v8, v9
	s_mov_b32 s10, 0x7ffffffe
	s_and_b32 s9, s9, 0xffff
	buffer_load_ubyte v92, v5, s[8:11], 0 offen
	v_or_b32_e32 v5, 2, v6
	v_sub_co_u32_e32 v6, vcc, -3, v6
	s_nop 1
	v_subb_co_u32_e32 v8, vcc, -1, v7, vcc
	v_cmp_gt_i32_e32 vcc, 0, v7
	s_nop 1
	v_cndmask_b32_e32 v9, v7, v8, vcc
	v_cndmask_b32_e32 v8, v5, v6, vcc
	v_or_b32_e32 v5, s79, v9
	v_cmp_ne_u32_e32 vcc, 0, v5
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[22:23], exec, s[0:1]
	s_cbranch_execz .LBB0_20
	s_add_u32 s0, s78, s82
	s_addc_u32 s1, s79, s82
	s_xor_b64 s[62:63], s[0:1], s[82:83]
	v_cvt_f32_u32_e32 v5, s62
	v_cvt_f32_u32_e32 v6, s63
	s_sub_u32 s0, 0, s62
	s_subb_u32 s1, 0, s63
	v_mov_b32_e32 v11, v47
	v_fmac_f32_e32 v5, 0x4f800000, v6
	v_rcp_f32_e32 v5, v5
	s_nop 0
	v_mul_f32_e32 v5, 0x5f7ffffc, v5
	v_mul_f32_e32 v6, 0x2f800000, v5
	v_trunc_f32_e32 v6, v6
	v_fmac_f32_e32 v5, 0xcf800000, v6
	v_cvt_u32_f32_e32 v6, v6
	v_cvt_u32_f32_e32 v5, v5
	v_readfirstlane_b32 s14, v6
	v_readfirstlane_b32 s15, v5
	s_mul_i32 s56, s0, s14
	s_mul_hi_u32 s58, s0, s15
	s_mul_i32 s57, s1, s15
	s_add_i32 s56, s58, s56
	s_mul_i32 s59, s0, s15
	s_add_i32 s56, s56, s57
	s_mul_i32 s58, s15, s56
	s_mul_hi_u32 s65, s15, s59
	s_mul_hi_u32 s57, s15, s56
	s_add_u32 s58, s65, s58
	s_addc_u32 s57, 0, s57
	s_mul_hi_u32 s68, s14, s59
	s_mul_i32 s59, s14, s59
	s_add_u32 s58, s58, s59
	s_mul_hi_u32 s65, s14, s56
	s_addc_u32 s57, s57, s68
	s_addc_u32 s58, s65, 0
	s_mul_i32 s56, s14, s56
	s_add_u32 s56, s57, s56
	s_addc_u32 s57, 0, s58
	s_add_u32 s15, s15, s56
	s_addc_u32 s14, s14, s57
	s_mul_i32 s56, s0, s14
	s_mul_hi_u32 s57, s0, s15
	s_add_i32 s56, s57, s56
	s_mul_i32 s1, s1, s15
	s_add_i32 s56, s56, s1
	s_mul_i32 s0, s0, s15
	s_mul_hi_u32 s57, s14, s0
	s_mul_i32 s58, s14, s0
	s_mul_i32 s65, s15, s56
	s_mul_hi_u32 s0, s15, s0
	s_mul_hi_u32 s59, s15, s56
	s_add_u32 s0, s0, s65
	s_addc_u32 s59, 0, s59
	s_add_u32 s0, s0, s58
	s_mul_hi_u32 s1, s14, s56
	s_addc_u32 s0, s59, s57
	s_addc_u32 s1, s1, 0
	s_mul_i32 s56, s14, s56
	s_add_u32 s0, s0, s56
	v_ashrrev_i32_e32 v6, 31, v9
	s_addc_u32 s1, 0, s1
	v_mov_b32_e32 v7, v6
	s_add_u32 s15, s15, s0
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[6:7]
	s_addc_u32 s14, s14, s1
	v_xor_b32_e32 v7, v8, v6
	v_xor_b32_e32 v5, v9, v6
	v_mad_u64_u32 v[8:9], s[0:1], v7, s14, 0
	v_mul_hi_u32 v10, v7, s15
	v_lshl_add_u64 v[8:9], v[10:11], 0, v[8:9]
	v_mad_u64_u32 v[16:17], s[0:1], v5, s15, 0
	v_mad_u64_u32 v[10:11], s[0:1], v5, s14, 0
	v_add_co_u32_e32 v8, vcc, v8, v16
	s_mov_b32 s0, 0
	s_nop 0
	v_addc_co_u32_e32 v8, vcc, v9, v17, vcc
	v_mov_b32_e32 v9, s0
	s_nop 0
	v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[10:11]
	v_mul_lo_u32 v16, s63, v8
	v_mul_lo_u32 v17, s62, v9
	v_mad_u64_u32 v[10:11], s[0:1], s62, v8, 0
	v_add3_u32 v67, v11, v17, v16
	v_sub_u32_e32 v11, v5, v67
	v_mov_b32_e32 v16, s63
	v_sub_co_u32_e32 v7, vcc, v7, v10
	v_xor_b32_e32 v6, s82, v6
	s_nop 0
	v_subb_co_u32_e64 v10, s[14:15], v11, v16, vcc
	v_subrev_co_u32_e64 v11, s[14:15], s62, v7
	v_subb_co_u32_e32 v5, vcc, v5, v67, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v10, s[14:15], 0, v10, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s63, v10
	v_cmp_le_u32_e32 vcc, s63, v5
	s_nop 0
	v_cndmask_b32_e64 v16, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v11
	s_nop 1
	v_cndmask_b32_e64 v11, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v10
	s_nop 1
	v_cndmask_b32_e64 v68, v16, v11, s[14:15]
	v_lshl_add_u64 v[10:11], v[8:9], 0, 2
	v_lshl_add_u64 v[16:17], v[8:9], 0, 1
	v_cndmask_b32_e64 v9, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s62, v7
	s_nop 1
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v5
	s_nop 1
	v_cndmask_b32_e32 v5, v9, v7, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e32 v7, v16, v10, vcc
	v_cmp_ne_u32_e32 vcc, 0, v5
	s_nop 1
	v_cndmask_b32_e32 v5, v8, v7, vcc
	v_xor_b32_e32 v5, v5, v6
	v_sub_co_u32_e32 v6, vcc, v5, v6
.LBB0_20:
	s_andn2_saveexec_b64 s[14:15], s[22:23]
	s_cbranch_execz .LBB0_22
	v_rcp_iflag_f32_e32 v5, v100
	s_sub_i32 s0, 0, s78
	v_mul_f32_e32 v5, 0x4f7ffffe, v5
	v_cvt_u32_f32_e32 v5, v5
	v_mul_lo_u32 v6, s0, v5
	v_mul_hi_u32 v6, v5, v6
	v_add_u32_e32 v5, v5, v6
	v_mul_hi_u32 v5, v8, v5
	v_mul_lo_u32 v6, v5, s78
	v_sub_u32_e32 v6, v8, v6
	v_add_u32_e32 v7, 1, v5
	v_subrev_u32_e32 v8, s78, v6
	v_cmp_le_u32_e32 vcc, s78, v6
	s_nop 1
	v_cndmask_b32_e32 v6, v6, v8, vcc
	v_cndmask_b32_e32 v5, v5, v7, vcc
	v_add_u32_e32 v7, 1, v5
	v_cmp_le_u32_e32 vcc, s78, v6
	s_nop 1
	v_cndmask_b32_e32 v6, v5, v7, vcc
.LBB0_22:
	s_or_b64 exec, exec, s[14:15]
	v_or_b32_e32 v4, 2, v4
	v_ashrrev_i32_e32 v5, 31, v4
	v_or_b32_e32 v7, s79, v5
	v_cmp_ne_u32_e32 vcc, 0, v7
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[62:63], exec, s[0:1]
	s_cbranch_execz .LBB0_24
	s_add_u32 s0, s78, s82
	s_mov_b32 s83, s82
	s_addc_u32 s1, s79, s82
	s_xor_b64 s[68:69], s[0:1], s[82:83]
	v_cvt_f32_u32_e32 v7, s68
	v_cvt_f32_u32_e32 v8, s69
	s_sub_u32 s0, 0, s68
	s_subb_u32 s1, 0, s69
	v_ashrrev_i32_e32 v10, 31, v5
	v_fmamk_f32 v7, v8, 0x4f800000, v7
	v_rcp_f32_e32 v7, v7
	v_mov_b32_e32 v11, v10
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[10:11]
	v_xor_b32_e32 v11, v4, v10
	v_mul_f32_e32 v7, 0x5f7ffffc, v7
	v_mul_f32_e32 v8, 0x2f800000, v7
	v_trunc_f32_e32 v8, v8
	v_fmamk_f32 v7, v8, 0xcf800000, v7
	v_cvt_u32_f32_e32 v8, v8
	v_cvt_u32_f32_e32 v7, v7
	v_mov_b32_e32 v9, 0
	v_readfirstlane_b32 s14, v8
	v_readfirstlane_b32 s15, v7
	s_mul_i32 s22, s0, s14
	s_mul_hi_u32 s56, s0, s15
	s_mul_i32 s23, s1, s15
	s_add_i32 s22, s56, s22
	s_add_i32 s22, s22, s23
	s_mul_i32 s57, s0, s15
	s_mul_i32 s56, s15, s22
	s_mul_hi_u32 s58, s15, s57
	s_mul_hi_u32 s23, s15, s22
	s_add_u32 s56, s58, s56
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 s59, s14, s57
	s_mul_i32 s57, s14, s57
	s_add_u32 s56, s56, s57
	s_mul_hi_u32 s58, s14, s22
	s_addc_u32 s23, s23, s59
	s_addc_u32 s56, s58, 0
	s_mul_i32 s22, s14, s22
	s_add_u32 s22, s23, s22
	s_addc_u32 s23, 0, s56
	s_add_u32 s15, s15, s22
	s_addc_u32 s14, s14, s23
	s_mul_i32 s22, s0, s14
	s_mul_hi_u32 s23, s0, s15
	s_add_i32 s22, s23, s22
	s_mul_i32 s1, s1, s15
	s_add_i32 s22, s22, s1
	s_mul_i32 s0, s0, s15
	s_mul_hi_u32 s23, s14, s0
	s_mul_i32 s56, s14, s0
	s_mul_i32 s58, s15, s22
	s_mul_hi_u32 s0, s15, s0
	s_mul_hi_u32 s57, s15, s22
	s_add_u32 s0, s0, s58
	s_addc_u32 s57, 0, s57
	s_add_u32 s0, s0, s56
	s_mul_hi_u32 s1, s14, s22
	s_addc_u32 s0, s57, s23
	s_addc_u32 s1, s1, 0
	s_mul_i32 s22, s14, s22
	s_add_u32 s0, s0, s22
	s_addc_u32 s1, 0, s1
	s_add_u32 s15, s15, s0
	s_addc_u32 s14, s14, s1
	v_xor_b32_e32 v7, v5, v10
	v_mad_u64_u32 v[4:5], s[0:1], v11, s14, 0
	v_mul_hi_u32 v8, v11, s15
	v_lshl_add_u64 v[4:5], v[8:9], 0, v[4:5]
	v_mad_u64_u32 v[16:17], s[0:1], v7, s15, 0
	v_mad_u64_u32 v[8:9], s[0:1], v7, s14, 0
	v_add_co_u32_e32 v4, vcc, v4, v16
	s_mov_b32 s0, 0
	s_nop 0
	v_addc_co_u32_e32 v4, vcc, v5, v17, vcc
	v_mov_b32_e32 v5, s0
	s_nop 0
	v_addc_co_u32_e32 v9, vcc, 0, v9, vcc
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[8:9]
	v_mul_lo_u32 v8, s69, v4
	v_mul_lo_u32 v9, s68, v5
	v_mad_u64_u32 v[4:5], s[0:1], s68, v4, 0
	v_add3_u32 v5, v5, v9, v8
	v_sub_u32_e32 v8, v7, v5
	v_mov_b32_e32 v9, s69
	v_sub_co_u32_e32 v4, vcc, v11, v4
	s_nop 1
	v_subb_co_u32_e64 v8, s[14:15], v8, v9, vcc
	v_subrev_co_u32_e64 v11, s[14:15], s68, v4
	v_subb_co_u32_e32 v5, vcc, v7, v5, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v16, s[22:23], 0, v8, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s69, v16
	v_subb_co_u32_e64 v8, s[14:15], v8, v9, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s68, v11
	v_subrev_co_u32_e64 v9, s[14:15], s68, v11
	s_nop 0
	v_cndmask_b32_e64 v67, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s69, v16
	v_subbrev_co_u32_e64 v8, s[14:15], 0, v8, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v17, v17, v67, s[22:23]
	v_cmp_le_u32_e32 vcc, s69, v5
	v_cmp_ne_u32_e64 s[14:15], 0, v17
	s_nop 0
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v4
	v_cndmask_b32_e64 v8, v16, v8, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v5
	s_nop 1
	v_cndmask_b32_e32 v7, v7, v16, vcc
	v_cmp_ne_u32_e32 vcc, 0, v7
	v_cndmask_b32_e64 v7, v11, v9, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v4, v4, v7, vcc
	v_cndmask_b32_e32 v5, v5, v8, vcc
	v_xor_b32_e32 v4, v4, v10
	v_xor_b32_e32 v5, v5, v10
	v_sub_co_u32_e32 v8, vcc, v4, v10
	s_nop 1
	v_subb_co_u32_e32 v9, vcc, v5, v10, vcc
.LBB0_24:
	s_andn2_saveexec_b64 s[14:15], s[62:63]
	s_cbranch_execz .LBB0_26
	v_rcp_iflag_f32_e32 v5, v100
	s_sub_i32 s0, 0, s78
	v_mov_b32_e32 v9, 0
	v_mul_f32_e32 v5, 0x4f7ffffe, v5
	v_cvt_u32_f32_e32 v5, v5
	v_mul_lo_u32 v7, s0, v5
	v_mul_hi_u32 v7, v5, v7
	v_add_u32_e32 v5, v5, v7
	v_mul_hi_u32 v5, v4, v5
	v_mul_lo_u32 v5, v5, s78
	v_sub_u32_e32 v4, v4, v5
	v_subrev_u32_e32 v5, s78, v4
	v_cmp_le_u32_e32 vcc, s78, v4
	s_nop 1
	v_cndmask_b32_e32 v4, v4, v5, vcc
	v_subrev_u32_e32 v5, s78, v4
	v_cmp_le_u32_e32 vcc, s78, v4
	s_nop 1
	v_cndmask_b32_e32 v8, v4, v5, vcc
.LBB0_26:
	s_or_b64 exec, exec, s[14:15]
	v_xor_b32_e32 v0, v6, v0
	v_ashrrev_i32_e32 v4, 31, v9
	v_mul_lo_u32 v0, v0, s66
	v_and_b32_e32 v4, s78, v4
	v_add3_u32 v0, v0, v8, v4
	buffer_load_ubyte v81, v0, s[8:11], 0 offen
	v_readlane_b32 s0, v144, 10
	v_readlane_b32 s1, v144, 11
	v_readlane_b32 s14, v144, 12
	v_mov_b32_e32 v9, 0
	s_lshl_b32 s0, s0, 7
	s_lshl_b32 s1, s1, 7
	s_lshl_b32 s58, s14, 7
	s_waitcnt vmcnt(0)
	s_barrier
	v_or_b32_e32 v8, 0x100, v32
	v_or_b32_e32 v0, 0x80, v28
	v_cmp_gt_i64_e32 vcc, s[30:31], v[8:9]
	v_add_u32_e32 v4, v0, v30
	v_bfrev_b32_e32 v5, -2
	s_and_b64 s[14:15], vcc, s[2:3]
	s_add_i32 s65, s54, 0x4000
	v_cndmask_b32_e64 v4, v5, v4, s[14:15]
	s_mov_b32 m0, s65
	s_and_b64 s[14:15], vcc, s[26:27]
	buffer_load_dwordx4 v4, s[40:43], 0 offen lds
	v_add_u32_e32 v4, v0, v34
	s_addk_i32 s0, 0x4000
	v_cndmask_b32_e64 v4, v5, v4, s[14:15]
	s_mov_b32 m0, s0
	s_and_b64 s[14:15], vcc, s[18:19]
	buffer_load_dwordx4 v4, s[40:43], 0 offen lds
	v_add_u32_e32 v4, v0, v36
	s_addk_i32 s1, 0x4000
	v_cndmask_b32_e64 v4, v5, v4, s[14:15]
	s_mov_b32 m0, s1
	v_add_u32_e32 v0, v0, v38
	s_and_b64 vcc, vcc, s[20:21]
	s_addk_i32 s58, 0x4000
	buffer_load_dwordx4 v4, s[40:43], 0 offen lds
	v_cndmask_b32_e32 v0, v5, v0, vcc
	s_mov_b32 m0, s58
	s_mov_b64 s[14:15], 0x2000
	buffer_load_dwordx4 v0, s[40:43], 0 offen lds
	v_lshl_add_u64 v[4:5], v[40:41], 0, s[14:15]
	s_movk_i32 s14, 0xdfff
	v_sub_co_u32_e32 v0, vcc, s14, v40
	s_nop 1
	v_subb_co_u32_e32 v6, vcc, -1, v41, vcc
	v_cmp_gt_i32_e32 vcc, 0, v5
	s_nop 1
	v_cndmask_b32_e32 v11, v5, v6, vcc
	v_cndmask_b32_e32 v10, v4, v0, vcc
	v_or_b32_e32 v0, s31, v11
	v_cmp_ne_u32_e32 vcc, 0, v0
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[22:23], exec, s[14:15]
	s_cbranch_execz .LBB0_28
	s_add_u32 s14, s30, s80
	s_mov_b32 s81, s80
	s_addc_u32 s15, s31, s80
	s_xor_b64 s[56:57], s[14:15], s[80:81]
	v_cvt_f32_u32_e32 v0, s56
	v_cvt_f32_u32_e32 v6, s57
	s_sub_u32 s14, 0, s56
	s_subb_u32 s15, 0, s57
	v_fmamk_f32 v0, v6, 0x4f800000, v0
	v_rcp_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x5f7ffffc, v0
	v_mul_f32_e32 v6, 0x2f800000, v0
	v_trunc_f32_e32 v6, v6
	v_fmamk_f32 v0, v6, 0xcf800000, v0
	v_cvt_u32_f32_e32 v6, v6
	v_cvt_u32_f32_e32 v0, v0
	v_readfirstlane_b32 s59, v6
	v_readfirstlane_b32 s62, v0
	s_mul_i32 s63, s14, s59
	s_mul_hi_u32 s69, s14, s62
	s_mul_i32 s68, s15, s62
	s_add_i32 s63, s69, s63
	s_add_i32 s63, s63, s68
	s_mul_i32 s75, s14, s62
	s_mul_i32 s69, s62, s63
	s_mul_hi_u32 s81, s62, s75
	s_mul_hi_u32 s68, s62, s63
	s_add_u32 s69, s81, s69
	s_addc_u32 s68, 0, s68
	s_mul_hi_u32 s83, s59, s75
	s_mul_i32 s75, s59, s75
	s_add_u32 s69, s69, s75
	s_mul_hi_u32 s81, s59, s63
	s_addc_u32 s68, s68, s83
	s_addc_u32 s69, s81, 0
	s_mul_i32 s63, s59, s63
	s_add_u32 s63, s68, s63
	s_addc_u32 s68, 0, s69
	s_add_u32 s62, s62, s63
	s_addc_u32 s59, s59, s68
	s_mul_i32 s63, s14, s59
	s_mul_hi_u32 s68, s14, s62
	s_add_i32 s63, s68, s63
	s_mul_i32 s15, s15, s62
	s_add_i32 s63, s63, s15
	s_mul_i32 s14, s14, s62
	s_mul_hi_u32 s68, s59, s14
	s_mul_i32 s69, s59, s14
	s_mul_i32 s81, s62, s63
	s_mul_hi_u32 s14, s62, s14
	s_mul_hi_u32 s75, s62, s63
	s_add_u32 s14, s14, s81
	s_addc_u32 s75, 0, s75
	s_add_u32 s14, s14, s69
	s_mul_hi_u32 s15, s59, s63
	s_addc_u32 s14, s75, s68
	s_addc_u32 s15, s15, 0
	s_mul_i32 s63, s59, s63
	s_add_u32 s14, s14, s63
	v_ashrrev_i32_e32 v6, 31, v11
	s_addc_u32 s15, 0, s15
	v_mov_b32_e32 v7, v6
	s_add_u32 s62, s62, s14
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[6:7]
	s_addc_u32 s59, s59, s15
	v_xor_b32_e32 v7, v10, v6
	v_xor_b32_e32 v0, v11, v6
	v_mad_u64_u32 v[10:11], s[14:15], v7, s59, 0
	v_mul_hi_u32 v8, v7, s62
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[10:11]
	v_mad_u64_u32 v[16:17], s[14:15], v0, s62, 0
	v_mad_u64_u32 v[10:11], s[14:15], v0, s59, 0
	v_add_co_u32_e32 v8, vcc, v8, v16
	s_mov_b32 s14, 0
	s_nop 0
	v_addc_co_u32_e32 v8, vcc, v9, v17, vcc
	v_mov_b32_e32 v9, s14
	s_nop 0
	v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[10:11]
	v_mul_lo_u32 v16, s57, v8
	v_mul_lo_u32 v17, s56, v9
	v_mad_u64_u32 v[10:11], s[14:15], s56, v8, 0
	v_add3_u32 v67, v11, v17, v16
	v_sub_u32_e32 v11, v0, v67
	v_mov_b32_e32 v16, s57
	v_sub_co_u32_e32 v7, vcc, v7, v10
	v_xor_b32_e32 v6, s80, v6
	s_nop 0
	v_subb_co_u32_e64 v10, s[14:15], v11, v16, vcc
	v_subrev_co_u32_e64 v11, s[14:15], s56, v7
	v_subb_co_u32_e32 v0, vcc, v0, v67, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v10, s[14:15], 0, v10, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s57, v10
	v_cmp_le_u32_e32 vcc, s57, v0
	s_nop 0
	v_cndmask_b32_e64 v16, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s56, v11
	s_nop 1
	v_cndmask_b32_e64 v11, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s57, v10
	s_nop 1
	v_cndmask_b32_e64 v68, v16, v11, s[14:15]
	v_lshl_add_u64 v[10:11], v[8:9], 0, 2
	v_lshl_add_u64 v[16:17], v[8:9], 0, 1
	v_cndmask_b32_e64 v9, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s56, v7
	s_nop 1
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s57, v0
	s_nop 1
	v_cndmask_b32_e32 v0, v9, v7, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e32 v7, v16, v10, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	s_nop 1
	v_cndmask_b32_e32 v0, v8, v7, vcc
	v_xor_b32_e32 v0, v0, v6
	v_sub_co_u32_e32 v6, vcc, v0, v6
.LBB0_28:
	s_andn2_saveexec_b64 s[14:15], s[22:23]
	s_cbranch_execz .LBB0_30
	v_cvt_f32_u32_e32 v0, s30
	s_sub_i32 s22, 0, s30
	v_rcp_iflag_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	v_mul_lo_u32 v6, s22, v0
	v_mul_hi_u32 v6, v0, v6
	v_add_u32_e32 v0, v0, v6
	v_mul_hi_u32 v0, v10, v0
	v_mul_lo_u32 v6, v0, s30
	v_sub_u32_e32 v6, v10, v6
	v_add_u32_e32 v7, 1, v0
	v_subrev_u32_e32 v8, s30, v6
	v_cmp_le_u32_e32 vcc, s30, v6
	s_nop 1
	v_cndmask_b32_e32 v6, v6, v8, vcc
	v_cndmask_b32_e32 v0, v0, v7, vcc
	v_add_u32_e32 v7, 1, v0
	v_cmp_le_u32_e32 vcc, s30, v6
	s_nop 1
	v_cndmask_b32_e32 v6, v0, v7, vcc
.LBB0_30:
	s_or_b64 exec, exec, s[14:15]
	v_or_b32_e32 v0, s31, v5
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_ashrrev_i32_e32 v8, 31, v5
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[62:63], exec, s[14:15]
	s_cbranch_execz .LBB0_32
	s_add_u32 s14, s30, s80
	s_mov_b32 s81, s80
	s_addc_u32 s15, s31, s80
	s_xor_b64 s[56:57], s[14:15], s[80:81]
	v_cvt_f32_u32_e32 v0, s56
	v_cvt_f32_u32_e32 v7, s57
	s_sub_u32 s14, 0, s56
	s_subb_u32 s15, 0, s57
	v_mov_b32_e32 v9, v8
	v_fmamk_f32 v0, v7, 0x4f800000, v0
	v_rcp_f32_e32 v0, v0
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[8:9]
	v_mov_b32_e32 v11, 0
	v_mul_f32_e32 v0, 0x5f7ffffc, v0
	v_mul_f32_e32 v7, 0x2f800000, v0
	v_trunc_f32_e32 v7, v7
	v_fmamk_f32 v0, v7, 0xcf800000, v0
	v_cvt_u32_f32_e32 v7, v7
	v_cvt_u32_f32_e32 v0, v0
	v_readfirstlane_b32 s22, v7
	v_readfirstlane_b32 s23, v0
	s_mul_i32 s59, s14, s22
	s_mul_hi_u32 s69, s14, s23
	s_mul_i32 s68, s15, s23
	s_add_i32 s59, s69, s59
	s_add_i32 s59, s59, s68
	s_mul_i32 s75, s14, s23
	s_mul_i32 s69, s23, s59
	s_mul_hi_u32 s81, s23, s75
	s_mul_hi_u32 s68, s23, s59
	s_add_u32 s69, s81, s69
	s_addc_u32 s68, 0, s68
	s_mul_hi_u32 s83, s22, s75
	s_mul_i32 s75, s22, s75
	s_add_u32 s69, s69, s75
	s_mul_hi_u32 s81, s22, s59
	s_addc_u32 s68, s68, s83
	s_addc_u32 s69, s81, 0
	s_mul_i32 s59, s22, s59
	s_add_u32 s59, s68, s59
	s_addc_u32 s68, 0, s69
	s_add_u32 s23, s23, s59
	s_addc_u32 s22, s22, s68
	s_mul_i32 s59, s14, s22
	s_mul_hi_u32 s68, s14, s23
	s_add_i32 s59, s68, s59
	s_mul_i32 s15, s15, s23
	s_add_i32 s59, s59, s15
	s_mul_i32 s14, s14, s23
	s_mul_hi_u32 s68, s22, s14
	s_mul_i32 s69, s22, s14
	s_mul_i32 s81, s23, s59
	s_mul_hi_u32 s14, s23, s14
	s_mul_hi_u32 s75, s23, s59
	s_add_u32 s14, s14, s81
	s_addc_u32 s75, 0, s75
	s_add_u32 s14, s14, s69
	s_mul_hi_u32 s15, s22, s59
	s_addc_u32 s14, s75, s68
	s_addc_u32 s15, s15, 0
	s_mul_i32 s59, s22, s59
	s_add_u32 s14, s14, s59
	s_addc_u32 s15, 0, s15
	s_add_u32 s23, s23, s14
	s_addc_u32 s22, s22, s15
	v_xor_b32_e32 v7, v4, v8
	v_xor_b32_e32 v0, v5, v8
	v_mad_u64_u32 v[4:5], s[14:15], v7, s22, 0
	v_mul_hi_u32 v10, v7, s23
	v_lshl_add_u64 v[4:5], v[10:11], 0, v[4:5]
	v_mad_u64_u32 v[16:17], s[14:15], v0, s23, 0
	v_mad_u64_u32 v[10:11], s[14:15], v0, s22, 0
	v_add_co_u32_e32 v4, vcc, v4, v16
	s_mov_b32 s14, 0
	s_nop 0
	v_addc_co_u32_e32 v4, vcc, v5, v17, vcc
	v_mov_b32_e32 v5, s14
	s_nop 0
	v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[10:11]
	v_mul_lo_u32 v9, s57, v4
	v_mul_lo_u32 v10, s56, v5
	v_mad_u64_u32 v[4:5], s[14:15], s56, v4, 0
	v_add3_u32 v5, v5, v10, v9
	v_sub_u32_e32 v9, v0, v5
	v_mov_b32_e32 v10, s57
	v_sub_co_u32_e32 v4, vcc, v7, v4
	s_nop 1
	v_subb_co_u32_e64 v7, s[14:15], v9, v10, vcc
	v_subrev_co_u32_e64 v9, s[14:15], s56, v4
	v_subb_co_u32_e32 v0, vcc, v0, v5, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v11, s[22:23], 0, v7, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s57, v11
	v_subb_co_u32_e64 v7, s[14:15], v7, v10, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v16, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s56, v9
	v_subrev_co_u32_e64 v10, s[14:15], s56, v9
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s57, v11
	v_subbrev_co_u32_e64 v7, s[14:15], 0, v7, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v16, v16, v17, s[22:23]
	v_cmp_le_u32_e32 vcc, s57, v0
	v_cmp_ne_u32_e64 s[14:15], 0, v16
	s_nop 0
	v_cndmask_b32_e64 v5, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s56, v4
	v_cndmask_b32_e64 v7, v11, v7, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v11, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s57, v0
	s_nop 1
	v_cndmask_b32_e32 v5, v5, v11, vcc
	v_cmp_ne_u32_e32 vcc, 0, v5
	v_cndmask_b32_e64 v5, v9, v10, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v4, v4, v5, vcc
	v_cndmask_b32_e32 v0, v0, v7, vcc
	v_xor_b32_e32 v4, v4, v8
	v_xor_b32_e32 v0, v0, v8
	v_sub_co_u32_e32 v16, vcc, v4, v8
	s_nop 1
	v_subb_co_u32_e32 v17, vcc, v0, v8, vcc
.LBB0_32:
	s_or_saveexec_b64 s[14:15], s[62:63]
	v_mul_i32_i24_e32 v68, 0xffffff00, v45
	s_xor_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB0_34
	v_cvt_f32_u32_e32 v0, s30
	s_sub_i32 s22, 0, s30
	v_mov_b32_e32 v17, 0
	v_rcp_iflag_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	v_mul_lo_u32 v5, s22, v0
	v_mul_hi_u32 v5, v0, v5
	v_add_u32_e32 v0, v0, v5
	v_mul_hi_u32 v0, v4, v0
	v_mul_lo_u32 v0, v0, s30
	v_sub_u32_e32 v0, v4, v0
	v_subrev_u32_e32 v4, s30, v0
	v_cmp_le_u32_e32 vcc, s30, v0
	s_nop 1
	v_cndmask_b32_e32 v0, v0, v4, vcc
	v_subrev_u32_e32 v4, s30, v0
	v_cmp_le_u32_e32 vcc, s30, v0
	s_nop 1
	v_cndmask_b32_e32 v16, v0, v4, vcc
.LBB0_34:
	s_or_b64 exec, exec, s[14:15]
	v_lshlrev_b16_e32 v4, 8, v35
	v_lshlrev_b16_e32 v5, 8, v39
	v_or_b32_e32 v4, v31, v4
	v_or_b32_sdwa v5, v33, v5 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_xor_b32_e32 v0, v6, v8
	v_or_b32_sdwa v11, v4, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v4, 8, v23
	v_or_b32_e32 v3, v3, v4
	v_lshlrev_b16_e32 v4, 8, v29
	v_or_b32_sdwa v4, v22, v4 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_mov_b32 s14, 0
	v_or_b32_sdwa v10, v3, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v3, 8, v20
	v_lshlrev_b16_e32 v4, 8, v21
	v_or_b32_e32 v3, v18, v3
	v_or_b32_sdwa v4, v19, v4 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_mov_b32_e32 v7, s14
	v_or_b32_sdwa v9, v3, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v3, 8, v14
	v_lshlrev_b16_e32 v4, 8, v15
	v_or_b32_e32 v3, v12, v3
	v_or_b32_sdwa v4, v13, v4 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_cmp_lt_i64_e64 s[14:15], s[28:29], 1
	v_or_b32_sdwa v8, v3, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v3, 8, v64
	v_lshlrev_b16_e32 v4, 8, v65
	v_or_b32_e32 v3, v62, v3
	v_or_b32_sdwa v4, v63, v4 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_and_b64 vcc, s[52:53], s[14:15]
	v_or_b32_sdwa v15, v3, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v3, 8, v60
	v_lshlrev_b16_e32 v4, 8, v61
	v_or_b32_e32 v3, v58, v3
	v_or_b32_sdwa v4, v59, v4 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_cmp_gt_i64_e64 s[14:15], s[28:29], 0
	v_or_b32_sdwa v14, v3, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v3, 8, v56
	v_lshlrev_b16_e32 v4, 8, v57
	v_or_b32_e32 v3, v54, v3
	v_or_b32_sdwa v4, v55, v4 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_add_i32 s59, s67, 0xc400
	v_or_b32_sdwa v13, v3, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v3, 8, v52
	v_lshlrev_b16_e32 v4, 8, v53
	v_or_b32_e32 v3, v50, v3
	v_or_b32_sdwa v4, v51, v4 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_ashr_i32 s22, s51, 31
	v_or_b32_sdwa v12, v3, v4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_ashrrev_i32_e32 v4, 31, v17
	v_and_b32_e32 v5, s31, v4
	v_and_b32_e32 v4, s30, v4
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[16:17]
	v_ashrrev_i32_e32 v16, 31, v5
	v_xor_b32_e32 v5, v16, v5
	v_ashrrev_i32_e32 v6, 31, v5
	v_xor_b32_e32 v4, v16, v4
	v_lshrrev_b32_e32 v6, 27, v6
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[6:7]
	v_cndmask_b32_e64 v6, 0, -1, s[14:15]
	s_mov_b32 s23, s22
	v_readfirstlane_b32 s14, v6
	s_mov_b32 s15, s14
	s_xor_b64 s[28:29], s[24:25], s[14:15]
	s_add_u32 s14, s50, s22
	s_addc_u32 s15, s51, s22
	s_xor_b64 s[24:25], s[14:15], s[22:23]
	v_cvt_f32_u32_e32 v7, s24
	v_cvt_f32_u32_e32 v17, s25
	v_alignbit_b32 v4, v5, v4, 5
	v_mul_lo_u32 v0, v0, s46
	v_xad_u32 v0, v4, v16, v0
	v_fmamk_f32 v4, v17, 0x4f800000, v7
	v_rcp_f32_e32 v4, v4
	s_mov_b32 m0, s59
	s_sub_u32 s52, 0, s24
	buffer_load_dword v0, s[4:7], 0 offen lds
	v_mul_f32_e32 v0, 0x5f7ffffc, v4
	v_mul_f32_e32 v4, 0x2f800000, v0
	v_trunc_f32_e32 v4, v4
	v_fmamk_f32 v0, v4, 0xcf800000, v0
	v_cvt_u32_f32_e32 v0, v0
	v_cvt_u32_f32_e32 v4, v4
	s_subb_u32 s53, 0, s25
	v_and_b32_e32 v3, 7, v96
	v_readfirstlane_b32 s56, v0
	v_readfirstlane_b32 s63, v4
	s_mul_hi_u32 s62, s52, s56
	s_mul_i32 s68, s52, s63
	s_mul_i32 s57, s53, s56
	s_add_i32 s62, s62, s68
	s_add_i32 s62, s62, s57
	s_mul_i32 s69, s52, s56
	s_mul_i32 s68, s56, s62
	s_mul_hi_u32 s75, s56, s69
	s_mul_hi_u32 s57, s56, s62
	s_add_u32 s68, s75, s68
	s_addc_u32 s57, 0, s57
	s_mul_hi_u32 s81, s63, s69
	s_mul_i32 s69, s63, s69
	s_add_u32 s68, s68, s69
	s_mul_hi_u32 s75, s63, s62
	s_addc_u32 s57, s57, s81
	s_addc_u32 s68, s75, 0
	s_mul_i32 s62, s63, s62
	s_add_u32 s57, s57, s62
	s_addc_u32 s62, 0, s68
	s_add_u32 s56, s56, s57
	s_addc_u32 s57, s63, s62
	s_mul_i32 s62, s52, s57
	s_mul_hi_u32 s63, s52, s56
	s_add_i32 s62, s63, s62
	s_mul_i32 s53, s53, s56
	s_add_i32 s62, s62, s53
	s_mul_i32 s52, s52, s56
	s_mul_hi_u32 s63, s57, s52
	s_mul_i32 s68, s57, s52
	s_mul_i32 s75, s56, s62
	s_mul_hi_u32 s52, s56, s52
	s_mul_hi_u32 s69, s56, s62
	s_add_u32 s52, s52, s75
	s_addc_u32 s69, 0, s69
	s_add_u32 s52, s52, s68
	s_mul_hi_u32 s53, s57, s62
	s_addc_u32 s52, s69, s63
	s_addc_u32 s53, s53, 0
	s_mul_i32 s62, s57, s62
	s_add_u32 s52, s52, s62
	s_addc_u32 s53, 0, s53
	s_add_u32 s56, s56, s52
	s_addc_u32 s57, s57, s53
	s_ashr_i32 s52, s29, 31
	s_add_u32 s28, s28, s52
	s_mov_b32 s53, s52
	s_addc_u32 s29, s29, s52
	s_xor_b64 s[28:29], s[28:29], s[52:53]
	s_mul_i32 s63, s28, s57
	s_mul_hi_u32 s68, s28, s56
	s_mul_hi_u32 s62, s28, s57
	s_add_u32 s63, s68, s63
	s_addc_u32 s62, 0, s62
	s_mul_hi_u32 s69, s29, s56
	s_mul_i32 s56, s29, s56
	s_add_u32 s56, s63, s56
	s_mul_hi_u32 s68, s29, s57
	s_addc_u32 s56, s62, s69
	s_addc_u32 s62, s68, 0
	s_mul_i32 s57, s29, s57
	s_add_u32 s63, s56, s57
	s_addc_u32 s62, 0, s62
	s_mul_i32 s56, s24, s62
	s_mul_hi_u32 s57, s24, s63
	s_add_i32 s56, s57, s56
	s_mul_i32 s57, s25, s63
	s_add_i32 s68, s56, s57
	s_sub_i32 s69, s29, s68
	s_mul_i32 s56, s24, s63
	s_sub_u32 s28, s28, s56
	s_cselect_b64 s[56:57], -1, 0
	s_subb_u32 s69, s69, s25
	s_sub_u32 s75, s28, s24
	s_subb_u32 s69, s69, 0
	s_cmp_ge_u32 s69, s25
	s_cselect_b32 s81, -1, 0
	s_cmp_ge_u32 s75, s24
	s_cselect_b32 s75, -1, 0
	s_cmp_eq_u32 s69, s25
	s_cselect_b32 s69, s75, s81
	s_add_u32 s75, s63, 1
	s_addc_u32 s81, s62, 0
	s_add_u32 s83, s63, 2
	s_addc_u32 s84, s62, 0
	s_cmp_lg_u32 s69, 0
	s_cselect_b32 s69, s83, s75
	s_cselect_b32 s75, s84, s81
	s_cmp_lg_u64 s[56:57], 0
	s_subb_u32 s29, s29, s68
	s_cmp_ge_u32 s29, s25
	s_cselect_b32 s56, -1, 0
	s_cmp_ge_u32 s28, s24
	s_cselect_b32 s24, -1, 0
	s_cmp_eq_u32 s29, s25
	s_cselect_b32 s24, s24, s56
	s_cmp_lg_u32 s24, 0
	s_cselect_b32 s25, s75, s62
	s_cselect_b32 s24, s69, s63
	s_xor_b64 s[22:23], s[52:53], s[22:23]
	s_xor_b64 s[24:25], s[24:25], s[22:23]
	s_sub_u32 s22, s24, s22
	s_subb_u32 s23, s25, s23
	v_xor_b32_e32 v0, s22, v6
	v_xor_b32_e32 v4, s23, v6
	v_cndmask_b32_e32 v5, 0, v4, vcc
	v_cndmask_b32_e32 v4, 0, v0, vcc
	s_ashr_i32 s22, s35, 31
	v_lshlrev_b64 v[50:51], 10, v[4:5]
	v_lshlrev_b64 v[6:7], 14, v[4:5]
	v_mul_i32_i24_e32 v4, 0xfffff800, v45
	v_mov_b32_e32 v5, 0
	s_mov_b32 s23, s22
	v_lshl_add_u32 v0, s86, 8, v70
	v_lshl_add_u64 v[16:17], v[4:5], 0, v[24:25]
	v_or_b32_e32 v6, s74, v6
	s_xor_b64 s[24:25], s[22:23], s[34:35]
	v_or_b32_e32 v50, v50, v0
	v_bitop3_b32 v0, v98, v96, 7 bitop3:0x78
	v_lshl_add_u64 v[52:53], v[6:7], 0, v[16:17]
	s_lshr_b32 s28, s25, 31
	v_lshl_or_b32 v54, v0, 4, v52
	v_bitop3_b32 v0, v45, v3, 4 bitop3:0x36
	s_add_u32 s24, s24, s28
	v_lshl_or_b32 v52, v0, 4, v52
	s_addc_u32 s25, s25, 0
	ds_read_b128 v[132:135], v52
	ds_read_b128 v[128:131], v54 offset:2048
	ds_read_b128 v[124:127], v52 offset:2048
	s_ashr_i64 s[24:25], s[24:25], 1
	v_mov_b32_e32 v55, v53
	ds_read_b32 v80, v50 offset:49152
	ds_read_b128 v[88:91], v54
	s_xor_b64 s[22:23], s[24:25], s[22:23]
	s_mov_b64 s[24:25], 0x1000
	v_lshl_add_u64 v[56:57], v[54:55], 0, s[24:25]
	v_lshl_add_u64 v[58:59], v[52:53], 0, s[24:25]
	s_mov_b64 s[24:25], 0x1800
	v_lshl_add_u64 v[60:61], v[54:55], 0, s[24:25]
	v_lshl_add_u64 v[62:63], v[52:53], 0, s[24:25]
	s_mov_b64 s[24:25], 0x100
	v_lshl_add_u64 v[64:65], v[50:51], 0, s[24:25]
	v_cmp_gt_i64_e64 s[24:25], s[22:23], 1
	s_mov_b64 s[14:15], 0
	s_and_b64 vcc, exec, s[24:25]
	v_ashrrev_i32_e32 v69, 31, v68
	v_ashrrev_i32_e32 v67, 31, v66
	s_cbranch_vccnz .LBB0_37
	v_mov_b32_e32 v25, v5
	s_lshl_b64 s[52:53], s[22:23], 1
	s_andn2_b64 vcc, exec, s[14:15]
	v_mov_b32_e32 v71, 0
	s_cbranch_vccz .LBB0_38
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, v71
	v_mov_b32_e32 v1, v71
	v_mov_b32_e32 v0, v71
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, v71
	v_mov_b32_e32 v21, v71
	v_mov_b32_e32 v20, v71
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v6, v71
	v_mov_b32_e32 v5, v71
	v_mov_b32_e32 v4, v71
	v_mov_b32_e32 v19, 0
	v_mov_b32_e32 v18, v71
	v_mov_b32_e32 v17, v71
	v_mov_b32_e32 v16, v71
	s_branch .LBB0_89
.LBB0_37:
	s_lshl_b64 s[52:53], s[22:23], 1
	v_mov_b32_e32 v71, 0
.LBB0_38:
	v_mov_b32_e32 v3, v1
	v_lshlrev_b32_e32 v1, 8, v45
	v_and_b32_e32 v1, 0x200, v1
	v_mul_i32_i24_e32 v0, 0xffffffc0, v45
	v_or_b32_e32 v2, v2, v70
	v_or_b32_e32 v70, 0x800, v1
	v_mad_u64_u32 v[72:73], s[14:15], s73, v72, 0
	s_add_u32 s84, s52, -2
	v_lshl_add_u64 v[4:5], v[68:69], 0, v[70:71]
	v_ashrrev_i32_e32 v1, 31, v0
	v_writelane_b32 v144, s86, 28
	v_add3_u32 v73, v73, v75, v74
	s_addc_u32 s85, s53, -1
	v_mov_b32_e32 v25, v71
	v_lshl_add_u64 v[74:75], v[4:5], 0, v[66:67]
	v_lshl_add_u64 v[76:77], v[2:3], 0, v[0:1]
	v_or_b32_e32 v72, v72, v46
	s_mov_b64 s[86:87], 0
	s_mov_b64 s[88:89], 0x800
	s_mov_b32 s74, 0
	s_mov_b32 s90, s72
	s_mov_b32 s91, s72
	s_mov_b64 s[92:93], 0xc00
	v_bfrev_b32_e32 v29, -2
	v_mov_b32_e32 v0, v71
	v_mov_b32_e32 v1, v71
	v_mov_b32_e32 v2, v71
	v_mov_b32_e32 v3, v71
	v_mov_b32_e32 v20, v71
	v_mov_b32_e32 v21, v71
	v_mov_b32_e32 v22, v71
	v_mov_b32_e32 v23, v71
	v_mov_b32_e32 v4, v71
	v_mov_b32_e32 v5, v71
	v_mov_b32_e32 v6, v71
	v_mov_b32_e32 v7, v71
	v_mov_b32_e32 v16, v71
	v_mov_b32_e32 v17, v71
	v_mov_b32_e32 v18, v71
	v_mov_b32_e32 v19, v71
	s_mov_b64 s[94:95], 0x100
	s_mov_b64 s[96:97], 0x102
	s_mov_b64 s[24:25], 0x200
	s_mov_b64 s[62:63], 0x202
	s_branch .LBB0_40
.LBB0_39:
	s_or_b64 exec, exec, s[14:15]
	v_ashrrev_i32_e32 v8, 31, v15
	v_and_b32_e32 v9, s31, v8
	v_and_b32_e32 v8, s30, v8
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[14:15]
	v_ashrrev_i32_e32 v11, 31, v9
	v_xor_b32_e32 v9, v11, v9
	v_xor_b32_e32 v10, v10, v12
	v_ashrrev_i32_e32 v12, 31, v9
	v_xor_b32_e32 v8, v11, v8
	v_lshrrev_b32_e32 v70, 27, v12
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[70:71]
	v_alignbit_b32 v8, v9, v8, 5
	v_mul_lo_u32 v9, v10, s46
	v_xad_u32 v8, v8, v11, v9
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dword v8, s[4:7], 0 offen lds
	s_waitcnt vmcnt(5) lgkmcnt(0)
	s_barrier
	s_add_u32 s86, s86, 2
	s_addc_u32 s87, s87, 0
	v_mov_b64_e32 v[8:9], s[84:85]
	v_cmp_lt_i64_e32 vcc, s[86:87], v[8:9]
	v_lshlrev_b16_e32 v8, 8, v61
	v_lshlrev_b16_e32 v9, 8, v63
	v_or_b32_e32 v8, v53, v8
	v_or_b32_sdwa v9, v55, v9 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v10, 8, v59
	v_or_b32_sdwa v8, v8, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v9, 8, v57
	v_or_b32_e32 v9, v39, v9
	v_or_b32_sdwa v10, v51, v10 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v11, 8, v106
	v_or_b32_sdwa v9, v9, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v10, 8, v105
	v_or_b32_e32 v10, v101, v10
	v_or_b32_sdwa v11, v102, v11 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v12, 8, v104
	v_or_b32_sdwa v10, v10, v11 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v11, 8, v103
	v_or_b32_e32 v11, v65, v11
	v_or_b32_sdwa v12, v85, v12 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v13, 8, v115
	v_or_b32_sdwa v11, v11, v12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v12, 8, v114
	v_or_b32_e32 v12, v110, v12
	v_or_b32_sdwa v13, v111, v13 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v14, 8, v113
	v_or_b32_sdwa v12, v12, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v13, 8, v112
	v_or_b32_e32 v13, v108, v13
	v_or_b32_sdwa v14, v109, v14 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v15, 8, v122
	v_or_b32_sdwa v13, v13, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v14, 8, v121
	v_or_b32_e32 v14, v119, v14
	v_or_b32_sdwa v15, v120, v15 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v31, 8, v118
	v_or_b32_sdwa v14, v14, v15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v15, 8, v117
	v_or_b32_e32 v15, v35, v15
	v_or_b32_sdwa v31, v116, v31 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_nop 0
	v_or_b32_sdwa v15, v15, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_cbranch_vccz .LBB0_88
.LBB0_40:
	s_waitcnt vmcnt(5) lgkmcnt(0)
	s_barrier
	s_lshl_b64 s[14:15], s[86:87], 11
	v_or_b32_e32 v78, s14, v44
	s_add_u32 s14, s44, s72
	v_mov_b32_e32 v79, s15
	s_addc_u32 s15, s45, s72
	s_xor_b64 s[68:69], s[14:15], s[90:91]
	v_cvt_f32_u32_e32 v33, s68
	v_cvt_f32_u32_e32 v35, s69
	v_and_b32_e32 v31, 0xff, v92
	s_movk_i32 s14, 0xf7ff
	v_and_b32_e32 v106, 0xff, v81
	v_fmac_f32_e32 v33, 0x4f800000, v35
	v_rcp_f32_e32 v33, v33
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[88:91], v[8:11], v[0:3], v80, v31 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshl_add_u64 v[90:91], v[78:79], 0, v[68:69]
	v_sub_co_u32_e32 v39, vcc, s14, v90
	v_mul_f32_e32 v33, 0x5f7ffffc, v33
	v_mul_f32_e32 v35, 0x2f800000, v33
	v_trunc_f32_e32 v35, v35
	v_fmac_f32_e32 v33, 0xcf800000, v35
	v_cvt_u32_f32_e32 v33, v33
	v_cvt_u32_f32_e32 v35, v35
	s_sub_u32 s14, 0, s68
	s_subb_u32 s15, 0, s69
	v_readfirstlane_b32 s22, v33
	v_readfirstlane_b32 s29, v35
	s_mul_hi_u32 s28, s14, s22
	s_mul_i32 s56, s14, s29
	s_mul_i32 s23, s15, s22
	s_add_i32 s28, s28, s56
	s_add_i32 s28, s28, s23
	s_mul_i32 s57, s14, s22
	s_mul_i32 s56, s22, s28
	s_mul_hi_u32 s73, s22, s57
	s_mul_hi_u32 s23, s22, s28
	s_add_u32 s56, s73, s56
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 s75, s29, s57
	s_mul_i32 s57, s29, s57
	s_add_u32 s56, s56, s57
	s_mul_hi_u32 s73, s29, s28
	s_addc_u32 s23, s23, s75
	s_addc_u32 s56, s73, 0
	s_mul_i32 s28, s29, s28
	s_add_u32 s23, s23, s28
	s_addc_u32 s28, 0, s56
	s_add_u32 s22, s22, s23
	s_addc_u32 s23, s29, s28
	s_mul_i32 s28, s14, s23
	s_mul_hi_u32 s29, s14, s22
	s_add_i32 s28, s29, s28
	s_mul_i32 s15, s15, s22
	s_add_i32 s28, s28, s15
	s_mul_i32 s14, s14, s22
	s_mul_hi_u32 s29, s23, s14
	s_mul_i32 s56, s23, s14
	s_mul_i32 s73, s22, s28
	s_mul_hi_u32 s14, s22, s14
	s_mul_hi_u32 s57, s22, s28
	s_add_u32 s14, s14, s73
	s_addc_u32 s57, 0, s57
	v_lshl_add_u64 v[82:83], v[90:91], 0, s[88:89]
	v_subb_co_u32_e32 v51, vcc, -1, v91, vcc
	s_add_u32 s14, s14, s56
	v_cmp_gt_i32_e32 vcc, 0, v83
	s_mul_hi_u32 s15, s23, s28
	s_addc_u32 s14, s57, s29
	v_cndmask_b32_e32 v85, v83, v51, vcc
	s_addc_u32 s15, s15, 0
	s_mul_i32 s28, s23, s28
	s_add_u32 s14, s14, s28
	v_ashrrev_i32_e32 v86, 31, v85
	v_cndmask_b32_e32 v84, v82, v39, vcc
	s_addc_u32 s15, 0, s15
	v_mov_b32_e32 v87, v86
	s_add_u32 s56, s22, s14
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	s_addc_u32 s57, s23, s15
	v_xor_b32_e32 v35, v84, v86
	v_xor_b32_e32 v33, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v35, s57, 0
	v_mul_hi_u32 v70, v35, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[92:93], s[14:15], v33, s56, 0
	v_add_co_u32_e32 v39, vcc, v84, v92
	v_mad_u64_u32 v[88:89], s[14:15], v33, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v93, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v89, vcc, 0, v89, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[88:89]
	v_mul_lo_u32 v39, s69, v84
	v_mul_lo_u32 v51, s68, v85
	v_mad_u64_u32 v[88:89], s[14:15], s68, v84, 0
	v_add3_u32 v39, v89, v51, v39
	v_sub_co_u32_e32 v35, vcc, v35, v88
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[128:131], v[8:11], v[20:23], v80, v31 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_subb_co_u32_e64 v51, s[14:15], v33, v39, vcc
	v_sub_u32_e32 v39, v33, v39
	v_mov_b32_e32 v33, s69
	v_cmp_le_u32_e64 s[14:15], s69, v51
	v_subb_co_u32_e32 v39, vcc, v39, v33, vcc
	s_nop 0
	v_cndmask_b32_e64 v53, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v35
	v_subrev_co_u32_e32 v35, vcc, s68, v35
	s_nop 0
	v_cndmask_b32_e64 v55, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v39, vcc, 0, v39, vcc
	v_cmp_eq_u32_e64 s[14:15], s69, v51
	v_cmp_le_u32_e32 vcc, s69, v39
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[132:135], v[12:15], v[0:3], v80, v106 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e64 v51, v53, v55, s[14:15]
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v35
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[124:127], v[12:15], v[20:23], v80, v106 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v39
	v_xor_b32_e32 v39, s72, v86
	s_nop 0
	v_cndmask_b32_e32 v35, v53, v35, vcc
	v_cmp_ne_u32_e32 vcc, 0, v35
	s_nop 1
	v_cndmask_b32_e64 v35, 1, 2, vcc
	v_add_u32_e32 v35, v84, v35
	v_cmp_ne_u32_e32 vcc, 0, v51
	s_nop 1
	v_cndmask_b32_e32 v35, v84, v35, vcc
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v35, v35, v39
	v_xor_b32_e32 v51, v82, v84
	v_sub_u32_e32 v35, v35, v39
	v_xor_b32_e32 v39, v83, v84
	v_mad_u64_u32 v[82:83], s[14:15], v51, s57, 0
	v_mul_hi_u32 v70, v51, s56
	v_lshl_add_u64 v[82:83], v[70:71], 0, v[82:83]
	v_mad_u64_u32 v[88:89], s[14:15], v39, s56, 0
	v_add_co_u32_e32 v53, vcc, v82, v88
	v_mad_u64_u32 v[86:87], s[14:15], v39, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v89, vcc
	v_mov_b32_e32 v83, s74
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[86:87]
	v_mul_lo_u32 v53, s69, v82
	v_mul_lo_u32 v55, s68, v83
	v_mad_u64_u32 v[82:83], s[14:15], s68, v82, 0
	v_add3_u32 v53, v83, v55, v53
	v_sub_u32_e32 v55, v39, v53
	v_sub_co_u32_e32 v51, vcc, v51, v82
	v_xad_u32 v35, v35, v84, v37
	s_nop 0
	v_subb_co_u32_e64 v55, s[14:15], v55, v33, vcc
	v_subrev_co_u32_e64 v57, s[14:15], s68, v51
	v_subb_co_u32_e32 v39, vcc, v39, v53, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v59, s[22:23], 0, v55, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s69, v59
	v_subb_co_u32_e64 v55, s[14:15], v55, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s68, v57
	v_cmp_le_u32_e32 vcc, s69, v39
	v_mul_lo_u32 v35, v35, s33
	v_cndmask_b32_e64 v63, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s69, v59
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v51
	v_cndmask_b32_e64 v61, v61, v63, s[22:23]
	v_subrev_co_u32_e64 v63, s[14:15], s68, v57
	v_lshl_add_u64 v[88:89], v[78:79], 0, v[74:75]
	s_nop 0
	v_subbrev_co_u32_e64 v55, s[14:15], 0, v55, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v61
	v_cmp_gt_i32_e64 s[22:23], 0, v89
	v_or_b32_e32 v82, 1, v88
	v_cndmask_b32_e64 v55, v59, v55, s[14:15]
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v39
	v_mov_b32_e32 v83, v89
	s_nop 0
	v_cndmask_b32_e32 v53, v53, v59, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_cndmask_b32_e64 v53, v57, v63, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v51, v51, v53, vcc
	v_cndmask_b32_e32 v39, v39, v55, vcc
	v_xor_b32_e32 v51, v51, v84
	v_xor_b32_e32 v39, v39, v84
	v_sub_co_u32_e32 v51, vcc, v51, v84
	s_nop 1
	v_subb_co_u32_e32 v39, vcc, v39, v84, vcc
	v_ashrrev_i32_e32 v39, 31, v39
	v_and_b32_e32 v39, s44, v39
	v_add3_u32 v35, v39, v51, v35
	v_sub_co_u32_e32 v39, vcc, -2, v88
	v_cndmask_b32_e64 v78, v82, v39, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v51, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v79, v89, v51, s[22:23]
	v_ashrrev_i32_e32 v84, 31, v79
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[78:79], v[78:79], 0, v[84:85]
	v_xor_b32_e32 v51, v78, v84
	v_xor_b32_e32 v39, v79, v84
	v_mad_u64_u32 v[78:79], s[14:15], v51, s57, 0
	v_mul_hi_u32 v70, v51, s56
	v_lshl_add_u64 v[78:79], v[70:71], 0, v[78:79]
	v_mad_u64_u32 v[92:93], s[14:15], v39, s56, 0
	v_add_co_u32_e32 v53, vcc, v78, v92
	v_mad_u64_u32 v[86:87], s[14:15], v39, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v78, vcc, v79, v93, vcc
	v_mov_b32_e32 v79, s74
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[78:79], v[78:79], 0, v[86:87]
	v_mul_lo_u32 v53, s69, v78
	v_mul_lo_u32 v55, s68, v79
	v_mad_u64_u32 v[86:87], s[14:15], s68, v78, 0
	v_add3_u32 v53, v87, v55, v53
	v_sub_co_u32_e32 v51, vcc, v51, v86
	s_nop 1
	v_subb_co_u32_e64 v55, s[14:15], v39, v53, vcc
	v_sub_u32_e32 v39, v39, v53
	v_cmp_le_u32_e64 s[14:15], s69, v55
	v_subb_co_u32_e32 v39, vcc, v39, v33, vcc
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v51
	v_subrev_co_u32_e32 v51, vcc, s68, v51
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v39, vcc, 0, v39, vcc
	v_cmp_le_u32_e32 vcc, s69, v39
	v_cmp_eq_u32_e64 s[14:15], s69, v55
	s_nop 0
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v51
	v_cndmask_b32_e64 v55, v57, v59, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v39
	s_nop 1
	v_cndmask_b32_e32 v39, v53, v51, vcc
	v_cmp_ne_u32_e32 vcc, 0, v39
	v_xor_b32_e32 v51, s72, v84
	s_nop 0
	v_cndmask_b32_e64 v39, 1, 2, vcc
	v_add_u32_e32 v39, v78, v39
	v_cmp_ne_u32_e32 vcc, 0, v55
	s_nop 1
	v_cndmask_b32_e32 v39, v78, v39, vcc
	v_ashrrev_i32_e32 v78, 31, v89
	v_mov_b32_e32 v79, v78
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[78:79]
	v_xor_b32_e32 v39, v39, v51
	v_xor_b32_e32 v53, v82, v78
	v_sub_u32_e32 v39, v39, v51
	v_xor_b32_e32 v51, v83, v78
	v_mad_u64_u32 v[82:83], s[14:15], v53, s57, 0
	v_mul_hi_u32 v70, v53, s56
	v_lshl_add_u64 v[82:83], v[70:71], 0, v[82:83]
	v_mad_u64_u32 v[86:87], s[14:15], v51, s56, 0
	v_add_co_u32_e32 v55, vcc, v82, v86
	v_mad_u64_u32 v[84:85], s[14:15], v51, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v87, vcc
	v_mov_b32_e32 v83, s74
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_mul_lo_u32 v55, s69, v82
	v_mul_lo_u32 v57, s68, v83
	v_mad_u64_u32 v[82:83], s[14:15], s68, v82, 0
	v_add3_u32 v55, v83, v57, v55
	v_sub_u32_e32 v57, v51, v55
	v_sub_co_u32_e32 v53, vcc, v53, v82
	v_xad_u32 v39, v39, v78, v37
	s_nop 0
	v_subb_co_u32_e64 v57, s[14:15], v57, v33, vcc
	v_subrev_co_u32_e64 v59, s[14:15], s68, v53
	v_subb_co_u32_e32 v51, vcc, v51, v55, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v61, s[28:29], 0, v57, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v61
	v_subb_co_u32_e64 v57, s[14:15], v57, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v59
	v_cmp_le_u32_e32 vcc, s69, v51
	v_mul_lo_u32 v39, v39, s33
	v_cndmask_b32_e64 v65, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v61
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v53
	v_cndmask_b32_e64 v63, v63, v65, s[28:29]
	v_subrev_co_u32_e64 v65, s[14:15], s68, v59
	v_or_b32_e32 v82, 2, v88
	s_nop 0
	v_subbrev_co_u32_e64 v57, s[14:15], 0, v57, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v63
	v_mov_b32_e32 v83, v89
	s_nop 0
	v_cndmask_b32_e64 v57, v61, v57, s[14:15]
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v51
	s_nop 1
	v_cndmask_b32_e32 v55, v55, v61, vcc
	v_cmp_ne_u32_e32 vcc, 0, v55
	v_cndmask_b32_e64 v55, v59, v65, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v53, v53, v55, vcc
	v_cndmask_b32_e32 v51, v51, v57, vcc
	v_xor_b32_e32 v53, v53, v78
	v_xor_b32_e32 v51, v51, v78
	v_sub_co_u32_e32 v53, vcc, v53, v78
	s_nop 1
	v_subb_co_u32_e32 v51, vcc, v51, v78, vcc
	v_ashrrev_i32_e32 v51, 31, v51
	v_and_b32_e32 v51, s44, v51
	v_add3_u32 v39, v51, v53, v39
	v_sub_co_u32_e32 v51, vcc, -3, v88
	v_cndmask_b32_e64 v84, v82, v51, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v53, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v85, v89, v53, s[22:23]
	v_ashrrev_i32_e32 v86, 31, v85
	v_mov_b32_e32 v87, v86
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_xor_b32_e32 v53, v84, v86
	v_xor_b32_e32 v51, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v53, s57, 0
	v_mul_hi_u32 v70, v53, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[94:95], s[14:15], v51, s56, 0
	v_add_co_u32_e32 v55, vcc, v84, v94
	v_mad_u64_u32 v[92:93], s[14:15], v51, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v95, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v55, s69, v84
	v_mul_lo_u32 v57, s68, v85
	v_mad_u64_u32 v[92:93], s[14:15], s68, v84, 0
	v_add3_u32 v55, v93, v57, v55
	v_sub_co_u32_e32 v53, vcc, v53, v92
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[78:79]
	s_nop 0
	v_subb_co_u32_e64 v57, s[14:15], v51, v55, vcc
	v_sub_u32_e32 v51, v51, v55
	v_cmp_le_u32_e64 s[14:15], s69, v57
	v_subb_co_u32_e32 v51, vcc, v51, v33, vcc
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v53
	v_subrev_co_u32_e32 v53, vcc, s68, v53
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v51, vcc, 0, v51, vcc
	v_cmp_le_u32_e32 vcc, s69, v51
	v_cmp_eq_u32_e64 s[14:15], s69, v57
	s_nop 0
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v53
	v_cndmask_b32_e64 v57, v59, v61, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v51
	s_nop 1
	v_cndmask_b32_e32 v51, v55, v53, vcc
	v_cmp_ne_u32_e32 vcc, 0, v51
	v_xor_b32_e32 v53, s72, v86
	v_xor_b32_e32 v55, v82, v78
	v_cndmask_b32_e64 v51, 1, 2, vcc
	v_add_u32_e32 v51, v84, v51
	v_cmp_ne_u32_e32 vcc, 0, v57
	v_mul_hi_u32 v70, v55, s56
	s_nop 0
	v_cndmask_b32_e32 v51, v84, v51, vcc
	v_xor_b32_e32 v51, v51, v53
	v_sub_u32_e32 v51, v51, v53
	v_xor_b32_e32 v53, v83, v78
	v_mad_u64_u32 v[82:83], s[14:15], v55, s57, 0
	v_lshl_add_u64 v[82:83], v[70:71], 0, v[82:83]
	v_mad_u64_u32 v[86:87], s[14:15], v53, s56, 0
	v_add_co_u32_e32 v57, vcc, v82, v86
	v_mad_u64_u32 v[84:85], s[14:15], v53, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v87, vcc
	v_mov_b32_e32 v83, s74
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_mul_lo_u32 v57, s69, v82
	v_mul_lo_u32 v59, s68, v83
	v_mad_u64_u32 v[82:83], s[14:15], s68, v82, 0
	v_add3_u32 v57, v83, v59, v57
	v_sub_u32_e32 v59, v53, v57
	v_sub_co_u32_e32 v55, vcc, v55, v82
	v_xad_u32 v51, v51, v78, v37
	s_nop 0
	v_subb_co_u32_e64 v59, s[14:15], v59, v33, vcc
	v_subrev_co_u32_e64 v61, s[14:15], s68, v55
	v_subb_co_u32_e32 v53, vcc, v53, v57, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v63, s[28:29], 0, v59, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v63
	v_subb_co_u32_e64 v59, s[14:15], v59, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v61
	v_cmp_le_u32_e32 vcc, s69, v53
	v_mul_lo_u32 v51, v51, s33
	v_cndmask_b32_e64 v70, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v63
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v55
	v_cndmask_b32_e64 v65, v65, v70, s[28:29]
	v_subrev_co_u32_e64 v70, s[14:15], s68, v61
	v_or_b32_e32 v82, 3, v88
	s_nop 0
	v_subbrev_co_u32_e64 v59, s[14:15], 0, v59, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v65
	v_mov_b32_e32 v83, v89
	s_nop 0
	v_cndmask_b32_e64 v59, v63, v59, s[14:15]
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v53
	s_nop 1
	v_cndmask_b32_e32 v57, v57, v63, vcc
	v_cmp_ne_u32_e32 vcc, 0, v57
	v_cndmask_b32_e64 v57, v61, v70, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v55, v55, v57, vcc
	v_cndmask_b32_e32 v53, v53, v59, vcc
	v_xor_b32_e32 v55, v55, v78
	v_xor_b32_e32 v53, v53, v78
	v_sub_co_u32_e32 v55, vcc, v55, v78
	s_nop 1
	v_subb_co_u32_e32 v53, vcc, v53, v78, vcc
	v_ashrrev_i32_e32 v53, 31, v53
	v_and_b32_e32 v53, s44, v53
	v_add3_u32 v53, v53, v55, v51
	v_sub_co_u32_e32 v51, vcc, -4, v88
	v_cndmask_b32_e64 v84, v82, v51, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v55, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v85, v89, v55, s[22:23]
	v_ashrrev_i32_e32 v86, 31, v85
	v_mov_b32_e32 v87, v86
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_xor_b32_e32 v55, v84, v86
	v_xor_b32_e32 v51, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v55, s57, 0
	v_mul_hi_u32 v70, v55, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[94:95], s[14:15], v51, s56, 0
	v_add_co_u32_e32 v57, vcc, v84, v94
	v_mad_u64_u32 v[92:93], s[14:15], v51, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v95, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v57, s69, v84
	v_mul_lo_u32 v59, s68, v85
	v_mad_u64_u32 v[92:93], s[14:15], s68, v84, 0
	v_add3_u32 v57, v93, v59, v57
	v_sub_co_u32_e32 v55, vcc, v55, v92
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[78:79]
	s_nop 0
	v_subb_co_u32_e64 v59, s[14:15], v51, v57, vcc
	v_sub_u32_e32 v51, v51, v57
	v_cmp_le_u32_e64 s[14:15], s69, v59
	v_subb_co_u32_e32 v51, vcc, v51, v33, vcc
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v55
	v_subrev_co_u32_e32 v55, vcc, s68, v55
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v51, vcc, 0, v51, vcc
	v_cmp_le_u32_e32 vcc, s69, v51
	v_cmp_eq_u32_e64 s[14:15], s69, v59
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v55
	v_cndmask_b32_e64 v59, v61, v63, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v51
	s_nop 1
	v_cndmask_b32_e32 v51, v57, v55, vcc
	v_cmp_ne_u32_e32 vcc, 0, v51
	v_xor_b32_e32 v55, s72, v86
	v_xor_b32_e32 v57, v82, v78
	v_cndmask_b32_e64 v51, 1, 2, vcc
	v_add_u32_e32 v51, v84, v51
	v_cmp_ne_u32_e32 vcc, 0, v59
	v_mul_hi_u32 v70, v57, s56
	s_nop 0
	v_cndmask_b32_e32 v51, v84, v51, vcc
	v_xor_b32_e32 v51, v51, v55
	v_sub_u32_e32 v51, v51, v55
	v_xor_b32_e32 v55, v83, v78
	v_mad_u64_u32 v[82:83], s[14:15], v57, s57, 0
	v_lshl_add_u64 v[82:83], v[70:71], 0, v[82:83]
	v_mad_u64_u32 v[86:87], s[14:15], v55, s56, 0
	v_add_co_u32_e32 v59, vcc, v82, v86
	v_mad_u64_u32 v[84:85], s[14:15], v55, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v87, vcc
	v_mov_b32_e32 v83, s74
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_mul_lo_u32 v59, s69, v82
	v_mul_lo_u32 v61, s68, v83
	v_mad_u64_u32 v[82:83], s[14:15], s68, v82, 0
	v_add3_u32 v59, v83, v61, v59
	v_sub_u32_e32 v61, v55, v59
	v_sub_co_u32_e32 v57, vcc, v57, v82
	v_xad_u32 v51, v51, v78, v37
	s_nop 0
	v_subb_co_u32_e64 v61, s[14:15], v61, v33, vcc
	v_subrev_co_u32_e64 v63, s[14:15], s68, v57
	v_subb_co_u32_e32 v55, vcc, v55, v59, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[28:29], 0, v61, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v65
	v_subb_co_u32_e64 v61, s[14:15], v61, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v63
	v_cmp_le_u32_e32 vcc, s69, v55
	v_mul_lo_u32 v51, v51, s33
	v_cndmask_b32_e64 v82, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v65
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v57
	v_cndmask_b32_e64 v70, v70, v82, s[28:29]
	v_subrev_co_u32_e64 v82, s[14:15], s68, v63
	v_mov_b32_e32 v83, v89
	s_nop 0
	v_subbrev_co_u32_e64 v61, s[14:15], 0, v61, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v70
	s_nop 1
	v_cndmask_b32_e64 v61, v65, v61, s[14:15]
	v_cndmask_b32_e64 v65, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v55
	s_nop 1
	v_cndmask_b32_e32 v59, v59, v65, vcc
	v_cmp_ne_u32_e32 vcc, 0, v59
	v_cndmask_b32_e64 v59, v63, v82, s[14:15]
	v_or_b32_e32 v82, 4, v88
	v_cndmask_b32_e32 v57, v57, v59, vcc
	v_cndmask_b32_e32 v55, v55, v61, vcc
	v_xor_b32_e32 v57, v57, v78
	v_xor_b32_e32 v55, v55, v78
	v_sub_co_u32_e32 v57, vcc, v57, v78
	s_nop 1
	v_subb_co_u32_e32 v55, vcc, v55, v78, vcc
	v_ashrrev_i32_e32 v55, 31, v55
	v_and_b32_e32 v55, s44, v55
	v_add3_u32 v55, v55, v57, v51
	v_sub_co_u32_e32 v51, vcc, -5, v88
	v_cndmask_b32_e64 v84, v82, v51, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v57, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v85, v89, v57, s[22:23]
	v_ashrrev_i32_e32 v86, 31, v85
	v_mov_b32_e32 v87, v86
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_xor_b32_e32 v57, v84, v86
	v_xor_b32_e32 v51, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v57, s57, 0
	v_mul_hi_u32 v70, v57, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[94:95], s[14:15], v51, s56, 0
	v_add_co_u32_e32 v59, vcc, v84, v94
	v_mad_u64_u32 v[92:93], s[14:15], v51, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v95, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v59, s69, v84
	v_mul_lo_u32 v61, s68, v85
	v_mad_u64_u32 v[92:93], s[14:15], s68, v84, 0
	v_add3_u32 v59, v93, v61, v59
	v_sub_co_u32_e32 v57, vcc, v57, v92
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[78:79]
	s_nop 0
	v_subb_co_u32_e64 v61, s[14:15], v51, v59, vcc
	v_sub_u32_e32 v51, v51, v59
	v_cmp_le_u32_e64 s[14:15], s69, v61
	v_subb_co_u32_e32 v51, vcc, v51, v33, vcc
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v57
	v_subrev_co_u32_e32 v57, vcc, s68, v57
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v51, vcc, 0, v51, vcc
	v_cmp_le_u32_e32 vcc, s69, v51
	v_cmp_eq_u32_e64 s[14:15], s69, v61
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v57
	v_cndmask_b32_e64 v61, v63, v65, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v51
	s_nop 1
	v_cndmask_b32_e32 v51, v59, v57, vcc
	v_cmp_ne_u32_e32 vcc, 0, v51
	v_xor_b32_e32 v57, s72, v86
	v_xor_b32_e32 v59, v82, v78
	v_cndmask_b32_e64 v51, 1, 2, vcc
	v_add_u32_e32 v51, v84, v51
	v_cmp_ne_u32_e32 vcc, 0, v61
	v_mul_hi_u32 v70, v59, s56
	s_nop 0
	v_cndmask_b32_e32 v51, v84, v51, vcc
	v_xor_b32_e32 v51, v51, v57
	v_sub_u32_e32 v51, v51, v57
	v_xor_b32_e32 v57, v83, v78
	v_mad_u64_u32 v[82:83], s[14:15], v59, s57, 0
	v_lshl_add_u64 v[82:83], v[70:71], 0, v[82:83]
	v_mad_u64_u32 v[86:87], s[14:15], v57, s56, 0
	v_add_co_u32_e32 v61, vcc, v82, v86
	v_mad_u64_u32 v[84:85], s[14:15], v57, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v87, vcc
	v_mov_b32_e32 v83, s74
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_mul_lo_u32 v61, s69, v82
	v_mul_lo_u32 v63, s68, v83
	v_mad_u64_u32 v[82:83], s[14:15], s68, v82, 0
	v_add3_u32 v61, v83, v63, v61
	v_sub_u32_e32 v63, v57, v61
	v_sub_co_u32_e32 v59, vcc, v59, v82
	v_xad_u32 v51, v51, v78, v37
	s_nop 0
	v_subb_co_u32_e64 v63, s[14:15], v63, v33, vcc
	v_subrev_co_u32_e64 v65, s[14:15], s68, v59
	v_subb_co_u32_e32 v57, vcc, v57, v61, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v70, s[28:29], 0, v63, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v70
	v_subb_co_u32_e64 v63, s[14:15], v63, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v82, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v65
	v_cmp_le_u32_e32 vcc, s69, v57
	v_mul_lo_u32 v51, v51, s33
	v_cndmask_b32_e64 v83, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v70
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v59
	v_cndmask_b32_e64 v82, v82, v83, s[28:29]
	v_subrev_co_u32_e64 v83, s[14:15], s68, v65
	s_nop 1
	v_subbrev_co_u32_e64 v63, s[14:15], 0, v63, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v82
	v_or_b32_e32 v82, 5, v88
	s_nop 0
	v_cndmask_b32_e64 v63, v70, v63, s[14:15]
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v57
	s_nop 1
	v_cndmask_b32_e32 v61, v61, v70, vcc
	v_cmp_ne_u32_e32 vcc, 0, v61
	v_cndmask_b32_e64 v61, v65, v83, s[14:15]
	v_mov_b32_e32 v83, v89
	v_cndmask_b32_e32 v59, v59, v61, vcc
	v_cndmask_b32_e32 v57, v57, v63, vcc
	v_xor_b32_e32 v59, v59, v78
	v_xor_b32_e32 v57, v57, v78
	v_sub_co_u32_e32 v59, vcc, v59, v78
	s_nop 1
	v_subb_co_u32_e32 v57, vcc, v57, v78, vcc
	v_ashrrev_i32_e32 v57, 31, v57
	v_and_b32_e32 v57, s44, v57
	v_add3_u32 v57, v57, v59, v51
	v_sub_co_u32_e32 v51, vcc, -6, v88
	v_cndmask_b32_e64 v84, v82, v51, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v59, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v85, v89, v59, s[22:23]
	v_ashrrev_i32_e32 v86, 31, v85
	v_mov_b32_e32 v87, v86
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_xor_b32_e32 v59, v84, v86
	v_xor_b32_e32 v51, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v59, s57, 0
	v_mul_hi_u32 v70, v59, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[94:95], s[14:15], v51, s56, 0
	v_add_co_u32_e32 v61, vcc, v84, v94
	v_mad_u64_u32 v[92:93], s[14:15], v51, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v95, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v61, s69, v84
	v_mul_lo_u32 v63, s68, v85
	v_mad_u64_u32 v[92:93], s[14:15], s68, v84, 0
	v_add3_u32 v61, v93, v63, v61
	v_sub_co_u32_e32 v59, vcc, v59, v92
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[78:79]
	s_nop 0
	v_subb_co_u32_e64 v63, s[14:15], v51, v61, vcc
	v_sub_u32_e32 v51, v51, v61
	v_cmp_le_u32_e64 s[14:15], s69, v63
	v_subb_co_u32_e32 v51, vcc, v51, v33, vcc
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v59
	v_subrev_co_u32_e32 v59, vcc, s68, v59
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v51, vcc, 0, v51, vcc
	v_cmp_le_u32_e32 vcc, s69, v51
	v_cmp_eq_u32_e64 s[14:15], s69, v63
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v59
	v_cndmask_b32_e64 v63, v65, v70, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v51
	s_nop 1
	v_cndmask_b32_e32 v51, v61, v59, vcc
	v_cmp_ne_u32_e32 vcc, 0, v51
	v_xor_b32_e32 v59, s72, v86
	v_xor_b32_e32 v61, v82, v78
	v_cndmask_b32_e64 v51, 1, 2, vcc
	v_add_u32_e32 v51, v84, v51
	v_cmp_ne_u32_e32 vcc, 0, v63
	v_mul_hi_u32 v70, v61, s56
	s_nop 0
	v_cndmask_b32_e32 v51, v84, v51, vcc
	v_xor_b32_e32 v51, v51, v59
	v_sub_u32_e32 v51, v51, v59
	v_xor_b32_e32 v59, v83, v78
	v_mad_u64_u32 v[82:83], s[14:15], v61, s57, 0
	v_lshl_add_u64 v[82:83], v[70:71], 0, v[82:83]
	v_mad_u64_u32 v[86:87], s[14:15], v59, s56, 0
	v_add_co_u32_e32 v63, vcc, v82, v86
	v_mad_u64_u32 v[84:85], s[14:15], v59, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v87, vcc
	v_mov_b32_e32 v83, s74
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_mul_lo_u32 v63, s69, v82
	v_mul_lo_u32 v65, s68, v83
	v_mad_u64_u32 v[82:83], s[14:15], s68, v82, 0
	v_add3_u32 v63, v83, v65, v63
	v_sub_u32_e32 v65, v59, v63
	v_sub_co_u32_e32 v61, vcc, v61, v82
	v_xad_u32 v51, v51, v78, v37
	s_nop 0
	v_subb_co_u32_e64 v65, s[14:15], v65, v33, vcc
	v_subrev_co_u32_e64 v70, s[14:15], s68, v61
	v_subb_co_u32_e32 v59, vcc, v59, v63, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v82, s[28:29], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v82
	v_subb_co_u32_e64 v65, s[14:15], v65, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v70
	v_cmp_le_u32_e32 vcc, s69, v59
	v_mul_lo_u32 v51, v51, s33
	v_cndmask_b32_e64 v84, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v82
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v61
	v_cndmask_b32_e64 v83, v83, v84, s[28:29]
	v_subrev_co_u32_e64 v84, s[14:15], s68, v70
	s_nop 1
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v83
	v_mov_b32_e32 v83, v89
	s_nop 0
	v_cndmask_b32_e64 v65, v82, v65, s[14:15]
	v_cndmask_b32_e64 v82, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v59
	s_nop 1
	v_cndmask_b32_e32 v63, v63, v82, vcc
	v_cmp_ne_u32_e32 vcc, 0, v63
	v_cndmask_b32_e64 v63, v70, v84, s[14:15]
	v_or_b32_e32 v82, 6, v88
	v_cndmask_b32_e32 v61, v61, v63, vcc
	v_cndmask_b32_e32 v59, v59, v65, vcc
	v_xor_b32_e32 v61, v61, v78
	v_xor_b32_e32 v59, v59, v78
	v_sub_co_u32_e32 v61, vcc, v61, v78
	s_nop 1
	v_subb_co_u32_e32 v59, vcc, v59, v78, vcc
	v_ashrrev_i32_e32 v59, 31, v59
	v_and_b32_e32 v59, s44, v59
	v_add3_u32 v59, v59, v61, v51
	v_sub_co_u32_e32 v51, vcc, -7, v88
	v_cndmask_b32_e64 v84, v82, v51, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v61, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v85, v89, v61, s[22:23]
	v_ashrrev_i32_e32 v86, 31, v85
	v_mov_b32_e32 v87, v86
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_xor_b32_e32 v61, v84, v86
	v_xor_b32_e32 v51, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v61, s57, 0
	v_mul_hi_u32 v70, v61, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[94:95], s[14:15], v51, s56, 0
	v_add_co_u32_e32 v63, vcc, v84, v94
	v_mad_u64_u32 v[92:93], s[14:15], v51, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v95, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v63, s69, v84
	v_mul_lo_u32 v65, s68, v85
	v_mad_u64_u32 v[92:93], s[14:15], s68, v84, 0
	v_add3_u32 v63, v93, v65, v63
	v_sub_co_u32_e32 v61, vcc, v61, v92
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[78:79]
	s_nop 0
	v_subb_co_u32_e64 v65, s[14:15], v51, v63, vcc
	v_sub_u32_e32 v51, v51, v63
	v_cmp_le_u32_e64 s[14:15], s69, v65
	v_subb_co_u32_e32 v51, vcc, v51, v33, vcc
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v61
	v_subrev_co_u32_e32 v61, vcc, s68, v61
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v51, vcc, 0, v51, vcc
	v_cmp_le_u32_e32 vcc, s69, v51
	v_cmp_eq_u32_e64 s[14:15], s69, v65
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v61
	v_cndmask_b32_e64 v65, v70, v85, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v51
	s_nop 1
	v_cndmask_b32_e32 v51, v63, v61, vcc
	v_cmp_ne_u32_e32 vcc, 0, v51
	v_xor_b32_e32 v61, s72, v86
	v_xor_b32_e32 v63, v82, v78
	v_cndmask_b32_e64 v51, 1, 2, vcc
	v_add_u32_e32 v51, v84, v51
	v_cmp_ne_u32_e32 vcc, 0, v65
	v_mul_hi_u32 v70, v63, s56
	s_nop 0
	v_cndmask_b32_e32 v51, v84, v51, vcc
	v_xor_b32_e32 v51, v51, v61
	v_sub_u32_e32 v51, v51, v61
	v_xor_b32_e32 v61, v83, v78
	v_mad_u64_u32 v[82:83], s[14:15], v63, s57, 0
	v_lshl_add_u64 v[82:83], v[70:71], 0, v[82:83]
	v_mad_u64_u32 v[86:87], s[14:15], v61, s56, 0
	v_add_co_u32_e32 v65, vcc, v82, v86
	v_mad_u64_u32 v[84:85], s[14:15], v61, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v87, vcc
	v_mov_b32_e32 v83, s74
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_mul_lo_u32 v65, s69, v82
	v_mul_lo_u32 v70, s68, v83
	v_mad_u64_u32 v[82:83], s[14:15], s68, v82, 0
	v_add3_u32 v65, v83, v70, v65
	v_sub_u32_e32 v70, v61, v65
	v_sub_co_u32_e32 v63, vcc, v63, v82
	v_xad_u32 v51, v51, v78, v37
	s_nop 0
	v_subb_co_u32_e64 v70, s[14:15], v70, v33, vcc
	v_subrev_co_u32_e64 v82, s[14:15], s68, v63
	v_subb_co_u32_e32 v61, vcc, v61, v65, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v83, s[28:29], 0, v70, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v83
	v_subb_co_u32_e64 v70, s[14:15], v70, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v84, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v82
	v_cmp_le_u32_e32 vcc, s69, v61
	v_mul_lo_u32 v51, v51, s33
	v_cndmask_b32_e64 v85, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v83
	v_cndmask_b32_e64 v65, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v63
	v_cndmask_b32_e64 v84, v84, v85, s[28:29]
	v_subrev_co_u32_e64 v85, s[14:15], s68, v82
	s_nop 1
	v_subbrev_co_u32_e64 v70, s[14:15], 0, v70, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v84
	s_nop 1
	v_cndmask_b32_e64 v70, v83, v70, s[14:15]
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v61
	s_nop 1
	v_cndmask_b32_e32 v65, v65, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v65
	v_cndmask_b32_e64 v65, v82, v85, s[14:15]
	v_or_b32_e32 v82, 7, v88
	v_cndmask_b32_e32 v63, v63, v65, vcc
	v_cndmask_b32_e32 v61, v61, v70, vcc
	v_xor_b32_e32 v63, v63, v78
	v_xor_b32_e32 v61, v61, v78
	v_sub_co_u32_e32 v63, vcc, v63, v78
	v_mov_b32_e32 v83, v89
	s_nop 0
	v_subb_co_u32_e32 v61, vcc, v61, v78, vcc
	v_ashrrev_i32_e32 v61, 31, v61
	v_and_b32_e32 v61, s44, v61
	v_add3_u32 v61, v61, v63, v51
	v_sub_co_u32_e32 v51, vcc, -8, v88
	v_cndmask_b32_e64 v84, v82, v51, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v63, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v85, v89, v63, s[22:23]
	v_ashrrev_i32_e32 v86, 31, v85
	v_mov_b32_e32 v87, v86
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_xor_b32_e32 v63, v84, v86
	v_xor_b32_e32 v51, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v63, s57, 0
	v_mul_hi_u32 v70, v63, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[94:95], s[14:15], v51, s56, 0
	v_add_co_u32_e32 v65, vcc, v84, v94
	v_mad_u64_u32 v[92:93], s[14:15], v51, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v95, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v65, s69, v84
	v_mul_lo_u32 v70, s68, v85
	v_mad_u64_u32 v[92:93], s[14:15], s68, v84, 0
	v_add3_u32 v65, v93, v70, v65
	v_sub_co_u32_e32 v63, vcc, v63, v92
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[78:79]
	s_nop 0
	v_subb_co_u32_e64 v70, s[14:15], v51, v65, vcc
	v_sub_u32_e32 v51, v51, v65
	v_cmp_le_u32_e64 s[14:15], s69, v70
	v_subb_co_u32_e32 v51, vcc, v51, v33, vcc
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v63
	v_subrev_co_u32_e32 v63, vcc, s68, v63
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v51, vcc, 0, v51, vcc
	v_cmp_le_u32_e32 vcc, s69, v51
	v_cmp_eq_u32_e64 s[14:15], s69, v70
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v63
	v_cndmask_b32_e64 v70, v85, v87, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v51
	s_nop 1
	v_cndmask_b32_e32 v51, v65, v63, vcc
	v_cmp_ne_u32_e32 vcc, 0, v51
	v_xor_b32_e32 v63, s72, v86
	v_xor_b32_e32 v65, v82, v78
	v_cndmask_b32_e64 v51, 1, 2, vcc
	v_add_u32_e32 v51, v84, v51
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_mul_hi_u32 v70, v65, s56
	s_nop 0
	v_cndmask_b32_e32 v51, v84, v51, vcc
	v_xor_b32_e32 v51, v51, v63
	v_sub_u32_e32 v51, v51, v63
	v_xor_b32_e32 v63, v83, v78
	v_mad_u64_u32 v[82:83], s[14:15], v65, s57, 0
	v_lshl_add_u64 v[82:83], v[70:71], 0, v[82:83]
	v_mad_u64_u32 v[86:87], s[14:15], v63, s56, 0
	v_add_co_u32_e32 v70, vcc, v82, v86
	v_mad_u64_u32 v[84:85], s[14:15], v63, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v87, vcc
	v_mov_b32_e32 v83, s74
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_mul_lo_u32 v70, s69, v82
	v_mul_lo_u32 v84, s68, v83
	v_mad_u64_u32 v[82:83], s[14:15], s68, v82, 0
	v_add3_u32 v70, v83, v84, v70
	v_sub_u32_e32 v83, v63, v70
	v_sub_co_u32_e32 v65, vcc, v65, v82
	v_xad_u32 v51, v51, v78, v37
	s_nop 0
	v_subb_co_u32_e64 v82, s[14:15], v83, v33, vcc
	v_subrev_co_u32_e64 v83, s[14:15], s68, v65
	v_subb_co_u32_e32 v63, vcc, v63, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v84, s[28:29], 0, v82, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v84
	v_subb_co_u32_e64 v82, s[14:15], v82, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v83
	v_cmp_le_u32_e32 vcc, s69, v63
	v_mul_lo_u32 v51, v51, s33
	v_cndmask_b32_e64 v86, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v84
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v65
	v_cndmask_b32_e64 v85, v85, v86, s[28:29]
	v_subrev_co_u32_e64 v86, s[14:15], s68, v83
	s_nop 1
	v_subbrev_co_u32_e64 v82, s[14:15], 0, v82, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v85
	s_nop 1
	v_cndmask_b32_e64 v82, v84, v82, s[14:15]
	v_cndmask_b32_e64 v84, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v63
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v84, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_cndmask_b32_e64 v70, v83, v86, s[14:15]
	v_mov_b32_e32 v83, v89
	v_cndmask_b32_e32 v65, v65, v70, vcc
	v_cndmask_b32_e32 v63, v63, v82, vcc
	v_xor_b32_e32 v65, v65, v78
	v_xor_b32_e32 v63, v63, v78
	v_sub_co_u32_e32 v65, vcc, v65, v78
	v_or_b32_e32 v82, 8, v88
	s_nop 0
	v_subb_co_u32_e32 v63, vcc, v63, v78, vcc
	v_ashrrev_i32_e32 v63, 31, v63
	v_and_b32_e32 v63, s44, v63
	v_add3_u32 v63, v63, v65, v51
	buffer_load_ubyte v35, v35, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v51, v39, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v39, v53, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v53, v55, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v55, v57, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v59, v59, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v57, v61, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v61, v63, s[36:39], 0 offen
	v_sub_co_u32_e32 v63, vcc, -9, v88
	v_cndmask_b32_e64 v84, v82, v63, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v65, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v85, v89, v65, s[22:23]
	v_ashrrev_i32_e32 v86, 31, v85
	v_mov_b32_e32 v87, v86
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_xor_b32_e32 v65, v84, v86
	v_xor_b32_e32 v63, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v65, s57, 0
	v_mul_hi_u32 v70, v65, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[94:95], s[14:15], v63, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v94
	v_mad_u64_u32 v[92:93], s[14:15], v63, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v95, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v85, s68, v85
	v_mad_u64_u32 v[92:93], s[14:15], s68, v84, 0
	v_add3_u32 v70, v93, v85, v70
	v_sub_co_u32_e32 v65, vcc, v65, v92
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[78:79]
	s_nop 0
	v_subb_co_u32_e64 v85, s[14:15], v63, v70, vcc
	v_sub_u32_e32 v63, v63, v70
	v_cmp_le_u32_e64 s[14:15], s69, v85
	v_subb_co_u32_e32 v63, vcc, v63, v33, vcc
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v65
	v_subrev_co_u32_e32 v65, vcc, s68, v65
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v63, vcc, 0, v63, vcc
	v_cmp_le_u32_e32 vcc, s69, v63
	v_cmp_eq_u32_e64 s[14:15], s69, v85
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v65
	v_cndmask_b32_e64 v85, v87, v92, s[14:15]
	v_xor_b32_e32 v92, v82, v78
	v_cndmask_b32_e64 v65, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v63
	s_nop 1
	v_cndmask_b32_e32 v63, v70, v65, vcc
	v_cmp_ne_u32_e32 vcc, 0, v63
	v_xor_b32_e32 v65, s72, v86
	v_mul_hi_u32 v70, v92, s56
	v_cndmask_b32_e64 v63, 1, 2, vcc
	v_add_u32_e32 v63, v84, v63
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v63, v84, v63, vcc
	v_xor_b32_e32 v63, v63, v65
	v_sub_u32_e32 v63, v63, v65
	v_xor_b32_e32 v65, v83, v78
	v_mad_u64_u32 v[82:83], s[14:15], v92, s57, 0
	v_lshl_add_u64 v[82:83], v[70:71], 0, v[82:83]
	v_mad_u64_u32 v[86:87], s[14:15], v65, s56, 0
	v_add_co_u32_e32 v70, vcc, v82, v86
	v_mad_u64_u32 v[84:85], s[14:15], v65, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v87, vcc
	v_mov_b32_e32 v83, s74
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_mul_lo_u32 v70, s69, v82
	v_mul_lo_u32 v84, s68, v83
	v_mad_u64_u32 v[82:83], s[14:15], s68, v82, 0
	v_add3_u32 v70, v83, v84, v70
	v_sub_u32_e32 v83, v65, v70
	v_sub_co_u32_e32 v82, vcc, v92, v82
	v_xad_u32 v63, v63, v78, v37
	s_nop 0
	v_subb_co_u32_e64 v83, s[14:15], v83, v33, vcc
	v_subrev_co_u32_e64 v84, s[14:15], s68, v82
	v_subb_co_u32_e32 v65, vcc, v65, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[28:29], 0, v83, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v85
	v_subb_co_u32_e64 v83, s[14:15], v83, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v86, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v84
	v_cmp_le_u32_e32 vcc, s69, v65
	v_mul_lo_u32 v63, v63, s33
	v_cndmask_b32_e64 v87, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v85
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v82
	v_cndmask_b32_e64 v86, v86, v87, s[28:29]
	v_subrev_co_u32_e64 v87, s[14:15], s68, v84
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[14:15], 0, v83, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v86
	s_nop 1
	v_cndmask_b32_e64 v83, v85, v83, s[14:15]
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v65
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_cndmask_b32_e64 v70, v84, v87, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v82, v70, vcc
	v_cndmask_b32_e32 v65, v65, v83, vcc
	v_xor_b32_e32 v70, v70, v78
	v_xor_b32_e32 v65, v65, v78
	v_sub_co_u32_e32 v70, vcc, v70, v78
	v_or_b32_e32 v82, 9, v88
	s_nop 0
	v_subb_co_u32_e32 v65, vcc, v65, v78, vcc
	v_ashrrev_i32_e32 v65, 31, v65
	v_and_b32_e32 v65, s44, v65
	v_add3_u32 v63, v65, v70, v63
	v_sub_co_u32_e32 v65, vcc, -10, v88
	v_cndmask_b32_e64 v84, v82, v65, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v70, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v85, v89, v70, s[22:23]
	v_ashrrev_i32_e32 v86, 31, v85
	v_mov_b32_e32 v87, v86
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_xor_b32_e32 v87, v84, v86
	v_xor_b32_e32 v65, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v87, s57, 0
	v_mul_hi_u32 v70, v87, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[94:95], s[14:15], v65, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v94
	v_mad_u64_u32 v[92:93], s[14:15], v65, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v95, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v85, s68, v85
	v_mad_u64_u32 v[92:93], s[14:15], s68, v84, 0
	v_add3_u32 v70, v93, v85, v70
	v_sub_co_u32_e32 v85, vcc, v87, v92
	v_mov_b32_e32 v83, v89
	s_nop 0
	v_subb_co_u32_e64 v87, s[14:15], v65, v70, vcc
	v_sub_u32_e32 v65, v65, v70
	v_subb_co_u32_e32 v65, vcc, v65, v33, vcc
	v_subrev_co_u32_e32 v70, vcc, s68, v85
	v_cmp_le_u32_e64 s[14:15], s69, v87
	s_nop 0
	v_subbrev_co_u32_e32 v65, vcc, 0, v65, vcc
	v_cmp_le_u32_e32 vcc, s69, v65
	v_cndmask_b32_e64 v92, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v85
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v70
	v_cndmask_b32_e64 v93, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v87
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v65
	v_cndmask_b32_e64 v87, v92, v93, s[14:15]
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[78:79]
	v_cndmask_b32_e32 v65, v85, v70, vcc
	v_cmp_ne_u32_e32 vcc, 0, v65
	v_xor_b32_e32 v70, s72, v86
	v_xor_b32_e32 v93, v82, v78
	v_cndmask_b32_e64 v65, 1, 2, vcc
	v_add_u32_e32 v65, v84, v65
	v_cmp_ne_u32_e32 vcc, 0, v87
	v_xor_b32_e32 v92, v83, v78
	v_mad_u64_u32 v[82:83], s[14:15], v93, s57, 0
	v_cndmask_b32_e32 v65, v84, v65, vcc
	v_xor_b32_e32 v65, v65, v70
	v_sub_u32_e32 v65, v65, v70
	v_mul_hi_u32 v70, v93, s56
	v_lshl_add_u64 v[82:83], v[70:71], 0, v[82:83]
	v_mad_u64_u32 v[86:87], s[14:15], v92, s56, 0
	v_add_co_u32_e32 v70, vcc, v82, v86
	v_mad_u64_u32 v[84:85], s[14:15], v92, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v87, vcc
	v_mov_b32_e32 v83, s74
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_mul_lo_u32 v70, s69, v82
	v_mul_lo_u32 v84, s68, v83
	v_mad_u64_u32 v[82:83], s[14:15], s68, v82, 0
	v_add3_u32 v70, v83, v84, v70
	v_sub_u32_e32 v83, v92, v70
	v_sub_co_u32_e32 v82, vcc, v93, v82
	v_xad_u32 v65, v65, v78, v37
	s_nop 0
	v_subb_co_u32_e64 v83, s[14:15], v83, v33, vcc
	v_subrev_co_u32_e64 v84, s[14:15], s68, v82
	v_subb_co_u32_e32 v70, vcc, v92, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[28:29], 0, v83, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v85
	v_subb_co_u32_e64 v83, s[14:15], v83, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v86, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v84
	v_cmp_le_u32_e32 vcc, s69, v70
	v_mul_lo_u32 v65, v65, s33
	v_cndmask_b32_e64 v87, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v85
	s_nop 1
	v_cndmask_b32_e64 v86, v86, v87, s[28:29]
	v_subrev_co_u32_e64 v87, s[14:15], s68, v84
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[14:15], 0, v83, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v86
	s_nop 1
	v_cndmask_b32_e64 v83, v85, v83, s[14:15]
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v82
	s_nop 1
	v_cndmask_b32_e64 v86, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v85, v85, v86, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v83, vcc
	v_cndmask_b32_e64 v83, v84, v87, s[14:15]
	v_cndmask_b32_e32 v82, v82, v83, vcc
	v_xor_b32_e32 v82, v82, v78
	v_xor_b32_e32 v70, v70, v78
	v_sub_co_u32_e32 v82, vcc, v82, v78
	v_mov_b32_e32 v83, v89
	s_nop 0
	v_subb_co_u32_e32 v70, vcc, v70, v78, vcc
	v_ashrrev_i32_e32 v70, 31, v70
	v_and_b32_e32 v70, s44, v70
	v_add3_u32 v65, v70, v82, v65
	v_sub_co_u32_e32 v70, vcc, -11, v88
	v_or_b32_e32 v82, 10, v88
	s_nop 0
	v_subb_co_u32_e32 v84, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v85, v89, v84, s[22:23]
	v_ashrrev_i32_e32 v86, 31, v85
	v_cndmask_b32_e64 v84, v82, v70, s[22:23]
	v_mov_b32_e32 v87, v86
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_xor_b32_e32 v101, v84, v86
	v_xor_b32_e32 v87, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v101, s57, 0
	v_mul_hi_u32 v70, v101, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[94:95], s[14:15], v87, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v94
	v_mad_u64_u32 v[92:93], s[14:15], v87, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v95, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v85, s68, v85
	v_mad_u64_u32 v[92:93], s[14:15], s68, v84, 0
	v_add3_u32 v70, v93, v85, v70
	v_sub_co_u32_e32 v85, vcc, v101, v92
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[78:79]
	s_nop 0
	v_subb_co_u32_e64 v92, s[14:15], v87, v70, vcc
	v_sub_u32_e32 v70, v87, v70
	v_cmp_le_u32_e64 s[14:15], s69, v92
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v85
	v_subrev_co_u32_e32 v85, vcc, s68, v85
	s_nop 0
	v_cndmask_b32_e64 v94, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cmp_eq_u32_e64 s[14:15], s69, v92
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v85
	v_cndmask_b32_e64 v92, v93, v94, s[14:15]
	v_xor_b32_e32 v94, v82, v78
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_xor_b32_e32 v93, v83, v78
	v_mad_u64_u32 v[82:83], s[14:15], v94, s57, 0
	v_cndmask_b32_e32 v70, v87, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_nop 1
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v84, v70
	v_cmp_ne_u32_e32 vcc, 0, v92
	s_nop 1
	v_cndmask_b32_e32 v70, v84, v70, vcc
	v_xor_b32_e32 v84, s72, v86
	v_xor_b32_e32 v70, v70, v84
	v_sub_u32_e32 v70, v70, v84
	v_xad_u32 v92, v70, v78, v37
	v_mul_hi_u32 v70, v94, s56
	v_lshl_add_u64 v[82:83], v[70:71], 0, v[82:83]
	v_mad_u64_u32 v[86:87], s[14:15], v93, s56, 0
	v_add_co_u32_e32 v70, vcc, v82, v86
	v_mad_u64_u32 v[84:85], s[14:15], v93, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v87, vcc
	v_mov_b32_e32 v83, s74
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_mul_lo_u32 v70, s69, v82
	v_mul_lo_u32 v84, s68, v83
	v_mad_u64_u32 v[82:83], s[14:15], s68, v82, 0
	v_add3_u32 v70, v83, v84, v70
	v_sub_u32_e32 v83, v93, v70
	v_sub_co_u32_e32 v82, vcc, v94, v82
	s_nop 1
	v_subb_co_u32_e64 v83, s[14:15], v83, v33, vcc
	v_subrev_co_u32_e64 v84, s[14:15], s68, v82
	v_subb_co_u32_e32 v70, vcc, v93, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[28:29], 0, v83, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v85
	v_subb_co_u32_e64 v83, s[14:15], v83, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v86, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v84
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v85
	s_nop 1
	v_cndmask_b32_e64 v86, v86, v87, s[28:29]
	v_subrev_co_u32_e64 v87, s[14:15], s68, v84
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[14:15], 0, v83, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v86
	s_nop 1
	v_cndmask_b32_e64 v83, v85, v83, s[14:15]
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v82
	s_nop 1
	v_cndmask_b32_e64 v86, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v85, v85, v86, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	v_mov_b32_e32 v85, v89
	s_nop 0
	v_cndmask_b32_e32 v70, v70, v83, vcc
	v_cndmask_b32_e64 v83, v84, v87, s[14:15]
	v_cndmask_b32_e32 v82, v82, v83, vcc
	v_xor_b32_e32 v82, v82, v78
	v_xor_b32_e32 v70, v70, v78
	v_sub_co_u32_e32 v82, vcc, v82, v78
	v_mul_lo_u32 v83, v92, s33
	s_nop 0
	v_subb_co_u32_e32 v70, vcc, v70, v78, vcc
	v_ashrrev_i32_e32 v70, 31, v70
	v_and_b32_e32 v70, s44, v70
	v_add3_u32 v82, v70, v82, v83
	v_sub_co_u32_e32 v70, vcc, -12, v88
	v_or_b32_e32 v84, 11, v88
	s_nop 0
	v_subb_co_u32_e32 v83, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v87, v89, v83, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v87
	v_cndmask_b32_e64 v86, v84, v70, s[22:23]
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[92:93]
	v_xor_b32_e32 v93, v86, v92
	v_xor_b32_e32 v83, v87, v92
	v_mad_u64_u32 v[86:87], s[14:15], v93, s57, 0
	v_mul_hi_u32 v70, v93, s56
	v_lshl_add_u64 v[86:87], v[70:71], 0, v[86:87]
	v_mad_u64_u32 v[102:103], s[14:15], v83, s56, 0
	v_add_co_u32_e32 v70, vcc, v86, v102
	v_mad_u64_u32 v[94:95], s[14:15], v83, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v86, vcc, v87, v103, vcc
	v_mov_b32_e32 v87, s74
	s_nop 0
	v_addc_co_u32_e32 v95, vcc, 0, v95, vcc
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[94:95]
	v_mul_lo_u32 v70, s69, v86
	v_mul_lo_u32 v87, s68, v87
	v_mad_u64_u32 v[94:95], s[14:15], s68, v86, 0
	v_add3_u32 v70, v95, v87, v70
	v_sub_co_u32_e32 v87, vcc, v93, v94
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[78:79]
	s_nop 0
	v_subb_co_u32_e64 v93, s[14:15], v83, v70, vcc
	v_sub_u32_e32 v70, v83, v70
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	v_subrev_co_u32_e32 v83, vcc, s68, v87
	v_cmp_le_u32_e64 s[14:15], s69, v93
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v94, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v87
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v83
	v_cndmask_b32_e64 v95, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v93
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v93, v94, v95, s[14:15]
	v_xor_b32_e32 v95, v84, v78
	v_cndmask_b32_e32 v70, v87, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v83, s72, v92
	v_xor_b32_e32 v94, v85, v78
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v86, v70
	v_cmp_ne_u32_e32 vcc, 0, v93
	v_mad_u64_u32 v[84:85], s[14:15], v95, s57, 0
	s_nop 0
	v_cndmask_b32_e32 v70, v86, v70, vcc
	v_xor_b32_e32 v70, v70, v83
	v_sub_u32_e32 v70, v70, v83
	v_xad_u32 v83, v70, v78, v37
	v_mul_hi_u32 v70, v95, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[92:93], s[14:15], v94, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v92
	v_mad_u64_u32 v[86:87], s[14:15], v94, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v93, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v86, s68, v85
	v_mad_u64_u32 v[84:85], s[14:15], s68, v84, 0
	v_add3_u32 v70, v85, v86, v70
	v_sub_u32_e32 v85, v94, v70
	v_sub_co_u32_e32 v84, vcc, v95, v84
	v_mul_lo_u32 v83, v83, s33
	s_nop 0
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, vcc
	v_subrev_co_u32_e64 v86, s[14:15], s68, v84
	v_subb_co_u32_e32 v70, vcc, v94, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v87, s[28:29], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v87
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v86
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v87
	s_nop 1
	v_cndmask_b32_e64 v92, v92, v93, s[28:29]
	v_subrev_co_u32_e64 v93, s[14:15], s68, v86
	s_nop 1
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v85, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v92
	s_nop 1
	v_cndmask_b32_e64 v85, v87, v85, s[14:15]
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v84
	s_nop 1
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v87, v87, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v87
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v85, vcc
	v_cndmask_b32_e64 v85, v86, v93, s[14:15]
	v_cndmask_b32_e32 v84, v84, v85, vcc
	v_xor_b32_e32 v84, v84, v78
	v_xor_b32_e32 v70, v70, v78
	v_sub_co_u32_e32 v84, vcc, v84, v78
	v_mov_b32_e32 v85, v89
	s_nop 0
	v_subb_co_u32_e32 v70, vcc, v70, v78, vcc
	v_ashrrev_i32_e32 v70, 31, v70
	v_and_b32_e32 v70, s44, v70
	v_add3_u32 v83, v70, v84, v83
	v_sub_co_u32_e32 v70, vcc, -13, v88
	v_or_b32_e32 v84, 12, v88
	s_nop 0
	v_subb_co_u32_e32 v86, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v87, v89, v86, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v87
	v_cndmask_b32_e64 v86, v84, v70, s[22:23]
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[92:93]
	v_xor_b32_e32 v101, v86, v92
	v_xor_b32_e32 v93, v87, v92
	v_mad_u64_u32 v[86:87], s[14:15], v101, s57, 0
	v_mul_hi_u32 v70, v101, s56
	v_lshl_add_u64 v[86:87], v[70:71], 0, v[86:87]
	v_mad_u64_u32 v[102:103], s[14:15], v93, s56, 0
	v_add_co_u32_e32 v70, vcc, v86, v102
	v_mad_u64_u32 v[94:95], s[14:15], v93, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v86, vcc, v87, v103, vcc
	v_mov_b32_e32 v87, s74
	s_nop 0
	v_addc_co_u32_e32 v95, vcc, 0, v95, vcc
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[94:95]
	v_mul_lo_u32 v70, s69, v86
	v_mul_lo_u32 v87, s68, v87
	v_mad_u64_u32 v[94:95], s[14:15], s68, v86, 0
	v_add3_u32 v70, v95, v87, v70
	v_sub_co_u32_e32 v87, vcc, v101, v94
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[78:79]
	s_nop 0
	v_subb_co_u32_e64 v94, s[14:15], v93, v70, vcc
	v_sub_u32_e32 v70, v93, v70
	v_cmp_le_u32_e64 s[14:15], s69, v94
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v87
	v_subrev_co_u32_e32 v87, vcc, s68, v87
	s_nop 0
	v_cndmask_b32_e64 v101, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cmp_eq_u32_e64 s[14:15], s69, v94
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v87
	v_cndmask_b32_e64 v94, v95, v101, s[14:15]
	v_xor_b32_e32 v101, v84, v78
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_xor_b32_e32 v95, v85, v78
	v_mad_u64_u32 v[84:85], s[14:15], v101, s57, 0
	v_cndmask_b32_e32 v70, v93, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_nop 1
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v86, v70
	v_cmp_ne_u32_e32 vcc, 0, v94
	s_nop 1
	v_cndmask_b32_e32 v70, v86, v70, vcc
	v_xor_b32_e32 v86, s72, v92
	v_xor_b32_e32 v70, v70, v86
	v_sub_u32_e32 v70, v70, v86
	v_xad_u32 v94, v70, v78, v37
	v_mul_hi_u32 v70, v101, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[92:93], s[14:15], v95, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v92
	v_mad_u64_u32 v[86:87], s[14:15], v95, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v93, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v86, s68, v85
	v_mad_u64_u32 v[84:85], s[14:15], s68, v84, 0
	v_add3_u32 v70, v85, v86, v70
	v_sub_u32_e32 v85, v95, v70
	v_sub_co_u32_e32 v84, vcc, v101, v84
	s_nop 1
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, vcc
	v_subrev_co_u32_e64 v86, s[14:15], s68, v84
	v_subb_co_u32_e32 v70, vcc, v95, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v87, s[28:29], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v87
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v86
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v87
	s_nop 1
	v_cndmask_b32_e64 v92, v92, v93, s[28:29]
	v_subrev_co_u32_e64 v93, s[14:15], s68, v86
	s_nop 1
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v85, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v92
	s_nop 1
	v_cndmask_b32_e64 v85, v87, v85, s[14:15]
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v84
	s_nop 1
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v87, v87, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v87
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v85, vcc
	v_cndmask_b32_e64 v85, v86, v93, s[14:15]
	v_cndmask_b32_e32 v84, v84, v85, vcc
	v_xor_b32_e32 v84, v84, v78
	v_xor_b32_e32 v70, v70, v78
	v_sub_co_u32_e32 v84, vcc, v84, v78
	v_mul_lo_u32 v85, v94, s33
	s_nop 0
	v_subb_co_u32_e32 v70, vcc, v70, v78, vcc
	v_ashrrev_i32_e32 v70, 31, v70
	v_and_b32_e32 v70, s44, v70
	v_add3_u32 v104, v70, v84, v85
	v_sub_co_u32_e32 v70, vcc, -14, v88
	v_or_b32_e32 v84, 13, v88
	s_nop 0
	v_subb_co_u32_e32 v86, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v87, v89, v86, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v87
	v_cndmask_b32_e64 v86, v84, v70, s[22:23]
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[92:93]
	v_xor_b32_e32 v101, v86, v92
	v_xor_b32_e32 v93, v87, v92
	v_mad_u64_u32 v[86:87], s[14:15], v101, s57, 0
	v_mul_hi_u32 v70, v101, s56
	v_lshl_add_u64 v[86:87], v[70:71], 0, v[86:87]
	v_mad_u64_u32 v[102:103], s[14:15], v93, s56, 0
	v_add_co_u32_e32 v70, vcc, v86, v102
	v_mad_u64_u32 v[94:95], s[14:15], v93, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v86, vcc, v87, v103, vcc
	v_mov_b32_e32 v87, s74
	s_nop 0
	v_addc_co_u32_e32 v95, vcc, 0, v95, vcc
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[94:95]
	v_mul_lo_u32 v70, s69, v86
	v_mul_lo_u32 v87, s68, v87
	v_mad_u64_u32 v[94:95], s[14:15], s68, v86, 0
	v_add3_u32 v70, v95, v87, v70
	v_sub_co_u32_e32 v87, vcc, v101, v94
	v_mov_b32_e32 v85, v89
	s_nop 0
	v_subb_co_u32_e64 v94, s[14:15], v93, v70, vcc
	v_sub_u32_e32 v70, v93, v70
	v_cmp_le_u32_e64 s[14:15], s69, v94
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v87
	v_subrev_co_u32_e32 v87, vcc, s68, v87
	s_nop 0
	v_cndmask_b32_e64 v101, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cmp_eq_u32_e64 s[14:15], s69, v94
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[78:79]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v87
	v_cndmask_b32_e64 v94, v95, v101, s[14:15]
	v_xor_b32_e32 v101, v84, v78
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_xor_b32_e32 v95, v85, v78
	v_mad_u64_u32 v[84:85], s[14:15], v101, s57, 0
	v_cndmask_b32_e32 v70, v93, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_nop 1
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v86, v70
	v_cmp_ne_u32_e32 vcc, 0, v94
	s_nop 1
	v_cndmask_b32_e32 v70, v86, v70, vcc
	v_xor_b32_e32 v86, s72, v92
	v_xor_b32_e32 v70, v70, v86
	v_sub_u32_e32 v70, v70, v86
	v_xad_u32 v94, v70, v78, v37
	v_mul_hi_u32 v70, v101, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[92:93], s[14:15], v95, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v92
	v_mad_u64_u32 v[86:87], s[14:15], v95, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v93, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v86, s68, v85
	v_mad_u64_u32 v[84:85], s[14:15], s68, v84, 0
	v_add3_u32 v70, v85, v86, v70
	v_sub_u32_e32 v85, v95, v70
	v_sub_co_u32_e32 v84, vcc, v101, v84
	s_nop 1
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, vcc
	v_subrev_co_u32_e64 v86, s[14:15], s68, v84
	v_subb_co_u32_e32 v70, vcc, v95, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v87, s[28:29], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v87
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v86
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v87
	s_nop 1
	v_cndmask_b32_e64 v92, v92, v93, s[28:29]
	v_subrev_co_u32_e64 v93, s[14:15], s68, v86
	s_nop 1
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v85, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v92
	s_nop 1
	v_cndmask_b32_e64 v85, v87, v85, s[14:15]
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v84
	s_nop 1
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v87, v87, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v87
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v85, vcc
	v_cndmask_b32_e64 v85, v86, v93, s[14:15]
	v_cndmask_b32_e32 v84, v84, v85, vcc
	v_xor_b32_e32 v84, v84, v78
	v_xor_b32_e32 v70, v70, v78
	v_sub_co_u32_e32 v84, vcc, v84, v78
	v_mul_lo_u32 v85, v94, s33
	s_nop 0
	v_subb_co_u32_e32 v70, vcc, v70, v78, vcc
	v_ashrrev_i32_e32 v70, 31, v70
	v_and_b32_e32 v70, s44, v70
	v_add3_u32 v105, v70, v84, v85
	v_sub_co_u32_e32 v70, vcc, -15, v88
	v_or_b32_e32 v84, 14, v88
	s_nop 0
	v_subb_co_u32_e32 v86, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v87, v89, v86, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v87
	v_cndmask_b32_e64 v86, v84, v70, s[22:23]
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[92:93]
	v_xor_b32_e32 v101, v86, v92
	v_xor_b32_e32 v93, v87, v92
	v_mad_u64_u32 v[86:87], s[14:15], v101, s57, 0
	v_mul_hi_u32 v70, v101, s56
	v_lshl_add_u64 v[86:87], v[70:71], 0, v[86:87]
	v_mad_u64_u32 v[102:103], s[14:15], v93, s56, 0
	v_add_co_u32_e32 v70, vcc, v86, v102
	v_mad_u64_u32 v[94:95], s[14:15], v93, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v86, vcc, v87, v103, vcc
	v_mov_b32_e32 v87, s74
	s_nop 0
	v_addc_co_u32_e32 v95, vcc, 0, v95, vcc
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[94:95]
	v_mul_lo_u32 v70, s69, v86
	v_mul_lo_u32 v87, s68, v87
	v_mad_u64_u32 v[94:95], s[14:15], s68, v86, 0
	v_add3_u32 v70, v95, v87, v70
	v_sub_co_u32_e32 v87, vcc, v101, v94
	v_mov_b32_e32 v85, v89
	s_nop 0
	v_subb_co_u32_e64 v94, s[14:15], v93, v70, vcc
	v_sub_u32_e32 v70, v93, v70
	v_cmp_le_u32_e64 s[14:15], s69, v94
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v87
	v_subrev_co_u32_e32 v87, vcc, s68, v87
	s_nop 0
	v_cndmask_b32_e64 v101, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cmp_eq_u32_e64 s[14:15], s69, v94
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[78:79]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v87
	v_cndmask_b32_e64 v94, v95, v101, s[14:15]
	v_xor_b32_e32 v101, v84, v78
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_xor_b32_e32 v95, v85, v78
	v_mad_u64_u32 v[84:85], s[14:15], v101, s57, 0
	v_cndmask_b32_e32 v70, v93, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_nop 1
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v86, v70
	v_cmp_ne_u32_e32 vcc, 0, v94
	s_nop 1
	v_cndmask_b32_e32 v70, v86, v70, vcc
	v_xor_b32_e32 v86, s72, v92
	v_xor_b32_e32 v70, v70, v86
	v_sub_u32_e32 v70, v70, v86
	v_xad_u32 v94, v70, v78, v37
	v_mul_hi_u32 v70, v101, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[92:93], s[14:15], v95, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v92
	v_mad_u64_u32 v[86:87], s[14:15], v95, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v93, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v86, s68, v85
	v_mad_u64_u32 v[84:85], s[14:15], s68, v84, 0
	v_add3_u32 v70, v85, v86, v70
	v_sub_u32_e32 v85, v95, v70
	v_sub_co_u32_e32 v84, vcc, v101, v84
	s_nop 1
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, vcc
	v_subrev_co_u32_e64 v86, s[14:15], s68, v84
	v_subb_co_u32_e32 v70, vcc, v95, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v87, s[28:29], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v87
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v86
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v87
	s_nop 1
	v_cndmask_b32_e64 v92, v92, v93, s[28:29]
	v_subrev_co_u32_e64 v93, s[14:15], s68, v86
	s_nop 1
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v85, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v92
	s_nop 1
	v_cndmask_b32_e64 v85, v87, v85, s[14:15]
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v84
	s_nop 1
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v87, v87, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v87
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v85, vcc
	v_cndmask_b32_e64 v85, v86, v93, s[14:15]
	v_cndmask_b32_e32 v84, v84, v85, vcc
	v_xor_b32_e32 v84, v84, v78
	v_xor_b32_e32 v70, v70, v78
	v_sub_co_u32_e32 v84, vcc, v84, v78
	v_mul_lo_u32 v85, v94, s33
	s_nop 0
	v_subb_co_u32_e32 v70, vcc, v70, v78, vcc
	v_ashrrev_i32_e32 v70, 31, v70
	v_and_b32_e32 v70, s44, v70
	v_add3_u32 v107, v70, v84, v85
	v_sub_co_u32_e32 v70, vcc, -16, v88
	v_or_b32_e32 v84, 15, v88
	s_nop 0
	v_subb_co_u32_e32 v86, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v87, v89, v86, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v87
	v_cndmask_b32_e64 v86, v84, v70, s[22:23]
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[92:93]
	v_xor_b32_e32 v101, v86, v92
	v_xor_b32_e32 v93, v87, v92
	v_mad_u64_u32 v[86:87], s[14:15], v101, s57, 0
	v_mul_hi_u32 v70, v101, s56
	v_lshl_add_u64 v[86:87], v[70:71], 0, v[86:87]
	v_mad_u64_u32 v[102:103], s[14:15], v93, s56, 0
	v_add_co_u32_e32 v70, vcc, v86, v102
	v_mad_u64_u32 v[94:95], s[14:15], v93, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v86, vcc, v87, v103, vcc
	v_mov_b32_e32 v87, s74
	s_nop 0
	v_addc_co_u32_e32 v95, vcc, 0, v95, vcc
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[94:95]
	v_mul_lo_u32 v70, s69, v86
	v_mul_lo_u32 v87, s68, v87
	v_mad_u64_u32 v[94:95], s[14:15], s68, v86, 0
	v_add3_u32 v70, v95, v87, v70
	v_sub_co_u32_e32 v87, vcc, v101, v94
	v_mov_b32_e32 v85, v89
	s_nop 0
	v_subb_co_u32_e64 v94, s[14:15], v93, v70, vcc
	v_sub_u32_e32 v70, v93, v70
	v_cmp_le_u32_e64 s[14:15], s69, v94
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v87
	v_subrev_co_u32_e32 v87, vcc, s68, v87
	s_nop 0
	v_cndmask_b32_e64 v101, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cmp_eq_u32_e64 s[14:15], s69, v94
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[78:79]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v87
	v_cndmask_b32_e64 v94, v95, v101, s[14:15]
	v_xor_b32_e32 v95, v84, v78
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_xor_b32_e32 v79, v85, v78
	v_mad_u64_u32 v[84:85], s[14:15], v95, s57, 0
	v_cndmask_b32_e32 v70, v93, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_nop 1
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v86, v70
	v_cmp_ne_u32_e32 vcc, 0, v94
	s_nop 1
	v_cndmask_b32_e32 v70, v86, v70, vcc
	v_xor_b32_e32 v86, s72, v92
	v_xor_b32_e32 v70, v70, v86
	v_sub_u32_e32 v70, v70, v86
	v_xad_u32 v94, v70, v78, v37
	v_mul_hi_u32 v70, v95, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[92:93], s[14:15], v79, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v92
	v_mad_u64_u32 v[86:87], s[14:15], v79, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v93, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v86, s68, v85
	v_mad_u64_u32 v[84:85], s[14:15], s68, v84, 0
	v_add3_u32 v70, v85, v86, v70
	v_sub_u32_e32 v85, v79, v70
	v_sub_co_u32_e32 v84, vcc, v95, v84
	s_nop 1
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, vcc
	v_subrev_co_u32_e64 v86, s[14:15], s68, v84
	v_subb_co_u32_e32 v70, vcc, v79, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v87, s[22:23], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s69, v87
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s68, v86
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s69, v87
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v84
	v_cndmask_b32_e64 v92, v92, v93, s[22:23]
	v_subrev_co_u32_e64 v93, s[14:15], s68, v86
	s_nop 1
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v85, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v92
	s_nop 1
	v_cndmask_b32_e64 v85, v87, v85, s[14:15]
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v79, v79, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v79
	v_cndmask_b32_e64 v79, v86, v93, s[14:15]
	s_movk_i32 s14, 0xf3ff
	v_cndmask_b32_e32 v79, v84, v79, vcc
	v_cndmask_b32_e32 v70, v70, v85, vcc
	v_xor_b32_e32 v79, v79, v78
	v_xor_b32_e32 v70, v70, v78
	v_sub_co_u32_e32 v79, vcc, v79, v78
	s_nop 1
	v_subb_co_u32_e32 v70, vcc, v70, v78, vcc
	v_ashrrev_i32_e32 v70, 31, v70
	v_and_b32_e32 v70, s44, v70
	v_mul_lo_u32 v78, v94, s33
	v_add3_u32 v70, v70, v79, v78
	buffer_load_ubyte v63, v63, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v95, v65, s[36:39], 0 offen
	buffer_load_ubyte v94, v82, s[36:39], 0 offen
	buffer_load_ubyte v101, v83, s[36:39], 0 offen
	buffer_load_ubyte v102, v104, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v104, v105, s[36:39], 0 offen
	buffer_load_ubyte v103, v107, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v105, v70, s[36:39], 0 offen
	v_sub_co_u32_e32 v65, vcc, s14, v90
	v_lshl_add_u64 v[78:79], v[90:91], 0, s[92:93]
	s_nop 0
	v_subb_co_u32_e32 v70, vcc, -1, v91, vcc
	v_cmp_gt_i32_e32 vcc, 0, v79
	s_nop 1
	v_cndmask_b32_e32 v83, v79, v70, vcc
	v_ashrrev_i32_e32 v84, 31, v83
	v_cndmask_b32_e32 v82, v78, v65, vcc
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v81, v82, v84
	v_xor_b32_e32 v65, v83, v84
	v_mad_u64_u32 v[82:83], s[14:15], v81, s57, 0
	v_mul_hi_u32 v70, v81, s56
	v_lshl_add_u64 v[82:83], v[70:71], 0, v[82:83]
	v_mad_u64_u32 v[92:93], s[14:15], v65, s56, 0
	v_add_co_u32_e32 v70, vcc, v82, v92
	v_mad_u64_u32 v[86:87], s[14:15], v65, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v93, vcc
	v_mov_b32_e32 v83, s74
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[86:87]
	v_mul_lo_u32 v70, s69, v82
	v_mul_lo_u32 v83, s68, v83
	v_mad_u64_u32 v[86:87], s[14:15], s68, v82, 0
	v_add3_u32 v70, v87, v83, v70
	v_sub_co_u32_e32 v81, vcc, v81, v86
	s_nop 1
	v_subb_co_u32_e64 v83, s[14:15], v65, v70, vcc
	v_sub_u32_e32 v65, v65, v70
	v_subb_co_u32_e32 v65, vcc, v65, v33, vcc
	v_subrev_co_u32_e32 v70, vcc, s68, v81
	v_cmp_le_u32_e64 s[14:15], s69, v83
	s_nop 0
	v_subbrev_co_u32_e32 v65, vcc, 0, v65, vcc
	v_cmp_le_u32_e32 vcc, s69, v65
	v_cndmask_b32_e64 v85, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v81
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v70
	v_cndmask_b32_e64 v86, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v83
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v65
	v_cndmask_b32_e64 v83, v85, v86, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v65, v81, v70, vcc
	v_cmp_ne_u32_e32 vcc, 0, v65
	v_xor_b32_e32 v70, s72, v84
	s_nop 0
	v_cndmask_b32_e64 v65, 1, 2, vcc
	v_add_u32_e32 v65, v82, v65
	v_cmp_ne_u32_e32 vcc, 0, v83
	s_nop 1
	v_cndmask_b32_e32 v65, v82, v65, vcc
	v_ashrrev_i32_e32 v82, 31, v79
	v_mov_b32_e32 v83, v82
	v_lshl_add_u64 v[78:79], v[78:79], 0, v[82:83]
	v_xor_b32_e32 v65, v65, v70
	v_xor_b32_e32 v83, v78, v82
	v_sub_u32_e32 v65, v65, v70
	v_xor_b32_e32 v81, v79, v82
	v_mad_u64_u32 v[78:79], s[14:15], v83, s57, 0
	v_mul_hi_u32 v70, v83, s56
	v_lshl_add_u64 v[78:79], v[70:71], 0, v[78:79]
	v_mad_u64_u32 v[86:87], s[14:15], v81, s56, 0
	v_add_co_u32_e32 v70, vcc, v78, v86
	v_mad_u64_u32 v[84:85], s[14:15], v81, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v78, vcc, v79, v87, vcc
	v_mov_b32_e32 v79, s74
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[78:79], v[78:79], 0, v[84:85]
	v_mul_lo_u32 v70, s69, v78
	v_mul_lo_u32 v84, s68, v79
	v_mad_u64_u32 v[78:79], s[14:15], s68, v78, 0
	v_add3_u32 v70, v79, v84, v70
	v_sub_u32_e32 v79, v81, v70
	v_sub_co_u32_e32 v78, vcc, v83, v78
	v_xad_u32 v65, v65, v82, v37
	s_nop 0
	v_subb_co_u32_e64 v79, s[14:15], v79, v33, vcc
	v_subrev_co_u32_e64 v83, s[14:15], s68, v78
	v_subb_co_u32_e32 v70, vcc, v81, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v84, s[22:23], 0, v79, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s69, v84
	v_subb_co_u32_e64 v79, s[14:15], v79, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s68, v83
	v_cmp_le_u32_e32 vcc, s69, v70
	v_mul_lo_u32 v65, v65, s33
	v_cndmask_b32_e64 v86, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s69, v84
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v78
	v_cndmask_b32_e64 v85, v85, v86, s[22:23]
	v_subrev_co_u32_e64 v86, s[14:15], s68, v83
	s_nop 1
	v_subbrev_co_u32_e64 v79, s[14:15], 0, v79, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v85
	s_nop 1
	v_cndmask_b32_e64 v79, v84, v79, s[14:15]
	v_cndmask_b32_e64 v84, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v81, v81, v84, vcc
	v_cmp_ne_u32_e32 vcc, 0, v81
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v79, vcc
	v_cndmask_b32_e64 v79, v83, v86, s[14:15]
	v_cndmask_b32_e32 v78, v78, v79, vcc
	v_xor_b32_e32 v78, v78, v82
	v_xor_b32_e32 v70, v70, v82
	v_sub_co_u32_e32 v78, vcc, v78, v82
	s_mov_b64 s[14:15], 0x400
	s_nop 0
	v_subb_co_u32_e32 v70, vcc, v70, v82, vcc
	v_ashrrev_i32_e32 v70, 31, v70
	v_lshl_add_u64 v[82:83], v[88:89], 0, s[14:15]
	s_mov_b64 s[14:15], 0x401
	v_and_b32_e32 v70, s44, v70
	v_lshl_add_u64 v[84:85], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xfbfe
	v_add3_u32 v78, v70, v78, v65
	v_sub_co_u32_e32 v65, vcc, s14, v88
	v_cmp_gt_i32_e64 s[22:23], 0, v83
	s_nop 0
	v_subb_co_u32_e32 v70, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v87, v85, v70, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v87
	v_cndmask_b32_e64 v86, v84, v65, s[22:23]
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[92:93]
	v_xor_b32_e32 v79, v86, v92
	v_xor_b32_e32 v65, v87, v92
	v_mad_u64_u32 v[86:87], s[14:15], v79, s57, 0
	v_mul_hi_u32 v70, v79, s56
	v_lshl_add_u64 v[86:87], v[70:71], 0, v[86:87]
	v_mad_u64_u32 v[110:111], s[14:15], v65, s56, 0
	v_add_co_u32_e32 v70, vcc, v86, v110
	v_mad_u64_u32 v[108:109], s[14:15], v65, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v86, vcc, v87, v111, vcc
	v_mov_b32_e32 v87, s74
	s_nop 0
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[108:109]
	v_mul_lo_u32 v70, s69, v86
	v_mul_lo_u32 v81, s68, v87
	v_mad_u64_u32 v[108:109], s[14:15], s68, v86, 0
	v_add3_u32 v70, v109, v81, v70
	v_sub_co_u32_e32 v79, vcc, v79, v108
	s_nop 1
	v_subb_co_u32_e64 v81, s[14:15], v65, v70, vcc
	v_sub_u32_e32 v65, v65, v70
	v_subb_co_u32_e32 v65, vcc, v65, v33, vcc
	v_subrev_co_u32_e32 v70, vcc, s68, v79
	v_cmp_le_u32_e64 s[14:15], s69, v81
	s_nop 0
	v_subbrev_co_u32_e32 v65, vcc, 0, v65, vcc
	v_cmp_le_u32_e32 vcc, s69, v65
	v_cndmask_b32_e64 v82, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v79
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v70
	v_cndmask_b32_e64 v87, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v81
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v65
	v_cndmask_b32_e64 v81, v82, v87, s[14:15]
	v_ashrrev_i32_e32 v82, 31, v85
	v_cndmask_b32_e32 v65, v79, v70, vcc
	v_cmp_ne_u32_e32 vcc, 0, v65
	v_xor_b32_e32 v70, s72, v92
	s_nop 0
	v_cndmask_b32_e64 v65, 1, 2, vcc
	v_add_u32_e32 v65, v86, v65
	v_cmp_ne_u32_e32 vcc, 0, v81
	s_nop 1
	v_cndmask_b32_e32 v65, v86, v65, vcc
	v_xor_b32_e32 v65, v65, v70
	v_sub_u32_e32 v70, v65, v70
	v_ashrrev_i32_e32 v65, 31, v83
	v_mov_b32_e32 v83, v82
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[82:83]
	v_xor_b32_e32 v83, v84, v82
	v_xad_u32 v81, v70, v65, v37
	v_xor_b32_e32 v79, v85, v82
	v_mad_u64_u32 v[84:85], s[14:15], v83, s57, 0
	v_mul_hi_u32 v70, v83, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[92:93], s[14:15], v79, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v92
	v_mad_u64_u32 v[86:87], s[14:15], v79, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v93, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v86, s68, v85
	v_mad_u64_u32 v[84:85], s[14:15], s68, v84, 0
	v_add3_u32 v70, v85, v86, v70
	v_sub_u32_e32 v85, v79, v70
	v_sub_co_u32_e32 v83, vcc, v83, v84
	v_mul_lo_u32 v81, v81, s33
	s_nop 0
	v_subb_co_u32_e64 v84, s[14:15], v85, v33, vcc
	v_subrev_co_u32_e64 v85, s[14:15], s68, v83
	v_subb_co_u32_e32 v70, vcc, v79, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v86, s[28:29], 0, v84, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v86
	v_subb_co_u32_e64 v84, s[14:15], v84, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v85
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v86
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v83
	v_cndmask_b32_e64 v87, v87, v92, s[28:29]
	v_subrev_co_u32_e64 v92, s[14:15], s68, v85
	s_nop 1
	v_subbrev_co_u32_e64 v84, s[14:15], 0, v84, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v87
	s_nop 1
	v_cndmask_b32_e64 v84, v86, v84, s[14:15]
	v_cndmask_b32_e64 v86, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v79, v79, v86, vcc
	v_cmp_ne_u32_e32 vcc, 0, v79
	v_cndmask_b32_e64 v79, v85, v92, s[14:15]
	s_mov_b64 s[14:15], 0x402
	v_cndmask_b32_e32 v70, v70, v84, vcc
	v_cndmask_b32_e32 v79, v83, v79, vcc
	v_xor_b32_e32 v83, v70, v82
	v_xor_b32_e32 v70, v79, v82
	v_sub_co_u32_e32 v70, vcc, v70, v82
	s_nop 1
	v_subb_co_u32_e32 v79, vcc, v83, v82, vcc
	v_ashrrev_i32_e32 v79, 31, v79
	v_and_b32_e32 v79, s44, v79
	v_lshl_add_u64 v[82:83], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xfbfd
	v_add3_u32 v79, v79, v70, v81
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v84, v82, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v81, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v85, v83, v81, s[22:23]
	v_ashrrev_i32_e32 v86, 31, v85
	v_mov_b32_e32 v87, v86
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_xor_b32_e32 v87, v84, v86
	v_xor_b32_e32 v81, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v87, s57, 0
	v_mul_hi_u32 v70, v87, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[108:109], s[14:15], v81, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v108
	v_mad_u64_u32 v[92:93], s[14:15], v81, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v109, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v85, s68, v85
	v_mad_u64_u32 v[92:93], s[14:15], s68, v84, 0
	v_add3_u32 v70, v93, v85, v70
	v_sub_co_u32_e32 v85, vcc, v87, v92
	s_nop 1
	v_subb_co_u32_e64 v87, s[14:15], v81, v70, vcc
	v_sub_u32_e32 v70, v81, v70
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	v_subrev_co_u32_e32 v81, vcc, s68, v85
	v_cmp_le_u32_e64 s[14:15], s69, v87
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v92, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v85
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v81
	v_cndmask_b32_e64 v93, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v87
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v87, v92, v93, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v85, v81, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v81, s72, v86
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v84, v70
	v_cmp_ne_u32_e32 vcc, 0, v87
	s_nop 1
	v_cndmask_b32_e32 v70, v84, v70, vcc
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_xor_b32_e32 v70, v70, v81
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_sub_u32_e32 v70, v70, v81
	v_xor_b32_e32 v85, v82, v84
	v_xad_u32 v107, v70, v65, v37
	v_xor_b32_e32 v81, v83, v84
	v_mad_u64_u32 v[82:83], s[14:15], v85, s57, 0
	v_mul_hi_u32 v70, v85, s56
	v_lshl_add_u64 v[82:83], v[70:71], 0, v[82:83]
	v_mad_u64_u32 v[92:93], s[14:15], v81, s56, 0
	v_add_co_u32_e32 v70, vcc, v82, v92
	v_mad_u64_u32 v[86:87], s[14:15], v81, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v93, vcc
	v_mov_b32_e32 v83, s74
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[86:87]
	v_mul_lo_u32 v70, s69, v82
	v_mul_lo_u32 v86, s68, v83
	v_mad_u64_u32 v[82:83], s[14:15], s68, v82, 0
	v_add3_u32 v70, v83, v86, v70
	v_sub_u32_e32 v83, v81, v70
	v_sub_co_u32_e32 v82, vcc, v85, v82
	s_nop 1
	v_subb_co_u32_e64 v83, s[14:15], v83, v33, vcc
	v_subrev_co_u32_e64 v85, s[14:15], s68, v82
	v_subb_co_u32_e32 v70, vcc, v81, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v86, s[28:29], 0, v83, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v86
	v_subb_co_u32_e64 v83, s[14:15], v83, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v85
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v86
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v82
	v_cndmask_b32_e64 v87, v87, v92, s[28:29]
	v_subrev_co_u32_e64 v92, s[14:15], s68, v85
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[14:15], 0, v83, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v87
	s_nop 1
	v_cndmask_b32_e64 v83, v86, v83, s[14:15]
	v_cndmask_b32_e64 v86, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v81, v81, v86, vcc
	v_cmp_ne_u32_e32 vcc, 0, v81
	v_cndmask_b32_e64 v81, v85, v92, s[14:15]
	s_mov_b64 s[14:15], 0x403
	v_cndmask_b32_e32 v70, v70, v83, vcc
	v_cndmask_b32_e32 v81, v82, v81, vcc
	v_xor_b32_e32 v82, v70, v84
	v_xor_b32_e32 v70, v81, v84
	v_sub_co_u32_e32 v70, vcc, v70, v84
	s_nop 1
	v_subb_co_u32_e32 v81, vcc, v82, v84, vcc
	v_ashrrev_i32_e32 v81, 31, v81
	v_and_b32_e32 v81, s44, v81
	v_mul_lo_u32 v82, v107, s33
	v_add3_u32 v81, v81, v70, v82
	v_lshl_add_u64 v[82:83], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xfbfc
	v_sub_co_u32_e32 v70, vcc, s14, v88
	s_nop 1
	v_subb_co_u32_e32 v84, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v85, v83, v84, s[22:23]
	v_ashrrev_i32_e32 v86, 31, v85
	v_cndmask_b32_e64 v84, v82, v70, s[22:23]
	v_mov_b32_e32 v87, v86
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_xor_b32_e32 v107, v84, v86
	v_xor_b32_e32 v87, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v107, s57, 0
	v_mul_hi_u32 v70, v107, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[108:109], s[14:15], v87, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v108
	v_mad_u64_u32 v[92:93], s[14:15], v87, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v109, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v85, s68, v85
	v_mad_u64_u32 v[92:93], s[14:15], s68, v84, 0
	v_add3_u32 v70, v93, v85, v70
	v_sub_co_u32_e32 v85, vcc, v107, v92
	s_nop 1
	v_subb_co_u32_e64 v92, s[14:15], v87, v70, vcc
	v_sub_u32_e32 v70, v87, v70
	v_cmp_le_u32_e64 s[14:15], s69, v92
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v85
	v_subrev_co_u32_e32 v85, vcc, s68, v85
	s_nop 0
	v_cndmask_b32_e64 v107, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cmp_eq_u32_e64 s[14:15], s69, v92
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v85
	v_cndmask_b32_e64 v92, v93, v107, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v70, v87, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_nop 1
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v84, v70
	v_cmp_ne_u32_e32 vcc, 0, v92
	s_nop 1
	v_cndmask_b32_e32 v70, v84, v70, vcc
	v_xor_b32_e32 v84, s72, v86
	v_xor_b32_e32 v70, v70, v84
	v_sub_u32_e32 v70, v70, v84
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v108, v82, v84
	v_xad_u32 v107, v70, v65, v37
	v_xor_b32_e32 v85, v83, v84
	v_mad_u64_u32 v[82:83], s[14:15], v108, s57, 0
	v_mul_hi_u32 v70, v108, s56
	v_lshl_add_u64 v[82:83], v[70:71], 0, v[82:83]
	v_mad_u64_u32 v[92:93], s[14:15], v85, s56, 0
	v_add_co_u32_e32 v70, vcc, v82, v92
	v_mad_u64_u32 v[86:87], s[14:15], v85, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v93, vcc
	v_mov_b32_e32 v83, s74
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[86:87]
	v_mul_lo_u32 v70, s69, v82
	v_mul_lo_u32 v86, s68, v83
	v_mad_u64_u32 v[82:83], s[14:15], s68, v82, 0
	v_add3_u32 v70, v83, v86, v70
	v_sub_u32_e32 v83, v85, v70
	v_sub_co_u32_e32 v82, vcc, v108, v82
	s_nop 1
	v_subb_co_u32_e64 v83, s[14:15], v83, v33, vcc
	v_subrev_co_u32_e64 v86, s[14:15], s68, v82
	v_subb_co_u32_e32 v70, vcc, v85, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v87, s[28:29], 0, v83, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v87
	v_subb_co_u32_e64 v83, s[14:15], v83, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v86
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v87
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v82
	v_cndmask_b32_e64 v92, v92, v93, s[28:29]
	v_subrev_co_u32_e64 v93, s[14:15], s68, v86
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[14:15], 0, v83, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v92
	s_nop 1
	v_cndmask_b32_e64 v83, v87, v83, s[14:15]
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v85, v85, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v83, vcc
	v_cndmask_b32_e64 v83, v86, v93, s[14:15]
	v_cndmask_b32_e32 v82, v82, v83, vcc
	v_xor_b32_e32 v83, v70, v84
	v_xor_b32_e32 v70, v82, v84
	v_sub_co_u32_e32 v70, vcc, v70, v84
	s_mov_b64 s[14:15], 0x404
	s_nop 0
	v_subb_co_u32_e32 v83, vcc, v83, v84, vcc
	v_ashrrev_i32_e32 v82, 31, v83
	v_and_b32_e32 v82, s44, v82
	v_mul_lo_u32 v83, v107, s33
	v_lshl_add_u64 v[84:85], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xfbfb
	v_add3_u32 v82, v82, v70, v83
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v86, v84, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v83, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v87, v85, v83, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v87
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[92:93]
	v_xor_b32_e32 v93, v86, v92
	v_xor_b32_e32 v83, v87, v92
	v_mad_u64_u32 v[86:87], s[14:15], v93, s57, 0
	v_mul_hi_u32 v70, v93, s56
	v_lshl_add_u64 v[86:87], v[70:71], 0, v[86:87]
	v_mad_u64_u32 v[110:111], s[14:15], v83, s56, 0
	v_add_co_u32_e32 v70, vcc, v86, v110
	v_mad_u64_u32 v[108:109], s[14:15], v83, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v86, vcc, v87, v111, vcc
	v_mov_b32_e32 v87, s74
	s_nop 0
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[108:109]
	v_mul_lo_u32 v70, s69, v86
	v_mul_lo_u32 v87, s68, v87
	v_mad_u64_u32 v[108:109], s[14:15], s68, v86, 0
	v_add3_u32 v70, v109, v87, v70
	v_sub_co_u32_e32 v87, vcc, v93, v108
	s_nop 1
	v_subb_co_u32_e64 v93, s[14:15], v83, v70, vcc
	v_sub_u32_e32 v70, v83, v70
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	v_subrev_co_u32_e32 v83, vcc, s68, v87
	v_cmp_le_u32_e64 s[14:15], s69, v93
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v107, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v87
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v83
	v_cndmask_b32_e64 v108, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v93
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v93, v107, v108, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v87, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v83, s72, v92
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v86, v70
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v70, v86, v70, vcc
	v_ashrrev_i32_e32 v86, 31, v85
	v_mov_b32_e32 v87, v86
	v_xor_b32_e32 v70, v70, v83
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_sub_u32_e32 v70, v70, v83
	v_xor_b32_e32 v87, v84, v86
	v_xad_u32 v107, v70, v65, v37
	v_xor_b32_e32 v83, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v87, s57, 0
	v_mul_hi_u32 v70, v87, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[108:109], s[14:15], v83, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v108
	v_mad_u64_u32 v[92:93], s[14:15], v83, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v109, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v92, s68, v85
	v_mad_u64_u32 v[84:85], s[14:15], s68, v84, 0
	v_add3_u32 v70, v85, v92, v70
	v_sub_u32_e32 v85, v83, v70
	v_sub_co_u32_e32 v84, vcc, v87, v84
	s_nop 1
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, vcc
	v_subrev_co_u32_e64 v87, s[14:15], s68, v84
	v_subb_co_u32_e32 v70, vcc, v83, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v92, s[28:29], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v92
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v87
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v108, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v92
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v84
	v_cndmask_b32_e64 v93, v93, v108, s[28:29]
	v_subrev_co_u32_e64 v108, s[14:15], s68, v87
	s_nop 1
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v85, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v93
	s_nop 1
	v_cndmask_b32_e64 v85, v92, v85, s[14:15]
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	v_cndmask_b32_e64 v83, v87, v108, s[14:15]
	s_mov_b64 s[14:15], 0x405
	v_cndmask_b32_e32 v70, v70, v85, vcc
	v_cndmask_b32_e32 v83, v84, v83, vcc
	v_xor_b32_e32 v84, v70, v86
	v_xor_b32_e32 v70, v83, v86
	v_sub_co_u32_e32 v70, vcc, v70, v86
	s_nop 1
	v_subb_co_u32_e32 v83, vcc, v84, v86, vcc
	v_ashrrev_i32_e32 v83, 31, v83
	v_and_b32_e32 v83, s44, v83
	v_mul_lo_u32 v84, v107, s33
	v_add3_u32 v112, v83, v70, v84
	v_lshl_add_u64 v[84:85], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xfbfa
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v86, v84, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v83, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v87, v85, v83, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v87
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[92:93]
	v_xor_b32_e32 v93, v86, v92
	v_xor_b32_e32 v83, v87, v92
	v_mad_u64_u32 v[86:87], s[14:15], v93, s57, 0
	v_mul_hi_u32 v70, v93, s56
	v_lshl_add_u64 v[86:87], v[70:71], 0, v[86:87]
	v_mad_u64_u32 v[110:111], s[14:15], v83, s56, 0
	v_add_co_u32_e32 v70, vcc, v86, v110
	v_mad_u64_u32 v[108:109], s[14:15], v83, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v86, vcc, v87, v111, vcc
	v_mov_b32_e32 v87, s74
	s_nop 0
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[108:109]
	v_mul_lo_u32 v70, s69, v86
	v_mul_lo_u32 v87, s68, v87
	v_mad_u64_u32 v[108:109], s[14:15], s68, v86, 0
	v_add3_u32 v70, v109, v87, v70
	v_sub_co_u32_e32 v87, vcc, v93, v108
	s_nop 1
	v_subb_co_u32_e64 v93, s[14:15], v83, v70, vcc
	v_sub_u32_e32 v70, v83, v70
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	v_subrev_co_u32_e32 v83, vcc, s68, v87
	v_cmp_le_u32_e64 s[14:15], s69, v93
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v107, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v87
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v83
	v_cndmask_b32_e64 v108, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v93
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v93, v107, v108, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v87, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v83, s72, v92
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v86, v70
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v70, v86, v70, vcc
	v_ashrrev_i32_e32 v86, 31, v85
	v_mov_b32_e32 v87, v86
	v_xor_b32_e32 v70, v70, v83
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_sub_u32_e32 v70, v70, v83
	v_xor_b32_e32 v87, v84, v86
	v_xad_u32 v107, v70, v65, v37
	v_xor_b32_e32 v83, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v87, s57, 0
	v_mul_hi_u32 v70, v87, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[108:109], s[14:15], v83, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v108
	v_mad_u64_u32 v[92:93], s[14:15], v83, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v109, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v92, s68, v85
	v_mad_u64_u32 v[84:85], s[14:15], s68, v84, 0
	v_add3_u32 v70, v85, v92, v70
	v_sub_u32_e32 v85, v83, v70
	v_sub_co_u32_e32 v84, vcc, v87, v84
	s_nop 1
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, vcc
	v_subrev_co_u32_e64 v87, s[14:15], s68, v84
	v_subb_co_u32_e32 v70, vcc, v83, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v92, s[28:29], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v92
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v87
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v108, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v92
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v84
	v_cndmask_b32_e64 v93, v93, v108, s[28:29]
	v_subrev_co_u32_e64 v108, s[14:15], s68, v87
	s_nop 1
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v85, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v93
	s_nop 1
	v_cndmask_b32_e64 v85, v92, v85, s[14:15]
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	v_cndmask_b32_e64 v83, v87, v108, s[14:15]
	s_mov_b64 s[14:15], 0x406
	v_cndmask_b32_e32 v70, v70, v85, vcc
	v_cndmask_b32_e32 v83, v84, v83, vcc
	v_xor_b32_e32 v84, v70, v86
	v_xor_b32_e32 v70, v83, v86
	v_sub_co_u32_e32 v70, vcc, v70, v86
	s_nop 1
	v_subb_co_u32_e32 v83, vcc, v84, v86, vcc
	v_ashrrev_i32_e32 v83, 31, v83
	v_and_b32_e32 v83, s44, v83
	v_mul_lo_u32 v84, v107, s33
	v_add3_u32 v113, v83, v70, v84
	v_lshl_add_u64 v[84:85], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xfbf9
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v86, v84, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v83, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v87, v85, v83, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v87
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[92:93]
	v_xor_b32_e32 v93, v86, v92
	v_xor_b32_e32 v83, v87, v92
	v_mad_u64_u32 v[86:87], s[14:15], v93, s57, 0
	v_mul_hi_u32 v70, v93, s56
	v_lshl_add_u64 v[86:87], v[70:71], 0, v[86:87]
	v_mad_u64_u32 v[110:111], s[14:15], v83, s56, 0
	v_add_co_u32_e32 v70, vcc, v86, v110
	v_mad_u64_u32 v[108:109], s[14:15], v83, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v86, vcc, v87, v111, vcc
	v_mov_b32_e32 v87, s74
	s_nop 0
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[108:109]
	v_mul_lo_u32 v70, s69, v86
	v_mul_lo_u32 v87, s68, v87
	v_mad_u64_u32 v[108:109], s[14:15], s68, v86, 0
	v_add3_u32 v70, v109, v87, v70
	v_sub_co_u32_e32 v87, vcc, v93, v108
	s_nop 1
	v_subb_co_u32_e64 v93, s[14:15], v83, v70, vcc
	v_sub_u32_e32 v70, v83, v70
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	v_subrev_co_u32_e32 v83, vcc, s68, v87
	v_cmp_le_u32_e64 s[14:15], s69, v93
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v107, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v87
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v83
	v_cndmask_b32_e64 v108, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v93
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v93, v107, v108, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v87, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v83, s72, v92
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v86, v70
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v70, v86, v70, vcc
	v_ashrrev_i32_e32 v86, 31, v85
	v_mov_b32_e32 v87, v86
	v_xor_b32_e32 v70, v70, v83
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_sub_u32_e32 v70, v70, v83
	v_xor_b32_e32 v87, v84, v86
	v_xad_u32 v107, v70, v65, v37
	v_xor_b32_e32 v83, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v87, s57, 0
	v_mul_hi_u32 v70, v87, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[108:109], s[14:15], v83, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v108
	v_mad_u64_u32 v[92:93], s[14:15], v83, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v109, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v92, s68, v85
	v_mad_u64_u32 v[84:85], s[14:15], s68, v84, 0
	v_add3_u32 v70, v85, v92, v70
	v_sub_u32_e32 v85, v83, v70
	v_sub_co_u32_e32 v84, vcc, v87, v84
	s_nop 1
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, vcc
	v_subrev_co_u32_e64 v87, s[14:15], s68, v84
	v_subb_co_u32_e32 v70, vcc, v83, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v92, s[28:29], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v92
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v87
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v108, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v92
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v84
	v_cndmask_b32_e64 v93, v93, v108, s[28:29]
	v_subrev_co_u32_e64 v108, s[14:15], s68, v87
	s_nop 1
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v85, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v93
	s_nop 1
	v_cndmask_b32_e64 v85, v92, v85, s[14:15]
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	v_cndmask_b32_e64 v83, v87, v108, s[14:15]
	s_mov_b64 s[14:15], 0x407
	v_cndmask_b32_e32 v70, v70, v85, vcc
	v_cndmask_b32_e32 v83, v84, v83, vcc
	v_xor_b32_e32 v84, v70, v86
	v_xor_b32_e32 v70, v83, v86
	v_sub_co_u32_e32 v70, vcc, v70, v86
	s_nop 1
	v_subb_co_u32_e32 v83, vcc, v84, v86, vcc
	v_ashrrev_i32_e32 v83, 31, v83
	v_and_b32_e32 v83, s44, v83
	v_mul_lo_u32 v84, v107, s33
	v_add3_u32 v114, v83, v70, v84
	v_lshl_add_u64 v[84:85], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xfbf8
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v86, v84, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v83, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v87, v85, v83, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v87
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[92:93]
	v_xor_b32_e32 v93, v86, v92
	v_xor_b32_e32 v83, v87, v92
	v_mad_u64_u32 v[86:87], s[14:15], v93, s57, 0
	v_mul_hi_u32 v70, v93, s56
	v_lshl_add_u64 v[86:87], v[70:71], 0, v[86:87]
	v_mad_u64_u32 v[110:111], s[14:15], v83, s56, 0
	v_add_co_u32_e32 v70, vcc, v86, v110
	v_mad_u64_u32 v[108:109], s[14:15], v83, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v86, vcc, v87, v111, vcc
	v_mov_b32_e32 v87, s74
	s_nop 0
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[108:109]
	v_mul_lo_u32 v70, s69, v86
	v_mul_lo_u32 v87, s68, v87
	v_mad_u64_u32 v[108:109], s[14:15], s68, v86, 0
	v_add3_u32 v70, v109, v87, v70
	v_sub_co_u32_e32 v87, vcc, v93, v108
	s_nop 1
	v_subb_co_u32_e64 v93, s[14:15], v83, v70, vcc
	v_sub_u32_e32 v70, v83, v70
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	v_subrev_co_u32_e32 v83, vcc, s68, v87
	v_cmp_le_u32_e64 s[14:15], s69, v93
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v107, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v87
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v83
	v_cndmask_b32_e64 v108, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v93
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v93, v107, v108, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v87, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v83, s72, v92
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v86, v70
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v70, v86, v70, vcc
	v_ashrrev_i32_e32 v86, 31, v85
	v_mov_b32_e32 v87, v86
	v_xor_b32_e32 v70, v70, v83
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_sub_u32_e32 v70, v70, v83
	v_xor_b32_e32 v87, v84, v86
	v_xad_u32 v107, v70, v65, v37
	v_xor_b32_e32 v83, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v87, s57, 0
	v_mul_hi_u32 v70, v87, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[108:109], s[14:15], v83, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v108
	v_mad_u64_u32 v[92:93], s[14:15], v83, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v109, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v92, s68, v85
	v_mad_u64_u32 v[84:85], s[14:15], s68, v84, 0
	v_add3_u32 v70, v85, v92, v70
	v_sub_u32_e32 v85, v83, v70
	v_sub_co_u32_e32 v84, vcc, v87, v84
	s_nop 1
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, vcc
	v_subrev_co_u32_e64 v87, s[14:15], s68, v84
	v_subb_co_u32_e32 v70, vcc, v83, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v92, s[28:29], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v92
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v87
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v108, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v92
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v84
	v_cndmask_b32_e64 v93, v93, v108, s[28:29]
	v_subrev_co_u32_e64 v108, s[14:15], s68, v87
	s_nop 1
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v85, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v93
	s_nop 1
	v_cndmask_b32_e64 v85, v92, v85, s[14:15]
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	v_cndmask_b32_e64 v83, v87, v108, s[14:15]
	s_mov_b64 s[14:15], 0x408
	v_cndmask_b32_e32 v70, v70, v85, vcc
	v_cndmask_b32_e32 v83, v84, v83, vcc
	v_xor_b32_e32 v84, v70, v86
	v_xor_b32_e32 v70, v83, v86
	v_sub_co_u32_e32 v70, vcc, v70, v86
	s_nop 1
	v_subb_co_u32_e32 v83, vcc, v84, v86, vcc
	v_ashrrev_i32_e32 v83, 31, v83
	v_and_b32_e32 v83, s44, v83
	v_mul_lo_u32 v84, v107, s33
	v_add3_u32 v70, v83, v70, v84
	buffer_load_ubyte v107, v78, s[36:39], 0 offen
	buffer_load_ubyte v109, v79, s[36:39], 0 offen
	buffer_load_ubyte v108, v81, s[36:39], 0 offen
	buffer_load_ubyte v110, v82, s[36:39], 0 offen
	buffer_load_ubyte v111, v112, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v113, v113, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v112, v114, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v114, v70, s[36:39], 0 offen
	v_lshl_add_u64 v[78:79], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xfbf7
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v82, v78, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v81, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v83, v79, v81, s[22:23]
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v85, v82, v84
	v_xor_b32_e32 v81, v83, v84
	v_mad_u64_u32 v[82:83], s[14:15], v85, s57, 0
	v_mul_hi_u32 v70, v85, s56
	v_lshl_add_u64 v[82:83], v[70:71], 0, v[82:83]
	v_mad_u64_u32 v[92:93], s[14:15], v81, s56, 0
	v_add_co_u32_e32 v70, vcc, v82, v92
	v_mad_u64_u32 v[86:87], s[14:15], v81, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v93, vcc
	v_mov_b32_e32 v83, s74
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[86:87]
	v_mul_lo_u32 v70, s69, v82
	v_mul_lo_u32 v83, s68, v83
	v_mad_u64_u32 v[86:87], s[14:15], s68, v82, 0
	v_add3_u32 v70, v87, v83, v70
	v_sub_co_u32_e32 v83, vcc, v85, v86
	s_nop 1
	v_subb_co_u32_e64 v85, s[14:15], v81, v70, vcc
	v_sub_u32_e32 v70, v81, v70
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	v_subrev_co_u32_e32 v81, vcc, s68, v83
	v_cmp_le_u32_e64 s[14:15], s69, v85
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v86, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v83
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v81
	v_cndmask_b32_e64 v87, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v85
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v85, v86, v87, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v83, v81, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v81, s72, v84
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v82, v70
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v70, v82, v70, vcc
	v_ashrrev_i32_e32 v82, 31, v79
	v_mov_b32_e32 v83, v82
	v_xor_b32_e32 v70, v70, v81
	v_lshl_add_u64 v[78:79], v[78:79], 0, v[82:83]
	v_sub_u32_e32 v70, v70, v81
	v_xor_b32_e32 v92, v78, v82
	v_xad_u32 v81, v70, v65, v37
	v_xor_b32_e32 v83, v79, v82
	v_mad_u64_u32 v[78:79], s[14:15], v92, s57, 0
	v_mul_hi_u32 v70, v92, s56
	v_lshl_add_u64 v[78:79], v[70:71], 0, v[78:79]
	v_mad_u64_u32 v[86:87], s[14:15], v83, s56, 0
	v_add_co_u32_e32 v70, vcc, v78, v86
	v_mad_u64_u32 v[84:85], s[14:15], v83, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v78, vcc, v79, v87, vcc
	v_mov_b32_e32 v79, s74
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[78:79], v[78:79], 0, v[84:85]
	v_mul_lo_u32 v70, s69, v78
	v_mul_lo_u32 v84, s68, v79
	v_mad_u64_u32 v[78:79], s[14:15], s68, v78, 0
	v_add3_u32 v70, v79, v84, v70
	v_sub_u32_e32 v79, v83, v70
	v_sub_co_u32_e32 v78, vcc, v92, v78
	s_nop 1
	v_subb_co_u32_e64 v79, s[14:15], v79, v33, vcc
	v_subrev_co_u32_e64 v84, s[14:15], s68, v78
	v_subb_co_u32_e32 v70, vcc, v83, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[28:29], 0, v79, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v85
	v_subb_co_u32_e64 v79, s[14:15], v79, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v86, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v84
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v85
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v78
	v_cndmask_b32_e64 v86, v86, v87, s[28:29]
	v_subrev_co_u32_e64 v87, s[14:15], s68, v84
	s_nop 1
	v_subbrev_co_u32_e64 v79, s[14:15], 0, v79, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v86
	s_nop 1
	v_cndmask_b32_e64 v79, v85, v79, s[14:15]
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v79, vcc
	v_cndmask_b32_e64 v79, v84, v87, s[14:15]
	v_cndmask_b32_e32 v78, v78, v79, vcc
	v_xor_b32_e32 v79, v70, v82
	v_xor_b32_e32 v70, v78, v82
	v_sub_co_u32_e32 v70, vcc, v70, v82
	s_mov_b64 s[14:15], 0x409
	s_nop 0
	v_subb_co_u32_e32 v79, vcc, v79, v82, vcc
	v_ashrrev_i32_e32 v78, 31, v79
	v_and_b32_e32 v78, s44, v78
	v_mul_lo_u32 v79, v81, s33
	v_lshl_add_u64 v[82:83], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xfbf6
	v_add3_u32 v78, v78, v70, v79
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v84, v82, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v79, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v85, v83, v79, s[22:23]
	v_ashrrev_i32_e32 v86, 31, v85
	v_mov_b32_e32 v87, v86
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_xor_b32_e32 v81, v84, v86
	v_xor_b32_e32 v79, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v81, s57, 0
	v_mul_hi_u32 v70, v81, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[116:117], s[14:15], v79, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v116
	v_mad_u64_u32 v[92:93], s[14:15], v79, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v117, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v85, s68, v85
	v_mad_u64_u32 v[92:93], s[14:15], s68, v84, 0
	v_add3_u32 v70, v93, v85, v70
	v_sub_co_u32_e32 v81, vcc, v81, v92
	s_nop 1
	v_subb_co_u32_e64 v85, s[14:15], v79, v70, vcc
	v_sub_u32_e32 v70, v79, v70
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	v_subrev_co_u32_e32 v79, vcc, s68, v81
	v_cmp_le_u32_e64 s[14:15], s69, v85
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v87, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v81
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v79
	v_cndmask_b32_e64 v92, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v85
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v85, v87, v92, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v81, v79, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v79, s72, v86
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v84, v70
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v70, v84, v70, vcc
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_xor_b32_e32 v70, v70, v79
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_sub_u32_e32 v70, v70, v79
	v_xor_b32_e32 v85, v82, v84
	v_xad_u32 v81, v70, v65, v37
	v_xor_b32_e32 v79, v83, v84
	v_mad_u64_u32 v[82:83], s[14:15], v85, s57, 0
	v_mul_hi_u32 v70, v85, s56
	v_lshl_add_u64 v[82:83], v[70:71], 0, v[82:83]
	v_mad_u64_u32 v[92:93], s[14:15], v79, s56, 0
	v_add_co_u32_e32 v70, vcc, v82, v92
	v_mad_u64_u32 v[86:87], s[14:15], v79, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v93, vcc
	v_mov_b32_e32 v83, s74
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[86:87]
	v_mul_lo_u32 v70, s69, v82
	v_mul_lo_u32 v86, s68, v83
	v_mad_u64_u32 v[82:83], s[14:15], s68, v82, 0
	v_add3_u32 v70, v83, v86, v70
	v_sub_u32_e32 v83, v79, v70
	v_sub_co_u32_e32 v82, vcc, v85, v82
	v_mul_lo_u32 v81, v81, s33
	s_nop 0
	v_subb_co_u32_e64 v83, s[14:15], v83, v33, vcc
	v_subrev_co_u32_e64 v85, s[14:15], s68, v82
	v_subb_co_u32_e32 v70, vcc, v79, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v86, s[28:29], 0, v83, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v86
	v_subb_co_u32_e64 v83, s[14:15], v83, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v85
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v86
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v82
	v_cndmask_b32_e64 v87, v87, v92, s[28:29]
	v_subrev_co_u32_e64 v92, s[14:15], s68, v85
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[14:15], 0, v83, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v87
	s_nop 1
	v_cndmask_b32_e64 v83, v86, v83, s[14:15]
	v_cndmask_b32_e64 v86, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v79, v79, v86, vcc
	v_cmp_ne_u32_e32 vcc, 0, v79
	v_cndmask_b32_e64 v79, v85, v92, s[14:15]
	s_mov_b64 s[14:15], 0x40a
	v_cndmask_b32_e32 v70, v70, v83, vcc
	v_cndmask_b32_e32 v79, v82, v79, vcc
	v_xor_b32_e32 v82, v70, v84
	v_xor_b32_e32 v70, v79, v84
	v_sub_co_u32_e32 v70, vcc, v70, v84
	s_nop 1
	v_subb_co_u32_e32 v79, vcc, v82, v84, vcc
	v_ashrrev_i32_e32 v79, 31, v79
	v_and_b32_e32 v79, s44, v79
	v_lshl_add_u64 v[82:83], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xfbf5
	v_add3_u32 v79, v79, v70, v81
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v84, v82, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v81, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v85, v83, v81, s[22:23]
	v_ashrrev_i32_e32 v86, 31, v85
	v_mov_b32_e32 v87, v86
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_xor_b32_e32 v87, v84, v86
	v_xor_b32_e32 v81, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v87, s57, 0
	v_mul_hi_u32 v70, v87, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[116:117], s[14:15], v81, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v116
	v_mad_u64_u32 v[92:93], s[14:15], v81, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v117, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v85, s68, v85
	v_mad_u64_u32 v[92:93], s[14:15], s68, v84, 0
	v_add3_u32 v70, v93, v85, v70
	v_sub_co_u32_e32 v85, vcc, v87, v92
	s_nop 1
	v_subb_co_u32_e64 v87, s[14:15], v81, v70, vcc
	v_sub_u32_e32 v70, v81, v70
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	v_subrev_co_u32_e32 v81, vcc, s68, v85
	v_cmp_le_u32_e64 s[14:15], s69, v87
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v92, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v85
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v81
	v_cndmask_b32_e64 v93, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v87
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v87, v92, v93, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v85, v81, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v81, s72, v86
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v84, v70
	v_cmp_ne_u32_e32 vcc, 0, v87
	s_nop 1
	v_cndmask_b32_e32 v70, v84, v70, vcc
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_xor_b32_e32 v70, v70, v81
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_sub_u32_e32 v70, v70, v81
	v_xor_b32_e32 v85, v82, v84
	v_xad_u32 v115, v70, v65, v37
	v_xor_b32_e32 v81, v83, v84
	v_mad_u64_u32 v[82:83], s[14:15], v85, s57, 0
	v_mul_hi_u32 v70, v85, s56
	v_lshl_add_u64 v[82:83], v[70:71], 0, v[82:83]
	v_mad_u64_u32 v[92:93], s[14:15], v81, s56, 0
	v_add_co_u32_e32 v70, vcc, v82, v92
	v_mad_u64_u32 v[86:87], s[14:15], v81, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v93, vcc
	v_mov_b32_e32 v83, s74
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[86:87]
	v_mul_lo_u32 v70, s69, v82
	v_mul_lo_u32 v86, s68, v83
	v_mad_u64_u32 v[82:83], s[14:15], s68, v82, 0
	v_add3_u32 v70, v83, v86, v70
	v_sub_u32_e32 v83, v81, v70
	v_sub_co_u32_e32 v82, vcc, v85, v82
	s_nop 1
	v_subb_co_u32_e64 v83, s[14:15], v83, v33, vcc
	v_subrev_co_u32_e64 v85, s[14:15], s68, v82
	v_subb_co_u32_e32 v70, vcc, v81, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v86, s[28:29], 0, v83, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v86
	v_subb_co_u32_e64 v83, s[14:15], v83, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v85
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v86
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v82
	v_cndmask_b32_e64 v87, v87, v92, s[28:29]
	v_subrev_co_u32_e64 v92, s[14:15], s68, v85
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[14:15], 0, v83, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v87
	s_nop 1
	v_cndmask_b32_e64 v83, v86, v83, s[14:15]
	v_cndmask_b32_e64 v86, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v81, v81, v86, vcc
	v_cmp_ne_u32_e32 vcc, 0, v81
	v_cndmask_b32_e64 v81, v85, v92, s[14:15]
	s_mov_b64 s[14:15], 0x40b
	v_cndmask_b32_e32 v70, v70, v83, vcc
	v_cndmask_b32_e32 v81, v82, v81, vcc
	v_xor_b32_e32 v82, v70, v84
	v_xor_b32_e32 v70, v81, v84
	v_sub_co_u32_e32 v70, vcc, v70, v84
	s_nop 1
	v_subb_co_u32_e32 v81, vcc, v82, v84, vcc
	v_ashrrev_i32_e32 v81, 31, v81
	v_and_b32_e32 v81, s44, v81
	v_mul_lo_u32 v82, v115, s33
	v_add3_u32 v81, v81, v70, v82
	v_lshl_add_u64 v[82:83], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xfbf4
	v_sub_co_u32_e32 v70, vcc, s14, v88
	s_nop 1
	v_subb_co_u32_e32 v84, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v85, v83, v84, s[22:23]
	v_ashrrev_i32_e32 v86, 31, v85
	v_cndmask_b32_e64 v84, v82, v70, s[22:23]
	v_mov_b32_e32 v87, v86
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_xor_b32_e32 v115, v84, v86
	v_xor_b32_e32 v87, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v115, s57, 0
	v_mul_hi_u32 v70, v115, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[116:117], s[14:15], v87, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v116
	v_mad_u64_u32 v[92:93], s[14:15], v87, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v117, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v85, s68, v85
	v_mad_u64_u32 v[92:93], s[14:15], s68, v84, 0
	v_add3_u32 v70, v93, v85, v70
	v_sub_co_u32_e32 v85, vcc, v115, v92
	s_nop 1
	v_subb_co_u32_e64 v92, s[14:15], v87, v70, vcc
	v_sub_u32_e32 v70, v87, v70
	v_cmp_le_u32_e64 s[14:15], s69, v92
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v85
	v_subrev_co_u32_e32 v85, vcc, s68, v85
	s_nop 0
	v_cndmask_b32_e64 v115, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cmp_eq_u32_e64 s[14:15], s69, v92
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v85
	v_cndmask_b32_e64 v92, v93, v115, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v70, v87, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_nop 1
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v84, v70
	v_cmp_ne_u32_e32 vcc, 0, v92
	s_nop 1
	v_cndmask_b32_e32 v70, v84, v70, vcc
	v_xor_b32_e32 v84, s72, v86
	v_xor_b32_e32 v70, v70, v84
	v_sub_u32_e32 v70, v70, v84
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v116, v82, v84
	v_xad_u32 v115, v70, v65, v37
	v_xor_b32_e32 v85, v83, v84
	v_mad_u64_u32 v[82:83], s[14:15], v116, s57, 0
	v_mul_hi_u32 v70, v116, s56
	v_lshl_add_u64 v[82:83], v[70:71], 0, v[82:83]
	v_mad_u64_u32 v[92:93], s[14:15], v85, s56, 0
	v_add_co_u32_e32 v70, vcc, v82, v92
	v_mad_u64_u32 v[86:87], s[14:15], v85, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v93, vcc
	v_mov_b32_e32 v83, s74
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[86:87]
	v_mul_lo_u32 v70, s69, v82
	v_mul_lo_u32 v86, s68, v83
	v_mad_u64_u32 v[82:83], s[14:15], s68, v82, 0
	v_add3_u32 v70, v83, v86, v70
	v_sub_u32_e32 v83, v85, v70
	v_sub_co_u32_e32 v82, vcc, v116, v82
	s_nop 1
	v_subb_co_u32_e64 v83, s[14:15], v83, v33, vcc
	v_subrev_co_u32_e64 v86, s[14:15], s68, v82
	v_subb_co_u32_e32 v70, vcc, v85, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v87, s[28:29], 0, v83, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v87
	v_subb_co_u32_e64 v83, s[14:15], v83, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v86
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v87
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v82
	v_cndmask_b32_e64 v92, v92, v93, s[28:29]
	v_subrev_co_u32_e64 v93, s[14:15], s68, v86
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[14:15], 0, v83, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v92
	s_nop 1
	v_cndmask_b32_e64 v83, v87, v83, s[14:15]
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v85, v85, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v83, vcc
	v_cndmask_b32_e64 v83, v86, v93, s[14:15]
	v_cndmask_b32_e32 v82, v82, v83, vcc
	v_xor_b32_e32 v83, v70, v84
	v_xor_b32_e32 v70, v82, v84
	v_sub_co_u32_e32 v70, vcc, v70, v84
	s_mov_b64 s[14:15], 0x40c
	s_nop 0
	v_subb_co_u32_e32 v83, vcc, v83, v84, vcc
	v_ashrrev_i32_e32 v82, 31, v83
	v_and_b32_e32 v82, s44, v82
	v_mul_lo_u32 v83, v115, s33
	v_lshl_add_u64 v[84:85], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xfbf3
	v_add3_u32 v82, v82, v70, v83
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v86, v84, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v83, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v87, v85, v83, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v87
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[92:93]
	v_xor_b32_e32 v93, v86, v92
	v_xor_b32_e32 v83, v87, v92
	v_mad_u64_u32 v[86:87], s[14:15], v93, s57, 0
	v_mul_hi_u32 v70, v93, s56
	v_lshl_add_u64 v[86:87], v[70:71], 0, v[86:87]
	v_mad_u64_u32 v[118:119], s[14:15], v83, s56, 0
	v_add_co_u32_e32 v70, vcc, v86, v118
	v_mad_u64_u32 v[116:117], s[14:15], v83, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v86, vcc, v87, v119, vcc
	v_mov_b32_e32 v87, s74
	s_nop 0
	v_addc_co_u32_e32 v117, vcc, 0, v117, vcc
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[116:117]
	v_mul_lo_u32 v70, s69, v86
	v_mul_lo_u32 v87, s68, v87
	v_mad_u64_u32 v[116:117], s[14:15], s68, v86, 0
	v_add3_u32 v70, v117, v87, v70
	v_sub_co_u32_e32 v87, vcc, v93, v116
	s_nop 1
	v_subb_co_u32_e64 v93, s[14:15], v83, v70, vcc
	v_sub_u32_e32 v70, v83, v70
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	v_subrev_co_u32_e32 v83, vcc, s68, v87
	v_cmp_le_u32_e64 s[14:15], s69, v93
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v115, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v87
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v83
	v_cndmask_b32_e64 v116, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v93
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v93, v115, v116, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v87, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v83, s72, v92
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v86, v70
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v70, v86, v70, vcc
	v_ashrrev_i32_e32 v86, 31, v85
	v_mov_b32_e32 v87, v86
	v_xor_b32_e32 v70, v70, v83
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_sub_u32_e32 v70, v70, v83
	v_xor_b32_e32 v87, v84, v86
	v_xad_u32 v115, v70, v65, v37
	v_xor_b32_e32 v83, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v87, s57, 0
	v_mul_hi_u32 v70, v87, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[116:117], s[14:15], v83, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v116
	v_mad_u64_u32 v[92:93], s[14:15], v83, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v117, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v92, s68, v85
	v_mad_u64_u32 v[84:85], s[14:15], s68, v84, 0
	v_add3_u32 v70, v85, v92, v70
	v_sub_u32_e32 v85, v83, v70
	v_sub_co_u32_e32 v84, vcc, v87, v84
	s_nop 1
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, vcc
	v_subrev_co_u32_e64 v87, s[14:15], s68, v84
	v_subb_co_u32_e32 v70, vcc, v83, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v92, s[28:29], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v92
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v87
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v116, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v92
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v84
	v_cndmask_b32_e64 v93, v93, v116, s[28:29]
	v_subrev_co_u32_e64 v116, s[14:15], s68, v87
	s_nop 1
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v85, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v93
	s_nop 1
	v_cndmask_b32_e64 v85, v92, v85, s[14:15]
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	v_cndmask_b32_e64 v83, v87, v116, s[14:15]
	s_mov_b64 s[14:15], 0x40d
	v_cndmask_b32_e32 v70, v70, v85, vcc
	v_cndmask_b32_e32 v83, v84, v83, vcc
	v_xor_b32_e32 v84, v70, v86
	v_xor_b32_e32 v70, v83, v86
	v_sub_co_u32_e32 v70, vcc, v70, v86
	s_nop 1
	v_subb_co_u32_e32 v83, vcc, v84, v86, vcc
	v_ashrrev_i32_e32 v83, 31, v83
	v_and_b32_e32 v83, s44, v83
	v_mul_lo_u32 v84, v115, s33
	v_add3_u32 v120, v83, v70, v84
	v_lshl_add_u64 v[84:85], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xfbf2
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v86, v84, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v83, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v87, v85, v83, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v87
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[92:93]
	v_xor_b32_e32 v93, v86, v92
	v_xor_b32_e32 v83, v87, v92
	v_mad_u64_u32 v[86:87], s[14:15], v93, s57, 0
	v_mul_hi_u32 v70, v93, s56
	v_lshl_add_u64 v[86:87], v[70:71], 0, v[86:87]
	v_mad_u64_u32 v[118:119], s[14:15], v83, s56, 0
	v_add_co_u32_e32 v70, vcc, v86, v118
	v_mad_u64_u32 v[116:117], s[14:15], v83, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v86, vcc, v87, v119, vcc
	v_mov_b32_e32 v87, s74
	s_nop 0
	v_addc_co_u32_e32 v117, vcc, 0, v117, vcc
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[116:117]
	v_mul_lo_u32 v70, s69, v86
	v_mul_lo_u32 v87, s68, v87
	v_mad_u64_u32 v[116:117], s[14:15], s68, v86, 0
	v_add3_u32 v70, v117, v87, v70
	v_sub_co_u32_e32 v87, vcc, v93, v116
	s_nop 1
	v_subb_co_u32_e64 v93, s[14:15], v83, v70, vcc
	v_sub_u32_e32 v70, v83, v70
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	v_subrev_co_u32_e32 v83, vcc, s68, v87
	v_cmp_le_u32_e64 s[14:15], s69, v93
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v115, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v87
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v83
	v_cndmask_b32_e64 v116, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v93
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v93, v115, v116, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v87, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v83, s72, v92
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v86, v70
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v70, v86, v70, vcc
	v_ashrrev_i32_e32 v86, 31, v85
	v_mov_b32_e32 v87, v86
	v_xor_b32_e32 v70, v70, v83
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_sub_u32_e32 v70, v70, v83
	v_xor_b32_e32 v87, v84, v86
	v_xad_u32 v115, v70, v65, v37
	v_xor_b32_e32 v83, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v87, s57, 0
	v_mul_hi_u32 v70, v87, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[116:117], s[14:15], v83, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v116
	v_mad_u64_u32 v[92:93], s[14:15], v83, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v117, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v92, s68, v85
	v_mad_u64_u32 v[84:85], s[14:15], s68, v84, 0
	v_add3_u32 v70, v85, v92, v70
	v_sub_u32_e32 v85, v83, v70
	v_sub_co_u32_e32 v84, vcc, v87, v84
	s_nop 1
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, vcc
	v_subrev_co_u32_e64 v87, s[14:15], s68, v84
	v_subb_co_u32_e32 v70, vcc, v83, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v92, s[28:29], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v92
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v87
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v116, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v92
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v84
	v_cndmask_b32_e64 v93, v93, v116, s[28:29]
	v_subrev_co_u32_e64 v116, s[14:15], s68, v87
	s_nop 1
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v85, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v93
	s_nop 1
	v_cndmask_b32_e64 v85, v92, v85, s[14:15]
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	v_cndmask_b32_e64 v83, v87, v116, s[14:15]
	s_mov_b64 s[14:15], 0x40e
	v_cndmask_b32_e32 v70, v70, v85, vcc
	v_cndmask_b32_e32 v83, v84, v83, vcc
	v_xor_b32_e32 v84, v70, v86
	v_xor_b32_e32 v70, v83, v86
	v_sub_co_u32_e32 v70, vcc, v70, v86
	s_nop 1
	v_subb_co_u32_e32 v83, vcc, v84, v86, vcc
	v_ashrrev_i32_e32 v83, 31, v83
	v_and_b32_e32 v83, s44, v83
	v_mul_lo_u32 v84, v115, s33
	v_add3_u32 v121, v83, v70, v84
	v_lshl_add_u64 v[84:85], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xfbf1
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v86, v84, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v83, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v87, v85, v83, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v87
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[92:93]
	v_xor_b32_e32 v93, v86, v92
	v_xor_b32_e32 v83, v87, v92
	v_mad_u64_u32 v[86:87], s[14:15], v93, s57, 0
	v_mul_hi_u32 v70, v93, s56
	v_lshl_add_u64 v[86:87], v[70:71], 0, v[86:87]
	v_mad_u64_u32 v[118:119], s[14:15], v83, s56, 0
	v_add_co_u32_e32 v70, vcc, v86, v118
	v_mad_u64_u32 v[116:117], s[14:15], v83, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v86, vcc, v87, v119, vcc
	v_mov_b32_e32 v87, s74
	s_nop 0
	v_addc_co_u32_e32 v117, vcc, 0, v117, vcc
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[116:117]
	v_mul_lo_u32 v70, s69, v86
	v_mul_lo_u32 v87, s68, v87
	v_mad_u64_u32 v[116:117], s[14:15], s68, v86, 0
	v_add3_u32 v70, v117, v87, v70
	v_sub_co_u32_e32 v87, vcc, v93, v116
	s_nop 1
	v_subb_co_u32_e64 v93, s[14:15], v83, v70, vcc
	v_sub_u32_e32 v70, v83, v70
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	v_subrev_co_u32_e32 v83, vcc, s68, v87
	v_cmp_le_u32_e64 s[14:15], s69, v93
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v115, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v87
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v83
	v_cndmask_b32_e64 v116, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v93
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v93, v115, v116, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v87, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v83, s72, v92
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v86, v70
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v70, v86, v70, vcc
	v_ashrrev_i32_e32 v86, 31, v85
	v_mov_b32_e32 v87, v86
	v_xor_b32_e32 v70, v70, v83
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_sub_u32_e32 v70, v70, v83
	v_xor_b32_e32 v87, v84, v86
	v_xad_u32 v115, v70, v65, v37
	v_xor_b32_e32 v83, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v87, s57, 0
	v_mul_hi_u32 v70, v87, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[116:117], s[14:15], v83, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v116
	v_mad_u64_u32 v[92:93], s[14:15], v83, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v117, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v92, s68, v85
	v_mad_u64_u32 v[84:85], s[14:15], s68, v84, 0
	v_add3_u32 v70, v85, v92, v70
	v_sub_u32_e32 v85, v83, v70
	v_sub_co_u32_e32 v84, vcc, v87, v84
	s_nop 1
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, vcc
	v_subrev_co_u32_e64 v87, s[14:15], s68, v84
	v_subb_co_u32_e32 v70, vcc, v83, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v92, s[28:29], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v92
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v87
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v116, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v92
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v84
	v_cndmask_b32_e64 v93, v93, v116, s[28:29]
	v_subrev_co_u32_e64 v116, s[14:15], s68, v87
	s_lshl_b64 s[28:29], s[86:87], 8
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v85, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v93
	s_nop 1
	v_cndmask_b32_e64 v85, v92, v85, s[14:15]
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	v_cndmask_b32_e64 v83, v87, v116, s[14:15]
	s_mov_b64 s[14:15], 0x40f
	v_cndmask_b32_e32 v70, v70, v85, vcc
	v_cndmask_b32_e32 v83, v84, v83, vcc
	v_xor_b32_e32 v84, v70, v86
	v_xor_b32_e32 v70, v83, v86
	v_sub_co_u32_e32 v70, vcc, v70, v86
	s_nop 1
	v_subb_co_u32_e32 v83, vcc, v84, v86, vcc
	v_ashrrev_i32_e32 v83, 31, v83
	v_and_b32_e32 v83, s44, v83
	v_mul_lo_u32 v84, v115, s33
	v_add3_u32 v83, v83, v70, v84
	v_lshl_add_u64 v[84:85], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xfbf0
	v_sub_co_u32_e32 v70, vcc, s14, v88
	s_nop 1
	v_subb_co_u32_e32 v86, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v87, v85, v86, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v87
	v_cndmask_b32_e64 v86, v84, v70, s[22:23]
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[92:93]
	v_xor_b32_e32 v115, v86, v92
	v_xor_b32_e32 v93, v87, v92
	v_mad_u64_u32 v[86:87], s[14:15], v115, s57, 0
	v_mul_hi_u32 v70, v115, s56
	v_lshl_add_u64 v[86:87], v[70:71], 0, v[86:87]
	v_mad_u64_u32 v[118:119], s[14:15], v93, s56, 0
	v_add_co_u32_e32 v70, vcc, v86, v118
	v_mad_u64_u32 v[116:117], s[14:15], v93, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v86, vcc, v87, v119, vcc
	v_mov_b32_e32 v87, s74
	s_nop 0
	v_addc_co_u32_e32 v117, vcc, 0, v117, vcc
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[116:117]
	v_mul_lo_u32 v70, s69, v86
	v_mul_lo_u32 v87, s68, v87
	v_mad_u64_u32 v[116:117], s[14:15], s68, v86, 0
	v_add3_u32 v70, v117, v87, v70
	v_sub_co_u32_e32 v87, vcc, v115, v116
	s_nop 1
	v_subb_co_u32_e64 v115, s[14:15], v93, v70, vcc
	v_sub_u32_e32 v70, v93, v70
	v_cmp_le_u32_e64 s[14:15], s69, v115
	v_subb_co_u32_e32 v70, vcc, v70, v33, vcc
	s_nop 0
	v_cndmask_b32_e64 v116, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v87
	v_subrev_co_u32_e32 v87, vcc, s68, v87
	s_nop 0
	v_cndmask_b32_e64 v117, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cmp_eq_u32_e64 s[14:15], s69, v115
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v87
	v_cndmask_b32_e64 v115, v116, v117, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v70, v93, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_nop 1
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v86, v70
	v_cmp_ne_u32_e32 vcc, 0, v115
	s_nop 1
	v_cndmask_b32_e32 v70, v86, v70, vcc
	v_xor_b32_e32 v86, s72, v92
	v_xor_b32_e32 v70, v70, v86
	v_sub_u32_e32 v70, v70, v86
	v_ashrrev_i32_e32 v86, 31, v85
	v_mov_b32_e32 v87, v86
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_xor_b32_e32 v115, v84, v86
	v_xad_u32 v65, v70, v65, v37
	v_xor_b32_e32 v87, v85, v86
	v_mad_u64_u32 v[84:85], s[14:15], v115, s57, 0
	v_mul_hi_u32 v70, v115, s56
	v_lshl_add_u64 v[84:85], v[70:71], 0, v[84:85]
	v_mad_u64_u32 v[116:117], s[14:15], v87, s56, 0
	v_add_co_u32_e32 v70, vcc, v84, v116
	v_mad_u64_u32 v[92:93], s[14:15], v87, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v117, vcc
	v_mov_b32_e32 v85, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v70, s69, v84
	v_mul_lo_u32 v92, s68, v85
	v_mad_u64_u32 v[84:85], s[14:15], s68, v84, 0
	v_add3_u32 v70, v85, v92, v70
	v_sub_u32_e32 v85, v87, v70
	v_sub_co_u32_e32 v84, vcc, v115, v84
	v_mul_lo_u32 v65, v65, s33
	s_nop 0
	v_subb_co_u32_e64 v85, s[14:15], v85, v33, vcc
	v_subrev_co_u32_e64 v92, s[14:15], s68, v84
	v_subb_co_u32_e32 v70, vcc, v87, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v93, s[22:23], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s69, v93
	v_subb_co_u32_e64 v33, s[14:15], v85, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v115, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s68, v92
	v_subrev_co_u32_e64 v85, s[14:15], s68, v92
	s_nop 0
	v_cndmask_b32_e64 v116, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s69, v93
	v_subbrev_co_u32_e64 v33, s[14:15], 0, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v115, v115, v116, s[22:23]
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cmp_ne_u32_e64 s[14:15], 0, v115
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v84
	v_cndmask_b32_e64 v33, v93, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v87, v87, v93, vcc
	v_cmp_ne_u32_e32 vcc, 0, v87
	s_nop 1
	v_cndmask_b32_e32 v33, v70, v33, vcc
	v_cndmask_b32_e64 v70, v92, v85, s[14:15]
	v_cndmask_b32_e32 v70, v84, v70, vcc
	v_xor_b32_e32 v70, v70, v86
	v_xor_b32_e32 v33, v33, v86
	v_sub_co_u32_e32 v70, vcc, v70, v86
	s_nop 1
	v_subb_co_u32_e32 v33, vcc, v33, v86, vcc
	v_ashrrev_i32_e32 v33, 31, v33
	v_and_b32_e32 v33, s44, v33
	v_add3_u32 v33, v33, v70, v65
	buffer_load_ubyte v115, v78, s[36:39], 0 offen
	buffer_load_ubyte v117, v79, s[36:39], 0 offen
	buffer_load_ubyte v116, v81, s[36:39], 0 offen
	buffer_load_ubyte v118, v82, s[36:39], 0 offen
	buffer_load_ubyte v119, v120, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v121, v121, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v120, v83, s[36:39], 0 offen
	buffer_load_ubyte v122, v33, s[36:39], 0 offen
	ds_read_b128 v[132:135], v52 offset:4096
	ds_read_b128 v[128:131], v54 offset:6144
	ds_read_b128 v[124:127], v52 offset:6144
	ds_read_b128 v[136:139], v54 offset:4096
	ds_read_b32 v33, v50 offset:49408
	v_lshl_add_u64 v[80:81], v[76:77], 0, s[28:29]
	v_lshl_add_u64 v[78:79], v[72:73], 0, v[80:81]
	v_sub_co_u32_e32 v65, vcc, 0xfffffeff, v78
	v_lshl_add_u64 v[82:83], v[78:79], 0, s[94:95]
	s_nop 0
	v_subb_co_u32_e32 v70, vcc, -1, v79, vcc
	v_cmp_gt_i32_e32 vcc, 0, v83
	s_nop 1
	v_cndmask_b32_e32 v87, v83, v70, vcc
	v_cndmask_b32_e32 v86, v82, v65, vcc
	v_or_b32_e32 v65, s79, v87
	v_cmp_ne_u32_e32 vcc, 0, v65
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[22:23], exec, s[14:15]
	s_cbranch_execz .LBB0_42
	s_add_u32 s14, s78, s82
	s_mov_b32 s83, s82
	s_addc_u32 s15, s79, s82
	s_xor_b64 s[56:57], s[14:15], s[82:83]
	v_cvt_f32_u32_e32 v65, s56
	v_cvt_f32_u32_e32 v70, s57
	s_sub_u32 s14, 0, s56
	s_subb_u32 s15, 0, s57
	v_ashrrev_i32_e32 v84, 31, v87
	v_fmac_f32_e32 v65, 0x4f800000, v70
	v_rcp_f32_e32 v65, v65
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[84:85]
	v_xor_b32_e32 v82, v86, v84
	v_mul_f32_e32 v65, 0x5f7ffffc, v65
	v_mul_f32_e32 v70, 0x2f800000, v65
	v_trunc_f32_e32 v70, v70
	v_fmac_f32_e32 v65, 0xcf800000, v70
	v_cvt_u32_f32_e32 v70, v70
	v_cvt_u32_f32_e32 v65, v65
	v_readfirstlane_b32 s68, v70
	v_readfirstlane_b32 s69, v65
	s_mul_i32 s73, s14, s68
	s_mul_hi_u32 s81, s14, s69
	s_mul_i32 s75, s15, s69
	s_add_i32 s73, s81, s73
	s_add_i32 s73, s73, s75
	s_mul_i32 s83, s14, s69
	s_mul_i32 s81, s69, s73
	s_mul_hi_u32 vcc_lo, s69, s83
	s_mul_hi_u32 s75, s69, s73
	s_add_u32 s81, vcc_lo, s81
	s_addc_u32 s75, 0, s75
	s_mul_hi_u32 vcc_hi, s68, s83
	s_mul_i32 s83, s68, s83
	s_add_u32 s81, s81, s83
	s_mul_hi_u32 vcc_lo, s68, s73
	s_addc_u32 s75, s75, vcc_hi
	s_addc_u32 s81, vcc_lo, 0
	s_mul_i32 s73, s68, s73
	s_add_u32 s73, s75, s73
	s_addc_u32 s75, 0, s81
	s_add_u32 s69, s69, s73
	s_addc_u32 s68, s68, s75
	s_mul_i32 s73, s14, s68
	s_mul_hi_u32 s75, s14, s69
	s_add_i32 s73, s75, s73
	s_mul_i32 s15, s15, s69
	s_add_i32 s73, s73, s15
	s_mul_i32 s14, s14, s69
	s_mul_hi_u32 s75, s68, s14
	s_mul_i32 s81, s68, s14
	s_mul_i32 vcc_lo, s69, s73
	s_mul_hi_u32 s14, s69, s14
	s_mul_hi_u32 s83, s69, s73
	s_add_u32 s14, s14, vcc_lo
	s_addc_u32 s83, 0, s83
	s_add_u32 s14, s14, s81
	s_mul_hi_u32 s15, s68, s73
	s_addc_u32 s14, s83, s75
	s_addc_u32 s15, s15, 0
	s_mul_i32 s73, s68, s73
	s_add_u32 s14, s14, s73
	s_addc_u32 s15, 0, s15
	s_add_u32 s69, s69, s14
	s_addc_u32 s68, s68, s15
	v_xor_b32_e32 v65, v87, v84
	v_mad_u64_u32 v[86:87], s[14:15], v82, s68, 0
	v_mul_hi_u32 v70, v82, s69
	v_lshl_add_u64 v[86:87], v[70:71], 0, v[86:87]
	v_mad_u64_u32 v[140:141], s[14:15], v65, s69, 0
	v_add_co_u32_e32 v70, vcc, v86, v140
	v_mad_u64_u32 v[92:93], s[14:15], v65, s68, 0
	s_nop 0
	v_addc_co_u32_e32 v86, vcc, v87, v141, vcc
	v_mov_b32_e32 v87, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[92:93]
	v_mul_lo_u32 v70, s57, v86
	v_mul_lo_u32 v85, s56, v87
	v_mad_u64_u32 v[92:93], s[14:15], s56, v86, 0
	v_add3_u32 v70, v93, v85, v70
	v_sub_u32_e32 v85, v65, v70
	v_mov_b32_e32 v93, s57
	v_sub_co_u32_e32 v82, vcc, v82, v92
	v_lshl_add_u64 v[140:141], v[86:87], 0, 1
	s_nop 0
	v_subb_co_u32_e64 v85, s[14:15], v85, v93, vcc
	v_subrev_co_u32_e64 v92, s[14:15], s56, v82
	v_subb_co_u32_e32 v65, vcc, v65, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s57, v85
	v_cmp_le_u32_e32 vcc, s57, v65
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s56, v92
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s56, v82
	v_cndmask_b32_e64 v92, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s57, v85
	v_cndmask_b32_e64 v82, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s57, v65
	v_cndmask_b32_e64 v85, v93, v92, s[14:15]
	v_lshl_add_u64 v[92:93], v[86:87], 0, 2
	v_cndmask_b32_e32 v65, v70, v82, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v70, v140, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v65
	s_nop 1
	v_cndmask_b32_e32 v65, v86, v70, vcc
	v_xor_b32_e32 v70, s82, v84
	v_xor_b32_e32 v65, v65, v70
	v_sub_co_u32_e32 v84, vcc, v65, v70
.LBB0_42:
	s_andn2_saveexec_b64 s[14:15], s[22:23]
	s_cbranch_execz .LBB0_44
	v_rcp_iflag_f32_e32 v65, v100
	s_sub_i32 s22, 0, s78
	v_mul_f32_e32 v65, 0x4f7ffffe, v65
	v_cvt_u32_f32_e32 v65, v65
	v_mul_lo_u32 v70, s22, v65
	v_mul_hi_u32 v70, v65, v70
	v_add_u32_e32 v65, v65, v70
	v_mul_hi_u32 v65, v86, v65
	v_mul_lo_u32 v70, v65, s78
	v_sub_u32_e32 v70, v86, v70
	v_add_u32_e32 v82, 1, v65
	v_subrev_u32_e32 v84, s78, v70
	v_cmp_le_u32_e32 vcc, s78, v70
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v84, vcc
	v_cndmask_b32_e32 v65, v65, v82, vcc
	v_add_u32_e32 v82, 1, v65
	v_cmp_le_u32_e32 vcc, s78, v70
	s_nop 1
	v_cndmask_b32_e32 v84, v65, v82, vcc
.LBB0_44:
	s_or_b64 exec, exec, s[14:15]
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[46:47]
	v_lshl_add_u64 v[86:87], v[80:81], 0, s[94:95]
	v_or_b32_e32 v65, s79, v87
	v_cmp_ne_u32_e32 vcc, 0, v65
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[68:69], exec, s[14:15]
	s_cbranch_execz .LBB0_46
	s_add_u32 s14, s78, s82
	s_mov_b32 s83, s82
	s_addc_u32 s15, s79, s82
	s_xor_b64 s[56:57], s[14:15], s[82:83]
	v_cvt_f32_u32_e32 v65, s56
	v_cvt_f32_u32_e32 v70, s57
	s_sub_u32 s14, 0, s56
	s_subb_u32 s15, 0, s57
	v_ashrrev_i32_e32 v140, 31, v87
	v_fmac_f32_e32 v65, 0x4f800000, v70
	v_rcp_f32_e32 v65, v65
	v_mov_b32_e32 v141, v140
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[140:141]
	v_xor_b32_e32 v82, v86, v140
	v_mul_f32_e32 v65, 0x5f7ffffc, v65
	v_mul_f32_e32 v70, 0x2f800000, v65
	v_trunc_f32_e32 v70, v70
	v_fmac_f32_e32 v65, 0xcf800000, v70
	v_cvt_u32_f32_e32 v70, v70
	v_cvt_u32_f32_e32 v65, v65
	v_readfirstlane_b32 s22, v70
	v_readfirstlane_b32 s23, v65
	s_mul_i32 s73, s14, s22
	s_mul_hi_u32 s81, s14, s23
	s_mul_i32 s75, s15, s23
	s_add_i32 s73, s81, s73
	s_add_i32 s73, s73, s75
	s_mul_i32 s83, s14, s23
	s_mul_i32 s81, s23, s73
	s_mul_hi_u32 vcc_lo, s23, s83
	s_mul_hi_u32 s75, s23, s73
	s_add_u32 s81, vcc_lo, s81
	s_addc_u32 s75, 0, s75
	s_mul_hi_u32 vcc_hi, s22, s83
	s_mul_i32 s83, s22, s83
	s_add_u32 s81, s81, s83
	s_mul_hi_u32 vcc_lo, s22, s73
	s_addc_u32 s75, s75, vcc_hi
	s_addc_u32 s81, vcc_lo, 0
	s_mul_i32 s73, s22, s73
	s_add_u32 s73, s75, s73
	s_addc_u32 s75, 0, s81
	s_add_u32 s23, s23, s73
	s_addc_u32 s22, s22, s75
	s_mul_i32 s73, s14, s22
	s_mul_hi_u32 s75, s14, s23
	s_add_i32 s73, s75, s73
	s_mul_i32 s15, s15, s23
	s_add_i32 s73, s73, s15
	s_mul_i32 s14, s14, s23
	s_mul_hi_u32 s75, s22, s14
	s_mul_i32 s81, s22, s14
	s_mul_i32 vcc_lo, s23, s73
	s_mul_hi_u32 s14, s23, s14
	s_mul_hi_u32 s83, s23, s73
	s_add_u32 s14, s14, vcc_lo
	s_addc_u32 s83, 0, s83
	s_add_u32 s14, s14, s81
	s_mul_hi_u32 s15, s22, s73
	s_addc_u32 s14, s83, s75
	s_addc_u32 s15, s15, 0
	s_mul_i32 s73, s22, s73
	s_add_u32 s14, s14, s73
	s_addc_u32 s15, 0, s15
	s_add_u32 s23, s23, s14
	s_addc_u32 s22, s22, s15
	v_xor_b32_e32 v65, v87, v140
	v_mad_u64_u32 v[86:87], s[14:15], v82, s22, 0
	v_mul_hi_u32 v70, v82, s23
	v_lshl_add_u64 v[86:87], v[70:71], 0, v[86:87]
	v_mad_u64_u32 v[142:143], s[14:15], v65, s23, 0
	v_add_co_u32_e32 v70, vcc, v86, v142
	v_mad_u64_u32 v[92:93], s[14:15], v65, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v86, vcc, v87, v143, vcc
	v_mov_b32_e32 v87, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[92:93]
	v_mul_lo_u32 v70, s57, v86
	v_mul_lo_u32 v85, s56, v87
	v_mad_u64_u32 v[86:87], s[14:15], s56, v86, 0
	v_add3_u32 v70, v87, v85, v70
	v_sub_u32_e32 v85, v65, v70
	v_mov_b32_e32 v87, s57
	v_sub_co_u32_e32 v82, vcc, v82, v86
	s_nop 1
	v_subb_co_u32_e64 v85, s[14:15], v85, v87, vcc
	v_subrev_co_u32_e64 v86, s[14:15], s56, v82
	v_subb_co_u32_e32 v65, vcc, v65, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v92, s[22:23], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s57, v92
	v_subb_co_u32_e64 v85, s[14:15], v85, v87, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s56, v86
	v_subrev_co_u32_e64 v87, s[14:15], s56, v86
	s_nop 0
	v_cndmask_b32_e64 v123, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s57, v92
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v85, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v93, v93, v123, s[22:23]
	v_cmp_le_u32_e32 vcc, s57, v65
	v_cmp_ne_u32_e64 s[14:15], 0, v93
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s56, v82
	v_cndmask_b32_e64 v85, v92, v85, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s57, v65
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_cndmask_b32_e64 v70, v86, v87, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v82, v70, vcc
	v_cndmask_b32_e32 v65, v65, v85, vcc
	v_xor_b32_e32 v70, v70, v140
	v_xor_b32_e32 v65, v65, v140
	v_sub_co_u32_e32 v92, vcc, v70, v140
	s_nop 1
	v_subb_co_u32_e32 v93, vcc, v65, v140, vcc
.LBB0_46:
	s_andn2_saveexec_b64 s[14:15], s[68:69]
	s_cbranch_execz .LBB0_48
	v_rcp_iflag_f32_e32 v65, v100
	s_sub_i32 s22, 0, s78
	v_mul_f32_e32 v65, 0x4f7ffffe, v65
	v_cvt_u32_f32_e32 v65, v65
	v_mul_lo_u32 v70, s22, v65
	v_mul_hi_u32 v70, v65, v70
	v_add_u32_e32 v65, v65, v70
	v_mul_hi_u32 v65, v86, v65
	v_mul_lo_u32 v65, v65, s78
	v_sub_u32_e32 v65, v86, v65
	v_subrev_u32_e32 v70, s78, v65
	v_cmp_le_u32_e32 vcc, s78, v65
	s_nop 1
	v_cndmask_b32_e32 v65, v65, v70, vcc
	v_subrev_u32_e32 v70, s78, v65
	v_cmp_le_u32_e32 vcc, s78, v65
	s_nop 1
	v_cndmask_b32_e32 v70, v65, v70, vcc
	v_mov_b64_e32 v[92:93], v[70:71]
.LBB0_48:
	s_or_b64 exec, exec, s[14:15]
	v_ashrrev_i32_e32 v65, 31, v83
	v_xor_b32_e32 v65, v84, v65
	v_ashrrev_i32_e32 v70, 31, v93
	v_mul_lo_u32 v65, v65, s66
	v_and_b32_e32 v70, s78, v70
	v_add3_u32 v65, v65, v92, v70
	buffer_load_ubyte v123, v65, s[8:11], 0 offen
	v_sub_co_u32_e32 v65, vcc, 0xfffffefd, v78
	v_lshl_add_u64 v[82:83], v[78:79], 0, s[96:97]
	s_nop 0
	v_subb_co_u32_e32 v70, vcc, -1, v79, vcc
	v_cmp_gt_i32_e32 vcc, 0, v83
	s_nop 1
	v_cndmask_b32_e32 v87, v83, v70, vcc
	v_cndmask_b32_e32 v86, v82, v65, vcc
	v_or_b32_e32 v65, s79, v87
	v_cmp_ne_u32_e32 vcc, 0, v65
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[22:23], exec, s[14:15]
	s_cbranch_execz .LBB0_50
	s_add_u32 s14, s78, s82
	s_mov_b32 s83, s82
	s_addc_u32 s15, s79, s82
	s_xor_b64 s[56:57], s[14:15], s[82:83]
	v_cvt_f32_u32_e32 v65, s56
	v_cvt_f32_u32_e32 v70, s57
	s_sub_u32 s14, 0, s56
	s_subb_u32 s15, 0, s57
	v_ashrrev_i32_e32 v84, 31, v87
	v_fmac_f32_e32 v65, 0x4f800000, v70
	v_rcp_f32_e32 v65, v65
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[84:85]
	v_xor_b32_e32 v82, v86, v84
	v_mul_f32_e32 v65, 0x5f7ffffc, v65
	v_mul_f32_e32 v70, 0x2f800000, v65
	v_trunc_f32_e32 v70, v70
	v_fmac_f32_e32 v65, 0xcf800000, v70
	v_cvt_u32_f32_e32 v70, v70
	v_cvt_u32_f32_e32 v65, v65
	v_readfirstlane_b32 s68, v70
	v_readfirstlane_b32 s69, v65
	s_mul_i32 s73, s14, s68
	s_mul_hi_u32 s81, s14, s69
	s_mul_i32 s75, s15, s69
	s_add_i32 s73, s81, s73
	s_add_i32 s73, s73, s75
	s_mul_i32 s83, s14, s69
	s_mul_i32 s81, s69, s73
	s_mul_hi_u32 vcc_lo, s69, s83
	s_mul_hi_u32 s75, s69, s73
	s_add_u32 s81, vcc_lo, s81
	s_addc_u32 s75, 0, s75
	s_mul_hi_u32 vcc_hi, s68, s83
	s_mul_i32 s83, s68, s83
	s_add_u32 s81, s81, s83
	s_mul_hi_u32 vcc_lo, s68, s73
	s_addc_u32 s75, s75, vcc_hi
	s_addc_u32 s81, vcc_lo, 0
	s_mul_i32 s73, s68, s73
	s_add_u32 s73, s75, s73
	s_addc_u32 s75, 0, s81
	s_add_u32 s69, s69, s73
	s_addc_u32 s68, s68, s75
	s_mul_i32 s73, s14, s68
	s_mul_hi_u32 s75, s14, s69
	s_add_i32 s73, s75, s73
	s_mul_i32 s15, s15, s69
	s_add_i32 s73, s73, s15
	s_mul_i32 s14, s14, s69
	s_mul_hi_u32 s75, s68, s14
	s_mul_i32 s81, s68, s14
	s_mul_i32 vcc_lo, s69, s73
	s_mul_hi_u32 s14, s69, s14
	s_mul_hi_u32 s83, s69, s73
	s_add_u32 s14, s14, vcc_lo
	s_addc_u32 s83, 0, s83
	s_add_u32 s14, s14, s81
	s_mul_hi_u32 s15, s68, s73
	s_addc_u32 s14, s83, s75
	s_addc_u32 s15, s15, 0
	s_mul_i32 s73, s68, s73
	s_add_u32 s14, s14, s73
	s_addc_u32 s15, 0, s15
	s_add_u32 s69, s69, s14
	s_addc_u32 s68, s68, s15
	v_xor_b32_e32 v65, v87, v84
	v_mad_u64_u32 v[86:87], s[14:15], v82, s68, 0
	v_mul_hi_u32 v70, v82, s69
	v_lshl_add_u64 v[86:87], v[70:71], 0, v[86:87]
	v_mad_u64_u32 v[140:141], s[14:15], v65, s69, 0
	v_add_co_u32_e32 v70, vcc, v86, v140
	v_mad_u64_u32 v[92:93], s[14:15], v65, s68, 0
	s_nop 0
	v_addc_co_u32_e32 v86, vcc, v87, v141, vcc
	v_mov_b32_e32 v87, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[92:93]
	v_mul_lo_u32 v70, s57, v86
	v_mul_lo_u32 v85, s56, v87
	v_mad_u64_u32 v[92:93], s[14:15], s56, v86, 0
	v_add3_u32 v70, v93, v85, v70
	v_sub_u32_e32 v85, v65, v70
	v_mov_b32_e32 v93, s57
	v_sub_co_u32_e32 v82, vcc, v82, v92
	v_lshl_add_u64 v[140:141], v[86:87], 0, 1
	s_nop 0
	v_subb_co_u32_e64 v85, s[14:15], v85, v93, vcc
	v_subrev_co_u32_e64 v92, s[14:15], s56, v82
	v_subb_co_u32_e32 v65, vcc, v65, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s57, v85
	v_cmp_le_u32_e32 vcc, s57, v65
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s56, v92
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s56, v82
	v_cndmask_b32_e64 v92, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s57, v85
	v_cndmask_b32_e64 v82, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s57, v65
	v_cndmask_b32_e64 v85, v93, v92, s[14:15]
	v_lshl_add_u64 v[92:93], v[86:87], 0, 2
	v_cndmask_b32_e32 v65, v70, v82, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v70, v140, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v65
	s_nop 1
	v_cndmask_b32_e32 v65, v86, v70, vcc
	v_xor_b32_e32 v70, s82, v84
	v_xor_b32_e32 v65, v65, v70
	v_sub_co_u32_e32 v84, vcc, v65, v70
.LBB0_50:
	s_andn2_saveexec_b64 s[14:15], s[22:23]
	s_cbranch_execz .LBB0_52
	v_rcp_iflag_f32_e32 v65, v100
	s_sub_i32 s22, 0, s78
	v_mul_f32_e32 v65, 0x4f7ffffe, v65
	v_cvt_u32_f32_e32 v65, v65
	v_mul_lo_u32 v70, s22, v65
	v_mul_hi_u32 v70, v65, v70
	v_add_u32_e32 v65, v65, v70
	v_mul_hi_u32 v65, v86, v65
	v_mul_lo_u32 v70, v65, s78
	v_sub_u32_e32 v70, v86, v70
	v_add_u32_e32 v82, 1, v65
	v_subrev_u32_e32 v84, s78, v70
	v_cmp_le_u32_e32 vcc, s78, v70
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v84, vcc
	v_cndmask_b32_e32 v65, v65, v82, vcc
	v_add_u32_e32 v82, 1, v65
	v_cmp_le_u32_e32 vcc, s78, v70
	s_nop 1
	v_cndmask_b32_e32 v84, v65, v82, vcc
.LBB0_52:
	s_or_b64 exec, exec, s[14:15]
	v_lshl_add_u64 v[86:87], v[80:81], 0, s[96:97]
	v_or_b32_e32 v65, s79, v87
	v_cmp_ne_u32_e32 vcc, 0, v65
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[68:69], exec, s[14:15]
	s_cbranch_execz .LBB0_54
	s_add_u32 s14, s78, s82
	s_mov_b32 s83, s82
	s_addc_u32 s15, s79, s82
	s_xor_b64 s[56:57], s[14:15], s[82:83]
	v_cvt_f32_u32_e32 v65, s56
	v_cvt_f32_u32_e32 v70, s57
	s_sub_u32 s14, 0, s56
	s_subb_u32 s15, 0, s57
	v_ashrrev_i32_e32 v140, 31, v87
	v_fmac_f32_e32 v65, 0x4f800000, v70
	v_rcp_f32_e32 v65, v65
	v_mov_b32_e32 v141, v140
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[140:141]
	v_xor_b32_e32 v82, v86, v140
	v_mul_f32_e32 v65, 0x5f7ffffc, v65
	v_mul_f32_e32 v70, 0x2f800000, v65
	v_trunc_f32_e32 v70, v70
	v_fmac_f32_e32 v65, 0xcf800000, v70
	v_cvt_u32_f32_e32 v70, v70
	v_cvt_u32_f32_e32 v65, v65
	v_readfirstlane_b32 s22, v70
	v_readfirstlane_b32 s23, v65
	s_mul_i32 s73, s14, s22
	s_mul_hi_u32 s81, s14, s23
	s_mul_i32 s75, s15, s23
	s_add_i32 s73, s81, s73
	s_add_i32 s73, s73, s75
	s_mul_i32 s83, s14, s23
	s_mul_i32 s81, s23, s73
	s_mul_hi_u32 vcc_lo, s23, s83
	s_mul_hi_u32 s75, s23, s73
	s_add_u32 s81, vcc_lo, s81
	s_addc_u32 s75, 0, s75
	s_mul_hi_u32 vcc_hi, s22, s83
	s_mul_i32 s83, s22, s83
	s_add_u32 s81, s81, s83
	s_mul_hi_u32 vcc_lo, s22, s73
	s_addc_u32 s75, s75, vcc_hi
	s_addc_u32 s81, vcc_lo, 0
	s_mul_i32 s73, s22, s73
	s_add_u32 s73, s75, s73
	s_addc_u32 s75, 0, s81
	s_add_u32 s23, s23, s73
	s_addc_u32 s22, s22, s75
	s_mul_i32 s73, s14, s22
	s_mul_hi_u32 s75, s14, s23
	s_add_i32 s73, s75, s73
	s_mul_i32 s15, s15, s23
	s_add_i32 s73, s73, s15
	s_mul_i32 s14, s14, s23
	s_mul_hi_u32 s75, s22, s14
	s_mul_i32 s81, s22, s14
	s_mul_i32 vcc_lo, s23, s73
	s_mul_hi_u32 s14, s23, s14
	s_mul_hi_u32 s83, s23, s73
	s_add_u32 s14, s14, vcc_lo
	s_addc_u32 s83, 0, s83
	s_add_u32 s14, s14, s81
	s_mul_hi_u32 s15, s22, s73
	s_addc_u32 s14, s83, s75
	s_addc_u32 s15, s15, 0
	s_mul_i32 s73, s22, s73
	s_add_u32 s14, s14, s73
	s_addc_u32 s15, 0, s15
	s_add_u32 s23, s23, s14
	s_addc_u32 s22, s22, s15
	v_xor_b32_e32 v65, v87, v140
	v_mad_u64_u32 v[86:87], s[14:15], v82, s22, 0
	v_mul_hi_u32 v70, v82, s23
	v_lshl_add_u64 v[86:87], v[70:71], 0, v[86:87]
	v_mad_u64_u32 v[142:143], s[14:15], v65, s23, 0
	v_add_co_u32_e32 v70, vcc, v86, v142
	v_mad_u64_u32 v[92:93], s[14:15], v65, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v86, vcc, v87, v143, vcc
	v_mov_b32_e32 v87, s74
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[92:93]
	v_mul_lo_u32 v70, s57, v86
	v_mul_lo_u32 v85, s56, v87
	v_mad_u64_u32 v[86:87], s[14:15], s56, v86, 0
	v_add3_u32 v70, v87, v85, v70
	v_sub_u32_e32 v85, v65, v70
	v_mov_b32_e32 v87, s57
	v_sub_co_u32_e32 v82, vcc, v82, v86
	s_nop 1
	v_subb_co_u32_e64 v85, s[14:15], v85, v87, vcc
	v_subrev_co_u32_e64 v86, s[14:15], s56, v82
	v_subb_co_u32_e32 v65, vcc, v65, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v92, s[22:23], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s57, v92
	v_subb_co_u32_e64 v85, s[14:15], v85, v87, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s56, v86
	v_subrev_co_u32_e64 v87, s[14:15], s56, v86
	s_nop 0
	v_cndmask_b32_e64 v141, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s57, v92
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v85, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v93, v93, v141, s[22:23]
	v_cmp_le_u32_e32 vcc, s57, v65
	v_cmp_ne_u32_e64 s[14:15], 0, v93
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s56, v82
	v_cndmask_b32_e64 v85, v92, v85, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s57, v65
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_cndmask_b32_e64 v70, v86, v87, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v82, v70, vcc
	v_cndmask_b32_e32 v65, v65, v85, vcc
	v_xor_b32_e32 v70, v70, v140
	v_xor_b32_e32 v65, v65, v140
	v_sub_co_u32_e32 v92, vcc, v70, v140
	s_nop 1
	v_subb_co_u32_e32 v93, vcc, v65, v140, vcc
.LBB0_54:
	s_andn2_saveexec_b64 s[14:15], s[68:69]
	s_cbranch_execz .LBB0_56
	v_rcp_iflag_f32_e32 v65, v100
	s_sub_i32 s22, 0, s78
	v_mul_f32_e32 v65, 0x4f7ffffe, v65
	v_cvt_u32_f32_e32 v65, v65
	v_mul_lo_u32 v70, s22, v65
	v_mul_hi_u32 v70, v65, v70
	v_add_u32_e32 v65, v65, v70
	v_mul_hi_u32 v65, v86, v65
	v_mul_lo_u32 v65, v65, s78
	v_sub_u32_e32 v65, v86, v65
	v_subrev_u32_e32 v70, s78, v65
	v_cmp_le_u32_e32 vcc, s78, v65
	s_nop 1
	v_cndmask_b32_e32 v65, v65, v70, vcc
	v_subrev_u32_e32 v70, s78, v65
	v_cmp_le_u32_e32 vcc, s78, v65
	s_nop 1
	v_cndmask_b32_e32 v70, v65, v70, vcc
	v_mov_b64_e32 v[92:93], v[70:71]
.LBB0_56:
	s_or_b64 exec, exec, s[14:15]
	v_ashrrev_i32_e32 v65, 31, v83
	v_xor_b32_e32 v65, v84, v65
	v_ashrrev_i32_e32 v70, 31, v93
	v_mul_lo_u32 v65, v65, s66
	v_and_b32_e32 v70, s78, v70
	v_add3_u32 v65, v65, v92, v70
	buffer_load_ubyte v65, v65, s[8:11], 0 offen
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	s_lshl_b64 s[14:15], s[86:87], 7
	v_mov_b32_e32 v87, s29
	v_or_b32_e32 v86, s28, v32
	v_or_b32_e32 v84, s14, v28
	v_lshl_add_u64 v[82:83], v[86:87], 0, s[24:25]
	v_add_u32_e32 v70, 0x100, v84
	v_cmp_gt_i64_e32 vcc, s[30:31], v[82:83]
	v_add_u32_e32 v85, v70, v30
	s_and_b64 s[14:15], s[2:3], vcc
	s_mov_b32 m0, s54
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[136:139], v[8:11], v[4:7], v33, v31 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e64 v82, v29, v85, s[14:15]
	s_and_b64 s[14:15], s[26:27], vcc
	buffer_load_dwordx4 v82, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[128:131], v[8:11], v[16:19], v33, v31 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s16
	s_nop 1
	v_add_u32_e32 v16, v70, v34
	v_cndmask_b32_e64 v16, v29, v16, s[14:15]
	buffer_load_dwordx4 v16, s[40:43], 0 offen lds
	v_add_u32_e32 v16, v70, v36
	s_and_b64 s[14:15], s[18:19], vcc
	v_cndmask_b32_e64 v16, v29, v16, s[14:15]
	s_mov_b32 m0, s60
	s_and_b64 vcc, s[20:21], vcc
	buffer_load_dwordx4 v16, s[40:43], 0 offen lds
	v_add_u32_e32 v16, v70, v38
	v_cndmask_b32_e32 v16, v29, v16, vcc
	s_mov_b32 m0, s61
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[132:135], v[12:15], v[4:7], v33, v106 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v16, s[40:43], 0 offen lds
	s_lshl_b64 s[14:15], s[86:87], 13
	v_lshl_add_u64 v[82:83], v[40:41], 0, s[14:15]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[124:127], v[12:15], v[8:11], v33, v106 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 v[132:135], v52 offset:16384
	ds_read_b128 v[128:131], v54 offset:18432
	ds_read_b128 v[124:127], v52 offset:18432
	ds_read_b128 v[136:139], v54 offset:16384
	ds_read_b32 v33, v50 offset:50176
	s_mov_b64 s[14:15], 0x4000
	v_sub_co_u32_e32 v14, vcc, 0xffffbfff, v82
	v_lshl_add_u64 v[12:13], v[82:83], 0, s[14:15]
	s_nop 0
	v_subb_co_u32_e32 v15, vcc, -1, v83, vcc
	v_cmp_gt_i32_e32 vcc, 0, v13
	s_nop 1
	v_cndmask_b32_e32 v17, v13, v15, vcc
	v_cndmask_b32_e32 v16, v12, v14, vcc
	v_or_b32_e32 v14, s31, v17
	v_cmp_ne_u32_e32 vcc, 0, v14
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[22:23], exec, s[14:15]
	s_cbranch_execz .LBB0_58
	s_add_u32 s14, s30, s80
	s_mov_b32 s81, s80
	s_addc_u32 s15, s31, s80
	s_xor_b64 s[28:29], s[14:15], s[80:81]
	v_cvt_f32_u32_e32 v14, s28
	v_cvt_f32_u32_e32 v15, s29
	s_sub_u32 s14, 0, s28
	s_subb_u32 s15, 0, s29
	v_fmac_f32_e32 v14, 0x4f800000, v15
	v_rcp_f32_e32 v14, v14
	s_nop 0
	v_mul_f32_e32 v14, 0x5f7ffffc, v14
	v_mul_f32_e32 v15, 0x2f800000, v14
	v_trunc_f32_e32 v15, v15
	v_fmac_f32_e32 v14, 0xcf800000, v15
	v_cvt_u32_f32_e32 v15, v15
	v_cvt_u32_f32_e32 v14, v14
	v_readfirstlane_b32 s56, v15
	v_readfirstlane_b32 s57, v14
	s_mul_i32 s68, s14, s56
	s_mul_hi_u32 s73, s14, s57
	s_mul_i32 s69, s15, s57
	s_add_i32 s68, s73, s68
	s_add_i32 s68, s68, s69
	s_mul_i32 s75, s14, s57
	s_mul_i32 s73, s57, s68
	s_mul_hi_u32 s81, s57, s75
	s_mul_hi_u32 s69, s57, s68
	s_add_u32 s73, s81, s73
	s_addc_u32 s69, 0, s69
	s_mul_hi_u32 s83, s56, s75
	s_mul_i32 s75, s56, s75
	s_add_u32 s73, s73, s75
	s_mul_hi_u32 s81, s56, s68
	s_addc_u32 s69, s69, s83
	s_addc_u32 s73, s81, 0
	s_mul_i32 s68, s56, s68
	s_add_u32 s68, s69, s68
	s_addc_u32 s69, 0, s73
	s_add_u32 s57, s57, s68
	s_addc_u32 s56, s56, s69
	s_mul_i32 s68, s14, s56
	s_mul_hi_u32 s69, s14, s57
	s_add_i32 s68, s69, s68
	s_mul_i32 s15, s15, s57
	s_add_i32 s68, s68, s15
	s_mul_i32 s14, s14, s57
	s_mul_hi_u32 s69, s56, s14
	s_mul_i32 s73, s56, s14
	s_mul_i32 s81, s57, s68
	s_mul_hi_u32 s14, s57, s14
	s_mul_hi_u32 s75, s57, s68
	s_add_u32 s14, s14, s81
	s_addc_u32 s75, 0, s75
	s_add_u32 s14, s14, s73
	s_mul_hi_u32 s15, s56, s68
	s_addc_u32 s14, s75, s69
	s_addc_u32 s15, s15, 0
	s_mul_i32 s68, s56, s68
	s_add_u32 s14, s14, s68
	v_ashrrev_i32_e32 v14, 31, v17
	s_addc_u32 s15, 0, s15
	v_mov_b32_e32 v15, v14
	s_add_u32 s57, s57, s14
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[14:15]
	s_addc_u32 s56, s56, s15
	v_xor_b32_e32 v31, v16, v14
	v_xor_b32_e32 v15, v17, v14
	v_mad_u64_u32 v[16:17], s[14:15], v31, s56, 0
	v_mul_hi_u32 v70, v31, s57
	v_lshl_add_u64 v[16:17], v[70:71], 0, v[16:17]
	v_mad_u64_u32 v[92:93], s[14:15], v15, s57, 0
	v_add_co_u32_e32 v16, vcc, v16, v92
	v_mad_u64_u32 v[18:19], s[14:15], v15, s56, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v93, vcc
	v_mov_b32_e32 v17, s74
	s_nop 0
	v_addc_co_u32_e32 v19, vcc, 0, v19, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[18:19]
	v_mul_lo_u32 v70, s29, v16
	v_mul_lo_u32 v85, s28, v17
	v_mad_u64_u32 v[18:19], s[14:15], s28, v16, 0
	v_add3_u32 v70, v19, v85, v70
	v_sub_u32_e32 v19, v15, v70
	v_mov_b32_e32 v85, s29
	v_sub_co_u32_e32 v31, vcc, v31, v18
	v_lshl_add_u64 v[92:93], v[16:17], 0, 1
	s_nop 0
	v_subb_co_u32_e64 v18, s[14:15], v19, v85, vcc
	v_subrev_co_u32_e64 v19, s[14:15], s28, v31
	v_subb_co_u32_e32 v15, vcc, v15, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v18, s[14:15], 0, v18, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s29, v18
	v_cmp_le_u32_e32 vcc, s29, v15
	v_xor_b32_e32 v14, s80, v14
	v_cndmask_b32_e64 v85, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v19
	s_nop 1
	v_cndmask_b32_e64 v19, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v18
	s_nop 1
	v_cndmask_b32_e64 v85, v85, v19, s[14:15]
	v_lshl_add_u64 v[18:19], v[16:17], 0, 2
	v_cndmask_b32_e64 v17, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v31
	s_nop 1
	v_cndmask_b32_e64 v19, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v15
	s_nop 1
	v_cndmask_b32_e32 v15, v17, v19, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v17, v92, v18, vcc
	v_cmp_ne_u32_e32 vcc, 0, v15
	s_nop 1
	v_cndmask_b32_e32 v15, v16, v17, vcc
	v_xor_b32_e32 v15, v15, v14
	v_sub_co_u32_e32 v14, vcc, v15, v14
.LBB0_58:
	s_andn2_saveexec_b64 s[14:15], s[22:23]
	s_cbranch_execz .LBB0_60
	v_cvt_f32_u32_e32 v14, s30
	s_sub_i32 s22, 0, s30
	v_rcp_iflag_f32_e32 v14, v14
	s_nop 0
	v_mul_f32_e32 v14, 0x4f7ffffe, v14
	v_cvt_u32_f32_e32 v14, v14
	v_mul_lo_u32 v15, s22, v14
	v_mul_hi_u32 v15, v14, v15
	v_add_u32_e32 v14, v14, v15
	v_mul_hi_u32 v14, v16, v14
	v_mul_lo_u32 v15, v14, s30
	v_sub_u32_e32 v15, v16, v15
	v_add_u32_e32 v17, 1, v14
	v_subrev_u32_e32 v16, s30, v15
	v_cmp_le_u32_e32 vcc, s30, v15
	s_nop 1
	v_cndmask_b32_e32 v15, v15, v16, vcc
	v_cndmask_b32_e32 v14, v14, v17, vcc
	v_add_u32_e32 v16, 1, v14
	v_cmp_le_u32_e32 vcc, s30, v15
	s_nop 1
	v_cndmask_b32_e32 v14, v14, v16, vcc
.LBB0_60:
	s_or_b64 exec, exec, s[14:15]
	v_or_b32_e32 v15, s31, v13
	v_cmp_ne_u32_e32 vcc, 0, v15
	v_ashrrev_i32_e32 v16, 31, v13
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[28:29], exec, s[14:15]
	s_cbranch_execz .LBB0_62
	s_add_u32 s14, s30, s80
	s_mov_b32 s81, s80
	s_addc_u32 s15, s31, s80
	s_xor_b64 s[56:57], s[14:15], s[80:81]
	v_cvt_f32_u32_e32 v15, s56
	v_cvt_f32_u32_e32 v17, s57
	s_sub_u32 s14, 0, s56
	s_subb_u32 s15, 0, s57
	v_fmac_f32_e32 v15, 0x4f800000, v17
	v_rcp_f32_e32 v15, v15
	s_nop 0
	v_mul_f32_e32 v15, 0x5f7ffffc, v15
	v_mul_f32_e32 v17, 0x2f800000, v15
	v_trunc_f32_e32 v17, v17
	v_fmac_f32_e32 v15, 0xcf800000, v17
	v_cvt_u32_f32_e32 v17, v17
	v_cvt_u32_f32_e32 v15, v15
	v_readfirstlane_b32 s22, v17
	v_readfirstlane_b32 s23, v15
	s_mul_i32 s68, s14, s22
	s_mul_hi_u32 s73, s14, s23
	s_mul_i32 s69, s15, s23
	s_add_i32 s68, s73, s68
	s_add_i32 s68, s68, s69
	s_mul_i32 s75, s14, s23
	s_mul_i32 s73, s23, s68
	s_mul_hi_u32 s81, s23, s75
	s_mul_hi_u32 s69, s23, s68
	s_add_u32 s73, s81, s73
	s_addc_u32 s69, 0, s69
	s_mul_hi_u32 s83, s22, s75
	s_mul_i32 s75, s22, s75
	s_add_u32 s73, s73, s75
	s_mul_hi_u32 s81, s22, s68
	s_addc_u32 s69, s69, s83
	s_addc_u32 s73, s81, 0
	s_mul_i32 s68, s22, s68
	s_add_u32 s68, s69, s68
	s_addc_u32 s69, 0, s73
	s_add_u32 s23, s23, s68
	s_addc_u32 s22, s22, s69
	s_mul_i32 s68, s14, s22
	s_mul_hi_u32 s69, s14, s23
	s_add_i32 s68, s69, s68
	s_mul_i32 s15, s15, s23
	s_add_i32 s68, s68, s15
	s_mul_i32 s14, s14, s23
	s_mul_hi_u32 s69, s22, s14
	s_mul_i32 s73, s22, s14
	s_mul_i32 s81, s23, s68
	s_mul_hi_u32 s14, s23, s14
	s_mul_hi_u32 s75, s23, s68
	s_add_u32 s14, s14, s81
	s_addc_u32 s75, 0, s75
	s_add_u32 s14, s14, s73
	s_mul_hi_u32 s15, s22, s68
	s_addc_u32 s14, s75, s69
	s_addc_u32 s15, s15, 0
	s_mul_i32 s68, s22, s68
	s_add_u32 s14, s14, s68
	s_addc_u32 s15, 0, s15
	v_mov_b32_e32 v17, v16
	s_add_u32 s23, s23, s14
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[16:17]
	s_addc_u32 s22, s22, s15
	v_xor_b32_e32 v17, v12, v16
	v_xor_b32_e32 v15, v13, v16
	v_mad_u64_u32 v[12:13], s[14:15], v17, s22, 0
	v_mul_hi_u32 v70, v17, s23
	v_lshl_add_u64 v[12:13], v[70:71], 0, v[12:13]
	v_mad_u64_u32 v[92:93], s[14:15], v15, s23, 0
	v_add_co_u32_e32 v12, vcc, v12, v92
	v_mad_u64_u32 v[18:19], s[14:15], v15, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v13, v93, vcc
	v_mov_b32_e32 v13, s74
	s_nop 0
	v_addc_co_u32_e32 v19, vcc, 0, v19, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[18:19]
	v_mul_lo_u32 v18, s57, v12
	v_mul_lo_u32 v19, s56, v13
	v_mad_u64_u32 v[12:13], s[14:15], s56, v12, 0
	v_add3_u32 v13, v13, v19, v18
	v_sub_u32_e32 v18, v15, v13
	v_mov_b32_e32 v19, s57
	v_sub_co_u32_e32 v12, vcc, v17, v12
	s_nop 1
	v_subb_co_u32_e64 v17, s[14:15], v18, v19, vcc
	v_subrev_co_u32_e64 v18, s[14:15], s56, v12
	v_subb_co_u32_e32 v13, vcc, v15, v13, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v31, s[22:23], 0, v17, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s57, v31
	v_subb_co_u32_e64 v17, s[14:15], v17, v19, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s56, v18
	v_subrev_co_u32_e64 v19, s[14:15], s56, v18
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s57, v31
	v_subbrev_co_u32_e64 v17, s[14:15], 0, v17, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v70, v70, v85, s[22:23]
	v_cmp_le_u32_e32 vcc, s57, v13
	v_cmp_ne_u32_e64 s[14:15], 0, v70
	s_nop 0
	v_cndmask_b32_e64 v15, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s56, v12
	v_cndmask_b32_e64 v17, v31, v17, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s57, v13
	s_nop 1
	v_cndmask_b32_e32 v15, v15, v31, vcc
	v_cmp_ne_u32_e32 vcc, 0, v15
	v_cndmask_b32_e64 v15, v18, v19, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v12, v12, v15, vcc
	v_cndmask_b32_e32 v13, v13, v17, vcc
	v_xor_b32_e32 v12, v12, v16
	v_xor_b32_e32 v13, v13, v16
	v_sub_co_u32_e32 v18, vcc, v12, v16
	s_nop 1
	v_subb_co_u32_e32 v19, vcc, v13, v16, vcc
.LBB0_62:
	s_andn2_saveexec_b64 s[14:15], s[28:29]
	s_cbranch_execz .LBB0_64
	v_cvt_f32_u32_e32 v13, s30
	s_sub_i32 s22, 0, s30
	v_rcp_iflag_f32_e32 v13, v13
	s_nop 0
	v_mul_f32_e32 v13, 0x4f7ffffe, v13
	v_cvt_u32_f32_e32 v13, v13
	v_mul_lo_u32 v15, s22, v13
	v_mul_hi_u32 v15, v13, v15
	v_add_u32_e32 v13, v13, v15
	v_mul_hi_u32 v13, v12, v13
	v_mul_lo_u32 v13, v13, s30
	v_sub_u32_e32 v12, v12, v13
	v_subrev_u32_e32 v13, s30, v12
	v_cmp_le_u32_e32 vcc, s30, v12
	s_nop 1
	v_cndmask_b32_e32 v12, v12, v13, vcc
	v_subrev_u32_e32 v13, s30, v12
	v_cmp_le_u32_e32 vcc, s30, v12
	s_nop 1
	v_cndmask_b32_e32 v70, v12, v13, vcc
	v_mov_b64_e32 v[18:19], v[70:71]
.LBB0_64:
	s_or_b64 exec, exec, s[14:15]
	v_ashrrev_i32_e32 v12, 31, v19
	v_and_b32_e32 v13, s31, v12
	v_and_b32_e32 v12, s30, v12
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[18:19]
	v_ashrrev_i32_e32 v15, 31, v13
	v_xor_b32_e32 v13, v15, v13
	v_xor_b32_e32 v14, v14, v16
	v_ashrrev_i32_e32 v16, 31, v13
	v_xor_b32_e32 v12, v15, v12
	v_lshrrev_b32_e32 v70, 27, v16
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[70:71]
	v_alignbit_b32 v12, v13, v12, 5
	v_mul_lo_u32 v13, v14, s46
	s_mov_b32 m0, s64
	v_xad_u32 v12, v12, v15, v13
	buffer_load_dword v12, s[4:7], 0 offen lds
	v_lshlrev_b16_e32 v12, 8, v104
	v_lshlrev_b16_e32 v13, 8, v105
	v_or_b32_e32 v12, v102, v12
	v_or_b32_sdwa v13, v103, v13 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v16, 8, v53
	v_or_b32_sdwa v15, v12, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v12, 8, v95
	v_lshlrev_b16_e32 v13, 8, v101
	v_or_b32_e32 v12, v63, v12
	v_or_b32_sdwa v13, v94, v13 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v16, v39, v16 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v14, v12, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v12, 8, v59
	v_lshlrev_b16_e32 v13, 8, v61
	v_or_b32_e32 v12, v55, v12
	v_or_b32_sdwa v13, v57, v13 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt vmcnt(7)
	v_lshlrev_b16_e32 v17, 8, v122
	v_or_b32_sdwa v13, v12, v13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v12, 8, v51
	v_or_b32_e32 v12, v35, v12
	v_or_b32_sdwa v12, v12, v16 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v16, 8, v121
	v_or_b32_e32 v16, v119, v16
	v_or_b32_sdwa v17, v120, v17 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v31, 8, v110
	v_or_b32_sdwa v19, v16, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v16, 8, v117
	v_lshlrev_b16_e32 v17, 8, v118
	v_or_b32_e32 v16, v115, v16
	v_or_b32_sdwa v17, v116, v17 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v31, v108, v31 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v18, v16, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v16, 8, v113
	v_lshlrev_b16_e32 v17, 8, v114
	v_or_b32_e32 v16, v111, v16
	v_or_b32_sdwa v17, v112, v17 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_nop 0
	v_or_b32_sdwa v17, v16, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v16, 8, v109
	v_or_b32_e32 v16, v107, v16
	v_or_b32_sdwa v16, v16, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_waitcnt vmcnt(5) lgkmcnt(0)
	s_barrier
	s_mov_b64 s[14:15], 0x1000
	v_lshl_add_u64 v[92:93], v[90:91], 0, s[14:15]
	s_movk_i32 s14, 0xefff
	v_sub_co_u32_e32 v35, vcc, s14, v90
	s_add_u32 s14, s44, s72
	s_nop 0
	v_subb_co_u32_e32 v39, vcc, -1, v91, vcc
	s_mov_b32 s73, s72
	s_addc_u32 s15, s45, s72
	v_cmp_gt_i32_e32 vcc, 0, v93
	s_xor_b64 s[68:69], s[14:15], s[72:73]
	v_cvt_f32_u32_e32 v51, s69
	v_cndmask_b32_e32 v95, v93, v39, vcc
	v_cvt_f32_u32_e32 v39, s68
	v_cndmask_b32_e32 v94, v92, v35, vcc
	s_sub_u32 s14, 0, s68
	s_subb_u32 s15, 0, s69
	v_fmac_f32_e32 v39, 0x4f800000, v51
	v_rcp_f32_e32 v35, v39
	v_ashrrev_i32_e32 v102, 31, v95
	v_mov_b32_e32 v103, v102
	v_lshl_add_u64 v[94:95], v[94:95], 0, v[102:103]
	v_mul_f32_e32 v35, 0x5f7ffffc, v35
	v_mul_f32_e32 v53, 0x2f800000, v35
	v_trunc_f32_e32 v53, v53
	v_fmac_f32_e32 v35, 0xcf800000, v53
	v_cvt_u32_f32_e32 v35, v35
	v_cvt_u32_f32_e32 v53, v53
	v_xor_b32_e32 v51, v94, v102
	v_xor_b32_e32 v39, v95, v102
	v_readfirstlane_b32 s22, v35
	v_readfirstlane_b32 s29, v53
	s_mul_hi_u32 s28, s14, s22
	s_mul_i32 s56, s14, s29
	s_mul_i32 s23, s15, s22
	s_add_i32 s28, s28, s56
	s_add_i32 s28, s28, s23
	s_mul_i32 s57, s14, s22
	s_mul_i32 s56, s22, s28
	s_mul_hi_u32 s73, s22, s57
	s_mul_hi_u32 s23, s22, s28
	s_add_u32 s56, s73, s56
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 s75, s29, s57
	s_mul_i32 s57, s29, s57
	s_add_u32 s56, s56, s57
	s_mul_hi_u32 s73, s29, s28
	s_addc_u32 s23, s23, s75
	s_addc_u32 s56, s73, 0
	s_mul_i32 s28, s29, s28
	s_add_u32 s23, s23, s28
	s_addc_u32 s28, 0, s56
	s_add_u32 s22, s22, s23
	s_addc_u32 s23, s29, s28
	s_mul_i32 s28, s14, s23
	s_mul_hi_u32 s29, s14, s22
	s_add_i32 s28, s29, s28
	s_mul_i32 s15, s15, s22
	s_add_i32 s28, s28, s15
	s_mul_i32 s14, s14, s22
	s_mul_hi_u32 s29, s23, s14
	s_mul_i32 s56, s23, s14
	s_mul_i32 s73, s22, s28
	s_mul_hi_u32 s14, s22, s14
	s_mul_hi_u32 s57, s22, s28
	s_add_u32 s14, s14, s73
	s_addc_u32 s57, 0, s57
	s_add_u32 s14, s14, s56
	s_mul_hi_u32 s15, s23, s28
	s_addc_u32 s14, s57, s29
	s_addc_u32 s15, s15, 0
	s_mul_i32 s28, s23, s28
	s_add_u32 s14, s14, s28
	s_addc_u32 s15, 0, s15
	s_add_u32 s56, s22, s14
	s_addc_u32 s57, s23, s15
	v_mad_u64_u32 v[94:95], s[14:15], v51, s57, 0
	v_mul_hi_u32 v70, v51, s56
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[94:95]
	v_mad_u64_u32 v[106:107], s[14:15], v39, s56, 0
	v_add_co_u32_e32 v35, vcc, v94, v106
	v_mad_u64_u32 v[104:105], s[14:15], v39, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v95, v107, vcc
	v_and_b32_e32 v31, 0xffff, v123
	s_nop 0
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[104:105]
	v_mul_lo_u32 v35, s69, v94
	v_mul_lo_u32 v53, s68, v95
	v_mad_u64_u32 v[104:105], s[14:15], s68, v94, 0
	v_add3_u32 v35, v105, v53, v35
	v_sub_co_u32_e32 v51, vcc, v51, v104
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[128:131], v[12:15], v[20:23], v33, v31 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_subb_co_u32_e64 v53, s[14:15], v39, v35, vcc
	v_sub_u32_e32 v39, v39, v35
	v_mov_b32_e32 v35, s69
	v_cmp_le_u32_e64 s[14:15], s69, v53
	v_subb_co_u32_e32 v39, vcc, v39, v35, vcc
	s_nop 0
	v_cndmask_b32_e64 v55, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v51
	v_subrev_co_u32_e32 v51, vcc, s68, v51
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v39, vcc, 0, v39, vcc
	v_cmp_eq_u32_e64 s[14:15], s69, v53
	v_cmp_le_u32_e32 vcc, s69, v39
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[136:139], v[12:15], v[0:3], v33, v31 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e64 v53, v55, v57, s[14:15]
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v51
	s_nop 1
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v39
	s_nop 1
	v_cndmask_b32_e32 v39, v55, v51, vcc
	v_cmp_ne_u32_e32 vcc, 0, v39
	v_xor_b32_e32 v51, s72, v102
	s_nop 0
	v_cndmask_b32_e64 v39, 1, 2, vcc
	v_add_u32_e32 v39, v94, v39
	v_cmp_ne_u32_e32 vcc, 0, v53
	s_nop 1
	v_cndmask_b32_e32 v39, v94, v39, vcc
	v_ashrrev_i32_e32 v94, 31, v93
	v_mov_b32_e32 v95, v94
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[94:95]
	v_xor_b32_e32 v39, v39, v51
	v_xor_b32_e32 v53, v92, v94
	v_sub_u32_e32 v39, v39, v51
	v_xor_b32_e32 v51, v93, v94
	v_mad_u64_u32 v[92:93], s[14:15], v53, s57, 0
	v_mul_hi_u32 v70, v53, s56
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[92:93]
	v_mad_u64_u32 v[104:105], s[14:15], v51, s56, 0
	v_add_co_u32_e32 v55, vcc, v92, v104
	v_mad_u64_u32 v[102:103], s[14:15], v51, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v93, v105, vcc
	v_xad_u32 v39, v39, v94, v37
	s_nop 0
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[102:103]
	v_mul_lo_u32 v55, s69, v92
	v_mul_lo_u32 v57, s68, v93
	v_mad_u64_u32 v[92:93], s[14:15], s68, v92, 0
	v_add3_u32 v55, v93, v57, v55
	v_sub_u32_e32 v57, v51, v55
	v_sub_co_u32_e32 v53, vcc, v53, v92
	v_mul_lo_u32 v39, v39, s33
	s_nop 0
	v_subb_co_u32_e64 v57, s[14:15], v57, v35, vcc
	v_subrev_co_u32_e64 v59, s[14:15], s68, v53
	v_subb_co_u32_e32 v51, vcc, v51, v55, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v61, s[22:23], 0, v57, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s69, v61
	v_subb_co_u32_e64 v57, s[14:15], v57, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s68, v59
	v_cmp_le_u32_e32 vcc, s69, v51
	v_lshl_add_u64 v[92:93], v[88:89], 0, s[88:89]
	v_cndmask_b32_e64 v70, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s69, v61
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v53
	v_cndmask_b32_e64 v63, v63, v70, s[22:23]
	v_subrev_co_u32_e64 v70, s[14:15], s68, v59
	v_cmp_gt_i32_e64 s[22:23], 0, v93
	s_nop 0
	v_subbrev_co_u32_e64 v57, s[14:15], 0, v57, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v63
	v_ashrrev_i32_e32 v85, 31, v93
	s_nop 0
	v_cndmask_b32_e64 v57, v61, v57, s[14:15]
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v51
	s_nop 1
	v_cndmask_b32_e32 v55, v55, v61, vcc
	v_cmp_ne_u32_e32 vcc, 0, v55
	v_cndmask_b32_e64 v55, v59, v70, s[14:15]
	s_mov_b64 s[14:15], 0x801
	v_cndmask_b32_e32 v53, v53, v55, vcc
	v_cndmask_b32_e32 v51, v51, v57, vcc
	v_xor_b32_e32 v53, v53, v94
	v_xor_b32_e32 v51, v51, v94
	v_sub_co_u32_e32 v53, vcc, v53, v94
	s_nop 1
	v_subb_co_u32_e32 v51, vcc, v51, v94, vcc
	v_ashrrev_i32_e32 v51, 31, v51
	v_and_b32_e32 v51, s44, v51
	v_lshl_add_u64 v[94:95], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf7fe
	v_add3_u32 v39, v51, v53, v39
	v_sub_co_u32_e32 v51, vcc, s14, v88
	v_cndmask_b32_e64 v102, v94, v51, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v53, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v103, v95, v53, s[22:23]
	v_ashrrev_i32_e32 v104, 31, v103
	v_mov_b32_e32 v105, v104
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[104:105]
	v_xor_b32_e32 v53, v102, v104
	v_xor_b32_e32 v51, v103, v104
	v_mad_u64_u32 v[102:103], s[14:15], v53, s57, 0
	v_mul_hi_u32 v70, v53, s56
	v_lshl_add_u64 v[102:103], v[70:71], 0, v[102:103]
	v_mad_u64_u32 v[108:109], s[14:15], v51, s56, 0
	v_add_co_u32_e32 v55, vcc, v102, v108
	v_mad_u64_u32 v[106:107], s[14:15], v51, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v103, v109, vcc
	v_ashrrev_i32_e32 v92, 31, v95
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[102:103], v[70:71], 0, v[106:107]
	v_mul_lo_u32 v55, s69, v102
	v_mul_lo_u32 v57, s68, v103
	v_mad_u64_u32 v[106:107], s[14:15], s68, v102, 0
	v_add3_u32 v55, v107, v57, v55
	v_sub_co_u32_e32 v53, vcc, v53, v106
	v_mov_b32_e32 v93, v92
	s_nop 0
	v_subb_co_u32_e64 v57, s[14:15], v51, v55, vcc
	v_sub_u32_e32 v51, v51, v55
	v_cmp_le_u32_e64 s[14:15], s69, v57
	v_subb_co_u32_e32 v51, vcc, v51, v35, vcc
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v53
	v_subrev_co_u32_e32 v53, vcc, s68, v53
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v51, vcc, 0, v51, vcc
	v_cmp_le_u32_e32 vcc, s69, v51
	v_cmp_eq_u32_e64 s[14:15], s69, v57
	v_lshl_add_u64 v[94:95], v[94:95], 0, v[92:93]
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v53
	v_cndmask_b32_e64 v57, v59, v61, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v51
	s_nop 1
	v_cndmask_b32_e32 v51, v55, v53, vcc
	v_cmp_ne_u32_e32 vcc, 0, v51
	v_xor_b32_e32 v53, s72, v104
	v_xor_b32_e32 v55, v94, v92
	v_cndmask_b32_e64 v51, 1, 2, vcc
	v_add_u32_e32 v51, v102, v51
	v_cmp_ne_u32_e32 vcc, 0, v57
	v_mul_hi_u32 v70, v55, s56
	s_nop 0
	v_cndmask_b32_e32 v51, v102, v51, vcc
	v_xor_b32_e32 v51, v51, v53
	v_sub_u32_e32 v51, v51, v53
	v_xad_u32 v53, v51, v85, v37
	v_xor_b32_e32 v51, v95, v92
	v_mad_u64_u32 v[94:95], s[14:15], v55, s57, 0
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[94:95]
	v_mad_u64_u32 v[104:105], s[14:15], v51, s56, 0
	v_add_co_u32_e32 v57, vcc, v94, v104
	v_mad_u64_u32 v[102:103], s[14:15], v51, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v95, v105, vcc
	v_mul_lo_u32 v53, v53, s33
	s_nop 0
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[102:103]
	v_mul_lo_u32 v57, s69, v94
	v_mul_lo_u32 v59, s68, v95
	v_mad_u64_u32 v[94:95], s[14:15], s68, v94, 0
	v_add3_u32 v57, v95, v59, v57
	v_sub_u32_e32 v59, v51, v57
	v_sub_co_u32_e32 v55, vcc, v55, v94
	s_nop 1
	v_subb_co_u32_e64 v59, s[14:15], v59, v35, vcc
	v_subrev_co_u32_e64 v61, s[14:15], s68, v55
	v_subb_co_u32_e32 v51, vcc, v51, v57, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v63, s[28:29], 0, v59, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v63
	v_subb_co_u32_e64 v59, s[14:15], v59, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v61
	v_cmp_le_u32_e32 vcc, s69, v51
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v63
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v55
	v_cndmask_b32_e64 v70, v70, v93, s[28:29]
	v_subrev_co_u32_e64 v93, s[14:15], s68, v61
	s_nop 1
	v_subbrev_co_u32_e64 v59, s[14:15], 0, v59, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v70
	s_nop 1
	v_cndmask_b32_e64 v59, v63, v59, s[14:15]
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v51
	s_nop 1
	v_cndmask_b32_e32 v57, v57, v63, vcc
	v_cmp_ne_u32_e32 vcc, 0, v57
	v_cndmask_b32_e64 v57, v61, v93, s[14:15]
	s_mov_b64 s[14:15], 0x802
	v_cndmask_b32_e32 v55, v55, v57, vcc
	v_cndmask_b32_e32 v51, v51, v59, vcc
	v_xor_b32_e32 v55, v55, v92
	v_xor_b32_e32 v51, v51, v92
	v_sub_co_u32_e32 v70, vcc, v55, v92
	s_nop 1
	v_subb_co_u32_e32 v51, vcc, v51, v92, vcc
	v_ashrrev_i32_e32 v51, 31, v51
	v_and_b32_e32 v51, s44, v51
	v_lshl_add_u64 v[92:93], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf7fd
	v_add3_u32 v51, v51, v70, v53
	v_sub_co_u32_e32 v53, vcc, s14, v88
	v_cndmask_b32_e64 v94, v92, v53, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v55, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v95, v93, v55, s[22:23]
	v_ashrrev_i32_e32 v102, 31, v95
	v_mov_b32_e32 v103, v102
	v_lshl_add_u64 v[94:95], v[94:95], 0, v[102:103]
	v_xor_b32_e32 v55, v94, v102
	v_xor_b32_e32 v53, v95, v102
	v_mad_u64_u32 v[94:95], s[14:15], v55, s57, 0
	v_mul_hi_u32 v70, v55, s56
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[94:95]
	v_mad_u64_u32 v[106:107], s[14:15], v53, s56, 0
	v_add_co_u32_e32 v57, vcc, v94, v106
	v_mad_u64_u32 v[104:105], s[14:15], v53, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v95, v107, vcc
	s_nop 1
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[104:105]
	v_mul_lo_u32 v57, s69, v94
	v_mul_lo_u32 v59, s68, v95
	v_mad_u64_u32 v[104:105], s[14:15], s68, v94, 0
	v_add3_u32 v57, v105, v59, v57
	v_sub_co_u32_e32 v55, vcc, v55, v104
	s_nop 1
	v_subb_co_u32_e64 v59, s[14:15], v53, v57, vcc
	v_sub_u32_e32 v53, v53, v57
	v_cmp_le_u32_e64 s[14:15], s69, v59
	v_subb_co_u32_e32 v53, vcc, v53, v35, vcc
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v55
	v_subrev_co_u32_e32 v55, vcc, s68, v55
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v53, vcc, 0, v53, vcc
	v_cmp_le_u32_e32 vcc, s69, v53
	v_cmp_eq_u32_e64 s[14:15], s69, v59
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v55
	v_cndmask_b32_e64 v59, v61, v63, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v53
	s_nop 1
	v_cndmask_b32_e32 v53, v57, v55, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_xor_b32_e32 v55, s72, v102
	s_nop 0
	v_cndmask_b32_e64 v53, 1, 2, vcc
	v_add_u32_e32 v53, v94, v53
	v_cmp_ne_u32_e32 vcc, 0, v59
	s_nop 1
	v_cndmask_b32_e32 v53, v94, v53, vcc
	v_ashrrev_i32_e32 v94, 31, v93
	v_mov_b32_e32 v95, v94
	v_xor_b32_e32 v53, v53, v55
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[94:95]
	v_sub_u32_e32 v53, v53, v55
	v_xor_b32_e32 v57, v92, v94
	v_xad_u32 v55, v53, v85, v37
	v_xor_b32_e32 v53, v93, v94
	v_mad_u64_u32 v[92:93], s[14:15], v57, s57, 0
	v_mul_hi_u32 v70, v57, s56
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[92:93]
	v_mad_u64_u32 v[104:105], s[14:15], v53, s56, 0
	v_add_co_u32_e32 v59, vcc, v92, v104
	v_mad_u64_u32 v[102:103], s[14:15], v53, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v93, v105, vcc
	v_mul_lo_u32 v55, v55, s33
	s_nop 0
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[102:103]
	v_mul_lo_u32 v59, s69, v92
	v_mul_lo_u32 v61, s68, v93
	v_mad_u64_u32 v[92:93], s[14:15], s68, v92, 0
	v_add3_u32 v59, v93, v61, v59
	v_sub_u32_e32 v61, v53, v59
	v_sub_co_u32_e32 v57, vcc, v57, v92
	s_nop 1
	v_subb_co_u32_e64 v61, s[14:15], v61, v35, vcc
	v_subrev_co_u32_e64 v63, s[14:15], s68, v57
	v_subb_co_u32_e32 v53, vcc, v53, v59, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v70, s[28:29], 0, v61, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v70
	v_subb_co_u32_e64 v61, s[14:15], v61, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v63
	v_cmp_le_u32_e32 vcc, s69, v53
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v70
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v57
	v_cndmask_b32_e64 v92, v92, v93, s[28:29]
	v_subrev_co_u32_e64 v93, s[14:15], s68, v63
	s_nop 1
	v_subbrev_co_u32_e64 v61, s[14:15], 0, v61, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v92
	s_nop 1
	v_cndmask_b32_e64 v61, v70, v61, s[14:15]
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v53
	s_nop 1
	v_cndmask_b32_e32 v59, v59, v70, vcc
	v_cmp_ne_u32_e32 vcc, 0, v59
	v_cndmask_b32_e64 v59, v63, v93, s[14:15]
	s_mov_b64 s[14:15], 0x803
	v_cndmask_b32_e32 v57, v57, v59, vcc
	v_cndmask_b32_e32 v53, v53, v61, vcc
	v_xor_b32_e32 v57, v57, v94
	v_xor_b32_e32 v53, v53, v94
	v_sub_co_u32_e32 v70, vcc, v57, v94
	v_lshl_add_u64 v[92:93], v[88:89], 0, s[14:15]
	s_nop 0
	v_subb_co_u32_e32 v53, vcc, v53, v94, vcc
	v_ashrrev_i32_e32 v53, 31, v53
	v_and_b32_e32 v53, s44, v53
	s_movk_i32 s14, 0xf7fc
	v_add3_u32 v55, v53, v70, v55
	v_sub_co_u32_e32 v53, vcc, s14, v88
	v_cndmask_b32_e64 v94, v92, v53, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v57, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v95, v93, v57, s[22:23]
	v_ashrrev_i32_e32 v102, 31, v95
	v_mov_b32_e32 v103, v102
	v_lshl_add_u64 v[94:95], v[94:95], 0, v[102:103]
	v_xor_b32_e32 v57, v94, v102
	v_xor_b32_e32 v53, v95, v102
	v_mad_u64_u32 v[94:95], s[14:15], v57, s57, 0
	v_mul_hi_u32 v70, v57, s56
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[94:95]
	v_mad_u64_u32 v[106:107], s[14:15], v53, s56, 0
	v_add_co_u32_e32 v59, vcc, v94, v106
	v_mad_u64_u32 v[104:105], s[14:15], v53, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v95, v107, vcc
	s_nop 1
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[104:105]
	v_mul_lo_u32 v59, s69, v94
	v_mul_lo_u32 v61, s68, v95
	v_mad_u64_u32 v[104:105], s[14:15], s68, v94, 0
	v_add3_u32 v59, v105, v61, v59
	v_sub_co_u32_e32 v57, vcc, v57, v104
	s_nop 1
	v_subb_co_u32_e64 v61, s[14:15], v53, v59, vcc
	v_sub_u32_e32 v53, v53, v59
	v_cmp_le_u32_e64 s[14:15], s69, v61
	v_subb_co_u32_e32 v53, vcc, v53, v35, vcc
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v57
	v_subrev_co_u32_e32 v57, vcc, s68, v57
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v53, vcc, 0, v53, vcc
	v_cmp_le_u32_e32 vcc, s69, v53
	v_cmp_eq_u32_e64 s[14:15], s69, v61
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v57
	v_cndmask_b32_e64 v61, v63, v70, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v53
	s_nop 1
	v_cndmask_b32_e32 v53, v59, v57, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_xor_b32_e32 v57, s72, v102
	s_nop 0
	v_cndmask_b32_e64 v53, 1, 2, vcc
	v_add_u32_e32 v53, v94, v53
	v_cmp_ne_u32_e32 vcc, 0, v61
	s_nop 1
	v_cndmask_b32_e32 v53, v94, v53, vcc
	v_ashrrev_i32_e32 v94, 31, v93
	v_mov_b32_e32 v95, v94
	v_xor_b32_e32 v53, v53, v57
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[94:95]
	v_sub_u32_e32 v53, v53, v57
	v_xor_b32_e32 v59, v92, v94
	v_xad_u32 v57, v53, v85, v37
	v_xor_b32_e32 v53, v93, v94
	v_mad_u64_u32 v[92:93], s[14:15], v59, s57, 0
	v_mul_hi_u32 v70, v59, s56
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[92:93]
	v_mad_u64_u32 v[104:105], s[14:15], v53, s56, 0
	v_add_co_u32_e32 v61, vcc, v92, v104
	v_mad_u64_u32 v[102:103], s[14:15], v53, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v93, v105, vcc
	v_mul_lo_u32 v57, v57, s33
	s_nop 0
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[102:103]
	v_mul_lo_u32 v61, s69, v92
	v_mul_lo_u32 v63, s68, v93
	v_mad_u64_u32 v[92:93], s[14:15], s68, v92, 0
	v_add3_u32 v61, v93, v63, v61
	v_sub_u32_e32 v63, v53, v61
	v_sub_co_u32_e32 v59, vcc, v59, v92
	s_nop 1
	v_subb_co_u32_e64 v63, s[14:15], v63, v35, vcc
	v_subrev_co_u32_e64 v70, s[14:15], s68, v59
	v_subb_co_u32_e32 v53, vcc, v53, v61, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v92, s[28:29], 0, v63, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v92
	v_subb_co_u32_e64 v63, s[14:15], v63, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v70
	v_cmp_le_u32_e32 vcc, s69, v53
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v92
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v59
	v_cndmask_b32_e64 v93, v93, v95, s[28:29]
	v_subrev_co_u32_e64 v95, s[14:15], s68, v70
	s_nop 1
	v_subbrev_co_u32_e64 v63, s[14:15], 0, v63, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v93
	s_nop 1
	v_cndmask_b32_e64 v63, v92, v63, s[14:15]
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v53
	s_nop 1
	v_cndmask_b32_e32 v61, v61, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v61
	v_cndmask_b32_e64 v61, v70, v95, s[14:15]
	s_mov_b64 s[14:15], 0x804
	v_cndmask_b32_e32 v59, v59, v61, vcc
	v_cndmask_b32_e32 v53, v53, v63, vcc
	v_xor_b32_e32 v59, v59, v94
	v_xor_b32_e32 v53, v53, v94
	v_sub_co_u32_e32 v70, vcc, v59, v94
	v_lshl_add_u64 v[92:93], v[88:89], 0, s[14:15]
	s_nop 0
	v_subb_co_u32_e32 v53, vcc, v53, v94, vcc
	v_ashrrev_i32_e32 v53, 31, v53
	v_and_b32_e32 v53, s44, v53
	s_movk_i32 s14, 0xf7fb
	v_add3_u32 v57, v53, v70, v57
	v_sub_co_u32_e32 v53, vcc, s14, v88
	v_cndmask_b32_e64 v94, v92, v53, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v59, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v95, v93, v59, s[22:23]
	v_ashrrev_i32_e32 v102, 31, v95
	v_mov_b32_e32 v103, v102
	v_lshl_add_u64 v[94:95], v[94:95], 0, v[102:103]
	v_xor_b32_e32 v59, v94, v102
	v_xor_b32_e32 v53, v95, v102
	v_mad_u64_u32 v[94:95], s[14:15], v59, s57, 0
	v_mul_hi_u32 v70, v59, s56
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[94:95]
	v_mad_u64_u32 v[106:107], s[14:15], v53, s56, 0
	v_add_co_u32_e32 v61, vcc, v94, v106
	v_mad_u64_u32 v[104:105], s[14:15], v53, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v95, v107, vcc
	s_nop 1
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[104:105]
	v_mul_lo_u32 v61, s69, v94
	v_mul_lo_u32 v63, s68, v95
	v_mad_u64_u32 v[104:105], s[14:15], s68, v94, 0
	v_add3_u32 v61, v105, v63, v61
	v_sub_co_u32_e32 v59, vcc, v59, v104
	s_nop 1
	v_subb_co_u32_e64 v63, s[14:15], v53, v61, vcc
	v_sub_u32_e32 v53, v53, v61
	v_cmp_le_u32_e64 s[14:15], s69, v63
	v_subb_co_u32_e32 v53, vcc, v53, v35, vcc
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v59
	v_subrev_co_u32_e32 v59, vcc, s68, v59
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v53, vcc, 0, v53, vcc
	v_cmp_le_u32_e32 vcc, s69, v53
	v_cmp_eq_u32_e64 s[14:15], s69, v63
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v59
	v_cndmask_b32_e64 v63, v70, v95, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v53
	s_nop 1
	v_cndmask_b32_e32 v53, v61, v59, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_xor_b32_e32 v59, s72, v102
	s_nop 0
	v_cndmask_b32_e64 v53, 1, 2, vcc
	v_add_u32_e32 v53, v94, v53
	v_cmp_ne_u32_e32 vcc, 0, v63
	s_nop 1
	v_cndmask_b32_e32 v53, v94, v53, vcc
	v_ashrrev_i32_e32 v94, 31, v93
	v_mov_b32_e32 v95, v94
	v_xor_b32_e32 v53, v53, v59
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[94:95]
	v_sub_u32_e32 v53, v53, v59
	v_xor_b32_e32 v61, v92, v94
	v_xad_u32 v59, v53, v85, v37
	v_xor_b32_e32 v53, v93, v94
	v_mad_u64_u32 v[92:93], s[14:15], v61, s57, 0
	v_mul_hi_u32 v70, v61, s56
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[92:93]
	v_mad_u64_u32 v[104:105], s[14:15], v53, s56, 0
	v_add_co_u32_e32 v63, vcc, v92, v104
	v_mad_u64_u32 v[102:103], s[14:15], v53, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v93, v105, vcc
	v_mul_lo_u32 v59, v59, s33
	s_nop 0
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[102:103]
	v_mul_lo_u32 v63, s69, v92
	v_mul_lo_u32 v70, s68, v93
	v_mad_u64_u32 v[92:93], s[14:15], s68, v92, 0
	v_add3_u32 v63, v93, v70, v63
	v_sub_u32_e32 v70, v53, v63
	v_sub_co_u32_e32 v61, vcc, v61, v92
	s_nop 1
	v_subb_co_u32_e64 v70, s[14:15], v70, v35, vcc
	v_subrev_co_u32_e64 v92, s[14:15], s68, v61
	v_subb_co_u32_e32 v53, vcc, v53, v63, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v93, s[28:29], 0, v70, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v93
	v_subb_co_u32_e64 v70, s[14:15], v70, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v92
	v_cmp_le_u32_e32 vcc, s69, v53
	s_nop 0
	v_cndmask_b32_e64 v101, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v93
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v61
	v_cndmask_b32_e64 v95, v95, v101, s[28:29]
	v_subrev_co_u32_e64 v101, s[14:15], s68, v92
	s_nop 1
	v_subbrev_co_u32_e64 v70, s[14:15], 0, v70, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v95
	s_nop 1
	v_cndmask_b32_e64 v70, v93, v70, s[14:15]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v53
	s_nop 1
	v_cndmask_b32_e32 v63, v63, v93, vcc
	v_cmp_ne_u32_e32 vcc, 0, v63
	v_cndmask_b32_e64 v63, v92, v101, s[14:15]
	s_mov_b64 s[14:15], 0x805
	v_cndmask_b32_e32 v61, v61, v63, vcc
	v_cndmask_b32_e32 v53, v53, v70, vcc
	v_xor_b32_e32 v61, v61, v94
	v_xor_b32_e32 v53, v53, v94
	v_sub_co_u32_e32 v70, vcc, v61, v94
	v_lshl_add_u64 v[92:93], v[88:89], 0, s[14:15]
	s_nop 0
	v_subb_co_u32_e32 v53, vcc, v53, v94, vcc
	v_ashrrev_i32_e32 v53, 31, v53
	v_and_b32_e32 v53, s44, v53
	s_movk_i32 s14, 0xf7fa
	v_add3_u32 v59, v53, v70, v59
	v_sub_co_u32_e32 v53, vcc, s14, v88
	v_cndmask_b32_e64 v94, v92, v53, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v61, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v95, v93, v61, s[22:23]
	v_ashrrev_i32_e32 v102, 31, v95
	v_mov_b32_e32 v103, v102
	v_lshl_add_u64 v[94:95], v[94:95], 0, v[102:103]
	v_xor_b32_e32 v61, v94, v102
	v_xor_b32_e32 v53, v95, v102
	v_mad_u64_u32 v[94:95], s[14:15], v61, s57, 0
	v_mul_hi_u32 v70, v61, s56
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[94:95]
	v_mad_u64_u32 v[106:107], s[14:15], v53, s56, 0
	v_add_co_u32_e32 v63, vcc, v94, v106
	v_mad_u64_u32 v[104:105], s[14:15], v53, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v95, v107, vcc
	s_nop 1
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[104:105]
	v_mul_lo_u32 v63, s69, v94
	v_mul_lo_u32 v70, s68, v95
	v_mad_u64_u32 v[104:105], s[14:15], s68, v94, 0
	v_add3_u32 v63, v105, v70, v63
	v_sub_co_u32_e32 v61, vcc, v61, v104
	s_nop 1
	v_subb_co_u32_e64 v70, s[14:15], v53, v63, vcc
	v_sub_u32_e32 v53, v53, v63
	v_cmp_le_u32_e64 s[14:15], s69, v70
	v_subb_co_u32_e32 v53, vcc, v53, v35, vcc
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v61
	v_subrev_co_u32_e32 v61, vcc, s68, v61
	s_nop 0
	v_cndmask_b32_e64 v101, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v53, vcc, 0, v53, vcc
	v_cmp_le_u32_e32 vcc, s69, v53
	v_cmp_eq_u32_e64 s[14:15], s69, v70
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v61
	v_cndmask_b32_e64 v70, v95, v101, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v53
	s_nop 1
	v_cndmask_b32_e32 v53, v63, v61, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_xor_b32_e32 v61, s72, v102
	s_nop 0
	v_cndmask_b32_e64 v53, 1, 2, vcc
	v_add_u32_e32 v53, v94, v53
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_nop 1
	v_cndmask_b32_e32 v53, v94, v53, vcc
	v_ashrrev_i32_e32 v94, 31, v93
	v_mov_b32_e32 v95, v94
	v_xor_b32_e32 v53, v53, v61
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[94:95]
	v_sub_u32_e32 v53, v53, v61
	v_xor_b32_e32 v63, v92, v94
	v_xad_u32 v61, v53, v85, v37
	v_xor_b32_e32 v53, v93, v94
	v_mad_u64_u32 v[92:93], s[14:15], v63, s57, 0
	v_mul_hi_u32 v70, v63, s56
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[92:93]
	v_mad_u64_u32 v[104:105], s[14:15], v53, s56, 0
	v_add_co_u32_e32 v70, vcc, v92, v104
	v_mad_u64_u32 v[102:103], s[14:15], v53, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v93, v105, vcc
	v_mul_lo_u32 v61, v61, s33
	s_nop 0
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[102:103]
	v_mul_lo_u32 v70, s69, v92
	v_mul_lo_u32 v95, s68, v93
	v_mad_u64_u32 v[92:93], s[14:15], s68, v92, 0
	v_add3_u32 v70, v93, v95, v70
	v_sub_u32_e32 v93, v53, v70
	v_sub_co_u32_e32 v63, vcc, v63, v92
	s_nop 1
	v_subb_co_u32_e64 v92, s[14:15], v93, v35, vcc
	v_subrev_co_u32_e64 v93, s[14:15], s68, v63
	v_subb_co_u32_e32 v53, vcc, v53, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v95, s[28:29], 0, v92, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v95
	v_subb_co_u32_e64 v92, s[14:15], v92, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v101, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v93
	v_cmp_le_u32_e32 vcc, s69, v53
	s_nop 0
	v_cndmask_b32_e64 v102, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v95
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v63
	v_cndmask_b32_e64 v101, v101, v102, s[28:29]
	v_subrev_co_u32_e64 v102, s[14:15], s68, v93
	s_nop 1
	v_subbrev_co_u32_e64 v92, s[14:15], 0, v92, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v101
	s_nop 1
	v_cndmask_b32_e64 v92, v95, v92, s[14:15]
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v53
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v95, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_cndmask_b32_e64 v70, v93, v102, s[14:15]
	s_mov_b64 s[14:15], 0x806
	v_cndmask_b32_e32 v63, v63, v70, vcc
	v_cndmask_b32_e32 v53, v53, v92, vcc
	v_xor_b32_e32 v63, v63, v94
	v_xor_b32_e32 v53, v53, v94
	v_sub_co_u32_e32 v70, vcc, v63, v94
	s_nop 1
	v_subb_co_u32_e32 v53, vcc, v53, v94, vcc
	v_ashrrev_i32_e32 v53, 31, v53
	v_and_b32_e32 v53, s44, v53
	v_lshl_add_u64 v[94:95], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf7f9
	v_add3_u32 v92, v53, v70, v61
	v_sub_co_u32_e32 v53, vcc, s14, v88
	v_cndmask_b32_e64 v102, v94, v53, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v61, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v103, v95, v61, s[22:23]
	v_ashrrev_i32_e32 v104, 31, v103
	v_mov_b32_e32 v105, v104
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[104:105]
	v_xor_b32_e32 v61, v102, v104
	v_xor_b32_e32 v53, v103, v104
	v_mad_u64_u32 v[102:103], s[14:15], v61, s57, 0
	v_mul_hi_u32 v70, v61, s56
	v_lshl_add_u64 v[102:103], v[70:71], 0, v[102:103]
	v_mad_u64_u32 v[108:109], s[14:15], v53, s56, 0
	v_add_co_u32_e32 v63, vcc, v102, v108
	v_mad_u64_u32 v[106:107], s[14:15], v53, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v103, v109, vcc
	s_nop 1
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[102:103], v[70:71], 0, v[106:107]
	v_mul_lo_u32 v63, s69, v102
	v_mul_lo_u32 v70, s68, v103
	v_mad_u64_u32 v[106:107], s[14:15], s68, v102, 0
	v_add3_u32 v63, v107, v70, v63
	v_sub_co_u32_e32 v61, vcc, v61, v106
	s_nop 1
	v_subb_co_u32_e64 v70, s[14:15], v53, v63, vcc
	v_sub_u32_e32 v53, v53, v63
	v_cmp_le_u32_e64 s[14:15], s69, v70
	v_subb_co_u32_e32 v53, vcc, v53, v35, vcc
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v61
	v_subrev_co_u32_e32 v61, vcc, s68, v61
	s_nop 0
	v_cndmask_b32_e64 v101, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v53, vcc, 0, v53, vcc
	v_cmp_le_u32_e32 vcc, s69, v53
	v_cmp_eq_u32_e64 s[14:15], s69, v70
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v61
	v_cndmask_b32_e64 v70, v93, v101, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v53
	s_nop 1
	v_cndmask_b32_e32 v53, v63, v61, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_xor_b32_e32 v61, s72, v104
	s_nop 0
	v_cndmask_b32_e64 v53, 1, 2, vcc
	v_add_u32_e32 v53, v102, v53
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_nop 1
	v_cndmask_b32_e32 v53, v102, v53, vcc
	v_ashrrev_i32_e32 v102, 31, v95
	v_mov_b32_e32 v103, v102
	v_xor_b32_e32 v53, v53, v61
	v_lshl_add_u64 v[94:95], v[94:95], 0, v[102:103]
	v_sub_u32_e32 v53, v53, v61
	v_xor_b32_e32 v63, v94, v102
	v_xad_u32 v61, v53, v85, v37
	v_xor_b32_e32 v53, v95, v102
	v_mad_u64_u32 v[94:95], s[14:15], v63, s57, 0
	v_mul_hi_u32 v70, v63, s56
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[94:95]
	v_mad_u64_u32 v[106:107], s[14:15], v53, s56, 0
	v_add_co_u32_e32 v70, vcc, v94, v106
	v_mad_u64_u32 v[104:105], s[14:15], v53, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v95, v107, vcc
	v_mul_lo_u32 v61, v61, s33
	s_nop 0
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[104:105]
	v_mul_lo_u32 v70, s69, v94
	v_mul_lo_u32 v93, s68, v95
	v_mad_u64_u32 v[94:95], s[14:15], s68, v94, 0
	v_add3_u32 v70, v95, v93, v70
	v_sub_u32_e32 v93, v53, v70
	v_sub_co_u32_e32 v63, vcc, v63, v94
	s_nop 1
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, vcc
	v_subrev_co_u32_e64 v94, s[14:15], s68, v63
	v_subb_co_u32_e32 v53, vcc, v53, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v95, s[28:29], 0, v93, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v95
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v101, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v94
	v_cmp_le_u32_e32 vcc, s69, v53
	s_nop 0
	v_cndmask_b32_e64 v103, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v95
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v63
	v_cndmask_b32_e64 v101, v101, v103, s[28:29]
	v_subrev_co_u32_e64 v103, s[14:15], s68, v94
	s_nop 1
	v_subbrev_co_u32_e64 v93, s[14:15], 0, v93, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v101
	s_nop 1
	v_cndmask_b32_e64 v93, v95, v93, s[14:15]
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v53
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v95, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_cndmask_b32_e64 v70, v94, v103, s[14:15]
	s_mov_b64 s[14:15], 0x807
	v_cndmask_b32_e32 v63, v63, v70, vcc
	v_cndmask_b32_e32 v53, v53, v93, vcc
	v_xor_b32_e32 v63, v63, v102
	v_xor_b32_e32 v53, v53, v102
	v_sub_co_u32_e32 v70, vcc, v63, v102
	v_lshl_add_u64 v[94:95], v[88:89], 0, s[14:15]
	s_nop 0
	v_subb_co_u32_e32 v53, vcc, v53, v102, vcc
	v_ashrrev_i32_e32 v53, 31, v53
	v_and_b32_e32 v53, s44, v53
	s_movk_i32 s14, 0xf7f8
	v_add3_u32 v93, v53, v70, v61
	v_sub_co_u32_e32 v53, vcc, s14, v88
	v_cndmask_b32_e64 v102, v94, v53, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v61, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v103, v95, v61, s[22:23]
	v_ashrrev_i32_e32 v104, 31, v103
	v_mov_b32_e32 v105, v104
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[104:105]
	v_xor_b32_e32 v61, v102, v104
	v_xor_b32_e32 v53, v103, v104
	v_mad_u64_u32 v[102:103], s[14:15], v61, s57, 0
	v_mul_hi_u32 v70, v61, s56
	v_lshl_add_u64 v[102:103], v[70:71], 0, v[102:103]
	v_mad_u64_u32 v[108:109], s[14:15], v53, s56, 0
	v_add_co_u32_e32 v63, vcc, v102, v108
	v_mad_u64_u32 v[106:107], s[14:15], v53, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v103, v109, vcc
	s_nop 1
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[102:103], v[70:71], 0, v[106:107]
	v_mul_lo_u32 v63, s69, v102
	v_mul_lo_u32 v70, s68, v103
	v_mad_u64_u32 v[106:107], s[14:15], s68, v102, 0
	v_add3_u32 v63, v107, v70, v63
	v_sub_co_u32_e32 v61, vcc, v61, v106
	s_nop 1
	v_subb_co_u32_e64 v70, s[14:15], v53, v63, vcc
	v_sub_u32_e32 v53, v53, v63
	v_cmp_le_u32_e64 s[14:15], s69, v70
	v_subb_co_u32_e32 v53, vcc, v53, v35, vcc
	s_nop 0
	v_cndmask_b32_e64 v101, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v61
	v_subrev_co_u32_e32 v61, vcc, s68, v61
	s_nop 0
	v_cndmask_b32_e64 v103, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v53, vcc, 0, v53, vcc
	v_cmp_le_u32_e32 vcc, s69, v53
	v_cmp_eq_u32_e64 s[14:15], s69, v70
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v61
	v_cndmask_b32_e64 v70, v101, v103, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v53
	s_nop 1
	v_cndmask_b32_e32 v53, v63, v61, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_xor_b32_e32 v61, s72, v104
	s_nop 0
	v_cndmask_b32_e64 v53, 1, 2, vcc
	v_add_u32_e32 v53, v102, v53
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_nop 1
	v_cndmask_b32_e32 v53, v102, v53, vcc
	v_ashrrev_i32_e32 v102, 31, v95
	v_mov_b32_e32 v103, v102
	v_xor_b32_e32 v53, v53, v61
	v_lshl_add_u64 v[94:95], v[94:95], 0, v[102:103]
	v_sub_u32_e32 v53, v53, v61
	v_xor_b32_e32 v63, v94, v102
	v_xad_u32 v61, v53, v85, v37
	v_xor_b32_e32 v53, v95, v102
	v_mad_u64_u32 v[94:95], s[14:15], v63, s57, 0
	v_mul_hi_u32 v70, v63, s56
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[94:95]
	v_mad_u64_u32 v[106:107], s[14:15], v53, s56, 0
	v_add_co_u32_e32 v70, vcc, v94, v106
	v_mad_u64_u32 v[104:105], s[14:15], v53, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v95, v107, vcc
	v_mul_lo_u32 v61, v61, s33
	s_nop 0
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[104:105]
	v_mul_lo_u32 v70, s69, v94
	v_mul_lo_u32 v101, s68, v95
	v_mad_u64_u32 v[94:95], s[14:15], s68, v94, 0
	v_add3_u32 v70, v95, v101, v70
	v_sub_u32_e32 v95, v53, v70
	v_sub_co_u32_e32 v63, vcc, v63, v94
	s_nop 1
	v_subb_co_u32_e64 v94, s[14:15], v95, v35, vcc
	v_subrev_co_u32_e64 v95, s[14:15], s68, v63
	v_subb_co_u32_e32 v53, vcc, v53, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v101, s[28:29], 0, v94, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v101
	v_subb_co_u32_e64 v94, s[14:15], v94, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v103, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v95
	v_cmp_le_u32_e32 vcc, s69, v53
	s_nop 0
	v_cndmask_b32_e64 v104, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v101
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v63
	v_cndmask_b32_e64 v103, v103, v104, s[28:29]
	v_subrev_co_u32_e64 v104, s[14:15], s68, v95
	s_nop 1
	v_subbrev_co_u32_e64 v94, s[14:15], 0, v94, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v103
	s_nop 1
	v_cndmask_b32_e64 v94, v101, v94, s[14:15]
	v_cndmask_b32_e64 v101, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v53
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v101, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_cndmask_b32_e64 v70, v95, v104, s[14:15]
	s_mov_b64 s[14:15], 0x808
	v_cndmask_b32_e32 v63, v63, v70, vcc
	v_cndmask_b32_e32 v53, v53, v94, vcc
	v_xor_b32_e32 v63, v63, v102
	v_xor_b32_e32 v53, v53, v102
	v_sub_co_u32_e32 v70, vcc, v63, v102
	s_nop 1
	v_subb_co_u32_e32 v53, vcc, v53, v102, vcc
	v_ashrrev_i32_e32 v53, 31, v53
	v_and_b32_e32 v53, s44, v53
	v_add3_u32 v70, v53, v70, v61
	buffer_load_ubyte v53, v39, s[36:39], 0 offen
	buffer_load_ubyte v61, v51, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v55, v55, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v63, v57, s[36:39], 0 offen
	buffer_load_ubyte v39, v59, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v57, v92, s[36:39], 0 offen
	buffer_load_ubyte v51, v93, s[36:39], 0 offen
	buffer_load_ubyte v59, v70, s[36:39], 0 offen
	v_lshl_add_u64 v[92:93], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf7f7
	v_sub_co_u32_e32 v70, vcc, s14, v88
	s_nop 1
	v_subb_co_u32_e32 v94, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v95, v93, v94, s[22:23]
	v_ashrrev_i32_e32 v102, 31, v95
	v_cndmask_b32_e64 v94, v92, v70, s[22:23]
	v_mov_b32_e32 v103, v102
	v_lshl_add_u64 v[94:95], v[94:95], 0, v[102:103]
	v_xor_b32_e32 v103, v94, v102
	v_xor_b32_e32 v101, v95, v102
	v_mad_u64_u32 v[94:95], s[14:15], v103, s57, 0
	v_mul_hi_u32 v70, v103, s56
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[94:95]
	v_mad_u64_u32 v[106:107], s[14:15], v101, s56, 0
	v_add_co_u32_e32 v70, vcc, v94, v106
	v_mad_u64_u32 v[104:105], s[14:15], v101, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v95, v107, vcc
	s_nop 1
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[104:105]
	v_mul_lo_u32 v70, s69, v94
	v_mul_lo_u32 v95, s68, v95
	v_mad_u64_u32 v[104:105], s[14:15], s68, v94, 0
	v_add3_u32 v70, v105, v95, v70
	v_sub_co_u32_e32 v95, vcc, v103, v104
	s_nop 1
	v_subb_co_u32_e64 v103, s[14:15], v101, v70, vcc
	v_sub_u32_e32 v70, v101, v70
	v_cmp_le_u32_e64 s[14:15], s69, v103
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	s_nop 0
	v_cndmask_b32_e64 v104, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v95
	v_subrev_co_u32_e32 v95, vcc, s68, v95
	s_nop 0
	v_cndmask_b32_e64 v105, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cmp_eq_u32_e64 s[14:15], s69, v103
	s_nop 0
	v_cndmask_b32_e64 v101, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v95
	v_cndmask_b32_e64 v103, v104, v105, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v70, v101, v95, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_nop 1
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v94, v70
	v_cmp_ne_u32_e32 vcc, 0, v103
	s_nop 1
	v_cndmask_b32_e32 v70, v94, v70, vcc
	v_xor_b32_e32 v94, s72, v102
	v_xor_b32_e32 v70, v70, v94
	v_sub_u32_e32 v70, v70, v94
	v_ashrrev_i32_e32 v94, 31, v93
	v_mov_b32_e32 v95, v94
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[94:95]
	v_xor_b32_e32 v106, v92, v94
	v_xad_u32 v101, v70, v85, v37
	v_xor_b32_e32 v95, v93, v94
	v_mad_u64_u32 v[92:93], s[14:15], v106, s57, 0
	v_mul_hi_u32 v70, v106, s56
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[92:93]
	v_mad_u64_u32 v[104:105], s[14:15], v95, s56, 0
	v_add_co_u32_e32 v70, vcc, v92, v104
	v_mad_u64_u32 v[102:103], s[14:15], v95, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v93, v105, vcc
	s_nop 1
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[102:103]
	v_mul_lo_u32 v70, s69, v92
	v_mul_lo_u32 v102, s68, v93
	v_mad_u64_u32 v[92:93], s[14:15], s68, v92, 0
	v_add3_u32 v70, v93, v102, v70
	v_sub_u32_e32 v93, v95, v70
	v_sub_co_u32_e32 v92, vcc, v106, v92
	s_nop 1
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, vcc
	v_subrev_co_u32_e64 v102, s[14:15], s68, v92
	v_subb_co_u32_e32 v70, vcc, v95, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v103, s[28:29], 0, v93, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v103
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v104, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v102
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v105, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v103
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v92
	v_cndmask_b32_e64 v104, v104, v105, s[28:29]
	v_subrev_co_u32_e64 v105, s[14:15], s68, v102
	s_nop 1
	v_subbrev_co_u32_e64 v93, s[14:15], 0, v93, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v104
	s_nop 1
	v_cndmask_b32_e64 v93, v103, v93, s[14:15]
	v_cndmask_b32_e64 v103, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v95, v95, v103, vcc
	v_cmp_ne_u32_e32 vcc, 0, v95
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v93, vcc
	v_cndmask_b32_e64 v93, v102, v105, s[14:15]
	v_cndmask_b32_e32 v92, v92, v93, vcc
	v_xor_b32_e32 v93, v70, v94
	v_xor_b32_e32 v70, v92, v94
	v_sub_co_u32_e32 v70, vcc, v70, v94
	s_mov_b64 s[14:15], 0x809
	s_nop 0
	v_subb_co_u32_e32 v93, vcc, v93, v94, vcc
	v_ashrrev_i32_e32 v92, 31, v93
	v_and_b32_e32 v92, s44, v92
	v_mul_lo_u32 v93, v101, s33
	v_add3_u32 v94, v92, v70, v93
	v_lshl_add_u64 v[92:93], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf7f6
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v102, v92, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v95, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v103, v93, v95, s[22:23]
	v_ashrrev_i32_e32 v104, 31, v103
	v_mov_b32_e32 v105, v104
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[104:105]
	v_xor_b32_e32 v101, v102, v104
	v_xor_b32_e32 v95, v103, v104
	v_mad_u64_u32 v[102:103], s[14:15], v101, s57, 0
	v_mul_hi_u32 v70, v101, s56
	v_lshl_add_u64 v[102:103], v[70:71], 0, v[102:103]
	v_mad_u64_u32 v[108:109], s[14:15], v95, s56, 0
	v_add_co_u32_e32 v70, vcc, v102, v108
	v_mad_u64_u32 v[106:107], s[14:15], v95, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v103, v109, vcc
	s_nop 1
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[102:103], v[70:71], 0, v[106:107]
	v_mul_lo_u32 v70, s69, v102
	v_mul_lo_u32 v103, s68, v103
	v_mad_u64_u32 v[106:107], s[14:15], s68, v102, 0
	v_add3_u32 v70, v107, v103, v70
	v_sub_co_u32_e32 v101, vcc, v101, v106
	s_nop 1
	v_subb_co_u32_e64 v103, s[14:15], v95, v70, vcc
	v_sub_u32_e32 v70, v95, v70
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	v_subrev_co_u32_e32 v95, vcc, s68, v101
	v_cmp_le_u32_e64 s[14:15], s69, v103
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v105, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v101
	v_cndmask_b32_e64 v101, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v95
	v_cndmask_b32_e64 v106, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v103
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v103, v105, v106, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v101, v95, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v95, s72, v104
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v102, v70
	v_cmp_ne_u32_e32 vcc, 0, v103
	s_nop 1
	v_cndmask_b32_e32 v70, v102, v70, vcc
	v_ashrrev_i32_e32 v102, 31, v93
	v_mov_b32_e32 v103, v102
	v_xor_b32_e32 v70, v70, v95
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[102:103]
	v_sub_u32_e32 v70, v70, v95
	v_xor_b32_e32 v103, v92, v102
	v_xad_u32 v95, v70, v85, v37
	v_xor_b32_e32 v101, v93, v102
	v_mad_u64_u32 v[92:93], s[14:15], v103, s57, 0
	v_mul_hi_u32 v70, v103, s56
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[92:93]
	v_mad_u64_u32 v[106:107], s[14:15], v101, s56, 0
	v_add_co_u32_e32 v70, vcc, v92, v106
	v_mad_u64_u32 v[104:105], s[14:15], v101, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v93, v107, vcc
	s_nop 1
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[104:105]
	v_mul_lo_u32 v70, s69, v92
	v_mul_lo_u32 v104, s68, v93
	v_mad_u64_u32 v[92:93], s[14:15], s68, v92, 0
	v_add3_u32 v70, v93, v104, v70
	v_sub_u32_e32 v93, v101, v70
	v_sub_co_u32_e32 v92, vcc, v103, v92
	s_nop 1
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, vcc
	v_subrev_co_u32_e64 v103, s[14:15], s68, v92
	v_subb_co_u32_e32 v70, vcc, v101, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v104, s[28:29], 0, v93, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v104
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v105, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v103
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v106, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v104
	v_cndmask_b32_e64 v101, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v92
	v_cndmask_b32_e64 v105, v105, v106, s[28:29]
	v_subrev_co_u32_e64 v106, s[14:15], s68, v103
	s_nop 1
	v_subbrev_co_u32_e64 v93, s[14:15], 0, v93, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v105
	s_nop 1
	v_cndmask_b32_e64 v93, v104, v93, s[14:15]
	v_cndmask_b32_e64 v104, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v101, v101, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v93, vcc
	v_cndmask_b32_e64 v93, v103, v106, s[14:15]
	v_cndmask_b32_e32 v92, v92, v93, vcc
	v_xor_b32_e32 v93, v70, v102
	v_xor_b32_e32 v70, v92, v102
	v_sub_co_u32_e32 v70, vcc, v70, v102
	s_mov_b64 s[14:15], 0x80a
	s_nop 0
	v_subb_co_u32_e32 v93, vcc, v93, v102, vcc
	v_ashrrev_i32_e32 v92, 31, v93
	v_and_b32_e32 v92, s44, v92
	v_mul_lo_u32 v93, v95, s33
	v_add3_u32 v95, v92, v70, v93
	v_lshl_add_u64 v[92:93], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf7f5
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v102, v92, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v101, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v103, v93, v101, s[22:23]
	v_ashrrev_i32_e32 v104, 31, v103
	v_mov_b32_e32 v105, v104
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[104:105]
	v_xor_b32_e32 v105, v102, v104
	v_xor_b32_e32 v101, v103, v104
	v_mad_u64_u32 v[102:103], s[14:15], v105, s57, 0
	v_mul_hi_u32 v70, v105, s56
	v_lshl_add_u64 v[102:103], v[70:71], 0, v[102:103]
	v_mad_u64_u32 v[108:109], s[14:15], v101, s56, 0
	v_add_co_u32_e32 v70, vcc, v102, v108
	v_mad_u64_u32 v[106:107], s[14:15], v101, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v103, v109, vcc
	s_nop 1
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[102:103], v[70:71], 0, v[106:107]
	v_mul_lo_u32 v70, s69, v102
	v_mul_lo_u32 v103, s68, v103
	v_mad_u64_u32 v[106:107], s[14:15], s68, v102, 0
	v_add3_u32 v70, v107, v103, v70
	v_sub_co_u32_e32 v103, vcc, v105, v106
	s_nop 1
	v_subb_co_u32_e64 v105, s[14:15], v101, v70, vcc
	v_sub_u32_e32 v70, v101, v70
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	v_subrev_co_u32_e32 v101, vcc, s68, v103
	v_cmp_le_u32_e64 s[14:15], s69, v105
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v106, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v103
	v_cndmask_b32_e64 v103, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v101
	v_cndmask_b32_e64 v107, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v105
	v_cndmask_b32_e64 v101, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v105, v106, v107, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v103, v101, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v101, s72, v104
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v102, v70
	v_cmp_ne_u32_e32 vcc, 0, v105
	s_nop 1
	v_cndmask_b32_e32 v70, v102, v70, vcc
	v_ashrrev_i32_e32 v102, 31, v93
	v_mov_b32_e32 v103, v102
	v_xor_b32_e32 v70, v70, v101
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[102:103]
	v_sub_u32_e32 v70, v70, v101
	v_xor_b32_e32 v108, v92, v102
	v_xad_u32 v101, v70, v85, v37
	v_xor_b32_e32 v103, v93, v102
	v_mad_u64_u32 v[92:93], s[14:15], v108, s57, 0
	v_mul_hi_u32 v70, v108, s56
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[92:93]
	v_mad_u64_u32 v[106:107], s[14:15], v103, s56, 0
	v_add_co_u32_e32 v70, vcc, v92, v106
	v_mad_u64_u32 v[104:105], s[14:15], v103, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v93, v107, vcc
	s_nop 1
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[104:105]
	v_mul_lo_u32 v70, s69, v92
	v_mul_lo_u32 v104, s68, v93
	v_mad_u64_u32 v[92:93], s[14:15], s68, v92, 0
	v_add3_u32 v70, v93, v104, v70
	v_sub_u32_e32 v93, v103, v70
	v_sub_co_u32_e32 v92, vcc, v108, v92
	s_nop 1
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, vcc
	v_subrev_co_u32_e64 v104, s[14:15], s68, v92
	v_subb_co_u32_e32 v70, vcc, v103, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v105, s[28:29], 0, v93, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v105
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v106, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v104
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v107, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v105
	v_cndmask_b32_e64 v103, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v92
	v_cndmask_b32_e64 v106, v106, v107, s[28:29]
	v_subrev_co_u32_e64 v107, s[14:15], s68, v104
	s_nop 1
	v_subbrev_co_u32_e64 v93, s[14:15], 0, v93, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v106
	s_nop 1
	v_cndmask_b32_e64 v93, v105, v93, s[14:15]
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v103, v103, v105, vcc
	v_cmp_ne_u32_e32 vcc, 0, v103
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v93, vcc
	v_cndmask_b32_e64 v93, v104, v107, s[14:15]
	v_cndmask_b32_e32 v92, v92, v93, vcc
	v_xor_b32_e32 v93, v70, v102
	v_xor_b32_e32 v70, v92, v102
	v_sub_co_u32_e32 v70, vcc, v70, v102
	s_mov_b64 s[14:15], 0x80b
	s_nop 0
	v_subb_co_u32_e32 v93, vcc, v93, v102, vcc
	v_ashrrev_i32_e32 v92, 31, v93
	v_and_b32_e32 v92, s44, v92
	v_mul_lo_u32 v93, v101, s33
	v_add3_u32 v102, v92, v70, v93
	v_lshl_add_u64 v[92:93], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf7f4
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v104, v92, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v101, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v105, v93, v101, s[22:23]
	v_ashrrev_i32_e32 v106, 31, v105
	v_mov_b32_e32 v107, v106
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[106:107]
	v_xor_b32_e32 v103, v104, v106
	v_xor_b32_e32 v101, v105, v106
	v_mad_u64_u32 v[104:105], s[14:15], v103, s57, 0
	v_mul_hi_u32 v70, v103, s56
	v_lshl_add_u64 v[104:105], v[70:71], 0, v[104:105]
	v_mad_u64_u32 v[110:111], s[14:15], v101, s56, 0
	v_add_co_u32_e32 v70, vcc, v104, v110
	v_mad_u64_u32 v[108:109], s[14:15], v101, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v105, v111, vcc
	s_nop 1
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[104:105], v[70:71], 0, v[108:109]
	v_mul_lo_u32 v70, s69, v104
	v_mul_lo_u32 v105, s68, v105
	v_mad_u64_u32 v[108:109], s[14:15], s68, v104, 0
	v_add3_u32 v70, v109, v105, v70
	v_sub_co_u32_e32 v103, vcc, v103, v108
	s_nop 1
	v_subb_co_u32_e64 v105, s[14:15], v101, v70, vcc
	v_sub_u32_e32 v70, v101, v70
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	v_subrev_co_u32_e32 v101, vcc, s68, v103
	v_cmp_le_u32_e64 s[14:15], s69, v105
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v107, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v103
	v_cndmask_b32_e64 v103, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v101
	v_cndmask_b32_e64 v108, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v105
	v_cndmask_b32_e64 v101, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v105, v107, v108, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v103, v101, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v101, s72, v106
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v104, v70
	v_cmp_ne_u32_e32 vcc, 0, v105
	s_nop 1
	v_cndmask_b32_e32 v70, v104, v70, vcc
	v_ashrrev_i32_e32 v104, 31, v93
	v_mov_b32_e32 v105, v104
	v_xor_b32_e32 v70, v70, v101
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[104:105]
	v_sub_u32_e32 v70, v70, v101
	v_xor_b32_e32 v105, v92, v104
	v_xad_u32 v101, v70, v85, v37
	v_xor_b32_e32 v103, v93, v104
	v_mad_u64_u32 v[92:93], s[14:15], v105, s57, 0
	v_mul_hi_u32 v70, v105, s56
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[92:93]
	v_mad_u64_u32 v[108:109], s[14:15], v103, s56, 0
	v_add_co_u32_e32 v70, vcc, v92, v108
	v_mad_u64_u32 v[106:107], s[14:15], v103, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v93, v109, vcc
	s_nop 1
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[106:107]
	v_mul_lo_u32 v70, s69, v92
	v_mul_lo_u32 v106, s68, v93
	v_mad_u64_u32 v[92:93], s[14:15], s68, v92, 0
	v_add3_u32 v70, v93, v106, v70
	v_sub_u32_e32 v93, v103, v70
	v_sub_co_u32_e32 v92, vcc, v105, v92
	s_nop 1
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, vcc
	v_subrev_co_u32_e64 v105, s[14:15], s68, v92
	v_subb_co_u32_e32 v70, vcc, v103, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v106, s[28:29], 0, v93, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v106
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v107, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v105
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v108, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v106
	v_cndmask_b32_e64 v103, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v92
	v_cndmask_b32_e64 v107, v107, v108, s[28:29]
	v_subrev_co_u32_e64 v108, s[14:15], s68, v105
	s_nop 1
	v_subbrev_co_u32_e64 v93, s[14:15], 0, v93, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v107
	s_nop 1
	v_cndmask_b32_e64 v93, v106, v93, s[14:15]
	v_cndmask_b32_e64 v106, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v103, v103, v106, vcc
	v_cmp_ne_u32_e32 vcc, 0, v103
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v93, vcc
	v_cndmask_b32_e64 v93, v105, v108, s[14:15]
	v_cndmask_b32_e32 v92, v92, v93, vcc
	v_xor_b32_e32 v93, v70, v104
	v_xor_b32_e32 v70, v92, v104
	v_sub_co_u32_e32 v70, vcc, v70, v104
	s_mov_b64 s[14:15], 0x80c
	s_nop 0
	v_subb_co_u32_e32 v93, vcc, v93, v104, vcc
	v_ashrrev_i32_e32 v92, 31, v93
	v_and_b32_e32 v92, s44, v92
	v_mul_lo_u32 v93, v101, s33
	v_add3_u32 v103, v92, v70, v93
	v_lshl_add_u64 v[92:93], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf7f3
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v104, v92, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v101, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v105, v93, v101, s[22:23]
	v_ashrrev_i32_e32 v106, 31, v105
	v_mov_b32_e32 v107, v106
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[106:107]
	v_xor_b32_e32 v107, v104, v106
	v_xor_b32_e32 v101, v105, v106
	v_mad_u64_u32 v[104:105], s[14:15], v107, s57, 0
	v_mul_hi_u32 v70, v107, s56
	v_lshl_add_u64 v[104:105], v[70:71], 0, v[104:105]
	v_mad_u64_u32 v[110:111], s[14:15], v101, s56, 0
	v_add_co_u32_e32 v70, vcc, v104, v110
	v_mad_u64_u32 v[108:109], s[14:15], v101, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v105, v111, vcc
	s_nop 1
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[104:105], v[70:71], 0, v[108:109]
	v_mul_lo_u32 v70, s69, v104
	v_mul_lo_u32 v105, s68, v105
	v_mad_u64_u32 v[108:109], s[14:15], s68, v104, 0
	v_add3_u32 v70, v109, v105, v70
	v_sub_co_u32_e32 v105, vcc, v107, v108
	s_nop 1
	v_subb_co_u32_e64 v107, s[14:15], v101, v70, vcc
	v_sub_u32_e32 v70, v101, v70
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	v_subrev_co_u32_e32 v101, vcc, s68, v105
	v_cmp_le_u32_e64 s[14:15], s69, v107
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v108, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v105
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v101
	v_cndmask_b32_e64 v109, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v107
	v_cndmask_b32_e64 v101, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v107, v108, v109, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v105, v101, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v101, s72, v106
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v104, v70
	v_cmp_ne_u32_e32 vcc, 0, v107
	s_nop 1
	v_cndmask_b32_e32 v70, v104, v70, vcc
	v_ashrrev_i32_e32 v104, 31, v93
	v_mov_b32_e32 v105, v104
	v_xor_b32_e32 v70, v70, v101
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[104:105]
	v_sub_u32_e32 v70, v70, v101
	v_xor_b32_e32 v110, v92, v104
	v_xad_u32 v101, v70, v85, v37
	v_xor_b32_e32 v105, v93, v104
	v_mad_u64_u32 v[92:93], s[14:15], v110, s57, 0
	v_mul_hi_u32 v70, v110, s56
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[92:93]
	v_mad_u64_u32 v[108:109], s[14:15], v105, s56, 0
	v_add_co_u32_e32 v70, vcc, v92, v108
	v_mad_u64_u32 v[106:107], s[14:15], v105, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v93, v109, vcc
	s_nop 1
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[106:107]
	v_mul_lo_u32 v70, s69, v92
	v_mul_lo_u32 v106, s68, v93
	v_mad_u64_u32 v[92:93], s[14:15], s68, v92, 0
	v_add3_u32 v70, v93, v106, v70
	v_sub_u32_e32 v93, v105, v70
	v_sub_co_u32_e32 v92, vcc, v110, v92
	s_nop 1
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, vcc
	v_subrev_co_u32_e64 v106, s[14:15], s68, v92
	v_subb_co_u32_e32 v70, vcc, v105, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v107, s[28:29], 0, v93, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v107
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v108, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v106
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v109, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v107
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v92
	v_cndmask_b32_e64 v108, v108, v109, s[28:29]
	v_subrev_co_u32_e64 v109, s[14:15], s68, v106
	s_nop 1
	v_subbrev_co_u32_e64 v93, s[14:15], 0, v93, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v108
	s_nop 1
	v_cndmask_b32_e64 v93, v107, v93, s[14:15]
	v_cndmask_b32_e64 v107, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v105, v105, v107, vcc
	v_cmp_ne_u32_e32 vcc, 0, v105
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v93, vcc
	v_cndmask_b32_e64 v93, v106, v109, s[14:15]
	v_cndmask_b32_e32 v92, v92, v93, vcc
	v_xor_b32_e32 v93, v70, v104
	v_xor_b32_e32 v70, v92, v104
	v_sub_co_u32_e32 v70, vcc, v70, v104
	s_mov_b64 s[14:15], 0x80d
	s_nop 0
	v_subb_co_u32_e32 v93, vcc, v93, v104, vcc
	v_ashrrev_i32_e32 v92, 31, v93
	v_and_b32_e32 v92, s44, v92
	v_mul_lo_u32 v93, v101, s33
	v_add3_u32 v104, v92, v70, v93
	v_lshl_add_u64 v[92:93], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf7f2
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v106, v92, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v101, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v107, v93, v101, s[22:23]
	v_ashrrev_i32_e32 v108, 31, v107
	v_mov_b32_e32 v109, v108
	v_lshl_add_u64 v[106:107], v[106:107], 0, v[108:109]
	v_xor_b32_e32 v105, v106, v108
	v_xor_b32_e32 v101, v107, v108
	v_mad_u64_u32 v[106:107], s[14:15], v105, s57, 0
	v_mul_hi_u32 v70, v105, s56
	v_lshl_add_u64 v[106:107], v[70:71], 0, v[106:107]
	v_mad_u64_u32 v[112:113], s[14:15], v101, s56, 0
	v_add_co_u32_e32 v70, vcc, v106, v112
	v_mad_u64_u32 v[110:111], s[14:15], v101, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v107, v113, vcc
	s_nop 1
	v_addc_co_u32_e32 v111, vcc, 0, v111, vcc
	v_lshl_add_u64 v[106:107], v[70:71], 0, v[110:111]
	v_mul_lo_u32 v70, s69, v106
	v_mul_lo_u32 v107, s68, v107
	v_mad_u64_u32 v[110:111], s[14:15], s68, v106, 0
	v_add3_u32 v70, v111, v107, v70
	v_sub_co_u32_e32 v105, vcc, v105, v110
	s_nop 1
	v_subb_co_u32_e64 v107, s[14:15], v101, v70, vcc
	v_sub_u32_e32 v70, v101, v70
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	v_subrev_co_u32_e32 v101, vcc, s68, v105
	v_cmp_le_u32_e64 s[14:15], s69, v107
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v109, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v105
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v101
	v_cndmask_b32_e64 v110, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v107
	v_cndmask_b32_e64 v101, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v107, v109, v110, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v105, v101, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v101, s72, v108
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v106, v70
	v_cmp_ne_u32_e32 vcc, 0, v107
	s_nop 1
	v_cndmask_b32_e32 v70, v106, v70, vcc
	v_ashrrev_i32_e32 v106, 31, v93
	v_mov_b32_e32 v107, v106
	v_xor_b32_e32 v70, v70, v101
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[106:107]
	v_sub_u32_e32 v70, v70, v101
	v_xor_b32_e32 v107, v92, v106
	v_xad_u32 v101, v70, v85, v37
	v_xor_b32_e32 v105, v93, v106
	v_mad_u64_u32 v[92:93], s[14:15], v107, s57, 0
	v_mul_hi_u32 v70, v107, s56
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[92:93]
	v_mad_u64_u32 v[110:111], s[14:15], v105, s56, 0
	v_add_co_u32_e32 v70, vcc, v92, v110
	v_mad_u64_u32 v[108:109], s[14:15], v105, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v93, v111, vcc
	s_nop 1
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[108:109]
	v_mul_lo_u32 v70, s69, v92
	v_mul_lo_u32 v108, s68, v93
	v_mad_u64_u32 v[92:93], s[14:15], s68, v92, 0
	v_add3_u32 v70, v93, v108, v70
	v_sub_u32_e32 v93, v105, v70
	v_sub_co_u32_e32 v92, vcc, v107, v92
	s_nop 1
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, vcc
	v_subrev_co_u32_e64 v107, s[14:15], s68, v92
	v_subb_co_u32_e32 v70, vcc, v105, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v108, s[28:29], 0, v93, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v108
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v109, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v107
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v110, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v108
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v92
	v_cndmask_b32_e64 v109, v109, v110, s[28:29]
	v_subrev_co_u32_e64 v110, s[14:15], s68, v107
	s_nop 1
	v_subbrev_co_u32_e64 v93, s[14:15], 0, v93, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v109
	s_nop 1
	v_cndmask_b32_e64 v93, v108, v93, s[14:15]
	v_cndmask_b32_e64 v108, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v105, v105, v108, vcc
	v_cmp_ne_u32_e32 vcc, 0, v105
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v93, vcc
	v_cndmask_b32_e64 v93, v107, v110, s[14:15]
	v_cndmask_b32_e32 v92, v92, v93, vcc
	v_xor_b32_e32 v93, v70, v106
	v_xor_b32_e32 v70, v92, v106
	v_sub_co_u32_e32 v70, vcc, v70, v106
	s_mov_b64 s[14:15], 0x80e
	s_nop 0
	v_subb_co_u32_e32 v93, vcc, v93, v106, vcc
	v_ashrrev_i32_e32 v92, 31, v93
	v_and_b32_e32 v92, s44, v92
	v_mul_lo_u32 v93, v101, s33
	v_add3_u32 v108, v92, v70, v93
	v_lshl_add_u64 v[92:93], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf7f1
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v106, v92, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v101, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v107, v93, v101, s[22:23]
	v_ashrrev_i32_e32 v110, 31, v107
	v_mov_b32_e32 v111, v110
	v_lshl_add_u64 v[106:107], v[106:107], 0, v[110:111]
	v_xor_b32_e32 v105, v106, v110
	v_xor_b32_e32 v101, v107, v110
	v_mad_u64_u32 v[106:107], s[14:15], v105, s57, 0
	v_mul_hi_u32 v70, v105, s56
	v_lshl_add_u64 v[106:107], v[70:71], 0, v[106:107]
	v_mad_u64_u32 v[114:115], s[14:15], v101, s56, 0
	v_add_co_u32_e32 v70, vcc, v106, v114
	v_mad_u64_u32 v[112:113], s[14:15], v101, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v107, v115, vcc
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	v_lshl_add_u64 v[106:107], v[70:71], 0, v[112:113]
	v_mul_lo_u32 v70, s69, v106
	v_mul_lo_u32 v107, s68, v107
	v_mad_u64_u32 v[112:113], s[14:15], s68, v106, 0
	v_add3_u32 v70, v113, v107, v70
	v_sub_co_u32_e32 v105, vcc, v105, v112
	s_nop 1
	v_subb_co_u32_e64 v107, s[14:15], v101, v70, vcc
	v_sub_u32_e32 v70, v101, v70
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	v_subrev_co_u32_e32 v101, vcc, s68, v105
	v_cmp_le_u32_e64 s[14:15], s69, v107
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v109, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v105
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v101
	v_cndmask_b32_e64 v111, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v107
	v_cndmask_b32_e64 v101, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v107, v109, v111, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v105, v101, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v101, s72, v110
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v106, v70
	v_cmp_ne_u32_e32 vcc, 0, v107
	s_nop 1
	v_cndmask_b32_e32 v70, v106, v70, vcc
	v_ashrrev_i32_e32 v106, 31, v93
	v_mov_b32_e32 v107, v106
	v_xor_b32_e32 v70, v70, v101
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[106:107]
	v_sub_u32_e32 v70, v70, v101
	v_xor_b32_e32 v107, v92, v106
	v_xad_u32 v101, v70, v85, v37
	v_xor_b32_e32 v105, v93, v106
	v_mad_u64_u32 v[92:93], s[14:15], v107, s57, 0
	v_mul_hi_u32 v70, v107, s56
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[92:93]
	v_mad_u64_u32 v[112:113], s[14:15], v105, s56, 0
	v_add_co_u32_e32 v70, vcc, v92, v112
	v_mad_u64_u32 v[110:111], s[14:15], v105, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v93, v113, vcc
	s_nop 1
	v_addc_co_u32_e32 v111, vcc, 0, v111, vcc
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[110:111]
	v_mul_lo_u32 v70, s69, v92
	v_mul_lo_u32 v109, s68, v93
	v_mad_u64_u32 v[92:93], s[14:15], s68, v92, 0
	v_add3_u32 v70, v93, v109, v70
	v_sub_u32_e32 v93, v105, v70
	v_sub_co_u32_e32 v92, vcc, v107, v92
	s_nop 1
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, vcc
	v_subrev_co_u32_e64 v107, s[14:15], s68, v92
	v_subb_co_u32_e32 v70, vcc, v105, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v109, s[28:29], 0, v93, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v109
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v110, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v107
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v111, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v109
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v92
	v_cndmask_b32_e64 v110, v110, v111, s[28:29]
	v_subrev_co_u32_e64 v111, s[14:15], s68, v107
	s_nop 1
	v_subbrev_co_u32_e64 v93, s[14:15], 0, v93, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v110
	s_nop 1
	v_cndmask_b32_e64 v93, v109, v93, s[14:15]
	v_cndmask_b32_e64 v109, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v105, v105, v109, vcc
	v_cmp_ne_u32_e32 vcc, 0, v105
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v93, vcc
	v_cndmask_b32_e64 v93, v107, v111, s[14:15]
	v_cndmask_b32_e32 v92, v92, v93, vcc
	v_xor_b32_e32 v93, v70, v106
	v_xor_b32_e32 v70, v92, v106
	v_sub_co_u32_e32 v70, vcc, v70, v106
	s_mov_b64 s[14:15], 0x80f
	s_nop 0
	v_subb_co_u32_e32 v93, vcc, v93, v106, vcc
	v_ashrrev_i32_e32 v92, 31, v93
	v_and_b32_e32 v92, s44, v92
	v_mul_lo_u32 v93, v101, s33
	v_add3_u32 v109, v92, v70, v93
	v_lshl_add_u64 v[92:93], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf7f0
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v106, v92, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v101, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v107, v93, v101, s[22:23]
	v_ashrrev_i32_e32 v110, 31, v107
	v_mov_b32_e32 v111, v110
	v_lshl_add_u64 v[106:107], v[106:107], 0, v[110:111]
	v_xor_b32_e32 v105, v106, v110
	v_xor_b32_e32 v101, v107, v110
	v_mad_u64_u32 v[106:107], s[14:15], v105, s57, 0
	v_mul_hi_u32 v70, v105, s56
	v_lshl_add_u64 v[106:107], v[70:71], 0, v[106:107]
	v_mad_u64_u32 v[114:115], s[14:15], v101, s56, 0
	v_add_co_u32_e32 v70, vcc, v106, v114
	v_mad_u64_u32 v[112:113], s[14:15], v101, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v107, v115, vcc
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	v_lshl_add_u64 v[106:107], v[70:71], 0, v[112:113]
	v_mul_lo_u32 v70, s69, v106
	v_mul_lo_u32 v107, s68, v107
	v_mad_u64_u32 v[112:113], s[14:15], s68, v106, 0
	v_add3_u32 v70, v113, v107, v70
	v_sub_co_u32_e32 v105, vcc, v105, v112
	s_nop 1
	v_subb_co_u32_e64 v107, s[14:15], v101, v70, vcc
	v_sub_u32_e32 v70, v101, v70
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	v_subrev_co_u32_e32 v101, vcc, s68, v105
	v_cmp_le_u32_e64 s[14:15], s69, v107
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v111, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v105
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v101
	v_cndmask_b32_e64 v112, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v107
	v_cndmask_b32_e64 v101, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v107, v111, v112, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v105, v101, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v101, s72, v110
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v106, v70
	v_cmp_ne_u32_e32 vcc, 0, v107
	s_nop 1
	v_cndmask_b32_e32 v70, v106, v70, vcc
	v_ashrrev_i32_e32 v106, 31, v93
	v_mov_b32_e32 v107, v106
	v_xor_b32_e32 v70, v70, v101
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[106:107]
	v_sub_u32_e32 v70, v70, v101
	v_xor_b32_e32 v105, v92, v106
	v_xad_u32 v101, v70, v85, v37
	v_xor_b32_e32 v85, v93, v106
	v_mad_u64_u32 v[92:93], s[14:15], v105, s57, 0
	v_mul_hi_u32 v70, v105, s56
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[92:93]
	v_mad_u64_u32 v[112:113], s[14:15], v85, s56, 0
	v_add_co_u32_e32 v70, vcc, v92, v112
	v_mad_u64_u32 v[110:111], s[14:15], v85, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v93, v113, vcc
	s_nop 1
	v_addc_co_u32_e32 v111, vcc, 0, v111, vcc
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[110:111]
	v_mul_lo_u32 v70, s69, v92
	v_mul_lo_u32 v107, s68, v93
	v_mad_u64_u32 v[92:93], s[14:15], s68, v92, 0
	v_add3_u32 v70, v93, v107, v70
	v_sub_u32_e32 v93, v85, v70
	v_sub_co_u32_e32 v92, vcc, v105, v92
	s_nop 1
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, vcc
	v_subrev_co_u32_e64 v105, s[14:15], s68, v92
	v_subb_co_u32_e32 v70, vcc, v85, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v107, s[22:23], 0, v93, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s69, v107
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v110, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s68, v105
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v111, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s69, v107
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v92
	v_cndmask_b32_e64 v110, v110, v111, s[22:23]
	v_subrev_co_u32_e64 v111, s[14:15], s68, v105
	s_nop 1
	v_subbrev_co_u32_e64 v93, s[14:15], 0, v93, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v110
	s_nop 1
	v_cndmask_b32_e64 v93, v107, v93, s[14:15]
	v_cndmask_b32_e64 v107, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v85, v85, v107, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	v_cndmask_b32_e64 v85, v105, v111, s[14:15]
	s_mov_b64 s[14:15], 0x1400
	v_cndmask_b32_e32 v70, v70, v93, vcc
	v_cndmask_b32_e32 v85, v92, v85, vcc
	v_xor_b32_e32 v92, v70, v106
	v_xor_b32_e32 v70, v85, v106
	v_sub_co_u32_e32 v70, vcc, v70, v106
	v_and_b32_e32 v107, 0xffff, v65
	s_nop 0
	v_subb_co_u32_e32 v85, vcc, v92, v106, vcc
	v_ashrrev_i32_e32 v85, 31, v85
	v_and_b32_e32 v85, s44, v85
	v_mul_lo_u32 v92, v101, s33
	v_add3_u32 v70, v85, v70, v92
	v_lshl_add_u64 v[92:93], v[90:91], 0, s[14:15]
	s_movk_i32 s14, 0xebff
	buffer_load_ubyte v101, v94, s[36:39], 0 offen
	buffer_load_ubyte v105, v95, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v102, v102, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v106, v103, s[36:39], 0 offen
	buffer_load_ubyte v65, v104, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v103, v108, s[36:39], 0 offen
	buffer_load_ubyte v85, v109, s[36:39], 0 offen
	buffer_load_ubyte v104, v70, s[36:39], 0 offen
	v_sub_co_u32_e32 v70, vcc, s14, v90
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[132:135], v[16:19], v[0:3], v33, v107 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_subb_co_u32_e32 v90, vcc, -1, v91, vcc
	v_cmp_gt_i32_e32 vcc, 0, v93
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[124:127], v[16:19], v[20:23], v33, v107 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v91, v93, v90, vcc
	v_ashrrev_i32_e32 v94, 31, v91
	v_cndmask_b32_e32 v90, v92, v70, vcc
	v_mov_b32_e32 v95, v94
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[94:95]
	v_xor_b32_e32 v112, v90, v94
	v_xor_b32_e32 v95, v91, v94
	v_mad_u64_u32 v[90:91], s[14:15], v112, s57, 0
	v_mul_hi_u32 v70, v112, s56
	v_lshl_add_u64 v[90:91], v[70:71], 0, v[90:91]
	v_mad_u64_u32 v[110:111], s[14:15], v95, s56, 0
	v_add_co_u32_e32 v70, vcc, v90, v110
	v_mad_u64_u32 v[108:109], s[14:15], v95, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v91, v111, vcc
	s_nop 1
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[90:91], v[70:71], 0, v[108:109]
	v_mul_lo_u32 v70, s69, v90
	v_mul_lo_u32 v91, s68, v91
	v_mad_u64_u32 v[108:109], s[14:15], s68, v90, 0
	v_add3_u32 v70, v109, v91, v70
	v_sub_co_u32_e32 v91, vcc, v112, v108
	s_nop 1
	v_subb_co_u32_e64 v108, s[14:15], v95, v70, vcc
	v_sub_u32_e32 v70, v95, v70
	v_cmp_le_u32_e64 s[14:15], s69, v108
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	s_nop 0
	v_cndmask_b32_e64 v109, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v91
	v_subrev_co_u32_e32 v91, vcc, s68, v91
	s_nop 0
	v_cndmask_b32_e64 v110, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cmp_eq_u32_e64 s[14:15], s69, v108
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v91
	v_cndmask_b32_e64 v108, v109, v110, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v70, v95, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_nop 1
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v90, v70
	v_cmp_ne_u32_e32 vcc, 0, v108
	s_nop 1
	v_cndmask_b32_e32 v70, v90, v70, vcc
	v_xor_b32_e32 v90, s72, v94
	v_xor_b32_e32 v70, v70, v90
	v_sub_u32_e32 v70, v70, v90
	v_ashrrev_i32_e32 v90, 31, v93
	v_mov_b32_e32 v91, v90
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[90:91]
	v_xor_b32_e32 v111, v92, v90
	v_xad_u32 v110, v70, v90, v37
	v_xor_b32_e32 v91, v93, v90
	v_mad_u64_u32 v[92:93], s[14:15], v111, s57, 0
	v_mul_hi_u32 v70, v111, s56
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[92:93]
	v_mad_u64_u32 v[108:109], s[14:15], v91, s56, 0
	v_add_co_u32_e32 v70, vcc, v92, v108
	v_mad_u64_u32 v[94:95], s[14:15], v91, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v93, v109, vcc
	s_nop 1
	v_addc_co_u32_e32 v95, vcc, 0, v95, vcc
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[94:95]
	v_mul_lo_u32 v70, s69, v92
	v_mul_lo_u32 v94, s68, v93
	v_mad_u64_u32 v[92:93], s[14:15], s68, v92, 0
	v_add3_u32 v70, v93, v94, v70
	v_sub_u32_e32 v93, v91, v70
	v_sub_co_u32_e32 v92, vcc, v111, v92
	s_nop 1
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, vcc
	v_subrev_co_u32_e64 v94, s[14:15], s68, v92
	v_subb_co_u32_e32 v70, vcc, v91, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v95, s[22:23], 0, v93, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s69, v95
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v108, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s68, v94
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v109, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s69, v95
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v92
	v_cndmask_b32_e64 v108, v108, v109, s[22:23]
	v_subrev_co_u32_e64 v109, s[14:15], s68, v94
	s_nop 1
	v_subbrev_co_u32_e64 v93, s[14:15], 0, v93, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v108
	s_nop 1
	v_cndmask_b32_e64 v93, v95, v93, s[14:15]
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v91, v91, v95, vcc
	v_cmp_ne_u32_e32 vcc, 0, v91
	v_cndmask_b32_e64 v91, v94, v109, s[14:15]
	s_mov_b64 s[14:15], 0xc01
	v_cndmask_b32_e32 v91, v92, v91, vcc
	v_cndmask_b32_e32 v70, v70, v93, vcc
	v_xor_b32_e32 v91, v91, v90
	v_xor_b32_e32 v70, v70, v90
	v_sub_co_u32_e32 v91, vcc, v91, v90
	v_lshl_add_u64 v[94:95], v[88:89], 0, s[14:15]
	s_nop 0
	v_subb_co_u32_e32 v70, vcc, v70, v90, vcc
	v_ashrrev_i32_e32 v70, 31, v70
	v_and_b32_e32 v70, s44, v70
	v_mul_lo_u32 v90, v110, s33
	s_movk_i32 s14, 0xf3fe
	v_add3_u32 v91, v70, v91, v90
	v_lshl_add_u64 v[92:93], v[88:89], 0, s[92:93]
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cmp_gt_i32_e64 s[22:23], 0, v93
	s_nop 0
	v_subb_co_u32_e32 v90, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v109, v95, v90, s[22:23]
	v_ashrrev_i32_e32 v110, 31, v109
	v_cndmask_b32_e64 v108, v94, v70, s[22:23]
	v_mov_b32_e32 v111, v110
	v_lshl_add_u64 v[108:109], v[108:109], 0, v[110:111]
	v_xor_b32_e32 v92, v108, v110
	v_xor_b32_e32 v90, v109, v110
	v_mad_u64_u32 v[108:109], s[14:15], v92, s57, 0
	v_mul_hi_u32 v70, v92, s56
	v_lshl_add_u64 v[108:109], v[70:71], 0, v[108:109]
	v_mad_u64_u32 v[114:115], s[14:15], v90, s56, 0
	v_add_co_u32_e32 v70, vcc, v108, v114
	v_mad_u64_u32 v[112:113], s[14:15], v90, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v109, v115, vcc
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	v_lshl_add_u64 v[108:109], v[70:71], 0, v[112:113]
	v_mul_lo_u32 v70, s69, v108
	v_mul_lo_u32 v109, s68, v109
	v_mad_u64_u32 v[112:113], s[14:15], s68, v108, 0
	v_add3_u32 v70, v113, v109, v70
	v_sub_co_u32_e32 v92, vcc, v92, v112
	s_nop 1
	v_subb_co_u32_e64 v109, s[14:15], v90, v70, vcc
	v_sub_u32_e32 v70, v90, v70
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	v_subrev_co_u32_e32 v90, vcc, s68, v92
	v_cmp_le_u32_e64 s[14:15], s69, v109
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v111, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v92
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v90
	v_cndmask_b32_e64 v112, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v109
	v_cndmask_b32_e64 v90, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v109, v111, v112, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v92, v90, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v90, s72, v110
	v_ashrrev_i32_e32 v92, 31, v95
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v108, v70
	v_cmp_ne_u32_e32 vcc, 0, v109
	s_nop 1
	v_cndmask_b32_e32 v70, v108, v70, vcc
	v_xor_b32_e32 v70, v70, v90
	v_sub_u32_e32 v70, v70, v90
	v_ashrrev_i32_e32 v90, 31, v93
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[94:95], v[94:95], 0, v[92:93]
	v_xor_b32_e32 v113, v94, v92
	v_xad_u32 v112, v70, v90, v37
	v_xor_b32_e32 v93, v95, v92
	v_mad_u64_u32 v[94:95], s[14:15], v113, s57, 0
	v_mul_hi_u32 v70, v113, s56
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[94:95]
	v_mad_u64_u32 v[110:111], s[14:15], v93, s56, 0
	v_add_co_u32_e32 v70, vcc, v94, v110
	v_mad_u64_u32 v[108:109], s[14:15], v93, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v95, v111, vcc
	s_nop 1
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[108:109]
	v_mul_lo_u32 v70, s69, v94
	v_mul_lo_u32 v108, s68, v95
	v_mad_u64_u32 v[94:95], s[14:15], s68, v94, 0
	v_add3_u32 v70, v95, v108, v70
	v_sub_u32_e32 v95, v93, v70
	v_sub_co_u32_e32 v94, vcc, v113, v94
	s_nop 1
	v_subb_co_u32_e64 v95, s[14:15], v95, v35, vcc
	v_subrev_co_u32_e64 v108, s[14:15], s68, v94
	v_subb_co_u32_e32 v70, vcc, v93, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v109, s[28:29], 0, v95, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v109
	v_subb_co_u32_e64 v95, s[14:15], v95, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v110, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v108
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v111, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v109
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v94
	v_cndmask_b32_e64 v110, v110, v111, s[28:29]
	v_subrev_co_u32_e64 v111, s[14:15], s68, v108
	s_nop 1
	v_subbrev_co_u32_e64 v95, s[14:15], 0, v95, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v110
	s_nop 1
	v_cndmask_b32_e64 v95, v109, v95, s[14:15]
	v_cndmask_b32_e64 v109, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v93, v93, v109, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	v_cndmask_b32_e64 v93, v108, v111, s[14:15]
	s_mov_b64 s[14:15], 0xc02
	v_cndmask_b32_e32 v70, v70, v95, vcc
	v_cndmask_b32_e32 v93, v94, v93, vcc
	v_xor_b32_e32 v94, v70, v92
	v_xor_b32_e32 v70, v93, v92
	v_sub_co_u32_e32 v70, vcc, v70, v92
	s_nop 1
	v_subb_co_u32_e32 v93, vcc, v94, v92, vcc
	v_ashrrev_i32_e32 v92, 31, v93
	v_and_b32_e32 v92, s44, v92
	v_mul_lo_u32 v93, v112, s33
	v_lshl_add_u64 v[94:95], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf3fd
	v_add3_u32 v92, v92, v70, v93
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v108, v94, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v93, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v109, v95, v93, s[22:23]
	v_ashrrev_i32_e32 v110, 31, v109
	v_mov_b32_e32 v111, v110
	v_lshl_add_u64 v[108:109], v[108:109], 0, v[110:111]
	v_xor_b32_e32 v111, v108, v110
	v_xor_b32_e32 v93, v109, v110
	v_mad_u64_u32 v[108:109], s[14:15], v111, s57, 0
	v_mul_hi_u32 v70, v111, s56
	v_lshl_add_u64 v[108:109], v[70:71], 0, v[108:109]
	v_mad_u64_u32 v[114:115], s[14:15], v93, s56, 0
	v_add_co_u32_e32 v70, vcc, v108, v114
	v_mad_u64_u32 v[112:113], s[14:15], v93, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v109, v115, vcc
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	v_lshl_add_u64 v[108:109], v[70:71], 0, v[112:113]
	v_mul_lo_u32 v70, s69, v108
	v_mul_lo_u32 v109, s68, v109
	v_mad_u64_u32 v[112:113], s[14:15], s68, v108, 0
	v_add3_u32 v70, v113, v109, v70
	v_sub_co_u32_e32 v109, vcc, v111, v112
	s_nop 1
	v_subb_co_u32_e64 v111, s[14:15], v93, v70, vcc
	v_sub_u32_e32 v70, v93, v70
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	v_subrev_co_u32_e32 v93, vcc, s68, v109
	v_cmp_le_u32_e64 s[14:15], s69, v111
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v112, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v109
	v_cndmask_b32_e64 v109, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v93
	v_cndmask_b32_e64 v113, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v111
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v111, v112, v113, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v109, v93, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v93, s72, v110
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v108, v70
	v_cmp_ne_u32_e32 vcc, 0, v111
	s_nop 1
	v_cndmask_b32_e32 v70, v108, v70, vcc
	v_ashrrev_i32_e32 v108, 31, v95
	v_mov_b32_e32 v109, v108
	v_xor_b32_e32 v70, v70, v93
	v_lshl_add_u64 v[94:95], v[94:95], 0, v[108:109]
	v_sub_u32_e32 v70, v70, v93
	v_xor_b32_e32 v109, v94, v108
	v_xad_u32 v114, v70, v90, v37
	v_xor_b32_e32 v93, v95, v108
	v_mad_u64_u32 v[94:95], s[14:15], v109, s57, 0
	v_mul_hi_u32 v70, v109, s56
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[94:95]
	v_mad_u64_u32 v[112:113], s[14:15], v93, s56, 0
	v_add_co_u32_e32 v70, vcc, v94, v112
	v_mad_u64_u32 v[110:111], s[14:15], v93, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v95, v113, vcc
	s_nop 1
	v_addc_co_u32_e32 v111, vcc, 0, v111, vcc
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[110:111]
	v_mul_lo_u32 v70, s69, v94
	v_mul_lo_u32 v110, s68, v95
	v_mad_u64_u32 v[94:95], s[14:15], s68, v94, 0
	v_add3_u32 v70, v95, v110, v70
	v_sub_u32_e32 v95, v93, v70
	v_sub_co_u32_e32 v94, vcc, v109, v94
	s_nop 1
	v_subb_co_u32_e64 v95, s[14:15], v95, v35, vcc
	v_subrev_co_u32_e64 v109, s[14:15], s68, v94
	v_subb_co_u32_e32 v70, vcc, v93, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v110, s[28:29], 0, v95, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v110
	v_subb_co_u32_e64 v95, s[14:15], v95, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v111, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v109
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v112, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v110
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v94
	v_cndmask_b32_e64 v111, v111, v112, s[28:29]
	v_subrev_co_u32_e64 v112, s[14:15], s68, v109
	s_nop 1
	v_subbrev_co_u32_e64 v95, s[14:15], 0, v95, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v111
	s_nop 1
	v_cndmask_b32_e64 v95, v110, v95, s[14:15]
	v_cndmask_b32_e64 v110, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v93, v93, v110, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	v_cndmask_b32_e64 v93, v109, v112, s[14:15]
	s_mov_b64 s[14:15], 0xc03
	v_cndmask_b32_e32 v70, v70, v95, vcc
	v_cndmask_b32_e32 v93, v94, v93, vcc
	v_xor_b32_e32 v94, v70, v108
	v_xor_b32_e32 v70, v93, v108
	v_sub_co_u32_e32 v70, vcc, v70, v108
	s_nop 1
	v_subb_co_u32_e32 v93, vcc, v94, v108, vcc
	v_ashrrev_i32_e32 v93, 31, v93
	v_and_b32_e32 v93, s44, v93
	v_mul_lo_u32 v94, v114, s33
	v_add3_u32 v93, v93, v70, v94
	v_lshl_add_u64 v[94:95], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf3fc
	v_sub_co_u32_e32 v70, vcc, s14, v88
	s_nop 1
	v_subb_co_u32_e32 v108, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v109, v95, v108, s[22:23]
	v_ashrrev_i32_e32 v110, 31, v109
	v_cndmask_b32_e64 v108, v94, v70, s[22:23]
	v_mov_b32_e32 v111, v110
	v_lshl_add_u64 v[108:109], v[108:109], 0, v[110:111]
	v_xor_b32_e32 v116, v108, v110
	v_xor_b32_e32 v111, v109, v110
	v_mad_u64_u32 v[108:109], s[14:15], v116, s57, 0
	v_mul_hi_u32 v70, v116, s56
	v_lshl_add_u64 v[108:109], v[70:71], 0, v[108:109]
	v_mad_u64_u32 v[114:115], s[14:15], v111, s56, 0
	v_add_co_u32_e32 v70, vcc, v108, v114
	v_mad_u64_u32 v[112:113], s[14:15], v111, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v109, v115, vcc
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	v_lshl_add_u64 v[108:109], v[70:71], 0, v[112:113]
	v_mul_lo_u32 v70, s69, v108
	v_mul_lo_u32 v109, s68, v109
	v_mad_u64_u32 v[112:113], s[14:15], s68, v108, 0
	v_add3_u32 v70, v113, v109, v70
	v_sub_co_u32_e32 v109, vcc, v116, v112
	s_nop 1
	v_subb_co_u32_e64 v112, s[14:15], v111, v70, vcc
	v_sub_u32_e32 v70, v111, v70
	v_cmp_le_u32_e64 s[14:15], s69, v112
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	s_nop 0
	v_cndmask_b32_e64 v113, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v109
	v_subrev_co_u32_e32 v109, vcc, s68, v109
	s_nop 0
	v_cndmask_b32_e64 v114, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cmp_eq_u32_e64 s[14:15], s69, v112
	s_nop 0
	v_cndmask_b32_e64 v111, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v109
	v_cndmask_b32_e64 v112, v113, v114, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v109, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v70, v111, v109, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_nop 1
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v108, v70
	v_cmp_ne_u32_e32 vcc, 0, v112
	s_nop 1
	v_cndmask_b32_e32 v70, v108, v70, vcc
	v_xor_b32_e32 v108, s72, v110
	v_xor_b32_e32 v70, v70, v108
	v_sub_u32_e32 v70, v70, v108
	v_ashrrev_i32_e32 v108, 31, v95
	v_mov_b32_e32 v109, v108
	v_lshl_add_u64 v[94:95], v[94:95], 0, v[108:109]
	v_xor_b32_e32 v115, v94, v108
	v_xad_u32 v114, v70, v90, v37
	v_xor_b32_e32 v109, v95, v108
	v_mad_u64_u32 v[94:95], s[14:15], v115, s57, 0
	v_mul_hi_u32 v70, v115, s56
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[94:95]
	v_mad_u64_u32 v[112:113], s[14:15], v109, s56, 0
	v_add_co_u32_e32 v70, vcc, v94, v112
	v_mad_u64_u32 v[110:111], s[14:15], v109, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v95, v113, vcc
	s_nop 1
	v_addc_co_u32_e32 v111, vcc, 0, v111, vcc
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[110:111]
	v_mul_lo_u32 v70, s69, v94
	v_mul_lo_u32 v110, s68, v95
	v_mad_u64_u32 v[94:95], s[14:15], s68, v94, 0
	v_add3_u32 v70, v95, v110, v70
	v_sub_u32_e32 v95, v109, v70
	v_sub_co_u32_e32 v94, vcc, v115, v94
	s_nop 1
	v_subb_co_u32_e64 v95, s[14:15], v95, v35, vcc
	v_subrev_co_u32_e64 v110, s[14:15], s68, v94
	v_subb_co_u32_e32 v70, vcc, v109, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v111, s[28:29], 0, v95, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v111
	v_subb_co_u32_e64 v95, s[14:15], v95, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v112, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v110
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v113, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v111
	v_cndmask_b32_e64 v109, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v94
	v_cndmask_b32_e64 v112, v112, v113, s[28:29]
	v_subrev_co_u32_e64 v113, s[14:15], s68, v110
	s_nop 1
	v_subbrev_co_u32_e64 v95, s[14:15], 0, v95, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v112
	s_nop 1
	v_cndmask_b32_e64 v95, v111, v95, s[14:15]
	v_cndmask_b32_e64 v111, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v109, v109, v111, vcc
	v_cmp_ne_u32_e32 vcc, 0, v109
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v95, vcc
	v_cndmask_b32_e64 v95, v110, v113, s[14:15]
	v_cndmask_b32_e32 v94, v94, v95, vcc
	v_xor_b32_e32 v95, v70, v108
	v_xor_b32_e32 v70, v94, v108
	v_sub_co_u32_e32 v70, vcc, v70, v108
	s_mov_b64 s[14:15], 0xc04
	s_nop 0
	v_subb_co_u32_e32 v95, vcc, v95, v108, vcc
	v_ashrrev_i32_e32 v94, 31, v95
	v_and_b32_e32 v94, s44, v94
	v_mul_lo_u32 v95, v114, s33
	v_lshl_add_u64 v[108:109], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf3fb
	v_add3_u32 v94, v94, v70, v95
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v110, v108, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v95, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v111, v109, v95, s[22:23]
	v_ashrrev_i32_e32 v112, 31, v111
	v_mov_b32_e32 v113, v112
	v_lshl_add_u64 v[110:111], v[110:111], 0, v[112:113]
	v_xor_b32_e32 v113, v110, v112
	v_xor_b32_e32 v95, v111, v112
	v_mad_u64_u32 v[110:111], s[14:15], v113, s57, 0
	v_mul_hi_u32 v70, v113, s56
	v_lshl_add_u64 v[110:111], v[70:71], 0, v[110:111]
	v_mad_u64_u32 v[116:117], s[14:15], v95, s56, 0
	v_add_co_u32_e32 v70, vcc, v110, v116
	v_mad_u64_u32 v[114:115], s[14:15], v95, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v111, v117, vcc
	s_nop 1
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	v_lshl_add_u64 v[110:111], v[70:71], 0, v[114:115]
	v_mul_lo_u32 v70, s69, v110
	v_mul_lo_u32 v111, s68, v111
	v_mad_u64_u32 v[114:115], s[14:15], s68, v110, 0
	v_add3_u32 v70, v115, v111, v70
	v_sub_co_u32_e32 v111, vcc, v113, v114
	s_nop 1
	v_subb_co_u32_e64 v113, s[14:15], v95, v70, vcc
	v_sub_u32_e32 v70, v95, v70
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	v_subrev_co_u32_e32 v95, vcc, s68, v111
	v_cmp_le_u32_e64 s[14:15], s69, v113
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v114, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v111
	v_cndmask_b32_e64 v111, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v95
	v_cndmask_b32_e64 v115, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v113
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v113, v114, v115, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v111, v95, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v95, s72, v112
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v110, v70
	v_cmp_ne_u32_e32 vcc, 0, v113
	s_nop 1
	v_cndmask_b32_e32 v70, v110, v70, vcc
	v_ashrrev_i32_e32 v110, 31, v109
	v_mov_b32_e32 v111, v110
	v_xor_b32_e32 v70, v70, v95
	v_lshl_add_u64 v[108:109], v[108:109], 0, v[110:111]
	v_sub_u32_e32 v70, v70, v95
	v_xor_b32_e32 v111, v108, v110
	v_xad_u32 v116, v70, v90, v37
	v_xor_b32_e32 v95, v109, v110
	v_mad_u64_u32 v[108:109], s[14:15], v111, s57, 0
	v_mul_hi_u32 v70, v111, s56
	v_lshl_add_u64 v[108:109], v[70:71], 0, v[108:109]
	v_mad_u64_u32 v[114:115], s[14:15], v95, s56, 0
	v_add_co_u32_e32 v70, vcc, v108, v114
	v_mad_u64_u32 v[112:113], s[14:15], v95, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v109, v115, vcc
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	v_lshl_add_u64 v[108:109], v[70:71], 0, v[112:113]
	v_mul_lo_u32 v70, s69, v108
	v_mul_lo_u32 v112, s68, v109
	v_mad_u64_u32 v[108:109], s[14:15], s68, v108, 0
	v_add3_u32 v70, v109, v112, v70
	v_sub_u32_e32 v109, v95, v70
	v_sub_co_u32_e32 v108, vcc, v111, v108
	s_nop 1
	v_subb_co_u32_e64 v109, s[14:15], v109, v35, vcc
	v_subrev_co_u32_e64 v111, s[14:15], s68, v108
	v_subb_co_u32_e32 v70, vcc, v95, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v112, s[28:29], 0, v109, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v112
	v_subb_co_u32_e64 v109, s[14:15], v109, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v113, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v111
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v114, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v112
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v108
	v_cndmask_b32_e64 v113, v113, v114, s[28:29]
	v_subrev_co_u32_e64 v114, s[14:15], s68, v111
	s_nop 1
	v_subbrev_co_u32_e64 v109, s[14:15], 0, v109, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v113
	s_nop 1
	v_cndmask_b32_e64 v109, v112, v109, s[14:15]
	v_cndmask_b32_e64 v112, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v95, v95, v112, vcc
	v_cmp_ne_u32_e32 vcc, 0, v95
	v_cndmask_b32_e64 v95, v111, v114, s[14:15]
	s_mov_b64 s[14:15], 0xc05
	v_cndmask_b32_e32 v70, v70, v109, vcc
	v_cndmask_b32_e32 v95, v108, v95, vcc
	v_xor_b32_e32 v108, v70, v110
	v_xor_b32_e32 v70, v95, v110
	v_sub_co_u32_e32 v70, vcc, v70, v110
	s_nop 1
	v_subb_co_u32_e32 v95, vcc, v108, v110, vcc
	v_ashrrev_i32_e32 v95, 31, v95
	v_and_b32_e32 v95, s44, v95
	v_mul_lo_u32 v108, v116, s33
	v_add3_u32 v95, v95, v70, v108
	v_lshl_add_u64 v[108:109], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf3fa
	v_sub_co_u32_e32 v70, vcc, s14, v88
	s_nop 1
	v_subb_co_u32_e32 v110, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v111, v109, v110, s[22:23]
	v_ashrrev_i32_e32 v112, 31, v111
	v_cndmask_b32_e64 v110, v108, v70, s[22:23]
	v_mov_b32_e32 v113, v112
	v_lshl_add_u64 v[110:111], v[110:111], 0, v[112:113]
	v_xor_b32_e32 v118, v110, v112
	v_xor_b32_e32 v113, v111, v112
	v_mad_u64_u32 v[110:111], s[14:15], v118, s57, 0
	v_mul_hi_u32 v70, v118, s56
	v_lshl_add_u64 v[110:111], v[70:71], 0, v[110:111]
	v_mad_u64_u32 v[116:117], s[14:15], v113, s56, 0
	v_add_co_u32_e32 v70, vcc, v110, v116
	v_mad_u64_u32 v[114:115], s[14:15], v113, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v111, v117, vcc
	s_nop 1
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	v_lshl_add_u64 v[110:111], v[70:71], 0, v[114:115]
	v_mul_lo_u32 v70, s69, v110
	v_mul_lo_u32 v111, s68, v111
	v_mad_u64_u32 v[114:115], s[14:15], s68, v110, 0
	v_add3_u32 v70, v115, v111, v70
	v_sub_co_u32_e32 v111, vcc, v118, v114
	s_nop 1
	v_subb_co_u32_e64 v114, s[14:15], v113, v70, vcc
	v_sub_u32_e32 v70, v113, v70
	v_cmp_le_u32_e64 s[14:15], s69, v114
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	s_nop 0
	v_cndmask_b32_e64 v115, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v111
	v_subrev_co_u32_e32 v111, vcc, s68, v111
	s_nop 0
	v_cndmask_b32_e64 v116, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cmp_eq_u32_e64 s[14:15], s69, v114
	s_nop 0
	v_cndmask_b32_e64 v113, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v111
	v_cndmask_b32_e64 v114, v115, v116, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v111, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v70, v113, v111, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_nop 1
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v110, v70
	v_cmp_ne_u32_e32 vcc, 0, v114
	s_nop 1
	v_cndmask_b32_e32 v70, v110, v70, vcc
	v_xor_b32_e32 v110, s72, v112
	v_xor_b32_e32 v70, v70, v110
	v_sub_u32_e32 v70, v70, v110
	v_ashrrev_i32_e32 v110, 31, v109
	v_mov_b32_e32 v111, v110
	v_lshl_add_u64 v[108:109], v[108:109], 0, v[110:111]
	v_xor_b32_e32 v117, v108, v110
	v_xad_u32 v116, v70, v90, v37
	v_xor_b32_e32 v111, v109, v110
	v_mad_u64_u32 v[108:109], s[14:15], v117, s57, 0
	v_mul_hi_u32 v70, v117, s56
	v_lshl_add_u64 v[108:109], v[70:71], 0, v[108:109]
	v_mad_u64_u32 v[114:115], s[14:15], v111, s56, 0
	v_add_co_u32_e32 v70, vcc, v108, v114
	v_mad_u64_u32 v[112:113], s[14:15], v111, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v109, v115, vcc
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	v_lshl_add_u64 v[108:109], v[70:71], 0, v[112:113]
	v_mul_lo_u32 v70, s69, v108
	v_mul_lo_u32 v112, s68, v109
	v_mad_u64_u32 v[108:109], s[14:15], s68, v108, 0
	v_add3_u32 v70, v109, v112, v70
	v_sub_u32_e32 v109, v111, v70
	v_sub_co_u32_e32 v108, vcc, v117, v108
	s_nop 1
	v_subb_co_u32_e64 v109, s[14:15], v109, v35, vcc
	v_subrev_co_u32_e64 v112, s[14:15], s68, v108
	v_subb_co_u32_e32 v70, vcc, v111, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v113, s[28:29], 0, v109, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v113
	v_subb_co_u32_e64 v109, s[14:15], v109, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v114, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v112
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v115, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v113
	v_cndmask_b32_e64 v111, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v108
	v_cndmask_b32_e64 v114, v114, v115, s[28:29]
	v_subrev_co_u32_e64 v115, s[14:15], s68, v112
	s_nop 1
	v_subbrev_co_u32_e64 v109, s[14:15], 0, v109, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v114
	s_nop 1
	v_cndmask_b32_e64 v109, v113, v109, s[14:15]
	v_cndmask_b32_e64 v113, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v111, v111, v113, vcc
	v_cmp_ne_u32_e32 vcc, 0, v111
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v109, vcc
	v_cndmask_b32_e64 v109, v112, v115, s[14:15]
	v_cndmask_b32_e32 v108, v108, v109, vcc
	v_xor_b32_e32 v109, v70, v110
	v_xor_b32_e32 v70, v108, v110
	v_sub_co_u32_e32 v70, vcc, v70, v110
	s_mov_b64 s[14:15], 0xc06
	s_nop 0
	v_subb_co_u32_e32 v109, vcc, v109, v110, vcc
	v_ashrrev_i32_e32 v108, 31, v109
	v_and_b32_e32 v108, s44, v108
	v_mul_lo_u32 v109, v116, s33
	v_lshl_add_u64 v[110:111], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf3f9
	v_add3_u32 v109, v108, v70, v109
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v112, v110, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v108, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v113, v111, v108, s[22:23]
	v_ashrrev_i32_e32 v114, 31, v113
	v_mov_b32_e32 v115, v114
	v_lshl_add_u64 v[112:113], v[112:113], 0, v[114:115]
	v_xor_b32_e32 v115, v112, v114
	v_xor_b32_e32 v108, v113, v114
	v_mad_u64_u32 v[112:113], s[14:15], v115, s57, 0
	v_mul_hi_u32 v70, v115, s56
	v_lshl_add_u64 v[112:113], v[70:71], 0, v[112:113]
	v_mad_u64_u32 v[118:119], s[14:15], v108, s56, 0
	v_add_co_u32_e32 v70, vcc, v112, v118
	v_mad_u64_u32 v[116:117], s[14:15], v108, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v113, v119, vcc
	s_nop 1
	v_addc_co_u32_e32 v117, vcc, 0, v117, vcc
	v_lshl_add_u64 v[112:113], v[70:71], 0, v[116:117]
	v_mul_lo_u32 v70, s69, v112
	v_mul_lo_u32 v113, s68, v113
	v_mad_u64_u32 v[116:117], s[14:15], s68, v112, 0
	v_add3_u32 v70, v117, v113, v70
	v_sub_co_u32_e32 v113, vcc, v115, v116
	s_nop 1
	v_subb_co_u32_e64 v115, s[14:15], v108, v70, vcc
	v_sub_u32_e32 v70, v108, v70
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	v_subrev_co_u32_e32 v108, vcc, s68, v113
	v_cmp_le_u32_e64 s[14:15], s69, v115
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v116, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v113
	v_cndmask_b32_e64 v113, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v108
	v_cndmask_b32_e64 v117, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v115
	v_cndmask_b32_e64 v108, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v115, v116, v117, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v113, v108, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v108, s72, v114
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v112, v70
	v_cmp_ne_u32_e32 vcc, 0, v115
	s_nop 1
	v_cndmask_b32_e32 v70, v112, v70, vcc
	v_ashrrev_i32_e32 v112, 31, v111
	v_mov_b32_e32 v113, v112
	v_xor_b32_e32 v70, v70, v108
	v_lshl_add_u64 v[110:111], v[110:111], 0, v[112:113]
	v_sub_u32_e32 v70, v70, v108
	v_xor_b32_e32 v118, v110, v112
	v_xad_u32 v108, v70, v90, v37
	v_xor_b32_e32 v113, v111, v112
	v_mad_u64_u32 v[110:111], s[14:15], v118, s57, 0
	v_mul_hi_u32 v70, v118, s56
	v_lshl_add_u64 v[110:111], v[70:71], 0, v[110:111]
	v_mad_u64_u32 v[116:117], s[14:15], v113, s56, 0
	v_add_co_u32_e32 v70, vcc, v110, v116
	v_mad_u64_u32 v[114:115], s[14:15], v113, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v111, v117, vcc
	v_mul_lo_u32 v108, v108, s33
	s_nop 0
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	v_lshl_add_u64 v[110:111], v[70:71], 0, v[114:115]
	v_mul_lo_u32 v70, s69, v110
	v_mul_lo_u32 v114, s68, v111
	v_mad_u64_u32 v[110:111], s[14:15], s68, v110, 0
	v_add3_u32 v70, v111, v114, v70
	v_sub_u32_e32 v111, v113, v70
	v_sub_co_u32_e32 v110, vcc, v118, v110
	s_nop 1
	v_subb_co_u32_e64 v111, s[14:15], v111, v35, vcc
	v_subrev_co_u32_e64 v114, s[14:15], s68, v110
	v_subb_co_u32_e32 v70, vcc, v113, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v115, s[28:29], 0, v111, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v115
	v_subb_co_u32_e64 v111, s[14:15], v111, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v116, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v114
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v117, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v115
	v_cndmask_b32_e64 v113, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v110
	v_cndmask_b32_e64 v116, v116, v117, s[28:29]
	v_subrev_co_u32_e64 v117, s[14:15], s68, v114
	s_nop 1
	v_subbrev_co_u32_e64 v111, s[14:15], 0, v111, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v116
	s_nop 1
	v_cndmask_b32_e64 v111, v115, v111, s[14:15]
	v_cndmask_b32_e64 v115, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v113, v113, v115, vcc
	v_cmp_ne_u32_e32 vcc, 0, v113
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v111, vcc
	v_cndmask_b32_e64 v111, v114, v117, s[14:15]
	v_cndmask_b32_e32 v110, v110, v111, vcc
	v_xor_b32_e32 v111, v70, v112
	v_xor_b32_e32 v70, v110, v112
	v_sub_co_u32_e32 v70, vcc, v70, v112
	s_mov_b64 s[14:15], 0xc07
	s_nop 0
	v_subb_co_u32_e32 v111, vcc, v111, v112, vcc
	v_ashrrev_i32_e32 v110, 31, v111
	v_and_b32_e32 v110, s44, v110
	v_add3_u32 v120, v110, v70, v108
	v_lshl_add_u64 v[110:111], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf3f8
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v112, v110, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v108, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v113, v111, v108, s[22:23]
	v_ashrrev_i32_e32 v114, 31, v113
	v_mov_b32_e32 v115, v114
	v_lshl_add_u64 v[112:113], v[112:113], 0, v[114:115]
	v_xor_b32_e32 v115, v112, v114
	v_xor_b32_e32 v108, v113, v114
	v_mad_u64_u32 v[112:113], s[14:15], v115, s57, 0
	v_mul_hi_u32 v70, v115, s56
	v_lshl_add_u64 v[112:113], v[70:71], 0, v[112:113]
	v_mad_u64_u32 v[118:119], s[14:15], v108, s56, 0
	v_add_co_u32_e32 v70, vcc, v112, v118
	v_mad_u64_u32 v[116:117], s[14:15], v108, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v113, v119, vcc
	s_nop 1
	v_addc_co_u32_e32 v117, vcc, 0, v117, vcc
	v_lshl_add_u64 v[112:113], v[70:71], 0, v[116:117]
	v_mul_lo_u32 v70, s69, v112
	v_mul_lo_u32 v113, s68, v113
	v_mad_u64_u32 v[116:117], s[14:15], s68, v112, 0
	v_add3_u32 v70, v117, v113, v70
	v_sub_co_u32_e32 v113, vcc, v115, v116
	s_nop 1
	v_subb_co_u32_e64 v115, s[14:15], v108, v70, vcc
	v_sub_u32_e32 v70, v108, v70
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	v_subrev_co_u32_e32 v108, vcc, s68, v113
	v_cmp_le_u32_e64 s[14:15], s69, v115
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v116, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v113
	v_cndmask_b32_e64 v113, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v108
	v_cndmask_b32_e64 v117, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v115
	v_cndmask_b32_e64 v108, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v115, v116, v117, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v113, v108, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v108, s72, v114
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v112, v70
	v_cmp_ne_u32_e32 vcc, 0, v115
	s_nop 1
	v_cndmask_b32_e32 v70, v112, v70, vcc
	v_ashrrev_i32_e32 v112, 31, v111
	v_mov_b32_e32 v113, v112
	v_xor_b32_e32 v70, v70, v108
	v_lshl_add_u64 v[110:111], v[110:111], 0, v[112:113]
	v_sub_u32_e32 v70, v70, v108
	v_xor_b32_e32 v118, v110, v112
	v_xad_u32 v108, v70, v90, v37
	v_xor_b32_e32 v113, v111, v112
	v_mad_u64_u32 v[110:111], s[14:15], v118, s57, 0
	v_mul_hi_u32 v70, v118, s56
	v_lshl_add_u64 v[110:111], v[70:71], 0, v[110:111]
	v_mad_u64_u32 v[116:117], s[14:15], v113, s56, 0
	v_add_co_u32_e32 v70, vcc, v110, v116
	v_mad_u64_u32 v[114:115], s[14:15], v113, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v111, v117, vcc
	v_mul_lo_u32 v108, v108, s33
	s_nop 0
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	v_lshl_add_u64 v[110:111], v[70:71], 0, v[114:115]
	v_mul_lo_u32 v70, s69, v110
	v_mul_lo_u32 v114, s68, v111
	v_mad_u64_u32 v[110:111], s[14:15], s68, v110, 0
	v_add3_u32 v70, v111, v114, v70
	v_sub_u32_e32 v111, v113, v70
	v_sub_co_u32_e32 v110, vcc, v118, v110
	s_nop 1
	v_subb_co_u32_e64 v111, s[14:15], v111, v35, vcc
	v_subrev_co_u32_e64 v114, s[14:15], s68, v110
	v_subb_co_u32_e32 v70, vcc, v113, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v115, s[28:29], 0, v111, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v115
	v_subb_co_u32_e64 v111, s[14:15], v111, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v116, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v114
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v117, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v115
	v_cndmask_b32_e64 v113, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v110
	v_cndmask_b32_e64 v116, v116, v117, s[28:29]
	v_subrev_co_u32_e64 v117, s[14:15], s68, v114
	s_nop 1
	v_subbrev_co_u32_e64 v111, s[14:15], 0, v111, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v116
	s_nop 1
	v_cndmask_b32_e64 v111, v115, v111, s[14:15]
	v_cndmask_b32_e64 v115, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v113, v113, v115, vcc
	v_cmp_ne_u32_e32 vcc, 0, v113
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v111, vcc
	v_cndmask_b32_e64 v111, v114, v117, s[14:15]
	v_cndmask_b32_e32 v110, v110, v111, vcc
	v_xor_b32_e32 v111, v70, v112
	v_xor_b32_e32 v70, v110, v112
	v_sub_co_u32_e32 v70, vcc, v70, v112
	s_mov_b64 s[14:15], 0xc08
	s_nop 0
	v_subb_co_u32_e32 v111, vcc, v111, v112, vcc
	v_ashrrev_i32_e32 v110, 31, v111
	v_and_b32_e32 v110, s44, v110
	v_add3_u32 v70, v110, v70, v108
	buffer_load_ubyte v110, v91, s[36:39], 0 offen
	buffer_load_ubyte v114, v92, s[36:39], 0 offen
	buffer_load_ubyte v111, v93, s[36:39], 0 offen
	buffer_load_ubyte v115, v94, s[36:39], 0 offen
	buffer_load_ubyte v108, v95, s[36:39], 0 offen
	buffer_load_ubyte v112, v109, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v109, v120, s[36:39], 0 offen
	buffer_load_ubyte v113, v70, s[36:39], 0 offen
	v_lshl_add_u64 v[92:93], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf3f7
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v94, v92, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v95, v93, v91, s[22:23]
	v_ashrrev_i32_e32 v116, 31, v95
	v_mov_b32_e32 v117, v116
	v_lshl_add_u64 v[94:95], v[94:95], 0, v[116:117]
	v_xor_b32_e32 v117, v94, v116
	v_xor_b32_e32 v91, v95, v116
	v_mad_u64_u32 v[94:95], s[14:15], v117, s57, 0
	v_mul_hi_u32 v70, v117, s56
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[94:95]
	v_mad_u64_u32 v[120:121], s[14:15], v91, s56, 0
	v_add_co_u32_e32 v70, vcc, v94, v120
	v_mad_u64_u32 v[118:119], s[14:15], v91, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v95, v121, vcc
	s_nop 1
	v_addc_co_u32_e32 v119, vcc, 0, v119, vcc
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[118:119]
	v_mul_lo_u32 v70, s69, v94
	v_mul_lo_u32 v95, s68, v95
	v_mad_u64_u32 v[118:119], s[14:15], s68, v94, 0
	v_add3_u32 v70, v119, v95, v70
	v_sub_co_u32_e32 v95, vcc, v117, v118
	s_nop 1
	v_subb_co_u32_e64 v117, s[14:15], v91, v70, vcc
	v_sub_u32_e32 v70, v91, v70
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	v_subrev_co_u32_e32 v91, vcc, s68, v95
	v_cmp_le_u32_e64 s[14:15], s69, v117
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v118, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v95
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v91
	v_cndmask_b32_e64 v119, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v117
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v117, v118, v119, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v95, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v91, s72, v116
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v94, v70
	v_cmp_ne_u32_e32 vcc, 0, v117
	s_nop 1
	v_cndmask_b32_e32 v70, v94, v70, vcc
	v_ashrrev_i32_e32 v94, 31, v93
	v_mov_b32_e32 v95, v94
	v_xor_b32_e32 v70, v70, v91
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[94:95]
	v_sub_u32_e32 v70, v70, v91
	v_xor_b32_e32 v95, v92, v94
	v_xad_u32 v120, v70, v90, v37
	v_xor_b32_e32 v91, v93, v94
	v_mad_u64_u32 v[92:93], s[14:15], v95, s57, 0
	v_mul_hi_u32 v70, v95, s56
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[92:93]
	v_mad_u64_u32 v[118:119], s[14:15], v91, s56, 0
	v_add_co_u32_e32 v70, vcc, v92, v118
	v_mad_u64_u32 v[116:117], s[14:15], v91, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v93, v119, vcc
	s_nop 1
	v_addc_co_u32_e32 v117, vcc, 0, v117, vcc
	v_lshl_add_u64 v[92:93], v[70:71], 0, v[116:117]
	v_mul_lo_u32 v70, s69, v92
	v_mul_lo_u32 v116, s68, v93
	v_mad_u64_u32 v[92:93], s[14:15], s68, v92, 0
	v_add3_u32 v70, v93, v116, v70
	v_sub_u32_e32 v93, v91, v70
	v_sub_co_u32_e32 v92, vcc, v95, v92
	s_nop 1
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, vcc
	v_subrev_co_u32_e64 v95, s[14:15], s68, v92
	v_subb_co_u32_e32 v70, vcc, v91, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v116, s[28:29], 0, v93, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v116
	v_subb_co_u32_e64 v93, s[14:15], v93, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v117, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v95
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v118, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v116
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v92
	v_cndmask_b32_e64 v117, v117, v118, s[28:29]
	v_subrev_co_u32_e64 v118, s[14:15], s68, v95
	s_nop 1
	v_subbrev_co_u32_e64 v93, s[14:15], 0, v93, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v117
	s_nop 1
	v_cndmask_b32_e64 v93, v116, v93, s[14:15]
	v_cndmask_b32_e64 v116, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v91, v91, v116, vcc
	v_cmp_ne_u32_e32 vcc, 0, v91
	v_cndmask_b32_e64 v91, v95, v118, s[14:15]
	s_mov_b64 s[14:15], 0xc09
	v_cndmask_b32_e32 v70, v70, v93, vcc
	v_cndmask_b32_e32 v91, v92, v91, vcc
	v_xor_b32_e32 v92, v70, v94
	v_xor_b32_e32 v70, v91, v94
	v_sub_co_u32_e32 v70, vcc, v70, v94
	s_nop 1
	v_subb_co_u32_e32 v91, vcc, v92, v94, vcc
	v_ashrrev_i32_e32 v91, 31, v91
	v_and_b32_e32 v91, s44, v91
	v_mul_lo_u32 v92, v120, s33
	v_lshl_add_u64 v[94:95], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf3f6
	v_add3_u32 v92, v91, v70, v92
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v116, v94, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v117, v95, v91, s[22:23]
	v_ashrrev_i32_e32 v118, 31, v117
	v_mov_b32_e32 v119, v118
	v_lshl_add_u64 v[116:117], v[116:117], 0, v[118:119]
	v_xor_b32_e32 v93, v116, v118
	v_xor_b32_e32 v91, v117, v118
	v_mad_u64_u32 v[116:117], s[14:15], v93, s57, 0
	v_mul_hi_u32 v70, v93, s56
	v_lshl_add_u64 v[116:117], v[70:71], 0, v[116:117]
	v_mad_u64_u32 v[122:123], s[14:15], v91, s56, 0
	v_add_co_u32_e32 v70, vcc, v116, v122
	v_mad_u64_u32 v[120:121], s[14:15], v91, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v117, v123, vcc
	s_nop 1
	v_addc_co_u32_e32 v121, vcc, 0, v121, vcc
	v_lshl_add_u64 v[116:117], v[70:71], 0, v[120:121]
	v_mul_lo_u32 v70, s69, v116
	v_mul_lo_u32 v117, s68, v117
	v_mad_u64_u32 v[120:121], s[14:15], s68, v116, 0
	v_add3_u32 v70, v121, v117, v70
	v_sub_co_u32_e32 v93, vcc, v93, v120
	s_nop 1
	v_subb_co_u32_e64 v117, s[14:15], v91, v70, vcc
	v_sub_u32_e32 v70, v91, v70
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	v_subrev_co_u32_e32 v91, vcc, s68, v93
	v_cmp_le_u32_e64 s[14:15], s69, v117
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v119, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v93
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v91
	v_cndmask_b32_e64 v120, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v117
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v117, v119, v120, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v93, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v91, s72, v118
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v116, v70
	v_cmp_ne_u32_e32 vcc, 0, v117
	s_nop 1
	v_cndmask_b32_e32 v70, v116, v70, vcc
	v_ashrrev_i32_e32 v116, 31, v95
	v_mov_b32_e32 v117, v116
	v_xor_b32_e32 v70, v70, v91
	v_lshl_add_u64 v[94:95], v[94:95], 0, v[116:117]
	v_sub_u32_e32 v70, v70, v91
	v_xor_b32_e32 v117, v94, v116
	v_xad_u32 v93, v70, v90, v37
	v_xor_b32_e32 v91, v95, v116
	v_mad_u64_u32 v[94:95], s[14:15], v117, s57, 0
	v_mul_hi_u32 v70, v117, s56
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[94:95]
	v_mad_u64_u32 v[120:121], s[14:15], v91, s56, 0
	v_add_co_u32_e32 v70, vcc, v94, v120
	v_mad_u64_u32 v[118:119], s[14:15], v91, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v95, v121, vcc
	v_mul_lo_u32 v93, v93, s33
	s_nop 0
	v_addc_co_u32_e32 v119, vcc, 0, v119, vcc
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[118:119]
	v_mul_lo_u32 v70, s69, v94
	v_mul_lo_u32 v118, s68, v95
	v_mad_u64_u32 v[94:95], s[14:15], s68, v94, 0
	v_add3_u32 v70, v95, v118, v70
	v_sub_u32_e32 v95, v91, v70
	v_sub_co_u32_e32 v94, vcc, v117, v94
	s_nop 1
	v_subb_co_u32_e64 v95, s[14:15], v95, v35, vcc
	v_subrev_co_u32_e64 v117, s[14:15], s68, v94
	v_subb_co_u32_e32 v70, vcc, v91, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v118, s[28:29], 0, v95, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v118
	v_subb_co_u32_e64 v95, s[14:15], v95, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v119, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v117
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v120, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v118
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v94
	v_cndmask_b32_e64 v119, v119, v120, s[28:29]
	v_subrev_co_u32_e64 v120, s[14:15], s68, v117
	s_nop 1
	v_subbrev_co_u32_e64 v95, s[14:15], 0, v95, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v119
	s_nop 1
	v_cndmask_b32_e64 v95, v118, v95, s[14:15]
	v_cndmask_b32_e64 v118, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v91, v91, v118, vcc
	v_cmp_ne_u32_e32 vcc, 0, v91
	v_cndmask_b32_e64 v91, v117, v120, s[14:15]
	s_mov_b64 s[14:15], 0xc0a
	v_cndmask_b32_e32 v70, v70, v95, vcc
	v_cndmask_b32_e32 v91, v94, v91, vcc
	v_xor_b32_e32 v94, v70, v116
	v_xor_b32_e32 v70, v91, v116
	v_sub_co_u32_e32 v70, vcc, v70, v116
	s_nop 1
	v_subb_co_u32_e32 v91, vcc, v94, v116, vcc
	v_ashrrev_i32_e32 v91, 31, v91
	v_and_b32_e32 v91, s44, v91
	v_lshl_add_u64 v[94:95], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf3f5
	v_add3_u32 v93, v91, v70, v93
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v116, v94, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v117, v95, v91, s[22:23]
	v_ashrrev_i32_e32 v118, 31, v117
	v_mov_b32_e32 v119, v118
	v_lshl_add_u64 v[116:117], v[116:117], 0, v[118:119]
	v_xor_b32_e32 v119, v116, v118
	v_xor_b32_e32 v91, v117, v118
	v_mad_u64_u32 v[116:117], s[14:15], v119, s57, 0
	v_mul_hi_u32 v70, v119, s56
	v_lshl_add_u64 v[116:117], v[70:71], 0, v[116:117]
	v_mad_u64_u32 v[122:123], s[14:15], v91, s56, 0
	v_add_co_u32_e32 v70, vcc, v116, v122
	v_mad_u64_u32 v[120:121], s[14:15], v91, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v117, v123, vcc
	s_nop 1
	v_addc_co_u32_e32 v121, vcc, 0, v121, vcc
	v_lshl_add_u64 v[116:117], v[70:71], 0, v[120:121]
	v_mul_lo_u32 v70, s69, v116
	v_mul_lo_u32 v117, s68, v117
	v_mad_u64_u32 v[120:121], s[14:15], s68, v116, 0
	v_add3_u32 v70, v121, v117, v70
	v_sub_co_u32_e32 v117, vcc, v119, v120
	s_nop 1
	v_subb_co_u32_e64 v119, s[14:15], v91, v70, vcc
	v_sub_u32_e32 v70, v91, v70
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	v_subrev_co_u32_e32 v91, vcc, s68, v117
	v_cmp_le_u32_e64 s[14:15], s69, v119
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v120, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v117
	v_cndmask_b32_e64 v117, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v91
	v_cndmask_b32_e64 v121, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v119
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v119, v120, v121, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v117, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v91, s72, v118
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v116, v70
	v_cmp_ne_u32_e32 vcc, 0, v119
	s_nop 1
	v_cndmask_b32_e32 v70, v116, v70, vcc
	v_ashrrev_i32_e32 v116, 31, v95
	v_mov_b32_e32 v117, v116
	v_xor_b32_e32 v70, v70, v91
	v_lshl_add_u64 v[94:95], v[94:95], 0, v[116:117]
	v_sub_u32_e32 v70, v70, v91
	v_xor_b32_e32 v117, v94, v116
	v_xad_u32 v122, v70, v90, v37
	v_xor_b32_e32 v91, v95, v116
	v_mad_u64_u32 v[94:95], s[14:15], v117, s57, 0
	v_mul_hi_u32 v70, v117, s56
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[94:95]
	v_mad_u64_u32 v[120:121], s[14:15], v91, s56, 0
	v_add_co_u32_e32 v70, vcc, v94, v120
	v_mad_u64_u32 v[118:119], s[14:15], v91, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v95, v121, vcc
	s_nop 1
	v_addc_co_u32_e32 v119, vcc, 0, v119, vcc
	v_lshl_add_u64 v[94:95], v[70:71], 0, v[118:119]
	v_mul_lo_u32 v70, s69, v94
	v_mul_lo_u32 v118, s68, v95
	v_mad_u64_u32 v[94:95], s[14:15], s68, v94, 0
	v_add3_u32 v70, v95, v118, v70
	v_sub_u32_e32 v95, v91, v70
	v_sub_co_u32_e32 v94, vcc, v117, v94
	s_nop 1
	v_subb_co_u32_e64 v95, s[14:15], v95, v35, vcc
	v_subrev_co_u32_e64 v117, s[14:15], s68, v94
	v_subb_co_u32_e32 v70, vcc, v91, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v118, s[28:29], 0, v95, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v118
	v_subb_co_u32_e64 v95, s[14:15], v95, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v119, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v117
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v120, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v118
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v94
	v_cndmask_b32_e64 v119, v119, v120, s[28:29]
	v_subrev_co_u32_e64 v120, s[14:15], s68, v117
	s_nop 1
	v_subbrev_co_u32_e64 v95, s[14:15], 0, v95, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v119
	s_nop 1
	v_cndmask_b32_e64 v95, v118, v95, s[14:15]
	v_cndmask_b32_e64 v118, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v91, v91, v118, vcc
	v_cmp_ne_u32_e32 vcc, 0, v91
	v_cndmask_b32_e64 v91, v117, v120, s[14:15]
	s_mov_b64 s[14:15], 0xc0b
	v_cndmask_b32_e32 v70, v70, v95, vcc
	v_cndmask_b32_e32 v91, v94, v91, vcc
	v_xor_b32_e32 v94, v70, v116
	v_xor_b32_e32 v70, v91, v116
	v_sub_co_u32_e32 v70, vcc, v70, v116
	s_nop 1
	v_subb_co_u32_e32 v91, vcc, v94, v116, vcc
	v_ashrrev_i32_e32 v91, 31, v91
	v_and_b32_e32 v91, s44, v91
	v_mul_lo_u32 v94, v122, s33
	v_lshl_add_u64 v[116:117], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf3f4
	v_add3_u32 v94, v91, v70, v94
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v118, v116, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v119, v117, v91, s[22:23]
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v95, v118, v120
	v_xor_b32_e32 v91, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v95, s57, 0
	v_mul_hi_u32 v70, v95, s56
	v_lshl_add_u64 v[118:119], v[70:71], 0, v[118:119]
	v_mad_u64_u32 v[128:129], s[14:15], v91, s56, 0
	v_add_co_u32_e32 v70, vcc, v118, v128
	v_mad_u64_u32 v[122:123], s[14:15], v91, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v119, v129, vcc
	s_nop 1
	v_addc_co_u32_e32 v123, vcc, 0, v123, vcc
	v_lshl_add_u64 v[118:119], v[70:71], 0, v[122:123]
	v_mul_lo_u32 v70, s69, v118
	v_mul_lo_u32 v119, s68, v119
	v_mad_u64_u32 v[122:123], s[14:15], s68, v118, 0
	v_add3_u32 v70, v123, v119, v70
	v_sub_co_u32_e32 v95, vcc, v95, v122
	s_nop 1
	v_subb_co_u32_e64 v119, s[14:15], v91, v70, vcc
	v_sub_u32_e32 v70, v91, v70
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	v_subrev_co_u32_e32 v91, vcc, s68, v95
	v_cmp_le_u32_e64 s[14:15], s69, v119
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v121, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v95
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v91
	v_cndmask_b32_e64 v122, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v119
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v119, v121, v122, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v95, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v91, s72, v120
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v118, v70
	v_cmp_ne_u32_e32 vcc, 0, v119
	s_nop 1
	v_cndmask_b32_e32 v70, v118, v70, vcc
	v_ashrrev_i32_e32 v118, 31, v117
	v_mov_b32_e32 v119, v118
	v_xor_b32_e32 v70, v70, v91
	v_lshl_add_u64 v[116:117], v[116:117], 0, v[118:119]
	v_sub_u32_e32 v70, v70, v91
	v_xor_b32_e32 v119, v116, v118
	v_xad_u32 v95, v70, v90, v37
	v_xor_b32_e32 v91, v117, v118
	v_mad_u64_u32 v[116:117], s[14:15], v119, s57, 0
	v_mul_hi_u32 v70, v119, s56
	v_lshl_add_u64 v[116:117], v[70:71], 0, v[116:117]
	v_mad_u64_u32 v[122:123], s[14:15], v91, s56, 0
	v_add_co_u32_e32 v70, vcc, v116, v122
	v_mad_u64_u32 v[120:121], s[14:15], v91, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v117, v123, vcc
	v_mul_lo_u32 v95, v95, s33
	s_nop 0
	v_addc_co_u32_e32 v121, vcc, 0, v121, vcc
	v_lshl_add_u64 v[116:117], v[70:71], 0, v[120:121]
	v_mul_lo_u32 v70, s69, v116
	v_mul_lo_u32 v120, s68, v117
	v_mad_u64_u32 v[116:117], s[14:15], s68, v116, 0
	v_add3_u32 v70, v117, v120, v70
	v_sub_u32_e32 v117, v91, v70
	v_sub_co_u32_e32 v116, vcc, v119, v116
	s_nop 1
	v_subb_co_u32_e64 v117, s[14:15], v117, v35, vcc
	v_subrev_co_u32_e64 v119, s[14:15], s68, v116
	v_subb_co_u32_e32 v70, vcc, v91, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v120, s[28:29], 0, v117, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v120
	v_subb_co_u32_e64 v117, s[14:15], v117, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v121, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v119
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v122, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v120
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v116
	v_cndmask_b32_e64 v121, v121, v122, s[28:29]
	v_subrev_co_u32_e64 v122, s[14:15], s68, v119
	s_nop 1
	v_subbrev_co_u32_e64 v117, s[14:15], 0, v117, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v121
	s_nop 1
	v_cndmask_b32_e64 v117, v120, v117, s[14:15]
	v_cndmask_b32_e64 v120, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v91, v91, v120, vcc
	v_cmp_ne_u32_e32 vcc, 0, v91
	v_cndmask_b32_e64 v91, v119, v122, s[14:15]
	s_mov_b64 s[14:15], 0xc0c
	v_cndmask_b32_e32 v70, v70, v117, vcc
	v_cndmask_b32_e32 v91, v116, v91, vcc
	v_xor_b32_e32 v116, v70, v118
	v_xor_b32_e32 v70, v91, v118
	v_sub_co_u32_e32 v70, vcc, v70, v118
	s_nop 1
	v_subb_co_u32_e32 v91, vcc, v116, v118, vcc
	v_ashrrev_i32_e32 v91, 31, v91
	v_and_b32_e32 v91, s44, v91
	v_lshl_add_u64 v[116:117], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf3f3
	v_add3_u32 v95, v91, v70, v95
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v118, v116, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v119, v117, v91, s[22:23]
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v121, v118, v120
	v_xor_b32_e32 v91, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v121, s57, 0
	v_mul_hi_u32 v70, v121, s56
	v_lshl_add_u64 v[118:119], v[70:71], 0, v[118:119]
	v_mad_u64_u32 v[128:129], s[14:15], v91, s56, 0
	v_add_co_u32_e32 v70, vcc, v118, v128
	v_mad_u64_u32 v[122:123], s[14:15], v91, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v119, v129, vcc
	s_nop 1
	v_addc_co_u32_e32 v123, vcc, 0, v123, vcc
	v_lshl_add_u64 v[118:119], v[70:71], 0, v[122:123]
	v_mul_lo_u32 v70, s69, v118
	v_mul_lo_u32 v119, s68, v119
	v_mad_u64_u32 v[122:123], s[14:15], s68, v118, 0
	v_add3_u32 v70, v123, v119, v70
	v_sub_co_u32_e32 v119, vcc, v121, v122
	s_nop 1
	v_subb_co_u32_e64 v121, s[14:15], v91, v70, vcc
	v_sub_u32_e32 v70, v91, v70
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	v_subrev_co_u32_e32 v91, vcc, s68, v119
	v_cmp_le_u32_e64 s[14:15], s69, v121
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v122, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v119
	v_cndmask_b32_e64 v119, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v91
	v_cndmask_b32_e64 v123, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v121
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v121, v122, v123, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v119, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v91, s72, v120
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v118, v70
	v_cmp_ne_u32_e32 vcc, 0, v121
	s_nop 1
	v_cndmask_b32_e32 v70, v118, v70, vcc
	v_ashrrev_i32_e32 v118, 31, v117
	v_mov_b32_e32 v119, v118
	v_xor_b32_e32 v70, v70, v91
	v_lshl_add_u64 v[116:117], v[116:117], 0, v[118:119]
	v_sub_u32_e32 v70, v70, v91
	v_xor_b32_e32 v119, v116, v118
	v_xad_u32 v128, v70, v90, v37
	v_xor_b32_e32 v91, v117, v118
	v_mad_u64_u32 v[116:117], s[14:15], v119, s57, 0
	v_mul_hi_u32 v70, v119, s56
	v_lshl_add_u64 v[116:117], v[70:71], 0, v[116:117]
	v_mad_u64_u32 v[122:123], s[14:15], v91, s56, 0
	v_add_co_u32_e32 v70, vcc, v116, v122
	v_mad_u64_u32 v[120:121], s[14:15], v91, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v117, v123, vcc
	s_nop 1
	v_addc_co_u32_e32 v121, vcc, 0, v121, vcc
	v_lshl_add_u64 v[116:117], v[70:71], 0, v[120:121]
	v_mul_lo_u32 v70, s69, v116
	v_mul_lo_u32 v120, s68, v117
	v_mad_u64_u32 v[116:117], s[14:15], s68, v116, 0
	v_add3_u32 v70, v117, v120, v70
	v_sub_u32_e32 v117, v91, v70
	v_sub_co_u32_e32 v116, vcc, v119, v116
	s_nop 1
	v_subb_co_u32_e64 v117, s[14:15], v117, v35, vcc
	v_subrev_co_u32_e64 v119, s[14:15], s68, v116
	v_subb_co_u32_e32 v70, vcc, v91, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v120, s[28:29], 0, v117, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v120
	v_subb_co_u32_e64 v117, s[14:15], v117, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v121, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v119
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v122, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v120
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v116
	v_cndmask_b32_e64 v121, v121, v122, s[28:29]
	v_subrev_co_u32_e64 v122, s[14:15], s68, v119
	s_nop 1
	v_subbrev_co_u32_e64 v117, s[14:15], 0, v117, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v121
	s_nop 1
	v_cndmask_b32_e64 v117, v120, v117, s[14:15]
	v_cndmask_b32_e64 v120, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v91, v91, v120, vcc
	v_cmp_ne_u32_e32 vcc, 0, v91
	v_cndmask_b32_e64 v91, v119, v122, s[14:15]
	s_mov_b64 s[14:15], 0xc0d
	v_cndmask_b32_e32 v70, v70, v117, vcc
	v_cndmask_b32_e32 v91, v116, v91, vcc
	v_xor_b32_e32 v116, v70, v118
	v_xor_b32_e32 v70, v91, v118
	v_sub_co_u32_e32 v70, vcc, v70, v118
	s_nop 1
	v_subb_co_u32_e32 v91, vcc, v116, v118, vcc
	v_ashrrev_i32_e32 v91, 31, v91
	v_and_b32_e32 v91, s44, v91
	v_mul_lo_u32 v116, v128, s33
	v_lshl_add_u64 v[118:119], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf3f2
	v_add3_u32 v116, v91, v70, v116
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v120, v118, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v121, v119, v91, s[22:23]
	v_ashrrev_i32_e32 v122, 31, v121
	v_mov_b32_e32 v123, v122
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[122:123]
	v_xor_b32_e32 v117, v120, v122
	v_xor_b32_e32 v91, v121, v122
	v_mad_u64_u32 v[120:121], s[14:15], v117, s57, 0
	v_mul_hi_u32 v70, v117, s56
	v_lshl_add_u64 v[120:121], v[70:71], 0, v[120:121]
	v_mad_u64_u32 v[130:131], s[14:15], v91, s56, 0
	v_add_co_u32_e32 v70, vcc, v120, v130
	v_mad_u64_u32 v[128:129], s[14:15], v91, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v121, v131, vcc
	s_nop 1
	v_addc_co_u32_e32 v129, vcc, 0, v129, vcc
	v_lshl_add_u64 v[120:121], v[70:71], 0, v[128:129]
	v_mul_lo_u32 v70, s69, v120
	v_mul_lo_u32 v121, s68, v121
	v_mad_u64_u32 v[128:129], s[14:15], s68, v120, 0
	v_add3_u32 v70, v129, v121, v70
	v_sub_co_u32_e32 v117, vcc, v117, v128
	s_nop 1
	v_subb_co_u32_e64 v121, s[14:15], v91, v70, vcc
	v_sub_u32_e32 v70, v91, v70
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	v_subrev_co_u32_e32 v91, vcc, s68, v117
	v_cmp_le_u32_e64 s[14:15], s69, v121
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v123, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v117
	v_cndmask_b32_e64 v117, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v91
	v_cndmask_b32_e64 v128, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v121
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v121, v123, v128, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v117, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v91, s72, v122
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v120, v70
	v_cmp_ne_u32_e32 vcc, 0, v121
	s_nop 1
	v_cndmask_b32_e32 v70, v120, v70, vcc
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_xor_b32_e32 v70, v70, v91
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_sub_u32_e32 v70, v70, v91
	v_xor_b32_e32 v121, v118, v120
	v_xad_u32 v117, v70, v90, v37
	v_xor_b32_e32 v91, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v121, s57, 0
	v_mul_hi_u32 v70, v121, s56
	v_lshl_add_u64 v[118:119], v[70:71], 0, v[118:119]
	v_mad_u64_u32 v[128:129], s[14:15], v91, s56, 0
	v_add_co_u32_e32 v70, vcc, v118, v128
	v_mad_u64_u32 v[122:123], s[14:15], v91, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v119, v129, vcc
	v_mul_lo_u32 v117, v117, s33
	s_nop 0
	v_addc_co_u32_e32 v123, vcc, 0, v123, vcc
	v_lshl_add_u64 v[118:119], v[70:71], 0, v[122:123]
	v_mul_lo_u32 v70, s69, v118
	v_mul_lo_u32 v122, s68, v119
	v_mad_u64_u32 v[118:119], s[14:15], s68, v118, 0
	v_add3_u32 v70, v119, v122, v70
	v_sub_u32_e32 v119, v91, v70
	v_sub_co_u32_e32 v118, vcc, v121, v118
	s_nop 1
	v_subb_co_u32_e64 v119, s[14:15], v119, v35, vcc
	v_subrev_co_u32_e64 v121, s[14:15], s68, v118
	v_subb_co_u32_e32 v70, vcc, v91, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v122, s[28:29], 0, v119, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v122
	v_subb_co_u32_e64 v119, s[14:15], v119, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v123, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v121
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v128, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v122
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v118
	v_cndmask_b32_e64 v123, v123, v128, s[28:29]
	v_subrev_co_u32_e64 v128, s[14:15], s68, v121
	s_nop 1
	v_subbrev_co_u32_e64 v119, s[14:15], 0, v119, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v123
	s_nop 1
	v_cndmask_b32_e64 v119, v122, v119, s[14:15]
	v_cndmask_b32_e64 v122, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v91, v91, v122, vcc
	v_cmp_ne_u32_e32 vcc, 0, v91
	v_cndmask_b32_e64 v91, v121, v128, s[14:15]
	s_mov_b64 s[14:15], 0xc0e
	v_cndmask_b32_e32 v70, v70, v119, vcc
	v_cndmask_b32_e32 v91, v118, v91, vcc
	v_xor_b32_e32 v118, v70, v120
	v_xor_b32_e32 v70, v91, v120
	v_sub_co_u32_e32 v70, vcc, v70, v120
	s_nop 1
	v_subb_co_u32_e32 v91, vcc, v118, v120, vcc
	v_ashrrev_i32_e32 v91, 31, v91
	v_and_b32_e32 v91, s44, v91
	v_lshl_add_u64 v[118:119], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf3f1
	v_add3_u32 v117, v91, v70, v117
	v_sub_co_u32_e32 v70, vcc, s14, v88
	v_cndmask_b32_e64 v120, v118, v70, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v121, v119, v91, s[22:23]
	v_ashrrev_i32_e32 v122, 31, v121
	v_mov_b32_e32 v123, v122
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[122:123]
	v_xor_b32_e32 v123, v120, v122
	v_xor_b32_e32 v91, v121, v122
	v_mad_u64_u32 v[120:121], s[14:15], v123, s57, 0
	v_mul_hi_u32 v70, v123, s56
	v_lshl_add_u64 v[120:121], v[70:71], 0, v[120:121]
	v_mad_u64_u32 v[130:131], s[14:15], v91, s56, 0
	v_add_co_u32_e32 v70, vcc, v120, v130
	v_mad_u64_u32 v[128:129], s[14:15], v91, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v121, v131, vcc
	s_nop 1
	v_addc_co_u32_e32 v129, vcc, 0, v129, vcc
	v_lshl_add_u64 v[120:121], v[70:71], 0, v[128:129]
	v_mul_lo_u32 v70, s69, v120
	v_mul_lo_u32 v121, s68, v121
	v_mad_u64_u32 v[128:129], s[14:15], s68, v120, 0
	v_add3_u32 v70, v129, v121, v70
	v_sub_co_u32_e32 v121, vcc, v123, v128
	s_nop 1
	v_subb_co_u32_e64 v123, s[14:15], v91, v70, vcc
	v_sub_u32_e32 v70, v91, v70
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	v_subrev_co_u32_e32 v91, vcc, s68, v121
	v_cmp_le_u32_e64 s[14:15], s69, v123
	s_nop 0
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v128, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v121
	v_cndmask_b32_e64 v121, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v91
	v_cndmask_b32_e64 v129, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s69, v123
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	v_cndmask_b32_e64 v123, v128, v129, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v70, v121, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	v_xor_b32_e32 v91, s72, v122
	s_nop 0
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v120, v70
	v_cmp_ne_u32_e32 vcc, 0, v123
	s_nop 1
	v_cndmask_b32_e32 v70, v120, v70, vcc
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_xor_b32_e32 v70, v70, v91
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_sub_u32_e32 v70, v70, v91
	v_xor_b32_e32 v121, v118, v120
	v_xad_u32 v130, v70, v90, v37
	v_xor_b32_e32 v91, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v121, s57, 0
	v_mul_hi_u32 v70, v121, s56
	v_lshl_add_u64 v[118:119], v[70:71], 0, v[118:119]
	v_mad_u64_u32 v[128:129], s[14:15], v91, s56, 0
	v_add_co_u32_e32 v70, vcc, v118, v128
	v_mad_u64_u32 v[122:123], s[14:15], v91, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v119, v129, vcc
	s_nop 1
	v_addc_co_u32_e32 v123, vcc, 0, v123, vcc
	v_lshl_add_u64 v[118:119], v[70:71], 0, v[122:123]
	v_mul_lo_u32 v70, s69, v118
	v_mul_lo_u32 v122, s68, v119
	v_mad_u64_u32 v[118:119], s[14:15], s68, v118, 0
	v_add3_u32 v70, v119, v122, v70
	v_sub_u32_e32 v119, v91, v70
	v_sub_co_u32_e32 v118, vcc, v121, v118
	s_nop 1
	v_subb_co_u32_e64 v119, s[14:15], v119, v35, vcc
	v_subrev_co_u32_e64 v121, s[14:15], s68, v118
	v_subb_co_u32_e32 v70, vcc, v91, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v122, s[28:29], 0, v119, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s69, v122
	v_subb_co_u32_e64 v119, s[14:15], v119, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v123, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s68, v121
	v_cmp_le_u32_e32 vcc, s69, v70
	s_nop 0
	v_cndmask_b32_e64 v128, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s69, v122
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v118
	v_cndmask_b32_e64 v123, v123, v128, s[28:29]
	v_subrev_co_u32_e64 v128, s[14:15], s68, v121
	s_nop 1
	v_subbrev_co_u32_e64 v119, s[14:15], 0, v119, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v123
	s_nop 1
	v_cndmask_b32_e64 v119, v122, v119, s[14:15]
	v_cndmask_b32_e64 v122, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v91, v91, v122, vcc
	v_cmp_ne_u32_e32 vcc, 0, v91
	v_cndmask_b32_e64 v91, v121, v128, s[14:15]
	s_mov_b64 s[14:15], 0xc0f
	v_cndmask_b32_e32 v70, v70, v119, vcc
	v_cndmask_b32_e32 v91, v118, v91, vcc
	v_xor_b32_e32 v118, v70, v120
	v_xor_b32_e32 v70, v91, v120
	v_sub_co_u32_e32 v70, vcc, v70, v120
	s_nop 1
	v_subb_co_u32_e32 v91, vcc, v118, v120, vcc
	v_ashrrev_i32_e32 v91, 31, v91
	v_and_b32_e32 v91, s44, v91
	v_mul_lo_u32 v118, v130, s33
	v_lshl_add_u64 v[120:121], v[88:89], 0, s[14:15]
	s_movk_i32 s14, 0xf3f0
	v_add3_u32 v118, v91, v70, v118
	v_sub_co_u32_e32 v70, vcc, s14, v88
	s_nop 1
	v_subb_co_u32_e32 v88, vcc, -1, v89, vcc
	v_cndmask_b32_e64 v89, v121, v88, s[22:23]
	v_ashrrev_i32_e32 v122, 31, v89
	v_cndmask_b32_e64 v88, v120, v70, s[22:23]
	v_mov_b32_e32 v123, v122
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[122:123]
	v_xor_b32_e32 v119, v88, v122
	v_xor_b32_e32 v91, v89, v122
	v_mad_u64_u32 v[88:89], s[14:15], v119, s57, 0
	v_mul_hi_u32 v70, v119, s56
	v_lshl_add_u64 v[88:89], v[70:71], 0, v[88:89]
	v_mad_u64_u32 v[130:131], s[14:15], v91, s56, 0
	v_add_co_u32_e32 v70, vcc, v88, v130
	v_mad_u64_u32 v[128:129], s[14:15], v91, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v89, v131, vcc
	s_nop 1
	v_addc_co_u32_e32 v129, vcc, 0, v129, vcc
	v_lshl_add_u64 v[88:89], v[70:71], 0, v[128:129]
	v_mul_lo_u32 v70, s69, v88
	v_mul_lo_u32 v89, s68, v89
	v_mad_u64_u32 v[128:129], s[14:15], s68, v88, 0
	v_add3_u32 v70, v129, v89, v70
	v_sub_co_u32_e32 v89, vcc, v119, v128
	s_nop 1
	v_subb_co_u32_e64 v119, s[14:15], v91, v70, vcc
	v_sub_u32_e32 v70, v91, v70
	v_cmp_le_u32_e64 s[14:15], s69, v119
	v_subb_co_u32_e32 v70, vcc, v70, v35, vcc
	s_nop 0
	v_cndmask_b32_e64 v123, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s68, v89
	v_subrev_co_u32_e32 v89, vcc, s68, v89
	s_nop 0
	v_cndmask_b32_e64 v128, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v70, vcc, 0, v70, vcc
	v_cmp_le_u32_e32 vcc, s69, v70
	v_cmp_eq_u32_e64 s[14:15], s69, v119
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v89
	v_cndmask_b32_e64 v119, v123, v128, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v89, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v70
	s_nop 1
	v_cndmask_b32_e32 v70, v91, v89, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_nop 1
	v_cndmask_b32_e64 v70, 1, 2, vcc
	v_add_u32_e32 v70, v88, v70
	v_cmp_ne_u32_e32 vcc, 0, v119
	s_nop 1
	v_cndmask_b32_e32 v70, v88, v70, vcc
	v_xor_b32_e32 v88, s72, v122
	v_xor_b32_e32 v70, v70, v88
	v_sub_u32_e32 v70, v70, v88
	v_ashrrev_i32_e32 v88, 31, v121
	v_mov_b32_e32 v89, v88
	v_xad_u32 v119, v70, v90, v37
	v_lshl_add_u64 v[90:91], v[120:121], 0, v[88:89]
	v_xor_b32_e32 v120, v90, v88
	v_xor_b32_e32 v89, v91, v88
	v_mad_u64_u32 v[90:91], s[14:15], v120, s57, 0
	v_mul_hi_u32 v70, v120, s56
	v_lshl_add_u64 v[122:123], v[70:71], 0, v[90:91]
	v_mad_u64_u32 v[128:129], s[14:15], v89, s56, 0
	v_add_co_u32_e32 v70, vcc, v122, v128
	v_mad_u64_u32 v[90:91], s[14:15], v89, s57, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v123, v129, vcc
	ds_read_b128 v[132:135], v52 offset:20480
	ds_read_b128 v[128:131], v54 offset:22528
	ds_read_b128 v[124:127], v52 offset:22528
	ds_read_b128 v[136:139], v54 offset:20480
	ds_read_b32 v33, v50 offset:50432
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[90:91], v[70:71], 0, v[90:91]
	v_mul_lo_u32 v70, s69, v90
	v_mul_lo_u32 v91, s68, v91
	v_mad_u64_u32 v[122:123], s[14:15], s68, v90, 0
	v_add3_u32 v90, v123, v91, v70
	v_sub_u32_e32 v91, v89, v90
	v_sub_co_u32_e32 v70, vcc, v120, v122
	s_nop 1
	v_subb_co_u32_e64 v91, s[14:15], v91, v35, vcc
	v_subrev_co_u32_e64 v120, s[14:15], s68, v70
	v_subb_co_u32_e32 v89, vcc, v89, v90, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v121, s[22:23], 0, v91, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s69, v121
	v_subb_co_u32_e64 v35, s[14:15], v91, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v122, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s68, v120
	v_subrev_co_u32_e64 v91, s[14:15], s68, v120
	s_nop 0
	v_cndmask_b32_e64 v123, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s69, v121
	v_subbrev_co_u32_e64 v35, s[14:15], 0, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v122, v122, v123, s[22:23]
	v_cmp_le_u32_e32 vcc, s69, v89
	v_cmp_ne_u32_e64 s[14:15], 0, v122
	s_nop 0
	v_cndmask_b32_e64 v90, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s68, v70
	v_cndmask_b32_e64 v35, v121, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v121, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s69, v89
	s_nop 1
	v_cndmask_b32_e32 v90, v90, v121, vcc
	v_cmp_ne_u32_e32 vcc, 0, v90
	s_nop 1
	v_cndmask_b32_e32 v35, v89, v35, vcc
	v_cndmask_b32_e64 v89, v120, v91, s[14:15]
	v_cndmask_b32_e32 v70, v70, v89, vcc
	v_xor_b32_e32 v70, v70, v88
	v_xor_b32_e32 v35, v35, v88
	v_sub_co_u32_e32 v70, vcc, v70, v88
	s_nop 1
	v_subb_co_u32_e32 v35, vcc, v35, v88, vcc
	v_ashrrev_i32_e32 v35, 31, v35
	v_and_b32_e32 v35, s44, v35
	v_mul_lo_u32 v88, v119, s33
	v_add3_u32 v70, v35, v70, v88
	buffer_load_ubyte v119, v92, s[36:39], 0 offen
	buffer_load_ubyte v121, v93, s[36:39], 0 offen
	buffer_load_ubyte v120, v94, s[36:39], 0 offen
	buffer_load_ubyte v122, v95, s[36:39], 0 offen
	buffer_load_ubyte v35, v116, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v117, v117, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v116, v118, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v118, v70, s[36:39], 0 offen
	v_sub_co_u32_e32 v70, vcc, 0xfffffdff, v78
	v_lshl_add_u64 v[88:89], v[78:79], 0, s[24:25]
	s_nop 0
	v_subb_co_u32_e32 v90, vcc, -1, v79, vcc
	v_cmp_gt_i32_e32 vcc, 0, v89
	s_nop 1
	v_cndmask_b32_e32 v91, v89, v90, vcc
	v_cndmask_b32_e32 v90, v88, v70, vcc
	v_or_b32_e32 v70, s79, v91
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[22:23], exec, s[14:15]
	s_cbranch_execz .LBB0_66
	s_add_u32 s14, s78, s82
	s_mov_b32 s83, s82
	s_addc_u32 s15, s79, s82
	s_xor_b64 s[28:29], s[14:15], s[82:83]
	v_cvt_f32_u32_e32 v70, s28
	v_cvt_f32_u32_e32 v88, s29
	s_sub_u32 s14, 0, s28
	s_subb_u32 s15, 0, s29
	v_ashrrev_i32_e32 v92, 31, v91
	v_fmac_f32_e32 v70, 0x4f800000, v88
	v_rcp_f32_e32 v70, v70
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[92:93]
	v_xor_b32_e32 v93, v90, v92
	v_mul_f32_e32 v70, 0x5f7ffffc, v70
	v_mul_f32_e32 v88, 0x2f800000, v70
	v_trunc_f32_e32 v88, v88
	v_fmac_f32_e32 v70, 0xcf800000, v88
	v_cvt_u32_f32_e32 v88, v88
	v_cvt_u32_f32_e32 v70, v70
	v_readfirstlane_b32 s56, v88
	v_readfirstlane_b32 s57, v70
	s_mul_i32 s68, s14, s56
	s_mul_hi_u32 s73, s14, s57
	s_mul_i32 s69, s15, s57
	s_add_i32 s68, s73, s68
	s_add_i32 s68, s68, s69
	s_mul_i32 s75, s14, s57
	s_mul_i32 s73, s57, s68
	s_mul_hi_u32 s81, s57, s75
	s_mul_hi_u32 s69, s57, s68
	s_add_u32 s73, s81, s73
	s_addc_u32 s69, 0, s69
	s_mul_hi_u32 s83, s56, s75
	s_mul_i32 s75, s56, s75
	s_add_u32 s73, s73, s75
	s_mul_hi_u32 s81, s56, s68
	s_addc_u32 s69, s69, s83
	s_addc_u32 s73, s81, 0
	s_mul_i32 s68, s56, s68
	s_add_u32 s68, s69, s68
	s_addc_u32 s69, 0, s73
	s_add_u32 s57, s57, s68
	s_addc_u32 s56, s56, s69
	s_mul_i32 s68, s14, s56
	s_mul_hi_u32 s69, s14, s57
	s_add_i32 s68, s69, s68
	s_mul_i32 s15, s15, s57
	s_add_i32 s68, s68, s15
	s_mul_i32 s14, s14, s57
	s_mul_hi_u32 s69, s56, s14
	s_mul_i32 s73, s56, s14
	s_mul_i32 s81, s57, s68
	s_mul_hi_u32 s14, s57, s14
	s_mul_hi_u32 s75, s57, s68
	s_add_u32 s14, s14, s81
	s_addc_u32 s75, 0, s75
	s_add_u32 s14, s14, s73
	s_mul_hi_u32 s15, s56, s68
	s_addc_u32 s14, s75, s69
	s_addc_u32 s15, s15, 0
	s_mul_i32 s68, s56, s68
	s_add_u32 s14, s14, s68
	s_addc_u32 s15, 0, s15
	s_add_u32 s57, s57, s14
	s_addc_u32 s56, s56, s15
	v_xor_b32_e32 v88, v91, v92
	v_mad_u64_u32 v[90:91], s[14:15], v93, s56, 0
	v_mul_hi_u32 v70, v93, s57
	v_lshl_add_u64 v[90:91], v[70:71], 0, v[90:91]
	v_mad_u64_u32 v[140:141], s[14:15], v88, s57, 0
	v_add_co_u32_e32 v70, vcc, v90, v140
	v_mad_u64_u32 v[94:95], s[14:15], v88, s56, 0
	s_nop 0
	v_addc_co_u32_e32 v90, vcc, v91, v141, vcc
	v_mov_b32_e32 v91, s74
	s_nop 0
	v_addc_co_u32_e32 v95, vcc, 0, v95, vcc
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[94:95]
	v_mul_lo_u32 v70, s29, v90
	v_mul_lo_u32 v123, s28, v91
	v_mad_u64_u32 v[94:95], s[14:15], s28, v90, 0
	v_add3_u32 v70, v95, v123, v70
	v_sub_u32_e32 v95, v88, v70
	v_mov_b32_e32 v123, s29
	v_sub_co_u32_e32 v93, vcc, v93, v94
	v_lshl_add_u64 v[140:141], v[90:91], 0, 1
	s_nop 0
	v_subb_co_u32_e64 v94, s[14:15], v95, v123, vcc
	v_subrev_co_u32_e64 v95, s[14:15], s28, v93
	v_subb_co_u32_e32 v70, vcc, v88, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v94, s[14:15], 0, v94, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s29, v94
	v_cmp_le_u32_e32 vcc, s29, v70
	s_nop 0
	v_cndmask_b32_e64 v123, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v95
	v_cndmask_b32_e64 v88, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v93
	v_cndmask_b32_e64 v95, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v94
	s_nop 1
	v_cndmask_b32_e64 v123, v123, v95, s[14:15]
	v_lshl_add_u64 v[94:95], v[90:91], 0, 2
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v70
	s_nop 1
	v_cndmask_b32_e32 v70, v88, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v123
	s_nop 1
	v_cndmask_b32_e32 v88, v140, v94, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_nop 1
	v_cndmask_b32_e32 v70, v90, v88, vcc
	v_xor_b32_e32 v88, s82, v92
	v_xor_b32_e32 v70, v70, v88
	v_sub_co_u32_e32 v92, vcc, v70, v88
.LBB0_66:
	s_andn2_saveexec_b64 s[14:15], s[22:23]
	s_cbranch_execz .LBB0_68
	v_rcp_iflag_f32_e32 v70, v100
	s_sub_i32 s22, 0, s78
	v_mul_f32_e32 v70, 0x4f7ffffe, v70
	v_cvt_u32_f32_e32 v70, v70
	v_mul_lo_u32 v88, s22, v70
	v_mul_hi_u32 v88, v70, v88
	v_add_u32_e32 v70, v70, v88
	v_mul_hi_u32 v70, v90, v70
	v_mul_lo_u32 v88, v70, s78
	v_sub_u32_e32 v88, v90, v88
	v_add_u32_e32 v91, 1, v70
	v_subrev_u32_e32 v90, s78, v88
	v_cmp_le_u32_e32 vcc, s78, v88
	s_nop 1
	v_cndmask_b32_e32 v88, v88, v90, vcc
	v_cndmask_b32_e32 v70, v70, v91, vcc
	v_add_u32_e32 v90, 1, v70
	v_cmp_le_u32_e32 vcc, s78, v88
	s_nop 1
	v_cndmask_b32_e32 v92, v70, v90, vcc
.LBB0_68:
	s_or_b64 exec, exec, s[14:15]
	v_lshl_add_u64 v[90:91], v[80:81], 0, s[24:25]
	v_or_b32_e32 v70, s79, v91
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[28:29], exec, s[14:15]
	s_cbranch_execz .LBB0_70
	s_add_u32 s14, s78, s82
	s_mov_b32 s83, s82
	s_addc_u32 s15, s79, s82
	s_xor_b64 s[56:57], s[14:15], s[82:83]
	v_cvt_f32_u32_e32 v70, s56
	v_cvt_f32_u32_e32 v88, s57
	s_sub_u32 s14, 0, s56
	s_subb_u32 s15, 0, s57
	v_ashrrev_i32_e32 v140, 31, v91
	v_fmac_f32_e32 v70, 0x4f800000, v88
	v_rcp_f32_e32 v70, v70
	v_mov_b32_e32 v141, v140
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[140:141]
	v_xor_b32_e32 v93, v90, v140
	v_mul_f32_e32 v70, 0x5f7ffffc, v70
	v_mul_f32_e32 v88, 0x2f800000, v70
	v_trunc_f32_e32 v88, v88
	v_fmac_f32_e32 v70, 0xcf800000, v88
	v_cvt_u32_f32_e32 v88, v88
	v_cvt_u32_f32_e32 v70, v70
	v_readfirstlane_b32 s22, v88
	v_readfirstlane_b32 s23, v70
	s_mul_i32 s68, s14, s22
	s_mul_hi_u32 s73, s14, s23
	s_mul_i32 s69, s15, s23
	s_add_i32 s68, s73, s68
	s_add_i32 s68, s68, s69
	s_mul_i32 s75, s14, s23
	s_mul_i32 s73, s23, s68
	s_mul_hi_u32 s81, s23, s75
	s_mul_hi_u32 s69, s23, s68
	s_add_u32 s73, s81, s73
	s_addc_u32 s69, 0, s69
	s_mul_hi_u32 s83, s22, s75
	s_mul_i32 s75, s22, s75
	s_add_u32 s73, s73, s75
	s_mul_hi_u32 s81, s22, s68
	s_addc_u32 s69, s69, s83
	s_addc_u32 s73, s81, 0
	s_mul_i32 s68, s22, s68
	s_add_u32 s68, s69, s68
	s_addc_u32 s69, 0, s73
	s_add_u32 s23, s23, s68
	s_addc_u32 s22, s22, s69
	s_mul_i32 s68, s14, s22
	s_mul_hi_u32 s69, s14, s23
	s_add_i32 s68, s69, s68
	s_mul_i32 s15, s15, s23
	s_add_i32 s68, s68, s15
	s_mul_i32 s14, s14, s23
	s_mul_hi_u32 s69, s22, s14
	s_mul_i32 s73, s22, s14
	s_mul_i32 s81, s23, s68
	s_mul_hi_u32 s14, s23, s14
	s_mul_hi_u32 s75, s23, s68
	s_add_u32 s14, s14, s81
	s_addc_u32 s75, 0, s75
	s_add_u32 s14, s14, s73
	s_mul_hi_u32 s15, s22, s68
	s_addc_u32 s14, s75, s69
	s_addc_u32 s15, s15, 0
	s_mul_i32 s68, s22, s68
	s_add_u32 s14, s14, s68
	s_addc_u32 s15, 0, s15
	s_add_u32 s23, s23, s14
	s_addc_u32 s22, s22, s15
	v_xor_b32_e32 v88, v91, v140
	v_mad_u64_u32 v[90:91], s[14:15], v93, s22, 0
	v_mul_hi_u32 v70, v93, s23
	v_lshl_add_u64 v[90:91], v[70:71], 0, v[90:91]
	v_mad_u64_u32 v[142:143], s[14:15], v88, s23, 0
	v_add_co_u32_e32 v70, vcc, v90, v142
	v_mad_u64_u32 v[94:95], s[14:15], v88, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v90, vcc, v91, v143, vcc
	v_mov_b32_e32 v91, s74
	s_nop 0
	v_addc_co_u32_e32 v95, vcc, 0, v95, vcc
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[94:95]
	v_mul_lo_u32 v70, s57, v90
	v_mul_lo_u32 v94, s56, v91
	v_mad_u64_u32 v[90:91], s[14:15], s56, v90, 0
	v_add3_u32 v70, v91, v94, v70
	v_sub_u32_e32 v91, v88, v70
	v_mov_b32_e32 v94, s57
	v_sub_co_u32_e32 v90, vcc, v93, v90
	s_nop 1
	v_subb_co_u32_e64 v91, s[14:15], v91, v94, vcc
	v_subrev_co_u32_e64 v93, s[14:15], s56, v90
	v_subb_co_u32_e32 v70, vcc, v88, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v95, s[22:23], 0, v91, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s57, v95
	v_subb_co_u32_e64 v91, s[14:15], v91, v94, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v123, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s56, v93
	v_subrev_co_u32_e64 v94, s[14:15], s56, v93
	s_nop 0
	v_cndmask_b32_e64 v141, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s57, v95
	v_subbrev_co_u32_e64 v91, s[14:15], 0, v91, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v123, v123, v141, s[22:23]
	v_cmp_le_u32_e32 vcc, s57, v70
	v_cmp_ne_u32_e64 s[14:15], 0, v123
	s_nop 0
	v_cndmask_b32_e64 v88, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s56, v90
	v_cndmask_b32_e64 v91, v95, v91, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s57, v70
	s_nop 1
	v_cndmask_b32_e32 v88, v88, v95, vcc
	v_cmp_ne_u32_e32 vcc, 0, v88
	v_cndmask_b32_e64 v88, v93, v94, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v88, v90, v88, vcc
	v_cndmask_b32_e32 v70, v70, v91, vcc
	v_xor_b32_e32 v88, v88, v140
	v_xor_b32_e32 v70, v70, v140
	v_sub_co_u32_e32 v94, vcc, v88, v140
	s_nop 1
	v_subb_co_u32_e32 v95, vcc, v70, v140, vcc
.LBB0_70:
	s_andn2_saveexec_b64 s[14:15], s[28:29]
	s_cbranch_execz .LBB0_72
	v_rcp_iflag_f32_e32 v70, v100
	s_sub_i32 s22, 0, s78
	v_mul_f32_e32 v70, 0x4f7ffffe, v70
	v_cvt_u32_f32_e32 v70, v70
	v_mul_lo_u32 v88, s22, v70
	v_mul_hi_u32 v88, v70, v88
	v_add_u32_e32 v70, v70, v88
	v_mul_hi_u32 v70, v90, v70
	v_mul_lo_u32 v70, v70, s78
	v_sub_u32_e32 v70, v90, v70
	v_subrev_u32_e32 v88, s78, v70
	v_cmp_le_u32_e32 vcc, s78, v70
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v88, vcc
	v_subrev_u32_e32 v88, s78, v70
	v_cmp_le_u32_e32 vcc, s78, v70
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v88, vcc
	v_mov_b64_e32 v[94:95], v[70:71]
.LBB0_72:
	s_or_b64 exec, exec, s[14:15]
	v_ashrrev_i32_e32 v70, 31, v89
	v_xor_b32_e32 v70, v92, v70
	v_ashrrev_i32_e32 v88, 31, v95
	v_mul_lo_u32 v70, v70, s66
	v_and_b32_e32 v88, s78, v88
	v_add3_u32 v70, v70, v94, v88
	buffer_load_ubyte v92, v70, s[8:11], 0 offen
	v_sub_co_u32_e32 v70, vcc, 0xfffffdfd, v78
	v_lshl_add_u64 v[88:89], v[78:79], 0, s[62:63]
	s_nop 0
	v_subb_co_u32_e32 v78, vcc, -1, v79, vcc
	v_cmp_gt_i32_e32 vcc, 0, v89
	s_nop 1
	v_cndmask_b32_e32 v91, v89, v78, vcc
	v_cndmask_b32_e32 v90, v88, v70, vcc
	v_or_b32_e32 v70, s79, v91
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[22:23], exec, s[14:15]
	s_cbranch_execz .LBB0_74
	s_add_u32 s14, s78, s82
	s_mov_b32 s83, s82
	s_addc_u32 s15, s79, s82
	s_xor_b64 s[28:29], s[14:15], s[82:83]
	v_cvt_f32_u32_e32 v70, s28
	v_cvt_f32_u32_e32 v78, s29
	s_sub_u32 s14, 0, s28
	s_subb_u32 s15, 0, s29
	v_fmac_f32_e32 v70, 0x4f800000, v78
	v_rcp_f32_e32 v70, v70
	s_nop 0
	v_mul_f32_e32 v70, 0x5f7ffffc, v70
	v_mul_f32_e32 v78, 0x2f800000, v70
	v_trunc_f32_e32 v78, v78
	v_fmac_f32_e32 v70, 0xcf800000, v78
	v_cvt_u32_f32_e32 v78, v78
	v_cvt_u32_f32_e32 v70, v70
	v_readfirstlane_b32 s56, v78
	v_readfirstlane_b32 s57, v70
	s_mul_i32 s68, s14, s56
	s_mul_hi_u32 s73, s14, s57
	s_mul_i32 s69, s15, s57
	s_add_i32 s68, s73, s68
	s_add_i32 s68, s68, s69
	s_mul_i32 s75, s14, s57
	s_mul_i32 s73, s57, s68
	s_mul_hi_u32 s81, s57, s75
	s_mul_hi_u32 s69, s57, s68
	s_add_u32 s73, s81, s73
	s_addc_u32 s69, 0, s69
	s_mul_hi_u32 s83, s56, s75
	s_mul_i32 s75, s56, s75
	s_add_u32 s73, s73, s75
	s_mul_hi_u32 s81, s56, s68
	s_addc_u32 s69, s69, s83
	s_addc_u32 s73, s81, 0
	s_mul_i32 s68, s56, s68
	s_add_u32 s68, s69, s68
	s_addc_u32 s69, 0, s73
	s_add_u32 s57, s57, s68
	s_addc_u32 s56, s56, s69
	s_mul_i32 s68, s14, s56
	s_mul_hi_u32 s69, s14, s57
	s_add_i32 s68, s69, s68
	s_mul_i32 s15, s15, s57
	s_add_i32 s68, s68, s15
	s_mul_i32 s14, s14, s57
	s_mul_hi_u32 s69, s56, s14
	s_mul_i32 s73, s56, s14
	s_mul_i32 s81, s57, s68
	s_mul_hi_u32 s14, s57, s14
	s_mul_hi_u32 s75, s57, s68
	s_add_u32 s14, s14, s81
	s_addc_u32 s75, 0, s75
	s_add_u32 s14, s14, s73
	s_mul_hi_u32 s15, s56, s68
	s_addc_u32 s14, s75, s69
	s_addc_u32 s15, s15, 0
	s_mul_i32 s68, s56, s68
	s_add_u32 s14, s14, s68
	v_ashrrev_i32_e32 v78, 31, v91
	s_addc_u32 s15, 0, s15
	v_mov_b32_e32 v79, v78
	s_add_u32 s57, s57, s14
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[78:79]
	s_addc_u32 s56, s56, s15
	v_xor_b32_e32 v88, v90, v78
	v_xor_b32_e32 v79, v91, v78
	v_mad_u64_u32 v[90:91], s[14:15], v88, s56, 0
	v_mul_hi_u32 v70, v88, s57
	v_lshl_add_u64 v[90:91], v[70:71], 0, v[90:91]
	v_mad_u64_u32 v[140:141], s[14:15], v79, s57, 0
	v_add_co_u32_e32 v70, vcc, v90, v140
	v_mad_u64_u32 v[94:95], s[14:15], v79, s56, 0
	s_nop 0
	v_addc_co_u32_e32 v90, vcc, v91, v141, vcc
	v_mov_b32_e32 v91, s74
	s_nop 0
	v_addc_co_u32_e32 v95, vcc, 0, v95, vcc
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[94:95]
	v_mul_lo_u32 v70, s29, v90
	v_mul_lo_u32 v93, s28, v91
	v_mad_u64_u32 v[94:95], s[14:15], s28, v90, 0
	v_add3_u32 v70, v95, v93, v70
	v_sub_u32_e32 v93, v79, v70
	v_mov_b32_e32 v95, s29
	v_sub_co_u32_e32 v88, vcc, v88, v94
	v_lshl_add_u64 v[140:141], v[90:91], 0, 1
	s_nop 0
	v_subb_co_u32_e64 v93, s[14:15], v93, v95, vcc
	v_subrev_co_u32_e64 v94, s[14:15], s28, v88
	v_subb_co_u32_e32 v70, vcc, v79, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v93, s[14:15], 0, v93, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s29, v93
	v_cmp_le_u32_e32 vcc, s29, v70
	v_xor_b32_e32 v78, s82, v78
	v_cndmask_b32_e64 v95, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v94
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v88
	v_cndmask_b32_e64 v94, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v93
	v_cndmask_b32_e64 v88, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v70
	v_cndmask_b32_e64 v93, v95, v94, s[14:15]
	v_lshl_add_u64 v[94:95], v[90:91], 0, 2
	v_cndmask_b32_e32 v70, v79, v88, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v79, v140, v94, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_nop 1
	v_cndmask_b32_e32 v70, v90, v79, vcc
	v_xor_b32_e32 v70, v70, v78
	v_sub_co_u32_e32 v78, vcc, v70, v78
.LBB0_74:
	s_andn2_saveexec_b64 s[14:15], s[22:23]
	s_cbranch_execz .LBB0_76
	v_rcp_iflag_f32_e32 v70, v100
	s_sub_i32 s22, 0, s78
	v_mul_f32_e32 v70, 0x4f7ffffe, v70
	v_cvt_u32_f32_e32 v70, v70
	v_mul_lo_u32 v78, s22, v70
	v_mul_hi_u32 v78, v70, v78
	v_add_u32_e32 v70, v70, v78
	v_mul_hi_u32 v70, v90, v70
	v_mul_lo_u32 v78, v70, s78
	v_sub_u32_e32 v78, v90, v78
	v_add_u32_e32 v79, 1, v70
	v_subrev_u32_e32 v88, s78, v78
	v_cmp_le_u32_e32 vcc, s78, v78
	s_nop 1
	v_cndmask_b32_e32 v78, v78, v88, vcc
	v_cndmask_b32_e32 v70, v70, v79, vcc
	v_add_u32_e32 v79, 1, v70
	v_cmp_le_u32_e32 vcc, s78, v78
	s_nop 1
	v_cndmask_b32_e32 v78, v70, v79, vcc
.LBB0_76:
	s_or_b64 exec, exec, s[14:15]
	v_lshl_add_u64 v[80:81], v[80:81], 0, s[62:63]
	v_or_b32_e32 v70, s79, v81
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[28:29], exec, s[14:15]
	s_cbranch_execz .LBB0_78
	s_add_u32 s14, s78, s82
	s_mov_b32 s83, s82
	s_addc_u32 s15, s79, s82
	s_xor_b64 s[56:57], s[14:15], s[82:83]
	v_cvt_f32_u32_e32 v70, s56
	v_cvt_f32_u32_e32 v79, s57
	s_sub_u32 s14, 0, s56
	s_subb_u32 s15, 0, s57
	v_ashrrev_i32_e32 v94, 31, v81
	v_fmac_f32_e32 v70, 0x4f800000, v79
	v_rcp_f32_e32 v70, v70
	v_mov_b32_e32 v95, v94
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[94:95]
	v_xor_b32_e32 v88, v80, v94
	v_mul_f32_e32 v70, 0x5f7ffffc, v70
	v_mul_f32_e32 v79, 0x2f800000, v70
	v_trunc_f32_e32 v79, v79
	v_fmac_f32_e32 v70, 0xcf800000, v79
	v_cvt_u32_f32_e32 v79, v79
	v_cvt_u32_f32_e32 v70, v70
	v_readfirstlane_b32 s22, v79
	v_readfirstlane_b32 s23, v70
	s_mul_i32 s68, s14, s22
	s_mul_hi_u32 s73, s14, s23
	s_mul_i32 s69, s15, s23
	s_add_i32 s68, s73, s68
	s_add_i32 s68, s68, s69
	s_mul_i32 s75, s14, s23
	s_mul_i32 s73, s23, s68
	s_mul_hi_u32 s81, s23, s75
	s_mul_hi_u32 s69, s23, s68
	s_add_u32 s73, s81, s73
	s_addc_u32 s69, 0, s69
	s_mul_hi_u32 s83, s22, s75
	s_mul_i32 s75, s22, s75
	s_add_u32 s73, s73, s75
	s_mul_hi_u32 s81, s22, s68
	s_addc_u32 s69, s69, s83
	s_addc_u32 s73, s81, 0
	s_mul_i32 s68, s22, s68
	s_add_u32 s68, s69, s68
	s_addc_u32 s69, 0, s73
	s_add_u32 s23, s23, s68
	s_addc_u32 s22, s22, s69
	s_mul_i32 s68, s14, s22
	s_mul_hi_u32 s69, s14, s23
	s_add_i32 s68, s69, s68
	s_mul_i32 s15, s15, s23
	s_add_i32 s68, s68, s15
	s_mul_i32 s14, s14, s23
	s_mul_hi_u32 s69, s22, s14
	s_mul_i32 s73, s22, s14
	s_mul_i32 s81, s23, s68
	s_mul_hi_u32 s14, s23, s14
	s_mul_hi_u32 s75, s23, s68
	s_add_u32 s14, s14, s81
	s_addc_u32 s75, 0, s75
	s_add_u32 s14, s14, s73
	s_mul_hi_u32 s15, s22, s68
	s_addc_u32 s14, s75, s69
	s_addc_u32 s15, s15, 0
	s_mul_i32 s68, s22, s68
	s_add_u32 s14, s14, s68
	s_addc_u32 s15, 0, s15
	s_add_u32 s23, s23, s14
	s_addc_u32 s22, s22, s15
	v_xor_b32_e32 v79, v81, v94
	v_mad_u64_u32 v[80:81], s[14:15], v88, s22, 0
	v_mul_hi_u32 v70, v88, s23
	v_lshl_add_u64 v[80:81], v[70:71], 0, v[80:81]
	v_mad_u64_u32 v[140:141], s[14:15], v79, s23, 0
	v_add_co_u32_e32 v70, vcc, v80, v140
	v_mad_u64_u32 v[90:91], s[14:15], v79, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v80, vcc, v81, v141, vcc
	v_mov_b32_e32 v81, s74
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[90:91]
	v_mul_lo_u32 v70, s57, v80
	v_mul_lo_u32 v90, s56, v81
	v_mad_u64_u32 v[80:81], s[14:15], s56, v80, 0
	v_add3_u32 v70, v81, v90, v70
	v_sub_u32_e32 v81, v79, v70
	v_mov_b32_e32 v90, s57
	v_sub_co_u32_e32 v80, vcc, v88, v80
	s_nop 1
	v_subb_co_u32_e64 v81, s[14:15], v81, v90, vcc
	v_subrev_co_u32_e64 v88, s[14:15], s56, v80
	v_subb_co_u32_e32 v70, vcc, v79, v70, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[22:23], 0, v81, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s57, v91
	v_subb_co_u32_e64 v81, s[14:15], v81, v90, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s56, v88
	v_subrev_co_u32_e64 v90, s[14:15], s56, v88
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s57, v91
	v_subbrev_co_u32_e64 v81, s[14:15], 0, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v93, v93, v95, s[22:23]
	v_cmp_le_u32_e32 vcc, s57, v70
	v_cmp_ne_u32_e64 s[14:15], 0, v93
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s56, v80
	v_cndmask_b32_e64 v81, v91, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s57, v70
	s_nop 1
	v_cndmask_b32_e32 v79, v79, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v79
	v_cndmask_b32_e64 v79, v88, v90, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v79, v80, v79, vcc
	v_cndmask_b32_e32 v70, v70, v81, vcc
	v_xor_b32_e32 v79, v79, v94
	v_xor_b32_e32 v70, v70, v94
	v_sub_co_u32_e32 v90, vcc, v79, v94
	s_nop 1
	v_subb_co_u32_e32 v91, vcc, v70, v94, vcc
.LBB0_78:
	s_andn2_saveexec_b64 s[14:15], s[28:29]
	s_cbranch_execz .LBB0_80
	v_rcp_iflag_f32_e32 v70, v100
	s_sub_i32 s22, 0, s78
	v_mul_f32_e32 v70, 0x4f7ffffe, v70
	v_cvt_u32_f32_e32 v70, v70
	v_mul_lo_u32 v79, s22, v70
	v_mul_hi_u32 v79, v70, v79
	v_add_u32_e32 v70, v70, v79
	v_mul_hi_u32 v70, v80, v70
	v_mul_lo_u32 v70, v70, s78
	v_sub_u32_e32 v70, v80, v70
	v_subrev_u32_e32 v79, s78, v70
	v_cmp_le_u32_e32 vcc, s78, v70
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v79, vcc
	v_subrev_u32_e32 v79, s78, v70
	v_cmp_le_u32_e32 vcc, s78, v70
	s_nop 1
	v_cndmask_b32_e32 v70, v70, v79, vcc
	v_mov_b64_e32 v[90:91], v[70:71]
.LBB0_80:
	s_or_b64 exec, exec, s[14:15]
	v_ashrrev_i32_e32 v70, 31, v89
	v_xor_b32_e32 v70, v78, v70
	v_ashrrev_i32_e32 v78, 31, v91
	v_mul_lo_u32 v70, v70, s66
	v_and_b32_e32 v78, s78, v78
	v_add3_u32 v70, v70, v90, v78
	buffer_load_ubyte v81, v70, s[8:11], 0 offen
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	s_mov_b64 s[14:15], 0x300
	v_lshl_add_u64 v[78:79], v[86:87], 0, s[14:15]
	v_add_u32_e32 v70, 0x180, v84
	v_cmp_gt_i64_e32 vcc, s[30:31], v[78:79]
	v_add_u32_e32 v80, v70, v30
	s_and_b64 s[14:15], s[2:3], vcc
	s_mov_b32 m0, s65
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[136:139], v[12:15], v[4:7], v33, v31 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e64 v78, v29, v80, s[14:15]
	s_and_b64 s[14:15], s[26:27], vcc
	buffer_load_dwordx4 v78, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[128:131], v[12:15], v[8:11], v33, v31 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v12, v70, v34
	v_cndmask_b32_e64 v12, v29, v12, s[14:15]
	s_mov_b32 m0, s0
	s_and_b64 s[14:15], s[18:19], vcc
	buffer_load_dwordx4 v12, s[40:43], 0 offen lds
	v_add_u32_e32 v12, v70, v36
	v_cndmask_b32_e64 v12, v29, v12, s[14:15]
	s_mov_b32 m0, s1
	s_and_b64 vcc, s[20:21], vcc
	buffer_load_dwordx4 v12, s[40:43], 0 offen lds
	v_add_u32_e32 v12, v70, v38
	v_cndmask_b32_e32 v12, v29, v12, vcc
	s_mov_b32 m0, s58
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[132:135], v[16:19], v[4:7], v33, v107 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v12, s[40:43], 0 offen lds
	s_mov_b64 s[14:15], 0x6000
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[124:127], v[16:19], v[8:11], v33, v107 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 v[132:135], v52
	ds_read_b128 v[128:131], v54 offset:2048
	ds_read_b128 v[124:127], v52 offset:2048
	ds_read_b128 v[88:91], v54
	ds_read_b32 v80, v50 offset:49152
	v_sub_co_u32_e32 v10, vcc, 0xffff9fff, v82
	v_lshl_add_u64 v[8:9], v[82:83], 0, s[14:15]
	s_nop 0
	v_subb_co_u32_e32 v11, vcc, -1, v83, vcc
	v_cmp_gt_i32_e32 vcc, 0, v9
	s_nop 1
	v_cndmask_b32_e32 v13, v9, v11, vcc
	v_cndmask_b32_e32 v12, v8, v10, vcc
	v_or_b32_e32 v10, s31, v13
	v_cmp_ne_u32_e32 vcc, 0, v10
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[22:23], exec, s[14:15]
	s_cbranch_execz .LBB0_82
	s_add_u32 s14, s30, s80
	s_mov_b32 s81, s80
	s_addc_u32 s15, s31, s80
	s_xor_b64 s[28:29], s[14:15], s[80:81]
	v_cvt_f32_u32_e32 v10, s28
	v_cvt_f32_u32_e32 v11, s29
	s_sub_u32 s14, 0, s28
	s_subb_u32 s15, 0, s29
	v_fmac_f32_e32 v10, 0x4f800000, v11
	v_rcp_f32_e32 v10, v10
	s_nop 0
	v_mul_f32_e32 v10, 0x5f7ffffc, v10
	v_mul_f32_e32 v11, 0x2f800000, v10
	v_trunc_f32_e32 v11, v11
	v_fmac_f32_e32 v10, 0xcf800000, v11
	v_cvt_u32_f32_e32 v11, v11
	v_cvt_u32_f32_e32 v10, v10
	v_readfirstlane_b32 s56, v11
	v_readfirstlane_b32 s57, v10
	s_mul_i32 s68, s14, s56
	s_mul_hi_u32 s73, s14, s57
	s_mul_i32 s69, s15, s57
	s_add_i32 s68, s73, s68
	s_add_i32 s68, s68, s69
	s_mul_i32 s75, s14, s57
	s_mul_i32 s73, s57, s68
	s_mul_hi_u32 s81, s57, s75
	s_mul_hi_u32 s69, s57, s68
	s_add_u32 s73, s81, s73
	s_addc_u32 s69, 0, s69
	s_mul_hi_u32 s83, s56, s75
	s_mul_i32 s75, s56, s75
	s_add_u32 s73, s73, s75
	s_mul_hi_u32 s81, s56, s68
	s_addc_u32 s69, s69, s83
	s_addc_u32 s73, s81, 0
	s_mul_i32 s68, s56, s68
	s_add_u32 s68, s69, s68
	s_addc_u32 s69, 0, s73
	s_add_u32 s57, s57, s68
	s_addc_u32 s56, s56, s69
	s_mul_i32 s68, s14, s56
	s_mul_hi_u32 s69, s14, s57
	s_add_i32 s68, s69, s68
	s_mul_i32 s15, s15, s57
	s_add_i32 s68, s68, s15
	s_mul_i32 s14, s14, s57
	s_mul_hi_u32 s69, s56, s14
	s_mul_i32 s73, s56, s14
	s_mul_i32 s81, s57, s68
	s_mul_hi_u32 s14, s57, s14
	s_mul_hi_u32 s75, s57, s68
	s_add_u32 s14, s14, s81
	s_addc_u32 s75, 0, s75
	s_add_u32 s14, s14, s73
	s_mul_hi_u32 s15, s56, s68
	s_addc_u32 s14, s75, s69
	s_addc_u32 s15, s15, 0
	s_mul_i32 s68, s56, s68
	s_add_u32 s14, s14, s68
	v_ashrrev_i32_e32 v10, 31, v13
	s_addc_u32 s15, 0, s15
	v_mov_b32_e32 v11, v10
	s_add_u32 s57, s57, s14
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[10:11]
	s_addc_u32 s56, s56, s15
	v_xor_b32_e32 v31, v12, v10
	v_xor_b32_e32 v11, v13, v10
	v_mad_u64_u32 v[12:13], s[14:15], v31, s56, 0
	v_mul_hi_u32 v70, v31, s57
	v_lshl_add_u64 v[12:13], v[70:71], 0, v[12:13]
	v_mad_u64_u32 v[78:79], s[14:15], v11, s57, 0
	v_add_co_u32_e32 v12, vcc, v12, v78
	v_mad_u64_u32 v[14:15], s[14:15], v11, s56, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v13, v79, vcc
	v_mov_b32_e32 v13, s74
	s_nop 0
	v_addc_co_u32_e32 v15, vcc, 0, v15, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[14:15]
	v_mul_lo_u32 v33, s29, v12
	v_mul_lo_u32 v70, s28, v13
	v_mad_u64_u32 v[14:15], s[14:15], s28, v12, 0
	v_add3_u32 v33, v15, v70, v33
	v_sub_u32_e32 v15, v11, v33
	v_mov_b32_e32 v70, s29
	v_sub_co_u32_e32 v31, vcc, v31, v14
	v_lshl_add_u64 v[78:79], v[12:13], 0, 1
	s_nop 0
	v_subb_co_u32_e64 v14, s[14:15], v15, v70, vcc
	v_subrev_co_u32_e64 v15, s[14:15], s28, v31
	v_subb_co_u32_e32 v11, vcc, v11, v33, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v14, s[14:15], 0, v14, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s29, v14
	v_cmp_le_u32_e32 vcc, s29, v11
	v_xor_b32_e32 v10, s80, v10
	v_cndmask_b32_e64 v70, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v15
	s_nop 1
	v_cndmask_b32_e64 v15, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v14
	s_nop 1
	v_cndmask_b32_e64 v70, v70, v15, s[14:15]
	v_lshl_add_u64 v[14:15], v[12:13], 0, 2
	v_cndmask_b32_e64 v13, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v31
	s_nop 1
	v_cndmask_b32_e64 v15, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v11
	s_nop 1
	v_cndmask_b32_e32 v11, v13, v15, vcc
	v_cmp_ne_u32_e32 vcc, 0, v70
	s_nop 1
	v_cndmask_b32_e32 v13, v78, v14, vcc
	v_cmp_ne_u32_e32 vcc, 0, v11
	s_nop 1
	v_cndmask_b32_e32 v11, v12, v13, vcc
	v_xor_b32_e32 v11, v11, v10
	v_sub_co_u32_e32 v10, vcc, v11, v10
.LBB0_82:
	s_andn2_saveexec_b64 s[14:15], s[22:23]
	s_cbranch_execz .LBB0_84
	v_cvt_f32_u32_e32 v10, s30
	s_sub_i32 s22, 0, s30
	v_rcp_iflag_f32_e32 v10, v10
	s_nop 0
	v_mul_f32_e32 v10, 0x4f7ffffe, v10
	v_cvt_u32_f32_e32 v10, v10
	v_mul_lo_u32 v11, s22, v10
	v_mul_hi_u32 v11, v10, v11
	v_add_u32_e32 v10, v10, v11
	v_mul_hi_u32 v10, v12, v10
	v_mul_lo_u32 v11, v10, s30
	v_sub_u32_e32 v11, v12, v11
	v_add_u32_e32 v13, 1, v10
	v_subrev_u32_e32 v12, s30, v11
	v_cmp_le_u32_e32 vcc, s30, v11
	s_nop 1
	v_cndmask_b32_e32 v11, v11, v12, vcc
	v_cndmask_b32_e32 v10, v10, v13, vcc
	v_add_u32_e32 v12, 1, v10
	v_cmp_le_u32_e32 vcc, s30, v11
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v12, vcc
.LBB0_84:
	s_or_b64 exec, exec, s[14:15]
	v_or_b32_e32 v11, s31, v9
	v_cmp_ne_u32_e32 vcc, 0, v11
	v_ashrrev_i32_e32 v12, 31, v9
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[28:29], exec, s[14:15]
	s_cbranch_execz .LBB0_86
	s_add_u32 s14, s30, s80
	s_mov_b32 s81, s80
	s_addc_u32 s15, s31, s80
	s_xor_b64 s[56:57], s[14:15], s[80:81]
	v_cvt_f32_u32_e32 v11, s56
	v_cvt_f32_u32_e32 v13, s57
	s_sub_u32 s14, 0, s56
	s_subb_u32 s15, 0, s57
	v_fmac_f32_e32 v11, 0x4f800000, v13
	v_rcp_f32_e32 v11, v11
	s_nop 0
	v_mul_f32_e32 v11, 0x5f7ffffc, v11
	v_mul_f32_e32 v13, 0x2f800000, v11
	v_trunc_f32_e32 v13, v13
	v_fmac_f32_e32 v11, 0xcf800000, v13
	v_cvt_u32_f32_e32 v13, v13
	v_cvt_u32_f32_e32 v11, v11
	v_readfirstlane_b32 s22, v13
	v_readfirstlane_b32 s23, v11
	s_mul_i32 s68, s14, s22
	s_mul_hi_u32 s73, s14, s23
	s_mul_i32 s69, s15, s23
	s_add_i32 s68, s73, s68
	s_add_i32 s68, s68, s69
	s_mul_i32 s75, s14, s23
	s_mul_i32 s73, s23, s68
	s_mul_hi_u32 s81, s23, s75
	s_mul_hi_u32 s69, s23, s68
	s_add_u32 s73, s81, s73
	s_addc_u32 s69, 0, s69
	s_mul_hi_u32 s83, s22, s75
	s_mul_i32 s75, s22, s75
	s_add_u32 s73, s73, s75
	s_mul_hi_u32 s81, s22, s68
	s_addc_u32 s69, s69, s83
	s_addc_u32 s73, s81, 0
	s_mul_i32 s68, s22, s68
	s_add_u32 s68, s69, s68
	s_addc_u32 s69, 0, s73
	s_add_u32 s23, s23, s68
	s_addc_u32 s22, s22, s69
	s_mul_i32 s68, s14, s22
	s_mul_hi_u32 s69, s14, s23
	s_add_i32 s68, s69, s68
	s_mul_i32 s15, s15, s23
	s_add_i32 s68, s68, s15
	s_mul_i32 s14, s14, s23
	s_mul_hi_u32 s69, s22, s14
	s_mul_i32 s73, s22, s14
	s_mul_i32 s81, s23, s68
	s_mul_hi_u32 s14, s23, s14
	s_mul_hi_u32 s75, s23, s68
	s_add_u32 s14, s14, s81
	s_addc_u32 s75, 0, s75
	s_add_u32 s14, s14, s73
	s_mul_hi_u32 s15, s22, s68
	s_addc_u32 s14, s75, s69
	s_addc_u32 s15, s15, 0
	s_mul_i32 s68, s22, s68
	s_add_u32 s14, s14, s68
	s_addc_u32 s15, 0, s15
	v_mov_b32_e32 v13, v12
	s_add_u32 s23, s23, s14
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[12:13]
	s_addc_u32 s22, s22, s15
	v_xor_b32_e32 v13, v8, v12
	v_xor_b32_e32 v11, v9, v12
	v_mad_u64_u32 v[8:9], s[14:15], v13, s22, 0
	v_mul_hi_u32 v70, v13, s23
	v_lshl_add_u64 v[8:9], v[70:71], 0, v[8:9]
	v_mad_u64_u32 v[78:79], s[14:15], v11, s23, 0
	v_add_co_u32_e32 v8, vcc, v8, v78
	v_mad_u64_u32 v[14:15], s[14:15], v11, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v8, vcc, v9, v79, vcc
	v_mov_b32_e32 v9, s74
	s_nop 0
	v_addc_co_u32_e32 v15, vcc, 0, v15, vcc
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[14:15]
	v_mul_lo_u32 v14, s57, v8
	v_mul_lo_u32 v15, s56, v9
	v_mad_u64_u32 v[8:9], s[14:15], s56, v8, 0
	v_add3_u32 v9, v9, v15, v14
	v_sub_u32_e32 v14, v11, v9
	v_mov_b32_e32 v15, s57
	v_sub_co_u32_e32 v8, vcc, v13, v8
	s_nop 1
	v_subb_co_u32_e64 v13, s[14:15], v14, v15, vcc
	v_subrev_co_u32_e64 v14, s[14:15], s56, v8
	v_subb_co_u32_e32 v9, vcc, v11, v9, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v31, s[22:23], 0, v13, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s57, v31
	v_subb_co_u32_e64 v13, s[14:15], v13, v15, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v33, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s56, v14
	v_subrev_co_u32_e64 v15, s[14:15], s56, v14
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s57, v31
	v_subbrev_co_u32_e64 v13, s[14:15], 0, v13, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v33, v33, v70, s[22:23]
	v_cmp_le_u32_e32 vcc, s57, v9
	v_cmp_ne_u32_e64 s[14:15], 0, v33
	s_nop 0
	v_cndmask_b32_e64 v11, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s56, v8
	v_cndmask_b32_e64 v13, v31, v13, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s57, v9
	s_nop 1
	v_cndmask_b32_e32 v11, v11, v31, vcc
	v_cmp_ne_u32_e32 vcc, 0, v11
	v_cndmask_b32_e64 v11, v14, v15, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v8, v8, v11, vcc
	v_cndmask_b32_e32 v9, v9, v13, vcc
	v_xor_b32_e32 v8, v8, v12
	v_xor_b32_e32 v9, v9, v12
	v_sub_co_u32_e32 v14, vcc, v8, v12
	s_nop 1
	v_subb_co_u32_e32 v15, vcc, v9, v12, vcc
.LBB0_86:
	s_andn2_saveexec_b64 s[14:15], s[28:29]
	s_cbranch_execz .LBB0_39
	v_cvt_f32_u32_e32 v9, s30
	s_sub_i32 s22, 0, s30
	v_rcp_iflag_f32_e32 v9, v9
	s_nop 0
	v_mul_f32_e32 v9, 0x4f7ffffe, v9
	v_cvt_u32_f32_e32 v9, v9
	v_mul_lo_u32 v11, s22, v9
	v_mul_hi_u32 v11, v9, v11
	v_add_u32_e32 v9, v9, v11
	v_mul_hi_u32 v9, v8, v9
	v_mul_lo_u32 v9, v9, s30
	v_sub_u32_e32 v8, v8, v9
	v_subrev_u32_e32 v9, s30, v8
	v_cmp_le_u32_e32 vcc, s30, v8
	s_nop 1
	v_cndmask_b32_e32 v8, v8, v9, vcc
	v_subrev_u32_e32 v9, s30, v8
	v_cmp_le_u32_e32 vcc, s30, v8
	s_nop 1
	v_cndmask_b32_e32 v70, v8, v9, vcc
	v_mov_b64_e32 v[14:15], v[70:71]
	s_branch .LBB0_39
.LBB0_88:
	v_readlane_b32 s86, v144, 28
.LBB0_89:
	v_readlane_b32 s0, v144, 26
	s_lshr_b32 s1, s0, 23
	s_add_u32 s2, s76, s1
	s_addc_u32 s3, s77, 0
	v_readlane_b32 s14, v144, 16
	s_ashr_i64 s[2:3], s[2:3], 9
	v_readlane_b32 s15, v144, 17
	s_xor_b64 s[18:19], s[2:3], s[14:15]
	s_lshl_b64 s[2:3], s[18:19], 12
	v_or_b32_e32 v29, s3, v25
	v_or_b32_e32 v28, s2, v44
	s_movk_i32 s2, 0xf800
	v_lshl_add_u64 v[30:31], v[28:29], 0, v[68:69]
	s_mov_b32 s3, -1
	v_lshl_add_u64 v[32:33], v[30:31], 0, s[2:3]
	s_add_u32 s2, s44, s72
	s_mov_b32 s73, s72
	s_addc_u32 s3, s45, s72
	s_xor_b64 s[20:21], s[2:3], s[72:73]
	v_cvt_f32_u32_e32 v36, s20
	v_cvt_f32_u32_e32 v38, s21
	s_movk_i32 s1, 0x7ff
	v_sub_co_u32_e32 v25, vcc, s1, v30
	v_fmac_f32_e32 v36, 0x4f800000, v38
	s_nop 0
	v_subb_co_u32_e32 v28, vcc, 0, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v33
	s_sub_u32 s1, 0, s20
	s_subb_u32 s2, 0, s21
	v_cndmask_b32_e32 v29, v33, v28, vcc
	v_cndmask_b32_e32 v28, v32, v25, vcc
	v_rcp_f32_e32 v25, v36
	v_ashrrev_i32_e32 v34, 31, v29
	v_mov_b32_e32 v35, v34
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[34:35]
	v_mul_f32_e32 v25, 0x5f7ffffc, v25
	v_xor_b32_e32 v36, v28, v34
	v_mul_f32_e32 v28, 0x2f800000, v25
	v_trunc_f32_e32 v28, v28
	v_fmac_f32_e32 v25, 0xcf800000, v28
	v_cvt_u32_f32_e32 v25, v25
	v_cvt_u32_f32_e32 v28, v28
	v_xor_b32_e32 v35, v29, v34
	v_mov_b32_e32 v29, 0
	v_readfirstlane_b32 s3, v25
	v_readfirstlane_b32 s16, v28
	s_mul_hi_u32 s15, s1, s3
	s_mul_i32 s22, s1, s16
	s_mul_i32 s14, s2, s3
	s_add_i32 s15, s15, s22
	s_add_i32 s15, s15, s14
	s_mul_i32 s23, s1, s3
	s_mul_i32 s22, s3, s15
	s_mul_hi_u32 s24, s3, s23
	s_mul_hi_u32 s14, s3, s15
	s_add_u32 s22, s24, s22
	s_addc_u32 s14, 0, s14
	s_mul_hi_u32 s25, s16, s23
	s_mul_i32 s23, s16, s23
	s_add_u32 s22, s22, s23
	s_mul_hi_u32 s24, s16, s15
	s_addc_u32 s14, s14, s25
	s_addc_u32 s22, s24, 0
	s_mul_i32 s15, s16, s15
	s_add_u32 s14, s14, s15
	s_addc_u32 s15, 0, s22
	s_add_u32 s3, s3, s14
	s_addc_u32 s14, s16, s15
	s_mul_i32 s15, s1, s14
	s_mul_hi_u32 s16, s1, s3
	s_add_i32 s15, s16, s15
	s_mul_i32 s2, s2, s3
	s_add_i32 s15, s15, s2
	s_mul_i32 s1, s1, s3
	s_mul_hi_u32 s16, s14, s1
	s_mul_i32 s22, s14, s1
	s_mul_i32 s24, s3, s15
	s_mul_hi_u32 s1, s3, s1
	s_mul_hi_u32 s23, s3, s15
	s_add_u32 s1, s1, s24
	s_addc_u32 s23, 0, s23
	s_add_u32 s1, s1, s22
	s_mul_hi_u32 s2, s14, s15
	s_addc_u32 s1, s23, s16
	s_addc_u32 s2, s2, 0
	s_mul_i32 s15, s14, s15
	s_add_u32 s1, s1, s15
	s_addc_u32 s2, 0, s2
	s_add_u32 s1, s3, s1
	s_addc_u32 s16, s14, s2
	v_mad_u64_u32 v[38:39], s[2:3], v36, s16, 0
	v_mul_hi_u32 v28, v36, s1
	v_lshl_add_u64 v[38:39], v[28:29], 0, v[38:39]
	v_mad_u64_u32 v[68:69], s[2:3], v35, s1, 0
	v_add_co_u32_e32 v25, vcc, v38, v68
	s_mov_b32 s0, 0
	v_mad_u64_u32 v[40:41], s[2:3], v35, s16, 0
	v_addc_co_u32_e32 v38, vcc, v39, v69, vcc
	v_mov_b32_e32 v39, s0
	s_nop 0
	v_addc_co_u32_e32 v41, vcc, 0, v41, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[40:41]
	v_mul_lo_u32 v25, s21, v38
	v_mul_lo_u32 v28, s20, v39
	v_mad_u64_u32 v[40:41], s[2:3], s20, v38, 0
	v_add3_u32 v28, v41, v28, v25
	v_sub_u32_e32 v39, v35, v28
	v_mov_b32_e32 v25, s21
	v_sub_co_u32_e32 v36, vcc, v36, v40
	v_xor_b32_e32 v34, s72, v34
	s_nop 0
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, vcc
	v_subrev_co_u32_e64 v40, s[2:3], s20, v36
	v_subb_co_u32_e32 v28, vcc, v35, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v39, s[2:3], 0, v39, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v39
	v_cmp_le_u32_e32 vcc, s21, v28
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_cndmask_b32_e64 v41, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v40
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v36
	v_cndmask_b32_e64 v40, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v39
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v39, v41, v40, s[2:3]
	v_add_u32_e32 v40, 2, v38
	v_add_u32_e32 v41, 1, v38
	v_cmp_ne_u32_e64 s[2:3], 0, v39
	v_cndmask_b32_e32 v28, v35, v36, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v39, v41, v40, s[2:3]
	s_barrier
	v_cndmask_b32_e32 v28, v38, v39, vcc
	v_xor_b32_e32 v28, v28, v34
	v_sub_u32_e32 v28, v28, v34
	v_ashrrev_i32_e32 v34, 31, v33
	v_mov_b32_e32 v35, v34
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[34:35]
	v_xor_b32_e32 v44, v32, v34
	v_xad_u32 v36, v28, v34, v37
	v_xor_b32_e32 v35, v33, v34
	v_mad_u64_u32 v[32:33], s[2:3], v44, s16, 0
	v_mul_hi_u32 v28, v44, s1
	v_lshl_add_u64 v[32:33], v[28:29], 0, v[32:33]
	v_mad_u64_u32 v[40:41], s[2:3], v35, s1, 0
	v_add_co_u32_e32 v28, vcc, v32, v40
	v_mad_u64_u32 v[38:39], s[2:3], v35, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v33, v41, vcc
	v_mov_b32_e32 v33, s0
	s_nop 0
	v_addc_co_u32_e32 v39, vcc, 0, v39, vcc
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[38:39]
	v_mul_lo_u32 v28, s21, v32
	v_mul_lo_u32 v38, s20, v33
	v_mad_u64_u32 v[32:33], s[2:3], s20, v32, 0
	v_add3_u32 v28, v33, v38, v28
	v_sub_u32_e32 v33, v35, v28
	v_sub_co_u32_e32 v32, vcc, v44, v32
	s_nop 1
	v_subb_co_u32_e64 v33, s[2:3], v33, v25, vcc
	v_subrev_co_u32_e64 v38, s[2:3], s20, v32
	v_subb_co_u32_e32 v28, vcc, v35, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v39, s[14:15], 0, v33, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v39
	v_subb_co_u32_e64 v33, s[2:3], v33, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v40, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v38
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v41, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v39
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v32
	v_cndmask_b32_e64 v40, v40, v41, s[14:15]
	v_subrev_co_u32_e64 v41, s[2:3], s20, v38
	s_nop 1
	v_subbrev_co_u32_e64 v33, s[2:3], 0, v33, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v40
	s_nop 1
	v_cndmask_b32_e64 v33, v39, v33, s[2:3]
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v35, v35, v39, vcc
	v_cmp_ne_u32_e32 vcc, 0, v35
	s_nop 1
	v_cndmask_b32_e32 v28, v28, v33, vcc
	v_cndmask_b32_e64 v33, v38, v41, s[2:3]
	v_cndmask_b32_e32 v32, v32, v33, vcc
	v_xor_b32_e32 v32, v32, v34
	v_xor_b32_e32 v28, v28, v34
	v_sub_co_u32_e32 v32, vcc, v32, v34
	v_mul_lo_u32 v33, v36, s33
	s_nop 0
	v_subb_co_u32_e32 v28, vcc, v28, v34, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v38, v28, v32, v33
	v_lshlrev_b32_e32 v28, 8, v98
	v_and_b32_e32 v36, 0x200, v28
	v_or_b32_e32 v34, 0xfffff800, v36
	v_ashrrev_i32_e32 v35, 31, v34
	v_lshl_add_u64 v[32:33], v[30:31], 0, v[66:67]
	v_lshl_add_u64 v[34:35], v[32:33], 0, v[34:35]
	v_sub_co_u32_e32 v28, vcc, -2, v34
	v_lshl_add_u64 v[40:41], v[34:35], 0, 1
	s_nop 0
	v_subb_co_u32_e32 v39, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v41
	s_nop 1
	v_cndmask_b32_e32 v67, v41, v39, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v40, v28, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v44, v66, v68
	v_xor_b32_e32 v39, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v44, s16, 0
	v_mul_hi_u32 v28, v44, s1
	v_lshl_add_u64 v[66:67], v[28:29], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v39, s1, 0
	v_add_co_u32_e32 v28, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v39, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s0
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v28, s21, v66
	v_mul_lo_u32 v47, s20, v67
	v_mad_u64_u32 v[70:71], s[2:3], s20, v66, 0
	v_add3_u32 v28, v71, v47, v28
	v_sub_u32_e32 v47, v39, v28
	v_sub_co_u32_e32 v44, vcc, v44, v70
	s_nop 1
	v_subb_co_u32_e64 v47, s[2:3], v47, v25, vcc
	v_subrev_co_u32_e64 v51, s[2:3], s20, v44
	v_subb_co_u32_e32 v28, vcc, v39, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[2:3], 0, v47, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v47
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v53, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v51
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v44
	v_cndmask_b32_e64 v51, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v47
	v_cndmask_b32_e64 v44, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v47, v53, v51, s[2:3]
	v_add_u32_e32 v51, 2, v66
	v_add_u32_e32 v53, 1, v66
	v_cmp_ne_u32_e64 s[2:3], 0, v47
	v_cndmask_b32_e32 v28, v39, v44, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v47, v53, v51, s[2:3]
	v_xor_b32_e32 v39, s72, v68
	v_cndmask_b32_e32 v28, v66, v47, vcc
	v_ashrrev_i32_e32 v66, 31, v41
	v_mov_b32_e32 v67, v66
	v_xor_b32_e32 v28, v28, v39
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[66:67]
	v_sub_u32_e32 v28, v28, v39
	v_xor_b32_e32 v47, v40, v66
	v_xad_u32 v39, v28, v66, v37
	v_xor_b32_e32 v44, v41, v66
	v_mad_u64_u32 v[40:41], s[2:3], v47, s16, 0
	v_mul_hi_u32 v28, v47, s1
	v_lshl_add_u64 v[40:41], v[28:29], 0, v[40:41]
	v_mad_u64_u32 v[70:71], s[2:3], v44, s1, 0
	v_add_co_u32_e32 v28, vcc, v40, v70
	v_mad_u64_u32 v[68:69], s[2:3], v44, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v40, vcc, v41, v71, vcc
	v_mov_b32_e32 v41, s0
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[68:69]
	v_mul_lo_u32 v28, s21, v40
	v_mul_lo_u32 v51, s20, v41
	v_mad_u64_u32 v[40:41], s[2:3], s20, v40, 0
	v_add3_u32 v28, v41, v51, v28
	v_sub_u32_e32 v41, v44, v28
	v_sub_co_u32_e32 v40, vcc, v47, v40
	v_mul_lo_u32 v39, v39, s33
	s_nop 0
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, vcc
	v_subrev_co_u32_e64 v47, s[2:3], s20, v40
	v_subb_co_u32_e32 v28, vcc, v44, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v51, s[14:15], 0, v41, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v51
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v53, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v47
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v55, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v51
	v_cndmask_b32_e64 v44, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v40
	v_cndmask_b32_e64 v53, v53, v55, s[14:15]
	v_subrev_co_u32_e64 v55, s[2:3], s20, v47
	s_nop 1
	v_subbrev_co_u32_e64 v41, s[2:3], 0, v41, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v53
	s_nop 1
	v_cndmask_b32_e64 v41, v51, v41, s[2:3]
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v44, v44, v51, vcc
	v_cmp_ne_u32_e32 vcc, 0, v44
	s_nop 1
	v_cndmask_b32_e32 v28, v28, v41, vcc
	v_cndmask_b32_e64 v41, v47, v55, s[2:3]
	v_cndmask_b32_e32 v40, v40, v41, vcc
	v_xor_b32_e32 v40, v40, v66
	v_xor_b32_e32 v28, v28, v66
	v_sub_co_u32_e32 v40, vcc, v40, v66
	s_nop 1
	v_subb_co_u32_e32 v28, vcc, v28, v66, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v39, v28, v40, v39
	v_sub_co_u32_e32 v28, vcc, -3, v34
	v_lshl_add_u64 v[40:41], v[34:35], 0, 2
	s_nop 0
	v_subb_co_u32_e32 v44, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v41
	s_nop 1
	v_cndmask_b32_e32 v67, v41, v44, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v40, v28, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v47, v66, v68
	v_xor_b32_e32 v44, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v47, s16, 0
	v_mul_hi_u32 v28, v47, s1
	v_lshl_add_u64 v[66:67], v[28:29], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v44, s1, 0
	v_add_co_u32_e32 v28, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v44, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s0
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v28, s21, v66
	v_mul_lo_u32 v51, s20, v67
	v_mad_u64_u32 v[70:71], s[2:3], s20, v66, 0
	v_add3_u32 v28, v71, v51, v28
	v_sub_u32_e32 v51, v44, v28
	v_sub_co_u32_e32 v47, vcc, v47, v70
	s_nop 1
	v_subb_co_u32_e64 v51, s[2:3], v51, v25, vcc
	v_subrev_co_u32_e64 v53, s[2:3], s20, v47
	v_subb_co_u32_e32 v28, vcc, v44, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v51, s[2:3], 0, v51, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v51
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v55, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v53
	v_cndmask_b32_e64 v44, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v47
	v_cndmask_b32_e64 v53, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v51
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v51, v55, v53, s[2:3]
	v_add_u32_e32 v53, 2, v66
	v_add_u32_e32 v55, 1, v66
	v_cmp_ne_u32_e64 s[2:3], 0, v51
	v_cndmask_b32_e32 v28, v44, v47, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v51, v55, v53, s[2:3]
	v_xor_b32_e32 v44, s72, v68
	v_cndmask_b32_e32 v28, v66, v51, vcc
	v_ashrrev_i32_e32 v66, 31, v41
	v_mov_b32_e32 v67, v66
	v_xor_b32_e32 v28, v28, v44
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[66:67]
	v_sub_u32_e32 v28, v28, v44
	v_xor_b32_e32 v51, v40, v66
	v_xad_u32 v44, v28, v66, v37
	v_xor_b32_e32 v47, v41, v66
	v_mad_u64_u32 v[40:41], s[2:3], v51, s16, 0
	v_mul_hi_u32 v28, v51, s1
	v_lshl_add_u64 v[40:41], v[28:29], 0, v[40:41]
	v_mad_u64_u32 v[70:71], s[2:3], v47, s1, 0
	v_add_co_u32_e32 v28, vcc, v40, v70
	v_mad_u64_u32 v[68:69], s[2:3], v47, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v40, vcc, v41, v71, vcc
	v_mov_b32_e32 v41, s0
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[68:69]
	v_mul_lo_u32 v28, s21, v40
	v_mul_lo_u32 v53, s20, v41
	v_mad_u64_u32 v[40:41], s[2:3], s20, v40, 0
	v_add3_u32 v28, v41, v53, v28
	v_sub_u32_e32 v41, v47, v28
	v_sub_co_u32_e32 v40, vcc, v51, v40
	s_nop 1
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, vcc
	v_subrev_co_u32_e64 v51, s[2:3], s20, v40
	v_subb_co_u32_e32 v28, vcc, v47, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v53, s[14:15], 0, v41, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v53
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v55, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v51
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v53
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v40
	v_cndmask_b32_e64 v55, v55, v57, s[14:15]
	v_subrev_co_u32_e64 v57, s[2:3], s20, v51
	s_nop 1
	v_subbrev_co_u32_e64 v41, s[2:3], 0, v41, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v55
	s_nop 1
	v_cndmask_b32_e64 v41, v53, v41, s[2:3]
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v47, v47, v53, vcc
	v_cmp_ne_u32_e32 vcc, 0, v47
	s_nop 1
	v_cndmask_b32_e32 v28, v28, v41, vcc
	v_cndmask_b32_e64 v41, v51, v57, s[2:3]
	v_cndmask_b32_e32 v40, v40, v41, vcc
	v_xor_b32_e32 v40, v40, v66
	v_xor_b32_e32 v28, v28, v66
	v_sub_co_u32_e32 v40, vcc, v40, v66
	v_mul_lo_u32 v41, v44, s33
	s_nop 0
	v_subb_co_u32_e32 v28, vcc, v28, v66, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v41, v28, v40, v41
	v_sub_co_u32_e32 v28, vcc, -4, v34
	v_lshl_add_u64 v[66:67], v[34:35], 0, 3
	s_nop 0
	v_subb_co_u32_e32 v40, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v69, v67, v40, vcc
	v_ashrrev_i32_e32 v70, 31, v69
	v_cndmask_b32_e32 v68, v66, v28, vcc
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[70:71]
	v_xor_b32_e32 v44, v68, v70
	v_xor_b32_e32 v40, v69, v70
	v_mad_u64_u32 v[68:69], s[2:3], v44, s16, 0
	v_mul_hi_u32 v28, v44, s1
	v_lshl_add_u64 v[68:69], v[28:29], 0, v[68:69]
	v_mad_u64_u32 v[74:75], s[2:3], v40, s1, 0
	v_add_co_u32_e32 v28, vcc, v68, v74
	v_mad_u64_u32 v[72:73], s[2:3], v40, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v69, v75, vcc
	v_mov_b32_e32 v69, s0
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[72:73]
	v_mul_lo_u32 v28, s21, v68
	v_mul_lo_u32 v47, s20, v69
	v_mad_u64_u32 v[72:73], s[2:3], s20, v68, 0
	v_add3_u32 v28, v73, v47, v28
	v_sub_u32_e32 v47, v40, v28
	v_sub_co_u32_e32 v44, vcc, v44, v72
	s_nop 1
	v_subb_co_u32_e64 v47, s[2:3], v47, v25, vcc
	v_subrev_co_u32_e64 v51, s[2:3], s20, v44
	v_subb_co_u32_e32 v28, vcc, v40, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[2:3], 0, v47, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v47
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v53, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v51
	v_cndmask_b32_e64 v40, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v44
	v_cndmask_b32_e64 v51, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v47
	v_cndmask_b32_e64 v44, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v47, v53, v51, s[2:3]
	v_add_u32_e32 v51, 2, v68
	v_add_u32_e32 v53, 1, v68
	v_cmp_ne_u32_e64 s[2:3], 0, v47
	v_cndmask_b32_e32 v28, v40, v44, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v47, v53, v51, s[2:3]
	v_xor_b32_e32 v40, s72, v70
	v_cndmask_b32_e32 v28, v68, v47, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_mov_b32_e32 v69, v68
	v_xor_b32_e32 v28, v28, v40
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_sub_u32_e32 v28, v28, v40
	v_xor_b32_e32 v47, v66, v68
	v_xad_u32 v40, v28, v68, v37
	v_xor_b32_e32 v44, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v47, s16, 0
	v_mul_hi_u32 v28, v47, s1
	v_lshl_add_u64 v[66:67], v[28:29], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v44, s1, 0
	v_add_co_u32_e32 v28, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v44, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s0
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v28, s21, v66
	v_mul_lo_u32 v51, s20, v67
	v_mad_u64_u32 v[66:67], s[2:3], s20, v66, 0
	v_add3_u32 v28, v67, v51, v28
	v_sub_u32_e32 v51, v44, v28
	v_sub_co_u32_e32 v47, vcc, v47, v66
	v_mul_lo_u32 v40, v40, s33
	s_nop 0
	v_subb_co_u32_e64 v51, s[2:3], v51, v25, vcc
	v_subrev_co_u32_e64 v53, s[2:3], s20, v47
	v_subb_co_u32_e32 v28, vcc, v44, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v55, s[14:15], 0, v51, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v55
	v_subb_co_u32_e64 v51, s[2:3], v51, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v53
	v_cmp_le_u32_e32 vcc, s21, v28
	v_lshl_add_u64 v[66:67], v[34:35], 0, 4
	v_cndmask_b32_e64 v59, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v55
	v_cndmask_b32_e64 v44, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v47
	v_cndmask_b32_e64 v57, v57, v59, s[14:15]
	v_subrev_co_u32_e64 v59, s[2:3], s20, v53
	s_nop 1
	v_subbrev_co_u32_e64 v51, s[2:3], 0, v51, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v57
	s_nop 1
	v_cndmask_b32_e64 v51, v55, v51, s[2:3]
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v44, v44, v55, vcc
	v_cmp_ne_u32_e32 vcc, 0, v44
	v_cndmask_b32_e64 v44, v53, v59, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v44, v47, v44, vcc
	v_cndmask_b32_e32 v28, v28, v51, vcc
	v_xor_b32_e32 v44, v44, v68
	v_xor_b32_e32 v28, v28, v68
	v_sub_co_u32_e32 v44, vcc, v44, v68
	s_nop 1
	v_subb_co_u32_e32 v28, vcc, v28, v68, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v47, v28, v44, v40
	v_sub_co_u32_e32 v28, vcc, -5, v34
	s_nop 1
	v_subb_co_u32_e32 v40, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v69, v67, v40, vcc
	v_ashrrev_i32_e32 v70, 31, v69
	v_cndmask_b32_e32 v68, v66, v28, vcc
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[70:71]
	v_xor_b32_e32 v44, v68, v70
	v_xor_b32_e32 v40, v69, v70
	v_mad_u64_u32 v[68:69], s[2:3], v44, s16, 0
	v_mul_hi_u32 v28, v44, s1
	v_lshl_add_u64 v[68:69], v[28:29], 0, v[68:69]
	v_mad_u64_u32 v[74:75], s[2:3], v40, s1, 0
	v_add_co_u32_e32 v28, vcc, v68, v74
	v_mad_u64_u32 v[72:73], s[2:3], v40, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v69, v75, vcc
	v_mov_b32_e32 v69, s0
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[72:73]
	v_mul_lo_u32 v28, s21, v68
	v_mul_lo_u32 v51, s20, v69
	v_mad_u64_u32 v[72:73], s[2:3], s20, v68, 0
	v_add3_u32 v28, v73, v51, v28
	v_sub_u32_e32 v51, v40, v28
	v_sub_co_u32_e32 v44, vcc, v44, v72
	s_nop 1
	v_subb_co_u32_e64 v51, s[2:3], v51, v25, vcc
	v_subrev_co_u32_e64 v53, s[2:3], s20, v44
	v_subb_co_u32_e32 v28, vcc, v40, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v51, s[2:3], 0, v51, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v51
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v55, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v53
	v_cndmask_b32_e64 v40, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v44
	v_cndmask_b32_e64 v53, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v51
	v_cndmask_b32_e64 v44, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v51, v55, v53, s[2:3]
	v_add_u32_e32 v53, 2, v68
	v_add_u32_e32 v55, 1, v68
	v_cmp_ne_u32_e64 s[2:3], 0, v51
	v_cndmask_b32_e32 v28, v40, v44, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v51, v55, v53, s[2:3]
	v_xor_b32_e32 v40, s72, v70
	v_cndmask_b32_e32 v28, v68, v51, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_mov_b32_e32 v69, v68
	v_xor_b32_e32 v28, v28, v40
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_sub_u32_e32 v28, v28, v40
	v_xor_b32_e32 v51, v66, v68
	v_xad_u32 v40, v28, v68, v37
	v_xor_b32_e32 v44, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v51, s16, 0
	v_mul_hi_u32 v28, v51, s1
	v_lshl_add_u64 v[66:67], v[28:29], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v44, s1, 0
	v_add_co_u32_e32 v28, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v44, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s0
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v28, s21, v66
	v_mul_lo_u32 v53, s20, v67
	v_mad_u64_u32 v[66:67], s[2:3], s20, v66, 0
	v_add3_u32 v28, v67, v53, v28
	v_sub_u32_e32 v53, v44, v28
	v_sub_co_u32_e32 v51, vcc, v51, v66
	v_mul_lo_u32 v40, v40, s33
	s_nop 0
	v_subb_co_u32_e64 v53, s[2:3], v53, v25, vcc
	v_subrev_co_u32_e64 v55, s[2:3], s20, v51
	v_subb_co_u32_e32 v28, vcc, v44, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v57, s[14:15], 0, v53, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v57
	v_subb_co_u32_e64 v53, s[2:3], v53, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v55
	v_cmp_le_u32_e32 vcc, s21, v28
	v_lshl_add_u64 v[66:67], v[34:35], 0, 5
	v_cndmask_b32_e64 v61, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v57
	v_cndmask_b32_e64 v44, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v51
	v_cndmask_b32_e64 v59, v59, v61, s[14:15]
	v_subrev_co_u32_e64 v61, s[2:3], s20, v55
	s_nop 1
	v_subbrev_co_u32_e64 v53, s[2:3], 0, v53, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v59
	s_nop 1
	v_cndmask_b32_e64 v53, v57, v53, s[2:3]
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v44, v44, v57, vcc
	v_cmp_ne_u32_e32 vcc, 0, v44
	v_cndmask_b32_e64 v44, v55, v61, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v44, v51, v44, vcc
	v_cndmask_b32_e32 v28, v28, v53, vcc
	v_xor_b32_e32 v44, v44, v68
	v_xor_b32_e32 v28, v28, v68
	v_sub_co_u32_e32 v44, vcc, v44, v68
	s_nop 1
	v_subb_co_u32_e32 v28, vcc, v28, v68, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v51, v28, v44, v40
	v_sub_co_u32_e32 v28, vcc, -6, v34
	s_nop 1
	v_subb_co_u32_e32 v40, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v69, v67, v40, vcc
	v_ashrrev_i32_e32 v70, 31, v69
	v_cndmask_b32_e32 v68, v66, v28, vcc
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[70:71]
	v_xor_b32_e32 v44, v68, v70
	v_xor_b32_e32 v40, v69, v70
	v_mad_u64_u32 v[68:69], s[2:3], v44, s16, 0
	v_mul_hi_u32 v28, v44, s1
	v_lshl_add_u64 v[68:69], v[28:29], 0, v[68:69]
	v_mad_u64_u32 v[74:75], s[2:3], v40, s1, 0
	v_add_co_u32_e32 v28, vcc, v68, v74
	v_mad_u64_u32 v[72:73], s[2:3], v40, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v69, v75, vcc
	v_mov_b32_e32 v69, s0
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[72:73]
	v_mul_lo_u32 v28, s21, v68
	v_mul_lo_u32 v53, s20, v69
	v_mad_u64_u32 v[72:73], s[2:3], s20, v68, 0
	v_add3_u32 v28, v73, v53, v28
	v_sub_u32_e32 v53, v40, v28
	v_sub_co_u32_e32 v44, vcc, v44, v72
	s_nop 1
	v_subb_co_u32_e64 v53, s[2:3], v53, v25, vcc
	v_subrev_co_u32_e64 v55, s[2:3], s20, v44
	v_subb_co_u32_e32 v28, vcc, v40, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v53, s[2:3], 0, v53, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v53
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v55
	v_cndmask_b32_e64 v40, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v44
	v_cndmask_b32_e64 v55, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v53
	v_cndmask_b32_e64 v44, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v53, v57, v55, s[2:3]
	v_add_u32_e32 v55, 2, v68
	v_add_u32_e32 v57, 1, v68
	v_cmp_ne_u32_e64 s[2:3], 0, v53
	v_cndmask_b32_e32 v28, v40, v44, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v53, v57, v55, s[2:3]
	v_xor_b32_e32 v40, s72, v70
	v_cndmask_b32_e32 v28, v68, v53, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_mov_b32_e32 v69, v68
	v_xor_b32_e32 v28, v28, v40
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_sub_u32_e32 v28, v28, v40
	v_xor_b32_e32 v53, v66, v68
	v_xad_u32 v40, v28, v68, v37
	v_xor_b32_e32 v44, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v53, s16, 0
	v_mul_hi_u32 v28, v53, s1
	v_lshl_add_u64 v[66:67], v[28:29], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v44, s1, 0
	v_add_co_u32_e32 v28, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v44, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s0
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v28, s21, v66
	v_mul_lo_u32 v55, s20, v67
	v_mad_u64_u32 v[66:67], s[2:3], s20, v66, 0
	v_add3_u32 v28, v67, v55, v28
	v_sub_u32_e32 v55, v44, v28
	v_sub_co_u32_e32 v53, vcc, v53, v66
	v_mul_lo_u32 v40, v40, s33
	s_nop 0
	v_subb_co_u32_e64 v55, s[2:3], v55, v25, vcc
	v_subrev_co_u32_e64 v57, s[2:3], s20, v53
	v_subb_co_u32_e32 v28, vcc, v44, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v59, s[14:15], 0, v55, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v59
	v_subb_co_u32_e64 v55, s[2:3], v55, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v57
	v_cmp_le_u32_e32 vcc, s21, v28
	v_lshl_add_u64 v[66:67], v[34:35], 0, 6
	v_cndmask_b32_e64 v63, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v59
	v_cndmask_b32_e64 v44, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v53
	v_cndmask_b32_e64 v61, v61, v63, s[14:15]
	v_subrev_co_u32_e64 v63, s[2:3], s20, v57
	s_nop 1
	v_subbrev_co_u32_e64 v55, s[2:3], 0, v55, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v61
	s_nop 1
	v_cndmask_b32_e64 v55, v59, v55, s[2:3]
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v44, v44, v59, vcc
	v_cmp_ne_u32_e32 vcc, 0, v44
	v_cndmask_b32_e64 v44, v57, v63, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v44, v53, v44, vcc
	v_cndmask_b32_e32 v28, v28, v55, vcc
	v_xor_b32_e32 v44, v44, v68
	v_xor_b32_e32 v28, v28, v68
	v_sub_co_u32_e32 v44, vcc, v44, v68
	s_nop 1
	v_subb_co_u32_e32 v28, vcc, v28, v68, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v53, v28, v44, v40
	v_sub_co_u32_e32 v28, vcc, -7, v34
	s_nop 1
	v_subb_co_u32_e32 v40, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v69, v67, v40, vcc
	v_ashrrev_i32_e32 v70, 31, v69
	v_cndmask_b32_e32 v68, v66, v28, vcc
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[70:71]
	v_xor_b32_e32 v44, v68, v70
	v_xor_b32_e32 v40, v69, v70
	v_mad_u64_u32 v[68:69], s[2:3], v44, s16, 0
	v_mul_hi_u32 v28, v44, s1
	v_lshl_add_u64 v[68:69], v[28:29], 0, v[68:69]
	v_mad_u64_u32 v[74:75], s[2:3], v40, s1, 0
	v_add_co_u32_e32 v28, vcc, v68, v74
	v_mad_u64_u32 v[72:73], s[2:3], v40, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v69, v75, vcc
	v_mov_b32_e32 v69, s0
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[72:73]
	v_mul_lo_u32 v28, s21, v68
	v_mul_lo_u32 v55, s20, v69
	v_mad_u64_u32 v[72:73], s[2:3], s20, v68, 0
	v_add3_u32 v28, v73, v55, v28
	v_sub_u32_e32 v55, v40, v28
	v_sub_co_u32_e32 v44, vcc, v44, v72
	s_nop 1
	v_subb_co_u32_e64 v55, s[2:3], v55, v25, vcc
	v_subrev_co_u32_e64 v57, s[2:3], s20, v44
	v_subb_co_u32_e32 v28, vcc, v40, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v55, s[2:3], 0, v55, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v55
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v57
	v_cndmask_b32_e64 v40, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v44
	v_cndmask_b32_e64 v57, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v55
	v_cndmask_b32_e64 v44, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v55, v59, v57, s[2:3]
	v_add_u32_e32 v57, 2, v68
	v_add_u32_e32 v59, 1, v68
	v_cmp_ne_u32_e64 s[2:3], 0, v55
	v_cndmask_b32_e32 v28, v40, v44, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v55, v59, v57, s[2:3]
	v_xor_b32_e32 v40, s72, v70
	v_cndmask_b32_e32 v28, v68, v55, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_mov_b32_e32 v69, v68
	v_xor_b32_e32 v28, v28, v40
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_sub_u32_e32 v28, v28, v40
	v_xor_b32_e32 v55, v66, v68
	v_xad_u32 v40, v28, v68, v37
	v_xor_b32_e32 v44, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v55, s16, 0
	v_mul_hi_u32 v28, v55, s1
	v_lshl_add_u64 v[66:67], v[28:29], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v44, s1, 0
	v_add_co_u32_e32 v28, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v44, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s0
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v28, s21, v66
	v_mul_lo_u32 v57, s20, v67
	v_mad_u64_u32 v[66:67], s[2:3], s20, v66, 0
	v_add3_u32 v28, v67, v57, v28
	v_sub_u32_e32 v57, v44, v28
	v_sub_co_u32_e32 v55, vcc, v55, v66
	v_mul_lo_u32 v40, v40, s33
	s_nop 0
	v_subb_co_u32_e64 v57, s[2:3], v57, v25, vcc
	v_subrev_co_u32_e64 v59, s[2:3], s20, v55
	v_subb_co_u32_e32 v28, vcc, v44, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v61, s[14:15], 0, v57, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v61
	v_subb_co_u32_e64 v57, s[2:3], v57, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v59
	v_cmp_le_u32_e32 vcc, s21, v28
	v_lshl_add_u64 v[66:67], v[34:35], 0, 7
	v_cndmask_b32_e64 v65, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v61
	v_cndmask_b32_e64 v44, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v55
	v_cndmask_b32_e64 v63, v63, v65, s[14:15]
	v_subrev_co_u32_e64 v65, s[2:3], s20, v59
	s_nop 1
	v_subbrev_co_u32_e64 v57, s[2:3], 0, v57, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v63
	s_nop 1
	v_cndmask_b32_e64 v57, v61, v57, s[2:3]
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v44, v44, v61, vcc
	v_cmp_ne_u32_e32 vcc, 0, v44
	v_cndmask_b32_e64 v44, v59, v65, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v44, v55, v44, vcc
	v_cndmask_b32_e32 v28, v28, v57, vcc
	v_xor_b32_e32 v44, v44, v68
	v_xor_b32_e32 v28, v28, v68
	v_sub_co_u32_e32 v44, vcc, v44, v68
	s_nop 1
	v_subb_co_u32_e32 v28, vcc, v28, v68, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v57, v28, v44, v40
	v_sub_co_u32_e32 v28, vcc, -8, v34
	s_nop 1
	v_subb_co_u32_e32 v40, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v69, v67, v40, vcc
	v_ashrrev_i32_e32 v70, 31, v69
	v_cndmask_b32_e32 v68, v66, v28, vcc
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[70:71]
	v_xor_b32_e32 v44, v68, v70
	v_xor_b32_e32 v40, v69, v70
	v_mad_u64_u32 v[68:69], s[2:3], v44, s16, 0
	v_mul_hi_u32 v28, v44, s1
	v_lshl_add_u64 v[68:69], v[28:29], 0, v[68:69]
	v_mad_u64_u32 v[74:75], s[2:3], v40, s1, 0
	v_add_co_u32_e32 v28, vcc, v68, v74
	v_mad_u64_u32 v[72:73], s[2:3], v40, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v69, v75, vcc
	v_mov_b32_e32 v69, s0
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[72:73]
	v_mul_lo_u32 v28, s21, v68
	v_mul_lo_u32 v55, s20, v69
	v_mad_u64_u32 v[72:73], s[2:3], s20, v68, 0
	v_add3_u32 v28, v73, v55, v28
	v_sub_u32_e32 v55, v40, v28
	v_sub_co_u32_e32 v44, vcc, v44, v72
	s_nop 1
	v_subb_co_u32_e64 v55, s[2:3], v55, v25, vcc
	v_subrev_co_u32_e64 v59, s[2:3], s20, v44
	v_subb_co_u32_e32 v28, vcc, v40, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v55, s[2:3], 0, v55, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v55
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v59
	v_cndmask_b32_e64 v40, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v44
	v_cndmask_b32_e64 v59, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v55
	v_cndmask_b32_e64 v44, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v55, v61, v59, s[2:3]
	v_add_u32_e32 v59, 2, v68
	v_add_u32_e32 v61, 1, v68
	v_cmp_ne_u32_e64 s[2:3], 0, v55
	v_cndmask_b32_e32 v28, v40, v44, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v55, v61, v59, s[2:3]
	v_xor_b32_e32 v40, s72, v70
	v_cndmask_b32_e32 v28, v68, v55, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_mov_b32_e32 v69, v68
	v_xor_b32_e32 v28, v28, v40
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_sub_u32_e32 v28, v28, v40
	v_xor_b32_e32 v55, v66, v68
	v_xad_u32 v40, v28, v68, v37
	v_xor_b32_e32 v44, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v55, s16, 0
	v_mul_hi_u32 v28, v55, s1
	v_lshl_add_u64 v[66:67], v[28:29], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v44, s1, 0
	v_add_co_u32_e32 v28, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v44, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s0
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v28, s21, v66
	v_mul_lo_u32 v59, s20, v67
	v_mad_u64_u32 v[66:67], s[2:3], s20, v66, 0
	v_add3_u32 v28, v67, v59, v28
	v_sub_u32_e32 v59, v44, v28
	v_sub_co_u32_e32 v55, vcc, v55, v66
	v_mul_lo_u32 v40, v40, s33
	s_nop 0
	v_subb_co_u32_e64 v59, s[2:3], v59, v25, vcc
	v_subrev_co_u32_e64 v61, s[2:3], s20, v55
	v_subb_co_u32_e32 v28, vcc, v44, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v63, s[14:15], 0, v59, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v63
	v_subb_co_u32_e64 v59, s[2:3], v59, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v61
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v66, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v63
	v_cndmask_b32_e64 v44, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v55
	v_cndmask_b32_e64 v65, v65, v66, s[14:15]
	v_subrev_co_u32_e64 v66, s[2:3], s20, v61
	s_nop 1
	v_subbrev_co_u32_e64 v59, s[2:3], 0, v59, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v65
	s_nop 1
	v_cndmask_b32_e64 v59, v63, v59, s[2:3]
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v44, v44, v63, vcc
	v_cmp_ne_u32_e32 vcc, 0, v44
	v_cndmask_b32_e64 v44, v61, v66, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v44, v55, v44, vcc
	v_cndmask_b32_e32 v28, v28, v59, vcc
	v_xor_b32_e32 v44, v44, v68
	v_xor_b32_e32 v28, v28, v68
	v_sub_co_u32_e32 v44, vcc, v44, v68
	s_nop 1
	v_subb_co_u32_e32 v28, vcc, v28, v68, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v28, v28, v44, v40
	buffer_load_ubyte v40, v38, s[36:39], 0 offen
	buffer_load_ubyte v44, v39, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v41, v41, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v47, v47, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v51, v51, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v55, v53, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v53, v57, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v57, v28, s[36:39], 0 offen
	v_sub_co_u32_e32 v28, vcc, -9, v34
	v_lshl_add_u64 v[38:39], v[34:35], 0, 8
	s_nop 0
	v_subb_co_u32_e32 v59, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v39
	s_nop 1
	v_cndmask_b32_e32 v67, v39, v59, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v38, v28, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v61, v66, v68
	v_xor_b32_e32 v59, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v61, s16, 0
	v_mul_hi_u32 v28, v61, s1
	v_lshl_add_u64 v[66:67], v[28:29], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v59, s1, 0
	v_add_co_u32_e32 v28, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v59, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s0
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v28, s21, v66
	v_mul_lo_u32 v63, s20, v67
	v_mad_u64_u32 v[70:71], s[2:3], s20, v66, 0
	v_add3_u32 v28, v71, v63, v28
	v_sub_u32_e32 v63, v59, v28
	v_sub_co_u32_e32 v61, vcc, v61, v70
	s_nop 1
	v_subb_co_u32_e64 v63, s[2:3], v63, v25, vcc
	v_subrev_co_u32_e64 v65, s[2:3], s20, v61
	v_subb_co_u32_e32 v28, vcc, v59, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v63, s[2:3], 0, v63, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v63
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v67, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v65
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v61
	v_cndmask_b32_e64 v65, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v63
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v63, v67, v65, s[2:3]
	v_add_u32_e32 v65, 2, v66
	v_add_u32_e32 v67, 1, v66
	v_cmp_ne_u32_e64 s[2:3], 0, v63
	v_cndmask_b32_e32 v28, v59, v61, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v63, v67, v65, s[2:3]
	v_xor_b32_e32 v59, s72, v68
	v_cndmask_b32_e32 v28, v66, v63, vcc
	v_ashrrev_i32_e32 v66, 31, v39
	v_mov_b32_e32 v67, v66
	v_xor_b32_e32 v28, v28, v59
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[66:67]
	v_sub_u32_e32 v28, v28, v59
	v_xor_b32_e32 v63, v38, v66
	v_xad_u32 v59, v28, v66, v37
	v_xor_b32_e32 v61, v39, v66
	v_mad_u64_u32 v[38:39], s[2:3], v63, s16, 0
	v_mul_hi_u32 v28, v63, s1
	v_lshl_add_u64 v[38:39], v[28:29], 0, v[38:39]
	v_mad_u64_u32 v[70:71], s[2:3], v61, s1, 0
	v_add_co_u32_e32 v28, vcc, v38, v70
	v_mad_u64_u32 v[68:69], s[2:3], v61, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v71, vcc
	v_mov_b32_e32 v39, s0
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[68:69]
	v_mul_lo_u32 v28, s21, v38
	v_mul_lo_u32 v65, s20, v39
	v_mad_u64_u32 v[38:39], s[2:3], s20, v38, 0
	v_add3_u32 v28, v39, v65, v28
	v_sub_u32_e32 v39, v61, v28
	v_sub_co_u32_e32 v38, vcc, v63, v38
	s_nop 1
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, vcc
	v_subrev_co_u32_e64 v63, s[2:3], s20, v38
	v_subb_co_u32_e32 v28, vcc, v61, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v39, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v65
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v67, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v63
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v68, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v65
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v38
	v_cndmask_b32_e64 v67, v67, v68, s[14:15]
	v_subrev_co_u32_e64 v68, s[2:3], s20, v63
	s_nop 1
	v_subbrev_co_u32_e64 v39, s[2:3], 0, v39, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v67
	s_nop 1
	v_cndmask_b32_e64 v39, v65, v39, s[2:3]
	v_cndmask_b32_e64 v65, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v61, v61, v65, vcc
	v_cmp_ne_u32_e32 vcc, 0, v61
	s_nop 1
	v_cndmask_b32_e32 v28, v28, v39, vcc
	v_cndmask_b32_e64 v39, v63, v68, s[2:3]
	v_cndmask_b32_e32 v38, v38, v39, vcc
	v_xor_b32_e32 v38, v38, v66
	v_xor_b32_e32 v28, v28, v66
	v_sub_co_u32_e32 v38, vcc, v38, v66
	v_mul_lo_u32 v39, v59, s33
	s_nop 0
	v_subb_co_u32_e32 v28, vcc, v28, v66, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v38, v28, v38, v39
	v_sub_co_u32_e32 v28, vcc, -10, v34
	v_lshl_add_u64 v[66:67], v[34:35], 0, 9
	s_nop 0
	v_subb_co_u32_e32 v39, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v69, v67, v39, vcc
	v_ashrrev_i32_e32 v70, 31, v69
	v_cndmask_b32_e32 v68, v66, v28, vcc
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[70:71]
	v_xor_b32_e32 v59, v68, v70
	v_xor_b32_e32 v39, v69, v70
	v_mad_u64_u32 v[68:69], s[2:3], v59, s16, 0
	v_mul_hi_u32 v28, v59, s1
	v_lshl_add_u64 v[68:69], v[28:29], 0, v[68:69]
	v_mad_u64_u32 v[74:75], s[2:3], v39, s1, 0
	v_add_co_u32_e32 v28, vcc, v68, v74
	v_mad_u64_u32 v[72:73], s[2:3], v39, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v69, v75, vcc
	v_mov_b32_e32 v69, s0
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[72:73]
	v_mul_lo_u32 v28, s21, v68
	v_mul_lo_u32 v61, s20, v69
	v_mad_u64_u32 v[72:73], s[2:3], s20, v68, 0
	v_add3_u32 v28, v73, v61, v28
	v_sub_u32_e32 v61, v39, v28
	v_sub_co_u32_e32 v59, vcc, v59, v72
	s_nop 1
	v_subb_co_u32_e64 v61, s[2:3], v61, v25, vcc
	v_subrev_co_u32_e64 v63, s[2:3], s20, v59
	v_subb_co_u32_e32 v28, vcc, v39, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v61, s[2:3], 0, v61, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v61
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v63
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v59
	v_cndmask_b32_e64 v63, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v61
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v61, v65, v63, s[2:3]
	v_add_u32_e32 v63, 2, v68
	v_add_u32_e32 v65, 1, v68
	v_cmp_ne_u32_e64 s[2:3], 0, v61
	v_cndmask_b32_e32 v28, v39, v59, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v61, v65, v63, s[2:3]
	v_xor_b32_e32 v39, s72, v70
	v_cndmask_b32_e32 v28, v68, v61, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_mov_b32_e32 v69, v68
	v_xor_b32_e32 v28, v28, v39
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_sub_u32_e32 v28, v28, v39
	v_xor_b32_e32 v61, v66, v68
	v_xad_u32 v39, v28, v68, v37
	v_xor_b32_e32 v59, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v61, s16, 0
	v_mul_hi_u32 v28, v61, s1
	v_lshl_add_u64 v[66:67], v[28:29], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v59, s1, 0
	v_add_co_u32_e32 v28, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v59, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s0
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v28, s21, v66
	v_mul_lo_u32 v63, s20, v67
	v_mad_u64_u32 v[66:67], s[2:3], s20, v66, 0
	v_add3_u32 v28, v67, v63, v28
	v_sub_u32_e32 v63, v59, v28
	v_sub_co_u32_e32 v61, vcc, v61, v66
	v_mul_lo_u32 v39, v39, s33
	s_nop 0
	v_subb_co_u32_e64 v63, s[2:3], v63, v25, vcc
	v_subrev_co_u32_e64 v65, s[2:3], s20, v61
	v_subb_co_u32_e32 v28, vcc, v59, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v66, s[14:15], 0, v63, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v66
	v_subb_co_u32_e64 v63, s[2:3], v63, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v67, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v65
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v66
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v61
	v_cndmask_b32_e64 v67, v67, v69, s[14:15]
	v_subrev_co_u32_e64 v69, s[2:3], s20, v65
	s_nop 1
	v_subbrev_co_u32_e64 v63, s[2:3], 0, v63, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v67
	s_nop 1
	v_cndmask_b32_e64 v63, v66, v63, s[2:3]
	v_cndmask_b32_e64 v66, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v59, v59, v66, vcc
	v_cmp_ne_u32_e32 vcc, 0, v59
	v_cndmask_b32_e64 v59, v65, v69, s[2:3]
	v_lshl_add_u64 v[66:67], v[34:35], 0, 10
	v_cndmask_b32_e32 v59, v61, v59, vcc
	v_cndmask_b32_e32 v28, v28, v63, vcc
	v_xor_b32_e32 v59, v59, v68
	v_xor_b32_e32 v28, v28, v68
	v_sub_co_u32_e32 v59, vcc, v59, v68
	s_nop 1
	v_subb_co_u32_e32 v28, vcc, v28, v68, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v39, v28, v59, v39
	v_sub_co_u32_e32 v28, vcc, -11, v34
	s_nop 1
	v_subb_co_u32_e32 v59, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v69, v67, v59, vcc
	v_ashrrev_i32_e32 v70, 31, v69
	v_cndmask_b32_e32 v68, v66, v28, vcc
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[70:71]
	v_xor_b32_e32 v61, v68, v70
	v_xor_b32_e32 v59, v69, v70
	v_mad_u64_u32 v[68:69], s[2:3], v61, s16, 0
	v_mul_hi_u32 v28, v61, s1
	v_lshl_add_u64 v[68:69], v[28:29], 0, v[68:69]
	v_mad_u64_u32 v[74:75], s[2:3], v59, s1, 0
	v_add_co_u32_e32 v28, vcc, v68, v74
	v_mad_u64_u32 v[72:73], s[2:3], v59, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v69, v75, vcc
	v_mov_b32_e32 v69, s0
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[72:73]
	v_mul_lo_u32 v28, s21, v68
	v_mul_lo_u32 v63, s20, v69
	v_mad_u64_u32 v[72:73], s[2:3], s20, v68, 0
	v_add3_u32 v28, v73, v63, v28
	v_sub_u32_e32 v63, v59, v28
	v_sub_co_u32_e32 v61, vcc, v61, v72
	s_nop 1
	v_subb_co_u32_e64 v63, s[2:3], v63, v25, vcc
	v_subrev_co_u32_e64 v65, s[2:3], s20, v61
	v_subb_co_u32_e32 v28, vcc, v59, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v63, s[2:3], 0, v63, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v63
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v65
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v61
	v_cndmask_b32_e64 v65, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v63
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v63, v69, v65, s[2:3]
	v_add_u32_e32 v65, 2, v68
	v_add_u32_e32 v69, 1, v68
	v_cmp_ne_u32_e64 s[2:3], 0, v63
	v_cndmask_b32_e32 v28, v59, v61, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v63, v69, v65, s[2:3]
	v_xor_b32_e32 v59, s72, v70
	v_cndmask_b32_e32 v28, v68, v63, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_mov_b32_e32 v69, v68
	v_xor_b32_e32 v28, v28, v59
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_sub_u32_e32 v28, v28, v59
	v_xor_b32_e32 v63, v66, v68
	v_xad_u32 v59, v28, v68, v37
	v_xor_b32_e32 v61, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v63, s16, 0
	v_mul_hi_u32 v28, v63, s1
	v_lshl_add_u64 v[66:67], v[28:29], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v61, s1, 0
	v_add_co_u32_e32 v28, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v61, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s0
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v28, s21, v66
	v_mul_lo_u32 v65, s20, v67
	v_mad_u64_u32 v[66:67], s[2:3], s20, v66, 0
	v_add3_u32 v28, v67, v65, v28
	v_sub_u32_e32 v65, v61, v28
	v_sub_co_u32_e32 v63, vcc, v63, v66
	v_mul_lo_u32 v59, v59, s33
	s_nop 0
	v_subb_co_u32_e64 v65, s[2:3], v65, v25, vcc
	v_subrev_co_u32_e64 v66, s[2:3], s20, v63
	v_subb_co_u32_e32 v28, vcc, v61, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v65, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v67
	v_subb_co_u32_e64 v65, s[2:3], v65, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v66
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v67
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v63
	v_cndmask_b32_e64 v69, v69, v70, s[14:15]
	v_subrev_co_u32_e64 v70, s[2:3], s20, v66
	s_nop 1
	v_subbrev_co_u32_e64 v65, s[2:3], 0, v65, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v69
	s_nop 1
	v_cndmask_b32_e64 v65, v67, v65, s[2:3]
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v61, v61, v67, vcc
	v_cmp_ne_u32_e32 vcc, 0, v61
	v_cndmask_b32_e64 v61, v66, v70, s[2:3]
	v_lshl_add_u64 v[66:67], v[34:35], 0, 11
	v_cndmask_b32_e32 v61, v63, v61, vcc
	v_cndmask_b32_e32 v28, v28, v65, vcc
	v_xor_b32_e32 v61, v61, v68
	v_xor_b32_e32 v28, v28, v68
	v_sub_co_u32_e32 v61, vcc, v61, v68
	s_nop 1
	v_subb_co_u32_e32 v28, vcc, v28, v68, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v61, v28, v61, v59
	v_sub_co_u32_e32 v28, vcc, -12, v34
	s_nop 1
	v_subb_co_u32_e32 v59, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v69, v67, v59, vcc
	v_ashrrev_i32_e32 v70, 31, v69
	v_cndmask_b32_e32 v68, v66, v28, vcc
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[70:71]
	v_xor_b32_e32 v63, v68, v70
	v_xor_b32_e32 v59, v69, v70
	v_mad_u64_u32 v[68:69], s[2:3], v63, s16, 0
	v_mul_hi_u32 v28, v63, s1
	v_lshl_add_u64 v[68:69], v[28:29], 0, v[68:69]
	v_mad_u64_u32 v[74:75], s[2:3], v59, s1, 0
	v_add_co_u32_e32 v28, vcc, v68, v74
	v_mad_u64_u32 v[72:73], s[2:3], v59, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v69, v75, vcc
	v_mov_b32_e32 v69, s0
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[72:73]
	v_mul_lo_u32 v28, s21, v68
	v_mul_lo_u32 v65, s20, v69
	v_mad_u64_u32 v[72:73], s[2:3], s20, v68, 0
	v_add3_u32 v28, v73, v65, v28
	v_sub_u32_e32 v65, v59, v28
	v_sub_co_u32_e32 v63, vcc, v63, v72
	s_nop 1
	v_subb_co_u32_e64 v65, s[2:3], v65, v25, vcc
	v_subrev_co_u32_e64 v69, s[2:3], s20, v63
	v_subb_co_u32_e32 v28, vcc, v59, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[2:3], 0, v65, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v65
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v69
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v63
	v_cndmask_b32_e64 v69, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v65
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v65, v71, v69, s[2:3]
	v_add_u32_e32 v69, 2, v68
	v_add_u32_e32 v71, 1, v68
	v_cmp_ne_u32_e64 s[2:3], 0, v65
	v_cndmask_b32_e32 v28, v59, v63, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v65, v71, v69, s[2:3]
	v_xor_b32_e32 v59, s72, v70
	v_cndmask_b32_e32 v28, v68, v65, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_mov_b32_e32 v69, v68
	v_xor_b32_e32 v28, v28, v59
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_sub_u32_e32 v28, v28, v59
	v_xor_b32_e32 v65, v66, v68
	v_xad_u32 v59, v28, v68, v37
	v_xor_b32_e32 v63, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v65, s16, 0
	v_mul_hi_u32 v28, v65, s1
	v_lshl_add_u64 v[66:67], v[28:29], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v63, s1, 0
	v_add_co_u32_e32 v28, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v63, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s0
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v28, s21, v66
	v_mul_lo_u32 v69, s20, v67
	v_mad_u64_u32 v[66:67], s[2:3], s20, v66, 0
	v_add3_u32 v28, v67, v69, v28
	v_sub_u32_e32 v67, v63, v28
	v_sub_co_u32_e32 v65, vcc, v65, v66
	v_mul_lo_u32 v59, v59, s33
	s_nop 0
	v_subb_co_u32_e64 v66, s[2:3], v67, v25, vcc
	v_subrev_co_u32_e64 v67, s[2:3], s20, v65
	v_subb_co_u32_e32 v28, vcc, v63, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v66, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v69
	v_subb_co_u32_e64 v66, s[2:3], v66, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v67
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v69
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v65
	v_cndmask_b32_e64 v70, v70, v71, s[14:15]
	v_subrev_co_u32_e64 v71, s[2:3], s20, v67
	s_nop 1
	v_subbrev_co_u32_e64 v66, s[2:3], 0, v66, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v70
	s_nop 1
	v_cndmask_b32_e64 v66, v69, v66, s[2:3]
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v63, v63, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v63
	v_cndmask_b32_e64 v63, v67, v71, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v63, v65, v63, vcc
	v_cndmask_b32_e32 v28, v28, v66, vcc
	v_xor_b32_e32 v63, v63, v68
	v_xor_b32_e32 v28, v28, v68
	v_sub_co_u32_e32 v63, vcc, v63, v68
	v_lshl_add_u64 v[66:67], v[34:35], 0, 12
	s_nop 0
	v_subb_co_u32_e32 v28, vcc, v28, v68, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v65, v28, v63, v59
	v_sub_co_u32_e32 v28, vcc, -13, v34
	s_nop 1
	v_subb_co_u32_e32 v59, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v69, v67, v59, vcc
	v_ashrrev_i32_e32 v70, 31, v69
	v_cndmask_b32_e32 v68, v66, v28, vcc
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[70:71]
	v_xor_b32_e32 v63, v68, v70
	v_xor_b32_e32 v59, v69, v70
	v_mad_u64_u32 v[68:69], s[2:3], v63, s16, 0
	v_mul_hi_u32 v28, v63, s1
	v_lshl_add_u64 v[68:69], v[28:29], 0, v[68:69]
	v_mad_u64_u32 v[74:75], s[2:3], v59, s1, 0
	v_add_co_u32_e32 v28, vcc, v68, v74
	v_mad_u64_u32 v[72:73], s[2:3], v59, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v69, v75, vcc
	v_mov_b32_e32 v69, s0
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[72:73]
	v_mul_lo_u32 v28, s21, v68
	v_mul_lo_u32 v69, s20, v69
	v_mad_u64_u32 v[72:73], s[2:3], s20, v68, 0
	v_add3_u32 v28, v73, v69, v28
	v_sub_u32_e32 v69, v59, v28
	v_sub_co_u32_e32 v63, vcc, v63, v72
	s_nop 1
	v_subb_co_u32_e64 v69, s[2:3], v69, v25, vcc
	v_subrev_co_u32_e64 v71, s[2:3], s20, v63
	v_subb_co_u32_e32 v28, vcc, v59, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[2:3], 0, v69, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v69
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v71
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v63
	v_cndmask_b32_e64 v71, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v69
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v69, v72, v71, s[2:3]
	v_add_u32_e32 v71, 2, v68
	v_add_u32_e32 v72, 1, v68
	v_cmp_ne_u32_e64 s[2:3], 0, v69
	v_cndmask_b32_e32 v28, v59, v63, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v69, v72, v71, s[2:3]
	v_xor_b32_e32 v59, s72, v70
	v_cndmask_b32_e32 v28, v68, v69, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_mov_b32_e32 v69, v68
	v_xor_b32_e32 v28, v28, v59
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_sub_u32_e32 v28, v28, v59
	v_xor_b32_e32 v69, v66, v68
	v_xad_u32 v59, v28, v68, v37
	v_xor_b32_e32 v63, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v69, s16, 0
	v_mul_hi_u32 v28, v69, s1
	v_lshl_add_u64 v[66:67], v[28:29], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v63, s1, 0
	v_add_co_u32_e32 v28, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v63, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s0
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v28, s21, v66
	v_mul_lo_u32 v70, s20, v67
	v_mad_u64_u32 v[66:67], s[2:3], s20, v66, 0
	v_add3_u32 v28, v67, v70, v28
	v_sub_u32_e32 v67, v63, v28
	v_sub_co_u32_e32 v66, vcc, v69, v66
	v_mul_lo_u32 v59, v59, s33
	s_nop 0
	v_subb_co_u32_e64 v67, s[2:3], v67, v25, vcc
	v_subrev_co_u32_e64 v69, s[2:3], s20, v66
	v_subb_co_u32_e32 v28, vcc, v63, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v70, s[14:15], 0, v67, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v70
	v_subb_co_u32_e64 v67, s[2:3], v67, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v69
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v70
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v66
	v_cndmask_b32_e64 v71, v71, v72, s[14:15]
	v_subrev_co_u32_e64 v72, s[2:3], s20, v69
	s_nop 1
	v_subbrev_co_u32_e64 v67, s[2:3], 0, v67, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v71
	s_nop 1
	v_cndmask_b32_e64 v67, v70, v67, s[2:3]
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v63, v63, v70, vcc
	v_cmp_ne_u32_e32 vcc, 0, v63
	v_cndmask_b32_e64 v63, v69, v72, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v63, v66, v63, vcc
	v_cndmask_b32_e32 v28, v28, v67, vcc
	v_xor_b32_e32 v63, v63, v68
	v_xor_b32_e32 v28, v28, v68
	v_sub_co_u32_e32 v63, vcc, v63, v68
	v_lshl_add_u64 v[66:67], v[34:35], 0, 13
	s_nop 0
	v_subb_co_u32_e32 v28, vcc, v28, v68, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v76, v28, v63, v59
	v_sub_co_u32_e32 v28, vcc, -14, v34
	s_nop 1
	v_subb_co_u32_e32 v59, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v69, v67, v59, vcc
	v_ashrrev_i32_e32 v70, 31, v69
	v_cndmask_b32_e32 v68, v66, v28, vcc
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[70:71]
	v_xor_b32_e32 v63, v68, v70
	v_xor_b32_e32 v59, v69, v70
	v_mad_u64_u32 v[68:69], s[2:3], v63, s16, 0
	v_mul_hi_u32 v28, v63, s1
	v_lshl_add_u64 v[68:69], v[28:29], 0, v[68:69]
	v_mad_u64_u32 v[74:75], s[2:3], v59, s1, 0
	v_add_co_u32_e32 v28, vcc, v68, v74
	v_mad_u64_u32 v[72:73], s[2:3], v59, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v69, v75, vcc
	v_mov_b32_e32 v69, s0
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[72:73]
	v_mul_lo_u32 v28, s21, v68
	v_mul_lo_u32 v69, s20, v69
	v_mad_u64_u32 v[72:73], s[2:3], s20, v68, 0
	v_add3_u32 v28, v73, v69, v28
	v_sub_u32_e32 v69, v59, v28
	v_sub_co_u32_e32 v63, vcc, v63, v72
	s_nop 1
	v_subb_co_u32_e64 v69, s[2:3], v69, v25, vcc
	v_subrev_co_u32_e64 v71, s[2:3], s20, v63
	v_subb_co_u32_e32 v28, vcc, v59, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[2:3], 0, v69, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v69
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v71
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v63
	v_cndmask_b32_e64 v71, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v69
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v69, v72, v71, s[2:3]
	v_add_u32_e32 v71, 2, v68
	v_add_u32_e32 v72, 1, v68
	v_cmp_ne_u32_e64 s[2:3], 0, v69
	v_cndmask_b32_e32 v28, v59, v63, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v69, v72, v71, s[2:3]
	v_xor_b32_e32 v59, s72, v70
	v_cndmask_b32_e32 v28, v68, v69, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_mov_b32_e32 v69, v68
	v_xor_b32_e32 v28, v28, v59
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_sub_u32_e32 v28, v28, v59
	v_xor_b32_e32 v69, v66, v68
	v_xad_u32 v59, v28, v68, v37
	v_xor_b32_e32 v63, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v69, s16, 0
	v_mul_hi_u32 v28, v69, s1
	v_lshl_add_u64 v[66:67], v[28:29], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v63, s1, 0
	v_add_co_u32_e32 v28, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v63, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s0
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v28, s21, v66
	v_mul_lo_u32 v70, s20, v67
	v_mad_u64_u32 v[66:67], s[2:3], s20, v66, 0
	v_add3_u32 v28, v67, v70, v28
	v_sub_u32_e32 v67, v63, v28
	v_sub_co_u32_e32 v66, vcc, v69, v66
	v_mul_lo_u32 v59, v59, s33
	s_nop 0
	v_subb_co_u32_e64 v67, s[2:3], v67, v25, vcc
	v_subrev_co_u32_e64 v69, s[2:3], s20, v66
	v_subb_co_u32_e32 v28, vcc, v63, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v70, s[14:15], 0, v67, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v70
	v_subb_co_u32_e64 v67, s[2:3], v67, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v69
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v70
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v66
	v_cndmask_b32_e64 v71, v71, v72, s[14:15]
	v_subrev_co_u32_e64 v72, s[2:3], s20, v69
	s_nop 1
	v_subbrev_co_u32_e64 v67, s[2:3], 0, v67, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v71
	s_nop 1
	v_cndmask_b32_e64 v67, v70, v67, s[2:3]
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v63, v63, v70, vcc
	v_cmp_ne_u32_e32 vcc, 0, v63
	v_cndmask_b32_e64 v63, v69, v72, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v63, v66, v63, vcc
	v_cndmask_b32_e32 v28, v28, v67, vcc
	v_xor_b32_e32 v63, v63, v68
	v_xor_b32_e32 v28, v28, v68
	v_sub_co_u32_e32 v63, vcc, v63, v68
	v_lshl_add_u64 v[66:67], v[34:35], 0, 14
	s_nop 0
	v_subb_co_u32_e32 v28, vcc, v28, v68, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v77, v28, v63, v59
	v_sub_co_u32_e32 v28, vcc, -15, v34
	s_nop 1
	v_subb_co_u32_e32 v59, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v69, v67, v59, vcc
	v_ashrrev_i32_e32 v70, 31, v69
	v_cndmask_b32_e32 v68, v66, v28, vcc
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[70:71]
	v_xor_b32_e32 v63, v68, v70
	v_xor_b32_e32 v59, v69, v70
	v_mad_u64_u32 v[68:69], s[2:3], v63, s16, 0
	v_mul_hi_u32 v28, v63, s1
	v_lshl_add_u64 v[68:69], v[28:29], 0, v[68:69]
	v_mad_u64_u32 v[74:75], s[2:3], v59, s1, 0
	v_add_co_u32_e32 v28, vcc, v68, v74
	v_mad_u64_u32 v[72:73], s[2:3], v59, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v69, v75, vcc
	v_mov_b32_e32 v69, s0
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[72:73]
	v_mul_lo_u32 v28, s21, v68
	v_mul_lo_u32 v69, s20, v69
	v_mad_u64_u32 v[72:73], s[2:3], s20, v68, 0
	v_add3_u32 v28, v73, v69, v28
	v_sub_u32_e32 v69, v59, v28
	v_sub_co_u32_e32 v63, vcc, v63, v72
	s_nop 1
	v_subb_co_u32_e64 v69, s[2:3], v69, v25, vcc
	v_subrev_co_u32_e64 v71, s[2:3], s20, v63
	v_subb_co_u32_e32 v28, vcc, v59, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[2:3], 0, v69, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v69
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v71
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v63
	v_cndmask_b32_e64 v71, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v69
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v69, v72, v71, s[2:3]
	v_add_u32_e32 v71, 2, v68
	v_add_u32_e32 v72, 1, v68
	v_cmp_ne_u32_e64 s[2:3], 0, v69
	v_cndmask_b32_e32 v28, v59, v63, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v69, v72, v71, s[2:3]
	v_xor_b32_e32 v59, s72, v70
	v_cndmask_b32_e32 v28, v68, v69, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_mov_b32_e32 v69, v68
	v_xor_b32_e32 v28, v28, v59
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_sub_u32_e32 v28, v28, v59
	v_xor_b32_e32 v69, v66, v68
	v_xad_u32 v59, v28, v68, v37
	v_xor_b32_e32 v63, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v69, s16, 0
	v_mul_hi_u32 v28, v69, s1
	v_lshl_add_u64 v[66:67], v[28:29], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v63, s1, 0
	v_add_co_u32_e32 v28, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v63, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s0
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v28, s21, v66
	v_mul_lo_u32 v70, s20, v67
	v_mad_u64_u32 v[66:67], s[2:3], s20, v66, 0
	v_add3_u32 v28, v67, v70, v28
	v_sub_u32_e32 v67, v63, v28
	v_sub_co_u32_e32 v66, vcc, v69, v66
	v_mul_lo_u32 v59, v59, s33
	s_nop 0
	v_subb_co_u32_e64 v67, s[2:3], v67, v25, vcc
	v_subrev_co_u32_e64 v69, s[2:3], s20, v66
	v_subb_co_u32_e32 v28, vcc, v63, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v70, s[14:15], 0, v67, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v70
	v_subb_co_u32_e64 v67, s[2:3], v67, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v69
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v70
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v66
	v_cndmask_b32_e64 v71, v71, v72, s[14:15]
	v_subrev_co_u32_e64 v72, s[2:3], s20, v69
	s_nop 1
	v_subbrev_co_u32_e64 v67, s[2:3], 0, v67, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v71
	s_nop 1
	v_cndmask_b32_e64 v67, v70, v67, s[2:3]
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v63, v63, v70, vcc
	v_cmp_ne_u32_e32 vcc, 0, v63
	v_cndmask_b32_e64 v63, v69, v72, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v63, v66, v63, vcc
	v_cndmask_b32_e32 v28, v28, v67, vcc
	v_xor_b32_e32 v63, v63, v68
	v_xor_b32_e32 v28, v28, v68
	v_sub_co_u32_e32 v63, vcc, v63, v68
	v_lshl_add_u64 v[66:67], v[34:35], 0, 15
	s_nop 0
	v_subb_co_u32_e32 v28, vcc, v28, v68, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v74, v28, v63, v59
	v_sub_co_u32_e32 v28, vcc, -16, v34
	s_nop 1
	v_subb_co_u32_e32 v34, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v35, v67, v34, vcc
	v_ashrrev_i32_e32 v68, 31, v35
	v_cndmask_b32_e32 v34, v66, v28, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[68:69]
	v_xor_b32_e32 v63, v34, v68
	v_xor_b32_e32 v59, v35, v68
	v_mad_u64_u32 v[34:35], s[2:3], v63, s16, 0
	v_mul_hi_u32 v28, v63, s1
	v_lshl_add_u64 v[34:35], v[28:29], 0, v[34:35]
	v_mad_u64_u32 v[72:73], s[2:3], v59, s1, 0
	v_add_co_u32_e32 v28, vcc, v34, v72
	v_mad_u64_u32 v[70:71], s[2:3], v59, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v34, vcc, v35, v73, vcc
	v_mov_b32_e32 v35, s0
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[70:71]
	v_mul_lo_u32 v28, s21, v34
	v_mul_lo_u32 v35, s20, v35
	v_mad_u64_u32 v[70:71], s[2:3], s20, v34, 0
	v_add3_u32 v28, v71, v35, v28
	v_sub_u32_e32 v35, v59, v28
	v_sub_co_u32_e32 v63, vcc, v63, v70
	s_nop 1
	v_subb_co_u32_e64 v35, s[2:3], v35, v25, vcc
	v_subrev_co_u32_e64 v69, s[2:3], s20, v63
	v_subb_co_u32_e32 v28, vcc, v59, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v35, s[2:3], 0, v35, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v35
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v69
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v63
	v_cndmask_b32_e64 v69, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v35
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v35, v70, v69, s[2:3]
	v_add_u32_e32 v69, 2, v34
	v_add_u32_e32 v70, 1, v34
	v_cmp_ne_u32_e64 s[2:3], 0, v35
	v_cndmask_b32_e32 v28, v59, v63, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v35, v70, v69, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v28, v34, v35, vcc
	v_xor_b32_e32 v34, s72, v68
	v_xor_b32_e32 v28, v28, v34
	v_sub_u32_e32 v28, v28, v34
	v_ashrrev_i32_e32 v34, 31, v67
	v_mov_b32_e32 v35, v34
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[34:35]
	v_xor_b32_e32 v63, v66, v34
	v_xad_u32 v59, v28, v34, v37
	v_xor_b32_e32 v35, v67, v34
	v_mad_u64_u32 v[66:67], s[2:3], v63, s16, 0
	v_mul_hi_u32 v28, v63, s1
	v_lshl_add_u64 v[66:67], v[28:29], 0, v[66:67]
	v_mad_u64_u32 v[70:71], s[2:3], v35, s1, 0
	v_add_co_u32_e32 v28, vcc, v66, v70
	v_mad_u64_u32 v[68:69], s[2:3], v35, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v71, vcc
	v_mov_b32_e32 v67, s0
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_mul_lo_u32 v28, s21, v66
	v_mul_lo_u32 v68, s20, v67
	v_mad_u64_u32 v[66:67], s[2:3], s20, v66, 0
	v_add3_u32 v28, v67, v68, v28
	v_sub_u32_e32 v67, v35, v28
	v_sub_co_u32_e32 v63, vcc, v63, v66
	s_nop 1
	v_subb_co_u32_e64 v66, s[2:3], v67, v25, vcc
	v_subrev_co_u32_e64 v67, s[2:3], s20, v63
	v_subb_co_u32_e32 v28, vcc, v35, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v68, s[14:15], 0, v66, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v68
	v_subb_co_u32_e64 v66, s[2:3], v66, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v67
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v68
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v63
	v_cndmask_b32_e64 v69, v69, v70, s[14:15]
	v_subrev_co_u32_e64 v70, s[2:3], s20, v67
	s_nop 1
	v_subbrev_co_u32_e64 v66, s[2:3], 0, v66, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v69
	s_nop 1
	v_cndmask_b32_e64 v66, v68, v66, s[2:3]
	v_cndmask_b32_e64 v68, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v35, v35, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v35
	v_cndmask_b32_e64 v35, v67, v70, s[2:3]
	s_movk_i32 s2, 0xfc00
	v_cndmask_b32_e32 v35, v63, v35, vcc
	v_cndmask_b32_e32 v28, v28, v66, vcc
	v_xor_b32_e32 v35, v35, v34
	v_xor_b32_e32 v28, v28, v34
	v_sub_co_u32_e32 v35, vcc, v35, v34
	s_mov_b32 s3, -1
	s_nop 0
	v_subb_co_u32_e32 v28, vcc, v28, v34, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_mul_lo_u32 v34, v59, s33
	v_add3_u32 v28, v28, v35, v34
	v_lshl_add_u64 v[34:35], v[30:31], 0, s[2:3]
	s_movk_i32 s2, 0x3ff
	buffer_load_ubyte v59, v38, s[36:39], 0 offen
	buffer_load_ubyte v63, v39, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v61, v61, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v65, v65, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v66, v76, s[36:39], 0 offen
	buffer_load_ubyte v68, v77, s[36:39], 0 offen
	buffer_load_ubyte v67, v74, s[36:39], 0 offen
	buffer_load_ubyte v69, v28, s[36:39], 0 offen
	v_sub_co_u32_e32 v28, vcc, s2, v30
	s_nop 1
	v_subb_co_u32_e32 v31, vcc, 0, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v35
	s_nop 1
	v_cndmask_b32_e32 v31, v35, v31, vcc
	v_ashrrev_i32_e32 v38, 31, v31
	v_cndmask_b32_e32 v30, v34, v28, vcc
	v_mov_b32_e32 v39, v38
	v_lshl_add_u64 v[30:31], v[30:31], 0, v[38:39]
	v_xor_b32_e32 v74, v30, v38
	v_xor_b32_e32 v39, v31, v38
	v_mad_u64_u32 v[30:31], s[2:3], v74, s16, 0
	v_mul_hi_u32 v28, v74, s1
	v_lshl_add_u64 v[30:31], v[28:29], 0, v[30:31]
	v_mad_u64_u32 v[72:73], s[2:3], v39, s1, 0
	v_add_co_u32_e32 v28, vcc, v30, v72
	v_mad_u64_u32 v[70:71], s[2:3], v39, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v30, vcc, v31, v73, vcc
	v_mov_b32_e32 v31, s0
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[30:31], v[30:31], 0, v[70:71]
	v_mul_lo_u32 v28, s21, v30
	v_mul_lo_u32 v31, s20, v31
	v_mad_u64_u32 v[70:71], s[2:3], s20, v30, 0
	v_add3_u32 v28, v71, v31, v28
	v_sub_u32_e32 v31, v39, v28
	v_sub_co_u32_e32 v70, vcc, v74, v70
	s_nop 1
	v_subb_co_u32_e64 v31, s[2:3], v31, v25, vcc
	v_subrev_co_u32_e64 v71, s[2:3], s20, v70
	v_subb_co_u32_e32 v28, vcc, v39, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v31, s[2:3], 0, v31, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v31
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v71
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v70
	v_cndmask_b32_e64 v71, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v31
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v31, v72, v71, s[2:3]
	v_add_u32_e32 v71, 2, v30
	v_add_u32_e32 v72, 1, v30
	v_cmp_ne_u32_e64 s[2:3], 0, v31
	v_cndmask_b32_e32 v28, v39, v70, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v31, v72, v71, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v28, v30, v31, vcc
	v_xor_b32_e32 v30, s72, v38
	v_xor_b32_e32 v28, v28, v30
	v_sub_u32_e32 v28, v28, v30
	v_ashrrev_i32_e32 v30, 31, v35
	v_mov_b32_e32 v31, v30
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[30:31]
	v_xor_b32_e32 v73, v34, v30
	v_xad_u32 v72, v28, v30, v37
	v_xor_b32_e32 v31, v35, v30
	v_mad_u64_u32 v[34:35], s[2:3], v73, s16, 0
	v_mul_hi_u32 v28, v73, s1
	v_lshl_add_u64 v[34:35], v[28:29], 0, v[34:35]
	v_mad_u64_u32 v[70:71], s[2:3], v31, s1, 0
	v_add_co_u32_e32 v28, vcc, v34, v70
	v_mad_u64_u32 v[38:39], s[2:3], v31, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v34, vcc, v35, v71, vcc
	v_mov_b32_e32 v35, s0
	s_nop 0
	v_addc_co_u32_e32 v39, vcc, 0, v39, vcc
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[38:39]
	v_mul_lo_u32 v28, s21, v34
	v_mul_lo_u32 v38, s20, v35
	v_mad_u64_u32 v[34:35], s[2:3], s20, v34, 0
	v_add3_u32 v28, v35, v38, v28
	v_sub_u32_e32 v35, v31, v28
	v_sub_co_u32_e32 v34, vcc, v73, v34
	s_nop 1
	v_subb_co_u32_e64 v35, s[2:3], v35, v25, vcc
	v_subrev_co_u32_e64 v38, s[2:3], s20, v34
	v_subb_co_u32_e32 v28, vcc, v31, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v39, s[14:15], 0, v35, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v39
	v_subb_co_u32_e64 v35, s[2:3], v35, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v38
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v39
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v34
	v_cndmask_b32_e64 v70, v70, v71, s[14:15]
	v_subrev_co_u32_e64 v71, s[2:3], s20, v38
	s_nop 1
	v_subbrev_co_u32_e64 v35, s[2:3], 0, v35, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v70
	s_nop 1
	v_cndmask_b32_e64 v35, v39, v35, s[2:3]
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v31, v31, v39, vcc
	v_cmp_ne_u32_e32 vcc, 0, v31
	v_cndmask_b32_e64 v31, v38, v71, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v31, v34, v31, vcc
	v_cndmask_b32_e32 v28, v28, v35, vcc
	v_xor_b32_e32 v31, v31, v30
	v_xor_b32_e32 v28, v28, v30
	v_sub_co_u32_e32 v31, vcc, v31, v30
	s_nop 1
	v_subb_co_u32_e32 v28, vcc, v28, v30, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_mul_lo_u32 v30, v72, s33
	v_add3_u32 v34, v28, v31, v30
	v_or_b32_e32 v30, 0xfffffc00, v36
	v_ashrrev_i32_e32 v31, 31, v30
	v_lshl_add_u64 v[30:31], v[32:33], 0, v[30:31]
	v_sub_co_u32_e32 v28, vcc, -2, v30
	v_lshl_add_u64 v[32:33], v[30:31], 0, 1
	s_nop 0
	v_subb_co_u32_e32 v35, vcc, -1, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v33
	s_nop 1
	v_cndmask_b32_e32 v39, v33, v35, vcc
	v_ashrrev_i32_e32 v70, 31, v39
	v_cndmask_b32_e32 v38, v32, v28, vcc
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[70:71]
	v_xor_b32_e32 v36, v38, v70
	v_xor_b32_e32 v35, v39, v70
	v_mad_u64_u32 v[38:39], s[2:3], v36, s16, 0
	v_mul_hi_u32 v28, v36, s1
	v_lshl_add_u64 v[38:39], v[28:29], 0, v[38:39]
	v_mad_u64_u32 v[74:75], s[2:3], v35, s1, 0
	v_add_co_u32_e32 v28, vcc, v38, v74
	v_mad_u64_u32 v[72:73], s[2:3], v35, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v75, vcc
	v_mov_b32_e32 v39, s0
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[72:73]
	v_mul_lo_u32 v28, s21, v38
	v_mul_lo_u32 v39, s20, v39
	v_mad_u64_u32 v[72:73], s[2:3], s20, v38, 0
	v_add3_u32 v28, v73, v39, v28
	v_sub_u32_e32 v39, v35, v28
	v_sub_co_u32_e32 v36, vcc, v36, v72
	s_nop 1
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, vcc
	v_subrev_co_u32_e64 v71, s[2:3], s20, v36
	v_subb_co_u32_e32 v28, vcc, v35, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v39, s[2:3], 0, v39, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v39
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v71
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v36
	v_cndmask_b32_e64 v71, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v39
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v39, v72, v71, s[2:3]
	v_add_u32_e32 v71, 2, v38
	v_add_u32_e32 v72, 1, v38
	v_cmp_ne_u32_e64 s[2:3], 0, v39
	v_cndmask_b32_e32 v28, v35, v36, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v39, v72, v71, s[2:3]
	v_xor_b32_e32 v35, s72, v70
	v_cndmask_b32_e32 v28, v38, v39, vcc
	v_ashrrev_i32_e32 v38, 31, v33
	v_mov_b32_e32 v39, v38
	v_xor_b32_e32 v28, v28, v35
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[38:39]
	v_sub_u32_e32 v28, v28, v35
	v_xor_b32_e32 v39, v32, v38
	v_xad_u32 v35, v28, v38, v37
	v_xor_b32_e32 v36, v33, v38
	v_mad_u64_u32 v[32:33], s[2:3], v39, s16, 0
	v_mul_hi_u32 v28, v39, s1
	v_lshl_add_u64 v[32:33], v[28:29], 0, v[32:33]
	v_mad_u64_u32 v[72:73], s[2:3], v36, s1, 0
	v_add_co_u32_e32 v28, vcc, v32, v72
	v_mad_u64_u32 v[70:71], s[2:3], v36, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v33, v73, vcc
	v_mov_b32_e32 v33, s0
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[70:71]
	v_mul_lo_u32 v28, s21, v32
	v_mul_lo_u32 v70, s20, v33
	v_mad_u64_u32 v[32:33], s[2:3], s20, v32, 0
	v_add3_u32 v28, v33, v70, v28
	v_sub_u32_e32 v33, v36, v28
	v_sub_co_u32_e32 v32, vcc, v39, v32
	s_nop 1
	v_subb_co_u32_e64 v33, s[2:3], v33, v25, vcc
	v_subrev_co_u32_e64 v39, s[2:3], s20, v32
	v_subb_co_u32_e32 v28, vcc, v36, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v70, s[14:15], 0, v33, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v70
	v_subb_co_u32_e64 v33, s[2:3], v33, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v39
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v70
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v32
	v_cndmask_b32_e64 v71, v71, v72, s[14:15]
	v_subrev_co_u32_e64 v72, s[2:3], s20, v39
	s_nop 1
	v_subbrev_co_u32_e64 v33, s[2:3], 0, v33, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v71
	s_nop 1
	v_cndmask_b32_e64 v33, v70, v33, s[2:3]
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v36, v36, v70, vcc
	v_cmp_ne_u32_e32 vcc, 0, v36
	s_nop 1
	v_cndmask_b32_e32 v28, v28, v33, vcc
	v_cndmask_b32_e64 v33, v39, v72, s[2:3]
	v_cndmask_b32_e32 v32, v32, v33, vcc
	v_xor_b32_e32 v32, v32, v38
	v_xor_b32_e32 v28, v28, v38
	v_sub_co_u32_e32 v32, vcc, v32, v38
	v_mul_lo_u32 v33, v35, s33
	s_nop 0
	v_subb_co_u32_e32 v28, vcc, v28, v38, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v32, v28, v32, v33
	v_sub_co_u32_e32 v28, vcc, -3, v30
	v_lshl_add_u64 v[38:39], v[30:31], 0, 2
	s_nop 0
	v_subb_co_u32_e32 v33, vcc, -1, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v39
	s_nop 1
	v_cndmask_b32_e32 v71, v39, v33, vcc
	v_ashrrev_i32_e32 v72, 31, v71
	v_cndmask_b32_e32 v70, v38, v28, vcc
	v_mov_b32_e32 v73, v72
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[72:73]
	v_xor_b32_e32 v35, v70, v72
	v_xor_b32_e32 v33, v71, v72
	v_mad_u64_u32 v[70:71], s[2:3], v35, s16, 0
	v_mul_hi_u32 v28, v35, s1
	v_lshl_add_u64 v[70:71], v[28:29], 0, v[70:71]
	v_mad_u64_u32 v[76:77], s[2:3], v33, s1, 0
	v_add_co_u32_e32 v28, vcc, v70, v76
	v_mad_u64_u32 v[74:75], s[2:3], v33, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v71, v77, vcc
	v_mov_b32_e32 v71, s0
	s_nop 0
	v_addc_co_u32_e32 v75, vcc, 0, v75, vcc
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[74:75]
	v_mul_lo_u32 v28, s21, v70
	v_mul_lo_u32 v36, s20, v71
	v_mad_u64_u32 v[74:75], s[2:3], s20, v70, 0
	v_add3_u32 v28, v75, v36, v28
	v_sub_u32_e32 v36, v33, v28
	v_sub_co_u32_e32 v35, vcc, v35, v74
	s_nop 1
	v_subb_co_u32_e64 v36, s[2:3], v36, v25, vcc
	v_subrev_co_u32_e64 v71, s[2:3], s20, v35
	v_subb_co_u32_e32 v28, vcc, v33, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v36, s[2:3], 0, v36, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v36
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v73, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v71
	v_cndmask_b32_e64 v33, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v35
	v_cndmask_b32_e64 v71, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v36
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v36, v73, v71, s[2:3]
	v_add_u32_e32 v71, 2, v70
	v_add_u32_e32 v73, 1, v70
	v_cmp_ne_u32_e64 s[2:3], 0, v36
	v_cndmask_b32_e32 v28, v33, v35, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v36, v73, v71, s[2:3]
	v_xor_b32_e32 v33, s72, v72
	v_cndmask_b32_e32 v28, v70, v36, vcc
	v_ashrrev_i32_e32 v70, 31, v39
	v_mov_b32_e32 v71, v70
	v_xor_b32_e32 v28, v28, v33
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[70:71]
	v_sub_u32_e32 v28, v28, v33
	v_xor_b32_e32 v36, v38, v70
	v_xad_u32 v33, v28, v70, v37
	v_xor_b32_e32 v35, v39, v70
	v_mad_u64_u32 v[38:39], s[2:3], v36, s16, 0
	v_mul_hi_u32 v28, v36, s1
	v_lshl_add_u64 v[38:39], v[28:29], 0, v[38:39]
	v_mad_u64_u32 v[74:75], s[2:3], v35, s1, 0
	v_add_co_u32_e32 v28, vcc, v38, v74
	v_mad_u64_u32 v[72:73], s[2:3], v35, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v75, vcc
	v_mov_b32_e32 v39, s0
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[72:73]
	v_mul_lo_u32 v28, s21, v38
	v_mul_lo_u32 v71, s20, v39
	v_mad_u64_u32 v[38:39], s[2:3], s20, v38, 0
	v_add3_u32 v28, v39, v71, v28
	v_sub_u32_e32 v39, v35, v28
	v_sub_co_u32_e32 v36, vcc, v36, v38
	v_mul_lo_u32 v33, v33, s33
	s_nop 0
	v_subb_co_u32_e64 v38, s[2:3], v39, v25, vcc
	v_subrev_co_u32_e64 v39, s[2:3], s20, v36
	v_subb_co_u32_e32 v28, vcc, v35, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v71, s[14:15], 0, v38, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v71
	v_subb_co_u32_e64 v38, s[2:3], v38, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v39
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v73, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v71
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v36
	v_cndmask_b32_e64 v72, v72, v73, s[14:15]
	v_subrev_co_u32_e64 v73, s[2:3], s20, v39
	s_nop 1
	v_subbrev_co_u32_e64 v38, s[2:3], 0, v38, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v72
	s_nop 1
	v_cndmask_b32_e64 v38, v71, v38, s[2:3]
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v35, v35, v71, vcc
	v_cmp_ne_u32_e32 vcc, 0, v35
	v_cndmask_b32_e64 v35, v39, v73, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v35, v36, v35, vcc
	v_cndmask_b32_e32 v28, v28, v38, vcc
	v_xor_b32_e32 v35, v35, v70
	v_xor_b32_e32 v28, v28, v70
	v_sub_co_u32_e32 v35, vcc, v35, v70
	v_lshl_add_u64 v[38:39], v[30:31], 0, 3
	s_nop 0
	v_subb_co_u32_e32 v28, vcc, v28, v70, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v33, v28, v35, v33
	v_sub_co_u32_e32 v28, vcc, -4, v30
	s_nop 1
	v_subb_co_u32_e32 v35, vcc, -1, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v39
	s_nop 1
	v_cndmask_b32_e32 v71, v39, v35, vcc
	v_ashrrev_i32_e32 v72, 31, v71
	v_cndmask_b32_e32 v70, v38, v28, vcc
	v_mov_b32_e32 v73, v72
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[72:73]
	v_xor_b32_e32 v36, v70, v72
	v_xor_b32_e32 v35, v71, v72
	v_mad_u64_u32 v[70:71], s[2:3], v36, s16, 0
	v_mul_hi_u32 v28, v36, s1
	v_lshl_add_u64 v[70:71], v[28:29], 0, v[70:71]
	v_mad_u64_u32 v[76:77], s[2:3], v35, s1, 0
	v_add_co_u32_e32 v28, vcc, v70, v76
	v_mad_u64_u32 v[74:75], s[2:3], v35, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v71, v77, vcc
	v_mov_b32_e32 v71, s0
	s_nop 0
	v_addc_co_u32_e32 v75, vcc, 0, v75, vcc
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[74:75]
	v_mul_lo_u32 v28, s21, v70
	v_mul_lo_u32 v71, s20, v71
	v_mad_u64_u32 v[74:75], s[2:3], s20, v70, 0
	v_add3_u32 v28, v75, v71, v28
	v_sub_u32_e32 v71, v35, v28
	v_sub_co_u32_e32 v36, vcc, v36, v74
	s_nop 1
	v_subb_co_u32_e64 v71, s[2:3], v71, v25, vcc
	v_subrev_co_u32_e64 v73, s[2:3], s20, v36
	v_subb_co_u32_e32 v28, vcc, v35, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v71, s[2:3], 0, v71, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v71
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v74, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v73
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v36
	v_cndmask_b32_e64 v73, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v71
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v71, v74, v73, s[2:3]
	v_add_u32_e32 v73, 2, v70
	v_add_u32_e32 v74, 1, v70
	v_cmp_ne_u32_e64 s[2:3], 0, v71
	v_cndmask_b32_e32 v28, v35, v36, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v71, v74, v73, s[2:3]
	v_xor_b32_e32 v35, s72, v72
	v_cndmask_b32_e32 v28, v70, v71, vcc
	v_ashrrev_i32_e32 v70, 31, v39
	v_mov_b32_e32 v71, v70
	v_xor_b32_e32 v28, v28, v35
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[70:71]
	v_sub_u32_e32 v28, v28, v35
	v_xor_b32_e32 v71, v38, v70
	v_xad_u32 v35, v28, v70, v37
	v_xor_b32_e32 v36, v39, v70
	v_mad_u64_u32 v[38:39], s[2:3], v71, s16, 0
	v_mul_hi_u32 v28, v71, s1
	v_lshl_add_u64 v[38:39], v[28:29], 0, v[38:39]
	v_mad_u64_u32 v[74:75], s[2:3], v36, s1, 0
	v_add_co_u32_e32 v28, vcc, v38, v74
	v_mad_u64_u32 v[72:73], s[2:3], v36, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v75, vcc
	v_mov_b32_e32 v39, s0
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[72:73]
	v_mul_lo_u32 v28, s21, v38
	v_mul_lo_u32 v72, s20, v39
	v_mad_u64_u32 v[38:39], s[2:3], s20, v38, 0
	v_add3_u32 v28, v39, v72, v28
	v_sub_u32_e32 v39, v36, v28
	v_sub_co_u32_e32 v38, vcc, v71, v38
	v_mul_lo_u32 v35, v35, s33
	s_nop 0
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, vcc
	v_subrev_co_u32_e64 v71, s[2:3], s20, v38
	v_subb_co_u32_e32 v28, vcc, v36, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v72, s[14:15], 0, v39, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v72
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v73, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v71
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v74, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v72
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v38
	v_cndmask_b32_e64 v73, v73, v74, s[14:15]
	v_subrev_co_u32_e64 v74, s[2:3], s20, v71
	s_nop 1
	v_subbrev_co_u32_e64 v39, s[2:3], 0, v39, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v73
	s_nop 1
	v_cndmask_b32_e64 v39, v72, v39, s[2:3]
	v_cndmask_b32_e64 v72, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v36, v36, v72, vcc
	v_cmp_ne_u32_e32 vcc, 0, v36
	v_cndmask_b32_e64 v36, v71, v74, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v36, v38, v36, vcc
	v_cndmask_b32_e32 v28, v28, v39, vcc
	v_xor_b32_e32 v36, v36, v70
	v_xor_b32_e32 v28, v28, v70
	v_sub_co_u32_e32 v36, vcc, v36, v70
	v_lshl_add_u64 v[38:39], v[30:31], 0, 4
	s_nop 0
	v_subb_co_u32_e32 v28, vcc, v28, v70, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v35, v28, v36, v35
	v_sub_co_u32_e32 v28, vcc, -5, v30
	s_nop 1
	v_subb_co_u32_e32 v36, vcc, -1, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v39
	s_nop 1
	v_cndmask_b32_e32 v71, v39, v36, vcc
	v_ashrrev_i32_e32 v72, 31, v71
	v_cndmask_b32_e32 v70, v38, v28, vcc
	v_mov_b32_e32 v73, v72
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[72:73]
	v_xor_b32_e32 v73, v70, v72
	v_xor_b32_e32 v36, v71, v72
	v_mad_u64_u32 v[70:71], s[2:3], v73, s16, 0
	v_mul_hi_u32 v28, v73, s1
	v_lshl_add_u64 v[70:71], v[28:29], 0, v[70:71]
	v_mad_u64_u32 v[76:77], s[2:3], v36, s1, 0
	v_add_co_u32_e32 v28, vcc, v70, v76
	v_mad_u64_u32 v[74:75], s[2:3], v36, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v71, v77, vcc
	v_mov_b32_e32 v71, s0
	s_nop 0
	v_addc_co_u32_e32 v75, vcc, 0, v75, vcc
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[74:75]
	v_mul_lo_u32 v28, s21, v70
	v_mul_lo_u32 v71, s20, v71
	v_mad_u64_u32 v[74:75], s[2:3], s20, v70, 0
	v_add3_u32 v28, v75, v71, v28
	v_sub_u32_e32 v71, v36, v28
	v_sub_co_u32_e32 v73, vcc, v73, v74
	s_nop 1
	v_subb_co_u32_e64 v71, s[2:3], v71, v25, vcc
	v_subrev_co_u32_e64 v74, s[2:3], s20, v73
	v_subb_co_u32_e32 v28, vcc, v36, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v71, s[2:3], 0, v71, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v71
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v75, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v74
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v73
	v_cndmask_b32_e64 v74, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v71
	v_cndmask_b32_e64 v73, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v71, v75, v74, s[2:3]
	v_add_u32_e32 v74, 2, v70
	v_add_u32_e32 v75, 1, v70
	v_cmp_ne_u32_e64 s[2:3], 0, v71
	v_cndmask_b32_e32 v28, v36, v73, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v71, v75, v74, s[2:3]
	v_xor_b32_e32 v36, s72, v72
	v_cndmask_b32_e32 v28, v70, v71, vcc
	v_ashrrev_i32_e32 v70, 31, v39
	v_mov_b32_e32 v71, v70
	v_xor_b32_e32 v28, v28, v36
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[70:71]
	v_sub_u32_e32 v28, v28, v36
	v_xor_b32_e32 v76, v38, v70
	v_xad_u32 v36, v28, v70, v37
	v_xor_b32_e32 v71, v39, v70
	v_mad_u64_u32 v[38:39], s[2:3], v76, s16, 0
	v_mul_hi_u32 v28, v76, s1
	v_lshl_add_u64 v[38:39], v[28:29], 0, v[38:39]
	v_mad_u64_u32 v[74:75], s[2:3], v71, s1, 0
	v_add_co_u32_e32 v28, vcc, v38, v74
	v_mad_u64_u32 v[72:73], s[2:3], v71, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v75, vcc
	v_mov_b32_e32 v39, s0
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[72:73]
	v_mul_lo_u32 v28, s21, v38
	v_mul_lo_u32 v72, s20, v39
	v_mad_u64_u32 v[38:39], s[2:3], s20, v38, 0
	v_add3_u32 v28, v39, v72, v28
	v_sub_u32_e32 v39, v71, v28
	v_sub_co_u32_e32 v38, vcc, v76, v38
	v_mul_lo_u32 v36, v36, s33
	s_nop 0
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, vcc
	v_subrev_co_u32_e64 v72, s[2:3], s20, v38
	v_subb_co_u32_e32 v28, vcc, v71, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v73, s[14:15], 0, v39, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v73
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v74, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v72
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v75, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v73
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v38
	v_cndmask_b32_e64 v74, v74, v75, s[14:15]
	v_subrev_co_u32_e64 v75, s[2:3], s20, v72
	s_nop 1
	v_subbrev_co_u32_e64 v39, s[2:3], 0, v39, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v74
	s_nop 1
	v_cndmask_b32_e64 v39, v73, v39, s[2:3]
	v_cndmask_b32_e64 v73, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v71, v71, v73, vcc
	v_cmp_ne_u32_e32 vcc, 0, v71
	s_nop 1
	v_cndmask_b32_e32 v28, v28, v39, vcc
	v_cndmask_b32_e64 v39, v72, v75, s[2:3]
	v_cndmask_b32_e32 v38, v38, v39, vcc
	v_xor_b32_e32 v38, v38, v70
	v_xor_b32_e32 v28, v28, v70
	v_sub_co_u32_e32 v38, vcc, v38, v70
	s_nop 1
	v_subb_co_u32_e32 v28, vcc, v28, v70, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v36, v28, v38, v36
	v_sub_co_u32_e32 v28, vcc, -6, v30
	v_lshl_add_u64 v[38:39], v[30:31], 0, 5
	s_nop 0
	v_subb_co_u32_e32 v70, vcc, -1, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v39
	s_nop 1
	v_cndmask_b32_e32 v71, v39, v70, vcc
	v_ashrrev_i32_e32 v72, 31, v71
	v_cndmask_b32_e32 v70, v38, v28, vcc
	v_mov_b32_e32 v73, v72
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[72:73]
	v_xor_b32_e32 v78, v70, v72
	v_xor_b32_e32 v73, v71, v72
	v_mad_u64_u32 v[70:71], s[2:3], v78, s16, 0
	v_mul_hi_u32 v28, v78, s1
	v_lshl_add_u64 v[70:71], v[28:29], 0, v[70:71]
	v_mad_u64_u32 v[76:77], s[2:3], v73, s1, 0
	v_add_co_u32_e32 v28, vcc, v70, v76
	v_mad_u64_u32 v[74:75], s[2:3], v73, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v71, v77, vcc
	v_mov_b32_e32 v71, s0
	s_nop 0
	v_addc_co_u32_e32 v75, vcc, 0, v75, vcc
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[74:75]
	v_mul_lo_u32 v28, s21, v70
	v_mul_lo_u32 v71, s20, v71
	v_mad_u64_u32 v[74:75], s[2:3], s20, v70, 0
	v_add3_u32 v28, v75, v71, v28
	v_sub_u32_e32 v71, v73, v28
	v_sub_co_u32_e32 v74, vcc, v78, v74
	s_nop 1
	v_subb_co_u32_e64 v71, s[2:3], v71, v25, vcc
	v_subrev_co_u32_e64 v75, s[2:3], s20, v74
	v_subb_co_u32_e32 v28, vcc, v73, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v71, s[2:3], 0, v71, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v71
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v76, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v75
	v_cndmask_b32_e64 v73, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v74
	v_cndmask_b32_e64 v75, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v71
	v_cndmask_b32_e64 v74, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v71, v76, v75, s[2:3]
	v_add_u32_e32 v75, 2, v70
	v_add_u32_e32 v76, 1, v70
	v_cmp_ne_u32_e64 s[2:3], 0, v71
	v_cndmask_b32_e32 v28, v73, v74, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v71, v76, v75, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v28, v70, v71, vcc
	v_xor_b32_e32 v70, s72, v72
	v_xor_b32_e32 v28, v28, v70
	v_sub_u32_e32 v28, v28, v70
	v_ashrrev_i32_e32 v70, 31, v39
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[70:71]
	v_xor_b32_e32 v77, v38, v70
	v_xad_u32 v76, v28, v70, v37
	v_xor_b32_e32 v71, v39, v70
	v_mad_u64_u32 v[38:39], s[2:3], v77, s16, 0
	v_mul_hi_u32 v28, v77, s1
	v_lshl_add_u64 v[38:39], v[28:29], 0, v[38:39]
	v_mad_u64_u32 v[74:75], s[2:3], v71, s1, 0
	v_add_co_u32_e32 v28, vcc, v38, v74
	v_mad_u64_u32 v[72:73], s[2:3], v71, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v75, vcc
	v_mov_b32_e32 v39, s0
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[72:73]
	v_mul_lo_u32 v28, s21, v38
	v_mul_lo_u32 v72, s20, v39
	v_mad_u64_u32 v[38:39], s[2:3], s20, v38, 0
	v_add3_u32 v28, v39, v72, v28
	v_sub_u32_e32 v39, v71, v28
	v_sub_co_u32_e32 v38, vcc, v77, v38
	s_nop 1
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, vcc
	v_subrev_co_u32_e64 v72, s[2:3], s20, v38
	v_subb_co_u32_e32 v28, vcc, v71, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v73, s[14:15], 0, v39, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v73
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v74, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v72
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v75, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v73
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v38
	v_cndmask_b32_e64 v74, v74, v75, s[14:15]
	v_subrev_co_u32_e64 v75, s[2:3], s20, v72
	s_nop 1
	v_subbrev_co_u32_e64 v39, s[2:3], 0, v39, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v74
	s_nop 1
	v_cndmask_b32_e64 v39, v73, v39, s[2:3]
	v_cndmask_b32_e64 v73, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v71, v71, v73, vcc
	v_cmp_ne_u32_e32 vcc, 0, v71
	s_nop 1
	v_cndmask_b32_e32 v28, v28, v39, vcc
	v_cndmask_b32_e64 v39, v72, v75, s[2:3]
	v_cndmask_b32_e32 v38, v38, v39, vcc
	v_xor_b32_e32 v38, v38, v70
	v_xor_b32_e32 v28, v28, v70
	v_sub_co_u32_e32 v38, vcc, v38, v70
	v_mul_lo_u32 v39, v76, s33
	s_nop 0
	v_subb_co_u32_e32 v28, vcc, v28, v70, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v78, v28, v38, v39
	v_sub_co_u32_e32 v28, vcc, -7, v30
	v_lshl_add_u64 v[38:39], v[30:31], 0, 6
	s_nop 0
	v_subb_co_u32_e32 v70, vcc, -1, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v39
	s_nop 1
	v_cndmask_b32_e32 v71, v39, v70, vcc
	v_ashrrev_i32_e32 v72, 31, v71
	v_cndmask_b32_e32 v70, v38, v28, vcc
	v_mov_b32_e32 v73, v72
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[72:73]
	v_xor_b32_e32 v79, v70, v72
	v_xor_b32_e32 v73, v71, v72
	v_mad_u64_u32 v[70:71], s[2:3], v79, s16, 0
	v_mul_hi_u32 v28, v79, s1
	v_lshl_add_u64 v[70:71], v[28:29], 0, v[70:71]
	v_mad_u64_u32 v[76:77], s[2:3], v73, s1, 0
	v_add_co_u32_e32 v28, vcc, v70, v76
	v_mad_u64_u32 v[74:75], s[2:3], v73, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v71, v77, vcc
	v_mov_b32_e32 v71, s0
	s_nop 0
	v_addc_co_u32_e32 v75, vcc, 0, v75, vcc
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[74:75]
	v_mul_lo_u32 v28, s21, v70
	v_mul_lo_u32 v71, s20, v71
	v_mad_u64_u32 v[74:75], s[2:3], s20, v70, 0
	v_add3_u32 v28, v75, v71, v28
	v_sub_u32_e32 v71, v73, v28
	v_sub_co_u32_e32 v74, vcc, v79, v74
	s_nop 1
	v_subb_co_u32_e64 v71, s[2:3], v71, v25, vcc
	v_subrev_co_u32_e64 v75, s[2:3], s20, v74
	v_subb_co_u32_e32 v28, vcc, v73, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v71, s[2:3], 0, v71, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v71
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v76, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v75
	v_cndmask_b32_e64 v73, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v74
	v_cndmask_b32_e64 v75, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v71
	v_cndmask_b32_e64 v74, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v71, v76, v75, s[2:3]
	v_add_u32_e32 v75, 2, v70
	v_add_u32_e32 v76, 1, v70
	v_cmp_ne_u32_e64 s[2:3], 0, v71
	v_cndmask_b32_e32 v28, v73, v74, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v71, v76, v75, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v28, v70, v71, vcc
	v_xor_b32_e32 v70, s72, v72
	v_xor_b32_e32 v28, v28, v70
	v_sub_u32_e32 v28, v28, v70
	v_ashrrev_i32_e32 v70, 31, v39
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[70:71]
	v_xor_b32_e32 v77, v38, v70
	v_xad_u32 v76, v28, v70, v37
	v_xor_b32_e32 v71, v39, v70
	v_mad_u64_u32 v[38:39], s[2:3], v77, s16, 0
	v_mul_hi_u32 v28, v77, s1
	v_lshl_add_u64 v[38:39], v[28:29], 0, v[38:39]
	v_mad_u64_u32 v[74:75], s[2:3], v71, s1, 0
	v_add_co_u32_e32 v28, vcc, v38, v74
	v_mad_u64_u32 v[72:73], s[2:3], v71, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v75, vcc
	v_mov_b32_e32 v39, s0
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[72:73]
	v_mul_lo_u32 v28, s21, v38
	v_mul_lo_u32 v72, s20, v39
	v_mad_u64_u32 v[38:39], s[2:3], s20, v38, 0
	v_add3_u32 v28, v39, v72, v28
	v_sub_u32_e32 v39, v71, v28
	v_sub_co_u32_e32 v38, vcc, v77, v38
	s_nop 1
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, vcc
	v_subrev_co_u32_e64 v72, s[2:3], s20, v38
	v_subb_co_u32_e32 v28, vcc, v71, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v73, s[14:15], 0, v39, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v73
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v74, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v72
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v75, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v73
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v38
	v_cndmask_b32_e64 v74, v74, v75, s[14:15]
	v_subrev_co_u32_e64 v75, s[2:3], s20, v72
	s_nop 1
	v_subbrev_co_u32_e64 v39, s[2:3], 0, v39, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v74
	s_nop 1
	v_cndmask_b32_e64 v39, v73, v39, s[2:3]
	v_cndmask_b32_e64 v73, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v71, v71, v73, vcc
	v_cmp_ne_u32_e32 vcc, 0, v71
	s_nop 1
	v_cndmask_b32_e32 v28, v28, v39, vcc
	v_cndmask_b32_e64 v39, v72, v75, s[2:3]
	v_cndmask_b32_e32 v38, v38, v39, vcc
	v_xor_b32_e32 v38, v38, v70
	v_xor_b32_e32 v28, v28, v70
	v_sub_co_u32_e32 v38, vcc, v38, v70
	v_mul_lo_u32 v39, v76, s33
	s_nop 0
	v_subb_co_u32_e32 v28, vcc, v28, v70, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v79, v28, v38, v39
	v_sub_co_u32_e32 v28, vcc, -8, v30
	v_lshl_add_u64 v[38:39], v[30:31], 0, 7
	s_nop 0
	v_subb_co_u32_e32 v70, vcc, -1, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v39
	s_nop 1
	v_cndmask_b32_e32 v71, v39, v70, vcc
	v_ashrrev_i32_e32 v72, 31, v71
	v_cndmask_b32_e32 v70, v38, v28, vcc
	v_mov_b32_e32 v73, v72
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[72:73]
	v_xor_b32_e32 v82, v70, v72
	v_xor_b32_e32 v73, v71, v72
	v_mad_u64_u32 v[70:71], s[2:3], v82, s16, 0
	v_mul_hi_u32 v28, v82, s1
	v_lshl_add_u64 v[70:71], v[28:29], 0, v[70:71]
	v_mad_u64_u32 v[76:77], s[2:3], v73, s1, 0
	v_add_co_u32_e32 v28, vcc, v70, v76
	v_mad_u64_u32 v[74:75], s[2:3], v73, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v71, v77, vcc
	v_mov_b32_e32 v71, s0
	s_nop 0
	v_addc_co_u32_e32 v75, vcc, 0, v75, vcc
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[74:75]
	v_mul_lo_u32 v28, s21, v70
	v_mul_lo_u32 v71, s20, v71
	v_mad_u64_u32 v[74:75], s[2:3], s20, v70, 0
	v_add3_u32 v28, v75, v71, v28
	v_sub_u32_e32 v71, v73, v28
	v_sub_co_u32_e32 v74, vcc, v82, v74
	s_nop 1
	v_subb_co_u32_e64 v71, s[2:3], v71, v25, vcc
	v_subrev_co_u32_e64 v75, s[2:3], s20, v74
	v_subb_co_u32_e32 v28, vcc, v73, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v71, s[2:3], 0, v71, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v71
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v76, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v75
	v_cndmask_b32_e64 v73, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v74
	v_cndmask_b32_e64 v75, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v71
	v_cndmask_b32_e64 v74, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v71, v76, v75, s[2:3]
	v_add_u32_e32 v75, 2, v70
	v_add_u32_e32 v76, 1, v70
	v_cmp_ne_u32_e64 s[2:3], 0, v71
	v_cndmask_b32_e32 v28, v73, v74, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v71, v76, v75, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v28, v70, v71, vcc
	v_xor_b32_e32 v70, s72, v72
	v_xor_b32_e32 v28, v28, v70
	v_sub_u32_e32 v28, v28, v70
	v_ashrrev_i32_e32 v70, 31, v39
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[70:71]
	v_xor_b32_e32 v77, v38, v70
	v_xad_u32 v76, v28, v70, v37
	v_xor_b32_e32 v71, v39, v70
	v_mad_u64_u32 v[38:39], s[2:3], v77, s16, 0
	v_mul_hi_u32 v28, v77, s1
	v_lshl_add_u64 v[38:39], v[28:29], 0, v[38:39]
	v_mad_u64_u32 v[74:75], s[2:3], v71, s1, 0
	v_add_co_u32_e32 v28, vcc, v38, v74
	v_mad_u64_u32 v[72:73], s[2:3], v71, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v75, vcc
	v_mov_b32_e32 v39, s0
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[72:73]
	v_mul_lo_u32 v28, s21, v38
	v_mul_lo_u32 v72, s20, v39
	v_mad_u64_u32 v[38:39], s[2:3], s20, v38, 0
	v_add3_u32 v28, v39, v72, v28
	v_sub_u32_e32 v39, v71, v28
	v_sub_co_u32_e32 v38, vcc, v77, v38
	s_nop 1
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, vcc
	v_subrev_co_u32_e64 v72, s[2:3], s20, v38
	v_subb_co_u32_e32 v28, vcc, v71, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v73, s[14:15], 0, v39, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v73
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v74, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v72
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v75, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v73
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v38
	v_cndmask_b32_e64 v74, v74, v75, s[14:15]
	v_subrev_co_u32_e64 v75, s[2:3], s20, v72
	s_nop 1
	v_subbrev_co_u32_e64 v39, s[2:3], 0, v39, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v74
	s_nop 1
	v_cndmask_b32_e64 v39, v73, v39, s[2:3]
	v_cndmask_b32_e64 v73, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v71, v71, v73, vcc
	v_cmp_ne_u32_e32 vcc, 0, v71
	s_nop 1
	v_cndmask_b32_e32 v28, v28, v39, vcc
	v_cndmask_b32_e64 v39, v72, v75, s[2:3]
	v_cndmask_b32_e32 v38, v38, v39, vcc
	v_xor_b32_e32 v38, v38, v70
	v_xor_b32_e32 v28, v28, v70
	v_sub_co_u32_e32 v38, vcc, v38, v70
	v_mul_lo_u32 v39, v76, s33
	s_nop 0
	v_subb_co_u32_e32 v28, vcc, v28, v70, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v28, v28, v38, v39
	buffer_load_ubyte v70, v34, s[36:39], 0 offen
	buffer_load_ubyte v72, v32, s[36:39], 0 offen
	buffer_load_ubyte v71, v33, s[36:39], 0 offen
	buffer_load_ubyte v73, v35, s[36:39], 0 offen
	buffer_load_ubyte v74, v36, s[36:39], 0 offen
	buffer_load_ubyte v76, v78, s[36:39], 0 offen
	buffer_load_ubyte v75, v79, s[36:39], 0 offen
	buffer_load_ubyte v77, v28, s[36:39], 0 offen
	v_sub_co_u32_e32 v28, vcc, -9, v30
	v_lshl_add_u64 v[32:33], v[30:31], 0, 8
	s_nop 0
	v_subb_co_u32_e32 v34, vcc, -1, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v33
	s_nop 1
	v_cndmask_b32_e32 v35, v33, v34, vcc
	v_ashrrev_i32_e32 v38, 31, v35
	v_cndmask_b32_e32 v34, v32, v28, vcc
	v_mov_b32_e32 v39, v38
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[38:39]
	v_xor_b32_e32 v39, v34, v38
	v_xor_b32_e32 v36, v35, v38
	v_mad_u64_u32 v[34:35], s[2:3], v39, s16, 0
	v_mul_hi_u32 v28, v39, s1
	v_lshl_add_u64 v[34:35], v[28:29], 0, v[34:35]
	v_mad_u64_u32 v[82:83], s[2:3], v36, s1, 0
	v_add_co_u32_e32 v28, vcc, v34, v82
	v_mad_u64_u32 v[78:79], s[2:3], v36, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v34, vcc, v35, v83, vcc
	v_mov_b32_e32 v35, s0
	s_nop 0
	v_addc_co_u32_e32 v79, vcc, 0, v79, vcc
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[78:79]
	v_mul_lo_u32 v28, s21, v34
	v_mul_lo_u32 v35, s20, v35
	v_mad_u64_u32 v[78:79], s[2:3], s20, v34, 0
	v_add3_u32 v28, v79, v35, v28
	v_sub_u32_e32 v35, v36, v28
	v_sub_co_u32_e32 v39, vcc, v39, v78
	s_nop 1
	v_subb_co_u32_e64 v35, s[2:3], v35, v25, vcc
	v_subrev_co_u32_e64 v78, s[2:3], s20, v39
	v_subb_co_u32_e32 v28, vcc, v36, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v35, s[2:3], 0, v35, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v35
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v78
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v39
	v_cndmask_b32_e64 v78, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v35
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v35, v79, v78, s[2:3]
	v_add_u32_e32 v78, 2, v34
	v_add_u32_e32 v79, 1, v34
	v_cmp_ne_u32_e64 s[2:3], 0, v35
	v_cndmask_b32_e32 v28, v36, v39, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v35, v79, v78, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v28, v34, v35, vcc
	v_xor_b32_e32 v34, s72, v38
	v_xor_b32_e32 v28, v28, v34
	v_sub_u32_e32 v28, v28, v34
	v_ashrrev_i32_e32 v34, 31, v33
	v_mov_b32_e32 v35, v34
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[34:35]
	v_xor_b32_e32 v82, v32, v34
	v_xad_u32 v36, v28, v34, v37
	v_xor_b32_e32 v35, v33, v34
	v_mad_u64_u32 v[32:33], s[2:3], v82, s16, 0
	v_mul_hi_u32 v28, v82, s1
	v_lshl_add_u64 v[32:33], v[28:29], 0, v[32:33]
	v_mad_u64_u32 v[78:79], s[2:3], v35, s1, 0
	v_add_co_u32_e32 v28, vcc, v32, v78
	v_mad_u64_u32 v[38:39], s[2:3], v35, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v33, v79, vcc
	v_mov_b32_e32 v33, s0
	s_nop 0
	v_addc_co_u32_e32 v39, vcc, 0, v39, vcc
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[38:39]
	v_mul_lo_u32 v28, s21, v32
	v_mul_lo_u32 v38, s20, v33
	v_mad_u64_u32 v[32:33], s[2:3], s20, v32, 0
	v_add3_u32 v28, v33, v38, v28
	v_sub_u32_e32 v33, v35, v28
	v_sub_co_u32_e32 v32, vcc, v82, v32
	s_nop 1
	v_subb_co_u32_e64 v33, s[2:3], v33, v25, vcc
	v_subrev_co_u32_e64 v38, s[2:3], s20, v32
	v_subb_co_u32_e32 v28, vcc, v35, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v39, s[14:15], 0, v33, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v39
	v_subb_co_u32_e64 v33, s[2:3], v33, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v78, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v38
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v39
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v32
	v_cndmask_b32_e64 v78, v78, v79, s[14:15]
	v_subrev_co_u32_e64 v79, s[2:3], s20, v38
	s_nop 1
	v_subbrev_co_u32_e64 v33, s[2:3], 0, v33, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v78
	s_nop 1
	v_cndmask_b32_e64 v33, v39, v33, s[2:3]
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v35, v35, v39, vcc
	v_cmp_ne_u32_e32 vcc, 0, v35
	s_nop 1
	v_cndmask_b32_e32 v28, v28, v33, vcc
	v_cndmask_b32_e64 v33, v38, v79, s[2:3]
	v_cndmask_b32_e32 v32, v32, v33, vcc
	v_xor_b32_e32 v32, v32, v34
	v_xor_b32_e32 v28, v28, v34
	v_sub_co_u32_e32 v32, vcc, v32, v34
	v_mul_lo_u32 v33, v36, s33
	s_nop 0
	v_subb_co_u32_e32 v28, vcc, v28, v34, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v32, v28, v32, v33
	v_sub_co_u32_e32 v28, vcc, -10, v30
	v_lshl_add_u64 v[34:35], v[30:31], 0, 9
	s_nop 0
	v_subb_co_u32_e32 v33, vcc, -1, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v35
	s_nop 1
	v_cndmask_b32_e32 v39, v35, v33, vcc
	v_ashrrev_i32_e32 v78, 31, v39
	v_cndmask_b32_e32 v38, v34, v28, vcc
	v_mov_b32_e32 v79, v78
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[78:79]
	v_xor_b32_e32 v36, v38, v78
	v_xor_b32_e32 v33, v39, v78
	v_mad_u64_u32 v[38:39], s[2:3], v36, s16, 0
	v_mul_hi_u32 v28, v36, s1
	v_lshl_add_u64 v[38:39], v[28:29], 0, v[38:39]
	v_mad_u64_u32 v[84:85], s[2:3], v33, s1, 0
	v_add_co_u32_e32 v28, vcc, v38, v84
	v_mad_u64_u32 v[82:83], s[2:3], v33, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v85, vcc
	v_mov_b32_e32 v39, s0
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[82:83]
	v_mul_lo_u32 v28, s21, v38
	v_mul_lo_u32 v39, s20, v39
	v_mad_u64_u32 v[82:83], s[2:3], s20, v38, 0
	v_add3_u32 v28, v83, v39, v28
	v_sub_u32_e32 v39, v33, v28
	v_sub_co_u32_e32 v36, vcc, v36, v82
	s_nop 1
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, vcc
	v_subrev_co_u32_e64 v79, s[2:3], s20, v36
	v_subb_co_u32_e32 v28, vcc, v33, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v39, s[2:3], 0, v39, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v39
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v82, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v79
	v_cndmask_b32_e64 v33, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v36
	v_cndmask_b32_e64 v79, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v39
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v39, v82, v79, s[2:3]
	v_add_u32_e32 v79, 2, v38
	v_add_u32_e32 v82, 1, v38
	v_cmp_ne_u32_e64 s[2:3], 0, v39
	v_cndmask_b32_e32 v28, v33, v36, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v39, v82, v79, s[2:3]
	v_xor_b32_e32 v33, s72, v78
	v_cndmask_b32_e32 v28, v38, v39, vcc
	v_ashrrev_i32_e32 v38, 31, v35
	v_mov_b32_e32 v39, v38
	v_xor_b32_e32 v28, v28, v33
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[38:39]
	v_sub_u32_e32 v28, v28, v33
	v_xor_b32_e32 v39, v34, v38
	v_xad_u32 v33, v28, v38, v37
	v_xor_b32_e32 v36, v35, v38
	v_mad_u64_u32 v[34:35], s[2:3], v39, s16, 0
	v_mul_hi_u32 v28, v39, s1
	v_lshl_add_u64 v[34:35], v[28:29], 0, v[34:35]
	v_mad_u64_u32 v[82:83], s[2:3], v36, s1, 0
	v_add_co_u32_e32 v28, vcc, v34, v82
	v_mad_u64_u32 v[78:79], s[2:3], v36, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v34, vcc, v35, v83, vcc
	v_mov_b32_e32 v35, s0
	s_nop 0
	v_addc_co_u32_e32 v79, vcc, 0, v79, vcc
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[78:79]
	v_mul_lo_u32 v28, s21, v34
	v_mul_lo_u32 v78, s20, v35
	v_mad_u64_u32 v[34:35], s[2:3], s20, v34, 0
	v_add3_u32 v28, v35, v78, v28
	v_sub_u32_e32 v35, v36, v28
	v_sub_co_u32_e32 v34, vcc, v39, v34
	v_mul_lo_u32 v33, v33, s33
	s_nop 0
	v_subb_co_u32_e64 v35, s[2:3], v35, v25, vcc
	v_subrev_co_u32_e64 v39, s[2:3], s20, v34
	v_subb_co_u32_e32 v28, vcc, v36, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v78, s[14:15], 0, v35, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v78
	v_subb_co_u32_e64 v35, s[2:3], v35, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v39
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v82, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v78
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v34
	v_cndmask_b32_e64 v79, v79, v82, s[14:15]
	v_subrev_co_u32_e64 v82, s[2:3], s20, v39
	s_nop 1
	v_subbrev_co_u32_e64 v35, s[2:3], 0, v35, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v79
	s_nop 1
	v_cndmask_b32_e64 v35, v78, v35, s[2:3]
	v_cndmask_b32_e64 v78, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v36, v36, v78, vcc
	v_cmp_ne_u32_e32 vcc, 0, v36
	s_nop 1
	v_cndmask_b32_e32 v28, v28, v35, vcc
	v_cndmask_b32_e64 v35, v39, v82, s[2:3]
	v_cndmask_b32_e32 v34, v34, v35, vcc
	v_xor_b32_e32 v34, v34, v38
	v_xor_b32_e32 v28, v28, v38
	v_sub_co_u32_e32 v34, vcc, v34, v38
	s_nop 1
	v_subb_co_u32_e32 v28, vcc, v28, v38, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v33, v28, v34, v33
	v_sub_co_u32_e32 v28, vcc, -11, v30
	v_lshl_add_u64 v[34:35], v[30:31], 0, 10
	s_nop 0
	v_subb_co_u32_e32 v36, vcc, -1, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v35
	s_nop 1
	v_cndmask_b32_e32 v39, v35, v36, vcc
	v_ashrrev_i32_e32 v78, 31, v39
	v_cndmask_b32_e32 v38, v34, v28, vcc
	v_mov_b32_e32 v79, v78
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[78:79]
	v_xor_b32_e32 v79, v38, v78
	v_xor_b32_e32 v36, v39, v78
	v_mad_u64_u32 v[38:39], s[2:3], v79, s16, 0
	v_mul_hi_u32 v28, v79, s1
	v_lshl_add_u64 v[38:39], v[28:29], 0, v[38:39]
	v_mad_u64_u32 v[84:85], s[2:3], v36, s1, 0
	v_add_co_u32_e32 v28, vcc, v38, v84
	v_mad_u64_u32 v[82:83], s[2:3], v36, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v85, vcc
	v_mov_b32_e32 v39, s0
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[82:83]
	v_mul_lo_u32 v28, s21, v38
	v_mul_lo_u32 v39, s20, v39
	v_mad_u64_u32 v[82:83], s[2:3], s20, v38, 0
	v_add3_u32 v28, v83, v39, v28
	v_sub_u32_e32 v39, v36, v28
	v_sub_co_u32_e32 v79, vcc, v79, v82
	s_nop 1
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, vcc
	v_subrev_co_u32_e64 v82, s[2:3], s20, v79
	v_subb_co_u32_e32 v28, vcc, v36, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v39, s[2:3], 0, v39, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v39
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v82
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v79
	v_cndmask_b32_e64 v82, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v39
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v39, v83, v82, s[2:3]
	v_add_u32_e32 v82, 2, v38
	v_add_u32_e32 v83, 1, v38
	v_cmp_ne_u32_e64 s[2:3], 0, v39
	v_cndmask_b32_e32 v28, v36, v79, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v39, v83, v82, s[2:3]
	v_xor_b32_e32 v36, s72, v78
	v_cndmask_b32_e32 v28, v38, v39, vcc
	v_ashrrev_i32_e32 v38, 31, v35
	v_mov_b32_e32 v39, v38
	v_xor_b32_e32 v28, v28, v36
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[38:39]
	v_sub_u32_e32 v28, v28, v36
	v_xor_b32_e32 v84, v34, v38
	v_xad_u32 v36, v28, v38, v37
	v_xor_b32_e32 v39, v35, v38
	v_mad_u64_u32 v[34:35], s[2:3], v84, s16, 0
	v_mul_hi_u32 v28, v84, s1
	v_lshl_add_u64 v[34:35], v[28:29], 0, v[34:35]
	v_mad_u64_u32 v[82:83], s[2:3], v39, s1, 0
	v_add_co_u32_e32 v28, vcc, v34, v82
	v_mad_u64_u32 v[78:79], s[2:3], v39, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v34, vcc, v35, v83, vcc
	v_mov_b32_e32 v35, s0
	s_nop 0
	v_addc_co_u32_e32 v79, vcc, 0, v79, vcc
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[78:79]
	v_mul_lo_u32 v28, s21, v34
	v_mul_lo_u32 v78, s20, v35
	v_mad_u64_u32 v[34:35], s[2:3], s20, v34, 0
	v_add3_u32 v28, v35, v78, v28
	v_sub_u32_e32 v35, v39, v28
	v_sub_co_u32_e32 v34, vcc, v84, v34
	s_nop 1
	v_subb_co_u32_e64 v35, s[2:3], v35, v25, vcc
	v_subrev_co_u32_e64 v78, s[2:3], s20, v34
	v_subb_co_u32_e32 v28, vcc, v39, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v79, s[14:15], 0, v35, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v79
	v_subb_co_u32_e64 v35, s[2:3], v35, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v82, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v78
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v79
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v34
	v_cndmask_b32_e64 v82, v82, v83, s[14:15]
	v_subrev_co_u32_e64 v83, s[2:3], s20, v78
	s_nop 1
	v_subbrev_co_u32_e64 v35, s[2:3], 0, v35, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v82
	s_nop 1
	v_cndmask_b32_e64 v35, v79, v35, s[2:3]
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v39, v39, v79, vcc
	v_cmp_ne_u32_e32 vcc, 0, v39
	s_nop 1
	v_cndmask_b32_e32 v28, v28, v35, vcc
	v_cndmask_b32_e64 v35, v78, v83, s[2:3]
	v_cndmask_b32_e32 v34, v34, v35, vcc
	v_xor_b32_e32 v34, v34, v38
	v_xor_b32_e32 v28, v28, v38
	v_sub_co_u32_e32 v34, vcc, v34, v38
	v_mul_lo_u32 v35, v36, s33
	s_nop 0
	v_subb_co_u32_e32 v28, vcc, v28, v38, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v34, v28, v34, v35
	v_sub_co_u32_e32 v28, vcc, -12, v30
	v_lshl_add_u64 v[38:39], v[30:31], 0, 11
	s_nop 0
	v_subb_co_u32_e32 v35, vcc, -1, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v39
	s_nop 1
	v_cndmask_b32_e32 v79, v39, v35, vcc
	v_ashrrev_i32_e32 v82, 31, v79
	v_cndmask_b32_e32 v78, v38, v28, vcc
	v_mov_b32_e32 v83, v82
	v_lshl_add_u64 v[78:79], v[78:79], 0, v[82:83]
	v_xor_b32_e32 v36, v78, v82
	v_xor_b32_e32 v35, v79, v82
	v_mad_u64_u32 v[78:79], s[2:3], v36, s16, 0
	v_mul_hi_u32 v28, v36, s1
	v_lshl_add_u64 v[78:79], v[28:29], 0, v[78:79]
	v_mad_u64_u32 v[86:87], s[2:3], v35, s1, 0
	v_add_co_u32_e32 v28, vcc, v78, v86
	v_mad_u64_u32 v[84:85], s[2:3], v35, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v78, vcc, v79, v87, vcc
	v_mov_b32_e32 v79, s0
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[78:79], v[78:79], 0, v[84:85]
	v_mul_lo_u32 v28, s21, v78
	v_mul_lo_u32 v79, s20, v79
	v_mad_u64_u32 v[84:85], s[2:3], s20, v78, 0
	v_add3_u32 v28, v85, v79, v28
	v_sub_u32_e32 v79, v35, v28
	v_sub_co_u32_e32 v36, vcc, v36, v84
	s_nop 1
	v_subb_co_u32_e64 v79, s[2:3], v79, v25, vcc
	v_subrev_co_u32_e64 v83, s[2:3], s20, v36
	v_subb_co_u32_e32 v28, vcc, v35, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v79, s[2:3], 0, v79, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v79
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v84, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v83
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v36
	v_cndmask_b32_e64 v83, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v79
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v79, v84, v83, s[2:3]
	v_add_u32_e32 v83, 2, v78
	v_add_u32_e32 v84, 1, v78
	v_cmp_ne_u32_e64 s[2:3], 0, v79
	v_cndmask_b32_e32 v28, v35, v36, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v79, v84, v83, s[2:3]
	v_xor_b32_e32 v35, s72, v82
	v_cndmask_b32_e32 v28, v78, v79, vcc
	v_ashrrev_i32_e32 v78, 31, v39
	v_mov_b32_e32 v79, v78
	v_xor_b32_e32 v28, v28, v35
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[78:79]
	v_sub_u32_e32 v28, v28, v35
	v_xor_b32_e32 v79, v38, v78
	v_xad_u32 v35, v28, v78, v37
	v_xor_b32_e32 v36, v39, v78
	v_mad_u64_u32 v[38:39], s[2:3], v79, s16, 0
	v_mul_hi_u32 v28, v79, s1
	v_lshl_add_u64 v[38:39], v[28:29], 0, v[38:39]
	v_mad_u64_u32 v[84:85], s[2:3], v36, s1, 0
	v_add_co_u32_e32 v28, vcc, v38, v84
	v_mad_u64_u32 v[82:83], s[2:3], v36, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v85, vcc
	v_mov_b32_e32 v39, s0
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[82:83]
	v_mul_lo_u32 v28, s21, v38
	v_mul_lo_u32 v82, s20, v39
	v_mad_u64_u32 v[38:39], s[2:3], s20, v38, 0
	v_add3_u32 v28, v39, v82, v28
	v_sub_u32_e32 v39, v36, v28
	v_sub_co_u32_e32 v38, vcc, v79, v38
	v_mul_lo_u32 v35, v35, s33
	s_nop 0
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, vcc
	v_subrev_co_u32_e64 v79, s[2:3], s20, v38
	v_subb_co_u32_e32 v28, vcc, v36, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v82, s[14:15], 0, v39, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v82
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v79
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v84, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v82
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v38
	v_cndmask_b32_e64 v83, v83, v84, s[14:15]
	v_subrev_co_u32_e64 v84, s[2:3], s20, v79
	s_nop 1
	v_subbrev_co_u32_e64 v39, s[2:3], 0, v39, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v83
	s_nop 1
	v_cndmask_b32_e64 v39, v82, v39, s[2:3]
	v_cndmask_b32_e64 v82, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v36, v36, v82, vcc
	v_cmp_ne_u32_e32 vcc, 0, v36
	v_cndmask_b32_e64 v36, v79, v84, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v36, v38, v36, vcc
	v_cndmask_b32_e32 v28, v28, v39, vcc
	v_xor_b32_e32 v36, v36, v78
	v_xor_b32_e32 v28, v28, v78
	v_sub_co_u32_e32 v36, vcc, v36, v78
	v_lshl_add_u64 v[38:39], v[30:31], 0, 12
	s_nop 0
	v_subb_co_u32_e32 v28, vcc, v28, v78, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v35, v28, v36, v35
	v_sub_co_u32_e32 v28, vcc, -13, v30
	s_nop 1
	v_subb_co_u32_e32 v36, vcc, -1, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v39
	s_nop 1
	v_cndmask_b32_e32 v79, v39, v36, vcc
	v_ashrrev_i32_e32 v82, 31, v79
	v_cndmask_b32_e32 v78, v38, v28, vcc
	v_mov_b32_e32 v83, v82
	v_lshl_add_u64 v[78:79], v[78:79], 0, v[82:83]
	v_xor_b32_e32 v83, v78, v82
	v_xor_b32_e32 v36, v79, v82
	v_mad_u64_u32 v[78:79], s[2:3], v83, s16, 0
	v_mul_hi_u32 v28, v83, s1
	v_lshl_add_u64 v[78:79], v[28:29], 0, v[78:79]
	v_mad_u64_u32 v[86:87], s[2:3], v36, s1, 0
	v_add_co_u32_e32 v28, vcc, v78, v86
	v_mad_u64_u32 v[84:85], s[2:3], v36, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v78, vcc, v79, v87, vcc
	v_mov_b32_e32 v79, s0
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[78:79], v[78:79], 0, v[84:85]
	v_mul_lo_u32 v28, s21, v78
	v_mul_lo_u32 v79, s20, v79
	v_mad_u64_u32 v[84:85], s[2:3], s20, v78, 0
	v_add3_u32 v28, v85, v79, v28
	v_sub_u32_e32 v79, v36, v28
	v_sub_co_u32_e32 v83, vcc, v83, v84
	s_nop 1
	v_subb_co_u32_e64 v79, s[2:3], v79, v25, vcc
	v_subrev_co_u32_e64 v84, s[2:3], s20, v83
	v_subb_co_u32_e32 v28, vcc, v36, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v79, s[2:3], 0, v79, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v79
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v84
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v83
	v_cndmask_b32_e64 v84, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v79
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v79, v85, v84, s[2:3]
	v_add_u32_e32 v84, 2, v78
	v_add_u32_e32 v85, 1, v78
	v_cmp_ne_u32_e64 s[2:3], 0, v79
	v_cndmask_b32_e32 v28, v36, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v79, v85, v84, s[2:3]
	v_xor_b32_e32 v36, s72, v82
	v_cndmask_b32_e32 v28, v78, v79, vcc
	v_ashrrev_i32_e32 v78, 31, v39
	v_mov_b32_e32 v79, v78
	v_xor_b32_e32 v28, v28, v36
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[78:79]
	v_sub_u32_e32 v28, v28, v36
	v_xor_b32_e32 v86, v38, v78
	v_xad_u32 v36, v28, v78, v37
	v_xor_b32_e32 v79, v39, v78
	v_mad_u64_u32 v[38:39], s[2:3], v86, s16, 0
	v_mul_hi_u32 v28, v86, s1
	v_lshl_add_u64 v[38:39], v[28:29], 0, v[38:39]
	v_mad_u64_u32 v[84:85], s[2:3], v79, s1, 0
	v_add_co_u32_e32 v28, vcc, v38, v84
	v_mad_u64_u32 v[82:83], s[2:3], v79, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v85, vcc
	v_mov_b32_e32 v39, s0
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[82:83]
	v_mul_lo_u32 v28, s21, v38
	v_mul_lo_u32 v82, s20, v39
	v_mad_u64_u32 v[38:39], s[2:3], s20, v38, 0
	v_add3_u32 v28, v39, v82, v28
	v_sub_u32_e32 v39, v79, v28
	v_sub_co_u32_e32 v38, vcc, v86, v38
	v_mul_lo_u32 v36, v36, s33
	s_nop 0
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, vcc
	v_subrev_co_u32_e64 v82, s[2:3], s20, v38
	v_subb_co_u32_e32 v28, vcc, v79, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v83, s[14:15], 0, v39, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v83
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v84, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v82
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v83
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v38
	v_cndmask_b32_e64 v84, v84, v85, s[14:15]
	v_subrev_co_u32_e64 v85, s[2:3], s20, v82
	s_nop 1
	v_subbrev_co_u32_e64 v39, s[2:3], 0, v39, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v84
	s_nop 1
	v_cndmask_b32_e64 v39, v83, v39, s[2:3]
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v79, v79, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v79
	s_nop 1
	v_cndmask_b32_e32 v28, v28, v39, vcc
	v_cndmask_b32_e64 v39, v82, v85, s[2:3]
	v_cndmask_b32_e32 v38, v38, v39, vcc
	v_xor_b32_e32 v38, v38, v78
	v_xor_b32_e32 v28, v28, v78
	v_sub_co_u32_e32 v38, vcc, v38, v78
	s_nop 1
	v_subb_co_u32_e32 v28, vcc, v28, v78, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v93, v28, v38, v36
	v_sub_co_u32_e32 v28, vcc, -14, v30
	v_lshl_add_u64 v[38:39], v[30:31], 0, 13
	s_nop 0
	v_subb_co_u32_e32 v36, vcc, -1, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v39
	s_nop 1
	v_cndmask_b32_e32 v79, v39, v36, vcc
	v_ashrrev_i32_e32 v82, 31, v79
	v_cndmask_b32_e32 v78, v38, v28, vcc
	v_mov_b32_e32 v83, v82
	v_lshl_add_u64 v[78:79], v[78:79], 0, v[82:83]
	v_xor_b32_e32 v83, v78, v82
	v_xor_b32_e32 v36, v79, v82
	v_mad_u64_u32 v[78:79], s[2:3], v83, s16, 0
	v_mul_hi_u32 v28, v83, s1
	v_lshl_add_u64 v[78:79], v[28:29], 0, v[78:79]
	v_mad_u64_u32 v[86:87], s[2:3], v36, s1, 0
	v_add_co_u32_e32 v28, vcc, v78, v86
	v_mad_u64_u32 v[84:85], s[2:3], v36, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v78, vcc, v79, v87, vcc
	v_mov_b32_e32 v79, s0
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[78:79], v[78:79], 0, v[84:85]
	v_mul_lo_u32 v28, s21, v78
	v_mul_lo_u32 v79, s20, v79
	v_mad_u64_u32 v[84:85], s[2:3], s20, v78, 0
	v_add3_u32 v28, v85, v79, v28
	v_sub_u32_e32 v79, v36, v28
	v_sub_co_u32_e32 v83, vcc, v83, v84
	s_nop 1
	v_subb_co_u32_e64 v79, s[2:3], v79, v25, vcc
	v_subrev_co_u32_e64 v84, s[2:3], s20, v83
	v_subb_co_u32_e32 v28, vcc, v36, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v79, s[2:3], 0, v79, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v79
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v84
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v83
	v_cndmask_b32_e64 v84, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v79
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v79, v85, v84, s[2:3]
	v_add_u32_e32 v84, 2, v78
	v_add_u32_e32 v85, 1, v78
	v_cmp_ne_u32_e64 s[2:3], 0, v79
	v_cndmask_b32_e32 v28, v36, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v79, v85, v84, s[2:3]
	v_xor_b32_e32 v36, s72, v82
	v_cndmask_b32_e32 v28, v78, v79, vcc
	v_ashrrev_i32_e32 v78, 31, v39
	v_mov_b32_e32 v79, v78
	v_xor_b32_e32 v28, v28, v36
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[78:79]
	v_sub_u32_e32 v28, v28, v36
	v_xor_b32_e32 v86, v38, v78
	v_xad_u32 v36, v28, v78, v37
	v_xor_b32_e32 v79, v39, v78
	v_mad_u64_u32 v[38:39], s[2:3], v86, s16, 0
	v_mul_hi_u32 v28, v86, s1
	v_lshl_add_u64 v[38:39], v[28:29], 0, v[38:39]
	v_mad_u64_u32 v[84:85], s[2:3], v79, s1, 0
	v_add_co_u32_e32 v28, vcc, v38, v84
	v_mad_u64_u32 v[82:83], s[2:3], v79, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v85, vcc
	v_mov_b32_e32 v39, s0
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[82:83]
	v_mul_lo_u32 v28, s21, v38
	v_mul_lo_u32 v82, s20, v39
	v_mad_u64_u32 v[38:39], s[2:3], s20, v38, 0
	v_add3_u32 v28, v39, v82, v28
	v_sub_u32_e32 v39, v79, v28
	v_sub_co_u32_e32 v38, vcc, v86, v38
	v_mul_lo_u32 v36, v36, s33
	s_nop 0
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, vcc
	v_subrev_co_u32_e64 v82, s[2:3], s20, v38
	v_subb_co_u32_e32 v28, vcc, v79, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v83, s[14:15], 0, v39, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v83
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v84, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v82
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v83
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v38
	v_cndmask_b32_e64 v84, v84, v85, s[14:15]
	v_subrev_co_u32_e64 v85, s[2:3], s20, v82
	s_nop 1
	v_subbrev_co_u32_e64 v39, s[2:3], 0, v39, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v84
	s_nop 1
	v_cndmask_b32_e64 v39, v83, v39, s[2:3]
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v79, v79, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v79
	s_nop 1
	v_cndmask_b32_e32 v28, v28, v39, vcc
	v_cndmask_b32_e64 v39, v82, v85, s[2:3]
	v_cndmask_b32_e32 v38, v38, v39, vcc
	v_xor_b32_e32 v38, v38, v78
	v_xor_b32_e32 v28, v28, v78
	v_sub_co_u32_e32 v38, vcc, v38, v78
	s_nop 1
	v_subb_co_u32_e32 v28, vcc, v28, v78, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v94, v28, v38, v36
	v_sub_co_u32_e32 v28, vcc, -15, v30
	v_lshl_add_u64 v[38:39], v[30:31], 0, 14
	s_nop 0
	v_subb_co_u32_e32 v36, vcc, -1, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v39
	s_nop 1
	v_cndmask_b32_e32 v79, v39, v36, vcc
	v_ashrrev_i32_e32 v82, 31, v79
	v_cndmask_b32_e32 v78, v38, v28, vcc
	v_mov_b32_e32 v83, v82
	v_lshl_add_u64 v[78:79], v[78:79], 0, v[82:83]
	v_xor_b32_e32 v83, v78, v82
	v_xor_b32_e32 v36, v79, v82
	v_mad_u64_u32 v[78:79], s[2:3], v83, s16, 0
	v_mul_hi_u32 v28, v83, s1
	v_lshl_add_u64 v[78:79], v[28:29], 0, v[78:79]
	v_mad_u64_u32 v[86:87], s[2:3], v36, s1, 0
	v_add_co_u32_e32 v28, vcc, v78, v86
	v_mad_u64_u32 v[84:85], s[2:3], v36, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v78, vcc, v79, v87, vcc
	v_mov_b32_e32 v79, s0
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[78:79], v[78:79], 0, v[84:85]
	v_mul_lo_u32 v28, s21, v78
	v_mul_lo_u32 v79, s20, v79
	v_mad_u64_u32 v[84:85], s[2:3], s20, v78, 0
	v_add3_u32 v28, v85, v79, v28
	v_sub_u32_e32 v79, v36, v28
	v_sub_co_u32_e32 v83, vcc, v83, v84
	s_nop 1
	v_subb_co_u32_e64 v79, s[2:3], v79, v25, vcc
	v_subrev_co_u32_e64 v84, s[2:3], s20, v83
	v_subb_co_u32_e32 v28, vcc, v36, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v79, s[2:3], 0, v79, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v79
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v84
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v83
	v_cndmask_b32_e64 v84, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v79
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v79, v85, v84, s[2:3]
	v_add_u32_e32 v84, 2, v78
	v_add_u32_e32 v85, 1, v78
	v_cmp_ne_u32_e64 s[2:3], 0, v79
	v_cndmask_b32_e32 v28, v36, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v79, v85, v84, s[2:3]
	v_xor_b32_e32 v36, s72, v82
	v_cndmask_b32_e32 v28, v78, v79, vcc
	v_ashrrev_i32_e32 v78, 31, v39
	v_mov_b32_e32 v79, v78
	v_xor_b32_e32 v28, v28, v36
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[78:79]
	v_sub_u32_e32 v28, v28, v36
	v_xor_b32_e32 v86, v38, v78
	v_xad_u32 v36, v28, v78, v37
	v_xor_b32_e32 v79, v39, v78
	v_mad_u64_u32 v[38:39], s[2:3], v86, s16, 0
	v_mul_hi_u32 v28, v86, s1
	v_lshl_add_u64 v[38:39], v[28:29], 0, v[38:39]
	v_mad_u64_u32 v[84:85], s[2:3], v79, s1, 0
	v_add_co_u32_e32 v28, vcc, v38, v84
	v_mad_u64_u32 v[82:83], s[2:3], v79, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v85, vcc
	v_mov_b32_e32 v39, s0
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[82:83]
	v_mul_lo_u32 v28, s21, v38
	v_mul_lo_u32 v82, s20, v39
	v_mad_u64_u32 v[38:39], s[2:3], s20, v38, 0
	v_add3_u32 v28, v39, v82, v28
	v_sub_u32_e32 v39, v79, v28
	v_sub_co_u32_e32 v38, vcc, v86, v38
	v_mul_lo_u32 v36, v36, s33
	s_nop 0
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, vcc
	v_subrev_co_u32_e64 v82, s[2:3], s20, v38
	v_subb_co_u32_e32 v28, vcc, v79, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v83, s[14:15], 0, v39, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v83
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v84, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v82
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v83
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v38
	v_cndmask_b32_e64 v84, v84, v85, s[14:15]
	v_subrev_co_u32_e64 v85, s[2:3], s20, v82
	s_nop 1
	v_subbrev_co_u32_e64 v39, s[2:3], 0, v39, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v84
	s_nop 1
	v_cndmask_b32_e64 v39, v83, v39, s[2:3]
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v79, v79, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v79
	s_nop 1
	v_cndmask_b32_e32 v28, v28, v39, vcc
	v_cndmask_b32_e64 v39, v82, v85, s[2:3]
	v_cndmask_b32_e32 v38, v38, v39, vcc
	v_xor_b32_e32 v38, v38, v78
	v_xor_b32_e32 v28, v28, v78
	v_sub_co_u32_e32 v38, vcc, v38, v78
	s_nop 1
	v_subb_co_u32_e32 v28, vcc, v28, v78, vcc
	v_ashrrev_i32_e32 v28, 31, v28
	v_and_b32_e32 v28, s44, v28
	v_add3_u32 v87, v28, v38, v36
	v_sub_co_u32_e32 v28, vcc, -16, v30
	v_lshl_add_u64 v[38:39], v[30:31], 0, 15
	s_nop 0
	v_subb_co_u32_e32 v30, vcc, -1, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v39
	s_nop 1
	v_cndmask_b32_e32 v31, v39, v30, vcc
	v_ashrrev_i32_e32 v78, 31, v31
	v_cndmask_b32_e32 v30, v38, v28, vcc
	v_mov_b32_e32 v79, v78
	v_lshl_add_u64 v[30:31], v[30:31], 0, v[78:79]
	v_xor_b32_e32 v79, v30, v78
	v_xor_b32_e32 v36, v31, v78
	v_mad_u64_u32 v[30:31], s[2:3], v79, s16, 0
	v_mul_hi_u32 v28, v79, s1
	v_lshl_add_u64 v[30:31], v[28:29], 0, v[30:31]
	v_mad_u64_u32 v[84:85], s[2:3], v36, s1, 0
	v_add_co_u32_e32 v28, vcc, v30, v84
	v_mad_u64_u32 v[82:83], s[2:3], v36, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v30, vcc, v31, v85, vcc
	v_mov_b32_e32 v31, s0
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[30:31], v[30:31], 0, v[82:83]
	v_mul_lo_u32 v28, s21, v30
	v_mul_lo_u32 v31, s20, v31
	v_mad_u64_u32 v[82:83], s[2:3], s20, v30, 0
	v_add3_u32 v28, v83, v31, v28
	v_sub_u32_e32 v31, v36, v28
	v_sub_co_u32_e32 v79, vcc, v79, v82
	s_nop 1
	v_subb_co_u32_e64 v31, s[2:3], v31, v25, vcc
	v_subrev_co_u32_e64 v82, s[2:3], s20, v79
	v_subb_co_u32_e32 v28, vcc, v36, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v31, s[2:3], 0, v31, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s21, v31
	v_cmp_le_u32_e32 vcc, s21, v28
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s20, v82
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v79
	v_cndmask_b32_e64 v82, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s21, v31
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	v_cndmask_b32_e64 v31, v83, v82, s[2:3]
	v_add_u32_e32 v82, 2, v30
	v_add_u32_e32 v83, 1, v30
	v_cmp_ne_u32_e64 s[2:3], 0, v31
	v_cndmask_b32_e32 v28, v36, v79, vcc
	v_cmp_ne_u32_e32 vcc, 0, v28
	v_cndmask_b32_e64 v31, v83, v82, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v28, v30, v31, vcc
	v_xor_b32_e32 v30, s72, v78
	v_xor_b32_e32 v28, v28, v30
	v_sub_u32_e32 v28, v28, v30
	v_ashrrev_i32_e32 v30, 31, v39
	v_mov_b32_e32 v31, v30
	v_xad_u32 v82, v28, v30, v37
	v_lshl_add_u64 v[36:37], v[38:39], 0, v[30:31]
	v_xor_b32_e32 v83, v36, v30
	v_xor_b32_e32 v31, v37, v30
	v_mad_u64_u32 v[36:37], s[2:3], v83, s16, 0
	v_mul_hi_u32 v28, v83, s1
	v_lshl_add_u64 v[36:37], v[28:29], 0, v[36:37]
	v_mad_u64_u32 v[78:79], s[2:3], v31, s1, 0
	v_add_co_u32_e32 v28, vcc, v36, v78
	v_mad_u64_u32 v[38:39], s[2:3], v31, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v36, vcc, v37, v79, vcc
	v_mov_b32_e32 v37, s0
	s_nop 0
	v_addc_co_u32_e32 v39, vcc, 0, v39, vcc
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[38:39]
	v_mul_lo_u32 v28, s21, v36
	v_mul_lo_u32 v38, s20, v37
	v_mad_u64_u32 v[36:37], s[2:3], s20, v36, 0
	v_add3_u32 v28, v37, v38, v28
	v_sub_u32_e32 v37, v31, v28
	v_sub_co_u32_e32 v36, vcc, v83, v36
	s_movk_i32 s1, 0xff
	s_nop 0
	v_subb_co_u32_e64 v37, s[2:3], v37, v25, vcc
	v_subrev_co_u32_e64 v38, s[2:3], s20, v36
	v_subb_co_u32_e32 v28, vcc, v31, v28, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v39, s[14:15], 0, v37, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v39
	v_subb_co_u32_e64 v25, s[2:3], v37, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v78, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v38
	v_subrev_co_u32_e64 v37, s[2:3], s20, v38
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v39
	v_subbrev_co_u32_e64 v25, s[2:3], 0, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v78, v78, v79, s[14:15]
	v_cmp_le_u32_e32 vcc, s21, v28
	v_cmp_ne_u32_e64 s[2:3], 0, v78
	s_nop 0
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v36
	v_cndmask_b32_e64 v25, v39, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v28
	s_nop 1
	v_cndmask_b32_e32 v31, v31, v39, vcc
	v_cmp_ne_u32_e32 vcc, 0, v31
	s_nop 1
	v_cndmask_b32_e32 v25, v28, v25, vcc
	v_cndmask_b32_e64 v28, v38, v37, s[2:3]
	v_cndmask_b32_e32 v28, v36, v28, vcc
	v_xor_b32_e32 v28, v28, v30
	v_xor_b32_e32 v25, v25, v30
	v_sub_co_u32_e32 v28, vcc, v28, v30
	s_lshl_b64 s[2:3], s[18:19], 9
	s_nop 0
	v_subb_co_u32_e32 v25, vcc, v25, v30, vcc
	v_ashrrev_i32_e32 v25, 31, v25
	v_and_b32_e32 v25, s44, v25
	v_mul_lo_u32 v30, v82, s33
	v_add3_u32 v25, v25, v28, v30
	buffer_load_ubyte v78, v32, s[36:39], 0 offen
	buffer_load_ubyte v82, v33, s[36:39], 0 offen
	buffer_load_ubyte v79, v34, s[36:39], 0 offen
	buffer_load_ubyte v83, v35, s[36:39], 0 offen
	buffer_load_ubyte v84, v93, s[36:39], 0 offen
	buffer_load_ubyte v86, v94, s[36:39], 0 offen
	buffer_load_ubyte v85, v87, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v87, v25, s[36:39], 0 offen
	ds_read_b32 v25, v64 offset:49152
	ds_read_b128 v[114:117], v56
	ds_read_b128 v[110:113], v58
	ds_read_b128 v[106:109], v60
	ds_read_b128 v[102:105], v62
	v_mov_b32_e32 v37, s3
	v_or_b32_e32 v36, s2, v46
	v_lshl_add_u64 v[30:31], v[36:37], 0, v[48:49]
	s_movk_i32 s2, 0xff00
	s_mov_b32 s3, -1
	v_sub_co_u32_e32 v28, vcc, s1, v30
	v_lshl_add_u64 v[32:33], v[30:31], 0, s[2:3]
	s_nop 0
	v_subb_co_u32_e32 v34, vcc, 0, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v33
	s_nop 1
	v_cndmask_b32_e32 v39, v33, v34, vcc
	v_cndmask_b32_e32 v38, v32, v28, vcc
	v_or_b32_e32 v28, s79, v39
	v_cmp_ne_u32_e32 vcc, 0, v28
	s_and_saveexec_b64 s[2:3], vcc
	s_xor_b64 s[14:15], exec, s[2:3]
	s_cbranch_execz .LBB0_91
	s_add_u32 s2, s78, s82
	s_mov_b32 s83, s82
	s_addc_u32 s3, s79, s82
	s_xor_b64 s[18:19], s[2:3], s[82:83]
	v_cvt_f32_u32_e32 v28, s18
	v_cvt_f32_u32_e32 v32, s19
	s_sub_u32 s1, 0, s18
	s_subb_u32 s2, 0, s19
	v_ashrrev_i32_e32 v34, 31, v39
	v_fmac_f32_e32 v28, 0x4f800000, v32
	v_rcp_f32_e32 v28, v28
	v_mov_b32_e32 v35, v34
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[34:35]
	v_xor_b32_e32 v35, v38, v34
	v_mul_f32_e32 v28, 0x5f7ffffc, v28
	v_mul_f32_e32 v32, 0x2f800000, v28
	v_trunc_f32_e32 v32, v32
	v_fmac_f32_e32 v28, 0xcf800000, v32
	v_cvt_u32_f32_e32 v32, v32
	v_cvt_u32_f32_e32 v28, v28
	v_readfirstlane_b32 s3, v32
	v_readfirstlane_b32 s16, v28
	s_mul_i32 s20, s1, s3
	s_mul_hi_u32 s22, s1, s16
	s_mul_i32 s21, s2, s16
	s_add_i32 s20, s22, s20
	s_add_i32 s20, s20, s21
	s_mul_i32 s23, s1, s16
	s_mul_i32 s22, s16, s20
	s_mul_hi_u32 s24, s16, s23
	s_mul_hi_u32 s21, s16, s20
	s_add_u32 s22, s24, s22
	s_addc_u32 s21, 0, s21
	s_mul_hi_u32 s25, s3, s23
	s_mul_i32 s23, s3, s23
	s_add_u32 s22, s22, s23
	s_mul_hi_u32 s24, s3, s20
	s_addc_u32 s21, s21, s25
	s_addc_u32 s22, s24, 0
	s_mul_i32 s20, s3, s20
	s_add_u32 s20, s21, s20
	s_addc_u32 s21, 0, s22
	s_add_u32 s16, s16, s20
	s_addc_u32 s3, s3, s21
	s_mul_i32 s20, s1, s3
	s_mul_hi_u32 s21, s1, s16
	s_add_i32 s20, s21, s20
	s_mul_i32 s2, s2, s16
	s_add_i32 s20, s20, s2
	s_mul_i32 s1, s1, s16
	s_mul_hi_u32 s21, s3, s1
	s_mul_i32 s22, s3, s1
	s_mul_i32 s24, s16, s20
	s_mul_hi_u32 s1, s16, s1
	s_mul_hi_u32 s23, s16, s20
	s_add_u32 s1, s1, s24
	s_addc_u32 s23, 0, s23
	s_add_u32 s1, s1, s22
	s_mul_hi_u32 s2, s3, s20
	s_addc_u32 s1, s23, s21
	s_addc_u32 s2, s2, 0
	s_mul_i32 s20, s3, s20
	s_add_u32 s1, s1, s20
	s_addc_u32 s2, 0, s2
	s_add_u32 s1, s16, s1
	s_addc_u32 s16, s3, s2
	v_xor_b32_e32 v32, v39, v34
	v_mad_u64_u32 v[38:39], s[2:3], v35, s16, 0
	v_mul_hi_u32 v28, v35, s1
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[38:39]
	v_mad_u64_u32 v[48:49], s[2:3], v32, s1, 0
	v_add_co_u32_e32 v28, vcc, v28, v48
	v_mad_u64_u32 v[38:39], s[2:3], v32, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v28, vcc, v29, v49, vcc
	v_mov_b32_e32 v29, s0
	s_nop 0
	v_addc_co_u32_e32 v39, vcc, 0, v39, vcc
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[38:39]
	v_mul_lo_u32 v46, s19, v28
	v_mul_lo_u32 v48, s18, v29
	v_mad_u64_u32 v[38:39], s[0:1], s18, v28, 0
	v_add3_u32 v46, v39, v48, v46
	v_sub_u32_e32 v39, v32, v46
	v_mov_b32_e32 v48, s19
	v_sub_co_u32_e32 v35, vcc, v35, v38
	s_nop 1
	v_subb_co_u32_e64 v38, s[2:3], v39, v48, vcc
	v_subrev_co_u32_e64 v39, s[2:3], s18, v35
	s_nop 1
	v_subbrev_co_u32_e64 v38, s[2:3], 0, v38, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s19, v38
	s_nop 1
	v_cndmask_b32_e64 v48, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s18, v39
	s_nop 1
	v_cndmask_b32_e64 v39, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s19, v38
	s_nop 1
	v_cndmask_b32_e64 v93, v48, v39, s[2:3]
	v_lshl_add_u64 v[38:39], v[28:29], 0, 2
	v_lshl_add_u64 v[48:49], v[28:29], 0, 1
	v_subb_co_u32_e32 v29, vcc, v32, v46, vcc
	v_cmp_le_u32_e32 vcc, s19, v29
	s_nop 1
	v_cndmask_b32_e64 v32, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s18, v35
	s_nop 1
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s19, v29
	s_nop 1
	v_cndmask_b32_e32 v29, v32, v35, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v32, v48, v38, vcc
	v_cmp_ne_u32_e32 vcc, 0, v29
	v_xor_b32_e32 v29, s82, v34
	s_nop 0
	v_cndmask_b32_e32 v28, v28, v32, vcc
	v_xor_b32_e32 v28, v28, v29
	v_sub_co_u32_e32 v34, vcc, v28, v29
.LBB0_91:
	s_or_saveexec_b64 s[2:3], s[14:15]
	v_readlane_b32 s56, v144, 24
	v_readlane_b32 s58, v144, 22
	v_readlane_b32 s60, v144, 20
	v_readlane_b32 s72, v144, 18
	v_readlane_b32 s57, v144, 25
	v_readlane_b32 s59, v144, 23
	v_readlane_b32 s61, v144, 21
	v_readlane_b32 s73, v144, 19
	s_xor_b64 exec, exec, s[2:3]
	s_cbranch_execz .LBB0_93
	v_rcp_iflag_f32_e32 v28, v100
	s_sub_i32 s0, 0, s78
	v_mul_f32_e32 v28, 0x4f7ffffe, v28
	v_cvt_u32_f32_e32 v28, v28
	v_mul_lo_u32 v29, s0, v28
	v_mul_hi_u32 v29, v28, v29
	v_add_u32_e32 v28, v28, v29
	v_mul_hi_u32 v28, v38, v28
	v_mul_lo_u32 v29, v28, s78
	v_sub_u32_e32 v29, v38, v29
	v_add_u32_e32 v32, 1, v28
	v_subrev_u32_e32 v34, s78, v29
	v_cmp_le_u32_e32 vcc, s78, v29
	s_nop 1
	v_cndmask_b32_e32 v29, v29, v34, vcc
	v_cndmask_b32_e32 v28, v28, v32, vcc
	v_add_u32_e32 v32, 1, v28
	v_cmp_le_u32_e32 vcc, s78, v29
	s_nop 1
	v_cndmask_b32_e32 v34, v28, v32, vcc
.LBB0_93:
	s_or_b64 exec, exec, s[2:3]
	s_movk_i32 s0, 0xff00
	v_lshl_add_u64 v[28:29], v[36:37], 0, v[42:43]
	s_mov_b32 s1, -1
	v_lshl_add_u64 v[36:37], v[28:29], 0, s[0:1]
	v_or_b32_e32 v32, s79, v37
	s_mov_b32 s0, 0
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_and_saveexec_b64 s[2:3], vcc
	s_xor_b64 s[18:19], exec, s[2:3]
	s_cbranch_execz .LBB0_95
	s_add_u32 s2, s78, s82
	s_mov_b32 s83, s82
	s_addc_u32 s3, s79, s82
	s_xor_b64 s[20:21], s[2:3], s[82:83]
	v_cvt_f32_u32_e32 v32, s20
	v_cvt_f32_u32_e32 v35, s21
	s_sub_u32 s1, 0, s20
	s_subb_u32 s2, 0, s21
	v_ashrrev_i32_e32 v42, 31, v37
	v_fmamk_f32 v32, v35, 0x4f800000, v32
	v_rcp_f32_e32 v32, v32
	v_mov_b32_e32 v43, v42
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[42:43]
	v_mov_b32_e32 v39, 0
	v_mul_f32_e32 v32, 0x5f7ffffc, v32
	v_mul_f32_e32 v35, 0x2f800000, v32
	v_trunc_f32_e32 v35, v35
	v_fmamk_f32 v32, v35, 0xcf800000, v32
	v_cvt_u32_f32_e32 v35, v35
	v_cvt_u32_f32_e32 v32, v32
	v_readfirstlane_b32 s3, v35
	v_readfirstlane_b32 s14, v32
	s_mul_i32 s15, s1, s3
	s_mul_hi_u32 s22, s1, s14
	s_mul_i32 s16, s2, s14
	s_add_i32 s15, s22, s15
	s_add_i32 s15, s15, s16
	s_mul_i32 s23, s1, s14
	s_mul_i32 s22, s14, s15
	s_mul_hi_u32 s24, s14, s23
	s_mul_hi_u32 s16, s14, s15
	s_add_u32 s22, s24, s22
	s_addc_u32 s16, 0, s16
	s_mul_hi_u32 s25, s3, s23
	s_mul_i32 s23, s3, s23
	s_add_u32 s22, s22, s23
	s_mul_hi_u32 s24, s3, s15
	s_addc_u32 s16, s16, s25
	s_addc_u32 s22, s24, 0
	s_mul_i32 s15, s3, s15
	s_add_u32 s15, s16, s15
	s_addc_u32 s16, 0, s22
	s_add_u32 s14, s14, s15
	s_addc_u32 s3, s3, s16
	s_mul_i32 s15, s1, s3
	s_mul_hi_u32 s16, s1, s14
	s_add_i32 s15, s16, s15
	s_mul_i32 s2, s2, s14
	s_add_i32 s15, s15, s2
	s_mul_i32 s1, s1, s14
	s_mul_hi_u32 s16, s3, s1
	s_mul_i32 s22, s3, s1
	s_mul_i32 s24, s14, s15
	s_mul_hi_u32 s1, s14, s1
	s_mul_hi_u32 s23, s14, s15
	s_add_u32 s1, s1, s24
	s_addc_u32 s23, 0, s23
	s_add_u32 s1, s1, s22
	s_mul_hi_u32 s2, s3, s15
	s_addc_u32 s1, s23, s16
	s_addc_u32 s2, s2, 0
	s_mul_i32 s15, s3, s15
	s_add_u32 s1, s1, s15
	s_addc_u32 s2, 0, s2
	s_add_u32 s1, s14, s1
	s_addc_u32 s14, s3, s2
	v_xor_b32_e32 v35, v36, v42
	v_xor_b32_e32 v32, v37, v42
	v_mad_u64_u32 v[36:37], s[2:3], v35, s14, 0
	v_mul_hi_u32 v38, v35, s1
	v_lshl_add_u64 v[36:37], v[38:39], 0, v[36:37]
	v_mad_u64_u32 v[48:49], s[2:3], v32, s1, 0
	v_add_co_u32_e32 v36, vcc, v36, v48
	v_mad_u64_u32 v[38:39], s[2:3], v32, s14, 0
	s_nop 0
	v_addc_co_u32_e32 v36, vcc, v37, v49, vcc
	v_mov_b32_e32 v37, s0
	s_nop 0
	v_addc_co_u32_e32 v39, vcc, 0, v39, vcc
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[38:39]
	v_mul_lo_u32 v38, s21, v36
	v_mul_lo_u32 v39, s20, v37
	v_mad_u64_u32 v[36:37], s[0:1], s20, v36, 0
	v_add3_u32 v37, v37, v39, v38
	v_sub_u32_e32 v38, v32, v37
	v_mov_b32_e32 v39, s21
	v_sub_co_u32_e32 v35, vcc, v35, v36
	s_nop 1
	v_subb_co_u32_e64 v36, s[2:3], v38, v39, vcc
	v_subrev_co_u32_e64 v38, s[2:3], s20, v35
	v_subb_co_u32_e32 v32, vcc, v32, v37, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[14:15], 0, v36, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v43
	v_subb_co_u32_e64 v36, s[2:3], v36, v39, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v46, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v38
	v_subrev_co_u32_e64 v39, s[2:3], s20, v38
	s_nop 0
	v_cndmask_b32_e64 v48, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v43
	v_subbrev_co_u32_e64 v36, s[2:3], 0, v36, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v46, v46, v48, s[14:15]
	v_cmp_le_u32_e32 vcc, s21, v32
	v_cmp_ne_u32_e64 s[2:3], 0, v46
	s_nop 0
	v_cndmask_b32_e64 v37, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v35
	v_cndmask_b32_e64 v36, v43, v36, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v32
	s_nop 1
	v_cndmask_b32_e32 v37, v37, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v37
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v36, vcc
	v_cndmask_b32_e64 v36, v38, v39, s[2:3]
	v_cndmask_b32_e32 v35, v35, v36, vcc
	v_xor_b32_e32 v35, v35, v42
	v_xor_b32_e32 v32, v32, v42
	v_sub_co_u32_e32 v38, vcc, v35, v42
	s_nop 1
	v_subb_co_u32_e32 v39, vcc, v32, v42, vcc
.LBB0_95:
	s_andn2_saveexec_b64 s[2:3], s[18:19]
	s_cbranch_execz .LBB0_97
	v_rcp_iflag_f32_e32 v32, v100
	s_sub_i32 s0, 0, s78
	v_mov_b32_e32 v39, 0
	v_mul_f32_e32 v32, 0x4f7ffffe, v32
	v_cvt_u32_f32_e32 v32, v32
	v_mul_lo_u32 v35, s0, v32
	v_mul_hi_u32 v35, v32, v35
	v_add_u32_e32 v32, v32, v35
	v_mul_hi_u32 v32, v36, v32
	v_mul_lo_u32 v32, v32, s78
	v_sub_u32_e32 v32, v36, v32
	v_subrev_u32_e32 v35, s78, v32
	v_cmp_le_u32_e32 vcc, s78, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v35, vcc
	v_subrev_u32_e32 v35, s78, v32
	v_cmp_le_u32_e32 vcc, s78, v32
	s_nop 1
	v_cndmask_b32_e32 v38, v32, v35, vcc
.LBB0_97:
	s_or_b64 exec, exec, s[2:3]
	v_ashrrev_i32_e32 v32, 31, v33
	v_xor_b32_e32 v32, v34, v32
	v_ashrrev_i32_e32 v33, 31, v39
	v_mul_lo_u32 v32, v32, s66
	v_and_b32_e32 v33, s78, v33
	v_add3_u32 v32, v32, v38, v33
	buffer_load_ubyte v36, v32, s[8:11], 0 offen
	s_movk_i32 s2, 0xff02
	s_mov_b32 s3, -1
	s_movk_i32 s1, 0xfd
	v_lshl_add_u64 v[32:33], v[30:31], 0, s[2:3]
	v_sub_co_u32_e32 v30, vcc, s1, v30
	s_mov_b32 s0, 0
	s_nop 0
	v_subb_co_u32_e32 v31, vcc, 0, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v33
	s_nop 1
	v_cndmask_b32_e32 v35, v33, v31, vcc
	v_cndmask_b32_e32 v34, v32, v30, vcc
	v_or_b32_e32 v30, s79, v35
	v_cmp_ne_u32_e32 vcc, 0, v30
	s_and_saveexec_b64 s[2:3], vcc
	s_xor_b64 s[14:15], exec, s[2:3]
	s_cbranch_execz .LBB0_99
	s_add_u32 s2, s78, s82
	s_mov_b32 s83, s82
	s_addc_u32 s3, s79, s82
	s_xor_b64 s[18:19], s[2:3], s[82:83]
	v_cvt_f32_u32_e32 v30, s18
	v_cvt_f32_u32_e32 v31, s19
	s_sub_u32 s1, 0, s18
	s_subb_u32 s2, 0, s19
	v_mov_b32_e32 v39, 0
	v_fmamk_f32 v30, v31, 0x4f800000, v30
	v_rcp_f32_e32 v30, v30
	s_nop 0
	v_mul_f32_e32 v30, 0x5f7ffffc, v30
	v_mul_f32_e32 v31, 0x2f800000, v30
	v_trunc_f32_e32 v31, v31
	v_fmamk_f32 v30, v31, 0xcf800000, v30
	v_cvt_u32_f32_e32 v31, v31
	v_cvt_u32_f32_e32 v30, v30
	v_readfirstlane_b32 s3, v31
	v_readfirstlane_b32 s16, v30
	s_mul_i32 s20, s1, s3
	s_mul_hi_u32 s22, s1, s16
	s_mul_i32 s21, s2, s16
	s_add_i32 s20, s22, s20
	s_add_i32 s20, s20, s21
	s_mul_i32 s23, s1, s16
	s_mul_i32 s22, s16, s20
	s_mul_hi_u32 s24, s16, s23
	s_mul_hi_u32 s21, s16, s20
	s_add_u32 s22, s24, s22
	s_addc_u32 s21, 0, s21
	s_mul_hi_u32 s25, s3, s23
	s_mul_i32 s23, s3, s23
	s_add_u32 s22, s22, s23
	s_mul_hi_u32 s24, s3, s20
	s_addc_u32 s21, s21, s25
	s_addc_u32 s22, s24, 0
	s_mul_i32 s20, s3, s20
	s_add_u32 s20, s21, s20
	s_addc_u32 s21, 0, s22
	s_add_u32 s16, s16, s20
	s_addc_u32 s3, s3, s21
	s_mul_i32 s20, s1, s3
	s_mul_hi_u32 s21, s1, s16
	s_add_i32 s20, s21, s20
	s_mul_i32 s2, s2, s16
	s_add_i32 s20, s20, s2
	s_mul_i32 s1, s1, s16
	s_mul_hi_u32 s21, s3, s1
	s_mul_i32 s22, s3, s1
	s_mul_i32 s24, s16, s20
	s_mul_hi_u32 s1, s16, s1
	s_mul_hi_u32 s23, s16, s20
	s_add_u32 s1, s1, s24
	s_addc_u32 s23, 0, s23
	s_add_u32 s1, s1, s22
	s_mul_hi_u32 s2, s3, s20
	s_addc_u32 s1, s23, s21
	s_addc_u32 s2, s2, 0
	s_mul_i32 s20, s3, s20
	s_add_u32 s1, s1, s20
	v_ashrrev_i32_e32 v30, 31, v35
	s_addc_u32 s2, 0, s2
	v_mov_b32_e32 v31, v30
	s_add_u32 s1, s16, s1
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[30:31]
	s_addc_u32 s16, s3, s2
	v_xor_b32_e32 v32, v34, v30
	v_xor_b32_e32 v31, v35, v30
	v_mad_u64_u32 v[34:35], s[2:3], v32, s16, 0
	v_mul_hi_u32 v38, v32, s1
	v_lshl_add_u64 v[34:35], v[38:39], 0, v[34:35]
	v_mad_u64_u32 v[42:43], s[2:3], v31, s1, 0
	v_add_co_u32_e32 v34, vcc, v34, v42
	v_mad_u64_u32 v[38:39], s[2:3], v31, s16, 0
	s_nop 0
	v_addc_co_u32_e32 v34, vcc, v35, v43, vcc
	v_mov_b32_e32 v35, s0
	s_nop 0
	v_addc_co_u32_e32 v39, vcc, 0, v39, vcc
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[38:39]
	v_mul_lo_u32 v37, s19, v34
	v_mul_lo_u32 v42, s18, v35
	v_mad_u64_u32 v[38:39], s[0:1], s18, v34, 0
	v_add3_u32 v37, v39, v42, v37
	v_sub_u32_e32 v39, v31, v37
	v_mov_b32_e32 v42, s19
	v_sub_co_u32_e32 v32, vcc, v32, v38
	v_xor_b32_e32 v30, s82, v30
	s_nop 0
	v_subb_co_u32_e64 v38, s[2:3], v39, v42, vcc
	v_subrev_co_u32_e64 v39, s[2:3], s18, v32
	v_subb_co_u32_e32 v31, vcc, v31, v37, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v38, s[2:3], 0, v38, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s19, v38
	v_cmp_le_u32_e32 vcc, s19, v31
	s_nop 0
	v_cndmask_b32_e64 v42, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s18, v39
	s_nop 1
	v_cndmask_b32_e64 v39, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s19, v38
	s_nop 1
	v_cndmask_b32_e64 v46, v42, v39, s[2:3]
	v_lshl_add_u64 v[38:39], v[34:35], 0, 2
	v_lshl_add_u64 v[42:43], v[34:35], 0, 1
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s18, v32
	s_nop 1
	v_cndmask_b32_e64 v32, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s19, v31
	s_nop 1
	v_cndmask_b32_e32 v31, v35, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v46
	s_nop 1
	v_cndmask_b32_e32 v32, v42, v38, vcc
	v_cmp_ne_u32_e32 vcc, 0, v31
	s_nop 1
	v_cndmask_b32_e32 v31, v34, v32, vcc
	v_xor_b32_e32 v31, v31, v30
	v_sub_co_u32_e32 v30, vcc, v31, v30
.LBB0_99:
	s_andn2_saveexec_b64 s[2:3], s[14:15]
	s_cbranch_execz .LBB0_101
	v_rcp_iflag_f32_e32 v30, v100
	s_sub_i32 s0, 0, s78
	v_mul_f32_e32 v30, 0x4f7ffffe, v30
	v_cvt_u32_f32_e32 v30, v30
	v_mul_lo_u32 v31, s0, v30
	v_mul_hi_u32 v31, v30, v31
	v_add_u32_e32 v30, v30, v31
	v_mul_hi_u32 v30, v34, v30
	v_mul_lo_u32 v31, v30, s78
	v_sub_u32_e32 v31, v34, v31
	v_add_u32_e32 v32, 1, v30
	v_subrev_u32_e32 v34, s78, v31
	v_cmp_le_u32_e32 vcc, s78, v31
	s_nop 1
	v_cndmask_b32_e32 v31, v31, v34, vcc
	v_cndmask_b32_e32 v30, v30, v32, vcc
	v_add_u32_e32 v32, 1, v30
	v_cmp_le_u32_e32 vcc, s78, v31
	s_nop 1
	v_cndmask_b32_e32 v30, v30, v32, vcc
.LBB0_101:
	s_or_b64 exec, exec, s[2:3]
	s_movk_i32 s0, 0xff02
	s_mov_b32 s1, -1
	v_lshl_add_u64 v[28:29], v[28:29], 0, s[0:1]
	v_or_b32_e32 v31, s79, v29
	s_mov_b32 s0, 0
	v_cmp_ne_u32_e32 vcc, 0, v31
	s_and_saveexec_b64 s[2:3], vcc
	s_xor_b64 s[18:19], exec, s[2:3]
	s_cbranch_execz .LBB0_103
	s_add_u32 s2, s78, s82
	s_mov_b32 s83, s82
	s_addc_u32 s3, s79, s82
	s_xor_b64 s[20:21], s[2:3], s[82:83]
	v_cvt_f32_u32_e32 v31, s20
	v_cvt_f32_u32_e32 v32, s21
	s_sub_u32 s1, 0, s20
	s_subb_u32 s2, 0, s21
	v_ashrrev_i32_e32 v38, 31, v29
	v_fmamk_f32 v31, v32, 0x4f800000, v31
	v_rcp_f32_e32 v31, v31
	v_mov_b32_e32 v39, v38
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[38:39]
	v_mov_b32_e32 v35, 0
	v_mul_f32_e32 v31, 0x5f7ffffc, v31
	v_mul_f32_e32 v32, 0x2f800000, v31
	v_trunc_f32_e32 v32, v32
	v_fmamk_f32 v31, v32, 0xcf800000, v31
	v_cvt_u32_f32_e32 v32, v32
	v_cvt_u32_f32_e32 v31, v31
	v_readfirstlane_b32 s3, v32
	v_readfirstlane_b32 s14, v31
	s_mul_i32 s15, s1, s3
	s_mul_hi_u32 s22, s1, s14
	s_mul_i32 s16, s2, s14
	s_add_i32 s15, s22, s15
	s_add_i32 s15, s15, s16
	s_mul_i32 s23, s1, s14
	s_mul_i32 s22, s14, s15
	s_mul_hi_u32 s24, s14, s23
	s_mul_hi_u32 s16, s14, s15
	s_add_u32 s22, s24, s22
	s_addc_u32 s16, 0, s16
	s_mul_hi_u32 s25, s3, s23
	s_mul_i32 s23, s3, s23
	s_add_u32 s22, s22, s23
	s_mul_hi_u32 s24, s3, s15
	s_addc_u32 s16, s16, s25
	s_addc_u32 s22, s24, 0
	s_mul_i32 s15, s3, s15
	s_add_u32 s15, s16, s15
	s_addc_u32 s16, 0, s22
	s_add_u32 s14, s14, s15
	s_addc_u32 s3, s3, s16
	s_mul_i32 s15, s1, s3
	s_mul_hi_u32 s16, s1, s14
	s_add_i32 s15, s16, s15
	s_mul_i32 s2, s2, s14
	s_add_i32 s15, s15, s2
	s_mul_i32 s1, s1, s14
	s_mul_hi_u32 s16, s3, s1
	s_mul_i32 s22, s3, s1
	s_mul_i32 s24, s14, s15
	s_mul_hi_u32 s1, s14, s1
	s_mul_hi_u32 s23, s14, s15
	s_add_u32 s1, s1, s24
	s_addc_u32 s23, 0, s23
	s_add_u32 s1, s1, s22
	s_mul_hi_u32 s2, s3, s15
	s_addc_u32 s1, s23, s16
	s_addc_u32 s2, s2, 0
	s_mul_i32 s15, s3, s15
	s_add_u32 s1, s1, s15
	s_addc_u32 s2, 0, s2
	s_add_u32 s1, s14, s1
	s_addc_u32 s14, s3, s2
	v_xor_b32_e32 v32, v28, v38
	v_xor_b32_e32 v31, v29, v38
	v_mad_u64_u32 v[28:29], s[2:3], v32, s14, 0
	v_mul_hi_u32 v34, v32, s1
	v_lshl_add_u64 v[28:29], v[34:35], 0, v[28:29]
	v_mad_u64_u32 v[42:43], s[2:3], v31, s1, 0
	v_add_co_u32_e32 v28, vcc, v28, v42
	v_mad_u64_u32 v[34:35], s[2:3], v31, s14, 0
	s_nop 0
	v_addc_co_u32_e32 v28, vcc, v29, v43, vcc
	v_mov_b32_e32 v29, s0
	s_nop 0
	v_addc_co_u32_e32 v35, vcc, 0, v35, vcc
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[34:35]
	v_mul_lo_u32 v34, s21, v28
	v_mul_lo_u32 v35, s20, v29
	v_mad_u64_u32 v[28:29], s[0:1], s20, v28, 0
	v_add3_u32 v29, v29, v35, v34
	v_sub_u32_e32 v34, v31, v29
	v_mov_b32_e32 v35, s21
	v_sub_co_u32_e32 v28, vcc, v32, v28
	s_nop 1
	v_subb_co_u32_e64 v32, s[2:3], v34, v35, vcc
	v_subrev_co_u32_e64 v34, s[2:3], s20, v28
	v_subb_co_u32_e32 v29, vcc, v31, v29, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v37, s[14:15], 0, v32, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s21, v37
	v_subb_co_u32_e64 v32, s[2:3], v32, v35, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v39, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s20, v34
	v_subrev_co_u32_e64 v35, s[2:3], s20, v34
	s_nop 0
	v_cndmask_b32_e64 v42, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s21, v37
	v_subbrev_co_u32_e64 v32, s[2:3], 0, v32, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v39, v39, v42, s[14:15]
	v_cmp_le_u32_e32 vcc, s21, v29
	v_cmp_ne_u32_e64 s[2:3], 0, v39
	s_nop 0
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s20, v28
	v_cndmask_b32_e64 v32, v37, v32, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v37, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s21, v29
	s_nop 1
	v_cndmask_b32_e32 v31, v31, v37, vcc
	v_cmp_ne_u32_e32 vcc, 0, v31
	v_cndmask_b32_e64 v31, v34, v35, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v28, v28, v31, vcc
	v_cndmask_b32_e32 v29, v29, v32, vcc
	v_xor_b32_e32 v28, v28, v38
	v_xor_b32_e32 v29, v29, v38
	v_sub_co_u32_e32 v34, vcc, v28, v38
	s_nop 1
	v_subb_co_u32_e32 v35, vcc, v29, v38, vcc
.LBB0_103:
	s_andn2_saveexec_b64 s[2:3], s[18:19]
	s_cbranch_execz .LBB0_105
	v_rcp_iflag_f32_e32 v29, v100
	s_sub_i32 s0, 0, s78
	v_mov_b32_e32 v35, 0
	v_mul_f32_e32 v29, 0x4f7ffffe, v29
	v_cvt_u32_f32_e32 v29, v29
	v_mul_lo_u32 v31, s0, v29
	v_mul_hi_u32 v31, v29, v31
	v_add_u32_e32 v29, v29, v31
	v_mul_hi_u32 v29, v28, v29
	v_mul_lo_u32 v29, v29, s78
	v_sub_u32_e32 v28, v28, v29
	v_subrev_u32_e32 v29, s78, v28
	v_cmp_le_u32_e32 vcc, s78, v28
	s_nop 1
	v_cndmask_b32_e32 v28, v28, v29, vcc
	v_subrev_u32_e32 v29, s78, v28
	v_cmp_le_u32_e32 vcc, s78, v28
	s_nop 1
	v_cndmask_b32_e32 v34, v28, v29, vcc
.LBB0_105:
	s_or_b64 exec, exec, s[2:3]
	v_ashrrev_i32_e32 v28, 31, v33
	v_xor_b32_e32 v28, v30, v28
	v_ashrrev_i32_e32 v29, 31, v35
	v_mul_lo_u32 v28, v28, s66
	v_and_b32_e32 v29, s78, v29
	v_add3_u32 v28, v28, v34, v29
	buffer_load_ubyte v37, v28, s[8:11], 0 offen
	s_waitcnt vmcnt(20)
	v_lshlrev_b16_e32 v28, 8, v68
	s_waitcnt vmcnt(18)
	v_lshlrev_b16_e32 v29, 8, v69
	v_or_b32_e32 v28, v66, v28
	v_or_b32_sdwa v29, v67, v29 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v32, 8, v47
	v_or_b32_sdwa v31, v28, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v28, 8, v63
	v_lshlrev_b16_e32 v29, 8, v65
	v_or_b32_e32 v28, v59, v28
	v_or_b32_sdwa v29, v61, v29 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v32, v41, v32 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v30, v28, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v28, 8, v55
	v_lshlrev_b16_e32 v29, 8, v57
	v_or_b32_e32 v28, v51, v28
	v_or_b32_sdwa v29, v53, v29 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt vmcnt(2)
	v_lshlrev_b16_e32 v33, 8, v87
	v_or_b32_sdwa v29, v28, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v28, 8, v44
	v_or_b32_e32 v28, v40, v28
	v_or_b32_sdwa v28, v28, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v32, 8, v86
	v_or_b32_e32 v32, v84, v32
	v_or_b32_sdwa v33, v85, v33 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v38, 8, v73
	v_or_b32_sdwa v35, v32, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v32, 8, v82
	v_lshlrev_b16_e32 v33, 8, v83
	v_or_b32_e32 v32, v78, v32
	v_or_b32_sdwa v33, v79, v33 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v38, v71, v38 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v34, v32, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v32, 8, v76
	v_lshlrev_b16_e32 v33, 8, v77
	v_or_b32_e32 v32, v74, v32
	v_or_b32_sdwa v33, v75, v33 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_or_b32_sdwa v33, v32, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v32, 8, v72
	v_or_b32_e32 v32, v70, v32
	v_or_b32_sdwa v32, v32, v38 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_barrier
	v_and_b32_e32 v38, 0xff, v92
	ds_read_b32 v40, v50 offset:50176
	ds_read_b128 v[46:49], v54 offset:16384
	ds_read_b128 v[66:69], v52 offset:16384
	ds_read_b128 v[70:73], v54 offset:18432
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[88:91], v[8:11], v[0:3], v80, v38 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[50:53], v52 offset:18432
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[128:131], v[8:11], v[20:23], v80, v38 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v41, v64 offset:50176
	ds_read_b128 v[54:57], v56 offset:16384
	ds_read_b128 v[74:77], v58 offset:16384
	v_and_b32_e32 v39, 0xff, v81
	ds_read_b128 v[58:61], v60 offset:16384
	ds_read_b128 v[62:65], v62 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[114:117], v[8:11], v[4:7], v25, v38 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[106:109], v[8:11], v[16:19], v25, v38 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[132:135], v[12:15], v[0:3], v80, v39 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[110:113], v[12:15], v[4:7], v25, v39 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[102:105], v[12:15], v[8:11], v25, v39 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[124:127], v[12:15], v[20:23], v80, v39 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_and_b32_e32 v25, 0xffff, v36
	v_readlane_b32 s0, v144, 27
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[46:49], v[28:31], v[0:3], v40, v25 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[66:69], v[32:35], v[0:3], v40, v37 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[70:73], v[28:31], v[20:23], v40, v25 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[50:53], v[32:35], v[0:3], v40, v37 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[54:57], v[28:31], v[4:7], v41, v25 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[74:77], v[32:35], v[0:3], v41, v37 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[58:61], v[28:31], v[16:19], v41, v25 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[62:65], v[32:35], v[0:3], v41, v37 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 1
	v_mov_b32_e32 v16, s0
.LBB0_106:
	s_and_b64 s[0:1], s[72:73], exec
	s_cselect_b32 s1, s59, s61
	s_cselect_b32 s0, s58, s60
	s_ashr_i32 s2, s1, 31
	s_lshr_b32 s2, s2, 25
	s_add_u32 s0, s0, s2
	s_addc_u32 s1, s1, 0
	s_ashr_i64 s[0:1], s[0:1], 7
	s_sub_u32 s2, 0, s0
	s_subb_u32 s3, 0, s1
	s_add_u32 s14, s0, 1
	s_addc_u32 s15, s1, 0
	s_and_b64 s[0:1], s[72:73], exec
	s_cselect_b32 s15, s3, s15
	s_cselect_b32 s14, s2, s14
	s_mul_i32 s0, s15, s17
	s_mul_hi_u32 s1, s14, s17
	s_add_i32 s16, s1, s0
	s_mul_i32 s17, s14, s17
	v_readlane_b32 s0, v144, 0
	s_add_u32 s18, s17, s0
	s_mul_i32 s0, s70, s15
	s_mul_hi_u32 s1, s70, s14
	s_addc_u32 s19, s16, 0
	s_add_i32 s0, s1, s0
	s_mul_i32 s1, s71, s14
	s_add_i32 s3, s0, s1
	s_mul_i32 s2, s70, s14
	s_lshl_b64 s[62:63], s[2:3], 5
	s_sub_u32 s64, 0, s62
	s_load_dwordx2 s[24:25], s[56:57], 0x60
	s_subb_u32 s65, 0, s63
	s_sub_u32 s0, s18, s62
	s_subb_u32 s1, s19, s63
	s_or_b64 s[20:21], s[0:1], s[50:51]
	s_cmp_lg_u32 s21, 0
	s_cbranch_scc0 .LBB0_164
	s_ashr_i32 s20, s51, 31
	s_add_u32 s22, s50, s20
	s_mov_b32 s21, s20
	s_addc_u32 s23, s51, s20
	s_xor_b64 s[22:23], s[22:23], s[20:21]
	v_cvt_f32_u32_e32 v17, s22
	v_cvt_f32_u32_e32 v18, s23
	s_sub_u32 s26, 0, s22
	s_subb_u32 s27, 0, s23
	v_fmamk_f32 v17, v18, 0x4f800000, v17
	v_rcp_f32_e32 v17, v17
	s_nop 0
	v_mul_f32_e32 v17, 0x5f7ffffc, v17
	v_mul_f32_e32 v18, 0x2f800000, v17
	v_trunc_f32_e32 v18, v18
	v_fmamk_f32 v17, v18, 0xcf800000, v17
	v_cvt_u32_f32_e32 v18, v18
	v_cvt_u32_f32_e32 v17, v17
	v_readfirstlane_b32 s28, v18
	v_readfirstlane_b32 s29, v17
	s_mul_i32 s54, s26, s28
	s_mul_hi_u32 s57, s26, s29
	s_mul_i32 s56, s27, s29
	s_add_i32 s54, s57, s54
	s_add_i32 s54, s54, s56
	s_mul_i32 s58, s26, s29
	s_mul_i32 s57, s29, s54
	s_mul_hi_u32 s59, s29, s58
	s_mul_hi_u32 s56, s29, s54
	s_add_u32 s57, s59, s57
	s_addc_u32 s56, 0, s56
	s_mul_hi_u32 s60, s28, s58
	s_mul_i32 s58, s28, s58
	s_add_u32 s57, s57, s58
	s_mul_hi_u32 s59, s28, s54
	s_addc_u32 s56, s56, s60
	s_addc_u32 s57, s59, 0
	s_mul_i32 s54, s28, s54
	s_add_u32 s54, s56, s54
	s_addc_u32 s56, 0, s57
	s_add_u32 s29, s29, s54
	s_addc_u32 s28, s28, s56
	s_mul_i32 s54, s26, s28
	s_mul_hi_u32 s56, s26, s29
	s_add_i32 s54, s56, s54
	s_mul_i32 s27, s27, s29
	s_add_i32 s54, s54, s27
	s_mul_i32 s26, s26, s29
	s_mul_hi_u32 s56, s28, s26
	s_mul_i32 s57, s28, s26
	s_mul_i32 s59, s29, s54
	s_mul_hi_u32 s26, s29, s26
	s_mul_hi_u32 s58, s29, s54
	s_add_u32 s26, s26, s59
	s_addc_u32 s58, 0, s58
	s_add_u32 s26, s26, s57
	s_mul_hi_u32 s27, s28, s54
	s_addc_u32 s26, s58, s56
	s_addc_u32 s27, s27, 0
	s_mul_i32 s54, s28, s54
	s_add_u32 s26, s26, s54
	s_addc_u32 s27, 0, s27
	s_add_u32 s54, s29, s26
	s_addc_u32 s56, s28, s27
	s_ashr_i32 s26, s1, 31
	s_add_u32 s28, s0, s26
	s_mov_b32 s27, s26
	s_addc_u32 s29, s1, s26
	s_xor_b64 s[28:29], s[28:29], s[26:27]
	s_mul_i32 s57, s28, s56
	s_mul_hi_u32 s58, s28, s54
	s_mul_hi_u32 s1, s28, s56
	s_add_u32 s57, s58, s57
	s_addc_u32 s1, 0, s1
	s_mul_hi_u32 s59, s29, s54
	s_mul_i32 s54, s29, s54
	s_add_u32 s54, s57, s54
	s_mul_hi_u32 s58, s29, s56
	s_addc_u32 s1, s1, s59
	s_addc_u32 s54, s58, 0
	s_mul_i32 s56, s29, s56
	s_add_u32 s1, s1, s56
	s_addc_u32 s54, 0, s54
	s_mul_i32 s54, s22, s54
	s_mul_hi_u32 s56, s22, s1
	s_add_i32 s54, s56, s54
	s_mul_i32 s56, s23, s1
	s_add_i32 s54, s54, s56
	s_sub_i32 s58, s29, s54
	s_mul_i32 s1, s22, s1
	s_sub_u32 s1, s28, s1
	s_cselect_b64 s[56:57], -1, 0
	s_subb_u32 s28, s58, s23
	s_sub_u32 s60, s1, s22
	s_cselect_b64 s[58:59], -1, 0
	s_subb_u32 s61, s28, 0
	s_cmp_ge_u32 s61, s23
	s_cselect_b32 s68, -1, 0
	s_cmp_ge_u32 s60, s22
	s_cselect_b32 s69, -1, 0
	s_cmp_eq_u32 s61, s23
	s_cselect_b32 s68, s69, s68
	s_cmp_lg_u64 s[58:59], 0
	s_subb_u32 s28, s28, s23
	s_sub_u32 s58, s60, s22
	s_subb_u32 s28, s28, 0
	s_cmp_lg_u32 s68, 0
	s_cselect_b32 s58, s58, s60
	s_cselect_b32 s28, s28, s61
	s_cmp_lg_u64 s[56:57], 0
	s_subb_u32 s29, s29, s54
	s_cmp_ge_u32 s29, s23
	s_cselect_b32 s54, -1, 0
	s_cmp_ge_u32 s1, s22
	s_cselect_b32 s22, -1, 0
	s_cmp_eq_u32 s29, s23
	s_cselect_b32 s22, s22, s54
	s_cmp_lg_u32 s22, 0
	s_cselect_b32 s23, s28, s29
	s_cselect_b32 s22, s58, s1
	s_xor_b64 s[22:23], s[22:23], s[26:27]
	s_sub_u32 s22, s22, s26
	s_subb_u32 s23, s23, s26
	v_cvt_f32_u32_e32 v48, s50
	s_cbranch_execnz .LBB0_109
.LBB0_108:
	v_rcp_iflag_f32_e32 v17, v48
	s_sub_i32 s1, 0, s50
	s_mov_b32 s23, 0
	v_mul_f32_e32 v17, 0x4f7ffffe, v17
	v_cvt_u32_f32_e32 v17, v17
	s_nop 0
	v_readfirstlane_b32 s20, v17
	s_mul_i32 s1, s1, s20
	s_mul_hi_u32 s1, s20, s1
	s_add_i32 s20, s20, s1
	s_mul_hi_u32 s1, s0, s20
	s_mul_i32 s1, s1, s50
	s_sub_i32 s0, s0, s1
	s_sub_i32 s1, s0, s50
	s_cmp_ge_u32 s0, s50
	s_cselect_b32 s0, s1, s0
	s_sub_i32 s1, s0, s50
	s_cmp_ge_u32 s0, s50
	s_cselect_b32 s22, s1, s0
.LBB0_109:
	s_ashr_i32 s0, s19, 31
	s_mov_b32 s1, s0
	s_xor_b64 s[20:21], s[0:1], s[18:19]
	s_ashr_i32 s26, s21, 31
	s_lshr_b32 s26, s26, 27
	s_add_u32 s20, s20, s26
	s_addc_u32 s21, s21, 0
	s_ashr_i64 s[20:21], s[20:21], 5
	s_xor_b64 s[68:69], s[20:21], s[0:1]
	s_ashr_i32 s20, s69, 31
	s_add_u32 s0, s68, s20
	s_mov_b32 s21, s20
	s_addc_u32 s1, s69, s20
	s_xor_b64 s[0:1], s[0:1], s[20:21]
	s_ashr_i32 s26, s15, 31
	s_add_u32 s28, s14, s26
	s_mov_b32 s27, s26
	s_addc_u32 s29, s15, s26
	s_xor_b64 s[28:29], s[28:29], s[26:27]
	v_cvt_f32_u32_e32 v17, s28
	v_cvt_f32_u32_e32 v18, s29
	s_sub_u32 s54, 0, s28
	s_subb_u32 s56, 0, s29
	v_cmp_ne_u32_e32 vcc, 0, v16
	v_fmac_f32_e32 v17, 0x4f800000, v18
	v_rcp_f32_e32 v17, v17
	s_nop 0
	v_mul_f32_e32 v17, 0x5f7ffffc, v17
	v_mul_f32_e32 v18, 0x2f800000, v17
	v_trunc_f32_e32 v18, v18
	v_fmac_f32_e32 v17, 0xcf800000, v18
	v_cvt_u32_f32_e32 v18, v18
	v_cvt_u32_f32_e32 v17, v17
	v_readfirstlane_b32 s57, v18
	v_readfirstlane_b32 s58, v17
	s_mul_hi_u32 s60, s54, s58
	s_mul_i32 s61, s54, s57
	s_mul_i32 s59, s56, s58
	s_add_i32 s60, s60, s61
	s_add_i32 s60, s60, s59
	s_mul_i32 s70, s54, s58
	s_mul_i32 s61, s58, s60
	s_mul_hi_u32 s71, s58, s70
	s_mul_hi_u32 s59, s58, s60
	s_add_u32 s61, s71, s61
	s_addc_u32 s59, 0, s59
	s_mul_hi_u32 s72, s57, s70
	s_mul_i32 s70, s57, s70
	s_add_u32 s61, s61, s70
	s_mul_hi_u32 s71, s57, s60
	s_addc_u32 s59, s59, s72
	s_addc_u32 s61, s71, 0
	s_mul_i32 s60, s57, s60
	s_add_u32 s59, s59, s60
	s_addc_u32 s60, 0, s61
	s_add_u32 s58, s58, s59
	s_addc_u32 s57, s57, s60
	s_mul_i32 s59, s54, s57
	s_mul_hi_u32 s60, s54, s58
	s_add_i32 s59, s60, s59
	s_mul_i32 s56, s56, s58
	s_add_i32 s59, s59, s56
	s_mul_i32 s54, s54, s58
	s_mul_hi_u32 s60, s57, s54
	s_mul_i32 s61, s57, s54
	s_mul_i32 s71, s58, s59
	s_mul_hi_u32 s54, s58, s54
	s_mul_hi_u32 s70, s58, s59
	s_add_u32 s54, s54, s71
	s_addc_u32 s70, 0, s70
	s_add_u32 s54, s54, s61
	s_mul_hi_u32 s56, s57, s59
	s_addc_u32 s54, s70, s60
	s_addc_u32 s56, s56, 0
	s_mul_i32 s59, s57, s59
	s_add_u32 s54, s54, s59
	s_addc_u32 s59, 0, s56
	s_add_u32 s56, s58, s54
	s_addc_u32 s54, s57, s59
	s_mul_i32 s58, s0, s54
	s_mul_hi_u32 s59, s0, s56
	s_mul_hi_u32 s57, s0, s54
	s_add_u32 s58, s59, s58
	s_addc_u32 s57, 0, s57
	s_mul_i32 s61, s1, s56
	s_mul_hi_u32 s60, s1, s56
	s_add_u32 s58, s58, s61
	s_mul_hi_u32 s59, s1, s54
	s_addc_u32 s57, s57, s60
	s_addc_u32 s58, s59, 0
	s_mul_i32 s59, s1, s54
	s_add_u32 s57, s57, s59
	s_addc_u32 s58, 0, s58
	s_mul_i32 s58, s28, s58
	s_mul_hi_u32 s59, s28, s57
	s_add_i32 s58, s59, s58
	s_mul_i32 s59, s29, s57
	s_add_i32 s70, s58, s59
	s_sub_i32 s60, s1, s70
	s_mul_i32 s57, s28, s57
	s_sub_u32 s0, s0, s57
	s_cselect_b64 s[58:59], -1, 0
	s_subb_u32 s57, s60, s29
	s_sub_u32 s71, s0, s28
	s_cselect_b64 s[60:61], -1, 0
	s_subb_u32 s72, s57, 0
	s_cmp_ge_u32 s72, s29
	s_cselect_b32 s73, -1, 0
	s_cmp_ge_u32 s71, s28
	s_cselect_b32 s74, -1, 0
	s_cmp_eq_u32 s72, s29
	s_cselect_b32 s73, s74, s73
	s_cmp_lg_u64 s[60:61], 0
	s_subb_u32 s57, s57, s29
	s_sub_u32 s60, s71, s28
	s_subb_u32 s57, s57, 0
	s_cmp_lg_u32 s73, 0
	s_cselect_b32 s60, s60, s71
	s_cselect_b32 s57, s57, s72
	s_cmp_lg_u64 s[58:59], 0
	s_subb_u32 s1, s1, s70
	s_cmp_ge_u32 s1, s29
	s_cselect_b32 s58, -1, 0
	s_cmp_ge_u32 s0, s28
	s_cselect_b32 s59, -1, 0
	s_cmp_eq_u32 s1, s29
	s_cselect_b32 s58, s59, s58
	s_cmp_lg_u32 s58, 0
	s_cselect_b32 s1, s57, s1
	s_cselect_b32 s0, s60, s0
	s_xor_b64 s[0:1], s[0:1], s[20:21]
	s_sub_u32 s57, s0, s20
	s_subb_u32 s58, s1, s20
	s_ashr_i32 s0, s58, 31
	s_mov_b32 s1, s0
	s_and_b64 s[0:1], s[0:1], s[14:15]
	s_add_u32 s90, s0, s57
	s_addc_u32 s91, s1, s58
	v_cmp_lt_i64_e64 s[0:1], s[2:3], 1
	v_cmp_gt_i64_e64 s[2:3], s[2:3], 0
	s_and_b64 s[0:1], vcc, s[0:1]
	s_ashr_i32 s70, s51, 31
	v_cndmask_b32_e64 v16, 0, -1, s[2:3]
	s_mov_b32 s71, s70
	v_readfirstlane_b32 s2, v16
	s_mov_b32 s3, s2
	s_xor_b64 s[58:59], s[64:65], s[2:3]
	s_add_u32 s2, s50, s70
	s_addc_u32 s3, s51, s70
	s_xor_b64 s[2:3], s[2:3], s[70:71]
	v_cvt_f32_u32_e32 v17, s2
	v_cvt_f32_u32_e32 v18, s3
	s_sub_u32 s57, 0, s2
	s_subb_u32 s60, 0, s3
	s_mov_b32 s73, 0
	v_fmac_f32_e32 v17, 0x4f800000, v18
	v_rcp_f32_e32 v17, v17
	s_nop 0
	v_mul_f32_e32 v17, 0x5f7ffffc, v17
	v_mul_f32_e32 v18, 0x2f800000, v17
	v_trunc_f32_e32 v18, v18
	v_fmac_f32_e32 v17, 0xcf800000, v18
	v_cvt_u32_f32_e32 v17, v17
	v_cvt_u32_f32_e32 v18, v18
	v_readfirstlane_b32 s61, v17
	v_readfirstlane_b32 s75, v18
	s_mul_hi_u32 s74, s57, s61
	s_mul_i32 s76, s57, s75
	s_mul_i32 s72, s60, s61
	s_add_i32 s74, s74, s76
	s_add_i32 s74, s74, s72
	s_mul_i32 s77, s57, s61
	s_mul_i32 s76, s61, s74
	s_mul_hi_u32 s78, s61, s77
	s_mul_hi_u32 s72, s61, s74
	s_add_u32 s76, s78, s76
	s_addc_u32 s72, 0, s72
	s_mul_hi_u32 s79, s75, s77
	s_mul_i32 s77, s75, s77
	s_add_u32 s76, s76, s77
	s_mul_hi_u32 s78, s75, s74
	s_addc_u32 s72, s72, s79
	s_addc_u32 s76, s78, 0
	s_mul_i32 s74, s75, s74
	s_add_u32 s72, s72, s74
	s_addc_u32 s74, 0, s76
	s_add_u32 s61, s61, s72
	s_addc_u32 s72, s75, s74
	s_mul_i32 s74, s57, s72
	s_mul_hi_u32 s75, s57, s61
	s_add_i32 s74, s75, s74
	s_mul_i32 s60, s60, s61
	s_add_i32 s74, s74, s60
	s_mul_i32 s57, s57, s61
	s_mul_hi_u32 s75, s72, s57
	s_mul_i32 s76, s72, s57
	s_mul_i32 s78, s61, s74
	s_mul_hi_u32 s57, s61, s57
	s_mul_hi_u32 s77, s61, s74
	s_add_u32 s57, s57, s78
	s_addc_u32 s77, 0, s77
	s_add_u32 s57, s57, s76
	s_mul_hi_u32 s60, s72, s74
	s_addc_u32 s57, s77, s75
	s_addc_u32 s60, s60, 0
	s_mul_i32 s74, s72, s74
	s_add_u32 s57, s57, s74
	s_addc_u32 s60, 0, s60
	s_add_u32 s57, s61, s57
	s_addc_u32 s72, s72, s60
	s_ashr_i32 s60, s59, 31
	s_add_u32 s58, s58, s60
	s_mov_b32 s61, s60
	s_addc_u32 s59, s59, s60
	s_xor_b64 s[58:59], s[58:59], s[60:61]
	s_mul_i32 s75, s58, s72
	s_mul_hi_u32 s76, s58, s57
	s_mul_hi_u32 s74, s58, s72
	s_add_u32 s75, s76, s75
	s_addc_u32 s74, 0, s74
	s_mul_hi_u32 s77, s59, s57
	s_mul_i32 s57, s59, s57
	s_add_u32 s57, s75, s57
	s_mul_hi_u32 s76, s59, s72
	s_addc_u32 s57, s74, s77
	s_addc_u32 s74, s76, 0
	s_mul_i32 s72, s59, s72
	s_add_u32 s57, s57, s72
	s_addc_u32 s72, 0, s74
	s_mul_i32 s74, s2, s72
	s_mul_hi_u32 s75, s2, s57
	s_add_i32 s74, s75, s74
	s_mul_i32 s75, s3, s57
	s_add_i32 s76, s74, s75
	s_sub_i32 s77, s59, s76
	s_mul_i32 s74, s2, s57
	s_sub_u32 s58, s58, s74
	s_cselect_b64 s[74:75], -1, 0
	s_subb_u32 s77, s77, s3
	s_sub_u32 s78, s58, s2
	s_subb_u32 s77, s77, 0
	s_cmp_ge_u32 s77, s3
	s_cselect_b32 s79, -1, 0
	s_cmp_ge_u32 s78, s2
	s_cselect_b32 s78, -1, 0
	s_cmp_eq_u32 s77, s3
	s_cselect_b32 s77, s78, s79
	s_add_u32 s78, s57, 1
	s_addc_u32 s79, s72, 0
	s_add_u32 s80, s57, 2
	s_addc_u32 s81, s72, 0
	s_cmp_lg_u32 s77, 0
	s_cselect_b32 s77, s80, s78
	s_cselect_b32 s78, s81, s79
	s_cmp_lg_u64 s[74:75], 0
	s_subb_u32 s59, s59, s76
	s_cmp_ge_u32 s59, s3
	s_cselect_b32 s74, -1, 0
	s_cmp_ge_u32 s58, s2
	s_cselect_b32 s2, -1, 0
	s_cmp_eq_u32 s59, s3
	s_cselect_b32 s2, s2, s74
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s3, s78, s72
	s_cselect_b32 s2, s77, s57
	s_xor_b64 s[58:59], s[60:61], s[70:71]
	s_xor_b64 s[2:3], s[2:3], s[58:59]
	s_sub_u32 s2, s2, s58
	s_subb_u32 s3, s3, s59
	v_xor_b32_e32 v17, s2, v16
	v_xor_b32_e32 v16, s3, v16
	s_mul_i32 s2, s48, s15
	s_mul_hi_u32 s3, s48, s14
	s_add_i32 s2, s3, s2
	s_mul_i32 s3, s49, s14
	s_add_i32 s2, s2, s3
	s_mul_i32 s3, s48, s14
	v_cndmask_b32_e64 v50, 0, v16, s[0:1]
	v_cndmask_b32_e64 v49, 0, v17, s[0:1]
	v_mov_b32_e32 v16, s3
	v_mov_b32_e32 v17, s2
	v_cmp_ge_i64_e64 s[2:3], s[18:19], v[16:17]
	s_and_b64 s[88:89], vcc, s[2:3]
	s_ashr_i32 s2, s23, 31
	s_mov_b32 s3, s2
	s_and_b64 s[2:3], s[2:3], s[50:51]
	s_add_u32 s22, s22, s48
	s_addc_u32 s23, s23, s49
	s_add_u32 s58, s22, s2
	s_addc_u32 s59, s23, s3
	s_xor_b64 s[2:3], s[20:21], s[68:69]
	s_ashr_i32 s22, s3, 31
	s_add_u32 s2, s2, s22
	s_mov_b32 s23, s22
	s_addc_u32 s3, s3, s22
	s_xor_b64 s[2:3], s[2:3], s[22:23]
	s_mul_i32 s60, s2, s54
	s_mul_hi_u32 s61, s2, s56
	s_mul_hi_u32 s57, s2, s54
	s_add_u32 s60, s61, s60
	s_addc_u32 s57, 0, s57
	s_mul_hi_u32 s68, s3, s56
	s_mul_i32 s56, s3, s56
	s_add_u32 s56, s60, s56
	s_mul_hi_u32 s61, s3, s54
	s_addc_u32 s56, s57, s68
	s_addc_u32 s57, s61, 0
	s_mul_i32 s54, s3, s54
	s_add_u32 s54, s56, s54
	s_addc_u32 s60, 0, s57
	s_mul_i32 s56, s28, s60
	s_mul_hi_u32 s57, s28, s54
	s_add_i32 s56, s57, s56
	s_mul_i32 s57, s29, s54
	s_add_i32 s61, s56, s57
	s_sub_i32 s68, s3, s61
	s_mul_i32 s56, s28, s54
	s_sub_u32 s2, s2, s56
	s_cselect_b64 s[56:57], -1, 0
	s_subb_u32 s68, s68, s29
	s_sub_u32 s69, s2, s28
	s_subb_u32 s68, s68, 0
	s_cmp_ge_u32 s68, s29
	s_cselect_b32 s71, -1, 0
	s_cmp_ge_u32 s69, s28
	s_cselect_b32 s69, -1, 0
	s_cmp_eq_u32 s68, s29
	s_cselect_b32 s68, s69, s71
	s_add_u32 s69, s54, 1
	s_addc_u32 s71, s60, 0
	s_add_u32 s72, s54, 2
	s_addc_u32 s74, s60, 0
	s_cmp_lg_u32 s68, 0
	s_cselect_b32 s68, s72, s69
	s_cselect_b32 s69, s74, s71
	s_cmp_lg_u64 s[56:57], 0
	s_subb_u32 s3, s3, s61
	s_cmp_ge_u32 s3, s29
	s_cselect_b32 s56, -1, 0
	s_cmp_ge_u32 s2, s28
	s_cselect_b32 s2, -1, 0
	s_cmp_eq_u32 s3, s29
	s_cselect_b32 s2, s2, s56
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s3, s69, s60
	s_cselect_b32 s2, s68, s54
	s_xor_b64 s[22:23], s[22:23], s[26:27]
	s_xor_b64 s[2:3], s[2:3], s[22:23]
	s_sub_u32 s2, s2, s22
	s_subb_u32 s3, s3, s23
	s_xor_b64 s[2:3], s[2:3], s[20:21]
	s_lshl_b64 s[2:3], s[2:3], 5
	s_and_b32 s72, s18, 31
	s_or_b64 s[2:3], s[2:3], s[72:73]
	v_mov_b64_e32 v[16:17], s[34:35]
	s_and_b64 s[20:21], s[88:89], exec
	v_cmp_ge_i64_e32 vcc, s[52:53], v[16:17]
	s_cselect_b32 s69, s59, s3
	s_cselect_b32 s68, s58, s2
	s_cbranch_vccnz .LBB0_128
	v_readlane_b32 s26, v144, 13
	v_readlane_b32 s27, v144, 14
	s_and_b64 s[2:3], s[26:27], exec
	v_readlane_b32 s2, v144, 6
	v_readlane_b32 s20, v144, 8
	v_readlane_b32 s3, v144, 7
	v_readlane_b32 s21, v144, 9
	s_cselect_b32 s3, s3, s21
	s_cselect_b32 s2, s2, s20
	s_ashr_i32 s20, s3, 31
	s_lshr_b32 s20, s20, 27
	s_add_u32 s2, s2, s20
	s_addc_u32 s3, s3, 0
	s_ashr_i64 s[2:3], s[2:3], 5
	s_sub_u32 s20, 0, s2
	s_subb_u32 s21, 0, s3
	s_add_u32 s22, s2, 1
	s_addc_u32 s23, s3, 0
	s_and_b64 s[2:3], s[26:27], exec
	s_cselect_b32 s2, s20, s22
	s_cselect_b32 s3, s21, s23
	s_and_b32 s20, s2, 31
	s_max_u32 s26, s20, 1
	s_cmp_lg_u32 s20, 0
	s_cselect_b64 s[20:21], -1, 0
	s_ashr_i32 s22, s3, 31
	s_mov_b32 s23, s22
	s_xor_b64 s[2:3], s[22:23], s[2:3]
	s_ashr_i32 s27, s3, 31
	s_lshr_b32 s27, s27, 27
	s_add_u32 s2, s2, s27
	s_addc_u32 s3, s3, 0
	s_ashr_i64 s[2:3], s[2:3], 5
	s_xor_b64 s[2:3], s[2:3], s[22:23]
	s_mul_i32 s3, s14, s3
	s_mul_hi_u32 s22, s14, s2
	s_add_i32 s3, s22, s3
	s_mul_i32 s15, s15, s2
	s_add_i32 s3, s3, s15
	s_mul_i32 s2, s14, s2
	s_lshl_b64 s[14:15], s[2:3], 5
	v_mov_b64_e32 v[16:17], s[14:15]
	v_cmp_ge_i64_e32 vcc, s[18:19], v[16:17]
	v_cvt_f32_u32_e32 v16, s26
	v_mov_b32_e32 v17, 0x4f800000
	s_and_b64 s[2:3], s[20:21], vcc
	v_readlane_b32 s18, v144, 0
	v_fmac_f32_e32 v16, 0, v17
	v_rcp_f32_e32 v16, v16
	s_sub_u32 s14, s18, s14
	s_subb_u32 s15, 0, s15
	s_add_u32 s18, s14, s17
	v_mul_f32_e32 v16, 0x5f7ffffc, v16
	v_mul_f32_e32 v17, 0x2f800000, v16
	v_trunc_f32_e32 v17, v17
	v_fmac_f32_e32 v16, 0xcf800000, v17
	s_addc_u32 s19, s15, s16
	v_cvt_u32_f32_e32 v18, v17
	v_cvt_u32_f32_e32 v16, v16
	s_ashr_i32 s14, s19, 31
	s_mov_b32 s15, s14
	s_xor_b64 s[18:19], s[14:15], s[18:19]
	s_sub_u32 s20, 0, s26
	v_readfirstlane_b32 s22, v18
	v_readfirstlane_b32 s27, v16
	s_subb_u32 s21, 0, 0
	s_mul_i32 s23, s20, s22
	s_mul_hi_u32 s28, s20, s27
	s_add_i32 s23, s28, s23
	s_mul_i32 s28, s21, s27
	s_add_i32 s23, s23, s28
	s_mul_i32 s54, s20, s27
	s_mul_i32 s29, s27, s23
	s_mul_hi_u32 s56, s27, s54
	s_mul_hi_u32 s28, s27, s23
	s_add_u32 s29, s56, s29
	s_addc_u32 s28, 0, s28
	s_mul_hi_u32 s57, s22, s54
	s_mul_i32 s54, s22, s54
	s_add_u32 s29, s29, s54
	s_mul_hi_u32 s56, s22, s23
	s_addc_u32 s28, s28, s57
	s_addc_u32 s29, s56, 0
	s_mul_i32 s23, s22, s23
	s_add_u32 s23, s28, s23
	s_addc_u32 s28, 0, s29
	s_add_u32 s23, s27, s23
	s_addc_u32 s22, s22, s28
	s_mul_i32 s27, s20, s22
	s_mul_hi_u32 s28, s20, s23
	s_add_i32 s27, s28, s27
	s_mul_i32 s21, s21, s23
	s_add_i32 s27, s27, s21
	s_mul_i32 s20, s20, s23
	s_mul_hi_u32 s28, s22, s20
	s_mul_i32 s29, s22, s20
	s_mul_i32 s56, s23, s27
	s_mul_hi_u32 s20, s23, s20
	s_mul_hi_u32 s54, s23, s27
	s_add_u32 s20, s20, s56
	s_addc_u32 s54, 0, s54
	s_add_u32 s20, s20, s29
	s_mul_hi_u32 s21, s22, s27
	s_addc_u32 s20, s54, s28
	s_addc_u32 s21, s21, 0
	s_mul_i32 s27, s22, s27
	s_add_u32 s20, s20, s27
	s_addc_u32 s21, 0, s21
	s_add_u32 s23, s23, s20
	s_addc_u32 s22, s22, s21
	s_ashr_i32 s20, s19, 31
	s_add_u32 s18, s18, s20
	s_mov_b32 s21, s20
	s_addc_u32 s19, s19, s20
	s_xor_b64 s[18:19], s[18:19], s[20:21]
	s_mul_i32 s28, s18, s22
	s_mul_hi_u32 s29, s18, s23
	s_mul_hi_u32 s27, s18, s22
	s_add_u32 s28, s29, s28
	s_addc_u32 s27, 0, s27
	s_mul_hi_u32 s54, s19, s23
	s_mul_i32 s23, s19, s23
	s_add_u32 s23, s28, s23
	s_mul_hi_u32 s29, s19, s22
	s_addc_u32 s23, s27, s54
	s_addc_u32 s27, s29, 0
	s_mul_i32 s22, s19, s22
	s_add_u32 s22, s23, s22
	s_addc_u32 s23, 0, s27
	s_add_u32 s27, s22, 1
	s_addc_u32 s28, s23, 0
	s_add_u32 s29, s22, 2
	s_mul_i32 s56, s26, s23
	s_mul_hi_u32 s57, s26, s22
	s_addc_u32 s54, s23, 0
	s_add_i32 s57, s57, s56
	s_mul_i32 s56, s26, s22
	s_sub_u32 s18, s18, s56
	s_subb_u32 s19, s19, s57
	s_sub_u32 s56, s18, s26
	s_subb_u32 s57, s19, 0
	s_cmp_ge_u32 s56, s26
	s_cselect_b32 s56, -1, 0
	s_cmp_eq_u32 s57, 0
	s_cselect_b32 s56, s56, -1
	s_cmp_lg_u32 s56, 0
	s_cselect_b32 s27, s29, s27
	s_cselect_b32 s28, s54, s28
	s_cmp_ge_u32 s18, s26
	s_cselect_b32 s18, -1, 0
	s_cmp_eq_u32 s19, 0
	s_cselect_b32 s18, s18, -1
	s_cmp_lg_u32 s18, 0
	s_cselect_b32 s19, s28, s23
	s_cselect_b32 s18, s27, s22
	s_xor_b64 s[18:19], s[18:19], s[20:21]
	s_sub_u32 s18, s18, s20
	s_subb_u32 s19, s19, s20
	s_xor_b64 s[14:15], s[18:19], s[14:15]
	s_and_b64 s[2:3], s[2:3], exec
	s_cselect_b32 s15, s15, s91
	s_cselect_b32 s14, s14, s90
	s_lshl_b64 s[2:3], s[14:15], 7
	v_or_b32_e32 v17, s3, v27
	v_or_b32_e32 v16, s2, v26
	v_or_b32_e32 v18, 32, v16
	v_mov_b32_e32 v19, v17
	v_readlane_b32 s22, v144, 15
	v_cmp_gt_i64_e64 s[2:3], s[12:13], v[16:17]
	v_cmp_gt_i64_e64 s[26:27], s[12:13], v[18:19]
	v_or_b32_e32 v18, 64, v16
	v_or_b32_e32 v16, 0x60, v16
	v_lshlrev_b32_e32 v32, 2, v96
	s_lshl_b32 s71, s22, 7
	v_readlane_b32 s22, v144, 10
	v_cmp_gt_i64_e64 s[20:21], s[12:13], v[16:17]
	v_lshl_add_u32 v16, s86, 8, v32
	s_lshl_b32 s72, s22, 7
	v_readlane_b32 s22, v144, 11
	v_or_b32_e32 v27, 0xc800, v16
	v_and_b32_e32 v16, 7, v96
	s_lshl_b32 s28, s86, 13
	v_lshlrev_b32_e32 v17, 14, v49
	s_lshl_b32 s74, s22, 7
	v_readlane_b32 s22, v144, 12
	v_cmp_gt_i64_e64 s[18:19], s[12:13], v[18:19]
	v_bitop3_b32 v18, v45, v16, 4 bitop3:0x36
	v_or_b32_e32 v17, s28, v17
	s_lshl_b32 s92, s22, 7
	s_lshl_b64 s[22:23], s[14:15], 2
	s_sub_i32 s15, 0, s28
	s_mov_b32 s28, 0x8000
	v_lshl_add_u32 v47, v18, 4, s28
	v_lshrrev_b32_e32 v18, 1, v98
	v_bitop3_b32 v19, v98, v96, 7 bitop3:0x78
	v_mov_b32_e32 v25, 0
	v_mul_i32_i24_e32 v20, 0xfffffe00, v18
	v_mul_i32_i24_e32 v18, -16, v45
	v_lshl_add_u32 v46, v19, 4, s28
	v_ashrrev_i32_e32 v19, 31, v18
	v_add_u32_e32 v30, v97, v96
	v_mov_b32_e32 v31, v25
	s_lshl_b64 s[28:29], s[68:69], 5
	v_lshl_add_u64 v[18:19], v[30:31], 0, v[18:19]
	v_lshl_add_u64 v[30:31], v[18:19], 0, s[28:29]
	v_ashrrev_i32_e32 v34, 31, v31
	s_ashr_i32 s75, s15, 31
	v_xor_b32_e32 v19, v34, v31
	v_bfe_u32 v38, v18, 4, 1
	v_xor_b32_e32 v18, v34, v30
	v_ashrrev_i32_e32 v30, 31, v19
	s_add_u32 s54, s46, 7
	v_lshrrev_b32_e32 v30, 27, v30
	v_mov_b32_e32 v31, s73
	s_addc_u32 s56, s47, 0
	v_lshl_add_u64 v[18:19], v[18:19], 0, v[30:31]
	s_sub_u32 s57, -8, s46
	v_or_b32_e32 v51, s28, v97
	v_ashrrev_i64 v[18:19], 5, v[18:19]
	s_subb_u32 s58, -1, s47
	v_cmp_lt_i64_e64 s[28:29], s[46:47], -7
	v_xor_b32_e32 v52, v18, v34
	v_mul_i32_i24_e32 v22, 0xffffff00, v45
	v_cndmask_b32_e64 v18, 0, -1, s[28:29]
	s_and_b64 s[28:29], s[28:29], exec
	s_cselect_b32 s29, s58, s56
	s_cselect_b32 s28, s57, s54
	s_ashr_i32 s47, s29, 31
	s_lshr_b32 s47, s47, 29
	s_add_u32 s28, s28, s47
	s_addc_u32 s29, s29, 0
	v_readfirstlane_b32 s56, v18
	s_ashr_i64 s[28:29], s[28:29], 3
	s_mov_b32 s57, s56
	s_xor_b64 s[28:29], s[28:29], s[56:57]
	s_lshl_b32 s47, s28, 8
	s_lshl_b64 s[58:59], s[28:29], 3
	s_lshl_b64 s[56:57], s[52:53], 11
	v_ashrrev_i32_e32 v23, 31, v22
	s_sub_u32 s60, 0, s56
	v_sub_co_u32_e32 v41, vcc, 0, v22
	s_subb_u32 s61, 0, s57
	s_nop 0
	v_subb_co_u32_e32 v53, vcc, 0, v23, vcc
	v_lshl_add_u64 v[22:23], s[56:57], 0, v[22:23]
	v_readlane_b32 s56, v144, 3
	v_readlane_b32 s57, v144, 4
	s_lshl_b32 s55, s55, 5
	s_mov_b32 s54, s73
	s_lshl_b64 s[56:57], s[56:57], 5
	s_or_b64 s[54:55], s[54:55], s[56:57]
	v_readlane_b32 s54, v144, 5
	s_add_i32 s54, s86, s54
	s_add_u32 s22, s22, s54
	s_addc_u32 s23, s23, 0
	s_mul_i32 s23, s56, s23
	s_mul_hi_u32 s54, s56, s22
	v_lshlrev_b32_e32 v40, 4, v96
	v_ashrrev_i32_e32 v21, 31, v20
	v_sub_co_u32_e32 v18, vcc, v41, v20
	s_add_i32 s23, s54, s23
	s_mul_i32 s54, s55, s22
	v_mul_i32_i24_e32 v16, 0xfffff800, v45
	v_lshl_add_u32 v28, v98, 8, v40
	v_mul_i32_i24_e32 v36, 0xffffffc0, v45
	v_xor_b32_e32 v45, v19, v34
	v_subb_co_u32_e32 v19, vcc, v53, v21, vcc
	s_add_i32 s54, s23, s54
	s_mul_i32 s55, s56, s22
	s_lshl_b64 s[22:23], s[52:53], 13
	v_sub_co_u32_e32 v18, vcc, v18, v28
	s_add_u32 s22, s55, s22
	s_nop 0
	v_subbrev_co_u32_e32 v19, vcc, 0, v19, vcc
	v_and_b32_e32 v30, 0x200, v40
	s_addc_u32 s23, s54, s23
	v_sub_co_u32_e32 v18, vcc, v18, v30
	s_add_u32 s22, s15, s22
	v_mov_b32_e32 v33, v25
	v_mov_b32_e32 v29, v25
	v_lshlrev_b32_e32 v34, 6, v98
	v_mov_b32_e32 v35, v25
	v_subbrev_co_u32_e32 v19, vcc, 0, v19, vcc
	s_addc_u32 s23, s75, s23
	s_lshl_b64 s[54:55], s[52:53], 8
	v_mov_b32_e32 v31, v25
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[28:29]
	v_sub_co_u32_e32 v28, vcc, v41, v28
	v_lshl_add_u64 v[40:41], v[34:35], 0, v[32:33]
	v_mov_b64_e32 v[34:35], s[54:55]
	v_lshl_add_u64 v[20:21], v[30:31], 0, v[20:21]
	v_lshl_add_u64 v[30:31], s[22:23], 0, v[24:25]
	v_mad_u64_u32 v[34:35], s[22:23], s47, v52, v[34:35]
	v_ashrrev_i32_e32 v37, 31, v36
	s_lshr_b64 s[22:23], s[28:29], 24
	v_add3_u32 v42, v16, v24, v17
	v_lshl_add_u64 v[32:33], v[40:41], 0, v[36:37]
	v_mul_lo_u32 v24, s47, v45
	v_mul_lo_u32 v45, s22, v52
	v_mad_u64_u32 v[40:41], s[22:23], s47, v52, v[40:41]
	v_add3_u32 v35, v45, v35, v24
	v_add3_u32 v41, v45, v41, v24
	v_lshl_add_u32 v24, s14, 7, v26
	v_bitop3_b32 v16, v99, 7, v96 bitop3:0x48
	v_add_u32_e32 v26, 0x60, v24
	v_lshlrev_b32_e32 v43, 4, v16
	v_mul_lo_u32 v26, v26, s44
	s_lshl_b32 s14, s52, 7
	v_add3_u32 v52, v26, s14, v43
	v_add_u32_e32 v26, 64, v24
	v_or_b32_e32 v32, v32, v38
	v_mul_lo_u32 v26, v26, s44
	v_subbrev_co_u32_e32 v29, vcc, 0, v53, vcc
	v_lshl_add_u64 v[34:35], v[32:33], 0, v[34:35]
	v_add3_u32 v53, v26, s14, v43
	v_add_u32_e32 v26, 32, v24
	v_lshlrev_b32_e32 v44, 10, v49
	v_mov_b32_e32 v39, v25
	v_sub_co_u32_e32 v34, vcc, -3, v34
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[36:37]
	v_mul_lo_u32 v26, v26, s44
	v_mul_lo_u32 v24, v24, s44
	s_ashr_i32 s56, s45, 31
	s_ashr_i32 s76, s59, 31
	s_movk_i32 s80, 0xf800
	s_movk_i32 s84, 0xff00
	v_lshlrev_b32_e32 v16, 5, v16
	v_mov_b32_e32 v17, v25
	v_subb_co_u32_e32 v35, vcc, -1, v35, vcc
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[38:39]
	v_add3_u32 v54, v26, s14, v43
	v_add3_u32 v55, v24, s14, v43
	v_bfrev_b32_e32 v56, -2
	s_add_i32 s47, s71, 0x8000
	s_add_i32 s71, s72, 0x8000
	s_add_i32 s72, s74, 0x8000
	s_add_i32 s92, s92, 0x8000
	v_add_u32_e32 v57, v27, v44
	v_add_u32_e32 v58, v46, v42
	v_add_u32_e32 v59, v47, v42
	s_mov_b32 s57, s56
	s_mov_b64 s[74:75], 0x400
	s_mov_b32 s78, s76
	s_mov_b32 s79, s76
	s_mov_b32 s81, -1
	s_mov_b64 s[82:83], 0x100
	s_mov_b32 s85, -1
	s_branch .LBB0_112
	.p2align	5
.LBB0_111:
	s_or_b64 exec, exec, s[14:15]
	s_add_u32 s14, s58, s76
	s_addc_u32 s15, s59, s76
	s_xor_b64 s[22:23], s[14:15], s[78:79]
	v_cvt_f32_u32_e32 v24, s22
	v_cvt_f32_u32_e32 v45, s23
	v_ashrrev_i32_e32 v94, 31, v41
	v_xor_b32_e32 v95, v44, v94
	s_sub_u32 s14, 0, s22
	v_fmac_f32_e32 v24, 0x4f800000, v45
	v_rcp_f32_e32 v24, v24
	s_subb_u32 s15, 0, s23
	v_xor_b32_e32 v41, v94, v41
	v_xor_b32_e32 v40, v94, v40
	v_mul_f32_e32 v24, 0x5f7ffffc, v24
	v_mul_f32_e32 v44, 0x2f800000, v24
	v_trunc_f32_e32 v44, v44
	v_fmac_f32_e32 v24, 0xcf800000, v44
	v_cvt_u32_f32_e32 v24, v24
	v_cvt_u32_f32_e32 v44, v44
	v_ashrrev_i32_e32 v39, 31, v39
	v_and_b32_e32 v39, s58, v39
	v_readfirstlane_b32 s28, v24
	v_readfirstlane_b32 s86, v44
	s_mul_hi_u32 s77, s14, s28
	s_mul_i32 s87, s14, s86
	s_mul_i32 s29, s15, s28
	s_add_i32 s77, s77, s87
	s_add_i32 s77, s77, s29
	s_mul_i32 s93, s14, s28
	s_mul_i32 s87, s28, s77
	s_mul_hi_u32 s94, s28, s93
	s_mul_hi_u32 s29, s28, s77
	s_add_u32 s87, s94, s87
	s_addc_u32 s29, 0, s29
	s_mul_hi_u32 s95, s86, s93
	s_mul_i32 s93, s86, s93
	s_add_u32 s87, s87, s93
	s_mul_hi_u32 s94, s86, s77
	s_addc_u32 s29, s29, s95
	s_addc_u32 s87, s94, 0
	s_mul_i32 s77, s86, s77
	s_add_u32 s29, s29, s77
	s_addc_u32 s77, 0, s87
	s_add_u32 s28, s28, s29
	s_addc_u32 s29, s86, s77
	s_mul_i32 s77, s14, s29
	s_mul_hi_u32 s86, s14, s28
	s_add_i32 s77, s86, s77
	s_mul_i32 s15, s15, s28
	s_add_i32 s77, s77, s15
	s_mul_i32 s14, s14, s28
	s_mul_hi_u32 s86, s29, s14
	s_mul_i32 s87, s29, s14
	s_mul_i32 s94, s28, s77
	s_mul_hi_u32 s14, s28, s14
	s_mul_hi_u32 s93, s28, s77
	s_add_u32 s14, s14, s94
	s_addc_u32 s93, 0, s93
	s_add_u32 s14, s14, s87
	s_mul_hi_u32 s15, s29, s77
	s_addc_u32 s14, s93, s86
	s_addc_u32 s15, s15, 0
	s_mul_i32 s77, s29, s77
	s_add_u32 s14, s14, s77
	v_ashrrev_i32_e32 v44, 31, v41
	s_addc_u32 s15, 0, s15
	v_mov_b32_e32 v45, v44
	s_add_u32 s28, s28, s14
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[44:45]
	s_addc_u32 s29, s29, s15
	v_xor_b32_e32 v99, v40, v44
	v_xor_b32_e32 v45, v41, v44
	v_mad_u64_u32 v[40:41], s[14:15], v99, s29, 0
	v_mul_hi_u32 v24, v99, s28
	v_lshl_add_u64 v[40:41], v[24:25], 0, v[40:41]
	v_mad_u64_u32 v[92:93], s[14:15], v45, s28, 0
	v_add_co_u32_e32 v24, vcc, v40, v92
	v_mad_u64_u32 v[46:47], s[14:15], v45, s29, 0
	s_nop 0
	v_addc_co_u32_e32 v40, vcc, v41, v93, vcc
	v_mov_b32_e32 v41, s73
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[46:47]
	v_mul_lo_u32 v24, s23, v40
	v_mul_lo_u32 v41, s22, v41
	v_mad_u64_u32 v[46:47], s[14:15], s22, v40, 0
	v_add3_u32 v24, v47, v41, v24
	v_sub_u32_e32 v41, v45, v24
	v_mov_b32_e32 v47, s23
	v_sub_co_u32_e32 v46, vcc, v99, v46
	s_add_u32 s52, s52, 1
	s_nop 0
	v_subb_co_u32_e64 v41, s[14:15], v41, v47, vcc
	v_subrev_co_u32_e64 v47, s[14:15], s22, v46
	v_subb_co_u32_e32 v24, vcc, v45, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v41, s[14:15], 0, v41, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s23, v41
	v_cmp_le_u32_e32 vcc, s23, v24
	s_addc_u32 s53, s53, 0
	v_cndmask_b32_e64 v92, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s22, v47
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s22, v46
	v_cndmask_b32_e64 v47, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s23, v41
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s23, v24
	v_cndmask_b32_e64 v41, v92, v47, s[14:15]
	v_add_u32_e32 v47, 2, v40
	v_add_u32_e32 v92, 1, v40
	v_cmp_ne_u32_e64 s[14:15], 0, v41
	v_cndmask_b32_e32 v24, v45, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v41, v92, v47, s[14:15]
	s_waitcnt vmcnt(12)
	v_lshlrev_b16_e32 v47, 8, v78
	v_cndmask_b32_e32 v24, v40, v41, vcc
	v_xor_b32_e32 v40, s76, v44
	v_xor_b32_e32 v24, v24, v40
	v_sub_u32_e32 v24, v24, v40
	v_xor_b32_e32 v24, v24, v94
	v_mul_lo_u32 v24, v24, s66
	v_add3_u32 v24, v24, v38, v39
	buffer_load_ubyte v24, v24, s[8:11], 0 offen
	v_ashrrev_i32_e32 v39, 31, v43
	v_mul_lo_u32 v38, v95, s66
	v_and_b32_e32 v39, s58, v39
	v_add3_u32 v38, v38, v42, v39
	buffer_load_ubyte v46, v38, s[8:11], 0 offen
	v_lshlrev_b16_e32 v38, 8, v74
	v_lshlrev_b16_e32 v39, 8, v75
	v_or_b32_e32 v38, v72, v38
	v_or_b32_sdwa v39, v73, v39 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v42, 8, v63
	v_or_b32_sdwa v41, v38, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v38, 8, v70
	v_lshlrev_b16_e32 v39, 8, v71
	v_or_b32_e32 v38, v68, v38
	v_or_b32_sdwa v39, v69, v39 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v42, v61, v42 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v40, v38, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v38, 8, v66
	v_lshlrev_b16_e32 v39, 8, v67
	v_or_b32_e32 v38, v64, v38
	v_or_b32_sdwa v39, v65, v39 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt vmcnt(2)
	v_lshlrev_b16_e32 v43, 8, v91
	v_or_b32_sdwa v39, v38, v39 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v38, 8, v62
	v_or_b32_e32 v38, v60, v38
	v_or_b32_sdwa v38, v38, v42 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v42, 8, v90
	v_or_b32_e32 v42, v88, v42
	v_or_b32_sdwa v43, v89, v43 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt vmcnt(1) lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[128:131], v[38:41], v[12:15], v26, v24 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_or_b32_sdwa v45, v42, v43 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v42, 8, v86
	v_lshlrev_b16_e32 v43, 8, v87
	v_or_b32_e32 v42, v84, v42
	v_or_b32_sdwa v43, v85, v43 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[100:103], v[38:41], v[8:11], v26, v24 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_or_b32_sdwa v44, v42, v43 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v42, 8, v82
	v_lshlrev_b16_e32 v43, 8, v83
	v_or_b32_e32 v42, v80, v42
	v_or_b32_sdwa v43, v81, v43 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[112:115], v[38:41], v[4:7], v27, v24 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_or_b32_sdwa v43, v42, v43 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v42, 8, v79
	v_or_b32_e32 v42, v77, v42
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[104:107], v[38:41], v[0:3], v27, v24 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_or_b32_sdwa v47, v76, v47 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_mov_b64 s[14:15], 0x800
	v_or_b32_sdwa v42, v42, v47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshl_add_u64 v[22:23], v[22:23], 0, s[14:15]
	s_mov_b64 s[14:15], 0x2000
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[120:123], v[42:45], v[12:15], v26, v46 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshl_add_u64 v[18:19], v[18:19], 0, s[80:81]
	v_lshl_add_u64 v[28:29], v[28:29], 0, s[80:81]
	v_lshl_add_u64 v[30:31], v[30:31], 0, s[14:15]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[108:111], v[42:45], v[8:11], v26, v46 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshl_add_u64 v[32:33], v[32:33], 0, s[82:83]
	v_lshl_add_u64 v[34:35], v[34:35], 0, s[84:85]
	v_lshl_add_u64 v[36:37], v[36:37], 0, s[82:83]
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[124:127], v[42:45], v[4:7], v27, v46 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshl_add_u64 v[16:17], v[16:17], 0, s[82:83]
	v_add_u32_e32 v52, 0x80, v52
	v_add_u32_e32 v53, 0x80, v53
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[116:119], v[42:45], v[0:3], v27, v46 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mov_b64_e32 v[26:27], s[34:35]
	v_cmp_lt_i64_e32 vcc, s[52:53], v[26:27]
	v_add_u32_e32 v54, 0x80, v54
	v_add_u32_e32 v55, 0x80, v55
	s_cbranch_vccz .LBB0_128
.LBB0_112:
	v_lshl_add_u64 v[26:27], s[54:55], 0, v[16:17]
	v_cmp_gt_i64_e32 vcc, s[30:31], v[26:27]
	s_and_b64 s[14:15], s[2:3], vcc
	s_mov_b32 m0, s47
	v_cndmask_b32_e64 v24, v56, v55, s[14:15]
	s_and_b64 s[14:15], s[26:27], vcc
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v24, s[40:43], 0 offen lds
	v_cndmask_b32_e64 v24, v56, v54, s[14:15]
	s_mov_b32 m0, s71
	s_and_b64 s[14:15], s[18:19], vcc
	buffer_load_dwordx4 v24, s[40:43], 0 offen lds
	v_cndmask_b32_e64 v24, v56, v53, s[14:15]
	s_mov_b32 m0, s72
	s_and_b64 vcc, s[20:21], vcc
	buffer_load_dwordx4 v24, s[40:43], 0 offen lds
	v_cndmask_b32_e32 v24, v56, v52, vcc
	s_mov_b32 m0, s92
	v_ashrrev_i32_e32 v26, 31, v31
	buffer_load_dwordx4 v24, s[40:43], 0 offen lds
	v_or_b32_e32 v24, s31, v31
	v_cmp_ne_u32_e32 vcc, 0, v24
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[28:29], exec, s[14:15]
	s_cbranch_execz .LBB0_114
	s_ashr_i32 s14, s31, 31
	s_add_u32 s22, s30, s14
	s_mov_b32 s15, s14
	s_addc_u32 s23, s31, s14
	s_xor_b64 s[86:87], s[22:23], s[14:15]
	v_cvt_f32_u32_e32 v24, s86
	v_cvt_f32_u32_e32 v27, s87
	s_sub_u32 s14, 0, s86
	s_subb_u32 s15, 0, s87
	v_fmac_f32_e32 v24, 0x4f800000, v27
	v_rcp_f32_e32 v24, v24
	s_nop 0
	v_mul_f32_e32 v24, 0x5f7ffffc, v24
	v_mul_f32_e32 v27, 0x2f800000, v24
	v_trunc_f32_e32 v27, v27
	v_fmac_f32_e32 v24, 0xcf800000, v27
	v_cvt_u32_f32_e32 v27, v27
	v_cvt_u32_f32_e32 v24, v24
	v_readfirstlane_b32 s22, v27
	v_readfirstlane_b32 s23, v24
	s_mul_i32 s77, s14, s22
	s_mul_hi_u32 s94, s14, s23
	s_mul_i32 s93, s15, s23
	s_add_i32 s77, s94, s77
	s_add_i32 s77, s77, s93
	s_mul_i32 s95, s14, s23
	s_mul_i32 s94, s23, s77
	s_mul_hi_u32 s96, s23, s95
	s_mul_hi_u32 s93, s23, s77
	s_add_u32 s94, s96, s94
	s_addc_u32 s93, 0, s93
	s_mul_hi_u32 s97, s22, s95
	s_mul_i32 s95, s22, s95
	s_add_u32 s94, s94, s95
	s_mul_hi_u32 s96, s22, s77
	s_addc_u32 s93, s93, s97
	s_addc_u32 s94, s96, 0
	s_mul_i32 s77, s22, s77
	s_add_u32 s77, s93, s77
	s_addc_u32 s93, 0, s94
	s_add_u32 s23, s23, s77
	s_addc_u32 s22, s22, s93
	s_mul_i32 s77, s14, s22
	s_mul_hi_u32 s93, s14, s23
	s_add_i32 s77, s93, s77
	s_mul_i32 s15, s15, s23
	s_add_i32 s77, s77, s15
	s_mul_i32 s14, s14, s23
	s_mul_hi_u32 s93, s22, s14
	s_mul_i32 s94, s22, s14
	s_mul_i32 s96, s23, s77
	s_mul_hi_u32 s14, s23, s14
	s_mul_hi_u32 s95, s23, s77
	s_add_u32 s14, s14, s96
	s_addc_u32 s95, 0, s95
	s_add_u32 s14, s14, s94
	s_mul_hi_u32 s15, s22, s77
	s_addc_u32 s14, s95, s93
	s_addc_u32 s15, s15, 0
	s_mul_i32 s77, s22, s77
	s_add_u32 s14, s14, s77
	s_addc_u32 s15, 0, s15
	v_mov_b32_e32 v27, v26
	s_add_u32 s23, s23, s14
	v_lshl_add_u64 v[38:39], v[30:31], 0, v[26:27]
	s_addc_u32 s22, s22, s15
	v_xor_b32_e32 v44, v38, v26
	v_xor_b32_e32 v27, v39, v26
	v_mad_u64_u32 v[38:39], s[14:15], v44, s22, 0
	v_mul_hi_u32 v24, v44, s23
	v_lshl_add_u64 v[38:39], v[24:25], 0, v[38:39]
	v_mad_u64_u32 v[42:43], s[14:15], v27, s23, 0
	v_add_co_u32_e32 v24, vcc, v38, v42
	v_mad_u64_u32 v[40:41], s[14:15], v27, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v43, vcc
	v_mov_b32_e32 v39, s73
	s_nop 0
	v_addc_co_u32_e32 v41, vcc, 0, v41, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[40:41]
	v_mul_lo_u32 v24, s87, v38
	v_mul_lo_u32 v40, s86, v39
	v_mad_u64_u32 v[38:39], s[14:15], s86, v38, 0
	v_add3_u32 v24, v39, v40, v24
	v_sub_u32_e32 v39, v27, v24
	v_mov_b32_e32 v40, s87
	v_sub_co_u32_e32 v38, vcc, v44, v38
	s_nop 1
	v_subb_co_u32_e64 v39, s[14:15], v39, v40, vcc
	v_subrev_co_u32_e64 v41, s[14:15], s86, v38
	v_subb_co_u32_e32 v24, vcc, v27, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v42, s[22:23], 0, v39, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s87, v42
	v_subb_co_u32_e64 v39, s[14:15], v39, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v43, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s86, v41
	v_subrev_co_u32_e64 v40, s[14:15], s86, v41
	s_nop 0
	v_cndmask_b32_e64 v44, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s87, v42
	v_subbrev_co_u32_e64 v39, s[14:15], 0, v39, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v43, v43, v44, s[22:23]
	v_cmp_le_u32_e32 vcc, s87, v24
	v_cmp_ne_u32_e64 s[14:15], 0, v43
	s_nop 0
	v_cndmask_b32_e64 v27, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v38
	v_cndmask_b32_e64 v39, v42, v39, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v42, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v27, v27, v42, vcc
	v_cmp_ne_u32_e32 vcc, 0, v27
	v_cndmask_b32_e64 v27, v41, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v27, v38, v27, vcc
	v_cndmask_b32_e32 v24, v24, v39, vcc
	v_xor_b32_e32 v27, v27, v26
	v_xor_b32_e32 v24, v24, v26
	v_sub_co_u32_e32 v38, vcc, v27, v26
	s_nop 1
	v_subb_co_u32_e32 v39, vcc, v24, v26, vcc
.LBB0_114:
	s_andn2_saveexec_b64 s[14:15], s[28:29]
	s_cbranch_execz .LBB0_116
	v_cvt_f32_u32_e32 v24, s30
	s_sub_i32 s22, 0, s30
	v_rcp_iflag_f32_e32 v24, v24
	s_nop 0
	v_mul_f32_e32 v24, 0x4f7ffffe, v24
	v_cvt_u32_f32_e32 v24, v24
	v_mul_lo_u32 v27, s22, v24
	v_mul_hi_u32 v27, v24, v27
	v_add_u32_e32 v24, v24, v27
	v_mul_hi_u32 v24, v30, v24
	v_mul_lo_u32 v24, v24, s30
	v_sub_u32_e32 v24, v30, v24
	v_subrev_u32_e32 v27, s30, v24
	v_cmp_le_u32_e32 vcc, s30, v24
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v27, vcc
	v_subrev_u32_e32 v27, s30, v24
	v_cmp_le_u32_e32 vcc, s30, v24
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v27, vcc
	v_mov_b64_e32 v[38:39], v[24:25]
.LBB0_116:
	s_or_b64 exec, exec, s[14:15]
	s_ashr_i32 s22, s31, 31
	s_add_u32 s14, s30, s22
	s_mov_b32 s23, s22
	s_addc_u32 s15, s31, s22
	s_xor_b64 s[28:29], s[14:15], s[22:23]
	v_cvt_f32_u32_e32 v24, s28
	v_cvt_f32_u32_e32 v27, s29
	s_sub_u32 s14, 0, s28
	s_subb_u32 s15, 0, s29
	v_xor_b32_e32 v41, v26, v31
	v_fmac_f32_e32 v24, 0x4f800000, v27
	v_rcp_f32_e32 v24, v24
	v_ashrrev_i32_e32 v42, 31, v41
	v_xor_b32_e32 v40, v26, v30
	v_mov_b32_e32 v43, v42
	v_mul_f32_e32 v24, 0x5f7ffffc, v24
	v_mul_f32_e32 v27, 0x2f800000, v24
	v_trunc_f32_e32 v27, v27
	v_fmac_f32_e32 v24, 0xcf800000, v27
	v_cvt_u32_f32_e32 v27, v27
	v_cvt_u32_f32_e32 v24, v24
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[42:43]
	v_xor_b32_e32 v43, v40, v42
	v_readfirstlane_b32 s23, v27
	v_readfirstlane_b32 s77, v24
	s_mul_hi_u32 s87, s14, s77
	s_mul_i32 s93, s14, s23
	s_mul_i32 s86, s15, s77
	s_add_i32 s87, s87, s93
	s_add_i32 s87, s87, s86
	s_mul_i32 s94, s14, s77
	s_mul_i32 s93, s77, s87
	s_mul_hi_u32 s95, s77, s94
	s_mul_hi_u32 s86, s77, s87
	s_add_u32 s93, s95, s93
	s_addc_u32 s86, 0, s86
	s_mul_hi_u32 s96, s23, s94
	s_mul_i32 s94, s23, s94
	s_add_u32 s93, s93, s94
	s_mul_hi_u32 s95, s23, s87
	s_addc_u32 s86, s86, s96
	s_addc_u32 s93, s95, 0
	s_mul_i32 s87, s23, s87
	s_add_u32 s86, s86, s87
	s_addc_u32 s87, 0, s93
	s_add_u32 s77, s77, s86
	s_addc_u32 s23, s23, s87
	s_mul_i32 s86, s14, s23
	s_mul_hi_u32 s87, s14, s77
	s_add_i32 s86, s87, s86
	s_mul_i32 s15, s15, s77
	s_add_i32 s86, s86, s15
	s_mul_i32 s14, s14, s77
	s_mul_hi_u32 s87, s23, s14
	s_mul_i32 s93, s23, s14
	s_mul_i32 s95, s77, s86
	s_mul_hi_u32 s14, s77, s14
	s_mul_hi_u32 s94, s77, s86
	s_add_u32 s14, s14, s95
	s_addc_u32 s94, 0, s94
	s_add_u32 s14, s14, s93
	s_mul_hi_u32 s15, s23, s86
	s_addc_u32 s14, s94, s87
	s_addc_u32 s15, s15, 0
	s_mul_i32 s86, s23, s86
	s_add_u32 s14, s14, s86
	s_addc_u32 s15, 0, s15
	s_add_u32 s77, s77, s14
	s_addc_u32 s23, s23, s15
	v_xor_b32_e32 v27, v41, v42
	v_mad_u64_u32 v[40:41], s[14:15], v43, s23, 0
	v_mul_hi_u32 v24, v43, s77
	v_lshl_add_u64 v[40:41], v[24:25], 0, v[40:41]
	v_mad_u64_u32 v[46:47], s[14:15], v27, s77, 0
	v_add_co_u32_e32 v24, vcc, v40, v46
	v_mad_u64_u32 v[44:45], s[14:15], v27, s23, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v41, v47, vcc
	s_add_i32 m0, s67, 0xc800
	s_nop 0
	v_addc_co_u32_e32 v45, vcc, 0, v45, vcc
	v_lshl_add_u64 v[40:41], v[24:25], 0, v[44:45]
	v_mul_lo_u32 v24, s29, v40
	v_mul_lo_u32 v41, s28, v41
	v_mad_u64_u32 v[44:45], s[14:15], s28, v40, 0
	v_add3_u32 v24, v45, v41, v24
	v_sub_u32_e32 v41, v27, v24
	v_mov_b32_e32 v45, s29
	v_sub_co_u32_e32 v43, vcc, v43, v44
	s_nop 1
	v_subb_co_u32_e64 v41, s[14:15], v41, v45, vcc
	v_subrev_co_u32_e64 v44, s[14:15], s28, v43
	v_subb_co_u32_e32 v24, vcc, v27, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v41, s[14:15], 0, v41, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s29, v41
	v_cmp_le_u32_e32 vcc, s29, v24
	s_nop 0
	v_cndmask_b32_e64 v45, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v44
	v_cndmask_b32_e64 v27, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v43
	v_cndmask_b32_e64 v44, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v41
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v24
	v_cndmask_b32_e64 v41, v45, v44, s[14:15]
	v_add_u32_e32 v44, 2, v40
	v_add_u32_e32 v45, 1, v40
	v_cmp_ne_u32_e64 s[14:15], 0, v41
	v_cndmask_b32_e32 v24, v27, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v41, v45, v44, s[14:15]
	v_xor_b32_e32 v27, s22, v42
	v_cndmask_b32_e32 v24, v40, v41, vcc
	v_xor_b32_e32 v24, v24, v27
	v_sub_u32_e32 v24, v24, v27
	v_xor_b32_e32 v40, v24, v26
	v_ashrrev_i32_e32 v24, 31, v39
	v_and_b32_e32 v27, s31, v24
	v_and_b32_e32 v26, s30, v24
	v_lshl_add_u64 v[26:27], v[26:27], 0, v[38:39]
	v_ashrrev_i32_e32 v38, 31, v27
	v_xor_b32_e32 v27, v38, v27
	v_ashrrev_i32_e32 v24, 31, v27
	v_xor_b32_e32 v26, v38, v26
	v_lshrrev_b32_e32 v24, 27, v24
	v_lshl_add_u64 v[26:27], v[26:27], 0, v[24:25]
	s_add_u32 s14, s44, s56
	v_alignbit_b32 v24, v27, v26, 5
	v_mul_lo_u32 v26, v40, s46
	s_addc_u32 s15, s45, s56
	v_xad_u32 v24, v24, v38, v26
	s_xor_b64 s[86:87], s[14:15], s[56:57]
	buffer_load_dword v24, s[4:7], 0 offen lds
	v_cvt_f32_u32_e32 v24, s86
	v_cvt_f32_u32_e32 v40, s87
	v_ashrrev_i32_e32 v26, 31, v23
	v_xor_b32_e32 v39, v26, v23
	v_ashrrev_i32_e32 v42, 31, v39
	v_fmac_f32_e32 v24, 0x4f800000, v40
	v_rcp_f32_e32 v24, v24
	v_xor_b32_e32 v38, v26, v22
	v_mov_b32_e32 v43, v42
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[42:43]
	v_mul_f32_e32 v24, 0x5f7ffffc, v24
	v_xor_b32_e32 v46, v38, v42
	v_mul_f32_e32 v38, 0x2f800000, v24
	v_trunc_f32_e32 v38, v38
	v_fmac_f32_e32 v24, 0xcf800000, v38
	v_cvt_u32_f32_e32 v24, v24
	v_cvt_u32_f32_e32 v38, v38
	s_sub_u32 s14, 0, s86
	s_subb_u32 s15, 0, s87
	v_readfirstlane_b32 s22, v24
	v_readfirstlane_b32 s29, v38
	s_mul_hi_u32 s28, s14, s22
	s_mul_i32 s77, s14, s29
	s_mul_i32 s23, s15, s22
	s_add_i32 s28, s28, s77
	s_add_i32 s28, s28, s23
	s_mul_i32 s93, s14, s22
	s_mul_i32 s77, s22, s28
	s_mul_hi_u32 s94, s22, s93
	s_mul_hi_u32 s23, s22, s28
	s_add_u32 s77, s94, s77
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 s95, s29, s93
	s_mul_i32 s93, s29, s93
	s_add_u32 s77, s77, s93
	s_mul_hi_u32 s94, s29, s28
	s_addc_u32 s23, s23, s95
	s_addc_u32 s77, s94, 0
	s_mul_i32 s28, s29, s28
	s_add_u32 s23, s23, s28
	s_addc_u32 s28, 0, s77
	s_add_u32 s22, s22, s23
	s_addc_u32 s23, s29, s28
	s_mul_i32 s28, s14, s23
	s_mul_hi_u32 s29, s14, s22
	s_add_i32 s28, s29, s28
	s_mul_i32 s15, s15, s22
	s_add_i32 s28, s28, s15
	s_mul_i32 s14, s14, s22
	s_mul_hi_u32 s29, s23, s14
	s_mul_i32 s77, s23, s14
	s_mul_i32 s94, s22, s28
	s_mul_hi_u32 s14, s22, s14
	s_mul_hi_u32 s93, s22, s28
	s_add_u32 s14, s14, s94
	s_addc_u32 s93, 0, s93
	s_add_u32 s14, s14, s77
	s_mul_hi_u32 s15, s23, s28
	s_addc_u32 s14, s93, s29
	s_addc_u32 s15, s15, 0
	s_mul_i32 s28, s23, s28
	s_add_u32 s14, s14, s28
	s_addc_u32 s15, 0, s15
	s_add_u32 s77, s22, s14
	s_addc_u32 s93, s23, s15
	v_xor_b32_e32 v43, v39, v42
	v_mad_u64_u32 v[38:39], s[14:15], v46, s93, 0
	v_mul_hi_u32 v24, v46, s77
	v_lshl_add_u64 v[38:39], v[24:25], 0, v[38:39]
	v_mad_u64_u32 v[44:45], s[14:15], v43, s77, 0
	v_add_co_u32_e32 v24, vcc, v38, v44
	v_mad_u64_u32 v[40:41], s[14:15], v43, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v39, v45, vcc
	v_mov_b32_e32 v27, v26
	s_nop 0
	v_addc_co_u32_e32 v41, vcc, 0, v41, vcc
	v_lshl_add_u64 v[38:39], v[24:25], 0, v[40:41]
	v_mul_lo_u32 v24, s87, v38
	v_mul_lo_u32 v39, s86, v39
	v_mad_u64_u32 v[44:45], s[14:15], s86, v38, 0
	v_add3_u32 v24, v45, v39, v24
	v_sub_u32_e32 v39, v43, v24
	v_mov_b32_e32 v40, s87
	v_sub_co_u32_e32 v41, vcc, v46, v44
	s_waitcnt vmcnt(0)
	s_nop 0
	v_subb_co_u32_e64 v39, s[14:15], v39, v40, vcc
	v_subrev_co_u32_e64 v44, s[14:15], s86, v41
	v_subb_co_u32_e32 v24, vcc, v43, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v39, s[14:15], 0, v39, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v39
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v45, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v44
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v41
	v_cndmask_b32_e64 v44, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v39
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v39, v45, v44, s[14:15]
	v_add_u32_e32 v44, 2, v38
	v_add_u32_e32 v45, 1, v38
	v_cmp_ne_u32_e64 s[14:15], 0, v39
	v_cndmask_b32_e32 v24, v43, v41, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v39, v45, v44, s[14:15]
	s_barrier
	v_cndmask_b32_e32 v24, v38, v39, vcc
	v_xor_b32_e32 v38, s56, v42
	v_xor_b32_e32 v24, v24, v38
	v_sub_u32_e32 v24, v24, v38
	v_lshl_add_u64 v[38:39], v[22:23], 0, v[26:27]
	v_xor_b32_e32 v46, v38, v26
	v_xad_u32 v41, v24, v26, v51
	v_xor_b32_e32 v27, v39, v26
	v_mad_u64_u32 v[38:39], s[14:15], v46, s93, 0
	v_mul_hi_u32 v24, v46, s77
	v_lshl_add_u64 v[38:39], v[24:25], 0, v[38:39]
	v_mad_u64_u32 v[44:45], s[14:15], v27, s77, 0
	v_add_co_u32_e32 v24, vcc, v38, v44
	v_mad_u64_u32 v[42:43], s[14:15], v27, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v39, v45, vcc
	s_nop 1
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[38:39], v[24:25], 0, v[42:43]
	v_mul_lo_u32 v24, s87, v38
	v_mul_lo_u32 v42, s86, v39
	v_mad_u64_u32 v[38:39], s[14:15], s86, v38, 0
	v_add3_u32 v24, v39, v42, v24
	v_sub_u32_e32 v39, v27, v24
	v_sub_co_u32_e32 v38, vcc, v46, v38
	s_nop 1
	v_subb_co_u32_e64 v39, s[14:15], v39, v40, vcc
	v_subrev_co_u32_e64 v42, s[14:15], s86, v38
	v_subb_co_u32_e32 v24, vcc, v27, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[22:23], 0, v39, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s87, v43
	v_subb_co_u32_e64 v39, s[14:15], v39, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v44, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s86, v42
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v45, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s87, v43
	v_cndmask_b32_e64 v27, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v38
	v_cndmask_b32_e64 v44, v44, v45, s[22:23]
	v_subrev_co_u32_e64 v45, s[14:15], s86, v42
	v_cmp_gt_i32_e64 s[22:23], 0, v23
	s_nop 0
	v_subbrev_co_u32_e64 v39, s[14:15], 0, v39, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v44
	s_nop 1
	v_cndmask_b32_e64 v39, v43, v39, s[14:15]
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v27, v27, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v27
	v_cndmask_b32_e64 v27, v42, v45, s[14:15]
	v_lshl_add_u64 v[42:43], v[22:23], 0, 1
	v_cndmask_b32_e32 v27, v38, v27, vcc
	v_cndmask_b32_e32 v24, v24, v39, vcc
	v_xor_b32_e32 v27, v27, v26
	v_xor_b32_e32 v24, v24, v26
	v_sub_co_u32_e32 v27, vcc, v27, v26
	v_mul_lo_u32 v38, v41, s33
	s_nop 0
	v_subb_co_u32_e32 v24, vcc, v24, v26, vcc
	v_ashrrev_i32_e32 v24, 31, v24
	v_and_b32_e32 v24, s44, v24
	v_add3_u32 v27, v24, v27, v38
	v_lshl_add_u64 v[38:39], s[60:61], 0, v[28:29]
	v_lshl_add_u64 v[44:45], v[38:39], 0, -2
	v_cndmask_b32_e64 v45, v43, v45, s[22:23]
	v_ashrrev_i32_e32 v46, 31, v45
	v_cndmask_b32_e64 v44, v42, v44, s[22:23]
	v_mov_b32_e32 v47, v46
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_xor_b32_e32 v47, v44, v46
	v_xor_b32_e32 v41, v45, v46
	v_mad_u64_u32 v[44:45], s[14:15], v47, s93, 0
	v_mul_hi_u32 v24, v47, s77
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[44:45]
	v_mad_u64_u32 v[62:63], s[14:15], v41, s77, 0
	v_add_co_u32_e32 v24, vcc, v44, v62
	v_mad_u64_u32 v[60:61], s[14:15], v41, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v45, v63, vcc
	s_nop 1
	v_addc_co_u32_e32 v61, vcc, 0, v61, vcc
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[60:61]
	v_mul_lo_u32 v24, s87, v44
	v_mul_lo_u32 v45, s86, v45
	v_mad_u64_u32 v[60:61], s[14:15], s86, v44, 0
	v_add3_u32 v24, v61, v45, v24
	v_sub_u32_e32 v45, v41, v24
	v_sub_co_u32_e32 v47, vcc, v47, v60
	s_nop 1
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, vcc
	v_subrev_co_u32_e64 v60, s[14:15], s86, v47
	v_subb_co_u32_e32 v24, vcc, v41, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v45, s[14:15], 0, v45, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v45
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v60
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v47
	v_cndmask_b32_e64 v60, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v45
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v45, v61, v60, s[14:15]
	v_add_u32_e32 v60, 2, v44
	v_add_u32_e32 v61, 1, v44
	v_cmp_ne_u32_e64 s[14:15], 0, v45
	v_cndmask_b32_e32 v24, v41, v47, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v45, v61, v60, s[14:15]
	v_xor_b32_e32 v41, s56, v46
	v_cndmask_b32_e32 v24, v44, v45, vcc
	v_ashrrev_i32_e32 v44, 31, v43
	v_mov_b32_e32 v45, v44
	v_xor_b32_e32 v24, v24, v41
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[44:45]
	v_sub_u32_e32 v24, v24, v41
	v_xor_b32_e32 v45, v42, v44
	v_xad_u32 v62, v24, v26, v51
	v_xor_b32_e32 v41, v43, v44
	v_mad_u64_u32 v[42:43], s[14:15], v45, s93, 0
	v_mul_hi_u32 v24, v45, s77
	v_lshl_add_u64 v[42:43], v[24:25], 0, v[42:43]
	v_mad_u64_u32 v[60:61], s[14:15], v41, s77, 0
	v_add_co_u32_e32 v24, vcc, v42, v60
	v_mad_u64_u32 v[46:47], s[14:15], v41, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v43, v61, vcc
	s_nop 1
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[24:25], 0, v[46:47]
	v_mul_lo_u32 v24, s87, v42
	v_mul_lo_u32 v46, s86, v43
	v_mad_u64_u32 v[42:43], s[14:15], s86, v42, 0
	v_add3_u32 v24, v43, v46, v24
	v_sub_u32_e32 v43, v41, v24
	v_sub_co_u32_e32 v42, vcc, v45, v42
	s_nop 1
	v_subb_co_u32_e64 v43, s[14:15], v43, v40, vcc
	v_subrev_co_u32_e64 v45, s[14:15], s86, v42
	v_subb_co_u32_e32 v24, vcc, v41, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v46, s[28:29], 0, v43, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v46
	v_subb_co_u32_e64 v43, s[14:15], v43, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v45
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v60, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v46
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v42
	v_cndmask_b32_e64 v47, v47, v60, s[28:29]
	v_subrev_co_u32_e64 v60, s[14:15], s86, v45
	s_nop 1
	v_subbrev_co_u32_e64 v43, s[14:15], 0, v43, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v47
	s_nop 1
	v_cndmask_b32_e64 v43, v46, v43, s[14:15]
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v41, v41, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v41
	v_cndmask_b32_e64 v41, v45, v60, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v24, v43, vcc
	v_cndmask_b32_e32 v41, v42, v41, vcc
	v_xor_b32_e32 v42, v24, v44
	v_xor_b32_e32 v24, v41, v44
	v_sub_co_u32_e32 v24, vcc, v24, v44
	s_nop 1
	v_subb_co_u32_e32 v41, vcc, v42, v44, vcc
	v_ashrrev_i32_e32 v41, 31, v41
	v_and_b32_e32 v41, s44, v41
	v_mul_lo_u32 v42, v62, s33
	v_add3_u32 v41, v41, v24, v42
	v_lshl_add_u64 v[42:43], v[22:23], 0, 2
	v_lshl_add_u64 v[44:45], v[38:39], 0, -3
	v_cndmask_b32_e64 v45, v43, v45, s[22:23]
	v_ashrrev_i32_e32 v46, 31, v45
	v_cndmask_b32_e64 v44, v42, v44, s[22:23]
	v_mov_b32_e32 v47, v46
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_xor_b32_e32 v64, v44, v46
	v_xor_b32_e32 v47, v45, v46
	v_mad_u64_u32 v[44:45], s[14:15], v64, s93, 0
	v_mul_hi_u32 v24, v64, s77
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[44:45]
	v_mad_u64_u32 v[62:63], s[14:15], v47, s77, 0
	v_add_co_u32_e32 v24, vcc, v44, v62
	v_mad_u64_u32 v[60:61], s[14:15], v47, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v45, v63, vcc
	s_nop 1
	v_addc_co_u32_e32 v61, vcc, 0, v61, vcc
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[60:61]
	v_mul_lo_u32 v24, s87, v44
	v_mul_lo_u32 v45, s86, v45
	v_mad_u64_u32 v[60:61], s[14:15], s86, v44, 0
	v_add3_u32 v24, v61, v45, v24
	v_sub_u32_e32 v45, v47, v24
	v_sub_co_u32_e32 v60, vcc, v64, v60
	s_nop 1
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, vcc
	v_subrev_co_u32_e64 v61, s[14:15], s86, v60
	v_subb_co_u32_e32 v24, vcc, v47, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v45, s[14:15], 0, v45, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v45
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v62, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v61
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v60
	v_cndmask_b32_e64 v61, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v45
	v_cndmask_b32_e64 v60, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v45, v62, v61, s[14:15]
	v_add_u32_e32 v61, 2, v44
	v_add_u32_e32 v62, 1, v44
	v_cmp_ne_u32_e64 s[14:15], 0, v45
	v_cndmask_b32_e32 v24, v47, v60, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v45, v62, v61, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v44, v45, vcc
	v_xor_b32_e32 v44, s56, v46
	v_xor_b32_e32 v24, v24, v44
	v_sub_u32_e32 v24, v24, v44
	v_ashrrev_i32_e32 v44, 31, v43
	v_mov_b32_e32 v45, v44
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[44:45]
	v_xor_b32_e32 v63, v42, v44
	v_xad_u32 v62, v24, v26, v51
	v_xor_b32_e32 v45, v43, v44
	v_mad_u64_u32 v[42:43], s[14:15], v63, s93, 0
	v_mul_hi_u32 v24, v63, s77
	v_lshl_add_u64 v[42:43], v[24:25], 0, v[42:43]
	v_mad_u64_u32 v[60:61], s[14:15], v45, s77, 0
	v_add_co_u32_e32 v24, vcc, v42, v60
	v_mad_u64_u32 v[46:47], s[14:15], v45, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v43, v61, vcc
	s_nop 1
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[24:25], 0, v[46:47]
	v_mul_lo_u32 v24, s87, v42
	v_mul_lo_u32 v46, s86, v43
	v_mad_u64_u32 v[42:43], s[14:15], s86, v42, 0
	v_add3_u32 v24, v43, v46, v24
	v_sub_u32_e32 v43, v45, v24
	v_sub_co_u32_e32 v42, vcc, v63, v42
	s_nop 1
	v_subb_co_u32_e64 v43, s[14:15], v43, v40, vcc
	v_subrev_co_u32_e64 v46, s[14:15], s86, v42
	v_subb_co_u32_e32 v24, vcc, v45, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[28:29], 0, v43, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v47
	v_subb_co_u32_e64 v43, s[14:15], v43, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v60, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v46
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v47
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v42
	v_cndmask_b32_e64 v60, v60, v61, s[28:29]
	v_subrev_co_u32_e64 v61, s[14:15], s86, v46
	s_nop 1
	v_subbrev_co_u32_e64 v43, s[14:15], 0, v43, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v60
	s_nop 1
	v_cndmask_b32_e64 v43, v47, v43, s[14:15]
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v45, v45, v47, vcc
	v_cmp_ne_u32_e32 vcc, 0, v45
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v43, vcc
	v_cndmask_b32_e64 v43, v46, v61, s[14:15]
	v_cndmask_b32_e32 v42, v42, v43, vcc
	v_xor_b32_e32 v43, v24, v44
	v_xor_b32_e32 v24, v42, v44
	v_sub_co_u32_e32 v24, vcc, v24, v44
	v_lshl_add_u64 v[46:47], v[38:39], 0, -4
	s_nop 0
	v_subb_co_u32_e32 v43, vcc, v43, v44, vcc
	v_lshl_add_u64 v[44:45], v[22:23], 0, 3
	v_cndmask_b32_e64 v47, v45, v47, s[22:23]
	v_ashrrev_i32_e32 v60, 31, v47
	v_cndmask_b32_e64 v46, v44, v46, s[22:23]
	v_mov_b32_e32 v61, v60
	v_ashrrev_i32_e32 v42, 31, v43
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[60:61]
	v_and_b32_e32 v42, s44, v42
	v_mul_lo_u32 v43, v62, s33
	v_xor_b32_e32 v61, v46, v60
	v_add3_u32 v42, v42, v24, v43
	v_xor_b32_e32 v43, v47, v60
	v_mad_u64_u32 v[46:47], s[14:15], v61, s93, 0
	v_mul_hi_u32 v24, v61, s77
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[64:65], s[14:15], v43, s77, 0
	v_add_co_u32_e32 v24, vcc, v46, v64
	v_mad_u64_u32 v[62:63], s[14:15], v43, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v47, v65, vcc
	s_nop 1
	v_addc_co_u32_e32 v63, vcc, 0, v63, vcc
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[62:63]
	v_mul_lo_u32 v24, s87, v46
	v_mul_lo_u32 v47, s86, v47
	v_mad_u64_u32 v[62:63], s[14:15], s86, v46, 0
	v_add3_u32 v24, v63, v47, v24
	v_sub_u32_e32 v47, v43, v24
	v_sub_co_u32_e32 v61, vcc, v61, v62
	s_nop 1
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, vcc
	v_subrev_co_u32_e64 v62, s[14:15], s86, v61
	v_subb_co_u32_e32 v24, vcc, v43, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v47
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v62
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v61
	v_cndmask_b32_e64 v62, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v47
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v47, v63, v62, s[14:15]
	v_add_u32_e32 v62, 2, v46
	v_add_u32_e32 v63, 1, v46
	v_cmp_ne_u32_e64 s[14:15], 0, v47
	v_cndmask_b32_e32 v24, v43, v61, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v47, v63, v62, s[14:15]
	v_xor_b32_e32 v43, s56, v60
	v_cndmask_b32_e32 v24, v46, v47, vcc
	v_ashrrev_i32_e32 v46, 31, v45
	v_mov_b32_e32 v47, v46
	v_xor_b32_e32 v24, v24, v43
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_sub_u32_e32 v24, v24, v43
	v_xor_b32_e32 v47, v44, v46
	v_xad_u32 v64, v24, v26, v51
	v_xor_b32_e32 v43, v45, v46
	v_mad_u64_u32 v[44:45], s[14:15], v47, s93, 0
	v_mul_hi_u32 v24, v47, s77
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[44:45]
	v_mad_u64_u32 v[62:63], s[14:15], v43, s77, 0
	v_add_co_u32_e32 v24, vcc, v44, v62
	v_mad_u64_u32 v[60:61], s[14:15], v43, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v45, v63, vcc
	s_nop 1
	v_addc_co_u32_e32 v61, vcc, 0, v61, vcc
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[60:61]
	v_mul_lo_u32 v24, s87, v44
	v_mul_lo_u32 v60, s86, v45
	v_mad_u64_u32 v[44:45], s[14:15], s86, v44, 0
	v_add3_u32 v24, v45, v60, v24
	v_sub_u32_e32 v45, v43, v24
	v_sub_co_u32_e32 v44, vcc, v47, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, vcc
	v_subrev_co_u32_e64 v47, s[14:15], s86, v44
	v_subb_co_u32_e32 v24, vcc, v43, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v60, s[28:29], 0, v45, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v60
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v47
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v62, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v60
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v44
	v_cndmask_b32_e64 v61, v61, v62, s[28:29]
	v_subrev_co_u32_e64 v62, s[14:15], s86, v47
	s_nop 1
	v_subbrev_co_u32_e64 v45, s[14:15], 0, v45, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v61
	s_nop 1
	v_cndmask_b32_e64 v45, v60, v45, s[14:15]
	v_cndmask_b32_e64 v60, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v43, v43, v60, vcc
	v_cmp_ne_u32_e32 vcc, 0, v43
	v_cndmask_b32_e64 v43, v47, v62, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v24, v45, vcc
	v_cndmask_b32_e32 v43, v44, v43, vcc
	v_xor_b32_e32 v44, v24, v46
	v_xor_b32_e32 v24, v43, v46
	v_sub_co_u32_e32 v24, vcc, v24, v46
	s_nop 1
	v_subb_co_u32_e32 v43, vcc, v44, v46, vcc
	v_ashrrev_i32_e32 v43, 31, v43
	v_and_b32_e32 v43, s44, v43
	v_mul_lo_u32 v44, v64, s33
	v_add3_u32 v43, v43, v24, v44
	v_lshl_add_u64 v[44:45], v[22:23], 0, 4
	v_lshl_add_u64 v[46:47], v[38:39], 0, -5
	v_cndmask_b32_e64 v47, v45, v47, s[22:23]
	v_ashrrev_i32_e32 v60, 31, v47
	v_cndmask_b32_e64 v46, v44, v46, s[22:23]
	v_mov_b32_e32 v61, v60
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[60:61]
	v_xor_b32_e32 v66, v46, v60
	v_xor_b32_e32 v61, v47, v60
	v_mad_u64_u32 v[46:47], s[14:15], v66, s93, 0
	v_mul_hi_u32 v24, v66, s77
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[64:65], s[14:15], v61, s77, 0
	v_add_co_u32_e32 v24, vcc, v46, v64
	v_mad_u64_u32 v[62:63], s[14:15], v61, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v47, v65, vcc
	s_nop 1
	v_addc_co_u32_e32 v63, vcc, 0, v63, vcc
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[62:63]
	v_mul_lo_u32 v24, s87, v46
	v_mul_lo_u32 v47, s86, v47
	v_mad_u64_u32 v[62:63], s[14:15], s86, v46, 0
	v_add3_u32 v24, v63, v47, v24
	v_sub_u32_e32 v47, v61, v24
	v_sub_co_u32_e32 v62, vcc, v66, v62
	s_nop 1
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, vcc
	v_subrev_co_u32_e64 v63, s[14:15], s86, v62
	v_subb_co_u32_e32 v24, vcc, v61, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v47
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v64, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v63
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v62
	v_cndmask_b32_e64 v63, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v47
	v_cndmask_b32_e64 v62, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v47, v64, v63, s[14:15]
	v_add_u32_e32 v63, 2, v46
	v_add_u32_e32 v64, 1, v46
	v_cmp_ne_u32_e64 s[14:15], 0, v47
	v_cndmask_b32_e32 v24, v61, v62, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v47, v64, v63, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v46, v47, vcc
	v_xor_b32_e32 v46, s56, v60
	v_xor_b32_e32 v24, v24, v46
	v_sub_u32_e32 v24, v24, v46
	v_ashrrev_i32_e32 v46, 31, v45
	v_mov_b32_e32 v47, v46
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_xor_b32_e32 v65, v44, v46
	v_xad_u32 v64, v24, v26, v51
	v_xor_b32_e32 v47, v45, v46
	v_mad_u64_u32 v[44:45], s[14:15], v65, s93, 0
	v_mul_hi_u32 v24, v65, s77
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[44:45]
	v_mad_u64_u32 v[62:63], s[14:15], v47, s77, 0
	v_add_co_u32_e32 v24, vcc, v44, v62
	v_mad_u64_u32 v[60:61], s[14:15], v47, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v45, v63, vcc
	s_nop 1
	v_addc_co_u32_e32 v61, vcc, 0, v61, vcc
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[60:61]
	v_mul_lo_u32 v24, s87, v44
	v_mul_lo_u32 v60, s86, v45
	v_mad_u64_u32 v[44:45], s[14:15], s86, v44, 0
	v_add3_u32 v24, v45, v60, v24
	v_sub_u32_e32 v45, v47, v24
	v_sub_co_u32_e32 v44, vcc, v65, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, vcc
	v_subrev_co_u32_e64 v60, s[14:15], s86, v44
	v_subb_co_u32_e32 v24, vcc, v47, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v61, s[28:29], 0, v45, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v61
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v62, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v60
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v61
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v44
	v_cndmask_b32_e64 v62, v62, v63, s[28:29]
	v_subrev_co_u32_e64 v63, s[14:15], s86, v60
	s_nop 1
	v_subbrev_co_u32_e64 v45, s[14:15], 0, v45, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v62
	s_nop 1
	v_cndmask_b32_e64 v45, v61, v45, s[14:15]
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v47, v47, v61, vcc
	v_cmp_ne_u32_e32 vcc, 0, v47
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v45, vcc
	v_cndmask_b32_e64 v45, v60, v63, s[14:15]
	v_cndmask_b32_e32 v44, v44, v45, vcc
	v_xor_b32_e32 v45, v24, v46
	v_xor_b32_e32 v24, v44, v46
	v_sub_co_u32_e32 v24, vcc, v24, v46
	v_lshl_add_u64 v[60:61], v[38:39], 0, -6
	s_nop 0
	v_subb_co_u32_e32 v45, vcc, v45, v46, vcc
	v_lshl_add_u64 v[46:47], v[22:23], 0, 5
	v_cndmask_b32_e64 v61, v47, v61, s[22:23]
	v_ashrrev_i32_e32 v62, 31, v61
	v_cndmask_b32_e64 v60, v46, v60, s[22:23]
	v_mov_b32_e32 v63, v62
	v_ashrrev_i32_e32 v44, 31, v45
	v_lshl_add_u64 v[60:61], v[60:61], 0, v[62:63]
	v_and_b32_e32 v44, s44, v44
	v_mul_lo_u32 v45, v64, s33
	v_xor_b32_e32 v63, v60, v62
	v_add3_u32 v44, v44, v24, v45
	v_xor_b32_e32 v45, v61, v62
	v_mad_u64_u32 v[60:61], s[14:15], v63, s93, 0
	v_mul_hi_u32 v24, v63, s77
	v_lshl_add_u64 v[60:61], v[24:25], 0, v[60:61]
	v_mad_u64_u32 v[66:67], s[14:15], v45, s77, 0
	v_add_co_u32_e32 v24, vcc, v60, v66
	v_mad_u64_u32 v[64:65], s[14:15], v45, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v61, v67, vcc
	s_nop 1
	v_addc_co_u32_e32 v65, vcc, 0, v65, vcc
	v_lshl_add_u64 v[60:61], v[24:25], 0, v[64:65]
	v_mul_lo_u32 v24, s87, v60
	v_mul_lo_u32 v61, s86, v61
	v_mad_u64_u32 v[64:65], s[14:15], s86, v60, 0
	v_add3_u32 v24, v65, v61, v24
	v_sub_u32_e32 v61, v45, v24
	v_sub_co_u32_e32 v63, vcc, v63, v64
	s_nop 1
	v_subb_co_u32_e64 v61, s[14:15], v61, v40, vcc
	v_subrev_co_u32_e64 v64, s[14:15], s86, v63
	v_subb_co_u32_e32 v24, vcc, v45, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v61, s[14:15], 0, v61, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v61
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v64
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v63
	v_cndmask_b32_e64 v64, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v61
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v61, v65, v64, s[14:15]
	v_add_u32_e32 v64, 2, v60
	v_add_u32_e32 v65, 1, v60
	v_cmp_ne_u32_e64 s[14:15], 0, v61
	v_cndmask_b32_e32 v24, v45, v63, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v61, v65, v64, s[14:15]
	v_xor_b32_e32 v45, s56, v62
	v_cndmask_b32_e32 v24, v60, v61, vcc
	v_ashrrev_i32_e32 v60, 31, v47
	v_mov_b32_e32 v61, v60
	v_xor_b32_e32 v24, v24, v45
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[60:61]
	v_sub_u32_e32 v24, v24, v45
	v_xor_b32_e32 v61, v46, v60
	v_xad_u32 v66, v24, v26, v51
	v_xor_b32_e32 v45, v47, v60
	v_mad_u64_u32 v[46:47], s[14:15], v61, s93, 0
	v_mul_hi_u32 v24, v61, s77
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[64:65], s[14:15], v45, s77, 0
	v_add_co_u32_e32 v24, vcc, v46, v64
	v_mad_u64_u32 v[62:63], s[14:15], v45, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v47, v65, vcc
	s_nop 1
	v_addc_co_u32_e32 v63, vcc, 0, v63, vcc
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[62:63]
	v_mul_lo_u32 v24, s87, v46
	v_mul_lo_u32 v62, s86, v47
	v_mad_u64_u32 v[46:47], s[14:15], s86, v46, 0
	v_add3_u32 v24, v47, v62, v24
	v_sub_u32_e32 v47, v45, v24
	v_sub_co_u32_e32 v46, vcc, v61, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, vcc
	v_subrev_co_u32_e64 v61, s[14:15], s86, v46
	v_subb_co_u32_e32 v24, vcc, v45, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v62, s[28:29], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v62
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v61
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v64, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v62
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v46
	v_cndmask_b32_e64 v63, v63, v64, s[28:29]
	v_subrev_co_u32_e64 v64, s[14:15], s86, v61
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v63
	s_nop 1
	v_cndmask_b32_e64 v47, v62, v47, s[14:15]
	v_cndmask_b32_e64 v62, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v45, v45, v62, vcc
	v_cmp_ne_u32_e32 vcc, 0, v45
	v_cndmask_b32_e64 v45, v61, v64, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v24, v47, vcc
	v_cndmask_b32_e32 v45, v46, v45, vcc
	v_xor_b32_e32 v46, v24, v60
	v_xor_b32_e32 v24, v45, v60
	v_sub_co_u32_e32 v24, vcc, v24, v60
	s_nop 1
	v_subb_co_u32_e32 v45, vcc, v46, v60, vcc
	v_ashrrev_i32_e32 v45, 31, v45
	v_and_b32_e32 v45, s44, v45
	v_mul_lo_u32 v46, v66, s33
	v_add3_u32 v45, v45, v24, v46
	v_lshl_add_u64 v[46:47], v[22:23], 0, 6
	v_lshl_add_u64 v[60:61], v[38:39], 0, -7
	v_cndmask_b32_e64 v61, v47, v61, s[22:23]
	v_ashrrev_i32_e32 v62, 31, v61
	v_cndmask_b32_e64 v60, v46, v60, s[22:23]
	v_mov_b32_e32 v63, v62
	v_lshl_add_u64 v[60:61], v[60:61], 0, v[62:63]
	v_xor_b32_e32 v68, v60, v62
	v_xor_b32_e32 v63, v61, v62
	v_mad_u64_u32 v[60:61], s[14:15], v68, s93, 0
	v_mul_hi_u32 v24, v68, s77
	v_lshl_add_u64 v[60:61], v[24:25], 0, v[60:61]
	v_mad_u64_u32 v[66:67], s[14:15], v63, s77, 0
	v_add_co_u32_e32 v24, vcc, v60, v66
	v_mad_u64_u32 v[64:65], s[14:15], v63, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v61, v67, vcc
	s_nop 1
	v_addc_co_u32_e32 v65, vcc, 0, v65, vcc
	v_lshl_add_u64 v[60:61], v[24:25], 0, v[64:65]
	v_mul_lo_u32 v24, s87, v60
	v_mul_lo_u32 v61, s86, v61
	v_mad_u64_u32 v[64:65], s[14:15], s86, v60, 0
	v_add3_u32 v24, v65, v61, v24
	v_sub_u32_e32 v61, v63, v24
	v_sub_co_u32_e32 v64, vcc, v68, v64
	s_nop 1
	v_subb_co_u32_e64 v61, s[14:15], v61, v40, vcc
	v_subrev_co_u32_e64 v65, s[14:15], s86, v64
	v_subb_co_u32_e32 v24, vcc, v63, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v61, s[14:15], 0, v61, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v61
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v66, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v65
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v64
	v_cndmask_b32_e64 v65, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v61
	v_cndmask_b32_e64 v64, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v61, v66, v65, s[14:15]
	v_add_u32_e32 v65, 2, v60
	v_add_u32_e32 v66, 1, v60
	v_cmp_ne_u32_e64 s[14:15], 0, v61
	v_cndmask_b32_e32 v24, v63, v64, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v61, v66, v65, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v60, v61, vcc
	v_xor_b32_e32 v60, s56, v62
	v_xor_b32_e32 v24, v24, v60
	v_sub_u32_e32 v24, v24, v60
	v_ashrrev_i32_e32 v60, 31, v47
	v_mov_b32_e32 v61, v60
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[60:61]
	v_xor_b32_e32 v67, v46, v60
	v_xad_u32 v66, v24, v26, v51
	v_xor_b32_e32 v61, v47, v60
	v_mad_u64_u32 v[46:47], s[14:15], v67, s93, 0
	v_mul_hi_u32 v24, v67, s77
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[64:65], s[14:15], v61, s77, 0
	v_add_co_u32_e32 v24, vcc, v46, v64
	v_mad_u64_u32 v[62:63], s[14:15], v61, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v47, v65, vcc
	s_nop 1
	v_addc_co_u32_e32 v63, vcc, 0, v63, vcc
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[62:63]
	v_mul_lo_u32 v24, s87, v46
	v_mul_lo_u32 v62, s86, v47
	v_mad_u64_u32 v[46:47], s[14:15], s86, v46, 0
	v_add3_u32 v24, v47, v62, v24
	v_sub_u32_e32 v47, v61, v24
	v_sub_co_u32_e32 v46, vcc, v67, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, vcc
	v_subrev_co_u32_e64 v62, s[14:15], s86, v46
	v_subb_co_u32_e32 v24, vcc, v61, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v63, s[28:29], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v63
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v64, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v62
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v63
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v46
	v_cndmask_b32_e64 v64, v64, v65, s[28:29]
	v_subrev_co_u32_e64 v65, s[14:15], s86, v62
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v64
	s_nop 1
	v_cndmask_b32_e64 v47, v63, v47, s[14:15]
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v61, v61, v63, vcc
	v_cmp_ne_u32_e32 vcc, 0, v61
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v47, vcc
	v_cndmask_b32_e64 v47, v62, v65, s[14:15]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v24, v60
	v_xor_b32_e32 v24, v46, v60
	v_sub_co_u32_e32 v24, vcc, v24, v60
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v47, v60, vcc
	v_ashrrev_i32_e32 v46, 31, v47
	v_and_b32_e32 v46, s44, v46
	v_mul_lo_u32 v47, v66, s33
	v_add3_u32 v68, v46, v24, v47
	v_lshl_add_u64 v[46:47], v[22:23], 0, 7
	v_lshl_add_u64 v[60:61], v[38:39], 0, -8
	v_cndmask_b32_e64 v61, v47, v61, s[22:23]
	v_ashrrev_i32_e32 v62, 31, v61
	v_cndmask_b32_e64 v60, v46, v60, s[22:23]
	v_mov_b32_e32 v63, v62
	v_lshl_add_u64 v[60:61], v[60:61], 0, v[62:63]
	v_xor_b32_e32 v69, v60, v62
	v_xor_b32_e32 v63, v61, v62
	v_mad_u64_u32 v[60:61], s[14:15], v69, s93, 0
	v_mul_hi_u32 v24, v69, s77
	v_lshl_add_u64 v[60:61], v[24:25], 0, v[60:61]
	v_mad_u64_u32 v[66:67], s[14:15], v63, s77, 0
	v_add_co_u32_e32 v24, vcc, v60, v66
	v_mad_u64_u32 v[64:65], s[14:15], v63, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v61, v67, vcc
	s_nop 1
	v_addc_co_u32_e32 v65, vcc, 0, v65, vcc
	v_lshl_add_u64 v[60:61], v[24:25], 0, v[64:65]
	v_mul_lo_u32 v24, s87, v60
	v_mul_lo_u32 v61, s86, v61
	v_mad_u64_u32 v[64:65], s[14:15], s86, v60, 0
	v_add3_u32 v24, v65, v61, v24
	v_sub_u32_e32 v61, v63, v24
	v_sub_co_u32_e32 v64, vcc, v69, v64
	s_nop 1
	v_subb_co_u32_e64 v61, s[14:15], v61, v40, vcc
	v_subrev_co_u32_e64 v65, s[14:15], s86, v64
	v_subb_co_u32_e32 v24, vcc, v63, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v61, s[14:15], 0, v61, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v61
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v66, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v65
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v64
	v_cndmask_b32_e64 v65, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v61
	v_cndmask_b32_e64 v64, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v61, v66, v65, s[14:15]
	v_add_u32_e32 v65, 2, v60
	v_add_u32_e32 v66, 1, v60
	v_cmp_ne_u32_e64 s[14:15], 0, v61
	v_cndmask_b32_e32 v24, v63, v64, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v61, v66, v65, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v60, v61, vcc
	v_xor_b32_e32 v60, s56, v62
	v_xor_b32_e32 v24, v24, v60
	v_sub_u32_e32 v24, v24, v60
	v_ashrrev_i32_e32 v60, 31, v47
	v_mov_b32_e32 v61, v60
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[60:61]
	v_xor_b32_e32 v67, v46, v60
	v_xad_u32 v66, v24, v26, v51
	v_xor_b32_e32 v61, v47, v60
	v_mad_u64_u32 v[46:47], s[14:15], v67, s93, 0
	v_mul_hi_u32 v24, v67, s77
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[64:65], s[14:15], v61, s77, 0
	v_add_co_u32_e32 v24, vcc, v46, v64
	v_mad_u64_u32 v[62:63], s[14:15], v61, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v47, v65, vcc
	s_nop 1
	v_addc_co_u32_e32 v63, vcc, 0, v63, vcc
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[62:63]
	v_mul_lo_u32 v24, s87, v46
	v_mul_lo_u32 v62, s86, v47
	v_mad_u64_u32 v[46:47], s[14:15], s86, v46, 0
	v_add3_u32 v24, v47, v62, v24
	v_sub_u32_e32 v47, v61, v24
	v_sub_co_u32_e32 v46, vcc, v67, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, vcc
	v_subrev_co_u32_e64 v62, s[14:15], s86, v46
	v_subb_co_u32_e32 v24, vcc, v61, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v63, s[28:29], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v63
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v64, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v62
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v63
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v46
	v_cndmask_b32_e64 v64, v64, v65, s[28:29]
	v_subrev_co_u32_e64 v65, s[14:15], s86, v62
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v64
	s_nop 1
	v_cndmask_b32_e64 v47, v63, v47, s[14:15]
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v61, v61, v63, vcc
	v_cmp_ne_u32_e32 vcc, 0, v61
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v47, vcc
	v_cndmask_b32_e64 v47, v62, v65, s[14:15]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v24, v60
	v_xor_b32_e32 v24, v46, v60
	v_sub_co_u32_e32 v24, vcc, v24, v60
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v47, v60, vcc
	v_ashrrev_i32_e32 v46, 31, v47
	v_and_b32_e32 v46, s44, v46
	v_mul_lo_u32 v47, v66, s33
	v_add3_u32 v24, v46, v24, v47
	buffer_load_ubyte v60, v27, s[36:39], 0 offen
	buffer_load_ubyte v62, v41, s[36:39], 0 offen
	buffer_load_ubyte v61, v42, s[36:39], 0 offen
	buffer_load_ubyte v63, v43, s[36:39], 0 offen
	buffer_load_ubyte v64, v44, s[36:39], 0 offen
	buffer_load_ubyte v66, v45, s[36:39], 0 offen
	buffer_load_ubyte v65, v68, s[36:39], 0 offen
	buffer_load_ubyte v67, v24, s[36:39], 0 offen
	v_lshl_add_u64 v[42:43], v[22:23], 0, 8
	v_lshl_add_u64 v[44:45], v[38:39], 0, -9
	v_cndmask_b32_e64 v45, v43, v45, s[22:23]
	v_ashrrev_i32_e32 v46, 31, v45
	v_cndmask_b32_e64 v44, v42, v44, s[22:23]
	v_mov_b32_e32 v47, v46
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_xor_b32_e32 v41, v44, v46
	v_xor_b32_e32 v27, v45, v46
	v_mad_u64_u32 v[44:45], s[14:15], v41, s93, 0
	v_mul_hi_u32 v24, v41, s77
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[44:45]
	v_mad_u64_u32 v[70:71], s[14:15], v27, s77, 0
	v_add_co_u32_e32 v24, vcc, v44, v70
	v_mad_u64_u32 v[68:69], s[14:15], v27, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v45, v71, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[68:69]
	v_mul_lo_u32 v24, s87, v44
	v_mul_lo_u32 v45, s86, v45
	v_mad_u64_u32 v[68:69], s[14:15], s86, v44, 0
	v_add3_u32 v24, v69, v45, v24
	v_sub_u32_e32 v45, v27, v24
	v_sub_co_u32_e32 v41, vcc, v41, v68
	s_nop 1
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, vcc
	v_subrev_co_u32_e64 v47, s[14:15], s86, v41
	v_subb_co_u32_e32 v24, vcc, v27, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v45, s[14:15], 0, v45, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v45
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v68, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v47
	v_cndmask_b32_e64 v27, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v41
	v_cndmask_b32_e64 v47, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v45
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v45, v68, v47, s[14:15]
	v_add_u32_e32 v47, 2, v44
	v_add_u32_e32 v68, 1, v44
	v_cmp_ne_u32_e64 s[14:15], 0, v45
	v_cndmask_b32_e32 v24, v27, v41, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v45, v68, v47, s[14:15]
	v_xor_b32_e32 v27, s56, v46
	v_cndmask_b32_e32 v24, v44, v45, vcc
	v_ashrrev_i32_e32 v44, 31, v43
	v_mov_b32_e32 v45, v44
	v_xor_b32_e32 v24, v24, v27
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[44:45]
	v_sub_u32_e32 v24, v24, v27
	v_xor_b32_e32 v45, v42, v44
	v_xad_u32 v41, v24, v26, v51
	v_xor_b32_e32 v27, v43, v44
	v_mad_u64_u32 v[42:43], s[14:15], v45, s93, 0
	v_mul_hi_u32 v24, v45, s77
	v_lshl_add_u64 v[42:43], v[24:25], 0, v[42:43]
	v_mad_u64_u32 v[68:69], s[14:15], v27, s77, 0
	v_add_co_u32_e32 v24, vcc, v42, v68
	v_mad_u64_u32 v[46:47], s[14:15], v27, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v43, v69, vcc
	v_mul_lo_u32 v41, v41, s33
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[24:25], 0, v[46:47]
	v_mul_lo_u32 v24, s87, v42
	v_mul_lo_u32 v46, s86, v43
	v_mad_u64_u32 v[42:43], s[14:15], s86, v42, 0
	v_add3_u32 v24, v43, v46, v24
	v_sub_u32_e32 v43, v27, v24
	v_sub_co_u32_e32 v42, vcc, v45, v42
	s_nop 1
	v_subb_co_u32_e64 v43, s[14:15], v43, v40, vcc
	v_subrev_co_u32_e64 v45, s[14:15], s86, v42
	v_subb_co_u32_e32 v24, vcc, v27, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v46, s[28:29], 0, v43, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v46
	v_subb_co_u32_e64 v43, s[14:15], v43, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v45
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v68, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v46
	v_cndmask_b32_e64 v27, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v42
	v_cndmask_b32_e64 v47, v47, v68, s[28:29]
	v_subrev_co_u32_e64 v68, s[14:15], s86, v45
	s_nop 1
	v_subbrev_co_u32_e64 v43, s[14:15], 0, v43, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v47
	s_nop 1
	v_cndmask_b32_e64 v43, v46, v43, s[14:15]
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v27, v27, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v27
	v_cndmask_b32_e64 v27, v45, v68, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v24, v43, vcc
	v_cndmask_b32_e32 v27, v42, v27, vcc
	v_xor_b32_e32 v42, v24, v44
	v_xor_b32_e32 v24, v27, v44
	v_sub_co_u32_e32 v24, vcc, v24, v44
	s_nop 1
	v_subb_co_u32_e32 v27, vcc, v42, v44, vcc
	v_lshl_add_u64 v[42:43], v[22:23], 0, 9
	v_lshl_add_u64 v[44:45], v[38:39], 0, -10
	v_cndmask_b32_e64 v45, v43, v45, s[22:23]
	v_ashrrev_i32_e32 v46, 31, v45
	v_cndmask_b32_e64 v44, v42, v44, s[22:23]
	v_mov_b32_e32 v47, v46
	v_ashrrev_i32_e32 v27, 31, v27
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_and_b32_e32 v27, s44, v27
	v_xor_b32_e32 v47, v44, v46
	v_add3_u32 v27, v27, v24, v41
	v_xor_b32_e32 v41, v45, v46
	v_mad_u64_u32 v[44:45], s[14:15], v47, s93, 0
	v_mul_hi_u32 v24, v47, s77
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[44:45]
	v_mad_u64_u32 v[70:71], s[14:15], v41, s77, 0
	v_add_co_u32_e32 v24, vcc, v44, v70
	v_mad_u64_u32 v[68:69], s[14:15], v41, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v45, v71, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[68:69]
	v_mul_lo_u32 v24, s87, v44
	v_mul_lo_u32 v45, s86, v45
	v_mad_u64_u32 v[68:69], s[14:15], s86, v44, 0
	v_add3_u32 v24, v69, v45, v24
	v_sub_u32_e32 v45, v41, v24
	v_sub_co_u32_e32 v47, vcc, v47, v68
	s_nop 1
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, vcc
	v_subrev_co_u32_e64 v68, s[14:15], s86, v47
	v_subb_co_u32_e32 v24, vcc, v41, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v45, s[14:15], 0, v45, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v45
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v68
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v47
	v_cndmask_b32_e64 v68, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v45
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v45, v69, v68, s[14:15]
	v_add_u32_e32 v68, 2, v44
	v_add_u32_e32 v69, 1, v44
	v_cmp_ne_u32_e64 s[14:15], 0, v45
	v_cndmask_b32_e32 v24, v41, v47, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v45, v69, v68, s[14:15]
	v_xor_b32_e32 v41, s56, v46
	v_cndmask_b32_e32 v24, v44, v45, vcc
	v_ashrrev_i32_e32 v44, 31, v43
	v_mov_b32_e32 v45, v44
	v_xor_b32_e32 v24, v24, v41
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[44:45]
	v_sub_u32_e32 v24, v24, v41
	v_xor_b32_e32 v45, v42, v44
	v_xad_u32 v70, v24, v26, v51
	v_xor_b32_e32 v41, v43, v44
	v_mad_u64_u32 v[42:43], s[14:15], v45, s93, 0
	v_mul_hi_u32 v24, v45, s77
	v_lshl_add_u64 v[42:43], v[24:25], 0, v[42:43]
	v_mad_u64_u32 v[68:69], s[14:15], v41, s77, 0
	v_add_co_u32_e32 v24, vcc, v42, v68
	v_mad_u64_u32 v[46:47], s[14:15], v41, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v43, v69, vcc
	s_nop 1
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[24:25], 0, v[46:47]
	v_mul_lo_u32 v24, s87, v42
	v_mul_lo_u32 v46, s86, v43
	v_mad_u64_u32 v[42:43], s[14:15], s86, v42, 0
	v_add3_u32 v24, v43, v46, v24
	v_sub_u32_e32 v43, v41, v24
	v_sub_co_u32_e32 v42, vcc, v45, v42
	s_nop 1
	v_subb_co_u32_e64 v43, s[14:15], v43, v40, vcc
	v_subrev_co_u32_e64 v45, s[14:15], s86, v42
	v_subb_co_u32_e32 v24, vcc, v41, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v46, s[28:29], 0, v43, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v46
	v_subb_co_u32_e64 v43, s[14:15], v43, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v45
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v68, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v46
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v42
	v_cndmask_b32_e64 v47, v47, v68, s[28:29]
	v_subrev_co_u32_e64 v68, s[14:15], s86, v45
	s_nop 1
	v_subbrev_co_u32_e64 v43, s[14:15], 0, v43, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v47
	s_nop 1
	v_cndmask_b32_e64 v43, v46, v43, s[14:15]
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v41, v41, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v41
	v_cndmask_b32_e64 v41, v45, v68, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v24, v43, vcc
	v_cndmask_b32_e32 v41, v42, v41, vcc
	v_xor_b32_e32 v42, v24, v44
	v_xor_b32_e32 v24, v41, v44
	v_sub_co_u32_e32 v24, vcc, v24, v44
	s_nop 1
	v_subb_co_u32_e32 v41, vcc, v42, v44, vcc
	v_ashrrev_i32_e32 v41, 31, v41
	v_and_b32_e32 v41, s44, v41
	v_mul_lo_u32 v42, v70, s33
	v_add3_u32 v41, v41, v24, v42
	v_lshl_add_u64 v[42:43], v[22:23], 0, 10
	v_lshl_add_u64 v[44:45], v[38:39], 0, -11
	v_cndmask_b32_e64 v45, v43, v45, s[22:23]
	v_ashrrev_i32_e32 v46, 31, v45
	v_cndmask_b32_e64 v44, v42, v44, s[22:23]
	v_mov_b32_e32 v47, v46
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_xor_b32_e32 v72, v44, v46
	v_xor_b32_e32 v47, v45, v46
	v_mad_u64_u32 v[44:45], s[14:15], v72, s93, 0
	v_mul_hi_u32 v24, v72, s77
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[44:45]
	v_mad_u64_u32 v[70:71], s[14:15], v47, s77, 0
	v_add_co_u32_e32 v24, vcc, v44, v70
	v_mad_u64_u32 v[68:69], s[14:15], v47, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v45, v71, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[68:69]
	v_mul_lo_u32 v24, s87, v44
	v_mul_lo_u32 v45, s86, v45
	v_mad_u64_u32 v[68:69], s[14:15], s86, v44, 0
	v_add3_u32 v24, v69, v45, v24
	v_sub_u32_e32 v45, v47, v24
	v_sub_co_u32_e32 v68, vcc, v72, v68
	s_nop 1
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, vcc
	v_subrev_co_u32_e64 v69, s[14:15], s86, v68
	v_subb_co_u32_e32 v24, vcc, v47, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v45, s[14:15], 0, v45, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v45
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v69
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v68
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v45
	v_cndmask_b32_e64 v68, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v45, v70, v69, s[14:15]
	v_add_u32_e32 v69, 2, v44
	v_add_u32_e32 v70, 1, v44
	v_cmp_ne_u32_e64 s[14:15], 0, v45
	v_cndmask_b32_e32 v24, v47, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v45, v70, v69, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v44, v45, vcc
	v_xor_b32_e32 v44, s56, v46
	v_xor_b32_e32 v24, v24, v44
	v_sub_u32_e32 v24, v24, v44
	v_ashrrev_i32_e32 v44, 31, v43
	v_mov_b32_e32 v45, v44
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[44:45]
	v_xor_b32_e32 v71, v42, v44
	v_xad_u32 v70, v24, v26, v51
	v_xor_b32_e32 v45, v43, v44
	v_mad_u64_u32 v[42:43], s[14:15], v71, s93, 0
	v_mul_hi_u32 v24, v71, s77
	v_lshl_add_u64 v[42:43], v[24:25], 0, v[42:43]
	v_mad_u64_u32 v[68:69], s[14:15], v45, s77, 0
	v_add_co_u32_e32 v24, vcc, v42, v68
	v_mad_u64_u32 v[46:47], s[14:15], v45, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v43, v69, vcc
	s_nop 1
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[24:25], 0, v[46:47]
	v_mul_lo_u32 v24, s87, v42
	v_mul_lo_u32 v46, s86, v43
	v_mad_u64_u32 v[42:43], s[14:15], s86, v42, 0
	v_add3_u32 v24, v43, v46, v24
	v_sub_u32_e32 v43, v45, v24
	v_sub_co_u32_e32 v42, vcc, v71, v42
	s_nop 1
	v_subb_co_u32_e64 v43, s[14:15], v43, v40, vcc
	v_subrev_co_u32_e64 v46, s[14:15], s86, v42
	v_subb_co_u32_e32 v24, vcc, v45, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[28:29], 0, v43, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v47
	v_subb_co_u32_e64 v43, s[14:15], v43, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v68, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v46
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v47
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v42
	v_cndmask_b32_e64 v68, v68, v69, s[28:29]
	v_subrev_co_u32_e64 v69, s[14:15], s86, v46
	s_nop 1
	v_subbrev_co_u32_e64 v43, s[14:15], 0, v43, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v68
	s_nop 1
	v_cndmask_b32_e64 v43, v47, v43, s[14:15]
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v45, v45, v47, vcc
	v_cmp_ne_u32_e32 vcc, 0, v45
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v43, vcc
	v_cndmask_b32_e64 v43, v46, v69, s[14:15]
	v_cndmask_b32_e32 v42, v42, v43, vcc
	v_xor_b32_e32 v43, v24, v44
	v_xor_b32_e32 v24, v42, v44
	v_sub_co_u32_e32 v24, vcc, v24, v44
	v_lshl_add_u64 v[46:47], v[38:39], 0, -12
	s_nop 0
	v_subb_co_u32_e32 v43, vcc, v43, v44, vcc
	v_lshl_add_u64 v[44:45], v[22:23], 0, 11
	v_cndmask_b32_e64 v47, v45, v47, s[22:23]
	v_ashrrev_i32_e32 v68, 31, v47
	v_cndmask_b32_e64 v46, v44, v46, s[22:23]
	v_mov_b32_e32 v69, v68
	v_ashrrev_i32_e32 v42, 31, v43
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[68:69]
	v_and_b32_e32 v42, s44, v42
	v_mul_lo_u32 v43, v70, s33
	v_xor_b32_e32 v69, v46, v68
	v_add3_u32 v42, v42, v24, v43
	v_xor_b32_e32 v43, v47, v68
	v_mad_u64_u32 v[46:47], s[14:15], v69, s93, 0
	v_mul_hi_u32 v24, v69, s77
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[72:73], s[14:15], v43, s77, 0
	v_add_co_u32_e32 v24, vcc, v46, v72
	v_mad_u64_u32 v[70:71], s[14:15], v43, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v47, v73, vcc
	s_nop 1
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[70:71]
	v_mul_lo_u32 v24, s87, v46
	v_mul_lo_u32 v47, s86, v47
	v_mad_u64_u32 v[70:71], s[14:15], s86, v46, 0
	v_add3_u32 v24, v71, v47, v24
	v_sub_u32_e32 v47, v43, v24
	v_sub_co_u32_e32 v69, vcc, v69, v70
	s_nop 1
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, vcc
	v_subrev_co_u32_e64 v70, s[14:15], s86, v69
	v_subb_co_u32_e32 v24, vcc, v43, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v47
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v70
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v69
	v_cndmask_b32_e64 v70, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v47
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v47, v71, v70, s[14:15]
	v_add_u32_e32 v70, 2, v46
	v_add_u32_e32 v71, 1, v46
	v_cmp_ne_u32_e64 s[14:15], 0, v47
	v_cndmask_b32_e32 v24, v43, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v47, v71, v70, s[14:15]
	v_xor_b32_e32 v43, s56, v68
	v_cndmask_b32_e32 v24, v46, v47, vcc
	v_ashrrev_i32_e32 v46, 31, v45
	v_mov_b32_e32 v47, v46
	v_xor_b32_e32 v24, v24, v43
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_sub_u32_e32 v24, v24, v43
	v_xor_b32_e32 v47, v44, v46
	v_xad_u32 v72, v24, v26, v51
	v_xor_b32_e32 v43, v45, v46
	v_mad_u64_u32 v[44:45], s[14:15], v47, s93, 0
	v_mul_hi_u32 v24, v47, s77
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[44:45]
	v_mad_u64_u32 v[70:71], s[14:15], v43, s77, 0
	v_add_co_u32_e32 v24, vcc, v44, v70
	v_mad_u64_u32 v[68:69], s[14:15], v43, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v45, v71, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[68:69]
	v_mul_lo_u32 v24, s87, v44
	v_mul_lo_u32 v68, s86, v45
	v_mad_u64_u32 v[44:45], s[14:15], s86, v44, 0
	v_add3_u32 v24, v45, v68, v24
	v_sub_u32_e32 v45, v43, v24
	v_sub_co_u32_e32 v44, vcc, v47, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, vcc
	v_subrev_co_u32_e64 v47, s[14:15], s86, v44
	v_subb_co_u32_e32 v24, vcc, v43, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v68, s[28:29], 0, v45, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v68
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v47
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v68
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v44
	v_cndmask_b32_e64 v69, v69, v70, s[28:29]
	v_subrev_co_u32_e64 v70, s[14:15], s86, v47
	s_nop 1
	v_subbrev_co_u32_e64 v45, s[14:15], 0, v45, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v69
	s_nop 1
	v_cndmask_b32_e64 v45, v68, v45, s[14:15]
	v_cndmask_b32_e64 v68, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v43, v43, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v43
	v_cndmask_b32_e64 v43, v47, v70, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v24, v45, vcc
	v_cndmask_b32_e32 v43, v44, v43, vcc
	v_xor_b32_e32 v44, v24, v46
	v_xor_b32_e32 v24, v43, v46
	v_sub_co_u32_e32 v24, vcc, v24, v46
	s_nop 1
	v_subb_co_u32_e32 v43, vcc, v44, v46, vcc
	v_ashrrev_i32_e32 v43, 31, v43
	v_and_b32_e32 v43, s44, v43
	v_mul_lo_u32 v44, v72, s33
	v_add3_u32 v43, v43, v24, v44
	v_lshl_add_u64 v[44:45], v[22:23], 0, 12
	v_lshl_add_u64 v[46:47], v[38:39], 0, -13
	v_cndmask_b32_e64 v47, v45, v47, s[22:23]
	v_ashrrev_i32_e32 v68, 31, v47
	v_cndmask_b32_e64 v46, v44, v46, s[22:23]
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[68:69]
	v_xor_b32_e32 v74, v46, v68
	v_xor_b32_e32 v69, v47, v68
	v_mad_u64_u32 v[46:47], s[14:15], v74, s93, 0
	v_mul_hi_u32 v24, v74, s77
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[72:73], s[14:15], v69, s77, 0
	v_add_co_u32_e32 v24, vcc, v46, v72
	v_mad_u64_u32 v[70:71], s[14:15], v69, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v47, v73, vcc
	s_nop 1
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[70:71]
	v_mul_lo_u32 v24, s87, v46
	v_mul_lo_u32 v47, s86, v47
	v_mad_u64_u32 v[70:71], s[14:15], s86, v46, 0
	v_add3_u32 v24, v71, v47, v24
	v_sub_u32_e32 v47, v69, v24
	v_sub_co_u32_e32 v70, vcc, v74, v70
	s_nop 1
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, vcc
	v_subrev_co_u32_e64 v71, s[14:15], s86, v70
	v_subb_co_u32_e32 v24, vcc, v69, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v47
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v71
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v70
	v_cndmask_b32_e64 v71, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v47
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v47, v72, v71, s[14:15]
	v_add_u32_e32 v71, 2, v46
	v_add_u32_e32 v72, 1, v46
	v_cmp_ne_u32_e64 s[14:15], 0, v47
	v_cndmask_b32_e32 v24, v69, v70, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v47, v72, v71, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v46, v47, vcc
	v_xor_b32_e32 v46, s56, v68
	v_xor_b32_e32 v24, v24, v46
	v_sub_u32_e32 v24, v24, v46
	v_ashrrev_i32_e32 v46, 31, v45
	v_mov_b32_e32 v47, v46
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_xor_b32_e32 v73, v44, v46
	v_xad_u32 v72, v24, v26, v51
	v_xor_b32_e32 v47, v45, v46
	v_mad_u64_u32 v[44:45], s[14:15], v73, s93, 0
	v_mul_hi_u32 v24, v73, s77
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[44:45]
	v_mad_u64_u32 v[70:71], s[14:15], v47, s77, 0
	v_add_co_u32_e32 v24, vcc, v44, v70
	v_mad_u64_u32 v[68:69], s[14:15], v47, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v45, v71, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[68:69]
	v_mul_lo_u32 v24, s87, v44
	v_mul_lo_u32 v68, s86, v45
	v_mad_u64_u32 v[44:45], s[14:15], s86, v44, 0
	v_add3_u32 v24, v45, v68, v24
	v_sub_u32_e32 v45, v47, v24
	v_sub_co_u32_e32 v44, vcc, v73, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, vcc
	v_subrev_co_u32_e64 v68, s[14:15], s86, v44
	v_subb_co_u32_e32 v24, vcc, v47, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[28:29], 0, v45, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v69
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v68
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v69
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v44
	v_cndmask_b32_e64 v70, v70, v71, s[28:29]
	v_subrev_co_u32_e64 v71, s[14:15], s86, v68
	s_nop 1
	v_subbrev_co_u32_e64 v45, s[14:15], 0, v45, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v70
	s_nop 1
	v_cndmask_b32_e64 v45, v69, v45, s[14:15]
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v47, v47, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v47
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v45, vcc
	v_cndmask_b32_e64 v45, v68, v71, s[14:15]
	v_cndmask_b32_e32 v44, v44, v45, vcc
	v_xor_b32_e32 v45, v24, v46
	v_xor_b32_e32 v24, v44, v46
	v_sub_co_u32_e32 v24, vcc, v24, v46
	v_lshl_add_u64 v[68:69], v[38:39], 0, -14
	s_nop 0
	v_subb_co_u32_e32 v45, vcc, v45, v46, vcc
	v_lshl_add_u64 v[46:47], v[22:23], 0, 13
	v_cndmask_b32_e64 v69, v47, v69, s[22:23]
	v_ashrrev_i32_e32 v70, 31, v69
	v_cndmask_b32_e64 v68, v46, v68, s[22:23]
	v_mov_b32_e32 v71, v70
	v_ashrrev_i32_e32 v44, 31, v45
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[70:71]
	v_and_b32_e32 v44, s44, v44
	v_mul_lo_u32 v45, v72, s33
	v_xor_b32_e32 v71, v68, v70
	v_add3_u32 v44, v44, v24, v45
	v_xor_b32_e32 v45, v69, v70
	v_mad_u64_u32 v[68:69], s[14:15], v71, s93, 0
	v_mul_hi_u32 v24, v71, s77
	v_lshl_add_u64 v[68:69], v[24:25], 0, v[68:69]
	v_mad_u64_u32 v[74:75], s[14:15], v45, s77, 0
	v_add_co_u32_e32 v24, vcc, v68, v74
	v_mad_u64_u32 v[72:73], s[14:15], v45, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v69, v75, vcc
	s_nop 1
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[68:69], v[24:25], 0, v[72:73]
	v_mul_lo_u32 v24, s87, v68
	v_mul_lo_u32 v69, s86, v69
	v_mad_u64_u32 v[72:73], s[14:15], s86, v68, 0
	v_add3_u32 v24, v73, v69, v24
	v_sub_u32_e32 v69, v45, v24
	v_sub_co_u32_e32 v71, vcc, v71, v72
	s_nop 1
	v_subb_co_u32_e64 v69, s[14:15], v69, v40, vcc
	v_subrev_co_u32_e64 v72, s[14:15], s86, v71
	v_subb_co_u32_e32 v24, vcc, v45, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v69
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v73, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v72
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v71
	v_cndmask_b32_e64 v72, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v69
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v69, v73, v72, s[14:15]
	v_add_u32_e32 v72, 2, v68
	v_add_u32_e32 v73, 1, v68
	v_cmp_ne_u32_e64 s[14:15], 0, v69
	v_cndmask_b32_e32 v24, v45, v71, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v69, v73, v72, s[14:15]
	v_xor_b32_e32 v45, s56, v70
	v_cndmask_b32_e32 v24, v68, v69, vcc
	v_ashrrev_i32_e32 v68, 31, v47
	v_mov_b32_e32 v69, v68
	v_xor_b32_e32 v24, v24, v45
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[68:69]
	v_sub_u32_e32 v24, v24, v45
	v_xor_b32_e32 v69, v46, v68
	v_xad_u32 v74, v24, v26, v51
	v_xor_b32_e32 v45, v47, v68
	v_mad_u64_u32 v[46:47], s[14:15], v69, s93, 0
	v_mul_hi_u32 v24, v69, s77
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[72:73], s[14:15], v45, s77, 0
	v_add_co_u32_e32 v24, vcc, v46, v72
	v_mad_u64_u32 v[70:71], s[14:15], v45, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v47, v73, vcc
	s_nop 1
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[70:71]
	v_mul_lo_u32 v24, s87, v46
	v_mul_lo_u32 v70, s86, v47
	v_mad_u64_u32 v[46:47], s[14:15], s86, v46, 0
	v_add3_u32 v24, v47, v70, v24
	v_sub_u32_e32 v47, v45, v24
	v_sub_co_u32_e32 v46, vcc, v69, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, vcc
	v_subrev_co_u32_e64 v69, s[14:15], s86, v46
	v_subb_co_u32_e32 v24, vcc, v45, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v70, s[28:29], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v70
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v69
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v70
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v46
	v_cndmask_b32_e64 v71, v71, v72, s[28:29]
	v_subrev_co_u32_e64 v72, s[14:15], s86, v69
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v71
	s_nop 1
	v_cndmask_b32_e64 v47, v70, v47, s[14:15]
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v45, v45, v70, vcc
	v_cmp_ne_u32_e32 vcc, 0, v45
	v_cndmask_b32_e64 v45, v69, v72, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v24, v47, vcc
	v_cndmask_b32_e32 v45, v46, v45, vcc
	v_xor_b32_e32 v46, v24, v68
	v_xor_b32_e32 v24, v45, v68
	v_sub_co_u32_e32 v24, vcc, v24, v68
	s_nop 1
	v_subb_co_u32_e32 v45, vcc, v46, v68, vcc
	v_ashrrev_i32_e32 v45, 31, v45
	v_and_b32_e32 v45, s44, v45
	v_mul_lo_u32 v46, v74, s33
	v_add3_u32 v45, v45, v24, v46
	v_lshl_add_u64 v[46:47], v[22:23], 0, 14
	v_lshl_add_u64 v[68:69], v[38:39], 0, -15
	v_cndmask_b32_e64 v69, v47, v69, s[22:23]
	v_ashrrev_i32_e32 v70, 31, v69
	v_cndmask_b32_e64 v68, v46, v68, s[22:23]
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[70:71]
	v_xor_b32_e32 v76, v68, v70
	v_xor_b32_e32 v71, v69, v70
	v_mad_u64_u32 v[68:69], s[14:15], v76, s93, 0
	v_mul_hi_u32 v24, v76, s77
	v_lshl_add_u64 v[68:69], v[24:25], 0, v[68:69]
	v_mad_u64_u32 v[74:75], s[14:15], v71, s77, 0
	v_add_co_u32_e32 v24, vcc, v68, v74
	v_mad_u64_u32 v[72:73], s[14:15], v71, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v69, v75, vcc
	s_nop 1
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[68:69], v[24:25], 0, v[72:73]
	v_mul_lo_u32 v24, s87, v68
	v_mul_lo_u32 v69, s86, v69
	v_mad_u64_u32 v[72:73], s[14:15], s86, v68, 0
	v_add3_u32 v24, v73, v69, v24
	v_sub_u32_e32 v69, v71, v24
	v_sub_co_u32_e32 v72, vcc, v76, v72
	s_nop 1
	v_subb_co_u32_e64 v69, s[14:15], v69, v40, vcc
	v_subrev_co_u32_e64 v73, s[14:15], s86, v72
	v_subb_co_u32_e32 v24, vcc, v71, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v69
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v74, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v73
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v72
	v_cndmask_b32_e64 v73, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v69
	v_cndmask_b32_e64 v72, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v69, v74, v73, s[14:15]
	v_add_u32_e32 v73, 2, v68
	v_add_u32_e32 v74, 1, v68
	v_cmp_ne_u32_e64 s[14:15], 0, v69
	v_cndmask_b32_e32 v24, v71, v72, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v69, v74, v73, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v68, v69, vcc
	v_xor_b32_e32 v68, s56, v70
	v_xor_b32_e32 v24, v24, v68
	v_sub_u32_e32 v24, v24, v68
	v_ashrrev_i32_e32 v68, 31, v47
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[68:69]
	v_xor_b32_e32 v75, v46, v68
	v_xad_u32 v74, v24, v26, v51
	v_xor_b32_e32 v69, v47, v68
	v_mad_u64_u32 v[46:47], s[14:15], v75, s93, 0
	v_mul_hi_u32 v24, v75, s77
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[72:73], s[14:15], v69, s77, 0
	v_add_co_u32_e32 v24, vcc, v46, v72
	v_mad_u64_u32 v[70:71], s[14:15], v69, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v47, v73, vcc
	s_nop 1
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[70:71]
	v_mul_lo_u32 v24, s87, v46
	v_mul_lo_u32 v70, s86, v47
	v_mad_u64_u32 v[46:47], s[14:15], s86, v46, 0
	v_add3_u32 v24, v47, v70, v24
	v_sub_u32_e32 v47, v69, v24
	v_sub_co_u32_e32 v46, vcc, v75, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, vcc
	v_subrev_co_u32_e64 v70, s[14:15], s86, v46
	v_subb_co_u32_e32 v24, vcc, v69, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v71, s[28:29], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v71
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v70
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v73, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v71
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v46
	v_cndmask_b32_e64 v72, v72, v73, s[28:29]
	v_subrev_co_u32_e64 v73, s[14:15], s86, v70
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v72
	s_nop 1
	v_cndmask_b32_e64 v47, v71, v47, s[14:15]
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v69, v69, v71, vcc
	v_cmp_ne_u32_e32 vcc, 0, v69
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v47, vcc
	v_cndmask_b32_e64 v47, v70, v73, s[14:15]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v24, v68
	v_xor_b32_e32 v24, v46, v68
	v_sub_co_u32_e32 v24, vcc, v24, v68
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v47, v68, vcc
	v_ashrrev_i32_e32 v46, 31, v47
	v_and_b32_e32 v46, s44, v46
	v_mul_lo_u32 v47, v74, s33
	v_add3_u32 v76, v46, v24, v47
	v_lshl_add_u64 v[46:47], v[22:23], 0, 15
	v_lshl_add_u64 v[68:69], v[38:39], 0, -16
	v_cndmask_b32_e64 v69, v47, v69, s[22:23]
	v_ashrrev_i32_e32 v70, 31, v69
	v_cndmask_b32_e64 v68, v46, v68, s[22:23]
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[70:71]
	v_xor_b32_e32 v77, v68, v70
	v_xor_b32_e32 v71, v69, v70
	v_mad_u64_u32 v[68:69], s[14:15], v77, s93, 0
	v_mul_hi_u32 v24, v77, s77
	v_lshl_add_u64 v[68:69], v[24:25], 0, v[68:69]
	v_mad_u64_u32 v[74:75], s[14:15], v71, s77, 0
	v_add_co_u32_e32 v24, vcc, v68, v74
	v_mad_u64_u32 v[72:73], s[14:15], v71, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v69, v75, vcc
	s_nop 1
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[68:69], v[24:25], 0, v[72:73]
	v_mul_lo_u32 v24, s87, v68
	v_mul_lo_u32 v69, s86, v69
	v_mad_u64_u32 v[72:73], s[14:15], s86, v68, 0
	v_add3_u32 v24, v73, v69, v24
	v_sub_u32_e32 v69, v71, v24
	v_sub_co_u32_e32 v72, vcc, v77, v72
	s_nop 1
	v_subb_co_u32_e64 v69, s[14:15], v69, v40, vcc
	v_subrev_co_u32_e64 v73, s[14:15], s86, v72
	v_subb_co_u32_e32 v24, vcc, v71, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v69
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v74, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v73
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v72
	v_cndmask_b32_e64 v73, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v69
	v_cndmask_b32_e64 v72, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v69, v74, v73, s[14:15]
	v_add_u32_e32 v73, 2, v68
	v_add_u32_e32 v74, 1, v68
	v_cmp_ne_u32_e64 s[14:15], 0, v69
	v_cndmask_b32_e32 v24, v71, v72, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v69, v74, v73, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v68, v69, vcc
	v_xor_b32_e32 v68, s56, v70
	v_xor_b32_e32 v24, v24, v68
	v_sub_u32_e32 v24, v24, v68
	v_ashrrev_i32_e32 v68, 31, v47
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[68:69]
	v_xor_b32_e32 v74, v46, v68
	v_xad_u32 v26, v24, v26, v51
	v_xor_b32_e32 v69, v47, v68
	v_mad_u64_u32 v[46:47], s[14:15], v74, s93, 0
	v_mul_hi_u32 v24, v74, s77
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[72:73], s[14:15], v69, s77, 0
	v_add_co_u32_e32 v24, vcc, v46, v72
	v_mad_u64_u32 v[70:71], s[14:15], v69, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v47, v73, vcc
	v_mul_lo_u32 v26, v26, s33
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[70:71]
	v_mul_lo_u32 v24, s87, v46
	v_mul_lo_u32 v70, s86, v47
	v_mad_u64_u32 v[46:47], s[14:15], s86, v46, 0
	v_add3_u32 v24, v47, v70, v24
	v_sub_u32_e32 v47, v69, v24
	v_sub_co_u32_e32 v46, vcc, v74, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, vcc
	v_subrev_co_u32_e64 v70, s[14:15], s86, v46
	v_subb_co_u32_e32 v24, vcc, v69, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v71, s[22:23], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s87, v71
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s86, v70
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v73, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s87, v71
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v46
	v_cndmask_b32_e64 v72, v72, v73, s[22:23]
	v_subrev_co_u32_e64 v73, s[14:15], s86, v70
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v72
	s_nop 1
	v_cndmask_b32_e64 v47, v71, v47, s[14:15]
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v69, v69, v71, vcc
	v_cmp_ne_u32_e32 vcc, 0, v69
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v47, vcc
	v_cndmask_b32_e64 v47, v70, v73, s[14:15]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v24, v68
	v_xor_b32_e32 v24, v46, v68
	v_sub_co_u32_e32 v24, vcc, v24, v68
	s_movk_i32 s14, 0xfbff
	s_nop 0
	v_subb_co_u32_e32 v47, vcc, v47, v68, vcc
	v_ashrrev_i32_e32 v46, 31, v47
	v_and_b32_e32 v46, s44, v46
	v_add3_u32 v24, v46, v24, v26
	buffer_load_ubyte v68, v27, s[36:39], 0 offen
	buffer_load_ubyte v70, v41, s[36:39], 0 offen
	buffer_load_ubyte v69, v42, s[36:39], 0 offen
	buffer_load_ubyte v71, v43, s[36:39], 0 offen
	buffer_load_ubyte v72, v44, s[36:39], 0 offen
	buffer_load_ubyte v74, v45, s[36:39], 0 offen
	buffer_load_ubyte v73, v76, s[36:39], 0 offen
	buffer_load_ubyte v75, v24, s[36:39], 0 offen
	v_lshl_add_u64 v[26:27], v[22:23], 0, s[74:75]
	s_mov_b32 s15, -1
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[14:15]
	v_cmp_gt_i32_e32 vcc, 0, v27
	s_nop 1
	v_cndmask_b32_e32 v39, v27, v39, vcc
	v_ashrrev_i32_e32 v42, 31, v39
	v_cndmask_b32_e32 v38, v26, v38, vcc
	v_mov_b32_e32 v43, v42
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[42:43]
	v_xor_b32_e32 v43, v38, v42
	v_xor_b32_e32 v41, v39, v42
	v_mad_u64_u32 v[38:39], s[14:15], v43, s93, 0
	v_mul_hi_u32 v24, v43, s77
	v_lshl_add_u64 v[38:39], v[24:25], 0, v[38:39]
	v_mad_u64_u32 v[46:47], s[14:15], v41, s77, 0
	v_add_co_u32_e32 v24, vcc, v38, v46
	v_mad_u64_u32 v[44:45], s[14:15], v41, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v39, v47, vcc
	s_nop 1
	v_addc_co_u32_e32 v45, vcc, 0, v45, vcc
	v_lshl_add_u64 v[38:39], v[24:25], 0, v[44:45]
	v_mul_lo_u32 v24, s87, v38
	v_mul_lo_u32 v39, s86, v39
	v_mad_u64_u32 v[44:45], s[14:15], s86, v38, 0
	v_add3_u32 v24, v45, v39, v24
	v_sub_u32_e32 v39, v41, v24
	v_sub_co_u32_e32 v43, vcc, v43, v44
	s_nop 1
	v_subb_co_u32_e64 v39, s[14:15], v39, v40, vcc
	v_subrev_co_u32_e64 v44, s[14:15], s86, v43
	v_subb_co_u32_e32 v24, vcc, v41, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v39, s[14:15], 0, v39, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v39
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v45, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v44
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v43
	v_cndmask_b32_e64 v44, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v39
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v39, v45, v44, s[14:15]
	v_add_u32_e32 v44, 2, v38
	v_add_u32_e32 v45, 1, v38
	v_cmp_ne_u32_e64 s[14:15], 0, v39
	v_cndmask_b32_e32 v24, v41, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v39, v45, v44, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v38, v39, vcc
	v_xor_b32_e32 v38, s56, v42
	v_xor_b32_e32 v24, v24, v38
	v_sub_u32_e32 v24, v24, v38
	v_ashrrev_i32_e32 v38, 31, v27
	v_mov_b32_e32 v39, v38
	v_lshl_add_u64 v[26:27], v[26:27], 0, v[38:39]
	v_xor_b32_e32 v46, v26, v38
	v_xad_u32 v41, v24, v38, v51
	v_xor_b32_e32 v39, v27, v38
	v_mad_u64_u32 v[26:27], s[14:15], v46, s93, 0
	v_mul_hi_u32 v24, v46, s77
	v_lshl_add_u64 v[26:27], v[24:25], 0, v[26:27]
	v_mad_u64_u32 v[44:45], s[14:15], v39, s77, 0
	v_add_co_u32_e32 v24, vcc, v26, v44
	v_mad_u64_u32 v[42:43], s[14:15], v39, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v27, v45, vcc
	s_nop 1
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[26:27], v[24:25], 0, v[42:43]
	v_mul_lo_u32 v24, s87, v26
	v_mul_lo_u32 v42, s86, v27
	v_mad_u64_u32 v[26:27], s[14:15], s86, v26, 0
	v_add3_u32 v24, v27, v42, v24
	v_sub_u32_e32 v27, v39, v24
	v_sub_co_u32_e32 v26, vcc, v46, v26
	s_nop 1
	v_subb_co_u32_e64 v27, s[14:15], v27, v40, vcc
	v_subrev_co_u32_e64 v42, s[14:15], s86, v26
	v_subb_co_u32_e32 v24, vcc, v39, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[22:23], 0, v27, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s87, v43
	v_subb_co_u32_e64 v27, s[14:15], v27, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v44, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s86, v42
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v45, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s87, v43
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v26
	v_cndmask_b32_e64 v44, v44, v45, s[22:23]
	v_subrev_co_u32_e64 v45, s[14:15], s86, v42
	s_nop 1
	v_subbrev_co_u32_e64 v27, s[14:15], 0, v27, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v44
	s_nop 1
	v_cndmask_b32_e64 v27, v43, v27, s[14:15]
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v39, v39, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v39
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v27, vcc
	v_cndmask_b32_e64 v27, v42, v45, s[14:15]
	v_cndmask_b32_e32 v26, v26, v27, vcc
	v_xor_b32_e32 v26, v26, v38
	v_xor_b32_e32 v24, v24, v38
	v_sub_co_u32_e32 v26, vcc, v26, v38
	v_mul_lo_u32 v27, v41, s33
	s_nop 0
	v_subb_co_u32_e32 v24, vcc, v24, v38, vcc
	v_ashrrev_i32_e32 v24, 31, v24
	v_and_b32_e32 v24, s44, v24
	v_add3_u32 v41, v24, v26, v27
	v_lshl_add_u64 v[26:27], v[20:21], 0, v[22:23]
	s_mov_b64 s[14:15], 0x401
	v_lshl_add_u64 v[44:45], v[26:27], 0, s[14:15]
	s_movk_i32 s14, 0xfbfe
	v_lshl_add_u64 v[42:43], v[26:27], 0, s[74:75]
	v_lshl_add_u64 v[38:39], s[60:61], 0, v[18:19]
	s_mov_b32 s15, -1
	v_lshl_add_u64 v[46:47], v[38:39], 0, s[14:15]
	v_cmp_gt_i32_e64 s[22:23], 0, v43
	v_ashrrev_i32_e32 v84, 31, v43
	s_nop 0
	v_cndmask_b32_e64 v47, v45, v47, s[22:23]
	v_ashrrev_i32_e32 v76, 31, v47
	v_cndmask_b32_e64 v46, v44, v46, s[22:23]
	v_mov_b32_e32 v77, v76
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[76:77]
	v_xor_b32_e32 v77, v46, v76
	v_xor_b32_e32 v42, v47, v76
	v_mad_u64_u32 v[46:47], s[14:15], v77, s93, 0
	v_mul_hi_u32 v24, v77, s77
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[80:81], s[14:15], v42, s77, 0
	v_add_co_u32_e32 v24, vcc, v46, v80
	v_mad_u64_u32 v[78:79], s[14:15], v42, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v47, v81, vcc
	s_nop 1
	v_addc_co_u32_e32 v79, vcc, 0, v79, vcc
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[78:79]
	v_mul_lo_u32 v24, s87, v46
	v_mul_lo_u32 v47, s86, v47
	v_mad_u64_u32 v[78:79], s[14:15], s86, v46, 0
	v_add3_u32 v24, v79, v47, v24
	v_sub_u32_e32 v47, v42, v24
	v_sub_co_u32_e32 v77, vcc, v77, v78
	s_nop 1
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, vcc
	v_subrev_co_u32_e64 v78, s[14:15], s86, v77
	v_subb_co_u32_e32 v24, vcc, v42, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v47
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v78
	v_cndmask_b32_e64 v42, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v77
	v_cndmask_b32_e64 v78, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v47
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v47, v79, v78, s[14:15]
	v_add_u32_e32 v78, 2, v46
	v_add_u32_e32 v79, 1, v46
	v_cmp_ne_u32_e64 s[14:15], 0, v47
	v_cndmask_b32_e32 v24, v42, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v47, v79, v78, s[14:15]
	v_xor_b32_e32 v42, s56, v76
	v_cndmask_b32_e32 v24, v46, v47, vcc
	v_xor_b32_e32 v24, v24, v42
	v_sub_u32_e32 v24, v24, v42
	v_ashrrev_i32_e32 v42, 31, v45
	v_mov_b32_e32 v43, v42
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[42:43]
	v_xor_b32_e32 v79, v44, v42
	v_xad_u32 v78, v24, v84, v51
	v_xor_b32_e32 v43, v45, v42
	v_mad_u64_u32 v[44:45], s[14:15], v79, s93, 0
	v_mul_hi_u32 v24, v79, s77
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[44:45]
	v_mad_u64_u32 v[76:77], s[14:15], v43, s77, 0
	v_add_co_u32_e32 v24, vcc, v44, v76
	v_mad_u64_u32 v[46:47], s[14:15], v43, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v45, v77, vcc
	s_nop 1
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[46:47]
	v_mul_lo_u32 v24, s87, v44
	v_mul_lo_u32 v46, s86, v45
	v_mad_u64_u32 v[44:45], s[14:15], s86, v44, 0
	v_add3_u32 v24, v45, v46, v24
	v_sub_u32_e32 v45, v43, v24
	v_sub_co_u32_e32 v44, vcc, v79, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, vcc
	v_subrev_co_u32_e64 v46, s[14:15], s86, v44
	v_subb_co_u32_e32 v24, vcc, v43, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[28:29], 0, v45, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v47
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v76, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v46
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v77, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v47
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v44
	v_cndmask_b32_e64 v76, v76, v77, s[28:29]
	v_subrev_co_u32_e64 v77, s[14:15], s86, v46
	s_nop 1
	v_subbrev_co_u32_e64 v45, s[14:15], 0, v45, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v76
	s_nop 1
	v_cndmask_b32_e64 v45, v47, v45, s[14:15]
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v43, v43, v47, vcc
	v_cmp_ne_u32_e32 vcc, 0, v43
	v_cndmask_b32_e64 v43, v46, v77, s[14:15]
	s_mov_b64 s[14:15], 0x402
	v_cndmask_b32_e32 v24, v24, v45, vcc
	v_cndmask_b32_e32 v43, v44, v43, vcc
	v_xor_b32_e32 v44, v24, v42
	v_xor_b32_e32 v24, v43, v42
	v_sub_co_u32_e32 v24, vcc, v24, v42
	s_nop 1
	v_subb_co_u32_e32 v43, vcc, v44, v42, vcc
	v_lshl_add_u64 v[44:45], v[26:27], 0, s[14:15]
	s_movk_i32 s14, 0xfbfd
	s_mov_b32 s15, -1
	v_lshl_add_u64 v[46:47], v[38:39], 0, s[14:15]
	v_cndmask_b32_e64 v47, v45, v47, s[22:23]
	v_ashrrev_i32_e32 v76, 31, v47
	v_cndmask_b32_e64 v46, v44, v46, s[22:23]
	v_mov_b32_e32 v77, v76
	v_ashrrev_i32_e32 v42, 31, v43
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[76:77]
	v_and_b32_e32 v42, s44, v42
	v_mul_lo_u32 v43, v78, s33
	v_xor_b32_e32 v77, v46, v76
	v_add3_u32 v42, v42, v24, v43
	v_xor_b32_e32 v43, v47, v76
	v_mad_u64_u32 v[46:47], s[14:15], v77, s93, 0
	v_mul_hi_u32 v24, v77, s77
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[80:81], s[14:15], v43, s77, 0
	v_add_co_u32_e32 v24, vcc, v46, v80
	v_mad_u64_u32 v[78:79], s[14:15], v43, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v47, v81, vcc
	s_nop 1
	v_addc_co_u32_e32 v79, vcc, 0, v79, vcc
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[78:79]
	v_mul_lo_u32 v24, s87, v46
	v_mul_lo_u32 v47, s86, v47
	v_mad_u64_u32 v[78:79], s[14:15], s86, v46, 0
	v_add3_u32 v24, v79, v47, v24
	v_sub_u32_e32 v47, v43, v24
	v_sub_co_u32_e32 v77, vcc, v77, v78
	s_nop 1
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, vcc
	v_subrev_co_u32_e64 v78, s[14:15], s86, v77
	v_subb_co_u32_e32 v24, vcc, v43, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v47
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v78
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v77
	v_cndmask_b32_e64 v78, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v47
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v47, v79, v78, s[14:15]
	v_add_u32_e32 v78, 2, v46
	v_add_u32_e32 v79, 1, v46
	v_cmp_ne_u32_e64 s[14:15], 0, v47
	v_cndmask_b32_e32 v24, v43, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v47, v79, v78, s[14:15]
	v_xor_b32_e32 v43, s56, v76
	v_cndmask_b32_e32 v24, v46, v47, vcc
	v_ashrrev_i32_e32 v46, 31, v45
	v_mov_b32_e32 v47, v46
	v_xor_b32_e32 v24, v24, v43
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_sub_u32_e32 v24, v24, v43
	v_xor_b32_e32 v47, v44, v46
	v_xad_u32 v80, v24, v84, v51
	v_xor_b32_e32 v43, v45, v46
	v_mad_u64_u32 v[44:45], s[14:15], v47, s93, 0
	v_mul_hi_u32 v24, v47, s77
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[44:45]
	v_mad_u64_u32 v[78:79], s[14:15], v43, s77, 0
	v_add_co_u32_e32 v24, vcc, v44, v78
	v_mad_u64_u32 v[76:77], s[14:15], v43, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v45, v79, vcc
	s_nop 1
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[76:77]
	v_mul_lo_u32 v24, s87, v44
	v_mul_lo_u32 v76, s86, v45
	v_mad_u64_u32 v[44:45], s[14:15], s86, v44, 0
	v_add3_u32 v24, v45, v76, v24
	v_sub_u32_e32 v45, v43, v24
	v_sub_co_u32_e32 v44, vcc, v47, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, vcc
	v_subrev_co_u32_e64 v47, s[14:15], s86, v44
	v_subb_co_u32_e32 v24, vcc, v43, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v76, s[28:29], 0, v45, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v76
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v77, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v47
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v78, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v76
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v44
	v_cndmask_b32_e64 v77, v77, v78, s[28:29]
	v_subrev_co_u32_e64 v78, s[14:15], s86, v47
	s_nop 1
	v_subbrev_co_u32_e64 v45, s[14:15], 0, v45, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v77
	s_nop 1
	v_cndmask_b32_e64 v45, v76, v45, s[14:15]
	v_cndmask_b32_e64 v76, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v43, v43, v76, vcc
	v_cmp_ne_u32_e32 vcc, 0, v43
	v_cndmask_b32_e64 v43, v47, v78, s[14:15]
	s_mov_b64 s[14:15], 0x403
	v_cndmask_b32_e32 v24, v24, v45, vcc
	v_cndmask_b32_e32 v43, v44, v43, vcc
	v_xor_b32_e32 v44, v24, v46
	v_xor_b32_e32 v24, v43, v46
	v_sub_co_u32_e32 v24, vcc, v24, v46
	s_nop 1
	v_subb_co_u32_e32 v43, vcc, v44, v46, vcc
	v_ashrrev_i32_e32 v43, 31, v43
	v_and_b32_e32 v43, s44, v43
	v_mul_lo_u32 v44, v80, s33
	v_add3_u32 v43, v43, v24, v44
	v_lshl_add_u64 v[44:45], v[26:27], 0, s[14:15]
	s_movk_i32 s14, 0xfbfc
	s_mov_b32 s15, -1
	v_lshl_add_u64 v[46:47], v[38:39], 0, s[14:15]
	v_cndmask_b32_e64 v47, v45, v47, s[22:23]
	v_ashrrev_i32_e32 v76, 31, v47
	v_cndmask_b32_e64 v46, v44, v46, s[22:23]
	v_mov_b32_e32 v77, v76
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[76:77]
	v_xor_b32_e32 v82, v46, v76
	v_xor_b32_e32 v77, v47, v76
	v_mad_u64_u32 v[46:47], s[14:15], v82, s93, 0
	v_mul_hi_u32 v24, v82, s77
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[80:81], s[14:15], v77, s77, 0
	v_add_co_u32_e32 v24, vcc, v46, v80
	v_mad_u64_u32 v[78:79], s[14:15], v77, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v47, v81, vcc
	s_nop 1
	v_addc_co_u32_e32 v79, vcc, 0, v79, vcc
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[78:79]
	v_mul_lo_u32 v24, s87, v46
	v_mul_lo_u32 v47, s86, v47
	v_mad_u64_u32 v[78:79], s[14:15], s86, v46, 0
	v_add3_u32 v24, v79, v47, v24
	v_sub_u32_e32 v47, v77, v24
	v_sub_co_u32_e32 v78, vcc, v82, v78
	s_nop 1
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, vcc
	v_subrev_co_u32_e64 v79, s[14:15], s86, v78
	v_subb_co_u32_e32 v24, vcc, v77, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v47
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v80, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v79
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v78
	v_cndmask_b32_e64 v79, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v47
	v_cndmask_b32_e64 v78, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v47, v80, v79, s[14:15]
	v_add_u32_e32 v79, 2, v46
	v_add_u32_e32 v80, 1, v46
	v_cmp_ne_u32_e64 s[14:15], 0, v47
	v_cndmask_b32_e32 v24, v77, v78, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v47, v80, v79, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v46, v47, vcc
	v_xor_b32_e32 v46, s56, v76
	v_xor_b32_e32 v24, v24, v46
	v_sub_u32_e32 v24, v24, v46
	v_ashrrev_i32_e32 v46, 31, v45
	v_mov_b32_e32 v47, v46
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_xor_b32_e32 v81, v44, v46
	v_xad_u32 v80, v24, v84, v51
	v_xor_b32_e32 v47, v45, v46
	v_mad_u64_u32 v[44:45], s[14:15], v81, s93, 0
	v_mul_hi_u32 v24, v81, s77
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[44:45]
	v_mad_u64_u32 v[78:79], s[14:15], v47, s77, 0
	v_add_co_u32_e32 v24, vcc, v44, v78
	v_mad_u64_u32 v[76:77], s[14:15], v47, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v45, v79, vcc
	s_nop 1
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[76:77]
	v_mul_lo_u32 v24, s87, v44
	v_mul_lo_u32 v76, s86, v45
	v_mad_u64_u32 v[44:45], s[14:15], s86, v44, 0
	v_add3_u32 v24, v45, v76, v24
	v_sub_u32_e32 v45, v47, v24
	v_sub_co_u32_e32 v44, vcc, v81, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, vcc
	v_subrev_co_u32_e64 v76, s[14:15], s86, v44
	v_subb_co_u32_e32 v24, vcc, v47, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v77, s[28:29], 0, v45, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v77
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v78, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v76
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v77
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v44
	v_cndmask_b32_e64 v78, v78, v79, s[28:29]
	v_subrev_co_u32_e64 v79, s[14:15], s86, v76
	s_nop 1
	v_subbrev_co_u32_e64 v45, s[14:15], 0, v45, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v78
	s_nop 1
	v_cndmask_b32_e64 v45, v77, v45, s[14:15]
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v47, v47, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v47
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v45, vcc
	v_cndmask_b32_e64 v45, v76, v79, s[14:15]
	v_cndmask_b32_e32 v44, v44, v45, vcc
	v_xor_b32_e32 v45, v24, v46
	v_xor_b32_e32 v24, v44, v46
	v_sub_co_u32_e32 v24, vcc, v24, v46
	s_mov_b64 s[14:15], 0x404
	s_nop 0
	v_subb_co_u32_e32 v45, vcc, v45, v46, vcc
	v_lshl_add_u64 v[46:47], v[26:27], 0, s[14:15]
	s_movk_i32 s14, 0xfbfb
	s_mov_b32 s15, -1
	v_lshl_add_u64 v[76:77], v[38:39], 0, s[14:15]
	v_cndmask_b32_e64 v77, v47, v77, s[22:23]
	v_ashrrev_i32_e32 v78, 31, v77
	v_cndmask_b32_e64 v76, v46, v76, s[22:23]
	v_mov_b32_e32 v79, v78
	v_ashrrev_i32_e32 v44, 31, v45
	v_lshl_add_u64 v[76:77], v[76:77], 0, v[78:79]
	v_and_b32_e32 v44, s44, v44
	v_mul_lo_u32 v45, v80, s33
	v_xor_b32_e32 v79, v76, v78
	v_add3_u32 v44, v44, v24, v45
	v_xor_b32_e32 v45, v77, v78
	v_mad_u64_u32 v[76:77], s[14:15], v79, s93, 0
	v_mul_hi_u32 v24, v79, s77
	v_lshl_add_u64 v[76:77], v[24:25], 0, v[76:77]
	v_mad_u64_u32 v[82:83], s[14:15], v45, s77, 0
	v_add_co_u32_e32 v24, vcc, v76, v82
	v_mad_u64_u32 v[80:81], s[14:15], v45, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v77, v83, vcc
	s_nop 1
	v_addc_co_u32_e32 v81, vcc, 0, v81, vcc
	v_lshl_add_u64 v[76:77], v[24:25], 0, v[80:81]
	v_mul_lo_u32 v24, s87, v76
	v_mul_lo_u32 v77, s86, v77
	v_mad_u64_u32 v[80:81], s[14:15], s86, v76, 0
	v_add3_u32 v24, v81, v77, v24
	v_sub_u32_e32 v77, v45, v24
	v_sub_co_u32_e32 v79, vcc, v79, v80
	s_nop 1
	v_subb_co_u32_e64 v77, s[14:15], v77, v40, vcc
	v_subrev_co_u32_e64 v80, s[14:15], s86, v79
	v_subb_co_u32_e32 v24, vcc, v45, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v77, s[14:15], 0, v77, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v77
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v81, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v80
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v79
	v_cndmask_b32_e64 v80, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v77
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v77, v81, v80, s[14:15]
	v_add_u32_e32 v80, 2, v76
	v_add_u32_e32 v81, 1, v76
	v_cmp_ne_u32_e64 s[14:15], 0, v77
	v_cndmask_b32_e32 v24, v45, v79, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v77, v81, v80, s[14:15]
	v_xor_b32_e32 v45, s56, v78
	v_cndmask_b32_e32 v24, v76, v77, vcc
	v_ashrrev_i32_e32 v76, 31, v47
	v_mov_b32_e32 v77, v76
	v_xor_b32_e32 v24, v24, v45
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[76:77]
	v_sub_u32_e32 v24, v24, v45
	v_xor_b32_e32 v77, v46, v76
	v_xad_u32 v82, v24, v84, v51
	v_xor_b32_e32 v45, v47, v76
	v_mad_u64_u32 v[46:47], s[14:15], v77, s93, 0
	v_mul_hi_u32 v24, v77, s77
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[80:81], s[14:15], v45, s77, 0
	v_add_co_u32_e32 v24, vcc, v46, v80
	v_mad_u64_u32 v[78:79], s[14:15], v45, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v47, v81, vcc
	s_nop 1
	v_addc_co_u32_e32 v79, vcc, 0, v79, vcc
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[78:79]
	v_mul_lo_u32 v24, s87, v46
	v_mul_lo_u32 v78, s86, v47
	v_mad_u64_u32 v[46:47], s[14:15], s86, v46, 0
	v_add3_u32 v24, v47, v78, v24
	v_sub_u32_e32 v47, v45, v24
	v_sub_co_u32_e32 v46, vcc, v77, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, vcc
	v_subrev_co_u32_e64 v77, s[14:15], s86, v46
	v_subb_co_u32_e32 v24, vcc, v45, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v78, s[28:29], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v78
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v77
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v80, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v78
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v46
	v_cndmask_b32_e64 v79, v79, v80, s[28:29]
	v_subrev_co_u32_e64 v80, s[14:15], s86, v77
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v79
	s_nop 1
	v_cndmask_b32_e64 v47, v78, v47, s[14:15]
	v_cndmask_b32_e64 v78, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v45, v45, v78, vcc
	v_cmp_ne_u32_e32 vcc, 0, v45
	v_cndmask_b32_e64 v45, v77, v80, s[14:15]
	s_mov_b64 s[14:15], 0x405
	v_cndmask_b32_e32 v24, v24, v47, vcc
	v_cndmask_b32_e32 v45, v46, v45, vcc
	v_xor_b32_e32 v46, v24, v76
	v_xor_b32_e32 v24, v45, v76
	v_sub_co_u32_e32 v24, vcc, v24, v76
	s_nop 1
	v_subb_co_u32_e32 v45, vcc, v46, v76, vcc
	v_ashrrev_i32_e32 v45, 31, v45
	v_and_b32_e32 v45, s44, v45
	v_mul_lo_u32 v46, v82, s33
	v_add3_u32 v45, v45, v24, v46
	v_lshl_add_u64 v[46:47], v[26:27], 0, s[14:15]
	s_movk_i32 s14, 0xfbfa
	s_mov_b32 s15, -1
	v_lshl_add_u64 v[76:77], v[38:39], 0, s[14:15]
	v_cndmask_b32_e64 v77, v47, v77, s[22:23]
	v_ashrrev_i32_e32 v78, 31, v77
	v_cndmask_b32_e64 v76, v46, v76, s[22:23]
	v_mov_b32_e32 v79, v78
	v_lshl_add_u64 v[76:77], v[76:77], 0, v[78:79]
	v_xor_b32_e32 v85, v76, v78
	v_xor_b32_e32 v79, v77, v78
	v_mad_u64_u32 v[76:77], s[14:15], v85, s93, 0
	v_mul_hi_u32 v24, v85, s77
	v_lshl_add_u64 v[76:77], v[24:25], 0, v[76:77]
	v_mad_u64_u32 v[82:83], s[14:15], v79, s77, 0
	v_add_co_u32_e32 v24, vcc, v76, v82
	v_mad_u64_u32 v[80:81], s[14:15], v79, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v77, v83, vcc
	s_nop 1
	v_addc_co_u32_e32 v81, vcc, 0, v81, vcc
	v_lshl_add_u64 v[76:77], v[24:25], 0, v[80:81]
	v_mul_lo_u32 v24, s87, v76
	v_mul_lo_u32 v77, s86, v77
	v_mad_u64_u32 v[80:81], s[14:15], s86, v76, 0
	v_add3_u32 v24, v81, v77, v24
	v_sub_u32_e32 v77, v79, v24
	v_sub_co_u32_e32 v80, vcc, v85, v80
	s_nop 1
	v_subb_co_u32_e64 v77, s[14:15], v77, v40, vcc
	v_subrev_co_u32_e64 v81, s[14:15], s86, v80
	v_subb_co_u32_e32 v24, vcc, v79, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v77, s[14:15], 0, v77, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v77
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v82, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v81
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v80
	v_cndmask_b32_e64 v81, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v77
	v_cndmask_b32_e64 v80, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v77, v82, v81, s[14:15]
	v_add_u32_e32 v81, 2, v76
	v_add_u32_e32 v82, 1, v76
	v_cmp_ne_u32_e64 s[14:15], 0, v77
	v_cndmask_b32_e32 v24, v79, v80, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v77, v82, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v76, v77, vcc
	v_xor_b32_e32 v76, s56, v78
	v_xor_b32_e32 v24, v24, v76
	v_sub_u32_e32 v24, v24, v76
	v_ashrrev_i32_e32 v76, 31, v47
	v_mov_b32_e32 v77, v76
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[76:77]
	v_xor_b32_e32 v83, v46, v76
	v_xad_u32 v82, v24, v84, v51
	v_xor_b32_e32 v77, v47, v76
	v_mad_u64_u32 v[46:47], s[14:15], v83, s93, 0
	v_mul_hi_u32 v24, v83, s77
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[80:81], s[14:15], v77, s77, 0
	v_add_co_u32_e32 v24, vcc, v46, v80
	v_mad_u64_u32 v[78:79], s[14:15], v77, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v47, v81, vcc
	s_nop 1
	v_addc_co_u32_e32 v79, vcc, 0, v79, vcc
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[78:79]
	v_mul_lo_u32 v24, s87, v46
	v_mul_lo_u32 v78, s86, v47
	v_mad_u64_u32 v[46:47], s[14:15], s86, v46, 0
	v_add3_u32 v24, v47, v78, v24
	v_sub_u32_e32 v47, v77, v24
	v_sub_co_u32_e32 v46, vcc, v83, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, vcc
	v_subrev_co_u32_e64 v78, s[14:15], s86, v46
	v_subb_co_u32_e32 v24, vcc, v77, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v79, s[28:29], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v79
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v80, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v78
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v81, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v79
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v46
	v_cndmask_b32_e64 v80, v80, v81, s[28:29]
	v_subrev_co_u32_e64 v81, s[14:15], s86, v78
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v80
	s_nop 1
	v_cndmask_b32_e64 v47, v79, v47, s[14:15]
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v77, v77, v79, vcc
	v_cmp_ne_u32_e32 vcc, 0, v77
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v47, vcc
	v_cndmask_b32_e64 v47, v78, v81, s[14:15]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v24, v76
	v_xor_b32_e32 v24, v46, v76
	v_sub_co_u32_e32 v24, vcc, v24, v76
	s_mov_b64 s[14:15], 0x406
	s_nop 0
	v_subb_co_u32_e32 v47, vcc, v47, v76, vcc
	v_lshl_add_u64 v[76:77], v[26:27], 0, s[14:15]
	s_movk_i32 s14, 0xfbf9
	s_mov_b32 s15, -1
	v_lshl_add_u64 v[78:79], v[38:39], 0, s[14:15]
	v_cndmask_b32_e64 v79, v77, v79, s[22:23]
	v_ashrrev_i32_e32 v80, 31, v79
	v_cndmask_b32_e64 v78, v76, v78, s[22:23]
	v_mov_b32_e32 v81, v80
	v_ashrrev_i32_e32 v46, 31, v47
	v_lshl_add_u64 v[78:79], v[78:79], 0, v[80:81]
	v_and_b32_e32 v46, s44, v46
	v_mul_lo_u32 v47, v82, s33
	v_xor_b32_e32 v81, v78, v80
	v_add3_u32 v46, v46, v24, v47
	v_xor_b32_e32 v47, v79, v80
	v_mad_u64_u32 v[78:79], s[14:15], v81, s93, 0
	v_mul_hi_u32 v24, v81, s77
	v_lshl_add_u64 v[78:79], v[24:25], 0, v[78:79]
	v_mad_u64_u32 v[86:87], s[14:15], v47, s77, 0
	v_add_co_u32_e32 v24, vcc, v78, v86
	v_mad_u64_u32 v[82:83], s[14:15], v47, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v79, v87, vcc
	s_nop 1
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[78:79], v[24:25], 0, v[82:83]
	v_mul_lo_u32 v24, s87, v78
	v_mul_lo_u32 v79, s86, v79
	v_mad_u64_u32 v[82:83], s[14:15], s86, v78, 0
	v_add3_u32 v24, v83, v79, v24
	v_sub_u32_e32 v79, v47, v24
	v_sub_co_u32_e32 v81, vcc, v81, v82
	s_nop 1
	v_subb_co_u32_e64 v79, s[14:15], v79, v40, vcc
	v_subrev_co_u32_e64 v82, s[14:15], s86, v81
	v_subb_co_u32_e32 v24, vcc, v47, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v79, s[14:15], 0, v79, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v79
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v82
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v81
	v_cndmask_b32_e64 v82, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v79
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v79, v83, v82, s[14:15]
	v_add_u32_e32 v82, 2, v78
	v_add_u32_e32 v83, 1, v78
	v_cmp_ne_u32_e64 s[14:15], 0, v79
	v_cndmask_b32_e32 v24, v47, v81, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v79, v83, v82, s[14:15]
	v_xor_b32_e32 v47, s56, v80
	v_cndmask_b32_e32 v24, v78, v79, vcc
	v_ashrrev_i32_e32 v78, 31, v77
	v_mov_b32_e32 v79, v78
	v_xor_b32_e32 v24, v24, v47
	v_lshl_add_u64 v[76:77], v[76:77], 0, v[78:79]
	v_sub_u32_e32 v24, v24, v47
	v_xor_b32_e32 v79, v76, v78
	v_xad_u32 v85, v24, v84, v51
	v_xor_b32_e32 v47, v77, v78
	v_mad_u64_u32 v[76:77], s[14:15], v79, s93, 0
	v_mul_hi_u32 v24, v79, s77
	v_lshl_add_u64 v[76:77], v[24:25], 0, v[76:77]
	v_mad_u64_u32 v[82:83], s[14:15], v47, s77, 0
	v_add_co_u32_e32 v24, vcc, v76, v82
	v_mad_u64_u32 v[80:81], s[14:15], v47, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v77, v83, vcc
	s_nop 1
	v_addc_co_u32_e32 v81, vcc, 0, v81, vcc
	v_lshl_add_u64 v[76:77], v[24:25], 0, v[80:81]
	v_mul_lo_u32 v24, s87, v76
	v_mul_lo_u32 v80, s86, v77
	v_mad_u64_u32 v[76:77], s[14:15], s86, v76, 0
	v_add3_u32 v24, v77, v80, v24
	v_sub_u32_e32 v77, v47, v24
	v_sub_co_u32_e32 v76, vcc, v79, v76
	s_nop 1
	v_subb_co_u32_e64 v77, s[14:15], v77, v40, vcc
	v_subrev_co_u32_e64 v79, s[14:15], s86, v76
	v_subb_co_u32_e32 v24, vcc, v47, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v80, s[28:29], 0, v77, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v80
	v_subb_co_u32_e64 v77, s[14:15], v77, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v81, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v79
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v82, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v80
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v76
	v_cndmask_b32_e64 v81, v81, v82, s[28:29]
	v_subrev_co_u32_e64 v82, s[14:15], s86, v79
	s_nop 1
	v_subbrev_co_u32_e64 v77, s[14:15], 0, v77, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v81
	s_nop 1
	v_cndmask_b32_e64 v77, v80, v77, s[14:15]
	v_cndmask_b32_e64 v80, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v47, v47, v80, vcc
	v_cmp_ne_u32_e32 vcc, 0, v47
	v_cndmask_b32_e64 v47, v79, v82, s[14:15]
	s_mov_b64 s[14:15], 0x407
	v_cndmask_b32_e32 v24, v24, v77, vcc
	v_cndmask_b32_e32 v47, v76, v47, vcc
	v_xor_b32_e32 v76, v24, v78
	v_xor_b32_e32 v24, v47, v78
	v_sub_co_u32_e32 v24, vcc, v24, v78
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v76, v78, vcc
	v_ashrrev_i32_e32 v47, 31, v47
	v_and_b32_e32 v47, s44, v47
	v_mul_lo_u32 v76, v85, s33
	v_add3_u32 v85, v47, v24, v76
	v_lshl_add_u64 v[76:77], v[26:27], 0, s[14:15]
	s_movk_i32 s14, 0xfbf8
	s_mov_b32 s15, -1
	v_lshl_add_u64 v[78:79], v[38:39], 0, s[14:15]
	v_cndmask_b32_e64 v79, v77, v79, s[22:23]
	v_ashrrev_i32_e32 v80, 31, v79
	v_cndmask_b32_e64 v78, v76, v78, s[22:23]
	v_mov_b32_e32 v81, v80
	v_lshl_add_u64 v[78:79], v[78:79], 0, v[80:81]
	v_xor_b32_e32 v81, v78, v80
	v_xor_b32_e32 v47, v79, v80
	v_mad_u64_u32 v[78:79], s[14:15], v81, s93, 0
	v_mul_hi_u32 v24, v81, s77
	v_lshl_add_u64 v[78:79], v[24:25], 0, v[78:79]
	v_mad_u64_u32 v[86:87], s[14:15], v47, s77, 0
	v_add_co_u32_e32 v24, vcc, v78, v86
	v_mad_u64_u32 v[82:83], s[14:15], v47, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v79, v87, vcc
	s_nop 1
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[78:79], v[24:25], 0, v[82:83]
	v_mul_lo_u32 v24, s87, v78
	v_mul_lo_u32 v79, s86, v79
	v_mad_u64_u32 v[82:83], s[14:15], s86, v78, 0
	v_add3_u32 v24, v83, v79, v24
	v_sub_u32_e32 v79, v47, v24
	v_sub_co_u32_e32 v81, vcc, v81, v82
	s_nop 1
	v_subb_co_u32_e64 v79, s[14:15], v79, v40, vcc
	v_subrev_co_u32_e64 v82, s[14:15], s86, v81
	v_subb_co_u32_e32 v24, vcc, v47, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v79, s[14:15], 0, v79, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v79
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v82
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v81
	v_cndmask_b32_e64 v82, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v79
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v79, v83, v82, s[14:15]
	v_add_u32_e32 v82, 2, v78
	v_add_u32_e32 v83, 1, v78
	v_cmp_ne_u32_e64 s[14:15], 0, v79
	v_cndmask_b32_e32 v24, v47, v81, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v79, v83, v82, s[14:15]
	v_xor_b32_e32 v47, s56, v80
	v_cndmask_b32_e32 v24, v78, v79, vcc
	v_ashrrev_i32_e32 v78, 31, v77
	v_mov_b32_e32 v79, v78
	v_xor_b32_e32 v24, v24, v47
	v_lshl_add_u64 v[76:77], v[76:77], 0, v[78:79]
	v_sub_u32_e32 v24, v24, v47
	v_xor_b32_e32 v79, v76, v78
	v_xad_u32 v86, v24, v84, v51
	v_xor_b32_e32 v47, v77, v78
	v_mad_u64_u32 v[76:77], s[14:15], v79, s93, 0
	v_mul_hi_u32 v24, v79, s77
	v_lshl_add_u64 v[76:77], v[24:25], 0, v[76:77]
	v_mad_u64_u32 v[82:83], s[14:15], v47, s77, 0
	v_add_co_u32_e32 v24, vcc, v76, v82
	v_mad_u64_u32 v[80:81], s[14:15], v47, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v77, v83, vcc
	s_nop 1
	v_addc_co_u32_e32 v81, vcc, 0, v81, vcc
	v_lshl_add_u64 v[76:77], v[24:25], 0, v[80:81]
	v_mul_lo_u32 v24, s87, v76
	v_mul_lo_u32 v80, s86, v77
	v_mad_u64_u32 v[76:77], s[14:15], s86, v76, 0
	v_add3_u32 v24, v77, v80, v24
	v_sub_u32_e32 v77, v47, v24
	v_sub_co_u32_e32 v76, vcc, v79, v76
	s_nop 1
	v_subb_co_u32_e64 v77, s[14:15], v77, v40, vcc
	v_subrev_co_u32_e64 v79, s[14:15], s86, v76
	v_subb_co_u32_e32 v24, vcc, v47, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v80, s[28:29], 0, v77, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v80
	v_subb_co_u32_e64 v77, s[14:15], v77, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v81, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v79
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v82, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v80
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v76
	v_cndmask_b32_e64 v81, v81, v82, s[28:29]
	v_subrev_co_u32_e64 v82, s[14:15], s86, v79
	s_nop 1
	v_subbrev_co_u32_e64 v77, s[14:15], 0, v77, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v81
	s_nop 1
	v_cndmask_b32_e64 v77, v80, v77, s[14:15]
	v_cndmask_b32_e64 v80, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v47, v47, v80, vcc
	v_cmp_ne_u32_e32 vcc, 0, v47
	v_cndmask_b32_e64 v47, v79, v82, s[14:15]
	s_mov_b64 s[14:15], 0x408
	v_cndmask_b32_e32 v24, v24, v77, vcc
	v_cndmask_b32_e32 v47, v76, v47, vcc
	v_xor_b32_e32 v76, v24, v78
	v_xor_b32_e32 v24, v47, v78
	v_sub_co_u32_e32 v24, vcc, v24, v78
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v76, v78, vcc
	v_ashrrev_i32_e32 v47, 31, v47
	v_and_b32_e32 v47, s44, v47
	v_mul_lo_u32 v76, v86, s33
	v_add3_u32 v24, v47, v24, v76
	buffer_load_ubyte v77, v41, s[36:39], 0 offen
	buffer_load_ubyte v79, v42, s[36:39], 0 offen
	buffer_load_ubyte v76, v43, s[36:39], 0 offen
	buffer_load_ubyte v78, v44, s[36:39], 0 offen
	buffer_load_ubyte v80, v45, s[36:39], 0 offen
	buffer_load_ubyte v82, v46, s[36:39], 0 offen
	buffer_load_ubyte v81, v85, s[36:39], 0 offen
	buffer_load_ubyte v83, v24, s[36:39], 0 offen
	v_lshl_add_u64 v[42:43], v[26:27], 0, s[14:15]
	s_movk_i32 s14, 0xfbf7
	s_mov_b32 s15, -1
	v_lshl_add_u64 v[44:45], v[38:39], 0, s[14:15]
	v_cndmask_b32_e64 v45, v43, v45, s[22:23]
	v_ashrrev_i32_e32 v46, 31, v45
	v_cndmask_b32_e64 v44, v42, v44, s[22:23]
	v_mov_b32_e32 v47, v46
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_xor_b32_e32 v47, v44, v46
	v_xor_b32_e32 v41, v45, v46
	v_mad_u64_u32 v[44:45], s[14:15], v47, s93, 0
	v_mul_hi_u32 v24, v47, s77
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[44:45]
	v_mad_u64_u32 v[88:89], s[14:15], v41, s77, 0
	v_add_co_u32_e32 v24, vcc, v44, v88
	v_mad_u64_u32 v[86:87], s[14:15], v41, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v45, v89, vcc
	s_nop 1
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[86:87]
	v_mul_lo_u32 v24, s87, v44
	v_mul_lo_u32 v45, s86, v45
	v_mad_u64_u32 v[86:87], s[14:15], s86, v44, 0
	v_add3_u32 v24, v87, v45, v24
	v_sub_u32_e32 v45, v41, v24
	v_sub_co_u32_e32 v47, vcc, v47, v86
	s_nop 1
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, vcc
	v_subrev_co_u32_e64 v85, s[14:15], s86, v47
	v_subb_co_u32_e32 v24, vcc, v41, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v45, s[14:15], 0, v45, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v45
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v86, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v85
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v47
	v_cndmask_b32_e64 v85, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v45
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v45, v86, v85, s[14:15]
	v_add_u32_e32 v85, 2, v44
	v_add_u32_e32 v86, 1, v44
	v_cmp_ne_u32_e64 s[14:15], 0, v45
	v_cndmask_b32_e32 v24, v41, v47, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v45, v86, v85, s[14:15]
	v_xor_b32_e32 v41, s56, v46
	v_cndmask_b32_e32 v24, v44, v45, vcc
	v_ashrrev_i32_e32 v44, 31, v43
	v_mov_b32_e32 v45, v44
	v_xor_b32_e32 v24, v24, v41
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[44:45]
	v_sub_u32_e32 v24, v24, v41
	v_xor_b32_e32 v45, v42, v44
	v_xad_u32 v85, v24, v84, v51
	v_xor_b32_e32 v41, v43, v44
	v_mad_u64_u32 v[42:43], s[14:15], v45, s93, 0
	v_mul_hi_u32 v24, v45, s77
	v_lshl_add_u64 v[42:43], v[24:25], 0, v[42:43]
	v_mad_u64_u32 v[86:87], s[14:15], v41, s77, 0
	v_add_co_u32_e32 v24, vcc, v42, v86
	v_mad_u64_u32 v[46:47], s[14:15], v41, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v43, v87, vcc
	s_nop 1
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[24:25], 0, v[46:47]
	v_mul_lo_u32 v24, s87, v42
	v_mul_lo_u32 v46, s86, v43
	v_mad_u64_u32 v[42:43], s[14:15], s86, v42, 0
	v_add3_u32 v24, v43, v46, v24
	v_sub_u32_e32 v43, v41, v24
	v_sub_co_u32_e32 v42, vcc, v45, v42
	s_nop 1
	v_subb_co_u32_e64 v43, s[14:15], v43, v40, vcc
	v_subrev_co_u32_e64 v45, s[14:15], s86, v42
	v_subb_co_u32_e32 v24, vcc, v41, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v46, s[28:29], 0, v43, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v46
	v_subb_co_u32_e64 v43, s[14:15], v43, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v45
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v86, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v46
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v42
	v_cndmask_b32_e64 v47, v47, v86, s[28:29]
	v_subrev_co_u32_e64 v86, s[14:15], s86, v45
	s_nop 1
	v_subbrev_co_u32_e64 v43, s[14:15], 0, v43, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v47
	s_nop 1
	v_cndmask_b32_e64 v43, v46, v43, s[14:15]
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v41, v41, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v41
	v_cndmask_b32_e64 v41, v45, v86, s[14:15]
	s_mov_b64 s[14:15], 0x409
	v_cndmask_b32_e32 v24, v24, v43, vcc
	v_cndmask_b32_e32 v41, v42, v41, vcc
	v_xor_b32_e32 v42, v24, v44
	v_xor_b32_e32 v24, v41, v44
	v_sub_co_u32_e32 v24, vcc, v24, v44
	s_nop 1
	v_subb_co_u32_e32 v41, vcc, v42, v44, vcc
	v_ashrrev_i32_e32 v41, 31, v41
	v_and_b32_e32 v41, s44, v41
	v_mul_lo_u32 v42, v85, s33
	v_add3_u32 v41, v41, v24, v42
	v_lshl_add_u64 v[42:43], v[26:27], 0, s[14:15]
	s_movk_i32 s14, 0xfbf6
	s_mov_b32 s15, -1
	v_lshl_add_u64 v[44:45], v[38:39], 0, s[14:15]
	v_cndmask_b32_e64 v45, v43, v45, s[22:23]
	v_ashrrev_i32_e32 v46, 31, v45
	v_cndmask_b32_e64 v44, v42, v44, s[22:23]
	v_mov_b32_e32 v47, v46
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_xor_b32_e32 v85, v44, v46
	v_xor_b32_e32 v47, v45, v46
	v_mad_u64_u32 v[44:45], s[14:15], v85, s93, 0
	v_mul_hi_u32 v24, v85, s77
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[44:45]
	v_mad_u64_u32 v[88:89], s[14:15], v47, s77, 0
	v_add_co_u32_e32 v24, vcc, v44, v88
	v_mad_u64_u32 v[86:87], s[14:15], v47, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v45, v89, vcc
	s_nop 1
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[86:87]
	v_mul_lo_u32 v24, s87, v44
	v_mul_lo_u32 v45, s86, v45
	v_mad_u64_u32 v[86:87], s[14:15], s86, v44, 0
	v_add3_u32 v24, v87, v45, v24
	v_sub_u32_e32 v45, v47, v24
	v_sub_co_u32_e32 v85, vcc, v85, v86
	s_nop 1
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, vcc
	v_subrev_co_u32_e64 v86, s[14:15], s86, v85
	v_subb_co_u32_e32 v24, vcc, v47, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v45, s[14:15], 0, v45, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v45
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v86
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v85
	v_cndmask_b32_e64 v86, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v45
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v45, v87, v86, s[14:15]
	v_add_u32_e32 v86, 2, v44
	v_add_u32_e32 v87, 1, v44
	v_cmp_ne_u32_e64 s[14:15], 0, v45
	v_cndmask_b32_e32 v24, v47, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v45, v87, v86, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v44, v45, vcc
	v_xor_b32_e32 v44, s56, v46
	v_xor_b32_e32 v24, v24, v44
	v_sub_u32_e32 v24, v24, v44
	v_ashrrev_i32_e32 v44, 31, v43
	v_mov_b32_e32 v45, v44
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[44:45]
	v_xor_b32_e32 v88, v42, v44
	v_xad_u32 v85, v24, v84, v51
	v_xor_b32_e32 v45, v43, v44
	v_mad_u64_u32 v[42:43], s[14:15], v88, s93, 0
	v_mul_hi_u32 v24, v88, s77
	v_lshl_add_u64 v[42:43], v[24:25], 0, v[42:43]
	v_mad_u64_u32 v[86:87], s[14:15], v45, s77, 0
	v_add_co_u32_e32 v24, vcc, v42, v86
	v_mad_u64_u32 v[46:47], s[14:15], v45, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v43, v87, vcc
	s_nop 1
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[24:25], 0, v[46:47]
	v_mul_lo_u32 v24, s87, v42
	v_mul_lo_u32 v46, s86, v43
	v_mad_u64_u32 v[42:43], s[14:15], s86, v42, 0
	v_add3_u32 v24, v43, v46, v24
	v_sub_u32_e32 v43, v45, v24
	v_sub_co_u32_e32 v42, vcc, v88, v42
	s_nop 1
	v_subb_co_u32_e64 v43, s[14:15], v43, v40, vcc
	v_subrev_co_u32_e64 v46, s[14:15], s86, v42
	v_subb_co_u32_e32 v24, vcc, v45, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[28:29], 0, v43, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v47
	v_subb_co_u32_e64 v43, s[14:15], v43, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v86, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v46
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v47
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v42
	v_cndmask_b32_e64 v86, v86, v87, s[28:29]
	v_subrev_co_u32_e64 v87, s[14:15], s86, v46
	s_nop 1
	v_subbrev_co_u32_e64 v43, s[14:15], 0, v43, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v86
	s_nop 1
	v_cndmask_b32_e64 v43, v47, v43, s[14:15]
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v45, v45, v47, vcc
	v_cmp_ne_u32_e32 vcc, 0, v45
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v43, vcc
	v_cndmask_b32_e64 v43, v46, v87, s[14:15]
	v_cndmask_b32_e32 v42, v42, v43, vcc
	v_xor_b32_e32 v43, v24, v44
	v_xor_b32_e32 v24, v42, v44
	v_sub_co_u32_e32 v24, vcc, v24, v44
	s_mov_b64 s[14:15], 0x40a
	s_nop 0
	v_subb_co_u32_e32 v43, vcc, v43, v44, vcc
	v_lshl_add_u64 v[44:45], v[26:27], 0, s[14:15]
	s_movk_i32 s14, 0xfbf5
	s_mov_b32 s15, -1
	v_lshl_add_u64 v[46:47], v[38:39], 0, s[14:15]
	v_cndmask_b32_e64 v47, v45, v47, s[22:23]
	v_ashrrev_i32_e32 v86, 31, v47
	v_cndmask_b32_e64 v46, v44, v46, s[22:23]
	v_mov_b32_e32 v87, v86
	v_ashrrev_i32_e32 v42, 31, v43
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[86:87]
	v_and_b32_e32 v42, s44, v42
	v_mul_lo_u32 v43, v85, s33
	v_xor_b32_e32 v85, v46, v86
	v_add3_u32 v42, v42, v24, v43
	v_xor_b32_e32 v43, v47, v86
	v_mad_u64_u32 v[46:47], s[14:15], v85, s93, 0
	v_mul_hi_u32 v24, v85, s77
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[90:91], s[14:15], v43, s77, 0
	v_add_co_u32_e32 v24, vcc, v46, v90
	v_mad_u64_u32 v[88:89], s[14:15], v43, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v47, v91, vcc
	s_nop 1
	v_addc_co_u32_e32 v89, vcc, 0, v89, vcc
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[88:89]
	v_mul_lo_u32 v24, s87, v46
	v_mul_lo_u32 v47, s86, v47
	v_mad_u64_u32 v[88:89], s[14:15], s86, v46, 0
	v_add3_u32 v24, v89, v47, v24
	v_sub_u32_e32 v47, v43, v24
	v_sub_co_u32_e32 v85, vcc, v85, v88
	s_nop 1
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, vcc
	v_subrev_co_u32_e64 v87, s[14:15], s86, v85
	v_subb_co_u32_e32 v24, vcc, v43, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v47
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v88, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v87
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v85
	v_cndmask_b32_e64 v87, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v47
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v47, v88, v87, s[14:15]
	v_add_u32_e32 v87, 2, v46
	v_add_u32_e32 v88, 1, v46
	v_cmp_ne_u32_e64 s[14:15], 0, v47
	v_cndmask_b32_e32 v24, v43, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v47, v88, v87, s[14:15]
	v_xor_b32_e32 v43, s56, v86
	v_cndmask_b32_e32 v24, v46, v47, vcc
	v_ashrrev_i32_e32 v46, 31, v45
	v_mov_b32_e32 v47, v46
	v_xor_b32_e32 v24, v24, v43
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_sub_u32_e32 v24, v24, v43
	v_xor_b32_e32 v47, v44, v46
	v_xad_u32 v85, v24, v84, v51
	v_xor_b32_e32 v43, v45, v46
	v_mad_u64_u32 v[44:45], s[14:15], v47, s93, 0
	v_mul_hi_u32 v24, v47, s77
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[44:45]
	v_mad_u64_u32 v[88:89], s[14:15], v43, s77, 0
	v_add_co_u32_e32 v24, vcc, v44, v88
	v_mad_u64_u32 v[86:87], s[14:15], v43, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v45, v89, vcc
	s_nop 1
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[86:87]
	v_mul_lo_u32 v24, s87, v44
	v_mul_lo_u32 v86, s86, v45
	v_mad_u64_u32 v[44:45], s[14:15], s86, v44, 0
	v_add3_u32 v24, v45, v86, v24
	v_sub_u32_e32 v45, v43, v24
	v_sub_co_u32_e32 v44, vcc, v47, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, vcc
	v_subrev_co_u32_e64 v47, s[14:15], s86, v44
	v_subb_co_u32_e32 v24, vcc, v43, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v86, s[28:29], 0, v45, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v86
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v47
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v88, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v86
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v44
	v_cndmask_b32_e64 v87, v87, v88, s[28:29]
	v_subrev_co_u32_e64 v88, s[14:15], s86, v47
	s_nop 1
	v_subbrev_co_u32_e64 v45, s[14:15], 0, v45, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v87
	s_nop 1
	v_cndmask_b32_e64 v45, v86, v45, s[14:15]
	v_cndmask_b32_e64 v86, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v43, v43, v86, vcc
	v_cmp_ne_u32_e32 vcc, 0, v43
	v_cndmask_b32_e64 v43, v47, v88, s[14:15]
	s_mov_b64 s[14:15], 0x40b
	v_cndmask_b32_e32 v24, v24, v45, vcc
	v_cndmask_b32_e32 v43, v44, v43, vcc
	v_xor_b32_e32 v44, v24, v46
	v_xor_b32_e32 v24, v43, v46
	v_sub_co_u32_e32 v24, vcc, v24, v46
	s_nop 1
	v_subb_co_u32_e32 v43, vcc, v44, v46, vcc
	v_ashrrev_i32_e32 v43, 31, v43
	v_and_b32_e32 v43, s44, v43
	v_mul_lo_u32 v44, v85, s33
	v_add3_u32 v43, v43, v24, v44
	v_lshl_add_u64 v[44:45], v[26:27], 0, s[14:15]
	s_movk_i32 s14, 0xfbf4
	s_mov_b32 s15, -1
	v_lshl_add_u64 v[46:47], v[38:39], 0, s[14:15]
	v_cndmask_b32_e64 v47, v45, v47, s[22:23]
	v_ashrrev_i32_e32 v86, 31, v47
	v_cndmask_b32_e64 v46, v44, v46, s[22:23]
	v_mov_b32_e32 v87, v86
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[86:87]
	v_xor_b32_e32 v87, v46, v86
	v_xor_b32_e32 v85, v47, v86
	v_mad_u64_u32 v[46:47], s[14:15], v87, s93, 0
	v_mul_hi_u32 v24, v87, s77
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[90:91], s[14:15], v85, s77, 0
	v_add_co_u32_e32 v24, vcc, v46, v90
	v_mad_u64_u32 v[88:89], s[14:15], v85, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v47, v91, vcc
	s_nop 1
	v_addc_co_u32_e32 v89, vcc, 0, v89, vcc
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[88:89]
	v_mul_lo_u32 v24, s87, v46
	v_mul_lo_u32 v47, s86, v47
	v_mad_u64_u32 v[88:89], s[14:15], s86, v46, 0
	v_add3_u32 v24, v89, v47, v24
	v_sub_u32_e32 v47, v85, v24
	v_sub_co_u32_e32 v87, vcc, v87, v88
	s_nop 1
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, vcc
	v_subrev_co_u32_e64 v88, s[14:15], s86, v87
	v_subb_co_u32_e32 v24, vcc, v85, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v47
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v89, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v88
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v87
	v_cndmask_b32_e64 v88, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v47
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v47, v89, v88, s[14:15]
	v_add_u32_e32 v88, 2, v46
	v_add_u32_e32 v89, 1, v46
	v_cmp_ne_u32_e64 s[14:15], 0, v47
	v_cndmask_b32_e32 v24, v85, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v47, v89, v88, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v46, v47, vcc
	v_xor_b32_e32 v46, s56, v86
	v_xor_b32_e32 v24, v24, v46
	v_sub_u32_e32 v24, v24, v46
	v_ashrrev_i32_e32 v46, 31, v45
	v_mov_b32_e32 v47, v46
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_xor_b32_e32 v90, v44, v46
	v_xad_u32 v85, v24, v84, v51
	v_xor_b32_e32 v47, v45, v46
	v_mad_u64_u32 v[44:45], s[14:15], v90, s93, 0
	v_mul_hi_u32 v24, v90, s77
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[44:45]
	v_mad_u64_u32 v[88:89], s[14:15], v47, s77, 0
	v_add_co_u32_e32 v24, vcc, v44, v88
	v_mad_u64_u32 v[86:87], s[14:15], v47, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v45, v89, vcc
	s_nop 1
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[44:45], v[24:25], 0, v[86:87]
	v_mul_lo_u32 v24, s87, v44
	v_mul_lo_u32 v86, s86, v45
	v_mad_u64_u32 v[44:45], s[14:15], s86, v44, 0
	v_add3_u32 v24, v45, v86, v24
	v_sub_u32_e32 v45, v47, v24
	v_sub_co_u32_e32 v44, vcc, v90, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, vcc
	v_subrev_co_u32_e64 v86, s[14:15], s86, v44
	v_subb_co_u32_e32 v24, vcc, v47, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v87, s[28:29], 0, v45, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v87
	v_subb_co_u32_e64 v45, s[14:15], v45, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v88, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v86
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v89, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v87
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v44
	v_cndmask_b32_e64 v88, v88, v89, s[28:29]
	v_subrev_co_u32_e64 v89, s[14:15], s86, v86
	s_nop 1
	v_subbrev_co_u32_e64 v45, s[14:15], 0, v45, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v88
	s_nop 1
	v_cndmask_b32_e64 v45, v87, v45, s[14:15]
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v47, v47, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v47
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v45, vcc
	v_cndmask_b32_e64 v45, v86, v89, s[14:15]
	v_cndmask_b32_e32 v44, v44, v45, vcc
	v_xor_b32_e32 v45, v24, v46
	v_xor_b32_e32 v24, v44, v46
	v_sub_co_u32_e32 v24, vcc, v24, v46
	s_mov_b64 s[14:15], 0x40c
	s_nop 0
	v_subb_co_u32_e32 v45, vcc, v45, v46, vcc
	v_lshl_add_u64 v[46:47], v[26:27], 0, s[14:15]
	s_movk_i32 s14, 0xfbf3
	s_mov_b32 s15, -1
	v_lshl_add_u64 v[86:87], v[38:39], 0, s[14:15]
	v_cndmask_b32_e64 v87, v47, v87, s[22:23]
	v_ashrrev_i32_e32 v88, 31, v87
	v_cndmask_b32_e64 v86, v46, v86, s[22:23]
	v_mov_b32_e32 v89, v88
	v_ashrrev_i32_e32 v44, 31, v45
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[88:89]
	v_and_b32_e32 v44, s44, v44
	v_mul_lo_u32 v45, v85, s33
	v_xor_b32_e32 v85, v86, v88
	v_add3_u32 v44, v44, v24, v45
	v_xor_b32_e32 v45, v87, v88
	v_mad_u64_u32 v[86:87], s[14:15], v85, s93, 0
	v_mul_hi_u32 v24, v85, s77
	v_lshl_add_u64 v[86:87], v[24:25], 0, v[86:87]
	v_mad_u64_u32 v[92:93], s[14:15], v45, s77, 0
	v_add_co_u32_e32 v24, vcc, v86, v92
	v_mad_u64_u32 v[90:91], s[14:15], v45, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v87, v93, vcc
	s_nop 1
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[86:87], v[24:25], 0, v[90:91]
	v_mul_lo_u32 v24, s87, v86
	v_mul_lo_u32 v87, s86, v87
	v_mad_u64_u32 v[90:91], s[14:15], s86, v86, 0
	v_add3_u32 v24, v91, v87, v24
	v_sub_u32_e32 v87, v45, v24
	v_sub_co_u32_e32 v85, vcc, v85, v90
	s_nop 1
	v_subb_co_u32_e64 v87, s[14:15], v87, v40, vcc
	v_subrev_co_u32_e64 v89, s[14:15], s86, v85
	v_subb_co_u32_e32 v24, vcc, v45, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v87, s[14:15], 0, v87, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v87
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v90, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v89
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v85
	v_cndmask_b32_e64 v89, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v87
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v87, v90, v89, s[14:15]
	v_add_u32_e32 v89, 2, v86
	v_add_u32_e32 v90, 1, v86
	v_cmp_ne_u32_e64 s[14:15], 0, v87
	v_cndmask_b32_e32 v24, v45, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v87, v90, v89, s[14:15]
	v_xor_b32_e32 v45, s56, v88
	v_cndmask_b32_e32 v24, v86, v87, vcc
	v_ashrrev_i32_e32 v86, 31, v47
	v_mov_b32_e32 v87, v86
	v_xor_b32_e32 v24, v24, v45
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[86:87]
	v_sub_u32_e32 v24, v24, v45
	v_xor_b32_e32 v87, v46, v86
	v_xad_u32 v85, v24, v84, v51
	v_xor_b32_e32 v45, v47, v86
	v_mad_u64_u32 v[46:47], s[14:15], v87, s93, 0
	v_mul_hi_u32 v24, v87, s77
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[90:91], s[14:15], v45, s77, 0
	v_add_co_u32_e32 v24, vcc, v46, v90
	v_mad_u64_u32 v[88:89], s[14:15], v45, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v47, v91, vcc
	s_nop 1
	v_addc_co_u32_e32 v89, vcc, 0, v89, vcc
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[88:89]
	v_mul_lo_u32 v24, s87, v46
	v_mul_lo_u32 v88, s86, v47
	v_mad_u64_u32 v[46:47], s[14:15], s86, v46, 0
	v_add3_u32 v24, v47, v88, v24
	v_sub_u32_e32 v47, v45, v24
	v_sub_co_u32_e32 v46, vcc, v87, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, vcc
	v_subrev_co_u32_e64 v87, s[14:15], s86, v46
	v_subb_co_u32_e32 v24, vcc, v45, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v88, s[28:29], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v88
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v89, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v87
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v90, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v88
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v46
	v_cndmask_b32_e64 v89, v89, v90, s[28:29]
	v_subrev_co_u32_e64 v90, s[14:15], s86, v87
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v89
	s_nop 1
	v_cndmask_b32_e64 v47, v88, v47, s[14:15]
	v_cndmask_b32_e64 v88, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v45, v45, v88, vcc
	v_cmp_ne_u32_e32 vcc, 0, v45
	v_cndmask_b32_e64 v45, v87, v90, s[14:15]
	s_mov_b64 s[14:15], 0x40d
	v_cndmask_b32_e32 v24, v24, v47, vcc
	v_cndmask_b32_e32 v45, v46, v45, vcc
	v_xor_b32_e32 v46, v24, v86
	v_xor_b32_e32 v24, v45, v86
	v_sub_co_u32_e32 v24, vcc, v24, v86
	s_nop 1
	v_subb_co_u32_e32 v45, vcc, v46, v86, vcc
	v_ashrrev_i32_e32 v45, 31, v45
	v_and_b32_e32 v45, s44, v45
	v_mul_lo_u32 v46, v85, s33
	v_add3_u32 v45, v45, v24, v46
	v_lshl_add_u64 v[46:47], v[26:27], 0, s[14:15]
	s_movk_i32 s14, 0xfbf2
	s_mov_b32 s15, -1
	v_lshl_add_u64 v[86:87], v[38:39], 0, s[14:15]
	v_cndmask_b32_e64 v87, v47, v87, s[22:23]
	v_ashrrev_i32_e32 v88, 31, v87
	v_cndmask_b32_e64 v86, v46, v86, s[22:23]
	v_mov_b32_e32 v89, v88
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[88:89]
	v_xor_b32_e32 v89, v86, v88
	v_xor_b32_e32 v85, v87, v88
	v_mad_u64_u32 v[86:87], s[14:15], v89, s93, 0
	v_mul_hi_u32 v24, v89, s77
	v_lshl_add_u64 v[86:87], v[24:25], 0, v[86:87]
	v_mad_u64_u32 v[92:93], s[14:15], v85, s77, 0
	v_add_co_u32_e32 v24, vcc, v86, v92
	v_mad_u64_u32 v[90:91], s[14:15], v85, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v87, v93, vcc
	s_nop 1
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[86:87], v[24:25], 0, v[90:91]
	v_mul_lo_u32 v24, s87, v86
	v_mul_lo_u32 v87, s86, v87
	v_mad_u64_u32 v[90:91], s[14:15], s86, v86, 0
	v_add3_u32 v24, v91, v87, v24
	v_sub_u32_e32 v87, v85, v24
	v_sub_co_u32_e32 v89, vcc, v89, v90
	s_nop 1
	v_subb_co_u32_e64 v87, s[14:15], v87, v40, vcc
	v_subrev_co_u32_e64 v90, s[14:15], s86, v89
	v_subb_co_u32_e32 v24, vcc, v85, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v87, s[14:15], 0, v87, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v87
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v90
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v89
	v_cndmask_b32_e64 v90, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v87
	v_cndmask_b32_e64 v89, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v87, v91, v90, s[14:15]
	v_add_u32_e32 v90, 2, v86
	v_add_u32_e32 v91, 1, v86
	v_cmp_ne_u32_e64 s[14:15], 0, v87
	v_cndmask_b32_e32 v24, v85, v89, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v87, v91, v90, s[14:15]
	v_xor_b32_e32 v85, s56, v88
	v_cndmask_b32_e32 v24, v86, v87, vcc
	v_ashrrev_i32_e32 v86, 31, v47
	v_mov_b32_e32 v87, v86
	v_xor_b32_e32 v24, v24, v85
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[86:87]
	v_sub_u32_e32 v24, v24, v85
	v_xor_b32_e32 v92, v46, v86
	v_xad_u32 v85, v24, v84, v51
	v_xor_b32_e32 v87, v47, v86
	v_mad_u64_u32 v[46:47], s[14:15], v92, s93, 0
	v_mul_hi_u32 v24, v92, s77
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[90:91], s[14:15], v87, s77, 0
	v_add_co_u32_e32 v24, vcc, v46, v90
	v_mad_u64_u32 v[88:89], s[14:15], v87, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v47, v91, vcc
	s_nop 1
	v_addc_co_u32_e32 v89, vcc, 0, v89, vcc
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[88:89]
	v_mul_lo_u32 v24, s87, v46
	v_mul_lo_u32 v88, s86, v47
	v_mad_u64_u32 v[46:47], s[14:15], s86, v46, 0
	v_add3_u32 v24, v47, v88, v24
	v_sub_u32_e32 v47, v87, v24
	v_sub_co_u32_e32 v46, vcc, v92, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, vcc
	v_subrev_co_u32_e64 v88, s[14:15], s86, v46
	v_subb_co_u32_e32 v24, vcc, v87, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[28:29], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v89
	v_subb_co_u32_e64 v47, s[14:15], v47, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v90, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v88
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v89
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v46
	v_cndmask_b32_e64 v90, v90, v91, s[28:29]
	v_subrev_co_u32_e64 v91, s[14:15], s86, v88
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v90
	s_nop 1
	v_cndmask_b32_e64 v47, v89, v47, s[14:15]
	v_cndmask_b32_e64 v89, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v87, v87, v89, vcc
	v_cmp_ne_u32_e32 vcc, 0, v87
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v47, vcc
	v_cndmask_b32_e64 v47, v88, v91, s[14:15]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v24, v86
	v_xor_b32_e32 v24, v46, v86
	v_sub_co_u32_e32 v24, vcc, v24, v86
	s_mov_b64 s[14:15], 0x40e
	s_nop 0
	v_subb_co_u32_e32 v47, vcc, v47, v86, vcc
	v_lshl_add_u64 v[86:87], v[26:27], 0, s[14:15]
	s_movk_i32 s14, 0xfbf1
	s_mov_b32 s15, -1
	v_lshl_add_u64 v[88:89], v[38:39], 0, s[14:15]
	v_cndmask_b32_e64 v89, v87, v89, s[22:23]
	v_ashrrev_i32_e32 v90, 31, v89
	v_cndmask_b32_e64 v88, v86, v88, s[22:23]
	v_mov_b32_e32 v91, v90
	v_ashrrev_i32_e32 v46, 31, v47
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[90:91]
	v_and_b32_e32 v46, s44, v46
	v_mul_lo_u32 v47, v85, s33
	v_xor_b32_e32 v85, v88, v90
	v_add3_u32 v46, v46, v24, v47
	v_xor_b32_e32 v47, v89, v90
	v_mad_u64_u32 v[88:89], s[14:15], v85, s93, 0
	v_mul_hi_u32 v24, v85, s77
	v_lshl_add_u64 v[88:89], v[24:25], 0, v[88:89]
	v_mad_u64_u32 v[94:95], s[14:15], v47, s77, 0
	v_add_co_u32_e32 v24, vcc, v88, v94
	v_mad_u64_u32 v[92:93], s[14:15], v47, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v89, v95, vcc
	s_nop 1
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[88:89], v[24:25], 0, v[92:93]
	v_mul_lo_u32 v24, s87, v88
	v_mul_lo_u32 v89, s86, v89
	v_mad_u64_u32 v[92:93], s[14:15], s86, v88, 0
	v_add3_u32 v24, v93, v89, v24
	v_sub_u32_e32 v89, v47, v24
	v_sub_co_u32_e32 v85, vcc, v85, v92
	s_nop 1
	v_subb_co_u32_e64 v89, s[14:15], v89, v40, vcc
	v_subrev_co_u32_e64 v91, s[14:15], s86, v85
	v_subb_co_u32_e32 v24, vcc, v47, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[14:15], 0, v89, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v89
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v91
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v85
	v_cndmask_b32_e64 v91, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v89
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v89, v92, v91, s[14:15]
	v_add_u32_e32 v91, 2, v88
	v_add_u32_e32 v92, 1, v88
	v_cmp_ne_u32_e64 s[14:15], 0, v89
	v_cndmask_b32_e32 v24, v47, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v89, v92, v91, s[14:15]
	v_xor_b32_e32 v47, s56, v90
	v_cndmask_b32_e32 v24, v88, v89, vcc
	v_ashrrev_i32_e32 v88, 31, v87
	v_mov_b32_e32 v89, v88
	v_xor_b32_e32 v24, v24, v47
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[88:89]
	v_sub_u32_e32 v24, v24, v47
	v_xor_b32_e32 v89, v86, v88
	v_xad_u32 v85, v24, v84, v51
	v_xor_b32_e32 v47, v87, v88
	v_mad_u64_u32 v[86:87], s[14:15], v89, s93, 0
	v_mul_hi_u32 v24, v89, s77
	v_lshl_add_u64 v[86:87], v[24:25], 0, v[86:87]
	v_mad_u64_u32 v[92:93], s[14:15], v47, s77, 0
	v_add_co_u32_e32 v24, vcc, v86, v92
	v_mad_u64_u32 v[90:91], s[14:15], v47, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v87, v93, vcc
	v_mul_lo_u32 v85, v85, s33
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[86:87], v[24:25], 0, v[90:91]
	v_mul_lo_u32 v24, s87, v86
	v_mul_lo_u32 v90, s86, v87
	v_mad_u64_u32 v[86:87], s[14:15], s86, v86, 0
	v_add3_u32 v24, v87, v90, v24
	v_sub_u32_e32 v87, v47, v24
	v_sub_co_u32_e32 v86, vcc, v89, v86
	s_nop 1
	v_subb_co_u32_e64 v87, s[14:15], v87, v40, vcc
	v_subrev_co_u32_e64 v89, s[14:15], s86, v86
	v_subb_co_u32_e32 v24, vcc, v47, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v90, s[28:29], 0, v87, s[14:15]
	v_cmp_le_u32_e64 s[28:29], s87, v90
	v_subb_co_u32_e64 v87, s[14:15], v87, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s86, v89
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s87, v90
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v86
	v_cndmask_b32_e64 v91, v91, v92, s[28:29]
	v_subrev_co_u32_e64 v92, s[14:15], s86, v89
	s_nop 1
	v_subbrev_co_u32_e64 v87, s[14:15], 0, v87, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v91
	s_nop 1
	v_cndmask_b32_e64 v87, v90, v87, s[14:15]
	v_cndmask_b32_e64 v90, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v47, v47, v90, vcc
	v_cmp_ne_u32_e32 vcc, 0, v47
	v_cndmask_b32_e64 v47, v89, v92, s[14:15]
	s_mov_b64 s[14:15], 0x40f
	v_cndmask_b32_e32 v24, v24, v87, vcc
	v_cndmask_b32_e32 v47, v86, v47, vcc
	v_xor_b32_e32 v86, v24, v88
	v_xor_b32_e32 v24, v47, v88
	v_sub_co_u32_e32 v24, vcc, v24, v88
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v86, v88, vcc
	v_lshl_add_u64 v[86:87], v[26:27], 0, s[14:15]
	s_movk_i32 s14, 0xfbf0
	s_mov_b32 s15, -1
	v_lshl_add_u64 v[26:27], v[38:39], 0, s[14:15]
	v_cndmask_b32_e64 v27, v87, v27, s[22:23]
	v_ashrrev_i32_e32 v38, 31, v27
	v_ashrrev_i32_e32 v47, 31, v47
	v_cndmask_b32_e64 v26, v86, v26, s[22:23]
	v_mov_b32_e32 v39, v38
	v_and_b32_e32 v47, s44, v47
	v_lshl_add_u64 v[26:27], v[26:27], 0, v[38:39]
	v_add3_u32 v47, v47, v24, v85
	v_xor_b32_e32 v85, v26, v38
	v_xor_b32_e32 v39, v27, v38
	v_mad_u64_u32 v[26:27], s[14:15], v85, s93, 0
	v_mul_hi_u32 v24, v85, s77
	v_lshl_add_u64 v[26:27], v[24:25], 0, v[26:27]
	v_mad_u64_u32 v[90:91], s[14:15], v39, s77, 0
	v_add_co_u32_e32 v24, vcc, v26, v90
	v_mad_u64_u32 v[88:89], s[14:15], v39, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v27, v91, vcc
	s_nop 1
	v_addc_co_u32_e32 v89, vcc, 0, v89, vcc
	v_lshl_add_u64 v[26:27], v[24:25], 0, v[88:89]
	v_mul_lo_u32 v24, s87, v26
	v_mul_lo_u32 v27, s86, v27
	v_mad_u64_u32 v[88:89], s[14:15], s86, v26, 0
	v_add3_u32 v24, v89, v27, v24
	v_sub_u32_e32 v27, v39, v24
	v_sub_co_u32_e32 v85, vcc, v85, v88
	s_nop 1
	v_subb_co_u32_e64 v27, s[14:15], v27, v40, vcc
	v_subrev_co_u32_e64 v88, s[14:15], s86, v85
	v_subb_co_u32_e32 v24, vcc, v39, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v27, s[14:15], 0, v27, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s87, v27
	v_cmp_le_u32_e32 vcc, s87, v24
	s_nop 0
	v_cndmask_b32_e64 v89, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s86, v88
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v85
	v_cndmask_b32_e64 v88, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s87, v27
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	v_cndmask_b32_e64 v27, v89, v88, s[14:15]
	v_add_u32_e32 v88, 2, v26
	v_add_u32_e32 v89, 1, v26
	v_cmp_ne_u32_e64 s[14:15], 0, v27
	v_cndmask_b32_e32 v24, v39, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	v_cndmask_b32_e64 v27, v89, v88, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v26, v27, vcc
	v_xor_b32_e32 v26, s56, v38
	v_xor_b32_e32 v24, v24, v26
	v_sub_u32_e32 v24, v24, v26
	v_ashrrev_i32_e32 v26, 31, v87
	v_mov_b32_e32 v27, v26
	v_lshl_add_u64 v[38:39], v[86:87], 0, v[26:27]
	v_xor_b32_e32 v85, v38, v26
	v_xad_u32 v84, v24, v84, v51
	v_xor_b32_e32 v27, v39, v26
	v_mad_u64_u32 v[38:39], s[14:15], v85, s93, 0
	v_mul_hi_u32 v24, v85, s77
	v_lshl_add_u64 v[86:87], v[24:25], 0, v[38:39]
	v_mad_u64_u32 v[88:89], s[14:15], v27, s77, 0
	v_add_co_u32_e32 v24, vcc, v86, v88
	v_mad_u64_u32 v[38:39], s[14:15], v27, s93, 0
	s_nop 0
	v_addc_co_u32_e32 v24, vcc, v87, v89, vcc
	s_nop 1
	v_addc_co_u32_e32 v39, vcc, 0, v39, vcc
	v_lshl_add_u64 v[38:39], v[24:25], 0, v[38:39]
	v_mul_lo_u32 v24, s87, v38
	v_mul_lo_u32 v86, s86, v39
	v_mad_u64_u32 v[38:39], s[14:15], s86, v38, 0
	v_add3_u32 v24, v39, v86, v24
	v_sub_u32_e32 v39, v27, v24
	v_sub_co_u32_e32 v38, vcc, v85, v38
	s_nop 1
	v_subb_co_u32_e64 v39, s[14:15], v39, v40, vcc
	v_subrev_co_u32_e64 v85, s[14:15], s86, v38
	v_subb_co_u32_e32 v24, vcc, v27, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v86, s[22:23], 0, v39, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s87, v86
	v_subb_co_u32_e64 v39, s[14:15], v39, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s86, v85
	v_subrev_co_u32_e64 v40, s[14:15], s86, v85
	s_nop 0
	v_cndmask_b32_e64 v88, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s87, v86
	v_subbrev_co_u32_e64 v39, s[14:15], 0, v39, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v87, v87, v88, s[22:23]
	v_cmp_le_u32_e32 vcc, s87, v24
	v_cmp_ne_u32_e64 s[14:15], 0, v87
	s_nop 0
	v_cndmask_b32_e64 v27, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v38
	v_cndmask_b32_e64 v39, v86, v39, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v86, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v27, v27, v86, vcc
	v_cmp_ne_u32_e32 vcc, 0, v27
	v_cndmask_b32_e64 v27, v85, v40, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v24, v24, v39, vcc
	v_cndmask_b32_e32 v27, v38, v27, vcc
	v_xor_b32_e32 v38, v24, v26
	v_xor_b32_e32 v24, v27, v26
	v_sub_co_u32_e32 v24, vcc, v24, v26
	s_nop 1
	v_subb_co_u32_e32 v27, vcc, v38, v26, vcc
	v_ashrrev_i32_e32 v26, 31, v27
	v_and_b32_e32 v26, s44, v26
	v_mul_lo_u32 v27, v84, s33
	v_add3_u32 v24, v26, v24, v27
	buffer_load_ubyte v84, v41, s[36:39], 0 offen
	buffer_load_ubyte v86, v42, s[36:39], 0 offen
	buffer_load_ubyte v85, v43, s[36:39], 0 offen
	buffer_load_ubyte v87, v44, s[36:39], 0 offen
	buffer_load_ubyte v88, v45, s[36:39], 0 offen
	buffer_load_ubyte v90, v46, s[36:39], 0 offen
	buffer_load_ubyte v89, v47, s[36:39], 0 offen
	buffer_load_ubyte v91, v24, s[36:39], 0 offen
	ds_read2st64_b32 v[26:27], v57 offset1:1
	ds_read_b128 v[128:131], v58
	ds_read_b128 v[100:103], v58 offset:2048
	ds_read_b128 v[120:123], v59
	ds_read_b128 v[108:111], v59 offset:2048
	ds_read_b128 v[112:115], v58 offset:4096
	ds_read_b128 v[104:107], v58 offset:6144
	ds_read_b128 v[124:127], v59 offset:4096
	ds_read_b128 v[116:119], v59 offset:6144
	v_lshl_add_u64 v[42:43], s[54:55], 0, v[32:33]
	v_or_b32_e32 v24, s59, v43
	v_cmp_ne_u32_e32 vcc, 0, v24
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[28:29], exec, s[14:15]
	s_cbranch_execz .LBB0_118
	s_add_u32 s14, s58, s76
	s_mov_b32 s77, s76
	s_addc_u32 s15, s59, s76
	s_xor_b64 s[86:87], s[14:15], s[76:77]
	v_cvt_f32_u32_e32 v24, s86
	v_cvt_f32_u32_e32 v38, s87
	s_sub_u32 s14, 0, s86
	s_subb_u32 s15, 0, s87
	v_ashrrev_i32_e32 v40, 31, v43
	v_fmac_f32_e32 v24, 0x4f800000, v38
	v_rcp_f32_e32 v24, v24
	v_mov_b32_e32 v41, v40
	v_mul_f32_e32 v24, 0x5f7ffffc, v24
	v_mul_f32_e32 v38, 0x2f800000, v24
	v_trunc_f32_e32 v38, v38
	v_fmac_f32_e32 v24, 0xcf800000, v38
	v_cvt_u32_f32_e32 v38, v38
	v_cvt_u32_f32_e32 v24, v24
	v_readfirstlane_b32 s22, v38
	v_readfirstlane_b32 s23, v24
	s_mul_i32 s77, s14, s22
	s_mul_hi_u32 s94, s14, s23
	s_mul_i32 s93, s15, s23
	s_add_i32 s77, s94, s77
	s_add_i32 s77, s77, s93
	s_mul_i32 s95, s14, s23
	s_mul_i32 s94, s23, s77
	s_mul_hi_u32 s96, s23, s95
	s_mul_hi_u32 s93, s23, s77
	s_add_u32 s94, s96, s94
	s_addc_u32 s93, 0, s93
	s_mul_hi_u32 s97, s22, s95
	s_mul_i32 s95, s22, s95
	s_add_u32 s94, s94, s95
	s_mul_hi_u32 s96, s22, s77
	s_addc_u32 s93, s93, s97
	s_addc_u32 s94, s96, 0
	s_mul_i32 s77, s22, s77
	s_add_u32 s77, s93, s77
	s_addc_u32 s93, 0, s94
	s_add_u32 s23, s23, s77
	s_addc_u32 s22, s22, s93
	s_mul_i32 s77, s14, s22
	s_mul_hi_u32 s93, s14, s23
	s_add_i32 s77, s93, s77
	s_mul_i32 s15, s15, s23
	s_add_i32 s77, s77, s15
	s_mul_i32 s14, s14, s23
	s_mul_hi_u32 s93, s22, s14
	s_mul_i32 s94, s22, s14
	s_mul_i32 s96, s23, s77
	s_mul_hi_u32 s14, s23, s14
	s_mul_hi_u32 s95, s23, s77
	s_add_u32 s14, s14, s96
	s_addc_u32 s95, 0, s95
	s_add_u32 s14, s14, s94
	s_mul_hi_u32 s15, s22, s77
	s_addc_u32 s14, s95, s93
	s_addc_u32 s15, s15, 0
	s_mul_i32 s77, s22, s77
	s_add_u32 s14, s14, s77
	s_addc_u32 s15, 0, s15
	s_add_u32 s23, s23, s14
	v_lshl_add_u64 v[38:39], v[42:43], 0, v[40:41]
	s_addc_u32 s22, s22, s15
	v_xor_b32_e32 v92, v38, v40
	v_xor_b32_e32 v41, v39, v40
	v_mad_u64_u32 v[38:39], s[14:15], v92, s22, 0
	v_mul_hi_u32 v24, v92, s23
	v_lshl_add_u64 v[38:39], v[24:25], 0, v[38:39]
	v_mad_u64_u32 v[46:47], s[14:15], v41, s23, 0
	v_add_co_u32_e32 v24, vcc, v38, v46
	v_mad_u64_u32 v[44:45], s[14:15], v41, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v47, vcc
	v_mov_b32_e32 v39, s73
	s_nop 0
	v_addc_co_u32_e32 v45, vcc, 0, v45, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[44:45]
	v_mul_lo_u32 v24, s87, v38
	v_mul_lo_u32 v44, s86, v39
	v_mad_u64_u32 v[38:39], s[14:15], s86, v38, 0
	v_add3_u32 v24, v39, v44, v24
	v_sub_u32_e32 v39, v41, v24
	v_mov_b32_e32 v44, s87
	v_sub_co_u32_e32 v38, vcc, v92, v38
	s_nop 1
	v_subb_co_u32_e64 v39, s[14:15], v39, v44, vcc
	v_subrev_co_u32_e64 v45, s[14:15], s86, v38
	v_subb_co_u32_e32 v24, vcc, v41, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v46, s[22:23], 0, v39, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s87, v46
	v_subb_co_u32_e64 v39, s[14:15], v39, v44, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s86, v45
	v_subrev_co_u32_e64 v44, s[14:15], s86, v45
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s87, v46
	v_subbrev_co_u32_e64 v39, s[14:15], 0, v39, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v47, v47, v92, s[22:23]
	v_cmp_le_u32_e32 vcc, s87, v24
	v_cmp_ne_u32_e64 s[14:15], 0, v47
	s_nop 0
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v38
	v_cndmask_b32_e64 v39, v46, v39, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v41, v41, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v41
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v39, vcc
	v_cndmask_b32_e64 v39, v45, v44, s[14:15]
	v_cndmask_b32_e32 v38, v38, v39, vcc
	v_xor_b32_e32 v38, v38, v40
	v_xor_b32_e32 v24, v24, v40
	v_sub_co_u32_e32 v38, vcc, v38, v40
	s_nop 1
	v_subb_co_u32_e32 v39, vcc, v24, v40, vcc
.LBB0_118:
	s_or_saveexec_b64 s[14:15], s[28:29]
	v_cvt_f32_u32_e32 v92, s58
	s_xor_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB0_120
	v_rcp_iflag_f32_e32 v24, v92
	s_sub_i32 s22, 0, s58
	v_mul_f32_e32 v24, 0x4f7ffffe, v24
	v_cvt_u32_f32_e32 v24, v24
	v_mul_lo_u32 v38, s22, v24
	v_mul_hi_u32 v38, v24, v38
	v_add_u32_e32 v24, v24, v38
	v_mul_hi_u32 v24, v42, v24
	v_mul_lo_u32 v24, v24, s58
	v_sub_u32_e32 v24, v42, v24
	v_subrev_u32_e32 v38, s58, v24
	v_cmp_le_u32_e32 vcc, s58, v24
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v38, vcc
	v_subrev_u32_e32 v38, s58, v24
	v_cmp_le_u32_e32 vcc, s58, v24
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v38, vcc
	v_mov_b64_e32 v[38:39], v[24:25]
.LBB0_120:
	s_or_b64 exec, exec, s[14:15]
	v_lshl_add_u64 v[40:41], s[54:55], 0, v[36:37]
	v_lshl_add_u64 v[44:45], v[40:41], 0, 2
	v_cmp_gt_i32_e32 vcc, 0, v41
	s_nop 1
	v_cndmask_b32_e32 v47, v45, v35, vcc
	v_or_b32_e32 v24, s59, v47
	v_cndmask_b32_e32 v46, v44, v34, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[22:23], exec, s[14:15]
	s_cbranch_execz .LBB0_122
	s_add_u32 s14, s58, s76
	s_mov_b32 s77, s76
	s_addc_u32 s15, s59, s76
	s_xor_b64 s[28:29], s[14:15], s[76:77]
	v_cvt_f32_u32_e32 v24, s28
	v_cvt_f32_u32_e32 v44, s29
	s_sub_u32 s14, 0, s28
	s_subb_u32 s15, 0, s29
	v_fmac_f32_e32 v24, 0x4f800000, v44
	v_rcp_f32_e32 v24, v24
	s_nop 0
	v_mul_f32_e32 v24, 0x5f7ffffc, v24
	v_mul_f32_e32 v44, 0x2f800000, v24
	v_trunc_f32_e32 v44, v44
	v_fmac_f32_e32 v24, 0xcf800000, v44
	v_cvt_u32_f32_e32 v44, v44
	v_cvt_u32_f32_e32 v24, v24
	v_readfirstlane_b32 s77, v44
	v_readfirstlane_b32 s86, v24
	s_mul_i32 s87, s14, s77
	s_mul_hi_u32 s94, s14, s86
	s_mul_i32 s93, s15, s86
	s_add_i32 s87, s94, s87
	s_add_i32 s87, s87, s93
	s_mul_i32 s95, s14, s86
	s_mul_i32 s94, s86, s87
	s_mul_hi_u32 s96, s86, s95
	s_mul_hi_u32 s93, s86, s87
	s_add_u32 s94, s96, s94
	s_addc_u32 s93, 0, s93
	s_mul_hi_u32 s97, s77, s95
	s_mul_i32 s95, s77, s95
	s_add_u32 s94, s94, s95
	s_mul_hi_u32 s96, s77, s87
	s_addc_u32 s93, s93, s97
	s_addc_u32 s94, s96, 0
	s_mul_i32 s87, s77, s87
	s_add_u32 s87, s93, s87
	s_addc_u32 s93, 0, s94
	s_add_u32 s86, s86, s87
	s_addc_u32 s77, s77, s93
	s_mul_i32 s87, s14, s77
	s_mul_hi_u32 s93, s14, s86
	s_add_i32 s87, s93, s87
	s_mul_i32 s15, s15, s86
	s_add_i32 s87, s87, s15
	s_mul_i32 s14, s14, s86
	s_mul_hi_u32 s93, s77, s14
	s_mul_i32 s94, s77, s14
	s_mul_i32 s96, s86, s87
	s_mul_hi_u32 s14, s86, s14
	s_mul_hi_u32 s95, s86, s87
	s_add_u32 s14, s14, s96
	s_addc_u32 s95, 0, s95
	s_add_u32 s14, s14, s94
	s_mul_hi_u32 s15, s77, s87
	s_addc_u32 s14, s95, s93
	s_addc_u32 s15, s15, 0
	s_mul_i32 s87, s77, s87
	s_add_u32 s14, s14, s87
	v_ashrrev_i32_e32 v44, 31, v47
	s_addc_u32 s15, 0, s15
	v_mov_b32_e32 v45, v44
	s_add_u32 s86, s86, s14
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[44:45]
	s_addc_u32 s77, s77, s15
	v_xor_b32_e32 v93, v46, v44
	v_xor_b32_e32 v45, v47, v44
	v_mad_u64_u32 v[46:47], s[14:15], v93, s77, 0
	v_mul_hi_u32 v24, v93, s86
	v_lshl_add_u64 v[46:47], v[24:25], 0, v[46:47]
	v_mad_u64_u32 v[132:133], s[14:15], v45, s86, 0
	v_add_co_u32_e32 v24, vcc, v46, v132
	v_mad_u64_u32 v[94:95], s[14:15], v45, s77, 0
	s_nop 0
	v_addc_co_u32_e32 v46, vcc, v47, v133, vcc
	v_mov_b32_e32 v47, s73
	s_nop 0
	v_addc_co_u32_e32 v95, vcc, 0, v95, vcc
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[94:95]
	v_mul_lo_u32 v24, s29, v46
	v_mul_lo_u32 v99, s28, v47
	v_mad_u64_u32 v[94:95], s[14:15], s28, v46, 0
	v_add3_u32 v24, v95, v99, v24
	v_sub_u32_e32 v95, v45, v24
	v_mov_b32_e32 v99, s29
	v_sub_co_u32_e32 v93, vcc, v93, v94
	v_lshl_add_u64 v[132:133], v[46:47], 0, 1
	s_nop 0
	v_subb_co_u32_e64 v94, s[14:15], v95, v99, vcc
	v_subrev_co_u32_e64 v95, s[14:15], s28, v93
	v_subb_co_u32_e32 v24, vcc, v45, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v94, s[14:15], 0, v94, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s29, v94
	v_cmp_le_u32_e32 vcc, s29, v24
	v_xor_b32_e32 v44, s76, v44
	v_cndmask_b32_e64 v99, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v95
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v93
	v_cndmask_b32_e64 v95, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v94
	s_nop 1
	v_cndmask_b32_e64 v99, v99, v95, s[14:15]
	v_lshl_add_u64 v[94:95], v[46:47], 0, 2
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v24
	s_nop 1
	v_cndmask_b32_e32 v24, v45, v47, vcc
	v_cmp_ne_u32_e32 vcc, 0, v99
	s_nop 1
	v_cndmask_b32_e32 v45, v132, v94, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	s_nop 1
	v_cndmask_b32_e32 v24, v46, v45, vcc
	v_xor_b32_e32 v24, v24, v44
	v_sub_co_u32_e32 v44, vcc, v24, v44
.LBB0_122:
	s_andn2_saveexec_b64 s[14:15], s[22:23]
	s_cbranch_execz .LBB0_124
	v_rcp_iflag_f32_e32 v24, v92
	s_sub_i32 s22, 0, s58
	v_mul_f32_e32 v24, 0x4f7ffffe, v24
	v_cvt_u32_f32_e32 v24, v24
	v_mul_lo_u32 v44, s22, v24
	v_mul_hi_u32 v44, v24, v44
	v_add_u32_e32 v24, v24, v44
	v_mul_hi_u32 v24, v46, v24
	v_mul_lo_u32 v44, v24, s58
	v_sub_u32_e32 v44, v46, v44
	v_add_u32_e32 v45, 1, v24
	v_subrev_u32_e32 v46, s58, v44
	v_cmp_le_u32_e32 vcc, s58, v44
	s_nop 1
	v_cndmask_b32_e32 v44, v44, v46, vcc
	v_cndmask_b32_e32 v24, v24, v45, vcc
	v_add_u32_e32 v45, 1, v24
	v_cmp_le_u32_e32 vcc, s58, v44
	s_nop 1
	v_cndmask_b32_e32 v44, v24, v45, vcc
.LBB0_124:
	s_or_b64 exec, exec, s[14:15]
	v_lshl_add_u64 v[46:47], v[42:43], 0, 2
	v_or_b32_e32 v24, s59, v47
	v_cmp_ne_u32_e32 vcc, 0, v24
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[28:29], exec, s[14:15]
	s_cbranch_execz .LBB0_126
	s_add_u32 s14, s58, s76
	s_mov_b32 s77, s76
	s_addc_u32 s15, s59, s76
	s_xor_b64 s[86:87], s[14:15], s[76:77]
	v_cvt_f32_u32_e32 v24, s86
	v_cvt_f32_u32_e32 v42, s87
	s_sub_u32 s14, 0, s86
	s_subb_u32 s15, 0, s87
	v_ashrrev_i32_e32 v92, 31, v47
	v_fmac_f32_e32 v24, 0x4f800000, v42
	v_rcp_f32_e32 v24, v24
	v_mov_b32_e32 v93, v92
	v_mul_f32_e32 v24, 0x5f7ffffc, v24
	v_mul_f32_e32 v42, 0x2f800000, v24
	v_trunc_f32_e32 v42, v42
	v_fmac_f32_e32 v24, 0xcf800000, v42
	v_cvt_u32_f32_e32 v42, v42
	v_cvt_u32_f32_e32 v24, v24
	v_readfirstlane_b32 s22, v42
	v_readfirstlane_b32 s23, v24
	s_mul_i32 s77, s14, s22
	s_mul_hi_u32 s94, s14, s23
	s_mul_i32 s93, s15, s23
	s_add_i32 s77, s94, s77
	s_add_i32 s77, s77, s93
	s_mul_i32 s95, s14, s23
	s_mul_i32 s94, s23, s77
	s_mul_hi_u32 s96, s23, s95
	s_mul_hi_u32 s93, s23, s77
	s_add_u32 s94, s96, s94
	s_addc_u32 s93, 0, s93
	s_mul_hi_u32 s97, s22, s95
	s_mul_i32 s95, s22, s95
	s_add_u32 s94, s94, s95
	s_mul_hi_u32 s96, s22, s77
	s_addc_u32 s93, s93, s97
	s_addc_u32 s94, s96, 0
	s_mul_i32 s77, s22, s77
	s_add_u32 s77, s93, s77
	s_addc_u32 s93, 0, s94
	s_add_u32 s23, s23, s77
	s_addc_u32 s22, s22, s93
	s_mul_i32 s77, s14, s22
	s_mul_hi_u32 s93, s14, s23
	s_add_i32 s77, s93, s77
	s_mul_i32 s15, s15, s23
	s_add_i32 s77, s77, s15
	s_mul_i32 s14, s14, s23
	s_mul_hi_u32 s93, s22, s14
	s_mul_i32 s94, s22, s14
	s_mul_i32 s96, s23, s77
	s_mul_hi_u32 s14, s23, s14
	s_mul_hi_u32 s95, s23, s77
	s_add_u32 s14, s14, s96
	s_addc_u32 s95, 0, s95
	s_add_u32 s14, s14, s94
	s_mul_hi_u32 s15, s22, s77
	s_addc_u32 s14, s95, s93
	s_addc_u32 s15, s15, 0
	s_mul_i32 s77, s22, s77
	s_add_u32 s14, s14, s77
	s_addc_u32 s15, 0, s15
	s_add_u32 s23, s23, s14
	v_lshl_add_u64 v[42:43], v[46:47], 0, v[92:93]
	s_addc_u32 s22, s22, s15
	v_xor_b32_e32 v93, v42, v92
	v_xor_b32_e32 v45, v43, v92
	v_mad_u64_u32 v[42:43], s[14:15], v93, s22, 0
	v_mul_hi_u32 v24, v93, s23
	v_lshl_add_u64 v[42:43], v[24:25], 0, v[42:43]
	v_mad_u64_u32 v[94:95], s[14:15], v45, s23, 0
	v_add_co_u32_e32 v24, vcc, v42, v94
	v_mad_u64_u32 v[46:47], s[14:15], v45, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v95, vcc
	v_mov_b32_e32 v43, s73
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	v_mul_lo_u32 v24, s87, v42
	v_mul_lo_u32 v46, s86, v43
	v_mad_u64_u32 v[42:43], s[14:15], s86, v42, 0
	v_add3_u32 v24, v43, v46, v24
	v_sub_u32_e32 v43, v45, v24
	v_mov_b32_e32 v46, s87
	v_sub_co_u32_e32 v42, vcc, v93, v42
	s_nop 1
	v_subb_co_u32_e64 v43, s[14:15], v43, v46, vcc
	v_subrev_co_u32_e64 v47, s[14:15], s86, v42
	v_subb_co_u32_e32 v24, vcc, v45, v24, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v93, s[22:23], 0, v43, s[14:15]
	v_cmp_le_u32_e64 s[22:23], s87, v93
	v_subb_co_u32_e64 v43, s[14:15], v43, v46, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v94, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s86, v47
	v_subrev_co_u32_e64 v46, s[14:15], s86, v47
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s87, v93
	v_subbrev_co_u32_e64 v43, s[14:15], 0, v43, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v94, v94, v95, s[22:23]
	v_cmp_le_u32_e32 vcc, s87, v24
	v_cmp_ne_u32_e64 s[14:15], 0, v94
	s_nop 0
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s86, v42
	v_cndmask_b32_e64 v43, v93, v43, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s87, v24
	s_nop 1
	v_cndmask_b32_e32 v45, v45, v93, vcc
	v_cmp_ne_u32_e32 vcc, 0, v45
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v43, vcc
	v_cndmask_b32_e64 v43, v47, v46, s[14:15]
	v_cndmask_b32_e32 v42, v42, v43, vcc
	v_xor_b32_e32 v42, v42, v92
	v_xor_b32_e32 v24, v24, v92
	v_sub_co_u32_e32 v42, vcc, v42, v92
	s_nop 1
	v_subb_co_u32_e32 v43, vcc, v24, v92, vcc
.LBB0_126:
	s_andn2_saveexec_b64 s[14:15], s[28:29]
	s_cbranch_execz .LBB0_111
	v_rcp_iflag_f32_e32 v24, v92
	s_sub_i32 s22, 0, s58
	v_mul_f32_e32 v24, 0x4f7ffffe, v24
	v_cvt_u32_f32_e32 v24, v24
	v_mul_lo_u32 v42, s22, v24
	v_mul_hi_u32 v42, v24, v42
	v_add_u32_e32 v24, v24, v42
	v_mul_hi_u32 v24, v46, v24
	v_mul_lo_u32 v24, v24, s58
	v_sub_u32_e32 v24, v46, v24
	v_subrev_u32_e32 v42, s58, v24
	v_cmp_le_u32_e32 vcc, s58, v24
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v42, vcc
	v_subrev_u32_e32 v42, s58, v24
	v_cmp_le_u32_e32 vcc, s58, v24
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v42, vcc
	v_mov_b64_e32 v[42:43], v[24:25]
	s_branch .LBB0_111
.LBB0_128:
	s_or_b64 s[2:3], s[64:65], s[50:51]
	s_cmp_lg_u32 s3, 0
	s_cbranch_scc0 .LBB0_165
	s_add_u32 s2, s50, s70
	s_mov_b32 s71, s70
	s_addc_u32 s3, s51, s70
	s_xor_b64 s[4:5], s[2:3], s[70:71]
	v_cvt_f32_u32_e32 v16, s4
	v_cvt_f32_u32_e32 v17, s5
	s_sub_u32 s6, 0, s4
	s_subb_u32 s7, 0, s5
	v_fmamk_f32 v16, v17, 0x4f800000, v16
	v_rcp_f32_e32 v16, v16
	s_nop 0
	v_mul_f32_e32 v16, 0x5f7ffffc, v16
	v_mul_f32_e32 v17, 0x2f800000, v16
	v_trunc_f32_e32 v17, v17
	v_fmamk_f32 v16, v17, 0xcf800000, v16
	v_cvt_u32_f32_e32 v17, v17
	v_cvt_u32_f32_e32 v16, v16
	v_readfirstlane_b32 s8, v17
	v_readfirstlane_b32 s9, v16
	s_mul_i32 s10, s6, s8
	s_mul_hi_u32 s14, s6, s9
	s_mul_i32 s11, s7, s9
	s_add_i32 s10, s14, s10
	s_add_i32 s10, s10, s11
	s_mul_i32 s15, s6, s9
	s_mul_i32 s14, s9, s10
	s_mul_hi_u32 s18, s9, s15
	s_mul_hi_u32 s11, s9, s10
	s_add_u32 s14, s18, s14
	s_addc_u32 s11, 0, s11
	s_mul_hi_u32 s19, s8, s15
	s_mul_i32 s15, s8, s15
	s_add_u32 s14, s14, s15
	s_mul_hi_u32 s18, s8, s10
	s_addc_u32 s11, s11, s19
	s_addc_u32 s14, s18, 0
	s_mul_i32 s10, s8, s10
	s_add_u32 s10, s11, s10
	s_addc_u32 s11, 0, s14
	s_add_u32 s9, s9, s10
	s_addc_u32 s8, s8, s11
	s_mul_i32 s10, s6, s8
	s_mul_hi_u32 s11, s6, s9
	s_add_i32 s10, s11, s10
	s_mul_i32 s7, s7, s9
	s_add_i32 s10, s10, s7
	s_mul_i32 s6, s6, s9
	s_mul_hi_u32 s11, s8, s6
	s_mul_i32 s14, s8, s6
	s_mul_i32 s18, s9, s10
	s_mul_hi_u32 s6, s9, s6
	s_mul_hi_u32 s15, s9, s10
	s_add_u32 s6, s6, s18
	s_addc_u32 s15, 0, s15
	s_add_u32 s6, s6, s14
	s_mul_hi_u32 s7, s8, s10
	s_addc_u32 s6, s15, s11
	s_addc_u32 s7, s7, 0
	s_mul_i32 s10, s8, s10
	s_add_u32 s6, s6, s10
	s_addc_u32 s7, 0, s7
	s_add_u32 s10, s9, s6
	s_addc_u32 s11, s8, s7
	s_ashr_i32 s6, s65, 31
	s_add_u32 s8, s64, s6
	s_mov_b32 s7, s6
	s_addc_u32 s9, s65, s6
	s_xor_b64 s[8:9], s[8:9], s[6:7]
	s_mul_i32 s15, s8, s11
	s_mul_hi_u32 s18, s8, s10
	s_mul_hi_u32 s14, s8, s11
	s_add_u32 s15, s18, s15
	s_addc_u32 s14, 0, s14
	s_mul_hi_u32 s19, s9, s10
	s_mul_i32 s10, s9, s10
	s_add_u32 s10, s15, s10
	s_mul_hi_u32 s18, s9, s11
	s_addc_u32 s10, s14, s19
	s_addc_u32 s14, s18, 0
	s_mul_i32 s11, s9, s11
	s_add_u32 s10, s10, s11
	s_addc_u32 s11, 0, s14
	s_mul_i32 s11, s4, s11
	s_mul_hi_u32 s14, s4, s10
	s_add_i32 s11, s14, s11
	s_mul_i32 s14, s5, s10
	s_add_i32 s18, s11, s14
	s_sub_i32 s14, s9, s18
	s_mul_i32 s10, s4, s10
	s_sub_u32 s8, s8, s10
	s_cselect_b64 s[10:11], -1, 0
	s_subb_u32 s19, s14, s5
	s_sub_u32 s20, s8, s4
	s_cselect_b64 s[14:15], -1, 0
	s_subb_u32 s21, s19, 0
	s_cmp_ge_u32 s21, s5
	s_cselect_b32 s22, -1, 0
	s_cmp_ge_u32 s20, s4
	s_cselect_b32 s23, -1, 0
	s_cmp_eq_u32 s21, s5
	s_cselect_b32 s22, s23, s22
	s_cmp_lg_u64 s[14:15], 0
	s_subb_u32 s14, s19, s5
	s_sub_u32 s15, s20, s4
	s_subb_u32 s14, s14, 0
	s_cmp_lg_u32 s22, 0
	s_cselect_b32 s15, s15, s20
	s_cselect_b32 s14, s14, s21
	s_cmp_lg_u64 s[10:11], 0
	s_subb_u32 s9, s9, s18
	s_cmp_ge_u32 s9, s5
	s_cselect_b32 s10, -1, 0
	s_cmp_ge_u32 s8, s4
	s_cselect_b32 s4, -1, 0
	s_cmp_eq_u32 s9, s5
	s_cselect_b32 s4, s4, s10
	s_cmp_lg_u32 s4, 0
	s_cselect_b32 s5, s14, s9
	s_cselect_b32 s4, s15, s8
	s_xor_b64 s[4:5], s[4:5], s[6:7]
	s_sub_u32 s4, s4, s6
	s_subb_u32 s5, s5, s6
	s_cbranch_execnz .LBB0_131
.LBB0_130:
	v_rcp_iflag_f32_e32 v16, v48
	s_sub_i32 s2, 0, s50
	s_mov_b32 s5, 0
	v_mul_f32_e32 v16, 0x4f7ffffe, v16
	v_cvt_u32_f32_e32 v16, v16
	s_nop 0
	v_readfirstlane_b32 s3, v16
	s_mul_i32 s2, s2, s3
	s_mul_hi_u32 s2, s3, s2
	s_add_i32 s3, s3, s2
	s_mul_hi_u32 s2, s64, s3
	s_mul_i32 s2, s2, s50
	s_sub_i32 s2, s64, s2
	s_sub_i32 s3, s2, s50
	s_cmp_ge_u32 s2, s50
	s_cselect_b32 s2, s3, s2
	s_sub_i32 s3, s2, s50
	s_cmp_ge_u32 s2, s50
	s_cselect_b32 s4, s3, s2
.LBB0_131:
	v_bfe_u32 v16, v15, 16, 1
	s_movk_i32 s2, 0x7fff
	v_bfe_u32 v17, v14, 16, 1
	v_add3_u32 v16, v15, v16, s2
	v_bfe_u32 v18, v13, 16, 1
	v_add3_u32 v17, v14, v17, s2
	v_lshrrev_b32_e32 v16, 16, v16
	v_mov_b32_e32 v24, 0x7fc0
	v_cmp_o_f32_e32 vcc, v15, v15
	v_add3_u32 v18, v13, v18, s2
	v_lshrrev_b32_e32 v17, 16, v17
	v_cndmask_b32_e32 v20, v24, v16, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_bfe_u32 v19, v12, 16, 1
	v_lshrrev_b32_e32 v18, 16, v18
	v_cndmask_b32_e32 v21, v24, v17, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_add3_u32 v19, v12, v19, s2
	v_lshrrev_b32_e32 v19, 16, v19
	v_cndmask_b32_e32 v22, v24, v18, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v12, v11, 16, 1
	v_bfe_u32 v13, v10, 16, 1
	v_add3_u32 v12, v11, v12, s2
	v_cndmask_b32_e32 v23, v24, v19, vcc
	v_bfe_u32 v14, v9, 16, 1
	v_add3_u32 v13, v10, v13, s2
	v_lshrrev_b32_e32 v12, 16, v12
	v_cmp_o_f32_e32 vcc, v11, v11
	v_add3_u32 v14, v9, v14, s2
	v_lshrrev_b32_e32 v13, 16, v13
	v_cndmask_b32_e32 v16, v24, v12, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_bfe_u32 v15, v8, 16, 1
	v_lshrrev_b32_e32 v14, 16, v14
	v_cndmask_b32_e32 v17, v24, v13, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_add3_u32 v15, v8, v15, s2
	v_lshrrev_b32_e32 v15, 16, v15
	v_cndmask_b32_e32 v18, v24, v14, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v8, v7, 16, 1
	v_bfe_u32 v9, v6, 16, 1
	v_bfe_u32 v10, v5, 16, 1
	v_add3_u32 v8, v7, v8, s2
	v_cndmask_b32_e32 v19, v24, v15, vcc
	v_add3_u32 v10, v5, v10, s2
	v_add3_u32 v9, v6, v9, s2
	v_lshrrev_b32_e32 v8, 16, v8
	v_cmp_o_f32_e32 vcc, v7, v7
	v_lshrrev_b32_e32 v9, 16, v9
	v_lshrrev_b32_e32 v13, 16, v10
	v_cndmask_b32_e32 v10, v24, v8, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_bfe_u32 v11, v4, 16, 1
	v_add3_u32 v11, v4, v11, s2
	v_cndmask_b32_e32 v12, v24, v9, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_bfe_u32 v5, v2, 16, 1
	v_bfe_u32 v6, v1, 16, 1
	v_cndmask_b32_e32 v13, v24, v13, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v4, v3, 16, 1
	v_bfe_u32 v7, v0, 16, 1
	v_add3_u32 v7, v0, v7, s2
	v_add3_u32 v6, v1, v6, s2
	v_add3_u32 v5, v2, v5, s2
	v_add3_u32 v4, v3, v4, s2
	s_sub_u32 s2, s17, s62
	s_mul_i32 s6, s50, s51
	s_mul_hi_u32 s7, s50, s50
	s_subb_u32 s3, s16, s63
	s_add_i32 s7, s7, s6
	s_mul_i32 s9, s50, s50
	s_add_i32 s8, s7, s6
	s_mul_i32 s3, s9, s3
	s_mul_hi_u32 s7, s9, s2
	s_mul_i32 s6, s8, s2
	s_add_i32 s3, s7, s3
	v_readlane_b32 s10, v144, 0
	s_add_i32 s3, s3, s6
	s_mul_i32 s6, s8, s10
	s_mul_hi_u32 s7, s9, s10
	s_mul_i32 s2, s9, s2
	s_add_i32 s7, s7, s6
	s_mul_i32 s6, s9, s10
	s_add_u32 s6, s2, s6
	s_addc_u32 s7, s3, s7
	s_mul_i32 s2, s9, s51
	s_mul_hi_u32 s3, s9, s50
	s_add_i32 s2, s3, s2
	s_mul_i32 s8, s8, s50
	s_add_i32 s14, s2, s8
	s_ashr_i32 s2, s7, 31
	s_mov_b32 s3, s2
	s_mul_i32 s9, s9, s50
	s_xor_b64 s[10:11], s[2:3], s[6:7]
	s_ashr_i32 s6, s14, 31
	s_add_u32 s8, s9, s6
	s_mov_b32 s7, s6
	s_addc_u32 s9, s14, s6
	s_xor_b64 s[8:9], s[8:9], s[6:7]
	v_cvt_f32_u32_e32 v9, s8
	v_cvt_f32_u32_e32 v14, s9
	v_lshrrev_b32_e32 v11, 16, v11
	v_cndmask_b32_e32 v15, v24, v11, vcc
	v_lshrrev_b32_e32 v4, 16, v4
	v_cmp_o_f32_e32 vcc, v3, v3
	v_lshrrev_b32_e32 v11, 16, v7
	v_lshrrev_b32_e32 v5, 16, v5
	v_cndmask_b32_e32 v7, v24, v4, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	v_fmamk_f32 v2, v14, 0x4f800000, v9
	v_rcp_f32_e32 v2, v2
	v_lshrrev_b32_e32 v6, 16, v6
	v_cndmask_b32_e32 v8, v24, v5, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	s_sub_u32 s14, 0, s8
	s_subb_u32 s15, 0, s9
	v_cndmask_b32_e32 v9, v24, v6, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	v_mul_f32_e32 v0, 0x5f7ffffc, v2
	v_mul_f32_e32 v1, 0x2f800000, v0
	v_trunc_f32_e32 v1, v1
	v_fmamk_f32 v0, v1, 0xcf800000, v0
	v_cvt_u32_f32_e32 v0, v0
	v_cvt_u32_f32_e32 v1, v1
	v_cndmask_b32_e32 v11, v24, v11, vcc
	v_lshlrev_b32_e32 v6, 7, v49
	v_readfirstlane_b32 s16, v0
	v_readfirstlane_b32 s19, v1
	s_mul_hi_u32 s18, s14, s16
	s_mul_i32 s20, s14, s19
	s_mul_i32 s17, s15, s16
	s_add_i32 s18, s18, s20
	s_add_i32 s18, s18, s17
	s_mul_i32 s21, s14, s16
	s_mul_i32 s20, s16, s18
	s_mul_hi_u32 s22, s16, s21
	s_mul_hi_u32 s17, s16, s18
	s_add_u32 s20, s22, s20
	s_addc_u32 s17, 0, s17
	s_mul_hi_u32 s23, s19, s21
	s_mul_i32 s21, s19, s21
	s_add_u32 s20, s20, s21
	s_mul_hi_u32 s22, s19, s18
	s_addc_u32 s17, s17, s23
	s_addc_u32 s20, s22, 0
	s_mul_i32 s18, s19, s18
	s_add_u32 s17, s17, s18
	s_addc_u32 s18, 0, s20
	s_add_u32 s16, s16, s17
	s_addc_u32 s17, s19, s18
	s_mul_i32 s18, s14, s17
	s_mul_hi_u32 s19, s14, s16
	s_add_i32 s18, s19, s18
	s_mul_i32 s15, s15, s16
	s_add_i32 s18, s18, s15
	s_mul_i32 s14, s14, s16
	s_mul_hi_u32 s19, s17, s14
	s_mul_i32 s20, s17, s14
	s_mul_i32 s22, s16, s18
	s_mul_hi_u32 s14, s16, s14
	s_mul_hi_u32 s21, s16, s18
	s_add_u32 s14, s14, s22
	s_addc_u32 s21, 0, s21
	s_add_u32 s14, s14, s20
	s_mul_hi_u32 s15, s17, s18
	s_addc_u32 s14, s21, s19
	s_addc_u32 s15, s15, 0
	s_mul_i32 s18, s17, s18
	s_add_u32 s14, s14, s18
	s_addc_u32 s15, 0, s15
	s_add_u32 s16, s16, s14
	s_addc_u32 s17, s17, s15
	s_ashr_i32 s14, s11, 31
	s_add_u32 s10, s10, s14
	s_mov_b32 s15, s14
	s_addc_u32 s11, s11, s14
	s_xor_b64 s[10:11], s[10:11], s[14:15]
	s_mul_i32 s19, s10, s17
	s_mul_hi_u32 s20, s10, s16
	s_mul_hi_u32 s18, s10, s17
	s_add_u32 s19, s20, s19
	s_addc_u32 s18, 0, s18
	s_mul_hi_u32 s21, s11, s16
	s_mul_i32 s16, s11, s16
	s_add_u32 s16, s19, s16
	s_mul_hi_u32 s20, s11, s17
	s_addc_u32 s16, s18, s21
	s_addc_u32 s18, s20, 0
	s_mul_i32 s17, s11, s17
	s_add_u32 s19, s16, s17
	s_addc_u32 s18, 0, s18
	s_mul_i32 s16, s8, s18
	s_mul_hi_u32 s17, s8, s19
	s_add_i32 s16, s17, s16
	s_mul_i32 s17, s9, s19
	s_add_i32 s20, s16, s17
	s_sub_i32 s21, s11, s20
	s_mul_i32 s16, s8, s19
	s_sub_u32 s10, s10, s16
	s_cselect_b64 s[16:17], -1, 0
	s_subb_u32 s21, s21, s9
	s_sub_u32 s22, s10, s8
	s_subb_u32 s21, s21, 0
	s_cmp_ge_u32 s21, s9
	s_cselect_b32 s23, -1, 0
	s_cmp_ge_u32 s22, s8
	s_cselect_b32 s22, -1, 0
	s_cmp_eq_u32 s21, s9
	s_cselect_b32 s21, s22, s23
	s_add_u32 s22, s19, 1
	s_addc_u32 s23, s18, 0
	s_add_u32 s26, s19, 2
	s_addc_u32 s27, s18, 0
	s_cmp_lg_u32 s21, 0
	s_cselect_b32 s21, s26, s22
	s_cselect_b32 s22, s27, s23
	s_cmp_lg_u64 s[16:17], 0
	s_subb_u32 s11, s11, s20
	s_cmp_ge_u32 s11, s9
	s_cselect_b32 s16, -1, 0
	s_cmp_ge_u32 s10, s8
	s_cselect_b32 s8, -1, 0
	s_cmp_eq_u32 s11, s9
	s_cselect_b32 s8, s8, s16
	s_cmp_lg_u32 s8, 0
	s_cselect_b32 s9, s22, s18
	s_cselect_b32 s8, s21, s19
	s_xor_b64 s[6:7], s[14:15], s[6:7]
	s_xor_b64 s[8:9], s[8:9], s[6:7]
	s_sub_u32 s6, s8, s6
	s_subb_u32 s7, s9, s7
	s_xor_b64 s[2:3], s[6:7], s[2:3]
	s_and_b64 s[6:7], s[88:89], exec
	s_cselect_b32 s7, s3, s91
	s_cselect_b32 s6, s2, s90
	s_lshl_b64 s[2:3], s[6:7], 7
	v_sub_co_u32_e32 v3, vcc, s6, v49
	s_ashr_i32 s6, s5, 31
	v_mov_b32_e32 v2, s7
	s_mov_b32 s7, s6
	v_lshlrev_b32_e32 v0, 2, v98
	v_and_b32_e32 v1, 64, v96
	s_and_b64 s[6:7], s[6:7], s[50:51]
	v_or3_b32 v4, s2, v0, v1
	s_add_u32 s2, s4, s48
	s_addc_u32 s4, s5, s49
	s_add_u32 s2, s2, s6
	s_addc_u32 s4, s4, s7
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s2, s2, 0
	v_or3_b32 v14, v6, v0, v1
	v_and_b32_e32 v0, 15, v96
	v_subb_co_u32_e32 v2, vcc, v2, v50, vcc
	v_lshl_or_b32 v0, s2, 5, v0
	v_or_b32_e32 v6, v0, v97
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v0, s24, v2
	v_mul_hi_u32 v1, s24, v3
	s_cselect_b32 s1, s4, 0
	v_add_u32_e32 v0, v1, v0
	v_mul_lo_u32 v1, s25, v3
	v_mad_u64_u32 v[24:25], s[4:5], v14, s24, v[6:7]
	s_sub_u32 s0, s68, s2
	v_add_u32_e32 v1, v0, v1
	v_mul_lo_u32 v0, s24, v3
	v_readlane_b32 s4, v144, 1
	v_mov_b32_e32 v5, s3
	s_subb_u32 s1, s69, s1
	v_lshlrev_b64 v[0:1], 8, v[0:1]
	v_readlane_b32 s5, v144, 2
	s_lshl_b64 s[0:1], s[0:1], 6
	v_not_b32_e32 v25, -2.0
	v_lshl_add_u64 v[0:1], s[4:5], 0, v[0:1]
	v_cmp_gt_i64_e32 vcc, s[12:13], v[4:5]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[0:1]
	v_and_b32_e32 v1, 0xffff, v1
	v_cndmask_b32_e32 v5, v25, v24, vcc
	v_mov_b32_e32 v2, 0x7ffffffd
	v_mov_b32_e32 v3, 0x27000
	v_lshlrev_b32_e32 v5, 1, v5
	s_mov_b64 s[8:9], exec
.LBB0_132:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v23, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_132
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v24, 1, v4
	v_mov_b32_e32 v25, s3
	v_or_b32_e32 v5, 1, v14
	v_mad_u64_u32 v[26:27], s[0:1], v5, s24, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[12:13], v[24:25]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v26, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_134:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v22, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_134
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v22, 2, v4
	v_mov_b32_e32 v23, s3
	v_or_b32_e32 v5, 2, v14
	v_mad_u64_u32 v[24:25], s[0:1], v5, s24, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[12:13], v[22:23]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v24, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_136:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v21, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_136
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v22, 3, v4
	v_mov_b32_e32 v23, s3
	v_or_b32_e32 v5, 3, v14
	v_mad_u64_u32 v[24:25], s[0:1], v5, s24, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[12:13], v[22:23]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v24, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_138:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v20, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_138
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v20, 16, v4
	v_mov_b32_e32 v21, s3
	v_or_b32_e32 v5, 16, v14
	v_mad_u64_u32 v[22:23], s[0:1], v5, s24, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[12:13], v[20:21]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v22, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_140:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v19, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_140
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v20, 17, v4
	v_mov_b32_e32 v21, s3
	v_or_b32_e32 v5, 17, v14
	v_mad_u64_u32 v[22:23], s[0:1], v5, s24, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[12:13], v[20:21]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v22, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_142:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v18, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_142
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v18, 18, v4
	v_mov_b32_e32 v19, s3
	v_or_b32_e32 v5, 18, v14
	v_mad_u64_u32 v[20:21], s[0:1], v5, s24, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[12:13], v[18:19]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v20, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_144:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v17, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_144
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v18, 19, v4
	v_mov_b32_e32 v19, s3
	v_or_b32_e32 v5, 19, v14
	v_mad_u64_u32 v[20:21], s[0:1], v5, s24, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[12:13], v[18:19]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v20, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_146:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v16, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_146
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v16, 32, v4
	v_mov_b32_e32 v17, s3
	v_or_b32_e32 v5, 32, v14
	v_mad_u64_u32 v[18:19], s[0:1], v5, s24, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[12:13], v[16:17]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v18, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_148:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v15, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_148
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v16, 33, v4
	v_mov_b32_e32 v17, s3
	v_or_b32_e32 v5, 33, v14
	v_mad_u64_u32 v[18:19], s[0:1], v5, s24, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[12:13], v[16:17]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v18, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_150:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v13, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_150
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v16, 34, v4
	v_mov_b32_e32 v17, s3
	v_or_b32_e32 v5, 34, v14
	v_mad_u64_u32 v[18:19], s[0:1], v5, s24, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[12:13], v[16:17]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v18, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_152:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v12, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_152
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v12, 35, v4
	v_mov_b32_e32 v13, s3
	v_or_b32_e32 v5, 35, v14
	v_mad_u64_u32 v[16:17], s[0:1], v5, s24, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[12:13], v[12:13]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v16, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_154:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v10, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_154
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v12, 48, v4
	v_mov_b32_e32 v13, s3
	v_or_b32_e32 v5, 48, v14
	v_mad_u64_u32 v[16:17], s[0:1], v5, s24, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[12:13], v[12:13]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v16, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_156:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v11, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_156
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v10, 49, v4
	v_mov_b32_e32 v11, s3
	v_or_b32_e32 v5, 49, v14
	v_mad_u64_u32 v[12:13], s[0:1], v5, s24, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[12:13], v[10:11]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v12, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_158:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v9, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_158
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v10, 50, v4
	v_mov_b32_e32 v11, s3
	v_or_b32_e32 v5, 50, v14
	v_mad_u64_u32 v[12:13], s[0:1], v5, s24, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[12:13], v[10:11]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v12, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_160:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v8, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_160
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v4, 51, v4
	v_mov_b32_e32 v5, s3
	v_or_b32_e32 v8, 51, v14
	v_mad_u64_u32 v[8:9], s[0:1], v8, s24, v[6:7]
	v_not_b32_e32 v6, -2.0
	v_cmp_gt_i64_e32 vcc, s[12:13], v[4:5]
	s_mov_b64 s[0:1], exec
	s_nop 0
	v_cndmask_b32_e32 v4, v6, v8, vcc
	v_lshlrev_b32_e32 v4, 1, v4
.LBB0_162:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v7, v4, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_162
	s_endpgm
.LBB0_164:
	v_cvt_f32_u32_e32 v48, s50
	s_branch .LBB0_108
.LBB0_165:
	s_branch .LBB0_130
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_dynamic_gemm_128x32x256
		.amdhsa_group_segment_fixed_size 52224
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
		.amdhsa_next_free_vgpr 145
		.amdhsa_next_free_sgpr 98
		.amdhsa_accum_offset 148
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

	.set wave_mxfp4_dynamic_gemm_128x32x256.num_vgpr, 145
	.set wave_mxfp4_dynamic_gemm_128x32x256.num_agpr, 0
	.set wave_mxfp4_dynamic_gemm_128x32x256.numbered_sgpr, 98
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
    .group_segment_fixed_size: 52224
    .kernarg_segment_align: 8
    .kernarg_segment_size: 104
    .max_flat_workgroup_size: 256
    .name:           wave_mxfp4_dynamic_gemm_128x32x256
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 128
      - 2
      - 1
    .sgpr_count:     104
    .sgpr_spill_count: 33
    .symbol:         wave_mxfp4_dynamic_gemm_128x32x256.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     145
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
