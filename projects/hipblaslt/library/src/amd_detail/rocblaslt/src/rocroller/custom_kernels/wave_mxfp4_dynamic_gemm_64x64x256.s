; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_dynamic_gemm_64x64x256
	.p2align	8
	.type	wave_mxfp4_dynamic_gemm_64x64x256,@function
wave_mxfp4_dynamic_gemm_64x64x256:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	s_load_dwordx2 s[46:47], s[0:1], 0x38
	s_load_dword s33, s[0:1], 0x50
	s_mov_b64 s[96:97], s[0:1]
	s_load_dword s0, s[0:1], 0x58
	v_writelane_b32 v144, s17, 0
	v_writelane_b32 v144, s16, 2
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s66, s47, 31
	v_writelane_b32 v144, s10, 3
	s_mov_b32 s34, s46
	s_mov_b32 s35, s47
	s_mov_b32 s67, s66
	v_writelane_b32 v144, s11, 4
	s_xor_b64 s[68:69], s[34:35], s[66:67]
	v_writelane_b32 v144, s0, 5
	s_lshr_b32 s0, s69, 31
	s_add_u32 s0, s68, s0
	s_mov_b64 s[36:37], s[2:3]
	s_addc_u32 s1, s69, 0
	s_ashr_i32 s2, s69, 31
	s_ashr_i64 s[0:1], s[0:1], 1
	v_writelane_b32 v144, s2, 6
	s_lshr_b32 s2, s2, 27
	s_add_u32 s2, s68, s2
	s_addc_u32 s3, s69, 0
	s_mov_b64 s[92:93], s[4:5]
	s_ashr_i64 s[4:5], s[2:3], 5
	s_mov_b32 s60, s12
	s_sub_u32 s12, 0, s46
	s_mov_b32 s61, s13
	s_subb_u32 s13, 0, s47
	s_add_u32 s16, s46, -1
	v_cmp_lt_i64_e64 s[2:3], s[34:35], 1
	s_addc_u32 s17, s47, -1
	s_and_b64 s[10:11], s[2:3], exec
	s_cselect_b32 s11, s13, s17
	s_cselect_b32 s10, s12, s16
	s_ashr_i32 s12, s11, 31
	s_lshr_b32 s12, s12, 24
	s_add_u32 s10, s10, s12
	s_addc_u32 s11, s11, 0
	s_ashr_i64 s[10:11], s[10:11], 8
	s_sub_u32 s12, 0, s10
	s_subb_u32 s13, 0, s11
	s_add_u32 s10, s10, 1
	s_addc_u32 s11, s11, 0
	v_bfe_u32 v95, v0, 10, 10
	s_and_b64 s[2:3], s[2:3], exec
	v_mov_b32_e32 v1, 0x100
	v_cmp_ne_u32_e64 s[2:3], 0, v95
	s_cselect_b32 s10, s12, s10
	v_and_b32_e32 v24, 0x3ff, v0
	v_cndmask_b32_e64 v1, 0, v1, s[2:3]
	s_cselect_b32 s11, s13, s11
	v_readfirstlane_b32 s12, v1
	v_lshlrev_b32_e32 v0, 3, v95
	s_mov_b64 s[28:29], s[6:7]
	v_writelane_b32 v144, s12, 7
	s_mov_b32 s64, s46
	s_mov_b32 s65, 0
	v_mov_b32_e32 v25, 0
	s_mov_b64 s[6:7], -1
	v_readfirstlane_b32 s49, v0
	s_cmp_gt_i32 s10, 5
	v_lshrrev_b32_e32 v45, 3, v24
	v_lshrrev_b32_e32 v63, 4, v24
	v_writelane_b32 v144, s10, 8
	s_nop 1
	v_writelane_b32 v144, s11, 9
	s_cbranch_scc1 .LBB0_2
	s_ashr_i32 s6, s11, 31
	s_mov_b32 s7, s6
	s_xor_b64 s[10:11], s[6:7], s[10:11]
	s_lshr_b32 s12, s11, 31
	s_add_u32 s10, s10, s12
	s_addc_u32 s11, s11, 0
	s_lshr_b64 s[10:11], s[10:11], 1
	s_xor_b64 s[6:7], s[10:11], s[6:7]
	s_lshl_b64 s[62:63], s[6:7], 1
	s_sub_u32 s10, 0, s14
	s_subb_u32 s11, 0, s15
	v_writelane_b32 v144, s10, 10
	s_mov_b32 s39, 0x27000
	s_mov_b32 s38, 0x7ffffffe
	v_writelane_b32 v144, s11, 11
	s_add_u32 s10, s14, -1
	s_addc_u32 s11, s15, -1
	v_writelane_b32 v144, s10, 12
	s_mov_b64 s[6:7], 0
	v_or_b32_e32 v28, v45, v0
	v_writelane_b32 v144, s11, 13
	s_sub_u32 s10, 0, s60
	s_subb_u32 s11, 0, s61
	v_writelane_b32 v144, s10, 14
	v_mov_b32_e32 v29, v25
	s_mov_b32 s12, s36
	v_writelane_b32 v144, s11, 15
	s_add_u32 s10, s60, -1
	s_addc_u32 s11, s61, -1
	v_writelane_b32 v144, s10, 16
	s_and_b32 s13, s37, 0xffff
	s_and_b32 s17, s93, 0xffff
	v_writelane_b32 v144, s11, 17
	s_mov_b32 s16, s92
	s_and_b32 s19, s29, 0xffff
	s_mov_b32 s18, s28
	s_and_b32 s21, s9, 0xffff
	s_mov_b32 s20, s8
	s_mov_b64 s[94:95], s[38:39]
	s_mov_b64 s[30:31], s[38:39]
	s_mov_b64 s[10:11], s[38:39]
	s_branch .LBB0_3
.LBB0_2:
.LBB0_3:
	s_xor_b64 s[42:43], s[0:1], s[66:67]
	s_xor_b64 s[0:1], s[4:5], s[66:67]
	v_writelane_b32 v144, s0, 18
	v_cmp_lt_i64_e64 s[58:59], s[60:61], 1
	s_andn2_b64 vcc, exec, s[6:7]
	v_writelane_b32 v144, s1, 19
	s_or_b32 s0, s49, 32
	v_writelane_b32 v144, s0, 20
	v_writelane_b32 v144, s60, 22
	v_cmp_lt_i64_e64 s[6:7], s[14:15], 1
	v_lshlrev_b32_e32 v26, 7, v24
	v_writelane_b32 v144, s61, 23
	v_writelane_b32 v144, s49, 24
	v_cndmask_b32_e64 v94, 0, 1, s[2:3]
	v_writelane_b32 v144, s58, 25
	s_nop 1
	v_writelane_b32 v144, s59, 26
	s_cbranch_vccnz .LBB0_10
	s_sub_u32 s2, 0, s14
	s_subb_u32 s3, 0, s15
	s_add_u32 s4, s14, -1
	s_addc_u32 s5, s15, -1
	s_and_b64 s[0:1], s[6:7], exec
	s_cselect_b32 s0, s2, s4
	v_writelane_b32 v144, s2, 10
	v_or_b32_e32 v28, v45, v0
	v_mov_b32_e32 v29, 0
	v_writelane_b32 v144, s3, 11
	v_writelane_b32 v144, s4, 12
	s_cselect_b32 s1, s3, s5
	s_ashr_i32 s2, s1, 31
	s_lshr_b32 s2, s2, 26
	s_add_u32 s0, s0, s2
	s_addc_u32 s1, s1, 0
	s_ashr_i64 s[0:1], s[0:1], 6
	s_sub_u32 s2, 0, s0
	s_subb_u32 s3, 0, s1
	v_writelane_b32 v144, s5, 13
	s_add_u32 s4, s0, 1
	s_addc_u32 s5, s1, 0
	v_writelane_b32 v144, s6, 27
	s_and_b64 s[0:1], s[6:7], exec
	s_cselect_b32 s2, s2, s4
	s_cselect_b32 s3, s3, s5
	s_and_b32 s0, s2, 31
	s_max_u32 s50, s0, 1
	s_cmp_lg_u32 s0, 0
	v_writelane_b32 v144, s7, 28
	s_cselect_b64 s[0:1], -1, 0
	s_sub_u32 s6, 0, s60
	s_subb_u32 s7, 0, s61
	s_add_u32 s10, s60, -1
	s_addc_u32 s11, s61, -1
	s_and_b64 s[4:5], s[58:59], exec
	s_cselect_b32 s4, s6, s10
	v_writelane_b32 v144, s6, 14
	v_cvt_f32_u32_e32 v1, s50
	v_mov_b32_e32 v37, v29
	v_writelane_b32 v144, s7, 15
	v_writelane_b32 v144, s10, 16
	s_cselect_b32 s5, s7, s11
	s_ashr_i32 s6, s5, 31
	s_lshr_b32 s6, s6, 26
	s_add_u32 s4, s4, s6
	s_addc_u32 s5, s5, 0
	s_ashr_i64 s[4:5], s[4:5], 6
	s_sub_u32 s6, 0, s4
	s_subb_u32 s7, 0, s5
	v_writelane_b32 v144, s11, 17
	s_add_u32 s10, s4, 1
	s_addc_u32 s11, s5, 0
	s_and_b64 s[4:5], s[58:59], exec
	s_cselect_b32 s19, s7, s11
	s_cselect_b32 s18, s6, s10
	v_readlane_b32 s6, v144, 0
	s_mul_i32 s4, s19, s6
	s_mul_hi_u32 s5, s18, s6
	s_add_i32 s5, s5, s4
	s_mul_i32 s4, s18, s6
	v_readlane_b32 s6, v144, 2
	s_add_u32 s4, s4, s6
	s_addc_u32 s5, s5, 0
	s_ashr_i32 s6, s3, 31
	s_mov_b32 s7, s6
	s_xor_b64 s[2:3], s[6:7], s[2:3]
	s_lshr_b32 s10, s3, 27
	s_add_u32 s2, s2, s10
	s_addc_u32 s3, s3, 0
	s_ashr_i64 s[2:3], s[2:3], 5
	s_xor_b64 s[2:3], s[2:3], s[6:7]
	s_lshl_b64 s[22:23], s[2:3], 5
	s_mul_i32 s6, s22, s19
	s_mul_hi_u32 s7, s22, s18
	s_add_i32 s10, s7, s6
	s_lshr_b64 s[6:7], s[2:3], 27
	s_mul_i32 s6, s6, s18
	s_add_i32 s10, s10, s6
	s_mul_i32 s6, s22, s18
	v_mov_b32_e32 v2, s6
	v_mov_b32_e32 v3, s10
	v_cmp_ge_i64_e32 vcc, s[4:5], v[2:3]
	v_cvt_f32_ubyte0_e32 v2, 0
	v_fmac_f32_e32 v1, 0x4f800000, v2
	v_rcp_f32_e32 v1, v1
	s_mul_i32 s6, s2, s19
	s_mul_hi_u32 s7, s2, s18
	s_add_i32 s6, s7, s6
	s_mul_i32 s3, s3, s18
	s_add_i32 s21, s6, s3
	s_mul_i32 s20, s2, s18
	s_and_b64 s[12:13], s[0:1], vcc
	s_lshl_b64 s[2:3], s[20:21], 5
	v_mul_f32_e32 v1, 0x5f7ffffc, v1
	s_sub_u32 s16, 0, s2
	v_mul_f32_e32 v2, 0x2f800000, v1
	s_subb_u32 s17, 0, s3
	v_trunc_f32_e32 v2, v2
	s_sub_u32 s24, s4, s2
	v_fmac_f32_e32 v1, 0xcf800000, v2
	s_subb_u32 s25, s5, s3
	v_cvt_u32_f32_e32 v3, v2
	v_cvt_u32_f32_e32 v1, v1
	s_ashr_i32 s26, s25, 31
	s_mov_b32 s27, s26
	s_xor_b64 s[2:3], s[26:27], s[24:25]
	s_sub_u32 s6, 0, s50
	v_readfirstlane_b32 s10, v3
	v_readfirstlane_b32 s14, v1
	s_subb_u32 s7, 0, 0
	s_mul_i32 s11, s6, s10
	s_mul_hi_u32 s15, s6, s14
	s_add_i32 s11, s15, s11
	s_mul_i32 s15, s7, s14
	s_add_i32 s11, s11, s15
	s_mul_i32 s31, s6, s14
	s_mul_i32 s30, s14, s11
	s_mul_hi_u32 s38, s14, s31
	s_mul_hi_u32 s15, s14, s11
	s_add_u32 s30, s38, s30
	s_addc_u32 s15, 0, s15
	s_mul_hi_u32 s39, s10, s31
	s_mul_i32 s31, s10, s31
	s_add_u32 s30, s30, s31
	s_mul_hi_u32 s38, s10, s11
	s_addc_u32 s15, s15, s39
	s_addc_u32 s30, s38, 0
	s_mul_i32 s11, s10, s11
	s_add_u32 s11, s15, s11
	s_addc_u32 s15, 0, s30
	s_add_u32 s11, s14, s11
	s_addc_u32 s10, s10, s15
	s_mul_i32 s14, s6, s10
	s_mul_hi_u32 s15, s6, s11
	s_add_i32 s14, s15, s14
	s_mul_i32 s7, s7, s11
	s_add_i32 s14, s14, s7
	s_mul_i32 s6, s6, s11
	s_mul_hi_u32 s15, s10, s6
	s_mul_i32 s30, s10, s6
	s_mul_i32 s38, s11, s14
	s_mul_hi_u32 s6, s11, s6
	s_mul_hi_u32 s31, s11, s14
	s_add_u32 s6, s6, s38
	s_addc_u32 s31, 0, s31
	s_add_u32 s6, s6, s30
	s_mul_hi_u32 s7, s10, s14
	s_addc_u32 s6, s31, s15
	s_addc_u32 s7, s7, 0
	s_mul_i32 s14, s10, s14
	s_add_u32 s6, s6, s14
	s_addc_u32 s7, 0, s7
	s_add_u32 s11, s11, s6
	s_addc_u32 s10, s10, s7
	s_ashr_i32 s6, s3, 31
	s_add_u32 s2, s2, s6
	s_mov_b32 s7, s6
	s_addc_u32 s3, s3, s6
	s_xor_b64 s[2:3], s[2:3], s[6:7]
	s_mul_i32 s15, s2, s10
	s_mul_hi_u32 s30, s2, s11
	s_mul_hi_u32 s14, s2, s10
	s_add_u32 s15, s30, s15
	s_addc_u32 s14, 0, s14
	s_mul_hi_u32 s31, s3, s11
	s_mul_i32 s11, s3, s11
	s_add_u32 s11, s15, s11
	s_mul_hi_u32 s30, s3, s10
	s_addc_u32 s11, s14, s31
	s_addc_u32 s14, s30, 0
	s_mul_i32 s10, s3, s10
	s_add_u32 s10, s11, s10
	s_addc_u32 s11, 0, s14
	s_add_u32 s14, s10, 1
	s_addc_u32 s15, s11, 0
	s_add_u32 s30, s10, 2
	s_mul_i32 s38, s50, s11
	s_mul_hi_u32 s39, s50, s10
	s_addc_u32 s31, s11, 0
	s_add_i32 s39, s39, s38
	s_mul_i32 s38, s50, s10
	s_sub_u32 s2, s2, s38
	s_subb_u32 s3, s3, s39
	s_sub_u32 s38, s2, s50
	s_subb_u32 s39, s3, 0
	s_cmp_ge_u32 s38, s50
	s_cselect_b32 s38, -1, 0
	s_cmp_eq_u32 s39, 0
	s_cselect_b32 s38, s38, -1
	s_cmp_lg_u32 s38, 0
	s_cselect_b32 s14, s30, s14
	s_cselect_b32 s15, s31, s15
	s_cmp_ge_u32 s2, s50
	s_cselect_b32 s2, -1, 0
	s_cmp_eq_u32 s3, 0
	s_cselect_b32 s2, s2, -1
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s3, s15, s11
	s_cselect_b32 s2, s14, s10
	s_xor_b64 s[2:3], s[2:3], s[6:7]
	s_sub_u32 s2, s2, s6
	s_subb_u32 s3, s3, s6
	s_ashr_i32 s6, s5, 31
	s_mov_b32 s7, s6
	s_xor_b64 s[10:11], s[6:7], s[4:5]
	s_ashr_i32 s5, s11, 31
	s_xor_b64 s[2:3], s[2:3], s[26:27]
	s_lshr_b32 s5, s5, 27
	s_add_u32 s10, s10, s5
	s_addc_u32 s11, s11, 0
	s_ashr_i64 s[10:11], s[10:11], 5
	s_xor_b64 s[40:41], s[10:11], s[6:7]
	s_ashr_i32 s6, s19, 31
	s_add_u32 s10, s18, s6
	s_mov_b32 s7, s6
	s_addc_u32 s11, s19, s6
	s_xor_b64 s[6:7], s[10:11], s[6:7]
	v_cvt_f32_u32_e32 v1, s6
	v_cvt_f32_u32_e32 v2, s7
	s_sub_u32 s5, 0, s6
	s_subb_u32 s10, 0, s7
	v_mov_b32_e32 v27, v29
	v_fmac_f32_e32 v1, 0x4f800000, v2
	v_rcp_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x5f7ffffc, v1
	v_mul_f32_e32 v2, 0x2f800000, v1
	v_trunc_f32_e32 v2, v2
	v_fmac_f32_e32 v1, 0xcf800000, v2
	v_cvt_u32_f32_e32 v1, v1
	v_cvt_u32_f32_e32 v2, v2
	v_readfirstlane_b32 s11, v1
	v_readfirstlane_b32 s27, v2
	s_mul_hi_u32 s15, s5, s11
	s_mul_i32 s30, s5, s27
	s_mul_i32 s14, s10, s11
	s_add_i32 s15, s15, s30
	s_add_i32 s15, s15, s14
	s_mul_i32 s31, s5, s11
	s_mul_i32 s30, s11, s15
	s_mul_hi_u32 s38, s11, s31
	s_mul_hi_u32 s14, s11, s15
	s_add_u32 s30, s38, s30
	s_addc_u32 s14, 0, s14
	s_mul_hi_u32 s39, s27, s31
	s_mul_i32 s31, s27, s31
	s_add_u32 s30, s30, s31
	s_mul_hi_u32 s38, s27, s15
	s_addc_u32 s14, s14, s39
	s_addc_u32 s30, s38, 0
	s_mul_i32 s15, s27, s15
	s_add_u32 s14, s14, s15
	s_addc_u32 s15, 0, s30
	s_add_u32 s11, s11, s14
	s_addc_u32 s14, s27, s15
	s_mul_i32 s15, s5, s14
	s_mul_hi_u32 s27, s5, s11
	s_add_i32 s15, s27, s15
	s_mul_i32 s10, s10, s11
	s_add_i32 s15, s15, s10
	s_mul_i32 s5, s5, s11
	s_mul_hi_u32 s27, s14, s5
	s_mul_i32 s30, s14, s5
	s_mul_i32 s38, s11, s15
	s_mul_hi_u32 s5, s11, s5
	s_mul_hi_u32 s31, s11, s15
	s_add_u32 s5, s5, s38
	s_addc_u32 s31, 0, s31
	s_add_u32 s5, s5, s30
	s_mul_hi_u32 s10, s14, s15
	s_addc_u32 s5, s31, s27
	s_addc_u32 s10, s10, 0
	s_mul_i32 s15, s14, s15
	s_add_u32 s5, s5, s15
	s_addc_u32 s10, 0, s10
	s_add_u32 s5, s11, s5
	s_addc_u32 s14, s14, s10
	s_ashr_i32 s30, s41, 31
	s_add_u32 s10, s40, s30
	s_mov_b32 s31, s30
	s_addc_u32 s11, s41, s30
	s_xor_b64 s[10:11], s[10:11], s[30:31]
	s_mul_i32 s27, s10, s14
	s_mul_hi_u32 s38, s10, s5
	s_mul_hi_u32 s15, s10, s14
	s_add_u32 s27, s38, s27
	s_addc_u32 s15, 0, s15
	s_mul_hi_u32 s39, s11, s5
	s_mul_i32 s5, s11, s5
	s_add_u32 s5, s27, s5
	s_mul_hi_u32 s38, s11, s14
	s_addc_u32 s5, s15, s39
	s_addc_u32 s15, s38, 0
	s_mul_i32 s14, s11, s14
	s_add_u32 s5, s5, s14
	s_addc_u32 s14, 0, s15
	s_mul_i32 s14, s6, s14
	s_mul_hi_u32 s15, s6, s5
	s_add_i32 s14, s15, s14
	s_mul_i32 s15, s7, s5
	s_add_i32 s27, s14, s15
	s_sub_i32 s38, s11, s27
	s_mul_i32 s5, s6, s5
	s_sub_u32 s5, s10, s5
	s_cselect_b64 s[14:15], -1, 0
	s_subb_u32 s10, s38, s7
	s_sub_u32 s44, s5, s6
	s_cselect_b64 s[38:39], -1, 0
	s_subb_u32 s45, s10, 0
	s_cmp_ge_u32 s45, s7
	s_cselect_b32 s46, -1, 0
	s_cmp_ge_u32 s44, s6
	s_cselect_b32 s48, -1, 0
	s_cmp_eq_u32 s45, s7
	s_cselect_b32 s46, s48, s46
	s_cmp_lg_u64 s[38:39], 0
	s_subb_u32 s10, s10, s7
	s_sub_u32 s38, s44, s6
	s_subb_u32 s10, s10, 0
	s_cmp_lg_u32 s46, 0
	s_cselect_b32 s38, s38, s44
	s_cselect_b32 s10, s10, s45
	s_cmp_lg_u64 s[14:15], 0
	s_subb_u32 s11, s11, s27
	s_cmp_ge_u32 s11, s7
	s_cselect_b32 s14, -1, 0
	s_cmp_ge_u32 s5, s6
	s_cselect_b32 s6, -1, 0
	s_cmp_eq_u32 s11, s7
	s_cselect_b32 s6, s6, s14
	s_cmp_lg_u32 s6, 0
	s_cselect_b32 s7, s10, s11
	s_cselect_b32 s6, s38, s5
	s_xor_b64 s[6:7], s[6:7], s[30:31]
	s_sub_u32 s5, s6, s30
	s_subb_u32 s10, s7, s30
	s_ashr_i32 s6, s10, 31
	s_mov_b32 s7, s6
	s_and_b64 s[6:7], s[6:7], s[18:19]
	s_add_u32 s5, s6, s5
	s_addc_u32 s10, s7, s10
	s_and_b64 s[6:7], s[12:13], exec
	s_cselect_b32 s11, s3, s10
	s_cselect_b32 s10, s2, s5
	s_lshl_b64 s[14:15], s[10:11], 6
	v_or_b32_e32 v0, s14, v28
	v_bitop3_b32 v2, v24, v45, 7 bitop3:0x6c
	v_mov_b32_e32 v1, s15
	v_mad_u64_u32 v[34:35], s[2:3], v0, s42, 0
	v_lshlrev_b32_e32 v36, 5, v2
	v_lshlrev_b32_e32 v32, 4, v2
	v_cmp_gt_i64_e32 vcc, s[34:35], v[36:37]
	v_cmp_gt_i64_e64 s[2:3], s[60:61], v[0:1]
	v_add_u32_e32 v1, v32, v34
	v_bfrev_b32_e32 v2, -2
	s_and_b64 s[6:7], vcc, s[2:3]
	s_lshl_b32 s46, s49, 7
	s_mov_b32 s39, 0x27000
	s_mov_b32 s38, 0x7ffffffe
	s_and_b32 s37, s37, 0xffff
	v_cndmask_b32_e64 v1, v2, v1, s[6:7]
	s_mov_b32 m0, s46
	v_or_b32_e32 v0, 32, v0
	buffer_load_dwordx4 v1, s[36:39], 0 offen lds
	v_mov_b32_e32 v1, s15
	v_mad_u64_u32 v[38:39], s[6:7], v0, s42, 0
	v_cmp_gt_i64_e64 s[14:15], s[60:61], v[0:1]
	v_add_u32_e32 v0, v32, v38
	s_and_b64 vcc, vcc, s[14:15]
	s_add_i32 s54, s46, 0x1000
	v_cndmask_b32_e32 v0, v2, v0, vcc
	s_mov_b32 m0, s54
	s_lshl_b64 s[6:7], s[10:11], 1
	buffer_load_dwordx4 v0, s[36:39], 0 offen lds
	s_lshl_b32 s5, s34, 5
	v_or_b32_e32 v0, s6, v94
	s_mul_i32 s10, s5, s7
	s_lshr_b64 s[6:7], s[34:35], 27
	v_mul_lo_u32 v1, s6, v0
	v_add_u32_e32 v1, s10, v1
	v_mad_u64_u32 v[40:41], s[6:7], s5, v0, v[26:27]
	v_add_u32_e32 v41, v1, v41
	v_or_b32_e32 v0, s35, v41
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_ashrrev_i32_e32 v0, 31, v41
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[44:45], exec, s[6:7]
	s_cbranch_execz .LBB0_6
	s_ashr_i32 s6, s35, 31
	s_add_u32 s10, s34, s6
	s_mov_b32 s7, s6
	s_addc_u32 s11, s35, s6
	s_xor_b64 s[48:49], s[10:11], s[6:7]
	v_cvt_f32_u32_e32 v1, s48
	v_cvt_f32_u32_e32 v2, s49
	s_sub_u32 s5, 0, s48
	s_subb_u32 s6, 0, s49
	v_mov_b32_e32 v5, v29
	v_fmac_f32_e32 v1, 0x4f800000, v2
	v_rcp_f32_e32 v1, v1
	s_mov_b32 s53, 0
	v_mul_f32_e32 v1, 0x5f7ffffc, v1
	v_mul_f32_e32 v2, 0x2f800000, v1
	v_trunc_f32_e32 v2, v2
	v_fmac_f32_e32 v1, 0xcf800000, v2
	v_cvt_u32_f32_e32 v2, v2
	v_cvt_u32_f32_e32 v1, v1
	v_readfirstlane_b32 s7, v2
	v_readfirstlane_b32 s10, v1
	s_mul_i32 s11, s5, s7
	s_mul_hi_u32 s31, s5, s10
	s_mul_i32 s27, s6, s10
	s_add_i32 s11, s31, s11
	s_add_i32 s11, s11, s27
	s_mul_i32 s51, s5, s10
	s_mul_i32 s31, s10, s11
	s_mul_hi_u32 s52, s10, s51
	s_mul_hi_u32 s27, s10, s11
	s_add_u32 s31, s52, s31
	s_addc_u32 s27, 0, s27
	s_mul_hi_u32 s55, s7, s51
	s_mul_i32 s51, s7, s51
	s_add_u32 s31, s31, s51
	s_mul_hi_u32 s52, s7, s11
	s_addc_u32 s27, s27, s55
	s_addc_u32 s31, s52, 0
	s_mul_i32 s11, s7, s11
	s_add_u32 s11, s27, s11
	s_addc_u32 s27, 0, s31
	s_add_u32 s10, s10, s11
	s_addc_u32 s7, s7, s27
	s_mul_i32 s11, s5, s7
	s_mul_hi_u32 s27, s5, s10
	s_add_i32 s11, s27, s11
	s_mul_i32 s6, s6, s10
	s_add_i32 s11, s11, s6
	s_mul_i32 s5, s5, s10
	s_mul_hi_u32 s27, s7, s5
	s_mul_i32 s31, s7, s5
	s_mul_i32 s52, s10, s11
	s_mul_hi_u32 s5, s10, s5
	s_mul_hi_u32 s51, s10, s11
	s_add_u32 s5, s5, s52
	s_addc_u32 s51, 0, s51
	s_add_u32 s5, s5, s31
	s_mul_hi_u32 s6, s7, s11
	s_addc_u32 s5, s51, s27
	s_addc_u32 s6, s6, 0
	s_mul_i32 s11, s7, s11
	s_add_u32 s5, s5, s11
	s_addc_u32 s6, 0, s6
	v_mov_b32_e32 v1, v0
	s_add_u32 s5, s10, s5
	v_lshl_add_u64 v[2:3], v[40:41], 0, v[0:1]
	s_addc_u32 s10, s7, s6
	v_xor_b32_e32 v8, v2, v0
	v_xor_b32_e32 v1, v3, v0
	v_mad_u64_u32 v[2:3], s[6:7], v8, s10, 0
	v_mul_hi_u32 v4, v8, s5
	v_lshl_add_u64 v[2:3], v[4:5], 0, v[2:3]
	v_mad_u64_u32 v[6:7], s[6:7], v1, s5, 0
	v_add_co_u32_e32 v2, vcc, v2, v6
	v_mad_u64_u32 v[4:5], s[6:7], v1, s10, 0
	s_nop 0
	v_addc_co_u32_e32 v2, vcc, v3, v7, vcc
	v_mov_b32_e32 v3, s53
	s_nop 0
	v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	v_mul_lo_u32 v4, s49, v2
	v_mul_lo_u32 v5, s48, v3
	v_mad_u64_u32 v[2:3], s[6:7], s48, v2, 0
	v_add3_u32 v3, v3, v5, v4
	v_sub_u32_e32 v4, v1, v3
	v_mov_b32_e32 v5, s49
	v_sub_co_u32_e32 v2, vcc, v8, v2
	s_nop 1
	v_subb_co_u32_e64 v4, s[6:7], v4, v5, vcc
	v_subrev_co_u32_e64 v6, s[6:7], s48, v2
	v_subb_co_u32_e32 v1, vcc, v1, v3, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v7, s[10:11], 0, v4, s[6:7]
	v_cmp_le_u32_e64 s[10:11], s49, v7
	v_subb_co_u32_e64 v4, s[6:7], v4, v5, s[6:7]
	s_nop 0
	v_cndmask_b32_e64 v8, 0, -1, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s48, v6
	v_subrev_co_u32_e64 v5, s[6:7], s48, v6
	s_nop 0
	v_cndmask_b32_e64 v9, 0, -1, s[10:11]
	v_cmp_eq_u32_e64 s[10:11], s49, v7
	v_subbrev_co_u32_e64 v4, s[6:7], 0, v4, s[6:7]
	s_nop 0
	v_cndmask_b32_e64 v8, v8, v9, s[10:11]
	v_cmp_le_u32_e32 vcc, s49, v1
	v_cmp_ne_u32_e64 s[6:7], 0, v8
	s_nop 0
	v_cndmask_b32_e64 v3, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s48, v2
	v_cndmask_b32_e64 v4, v7, v4, s[6:7]
	s_nop 0
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s49, v1
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
	s_andn2_saveexec_b64 s[6:7], s[44:45]
	s_cbranch_execz .LBB0_8
	v_cvt_f32_u32_e32 v1, s34
	s_sub_i32 s5, 0, s34
	v_mov_b32_e32 v3, 0
	v_rcp_iflag_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_mul_lo_u32 v2, s5, v1
	v_mul_hi_u32 v2, v1, v2
	v_add_u32_e32 v1, v1, v2
	v_mul_hi_u32 v1, v40, v1
	v_mul_lo_u32 v1, v1, s34
	v_sub_u32_e32 v1, v40, v1
	v_subrev_u32_e32 v2, s34, v1
	v_cmp_le_u32_e32 vcc, s34, v1
	s_nop 1
	v_cndmask_b32_e32 v1, v1, v2, vcc
	v_subrev_u32_e32 v2, s34, v1
	v_cmp_le_u32_e32 vcc, s34, v1
	s_nop 1
	v_cndmask_b32_e32 v2, v1, v2, vcc
.LBB0_8:
	s_or_b64 exec, exec, s[6:7]
	s_ashr_i32 s72, s35, 31
	s_add_u32 s6, s34, s72
	s_mov_b32 s73, s72
	s_addc_u32 s7, s35, s72
	s_xor_b64 s[10:11], s[6:7], s[72:73]
	v_cvt_f32_u32_e32 v1, s10
	v_cvt_f32_u32_e32 v4, s11
	s_sub_u32 s5, 0, s10
	s_subb_u32 s6, 0, s11
	v_xor_b32_e32 v5, v0, v41
	v_fmac_f32_e32 v1, 0x4f800000, v4
	v_rcp_f32_e32 v1, v1
	v_xor_b32_e32 v4, v0, v40
	v_mov_b32_e32 v9, 0
	s_mov_b32 s51, 0
	v_mul_f32_e32 v1, 0x5f7ffffc, v1
	v_mul_f32_e32 v6, 0x2f800000, v1
	v_trunc_f32_e32 v6, v6
	v_fmac_f32_e32 v1, 0xcf800000, v6
	v_cvt_u32_f32_e32 v6, v6
	v_cvt_u32_f32_e32 v1, v1
	s_mov_b32 s95, 0x27000
	s_mov_b32 s94, 0x7ffffffe
	v_readfirstlane_b32 s7, v6
	v_readfirstlane_b32 s27, v1
	s_mul_hi_u32 s44, s5, s27
	s_mul_i32 s45, s5, s7
	s_mul_i32 s31, s6, s27
	s_add_i32 s44, s44, s45
	s_add_i32 s44, s44, s31
	s_mul_i32 s48, s5, s27
	s_mul_i32 s45, s27, s44
	s_mul_hi_u32 s49, s27, s48
	s_mul_hi_u32 s31, s27, s44
	s_add_u32 s45, s49, s45
	s_addc_u32 s31, 0, s31
	s_mul_hi_u32 s52, s7, s48
	s_mul_i32 s48, s7, s48
	s_add_u32 s45, s45, s48
	s_mul_hi_u32 s49, s7, s44
	s_addc_u32 s31, s31, s52
	s_addc_u32 s45, s49, 0
	s_mul_i32 s44, s7, s44
	s_add_u32 s31, s31, s44
	s_addc_u32 s44, 0, s45
	s_add_u32 s27, s27, s31
	s_addc_u32 s7, s7, s44
	s_mul_i32 s31, s5, s7
	s_mul_hi_u32 s44, s5, s27
	s_add_i32 s31, s44, s31
	s_mul_i32 s6, s6, s27
	s_add_i32 s31, s31, s6
	s_mul_i32 s5, s5, s27
	s_mul_hi_u32 s44, s7, s5
	s_mul_i32 s45, s7, s5
	s_mul_i32 s49, s27, s31
	s_mul_hi_u32 s5, s27, s5
	s_mul_hi_u32 s48, s27, s31
	s_add_u32 s5, s5, s49
	s_addc_u32 s48, 0, s48
	s_add_u32 s5, s5, s45
	s_mul_hi_u32 s6, s7, s31
	s_addc_u32 s5, s48, s44
	s_addc_u32 s6, s6, 0
	s_mul_i32 s31, s7, s31
	s_add_u32 s5, s5, s31
	v_ashrrev_i32_e32 v6, 31, v5
	s_addc_u32 s6, 0, s6
	v_mov_b32_e32 v7, v6
	s_add_u32 s5, s27, s5
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[6:7]
	s_addc_u32 s27, s7, s6
	v_xor_b32_e32 v7, v4, v6
	v_xor_b32_e32 v1, v5, v6
	v_mad_u64_u32 v[4:5], s[6:7], v7, s27, 0
	v_mul_hi_u32 v8, v7, s5
	v_lshl_add_u64 v[4:5], v[8:9], 0, v[4:5]
	v_mad_u64_u32 v[10:11], s[6:7], v1, s5, 0
	v_add_co_u32_e32 v4, vcc, v4, v10
	v_mad_u64_u32 v[8:9], s[6:7], v1, s27, 0
	s_nop 0
	v_addc_co_u32_e32 v4, vcc, v5, v11, vcc
	v_mov_b32_e32 v5, s51
	s_nop 0
	v_addc_co_u32_e32 v9, vcc, 0, v9, vcc
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[8:9]
	v_mul_lo_u32 v10, s11, v4
	v_mul_lo_u32 v5, s10, v5
	v_mad_u64_u32 v[8:9], s[6:7], s10, v4, 0
	v_add3_u32 v5, v9, v5, v10
	v_sub_u32_e32 v9, v1, v5
	v_mov_b32_e32 v10, s11
	v_sub_co_u32_e32 v7, vcc, v7, v8
	v_readlane_b32 s5, v144, 7
	s_nop 0
	v_subb_co_u32_e64 v8, s[6:7], v9, v10, vcc
	v_subrev_co_u32_e64 v9, s[6:7], s10, v7
	v_subb_co_u32_e32 v1, vcc, v1, v5, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v8, s[6:7], 0, v8, s[6:7]
	v_cmp_le_u32_e64 s[6:7], s11, v8
	v_cmp_le_u32_e32 vcc, s11, v1
	s_add_i32 s44, s5, 0x6000
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
	v_readlane_b32 s6, v144, 18
	v_cndmask_b32_e32 v1, v4, v8, vcc
	v_xor_b32_e32 v4, s72, v6
	v_xor_b32_e32 v1, v1, v4
	v_sub_u32_e32 v1, v1, v4
	v_xor_b32_e32 v4, v0, v1
	v_ashrrev_i32_e32 v0, 31, v3
	v_and_b32_e32 v1, s35, v0
	v_and_b32_e32 v0, s34, v0
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[2:3]
	v_ashrrev_i32_e32 v5, 31, v1
	v_xor_b32_e32 v1, v5, v1
	v_ashrrev_i32_e32 v2, 31, v1
	v_xor_b32_e32 v0, v5, v0
	v_lshrrev_b32_e32 v2, 27, v2
	v_mov_b32_e32 v3, s51
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[2:3]
	v_alignbit_b32 v0, v1, v0, 5
	v_mul_lo_u32 v1, v4, s6
	v_xad_u32 v0, v0, v5, v1
	s_and_b32 s93, s93, 0xffff
	s_mov_b32 m0, s44
	s_cmp_lg_u32 s25, 0
	buffer_load_dword v0, s[92:95], 0 offen lds
	v_readlane_b32 s7, v144, 19
	s_cbranch_scc0 .LBB0_11
	s_add_u32 s6, s50, 0
	s_addc_u32 s7, 0, 0
	s_xor_b64 s[10:11], s[6:7], 0
	v_cvt_f32_u32_e32 v0, s10
	v_cvt_f32_u32_e32 v1, s11
	s_sub_u32 s5, 0, s10
	s_subb_u32 s27, 0, s11
	s_mov_b64 s[6:7], 0
	v_fmac_f32_e32 v0, 0x4f800000, v1
	v_rcp_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x5f7ffffc, v0
	v_mul_f32_e32 v1, 0x2f800000, v0
	v_trunc_f32_e32 v1, v1
	v_fmac_f32_e32 v0, 0xcf800000, v1
	v_cvt_u32_f32_e32 v1, v1
	v_cvt_u32_f32_e32 v0, v0
	v_readfirstlane_b32 s31, v1
	v_readfirstlane_b32 s45, v0
	s_mul_i32 s48, s5, s31
	s_mul_hi_u32 s51, s5, s45
	s_mul_i32 s49, s27, s45
	s_add_i32 s48, s51, s48
	s_add_i32 s48, s48, s49
	s_mul_i32 s52, s5, s45
	s_mul_i32 s51, s45, s48
	s_mul_hi_u32 s53, s45, s52
	s_mul_hi_u32 s49, s45, s48
	s_add_u32 s51, s53, s51
	s_addc_u32 s49, 0, s49
	s_mul_hi_u32 s55, s31, s52
	s_mul_i32 s52, s31, s52
	s_add_u32 s51, s51, s52
	s_mul_hi_u32 s53, s31, s48
	s_addc_u32 s49, s49, s55
	s_addc_u32 s51, s53, 0
	s_mul_i32 s48, s31, s48
	s_add_u32 s48, s49, s48
	s_addc_u32 s49, 0, s51
	s_add_u32 s45, s45, s48
	s_addc_u32 s31, s31, s49
	s_mul_i32 s48, s5, s31
	s_mul_hi_u32 s49, s5, s45
	s_add_i32 s48, s49, s48
	s_mul_i32 s27, s27, s45
	s_add_i32 s48, s48, s27
	s_mul_i32 s5, s5, s45
	s_mul_hi_u32 s49, s31, s5
	s_mul_i32 s51, s31, s5
	s_mul_i32 s53, s45, s48
	s_mul_hi_u32 s5, s45, s5
	s_mul_hi_u32 s52, s45, s48
	s_add_u32 s5, s5, s53
	s_addc_u32 s52, 0, s52
	s_add_u32 s5, s5, s51
	s_mul_hi_u32 s27, s31, s48
	s_addc_u32 s5, s52, s49
	s_addc_u32 s27, s27, 0
	s_mul_i32 s48, s31, s48
	s_add_u32 s5, s5, s48
	s_addc_u32 s27, 0, s27
	s_add_u32 s5, s45, s5
	s_addc_u32 s31, s31, s27
	s_add_u32 s48, s24, s26
	s_mov_b32 s27, s26
	s_addc_u32 s49, s25, s26
	s_xor_b64 s[48:49], s[48:49], s[26:27]
	s_mul_i32 s45, s48, s31
	s_mul_hi_u32 s51, s48, s5
	s_mul_hi_u32 s25, s48, s31
	s_add_u32 s45, s51, s45
	s_addc_u32 s25, 0, s25
	s_mul_hi_u32 s52, s49, s5
	s_mul_i32 s5, s49, s5
	s_add_u32 s5, s45, s5
	s_mul_hi_u32 s51, s49, s31
	s_addc_u32 s5, s25, s52
	s_addc_u32 s25, s51, 0
	s_mul_i32 s31, s49, s31
	s_add_u32 s5, s5, s31
	s_addc_u32 s25, 0, s25
	s_mul_i32 s25, s10, s25
	s_mul_hi_u32 s31, s10, s5
	s_add_i32 s25, s31, s25
	s_mul_i32 s31, s11, s5
	s_add_i32 s25, s25, s31
	s_sub_i32 s31, s49, s25
	s_mul_i32 s5, s10, s5
	s_sub_u32 s5, s48, s5
	s_cselect_b64 s[52:53], -1, 0
	s_subb_u32 s31, s31, s11
	s_sub_u32 s45, s5, s10
	s_cselect_b64 s[56:57], -1, 0
	s_subb_u32 s48, s31, 0
	s_cmp_ge_u32 s48, s11
	s_cselect_b32 s51, -1, 0
	s_cmp_ge_u32 s45, s10
	s_cselect_b32 s55, -1, 0
	s_cmp_eq_u32 s48, s11
	s_cselect_b32 s51, s55, s51
	s_cmp_lg_u64 s[56:57], 0
	s_subb_u32 s31, s31, s11
	s_sub_u32 s55, s45, s10
	s_subb_u32 s31, s31, 0
	s_cmp_lg_u32 s51, 0
	s_cselect_b32 s45, s55, s45
	s_cselect_b32 s31, s31, s48
	s_cmp_lg_u64 s[52:53], 0
	s_subb_u32 s25, s49, s25
	s_cmp_ge_u32 s25, s11
	s_cselect_b32 s48, -1, 0
	s_cmp_ge_u32 s5, s10
	s_cselect_b32 s10, -1, 0
	s_cmp_eq_u32 s25, s11
	s_cselect_b32 s10, s10, s48
	s_cmp_lg_u32 s10, 0
	s_cselect_b32 s11, s31, s25
	s_cselect_b32 s10, s45, s5
	s_xor_b64 s[10:11], s[10:11], s[26:27]
	s_sub_u32 s10, s10, s26
	s_subb_u32 s11, s11, s26
	s_branch .LBB0_12
.LBB0_10:
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v1, v0
	v_mov_b64_e32 v[6:7], v[2:3]
	v_mov_b64_e32 v[10:11], v[2:3]
	v_mov_b64_e32 v[14:15], v[2:3]
	s_mov_b64 s[36:37], s[12:13]
	s_mov_b64 s[92:93], s[16:17]
	s_mov_b64 s[28:29], s[18:19]
	s_mov_b64 s[8:9], s[20:21]
	v_mov_b64_e32 v[4:5], v[0:1]
	v_mov_b64_e32 v[8:9], v[0:1]
	v_mov_b64_e32 v[12:13], v[0:1]
	s_branch .LBB0_58
.LBB0_11:
	s_mov_b64 s[6:7], -1
.LBB0_12:
	v_writelane_b32 v144, s64, 29
	s_andn2_b64 vcc, exec, s[6:7]
	s_nop 0
	v_writelane_b32 v144, s65, 30
	s_cbranch_vccnz .LBB0_14
	v_cvt_f32_u32_e32 v0, s50
	s_sub_i32 s5, 0, s50
	s_mov_b32 s11, 0
	v_rcp_iflag_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_nop 0
	v_readfirstlane_b32 s6, v0
	s_mul_i32 s5, s5, s6
	s_mul_hi_u32 s5, s6, s5
	s_add_i32 s6, s6, s5
	s_mul_hi_u32 s5, s24, s6
	s_mul_i32 s5, s5, s50
	s_sub_i32 s5, s24, s5
	s_sub_i32 s6, s5, s50
	s_cmp_ge_u32 s5, s50
	s_cselect_b32 s5, s6, s5
	s_sub_i32 s6, s5, s50
	s_cmp_ge_u32 s5, s50
	s_cselect_b32 s10, s6, s5
.LBB0_14:
	s_add_i32 s5, s50, s10
	s_ashr_i32 s6, s5, 31
	s_cmp_lt_i32 s11, 0
	s_cselect_b32 s5, s5, s10
	s_cselect_b32 s6, s6, s11
	s_add_u32 s26, s5, s22
	s_mov_b32 s31, s30
	s_addc_u32 s27, s6, s23
	s_xor_b64 s[22:23], s[30:31], s[40:41]
	s_ashr_i32 s6, s19, 31
	s_add_u32 s10, s18, s6
	s_mov_b32 s7, s6
	s_addc_u32 s11, s19, s6
	s_xor_b64 s[10:11], s[10:11], s[6:7]
	v_cvt_f32_u32_e32 v0, s10
	v_cvt_f32_u32_e32 v1, s11
	s_sub_u32 s18, 0, s10
	s_subb_u32 s19, 0, s11
	s_mov_b32 s5, 0
	v_fmac_f32_e32 v0, 0x4f800000, v1
	v_rcp_f32_e32 v0, v0
	v_lshlrev_b32_e32 v27, 4, v24
	v_lshl_add_u32 v44, v63, 8, v27
	v_mul_f32_e32 v0, 0x5f7ffffc, v0
	v_mul_f32_e32 v1, 0x2f800000, v0
	v_trunc_f32_e32 v1, v1
	v_fmac_f32_e32 v0, 0xcf800000, v1
	v_cvt_u32_f32_e32 v0, v0
	v_cvt_u32_f32_e32 v1, v1
	v_readfirstlane_b32 s24, v0
	v_readfirstlane_b32 s41, v1
	s_mul_hi_u32 s40, s18, s24
	s_mul_i32 s45, s18, s41
	s_mul_i32 s25, s19, s24
	s_add_i32 s40, s40, s45
	s_add_i32 s40, s40, s25
	s_mul_i32 s48, s18, s24
	s_mul_i32 s45, s24, s40
	s_mul_hi_u32 s49, s24, s48
	s_mul_hi_u32 s25, s24, s40
	s_add_u32 s45, s49, s45
	s_addc_u32 s25, 0, s25
	s_mul_hi_u32 s51, s41, s48
	s_mul_i32 s48, s41, s48
	s_add_u32 s45, s45, s48
	s_mul_hi_u32 s49, s41, s40
	s_addc_u32 s25, s25, s51
	s_addc_u32 s45, s49, 0
	s_mul_i32 s40, s41, s40
	s_add_u32 s25, s25, s40
	s_addc_u32 s40, 0, s45
	s_add_u32 s24, s24, s25
	s_addc_u32 s25, s41, s40
	s_mul_i32 s40, s18, s25
	s_mul_hi_u32 s41, s18, s24
	s_add_i32 s40, s41, s40
	s_mul_i32 s19, s19, s24
	s_add_i32 s40, s40, s19
	s_mul_i32 s18, s18, s24
	s_mul_hi_u32 s41, s25, s18
	s_mul_i32 s45, s25, s18
	s_mul_i32 s49, s24, s40
	s_mul_hi_u32 s18, s24, s18
	s_mul_hi_u32 s48, s24, s40
	s_add_u32 s18, s18, s49
	s_addc_u32 s48, 0, s48
	s_add_u32 s18, s18, s45
	s_mul_hi_u32 s19, s25, s40
	s_addc_u32 s18, s48, s41
	s_addc_u32 s19, s19, 0
	s_mul_i32 s40, s25, s40
	s_add_u32 s18, s18, s40
	s_addc_u32 s19, 0, s19
	s_add_u32 s24, s24, s18
	s_addc_u32 s25, s25, s19
	s_ashr_i32 s18, s23, 31
	s_add_u32 s22, s22, s18
	s_mov_b32 s19, s18
	s_addc_u32 s23, s23, s18
	s_xor_b64 s[22:23], s[22:23], s[18:19]
	s_mul_i32 s41, s22, s25
	s_mul_hi_u32 s45, s22, s24
	s_mul_hi_u32 s40, s22, s25
	s_add_u32 s41, s45, s41
	s_addc_u32 s40, 0, s40
	s_mul_hi_u32 s48, s23, s24
	s_mul_i32 s24, s23, s24
	s_add_u32 s24, s41, s24
	s_mul_hi_u32 s45, s23, s25
	s_addc_u32 s24, s40, s48
	s_addc_u32 s40, s45, 0
	s_mul_i32 s25, s23, s25
	s_add_u32 s41, s24, s25
	s_addc_u32 s40, 0, s40
	s_mul_i32 s24, s10, s40
	s_mul_hi_u32 s25, s10, s41
	s_add_i32 s24, s25, s24
	s_mul_i32 s25, s11, s41
	s_add_i32 s45, s24, s25
	s_sub_i32 s48, s23, s45
	s_mul_i32 s24, s10, s41
	s_sub_u32 s22, s22, s24
	s_cselect_b64 s[24:25], -1, 0
	s_subb_u32 s48, s48, s11
	s_sub_u32 s49, s22, s10
	s_subb_u32 s48, s48, 0
	s_cmp_ge_u32 s48, s11
	s_cselect_b32 s51, -1, 0
	s_cmp_ge_u32 s49, s10
	s_cselect_b32 s49, -1, 0
	s_cmp_eq_u32 s48, s11
	s_cselect_b32 s48, s49, s51
	s_add_u32 s49, s41, 1
	s_addc_u32 s51, s40, 0
	s_add_u32 s52, s41, 2
	s_addc_u32 s53, s40, 0
	s_cmp_lg_u32 s48, 0
	s_cselect_b32 s48, s52, s49
	s_cselect_b32 s49, s53, s51
	s_cmp_lg_u64 s[24:25], 0
	s_subb_u32 s23, s23, s45
	s_cmp_ge_u32 s23, s11
	s_cselect_b32 s24, -1, 0
	s_cmp_ge_u32 s22, s10
	s_cselect_b32 s10, -1, 0
	s_cmp_eq_u32 s23, s11
	s_cselect_b32 s10, s10, s24
	s_cmp_lg_u32 s10, 0
	s_cselect_b32 s11, s49, s40
	s_cselect_b32 s10, s48, s41
	s_xor_b64 s[6:7], s[18:19], s[6:7]
	s_xor_b64 s[10:11], s[10:11], s[6:7]
	s_sub_u32 s6, s10, s6
	s_subb_u32 s7, s11, s7
	s_xor_b64 s[6:7], s[6:7], s[30:31]
	s_lshl_b64 s[6:7], s[6:7], 5
	s_and_b32 s4, s4, 31
	s_or_b64 s[6:7], s[6:7], s[4:5]
	s_and_b64 s[10:11], s[12:13], exec
	s_cselect_b32 s13, s27, s7
	s_cselect_b32 s12, s26, s6
	s_ashr_i32 s64, s43, 31
	s_add_u32 s6, s42, s64
	s_mov_b32 s65, s64
	s_addc_u32 s7, s43, s64
	s_xor_b64 s[6:7], s[6:7], s[64:65]
	v_cvt_f32_u32_e32 v6, s6
	v_cvt_f32_u32_e32 v7, s7
	s_movk_i32 s4, 0xff00
	v_mad_i32_i24 v2, v63, s4, v44
	v_not_b32_e32 v0, v2
	v_fmac_f32_e32 v6, 0x4f800000, v7
	v_ashrrev_i32_e32 v3, 31, v2
	v_ashrrev_i32_e32 v1, 31, v0
	v_cmp_gt_i32_e64 s[10:11], 0, v2
	v_rcp_f32_e32 v6, v6
	s_sub_u32 s4, 0, s6
	v_cndmask_b32_e64 v5, v3, v1, s[10:11]
	v_cndmask_b32_e64 v4, v2, v0, s[10:11]
	v_mov_b32_e32 v0, v5
	v_mov_b32_e32 v1, v5
	v_lshl_add_u64 v[0:1], v[4:5], 0, v[0:1]
	v_xor_b32_e32 v13, v0, v5
	v_mul_f32_e32 v0, 0x5f7ffffc, v6
	v_xor_b32_e32 v12, v1, v5
	v_mul_f32_e32 v1, 0x2f800000, v0
	v_trunc_f32_e32 v1, v1
	v_fmac_f32_e32 v0, 0xcf800000, v1
	v_cvt_u32_f32_e32 v0, v0
	v_cvt_u32_f32_e32 v1, v1
	s_subb_u32 s18, 0, s7
	v_xor_b32_e32 v5, s64, v5
	v_readfirstlane_b32 s19, v0
	v_readfirstlane_b32 s24, v1
	s_mul_hi_u32 s23, s4, s19
	s_mul_i32 s25, s4, s24
	s_mul_i32 s22, s18, s19
	s_add_i32 s23, s23, s25
	s_add_i32 s23, s23, s22
	s_mul_i32 s26, s4, s19
	s_mul_i32 s25, s19, s23
	s_mul_hi_u32 s27, s19, s26
	s_mul_hi_u32 s22, s19, s23
	s_add_u32 s25, s27, s25
	s_addc_u32 s22, 0, s22
	s_mul_hi_u32 s30, s24, s26
	s_mul_i32 s26, s24, s26
	s_add_u32 s25, s25, s26
	s_mul_hi_u32 s27, s24, s23
	s_addc_u32 s22, s22, s30
	s_addc_u32 s25, s27, 0
	s_mul_i32 s23, s24, s23
	s_add_u32 s22, s22, s23
	s_addc_u32 s23, 0, s25
	s_add_u32 s19, s19, s22
	s_addc_u32 s22, s24, s23
	s_mul_i32 s23, s4, s22
	s_mul_hi_u32 s24, s4, s19
	s_add_i32 s23, s24, s23
	s_mul_i32 s18, s18, s19
	s_add_i32 s23, s23, s18
	s_mul_i32 s4, s4, s19
	s_mul_hi_u32 s24, s22, s4
	s_mul_i32 s25, s22, s4
	s_mul_i32 s27, s19, s23
	s_mul_hi_u32 s4, s19, s4
	s_mul_hi_u32 s26, s19, s23
	s_add_u32 s4, s4, s27
	s_addc_u32 s26, 0, s26
	s_add_u32 s4, s4, s25
	s_mul_hi_u32 s18, s22, s23
	s_addc_u32 s4, s26, s24
	s_addc_u32 s18, s18, 0
	s_mul_i32 s23, s22, s23
	s_add_u32 s4, s4, s23
	s_addc_u32 s18, 0, s18
	s_add_u32 s4, s19, s4
	s_addc_u32 s24, s22, s18
	v_mad_u64_u32 v[6:7], s[18:19], v13, s24, 0
	v_mul_hi_u32 v0, v13, s4
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[6:7]
	v_mad_u64_u32 v[10:11], s[18:19], v12, s4, 0
	v_add_co_u32_e32 v0, vcc, v6, v10
	v_mad_u64_u32 v[8:9], s[18:19], v12, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v6, vcc, v7, v11, vcc
	v_mov_b32_e32 v7, s5
	s_nop 0
	v_addc_co_u32_e32 v9, vcc, 0, v9, vcc
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[8:9]
	v_mul_lo_u32 v0, s7, v6
	v_mul_lo_u32 v4, s6, v7
	v_mad_u64_u32 v[8:9], s[18:19], s6, v6, 0
	v_add3_u32 v0, v9, v4, v0
	v_sub_u32_e32 v7, v12, v0
	v_mov_b32_e32 v4, s7
	v_sub_co_u32_e32 v8, vcc, v13, v8
	s_lshl_b64 s[12:13], s[12:13], 6
	s_nop 0
	v_subb_co_u32_e64 v7, s[18:19], v7, v4, vcc
	v_subrev_co_u32_e64 v9, s[18:19], s6, v8
	v_subb_co_u32_e32 v0, vcc, v12, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v7, s[18:19], 0, v7, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v7
	v_cmp_le_u32_e32 vcc, s7, v0
	s_mov_b32 s31, 0x27000
	v_cndmask_b32_e64 v10, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v9
	s_mov_b32 s30, 0x7ffffffe
	s_and_b32 s29, s29, 0xffff
	v_cndmask_b32_e64 v9, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v7
	s_cmp_lg_u32 s43, 0
	s_nop 0
	v_cndmask_b32_e64 v7, v10, v9, s[18:19]
	v_add_u32_e32 v9, 2, v6
	v_add_u32_e32 v10, 1, v6
	v_cmp_ne_u32_e64 s[18:19], 0, v7
	s_nop 1
	v_cndmask_b32_e64 v7, v10, v9, s[18:19]
	v_cndmask_b32_e64 v9, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v8
	s_nop 1
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	s_nop 1
	v_cndmask_b32_e32 v0, v9, v8, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	s_nop 1
	v_cndmask_b32_e32 v0, v6, v7, vcc
	v_xor_b32_e32 v0, v0, v5
	v_mov_b32_e32 v6, v3
	v_mov_b32_e32 v7, v3
	v_sub_u32_e32 v0, v0, v5
	v_lshlrev_b32_e32 v5, 4, v95
	v_lshl_add_u64 v[6:7], v[2:3], 0, v[6:7]
	v_or_b32_e32 v30, s12, v5
	v_xor_b32_e32 v13, v6, v3
	v_xad_u32 v5, v0, v3, v30
	v_xor_b32_e32 v12, v7, v3
	v_mad_u64_u32 v[6:7], s[18:19], v13, s24, 0
	v_mul_hi_u32 v0, v13, s4
	v_lshl_add_u64 v[6:7], v[0:1], 0, v[6:7]
	v_mad_u64_u32 v[10:11], s[18:19], v12, s4, 0
	v_add_co_u32_e32 v0, vcc, v6, v10
	v_mad_u64_u32 v[8:9], s[18:19], v12, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v6, vcc, v7, v11, vcc
	v_mov_b32_e32 v7, s5
	s_nop 0
	v_addc_co_u32_e32 v9, vcc, 0, v9, vcc
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[8:9]
	v_mul_lo_u32 v0, s7, v6
	v_mul_lo_u32 v8, s6, v7
	v_mad_u64_u32 v[6:7], s[18:19], s6, v6, 0
	v_add3_u32 v0, v7, v8, v0
	v_sub_u32_e32 v7, v12, v0
	v_sub_co_u32_e32 v6, vcc, v13, v6
	v_mul_lo_u32 v5, v5, s33
	s_nop 0
	v_subb_co_u32_e64 v7, s[18:19], v7, v4, vcc
	v_subrev_co_u32_e64 v8, s[18:19], s6, v6
	v_subb_co_u32_e32 v0, vcc, v12, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v9, s[22:23], 0, v7, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v9
	v_subb_co_u32_e64 v7, s[18:19], v7, v4, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v10, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v8
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v11, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v9
	s_nop 1
	v_cndmask_b32_e64 v10, v10, v11, s[22:23]
	v_subrev_co_u32_e64 v11, s[18:19], s6, v8
	s_nop 1
	v_subbrev_co_u32_e64 v7, s[18:19], 0, v7, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v10
	s_nop 1
	v_cndmask_b32_e64 v7, v9, v7, s[18:19]
	v_cndmask_b32_e64 v9, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v6
	s_nop 1
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	s_nop 1
	v_cndmask_b32_e32 v9, v9, v10, vcc
	v_cmp_ne_u32_e32 vcc, 0, v9
	s_nop 1
	v_cndmask_b32_e32 v0, v0, v7, vcc
	v_cndmask_b32_e64 v7, v8, v11, s[18:19]
	v_cndmask_b32_e32 v6, v6, v7, vcc
	v_xor_b32_e32 v6, v6, v3
	v_xor_b32_e32 v0, v0, v3
	v_sub_co_u32_e32 v6, vcc, v6, v3
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v3, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v5, v0, v6, v5
	v_or_b32_e32 v6, 1, v2
	v_sub_u32_e32 v0, -2, v2
	v_ashrrev_i32_e32 v7, 31, v6
	v_ashrrev_i32_e32 v8, 31, v0
	v_cndmask_b32_e64 v9, v7, v8, s[10:11]
	v_cndmask_b32_e64 v8, v6, v0, s[10:11]
	v_mov_b32_e32 v10, v9
	v_mov_b32_e32 v11, v9
	v_lshl_add_u64 v[10:11], v[8:9], 0, v[10:11]
	v_xor_b32_e32 v16, v10, v9
	v_xor_b32_e32 v8, v11, v9
	v_mad_u64_u32 v[10:11], s[18:19], v16, s24, 0
	v_mul_hi_u32 v0, v16, s4
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[10:11]
	v_mad_u64_u32 v[14:15], s[18:19], v8, s4, 0
	v_add_co_u32_e32 v0, vcc, v10, v14
	v_mad_u64_u32 v[12:13], s[18:19], v8, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v15, vcc
	v_mov_b32_e32 v11, s5
	s_nop 0
	v_addc_co_u32_e32 v13, vcc, 0, v13, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[12:13]
	v_mul_lo_u32 v0, s7, v10
	v_mul_lo_u32 v11, s6, v11
	v_mad_u64_u32 v[12:13], s[18:19], s6, v10, 0
	v_add3_u32 v0, v13, v11, v0
	v_sub_u32_e32 v11, v8, v0
	v_sub_co_u32_e32 v12, vcc, v16, v12
	s_nop 1
	v_subb_co_u32_e64 v11, s[18:19], v11, v4, vcc
	v_subrev_co_u32_e64 v13, s[18:19], s6, v12
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v11, s[18:19], 0, v11, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v11
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v14, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v13
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v12
	v_cndmask_b32_e64 v13, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v11
	v_cndmask_b32_e64 v12, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	v_cndmask_b32_e64 v11, v14, v13, s[18:19]
	v_add_u32_e32 v13, 2, v10
	v_add_u32_e32 v14, 1, v10
	v_cmp_ne_u32_e64 s[18:19], 0, v11
	v_cndmask_b32_e32 v0, v8, v12, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v11, v14, v13, s[18:19]
	v_xor_b32_e32 v8, s64, v9
	v_cndmask_b32_e32 v0, v10, v11, vcc
	v_xor_b32_e32 v0, v0, v8
	v_sub_u32_e32 v0, v0, v8
	v_mov_b32_e32 v8, v7
	v_mov_b32_e32 v9, v7
	v_lshl_add_u64 v[8:9], v[6:7], 0, v[8:9]
	v_xor_b32_e32 v15, v8, v7
	v_xad_u32 v14, v0, v3, v30
	v_xor_b32_e32 v6, v9, v7
	v_mad_u64_u32 v[8:9], s[18:19], v15, s24, 0
	v_mul_hi_u32 v0, v15, s4
	v_lshl_add_u64 v[8:9], v[0:1], 0, v[8:9]
	v_mad_u64_u32 v[12:13], s[18:19], v6, s4, 0
	v_add_co_u32_e32 v0, vcc, v8, v12
	v_mad_u64_u32 v[10:11], s[18:19], v6, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v8, vcc, v9, v13, vcc
	v_mov_b32_e32 v9, s5
	s_nop 0
	v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[10:11]
	v_mul_lo_u32 v0, s7, v8
	v_mul_lo_u32 v10, s6, v9
	v_mad_u64_u32 v[8:9], s[18:19], s6, v8, 0
	v_add3_u32 v0, v9, v10, v0
	v_sub_u32_e32 v9, v6, v0
	v_sub_co_u32_e32 v8, vcc, v15, v8
	s_nop 1
	v_subb_co_u32_e64 v9, s[18:19], v9, v4, vcc
	v_subrev_co_u32_e64 v10, s[18:19], s6, v8
	v_subb_co_u32_e32 v0, vcc, v6, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v11, s[22:23], 0, v9, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v11
	v_subb_co_u32_e64 v9, s[18:19], v9, v4, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v12, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v10
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v13, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v11
	v_cndmask_b32_e64 v6, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v8
	v_cndmask_b32_e64 v12, v12, v13, s[22:23]
	v_subrev_co_u32_e64 v13, s[18:19], s6, v10
	s_nop 1
	v_subbrev_co_u32_e64 v9, s[18:19], 0, v9, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v12
	s_nop 1
	v_cndmask_b32_e64 v9, v11, v9, s[18:19]
	v_cndmask_b32_e64 v11, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	s_nop 1
	v_cndmask_b32_e32 v6, v6, v11, vcc
	v_cmp_ne_u32_e32 vcc, 0, v6
	v_cndmask_b32_e64 v6, v10, v13, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v6, v8, v6, vcc
	v_cndmask_b32_e32 v0, v0, v9, vcc
	v_xor_b32_e32 v6, v6, v7
	v_xor_b32_e32 v0, v0, v7
	v_sub_co_u32_e32 v6, vcc, v6, v7
	v_or_b32_e32 v8, 2, v2
	s_nop 0
	v_subb_co_u32_e32 v0, vcc, v0, v7, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_mul_lo_u32 v7, v14, s33
	v_add3_u32 v6, v0, v6, v7
	v_sub_u32_e32 v0, -3, v2
	v_ashrrev_i32_e32 v9, 31, v8
	v_ashrrev_i32_e32 v7, 31, v0
	v_cndmask_b32_e64 v11, v9, v7, s[10:11]
	v_cndmask_b32_e64 v10, v8, v0, s[10:11]
	v_mov_b32_e32 v12, v11
	v_mov_b32_e32 v13, v11
	v_lshl_add_u64 v[12:13], v[10:11], 0, v[12:13]
	v_xor_b32_e32 v10, v12, v11
	v_xor_b32_e32 v7, v13, v11
	v_mad_u64_u32 v[12:13], s[18:19], v10, s24, 0
	v_mul_hi_u32 v0, v10, s4
	v_lshl_add_u64 v[12:13], v[0:1], 0, v[12:13]
	v_mad_u64_u32 v[16:17], s[18:19], v7, s4, 0
	v_add_co_u32_e32 v0, vcc, v12, v16
	v_mad_u64_u32 v[14:15], s[18:19], v7, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v13, v17, vcc
	v_mov_b32_e32 v13, s5
	s_nop 0
	v_addc_co_u32_e32 v15, vcc, 0, v15, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[14:15]
	v_mul_lo_u32 v0, s7, v12
	v_mul_lo_u32 v13, s6, v13
	v_mad_u64_u32 v[14:15], s[18:19], s6, v12, 0
	v_add3_u32 v0, v15, v13, v0
	v_sub_u32_e32 v13, v7, v0
	v_sub_co_u32_e32 v10, vcc, v10, v14
	s_nop 1
	v_subb_co_u32_e64 v13, s[18:19], v13, v4, vcc
	v_subrev_co_u32_e64 v14, s[18:19], s6, v10
	v_subb_co_u32_e32 v0, vcc, v7, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v13, s[18:19], 0, v13, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v13
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v15, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v14
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v10
	v_cndmask_b32_e64 v14, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v13
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	v_cndmask_b32_e64 v13, v15, v14, s[18:19]
	v_add_u32_e32 v14, 2, v12
	v_add_u32_e32 v15, 1, v12
	v_cmp_ne_u32_e64 s[18:19], 0, v13
	v_cndmask_b32_e32 v0, v7, v10, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v13, v15, v14, s[18:19]
	v_xor_b32_e32 v7, s64, v11
	v_cndmask_b32_e32 v0, v12, v13, vcc
	v_mov_b32_e32 v10, v9
	v_mov_b32_e32 v11, v9
	v_xor_b32_e32 v0, v0, v7
	v_lshl_add_u64 v[10:11], v[8:9], 0, v[10:11]
	v_sub_u32_e32 v0, v0, v7
	v_xor_b32_e32 v16, v10, v9
	v_xad_u32 v7, v0, v3, v30
	v_xor_b32_e32 v8, v11, v9
	v_mad_u64_u32 v[10:11], s[18:19], v16, s24, 0
	v_mul_hi_u32 v0, v16, s4
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[10:11]
	v_mad_u64_u32 v[14:15], s[18:19], v8, s4, 0
	v_add_co_u32_e32 v0, vcc, v10, v14
	v_mad_u64_u32 v[12:13], s[18:19], v8, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v15, vcc
	v_mov_b32_e32 v11, s5
	s_nop 0
	v_addc_co_u32_e32 v13, vcc, 0, v13, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[12:13]
	v_mul_lo_u32 v0, s7, v10
	v_mul_lo_u32 v12, s6, v11
	v_mad_u64_u32 v[10:11], s[18:19], s6, v10, 0
	v_add3_u32 v0, v11, v12, v0
	v_sub_u32_e32 v11, v8, v0
	v_sub_co_u32_e32 v10, vcc, v16, v10
	v_mul_lo_u32 v7, v7, s33
	s_nop 0
	v_subb_co_u32_e64 v11, s[18:19], v11, v4, vcc
	v_subrev_co_u32_e64 v12, s[18:19], s6, v10
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v13, s[22:23], 0, v11, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v13
	v_subb_co_u32_e64 v11, s[18:19], v11, v4, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v14, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v12
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v15, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v13
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v10
	v_cndmask_b32_e64 v14, v14, v15, s[22:23]
	v_subrev_co_u32_e64 v15, s[18:19], s6, v12
	s_nop 1
	v_subbrev_co_u32_e64 v11, s[18:19], 0, v11, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v14
	s_nop 1
	v_cndmask_b32_e64 v11, v13, v11, s[18:19]
	v_cndmask_b32_e64 v13, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	s_nop 1
	v_cndmask_b32_e32 v8, v8, v13, vcc
	v_cmp_ne_u32_e32 vcc, 0, v8
	v_cndmask_b32_e64 v8, v12, v15, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v8, v10, v8, vcc
	v_cndmask_b32_e32 v0, v0, v11, vcc
	v_xor_b32_e32 v8, v8, v9
	v_xor_b32_e32 v0, v0, v9
	v_sub_co_u32_e32 v8, vcc, v8, v9
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v9, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v7, v0, v8, v7
	v_or_b32_e32 v8, 3, v2
	v_sub_u32_e32 v0, -4, v2
	v_ashrrev_i32_e32 v9, 31, v8
	v_ashrrev_i32_e32 v10, 31, v0
	v_cndmask_b32_e64 v11, v9, v10, s[10:11]
	v_cndmask_b32_e64 v10, v8, v0, s[10:11]
	v_mov_b32_e32 v12, v11
	v_mov_b32_e32 v13, v11
	v_lshl_add_u64 v[12:13], v[10:11], 0, v[12:13]
	v_xor_b32_e32 v18, v12, v11
	v_xor_b32_e32 v10, v13, v11
	v_mad_u64_u32 v[12:13], s[18:19], v18, s24, 0
	v_mul_hi_u32 v0, v18, s4
	v_lshl_add_u64 v[12:13], v[0:1], 0, v[12:13]
	v_mad_u64_u32 v[16:17], s[18:19], v10, s4, 0
	v_add_co_u32_e32 v0, vcc, v12, v16
	v_mad_u64_u32 v[14:15], s[18:19], v10, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v13, v17, vcc
	v_mov_b32_e32 v13, s5
	s_nop 0
	v_addc_co_u32_e32 v15, vcc, 0, v15, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[14:15]
	v_mul_lo_u32 v0, s7, v12
	v_mul_lo_u32 v13, s6, v13
	v_mad_u64_u32 v[14:15], s[18:19], s6, v12, 0
	v_add3_u32 v0, v15, v13, v0
	v_sub_u32_e32 v13, v10, v0
	v_sub_co_u32_e32 v14, vcc, v18, v14
	s_nop 1
	v_subb_co_u32_e64 v13, s[18:19], v13, v4, vcc
	v_subrev_co_u32_e64 v15, s[18:19], s6, v14
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v13, s[18:19], 0, v13, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v13
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v16, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v15
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v14
	v_cndmask_b32_e64 v15, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v13
	v_cndmask_b32_e64 v14, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	v_cndmask_b32_e64 v13, v16, v15, s[18:19]
	v_add_u32_e32 v15, 2, v12
	v_add_u32_e32 v16, 1, v12
	v_cmp_ne_u32_e64 s[18:19], 0, v13
	v_cndmask_b32_e32 v0, v10, v14, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v13, v16, v15, s[18:19]
	v_xor_b32_e32 v10, s64, v11
	v_cndmask_b32_e32 v0, v12, v13, vcc
	v_xor_b32_e32 v0, v0, v10
	v_sub_u32_e32 v0, v0, v10
	v_mov_b32_e32 v10, v9
	v_mov_b32_e32 v11, v9
	v_lshl_add_u64 v[10:11], v[8:9], 0, v[10:11]
	v_xor_b32_e32 v17, v10, v9
	v_xad_u32 v16, v0, v3, v30
	v_xor_b32_e32 v8, v11, v9
	v_mad_u64_u32 v[10:11], s[18:19], v17, s24, 0
	v_mul_hi_u32 v0, v17, s4
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[10:11]
	v_mad_u64_u32 v[14:15], s[18:19], v8, s4, 0
	v_add_co_u32_e32 v0, vcc, v10, v14
	v_mad_u64_u32 v[12:13], s[18:19], v8, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v15, vcc
	v_mov_b32_e32 v11, s5
	s_nop 0
	v_addc_co_u32_e32 v13, vcc, 0, v13, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[12:13]
	v_mul_lo_u32 v0, s7, v10
	v_mul_lo_u32 v12, s6, v11
	v_mad_u64_u32 v[10:11], s[18:19], s6, v10, 0
	v_add3_u32 v0, v11, v12, v0
	v_sub_u32_e32 v11, v8, v0
	v_sub_co_u32_e32 v10, vcc, v17, v10
	s_nop 1
	v_subb_co_u32_e64 v11, s[18:19], v11, v4, vcc
	v_subrev_co_u32_e64 v12, s[18:19], s6, v10
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v13, s[22:23], 0, v11, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v13
	v_subb_co_u32_e64 v11, s[18:19], v11, v4, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v14, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v12
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v15, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v13
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v10
	v_cndmask_b32_e64 v14, v14, v15, s[22:23]
	v_subrev_co_u32_e64 v15, s[18:19], s6, v12
	s_nop 1
	v_subbrev_co_u32_e64 v11, s[18:19], 0, v11, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v14
	s_nop 1
	v_cndmask_b32_e64 v11, v13, v11, s[18:19]
	v_cndmask_b32_e64 v13, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	s_nop 1
	v_cndmask_b32_e32 v8, v8, v13, vcc
	v_cmp_ne_u32_e32 vcc, 0, v8
	v_cndmask_b32_e64 v8, v12, v15, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v8, v10, v8, vcc
	v_cndmask_b32_e32 v0, v0, v11, vcc
	v_xor_b32_e32 v8, v8, v9
	v_xor_b32_e32 v0, v0, v9
	v_sub_co_u32_e32 v8, vcc, v8, v9
	v_or_b32_e32 v10, 4, v2
	s_nop 0
	v_subb_co_u32_e32 v0, vcc, v0, v9, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_mul_lo_u32 v9, v16, s33
	v_add3_u32 v8, v0, v8, v9
	v_sub_u32_e32 v0, -5, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v9, 31, v0
	v_cndmask_b32_e64 v13, v11, v9, s[10:11]
	v_cndmask_b32_e64 v12, v10, v0, s[10:11]
	v_mov_b32_e32 v14, v13
	v_mov_b32_e32 v15, v13
	v_lshl_add_u64 v[14:15], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v12, v14, v13
	v_xor_b32_e32 v9, v15, v13
	v_mad_u64_u32 v[14:15], s[18:19], v12, s24, 0
	v_mul_hi_u32 v0, v12, s4
	v_lshl_add_u64 v[14:15], v[0:1], 0, v[14:15]
	v_mad_u64_u32 v[18:19], s[18:19], v9, s4, 0
	v_add_co_u32_e32 v0, vcc, v14, v18
	v_mad_u64_u32 v[16:17], s[18:19], v9, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v14, vcc, v15, v19, vcc
	v_mov_b32_e32 v15, s5
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[16:17]
	v_mul_lo_u32 v0, s7, v14
	v_mul_lo_u32 v15, s6, v15
	v_mad_u64_u32 v[16:17], s[18:19], s6, v14, 0
	v_add3_u32 v0, v17, v15, v0
	v_sub_u32_e32 v15, v9, v0
	v_sub_co_u32_e32 v12, vcc, v12, v16
	s_nop 1
	v_subb_co_u32_e64 v15, s[18:19], v15, v4, vcc
	v_subrev_co_u32_e64 v16, s[18:19], s6, v12
	v_subb_co_u32_e32 v0, vcc, v9, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v15, s[18:19], 0, v15, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v15
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v16
	v_cndmask_b32_e64 v9, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v12
	v_cndmask_b32_e64 v16, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v15
	v_cndmask_b32_e64 v12, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	v_cndmask_b32_e64 v15, v17, v16, s[18:19]
	v_add_u32_e32 v16, 2, v14
	v_add_u32_e32 v17, 1, v14
	v_cmp_ne_u32_e64 s[18:19], 0, v15
	v_cndmask_b32_e32 v0, v9, v12, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v15, v17, v16, s[18:19]
	v_xor_b32_e32 v9, s64, v13
	v_cndmask_b32_e32 v0, v14, v15, vcc
	v_mov_b32_e32 v12, v11
	v_mov_b32_e32 v13, v11
	v_xor_b32_e32 v0, v0, v9
	v_lshl_add_u64 v[12:13], v[10:11], 0, v[12:13]
	v_sub_u32_e32 v0, v0, v9
	v_xor_b32_e32 v18, v12, v11
	v_xad_u32 v9, v0, v3, v30
	v_xor_b32_e32 v10, v13, v11
	v_mad_u64_u32 v[12:13], s[18:19], v18, s24, 0
	v_mul_hi_u32 v0, v18, s4
	v_lshl_add_u64 v[12:13], v[0:1], 0, v[12:13]
	v_mad_u64_u32 v[16:17], s[18:19], v10, s4, 0
	v_add_co_u32_e32 v0, vcc, v12, v16
	v_mad_u64_u32 v[14:15], s[18:19], v10, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v13, v17, vcc
	v_mov_b32_e32 v13, s5
	s_nop 0
	v_addc_co_u32_e32 v15, vcc, 0, v15, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[14:15]
	v_mul_lo_u32 v0, s7, v12
	v_mul_lo_u32 v14, s6, v13
	v_mad_u64_u32 v[12:13], s[18:19], s6, v12, 0
	v_add3_u32 v0, v13, v14, v0
	v_sub_u32_e32 v13, v10, v0
	v_sub_co_u32_e32 v12, vcc, v18, v12
	v_mul_lo_u32 v9, v9, s33
	s_nop 0
	v_subb_co_u32_e64 v13, s[18:19], v13, v4, vcc
	v_subrev_co_u32_e64 v14, s[18:19], s6, v12
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v15, s[22:23], 0, v13, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v15
	v_subb_co_u32_e64 v13, s[18:19], v13, v4, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v16, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v14
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v15
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v12
	v_cndmask_b32_e64 v16, v16, v17, s[22:23]
	v_subrev_co_u32_e64 v17, s[18:19], s6, v14
	s_nop 1
	v_subbrev_co_u32_e64 v13, s[18:19], 0, v13, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v16
	s_nop 1
	v_cndmask_b32_e64 v13, v15, v13, s[18:19]
	v_cndmask_b32_e64 v15, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v15, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v14, v17, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v10, v12, v10, vcc
	v_cndmask_b32_e32 v0, v0, v13, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v20, v0, v10, v9
	v_or_b32_e32 v10, 5, v2
	v_sub_u32_e32 v0, -6, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v9, 31, v0
	v_cndmask_b32_e64 v13, v11, v9, s[10:11]
	v_cndmask_b32_e64 v12, v10, v0, s[10:11]
	v_mov_b32_e32 v14, v13
	v_mov_b32_e32 v15, v13
	v_lshl_add_u64 v[14:15], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v12, v14, v13
	v_xor_b32_e32 v9, v15, v13
	v_mad_u64_u32 v[14:15], s[18:19], v12, s24, 0
	v_mul_hi_u32 v0, v12, s4
	v_lshl_add_u64 v[14:15], v[0:1], 0, v[14:15]
	v_mad_u64_u32 v[18:19], s[18:19], v9, s4, 0
	v_add_co_u32_e32 v0, vcc, v14, v18
	v_mad_u64_u32 v[16:17], s[18:19], v9, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v14, vcc, v15, v19, vcc
	v_mov_b32_e32 v15, s5
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[16:17]
	v_mul_lo_u32 v0, s7, v14
	v_mul_lo_u32 v15, s6, v15
	v_mad_u64_u32 v[16:17], s[18:19], s6, v14, 0
	v_add3_u32 v0, v17, v15, v0
	v_sub_u32_e32 v15, v9, v0
	v_sub_co_u32_e32 v12, vcc, v12, v16
	s_nop 1
	v_subb_co_u32_e64 v15, s[18:19], v15, v4, vcc
	v_subrev_co_u32_e64 v16, s[18:19], s6, v12
	v_subb_co_u32_e32 v0, vcc, v9, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v15, s[18:19], 0, v15, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v15
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v16
	v_cndmask_b32_e64 v9, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v12
	v_cndmask_b32_e64 v16, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v15
	v_cndmask_b32_e64 v12, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	v_cndmask_b32_e64 v15, v17, v16, s[18:19]
	v_add_u32_e32 v16, 2, v14
	v_add_u32_e32 v17, 1, v14
	v_cmp_ne_u32_e64 s[18:19], 0, v15
	v_cndmask_b32_e32 v0, v9, v12, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v15, v17, v16, s[18:19]
	v_xor_b32_e32 v9, s64, v13
	v_cndmask_b32_e32 v0, v14, v15, vcc
	v_mov_b32_e32 v12, v11
	v_mov_b32_e32 v13, v11
	v_xor_b32_e32 v0, v0, v9
	v_lshl_add_u64 v[12:13], v[10:11], 0, v[12:13]
	v_sub_u32_e32 v0, v0, v9
	v_xor_b32_e32 v18, v12, v11
	v_xad_u32 v9, v0, v3, v30
	v_xor_b32_e32 v10, v13, v11
	v_mad_u64_u32 v[12:13], s[18:19], v18, s24, 0
	v_mul_hi_u32 v0, v18, s4
	v_lshl_add_u64 v[12:13], v[0:1], 0, v[12:13]
	v_mad_u64_u32 v[16:17], s[18:19], v10, s4, 0
	v_add_co_u32_e32 v0, vcc, v12, v16
	v_mad_u64_u32 v[14:15], s[18:19], v10, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v13, v17, vcc
	v_mov_b32_e32 v13, s5
	s_nop 0
	v_addc_co_u32_e32 v15, vcc, 0, v15, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[14:15]
	v_mul_lo_u32 v0, s7, v12
	v_mul_lo_u32 v14, s6, v13
	v_mad_u64_u32 v[12:13], s[18:19], s6, v12, 0
	v_add3_u32 v0, v13, v14, v0
	v_sub_u32_e32 v13, v10, v0
	v_sub_co_u32_e32 v12, vcc, v18, v12
	v_mul_lo_u32 v9, v9, s33
	s_nop 0
	v_subb_co_u32_e64 v13, s[18:19], v13, v4, vcc
	v_subrev_co_u32_e64 v14, s[18:19], s6, v12
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v15, s[22:23], 0, v13, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v15
	v_subb_co_u32_e64 v13, s[18:19], v13, v4, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v16, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v14
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v15
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v12
	v_cndmask_b32_e64 v16, v16, v17, s[22:23]
	v_subrev_co_u32_e64 v17, s[18:19], s6, v14
	s_nop 1
	v_subbrev_co_u32_e64 v13, s[18:19], 0, v13, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v16
	s_nop 1
	v_cndmask_b32_e64 v13, v15, v13, s[18:19]
	v_cndmask_b32_e64 v15, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v15, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v14, v17, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v10, v12, v10, vcc
	v_cndmask_b32_e32 v0, v0, v13, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v21, v0, v10, v9
	v_or_b32_e32 v10, 6, v2
	v_sub_u32_e32 v0, -7, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v9, 31, v0
	v_cndmask_b32_e64 v13, v11, v9, s[10:11]
	v_cndmask_b32_e64 v12, v10, v0, s[10:11]
	v_mov_b32_e32 v14, v13
	v_mov_b32_e32 v15, v13
	v_lshl_add_u64 v[14:15], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v12, v14, v13
	v_xor_b32_e32 v9, v15, v13
	v_mad_u64_u32 v[14:15], s[18:19], v12, s24, 0
	v_mul_hi_u32 v0, v12, s4
	v_lshl_add_u64 v[14:15], v[0:1], 0, v[14:15]
	v_mad_u64_u32 v[18:19], s[18:19], v9, s4, 0
	v_add_co_u32_e32 v0, vcc, v14, v18
	v_mad_u64_u32 v[16:17], s[18:19], v9, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v14, vcc, v15, v19, vcc
	v_mov_b32_e32 v15, s5
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[16:17]
	v_mul_lo_u32 v0, s7, v14
	v_mul_lo_u32 v15, s6, v15
	v_mad_u64_u32 v[16:17], s[18:19], s6, v14, 0
	v_add3_u32 v0, v17, v15, v0
	v_sub_u32_e32 v15, v9, v0
	v_sub_co_u32_e32 v12, vcc, v12, v16
	s_nop 1
	v_subb_co_u32_e64 v15, s[18:19], v15, v4, vcc
	v_subrev_co_u32_e64 v16, s[18:19], s6, v12
	v_subb_co_u32_e32 v0, vcc, v9, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v15, s[18:19], 0, v15, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v15
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v16
	v_cndmask_b32_e64 v9, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v12
	v_cndmask_b32_e64 v16, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v15
	v_cndmask_b32_e64 v12, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	v_cndmask_b32_e64 v15, v17, v16, s[18:19]
	v_add_u32_e32 v16, 2, v14
	v_add_u32_e32 v17, 1, v14
	v_cmp_ne_u32_e64 s[18:19], 0, v15
	v_cndmask_b32_e32 v0, v9, v12, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v15, v17, v16, s[18:19]
	v_xor_b32_e32 v9, s64, v13
	v_cndmask_b32_e32 v0, v14, v15, vcc
	v_mov_b32_e32 v12, v11
	v_mov_b32_e32 v13, v11
	v_xor_b32_e32 v0, v0, v9
	v_lshl_add_u64 v[12:13], v[10:11], 0, v[12:13]
	v_sub_u32_e32 v0, v0, v9
	v_xor_b32_e32 v18, v12, v11
	v_xad_u32 v9, v0, v3, v30
	v_xor_b32_e32 v10, v13, v11
	v_mad_u64_u32 v[12:13], s[18:19], v18, s24, 0
	v_mul_hi_u32 v0, v18, s4
	v_lshl_add_u64 v[12:13], v[0:1], 0, v[12:13]
	v_mad_u64_u32 v[16:17], s[18:19], v10, s4, 0
	v_add_co_u32_e32 v0, vcc, v12, v16
	v_mad_u64_u32 v[14:15], s[18:19], v10, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v13, v17, vcc
	v_mov_b32_e32 v13, s5
	s_nop 0
	v_addc_co_u32_e32 v15, vcc, 0, v15, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[14:15]
	v_mul_lo_u32 v0, s7, v12
	v_mul_lo_u32 v14, s6, v13
	v_mad_u64_u32 v[12:13], s[18:19], s6, v12, 0
	v_add3_u32 v0, v13, v14, v0
	v_sub_u32_e32 v13, v10, v0
	v_sub_co_u32_e32 v12, vcc, v18, v12
	v_mul_lo_u32 v9, v9, s33
	s_nop 0
	v_subb_co_u32_e64 v13, s[18:19], v13, v4, vcc
	v_subrev_co_u32_e64 v14, s[18:19], s6, v12
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v15, s[22:23], 0, v13, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v15
	v_subb_co_u32_e64 v13, s[18:19], v13, v4, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v16, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v14
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v15
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v12
	v_cndmask_b32_e64 v16, v16, v17, s[22:23]
	v_subrev_co_u32_e64 v17, s[18:19], s6, v14
	s_nop 1
	v_subbrev_co_u32_e64 v13, s[18:19], 0, v13, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v16
	s_nop 1
	v_cndmask_b32_e64 v13, v15, v13, s[18:19]
	v_cndmask_b32_e64 v15, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v15, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v14, v17, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v10, v12, v10, vcc
	v_cndmask_b32_e32 v0, v0, v13, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v22, v0, v10, v9
	v_or_b32_e32 v10, 7, v2
	v_sub_u32_e32 v0, -8, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v9, 31, v0
	v_cndmask_b32_e64 v13, v11, v9, s[10:11]
	v_cndmask_b32_e64 v12, v10, v0, s[10:11]
	v_mov_b32_e32 v14, v13
	v_mov_b32_e32 v15, v13
	v_lshl_add_u64 v[14:15], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v12, v14, v13
	v_xor_b32_e32 v9, v15, v13
	v_mad_u64_u32 v[14:15], s[18:19], v12, s24, 0
	v_mul_hi_u32 v0, v12, s4
	v_lshl_add_u64 v[14:15], v[0:1], 0, v[14:15]
	v_mad_u64_u32 v[18:19], s[18:19], v9, s4, 0
	v_add_co_u32_e32 v0, vcc, v14, v18
	v_mad_u64_u32 v[16:17], s[18:19], v9, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v14, vcc, v15, v19, vcc
	v_mov_b32_e32 v15, s5
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[14:15], v[14:15], 0, v[16:17]
	v_mul_lo_u32 v0, s7, v14
	v_mul_lo_u32 v15, s6, v15
	v_mad_u64_u32 v[16:17], s[18:19], s6, v14, 0
	v_add3_u32 v0, v17, v15, v0
	v_sub_u32_e32 v15, v9, v0
	v_sub_co_u32_e32 v12, vcc, v12, v16
	s_nop 1
	v_subb_co_u32_e64 v15, s[18:19], v15, v4, vcc
	v_subrev_co_u32_e64 v16, s[18:19], s6, v12
	v_subb_co_u32_e32 v0, vcc, v9, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v15, s[18:19], 0, v15, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v15
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v16
	v_cndmask_b32_e64 v9, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v12
	v_cndmask_b32_e64 v16, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v15
	v_cndmask_b32_e64 v12, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	v_cndmask_b32_e64 v15, v17, v16, s[18:19]
	v_add_u32_e32 v16, 2, v14
	v_add_u32_e32 v17, 1, v14
	v_cmp_ne_u32_e64 s[18:19], 0, v15
	v_cndmask_b32_e32 v0, v9, v12, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v15, v17, v16, s[18:19]
	v_xor_b32_e32 v9, s64, v13
	v_cndmask_b32_e32 v0, v14, v15, vcc
	v_mov_b32_e32 v12, v11
	v_mov_b32_e32 v13, v11
	v_xor_b32_e32 v0, v0, v9
	v_lshl_add_u64 v[12:13], v[10:11], 0, v[12:13]
	v_sub_u32_e32 v0, v0, v9
	v_xor_b32_e32 v18, v12, v11
	v_xad_u32 v9, v0, v3, v30
	v_xor_b32_e32 v10, v13, v11
	v_mad_u64_u32 v[12:13], s[18:19], v18, s24, 0
	v_mul_hi_u32 v0, v18, s4
	v_lshl_add_u64 v[12:13], v[0:1], 0, v[12:13]
	v_mad_u64_u32 v[16:17], s[18:19], v10, s4, 0
	v_add_co_u32_e32 v0, vcc, v12, v16
	v_mad_u64_u32 v[14:15], s[18:19], v10, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v13, v17, vcc
	v_mov_b32_e32 v13, s5
	s_nop 0
	v_addc_co_u32_e32 v15, vcc, 0, v15, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[14:15]
	v_mul_lo_u32 v0, s7, v12
	v_mul_lo_u32 v14, s6, v13
	v_mad_u64_u32 v[12:13], s[18:19], s6, v12, 0
	v_add3_u32 v0, v13, v14, v0
	v_sub_u32_e32 v13, v10, v0
	v_sub_co_u32_e32 v12, vcc, v18, v12
	v_mul_lo_u32 v9, v9, s33
	s_nop 0
	v_subb_co_u32_e64 v13, s[18:19], v13, v4, vcc
	v_subrev_co_u32_e64 v14, s[18:19], s6, v12
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v15, s[22:23], 0, v13, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v15
	v_subb_co_u32_e64 v13, s[18:19], v13, v4, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v16, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v14
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v15
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v12
	v_cndmask_b32_e64 v16, v16, v17, s[22:23]
	v_subrev_co_u32_e64 v17, s[18:19], s6, v14
	s_nop 1
	v_subbrev_co_u32_e64 v13, s[18:19], 0, v13, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v16
	s_nop 1
	v_cndmask_b32_e64 v13, v15, v13, s[18:19]
	v_cndmask_b32_e64 v15, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v15, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v14, v17, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v10, v12, v10, vcc
	v_cndmask_b32_e32 v0, v0, v13, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v0, v0, v10, v9
	buffer_load_ubyte v9, v5, s[28:31], 0 offen
	buffer_load_ubyte v13, v6, s[28:31], 0 offen
	buffer_load_ubyte v12, v7, s[28:31], 0 offen
	buffer_load_ubyte v14, v8, s[28:31], 0 offen
	buffer_load_ubyte v15, v20, s[28:31], 0 offen
	buffer_load_ubyte v17, v21, s[28:31], 0 offen
	buffer_load_ubyte v16, v22, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v20, v0, s[28:31], 0 offen
	v_or_b32_e32 v6, 8, v2
	v_sub_u32_e32 v0, -9, v2
	v_ashrrev_i32_e32 v7, 31, v6
	v_ashrrev_i32_e32 v5, 31, v0
	v_cndmask_b32_e64 v11, v7, v5, s[10:11]
	v_cndmask_b32_e64 v10, v6, v0, s[10:11]
	v_mov_b32_e32 v18, v11
	v_mov_b32_e32 v19, v11
	v_lshl_add_u64 v[18:19], v[10:11], 0, v[18:19]
	v_xor_b32_e32 v8, v18, v11
	v_xor_b32_e32 v5, v19, v11
	v_mad_u64_u32 v[18:19], s[18:19], v8, s24, 0
	v_mul_hi_u32 v0, v8, s4
	v_lshl_add_u64 v[18:19], v[0:1], 0, v[18:19]
	v_mad_u64_u32 v[42:43], s[18:19], v5, s4, 0
	v_add_co_u32_e32 v0, vcc, v18, v42
	v_mad_u64_u32 v[22:23], s[18:19], v5, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v19, v43, vcc
	v_mov_b32_e32 v19, s5
	s_nop 0
	v_addc_co_u32_e32 v23, vcc, 0, v23, vcc
	v_lshl_add_u64 v[18:19], v[18:19], 0, v[22:23]
	v_mul_lo_u32 v0, s7, v18
	v_mul_lo_u32 v10, s6, v19
	v_mad_u64_u32 v[22:23], s[18:19], s6, v18, 0
	v_add3_u32 v0, v23, v10, v0
	v_sub_u32_e32 v10, v5, v0
	v_sub_co_u32_e32 v8, vcc, v8, v22
	s_nop 1
	v_subb_co_u32_e64 v10, s[18:19], v10, v4, vcc
	v_subrev_co_u32_e64 v19, s[18:19], s6, v8
	v_subb_co_u32_e32 v0, vcc, v5, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v10, s[18:19], 0, v10, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v10
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v21, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v19
	v_cndmask_b32_e64 v5, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v8
	v_cndmask_b32_e64 v19, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v10
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	v_cndmask_b32_e64 v10, v21, v19, s[18:19]
	v_add_u32_e32 v19, 2, v18
	v_add_u32_e32 v21, 1, v18
	v_cmp_ne_u32_e64 s[18:19], 0, v10
	v_cndmask_b32_e32 v0, v5, v8, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v10, v21, v19, s[18:19]
	v_xor_b32_e32 v5, s64, v11
	v_cndmask_b32_e32 v0, v18, v10, vcc
	v_mov_b32_e32 v10, v7
	v_mov_b32_e32 v11, v7
	v_xor_b32_e32 v0, v0, v5
	v_lshl_add_u64 v[10:11], v[6:7], 0, v[10:11]
	v_sub_u32_e32 v0, v0, v5
	v_xor_b32_e32 v8, v10, v7
	v_xad_u32 v5, v0, v3, v30
	v_xor_b32_e32 v6, v11, v7
	v_mad_u64_u32 v[10:11], s[18:19], v8, s24, 0
	v_mul_hi_u32 v0, v8, s4
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[10:11]
	v_mad_u64_u32 v[22:23], s[18:19], v6, s4, 0
	v_add_co_u32_e32 v0, vcc, v10, v22
	v_mad_u64_u32 v[18:19], s[18:19], v6, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v23, vcc
	v_mov_b32_e32 v11, s5
	s_nop 0
	v_addc_co_u32_e32 v19, vcc, 0, v19, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[18:19]
	v_mul_lo_u32 v0, s7, v10
	v_mul_lo_u32 v18, s6, v11
	v_mad_u64_u32 v[10:11], s[18:19], s6, v10, 0
	v_add3_u32 v0, v11, v18, v0
	v_sub_u32_e32 v11, v6, v0
	v_sub_co_u32_e32 v8, vcc, v8, v10
	v_mul_lo_u32 v5, v5, s33
	s_nop 0
	v_subb_co_u32_e64 v10, s[18:19], v11, v4, vcc
	v_subrev_co_u32_e64 v11, s[18:19], s6, v8
	v_subb_co_u32_e32 v0, vcc, v6, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v18, s[22:23], 0, v10, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v18
	v_subb_co_u32_e64 v10, s[18:19], v10, v4, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v19, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v11
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v21, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v18
	v_cndmask_b32_e64 v6, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v8
	v_cndmask_b32_e64 v19, v19, v21, s[22:23]
	v_subrev_co_u32_e64 v21, s[18:19], s6, v11
	s_nop 1
	v_subbrev_co_u32_e64 v10, s[18:19], 0, v10, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v19
	s_nop 1
	v_cndmask_b32_e64 v10, v18, v10, s[18:19]
	v_cndmask_b32_e64 v18, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	s_nop 1
	v_cndmask_b32_e32 v6, v6, v18, vcc
	v_cmp_ne_u32_e32 vcc, 0, v6
	v_cndmask_b32_e64 v6, v11, v21, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v6, v8, v6, vcc
	v_cndmask_b32_e32 v0, v0, v10, vcc
	v_xor_b32_e32 v6, v6, v7
	v_xor_b32_e32 v0, v0, v7
	v_sub_co_u32_e32 v6, vcc, v6, v7
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v7, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v5, v0, v6, v5
	v_or_b32_e32 v6, 9, v2
	v_sub_u32_e32 v0, -10, v2
	v_ashrrev_i32_e32 v7, 31, v6
	v_ashrrev_i32_e32 v8, 31, v0
	v_cndmask_b32_e64 v11, v7, v8, s[10:11]
	v_cndmask_b32_e64 v10, v6, v0, s[10:11]
	v_mov_b32_e32 v18, v11
	v_mov_b32_e32 v19, v11
	v_lshl_add_u64 v[18:19], v[10:11], 0, v[18:19]
	v_xor_b32_e32 v10, v18, v11
	v_xor_b32_e32 v8, v19, v11
	v_mad_u64_u32 v[18:19], s[18:19], v10, s24, 0
	v_mul_hi_u32 v0, v10, s4
	v_lshl_add_u64 v[18:19], v[0:1], 0, v[18:19]
	v_mad_u64_u32 v[42:43], s[18:19], v8, s4, 0
	v_add_co_u32_e32 v0, vcc, v18, v42
	v_mad_u64_u32 v[22:23], s[18:19], v8, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v19, v43, vcc
	v_mov_b32_e32 v19, s5
	s_nop 0
	v_addc_co_u32_e32 v23, vcc, 0, v23, vcc
	v_lshl_add_u64 v[18:19], v[18:19], 0, v[22:23]
	v_mul_lo_u32 v0, s7, v18
	v_mul_lo_u32 v19, s6, v19
	v_mad_u64_u32 v[22:23], s[18:19], s6, v18, 0
	v_add3_u32 v0, v23, v19, v0
	v_sub_u32_e32 v19, v8, v0
	v_sub_co_u32_e32 v10, vcc, v10, v22
	s_nop 1
	v_subb_co_u32_e64 v19, s[18:19], v19, v4, vcc
	v_subrev_co_u32_e64 v21, s[18:19], s6, v10
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v19, s[18:19], 0, v19, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v19
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v22, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v21
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v10
	v_cndmask_b32_e64 v21, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v19
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	v_cndmask_b32_e64 v19, v22, v21, s[18:19]
	v_add_u32_e32 v21, 2, v18
	v_add_u32_e32 v22, 1, v18
	v_cmp_ne_u32_e64 s[18:19], 0, v19
	v_cndmask_b32_e32 v0, v8, v10, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v19, v22, v21, s[18:19]
	v_xor_b32_e32 v8, s64, v11
	v_cndmask_b32_e32 v0, v18, v19, vcc
	v_mov_b32_e32 v10, v7
	v_mov_b32_e32 v11, v7
	v_xor_b32_e32 v0, v0, v8
	v_lshl_add_u64 v[10:11], v[6:7], 0, v[10:11]
	v_sub_u32_e32 v0, v0, v8
	v_xor_b32_e32 v21, v10, v7
	v_xad_u32 v8, v0, v3, v30
	v_xor_b32_e32 v6, v11, v7
	v_mad_u64_u32 v[10:11], s[18:19], v21, s24, 0
	v_mul_hi_u32 v0, v21, s4
	v_lshl_add_u64 v[10:11], v[0:1], 0, v[10:11]
	v_mad_u64_u32 v[22:23], s[18:19], v6, s4, 0
	v_add_co_u32_e32 v0, vcc, v10, v22
	v_mad_u64_u32 v[18:19], s[18:19], v6, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v23, vcc
	v_mov_b32_e32 v11, s5
	s_nop 0
	v_addc_co_u32_e32 v19, vcc, 0, v19, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[18:19]
	v_mul_lo_u32 v0, s7, v10
	v_mul_lo_u32 v18, s6, v11
	v_mad_u64_u32 v[10:11], s[18:19], s6, v10, 0
	v_add3_u32 v0, v11, v18, v0
	v_sub_u32_e32 v11, v6, v0
	v_sub_co_u32_e32 v10, vcc, v21, v10
	s_nop 1
	v_subb_co_u32_e64 v11, s[18:19], v11, v4, vcc
	v_subrev_co_u32_e64 v18, s[18:19], s6, v10
	v_subb_co_u32_e32 v0, vcc, v6, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v19, s[22:23], 0, v11, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v19
	v_subb_co_u32_e64 v11, s[18:19], v11, v4, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v21, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v18
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v22, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v19
	v_cndmask_b32_e64 v6, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v10
	v_cndmask_b32_e64 v21, v21, v22, s[22:23]
	v_subrev_co_u32_e64 v22, s[18:19], s6, v18
	s_nop 1
	v_subbrev_co_u32_e64 v11, s[18:19], 0, v11, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v21
	s_nop 1
	v_cndmask_b32_e64 v11, v19, v11, s[18:19]
	v_cndmask_b32_e64 v19, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	s_nop 1
	v_cndmask_b32_e32 v6, v6, v19, vcc
	v_cmp_ne_u32_e32 vcc, 0, v6
	v_cndmask_b32_e64 v6, v18, v22, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v6, v10, v6, vcc
	v_cndmask_b32_e32 v0, v0, v11, vcc
	v_xor_b32_e32 v6, v6, v7
	v_xor_b32_e32 v0, v0, v7
	v_sub_co_u32_e32 v6, vcc, v6, v7
	v_or_b32_e32 v10, 10, v2
	s_nop 0
	v_subb_co_u32_e32 v0, vcc, v0, v7, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_mul_lo_u32 v7, v8, s33
	v_add3_u32 v6, v0, v6, v7
	v_sub_u32_e32 v0, -11, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v7, 31, v0
	v_cndmask_b32_e64 v19, v11, v7, s[10:11]
	v_cndmask_b32_e64 v18, v10, v0, s[10:11]
	v_mov_b32_e32 v22, v19
	v_mov_b32_e32 v23, v19
	v_lshl_add_u64 v[22:23], v[18:19], 0, v[22:23]
	v_xor_b32_e32 v8, v22, v19
	v_xor_b32_e32 v7, v23, v19
	v_mad_u64_u32 v[22:23], s[18:19], v8, s24, 0
	v_mul_hi_u32 v0, v8, s4
	v_lshl_add_u64 v[22:23], v[0:1], 0, v[22:23]
	v_mad_u64_u32 v[46:47], s[18:19], v7, s4, 0
	v_add_co_u32_e32 v0, vcc, v22, v46
	v_mad_u64_u32 v[42:43], s[18:19], v7, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v22, vcc, v23, v47, vcc
	v_mov_b32_e32 v23, s5
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[42:43]
	v_mul_lo_u32 v0, s7, v22
	v_mul_lo_u32 v18, s6, v23
	v_mad_u64_u32 v[42:43], s[18:19], s6, v22, 0
	v_add3_u32 v0, v43, v18, v0
	v_sub_u32_e32 v18, v7, v0
	v_sub_co_u32_e32 v8, vcc, v8, v42
	s_nop 1
	v_subb_co_u32_e64 v18, s[18:19], v18, v4, vcc
	v_subrev_co_u32_e64 v21, s[18:19], s6, v8
	v_subb_co_u32_e32 v0, vcc, v7, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v18, s[18:19], 0, v18, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v18
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v23, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v21
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v8
	v_cndmask_b32_e64 v21, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v18
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	v_cndmask_b32_e64 v18, v23, v21, s[18:19]
	v_add_u32_e32 v21, 2, v22
	v_add_u32_e32 v23, 1, v22
	v_cmp_ne_u32_e64 s[18:19], 0, v18
	v_cndmask_b32_e32 v0, v7, v8, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v18, v23, v21, s[18:19]
	v_xor_b32_e32 v7, s64, v19
	v_cndmask_b32_e32 v0, v22, v18, vcc
	v_mov_b32_e32 v18, v11
	v_mov_b32_e32 v19, v11
	v_xor_b32_e32 v0, v0, v7
	v_lshl_add_u64 v[18:19], v[10:11], 0, v[18:19]
	v_sub_u32_e32 v0, v0, v7
	v_xor_b32_e32 v10, v18, v11
	v_xad_u32 v7, v0, v3, v30
	v_xor_b32_e32 v8, v19, v11
	v_mad_u64_u32 v[18:19], s[18:19], v10, s24, 0
	v_mul_hi_u32 v0, v10, s4
	v_lshl_add_u64 v[18:19], v[0:1], 0, v[18:19]
	v_mad_u64_u32 v[42:43], s[18:19], v8, s4, 0
	v_add_co_u32_e32 v0, vcc, v18, v42
	v_mad_u64_u32 v[22:23], s[18:19], v8, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v19, v43, vcc
	v_mov_b32_e32 v19, s5
	s_nop 0
	v_addc_co_u32_e32 v23, vcc, 0, v23, vcc
	v_lshl_add_u64 v[18:19], v[18:19], 0, v[22:23]
	v_mul_lo_u32 v0, s7, v18
	v_mul_lo_u32 v21, s6, v19
	v_mad_u64_u32 v[18:19], s[18:19], s6, v18, 0
	v_add3_u32 v0, v19, v21, v0
	v_sub_u32_e32 v19, v8, v0
	v_sub_co_u32_e32 v10, vcc, v10, v18
	v_mul_lo_u32 v7, v7, s33
	s_nop 0
	v_subb_co_u32_e64 v18, s[18:19], v19, v4, vcc
	v_subrev_co_u32_e64 v19, s[18:19], s6, v10
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v21, s[22:23], 0, v18, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v21
	v_subb_co_u32_e64 v18, s[18:19], v18, v4, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v22, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v19
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v23, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v21
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v10
	v_cndmask_b32_e64 v22, v22, v23, s[22:23]
	v_subrev_co_u32_e64 v23, s[18:19], s6, v19
	s_nop 1
	v_subbrev_co_u32_e64 v18, s[18:19], 0, v18, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v22
	s_nop 1
	v_cndmask_b32_e64 v18, v21, v18, s[18:19]
	v_cndmask_b32_e64 v21, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	s_nop 1
	v_cndmask_b32_e32 v8, v8, v21, vcc
	v_cmp_ne_u32_e32 vcc, 0, v8
	v_cndmask_b32_e64 v8, v19, v23, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v8, v10, v8, vcc
	v_cndmask_b32_e32 v0, v0, v18, vcc
	v_xor_b32_e32 v8, v8, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v8, vcc, v8, v11
	v_or_b32_e32 v10, 11, v2
	s_nop 0
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v7, v0, v8, v7
	v_sub_u32_e32 v0, -12, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v8, 31, v0
	v_cndmask_b32_e64 v19, v11, v8, s[10:11]
	v_cndmask_b32_e64 v18, v10, v0, s[10:11]
	v_mov_b32_e32 v22, v19
	v_mov_b32_e32 v23, v19
	v_lshl_add_u64 v[22:23], v[18:19], 0, v[22:23]
	v_xor_b32_e32 v18, v22, v19
	v_xor_b32_e32 v8, v23, v19
	v_mad_u64_u32 v[22:23], s[18:19], v18, s24, 0
	v_mul_hi_u32 v0, v18, s4
	v_lshl_add_u64 v[22:23], v[0:1], 0, v[22:23]
	v_mad_u64_u32 v[46:47], s[18:19], v8, s4, 0
	v_add_co_u32_e32 v0, vcc, v22, v46
	v_mad_u64_u32 v[42:43], s[18:19], v8, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v22, vcc, v23, v47, vcc
	v_mov_b32_e32 v23, s5
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[42:43]
	v_mul_lo_u32 v0, s7, v22
	v_mul_lo_u32 v21, s6, v23
	v_mad_u64_u32 v[42:43], s[18:19], s6, v22, 0
	v_add3_u32 v0, v43, v21, v0
	v_sub_u32_e32 v21, v8, v0
	v_sub_co_u32_e32 v18, vcc, v18, v42
	s_nop 1
	v_subb_co_u32_e64 v21, s[18:19], v21, v4, vcc
	v_subrev_co_u32_e64 v23, s[18:19], s6, v18
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v21, s[18:19], 0, v21, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v21
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v31, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v23
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v18
	v_cndmask_b32_e64 v23, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v21
	v_cndmask_b32_e64 v18, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	v_cndmask_b32_e64 v21, v31, v23, s[18:19]
	v_add_u32_e32 v23, 2, v22
	v_add_u32_e32 v31, 1, v22
	v_cmp_ne_u32_e64 s[18:19], 0, v21
	v_cndmask_b32_e32 v0, v8, v18, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v21, v31, v23, s[18:19]
	v_xor_b32_e32 v8, s64, v19
	v_cndmask_b32_e32 v0, v22, v21, vcc
	v_mov_b32_e32 v18, v11
	v_mov_b32_e32 v19, v11
	v_xor_b32_e32 v0, v0, v8
	v_lshl_add_u64 v[18:19], v[10:11], 0, v[18:19]
	v_sub_u32_e32 v0, v0, v8
	v_xor_b32_e32 v21, v18, v11
	v_xad_u32 v8, v0, v3, v30
	v_xor_b32_e32 v10, v19, v11
	v_mad_u64_u32 v[18:19], s[18:19], v21, s24, 0
	v_mul_hi_u32 v0, v21, s4
	v_lshl_add_u64 v[18:19], v[0:1], 0, v[18:19]
	v_mad_u64_u32 v[42:43], s[18:19], v10, s4, 0
	v_add_co_u32_e32 v0, vcc, v18, v42
	v_mad_u64_u32 v[22:23], s[18:19], v10, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v19, v43, vcc
	v_mov_b32_e32 v19, s5
	s_nop 0
	v_addc_co_u32_e32 v23, vcc, 0, v23, vcc
	v_lshl_add_u64 v[18:19], v[18:19], 0, v[22:23]
	v_mul_lo_u32 v0, s7, v18
	v_mul_lo_u32 v22, s6, v19
	v_mad_u64_u32 v[18:19], s[18:19], s6, v18, 0
	v_add3_u32 v0, v19, v22, v0
	v_sub_u32_e32 v19, v10, v0
	v_sub_co_u32_e32 v18, vcc, v21, v18
	v_mul_lo_u32 v8, v8, s33
	s_nop 0
	v_subb_co_u32_e64 v19, s[18:19], v19, v4, vcc
	v_subrev_co_u32_e64 v21, s[18:19], s6, v18
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v22, s[22:23], 0, v19, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v22
	v_subb_co_u32_e64 v19, s[18:19], v19, v4, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v23, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v21
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v31, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v22
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v18
	v_cndmask_b32_e64 v23, v23, v31, s[22:23]
	v_subrev_co_u32_e64 v31, s[18:19], s6, v21
	s_nop 1
	v_subbrev_co_u32_e64 v19, s[18:19], 0, v19, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v23
	s_nop 1
	v_cndmask_b32_e64 v19, v22, v19, s[18:19]
	v_cndmask_b32_e64 v22, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v22, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v21, v31, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v10, v18, v10, vcc
	v_cndmask_b32_e32 v0, v0, v19, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v8, v0, v10, v8
	v_or_b32_e32 v10, 12, v2
	v_sub_u32_e32 v0, -13, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v18, 31, v0
	v_cndmask_b32_e64 v19, v11, v18, s[10:11]
	v_cndmask_b32_e64 v18, v10, v0, s[10:11]
	v_mov_b32_e32 v22, v19
	v_mov_b32_e32 v23, v19
	v_lshl_add_u64 v[22:23], v[18:19], 0, v[22:23]
	v_xor_b32_e32 v21, v22, v19
	v_xor_b32_e32 v18, v23, v19
	v_mad_u64_u32 v[22:23], s[18:19], v21, s24, 0
	v_mul_hi_u32 v0, v21, s4
	v_lshl_add_u64 v[22:23], v[0:1], 0, v[22:23]
	v_mad_u64_u32 v[46:47], s[18:19], v18, s4, 0
	v_add_co_u32_e32 v0, vcc, v22, v46
	v_mad_u64_u32 v[42:43], s[18:19], v18, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v22, vcc, v23, v47, vcc
	v_mov_b32_e32 v23, s5
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[42:43]
	v_mul_lo_u32 v0, s7, v22
	v_mul_lo_u32 v23, s6, v23
	v_mad_u64_u32 v[42:43], s[18:19], s6, v22, 0
	v_add3_u32 v0, v43, v23, v0
	v_sub_u32_e32 v23, v18, v0
	v_sub_co_u32_e32 v21, vcc, v21, v42
	s_nop 1
	v_subb_co_u32_e64 v23, s[18:19], v23, v4, vcc
	v_subrev_co_u32_e64 v31, s[18:19], s6, v21
	v_subb_co_u32_e32 v0, vcc, v18, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v23, s[18:19], 0, v23, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v23
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v33, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v31
	v_cndmask_b32_e64 v18, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v21
	v_cndmask_b32_e64 v31, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v23
	v_cndmask_b32_e64 v21, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	v_cndmask_b32_e64 v23, v33, v31, s[18:19]
	v_add_u32_e32 v31, 2, v22
	v_add_u32_e32 v33, 1, v22
	v_cmp_ne_u32_e64 s[18:19], 0, v23
	v_cndmask_b32_e32 v0, v18, v21, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v23, v33, v31, s[18:19]
	v_xor_b32_e32 v18, s64, v19
	v_cndmask_b32_e32 v0, v22, v23, vcc
	v_xor_b32_e32 v0, v0, v18
	v_sub_u32_e32 v0, v0, v18
	v_mov_b32_e32 v18, v11
	v_mov_b32_e32 v19, v11
	v_lshl_add_u64 v[18:19], v[10:11], 0, v[18:19]
	v_xor_b32_e32 v31, v18, v11
	v_xad_u32 v21, v0, v3, v30
	v_xor_b32_e32 v10, v19, v11
	v_mad_u64_u32 v[18:19], s[18:19], v31, s24, 0
	v_mul_hi_u32 v0, v31, s4
	v_lshl_add_u64 v[18:19], v[0:1], 0, v[18:19]
	v_mad_u64_u32 v[42:43], s[18:19], v10, s4, 0
	v_add_co_u32_e32 v0, vcc, v18, v42
	v_mad_u64_u32 v[22:23], s[18:19], v10, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v19, v43, vcc
	v_mov_b32_e32 v19, s5
	s_nop 0
	v_addc_co_u32_e32 v23, vcc, 0, v23, vcc
	v_lshl_add_u64 v[18:19], v[18:19], 0, v[22:23]
	v_mul_lo_u32 v0, s7, v18
	v_mul_lo_u32 v22, s6, v19
	v_mad_u64_u32 v[18:19], s[18:19], s6, v18, 0
	v_add3_u32 v0, v19, v22, v0
	v_sub_u32_e32 v19, v10, v0
	v_sub_co_u32_e32 v18, vcc, v31, v18
	s_nop 1
	v_subb_co_u32_e64 v19, s[18:19], v19, v4, vcc
	v_subrev_co_u32_e64 v22, s[18:19], s6, v18
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v23, s[22:23], 0, v19, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v23
	v_subb_co_u32_e64 v19, s[18:19], v19, v4, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v31, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v22
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v33, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v23
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v18
	v_cndmask_b32_e64 v31, v31, v33, s[22:23]
	v_subrev_co_u32_e64 v33, s[18:19], s6, v22
	s_nop 1
	v_subbrev_co_u32_e64 v19, s[18:19], 0, v19, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v31
	s_nop 1
	v_cndmask_b32_e64 v19, v23, v19, s[18:19]
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v23, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v22, v33, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v10, v18, v10, vcc
	v_cndmask_b32_e32 v0, v0, v19, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_mul_lo_u32 v11, v21, s33
	v_add3_u32 v31, v0, v10, v11
	v_or_b32_e32 v10, 13, v2
	v_sub_u32_e32 v0, -14, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v18, 31, v0
	v_cndmask_b32_e64 v19, v11, v18, s[10:11]
	v_cndmask_b32_e64 v18, v10, v0, s[10:11]
	v_mov_b32_e32 v22, v19
	v_mov_b32_e32 v23, v19
	v_lshl_add_u64 v[22:23], v[18:19], 0, v[22:23]
	v_xor_b32_e32 v21, v22, v19
	v_xor_b32_e32 v18, v23, v19
	v_mad_u64_u32 v[22:23], s[18:19], v21, s24, 0
	v_mul_hi_u32 v0, v21, s4
	v_lshl_add_u64 v[22:23], v[0:1], 0, v[22:23]
	v_mad_u64_u32 v[46:47], s[18:19], v18, s4, 0
	v_add_co_u32_e32 v0, vcc, v22, v46
	v_mad_u64_u32 v[42:43], s[18:19], v18, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v22, vcc, v23, v47, vcc
	v_mov_b32_e32 v23, s5
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[42:43]
	v_mul_lo_u32 v0, s7, v22
	v_mul_lo_u32 v23, s6, v23
	v_mad_u64_u32 v[42:43], s[18:19], s6, v22, 0
	v_add3_u32 v0, v43, v23, v0
	v_sub_u32_e32 v23, v18, v0
	v_sub_co_u32_e32 v21, vcc, v21, v42
	s_nop 1
	v_subb_co_u32_e64 v23, s[18:19], v23, v4, vcc
	v_subrev_co_u32_e64 v33, s[18:19], s6, v21
	v_subb_co_u32_e32 v0, vcc, v18, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v23, s[18:19], 0, v23, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v23
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v35, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v33
	v_cndmask_b32_e64 v18, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v21
	v_cndmask_b32_e64 v33, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v23
	v_cndmask_b32_e64 v21, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	v_cndmask_b32_e64 v23, v35, v33, s[18:19]
	v_add_u32_e32 v33, 2, v22
	v_add_u32_e32 v35, 1, v22
	v_cmp_ne_u32_e64 s[18:19], 0, v23
	v_cndmask_b32_e32 v0, v18, v21, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v23, v35, v33, s[18:19]
	v_xor_b32_e32 v18, s64, v19
	v_cndmask_b32_e32 v0, v22, v23, vcc
	v_xor_b32_e32 v0, v0, v18
	v_sub_u32_e32 v0, v0, v18
	v_mov_b32_e32 v18, v11
	v_mov_b32_e32 v19, v11
	v_lshl_add_u64 v[18:19], v[10:11], 0, v[18:19]
	v_xor_b32_e32 v33, v18, v11
	v_xad_u32 v21, v0, v3, v30
	v_xor_b32_e32 v10, v19, v11
	v_mad_u64_u32 v[18:19], s[18:19], v33, s24, 0
	v_mul_hi_u32 v0, v33, s4
	v_lshl_add_u64 v[18:19], v[0:1], 0, v[18:19]
	v_mad_u64_u32 v[42:43], s[18:19], v10, s4, 0
	v_add_co_u32_e32 v0, vcc, v18, v42
	v_mad_u64_u32 v[22:23], s[18:19], v10, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v19, v43, vcc
	v_mov_b32_e32 v19, s5
	s_nop 0
	v_addc_co_u32_e32 v23, vcc, 0, v23, vcc
	v_lshl_add_u64 v[18:19], v[18:19], 0, v[22:23]
	v_mul_lo_u32 v0, s7, v18
	v_mul_lo_u32 v22, s6, v19
	v_mad_u64_u32 v[18:19], s[18:19], s6, v18, 0
	v_add3_u32 v0, v19, v22, v0
	v_sub_u32_e32 v19, v10, v0
	v_sub_co_u32_e32 v18, vcc, v33, v18
	s_nop 1
	v_subb_co_u32_e64 v19, s[18:19], v19, v4, vcc
	v_subrev_co_u32_e64 v22, s[18:19], s6, v18
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v23, s[22:23], 0, v19, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v23
	v_subb_co_u32_e64 v19, s[18:19], v19, v4, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v33, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v22
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v35, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v23
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v18
	v_cndmask_b32_e64 v33, v33, v35, s[22:23]
	v_subrev_co_u32_e64 v35, s[18:19], s6, v22
	s_nop 1
	v_subbrev_co_u32_e64 v19, s[18:19], 0, v19, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v33
	s_nop 1
	v_cndmask_b32_e64 v19, v23, v19, s[18:19]
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v23, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v22, v35, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v10, v18, v10, vcc
	v_cndmask_b32_e32 v0, v0, v19, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_mul_lo_u32 v11, v21, s33
	v_add3_u32 v33, v0, v10, v11
	v_or_b32_e32 v10, 14, v2
	v_sub_u32_e32 v0, -15, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v18, 31, v0
	v_cndmask_b32_e64 v19, v11, v18, s[10:11]
	v_cndmask_b32_e64 v18, v10, v0, s[10:11]
	v_mov_b32_e32 v22, v19
	v_mov_b32_e32 v23, v19
	v_lshl_add_u64 v[22:23], v[18:19], 0, v[22:23]
	v_xor_b32_e32 v21, v22, v19
	v_xor_b32_e32 v18, v23, v19
	v_mad_u64_u32 v[22:23], s[18:19], v21, s24, 0
	v_mul_hi_u32 v0, v21, s4
	v_lshl_add_u64 v[22:23], v[0:1], 0, v[22:23]
	v_mad_u64_u32 v[46:47], s[18:19], v18, s4, 0
	v_add_co_u32_e32 v0, vcc, v22, v46
	v_mad_u64_u32 v[42:43], s[18:19], v18, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v22, vcc, v23, v47, vcc
	v_mov_b32_e32 v23, s5
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[42:43]
	v_mul_lo_u32 v0, s7, v22
	v_mul_lo_u32 v23, s6, v23
	v_mad_u64_u32 v[42:43], s[18:19], s6, v22, 0
	v_add3_u32 v0, v43, v23, v0
	v_sub_u32_e32 v23, v18, v0
	v_sub_co_u32_e32 v21, vcc, v21, v42
	s_nop 1
	v_subb_co_u32_e64 v23, s[18:19], v23, v4, vcc
	v_subrev_co_u32_e64 v35, s[18:19], s6, v21
	v_subb_co_u32_e32 v0, vcc, v18, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v23, s[18:19], 0, v23, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v23
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v37, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v35
	v_cndmask_b32_e64 v18, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v21
	v_cndmask_b32_e64 v35, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v23
	v_cndmask_b32_e64 v21, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	v_cndmask_b32_e64 v23, v37, v35, s[18:19]
	v_add_u32_e32 v35, 2, v22
	v_add_u32_e32 v37, 1, v22
	v_cmp_ne_u32_e64 s[18:19], 0, v23
	v_cndmask_b32_e32 v0, v18, v21, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v23, v37, v35, s[18:19]
	v_xor_b32_e32 v18, s64, v19
	v_cndmask_b32_e32 v0, v22, v23, vcc
	v_xor_b32_e32 v0, v0, v18
	v_sub_u32_e32 v0, v0, v18
	v_mov_b32_e32 v18, v11
	v_mov_b32_e32 v19, v11
	v_lshl_add_u64 v[18:19], v[10:11], 0, v[18:19]
	v_xor_b32_e32 v35, v18, v11
	v_xad_u32 v21, v0, v3, v30
	v_xor_b32_e32 v10, v19, v11
	v_mad_u64_u32 v[18:19], s[18:19], v35, s24, 0
	v_mul_hi_u32 v0, v35, s4
	v_lshl_add_u64 v[18:19], v[0:1], 0, v[18:19]
	v_mad_u64_u32 v[42:43], s[18:19], v10, s4, 0
	v_add_co_u32_e32 v0, vcc, v18, v42
	v_mad_u64_u32 v[22:23], s[18:19], v10, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v19, v43, vcc
	v_mov_b32_e32 v19, s5
	s_nop 0
	v_addc_co_u32_e32 v23, vcc, 0, v23, vcc
	v_lshl_add_u64 v[18:19], v[18:19], 0, v[22:23]
	v_mul_lo_u32 v0, s7, v18
	v_mul_lo_u32 v22, s6, v19
	v_mad_u64_u32 v[18:19], s[18:19], s6, v18, 0
	v_add3_u32 v0, v19, v22, v0
	v_sub_u32_e32 v19, v10, v0
	v_sub_co_u32_e32 v18, vcc, v35, v18
	s_nop 1
	v_subb_co_u32_e64 v19, s[18:19], v19, v4, vcc
	v_subrev_co_u32_e64 v22, s[18:19], s6, v18
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v23, s[22:23], 0, v19, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v23
	v_subb_co_u32_e64 v19, s[18:19], v19, v4, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v35, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v22
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v37, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v23
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v18
	v_cndmask_b32_e64 v35, v35, v37, s[22:23]
	v_subrev_co_u32_e64 v37, s[18:19], s6, v22
	s_nop 1
	v_subbrev_co_u32_e64 v19, s[18:19], 0, v19, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v35
	s_nop 1
	v_cndmask_b32_e64 v19, v23, v19, s[18:19]
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v23, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v22, v37, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v10, v18, v10, vcc
	v_cndmask_b32_e32 v0, v0, v19, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_mul_lo_u32 v11, v21, s33
	v_add3_u32 v37, v0, v10, v11
	v_or_b32_e32 v10, 15, v2
	v_sub_u32_e32 v0, -16, v2
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v18, 31, v0
	v_cndmask_b32_e64 v19, v11, v18, s[10:11]
	v_cndmask_b32_e64 v18, v10, v0, s[10:11]
	v_mov_b32_e32 v22, v19
	v_mov_b32_e32 v23, v19
	v_lshl_add_u64 v[22:23], v[18:19], 0, v[22:23]
	v_xor_b32_e32 v21, v22, v19
	v_xor_b32_e32 v18, v23, v19
	v_mad_u64_u32 v[22:23], s[10:11], v21, s24, 0
	v_mul_hi_u32 v0, v21, s4
	v_lshl_add_u64 v[22:23], v[0:1], 0, v[22:23]
	v_mad_u64_u32 v[46:47], s[10:11], v18, s4, 0
	v_add_co_u32_e32 v0, vcc, v22, v46
	v_mad_u64_u32 v[42:43], s[10:11], v18, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v22, vcc, v23, v47, vcc
	v_mov_b32_e32 v23, s5
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[42:43]
	v_mul_lo_u32 v0, s7, v22
	v_mul_lo_u32 v23, s6, v23
	v_mad_u64_u32 v[42:43], s[10:11], s6, v22, 0
	v_add3_u32 v0, v43, v23, v0
	v_sub_u32_e32 v23, v18, v0
	v_sub_co_u32_e32 v21, vcc, v21, v42
	s_nop 1
	v_subb_co_u32_e64 v23, s[10:11], v23, v4, vcc
	v_subrev_co_u32_e64 v35, s[10:11], s6, v21
	v_subb_co_u32_e32 v0, vcc, v18, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v23, s[10:11], 0, v23, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s7, v23
	v_cmp_le_u32_e32 vcc, s7, v0
	s_nop 0
	v_cndmask_b32_e64 v39, 0, -1, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s6, v35
	v_cndmask_b32_e64 v18, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v21
	v_cndmask_b32_e64 v35, 0, -1, s[10:11]
	v_cmp_eq_u32_e64 s[10:11], s7, v23
	v_cndmask_b32_e64 v21, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	v_cndmask_b32_e64 v23, v39, v35, s[10:11]
	v_add_u32_e32 v35, 2, v22
	v_add_u32_e32 v39, 1, v22
	v_cmp_ne_u32_e64 s[10:11], 0, v23
	v_cndmask_b32_e32 v0, v18, v21, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v23, v39, v35, s[10:11]
	v_xor_b32_e32 v18, s64, v19
	v_cndmask_b32_e32 v0, v22, v23, vcc
	v_xor_b32_e32 v0, v0, v18
	v_sub_u32_e32 v0, v0, v18
	v_mov_b32_e32 v18, v11
	v_mov_b32_e32 v19, v11
	v_lshl_add_u64 v[18:19], v[10:11], 0, v[18:19]
	v_xor_b32_e32 v21, v18, v11
	v_xad_u32 v3, v0, v3, v30
	v_xor_b32_e32 v10, v19, v11
	v_mad_u64_u32 v[18:19], s[10:11], v21, s24, 0
	v_mul_hi_u32 v0, v21, s4
	v_lshl_add_u64 v[18:19], v[0:1], 0, v[18:19]
	v_mad_u64_u32 v[42:43], s[10:11], v10, s4, 0
	v_add_co_u32_e32 v0, vcc, v18, v42
	v_mad_u64_u32 v[22:23], s[10:11], v10, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v19, v43, vcc
	v_mov_b32_e32 v19, s5
	s_nop 0
	v_addc_co_u32_e32 v23, vcc, 0, v23, vcc
	v_lshl_add_u64 v[18:19], v[18:19], 0, v[22:23]
	v_mul_lo_u32 v0, s7, v18
	v_mul_lo_u32 v22, s6, v19
	v_mad_u64_u32 v[18:19], s[10:11], s6, v18, 0
	v_add3_u32 v0, v19, v22, v0
	v_sub_u32_e32 v19, v10, v0
	v_sub_co_u32_e32 v18, vcc, v21, v18
	v_mul_lo_u32 v3, v3, s33
	s_nop 0
	v_subb_co_u32_e64 v19, s[10:11], v19, v4, vcc
	v_subrev_co_u32_e64 v21, s[10:11], s6, v18
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v22, s[18:19], 0, v19, s[10:11]
	v_cmp_le_u32_e64 s[18:19], s7, v22
	v_subb_co_u32_e64 v4, s[10:11], v19, v4, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v23, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v21
	v_subrev_co_u32_e64 v19, s[10:11], s6, v21
	s_nop 0
	v_cndmask_b32_e64 v35, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v22
	v_subbrev_co_u32_e64 v4, s[10:11], 0, v4, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v23, v23, v35, s[18:19]
	v_cmp_le_u32_e32 vcc, s7, v0
	v_cmp_ne_u32_e64 s[10:11], 0, v23
	s_nop 0
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v18
	v_cndmask_b32_e64 v4, v22, v4, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v22, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v0
	s_cselect_b64 s[6:7], -1, 0
	s_nop 0
	v_cndmask_b32_e32 v10, v10, v22, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	s_nop 1
	v_cndmask_b32_e32 v0, v0, v4, vcc
	v_cndmask_b32_e64 v4, v21, v19, s[10:11]
	v_cndmask_b32_e32 v4, v18, v4, vcc
	v_xor_b32_e32 v4, v4, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v4, vcc, v4, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v0, v0, v4, v3
	buffer_load_ubyte v18, v5, s[28:31], 0 offen
	buffer_load_ubyte v22, v6, s[28:31], 0 offen
	buffer_load_ubyte v21, v7, s[28:31], 0 offen
	buffer_load_ubyte v23, v8, s[28:31], 0 offen
	buffer_load_ubyte v19, v31, s[28:31], 0 offen
	buffer_load_ubyte v35, v33, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v33, v37, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v37, v0, s[28:31], 0 offen
	v_mov_b32_e32 v31, s13
	v_add_u32_e32 v0, 0x400, v2
	s_and_saveexec_b64 s[10:11], s[6:7]
	s_xor_b64 s[6:7], exec, s[10:11]
	s_cbranch_execz .LBB0_16
	s_add_u32 s10, s42, s64
	s_addc_u32 s11, s43, s64
	s_xor_b64 s[12:13], s[10:11], s[64:65]
	v_cvt_f32_u32_e32 v2, s12
	v_cvt_f32_u32_e32 v3, s13
	s_sub_u32 s4, 0, s12
	s_subb_u32 s10, 0, s13
	v_mov_b32_e32 v7, v1
	v_fmac_f32_e32 v2, 0x4f800000, v3
	v_rcp_f32_e32 v2, v2
	s_nop 0
	v_mul_f32_e32 v2, 0x5f7ffffc, v2
	v_mul_f32_e32 v3, 0x2f800000, v2
	v_trunc_f32_e32 v3, v3
	v_fmac_f32_e32 v2, 0xcf800000, v3
	v_cvt_u32_f32_e32 v3, v3
	v_cvt_u32_f32_e32 v2, v2
	v_readfirstlane_b32 s11, v3
	v_readfirstlane_b32 s18, v2
	s_mul_i32 s19, s4, s11
	s_mul_hi_u32 s23, s4, s18
	s_mul_i32 s22, s10, s18
	s_add_i32 s19, s23, s19
	s_mul_i32 s24, s4, s18
	s_add_i32 s19, s19, s22
	s_mul_i32 s23, s18, s19
	s_mul_hi_u32 s25, s18, s24
	s_mul_hi_u32 s22, s18, s19
	s_add_u32 s23, s25, s23
	s_addc_u32 s22, 0, s22
	s_mul_hi_u32 s26, s11, s24
	s_mul_i32 s24, s11, s24
	s_add_u32 s23, s23, s24
	s_mul_hi_u32 s25, s11, s19
	s_addc_u32 s22, s22, s26
	s_addc_u32 s23, s25, 0
	s_mul_i32 s19, s11, s19
	s_add_u32 s19, s22, s19
	s_addc_u32 s22, 0, s23
	s_add_u32 s18, s18, s19
	s_addc_u32 s11, s11, s22
	s_mul_i32 s19, s4, s11
	s_mul_hi_u32 s22, s4, s18
	s_add_i32 s19, s22, s19
	s_mul_i32 s10, s10, s18
	s_add_i32 s19, s19, s10
	s_mul_i32 s4, s4, s18
	s_mul_hi_u32 s22, s11, s4
	s_mul_i32 s23, s11, s4
	s_mul_i32 s25, s18, s19
	s_mul_hi_u32 s4, s18, s4
	s_mul_hi_u32 s24, s18, s19
	s_add_u32 s4, s4, s25
	s_addc_u32 s24, 0, s24
	s_add_u32 s4, s4, s23
	s_mul_hi_u32 s10, s11, s19
	s_addc_u32 s4, s24, s22
	s_addc_u32 s10, s10, 0
	s_mul_i32 s19, s11, s19
	s_add_u32 s4, s4, s19
	s_addc_u32 s10, 0, s10
	s_add_u32 s4, s18, s4
	s_addc_u32 s18, s11, s10
	v_lshl_add_u64 v[2:3], v[0:1], 0, 0
	v_mad_u64_u32 v[4:5], s[10:11], v2, s18, 0
	v_mul_hi_u32 v6, v2, s4
	v_lshl_add_u64 v[4:5], v[6:7], 0, v[4:5]
	v_mad_u64_u32 v[10:11], s[10:11], v3, s4, 0
	v_add_co_u32_e32 v1, vcc, v4, v10
	v_mad_u64_u32 v[6:7], s[10:11], v3, s18, 0
	s_nop 0
	v_addc_co_u32_e32 v4, vcc, v5, v11, vcc
	v_mov_b32_e32 v5, s5
	s_nop 0
	v_addc_co_u32_e32 v7, vcc, 0, v7, vcc
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[6:7]
	v_mul_lo_u32 v1, s13, v4
	v_mul_lo_u32 v8, s12, v5
	v_mad_u64_u32 v[6:7], s[4:5], s12, v4, 0
	v_add3_u32 v1, v7, v8, v1
	v_sub_u32_e32 v7, v3, v1
	v_mov_b32_e32 v8, s13
	v_sub_co_u32_e32 v2, vcc, v2, v6
	v_lshl_add_u64 v[10:11], v[4:5], 0, 1
	s_nop 0
	v_subb_co_u32_e64 v6, s[10:11], v7, v8, vcc
	v_subrev_co_u32_e64 v7, s[10:11], s12, v2
	v_subb_co_u32_e32 v1, vcc, v3, v1, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v6, s[10:11], 0, v6, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s13, v6
	v_cmp_le_u32_e32 vcc, s13, v1
	s_nop 0
	v_cndmask_b32_e64 v8, 0, -1, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s12, v7
	v_cndmask_b32_e64 v3, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v2
	v_cndmask_b32_e64 v7, 0, -1, s[10:11]
	v_cmp_eq_u32_e64 s[10:11], s13, v6
	v_cndmask_b32_e64 v2, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v1
	v_cndmask_b32_e64 v8, v8, v7, s[10:11]
	v_lshl_add_u64 v[6:7], v[4:5], 0, 2
	v_cndmask_b32_e32 v1, v3, v2, vcc
	v_cmp_ne_u32_e32 vcc, 0, v8
	s_nop 1
	v_cndmask_b32_e32 v2, v10, v6, vcc
	v_cmp_ne_u32_e32 vcc, 0, v1
	s_nop 1
	v_cndmask_b32_e32 v1, v4, v2, vcc
	v_xor_b32_e32 v1, s64, v1
	v_subrev_co_u32_e32 v2, vcc, s64, v1
.LBB0_16:
	s_andn2_saveexec_b64 s[4:5], s[6:7]
	s_cbranch_execz .LBB0_18
	v_cvt_f32_u32_e32 v1, s42
	s_sub_i32 s6, 0, s42
	v_rcp_iflag_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_mul_lo_u32 v2, s6, v1
	v_mul_hi_u32 v2, v1, v2
	v_add_u32_e32 v1, v1, v2
	v_mul_hi_u32 v1, v0, v1
	v_mul_lo_u32 v2, v1, s42
	v_sub_u32_e32 v2, v0, v2
	v_add_u32_e32 v3, 1, v1
	v_subrev_u32_e32 v4, s42, v2
	v_cmp_le_u32_e32 vcc, s42, v2
	s_nop 1
	v_cndmask_b32_e32 v2, v2, v4, vcc
	v_cndmask_b32_e32 v1, v1, v3, vcc
	v_add_u32_e32 v3, 1, v1
	v_cmp_le_u32_e32 vcc, s42, v2
	s_nop 1
	v_cndmask_b32_e32 v2, v1, v3, vcc
.LBB0_18:
	s_or_b64 exec, exec, s[4:5]
	v_add_u32_e32 v1, v30, v2
	v_mul_lo_u32 v2, v2, s42
	v_sub_u32_e32 v2, v0, v2
	v_mad_u64_u32 v[2:3], s[4:5], v1, s33, v[2:3]
	s_add_u32 s4, s42, s64
	v_lshrrev_b32_e32 v1, 5, v24
	s_mov_b32 s65, s64
	s_addc_u32 s5, s43, s64
	v_and_b32_e32 v3, 0x200, v27
	v_mul_i32_i24_e32 v64, 0xfffffe00, v1
	s_xor_b64 s[4:5], s[4:5], s[64:65]
	v_add3_u32 v4, v0, v3, v64
	v_cvt_f32_u32_e32 v0, s4
	v_cvt_f32_u32_e32 v1, s5
	s_sub_u32 s6, 0, s4
	s_subb_u32 s7, 0, s5
	v_or_b32_e32 v10, 1, v4
	v_fmac_f32_e32 v0, 0x4f800000, v1
	v_rcp_f32_e32 v0, v0
	v_sub_u32_e32 v5, -2, v4
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v6, 31, v5
	v_mul_f32_e32 v0, 0x5f7ffffc, v0
	v_mul_f32_e32 v1, 0x2f800000, v0
	v_trunc_f32_e32 v1, v1
	v_fmac_f32_e32 v0, 0xcf800000, v1
	v_cvt_u32_f32_e32 v0, v0
	v_cvt_u32_f32_e32 v1, v1
	v_cmp_gt_i32_e64 s[10:11], 0, v4
	s_mov_b32 s24, 0
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s19, v1
	s_mul_hi_u32 s18, s6, s12
	s_mul_i32 s22, s6, s19
	s_mul_i32 s13, s7, s12
	s_add_i32 s18, s18, s22
	s_add_i32 s18, s18, s13
	s_mul_i32 s23, s6, s12
	s_mul_i32 s22, s12, s18
	s_mul_hi_u32 s25, s12, s23
	s_mul_hi_u32 s13, s12, s18
	s_add_u32 s22, s25, s22
	s_addc_u32 s13, 0, s13
	s_mul_hi_u32 s26, s19, s23
	s_mul_i32 s23, s19, s23
	s_add_u32 s22, s22, s23
	s_mul_hi_u32 s25, s19, s18
	s_addc_u32 s13, s13, s26
	s_addc_u32 s22, s25, 0
	s_mul_i32 s18, s19, s18
	s_add_u32 s13, s13, s18
	s_addc_u32 s18, 0, s22
	s_add_u32 s12, s12, s13
	s_addc_u32 s13, s19, s18
	s_mul_i32 s18, s6, s13
	s_mul_hi_u32 s19, s6, s12
	s_add_i32 s18, s19, s18
	s_mul_i32 s7, s7, s12
	s_add_i32 s18, s18, s7
	s_mul_i32 s6, s6, s12
	s_mul_hi_u32 s19, s13, s6
	s_mul_i32 s22, s13, s6
	s_mul_i32 s25, s12, s18
	s_mul_hi_u32 s6, s12, s6
	s_mul_hi_u32 s23, s12, s18
	s_add_u32 s6, s6, s25
	s_addc_u32 s23, 0, s23
	s_add_u32 s6, s6, s22
	s_mul_hi_u32 s7, s13, s18
	s_addc_u32 s6, s23, s19
	s_addc_u32 s7, s7, 0
	s_mul_i32 s18, s13, s18
	v_cndmask_b32_e64 v7, v11, v6, s[10:11]
	s_add_u32 s6, s6, s18
	v_cndmask_b32_e64 v6, v10, v5, s[10:11]
	s_addc_u32 s7, 0, s7
	v_mov_b32_e32 v0, v7
	v_mov_b32_e32 v1, v7
	s_add_u32 s6, s12, s6
	v_lshl_add_u64 v[0:1], v[6:7], 0, v[0:1]
	s_addc_u32 s7, s13, s7
	v_xor_b32_e32 v8, v0, v7
	v_xor_b32_e32 v6, v1, v7
	v_mad_u64_u32 v[42:43], s[12:13], v8, s7, 0
	v_mul_hi_u32 v0, v8, s6
	v_mov_b32_e32 v1, 0
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[42:43]
	v_mad_u64_u32 v[48:49], s[12:13], v6, s6, 0
	v_add_co_u32_e32 v0, vcc, v42, v48
	v_mad_u64_u32 v[46:47], s[12:13], v6, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v49, vcc
	v_mov_b32_e32 v43, s24
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	v_mul_lo_u32 v0, s5, v42
	v_mul_lo_u32 v5, s4, v43
	v_mad_u64_u32 v[46:47], s[12:13], s4, v42, 0
	v_add3_u32 v0, v47, v5, v0
	v_sub_u32_e32 v39, v6, v0
	v_mov_b32_e32 v5, s5
	v_sub_co_u32_e32 v8, vcc, v8, v46
	s_nop 1
	v_subb_co_u32_e64 v39, s[18:19], v39, v5, vcc
	v_subrev_co_u32_e64 v43, s[18:19], s4, v8
	v_subb_co_u32_e32 v0, vcc, v6, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v39, s[18:19], 0, v39, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s5, v39
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v46, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s4, v43
	v_cndmask_b32_e64 v6, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v8
	v_cndmask_b32_e64 v43, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s5, v39
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	v_cndmask_b32_e64 v39, v46, v43, s[18:19]
	v_add_u32_e32 v43, 2, v42
	v_add_u32_e32 v46, 1, v42
	v_cmp_ne_u32_e64 s[18:19], 0, v39
	v_cndmask_b32_e32 v0, v6, v8, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v39, v46, v43, s[18:19]
	v_xor_b32_e32 v6, s64, v7
	v_cndmask_b32_e32 v0, v42, v39, vcc
	v_mov_b32_e32 v42, v11
	v_mov_b32_e32 v43, v11
	v_xor_b32_e32 v0, v0, v6
	v_lshl_add_u64 v[42:43], v[10:11], 0, v[42:43]
	v_sub_u32_e32 v0, v0, v6
	v_ashrrev_i32_e32 v6, 31, v4
	v_xor_b32_e32 v10, v42, v11
	v_xad_u32 v7, v0, v6, v30
	v_xor_b32_e32 v8, v43, v11
	v_mad_u64_u32 v[42:43], s[12:13], v10, s7, 0
	v_mul_hi_u32 v0, v10, s6
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[42:43]
	v_mad_u64_u32 v[48:49], s[12:13], v8, s6, 0
	v_add_co_u32_e32 v0, vcc, v42, v48
	v_mad_u64_u32 v[46:47], s[12:13], v8, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v49, vcc
	v_mov_b32_e32 v43, s24
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	v_mul_lo_u32 v0, s5, v42
	v_mul_lo_u32 v39, s4, v43
	v_mad_u64_u32 v[42:43], s[12:13], s4, v42, 0
	v_add3_u32 v0, v43, v39, v0
	v_sub_u32_e32 v39, v8, v0
	v_sub_co_u32_e32 v10, vcc, v10, v42
	v_mul_lo_u32 v7, v7, s33
	s_nop 0
	v_subb_co_u32_e64 v39, s[18:19], v39, v5, vcc
	v_subrev_co_u32_e64 v42, s[18:19], s4, v10
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[22:23], 0, v39, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s5, v43
	v_subb_co_u32_e64 v39, s[18:19], v39, v5, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v46, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s4, v42
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s5, v43
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v10
	v_cndmask_b32_e64 v46, v46, v47, s[22:23]
	v_subrev_co_u32_e64 v47, s[18:19], s4, v42
	s_nop 1
	v_subbrev_co_u32_e64 v39, s[18:19], 0, v39, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v46
	s_nop 1
	v_cndmask_b32_e64 v39, v43, v39, s[18:19]
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	s_nop 1
	v_cndmask_b32_e32 v8, v8, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v8
	v_cndmask_b32_e64 v8, v42, v47, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v8, v10, v8, vcc
	v_cndmask_b32_e32 v0, v0, v39, vcc
	v_xor_b32_e32 v8, v8, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v8, vcc, v8, v11
	v_or_b32_e32 v10, 2, v4
	s_nop 0
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v7, v0, v8, v7
	v_sub_u32_e32 v0, -3, v4
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v8, 31, v0
	v_cndmask_b32_e64 v43, v11, v8, s[10:11]
	v_cndmask_b32_e64 v42, v10, v0, s[10:11]
	v_mov_b32_e32 v46, v43
	v_mov_b32_e32 v47, v43
	v_lshl_add_u64 v[46:47], v[42:43], 0, v[46:47]
	v_xor_b32_e32 v39, v46, v43
	v_xor_b32_e32 v8, v47, v43
	v_mad_u64_u32 v[46:47], s[12:13], v39, s7, 0
	v_mul_hi_u32 v0, v39, s6
	v_lshl_add_u64 v[46:47], v[0:1], 0, v[46:47]
	v_mad_u64_u32 v[50:51], s[12:13], v8, s6, 0
	v_add_co_u32_e32 v0, vcc, v46, v50
	v_mad_u64_u32 v[48:49], s[12:13], v8, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v46, vcc, v47, v51, vcc
	v_mov_b32_e32 v47, s24
	s_nop 0
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_mul_lo_u32 v0, s5, v46
	v_mul_lo_u32 v42, s4, v47
	v_mad_u64_u32 v[48:49], s[12:13], s4, v46, 0
	v_add3_u32 v0, v49, v42, v0
	v_sub_u32_e32 v42, v8, v0
	v_sub_co_u32_e32 v39, vcc, v39, v48
	s_nop 1
	v_subb_co_u32_e64 v42, s[18:19], v42, v5, vcc
	v_subrev_co_u32_e64 v47, s[18:19], s4, v39
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v42, s[18:19], 0, v42, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s5, v42
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v48, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s4, v47
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v39
	v_cndmask_b32_e64 v47, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s5, v42
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	v_cndmask_b32_e64 v42, v48, v47, s[18:19]
	v_add_u32_e32 v47, 2, v46
	v_add_u32_e32 v48, 1, v46
	v_cmp_ne_u32_e64 s[18:19], 0, v42
	v_cndmask_b32_e32 v0, v8, v39, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v42, v48, v47, s[18:19]
	v_xor_b32_e32 v8, s64, v43
	v_cndmask_b32_e32 v0, v46, v42, vcc
	v_mov_b32_e32 v42, v11
	v_mov_b32_e32 v43, v11
	v_xor_b32_e32 v0, v0, v8
	v_lshl_add_u64 v[42:43], v[10:11], 0, v[42:43]
	v_sub_u32_e32 v0, v0, v8
	v_xor_b32_e32 v39, v42, v11
	v_xad_u32 v8, v0, v6, v30
	v_xor_b32_e32 v10, v43, v11
	v_mad_u64_u32 v[42:43], s[12:13], v39, s7, 0
	v_mul_hi_u32 v0, v39, s6
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[42:43]
	v_mad_u64_u32 v[48:49], s[12:13], v10, s6, 0
	v_add_co_u32_e32 v0, vcc, v42, v48
	v_mad_u64_u32 v[46:47], s[12:13], v10, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v49, vcc
	v_mov_b32_e32 v43, s24
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	v_mul_lo_u32 v0, s5, v42
	v_mul_lo_u32 v46, s4, v43
	v_mad_u64_u32 v[42:43], s[12:13], s4, v42, 0
	v_add3_u32 v0, v43, v46, v0
	v_sub_u32_e32 v43, v10, v0
	v_sub_co_u32_e32 v39, vcc, v39, v42
	v_mul_lo_u32 v8, v8, s33
	s_nop 0
	v_subb_co_u32_e64 v42, s[18:19], v43, v5, vcc
	v_subrev_co_u32_e64 v43, s[18:19], s4, v39
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v46, s[22:23], 0, v42, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s5, v46
	v_subb_co_u32_e64 v42, s[18:19], v42, v5, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s4, v43
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v48, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s5, v46
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v39
	v_cndmask_b32_e64 v47, v47, v48, s[22:23]
	v_subrev_co_u32_e64 v48, s[18:19], s4, v43
	s_nop 1
	v_subbrev_co_u32_e64 v42, s[18:19], 0, v42, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v47
	s_nop 1
	v_cndmask_b32_e64 v42, v46, v42, s[18:19]
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v43, v48, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v10, v39, v10, vcc
	v_cndmask_b32_e32 v0, v0, v42, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v8, v0, v10, v8
	v_or_b32_e32 v10, 3, v4
	v_sub_u32_e32 v0, -4, v4
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v39, 31, v0
	v_cndmask_b32_e64 v43, v11, v39, s[10:11]
	v_cndmask_b32_e64 v42, v10, v0, s[10:11]
	v_mov_b32_e32 v46, v43
	v_mov_b32_e32 v47, v43
	v_lshl_add_u64 v[46:47], v[42:43], 0, v[46:47]
	v_xor_b32_e32 v42, v46, v43
	v_xor_b32_e32 v39, v47, v43
	v_mad_u64_u32 v[46:47], s[12:13], v42, s7, 0
	v_mul_hi_u32 v0, v42, s6
	v_lshl_add_u64 v[46:47], v[0:1], 0, v[46:47]
	v_mad_u64_u32 v[50:51], s[12:13], v39, s6, 0
	v_add_co_u32_e32 v0, vcc, v46, v50
	v_mad_u64_u32 v[48:49], s[12:13], v39, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v46, vcc, v47, v51, vcc
	v_mov_b32_e32 v47, s24
	s_nop 0
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_mul_lo_u32 v0, s5, v46
	v_mul_lo_u32 v47, s4, v47
	v_mad_u64_u32 v[48:49], s[12:13], s4, v46, 0
	v_add3_u32 v0, v49, v47, v0
	v_sub_u32_e32 v47, v39, v0
	v_sub_co_u32_e32 v42, vcc, v42, v48
	s_nop 1
	v_subb_co_u32_e64 v47, s[18:19], v47, v5, vcc
	v_subrev_co_u32_e64 v48, s[18:19], s4, v42
	v_subb_co_u32_e32 v0, vcc, v39, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[18:19], 0, v47, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s5, v47
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s4, v48
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v42
	v_cndmask_b32_e64 v48, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s5, v47
	v_cndmask_b32_e64 v42, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	v_cndmask_b32_e64 v47, v49, v48, s[18:19]
	v_add_u32_e32 v48, 2, v46
	v_add_u32_e32 v49, 1, v46
	v_cmp_ne_u32_e64 s[18:19], 0, v47
	v_cndmask_b32_e32 v0, v39, v42, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v47, v49, v48, s[18:19]
	v_xor_b32_e32 v39, s64, v43
	v_cndmask_b32_e32 v0, v46, v47, vcc
	v_mov_b32_e32 v42, v11
	v_mov_b32_e32 v43, v11
	v_xor_b32_e32 v0, v0, v39
	v_lshl_add_u64 v[42:43], v[10:11], 0, v[42:43]
	v_sub_u32_e32 v0, v0, v39
	v_xor_b32_e32 v50, v42, v11
	v_xad_u32 v39, v0, v6, v30
	v_xor_b32_e32 v10, v43, v11
	v_mad_u64_u32 v[42:43], s[12:13], v50, s7, 0
	v_mul_hi_u32 v0, v50, s6
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[42:43]
	v_mad_u64_u32 v[48:49], s[12:13], v10, s6, 0
	v_add_co_u32_e32 v0, vcc, v42, v48
	v_mad_u64_u32 v[46:47], s[12:13], v10, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v49, vcc
	v_mov_b32_e32 v43, s24
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	v_mul_lo_u32 v0, s5, v42
	v_mul_lo_u32 v46, s4, v43
	v_mad_u64_u32 v[42:43], s[12:13], s4, v42, 0
	v_add3_u32 v0, v43, v46, v0
	v_sub_u32_e32 v43, v10, v0
	v_sub_co_u32_e32 v42, vcc, v50, v42
	s_nop 1
	v_subb_co_u32_e64 v43, s[18:19], v43, v5, vcc
	v_subrev_co_u32_e64 v46, s[18:19], s4, v42
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v43, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s5, v47
	v_subb_co_u32_e64 v43, s[18:19], v43, v5, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v48, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s4, v46
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s5, v47
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v42
	v_cndmask_b32_e64 v48, v48, v49, s[22:23]
	v_subrev_co_u32_e64 v49, s[18:19], s4, v46
	s_nop 1
	v_subbrev_co_u32_e64 v43, s[18:19], 0, v43, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v48
	s_nop 1
	v_cndmask_b32_e64 v43, v47, v43, s[18:19]
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v47, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v46, v49, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v10, v42, v10, vcc
	v_cndmask_b32_e32 v0, v0, v43, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	v_or_b32_e32 v42, 4, v4
	s_nop 0
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_mul_lo_u32 v11, v39, s33
	v_add3_u32 v10, v0, v10, v11
	v_sub_u32_e32 v0, -5, v4
	v_ashrrev_i32_e32 v43, 31, v42
	v_ashrrev_i32_e32 v11, 31, v0
	v_cndmask_b32_e64 v47, v43, v11, s[10:11]
	v_cndmask_b32_e64 v46, v42, v0, s[10:11]
	v_mov_b32_e32 v48, v47
	v_mov_b32_e32 v49, v47
	v_lshl_add_u64 v[48:49], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v39, v48, v47
	v_xor_b32_e32 v11, v49, v47
	v_mad_u64_u32 v[48:49], s[12:13], v39, s7, 0
	v_mul_hi_u32 v0, v39, s6
	v_lshl_add_u64 v[48:49], v[0:1], 0, v[48:49]
	v_mad_u64_u32 v[52:53], s[12:13], v11, s6, 0
	v_add_co_u32_e32 v0, vcc, v48, v52
	v_mad_u64_u32 v[50:51], s[12:13], v11, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v48, vcc, v49, v53, vcc
	v_mov_b32_e32 v49, s24
	s_nop 0
	v_addc_co_u32_e32 v51, vcc, 0, v51, vcc
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[50:51]
	v_mul_lo_u32 v0, s5, v48
	v_mul_lo_u32 v46, s4, v49
	v_mad_u64_u32 v[50:51], s[12:13], s4, v48, 0
	v_add3_u32 v0, v51, v46, v0
	v_sub_u32_e32 v46, v11, v0
	v_sub_co_u32_e32 v39, vcc, v39, v50
	s_nop 1
	v_subb_co_u32_e64 v46, s[18:19], v46, v5, vcc
	v_subrev_co_u32_e64 v49, s[18:19], s4, v39
	v_subb_co_u32_e32 v0, vcc, v11, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v46, s[18:19], 0, v46, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s5, v46
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v50, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s4, v49
	v_cndmask_b32_e64 v11, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v39
	v_cndmask_b32_e64 v49, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s5, v46
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	v_cndmask_b32_e64 v46, v50, v49, s[18:19]
	v_add_u32_e32 v49, 2, v48
	v_add_u32_e32 v50, 1, v48
	v_cmp_ne_u32_e64 s[18:19], 0, v46
	v_cndmask_b32_e32 v0, v11, v39, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v46, v50, v49, s[18:19]
	v_xor_b32_e32 v11, s64, v47
	v_cndmask_b32_e32 v0, v48, v46, vcc
	v_mov_b32_e32 v46, v43
	v_mov_b32_e32 v47, v43
	v_xor_b32_e32 v0, v0, v11
	v_lshl_add_u64 v[46:47], v[42:43], 0, v[46:47]
	v_sub_u32_e32 v0, v0, v11
	v_xor_b32_e32 v42, v46, v43
	v_xad_u32 v11, v0, v6, v30
	v_xor_b32_e32 v39, v47, v43
	v_mad_u64_u32 v[46:47], s[12:13], v42, s7, 0
	v_mul_hi_u32 v0, v42, s6
	v_lshl_add_u64 v[46:47], v[0:1], 0, v[46:47]
	v_mad_u64_u32 v[50:51], s[12:13], v39, s6, 0
	v_add_co_u32_e32 v0, vcc, v46, v50
	v_mad_u64_u32 v[48:49], s[12:13], v39, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v46, vcc, v47, v51, vcc
	v_mov_b32_e32 v47, s24
	s_nop 0
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_mul_lo_u32 v0, s5, v46
	v_mul_lo_u32 v48, s4, v47
	v_mad_u64_u32 v[46:47], s[12:13], s4, v46, 0
	v_add3_u32 v0, v47, v48, v0
	v_sub_u32_e32 v47, v39, v0
	v_sub_co_u32_e32 v42, vcc, v42, v46
	v_mul_lo_u32 v11, v11, s33
	s_nop 0
	v_subb_co_u32_e64 v46, s[18:19], v47, v5, vcc
	v_subrev_co_u32_e64 v47, s[18:19], s4, v42
	v_subb_co_u32_e32 v0, vcc, v39, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v48, s[22:23], 0, v46, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s5, v48
	v_subb_co_u32_e64 v46, s[18:19], v46, v5, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s4, v47
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v50, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s5, v48
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v42
	v_cndmask_b32_e64 v49, v49, v50, s[22:23]
	v_subrev_co_u32_e64 v50, s[18:19], s4, v47
	s_nop 1
	v_subbrev_co_u32_e64 v46, s[18:19], 0, v46, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v49
	s_nop 1
	v_cndmask_b32_e64 v46, v48, v46, s[18:19]
	v_cndmask_b32_e64 v48, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	s_nop 1
	v_cndmask_b32_e32 v39, v39, v48, vcc
	v_cmp_ne_u32_e32 vcc, 0, v39
	v_cndmask_b32_e64 v39, v47, v50, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v39, v42, v39, vcc
	v_cndmask_b32_e32 v0, v0, v46, vcc
	v_xor_b32_e32 v39, v39, v43
	v_xor_b32_e32 v0, v0, v43
	v_sub_co_u32_e32 v39, vcc, v39, v43
	v_or_b32_e32 v42, 5, v4
	s_nop 0
	v_subb_co_u32_e32 v0, vcc, v0, v43, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v11, v0, v39, v11
	v_sub_u32_e32 v0, -6, v4
	v_ashrrev_i32_e32 v43, 31, v42
	v_ashrrev_i32_e32 v39, 31, v0
	v_cndmask_b32_e64 v47, v43, v39, s[10:11]
	v_cndmask_b32_e64 v46, v42, v0, s[10:11]
	v_mov_b32_e32 v48, v47
	v_mov_b32_e32 v49, v47
	v_lshl_add_u64 v[48:49], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v46, v48, v47
	v_xor_b32_e32 v39, v49, v47
	v_mad_u64_u32 v[48:49], s[12:13], v46, s7, 0
	v_mul_hi_u32 v0, v46, s6
	v_lshl_add_u64 v[48:49], v[0:1], 0, v[48:49]
	v_mad_u64_u32 v[52:53], s[12:13], v39, s6, 0
	v_add_co_u32_e32 v0, vcc, v48, v52
	v_mad_u64_u32 v[50:51], s[12:13], v39, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v48, vcc, v49, v53, vcc
	v_mov_b32_e32 v49, s24
	s_nop 0
	v_addc_co_u32_e32 v51, vcc, 0, v51, vcc
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[50:51]
	v_mul_lo_u32 v0, s5, v48
	v_mul_lo_u32 v49, s4, v49
	v_mad_u64_u32 v[50:51], s[12:13], s4, v48, 0
	v_add3_u32 v0, v51, v49, v0
	v_sub_u32_e32 v49, v39, v0
	v_sub_co_u32_e32 v46, vcc, v46, v50
	s_nop 1
	v_subb_co_u32_e64 v49, s[18:19], v49, v5, vcc
	v_subrev_co_u32_e64 v50, s[18:19], s4, v46
	v_subb_co_u32_e32 v0, vcc, v39, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[18:19], 0, v49, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s5, v49
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v51, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s4, v50
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v46
	v_cndmask_b32_e64 v50, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s5, v49
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	v_cndmask_b32_e64 v49, v51, v50, s[18:19]
	v_add_u32_e32 v50, 2, v48
	v_add_u32_e32 v51, 1, v48
	v_cmp_ne_u32_e64 s[18:19], 0, v49
	v_cndmask_b32_e32 v0, v39, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v49, v51, v50, s[18:19]
	v_xor_b32_e32 v39, s64, v47
	v_cndmask_b32_e32 v0, v48, v49, vcc
	v_mov_b32_e32 v46, v43
	v_mov_b32_e32 v47, v43
	v_xor_b32_e32 v0, v0, v39
	v_lshl_add_u64 v[46:47], v[42:43], 0, v[46:47]
	v_sub_u32_e32 v0, v0, v39
	v_xor_b32_e32 v52, v46, v43
	v_xad_u32 v39, v0, v6, v30
	v_xor_b32_e32 v42, v47, v43
	v_mad_u64_u32 v[46:47], s[12:13], v52, s7, 0
	v_mul_hi_u32 v0, v52, s6
	v_lshl_add_u64 v[46:47], v[0:1], 0, v[46:47]
	v_mad_u64_u32 v[50:51], s[12:13], v42, s6, 0
	v_add_co_u32_e32 v0, vcc, v46, v50
	v_mad_u64_u32 v[48:49], s[12:13], v42, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v46, vcc, v47, v51, vcc
	v_mov_b32_e32 v47, s24
	s_nop 0
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_mul_lo_u32 v0, s5, v46
	v_mul_lo_u32 v48, s4, v47
	v_mad_u64_u32 v[46:47], s[12:13], s4, v46, 0
	v_add3_u32 v0, v47, v48, v0
	v_sub_u32_e32 v47, v42, v0
	v_sub_co_u32_e32 v46, vcc, v52, v46
	v_mul_lo_u32 v39, v39, s33
	s_nop 0
	v_subb_co_u32_e64 v47, s[18:19], v47, v5, vcc
	v_subrev_co_u32_e64 v48, s[18:19], s4, v46
	v_subb_co_u32_e32 v0, vcc, v42, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v47, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s5, v49
	v_subb_co_u32_e64 v47, s[18:19], v47, v5, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v50, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s4, v48
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v51, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s5, v49
	v_cndmask_b32_e64 v42, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v46
	v_cndmask_b32_e64 v50, v50, v51, s[22:23]
	v_subrev_co_u32_e64 v51, s[18:19], s4, v48
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[18:19], 0, v47, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v50
	s_nop 1
	v_cndmask_b32_e64 v47, v49, v47, s[18:19]
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	s_nop 1
	v_cndmask_b32_e32 v42, v42, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v42
	v_cndmask_b32_e64 v42, v48, v51, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v42, v46, v42, vcc
	v_cndmask_b32_e32 v0, v0, v47, vcc
	v_xor_b32_e32 v42, v42, v43
	v_xor_b32_e32 v0, v0, v43
	v_sub_co_u32_e32 v42, vcc, v42, v43
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v43, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v54, v0, v42, v39
	v_or_b32_e32 v42, 6, v4
	v_sub_u32_e32 v0, -7, v4
	v_ashrrev_i32_e32 v43, 31, v42
	v_ashrrev_i32_e32 v39, 31, v0
	v_cndmask_b32_e64 v47, v43, v39, s[10:11]
	v_cndmask_b32_e64 v46, v42, v0, s[10:11]
	v_mov_b32_e32 v48, v47
	v_mov_b32_e32 v49, v47
	v_lshl_add_u64 v[48:49], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v46, v48, v47
	v_xor_b32_e32 v39, v49, v47
	v_mad_u64_u32 v[48:49], s[12:13], v46, s7, 0
	v_mul_hi_u32 v0, v46, s6
	v_lshl_add_u64 v[48:49], v[0:1], 0, v[48:49]
	v_mad_u64_u32 v[52:53], s[12:13], v39, s6, 0
	v_add_co_u32_e32 v0, vcc, v48, v52
	v_mad_u64_u32 v[50:51], s[12:13], v39, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v48, vcc, v49, v53, vcc
	v_mov_b32_e32 v49, s24
	s_nop 0
	v_addc_co_u32_e32 v51, vcc, 0, v51, vcc
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[50:51]
	v_mul_lo_u32 v0, s5, v48
	v_mul_lo_u32 v49, s4, v49
	v_mad_u64_u32 v[50:51], s[12:13], s4, v48, 0
	v_add3_u32 v0, v51, v49, v0
	v_sub_u32_e32 v49, v39, v0
	v_sub_co_u32_e32 v46, vcc, v46, v50
	s_nop 1
	v_subb_co_u32_e64 v49, s[18:19], v49, v5, vcc
	v_subrev_co_u32_e64 v50, s[18:19], s4, v46
	v_subb_co_u32_e32 v0, vcc, v39, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[18:19], 0, v49, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s5, v49
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v51, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s4, v50
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v46
	v_cndmask_b32_e64 v50, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s5, v49
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	v_cndmask_b32_e64 v49, v51, v50, s[18:19]
	v_add_u32_e32 v50, 2, v48
	v_add_u32_e32 v51, 1, v48
	v_cmp_ne_u32_e64 s[18:19], 0, v49
	v_cndmask_b32_e32 v0, v39, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v49, v51, v50, s[18:19]
	v_xor_b32_e32 v39, s64, v47
	v_cndmask_b32_e32 v0, v48, v49, vcc
	v_mov_b32_e32 v46, v43
	v_mov_b32_e32 v47, v43
	v_xor_b32_e32 v0, v0, v39
	v_lshl_add_u64 v[46:47], v[42:43], 0, v[46:47]
	v_sub_u32_e32 v0, v0, v39
	v_xor_b32_e32 v52, v46, v43
	v_xad_u32 v39, v0, v6, v30
	v_xor_b32_e32 v42, v47, v43
	v_mad_u64_u32 v[46:47], s[12:13], v52, s7, 0
	v_mul_hi_u32 v0, v52, s6
	v_lshl_add_u64 v[46:47], v[0:1], 0, v[46:47]
	v_mad_u64_u32 v[50:51], s[12:13], v42, s6, 0
	v_add_co_u32_e32 v0, vcc, v46, v50
	v_mad_u64_u32 v[48:49], s[12:13], v42, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v46, vcc, v47, v51, vcc
	v_mov_b32_e32 v47, s24
	s_nop 0
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_mul_lo_u32 v0, s5, v46
	v_mul_lo_u32 v48, s4, v47
	v_mad_u64_u32 v[46:47], s[12:13], s4, v46, 0
	v_add3_u32 v0, v47, v48, v0
	v_sub_u32_e32 v47, v42, v0
	v_sub_co_u32_e32 v46, vcc, v52, v46
	v_mul_lo_u32 v39, v39, s33
	s_nop 0
	v_subb_co_u32_e64 v47, s[18:19], v47, v5, vcc
	v_subrev_co_u32_e64 v48, s[18:19], s4, v46
	v_subb_co_u32_e32 v0, vcc, v42, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v47, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s5, v49
	v_subb_co_u32_e64 v47, s[18:19], v47, v5, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v50, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s4, v48
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v51, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s5, v49
	v_cndmask_b32_e64 v42, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v46
	v_cndmask_b32_e64 v50, v50, v51, s[22:23]
	v_subrev_co_u32_e64 v51, s[18:19], s4, v48
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[18:19], 0, v47, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v50
	s_nop 1
	v_cndmask_b32_e64 v47, v49, v47, s[18:19]
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	s_nop 1
	v_cndmask_b32_e32 v42, v42, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v42
	v_cndmask_b32_e64 v42, v48, v51, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v42, v46, v42, vcc
	v_cndmask_b32_e32 v0, v0, v47, vcc
	v_xor_b32_e32 v42, v42, v43
	v_xor_b32_e32 v0, v0, v43
	v_sub_co_u32_e32 v42, vcc, v42, v43
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v43, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v56, v0, v42, v39
	v_or_b32_e32 v42, 7, v4
	v_sub_u32_e32 v0, -8, v4
	v_ashrrev_i32_e32 v43, 31, v42
	v_ashrrev_i32_e32 v39, 31, v0
	v_cndmask_b32_e64 v47, v43, v39, s[10:11]
	v_cndmask_b32_e64 v46, v42, v0, s[10:11]
	v_mov_b32_e32 v48, v47
	v_mov_b32_e32 v49, v47
	v_lshl_add_u64 v[48:49], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v46, v48, v47
	v_xor_b32_e32 v39, v49, v47
	v_mad_u64_u32 v[48:49], s[12:13], v46, s7, 0
	v_mul_hi_u32 v0, v46, s6
	v_lshl_add_u64 v[48:49], v[0:1], 0, v[48:49]
	v_mad_u64_u32 v[52:53], s[12:13], v39, s6, 0
	v_add_co_u32_e32 v0, vcc, v48, v52
	v_mad_u64_u32 v[50:51], s[12:13], v39, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v48, vcc, v49, v53, vcc
	v_mov_b32_e32 v49, s24
	s_nop 0
	v_addc_co_u32_e32 v51, vcc, 0, v51, vcc
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[50:51]
	v_mul_lo_u32 v0, s5, v48
	v_mul_lo_u32 v49, s4, v49
	v_mad_u64_u32 v[50:51], s[12:13], s4, v48, 0
	v_add3_u32 v0, v51, v49, v0
	v_sub_u32_e32 v49, v39, v0
	v_sub_co_u32_e32 v46, vcc, v46, v50
	s_nop 1
	v_subb_co_u32_e64 v49, s[18:19], v49, v5, vcc
	v_subrev_co_u32_e64 v50, s[18:19], s4, v46
	v_subb_co_u32_e32 v0, vcc, v39, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[18:19], 0, v49, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s5, v49
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v51, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s4, v50
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v46
	v_cndmask_b32_e64 v50, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s5, v49
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	v_cndmask_b32_e64 v49, v51, v50, s[18:19]
	v_add_u32_e32 v50, 2, v48
	v_add_u32_e32 v51, 1, v48
	v_cmp_ne_u32_e64 s[18:19], 0, v49
	v_cndmask_b32_e32 v0, v39, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v49, v51, v50, s[18:19]
	v_xor_b32_e32 v39, s64, v47
	v_cndmask_b32_e32 v0, v48, v49, vcc
	v_mov_b32_e32 v46, v43
	v_mov_b32_e32 v47, v43
	v_xor_b32_e32 v0, v0, v39
	v_lshl_add_u64 v[46:47], v[42:43], 0, v[46:47]
	v_sub_u32_e32 v0, v0, v39
	v_xor_b32_e32 v52, v46, v43
	v_xad_u32 v39, v0, v6, v30
	v_xor_b32_e32 v42, v47, v43
	v_mad_u64_u32 v[46:47], s[12:13], v52, s7, 0
	v_mul_hi_u32 v0, v52, s6
	v_lshl_add_u64 v[46:47], v[0:1], 0, v[46:47]
	v_mad_u64_u32 v[50:51], s[12:13], v42, s6, 0
	v_add_co_u32_e32 v0, vcc, v46, v50
	v_mad_u64_u32 v[48:49], s[12:13], v42, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v46, vcc, v47, v51, vcc
	v_mov_b32_e32 v47, s24
	s_nop 0
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_mul_lo_u32 v0, s5, v46
	v_mul_lo_u32 v48, s4, v47
	v_mad_u64_u32 v[46:47], s[12:13], s4, v46, 0
	v_add3_u32 v0, v47, v48, v0
	v_sub_u32_e32 v47, v42, v0
	v_sub_co_u32_e32 v46, vcc, v52, v46
	v_mul_lo_u32 v39, v39, s33
	s_nop 0
	v_subb_co_u32_e64 v47, s[18:19], v47, v5, vcc
	v_subrev_co_u32_e64 v48, s[18:19], s4, v46
	v_subb_co_u32_e32 v0, vcc, v42, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v47, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s5, v49
	v_subb_co_u32_e64 v47, s[18:19], v47, v5, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v50, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s4, v48
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v51, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s5, v49
	v_cndmask_b32_e64 v42, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v46
	v_cndmask_b32_e64 v50, v50, v51, s[22:23]
	v_subrev_co_u32_e64 v51, s[18:19], s4, v48
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[18:19], 0, v47, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v50
	s_nop 1
	v_cndmask_b32_e64 v47, v49, v47, s[18:19]
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	s_nop 1
	v_cndmask_b32_e32 v42, v42, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v42
	v_cndmask_b32_e64 v42, v48, v51, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v42, v46, v42, vcc
	v_cndmask_b32_e32 v0, v0, v47, vcc
	v_xor_b32_e32 v42, v42, v43
	v_xor_b32_e32 v0, v0, v43
	v_sub_co_u32_e32 v42, vcc, v42, v43
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v43, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v0, v0, v42, v39
	buffer_load_ubyte v39, v2, s[28:31], 0 offen
	buffer_load_ubyte v51, v7, s[28:31], 0 offen
	buffer_load_ubyte v50, v8, s[28:31], 0 offen
	buffer_load_ubyte v52, v10, s[28:31], 0 offen
	buffer_load_ubyte v53, v11, s[28:31], 0 offen
	buffer_load_ubyte v55, v54, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v54, v56, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v56, v0, s[28:31], 0 offen
	v_or_b32_e32 v10, 8, v4
	v_sub_u32_e32 v0, -9, v4
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v2, 31, v0
	v_cndmask_b32_e64 v43, v11, v2, s[10:11]
	v_cndmask_b32_e64 v42, v10, v0, s[10:11]
	v_mov_b32_e32 v46, v43
	v_mov_b32_e32 v47, v43
	v_lshl_add_u64 v[46:47], v[42:43], 0, v[46:47]
	v_xor_b32_e32 v7, v46, v43
	v_xor_b32_e32 v2, v47, v43
	v_mad_u64_u32 v[46:47], s[12:13], v7, s7, 0
	v_mul_hi_u32 v0, v7, s6
	v_lshl_add_u64 v[46:47], v[0:1], 0, v[46:47]
	v_mad_u64_u32 v[58:59], s[12:13], v2, s6, 0
	v_add_co_u32_e32 v0, vcc, v46, v58
	v_mad_u64_u32 v[48:49], s[12:13], v2, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v46, vcc, v47, v59, vcc
	v_mov_b32_e32 v47, s24
	s_nop 0
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_mul_lo_u32 v0, s5, v46
	v_mul_lo_u32 v8, s4, v47
	v_mad_u64_u32 v[48:49], s[12:13], s4, v46, 0
	v_add3_u32 v0, v49, v8, v0
	v_sub_u32_e32 v8, v2, v0
	v_sub_co_u32_e32 v7, vcc, v7, v48
	s_nop 1
	v_subb_co_u32_e64 v8, s[18:19], v8, v5, vcc
	v_subrev_co_u32_e64 v42, s[18:19], s4, v7
	v_subb_co_u32_e32 v0, vcc, v2, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v8, s[18:19], 0, v8, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s5, v8
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s4, v42
	v_cndmask_b32_e64 v2, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v7
	v_cndmask_b32_e64 v42, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s5, v8
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	v_cndmask_b32_e64 v8, v47, v42, s[18:19]
	v_add_u32_e32 v42, 2, v46
	v_add_u32_e32 v47, 1, v46
	v_cmp_ne_u32_e64 s[18:19], 0, v8
	v_cndmask_b32_e32 v0, v2, v7, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v8, v47, v42, s[18:19]
	v_xor_b32_e32 v2, s64, v43
	v_cndmask_b32_e32 v0, v46, v8, vcc
	v_mov_b32_e32 v42, v11
	v_mov_b32_e32 v43, v11
	v_xor_b32_e32 v0, v0, v2
	v_lshl_add_u64 v[42:43], v[10:11], 0, v[42:43]
	v_sub_u32_e32 v0, v0, v2
	v_xor_b32_e32 v8, v42, v11
	v_xad_u32 v2, v0, v6, v30
	v_xor_b32_e32 v7, v43, v11
	v_mad_u64_u32 v[42:43], s[12:13], v8, s7, 0
	v_mul_hi_u32 v0, v8, s6
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[42:43]
	v_mad_u64_u32 v[48:49], s[12:13], v7, s6, 0
	v_add_co_u32_e32 v0, vcc, v42, v48
	v_mad_u64_u32 v[46:47], s[12:13], v7, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v49, vcc
	v_mov_b32_e32 v43, s24
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	v_mul_lo_u32 v0, s5, v42
	v_mul_lo_u32 v10, s4, v43
	v_mad_u64_u32 v[42:43], s[12:13], s4, v42, 0
	v_add3_u32 v0, v43, v10, v0
	v_sub_u32_e32 v10, v7, v0
	v_sub_co_u32_e32 v8, vcc, v8, v42
	v_mul_lo_u32 v2, v2, s33
	s_nop 0
	v_subb_co_u32_e64 v10, s[18:19], v10, v5, vcc
	v_subrev_co_u32_e64 v42, s[18:19], s4, v8
	v_subb_co_u32_e32 v0, vcc, v7, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[22:23], 0, v10, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s5, v43
	v_subb_co_u32_e64 v10, s[18:19], v10, v5, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v46, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s4, v42
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s5, v43
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v8
	v_cndmask_b32_e64 v46, v46, v47, s[22:23]
	v_subrev_co_u32_e64 v47, s[18:19], s4, v42
	s_nop 1
	v_subbrev_co_u32_e64 v10, s[18:19], 0, v10, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v46
	s_nop 1
	v_cndmask_b32_e64 v10, v43, v10, s[18:19]
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	s_nop 1
	v_cndmask_b32_e32 v7, v7, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v7
	v_cndmask_b32_e64 v7, v42, v47, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v7, v8, v7, vcc
	v_cndmask_b32_e32 v0, v0, v10, vcc
	v_xor_b32_e32 v7, v7, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v7, vcc, v7, v11
	v_or_b32_e32 v10, 9, v4
	s_nop 0
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v2, v0, v7, v2
	v_sub_u32_e32 v0, -10, v4
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v7, 31, v0
	v_cndmask_b32_e64 v43, v11, v7, s[10:11]
	v_cndmask_b32_e64 v42, v10, v0, s[10:11]
	v_mov_b32_e32 v46, v43
	v_mov_b32_e32 v47, v43
	v_lshl_add_u64 v[46:47], v[42:43], 0, v[46:47]
	v_xor_b32_e32 v8, v46, v43
	v_xor_b32_e32 v7, v47, v43
	v_mad_u64_u32 v[46:47], s[12:13], v8, s7, 0
	v_mul_hi_u32 v0, v8, s6
	v_lshl_add_u64 v[46:47], v[0:1], 0, v[46:47]
	v_mad_u64_u32 v[58:59], s[12:13], v7, s6, 0
	v_add_co_u32_e32 v0, vcc, v46, v58
	v_mad_u64_u32 v[48:49], s[12:13], v7, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v46, vcc, v47, v59, vcc
	v_mov_b32_e32 v47, s24
	s_nop 0
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_mul_lo_u32 v0, s5, v46
	v_mul_lo_u32 v42, s4, v47
	v_mad_u64_u32 v[48:49], s[12:13], s4, v46, 0
	v_add3_u32 v0, v49, v42, v0
	v_sub_u32_e32 v42, v7, v0
	v_sub_co_u32_e32 v8, vcc, v8, v48
	s_nop 1
	v_subb_co_u32_e64 v42, s[18:19], v42, v5, vcc
	v_subrev_co_u32_e64 v47, s[18:19], s4, v8
	v_subb_co_u32_e32 v0, vcc, v7, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v42, s[18:19], 0, v42, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s5, v42
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v48, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s4, v47
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v8
	v_cndmask_b32_e64 v47, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s5, v42
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	v_cndmask_b32_e64 v42, v48, v47, s[18:19]
	v_add_u32_e32 v47, 2, v46
	v_add_u32_e32 v48, 1, v46
	v_cmp_ne_u32_e64 s[18:19], 0, v42
	v_cndmask_b32_e32 v0, v7, v8, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v42, v48, v47, s[18:19]
	v_xor_b32_e32 v7, s64, v43
	v_cndmask_b32_e32 v0, v46, v42, vcc
	v_mov_b32_e32 v42, v11
	v_mov_b32_e32 v43, v11
	v_xor_b32_e32 v0, v0, v7
	v_lshl_add_u64 v[42:43], v[10:11], 0, v[42:43]
	v_sub_u32_e32 v0, v0, v7
	v_xor_b32_e32 v10, v42, v11
	v_xad_u32 v7, v0, v6, v30
	v_xor_b32_e32 v8, v43, v11
	v_mad_u64_u32 v[42:43], s[12:13], v10, s7, 0
	v_mul_hi_u32 v0, v10, s6
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[42:43]
	v_mad_u64_u32 v[48:49], s[12:13], v8, s6, 0
	v_add_co_u32_e32 v0, vcc, v42, v48
	v_mad_u64_u32 v[46:47], s[12:13], v8, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v49, vcc
	v_mov_b32_e32 v43, s24
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	v_mul_lo_u32 v0, s5, v42
	v_mul_lo_u32 v46, s4, v43
	v_mad_u64_u32 v[42:43], s[12:13], s4, v42, 0
	v_add3_u32 v0, v43, v46, v0
	v_sub_u32_e32 v43, v8, v0
	v_sub_co_u32_e32 v10, vcc, v10, v42
	v_mul_lo_u32 v7, v7, s33
	s_nop 0
	v_subb_co_u32_e64 v42, s[18:19], v43, v5, vcc
	v_subrev_co_u32_e64 v43, s[18:19], s4, v10
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v46, s[22:23], 0, v42, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s5, v46
	v_subb_co_u32_e64 v42, s[18:19], v42, v5, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s4, v43
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v48, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s5, v46
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v10
	v_cndmask_b32_e64 v47, v47, v48, s[22:23]
	v_subrev_co_u32_e64 v48, s[18:19], s4, v43
	s_nop 1
	v_subbrev_co_u32_e64 v42, s[18:19], 0, v42, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v47
	s_nop 1
	v_cndmask_b32_e64 v42, v46, v42, s[18:19]
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	s_nop 1
	v_cndmask_b32_e32 v8, v8, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v8
	v_cndmask_b32_e64 v8, v43, v48, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v8, v10, v8, vcc
	v_cndmask_b32_e32 v0, v0, v42, vcc
	v_xor_b32_e32 v8, v8, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v8, vcc, v8, v11
	v_or_b32_e32 v10, 10, v4
	s_nop 0
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v7, v0, v8, v7
	v_sub_u32_e32 v0, -11, v4
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v8, 31, v0
	v_cndmask_b32_e64 v43, v11, v8, s[10:11]
	v_cndmask_b32_e64 v42, v10, v0, s[10:11]
	v_mov_b32_e32 v46, v43
	v_mov_b32_e32 v47, v43
	v_lshl_add_u64 v[46:47], v[42:43], 0, v[46:47]
	v_xor_b32_e32 v42, v46, v43
	v_xor_b32_e32 v8, v47, v43
	v_mad_u64_u32 v[46:47], s[12:13], v42, s7, 0
	v_mul_hi_u32 v0, v42, s6
	v_lshl_add_u64 v[46:47], v[0:1], 0, v[46:47]
	v_mad_u64_u32 v[58:59], s[12:13], v8, s6, 0
	v_add_co_u32_e32 v0, vcc, v46, v58
	v_mad_u64_u32 v[48:49], s[12:13], v8, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v46, vcc, v47, v59, vcc
	v_mov_b32_e32 v47, s24
	s_nop 0
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_mul_lo_u32 v0, s5, v46
	v_mul_lo_u32 v47, s4, v47
	v_mad_u64_u32 v[48:49], s[12:13], s4, v46, 0
	v_add3_u32 v0, v49, v47, v0
	v_sub_u32_e32 v47, v8, v0
	v_sub_co_u32_e32 v42, vcc, v42, v48
	s_nop 1
	v_subb_co_u32_e64 v47, s[18:19], v47, v5, vcc
	v_subrev_co_u32_e64 v48, s[18:19], s4, v42
	v_subb_co_u32_e32 v0, vcc, v8, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[18:19], 0, v47, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s5, v47
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s4, v48
	v_cndmask_b32_e64 v8, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v42
	v_cndmask_b32_e64 v48, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s5, v47
	v_cndmask_b32_e64 v42, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	v_cndmask_b32_e64 v47, v49, v48, s[18:19]
	v_add_u32_e32 v48, 2, v46
	v_add_u32_e32 v49, 1, v46
	v_cmp_ne_u32_e64 s[18:19], 0, v47
	v_cndmask_b32_e32 v0, v8, v42, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v47, v49, v48, s[18:19]
	v_xor_b32_e32 v8, s64, v43
	v_cndmask_b32_e32 v0, v46, v47, vcc
	v_mov_b32_e32 v42, v11
	v_mov_b32_e32 v43, v11
	v_xor_b32_e32 v0, v0, v8
	v_lshl_add_u64 v[42:43], v[10:11], 0, v[42:43]
	v_sub_u32_e32 v0, v0, v8
	v_xor_b32_e32 v57, v42, v11
	v_xad_u32 v8, v0, v6, v30
	v_xor_b32_e32 v10, v43, v11
	v_mad_u64_u32 v[42:43], s[12:13], v57, s7, 0
	v_mul_hi_u32 v0, v57, s6
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[42:43]
	v_mad_u64_u32 v[48:49], s[12:13], v10, s6, 0
	v_add_co_u32_e32 v0, vcc, v42, v48
	v_mad_u64_u32 v[46:47], s[12:13], v10, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v49, vcc
	v_mov_b32_e32 v43, s24
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	v_mul_lo_u32 v0, s5, v42
	v_mul_lo_u32 v46, s4, v43
	v_mad_u64_u32 v[42:43], s[12:13], s4, v42, 0
	v_add3_u32 v0, v43, v46, v0
	v_sub_u32_e32 v43, v10, v0
	v_sub_co_u32_e32 v42, vcc, v57, v42
	v_mul_lo_u32 v8, v8, s33
	s_nop 0
	v_subb_co_u32_e64 v43, s[18:19], v43, v5, vcc
	v_subrev_co_u32_e64 v46, s[18:19], s4, v42
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v43, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s5, v47
	v_subb_co_u32_e64 v43, s[18:19], v43, v5, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v48, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s4, v46
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s5, v47
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v42
	v_cndmask_b32_e64 v48, v48, v49, s[22:23]
	v_subrev_co_u32_e64 v49, s[18:19], s4, v46
	s_nop 1
	v_subbrev_co_u32_e64 v43, s[18:19], 0, v43, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v48
	s_nop 1
	v_cndmask_b32_e64 v43, v47, v43, s[18:19]
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v47, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v46, v49, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v10, v42, v10, vcc
	v_cndmask_b32_e32 v0, v0, v43, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v8, v0, v10, v8
	v_or_b32_e32 v10, 11, v4
	v_sub_u32_e32 v0, -12, v4
	v_ashrrev_i32_e32 v11, 31, v10
	v_ashrrev_i32_e32 v42, 31, v0
	v_cndmask_b32_e64 v43, v11, v42, s[10:11]
	v_cndmask_b32_e64 v42, v10, v0, s[10:11]
	v_mov_b32_e32 v46, v43
	v_mov_b32_e32 v47, v43
	v_lshl_add_u64 v[46:47], v[42:43], 0, v[46:47]
	v_xor_b32_e32 v57, v46, v43
	v_xor_b32_e32 v42, v47, v43
	v_mad_u64_u32 v[46:47], s[12:13], v57, s7, 0
	v_mul_hi_u32 v0, v57, s6
	v_lshl_add_u64 v[46:47], v[0:1], 0, v[46:47]
	v_mad_u64_u32 v[58:59], s[12:13], v42, s6, 0
	v_add_co_u32_e32 v0, vcc, v46, v58
	v_mad_u64_u32 v[48:49], s[12:13], v42, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v46, vcc, v47, v59, vcc
	v_mov_b32_e32 v47, s24
	s_nop 0
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_mul_lo_u32 v0, s5, v46
	v_mul_lo_u32 v47, s4, v47
	v_mad_u64_u32 v[48:49], s[12:13], s4, v46, 0
	v_add3_u32 v0, v49, v47, v0
	v_sub_u32_e32 v47, v42, v0
	v_sub_co_u32_e32 v48, vcc, v57, v48
	s_nop 1
	v_subb_co_u32_e64 v47, s[18:19], v47, v5, vcc
	v_subrev_co_u32_e64 v49, s[18:19], s4, v48
	v_subb_co_u32_e32 v0, vcc, v42, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[18:19], 0, v47, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s5, v47
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s4, v49
	v_cndmask_b32_e64 v42, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v48
	v_cndmask_b32_e64 v49, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s5, v47
	v_cndmask_b32_e64 v48, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	v_cndmask_b32_e64 v47, v57, v49, s[18:19]
	v_add_u32_e32 v49, 2, v46
	v_add_u32_e32 v57, 1, v46
	v_cmp_ne_u32_e64 s[18:19], 0, v47
	v_cndmask_b32_e32 v0, v42, v48, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v47, v57, v49, s[18:19]
	v_xor_b32_e32 v42, s64, v43
	v_cndmask_b32_e32 v0, v46, v47, vcc
	v_xor_b32_e32 v0, v0, v42
	v_sub_u32_e32 v0, v0, v42
	v_mov_b32_e32 v42, v11
	v_mov_b32_e32 v43, v11
	v_lshl_add_u64 v[42:43], v[10:11], 0, v[42:43]
	v_xor_b32_e32 v58, v42, v11
	v_xad_u32 v57, v0, v6, v30
	v_xor_b32_e32 v10, v43, v11
	v_mad_u64_u32 v[42:43], s[12:13], v58, s7, 0
	v_mul_hi_u32 v0, v58, s6
	v_lshl_add_u64 v[42:43], v[0:1], 0, v[42:43]
	v_mad_u64_u32 v[48:49], s[12:13], v10, s6, 0
	v_add_co_u32_e32 v0, vcc, v42, v48
	v_mad_u64_u32 v[46:47], s[12:13], v10, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v49, vcc
	v_mov_b32_e32 v43, s24
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	v_mul_lo_u32 v0, s5, v42
	v_mul_lo_u32 v46, s4, v43
	v_mad_u64_u32 v[42:43], s[12:13], s4, v42, 0
	v_add3_u32 v0, v43, v46, v0
	v_sub_u32_e32 v43, v10, v0
	v_sub_co_u32_e32 v42, vcc, v58, v42
	s_nop 1
	v_subb_co_u32_e64 v43, s[18:19], v43, v5, vcc
	v_subrev_co_u32_e64 v46, s[18:19], s4, v42
	v_subb_co_u32_e32 v0, vcc, v10, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v43, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s5, v47
	v_subb_co_u32_e64 v43, s[18:19], v43, v5, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v48, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s4, v46
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s5, v47
	v_cndmask_b32_e64 v10, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v42
	v_cndmask_b32_e64 v48, v48, v49, s[22:23]
	v_subrev_co_u32_e64 v49, s[18:19], s4, v46
	s_nop 1
	v_subbrev_co_u32_e64 v43, s[18:19], 0, v43, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v48
	s_nop 1
	v_cndmask_b32_e64 v43, v47, v43, s[18:19]
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v10, v47, vcc
	v_cmp_ne_u32_e32 vcc, 0, v10
	v_cndmask_b32_e64 v10, v46, v49, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v10, v42, v10, vcc
	v_cndmask_b32_e32 v0, v0, v43, vcc
	v_xor_b32_e32 v10, v10, v11
	v_xor_b32_e32 v0, v0, v11
	v_sub_co_u32_e32 v10, vcc, v10, v11
	v_or_b32_e32 v42, 12, v4
	s_nop 0
	v_subb_co_u32_e32 v0, vcc, v0, v11, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_mul_lo_u32 v11, v57, s33
	v_add3_u32 v10, v0, v10, v11
	v_sub_u32_e32 v0, -13, v4
	v_ashrrev_i32_e32 v43, 31, v42
	v_ashrrev_i32_e32 v11, 31, v0
	v_cndmask_b32_e64 v47, v43, v11, s[10:11]
	v_cndmask_b32_e64 v46, v42, v0, s[10:11]
	v_mov_b32_e32 v48, v47
	v_mov_b32_e32 v49, v47
	v_lshl_add_u64 v[48:49], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v46, v48, v47
	v_xor_b32_e32 v11, v49, v47
	v_mad_u64_u32 v[48:49], s[12:13], v46, s7, 0
	v_mul_hi_u32 v0, v46, s6
	v_lshl_add_u64 v[48:49], v[0:1], 0, v[48:49]
	v_mad_u64_u32 v[60:61], s[12:13], v11, s6, 0
	v_add_co_u32_e32 v0, vcc, v48, v60
	v_mad_u64_u32 v[58:59], s[12:13], v11, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v48, vcc, v49, v61, vcc
	v_mov_b32_e32 v49, s24
	s_nop 0
	v_addc_co_u32_e32 v59, vcc, 0, v59, vcc
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[58:59]
	v_mul_lo_u32 v0, s5, v48
	v_mul_lo_u32 v49, s4, v49
	v_mad_u64_u32 v[58:59], s[12:13], s4, v48, 0
	v_add3_u32 v0, v59, v49, v0
	v_sub_u32_e32 v49, v11, v0
	v_sub_co_u32_e32 v46, vcc, v46, v58
	s_nop 1
	v_subb_co_u32_e64 v49, s[18:19], v49, v5, vcc
	v_subrev_co_u32_e64 v57, s[18:19], s4, v46
	v_subb_co_u32_e32 v0, vcc, v11, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[18:19], 0, v49, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s5, v49
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v58, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s4, v57
	v_cndmask_b32_e64 v11, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v46
	v_cndmask_b32_e64 v57, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s5, v49
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	v_cndmask_b32_e64 v49, v58, v57, s[18:19]
	v_add_u32_e32 v57, 2, v48
	v_add_u32_e32 v58, 1, v48
	v_cmp_ne_u32_e64 s[18:19], 0, v49
	v_cndmask_b32_e32 v0, v11, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v49, v58, v57, s[18:19]
	v_xor_b32_e32 v11, s64, v47
	v_cndmask_b32_e32 v0, v48, v49, vcc
	v_mov_b32_e32 v46, v43
	v_mov_b32_e32 v47, v43
	v_xor_b32_e32 v0, v0, v11
	v_lshl_add_u64 v[46:47], v[42:43], 0, v[46:47]
	v_sub_u32_e32 v0, v0, v11
	v_xor_b32_e32 v57, v46, v43
	v_xad_u32 v11, v0, v6, v30
	v_xor_b32_e32 v42, v47, v43
	v_mad_u64_u32 v[46:47], s[12:13], v57, s7, 0
	v_mul_hi_u32 v0, v57, s6
	v_lshl_add_u64 v[46:47], v[0:1], 0, v[46:47]
	v_mad_u64_u32 v[58:59], s[12:13], v42, s6, 0
	v_add_co_u32_e32 v0, vcc, v46, v58
	v_mad_u64_u32 v[48:49], s[12:13], v42, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v46, vcc, v47, v59, vcc
	v_mov_b32_e32 v47, s24
	s_nop 0
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_mul_lo_u32 v0, s5, v46
	v_mul_lo_u32 v48, s4, v47
	v_mad_u64_u32 v[46:47], s[12:13], s4, v46, 0
	v_add3_u32 v0, v47, v48, v0
	v_sub_u32_e32 v47, v42, v0
	v_sub_co_u32_e32 v46, vcc, v57, v46
	v_mul_lo_u32 v11, v11, s33
	s_nop 0
	v_subb_co_u32_e64 v47, s[18:19], v47, v5, vcc
	v_subrev_co_u32_e64 v48, s[18:19], s4, v46
	v_subb_co_u32_e32 v0, vcc, v42, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v47, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s5, v49
	v_subb_co_u32_e64 v47, s[18:19], v47, v5, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s4, v48
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v58, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s5, v49
	v_cndmask_b32_e64 v42, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v46
	v_cndmask_b32_e64 v57, v57, v58, s[22:23]
	v_subrev_co_u32_e64 v58, s[18:19], s4, v48
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[18:19], 0, v47, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v57
	s_nop 1
	v_cndmask_b32_e64 v47, v49, v47, s[18:19]
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	s_nop 1
	v_cndmask_b32_e32 v42, v42, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v42
	v_cndmask_b32_e64 v42, v48, v58, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v42, v46, v42, vcc
	v_cndmask_b32_e32 v0, v0, v47, vcc
	v_xor_b32_e32 v42, v42, v43
	v_xor_b32_e32 v0, v0, v43
	v_sub_co_u32_e32 v42, vcc, v42, v43
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v43, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v11, v0, v42, v11
	v_or_b32_e32 v42, 13, v4
	v_sub_u32_e32 v0, -14, v4
	v_ashrrev_i32_e32 v43, 31, v42
	v_ashrrev_i32_e32 v46, 31, v0
	v_cndmask_b32_e64 v47, v43, v46, s[10:11]
	v_cndmask_b32_e64 v46, v42, v0, s[10:11]
	v_mov_b32_e32 v48, v47
	v_mov_b32_e32 v49, v47
	v_lshl_add_u64 v[48:49], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v57, v48, v47
	v_xor_b32_e32 v46, v49, v47
	v_mad_u64_u32 v[48:49], s[12:13], v57, s7, 0
	v_mul_hi_u32 v0, v57, s6
	v_lshl_add_u64 v[48:49], v[0:1], 0, v[48:49]
	v_mad_u64_u32 v[60:61], s[12:13], v46, s6, 0
	v_add_co_u32_e32 v0, vcc, v48, v60
	v_mad_u64_u32 v[58:59], s[12:13], v46, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v48, vcc, v49, v61, vcc
	v_mov_b32_e32 v49, s24
	s_nop 0
	v_addc_co_u32_e32 v59, vcc, 0, v59, vcc
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[58:59]
	v_mul_lo_u32 v0, s5, v48
	v_mul_lo_u32 v49, s4, v49
	v_mad_u64_u32 v[58:59], s[12:13], s4, v48, 0
	v_add3_u32 v0, v59, v49, v0
	v_sub_u32_e32 v49, v46, v0
	v_sub_co_u32_e32 v57, vcc, v57, v58
	s_nop 1
	v_subb_co_u32_e64 v49, s[18:19], v49, v5, vcc
	v_subrev_co_u32_e64 v58, s[18:19], s4, v57
	v_subb_co_u32_e32 v0, vcc, v46, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[18:19], 0, v49, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s5, v49
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s4, v58
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v57
	v_cndmask_b32_e64 v58, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s5, v49
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	v_cndmask_b32_e64 v49, v59, v58, s[18:19]
	v_add_u32_e32 v58, 2, v48
	v_add_u32_e32 v59, 1, v48
	v_cmp_ne_u32_e64 s[18:19], 0, v49
	v_cndmask_b32_e32 v0, v46, v57, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v49, v59, v58, s[18:19]
	v_xor_b32_e32 v46, s64, v47
	v_cndmask_b32_e32 v0, v48, v49, vcc
	v_xor_b32_e32 v0, v0, v46
	v_sub_u32_e32 v0, v0, v46
	v_mov_b32_e32 v46, v43
	v_mov_b32_e32 v47, v43
	v_lshl_add_u64 v[46:47], v[42:43], 0, v[46:47]
	v_xor_b32_e32 v60, v46, v43
	v_xad_u32 v57, v0, v6, v30
	v_xor_b32_e32 v42, v47, v43
	v_mad_u64_u32 v[46:47], s[12:13], v60, s7, 0
	v_mul_hi_u32 v0, v60, s6
	v_lshl_add_u64 v[46:47], v[0:1], 0, v[46:47]
	v_mad_u64_u32 v[58:59], s[12:13], v42, s6, 0
	v_add_co_u32_e32 v0, vcc, v46, v58
	v_mad_u64_u32 v[48:49], s[12:13], v42, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v46, vcc, v47, v59, vcc
	v_mov_b32_e32 v47, s24
	s_nop 0
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_mul_lo_u32 v0, s5, v46
	v_mul_lo_u32 v48, s4, v47
	v_mad_u64_u32 v[46:47], s[12:13], s4, v46, 0
	v_add3_u32 v0, v47, v48, v0
	v_sub_u32_e32 v47, v42, v0
	v_sub_co_u32_e32 v46, vcc, v60, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[18:19], v47, v5, vcc
	v_subrev_co_u32_e64 v48, s[18:19], s4, v46
	v_subb_co_u32_e32 v0, vcc, v42, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v47, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s5, v49
	v_subb_co_u32_e64 v47, s[18:19], v47, v5, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v58, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s4, v48
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s5, v49
	v_cndmask_b32_e64 v42, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v46
	v_cndmask_b32_e64 v58, v58, v59, s[22:23]
	v_subrev_co_u32_e64 v59, s[18:19], s4, v48
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[18:19], 0, v47, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v58
	s_nop 1
	v_cndmask_b32_e64 v47, v49, v47, s[18:19]
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	s_nop 1
	v_cndmask_b32_e32 v42, v42, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v42
	v_cndmask_b32_e64 v42, v48, v59, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v42, v46, v42, vcc
	v_cndmask_b32_e32 v0, v0, v47, vcc
	v_xor_b32_e32 v42, v42, v43
	v_xor_b32_e32 v0, v0, v43
	v_sub_co_u32_e32 v42, vcc, v42, v43
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v43, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_mul_lo_u32 v43, v57, s33
	v_add3_u32 v62, v0, v42, v43
	v_or_b32_e32 v42, 14, v4
	v_sub_u32_e32 v0, -15, v4
	v_ashrrev_i32_e32 v43, 31, v42
	v_ashrrev_i32_e32 v46, 31, v0
	v_cndmask_b32_e64 v47, v43, v46, s[10:11]
	v_cndmask_b32_e64 v46, v42, v0, s[10:11]
	v_mov_b32_e32 v48, v47
	v_mov_b32_e32 v49, v47
	v_lshl_add_u64 v[48:49], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v57, v48, v47
	v_xor_b32_e32 v46, v49, v47
	v_mad_u64_u32 v[48:49], s[12:13], v57, s7, 0
	v_mul_hi_u32 v0, v57, s6
	v_lshl_add_u64 v[48:49], v[0:1], 0, v[48:49]
	v_mad_u64_u32 v[60:61], s[12:13], v46, s6, 0
	v_add_co_u32_e32 v0, vcc, v48, v60
	v_mad_u64_u32 v[58:59], s[12:13], v46, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v48, vcc, v49, v61, vcc
	v_mov_b32_e32 v49, s24
	s_nop 0
	v_addc_co_u32_e32 v59, vcc, 0, v59, vcc
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[58:59]
	v_mul_lo_u32 v0, s5, v48
	v_mul_lo_u32 v49, s4, v49
	v_mad_u64_u32 v[58:59], s[12:13], s4, v48, 0
	v_add3_u32 v0, v59, v49, v0
	v_sub_u32_e32 v49, v46, v0
	v_sub_co_u32_e32 v57, vcc, v57, v58
	s_nop 1
	v_subb_co_u32_e64 v49, s[18:19], v49, v5, vcc
	v_subrev_co_u32_e64 v58, s[18:19], s4, v57
	v_subb_co_u32_e32 v0, vcc, v46, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[18:19], 0, v49, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s5, v49
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s4, v58
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v57
	v_cndmask_b32_e64 v58, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s5, v49
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	v_cndmask_b32_e64 v49, v59, v58, s[18:19]
	v_add_u32_e32 v58, 2, v48
	v_add_u32_e32 v59, 1, v48
	v_cmp_ne_u32_e64 s[18:19], 0, v49
	v_cndmask_b32_e32 v0, v46, v57, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v49, v59, v58, s[18:19]
	v_xor_b32_e32 v46, s64, v47
	v_cndmask_b32_e32 v0, v48, v49, vcc
	v_xor_b32_e32 v0, v0, v46
	v_sub_u32_e32 v0, v0, v46
	v_mov_b32_e32 v46, v43
	v_mov_b32_e32 v47, v43
	v_lshl_add_u64 v[46:47], v[42:43], 0, v[46:47]
	v_xor_b32_e32 v60, v46, v43
	v_xad_u32 v57, v0, v6, v30
	v_xor_b32_e32 v42, v47, v43
	v_mad_u64_u32 v[46:47], s[12:13], v60, s7, 0
	v_mul_hi_u32 v0, v60, s6
	v_lshl_add_u64 v[46:47], v[0:1], 0, v[46:47]
	v_mad_u64_u32 v[58:59], s[12:13], v42, s6, 0
	v_add_co_u32_e32 v0, vcc, v46, v58
	v_mad_u64_u32 v[48:49], s[12:13], v42, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v46, vcc, v47, v59, vcc
	v_mov_b32_e32 v47, s24
	s_nop 0
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_mul_lo_u32 v0, s5, v46
	v_mul_lo_u32 v48, s4, v47
	v_mad_u64_u32 v[46:47], s[12:13], s4, v46, 0
	v_add3_u32 v0, v47, v48, v0
	v_sub_u32_e32 v47, v42, v0
	v_sub_co_u32_e32 v46, vcc, v60, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[18:19], v47, v5, vcc
	v_subrev_co_u32_e64 v48, s[18:19], s4, v46
	v_subb_co_u32_e32 v0, vcc, v42, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v47, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s5, v49
	v_subb_co_u32_e64 v47, s[18:19], v47, v5, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v58, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s4, v48
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s5, v49
	v_cndmask_b32_e64 v42, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v46
	v_cndmask_b32_e64 v58, v58, v59, s[22:23]
	v_subrev_co_u32_e64 v59, s[18:19], s4, v48
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[18:19], 0, v47, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v58
	s_nop 1
	v_cndmask_b32_e64 v47, v49, v47, s[18:19]
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	s_nop 1
	v_cndmask_b32_e32 v42, v42, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v42
	v_cndmask_b32_e64 v42, v48, v59, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v42, v46, v42, vcc
	v_cndmask_b32_e32 v0, v0, v47, vcc
	v_xor_b32_e32 v42, v42, v43
	v_xor_b32_e32 v0, v0, v43
	v_sub_co_u32_e32 v42, vcc, v42, v43
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v43, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_mul_lo_u32 v43, v57, s33
	v_add3_u32 v66, v0, v42, v43
	v_or_b32_e32 v42, 15, v4
	v_sub_u32_e32 v0, -16, v4
	v_ashrrev_i32_e32 v43, 31, v42
	v_ashrrev_i32_e32 v4, 31, v0
	v_cndmask_b32_e64 v47, v43, v4, s[10:11]
	v_cndmask_b32_e64 v46, v42, v0, s[10:11]
	v_mov_b32_e32 v48, v47
	v_mov_b32_e32 v49, v47
	v_lshl_add_u64 v[48:49], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v46, v48, v47
	v_xor_b32_e32 v4, v49, v47
	v_mad_u64_u32 v[48:49], s[10:11], v46, s7, 0
	v_mul_hi_u32 v0, v46, s6
	v_lshl_add_u64 v[48:49], v[0:1], 0, v[48:49]
	v_mad_u64_u32 v[60:61], s[10:11], v4, s6, 0
	v_add_co_u32_e32 v0, vcc, v48, v60
	v_mad_u64_u32 v[58:59], s[10:11], v4, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v48, vcc, v49, v61, vcc
	v_mov_b32_e32 v49, s24
	s_nop 0
	v_addc_co_u32_e32 v59, vcc, 0, v59, vcc
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[58:59]
	v_mul_lo_u32 v0, s5, v48
	v_mul_lo_u32 v49, s4, v49
	v_mad_u64_u32 v[58:59], s[10:11], s4, v48, 0
	v_add3_u32 v0, v59, v49, v0
	v_sub_u32_e32 v49, v4, v0
	v_sub_co_u32_e32 v46, vcc, v46, v58
	s_nop 1
	v_subb_co_u32_e64 v49, s[10:11], v49, v5, vcc
	v_subrev_co_u32_e64 v57, s[10:11], s4, v46
	v_subb_co_u32_e32 v0, vcc, v4, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[10:11], 0, v49, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s5, v49
	v_cmp_le_u32_e32 vcc, s5, v0
	s_nop 0
	v_cndmask_b32_e64 v58, 0, -1, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s4, v57
	v_cndmask_b32_e64 v4, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v46
	v_cndmask_b32_e64 v57, 0, -1, s[10:11]
	v_cmp_eq_u32_e64 s[10:11], s5, v49
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	v_cndmask_b32_e64 v49, v58, v57, s[10:11]
	v_add_u32_e32 v57, 2, v48
	v_add_u32_e32 v58, 1, v48
	v_cmp_ne_u32_e64 s[10:11], 0, v49
	v_cndmask_b32_e32 v0, v4, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_cndmask_b32_e64 v49, v58, v57, s[10:11]
	v_xor_b32_e32 v4, s64, v47
	v_cndmask_b32_e32 v0, v48, v49, vcc
	v_mov_b32_e32 v46, v43
	v_mov_b32_e32 v47, v43
	v_xor_b32_e32 v0, v0, v4
	v_lshl_add_u64 v[46:47], v[42:43], 0, v[46:47]
	v_sub_u32_e32 v0, v0, v4
	v_xor_b32_e32 v42, v46, v43
	v_xad_u32 v4, v0, v6, v30
	v_xor_b32_e32 v6, v47, v43
	v_mad_u64_u32 v[46:47], s[10:11], v42, s7, 0
	v_mul_hi_u32 v0, v42, s6
	v_lshl_add_u64 v[46:47], v[0:1], 0, v[46:47]
	v_mad_u64_u32 v[48:49], s[10:11], v6, s7, 0
	v_mad_u64_u32 v[58:59], s[6:7], v6, s6, 0
	v_add_co_u32_e32 v0, vcc, v46, v58
	v_mul_lo_u32 v4, v4, s33
	s_nop 0
	v_addc_co_u32_e32 v46, vcc, v47, v59, vcc
	v_mov_b32_e32 v47, s24
	s_nop 0
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_mul_lo_u32 v0, s5, v46
	v_mul_lo_u32 v48, s4, v47
	v_mad_u64_u32 v[46:47], s[6:7], s4, v46, 0
	v_add3_u32 v0, v47, v48, v0
	v_sub_u32_e32 v47, v6, v0
	v_sub_co_u32_e32 v42, vcc, v42, v46
	s_nop 1
	v_subb_co_u32_e64 v46, s[10:11], v47, v5, vcc
	v_subrev_co_u32_e64 v47, s[10:11], s4, v42
	v_subb_co_u32_e32 v0, vcc, v6, v0, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v48, s[18:19], 0, v46, s[10:11]
	v_cmp_le_u32_e64 s[18:19], s5, v48
	v_subb_co_u32_e64 v5, s[10:11], v46, v5, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s4, v47
	v_subrev_co_u32_e64 v46, s[10:11], s4, v47
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s5, v48
	v_subbrev_co_u32_e64 v5, s[10:11], 0, v5, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v49, v49, v57, s[18:19]
	v_cmp_le_u32_e32 vcc, s5, v0
	v_cmp_ne_u32_e64 s[10:11], 0, v49
	s_nop 0
	v_cndmask_b32_e64 v6, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v42
	v_cndmask_b32_e64 v5, v48, v5, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v48, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v0
	s_nop 1
	v_cndmask_b32_e32 v6, v6, v48, vcc
	v_cmp_ne_u32_e32 vcc, 0, v6
	s_nop 1
	v_cndmask_b32_e32 v0, v0, v5, vcc
	v_cndmask_b32_e64 v5, v47, v46, s[10:11]
	v_cndmask_b32_e32 v5, v42, v5, vcc
	v_xor_b32_e32 v5, v5, v43
	v_xor_b32_e32 v0, v0, v43
	v_sub_co_u32_e32 v5, vcc, v5, v43
	s_nop 1
	v_subb_co_u32_e32 v0, vcc, v0, v43, vcc
	v_ashrrev_i32_e32 v0, 31, v0
	v_and_b32_e32 v0, s42, v0
	v_add3_u32 v0, v0, v5, v4
	buffer_load_ubyte v57, v2, s[28:31], 0 offen
	buffer_load_ubyte v59, v7, s[28:31], 0 offen
	buffer_load_ubyte v58, v8, s[28:31], 0 offen
	buffer_load_ubyte v60, v10, s[28:31], 0 offen
	buffer_load_ubyte v61, v11, s[28:31], 0 offen
	buffer_load_ubyte v65, v62, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v62, v66, s[28:31], 0 offen
	buffer_load_ubyte v67, v0, s[28:31], 0 offen
	v_lshlrev_b32_e32 v2, 2, v24
	s_movk_i32 s4, 0xffc0
	v_mad_i32_i24 v0, v63, s4, v2
	v_readlane_b32 s4, v144, 18
	v_mul_i32_i24_e32 v4, -16, v63
	v_readlane_b32 s5, v144, 19
	s_add_u32 s6, s4, 7
	v_ashrrev_i32_e32 v5, 31, v4
	v_lshl_add_u64 v[6:7], v[30:31], 0, v[24:25]
	s_addc_u32 s7, s5, 0
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[4:5]
	s_sub_u32 s10, -8, s4
	v_bfe_u32 v8, v6, 4, 1
	s_subb_u32 s11, -1, s5
	v_cmp_lt_i64_e64 s[4:5], s[4:5], -7
	v_or_b32_e32 v42, v0, v8
	v_lshlrev_b32_e32 v46, 6, v63
	v_cndmask_b32_e64 v0, 0, -1, s[4:5]
	s_and_b64 s[4:5], s[4:5], exec
	s_cselect_b32 s5, s11, s7
	s_cselect_b32 s4, s10, s6
	s_ashr_i32 s6, s5, 31
	s_lshr_b32 s6, s6, 29
	s_add_u32 s4, s4, s6
	s_addc_u32 s5, s5, 0
	v_readfirstlane_b32 s6, v0
	s_ashr_i64 s[4:5], s[4:5], 3
	s_mov_b32 s7, s6
	s_xor_b64 s[4:5], s[4:5], s[6:7]
	v_add_u32_e32 v4, v42, v46
	s_lshl_b64 s[70:71], s[4:5], 3
	v_ashrrev_i32_e32 v5, 31, v4
	v_or_b32_e32 v0, s71, v5
	v_cmp_ne_u32_e32 vcc, 0, v0
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
	s_cbranch_execz .LBB0_20
	s_ashr_i32 s10, s71, 31
	s_add_u32 s12, s70, s10
	s_mov_b32 s11, s10
	s_addc_u32 s13, s71, s10
	s_xor_b64 s[12:13], s[12:13], s[10:11]
	v_cvt_f32_u32_e32 v0, s12
	v_cvt_f32_u32_e32 v10, s13
	s_sub_u32 s10, 0, s12
	s_subb_u32 s11, 0, s13
	v_ashrrev_i32_e32 v48, 31, v5
	v_fmac_f32_e32 v0, 0x4f800000, v10
	v_rcp_f32_e32 v0, v0
	v_mov_b32_e32 v49, v48
	v_mul_f32_e32 v0, 0x5f7ffffc, v0
	v_mul_f32_e32 v10, 0x2f800000, v0
	v_trunc_f32_e32 v10, v10
	v_fmac_f32_e32 v0, 0xcf800000, v10
	v_cvt_u32_f32_e32 v10, v10
	v_cvt_u32_f32_e32 v0, v0
	v_readfirstlane_b32 s18, v10
	v_readfirstlane_b32 s19, v0
	s_mul_i32 s22, s10, s18
	s_mul_hi_u32 s25, s10, s19
	s_mul_i32 s23, s11, s19
	s_add_i32 s22, s25, s22
	s_add_i32 s22, s22, s23
	s_mul_i32 s26, s10, s19
	s_mul_i32 s25, s19, s22
	s_mul_hi_u32 s27, s19, s26
	s_mul_hi_u32 s23, s19, s22
	s_add_u32 s25, s27, s25
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 s40, s18, s26
	s_mul_i32 s26, s18, s26
	s_add_u32 s25, s25, s26
	s_mul_hi_u32 s27, s18, s22
	s_addc_u32 s23, s23, s40
	s_addc_u32 s25, s27, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s22, s23, s22
	s_addc_u32 s23, 0, s25
	s_add_u32 s19, s19, s22
	s_addc_u32 s18, s18, s23
	s_mul_i32 s22, s10, s18
	s_mul_hi_u32 s23, s10, s19
	s_add_i32 s22, s23, s22
	s_mul_i32 s11, s11, s19
	s_add_i32 s22, s22, s11
	s_mul_i32 s10, s10, s19
	s_mul_hi_u32 s23, s18, s10
	s_mul_i32 s25, s18, s10
	s_mul_i32 s27, s19, s22
	s_mul_hi_u32 s10, s19, s10
	s_mul_hi_u32 s26, s19, s22
	s_add_u32 s10, s10, s27
	s_addc_u32 s26, 0, s26
	s_add_u32 s10, s10, s25
	s_mul_hi_u32 s11, s18, s22
	s_addc_u32 s10, s26, s23
	s_addc_u32 s11, s11, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s10, s10, s22
	s_addc_u32 s11, 0, s11
	s_add_u32 s19, s19, s10
	v_lshl_add_u64 v[10:11], v[4:5], 0, v[48:49]
	s_addc_u32 s18, s18, s11
	v_xor_b32_e32 v25, v10, v48
	v_xor_b32_e32 v5, v11, v48
	v_mad_u64_u32 v[10:11], s[10:11], v25, s18, 0
	v_mul_hi_u32 v0, v25, s19
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[10:11]
	v_mad_u64_u32 v[68:69], s[10:11], v5, s19, 0
	v_add_co_u32_e32 v0, vcc, v0, v68
	v_mad_u64_u32 v[10:11], s[10:11], v5, s18, 0
	s_nop 0
	v_addc_co_u32_e32 v0, vcc, v1, v69, vcc
	v_mov_b32_e32 v1, s24
	s_nop 0
	v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[10:11]
	v_mul_lo_u32 v10, s13, v0
	v_mul_lo_u32 v11, s12, v1
	v_mad_u64_u32 v[0:1], s[10:11], s12, v0, 0
	v_add3_u32 v1, v1, v11, v10
	v_sub_u32_e32 v10, v5, v1
	v_mov_b32_e32 v11, s13
	v_sub_co_u32_e32 v0, vcc, v25, v0
	s_nop 1
	v_subb_co_u32_e64 v10, s[10:11], v10, v11, vcc
	v_subrev_co_u32_e64 v25, s[10:11], s12, v0
	v_subb_co_u32_e32 v1, vcc, v5, v1, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v31, s[18:19], 0, v10, s[10:11]
	v_cmp_le_u32_e64 s[18:19], s13, v31
	v_subb_co_u32_e64 v10, s[10:11], v10, v11, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v43, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s12, v25
	v_subrev_co_u32_e64 v11, s[10:11], s12, v25
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s13, v31
	v_subbrev_co_u32_e64 v10, s[10:11], 0, v10, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v43, v43, v47, s[18:19]
	v_cmp_le_u32_e32 vcc, s13, v1
	v_cmp_ne_u32_e64 s[10:11], 0, v43
	s_nop 0
	v_cndmask_b32_e64 v5, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v0
	v_cndmask_b32_e64 v10, v31, v10, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v1
	s_nop 1
	v_cndmask_b32_e32 v5, v5, v31, vcc
	v_cmp_ne_u32_e32 vcc, 0, v5
	v_cndmask_b32_e64 v5, v25, v11, s[10:11]
	s_nop 0
	v_cndmask_b32_e32 v0, v0, v5, vcc
	v_cndmask_b32_e32 v1, v1, v10, vcc
	v_xor_b32_e32 v0, v0, v48
	v_xor_b32_e32 v1, v1, v48
	v_sub_co_u32_e32 v10, vcc, v0, v48
	s_nop 1
	v_subb_co_u32_e32 v11, vcc, v1, v48, vcc
.LBB0_20:
	s_or_saveexec_b64 s[6:7], s[6:7]
	v_cvt_f32_u32_e32 v96, s70
	s_xor_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB0_22
	v_rcp_iflag_f32_e32 v0, v96
	s_sub_i32 s10, 0, s70
	v_mov_b32_e32 v11, 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	v_mul_lo_u32 v1, s10, v0
	v_mul_hi_u32 v1, v0, v1
	v_add_u32_e32 v0, v0, v1
	v_mul_hi_u32 v0, v4, v0
	v_mul_lo_u32 v0, v0, s70
	v_sub_u32_e32 v0, v4, v0
	v_subrev_u32_e32 v1, s70, v0
	v_cmp_le_u32_e32 vcc, s70, v0
	s_nop 1
	v_cndmask_b32_e32 v0, v0, v1, vcc
	v_subrev_u32_e32 v1, s70, v0
	v_cmp_le_u32_e32 vcc, s70, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v0, v1, vcc
.LBB0_22:
	s_or_b64 exec, exec, s[6:7]
	v_ashrrev_i32_e32 v5, 31, v7
	v_xor_b32_e32 v1, v5, v7
	v_xor_b32_e32 v0, v5, v6
	v_ashrrev_i32_e32 v6, 31, v1
	v_mov_b32_e32 v47, 0
	v_lshrrev_b32_e32 v6, 27, v6
	v_mov_b32_e32 v7, v47
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[6:7]
	v_ashrrev_i64 v[0:1], 5, v[0:1]
	v_xor_b32_e32 v31, v0, v5
	s_lshl_b32 s12, s4, 8
	s_lshr_b64 s[4:5], s[4:5], 24
	s_ashr_i32 s74, s71, 31
	v_mul_lo_u32 v68, s4, v31
	s_add_u32 s4, s70, s74
	s_mov_b32 s75, s74
	s_addc_u32 s5, s71, s74
	v_xor_b32_e32 v1, v1, v5
	s_xor_b64 s[4:5], s[4:5], s[74:75]
	v_mul_lo_u32 v72, s12, v1
	v_cvt_f32_u32_e32 v0, s4
	v_cvt_f32_u32_e32 v1, s5
	v_ashrrev_i32_e32 v43, 31, v42
	v_mad_u64_u32 v[48:49], s[6:7], s12, v31, v[42:43]
	v_fmac_f32_e32 v0, 0x4f800000, v1
	v_rcp_f32_e32 v1, v0
	s_sub_u32 s6, 0, s4
	s_subb_u32 s7, 0, s5
	v_add3_u32 v49, v68, v49, v72
	v_mul_f32_e32 v1, 0x5f7ffffc, v1
	v_mul_f32_e32 v5, 0x2f800000, v1
	v_trunc_f32_e32 v5, v5
	v_fmac_f32_e32 v1, 0xcf800000, v5
	v_cvt_u32_f32_e32 v1, v1
	v_cvt_u32_f32_e32 v5, v5
	v_lshl_add_u64 v[6:7], v[48:49], 0, v[46:47]
	v_ashrrev_i32_e32 v0, 31, v7
	v_readfirstlane_b32 s10, v1
	v_readfirstlane_b32 s18, v5
	s_mul_hi_u32 s13, s6, s10
	s_mul_i32 s19, s6, s18
	s_mul_i32 s11, s7, s10
	s_add_i32 s13, s13, s19
	s_add_i32 s13, s13, s11
	s_mul_i32 s22, s6, s10
	s_mul_i32 s19, s10, s13
	s_mul_hi_u32 s23, s10, s22
	s_mul_hi_u32 s11, s10, s13
	s_add_u32 s19, s23, s19
	s_addc_u32 s11, 0, s11
	s_mul_hi_u32 s24, s18, s22
	s_mul_i32 s22, s18, s22
	s_add_u32 s19, s19, s22
	s_mul_hi_u32 s23, s18, s13
	s_addc_u32 s11, s11, s24
	s_addc_u32 s19, s23, 0
	s_mul_i32 s13, s18, s13
	s_add_u32 s11, s11, s13
	s_addc_u32 s13, 0, s19
	s_add_u32 s10, s10, s11
	s_addc_u32 s11, s18, s13
	s_mul_i32 s13, s6, s11
	s_mul_hi_u32 s18, s6, s10
	s_add_i32 s13, s18, s13
	s_mul_i32 s7, s7, s10
	s_add_i32 s13, s13, s7
	s_mul_i32 s6, s6, s10
	s_mul_hi_u32 s18, s11, s6
	s_mul_i32 s19, s11, s6
	s_mul_i32 s23, s10, s13
	s_mul_hi_u32 s6, s10, s6
	s_mul_hi_u32 s22, s10, s13
	s_add_u32 s6, s6, s23
	s_addc_u32 s22, 0, s22
	s_add_u32 s6, s6, s19
	s_mul_hi_u32 s7, s11, s13
	s_addc_u32 s6, s22, s18
	v_xor_b32_e32 v71, v0, v7
	s_addc_u32 s7, s7, 0
	s_mul_i32 s13, s11, s13
	s_add_u32 s6, s6, s13
	v_ashrrev_i32_e32 v74, 31, v71
	v_xor_b32_e32 v70, v0, v6
	s_addc_u32 s7, 0, s7
	v_mov_b32_e32 v75, v74
	s_add_u32 s10, s10, s6
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[74:75]
	s_addc_u32 s11, s11, s7
	v_xor_b32_e32 v5, v70, v74
	v_xor_b32_e32 v1, v71, v74
	v_mad_u64_u32 v[70:71], s[6:7], v5, s11, 0
	v_mul_hi_u32 v76, v5, s10
	v_mov_b32_e32 v77, v47
	v_lshl_add_u64 v[70:71], v[76:77], 0, v[70:71]
	v_mad_u64_u32 v[78:79], s[6:7], v1, s10, 0
	v_add_co_u32_e32 v25, vcc, v70, v78
	v_mad_u64_u32 v[76:77], s[6:7], v1, s11, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v71, v79, vcc
	v_mov_b32_e32 v71, v47
	s_nop 0
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[76:77]
	v_mul_lo_u32 v25, s5, v70
	v_mul_lo_u32 v66, s4, v71
	v_mad_u64_u32 v[76:77], s[6:7], s4, v70, 0
	v_add3_u32 v25, v77, v66, v25
	v_sub_u32_e32 v66, v1, v25
	v_mov_b32_e32 v69, s5
	v_sub_co_u32_e32 v5, vcc, v5, v76
	s_and_b32 s9, s9, 0xffff
	s_nop 0
	v_subb_co_u32_e64 v66, s[10:11], v66, v69, vcc
	v_subrev_co_u32_e64 v69, s[10:11], s4, v5
	v_subb_co_u32_e32 v1, vcc, v1, v25, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v66, s[10:11], 0, v66, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s5, v66
	v_cmp_le_u32_e32 vcc, s5, v1
	s_mov_b32 s13, 0
	v_cndmask_b32_e64 v71, 0, -1, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s4, v69
	v_cndmask_b32_e64 v25, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v5
	v_cndmask_b32_e64 v69, 0, -1, s[10:11]
	v_cmp_eq_u32_e64 s[10:11], s5, v66
	v_cndmask_b32_e64 v5, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v1
	v_cndmask_b32_e64 v66, v71, v69, s[10:11]
	v_add_u32_e32 v69, 2, v70
	v_add_u32_e32 v71, 1, v70
	v_cmp_ne_u32_e64 s[10:11], 0, v66
	v_cndmask_b32_e32 v1, v25, v5, vcc
	v_cmp_ne_u32_e32 vcc, 0, v1
	v_cndmask_b32_e64 v66, v71, v69, s[10:11]
	s_load_dword s4, s[96:97], 0x58
	v_cndmask_b32_e32 v1, v70, v66, vcc
	v_xor_b32_e32 v5, s74, v74
	v_xor_b32_e32 v1, v1, v5
	v_sub_u32_e32 v1, v1, v5
	v_xor_b32_e32 v1, v1, v0
	v_ashrrev_i32_e32 v5, 31, v11
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v1, v1, s4
	v_and_b32_e32 v5, s70, v5
	v_add3_u32 v1, v1, v10, v5
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, 0x7ffffffe
	buffer_load_ubyte v90, v1, s[8:11], 0 offen
	v_sub_co_u32_e32 v5, vcc, -3, v6
	v_or_b32_e32 v1, 2, v6
	s_nop 0
	v_subb_co_u32_e32 v6, vcc, -1, v7, vcc
	v_cmp_gt_i32_e32 vcc, 0, v7
	s_nop 1
	v_cndmask_b32_e32 v11, v7, v6, vcc
	v_cndmask_b32_e32 v10, v1, v5, vcc
	v_or_b32_e32 v1, s71, v11
	v_cmp_ne_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[4:5], exec, s[4:5]
	s_cbranch_execz .LBB0_24
	s_add_u32 s6, s70, s74
	s_addc_u32 s7, s71, s74
	s_xor_b64 s[6:7], s[6:7], s[74:75]
	v_cvt_f32_u32_e32 v1, s6
	v_cvt_f32_u32_e32 v5, s7
	s_sub_u32 s18, 0, s6
	s_subb_u32 s19, 0, s7
	v_ashrrev_i32_e32 v6, 31, v11
	v_fmac_f32_e32 v1, 0x4f800000, v5
	v_rcp_f32_e32 v1, v1
	v_mov_b32_e32 v7, v6
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[6:7]
	v_mov_b32_e32 v71, v47
	v_mul_f32_e32 v1, 0x5f7ffffc, v1
	v_mul_f32_e32 v5, 0x2f800000, v1
	v_trunc_f32_e32 v5, v5
	v_fmac_f32_e32 v1, 0xcf800000, v5
	v_cvt_u32_f32_e32 v5, v5
	v_cvt_u32_f32_e32 v1, v1
	v_mov_b32_e32 v66, s7
	v_readfirstlane_b32 s22, v5
	v_readfirstlane_b32 s23, v1
	s_mul_i32 s24, s18, s22
	s_mul_hi_u32 s26, s18, s23
	s_mul_i32 s25, s19, s23
	s_add_i32 s24, s26, s24
	s_mul_i32 s27, s18, s23
	s_add_i32 s24, s24, s25
	s_mul_i32 s26, s23, s24
	s_mul_hi_u32 s40, s23, s27
	s_mul_hi_u32 s25, s23, s24
	s_add_u32 s26, s40, s26
	s_addc_u32 s25, 0, s25
	s_mul_hi_u32 s41, s22, s27
	s_mul_i32 s27, s22, s27
	s_add_u32 s26, s26, s27
	s_mul_hi_u32 s40, s22, s24
	s_addc_u32 s25, s25, s41
	s_addc_u32 s26, s40, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s24, s25, s24
	s_addc_u32 s25, 0, s26
	s_add_u32 s23, s23, s24
	s_addc_u32 s22, s22, s25
	s_mul_i32 s24, s18, s22
	s_mul_hi_u32 s25, s18, s23
	s_add_i32 s24, s25, s24
	s_mul_i32 s19, s19, s23
	s_add_i32 s24, s24, s19
	s_mul_i32 s18, s18, s23
	s_mul_hi_u32 s25, s22, s18
	s_mul_i32 s26, s22, s18
	s_mul_i32 s40, s23, s24
	s_mul_hi_u32 s18, s23, s18
	s_mul_hi_u32 s27, s23, s24
	s_add_u32 s18, s18, s40
	s_addc_u32 s27, 0, s27
	s_add_u32 s18, s18, s26
	s_mul_hi_u32 s19, s22, s24
	s_addc_u32 s18, s27, s25
	s_addc_u32 s19, s19, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s18, s18, s24
	s_addc_u32 s19, 0, s19
	s_add_u32 s23, s23, s18
	s_addc_u32 s22, s22, s19
	v_xor_b32_e32 v5, v10, v6
	v_xor_b32_e32 v1, v11, v6
	v_mad_u64_u32 v[10:11], s[18:19], v5, s22, 0
	v_mul_hi_u32 v70, v5, s23
	v_lshl_add_u64 v[10:11], v[70:71], 0, v[10:11]
	v_mad_u64_u32 v[74:75], s[18:19], v1, s23, 0
	v_add_co_u32_e32 v7, vcc, v10, v74
	v_mad_u64_u32 v[70:71], s[18:19], v1, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v75, vcc
	v_mov_b32_e32 v11, s13
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[70:71]
	v_mul_lo_u32 v7, s7, v10
	v_mul_lo_u32 v25, s6, v11
	v_mad_u64_u32 v[70:71], s[18:19], s6, v10, 0
	v_add3_u32 v7, v71, v25, v7
	v_sub_u32_e32 v25, v1, v7
	v_sub_co_u32_e32 v5, vcc, v5, v70
	v_lshl_add_u64 v[70:71], v[10:11], 0, 2
	s_nop 0
	v_subb_co_u32_e64 v25, s[18:19], v25, v66, vcc
	v_subrev_co_u32_e64 v66, s[18:19], s6, v5
	v_subb_co_u32_e32 v1, vcc, v1, v7, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v25, s[18:19], 0, v25, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v25
	v_cmp_le_u32_e32 vcc, s7, v1
	v_lshl_add_u64 v[74:75], v[10:11], 0, 1
	v_cndmask_b32_e64 v69, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v66
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v5
	v_cndmask_b32_e64 v66, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v25
	v_cndmask_b32_e64 v5, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v1
	v_cndmask_b32_e64 v25, v69, v66, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v1, v7, v5, vcc
	v_cmp_ne_u32_e32 vcc, 0, v25
	s_nop 1
	v_cndmask_b32_e32 v5, v74, v70, vcc
	v_cmp_ne_u32_e32 vcc, 0, v1
	s_nop 1
	v_cndmask_b32_e32 v1, v10, v5, vcc
	v_xor_b32_e32 v5, s74, v6
	v_xor_b32_e32 v1, v1, v5
	v_sub_co_u32_e32 v6, vcc, v1, v5
.LBB0_24:
	s_andn2_saveexec_b64 s[4:5], s[4:5]
	s_cbranch_execz .LBB0_26
	v_rcp_iflag_f32_e32 v1, v96
	s_sub_i32 s6, 0, s70
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_mul_lo_u32 v5, s6, v1
	v_mul_hi_u32 v5, v1, v5
	v_add_u32_e32 v1, v1, v5
	v_mul_hi_u32 v1, v10, v1
	v_mul_lo_u32 v5, v1, s70
	v_sub_u32_e32 v5, v10, v5
	v_add_u32_e32 v6, 1, v1
	v_subrev_u32_e32 v7, s70, v5
	v_cmp_le_u32_e32 vcc, s70, v5
	s_nop 1
	v_cndmask_b32_e32 v5, v5, v7, vcc
	v_cndmask_b32_e32 v1, v1, v6, vcc
	v_add_u32_e32 v6, 1, v1
	v_cmp_le_u32_e32 vcc, s70, v5
	s_nop 1
	v_cndmask_b32_e32 v6, v1, v6, vcc
.LBB0_26:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v4, 2, v4
	v_ashrrev_i32_e32 v5, 31, v4
	v_or_b32_e32 v1, s71, v5
	v_cmp_ne_u32_e32 vcc, 0, v1
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[4:5], exec, s[4:5]
	s_cbranch_execz .LBB0_28
	s_add_u32 s6, s70, s74
	s_mov_b32 s75, s74
	s_addc_u32 s7, s71, s74
	s_xor_b64 s[6:7], s[6:7], s[74:75]
	v_cvt_f32_u32_e32 v1, s6
	v_cvt_f32_u32_e32 v7, s7
	s_sub_u32 s18, 0, s6
	s_subb_u32 s19, 0, s7
	v_ashrrev_i32_e32 v70, 31, v5
	v_fmamk_f32 v1, v7, 0x4f800000, v1
	v_rcp_f32_e32 v1, v1
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[70:71]
	v_mov_b32_e32 v11, 0
	v_mul_f32_e32 v1, 0x5f7ffffc, v1
	v_mul_f32_e32 v7, 0x2f800000, v1
	v_trunc_f32_e32 v7, v7
	v_fmamk_f32 v1, v7, 0xcf800000, v1
	v_cvt_u32_f32_e32 v7, v7
	v_cvt_u32_f32_e32 v1, v1
	v_readfirstlane_b32 s22, v7
	v_readfirstlane_b32 s23, v1
	s_mul_i32 s24, s18, s22
	s_mul_hi_u32 s26, s18, s23
	s_mul_i32 s25, s19, s23
	s_add_i32 s24, s26, s24
	s_add_i32 s24, s24, s25
	s_mul_i32 s27, s18, s23
	s_mul_i32 s26, s23, s24
	s_mul_hi_u32 s40, s23, s27
	s_mul_hi_u32 s25, s23, s24
	s_add_u32 s26, s40, s26
	s_addc_u32 s25, 0, s25
	s_mul_hi_u32 s41, s22, s27
	s_mul_i32 s27, s22, s27
	s_add_u32 s26, s26, s27
	s_mul_hi_u32 s40, s22, s24
	s_addc_u32 s25, s25, s41
	s_addc_u32 s26, s40, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s24, s25, s24
	s_addc_u32 s25, 0, s26
	s_add_u32 s23, s23, s24
	s_addc_u32 s22, s22, s25
	s_mul_i32 s24, s18, s22
	s_mul_hi_u32 s25, s18, s23
	s_add_i32 s24, s25, s24
	s_mul_i32 s19, s19, s23
	s_add_i32 s24, s24, s19
	s_mul_i32 s18, s18, s23
	s_mul_hi_u32 s25, s22, s18
	s_mul_i32 s26, s22, s18
	s_mul_i32 s40, s23, s24
	s_mul_hi_u32 s18, s23, s18
	s_mul_hi_u32 s27, s23, s24
	s_add_u32 s18, s18, s40
	s_addc_u32 s27, 0, s27
	s_add_u32 s18, s18, s26
	s_mul_hi_u32 s19, s22, s24
	s_addc_u32 s18, s27, s25
	s_addc_u32 s19, s19, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s18, s18, s24
	s_addc_u32 s19, 0, s19
	s_add_u32 s23, s23, s18
	s_addc_u32 s22, s22, s19
	v_xor_b32_e32 v7, v4, v70
	v_xor_b32_e32 v1, v5, v70
	v_mad_u64_u32 v[4:5], s[18:19], v7, s22, 0
	v_mul_hi_u32 v10, v7, s23
	v_lshl_add_u64 v[4:5], v[10:11], 0, v[4:5]
	v_mad_u64_u32 v[74:75], s[18:19], v1, s23, 0
	v_add_co_u32_e32 v4, vcc, v4, v74
	v_mad_u64_u32 v[10:11], s[18:19], v1, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v4, vcc, v5, v75, vcc
	v_mov_b32_e32 v5, s13
	s_nop 0
	v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[10:11]
	v_mul_lo_u32 v10, s7, v4
	v_mul_lo_u32 v11, s6, v5
	v_mad_u64_u32 v[4:5], s[18:19], s6, v4, 0
	v_add3_u32 v5, v5, v11, v10
	v_sub_u32_e32 v10, v1, v5
	v_mov_b32_e32 v11, s7
	v_sub_co_u32_e32 v4, vcc, v7, v4
	s_nop 1
	v_subb_co_u32_e64 v7, s[18:19], v10, v11, vcc
	v_subrev_co_u32_e64 v10, s[18:19], s6, v4
	v_subb_co_u32_e32 v1, vcc, v1, v5, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v25, s[22:23], 0, v7, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v25
	v_subb_co_u32_e64 v7, s[18:19], v7, v11, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v66, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v10
	v_subrev_co_u32_e64 v11, s[18:19], s6, v10
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v25
	v_subbrev_co_u32_e64 v7, s[18:19], 0, v7, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v66, v66, v69, s[22:23]
	v_cmp_le_u32_e32 vcc, s7, v1
	v_cmp_ne_u32_e64 s[18:19], 0, v66
	s_nop 0
	v_cndmask_b32_e64 v5, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v4
	v_cndmask_b32_e64 v7, v25, v7, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v25, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v1
	s_nop 1
	v_cndmask_b32_e32 v5, v5, v25, vcc
	v_cmp_ne_u32_e32 vcc, 0, v5
	v_cndmask_b32_e64 v5, v10, v11, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v4, v4, v5, vcc
	v_cndmask_b32_e32 v1, v1, v7, vcc
	v_xor_b32_e32 v4, v4, v70
	v_xor_b32_e32 v1, v1, v70
	v_sub_co_u32_e32 v10, vcc, v4, v70
	s_nop 1
	v_subb_co_u32_e32 v11, vcc, v1, v70, vcc
.LBB0_28:
	s_andn2_saveexec_b64 s[4:5], s[4:5]
	s_cbranch_execz .LBB0_30
	v_rcp_iflag_f32_e32 v1, v96
	s_sub_i32 s6, 0, s70
	v_mov_b32_e32 v11, 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_mul_lo_u32 v5, s6, v1
	v_mul_hi_u32 v5, v1, v5
	v_add_u32_e32 v1, v1, v5
	v_mul_hi_u32 v1, v4, v1
	v_mul_lo_u32 v1, v1, s70
	v_sub_u32_e32 v1, v4, v1
	v_subrev_u32_e32 v4, s70, v1
	v_cmp_le_u32_e32 vcc, s70, v1
	s_nop 1
	v_cndmask_b32_e32 v1, v1, v4, vcc
	v_subrev_u32_e32 v4, s70, v1
	v_cmp_le_u32_e32 vcc, s70, v1
	s_nop 1
	v_cndmask_b32_e32 v10, v1, v4, vcc
.LBB0_30:
	s_or_b64 exec, exec, s[4:5]
	s_load_dword s4, s[96:97], 0x58
	v_xor_b32_e32 v0, v6, v0
	v_ashrrev_i32_e32 v1, 31, v11
	v_and_b32_e32 v1, s70, v1
	v_mov_b32_e32 v7, 0
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v0, v0, s4
	v_add3_u32 v0, v0, v10, v1
	buffer_load_ubyte v79, v0, s[8:11], 0 offen
	v_readlane_b32 s4, v144, 20
	s_lshl_b32 s55, s4, 7
	s_waitcnt vmcnt(0)
	s_barrier
	v_or_b32_e32 v6, 0x100, v36
	v_or_b32_e32 v0, 0x80, v32
	v_cmp_gt_i64_e32 vcc, s[34:35], v[6:7]
	v_add_u32_e32 v1, v0, v34
	v_bfrev_b32_e32 v4, -2
	s_and_b64 s[18:19], vcc, s[2:3]
	s_add_i32 s45, s46, 0x2000
	v_cndmask_b32_e64 v1, v4, v1, s[18:19]
	s_mov_b32 m0, s45
	v_add_u32_e32 v0, v0, v38
	s_and_b64 vcc, vcc, s[14:15]
	s_addk_i32 s55, 0x2000
	buffer_load_dwordx4 v1, s[36:39], 0 offen lds
	v_cndmask_b32_e32 v0, v4, v0, vcc
	s_mov_b32 m0, s55
	s_mov_b64 s[4:5], 0x2000
	buffer_load_dwordx4 v0, s[36:39], 0 offen lds
	v_lshl_add_u64 v[0:1], v[40:41], 0, s[4:5]
	s_movk_i32 s4, 0xdfff
	v_sub_co_u32_e32 v4, vcc, s4, v40
	s_nop 1
	v_subb_co_u32_e32 v5, vcc, -1, v41, vcc
	v_cmp_gt_i32_e32 vcc, 0, v1
	s_nop 1
	v_cndmask_b32_e32 v11, v1, v5, vcc
	v_cndmask_b32_e32 v10, v0, v4, vcc
	v_or_b32_e32 v4, s35, v11
	v_cmp_ne_u32_e32 vcc, 0, v4
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[4:5], exec, s[4:5]
	s_cbranch_execz .LBB0_32
	s_add_u32 s6, s34, s72
	s_mov_b32 s73, s72
	s_addc_u32 s7, s35, s72
	s_xor_b64 s[6:7], s[6:7], s[72:73]
	v_cvt_f32_u32_e32 v4, s6
	v_cvt_f32_u32_e32 v5, s7
	s_sub_u32 s18, 0, s6
	s_subb_u32 s19, 0, s7
	v_fmamk_f32 v4, v5, 0x4f800000, v4
	v_rcp_f32_e32 v4, v4
	s_nop 0
	v_mul_f32_e32 v4, 0x5f7ffffc, v4
	v_mul_f32_e32 v5, 0x2f800000, v4
	v_trunc_f32_e32 v5, v5
	v_fmamk_f32 v4, v5, 0xcf800000, v4
	v_cvt_u32_f32_e32 v5, v5
	v_cvt_u32_f32_e32 v4, v4
	v_readfirstlane_b32 s22, v5
	v_readfirstlane_b32 s23, v4
	s_mul_i32 s24, s18, s22
	s_mul_hi_u32 s26, s18, s23
	s_mul_i32 s25, s19, s23
	s_add_i32 s24, s26, s24
	s_add_i32 s24, s24, s25
	s_mul_i32 s27, s18, s23
	s_mul_i32 s26, s23, s24
	s_mul_hi_u32 s40, s23, s27
	s_mul_hi_u32 s25, s23, s24
	s_add_u32 s26, s40, s26
	s_addc_u32 s25, 0, s25
	s_mul_hi_u32 s41, s22, s27
	s_mul_i32 s27, s22, s27
	s_add_u32 s26, s26, s27
	s_mul_hi_u32 s40, s22, s24
	s_addc_u32 s25, s25, s41
	s_addc_u32 s26, s40, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s24, s25, s24
	s_addc_u32 s25, 0, s26
	s_add_u32 s23, s23, s24
	s_addc_u32 s22, s22, s25
	s_mul_i32 s24, s18, s22
	s_mul_hi_u32 s25, s18, s23
	s_add_i32 s24, s25, s24
	s_mul_i32 s19, s19, s23
	s_add_i32 s24, s24, s19
	s_mul_i32 s18, s18, s23
	s_mul_hi_u32 s25, s22, s18
	s_mul_i32 s26, s22, s18
	s_mul_i32 s40, s23, s24
	s_mul_hi_u32 s18, s23, s18
	s_mul_hi_u32 s27, s23, s24
	s_add_u32 s18, s18, s40
	s_addc_u32 s27, 0, s27
	s_add_u32 s18, s18, s26
	s_mul_hi_u32 s19, s22, s24
	s_addc_u32 s18, s27, s25
	s_addc_u32 s19, s19, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s18, s18, s24
	v_ashrrev_i32_e32 v4, 31, v11
	s_addc_u32 s19, 0, s19
	v_mov_b32_e32 v5, v4
	s_add_u32 s23, s23, s18
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[4:5]
	s_addc_u32 s22, s22, s19
	v_xor_b32_e32 v25, v10, v4
	v_xor_b32_e32 v5, v11, v4
	v_mad_u64_u32 v[10:11], s[18:19], v25, s22, 0
	v_mul_hi_u32 v6, v25, s23
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[10:11]
	v_mad_u64_u32 v[70:71], s[18:19], v5, s23, 0
	v_add_co_u32_e32 v6, vcc, v6, v70
	v_mad_u64_u32 v[10:11], s[18:19], v5, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v6, vcc, v7, v71, vcc
	v_mov_b32_e32 v7, s13
	s_nop 0
	v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[10:11]
	v_mul_lo_u32 v66, s7, v6
	v_mul_lo_u32 v69, s6, v7
	v_mad_u64_u32 v[10:11], s[18:19], s6, v6, 0
	v_add3_u32 v66, v11, v69, v66
	v_sub_u32_e32 v11, v5, v66
	v_mov_b32_e32 v69, s7
	v_sub_co_u32_e32 v25, vcc, v25, v10
	v_lshl_add_u64 v[70:71], v[6:7], 0, 1
	s_nop 0
	v_subb_co_u32_e64 v10, s[18:19], v11, v69, vcc
	v_subrev_co_u32_e64 v11, s[18:19], s6, v25
	v_subb_co_u32_e32 v5, vcc, v5, v66, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v10, s[18:19], 0, v10, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v10
	v_cmp_le_u32_e32 vcc, s7, v5
	v_xor_b32_e32 v4, s72, v4
	v_cndmask_b32_e64 v69, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v11
	s_nop 1
	v_cndmask_b32_e64 v11, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v10
	s_nop 1
	v_cndmask_b32_e64 v69, v69, v11, s[18:19]
	v_lshl_add_u64 v[10:11], v[6:7], 0, 2
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v25
	s_nop 1
	v_cndmask_b32_e64 v11, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v5
	s_nop 1
	v_cndmask_b32_e32 v5, v7, v11, vcc
	v_cmp_ne_u32_e32 vcc, 0, v69
	s_nop 1
	v_cndmask_b32_e32 v7, v70, v10, vcc
	v_cmp_ne_u32_e32 vcc, 0, v5
	s_nop 1
	v_cndmask_b32_e32 v5, v6, v7, vcc
	v_xor_b32_e32 v5, v5, v4
	v_sub_co_u32_e32 v4, vcc, v5, v4
.LBB0_32:
	s_andn2_saveexec_b64 s[4:5], s[4:5]
	s_cbranch_execz .LBB0_34
	v_cvt_f32_u32_e32 v4, s34
	s_sub_i32 s6, 0, s34
	v_rcp_iflag_f32_e32 v4, v4
	s_nop 0
	v_mul_f32_e32 v4, 0x4f7ffffe, v4
	v_cvt_u32_f32_e32 v4, v4
	v_mul_lo_u32 v5, s6, v4
	v_mul_hi_u32 v5, v4, v5
	v_add_u32_e32 v4, v4, v5
	v_mul_hi_u32 v4, v10, v4
	v_mul_lo_u32 v5, v4, s34
	v_sub_u32_e32 v5, v10, v5
	v_add_u32_e32 v6, 1, v4
	v_subrev_u32_e32 v7, s34, v5
	v_cmp_le_u32_e32 vcc, s34, v5
	s_nop 1
	v_cndmask_b32_e32 v5, v5, v7, vcc
	v_cndmask_b32_e32 v4, v4, v6, vcc
	v_add_u32_e32 v6, 1, v4
	v_cmp_le_u32_e32 vcc, s34, v5
	s_nop 1
	v_cndmask_b32_e32 v4, v4, v6, vcc
.LBB0_34:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v5, s35, v1
	v_cmp_ne_u32_e32 vcc, 0, v5
	v_ashrrev_i32_e32 v6, 31, v1
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[4:5], exec, s[4:5]
	s_cbranch_execz .LBB0_36
	s_add_u32 s6, s34, s72
	s_mov_b32 s73, s72
	s_addc_u32 s7, s35, s72
	s_xor_b64 s[6:7], s[6:7], s[72:73]
	v_cvt_f32_u32_e32 v5, s6
	v_cvt_f32_u32_e32 v7, s7
	s_sub_u32 s18, 0, s6
	s_subb_u32 s19, 0, s7
	v_mov_b32_e32 v11, 0
	v_fmamk_f32 v5, v7, 0x4f800000, v5
	v_rcp_f32_e32 v5, v5
	s_nop 0
	v_mul_f32_e32 v5, 0x5f7ffffc, v5
	v_mul_f32_e32 v7, 0x2f800000, v5
	v_trunc_f32_e32 v7, v7
	v_fmamk_f32 v5, v7, 0xcf800000, v5
	v_cvt_u32_f32_e32 v7, v7
	v_cvt_u32_f32_e32 v5, v5
	v_readfirstlane_b32 s22, v7
	v_readfirstlane_b32 s23, v5
	s_mul_i32 s24, s18, s22
	s_mul_hi_u32 s26, s18, s23
	s_mul_i32 s25, s19, s23
	s_add_i32 s24, s26, s24
	s_add_i32 s24, s24, s25
	s_mul_i32 s27, s18, s23
	s_mul_i32 s26, s23, s24
	s_mul_hi_u32 s40, s23, s27
	s_mul_hi_u32 s25, s23, s24
	s_add_u32 s26, s40, s26
	s_addc_u32 s25, 0, s25
	s_mul_hi_u32 s41, s22, s27
	s_mul_i32 s27, s22, s27
	s_add_u32 s26, s26, s27
	s_mul_hi_u32 s40, s22, s24
	s_addc_u32 s25, s25, s41
	s_addc_u32 s26, s40, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s24, s25, s24
	s_addc_u32 s25, 0, s26
	s_add_u32 s23, s23, s24
	s_addc_u32 s22, s22, s25
	s_mul_i32 s24, s18, s22
	s_mul_hi_u32 s25, s18, s23
	s_add_i32 s24, s25, s24
	s_mul_i32 s19, s19, s23
	s_add_i32 s24, s24, s19
	s_mul_i32 s18, s18, s23
	s_mul_hi_u32 s25, s22, s18
	s_mul_i32 s26, s22, s18
	s_mul_i32 s40, s23, s24
	s_mul_hi_u32 s18, s23, s18
	s_mul_hi_u32 s27, s23, s24
	s_add_u32 s18, s18, s40
	s_addc_u32 s27, 0, s27
	s_add_u32 s18, s18, s26
	s_mul_hi_u32 s19, s22, s24
	s_addc_u32 s18, s27, s25
	s_addc_u32 s19, s19, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s18, s18, s24
	s_addc_u32 s19, 0, s19
	v_mov_b32_e32 v7, v6
	s_add_u32 s23, s23, s18
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[6:7]
	s_addc_u32 s22, s22, s19
	v_xor_b32_e32 v7, v0, v6
	v_xor_b32_e32 v5, v1, v6
	v_mad_u64_u32 v[0:1], s[18:19], v7, s22, 0
	v_mul_hi_u32 v10, v7, s23
	v_lshl_add_u64 v[0:1], v[10:11], 0, v[0:1]
	v_mad_u64_u32 v[70:71], s[18:19], v5, s23, 0
	v_add_co_u32_e32 v0, vcc, v0, v70
	v_mad_u64_u32 v[10:11], s[18:19], v5, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v0, vcc, v1, v71, vcc
	v_mov_b32_e32 v1, s13
	s_nop 0
	v_addc_co_u32_e32 v11, vcc, 0, v11, vcc
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[10:11]
	v_mul_lo_u32 v10, s7, v0
	v_mul_lo_u32 v11, s6, v1
	v_mad_u64_u32 v[0:1], s[18:19], s6, v0, 0
	v_add3_u32 v1, v1, v11, v10
	v_sub_u32_e32 v10, v5, v1
	v_mov_b32_e32 v11, s7
	v_sub_co_u32_e32 v0, vcc, v7, v0
	s_nop 1
	v_subb_co_u32_e64 v7, s[18:19], v10, v11, vcc
	v_subrev_co_u32_e64 v10, s[18:19], s6, v0
	v_subb_co_u32_e32 v1, vcc, v5, v1, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v25, s[22:23], 0, v7, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v25
	v_subb_co_u32_e64 v7, s[18:19], v7, v11, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v66, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v10
	v_subrev_co_u32_e64 v11, s[18:19], s6, v10
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v25
	v_subbrev_co_u32_e64 v7, s[18:19], 0, v7, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v66, v66, v69, s[22:23]
	v_cmp_le_u32_e32 vcc, s7, v1
	v_cmp_ne_u32_e64 s[18:19], 0, v66
	s_nop 0
	v_cndmask_b32_e64 v5, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v0
	v_cndmask_b32_e64 v7, v25, v7, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v25, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v1
	s_nop 1
	v_cndmask_b32_e32 v5, v5, v25, vcc
	v_cmp_ne_u32_e32 vcc, 0, v5
	v_cndmask_b32_e64 v5, v10, v11, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v0, v0, v5, vcc
	v_cndmask_b32_e32 v1, v1, v7, vcc
	v_xor_b32_e32 v0, v0, v6
	v_xor_b32_e32 v1, v1, v6
	v_sub_co_u32_e32 v10, vcc, v0, v6
	s_nop 1
	v_subb_co_u32_e32 v11, vcc, v1, v6, vcc
.LBB0_36:
	s_or_saveexec_b64 s[4:5], s[4:5]
	v_and_b32_e32 v25, 7, v24
	v_mul_i32_i24_e32 v66, 0xffffff00, v63
	s_xor_b64 exec, exec, s[4:5]
	s_cbranch_execz .LBB0_38
	v_cvt_f32_u32_e32 v1, s34
	s_sub_i32 s6, 0, s34
	v_mov_b32_e32 v11, 0
	v_rcp_iflag_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_mul_lo_u32 v5, s6, v1
	v_mul_hi_u32 v5, v1, v5
	v_add_u32_e32 v1, v1, v5
	v_mul_hi_u32 v1, v0, v1
	v_mul_lo_u32 v1, v1, s34
	v_sub_u32_e32 v0, v0, v1
	v_subrev_u32_e32 v1, s34, v0
	v_cmp_le_u32_e32 vcc, s34, v0
	s_nop 1
	v_cndmask_b32_e32 v0, v0, v1, vcc
	v_subrev_u32_e32 v1, s34, v0
	v_cmp_le_u32_e32 vcc, s34, v0
	s_nop 1
	v_cndmask_b32_e32 v10, v0, v1, vcc
.LBB0_38:
	s_or_b64 exec, exec, s[4:5]
	v_lshlrev_b16_e32 v0, 8, v65
	v_lshlrev_b16_e32 v1, 8, v67
	v_or_b32_e32 v0, v61, v0
	v_or_b32_sdwa v1, v62, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_xor_b32_e32 v69, v4, v6
	v_or_b32_sdwa v7, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v0, 8, v59
	v_lshlrev_b16_e32 v1, 8, v60
	v_or_b32_e32 v0, v57, v0
	v_or_b32_sdwa v1, v58, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_mov_b32 s4, 0
	v_or_b32_sdwa v6, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v0, 8, v55
	v_lshlrev_b16_e32 v1, 8, v56
	v_or_b32_e32 v0, v53, v0
	v_or_b32_sdwa v1, v54, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_ashrrev_i32_e32 v67, 31, v66
	v_or_b32_sdwa v5, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v0, 8, v51
	v_lshlrev_b16_e32 v1, 8, v52
	v_or_b32_e32 v0, v39, v0
	v_or_b32_sdwa v1, v50, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_ashrrev_i32_e32 v65, 31, v64
	v_or_b32_sdwa v4, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v0, 8, v35
	v_lshlrev_b16_e32 v1, 8, v37
	v_or_b32_e32 v0, v19, v0
	v_or_b32_sdwa v1, v33, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_nop 0
	v_or_b32_sdwa v19, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v0, 8, v22
	v_lshlrev_b16_e32 v1, 8, v23
	v_or_b32_e32 v0, v18, v0
	v_or_b32_sdwa v1, v21, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_nop 0
	v_or_b32_sdwa v18, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v0, 8, v17
	v_lshlrev_b16_e32 v1, 8, v20
	v_or_b32_e32 v0, v15, v0
	v_or_b32_sdwa v1, v16, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_nop 0
	v_or_b32_sdwa v17, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v0, 8, v13
	v_lshlrev_b16_e32 v1, 8, v14
	v_or_b32_e32 v0, v9, v0
	v_or_b32_sdwa v1, v12, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_nop 0
	v_or_b32_sdwa v16, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_ashrrev_i32_e32 v0, 31, v11
	v_and_b32_e32 v1, s35, v0
	v_and_b32_e32 v0, s34, v0
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[10:11]
	v_ashrrev_i32_e32 v9, 31, v1
	v_xor_b32_e32 v1, v9, v1
	v_mov_b32_e32 v11, s4
	v_readlane_b32 s4, v144, 7
	v_ashrrev_i32_e32 v10, 31, v1
	s_add_i32 s40, s4, 0x6200
	v_cmp_lt_i64_e64 s[4:5], s[20:21], 1
	v_xor_b32_e32 v0, v9, v0
	v_lshrrev_b32_e32 v10, 27, v10
	s_and_b64 vcc, s[0:1], s[4:5]
	v_cmp_gt_i64_e64 s[0:1], s[20:21], 0
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[10:11]
	v_alignbit_b32 v0, v1, v0, 5
	v_cndmask_b32_e64 v10, 0, -1, s[0:1]
	s_mov_b32 m0, s40
	v_readfirstlane_b32 s0, v10
	s_mov_b32 s1, s0
	s_xor_b64 s[6:7], s[16:17], s[0:1]
	s_add_u32 s0, s50, 0
	s_addc_u32 s1, 0, 0
	s_xor_b64 s[4:5], s[0:1], 0
	v_cvt_f32_u32_e32 v11, s4
	v_cvt_f32_u32_e32 v12, s5
	v_readlane_b32 s0, v144, 18
	s_sub_u32 s13, 0, s4
	s_subb_u32 s16, 0, s5
	v_mul_lo_u32 v1, v69, s0
	v_xad_u32 v0, v0, v9, v1
	v_fmamk_f32 v1, v12, 0x4f800000, v11
	v_rcp_f32_e32 v1, v1
	buffer_load_dword v0, s[92:95], 0 offen lds
	v_xor_b32_e32 v9, v63, v25
	v_readlane_b32 s1, v144, 19
	v_mul_f32_e32 v0, 0x5f7ffffc, v1
	v_mul_f32_e32 v1, 0x2f800000, v0
	v_trunc_f32_e32 v1, v1
	v_fmamk_f32 v0, v1, 0xcf800000, v0
	v_cvt_u32_f32_e32 v0, v0
	v_cvt_u32_f32_e32 v1, v1
	s_mov_b64 s[0:1], 0
	v_readfirstlane_b32 s17, v0
	v_readfirstlane_b32 s20, v1
	s_mul_hi_u32 s19, s13, s17
	s_mul_i32 s21, s13, s20
	s_mul_i32 s18, s16, s17
	s_add_i32 s19, s19, s21
	s_add_i32 s19, s19, s18
	s_mul_i32 s22, s13, s17
	s_mul_i32 s21, s17, s19
	s_mul_hi_u32 s23, s17, s22
	s_mul_hi_u32 s18, s17, s19
	s_add_u32 s21, s23, s21
	s_addc_u32 s18, 0, s18
	s_mul_hi_u32 s24, s20, s22
	s_mul_i32 s22, s20, s22
	s_add_u32 s21, s21, s22
	s_mul_hi_u32 s23, s20, s19
	s_addc_u32 s18, s18, s24
	s_addc_u32 s21, s23, 0
	s_mul_i32 s19, s20, s19
	s_add_u32 s18, s18, s19
	s_addc_u32 s19, 0, s21
	s_add_u32 s17, s17, s18
	s_addc_u32 s18, s20, s19
	s_mul_i32 s19, s13, s18
	s_mul_hi_u32 s20, s13, s17
	s_add_i32 s19, s20, s19
	s_mul_i32 s16, s16, s17
	s_add_i32 s19, s19, s16
	s_mul_i32 s13, s13, s17
	s_mul_hi_u32 s20, s18, s13
	s_mul_i32 s21, s18, s13
	s_mul_i32 s23, s17, s19
	s_mul_hi_u32 s13, s17, s13
	s_mul_hi_u32 s22, s17, s19
	s_add_u32 s13, s13, s23
	s_addc_u32 s22, 0, s22
	s_add_u32 s13, s13, s21
	s_mul_hi_u32 s16, s18, s19
	s_addc_u32 s13, s22, s20
	s_addc_u32 s16, s16, 0
	s_mul_i32 s19, s18, s19
	s_add_u32 s13, s13, s19
	s_addc_u32 s16, 0, s16
	s_add_u32 s13, s17, s13
	s_addc_u32 s18, s18, s16
	s_ashr_i32 s16, s7, 31
	s_add_u32 s6, s6, s16
	s_mov_b32 s17, s16
	s_addc_u32 s7, s7, s16
	s_xor_b64 s[6:7], s[6:7], s[16:17]
	s_mul_i32 s20, s6, s18
	s_mul_hi_u32 s21, s6, s13
	s_mul_hi_u32 s19, s6, s18
	s_add_u32 s20, s21, s20
	s_addc_u32 s19, 0, s19
	s_mul_hi_u32 s22, s7, s13
	s_mul_i32 s13, s7, s13
	s_add_u32 s13, s20, s13
	s_mul_hi_u32 s21, s7, s18
	s_addc_u32 s13, s19, s22
	s_addc_u32 s19, s21, 0
	s_mul_i32 s18, s7, s18
	s_add_u32 s13, s13, s18
	s_addc_u32 s20, 0, s19
	s_mul_i32 s18, s4, s20
	s_mul_hi_u32 s19, s4, s13
	s_add_i32 s18, s19, s18
	s_mul_i32 s19, s5, s13
	s_add_i32 s21, s18, s19
	s_sub_i32 s22, s7, s21
	s_mul_i32 s18, s4, s13
	s_sub_u32 s6, s6, s18
	s_cselect_b64 s[18:19], -1, 0
	s_subb_u32 s22, s22, s5
	s_sub_u32 s23, s6, s4
	s_subb_u32 s22, s22, 0
	s_cmp_ge_u32 s22, s5
	s_cselect_b32 s24, -1, 0
	s_cmp_ge_u32 s23, s4
	s_cselect_b32 s23, -1, 0
	s_cmp_eq_u32 s22, s5
	s_cselect_b32 s22, s23, s24
	s_add_u32 s23, s13, 1
	s_addc_u32 s24, s20, 0
	s_add_u32 s25, s13, 2
	s_addc_u32 s26, s20, 0
	s_cmp_lg_u32 s22, 0
	s_cselect_b32 s22, s25, s23
	s_cselect_b32 s23, s26, s24
	s_cmp_lg_u64 s[18:19], 0
	s_subb_u32 s7, s7, s21
	s_cmp_ge_u32 s7, s5
	s_cselect_b32 s18, -1, 0
	s_cmp_ge_u32 s6, s4
	s_cselect_b32 s4, -1, 0
	s_cmp_eq_u32 s7, s5
	s_cselect_b32 s4, s4, s18
	s_cmp_lg_u32 s4, 0
	s_cselect_b32 s5, s23, s20
	s_cselect_b32 s4, s22, s13
	s_xor_b64 s[6:7], s[16:17], 0
	s_xor_b64 s[4:5], s[4:5], s[6:7]
	s_sub_u32 s4, s4, s6
	s_subb_u32 s5, s5, s7
	v_readlane_b32 s6, v144, 8
	v_readlane_b32 s7, v144, 9
	v_xor_b32_e32 v0, s4, v10
	v_xor_b32_e32 v1, s5, v10
	s_ashr_i32 s4, s7, 31
	v_cndmask_b32_e32 v11, 0, v1, vcc
	v_cndmask_b32_e32 v10, 0, v0, vcc
	v_lshlrev_b32_e32 v0, 11, v63
	s_mov_b32 s5, s4
	v_mov_b32_e32 v1, 0
	v_lshlrev_b64 v[12:13], 9, v[10:11]
	v_sub_u32_e32 v0, v26, v0
	v_lshlrev_b64 v[10:11], 13, v[10:11]
	s_xor_b64 s[6:7], s[4:5], s[6:7]
	v_lshl_add_u64 v[50:51], v[10:11], 0, v[0:1]
	v_bitop3_b32 v0, v63, v25, 4 bitop3:0x36
	s_lshr_b32 s13, s7, 31
	v_lshl_or_b32 v52, v9, 4, v50
	v_lshl_or_b32 v50, v0, 4, v50
	s_add_u32 s6, s6, s13
	v_or_b32_e32 v97, v12, v2
	ds_read_b128 v[130:133], v50
	ds_read_b128 v[126:129], v52 offset:2048
	ds_read_b128 v[122:125], v50 offset:2048
	s_addc_u32 s7, s7, 0
	ds_read_b32 v78, v97 offset:24576
	ds_read_b128 v[134:137], v52
	s_ashr_i64 s[6:7], s[6:7], 1
	v_mov_b32_e32 v53, v51
	s_xor_b64 s[4:5], s[6:7], s[4:5]
	s_mov_b64 s[6:7], 0x1000
	v_lshl_add_u64 v[54:55], v[52:53], 0, s[6:7]
	v_lshl_add_u64 v[56:57], v[50:51], 0, s[6:7]
	s_mov_b64 s[6:7], 0x1800
	v_lshl_add_u64 v[58:59], v[52:53], 0, s[6:7]
	v_lshl_add_u64 v[60:61], v[50:51], 0, s[6:7]
	v_cmp_gt_i64_e64 s[6:7], s[4:5], 1
	v_or_b32_e32 v62, 0x100, v97
	s_and_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz .LBB0_122
	v_mov_b32_e32 v25, v1
	s_lshl_b64 s[62:63], s[4:5], 1
	s_andn2_b64 vcc, exec, s[0:1]
	v_mov_b32_e32 v69, 0
	s_cbranch_vccz .LBB0_123
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, v69
	v_mov_b32_e32 v13, v69
	v_mov_b32_e32 v12, v69
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, v69
	v_mov_b32_e32 v1, v69
	v_mov_b32_e32 v0, v69
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, v69
	v_mov_b32_e32 v9, v69
	v_mov_b32_e32 v8, v69
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, v69
	v_mov_b32_e32 v21, v69
	v_mov_b32_e32 v20, v69
.LBB0_41:
	v_readlane_b32 s0, v144, 6
	s_lshr_b32 s0, s0, 23
	s_add_u32 s0, s68, s0
	s_addc_u32 s1, s69, 0
	s_ashr_i64 s[0:1], s[0:1], 9
	s_xor_b64 s[0:1], s[0:1], s[66:67]
	s_lshl_b64 s[2:3], s[0:1], 12
	v_or_b32_e32 v33, s3, v25
	v_or_b32_e32 v32, s2, v44
	s_movk_i32 s2, 0xf800
	v_lshl_add_u64 v[36:37], v[32:33], 0, v[66:67]
	s_mov_b32 s3, -1
	v_lshl_add_u64 v[34:35], v[36:37], 0, s[2:3]
	s_movk_i32 s2, 0x7ff
	v_sub_co_u32_e32 v25, vcc, s2, v36
	s_add_u32 s2, s42, s64
	s_nop 0
	v_subb_co_u32_e32 v31, vcc, 0, v37, vcc
	s_mov_b32 s65, s64
	s_addc_u32 s3, s43, s64
	v_cmp_gt_i32_e32 vcc, 0, v35
	s_xor_b64 s[12:13], s[2:3], s[64:65]
	v_cvt_f32_u32_e32 v40, s13
	v_cndmask_b32_e32 v33, v35, v31, vcc
	v_cvt_f32_u32_e32 v31, s12
	v_cndmask_b32_e32 v32, v34, v25, vcc
	v_ashrrev_i32_e32 v38, 31, v33
	v_mov_b32_e32 v39, v38
	v_fmac_f32_e32 v31, 0x4f800000, v40
	v_rcp_f32_e32 v25, v31
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[38:39]
	v_xor_b32_e32 v39, v32, v38
	s_sub_u32 s2, 0, s12
	v_mul_f32_e32 v25, 0x5f7ffffc, v25
	v_mul_f32_e32 v32, 0x2f800000, v25
	v_trunc_f32_e32 v32, v32
	v_fmac_f32_e32 v25, 0xcf800000, v32
	v_cvt_u32_f32_e32 v25, v25
	v_cvt_u32_f32_e32 v32, v32
	s_subb_u32 s3, 0, s13
	v_xor_b32_e32 v31, v33, v38
	v_readfirstlane_b32 s4, v25
	v_readfirstlane_b32 s14, v32
	s_mul_hi_u32 s7, s2, s4
	s_mul_i32 s15, s2, s14
	s_mul_i32 s5, s3, s4
	s_add_i32 s7, s7, s15
	s_add_i32 s7, s7, s5
	s_mul_i32 s16, s2, s4
	s_mul_i32 s15, s4, s7
	s_mul_hi_u32 s17, s4, s16
	s_mul_hi_u32 s5, s4, s7
	s_add_u32 s15, s17, s15
	s_addc_u32 s5, 0, s5
	s_mul_hi_u32 s18, s14, s16
	s_mul_i32 s16, s14, s16
	s_add_u32 s15, s15, s16
	s_mul_hi_u32 s17, s14, s7
	s_addc_u32 s5, s5, s18
	s_addc_u32 s15, s17, 0
	s_mul_i32 s7, s14, s7
	s_add_u32 s5, s5, s7
	s_addc_u32 s7, 0, s15
	s_add_u32 s4, s4, s5
	s_addc_u32 s5, s14, s7
	s_mul_i32 s7, s2, s5
	s_mul_hi_u32 s14, s2, s4
	s_add_i32 s7, s14, s7
	s_mul_i32 s3, s3, s4
	s_add_i32 s7, s7, s3
	s_mul_i32 s2, s2, s4
	s_mul_hi_u32 s14, s5, s2
	s_mul_i32 s15, s5, s2
	s_mul_i32 s17, s4, s7
	s_mul_hi_u32 s2, s4, s2
	s_mul_hi_u32 s16, s4, s7
	s_add_u32 s2, s2, s17
	s_addc_u32 s16, 0, s16
	s_add_u32 s2, s2, s15
	s_mul_hi_u32 s3, s5, s7
	s_addc_u32 s2, s16, s14
	s_addc_u32 s3, s3, 0
	s_mul_i32 s7, s5, s7
	s_add_u32 s2, s2, s7
	s_addc_u32 s3, 0, s3
	s_add_u32 s4, s4, s2
	s_addc_u32 s5, s5, s3
	v_mad_u64_u32 v[40:41], s[2:3], v39, s5, 0
	v_mul_hi_u32 v32, v39, s4
	v_mov_b32_e32 v33, 0
	v_lshl_add_u64 v[40:41], v[32:33], 0, v[40:41]
	v_mad_u64_u32 v[68:69], s[2:3], v31, s4, 0
	v_add_co_u32_e32 v25, vcc, v40, v68
	s_mov_b32 s6, 0
	v_mad_u64_u32 v[66:67], s[2:3], v31, s5, 0
	v_addc_co_u32_e32 v40, vcc, v41, v69, vcc
	v_mov_b32_e32 v41, s6
	s_nop 0
	v_addc_co_u32_e32 v67, vcc, 0, v67, vcc
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[66:67]
	v_mul_lo_u32 v25, s13, v40
	v_mul_lo_u32 v32, s12, v41
	v_mad_u64_u32 v[66:67], s[2:3], s12, v40, 0
	v_add3_u32 v32, v67, v32, v25
	v_sub_u32_e32 v41, v31, v32
	v_mov_b32_e32 v25, s13
	v_sub_co_u32_e32 v39, vcc, v39, v66
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_nop 0
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, vcc
	v_subrev_co_u32_e64 v44, s[2:3], s12, v39
	v_subb_co_u32_e32 v31, vcc, v31, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v41, s[2:3], 0, v41, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v41
	v_cmp_le_u32_e32 vcc, s13, v31
	s_barrier
	v_cndmask_b32_e64 v47, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v44
	v_cndmask_b32_e64 v32, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v39
	v_cndmask_b32_e64 v44, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v41
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v31
	v_cndmask_b32_e64 v41, v47, v44, s[2:3]
	v_add_u32_e32 v44, 2, v40
	v_add_u32_e32 v47, 1, v40
	v_cmp_ne_u32_e64 s[2:3], 0, v41
	v_cndmask_b32_e32 v31, v32, v39, vcc
	v_xor_b32_e32 v32, s64, v38
	v_ashrrev_i32_e32 v38, 31, v35
	v_cndmask_b32_e64 v41, v47, v44, s[2:3]
	v_cmp_ne_u32_e32 vcc, 0, v31
	v_mov_b32_e32 v39, v38
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[38:39]
	v_cndmask_b32_e32 v31, v40, v41, vcc
	v_xor_b32_e32 v31, v31, v32
	v_xor_b32_e32 v44, v34, v38
	v_sub_u32_e32 v31, v31, v32
	v_xor_b32_e32 v39, v35, v38
	v_mad_u64_u32 v[34:35], s[2:3], v44, s5, 0
	v_mul_hi_u32 v32, v44, s4
	v_lshl_add_u64 v[34:35], v[32:33], 0, v[34:35]
	v_mad_u64_u32 v[66:67], s[2:3], v39, s4, 0
	v_add_co_u32_e32 v32, vcc, v34, v66
	v_mad_u64_u32 v[40:41], s[2:3], v39, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v34, vcc, v35, v67, vcc
	v_mov_b32_e32 v35, s6
	s_nop 0
	v_addc_co_u32_e32 v41, vcc, 0, v41, vcc
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[40:41]
	v_mul_lo_u32 v32, s13, v34
	v_mul_lo_u32 v40, s12, v35
	v_mad_u64_u32 v[34:35], s[2:3], s12, v34, 0
	v_add3_u32 v32, v35, v40, v32
	v_sub_u32_e32 v35, v39, v32
	v_sub_co_u32_e32 v34, vcc, v44, v34
	v_xad_u32 v31, v31, v38, v30
	s_nop 0
	v_subb_co_u32_e64 v35, s[2:3], v35, v25, vcc
	v_subrev_co_u32_e64 v40, s[2:3], s12, v34
	v_subb_co_u32_e32 v32, vcc, v39, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v41, s[14:15], 0, v35, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v41
	v_subb_co_u32_e64 v35, s[2:3], v35, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v44, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v40
	v_cmp_le_u32_e32 vcc, s13, v32
	v_mul_lo_u32 v31, v31, s33
	v_cndmask_b32_e64 v47, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v41
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v34
	v_cndmask_b32_e64 v44, v44, v47, s[14:15]
	v_subrev_co_u32_e64 v47, s[2:3], s12, v40
	s_lshl_b64 s[0:1], s[0:1], 9
	s_nop 0
	v_subbrev_co_u32_e64 v35, s[2:3], 0, v35, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v44
	s_nop 1
	v_cndmask_b32_e64 v35, v41, v35, s[2:3]
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v39, v39, v41, vcc
	v_cmp_ne_u32_e32 vcc, 0, v39
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v35, vcc
	v_cndmask_b32_e64 v35, v40, v47, s[2:3]
	v_cndmask_b32_e32 v34, v34, v35, vcc
	v_xor_b32_e32 v34, v34, v38
	v_xor_b32_e32 v32, v32, v38
	v_sub_co_u32_e32 v34, vcc, v34, v38
	s_movk_i32 s2, 0xfa00
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v38, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v44, v32, v34, v31
	v_mov_b32_e32 v31, 0xfffff800
	v_bitop3_b32 v38, v27, s2, v31 bitop3:0xc8
	v_ashrrev_i32_e32 v39, 31, v38
	v_lshl_add_u64 v[34:35], v[36:37], 0, v[64:65]
	v_lshl_add_u64 v[38:39], v[34:35], 0, v[38:39]
	v_sub_co_u32_e32 v31, vcc, -2, v38
	v_lshl_add_u64 v[40:41], v[38:39], 0, 1
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, -1, v39, vcc
	v_cmp_gt_i32_e32 vcc, 0, v41
	s_nop 1
	v_cndmask_b32_e32 v65, v41, v32, vcc
	v_ashrrev_i32_e32 v66, 31, v65
	v_cndmask_b32_e32 v64, v40, v31, vcc
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_xor_b32_e32 v47, v64, v66
	v_xor_b32_e32 v31, v65, v66
	v_mad_u64_u32 v[64:65], s[2:3], v47, s5, 0
	v_mul_hi_u32 v32, v47, s4
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[70:71], s[2:3], v31, s4, 0
	v_add_co_u32_e32 v32, vcc, v64, v70
	v_mad_u64_u32 v[68:69], s[2:3], v31, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v64, vcc, v65, v71, vcc
	v_mov_b32_e32 v65, s6
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[68:69]
	v_mul_lo_u32 v32, s13, v64
	v_mul_lo_u32 v51, s12, v65
	v_mad_u64_u32 v[68:69], s[2:3], s12, v64, 0
	v_add3_u32 v32, v69, v51, v32
	v_sub_u32_e32 v51, v31, v32
	v_sub_co_u32_e32 v47, vcc, v47, v68
	s_nop 1
	v_subb_co_u32_e64 v51, s[2:3], v51, v25, vcc
	v_subrev_co_u32_e64 v53, s[2:3], s12, v47
	v_subb_co_u32_e32 v31, vcc, v31, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v51, s[2:3], 0, v51, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v51
	v_cmp_le_u32_e32 vcc, s13, v31
	s_nop 0
	v_cndmask_b32_e64 v55, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v53
	v_cndmask_b32_e64 v32, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v47
	v_cndmask_b32_e64 v53, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v51
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v31
	v_cndmask_b32_e64 v51, v55, v53, s[2:3]
	v_add_u32_e32 v53, 2, v64
	v_add_u32_e32 v55, 1, v64
	v_cmp_ne_u32_e64 s[2:3], 0, v51
	v_cndmask_b32_e32 v31, v32, v47, vcc
	v_cmp_ne_u32_e32 vcc, 0, v31
	v_cndmask_b32_e64 v51, v55, v53, s[2:3]
	v_xor_b32_e32 v32, s64, v66
	v_cndmask_b32_e32 v31, v64, v51, vcc
	v_ashrrev_i32_e32 v64, 31, v41
	v_mov_b32_e32 v65, v64
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[64:65]
	v_xor_b32_e32 v31, v31, v32
	v_xor_b32_e32 v51, v40, v64
	v_sub_u32_e32 v31, v31, v32
	v_xor_b32_e32 v47, v41, v64
	v_mad_u64_u32 v[40:41], s[2:3], v51, s5, 0
	v_mul_hi_u32 v32, v51, s4
	v_lshl_add_u64 v[40:41], v[32:33], 0, v[40:41]
	v_mad_u64_u32 v[68:69], s[2:3], v47, s4, 0
	v_add_co_u32_e32 v32, vcc, v40, v68
	v_mad_u64_u32 v[66:67], s[2:3], v47, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v40, vcc, v41, v69, vcc
	v_mov_b32_e32 v41, s6
	s_nop 0
	v_addc_co_u32_e32 v67, vcc, 0, v67, vcc
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[66:67]
	v_mul_lo_u32 v32, s13, v40
	v_mul_lo_u32 v53, s12, v41
	v_mad_u64_u32 v[40:41], s[2:3], s12, v40, 0
	v_add3_u32 v32, v41, v53, v32
	v_sub_u32_e32 v41, v47, v32
	v_sub_co_u32_e32 v40, vcc, v51, v40
	v_xad_u32 v31, v31, v64, v30
	s_nop 0
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, vcc
	v_subrev_co_u32_e64 v51, s[2:3], s12, v40
	v_subb_co_u32_e32 v32, vcc, v47, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v53, s[14:15], 0, v41, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v53
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v55, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v51
	v_cmp_le_u32_e32 vcc, s13, v32
	v_mul_lo_u32 v31, v31, s33
	v_cndmask_b32_e64 v57, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v53
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v40
	v_cndmask_b32_e64 v55, v55, v57, s[14:15]
	v_subrev_co_u32_e64 v57, s[2:3], s12, v51
	s_nop 1
	v_subbrev_co_u32_e64 v41, s[2:3], 0, v41, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v55
	s_nop 1
	v_cndmask_b32_e64 v41, v53, v41, s[2:3]
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v47, v47, v53, vcc
	v_cmp_ne_u32_e32 vcc, 0, v47
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v41, vcc
	v_cndmask_b32_e64 v41, v51, v57, s[2:3]
	v_cndmask_b32_e32 v40, v40, v41, vcc
	v_xor_b32_e32 v40, v40, v64
	v_xor_b32_e32 v32, v32, v64
	v_sub_co_u32_e32 v40, vcc, v40, v64
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v64, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v47, v32, v40, v31
	v_lshlrev_b32_e32 v31, 8, v63
	v_and_b32_e32 v31, 0x200, v31
	v_or_b32_e32 v40, 0xfffff800, v31
	v_ashrrev_i32_e32 v41, 31, v40
	v_lshl_add_u64 v[40:41], v[34:35], 0, v[40:41]
	v_sub_co_u32_e32 v32, vcc, -3, v40
	v_lshl_add_u64 v[64:65], v[40:41], 0, 2
	s_nop 0
	v_subb_co_u32_e32 v51, vcc, -1, v41, vcc
	v_cmp_gt_i32_e32 vcc, 0, v65
	s_nop 1
	v_cndmask_b32_e32 v67, v65, v51, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v32, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v53, v66, v68
	v_xor_b32_e32 v51, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v53, s5, 0
	v_mul_hi_u32 v32, v53, s4
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v51, s4, 0
	v_add_co_u32_e32 v32, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v51, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s6
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v32, s13, v66
	v_mul_lo_u32 v55, s12, v67
	v_mad_u64_u32 v[70:71], s[2:3], s12, v66, 0
	v_add3_u32 v32, v71, v55, v32
	v_sub_u32_e32 v55, v51, v32
	v_sub_co_u32_e32 v53, vcc, v53, v70
	s_nop 1
	v_subb_co_u32_e64 v55, s[2:3], v55, v25, vcc
	v_subrev_co_u32_e64 v57, s[2:3], s12, v53
	v_subb_co_u32_e32 v32, vcc, v51, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v55, s[2:3], 0, v55, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v55
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v57
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v53
	v_cndmask_b32_e64 v57, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v55
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v55, v59, v57, s[2:3]
	v_add_u32_e32 v57, 2, v66
	v_add_u32_e32 v59, 1, v66
	v_cmp_ne_u32_e64 s[2:3], 0, v55
	v_cndmask_b32_e32 v32, v51, v53, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v55, v59, v57, s[2:3]
	v_xor_b32_e32 v51, s64, v68
	v_cndmask_b32_e32 v32, v66, v55, vcc
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_xor_b32_e32 v32, v32, v51
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_sub_u32_e32 v32, v32, v51
	v_xor_b32_e32 v55, v64, v66
	v_xad_u32 v51, v32, v66, v30
	v_xor_b32_e32 v53, v65, v66
	v_mad_u64_u32 v[64:65], s[2:3], v55, s5, 0
	v_mul_hi_u32 v32, v55, s4
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[70:71], s[2:3], v53, s4, 0
	v_add_co_u32_e32 v32, vcc, v64, v70
	v_mad_u64_u32 v[68:69], s[2:3], v53, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v64, vcc, v65, v71, vcc
	v_mov_b32_e32 v65, s6
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[68:69]
	v_mul_lo_u32 v32, s13, v64
	v_mul_lo_u32 v57, s12, v65
	v_mad_u64_u32 v[64:65], s[2:3], s12, v64, 0
	v_add3_u32 v32, v65, v57, v32
	v_sub_u32_e32 v57, v53, v32
	v_sub_co_u32_e32 v55, vcc, v55, v64
	v_mul_lo_u32 v51, v51, s33
	s_nop 0
	v_subb_co_u32_e64 v57, s[2:3], v57, v25, vcc
	v_subrev_co_u32_e64 v59, s[2:3], s12, v55
	v_subb_co_u32_e32 v32, vcc, v53, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v61, s[14:15], 0, v57, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v61
	v_subb_co_u32_e64 v57, s[2:3], v57, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v64, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v59
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v61
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v55
	v_cndmask_b32_e64 v64, v64, v65, s[14:15]
	v_subrev_co_u32_e64 v65, s[2:3], s12, v59
	s_nop 1
	v_subbrev_co_u32_e64 v57, s[2:3], 0, v57, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v64
	s_nop 1
	v_cndmask_b32_e64 v57, v61, v57, s[2:3]
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v53, v53, v61, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_cndmask_b32_e64 v53, v59, v65, s[2:3]
	v_lshl_add_u64 v[64:65], v[38:39], 0, 3
	v_cndmask_b32_e32 v53, v55, v53, vcc
	v_cndmask_b32_e32 v32, v32, v57, vcc
	v_xor_b32_e32 v53, v53, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v53, vcc, v53, v66
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v53, v32, v53, v51
	v_sub_co_u32_e32 v32, vcc, -4, v38
	s_nop 1
	v_subb_co_u32_e32 v51, vcc, -1, v39, vcc
	v_cmp_gt_i32_e32 vcc, 0, v65
	s_nop 1
	v_cndmask_b32_e32 v67, v65, v51, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v32, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v55, v66, v68
	v_xor_b32_e32 v51, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v55, s5, 0
	v_mul_hi_u32 v32, v55, s4
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v51, s4, 0
	v_add_co_u32_e32 v32, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v51, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s6
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v32, s13, v66
	v_mul_lo_u32 v57, s12, v67
	v_mad_u64_u32 v[70:71], s[2:3], s12, v66, 0
	v_add3_u32 v32, v71, v57, v32
	v_sub_u32_e32 v57, v51, v32
	v_sub_co_u32_e32 v55, vcc, v55, v70
	s_nop 1
	v_subb_co_u32_e64 v57, s[2:3], v57, v25, vcc
	v_subrev_co_u32_e64 v59, s[2:3], s12, v55
	v_subb_co_u32_e32 v32, vcc, v51, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v57, s[2:3], 0, v57, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v57
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v59
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v55
	v_cndmask_b32_e64 v59, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v57
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v57, v61, v59, s[2:3]
	v_add_u32_e32 v59, 2, v66
	v_add_u32_e32 v61, 1, v66
	v_cmp_ne_u32_e64 s[2:3], 0, v57
	v_cndmask_b32_e32 v32, v51, v55, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v57, v61, v59, s[2:3]
	v_xor_b32_e32 v51, s64, v68
	v_cndmask_b32_e32 v32, v66, v57, vcc
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_xor_b32_e32 v32, v32, v51
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_sub_u32_e32 v32, v32, v51
	v_xor_b32_e32 v57, v64, v66
	v_xad_u32 v51, v32, v66, v30
	v_xor_b32_e32 v55, v65, v66
	v_mad_u64_u32 v[64:65], s[2:3], v57, s5, 0
	v_mul_hi_u32 v32, v57, s4
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[70:71], s[2:3], v55, s4, 0
	v_add_co_u32_e32 v32, vcc, v64, v70
	v_mad_u64_u32 v[68:69], s[2:3], v55, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v64, vcc, v65, v71, vcc
	v_mov_b32_e32 v65, s6
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[68:69]
	v_mul_lo_u32 v32, s13, v64
	v_mul_lo_u32 v59, s12, v65
	v_mad_u64_u32 v[64:65], s[2:3], s12, v64, 0
	v_add3_u32 v32, v65, v59, v32
	v_sub_u32_e32 v59, v55, v32
	v_sub_co_u32_e32 v57, vcc, v57, v64
	v_mul_lo_u32 v51, v51, s33
	s_nop 0
	v_subb_co_u32_e64 v59, s[2:3], v59, v25, vcc
	v_subrev_co_u32_e64 v61, s[2:3], s12, v57
	v_subb_co_u32_e32 v32, vcc, v55, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v64, s[14:15], 0, v59, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v64
	v_subb_co_u32_e64 v59, s[2:3], v59, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v61
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v67, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v64
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v57
	v_cndmask_b32_e64 v65, v65, v67, s[14:15]
	v_subrev_co_u32_e64 v67, s[2:3], s12, v61
	s_nop 1
	v_subbrev_co_u32_e64 v59, s[2:3], 0, v59, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v65
	s_nop 1
	v_cndmask_b32_e64 v59, v64, v59, s[2:3]
	v_cndmask_b32_e64 v64, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v55, v55, v64, vcc
	v_cmp_ne_u32_e32 vcc, 0, v55
	v_cndmask_b32_e64 v55, v61, v67, s[2:3]
	v_lshl_add_u64 v[64:65], v[40:41], 0, 4
	v_cndmask_b32_e32 v55, v57, v55, vcc
	v_cndmask_b32_e32 v32, v32, v59, vcc
	v_xor_b32_e32 v55, v55, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v55, vcc, v55, v66
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v55, v32, v55, v51
	v_sub_co_u32_e32 v32, vcc, -5, v40
	s_nop 1
	v_subb_co_u32_e32 v51, vcc, -1, v41, vcc
	v_cmp_gt_i32_e32 vcc, 0, v65
	s_nop 1
	v_cndmask_b32_e32 v67, v65, v51, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v32, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v57, v66, v68
	v_xor_b32_e32 v51, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v57, s5, 0
	v_mul_hi_u32 v32, v57, s4
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v51, s4, 0
	v_add_co_u32_e32 v32, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v51, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s6
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v32, s13, v66
	v_mul_lo_u32 v59, s12, v67
	v_mad_u64_u32 v[70:71], s[2:3], s12, v66, 0
	v_add3_u32 v32, v71, v59, v32
	v_sub_u32_e32 v59, v51, v32
	v_sub_co_u32_e32 v57, vcc, v57, v70
	s_nop 1
	v_subb_co_u32_e64 v59, s[2:3], v59, v25, vcc
	v_subrev_co_u32_e64 v61, s[2:3], s12, v57
	v_subb_co_u32_e32 v32, vcc, v51, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v59, s[2:3], 0, v59, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v59
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v67, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v61
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v57
	v_cndmask_b32_e64 v61, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v59
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v59, v67, v61, s[2:3]
	v_add_u32_e32 v61, 2, v66
	v_add_u32_e32 v67, 1, v66
	v_cmp_ne_u32_e64 s[2:3], 0, v59
	v_cndmask_b32_e32 v32, v51, v57, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v59, v67, v61, s[2:3]
	v_xor_b32_e32 v51, s64, v68
	v_cndmask_b32_e32 v32, v66, v59, vcc
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_xor_b32_e32 v32, v32, v51
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_sub_u32_e32 v32, v32, v51
	v_xor_b32_e32 v59, v64, v66
	v_xad_u32 v51, v32, v66, v30
	v_xor_b32_e32 v57, v65, v66
	v_mad_u64_u32 v[64:65], s[2:3], v59, s5, 0
	v_mul_hi_u32 v32, v59, s4
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[70:71], s[2:3], v57, s4, 0
	v_add_co_u32_e32 v32, vcc, v64, v70
	v_mad_u64_u32 v[68:69], s[2:3], v57, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v64, vcc, v65, v71, vcc
	v_mov_b32_e32 v65, s6
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[68:69]
	v_mul_lo_u32 v32, s13, v64
	v_mul_lo_u32 v61, s12, v65
	v_mad_u64_u32 v[64:65], s[2:3], s12, v64, 0
	v_add3_u32 v32, v65, v61, v32
	v_sub_u32_e32 v61, v57, v32
	v_sub_co_u32_e32 v59, vcc, v59, v64
	v_mul_lo_u32 v51, v51, s33
	s_nop 0
	v_subb_co_u32_e64 v61, s[2:3], v61, v25, vcc
	v_subrev_co_u32_e64 v64, s[2:3], s12, v59
	v_subb_co_u32_e32 v32, vcc, v57, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v61, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v65
	v_subb_co_u32_e64 v61, s[2:3], v61, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v67, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v64
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v68, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v65
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v59
	v_cndmask_b32_e64 v67, v67, v68, s[14:15]
	v_subrev_co_u32_e64 v68, s[2:3], s12, v64
	s_nop 1
	v_subbrev_co_u32_e64 v61, s[2:3], 0, v61, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v67
	s_nop 1
	v_cndmask_b32_e64 v61, v65, v61, s[2:3]
	v_cndmask_b32_e64 v65, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v57, v57, v65, vcc
	v_cmp_ne_u32_e32 vcc, 0, v57
	v_cndmask_b32_e64 v57, v64, v68, s[2:3]
	v_lshl_add_u64 v[64:65], v[38:39], 0, 5
	v_cndmask_b32_e32 v57, v59, v57, vcc
	v_cndmask_b32_e32 v32, v32, v61, vcc
	v_xor_b32_e32 v57, v57, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v57, vcc, v57, v66
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v57, v32, v57, v51
	v_sub_co_u32_e32 v32, vcc, -6, v38
	s_nop 1
	v_subb_co_u32_e32 v51, vcc, -1, v39, vcc
	v_cmp_gt_i32_e32 vcc, 0, v65
	s_nop 1
	v_cndmask_b32_e32 v67, v65, v51, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v32, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v59, v66, v68
	v_xor_b32_e32 v51, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v59, s5, 0
	v_mul_hi_u32 v32, v59, s4
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v51, s4, 0
	v_add_co_u32_e32 v32, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v51, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s6
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v32, s13, v66
	v_mul_lo_u32 v61, s12, v67
	v_mad_u64_u32 v[70:71], s[2:3], s12, v66, 0
	v_add3_u32 v32, v71, v61, v32
	v_sub_u32_e32 v61, v51, v32
	v_sub_co_u32_e32 v59, vcc, v59, v70
	s_nop 1
	v_subb_co_u32_e64 v61, s[2:3], v61, v25, vcc
	v_subrev_co_u32_e64 v67, s[2:3], s12, v59
	v_subb_co_u32_e32 v32, vcc, v51, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v61, s[2:3], 0, v61, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v61
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v67
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v59
	v_cndmask_b32_e64 v67, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v61
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v61, v69, v67, s[2:3]
	v_add_u32_e32 v67, 2, v66
	v_add_u32_e32 v69, 1, v66
	v_cmp_ne_u32_e64 s[2:3], 0, v61
	v_cndmask_b32_e32 v32, v51, v59, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v61, v69, v67, s[2:3]
	v_xor_b32_e32 v51, s64, v68
	v_cndmask_b32_e32 v32, v66, v61, vcc
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_xor_b32_e32 v32, v32, v51
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_sub_u32_e32 v32, v32, v51
	v_xor_b32_e32 v61, v64, v66
	v_xad_u32 v51, v32, v66, v30
	v_xor_b32_e32 v59, v65, v66
	v_mad_u64_u32 v[64:65], s[2:3], v61, s5, 0
	v_mul_hi_u32 v32, v61, s4
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[70:71], s[2:3], v59, s4, 0
	v_add_co_u32_e32 v32, vcc, v64, v70
	v_mad_u64_u32 v[68:69], s[2:3], v59, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v64, vcc, v65, v71, vcc
	v_mov_b32_e32 v65, s6
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[68:69]
	v_mul_lo_u32 v32, s13, v64
	v_mul_lo_u32 v67, s12, v65
	v_mad_u64_u32 v[64:65], s[2:3], s12, v64, 0
	v_add3_u32 v32, v65, v67, v32
	v_sub_u32_e32 v65, v59, v32
	v_sub_co_u32_e32 v61, vcc, v61, v64
	v_mul_lo_u32 v51, v51, s33
	s_nop 0
	v_subb_co_u32_e64 v64, s[2:3], v65, v25, vcc
	v_subrev_co_u32_e64 v65, s[2:3], s12, v61
	v_subb_co_u32_e32 v32, vcc, v59, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v64, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v67
	v_subb_co_u32_e64 v64, s[2:3], v64, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v68, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v65
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v67
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v61
	v_cndmask_b32_e64 v68, v68, v69, s[14:15]
	v_subrev_co_u32_e64 v69, s[2:3], s12, v65
	s_nop 1
	v_subbrev_co_u32_e64 v64, s[2:3], 0, v64, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v68
	s_nop 1
	v_cndmask_b32_e64 v64, v67, v64, s[2:3]
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v59, v59, v67, vcc
	v_cmp_ne_u32_e32 vcc, 0, v59
	v_cndmask_b32_e64 v59, v65, v69, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v59, v61, v59, vcc
	v_cndmask_b32_e32 v32, v32, v64, vcc
	v_xor_b32_e32 v59, v59, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v59, vcc, v59, v66
	v_lshl_add_u64 v[64:65], v[40:41], 0, 6
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v59, v32, v59, v51
	v_sub_co_u32_e32 v32, vcc, -7, v40
	s_nop 1
	v_subb_co_u32_e32 v51, vcc, -1, v41, vcc
	v_cmp_gt_i32_e32 vcc, 0, v65
	s_nop 1
	v_cndmask_b32_e32 v67, v65, v51, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v32, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v61, v66, v68
	v_xor_b32_e32 v51, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v61, s5, 0
	v_mul_hi_u32 v32, v61, s4
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v51, s4, 0
	v_add_co_u32_e32 v32, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v51, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s6
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v32, s13, v66
	v_mul_lo_u32 v67, s12, v67
	v_mad_u64_u32 v[70:71], s[2:3], s12, v66, 0
	v_add3_u32 v32, v71, v67, v32
	v_sub_u32_e32 v67, v51, v32
	v_sub_co_u32_e32 v61, vcc, v61, v70
	s_nop 1
	v_subb_co_u32_e64 v67, s[2:3], v67, v25, vcc
	v_subrev_co_u32_e64 v69, s[2:3], s12, v61
	v_subb_co_u32_e32 v32, vcc, v51, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[2:3], 0, v67, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v67
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v69
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v61
	v_cndmask_b32_e64 v69, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v67
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v67, v70, v69, s[2:3]
	v_add_u32_e32 v69, 2, v66
	v_add_u32_e32 v70, 1, v66
	v_cmp_ne_u32_e64 s[2:3], 0, v67
	v_cndmask_b32_e32 v32, v51, v61, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v67, v70, v69, s[2:3]
	v_xor_b32_e32 v51, s64, v68
	v_cndmask_b32_e32 v32, v66, v67, vcc
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_xor_b32_e32 v32, v32, v51
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_sub_u32_e32 v32, v32, v51
	v_xor_b32_e32 v67, v64, v66
	v_xad_u32 v51, v32, v66, v30
	v_xor_b32_e32 v61, v65, v66
	v_mad_u64_u32 v[64:65], s[2:3], v67, s5, 0
	v_mul_hi_u32 v32, v67, s4
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[70:71], s[2:3], v61, s4, 0
	v_add_co_u32_e32 v32, vcc, v64, v70
	v_mad_u64_u32 v[68:69], s[2:3], v61, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v64, vcc, v65, v71, vcc
	v_mov_b32_e32 v65, s6
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[68:69]
	v_mul_lo_u32 v32, s13, v64
	v_mul_lo_u32 v68, s12, v65
	v_mad_u64_u32 v[64:65], s[2:3], s12, v64, 0
	v_add3_u32 v32, v65, v68, v32
	v_sub_u32_e32 v65, v61, v32
	v_sub_co_u32_e32 v64, vcc, v67, v64
	v_mul_lo_u32 v51, v51, s33
	s_nop 0
	v_subb_co_u32_e64 v65, s[2:3], v65, v25, vcc
	v_subrev_co_u32_e64 v67, s[2:3], s12, v64
	v_subb_co_u32_e32 v32, vcc, v61, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v68, s[14:15], 0, v65, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v68
	v_subb_co_u32_e64 v65, s[2:3], v65, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v67
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v68
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v64
	v_cndmask_b32_e64 v69, v69, v70, s[14:15]
	v_subrev_co_u32_e64 v70, s[2:3], s12, v67
	s_nop 1
	v_subbrev_co_u32_e64 v65, s[2:3], 0, v65, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v69
	s_nop 1
	v_cndmask_b32_e64 v65, v68, v65, s[2:3]
	v_cndmask_b32_e64 v68, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v61, v61, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v61
	v_cndmask_b32_e64 v61, v67, v70, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v61, v64, v61, vcc
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_xor_b32_e32 v61, v61, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v61, vcc, v61, v66
	v_lshl_add_u64 v[64:65], v[38:39], 0, 7
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v61, v32, v61, v51
	v_sub_co_u32_e32 v32, vcc, -8, v38
	s_nop 1
	v_subb_co_u32_e32 v51, vcc, -1, v39, vcc
	v_cmp_gt_i32_e32 vcc, 0, v65
	s_nop 1
	v_cndmask_b32_e32 v67, v65, v51, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v32, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v69, v66, v68
	v_xor_b32_e32 v51, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v69, s5, 0
	v_mul_hi_u32 v32, v69, s4
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v51, s4, 0
	v_add_co_u32_e32 v32, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v51, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s6
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v32, s13, v66
	v_mul_lo_u32 v67, s12, v67
	v_mad_u64_u32 v[70:71], s[2:3], s12, v66, 0
	v_add3_u32 v32, v71, v67, v32
	v_sub_u32_e32 v67, v51, v32
	v_sub_co_u32_e32 v69, vcc, v69, v70
	s_nop 1
	v_subb_co_u32_e64 v67, s[2:3], v67, v25, vcc
	v_subrev_co_u32_e64 v70, s[2:3], s12, v69
	v_subb_co_u32_e32 v32, vcc, v51, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[2:3], 0, v67, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v67
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v70
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v69
	v_cndmask_b32_e64 v70, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v67
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v67, v71, v70, s[2:3]
	v_add_u32_e32 v70, 2, v66
	v_add_u32_e32 v71, 1, v66
	v_cmp_ne_u32_e64 s[2:3], 0, v67
	v_cndmask_b32_e32 v32, v51, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v67, v71, v70, s[2:3]
	v_xor_b32_e32 v51, s64, v68
	v_cndmask_b32_e32 v32, v66, v67, vcc
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_xor_b32_e32 v32, v32, v51
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_sub_u32_e32 v32, v32, v51
	v_xor_b32_e32 v72, v64, v66
	v_xad_u32 v51, v32, v66, v30
	v_xor_b32_e32 v67, v65, v66
	v_mad_u64_u32 v[64:65], s[2:3], v72, s5, 0
	v_mul_hi_u32 v32, v72, s4
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[70:71], s[2:3], v67, s4, 0
	v_add_co_u32_e32 v32, vcc, v64, v70
	v_mad_u64_u32 v[68:69], s[2:3], v67, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v64, vcc, v65, v71, vcc
	v_mov_b32_e32 v65, s6
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[68:69]
	v_mul_lo_u32 v32, s13, v64
	v_mul_lo_u32 v68, s12, v65
	v_mad_u64_u32 v[64:65], s[2:3], s12, v64, 0
	v_add3_u32 v32, v65, v68, v32
	v_sub_u32_e32 v65, v67, v32
	v_sub_co_u32_e32 v64, vcc, v72, v64
	v_mul_lo_u32 v51, v51, s33
	s_nop 0
	v_subb_co_u32_e64 v65, s[2:3], v65, v25, vcc
	v_subrev_co_u32_e64 v68, s[2:3], s12, v64
	v_subb_co_u32_e32 v32, vcc, v67, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v65, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v69
	v_subb_co_u32_e64 v65, s[2:3], v65, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v68
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v69
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v64
	v_cndmask_b32_e64 v70, v70, v71, s[14:15]
	v_subrev_co_u32_e64 v71, s[2:3], s12, v68
	s_nop 1
	v_subbrev_co_u32_e64 v65, s[2:3], 0, v65, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v70
	s_nop 1
	v_cndmask_b32_e64 v65, v69, v65, s[2:3]
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v67, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_cndmask_b32_e64 v65, v68, v71, s[2:3]
	v_cndmask_b32_e32 v64, v64, v65, vcc
	v_xor_b32_e32 v64, v64, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v64, vcc, v64, v66
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v32, v32, v64, v51
	buffer_load_ubyte v44, v44, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v51, v47, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v47, v53, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v53, v55, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v55, v57, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v59, v59, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v57, v61, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v61, v32, s[28:31], 0 offen
	v_sub_co_u32_e32 v32, vcc, -9, v40
	v_lshl_add_u64 v[64:65], v[40:41], 0, 8
	s_nop 0
	v_subb_co_u32_e32 v66, vcc, -1, v41, vcc
	v_cmp_gt_i32_e32 vcc, 0, v65
	s_nop 1
	v_cndmask_b32_e32 v67, v65, v66, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v32, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v74, v66, v68
	v_xor_b32_e32 v69, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v74, s5, 0
	v_mul_hi_u32 v32, v74, s4
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v69, s4, 0
	v_add_co_u32_e32 v32, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v69, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s6
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v32, s13, v66
	v_mul_lo_u32 v67, s12, v67
	v_mad_u64_u32 v[70:71], s[2:3], s12, v66, 0
	v_add3_u32 v32, v71, v67, v32
	v_sub_u32_e32 v67, v69, v32
	v_sub_co_u32_e32 v70, vcc, v74, v70
	s_nop 1
	v_subb_co_u32_e64 v67, s[2:3], v67, v25, vcc
	v_subrev_co_u32_e64 v71, s[2:3], s12, v70
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[2:3], 0, v67, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v67
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v71
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v70
	v_cndmask_b32_e64 v71, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v67
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v67, v72, v71, s[2:3]
	v_add_u32_e32 v71, 2, v66
	v_add_u32_e32 v72, 1, v66
	v_cmp_ne_u32_e64 s[2:3], 0, v67
	v_cndmask_b32_e32 v32, v69, v70, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v67, v72, v71, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v32, v66, v67, vcc
	v_xor_b32_e32 v66, s64, v68
	v_xor_b32_e32 v32, v32, v66
	v_sub_u32_e32 v32, v32, v66
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_xor_b32_e32 v73, v64, v66
	v_xad_u32 v72, v32, v66, v30
	v_xor_b32_e32 v67, v65, v66
	v_mad_u64_u32 v[64:65], s[2:3], v73, s5, 0
	v_mul_hi_u32 v32, v73, s4
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[70:71], s[2:3], v67, s4, 0
	v_add_co_u32_e32 v32, vcc, v64, v70
	v_mad_u64_u32 v[68:69], s[2:3], v67, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v64, vcc, v65, v71, vcc
	v_mov_b32_e32 v65, s6
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[68:69]
	v_mul_lo_u32 v32, s13, v64
	v_mul_lo_u32 v68, s12, v65
	v_mad_u64_u32 v[64:65], s[2:3], s12, v64, 0
	v_add3_u32 v32, v65, v68, v32
	v_sub_u32_e32 v65, v67, v32
	v_sub_co_u32_e32 v64, vcc, v73, v64
	s_nop 1
	v_subb_co_u32_e64 v65, s[2:3], v65, v25, vcc
	v_subrev_co_u32_e64 v68, s[2:3], s12, v64
	v_subb_co_u32_e32 v32, vcc, v67, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v65, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v69
	v_subb_co_u32_e64 v65, s[2:3], v65, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v68
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v69
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v64
	v_cndmask_b32_e64 v70, v70, v71, s[14:15]
	v_subrev_co_u32_e64 v71, s[2:3], s12, v68
	s_nop 1
	v_subbrev_co_u32_e64 v65, s[2:3], 0, v65, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v70
	s_nop 1
	v_cndmask_b32_e64 v65, v69, v65, s[2:3]
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v67, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_cndmask_b32_e64 v65, v68, v71, s[2:3]
	v_cndmask_b32_e32 v64, v64, v65, vcc
	v_xor_b32_e32 v64, v64, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v64, vcc, v64, v66
	v_mul_lo_u32 v65, v72, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v64, v32, v64, v65
	v_sub_co_u32_e32 v32, vcc, -10, v38
	v_lshl_add_u64 v[66:67], v[38:39], 0, 9
	s_nop 0
	v_subb_co_u32_e32 v65, vcc, -1, v39, vcc
	v_cmp_gt_i32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v69, v67, v65, vcc
	v_ashrrev_i32_e32 v70, 31, v69
	v_cndmask_b32_e32 v68, v66, v32, vcc
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[70:71]
	v_xor_b32_e32 v71, v68, v70
	v_xor_b32_e32 v65, v69, v70
	v_mad_u64_u32 v[68:69], s[2:3], v71, s5, 0
	v_mul_hi_u32 v32, v71, s4
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[68:69]
	v_mad_u64_u32 v[74:75], s[2:3], v65, s4, 0
	v_add_co_u32_e32 v32, vcc, v68, v74
	v_mad_u64_u32 v[72:73], s[2:3], v65, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v69, v75, vcc
	v_mov_b32_e32 v69, s6
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[72:73]
	v_mul_lo_u32 v32, s13, v68
	v_mul_lo_u32 v69, s12, v69
	v_mad_u64_u32 v[72:73], s[2:3], s12, v68, 0
	v_add3_u32 v32, v73, v69, v32
	v_sub_u32_e32 v69, v65, v32
	v_sub_co_u32_e32 v71, vcc, v71, v72
	s_nop 1
	v_subb_co_u32_e64 v69, s[2:3], v69, v25, vcc
	v_subrev_co_u32_e64 v72, s[2:3], s12, v71
	v_subb_co_u32_e32 v32, vcc, v65, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[2:3], 0, v69, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v69
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v73, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v72
	v_cndmask_b32_e64 v65, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v71
	v_cndmask_b32_e64 v72, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v69
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v69, v73, v72, s[2:3]
	v_add_u32_e32 v72, 2, v68
	v_add_u32_e32 v73, 1, v68
	v_cmp_ne_u32_e64 s[2:3], 0, v69
	v_cndmask_b32_e32 v32, v65, v71, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v69, v73, v72, s[2:3]
	v_xor_b32_e32 v65, s64, v70
	v_cndmask_b32_e32 v32, v68, v69, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_mov_b32_e32 v69, v68
	v_xor_b32_e32 v32, v32, v65
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_sub_u32_e32 v32, v32, v65
	v_xor_b32_e32 v74, v66, v68
	v_xad_u32 v65, v32, v68, v30
	v_xor_b32_e32 v69, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v74, s5, 0
	v_mul_hi_u32 v32, v74, s4
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v69, s4, 0
	v_add_co_u32_e32 v32, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v69, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s6
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v32, s13, v66
	v_mul_lo_u32 v70, s12, v67
	v_mad_u64_u32 v[66:67], s[2:3], s12, v66, 0
	v_add3_u32 v32, v67, v70, v32
	v_sub_u32_e32 v67, v69, v32
	v_sub_co_u32_e32 v66, vcc, v74, v66
	v_mul_lo_u32 v65, v65, s33
	s_nop 0
	v_subb_co_u32_e64 v67, s[2:3], v67, v25, vcc
	v_subrev_co_u32_e64 v70, s[2:3], s12, v66
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v71, s[14:15], 0, v67, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v71
	v_subb_co_u32_e64 v67, s[2:3], v67, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v70
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v73, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v71
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v66
	v_cndmask_b32_e64 v72, v72, v73, s[14:15]
	v_subrev_co_u32_e64 v73, s[2:3], s12, v70
	s_nop 1
	v_subbrev_co_u32_e64 v67, s[2:3], 0, v67, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v72
	s_nop 1
	v_cndmask_b32_e64 v67, v71, v67, s[2:3]
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v69, v69, v71, vcc
	v_cmp_ne_u32_e32 vcc, 0, v69
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v67, vcc
	v_cndmask_b32_e64 v67, v70, v73, s[2:3]
	v_cndmask_b32_e32 v66, v66, v67, vcc
	v_xor_b32_e32 v66, v66, v68
	v_xor_b32_e32 v32, v32, v68
	v_sub_co_u32_e32 v66, vcc, v66, v68
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v68, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v65, v32, v66, v65
	v_sub_co_u32_e32 v32, vcc, -11, v40
	v_lshl_add_u64 v[66:67], v[40:41], 0, 10
	s_nop 0
	v_subb_co_u32_e32 v68, vcc, -1, v41, vcc
	v_cmp_gt_i32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v69, v67, v68, vcc
	v_ashrrev_i32_e32 v70, 31, v69
	v_cndmask_b32_e32 v68, v66, v32, vcc
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[70:71]
	v_xor_b32_e32 v76, v68, v70
	v_xor_b32_e32 v71, v69, v70
	v_mad_u64_u32 v[68:69], s[2:3], v76, s5, 0
	v_mul_hi_u32 v32, v76, s4
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[68:69]
	v_mad_u64_u32 v[74:75], s[2:3], v71, s4, 0
	v_add_co_u32_e32 v32, vcc, v68, v74
	v_mad_u64_u32 v[72:73], s[2:3], v71, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v69, v75, vcc
	v_mov_b32_e32 v69, s6
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[72:73]
	v_mul_lo_u32 v32, s13, v68
	v_mul_lo_u32 v69, s12, v69
	v_mad_u64_u32 v[72:73], s[2:3], s12, v68, 0
	v_add3_u32 v32, v73, v69, v32
	v_sub_u32_e32 v69, v71, v32
	v_sub_co_u32_e32 v72, vcc, v76, v72
	s_nop 1
	v_subb_co_u32_e64 v69, s[2:3], v69, v25, vcc
	v_subrev_co_u32_e64 v73, s[2:3], s12, v72
	v_subb_co_u32_e32 v32, vcc, v71, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[2:3], 0, v69, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v69
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v74, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v73
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v72
	v_cndmask_b32_e64 v73, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v69
	v_cndmask_b32_e64 v72, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v69, v74, v73, s[2:3]
	v_add_u32_e32 v73, 2, v68
	v_add_u32_e32 v74, 1, v68
	v_cmp_ne_u32_e64 s[2:3], 0, v69
	v_cndmask_b32_e32 v32, v71, v72, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v69, v74, v73, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v32, v68, v69, vcc
	v_xor_b32_e32 v68, s64, v70
	v_xor_b32_e32 v32, v32, v68
	v_sub_u32_e32 v32, v32, v68
	v_ashrrev_i32_e32 v68, 31, v67
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v75, v66, v68
	v_xad_u32 v74, v32, v68, v30
	v_xor_b32_e32 v69, v67, v68
	v_mad_u64_u32 v[66:67], s[2:3], v75, s5, 0
	v_mul_hi_u32 v32, v75, s4
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[72:73], s[2:3], v69, s4, 0
	v_add_co_u32_e32 v32, vcc, v66, v72
	v_mad_u64_u32 v[70:71], s[2:3], v69, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v73, vcc
	v_mov_b32_e32 v67, s6
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v32, s13, v66
	v_mul_lo_u32 v70, s12, v67
	v_mad_u64_u32 v[66:67], s[2:3], s12, v66, 0
	v_add3_u32 v32, v67, v70, v32
	v_sub_u32_e32 v67, v69, v32
	v_sub_co_u32_e32 v66, vcc, v75, v66
	s_nop 1
	v_subb_co_u32_e64 v67, s[2:3], v67, v25, vcc
	v_subrev_co_u32_e64 v70, s[2:3], s12, v66
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v71, s[14:15], 0, v67, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v71
	v_subb_co_u32_e64 v67, s[2:3], v67, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v70
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v73, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v71
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v66
	v_cndmask_b32_e64 v72, v72, v73, s[14:15]
	v_subrev_co_u32_e64 v73, s[2:3], s12, v70
	s_nop 1
	v_subbrev_co_u32_e64 v67, s[2:3], 0, v67, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v72
	s_nop 1
	v_cndmask_b32_e64 v67, v71, v67, s[2:3]
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v69, v69, v71, vcc
	v_cmp_ne_u32_e32 vcc, 0, v69
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v67, vcc
	v_cndmask_b32_e64 v67, v70, v73, s[2:3]
	v_cndmask_b32_e32 v66, v66, v67, vcc
	v_xor_b32_e32 v66, v66, v68
	v_xor_b32_e32 v32, v32, v68
	v_sub_co_u32_e32 v66, vcc, v66, v68
	v_mul_lo_u32 v67, v74, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v68, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v67, v32, v66, v67
	v_sub_co_u32_e32 v32, vcc, -12, v38
	v_lshl_add_u64 v[68:69], v[38:39], 0, 11
	s_nop 0
	v_subb_co_u32_e32 v66, vcc, -1, v39, vcc
	v_cmp_gt_i32_e32 vcc, 0, v69
	s_nop 1
	v_cndmask_b32_e32 v71, v69, v66, vcc
	v_ashrrev_i32_e32 v72, 31, v71
	v_cndmask_b32_e32 v70, v68, v32, vcc
	v_mov_b32_e32 v73, v72
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[72:73]
	v_xor_b32_e32 v73, v70, v72
	v_xor_b32_e32 v66, v71, v72
	v_mad_u64_u32 v[70:71], s[2:3], v73, s5, 0
	v_mul_hi_u32 v32, v73, s4
	v_lshl_add_u64 v[70:71], v[32:33], 0, v[70:71]
	v_mad_u64_u32 v[76:77], s[2:3], v66, s4, 0
	v_add_co_u32_e32 v32, vcc, v70, v76
	v_mad_u64_u32 v[74:75], s[2:3], v66, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v71, v77, vcc
	v_mov_b32_e32 v71, s6
	s_nop 0
	v_addc_co_u32_e32 v75, vcc, 0, v75, vcc
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[74:75]
	v_mul_lo_u32 v32, s13, v70
	v_mul_lo_u32 v71, s12, v71
	v_mad_u64_u32 v[74:75], s[2:3], s12, v70, 0
	v_add3_u32 v32, v75, v71, v32
	v_sub_u32_e32 v71, v66, v32
	v_sub_co_u32_e32 v73, vcc, v73, v74
	s_nop 1
	v_subb_co_u32_e64 v71, s[2:3], v71, v25, vcc
	v_subrev_co_u32_e64 v74, s[2:3], s12, v73
	v_subb_co_u32_e32 v32, vcc, v66, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v71, s[2:3], 0, v71, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v71
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v75, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v74
	v_cndmask_b32_e64 v66, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v73
	v_cndmask_b32_e64 v74, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v71
	v_cndmask_b32_e64 v73, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v71, v75, v74, s[2:3]
	v_add_u32_e32 v74, 2, v70
	v_add_u32_e32 v75, 1, v70
	v_cmp_ne_u32_e64 s[2:3], 0, v71
	v_cndmask_b32_e32 v32, v66, v73, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v71, v75, v74, s[2:3]
	v_xor_b32_e32 v66, s64, v72
	v_cndmask_b32_e32 v32, v70, v71, vcc
	v_ashrrev_i32_e32 v70, 31, v69
	v_mov_b32_e32 v71, v70
	v_xor_b32_e32 v32, v32, v66
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[70:71]
	v_sub_u32_e32 v32, v32, v66
	v_xor_b32_e32 v76, v68, v70
	v_xad_u32 v66, v32, v70, v30
	v_xor_b32_e32 v71, v69, v70
	v_mad_u64_u32 v[68:69], s[2:3], v76, s5, 0
	v_mul_hi_u32 v32, v76, s4
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[68:69]
	v_mad_u64_u32 v[74:75], s[2:3], v71, s4, 0
	v_add_co_u32_e32 v32, vcc, v68, v74
	v_mad_u64_u32 v[72:73], s[2:3], v71, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v69, v75, vcc
	v_mov_b32_e32 v69, s6
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[72:73]
	v_mul_lo_u32 v32, s13, v68
	v_mul_lo_u32 v72, s12, v69
	v_mad_u64_u32 v[68:69], s[2:3], s12, v68, 0
	v_add3_u32 v32, v69, v72, v32
	v_sub_u32_e32 v69, v71, v32
	v_sub_co_u32_e32 v68, vcc, v76, v68
	v_mul_lo_u32 v66, v66, s33
	s_nop 0
	v_subb_co_u32_e64 v69, s[2:3], v69, v25, vcc
	v_subrev_co_u32_e64 v72, s[2:3], s12, v68
	v_subb_co_u32_e32 v32, vcc, v71, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v73, s[14:15], 0, v69, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v73
	v_subb_co_u32_e64 v69, s[2:3], v69, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v74, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v72
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v75, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v73
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v68
	v_cndmask_b32_e64 v74, v74, v75, s[14:15]
	v_subrev_co_u32_e64 v75, s[2:3], s12, v72
	s_nop 1
	v_subbrev_co_u32_e64 v69, s[2:3], 0, v69, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v74
	s_nop 1
	v_cndmask_b32_e64 v69, v73, v69, s[2:3]
	v_cndmask_b32_e64 v73, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v71, v71, v73, vcc
	v_cmp_ne_u32_e32 vcc, 0, v71
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v69, vcc
	v_cndmask_b32_e64 v69, v72, v75, s[2:3]
	v_cndmask_b32_e32 v68, v68, v69, vcc
	v_xor_b32_e32 v68, v68, v70
	v_xor_b32_e32 v32, v32, v70
	v_sub_co_u32_e32 v68, vcc, v68, v70
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v70, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v68, v32, v68, v66
	v_sub_co_u32_e32 v32, vcc, -13, v40
	v_lshl_add_u64 v[70:71], v[40:41], 0, 12
	s_nop 0
	v_subb_co_u32_e32 v66, vcc, -1, v41, vcc
	v_cmp_gt_i32_e32 vcc, 0, v71
	s_nop 1
	v_cndmask_b32_e32 v73, v71, v66, vcc
	v_ashrrev_i32_e32 v74, 31, v73
	v_cndmask_b32_e32 v72, v70, v32, vcc
	v_mov_b32_e32 v75, v74
	v_lshl_add_u64 v[72:73], v[72:73], 0, v[74:75]
	v_xor_b32_e32 v69, v72, v74
	v_xor_b32_e32 v66, v73, v74
	v_mad_u64_u32 v[72:73], s[2:3], v69, s5, 0
	v_mul_hi_u32 v32, v69, s4
	v_lshl_add_u64 v[72:73], v[32:33], 0, v[72:73]
	v_mad_u64_u32 v[80:81], s[2:3], v66, s4, 0
	v_add_co_u32_e32 v32, vcc, v72, v80
	v_mad_u64_u32 v[76:77], s[2:3], v66, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v72, vcc, v73, v81, vcc
	v_mov_b32_e32 v73, s6
	s_nop 0
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[72:73], v[72:73], 0, v[76:77]
	v_mul_lo_u32 v32, s13, v72
	v_mul_lo_u32 v73, s12, v73
	v_mad_u64_u32 v[76:77], s[2:3], s12, v72, 0
	v_add3_u32 v32, v77, v73, v32
	v_sub_u32_e32 v73, v66, v32
	v_sub_co_u32_e32 v69, vcc, v69, v76
	s_nop 1
	v_subb_co_u32_e64 v73, s[2:3], v73, v25, vcc
	v_subrev_co_u32_e64 v75, s[2:3], s12, v69
	v_subb_co_u32_e32 v32, vcc, v66, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v73, s[2:3], 0, v73, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v73
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v76, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v75
	v_cndmask_b32_e64 v66, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v69
	v_cndmask_b32_e64 v75, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v73
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v73, v76, v75, s[2:3]
	v_add_u32_e32 v75, 2, v72
	v_add_u32_e32 v76, 1, v72
	v_cmp_ne_u32_e64 s[2:3], 0, v73
	v_cndmask_b32_e32 v32, v66, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v73, v76, v75, s[2:3]
	v_xor_b32_e32 v66, s64, v74
	v_cndmask_b32_e32 v32, v72, v73, vcc
	v_ashrrev_i32_e32 v72, 31, v71
	v_mov_b32_e32 v73, v72
	v_xor_b32_e32 v32, v32, v66
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[72:73]
	v_sub_u32_e32 v32, v32, v66
	v_xor_b32_e32 v73, v70, v72
	v_xad_u32 v66, v32, v72, v30
	v_xor_b32_e32 v69, v71, v72
	v_mad_u64_u32 v[70:71], s[2:3], v73, s5, 0
	v_mul_hi_u32 v32, v73, s4
	v_lshl_add_u64 v[70:71], v[32:33], 0, v[70:71]
	v_mad_u64_u32 v[76:77], s[2:3], v69, s4, 0
	v_add_co_u32_e32 v32, vcc, v70, v76
	v_mad_u64_u32 v[74:75], s[2:3], v69, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v71, v77, vcc
	v_mov_b32_e32 v71, s6
	s_nop 0
	v_addc_co_u32_e32 v75, vcc, 0, v75, vcc
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[74:75]
	v_mul_lo_u32 v32, s13, v70
	v_mul_lo_u32 v74, s12, v71
	v_mad_u64_u32 v[70:71], s[2:3], s12, v70, 0
	v_add3_u32 v32, v71, v74, v32
	v_sub_u32_e32 v71, v69, v32
	v_sub_co_u32_e32 v70, vcc, v73, v70
	v_mul_lo_u32 v66, v66, s33
	s_nop 0
	v_subb_co_u32_e64 v71, s[2:3], v71, v25, vcc
	v_subrev_co_u32_e64 v73, s[2:3], s12, v70
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v74, s[14:15], 0, v71, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v74
	v_subb_co_u32_e64 v71, s[2:3], v71, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v75, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v73
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v76, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v74
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v70
	v_cndmask_b32_e64 v75, v75, v76, s[14:15]
	v_subrev_co_u32_e64 v76, s[2:3], s12, v73
	s_nop 1
	v_subbrev_co_u32_e64 v71, s[2:3], 0, v71, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v75
	s_nop 1
	v_cndmask_b32_e64 v71, v74, v71, s[2:3]
	v_cndmask_b32_e64 v74, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v69, v69, v74, vcc
	v_cmp_ne_u32_e32 vcc, 0, v69
	v_cndmask_b32_e64 v69, v73, v76, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v69, v70, v69, vcc
	v_cndmask_b32_e32 v32, v32, v71, vcc
	v_xor_b32_e32 v69, v69, v72
	v_xor_b32_e32 v32, v32, v72
	v_sub_co_u32_e32 v69, vcc, v69, v72
	v_lshl_add_u64 v[70:71], v[38:39], 0, 13
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v72, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v69, v32, v69, v66
	v_sub_co_u32_e32 v32, vcc, -14, v38
	s_nop 1
	v_subb_co_u32_e32 v66, vcc, -1, v39, vcc
	v_cmp_gt_i32_e32 vcc, 0, v71
	s_nop 1
	v_cndmask_b32_e32 v73, v71, v66, vcc
	v_ashrrev_i32_e32 v74, 31, v73
	v_cndmask_b32_e32 v72, v70, v32, vcc
	v_mov_b32_e32 v75, v74
	v_lshl_add_u64 v[72:73], v[72:73], 0, v[74:75]
	v_xor_b32_e32 v75, v72, v74
	v_xor_b32_e32 v66, v73, v74
	v_mad_u64_u32 v[72:73], s[2:3], v75, s5, 0
	v_mul_hi_u32 v32, v75, s4
	v_lshl_add_u64 v[72:73], v[32:33], 0, v[72:73]
	v_mad_u64_u32 v[80:81], s[2:3], v66, s4, 0
	v_add_co_u32_e32 v32, vcc, v72, v80
	v_mad_u64_u32 v[76:77], s[2:3], v66, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v72, vcc, v73, v81, vcc
	v_mov_b32_e32 v73, s6
	s_nop 0
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[72:73], v[72:73], 0, v[76:77]
	v_mul_lo_u32 v32, s13, v72
	v_mul_lo_u32 v73, s12, v73
	v_mad_u64_u32 v[76:77], s[2:3], s12, v72, 0
	v_add3_u32 v32, v77, v73, v32
	v_sub_u32_e32 v73, v66, v32
	v_sub_co_u32_e32 v75, vcc, v75, v76
	s_nop 1
	v_subb_co_u32_e64 v73, s[2:3], v73, v25, vcc
	v_subrev_co_u32_e64 v76, s[2:3], s12, v75
	v_subb_co_u32_e32 v32, vcc, v66, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v73, s[2:3], 0, v73, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v73
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v77, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v76
	v_cndmask_b32_e64 v66, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v75
	v_cndmask_b32_e64 v76, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v73
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v73, v77, v76, s[2:3]
	v_add_u32_e32 v76, 2, v72
	v_add_u32_e32 v77, 1, v72
	v_cmp_ne_u32_e64 s[2:3], 0, v73
	v_cndmask_b32_e32 v32, v66, v75, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v73, v77, v76, s[2:3]
	v_xor_b32_e32 v66, s64, v74
	v_cndmask_b32_e32 v32, v72, v73, vcc
	v_ashrrev_i32_e32 v72, 31, v71
	v_mov_b32_e32 v73, v72
	v_xor_b32_e32 v32, v32, v66
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[72:73]
	v_sub_u32_e32 v32, v32, v66
	v_xor_b32_e32 v80, v70, v72
	v_xad_u32 v66, v32, v72, v30
	v_xor_b32_e32 v73, v71, v72
	v_mad_u64_u32 v[70:71], s[2:3], v80, s5, 0
	v_mul_hi_u32 v32, v80, s4
	v_lshl_add_u64 v[70:71], v[32:33], 0, v[70:71]
	v_mad_u64_u32 v[76:77], s[2:3], v73, s4, 0
	v_add_co_u32_e32 v32, vcc, v70, v76
	v_mad_u64_u32 v[74:75], s[2:3], v73, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v71, v77, vcc
	v_mov_b32_e32 v71, s6
	s_nop 0
	v_addc_co_u32_e32 v75, vcc, 0, v75, vcc
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[74:75]
	v_mul_lo_u32 v32, s13, v70
	v_mul_lo_u32 v74, s12, v71
	v_mad_u64_u32 v[70:71], s[2:3], s12, v70, 0
	v_add3_u32 v32, v71, v74, v32
	v_sub_u32_e32 v71, v73, v32
	v_sub_co_u32_e32 v70, vcc, v80, v70
	v_mul_lo_u32 v66, v66, s33
	s_nop 0
	v_subb_co_u32_e64 v71, s[2:3], v71, v25, vcc
	v_subrev_co_u32_e64 v74, s[2:3], s12, v70
	v_subb_co_u32_e32 v32, vcc, v73, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v75, s[14:15], 0, v71, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v75
	v_subb_co_u32_e64 v71, s[2:3], v71, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v76, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v74
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v77, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v75
	v_cndmask_b32_e64 v73, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v70
	v_cndmask_b32_e64 v76, v76, v77, s[14:15]
	v_subrev_co_u32_e64 v77, s[2:3], s12, v74
	s_nop 1
	v_subbrev_co_u32_e64 v71, s[2:3], 0, v71, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v76
	s_nop 1
	v_cndmask_b32_e64 v71, v75, v71, s[2:3]
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v73, v73, v75, vcc
	v_cmp_ne_u32_e32 vcc, 0, v73
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v71, vcc
	v_cndmask_b32_e64 v71, v74, v77, s[2:3]
	v_cndmask_b32_e32 v70, v70, v71, vcc
	v_xor_b32_e32 v70, v70, v72
	v_xor_b32_e32 v32, v32, v72
	v_sub_co_u32_e32 v70, vcc, v70, v72
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v72, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v80, v32, v70, v66
	v_sub_co_u32_e32 v32, vcc, -15, v40
	v_lshl_add_u64 v[70:71], v[40:41], 0, 14
	s_nop 0
	v_subb_co_u32_e32 v40, vcc, -1, v41, vcc
	v_cmp_gt_i32_e32 vcc, 0, v71
	s_nop 1
	v_cndmask_b32_e32 v41, v71, v40, vcc
	v_ashrrev_i32_e32 v72, 31, v41
	v_cndmask_b32_e32 v40, v70, v32, vcc
	v_mov_b32_e32 v73, v72
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[72:73]
	v_xor_b32_e32 v73, v40, v72
	v_xor_b32_e32 v66, v41, v72
	v_mad_u64_u32 v[40:41], s[2:3], v73, s5, 0
	v_mul_hi_u32 v32, v73, s4
	v_lshl_add_u64 v[40:41], v[32:33], 0, v[40:41]
	v_mad_u64_u32 v[76:77], s[2:3], v66, s4, 0
	v_add_co_u32_e32 v32, vcc, v40, v76
	v_mad_u64_u32 v[74:75], s[2:3], v66, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v40, vcc, v41, v77, vcc
	v_mov_b32_e32 v41, s6
	s_nop 0
	v_addc_co_u32_e32 v75, vcc, 0, v75, vcc
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[74:75]
	v_mul_lo_u32 v32, s13, v40
	v_mul_lo_u32 v41, s12, v41
	v_mad_u64_u32 v[74:75], s[2:3], s12, v40, 0
	v_add3_u32 v32, v75, v41, v32
	v_sub_u32_e32 v41, v66, v32
	v_sub_co_u32_e32 v73, vcc, v73, v74
	s_nop 1
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, vcc
	v_subrev_co_u32_e64 v74, s[2:3], s12, v73
	v_subb_co_u32_e32 v32, vcc, v66, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v41, s[2:3], 0, v41, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v41
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v75, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v74
	v_cndmask_b32_e64 v66, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v73
	v_cndmask_b32_e64 v74, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v41
	v_cndmask_b32_e64 v73, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v41, v75, v74, s[2:3]
	v_add_u32_e32 v74, 2, v40
	v_add_u32_e32 v75, 1, v40
	v_cmp_ne_u32_e64 s[2:3], 0, v41
	v_cndmask_b32_e32 v32, v66, v73, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v41, v75, v74, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v32, v40, v41, vcc
	v_xor_b32_e32 v40, s64, v72
	v_xor_b32_e32 v32, v32, v40
	v_sub_u32_e32 v32, v32, v40
	v_ashrrev_i32_e32 v40, 31, v71
	v_mov_b32_e32 v41, v40
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[40:41]
	v_xor_b32_e32 v76, v70, v40
	v_xad_u32 v66, v32, v40, v30
	v_xor_b32_e32 v41, v71, v40
	v_mad_u64_u32 v[70:71], s[2:3], v76, s5, 0
	v_mul_hi_u32 v32, v76, s4
	v_lshl_add_u64 v[70:71], v[32:33], 0, v[70:71]
	v_mad_u64_u32 v[74:75], s[2:3], v41, s4, 0
	v_add_co_u32_e32 v32, vcc, v70, v74
	v_mad_u64_u32 v[72:73], s[2:3], v41, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v71, v75, vcc
	v_mov_b32_e32 v71, s6
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[72:73]
	v_mul_lo_u32 v32, s13, v70
	v_mul_lo_u32 v72, s12, v71
	v_mad_u64_u32 v[70:71], s[2:3], s12, v70, 0
	v_add3_u32 v32, v71, v72, v32
	v_sub_u32_e32 v71, v41, v32
	v_sub_co_u32_e32 v70, vcc, v76, v70
	s_nop 1
	v_subb_co_u32_e64 v71, s[2:3], v71, v25, vcc
	v_subrev_co_u32_e64 v72, s[2:3], s12, v70
	v_subb_co_u32_e32 v32, vcc, v41, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v73, s[14:15], 0, v71, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v73
	v_subb_co_u32_e64 v71, s[2:3], v71, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v74, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v72
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v75, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v73
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v70
	v_cndmask_b32_e64 v74, v74, v75, s[14:15]
	v_subrev_co_u32_e64 v75, s[2:3], s12, v72
	s_nop 1
	v_subbrev_co_u32_e64 v71, s[2:3], 0, v71, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v74
	s_nop 1
	v_cndmask_b32_e64 v71, v73, v71, s[2:3]
	v_cndmask_b32_e64 v73, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v41, v41, v73, vcc
	v_cmp_ne_u32_e32 vcc, 0, v41
	v_cndmask_b32_e64 v41, v72, v75, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v41, v70, v41, vcc
	v_cndmask_b32_e32 v32, v32, v71, vcc
	v_xor_b32_e32 v41, v41, v40
	v_xor_b32_e32 v32, v32, v40
	v_sub_co_u32_e32 v41, vcc, v41, v40
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v40, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_mul_lo_u32 v40, v66, s33
	v_add3_u32 v76, v32, v41, v40
	v_sub_co_u32_e32 v32, vcc, -16, v38
	v_lshl_add_u64 v[40:41], v[38:39], 0, 15
	s_nop 0
	v_subb_co_u32_e32 v38, vcc, -1, v39, vcc
	v_cmp_gt_i32_e32 vcc, 0, v41
	s_nop 1
	v_cndmask_b32_e32 v39, v41, v38, vcc
	v_ashrrev_i32_e32 v70, 31, v39
	v_cndmask_b32_e32 v38, v40, v32, vcc
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[70:71]
	v_xor_b32_e32 v71, v38, v70
	v_xor_b32_e32 v66, v39, v70
	v_mad_u64_u32 v[38:39], s[2:3], v71, s5, 0
	v_mul_hi_u32 v32, v71, s4
	v_lshl_add_u64 v[38:39], v[32:33], 0, v[38:39]
	v_mad_u64_u32 v[74:75], s[2:3], v66, s4, 0
	v_add_co_u32_e32 v32, vcc, v38, v74
	v_mad_u64_u32 v[72:73], s[2:3], v66, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v75, vcc
	v_mov_b32_e32 v39, s6
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[72:73]
	v_mul_lo_u32 v32, s13, v38
	v_mul_lo_u32 v39, s12, v39
	v_mad_u64_u32 v[72:73], s[2:3], s12, v38, 0
	v_add3_u32 v32, v73, v39, v32
	v_sub_u32_e32 v39, v66, v32
	v_sub_co_u32_e32 v71, vcc, v71, v72
	s_nop 1
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, vcc
	v_subrev_co_u32_e64 v72, s[2:3], s12, v71
	v_subb_co_u32_e32 v32, vcc, v66, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v39, s[2:3], 0, v39, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v39
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v73, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v72
	v_cndmask_b32_e64 v66, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v71
	v_cndmask_b32_e64 v72, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v39
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v39, v73, v72, s[2:3]
	v_add_u32_e32 v72, 2, v38
	v_add_u32_e32 v73, 1, v38
	v_cmp_ne_u32_e64 s[2:3], 0, v39
	v_cndmask_b32_e32 v32, v66, v71, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v39, v73, v72, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v32, v38, v39, vcc
	v_xor_b32_e32 v38, s64, v70
	v_xor_b32_e32 v32, v32, v38
	v_sub_u32_e32 v32, v32, v38
	v_ashrrev_i32_e32 v38, 31, v41
	v_mov_b32_e32 v39, v38
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[38:39]
	v_xor_b32_e32 v74, v40, v38
	v_xad_u32 v66, v32, v38, v30
	v_xor_b32_e32 v39, v41, v38
	v_mad_u64_u32 v[40:41], s[2:3], v74, s5, 0
	v_mul_hi_u32 v32, v74, s4
	v_lshl_add_u64 v[40:41], v[32:33], 0, v[40:41]
	v_mad_u64_u32 v[72:73], s[2:3], v39, s4, 0
	v_add_co_u32_e32 v32, vcc, v40, v72
	v_mad_u64_u32 v[70:71], s[2:3], v39, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v40, vcc, v41, v73, vcc
	v_mov_b32_e32 v41, s6
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[70:71]
	v_mul_lo_u32 v32, s13, v40
	v_mul_lo_u32 v70, s12, v41
	v_mad_u64_u32 v[40:41], s[2:3], s12, v40, 0
	v_add3_u32 v32, v41, v70, v32
	v_sub_u32_e32 v41, v39, v32
	v_sub_co_u32_e32 v40, vcc, v74, v40
	s_nop 1
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, vcc
	v_subrev_co_u32_e64 v70, s[2:3], s12, v40
	v_subb_co_u32_e32 v32, vcc, v39, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v71, s[14:15], 0, v41, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v71
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v70
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v73, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v71
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v40
	v_cndmask_b32_e64 v72, v72, v73, s[14:15]
	v_subrev_co_u32_e64 v73, s[2:3], s12, v70
	s_nop 1
	v_subbrev_co_u32_e64 v41, s[2:3], 0, v41, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v72
	s_nop 1
	v_cndmask_b32_e64 v41, v71, v41, s[2:3]
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v39, v39, v71, vcc
	v_cmp_ne_u32_e32 vcc, 0, v39
	v_cndmask_b32_e64 v39, v70, v73, s[2:3]
	s_movk_i32 s2, 0xfc00
	v_cndmask_b32_e32 v39, v40, v39, vcc
	v_cndmask_b32_e32 v32, v32, v41, vcc
	v_xor_b32_e32 v39, v39, v38
	v_xor_b32_e32 v32, v32, v38
	v_sub_co_u32_e32 v39, vcc, v39, v38
	s_mov_b32 s3, -1
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v38, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_mul_lo_u32 v38, v66, s33
	v_add3_u32 v32, v32, v39, v38
	v_lshl_add_u64 v[38:39], v[36:37], 0, s[2:3]
	s_movk_i32 s2, 0x3ff
	buffer_load_ubyte v64, v64, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v66, v65, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v65, v67, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v67, v68, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v68, v69, s[28:31], 0 offen
	buffer_load_ubyte v70, v80, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v69, v76, s[28:31], 0 offen
	buffer_load_ubyte v71, v32, s[28:31], 0 offen
	v_sub_co_u32_e32 v32, vcc, s2, v36
	s_nop 1
	v_subb_co_u32_e32 v37, vcc, 0, v37, vcc
	v_cmp_gt_i32_e32 vcc, 0, v39
	s_nop 1
	v_cndmask_b32_e32 v37, v39, v37, vcc
	v_ashrrev_i32_e32 v40, 31, v37
	v_cndmask_b32_e32 v36, v38, v32, vcc
	v_mov_b32_e32 v41, v40
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[40:41]
	v_xor_b32_e32 v76, v36, v40
	v_xor_b32_e32 v41, v37, v40
	v_mad_u64_u32 v[36:37], s[2:3], v76, s5, 0
	v_mul_hi_u32 v32, v76, s4
	v_lshl_add_u64 v[36:37], v[32:33], 0, v[36:37]
	v_mad_u64_u32 v[74:75], s[2:3], v41, s4, 0
	v_add_co_u32_e32 v32, vcc, v36, v74
	v_mad_u64_u32 v[72:73], s[2:3], v41, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v36, vcc, v37, v75, vcc
	v_mov_b32_e32 v37, s6
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[72:73]
	v_mul_lo_u32 v32, s13, v36
	v_mul_lo_u32 v37, s12, v37
	v_mad_u64_u32 v[72:73], s[2:3], s12, v36, 0
	v_add3_u32 v32, v73, v37, v32
	v_sub_u32_e32 v37, v41, v32
	v_sub_co_u32_e32 v72, vcc, v76, v72
	s_nop 1
	v_subb_co_u32_e64 v37, s[2:3], v37, v25, vcc
	v_subrev_co_u32_e64 v73, s[2:3], s12, v72
	v_subb_co_u32_e32 v32, vcc, v41, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v37, s[2:3], 0, v37, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v37
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v74, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v73
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v72
	v_cndmask_b32_e64 v73, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v37
	v_cndmask_b32_e64 v72, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v37, v74, v73, s[2:3]
	v_add_u32_e32 v73, 2, v36
	v_add_u32_e32 v74, 1, v36
	v_cmp_ne_u32_e64 s[2:3], 0, v37
	v_cndmask_b32_e32 v32, v41, v72, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v37, v74, v73, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v32, v36, v37, vcc
	v_xor_b32_e32 v36, s64, v40
	v_xor_b32_e32 v32, v32, v36
	v_sub_u32_e32 v32, v32, v36
	v_ashrrev_i32_e32 v36, 31, v39
	v_mov_b32_e32 v37, v36
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[36:37]
	v_xor_b32_e32 v75, v38, v36
	v_xad_u32 v74, v32, v36, v30
	v_xor_b32_e32 v37, v39, v36
	v_mad_u64_u32 v[38:39], s[2:3], v75, s5, 0
	v_mul_hi_u32 v32, v75, s4
	v_lshl_add_u64 v[38:39], v[32:33], 0, v[38:39]
	v_mad_u64_u32 v[72:73], s[2:3], v37, s4, 0
	v_add_co_u32_e32 v32, vcc, v38, v72
	v_mad_u64_u32 v[40:41], s[2:3], v37, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v73, vcc
	v_mov_b32_e32 v39, s6
	s_nop 0
	v_addc_co_u32_e32 v41, vcc, 0, v41, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[40:41]
	v_mul_lo_u32 v32, s13, v38
	v_mul_lo_u32 v40, s12, v39
	v_mad_u64_u32 v[38:39], s[2:3], s12, v38, 0
	v_add3_u32 v32, v39, v40, v32
	v_sub_u32_e32 v39, v37, v32
	v_sub_co_u32_e32 v38, vcc, v75, v38
	s_nop 1
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, vcc
	v_subrev_co_u32_e64 v40, s[2:3], s12, v38
	v_subb_co_u32_e32 v32, vcc, v37, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v41, s[14:15], 0, v39, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v41
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v40
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v73, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v41
	v_cndmask_b32_e64 v37, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v38
	v_cndmask_b32_e64 v72, v72, v73, s[14:15]
	v_subrev_co_u32_e64 v73, s[2:3], s12, v40
	s_nop 1
	v_subbrev_co_u32_e64 v39, s[2:3], 0, v39, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v72
	s_nop 1
	v_cndmask_b32_e64 v39, v41, v39, s[2:3]
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v37, v37, v41, vcc
	v_cmp_ne_u32_e32 vcc, 0, v37
	v_cndmask_b32_e64 v37, v40, v73, s[2:3]
	s_movk_i32 s2, 0xfe00
	v_cndmask_b32_e32 v37, v38, v37, vcc
	v_cndmask_b32_e32 v32, v32, v39, vcc
	v_xor_b32_e32 v37, v37, v36
	v_xor_b32_e32 v32, v32, v36
	v_sub_co_u32_e32 v37, vcc, v37, v36
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v36, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_mul_lo_u32 v36, v74, s33
	v_add3_u32 v38, v32, v37, v36
	v_mov_b32_e32 v32, 0xfffffc00
	v_bitop3_b32 v36, v27, s2, v32 bitop3:0xc8
	v_ashrrev_i32_e32 v37, 31, v36
	v_lshl_add_u64 v[36:37], v[34:35], 0, v[36:37]
	v_sub_co_u32_e32 v27, vcc, -2, v36
	v_lshl_add_u64 v[40:41], v[36:37], 0, 1
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, -1, v37, vcc
	v_cmp_gt_i32_e32 vcc, 0, v41
	s_nop 1
	v_cndmask_b32_e32 v73, v41, v32, vcc
	v_ashrrev_i32_e32 v74, 31, v73
	v_cndmask_b32_e32 v72, v40, v27, vcc
	v_mov_b32_e32 v75, v74
	v_lshl_add_u64 v[72:73], v[72:73], 0, v[74:75]
	v_xor_b32_e32 v39, v72, v74
	v_xor_b32_e32 v27, v73, v74
	v_mad_u64_u32 v[72:73], s[2:3], v39, s5, 0
	v_mul_hi_u32 v32, v39, s4
	v_lshl_add_u64 v[72:73], v[32:33], 0, v[72:73]
	v_mad_u64_u32 v[80:81], s[2:3], v27, s4, 0
	v_add_co_u32_e32 v32, vcc, v72, v80
	v_mad_u64_u32 v[76:77], s[2:3], v27, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v72, vcc, v73, v81, vcc
	v_mov_b32_e32 v73, s6
	s_nop 0
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[72:73], v[72:73], 0, v[76:77]
	v_mul_lo_u32 v32, s13, v72
	v_mul_lo_u32 v73, s12, v73
	v_mad_u64_u32 v[76:77], s[2:3], s12, v72, 0
	v_add3_u32 v32, v77, v73, v32
	v_sub_u32_e32 v73, v27, v32
	v_sub_co_u32_e32 v39, vcc, v39, v76
	s_nop 1
	v_subb_co_u32_e64 v73, s[2:3], v73, v25, vcc
	v_subrev_co_u32_e64 v75, s[2:3], s12, v39
	v_subb_co_u32_e32 v27, vcc, v27, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v73, s[2:3], 0, v73, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v73
	v_cmp_le_u32_e32 vcc, s13, v27
	s_nop 0
	v_cndmask_b32_e64 v76, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v75
	v_cndmask_b32_e64 v32, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v39
	v_cndmask_b32_e64 v75, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v73
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v27
	v_cndmask_b32_e64 v73, v76, v75, s[2:3]
	v_add_u32_e32 v75, 2, v72
	v_add_u32_e32 v76, 1, v72
	v_cmp_ne_u32_e64 s[2:3], 0, v73
	v_cndmask_b32_e32 v27, v32, v39, vcc
	v_cmp_ne_u32_e32 vcc, 0, v27
	v_cndmask_b32_e64 v73, v76, v75, s[2:3]
	v_xor_b32_e32 v32, s64, v74
	v_cndmask_b32_e32 v27, v72, v73, vcc
	v_ashrrev_i32_e32 v72, 31, v41
	v_mov_b32_e32 v73, v72
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[72:73]
	v_xor_b32_e32 v27, v27, v32
	v_xor_b32_e32 v73, v40, v72
	v_sub_u32_e32 v27, v27, v32
	v_xor_b32_e32 v39, v41, v72
	v_mad_u64_u32 v[40:41], s[2:3], v73, s5, 0
	v_mul_hi_u32 v32, v73, s4
	v_lshl_add_u64 v[40:41], v[32:33], 0, v[40:41]
	v_mad_u64_u32 v[76:77], s[2:3], v39, s4, 0
	v_add_co_u32_e32 v32, vcc, v40, v76
	v_mad_u64_u32 v[74:75], s[2:3], v39, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v40, vcc, v41, v77, vcc
	v_mov_b32_e32 v41, s6
	s_nop 0
	v_addc_co_u32_e32 v75, vcc, 0, v75, vcc
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[74:75]
	v_mul_lo_u32 v32, s13, v40
	v_mul_lo_u32 v74, s12, v41
	v_mad_u64_u32 v[40:41], s[2:3], s12, v40, 0
	v_add3_u32 v32, v41, v74, v32
	v_sub_u32_e32 v41, v39, v32
	v_sub_co_u32_e32 v40, vcc, v73, v40
	v_xad_u32 v27, v27, v72, v30
	s_nop 0
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, vcc
	v_subrev_co_u32_e64 v73, s[2:3], s12, v40
	v_subb_co_u32_e32 v32, vcc, v39, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v74, s[14:15], 0, v41, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v74
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v75, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v73
	v_cmp_le_u32_e32 vcc, s13, v32
	v_mul_lo_u32 v27, v27, s33
	v_cndmask_b32_e64 v76, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v74
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v40
	v_cndmask_b32_e64 v75, v75, v76, s[14:15]
	v_subrev_co_u32_e64 v76, s[2:3], s12, v73
	s_nop 1
	v_subbrev_co_u32_e64 v41, s[2:3], 0, v41, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v75
	s_nop 1
	v_cndmask_b32_e64 v41, v74, v41, s[2:3]
	v_cndmask_b32_e64 v74, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v39, v39, v74, vcc
	v_cmp_ne_u32_e32 vcc, 0, v39
	v_cndmask_b32_e64 v39, v73, v76, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v39, v40, v39, vcc
	v_cndmask_b32_e32 v32, v32, v41, vcc
	v_xor_b32_e32 v39, v39, v72
	v_xor_b32_e32 v32, v32, v72
	v_sub_co_u32_e32 v39, vcc, v39, v72
	v_or_b32_e32 v40, 0xfffffc00, v31
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v72, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_ashrrev_i32_e32 v41, 31, v40
	v_and_b32_e32 v32, s42, v32
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[40:41]
	v_add3_u32 v39, v32, v39, v27
	v_sub_co_u32_e32 v27, vcc, -3, v34
	v_lshl_add_u64 v[40:41], v[34:35], 0, 2
	s_nop 0
	v_subb_co_u32_e32 v31, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v41
	s_nop 1
	v_cndmask_b32_e32 v73, v41, v31, vcc
	v_ashrrev_i32_e32 v74, 31, v73
	v_cndmask_b32_e32 v72, v40, v27, vcc
	v_mov_b32_e32 v75, v74
	v_lshl_add_u64 v[72:73], v[72:73], 0, v[74:75]
	v_xor_b32_e32 v31, v72, v74
	v_xor_b32_e32 v27, v73, v74
	v_mad_u64_u32 v[72:73], s[2:3], v31, s5, 0
	v_mul_hi_u32 v32, v31, s4
	v_lshl_add_u64 v[72:73], v[32:33], 0, v[72:73]
	v_mad_u64_u32 v[80:81], s[2:3], v27, s4, 0
	v_add_co_u32_e32 v32, vcc, v72, v80
	v_mad_u64_u32 v[76:77], s[2:3], v27, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v72, vcc, v73, v81, vcc
	v_mov_b32_e32 v73, s6
	s_nop 0
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[72:73], v[72:73], 0, v[76:77]
	v_mul_lo_u32 v32, s13, v72
	v_mul_lo_u32 v73, s12, v73
	v_mad_u64_u32 v[76:77], s[2:3], s12, v72, 0
	v_add3_u32 v32, v77, v73, v32
	v_sub_u32_e32 v73, v27, v32
	v_sub_co_u32_e32 v31, vcc, v31, v76
	s_nop 1
	v_subb_co_u32_e64 v73, s[2:3], v73, v25, vcc
	v_subrev_co_u32_e64 v75, s[2:3], s12, v31
	v_subb_co_u32_e32 v27, vcc, v27, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v73, s[2:3], 0, v73, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v73
	v_cmp_le_u32_e32 vcc, s13, v27
	s_nop 0
	v_cndmask_b32_e64 v76, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v75
	v_cndmask_b32_e64 v32, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v31
	v_cndmask_b32_e64 v75, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v73
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v27
	v_cndmask_b32_e64 v73, v76, v75, s[2:3]
	v_add_u32_e32 v75, 2, v72
	v_add_u32_e32 v76, 1, v72
	v_cmp_ne_u32_e64 s[2:3], 0, v73
	v_cndmask_b32_e32 v27, v32, v31, vcc
	v_cmp_ne_u32_e32 vcc, 0, v27
	v_cndmask_b32_e64 v73, v76, v75, s[2:3]
	v_xor_b32_e32 v31, s64, v74
	v_cndmask_b32_e32 v27, v72, v73, vcc
	v_ashrrev_i32_e32 v72, 31, v41
	v_mov_b32_e32 v73, v72
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[72:73]
	v_xor_b32_e32 v27, v27, v31
	v_xor_b32_e32 v73, v40, v72
	v_sub_u32_e32 v27, v27, v31
	v_xor_b32_e32 v31, v41, v72
	v_mad_u64_u32 v[40:41], s[2:3], v73, s5, 0
	v_mul_hi_u32 v32, v73, s4
	v_lshl_add_u64 v[40:41], v[32:33], 0, v[40:41]
	v_mad_u64_u32 v[76:77], s[2:3], v31, s4, 0
	v_add_co_u32_e32 v32, vcc, v40, v76
	v_mad_u64_u32 v[74:75], s[2:3], v31, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v40, vcc, v41, v77, vcc
	v_mov_b32_e32 v41, s6
	s_nop 0
	v_addc_co_u32_e32 v75, vcc, 0, v75, vcc
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[74:75]
	v_mul_lo_u32 v32, s13, v40
	v_mul_lo_u32 v74, s12, v41
	v_mad_u64_u32 v[40:41], s[2:3], s12, v40, 0
	v_add3_u32 v32, v41, v74, v32
	v_sub_u32_e32 v41, v31, v32
	v_sub_co_u32_e32 v40, vcc, v73, v40
	v_xad_u32 v27, v27, v72, v30
	s_nop 0
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, vcc
	v_subrev_co_u32_e64 v73, s[2:3], s12, v40
	v_subb_co_u32_e32 v31, vcc, v31, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v74, s[14:15], 0, v41, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v74
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v75, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v73
	v_cmp_le_u32_e32 vcc, s13, v31
	v_mul_lo_u32 v27, v27, s33
	v_cndmask_b32_e64 v76, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v74
	v_cndmask_b32_e64 v32, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v40
	v_cndmask_b32_e64 v75, v75, v76, s[14:15]
	v_subrev_co_u32_e64 v76, s[2:3], s12, v73
	s_nop 1
	v_subbrev_co_u32_e64 v41, s[2:3], 0, v41, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v75
	s_nop 1
	v_cndmask_b32_e64 v41, v74, v41, s[2:3]
	v_cndmask_b32_e64 v74, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v31
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v74, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v32, v73, v76, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v32, v40, v32, vcc
	v_cndmask_b32_e32 v31, v31, v41, vcc
	v_xor_b32_e32 v32, v32, v72
	v_xor_b32_e32 v31, v31, v72
	v_sub_co_u32_e32 v32, vcc, v32, v72
	v_lshl_add_u64 v[40:41], v[36:37], 0, 3
	s_nop 0
	v_subb_co_u32_e32 v31, vcc, v31, v72, vcc
	v_ashrrev_i32_e32 v31, 31, v31
	v_and_b32_e32 v31, s42, v31
	v_add3_u32 v31, v31, v32, v27
	v_sub_co_u32_e32 v27, vcc, -4, v36
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, -1, v37, vcc
	v_cmp_gt_i32_e32 vcc, 0, v41
	s_nop 1
	v_cndmask_b32_e32 v73, v41, v32, vcc
	v_ashrrev_i32_e32 v74, 31, v73
	v_cndmask_b32_e32 v72, v40, v27, vcc
	v_mov_b32_e32 v75, v74
	v_lshl_add_u64 v[72:73], v[72:73], 0, v[74:75]
	v_xor_b32_e32 v75, v72, v74
	v_xor_b32_e32 v27, v73, v74
	v_mad_u64_u32 v[72:73], s[2:3], v75, s5, 0
	v_mul_hi_u32 v32, v75, s4
	v_lshl_add_u64 v[72:73], v[32:33], 0, v[72:73]
	v_mad_u64_u32 v[80:81], s[2:3], v27, s4, 0
	v_add_co_u32_e32 v32, vcc, v72, v80
	v_mad_u64_u32 v[76:77], s[2:3], v27, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v72, vcc, v73, v81, vcc
	v_mov_b32_e32 v73, s6
	s_nop 0
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[72:73], v[72:73], 0, v[76:77]
	v_mul_lo_u32 v32, s13, v72
	v_mul_lo_u32 v73, s12, v73
	v_mad_u64_u32 v[76:77], s[2:3], s12, v72, 0
	v_add3_u32 v32, v77, v73, v32
	v_sub_u32_e32 v73, v27, v32
	v_sub_co_u32_e32 v75, vcc, v75, v76
	s_nop 1
	v_subb_co_u32_e64 v73, s[2:3], v73, v25, vcc
	v_subrev_co_u32_e64 v76, s[2:3], s12, v75
	v_subb_co_u32_e32 v27, vcc, v27, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v73, s[2:3], 0, v73, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v73
	v_cmp_le_u32_e32 vcc, s13, v27
	s_nop 0
	v_cndmask_b32_e64 v77, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v76
	v_cndmask_b32_e64 v32, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v75
	v_cndmask_b32_e64 v76, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v73
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v27
	v_cndmask_b32_e64 v73, v77, v76, s[2:3]
	v_add_u32_e32 v76, 2, v72
	v_add_u32_e32 v77, 1, v72
	v_cmp_ne_u32_e64 s[2:3], 0, v73
	v_cndmask_b32_e32 v27, v32, v75, vcc
	v_cmp_ne_u32_e32 vcc, 0, v27
	v_cndmask_b32_e64 v73, v77, v76, s[2:3]
	v_xor_b32_e32 v32, s64, v74
	v_cndmask_b32_e32 v27, v72, v73, vcc
	v_ashrrev_i32_e32 v72, 31, v41
	v_mov_b32_e32 v73, v72
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[72:73]
	v_xor_b32_e32 v27, v27, v32
	v_xor_b32_e32 v80, v40, v72
	v_sub_u32_e32 v27, v27, v32
	v_xor_b32_e32 v73, v41, v72
	v_mad_u64_u32 v[40:41], s[2:3], v80, s5, 0
	v_mul_hi_u32 v32, v80, s4
	v_lshl_add_u64 v[40:41], v[32:33], 0, v[40:41]
	v_mad_u64_u32 v[76:77], s[2:3], v73, s4, 0
	v_add_co_u32_e32 v32, vcc, v40, v76
	v_mad_u64_u32 v[74:75], s[2:3], v73, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v40, vcc, v41, v77, vcc
	v_mov_b32_e32 v41, s6
	s_nop 0
	v_addc_co_u32_e32 v75, vcc, 0, v75, vcc
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[74:75]
	v_mul_lo_u32 v32, s13, v40
	v_mul_lo_u32 v74, s12, v41
	v_mad_u64_u32 v[40:41], s[2:3], s12, v40, 0
	v_add3_u32 v32, v41, v74, v32
	v_sub_u32_e32 v41, v73, v32
	v_sub_co_u32_e32 v40, vcc, v80, v40
	v_xad_u32 v27, v27, v72, v30
	s_nop 0
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, vcc
	v_subrev_co_u32_e64 v74, s[2:3], s12, v40
	v_subb_co_u32_e32 v32, vcc, v73, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v75, s[14:15], 0, v41, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v75
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v76, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v74
	v_cmp_le_u32_e32 vcc, s13, v32
	v_mul_lo_u32 v27, v27, s33
	v_cndmask_b32_e64 v77, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v75
	v_cndmask_b32_e64 v73, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v40
	v_cndmask_b32_e64 v76, v76, v77, s[14:15]
	v_subrev_co_u32_e64 v77, s[2:3], s12, v74
	s_nop 1
	v_subbrev_co_u32_e64 v41, s[2:3], 0, v41, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v76
	s_nop 1
	v_cndmask_b32_e64 v41, v75, v41, s[2:3]
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v73, v73, v75, vcc
	v_cmp_ne_u32_e32 vcc, 0, v73
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v41, vcc
	v_cndmask_b32_e64 v41, v74, v77, s[2:3]
	v_cndmask_b32_e32 v40, v40, v41, vcc
	v_xor_b32_e32 v40, v40, v72
	v_xor_b32_e32 v32, v32, v72
	v_sub_co_u32_e32 v40, vcc, v40, v72
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v72, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v40, v32, v40, v27
	v_sub_co_u32_e32 v27, vcc, -5, v34
	v_lshl_add_u64 v[72:73], v[34:35], 0, 4
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v73
	s_nop 1
	v_cndmask_b32_e32 v75, v73, v32, vcc
	v_ashrrev_i32_e32 v76, 31, v75
	v_cndmask_b32_e32 v74, v72, v27, vcc
	v_mov_b32_e32 v77, v76
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[76:77]
	v_xor_b32_e32 v41, v74, v76
	v_xor_b32_e32 v27, v75, v76
	v_mad_u64_u32 v[74:75], s[2:3], v41, s5, 0
	v_mul_hi_u32 v32, v41, s4
	v_lshl_add_u64 v[74:75], v[32:33], 0, v[74:75]
	v_mad_u64_u32 v[82:83], s[2:3], v27, s4, 0
	v_add_co_u32_e32 v32, vcc, v74, v82
	v_mad_u64_u32 v[80:81], s[2:3], v27, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v74, vcc, v75, v83, vcc
	v_mov_b32_e32 v75, s6
	s_nop 0
	v_addc_co_u32_e32 v81, vcc, 0, v81, vcc
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[80:81]
	v_mul_lo_u32 v32, s13, v74
	v_mul_lo_u32 v75, s12, v75
	v_mad_u64_u32 v[80:81], s[2:3], s12, v74, 0
	v_add3_u32 v32, v81, v75, v32
	v_sub_u32_e32 v75, v27, v32
	v_sub_co_u32_e32 v41, vcc, v41, v80
	s_nop 1
	v_subb_co_u32_e64 v75, s[2:3], v75, v25, vcc
	v_subrev_co_u32_e64 v77, s[2:3], s12, v41
	v_subb_co_u32_e32 v27, vcc, v27, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v75, s[2:3], 0, v75, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v75
	v_cmp_le_u32_e32 vcc, s13, v27
	s_nop 0
	v_cndmask_b32_e64 v80, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v77
	v_cndmask_b32_e64 v32, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v41
	v_cndmask_b32_e64 v77, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v75
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v27
	v_cndmask_b32_e64 v75, v80, v77, s[2:3]
	v_add_u32_e32 v77, 2, v74
	v_add_u32_e32 v80, 1, v74
	v_cmp_ne_u32_e64 s[2:3], 0, v75
	v_cndmask_b32_e32 v27, v32, v41, vcc
	v_cmp_ne_u32_e32 vcc, 0, v27
	v_cndmask_b32_e64 v75, v80, v77, s[2:3]
	v_xor_b32_e32 v32, s64, v76
	v_cndmask_b32_e32 v27, v74, v75, vcc
	v_ashrrev_i32_e32 v74, 31, v73
	v_mov_b32_e32 v75, v74
	v_lshl_add_u64 v[72:73], v[72:73], 0, v[74:75]
	v_xor_b32_e32 v27, v27, v32
	v_xor_b32_e32 v75, v72, v74
	v_sub_u32_e32 v27, v27, v32
	v_xor_b32_e32 v41, v73, v74
	v_mad_u64_u32 v[72:73], s[2:3], v75, s5, 0
	v_mul_hi_u32 v32, v75, s4
	v_lshl_add_u64 v[72:73], v[32:33], 0, v[72:73]
	v_mad_u64_u32 v[80:81], s[2:3], v41, s4, 0
	v_add_co_u32_e32 v32, vcc, v72, v80
	v_mad_u64_u32 v[76:77], s[2:3], v41, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v72, vcc, v73, v81, vcc
	v_mov_b32_e32 v73, s6
	s_nop 0
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[72:73], v[72:73], 0, v[76:77]
	v_mul_lo_u32 v32, s13, v72
	v_mul_lo_u32 v76, s12, v73
	v_mad_u64_u32 v[72:73], s[2:3], s12, v72, 0
	v_add3_u32 v32, v73, v76, v32
	v_sub_u32_e32 v73, v41, v32
	v_sub_co_u32_e32 v72, vcc, v75, v72
	v_xad_u32 v27, v27, v74, v30
	s_nop 0
	v_subb_co_u32_e64 v73, s[2:3], v73, v25, vcc
	v_subrev_co_u32_e64 v75, s[2:3], s12, v72
	v_subb_co_u32_e32 v32, vcc, v41, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v76, s[14:15], 0, v73, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v76
	v_subb_co_u32_e64 v73, s[2:3], v73, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v77, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v75
	v_cmp_le_u32_e32 vcc, s13, v32
	v_mul_lo_u32 v27, v27, s33
	v_cndmask_b32_e64 v80, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v76
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v72
	v_cndmask_b32_e64 v77, v77, v80, s[14:15]
	v_subrev_co_u32_e64 v80, s[2:3], s12, v75
	s_nop 1
	v_subbrev_co_u32_e64 v73, s[2:3], 0, v73, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v77
	s_nop 1
	v_cndmask_b32_e64 v73, v76, v73, s[2:3]
	v_cndmask_b32_e64 v76, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v41, v41, v76, vcc
	v_cmp_ne_u32_e32 vcc, 0, v41
	v_cndmask_b32_e64 v41, v75, v80, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v41, v72, v41, vcc
	v_cndmask_b32_e32 v32, v32, v73, vcc
	v_xor_b32_e32 v41, v41, v74
	v_xor_b32_e32 v32, v32, v74
	v_sub_co_u32_e32 v41, vcc, v41, v74
	v_lshl_add_u64 v[72:73], v[36:37], 0, 5
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v74, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v41, v32, v41, v27
	v_sub_co_u32_e32 v27, vcc, -6, v36
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, -1, v37, vcc
	v_cmp_gt_i32_e32 vcc, 0, v73
	s_nop 1
	v_cndmask_b32_e32 v75, v73, v32, vcc
	v_ashrrev_i32_e32 v76, 31, v75
	v_cndmask_b32_e32 v74, v72, v27, vcc
	v_mov_b32_e32 v77, v76
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[76:77]
	v_xor_b32_e32 v77, v74, v76
	v_xor_b32_e32 v27, v75, v76
	v_mad_u64_u32 v[74:75], s[2:3], v77, s5, 0
	v_mul_hi_u32 v32, v77, s4
	v_lshl_add_u64 v[74:75], v[32:33], 0, v[74:75]
	v_mad_u64_u32 v[82:83], s[2:3], v27, s4, 0
	v_add_co_u32_e32 v32, vcc, v74, v82
	v_mad_u64_u32 v[80:81], s[2:3], v27, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v74, vcc, v75, v83, vcc
	v_mov_b32_e32 v75, s6
	s_nop 0
	v_addc_co_u32_e32 v81, vcc, 0, v81, vcc
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[80:81]
	v_mul_lo_u32 v32, s13, v74
	v_mul_lo_u32 v75, s12, v75
	v_mad_u64_u32 v[80:81], s[2:3], s12, v74, 0
	v_add3_u32 v32, v81, v75, v32
	v_sub_u32_e32 v75, v27, v32
	v_sub_co_u32_e32 v77, vcc, v77, v80
	s_nop 1
	v_subb_co_u32_e64 v75, s[2:3], v75, v25, vcc
	v_subrev_co_u32_e64 v80, s[2:3], s12, v77
	v_subb_co_u32_e32 v27, vcc, v27, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v75, s[2:3], 0, v75, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v75
	v_cmp_le_u32_e32 vcc, s13, v27
	s_nop 0
	v_cndmask_b32_e64 v81, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v80
	v_cndmask_b32_e64 v32, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v77
	v_cndmask_b32_e64 v80, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v75
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v27
	v_cndmask_b32_e64 v75, v81, v80, s[2:3]
	v_add_u32_e32 v80, 2, v74
	v_add_u32_e32 v81, 1, v74
	v_cmp_ne_u32_e64 s[2:3], 0, v75
	v_cndmask_b32_e32 v27, v32, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v27
	v_cndmask_b32_e64 v75, v81, v80, s[2:3]
	v_xor_b32_e32 v32, s64, v76
	v_cndmask_b32_e32 v27, v74, v75, vcc
	v_ashrrev_i32_e32 v74, 31, v73
	v_mov_b32_e32 v75, v74
	v_lshl_add_u64 v[72:73], v[72:73], 0, v[74:75]
	v_xor_b32_e32 v27, v27, v32
	v_xor_b32_e32 v82, v72, v74
	v_sub_u32_e32 v27, v27, v32
	v_xor_b32_e32 v75, v73, v74
	v_mad_u64_u32 v[72:73], s[2:3], v82, s5, 0
	v_mul_hi_u32 v32, v82, s4
	v_lshl_add_u64 v[72:73], v[32:33], 0, v[72:73]
	v_mad_u64_u32 v[80:81], s[2:3], v75, s4, 0
	v_add_co_u32_e32 v32, vcc, v72, v80
	v_mad_u64_u32 v[76:77], s[2:3], v75, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v72, vcc, v73, v81, vcc
	v_mov_b32_e32 v73, s6
	s_nop 0
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[72:73], v[72:73], 0, v[76:77]
	v_mul_lo_u32 v32, s13, v72
	v_mul_lo_u32 v76, s12, v73
	v_mad_u64_u32 v[72:73], s[2:3], s12, v72, 0
	v_add3_u32 v32, v73, v76, v32
	v_sub_u32_e32 v73, v75, v32
	v_sub_co_u32_e32 v72, vcc, v82, v72
	v_xad_u32 v27, v27, v74, v30
	s_nop 0
	v_subb_co_u32_e64 v73, s[2:3], v73, v25, vcc
	v_subrev_co_u32_e64 v76, s[2:3], s12, v72
	v_subb_co_u32_e32 v32, vcc, v75, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v77, s[14:15], 0, v73, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v77
	v_subb_co_u32_e64 v73, s[2:3], v73, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v80, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v76
	v_cmp_le_u32_e32 vcc, s13, v32
	v_mul_lo_u32 v27, v27, s33
	v_cndmask_b32_e64 v81, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v77
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v72
	v_cndmask_b32_e64 v80, v80, v81, s[14:15]
	v_subrev_co_u32_e64 v81, s[2:3], s12, v76
	s_nop 1
	v_subbrev_co_u32_e64 v73, s[2:3], 0, v73, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v80
	s_nop 1
	v_cndmask_b32_e64 v73, v77, v73, s[2:3]
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v75, v75, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v75
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v73, vcc
	v_cndmask_b32_e64 v73, v76, v81, s[2:3]
	v_cndmask_b32_e32 v72, v72, v73, vcc
	v_xor_b32_e32 v72, v72, v74
	v_xor_b32_e32 v32, v32, v74
	v_sub_co_u32_e32 v72, vcc, v72, v74
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v74, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v84, v32, v72, v27
	v_sub_co_u32_e32 v27, vcc, -7, v34
	v_lshl_add_u64 v[72:73], v[34:35], 0, 6
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v73
	s_nop 1
	v_cndmask_b32_e32 v75, v73, v32, vcc
	v_ashrrev_i32_e32 v76, 31, v75
	v_cndmask_b32_e32 v74, v72, v27, vcc
	v_mov_b32_e32 v77, v76
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[76:77]
	v_xor_b32_e32 v77, v74, v76
	v_xor_b32_e32 v27, v75, v76
	v_mad_u64_u32 v[74:75], s[2:3], v77, s5, 0
	v_mul_hi_u32 v32, v77, s4
	v_lshl_add_u64 v[74:75], v[32:33], 0, v[74:75]
	v_mad_u64_u32 v[82:83], s[2:3], v27, s4, 0
	v_add_co_u32_e32 v32, vcc, v74, v82
	v_mad_u64_u32 v[80:81], s[2:3], v27, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v74, vcc, v75, v83, vcc
	v_mov_b32_e32 v75, s6
	s_nop 0
	v_addc_co_u32_e32 v81, vcc, 0, v81, vcc
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[80:81]
	v_mul_lo_u32 v32, s13, v74
	v_mul_lo_u32 v75, s12, v75
	v_mad_u64_u32 v[80:81], s[2:3], s12, v74, 0
	v_add3_u32 v32, v81, v75, v32
	v_sub_u32_e32 v75, v27, v32
	v_sub_co_u32_e32 v77, vcc, v77, v80
	s_nop 1
	v_subb_co_u32_e64 v75, s[2:3], v75, v25, vcc
	v_subrev_co_u32_e64 v80, s[2:3], s12, v77
	v_subb_co_u32_e32 v27, vcc, v27, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v75, s[2:3], 0, v75, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v75
	v_cmp_le_u32_e32 vcc, s13, v27
	s_nop 0
	v_cndmask_b32_e64 v81, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v80
	v_cndmask_b32_e64 v32, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v77
	v_cndmask_b32_e64 v80, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v75
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v27
	v_cndmask_b32_e64 v75, v81, v80, s[2:3]
	v_add_u32_e32 v80, 2, v74
	v_add_u32_e32 v81, 1, v74
	v_cmp_ne_u32_e64 s[2:3], 0, v75
	v_cndmask_b32_e32 v27, v32, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v27
	v_cndmask_b32_e64 v75, v81, v80, s[2:3]
	v_xor_b32_e32 v32, s64, v76
	v_cndmask_b32_e32 v27, v74, v75, vcc
	v_ashrrev_i32_e32 v74, 31, v73
	v_mov_b32_e32 v75, v74
	v_lshl_add_u64 v[72:73], v[72:73], 0, v[74:75]
	v_xor_b32_e32 v27, v27, v32
	v_xor_b32_e32 v82, v72, v74
	v_sub_u32_e32 v27, v27, v32
	v_xor_b32_e32 v75, v73, v74
	v_mad_u64_u32 v[72:73], s[2:3], v82, s5, 0
	v_mul_hi_u32 v32, v82, s4
	v_lshl_add_u64 v[72:73], v[32:33], 0, v[72:73]
	v_mad_u64_u32 v[80:81], s[2:3], v75, s4, 0
	v_add_co_u32_e32 v32, vcc, v72, v80
	v_mad_u64_u32 v[76:77], s[2:3], v75, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v72, vcc, v73, v81, vcc
	v_mov_b32_e32 v73, s6
	s_nop 0
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[72:73], v[72:73], 0, v[76:77]
	v_mul_lo_u32 v32, s13, v72
	v_mul_lo_u32 v76, s12, v73
	v_mad_u64_u32 v[72:73], s[2:3], s12, v72, 0
	v_add3_u32 v32, v73, v76, v32
	v_sub_u32_e32 v73, v75, v32
	v_sub_co_u32_e32 v72, vcc, v82, v72
	v_xad_u32 v27, v27, v74, v30
	s_nop 0
	v_subb_co_u32_e64 v73, s[2:3], v73, v25, vcc
	v_subrev_co_u32_e64 v76, s[2:3], s12, v72
	v_subb_co_u32_e32 v32, vcc, v75, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v77, s[14:15], 0, v73, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v77
	v_subb_co_u32_e64 v73, s[2:3], v73, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v80, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v76
	v_cmp_le_u32_e32 vcc, s13, v32
	v_mul_lo_u32 v27, v27, s33
	v_cndmask_b32_e64 v81, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v77
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v72
	v_cndmask_b32_e64 v80, v80, v81, s[14:15]
	v_subrev_co_u32_e64 v81, s[2:3], s12, v76
	s_nop 1
	v_subbrev_co_u32_e64 v73, s[2:3], 0, v73, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v80
	s_nop 1
	v_cndmask_b32_e64 v73, v77, v73, s[2:3]
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v75, v75, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v75
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v73, vcc
	v_cndmask_b32_e64 v73, v76, v81, s[2:3]
	v_cndmask_b32_e32 v72, v72, v73, vcc
	v_xor_b32_e32 v72, v72, v74
	v_xor_b32_e32 v32, v32, v74
	v_sub_co_u32_e32 v72, vcc, v72, v74
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v74, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v85, v32, v72, v27
	v_sub_co_u32_e32 v27, vcc, -8, v36
	v_lshl_add_u64 v[72:73], v[36:37], 0, 7
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, -1, v37, vcc
	v_cmp_gt_i32_e32 vcc, 0, v73
	s_nop 1
	v_cndmask_b32_e32 v75, v73, v32, vcc
	v_ashrrev_i32_e32 v76, 31, v75
	v_cndmask_b32_e32 v74, v72, v27, vcc
	v_mov_b32_e32 v77, v76
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[76:77]
	v_xor_b32_e32 v77, v74, v76
	v_xor_b32_e32 v27, v75, v76
	v_mad_u64_u32 v[74:75], s[2:3], v77, s5, 0
	v_mul_hi_u32 v32, v77, s4
	v_lshl_add_u64 v[74:75], v[32:33], 0, v[74:75]
	v_mad_u64_u32 v[82:83], s[2:3], v27, s4, 0
	v_add_co_u32_e32 v32, vcc, v74, v82
	v_mad_u64_u32 v[80:81], s[2:3], v27, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v74, vcc, v75, v83, vcc
	v_mov_b32_e32 v75, s6
	s_nop 0
	v_addc_co_u32_e32 v81, vcc, 0, v81, vcc
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[80:81]
	v_mul_lo_u32 v32, s13, v74
	v_mul_lo_u32 v75, s12, v75
	v_mad_u64_u32 v[80:81], s[2:3], s12, v74, 0
	v_add3_u32 v32, v81, v75, v32
	v_sub_u32_e32 v75, v27, v32
	v_sub_co_u32_e32 v77, vcc, v77, v80
	s_nop 1
	v_subb_co_u32_e64 v75, s[2:3], v75, v25, vcc
	v_subrev_co_u32_e64 v80, s[2:3], s12, v77
	v_subb_co_u32_e32 v27, vcc, v27, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v75, s[2:3], 0, v75, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v75
	v_cmp_le_u32_e32 vcc, s13, v27
	s_nop 0
	v_cndmask_b32_e64 v81, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v80
	v_cndmask_b32_e64 v32, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v77
	v_cndmask_b32_e64 v80, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v75
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v27
	v_cndmask_b32_e64 v75, v81, v80, s[2:3]
	v_add_u32_e32 v80, 2, v74
	v_add_u32_e32 v81, 1, v74
	v_cmp_ne_u32_e64 s[2:3], 0, v75
	v_cndmask_b32_e32 v27, v32, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v27
	v_cndmask_b32_e64 v75, v81, v80, s[2:3]
	v_xor_b32_e32 v32, s64, v76
	v_cndmask_b32_e32 v27, v74, v75, vcc
	v_ashrrev_i32_e32 v74, 31, v73
	v_mov_b32_e32 v75, v74
	v_lshl_add_u64 v[72:73], v[72:73], 0, v[74:75]
	v_xor_b32_e32 v27, v27, v32
	v_xor_b32_e32 v82, v72, v74
	v_sub_u32_e32 v27, v27, v32
	v_xor_b32_e32 v75, v73, v74
	v_mad_u64_u32 v[72:73], s[2:3], v82, s5, 0
	v_mul_hi_u32 v32, v82, s4
	v_lshl_add_u64 v[72:73], v[32:33], 0, v[72:73]
	v_mad_u64_u32 v[80:81], s[2:3], v75, s4, 0
	v_add_co_u32_e32 v32, vcc, v72, v80
	v_mad_u64_u32 v[76:77], s[2:3], v75, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v72, vcc, v73, v81, vcc
	v_mov_b32_e32 v73, s6
	s_nop 0
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[72:73], v[72:73], 0, v[76:77]
	v_mul_lo_u32 v32, s13, v72
	v_mul_lo_u32 v76, s12, v73
	v_mad_u64_u32 v[72:73], s[2:3], s12, v72, 0
	v_add3_u32 v32, v73, v76, v32
	v_sub_u32_e32 v73, v75, v32
	v_sub_co_u32_e32 v72, vcc, v82, v72
	v_xad_u32 v27, v27, v74, v30
	s_nop 0
	v_subb_co_u32_e64 v73, s[2:3], v73, v25, vcc
	v_subrev_co_u32_e64 v76, s[2:3], s12, v72
	v_subb_co_u32_e32 v32, vcc, v75, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v77, s[14:15], 0, v73, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v77
	v_subb_co_u32_e64 v73, s[2:3], v73, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v80, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v76
	v_cmp_le_u32_e32 vcc, s13, v32
	v_mul_lo_u32 v27, v27, s33
	v_cndmask_b32_e64 v81, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v77
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v72
	v_cndmask_b32_e64 v80, v80, v81, s[14:15]
	v_subrev_co_u32_e64 v81, s[2:3], s12, v76
	s_nop 1
	v_subbrev_co_u32_e64 v73, s[2:3], 0, v73, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v80
	s_nop 1
	v_cndmask_b32_e64 v73, v77, v73, s[2:3]
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v75, v75, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v75
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v73, vcc
	v_cndmask_b32_e64 v73, v76, v81, s[2:3]
	v_cndmask_b32_e32 v72, v72, v73, vcc
	v_xor_b32_e32 v72, v72, v74
	v_xor_b32_e32 v32, v32, v74
	v_sub_co_u32_e32 v72, vcc, v72, v74
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v74, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v32, v32, v72, v27
	buffer_load_ubyte v27, v38, s[28:31], 0 offen
	buffer_load_ubyte v73, v39, s[28:31], 0 offen
	buffer_load_ubyte v72, v31, s[28:31], 0 offen
	buffer_load_ubyte v74, v40, s[28:31], 0 offen
	buffer_load_ubyte v75, v41, s[28:31], 0 offen
	buffer_load_ubyte v77, v84, s[28:31], 0 offen
	buffer_load_ubyte v76, v85, s[28:31], 0 offen
	buffer_load_ubyte v80, v32, s[28:31], 0 offen
	v_sub_co_u32_e32 v31, vcc, -9, v34
	v_lshl_add_u64 v[38:39], v[34:35], 0, 8
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v39
	s_nop 1
	v_cndmask_b32_e32 v41, v39, v32, vcc
	v_ashrrev_i32_e32 v82, 31, v41
	v_cndmask_b32_e32 v40, v38, v31, vcc
	v_mov_b32_e32 v83, v82
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[82:83]
	v_xor_b32_e32 v81, v40, v82
	v_xor_b32_e32 v31, v41, v82
	v_mad_u64_u32 v[40:41], s[2:3], v81, s5, 0
	v_mul_hi_u32 v32, v81, s4
	v_lshl_add_u64 v[40:41], v[32:33], 0, v[40:41]
	v_mad_u64_u32 v[86:87], s[2:3], v31, s4, 0
	v_add_co_u32_e32 v32, vcc, v40, v86
	v_mad_u64_u32 v[84:85], s[2:3], v31, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v40, vcc, v41, v87, vcc
	v_mov_b32_e32 v41, s6
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[84:85]
	v_mul_lo_u32 v32, s13, v40
	v_mul_lo_u32 v41, s12, v41
	v_mad_u64_u32 v[84:85], s[2:3], s12, v40, 0
	v_add3_u32 v32, v85, v41, v32
	v_sub_u32_e32 v41, v31, v32
	v_sub_co_u32_e32 v81, vcc, v81, v84
	s_nop 1
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, vcc
	v_subrev_co_u32_e64 v83, s[2:3], s12, v81
	v_subb_co_u32_e32 v31, vcc, v31, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v41, s[2:3], 0, v41, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v41
	v_cmp_le_u32_e32 vcc, s13, v31
	s_nop 0
	v_cndmask_b32_e64 v84, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v83
	v_cndmask_b32_e64 v32, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v81
	v_cndmask_b32_e64 v83, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v41
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v31
	v_cndmask_b32_e64 v41, v84, v83, s[2:3]
	v_add_u32_e32 v83, 2, v40
	v_add_u32_e32 v84, 1, v40
	v_cmp_ne_u32_e64 s[2:3], 0, v41
	v_cndmask_b32_e32 v31, v32, v81, vcc
	v_cmp_ne_u32_e32 vcc, 0, v31
	v_cndmask_b32_e64 v41, v84, v83, s[2:3]
	v_xor_b32_e32 v32, s64, v82
	v_cndmask_b32_e32 v31, v40, v41, vcc
	v_ashrrev_i32_e32 v40, 31, v39
	v_mov_b32_e32 v41, v40
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[40:41]
	v_xor_b32_e32 v31, v31, v32
	v_xor_b32_e32 v81, v38, v40
	v_sub_u32_e32 v31, v31, v32
	v_xor_b32_e32 v41, v39, v40
	v_mad_u64_u32 v[38:39], s[2:3], v81, s5, 0
	v_mul_hi_u32 v32, v81, s4
	v_lshl_add_u64 v[38:39], v[32:33], 0, v[38:39]
	v_mad_u64_u32 v[84:85], s[2:3], v41, s4, 0
	v_add_co_u32_e32 v32, vcc, v38, v84
	v_mad_u64_u32 v[82:83], s[2:3], v41, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v85, vcc
	v_mov_b32_e32 v39, s6
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[82:83]
	v_mul_lo_u32 v32, s13, v38
	v_mul_lo_u32 v82, s12, v39
	v_mad_u64_u32 v[38:39], s[2:3], s12, v38, 0
	v_add3_u32 v32, v39, v82, v32
	v_sub_u32_e32 v39, v41, v32
	v_sub_co_u32_e32 v38, vcc, v81, v38
	v_xad_u32 v31, v31, v40, v30
	s_nop 0
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, vcc
	v_subrev_co_u32_e64 v81, s[2:3], s12, v38
	v_subb_co_u32_e32 v32, vcc, v41, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v82, s[14:15], 0, v39, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v82
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v81
	v_cmp_le_u32_e32 vcc, s13, v32
	v_mul_lo_u32 v31, v31, s33
	v_cndmask_b32_e64 v84, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v82
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v38
	v_cndmask_b32_e64 v83, v83, v84, s[14:15]
	v_subrev_co_u32_e64 v84, s[2:3], s12, v81
	s_nop 1
	v_subbrev_co_u32_e64 v39, s[2:3], 0, v39, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v83
	s_nop 1
	v_cndmask_b32_e64 v39, v82, v39, s[2:3]
	v_cndmask_b32_e64 v82, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v41, v41, v82, vcc
	v_cmp_ne_u32_e32 vcc, 0, v41
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v39, vcc
	v_cndmask_b32_e64 v39, v81, v84, s[2:3]
	v_cndmask_b32_e32 v38, v38, v39, vcc
	v_xor_b32_e32 v38, v38, v40
	v_xor_b32_e32 v32, v32, v40
	v_sub_co_u32_e32 v38, vcc, v38, v40
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v40, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v31, v32, v38, v31
	v_sub_co_u32_e32 v32, vcc, -10, v36
	v_lshl_add_u64 v[38:39], v[36:37], 0, 9
	s_nop 0
	v_subb_co_u32_e32 v40, vcc, -1, v37, vcc
	v_cmp_gt_i32_e32 vcc, 0, v39
	s_nop 1
	v_cndmask_b32_e32 v41, v39, v40, vcc
	v_ashrrev_i32_e32 v82, 31, v41
	v_cndmask_b32_e32 v40, v38, v32, vcc
	v_mov_b32_e32 v83, v82
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[82:83]
	v_xor_b32_e32 v83, v40, v82
	v_xor_b32_e32 v81, v41, v82
	v_mad_u64_u32 v[40:41], s[2:3], v83, s5, 0
	v_mul_hi_u32 v32, v83, s4
	v_lshl_add_u64 v[40:41], v[32:33], 0, v[40:41]
	v_mad_u64_u32 v[86:87], s[2:3], v81, s4, 0
	v_add_co_u32_e32 v32, vcc, v40, v86
	v_mad_u64_u32 v[84:85], s[2:3], v81, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v40, vcc, v41, v87, vcc
	v_mov_b32_e32 v41, s6
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[84:85]
	v_mul_lo_u32 v32, s13, v40
	v_mul_lo_u32 v41, s12, v41
	v_mad_u64_u32 v[84:85], s[2:3], s12, v40, 0
	v_add3_u32 v32, v85, v41, v32
	v_sub_u32_e32 v41, v81, v32
	v_sub_co_u32_e32 v83, vcc, v83, v84
	s_nop 1
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, vcc
	v_subrev_co_u32_e64 v84, s[2:3], s12, v83
	v_subb_co_u32_e32 v32, vcc, v81, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v41, s[2:3], 0, v41, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v41
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v84
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v83
	v_cndmask_b32_e64 v84, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v41
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v41, v85, v84, s[2:3]
	v_add_u32_e32 v84, 2, v40
	v_add_u32_e32 v85, 1, v40
	v_cmp_ne_u32_e64 s[2:3], 0, v41
	v_cndmask_b32_e32 v32, v81, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v41, v85, v84, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v32, v40, v41, vcc
	v_xor_b32_e32 v40, s64, v82
	v_xor_b32_e32 v32, v32, v40
	v_sub_u32_e32 v32, v32, v40
	v_ashrrev_i32_e32 v40, 31, v39
	v_mov_b32_e32 v41, v40
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[40:41]
	v_xor_b32_e32 v86, v38, v40
	v_xad_u32 v81, v32, v40, v30
	v_xor_b32_e32 v41, v39, v40
	v_mad_u64_u32 v[38:39], s[2:3], v86, s5, 0
	v_mul_hi_u32 v32, v86, s4
	v_lshl_add_u64 v[38:39], v[32:33], 0, v[38:39]
	v_mad_u64_u32 v[84:85], s[2:3], v41, s4, 0
	v_add_co_u32_e32 v32, vcc, v38, v84
	v_mad_u64_u32 v[82:83], s[2:3], v41, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v85, vcc
	v_mov_b32_e32 v39, s6
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[82:83]
	v_mul_lo_u32 v32, s13, v38
	v_mul_lo_u32 v82, s12, v39
	v_mad_u64_u32 v[38:39], s[2:3], s12, v38, 0
	v_add3_u32 v32, v39, v82, v32
	v_sub_u32_e32 v39, v41, v32
	v_sub_co_u32_e32 v38, vcc, v86, v38
	s_nop 1
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, vcc
	v_subrev_co_u32_e64 v82, s[2:3], s12, v38
	v_subb_co_u32_e32 v32, vcc, v41, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v83, s[14:15], 0, v39, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v83
	v_subb_co_u32_e64 v39, s[2:3], v39, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v84, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v82
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v83
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v38
	v_cndmask_b32_e64 v84, v84, v85, s[14:15]
	v_subrev_co_u32_e64 v85, s[2:3], s12, v82
	s_nop 1
	v_subbrev_co_u32_e64 v39, s[2:3], 0, v39, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v84
	s_nop 1
	v_cndmask_b32_e64 v39, v83, v39, s[2:3]
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v41, v41, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v41
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v39, vcc
	v_cndmask_b32_e64 v39, v82, v85, s[2:3]
	v_cndmask_b32_e32 v38, v38, v39, vcc
	v_xor_b32_e32 v38, v38, v40
	v_xor_b32_e32 v32, v32, v40
	v_sub_co_u32_e32 v38, vcc, v38, v40
	v_mul_lo_u32 v39, v81, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v40, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v38, v32, v38, v39
	v_sub_co_u32_e32 v32, vcc, -11, v34
	v_lshl_add_u64 v[40:41], v[34:35], 0, 10
	s_nop 0
	v_subb_co_u32_e32 v39, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v41
	s_nop 1
	v_cndmask_b32_e32 v83, v41, v39, vcc
	v_ashrrev_i32_e32 v84, 31, v83
	v_cndmask_b32_e32 v82, v40, v32, vcc
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v81, v82, v84
	v_xor_b32_e32 v39, v83, v84
	v_mad_u64_u32 v[82:83], s[2:3], v81, s5, 0
	v_mul_hi_u32 v32, v81, s4
	v_lshl_add_u64 v[82:83], v[32:33], 0, v[82:83]
	v_mad_u64_u32 v[88:89], s[2:3], v39, s4, 0
	v_add_co_u32_e32 v32, vcc, v82, v88
	v_mad_u64_u32 v[86:87], s[2:3], v39, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v89, vcc
	v_mov_b32_e32 v83, s6
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[86:87]
	v_mul_lo_u32 v32, s13, v82
	v_mul_lo_u32 v83, s12, v83
	v_mad_u64_u32 v[86:87], s[2:3], s12, v82, 0
	v_add3_u32 v32, v87, v83, v32
	v_sub_u32_e32 v83, v39, v32
	v_sub_co_u32_e32 v81, vcc, v81, v86
	s_nop 1
	v_subb_co_u32_e64 v83, s[2:3], v83, v25, vcc
	v_subrev_co_u32_e64 v85, s[2:3], s12, v81
	v_subb_co_u32_e32 v32, vcc, v39, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v83, s[2:3], 0, v83, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v83
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v86, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v85
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v81
	v_cndmask_b32_e64 v85, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v83
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v83, v86, v85, s[2:3]
	v_add_u32_e32 v85, 2, v82
	v_add_u32_e32 v86, 1, v82
	v_cmp_ne_u32_e64 s[2:3], 0, v83
	v_cndmask_b32_e32 v32, v39, v81, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v83, v86, v85, s[2:3]
	v_xor_b32_e32 v39, s64, v84
	v_cndmask_b32_e32 v32, v82, v83, vcc
	v_ashrrev_i32_e32 v82, 31, v41
	v_mov_b32_e32 v83, v82
	v_xor_b32_e32 v32, v32, v39
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[82:83]
	v_sub_u32_e32 v32, v32, v39
	v_xor_b32_e32 v83, v40, v82
	v_xad_u32 v39, v32, v82, v30
	v_xor_b32_e32 v81, v41, v82
	v_mad_u64_u32 v[40:41], s[2:3], v83, s5, 0
	v_mul_hi_u32 v32, v83, s4
	v_lshl_add_u64 v[40:41], v[32:33], 0, v[40:41]
	v_mad_u64_u32 v[86:87], s[2:3], v81, s4, 0
	v_add_co_u32_e32 v32, vcc, v40, v86
	v_mad_u64_u32 v[84:85], s[2:3], v81, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v40, vcc, v41, v87, vcc
	v_mov_b32_e32 v41, s6
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[84:85]
	v_mul_lo_u32 v32, s13, v40
	v_mul_lo_u32 v84, s12, v41
	v_mad_u64_u32 v[40:41], s[2:3], s12, v40, 0
	v_add3_u32 v32, v41, v84, v32
	v_sub_u32_e32 v41, v81, v32
	v_sub_co_u32_e32 v40, vcc, v83, v40
	v_mul_lo_u32 v39, v39, s33
	s_nop 0
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, vcc
	v_subrev_co_u32_e64 v83, s[2:3], s12, v40
	v_subb_co_u32_e32 v32, vcc, v81, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v84, s[14:15], 0, v41, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v84
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v83
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v86, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v84
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v40
	v_cndmask_b32_e64 v85, v85, v86, s[14:15]
	v_subrev_co_u32_e64 v86, s[2:3], s12, v83
	s_nop 1
	v_subbrev_co_u32_e64 v41, s[2:3], 0, v41, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v85
	s_nop 1
	v_cndmask_b32_e64 v41, v84, v41, s[2:3]
	v_cndmask_b32_e64 v84, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v81, v81, v84, vcc
	v_cmp_ne_u32_e32 vcc, 0, v81
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v41, vcc
	v_cndmask_b32_e64 v41, v83, v86, s[2:3]
	v_cndmask_b32_e32 v40, v40, v41, vcc
	v_xor_b32_e32 v40, v40, v82
	v_xor_b32_e32 v32, v32, v82
	v_sub_co_u32_e32 v40, vcc, v40, v82
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v82, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v39, v32, v40, v39
	v_sub_co_u32_e32 v32, vcc, -12, v36
	v_lshl_add_u64 v[40:41], v[36:37], 0, 11
	s_nop 0
	v_subb_co_u32_e32 v81, vcc, -1, v37, vcc
	v_cmp_gt_i32_e32 vcc, 0, v41
	s_nop 1
	v_cndmask_b32_e32 v83, v41, v81, vcc
	v_ashrrev_i32_e32 v84, 31, v83
	v_cndmask_b32_e32 v82, v40, v32, vcc
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v85, v82, v84
	v_xor_b32_e32 v81, v83, v84
	v_mad_u64_u32 v[82:83], s[2:3], v85, s5, 0
	v_mul_hi_u32 v32, v85, s4
	v_lshl_add_u64 v[82:83], v[32:33], 0, v[82:83]
	v_mad_u64_u32 v[88:89], s[2:3], v81, s4, 0
	v_add_co_u32_e32 v32, vcc, v82, v88
	v_mad_u64_u32 v[86:87], s[2:3], v81, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v89, vcc
	v_mov_b32_e32 v83, s6
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[86:87]
	v_mul_lo_u32 v32, s13, v82
	v_mul_lo_u32 v83, s12, v83
	v_mad_u64_u32 v[86:87], s[2:3], s12, v82, 0
	v_add3_u32 v32, v87, v83, v32
	v_sub_u32_e32 v83, v81, v32
	v_sub_co_u32_e32 v85, vcc, v85, v86
	s_nop 1
	v_subb_co_u32_e64 v83, s[2:3], v83, v25, vcc
	v_subrev_co_u32_e64 v86, s[2:3], s12, v85
	v_subb_co_u32_e32 v32, vcc, v81, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v83, s[2:3], 0, v83, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v83
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v86
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v85
	v_cndmask_b32_e64 v86, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v83
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v83, v87, v86, s[2:3]
	v_add_u32_e32 v86, 2, v82
	v_add_u32_e32 v87, 1, v82
	v_cmp_ne_u32_e64 s[2:3], 0, v83
	v_cndmask_b32_e32 v32, v81, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v83, v87, v86, s[2:3]
	v_xor_b32_e32 v81, s64, v84
	v_cndmask_b32_e32 v32, v82, v83, vcc
	v_ashrrev_i32_e32 v82, 31, v41
	v_mov_b32_e32 v83, v82
	v_xor_b32_e32 v32, v32, v81
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[82:83]
	v_sub_u32_e32 v32, v32, v81
	v_xor_b32_e32 v88, v40, v82
	v_xad_u32 v81, v32, v82, v30
	v_xor_b32_e32 v83, v41, v82
	v_mad_u64_u32 v[40:41], s[2:3], v88, s5, 0
	v_mul_hi_u32 v32, v88, s4
	v_lshl_add_u64 v[40:41], v[32:33], 0, v[40:41]
	v_mad_u64_u32 v[86:87], s[2:3], v83, s4, 0
	v_add_co_u32_e32 v32, vcc, v40, v86
	v_mad_u64_u32 v[84:85], s[2:3], v83, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v40, vcc, v41, v87, vcc
	v_mov_b32_e32 v41, s6
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[84:85]
	v_mul_lo_u32 v32, s13, v40
	v_mul_lo_u32 v84, s12, v41
	v_mad_u64_u32 v[40:41], s[2:3], s12, v40, 0
	v_add3_u32 v32, v41, v84, v32
	v_sub_u32_e32 v41, v83, v32
	v_sub_co_u32_e32 v40, vcc, v88, v40
	s_nop 1
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, vcc
	v_subrev_co_u32_e64 v84, s[2:3], s12, v40
	v_subb_co_u32_e32 v32, vcc, v83, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v41, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v85
	v_subb_co_u32_e64 v41, s[2:3], v41, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v86, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v84
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v85
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v40
	v_cndmask_b32_e64 v86, v86, v87, s[14:15]
	v_subrev_co_u32_e64 v87, s[2:3], s12, v84
	s_nop 1
	v_subbrev_co_u32_e64 v41, s[2:3], 0, v41, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v86
	s_nop 1
	v_cndmask_b32_e64 v41, v85, v41, s[2:3]
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v41, vcc
	v_cndmask_b32_e64 v41, v84, v87, s[2:3]
	v_cndmask_b32_e32 v40, v40, v41, vcc
	v_xor_b32_e32 v40, v40, v82
	v_xor_b32_e32 v32, v32, v82
	v_sub_co_u32_e32 v40, vcc, v40, v82
	v_mul_lo_u32 v41, v81, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v82, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v40, v32, v40, v41
	v_sub_co_u32_e32 v32, vcc, -13, v34
	v_lshl_add_u64 v[82:83], v[34:35], 0, 12
	s_nop 0
	v_subb_co_u32_e32 v41, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v83
	s_nop 1
	v_cndmask_b32_e32 v85, v83, v41, vcc
	v_ashrrev_i32_e32 v86, 31, v85
	v_cndmask_b32_e32 v84, v82, v32, vcc
	v_mov_b32_e32 v87, v86
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_xor_b32_e32 v81, v84, v86
	v_xor_b32_e32 v41, v85, v86
	v_mad_u64_u32 v[84:85], s[2:3], v81, s5, 0
	v_mul_hi_u32 v32, v81, s4
	v_lshl_add_u64 v[84:85], v[32:33], 0, v[84:85]
	v_mad_u64_u32 v[92:93], s[2:3], v41, s4, 0
	v_add_co_u32_e32 v32, vcc, v84, v92
	v_mad_u64_u32 v[88:89], s[2:3], v41, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v93, vcc
	v_mov_b32_e32 v85, s6
	s_nop 0
	v_addc_co_u32_e32 v89, vcc, 0, v89, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[88:89]
	v_mul_lo_u32 v32, s13, v84
	v_mul_lo_u32 v85, s12, v85
	v_mad_u64_u32 v[88:89], s[2:3], s12, v84, 0
	v_add3_u32 v32, v89, v85, v32
	v_sub_u32_e32 v85, v41, v32
	v_sub_co_u32_e32 v81, vcc, v81, v88
	s_nop 1
	v_subb_co_u32_e64 v85, s[2:3], v85, v25, vcc
	v_subrev_co_u32_e64 v87, s[2:3], s12, v81
	v_subb_co_u32_e32 v32, vcc, v41, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[2:3], 0, v85, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v85
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v88, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v87
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v81
	v_cndmask_b32_e64 v87, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v85
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v85, v88, v87, s[2:3]
	v_add_u32_e32 v87, 2, v84
	v_add_u32_e32 v88, 1, v84
	v_cmp_ne_u32_e64 s[2:3], 0, v85
	v_cndmask_b32_e32 v32, v41, v81, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v85, v88, v87, s[2:3]
	v_xor_b32_e32 v41, s64, v86
	v_cndmask_b32_e32 v32, v84, v85, vcc
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_xor_b32_e32 v32, v32, v41
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_sub_u32_e32 v32, v32, v41
	v_xor_b32_e32 v85, v82, v84
	v_xad_u32 v41, v32, v84, v30
	v_xor_b32_e32 v81, v83, v84
	v_mad_u64_u32 v[82:83], s[2:3], v85, s5, 0
	v_mul_hi_u32 v32, v85, s4
	v_lshl_add_u64 v[82:83], v[32:33], 0, v[82:83]
	v_mad_u64_u32 v[88:89], s[2:3], v81, s4, 0
	v_add_co_u32_e32 v32, vcc, v82, v88
	v_mad_u64_u32 v[86:87], s[2:3], v81, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v89, vcc
	v_mov_b32_e32 v83, s6
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[86:87]
	v_mul_lo_u32 v32, s13, v82
	v_mul_lo_u32 v86, s12, v83
	v_mad_u64_u32 v[82:83], s[2:3], s12, v82, 0
	v_add3_u32 v32, v83, v86, v32
	v_sub_u32_e32 v83, v81, v32
	v_sub_co_u32_e32 v82, vcc, v85, v82
	v_mul_lo_u32 v41, v41, s33
	s_nop 0
	v_subb_co_u32_e64 v83, s[2:3], v83, v25, vcc
	v_subrev_co_u32_e64 v85, s[2:3], s12, v82
	v_subb_co_u32_e32 v32, vcc, v81, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v86, s[14:15], 0, v83, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v86
	v_subb_co_u32_e64 v83, s[2:3], v83, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v85
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v88, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v86
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v82
	v_cndmask_b32_e64 v87, v87, v88, s[14:15]
	v_subrev_co_u32_e64 v88, s[2:3], s12, v85
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[2:3], 0, v83, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v87
	s_nop 1
	v_cndmask_b32_e64 v83, v86, v83, s[2:3]
	v_cndmask_b32_e64 v86, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v81, v81, v86, vcc
	v_cmp_ne_u32_e32 vcc, 0, v81
	v_cndmask_b32_e64 v81, v85, v88, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v81, v82, v81, vcc
	v_cndmask_b32_e32 v32, v32, v83, vcc
	v_xor_b32_e32 v81, v81, v84
	v_xor_b32_e32 v32, v32, v84
	v_sub_co_u32_e32 v81, vcc, v81, v84
	v_lshl_add_u64 v[82:83], v[36:37], 0, 13
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v84, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v41, v32, v81, v41
	v_sub_co_u32_e32 v32, vcc, -14, v36
	s_nop 1
	v_subb_co_u32_e32 v81, vcc, -1, v37, vcc
	v_cmp_gt_i32_e32 vcc, 0, v83
	s_nop 1
	v_cndmask_b32_e32 v85, v83, v81, vcc
	v_ashrrev_i32_e32 v86, 31, v85
	v_cndmask_b32_e32 v84, v82, v32, vcc
	v_mov_b32_e32 v87, v86
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[86:87]
	v_xor_b32_e32 v87, v84, v86
	v_xor_b32_e32 v81, v85, v86
	v_mad_u64_u32 v[84:85], s[2:3], v87, s5, 0
	v_mul_hi_u32 v32, v87, s4
	v_lshl_add_u64 v[84:85], v[32:33], 0, v[84:85]
	v_mad_u64_u32 v[92:93], s[2:3], v81, s4, 0
	v_add_co_u32_e32 v32, vcc, v84, v92
	v_mad_u64_u32 v[88:89], s[2:3], v81, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v93, vcc
	v_mov_b32_e32 v85, s6
	s_nop 0
	v_addc_co_u32_e32 v89, vcc, 0, v89, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[88:89]
	v_mul_lo_u32 v32, s13, v84
	v_mul_lo_u32 v85, s12, v85
	v_mad_u64_u32 v[88:89], s[2:3], s12, v84, 0
	v_add3_u32 v32, v89, v85, v32
	v_sub_u32_e32 v85, v81, v32
	v_sub_co_u32_e32 v87, vcc, v87, v88
	s_nop 1
	v_subb_co_u32_e64 v85, s[2:3], v85, v25, vcc
	v_subrev_co_u32_e64 v88, s[2:3], s12, v87
	v_subb_co_u32_e32 v32, vcc, v81, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[2:3], 0, v85, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v85
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v89, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v88
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v87
	v_cndmask_b32_e64 v88, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v85
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v85, v89, v88, s[2:3]
	v_add_u32_e32 v88, 2, v84
	v_add_u32_e32 v89, 1, v84
	v_cmp_ne_u32_e64 s[2:3], 0, v85
	v_cndmask_b32_e32 v32, v81, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v85, v89, v88, s[2:3]
	v_xor_b32_e32 v81, s64, v86
	v_cndmask_b32_e32 v32, v84, v85, vcc
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_xor_b32_e32 v32, v32, v81
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_sub_u32_e32 v32, v32, v81
	v_xor_b32_e32 v91, v82, v84
	v_xad_u32 v81, v32, v84, v30
	v_xor_b32_e32 v85, v83, v84
	v_mad_u64_u32 v[82:83], s[2:3], v91, s5, 0
	v_mul_hi_u32 v32, v91, s4
	v_lshl_add_u64 v[82:83], v[32:33], 0, v[82:83]
	v_mad_u64_u32 v[88:89], s[2:3], v85, s4, 0
	v_add_co_u32_e32 v32, vcc, v82, v88
	v_mad_u64_u32 v[86:87], s[2:3], v85, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v89, vcc
	v_mov_b32_e32 v83, s6
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[86:87]
	v_mul_lo_u32 v32, s13, v82
	v_mul_lo_u32 v86, s12, v83
	v_mad_u64_u32 v[82:83], s[2:3], s12, v82, 0
	v_add3_u32 v32, v83, v86, v32
	v_sub_u32_e32 v83, v85, v32
	v_sub_co_u32_e32 v82, vcc, v91, v82
	v_mul_lo_u32 v81, v81, s33
	s_nop 0
	v_subb_co_u32_e64 v83, s[2:3], v83, v25, vcc
	v_subrev_co_u32_e64 v86, s[2:3], s12, v82
	v_subb_co_u32_e32 v32, vcc, v85, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v87, s[14:15], 0, v83, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v87
	v_subb_co_u32_e64 v83, s[2:3], v83, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v88, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v86
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v89, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v87
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v82
	v_cndmask_b32_e64 v88, v88, v89, s[14:15]
	v_subrev_co_u32_e64 v89, s[2:3], s12, v86
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[2:3], 0, v83, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v88
	s_nop 1
	v_cndmask_b32_e64 v83, v87, v83, s[2:3]
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v85, v85, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v83, vcc
	v_cndmask_b32_e64 v83, v86, v89, s[2:3]
	v_cndmask_b32_e32 v82, v82, v83, vcc
	v_xor_b32_e32 v82, v82, v84
	v_xor_b32_e32 v32, v32, v84
	v_sub_co_u32_e32 v82, vcc, v82, v84
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v84, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_add3_u32 v91, v32, v82, v81
	v_sub_co_u32_e32 v32, vcc, -15, v34
	v_lshl_add_u64 v[82:83], v[34:35], 0, 14
	s_nop 0
	v_subb_co_u32_e32 v34, vcc, -1, v35, vcc
	v_cmp_gt_i32_e32 vcc, 0, v83
	s_nop 1
	v_cndmask_b32_e32 v35, v83, v34, vcc
	v_ashrrev_i32_e32 v84, 31, v35
	v_cndmask_b32_e32 v34, v82, v32, vcc
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[84:85]
	v_xor_b32_e32 v85, v34, v84
	v_xor_b32_e32 v81, v35, v84
	v_mad_u64_u32 v[34:35], s[2:3], v85, s5, 0
	v_mul_hi_u32 v32, v85, s4
	v_lshl_add_u64 v[34:35], v[32:33], 0, v[34:35]
	v_mad_u64_u32 v[88:89], s[2:3], v81, s4, 0
	v_add_co_u32_e32 v32, vcc, v34, v88
	v_mad_u64_u32 v[86:87], s[2:3], v81, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v34, vcc, v35, v89, vcc
	v_mov_b32_e32 v35, s6
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[86:87]
	v_mul_lo_u32 v32, s13, v34
	v_mul_lo_u32 v35, s12, v35
	v_mad_u64_u32 v[86:87], s[2:3], s12, v34, 0
	v_add3_u32 v32, v87, v35, v32
	v_sub_u32_e32 v35, v81, v32
	v_sub_co_u32_e32 v85, vcc, v85, v86
	s_nop 1
	v_subb_co_u32_e64 v35, s[2:3], v35, v25, vcc
	v_subrev_co_u32_e64 v86, s[2:3], s12, v85
	v_subb_co_u32_e32 v32, vcc, v81, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v35, s[2:3], 0, v35, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v35
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v86
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v85
	v_cndmask_b32_e64 v86, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v35
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v35, v87, v86, s[2:3]
	v_add_u32_e32 v86, 2, v34
	v_add_u32_e32 v87, 1, v34
	v_cmp_ne_u32_e64 s[2:3], 0, v35
	v_cndmask_b32_e32 v32, v81, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v35, v87, v86, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v32, v34, v35, vcc
	v_xor_b32_e32 v34, s64, v84
	v_xor_b32_e32 v32, v32, v34
	v_sub_u32_e32 v32, v32, v34
	v_ashrrev_i32_e32 v34, 31, v83
	v_mov_b32_e32 v35, v34
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[34:35]
	v_xor_b32_e32 v88, v82, v34
	v_xad_u32 v81, v32, v34, v30
	v_xor_b32_e32 v35, v83, v34
	v_mad_u64_u32 v[82:83], s[2:3], v88, s5, 0
	v_mul_hi_u32 v32, v88, s4
	v_lshl_add_u64 v[82:83], v[32:33], 0, v[82:83]
	v_mad_u64_u32 v[86:87], s[2:3], v35, s4, 0
	v_add_co_u32_e32 v32, vcc, v82, v86
	v_mad_u64_u32 v[84:85], s[2:3], v35, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v87, vcc
	v_mov_b32_e32 v83, s6
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_mul_lo_u32 v32, s13, v82
	v_mul_lo_u32 v84, s12, v83
	v_mad_u64_u32 v[82:83], s[2:3], s12, v82, 0
	v_add3_u32 v32, v83, v84, v32
	v_sub_u32_e32 v83, v35, v32
	v_sub_co_u32_e32 v82, vcc, v88, v82
	s_nop 1
	v_subb_co_u32_e64 v83, s[2:3], v83, v25, vcc
	v_subrev_co_u32_e64 v84, s[2:3], s12, v82
	v_subb_co_u32_e32 v32, vcc, v35, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v83, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v85
	v_subb_co_u32_e64 v83, s[2:3], v83, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v86, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v84
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v85
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v82
	v_cndmask_b32_e64 v86, v86, v87, s[14:15]
	v_subrev_co_u32_e64 v87, s[2:3], s12, v84
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[2:3], 0, v83, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v86
	s_nop 1
	v_cndmask_b32_e64 v83, v85, v83, s[2:3]
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v35, v35, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v35
	v_cndmask_b32_e64 v35, v84, v87, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v35, v82, v35, vcc
	v_cndmask_b32_e32 v32, v32, v83, vcc
	v_xor_b32_e32 v35, v35, v34
	v_xor_b32_e32 v32, v32, v34
	v_sub_co_u32_e32 v35, vcc, v35, v34
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v34, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s42, v32
	v_mul_lo_u32 v34, v81, s33
	v_add3_u32 v88, v32, v35, v34
	v_sub_co_u32_e32 v32, vcc, -16, v36
	v_lshl_add_u64 v[34:35], v[36:37], 0, 15
	s_nop 0
	v_subb_co_u32_e32 v36, vcc, -1, v37, vcc
	v_cmp_gt_i32_e32 vcc, 0, v35
	s_nop 1
	v_cndmask_b32_e32 v37, v35, v36, vcc
	v_ashrrev_i32_e32 v82, 31, v37
	v_cndmask_b32_e32 v36, v34, v32, vcc
	v_mov_b32_e32 v83, v82
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[82:83]
	v_xor_b32_e32 v83, v36, v82
	v_xor_b32_e32 v81, v37, v82
	v_mad_u64_u32 v[36:37], s[2:3], v83, s5, 0
	v_mul_hi_u32 v32, v83, s4
	v_lshl_add_u64 v[36:37], v[32:33], 0, v[36:37]
	v_mad_u64_u32 v[86:87], s[2:3], v81, s4, 0
	v_add_co_u32_e32 v32, vcc, v36, v86
	v_mad_u64_u32 v[84:85], s[2:3], v81, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v36, vcc, v37, v87, vcc
	v_mov_b32_e32 v37, s6
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[84:85]
	v_mul_lo_u32 v32, s13, v36
	v_mul_lo_u32 v37, s12, v37
	v_mad_u64_u32 v[84:85], s[2:3], s12, v36, 0
	v_add3_u32 v32, v85, v37, v32
	v_sub_u32_e32 v37, v81, v32
	v_sub_co_u32_e32 v83, vcc, v83, v84
	s_nop 1
	v_subb_co_u32_e64 v37, s[2:3], v37, v25, vcc
	v_subrev_co_u32_e64 v84, s[2:3], s12, v83
	v_subb_co_u32_e32 v32, vcc, v81, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v37, s[2:3], 0, v37, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s13, v37
	v_cmp_le_u32_e32 vcc, s13, v32
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s12, v84
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v83
	v_cndmask_b32_e64 v84, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s13, v37
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	v_cndmask_b32_e64 v37, v85, v84, s[2:3]
	v_add_u32_e32 v84, 2, v36
	v_add_u32_e32 v85, 1, v36
	v_cmp_ne_u32_e64 s[2:3], 0, v37
	v_cndmask_b32_e32 v32, v81, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v37, v85, v84, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v32, v36, v37, vcc
	v_xor_b32_e32 v36, s64, v82
	v_xor_b32_e32 v32, v32, v36
	v_sub_u32_e32 v32, v32, v36
	v_ashrrev_i32_e32 v36, 31, v35
	v_mov_b32_e32 v37, v36
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[36:37]
	v_xor_b32_e32 v81, v34, v36
	v_xad_u32 v30, v32, v36, v30
	v_xor_b32_e32 v37, v35, v36
	v_mad_u64_u32 v[34:35], s[2:3], v81, s5, 0
	v_mul_hi_u32 v32, v81, s4
	v_lshl_add_u64 v[34:35], v[32:33], 0, v[34:35]
	v_mad_u64_u32 v[84:85], s[2:3], v37, s4, 0
	v_add_co_u32_e32 v32, vcc, v34, v84
	v_mad_u64_u32 v[82:83], s[2:3], v37, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v34, vcc, v35, v85, vcc
	v_mov_b32_e32 v35, s6
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[82:83]
	v_mul_lo_u32 v32, s13, v34
	v_mul_lo_u32 v82, s12, v35
	v_mad_u64_u32 v[34:35], s[2:3], s12, v34, 0
	v_add3_u32 v32, v35, v82, v32
	v_sub_u32_e32 v35, v37, v32
	v_sub_co_u32_e32 v34, vcc, v81, v34
	v_mul_lo_u32 v30, v30, s33
	s_nop 0
	v_subb_co_u32_e64 v35, s[2:3], v35, v25, vcc
	v_subrev_co_u32_e64 v81, s[2:3], s12, v34
	v_subb_co_u32_e32 v32, vcc, v37, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v82, s[14:15], 0, v35, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s13, v82
	v_subb_co_u32_e64 v25, s[2:3], v35, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s12, v81
	v_subrev_co_u32_e64 v35, s[2:3], s12, v81
	s_nop 0
	v_cndmask_b32_e64 v84, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s13, v82
	v_subbrev_co_u32_e64 v25, s[2:3], 0, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v83, v83, v84, s[14:15]
	v_cmp_le_u32_e32 vcc, s13, v32
	v_cmp_ne_u32_e64 s[2:3], 0, v83
	s_nop 0
	v_cndmask_b32_e64 v37, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v34
	v_cndmask_b32_e64 v25, v82, v25, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v82, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v32
	s_nop 1
	v_cndmask_b32_e32 v37, v37, v82, vcc
	v_cmp_ne_u32_e32 vcc, 0, v37
	s_nop 1
	v_cndmask_b32_e32 v25, v32, v25, vcc
	v_cndmask_b32_e64 v32, v81, v35, s[2:3]
	v_cndmask_b32_e32 v32, v34, v32, vcc
	v_xor_b32_e32 v32, v32, v36
	v_xor_b32_e32 v25, v25, v36
	v_sub_co_u32_e32 v32, vcc, v32, v36
	s_nop 1
	v_subb_co_u32_e32 v25, vcc, v25, v36, vcc
	v_ashrrev_i32_e32 v25, 31, v25
	v_and_b32_e32 v25, s42, v25
	v_add3_u32 v25, v25, v32, v30
	buffer_load_ubyte v81, v31, s[28:31], 0 offen
	buffer_load_ubyte v83, v38, s[28:31], 0 offen
	buffer_load_ubyte v82, v39, s[28:31], 0 offen
	buffer_load_ubyte v84, v40, s[28:31], 0 offen
	buffer_load_ubyte v85, v41, s[28:31], 0 offen
	buffer_load_ubyte v87, v91, s[28:31], 0 offen
	buffer_load_ubyte v86, v88, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v88, v25, s[28:31], 0 offen
	v_mov_b32_e32 v39, s1
	v_or_b32_e32 v38, s0, v46
	s_movk_i32 s0, 0xff00
	ds_read_b32 v25, v62 offset:24576
	ds_read_b128 v[110:113], v54
	ds_read_b128 v[106:109], v56
	ds_read_b128 v[102:105], v58
	ds_read_b128 v[98:101], v60
	v_lshl_add_u64 v[30:31], v[38:39], 0, v[48:49]
	s_mov_b32 s1, -1
	v_lshl_add_u64 v[34:35], v[30:31], 0, s[0:1]
	s_movk_i32 s0, 0xff
	v_sub_co_u32_e32 v32, vcc, s0, v30
	s_nop 1
	v_subb_co_u32_e32 v36, vcc, 0, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v35
	s_nop 1
	v_cndmask_b32_e32 v41, v35, v36, vcc
	v_cndmask_b32_e32 v40, v34, v32, vcc
	v_or_b32_e32 v32, s71, v41
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	s_cbranch_execz .LBB0_43
	s_add_u32 s2, s70, s74
	s_mov_b32 s75, s74
	s_addc_u32 s3, s71, s74
	s_xor_b64 s[4:5], s[2:3], s[74:75]
	v_cvt_f32_u32_e32 v32, s4
	v_cvt_f32_u32_e32 v34, s5
	s_sub_u32 s2, 0, s4
	s_subb_u32 s3, 0, s5
	v_ashrrev_i32_e32 v36, 31, v41
	v_fmac_f32_e32 v32, 0x4f800000, v34
	v_rcp_f32_e32 v32, v32
	v_mov_b32_e32 v37, v36
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[36:37]
	v_xor_b32_e32 v37, v40, v36
	v_mul_f32_e32 v32, 0x5f7ffffc, v32
	v_mul_f32_e32 v34, 0x2f800000, v32
	v_trunc_f32_e32 v34, v34
	v_fmac_f32_e32 v32, 0xcf800000, v34
	v_cvt_u32_f32_e32 v34, v34
	v_cvt_u32_f32_e32 v32, v32
	v_readfirstlane_b32 s7, v34
	v_readfirstlane_b32 s12, v32
	s_mul_i32 s13, s2, s7
	s_mul_hi_u32 s15, s2, s12
	s_mul_i32 s14, s3, s12
	s_add_i32 s13, s15, s13
	s_add_i32 s13, s13, s14
	s_mul_i32 s16, s2, s12
	s_mul_i32 s15, s12, s13
	s_mul_hi_u32 s17, s12, s16
	s_mul_hi_u32 s14, s12, s13
	s_add_u32 s15, s17, s15
	s_addc_u32 s14, 0, s14
	s_mul_hi_u32 s18, s7, s16
	s_mul_i32 s16, s7, s16
	s_add_u32 s15, s15, s16
	s_mul_hi_u32 s17, s7, s13
	s_addc_u32 s14, s14, s18
	s_addc_u32 s15, s17, 0
	s_mul_i32 s13, s7, s13
	s_add_u32 s13, s14, s13
	s_addc_u32 s14, 0, s15
	s_add_u32 s12, s12, s13
	s_addc_u32 s7, s7, s14
	s_mul_i32 s13, s2, s7
	s_mul_hi_u32 s14, s2, s12
	s_add_i32 s13, s14, s13
	s_mul_i32 s3, s3, s12
	s_add_i32 s13, s13, s3
	s_mul_i32 s2, s2, s12
	s_mul_hi_u32 s14, s7, s2
	s_mul_i32 s15, s7, s2
	s_mul_i32 s17, s12, s13
	s_mul_hi_u32 s2, s12, s2
	s_mul_hi_u32 s16, s12, s13
	s_add_u32 s2, s2, s17
	s_addc_u32 s16, 0, s16
	s_add_u32 s2, s2, s15
	s_mul_hi_u32 s3, s7, s13
	s_addc_u32 s2, s16, s14
	s_addc_u32 s3, s3, 0
	s_mul_i32 s13, s7, s13
	s_add_u32 s2, s2, s13
	s_addc_u32 s3, 0, s3
	s_add_u32 s12, s12, s2
	s_addc_u32 s7, s7, s3
	v_xor_b32_e32 v34, v41, v36
	v_mad_u64_u32 v[40:41], s[2:3], v37, s7, 0
	v_mul_hi_u32 v32, v37, s12
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[40:41]
	v_mad_u64_u32 v[48:49], s[2:3], v34, s12, 0
	v_add_co_u32_e32 v32, vcc, v32, v48
	v_mad_u64_u32 v[40:41], s[2:3], v34, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v33, v49, vcc
	v_mov_b32_e32 v33, s6
	s_nop 0
	v_addc_co_u32_e32 v41, vcc, 0, v41, vcc
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[40:41]
	v_mul_lo_u32 v46, s5, v32
	v_mul_lo_u32 v48, s4, v33
	v_mad_u64_u32 v[40:41], s[2:3], s4, v32, 0
	v_add3_u32 v46, v41, v48, v46
	v_sub_u32_e32 v41, v34, v46
	v_mov_b32_e32 v48, s5
	v_sub_co_u32_e32 v37, vcc, v37, v40
	s_nop 1
	v_subb_co_u32_e64 v40, s[2:3], v41, v48, vcc
	v_subrev_co_u32_e64 v41, s[2:3], s4, v37
	s_nop 1
	v_subbrev_co_u32_e64 v40, s[2:3], 0, v40, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s5, v40
	s_nop 1
	v_cndmask_b32_e64 v48, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s4, v41
	s_nop 1
	v_cndmask_b32_e64 v41, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s5, v40
	s_nop 1
	v_cndmask_b32_e64 v89, v48, v41, s[2:3]
	v_lshl_add_u64 v[40:41], v[32:33], 0, 2
	v_lshl_add_u64 v[48:49], v[32:33], 0, 1
	v_subb_co_u32_e32 v33, vcc, v34, v46, vcc
	v_cmp_le_u32_e32 vcc, s5, v33
	s_nop 1
	v_cndmask_b32_e64 v34, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v37
	s_nop 1
	v_cndmask_b32_e64 v37, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v33
	s_nop 1
	v_cndmask_b32_e32 v33, v34, v37, vcc
	v_cmp_ne_u32_e32 vcc, 0, v89
	s_nop 1
	v_cndmask_b32_e32 v34, v48, v40, vcc
	v_cmp_ne_u32_e32 vcc, 0, v33
	v_xor_b32_e32 v33, s74, v36
	s_nop 0
	v_cndmask_b32_e32 v32, v32, v34, vcc
	v_xor_b32_e32 v32, v32, v33
	v_sub_co_u32_e32 v36, vcc, v32, v33
.LBB0_43:
	s_or_saveexec_b64 s[0:1], s[0:1]
	v_readlane_b32 s60, v144, 22
	v_readlane_b32 s64, v144, 29
	v_readlane_b32 s61, v144, 23
	v_readlane_b32 s65, v144, 30
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_45
	v_rcp_iflag_f32_e32 v32, v96
	s_sub_i32 s2, 0, s70
	v_mul_f32_e32 v32, 0x4f7ffffe, v32
	v_cvt_u32_f32_e32 v32, v32
	v_mul_lo_u32 v33, s2, v32
	v_mul_hi_u32 v33, v32, v33
	v_add_u32_e32 v32, v32, v33
	v_mul_hi_u32 v32, v40, v32
	v_mul_lo_u32 v33, v32, s70
	v_sub_u32_e32 v33, v40, v33
	v_add_u32_e32 v34, 1, v32
	v_subrev_u32_e32 v36, s70, v33
	v_cmp_le_u32_e32 vcc, s70, v33
	s_nop 1
	v_cndmask_b32_e32 v33, v33, v36, vcc
	v_cndmask_b32_e32 v32, v32, v34, vcc
	v_add_u32_e32 v34, 1, v32
	v_cmp_le_u32_e32 vcc, s70, v33
	s_nop 1
	v_cndmask_b32_e32 v36, v32, v34, vcc
.LBB0_45:
	s_or_b64 exec, exec, s[0:1]
	s_movk_i32 s0, 0xff00
	v_lshl_add_u64 v[32:33], v[38:39], 0, v[42:43]
	s_mov_b32 s1, -1
	v_lshl_add_u64 v[38:39], v[32:33], 0, s[0:1]
	v_or_b32_e32 v34, s71, v39
	s_mov_b32 s2, 0
	v_cmp_ne_u32_e32 vcc, 0, v34
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	s_cbranch_execz .LBB0_47
	s_add_u32 s4, s70, s74
	s_mov_b32 s75, s74
	s_addc_u32 s5, s71, s74
	s_xor_b64 s[4:5], s[4:5], s[74:75]
	v_cvt_f32_u32_e32 v34, s4
	v_cvt_f32_u32_e32 v37, s5
	s_sub_u32 s3, 0, s4
	s_subb_u32 s6, 0, s5
	v_ashrrev_i32_e32 v42, 31, v39
	v_fmamk_f32 v34, v37, 0x4f800000, v34
	v_rcp_f32_e32 v34, v34
	v_mov_b32_e32 v43, v42
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[42:43]
	v_mov_b32_e32 v41, 0
	v_mul_f32_e32 v34, 0x5f7ffffc, v34
	v_mul_f32_e32 v37, 0x2f800000, v34
	v_trunc_f32_e32 v37, v37
	v_fmamk_f32 v34, v37, 0xcf800000, v34
	v_cvt_u32_f32_e32 v37, v37
	v_cvt_u32_f32_e32 v34, v34
	v_readfirstlane_b32 s7, v37
	v_readfirstlane_b32 s12, v34
	s_mul_i32 s13, s3, s7
	s_mul_hi_u32 s15, s3, s12
	s_mul_i32 s14, s6, s12
	s_add_i32 s13, s15, s13
	s_add_i32 s13, s13, s14
	s_mul_i32 s16, s3, s12
	s_mul_i32 s15, s12, s13
	s_mul_hi_u32 s17, s12, s16
	s_mul_hi_u32 s14, s12, s13
	s_add_u32 s15, s17, s15
	s_addc_u32 s14, 0, s14
	s_mul_hi_u32 s18, s7, s16
	s_mul_i32 s16, s7, s16
	s_add_u32 s15, s15, s16
	s_mul_hi_u32 s17, s7, s13
	s_addc_u32 s14, s14, s18
	s_addc_u32 s15, s17, 0
	s_mul_i32 s13, s7, s13
	s_add_u32 s13, s14, s13
	s_addc_u32 s14, 0, s15
	s_add_u32 s12, s12, s13
	s_addc_u32 s7, s7, s14
	s_mul_i32 s13, s3, s7
	s_mul_hi_u32 s14, s3, s12
	s_add_i32 s13, s14, s13
	s_mul_i32 s6, s6, s12
	s_add_i32 s13, s13, s6
	s_mul_i32 s3, s3, s12
	s_mul_hi_u32 s14, s7, s3
	s_mul_i32 s15, s7, s3
	s_mul_i32 s17, s12, s13
	s_mul_hi_u32 s3, s12, s3
	s_mul_hi_u32 s16, s12, s13
	s_add_u32 s3, s3, s17
	s_addc_u32 s16, 0, s16
	s_add_u32 s3, s3, s15
	s_mul_hi_u32 s6, s7, s13
	s_addc_u32 s3, s16, s14
	s_addc_u32 s6, s6, 0
	s_mul_i32 s13, s7, s13
	s_add_u32 s3, s3, s13
	s_addc_u32 s6, 0, s6
	s_add_u32 s3, s12, s3
	s_addc_u32 s12, s7, s6
	v_xor_b32_e32 v37, v38, v42
	v_xor_b32_e32 v34, v39, v42
	v_mad_u64_u32 v[38:39], s[6:7], v37, s12, 0
	v_mul_hi_u32 v40, v37, s3
	v_lshl_add_u64 v[38:39], v[40:41], 0, v[38:39]
	v_mad_u64_u32 v[48:49], s[6:7], v34, s3, 0
	v_add_co_u32_e32 v38, vcc, v38, v48
	v_mad_u64_u32 v[40:41], s[6:7], v34, s12, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v39, v49, vcc
	v_mov_b32_e32 v39, s2
	s_nop 0
	v_addc_co_u32_e32 v41, vcc, 0, v41, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[40:41]
	v_mul_lo_u32 v40, s5, v38
	v_mul_lo_u32 v41, s4, v39
	v_mad_u64_u32 v[38:39], s[2:3], s4, v38, 0
	v_add3_u32 v39, v39, v41, v40
	v_sub_u32_e32 v40, v34, v39
	v_mov_b32_e32 v41, s5
	v_sub_co_u32_e32 v37, vcc, v37, v38
	s_nop 1
	v_subb_co_u32_e64 v38, s[2:3], v40, v41, vcc
	v_subrev_co_u32_e64 v40, s[2:3], s4, v37
	v_subb_co_u32_e32 v34, vcc, v34, v39, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[14:15], 0, v38, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s5, v43
	v_subb_co_u32_e64 v38, s[2:3], v38, v41, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v46, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s4, v40
	v_subrev_co_u32_e64 v41, s[2:3], s4, v40
	s_nop 0
	v_cndmask_b32_e64 v48, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s5, v43
	v_subbrev_co_u32_e64 v38, s[2:3], 0, v38, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v46, v46, v48, s[14:15]
	v_cmp_le_u32_e32 vcc, s5, v34
	v_cmp_ne_u32_e64 s[2:3], 0, v46
	s_nop 0
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v37
	v_cndmask_b32_e64 v38, v43, v38, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v34
	s_nop 1
	v_cndmask_b32_e32 v39, v39, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v39
	s_nop 1
	v_cndmask_b32_e32 v34, v34, v38, vcc
	v_cndmask_b32_e64 v38, v40, v41, s[2:3]
	v_cndmask_b32_e32 v37, v37, v38, vcc
	v_xor_b32_e32 v37, v37, v42
	v_xor_b32_e32 v34, v34, v42
	v_sub_co_u32_e32 v40, vcc, v37, v42
	s_nop 1
	v_subb_co_u32_e32 v41, vcc, v34, v42, vcc
.LBB0_47:
	s_or_saveexec_b64 s[0:1], s[0:1]
	v_readlane_b32 s6, v144, 27
	v_readlane_b32 s7, v144, 28
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_49
	v_rcp_iflag_f32_e32 v34, v96
	s_sub_i32 s2, 0, s70
	v_mov_b32_e32 v41, 0
	v_mul_f32_e32 v34, 0x4f7ffffe, v34
	v_cvt_u32_f32_e32 v34, v34
	v_mul_lo_u32 v37, s2, v34
	v_mul_hi_u32 v37, v34, v37
	v_add_u32_e32 v34, v34, v37
	v_mul_hi_u32 v34, v38, v34
	v_mul_lo_u32 v34, v34, s70
	v_sub_u32_e32 v34, v38, v34
	v_subrev_u32_e32 v37, s70, v34
	v_cmp_le_u32_e32 vcc, s70, v34
	s_nop 1
	v_cndmask_b32_e32 v34, v34, v37, vcc
	v_subrev_u32_e32 v37, s70, v34
	v_cmp_le_u32_e32 vcc, s70, v34
	s_nop 1
	v_cndmask_b32_e32 v40, v34, v37, vcc
.LBB0_49:
	s_or_b64 exec, exec, s[0:1]
	s_load_dword s0, s[96:97], 0x58
	v_ashrrev_i32_e32 v34, 31, v35
	v_xor_b32_e32 v34, v36, v34
	v_ashrrev_i32_e32 v35, 31, v41
	v_and_b32_e32 v35, s70, v35
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v34, v34, s0
	v_add3_u32 v34, v34, v40, v35
	buffer_load_ubyte v38, v34, s[8:11], 0 offen
	s_movk_i32 s0, 0xff02
	s_mov_b32 s1, -1
	v_lshl_add_u64 v[34:35], v[30:31], 0, s[0:1]
	s_movk_i32 s0, 0xfd
	v_sub_co_u32_e32 v30, vcc, s0, v30
	s_mov_b32 s2, 0
	s_nop 0
	v_subb_co_u32_e32 v31, vcc, 0, v31, vcc
	v_cmp_gt_i32_e32 vcc, 0, v35
	s_nop 1
	v_cndmask_b32_e32 v37, v35, v31, vcc
	v_cndmask_b32_e32 v36, v34, v30, vcc
	v_or_b32_e32 v30, s71, v37
	v_cmp_ne_u32_e32 vcc, 0, v30
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	s_cbranch_execz .LBB0_51
	s_add_u32 s4, s70, s74
	s_mov_b32 s75, s74
	s_addc_u32 s5, s71, s74
	s_xor_b64 s[4:5], s[4:5], s[74:75]
	v_cvt_f32_u32_e32 v30, s4
	v_cvt_f32_u32_e32 v31, s5
	s_sub_u32 s3, 0, s4
	s_subb_u32 s6, 0, s5
	v_mov_b32_e32 v41, 0
	v_fmamk_f32 v30, v31, 0x4f800000, v30
	v_rcp_f32_e32 v30, v30
	s_nop 0
	v_mul_f32_e32 v30, 0x5f7ffffc, v30
	v_mul_f32_e32 v31, 0x2f800000, v30
	v_trunc_f32_e32 v31, v31
	v_fmamk_f32 v30, v31, 0xcf800000, v30
	v_cvt_u32_f32_e32 v31, v31
	v_cvt_u32_f32_e32 v30, v30
	v_readfirstlane_b32 s7, v31
	v_readfirstlane_b32 s12, v30
	s_mul_i32 s13, s3, s7
	s_mul_hi_u32 s15, s3, s12
	s_mul_i32 s14, s6, s12
	s_add_i32 s13, s15, s13
	s_add_i32 s13, s13, s14
	s_mul_i32 s16, s3, s12
	s_mul_i32 s15, s12, s13
	s_mul_hi_u32 s17, s12, s16
	s_mul_hi_u32 s14, s12, s13
	s_add_u32 s15, s17, s15
	s_addc_u32 s14, 0, s14
	s_mul_hi_u32 s18, s7, s16
	s_mul_i32 s16, s7, s16
	s_add_u32 s15, s15, s16
	s_mul_hi_u32 s17, s7, s13
	s_addc_u32 s14, s14, s18
	s_addc_u32 s15, s17, 0
	s_mul_i32 s13, s7, s13
	s_add_u32 s13, s14, s13
	s_addc_u32 s14, 0, s15
	s_add_u32 s12, s12, s13
	s_addc_u32 s7, s7, s14
	s_mul_i32 s13, s3, s7
	s_mul_hi_u32 s14, s3, s12
	s_add_i32 s13, s14, s13
	s_mul_i32 s6, s6, s12
	s_add_i32 s13, s13, s6
	s_mul_i32 s3, s3, s12
	s_mul_hi_u32 s14, s7, s3
	s_mul_i32 s15, s7, s3
	s_mul_i32 s17, s12, s13
	s_mul_hi_u32 s3, s12, s3
	s_mul_hi_u32 s16, s12, s13
	s_add_u32 s3, s3, s17
	s_addc_u32 s16, 0, s16
	s_add_u32 s3, s3, s15
	s_mul_hi_u32 s6, s7, s13
	s_addc_u32 s3, s16, s14
	s_addc_u32 s6, s6, 0
	s_mul_i32 s13, s7, s13
	s_add_u32 s3, s3, s13
	v_ashrrev_i32_e32 v30, 31, v37
	s_addc_u32 s6, 0, s6
	v_mov_b32_e32 v31, v30
	s_add_u32 s3, s12, s3
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[30:31]
	s_addc_u32 s12, s7, s6
	v_xor_b32_e32 v34, v36, v30
	v_xor_b32_e32 v31, v37, v30
	v_mad_u64_u32 v[36:37], s[6:7], v34, s12, 0
	v_mul_hi_u32 v40, v34, s3
	v_lshl_add_u64 v[36:37], v[40:41], 0, v[36:37]
	v_mad_u64_u32 v[42:43], s[6:7], v31, s3, 0
	v_add_co_u32_e32 v36, vcc, v36, v42
	v_mad_u64_u32 v[40:41], s[6:7], v31, s12, 0
	s_nop 0
	v_addc_co_u32_e32 v36, vcc, v37, v43, vcc
	v_mov_b32_e32 v37, s2
	s_nop 0
	v_addc_co_u32_e32 v41, vcc, 0, v41, vcc
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[40:41]
	v_mul_lo_u32 v39, s5, v36
	v_mul_lo_u32 v42, s4, v37
	v_mad_u64_u32 v[40:41], s[2:3], s4, v36, 0
	v_add3_u32 v39, v41, v42, v39
	v_sub_u32_e32 v41, v31, v39
	v_mov_b32_e32 v42, s5
	v_sub_co_u32_e32 v34, vcc, v34, v40
	v_xor_b32_e32 v30, s74, v30
	s_nop 0
	v_subb_co_u32_e64 v40, s[2:3], v41, v42, vcc
	v_subrev_co_u32_e64 v41, s[2:3], s4, v34
	v_subb_co_u32_e32 v31, vcc, v31, v39, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v40, s[2:3], 0, v40, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s5, v40
	v_cmp_le_u32_e32 vcc, s5, v31
	v_readlane_b32 s6, v144, 27
	v_cndmask_b32_e64 v42, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s4, v41
	v_readlane_b32 s7, v144, 28
	s_nop 0
	v_cndmask_b32_e64 v41, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s5, v40
	s_nop 1
	v_cndmask_b32_e64 v46, v42, v41, s[2:3]
	v_lshl_add_u64 v[40:41], v[36:37], 0, 2
	v_lshl_add_u64 v[42:43], v[36:37], 0, 1
	v_cndmask_b32_e64 v37, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v34
	s_nop 1
	v_cndmask_b32_e64 v34, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v31
	s_nop 1
	v_cndmask_b32_e32 v31, v37, v34, vcc
	v_cmp_ne_u32_e32 vcc, 0, v46
	s_nop 1
	v_cndmask_b32_e32 v34, v42, v40, vcc
	v_cmp_ne_u32_e32 vcc, 0, v31
	s_nop 1
	v_cndmask_b32_e32 v31, v36, v34, vcc
	v_xor_b32_e32 v31, v31, v30
	v_sub_co_u32_e32 v30, vcc, v31, v30
.LBB0_51:
	s_andn2_saveexec_b64 s[0:1], s[0:1]
	s_cbranch_execz .LBB0_53
	v_rcp_iflag_f32_e32 v30, v96
	s_sub_i32 s2, 0, s70
	v_mul_f32_e32 v30, 0x4f7ffffe, v30
	v_cvt_u32_f32_e32 v30, v30
	v_mul_lo_u32 v31, s2, v30
	v_mul_hi_u32 v31, v30, v31
	v_add_u32_e32 v30, v30, v31
	v_mul_hi_u32 v30, v36, v30
	v_mul_lo_u32 v31, v30, s70
	v_sub_u32_e32 v31, v36, v31
	v_add_u32_e32 v34, 1, v30
	v_subrev_u32_e32 v36, s70, v31
	v_cmp_le_u32_e32 vcc, s70, v31
	s_nop 1
	v_cndmask_b32_e32 v31, v31, v36, vcc
	v_cndmask_b32_e32 v30, v30, v34, vcc
	v_add_u32_e32 v34, 1, v30
	v_cmp_le_u32_e32 vcc, s70, v31
	s_nop 1
	v_cndmask_b32_e32 v30, v30, v34, vcc
.LBB0_53:
	s_or_b64 exec, exec, s[0:1]
	s_movk_i32 s0, 0xff02
	s_mov_b32 s1, -1
	v_lshl_add_u64 v[32:33], v[32:33], 0, s[0:1]
	v_or_b32_e32 v31, s71, v33
	s_mov_b32 s2, 0
	v_cmp_ne_u32_e32 vcc, 0, v31
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	s_cbranch_execz .LBB0_55
	s_add_u32 s4, s70, s74
	s_mov_b32 s75, s74
	s_addc_u32 s5, s71, s74
	s_xor_b64 s[4:5], s[4:5], s[74:75]
	v_cvt_f32_u32_e32 v31, s4
	v_cvt_f32_u32_e32 v34, s5
	s_sub_u32 s3, 0, s4
	s_subb_u32 s6, 0, s5
	v_ashrrev_i32_e32 v40, 31, v33
	v_fmamk_f32 v31, v34, 0x4f800000, v31
	v_rcp_f32_e32 v31, v31
	v_mov_b32_e32 v41, v40
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[40:41]
	v_mov_b32_e32 v37, 0
	v_mul_f32_e32 v31, 0x5f7ffffc, v31
	v_mul_f32_e32 v34, 0x2f800000, v31
	v_trunc_f32_e32 v34, v34
	v_fmamk_f32 v31, v34, 0xcf800000, v31
	v_cvt_u32_f32_e32 v34, v34
	v_cvt_u32_f32_e32 v31, v31
	v_readfirstlane_b32 s7, v34
	v_readfirstlane_b32 s12, v31
	s_mul_i32 s13, s3, s7
	s_mul_hi_u32 s15, s3, s12
	s_mul_i32 s14, s6, s12
	s_add_i32 s13, s15, s13
	s_add_i32 s13, s13, s14
	s_mul_i32 s16, s3, s12
	s_mul_i32 s15, s12, s13
	s_mul_hi_u32 s17, s12, s16
	s_mul_hi_u32 s14, s12, s13
	s_add_u32 s15, s17, s15
	s_addc_u32 s14, 0, s14
	s_mul_hi_u32 s18, s7, s16
	s_mul_i32 s16, s7, s16
	s_add_u32 s15, s15, s16
	s_mul_hi_u32 s17, s7, s13
	s_addc_u32 s14, s14, s18
	s_addc_u32 s15, s17, 0
	s_mul_i32 s13, s7, s13
	s_add_u32 s13, s14, s13
	s_addc_u32 s14, 0, s15
	s_add_u32 s12, s12, s13
	s_addc_u32 s7, s7, s14
	s_mul_i32 s13, s3, s7
	s_mul_hi_u32 s14, s3, s12
	s_add_i32 s13, s14, s13
	s_mul_i32 s6, s6, s12
	s_add_i32 s13, s13, s6
	s_mul_i32 s3, s3, s12
	s_mul_hi_u32 s14, s7, s3
	s_mul_i32 s15, s7, s3
	s_mul_i32 s17, s12, s13
	s_mul_hi_u32 s3, s12, s3
	s_mul_hi_u32 s16, s12, s13
	s_add_u32 s3, s3, s17
	s_addc_u32 s16, 0, s16
	s_add_u32 s3, s3, s15
	s_mul_hi_u32 s6, s7, s13
	s_addc_u32 s3, s16, s14
	s_addc_u32 s6, s6, 0
	s_mul_i32 s13, s7, s13
	s_add_u32 s3, s3, s13
	s_addc_u32 s6, 0, s6
	s_add_u32 s3, s12, s3
	s_addc_u32 s12, s7, s6
	v_xor_b32_e32 v34, v32, v40
	v_xor_b32_e32 v31, v33, v40
	v_mad_u64_u32 v[32:33], s[6:7], v34, s12, 0
	v_mul_hi_u32 v36, v34, s3
	v_lshl_add_u64 v[32:33], v[36:37], 0, v[32:33]
	v_mad_u64_u32 v[42:43], s[6:7], v31, s3, 0
	v_add_co_u32_e32 v32, vcc, v32, v42
	v_mad_u64_u32 v[36:37], s[6:7], v31, s12, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v33, v43, vcc
	v_mov_b32_e32 v33, s2
	s_nop 0
	v_addc_co_u32_e32 v37, vcc, 0, v37, vcc
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[36:37]
	v_mul_lo_u32 v36, s5, v32
	v_mul_lo_u32 v37, s4, v33
	v_mad_u64_u32 v[32:33], s[2:3], s4, v32, 0
	v_add3_u32 v33, v33, v37, v36
	v_sub_u32_e32 v36, v31, v33
	v_mov_b32_e32 v37, s5
	v_sub_co_u32_e32 v32, vcc, v34, v32
	v_readlane_b32 s6, v144, 27
	s_nop 0
	v_subb_co_u32_e64 v34, s[2:3], v36, v37, vcc
	v_subrev_co_u32_e64 v36, s[2:3], s4, v32
	v_subb_co_u32_e32 v31, vcc, v31, v33, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v39, s[14:15], 0, v34, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s5, v39
	v_subb_co_u32_e64 v34, s[2:3], v34, v37, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v41, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s4, v36
	v_subrev_co_u32_e64 v37, s[2:3], s4, v36
	s_nop 0
	v_cndmask_b32_e64 v42, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s5, v39
	v_subbrev_co_u32_e64 v34, s[2:3], 0, v34, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v41, v41, v42, s[14:15]
	v_cmp_le_u32_e32 vcc, s5, v31
	v_cmp_ne_u32_e64 s[2:3], 0, v41
	v_readlane_b32 s7, v144, 28
	v_cndmask_b32_e64 v33, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s4, v32
	v_cndmask_b32_e64 v34, v39, v34, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s5, v31
	s_nop 1
	v_cndmask_b32_e32 v33, v33, v39, vcc
	v_cmp_ne_u32_e32 vcc, 0, v33
	v_cndmask_b32_e64 v33, v36, v37, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v32, v32, v33, vcc
	v_cndmask_b32_e32 v31, v31, v34, vcc
	v_xor_b32_e32 v32, v32, v40
	v_xor_b32_e32 v31, v31, v40
	v_sub_co_u32_e32 v36, vcc, v32, v40
	s_nop 1
	v_subb_co_u32_e32 v37, vcc, v31, v40, vcc
.LBB0_55:
	s_andn2_saveexec_b64 s[0:1], s[0:1]
	s_cbranch_execz .LBB0_57
	v_rcp_iflag_f32_e32 v31, v96
	s_sub_i32 s2, 0, s70
	v_mov_b32_e32 v37, 0
	v_mul_f32_e32 v31, 0x4f7ffffe, v31
	v_cvt_u32_f32_e32 v31, v31
	v_mul_lo_u32 v33, s2, v31
	v_mul_hi_u32 v33, v31, v33
	v_add_u32_e32 v31, v31, v33
	v_mul_hi_u32 v31, v32, v31
	v_mul_lo_u32 v31, v31, s70
	v_sub_u32_e32 v31, v32, v31
	v_subrev_u32_e32 v32, s70, v31
	v_cmp_le_u32_e32 vcc, s70, v31
	s_nop 1
	v_cndmask_b32_e32 v31, v31, v32, vcc
	v_subrev_u32_e32 v32, s70, v31
	v_cmp_le_u32_e32 vcc, s70, v31
	s_nop 1
	v_cndmask_b32_e32 v36, v31, v32, vcc
.LBB0_57:
	s_or_b64 exec, exec, s[0:1]
	s_load_dword s0, s[96:97], 0x58
	v_ashrrev_i32_e32 v31, 31, v35
	v_xor_b32_e32 v30, v30, v31
	v_ashrrev_i32_e32 v31, 31, v37
	v_and_b32_e32 v31, s70, v31
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v30, v30, s0
	v_add3_u32 v30, v30, v36, v31
	buffer_load_ubyte v39, v30, s[8:11], 0 offen
	s_waitcnt vmcnt(20)
	v_lshlrev_b16_e32 v30, 8, v70
	s_waitcnt vmcnt(18)
	v_lshlrev_b16_e32 v31, 8, v71
	v_or_b32_e32 v30, v68, v30
	v_or_b32_sdwa v31, v69, v31 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v34, 8, v53
	v_or_b32_sdwa v33, v30, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v30, 8, v66
	v_lshlrev_b16_e32 v31, 8, v67
	v_or_b32_e32 v30, v64, v30
	v_or_b32_sdwa v31, v65, v31 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v34, v47, v34 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v32, v30, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v30, 8, v59
	v_lshlrev_b16_e32 v31, 8, v61
	v_or_b32_e32 v30, v55, v30
	v_or_b32_sdwa v31, v57, v31 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt vmcnt(2)
	v_lshlrev_b16_e32 v35, 8, v88
	v_or_b32_sdwa v31, v30, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v30, 8, v51
	v_or_b32_e32 v30, v44, v30
	v_or_b32_sdwa v30, v30, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v34, 8, v87
	v_or_b32_e32 v34, v85, v34
	v_or_b32_sdwa v35, v86, v35 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_or_b32_sdwa v37, v34, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v34, 8, v83
	v_lshlrev_b16_e32 v35, 8, v84
	v_or_b32_e32 v34, v81, v34
	v_or_b32_sdwa v35, v82, v35 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_barrier
	v_or_b32_sdwa v36, v34, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v34, 8, v77
	v_lshlrev_b16_e32 v35, 8, v80
	v_or_b32_e32 v34, v75, v34
	v_or_b32_sdwa v35, v76, v35 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_nop 0
	v_or_b32_sdwa v35, v34, v35 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v34, 8, v73
	v_or_b32_e32 v27, v27, v34
	v_lshlrev_b16_e32 v34, 8, v74
	v_or_b32_sdwa v34, v72, v34 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_nop 0
	v_or_b32_sdwa v34, v27, v34 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_and_b32_e32 v27, 0xff, v90
	v_and_b32_e32 v44, 0xff, v79
	ds_read_b32 v46, v97 offset:25088
	ds_read_b128 v[64:67], v52 offset:8192
	ds_read_b128 v[68:71], v50 offset:8192
	ds_read_b128 v[72:75], v52 offset:10240
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[134:137], v[16:19], v[12:15], v78, v27 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[48:51], v50 offset:10240
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[126:129], v[16:19], v[0:3], v78, v27 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[110:113], v[16:19], v[8:11], v25, v27 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[130:133], v[4:7], v[12:15], v78, v44 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[122:125], v[4:7], v[0:3], v78, v44 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b32 v47, v62 offset:25088
	ds_read_b128 v[52:55], v54 offset:8192
	ds_read_b128 v[76:79], v56 offset:8192
	ds_read_b128 v[56:59], v58 offset:8192
	ds_read_b128 v[80:83], v60 offset:8192
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[106:109], v[4:7], v[8:11], v25, v44 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[102:105], v[16:19], v[20:23], v25, v27 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[98:101], v[4:7], v[8:11], v25, v44 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 1
	v_and_b32_e32 v20, 0xffff, v38
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[72:75], v[30:33], v[0:3], v46, v20 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[64:67], v[30:33], v[12:15], v46, v20 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[48:51], v[34:37], v[0:3], v46, v39 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[52:55], v[30:33], v[40:43], v47, v20 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[68:71], v[34:37], v[4:7], v46, v39 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[76:79], v[34:37], v[0:3], v47, v39 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[56:59], v[30:33], v[16:19], v47, v20 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[80:83], v[34:37], v[0:3], v47, v39 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
.LBB0_58:
	s_and_b64 s[0:1], s[6:7], exec
	v_readlane_b32 s0, v144, 10
	v_readlane_b32 s2, v144, 12
	v_readlane_b32 s1, v144, 11
	v_readlane_b32 s3, v144, 13
	s_cselect_b32 s1, s1, s3
	s_cselect_b32 s0, s0, s2
	s_ashr_i32 s2, s1, 31
	s_lshr_b32 s2, s2, 26
	s_add_u32 s0, s0, s2
	s_addc_u32 s1, s1, 0
	s_ashr_i64 s[0:1], s[0:1], 6
	s_sub_u32 s2, 0, s0
	s_subb_u32 s3, 0, s1
	s_add_u32 s4, s0, 1
	s_addc_u32 s5, s1, 0
	s_and_b64 s[0:1], s[6:7], exec
	s_cselect_b32 s0, s2, s4
	s_cselect_b32 s1, s3, s5
	s_and_b32 s22, s0, 31
	v_readlane_b32 s12, v144, 25
	s_max_u32 s26, s22, 1
	v_readlane_b32 s13, v144, 26
	s_and_b64 s[2:3], s[12:13], exec
	v_readlane_b32 s2, v144, 14
	v_readlane_b32 s4, v144, 16
	v_readlane_b32 s3, v144, 15
	v_readlane_b32 s5, v144, 17
	s_cselect_b32 s3, s3, s5
	s_cselect_b32 s2, s2, s4
	s_ashr_i32 s4, s3, 31
	s_lshr_b32 s4, s4, 26
	s_add_u32 s2, s2, s4
	s_addc_u32 s3, s3, 0
	s_ashr_i64 s[2:3], s[2:3], 6
	s_sub_u32 s4, 0, s2
	s_subb_u32 s5, 0, s3
	s_add_u32 s6, s2, 1
	s_addc_u32 s7, s3, 0
	s_and_b64 s[2:3], s[12:13], exec
	s_cselect_b32 s7, s5, s7
	s_cselect_b32 s6, s4, s6
	v_readlane_b32 s4, v144, 0
	s_mul_i32 s2, s7, s4
	s_mul_hi_u32 s3, s6, s4
	s_add_i32 s5, s3, s2
	s_mul_i32 s3, s6, s4
	v_readlane_b32 s2, v144, 2
	s_add_u32 s2, s3, s2
	v_writelane_b32 v144, s3, 12
	s_addc_u32 s3, s5, 0
	s_ashr_i32 s4, s1, 31
	v_writelane_b32 v144, s5, 14
	s_mov_b32 s5, s4
	s_xor_b64 s[0:1], s[4:5], s[0:1]
	s_ashr_i32 s12, s1, 31
	s_lshr_b32 s12, s12, 27
	s_add_u32 s0, s0, s12
	s_addc_u32 s1, s1, 0
	s_ashr_i64 s[0:1], s[0:1], 5
	s_xor_b64 s[14:15], s[0:1], s[4:5]
	s_mul_i32 s0, s14, s7
	s_mul_hi_u32 s1, s14, s6
	s_add_i32 s0, s1, s0
	s_mul_i32 s1, s15, s6
	s_add_i32 s1, s0, s1
	s_mul_i32 s0, s14, s6
	v_writelane_b32 v144, s0, 10
	s_load_dwordx2 s[16:17], s[96:97], 0x60
	s_nop 0
	v_writelane_b32 v144, s1, 11
	s_lshl_b64 s[0:1], s[0:1], 5
	s_sub_u32 s4, 0, s0
	s_subb_u32 s5, 0, s1
	s_sub_u32 s0, s2, s0
	s_subb_u32 s1, s3, s1
	s_cmp_lg_u32 s1, 0
	s_cbranch_scc0 .LBB0_60
	s_add_u32 s12, s26, 0
	s_addc_u32 s13, 0, 0
	s_xor_b64 s[12:13], s[12:13], 0
	v_cvt_f32_u32_e32 v16, s12
	v_cvt_f32_u32_e32 v17, s13
	s_sub_u32 s23, 0, s12
	s_subb_u32 s24, 0, s13
	s_mov_b64 s[18:19], 0
	v_fmamk_f32 v16, v17, 0x4f800000, v16
	v_rcp_f32_e32 v16, v16
	s_nop 0
	v_mul_f32_e32 v16, 0x5f7ffffc, v16
	v_mul_f32_e32 v17, 0x2f800000, v16
	v_trunc_f32_e32 v17, v17
	v_fmamk_f32 v16, v17, 0xcf800000, v16
	v_cvt_u32_f32_e32 v17, v17
	v_cvt_u32_f32_e32 v16, v16
	v_readfirstlane_b32 s25, v17
	v_readfirstlane_b32 s27, v16
	s_mul_i32 s40, s23, s25
	s_mul_hi_u32 s44, s23, s27
	s_mul_i32 s41, s24, s27
	s_add_i32 s40, s44, s40
	s_add_i32 s40, s40, s41
	s_mul_i32 s45, s23, s27
	s_mul_i32 s44, s27, s40
	s_mul_hi_u32 s46, s27, s45
	s_mul_hi_u32 s41, s27, s40
	s_add_u32 s44, s46, s44
	s_addc_u32 s41, 0, s41
	s_mul_hi_u32 s48, s25, s45
	s_mul_i32 s45, s25, s45
	s_add_u32 s44, s44, s45
	s_mul_hi_u32 s46, s25, s40
	s_addc_u32 s41, s41, s48
	s_addc_u32 s44, s46, 0
	s_mul_i32 s40, s25, s40
	s_add_u32 s40, s41, s40
	s_addc_u32 s41, 0, s44
	s_add_u32 s27, s27, s40
	s_addc_u32 s25, s25, s41
	s_mul_i32 s40, s23, s25
	s_mul_hi_u32 s41, s23, s27
	s_add_i32 s40, s41, s40
	s_mul_i32 s24, s24, s27
	s_add_i32 s40, s40, s24
	s_mul_i32 s23, s23, s27
	s_mul_hi_u32 s41, s25, s23
	s_mul_i32 s44, s25, s23
	s_mul_i32 s46, s27, s40
	s_mul_hi_u32 s23, s27, s23
	s_mul_hi_u32 s45, s27, s40
	s_add_u32 s23, s23, s46
	s_addc_u32 s45, 0, s45
	s_add_u32 s23, s23, s44
	s_mul_hi_u32 s24, s25, s40
	s_addc_u32 s23, s45, s41
	s_addc_u32 s24, s24, 0
	s_mul_i32 s40, s25, s40
	s_add_u32 s23, s23, s40
	s_addc_u32 s24, 0, s24
	s_add_u32 s23, s27, s23
	s_addc_u32 s27, s25, s24
	s_ashr_i32 s24, s1, 31
	s_add_u32 s40, s0, s24
	s_mov_b32 s25, s24
	s_addc_u32 s41, s1, s24
	s_xor_b64 s[40:41], s[40:41], s[24:25]
	s_mul_i32 s45, s40, s27
	s_mul_hi_u32 s46, s40, s23
	s_mul_hi_u32 s44, s40, s27
	s_add_u32 s45, s46, s45
	s_addc_u32 s44, 0, s44
	s_mul_hi_u32 s48, s41, s23
	s_mul_i32 s23, s41, s23
	s_add_u32 s23, s45, s23
	s_mul_hi_u32 s46, s41, s27
	s_addc_u32 s23, s44, s48
	s_addc_u32 s44, s46, 0
	s_mul_i32 s27, s41, s27
	s_add_u32 s23, s23, s27
	s_addc_u32 s27, 0, s44
	s_mul_i32 s27, s12, s27
	s_mul_hi_u32 s44, s12, s23
	s_add_i32 s27, s44, s27
	s_mul_i32 s44, s13, s23
	s_add_i32 s27, s27, s44
	s_sub_i32 s46, s41, s27
	s_mul_i32 s23, s12, s23
	s_sub_u32 s23, s40, s23
	s_cselect_b64 s[44:45], -1, 0
	s_subb_u32 s40, s46, s13
	s_sub_u32 s46, s23, s12
	s_cselect_b64 s[48:49], -1, 0
	s_subb_u32 s50, s40, 0
	s_cmp_ge_u32 s50, s13
	s_cselect_b32 s51, -1, 0
	s_cmp_ge_u32 s46, s12
	s_cselect_b32 s52, -1, 0
	s_cmp_eq_u32 s50, s13
	s_cselect_b32 s51, s52, s51
	s_cmp_lg_u64 s[48:49], 0
	s_subb_u32 s40, s40, s13
	s_sub_u32 s48, s46, s12
	s_subb_u32 s40, s40, 0
	s_cmp_lg_u32 s51, 0
	s_cselect_b32 s46, s48, s46
	s_cselect_b32 s40, s40, s50
	s_cmp_lg_u64 s[44:45], 0
	s_subb_u32 s27, s41, s27
	s_cmp_ge_u32 s27, s13
	s_cselect_b32 s41, -1, 0
	s_cmp_ge_u32 s23, s12
	s_cselect_b32 s12, -1, 0
	s_cmp_eq_u32 s27, s13
	s_cselect_b32 s12, s12, s41
	s_cmp_lg_u32 s12, 0
	s_cselect_b32 s13, s40, s27
	s_cselect_b32 s12, s46, s23
	s_xor_b64 s[12:13], s[12:13], s[24:25]
	s_sub_u32 s12, s12, s24
	s_subb_u32 s13, s13, s24
	s_branch .LBB0_61
.LBB0_60:
	s_mov_b64 s[18:19], -1
.LBB0_61:
	v_cvt_f32_u32_e32 v17, s26
	v_writelane_b32 v144, s4, 0
	s_andn2_b64 vcc, exec, s[18:19]
	s_nop 0
	v_writelane_b32 v144, s5, 1
	s_cbranch_vccnz .LBB0_63
	v_rcp_iflag_f32_e32 v16, v17
	s_sub_i32 s12, 0, s26
	s_mov_b32 s13, 0
	v_mul_f32_e32 v16, 0x4f7ffffe, v16
	v_cvt_u32_f32_e32 v16, v16
	s_nop 0
	v_readfirstlane_b32 s18, v16
	s_mul_i32 s12, s12, s18
	s_mul_hi_u32 s12, s18, s12
	s_add_i32 s18, s18, s12
	s_mul_hi_u32 s12, s0, s18
	s_mul_i32 s12, s12, s26
	s_sub_i32 s12, s0, s12
	s_sub_i32 s18, s12, s26
	s_cmp_ge_u32 s12, s26
	s_cselect_b32 s12, s18, s12
	s_sub_i32 s18, s12, s26
	s_cmp_ge_u32 s12, s26
	s_cselect_b32 s12, s18, s12
.LBB0_63:
	s_cmp_lg_u32 s22, 0
	s_cselect_b64 s[40:41], -1, 0
	s_lshl_b64 s[20:21], s[14:15], 5
	s_mul_i32 s18, s20, s7
	s_mul_hi_u32 s19, s20, s6
	s_lshr_b64 s[14:15], s[14:15], 27
	s_add_i32 s18, s19, s18
	s_mul_i32 s14, s14, s6
	s_add_i32 s18, s18, s14
	s_mul_i32 s14, s20, s6
	v_mov_b32_e32 v18, s14
	s_ashr_i32 s14, s3, 31
	s_mov_b32 s15, s14
	v_mov_b32_e32 v19, s18
	s_xor_b64 s[18:19], s[14:15], s[2:3]
	v_cmp_ge_i64_e32 vcc, s[2:3], v[18:19]
	s_ashr_i32 s3, s19, 31
	s_and_b64 s[66:67], s[40:41], vcc
	s_lshr_b32 s3, s3, 27
	s_add_u32 s18, s18, s3
	s_addc_u32 s19, s19, 0
	s_ashr_i64 s[18:19], s[18:19], 5
	s_xor_b64 s[24:25], s[18:19], s[14:15]
	s_ashr_i32 s14, s25, 31
	s_add_u32 s18, s24, s14
	s_mov_b32 s15, s14
	s_addc_u32 s19, s25, s14
	s_xor_b64 s[44:45], s[18:19], s[14:15]
	s_ashr_i32 s18, s7, 31
	s_add_u32 s22, s6, s18
	s_mov_b32 s19, s18
	s_addc_u32 s23, s7, s18
	s_xor_b64 s[22:23], s[22:23], s[18:19]
	v_cvt_f32_u32_e32 v16, s22
	v_cvt_f32_u32_e32 v18, s23
	s_sub_u32 s3, 0, s22
	s_subb_u32 s27, 0, s23
	v_readlane_b32 s4, v144, 10
	v_fmac_f32_e32 v16, 0x4f800000, v18
	v_rcp_f32_e32 v16, v16
	v_readlane_b32 s5, v144, 11
	v_mul_f32_e32 v16, 0x5f7ffffc, v16
	v_mul_f32_e32 v18, 0x2f800000, v16
	v_trunc_f32_e32 v18, v18
	v_fmac_f32_e32 v16, 0xcf800000, v18
	v_cvt_u32_f32_e32 v16, v16
	v_cvt_u32_f32_e32 v18, v18
	v_readfirstlane_b32 s46, v16
	v_readfirstlane_b32 s50, v18
	s_mul_hi_u32 s49, s3, s46
	s_mul_i32 s51, s3, s50
	s_mul_i32 s48, s27, s46
	s_add_i32 s49, s49, s51
	s_add_i32 s49, s49, s48
	s_mul_i32 s52, s3, s46
	s_mul_i32 s51, s46, s49
	s_mul_hi_u32 s53, s46, s52
	s_mul_hi_u32 s48, s46, s49
	s_add_u32 s51, s53, s51
	s_addc_u32 s48, 0, s48
	s_mul_hi_u32 s54, s50, s52
	s_mul_i32 s52, s50, s52
	s_add_u32 s51, s51, s52
	s_mul_hi_u32 s53, s50, s49
	s_addc_u32 s48, s48, s54
	s_addc_u32 s51, s53, 0
	s_mul_i32 s49, s50, s49
	s_add_u32 s48, s48, s49
	s_addc_u32 s49, 0, s51
	s_add_u32 s46, s46, s48
	s_addc_u32 s48, s50, s49
	s_mul_i32 s49, s3, s48
	s_mul_hi_u32 s50, s3, s46
	s_add_i32 s49, s50, s49
	s_mul_i32 s27, s27, s46
	s_add_i32 s49, s49, s27
	s_mul_i32 s3, s3, s46
	s_mul_hi_u32 s50, s48, s3
	s_mul_i32 s51, s48, s3
	s_mul_i32 s53, s46, s49
	s_mul_hi_u32 s3, s46, s3
	s_mul_hi_u32 s52, s46, s49
	s_add_u32 s3, s3, s53
	s_addc_u32 s52, 0, s52
	s_add_u32 s3, s3, s51
	s_mul_hi_u32 s27, s48, s49
	s_addc_u32 s3, s52, s50
	s_addc_u32 s27, s27, 0
	s_mul_i32 s49, s48, s49
	s_add_u32 s3, s3, s49
	s_addc_u32 s27, 0, s27
	s_add_u32 s46, s46, s3
	s_addc_u32 s27, s48, s27
	s_mul_i32 s48, s44, s27
	s_mul_hi_u32 s49, s44, s46
	s_mul_hi_u32 s3, s44, s27
	s_add_u32 s48, s49, s48
	s_addc_u32 s3, 0, s3
	s_mul_i32 s51, s45, s46
	s_mul_hi_u32 s50, s45, s46
	s_add_u32 s48, s48, s51
	s_mul_hi_u32 s49, s45, s27
	s_addc_u32 s3, s3, s50
	s_addc_u32 s48, s49, 0
	s_mul_i32 s49, s45, s27
	s_add_u32 s3, s3, s49
	s_addc_u32 s48, 0, s48
	s_mul_i32 s48, s22, s48
	s_mul_hi_u32 s49, s22, s3
	s_add_i32 s48, s49, s48
	s_mul_i32 s49, s23, s3
	s_add_i32 s52, s48, s49
	s_sub_i32 s50, s45, s52
	s_mul_i32 s3, s22, s3
	s_sub_u32 s3, s44, s3
	s_cselect_b64 s[48:49], -1, 0
	s_subb_u32 s44, s50, s23
	s_sub_u32 s53, s3, s22
	s_cselect_b64 s[50:51], -1, 0
	s_subb_u32 s54, s44, 0
	s_cmp_ge_u32 s54, s23
	s_cselect_b32 s55, -1, 0
	s_cmp_ge_u32 s53, s22
	s_cselect_b32 s56, -1, 0
	s_cmp_eq_u32 s54, s23
	s_cselect_b32 s55, s56, s55
	s_cmp_lg_u64 s[50:51], 0
	s_subb_u32 s44, s44, s23
	s_sub_u32 s50, s53, s22
	s_subb_u32 s44, s44, 0
	s_cmp_lg_u32 s55, 0
	s_cselect_b32 s50, s50, s53
	s_cselect_b32 s44, s44, s54
	s_cmp_lg_u64 s[48:49], 0
	s_subb_u32 s45, s45, s52
	s_cmp_ge_u32 s45, s23
	s_cselect_b32 s48, -1, 0
	s_cmp_ge_u32 s3, s22
	s_cselect_b32 s49, -1, 0
	s_cmp_eq_u32 s45, s23
	s_cselect_b32 s48, s49, s48
	s_cmp_lg_u32 s48, 0
	s_cselect_b32 s45, s44, s45
	s_cselect_b32 s44, s50, s3
	s_xor_b64 s[44:45], s[44:45], s[14:15]
	s_sub_u32 s3, s44, s14
	s_subb_u32 s48, s45, s14
	s_ashr_i32 s44, s48, 31
	s_mov_b32 s45, s44
	s_and_b64 s[6:7], s[44:45], s[6:7]
	s_add_u32 s70, s6, s3
	s_addc_u32 s71, s7, s48
	v_cmp_lt_i64_e64 s[6:7], s[4:5], 1
	s_and_b64 s[68:69], s[40:41], s[6:7]
	v_cmp_gt_i64_e64 s[6:7], s[4:5], 0
	v_readlane_b32 s4, v144, 0
	v_readlane_b32 s5, v144, 1
	v_cndmask_b32_e64 v16, 0, -1, s[6:7]
	s_mov_b32 s3, 0
	v_readfirstlane_b32 s6, v16
	s_mov_b32 s7, s6
	s_xor_b64 s[44:45], s[4:5], s[6:7]
	s_add_u32 s6, s26, 0
	s_addc_u32 s7, 0, 0
	s_xor_b64 s[40:41], s[6:7], 0
	v_cvt_f32_u32_e32 v18, s40
	v_cvt_f32_u32_e32 v19, s41
	s_sub_u32 s48, 0, s40
	s_subb_u32 s49, 0, s41
	s_mov_b64 s[6:7], 0
	v_fmac_f32_e32 v18, 0x4f800000, v19
	v_rcp_f32_e32 v18, v18
	s_nop 0
	v_mul_f32_e32 v18, 0x5f7ffffc, v18
	v_mul_f32_e32 v19, 0x2f800000, v18
	v_trunc_f32_e32 v19, v19
	v_fmac_f32_e32 v18, 0xcf800000, v19
	v_cvt_u32_f32_e32 v18, v18
	v_cvt_u32_f32_e32 v19, v19
	v_readfirstlane_b32 s50, v18
	v_readfirstlane_b32 s53, v19
	s_mul_hi_u32 s52, s48, s50
	s_mul_i32 s54, s48, s53
	s_mul_i32 s51, s49, s50
	s_add_i32 s52, s52, s54
	s_add_i32 s52, s52, s51
	s_mul_i32 s55, s48, s50
	s_mul_i32 s54, s50, s52
	s_mul_hi_u32 s56, s50, s55
	s_mul_hi_u32 s51, s50, s52
	s_add_u32 s54, s56, s54
	s_addc_u32 s51, 0, s51
	s_mul_hi_u32 s57, s53, s55
	s_mul_i32 s55, s53, s55
	s_add_u32 s54, s54, s55
	s_mul_hi_u32 s56, s53, s52
	s_addc_u32 s51, s51, s57
	s_addc_u32 s54, s56, 0
	s_mul_i32 s52, s53, s52
	s_add_u32 s51, s51, s52
	s_addc_u32 s52, 0, s54
	s_add_u32 s50, s50, s51
	s_addc_u32 s51, s53, s52
	s_mul_i32 s52, s48, s51
	s_mul_hi_u32 s53, s48, s50
	s_add_i32 s52, s53, s52
	s_mul_i32 s49, s49, s50
	s_add_i32 s52, s52, s49
	s_mul_i32 s48, s48, s50
	s_mul_hi_u32 s53, s51, s48
	s_mul_i32 s54, s51, s48
	s_mul_i32 s56, s50, s52
	s_mul_hi_u32 s48, s50, s48
	s_mul_hi_u32 s55, s50, s52
	s_add_u32 s48, s48, s56
	s_addc_u32 s55, 0, s55
	s_add_u32 s48, s48, s54
	s_mul_hi_u32 s49, s51, s52
	s_addc_u32 s48, s55, s53
	s_addc_u32 s49, s49, 0
	s_mul_i32 s52, s51, s52
	s_add_u32 s48, s48, s52
	s_addc_u32 s49, 0, s49
	s_add_u32 s50, s50, s48
	s_addc_u32 s51, s51, s49
	s_ashr_i32 s48, s45, 31
	s_add_u32 s44, s44, s48
	s_mov_b32 s49, s48
	s_addc_u32 s45, s45, s48
	s_xor_b64 s[44:45], s[44:45], s[48:49]
	s_mul_i32 s53, s44, s51
	s_mul_hi_u32 s54, s44, s50
	s_mul_hi_u32 s52, s44, s51
	s_add_u32 s53, s54, s53
	s_addc_u32 s52, 0, s52
	s_mul_hi_u32 s55, s45, s50
	s_mul_i32 s50, s45, s50
	s_add_u32 s50, s53, s50
	s_mul_hi_u32 s54, s45, s51
	s_addc_u32 s50, s52, s55
	s_addc_u32 s52, s54, 0
	s_mul_i32 s51, s45, s51
	s_add_u32 s53, s50, s51
	s_addc_u32 s52, 0, s52
	s_mul_i32 s50, s40, s52
	s_mul_hi_u32 s51, s40, s53
	s_add_i32 s50, s51, s50
	s_mul_i32 s51, s41, s53
	s_add_i32 s54, s50, s51
	s_sub_i32 s55, s45, s54
	s_mul_i32 s50, s40, s53
	s_sub_u32 s44, s44, s50
	s_cselect_b64 s[50:51], -1, 0
	s_subb_u32 s55, s55, s41
	s_sub_u32 s56, s44, s40
	s_subb_u32 s55, s55, 0
	s_cmp_ge_u32 s55, s41
	s_cselect_b32 s57, -1, 0
	s_cmp_ge_u32 s56, s40
	s_cselect_b32 s56, -1, 0
	s_cmp_eq_u32 s55, s41
	s_cselect_b32 s55, s56, s57
	s_add_u32 s56, s53, 1
	s_addc_u32 s57, s52, 0
	s_add_u32 s58, s53, 2
	s_addc_u32 s59, s52, 0
	s_cmp_lg_u32 s55, 0
	s_cselect_b32 s55, s58, s56
	s_cselect_b32 s56, s59, s57
	s_cmp_lg_u64 s[50:51], 0
	s_subb_u32 s45, s45, s54
	s_cmp_ge_u32 s45, s41
	s_cselect_b32 s50, -1, 0
	s_cmp_ge_u32 s44, s40
	s_cselect_b32 s40, -1, 0
	s_cmp_eq_u32 s45, s41
	s_cselect_b32 s40, s40, s50
	s_cmp_lg_u32 s40, 0
	s_cselect_b32 s41, s56, s52
	s_cselect_b32 s40, s55, s53
	s_xor_b64 s[44:45], s[48:49], 0
	s_xor_b64 s[40:41], s[40:41], s[44:45]
	s_sub_u32 s40, s40, s44
	s_subb_u32 s41, s41, s45
	v_xor_b32_e32 v18, s40, v16
	s_add_i32 s40, s26, s12
	v_xor_b32_e32 v16, s41, v16
	s_ashr_i32 s41, s40, 31
	s_cmp_lt_i32 s13, 0
	v_cndmask_b32_e64 v50, 0, v16, s[68:69]
	v_writelane_b32 v144, s68, 25
	s_cselect_b32 s12, s40, s12
	s_cselect_b32 s13, s41, s13
	v_writelane_b32 v144, s69, 26
	s_add_u32 s44, s12, s20
	v_writelane_b32 v144, s20, 16
	s_addc_u32 s45, s13, s21
	s_xor_b64 s[12:13], s[14:15], s[24:25]
	s_ashr_i32 s24, s13, 31
	s_add_u32 s12, s12, s24
	s_mov_b32 s25, s24
	s_addc_u32 s13, s13, s24
	s_xor_b64 s[12:13], s[12:13], s[24:25]
	s_mul_i32 s41, s12, s27
	s_mul_hi_u32 s48, s12, s46
	s_mul_hi_u32 s40, s12, s27
	s_add_u32 s41, s48, s41
	s_addc_u32 s40, 0, s40
	s_mul_hi_u32 s49, s13, s46
	s_mul_i32 s46, s13, s46
	s_add_u32 s41, s41, s46
	s_mul_hi_u32 s48, s13, s27
	s_addc_u32 s40, s40, s49
	s_addc_u32 s41, s48, 0
	s_mul_i32 s27, s13, s27
	s_add_u32 s27, s40, s27
	s_addc_u32 s46, 0, s41
	s_mul_i32 s40, s22, s46
	s_mul_hi_u32 s41, s22, s27
	s_add_i32 s40, s41, s40
	s_mul_i32 s41, s23, s27
	s_add_i32 s48, s40, s41
	s_sub_i32 s49, s13, s48
	s_mul_i32 s40, s22, s27
	s_sub_u32 s12, s12, s40
	s_cselect_b64 s[40:41], -1, 0
	s_subb_u32 s49, s49, s23
	s_sub_u32 s50, s12, s22
	s_subb_u32 s49, s49, 0
	s_cmp_ge_u32 s49, s23
	s_cselect_b32 s51, -1, 0
	s_cmp_ge_u32 s50, s22
	s_cselect_b32 s50, -1, 0
	s_cmp_eq_u32 s49, s23
	s_cselect_b32 s49, s50, s51
	s_add_u32 s50, s27, 1
	s_addc_u32 s51, s46, 0
	s_add_u32 s52, s27, 2
	s_addc_u32 s53, s46, 0
	s_cmp_lg_u32 s49, 0
	s_cselect_b32 s49, s52, s50
	s_cselect_b32 s50, s53, s51
	s_cmp_lg_u64 s[40:41], 0
	s_subb_u32 s13, s13, s48
	s_cmp_ge_u32 s13, s23
	s_cselect_b32 s40, -1, 0
	s_cmp_ge_u32 s12, s22
	s_cselect_b32 s12, -1, 0
	s_cmp_eq_u32 s13, s23
	s_cselect_b32 s12, s12, s40
	s_cmp_lg_u32 s12, 0
	s_cselect_b32 s13, s50, s46
	s_cselect_b32 s12, s49, s27
	s_xor_b64 s[18:19], s[24:25], s[18:19]
	s_xor_b64 s[12:13], s[12:13], s[18:19]
	s_sub_u32 s12, s12, s18
	s_subb_u32 s13, s13, s19
	s_xor_b64 s[12:13], s[12:13], s[14:15]
	s_lshl_b64 s[12:13], s[12:13], 5
	s_and_b32 s2, s2, 31
	s_or_b64 s[2:3], s[12:13], s[2:3]
	v_writelane_b32 v144, s21, 17
	s_and_b64 s[12:13], s[66:67], exec
	s_cselect_b32 s45, s45, s3
	s_cselect_b32 s44, s44, s2
	v_readlane_b32 s2, v144, 8
	v_readlane_b32 s3, v144, 9
	v_cndmask_b32_e64 v25, 0, v18, s[68:69]
	v_lshlrev_b32_e32 v16, 4, v95
	v_mov_b64_e32 v[18:19], s[2:3]
	v_cmp_lt_i64_e32 vcc, s[62:63], v[18:19]
	s_cbranch_vccz .LBB0_65
	s_mov_b64 s[6:7], -1
.LBB0_65:
	s_andn2_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz .LBB0_85
	s_ashr_i32 s2, s1, 31
	s_mov_b32 s3, s2
	s_xor_b64 s[6:7], s[2:3], s[0:1]
	s_add_u32 s0, s26, 0
	s_addc_u32 s1, 0, 0
	s_xor_b64 s[0:1], s[0:1], 0
	v_cvt_f32_u32_e32 v18, s0
	v_cvt_f32_u32_e32 v19, s1
	s_sub_u32 s12, 0, s0
	s_subb_u32 s13, 0, s1
	v_writelane_b32 v144, s66, 27
	v_fmac_f32_e32 v18, 0x4f800000, v19
	v_rcp_f32_e32 v18, v18
	v_writelane_b32 v144, s67, 28
	v_writelane_b32 v144, s71, 29
	v_writelane_b32 v144, s70, 6
	v_mul_f32_e32 v18, 0x5f7ffffc, v18
	v_mul_f32_e32 v19, 0x2f800000, v18
	v_trunc_f32_e32 v19, v19
	v_fmac_f32_e32 v18, 0xcf800000, v19
	v_cvt_u32_f32_e32 v19, v19
	v_cvt_u32_f32_e32 v18, v18
	v_bitop3_b32 v23, v63, v24, 7 bitop3:0x78
	v_readlane_b32 s4, v144, 24
	v_readfirstlane_b32 s14, v19
	v_readfirstlane_b32 s15, v18
	s_mul_i32 s18, s12, s14
	s_mul_hi_u32 s22, s12, s15
	s_mul_i32 s19, s13, s15
	s_add_i32 s18, s22, s18
	s_add_i32 s18, s18, s19
	s_mul_i32 s23, s12, s15
	s_mul_i32 s22, s15, s18
	s_mul_hi_u32 s24, s15, s23
	s_mul_hi_u32 s19, s15, s18
	s_add_u32 s22, s24, s22
	s_addc_u32 s19, 0, s19
	s_mul_hi_u32 s25, s14, s23
	s_mul_i32 s23, s14, s23
	s_add_u32 s22, s22, s23
	s_mul_hi_u32 s24, s14, s18
	s_addc_u32 s19, s19, s25
	s_addc_u32 s22, s24, 0
	s_mul_i32 s18, s14, s18
	s_add_u32 s18, s19, s18
	s_addc_u32 s19, 0, s22
	s_add_u32 s15, s15, s18
	s_addc_u32 s14, s14, s19
	s_mul_i32 s18, s12, s14
	s_mul_hi_u32 s19, s12, s15
	s_add_i32 s18, s19, s18
	s_mul_i32 s13, s13, s15
	s_add_i32 s18, s18, s13
	s_mul_i32 s12, s12, s15
	s_mul_hi_u32 s19, s14, s12
	s_mul_i32 s22, s14, s12
	s_mul_i32 s24, s15, s18
	s_mul_hi_u32 s12, s15, s12
	s_mul_hi_u32 s23, s15, s18
	s_add_u32 s12, s12, s24
	s_addc_u32 s23, 0, s23
	s_add_u32 s12, s12, s22
	s_mul_hi_u32 s13, s14, s18
	s_addc_u32 s12, s23, s19
	s_addc_u32 s13, s13, 0
	s_mul_i32 s18, s14, s18
	s_add_u32 s12, s12, s18
	s_addc_u32 s13, 0, s13
	s_add_u32 s15, s15, s12
	s_addc_u32 s14, s14, s13
	s_ashr_i32 s12, s7, 31
	s_add_u32 s6, s6, s12
	s_mov_b32 s13, s12
	s_addc_u32 s7, s7, s12
	s_xor_b64 s[6:7], s[6:7], s[12:13]
	s_mul_i32 s19, s6, s14
	s_mul_hi_u32 s22, s6, s15
	s_mul_hi_u32 s18, s6, s14
	s_add_u32 s19, s22, s19
	s_addc_u32 s18, 0, s18
	s_mul_hi_u32 s23, s7, s15
	s_mul_i32 s15, s7, s15
	s_add_u32 s15, s19, s15
	s_mul_hi_u32 s22, s7, s14
	s_addc_u32 s15, s18, s23
	s_addc_u32 s18, s22, 0
	s_mul_i32 s14, s7, s14
	s_add_u32 s19, s15, s14
	s_addc_u32 s18, 0, s18
	s_mul_i32 s14, s0, s18
	s_mul_hi_u32 s15, s0, s19
	s_add_i32 s14, s15, s14
	s_mul_i32 s15, s1, s19
	s_add_i32 s22, s14, s15
	s_sub_i32 s23, s7, s22
	s_mul_i32 s14, s0, s19
	s_sub_u32 s6, s6, s14
	s_cselect_b64 s[14:15], -1, 0
	s_subb_u32 s23, s23, s1
	s_sub_u32 s24, s6, s0
	s_subb_u32 s23, s23, 0
	s_cmp_ge_u32 s23, s1
	s_cselect_b32 s25, -1, 0
	s_cmp_ge_u32 s24, s0
	s_cselect_b32 s24, -1, 0
	s_cmp_eq_u32 s23, s1
	s_cselect_b32 s23, s24, s25
	s_add_u32 s24, s19, 1
	s_addc_u32 s25, s18, 0
	s_add_u32 s27, s19, 2
	s_addc_u32 s40, s18, 0
	s_cmp_lg_u32 s23, 0
	s_cselect_b32 s23, s27, s24
	s_cselect_b32 s24, s40, s25
	s_cmp_lg_u64 s[14:15], 0
	s_subb_u32 s7, s7, s22
	s_cmp_ge_u32 s7, s1
	s_cselect_b32 s14, -1, 0
	s_cmp_ge_u32 s6, s0
	s_cselect_b32 s0, -1, 0
	s_cmp_eq_u32 s7, s1
	s_cselect_b32 s0, s0, s14
	s_cmp_lg_u32 s0, 0
	s_cselect_b32 s1, s24, s18
	s_cselect_b32 s0, s23, s19
	s_xor_b64 s[6:7], s[12:13], 0
	s_xor_b64 s[0:1], s[0:1], s[6:7]
	s_sub_u32 s0, s0, s6
	s_subb_u32 s1, s1, s7
	s_xor_b64 s[0:1], s[0:1], s[2:3]
	s_and_b64 s[2:3], s[66:67], exec
	s_cselect_b32 s1, s1, s71
	s_cselect_b32 s0, s0, s70
	s_lshl_b64 s[2:3], s[0:1], 6
	v_or_b32_e32 v19, s3, v29
	v_or_b32_e32 v18, s2, v28
	v_cmp_gt_i64_e64 s[2:3], s[60:61], v[18:19]
	v_or_b32_e32 v18, 32, v18
	v_cmp_gt_i64_e64 s[14:15], s[60:61], v[18:19]
	v_and_b32_e32 v19, 7, v24
	v_bitop3_b32 v22, v63, v19, 4 bitop3:0x36
	s_lshl_b64 s[6:7], s[0:1], 1
	s_movk_i32 s1, 0x4000
	v_lshlrev_b32_e32 v19, 11, v63
	v_lshl_add_u32 v48, v22, 4, s1
	v_lshlrev_b32_e32 v22, 8, v63
	v_sub_u32_e32 v30, v26, v19
	v_mov_b32_e32 v19, 0
	v_lshl_add_u32 v32, v24, 4, v22
	v_mul_i32_i24_e32 v22, -16, v63
	v_lshl_add_u32 v47, v23, 4, s1
	v_ashrrev_i32_e32 v23, 31, v22
	v_add_u32_e32 v34, v16, v24
	v_mov_b32_e32 v35, v19
	s_lshl_b32 s24, s4, 7
	v_readlane_b32 s4, v144, 20
	v_writelane_b32 v144, s44, 20
	s_lshl_b64 s[12:13], s[44:45], 6
	v_lshl_add_u64 v[22:23], v[34:35], 0, v[22:23]
	v_lshl_add_u64 v[34:35], v[22:23], 0, s[12:13]
	v_writelane_b32 v144, s45, 21
	v_ashrrev_i32_e32 v36, 31, v35
	s_lshl_b32 s27, s4, 7
	v_xor_b32_e32 v23, v36, v35
	v_readlane_b32 s4, v144, 18
	s_mov_b32 s46, 0
	v_bfe_u32 v40, v22, 4, 1
	v_xor_b32_e32 v22, v36, v34
	v_ashrrev_i32_e32 v34, 31, v23
	v_readlane_b32 s5, v144, 19
	s_add_u32 s1, s4, 7
	v_lshrrev_b32_e32 v34, 27, v34
	v_mov_b32_e32 v35, s46
	s_addc_u32 s18, s5, 0
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[34:35]
	s_sub_u32 s19, -8, s4
	v_or_b32_e32 v51, s12, v16
	v_ashrrev_i64 v[22:23], 5, v[22:23]
	s_subb_u32 s22, -1, s5
	v_cmp_lt_i64_e64 s[12:13], s[4:5], -7
	v_xor_b32_e32 v52, v22, v36
	v_lshl_add_u32 v46, v25, 13, v30
	v_cndmask_b32_e64 v22, 0, -1, s[12:13]
	s_and_b64 s[12:13], s[12:13], exec
	s_cselect_b32 s13, s22, s18
	s_cselect_b32 s1, s19, s1
	s_ashr_i32 s12, s13, 31
	s_lshr_b32 s12, s12, 29
	s_add_u32 s12, s1, s12
	s_addc_u32 s13, s13, 0
	v_readfirstlane_b32 s18, v22
	s_ashr_i64 s[12:13], s[12:13], 3
	s_mov_b32 s19, s18
	v_sub_co_u32_e32 v22, vcc, 0, v32
	v_mul_i32_i24_e32 v30, 0xffffff00, v63
	v_xor_b32_e32 v49, v23, v36
	s_xor_b64 s[12:13], s[12:13], s[18:19]
	v_subb_co_u32_e64 v23, s[18:19], 0, 0, vcc
	v_ashrrev_i32_e32 v31, 31, v30
	s_lshl_b64 s[18:19], s[62:63], 11
	v_mov_b32_e32 v33, v19
	v_lshl_add_u64 v[36:37], s[18:19], 0, v[30:31]
	v_lshl_add_u64 v[32:33], v[36:37], 0, v[32:33]
	s_mov_b64 s[18:19], 0x40f
	v_sub_co_u32_e32 v30, vcc, 0, v36
	v_lshl_add_u64 v[32:33], v[32:33], 0, s[18:19]
	s_lshl_b32 s47, s47, 5
	s_lshl_b64 s[18:19], s[64:65], 5
	v_or_b32_e32 v36, s6, v94
	v_subb_co_u32_e32 v31, vcc, 0, v37, vcc
	s_or_b64 s[22:23], s[46:47], s[18:19]
	s_mul_i32 s6, s18, s7
	v_mul_hi_u32 v37, s18, v36
	v_add_u32_e32 v37, s6, v37
	v_mul_lo_u32 v42, s23, v36
	v_add_u32_e32 v37, v37, v42
	v_mul_lo_u32 v36, s18, v36
	s_lshl_b64 s[6:7], s[62:63], 13
	v_mov_b32_e32 v27, v19
	v_lshl_add_u64 v[36:37], v[36:37], 0, s[6:7]
	s_lshl_b64 s[56:57], s[62:63], 8
	s_lshl_b32 s1, s12, 8
	v_lshl_add_u64 v[26:27], v[36:37], 0, v[26:27]
	v_mov_b64_e32 v[36:37], s[56:57]
	v_lshlrev_b32_e32 v18, 2, v24
	v_mul_i32_i24_e32 v38, 0xffffffc0, v63
	v_lshlrev_b32_e32 v34, 6, v63
	v_mov_b32_e32 v35, v19
	v_mad_u64_u32 v[36:37], s[6:7], s1, v52, v[36:37]
	v_ashrrev_i32_e32 v39, 31, v38
	v_lshl_add_u64 v[42:43], v[34:35], 0, v[18:19]
	s_lshr_b64 s[6:7], s[12:13], 24
	v_or_b32_e32 v29, 0x6400, v18
	v_lshl_add_u64 v[34:35], v[42:43], 0, v[38:39]
	v_mul_lo_u32 v18, s1, v49
	v_mul_lo_u32 v49, s6, v52
	v_mad_u64_u32 v[42:43], s[6:7], s1, v52, v[42:43]
	v_or_b32_e32 v34, v34, v40
	v_add3_u32 v37, v49, v37, v18
	v_add3_u32 v43, v49, v43, v18
	v_lshl_add_u32 v18, s0, 6, v28
	v_bitop3_b32 v20, v24, v45, 7 bitop3:0x6c
	s_lshl_b64 s[54:55], s[12:13], 3
	v_lshl_add_u64 v[36:37], v[34:35], 0, v[36:37]
	v_add_u32_e32 v28, 32, v18
	v_lshlrev_b32_e32 v44, 4, v20
	v_lshlrev_b32_e32 v45, 9, v25
	v_mov_b32_e32 v41, v19
	v_sub_co_u32_e32 v36, vcc, -3, v36
	v_lshl_add_u64 v[38:39], v[42:43], 0, v[38:39]
	v_mul_lo_u32 v28, v28, s42
	s_lshl_b32 s0, s62, 7
	v_mul_lo_u32 v18, v18, s42
	s_movk_i32 s58, 0xfbf1
	s_ashr_i32 s60, s43, 31
	s_movk_i32 s64, 0xfbf2
	s_movk_i32 s66, 0xfbf3
	s_movk_i32 s68, 0xfbf4
	s_movk_i32 s70, 0xfbf5
	s_movk_i32 s72, 0xfbf6
	s_movk_i32 s74, 0xfbf7
	s_movk_i32 s76, 0xfbf8
	s_movk_i32 s78, 0xfbf9
	s_movk_i32 s80, 0xfbfa
	s_movk_i32 s82, 0xfbfb
	s_movk_i32 s84, 0xfbfc
	s_movk_i32 s86, 0xfbfd
	s_movk_i32 s88, 0xfbfe
	s_movk_i32 s90, 0xfbff
	s_movk_i32 s20, 0xfc00
	s_ashr_i32 s96, s55, 31
	s_movk_i32 s48, 0xf800
	s_movk_i32 s52, 0xff00
	v_lshlrev_b32_e32 v20, 5, v20
	v_mov_b32_e32 v21, v19
	v_subb_co_u32_e32 v37, vcc, -1, v37, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[40:41]
	v_add3_u32 v52, v28, s0, v44
	v_add3_u32 v53, v18, s0, v44
	s_add_i32 s47, s24, 0x4000
	s_addk_i32 s27, 0x4000
	v_add_u32_e32 v54, v29, v45
	v_add_u32_e32 v55, v47, v46
	v_add_u32_e32 v56, v48, v46
	s_mov_b32 s59, -1
	s_mov_b32 s61, s60
	s_mov_b32 s65, -1
	s_mov_b32 s67, -1
	s_mov_b32 s69, -1
	s_mov_b32 s71, -1
	s_mov_b32 s73, -1
	s_mov_b32 s75, -1
	s_mov_b32 s77, -1
	s_mov_b32 s79, -1
	s_mov_b32 s81, -1
	s_mov_b32 s83, -1
	s_mov_b32 s85, -1
	s_mov_b32 s87, -1
	s_mov_b32 s89, -1
	s_mov_b32 s91, -1
	s_mov_b32 s21, -1
	s_mov_b32 s40, s96
	s_mov_b32 s41, s96
	s_mov_b32 s49, -1
	s_mov_b64 s[0:1], 0x800
	s_mov_b64 s[12:13], 0x2000
	s_mov_b64 s[44:45], 0x100
	s_mov_b32 s53, -1
	v_bfrev_b32_e32 v57, -2
	s_branch .LBB0_68
	.p2align	5
.LBB0_67:
	s_or_b64 exec, exec, s[6:7]
	s_add_u32 s6, s54, s96
	s_addc_u32 s7, s55, s96
	s_xor_b64 s[6:7], s[6:7], s[40:41]
	v_cvt_f32_u32_e32 v18, s6
	v_cvt_f32_u32_e32 v47, s7
	v_ashrrev_i32_e32 v91, 31, v43
	v_xor_b32_e32 v126, v46, v91
	s_sub_u32 s4, 0, s6
	v_fmac_f32_e32 v18, 0x4f800000, v47
	v_rcp_f32_e32 v18, v18
	s_subb_u32 s5, 0, s7
	v_xor_b32_e32 v43, v91, v43
	v_xor_b32_e32 v42, v91, v42
	v_mul_f32_e32 v18, 0x5f7ffffc, v18
	v_mul_f32_e32 v46, 0x2f800000, v18
	v_trunc_f32_e32 v46, v46
	v_fmac_f32_e32 v18, 0xcf800000, v46
	v_cvt_u32_f32_e32 v18, v18
	v_cvt_u32_f32_e32 v46, v46
	v_ashrrev_i32_e32 v41, 31, v41
	v_and_b32_e32 v41, s54, v41
	v_readfirstlane_b32 s18, v18
	v_readfirstlane_b32 s23, v46
	s_mul_hi_u32 s22, s4, s18
	s_mul_i32 s24, s4, s23
	s_mul_i32 s19, s5, s18
	s_add_i32 s22, s22, s24
	s_add_i32 s22, s22, s19
	s_mul_i32 s25, s4, s18
	s_mul_i32 s24, s18, s22
	s_mul_hi_u32 s50, s18, s25
	s_mul_hi_u32 s19, s18, s22
	s_add_u32 s24, s50, s24
	s_addc_u32 s19, 0, s19
	s_mul_hi_u32 s51, s23, s25
	s_mul_i32 s25, s23, s25
	s_add_u32 s24, s24, s25
	s_mul_hi_u32 s50, s23, s22
	s_addc_u32 s19, s19, s51
	s_addc_u32 s24, s50, 0
	s_mul_i32 s22, s23, s22
	s_add_u32 s19, s19, s22
	s_addc_u32 s22, 0, s24
	s_add_u32 s18, s18, s19
	s_addc_u32 s19, s23, s22
	s_mul_i32 s22, s4, s19
	s_mul_hi_u32 s23, s4, s18
	s_add_i32 s22, s23, s22
	s_mul_i32 s5, s5, s18
	s_add_i32 s22, s22, s5
	s_mul_i32 s4, s4, s18
	s_mul_hi_u32 s23, s19, s4
	s_mul_i32 s24, s19, s4
	s_mul_i32 s50, s18, s22
	s_mul_hi_u32 s4, s18, s4
	s_mul_hi_u32 s25, s18, s22
	s_add_u32 s4, s4, s50
	s_addc_u32 s25, 0, s25
	s_add_u32 s4, s4, s24
	s_mul_hi_u32 s5, s19, s22
	s_addc_u32 s4, s25, s23
	s_addc_u32 s5, s5, 0
	s_mul_i32 s22, s19, s22
	s_add_u32 s4, s4, s22
	v_ashrrev_i32_e32 v46, 31, v43
	s_addc_u32 s5, 0, s5
	v_mov_b32_e32 v47, v46
	s_add_u32 s4, s18, s4
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[46:47]
	s_addc_u32 s5, s19, s5
	v_xor_b32_e32 v127, v42, v46
	v_xor_b32_e32 v47, v43, v46
	v_mad_u64_u32 v[42:43], s[18:19], v127, s5, 0
	v_mul_hi_u32 v18, v127, s4
	v_lshl_add_u64 v[42:43], v[18:19], 0, v[42:43]
	v_mad_u64_u32 v[124:125], s[18:19], v47, s4, 0
	v_add_co_u32_e32 v18, vcc, v42, v124
	v_mad_u64_u32 v[48:49], s[18:19], v47, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v42, vcc, v43, v125, vcc
	v_mov_b32_e32 v43, s46
	s_nop 0
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[48:49]
	v_mul_lo_u32 v18, s7, v42
	v_mul_lo_u32 v43, s6, v43
	v_mad_u64_u32 v[48:49], s[18:19], s6, v42, 0
	v_add3_u32 v18, v49, v43, v18
	v_sub_u32_e32 v43, v47, v18
	v_mov_b32_e32 v49, s7
	v_sub_co_u32_e32 v48, vcc, v127, v48
	v_readlane_b32 s4, v144, 5
	s_nop 0
	v_subb_co_u32_e64 v43, s[18:19], v43, v49, vcc
	v_subrev_co_u32_e64 v49, s[18:19], s6, v48
	v_subb_co_u32_e32 v18, vcc, v47, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[18:19], 0, v43, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v43
	v_cmp_le_u32_e32 vcc, s7, v18
	s_add_u32 s62, s62, 1
	v_cndmask_b32_e64 v124, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v49
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v48
	v_cndmask_b32_e64 v49, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v43
	v_cndmask_b32_e64 v48, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v18
	v_cndmask_b32_e64 v43, v124, v49, s[18:19]
	v_add_u32_e32 v49, 2, v42
	v_add_u32_e32 v124, 1, v42
	v_cmp_ne_u32_e64 s[18:19], 0, v43
	v_cndmask_b32_e32 v18, v47, v48, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v43, v124, v49, s[18:19]
	s_waitcnt vmcnt(12)
	v_lshlrev_b16_e32 v49, 8, v77
	v_cndmask_b32_e32 v18, v42, v43, vcc
	v_xor_b32_e32 v42, s96, v46
	v_xor_b32_e32 v18, v18, v42
	v_sub_u32_e32 v18, v18, v42
	v_xor_b32_e32 v18, v18, v91
	v_mul_lo_u32 v18, v18, s4
	v_add3_u32 v18, v18, v40, v41
	buffer_load_ubyte v18, v18, s[8:11], 0 offen
	v_ashrrev_i32_e32 v41, 31, v45
	v_mul_lo_u32 v40, v126, s4
	v_and_b32_e32 v41, s54, v41
	v_add3_u32 v40, v40, v44, v41
	buffer_load_ubyte v48, v40, s[8:11], 0 offen
	v_lshlrev_b16_e32 v40, 8, v73
	v_lshlrev_b16_e32 v41, 8, v74
	v_or_b32_e32 v40, v71, v40
	v_or_b32_sdwa v41, v72, v41 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v44, 8, v61
	v_or_b32_sdwa v43, v40, v41 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v40, 8, v69
	v_lshlrev_b16_e32 v41, 8, v70
	v_or_b32_e32 v40, v67, v40
	v_or_b32_sdwa v41, v68, v41 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v44, v59, v44 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v42, v40, v41 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v40, 8, v65
	v_lshlrev_b16_e32 v41, 8, v66
	v_or_b32_e32 v40, v62, v40
	v_or_b32_sdwa v41, v64, v41 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt vmcnt(2)
	v_lshlrev_b16_e32 v45, 8, v90
	v_or_b32_sdwa v41, v40, v41 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v40, 8, v60
	v_or_b32_e32 v40, v58, v40
	v_or_b32_sdwa v40, v40, v44 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v44, 8, v89
	v_or_b32_e32 v44, v87, v44
	v_or_b32_sdwa v45, v88, v45 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt vmcnt(1) lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[120:123], v[40:43], v[12:15], v28, v18 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_or_b32_sdwa v47, v44, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v44, 8, v85
	v_lshlrev_b16_e32 v45, 8, v86
	v_or_b32_e32 v44, v83, v44
	v_or_b32_sdwa v45, v84, v45 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[92:95], v[40:43], v[8:11], v28, v18 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_or_b32_sdwa v46, v44, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v44, 8, v81
	v_lshlrev_b16_e32 v45, 8, v82
	v_or_b32_e32 v44, v79, v44
	v_or_b32_sdwa v45, v80, v45 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[104:107], v[40:43], v[4:7], v29, v18 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_or_b32_sdwa v45, v44, v45 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v44, 8, v78
	v_or_b32_e32 v44, v76, v44
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[96:99], v[40:43], v[0:3], v29, v18 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_or_b32_sdwa v49, v75, v49 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_readlane_b32 s4, v144, 8
	v_or_b32_sdwa v44, v44, v49 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_readlane_b32 s5, v144, 9
	s_addc_u32 s63, s63, 0
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[112:115], v[44:47], v[12:15], v28, v48 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshl_add_u64 v[30:31], v[30:31], 0, s[48:49]
	v_lshl_add_u64 v[32:33], v[32:33], 0, s[0:1]
	v_lshl_add_u64 v[26:27], v[26:27], 0, s[12:13]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[100:103], v[44:47], v[8:11], v28, v48 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshl_add_u64 v[34:35], v[34:35], 0, s[44:45]
	v_lshl_add_u64 v[36:37], v[36:37], 0, s[52:53]
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[44:45]
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[116:119], v[44:47], v[4:7], v29, v48 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshl_add_u64 v[20:21], v[20:21], 0, s[44:45]
	v_add_u32_e32 v52, 0x80, v52
	v_add_u32_e32 v53, 0x80, v53
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[108:111], v[44:47], v[0:3], v29, v48 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mov_b64_e32 v[28:29], s[4:5]
	v_cmp_lt_i64_e32 vcc, s[62:63], v[28:29]
	s_cbranch_vccz .LBB0_84
.LBB0_68:
	v_lshl_add_u64 v[28:29], s[56:57], 0, v[20:21]
	v_cmp_gt_i64_e32 vcc, s[34:35], v[28:29]
	s_and_b64 s[18:19], s[2:3], vcc
	s_mov_b32 m0, s47
	v_cndmask_b32_e64 v18, v57, v53, s[18:19]
	s_and_b64 vcc, s[14:15], vcc
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v18, s[36:39], 0 offen lds
	v_cndmask_b32_e32 v18, v57, v52, vcc
	s_mov_b32 m0, s27
	v_ashrrev_i32_e32 v28, 31, v27
	buffer_load_dwordx4 v18, s[36:39], 0 offen lds
	v_or_b32_e32 v18, s35, v27
	v_cmp_ne_u32_e32 vcc, 0, v18
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
	s_cbranch_execz .LBB0_70
	s_ashr_i32 s18, s35, 31
	s_add_u32 s22, s34, s18
	s_mov_b32 s19, s18
	s_addc_u32 s23, s35, s18
	s_xor_b64 s[24:25], s[22:23], s[18:19]
	v_cvt_f32_u32_e32 v18, s24
	v_cvt_f32_u32_e32 v29, s25
	s_sub_u32 s18, 0, s24
	s_subb_u32 s19, 0, s25
	v_fmac_f32_e32 v18, 0x4f800000, v29
	v_rcp_f32_e32 v18, v18
	s_nop 0
	v_mul_f32_e32 v18, 0x5f7ffffc, v18
	v_mul_f32_e32 v29, 0x2f800000, v18
	v_trunc_f32_e32 v29, v29
	v_fmac_f32_e32 v18, 0xcf800000, v29
	v_cvt_u32_f32_e32 v29, v29
	v_cvt_u32_f32_e32 v18, v18
	v_readfirstlane_b32 s22, v29
	v_readfirstlane_b32 s23, v18
	s_mul_i32 s50, s18, s22
	s_mul_hi_u32 s97, s18, s23
	s_mul_i32 s51, s19, s23
	s_add_i32 s50, s97, s50
	s_add_i32 s50, s50, s51
	s_mul_i32 vcc_lo, s18, s23
	s_mul_i32 s97, s23, s50
	s_mul_hi_u32 vcc_hi, s23, vcc_lo
	s_mul_hi_u32 s51, s23, s50
	s_add_u32 s97, vcc_hi, s97
	s_addc_u32 s51, 0, s51
	s_mul_i32 s5, s22, vcc_lo
	s_mul_hi_u32 s4, s22, vcc_lo
	s_add_u32 s5, s97, s5
	s_mul_hi_u32 vcc_hi, s22, s50
	s_addc_u32 s4, s51, s4
	s_addc_u32 s5, vcc_hi, 0
	s_mul_i32 s50, s22, s50
	s_add_u32 s4, s4, s50
	s_addc_u32 s5, 0, s5
	s_add_u32 s4, s23, s4
	s_addc_u32 s5, s22, s5
	s_mul_i32 s22, s18, s5
	s_mul_hi_u32 s23, s18, s4
	s_add_i32 s22, s23, s22
	s_mul_i32 s19, s19, s4
	s_add_i32 s22, s22, s19
	s_mul_i32 s18, s18, s4
	s_mul_hi_u32 s23, s5, s18
	s_mul_i32 s50, s5, s18
	s_mul_i32 s97, s4, s22
	s_mul_hi_u32 s18, s4, s18
	s_mul_hi_u32 s51, s4, s22
	s_add_u32 s18, s18, s97
	s_addc_u32 s51, 0, s51
	s_add_u32 s18, s18, s50
	s_mul_hi_u32 s19, s5, s22
	s_addc_u32 s18, s51, s23
	s_addc_u32 s19, s19, 0
	s_mul_i32 s22, s5, s22
	s_add_u32 s18, s18, s22
	s_addc_u32 s19, 0, s19
	v_mov_b32_e32 v29, v28
	s_add_u32 s4, s4, s18
	v_lshl_add_u64 v[40:41], v[26:27], 0, v[28:29]
	s_addc_u32 s5, s5, s19
	v_xor_b32_e32 v46, v40, v28
	v_xor_b32_e32 v29, v41, v28
	v_mad_u64_u32 v[40:41], s[18:19], v46, s5, 0
	v_mul_hi_u32 v18, v46, s4
	v_lshl_add_u64 v[40:41], v[18:19], 0, v[40:41]
	v_mad_u64_u32 v[44:45], s[18:19], v29, s4, 0
	v_add_co_u32_e32 v18, vcc, v40, v44
	v_mad_u64_u32 v[42:43], s[18:19], v29, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v40, vcc, v41, v45, vcc
	v_mov_b32_e32 v41, s46
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[42:43]
	v_mul_lo_u32 v18, s25, v40
	v_mul_lo_u32 v42, s24, v41
	v_mad_u64_u32 v[40:41], s[18:19], s24, v40, 0
	v_add3_u32 v18, v41, v42, v18
	v_sub_u32_e32 v41, v29, v18
	v_mov_b32_e32 v42, s25
	v_sub_co_u32_e32 v40, vcc, v46, v40
	s_nop 1
	v_subb_co_u32_e64 v41, s[18:19], v41, v42, vcc
	v_subrev_co_u32_e64 v43, s[18:19], s24, v40
	v_subb_co_u32_e32 v18, vcc, v29, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v44, s[22:23], 0, v41, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s25, v44
	v_subb_co_u32_e64 v41, s[18:19], v41, v42, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v45, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s24, v43
	v_subrev_co_u32_e64 v42, s[18:19], s24, v43
	s_nop 0
	v_cndmask_b32_e64 v46, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s25, v44
	v_subbrev_co_u32_e64 v41, s[18:19], 0, v41, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v45, v45, v46, s[22:23]
	v_cmp_le_u32_e32 vcc, s25, v18
	v_cmp_ne_u32_e64 s[18:19], 0, v45
	s_nop 0
	v_cndmask_b32_e64 v29, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s24, v40
	v_cndmask_b32_e64 v41, v44, v41, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v44, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s25, v18
	s_nop 1
	v_cndmask_b32_e32 v29, v29, v44, vcc
	v_cmp_ne_u32_e32 vcc, 0, v29
	v_cndmask_b32_e64 v29, v43, v42, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v29, v40, v29, vcc
	v_cndmask_b32_e32 v18, v18, v41, vcc
	v_xor_b32_e32 v29, v29, v28
	v_xor_b32_e32 v18, v18, v28
	v_sub_co_u32_e32 v40, vcc, v29, v28
	s_nop 1
	v_subb_co_u32_e32 v41, vcc, v18, v28, vcc
.LBB0_70:
	s_andn2_saveexec_b64 s[6:7], s[6:7]
	s_cbranch_execz .LBB0_72
	v_cvt_f32_u32_e32 v18, s34
	s_sub_i32 s18, 0, s34
	v_rcp_iflag_f32_e32 v18, v18
	s_nop 0
	v_mul_f32_e32 v18, 0x4f7ffffe, v18
	v_cvt_u32_f32_e32 v18, v18
	v_mul_lo_u32 v29, s18, v18
	v_mul_hi_u32 v29, v18, v29
	v_add_u32_e32 v18, v18, v29
	v_mul_hi_u32 v18, v26, v18
	v_mul_lo_u32 v18, v18, s34
	v_sub_u32_e32 v18, v26, v18
	v_subrev_u32_e32 v29, s34, v18
	v_cmp_le_u32_e32 vcc, s34, v18
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v29, vcc
	v_subrev_u32_e32 v29, s34, v18
	v_cmp_le_u32_e32 vcc, s34, v18
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v29, vcc
	v_mov_b64_e32 v[40:41], v[18:19]
.LBB0_72:
	s_or_b64 exec, exec, s[6:7]
	s_ashr_i32 s6, s35, 31
	s_add_u32 s18, s34, s6
	s_mov_b32 s7, s6
	s_addc_u32 s19, s35, s6
	s_xor_b64 s[22:23], s[18:19], s[6:7]
	v_cvt_f32_u32_e32 v18, s22
	v_cvt_f32_u32_e32 v29, s23
	s_sub_u32 s4, 0, s22
	s_subb_u32 s5, 0, s23
	v_xor_b32_e32 v43, v28, v27
	v_fmac_f32_e32 v18, 0x4f800000, v29
	v_rcp_f32_e32 v18, v18
	v_ashrrev_i32_e32 v44, 31, v43
	v_xor_b32_e32 v42, v28, v26
	v_mov_b32_e32 v45, v44
	v_mul_f32_e32 v18, 0x5f7ffffc, v18
	v_mul_f32_e32 v29, 0x2f800000, v18
	v_trunc_f32_e32 v29, v29
	v_fmac_f32_e32 v18, 0xcf800000, v29
	v_cvt_u32_f32_e32 v29, v29
	v_cvt_u32_f32_e32 v18, v18
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[44:45]
	v_xor_b32_e32 v45, v42, v44
	v_readfirstlane_b32 s7, v29
	v_readfirstlane_b32 s18, v18
	s_mul_hi_u32 s24, s4, s18
	s_mul_i32 s25, s4, s7
	s_mul_i32 s19, s5, s18
	s_add_i32 s24, s24, s25
	s_add_i32 s24, s24, s19
	s_mul_i32 s50, s4, s18
	s_mul_i32 s25, s18, s24
	s_mul_hi_u32 s51, s18, s50
	s_mul_hi_u32 s19, s18, s24
	s_add_u32 s25, s51, s25
	s_addc_u32 s19, 0, s19
	s_mul_hi_u32 s97, s7, s50
	s_mul_i32 s50, s7, s50
	s_add_u32 s25, s25, s50
	s_mul_hi_u32 s51, s7, s24
	s_addc_u32 s19, s19, s97
	s_addc_u32 s25, s51, 0
	s_mul_i32 s24, s7, s24
	s_add_u32 s19, s19, s24
	s_addc_u32 s24, 0, s25
	s_add_u32 s18, s18, s19
	s_addc_u32 s7, s7, s24
	s_mul_i32 s19, s4, s7
	s_mul_hi_u32 s24, s4, s18
	s_add_i32 s19, s24, s19
	s_mul_i32 s5, s5, s18
	s_add_i32 s19, s19, s5
	s_mul_i32 s4, s4, s18
	s_mul_hi_u32 s24, s7, s4
	s_mul_i32 s25, s7, s4
	s_mul_i32 s51, s18, s19
	s_mul_hi_u32 s4, s18, s4
	s_mul_hi_u32 s50, s18, s19
	s_add_u32 s4, s4, s51
	s_addc_u32 s50, 0, s50
	s_add_u32 s4, s4, s25
	s_mul_hi_u32 s5, s7, s19
	s_addc_u32 s4, s50, s24
	s_addc_u32 s5, s5, 0
	s_mul_i32 s19, s7, s19
	s_add_u32 s4, s4, s19
	s_addc_u32 s5, 0, s5
	s_add_u32 s4, s18, s4
	s_addc_u32 s5, s7, s5
	v_xor_b32_e32 v29, v43, v44
	v_mad_u64_u32 v[42:43], s[18:19], v45, s5, 0
	v_mul_hi_u32 v18, v45, s4
	v_lshl_add_u64 v[42:43], v[18:19], 0, v[42:43]
	v_mad_u64_u32 v[48:49], s[18:19], v29, s4, 0
	v_add_co_u32_e32 v18, vcc, v42, v48
	v_mad_u64_u32 v[46:47], s[18:19], v29, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v43, v49, vcc
	v_readlane_b32 s4, v144, 18
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[42:43], v[18:19], 0, v[46:47]
	v_mul_lo_u32 v18, s23, v42
	v_mul_lo_u32 v43, s22, v43
	v_mad_u64_u32 v[46:47], s[18:19], s22, v42, 0
	v_add3_u32 v18, v47, v43, v18
	v_sub_u32_e32 v43, v29, v18
	v_mov_b32_e32 v47, s23
	v_sub_co_u32_e32 v45, vcc, v45, v46
	v_readlane_b32 s5, v144, 19
	s_nop 0
	v_subb_co_u32_e64 v43, s[18:19], v43, v47, vcc
	v_subrev_co_u32_e64 v46, s[18:19], s22, v45
	v_subb_co_u32_e32 v18, vcc, v29, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v43, s[18:19], 0, v43, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s23, v43
	v_cmp_le_u32_e32 vcc, s23, v18
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s22, v46
	v_cndmask_b32_e64 v29, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s22, v45
	v_cndmask_b32_e64 v46, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s23, v43
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s23, v18
	v_cndmask_b32_e64 v43, v47, v46, s[18:19]
	v_add_u32_e32 v46, 2, v42
	v_add_u32_e32 v47, 1, v42
	v_cmp_ne_u32_e64 s[18:19], 0, v43
	v_cndmask_b32_e32 v18, v29, v45, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v43, v47, v46, s[18:19]
	v_xor_b32_e32 v29, s6, v44
	v_cndmask_b32_e32 v18, v42, v43, vcc
	v_xor_b32_e32 v18, v18, v29
	v_sub_u32_e32 v18, v18, v29
	v_xor_b32_e32 v42, v18, v28
	v_ashrrev_i32_e32 v18, 31, v41
	v_and_b32_e32 v29, s35, v18
	v_and_b32_e32 v28, s34, v18
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[40:41]
	v_ashrrev_i32_e32 v40, 31, v29
	v_xor_b32_e32 v29, v40, v29
	v_ashrrev_i32_e32 v18, 31, v29
	v_xor_b32_e32 v28, v40, v28
	v_lshrrev_b32_e32 v18, 27, v18
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[18:19]
	v_alignbit_b32 v18, v29, v28, 5
	v_mul_lo_u32 v28, v42, s4
	v_readlane_b32 s4, v144, 7
	s_add_i32 m0, s4, 0x6400
	s_add_u32 s6, s42, s60
	s_addc_u32 s7, s43, s60
	v_xad_u32 v18, v18, v40, v28
	s_xor_b64 s[50:51], s[6:7], s[60:61]
	buffer_load_dword v18, s[92:95], 0 offen lds
	v_cvt_f32_u32_e32 v18, s50
	v_cvt_f32_u32_e32 v42, s51
	v_lshl_add_u64 v[44:45], v[32:33], 0, s[58:59]
	v_ashrrev_i32_e32 v40, 31, v45
	v_xor_b32_e32 v29, v40, v45
	v_fmac_f32_e32 v18, 0x4f800000, v42
	v_rcp_f32_e32 v18, v18
	v_ashrrev_i32_e32 v46, 31, v29
	v_xor_b32_e32 v28, v40, v44
	v_mov_b32_e32 v47, v46
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[46:47]
	v_mul_f32_e32 v18, 0x5f7ffffc, v18
	v_xor_b32_e32 v58, v28, v46
	v_mul_f32_e32 v28, 0x2f800000, v18
	v_trunc_f32_e32 v28, v28
	v_fmac_f32_e32 v18, 0xcf800000, v28
	v_cvt_u32_f32_e32 v18, v18
	v_cvt_u32_f32_e32 v28, v28
	s_sub_u32 s4, 0, s50
	s_subb_u32 s5, 0, s51
	v_readfirstlane_b32 s6, v18
	v_readfirstlane_b32 s19, v28
	s_mul_hi_u32 s18, s4, s6
	s_mul_i32 s22, s4, s19
	s_mul_i32 s7, s5, s6
	s_add_i32 s18, s18, s22
	s_add_i32 s18, s18, s7
	s_mul_i32 s23, s4, s6
	s_mul_i32 s22, s6, s18
	s_mul_hi_u32 s24, s6, s23
	s_mul_hi_u32 s7, s6, s18
	s_add_u32 s22, s24, s22
	s_addc_u32 s7, 0, s7
	s_mul_hi_u32 s25, s19, s23
	s_mul_i32 s23, s19, s23
	s_add_u32 s22, s22, s23
	s_mul_hi_u32 s24, s19, s18
	s_addc_u32 s7, s7, s25
	s_addc_u32 s22, s24, 0
	s_mul_i32 s18, s19, s18
	s_add_u32 s7, s7, s18
	s_addc_u32 s18, 0, s22
	s_add_u32 s6, s6, s7
	s_addc_u32 s7, s19, s18
	s_mul_i32 s18, s4, s7
	s_mul_hi_u32 s19, s4, s6
	s_add_i32 s18, s19, s18
	s_mul_i32 s5, s5, s6
	s_add_i32 s18, s18, s5
	s_mul_i32 s4, s4, s6
	s_mul_hi_u32 s19, s7, s4
	s_mul_i32 s22, s7, s4
	s_mul_i32 s24, s6, s18
	s_mul_hi_u32 s4, s6, s4
	s_mul_hi_u32 s23, s6, s18
	s_add_u32 s4, s4, s24
	s_addc_u32 s23, 0, s23
	s_add_u32 s4, s4, s22
	s_mul_hi_u32 s5, s7, s18
	s_addc_u32 s4, s23, s19
	s_addc_u32 s5, s5, 0
	s_mul_i32 s18, s7, s18
	s_add_u32 s4, s4, s18
	s_addc_u32 s5, 0, s5
	s_add_u32 s6, s6, s4
	s_addc_u32 s7, s7, s5
	v_xor_b32_e32 v47, v29, v46
	v_mad_u64_u32 v[28:29], s[18:19], v58, s7, 0
	v_mul_hi_u32 v18, v58, s6
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[28:29]
	v_mad_u64_u32 v[48:49], s[18:19], v47, s6, 0
	v_add_co_u32_e32 v18, vcc, v28, v48
	v_mad_u64_u32 v[42:43], s[18:19], v47, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v29, v49, vcc
	v_mov_b32_e32 v41, v40
	s_nop 0
	v_addc_co_u32_e32 v43, vcc, 0, v43, vcc
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[42:43]
	v_mul_lo_u32 v18, s51, v28
	v_mul_lo_u32 v29, s50, v29
	v_mad_u64_u32 v[48:49], s[18:19], s50, v28, 0
	v_add3_u32 v18, v49, v29, v18
	v_sub_u32_e32 v29, v47, v18
	v_mov_b32_e32 v42, s51
	v_sub_co_u32_e32 v43, vcc, v58, v48
	s_waitcnt vmcnt(0)
	s_nop 0
	v_subb_co_u32_e64 v29, s[18:19], v29, v42, vcc
	v_subrev_co_u32_e64 v48, s[18:19], s50, v43
	v_subb_co_u32_e32 v18, vcc, v47, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v29, s[18:19], 0, v29, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s51, v29
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s50, v48
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v43
	v_cndmask_b32_e64 v48, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s51, v29
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v29, v49, v48, s[18:19]
	v_add_u32_e32 v48, 2, v28
	v_add_u32_e32 v49, 1, v28
	v_cmp_ne_u32_e64 s[18:19], 0, v29
	v_cndmask_b32_e32 v18, v47, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v29, v49, v48, s[18:19]
	s_barrier
	v_cndmask_b32_e32 v18, v28, v29, vcc
	v_xor_b32_e32 v28, s60, v46
	v_xor_b32_e32 v18, v18, v28
	v_sub_u32_e32 v18, v18, v28
	v_lshl_add_u64 v[28:29], v[44:45], 0, v[40:41]
	v_xor_b32_e32 v44, v28, v40
	v_xad_u32 v43, v18, v40, v51
	v_xor_b32_e32 v41, v29, v40
	v_mad_u64_u32 v[28:29], s[18:19], v44, s7, 0
	v_mul_hi_u32 v18, v44, s6
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[28:29]
	v_mad_u64_u32 v[48:49], s[18:19], v41, s6, 0
	v_add_co_u32_e32 v18, vcc, v28, v48
	v_mad_u64_u32 v[46:47], s[18:19], v41, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v29, v49, vcc
	s_nop 1
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[46:47]
	v_mul_lo_u32 v18, s51, v28
	v_mul_lo_u32 v46, s50, v29
	v_mad_u64_u32 v[28:29], s[18:19], s50, v28, 0
	v_add3_u32 v18, v29, v46, v18
	v_sub_u32_e32 v29, v41, v18
	v_sub_co_u32_e32 v28, vcc, v44, v28
	s_movk_i32 s4, 0xfbf0
	s_nop 0
	v_subb_co_u32_e64 v29, s[18:19], v29, v42, vcc
	v_subrev_co_u32_e64 v44, s[18:19], s50, v28
	v_subb_co_u32_e32 v18, vcc, v41, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v46, s[22:23], 0, v29, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s51, v46
	v_subb_co_u32_e64 v29, s[18:19], v29, v42, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v44
	v_cmp_le_u32_e32 vcc, s51, v18
	s_mov_b32 s5, -1
	v_cndmask_b32_e64 v48, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v46
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v28
	v_cndmask_b32_e64 v47, v47, v48, s[22:23]
	v_subrev_co_u32_e64 v48, s[18:19], s50, v44
	s_nop 1
	v_subbrev_co_u32_e64 v29, s[18:19], 0, v29, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v47
	s_nop 1
	v_cndmask_b32_e64 v29, v46, v29, s[18:19]
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v41, v41, v46, vcc
	v_cmp_ne_u32_e32 vcc, 0, v41
	v_lshl_add_u64 v[46:47], v[32:33], 0, s[64:65]
	s_nop 0
	v_cndmask_b32_e32 v18, v18, v29, vcc
	v_cndmask_b32_e64 v29, v44, v48, s[18:19]
	v_cndmask_b32_e32 v28, v28, v29, vcc
	v_xor_b32_e32 v28, v28, v40
	v_xor_b32_e32 v18, v18, v40
	v_sub_co_u32_e32 v28, vcc, v28, v40
	v_mul_lo_u32 v29, v43, s33
	s_nop 0
	v_subb_co_u32_e32 v18, vcc, v18, v40, vcc
	v_ashrrev_i32_e32 v18, 31, v18
	v_and_b32_e32 v18, s42, v18
	v_add3_u32 v41, v18, v28, v29
	v_lshl_add_u64 v[28:29], v[22:23], 0, v[30:31]
	v_lshl_add_u64 v[48:49], v[28:29], 0, -2
	v_cmp_gt_i32_e64 s[18:19], 0, v45
	s_nop 1
	v_cndmask_b32_e64 v45, v47, v49, s[18:19]
	v_cndmask_b32_e64 v44, v46, v48, s[18:19]
	v_ashrrev_i32_e32 v48, 31, v45
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[48:49]
	v_xor_b32_e32 v49, v44, v48
	v_xor_b32_e32 v43, v45, v48
	v_mad_u64_u32 v[44:45], s[22:23], v49, s7, 0
	v_mul_hi_u32 v18, v49, s6
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[44:45]
	v_mad_u64_u32 v[60:61], s[22:23], v43, s6, 0
	v_add_co_u32_e32 v18, vcc, v44, v60
	v_mad_u64_u32 v[58:59], s[22:23], v43, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v45, v61, vcc
	s_nop 1
	v_addc_co_u32_e32 v59, vcc, 0, v59, vcc
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[58:59]
	v_mul_lo_u32 v18, s51, v44
	v_mul_lo_u32 v45, s50, v45
	v_mad_u64_u32 v[58:59], s[22:23], s50, v44, 0
	v_add3_u32 v18, v59, v45, v18
	v_sub_u32_e32 v45, v43, v18
	v_sub_co_u32_e32 v49, vcc, v49, v58
	s_nop 1
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, vcc
	v_subrev_co_u32_e64 v58, s[22:23], s50, v49
	v_subb_co_u32_e32 v18, vcc, v43, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v45, s[22:23], 0, v45, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v45
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v58
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v49
	v_cndmask_b32_e64 v58, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v45
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v45, v59, v58, s[22:23]
	v_add_u32_e32 v58, 2, v44
	v_add_u32_e32 v59, 1, v44
	v_cmp_ne_u32_e64 s[22:23], 0, v45
	v_cndmask_b32_e32 v18, v43, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v45, v59, v58, s[22:23]
	v_xor_b32_e32 v43, s60, v48
	v_cndmask_b32_e32 v18, v44, v45, vcc
	v_ashrrev_i32_e32 v44, 31, v47
	v_mov_b32_e32 v45, v44
	v_xor_b32_e32 v18, v18, v43
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[44:45]
	v_sub_u32_e32 v18, v18, v43
	v_xor_b32_e32 v45, v46, v44
	v_xad_u32 v60, v18, v40, v51
	v_xor_b32_e32 v43, v47, v44
	v_mad_u64_u32 v[46:47], s[22:23], v45, s7, 0
	v_mul_hi_u32 v18, v45, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[58:59], s[22:23], v43, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v58
	v_mad_u64_u32 v[48:49], s[22:23], v43, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v59, vcc
	s_nop 1
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[48:49]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v48, s50, v47
	v_mad_u64_u32 v[46:47], s[22:23], s50, v46, 0
	v_add3_u32 v18, v47, v48, v18
	v_sub_u32_e32 v47, v43, v18
	v_sub_co_u32_e32 v45, vcc, v45, v46
	s_nop 1
	v_subb_co_u32_e64 v46, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v47, s[22:23], s50, v45
	v_subb_co_u32_e32 v18, vcc, v43, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v48, s[24:25], 0, v46, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v48
	v_subb_co_u32_e64 v46, s[22:23], v46, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v47
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v58, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v48
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v45
	v_cndmask_b32_e64 v49, v49, v58, s[24:25]
	v_subrev_co_u32_e64 v58, s[22:23], s50, v47
	s_nop 1
	v_subbrev_co_u32_e64 v46, s[22:23], 0, v46, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	s_nop 1
	v_cndmask_b32_e64 v46, v48, v46, s[22:23]
	v_cndmask_b32_e64 v48, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v43, v43, v48, vcc
	v_cmp_ne_u32_e32 vcc, 0, v43
	v_cndmask_b32_e64 v43, v47, v58, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v18, v46, vcc
	v_cndmask_b32_e32 v43, v45, v43, vcc
	v_xor_b32_e32 v45, v18, v44
	v_xor_b32_e32 v18, v43, v44
	v_sub_co_u32_e32 v18, vcc, v18, v44
	v_lshl_add_u64 v[46:47], v[28:29], 0, -3
	s_nop 0
	v_subb_co_u32_e32 v43, vcc, v45, v44, vcc
	v_ashrrev_i32_e32 v43, 31, v43
	v_and_b32_e32 v43, s42, v43
	v_mul_lo_u32 v44, v60, s33
	v_add3_u32 v43, v43, v18, v44
	v_lshl_add_u64 v[44:45], v[32:33], 0, s[66:67]
	v_cndmask_b32_e64 v47, v45, v47, s[18:19]
	v_ashrrev_i32_e32 v48, 31, v47
	v_cndmask_b32_e64 v46, v44, v46, s[18:19]
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v62, v46, v48
	v_xor_b32_e32 v49, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v62, s7, 0
	v_mul_hi_u32 v18, v62, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[60:61], s[22:23], v49, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v60
	v_mad_u64_u32 v[58:59], s[22:23], v49, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v61, vcc
	s_nop 1
	v_addc_co_u32_e32 v59, vcc, 0, v59, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[58:59]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v47, s50, v47
	v_mad_u64_u32 v[58:59], s[22:23], s50, v46, 0
	v_add3_u32 v18, v59, v47, v18
	v_sub_u32_e32 v47, v49, v18
	v_sub_co_u32_e32 v58, vcc, v62, v58
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v59, s[22:23], s50, v58
	v_subb_co_u32_e32 v18, vcc, v49, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v47
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v60, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v59
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v58
	v_cndmask_b32_e64 v59, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v47
	v_cndmask_b32_e64 v58, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v47, v60, v59, s[22:23]
	v_add_u32_e32 v59, 2, v46
	v_add_u32_e32 v60, 1, v46
	v_cmp_ne_u32_e64 s[22:23], 0, v47
	v_cndmask_b32_e32 v18, v49, v58, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v47, v60, v59, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v46, v47, vcc
	v_xor_b32_e32 v46, s60, v48
	v_xor_b32_e32 v18, v18, v46
	v_sub_u32_e32 v18, v18, v46
	v_ashrrev_i32_e32 v46, 31, v45
	v_mov_b32_e32 v47, v46
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_xor_b32_e32 v61, v44, v46
	v_xad_u32 v60, v18, v40, v51
	v_xor_b32_e32 v47, v45, v46
	v_mad_u64_u32 v[44:45], s[22:23], v61, s7, 0
	v_mul_hi_u32 v18, v61, s6
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[44:45]
	v_mad_u64_u32 v[58:59], s[22:23], v47, s6, 0
	v_add_co_u32_e32 v18, vcc, v44, v58
	v_mad_u64_u32 v[48:49], s[22:23], v47, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v45, v59, vcc
	s_nop 1
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[48:49]
	v_mul_lo_u32 v18, s51, v44
	v_mul_lo_u32 v48, s50, v45
	v_mad_u64_u32 v[44:45], s[22:23], s50, v44, 0
	v_add3_u32 v18, v45, v48, v18
	v_sub_u32_e32 v45, v47, v18
	v_sub_co_u32_e32 v44, vcc, v61, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, vcc
	v_subrev_co_u32_e64 v48, s[22:23], s50, v44
	v_subb_co_u32_e32 v18, vcc, v47, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[24:25], 0, v45, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v49
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v58, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v48
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v49
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v44
	v_cndmask_b32_e64 v58, v58, v59, s[24:25]
	v_subrev_co_u32_e64 v59, s[22:23], s50, v48
	s_nop 1
	v_subbrev_co_u32_e64 v45, s[22:23], 0, v45, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v58
	s_nop 1
	v_cndmask_b32_e64 v45, v49, v45, s[22:23]
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v47, v47, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v47
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v45, vcc
	v_cndmask_b32_e64 v45, v48, v59, s[22:23]
	v_cndmask_b32_e32 v44, v44, v45, vcc
	v_xor_b32_e32 v45, v18, v46
	v_xor_b32_e32 v18, v44, v46
	v_sub_co_u32_e32 v18, vcc, v18, v46
	v_lshl_add_u64 v[48:49], v[28:29], 0, -4
	s_nop 0
	v_subb_co_u32_e32 v45, vcc, v45, v46, vcc
	v_lshl_add_u64 v[46:47], v[32:33], 0, s[68:69]
	v_cndmask_b32_e64 v49, v47, v49, s[18:19]
	v_ashrrev_i32_e32 v58, 31, v49
	v_cndmask_b32_e64 v48, v46, v48, s[18:19]
	v_mov_b32_e32 v59, v58
	v_ashrrev_i32_e32 v44, 31, v45
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[58:59]
	v_and_b32_e32 v44, s42, v44
	v_mul_lo_u32 v45, v60, s33
	v_xor_b32_e32 v59, v48, v58
	v_add3_u32 v44, v44, v18, v45
	v_xor_b32_e32 v45, v49, v58
	v_mad_u64_u32 v[48:49], s[22:23], v59, s7, 0
	v_mul_hi_u32 v18, v59, s6
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[64:65], s[22:23], v45, s6, 0
	v_add_co_u32_e32 v18, vcc, v48, v64
	v_mad_u64_u32 v[60:61], s[22:23], v45, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v49, v65, vcc
	s_nop 1
	v_addc_co_u32_e32 v61, vcc, 0, v61, vcc
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[60:61]
	v_mul_lo_u32 v18, s51, v48
	v_mul_lo_u32 v49, s50, v49
	v_mad_u64_u32 v[60:61], s[22:23], s50, v48, 0
	v_add3_u32 v18, v61, v49, v18
	v_sub_u32_e32 v49, v45, v18
	v_sub_co_u32_e32 v59, vcc, v59, v60
	s_nop 1
	v_subb_co_u32_e64 v49, s[22:23], v49, v42, vcc
	v_subrev_co_u32_e64 v60, s[22:23], s50, v59
	v_subb_co_u32_e32 v18, vcc, v45, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v49, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v60
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v59
	v_cndmask_b32_e64 v60, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v49
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v49, v61, v60, s[22:23]
	v_add_u32_e32 v60, 2, v48
	v_add_u32_e32 v61, 1, v48
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	v_cndmask_b32_e32 v18, v45, v59, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v49, v61, v60, s[22:23]
	v_xor_b32_e32 v45, s60, v58
	v_cndmask_b32_e32 v18, v48, v49, vcc
	v_ashrrev_i32_e32 v48, 31, v47
	v_mov_b32_e32 v49, v48
	v_xor_b32_e32 v18, v18, v45
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_sub_u32_e32 v18, v18, v45
	v_xor_b32_e32 v49, v46, v48
	v_xad_u32 v62, v18, v40, v51
	v_xor_b32_e32 v45, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v49, s7, 0
	v_mul_hi_u32 v18, v49, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[60:61], s[22:23], v45, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v60
	v_mad_u64_u32 v[58:59], s[22:23], v45, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v61, vcc
	s_nop 1
	v_addc_co_u32_e32 v59, vcc, 0, v59, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[58:59]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v58, s50, v47
	v_mad_u64_u32 v[46:47], s[22:23], s50, v46, 0
	v_add3_u32 v18, v47, v58, v18
	v_sub_u32_e32 v47, v45, v18
	v_sub_co_u32_e32 v46, vcc, v49, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v49, s[22:23], s50, v46
	v_subb_co_u32_e32 v18, vcc, v45, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v58, s[24:25], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v58
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v60, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v58
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v59, v59, v60, s[24:25]
	v_subrev_co_u32_e64 v60, s[22:23], s50, v49
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v59
	s_nop 1
	v_cndmask_b32_e64 v47, v58, v47, s[22:23]
	v_cndmask_b32_e64 v58, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v45, v45, v58, vcc
	v_cmp_ne_u32_e32 vcc, 0, v45
	v_cndmask_b32_e64 v45, v49, v60, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_cndmask_b32_e32 v45, v46, v45, vcc
	v_xor_b32_e32 v46, v18, v48
	v_xor_b32_e32 v18, v45, v48
	v_sub_co_u32_e32 v18, vcc, v18, v48
	s_nop 1
	v_subb_co_u32_e32 v45, vcc, v46, v48, vcc
	v_ashrrev_i32_e32 v45, 31, v45
	v_and_b32_e32 v45, s42, v45
	v_mul_lo_u32 v46, v62, s33
	v_add3_u32 v45, v45, v18, v46
	v_lshl_add_u64 v[46:47], v[32:33], 0, s[70:71]
	v_lshl_add_u64 v[48:49], v[28:29], 0, -5
	v_cndmask_b32_e64 v49, v47, v49, s[18:19]
	v_ashrrev_i32_e32 v58, 31, v49
	v_cndmask_b32_e64 v48, v46, v48, s[18:19]
	v_mov_b32_e32 v59, v58
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[58:59]
	v_xor_b32_e32 v62, v48, v58
	v_xor_b32_e32 v59, v49, v58
	v_mad_u64_u32 v[48:49], s[22:23], v62, s7, 0
	v_mul_hi_u32 v18, v62, s6
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[64:65], s[22:23], v59, s6, 0
	v_add_co_u32_e32 v18, vcc, v48, v64
	v_mad_u64_u32 v[60:61], s[22:23], v59, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v49, v65, vcc
	s_nop 1
	v_addc_co_u32_e32 v61, vcc, 0, v61, vcc
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[60:61]
	v_mul_lo_u32 v18, s51, v48
	v_mul_lo_u32 v49, s50, v49
	v_mad_u64_u32 v[60:61], s[22:23], s50, v48, 0
	v_add3_u32 v18, v61, v49, v18
	v_sub_u32_e32 v49, v59, v18
	v_sub_co_u32_e32 v60, vcc, v62, v60
	s_nop 1
	v_subb_co_u32_e64 v49, s[22:23], v49, v42, vcc
	v_subrev_co_u32_e64 v61, s[22:23], s50, v60
	v_subb_co_u32_e32 v18, vcc, v59, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v49, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v62, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v61
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v60
	v_cndmask_b32_e64 v61, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v49
	v_cndmask_b32_e64 v60, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v49, v62, v61, s[22:23]
	v_add_u32_e32 v61, 2, v48
	v_add_u32_e32 v62, 1, v48
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	v_cndmask_b32_e32 v18, v59, v60, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v49, v62, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v48, v49, vcc
	v_xor_b32_e32 v48, s60, v58
	v_xor_b32_e32 v18, v18, v48
	v_sub_u32_e32 v18, v18, v48
	v_ashrrev_i32_e32 v48, 31, v47
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v64, v46, v48
	v_xad_u32 v62, v18, v40, v51
	v_xor_b32_e32 v49, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v64, s7, 0
	v_mul_hi_u32 v18, v64, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[60:61], s[22:23], v49, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v60
	v_mad_u64_u32 v[58:59], s[22:23], v49, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v61, vcc
	s_nop 1
	v_addc_co_u32_e32 v59, vcc, 0, v59, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[58:59]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v58, s50, v47
	v_mad_u64_u32 v[46:47], s[22:23], s50, v46, 0
	v_add3_u32 v18, v47, v58, v18
	v_sub_u32_e32 v47, v49, v18
	v_sub_co_u32_e32 v46, vcc, v64, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v58, s[22:23], s50, v46
	v_subb_co_u32_e32 v18, vcc, v49, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v59, s[24:25], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v59
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v60, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v58
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v59
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v60, v60, v61, s[24:25]
	v_subrev_co_u32_e64 v61, s[22:23], s50, v58
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v60
	s_nop 1
	v_cndmask_b32_e64 v47, v59, v47, s[22:23]
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v49, v49, v59, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_cndmask_b32_e64 v47, v58, v61, s[22:23]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v18, v48
	v_xor_b32_e32 v18, v46, v48
	v_sub_co_u32_e32 v18, vcc, v18, v48
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v47, v48, vcc
	v_ashrrev_i32_e32 v46, 31, v47
	v_and_b32_e32 v46, s42, v46
	v_mul_lo_u32 v47, v62, s33
	v_add3_u32 v62, v46, v18, v47
	v_lshl_add_u64 v[46:47], v[32:33], 0, s[72:73]
	v_lshl_add_u64 v[48:49], v[28:29], 0, -6
	v_cndmask_b32_e64 v49, v47, v49, s[18:19]
	v_ashrrev_i32_e32 v58, 31, v49
	v_cndmask_b32_e64 v48, v46, v48, s[18:19]
	v_mov_b32_e32 v59, v58
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[58:59]
	v_xor_b32_e32 v66, v48, v58
	v_xor_b32_e32 v59, v49, v58
	v_mad_u64_u32 v[48:49], s[22:23], v66, s7, 0
	v_mul_hi_u32 v18, v66, s6
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[64:65], s[22:23], v59, s6, 0
	v_add_co_u32_e32 v18, vcc, v48, v64
	v_mad_u64_u32 v[60:61], s[22:23], v59, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v49, v65, vcc
	s_nop 1
	v_addc_co_u32_e32 v61, vcc, 0, v61, vcc
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[60:61]
	v_mul_lo_u32 v18, s51, v48
	v_mul_lo_u32 v49, s50, v49
	v_mad_u64_u32 v[60:61], s[22:23], s50, v48, 0
	v_add3_u32 v18, v61, v49, v18
	v_sub_u32_e32 v49, v59, v18
	v_sub_co_u32_e32 v60, vcc, v66, v60
	s_nop 1
	v_subb_co_u32_e64 v49, s[22:23], v49, v42, vcc
	v_subrev_co_u32_e64 v61, s[22:23], s50, v60
	v_subb_co_u32_e32 v18, vcc, v59, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v49, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v64, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v61
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v60
	v_cndmask_b32_e64 v61, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v49
	v_cndmask_b32_e64 v60, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v49, v64, v61, s[22:23]
	v_add_u32_e32 v61, 2, v48
	v_add_u32_e32 v64, 1, v48
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	v_cndmask_b32_e32 v18, v59, v60, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v49, v64, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v48, v49, vcc
	v_xor_b32_e32 v48, s60, v58
	v_xor_b32_e32 v18, v18, v48
	v_sub_u32_e32 v18, v18, v48
	v_ashrrev_i32_e32 v48, 31, v47
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v65, v46, v48
	v_xad_u32 v64, v18, v40, v51
	v_xor_b32_e32 v49, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v65, s7, 0
	v_mul_hi_u32 v18, v65, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[60:61], s[22:23], v49, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v60
	v_mad_u64_u32 v[58:59], s[22:23], v49, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v61, vcc
	s_nop 1
	v_addc_co_u32_e32 v59, vcc, 0, v59, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[58:59]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v58, s50, v47
	v_mad_u64_u32 v[46:47], s[22:23], s50, v46, 0
	v_add3_u32 v18, v47, v58, v18
	v_sub_u32_e32 v47, v49, v18
	v_sub_co_u32_e32 v46, vcc, v65, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v58, s[22:23], s50, v46
	v_subb_co_u32_e32 v18, vcc, v49, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v59, s[24:25], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v59
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v60, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v58
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v59
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v60, v60, v61, s[24:25]
	v_subrev_co_u32_e64 v61, s[22:23], s50, v58
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v60
	s_nop 1
	v_cndmask_b32_e64 v47, v59, v47, s[22:23]
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v49, v49, v59, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_cndmask_b32_e64 v47, v58, v61, s[22:23]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v18, v48
	v_xor_b32_e32 v18, v46, v48
	v_sub_co_u32_e32 v18, vcc, v18, v48
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v47, v48, vcc
	v_ashrrev_i32_e32 v46, 31, v47
	v_and_b32_e32 v46, s42, v46
	v_mul_lo_u32 v47, v64, s33
	v_add3_u32 v66, v46, v18, v47
	v_lshl_add_u64 v[46:47], v[32:33], 0, s[74:75]
	v_lshl_add_u64 v[48:49], v[28:29], 0, -7
	v_cndmask_b32_e64 v49, v47, v49, s[18:19]
	v_ashrrev_i32_e32 v58, 31, v49
	v_cndmask_b32_e64 v48, v46, v48, s[18:19]
	v_mov_b32_e32 v59, v58
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[58:59]
	v_xor_b32_e32 v67, v48, v58
	v_xor_b32_e32 v59, v49, v58
	v_mad_u64_u32 v[48:49], s[22:23], v67, s7, 0
	v_mul_hi_u32 v18, v67, s6
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[64:65], s[22:23], v59, s6, 0
	v_add_co_u32_e32 v18, vcc, v48, v64
	v_mad_u64_u32 v[60:61], s[22:23], v59, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v49, v65, vcc
	s_nop 1
	v_addc_co_u32_e32 v61, vcc, 0, v61, vcc
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[60:61]
	v_mul_lo_u32 v18, s51, v48
	v_mul_lo_u32 v49, s50, v49
	v_mad_u64_u32 v[60:61], s[22:23], s50, v48, 0
	v_add3_u32 v18, v61, v49, v18
	v_sub_u32_e32 v49, v59, v18
	v_sub_co_u32_e32 v60, vcc, v67, v60
	s_nop 1
	v_subb_co_u32_e64 v49, s[22:23], v49, v42, vcc
	v_subrev_co_u32_e64 v61, s[22:23], s50, v60
	v_subb_co_u32_e32 v18, vcc, v59, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v49, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v64, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v61
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v60
	v_cndmask_b32_e64 v61, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v49
	v_cndmask_b32_e64 v60, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v49, v64, v61, s[22:23]
	v_add_u32_e32 v61, 2, v48
	v_add_u32_e32 v64, 1, v48
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	v_cndmask_b32_e32 v18, v59, v60, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v49, v64, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v48, v49, vcc
	v_xor_b32_e32 v48, s60, v58
	v_xor_b32_e32 v18, v18, v48
	v_sub_u32_e32 v18, v18, v48
	v_ashrrev_i32_e32 v48, 31, v47
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v65, v46, v48
	v_xad_u32 v64, v18, v40, v51
	v_xor_b32_e32 v49, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v65, s7, 0
	v_mul_hi_u32 v18, v65, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[60:61], s[22:23], v49, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v60
	v_mad_u64_u32 v[58:59], s[22:23], v49, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v61, vcc
	s_nop 1
	v_addc_co_u32_e32 v59, vcc, 0, v59, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[58:59]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v58, s50, v47
	v_mad_u64_u32 v[46:47], s[22:23], s50, v46, 0
	v_add3_u32 v18, v47, v58, v18
	v_sub_u32_e32 v47, v49, v18
	v_sub_co_u32_e32 v46, vcc, v65, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v58, s[22:23], s50, v46
	v_subb_co_u32_e32 v18, vcc, v49, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v59, s[24:25], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v59
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v60, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v58
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v59
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v60, v60, v61, s[24:25]
	v_subrev_co_u32_e64 v61, s[22:23], s50, v58
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v60
	s_nop 1
	v_cndmask_b32_e64 v47, v59, v47, s[22:23]
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v49, v49, v59, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_cndmask_b32_e64 v47, v58, v61, s[22:23]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v18, v48
	v_xor_b32_e32 v18, v46, v48
	v_sub_co_u32_e32 v18, vcc, v18, v48
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v47, v48, vcc
	v_ashrrev_i32_e32 v46, 31, v47
	v_and_b32_e32 v46, s42, v46
	v_mul_lo_u32 v47, v64, s33
	v_add3_u32 v67, v46, v18, v47
	v_lshl_add_u64 v[46:47], v[32:33], 0, s[76:77]
	v_lshl_add_u64 v[48:49], v[28:29], 0, -8
	v_cndmask_b32_e64 v49, v47, v49, s[18:19]
	v_ashrrev_i32_e32 v58, 31, v49
	v_cndmask_b32_e64 v48, v46, v48, s[18:19]
	v_mov_b32_e32 v59, v58
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[58:59]
	v_xor_b32_e32 v68, v48, v58
	v_xor_b32_e32 v59, v49, v58
	v_mad_u64_u32 v[48:49], s[22:23], v68, s7, 0
	v_mul_hi_u32 v18, v68, s6
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[64:65], s[22:23], v59, s6, 0
	v_add_co_u32_e32 v18, vcc, v48, v64
	v_mad_u64_u32 v[60:61], s[22:23], v59, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v49, v65, vcc
	s_nop 1
	v_addc_co_u32_e32 v61, vcc, 0, v61, vcc
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[60:61]
	v_mul_lo_u32 v18, s51, v48
	v_mul_lo_u32 v49, s50, v49
	v_mad_u64_u32 v[60:61], s[22:23], s50, v48, 0
	v_add3_u32 v18, v61, v49, v18
	v_sub_u32_e32 v49, v59, v18
	v_sub_co_u32_e32 v60, vcc, v68, v60
	s_nop 1
	v_subb_co_u32_e64 v49, s[22:23], v49, v42, vcc
	v_subrev_co_u32_e64 v61, s[22:23], s50, v60
	v_subb_co_u32_e32 v18, vcc, v59, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v49, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v64, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v61
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v60
	v_cndmask_b32_e64 v61, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v49
	v_cndmask_b32_e64 v60, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v49, v64, v61, s[22:23]
	v_add_u32_e32 v61, 2, v48
	v_add_u32_e32 v64, 1, v48
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	v_cndmask_b32_e32 v18, v59, v60, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v49, v64, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v48, v49, vcc
	v_xor_b32_e32 v48, s60, v58
	v_xor_b32_e32 v18, v18, v48
	v_sub_u32_e32 v18, v18, v48
	v_ashrrev_i32_e32 v48, 31, v47
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v65, v46, v48
	v_xad_u32 v64, v18, v40, v51
	v_xor_b32_e32 v49, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v65, s7, 0
	v_mul_hi_u32 v18, v65, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[60:61], s[22:23], v49, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v60
	v_mad_u64_u32 v[58:59], s[22:23], v49, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v61, vcc
	s_nop 1
	v_addc_co_u32_e32 v59, vcc, 0, v59, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[58:59]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v58, s50, v47
	v_mad_u64_u32 v[46:47], s[22:23], s50, v46, 0
	v_add3_u32 v18, v47, v58, v18
	v_sub_u32_e32 v47, v49, v18
	v_sub_co_u32_e32 v46, vcc, v65, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v58, s[22:23], s50, v46
	v_subb_co_u32_e32 v18, vcc, v49, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v59, s[24:25], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v59
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v60, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v58
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v61, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v59
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v60, v60, v61, s[24:25]
	v_subrev_co_u32_e64 v61, s[22:23], s50, v58
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v60
	s_nop 1
	v_cndmask_b32_e64 v47, v59, v47, s[22:23]
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v49, v49, v59, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_cndmask_b32_e64 v47, v58, v61, s[22:23]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v18, v48
	v_xor_b32_e32 v18, v46, v48
	v_sub_co_u32_e32 v18, vcc, v18, v48
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v47, v48, vcc
	v_ashrrev_i32_e32 v46, 31, v47
	v_and_b32_e32 v46, s42, v46
	v_mul_lo_u32 v47, v64, s33
	v_add3_u32 v18, v46, v18, v47
	buffer_load_ubyte v58, v41, s[28:31], 0 offen
	buffer_load_ubyte v60, v43, s[28:31], 0 offen
	buffer_load_ubyte v59, v44, s[28:31], 0 offen
	buffer_load_ubyte v61, v45, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v62, v62, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v65, v66, s[28:31], 0 offen
	buffer_load_ubyte v64, v67, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v66, v18, s[28:31], 0 offen
	v_lshl_add_u64 v[44:45], v[32:33], 0, s[78:79]
	v_lshl_add_u64 v[46:47], v[28:29], 0, -9
	v_cndmask_b32_e64 v47, v45, v47, s[18:19]
	v_ashrrev_i32_e32 v48, 31, v47
	v_cndmask_b32_e64 v46, v44, v46, s[18:19]
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v43, v46, v48
	v_xor_b32_e32 v41, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v43, s7, 0
	v_mul_hi_u32 v18, v43, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[70:71], s[22:23], v41, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v70
	v_mad_u64_u32 v[68:69], s[22:23], v41, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v71, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[68:69]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v47, s50, v47
	v_mad_u64_u32 v[68:69], s[22:23], s50, v46, 0
	v_add3_u32 v18, v69, v47, v18
	v_sub_u32_e32 v47, v41, v18
	v_sub_co_u32_e32 v43, vcc, v43, v68
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v49, s[22:23], s50, v43
	v_subb_co_u32_e32 v18, vcc, v41, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v47
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v67, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v49
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v43
	v_cndmask_b32_e64 v49, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v47
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v47, v67, v49, s[22:23]
	v_add_u32_e32 v49, 2, v46
	v_add_u32_e32 v67, 1, v46
	v_cmp_ne_u32_e64 s[22:23], 0, v47
	v_cndmask_b32_e32 v18, v41, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v47, v67, v49, s[22:23]
	v_xor_b32_e32 v41, s60, v48
	v_cndmask_b32_e32 v18, v46, v47, vcc
	v_ashrrev_i32_e32 v46, 31, v45
	v_mov_b32_e32 v47, v46
	v_xor_b32_e32 v18, v18, v41
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_sub_u32_e32 v18, v18, v41
	v_xor_b32_e32 v47, v44, v46
	v_xad_u32 v43, v18, v40, v51
	v_xor_b32_e32 v41, v45, v46
	v_mad_u64_u32 v[44:45], s[22:23], v47, s7, 0
	v_mul_hi_u32 v18, v47, s6
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[44:45]
	v_mad_u64_u32 v[68:69], s[22:23], v41, s6, 0
	v_add_co_u32_e32 v18, vcc, v44, v68
	v_mad_u64_u32 v[48:49], s[22:23], v41, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v45, v69, vcc
	v_mul_lo_u32 v43, v43, s33
	s_nop 0
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[48:49]
	v_mul_lo_u32 v18, s51, v44
	v_mul_lo_u32 v48, s50, v45
	v_mad_u64_u32 v[44:45], s[22:23], s50, v44, 0
	v_add3_u32 v18, v45, v48, v18
	v_sub_u32_e32 v45, v41, v18
	v_sub_co_u32_e32 v44, vcc, v47, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, vcc
	v_subrev_co_u32_e64 v47, s[22:23], s50, v44
	v_subb_co_u32_e32 v18, vcc, v41, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v48, s[24:25], 0, v45, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v48
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v47
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v67, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v48
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v44
	v_cndmask_b32_e64 v49, v49, v67, s[24:25]
	v_subrev_co_u32_e64 v67, s[22:23], s50, v47
	s_nop 1
	v_subbrev_co_u32_e64 v45, s[22:23], 0, v45, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	s_nop 1
	v_cndmask_b32_e64 v45, v48, v45, s[22:23]
	v_cndmask_b32_e64 v48, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v41, v41, v48, vcc
	v_cmp_ne_u32_e32 vcc, 0, v41
	v_cndmask_b32_e64 v41, v47, v67, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v18, v45, vcc
	v_cndmask_b32_e32 v41, v44, v41, vcc
	v_xor_b32_e32 v44, v18, v46
	v_xor_b32_e32 v18, v41, v46
	v_sub_co_u32_e32 v18, vcc, v18, v46
	s_nop 1
	v_subb_co_u32_e32 v41, vcc, v44, v46, vcc
	v_lshl_add_u64 v[44:45], v[32:33], 0, s[80:81]
	v_lshl_add_u64 v[46:47], v[28:29], 0, -10
	v_cndmask_b32_e64 v47, v45, v47, s[18:19]
	v_ashrrev_i32_e32 v48, 31, v47
	v_cndmask_b32_e64 v46, v44, v46, s[18:19]
	v_mov_b32_e32 v49, v48
	v_ashrrev_i32_e32 v41, 31, v41
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_and_b32_e32 v41, s42, v41
	v_xor_b32_e32 v49, v46, v48
	v_add3_u32 v41, v41, v18, v43
	v_xor_b32_e32 v43, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v49, s7, 0
	v_mul_hi_u32 v18, v49, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[70:71], s[22:23], v43, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v70
	v_mad_u64_u32 v[68:69], s[22:23], v43, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v71, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[68:69]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v47, s50, v47
	v_mad_u64_u32 v[68:69], s[22:23], s50, v46, 0
	v_add3_u32 v18, v69, v47, v18
	v_sub_u32_e32 v47, v43, v18
	v_sub_co_u32_e32 v49, vcc, v49, v68
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v67, s[22:23], s50, v49
	v_subb_co_u32_e32 v18, vcc, v43, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v47
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v68, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v67
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v49
	v_cndmask_b32_e64 v67, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v47
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v47, v68, v67, s[22:23]
	v_add_u32_e32 v67, 2, v46
	v_add_u32_e32 v68, 1, v46
	v_cmp_ne_u32_e64 s[22:23], 0, v47
	v_cndmask_b32_e32 v18, v43, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v47, v68, v67, s[22:23]
	v_xor_b32_e32 v43, s60, v48
	v_cndmask_b32_e32 v18, v46, v47, vcc
	v_ashrrev_i32_e32 v46, 31, v45
	v_mov_b32_e32 v47, v46
	v_xor_b32_e32 v18, v18, v43
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_sub_u32_e32 v18, v18, v43
	v_xor_b32_e32 v47, v44, v46
	v_xad_u32 v67, v18, v40, v51
	v_xor_b32_e32 v43, v45, v46
	v_mad_u64_u32 v[44:45], s[22:23], v47, s7, 0
	v_mul_hi_u32 v18, v47, s6
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[44:45]
	v_mad_u64_u32 v[68:69], s[22:23], v43, s6, 0
	v_add_co_u32_e32 v18, vcc, v44, v68
	v_mad_u64_u32 v[48:49], s[22:23], v43, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v45, v69, vcc
	s_nop 1
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[48:49]
	v_mul_lo_u32 v18, s51, v44
	v_mul_lo_u32 v48, s50, v45
	v_mad_u64_u32 v[44:45], s[22:23], s50, v44, 0
	v_add3_u32 v18, v45, v48, v18
	v_sub_u32_e32 v45, v43, v18
	v_sub_co_u32_e32 v44, vcc, v47, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, vcc
	v_subrev_co_u32_e64 v47, s[22:23], s50, v44
	v_subb_co_u32_e32 v18, vcc, v43, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v48, s[24:25], 0, v45, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v48
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v47
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v48
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v44
	v_cndmask_b32_e64 v49, v49, v68, s[24:25]
	v_subrev_co_u32_e64 v68, s[22:23], s50, v47
	s_nop 1
	v_subbrev_co_u32_e64 v45, s[22:23], 0, v45, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	s_nop 1
	v_cndmask_b32_e64 v45, v48, v45, s[22:23]
	v_cndmask_b32_e64 v48, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v43, v43, v48, vcc
	v_cmp_ne_u32_e32 vcc, 0, v43
	v_cndmask_b32_e64 v43, v47, v68, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v18, v45, vcc
	v_cndmask_b32_e32 v43, v44, v43, vcc
	v_xor_b32_e32 v44, v18, v46
	v_xor_b32_e32 v18, v43, v46
	v_sub_co_u32_e32 v18, vcc, v18, v46
	s_nop 1
	v_subb_co_u32_e32 v43, vcc, v44, v46, vcc
	v_ashrrev_i32_e32 v43, 31, v43
	v_and_b32_e32 v43, s42, v43
	v_mul_lo_u32 v44, v67, s33
	v_add3_u32 v43, v43, v18, v44
	v_lshl_add_u64 v[44:45], v[32:33], 0, s[82:83]
	v_lshl_add_u64 v[46:47], v[28:29], 0, -11
	v_cndmask_b32_e64 v47, v45, v47, s[18:19]
	v_ashrrev_i32_e32 v48, 31, v47
	v_cndmask_b32_e64 v46, v44, v46, s[18:19]
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v67, v46, v48
	v_xor_b32_e32 v49, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v67, s7, 0
	v_mul_hi_u32 v18, v67, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[70:71], s[22:23], v49, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v70
	v_mad_u64_u32 v[68:69], s[22:23], v49, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v71, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[68:69]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v47, s50, v47
	v_mad_u64_u32 v[68:69], s[22:23], s50, v46, 0
	v_add3_u32 v18, v69, v47, v18
	v_sub_u32_e32 v47, v49, v18
	v_sub_co_u32_e32 v67, vcc, v67, v68
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v68, s[22:23], s50, v67
	v_subb_co_u32_e32 v18, vcc, v49, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v47
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v68
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v67
	v_cndmask_b32_e64 v68, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v47
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v47, v69, v68, s[22:23]
	v_add_u32_e32 v68, 2, v46
	v_add_u32_e32 v69, 1, v46
	v_cmp_ne_u32_e64 s[22:23], 0, v47
	v_cndmask_b32_e32 v18, v49, v67, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v47, v69, v68, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v46, v47, vcc
	v_xor_b32_e32 v46, s60, v48
	v_xor_b32_e32 v18, v18, v46
	v_sub_u32_e32 v18, v18, v46
	v_ashrrev_i32_e32 v46, 31, v45
	v_mov_b32_e32 v47, v46
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_xor_b32_e32 v70, v44, v46
	v_xad_u32 v67, v18, v40, v51
	v_xor_b32_e32 v47, v45, v46
	v_mad_u64_u32 v[44:45], s[22:23], v70, s7, 0
	v_mul_hi_u32 v18, v70, s6
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[44:45]
	v_mad_u64_u32 v[68:69], s[22:23], v47, s6, 0
	v_add_co_u32_e32 v18, vcc, v44, v68
	v_mad_u64_u32 v[48:49], s[22:23], v47, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v45, v69, vcc
	s_nop 1
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[48:49]
	v_mul_lo_u32 v18, s51, v44
	v_mul_lo_u32 v48, s50, v45
	v_mad_u64_u32 v[44:45], s[22:23], s50, v44, 0
	v_add3_u32 v18, v45, v48, v18
	v_sub_u32_e32 v45, v47, v18
	v_sub_co_u32_e32 v44, vcc, v70, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, vcc
	v_subrev_co_u32_e64 v48, s[22:23], s50, v44
	v_subb_co_u32_e32 v18, vcc, v47, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[24:25], 0, v45, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v49
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v48
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v49
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v44
	v_cndmask_b32_e64 v68, v68, v69, s[24:25]
	v_subrev_co_u32_e64 v69, s[22:23], s50, v48
	s_nop 1
	v_subbrev_co_u32_e64 v45, s[22:23], 0, v45, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v68
	s_nop 1
	v_cndmask_b32_e64 v45, v49, v45, s[22:23]
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v47, v47, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v47
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v45, vcc
	v_cndmask_b32_e64 v45, v48, v69, s[22:23]
	v_cndmask_b32_e32 v44, v44, v45, vcc
	v_xor_b32_e32 v45, v18, v46
	v_xor_b32_e32 v18, v44, v46
	v_sub_co_u32_e32 v18, vcc, v18, v46
	v_lshl_add_u64 v[48:49], v[28:29], 0, -12
	s_nop 0
	v_subb_co_u32_e32 v45, vcc, v45, v46, vcc
	v_lshl_add_u64 v[46:47], v[32:33], 0, s[84:85]
	v_cndmask_b32_e64 v49, v47, v49, s[18:19]
	v_ashrrev_i32_e32 v68, 31, v49
	v_cndmask_b32_e64 v48, v46, v48, s[18:19]
	v_mov_b32_e32 v69, v68
	v_ashrrev_i32_e32 v44, 31, v45
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[68:69]
	v_and_b32_e32 v44, s42, v44
	v_mul_lo_u32 v45, v67, s33
	v_xor_b32_e32 v67, v48, v68
	v_add3_u32 v44, v44, v18, v45
	v_xor_b32_e32 v45, v49, v68
	v_mad_u64_u32 v[48:49], s[22:23], v67, s7, 0
	v_mul_hi_u32 v18, v67, s6
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[72:73], s[22:23], v45, s6, 0
	v_add_co_u32_e32 v18, vcc, v48, v72
	v_mad_u64_u32 v[70:71], s[22:23], v45, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v49, v73, vcc
	s_nop 1
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[70:71]
	v_mul_lo_u32 v18, s51, v48
	v_mul_lo_u32 v49, s50, v49
	v_mad_u64_u32 v[70:71], s[22:23], s50, v48, 0
	v_add3_u32 v18, v71, v49, v18
	v_sub_u32_e32 v49, v45, v18
	v_sub_co_u32_e32 v67, vcc, v67, v70
	s_nop 1
	v_subb_co_u32_e64 v49, s[22:23], v49, v42, vcc
	v_subrev_co_u32_e64 v69, s[22:23], s50, v67
	v_subb_co_u32_e32 v18, vcc, v45, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v49, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v69
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v67
	v_cndmask_b32_e64 v69, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v49
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v49, v70, v69, s[22:23]
	v_add_u32_e32 v69, 2, v48
	v_add_u32_e32 v70, 1, v48
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	v_cndmask_b32_e32 v18, v45, v67, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v49, v70, v69, s[22:23]
	v_xor_b32_e32 v45, s60, v68
	v_cndmask_b32_e32 v18, v48, v49, vcc
	v_ashrrev_i32_e32 v48, 31, v47
	v_mov_b32_e32 v49, v48
	v_xor_b32_e32 v18, v18, v45
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_sub_u32_e32 v18, v18, v45
	v_xor_b32_e32 v49, v46, v48
	v_xad_u32 v67, v18, v40, v51
	v_xor_b32_e32 v45, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v49, s7, 0
	v_mul_hi_u32 v18, v49, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[70:71], s[22:23], v45, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v70
	v_mad_u64_u32 v[68:69], s[22:23], v45, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v71, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[68:69]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v68, s50, v47
	v_mad_u64_u32 v[46:47], s[22:23], s50, v46, 0
	v_add3_u32 v18, v47, v68, v18
	v_sub_u32_e32 v47, v45, v18
	v_sub_co_u32_e32 v46, vcc, v49, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v49, s[22:23], s50, v46
	v_subb_co_u32_e32 v18, vcc, v45, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v68, s[24:25], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v68
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v68
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v69, v69, v70, s[24:25]
	v_subrev_co_u32_e64 v70, s[22:23], s50, v49
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v69
	s_nop 1
	v_cndmask_b32_e64 v47, v68, v47, s[22:23]
	v_cndmask_b32_e64 v68, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v45, v45, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v45
	v_cndmask_b32_e64 v45, v49, v70, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_cndmask_b32_e32 v45, v46, v45, vcc
	v_xor_b32_e32 v46, v18, v48
	v_xor_b32_e32 v18, v45, v48
	v_sub_co_u32_e32 v18, vcc, v18, v48
	s_nop 1
	v_subb_co_u32_e32 v45, vcc, v46, v48, vcc
	v_ashrrev_i32_e32 v45, 31, v45
	v_and_b32_e32 v45, s42, v45
	v_mul_lo_u32 v46, v67, s33
	v_add3_u32 v45, v45, v18, v46
	v_lshl_add_u64 v[46:47], v[32:33], 0, s[86:87]
	v_lshl_add_u64 v[48:49], v[28:29], 0, -13
	v_cndmask_b32_e64 v49, v47, v49, s[18:19]
	v_ashrrev_i32_e32 v68, 31, v49
	v_cndmask_b32_e64 v48, v46, v48, s[18:19]
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[68:69]
	v_xor_b32_e32 v69, v48, v68
	v_xor_b32_e32 v67, v49, v68
	v_mad_u64_u32 v[48:49], s[22:23], v69, s7, 0
	v_mul_hi_u32 v18, v69, s6
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[72:73], s[22:23], v67, s6, 0
	v_add_co_u32_e32 v18, vcc, v48, v72
	v_mad_u64_u32 v[70:71], s[22:23], v67, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v49, v73, vcc
	s_nop 1
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[70:71]
	v_mul_lo_u32 v18, s51, v48
	v_mul_lo_u32 v49, s50, v49
	v_mad_u64_u32 v[70:71], s[22:23], s50, v48, 0
	v_add3_u32 v18, v71, v49, v18
	v_sub_u32_e32 v49, v67, v18
	v_sub_co_u32_e32 v69, vcc, v69, v70
	s_nop 1
	v_subb_co_u32_e64 v49, s[22:23], v49, v42, vcc
	v_subrev_co_u32_e64 v70, s[22:23], s50, v69
	v_subb_co_u32_e32 v18, vcc, v67, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v49, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v70
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v69
	v_cndmask_b32_e64 v70, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v49
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v49, v71, v70, s[22:23]
	v_add_u32_e32 v70, 2, v48
	v_add_u32_e32 v71, 1, v48
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	v_cndmask_b32_e32 v18, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v49, v71, v70, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v48, v49, vcc
	v_xor_b32_e32 v48, s60, v68
	v_xor_b32_e32 v18, v18, v48
	v_sub_u32_e32 v18, v18, v48
	v_ashrrev_i32_e32 v48, 31, v47
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v72, v46, v48
	v_xad_u32 v67, v18, v40, v51
	v_xor_b32_e32 v49, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v72, s7, 0
	v_mul_hi_u32 v18, v72, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[70:71], s[22:23], v49, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v70
	v_mad_u64_u32 v[68:69], s[22:23], v49, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v71, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[68:69]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v68, s50, v47
	v_mad_u64_u32 v[46:47], s[22:23], s50, v46, 0
	v_add3_u32 v18, v47, v68, v18
	v_sub_u32_e32 v47, v49, v18
	v_sub_co_u32_e32 v46, vcc, v72, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v68, s[22:23], s50, v46
	v_subb_co_u32_e32 v18, vcc, v49, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[24:25], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v69
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v68
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v69
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v70, v70, v71, s[24:25]
	v_subrev_co_u32_e64 v71, s[22:23], s50, v68
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v70
	s_nop 1
	v_cndmask_b32_e64 v47, v69, v47, s[22:23]
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v49, v49, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_cndmask_b32_e64 v47, v68, v71, s[22:23]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v18, v48
	v_xor_b32_e32 v18, v46, v48
	v_sub_co_u32_e32 v18, vcc, v18, v48
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v47, v48, vcc
	v_ashrrev_i32_e32 v46, 31, v47
	v_and_b32_e32 v46, s42, v46
	v_mul_lo_u32 v47, v67, s33
	v_add3_u32 v74, v46, v18, v47
	v_lshl_add_u64 v[46:47], v[32:33], 0, s[88:89]
	v_lshl_add_u64 v[48:49], v[28:29], 0, -14
	v_cndmask_b32_e64 v49, v47, v49, s[18:19]
	v_ashrrev_i32_e32 v68, 31, v49
	v_cndmask_b32_e64 v48, v46, v48, s[18:19]
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[68:69]
	v_xor_b32_e32 v69, v48, v68
	v_xor_b32_e32 v67, v49, v68
	v_mad_u64_u32 v[48:49], s[22:23], v69, s7, 0
	v_mul_hi_u32 v18, v69, s6
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[72:73], s[22:23], v67, s6, 0
	v_add_co_u32_e32 v18, vcc, v48, v72
	v_mad_u64_u32 v[70:71], s[22:23], v67, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v49, v73, vcc
	s_nop 1
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[70:71]
	v_mul_lo_u32 v18, s51, v48
	v_mul_lo_u32 v49, s50, v49
	v_mad_u64_u32 v[70:71], s[22:23], s50, v48, 0
	v_add3_u32 v18, v71, v49, v18
	v_sub_u32_e32 v49, v67, v18
	v_sub_co_u32_e32 v69, vcc, v69, v70
	s_nop 1
	v_subb_co_u32_e64 v49, s[22:23], v49, v42, vcc
	v_subrev_co_u32_e64 v70, s[22:23], s50, v69
	v_subb_co_u32_e32 v18, vcc, v67, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v49, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v70
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v69
	v_cndmask_b32_e64 v70, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v49
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v49, v71, v70, s[22:23]
	v_add_u32_e32 v70, 2, v48
	v_add_u32_e32 v71, 1, v48
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	v_cndmask_b32_e32 v18, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v49, v71, v70, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v48, v49, vcc
	v_xor_b32_e32 v48, s60, v68
	v_xor_b32_e32 v18, v18, v48
	v_sub_u32_e32 v18, v18, v48
	v_ashrrev_i32_e32 v48, 31, v47
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v72, v46, v48
	v_xad_u32 v67, v18, v40, v51
	v_xor_b32_e32 v49, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v72, s7, 0
	v_mul_hi_u32 v18, v72, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[70:71], s[22:23], v49, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v70
	v_mad_u64_u32 v[68:69], s[22:23], v49, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v71, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[68:69]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v68, s50, v47
	v_mad_u64_u32 v[46:47], s[22:23], s50, v46, 0
	v_add3_u32 v18, v47, v68, v18
	v_sub_u32_e32 v47, v49, v18
	v_sub_co_u32_e32 v46, vcc, v72, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v68, s[22:23], s50, v46
	v_subb_co_u32_e32 v18, vcc, v49, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[24:25], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v69
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v68
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v69
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v70, v70, v71, s[24:25]
	v_subrev_co_u32_e64 v71, s[22:23], s50, v68
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v70
	s_nop 1
	v_cndmask_b32_e64 v47, v69, v47, s[22:23]
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v49, v49, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_cndmask_b32_e64 v47, v68, v71, s[22:23]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v18, v48
	v_xor_b32_e32 v18, v46, v48
	v_sub_co_u32_e32 v18, vcc, v18, v48
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v47, v48, vcc
	v_ashrrev_i32_e32 v46, 31, v47
	v_and_b32_e32 v46, s42, v46
	v_mul_lo_u32 v47, v67, s33
	v_add3_u32 v75, v46, v18, v47
	v_lshl_add_u64 v[46:47], v[32:33], 0, s[90:91]
	v_lshl_add_u64 v[48:49], v[28:29], 0, -15
	v_cndmask_b32_e64 v49, v47, v49, s[18:19]
	v_ashrrev_i32_e32 v68, 31, v49
	v_cndmask_b32_e64 v48, v46, v48, s[18:19]
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[68:69]
	v_xor_b32_e32 v69, v48, v68
	v_xor_b32_e32 v67, v49, v68
	v_mad_u64_u32 v[48:49], s[22:23], v69, s7, 0
	v_mul_hi_u32 v18, v69, s6
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[72:73], s[22:23], v67, s6, 0
	v_add_co_u32_e32 v18, vcc, v48, v72
	v_mad_u64_u32 v[70:71], s[22:23], v67, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v49, v73, vcc
	s_nop 1
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[70:71]
	v_mul_lo_u32 v18, s51, v48
	v_mul_lo_u32 v49, s50, v49
	v_mad_u64_u32 v[70:71], s[22:23], s50, v48, 0
	v_add3_u32 v18, v71, v49, v18
	v_sub_u32_e32 v49, v67, v18
	v_sub_co_u32_e32 v69, vcc, v69, v70
	s_nop 1
	v_subb_co_u32_e64 v49, s[22:23], v49, v42, vcc
	v_subrev_co_u32_e64 v70, s[22:23], s50, v69
	v_subb_co_u32_e32 v18, vcc, v67, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v49, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v70
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v69
	v_cndmask_b32_e64 v70, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v49
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v49, v71, v70, s[22:23]
	v_add_u32_e32 v70, 2, v48
	v_add_u32_e32 v71, 1, v48
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	v_cndmask_b32_e32 v18, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v49, v71, v70, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v48, v49, vcc
	v_xor_b32_e32 v48, s60, v68
	v_xor_b32_e32 v18, v18, v48
	v_sub_u32_e32 v18, v18, v48
	v_ashrrev_i32_e32 v48, 31, v47
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v72, v46, v48
	v_xad_u32 v67, v18, v40, v51
	v_xor_b32_e32 v49, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v72, s7, 0
	v_mul_hi_u32 v18, v72, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[70:71], s[22:23], v49, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v70
	v_mad_u64_u32 v[68:69], s[22:23], v49, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v71, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[68:69]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v68, s50, v47
	v_mad_u64_u32 v[46:47], s[22:23], s50, v46, 0
	v_add3_u32 v18, v47, v68, v18
	v_sub_u32_e32 v47, v49, v18
	v_sub_co_u32_e32 v46, vcc, v72, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v68, s[22:23], s50, v46
	v_subb_co_u32_e32 v18, vcc, v49, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[24:25], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v69
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v68
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v69
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v70, v70, v71, s[24:25]
	v_subrev_co_u32_e64 v71, s[22:23], s50, v68
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v70
	s_nop 1
	v_cndmask_b32_e64 v47, v69, v47, s[22:23]
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v49, v49, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_cndmask_b32_e64 v47, v68, v71, s[22:23]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v18, v48
	v_xor_b32_e32 v18, v46, v48
	v_sub_co_u32_e32 v18, vcc, v18, v48
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v47, v48, vcc
	v_ashrrev_i32_e32 v46, 31, v47
	v_and_b32_e32 v46, s42, v46
	v_mul_lo_u32 v47, v67, s33
	v_add3_u32 v76, v46, v18, v47
	v_lshl_add_u64 v[46:47], v[32:33], 0, s[20:21]
	v_lshl_add_u64 v[48:49], v[28:29], 0, -16
	v_cndmask_b32_e64 v49, v47, v49, s[18:19]
	v_ashrrev_i32_e32 v68, 31, v49
	v_cndmask_b32_e64 v48, v46, v48, s[18:19]
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[68:69]
	v_xor_b32_e32 v69, v48, v68
	v_xor_b32_e32 v67, v49, v68
	v_mad_u64_u32 v[48:49], s[18:19], v69, s7, 0
	v_mul_hi_u32 v18, v69, s6
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[72:73], s[18:19], v67, s6, 0
	v_add_co_u32_e32 v18, vcc, v48, v72
	v_mad_u64_u32 v[70:71], s[18:19], v67, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v49, v73, vcc
	s_nop 1
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[70:71]
	v_mul_lo_u32 v18, s51, v48
	v_mul_lo_u32 v49, s50, v49
	v_mad_u64_u32 v[70:71], s[18:19], s50, v48, 0
	v_add3_u32 v18, v71, v49, v18
	v_sub_u32_e32 v49, v67, v18
	v_sub_co_u32_e32 v69, vcc, v69, v70
	s_nop 1
	v_subb_co_u32_e64 v49, s[18:19], v49, v42, vcc
	v_subrev_co_u32_e64 v70, s[18:19], s50, v69
	v_subb_co_u32_e32 v18, vcc, v67, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[18:19], 0, v49, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s51, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s50, v70
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v69
	v_cndmask_b32_e64 v70, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s51, v49
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v49, v71, v70, s[18:19]
	v_add_u32_e32 v70, 2, v48
	v_add_u32_e32 v71, 1, v48
	v_cmp_ne_u32_e64 s[18:19], 0, v49
	v_cndmask_b32_e32 v18, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v49, v71, v70, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v18, v48, v49, vcc
	v_xor_b32_e32 v48, s60, v68
	v_xor_b32_e32 v18, v18, v48
	v_sub_u32_e32 v18, v18, v48
	v_ashrrev_i32_e32 v48, 31, v47
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v67, v46, v48
	v_xad_u32 v40, v18, v40, v51
	v_xor_b32_e32 v49, v47, v48
	v_mad_u64_u32 v[46:47], s[18:19], v67, s7, 0
	v_mul_hi_u32 v18, v67, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[70:71], s[18:19], v49, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v70
	v_mad_u64_u32 v[68:69], s[18:19], v49, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v71, vcc
	v_mul_lo_u32 v40, v40, s33
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[68:69]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v68, s50, v47
	v_mad_u64_u32 v[46:47], s[18:19], s50, v46, 0
	v_add3_u32 v18, v47, v68, v18
	v_sub_u32_e32 v47, v49, v18
	v_sub_co_u32_e32 v46, vcc, v67, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[18:19], v47, v42, vcc
	v_subrev_co_u32_e64 v67, s[18:19], s50, v46
	v_subb_co_u32_e32 v18, vcc, v49, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v68, s[22:23], 0, v47, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s51, v68
	v_subb_co_u32_e64 v47, s[18:19], v47, v42, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v67
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v68
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v69, v69, v70, s[22:23]
	v_subrev_co_u32_e64 v70, s[18:19], s50, v67
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[18:19], 0, v47, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v69
	s_nop 1
	v_cndmask_b32_e64 v47, v68, v47, s[18:19]
	v_cndmask_b32_e64 v68, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v49, v49, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_cndmask_b32_e64 v47, v67, v70, s[18:19]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v18, v48
	v_xor_b32_e32 v18, v46, v48
	v_sub_co_u32_e32 v18, vcc, v18, v48
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v47, v48, vcc
	v_ashrrev_i32_e32 v46, 31, v47
	v_and_b32_e32 v46, s42, v46
	v_add3_u32 v18, v46, v18, v40
	buffer_load_ubyte v67, v41, s[28:31], 0 offen
	buffer_load_ubyte v69, v43, s[28:31], 0 offen
	buffer_load_ubyte v68, v44, s[28:31], 0 offen
	buffer_load_ubyte v70, v45, s[28:31], 0 offen
	buffer_load_ubyte v71, v74, s[28:31], 0 offen
	buffer_load_ubyte v73, v75, s[28:31], 0 offen
	buffer_load_ubyte v72, v76, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v74, v18, s[28:31], 0 offen
	v_lshl_add_u64 v[44:45], v[32:33], 0, -15
	v_lshl_add_u64 v[40:41], v[28:29], 0, s[90:91]
	v_cmp_gt_i32_e64 s[18:19], 0, v45
	s_nop 1
	v_cndmask_b32_e64 v41, v45, v41, s[18:19]
	v_ashrrev_i32_e32 v46, 31, v41
	v_cndmask_b32_e64 v40, v44, v40, s[18:19]
	v_mov_b32_e32 v47, v46
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[46:47]
	v_xor_b32_e32 v47, v40, v46
	v_xor_b32_e32 v43, v41, v46
	v_mad_u64_u32 v[40:41], s[22:23], v47, s7, 0
	v_mul_hi_u32 v18, v47, s6
	v_lshl_add_u64 v[40:41], v[18:19], 0, v[40:41]
	v_mad_u64_u32 v[76:77], s[22:23], v43, s6, 0
	v_add_co_u32_e32 v18, vcc, v40, v76
	v_mad_u64_u32 v[48:49], s[22:23], v43, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v41, v77, vcc
	s_nop 1
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[40:41], v[18:19], 0, v[48:49]
	v_mul_lo_u32 v18, s51, v40
	v_mul_lo_u32 v41, s50, v41
	v_mad_u64_u32 v[48:49], s[22:23], s50, v40, 0
	v_add3_u32 v18, v49, v41, v18
	v_sub_u32_e32 v41, v43, v18
	v_sub_co_u32_e32 v47, vcc, v47, v48
	s_nop 1
	v_subb_co_u32_e64 v41, s[22:23], v41, v42, vcc
	v_subrev_co_u32_e64 v48, s[22:23], s50, v47
	v_subb_co_u32_e32 v18, vcc, v43, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v41, s[22:23], 0, v41, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v41
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v48
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v47
	v_cndmask_b32_e64 v48, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v41
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v41, v49, v48, s[22:23]
	v_add_u32_e32 v48, 2, v40
	v_add_u32_e32 v49, 1, v40
	v_cmp_ne_u32_e64 s[22:23], 0, v41
	v_cndmask_b32_e32 v18, v43, v47, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v41, v49, v48, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v40, v41, vcc
	v_xor_b32_e32 v40, s60, v46
	v_xor_b32_e32 v18, v18, v40
	v_sub_u32_e32 v18, v18, v40
	v_ashrrev_i32_e32 v40, 31, v45
	v_mov_b32_e32 v41, v40
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[40:41]
	v_xor_b32_e32 v75, v44, v40
	v_xad_u32 v43, v18, v40, v51
	v_xor_b32_e32 v41, v45, v40
	v_mad_u64_u32 v[44:45], s[22:23], v75, s7, 0
	v_mul_hi_u32 v18, v75, s6
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[44:45]
	v_mad_u64_u32 v[48:49], s[22:23], v41, s6, 0
	v_add_co_u32_e32 v18, vcc, v44, v48
	v_mad_u64_u32 v[46:47], s[22:23], v41, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v45, v49, vcc
	v_mul_lo_u32 v43, v43, s33
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[46:47]
	v_mul_lo_u32 v18, s51, v44
	v_mul_lo_u32 v46, s50, v45
	v_mad_u64_u32 v[44:45], s[22:23], s50, v44, 0
	v_add3_u32 v18, v45, v46, v18
	v_sub_u32_e32 v45, v41, v18
	v_sub_co_u32_e32 v44, vcc, v75, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, vcc
	v_subrev_co_u32_e64 v46, s[22:23], s50, v44
	v_subb_co_u32_e32 v18, vcc, v41, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[24:25], 0, v45, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v47
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v48, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v46
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v47
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v44
	v_cndmask_b32_e64 v48, v48, v49, s[24:25]
	v_subrev_co_u32_e64 v49, s[22:23], s50, v46
	s_nop 1
	v_subbrev_co_u32_e64 v45, s[22:23], 0, v45, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v48
	s_nop 1
	v_cndmask_b32_e64 v45, v47, v45, s[22:23]
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v41, v41, v47, vcc
	v_cmp_ne_u32_e32 vcc, 0, v41
	v_cndmask_b32_e64 v41, v46, v49, s[22:23]
	v_lshl_add_u64 v[46:47], v[28:29], 0, s[88:89]
	v_cndmask_b32_e32 v18, v18, v45, vcc
	v_cndmask_b32_e32 v41, v44, v41, vcc
	v_lshl_add_u64 v[44:45], v[32:33], 0, -14
	v_xor_b32_e32 v41, v41, v40
	v_cndmask_b32_e64 v47, v45, v47, s[18:19]
	v_xor_b32_e32 v18, v18, v40
	v_sub_co_u32_e32 v41, vcc, v41, v40
	v_ashrrev_i32_e32 v48, 31, v47
	s_nop 0
	v_subb_co_u32_e32 v18, vcc, v18, v40, vcc
	v_cndmask_b32_e64 v46, v44, v46, s[18:19]
	v_mov_b32_e32 v49, v48
	v_ashrrev_i32_e32 v18, 31, v18
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_and_b32_e32 v18, s42, v18
	v_xor_b32_e32 v49, v46, v48
	v_add3_u32 v41, v18, v41, v43
	v_xor_b32_e32 v43, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v49, s7, 0
	v_mul_hi_u32 v18, v49, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[78:79], s[22:23], v43, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v78
	v_mad_u64_u32 v[76:77], s[22:23], v43, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v79, vcc
	s_nop 1
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[76:77]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v47, s50, v47
	v_mad_u64_u32 v[76:77], s[22:23], s50, v46, 0
	v_add3_u32 v18, v77, v47, v18
	v_sub_u32_e32 v47, v43, v18
	v_sub_co_u32_e32 v49, vcc, v49, v76
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v75, s[22:23], s50, v49
	v_subb_co_u32_e32 v18, vcc, v43, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v47
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v76, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v75
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v49
	v_cndmask_b32_e64 v75, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v47
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v47, v76, v75, s[22:23]
	v_add_u32_e32 v75, 2, v46
	v_add_u32_e32 v76, 1, v46
	v_cmp_ne_u32_e64 s[22:23], 0, v47
	v_cndmask_b32_e32 v18, v43, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v47, v76, v75, s[22:23]
	v_xor_b32_e32 v43, s60, v48
	v_cndmask_b32_e32 v18, v46, v47, vcc
	v_ashrrev_i32_e32 v46, 31, v45
	v_mov_b32_e32 v47, v46
	v_xor_b32_e32 v18, v18, v43
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_sub_u32_e32 v18, v18, v43
	v_xor_b32_e32 v47, v44, v46
	v_xad_u32 v75, v18, v40, v51
	v_xor_b32_e32 v43, v45, v46
	v_mad_u64_u32 v[44:45], s[22:23], v47, s7, 0
	v_mul_hi_u32 v18, v47, s6
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[44:45]
	v_mad_u64_u32 v[76:77], s[22:23], v43, s6, 0
	v_add_co_u32_e32 v18, vcc, v44, v76
	v_mad_u64_u32 v[48:49], s[22:23], v43, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v45, v77, vcc
	s_nop 1
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[48:49]
	v_mul_lo_u32 v18, s51, v44
	v_mul_lo_u32 v48, s50, v45
	v_mad_u64_u32 v[44:45], s[22:23], s50, v44, 0
	v_add3_u32 v18, v45, v48, v18
	v_sub_u32_e32 v45, v43, v18
	v_sub_co_u32_e32 v44, vcc, v47, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, vcc
	v_subrev_co_u32_e64 v47, s[22:23], s50, v44
	v_subb_co_u32_e32 v18, vcc, v43, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v48, s[24:25], 0, v45, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v48
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v47
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v76, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v48
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v44
	v_cndmask_b32_e64 v49, v49, v76, s[24:25]
	v_subrev_co_u32_e64 v76, s[22:23], s50, v47
	s_nop 1
	v_subbrev_co_u32_e64 v45, s[22:23], 0, v45, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	s_nop 1
	v_cndmask_b32_e64 v45, v48, v45, s[22:23]
	v_cndmask_b32_e64 v48, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v43, v43, v48, vcc
	v_cmp_ne_u32_e32 vcc, 0, v43
	v_cndmask_b32_e64 v43, v47, v76, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v18, v45, vcc
	v_cndmask_b32_e32 v43, v44, v43, vcc
	v_xor_b32_e32 v44, v18, v46
	v_xor_b32_e32 v18, v43, v46
	v_sub_co_u32_e32 v18, vcc, v18, v46
	s_nop 1
	v_subb_co_u32_e32 v43, vcc, v44, v46, vcc
	v_ashrrev_i32_e32 v43, 31, v43
	v_and_b32_e32 v43, s42, v43
	v_mul_lo_u32 v44, v75, s33
	v_add3_u32 v43, v43, v18, v44
	v_lshl_add_u64 v[44:45], v[32:33], 0, -13
	v_lshl_add_u64 v[46:47], v[28:29], 0, s[86:87]
	v_cndmask_b32_e64 v47, v45, v47, s[18:19]
	v_ashrrev_i32_e32 v48, 31, v47
	v_cndmask_b32_e64 v46, v44, v46, s[18:19]
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v75, v46, v48
	v_xor_b32_e32 v49, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v75, s7, 0
	v_mul_hi_u32 v18, v75, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[78:79], s[22:23], v49, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v78
	v_mad_u64_u32 v[76:77], s[22:23], v49, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v79, vcc
	s_nop 1
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[76:77]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v47, s50, v47
	v_mad_u64_u32 v[76:77], s[22:23], s50, v46, 0
	v_add3_u32 v18, v77, v47, v18
	v_sub_u32_e32 v47, v49, v18
	v_sub_co_u32_e32 v75, vcc, v75, v76
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v76, s[22:23], s50, v75
	v_subb_co_u32_e32 v18, vcc, v49, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v47
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v77, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v76
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v75
	v_cndmask_b32_e64 v76, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v47
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v47, v77, v76, s[22:23]
	v_add_u32_e32 v76, 2, v46
	v_add_u32_e32 v77, 1, v46
	v_cmp_ne_u32_e64 s[22:23], 0, v47
	v_cndmask_b32_e32 v18, v49, v75, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v47, v77, v76, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v46, v47, vcc
	v_xor_b32_e32 v46, s60, v48
	v_xor_b32_e32 v18, v18, v46
	v_sub_u32_e32 v18, v18, v46
	v_ashrrev_i32_e32 v46, 31, v45
	v_mov_b32_e32 v47, v46
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_xor_b32_e32 v78, v44, v46
	v_xad_u32 v75, v18, v40, v51
	v_xor_b32_e32 v47, v45, v46
	v_mad_u64_u32 v[44:45], s[22:23], v78, s7, 0
	v_mul_hi_u32 v18, v78, s6
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[44:45]
	v_mad_u64_u32 v[76:77], s[22:23], v47, s6, 0
	v_add_co_u32_e32 v18, vcc, v44, v76
	v_mad_u64_u32 v[48:49], s[22:23], v47, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v45, v77, vcc
	s_nop 1
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[48:49]
	v_mul_lo_u32 v18, s51, v44
	v_mul_lo_u32 v48, s50, v45
	v_mad_u64_u32 v[44:45], s[22:23], s50, v44, 0
	v_add3_u32 v18, v45, v48, v18
	v_sub_u32_e32 v45, v47, v18
	v_sub_co_u32_e32 v44, vcc, v78, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, vcc
	v_subrev_co_u32_e64 v48, s[22:23], s50, v44
	v_subb_co_u32_e32 v18, vcc, v47, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[24:25], 0, v45, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v49
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v76, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v48
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v77, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v49
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v44
	v_cndmask_b32_e64 v76, v76, v77, s[24:25]
	v_subrev_co_u32_e64 v77, s[22:23], s50, v48
	s_nop 1
	v_subbrev_co_u32_e64 v45, s[22:23], 0, v45, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v76
	s_nop 1
	v_cndmask_b32_e64 v45, v49, v45, s[22:23]
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v47, v47, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v47
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v45, vcc
	v_cndmask_b32_e64 v45, v48, v77, s[22:23]
	v_cndmask_b32_e32 v44, v44, v45, vcc
	v_xor_b32_e32 v45, v18, v46
	v_xor_b32_e32 v18, v44, v46
	v_sub_co_u32_e32 v18, vcc, v18, v46
	v_lshl_add_u64 v[48:49], v[28:29], 0, s[84:85]
	s_nop 0
	v_subb_co_u32_e32 v45, vcc, v45, v46, vcc
	v_lshl_add_u64 v[46:47], v[32:33], 0, -12
	v_cndmask_b32_e64 v49, v47, v49, s[18:19]
	v_ashrrev_i32_e32 v76, 31, v49
	v_cndmask_b32_e64 v48, v46, v48, s[18:19]
	v_mov_b32_e32 v77, v76
	v_ashrrev_i32_e32 v44, 31, v45
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[76:77]
	v_and_b32_e32 v44, s42, v44
	v_mul_lo_u32 v45, v75, s33
	v_xor_b32_e32 v75, v48, v76
	v_add3_u32 v44, v44, v18, v45
	v_xor_b32_e32 v45, v49, v76
	v_mad_u64_u32 v[48:49], s[22:23], v75, s7, 0
	v_mul_hi_u32 v18, v75, s6
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[80:81], s[22:23], v45, s6, 0
	v_add_co_u32_e32 v18, vcc, v48, v80
	v_mad_u64_u32 v[78:79], s[22:23], v45, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v49, v81, vcc
	s_nop 1
	v_addc_co_u32_e32 v79, vcc, 0, v79, vcc
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[78:79]
	v_mul_lo_u32 v18, s51, v48
	v_mul_lo_u32 v49, s50, v49
	v_mad_u64_u32 v[78:79], s[22:23], s50, v48, 0
	v_add3_u32 v18, v79, v49, v18
	v_sub_u32_e32 v49, v45, v18
	v_sub_co_u32_e32 v75, vcc, v75, v78
	s_nop 1
	v_subb_co_u32_e64 v49, s[22:23], v49, v42, vcc
	v_subrev_co_u32_e64 v77, s[22:23], s50, v75
	v_subb_co_u32_e32 v18, vcc, v45, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v49, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v78, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v77
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v75
	v_cndmask_b32_e64 v77, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v49
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v49, v78, v77, s[22:23]
	v_add_u32_e32 v77, 2, v48
	v_add_u32_e32 v78, 1, v48
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	v_cndmask_b32_e32 v18, v45, v75, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v49, v78, v77, s[22:23]
	v_xor_b32_e32 v45, s60, v76
	v_cndmask_b32_e32 v18, v48, v49, vcc
	v_ashrrev_i32_e32 v48, 31, v47
	v_mov_b32_e32 v49, v48
	v_xor_b32_e32 v18, v18, v45
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_sub_u32_e32 v18, v18, v45
	v_xor_b32_e32 v49, v46, v48
	v_xad_u32 v75, v18, v40, v51
	v_xor_b32_e32 v45, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v49, s7, 0
	v_mul_hi_u32 v18, v49, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[78:79], s[22:23], v45, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v78
	v_mad_u64_u32 v[76:77], s[22:23], v45, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v79, vcc
	s_nop 1
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[76:77]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v76, s50, v47
	v_mad_u64_u32 v[46:47], s[22:23], s50, v46, 0
	v_add3_u32 v18, v47, v76, v18
	v_sub_u32_e32 v47, v45, v18
	v_sub_co_u32_e32 v46, vcc, v49, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v49, s[22:23], s50, v46
	v_subb_co_u32_e32 v18, vcc, v45, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v76, s[24:25], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v76
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v77, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v78, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v76
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v77, v77, v78, s[24:25]
	v_subrev_co_u32_e64 v78, s[22:23], s50, v49
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v77
	s_nop 1
	v_cndmask_b32_e64 v47, v76, v47, s[22:23]
	v_cndmask_b32_e64 v76, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v45, v45, v76, vcc
	v_cmp_ne_u32_e32 vcc, 0, v45
	v_cndmask_b32_e64 v45, v49, v78, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_cndmask_b32_e32 v45, v46, v45, vcc
	v_xor_b32_e32 v46, v18, v48
	v_xor_b32_e32 v18, v45, v48
	v_sub_co_u32_e32 v18, vcc, v18, v48
	s_nop 1
	v_subb_co_u32_e32 v45, vcc, v46, v48, vcc
	v_ashrrev_i32_e32 v45, 31, v45
	v_and_b32_e32 v45, s42, v45
	v_mul_lo_u32 v46, v75, s33
	v_add3_u32 v45, v45, v18, v46
	v_lshl_add_u64 v[46:47], v[32:33], 0, -11
	v_lshl_add_u64 v[48:49], v[28:29], 0, s[82:83]
	v_cndmask_b32_e64 v49, v47, v49, s[18:19]
	v_ashrrev_i32_e32 v76, 31, v49
	v_cndmask_b32_e64 v48, v46, v48, s[18:19]
	v_mov_b32_e32 v77, v76
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[76:77]
	v_xor_b32_e32 v77, v48, v76
	v_xor_b32_e32 v75, v49, v76
	v_mad_u64_u32 v[48:49], s[22:23], v77, s7, 0
	v_mul_hi_u32 v18, v77, s6
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[80:81], s[22:23], v75, s6, 0
	v_add_co_u32_e32 v18, vcc, v48, v80
	v_mad_u64_u32 v[78:79], s[22:23], v75, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v49, v81, vcc
	s_nop 1
	v_addc_co_u32_e32 v79, vcc, 0, v79, vcc
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[78:79]
	v_mul_lo_u32 v18, s51, v48
	v_mul_lo_u32 v49, s50, v49
	v_mad_u64_u32 v[78:79], s[22:23], s50, v48, 0
	v_add3_u32 v18, v79, v49, v18
	v_sub_u32_e32 v49, v75, v18
	v_sub_co_u32_e32 v77, vcc, v77, v78
	s_nop 1
	v_subb_co_u32_e64 v49, s[22:23], v49, v42, vcc
	v_subrev_co_u32_e64 v78, s[22:23], s50, v77
	v_subb_co_u32_e32 v18, vcc, v75, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v49, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v78
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v77
	v_cndmask_b32_e64 v78, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v49
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v49, v79, v78, s[22:23]
	v_add_u32_e32 v78, 2, v48
	v_add_u32_e32 v79, 1, v48
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	v_cndmask_b32_e32 v18, v75, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v49, v79, v78, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v48, v49, vcc
	v_xor_b32_e32 v48, s60, v76
	v_xor_b32_e32 v18, v18, v48
	v_sub_u32_e32 v18, v18, v48
	v_ashrrev_i32_e32 v48, 31, v47
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v80, v46, v48
	v_xad_u32 v75, v18, v40, v51
	v_xor_b32_e32 v49, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v80, s7, 0
	v_mul_hi_u32 v18, v80, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[78:79], s[22:23], v49, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v78
	v_mad_u64_u32 v[76:77], s[22:23], v49, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v79, vcc
	s_nop 1
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[76:77]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v76, s50, v47
	v_mad_u64_u32 v[46:47], s[22:23], s50, v46, 0
	v_add3_u32 v18, v47, v76, v18
	v_sub_u32_e32 v47, v49, v18
	v_sub_co_u32_e32 v46, vcc, v80, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v76, s[22:23], s50, v46
	v_subb_co_u32_e32 v18, vcc, v49, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v77, s[24:25], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v77
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v78, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v76
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v77
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v78, v78, v79, s[24:25]
	v_subrev_co_u32_e64 v79, s[22:23], s50, v76
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v78
	s_nop 1
	v_cndmask_b32_e64 v47, v77, v47, s[22:23]
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v49, v49, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_cndmask_b32_e64 v47, v76, v79, s[22:23]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v18, v48
	v_xor_b32_e32 v18, v46, v48
	v_sub_co_u32_e32 v18, vcc, v18, v48
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v47, v48, vcc
	v_ashrrev_i32_e32 v46, 31, v47
	v_and_b32_e32 v46, s42, v46
	v_mul_lo_u32 v47, v75, s33
	v_add3_u32 v82, v46, v18, v47
	v_lshl_add_u64 v[46:47], v[32:33], 0, -10
	v_lshl_add_u64 v[48:49], v[28:29], 0, s[80:81]
	v_cndmask_b32_e64 v49, v47, v49, s[18:19]
	v_ashrrev_i32_e32 v76, 31, v49
	v_cndmask_b32_e64 v48, v46, v48, s[18:19]
	v_mov_b32_e32 v77, v76
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[76:77]
	v_xor_b32_e32 v77, v48, v76
	v_xor_b32_e32 v75, v49, v76
	v_mad_u64_u32 v[48:49], s[22:23], v77, s7, 0
	v_mul_hi_u32 v18, v77, s6
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[80:81], s[22:23], v75, s6, 0
	v_add_co_u32_e32 v18, vcc, v48, v80
	v_mad_u64_u32 v[78:79], s[22:23], v75, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v49, v81, vcc
	s_nop 1
	v_addc_co_u32_e32 v79, vcc, 0, v79, vcc
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[78:79]
	v_mul_lo_u32 v18, s51, v48
	v_mul_lo_u32 v49, s50, v49
	v_mad_u64_u32 v[78:79], s[22:23], s50, v48, 0
	v_add3_u32 v18, v79, v49, v18
	v_sub_u32_e32 v49, v75, v18
	v_sub_co_u32_e32 v77, vcc, v77, v78
	s_nop 1
	v_subb_co_u32_e64 v49, s[22:23], v49, v42, vcc
	v_subrev_co_u32_e64 v78, s[22:23], s50, v77
	v_subb_co_u32_e32 v18, vcc, v75, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v49, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v78
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v77
	v_cndmask_b32_e64 v78, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v49
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v49, v79, v78, s[22:23]
	v_add_u32_e32 v78, 2, v48
	v_add_u32_e32 v79, 1, v48
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	v_cndmask_b32_e32 v18, v75, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v49, v79, v78, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v48, v49, vcc
	v_xor_b32_e32 v48, s60, v76
	v_xor_b32_e32 v18, v18, v48
	v_sub_u32_e32 v18, v18, v48
	v_ashrrev_i32_e32 v48, 31, v47
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v80, v46, v48
	v_xad_u32 v75, v18, v40, v51
	v_xor_b32_e32 v49, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v80, s7, 0
	v_mul_hi_u32 v18, v80, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[78:79], s[22:23], v49, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v78
	v_mad_u64_u32 v[76:77], s[22:23], v49, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v79, vcc
	s_nop 1
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[76:77]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v76, s50, v47
	v_mad_u64_u32 v[46:47], s[22:23], s50, v46, 0
	v_add3_u32 v18, v47, v76, v18
	v_sub_u32_e32 v47, v49, v18
	v_sub_co_u32_e32 v46, vcc, v80, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v76, s[22:23], s50, v46
	v_subb_co_u32_e32 v18, vcc, v49, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v77, s[24:25], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v77
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v78, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v76
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v77
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v78, v78, v79, s[24:25]
	v_subrev_co_u32_e64 v79, s[22:23], s50, v76
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v78
	s_nop 1
	v_cndmask_b32_e64 v47, v77, v47, s[22:23]
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v49, v49, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_cndmask_b32_e64 v47, v76, v79, s[22:23]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v18, v48
	v_xor_b32_e32 v18, v46, v48
	v_sub_co_u32_e32 v18, vcc, v18, v48
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v47, v48, vcc
	v_ashrrev_i32_e32 v46, 31, v47
	v_and_b32_e32 v46, s42, v46
	v_mul_lo_u32 v47, v75, s33
	v_add3_u32 v83, v46, v18, v47
	v_lshl_add_u64 v[46:47], v[32:33], 0, -9
	v_lshl_add_u64 v[48:49], v[28:29], 0, s[78:79]
	v_cndmask_b32_e64 v49, v47, v49, s[18:19]
	v_ashrrev_i32_e32 v76, 31, v49
	v_cndmask_b32_e64 v48, v46, v48, s[18:19]
	v_mov_b32_e32 v77, v76
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[76:77]
	v_xor_b32_e32 v77, v48, v76
	v_xor_b32_e32 v75, v49, v76
	v_mad_u64_u32 v[48:49], s[22:23], v77, s7, 0
	v_mul_hi_u32 v18, v77, s6
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[80:81], s[22:23], v75, s6, 0
	v_add_co_u32_e32 v18, vcc, v48, v80
	v_mad_u64_u32 v[78:79], s[22:23], v75, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v49, v81, vcc
	s_nop 1
	v_addc_co_u32_e32 v79, vcc, 0, v79, vcc
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[78:79]
	v_mul_lo_u32 v18, s51, v48
	v_mul_lo_u32 v49, s50, v49
	v_mad_u64_u32 v[78:79], s[22:23], s50, v48, 0
	v_add3_u32 v18, v79, v49, v18
	v_sub_u32_e32 v49, v75, v18
	v_sub_co_u32_e32 v77, vcc, v77, v78
	s_nop 1
	v_subb_co_u32_e64 v49, s[22:23], v49, v42, vcc
	v_subrev_co_u32_e64 v78, s[22:23], s50, v77
	v_subb_co_u32_e32 v18, vcc, v75, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v49, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v78
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v77
	v_cndmask_b32_e64 v78, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v49
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v49, v79, v78, s[22:23]
	v_add_u32_e32 v78, 2, v48
	v_add_u32_e32 v79, 1, v48
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	v_cndmask_b32_e32 v18, v75, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v49, v79, v78, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v48, v49, vcc
	v_xor_b32_e32 v48, s60, v76
	v_xor_b32_e32 v18, v18, v48
	v_sub_u32_e32 v18, v18, v48
	v_ashrrev_i32_e32 v48, 31, v47
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v80, v46, v48
	v_xad_u32 v75, v18, v40, v51
	v_xor_b32_e32 v49, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v80, s7, 0
	v_mul_hi_u32 v18, v80, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[78:79], s[22:23], v49, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v78
	v_mad_u64_u32 v[76:77], s[22:23], v49, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v79, vcc
	s_nop 1
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[76:77]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v76, s50, v47
	v_mad_u64_u32 v[46:47], s[22:23], s50, v46, 0
	v_add3_u32 v18, v47, v76, v18
	v_sub_u32_e32 v47, v49, v18
	v_sub_co_u32_e32 v46, vcc, v80, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v76, s[22:23], s50, v46
	v_subb_co_u32_e32 v18, vcc, v49, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v77, s[24:25], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v77
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v78, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v76
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v77
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v78, v78, v79, s[24:25]
	v_subrev_co_u32_e64 v79, s[22:23], s50, v76
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v78
	s_nop 1
	v_cndmask_b32_e64 v47, v77, v47, s[22:23]
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v49, v49, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_cndmask_b32_e64 v47, v76, v79, s[22:23]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v18, v48
	v_xor_b32_e32 v18, v46, v48
	v_sub_co_u32_e32 v18, vcc, v18, v48
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v47, v48, vcc
	v_ashrrev_i32_e32 v46, 31, v47
	v_and_b32_e32 v46, s42, v46
	v_mul_lo_u32 v47, v75, s33
	v_add3_u32 v84, v46, v18, v47
	v_lshl_add_u64 v[46:47], v[32:33], 0, -8
	v_lshl_add_u64 v[48:49], v[28:29], 0, s[76:77]
	v_cndmask_b32_e64 v49, v47, v49, s[18:19]
	v_ashrrev_i32_e32 v76, 31, v49
	v_cndmask_b32_e64 v48, v46, v48, s[18:19]
	v_mov_b32_e32 v77, v76
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[76:77]
	v_xor_b32_e32 v77, v48, v76
	v_xor_b32_e32 v75, v49, v76
	v_mad_u64_u32 v[48:49], s[22:23], v77, s7, 0
	v_mul_hi_u32 v18, v77, s6
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[80:81], s[22:23], v75, s6, 0
	v_add_co_u32_e32 v18, vcc, v48, v80
	v_mad_u64_u32 v[78:79], s[22:23], v75, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v49, v81, vcc
	s_nop 1
	v_addc_co_u32_e32 v79, vcc, 0, v79, vcc
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[78:79]
	v_mul_lo_u32 v18, s51, v48
	v_mul_lo_u32 v49, s50, v49
	v_mad_u64_u32 v[78:79], s[22:23], s50, v48, 0
	v_add3_u32 v18, v79, v49, v18
	v_sub_u32_e32 v49, v75, v18
	v_sub_co_u32_e32 v77, vcc, v77, v78
	s_nop 1
	v_subb_co_u32_e64 v49, s[22:23], v49, v42, vcc
	v_subrev_co_u32_e64 v78, s[22:23], s50, v77
	v_subb_co_u32_e32 v18, vcc, v75, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v49, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v78
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v77
	v_cndmask_b32_e64 v78, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v49
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v49, v79, v78, s[22:23]
	v_add_u32_e32 v78, 2, v48
	v_add_u32_e32 v79, 1, v48
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	v_cndmask_b32_e32 v18, v75, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v49, v79, v78, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v48, v49, vcc
	v_xor_b32_e32 v48, s60, v76
	v_xor_b32_e32 v18, v18, v48
	v_sub_u32_e32 v18, v18, v48
	v_ashrrev_i32_e32 v48, 31, v47
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v80, v46, v48
	v_xad_u32 v75, v18, v40, v51
	v_xor_b32_e32 v49, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v80, s7, 0
	v_mul_hi_u32 v18, v80, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[78:79], s[22:23], v49, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v78
	v_mad_u64_u32 v[76:77], s[22:23], v49, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v79, vcc
	s_nop 1
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[76:77]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v76, s50, v47
	v_mad_u64_u32 v[46:47], s[22:23], s50, v46, 0
	v_add3_u32 v18, v47, v76, v18
	v_sub_u32_e32 v47, v49, v18
	v_sub_co_u32_e32 v46, vcc, v80, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v76, s[22:23], s50, v46
	v_subb_co_u32_e32 v18, vcc, v49, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v77, s[24:25], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v77
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v78, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v76
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v77
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v78, v78, v79, s[24:25]
	v_subrev_co_u32_e64 v79, s[22:23], s50, v76
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v78
	s_nop 1
	v_cndmask_b32_e64 v47, v77, v47, s[22:23]
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v49, v49, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_cndmask_b32_e64 v47, v76, v79, s[22:23]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v18, v48
	v_xor_b32_e32 v18, v46, v48
	v_sub_co_u32_e32 v18, vcc, v18, v48
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v47, v48, vcc
	v_ashrrev_i32_e32 v46, 31, v47
	v_and_b32_e32 v46, s42, v46
	v_mul_lo_u32 v47, v75, s33
	v_add3_u32 v18, v46, v18, v47
	buffer_load_ubyte v76, v41, s[28:31], 0 offen
	buffer_load_ubyte v78, v43, s[28:31], 0 offen
	buffer_load_ubyte v75, v44, s[28:31], 0 offen
	buffer_load_ubyte v77, v45, s[28:31], 0 offen
	buffer_load_ubyte v79, v82, s[28:31], 0 offen
	buffer_load_ubyte v81, v83, s[28:31], 0 offen
	buffer_load_ubyte v80, v84, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v82, v18, s[28:31], 0 offen
	v_lshl_add_u64 v[44:45], v[32:33], 0, -7
	v_lshl_add_u64 v[46:47], v[28:29], 0, s[74:75]
	v_cndmask_b32_e64 v47, v45, v47, s[18:19]
	v_ashrrev_i32_e32 v48, 31, v47
	v_cndmask_b32_e64 v46, v44, v46, s[18:19]
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v43, v46, v48
	v_xor_b32_e32 v41, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v43, s7, 0
	v_mul_hi_u32 v18, v43, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[86:87], s[22:23], v41, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v86
	v_mad_u64_u32 v[84:85], s[22:23], v41, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v87, vcc
	s_nop 1
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[84:85]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v47, s50, v47
	v_mad_u64_u32 v[84:85], s[22:23], s50, v46, 0
	v_add3_u32 v18, v85, v47, v18
	v_sub_u32_e32 v47, v41, v18
	v_sub_co_u32_e32 v43, vcc, v43, v84
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v49, s[22:23], s50, v43
	v_subb_co_u32_e32 v18, vcc, v41, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v47
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v49
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v43
	v_cndmask_b32_e64 v49, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v47
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v47, v83, v49, s[22:23]
	v_add_u32_e32 v49, 2, v46
	v_add_u32_e32 v83, 1, v46
	v_cmp_ne_u32_e64 s[22:23], 0, v47
	v_cndmask_b32_e32 v18, v41, v43, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v47, v83, v49, s[22:23]
	v_xor_b32_e32 v41, s60, v48
	v_cndmask_b32_e32 v18, v46, v47, vcc
	v_ashrrev_i32_e32 v46, 31, v45
	v_mov_b32_e32 v47, v46
	v_xor_b32_e32 v18, v18, v41
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_sub_u32_e32 v18, v18, v41
	v_xor_b32_e32 v47, v44, v46
	v_xad_u32 v43, v18, v40, v51
	v_xor_b32_e32 v41, v45, v46
	v_mad_u64_u32 v[44:45], s[22:23], v47, s7, 0
	v_mul_hi_u32 v18, v47, s6
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[44:45]
	v_mad_u64_u32 v[84:85], s[22:23], v41, s6, 0
	v_add_co_u32_e32 v18, vcc, v44, v84
	v_mad_u64_u32 v[48:49], s[22:23], v41, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v45, v85, vcc
	v_mul_lo_u32 v43, v43, s33
	s_nop 0
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[48:49]
	v_mul_lo_u32 v18, s51, v44
	v_mul_lo_u32 v48, s50, v45
	v_mad_u64_u32 v[44:45], s[22:23], s50, v44, 0
	v_add3_u32 v18, v45, v48, v18
	v_sub_u32_e32 v45, v41, v18
	v_sub_co_u32_e32 v44, vcc, v47, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, vcc
	v_subrev_co_u32_e64 v47, s[22:23], s50, v44
	v_subb_co_u32_e32 v18, vcc, v41, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v48, s[24:25], 0, v45, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v48
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v47
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v48
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v44
	v_cndmask_b32_e64 v49, v49, v83, s[24:25]
	v_subrev_co_u32_e64 v83, s[22:23], s50, v47
	s_nop 1
	v_subbrev_co_u32_e64 v45, s[22:23], 0, v45, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	s_nop 1
	v_cndmask_b32_e64 v45, v48, v45, s[22:23]
	v_cndmask_b32_e64 v48, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v41, v41, v48, vcc
	v_cmp_ne_u32_e32 vcc, 0, v41
	v_cndmask_b32_e64 v41, v47, v83, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v18, v45, vcc
	v_cndmask_b32_e32 v41, v44, v41, vcc
	v_xor_b32_e32 v44, v18, v46
	v_xor_b32_e32 v18, v41, v46
	v_sub_co_u32_e32 v18, vcc, v18, v46
	s_nop 1
	v_subb_co_u32_e32 v41, vcc, v44, v46, vcc
	v_lshl_add_u64 v[44:45], v[32:33], 0, -6
	v_lshl_add_u64 v[46:47], v[28:29], 0, s[72:73]
	v_cndmask_b32_e64 v47, v45, v47, s[18:19]
	v_ashrrev_i32_e32 v48, 31, v47
	v_cndmask_b32_e64 v46, v44, v46, s[18:19]
	v_mov_b32_e32 v49, v48
	v_ashrrev_i32_e32 v41, 31, v41
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_and_b32_e32 v41, s42, v41
	v_xor_b32_e32 v49, v46, v48
	v_add3_u32 v41, v41, v18, v43
	v_xor_b32_e32 v43, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v49, s7, 0
	v_mul_hi_u32 v18, v49, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[86:87], s[22:23], v43, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v86
	v_mad_u64_u32 v[84:85], s[22:23], v43, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v87, vcc
	s_nop 1
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[84:85]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v47, s50, v47
	v_mad_u64_u32 v[84:85], s[22:23], s50, v46, 0
	v_add3_u32 v18, v85, v47, v18
	v_sub_u32_e32 v47, v43, v18
	v_sub_co_u32_e32 v49, vcc, v49, v84
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v83, s[22:23], s50, v49
	v_subb_co_u32_e32 v18, vcc, v43, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v47
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v84, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v83
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v49
	v_cndmask_b32_e64 v83, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v47
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v47, v84, v83, s[22:23]
	v_add_u32_e32 v83, 2, v46
	v_add_u32_e32 v84, 1, v46
	v_cmp_ne_u32_e64 s[22:23], 0, v47
	v_cndmask_b32_e32 v18, v43, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v47, v84, v83, s[22:23]
	v_xor_b32_e32 v43, s60, v48
	v_cndmask_b32_e32 v18, v46, v47, vcc
	v_ashrrev_i32_e32 v46, 31, v45
	v_mov_b32_e32 v47, v46
	v_xor_b32_e32 v18, v18, v43
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_sub_u32_e32 v18, v18, v43
	v_xor_b32_e32 v47, v44, v46
	v_xad_u32 v83, v18, v40, v51
	v_xor_b32_e32 v43, v45, v46
	v_mad_u64_u32 v[44:45], s[22:23], v47, s7, 0
	v_mul_hi_u32 v18, v47, s6
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[44:45]
	v_mad_u64_u32 v[84:85], s[22:23], v43, s6, 0
	v_add_co_u32_e32 v18, vcc, v44, v84
	v_mad_u64_u32 v[48:49], s[22:23], v43, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v45, v85, vcc
	s_nop 1
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[48:49]
	v_mul_lo_u32 v18, s51, v44
	v_mul_lo_u32 v48, s50, v45
	v_mad_u64_u32 v[44:45], s[22:23], s50, v44, 0
	v_add3_u32 v18, v45, v48, v18
	v_sub_u32_e32 v45, v43, v18
	v_sub_co_u32_e32 v44, vcc, v47, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, vcc
	v_subrev_co_u32_e64 v47, s[22:23], s50, v44
	v_subb_co_u32_e32 v18, vcc, v43, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v48, s[24:25], 0, v45, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v48
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v47
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v84, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v48
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v44
	v_cndmask_b32_e64 v49, v49, v84, s[24:25]
	v_subrev_co_u32_e64 v84, s[22:23], s50, v47
	s_nop 1
	v_subbrev_co_u32_e64 v45, s[22:23], 0, v45, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	s_nop 1
	v_cndmask_b32_e64 v45, v48, v45, s[22:23]
	v_cndmask_b32_e64 v48, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v43, v43, v48, vcc
	v_cmp_ne_u32_e32 vcc, 0, v43
	v_cndmask_b32_e64 v43, v47, v84, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v18, v45, vcc
	v_cndmask_b32_e32 v43, v44, v43, vcc
	v_xor_b32_e32 v44, v18, v46
	v_xor_b32_e32 v18, v43, v46
	v_sub_co_u32_e32 v18, vcc, v18, v46
	s_nop 1
	v_subb_co_u32_e32 v43, vcc, v44, v46, vcc
	v_ashrrev_i32_e32 v43, 31, v43
	v_and_b32_e32 v43, s42, v43
	v_mul_lo_u32 v44, v83, s33
	v_add3_u32 v43, v43, v18, v44
	v_lshl_add_u64 v[44:45], v[32:33], 0, -5
	v_lshl_add_u64 v[46:47], v[28:29], 0, s[70:71]
	v_cndmask_b32_e64 v47, v45, v47, s[18:19]
	v_ashrrev_i32_e32 v48, 31, v47
	v_cndmask_b32_e64 v46, v44, v46, s[18:19]
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v83, v46, v48
	v_xor_b32_e32 v49, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v83, s7, 0
	v_mul_hi_u32 v18, v83, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[86:87], s[22:23], v49, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v86
	v_mad_u64_u32 v[84:85], s[22:23], v49, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v87, vcc
	s_nop 1
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[84:85]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v47, s50, v47
	v_mad_u64_u32 v[84:85], s[22:23], s50, v46, 0
	v_add3_u32 v18, v85, v47, v18
	v_sub_u32_e32 v47, v49, v18
	v_sub_co_u32_e32 v83, vcc, v83, v84
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v84, s[22:23], s50, v83
	v_subb_co_u32_e32 v18, vcc, v49, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v47
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v84
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v83
	v_cndmask_b32_e64 v84, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v47
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v47, v85, v84, s[22:23]
	v_add_u32_e32 v84, 2, v46
	v_add_u32_e32 v85, 1, v46
	v_cmp_ne_u32_e64 s[22:23], 0, v47
	v_cndmask_b32_e32 v18, v49, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v47, v85, v84, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v46, v47, vcc
	v_xor_b32_e32 v46, s60, v48
	v_xor_b32_e32 v18, v18, v46
	v_sub_u32_e32 v18, v18, v46
	v_ashrrev_i32_e32 v46, 31, v45
	v_mov_b32_e32 v47, v46
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[46:47]
	v_xor_b32_e32 v86, v44, v46
	v_xad_u32 v83, v18, v40, v51
	v_xor_b32_e32 v47, v45, v46
	v_mad_u64_u32 v[44:45], s[22:23], v86, s7, 0
	v_mul_hi_u32 v18, v86, s6
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[44:45]
	v_mad_u64_u32 v[84:85], s[22:23], v47, s6, 0
	v_add_co_u32_e32 v18, vcc, v44, v84
	v_mad_u64_u32 v[48:49], s[22:23], v47, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v45, v85, vcc
	s_nop 1
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[48:49]
	v_mul_lo_u32 v18, s51, v44
	v_mul_lo_u32 v48, s50, v45
	v_mad_u64_u32 v[44:45], s[22:23], s50, v44, 0
	v_add3_u32 v18, v45, v48, v18
	v_sub_u32_e32 v45, v47, v18
	v_sub_co_u32_e32 v44, vcc, v86, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, vcc
	v_subrev_co_u32_e64 v48, s[22:23], s50, v44
	v_subb_co_u32_e32 v18, vcc, v47, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[24:25], 0, v45, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v49
	v_subb_co_u32_e64 v45, s[22:23], v45, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v84, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v48
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v49
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v44
	v_cndmask_b32_e64 v84, v84, v85, s[24:25]
	v_subrev_co_u32_e64 v85, s[22:23], s50, v48
	s_nop 1
	v_subbrev_co_u32_e64 v45, s[22:23], 0, v45, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v84
	s_nop 1
	v_cndmask_b32_e64 v45, v49, v45, s[22:23]
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v47, v47, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v47
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v45, vcc
	v_cndmask_b32_e64 v45, v48, v85, s[22:23]
	v_cndmask_b32_e32 v44, v44, v45, vcc
	v_xor_b32_e32 v45, v18, v46
	v_xor_b32_e32 v18, v44, v46
	v_sub_co_u32_e32 v18, vcc, v18, v46
	v_lshl_add_u64 v[48:49], v[28:29], 0, s[68:69]
	s_nop 0
	v_subb_co_u32_e32 v45, vcc, v45, v46, vcc
	v_lshl_add_u64 v[46:47], v[32:33], 0, -4
	v_cndmask_b32_e64 v49, v47, v49, s[18:19]
	v_ashrrev_i32_e32 v84, 31, v49
	v_cndmask_b32_e64 v48, v46, v48, s[18:19]
	v_mov_b32_e32 v85, v84
	v_ashrrev_i32_e32 v44, 31, v45
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[84:85]
	v_and_b32_e32 v44, s42, v44
	v_mul_lo_u32 v45, v83, s33
	v_xor_b32_e32 v83, v48, v84
	v_add3_u32 v44, v44, v18, v45
	v_xor_b32_e32 v45, v49, v84
	v_mad_u64_u32 v[48:49], s[22:23], v83, s7, 0
	v_mul_hi_u32 v18, v83, s6
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[88:89], s[22:23], v45, s6, 0
	v_add_co_u32_e32 v18, vcc, v48, v88
	v_mad_u64_u32 v[86:87], s[22:23], v45, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v49, v89, vcc
	s_nop 1
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[86:87]
	v_mul_lo_u32 v18, s51, v48
	v_mul_lo_u32 v49, s50, v49
	v_mad_u64_u32 v[86:87], s[22:23], s50, v48, 0
	v_add3_u32 v18, v87, v49, v18
	v_sub_u32_e32 v49, v45, v18
	v_sub_co_u32_e32 v83, vcc, v83, v86
	s_nop 1
	v_subb_co_u32_e64 v49, s[22:23], v49, v42, vcc
	v_subrev_co_u32_e64 v85, s[22:23], s50, v83
	v_subb_co_u32_e32 v18, vcc, v45, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v49, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v86, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v85
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v83
	v_cndmask_b32_e64 v85, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v49
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v49, v86, v85, s[22:23]
	v_add_u32_e32 v85, 2, v48
	v_add_u32_e32 v86, 1, v48
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	v_cndmask_b32_e32 v18, v45, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v49, v86, v85, s[22:23]
	v_xor_b32_e32 v45, s60, v84
	v_cndmask_b32_e32 v18, v48, v49, vcc
	v_ashrrev_i32_e32 v48, 31, v47
	v_mov_b32_e32 v49, v48
	v_xor_b32_e32 v18, v18, v45
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_sub_u32_e32 v18, v18, v45
	v_xor_b32_e32 v49, v46, v48
	v_xad_u32 v83, v18, v40, v51
	v_xor_b32_e32 v45, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v49, s7, 0
	v_mul_hi_u32 v18, v49, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[86:87], s[22:23], v45, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v86
	v_mad_u64_u32 v[84:85], s[22:23], v45, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v87, vcc
	s_nop 1
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[84:85]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v84, s50, v47
	v_mad_u64_u32 v[46:47], s[22:23], s50, v46, 0
	v_add3_u32 v18, v47, v84, v18
	v_sub_u32_e32 v47, v45, v18
	v_sub_co_u32_e32 v46, vcc, v49, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v49, s[22:23], s50, v46
	v_subb_co_u32_e32 v18, vcc, v45, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v84, s[24:25], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v84
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v86, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v84
	v_cndmask_b32_e64 v45, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v85, v85, v86, s[24:25]
	v_subrev_co_u32_e64 v86, s[22:23], s50, v49
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v85
	s_nop 1
	v_cndmask_b32_e64 v47, v84, v47, s[22:23]
	v_cndmask_b32_e64 v84, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v45, v45, v84, vcc
	v_cmp_ne_u32_e32 vcc, 0, v45
	v_cndmask_b32_e64 v45, v49, v86, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_cndmask_b32_e32 v45, v46, v45, vcc
	v_xor_b32_e32 v46, v18, v48
	v_xor_b32_e32 v18, v45, v48
	v_sub_co_u32_e32 v18, vcc, v18, v48
	s_nop 1
	v_subb_co_u32_e32 v45, vcc, v46, v48, vcc
	v_ashrrev_i32_e32 v45, 31, v45
	v_and_b32_e32 v45, s42, v45
	v_mul_lo_u32 v46, v83, s33
	v_add3_u32 v45, v45, v18, v46
	v_lshl_add_u64 v[46:47], v[32:33], 0, -3
	v_lshl_add_u64 v[48:49], v[28:29], 0, s[66:67]
	v_cndmask_b32_e64 v49, v47, v49, s[18:19]
	v_ashrrev_i32_e32 v84, 31, v49
	v_cndmask_b32_e64 v48, v46, v48, s[18:19]
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[84:85]
	v_xor_b32_e32 v85, v48, v84
	v_xor_b32_e32 v83, v49, v84
	v_mad_u64_u32 v[48:49], s[22:23], v85, s7, 0
	v_mul_hi_u32 v18, v85, s6
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[88:89], s[22:23], v83, s6, 0
	v_add_co_u32_e32 v18, vcc, v48, v88
	v_mad_u64_u32 v[86:87], s[22:23], v83, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v49, v89, vcc
	s_nop 1
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[86:87]
	v_mul_lo_u32 v18, s51, v48
	v_mul_lo_u32 v49, s50, v49
	v_mad_u64_u32 v[86:87], s[22:23], s50, v48, 0
	v_add3_u32 v18, v87, v49, v18
	v_sub_u32_e32 v49, v83, v18
	v_sub_co_u32_e32 v85, vcc, v85, v86
	s_nop 1
	v_subb_co_u32_e64 v49, s[22:23], v49, v42, vcc
	v_subrev_co_u32_e64 v86, s[22:23], s50, v85
	v_subb_co_u32_e32 v18, vcc, v83, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v49, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v86
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v85
	v_cndmask_b32_e64 v86, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v49
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v49, v87, v86, s[22:23]
	v_add_u32_e32 v86, 2, v48
	v_add_u32_e32 v87, 1, v48
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	v_cndmask_b32_e32 v18, v83, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v49, v87, v86, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v48, v49, vcc
	v_xor_b32_e32 v48, s60, v84
	v_xor_b32_e32 v18, v18, v48
	v_sub_u32_e32 v18, v18, v48
	v_ashrrev_i32_e32 v48, 31, v47
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v88, v46, v48
	v_xad_u32 v83, v18, v40, v51
	v_xor_b32_e32 v49, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v88, s7, 0
	v_mul_hi_u32 v18, v88, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[86:87], s[22:23], v49, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v86
	v_mad_u64_u32 v[84:85], s[22:23], v49, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v87, vcc
	s_nop 1
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[84:85]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v84, s50, v47
	v_mad_u64_u32 v[46:47], s[22:23], s50, v46, 0
	v_add3_u32 v18, v47, v84, v18
	v_sub_u32_e32 v47, v49, v18
	v_sub_co_u32_e32 v46, vcc, v88, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v84, s[22:23], s50, v46
	v_subb_co_u32_e32 v18, vcc, v49, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[24:25], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v85
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v86, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v84
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v85
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v86, v86, v87, s[24:25]
	v_subrev_co_u32_e64 v87, s[22:23], s50, v84
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v86
	s_nop 1
	v_cndmask_b32_e64 v47, v85, v47, s[22:23]
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v49, v49, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_cndmask_b32_e64 v47, v84, v87, s[22:23]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v18, v48
	v_xor_b32_e32 v18, v46, v48
	v_sub_co_u32_e32 v18, vcc, v18, v48
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v47, v48, vcc
	v_ashrrev_i32_e32 v46, 31, v47
	v_and_b32_e32 v46, s42, v46
	v_mul_lo_u32 v47, v83, s33
	v_add3_u32 v90, v46, v18, v47
	v_lshl_add_u64 v[46:47], v[32:33], 0, -2
	v_lshl_add_u64 v[48:49], v[28:29], 0, s[64:65]
	v_cndmask_b32_e64 v49, v47, v49, s[18:19]
	v_ashrrev_i32_e32 v84, 31, v49
	v_cndmask_b32_e64 v48, v46, v48, s[18:19]
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[84:85]
	v_xor_b32_e32 v85, v48, v84
	v_xor_b32_e32 v83, v49, v84
	v_mad_u64_u32 v[48:49], s[22:23], v85, s7, 0
	v_mul_hi_u32 v18, v85, s6
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[88:89], s[22:23], v83, s6, 0
	v_add_co_u32_e32 v18, vcc, v48, v88
	v_mad_u64_u32 v[86:87], s[22:23], v83, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v49, v89, vcc
	s_nop 1
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[86:87]
	v_mul_lo_u32 v18, s51, v48
	v_mul_lo_u32 v49, s50, v49
	v_mad_u64_u32 v[86:87], s[22:23], s50, v48, 0
	v_add3_u32 v18, v87, v49, v18
	v_sub_u32_e32 v49, v83, v18
	v_sub_co_u32_e32 v85, vcc, v85, v86
	s_nop 1
	v_subb_co_u32_e64 v49, s[22:23], v49, v42, vcc
	v_subrev_co_u32_e64 v86, s[22:23], s50, v85
	v_subb_co_u32_e32 v18, vcc, v83, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v49, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v86
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v85
	v_cndmask_b32_e64 v86, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v49
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v49, v87, v86, s[22:23]
	v_add_u32_e32 v86, 2, v48
	v_add_u32_e32 v87, 1, v48
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	v_cndmask_b32_e32 v18, v83, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v49, v87, v86, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v48, v49, vcc
	v_xor_b32_e32 v48, s60, v84
	v_xor_b32_e32 v18, v18, v48
	v_sub_u32_e32 v18, v18, v48
	v_ashrrev_i32_e32 v48, 31, v47
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v88, v46, v48
	v_xad_u32 v83, v18, v40, v51
	v_xor_b32_e32 v49, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v88, s7, 0
	v_mul_hi_u32 v18, v88, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[86:87], s[22:23], v49, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v86
	v_mad_u64_u32 v[84:85], s[22:23], v49, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v87, vcc
	s_nop 1
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[84:85]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v84, s50, v47
	v_mad_u64_u32 v[46:47], s[22:23], s50, v46, 0
	v_add3_u32 v18, v47, v84, v18
	v_sub_u32_e32 v47, v49, v18
	v_sub_co_u32_e32 v46, vcc, v88, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v84, s[22:23], s50, v46
	v_subb_co_u32_e32 v18, vcc, v49, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[24:25], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v85
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v86, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v84
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v85
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v86, v86, v87, s[24:25]
	v_subrev_co_u32_e64 v87, s[22:23], s50, v84
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v86
	s_nop 1
	v_cndmask_b32_e64 v47, v85, v47, s[22:23]
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v49, v49, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_cndmask_b32_e64 v47, v84, v87, s[22:23]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v18, v48
	v_xor_b32_e32 v18, v46, v48
	v_sub_co_u32_e32 v18, vcc, v18, v48
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v47, v48, vcc
	v_ashrrev_i32_e32 v46, 31, v47
	v_and_b32_e32 v46, s42, v46
	v_mul_lo_u32 v47, v83, s33
	v_add3_u32 v91, v46, v18, v47
	v_lshl_add_u64 v[46:47], v[32:33], 0, -1
	v_lshl_add_u64 v[48:49], v[28:29], 0, s[58:59]
	v_cndmask_b32_e64 v49, v47, v49, s[18:19]
	v_ashrrev_i32_e32 v84, 31, v49
	v_cndmask_b32_e64 v48, v46, v48, s[18:19]
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[84:85]
	v_xor_b32_e32 v85, v48, v84
	v_xor_b32_e32 v83, v49, v84
	v_mad_u64_u32 v[48:49], s[22:23], v85, s7, 0
	v_mul_hi_u32 v18, v85, s6
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[88:89], s[22:23], v83, s6, 0
	v_add_co_u32_e32 v18, vcc, v48, v88
	v_mad_u64_u32 v[86:87], s[22:23], v83, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v49, v89, vcc
	v_lshl_add_u64 v[28:29], v[28:29], 0, s[4:5]
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[86:87]
	v_mul_lo_u32 v18, s51, v48
	v_mul_lo_u32 v49, s50, v49
	v_mad_u64_u32 v[86:87], s[22:23], s50, v48, 0
	v_add3_u32 v18, v87, v49, v18
	v_sub_u32_e32 v49, v83, v18
	v_sub_co_u32_e32 v85, vcc, v85, v86
	v_cndmask_b32_e64 v29, v33, v29, s[18:19]
	s_nop 0
	v_subb_co_u32_e64 v49, s[22:23], v49, v42, vcc
	v_subrev_co_u32_e64 v86, s[22:23], s50, v85
	v_subb_co_u32_e32 v18, vcc, v83, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v49, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s51, v49
	v_cmp_le_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v28, v32, v28, s[18:19]
	v_cndmask_b32_e64 v87, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v86
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v85
	v_cndmask_b32_e64 v86, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v49
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v49, v87, v86, s[22:23]
	v_add_u32_e32 v86, 2, v48
	v_add_u32_e32 v87, 1, v48
	v_cmp_ne_u32_e64 s[22:23], 0, v49
	v_cndmask_b32_e32 v18, v83, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v49, v87, v86, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v18, v48, v49, vcc
	v_xor_b32_e32 v48, s60, v84
	v_xor_b32_e32 v18, v18, v48
	v_sub_u32_e32 v18, v18, v48
	v_ashrrev_i32_e32 v48, 31, v47
	v_mov_b32_e32 v49, v48
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[48:49]
	v_xor_b32_e32 v88, v46, v48
	v_xad_u32 v83, v18, v40, v51
	v_xor_b32_e32 v49, v47, v48
	v_mad_u64_u32 v[46:47], s[22:23], v88, s7, 0
	v_mul_hi_u32 v18, v88, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[86:87], s[22:23], v49, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v86
	v_mad_u64_u32 v[84:85], s[22:23], v49, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v47, v87, vcc
	s_nop 1
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[84:85]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v84, s50, v47
	v_mad_u64_u32 v[46:47], s[22:23], s50, v46, 0
	v_add3_u32 v18, v47, v84, v18
	v_sub_u32_e32 v47, v49, v18
	v_sub_co_u32_e32 v46, vcc, v88, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, vcc
	v_subrev_co_u32_e64 v84, s[22:23], s50, v46
	v_subb_co_u32_e32 v18, vcc, v49, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[24:25], 0, v47, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s51, v85
	v_subb_co_u32_e64 v47, s[22:23], v47, v42, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v86, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s50, v84
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s51, v85
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v86, v86, v87, s[24:25]
	v_subrev_co_u32_e64 v87, s[22:23], s50, v84
	s_nop 1
	v_subbrev_co_u32_e64 v47, s[22:23], 0, v47, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v86
	s_nop 1
	v_cndmask_b32_e64 v47, v85, v47, s[22:23]
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v49, v49, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_cndmask_b32_e64 v47, v84, v87, s[22:23]
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_xor_b32_e32 v47, v18, v48
	v_xor_b32_e32 v18, v46, v48
	v_sub_co_u32_e32 v18, vcc, v18, v48
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v47, v48, vcc
	v_ashrrev_i32_e32 v46, 31, v47
	v_and_b32_e32 v46, s42, v46
	v_mul_lo_u32 v47, v83, s33
	v_add3_u32 v88, v46, v18, v47
	v_ashrrev_i32_e32 v46, 31, v29
	v_mov_b32_e32 v47, v46
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[46:47]
	v_xor_b32_e32 v83, v28, v46
	v_xor_b32_e32 v47, v29, v46
	v_mad_u64_u32 v[28:29], s[18:19], v83, s7, 0
	v_mul_hi_u32 v18, v83, s6
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[28:29]
	v_mad_u64_u32 v[84:85], s[18:19], v47, s6, 0
	v_add_co_u32_e32 v18, vcc, v28, v84
	v_mad_u64_u32 v[48:49], s[18:19], v47, s7, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v29, v85, vcc
	s_nop 1
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[28:29], v[18:19], 0, v[48:49]
	v_mul_lo_u32 v18, s51, v28
	v_mul_lo_u32 v29, s50, v29
	v_mad_u64_u32 v[48:49], s[18:19], s50, v28, 0
	v_add3_u32 v18, v49, v29, v18
	v_sub_u32_e32 v29, v47, v18
	v_sub_co_u32_e32 v48, vcc, v83, v48
	s_nop 1
	v_subb_co_u32_e64 v29, s[18:19], v29, v42, vcc
	v_subrev_co_u32_e64 v49, s[18:19], s50, v48
	v_subb_co_u32_e32 v18, vcc, v47, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v29, s[18:19], 0, v29, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s51, v29
	v_cmp_le_u32_e32 vcc, s51, v18
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s50, v49
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v48
	v_cndmask_b32_e64 v49, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s51, v29
	v_cndmask_b32_e64 v48, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	v_cndmask_b32_e64 v29, v83, v49, s[18:19]
	v_add_u32_e32 v49, 2, v28
	v_add_u32_e32 v83, 1, v28
	v_cmp_ne_u32_e64 s[18:19], 0, v29
	v_cndmask_b32_e32 v18, v47, v48, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	v_cndmask_b32_e64 v29, v83, v49, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v18, v28, v29, vcc
	v_xor_b32_e32 v28, s60, v46
	v_xor_b32_e32 v18, v18, v28
	v_sub_u32_e32 v18, v18, v28
	v_ashrrev_i32_e32 v28, 31, v33
	v_mov_b32_e32 v29, v28
	v_lshl_add_u64 v[46:47], v[32:33], 0, v[28:29]
	v_xor_b32_e32 v83, v46, v28
	v_xad_u32 v40, v18, v40, v51
	v_xor_b32_e32 v29, v47, v28
	v_mad_u64_u32 v[46:47], s[18:19], v83, s7, 0
	v_mul_hi_u32 v18, v83, s6
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[46:47]
	v_mad_u64_u32 v[48:49], s[18:19], v29, s7, 0
	v_mad_u64_u32 v[84:85], s[6:7], v29, s6, 0
	v_add_co_u32_e32 v18, vcc, v46, v84
	s_nop 1
	v_addc_co_u32_e32 v18, vcc, v47, v85, vcc
	s_nop 1
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[46:47], v[18:19], 0, v[48:49]
	v_mul_lo_u32 v18, s51, v46
	v_mul_lo_u32 v48, s50, v47
	v_mad_u64_u32 v[46:47], s[6:7], s50, v46, 0
	v_add3_u32 v18, v47, v48, v18
	v_sub_u32_e32 v47, v29, v18
	v_sub_co_u32_e32 v46, vcc, v83, v46
	s_nop 1
	v_subb_co_u32_e64 v47, s[18:19], v47, v42, vcc
	v_subrev_co_u32_e64 v48, s[18:19], s50, v46
	v_subb_co_u32_e32 v18, vcc, v29, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[22:23], 0, v47, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s51, v49
	v_subb_co_u32_e64 v42, s[18:19], v47, v42, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s50, v48
	v_subrev_co_u32_e64 v47, s[18:19], s50, v48
	s_nop 0
	v_cndmask_b32_e64 v84, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s51, v49
	v_subbrev_co_u32_e64 v42, s[18:19], 0, v42, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v83, v83, v84, s[22:23]
	v_cmp_le_u32_e32 vcc, s51, v18
	v_cmp_ne_u32_e64 s[18:19], 0, v83
	s_nop 0
	v_cndmask_b32_e64 v29, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s50, v46
	v_cndmask_b32_e64 v42, v49, v42, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s51, v18
	s_nop 1
	v_cndmask_b32_e32 v29, v29, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v29
	v_cndmask_b32_e64 v29, v48, v47, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v29, v46, v29, vcc
	v_cndmask_b32_e32 v18, v18, v42, vcc
	v_xor_b32_e32 v29, v29, v28
	v_xor_b32_e32 v18, v18, v28
	v_sub_co_u32_e32 v29, vcc, v29, v28
	s_nop 1
	v_subb_co_u32_e32 v18, vcc, v18, v28, vcc
	v_ashrrev_i32_e32 v18, 31, v18
	v_and_b32_e32 v18, s42, v18
	v_mul_lo_u32 v28, v40, s33
	v_add3_u32 v18, v18, v29, v28
	buffer_load_ubyte v83, v41, s[28:31], 0 offen
	buffer_load_ubyte v85, v43, s[28:31], 0 offen
	buffer_load_ubyte v84, v44, s[28:31], 0 offen
	buffer_load_ubyte v86, v45, s[28:31], 0 offen
	buffer_load_ubyte v87, v90, s[28:31], 0 offen
	buffer_load_ubyte v89, v91, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v88, v88, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v90, v18, s[28:31], 0 offen
	ds_read2st64_b32 v[28:29], v54 offset1:1
	ds_read_b128 v[120:123], v55
	ds_read_b128 v[92:95], v55 offset:2048
	ds_read_b128 v[112:115], v56
	ds_read_b128 v[100:103], v56 offset:2048
	ds_read_b128 v[104:107], v55 offset:4096
	ds_read_b128 v[96:99], v55 offset:6144
	ds_read_b128 v[116:119], v56 offset:4096
	ds_read_b128 v[108:111], v56 offset:6144
	v_lshl_add_u64 v[44:45], s[56:57], 0, v[34:35]
	v_or_b32_e32 v18, s55, v45
	v_cmp_ne_u32_e32 vcc, 0, v18
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
	s_cbranch_execz .LBB0_74
	s_add_u32 s18, s54, s96
	s_mov_b32 s97, s96
	s_addc_u32 s19, s55, s96
	s_xor_b64 s[24:25], s[18:19], s[96:97]
	v_cvt_f32_u32_e32 v18, s24
	v_cvt_f32_u32_e32 v40, s25
	s_sub_u32 s4, 0, s24
	s_subb_u32 s5, 0, s25
	v_ashrrev_i32_e32 v42, 31, v45
	v_fmac_f32_e32 v18, 0x4f800000, v40
	v_rcp_f32_e32 v18, v18
	v_mov_b32_e32 v43, v42
	v_mul_f32_e32 v18, 0x5f7ffffc, v18
	v_mul_f32_e32 v40, 0x2f800000, v18
	v_trunc_f32_e32 v40, v40
	v_fmac_f32_e32 v18, 0xcf800000, v40
	v_cvt_u32_f32_e32 v40, v40
	v_cvt_u32_f32_e32 v18, v18
	v_readfirstlane_b32 s18, v40
	v_readfirstlane_b32 s19, v18
	s_mul_i32 s22, s4, s18
	s_mul_hi_u32 s50, s4, s19
	s_mul_i32 s23, s5, s19
	s_add_i32 s22, s50, s22
	s_add_i32 s22, s22, s23
	s_mul_i32 s51, s4, s19
	s_mul_i32 s50, s19, s22
	s_mul_hi_u32 s97, s19, s51
	s_mul_hi_u32 s23, s19, s22
	s_add_u32 s50, s97, s50
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 vcc_lo, s18, s51
	s_mul_i32 s51, s18, s51
	s_add_u32 s50, s50, s51
	s_mul_hi_u32 s97, s18, s22
	s_addc_u32 s23, s23, vcc_lo
	s_addc_u32 s50, s97, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s22, s23, s22
	s_addc_u32 s23, 0, s50
	s_add_u32 s19, s19, s22
	s_addc_u32 s18, s18, s23
	s_mul_i32 s22, s4, s18
	s_mul_hi_u32 s23, s4, s19
	s_add_i32 s22, s23, s22
	s_mul_i32 s5, s5, s19
	s_add_i32 s22, s22, s5
	s_mul_i32 s4, s4, s19
	s_mul_hi_u32 s23, s18, s4
	s_mul_i32 s50, s18, s4
	s_mul_i32 s97, s19, s22
	s_mul_hi_u32 s4, s19, s4
	s_mul_hi_u32 s51, s19, s22
	s_add_u32 s4, s4, s97
	s_addc_u32 s51, 0, s51
	s_add_u32 s4, s4, s50
	s_mul_hi_u32 s5, s18, s22
	s_addc_u32 s4, s51, s23
	s_addc_u32 s5, s5, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s4, s4, s22
	s_addc_u32 s5, 0, s5
	s_add_u32 s4, s19, s4
	v_lshl_add_u64 v[40:41], v[44:45], 0, v[42:43]
	s_addc_u32 s5, s18, s5
	v_xor_b32_e32 v91, v40, v42
	v_xor_b32_e32 v43, v41, v42
	v_mad_u64_u32 v[40:41], s[18:19], v91, s5, 0
	v_mul_hi_u32 v18, v91, s4
	v_lshl_add_u64 v[40:41], v[18:19], 0, v[40:41]
	v_mad_u64_u32 v[48:49], s[18:19], v43, s4, 0
	v_add_co_u32_e32 v18, vcc, v40, v48
	v_mad_u64_u32 v[46:47], s[18:19], v43, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v40, vcc, v41, v49, vcc
	v_mov_b32_e32 v41, s46
	s_nop 0
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[40:41], v[40:41], 0, v[46:47]
	v_mul_lo_u32 v18, s25, v40
	v_mul_lo_u32 v46, s24, v41
	v_mad_u64_u32 v[40:41], s[18:19], s24, v40, 0
	v_add3_u32 v18, v41, v46, v18
	v_sub_u32_e32 v41, v43, v18
	v_mov_b32_e32 v46, s25
	v_sub_co_u32_e32 v40, vcc, v91, v40
	s_nop 1
	v_subb_co_u32_e64 v41, s[18:19], v41, v46, vcc
	v_subrev_co_u32_e64 v47, s[18:19], s24, v40
	v_subb_co_u32_e32 v18, vcc, v43, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v48, s[22:23], 0, v41, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s25, v48
	v_subb_co_u32_e64 v41, s[18:19], v41, v46, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s24, v47
	v_subrev_co_u32_e64 v46, s[18:19], s24, v47
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s25, v48
	v_subbrev_co_u32_e64 v41, s[18:19], 0, v41, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v49, v49, v91, s[22:23]
	v_cmp_le_u32_e32 vcc, s25, v18
	v_cmp_ne_u32_e64 s[18:19], 0, v49
	s_nop 0
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s24, v40
	v_cndmask_b32_e64 v41, v48, v41, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v48, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s25, v18
	s_nop 1
	v_cndmask_b32_e32 v43, v43, v48, vcc
	v_cmp_ne_u32_e32 vcc, 0, v43
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v41, vcc
	v_cndmask_b32_e64 v41, v47, v46, s[18:19]
	v_cndmask_b32_e32 v40, v40, v41, vcc
	v_xor_b32_e32 v40, v40, v42
	v_xor_b32_e32 v18, v18, v42
	v_sub_co_u32_e32 v40, vcc, v40, v42
	s_nop 1
	v_subb_co_u32_e32 v41, vcc, v18, v42, vcc
.LBB0_74:
	s_or_saveexec_b64 s[6:7], s[6:7]
	v_cvt_f32_u32_e32 v91, s54
	s_xor_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB0_76
	v_rcp_iflag_f32_e32 v18, v91
	s_sub_i32 s4, 0, s54
	v_mul_f32_e32 v18, 0x4f7ffffe, v18
	v_cvt_u32_f32_e32 v18, v18
	v_mul_lo_u32 v40, s4, v18
	v_mul_hi_u32 v40, v18, v40
	v_add_u32_e32 v18, v18, v40
	v_mul_hi_u32 v18, v44, v18
	v_mul_lo_u32 v18, v18, s54
	v_sub_u32_e32 v18, v44, v18
	v_subrev_u32_e32 v40, s54, v18
	v_cmp_le_u32_e32 vcc, s54, v18
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v40, vcc
	v_subrev_u32_e32 v40, s54, v18
	v_cmp_le_u32_e32 vcc, s54, v18
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v40, vcc
	v_mov_b64_e32 v[40:41], v[18:19]
.LBB0_76:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[42:43], s[56:57], 0, v[38:39]
	v_lshl_add_u64 v[46:47], v[42:43], 0, 2
	v_cmp_gt_i32_e32 vcc, 0, v43
	s_nop 1
	v_cndmask_b32_e32 v49, v47, v37, vcc
	v_or_b32_e32 v18, s55, v49
	v_cndmask_b32_e32 v48, v46, v36, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
	s_cbranch_execz .LBB0_78
	s_add_u32 s18, s54, s96
	s_mov_b32 s97, s96
	s_addc_u32 s19, s55, s96
	s_xor_b64 s[22:23], s[18:19], s[96:97]
	v_cvt_f32_u32_e32 v18, s22
	v_cvt_f32_u32_e32 v46, s23
	s_sub_u32 s4, 0, s22
	s_subb_u32 s5, 0, s23
	v_fmac_f32_e32 v18, 0x4f800000, v46
	v_rcp_f32_e32 v18, v18
	s_nop 0
	v_mul_f32_e32 v18, 0x5f7ffffc, v18
	v_mul_f32_e32 v46, 0x2f800000, v18
	v_trunc_f32_e32 v46, v46
	v_fmac_f32_e32 v18, 0xcf800000, v46
	v_cvt_u32_f32_e32 v46, v46
	v_cvt_u32_f32_e32 v18, v18
	v_readfirstlane_b32 s18, v46
	v_readfirstlane_b32 s19, v18
	s_mul_i32 s24, s4, s18
	s_mul_hi_u32 s50, s4, s19
	s_mul_i32 s25, s5, s19
	s_add_i32 s24, s50, s24
	s_add_i32 s24, s24, s25
	s_mul_i32 s51, s4, s19
	s_mul_i32 s50, s19, s24
	s_mul_hi_u32 s97, s19, s51
	s_mul_hi_u32 s25, s19, s24
	s_add_u32 s50, s97, s50
	s_addc_u32 s25, 0, s25
	s_mul_hi_u32 vcc_lo, s18, s51
	s_mul_i32 s51, s18, s51
	s_add_u32 s50, s50, s51
	s_mul_hi_u32 s97, s18, s24
	s_addc_u32 s25, s25, vcc_lo
	s_addc_u32 s50, s97, 0
	s_mul_i32 s24, s18, s24
	s_add_u32 s24, s25, s24
	s_addc_u32 s25, 0, s50
	s_add_u32 s19, s19, s24
	s_addc_u32 s18, s18, s25
	s_mul_i32 s24, s4, s18
	s_mul_hi_u32 s25, s4, s19
	s_add_i32 s24, s25, s24
	s_mul_i32 s5, s5, s19
	s_add_i32 s24, s24, s5
	s_mul_i32 s4, s4, s19
	s_mul_hi_u32 s25, s18, s4
	s_mul_i32 s50, s18, s4
	s_mul_i32 s97, s19, s24
	s_mul_hi_u32 s4, s19, s4
	s_mul_hi_u32 s51, s19, s24
	s_add_u32 s4, s4, s97
	s_addc_u32 s51, 0, s51
	s_add_u32 s4, s4, s50
	s_mul_hi_u32 s5, s18, s24
	s_addc_u32 s4, s51, s25
	s_addc_u32 s5, s5, 0
	s_mul_i32 s24, s18, s24
	s_add_u32 s4, s4, s24
	v_ashrrev_i32_e32 v46, 31, v49
	s_addc_u32 s5, 0, s5
	v_mov_b32_e32 v47, v46
	s_add_u32 s4, s19, s4
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[46:47]
	s_addc_u32 s5, s18, s5
	v_xor_b32_e32 v128, v48, v46
	v_xor_b32_e32 v47, v49, v46
	v_mad_u64_u32 v[48:49], s[18:19], v128, s5, 0
	v_mul_hi_u32 v18, v128, s4
	v_lshl_add_u64 v[48:49], v[18:19], 0, v[48:49]
	v_mad_u64_u32 v[126:127], s[18:19], v47, s4, 0
	v_add_co_u32_e32 v18, vcc, v48, v126
	v_mad_u64_u32 v[124:125], s[18:19], v47, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v48, vcc, v49, v127, vcc
	v_mov_b32_e32 v49, s46
	s_nop 0
	v_addc_co_u32_e32 v125, vcc, 0, v125, vcc
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[124:125]
	v_mul_lo_u32 v18, s23, v48
	v_mul_lo_u32 v126, s22, v49
	v_mad_u64_u32 v[124:125], s[18:19], s22, v48, 0
	v_add3_u32 v18, v125, v126, v18
	v_sub_u32_e32 v125, v47, v18
	v_mov_b32_e32 v126, s23
	v_sub_co_u32_e32 v128, vcc, v128, v124
	v_xor_b32_e32 v46, s96, v46
	s_nop 0
	v_subb_co_u32_e64 v124, s[18:19], v125, v126, vcc
	v_subrev_co_u32_e64 v125, s[18:19], s22, v128
	v_subb_co_u32_e32 v18, vcc, v47, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v124, s[18:19], 0, v124, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s23, v124
	v_cmp_le_u32_e32 vcc, s23, v18
	s_nop 0
	v_cndmask_b32_e64 v126, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s22, v125
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s22, v128
	v_cndmask_b32_e64 v125, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s23, v124
	s_nop 1
	v_cndmask_b32_e64 v129, v126, v125, s[18:19]
	v_lshl_add_u64 v[124:125], v[48:49], 0, 2
	v_lshl_add_u64 v[126:127], v[48:49], 0, 1
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s23, v18
	s_nop 1
	v_cndmask_b32_e32 v18, v47, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v129
	s_nop 1
	v_cndmask_b32_e32 v47, v126, v124, vcc
	v_cmp_ne_u32_e32 vcc, 0, v18
	s_nop 1
	v_cndmask_b32_e32 v18, v48, v47, vcc
	v_xor_b32_e32 v18, v18, v46
	v_sub_co_u32_e32 v46, vcc, v18, v46
.LBB0_78:
	s_andn2_saveexec_b64 s[6:7], s[6:7]
	s_cbranch_execz .LBB0_80
	v_rcp_iflag_f32_e32 v18, v91
	s_sub_i32 s4, 0, s54
	v_mul_f32_e32 v18, 0x4f7ffffe, v18
	v_cvt_u32_f32_e32 v18, v18
	v_mul_lo_u32 v46, s4, v18
	v_mul_hi_u32 v46, v18, v46
	v_add_u32_e32 v18, v18, v46
	v_mul_hi_u32 v18, v48, v18
	v_mul_lo_u32 v46, v18, s54
	v_sub_u32_e32 v46, v48, v46
	v_add_u32_e32 v47, 1, v18
	v_subrev_u32_e32 v48, s54, v46
	v_cmp_le_u32_e32 vcc, s54, v46
	s_nop 1
	v_cndmask_b32_e32 v46, v46, v48, vcc
	v_cndmask_b32_e32 v18, v18, v47, vcc
	v_add_u32_e32 v47, 1, v18
	v_cmp_le_u32_e32 vcc, s54, v46
	s_nop 1
	v_cndmask_b32_e32 v46, v18, v47, vcc
.LBB0_80:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[48:49], v[44:45], 0, 2
	v_or_b32_e32 v18, s55, v49
	v_cmp_ne_u32_e32 vcc, 0, v18
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
	s_cbranch_execz .LBB0_82
	s_add_u32 s18, s54, s96
	s_mov_b32 s97, s96
	s_addc_u32 s19, s55, s96
	s_xor_b64 s[24:25], s[18:19], s[96:97]
	v_cvt_f32_u32_e32 v18, s24
	v_cvt_f32_u32_e32 v44, s25
	s_sub_u32 s4, 0, s24
	s_subb_u32 s5, 0, s25
	v_ashrrev_i32_e32 v124, 31, v49
	v_fmac_f32_e32 v18, 0x4f800000, v44
	v_rcp_f32_e32 v18, v18
	v_mov_b32_e32 v125, v124
	v_mul_f32_e32 v18, 0x5f7ffffc, v18
	v_mul_f32_e32 v44, 0x2f800000, v18
	v_trunc_f32_e32 v44, v44
	v_fmac_f32_e32 v18, 0xcf800000, v44
	v_cvt_u32_f32_e32 v44, v44
	v_cvt_u32_f32_e32 v18, v18
	v_readfirstlane_b32 s18, v44
	v_readfirstlane_b32 s19, v18
	s_mul_i32 s22, s4, s18
	s_mul_hi_u32 s50, s4, s19
	s_mul_i32 s23, s5, s19
	s_add_i32 s22, s50, s22
	s_add_i32 s22, s22, s23
	s_mul_i32 s51, s4, s19
	s_mul_i32 s50, s19, s22
	s_mul_hi_u32 s97, s19, s51
	s_mul_hi_u32 s23, s19, s22
	s_add_u32 s50, s97, s50
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 vcc_lo, s18, s51
	s_mul_i32 s51, s18, s51
	s_add_u32 s50, s50, s51
	s_mul_hi_u32 s97, s18, s22
	s_addc_u32 s23, s23, vcc_lo
	s_addc_u32 s50, s97, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s22, s23, s22
	s_addc_u32 s23, 0, s50
	s_add_u32 s19, s19, s22
	s_addc_u32 s18, s18, s23
	s_mul_i32 s22, s4, s18
	s_mul_hi_u32 s23, s4, s19
	s_add_i32 s22, s23, s22
	s_mul_i32 s5, s5, s19
	s_add_i32 s22, s22, s5
	s_mul_i32 s4, s4, s19
	s_mul_hi_u32 s23, s18, s4
	s_mul_i32 s50, s18, s4
	s_mul_i32 s97, s19, s22
	s_mul_hi_u32 s4, s19, s4
	s_mul_hi_u32 s51, s19, s22
	s_add_u32 s4, s4, s97
	s_addc_u32 s51, 0, s51
	s_add_u32 s4, s4, s50
	s_mul_hi_u32 s5, s18, s22
	s_addc_u32 s4, s51, s23
	s_addc_u32 s5, s5, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s4, s4, s22
	s_addc_u32 s5, 0, s5
	s_add_u32 s4, s19, s4
	v_lshl_add_u64 v[44:45], v[48:49], 0, v[124:125]
	s_addc_u32 s5, s18, s5
	v_xor_b32_e32 v91, v44, v124
	v_xor_b32_e32 v47, v45, v124
	v_mad_u64_u32 v[44:45], s[18:19], v91, s5, 0
	v_mul_hi_u32 v18, v91, s4
	v_lshl_add_u64 v[44:45], v[18:19], 0, v[44:45]
	v_mad_u64_u32 v[126:127], s[18:19], v47, s4, 0
	v_add_co_u32_e32 v18, vcc, v44, v126
	v_mad_u64_u32 v[48:49], s[18:19], v47, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v44, vcc, v45, v127, vcc
	v_mov_b32_e32 v45, s46
	s_nop 0
	v_addc_co_u32_e32 v49, vcc, 0, v49, vcc
	v_lshl_add_u64 v[44:45], v[44:45], 0, v[48:49]
	v_mul_lo_u32 v18, s25, v44
	v_mul_lo_u32 v48, s24, v45
	v_mad_u64_u32 v[44:45], s[18:19], s24, v44, 0
	v_add3_u32 v18, v45, v48, v18
	v_sub_u32_e32 v45, v47, v18
	v_mov_b32_e32 v48, s25
	v_sub_co_u32_e32 v44, vcc, v91, v44
	s_nop 1
	v_subb_co_u32_e64 v45, s[18:19], v45, v48, vcc
	v_subrev_co_u32_e64 v49, s[18:19], s24, v44
	v_subb_co_u32_e32 v18, vcc, v47, v18, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[22:23], 0, v45, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s25, v91
	v_subb_co_u32_e64 v45, s[18:19], v45, v48, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v125, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s24, v49
	v_subrev_co_u32_e64 v48, s[18:19], s24, v49
	s_nop 0
	v_cndmask_b32_e64 v126, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s25, v91
	v_subbrev_co_u32_e64 v45, s[18:19], 0, v45, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v125, v125, v126, s[22:23]
	v_cmp_le_u32_e32 vcc, s25, v18
	v_cmp_ne_u32_e64 s[18:19], 0, v125
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s24, v44
	v_cndmask_b32_e64 v45, v91, v45, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s25, v18
	s_nop 1
	v_cndmask_b32_e32 v47, v47, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v47
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v45, vcc
	v_cndmask_b32_e64 v45, v49, v48, s[18:19]
	v_cndmask_b32_e32 v44, v44, v45, vcc
	v_xor_b32_e32 v44, v44, v124
	v_xor_b32_e32 v18, v18, v124
	v_sub_co_u32_e32 v44, vcc, v44, v124
	s_nop 1
	v_subb_co_u32_e32 v45, vcc, v18, v124, vcc
.LBB0_82:
	s_andn2_saveexec_b64 s[6:7], s[6:7]
	s_cbranch_execz .LBB0_67
	v_rcp_iflag_f32_e32 v18, v91
	s_sub_i32 s4, 0, s54
	v_mul_f32_e32 v18, 0x4f7ffffe, v18
	v_cvt_u32_f32_e32 v18, v18
	v_mul_lo_u32 v44, s4, v18
	v_mul_hi_u32 v44, v18, v44
	v_add_u32_e32 v18, v18, v44
	v_mul_hi_u32 v18, v48, v18
	v_mul_lo_u32 v18, v18, s54
	v_sub_u32_e32 v18, v48, v18
	v_subrev_u32_e32 v44, s54, v18
	v_cmp_le_u32_e32 vcc, s54, v18
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v44, vcc
	v_subrev_u32_e32 v44, s54, v18
	v_cmp_le_u32_e32 vcc, s54, v18
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v44, vcc
	v_mov_b64_e32 v[44:45], v[18:19]
	s_branch .LBB0_67
.LBB0_84:
	v_readlane_b32 s60, v144, 22
	v_readlane_b32 s66, v144, 27
	v_readlane_b32 s44, v144, 20
	v_readlane_b32 s61, v144, 23
	v_readlane_b32 s67, v144, 28
	v_readlane_b32 s70, v144, 6
	v_readlane_b32 s71, v144, 29
	v_readlane_b32 s45, v144, 21
.LBB0_85:
	v_readlane_b32 s14, v144, 0
	v_readlane_b32 s15, v144, 1
	s_cmp_lg_u32 s15, 0
	s_cbranch_scc0 .LBB0_121
	s_add_u32 s0, s26, 0
	s_addc_u32 s1, 0, 0
	s_xor_b64 s[0:1], s[0:1], 0
	v_cvt_f32_u32_e32 v18, s0
	v_cvt_f32_u32_e32 v19, s1
	s_sub_u32 s4, 0, s0
	s_subb_u32 s5, 0, s1
	v_fmamk_f32 v18, v19, 0x4f800000, v18
	v_rcp_f32_e32 v18, v18
	s_nop 0
	v_mul_f32_e32 v18, 0x5f7ffffc, v18
	v_mul_f32_e32 v19, 0x2f800000, v18
	v_trunc_f32_e32 v19, v19
	v_fmamk_f32 v18, v19, 0xcf800000, v18
	v_cvt_u32_f32_e32 v19, v19
	v_cvt_u32_f32_e32 v18, v18
	v_readfirstlane_b32 s6, v19
	v_readfirstlane_b32 s7, v18
	s_mul_i32 s8, s4, s6
	s_mul_hi_u32 s10, s4, s7
	s_mul_i32 s9, s5, s7
	s_add_i32 s8, s10, s8
	s_add_i32 s8, s8, s9
	s_mul_i32 s11, s4, s7
	s_mul_i32 s10, s7, s8
	s_mul_hi_u32 s12, s7, s11
	s_mul_hi_u32 s9, s7, s8
	s_add_u32 s10, s12, s10
	s_addc_u32 s9, 0, s9
	s_mul_hi_u32 s13, s6, s11
	s_mul_i32 s11, s6, s11
	s_add_u32 s10, s10, s11
	s_mul_hi_u32 s12, s6, s8
	s_addc_u32 s9, s9, s13
	s_addc_u32 s10, s12, 0
	s_mul_i32 s8, s6, s8
	s_add_u32 s8, s9, s8
	s_addc_u32 s9, 0, s10
	s_add_u32 s7, s7, s8
	s_addc_u32 s6, s6, s9
	s_mul_i32 s8, s4, s6
	s_mul_hi_u32 s9, s4, s7
	s_add_i32 s8, s9, s8
	s_mul_i32 s5, s5, s7
	s_add_i32 s8, s8, s5
	s_mul_i32 s4, s4, s7
	s_mul_hi_u32 s9, s6, s4
	s_mul_i32 s10, s6, s4
	s_mul_i32 s12, s7, s8
	s_mul_hi_u32 s4, s7, s4
	s_mul_hi_u32 s11, s7, s8
	s_add_u32 s4, s4, s12
	s_addc_u32 s11, 0, s11
	s_add_u32 s4, s4, s10
	s_mul_hi_u32 s5, s6, s8
	s_addc_u32 s4, s11, s9
	s_addc_u32 s5, s5, 0
	s_mul_i32 s8, s6, s8
	s_add_u32 s4, s4, s8
	s_addc_u32 s5, 0, s5
	s_add_u32 s4, s7, s4
	s_addc_u32 s5, s6, s5
	s_ashr_i32 s6, s15, 31
	s_add_u32 s8, s14, s6
	s_mov_b32 s7, s6
	s_addc_u32 s9, s15, s6
	s_xor_b64 s[8:9], s[8:9], s[6:7]
	s_mul_i32 s11, s8, s5
	s_mul_hi_u32 s12, s8, s4
	s_mul_hi_u32 s10, s8, s5
	s_add_u32 s11, s12, s11
	s_addc_u32 s10, 0, s10
	s_mul_hi_u32 s13, s9, s4
	s_mul_i32 s4, s9, s4
	s_add_u32 s4, s11, s4
	s_mul_hi_u32 s12, s9, s5
	s_addc_u32 s4, s10, s13
	s_addc_u32 s10, s12, 0
	s_mul_i32 s5, s9, s5
	s_add_u32 s4, s4, s5
	s_addc_u32 s5, 0, s10
	s_mul_i32 s5, s0, s5
	s_mul_hi_u32 s10, s0, s4
	s_add_i32 s5, s10, s5
	s_mul_i32 s10, s1, s4
	s_add_i32 s5, s5, s10
	s_sub_i32 s12, s9, s5
	s_mul_i32 s4, s0, s4
	s_sub_u32 s4, s8, s4
	s_cselect_b64 s[10:11], -1, 0
	s_subb_u32 s8, s12, s1
	s_sub_u32 s14, s4, s0
	s_cselect_b64 s[12:13], -1, 0
	s_subb_u32 s15, s8, 0
	s_cmp_ge_u32 s15, s1
	s_cselect_b32 s18, -1, 0
	s_cmp_ge_u32 s14, s0
	s_cselect_b32 s19, -1, 0
	s_cmp_eq_u32 s15, s1
	s_cselect_b32 s18, s19, s18
	s_cmp_lg_u64 s[12:13], 0
	s_subb_u32 s8, s8, s1
	s_sub_u32 s12, s14, s0
	s_subb_u32 s8, s8, 0
	s_cmp_lg_u32 s18, 0
	s_cselect_b32 s12, s12, s14
	s_cselect_b32 s8, s8, s15
	s_cmp_lg_u64 s[10:11], 0
	s_subb_u32 s5, s9, s5
	s_cmp_ge_u32 s5, s1
	s_cselect_b32 s9, -1, 0
	s_cmp_ge_u32 s4, s0
	s_cselect_b32 s0, -1, 0
	s_cmp_eq_u32 s5, s1
	s_cselect_b32 s0, s0, s9
	s_cmp_lg_u32 s0, 0
	s_cselect_b32 s1, s8, s5
	s_cselect_b32 s0, s12, s4
	s_xor_b64 s[0:1], s[0:1], s[6:7]
	s_sub_u32 s0, s0, s6
	s_subb_u32 s1, s1, s6
	s_cbranch_execnz .LBB0_88
.LBB0_87:
	v_rcp_iflag_f32_e32 v17, v17
	s_sub_i32 s0, 0, s26
	v_readlane_b32 s4, v144, 0
	s_mov_b32 s1, 0
	v_mul_f32_e32 v17, 0x4f7ffffe, v17
	v_cvt_u32_f32_e32 v17, v17
	v_readlane_b32 s5, v144, 1
	v_readfirstlane_b32 s2, v17
	s_mul_i32 s0, s0, s2
	s_mul_hi_u32 s0, s2, s0
	s_add_i32 s2, s2, s0
	s_mul_hi_u32 s0, s4, s2
	s_mul_i32 s0, s0, s26
	s_sub_i32 s0, s4, s0
	s_sub_i32 s2, s0, s26
	s_cmp_ge_u32 s0, s26
	s_cselect_b32 s0, s2, s0
	s_sub_i32 s2, s0, s26
	s_cmp_ge_u32 s0, s26
	s_cselect_b32 s0, s2, s0
.LBB0_88:
	v_bfe_u32 v17, v15, 16, 1
	s_movk_i32 s2, 0x7fff
	v_bfe_u32 v18, v14, 16, 1
	v_add3_u32 v17, v15, v17, s2
	v_bfe_u32 v19, v13, 16, 1
	v_add3_u32 v18, v14, v18, s2
	v_lshrrev_b32_e32 v17, 16, v17
	v_mov_b32_e32 v27, 0x7fc0
	v_cmp_o_f32_e32 vcc, v15, v15
	v_add3_u32 v19, v13, v19, s2
	v_lshrrev_b32_e32 v18, 16, v18
	v_cndmask_b32_e32 v21, v27, v17, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_bfe_u32 v20, v12, 16, 1
	v_lshrrev_b32_e32 v19, 16, v19
	v_cndmask_b32_e32 v22, v27, v18, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_add3_u32 v20, v12, v20, s2
	v_lshrrev_b32_e32 v20, 16, v20
	v_cndmask_b32_e32 v23, v27, v19, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v12, v11, 16, 1
	v_bfe_u32 v13, v10, 16, 1
	v_add3_u32 v12, v11, v12, s2
	v_cndmask_b32_e32 v26, v27, v20, vcc
	v_bfe_u32 v14, v9, 16, 1
	v_add3_u32 v13, v10, v13, s2
	v_lshrrev_b32_e32 v12, 16, v12
	v_cmp_o_f32_e32 vcc, v11, v11
	v_add3_u32 v14, v9, v14, s2
	v_lshrrev_b32_e32 v13, 16, v13
	v_cndmask_b32_e32 v17, v27, v12, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_bfe_u32 v15, v8, 16, 1
	v_lshrrev_b32_e32 v14, 16, v14
	v_cndmask_b32_e32 v18, v27, v13, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_add3_u32 v15, v8, v15, s2
	v_lshrrev_b32_e32 v15, 16, v15
	v_cndmask_b32_e32 v19, v27, v14, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v8, v7, 16, 1
	v_bfe_u32 v9, v6, 16, 1
	v_bfe_u32 v10, v5, 16, 1
	v_add3_u32 v8, v7, v8, s2
	v_cndmask_b32_e32 v20, v27, v15, vcc
	v_add3_u32 v10, v5, v10, s2
	v_add3_u32 v9, v6, v9, s2
	v_lshrrev_b32_e32 v8, 16, v8
	v_cmp_o_f32_e32 vcc, v7, v7
	v_lshrrev_b32_e32 v9, 16, v9
	v_lshrrev_b32_e32 v13, 16, v10
	v_cndmask_b32_e32 v10, v27, v8, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_bfe_u32 v11, v4, 16, 1
	v_add3_u32 v11, v4, v11, s2
	v_cndmask_b32_e32 v12, v27, v9, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_bfe_u32 v5, v2, 16, 1
	v_bfe_u32 v6, v1, 16, 1
	v_cndmask_b32_e32 v13, v27, v13, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v4, v3, 16, 1
	v_bfe_u32 v7, v0, 16, 1
	v_add3_u32 v7, v0, v7, s2
	v_add3_u32 v6, v1, v6, s2
	v_add3_u32 v5, v2, v5, s2
	v_add3_u32 v4, v3, v4, s2
	v_readlane_b32 s2, v144, 14
	v_readlane_b32 s4, v144, 12
	s_mul_i32 s2, s2, s26
	s_mul_hi_u32 s3, s4, s26
	v_readlane_b32 s8, v144, 10
	s_add_i32 s3, s3, s2
	s_mul_i32 s2, s4, s26
	s_lshl_b32 s4, s26, 5
	v_readlane_b32 s9, v144, 11
	s_mul_i32 s5, s4, s9
	s_mul_hi_u32 s6, s4, s8
	s_add_i32 s6, s6, s5
	s_mul_i32 s4, s4, s8
	s_sub_u32 s2, s2, s4
	s_subb_u32 s3, s3, s6
	s_mul_i32 s3, s3, s26
	s_mul_hi_u32 s4, s2, s26
	s_add_i32 s3, s4, s3
	s_mul_i32 s6, s26, s26
	v_readlane_b32 s4, v144, 2
	s_mul_i32 s2, s2, s26
	s_mul_hi_u32 s5, s6, s4
	s_mul_i32 s4, s6, s4
	s_add_u32 s4, s2, s4
	s_addc_u32 s5, s3, s5
	s_ashr_i32 s2, s5, 31
	s_mov_b32 s3, s2
	s_mul_i32 s8, s6, s26
	s_xor_b64 s[6:7], s[2:3], s[4:5]
	s_add_u32 s4, s8, 0
	s_addc_u32 s5, 0, 0
	s_xor_b64 s[4:5], s[4:5], 0
	v_cvt_f32_u32_e32 v9, s4
	v_cvt_f32_u32_e32 v14, s5
	v_lshrrev_b32_e32 v11, 16, v11
	v_cndmask_b32_e32 v15, v27, v11, vcc
	v_lshrrev_b32_e32 v4, 16, v4
	v_cmp_o_f32_e32 vcc, v3, v3
	v_lshrrev_b32_e32 v11, 16, v7
	v_lshrrev_b32_e32 v5, 16, v5
	v_cndmask_b32_e32 v7, v27, v4, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	v_fmamk_f32 v2, v14, 0x4f800000, v9
	v_rcp_f32_e32 v2, v2
	v_lshrrev_b32_e32 v6, 16, v6
	v_cndmask_b32_e32 v8, v27, v5, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	s_sub_u32 s8, 0, s4
	s_subb_u32 s9, 0, s5
	v_cndmask_b32_e32 v9, v27, v6, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	v_mul_f32_e32 v0, 0x5f7ffffc, v2
	v_mul_f32_e32 v1, 0x2f800000, v0
	v_trunc_f32_e32 v1, v1
	v_fmamk_f32 v0, v1, 0xcf800000, v0
	v_cvt_u32_f32_e32 v0, v0
	v_cvt_u32_f32_e32 v1, v1
	v_cndmask_b32_e32 v11, v27, v11, vcc
	v_mov_b32_e32 v3, 0x27000
	v_readfirstlane_b32 s10, v0
	v_readfirstlane_b32 s13, v1
	s_mul_hi_u32 s12, s8, s10
	s_mul_i32 s14, s8, s13
	s_mul_i32 s11, s9, s10
	s_add_i32 s12, s12, s14
	s_add_i32 s12, s12, s11
	s_mul_i32 s15, s8, s10
	s_mul_i32 s14, s10, s12
	s_mul_hi_u32 s18, s10, s15
	s_mul_hi_u32 s11, s10, s12
	s_add_u32 s14, s18, s14
	s_addc_u32 s11, 0, s11
	s_mul_hi_u32 s19, s13, s15
	s_mul_i32 s15, s13, s15
	s_add_u32 s14, s14, s15
	s_mul_hi_u32 s18, s13, s12
	s_addc_u32 s11, s11, s19
	s_addc_u32 s14, s18, 0
	s_mul_i32 s12, s13, s12
	s_add_u32 s11, s11, s12
	s_addc_u32 s12, 0, s14
	s_add_u32 s10, s10, s11
	s_addc_u32 s11, s13, s12
	s_mul_i32 s12, s8, s11
	s_mul_hi_u32 s13, s8, s10
	s_add_i32 s12, s13, s12
	s_mul_i32 s9, s9, s10
	s_add_i32 s12, s12, s9
	s_mul_i32 s8, s8, s10
	s_mul_hi_u32 s13, s11, s8
	s_mul_i32 s14, s11, s8
	s_mul_i32 s18, s10, s12
	s_mul_hi_u32 s8, s10, s8
	s_mul_hi_u32 s15, s10, s12
	s_add_u32 s8, s8, s18
	s_addc_u32 s15, 0, s15
	s_add_u32 s8, s8, s14
	s_mul_hi_u32 s9, s11, s12
	s_addc_u32 s8, s15, s13
	s_addc_u32 s9, s9, 0
	s_mul_i32 s12, s11, s12
	s_add_u32 s8, s8, s12
	s_addc_u32 s9, 0, s9
	s_add_u32 s10, s10, s8
	s_addc_u32 s11, s11, s9
	s_ashr_i32 s8, s7, 31
	s_add_u32 s6, s6, s8
	s_mov_b32 s9, s8
	s_addc_u32 s7, s7, s8
	s_xor_b64 s[6:7], s[6:7], s[8:9]
	s_mul_i32 s13, s6, s11
	s_mul_hi_u32 s14, s6, s10
	s_mul_hi_u32 s12, s6, s11
	s_add_u32 s13, s14, s13
	s_addc_u32 s12, 0, s12
	s_mul_hi_u32 s15, s7, s10
	s_mul_i32 s10, s7, s10
	s_add_u32 s10, s13, s10
	s_mul_hi_u32 s14, s7, s11
	s_addc_u32 s10, s12, s15
	s_addc_u32 s12, s14, 0
	s_mul_i32 s11, s7, s11
	s_add_u32 s13, s10, s11
	s_addc_u32 s12, 0, s12
	s_mul_i32 s10, s4, s12
	s_mul_hi_u32 s11, s4, s13
	s_add_i32 s10, s11, s10
	s_mul_i32 s11, s5, s13
	s_add_i32 s14, s10, s11
	s_sub_i32 s15, s7, s14
	s_mul_i32 s10, s4, s13
	s_sub_u32 s6, s6, s10
	s_cselect_b64 s[10:11], -1, 0
	s_subb_u32 s15, s15, s5
	s_sub_u32 s18, s6, s4
	s_subb_u32 s15, s15, 0
	s_cmp_ge_u32 s15, s5
	s_cselect_b32 s19, -1, 0
	s_cmp_ge_u32 s18, s4
	s_cselect_b32 s18, -1, 0
	s_cmp_eq_u32 s15, s5
	s_cselect_b32 s15, s18, s19
	s_add_u32 s18, s13, 1
	s_addc_u32 s19, s12, 0
	s_add_u32 s20, s13, 2
	s_addc_u32 s21, s12, 0
	s_cmp_lg_u32 s15, 0
	s_cselect_b32 s15, s20, s18
	s_cselect_b32 s18, s21, s19
	s_cmp_lg_u64 s[10:11], 0
	s_subb_u32 s7, s7, s14
	s_cmp_ge_u32 s7, s5
	s_cselect_b32 s10, -1, 0
	s_cmp_ge_u32 s6, s4
	s_cselect_b32 s4, -1, 0
	s_cmp_eq_u32 s7, s5
	s_cselect_b32 s4, s4, s10
	s_cmp_lg_u32 s4, 0
	s_cselect_b32 s5, s18, s12
	s_cselect_b32 s4, s15, s13
	s_xor_b64 s[6:7], s[8:9], 0
	s_xor_b64 s[4:5], s[4:5], s[6:7]
	s_sub_u32 s4, s4, s6
	s_subb_u32 s5, s5, s7
	s_xor_b64 s[2:3], s[4:5], s[2:3]
	s_and_b64 s[4:5], s[66:67], exec
	s_cselect_b32 s5, s3, s71
	s_cselect_b32 s4, s2, s70
	v_lshlrev_b32_e32 v0, 2, v63
	s_lshl_b64 s[2:3], s[4:5], 6
	v_or_b32_e32 v4, s2, v0
	s_add_i32 s2, s26, s0
	v_sub_co_u32_e32 v2, vcc, s4, v25
	s_ashr_i32 s4, s2, 31
	s_cmp_lt_i32 s1, 0
	v_mov_b32_e32 v1, s5
	s_cselect_b32 s1, s4, s1
	s_cselect_b32 s0, s2, s0
	v_readlane_b32 s4, v144, 16
	v_readlane_b32 s5, v144, 17
	s_add_u32 s2, s0, s4
	s_addc_u32 s4, s1, s5
	v_readlane_b32 s0, v144, 25
	v_readlane_b32 s1, v144, 26
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s2, s2, 0
	v_lshl_or_b32 v14, v25, 6, v0
	v_and_b32_e32 v0, 15, v24
	v_subb_co_u32_e32 v1, vcc, v1, v50, vcc
	v_lshl_or_b32 v0, s2, 6, v0
	v_or_b32_e32 v6, v0, v16
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v0, s16, v1
	v_mul_hi_u32 v1, s16, v2
	s_cselect_b32 s1, s4, 0
	v_add_u32_e32 v0, v1, v0
	v_mul_lo_u32 v1, s17, v2
	v_mad_u64_u32 v[24:25], s[4:5], v14, s16, v[6:7]
	s_sub_u32 s0, s44, s2
	v_add_u32_e32 v1, v0, v1
	v_mul_lo_u32 v0, s16, v2
	v_readlane_b32 s4, v144, 3
	v_mov_b32_e32 v5, s3
	s_subb_u32 s1, s45, s1
	v_lshlrev_b64 v[0:1], 7, v[0:1]
	v_readlane_b32 s5, v144, 4
	s_lshl_b64 s[0:1], s[0:1], 7
	v_not_b32_e32 v16, -2.0
	v_lshl_add_u64 v[0:1], s[4:5], 0, v[0:1]
	v_cmp_gt_i64_e32 vcc, s[60:61], v[4:5]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[0:1]
	v_and_b32_e32 v1, 0xffff, v1
	v_cndmask_b32_e32 v5, v16, v24, vcc
	v_mov_b32_e32 v2, 0x7ffffffd
	v_lshlrev_b32_e32 v5, 1, v5
	s_mov_b64 s[8:9], exec
.LBB0_89:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v26, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_89
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v24, 1, v4
	v_mov_b32_e32 v25, s3
	v_or_b32_e32 v5, 1, v14
	v_mad_u64_u32 v[26:27], s[0:1], v5, s16, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[60:61], v[24:25]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v26, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_91:
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
	s_cbranch_execnz .LBB0_91
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v24, 2, v4
	v_mov_b32_e32 v25, s3
	v_or_b32_e32 v5, 2, v14
	v_mad_u64_u32 v[26:27], s[0:1], v5, s16, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[60:61], v[24:25]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v26, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_93:
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
	s_cbranch_execnz .LBB0_93
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v22, 3, v4
	v_mov_b32_e32 v23, s3
	v_or_b32_e32 v5, 3, v14
	v_mad_u64_u32 v[24:25], s[0:1], v5, s16, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[60:61], v[22:23]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v24, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_95:
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
	s_cbranch_execnz .LBB0_95
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v22, 16, v4
	v_mov_b32_e32 v23, s3
	v_or_b32_e32 v5, 16, v14
	v_mad_u64_u32 v[24:25], s[0:1], v5, s16, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[60:61], v[22:23]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v24, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_97:
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
	s_cbranch_execnz .LBB0_97
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v20, 17, v4
	v_mov_b32_e32 v21, s3
	v_or_b32_e32 v5, 17, v14
	v_mad_u64_u32 v[22:23], s[0:1], v5, s16, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[60:61], v[20:21]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v22, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_99:
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
	s_cbranch_execnz .LBB0_99
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v20, 18, v4
	v_mov_b32_e32 v21, s3
	v_or_b32_e32 v5, 18, v14
	v_mad_u64_u32 v[22:23], s[0:1], v5, s16, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[60:61], v[20:21]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v22, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_101:
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
	s_cbranch_execnz .LBB0_101
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v18, 19, v4
	v_mov_b32_e32 v19, s3
	v_or_b32_e32 v5, 19, v14
	v_mad_u64_u32 v[20:21], s[0:1], v5, s16, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[60:61], v[18:19]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v20, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_103:
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
	s_cbranch_execnz .LBB0_103
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v16, 32, v4
	v_mov_b32_e32 v17, s3
	v_or_b32_e32 v5, 32, v14
	v_mad_u64_u32 v[18:19], s[0:1], v5, s16, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[60:61], v[16:17]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v18, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_105:
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
	s_cbranch_execnz .LBB0_105
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v16, 33, v4
	v_mov_b32_e32 v17, s3
	v_or_b32_e32 v5, 33, v14
	v_mad_u64_u32 v[18:19], s[0:1], v5, s16, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[60:61], v[16:17]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v18, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_107:
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
	s_cbranch_execnz .LBB0_107
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v16, 34, v4
	v_mov_b32_e32 v17, s3
	v_or_b32_e32 v5, 34, v14
	v_mad_u64_u32 v[18:19], s[0:1], v5, s16, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[60:61], v[16:17]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v18, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_109:
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
	s_cbranch_execnz .LBB0_109
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v12, 35, v4
	v_mov_b32_e32 v13, s3
	v_or_b32_e32 v5, 35, v14
	v_mad_u64_u32 v[16:17], s[0:1], v5, s16, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[60:61], v[12:13]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v16, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_111:
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
	s_cbranch_execnz .LBB0_111
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v12, 48, v4
	v_mov_b32_e32 v13, s3
	v_or_b32_e32 v5, 48, v14
	v_mad_u64_u32 v[16:17], s[0:1], v5, s16, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[60:61], v[12:13]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v16, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_113:
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
	s_cbranch_execnz .LBB0_113
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v10, 49, v4
	v_mov_b32_e32 v11, s3
	v_or_b32_e32 v5, 49, v14
	v_mad_u64_u32 v[12:13], s[0:1], v5, s16, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[60:61], v[10:11]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v12, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_115:
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
	s_cbranch_execnz .LBB0_115
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v10, 50, v4
	v_mov_b32_e32 v11, s3
	v_or_b32_e32 v5, 50, v14
	v_mad_u64_u32 v[12:13], s[0:1], v5, s16, v[6:7]
	v_not_b32_e32 v5, -2.0
	v_cmp_gt_i64_e32 vcc, s[60:61], v[10:11]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v12, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_117:
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
	s_cbranch_execnz .LBB0_117
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v4, 51, v4
	v_mov_b32_e32 v5, s3
	v_or_b32_e32 v8, 51, v14
	v_mad_u64_u32 v[8:9], s[0:1], v8, s16, v[6:7]
	v_not_b32_e32 v6, -2.0
	v_cmp_gt_i64_e32 vcc, s[60:61], v[4:5]
	s_mov_b64 s[0:1], exec
	s_nop 0
	v_cndmask_b32_e32 v4, v6, v8, vcc
	v_lshlrev_b32_e32 v4, 1, v4
.LBB0_119:
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
	s_cbranch_execnz .LBB0_119
	s_endpgm
.LBB0_121:
	s_branch .LBB0_87
.LBB0_122:
	s_lshl_b64 s[62:63], s[4:5], 1
	v_mov_b32_e32 v69, 0
.LBB0_123:
	v_mad_u64_u32 v[70:71], s[0:1], s12, v31, 0
	v_mul_i32_i24_e32 v0, 0xffffffc0, v63
	v_add3_u32 v71, v71, v72, v68
	s_add_u32 s76, s62, -2
	v_or_b32_e32 v68, v8, v2
	v_or_b32_e32 v2, 0x800, v3
	v_mov_b32_e32 v3, v69
	v_lshl_add_u64 v[8:9], v[66:67], 0, v[64:65]
	v_ashrrev_i32_e32 v1, 31, v0
	s_addc_u32 s77, s63, -1
	v_mov_b32_e32 v25, v69
	v_lshl_add_u64 v[72:73], v[8:9], 0, v[2:3]
	v_lshl_add_u64 v[74:75], v[68:69], 0, v[0:1]
	v_or_b32_e32 v70, v70, v46
	s_mov_b64 s[78:79], 0
	s_mov_b64 s[80:81], 0x800
	s_mov_b32 s41, 0
	s_mov_b32 s82, s64
	s_mov_b32 s83, s64
	s_mov_b64 s[84:85], 0xc00
	v_bfrev_b32_e32 v31, -2
	v_mov_b32_e32 v12, v69
	v_mov_b32_e32 v13, v69
	v_mov_b32_e32 v14, v69
	v_mov_b32_e32 v15, v69
	v_mov_b32_e32 v0, v69
	v_mov_b32_e32 v1, v69
	v_mov_b32_e32 v2, v69
	v_mov_b32_e32 v8, v69
	v_mov_b32_e32 v9, v69
	v_mov_b32_e32 v10, v69
	v_mov_b32_e32 v11, v69
	v_mov_b32_e32 v20, v69
	v_mov_b32_e32 v21, v69
	v_mov_b32_e32 v22, v69
	v_mov_b32_e32 v23, v69
	s_mov_b64 s[86:87], 0x100
	s_mov_b64 s[88:89], 0x102
	s_mov_b64 s[90:91], 0x200
	s_mov_b64 s[56:57], 0xc0b
	s_mov_b64 s[20:21], 0xc0c
	s_mov_b64 s[58:59], 0xc0d
	s_mov_b64 s[60:61], 0xc0e
	s_mov_b64 s[48:49], 0xc0f
	s_mov_b64 s[50:51], 0x202
	s_mov_b64 s[0:1], 0x300
	s_mov_b64 s[16:17], 0x6000
	s_branch .LBB0_125
.LBB0_124:
	s_or_b64 exec, exec, s[4:5]
	v_ashrrev_i32_e32 v4, 31, v19
	v_and_b32_e32 v5, s35, v4
	v_and_b32_e32 v4, s34, v4
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[18:19]
	v_ashrrev_i32_e32 v7, 31, v5
	v_xor_b32_e32 v5, v7, v5
	v_xor_b32_e32 v6, v6, v16
	v_ashrrev_i32_e32 v16, 31, v5
	v_xor_b32_e32 v4, v7, v4
	v_lshrrev_b32_e32 v68, 27, v16
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[68:69]
	v_readlane_b32 s4, v144, 18
	v_alignbit_b32 v4, v5, v4, 5
	s_mov_b32 m0, s40
	v_mul_lo_u32 v5, v6, s4
	v_xad_u32 v4, v4, v7, v5
	buffer_load_dword v4, s[92:95], 0 offen lds
	v_readlane_b32 s5, v144, 19
	s_waitcnt vmcnt(3) lgkmcnt(0)
	s_barrier
	s_add_u32 s78, s78, 2
	s_addc_u32 s79, s79, 0
	v_mov_b64_e32 v[4:5], s[76:77]
	v_cmp_lt_i64_e32 vcc, s[78:79], v[4:5]
	v_lshlrev_b16_e32 v4, 8, v37
	v_lshlrev_b16_e32 v5, 8, v51
	v_or_b32_e32 v4, v33, v4
	v_or_b32_sdwa v5, v39, v5 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v6, 8, v113
	v_or_b32_sdwa v16, v4, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v4, 8, v55
	v_lshlrev_b16_e32 v5, 8, v61
	v_or_b32_e32 v4, v53, v4
	v_or_b32_sdwa v5, v59, v5 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v6, v112, v6 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v17, v4, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v4, 8, v98
	v_lshlrev_b16_e32 v5, 8, v100
	v_or_b32_e32 v4, v83, v4
	v_or_b32_sdwa v5, v99, v5 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v7, 8, v117
	v_or_b32_sdwa v18, v4, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v4, 8, v102
	v_lshlrev_b16_e32 v5, 8, v104
	v_or_b32_e32 v4, v101, v4
	v_or_b32_sdwa v5, v103, v5 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v7, v116, v7 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v19, v4, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v4, 8, v107
	v_lshlrev_b16_e32 v5, 8, v109
	v_or_b32_e32 v4, v106, v4
	v_or_b32_sdwa v5, v108, v5 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v33, 8, v57
	v_or_b32_sdwa v4, v4, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v5, 8, v111
	v_or_b32_e32 v5, v110, v5
	v_or_b32_sdwa v5, v5, v6 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v6, 8, v115
	v_or_b32_e32 v6, v114, v6
	v_or_b32_sdwa v6, v6, v7 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v7, 8, v119
	v_or_b32_e32 v7, v118, v7
	v_or_b32_sdwa v33, v120, v33 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_nop 0
	v_or_b32_sdwa v7, v7, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_cbranch_vccz .LBB0_41
.LBB0_125:
	s_waitcnt vmcnt(3) lgkmcnt(0)
	s_barrier
	s_lshl_b64 s[4:5], s[78:79], 11
	v_or_b32_e32 v76, s4, v44
	s_add_u32 s4, s42, s64
	v_mov_b32_e32 v77, s5
	s_addc_u32 s5, s43, s64
	s_xor_b64 s[12:13], s[4:5], s[82:83]
	v_cvt_f32_u32_e32 v35, s12
	v_cvt_f32_u32_e32 v37, s13
	v_lshl_add_u64 v[88:89], v[76:77], 0, v[66:67]
	s_movk_i32 s4, 0xf7ff
	v_sub_co_u32_e32 v39, vcc, s4, v88
	v_fmac_f32_e32 v35, 0x4f800000, v37
	v_rcp_f32_e32 v35, v35
	s_sub_u32 s4, 0, s12
	s_subb_u32 s5, 0, s13
	v_lshl_add_u64 v[80:81], v[88:89], 0, s[80:81]
	v_mul_f32_e32 v35, 0x5f7ffffc, v35
	v_mul_f32_e32 v37, 0x2f800000, v35
	v_trunc_f32_e32 v37, v37
	v_fmac_f32_e32 v35, 0xcf800000, v37
	v_cvt_u32_f32_e32 v35, v35
	v_cvt_u32_f32_e32 v37, v37
	v_subb_co_u32_e32 v51, vcc, -1, v89, vcc
	v_readfirstlane_b32 s6, v35
	v_readfirstlane_b32 s19, v37
	s_mul_hi_u32 s18, s4, s6
	s_mul_i32 s22, s4, s19
	s_mul_i32 s7, s5, s6
	s_add_i32 s18, s18, s22
	s_add_i32 s18, s18, s7
	s_mul_i32 s23, s4, s6
	s_mul_i32 s22, s6, s18
	s_mul_hi_u32 s24, s6, s23
	s_mul_hi_u32 s7, s6, s18
	s_add_u32 s22, s24, s22
	s_addc_u32 s7, 0, s7
	s_mul_hi_u32 s25, s19, s23
	s_mul_i32 s23, s19, s23
	s_add_u32 s22, s22, s23
	s_mul_hi_u32 s24, s19, s18
	s_addc_u32 s7, s7, s25
	s_addc_u32 s22, s24, 0
	s_mul_i32 s18, s19, s18
	s_add_u32 s7, s7, s18
	s_addc_u32 s18, 0, s22
	s_add_u32 s6, s6, s7
	s_addc_u32 s7, s19, s18
	s_mul_i32 s18, s4, s7
	s_mul_hi_u32 s19, s4, s6
	s_add_i32 s18, s19, s18
	s_mul_i32 s5, s5, s6
	s_add_i32 s18, s18, s5
	s_mul_i32 s4, s4, s6
	s_mul_hi_u32 s19, s7, s4
	s_mul_i32 s22, s7, s4
	s_mul_i32 s24, s6, s18
	s_mul_hi_u32 s4, s6, s4
	s_mul_hi_u32 s23, s6, s18
	s_add_u32 s4, s4, s24
	s_addc_u32 s23, 0, s23
	s_add_u32 s4, s4, s22
	v_cmp_gt_i32_e32 vcc, 0, v81
	s_mul_hi_u32 s5, s7, s18
	s_addc_u32 s4, s23, s19
	v_cndmask_b32_e32 v83, v81, v51, vcc
	s_addc_u32 s5, s5, 0
	s_mul_i32 s18, s7, s18
	s_add_u32 s4, s4, s18
	v_ashrrev_i32_e32 v84, 31, v83
	v_cndmask_b32_e32 v82, v80, v39, vcc
	s_addc_u32 s5, 0, s5
	v_mov_b32_e32 v85, v84
	s_add_u32 s4, s6, s4
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	s_addc_u32 s5, s7, s5
	v_xor_b32_e32 v37, v82, v84
	v_xor_b32_e32 v35, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v37, s5, 0
	v_mul_hi_u32 v68, v37, s4
	v_and_b32_e32 v33, 0xff, v90
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[90:91], s[6:7], v35, s4, 0
	v_add_co_u32_e32 v39, vcc, v82, v90
	v_mad_u64_u32 v[86:87], s[6:7], v35, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v91, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[86:87]
	v_mul_lo_u32 v39, s13, v82
	v_mul_lo_u32 v51, s12, v83
	v_mad_u64_u32 v[86:87], s[6:7], s12, v82, 0
	v_add3_u32 v39, v87, v51, v39
	v_sub_co_u32_e32 v37, vcc, v37, v86
	v_mov_b32_e32 v61, s13
	s_nop 0
	v_subb_co_u32_e64 v51, s[18:19], v35, v39, vcc
	v_sub_u32_e32 v35, v35, v39
	v_cmp_le_u32_e64 s[18:19], s13, v51
	v_subb_co_u32_e32 v35, vcc, v35, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v53, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s12, v37
	v_subrev_co_u32_e32 v37, vcc, s12, v37
	s_nop 0
	v_cndmask_b32_e64 v55, 0, -1, s[18:19]
	v_subbrev_co_u32_e32 v35, vcc, 0, v35, vcc
	v_cmp_le_u32_e32 vcc, s13, v35
	v_cmp_eq_u32_e64 s[18:19], s13, v51
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[134:137], v[16:19], v[12:15], v78, v33 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v37
	v_cndmask_b32_e64 v51, v53, v55, s[18:19]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[126:129], v[16:19], v[0:3], v78, v33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e64 v37, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v35
	s_nop 1
	v_cndmask_b32_e32 v35, v39, v37, vcc
	v_cmp_ne_u32_e32 vcc, 0, v35
	v_xor_b32_e32 v37, s64, v84
	s_nop 0
	v_cndmask_b32_e64 v35, 1, 2, vcc
	v_add_u32_e32 v35, v82, v35
	v_cmp_ne_u32_e32 vcc, 0, v51
	s_nop 1
	v_cndmask_b32_e32 v35, v82, v35, vcc
	v_ashrrev_i32_e32 v82, 31, v81
	v_mov_b32_e32 v83, v82
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[82:83]
	v_xor_b32_e32 v35, v35, v37
	v_xor_b32_e32 v39, v80, v82
	v_sub_u32_e32 v35, v35, v37
	v_xor_b32_e32 v37, v81, v82
	v_mad_u64_u32 v[80:81], s[6:7], v39, s5, 0
	v_mul_hi_u32 v68, v39, s4
	v_lshl_add_u64 v[80:81], v[68:69], 0, v[80:81]
	v_mad_u64_u32 v[86:87], s[6:7], v37, s4, 0
	v_add_co_u32_e32 v51, vcc, v80, v86
	v_mad_u64_u32 v[84:85], s[6:7], v37, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v80, vcc, v81, v87, vcc
	v_mov_b32_e32 v81, s41
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[84:85]
	v_mul_lo_u32 v51, s13, v80
	v_mul_lo_u32 v53, s12, v81
	v_mad_u64_u32 v[80:81], s[6:7], s12, v80, 0
	v_add3_u32 v51, v81, v53, v51
	v_sub_u32_e32 v53, v37, v51
	v_sub_co_u32_e32 v39, vcc, v39, v80
	v_xad_u32 v35, v35, v82, v30
	s_nop 0
	v_subb_co_u32_e64 v53, s[18:19], v53, v61, vcc
	v_subrev_co_u32_e64 v55, s[18:19], s12, v39
	v_subb_co_u32_e32 v37, vcc, v37, v51, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v57, s[22:23], 0, v53, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s13, v57
	v_subb_co_u32_e64 v53, s[18:19], v53, v61, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v55
	v_cmp_le_u32_e32 vcc, s13, v37
	v_mul_lo_u32 v35, v35, s33
	v_cndmask_b32_e64 v68, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s13, v57
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v39
	v_cndmask_b32_e64 v59, v59, v68, s[22:23]
	v_subrev_co_u32_e64 v68, s[18:19], s12, v55
	v_lshl_add_u64 v[86:87], v[76:77], 0, v[72:73]
	s_nop 0
	v_subbrev_co_u32_e64 v53, s[18:19], 0, v53, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v59
	v_or_b32_e32 v80, 1, v86
	v_mov_b32_e32 v81, v87
	v_cndmask_b32_e64 v53, v57, v53, s[18:19]
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v37
	s_nop 1
	v_cndmask_b32_e32 v51, v51, v57, vcc
	v_cmp_ne_u32_e32 vcc, 0, v51
	v_cndmask_b32_e64 v51, v55, v68, s[18:19]
	v_cmp_gt_i32_e64 s[18:19], 0, v87
	v_cndmask_b32_e32 v39, v39, v51, vcc
	v_cndmask_b32_e32 v37, v37, v53, vcc
	v_xor_b32_e32 v39, v39, v82
	v_xor_b32_e32 v37, v37, v82
	v_sub_co_u32_e32 v39, vcc, v39, v82
	s_nop 1
	v_subb_co_u32_e32 v37, vcc, v37, v82, vcc
	v_ashrrev_i32_e32 v37, 31, v37
	v_and_b32_e32 v37, s42, v37
	v_add3_u32 v35, v37, v39, v35
	v_sub_co_u32_e32 v37, vcc, -2, v86
	v_cndmask_b32_e64 v76, v80, v37, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v39, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v77, v87, v39, s[18:19]
	v_ashrrev_i32_e32 v82, 31, v77
	v_mov_b32_e32 v83, v82
	v_lshl_add_u64 v[76:77], v[76:77], 0, v[82:83]
	v_xor_b32_e32 v39, v76, v82
	v_xor_b32_e32 v37, v77, v82
	v_mad_u64_u32 v[76:77], s[6:7], v39, s5, 0
	v_mul_hi_u32 v68, v39, s4
	v_lshl_add_u64 v[76:77], v[68:69], 0, v[76:77]
	v_mad_u64_u32 v[90:91], s[6:7], v37, s4, 0
	v_add_co_u32_e32 v51, vcc, v76, v90
	v_mad_u64_u32 v[84:85], s[6:7], v37, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v76, vcc, v77, v91, vcc
	v_mov_b32_e32 v77, s41
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[76:77], v[76:77], 0, v[84:85]
	v_mul_lo_u32 v51, s13, v76
	v_mul_lo_u32 v53, s12, v77
	v_mad_u64_u32 v[84:85], s[6:7], s12, v76, 0
	v_add3_u32 v51, v85, v53, v51
	v_sub_co_u32_e32 v39, vcc, v39, v84
	s_nop 1
	v_subb_co_u32_e64 v53, s[22:23], v37, v51, vcc
	v_sub_u32_e32 v37, v37, v51
	v_cmp_le_u32_e64 s[22:23], s13, v53
	v_subb_co_u32_e32 v37, vcc, v37, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v55, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v39
	v_subrev_co_u32_e32 v39, vcc, s12, v39
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v37, vcc, 0, v37, vcc
	v_cmp_le_u32_e32 vcc, s13, v37
	v_cmp_eq_u32_e64 s[22:23], s13, v53
	s_nop 0
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v39
	v_cndmask_b32_e64 v53, v55, v57, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v37
	s_nop 1
	v_cndmask_b32_e32 v37, v51, v39, vcc
	v_cmp_ne_u32_e32 vcc, 0, v37
	v_xor_b32_e32 v39, s64, v82
	s_nop 0
	v_cndmask_b32_e64 v37, 1, 2, vcc
	v_add_u32_e32 v37, v76, v37
	v_cmp_ne_u32_e32 vcc, 0, v53
	s_nop 1
	v_cndmask_b32_e32 v37, v76, v37, vcc
	v_ashrrev_i32_e32 v76, 31, v87
	v_mov_b32_e32 v77, v76
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[76:77]
	v_xor_b32_e32 v37, v37, v39
	v_xor_b32_e32 v51, v80, v76
	v_sub_u32_e32 v37, v37, v39
	v_xor_b32_e32 v39, v81, v76
	v_mad_u64_u32 v[80:81], s[6:7], v51, s5, 0
	v_mul_hi_u32 v68, v51, s4
	v_lshl_add_u64 v[80:81], v[68:69], 0, v[80:81]
	v_mad_u64_u32 v[84:85], s[6:7], v39, s4, 0
	v_add_co_u32_e32 v53, vcc, v80, v84
	v_mad_u64_u32 v[82:83], s[6:7], v39, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v80, vcc, v81, v85, vcc
	v_mov_b32_e32 v81, s41
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[82:83]
	v_mul_lo_u32 v53, s13, v80
	v_mul_lo_u32 v55, s12, v81
	v_mad_u64_u32 v[80:81], s[6:7], s12, v80, 0
	v_add3_u32 v53, v81, v55, v53
	v_sub_u32_e32 v55, v39, v53
	v_sub_co_u32_e32 v51, vcc, v51, v80
	v_xad_u32 v37, v37, v76, v30
	s_nop 0
	v_subb_co_u32_e64 v55, s[22:23], v55, v61, vcc
	v_subrev_co_u32_e64 v57, s[22:23], s12, v51
	v_subb_co_u32_e32 v39, vcc, v39, v53, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v59, s[24:25], 0, v55, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v59
	v_subb_co_u32_e64 v55, s[22:23], v55, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v57
	v_cmp_le_u32_e32 vcc, s13, v39
	v_mul_lo_u32 v37, v37, s33
	v_cndmask_b32_e64 v80, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v59
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v51
	v_cndmask_b32_e64 v68, v68, v80, s[24:25]
	v_subrev_co_u32_e64 v80, s[22:23], s12, v57
	v_mov_b32_e32 v81, v87
	s_nop 0
	v_subbrev_co_u32_e64 v55, s[22:23], 0, v55, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v68
	s_nop 1
	v_cndmask_b32_e64 v55, v59, v55, s[22:23]
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v39
	s_nop 1
	v_cndmask_b32_e32 v53, v53, v59, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_cndmask_b32_e64 v53, v57, v80, s[22:23]
	v_or_b32_e32 v80, 2, v86
	v_cndmask_b32_e32 v51, v51, v53, vcc
	v_cndmask_b32_e32 v39, v39, v55, vcc
	v_xor_b32_e32 v51, v51, v76
	v_xor_b32_e32 v39, v39, v76
	v_sub_co_u32_e32 v51, vcc, v51, v76
	s_nop 1
	v_subb_co_u32_e32 v39, vcc, v39, v76, vcc
	v_ashrrev_i32_e32 v39, 31, v39
	v_and_b32_e32 v39, s42, v39
	v_add3_u32 v37, v39, v51, v37
	v_sub_co_u32_e32 v39, vcc, -3, v86
	v_cndmask_b32_e64 v82, v80, v39, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v51, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v83, v87, v51, s[18:19]
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v51, v82, v84
	v_xor_b32_e32 v39, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v51, s5, 0
	v_mul_hi_u32 v68, v51, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[92:93], s[6:7], v39, s4, 0
	v_add_co_u32_e32 v53, vcc, v82, v92
	v_mad_u64_u32 v[90:91], s[6:7], v39, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v93, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v53, s13, v82
	v_mul_lo_u32 v55, s12, v83
	v_mad_u64_u32 v[90:91], s[6:7], s12, v82, 0
	v_add3_u32 v53, v91, v55, v53
	v_sub_co_u32_e32 v51, vcc, v51, v90
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[76:77]
	s_nop 0
	v_subb_co_u32_e64 v55, s[22:23], v39, v53, vcc
	v_sub_u32_e32 v39, v39, v53
	v_cmp_le_u32_e64 s[22:23], s13, v55
	v_subb_co_u32_e32 v39, vcc, v39, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v51
	v_subrev_co_u32_e32 v51, vcc, s12, v51
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v39, vcc, 0, v39, vcc
	v_cmp_le_u32_e32 vcc, s13, v39
	v_cmp_eq_u32_e64 s[22:23], s13, v55
	s_nop 0
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v51
	v_cndmask_b32_e64 v55, v57, v59, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v39
	s_nop 1
	v_cndmask_b32_e32 v39, v53, v51, vcc
	v_cmp_ne_u32_e32 vcc, 0, v39
	v_xor_b32_e32 v51, s64, v84
	v_xor_b32_e32 v53, v80, v76
	v_cndmask_b32_e64 v39, 1, 2, vcc
	v_add_u32_e32 v39, v82, v39
	v_cmp_ne_u32_e32 vcc, 0, v55
	v_mul_hi_u32 v68, v53, s4
	s_nop 0
	v_cndmask_b32_e32 v39, v82, v39, vcc
	v_xor_b32_e32 v39, v39, v51
	v_sub_u32_e32 v39, v39, v51
	v_xor_b32_e32 v51, v81, v76
	v_mad_u64_u32 v[80:81], s[6:7], v53, s5, 0
	v_lshl_add_u64 v[80:81], v[68:69], 0, v[80:81]
	v_mad_u64_u32 v[84:85], s[6:7], v51, s4, 0
	v_add_co_u32_e32 v55, vcc, v80, v84
	v_mad_u64_u32 v[82:83], s[6:7], v51, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v80, vcc, v81, v85, vcc
	v_mov_b32_e32 v81, s41
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[82:83]
	v_mul_lo_u32 v55, s13, v80
	v_mul_lo_u32 v57, s12, v81
	v_mad_u64_u32 v[80:81], s[6:7], s12, v80, 0
	v_add3_u32 v55, v81, v57, v55
	v_sub_u32_e32 v57, v51, v55
	v_sub_co_u32_e32 v53, vcc, v53, v80
	v_xad_u32 v39, v39, v76, v30
	s_nop 0
	v_subb_co_u32_e64 v57, s[22:23], v57, v61, vcc
	v_subrev_co_u32_e64 v59, s[22:23], s12, v53
	v_subb_co_u32_e32 v51, vcc, v51, v55, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v68, s[24:25], 0, v57, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v68
	v_subb_co_u32_e64 v57, s[22:23], v57, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v80, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v59
	v_cmp_le_u32_e32 vcc, s13, v51
	v_mul_lo_u32 v39, v39, s33
	v_cndmask_b32_e64 v81, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v68
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v53
	v_cndmask_b32_e64 v80, v80, v81, s[24:25]
	v_subrev_co_u32_e64 v81, s[22:23], s12, v59
	s_nop 1
	v_subbrev_co_u32_e64 v57, s[22:23], 0, v57, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v80
	v_or_b32_e32 v80, 3, v86
	s_nop 0
	v_cndmask_b32_e64 v57, v68, v57, s[22:23]
	v_cndmask_b32_e64 v68, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v51
	s_nop 1
	v_cndmask_b32_e32 v55, v55, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v55
	v_cndmask_b32_e64 v55, v59, v81, s[22:23]
	v_mov_b32_e32 v81, v87
	v_cndmask_b32_e32 v53, v53, v55, vcc
	v_cndmask_b32_e32 v51, v51, v57, vcc
	v_xor_b32_e32 v53, v53, v76
	v_xor_b32_e32 v51, v51, v76
	v_sub_co_u32_e32 v53, vcc, v53, v76
	s_nop 1
	v_subb_co_u32_e32 v51, vcc, v51, v76, vcc
	v_ashrrev_i32_e32 v51, 31, v51
	v_and_b32_e32 v51, s42, v51
	v_add3_u32 v51, v51, v53, v39
	v_sub_co_u32_e32 v39, vcc, -4, v86
	v_cndmask_b32_e64 v82, v80, v39, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v53, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v83, v87, v53, s[18:19]
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v53, v82, v84
	v_xor_b32_e32 v39, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v53, s5, 0
	v_mul_hi_u32 v68, v53, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[92:93], s[6:7], v39, s4, 0
	v_add_co_u32_e32 v55, vcc, v82, v92
	v_mad_u64_u32 v[90:91], s[6:7], v39, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v93, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v55, s13, v82
	v_mul_lo_u32 v57, s12, v83
	v_mad_u64_u32 v[90:91], s[6:7], s12, v82, 0
	v_add3_u32 v55, v91, v57, v55
	v_sub_co_u32_e32 v53, vcc, v53, v90
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[76:77]
	s_nop 0
	v_subb_co_u32_e64 v57, s[22:23], v39, v55, vcc
	v_sub_u32_e32 v39, v39, v55
	v_cmp_le_u32_e64 s[22:23], s13, v57
	v_subb_co_u32_e32 v39, vcc, v39, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v53
	v_subrev_co_u32_e32 v53, vcc, s12, v53
	s_nop 0
	v_cndmask_b32_e64 v68, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v39, vcc, 0, v39, vcc
	v_cmp_le_u32_e32 vcc, s13, v39
	v_cmp_eq_u32_e64 s[22:23], s13, v57
	s_nop 0
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v53
	v_cndmask_b32_e64 v57, v59, v68, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v39
	s_nop 1
	v_cndmask_b32_e32 v39, v55, v53, vcc
	v_cmp_ne_u32_e32 vcc, 0, v39
	v_xor_b32_e32 v53, s64, v84
	v_xor_b32_e32 v55, v80, v76
	v_cndmask_b32_e64 v39, 1, 2, vcc
	v_add_u32_e32 v39, v82, v39
	v_cmp_ne_u32_e32 vcc, 0, v57
	v_mul_hi_u32 v68, v55, s4
	s_nop 0
	v_cndmask_b32_e32 v39, v82, v39, vcc
	v_xor_b32_e32 v39, v39, v53
	v_sub_u32_e32 v39, v39, v53
	v_xor_b32_e32 v53, v81, v76
	v_mad_u64_u32 v[80:81], s[6:7], v55, s5, 0
	v_lshl_add_u64 v[80:81], v[68:69], 0, v[80:81]
	v_mad_u64_u32 v[84:85], s[6:7], v53, s4, 0
	v_add_co_u32_e32 v57, vcc, v80, v84
	v_mad_u64_u32 v[82:83], s[6:7], v53, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v80, vcc, v81, v85, vcc
	v_mov_b32_e32 v81, s41
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[82:83]
	v_mul_lo_u32 v57, s13, v80
	v_mul_lo_u32 v59, s12, v81
	v_mad_u64_u32 v[80:81], s[6:7], s12, v80, 0
	v_add3_u32 v57, v81, v59, v57
	v_sub_u32_e32 v59, v53, v57
	v_sub_co_u32_e32 v55, vcc, v55, v80
	v_xad_u32 v39, v39, v76, v30
	s_nop 0
	v_subb_co_u32_e64 v59, s[22:23], v59, v61, vcc
	v_subrev_co_u32_e64 v68, s[22:23], s12, v55
	v_subb_co_u32_e32 v53, vcc, v53, v57, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v80, s[24:25], 0, v59, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v80
	v_subb_co_u32_e64 v59, s[22:23], v59, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v81, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v68
	v_cmp_le_u32_e32 vcc, s13, v53
	v_mul_lo_u32 v39, v39, s33
	v_cndmask_b32_e64 v82, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v80
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v55
	v_cndmask_b32_e64 v81, v81, v82, s[24:25]
	v_subrev_co_u32_e64 v82, s[22:23], s12, v68
	s_nop 1
	v_subbrev_co_u32_e64 v59, s[22:23], 0, v59, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v81
	v_mov_b32_e32 v81, v87
	s_nop 0
	v_cndmask_b32_e64 v59, v80, v59, s[22:23]
	v_cndmask_b32_e64 v80, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v53
	s_nop 1
	v_cndmask_b32_e32 v57, v57, v80, vcc
	v_cmp_ne_u32_e32 vcc, 0, v57
	v_cndmask_b32_e64 v57, v68, v82, s[22:23]
	v_or_b32_e32 v80, 4, v86
	v_cndmask_b32_e32 v55, v55, v57, vcc
	v_cndmask_b32_e32 v53, v53, v59, vcc
	v_xor_b32_e32 v55, v55, v76
	v_xor_b32_e32 v53, v53, v76
	v_sub_co_u32_e32 v55, vcc, v55, v76
	s_nop 1
	v_subb_co_u32_e32 v53, vcc, v53, v76, vcc
	v_ashrrev_i32_e32 v53, 31, v53
	v_and_b32_e32 v53, s42, v53
	v_add3_u32 v53, v53, v55, v39
	v_sub_co_u32_e32 v39, vcc, -5, v86
	v_cndmask_b32_e64 v82, v80, v39, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v55, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v83, v87, v55, s[18:19]
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v55, v82, v84
	v_xor_b32_e32 v39, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v55, s5, 0
	v_mul_hi_u32 v68, v55, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[92:93], s[6:7], v39, s4, 0
	v_add_co_u32_e32 v57, vcc, v82, v92
	v_mad_u64_u32 v[90:91], s[6:7], v39, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v93, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v57, s13, v82
	v_mul_lo_u32 v59, s12, v83
	v_mad_u64_u32 v[90:91], s[6:7], s12, v82, 0
	v_add3_u32 v57, v91, v59, v57
	v_sub_co_u32_e32 v55, vcc, v55, v90
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[76:77]
	s_nop 0
	v_subb_co_u32_e64 v59, s[22:23], v39, v57, vcc
	v_sub_u32_e32 v39, v39, v57
	v_cmp_le_u32_e64 s[22:23], s13, v59
	v_subb_co_u32_e32 v39, vcc, v39, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v68, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v55
	v_subrev_co_u32_e32 v55, vcc, s12, v55
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v39, vcc, 0, v39, vcc
	v_cmp_le_u32_e32 vcc, s13, v39
	v_cmp_eq_u32_e64 s[22:23], s13, v59
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v55
	v_cndmask_b32_e64 v59, v68, v83, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v39
	s_nop 1
	v_cndmask_b32_e32 v39, v57, v55, vcc
	v_cmp_ne_u32_e32 vcc, 0, v39
	v_xor_b32_e32 v55, s64, v84
	v_xor_b32_e32 v57, v80, v76
	v_cndmask_b32_e64 v39, 1, 2, vcc
	v_add_u32_e32 v39, v82, v39
	v_cmp_ne_u32_e32 vcc, 0, v59
	v_mul_hi_u32 v68, v57, s4
	s_nop 0
	v_cndmask_b32_e32 v39, v82, v39, vcc
	v_xor_b32_e32 v39, v39, v55
	v_sub_u32_e32 v39, v39, v55
	v_xor_b32_e32 v55, v81, v76
	v_mad_u64_u32 v[80:81], s[6:7], v57, s5, 0
	v_lshl_add_u64 v[80:81], v[68:69], 0, v[80:81]
	v_mad_u64_u32 v[84:85], s[6:7], v55, s4, 0
	v_add_co_u32_e32 v59, vcc, v80, v84
	v_mad_u64_u32 v[82:83], s[6:7], v55, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v80, vcc, v81, v85, vcc
	v_mov_b32_e32 v81, s41
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[82:83]
	v_mul_lo_u32 v59, s13, v80
	v_mul_lo_u32 v68, s12, v81
	v_mad_u64_u32 v[80:81], s[6:7], s12, v80, 0
	v_add3_u32 v59, v81, v68, v59
	v_sub_u32_e32 v68, v55, v59
	v_sub_co_u32_e32 v57, vcc, v57, v80
	v_xad_u32 v39, v39, v76, v30
	s_nop 0
	v_subb_co_u32_e64 v68, s[22:23], v68, v61, vcc
	v_subrev_co_u32_e64 v80, s[22:23], s12, v57
	v_subb_co_u32_e32 v55, vcc, v55, v59, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v81, s[24:25], 0, v68, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v81
	v_subb_co_u32_e64 v68, s[22:23], v68, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v82, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v80
	v_cmp_le_u32_e32 vcc, s13, v55
	v_mul_lo_u32 v39, v39, s33
	v_cndmask_b32_e64 v83, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v81
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v57
	v_cndmask_b32_e64 v82, v82, v83, s[24:25]
	v_subrev_co_u32_e64 v83, s[22:23], s12, v80
	s_nop 1
	v_subbrev_co_u32_e64 v68, s[22:23], 0, v68, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v82
	s_nop 1
	v_cndmask_b32_e64 v68, v81, v68, s[22:23]
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v55
	s_nop 1
	v_cndmask_b32_e32 v59, v59, v81, vcc
	v_cmp_ne_u32_e32 vcc, 0, v59
	v_cndmask_b32_e64 v59, v80, v83, s[22:23]
	v_or_b32_e32 v80, 5, v86
	v_cndmask_b32_e32 v57, v57, v59, vcc
	v_cndmask_b32_e32 v55, v55, v68, vcc
	v_xor_b32_e32 v57, v57, v76
	v_xor_b32_e32 v55, v55, v76
	v_sub_co_u32_e32 v57, vcc, v57, v76
	v_mov_b32_e32 v81, v87
	s_nop 0
	v_subb_co_u32_e32 v55, vcc, v55, v76, vcc
	v_ashrrev_i32_e32 v55, 31, v55
	v_and_b32_e32 v55, s42, v55
	v_add3_u32 v55, v55, v57, v39
	v_sub_co_u32_e32 v39, vcc, -6, v86
	v_cndmask_b32_e64 v82, v80, v39, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v57, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v83, v87, v57, s[18:19]
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v57, v82, v84
	v_xor_b32_e32 v39, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v57, s5, 0
	v_mul_hi_u32 v68, v57, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[92:93], s[6:7], v39, s4, 0
	v_add_co_u32_e32 v59, vcc, v82, v92
	v_mad_u64_u32 v[90:91], s[6:7], v39, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v93, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v59, s13, v82
	v_mul_lo_u32 v68, s12, v83
	v_mad_u64_u32 v[90:91], s[6:7], s12, v82, 0
	v_add3_u32 v59, v91, v68, v59
	v_sub_co_u32_e32 v57, vcc, v57, v90
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[76:77]
	s_nop 0
	v_subb_co_u32_e64 v68, s[22:23], v39, v59, vcc
	v_sub_u32_e32 v39, v39, v59
	v_cmp_le_u32_e64 s[22:23], s13, v68
	v_subb_co_u32_e32 v39, vcc, v39, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v57
	v_subrev_co_u32_e32 v57, vcc, s12, v57
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v39, vcc, 0, v39, vcc
	v_cmp_le_u32_e32 vcc, s13, v39
	v_cmp_eq_u32_e64 s[22:23], s13, v68
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v57
	v_cndmask_b32_e64 v68, v83, v85, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v39
	s_nop 1
	v_cndmask_b32_e32 v39, v59, v57, vcc
	v_cmp_ne_u32_e32 vcc, 0, v39
	v_xor_b32_e32 v57, s64, v84
	v_xor_b32_e32 v59, v80, v76
	v_cndmask_b32_e64 v39, 1, 2, vcc
	v_add_u32_e32 v39, v82, v39
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_mul_hi_u32 v68, v59, s4
	s_nop 0
	v_cndmask_b32_e32 v39, v82, v39, vcc
	v_xor_b32_e32 v39, v39, v57
	v_sub_u32_e32 v39, v39, v57
	v_xor_b32_e32 v57, v81, v76
	v_mad_u64_u32 v[80:81], s[6:7], v59, s5, 0
	v_lshl_add_u64 v[80:81], v[68:69], 0, v[80:81]
	v_mad_u64_u32 v[84:85], s[6:7], v57, s4, 0
	v_add_co_u32_e32 v68, vcc, v80, v84
	v_mad_u64_u32 v[82:83], s[6:7], v57, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v80, vcc, v81, v85, vcc
	v_mov_b32_e32 v81, s41
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[82:83]
	v_mul_lo_u32 v68, s13, v80
	v_mul_lo_u32 v82, s12, v81
	v_mad_u64_u32 v[80:81], s[6:7], s12, v80, 0
	v_add3_u32 v68, v81, v82, v68
	v_sub_u32_e32 v81, v57, v68
	v_sub_co_u32_e32 v59, vcc, v59, v80
	v_xad_u32 v39, v39, v76, v30
	s_nop 0
	v_subb_co_u32_e64 v80, s[22:23], v81, v61, vcc
	v_subrev_co_u32_e64 v81, s[22:23], s12, v59
	v_subb_co_u32_e32 v57, vcc, v57, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v82, s[24:25], 0, v80, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v82
	v_subb_co_u32_e64 v80, s[22:23], v80, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v81
	v_cmp_le_u32_e32 vcc, s13, v57
	v_mul_lo_u32 v39, v39, s33
	v_cndmask_b32_e64 v84, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v82
	v_cndmask_b32_e64 v68, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v59
	v_cndmask_b32_e64 v83, v83, v84, s[24:25]
	v_subrev_co_u32_e64 v84, s[22:23], s12, v81
	s_nop 1
	v_subbrev_co_u32_e64 v80, s[22:23], 0, v80, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v83
	s_nop 1
	v_cndmask_b32_e64 v80, v82, v80, s[22:23]
	v_cndmask_b32_e64 v82, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v57
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v82, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_cndmask_b32_e64 v68, v81, v84, s[22:23]
	v_mov_b32_e32 v81, v87
	v_cndmask_b32_e32 v59, v59, v68, vcc
	v_cndmask_b32_e32 v57, v57, v80, vcc
	v_xor_b32_e32 v59, v59, v76
	v_xor_b32_e32 v57, v57, v76
	v_sub_co_u32_e32 v59, vcc, v59, v76
	v_or_b32_e32 v80, 6, v86
	s_nop 0
	v_subb_co_u32_e32 v57, vcc, v57, v76, vcc
	v_ashrrev_i32_e32 v57, 31, v57
	v_and_b32_e32 v57, s42, v57
	v_add3_u32 v57, v57, v59, v39
	v_sub_co_u32_e32 v39, vcc, -7, v86
	v_cndmask_b32_e64 v82, v80, v39, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v59, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v83, v87, v59, s[18:19]
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v59, v82, v84
	v_xor_b32_e32 v39, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v59, s5, 0
	v_mul_hi_u32 v68, v59, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[92:93], s[6:7], v39, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v92
	v_mad_u64_u32 v[90:91], s[6:7], v39, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v93, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v83, s12, v83
	v_mad_u64_u32 v[90:91], s[6:7], s12, v82, 0
	v_add3_u32 v68, v91, v83, v68
	v_sub_co_u32_e32 v59, vcc, v59, v90
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[76:77]
	s_nop 0
	v_subb_co_u32_e64 v83, s[22:23], v39, v68, vcc
	v_sub_u32_e32 v39, v39, v68
	v_cmp_le_u32_e64 s[22:23], s13, v83
	v_subb_co_u32_e32 v39, vcc, v39, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v59
	v_subrev_co_u32_e32 v59, vcc, s12, v59
	s_nop 0
	v_cndmask_b32_e64 v90, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v39, vcc, 0, v39, vcc
	v_cmp_le_u32_e32 vcc, s13, v39
	v_cmp_eq_u32_e64 s[22:23], s13, v83
	s_nop 0
	v_cndmask_b32_e64 v68, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v59
	v_cndmask_b32_e64 v83, v85, v90, s[22:23]
	v_xor_b32_e32 v90, v80, v76
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v39
	s_nop 1
	v_cndmask_b32_e32 v39, v68, v59, vcc
	v_cmp_ne_u32_e32 vcc, 0, v39
	v_xor_b32_e32 v59, s64, v84
	v_mul_hi_u32 v68, v90, s4
	v_cndmask_b32_e64 v39, 1, 2, vcc
	v_add_u32_e32 v39, v82, v39
	v_cmp_ne_u32_e32 vcc, 0, v83
	s_nop 1
	v_cndmask_b32_e32 v39, v82, v39, vcc
	v_xor_b32_e32 v39, v39, v59
	v_sub_u32_e32 v39, v39, v59
	v_xor_b32_e32 v59, v81, v76
	v_mad_u64_u32 v[80:81], s[6:7], v90, s5, 0
	v_lshl_add_u64 v[80:81], v[68:69], 0, v[80:81]
	v_mad_u64_u32 v[84:85], s[6:7], v59, s4, 0
	v_add_co_u32_e32 v68, vcc, v80, v84
	v_mad_u64_u32 v[82:83], s[6:7], v59, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v80, vcc, v81, v85, vcc
	v_mov_b32_e32 v81, s41
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[82:83]
	v_mul_lo_u32 v68, s13, v80
	v_mul_lo_u32 v82, s12, v81
	v_mad_u64_u32 v[80:81], s[6:7], s12, v80, 0
	v_add3_u32 v68, v81, v82, v68
	v_sub_u32_e32 v81, v59, v68
	v_sub_co_u32_e32 v80, vcc, v90, v80
	v_xad_u32 v39, v39, v76, v30
	s_nop 0
	v_subb_co_u32_e64 v81, s[22:23], v81, v61, vcc
	v_subrev_co_u32_e64 v82, s[22:23], s12, v80
	v_subb_co_u32_e32 v59, vcc, v59, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v83, s[24:25], 0, v81, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v83
	v_subb_co_u32_e64 v81, s[22:23], v81, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v84, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v82
	v_cmp_le_u32_e32 vcc, s13, v59
	v_mul_lo_u32 v39, v39, s33
	v_cndmask_b32_e64 v85, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v83
	v_cndmask_b32_e64 v68, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v80
	v_cndmask_b32_e64 v84, v84, v85, s[24:25]
	v_subrev_co_u32_e64 v85, s[22:23], s12, v82
	s_nop 1
	v_subbrev_co_u32_e64 v81, s[22:23], 0, v81, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v84
	s_nop 1
	v_cndmask_b32_e64 v81, v83, v81, s[22:23]
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v59
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_cndmask_b32_e64 v68, v82, v85, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v68, v80, v68, vcc
	v_cndmask_b32_e32 v59, v59, v81, vcc
	v_xor_b32_e32 v68, v68, v76
	v_xor_b32_e32 v59, v59, v76
	v_sub_co_u32_e32 v68, vcc, v68, v76
	v_or_b32_e32 v80, 7, v86
	s_nop 0
	v_subb_co_u32_e32 v59, vcc, v59, v76, vcc
	v_ashrrev_i32_e32 v59, 31, v59
	v_and_b32_e32 v59, s42, v59
	v_add3_u32 v59, v59, v68, v39
	v_sub_co_u32_e32 v39, vcc, -8, v86
	v_cndmask_b32_e64 v82, v80, v39, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v68, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v83, v87, v68, s[18:19]
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v85, v82, v84
	v_xor_b32_e32 v39, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v85, s5, 0
	v_mul_hi_u32 v68, v85, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[92:93], s[6:7], v39, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v92
	v_mad_u64_u32 v[90:91], s[6:7], v39, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v93, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v83, s12, v83
	v_mad_u64_u32 v[90:91], s[6:7], s12, v82, 0
	v_add3_u32 v68, v91, v83, v68
	v_sub_co_u32_e32 v83, vcc, v85, v90
	v_mov_b32_e32 v81, v87
	s_nop 0
	v_subb_co_u32_e64 v85, s[22:23], v39, v68, vcc
	v_sub_u32_e32 v39, v39, v68
	v_subb_co_u32_e32 v39, vcc, v39, v61, vcc
	v_subrev_co_u32_e32 v68, vcc, s12, v83
	v_cmp_le_u32_e64 s[22:23], s13, v85
	s_nop 0
	v_subbrev_co_u32_e32 v39, vcc, 0, v39, vcc
	v_cmp_le_u32_e32 vcc, s13, v39
	v_cndmask_b32_e64 v90, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v83
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v68
	v_cndmask_b32_e64 v91, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s13, v85
	v_cndmask_b32_e64 v68, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v39
	v_cndmask_b32_e64 v85, v90, v91, s[22:23]
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[76:77]
	v_cndmask_b32_e32 v39, v83, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v39
	v_xor_b32_e32 v68, s64, v84
	v_xor_b32_e32 v91, v80, v76
	v_cndmask_b32_e64 v39, 1, 2, vcc
	v_add_u32_e32 v39, v82, v39
	v_cmp_ne_u32_e32 vcc, 0, v85
	v_xor_b32_e32 v90, v81, v76
	v_mad_u64_u32 v[80:81], s[6:7], v91, s5, 0
	v_cndmask_b32_e32 v39, v82, v39, vcc
	v_xor_b32_e32 v39, v39, v68
	v_sub_u32_e32 v39, v39, v68
	v_mul_hi_u32 v68, v91, s4
	v_lshl_add_u64 v[80:81], v[68:69], 0, v[80:81]
	v_mad_u64_u32 v[84:85], s[6:7], v90, s4, 0
	v_add_co_u32_e32 v68, vcc, v80, v84
	v_mad_u64_u32 v[82:83], s[6:7], v90, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v80, vcc, v81, v85, vcc
	v_mov_b32_e32 v81, s41
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[82:83]
	v_mul_lo_u32 v68, s13, v80
	v_mul_lo_u32 v82, s12, v81
	v_mad_u64_u32 v[80:81], s[6:7], s12, v80, 0
	v_add3_u32 v68, v81, v82, v68
	v_sub_u32_e32 v81, v90, v68
	v_sub_co_u32_e32 v80, vcc, v91, v80
	v_xad_u32 v39, v39, v76, v30
	s_nop 0
	v_subb_co_u32_e64 v81, s[22:23], v81, v61, vcc
	v_subrev_co_u32_e64 v82, s[22:23], s12, v80
	v_subb_co_u32_e32 v68, vcc, v90, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v83, s[24:25], 0, v81, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v83
	v_subb_co_u32_e64 v81, s[22:23], v81, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v84, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v82
	v_cmp_le_u32_e32 vcc, s13, v68
	v_mul_lo_u32 v39, v39, s33
	v_cndmask_b32_e64 v85, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v83
	s_nop 1
	v_cndmask_b32_e64 v84, v84, v85, s[24:25]
	v_subrev_co_u32_e64 v85, s[22:23], s12, v82
	s_nop 1
	v_subbrev_co_u32_e64 v81, s[22:23], 0, v81, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v84
	s_nop 1
	v_cndmask_b32_e64 v81, v83, v81, s[22:23]
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v80
	s_nop 1
	v_cndmask_b32_e64 v84, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v84, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v81, vcc
	v_cndmask_b32_e64 v81, v82, v85, s[22:23]
	v_cndmask_b32_e32 v80, v80, v81, vcc
	v_xor_b32_e32 v80, v80, v76
	v_xor_b32_e32 v68, v68, v76
	v_sub_co_u32_e32 v80, vcc, v80, v76
	v_mov_b32_e32 v81, v87
	s_nop 0
	v_subb_co_u32_e32 v68, vcc, v68, v76, vcc
	v_ashrrev_i32_e32 v68, 31, v68
	v_and_b32_e32 v68, s42, v68
	v_add3_u32 v68, v68, v80, v39
	buffer_load_ubyte v35, v35, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v39, v37, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v37, v51, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v51, v53, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v53, v55, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v57, v57, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v55, v59, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v59, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, -9, v86
	v_or_b32_e32 v80, 8, v86
	s_nop 0
	v_subb_co_u32_e32 v82, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v83, v87, v82, s[18:19]
	v_ashrrev_i32_e32 v84, 31, v83
	v_cndmask_b32_e64 v82, v80, v68, s[18:19]
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v98, v82, v84
	v_xor_b32_e32 v85, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v98, s5, 0
	v_mul_hi_u32 v68, v98, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[92:93], s[6:7], v85, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v92
	v_mad_u64_u32 v[90:91], s[6:7], v85, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v93, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v83, s12, v83
	v_mad_u64_u32 v[90:91], s[6:7], s12, v82, 0
	v_add3_u32 v68, v91, v83, v68
	v_sub_co_u32_e32 v83, vcc, v98, v90
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[76:77]
	s_nop 0
	v_subb_co_u32_e64 v90, s[22:23], v85, v68, vcc
	v_sub_u32_e32 v68, v85, v68
	v_cmp_le_u32_e64 s[22:23], s13, v90
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v83
	v_subrev_co_u32_e32 v83, vcc, s12, v83
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cmp_eq_u32_e64 s[22:23], s13, v90
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v83
	v_cndmask_b32_e64 v90, v91, v92, s[22:23]
	v_xor_b32_e32 v92, v80, v76
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	v_xor_b32_e32 v91, v81, v76
	v_mad_u64_u32 v[80:81], s[6:7], v92, s5, 0
	v_cndmask_b32_e32 v68, v85, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v82, v68
	v_cmp_ne_u32_e32 vcc, 0, v90
	s_nop 1
	v_cndmask_b32_e32 v68, v82, v68, vcc
	v_xor_b32_e32 v82, s64, v84
	v_xor_b32_e32 v68, v68, v82
	v_sub_u32_e32 v68, v68, v82
	v_xad_u32 v90, v68, v76, v30
	v_mul_hi_u32 v68, v92, s4
	v_lshl_add_u64 v[80:81], v[68:69], 0, v[80:81]
	v_mad_u64_u32 v[84:85], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v80, v84
	v_mad_u64_u32 v[82:83], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v80, vcc, v81, v85, vcc
	v_mov_b32_e32 v81, s41
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[82:83]
	v_mul_lo_u32 v68, s13, v80
	v_mul_lo_u32 v82, s12, v81
	v_mad_u64_u32 v[80:81], s[6:7], s12, v80, 0
	v_add3_u32 v68, v81, v82, v68
	v_sub_u32_e32 v81, v91, v68
	v_sub_co_u32_e32 v80, vcc, v92, v80
	s_nop 1
	v_subb_co_u32_e64 v81, s[22:23], v81, v61, vcc
	v_subrev_co_u32_e64 v82, s[22:23], s12, v80
	v_subb_co_u32_e32 v68, vcc, v91, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v83, s[24:25], 0, v81, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v83
	v_subb_co_u32_e64 v81, s[22:23], v81, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v84, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v82
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v83
	s_nop 1
	v_cndmask_b32_e64 v84, v84, v85, s[24:25]
	v_subrev_co_u32_e64 v85, s[22:23], s12, v82
	s_nop 1
	v_subbrev_co_u32_e64 v81, s[22:23], 0, v81, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v84
	s_nop 1
	v_cndmask_b32_e64 v81, v83, v81, s[22:23]
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v80
	s_nop 1
	v_cndmask_b32_e64 v84, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v84, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	v_mov_b32_e32 v83, v87
	s_nop 0
	v_cndmask_b32_e32 v68, v68, v81, vcc
	v_cndmask_b32_e64 v81, v82, v85, s[22:23]
	v_cndmask_b32_e32 v80, v80, v81, vcc
	v_xor_b32_e32 v80, v80, v76
	v_xor_b32_e32 v68, v68, v76
	v_sub_co_u32_e32 v80, vcc, v80, v76
	v_mul_lo_u32 v81, v90, s33
	s_nop 0
	v_subb_co_u32_e32 v68, vcc, v68, v76, vcc
	v_ashrrev_i32_e32 v68, 31, v68
	v_and_b32_e32 v68, s42, v68
	v_add3_u32 v80, v68, v80, v81
	v_sub_co_u32_e32 v68, vcc, -10, v86
	v_or_b32_e32 v82, 9, v86
	s_nop 0
	v_subb_co_u32_e32 v81, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v85, v87, v81, s[18:19]
	v_ashrrev_i32_e32 v90, 31, v85
	v_cndmask_b32_e64 v84, v82, v68, s[18:19]
	v_mov_b32_e32 v91, v90
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_xor_b32_e32 v91, v84, v90
	v_xor_b32_e32 v81, v85, v90
	v_mad_u64_u32 v[84:85], s[6:7], v91, s5, 0
	v_mul_hi_u32 v68, v91, s4
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[98:99], s[6:7], v81, s4, 0
	v_add_co_u32_e32 v68, vcc, v84, v98
	v_mad_u64_u32 v[92:93], s[6:7], v81, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v99, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v85, s12, v85
	v_mad_u64_u32 v[92:93], s[6:7], s12, v84, 0
	v_add3_u32 v68, v93, v85, v68
	v_sub_co_u32_e32 v85, vcc, v91, v92
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[76:77]
	s_nop 0
	v_subb_co_u32_e64 v91, s[22:23], v81, v68, vcc
	v_sub_u32_e32 v68, v81, v68
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	v_subrev_co_u32_e32 v81, vcc, s12, v85
	v_cmp_le_u32_e64 s[22:23], s13, v91
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cndmask_b32_e64 v92, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v85
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v81
	v_cndmask_b32_e64 v93, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s13, v91
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	v_cndmask_b32_e64 v91, v92, v93, s[22:23]
	v_xor_b32_e32 v93, v82, v76
	v_cndmask_b32_e32 v68, v85, v81, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v81, s64, v90
	v_xor_b32_e32 v92, v83, v76
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v84, v68
	v_cmp_ne_u32_e32 vcc, 0, v91
	v_mad_u64_u32 v[82:83], s[6:7], v93, s5, 0
	s_nop 0
	v_cndmask_b32_e32 v68, v84, v68, vcc
	v_xor_b32_e32 v68, v68, v81
	v_sub_u32_e32 v68, v68, v81
	v_xad_u32 v81, v68, v76, v30
	v_mul_hi_u32 v68, v93, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[90:91], s[6:7], v92, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v90
	v_mad_u64_u32 v[84:85], s[6:7], v92, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v91, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v84, s12, v83
	v_mad_u64_u32 v[82:83], s[6:7], s12, v82, 0
	v_add3_u32 v68, v83, v84, v68
	v_sub_u32_e32 v83, v92, v68
	v_sub_co_u32_e32 v82, vcc, v93, v82
	v_mul_lo_u32 v81, v81, s33
	s_nop 0
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, vcc
	v_subrev_co_u32_e64 v84, s[22:23], s12, v82
	v_subb_co_u32_e32 v68, vcc, v92, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[24:25], 0, v83, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v85
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v90, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v84
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v85
	s_nop 1
	v_cndmask_b32_e64 v90, v90, v91, s[24:25]
	v_subrev_co_u32_e64 v91, s[22:23], s12, v84
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[22:23], 0, v83, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v90
	s_nop 1
	v_cndmask_b32_e64 v83, v85, v83, s[22:23]
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v82
	s_nop 1
	v_cndmask_b32_e64 v90, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v85, v85, v90, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v83, vcc
	v_cndmask_b32_e64 v83, v84, v91, s[22:23]
	v_cndmask_b32_e32 v82, v82, v83, vcc
	v_xor_b32_e32 v82, v82, v76
	v_xor_b32_e32 v68, v68, v76
	v_sub_co_u32_e32 v82, vcc, v82, v76
	v_mov_b32_e32 v83, v87
	s_nop 0
	v_subb_co_u32_e32 v68, vcc, v68, v76, vcc
	v_ashrrev_i32_e32 v68, 31, v68
	v_and_b32_e32 v68, s42, v68
	v_add3_u32 v81, v68, v82, v81
	v_sub_co_u32_e32 v68, vcc, -11, v86
	v_or_b32_e32 v82, 10, v86
	s_nop 0
	v_subb_co_u32_e32 v84, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v85, v87, v84, s[18:19]
	v_ashrrev_i32_e32 v90, 31, v85
	v_cndmask_b32_e64 v84, v82, v68, s[18:19]
	v_mov_b32_e32 v91, v90
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_xor_b32_e32 v100, v84, v90
	v_xor_b32_e32 v91, v85, v90
	v_mad_u64_u32 v[84:85], s[6:7], v100, s5, 0
	v_mul_hi_u32 v68, v100, s4
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[98:99], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v84, v98
	v_mad_u64_u32 v[92:93], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v99, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[92:93]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v85, s12, v85
	v_mad_u64_u32 v[92:93], s[6:7], s12, v84, 0
	v_add3_u32 v68, v93, v85, v68
	v_sub_co_u32_e32 v85, vcc, v100, v92
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[76:77]
	s_nop 0
	v_subb_co_u32_e64 v92, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_cmp_le_u32_e64 s[22:23], s13, v92
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v85
	v_subrev_co_u32_e32 v85, vcc, s12, v85
	s_nop 0
	v_cndmask_b32_e64 v98, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cmp_eq_u32_e64 s[22:23], s13, v92
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v85
	v_cndmask_b32_e64 v92, v93, v98, s[22:23]
	v_xor_b32_e32 v98, v82, v76
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	v_xor_b32_e32 v93, v83, v76
	v_mad_u64_u32 v[82:83], s[6:7], v98, s5, 0
	v_cndmask_b32_e32 v68, v91, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v84, v68
	v_cmp_ne_u32_e32 vcc, 0, v92
	s_nop 1
	v_cndmask_b32_e32 v68, v84, v68, vcc
	v_xor_b32_e32 v84, s64, v90
	v_xor_b32_e32 v68, v68, v84
	v_sub_u32_e32 v68, v68, v84
	v_xad_u32 v92, v68, v76, v30
	v_mul_hi_u32 v68, v98, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[90:91], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v90
	v_mad_u64_u32 v[84:85], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v91, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v84, s12, v83
	v_mad_u64_u32 v[82:83], s[6:7], s12, v82, 0
	v_add3_u32 v68, v83, v84, v68
	v_sub_u32_e32 v83, v93, v68
	v_sub_co_u32_e32 v82, vcc, v98, v82
	s_nop 1
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, vcc
	v_subrev_co_u32_e64 v84, s[22:23], s12, v82
	v_subb_co_u32_e32 v68, vcc, v93, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[24:25], 0, v83, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v85
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v90, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v84
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v85
	s_nop 1
	v_cndmask_b32_e64 v90, v90, v91, s[24:25]
	v_subrev_co_u32_e64 v91, s[22:23], s12, v84
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[22:23], 0, v83, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v90
	s_nop 1
	v_cndmask_b32_e64 v83, v85, v83, s[22:23]
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v82
	s_nop 1
	v_cndmask_b32_e64 v90, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v85, v85, v90, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	v_mov_b32_e32 v85, v87
	s_nop 0
	v_cndmask_b32_e32 v68, v68, v83, vcc
	v_cndmask_b32_e64 v83, v84, v91, s[22:23]
	v_cndmask_b32_e32 v82, v82, v83, vcc
	v_xor_b32_e32 v82, v82, v76
	v_xor_b32_e32 v68, v68, v76
	v_sub_co_u32_e32 v82, vcc, v82, v76
	v_mul_lo_u32 v83, v92, s33
	s_nop 0
	v_subb_co_u32_e32 v68, vcc, v68, v76, vcc
	v_ashrrev_i32_e32 v68, 31, v68
	v_and_b32_e32 v68, s42, v68
	v_add3_u32 v82, v68, v82, v83
	v_sub_co_u32_e32 v68, vcc, -12, v86
	v_or_b32_e32 v84, 11, v86
	s_nop 0
	v_subb_co_u32_e32 v83, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v91, v87, v83, s[18:19]
	v_ashrrev_i32_e32 v92, 31, v91
	v_cndmask_b32_e64 v90, v84, v68, s[18:19]
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[92:93]
	v_xor_b32_e32 v93, v90, v92
	v_xor_b32_e32 v83, v91, v92
	v_mad_u64_u32 v[90:91], s[6:7], v93, s5, 0
	v_mul_hi_u32 v68, v93, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[100:101], s[6:7], v83, s4, 0
	v_add_co_u32_e32 v68, vcc, v90, v100
	v_mad_u64_u32 v[98:99], s[6:7], v83, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v90, vcc, v91, v101, vcc
	v_mov_b32_e32 v91, s41
	s_nop 0
	v_addc_co_u32_e32 v99, vcc, 0, v99, vcc
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[98:99]
	v_mul_lo_u32 v68, s13, v90
	v_mul_lo_u32 v91, s12, v91
	v_mad_u64_u32 v[98:99], s[6:7], s12, v90, 0
	v_add3_u32 v68, v99, v91, v68
	v_sub_co_u32_e32 v91, vcc, v93, v98
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[76:77]
	s_nop 0
	v_subb_co_u32_e64 v93, s[22:23], v83, v68, vcc
	v_sub_u32_e32 v68, v83, v68
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	v_subrev_co_u32_e32 v83, vcc, s12, v91
	v_cmp_le_u32_e64 s[22:23], s13, v93
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cndmask_b32_e64 v98, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v91
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v83
	v_cndmask_b32_e64 v99, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s13, v93
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	v_cndmask_b32_e64 v93, v98, v99, s[22:23]
	v_xor_b32_e32 v99, v84, v76
	v_cndmask_b32_e32 v68, v91, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v83, s64, v92
	v_xor_b32_e32 v98, v85, v76
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v90, v68
	v_cmp_ne_u32_e32 vcc, 0, v93
	v_mad_u64_u32 v[84:85], s[6:7], v99, s5, 0
	s_nop 0
	v_cndmask_b32_e32 v68, v90, v68, vcc
	v_xor_b32_e32 v68, v68, v83
	v_sub_u32_e32 v68, v68, v83
	v_xad_u32 v83, v68, v76, v30
	v_mul_hi_u32 v68, v99, s4
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[92:93], s[6:7], v98, s4, 0
	v_add_co_u32_e32 v68, vcc, v84, v92
	v_mad_u64_u32 v[90:91], s[6:7], v98, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v93, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v90, s12, v85
	v_mad_u64_u32 v[84:85], s[6:7], s12, v84, 0
	v_add3_u32 v68, v85, v90, v68
	v_sub_u32_e32 v85, v98, v68
	v_sub_co_u32_e32 v84, vcc, v99, v84
	v_mul_lo_u32 v83, v83, s33
	s_nop 0
	v_subb_co_u32_e64 v85, s[22:23], v85, v61, vcc
	v_subrev_co_u32_e64 v90, s[22:23], s12, v84
	v_subb_co_u32_e32 v68, vcc, v98, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[24:25], 0, v85, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v91
	v_subb_co_u32_e64 v85, s[22:23], v85, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v90
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v91
	s_nop 1
	v_cndmask_b32_e64 v92, v92, v93, s[24:25]
	v_subrev_co_u32_e64 v93, s[22:23], s12, v90
	s_nop 1
	v_subbrev_co_u32_e64 v85, s[22:23], 0, v85, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v92
	s_nop 1
	v_cndmask_b32_e64 v85, v91, v85, s[22:23]
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v84
	s_nop 1
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v91, v91, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v91
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v85, vcc
	v_cndmask_b32_e64 v85, v90, v93, s[22:23]
	v_cndmask_b32_e32 v84, v84, v85, vcc
	v_xor_b32_e32 v84, v84, v76
	v_xor_b32_e32 v68, v68, v76
	v_sub_co_u32_e32 v84, vcc, v84, v76
	v_mov_b32_e32 v85, v87
	s_nop 0
	v_subb_co_u32_e32 v68, vcc, v68, v76, vcc
	v_ashrrev_i32_e32 v68, 31, v68
	v_and_b32_e32 v68, s42, v68
	v_add3_u32 v83, v68, v84, v83
	v_sub_co_u32_e32 v68, vcc, -13, v86
	v_or_b32_e32 v84, 12, v86
	s_nop 0
	v_subb_co_u32_e32 v90, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v91, v87, v90, s[18:19]
	v_ashrrev_i32_e32 v92, 31, v91
	v_cndmask_b32_e64 v90, v84, v68, s[18:19]
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[92:93]
	v_xor_b32_e32 v102, v90, v92
	v_xor_b32_e32 v93, v91, v92
	v_mad_u64_u32 v[90:91], s[6:7], v102, s5, 0
	v_mul_hi_u32 v68, v102, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[100:101], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v90, v100
	v_mad_u64_u32 v[98:99], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v90, vcc, v91, v101, vcc
	v_mov_b32_e32 v91, s41
	s_nop 0
	v_addc_co_u32_e32 v99, vcc, 0, v99, vcc
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[98:99]
	v_mul_lo_u32 v68, s13, v90
	v_mul_lo_u32 v91, s12, v91
	v_mad_u64_u32 v[98:99], s[6:7], s12, v90, 0
	v_add3_u32 v68, v99, v91, v68
	v_sub_co_u32_e32 v91, vcc, v102, v98
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[76:77]
	s_nop 0
	v_subb_co_u32_e64 v98, s[22:23], v93, v68, vcc
	v_sub_u32_e32 v68, v93, v68
	v_cmp_le_u32_e64 s[22:23], s13, v98
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v99, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v91
	v_subrev_co_u32_e32 v91, vcc, s12, v91
	s_nop 0
	v_cndmask_b32_e64 v100, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cmp_eq_u32_e64 s[22:23], s13, v98
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v91
	v_cndmask_b32_e64 v98, v99, v100, s[22:23]
	v_xor_b32_e32 v100, v84, v76
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	v_xor_b32_e32 v99, v85, v76
	v_mad_u64_u32 v[84:85], s[6:7], v100, s5, 0
	v_cndmask_b32_e32 v68, v93, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v90, v68
	v_cmp_ne_u32_e32 vcc, 0, v98
	s_nop 1
	v_cndmask_b32_e32 v68, v90, v68, vcc
	v_xor_b32_e32 v90, s64, v92
	v_xor_b32_e32 v68, v68, v90
	v_sub_u32_e32 v68, v68, v90
	v_xad_u32 v98, v68, v76, v30
	v_mul_hi_u32 v68, v100, s4
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[92:93], s[6:7], v99, s4, 0
	v_add_co_u32_e32 v68, vcc, v84, v92
	v_mad_u64_u32 v[90:91], s[6:7], v99, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v93, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v90, s12, v85
	v_mad_u64_u32 v[84:85], s[6:7], s12, v84, 0
	v_add3_u32 v68, v85, v90, v68
	v_sub_u32_e32 v85, v99, v68
	v_sub_co_u32_e32 v84, vcc, v100, v84
	s_nop 1
	v_subb_co_u32_e64 v85, s[22:23], v85, v61, vcc
	v_subrev_co_u32_e64 v90, s[22:23], s12, v84
	v_subb_co_u32_e32 v68, vcc, v99, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[24:25], 0, v85, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v91
	v_subb_co_u32_e64 v85, s[22:23], v85, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v90
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v91
	s_nop 1
	v_cndmask_b32_e64 v92, v92, v93, s[24:25]
	v_subrev_co_u32_e64 v93, s[22:23], s12, v90
	s_nop 1
	v_subbrev_co_u32_e64 v85, s[22:23], 0, v85, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v92
	s_nop 1
	v_cndmask_b32_e64 v85, v91, v85, s[22:23]
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v84
	s_nop 1
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v91, v91, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v91
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v85, vcc
	v_cndmask_b32_e64 v85, v90, v93, s[22:23]
	v_cndmask_b32_e32 v84, v84, v85, vcc
	v_xor_b32_e32 v84, v84, v76
	v_xor_b32_e32 v68, v68, v76
	v_sub_co_u32_e32 v84, vcc, v84, v76
	v_mul_lo_u32 v85, v98, s33
	s_nop 0
	v_subb_co_u32_e32 v68, vcc, v68, v76, vcc
	v_ashrrev_i32_e32 v68, 31, v68
	v_and_b32_e32 v68, s42, v68
	v_add3_u32 v102, v68, v84, v85
	v_sub_co_u32_e32 v68, vcc, -14, v86
	v_or_b32_e32 v84, 13, v86
	s_nop 0
	v_subb_co_u32_e32 v90, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v91, v87, v90, s[18:19]
	v_ashrrev_i32_e32 v92, 31, v91
	v_cndmask_b32_e64 v90, v84, v68, s[18:19]
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[92:93]
	v_xor_b32_e32 v103, v90, v92
	v_xor_b32_e32 v93, v91, v92
	v_mad_u64_u32 v[90:91], s[6:7], v103, s5, 0
	v_mul_hi_u32 v68, v103, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[100:101], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v90, v100
	v_mad_u64_u32 v[98:99], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v90, vcc, v91, v101, vcc
	v_mov_b32_e32 v91, s41
	s_nop 0
	v_addc_co_u32_e32 v99, vcc, 0, v99, vcc
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[98:99]
	v_mul_lo_u32 v68, s13, v90
	v_mul_lo_u32 v91, s12, v91
	v_mad_u64_u32 v[98:99], s[6:7], s12, v90, 0
	v_add3_u32 v68, v99, v91, v68
	v_sub_co_u32_e32 v91, vcc, v103, v98
	v_mov_b32_e32 v85, v87
	s_nop 0
	v_subb_co_u32_e64 v98, s[22:23], v93, v68, vcc
	v_sub_u32_e32 v68, v93, v68
	v_cmp_le_u32_e64 s[22:23], s13, v98
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v99, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v91
	v_subrev_co_u32_e32 v91, vcc, s12, v91
	s_nop 0
	v_cndmask_b32_e64 v100, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cmp_eq_u32_e64 s[22:23], s13, v98
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[76:77]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v91
	v_cndmask_b32_e64 v98, v99, v100, s[22:23]
	v_xor_b32_e32 v100, v84, v76
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	v_xor_b32_e32 v99, v85, v76
	v_mad_u64_u32 v[84:85], s[6:7], v100, s5, 0
	v_cndmask_b32_e32 v68, v93, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v90, v68
	v_cmp_ne_u32_e32 vcc, 0, v98
	s_nop 1
	v_cndmask_b32_e32 v68, v90, v68, vcc
	v_xor_b32_e32 v90, s64, v92
	v_xor_b32_e32 v68, v68, v90
	v_sub_u32_e32 v68, v68, v90
	v_xad_u32 v98, v68, v76, v30
	v_mul_hi_u32 v68, v100, s4
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[92:93], s[6:7], v99, s4, 0
	v_add_co_u32_e32 v68, vcc, v84, v92
	v_mad_u64_u32 v[90:91], s[6:7], v99, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v93, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v90, s12, v85
	v_mad_u64_u32 v[84:85], s[6:7], s12, v84, 0
	v_add3_u32 v68, v85, v90, v68
	v_sub_u32_e32 v85, v99, v68
	v_sub_co_u32_e32 v84, vcc, v100, v84
	s_nop 1
	v_subb_co_u32_e64 v85, s[22:23], v85, v61, vcc
	v_subrev_co_u32_e64 v90, s[22:23], s12, v84
	v_subb_co_u32_e32 v68, vcc, v99, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[24:25], 0, v85, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v91
	v_subb_co_u32_e64 v85, s[22:23], v85, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v90
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v91
	s_nop 1
	v_cndmask_b32_e64 v92, v92, v93, s[24:25]
	v_subrev_co_u32_e64 v93, s[22:23], s12, v90
	s_nop 1
	v_subbrev_co_u32_e64 v85, s[22:23], 0, v85, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v92
	s_nop 1
	v_cndmask_b32_e64 v85, v91, v85, s[22:23]
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v84
	s_nop 1
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v91, v91, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v91
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v85, vcc
	v_cndmask_b32_e64 v85, v90, v93, s[22:23]
	v_cndmask_b32_e32 v84, v84, v85, vcc
	v_xor_b32_e32 v84, v84, v76
	v_xor_b32_e32 v68, v68, v76
	v_sub_co_u32_e32 v84, vcc, v84, v76
	v_mul_lo_u32 v85, v98, s33
	s_nop 0
	v_subb_co_u32_e32 v68, vcc, v68, v76, vcc
	v_ashrrev_i32_e32 v68, 31, v68
	v_and_b32_e32 v68, s42, v68
	v_add3_u32 v103, v68, v84, v85
	v_sub_co_u32_e32 v68, vcc, -15, v86
	v_or_b32_e32 v84, 14, v86
	s_nop 0
	v_subb_co_u32_e32 v90, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v91, v87, v90, s[18:19]
	v_ashrrev_i32_e32 v92, 31, v91
	v_cndmask_b32_e64 v90, v84, v68, s[18:19]
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[92:93]
	v_xor_b32_e32 v104, v90, v92
	v_xor_b32_e32 v93, v91, v92
	v_mad_u64_u32 v[90:91], s[6:7], v104, s5, 0
	v_mul_hi_u32 v68, v104, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[100:101], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v90, v100
	v_mad_u64_u32 v[98:99], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v90, vcc, v91, v101, vcc
	v_mov_b32_e32 v91, s41
	s_nop 0
	v_addc_co_u32_e32 v99, vcc, 0, v99, vcc
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[98:99]
	v_mul_lo_u32 v68, s13, v90
	v_mul_lo_u32 v91, s12, v91
	v_mad_u64_u32 v[98:99], s[6:7], s12, v90, 0
	v_add3_u32 v68, v99, v91, v68
	v_sub_co_u32_e32 v91, vcc, v104, v98
	v_mov_b32_e32 v85, v87
	s_nop 0
	v_subb_co_u32_e64 v98, s[22:23], v93, v68, vcc
	v_sub_u32_e32 v68, v93, v68
	v_cmp_le_u32_e64 s[22:23], s13, v98
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v99, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v91
	v_subrev_co_u32_e32 v91, vcc, s12, v91
	s_nop 0
	v_cndmask_b32_e64 v100, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cmp_eq_u32_e64 s[22:23], s13, v98
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[76:77]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v91
	v_cndmask_b32_e64 v98, v99, v100, s[22:23]
	v_xor_b32_e32 v100, v84, v76
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	v_xor_b32_e32 v99, v85, v76
	v_mad_u64_u32 v[84:85], s[6:7], v100, s5, 0
	v_cndmask_b32_e32 v68, v93, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v90, v68
	v_cmp_ne_u32_e32 vcc, 0, v98
	s_nop 1
	v_cndmask_b32_e32 v68, v90, v68, vcc
	v_xor_b32_e32 v90, s64, v92
	v_xor_b32_e32 v68, v68, v90
	v_sub_u32_e32 v68, v68, v90
	v_xad_u32 v98, v68, v76, v30
	v_mul_hi_u32 v68, v100, s4
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[92:93], s[6:7], v99, s4, 0
	v_add_co_u32_e32 v68, vcc, v84, v92
	v_mad_u64_u32 v[90:91], s[6:7], v99, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v93, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v90, s12, v85
	v_mad_u64_u32 v[84:85], s[6:7], s12, v84, 0
	v_add3_u32 v68, v85, v90, v68
	v_sub_u32_e32 v85, v99, v68
	v_sub_co_u32_e32 v84, vcc, v100, v84
	s_nop 1
	v_subb_co_u32_e64 v85, s[22:23], v85, v61, vcc
	v_subrev_co_u32_e64 v90, s[22:23], s12, v84
	v_subb_co_u32_e32 v68, vcc, v99, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[24:25], 0, v85, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v91
	v_subb_co_u32_e64 v85, s[22:23], v85, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v90
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v91
	s_nop 1
	v_cndmask_b32_e64 v92, v92, v93, s[24:25]
	v_subrev_co_u32_e64 v93, s[22:23], s12, v90
	s_nop 1
	v_subbrev_co_u32_e64 v85, s[22:23], 0, v85, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v92
	s_nop 1
	v_cndmask_b32_e64 v85, v91, v85, s[22:23]
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v84
	s_nop 1
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v91, v91, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v91
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v85, vcc
	v_cndmask_b32_e64 v85, v90, v93, s[22:23]
	v_cndmask_b32_e32 v84, v84, v85, vcc
	v_xor_b32_e32 v84, v84, v76
	v_xor_b32_e32 v68, v68, v76
	v_sub_co_u32_e32 v84, vcc, v84, v76
	v_mul_lo_u32 v85, v98, s33
	s_nop 0
	v_subb_co_u32_e32 v68, vcc, v68, v76, vcc
	v_ashrrev_i32_e32 v68, 31, v68
	v_and_b32_e32 v68, s42, v68
	v_add3_u32 v105, v68, v84, v85
	v_sub_co_u32_e32 v68, vcc, -16, v86
	v_or_b32_e32 v84, 15, v86
	s_nop 0
	v_subb_co_u32_e32 v90, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v91, v87, v90, s[18:19]
	v_ashrrev_i32_e32 v92, 31, v91
	v_cndmask_b32_e64 v90, v84, v68, s[18:19]
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[92:93]
	v_xor_b32_e32 v104, v90, v92
	v_xor_b32_e32 v93, v91, v92
	v_mad_u64_u32 v[90:91], s[6:7], v104, s5, 0
	v_mul_hi_u32 v68, v104, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[100:101], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v90, v100
	v_mad_u64_u32 v[98:99], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v90, vcc, v91, v101, vcc
	v_mov_b32_e32 v91, s41
	s_nop 0
	v_addc_co_u32_e32 v99, vcc, 0, v99, vcc
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[98:99]
	v_mul_lo_u32 v68, s13, v90
	v_mul_lo_u32 v91, s12, v91
	v_mad_u64_u32 v[98:99], s[6:7], s12, v90, 0
	v_add3_u32 v68, v99, v91, v68
	v_sub_co_u32_e32 v91, vcc, v104, v98
	v_mov_b32_e32 v85, v87
	s_nop 0
	v_subb_co_u32_e64 v98, s[18:19], v93, v68, vcc
	v_sub_u32_e32 v68, v93, v68
	v_cmp_le_u32_e64 s[18:19], s13, v98
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v99, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s12, v91
	v_subrev_co_u32_e32 v91, vcc, s12, v91
	s_nop 0
	v_cndmask_b32_e64 v100, 0, -1, s[18:19]
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cmp_eq_u32_e64 s[18:19], s13, v98
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[76:77]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v91
	v_cndmask_b32_e64 v98, v99, v100, s[18:19]
	v_xor_b32_e32 v99, v84, v76
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	v_xor_b32_e32 v77, v85, v76
	v_mad_u64_u32 v[84:85], s[6:7], v99, s5, 0
	v_cndmask_b32_e32 v68, v93, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v90, v68
	v_cmp_ne_u32_e32 vcc, 0, v98
	s_nop 1
	v_cndmask_b32_e32 v68, v90, v68, vcc
	v_xor_b32_e32 v90, s64, v92
	v_xor_b32_e32 v68, v68, v90
	v_sub_u32_e32 v68, v68, v90
	v_xad_u32 v98, v68, v76, v30
	v_mul_hi_u32 v68, v99, s4
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[92:93], s[6:7], v77, s4, 0
	v_add_co_u32_e32 v68, vcc, v84, v92
	v_mad_u64_u32 v[90:91], s[6:7], v77, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v93, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v90, s12, v85
	v_mad_u64_u32 v[84:85], s[6:7], s12, v84, 0
	v_add3_u32 v68, v85, v90, v68
	v_sub_u32_e32 v85, v77, v68
	v_sub_co_u32_e32 v84, vcc, v99, v84
	s_movk_i32 s6, 0xf3ff
	s_nop 0
	v_subb_co_u32_e64 v85, s[18:19], v85, v61, vcc
	v_subrev_co_u32_e64 v90, s[18:19], s12, v84
	v_subb_co_u32_e32 v68, vcc, v77, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[22:23], 0, v85, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s13, v91
	v_subb_co_u32_e64 v85, s[18:19], v85, v61, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v90
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s13, v91
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v84
	v_cndmask_b32_e64 v92, v92, v93, s[22:23]
	v_subrev_co_u32_e64 v93, s[18:19], s12, v90
	s_nop 1
	v_subbrev_co_u32_e64 v85, s[18:19], 0, v85, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v92
	s_nop 1
	v_cndmask_b32_e64 v85, v91, v85, s[18:19]
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v77, v77, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v77
	v_cndmask_b32_e64 v77, v90, v93, s[18:19]
	v_and_b32_e32 v93, 0xff, v79
	v_cndmask_b32_e32 v77, v84, v77, vcc
	v_cndmask_b32_e32 v68, v68, v85, vcc
	v_xor_b32_e32 v77, v77, v76
	v_xor_b32_e32 v68, v68, v76
	v_sub_co_u32_e32 v77, vcc, v77, v76
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[130:133], v[4:7], v[12:15], v78, v93 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_subb_co_u32_e32 v68, vcc, v68, v76, vcc
	v_ashrrev_i32_e32 v68, 31, v68
	v_and_b32_e32 v68, s42, v68
	v_mul_lo_u32 v76, v98, s33
	v_add3_u32 v68, v68, v77, v76
	buffer_load_ubyte v98, v80, s[28:31], 0 offen
	buffer_load_ubyte v100, v81, s[28:31], 0 offen
	buffer_load_ubyte v99, v82, s[28:31], 0 offen
	buffer_load_ubyte v101, v83, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v102, v102, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v104, v103, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v103, v105, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v105, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v88
	v_lshl_add_u64 v[76:77], v[88:89], 0, s[84:85]
	s_nop 0
	v_subb_co_u32_e32 v79, vcc, -1, v89, vcc
	v_cmp_gt_i32_e32 vcc, 0, v77
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[122:125], v[4:7], v[0:3], v78, v93 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 v[132:135], v50 offset:4096
	ds_read_b128 v[128:131], v52 offset:6144
	ds_read_b128 v[124:127], v50 offset:6144
	ds_read_b128 v[136:139], v52 offset:4096
	ds_read_b32 v122, v97 offset:24832
	v_cndmask_b32_e32 v81, v77, v79, vcc
	v_ashrrev_i32_e32 v82, 31, v81
	v_cndmask_b32_e32 v80, v76, v68, vcc
	v_mov_b32_e32 v83, v82
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[82:83]
	v_xor_b32_e32 v83, v80, v82
	v_xor_b32_e32 v79, v81, v82
	v_mad_u64_u32 v[80:81], s[6:7], v83, s5, 0
	v_mul_hi_u32 v68, v83, s4
	v_lshl_add_u64 v[80:81], v[68:69], 0, v[80:81]
	v_mad_u64_u32 v[90:91], s[6:7], v79, s4, 0
	v_add_co_u32_e32 v68, vcc, v80, v90
	v_mad_u64_u32 v[84:85], s[6:7], v79, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v80, vcc, v81, v91, vcc
	v_mov_b32_e32 v81, s41
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[84:85]
	v_mul_lo_u32 v68, s13, v80
	v_mul_lo_u32 v81, s12, v81
	v_mad_u64_u32 v[84:85], s[6:7], s12, v80, 0
	v_add3_u32 v68, v85, v81, v68
	v_sub_co_u32_e32 v81, vcc, v83, v84
	s_nop 1
	v_subb_co_u32_e64 v83, s[18:19], v79, v68, vcc
	v_sub_u32_e32 v68, v79, v68
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	v_subrev_co_u32_e32 v79, vcc, s12, v81
	v_cmp_le_u32_e64 s[18:19], s13, v83
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cndmask_b32_e64 v84, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s12, v81
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v79
	v_cndmask_b32_e64 v85, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s13, v83
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	v_cndmask_b32_e64 v83, v84, v85, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v68, v81, v79, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v79, s64, v82
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v80, v68
	v_cmp_ne_u32_e32 vcc, 0, v83
	s_nop 1
	v_cndmask_b32_e32 v68, v80, v68, vcc
	v_ashrrev_i32_e32 v80, 31, v77
	v_mov_b32_e32 v81, v80
	v_xor_b32_e32 v68, v68, v79
	v_lshl_add_u64 v[76:77], v[76:77], 0, v[80:81]
	v_sub_u32_e32 v68, v68, v79
	v_xor_b32_e32 v90, v76, v80
	v_xad_u32 v79, v68, v80, v30
	v_xor_b32_e32 v81, v77, v80
	v_mad_u64_u32 v[76:77], s[6:7], v90, s5, 0
	v_mul_hi_u32 v68, v90, s4
	v_lshl_add_u64 v[76:77], v[68:69], 0, v[76:77]
	v_mad_u64_u32 v[84:85], s[6:7], v81, s4, 0
	v_add_co_u32_e32 v68, vcc, v76, v84
	v_mad_u64_u32 v[82:83], s[6:7], v81, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v76, vcc, v77, v85, vcc
	v_mov_b32_e32 v77, s41
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[76:77], v[76:77], 0, v[82:83]
	v_mul_lo_u32 v68, s13, v76
	v_mul_lo_u32 v82, s12, v77
	v_mad_u64_u32 v[76:77], s[6:7], s12, v76, 0
	v_add3_u32 v68, v77, v82, v68
	v_sub_u32_e32 v77, v81, v68
	v_sub_co_u32_e32 v76, vcc, v90, v76
	s_mov_b64 s[6:7], 0x400
	s_nop 0
	v_subb_co_u32_e64 v77, s[18:19], v77, v61, vcc
	v_subrev_co_u32_e64 v82, s[18:19], s12, v76
	v_subb_co_u32_e32 v68, vcc, v81, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v83, s[22:23], 0, v77, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s13, v83
	v_subb_co_u32_e64 v77, s[18:19], v77, v61, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v84, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v82
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s13, v83
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v76
	v_cndmask_b32_e64 v84, v84, v85, s[22:23]
	v_subrev_co_u32_e64 v85, s[18:19], s12, v82
	s_nop 1
	v_subbrev_co_u32_e64 v77, s[18:19], 0, v77, s[18:19]
	v_cmp_ne_u32_e64 s[18:19], 0, v84
	s_nop 1
	v_cndmask_b32_e64 v77, v83, v77, s[18:19]
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v81, v81, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v81
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v77, vcc
	v_cndmask_b32_e64 v77, v82, v85, s[18:19]
	v_cndmask_b32_e32 v76, v76, v77, vcc
	v_xor_b32_e32 v76, v76, v80
	v_xor_b32_e32 v68, v68, v80
	v_sub_co_u32_e32 v76, vcc, v76, v80
	v_mul_lo_u32 v77, v79, s33
	s_nop 0
	v_subb_co_u32_e32 v68, vcc, v68, v80, vcc
	v_ashrrev_i32_e32 v68, 31, v68
	v_lshl_add_u64 v[80:81], v[86:87], 0, s[6:7]
	s_mov_b64 s[6:7], 0x401
	v_and_b32_e32 v68, s42, v68
	v_lshl_add_u64 v[82:83], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xfbfe
	v_add3_u32 v77, v68, v76, v77
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cmp_gt_i32_e64 s[18:19], 0, v81
	s_nop 0
	v_subb_co_u32_e32 v76, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v85, v83, v76, s[18:19]
	v_ashrrev_i32_e32 v90, 31, v85
	v_cndmask_b32_e64 v84, v82, v68, s[18:19]
	v_mov_b32_e32 v91, v90
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_xor_b32_e32 v79, v84, v90
	v_xor_b32_e32 v76, v85, v90
	v_mad_u64_u32 v[84:85], s[6:7], v79, s5, 0
	v_mul_hi_u32 v68, v79, s4
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[108:109], s[6:7], v76, s4, 0
	v_add_co_u32_e32 v68, vcc, v84, v108
	v_mad_u64_u32 v[106:107], s[6:7], v76, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v109, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[106:107]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v80, s12, v85
	v_mad_u64_u32 v[106:107], s[6:7], s12, v84, 0
	v_add3_u32 v68, v107, v80, v68
	v_sub_co_u32_e32 v79, vcc, v79, v106
	s_nop 1
	v_subb_co_u32_e64 v80, s[22:23], v76, v68, vcc
	v_sub_u32_e32 v68, v76, v68
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	v_subrev_co_u32_e32 v76, vcc, s12, v79
	v_cmp_le_u32_e64 s[22:23], s13, v80
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cndmask_b32_e64 v85, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v79
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v76
	v_cndmask_b32_e64 v91, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s13, v80
	v_cndmask_b32_e64 v76, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	v_cndmask_b32_e64 v80, v85, v91, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v68, v79, v76, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v76, s64, v90
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v84, v68
	v_cmp_ne_u32_e32 vcc, 0, v80
	v_ashrrev_i32_e32 v80, 31, v83
	s_nop 0
	v_cndmask_b32_e32 v68, v84, v68, vcc
	v_xor_b32_e32 v68, v68, v76
	v_sub_u32_e32 v68, v68, v76
	v_ashrrev_i32_e32 v76, 31, v81
	v_mov_b32_e32 v81, v80
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[80:81]
	v_xor_b32_e32 v81, v82, v80
	v_xad_u32 v92, v68, v76, v30
	v_xor_b32_e32 v79, v83, v80
	v_mad_u64_u32 v[82:83], s[6:7], v81, s5, 0
	v_mul_hi_u32 v68, v81, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[90:91], s[6:7], v79, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v90
	v_mad_u64_u32 v[84:85], s[6:7], v79, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v91, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v84, s12, v83
	v_mad_u64_u32 v[82:83], s[6:7], s12, v82, 0
	v_add3_u32 v68, v83, v84, v68
	v_sub_u32_e32 v83, v79, v68
	v_sub_co_u32_e32 v81, vcc, v81, v82
	s_mov_b64 s[6:7], 0x402
	s_nop 0
	v_subb_co_u32_e64 v82, s[22:23], v83, v61, vcc
	v_subrev_co_u32_e64 v83, s[22:23], s12, v81
	v_subb_co_u32_e32 v68, vcc, v79, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v84, s[24:25], 0, v82, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v84
	v_subb_co_u32_e64 v82, s[22:23], v82, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v83
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v90, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v84
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v81
	v_cndmask_b32_e64 v85, v85, v90, s[24:25]
	v_subrev_co_u32_e64 v90, s[22:23], s12, v83
	s_nop 1
	v_subbrev_co_u32_e64 v82, s[22:23], 0, v82, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v85
	s_nop 1
	v_cndmask_b32_e64 v82, v84, v82, s[22:23]
	v_cndmask_b32_e64 v84, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v79, v79, v84, vcc
	v_cmp_ne_u32_e32 vcc, 0, v79
	v_cndmask_b32_e64 v79, v83, v90, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v68, v68, v82, vcc
	v_cndmask_b32_e32 v79, v81, v79, vcc
	v_xor_b32_e32 v81, v68, v80
	v_xor_b32_e32 v68, v79, v80
	v_sub_co_u32_e32 v68, vcc, v68, v80
	s_nop 1
	v_subb_co_u32_e32 v79, vcc, v81, v80, vcc
	v_ashrrev_i32_e32 v79, 31, v79
	v_and_b32_e32 v79, s42, v79
	v_mul_lo_u32 v80, v92, s33
	v_add3_u32 v79, v79, v68, v80
	v_lshl_add_u64 v[80:81], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xfbfd
	v_sub_co_u32_e32 v68, vcc, s6, v86
	s_nop 1
	v_subb_co_u32_e32 v82, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v83, v81, v82, s[18:19]
	v_ashrrev_i32_e32 v84, 31, v83
	v_cndmask_b32_e64 v82, v80, v68, s[18:19]
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v92, v82, v84
	v_xor_b32_e32 v85, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v92, s5, 0
	v_mul_hi_u32 v68, v92, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[106:107], s[6:7], v85, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v106
	v_mad_u64_u32 v[90:91], s[6:7], v85, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v107, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v83, s12, v83
	v_mad_u64_u32 v[90:91], s[6:7], s12, v82, 0
	v_add3_u32 v68, v91, v83, v68
	v_sub_co_u32_e32 v83, vcc, v92, v90
	s_nop 1
	v_subb_co_u32_e64 v90, s[22:23], v85, v68, vcc
	v_sub_u32_e32 v68, v85, v68
	v_cmp_le_u32_e64 s[22:23], s13, v90
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v83
	v_subrev_co_u32_e32 v83, vcc, s12, v83
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cmp_eq_u32_e64 s[22:23], s13, v90
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v83
	v_cndmask_b32_e64 v90, v91, v92, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v85, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v82, v68
	v_cmp_ne_u32_e32 vcc, 0, v90
	s_nop 1
	v_cndmask_b32_e32 v68, v82, v68, vcc
	v_xor_b32_e32 v82, s64, v84
	v_xor_b32_e32 v68, v68, v82
	v_sub_u32_e32 v68, v68, v82
	v_ashrrev_i32_e32 v82, 31, v81
	v_mov_b32_e32 v83, v82
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[82:83]
	v_xor_b32_e32 v106, v80, v82
	v_xad_u32 v92, v68, v76, v30
	v_xor_b32_e32 v83, v81, v82
	v_mad_u64_u32 v[80:81], s[6:7], v106, s5, 0
	v_mul_hi_u32 v68, v106, s4
	v_lshl_add_u64 v[80:81], v[68:69], 0, v[80:81]
	v_mad_u64_u32 v[90:91], s[6:7], v83, s4, 0
	v_add_co_u32_e32 v68, vcc, v80, v90
	v_mad_u64_u32 v[84:85], s[6:7], v83, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v80, vcc, v81, v91, vcc
	v_mov_b32_e32 v81, s41
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[84:85]
	v_mul_lo_u32 v68, s13, v80
	v_mul_lo_u32 v84, s12, v81
	v_mad_u64_u32 v[80:81], s[6:7], s12, v80, 0
	v_add3_u32 v68, v81, v84, v68
	v_sub_u32_e32 v81, v83, v68
	v_sub_co_u32_e32 v80, vcc, v106, v80
	s_mov_b64 s[6:7], 0x403
	s_nop 0
	v_subb_co_u32_e64 v81, s[22:23], v81, v61, vcc
	v_subrev_co_u32_e64 v84, s[22:23], s12, v80
	v_subb_co_u32_e32 v68, vcc, v83, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[24:25], 0, v81, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v85
	v_subb_co_u32_e64 v81, s[22:23], v81, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v90, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v84
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v85
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v80
	v_cndmask_b32_e64 v90, v90, v91, s[24:25]
	v_subrev_co_u32_e64 v91, s[22:23], s12, v84
	s_nop 1
	v_subbrev_co_u32_e64 v81, s[22:23], 0, v81, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v90
	s_nop 1
	v_cndmask_b32_e64 v81, v85, v81, s[22:23]
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v81, vcc
	v_cndmask_b32_e64 v81, v84, v91, s[22:23]
	v_cndmask_b32_e32 v80, v80, v81, vcc
	v_xor_b32_e32 v81, v68, v82
	v_xor_b32_e32 v68, v80, v82
	v_sub_co_u32_e32 v68, vcc, v68, v82
	s_nop 1
	v_subb_co_u32_e32 v81, vcc, v81, v82, vcc
	v_ashrrev_i32_e32 v80, 31, v81
	v_and_b32_e32 v80, s42, v80
	v_mul_lo_u32 v81, v92, s33
	v_lshl_add_u64 v[82:83], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xfbfc
	v_add3_u32 v80, v80, v68, v81
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cndmask_b32_e64 v84, v82, v68, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v81, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v85, v83, v81, s[18:19]
	v_ashrrev_i32_e32 v90, 31, v85
	v_mov_b32_e32 v91, v90
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_xor_b32_e32 v91, v84, v90
	v_xor_b32_e32 v81, v85, v90
	v_mad_u64_u32 v[84:85], s[6:7], v91, s5, 0
	v_mul_hi_u32 v68, v91, s4
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[108:109], s[6:7], v81, s4, 0
	v_add_co_u32_e32 v68, vcc, v84, v108
	v_mad_u64_u32 v[106:107], s[6:7], v81, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v109, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[106:107]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v85, s12, v85
	v_mad_u64_u32 v[106:107], s[6:7], s12, v84, 0
	v_add3_u32 v68, v107, v85, v68
	v_sub_co_u32_e32 v85, vcc, v91, v106
	s_nop 1
	v_subb_co_u32_e64 v91, s[22:23], v81, v68, vcc
	v_sub_u32_e32 v68, v81, v68
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	v_subrev_co_u32_e32 v81, vcc, s12, v85
	v_cmp_le_u32_e64 s[22:23], s13, v91
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cndmask_b32_e64 v92, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v85
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v81
	v_cndmask_b32_e64 v106, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s13, v91
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	v_cndmask_b32_e64 v91, v92, v106, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v68, v85, v81, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v81, s64, v90
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v84, v68
	v_cmp_ne_u32_e32 vcc, 0, v91
	s_nop 1
	v_cndmask_b32_e32 v68, v84, v68, vcc
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_xor_b32_e32 v68, v68, v81
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_sub_u32_e32 v68, v68, v81
	v_xor_b32_e32 v85, v82, v84
	v_xad_u32 v92, v68, v76, v30
	v_xor_b32_e32 v81, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v85, s5, 0
	v_mul_hi_u32 v68, v85, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[106:107], s[6:7], v81, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v106
	v_mad_u64_u32 v[90:91], s[6:7], v81, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v107, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v90, s12, v83
	v_mad_u64_u32 v[82:83], s[6:7], s12, v82, 0
	v_add3_u32 v68, v83, v90, v68
	v_sub_u32_e32 v83, v81, v68
	v_sub_co_u32_e32 v82, vcc, v85, v82
	s_mov_b64 s[6:7], 0x404
	s_nop 0
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, vcc
	v_subrev_co_u32_e64 v85, s[22:23], s12, v82
	v_subb_co_u32_e32 v68, vcc, v81, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v90, s[24:25], 0, v83, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v90
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v85
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v106, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v90
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v82
	v_cndmask_b32_e64 v91, v91, v106, s[24:25]
	v_subrev_co_u32_e64 v106, s[22:23], s12, v85
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[22:23], 0, v83, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v91
	s_nop 1
	v_cndmask_b32_e64 v83, v90, v83, s[22:23]
	v_cndmask_b32_e64 v90, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v81, v81, v90, vcc
	v_cmp_ne_u32_e32 vcc, 0, v81
	v_cndmask_b32_e64 v81, v85, v106, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v68, v68, v83, vcc
	v_cndmask_b32_e32 v81, v82, v81, vcc
	v_xor_b32_e32 v82, v68, v84
	v_xor_b32_e32 v68, v81, v84
	v_sub_co_u32_e32 v68, vcc, v68, v84
	s_nop 1
	v_subb_co_u32_e32 v81, vcc, v82, v84, vcc
	v_ashrrev_i32_e32 v81, 31, v81
	v_and_b32_e32 v81, s42, v81
	v_mul_lo_u32 v82, v92, s33
	v_add3_u32 v81, v81, v68, v82
	v_lshl_add_u64 v[82:83], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xfbfb
	v_sub_co_u32_e32 v68, vcc, s6, v86
	s_nop 1
	v_subb_co_u32_e32 v84, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v85, v83, v84, s[18:19]
	v_ashrrev_i32_e32 v90, 31, v85
	v_cndmask_b32_e64 v84, v82, v68, s[18:19]
	v_mov_b32_e32 v91, v90
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_xor_b32_e32 v92, v84, v90
	v_xor_b32_e32 v91, v85, v90
	v_mad_u64_u32 v[84:85], s[6:7], v92, s5, 0
	v_mul_hi_u32 v68, v92, s4
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[108:109], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v84, v108
	v_mad_u64_u32 v[106:107], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v109, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[106:107]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v85, s12, v85
	v_mad_u64_u32 v[106:107], s[6:7], s12, v84, 0
	v_add3_u32 v68, v107, v85, v68
	v_sub_co_u32_e32 v85, vcc, v92, v106
	s_nop 1
	v_subb_co_u32_e64 v92, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_cmp_le_u32_e64 s[22:23], s13, v92
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v106, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v85
	v_subrev_co_u32_e32 v85, vcc, s12, v85
	s_nop 0
	v_cndmask_b32_e64 v107, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cmp_eq_u32_e64 s[22:23], s13, v92
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v85
	v_cndmask_b32_e64 v92, v106, v107, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v91, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v84, v68
	v_cmp_ne_u32_e32 vcc, 0, v92
	s_nop 1
	v_cndmask_b32_e32 v68, v84, v68, vcc
	v_xor_b32_e32 v84, s64, v90
	v_xor_b32_e32 v68, v68, v84
	v_sub_u32_e32 v68, v68, v84
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v108, v82, v84
	v_xad_u32 v92, v68, v76, v30
	v_xor_b32_e32 v85, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v108, s5, 0
	v_mul_hi_u32 v68, v108, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[106:107], s[6:7], v85, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v106
	v_mad_u64_u32 v[90:91], s[6:7], v85, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v107, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v90, s12, v83
	v_mad_u64_u32 v[82:83], s[6:7], s12, v82, 0
	v_add3_u32 v68, v83, v90, v68
	v_sub_u32_e32 v83, v85, v68
	v_sub_co_u32_e32 v82, vcc, v108, v82
	s_mov_b64 s[6:7], 0x405
	s_nop 0
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, vcc
	v_subrev_co_u32_e64 v90, s[22:23], s12, v82
	v_subb_co_u32_e32 v68, vcc, v85, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[24:25], 0, v83, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v91
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v106, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v90
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v107, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v91
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v82
	v_cndmask_b32_e64 v106, v106, v107, s[24:25]
	v_subrev_co_u32_e64 v107, s[22:23], s12, v90
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[22:23], 0, v83, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v106
	s_nop 1
	v_cndmask_b32_e64 v83, v91, v83, s[22:23]
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v85, v85, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v83, vcc
	v_cndmask_b32_e64 v83, v90, v107, s[22:23]
	v_cndmask_b32_e32 v82, v82, v83, vcc
	v_xor_b32_e32 v83, v68, v84
	v_xor_b32_e32 v68, v82, v84
	v_sub_co_u32_e32 v68, vcc, v68, v84
	s_nop 1
	v_subb_co_u32_e32 v83, vcc, v83, v84, vcc
	v_ashrrev_i32_e32 v82, 31, v83
	v_and_b32_e32 v82, s42, v82
	v_mul_lo_u32 v83, v92, s33
	v_add3_u32 v92, v82, v68, v83
	v_lshl_add_u64 v[82:83], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xfbfa
	v_sub_co_u32_e32 v68, vcc, s6, v86
	s_nop 1
	v_subb_co_u32_e32 v84, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v85, v83, v84, s[18:19]
	v_ashrrev_i32_e32 v90, 31, v85
	v_cndmask_b32_e64 v84, v82, v68, s[18:19]
	v_mov_b32_e32 v91, v90
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_xor_b32_e32 v110, v84, v90
	v_xor_b32_e32 v91, v85, v90
	v_mad_u64_u32 v[84:85], s[6:7], v110, s5, 0
	v_mul_hi_u32 v68, v110, s4
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[108:109], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v84, v108
	v_mad_u64_u32 v[106:107], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v109, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[106:107]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v85, s12, v85
	v_mad_u64_u32 v[106:107], s[6:7], s12, v84, 0
	v_add3_u32 v68, v107, v85, v68
	v_sub_co_u32_e32 v85, vcc, v110, v106
	s_nop 1
	v_subb_co_u32_e64 v106, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_cmp_le_u32_e64 s[22:23], s13, v106
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v107, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v85
	v_subrev_co_u32_e32 v85, vcc, s12, v85
	s_nop 0
	v_cndmask_b32_e64 v108, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cmp_eq_u32_e64 s[22:23], s13, v106
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v85
	v_cndmask_b32_e64 v106, v107, v108, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v91, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v84, v68
	v_cmp_ne_u32_e32 vcc, 0, v106
	s_nop 1
	v_cndmask_b32_e32 v68, v84, v68, vcc
	v_xor_b32_e32 v84, s64, v90
	v_xor_b32_e32 v68, v68, v84
	v_sub_u32_e32 v68, v68, v84
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v109, v82, v84
	v_xad_u32 v108, v68, v76, v30
	v_xor_b32_e32 v85, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v109, s5, 0
	v_mul_hi_u32 v68, v109, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[106:107], s[6:7], v85, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v106
	v_mad_u64_u32 v[90:91], s[6:7], v85, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v107, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v90, s12, v83
	v_mad_u64_u32 v[82:83], s[6:7], s12, v82, 0
	v_add3_u32 v68, v83, v90, v68
	v_sub_u32_e32 v83, v85, v68
	v_sub_co_u32_e32 v82, vcc, v109, v82
	s_mov_b64 s[6:7], 0x406
	s_nop 0
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, vcc
	v_subrev_co_u32_e64 v90, s[22:23], s12, v82
	v_subb_co_u32_e32 v68, vcc, v85, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[24:25], 0, v83, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v91
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v106, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v90
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v107, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v91
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v82
	v_cndmask_b32_e64 v106, v106, v107, s[24:25]
	v_subrev_co_u32_e64 v107, s[22:23], s12, v90
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[22:23], 0, v83, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v106
	s_nop 1
	v_cndmask_b32_e64 v83, v91, v83, s[22:23]
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v85, v85, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v83, vcc
	v_cndmask_b32_e64 v83, v90, v107, s[22:23]
	v_cndmask_b32_e32 v82, v82, v83, vcc
	v_xor_b32_e32 v83, v68, v84
	v_xor_b32_e32 v68, v82, v84
	v_sub_co_u32_e32 v68, vcc, v68, v84
	s_nop 1
	v_subb_co_u32_e32 v83, vcc, v83, v84, vcc
	v_ashrrev_i32_e32 v82, 31, v83
	v_and_b32_e32 v82, s42, v82
	v_mul_lo_u32 v83, v108, s33
	v_add3_u32 v111, v82, v68, v83
	v_lshl_add_u64 v[82:83], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xfbf9
	v_sub_co_u32_e32 v68, vcc, s6, v86
	s_nop 1
	v_subb_co_u32_e32 v84, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v85, v83, v84, s[18:19]
	v_ashrrev_i32_e32 v90, 31, v85
	v_cndmask_b32_e64 v84, v82, v68, s[18:19]
	v_mov_b32_e32 v91, v90
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_xor_b32_e32 v110, v84, v90
	v_xor_b32_e32 v91, v85, v90
	v_mad_u64_u32 v[84:85], s[6:7], v110, s5, 0
	v_mul_hi_u32 v68, v110, s4
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[108:109], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v84, v108
	v_mad_u64_u32 v[106:107], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v109, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[106:107]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v85, s12, v85
	v_mad_u64_u32 v[106:107], s[6:7], s12, v84, 0
	v_add3_u32 v68, v107, v85, v68
	v_sub_co_u32_e32 v85, vcc, v110, v106
	s_nop 1
	v_subb_co_u32_e64 v106, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_cmp_le_u32_e64 s[22:23], s13, v106
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v107, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v85
	v_subrev_co_u32_e32 v85, vcc, s12, v85
	s_nop 0
	v_cndmask_b32_e64 v108, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cmp_eq_u32_e64 s[22:23], s13, v106
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v85
	v_cndmask_b32_e64 v106, v107, v108, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v91, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v84, v68
	v_cmp_ne_u32_e32 vcc, 0, v106
	s_nop 1
	v_cndmask_b32_e32 v68, v84, v68, vcc
	v_xor_b32_e32 v84, s64, v90
	v_xor_b32_e32 v68, v68, v84
	v_sub_u32_e32 v68, v68, v84
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v109, v82, v84
	v_xad_u32 v108, v68, v76, v30
	v_xor_b32_e32 v85, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v109, s5, 0
	v_mul_hi_u32 v68, v109, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[106:107], s[6:7], v85, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v106
	v_mad_u64_u32 v[90:91], s[6:7], v85, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v107, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v90, s12, v83
	v_mad_u64_u32 v[82:83], s[6:7], s12, v82, 0
	v_add3_u32 v68, v83, v90, v68
	v_sub_u32_e32 v83, v85, v68
	v_sub_co_u32_e32 v82, vcc, v109, v82
	s_mov_b64 s[6:7], 0x407
	s_nop 0
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, vcc
	v_subrev_co_u32_e64 v90, s[22:23], s12, v82
	v_subb_co_u32_e32 v68, vcc, v85, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[24:25], 0, v83, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v91
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v106, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v90
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v107, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v91
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v82
	v_cndmask_b32_e64 v106, v106, v107, s[24:25]
	v_subrev_co_u32_e64 v107, s[22:23], s12, v90
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[22:23], 0, v83, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v106
	s_nop 1
	v_cndmask_b32_e64 v83, v91, v83, s[22:23]
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v85, v85, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v83, vcc
	v_cndmask_b32_e64 v83, v90, v107, s[22:23]
	v_cndmask_b32_e32 v82, v82, v83, vcc
	v_xor_b32_e32 v83, v68, v84
	v_xor_b32_e32 v68, v82, v84
	v_sub_co_u32_e32 v68, vcc, v68, v84
	s_nop 1
	v_subb_co_u32_e32 v83, vcc, v83, v84, vcc
	v_ashrrev_i32_e32 v82, 31, v83
	v_and_b32_e32 v82, s42, v82
	v_mul_lo_u32 v83, v108, s33
	v_add3_u32 v113, v82, v68, v83
	v_lshl_add_u64 v[82:83], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xfbf8
	v_sub_co_u32_e32 v68, vcc, s6, v86
	s_nop 1
	v_subb_co_u32_e32 v84, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v85, v83, v84, s[18:19]
	v_ashrrev_i32_e32 v90, 31, v85
	v_cndmask_b32_e64 v84, v82, v68, s[18:19]
	v_mov_b32_e32 v91, v90
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_xor_b32_e32 v110, v84, v90
	v_xor_b32_e32 v91, v85, v90
	v_mad_u64_u32 v[84:85], s[6:7], v110, s5, 0
	v_mul_hi_u32 v68, v110, s4
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[108:109], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v84, v108
	v_mad_u64_u32 v[106:107], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v109, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[106:107]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v85, s12, v85
	v_mad_u64_u32 v[106:107], s[6:7], s12, v84, 0
	v_add3_u32 v68, v107, v85, v68
	v_sub_co_u32_e32 v85, vcc, v110, v106
	s_nop 1
	v_subb_co_u32_e64 v106, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_cmp_le_u32_e64 s[22:23], s13, v106
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v107, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v85
	v_subrev_co_u32_e32 v85, vcc, s12, v85
	s_nop 0
	v_cndmask_b32_e64 v108, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cmp_eq_u32_e64 s[22:23], s13, v106
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v85
	v_cndmask_b32_e64 v106, v107, v108, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v91, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v84, v68
	v_cmp_ne_u32_e32 vcc, 0, v106
	s_nop 1
	v_cndmask_b32_e32 v68, v84, v68, vcc
	v_xor_b32_e32 v84, s64, v90
	v_xor_b32_e32 v68, v68, v84
	v_sub_u32_e32 v68, v68, v84
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v109, v82, v84
	v_xad_u32 v108, v68, v76, v30
	v_xor_b32_e32 v85, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v109, s5, 0
	v_mul_hi_u32 v68, v109, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[106:107], s[6:7], v85, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v106
	v_mad_u64_u32 v[90:91], s[6:7], v85, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v107, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v90, s12, v83
	v_mad_u64_u32 v[82:83], s[6:7], s12, v82, 0
	v_add3_u32 v68, v83, v90, v68
	v_sub_u32_e32 v83, v85, v68
	v_sub_co_u32_e32 v82, vcc, v109, v82
	s_mov_b64 s[6:7], 0x408
	s_nop 0
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, vcc
	v_subrev_co_u32_e64 v90, s[22:23], s12, v82
	v_subb_co_u32_e32 v68, vcc, v85, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[24:25], 0, v83, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v91
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v106, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v90
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v107, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v91
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v82
	v_cndmask_b32_e64 v106, v106, v107, s[24:25]
	v_subrev_co_u32_e64 v107, s[22:23], s12, v90
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[22:23], 0, v83, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v106
	s_nop 1
	v_cndmask_b32_e64 v83, v91, v83, s[22:23]
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v85, v85, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v83, vcc
	v_cndmask_b32_e64 v83, v90, v107, s[22:23]
	v_cndmask_b32_e32 v82, v82, v83, vcc
	v_xor_b32_e32 v83, v68, v84
	v_xor_b32_e32 v68, v82, v84
	v_sub_co_u32_e32 v68, vcc, v68, v84
	s_nop 1
	v_subb_co_u32_e32 v83, vcc, v83, v84, vcc
	v_ashrrev_i32_e32 v82, 31, v83
	v_and_b32_e32 v82, s42, v82
	v_mul_lo_u32 v83, v108, s33
	v_add3_u32 v68, v82, v68, v83
	buffer_load_ubyte v106, v77, s[28:31], 0 offen
	buffer_load_ubyte v108, v79, s[28:31], 0 offen
	buffer_load_ubyte v107, v80, s[28:31], 0 offen
	buffer_load_ubyte v109, v81, s[28:31], 0 offen
	buffer_load_ubyte v110, v92, s[28:31], 0 offen
	buffer_load_ubyte v112, v111, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v111, v113, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v113, v68, s[28:31], 0 offen
	v_lshl_add_u64 v[80:81], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xfbf7
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cndmask_b32_e64 v82, v80, v68, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v77, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v83, v81, v77, s[18:19]
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v79, v82, v84
	v_xor_b32_e32 v77, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v79, s5, 0
	v_mul_hi_u32 v68, v79, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[114:115], s[6:7], v77, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v114
	v_mad_u64_u32 v[90:91], s[6:7], v77, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v115, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v83, s12, v83
	v_mad_u64_u32 v[90:91], s[6:7], s12, v82, 0
	v_add3_u32 v68, v91, v83, v68
	v_sub_co_u32_e32 v79, vcc, v79, v90
	s_nop 1
	v_subb_co_u32_e64 v83, s[22:23], v77, v68, vcc
	v_sub_u32_e32 v68, v77, v68
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	v_subrev_co_u32_e32 v77, vcc, s12, v79
	v_cmp_le_u32_e64 s[22:23], s13, v83
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cndmask_b32_e64 v85, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v79
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v77
	v_cndmask_b32_e64 v90, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s13, v83
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	v_cndmask_b32_e64 v83, v85, v90, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v68, v79, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v77, s64, v84
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v82, v68
	v_cmp_ne_u32_e32 vcc, 0, v83
	s_nop 1
	v_cndmask_b32_e32 v68, v82, v68, vcc
	v_ashrrev_i32_e32 v82, 31, v81
	v_mov_b32_e32 v83, v82
	v_xor_b32_e32 v68, v68, v77
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[82:83]
	v_sub_u32_e32 v68, v68, v77
	v_xor_b32_e32 v83, v80, v82
	v_xad_u32 v79, v68, v76, v30
	v_xor_b32_e32 v77, v81, v82
	v_mad_u64_u32 v[80:81], s[6:7], v83, s5, 0
	v_mul_hi_u32 v68, v83, s4
	v_lshl_add_u64 v[80:81], v[68:69], 0, v[80:81]
	v_mad_u64_u32 v[90:91], s[6:7], v77, s4, 0
	v_add_co_u32_e32 v68, vcc, v80, v90
	v_mad_u64_u32 v[84:85], s[6:7], v77, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v80, vcc, v81, v91, vcc
	v_mov_b32_e32 v81, s41
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[84:85]
	v_mul_lo_u32 v68, s13, v80
	v_mul_lo_u32 v84, s12, v81
	v_mad_u64_u32 v[80:81], s[6:7], s12, v80, 0
	v_add3_u32 v68, v81, v84, v68
	v_sub_u32_e32 v81, v77, v68
	v_sub_co_u32_e32 v80, vcc, v83, v80
	s_mov_b64 s[6:7], 0x409
	s_nop 0
	v_subb_co_u32_e64 v81, s[22:23], v81, v61, vcc
	v_subrev_co_u32_e64 v83, s[22:23], s12, v80
	v_subb_co_u32_e32 v68, vcc, v77, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v84, s[24:25], 0, v81, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v84
	v_subb_co_u32_e64 v81, s[22:23], v81, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v83
	v_cmp_le_u32_e32 vcc, s13, v68
	v_mul_lo_u32 v79, v79, s33
	v_cndmask_b32_e64 v90, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v84
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v80
	v_cndmask_b32_e64 v85, v85, v90, s[24:25]
	v_subrev_co_u32_e64 v90, s[22:23], s12, v83
	s_nop 1
	v_subbrev_co_u32_e64 v81, s[22:23], 0, v81, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v85
	s_nop 1
	v_cndmask_b32_e64 v81, v84, v81, s[22:23]
	v_cndmask_b32_e64 v84, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v77, v77, v84, vcc
	v_cmp_ne_u32_e32 vcc, 0, v77
	v_cndmask_b32_e64 v77, v83, v90, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v68, v68, v81, vcc
	v_cndmask_b32_e32 v77, v80, v77, vcc
	v_xor_b32_e32 v80, v68, v82
	v_xor_b32_e32 v68, v77, v82
	v_sub_co_u32_e32 v68, vcc, v68, v82
	s_nop 1
	v_subb_co_u32_e32 v77, vcc, v80, v82, vcc
	v_ashrrev_i32_e32 v77, 31, v77
	v_and_b32_e32 v77, s42, v77
	v_lshl_add_u64 v[80:81], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xfbf6
	v_add3_u32 v77, v77, v68, v79
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cndmask_b32_e64 v82, v80, v68, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v79, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v83, v81, v79, s[18:19]
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v85, v82, v84
	v_xor_b32_e32 v79, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v85, s5, 0
	v_mul_hi_u32 v68, v85, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[114:115], s[6:7], v79, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v114
	v_mad_u64_u32 v[90:91], s[6:7], v79, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v115, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v83, s12, v83
	v_mad_u64_u32 v[90:91], s[6:7], s12, v82, 0
	v_add3_u32 v68, v91, v83, v68
	v_sub_co_u32_e32 v83, vcc, v85, v90
	s_nop 1
	v_subb_co_u32_e64 v85, s[22:23], v79, v68, vcc
	v_sub_u32_e32 v68, v79, v68
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	v_subrev_co_u32_e32 v79, vcc, s12, v83
	v_cmp_le_u32_e64 s[22:23], s13, v85
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cndmask_b32_e64 v90, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v83
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v79
	v_cndmask_b32_e64 v91, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s13, v85
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	v_cndmask_b32_e64 v85, v90, v91, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v68, v83, v79, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v79, s64, v84
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v82, v68
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v68, v82, v68, vcc
	v_ashrrev_i32_e32 v82, 31, v81
	v_mov_b32_e32 v83, v82
	v_xor_b32_e32 v68, v68, v79
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[82:83]
	v_sub_u32_e32 v68, v68, v79
	v_xor_b32_e32 v83, v80, v82
	v_xad_u32 v92, v68, v76, v30
	v_xor_b32_e32 v79, v81, v82
	v_mad_u64_u32 v[80:81], s[6:7], v83, s5, 0
	v_mul_hi_u32 v68, v83, s4
	v_lshl_add_u64 v[80:81], v[68:69], 0, v[80:81]
	v_mad_u64_u32 v[90:91], s[6:7], v79, s4, 0
	v_add_co_u32_e32 v68, vcc, v80, v90
	v_mad_u64_u32 v[84:85], s[6:7], v79, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v80, vcc, v81, v91, vcc
	v_mov_b32_e32 v81, s41
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[84:85]
	v_mul_lo_u32 v68, s13, v80
	v_mul_lo_u32 v84, s12, v81
	v_mad_u64_u32 v[80:81], s[6:7], s12, v80, 0
	v_add3_u32 v68, v81, v84, v68
	v_sub_u32_e32 v81, v79, v68
	v_sub_co_u32_e32 v80, vcc, v83, v80
	s_mov_b64 s[6:7], 0x40a
	s_nop 0
	v_subb_co_u32_e64 v81, s[22:23], v81, v61, vcc
	v_subrev_co_u32_e64 v83, s[22:23], s12, v80
	v_subb_co_u32_e32 v68, vcc, v79, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v84, s[24:25], 0, v81, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v84
	v_subb_co_u32_e64 v81, s[22:23], v81, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v83
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v90, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v84
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v80
	v_cndmask_b32_e64 v85, v85, v90, s[24:25]
	v_subrev_co_u32_e64 v90, s[22:23], s12, v83
	s_nop 1
	v_subbrev_co_u32_e64 v81, s[22:23], 0, v81, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v85
	s_nop 1
	v_cndmask_b32_e64 v81, v84, v81, s[22:23]
	v_cndmask_b32_e64 v84, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v79, v79, v84, vcc
	v_cmp_ne_u32_e32 vcc, 0, v79
	v_cndmask_b32_e64 v79, v83, v90, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v68, v68, v81, vcc
	v_cndmask_b32_e32 v79, v80, v79, vcc
	v_xor_b32_e32 v80, v68, v82
	v_xor_b32_e32 v68, v79, v82
	v_sub_co_u32_e32 v68, vcc, v68, v82
	s_nop 1
	v_subb_co_u32_e32 v79, vcc, v80, v82, vcc
	v_ashrrev_i32_e32 v79, 31, v79
	v_and_b32_e32 v79, s42, v79
	v_mul_lo_u32 v80, v92, s33
	v_add3_u32 v79, v79, v68, v80
	v_lshl_add_u64 v[80:81], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xfbf5
	v_sub_co_u32_e32 v68, vcc, s6, v86
	s_nop 1
	v_subb_co_u32_e32 v82, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v83, v81, v82, s[18:19]
	v_ashrrev_i32_e32 v84, 31, v83
	v_cndmask_b32_e64 v82, v80, v68, s[18:19]
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v92, v82, v84
	v_xor_b32_e32 v85, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v92, s5, 0
	v_mul_hi_u32 v68, v92, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[114:115], s[6:7], v85, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v114
	v_mad_u64_u32 v[90:91], s[6:7], v85, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v115, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v83, s12, v83
	v_mad_u64_u32 v[90:91], s[6:7], s12, v82, 0
	v_add3_u32 v68, v91, v83, v68
	v_sub_co_u32_e32 v83, vcc, v92, v90
	s_nop 1
	v_subb_co_u32_e64 v90, s[22:23], v85, v68, vcc
	v_sub_u32_e32 v68, v85, v68
	v_cmp_le_u32_e64 s[22:23], s13, v90
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v83
	v_subrev_co_u32_e32 v83, vcc, s12, v83
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cmp_eq_u32_e64 s[22:23], s13, v90
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v83
	v_cndmask_b32_e64 v90, v91, v92, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v85, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v82, v68
	v_cmp_ne_u32_e32 vcc, 0, v90
	s_nop 1
	v_cndmask_b32_e32 v68, v82, v68, vcc
	v_xor_b32_e32 v82, s64, v84
	v_xor_b32_e32 v68, v68, v82
	v_sub_u32_e32 v68, v68, v82
	v_ashrrev_i32_e32 v82, 31, v81
	v_mov_b32_e32 v83, v82
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[82:83]
	v_xor_b32_e32 v114, v80, v82
	v_xad_u32 v92, v68, v76, v30
	v_xor_b32_e32 v83, v81, v82
	v_mad_u64_u32 v[80:81], s[6:7], v114, s5, 0
	v_mul_hi_u32 v68, v114, s4
	v_lshl_add_u64 v[80:81], v[68:69], 0, v[80:81]
	v_mad_u64_u32 v[90:91], s[6:7], v83, s4, 0
	v_add_co_u32_e32 v68, vcc, v80, v90
	v_mad_u64_u32 v[84:85], s[6:7], v83, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v80, vcc, v81, v91, vcc
	v_mov_b32_e32 v81, s41
	s_nop 0
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[80:81], v[80:81], 0, v[84:85]
	v_mul_lo_u32 v68, s13, v80
	v_mul_lo_u32 v84, s12, v81
	v_mad_u64_u32 v[80:81], s[6:7], s12, v80, 0
	v_add3_u32 v68, v81, v84, v68
	v_sub_u32_e32 v81, v83, v68
	v_sub_co_u32_e32 v80, vcc, v114, v80
	s_mov_b64 s[6:7], 0x40b
	s_nop 0
	v_subb_co_u32_e64 v81, s[22:23], v81, v61, vcc
	v_subrev_co_u32_e64 v84, s[22:23], s12, v80
	v_subb_co_u32_e32 v68, vcc, v83, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[24:25], 0, v81, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v85
	v_subb_co_u32_e64 v81, s[22:23], v81, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v90, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v84
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v85
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v80
	v_cndmask_b32_e64 v90, v90, v91, s[24:25]
	v_subrev_co_u32_e64 v91, s[22:23], s12, v84
	s_nop 1
	v_subbrev_co_u32_e64 v81, s[22:23], 0, v81, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v90
	s_nop 1
	v_cndmask_b32_e64 v81, v85, v81, s[22:23]
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v81, vcc
	v_cndmask_b32_e64 v81, v84, v91, s[22:23]
	v_cndmask_b32_e32 v80, v80, v81, vcc
	v_xor_b32_e32 v81, v68, v82
	v_xor_b32_e32 v68, v80, v82
	v_sub_co_u32_e32 v68, vcc, v68, v82
	s_nop 1
	v_subb_co_u32_e32 v81, vcc, v81, v82, vcc
	v_ashrrev_i32_e32 v80, 31, v81
	v_and_b32_e32 v80, s42, v80
	v_mul_lo_u32 v81, v92, s33
	v_lshl_add_u64 v[82:83], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xfbf4
	v_add3_u32 v80, v80, v68, v81
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cndmask_b32_e64 v84, v82, v68, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v81, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v85, v83, v81, s[18:19]
	v_ashrrev_i32_e32 v90, 31, v85
	v_mov_b32_e32 v91, v90
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_xor_b32_e32 v91, v84, v90
	v_xor_b32_e32 v81, v85, v90
	v_mad_u64_u32 v[84:85], s[6:7], v91, s5, 0
	v_mul_hi_u32 v68, v91, s4
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[116:117], s[6:7], v81, s4, 0
	v_add_co_u32_e32 v68, vcc, v84, v116
	v_mad_u64_u32 v[114:115], s[6:7], v81, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v117, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[114:115]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v85, s12, v85
	v_mad_u64_u32 v[114:115], s[6:7], s12, v84, 0
	v_add3_u32 v68, v115, v85, v68
	v_sub_co_u32_e32 v85, vcc, v91, v114
	s_nop 1
	v_subb_co_u32_e64 v91, s[22:23], v81, v68, vcc
	v_sub_u32_e32 v68, v81, v68
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	v_subrev_co_u32_e32 v81, vcc, s12, v85
	v_cmp_le_u32_e64 s[22:23], s13, v91
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cndmask_b32_e64 v92, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v85
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v81
	v_cndmask_b32_e64 v114, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s13, v91
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	v_cndmask_b32_e64 v91, v92, v114, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v68, v85, v81, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v81, s64, v90
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v84, v68
	v_cmp_ne_u32_e32 vcc, 0, v91
	s_nop 1
	v_cndmask_b32_e32 v68, v84, v68, vcc
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_xor_b32_e32 v68, v68, v81
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_sub_u32_e32 v68, v68, v81
	v_xor_b32_e32 v85, v82, v84
	v_xad_u32 v92, v68, v76, v30
	v_xor_b32_e32 v81, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v85, s5, 0
	v_mul_hi_u32 v68, v85, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[114:115], s[6:7], v81, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v114
	v_mad_u64_u32 v[90:91], s[6:7], v81, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v115, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v90, s12, v83
	v_mad_u64_u32 v[82:83], s[6:7], s12, v82, 0
	v_add3_u32 v68, v83, v90, v68
	v_sub_u32_e32 v83, v81, v68
	v_sub_co_u32_e32 v82, vcc, v85, v82
	s_mov_b64 s[6:7], 0x40c
	s_nop 0
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, vcc
	v_subrev_co_u32_e64 v85, s[22:23], s12, v82
	v_subb_co_u32_e32 v68, vcc, v81, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v90, s[24:25], 0, v83, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v90
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v85
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v114, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v90
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v82
	v_cndmask_b32_e64 v91, v91, v114, s[24:25]
	v_subrev_co_u32_e64 v114, s[22:23], s12, v85
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[22:23], 0, v83, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v91
	s_nop 1
	v_cndmask_b32_e64 v83, v90, v83, s[22:23]
	v_cndmask_b32_e64 v90, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v81, v81, v90, vcc
	v_cmp_ne_u32_e32 vcc, 0, v81
	v_cndmask_b32_e64 v81, v85, v114, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v68, v68, v83, vcc
	v_cndmask_b32_e32 v81, v82, v81, vcc
	v_xor_b32_e32 v82, v68, v84
	v_xor_b32_e32 v68, v81, v84
	v_sub_co_u32_e32 v68, vcc, v68, v84
	s_nop 1
	v_subb_co_u32_e32 v81, vcc, v82, v84, vcc
	v_ashrrev_i32_e32 v81, 31, v81
	v_and_b32_e32 v81, s42, v81
	v_mul_lo_u32 v82, v92, s33
	v_add3_u32 v81, v81, v68, v82
	v_lshl_add_u64 v[82:83], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xfbf3
	v_sub_co_u32_e32 v68, vcc, s6, v86
	s_nop 1
	v_subb_co_u32_e32 v84, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v85, v83, v84, s[18:19]
	v_ashrrev_i32_e32 v90, 31, v85
	v_cndmask_b32_e64 v84, v82, v68, s[18:19]
	v_mov_b32_e32 v91, v90
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_xor_b32_e32 v92, v84, v90
	v_xor_b32_e32 v91, v85, v90
	v_mad_u64_u32 v[84:85], s[6:7], v92, s5, 0
	v_mul_hi_u32 v68, v92, s4
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[116:117], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v84, v116
	v_mad_u64_u32 v[114:115], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v117, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[114:115]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v85, s12, v85
	v_mad_u64_u32 v[114:115], s[6:7], s12, v84, 0
	v_add3_u32 v68, v115, v85, v68
	v_sub_co_u32_e32 v85, vcc, v92, v114
	s_nop 1
	v_subb_co_u32_e64 v92, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_cmp_le_u32_e64 s[22:23], s13, v92
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v114, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v85
	v_subrev_co_u32_e32 v85, vcc, s12, v85
	s_nop 0
	v_cndmask_b32_e64 v115, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cmp_eq_u32_e64 s[22:23], s13, v92
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v85
	v_cndmask_b32_e64 v92, v114, v115, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v91, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v84, v68
	v_cmp_ne_u32_e32 vcc, 0, v92
	s_nop 1
	v_cndmask_b32_e32 v68, v84, v68, vcc
	v_xor_b32_e32 v84, s64, v90
	v_xor_b32_e32 v68, v68, v84
	v_sub_u32_e32 v68, v68, v84
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v116, v82, v84
	v_xad_u32 v92, v68, v76, v30
	v_xor_b32_e32 v85, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v116, s5, 0
	v_mul_hi_u32 v68, v116, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[114:115], s[6:7], v85, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v114
	v_mad_u64_u32 v[90:91], s[6:7], v85, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v115, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v90, s12, v83
	v_mad_u64_u32 v[82:83], s[6:7], s12, v82, 0
	v_add3_u32 v68, v83, v90, v68
	v_sub_u32_e32 v83, v85, v68
	v_sub_co_u32_e32 v82, vcc, v116, v82
	s_mov_b64 s[6:7], 0x40d
	s_nop 0
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, vcc
	v_subrev_co_u32_e64 v90, s[22:23], s12, v82
	v_subb_co_u32_e32 v68, vcc, v85, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[24:25], 0, v83, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v91
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v114, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v90
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v115, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v91
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v82
	v_cndmask_b32_e64 v114, v114, v115, s[24:25]
	v_subrev_co_u32_e64 v115, s[22:23], s12, v90
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[22:23], 0, v83, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v114
	s_nop 1
	v_cndmask_b32_e64 v83, v91, v83, s[22:23]
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v85, v85, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v83, vcc
	v_cndmask_b32_e64 v83, v90, v115, s[22:23]
	v_cndmask_b32_e32 v82, v82, v83, vcc
	v_xor_b32_e32 v83, v68, v84
	v_xor_b32_e32 v68, v82, v84
	v_sub_co_u32_e32 v68, vcc, v68, v84
	s_nop 1
	v_subb_co_u32_e32 v83, vcc, v83, v84, vcc
	v_ashrrev_i32_e32 v82, 31, v83
	v_and_b32_e32 v82, s42, v82
	v_mul_lo_u32 v83, v92, s33
	v_add3_u32 v92, v82, v68, v83
	v_lshl_add_u64 v[82:83], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xfbf2
	v_sub_co_u32_e32 v68, vcc, s6, v86
	s_nop 1
	v_subb_co_u32_e32 v84, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v85, v83, v84, s[18:19]
	v_ashrrev_i32_e32 v90, 31, v85
	v_cndmask_b32_e64 v84, v82, v68, s[18:19]
	v_mov_b32_e32 v91, v90
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_xor_b32_e32 v118, v84, v90
	v_xor_b32_e32 v91, v85, v90
	v_mad_u64_u32 v[84:85], s[6:7], v118, s5, 0
	v_mul_hi_u32 v68, v118, s4
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[116:117], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v84, v116
	v_mad_u64_u32 v[114:115], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v117, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[114:115]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v85, s12, v85
	v_mad_u64_u32 v[114:115], s[6:7], s12, v84, 0
	v_add3_u32 v68, v115, v85, v68
	v_sub_co_u32_e32 v85, vcc, v118, v114
	s_nop 1
	v_subb_co_u32_e64 v114, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_cmp_le_u32_e64 s[22:23], s13, v114
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v115, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v85
	v_subrev_co_u32_e32 v85, vcc, s12, v85
	s_nop 0
	v_cndmask_b32_e64 v116, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cmp_eq_u32_e64 s[22:23], s13, v114
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v85
	v_cndmask_b32_e64 v114, v115, v116, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v91, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v84, v68
	v_cmp_ne_u32_e32 vcc, 0, v114
	s_nop 1
	v_cndmask_b32_e32 v68, v84, v68, vcc
	v_xor_b32_e32 v84, s64, v90
	v_xor_b32_e32 v68, v68, v84
	v_sub_u32_e32 v68, v68, v84
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v117, v82, v84
	v_xad_u32 v116, v68, v76, v30
	v_xor_b32_e32 v85, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v117, s5, 0
	v_mul_hi_u32 v68, v117, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[114:115], s[6:7], v85, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v114
	v_mad_u64_u32 v[90:91], s[6:7], v85, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v115, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v90, s12, v83
	v_mad_u64_u32 v[82:83], s[6:7], s12, v82, 0
	v_add3_u32 v68, v83, v90, v68
	v_sub_u32_e32 v83, v85, v68
	v_sub_co_u32_e32 v82, vcc, v117, v82
	s_mov_b64 s[6:7], 0x40e
	s_nop 0
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, vcc
	v_subrev_co_u32_e64 v90, s[22:23], s12, v82
	v_subb_co_u32_e32 v68, vcc, v85, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[24:25], 0, v83, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v91
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v114, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v90
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v115, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v91
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v82
	v_cndmask_b32_e64 v114, v114, v115, s[24:25]
	v_subrev_co_u32_e64 v115, s[22:23], s12, v90
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[22:23], 0, v83, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v114
	s_nop 1
	v_cndmask_b32_e64 v83, v91, v83, s[22:23]
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v85, v85, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v83, vcc
	v_cndmask_b32_e64 v83, v90, v115, s[22:23]
	v_cndmask_b32_e32 v82, v82, v83, vcc
	v_xor_b32_e32 v83, v68, v84
	v_xor_b32_e32 v68, v82, v84
	v_sub_co_u32_e32 v68, vcc, v68, v84
	s_nop 1
	v_subb_co_u32_e32 v83, vcc, v83, v84, vcc
	v_ashrrev_i32_e32 v82, 31, v83
	v_and_b32_e32 v82, s42, v82
	v_mul_lo_u32 v83, v116, s33
	v_add3_u32 v118, v82, v68, v83
	v_lshl_add_u64 v[82:83], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xfbf1
	v_sub_co_u32_e32 v68, vcc, s6, v86
	s_nop 1
	v_subb_co_u32_e32 v84, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v85, v83, v84, s[18:19]
	v_ashrrev_i32_e32 v90, 31, v85
	v_cndmask_b32_e64 v84, v82, v68, s[18:19]
	v_mov_b32_e32 v91, v90
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_xor_b32_e32 v119, v84, v90
	v_xor_b32_e32 v91, v85, v90
	v_mad_u64_u32 v[84:85], s[6:7], v119, s5, 0
	v_mul_hi_u32 v68, v119, s4
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[116:117], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v84, v116
	v_mad_u64_u32 v[114:115], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v117, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[114:115]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v85, s12, v85
	v_mad_u64_u32 v[114:115], s[6:7], s12, v84, 0
	v_add3_u32 v68, v115, v85, v68
	v_sub_co_u32_e32 v85, vcc, v119, v114
	s_nop 1
	v_subb_co_u32_e64 v114, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_cmp_le_u32_e64 s[22:23], s13, v114
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v115, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v85
	v_subrev_co_u32_e32 v85, vcc, s12, v85
	s_nop 0
	v_cndmask_b32_e64 v116, 0, -1, s[22:23]
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cmp_eq_u32_e64 s[22:23], s13, v114
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v85
	v_cndmask_b32_e64 v114, v115, v116, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v91, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v84, v68
	v_cmp_ne_u32_e32 vcc, 0, v114
	s_nop 1
	v_cndmask_b32_e32 v68, v84, v68, vcc
	v_xor_b32_e32 v84, s64, v90
	v_xor_b32_e32 v68, v68, v84
	v_sub_u32_e32 v68, v68, v84
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v117, v82, v84
	v_xad_u32 v116, v68, v76, v30
	v_xor_b32_e32 v85, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v117, s5, 0
	v_mul_hi_u32 v68, v117, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[114:115], s[6:7], v85, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v114
	v_mad_u64_u32 v[90:91], s[6:7], v85, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v115, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v90, s12, v83
	v_mad_u64_u32 v[82:83], s[6:7], s12, v82, 0
	v_add3_u32 v68, v83, v90, v68
	v_sub_u32_e32 v83, v85, v68
	v_sub_co_u32_e32 v82, vcc, v117, v82
	s_mov_b64 s[6:7], 0x40f
	s_nop 0
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, vcc
	v_subrev_co_u32_e64 v90, s[22:23], s12, v82
	v_subb_co_u32_e32 v68, vcc, v85, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[24:25], 0, v83, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s13, v91
	v_subb_co_u32_e64 v83, s[22:23], v83, v61, s[22:23]
	s_nop 0
	v_cndmask_b32_e64 v114, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s12, v90
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v115, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s13, v91
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v82
	v_cndmask_b32_e64 v114, v114, v115, s[24:25]
	v_subrev_co_u32_e64 v115, s[22:23], s12, v90
	s_nop 1
	v_subbrev_co_u32_e64 v83, s[22:23], 0, v83, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v114
	s_nop 1
	v_cndmask_b32_e64 v83, v91, v83, s[22:23]
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v85, v85, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v83, vcc
	v_cndmask_b32_e64 v83, v90, v115, s[22:23]
	v_cndmask_b32_e32 v82, v82, v83, vcc
	v_xor_b32_e32 v83, v68, v84
	v_xor_b32_e32 v68, v82, v84
	v_sub_co_u32_e32 v68, vcc, v68, v84
	s_nop 1
	v_subb_co_u32_e32 v83, vcc, v83, v84, vcc
	v_ashrrev_i32_e32 v82, 31, v83
	v_and_b32_e32 v82, s42, v82
	v_mul_lo_u32 v83, v116, s33
	v_add3_u32 v120, v82, v68, v83
	v_lshl_add_u64 v[82:83], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xfbf0
	v_sub_co_u32_e32 v68, vcc, s6, v86
	s_nop 1
	v_subb_co_u32_e32 v84, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v85, v83, v84, s[18:19]
	v_ashrrev_i32_e32 v90, 31, v85
	v_cndmask_b32_e64 v84, v82, v68, s[18:19]
	v_mov_b32_e32 v91, v90
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_xor_b32_e32 v119, v84, v90
	v_xor_b32_e32 v91, v85, v90
	v_mad_u64_u32 v[84:85], s[6:7], v119, s5, 0
	v_mul_hi_u32 v68, v119, s4
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[116:117], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v84, v116
	v_mad_u64_u32 v[114:115], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v117, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[114:115]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v85, s12, v85
	v_mad_u64_u32 v[114:115], s[6:7], s12, v84, 0
	v_add3_u32 v68, v115, v85, v68
	v_sub_co_u32_e32 v85, vcc, v119, v114
	s_nop 1
	v_subb_co_u32_e64 v114, s[18:19], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_cmp_le_u32_e64 s[18:19], s13, v114
	v_subb_co_u32_e32 v68, vcc, v68, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v115, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s12, v85
	v_subrev_co_u32_e32 v85, vcc, s12, v85
	s_nop 0
	v_cndmask_b32_e64 v116, 0, -1, s[18:19]
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cmp_eq_u32_e64 s[18:19], s13, v114
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v85
	v_cndmask_b32_e64 v114, v115, v116, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v91, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v84, v68
	v_cmp_ne_u32_e32 vcc, 0, v114
	s_nop 1
	v_cndmask_b32_e32 v68, v84, v68, vcc
	v_xor_b32_e32 v84, s64, v90
	v_xor_b32_e32 v68, v68, v84
	v_sub_u32_e32 v68, v68, v84
	v_ashrrev_i32_e32 v84, 31, v83
	v_mov_b32_e32 v85, v84
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[84:85]
	v_xor_b32_e32 v116, v82, v84
	v_xad_u32 v76, v68, v76, v30
	v_xor_b32_e32 v85, v83, v84
	v_mad_u64_u32 v[82:83], s[6:7], v116, s5, 0
	v_mul_hi_u32 v68, v116, s4
	v_lshl_add_u64 v[82:83], v[68:69], 0, v[82:83]
	v_mad_u64_u32 v[90:91], s[6:7], v85, s5, 0
	v_mad_u64_u32 v[114:115], s[4:5], v85, s4, 0
	v_add_co_u32_e32 v68, vcc, v82, v114
	v_mul_lo_u32 v76, v76, s33
	s_nop 0
	v_addc_co_u32_e32 v82, vcc, v83, v115, vcc
	v_mov_b32_e32 v83, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[82:83], v[82:83], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v82
	v_mul_lo_u32 v90, s12, v83
	v_mad_u64_u32 v[82:83], s[4:5], s12, v82, 0
	v_add3_u32 v68, v83, v90, v68
	v_sub_u32_e32 v83, v85, v68
	v_sub_co_u32_e32 v82, vcc, v116, v82
	s_lshl_b64 s[4:5], s[78:79], 8
	s_nop 0
	v_subb_co_u32_e64 v83, s[18:19], v83, v61, vcc
	v_subrev_co_u32_e64 v90, s[18:19], s12, v82
	v_subb_co_u32_e32 v68, vcc, v85, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[22:23], 0, v83, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s13, v91
	v_subb_co_u32_e64 v61, s[18:19], v83, v61, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v114, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v90
	v_subrev_co_u32_e64 v83, s[18:19], s12, v90
	s_nop 0
	v_cndmask_b32_e64 v115, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s13, v91
	v_subbrev_co_u32_e64 v61, s[18:19], 0, v61, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v114, v114, v115, s[22:23]
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cmp_ne_u32_e64 s[18:19], 0, v114
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v82
	v_cndmask_b32_e64 v61, v91, v61, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v85, v85, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v61, v68, v61, vcc
	v_cndmask_b32_e64 v68, v90, v83, s[18:19]
	v_cndmask_b32_e32 v68, v82, v68, vcc
	v_xor_b32_e32 v68, v68, v84
	v_xor_b32_e32 v61, v61, v84
	v_sub_co_u32_e32 v68, vcc, v68, v84
	s_nop 1
	v_subb_co_u32_e32 v61, vcc, v61, v84, vcc
	v_ashrrev_i32_e32 v61, 31, v61
	v_and_b32_e32 v61, s42, v61
	v_add3_u32 v68, v61, v68, v76
	buffer_load_ubyte v61, v77, s[28:31], 0 offen
	buffer_load_ubyte v115, v79, s[28:31], 0 offen
	buffer_load_ubyte v114, v80, s[28:31], 0 offen
	buffer_load_ubyte v116, v81, s[28:31], 0 offen
	buffer_load_ubyte v117, v92, s[28:31], 0 offen
	buffer_load_ubyte v119, v118, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v118, v120, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v120, v68, s[28:31], 0 offen
	v_lshl_add_u64 v[78:79], v[74:75], 0, s[4:5]
	v_lshl_add_u64 v[76:77], v[70:71], 0, v[78:79]
	v_sub_co_u32_e32 v68, vcc, 0xfffffeff, v76
	v_lshl_add_u64 v[80:81], v[76:77], 0, s[86:87]
	s_nop 0
	v_subb_co_u32_e32 v82, vcc, -1, v77, vcc
	v_cmp_gt_i32_e32 vcc, 0, v81
	s_nop 1
	v_cndmask_b32_e32 v85, v81, v82, vcc
	v_cndmask_b32_e32 v84, v80, v68, vcc
	v_or_b32_e32 v68, s71, v85
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
	s_cbranch_execz .LBB0_127
	s_add_u32 s12, s70, s74
	s_mov_b32 s75, s74
	s_addc_u32 s13, s71, s74
	s_xor_b64 s[12:13], s[12:13], s[74:75]
	v_cvt_f32_u32_e32 v68, s12
	v_cvt_f32_u32_e32 v80, s13
	s_sub_u32 s18, 0, s12
	s_subb_u32 s19, 0, s13
	v_ashrrev_i32_e32 v82, 31, v85
	v_fmac_f32_e32 v68, 0x4f800000, v80
	v_rcp_f32_e32 v68, v68
	v_mov_b32_e32 v83, v82
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[82:83]
	v_xor_b32_e32 v83, v84, v82
	v_mul_f32_e32 v68, 0x5f7ffffc, v68
	v_mul_f32_e32 v80, 0x2f800000, v68
	v_trunc_f32_e32 v80, v80
	v_fmac_f32_e32 v68, 0xcf800000, v80
	v_cvt_u32_f32_e32 v80, v80
	v_cvt_u32_f32_e32 v68, v68
	v_readfirstlane_b32 s22, v80
	v_readfirstlane_b32 s23, v68
	s_mul_i32 s24, s18, s22
	s_mul_hi_u32 s26, s18, s23
	s_mul_i32 s25, s19, s23
	s_add_i32 s24, s26, s24
	s_add_i32 s24, s24, s25
	s_mul_i32 s27, s18, s23
	s_mul_i32 s26, s23, s24
	s_mul_hi_u32 s52, s23, s27
	s_mul_hi_u32 s25, s23, s24
	s_add_u32 s26, s52, s26
	s_addc_u32 s25, 0, s25
	s_mul_hi_u32 s53, s22, s27
	s_mul_i32 s27, s22, s27
	s_add_u32 s26, s26, s27
	s_mul_hi_u32 s52, s22, s24
	s_addc_u32 s25, s25, s53
	s_addc_u32 s26, s52, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s24, s25, s24
	s_addc_u32 s25, 0, s26
	s_add_u32 s23, s23, s24
	s_addc_u32 s22, s22, s25
	s_mul_i32 s24, s18, s22
	s_mul_hi_u32 s25, s18, s23
	s_add_i32 s24, s25, s24
	s_mul_i32 s19, s19, s23
	s_add_i32 s24, s24, s19
	s_mul_i32 s18, s18, s23
	s_mul_hi_u32 s25, s22, s18
	s_mul_i32 s26, s22, s18
	s_mul_i32 s52, s23, s24
	s_mul_hi_u32 s18, s23, s18
	s_mul_hi_u32 s27, s23, s24
	s_add_u32 s18, s18, s52
	s_addc_u32 s27, 0, s27
	s_add_u32 s18, s18, s26
	s_mul_hi_u32 s19, s22, s24
	s_addc_u32 s18, s27, s25
	s_addc_u32 s19, s19, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s18, s18, s24
	s_addc_u32 s19, 0, s19
	s_add_u32 s23, s23, s18
	s_addc_u32 s22, s22, s19
	v_xor_b32_e32 v80, v85, v82
	v_mad_u64_u32 v[84:85], s[18:19], v83, s22, 0
	v_mul_hi_u32 v68, v83, s23
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[140:141], s[18:19], v80, s23, 0
	v_add_co_u32_e32 v68, vcc, v84, v140
	v_mad_u64_u32 v[90:91], s[18:19], v80, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v141, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v92, s12, v85
	v_mad_u64_u32 v[90:91], s[18:19], s12, v84, 0
	v_add3_u32 v68, v91, v92, v68
	v_sub_u32_e32 v91, v80, v68
	v_mov_b32_e32 v92, s13
	v_sub_co_u32_e32 v83, vcc, v83, v90
	v_lshl_add_u64 v[140:141], v[84:85], 0, 1
	s_nop 0
	v_subb_co_u32_e64 v90, s[18:19], v91, v92, vcc
	v_subrev_co_u32_e64 v91, s[18:19], s12, v83
	v_subb_co_u32_e32 v68, vcc, v80, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v90, s[18:19], 0, v90, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s13, v90
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s12, v91
	v_cndmask_b32_e64 v80, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v83
	v_cndmask_b32_e64 v91, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s13, v90
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	v_cndmask_b32_e64 v92, v92, v91, s[18:19]
	v_lshl_add_u64 v[90:91], v[84:85], 0, 2
	v_cndmask_b32_e32 v68, v80, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v92
	s_nop 1
	v_cndmask_b32_e32 v80, v140, v90, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v84, v80, vcc
	v_xor_b32_e32 v80, s74, v82
	v_xor_b32_e32 v68, v68, v80
	v_sub_co_u32_e32 v82, vcc, v68, v80
.LBB0_127:
	s_andn2_saveexec_b64 s[6:7], s[6:7]
	s_cbranch_execz .LBB0_129
	v_rcp_iflag_f32_e32 v68, v96
	s_sub_i32 s12, 0, s70
	v_mul_f32_e32 v68, 0x4f7ffffe, v68
	v_cvt_u32_f32_e32 v68, v68
	v_mul_lo_u32 v80, s12, v68
	v_mul_hi_u32 v80, v68, v80
	v_add_u32_e32 v68, v68, v80
	v_mul_hi_u32 v68, v84, v68
	v_mul_lo_u32 v80, v68, s70
	v_sub_u32_e32 v80, v84, v80
	v_add_u32_e32 v82, 1, v68
	v_subrev_u32_e32 v83, s70, v80
	v_cmp_le_u32_e32 vcc, s70, v80
	s_nop 1
	v_cndmask_b32_e32 v80, v80, v83, vcc
	v_cndmask_b32_e32 v68, v68, v82, vcc
	v_add_u32_e32 v82, 1, v68
	v_cmp_le_u32_e32 vcc, s70, v80
	s_nop 1
	v_cndmask_b32_e32 v82, v68, v82, vcc
.LBB0_129:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[78:79], v[78:79], 0, v[46:47]
	v_lshl_add_u64 v[84:85], v[78:79], 0, s[86:87]
	v_or_b32_e32 v68, s71, v85
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
	s_cbranch_execz .LBB0_131
	s_add_u32 s12, s70, s74
	s_mov_b32 s75, s74
	s_addc_u32 s13, s71, s74
	s_xor_b64 s[12:13], s[12:13], s[74:75]
	v_cvt_f32_u32_e32 v68, s12
	v_cvt_f32_u32_e32 v80, s13
	s_sub_u32 s18, 0, s12
	s_subb_u32 s19, 0, s13
	v_ashrrev_i32_e32 v140, 31, v85
	v_fmac_f32_e32 v68, 0x4f800000, v80
	v_rcp_f32_e32 v68, v68
	v_mov_b32_e32 v141, v140
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[140:141]
	v_xor_b32_e32 v83, v84, v140
	v_mul_f32_e32 v68, 0x5f7ffffc, v68
	v_mul_f32_e32 v80, 0x2f800000, v68
	v_trunc_f32_e32 v80, v80
	v_fmac_f32_e32 v68, 0xcf800000, v80
	v_cvt_u32_f32_e32 v80, v80
	v_cvt_u32_f32_e32 v68, v68
	v_readfirstlane_b32 s22, v80
	v_readfirstlane_b32 s23, v68
	s_mul_i32 s24, s18, s22
	s_mul_hi_u32 s26, s18, s23
	s_mul_i32 s25, s19, s23
	s_add_i32 s24, s26, s24
	s_add_i32 s24, s24, s25
	s_mul_i32 s27, s18, s23
	s_mul_i32 s26, s23, s24
	s_mul_hi_u32 s52, s23, s27
	s_mul_hi_u32 s25, s23, s24
	s_add_u32 s26, s52, s26
	s_addc_u32 s25, 0, s25
	s_mul_hi_u32 s53, s22, s27
	s_mul_i32 s27, s22, s27
	s_add_u32 s26, s26, s27
	s_mul_hi_u32 s52, s22, s24
	s_addc_u32 s25, s25, s53
	s_addc_u32 s26, s52, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s24, s25, s24
	s_addc_u32 s25, 0, s26
	s_add_u32 s23, s23, s24
	s_addc_u32 s22, s22, s25
	s_mul_i32 s24, s18, s22
	s_mul_hi_u32 s25, s18, s23
	s_add_i32 s24, s25, s24
	s_mul_i32 s19, s19, s23
	s_add_i32 s24, s24, s19
	s_mul_i32 s18, s18, s23
	s_mul_hi_u32 s25, s22, s18
	s_mul_i32 s26, s22, s18
	s_mul_i32 s52, s23, s24
	s_mul_hi_u32 s18, s23, s18
	s_mul_hi_u32 s27, s23, s24
	s_add_u32 s18, s18, s52
	s_addc_u32 s27, 0, s27
	s_add_u32 s18, s18, s26
	s_mul_hi_u32 s19, s22, s24
	s_addc_u32 s18, s27, s25
	s_addc_u32 s19, s19, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s18, s18, s24
	s_addc_u32 s19, 0, s19
	s_add_u32 s23, s23, s18
	s_addc_u32 s22, s22, s19
	v_xor_b32_e32 v80, v85, v140
	v_mad_u64_u32 v[84:85], s[18:19], v83, s22, 0
	v_mul_hi_u32 v68, v83, s23
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[142:143], s[18:19], v80, s23, 0
	v_add_co_u32_e32 v68, vcc, v84, v142
	v_mad_u64_u32 v[90:91], s[18:19], v80, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v143, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v90, s12, v85
	v_mad_u64_u32 v[84:85], s[18:19], s12, v84, 0
	v_add3_u32 v68, v85, v90, v68
	v_sub_u32_e32 v85, v80, v68
	v_mov_b32_e32 v90, s13
	v_sub_co_u32_e32 v83, vcc, v83, v84
	s_nop 1
	v_subb_co_u32_e64 v84, s[18:19], v85, v90, vcc
	v_subrev_co_u32_e64 v85, s[18:19], s12, v83
	v_subb_co_u32_e32 v68, vcc, v80, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[22:23], 0, v84, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s13, v91
	v_subb_co_u32_e64 v84, s[18:19], v84, v90, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v85
	v_subrev_co_u32_e64 v90, s[18:19], s12, v85
	s_nop 0
	v_cndmask_b32_e64 v121, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s13, v91
	v_subbrev_co_u32_e64 v84, s[18:19], 0, v84, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v92, v92, v121, s[22:23]
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cmp_ne_u32_e64 s[18:19], 0, v92
	s_nop 0
	v_cndmask_b32_e64 v80, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v83
	v_cndmask_b32_e64 v84, v91, v84, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v80, v80, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v80
	v_cndmask_b32_e64 v80, v85, v90, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v80, v83, v80, vcc
	v_cndmask_b32_e32 v68, v68, v84, vcc
	v_xor_b32_e32 v80, v80, v140
	v_xor_b32_e32 v68, v68, v140
	v_sub_co_u32_e32 v90, vcc, v80, v140
	s_nop 1
	v_subb_co_u32_e32 v91, vcc, v68, v140, vcc
.LBB0_131:
	s_andn2_saveexec_b64 s[6:7], s[6:7]
	s_cbranch_execz .LBB0_133
	v_rcp_iflag_f32_e32 v68, v96
	s_sub_i32 s12, 0, s70
	v_mul_f32_e32 v68, 0x4f7ffffe, v68
	v_cvt_u32_f32_e32 v68, v68
	v_mul_lo_u32 v80, s12, v68
	v_mul_hi_u32 v80, v68, v80
	v_add_u32_e32 v68, v68, v80
	v_mul_hi_u32 v68, v84, v68
	v_mul_lo_u32 v68, v68, s70
	v_sub_u32_e32 v68, v84, v68
	v_subrev_u32_e32 v80, s70, v68
	v_cmp_le_u32_e32 vcc, s70, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v80, vcc
	v_subrev_u32_e32 v80, s70, v68
	v_cmp_le_u32_e32 vcc, s70, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v80, vcc
	v_mov_b64_e32 v[90:91], v[68:69]
.LBB0_133:
	s_or_b64 exec, exec, s[6:7]
	s_load_dword s6, s[96:97], 0x58
	v_ashrrev_i32_e32 v68, 31, v81
	v_xor_b32_e32 v68, v82, v68
	v_ashrrev_i32_e32 v80, 31, v91
	v_and_b32_e32 v80, s70, v80
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v68, v68, s6
	v_add3_u32 v68, v68, v90, v80
	buffer_load_ubyte v121, v68, s[8:11], 0 offen
	v_sub_co_u32_e32 v68, vcc, 0xfffffefd, v76
	v_lshl_add_u64 v[80:81], v[76:77], 0, s[88:89]
	s_nop 0
	v_subb_co_u32_e32 v82, vcc, -1, v77, vcc
	v_cmp_gt_i32_e32 vcc, 0, v81
	s_nop 1
	v_cndmask_b32_e32 v85, v81, v82, vcc
	v_cndmask_b32_e32 v84, v80, v68, vcc
	v_or_b32_e32 v68, s71, v85
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
	s_cbranch_execz .LBB0_135
	s_add_u32 s12, s70, s74
	s_mov_b32 s75, s74
	s_addc_u32 s13, s71, s74
	s_xor_b64 s[12:13], s[12:13], s[74:75]
	v_cvt_f32_u32_e32 v68, s12
	v_cvt_f32_u32_e32 v80, s13
	s_sub_u32 s18, 0, s12
	s_subb_u32 s19, 0, s13
	v_ashrrev_i32_e32 v82, 31, v85
	v_fmac_f32_e32 v68, 0x4f800000, v80
	v_rcp_f32_e32 v68, v68
	v_mov_b32_e32 v83, v82
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[82:83]
	v_xor_b32_e32 v83, v84, v82
	v_mul_f32_e32 v68, 0x5f7ffffc, v68
	v_mul_f32_e32 v80, 0x2f800000, v68
	v_trunc_f32_e32 v80, v80
	v_fmac_f32_e32 v68, 0xcf800000, v80
	v_cvt_u32_f32_e32 v80, v80
	v_cvt_u32_f32_e32 v68, v68
	v_readfirstlane_b32 s22, v80
	v_readfirstlane_b32 s23, v68
	s_mul_i32 s24, s18, s22
	s_mul_hi_u32 s26, s18, s23
	s_mul_i32 s25, s19, s23
	s_add_i32 s24, s26, s24
	s_add_i32 s24, s24, s25
	s_mul_i32 s27, s18, s23
	s_mul_i32 s26, s23, s24
	s_mul_hi_u32 s52, s23, s27
	s_mul_hi_u32 s25, s23, s24
	s_add_u32 s26, s52, s26
	s_addc_u32 s25, 0, s25
	s_mul_hi_u32 s53, s22, s27
	s_mul_i32 s27, s22, s27
	s_add_u32 s26, s26, s27
	s_mul_hi_u32 s52, s22, s24
	s_addc_u32 s25, s25, s53
	s_addc_u32 s26, s52, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s24, s25, s24
	s_addc_u32 s25, 0, s26
	s_add_u32 s23, s23, s24
	s_addc_u32 s22, s22, s25
	s_mul_i32 s24, s18, s22
	s_mul_hi_u32 s25, s18, s23
	s_add_i32 s24, s25, s24
	s_mul_i32 s19, s19, s23
	s_add_i32 s24, s24, s19
	s_mul_i32 s18, s18, s23
	s_mul_hi_u32 s25, s22, s18
	s_mul_i32 s26, s22, s18
	s_mul_i32 s52, s23, s24
	s_mul_hi_u32 s18, s23, s18
	s_mul_hi_u32 s27, s23, s24
	s_add_u32 s18, s18, s52
	s_addc_u32 s27, 0, s27
	s_add_u32 s18, s18, s26
	s_mul_hi_u32 s19, s22, s24
	s_addc_u32 s18, s27, s25
	s_addc_u32 s19, s19, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s18, s18, s24
	s_addc_u32 s19, 0, s19
	s_add_u32 s23, s23, s18
	s_addc_u32 s22, s22, s19
	v_xor_b32_e32 v80, v85, v82
	v_mad_u64_u32 v[84:85], s[18:19], v83, s22, 0
	v_mul_hi_u32 v68, v83, s23
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[140:141], s[18:19], v80, s23, 0
	v_add_co_u32_e32 v68, vcc, v84, v140
	v_mad_u64_u32 v[90:91], s[18:19], v80, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v141, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v92, s12, v85
	v_mad_u64_u32 v[90:91], s[18:19], s12, v84, 0
	v_add3_u32 v68, v91, v92, v68
	v_sub_u32_e32 v91, v80, v68
	v_mov_b32_e32 v92, s13
	v_sub_co_u32_e32 v83, vcc, v83, v90
	v_lshl_add_u64 v[140:141], v[84:85], 0, 1
	s_nop 0
	v_subb_co_u32_e64 v90, s[18:19], v91, v92, vcc
	v_subrev_co_u32_e64 v91, s[18:19], s12, v83
	v_subb_co_u32_e32 v68, vcc, v80, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v90, s[18:19], 0, v90, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s13, v90
	v_cmp_le_u32_e32 vcc, s13, v68
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s12, v91
	v_cndmask_b32_e64 v80, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v83
	v_cndmask_b32_e64 v91, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s13, v90
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	v_cndmask_b32_e64 v92, v92, v91, s[18:19]
	v_lshl_add_u64 v[90:91], v[84:85], 0, 2
	v_cndmask_b32_e32 v68, v80, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v92
	s_nop 1
	v_cndmask_b32_e32 v80, v140, v90, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v84, v80, vcc
	v_xor_b32_e32 v80, s74, v82
	v_xor_b32_e32 v68, v68, v80
	v_sub_co_u32_e32 v82, vcc, v68, v80
.LBB0_135:
	s_andn2_saveexec_b64 s[6:7], s[6:7]
	s_cbranch_execz .LBB0_137
	v_rcp_iflag_f32_e32 v68, v96
	s_sub_i32 s12, 0, s70
	v_mul_f32_e32 v68, 0x4f7ffffe, v68
	v_cvt_u32_f32_e32 v68, v68
	v_mul_lo_u32 v80, s12, v68
	v_mul_hi_u32 v80, v68, v80
	v_add_u32_e32 v68, v68, v80
	v_mul_hi_u32 v68, v84, v68
	v_mul_lo_u32 v80, v68, s70
	v_sub_u32_e32 v80, v84, v80
	v_add_u32_e32 v82, 1, v68
	v_subrev_u32_e32 v83, s70, v80
	v_cmp_le_u32_e32 vcc, s70, v80
	s_nop 1
	v_cndmask_b32_e32 v80, v80, v83, vcc
	v_cndmask_b32_e32 v68, v68, v82, vcc
	v_add_u32_e32 v82, 1, v68
	v_cmp_le_u32_e32 vcc, s70, v80
	s_nop 1
	v_cndmask_b32_e32 v82, v68, v82, vcc
.LBB0_137:
	s_or_b64 exec, exec, s[6:7]
	v_lshl_add_u64 v[84:85], v[78:79], 0, s[88:89]
	v_or_b32_e32 v68, s71, v85
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
	s_cbranch_execz .LBB0_139
	s_add_u32 s12, s70, s74
	s_mov_b32 s75, s74
	s_addc_u32 s13, s71, s74
	s_xor_b64 s[12:13], s[12:13], s[74:75]
	v_cvt_f32_u32_e32 v68, s12
	v_cvt_f32_u32_e32 v80, s13
	s_sub_u32 s18, 0, s12
	s_subb_u32 s19, 0, s13
	v_ashrrev_i32_e32 v140, 31, v85
	v_fmac_f32_e32 v68, 0x4f800000, v80
	v_rcp_f32_e32 v68, v68
	v_mov_b32_e32 v141, v140
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[140:141]
	v_xor_b32_e32 v83, v84, v140
	v_mul_f32_e32 v68, 0x5f7ffffc, v68
	v_mul_f32_e32 v80, 0x2f800000, v68
	v_trunc_f32_e32 v80, v80
	v_fmac_f32_e32 v68, 0xcf800000, v80
	v_cvt_u32_f32_e32 v80, v80
	v_cvt_u32_f32_e32 v68, v68
	v_readfirstlane_b32 s22, v80
	v_readfirstlane_b32 s23, v68
	s_mul_i32 s24, s18, s22
	s_mul_hi_u32 s26, s18, s23
	s_mul_i32 s25, s19, s23
	s_add_i32 s24, s26, s24
	s_add_i32 s24, s24, s25
	s_mul_i32 s27, s18, s23
	s_mul_i32 s26, s23, s24
	s_mul_hi_u32 s52, s23, s27
	s_mul_hi_u32 s25, s23, s24
	s_add_u32 s26, s52, s26
	s_addc_u32 s25, 0, s25
	s_mul_hi_u32 s53, s22, s27
	s_mul_i32 s27, s22, s27
	s_add_u32 s26, s26, s27
	s_mul_hi_u32 s52, s22, s24
	s_addc_u32 s25, s25, s53
	s_addc_u32 s26, s52, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s24, s25, s24
	s_addc_u32 s25, 0, s26
	s_add_u32 s23, s23, s24
	s_addc_u32 s22, s22, s25
	s_mul_i32 s24, s18, s22
	s_mul_hi_u32 s25, s18, s23
	s_add_i32 s24, s25, s24
	s_mul_i32 s19, s19, s23
	s_add_i32 s24, s24, s19
	s_mul_i32 s18, s18, s23
	s_mul_hi_u32 s25, s22, s18
	s_mul_i32 s26, s22, s18
	s_mul_i32 s52, s23, s24
	s_mul_hi_u32 s18, s23, s18
	s_mul_hi_u32 s27, s23, s24
	s_add_u32 s18, s18, s52
	s_addc_u32 s27, 0, s27
	s_add_u32 s18, s18, s26
	s_mul_hi_u32 s19, s22, s24
	s_addc_u32 s18, s27, s25
	s_addc_u32 s19, s19, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s18, s18, s24
	s_addc_u32 s19, 0, s19
	s_add_u32 s23, s23, s18
	s_addc_u32 s22, s22, s19
	v_xor_b32_e32 v80, v85, v140
	v_mad_u64_u32 v[84:85], s[18:19], v83, s22, 0
	v_mul_hi_u32 v68, v83, s23
	v_lshl_add_u64 v[84:85], v[68:69], 0, v[84:85]
	v_mad_u64_u32 v[142:143], s[18:19], v80, s23, 0
	v_add_co_u32_e32 v68, vcc, v84, v142
	v_mad_u64_u32 v[90:91], s[18:19], v80, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v84, vcc, v85, v143, vcc
	v_mov_b32_e32 v85, s41
	s_nop 0
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[84:85], v[84:85], 0, v[90:91]
	v_mul_lo_u32 v68, s13, v84
	v_mul_lo_u32 v90, s12, v85
	v_mad_u64_u32 v[84:85], s[18:19], s12, v84, 0
	v_add3_u32 v68, v85, v90, v68
	v_sub_u32_e32 v85, v80, v68
	v_mov_b32_e32 v90, s13
	v_sub_co_u32_e32 v83, vcc, v83, v84
	s_nop 1
	v_subb_co_u32_e64 v84, s[18:19], v85, v90, vcc
	v_subrev_co_u32_e64 v85, s[18:19], s12, v83
	v_subb_co_u32_e32 v68, vcc, v80, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[22:23], 0, v84, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s13, v91
	v_subb_co_u32_e64 v84, s[18:19], v84, v90, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v92, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s12, v85
	v_subrev_co_u32_e64 v90, s[18:19], s12, v85
	s_nop 0
	v_cndmask_b32_e64 v123, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s13, v91
	v_subbrev_co_u32_e64 v84, s[18:19], 0, v84, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v92, v92, v123, s[22:23]
	v_cmp_le_u32_e32 vcc, s13, v68
	v_cmp_ne_u32_e64 s[18:19], 0, v92
	s_nop 0
	v_cndmask_b32_e64 v80, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s12, v83
	v_cndmask_b32_e64 v84, v91, v84, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s13, v68
	s_nop 1
	v_cndmask_b32_e32 v80, v80, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v80
	v_cndmask_b32_e64 v80, v85, v90, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v80, v83, v80, vcc
	v_cndmask_b32_e32 v68, v68, v84, vcc
	v_xor_b32_e32 v80, v80, v140
	v_xor_b32_e32 v68, v68, v140
	v_sub_co_u32_e32 v90, vcc, v80, v140
	s_nop 1
	v_subb_co_u32_e32 v91, vcc, v68, v140, vcc
.LBB0_139:
	s_andn2_saveexec_b64 s[6:7], s[6:7]
	s_cbranch_execz .LBB0_141
	v_rcp_iflag_f32_e32 v68, v96
	s_sub_i32 s12, 0, s70
	v_mul_f32_e32 v68, 0x4f7ffffe, v68
	v_cvt_u32_f32_e32 v68, v68
	v_mul_lo_u32 v80, s12, v68
	v_mul_hi_u32 v80, v68, v80
	v_add_u32_e32 v68, v68, v80
	v_mul_hi_u32 v68, v84, v68
	v_mul_lo_u32 v68, v68, s70
	v_sub_u32_e32 v68, v84, v68
	v_subrev_u32_e32 v80, s70, v68
	v_cmp_le_u32_e32 vcc, s70, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v80, vcc
	v_subrev_u32_e32 v80, s70, v68
	v_cmp_le_u32_e32 vcc, s70, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v80, vcc
	v_mov_b64_e32 v[90:91], v[68:69]
.LBB0_141:
	s_or_b64 exec, exec, s[6:7]
	s_load_dword s6, s[96:97], 0x58
	v_ashrrev_i32_e32 v68, 31, v81
	v_xor_b32_e32 v68, v82, v68
	v_ashrrev_i32_e32 v80, 31, v91
	v_and_b32_e32 v80, s70, v80
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v68, v68, s6
	v_add3_u32 v68, v68, v90, v80
	buffer_load_ubyte v92, v68, s[8:11], 0 offen
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	s_lshl_b64 s[6:7], s[78:79], 7
	v_mov_b32_e32 v85, s5
	v_or_b32_e32 v84, s4, v36
	v_or_b32_e32 v82, s6, v32
	v_lshl_add_u64 v[80:81], v[84:85], 0, s[90:91]
	v_add_u32_e32 v68, 0x100, v82
	v_cmp_gt_i64_e32 vcc, s[34:35], v[80:81]
	v_add_u32_e32 v83, v68, v34
	s_and_b64 s[18:19], s[2:3], vcc
	s_mov_b32 m0, s46
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[136:139], v[16:19], v[8:11], v122, v33 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e64 v80, v31, v83, s[18:19]
	s_and_b64 vcc, s[14:15], vcc
	buffer_load_dwordx4 v80, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[128:131], v[16:19], v[20:23], v122, v33 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s54
	s_lshl_b64 s[4:5], s[78:79], 13
	v_lshl_add_u64 v[80:81], v[40:41], 0, s[4:5]
	v_add_u32_e32 v20, v68, v38
	v_cndmask_b32_e32 v20, v31, v20, vcc
	buffer_load_dwordx4 v20, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[132:135], v[4:7], v[8:11], v122, v93 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_mov_b64 s[4:5], 0x4000
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[124:127], v[4:7], v[16:19], v122, v93 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 v[122:125], v50 offset:8192
	ds_read_b128 v[126:129], v52 offset:10240
	ds_read_b128 v[130:133], v50 offset:10240
	ds_read_b128 v[134:137], v52 offset:8192
	ds_read_b32 v93, v97 offset:25088
	v_sub_co_u32_e32 v18, vcc, 0xffffbfff, v80
	v_lshl_add_u64 v[16:17], v[80:81], 0, s[4:5]
	s_nop 0
	v_subb_co_u32_e32 v19, vcc, -1, v81, vcc
	v_cmp_gt_i32_e32 vcc, 0, v17
	s_nop 1
	v_cndmask_b32_e32 v21, v17, v19, vcc
	v_cndmask_b32_e32 v20, v16, v18, vcc
	v_or_b32_e32 v18, s35, v21
	v_cmp_ne_u32_e32 vcc, 0, v18
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[4:5], exec, s[4:5]
	s_cbranch_execz .LBB0_143
	s_add_u32 s6, s34, s72
	s_mov_b32 s73, s72
	s_addc_u32 s7, s35, s72
	s_xor_b64 s[6:7], s[6:7], s[72:73]
	v_cvt_f32_u32_e32 v18, s6
	v_cvt_f32_u32_e32 v19, s7
	s_sub_u32 s12, 0, s6
	s_subb_u32 s13, 0, s7
	v_fmac_f32_e32 v18, 0x4f800000, v19
	v_rcp_f32_e32 v18, v18
	s_nop 0
	v_mul_f32_e32 v18, 0x5f7ffffc, v18
	v_mul_f32_e32 v19, 0x2f800000, v18
	v_trunc_f32_e32 v19, v19
	v_fmac_f32_e32 v18, 0xcf800000, v19
	v_cvt_u32_f32_e32 v19, v19
	v_cvt_u32_f32_e32 v18, v18
	v_readfirstlane_b32 s18, v19
	v_readfirstlane_b32 s19, v18
	s_mul_i32 s22, s12, s18
	s_mul_hi_u32 s24, s12, s19
	s_mul_i32 s23, s13, s19
	s_add_i32 s22, s24, s22
	s_add_i32 s22, s22, s23
	s_mul_i32 s25, s12, s19
	s_mul_i32 s24, s19, s22
	s_mul_hi_u32 s26, s19, s25
	s_mul_hi_u32 s23, s19, s22
	s_add_u32 s24, s26, s24
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 s27, s18, s25
	s_mul_i32 s25, s18, s25
	s_add_u32 s24, s24, s25
	s_mul_hi_u32 s26, s18, s22
	s_addc_u32 s23, s23, s27
	s_addc_u32 s24, s26, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s22, s23, s22
	s_addc_u32 s23, 0, s24
	s_add_u32 s19, s19, s22
	s_addc_u32 s18, s18, s23
	s_mul_i32 s22, s12, s18
	s_mul_hi_u32 s23, s12, s19
	s_add_i32 s22, s23, s22
	s_mul_i32 s13, s13, s19
	s_add_i32 s22, s22, s13
	s_mul_i32 s12, s12, s19
	s_mul_hi_u32 s23, s18, s12
	s_mul_i32 s24, s18, s12
	s_mul_i32 s26, s19, s22
	s_mul_hi_u32 s12, s19, s12
	s_mul_hi_u32 s25, s19, s22
	s_add_u32 s12, s12, s26
	s_addc_u32 s25, 0, s25
	s_add_u32 s12, s12, s24
	s_mul_hi_u32 s13, s18, s22
	s_addc_u32 s12, s25, s23
	s_addc_u32 s13, s13, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s12, s12, s22
	v_ashrrev_i32_e32 v18, 31, v21
	s_addc_u32 s13, 0, s13
	v_mov_b32_e32 v19, v18
	s_add_u32 s19, s19, s12
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[18:19]
	s_addc_u32 s18, s18, s13
	v_xor_b32_e32 v33, v20, v18
	v_xor_b32_e32 v19, v21, v18
	v_mad_u64_u32 v[20:21], s[12:13], v33, s18, 0
	v_mul_hi_u32 v68, v33, s19
	v_lshl_add_u64 v[20:21], v[68:69], 0, v[20:21]
	v_mad_u64_u32 v[90:91], s[12:13], v19, s19, 0
	v_add_co_u32_e32 v20, vcc, v20, v90
	v_mad_u64_u32 v[22:23], s[12:13], v19, s18, 0
	s_nop 0
	v_addc_co_u32_e32 v20, vcc, v21, v91, vcc
	v_mov_b32_e32 v21, s41
	s_nop 0
	v_addc_co_u32_e32 v23, vcc, 0, v23, vcc
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[22:23]
	v_mul_lo_u32 v68, s7, v20
	v_mul_lo_u32 v83, s6, v21
	v_mad_u64_u32 v[22:23], s[12:13], s6, v20, 0
	v_add3_u32 v68, v23, v83, v68
	v_sub_u32_e32 v23, v19, v68
	v_mov_b32_e32 v83, s7
	v_sub_co_u32_e32 v33, vcc, v33, v22
	v_lshl_add_u64 v[90:91], v[20:21], 0, 1
	s_nop 0
	v_subb_co_u32_e64 v22, s[18:19], v23, v83, vcc
	v_subrev_co_u32_e64 v23, s[18:19], s6, v33
	v_subb_co_u32_e32 v19, vcc, v19, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v22, s[18:19], 0, v22, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v22
	v_cmp_le_u32_e32 vcc, s7, v19
	v_xor_b32_e32 v18, s72, v18
	v_cndmask_b32_e64 v83, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v23
	s_nop 1
	v_cndmask_b32_e64 v23, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v22
	s_nop 1
	v_cndmask_b32_e64 v83, v83, v23, s[18:19]
	v_lshl_add_u64 v[22:23], v[20:21], 0, 2
	v_cndmask_b32_e64 v21, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v33
	s_nop 1
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v19
	s_nop 1
	v_cndmask_b32_e32 v19, v21, v23, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	s_nop 1
	v_cndmask_b32_e32 v21, v90, v22, vcc
	v_cmp_ne_u32_e32 vcc, 0, v19
	s_nop 1
	v_cndmask_b32_e32 v19, v20, v21, vcc
	v_xor_b32_e32 v19, v19, v18
	v_sub_co_u32_e32 v18, vcc, v19, v18
.LBB0_143:
	s_andn2_saveexec_b64 s[4:5], s[4:5]
	s_cbranch_execz .LBB0_145
	v_cvt_f32_u32_e32 v18, s34
	s_sub_i32 s6, 0, s34
	v_rcp_iflag_f32_e32 v18, v18
	s_nop 0
	v_mul_f32_e32 v18, 0x4f7ffffe, v18
	v_cvt_u32_f32_e32 v18, v18
	v_mul_lo_u32 v19, s6, v18
	v_mul_hi_u32 v19, v18, v19
	v_add_u32_e32 v18, v18, v19
	v_mul_hi_u32 v18, v20, v18
	v_mul_lo_u32 v19, v18, s34
	v_sub_u32_e32 v19, v20, v19
	v_add_u32_e32 v21, 1, v18
	v_subrev_u32_e32 v20, s34, v19
	v_cmp_le_u32_e32 vcc, s34, v19
	s_nop 1
	v_cndmask_b32_e32 v19, v19, v20, vcc
	v_cndmask_b32_e32 v18, v18, v21, vcc
	v_add_u32_e32 v20, 1, v18
	v_cmp_le_u32_e32 vcc, s34, v19
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v20, vcc
.LBB0_145:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v19, s35, v17
	v_cmp_ne_u32_e32 vcc, 0, v19
	v_ashrrev_i32_e32 v20, 31, v17
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[4:5], exec, s[4:5]
	s_cbranch_execz .LBB0_147
	s_add_u32 s6, s34, s72
	s_mov_b32 s73, s72
	s_addc_u32 s7, s35, s72
	s_xor_b64 s[6:7], s[6:7], s[72:73]
	v_cvt_f32_u32_e32 v19, s6
	v_cvt_f32_u32_e32 v21, s7
	s_sub_u32 s12, 0, s6
	s_subb_u32 s13, 0, s7
	v_fmac_f32_e32 v19, 0x4f800000, v21
	v_rcp_f32_e32 v19, v19
	s_nop 0
	v_mul_f32_e32 v19, 0x5f7ffffc, v19
	v_mul_f32_e32 v21, 0x2f800000, v19
	v_trunc_f32_e32 v21, v21
	v_fmac_f32_e32 v19, 0xcf800000, v21
	v_cvt_u32_f32_e32 v21, v21
	v_cvt_u32_f32_e32 v19, v19
	v_readfirstlane_b32 s18, v21
	v_readfirstlane_b32 s19, v19
	s_mul_i32 s22, s12, s18
	s_mul_hi_u32 s24, s12, s19
	s_mul_i32 s23, s13, s19
	s_add_i32 s22, s24, s22
	s_add_i32 s22, s22, s23
	s_mul_i32 s25, s12, s19
	s_mul_i32 s24, s19, s22
	s_mul_hi_u32 s26, s19, s25
	s_mul_hi_u32 s23, s19, s22
	s_add_u32 s24, s26, s24
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 s27, s18, s25
	s_mul_i32 s25, s18, s25
	s_add_u32 s24, s24, s25
	s_mul_hi_u32 s26, s18, s22
	s_addc_u32 s23, s23, s27
	s_addc_u32 s24, s26, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s22, s23, s22
	s_addc_u32 s23, 0, s24
	s_add_u32 s19, s19, s22
	s_addc_u32 s18, s18, s23
	s_mul_i32 s22, s12, s18
	s_mul_hi_u32 s23, s12, s19
	s_add_i32 s22, s23, s22
	s_mul_i32 s13, s13, s19
	s_add_i32 s22, s22, s13
	s_mul_i32 s12, s12, s19
	s_mul_hi_u32 s23, s18, s12
	s_mul_i32 s24, s18, s12
	s_mul_i32 s26, s19, s22
	s_mul_hi_u32 s12, s19, s12
	s_mul_hi_u32 s25, s19, s22
	s_add_u32 s12, s12, s26
	s_addc_u32 s25, 0, s25
	s_add_u32 s12, s12, s24
	s_mul_hi_u32 s13, s18, s22
	s_addc_u32 s12, s25, s23
	s_addc_u32 s13, s13, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s12, s12, s22
	s_addc_u32 s13, 0, s13
	v_mov_b32_e32 v21, v20
	s_add_u32 s19, s19, s12
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[20:21]
	s_addc_u32 s18, s18, s13
	v_xor_b32_e32 v21, v16, v20
	v_xor_b32_e32 v19, v17, v20
	v_mad_u64_u32 v[16:17], s[12:13], v21, s18, 0
	v_mul_hi_u32 v68, v21, s19
	v_lshl_add_u64 v[16:17], v[68:69], 0, v[16:17]
	v_mad_u64_u32 v[90:91], s[12:13], v19, s19, 0
	v_add_co_u32_e32 v16, vcc, v16, v90
	v_mad_u64_u32 v[22:23], s[12:13], v19, s18, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v91, vcc
	v_mov_b32_e32 v17, s41
	s_nop 0
	v_addc_co_u32_e32 v23, vcc, 0, v23, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[22:23]
	v_mul_lo_u32 v22, s7, v16
	v_mul_lo_u32 v23, s6, v17
	v_mad_u64_u32 v[16:17], s[12:13], s6, v16, 0
	v_add3_u32 v17, v17, v23, v22
	v_sub_u32_e32 v22, v19, v17
	v_mov_b32_e32 v23, s7
	v_sub_co_u32_e32 v16, vcc, v21, v16
	s_nop 1
	v_subb_co_u32_e64 v21, s[18:19], v22, v23, vcc
	v_subrev_co_u32_e64 v22, s[18:19], s6, v16
	v_subb_co_u32_e32 v17, vcc, v19, v17, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v33, s[22:23], 0, v21, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v33
	v_subb_co_u32_e64 v21, s[18:19], v21, v23, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v68, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v22
	v_subrev_co_u32_e64 v23, s[18:19], s6, v22
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v33
	v_subbrev_co_u32_e64 v21, s[18:19], 0, v21, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v68, v68, v83, s[22:23]
	v_cmp_le_u32_e32 vcc, s7, v17
	v_cmp_ne_u32_e64 s[18:19], 0, v68
	s_nop 0
	v_cndmask_b32_e64 v19, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v16
	v_cndmask_b32_e64 v21, v33, v21, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v33, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v17
	s_nop 1
	v_cndmask_b32_e32 v19, v19, v33, vcc
	v_cmp_ne_u32_e32 vcc, 0, v19
	v_cndmask_b32_e64 v19, v22, v23, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v16, v16, v19, vcc
	v_cndmask_b32_e32 v17, v17, v21, vcc
	v_xor_b32_e32 v16, v16, v20
	v_xor_b32_e32 v17, v17, v20
	v_sub_co_u32_e32 v22, vcc, v16, v20
	s_nop 1
	v_subb_co_u32_e32 v23, vcc, v17, v20, vcc
.LBB0_147:
	s_andn2_saveexec_b64 s[4:5], s[4:5]
	s_cbranch_execz .LBB0_149
	v_cvt_f32_u32_e32 v17, s34
	s_sub_i32 s6, 0, s34
	v_rcp_iflag_f32_e32 v17, v17
	s_nop 0
	v_mul_f32_e32 v17, 0x4f7ffffe, v17
	v_cvt_u32_f32_e32 v17, v17
	v_mul_lo_u32 v19, s6, v17
	v_mul_hi_u32 v19, v17, v19
	v_add_u32_e32 v17, v17, v19
	v_mul_hi_u32 v17, v16, v17
	v_mul_lo_u32 v17, v17, s34
	v_sub_u32_e32 v16, v16, v17
	v_subrev_u32_e32 v17, s34, v16
	v_cmp_le_u32_e32 vcc, s34, v16
	s_nop 1
	v_cndmask_b32_e32 v16, v16, v17, vcc
	v_subrev_u32_e32 v17, s34, v16
	v_cmp_le_u32_e32 vcc, s34, v16
	s_nop 1
	v_cndmask_b32_e32 v68, v16, v17, vcc
	v_mov_b64_e32 v[22:23], v[68:69]
.LBB0_149:
	s_or_b64 exec, exec, s[4:5]
	v_ashrrev_i32_e32 v16, 31, v23
	v_and_b32_e32 v17, s35, v16
	v_and_b32_e32 v16, s34, v16
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[22:23]
	v_ashrrev_i32_e32 v19, 31, v17
	v_xor_b32_e32 v17, v19, v17
	v_xor_b32_e32 v18, v18, v20
	v_ashrrev_i32_e32 v20, 31, v17
	v_xor_b32_e32 v16, v19, v16
	v_lshrrev_b32_e32 v68, 27, v20
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[68:69]
	v_readlane_b32 s4, v144, 18
	v_alignbit_b32 v16, v17, v16, 5
	s_mov_b32 m0, s44
	v_mul_lo_u32 v17, v18, s4
	v_xad_u32 v16, v16, v19, v17
	buffer_load_dword v16, s[92:95], 0 offen lds
	v_lshlrev_b16_e32 v16, 8, v104
	v_lshlrev_b16_e32 v17, 8, v105
	v_or_b32_e32 v16, v102, v16
	v_or_b32_sdwa v17, v103, v17 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v20, 8, v51
	v_or_b32_sdwa v19, v16, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v16, 8, v100
	v_lshlrev_b16_e32 v17, 8, v101
	v_or_b32_e32 v16, v98, v16
	v_or_b32_sdwa v17, v99, v17 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v20, v37, v20 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v18, v16, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v16, 8, v57
	v_lshlrev_b16_e32 v17, 8, v59
	v_or_b32_e32 v16, v53, v16
	v_or_b32_sdwa v17, v55, v17 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt vmcnt(5)
	v_lshlrev_b16_e32 v21, 8, v120
	v_or_b32_sdwa v17, v16, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v16, 8, v39
	v_or_b32_e32 v16, v35, v16
	v_or_b32_sdwa v16, v16, v20 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v20, 8, v119
	v_or_b32_e32 v20, v117, v20
	v_or_b32_sdwa v21, v118, v21 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v33, 8, v109
	v_or_b32_sdwa v23, v20, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v20, 8, v115
	v_lshlrev_b16_e32 v21, 8, v116
	v_or_b32_e32 v20, v61, v20
	v_or_b32_sdwa v21, v114, v21 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v33, v107, v33 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v22, v20, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v20, 8, v112
	v_lshlrev_b16_e32 v21, 8, v113
	v_or_b32_e32 v20, v110, v20
	v_or_b32_sdwa v21, v111, v21 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_readlane_b32 s5, v144, 19
	v_or_b32_sdwa v21, v20, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v20, 8, v108
	v_or_b32_e32 v20, v106, v20
	v_or_b32_sdwa v20, v20, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_waitcnt vmcnt(3) lgkmcnt(0)
	s_barrier
	s_mov_b64 s[4:5], 0x1000
	v_lshl_add_u64 v[90:91], v[88:89], 0, s[4:5]
	s_movk_i32 s4, 0xefff
	v_sub_co_u32_e64 v33, s[18:19], s4, v88
	s_add_u32 s4, s42, s64
	s_mov_b32 s65, s64
	s_addc_u32 s5, s43, s64
	s_xor_b64 s[52:53], s[4:5], s[64:65]
	v_cvt_f32_u32_e32 v39, s52
	v_cvt_f32_u32_e32 v51, s53
	v_subb_co_u32_e64 v37, s[18:19], -1, v89, s[18:19]
	s_sub_u32 s4, 0, s52
	v_fmac_f32_e32 v39, 0x4f800000, v51
	v_rcp_f32_e32 v39, v39
	s_subb_u32 s5, 0, s53
	v_cmp_gt_i32_e32 vcc, 0, v91
	v_mov_b32_e32 v57, s53
	v_mul_f32_e32 v39, 0x5f7ffffc, v39
	v_mul_f32_e32 v51, 0x2f800000, v39
	v_trunc_f32_e32 v51, v51
	v_fmac_f32_e32 v39, 0xcf800000, v51
	v_cvt_u32_f32_e32 v39, v39
	v_cndmask_b32_e32 v99, v91, v37, vcc
	v_ashrrev_i32_e32 v100, 31, v99
	v_cndmask_b32_e32 v98, v90, v33, vcc
	v_readfirstlane_b32 s6, v39
	v_cvt_u32_f32_e32 v39, v51
	s_mul_hi_u32 s12, s4, s6
	s_mul_i32 s7, s5, s6
	s_mul_i32 s19, s4, s6
	v_readfirstlane_b32 s13, v39
	s_mul_i32 s18, s4, s13
	s_add_i32 s12, s12, s18
	s_add_i32 s12, s12, s7
	s_mul_i32 s18, s6, s12
	s_mul_hi_u32 s22, s6, s19
	s_mul_hi_u32 s7, s6, s12
	s_add_u32 s18, s22, s18
	s_addc_u32 s7, 0, s7
	s_mul_hi_u32 s23, s13, s19
	s_mul_i32 s19, s13, s19
	s_add_u32 s18, s18, s19
	s_mul_hi_u32 s22, s13, s12
	s_addc_u32 s7, s7, s23
	s_addc_u32 s18, s22, 0
	s_mul_i32 s12, s13, s12
	s_add_u32 s7, s7, s12
	s_addc_u32 s12, 0, s18
	s_add_u32 s6, s6, s7
	s_addc_u32 s7, s13, s12
	s_mul_i32 s12, s4, s7
	s_mul_hi_u32 s13, s4, s6
	s_add_i32 s12, s13, s12
	s_mul_i32 s5, s5, s6
	s_add_i32 s12, s12, s5
	s_mul_i32 s4, s4, s6
	s_mul_hi_u32 s13, s7, s4
	s_mul_i32 s18, s7, s4
	s_mul_i32 s22, s6, s12
	s_mul_hi_u32 s4, s6, s4
	s_mul_hi_u32 s19, s6, s12
	s_add_u32 s4, s4, s22
	s_addc_u32 s19, 0, s19
	s_add_u32 s4, s4, s18
	s_mul_hi_u32 s5, s7, s12
	s_addc_u32 s4, s19, s13
	s_addc_u32 s5, s5, 0
	s_mul_i32 s12, s7, s12
	s_add_u32 s4, s4, s12
	v_mov_b32_e32 v101, v100
	s_addc_u32 s5, 0, s5
	v_lshl_add_u64 v[98:99], v[98:99], 0, v[100:101]
	s_add_u32 s4, s6, s4
	v_xor_b32_e32 v37, v98, v100
	s_addc_u32 s5, s7, s5
	v_xor_b32_e32 v33, v99, v100
	v_mad_u64_u32 v[98:99], s[6:7], v37, s5, 0
	v_mul_hi_u32 v68, v37, s4
	v_lshl_add_u64 v[98:99], v[68:69], 0, v[98:99]
	v_mad_u64_u32 v[104:105], s[6:7], v33, s4, 0
	v_add_co_u32_e32 v39, vcc, v98, v104
	v_mad_u64_u32 v[102:103], s[6:7], v33, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v99, v105, vcc
	v_and_b32_e32 v35, 0xffff, v121
	s_nop 0
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[98:99], v[68:69], 0, v[102:103]
	v_mul_lo_u32 v39, s53, v98
	v_mul_lo_u32 v51, s52, v99
	v_mad_u64_u32 v[102:103], s[6:7], s52, v98, 0
	v_add3_u32 v39, v103, v51, v39
	v_sub_co_u32_e32 v37, vcc, v37, v102
	v_cmp_le_u32_e64 s[22:23], s52, v37
	s_nop 0
	v_subb_co_u32_e64 v51, s[18:19], v33, v39, vcc
	v_sub_u32_e32 v33, v33, v39
	v_subb_co_u32_e32 v33, vcc, v33, v57, vcc
	v_subrev_co_u32_e32 v37, vcc, s52, v37
	v_cmp_le_u32_e64 s[18:19], s53, v51
	s_nop 0
	v_subbrev_co_u32_e32 v33, vcc, 0, v33, vcc
	v_cndmask_b32_e64 v53, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s53, v51
	v_cndmask_b32_e64 v51, 0, -1, s[22:23]
	v_cmp_le_u32_e32 vcc, s53, v33
	v_cmp_le_u32_e64 s[22:23], s52, v37
	v_cndmask_b32_e64 v51, v53, v51, s[18:19]
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v33
	v_cndmask_b32_e64 v33, 0, -1, s[22:23]
	v_cmp_ne_u32_e64 s[18:19], 0, v51
	v_cndmask_b32_e32 v33, v39, v33, vcc
	v_cmp_ne_u32_e32 vcc, 0, v33
	v_xor_b32_e32 v37, s64, v100
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[134:137], v[16:19], v[12:15], v93, v35 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e64 v33, 1, 2, vcc
	v_add_u32_e32 v33, v98, v33
	v_cndmask_b32_e64 v33, v98, v33, s[18:19]
	v_ashrrev_i32_e32 v98, 31, v91
	v_mov_b32_e32 v99, v98
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[98:99]
	v_xor_b32_e32 v33, v33, v37
	v_xor_b32_e32 v39, v90, v98
	v_sub_u32_e32 v33, v33, v37
	v_xor_b32_e32 v37, v91, v98
	v_mad_u64_u32 v[90:91], s[6:7], v39, s5, 0
	v_mul_hi_u32 v68, v39, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[102:103], s[6:7], v37, s4, 0
	v_add_co_u32_e32 v51, vcc, v90, v102
	v_mad_u64_u32 v[100:101], s[6:7], v37, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v91, v103, vcc
	v_xad_u32 v33, v33, v98, v30
	s_nop 0
	v_addc_co_u32_e32 v101, vcc, 0, v101, vcc
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[100:101]
	v_mul_lo_u32 v51, s53, v90
	v_mul_lo_u32 v53, s52, v91
	v_mad_u64_u32 v[90:91], s[6:7], s52, v90, 0
	v_add3_u32 v51, v91, v53, v51
	v_sub_u32_e32 v53, v37, v51
	v_sub_co_u32_e32 v39, vcc, v39, v90
	s_mov_b64 s[6:7], 0x801
	s_nop 0
	v_subb_co_u32_e64 v53, s[18:19], v53, v57, vcc
	v_subrev_co_u32_e64 v55, s[18:19], s52, v39
	v_cmp_le_u32_e64 s[24:25], s52, v55
	s_nop 0
	v_subbrev_co_u32_e64 v59, s[22:23], 0, v53, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s53, v59
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_subb_co_u32_e64 v53, s[18:19], v53, v57, s[18:19]
	v_cndmask_b32_e64 v61, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v59
	v_subb_co_u32_e32 v37, vcc, v37, v51, vcc
	s_nop 0
	v_cndmask_b32_e64 v61, v61, v68, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v61
	v_subrev_co_u32_e64 v61, s[18:19], s52, v55
	v_cmp_le_u32_e32 vcc, s53, v37
	s_nop 0
	v_subbrev_co_u32_e64 v53, s[18:19], 0, v53, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s52, v39
	v_cndmask_b32_e64 v53, v59, v53, s[22:23]
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v37
	v_cndmask_b32_e64 v59, 0, -1, s[18:19]
	v_mul_lo_u32 v33, v33, s33
	v_cndmask_b32_e32 v51, v51, v59, vcc
	v_cmp_ne_u32_e32 vcc, 0, v51
	v_cndmask_b32_e64 v51, v55, v61, s[22:23]
	v_lshl_add_u64 v[90:91], v[86:87], 0, s[6:7]
	v_cndmask_b32_e32 v39, v39, v51, vcc
	v_cndmask_b32_e32 v37, v37, v53, vcc
	v_xor_b32_e32 v39, v39, v98
	v_xor_b32_e32 v37, v37, v98
	v_sub_co_u32_e32 v39, vcc, v39, v98
	s_movk_i32 s6, 0xf7fe
	s_nop 0
	v_subb_co_u32_e32 v37, vcc, v37, v98, vcc
	v_ashrrev_i32_e32 v37, 31, v37
	v_and_b32_e32 v37, s42, v37
	v_add3_u32 v33, v37, v39, v33
	v_lshl_add_u64 v[98:99], v[86:87], 0, s[80:81]
	v_sub_co_u32_e32 v37, vcc, s6, v86
	v_cmp_gt_i32_e64 s[18:19], 0, v99
	s_nop 0
	v_subb_co_u32_e32 v39, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v101, v91, v39, s[18:19]
	v_ashrrev_i32_e32 v102, 31, v101
	v_cndmask_b32_e64 v100, v90, v37, s[18:19]
	v_mov_b32_e32 v103, v102
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[102:103]
	v_xor_b32_e32 v39, v100, v102
	v_xor_b32_e32 v37, v101, v102
	v_mad_u64_u32 v[100:101], s[6:7], v39, s5, 0
	v_mul_hi_u32 v68, v39, s4
	v_lshl_add_u64 v[100:101], v[68:69], 0, v[100:101]
	v_mad_u64_u32 v[106:107], s[6:7], v37, s4, 0
	v_add_co_u32_e32 v51, vcc, v100, v106
	v_mad_u64_u32 v[104:105], s[6:7], v37, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v101, v107, vcc
	v_ashrrev_i32_e32 v98, 31, v91
	s_nop 0
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[100:101], v[68:69], 0, v[104:105]
	v_mul_lo_u32 v51, s53, v100
	v_mul_lo_u32 v53, s52, v101
	v_mad_u64_u32 v[104:105], s[6:7], s52, v100, 0
	v_add3_u32 v51, v105, v53, v51
	v_sub_co_u32_e32 v39, vcc, v39, v104
	v_cmp_le_u32_e64 s[24:25], s52, v39
	s_nop 0
	v_subb_co_u32_e64 v53, s[22:23], v37, v51, vcc
	v_sub_u32_e32 v37, v37, v51
	v_subb_co_u32_e32 v37, vcc, v37, v57, vcc
	v_subrev_co_u32_e32 v39, vcc, s52, v39
	v_cmp_le_u32_e64 s[22:23], s53, v53
	s_nop 0
	v_subbrev_co_u32_e32 v37, vcc, 0, v37, vcc
	v_cndmask_b32_e64 v55, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v53
	v_cndmask_b32_e64 v53, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v37
	v_cmp_le_u32_e64 s[24:25], s52, v39
	v_cndmask_b32_e64 v53, v55, v53, s[22:23]
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v37
	v_cndmask_b32_e64 v37, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v53
	v_cndmask_b32_e32 v37, v51, v37, vcc
	v_cmp_ne_u32_e32 vcc, 0, v37
	v_xor_b32_e32 v39, s64, v102
	v_ashrrev_i32_e32 v104, 31, v99
	v_cndmask_b32_e64 v37, 1, 2, vcc
	v_add_u32_e32 v37, v100, v37
	v_cndmask_b32_e64 v37, v100, v37, s[22:23]
	v_mov_b32_e32 v99, v98
	v_xor_b32_e32 v37, v37, v39
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[98:99]
	v_sub_u32_e32 v37, v37, v39
	v_xor_b32_e32 v51, v90, v98
	v_xad_u32 v39, v37, v104, v30
	v_xor_b32_e32 v37, v91, v98
	v_mad_u64_u32 v[90:91], s[6:7], v51, s5, 0
	v_mul_hi_u32 v68, v51, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[102:103], s[6:7], v37, s4, 0
	v_add_co_u32_e32 v53, vcc, v90, v102
	v_mad_u64_u32 v[100:101], s[6:7], v37, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v91, v103, vcc
	v_mul_lo_u32 v39, v39, s33
	s_nop 0
	v_addc_co_u32_e32 v101, vcc, 0, v101, vcc
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[100:101]
	v_mul_lo_u32 v53, s53, v90
	v_mul_lo_u32 v55, s52, v91
	v_mad_u64_u32 v[90:91], s[6:7], s52, v90, 0
	v_add3_u32 v53, v91, v55, v53
	v_sub_u32_e32 v55, v37, v53
	v_sub_co_u32_e32 v51, vcc, v51, v90
	s_mov_b64 s[6:7], 0x802
	s_nop 0
	v_subb_co_u32_e64 v55, s[22:23], v55, v57, vcc
	v_subrev_co_u32_e64 v59, s[22:23], s52, v51
	v_cmp_le_u32_e64 s[26:27], s52, v59
	s_nop 0
	v_subbrev_co_u32_e64 v61, s[24:25], 0, v55, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v61
	v_cndmask_b32_e64 v83, 0, -1, s[26:27]
	v_subb_co_u32_e64 v55, s[22:23], v55, v57, s[22:23]
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v61
	v_subb_co_u32_e32 v37, vcc, v37, v53, vcc
	s_nop 0
	v_cndmask_b32_e64 v68, v68, v83, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v68
	v_subrev_co_u32_e64 v68, s[22:23], s52, v59
	v_cmp_le_u32_e32 vcc, s53, v37
	s_nop 0
	v_subbrev_co_u32_e64 v55, s[22:23], 0, v55, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v51
	v_cndmask_b32_e64 v55, v61, v55, s[24:25]
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v37
	v_cndmask_b32_e64 v61, 0, -1, s[22:23]
	v_lshl_add_u64 v[90:91], v[86:87], 0, s[6:7]
	v_cndmask_b32_e32 v53, v53, v61, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_cndmask_b32_e64 v53, v59, v68, s[24:25]
	s_movk_i32 s6, 0xf7fd
	v_cndmask_b32_e32 v51, v51, v53, vcc
	v_cndmask_b32_e32 v37, v37, v55, vcc
	v_xor_b32_e32 v51, v51, v98
	v_xor_b32_e32 v37, v37, v98
	v_sub_co_u32_e32 v68, vcc, v51, v98
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[126:129], v[16:19], v[0:3], v93, v35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_subb_co_u32_e32 v37, vcc, v37, v98, vcc
	v_ashrrev_i32_e32 v37, 31, v37
	v_and_b32_e32 v37, s42, v37
	v_add3_u32 v37, v37, v68, v39
	v_sub_co_u32_e32 v39, vcc, s6, v86
	v_cndmask_b32_e64 v98, v90, v39, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v51, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v99, v91, v51, s[18:19]
	v_ashrrev_i32_e32 v100, 31, v99
	v_mov_b32_e32 v101, v100
	v_lshl_add_u64 v[98:99], v[98:99], 0, v[100:101]
	v_xor_b32_e32 v51, v98, v100
	v_xor_b32_e32 v39, v99, v100
	v_mad_u64_u32 v[98:99], s[6:7], v51, s5, 0
	v_mul_hi_u32 v68, v51, s4
	v_lshl_add_u64 v[98:99], v[68:69], 0, v[98:99]
	v_mad_u64_u32 v[106:107], s[6:7], v39, s4, 0
	v_add_co_u32_e32 v53, vcc, v98, v106
	v_mad_u64_u32 v[102:103], s[6:7], v39, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v99, v107, vcc
	buffer_load_ubyte v33, v33, s[28:31], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[98:99], v[68:69], 0, v[102:103]
	v_mul_lo_u32 v53, s53, v98
	v_mul_lo_u32 v55, s52, v99
	v_mad_u64_u32 v[102:103], s[6:7], s52, v98, 0
	v_add3_u32 v53, v103, v55, v53
	v_sub_co_u32_e32 v51, vcc, v51, v102
	v_cmp_le_u32_e64 s[24:25], s52, v51
	s_nop 0
	v_subb_co_u32_e64 v55, s[22:23], v39, v53, vcc
	v_sub_u32_e32 v39, v39, v53
	v_subb_co_u32_e32 v39, vcc, v39, v57, vcc
	v_subrev_co_u32_e32 v51, vcc, s52, v51
	v_cmp_le_u32_e64 s[22:23], s53, v55
	s_nop 0
	v_subbrev_co_u32_e32 v39, vcc, 0, v39, vcc
	v_cndmask_b32_e64 v59, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v55
	v_cndmask_b32_e64 v55, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v39
	v_cmp_le_u32_e64 s[24:25], s52, v51
	v_cndmask_b32_e64 v55, v59, v55, s[22:23]
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v39
	v_cndmask_b32_e64 v39, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v55
	v_cndmask_b32_e32 v39, v53, v39, vcc
	v_cmp_ne_u32_e32 vcc, 0, v39
	v_xor_b32_e32 v51, s64, v100
	buffer_load_ubyte v37, v37, s[28:31], 0 offen
	v_cndmask_b32_e64 v39, 1, 2, vcc
	v_add_u32_e32 v39, v98, v39
	v_cndmask_b32_e64 v39, v98, v39, s[22:23]
	v_ashrrev_i32_e32 v98, 31, v91
	v_mov_b32_e32 v99, v98
	v_xor_b32_e32 v39, v39, v51
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[98:99]
	v_sub_u32_e32 v39, v39, v51
	v_xor_b32_e32 v53, v90, v98
	v_xad_u32 v51, v39, v104, v30
	v_xor_b32_e32 v39, v91, v98
	v_mad_u64_u32 v[90:91], s[6:7], v53, s5, 0
	v_mul_hi_u32 v68, v53, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[102:103], s[6:7], v39, s4, 0
	v_add_co_u32_e32 v55, vcc, v90, v102
	v_mad_u64_u32 v[100:101], s[6:7], v39, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v91, v103, vcc
	v_mul_lo_u32 v51, v51, s33
	s_nop 0
	v_addc_co_u32_e32 v101, vcc, 0, v101, vcc
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[100:101]
	v_mul_lo_u32 v55, s53, v90
	v_mul_lo_u32 v59, s52, v91
	v_mad_u64_u32 v[90:91], s[6:7], s52, v90, 0
	v_add3_u32 v55, v91, v59, v55
	v_sub_u32_e32 v59, v39, v55
	v_sub_co_u32_e32 v53, vcc, v53, v90
	s_mov_b64 s[6:7], 0x803
	s_nop 0
	v_subb_co_u32_e64 v59, s[22:23], v59, v57, vcc
	v_subrev_co_u32_e64 v61, s[22:23], s52, v53
	v_cmp_le_u32_e64 s[26:27], s52, v61
	s_nop 0
	v_subbrev_co_u32_e64 v68, s[24:25], 0, v59, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v68
	v_cndmask_b32_e64 v90, 0, -1, s[26:27]
	v_subb_co_u32_e64 v59, s[22:23], v59, v57, s[22:23]
	v_cndmask_b32_e64 v83, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v68
	v_subb_co_u32_e32 v39, vcc, v39, v55, vcc
	s_nop 0
	v_cndmask_b32_e64 v83, v83, v90, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v83
	v_subrev_co_u32_e64 v83, s[22:23], s52, v61
	v_cmp_le_u32_e32 vcc, s53, v39
	s_nop 0
	v_subbrev_co_u32_e64 v59, s[22:23], 0, v59, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v53
	v_cndmask_b32_e64 v59, v68, v59, s[24:25]
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v39
	v_cndmask_b32_e64 v68, 0, -1, s[22:23]
	v_lshl_add_u64 v[90:91], v[86:87], 0, s[6:7]
	v_cndmask_b32_e32 v55, v55, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v55
	v_cndmask_b32_e64 v55, v61, v83, s[24:25]
	s_movk_i32 s6, 0xf7fc
	v_cndmask_b32_e32 v53, v53, v55, vcc
	v_cndmask_b32_e32 v39, v39, v59, vcc
	v_xor_b32_e32 v53, v53, v98
	v_xor_b32_e32 v39, v39, v98
	v_sub_co_u32_e32 v68, vcc, v53, v98
	s_nop 1
	v_subb_co_u32_e32 v39, vcc, v39, v98, vcc
	v_ashrrev_i32_e32 v39, 31, v39
	v_and_b32_e32 v39, s42, v39
	v_add3_u32 v39, v39, v68, v51
	v_sub_co_u32_e32 v51, vcc, s6, v86
	v_cndmask_b32_e64 v98, v90, v51, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v53, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v99, v91, v53, s[18:19]
	v_ashrrev_i32_e32 v100, 31, v99
	v_mov_b32_e32 v101, v100
	v_lshl_add_u64 v[98:99], v[98:99], 0, v[100:101]
	v_xor_b32_e32 v53, v98, v100
	v_xor_b32_e32 v51, v99, v100
	v_mad_u64_u32 v[98:99], s[6:7], v53, s5, 0
	v_mul_hi_u32 v68, v53, s4
	v_lshl_add_u64 v[98:99], v[68:69], 0, v[98:99]
	v_mad_u64_u32 v[106:107], s[6:7], v51, s4, 0
	v_add_co_u32_e32 v55, vcc, v98, v106
	v_mad_u64_u32 v[102:103], s[6:7], v51, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v99, v107, vcc
	buffer_load_ubyte v39, v39, s[28:31], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[98:99], v[68:69], 0, v[102:103]
	v_mul_lo_u32 v55, s53, v98
	v_mul_lo_u32 v59, s52, v99
	v_mad_u64_u32 v[102:103], s[6:7], s52, v98, 0
	v_add3_u32 v55, v103, v59, v55
	v_sub_co_u32_e32 v53, vcc, v53, v102
	v_cmp_le_u32_e64 s[24:25], s52, v53
	s_nop 0
	v_subb_co_u32_e64 v59, s[22:23], v51, v55, vcc
	v_sub_u32_e32 v51, v51, v55
	v_subb_co_u32_e32 v51, vcc, v51, v57, vcc
	v_subrev_co_u32_e32 v53, vcc, s52, v53
	v_cmp_le_u32_e64 s[22:23], s53, v59
	s_nop 0
	v_subbrev_co_u32_e32 v51, vcc, 0, v51, vcc
	v_cndmask_b32_e64 v61, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v59
	v_cndmask_b32_e64 v59, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v51
	v_cmp_le_u32_e64 s[24:25], s52, v53
	v_cndmask_b32_e64 v59, v61, v59, s[22:23]
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v51
	v_cndmask_b32_e64 v51, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v59
	v_cndmask_b32_e32 v51, v55, v51, vcc
	v_cmp_ne_u32_e32 vcc, 0, v51
	v_xor_b32_e32 v53, s64, v100
	s_nop 0
	v_cndmask_b32_e64 v51, 1, 2, vcc
	v_add_u32_e32 v51, v98, v51
	v_cndmask_b32_e64 v51, v98, v51, s[22:23]
	v_ashrrev_i32_e32 v98, 31, v91
	v_mov_b32_e32 v99, v98
	v_xor_b32_e32 v51, v51, v53
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[98:99]
	v_sub_u32_e32 v51, v51, v53
	v_xor_b32_e32 v55, v90, v98
	v_xad_u32 v53, v51, v104, v30
	v_xor_b32_e32 v51, v91, v98
	v_mad_u64_u32 v[90:91], s[6:7], v55, s5, 0
	v_mul_hi_u32 v68, v55, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[102:103], s[6:7], v51, s4, 0
	v_add_co_u32_e32 v59, vcc, v90, v102
	v_mad_u64_u32 v[100:101], s[6:7], v51, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v91, v103, vcc
	v_mul_lo_u32 v53, v53, s33
	s_nop 0
	v_addc_co_u32_e32 v101, vcc, 0, v101, vcc
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[100:101]
	v_mul_lo_u32 v59, s53, v90
	v_mul_lo_u32 v61, s52, v91
	v_mad_u64_u32 v[90:91], s[6:7], s52, v90, 0
	v_add3_u32 v59, v91, v61, v59
	v_sub_u32_e32 v61, v51, v59
	v_sub_co_u32_e32 v55, vcc, v55, v90
	s_mov_b64 s[6:7], 0x804
	s_nop 0
	v_subb_co_u32_e64 v61, s[22:23], v61, v57, vcc
	v_subrev_co_u32_e64 v68, s[22:23], s52, v55
	v_cmp_le_u32_e64 s[26:27], s52, v68
	s_nop 0
	v_subbrev_co_u32_e64 v83, s[24:25], 0, v61, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v83
	v_cndmask_b32_e64 v91, 0, -1, s[26:27]
	v_subb_co_u32_e64 v61, s[22:23], v61, v57, s[22:23]
	v_cndmask_b32_e64 v90, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v83
	v_subb_co_u32_e32 v51, vcc, v51, v59, vcc
	s_nop 0
	v_cndmask_b32_e64 v90, v90, v91, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v90
	v_subrev_co_u32_e64 v90, s[22:23], s52, v68
	v_cmp_le_u32_e32 vcc, s53, v51
	s_nop 0
	v_subbrev_co_u32_e64 v61, s[22:23], 0, v61, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v55
	v_cndmask_b32_e64 v61, v83, v61, s[24:25]
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v51
	v_cndmask_b32_e64 v83, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v59, v59, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v59
	v_cndmask_b32_e64 v59, v68, v90, s[24:25]
	v_lshl_add_u64 v[90:91], v[86:87], 0, s[6:7]
	v_cndmask_b32_e32 v55, v55, v59, vcc
	v_cndmask_b32_e32 v51, v51, v61, vcc
	v_xor_b32_e32 v55, v55, v98
	v_xor_b32_e32 v51, v51, v98
	v_sub_co_u32_e32 v68, vcc, v55, v98
	s_movk_i32 s6, 0xf7fb
	s_nop 0
	v_subb_co_u32_e32 v51, vcc, v51, v98, vcc
	v_ashrrev_i32_e32 v51, 31, v51
	v_and_b32_e32 v51, s42, v51
	v_add3_u32 v51, v51, v68, v53
	v_sub_co_u32_e32 v53, vcc, s6, v86
	v_cndmask_b32_e64 v98, v90, v53, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v55, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v99, v91, v55, s[18:19]
	v_ashrrev_i32_e32 v100, 31, v99
	v_mov_b32_e32 v101, v100
	v_lshl_add_u64 v[98:99], v[98:99], 0, v[100:101]
	v_xor_b32_e32 v55, v98, v100
	v_xor_b32_e32 v53, v99, v100
	v_mad_u64_u32 v[98:99], s[6:7], v55, s5, 0
	v_mul_hi_u32 v68, v55, s4
	v_lshl_add_u64 v[98:99], v[68:69], 0, v[98:99]
	v_mad_u64_u32 v[106:107], s[6:7], v53, s4, 0
	v_add_co_u32_e32 v59, vcc, v98, v106
	v_mad_u64_u32 v[102:103], s[6:7], v53, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v99, v107, vcc
	buffer_load_ubyte v51, v51, s[28:31], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[98:99], v[68:69], 0, v[102:103]
	v_mul_lo_u32 v59, s53, v98
	v_mul_lo_u32 v61, s52, v99
	v_mad_u64_u32 v[102:103], s[6:7], s52, v98, 0
	v_add3_u32 v59, v103, v61, v59
	v_sub_co_u32_e32 v55, vcc, v55, v102
	v_cmp_le_u32_e64 s[24:25], s52, v55
	s_nop 0
	v_subb_co_u32_e64 v61, s[22:23], v53, v59, vcc
	v_sub_u32_e32 v53, v53, v59
	v_subb_co_u32_e32 v53, vcc, v53, v57, vcc
	v_subrev_co_u32_e32 v55, vcc, s52, v55
	v_cmp_le_u32_e64 s[22:23], s53, v61
	s_nop 0
	v_subbrev_co_u32_e32 v53, vcc, 0, v53, vcc
	v_cndmask_b32_e64 v68, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v61
	v_cndmask_b32_e64 v61, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v53
	v_cmp_le_u32_e64 s[24:25], s52, v55
	v_cndmask_b32_e64 v61, v68, v61, s[22:23]
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v53
	v_cndmask_b32_e64 v53, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v61
	v_cndmask_b32_e32 v53, v59, v53, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_xor_b32_e32 v55, s64, v100
	s_nop 0
	v_cndmask_b32_e64 v53, 1, 2, vcc
	v_add_u32_e32 v53, v98, v53
	v_cndmask_b32_e64 v53, v98, v53, s[22:23]
	v_ashrrev_i32_e32 v98, 31, v91
	v_mov_b32_e32 v99, v98
	v_xor_b32_e32 v53, v53, v55
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[98:99]
	v_sub_u32_e32 v53, v53, v55
	v_xor_b32_e32 v59, v90, v98
	v_xad_u32 v55, v53, v104, v30
	v_xor_b32_e32 v53, v91, v98
	v_mad_u64_u32 v[90:91], s[6:7], v59, s5, 0
	v_mul_hi_u32 v68, v59, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[102:103], s[6:7], v53, s4, 0
	v_add_co_u32_e32 v61, vcc, v90, v102
	v_mad_u64_u32 v[100:101], s[6:7], v53, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v91, v103, vcc
	v_mul_lo_u32 v55, v55, s33
	s_nop 0
	v_addc_co_u32_e32 v101, vcc, 0, v101, vcc
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[100:101]
	v_mul_lo_u32 v61, s53, v90
	v_mul_lo_u32 v68, s52, v91
	v_mad_u64_u32 v[90:91], s[6:7], s52, v90, 0
	v_add3_u32 v61, v91, v68, v61
	v_sub_u32_e32 v68, v53, v61
	v_sub_co_u32_e32 v59, vcc, v59, v90
	s_mov_b64 s[6:7], 0x805
	s_nop 0
	v_subb_co_u32_e64 v68, s[22:23], v68, v57, vcc
	v_subrev_co_u32_e64 v83, s[22:23], s52, v59
	v_cmp_le_u32_e64 s[26:27], s52, v83
	s_nop 0
	v_subbrev_co_u32_e64 v90, s[24:25], 0, v68, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v90
	v_cndmask_b32_e64 v99, 0, -1, s[26:27]
	v_subb_co_u32_e64 v68, s[22:23], v68, v57, s[22:23]
	v_cndmask_b32_e64 v91, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v90
	v_subb_co_u32_e32 v53, vcc, v53, v61, vcc
	s_nop 0
	v_cndmask_b32_e64 v91, v91, v99, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v91
	v_subrev_co_u32_e64 v91, s[22:23], s52, v83
	v_cmp_le_u32_e32 vcc, s53, v53
	s_nop 0
	v_subbrev_co_u32_e64 v68, s[22:23], 0, v68, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v59
	v_cndmask_b32_e64 v68, v90, v68, s[24:25]
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v53
	v_cndmask_b32_e64 v90, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v61, v61, v90, vcc
	v_cmp_ne_u32_e32 vcc, 0, v61
	v_cndmask_b32_e64 v61, v83, v91, s[24:25]
	v_lshl_add_u64 v[90:91], v[86:87], 0, s[6:7]
	v_cndmask_b32_e32 v59, v59, v61, vcc
	v_cndmask_b32_e32 v53, v53, v68, vcc
	v_xor_b32_e32 v59, v59, v98
	v_xor_b32_e32 v53, v53, v98
	v_sub_co_u32_e32 v68, vcc, v59, v98
	s_movk_i32 s6, 0xf7fa
	s_nop 0
	v_subb_co_u32_e32 v53, vcc, v53, v98, vcc
	v_ashrrev_i32_e32 v53, 31, v53
	v_and_b32_e32 v53, s42, v53
	v_add3_u32 v53, v53, v68, v55
	v_sub_co_u32_e32 v55, vcc, s6, v86
	v_cndmask_b32_e64 v98, v90, v55, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v59, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v99, v91, v59, s[18:19]
	v_ashrrev_i32_e32 v100, 31, v99
	v_mov_b32_e32 v101, v100
	v_lshl_add_u64 v[98:99], v[98:99], 0, v[100:101]
	v_xor_b32_e32 v59, v98, v100
	v_xor_b32_e32 v55, v99, v100
	v_mad_u64_u32 v[98:99], s[6:7], v59, s5, 0
	v_mul_hi_u32 v68, v59, s4
	v_lshl_add_u64 v[98:99], v[68:69], 0, v[98:99]
	v_mad_u64_u32 v[106:107], s[6:7], v55, s4, 0
	v_add_co_u32_e32 v61, vcc, v98, v106
	v_mad_u64_u32 v[102:103], s[6:7], v55, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v99, v107, vcc
	buffer_load_ubyte v53, v53, s[28:31], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[98:99], v[68:69], 0, v[102:103]
	v_mul_lo_u32 v61, s53, v98
	v_mul_lo_u32 v68, s52, v99
	v_mad_u64_u32 v[102:103], s[6:7], s52, v98, 0
	v_add3_u32 v61, v103, v68, v61
	v_sub_co_u32_e32 v59, vcc, v59, v102
	v_cmp_le_u32_e64 s[24:25], s52, v59
	s_nop 0
	v_subb_co_u32_e64 v68, s[22:23], v55, v61, vcc
	v_sub_u32_e32 v55, v55, v61
	v_subb_co_u32_e32 v55, vcc, v55, v57, vcc
	v_subrev_co_u32_e32 v59, vcc, s52, v59
	v_cmp_le_u32_e64 s[22:23], s53, v68
	s_nop 0
	v_subbrev_co_u32_e32 v55, vcc, 0, v55, vcc
	v_cndmask_b32_e64 v83, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v55
	v_cmp_le_u32_e64 s[24:25], s52, v59
	v_cndmask_b32_e64 v68, v83, v68, s[22:23]
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v55
	v_cndmask_b32_e64 v55, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v68
	v_cndmask_b32_e32 v55, v61, v55, vcc
	v_cmp_ne_u32_e32 vcc, 0, v55
	v_xor_b32_e32 v59, s64, v100
	s_nop 0
	v_cndmask_b32_e64 v55, 1, 2, vcc
	v_add_u32_e32 v55, v98, v55
	v_cndmask_b32_e64 v55, v98, v55, s[22:23]
	v_ashrrev_i32_e32 v98, 31, v91
	v_mov_b32_e32 v99, v98
	v_xor_b32_e32 v55, v55, v59
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[98:99]
	v_sub_u32_e32 v55, v55, v59
	v_xor_b32_e32 v61, v90, v98
	v_xad_u32 v59, v55, v104, v30
	v_xor_b32_e32 v55, v91, v98
	v_mad_u64_u32 v[90:91], s[6:7], v61, s5, 0
	v_mul_hi_u32 v68, v61, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[102:103], s[6:7], v55, s4, 0
	v_add_co_u32_e32 v68, vcc, v90, v102
	v_mad_u64_u32 v[100:101], s[6:7], v55, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v91, v103, vcc
	v_mul_lo_u32 v59, v59, s33
	s_nop 0
	v_addc_co_u32_e32 v101, vcc, 0, v101, vcc
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[100:101]
	v_mul_lo_u32 v68, s53, v90
	v_mul_lo_u32 v83, s52, v91
	v_mad_u64_u32 v[90:91], s[6:7], s52, v90, 0
	v_add3_u32 v68, v91, v83, v68
	v_sub_u32_e32 v83, v55, v68
	v_sub_co_u32_e32 v61, vcc, v61, v90
	s_mov_b64 s[6:7], 0x806
	s_nop 0
	v_subb_co_u32_e64 v83, s[22:23], v83, v57, vcc
	v_subrev_co_u32_e64 v90, s[22:23], s52, v61
	v_cmp_le_u32_e64 s[26:27], s52, v90
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[24:25], 0, v83, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v91
	v_cndmask_b32_e64 v100, 0, -1, s[26:27]
	v_subb_co_u32_e64 v83, s[22:23], v83, v57, s[22:23]
	v_cndmask_b32_e64 v99, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v91
	v_subb_co_u32_e32 v55, vcc, v55, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v99, v99, v100, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v99
	v_subrev_co_u32_e64 v99, s[22:23], s52, v90
	v_cmp_le_u32_e32 vcc, s53, v55
	s_nop 0
	v_subbrev_co_u32_e64 v83, s[22:23], 0, v83, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v61
	v_cndmask_b32_e64 v83, v91, v83, s[24:25]
	v_cndmask_b32_e64 v68, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v55
	v_cndmask_b32_e64 v91, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v68, v68, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_cndmask_b32_e64 v68, v90, v99, s[24:25]
	v_lshl_add_u64 v[90:91], v[86:87], 0, s[6:7]
	v_cndmask_b32_e32 v61, v61, v68, vcc
	v_cndmask_b32_e32 v55, v55, v83, vcc
	v_xor_b32_e32 v61, v61, v98
	v_xor_b32_e32 v55, v55, v98
	v_sub_co_u32_e32 v68, vcc, v61, v98
	s_movk_i32 s6, 0xf7f9
	s_nop 0
	v_subb_co_u32_e32 v55, vcc, v55, v98, vcc
	v_ashrrev_i32_e32 v55, 31, v55
	v_and_b32_e32 v55, s42, v55
	v_add3_u32 v55, v55, v68, v59
	v_sub_co_u32_e32 v59, vcc, s6, v86
	v_cndmask_b32_e64 v98, v90, v59, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v61, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v99, v91, v61, s[18:19]
	v_ashrrev_i32_e32 v100, 31, v99
	v_mov_b32_e32 v101, v100
	v_lshl_add_u64 v[98:99], v[98:99], 0, v[100:101]
	v_xor_b32_e32 v61, v98, v100
	v_xor_b32_e32 v59, v99, v100
	v_mad_u64_u32 v[98:99], s[6:7], v61, s5, 0
	v_mul_hi_u32 v68, v61, s4
	v_lshl_add_u64 v[98:99], v[68:69], 0, v[98:99]
	v_mad_u64_u32 v[106:107], s[6:7], v59, s4, 0
	v_add_co_u32_e32 v68, vcc, v98, v106
	v_mad_u64_u32 v[102:103], s[6:7], v59, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v99, v107, vcc
	buffer_load_ubyte v55, v55, s[28:31], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[98:99], v[68:69], 0, v[102:103]
	v_mul_lo_u32 v68, s53, v98
	v_mul_lo_u32 v83, s52, v99
	v_mad_u64_u32 v[102:103], s[6:7], s52, v98, 0
	v_add3_u32 v68, v103, v83, v68
	v_sub_co_u32_e32 v61, vcc, v61, v102
	v_cmp_le_u32_e64 s[24:25], s52, v61
	s_nop 0
	v_subb_co_u32_e64 v83, s[22:23], v59, v68, vcc
	v_sub_u32_e32 v59, v59, v68
	v_subb_co_u32_e32 v59, vcc, v59, v57, vcc
	v_subrev_co_u32_e32 v61, vcc, s52, v61
	v_cmp_le_u32_e64 s[22:23], s53, v83
	s_nop 0
	v_subbrev_co_u32_e32 v59, vcc, 0, v59, vcc
	v_cndmask_b32_e64 v99, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v83
	v_cndmask_b32_e64 v83, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v59
	v_cmp_le_u32_e64 s[24:25], s52, v61
	v_cndmask_b32_e64 v83, v99, v83, s[22:23]
	v_cndmask_b32_e64 v68, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v59
	v_cndmask_b32_e64 v59, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v83
	v_cndmask_b32_e32 v59, v68, v59, vcc
	v_cmp_ne_u32_e32 vcc, 0, v59
	v_xor_b32_e32 v61, s64, v100
	s_nop 0
	v_cndmask_b32_e64 v59, 1, 2, vcc
	v_add_u32_e32 v59, v98, v59
	v_cndmask_b32_e64 v59, v98, v59, s[22:23]
	v_ashrrev_i32_e32 v98, 31, v91
	v_mov_b32_e32 v99, v98
	v_xor_b32_e32 v59, v59, v61
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[98:99]
	v_sub_u32_e32 v59, v59, v61
	v_xor_b32_e32 v83, v90, v98
	v_xad_u32 v61, v59, v104, v30
	v_xor_b32_e32 v59, v91, v98
	v_mad_u64_u32 v[90:91], s[6:7], v83, s5, 0
	v_mul_hi_u32 v68, v83, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[102:103], s[6:7], v59, s4, 0
	v_add_co_u32_e32 v68, vcc, v90, v102
	v_mad_u64_u32 v[100:101], s[6:7], v59, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v91, v103, vcc
	v_mul_lo_u32 v61, v61, s33
	s_nop 0
	v_addc_co_u32_e32 v101, vcc, 0, v101, vcc
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[100:101]
	v_mul_lo_u32 v68, s53, v90
	v_mul_lo_u32 v99, s52, v91
	v_mad_u64_u32 v[90:91], s[6:7], s52, v90, 0
	v_add3_u32 v68, v91, v99, v68
	v_sub_u32_e32 v91, v59, v68
	v_sub_co_u32_e32 v83, vcc, v83, v90
	s_mov_b64 s[6:7], 0x807
	s_nop 0
	v_subb_co_u32_e64 v90, s[22:23], v91, v57, vcc
	v_subrev_co_u32_e64 v91, s[22:23], s52, v83
	v_cmp_le_u32_e64 s[26:27], s52, v91
	s_nop 0
	v_subbrev_co_u32_e64 v99, s[24:25], 0, v90, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v99
	v_cndmask_b32_e64 v101, 0, -1, s[26:27]
	v_subb_co_u32_e64 v90, s[22:23], v90, v57, s[22:23]
	v_cndmask_b32_e64 v100, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v99
	v_subb_co_u32_e32 v59, vcc, v59, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v100, v100, v101, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v100
	v_subrev_co_u32_e64 v100, s[22:23], s52, v91
	v_cmp_le_u32_e32 vcc, s53, v59
	s_nop 0
	v_subbrev_co_u32_e64 v90, s[22:23], 0, v90, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v83
	v_cndmask_b32_e64 v90, v99, v90, s[24:25]
	v_cndmask_b32_e64 v68, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v59
	v_cndmask_b32_e64 v99, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v68, v68, v99, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_cndmask_b32_e64 v68, v91, v100, s[24:25]
	s_nop 0
	v_cndmask_b32_e32 v68, v83, v68, vcc
	v_cndmask_b32_e32 v59, v59, v90, vcc
	v_xor_b32_e32 v68, v68, v98
	v_xor_b32_e32 v59, v59, v98
	v_sub_co_u32_e32 v68, vcc, v68, v98
	v_lshl_add_u64 v[90:91], v[86:87], 0, s[6:7]
	s_nop 0
	v_subb_co_u32_e32 v59, vcc, v59, v98, vcc
	v_ashrrev_i32_e32 v59, 31, v59
	v_and_b32_e32 v59, s42, v59
	s_movk_i32 s6, 0xf7f8
	v_add3_u32 v59, v59, v68, v61
	v_sub_co_u32_e32 v61, vcc, s6, v86
	v_cndmask_b32_e64 v98, v90, v61, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v68, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v99, v91, v68, s[18:19]
	v_ashrrev_i32_e32 v100, 31, v99
	v_mov_b32_e32 v101, v100
	v_lshl_add_u64 v[98:99], v[98:99], 0, v[100:101]
	v_xor_b32_e32 v83, v98, v100
	v_xor_b32_e32 v61, v99, v100
	v_mad_u64_u32 v[98:99], s[6:7], v83, s5, 0
	v_mul_hi_u32 v68, v83, s4
	v_lshl_add_u64 v[98:99], v[68:69], 0, v[98:99]
	v_mad_u64_u32 v[106:107], s[6:7], v61, s4, 0
	v_add_co_u32_e32 v68, vcc, v98, v106
	v_mad_u64_u32 v[102:103], s[6:7], v61, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v99, v107, vcc
	buffer_load_ubyte v59, v59, s[28:31], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[98:99], v[68:69], 0, v[102:103]
	v_mul_lo_u32 v68, s53, v98
	v_mul_lo_u32 v99, s52, v99
	v_mad_u64_u32 v[102:103], s[6:7], s52, v98, 0
	v_add3_u32 v68, v103, v99, v68
	v_sub_co_u32_e32 v83, vcc, v83, v102
	v_cmp_le_u32_e64 s[24:25], s52, v83
	s_nop 0
	v_subb_co_u32_e64 v99, s[22:23], v61, v68, vcc
	v_sub_u32_e32 v61, v61, v68
	v_subb_co_u32_e32 v61, vcc, v61, v57, vcc
	v_subrev_co_u32_e32 v68, vcc, s52, v83
	v_cmp_le_u32_e64 s[22:23], s53, v99
	s_nop 0
	v_subbrev_co_u32_e32 v61, vcc, 0, v61, vcc
	v_cndmask_b32_e64 v101, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v99
	v_cndmask_b32_e64 v99, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v61
	v_cmp_le_u32_e64 s[24:25], s52, v68
	v_cndmask_b32_e64 v99, v101, v99, s[22:23]
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v61
	v_cndmask_b32_e64 v61, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v99
	v_cndmask_b32_e32 v61, v83, v61, vcc
	v_cmp_ne_u32_e32 vcc, 0, v61
	v_xor_b32_e32 v68, s64, v100
	s_nop 0
	v_cndmask_b32_e64 v61, 1, 2, vcc
	v_add_u32_e32 v61, v98, v61
	v_cndmask_b32_e64 v61, v98, v61, s[22:23]
	v_ashrrev_i32_e32 v98, 31, v91
	v_mov_b32_e32 v99, v98
	v_xor_b32_e32 v61, v61, v68
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[98:99]
	v_sub_u32_e32 v61, v61, v68
	v_xor_b32_e32 v99, v90, v98
	v_xad_u32 v83, v61, v104, v30
	v_xor_b32_e32 v61, v91, v98
	v_mad_u64_u32 v[90:91], s[6:7], v99, s5, 0
	v_mul_hi_u32 v68, v99, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[102:103], s[6:7], v61, s4, 0
	v_add_co_u32_e32 v68, vcc, v90, v102
	v_mad_u64_u32 v[100:101], s[6:7], v61, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v91, v103, vcc
	v_mul_lo_u32 v83, v83, s33
	s_nop 0
	v_addc_co_u32_e32 v101, vcc, 0, v101, vcc
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[100:101]
	v_mul_lo_u32 v68, s53, v90
	v_mul_lo_u32 v100, s52, v91
	v_mad_u64_u32 v[90:91], s[6:7], s52, v90, 0
	v_add3_u32 v68, v91, v100, v68
	v_sub_u32_e32 v91, v61, v68
	v_sub_co_u32_e32 v90, vcc, v99, v90
	s_mov_b64 s[6:7], 0x808
	s_nop 0
	v_subb_co_u32_e64 v91, s[22:23], v91, v57, vcc
	v_subrev_co_u32_e64 v99, s[22:23], s52, v90
	v_cmp_le_u32_e64 s[26:27], s52, v99
	s_nop 0
	v_subbrev_co_u32_e64 v100, s[24:25], 0, v91, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v100
	v_cndmask_b32_e64 v102, 0, -1, s[26:27]
	v_subb_co_u32_e64 v91, s[22:23], v91, v57, s[22:23]
	v_cndmask_b32_e64 v101, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v100
	v_subb_co_u32_e32 v61, vcc, v61, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v101, v101, v102, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v101
	v_subrev_co_u32_e64 v101, s[22:23], s52, v99
	v_cmp_le_u32_e32 vcc, s53, v61
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[22:23], 0, v91, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v90
	v_cndmask_b32_e64 v91, v100, v91, s[24:25]
	v_cndmask_b32_e64 v68, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v61
	v_cndmask_b32_e64 v100, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v68, v68, v100, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_cndmask_b32_e64 v68, v99, v101, s[24:25]
	s_nop 0
	v_cndmask_b32_e32 v68, v90, v68, vcc
	v_cndmask_b32_e32 v61, v61, v91, vcc
	v_xor_b32_e32 v68, v68, v98
	v_xor_b32_e32 v61, v61, v98
	v_sub_co_u32_e32 v68, vcc, v68, v98
	v_lshl_add_u64 v[90:91], v[86:87], 0, s[6:7]
	s_nop 0
	v_subb_co_u32_e32 v61, vcc, v61, v98, vcc
	v_ashrrev_i32_e32 v61, 31, v61
	v_and_b32_e32 v61, s42, v61
	s_movk_i32 s6, 0xf7f7
	v_add3_u32 v61, v61, v68, v83
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cndmask_b32_e64 v98, v90, v68, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v83, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v99, v91, v83, s[18:19]
	v_ashrrev_i32_e32 v100, 31, v99
	v_mov_b32_e32 v101, v100
	v_lshl_add_u64 v[98:99], v[98:99], 0, v[100:101]
	v_xor_b32_e32 v101, v98, v100
	v_xor_b32_e32 v83, v99, v100
	v_mad_u64_u32 v[98:99], s[6:7], v101, s5, 0
	v_mul_hi_u32 v68, v101, s4
	v_lshl_add_u64 v[98:99], v[68:69], 0, v[98:99]
	v_mad_u64_u32 v[106:107], s[6:7], v83, s4, 0
	v_add_co_u32_e32 v68, vcc, v98, v106
	v_mad_u64_u32 v[102:103], s[6:7], v83, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v99, v107, vcc
	buffer_load_ubyte v61, v61, s[28:31], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[98:99], v[68:69], 0, v[102:103]
	v_mul_lo_u32 v68, s53, v98
	v_mul_lo_u32 v99, s52, v99
	v_mad_u64_u32 v[102:103], s[6:7], s52, v98, 0
	v_add3_u32 v68, v103, v99, v68
	v_sub_co_u32_e32 v99, vcc, v101, v102
	v_cmp_le_u32_e64 s[24:25], s52, v99
	s_nop 0
	v_subb_co_u32_e64 v101, s[22:23], v83, v68, vcc
	v_sub_u32_e32 v68, v83, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v83, vcc, s52, v99
	v_cmp_le_u32_e64 s[22:23], s53, v101
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v102, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v101
	v_cndmask_b32_e64 v101, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v83
	v_cndmask_b32_e64 v101, v102, v101, s[22:23]
	v_cndmask_b32_e64 v99, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v101
	v_cndmask_b32_e32 v68, v99, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v83, s64, v100
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v98, v68
	v_cndmask_b32_e64 v68, v98, v68, s[22:23]
	v_ashrrev_i32_e32 v98, 31, v91
	v_mov_b32_e32 v99, v98
	v_xor_b32_e32 v68, v68, v83
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[98:99]
	v_sub_u32_e32 v68, v68, v83
	v_xor_b32_e32 v99, v90, v98
	v_xad_u32 v105, v68, v104, v30
	v_xor_b32_e32 v83, v91, v98
	v_mad_u64_u32 v[90:91], s[6:7], v99, s5, 0
	v_mul_hi_u32 v68, v99, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[102:103], s[6:7], v83, s4, 0
	v_add_co_u32_e32 v68, vcc, v90, v102
	v_mad_u64_u32 v[100:101], s[6:7], v83, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v91, v103, vcc
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v101, vcc
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[100:101]
	v_mul_lo_u32 v68, s53, v90
	v_mul_lo_u32 v100, s52, v91
	v_mad_u64_u32 v[90:91], s[6:7], s52, v90, 0
	v_add3_u32 v68, v91, v100, v68
	v_sub_u32_e32 v91, v83, v68
	v_sub_co_u32_e32 v90, vcc, v99, v90
	s_mov_b64 s[6:7], 0x809
	s_nop 0
	v_subb_co_u32_e64 v91, s[22:23], v91, v57, vcc
	v_subrev_co_u32_e64 v99, s[22:23], s52, v90
	v_cmp_le_u32_e64 s[26:27], s52, v99
	s_nop 0
	v_subbrev_co_u32_e64 v100, s[24:25], 0, v91, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v100
	v_cndmask_b32_e64 v102, 0, -1, s[26:27]
	v_subb_co_u32_e64 v91, s[22:23], v91, v57, s[22:23]
	v_cndmask_b32_e64 v101, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v100
	v_subb_co_u32_e32 v68, vcc, v83, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v101, v101, v102, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v101
	v_subrev_co_u32_e64 v101, s[22:23], s52, v99
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[22:23], 0, v91, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v90
	v_cndmask_b32_e64 v91, v100, v91, s[24:25]
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v100, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v83, v83, v100, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	v_cndmask_b32_e64 v83, v99, v101, s[24:25]
	s_nop 0
	v_cndmask_b32_e32 v68, v68, v91, vcc
	v_cndmask_b32_e32 v83, v90, v83, vcc
	v_xor_b32_e32 v90, v68, v98
	v_xor_b32_e32 v68, v83, v98
	v_sub_co_u32_e32 v68, vcc, v68, v98
	s_nop 1
	v_subb_co_u32_e32 v83, vcc, v90, v98, vcc
	v_ashrrev_i32_e32 v83, 31, v83
	v_and_b32_e32 v83, s42, v83
	v_mul_lo_u32 v90, v105, s33
	v_add3_u32 v68, v83, v68, v90
	v_lshl_add_u64 v[90:91], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xf7f6
	buffer_load_ubyte v83, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	s_nop 1
	v_subb_co_u32_e32 v98, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v99, v91, v98, s[18:19]
	v_ashrrev_i32_e32 v100, 31, v99
	v_cndmask_b32_e64 v98, v90, v68, s[18:19]
	v_mov_b32_e32 v101, v100
	v_lshl_add_u64 v[98:99], v[98:99], 0, v[100:101]
	v_xor_b32_e32 v105, v98, v100
	v_xor_b32_e32 v101, v99, v100
	v_mad_u64_u32 v[98:99], s[6:7], v105, s5, 0
	v_mul_hi_u32 v68, v105, s4
	v_lshl_add_u64 v[98:99], v[68:69], 0, v[98:99]
	v_mad_u64_u32 v[106:107], s[6:7], v101, s4, 0
	v_add_co_u32_e32 v68, vcc, v98, v106
	v_mad_u64_u32 v[102:103], s[6:7], v101, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v99, v107, vcc
	s_nop 1
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[98:99], v[68:69], 0, v[102:103]
	v_mul_lo_u32 v68, s53, v98
	v_mul_lo_u32 v99, s52, v99
	v_mad_u64_u32 v[102:103], s[6:7], s52, v98, 0
	v_add3_u32 v68, v103, v99, v68
	v_sub_co_u32_e32 v99, vcc, v105, v102
	v_cmp_le_u32_e64 s[24:25], s52, v99
	s_nop 0
	v_subb_co_u32_e64 v102, s[22:23], v101, v68, vcc
	v_sub_u32_e32 v68, v101, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v99, vcc, s52, v99
	v_cmp_le_u32_e64 s[22:23], s53, v102
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v103, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v102
	v_cndmask_b32_e64 v102, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v99
	v_cndmask_b32_e64 v102, v103, v102, s[22:23]
	v_cndmask_b32_e64 v101, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v102
	v_cndmask_b32_e32 v68, v101, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v98, v68
	v_cndmask_b32_e64 v68, v98, v68, s[22:23]
	v_xor_b32_e32 v98, s64, v100
	v_xor_b32_e32 v68, v68, v98
	v_sub_u32_e32 v68, v68, v98
	v_ashrrev_i32_e32 v98, 31, v91
	v_mov_b32_e32 v99, v98
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[98:99]
	v_xor_b32_e32 v106, v90, v98
	v_xad_u32 v105, v68, v104, v30
	v_xor_b32_e32 v99, v91, v98
	v_mad_u64_u32 v[90:91], s[6:7], v106, s5, 0
	v_mul_hi_u32 v68, v106, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[102:103], s[6:7], v99, s4, 0
	v_add_co_u32_e32 v68, vcc, v90, v102
	v_mad_u64_u32 v[100:101], s[6:7], v99, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v91, v103, vcc
	s_nop 1
	v_addc_co_u32_e32 v101, vcc, 0, v101, vcc
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[100:101]
	v_mul_lo_u32 v68, s53, v90
	v_mul_lo_u32 v100, s52, v91
	v_mad_u64_u32 v[90:91], s[6:7], s52, v90, 0
	v_add3_u32 v68, v91, v100, v68
	v_sub_u32_e32 v91, v99, v68
	v_sub_co_u32_e32 v90, vcc, v106, v90
	s_mov_b64 s[6:7], 0x80a
	s_nop 0
	v_subb_co_u32_e64 v91, s[22:23], v91, v57, vcc
	v_subrev_co_u32_e64 v100, s[22:23], s52, v90
	v_cmp_le_u32_e64 s[26:27], s52, v100
	s_nop 0
	v_subbrev_co_u32_e64 v101, s[24:25], 0, v91, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v101
	v_cndmask_b32_e64 v103, 0, -1, s[26:27]
	v_subb_co_u32_e64 v91, s[22:23], v91, v57, s[22:23]
	v_cndmask_b32_e64 v102, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v101
	v_subb_co_u32_e32 v68, vcc, v99, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v102, v102, v103, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v102
	v_subrev_co_u32_e64 v102, s[22:23], s52, v100
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[22:23], 0, v91, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v90
	v_cndmask_b32_e64 v91, v101, v91, s[24:25]
	v_cndmask_b32_e64 v99, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v101, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v99, v99, v101, vcc
	v_cmp_ne_u32_e32 vcc, 0, v99
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v91, vcc
	v_cndmask_b32_e64 v91, v100, v102, s[24:25]
	v_cndmask_b32_e32 v90, v90, v91, vcc
	v_xor_b32_e32 v91, v68, v98
	v_xor_b32_e32 v68, v90, v98
	v_sub_co_u32_e32 v68, vcc, v68, v98
	s_nop 1
	v_subb_co_u32_e32 v91, vcc, v91, v98, vcc
	v_ashrrev_i32_e32 v90, 31, v91
	v_and_b32_e32 v90, s42, v90
	v_mul_lo_u32 v91, v105, s33
	v_add3_u32 v68, v90, v68, v91
	v_lshl_add_u64 v[90:91], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xf7f5
	buffer_load_ubyte v98, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cndmask_b32_e64 v100, v90, v68, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v99, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v101, v91, v99, s[18:19]
	v_ashrrev_i32_e32 v102, 31, v101
	v_mov_b32_e32 v103, v102
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[102:103]
	v_xor_b32_e32 v103, v100, v102
	v_xor_b32_e32 v99, v101, v102
	v_mad_u64_u32 v[100:101], s[6:7], v103, s5, 0
	v_mul_hi_u32 v68, v103, s4
	v_lshl_add_u64 v[100:101], v[68:69], 0, v[100:101]
	v_mad_u64_u32 v[108:109], s[6:7], v99, s4, 0
	v_add_co_u32_e32 v68, vcc, v100, v108
	v_mad_u64_u32 v[106:107], s[6:7], v99, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v101, v109, vcc
	s_nop 1
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[100:101], v[68:69], 0, v[106:107]
	v_mul_lo_u32 v68, s53, v100
	v_mul_lo_u32 v101, s52, v101
	v_mad_u64_u32 v[106:107], s[6:7], s52, v100, 0
	v_add3_u32 v68, v107, v101, v68
	v_sub_co_u32_e32 v101, vcc, v103, v106
	v_cmp_le_u32_e64 s[24:25], s52, v101
	s_nop 0
	v_subb_co_u32_e64 v103, s[22:23], v99, v68, vcc
	v_sub_u32_e32 v68, v99, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v99, vcc, s52, v101
	v_cmp_le_u32_e64 s[22:23], s53, v103
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v105, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v103
	v_cndmask_b32_e64 v103, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v99
	v_cndmask_b32_e64 v103, v105, v103, s[22:23]
	v_cndmask_b32_e64 v101, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v103
	v_cndmask_b32_e32 v68, v101, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v99, s64, v102
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v100, v68
	v_cndmask_b32_e64 v68, v100, v68, s[22:23]
	v_ashrrev_i32_e32 v100, 31, v91
	v_mov_b32_e32 v101, v100
	v_xor_b32_e32 v68, v68, v99
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[100:101]
	v_sub_u32_e32 v68, v68, v99
	v_xor_b32_e32 v105, v90, v100
	v_xad_u32 v99, v68, v104, v30
	v_xor_b32_e32 v101, v91, v100
	v_mad_u64_u32 v[90:91], s[6:7], v105, s5, 0
	v_mul_hi_u32 v68, v105, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[106:107], s[6:7], v101, s4, 0
	v_add_co_u32_e32 v68, vcc, v90, v106
	v_mad_u64_u32 v[102:103], s[6:7], v101, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v91, v107, vcc
	s_nop 1
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[102:103]
	v_mul_lo_u32 v68, s53, v90
	v_mul_lo_u32 v102, s52, v91
	v_mad_u64_u32 v[90:91], s[6:7], s52, v90, 0
	v_add3_u32 v68, v91, v102, v68
	v_sub_u32_e32 v91, v101, v68
	v_sub_co_u32_e32 v90, vcc, v105, v90
	s_mov_b64 s[6:7], 0x80b
	s_nop 0
	v_subb_co_u32_e64 v91, s[22:23], v91, v57, vcc
	v_subrev_co_u32_e64 v102, s[22:23], s52, v90
	v_cmp_le_u32_e64 s[26:27], s52, v102
	s_nop 0
	v_subbrev_co_u32_e64 v103, s[24:25], 0, v91, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v103
	v_cndmask_b32_e64 v106, 0, -1, s[26:27]
	v_subb_co_u32_e64 v91, s[22:23], v91, v57, s[22:23]
	v_cndmask_b32_e64 v105, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v103
	v_subb_co_u32_e32 v68, vcc, v101, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v105, v105, v106, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v105
	v_subrev_co_u32_e64 v105, s[22:23], s52, v102
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[22:23], 0, v91, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v90
	v_cndmask_b32_e64 v91, v103, v91, s[24:25]
	v_cndmask_b32_e64 v101, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v103, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v101, v101, v103, vcc
	v_cmp_ne_u32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v91, vcc
	v_cndmask_b32_e64 v91, v102, v105, s[24:25]
	v_cndmask_b32_e32 v90, v90, v91, vcc
	v_xor_b32_e32 v91, v68, v100
	v_xor_b32_e32 v68, v90, v100
	v_sub_co_u32_e32 v68, vcc, v68, v100
	s_nop 1
	v_subb_co_u32_e32 v91, vcc, v91, v100, vcc
	v_ashrrev_i32_e32 v90, 31, v91
	v_and_b32_e32 v90, s42, v90
	v_mul_lo_u32 v91, v99, s33
	v_add3_u32 v68, v90, v68, v91
	v_lshl_add_u64 v[90:91], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xf7f4
	buffer_load_ubyte v99, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	s_nop 1
	v_subb_co_u32_e32 v100, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v101, v91, v100, s[18:19]
	v_ashrrev_i32_e32 v102, 31, v101
	v_cndmask_b32_e64 v100, v90, v68, s[18:19]
	v_mov_b32_e32 v103, v102
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[102:103]
	v_xor_b32_e32 v105, v100, v102
	v_xor_b32_e32 v103, v101, v102
	v_mad_u64_u32 v[100:101], s[6:7], v105, s5, 0
	v_mul_hi_u32 v68, v105, s4
	v_lshl_add_u64 v[100:101], v[68:69], 0, v[100:101]
	v_mad_u64_u32 v[108:109], s[6:7], v103, s4, 0
	v_add_co_u32_e32 v68, vcc, v100, v108
	v_mad_u64_u32 v[106:107], s[6:7], v103, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v101, v109, vcc
	s_nop 1
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[100:101], v[68:69], 0, v[106:107]
	v_mul_lo_u32 v68, s53, v100
	v_mul_lo_u32 v101, s52, v101
	v_mad_u64_u32 v[106:107], s[6:7], s52, v100, 0
	v_add3_u32 v68, v107, v101, v68
	v_sub_co_u32_e32 v101, vcc, v105, v106
	v_cmp_le_u32_e64 s[24:25], s52, v101
	s_nop 0
	v_subb_co_u32_e64 v105, s[22:23], v103, v68, vcc
	v_sub_u32_e32 v68, v103, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v101, vcc, s52, v101
	v_cmp_le_u32_e64 s[22:23], s53, v105
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v106, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v105
	v_cndmask_b32_e64 v105, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v101
	v_cndmask_b32_e64 v105, v106, v105, s[22:23]
	v_cndmask_b32_e64 v103, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v105
	v_cndmask_b32_e32 v68, v103, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v100, v68
	v_cndmask_b32_e64 v68, v100, v68, s[22:23]
	v_xor_b32_e32 v100, s64, v102
	v_xor_b32_e32 v68, v68, v100
	v_sub_u32_e32 v68, v68, v100
	v_ashrrev_i32_e32 v100, 31, v91
	v_mov_b32_e32 v101, v100
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[100:101]
	v_xor_b32_e32 v108, v90, v100
	v_xad_u32 v105, v68, v104, v30
	v_xor_b32_e32 v101, v91, v100
	v_mad_u64_u32 v[90:91], s[6:7], v108, s5, 0
	v_mul_hi_u32 v68, v108, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[106:107], s[6:7], v101, s4, 0
	v_add_co_u32_e32 v68, vcc, v90, v106
	v_mad_u64_u32 v[102:103], s[6:7], v101, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v91, v107, vcc
	s_nop 1
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[102:103]
	v_mul_lo_u32 v68, s53, v90
	v_mul_lo_u32 v102, s52, v91
	v_mad_u64_u32 v[90:91], s[6:7], s52, v90, 0
	v_add3_u32 v68, v91, v102, v68
	v_sub_u32_e32 v91, v101, v68
	v_sub_co_u32_e32 v90, vcc, v108, v90
	s_mov_b64 s[6:7], 0x80c
	s_nop 0
	v_subb_co_u32_e64 v91, s[22:23], v91, v57, vcc
	v_subrev_co_u32_e64 v102, s[22:23], s52, v90
	v_cmp_le_u32_e64 s[26:27], s52, v102
	s_nop 0
	v_subbrev_co_u32_e64 v103, s[24:25], 0, v91, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v103
	v_cndmask_b32_e64 v107, 0, -1, s[26:27]
	v_subb_co_u32_e64 v91, s[22:23], v91, v57, s[22:23]
	v_cndmask_b32_e64 v106, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v103
	v_subb_co_u32_e32 v68, vcc, v101, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v106, v106, v107, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v106
	v_subrev_co_u32_e64 v106, s[22:23], s52, v102
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[22:23], 0, v91, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v90
	v_cndmask_b32_e64 v91, v103, v91, s[24:25]
	v_cndmask_b32_e64 v101, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v103, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v101, v101, v103, vcc
	v_cmp_ne_u32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v91, vcc
	v_cndmask_b32_e64 v91, v102, v106, s[24:25]
	v_cndmask_b32_e32 v90, v90, v91, vcc
	v_xor_b32_e32 v91, v68, v100
	v_xor_b32_e32 v68, v90, v100
	v_sub_co_u32_e32 v68, vcc, v68, v100
	s_nop 1
	v_subb_co_u32_e32 v91, vcc, v91, v100, vcc
	v_ashrrev_i32_e32 v90, 31, v91
	v_and_b32_e32 v90, s42, v90
	v_mul_lo_u32 v91, v105, s33
	v_add3_u32 v68, v90, v68, v91
	v_lshl_add_u64 v[90:91], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xf7f3
	buffer_load_ubyte v100, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cndmask_b32_e64 v102, v90, v68, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v101, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v103, v91, v101, s[18:19]
	v_ashrrev_i32_e32 v106, 31, v103
	v_mov_b32_e32 v107, v106
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[106:107]
	v_xor_b32_e32 v105, v102, v106
	v_xor_b32_e32 v101, v103, v106
	v_mad_u64_u32 v[102:103], s[6:7], v105, s5, 0
	v_mul_hi_u32 v68, v105, s4
	v_lshl_add_u64 v[102:103], v[68:69], 0, v[102:103]
	v_mad_u64_u32 v[110:111], s[6:7], v101, s4, 0
	v_add_co_u32_e32 v68, vcc, v102, v110
	v_mad_u64_u32 v[108:109], s[6:7], v101, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v103, v111, vcc
	s_nop 1
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[102:103], v[68:69], 0, v[108:109]
	v_mul_lo_u32 v68, s53, v102
	v_mul_lo_u32 v103, s52, v103
	v_mad_u64_u32 v[108:109], s[6:7], s52, v102, 0
	v_add3_u32 v68, v109, v103, v68
	v_sub_co_u32_e32 v103, vcc, v105, v108
	v_cmp_le_u32_e64 s[24:25], s52, v103
	s_nop 0
	v_subb_co_u32_e64 v105, s[22:23], v101, v68, vcc
	v_sub_u32_e32 v68, v101, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v101, vcc, s52, v103
	v_cmp_le_u32_e64 s[22:23], s53, v105
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v107, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v105
	v_cndmask_b32_e64 v105, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v101
	v_cndmask_b32_e64 v105, v107, v105, s[22:23]
	v_cndmask_b32_e64 v103, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v105
	v_cndmask_b32_e32 v68, v103, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v101, s64, v106
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v102, v68
	v_cndmask_b32_e64 v68, v102, v68, s[22:23]
	v_ashrrev_i32_e32 v102, 31, v91
	v_mov_b32_e32 v103, v102
	v_xor_b32_e32 v68, v68, v101
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[102:103]
	v_sub_u32_e32 v68, v68, v101
	v_xor_b32_e32 v105, v90, v102
	v_xad_u32 v101, v68, v104, v30
	v_xor_b32_e32 v103, v91, v102
	v_mad_u64_u32 v[90:91], s[6:7], v105, s5, 0
	v_mul_hi_u32 v68, v105, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[108:109], s[6:7], v103, s4, 0
	v_add_co_u32_e32 v68, vcc, v90, v108
	v_mad_u64_u32 v[106:107], s[6:7], v103, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v91, v109, vcc
	s_nop 1
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[106:107]
	v_mul_lo_u32 v68, s53, v90
	v_mul_lo_u32 v106, s52, v91
	v_mad_u64_u32 v[90:91], s[6:7], s52, v90, 0
	v_add3_u32 v68, v91, v106, v68
	v_sub_u32_e32 v91, v103, v68
	v_sub_co_u32_e32 v90, vcc, v105, v90
	s_mov_b64 s[6:7], 0x80d
	s_nop 0
	v_subb_co_u32_e64 v91, s[22:23], v91, v57, vcc
	v_subrev_co_u32_e64 v105, s[22:23], s52, v90
	v_cmp_le_u32_e64 s[26:27], s52, v105
	s_nop 0
	v_subbrev_co_u32_e64 v106, s[24:25], 0, v91, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v106
	v_cndmask_b32_e64 v108, 0, -1, s[26:27]
	v_subb_co_u32_e64 v91, s[22:23], v91, v57, s[22:23]
	v_cndmask_b32_e64 v107, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v106
	v_subb_co_u32_e32 v68, vcc, v103, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v107, v107, v108, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v107
	v_subrev_co_u32_e64 v107, s[22:23], s52, v105
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[22:23], 0, v91, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v90
	v_cndmask_b32_e64 v91, v106, v91, s[24:25]
	v_cndmask_b32_e64 v103, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v106, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v103, v103, v106, vcc
	v_cmp_ne_u32_e32 vcc, 0, v103
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v91, vcc
	v_cndmask_b32_e64 v91, v105, v107, s[24:25]
	v_cndmask_b32_e32 v90, v90, v91, vcc
	v_xor_b32_e32 v91, v68, v102
	v_xor_b32_e32 v68, v90, v102
	v_sub_co_u32_e32 v68, vcc, v68, v102
	s_nop 1
	v_subb_co_u32_e32 v91, vcc, v91, v102, vcc
	v_ashrrev_i32_e32 v90, 31, v91
	v_and_b32_e32 v90, s42, v90
	v_mul_lo_u32 v91, v101, s33
	v_add3_u32 v68, v90, v68, v91
	v_lshl_add_u64 v[90:91], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xf7f2
	buffer_load_ubyte v101, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	s_nop 1
	v_subb_co_u32_e32 v102, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v103, v91, v102, s[18:19]
	v_ashrrev_i32_e32 v106, 31, v103
	v_cndmask_b32_e64 v102, v90, v68, s[18:19]
	v_mov_b32_e32 v107, v106
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[106:107]
	v_xor_b32_e32 v107, v102, v106
	v_xor_b32_e32 v105, v103, v106
	v_mad_u64_u32 v[102:103], s[6:7], v107, s5, 0
	v_mul_hi_u32 v68, v107, s4
	v_lshl_add_u64 v[102:103], v[68:69], 0, v[102:103]
	v_mad_u64_u32 v[110:111], s[6:7], v105, s4, 0
	v_add_co_u32_e32 v68, vcc, v102, v110
	v_mad_u64_u32 v[108:109], s[6:7], v105, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v103, v111, vcc
	s_nop 1
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[102:103], v[68:69], 0, v[108:109]
	v_mul_lo_u32 v68, s53, v102
	v_mul_lo_u32 v103, s52, v103
	v_mad_u64_u32 v[108:109], s[6:7], s52, v102, 0
	v_add3_u32 v68, v109, v103, v68
	v_sub_co_u32_e32 v103, vcc, v107, v108
	v_cmp_le_u32_e64 s[24:25], s52, v103
	s_nop 0
	v_subb_co_u32_e64 v107, s[22:23], v105, v68, vcc
	v_sub_u32_e32 v68, v105, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v103, vcc, s52, v103
	v_cmp_le_u32_e64 s[22:23], s53, v107
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v108, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v107
	v_cndmask_b32_e64 v107, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v103
	v_cndmask_b32_e64 v107, v108, v107, s[22:23]
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v107
	v_cndmask_b32_e32 v68, v105, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v102, v68
	v_cndmask_b32_e64 v68, v102, v68, s[22:23]
	v_xor_b32_e32 v102, s64, v106
	v_xor_b32_e32 v68, v68, v102
	v_sub_u32_e32 v68, v68, v102
	v_ashrrev_i32_e32 v102, 31, v91
	v_mov_b32_e32 v103, v102
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[102:103]
	v_xor_b32_e32 v110, v90, v102
	v_xad_u32 v105, v68, v104, v30
	v_xor_b32_e32 v103, v91, v102
	v_mad_u64_u32 v[90:91], s[6:7], v110, s5, 0
	v_mul_hi_u32 v68, v110, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[108:109], s[6:7], v103, s4, 0
	v_add_co_u32_e32 v68, vcc, v90, v108
	v_mad_u64_u32 v[106:107], s[6:7], v103, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v91, v109, vcc
	s_nop 1
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[106:107]
	v_mul_lo_u32 v68, s53, v90
	v_mul_lo_u32 v106, s52, v91
	v_mad_u64_u32 v[90:91], s[6:7], s52, v90, 0
	v_add3_u32 v68, v91, v106, v68
	v_sub_u32_e32 v91, v103, v68
	v_sub_co_u32_e32 v90, vcc, v110, v90
	s_mov_b64 s[6:7], 0x80e
	s_nop 0
	v_subb_co_u32_e64 v91, s[22:23], v91, v57, vcc
	v_subrev_co_u32_e64 v106, s[22:23], s52, v90
	v_cmp_le_u32_e64 s[26:27], s52, v106
	s_nop 0
	v_subbrev_co_u32_e64 v107, s[24:25], 0, v91, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v107
	v_cndmask_b32_e64 v109, 0, -1, s[26:27]
	v_subb_co_u32_e64 v91, s[22:23], v91, v57, s[22:23]
	v_cndmask_b32_e64 v108, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v107
	v_subb_co_u32_e32 v68, vcc, v103, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v108, v108, v109, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v108
	v_subrev_co_u32_e64 v108, s[22:23], s52, v106
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[22:23], 0, v91, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v90
	v_cndmask_b32_e64 v91, v107, v91, s[24:25]
	v_cndmask_b32_e64 v103, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v107, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v103, v103, v107, vcc
	v_cmp_ne_u32_e32 vcc, 0, v103
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v91, vcc
	v_cndmask_b32_e64 v91, v106, v108, s[24:25]
	v_cndmask_b32_e32 v90, v90, v91, vcc
	v_xor_b32_e32 v91, v68, v102
	v_xor_b32_e32 v68, v90, v102
	v_sub_co_u32_e32 v68, vcc, v68, v102
	s_nop 1
	v_subb_co_u32_e32 v91, vcc, v91, v102, vcc
	v_ashrrev_i32_e32 v90, 31, v91
	v_and_b32_e32 v90, s42, v90
	v_mul_lo_u32 v91, v105, s33
	v_add3_u32 v68, v90, v68, v91
	v_lshl_add_u64 v[90:91], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xf7f1
	buffer_load_ubyte v102, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cndmask_b32_e64 v106, v90, v68, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v103, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v107, v91, v103, s[18:19]
	v_ashrrev_i32_e32 v108, 31, v107
	v_mov_b32_e32 v109, v108
	v_lshl_add_u64 v[106:107], v[106:107], 0, v[108:109]
	v_xor_b32_e32 v105, v106, v108
	v_xor_b32_e32 v103, v107, v108
	v_mad_u64_u32 v[106:107], s[6:7], v105, s5, 0
	v_mul_hi_u32 v68, v105, s4
	v_lshl_add_u64 v[106:107], v[68:69], 0, v[106:107]
	v_mad_u64_u32 v[112:113], s[6:7], v103, s4, 0
	v_add_co_u32_e32 v68, vcc, v106, v112
	v_mad_u64_u32 v[110:111], s[6:7], v103, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v107, v113, vcc
	s_nop 1
	v_addc_co_u32_e32 v111, vcc, 0, v111, vcc
	v_lshl_add_u64 v[106:107], v[68:69], 0, v[110:111]
	v_mul_lo_u32 v68, s53, v106
	v_mul_lo_u32 v107, s52, v107
	v_mad_u64_u32 v[110:111], s[6:7], s52, v106, 0
	v_add3_u32 v68, v111, v107, v68
	v_sub_co_u32_e32 v105, vcc, v105, v110
	v_cmp_le_u32_e64 s[24:25], s52, v105
	s_nop 0
	v_subb_co_u32_e64 v107, s[22:23], v103, v68, vcc
	v_sub_u32_e32 v68, v103, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v103, vcc, s52, v105
	v_cmp_le_u32_e64 s[22:23], s53, v107
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v109, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v107
	v_cndmask_b32_e64 v107, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v103
	v_cndmask_b32_e64 v107, v109, v107, s[22:23]
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v107
	v_cndmask_b32_e32 v68, v105, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v103, s64, v108
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v106, v68
	v_cndmask_b32_e64 v68, v106, v68, s[22:23]
	v_ashrrev_i32_e32 v106, 31, v91
	v_mov_b32_e32 v107, v106
	v_xor_b32_e32 v68, v68, v103
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[106:107]
	v_sub_u32_e32 v68, v68, v103
	v_xor_b32_e32 v107, v90, v106
	v_xad_u32 v103, v68, v104, v30
	v_xor_b32_e32 v105, v91, v106
	v_mad_u64_u32 v[90:91], s[6:7], v107, s5, 0
	v_mul_hi_u32 v68, v107, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[110:111], s[6:7], v105, s4, 0
	v_add_co_u32_e32 v68, vcc, v90, v110
	v_mad_u64_u32 v[108:109], s[6:7], v105, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v91, v111, vcc
	s_nop 1
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[108:109]
	v_mul_lo_u32 v68, s53, v90
	v_mul_lo_u32 v108, s52, v91
	v_mad_u64_u32 v[90:91], s[6:7], s52, v90, 0
	v_add3_u32 v68, v91, v108, v68
	v_sub_u32_e32 v91, v105, v68
	v_sub_co_u32_e32 v90, vcc, v107, v90
	s_mov_b64 s[6:7], 0x80f
	s_nop 0
	v_subb_co_u32_e64 v91, s[22:23], v91, v57, vcc
	v_subrev_co_u32_e64 v107, s[22:23], s52, v90
	v_cmp_le_u32_e64 s[26:27], s52, v107
	s_nop 0
	v_subbrev_co_u32_e64 v108, s[24:25], 0, v91, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v108
	v_cndmask_b32_e64 v110, 0, -1, s[26:27]
	v_subb_co_u32_e64 v91, s[22:23], v91, v57, s[22:23]
	v_cndmask_b32_e64 v109, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v108
	v_subb_co_u32_e32 v68, vcc, v105, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v109, v109, v110, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v109
	v_subrev_co_u32_e64 v109, s[22:23], s52, v107
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[22:23], 0, v91, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v90
	v_cndmask_b32_e64 v91, v108, v91, s[24:25]
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v108, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v105, v105, v108, vcc
	v_cmp_ne_u32_e32 vcc, 0, v105
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v91, vcc
	v_cndmask_b32_e64 v91, v107, v109, s[24:25]
	v_cndmask_b32_e32 v90, v90, v91, vcc
	v_xor_b32_e32 v91, v68, v106
	v_xor_b32_e32 v68, v90, v106
	v_sub_co_u32_e32 v68, vcc, v68, v106
	s_nop 1
	v_subb_co_u32_e32 v91, vcc, v91, v106, vcc
	v_ashrrev_i32_e32 v90, 31, v91
	v_and_b32_e32 v90, s42, v90
	v_mul_lo_u32 v91, v103, s33
	v_add3_u32 v68, v90, v68, v91
	v_lshl_add_u64 v[90:91], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xf7f0
	buffer_load_ubyte v103, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cndmask_b32_e64 v106, v90, v68, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v105, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v107, v91, v105, s[18:19]
	v_ashrrev_i32_e32 v108, 31, v107
	v_mov_b32_e32 v109, v108
	v_lshl_add_u64 v[106:107], v[106:107], 0, v[108:109]
	v_xor_b32_e32 v109, v106, v108
	v_xor_b32_e32 v105, v107, v108
	v_mad_u64_u32 v[106:107], s[6:7], v109, s5, 0
	v_mul_hi_u32 v68, v109, s4
	v_lshl_add_u64 v[106:107], v[68:69], 0, v[106:107]
	v_mad_u64_u32 v[112:113], s[6:7], v105, s4, 0
	v_add_co_u32_e32 v68, vcc, v106, v112
	v_mad_u64_u32 v[110:111], s[6:7], v105, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v107, v113, vcc
	s_nop 1
	v_addc_co_u32_e32 v111, vcc, 0, v111, vcc
	v_lshl_add_u64 v[106:107], v[68:69], 0, v[110:111]
	v_mul_lo_u32 v68, s53, v106
	v_mul_lo_u32 v107, s52, v107
	v_mad_u64_u32 v[110:111], s[6:7], s52, v106, 0
	v_add3_u32 v68, v111, v107, v68
	v_sub_co_u32_e32 v107, vcc, v109, v110
	v_cmp_le_u32_e64 s[22:23], s52, v107
	s_nop 0
	v_subb_co_u32_e64 v109, s[18:19], v105, v68, vcc
	v_sub_u32_e32 v68, v105, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v105, vcc, s52, v107
	v_cmp_le_u32_e64 s[18:19], s53, v109
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v110, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s53, v109
	v_cndmask_b32_e64 v109, 0, -1, s[22:23]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[22:23], s52, v105
	v_cndmask_b32_e64 v109, v110, v109, s[18:19]
	v_cndmask_b32_e64 v107, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[22:23]
	v_cmp_ne_u32_e64 s[18:19], 0, v109
	v_cndmask_b32_e32 v68, v107, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v105, s64, v108
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v106, v68
	v_cndmask_b32_e64 v68, v106, v68, s[18:19]
	v_xor_b32_e32 v68, v68, v105
	v_sub_u32_e32 v68, v68, v105
	v_xad_u32 v110, v68, v104, v30
	v_ashrrev_i32_e32 v104, 31, v91
	v_mov_b32_e32 v105, v104
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[104:105]
	v_xor_b32_e32 v111, v90, v104
	v_xor_b32_e32 v105, v91, v104
	v_mad_u64_u32 v[90:91], s[6:7], v111, s5, 0
	v_mul_hi_u32 v68, v111, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[108:109], s[6:7], v105, s4, 0
	v_add_co_u32_e32 v68, vcc, v90, v108
	v_mad_u64_u32 v[106:107], s[6:7], v105, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v91, v109, vcc
	s_nop 1
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[106:107]
	v_mul_lo_u32 v68, s53, v90
	v_mul_lo_u32 v106, s52, v91
	v_mad_u64_u32 v[90:91], s[6:7], s52, v90, 0
	v_add3_u32 v68, v91, v106, v68
	v_sub_u32_e32 v91, v105, v68
	v_sub_co_u32_e32 v90, vcc, v111, v90
	s_mov_b64 s[6:7], 0x1400
	s_nop 0
	v_subb_co_u32_e64 v91, s[18:19], v91, v57, vcc
	v_subrev_co_u32_e64 v106, s[18:19], s52, v90
	v_cmp_le_u32_e64 s[24:25], s52, v106
	s_nop 0
	v_subbrev_co_u32_e64 v107, s[22:23], 0, v91, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s53, v107
	v_cndmask_b32_e64 v109, 0, -1, s[24:25]
	v_subb_co_u32_e64 v91, s[18:19], v91, v57, s[18:19]
	v_cndmask_b32_e64 v108, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v107
	v_subb_co_u32_e32 v68, vcc, v105, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v108, v108, v109, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v108
	v_subrev_co_u32_e64 v108, s[18:19], s52, v106
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[18:19], 0, v91, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s52, v90
	v_cndmask_b32_e64 v91, v107, v91, s[22:23]
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v107, 0, -1, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v105, v105, v107, vcc
	v_cmp_ne_u32_e32 vcc, 0, v105
	v_and_b32_e32 v105, 0xffff, v92
	s_nop 0
	v_cndmask_b32_e32 v68, v68, v91, vcc
	v_cndmask_b32_e64 v91, v106, v108, s[22:23]
	v_cndmask_b32_e32 v90, v90, v91, vcc
	v_xor_b32_e32 v91, v68, v104
	v_xor_b32_e32 v68, v90, v104
	v_sub_co_u32_e32 v68, vcc, v68, v104
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[122:125], v[20:23], v[12:15], v93, v105 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, v91, v104, vcc
	v_ashrrev_i32_e32 v90, 31, v91
	v_and_b32_e32 v90, s42, v90
	v_mul_lo_u32 v91, v110, s33
	v_add3_u32 v68, v90, v68, v91
	v_lshl_add_u64 v[90:91], v[88:89], 0, s[6:7]
	s_movk_i32 s6, 0xebff
	buffer_load_ubyte v104, v68, s[28:31], 0 offen
	v_sub_co_u32_e64 v68, s[18:19], s6, v88
	v_cmp_gt_i32_e32 vcc, 0, v91
	s_nop 0
	v_subb_co_u32_e64 v88, s[18:19], -1, v89, s[18:19]
	v_cndmask_b32_e32 v89, v91, v88, vcc
	v_ashrrev_i32_e32 v92, 31, v89
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[130:133], v[20:23], v[0:3], v93, v105 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v88, v90, v68, vcc
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[92:93]
	v_xor_b32_e32 v110, v88, v92
	v_xor_b32_e32 v93, v89, v92
	v_mad_u64_u32 v[88:89], s[6:7], v110, s5, 0
	v_mul_hi_u32 v68, v110, s4
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[88:89]
	v_mad_u64_u32 v[108:109], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v88, v108
	v_mad_u64_u32 v[106:107], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v89, v109, vcc
	ds_read_b128 v[122:125], v52 offset:12288
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[106:107]
	v_mul_lo_u32 v68, s53, v88
	v_mul_lo_u32 v89, s52, v89
	v_mad_u64_u32 v[106:107], s[6:7], s52, v88, 0
	v_add3_u32 v68, v107, v89, v68
	v_sub_co_u32_e32 v89, vcc, v110, v106
	v_cmp_le_u32_e64 s[22:23], s52, v89
	s_nop 0
	v_subb_co_u32_e64 v106, s[18:19], v93, v68, vcc
	v_sub_u32_e32 v68, v93, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v89, vcc, s52, v89
	v_cmp_le_u32_e64 s[18:19], s53, v106
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v107, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s53, v106
	v_cndmask_b32_e64 v106, 0, -1, s[22:23]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[22:23], s52, v89
	v_cndmask_b32_e64 v106, v107, v106, s[18:19]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[22:23]
	v_cmp_ne_u32_e64 s[18:19], 0, v106
	v_cndmask_b32_e32 v68, v93, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v88, v68
	v_cndmask_b32_e64 v68, v88, v68, s[18:19]
	v_xor_b32_e32 v88, s64, v92
	v_xor_b32_e32 v68, v68, v88
	v_sub_u32_e32 v68, v68, v88
	v_ashrrev_i32_e32 v88, 31, v91
	v_mov_b32_e32 v89, v88
	v_lshl_add_u64 v[90:91], v[90:91], 0, v[88:89]
	v_xor_b32_e32 v109, v90, v88
	v_xad_u32 v108, v68, v88, v30
	v_xor_b32_e32 v89, v91, v88
	v_mad_u64_u32 v[90:91], s[6:7], v109, s5, 0
	v_mul_hi_u32 v68, v109, s4
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[90:91]
	v_mad_u64_u32 v[106:107], s[6:7], v89, s4, 0
	v_add_co_u32_e32 v68, vcc, v90, v106
	v_mad_u64_u32 v[92:93], s[6:7], v89, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v91, v107, vcc
	s_nop 1
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[90:91], v[68:69], 0, v[92:93]
	v_mul_lo_u32 v68, s53, v90
	v_mul_lo_u32 v92, s52, v91
	v_mad_u64_u32 v[90:91], s[6:7], s52, v90, 0
	v_add3_u32 v68, v91, v92, v68
	v_sub_u32_e32 v91, v89, v68
	v_sub_co_u32_e32 v90, vcc, v109, v90
	s_mov_b64 s[6:7], 0xc01
	s_nop 0
	v_subb_co_u32_e64 v91, s[18:19], v91, v57, vcc
	v_subrev_co_u32_e64 v92, s[18:19], s52, v90
	v_cmp_le_u32_e64 s[24:25], s52, v92
	s_nop 0
	v_subbrev_co_u32_e64 v93, s[22:23], 0, v91, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s53, v93
	v_cndmask_b32_e64 v107, 0, -1, s[24:25]
	v_subb_co_u32_e64 v91, s[18:19], v91, v57, s[18:19]
	v_cndmask_b32_e64 v106, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v93
	v_subb_co_u32_e32 v68, vcc, v89, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v106, v106, v107, s[22:23]
	v_cmp_ne_u32_e64 s[22:23], 0, v106
	v_subrev_co_u32_e64 v106, s[18:19], s52, v92
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[18:19], 0, v91, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s52, v90
	v_cndmask_b32_e64 v91, v93, v91, s[22:23]
	v_cndmask_b32_e64 v89, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v93, 0, -1, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v89, v89, v93, vcc
	v_cmp_ne_u32_e32 vcc, 0, v89
	v_cndmask_b32_e64 v89, v92, v106, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v89, v90, v89, vcc
	v_cndmask_b32_e32 v68, v68, v91, vcc
	v_xor_b32_e32 v89, v89, v88
	v_xor_b32_e32 v68, v68, v88
	v_sub_co_u32_e32 v89, vcc, v89, v88
	v_lshl_add_u64 v[90:91], v[86:87], 0, s[84:85]
	s_nop 0
	v_subb_co_u32_e32 v68, vcc, v68, v88, vcc
	v_ashrrev_i32_e32 v68, 31, v68
	v_and_b32_e32 v68, s42, v68
	v_mul_lo_u32 v88, v108, s33
	v_add3_u32 v68, v68, v89, v88
	v_lshl_add_u64 v[88:89], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xf3fe
	buffer_load_ubyte v106, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cmp_gt_i32_e64 s[18:19], 0, v91
	s_nop 0
	v_subb_co_u32_e32 v90, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v93, v89, v90, s[18:19]
	v_ashrrev_i32_e32 v108, 31, v93
	v_cndmask_b32_e64 v92, v88, v68, s[18:19]
	v_mov_b32_e32 v109, v108
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[108:109]
	v_xor_b32_e32 v107, v92, v108
	v_xor_b32_e32 v90, v93, v108
	v_mad_u64_u32 v[92:93], s[6:7], v107, s5, 0
	v_mul_hi_u32 v68, v107, s4
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[92:93]
	v_mad_u64_u32 v[112:113], s[6:7], v90, s4, 0
	v_add_co_u32_e32 v68, vcc, v92, v112
	v_mad_u64_u32 v[110:111], s[6:7], v90, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v93, v113, vcc
	s_nop 1
	v_addc_co_u32_e32 v111, vcc, 0, v111, vcc
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[110:111]
	v_mul_lo_u32 v68, s53, v92
	v_mul_lo_u32 v93, s52, v93
	v_mad_u64_u32 v[110:111], s[6:7], s52, v92, 0
	v_add3_u32 v68, v111, v93, v68
	v_sub_co_u32_e32 v93, vcc, v107, v110
	v_cmp_le_u32_e64 s[24:25], s52, v93
	s_nop 0
	v_subb_co_u32_e64 v107, s[22:23], v90, v68, vcc
	v_sub_u32_e32 v68, v90, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v90, vcc, s52, v93
	v_cmp_le_u32_e64 s[22:23], s53, v107
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v109, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v107
	v_cndmask_b32_e64 v107, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v90
	v_cndmask_b32_e64 v107, v109, v107, s[22:23]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v107
	v_cndmask_b32_e32 v68, v93, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v90, s64, v108
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v92, v68
	v_cndmask_b32_e64 v68, v92, v68, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v89
	v_mov_b32_e32 v93, v92
	v_xor_b32_e32 v68, v68, v90
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[92:93]
	v_sub_u32_e32 v68, v68, v90
	v_ashrrev_i32_e32 v90, 31, v91
	v_xor_b32_e32 v107, v88, v92
	v_xad_u32 v91, v68, v90, v30
	v_xor_b32_e32 v93, v89, v92
	v_mad_u64_u32 v[88:89], s[6:7], v107, s5, 0
	v_mul_hi_u32 v68, v107, s4
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[88:89]
	v_mad_u64_u32 v[110:111], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v88, v110
	v_mad_u64_u32 v[108:109], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v89, v111, vcc
	s_nop 1
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[108:109]
	v_mul_lo_u32 v68, s53, v88
	v_mul_lo_u32 v108, s52, v89
	v_mad_u64_u32 v[88:89], s[6:7], s52, v88, 0
	v_add3_u32 v68, v89, v108, v68
	v_sub_u32_e32 v89, v93, v68
	v_sub_co_u32_e32 v88, vcc, v107, v88
	s_mov_b64 s[6:7], 0xc02
	s_nop 0
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, vcc
	v_subrev_co_u32_e64 v107, s[22:23], s52, v88
	v_cmp_le_u32_e64 s[26:27], s52, v107
	s_nop 0
	v_subbrev_co_u32_e64 v108, s[24:25], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v108
	v_cndmask_b32_e64 v110, 0, -1, s[26:27]
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, s[22:23]
	v_cndmask_b32_e64 v109, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v108
	v_subb_co_u32_e32 v68, vcc, v93, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v109, v109, v110, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v109
	v_subrev_co_u32_e64 v109, s[22:23], s52, v107
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[22:23], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v88
	v_cndmask_b32_e64 v89, v108, v89, s[24:25]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v108, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v93, v93, v108, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v89, vcc
	v_cndmask_b32_e64 v89, v107, v109, s[24:25]
	v_cndmask_b32_e32 v88, v88, v89, vcc
	v_xor_b32_e32 v89, v68, v92
	v_xor_b32_e32 v68, v88, v92
	v_sub_co_u32_e32 v68, vcc, v68, v92
	s_nop 1
	v_subb_co_u32_e32 v89, vcc, v89, v92, vcc
	v_ashrrev_i32_e32 v88, 31, v89
	v_and_b32_e32 v88, s42, v88
	v_mul_lo_u32 v89, v91, s33
	v_add3_u32 v68, v88, v68, v89
	v_lshl_add_u64 v[88:89], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xf3fd
	buffer_load_ubyte v107, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cndmask_b32_e64 v92, v88, v68, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v93, v89, v91, s[18:19]
	v_ashrrev_i32_e32 v108, 31, v93
	v_mov_b32_e32 v109, v108
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[108:109]
	v_xor_b32_e32 v109, v92, v108
	v_xor_b32_e32 v91, v93, v108
	v_mad_u64_u32 v[92:93], s[6:7], v109, s5, 0
	v_mul_hi_u32 v68, v109, s4
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[92:93]
	v_mad_u64_u32 v[112:113], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v92, v112
	v_mad_u64_u32 v[110:111], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v93, v113, vcc
	s_nop 1
	v_addc_co_u32_e32 v111, vcc, 0, v111, vcc
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[110:111]
	v_mul_lo_u32 v68, s53, v92
	v_mul_lo_u32 v93, s52, v93
	v_mad_u64_u32 v[110:111], s[6:7], s52, v92, 0
	v_add3_u32 v68, v111, v93, v68
	v_sub_co_u32_e32 v93, vcc, v109, v110
	v_cmp_le_u32_e64 s[24:25], s52, v93
	s_nop 0
	v_subb_co_u32_e64 v109, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v91, vcc, s52, v93
	v_cmp_le_u32_e64 s[22:23], s53, v109
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v110, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v109
	v_cndmask_b32_e64 v109, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v91
	v_cndmask_b32_e64 v109, v110, v109, s[22:23]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v109
	v_cndmask_b32_e32 v68, v93, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v91, s64, v108
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v92, v68
	v_cndmask_b32_e64 v68, v92, v68, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v89
	v_mov_b32_e32 v93, v92
	v_xor_b32_e32 v68, v68, v91
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[92:93]
	v_sub_u32_e32 v68, v68, v91
	v_xor_b32_e32 v112, v88, v92
	v_xad_u32 v91, v68, v90, v30
	v_xor_b32_e32 v93, v89, v92
	v_mad_u64_u32 v[88:89], s[6:7], v112, s5, 0
	v_mul_hi_u32 v68, v112, s4
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[88:89]
	v_mad_u64_u32 v[110:111], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v88, v110
	v_mad_u64_u32 v[108:109], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v89, v111, vcc
	s_nop 1
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[108:109]
	v_mul_lo_u32 v68, s53, v88
	v_mul_lo_u32 v108, s52, v89
	v_mad_u64_u32 v[88:89], s[6:7], s52, v88, 0
	v_add3_u32 v68, v89, v108, v68
	v_sub_u32_e32 v89, v93, v68
	v_sub_co_u32_e32 v88, vcc, v112, v88
	s_mov_b64 s[6:7], 0xc03
	s_nop 0
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, vcc
	v_subrev_co_u32_e64 v108, s[22:23], s52, v88
	v_cmp_le_u32_e64 s[26:27], s52, v108
	s_nop 0
	v_subbrev_co_u32_e64 v109, s[24:25], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v109
	v_cndmask_b32_e64 v111, 0, -1, s[26:27]
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, s[22:23]
	v_cndmask_b32_e64 v110, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v109
	v_subb_co_u32_e32 v68, vcc, v93, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v110, v110, v111, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v110
	v_subrev_co_u32_e64 v110, s[22:23], s52, v108
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[22:23], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v88
	v_cndmask_b32_e64 v89, v109, v89, s[24:25]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v109, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v93, v93, v109, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v89, vcc
	v_cndmask_b32_e64 v89, v108, v110, s[24:25]
	v_cndmask_b32_e32 v88, v88, v89, vcc
	v_xor_b32_e32 v89, v68, v92
	v_xor_b32_e32 v68, v88, v92
	v_sub_co_u32_e32 v68, vcc, v68, v92
	s_nop 1
	v_subb_co_u32_e32 v89, vcc, v89, v92, vcc
	v_ashrrev_i32_e32 v88, 31, v89
	v_and_b32_e32 v88, s42, v88
	v_mul_lo_u32 v89, v91, s33
	v_add3_u32 v68, v88, v68, v89
	v_lshl_add_u64 v[88:89], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xf3fc
	buffer_load_ubyte v108, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cndmask_b32_e64 v92, v88, v68, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v93, v89, v91, s[18:19]
	v_ashrrev_i32_e32 v110, 31, v93
	v_mov_b32_e32 v111, v110
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[110:111]
	v_xor_b32_e32 v109, v92, v110
	v_xor_b32_e32 v91, v93, v110
	v_mad_u64_u32 v[92:93], s[6:7], v109, s5, 0
	v_mul_hi_u32 v68, v109, s4
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[92:93]
	v_mad_u64_u32 v[114:115], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v92, v114
	v_mad_u64_u32 v[112:113], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v93, v115, vcc
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[112:113]
	v_mul_lo_u32 v68, s53, v92
	v_mul_lo_u32 v93, s52, v93
	v_mad_u64_u32 v[112:113], s[6:7], s52, v92, 0
	v_add3_u32 v68, v113, v93, v68
	v_sub_co_u32_e32 v93, vcc, v109, v112
	v_cmp_le_u32_e64 s[24:25], s52, v93
	s_nop 0
	v_subb_co_u32_e64 v109, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v91, vcc, s52, v93
	v_cmp_le_u32_e64 s[22:23], s53, v109
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v111, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v109
	v_cndmask_b32_e64 v109, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v91
	v_cndmask_b32_e64 v109, v111, v109, s[22:23]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v109
	v_cndmask_b32_e32 v68, v93, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v91, s64, v110
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v92, v68
	v_cndmask_b32_e64 v68, v92, v68, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v89
	v_mov_b32_e32 v93, v92
	v_xor_b32_e32 v68, v68, v91
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[92:93]
	v_sub_u32_e32 v68, v68, v91
	v_xor_b32_e32 v109, v88, v92
	v_xad_u32 v91, v68, v90, v30
	v_xor_b32_e32 v93, v89, v92
	v_mad_u64_u32 v[88:89], s[6:7], v109, s5, 0
	v_mul_hi_u32 v68, v109, s4
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[88:89]
	v_mad_u64_u32 v[112:113], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v88, v112
	v_mad_u64_u32 v[110:111], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v89, v113, vcc
	s_nop 1
	v_addc_co_u32_e32 v111, vcc, 0, v111, vcc
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[110:111]
	v_mul_lo_u32 v68, s53, v88
	v_mul_lo_u32 v110, s52, v89
	v_mad_u64_u32 v[88:89], s[6:7], s52, v88, 0
	v_add3_u32 v68, v89, v110, v68
	v_sub_u32_e32 v89, v93, v68
	v_sub_co_u32_e32 v88, vcc, v109, v88
	s_mov_b64 s[6:7], 0xc04
	s_nop 0
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, vcc
	v_subrev_co_u32_e64 v109, s[22:23], s52, v88
	v_cmp_le_u32_e64 s[26:27], s52, v109
	s_nop 0
	v_subbrev_co_u32_e64 v110, s[24:25], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v110
	v_cndmask_b32_e64 v112, 0, -1, s[26:27]
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, s[22:23]
	v_cndmask_b32_e64 v111, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v110
	v_subb_co_u32_e32 v68, vcc, v93, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v111, v111, v112, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v111
	v_subrev_co_u32_e64 v111, s[22:23], s52, v109
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[22:23], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v88
	v_cndmask_b32_e64 v89, v110, v89, s[24:25]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v110, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v93, v93, v110, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v89, vcc
	v_cndmask_b32_e64 v89, v109, v111, s[24:25]
	v_cndmask_b32_e32 v88, v88, v89, vcc
	v_xor_b32_e32 v89, v68, v92
	v_xor_b32_e32 v68, v88, v92
	v_sub_co_u32_e32 v68, vcc, v68, v92
	s_nop 1
	v_subb_co_u32_e32 v89, vcc, v89, v92, vcc
	v_ashrrev_i32_e32 v88, 31, v89
	v_and_b32_e32 v88, s42, v88
	v_mul_lo_u32 v89, v91, s33
	v_add3_u32 v68, v88, v68, v89
	v_lshl_add_u64 v[88:89], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xf3fb
	buffer_load_ubyte v109, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cndmask_b32_e64 v92, v88, v68, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v93, v89, v91, s[18:19]
	v_ashrrev_i32_e32 v110, 31, v93
	v_mov_b32_e32 v111, v110
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[110:111]
	v_xor_b32_e32 v111, v92, v110
	v_xor_b32_e32 v91, v93, v110
	v_mad_u64_u32 v[92:93], s[6:7], v111, s5, 0
	v_mul_hi_u32 v68, v111, s4
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[92:93]
	v_mad_u64_u32 v[114:115], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v92, v114
	v_mad_u64_u32 v[112:113], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v93, v115, vcc
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[112:113]
	v_mul_lo_u32 v68, s53, v92
	v_mul_lo_u32 v93, s52, v93
	v_mad_u64_u32 v[112:113], s[6:7], s52, v92, 0
	v_add3_u32 v68, v113, v93, v68
	v_sub_co_u32_e32 v93, vcc, v111, v112
	v_cmp_le_u32_e64 s[24:25], s52, v93
	s_nop 0
	v_subb_co_u32_e64 v111, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v91, vcc, s52, v93
	v_cmp_le_u32_e64 s[22:23], s53, v111
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v112, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v111
	v_cndmask_b32_e64 v111, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v91
	v_cndmask_b32_e64 v111, v112, v111, s[22:23]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v111
	v_cndmask_b32_e32 v68, v93, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v91, s64, v110
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v92, v68
	v_cndmask_b32_e64 v68, v92, v68, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v89
	v_mov_b32_e32 v93, v92
	v_xor_b32_e32 v68, v68, v91
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[92:93]
	v_sub_u32_e32 v68, v68, v91
	v_xor_b32_e32 v114, v88, v92
	v_xad_u32 v91, v68, v90, v30
	v_xor_b32_e32 v93, v89, v92
	v_mad_u64_u32 v[88:89], s[6:7], v114, s5, 0
	v_mul_hi_u32 v68, v114, s4
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[88:89]
	v_mad_u64_u32 v[112:113], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v88, v112
	v_mad_u64_u32 v[110:111], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v89, v113, vcc
	s_nop 1
	v_addc_co_u32_e32 v111, vcc, 0, v111, vcc
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[110:111]
	v_mul_lo_u32 v68, s53, v88
	v_mul_lo_u32 v110, s52, v89
	v_mad_u64_u32 v[88:89], s[6:7], s52, v88, 0
	v_add3_u32 v68, v89, v110, v68
	v_sub_u32_e32 v89, v93, v68
	v_sub_co_u32_e32 v88, vcc, v114, v88
	s_mov_b64 s[6:7], 0xc05
	s_nop 0
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, vcc
	v_subrev_co_u32_e64 v110, s[22:23], s52, v88
	v_cmp_le_u32_e64 s[26:27], s52, v110
	s_nop 0
	v_subbrev_co_u32_e64 v111, s[24:25], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v111
	v_cndmask_b32_e64 v113, 0, -1, s[26:27]
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, s[22:23]
	v_cndmask_b32_e64 v112, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v111
	v_subb_co_u32_e32 v68, vcc, v93, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v112, v112, v113, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v112
	v_subrev_co_u32_e64 v112, s[22:23], s52, v110
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[22:23], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v88
	v_cndmask_b32_e64 v89, v111, v89, s[24:25]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v111, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v93, v93, v111, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v89, vcc
	v_cndmask_b32_e64 v89, v110, v112, s[24:25]
	v_cndmask_b32_e32 v88, v88, v89, vcc
	v_xor_b32_e32 v89, v68, v92
	v_xor_b32_e32 v68, v88, v92
	v_sub_co_u32_e32 v68, vcc, v68, v92
	s_nop 1
	v_subb_co_u32_e32 v89, vcc, v89, v92, vcc
	v_ashrrev_i32_e32 v88, 31, v89
	v_and_b32_e32 v88, s42, v88
	v_mul_lo_u32 v89, v91, s33
	v_add3_u32 v68, v88, v68, v89
	v_lshl_add_u64 v[88:89], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xf3fa
	buffer_load_ubyte v110, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cndmask_b32_e64 v92, v88, v68, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v93, v89, v91, s[18:19]
	v_ashrrev_i32_e32 v112, 31, v93
	v_mov_b32_e32 v113, v112
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[112:113]
	v_xor_b32_e32 v111, v92, v112
	v_xor_b32_e32 v91, v93, v112
	v_mad_u64_u32 v[92:93], s[6:7], v111, s5, 0
	v_mul_hi_u32 v68, v111, s4
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[92:93]
	v_mad_u64_u32 v[116:117], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v92, v116
	v_mad_u64_u32 v[114:115], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v93, v117, vcc
	s_nop 1
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[114:115]
	v_mul_lo_u32 v68, s53, v92
	v_mul_lo_u32 v93, s52, v93
	v_mad_u64_u32 v[114:115], s[6:7], s52, v92, 0
	v_add3_u32 v68, v115, v93, v68
	v_sub_co_u32_e32 v93, vcc, v111, v114
	v_cmp_le_u32_e64 s[24:25], s52, v93
	s_nop 0
	v_subb_co_u32_e64 v111, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v91, vcc, s52, v93
	v_cmp_le_u32_e64 s[22:23], s53, v111
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v113, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v111
	v_cndmask_b32_e64 v111, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v91
	v_cndmask_b32_e64 v111, v113, v111, s[22:23]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v111
	v_cndmask_b32_e32 v68, v93, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v91, s64, v112
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v92, v68
	v_cndmask_b32_e64 v68, v92, v68, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v89
	v_mov_b32_e32 v93, v92
	v_xor_b32_e32 v68, v68, v91
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[92:93]
	v_sub_u32_e32 v68, v68, v91
	v_xor_b32_e32 v111, v88, v92
	v_xad_u32 v91, v68, v90, v30
	v_xor_b32_e32 v93, v89, v92
	v_mad_u64_u32 v[88:89], s[6:7], v111, s5, 0
	v_mul_hi_u32 v68, v111, s4
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[88:89]
	v_mad_u64_u32 v[114:115], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v88, v114
	v_mad_u64_u32 v[112:113], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v89, v115, vcc
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[112:113]
	v_mul_lo_u32 v68, s53, v88
	v_mul_lo_u32 v112, s52, v89
	v_mad_u64_u32 v[88:89], s[6:7], s52, v88, 0
	v_add3_u32 v68, v89, v112, v68
	v_sub_u32_e32 v89, v93, v68
	v_sub_co_u32_e32 v88, vcc, v111, v88
	s_mov_b64 s[6:7], 0xc06
	s_nop 0
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, vcc
	v_subrev_co_u32_e64 v111, s[22:23], s52, v88
	v_cmp_le_u32_e64 s[26:27], s52, v111
	s_nop 0
	v_subbrev_co_u32_e64 v112, s[24:25], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v112
	v_cndmask_b32_e64 v114, 0, -1, s[26:27]
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, s[22:23]
	v_cndmask_b32_e64 v113, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v112
	v_subb_co_u32_e32 v68, vcc, v93, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v113, v113, v114, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v113
	v_subrev_co_u32_e64 v113, s[22:23], s52, v111
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[22:23], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v88
	v_cndmask_b32_e64 v89, v112, v89, s[24:25]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v112, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v93, v93, v112, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v89, vcc
	v_cndmask_b32_e64 v89, v111, v113, s[24:25]
	v_cndmask_b32_e32 v88, v88, v89, vcc
	v_xor_b32_e32 v89, v68, v92
	v_xor_b32_e32 v68, v88, v92
	v_sub_co_u32_e32 v68, vcc, v68, v92
	s_nop 1
	v_subb_co_u32_e32 v89, vcc, v89, v92, vcc
	v_ashrrev_i32_e32 v88, 31, v89
	v_and_b32_e32 v88, s42, v88
	v_mul_lo_u32 v89, v91, s33
	v_add3_u32 v68, v88, v68, v89
	v_lshl_add_u64 v[88:89], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xf3f9
	buffer_load_ubyte v111, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cndmask_b32_e64 v92, v88, v68, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v93, v89, v91, s[18:19]
	v_ashrrev_i32_e32 v112, 31, v93
	v_mov_b32_e32 v113, v112
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[112:113]
	v_xor_b32_e32 v113, v92, v112
	v_xor_b32_e32 v91, v93, v112
	v_mad_u64_u32 v[92:93], s[6:7], v113, s5, 0
	v_mul_hi_u32 v68, v113, s4
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[92:93]
	v_mad_u64_u32 v[116:117], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v92, v116
	v_mad_u64_u32 v[114:115], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v93, v117, vcc
	s_nop 1
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[114:115]
	v_mul_lo_u32 v68, s53, v92
	v_mul_lo_u32 v93, s52, v93
	v_mad_u64_u32 v[114:115], s[6:7], s52, v92, 0
	v_add3_u32 v68, v115, v93, v68
	v_sub_co_u32_e32 v93, vcc, v113, v114
	v_cmp_le_u32_e64 s[24:25], s52, v93
	s_nop 0
	v_subb_co_u32_e64 v113, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v91, vcc, s52, v93
	v_cmp_le_u32_e64 s[22:23], s53, v113
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v114, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v113
	v_cndmask_b32_e64 v113, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v91
	v_cndmask_b32_e64 v113, v114, v113, s[22:23]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v113
	v_cndmask_b32_e32 v68, v93, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v91, s64, v112
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v92, v68
	v_cndmask_b32_e64 v68, v92, v68, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v89
	v_mov_b32_e32 v93, v92
	v_xor_b32_e32 v68, v68, v91
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[92:93]
	v_sub_u32_e32 v68, v68, v91
	v_xor_b32_e32 v116, v88, v92
	v_xad_u32 v91, v68, v90, v30
	v_xor_b32_e32 v93, v89, v92
	v_mad_u64_u32 v[88:89], s[6:7], v116, s5, 0
	v_mul_hi_u32 v68, v116, s4
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[88:89]
	v_mad_u64_u32 v[114:115], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v88, v114
	v_mad_u64_u32 v[112:113], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v89, v115, vcc
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[112:113]
	v_mul_lo_u32 v68, s53, v88
	v_mul_lo_u32 v112, s52, v89
	v_mad_u64_u32 v[88:89], s[6:7], s52, v88, 0
	v_add3_u32 v68, v89, v112, v68
	v_sub_u32_e32 v89, v93, v68
	v_sub_co_u32_e32 v88, vcc, v116, v88
	s_mov_b64 s[6:7], 0xc07
	s_nop 0
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, vcc
	v_subrev_co_u32_e64 v112, s[22:23], s52, v88
	v_cmp_le_u32_e64 s[26:27], s52, v112
	s_nop 0
	v_subbrev_co_u32_e64 v113, s[24:25], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v113
	v_cndmask_b32_e64 v115, 0, -1, s[26:27]
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, s[22:23]
	v_cndmask_b32_e64 v114, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v113
	v_subb_co_u32_e32 v68, vcc, v93, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v114, v114, v115, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v114
	v_subrev_co_u32_e64 v114, s[22:23], s52, v112
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[22:23], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v88
	v_cndmask_b32_e64 v89, v113, v89, s[24:25]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v113, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v93, v93, v113, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v89, vcc
	v_cndmask_b32_e64 v89, v112, v114, s[24:25]
	v_cndmask_b32_e32 v88, v88, v89, vcc
	v_xor_b32_e32 v89, v68, v92
	v_xor_b32_e32 v68, v88, v92
	v_sub_co_u32_e32 v68, vcc, v68, v92
	s_nop 1
	v_subb_co_u32_e32 v89, vcc, v89, v92, vcc
	v_ashrrev_i32_e32 v88, 31, v89
	v_and_b32_e32 v88, s42, v88
	v_mul_lo_u32 v89, v91, s33
	v_add3_u32 v68, v88, v68, v89
	v_lshl_add_u64 v[88:89], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xf3f8
	buffer_load_ubyte v112, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cndmask_b32_e64 v92, v88, v68, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v93, v89, v91, s[18:19]
	v_ashrrev_i32_e32 v114, 31, v93
	v_mov_b32_e32 v115, v114
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[114:115]
	v_xor_b32_e32 v113, v92, v114
	v_xor_b32_e32 v91, v93, v114
	v_mad_u64_u32 v[92:93], s[6:7], v113, s5, 0
	v_mul_hi_u32 v68, v113, s4
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[92:93]
	v_mad_u64_u32 v[118:119], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v92, v118
	v_mad_u64_u32 v[116:117], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v93, v119, vcc
	s_nop 1
	v_addc_co_u32_e32 v117, vcc, 0, v117, vcc
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[116:117]
	v_mul_lo_u32 v68, s53, v92
	v_mul_lo_u32 v93, s52, v93
	v_mad_u64_u32 v[116:117], s[6:7], s52, v92, 0
	v_add3_u32 v68, v117, v93, v68
	v_sub_co_u32_e32 v93, vcc, v113, v116
	v_cmp_le_u32_e64 s[24:25], s52, v93
	s_nop 0
	v_subb_co_u32_e64 v113, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v91, vcc, s52, v93
	v_cmp_le_u32_e64 s[22:23], s53, v113
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v115, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v113
	v_cndmask_b32_e64 v113, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v91
	v_cndmask_b32_e64 v113, v115, v113, s[22:23]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v113
	v_cndmask_b32_e32 v68, v93, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v91, s64, v114
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v92, v68
	v_cndmask_b32_e64 v68, v92, v68, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v89
	v_mov_b32_e32 v93, v92
	v_xor_b32_e32 v68, v68, v91
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[92:93]
	v_sub_u32_e32 v68, v68, v91
	v_xor_b32_e32 v113, v88, v92
	v_xad_u32 v91, v68, v90, v30
	v_xor_b32_e32 v93, v89, v92
	v_mad_u64_u32 v[88:89], s[6:7], v113, s5, 0
	v_mul_hi_u32 v68, v113, s4
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[88:89]
	v_mad_u64_u32 v[116:117], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v88, v116
	v_mad_u64_u32 v[114:115], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v89, v117, vcc
	s_nop 1
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[114:115]
	v_mul_lo_u32 v68, s53, v88
	v_mul_lo_u32 v114, s52, v89
	v_mad_u64_u32 v[88:89], s[6:7], s52, v88, 0
	v_add3_u32 v68, v89, v114, v68
	v_sub_u32_e32 v89, v93, v68
	v_sub_co_u32_e32 v88, vcc, v113, v88
	s_mov_b64 s[6:7], 0xc08
	s_nop 0
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, vcc
	v_subrev_co_u32_e64 v113, s[22:23], s52, v88
	v_cmp_le_u32_e64 s[26:27], s52, v113
	s_nop 0
	v_subbrev_co_u32_e64 v114, s[24:25], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v114
	v_cndmask_b32_e64 v116, 0, -1, s[26:27]
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, s[22:23]
	v_cndmask_b32_e64 v115, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v114
	v_subb_co_u32_e32 v68, vcc, v93, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v115, v115, v116, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v115
	v_subrev_co_u32_e64 v115, s[22:23], s52, v113
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[22:23], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v88
	v_cndmask_b32_e64 v89, v114, v89, s[24:25]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v114, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v93, v93, v114, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v89, vcc
	v_cndmask_b32_e64 v89, v113, v115, s[24:25]
	v_cndmask_b32_e32 v88, v88, v89, vcc
	v_xor_b32_e32 v89, v68, v92
	v_xor_b32_e32 v68, v88, v92
	v_sub_co_u32_e32 v68, vcc, v68, v92
	s_nop 1
	v_subb_co_u32_e32 v89, vcc, v89, v92, vcc
	v_ashrrev_i32_e32 v88, 31, v89
	v_and_b32_e32 v88, s42, v88
	v_mul_lo_u32 v89, v91, s33
	v_add3_u32 v68, v88, v68, v89
	v_lshl_add_u64 v[88:89], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xf3f7
	buffer_load_ubyte v113, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cndmask_b32_e64 v92, v88, v68, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v93, v89, v91, s[18:19]
	v_ashrrev_i32_e32 v114, 31, v93
	v_mov_b32_e32 v115, v114
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[114:115]
	v_xor_b32_e32 v115, v92, v114
	v_xor_b32_e32 v91, v93, v114
	v_mad_u64_u32 v[92:93], s[6:7], v115, s5, 0
	v_mul_hi_u32 v68, v115, s4
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[92:93]
	v_mad_u64_u32 v[118:119], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v92, v118
	v_mad_u64_u32 v[116:117], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v93, v119, vcc
	s_nop 1
	v_addc_co_u32_e32 v117, vcc, 0, v117, vcc
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[116:117]
	v_mul_lo_u32 v68, s53, v92
	v_mul_lo_u32 v93, s52, v93
	v_mad_u64_u32 v[116:117], s[6:7], s52, v92, 0
	v_add3_u32 v68, v117, v93, v68
	v_sub_co_u32_e32 v93, vcc, v115, v116
	v_cmp_le_u32_e64 s[24:25], s52, v93
	s_nop 0
	v_subb_co_u32_e64 v115, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v91, vcc, s52, v93
	v_cmp_le_u32_e64 s[22:23], s53, v115
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v116, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v115
	v_cndmask_b32_e64 v115, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v91
	v_cndmask_b32_e64 v115, v116, v115, s[22:23]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v115
	v_cndmask_b32_e32 v68, v93, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v91, s64, v114
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v92, v68
	v_cndmask_b32_e64 v68, v92, v68, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v89
	v_mov_b32_e32 v93, v92
	v_xor_b32_e32 v68, v68, v91
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[92:93]
	v_sub_u32_e32 v68, v68, v91
	v_xor_b32_e32 v118, v88, v92
	v_xad_u32 v91, v68, v90, v30
	v_xor_b32_e32 v93, v89, v92
	v_mad_u64_u32 v[88:89], s[6:7], v118, s5, 0
	v_mul_hi_u32 v68, v118, s4
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[88:89]
	v_mad_u64_u32 v[116:117], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v88, v116
	v_mad_u64_u32 v[114:115], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v89, v117, vcc
	s_nop 1
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[114:115]
	v_mul_lo_u32 v68, s53, v88
	v_mul_lo_u32 v114, s52, v89
	v_mad_u64_u32 v[88:89], s[6:7], s52, v88, 0
	v_add3_u32 v68, v89, v114, v68
	v_sub_u32_e32 v89, v93, v68
	v_sub_co_u32_e32 v88, vcc, v118, v88
	s_mov_b64 s[6:7], 0xc09
	s_nop 0
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, vcc
	v_subrev_co_u32_e64 v114, s[22:23], s52, v88
	v_cmp_le_u32_e64 s[26:27], s52, v114
	s_nop 0
	v_subbrev_co_u32_e64 v115, s[24:25], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v115
	v_cndmask_b32_e64 v117, 0, -1, s[26:27]
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, s[22:23]
	v_cndmask_b32_e64 v116, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v115
	v_subb_co_u32_e32 v68, vcc, v93, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v116, v116, v117, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v116
	v_subrev_co_u32_e64 v116, s[22:23], s52, v114
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[22:23], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v88
	v_cndmask_b32_e64 v89, v115, v89, s[24:25]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v115, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v93, v93, v115, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v89, vcc
	v_cndmask_b32_e64 v89, v114, v116, s[24:25]
	v_cndmask_b32_e32 v88, v88, v89, vcc
	v_xor_b32_e32 v89, v68, v92
	v_xor_b32_e32 v68, v88, v92
	v_sub_co_u32_e32 v68, vcc, v68, v92
	s_nop 1
	v_subb_co_u32_e32 v89, vcc, v89, v92, vcc
	v_ashrrev_i32_e32 v88, 31, v89
	v_and_b32_e32 v88, s42, v88
	v_mul_lo_u32 v89, v91, s33
	v_add3_u32 v68, v88, v68, v89
	v_lshl_add_u64 v[88:89], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xf3f6
	buffer_load_ubyte v114, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cndmask_b32_e64 v92, v88, v68, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v93, v89, v91, s[18:19]
	v_ashrrev_i32_e32 v116, 31, v93
	v_mov_b32_e32 v117, v116
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[116:117]
	v_xor_b32_e32 v115, v92, v116
	v_xor_b32_e32 v91, v93, v116
	v_mad_u64_u32 v[92:93], s[6:7], v115, s5, 0
	v_mul_hi_u32 v68, v115, s4
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[92:93]
	v_mad_u64_u32 v[120:121], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v92, v120
	v_mad_u64_u32 v[118:119], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v93, v121, vcc
	s_nop 1
	v_addc_co_u32_e32 v119, vcc, 0, v119, vcc
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[118:119]
	v_mul_lo_u32 v68, s53, v92
	v_mul_lo_u32 v93, s52, v93
	v_mad_u64_u32 v[118:119], s[6:7], s52, v92, 0
	v_add3_u32 v68, v119, v93, v68
	v_sub_co_u32_e32 v93, vcc, v115, v118
	v_cmp_le_u32_e64 s[24:25], s52, v93
	s_nop 0
	v_subb_co_u32_e64 v115, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v91, vcc, s52, v93
	v_cmp_le_u32_e64 s[22:23], s53, v115
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v117, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v115
	v_cndmask_b32_e64 v115, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v91
	v_cndmask_b32_e64 v115, v117, v115, s[22:23]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v115
	v_cndmask_b32_e32 v68, v93, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v91, s64, v116
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v92, v68
	v_cndmask_b32_e64 v68, v92, v68, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v89
	v_mov_b32_e32 v93, v92
	v_xor_b32_e32 v68, v68, v91
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[92:93]
	v_sub_u32_e32 v68, v68, v91
	v_xor_b32_e32 v115, v88, v92
	v_xad_u32 v91, v68, v90, v30
	v_xor_b32_e32 v93, v89, v92
	v_mad_u64_u32 v[88:89], s[6:7], v115, s5, 0
	v_mul_hi_u32 v68, v115, s4
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[88:89]
	v_mad_u64_u32 v[118:119], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v88, v118
	v_mad_u64_u32 v[116:117], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v89, v119, vcc
	s_nop 1
	v_addc_co_u32_e32 v117, vcc, 0, v117, vcc
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[116:117]
	v_mul_lo_u32 v68, s53, v88
	v_mul_lo_u32 v116, s52, v89
	v_mad_u64_u32 v[88:89], s[6:7], s52, v88, 0
	v_add3_u32 v68, v89, v116, v68
	v_sub_u32_e32 v89, v93, v68
	v_sub_co_u32_e32 v88, vcc, v115, v88
	s_mov_b64 s[6:7], 0xc0a
	s_nop 0
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, vcc
	v_subrev_co_u32_e64 v115, s[22:23], s52, v88
	v_cmp_le_u32_e64 s[26:27], s52, v115
	s_nop 0
	v_subbrev_co_u32_e64 v116, s[24:25], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v116
	v_cndmask_b32_e64 v118, 0, -1, s[26:27]
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, s[22:23]
	v_cndmask_b32_e64 v117, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v116
	v_subb_co_u32_e32 v68, vcc, v93, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v117, v117, v118, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v117
	v_subrev_co_u32_e64 v117, s[22:23], s52, v115
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[22:23], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v88
	v_cndmask_b32_e64 v89, v116, v89, s[24:25]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v116, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v93, v93, v116, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v89, vcc
	v_cndmask_b32_e64 v89, v115, v117, s[24:25]
	v_cndmask_b32_e32 v88, v88, v89, vcc
	v_xor_b32_e32 v89, v68, v92
	v_xor_b32_e32 v68, v88, v92
	v_sub_co_u32_e32 v68, vcc, v68, v92
	s_nop 1
	v_subb_co_u32_e32 v89, vcc, v89, v92, vcc
	v_ashrrev_i32_e32 v88, 31, v89
	v_and_b32_e32 v88, s42, v88
	v_mul_lo_u32 v89, v91, s33
	v_add3_u32 v68, v88, v68, v89
	v_lshl_add_u64 v[88:89], v[86:87], 0, s[6:7]
	s_movk_i32 s6, 0xf3f5
	buffer_load_ubyte v115, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_cndmask_b32_e64 v92, v88, v68, s[18:19]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v93, v89, v91, s[18:19]
	v_ashrrev_i32_e32 v116, 31, v93
	v_mov_b32_e32 v117, v116
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[116:117]
	v_xor_b32_e32 v117, v92, v116
	v_xor_b32_e32 v91, v93, v116
	v_mad_u64_u32 v[92:93], s[6:7], v117, s5, 0
	v_mul_hi_u32 v68, v117, s4
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[92:93]
	v_mad_u64_u32 v[120:121], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v92, v120
	v_mad_u64_u32 v[118:119], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v93, v121, vcc
	s_nop 1
	v_addc_co_u32_e32 v119, vcc, 0, v119, vcc
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[118:119]
	v_mul_lo_u32 v68, s53, v92
	v_mul_lo_u32 v93, s52, v93
	v_mad_u64_u32 v[118:119], s[6:7], s52, v92, 0
	v_add3_u32 v68, v119, v93, v68
	v_sub_co_u32_e32 v93, vcc, v117, v118
	v_cmp_le_u32_e64 s[24:25], s52, v93
	s_nop 0
	v_subb_co_u32_e64 v117, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v91, vcc, s52, v93
	v_cmp_le_u32_e64 s[22:23], s53, v117
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v118, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v117
	v_cndmask_b32_e64 v117, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v91
	v_cndmask_b32_e64 v117, v118, v117, s[22:23]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v117
	v_cndmask_b32_e32 v68, v93, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v91, s64, v116
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v92, v68
	v_cndmask_b32_e64 v68, v92, v68, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v89
	v_mov_b32_e32 v93, v92
	v_xor_b32_e32 v68, v68, v91
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[92:93]
	v_sub_u32_e32 v68, v68, v91
	v_xor_b32_e32 v120, v88, v92
	v_xad_u32 v91, v68, v90, v30
	v_xor_b32_e32 v93, v89, v92
	v_mad_u64_u32 v[88:89], s[6:7], v120, s5, 0
	v_mul_hi_u32 v68, v120, s4
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[88:89]
	v_mad_u64_u32 v[118:119], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v88, v118
	v_mad_u64_u32 v[116:117], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v89, v119, vcc
	s_nop 1
	v_addc_co_u32_e32 v117, vcc, 0, v117, vcc
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[116:117]
	v_mul_lo_u32 v68, s53, v88
	v_mul_lo_u32 v116, s52, v89
	v_mad_u64_u32 v[88:89], s[6:7], s52, v88, 0
	v_add3_u32 v68, v89, v116, v68
	v_sub_u32_e32 v89, v93, v68
	v_sub_co_u32_e32 v88, vcc, v120, v88
	s_movk_i32 s6, 0xf3f4
	s_nop 0
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, vcc
	v_subrev_co_u32_e64 v116, s[22:23], s52, v88
	v_cmp_le_u32_e64 s[26:27], s52, v116
	s_nop 0
	v_subbrev_co_u32_e64 v117, s[24:25], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v117
	v_cndmask_b32_e64 v119, 0, -1, s[26:27]
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, s[22:23]
	v_cndmask_b32_e64 v118, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v117
	v_subb_co_u32_e32 v68, vcc, v93, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v118, v118, v119, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v118
	v_subrev_co_u32_e64 v118, s[22:23], s52, v116
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[22:23], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v88
	v_cndmask_b32_e64 v89, v117, v89, s[24:25]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v117, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v93, v93, v117, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v89, vcc
	v_cndmask_b32_e64 v89, v116, v118, s[24:25]
	v_cndmask_b32_e32 v88, v88, v89, vcc
	v_xor_b32_e32 v89, v68, v92
	v_xor_b32_e32 v68, v88, v92
	v_sub_co_u32_e32 v68, vcc, v68, v92
	s_nop 1
	v_subb_co_u32_e32 v89, vcc, v89, v92, vcc
	v_ashrrev_i32_e32 v88, 31, v89
	v_and_b32_e32 v88, s42, v88
	v_mul_lo_u32 v89, v91, s33
	v_add3_u32 v68, v88, v68, v89
	buffer_load_ubyte v116, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_lshl_add_u64 v[88:89], v[86:87], 0, s[56:57]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v93, v89, v91, s[18:19]
	v_ashrrev_i32_e32 v118, 31, v93
	v_cndmask_b32_e64 v92, v88, v68, s[18:19]
	v_mov_b32_e32 v119, v118
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[118:119]
	v_xor_b32_e32 v117, v92, v118
	v_xor_b32_e32 v91, v93, v118
	v_mad_u64_u32 v[92:93], s[6:7], v117, s5, 0
	v_mul_hi_u32 v68, v117, s4
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[92:93]
	v_mad_u64_u32 v[126:127], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v92, v126
	v_mad_u64_u32 v[120:121], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v93, v127, vcc
	s_nop 1
	v_addc_co_u32_e32 v121, vcc, 0, v121, vcc
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[120:121]
	v_mul_lo_u32 v68, s53, v92
	v_mul_lo_u32 v93, s52, v93
	v_mad_u64_u32 v[120:121], s[6:7], s52, v92, 0
	v_add3_u32 v68, v121, v93, v68
	v_sub_co_u32_e32 v93, vcc, v117, v120
	v_cmp_le_u32_e64 s[24:25], s52, v93
	s_nop 0
	v_subb_co_u32_e64 v117, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v91, vcc, s52, v93
	v_cmp_le_u32_e64 s[22:23], s53, v117
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v119, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v117
	v_cndmask_b32_e64 v117, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v91
	v_cndmask_b32_e64 v117, v119, v117, s[22:23]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v117
	v_cndmask_b32_e32 v68, v93, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v91, s64, v118
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v92, v68
	v_cndmask_b32_e64 v68, v92, v68, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v89
	v_mov_b32_e32 v93, v92
	v_xor_b32_e32 v68, v68, v91
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[92:93]
	v_sub_u32_e32 v68, v68, v91
	v_xor_b32_e32 v117, v88, v92
	v_xad_u32 v91, v68, v90, v30
	v_xor_b32_e32 v93, v89, v92
	v_mad_u64_u32 v[88:89], s[6:7], v117, s5, 0
	v_mul_hi_u32 v68, v117, s4
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[88:89]
	v_mad_u64_u32 v[120:121], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v88, v120
	v_mad_u64_u32 v[118:119], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v89, v121, vcc
	s_nop 1
	v_addc_co_u32_e32 v119, vcc, 0, v119, vcc
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[118:119]
	v_mul_lo_u32 v68, s53, v88
	v_mul_lo_u32 v118, s52, v89
	v_mad_u64_u32 v[88:89], s[6:7], s52, v88, 0
	v_add3_u32 v68, v89, v118, v68
	v_sub_u32_e32 v89, v93, v68
	v_sub_co_u32_e32 v88, vcc, v117, v88
	s_movk_i32 s6, 0xf3f3
	s_nop 0
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, vcc
	v_subrev_co_u32_e64 v117, s[22:23], s52, v88
	v_cmp_le_u32_e64 s[26:27], s52, v117
	s_nop 0
	v_subbrev_co_u32_e64 v118, s[24:25], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v118
	v_cndmask_b32_e64 v120, 0, -1, s[26:27]
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, s[22:23]
	v_cndmask_b32_e64 v119, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v118
	v_subb_co_u32_e32 v68, vcc, v93, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v119, v119, v120, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v119
	v_subrev_co_u32_e64 v119, s[22:23], s52, v117
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[22:23], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v88
	v_cndmask_b32_e64 v89, v118, v89, s[24:25]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v118, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v93, v93, v118, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v89, vcc
	v_cndmask_b32_e64 v89, v117, v119, s[24:25]
	v_cndmask_b32_e32 v88, v88, v89, vcc
	v_xor_b32_e32 v89, v68, v92
	v_xor_b32_e32 v68, v88, v92
	v_sub_co_u32_e32 v68, vcc, v68, v92
	s_nop 1
	v_subb_co_u32_e32 v89, vcc, v89, v92, vcc
	v_ashrrev_i32_e32 v88, 31, v89
	v_and_b32_e32 v88, s42, v88
	v_mul_lo_u32 v89, v91, s33
	v_add3_u32 v68, v88, v68, v89
	buffer_load_ubyte v117, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_lshl_add_u64 v[88:89], v[86:87], 0, s[20:21]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v93, v89, v91, s[18:19]
	v_ashrrev_i32_e32 v118, 31, v93
	v_cndmask_b32_e64 v92, v88, v68, s[18:19]
	v_mov_b32_e32 v119, v118
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[118:119]
	v_xor_b32_e32 v119, v92, v118
	v_xor_b32_e32 v91, v93, v118
	v_mad_u64_u32 v[92:93], s[6:7], v119, s5, 0
	v_mul_hi_u32 v68, v119, s4
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[92:93]
	v_mad_u64_u32 v[126:127], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v92, v126
	v_mad_u64_u32 v[120:121], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v93, v127, vcc
	s_nop 1
	v_addc_co_u32_e32 v121, vcc, 0, v121, vcc
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[120:121]
	v_mul_lo_u32 v68, s53, v92
	v_mul_lo_u32 v93, s52, v93
	v_mad_u64_u32 v[120:121], s[6:7], s52, v92, 0
	v_add3_u32 v68, v121, v93, v68
	v_sub_co_u32_e32 v93, vcc, v119, v120
	v_cmp_le_u32_e64 s[24:25], s52, v93
	s_nop 0
	v_subb_co_u32_e64 v119, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v91, vcc, s52, v93
	v_cmp_le_u32_e64 s[22:23], s53, v119
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v120, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v119
	v_cndmask_b32_e64 v119, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v91
	v_cndmask_b32_e64 v119, v120, v119, s[22:23]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v119
	v_cndmask_b32_e32 v68, v93, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v91, s64, v118
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v92, v68
	v_cndmask_b32_e64 v68, v92, v68, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v89
	v_mov_b32_e32 v93, v92
	v_xor_b32_e32 v68, v68, v91
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[92:93]
	v_sub_u32_e32 v68, v68, v91
	v_xor_b32_e32 v126, v88, v92
	v_xad_u32 v91, v68, v90, v30
	v_xor_b32_e32 v93, v89, v92
	v_mad_u64_u32 v[88:89], s[6:7], v126, s5, 0
	v_mul_hi_u32 v68, v126, s4
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[88:89]
	v_mad_u64_u32 v[120:121], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v88, v120
	v_mad_u64_u32 v[118:119], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v89, v121, vcc
	s_nop 1
	v_addc_co_u32_e32 v119, vcc, 0, v119, vcc
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[118:119]
	v_mul_lo_u32 v68, s53, v88
	v_mul_lo_u32 v118, s52, v89
	v_mad_u64_u32 v[88:89], s[6:7], s52, v88, 0
	v_add3_u32 v68, v89, v118, v68
	v_sub_u32_e32 v89, v93, v68
	v_sub_co_u32_e32 v88, vcc, v126, v88
	s_movk_i32 s6, 0xf3f2
	s_nop 0
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, vcc
	v_subrev_co_u32_e64 v118, s[22:23], s52, v88
	v_cmp_le_u32_e64 s[26:27], s52, v118
	s_nop 0
	v_subbrev_co_u32_e64 v119, s[24:25], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v119
	v_cndmask_b32_e64 v121, 0, -1, s[26:27]
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, s[22:23]
	v_cndmask_b32_e64 v120, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v119
	v_subb_co_u32_e32 v68, vcc, v93, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v120, v120, v121, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v120
	v_subrev_co_u32_e64 v120, s[22:23], s52, v118
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[22:23], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v88
	v_cndmask_b32_e64 v89, v119, v89, s[24:25]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v119, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v93, v93, v119, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v89, vcc
	v_cndmask_b32_e64 v89, v118, v120, s[24:25]
	v_cndmask_b32_e32 v88, v88, v89, vcc
	v_xor_b32_e32 v89, v68, v92
	v_xor_b32_e32 v68, v88, v92
	v_sub_co_u32_e32 v68, vcc, v68, v92
	s_nop 1
	v_subb_co_u32_e32 v89, vcc, v89, v92, vcc
	v_ashrrev_i32_e32 v88, 31, v89
	v_and_b32_e32 v88, s42, v88
	v_mul_lo_u32 v89, v91, s33
	v_add3_u32 v68, v88, v68, v89
	buffer_load_ubyte v118, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_lshl_add_u64 v[88:89], v[86:87], 0, s[58:59]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v93, v89, v91, s[18:19]
	v_ashrrev_i32_e32 v120, 31, v93
	v_cndmask_b32_e64 v92, v88, v68, s[18:19]
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[120:121]
	v_xor_b32_e32 v119, v92, v120
	v_xor_b32_e32 v91, v93, v120
	v_mad_u64_u32 v[92:93], s[6:7], v119, s5, 0
	v_mul_hi_u32 v68, v119, s4
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[92:93]
	v_mad_u64_u32 v[128:129], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v92, v128
	v_mad_u64_u32 v[126:127], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v93, v129, vcc
	s_nop 1
	v_addc_co_u32_e32 v127, vcc, 0, v127, vcc
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[126:127]
	v_mul_lo_u32 v68, s53, v92
	v_mul_lo_u32 v93, s52, v93
	v_mad_u64_u32 v[126:127], s[6:7], s52, v92, 0
	v_add3_u32 v68, v127, v93, v68
	v_sub_co_u32_e32 v93, vcc, v119, v126
	v_cmp_le_u32_e64 s[24:25], s52, v93
	s_nop 0
	v_subb_co_u32_e64 v119, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v91, vcc, s52, v93
	v_cmp_le_u32_e64 s[22:23], s53, v119
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v121, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v119
	v_cndmask_b32_e64 v119, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v91
	v_cndmask_b32_e64 v119, v121, v119, s[22:23]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v119
	v_cndmask_b32_e32 v68, v93, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v91, s64, v120
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v92, v68
	v_cndmask_b32_e64 v68, v92, v68, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v89
	v_mov_b32_e32 v93, v92
	v_xor_b32_e32 v68, v68, v91
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[92:93]
	v_sub_u32_e32 v68, v68, v91
	v_xor_b32_e32 v119, v88, v92
	v_xad_u32 v91, v68, v90, v30
	v_xor_b32_e32 v93, v89, v92
	v_mad_u64_u32 v[88:89], s[6:7], v119, s5, 0
	v_mul_hi_u32 v68, v119, s4
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[88:89]
	v_mad_u64_u32 v[126:127], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v88, v126
	v_mad_u64_u32 v[120:121], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v89, v127, vcc
	s_nop 1
	v_addc_co_u32_e32 v121, vcc, 0, v121, vcc
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[120:121]
	v_mul_lo_u32 v68, s53, v88
	v_mul_lo_u32 v120, s52, v89
	v_mad_u64_u32 v[88:89], s[6:7], s52, v88, 0
	v_add3_u32 v68, v89, v120, v68
	v_sub_u32_e32 v89, v93, v68
	v_sub_co_u32_e32 v88, vcc, v119, v88
	s_movk_i32 s6, 0xf3f1
	s_nop 0
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, vcc
	v_subrev_co_u32_e64 v119, s[22:23], s52, v88
	v_cmp_le_u32_e64 s[26:27], s52, v119
	s_nop 0
	v_subbrev_co_u32_e64 v120, s[24:25], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v120
	v_cndmask_b32_e64 v126, 0, -1, s[26:27]
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, s[22:23]
	v_cndmask_b32_e64 v121, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v120
	v_subb_co_u32_e32 v68, vcc, v93, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v121, v121, v126, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v121
	v_subrev_co_u32_e64 v121, s[22:23], s52, v119
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[22:23], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v88
	v_cndmask_b32_e64 v89, v120, v89, s[24:25]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v120, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v93, v93, v120, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v89, vcc
	v_cndmask_b32_e64 v89, v119, v121, s[24:25]
	v_cndmask_b32_e32 v88, v88, v89, vcc
	v_xor_b32_e32 v89, v68, v92
	v_xor_b32_e32 v68, v88, v92
	v_sub_co_u32_e32 v68, vcc, v68, v92
	s_nop 1
	v_subb_co_u32_e32 v89, vcc, v89, v92, vcc
	v_ashrrev_i32_e32 v88, 31, v89
	v_and_b32_e32 v88, s42, v88
	v_mul_lo_u32 v89, v91, s33
	v_add3_u32 v68, v88, v68, v89
	buffer_load_ubyte v119, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_lshl_add_u64 v[88:89], v[86:87], 0, s[60:61]
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v93, v89, v91, s[18:19]
	v_ashrrev_i32_e32 v120, 31, v93
	v_cndmask_b32_e64 v92, v88, v68, s[18:19]
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[92:93], v[92:93], 0, v[120:121]
	v_xor_b32_e32 v121, v92, v120
	v_xor_b32_e32 v91, v93, v120
	v_mad_u64_u32 v[92:93], s[6:7], v121, s5, 0
	v_mul_hi_u32 v68, v121, s4
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[92:93]
	v_mad_u64_u32 v[128:129], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v92, v128
	v_mad_u64_u32 v[126:127], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v93, v129, vcc
	s_nop 1
	v_addc_co_u32_e32 v127, vcc, 0, v127, vcc
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[126:127]
	v_mul_lo_u32 v68, s53, v92
	v_mul_lo_u32 v93, s52, v93
	v_mad_u64_u32 v[126:127], s[6:7], s52, v92, 0
	v_add3_u32 v68, v127, v93, v68
	v_sub_co_u32_e32 v93, vcc, v121, v126
	v_cmp_le_u32_e64 s[24:25], s52, v93
	s_nop 0
	v_subb_co_u32_e64 v121, s[22:23], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v91, vcc, s52, v93
	v_cmp_le_u32_e64 s[22:23], s53, v121
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v126, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v121
	v_cndmask_b32_e64 v121, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[24:25], s52, v91
	v_cndmask_b32_e64 v121, v126, v121, s[22:23]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_ne_u32_e64 s[22:23], 0, v121
	v_cndmask_b32_e32 v68, v93, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	v_xor_b32_e32 v91, s64, v120
	s_nop 0
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v92, v68
	v_cndmask_b32_e64 v68, v92, v68, s[22:23]
	v_ashrrev_i32_e32 v92, 31, v89
	v_mov_b32_e32 v93, v92
	v_xor_b32_e32 v68, v68, v91
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[92:93]
	v_sub_u32_e32 v68, v68, v91
	v_xor_b32_e32 v128, v88, v92
	v_xad_u32 v91, v68, v90, v30
	v_xor_b32_e32 v93, v89, v92
	v_mad_u64_u32 v[88:89], s[6:7], v128, s5, 0
	v_mul_hi_u32 v68, v128, s4
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[88:89]
	v_mad_u64_u32 v[126:127], s[6:7], v93, s4, 0
	v_add_co_u32_e32 v68, vcc, v88, v126
	v_mad_u64_u32 v[120:121], s[6:7], v93, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v89, v127, vcc
	s_nop 1
	v_addc_co_u32_e32 v121, vcc, 0, v121, vcc
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[120:121]
	v_mul_lo_u32 v68, s53, v88
	v_mul_lo_u32 v120, s52, v89
	v_mad_u64_u32 v[88:89], s[6:7], s52, v88, 0
	v_add3_u32 v68, v89, v120, v68
	v_sub_u32_e32 v89, v93, v68
	v_sub_co_u32_e32 v88, vcc, v128, v88
	s_movk_i32 s6, 0xf3f0
	s_nop 0
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, vcc
	v_subrev_co_u32_e64 v120, s[22:23], s52, v88
	v_cmp_le_u32_e64 s[26:27], s52, v120
	s_nop 0
	v_subbrev_co_u32_e64 v121, s[24:25], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[24:25], s53, v121
	v_cndmask_b32_e64 v127, 0, -1, s[26:27]
	v_subb_co_u32_e64 v89, s[22:23], v89, v57, s[22:23]
	v_cndmask_b32_e64 v126, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s53, v121
	v_subb_co_u32_e32 v68, vcc, v93, v68, vcc
	s_nop 0
	v_cndmask_b32_e64 v126, v126, v127, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v126
	v_subrev_co_u32_e64 v126, s[22:23], s52, v120
	v_cmp_le_u32_e32 vcc, s53, v68
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[22:23], 0, v89, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s52, v88
	v_cndmask_b32_e64 v89, v121, v89, s[24:25]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v121, 0, -1, s[22:23]
	s_nop 0
	v_cndmask_b32_e32 v93, v93, v121, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v89, vcc
	v_cndmask_b32_e64 v89, v120, v126, s[24:25]
	v_cndmask_b32_e32 v88, v88, v89, vcc
	v_xor_b32_e32 v89, v68, v92
	v_xor_b32_e32 v68, v88, v92
	v_sub_co_u32_e32 v68, vcc, v68, v92
	s_nop 1
	v_subb_co_u32_e32 v89, vcc, v89, v92, vcc
	v_ashrrev_i32_e32 v88, 31, v89
	v_and_b32_e32 v88, s42, v88
	v_mul_lo_u32 v89, v91, s33
	v_add3_u32 v68, v88, v68, v89
	buffer_load_ubyte v120, v68, s[28:31], 0 offen
	v_sub_co_u32_e32 v68, vcc, s6, v86
	v_lshl_add_u64 v[88:89], v[86:87], 0, s[48:49]
	s_nop 0
	v_subb_co_u32_e32 v86, vcc, -1, v87, vcc
	v_cndmask_b32_e64 v87, v89, v86, s[18:19]
	v_ashrrev_i32_e32 v92, 31, v87
	v_cndmask_b32_e64 v86, v88, v68, s[18:19]
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[86:87], v[86:87], 0, v[92:93]
	v_xor_b32_e32 v93, v86, v92
	v_xor_b32_e32 v91, v87, v92
	v_mad_u64_u32 v[86:87], s[6:7], v93, s5, 0
	v_mul_hi_u32 v68, v93, s4
	v_lshl_add_u64 v[86:87], v[68:69], 0, v[86:87]
	v_mad_u64_u32 v[128:129], s[6:7], v91, s4, 0
	v_add_co_u32_e32 v68, vcc, v86, v128
	v_mad_u64_u32 v[126:127], s[6:7], v91, s5, 0
	s_nop 0
	v_addc_co_u32_e32 v68, vcc, v87, v129, vcc
	s_nop 1
	v_addc_co_u32_e32 v127, vcc, 0, v127, vcc
	v_lshl_add_u64 v[86:87], v[68:69], 0, v[126:127]
	v_mul_lo_u32 v68, s53, v86
	v_mul_lo_u32 v87, s52, v87
	v_mad_u64_u32 v[126:127], s[6:7], s52, v86, 0
	v_add3_u32 v68, v127, v87, v68
	v_sub_co_u32_e32 v87, vcc, v93, v126
	v_cmp_le_u32_e64 s[22:23], s52, v87
	s_nop 0
	v_subb_co_u32_e64 v93, s[18:19], v91, v68, vcc
	v_sub_u32_e32 v68, v91, v68
	v_subb_co_u32_e32 v68, vcc, v68, v57, vcc
	v_subrev_co_u32_e32 v87, vcc, s52, v87
	v_cmp_le_u32_e64 s[18:19], s53, v93
	s_nop 0
	v_subbrev_co_u32_e32 v68, vcc, 0, v68, vcc
	v_cndmask_b32_e64 v121, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s53, v93
	v_cndmask_b32_e64 v93, 0, -1, s[22:23]
	v_cmp_le_u32_e32 vcc, s53, v68
	v_cmp_le_u32_e64 s[22:23], s52, v87
	v_cndmask_b32_e64 v93, v121, v93, s[18:19]
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v68
	v_cndmask_b32_e64 v68, 0, -1, s[22:23]
	v_cmp_ne_u32_e64 s[18:19], 0, v93
	v_cndmask_b32_e32 v68, v91, v68, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e64 v68, 1, 2, vcc
	v_add_u32_e32 v68, v86, v68
	v_cndmask_b32_e64 v68, v86, v68, s[18:19]
	v_xor_b32_e32 v86, s64, v92
	v_xor_b32_e32 v68, v68, v86
	v_sub_u32_e32 v68, v68, v86
	v_ashrrev_i32_e32 v86, 31, v89
	v_mov_b32_e32 v87, v86
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[86:87]
	v_xor_b32_e32 v91, v88, v86
	v_xad_u32 v90, v68, v90, v30
	v_xor_b32_e32 v87, v89, v86
	v_mad_u64_u32 v[88:89], s[6:7], v91, s5, 0
	v_mul_hi_u32 v68, v91, s4
	v_lshl_add_u64 v[92:93], v[68:69], 0, v[88:89]
	v_mad_u64_u32 v[88:89], s[6:7], v87, s5, 0
	v_mad_u64_u32 v[126:127], s[4:5], v87, s4, 0
	v_add_co_u32_e32 v68, vcc, v92, v126
	s_nop 1
	v_addc_co_u32_e32 v68, vcc, v93, v127, vcc
	s_nop 1
	v_addc_co_u32_e32 v89, vcc, 0, v89, vcc
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[88:89]
	v_mul_lo_u32 v68, s53, v88
	v_mul_lo_u32 v89, s52, v89
	v_mad_u64_u32 v[92:93], s[4:5], s52, v88, 0
	v_add3_u32 v88, v93, v89, v68
	v_sub_u32_e32 v89, v87, v88
	v_sub_co_u32_e32 v68, vcc, v91, v92
	s_nop 1
	v_subb_co_u32_e64 v89, s[18:19], v89, v57, vcc
	v_subrev_co_u32_e64 v91, s[18:19], s52, v68
	v_cmp_le_u32_e64 s[24:25], s52, v91
	s_nop 0
	v_subbrev_co_u32_e64 v92, s[22:23], 0, v89, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s53, v92
	v_subb_co_u32_e64 v57, s[18:19], v89, v57, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s53, v92
	v_cndmask_b32_e64 v121, 0, -1, s[24:25]
	v_subrev_co_u32_e64 v89, s[18:19], s52, v91
	v_cndmask_b32_e64 v93, v93, v121, s[22:23]
	s_nop 0
	v_subbrev_co_u32_e64 v57, s[18:19], 0, v57, s[18:19]
	v_subb_co_u32_e32 v87, vcc, v87, v88, vcc
	v_cmp_ne_u32_e64 s[22:23], 0, v93
	v_cmp_le_u32_e32 vcc, s53, v87
	v_cmp_le_u32_e64 s[18:19], s52, v68
	v_cndmask_b32_e64 v57, v92, v57, s[22:23]
	v_cndmask_b32_e64 v88, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s53, v87
	v_cndmask_b32_e64 v92, 0, -1, s[18:19]
	ds_read_b128 v[134:137], v50 offset:12288
	ds_read_b128 v[130:133], v52 offset:14336
	ds_read_b128 v[126:129], v50 offset:14336
	ds_read_b32 v121, v97 offset:25344
	v_cndmask_b32_e32 v88, v88, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v88
	s_nop 1
	v_cndmask_b32_e32 v57, v87, v57, vcc
	v_cndmask_b32_e64 v87, v91, v89, s[22:23]
	v_cndmask_b32_e32 v68, v68, v87, vcc
	v_xor_b32_e32 v68, v68, v86
	v_xor_b32_e32 v57, v57, v86
	v_sub_co_u32_e32 v68, vcc, v68, v86
	s_nop 1
	v_subb_co_u32_e32 v57, vcc, v57, v86, vcc
	v_ashrrev_i32_e32 v57, 31, v57
	v_and_b32_e32 v57, s42, v57
	v_mul_lo_u32 v86, v90, s33
	v_add3_u32 v57, v57, v68, v86
	buffer_load_ubyte v57, v57, s[28:31], 0 offen
	v_lshl_add_u64 v[86:87], v[76:77], 0, s[90:91]
	v_sub_co_u32_e32 v68, vcc, 0xfffffdff, v76
	v_cmp_gt_i32_e64 s[18:19], 0, v87
	s_nop 0
	v_subb_co_u32_e32 v88, vcc, -1, v77, vcc
	v_cndmask_b32_e64 v89, v87, v88, s[18:19]
	v_cndmask_b32_e64 v88, v86, v68, s[18:19]
	v_or_b32_e32 v68, s71, v89
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[4:5], exec, s[4:5]
	s_cbranch_execz .LBB0_151
	s_add_u32 s6, s70, s74
	s_mov_b32 s75, s74
	s_addc_u32 s7, s71, s74
	s_xor_b64 s[6:7], s[6:7], s[74:75]
	v_cvt_f32_u32_e32 v68, s6
	v_cvt_f32_u32_e32 v86, s7
	s_sub_u32 s12, 0, s6
	s_subb_u32 s13, 0, s7
	v_ashrrev_i32_e32 v90, 31, v89
	v_fmac_f32_e32 v68, 0x4f800000, v86
	v_rcp_f32_e32 v68, v68
	v_mov_b32_e32 v91, v90
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[90:91]
	v_xor_b32_e32 v91, v88, v90
	v_mul_f32_e32 v68, 0x5f7ffffc, v68
	v_mul_f32_e32 v86, 0x2f800000, v68
	v_trunc_f32_e32 v86, v86
	v_fmac_f32_e32 v68, 0xcf800000, v86
	v_cvt_u32_f32_e32 v86, v86
	v_cvt_u32_f32_e32 v68, v68
	v_readfirstlane_b32 s18, v86
	v_readfirstlane_b32 s19, v68
	s_mul_i32 s22, s12, s18
	s_mul_hi_u32 s24, s12, s19
	s_mul_i32 s23, s13, s19
	s_add_i32 s22, s24, s22
	s_add_i32 s22, s22, s23
	s_mul_i32 s25, s12, s19
	s_mul_i32 s24, s19, s22
	s_mul_hi_u32 s26, s19, s25
	s_mul_hi_u32 s23, s19, s22
	s_add_u32 s24, s26, s24
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 s27, s18, s25
	s_mul_i32 s25, s18, s25
	s_add_u32 s24, s24, s25
	s_mul_hi_u32 s26, s18, s22
	s_addc_u32 s23, s23, s27
	s_addc_u32 s24, s26, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s22, s23, s22
	s_addc_u32 s23, 0, s24
	s_add_u32 s19, s19, s22
	s_addc_u32 s18, s18, s23
	s_mul_i32 s22, s12, s18
	s_mul_hi_u32 s23, s12, s19
	s_add_i32 s22, s23, s22
	s_mul_i32 s13, s13, s19
	s_add_i32 s22, s22, s13
	s_mul_i32 s12, s12, s19
	s_mul_hi_u32 s23, s18, s12
	s_mul_i32 s24, s18, s12
	s_mul_i32 s26, s19, s22
	s_mul_hi_u32 s12, s19, s12
	s_mul_hi_u32 s25, s19, s22
	s_add_u32 s12, s12, s26
	s_addc_u32 s25, 0, s25
	s_add_u32 s12, s12, s24
	s_mul_hi_u32 s13, s18, s22
	s_addc_u32 s12, s25, s23
	s_addc_u32 s13, s13, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s12, s12, s22
	s_addc_u32 s13, 0, s13
	s_add_u32 s19, s19, s12
	s_addc_u32 s18, s18, s13
	v_xor_b32_e32 v86, v89, v90
	v_mad_u64_u32 v[88:89], s[12:13], v91, s18, 0
	v_mul_hi_u32 v68, v91, s19
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[88:89]
	v_mad_u64_u32 v[138:139], s[12:13], v86, s19, 0
	v_add_co_u32_e32 v68, vcc, v88, v138
	v_mad_u64_u32 v[92:93], s[12:13], v86, s18, 0
	s_nop 0
	v_addc_co_u32_e32 v88, vcc, v89, v139, vcc
	v_mov_b32_e32 v89, s41
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[92:93]
	v_mul_lo_u32 v68, s7, v88
	v_mul_lo_u32 v138, s6, v89
	v_mad_u64_u32 v[92:93], s[12:13], s6, v88, 0
	v_add3_u32 v68, v93, v138, v68
	v_sub_u32_e32 v93, v86, v68
	v_mov_b32_e32 v138, s7
	v_sub_co_u32_e32 v91, vcc, v91, v92
	s_nop 1
	v_subb_co_u32_e64 v92, s[18:19], v93, v138, vcc
	v_subrev_co_u32_e64 v93, s[18:19], s6, v91
	v_subb_co_u32_e32 v68, vcc, v86, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v92, s[18:19], 0, v92, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v92
	v_cmp_le_u32_e32 vcc, s7, v68
	s_nop 0
	v_cndmask_b32_e64 v138, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v93
	v_cndmask_b32_e64 v86, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v91
	v_cndmask_b32_e64 v93, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v92
	s_nop 1
	v_cndmask_b32_e64 v140, v138, v93, s[18:19]
	v_lshl_add_u64 v[92:93], v[88:89], 0, 2
	v_lshl_add_u64 v[138:139], v[88:89], 0, 1
	v_cndmask_b32_e64 v89, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v86, v89, vcc
	v_cmp_ne_u32_e32 vcc, 0, v140
	s_nop 1
	v_cndmask_b32_e32 v86, v138, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v88, v86, vcc
	v_xor_b32_e32 v86, s74, v90
	v_xor_b32_e32 v68, v68, v86
	v_sub_co_u32_e32 v90, vcc, v68, v86
.LBB0_151:
	s_andn2_saveexec_b64 s[4:5], s[4:5]
	s_cbranch_execz .LBB0_153
	v_rcp_iflag_f32_e32 v68, v96
	s_sub_i32 s6, 0, s70
	v_mul_f32_e32 v68, 0x4f7ffffe, v68
	v_cvt_u32_f32_e32 v68, v68
	v_mul_lo_u32 v86, s6, v68
	v_mul_hi_u32 v86, v68, v86
	v_add_u32_e32 v68, v68, v86
	v_mul_hi_u32 v68, v88, v68
	v_mul_lo_u32 v86, v68, s70
	v_sub_u32_e32 v86, v88, v86
	v_add_u32_e32 v89, 1, v68
	v_subrev_u32_e32 v88, s70, v86
	v_cmp_le_u32_e32 vcc, s70, v86
	s_nop 1
	v_cndmask_b32_e32 v86, v86, v88, vcc
	v_cndmask_b32_e32 v68, v68, v89, vcc
	v_add_u32_e32 v88, 1, v68
	v_cmp_le_u32_e32 vcc, s70, v86
	s_nop 1
	v_cndmask_b32_e32 v90, v68, v88, vcc
.LBB0_153:
	s_or_b64 exec, exec, s[4:5]
	v_lshl_add_u64 v[88:89], v[78:79], 0, s[90:91]
	v_or_b32_e32 v68, s71, v89
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[4:5], exec, s[4:5]
	s_cbranch_execz .LBB0_155
	s_add_u32 s6, s70, s74
	s_mov_b32 s75, s74
	s_addc_u32 s7, s71, s74
	s_xor_b64 s[6:7], s[6:7], s[74:75]
	v_cvt_f32_u32_e32 v68, s6
	v_cvt_f32_u32_e32 v86, s7
	s_sub_u32 s12, 0, s6
	s_subb_u32 s13, 0, s7
	v_ashrrev_i32_e32 v138, 31, v89
	v_fmac_f32_e32 v68, 0x4f800000, v86
	v_rcp_f32_e32 v68, v68
	v_mov_b32_e32 v139, v138
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[138:139]
	v_xor_b32_e32 v91, v88, v138
	v_mul_f32_e32 v68, 0x5f7ffffc, v68
	v_mul_f32_e32 v86, 0x2f800000, v68
	v_trunc_f32_e32 v86, v86
	v_fmac_f32_e32 v68, 0xcf800000, v86
	v_cvt_u32_f32_e32 v86, v86
	v_cvt_u32_f32_e32 v68, v68
	v_readfirstlane_b32 s18, v86
	v_readfirstlane_b32 s19, v68
	s_mul_i32 s22, s12, s18
	s_mul_hi_u32 s24, s12, s19
	s_mul_i32 s23, s13, s19
	s_add_i32 s22, s24, s22
	s_add_i32 s22, s22, s23
	s_mul_i32 s25, s12, s19
	s_mul_i32 s24, s19, s22
	s_mul_hi_u32 s26, s19, s25
	s_mul_hi_u32 s23, s19, s22
	s_add_u32 s24, s26, s24
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 s27, s18, s25
	s_mul_i32 s25, s18, s25
	s_add_u32 s24, s24, s25
	s_mul_hi_u32 s26, s18, s22
	s_addc_u32 s23, s23, s27
	s_addc_u32 s24, s26, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s22, s23, s22
	s_addc_u32 s23, 0, s24
	s_add_u32 s19, s19, s22
	s_addc_u32 s18, s18, s23
	s_mul_i32 s22, s12, s18
	s_mul_hi_u32 s23, s12, s19
	s_add_i32 s22, s23, s22
	s_mul_i32 s13, s13, s19
	s_add_i32 s22, s22, s13
	s_mul_i32 s12, s12, s19
	s_mul_hi_u32 s23, s18, s12
	s_mul_i32 s24, s18, s12
	s_mul_i32 s26, s19, s22
	s_mul_hi_u32 s12, s19, s12
	s_mul_hi_u32 s25, s19, s22
	s_add_u32 s12, s12, s26
	s_addc_u32 s25, 0, s25
	s_add_u32 s12, s12, s24
	s_mul_hi_u32 s13, s18, s22
	s_addc_u32 s12, s25, s23
	s_addc_u32 s13, s13, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s12, s12, s22
	s_addc_u32 s13, 0, s13
	s_add_u32 s19, s19, s12
	s_addc_u32 s18, s18, s13
	v_xor_b32_e32 v86, v89, v138
	v_mad_u64_u32 v[88:89], s[12:13], v91, s18, 0
	v_mul_hi_u32 v68, v91, s19
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[88:89]
	v_mad_u64_u32 v[140:141], s[12:13], v86, s19, 0
	v_add_co_u32_e32 v68, vcc, v88, v140
	v_mad_u64_u32 v[92:93], s[12:13], v86, s18, 0
	s_nop 0
	v_addc_co_u32_e32 v88, vcc, v89, v141, vcc
	v_mov_b32_e32 v89, s41
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[92:93]
	v_mul_lo_u32 v68, s7, v88
	v_mul_lo_u32 v92, s6, v89
	v_mad_u64_u32 v[88:89], s[12:13], s6, v88, 0
	v_add3_u32 v68, v89, v92, v68
	v_sub_u32_e32 v89, v86, v68
	v_mov_b32_e32 v92, s7
	v_sub_co_u32_e32 v88, vcc, v91, v88
	s_nop 1
	v_subb_co_u32_e64 v89, s[18:19], v89, v92, vcc
	v_subrev_co_u32_e64 v91, s[18:19], s6, v88
	v_subb_co_u32_e32 v68, vcc, v86, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v93, s[22:23], 0, v89, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v93
	v_subb_co_u32_e64 v89, s[18:19], v89, v92, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v139, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v91
	v_subrev_co_u32_e64 v92, s[18:19], s6, v91
	s_nop 0
	v_cndmask_b32_e64 v140, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v93
	v_subbrev_co_u32_e64 v89, s[18:19], 0, v89, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v139, v139, v140, s[22:23]
	v_cmp_le_u32_e32 vcc, s7, v68
	v_cmp_ne_u32_e64 s[18:19], 0, v139
	s_nop 0
	v_cndmask_b32_e64 v86, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v88
	v_cndmask_b32_e64 v89, v93, v89, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v68
	s_nop 1
	v_cndmask_b32_e32 v86, v86, v93, vcc
	v_cmp_ne_u32_e32 vcc, 0, v86
	v_cndmask_b32_e64 v86, v91, v92, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v86, v88, v86, vcc
	v_cndmask_b32_e32 v68, v68, v89, vcc
	v_xor_b32_e32 v86, v86, v138
	v_xor_b32_e32 v68, v68, v138
	v_sub_co_u32_e32 v92, vcc, v86, v138
	s_nop 1
	v_subb_co_u32_e32 v93, vcc, v68, v138, vcc
.LBB0_155:
	s_andn2_saveexec_b64 s[4:5], s[4:5]
	s_cbranch_execz .LBB0_157
	v_rcp_iflag_f32_e32 v68, v96
	s_sub_i32 s6, 0, s70
	v_mul_f32_e32 v68, 0x4f7ffffe, v68
	v_cvt_u32_f32_e32 v68, v68
	v_mul_lo_u32 v86, s6, v68
	v_mul_hi_u32 v86, v68, v86
	v_add_u32_e32 v68, v68, v86
	v_mul_hi_u32 v68, v88, v68
	v_mul_lo_u32 v68, v68, s70
	v_sub_u32_e32 v68, v88, v68
	v_subrev_u32_e32 v86, s70, v68
	v_cmp_le_u32_e32 vcc, s70, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v86, vcc
	v_subrev_u32_e32 v86, s70, v68
	v_cmp_le_u32_e32 vcc, s70, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v86, vcc
	v_mov_b64_e32 v[92:93], v[68:69]
.LBB0_157:
	s_or_b64 exec, exec, s[4:5]
	s_load_dword s4, s[96:97], 0x58
	v_ashrrev_i32_e32 v68, 31, v87
	v_xor_b32_e32 v68, v90, v68
	v_ashrrev_i32_e32 v86, 31, v93
	v_and_b32_e32 v86, s70, v86
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v68, v68, s4
	v_add3_u32 v68, v68, v92, v86
	buffer_load_ubyte v90, v68, s[8:11], 0 offen
	v_sub_co_u32_e32 v68, vcc, 0xfffffdfd, v76
	v_lshl_add_u64 v[86:87], v[76:77], 0, s[50:51]
	s_nop 0
	v_subb_co_u32_e32 v76, vcc, -1, v77, vcc
	v_cmp_gt_i32_e32 vcc, 0, v87
	s_nop 1
	v_cndmask_b32_e32 v89, v87, v76, vcc
	v_cndmask_b32_e32 v88, v86, v68, vcc
	v_or_b32_e32 v68, s71, v89
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[4:5], exec, s[4:5]
	s_cbranch_execz .LBB0_159
	s_add_u32 s6, s70, s74
	s_mov_b32 s75, s74
	s_addc_u32 s7, s71, s74
	s_xor_b64 s[6:7], s[6:7], s[74:75]
	v_cvt_f32_u32_e32 v68, s6
	v_cvt_f32_u32_e32 v76, s7
	s_sub_u32 s12, 0, s6
	s_subb_u32 s13, 0, s7
	v_fmac_f32_e32 v68, 0x4f800000, v76
	v_rcp_f32_e32 v68, v68
	s_nop 0
	v_mul_f32_e32 v68, 0x5f7ffffc, v68
	v_mul_f32_e32 v76, 0x2f800000, v68
	v_trunc_f32_e32 v76, v76
	v_fmac_f32_e32 v68, 0xcf800000, v76
	v_cvt_u32_f32_e32 v76, v76
	v_cvt_u32_f32_e32 v68, v68
	v_readfirstlane_b32 s18, v76
	v_readfirstlane_b32 s19, v68
	s_mul_i32 s22, s12, s18
	s_mul_hi_u32 s24, s12, s19
	s_mul_i32 s23, s13, s19
	s_add_i32 s22, s24, s22
	s_add_i32 s22, s22, s23
	s_mul_i32 s25, s12, s19
	s_mul_i32 s24, s19, s22
	s_mul_hi_u32 s26, s19, s25
	s_mul_hi_u32 s23, s19, s22
	s_add_u32 s24, s26, s24
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 s27, s18, s25
	s_mul_i32 s25, s18, s25
	s_add_u32 s24, s24, s25
	s_mul_hi_u32 s26, s18, s22
	s_addc_u32 s23, s23, s27
	s_addc_u32 s24, s26, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s22, s23, s22
	s_addc_u32 s23, 0, s24
	s_add_u32 s19, s19, s22
	s_addc_u32 s18, s18, s23
	s_mul_i32 s22, s12, s18
	s_mul_hi_u32 s23, s12, s19
	s_add_i32 s22, s23, s22
	s_mul_i32 s13, s13, s19
	s_add_i32 s22, s22, s13
	s_mul_i32 s12, s12, s19
	s_mul_hi_u32 s23, s18, s12
	s_mul_i32 s24, s18, s12
	s_mul_i32 s26, s19, s22
	s_mul_hi_u32 s12, s19, s12
	s_mul_hi_u32 s25, s19, s22
	s_add_u32 s12, s12, s26
	s_addc_u32 s25, 0, s25
	s_add_u32 s12, s12, s24
	s_mul_hi_u32 s13, s18, s22
	s_addc_u32 s12, s25, s23
	s_addc_u32 s13, s13, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s12, s12, s22
	v_ashrrev_i32_e32 v76, 31, v89
	s_addc_u32 s13, 0, s13
	v_mov_b32_e32 v77, v76
	s_add_u32 s19, s19, s12
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[76:77]
	s_addc_u32 s18, s18, s13
	v_xor_b32_e32 v86, v88, v76
	v_xor_b32_e32 v77, v89, v76
	v_mad_u64_u32 v[88:89], s[12:13], v86, s18, 0
	v_mul_hi_u32 v68, v86, s19
	v_lshl_add_u64 v[88:89], v[68:69], 0, v[88:89]
	v_mad_u64_u32 v[138:139], s[12:13], v77, s19, 0
	v_add_co_u32_e32 v68, vcc, v88, v138
	v_mad_u64_u32 v[92:93], s[12:13], v77, s18, 0
	s_nop 0
	v_addc_co_u32_e32 v88, vcc, v89, v139, vcc
	v_mov_b32_e32 v89, s41
	s_nop 0
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[88:89], v[88:89], 0, v[92:93]
	v_mul_lo_u32 v68, s7, v88
	v_mul_lo_u32 v91, s6, v89
	v_mad_u64_u32 v[92:93], s[12:13], s6, v88, 0
	v_add3_u32 v68, v93, v91, v68
	v_sub_u32_e32 v91, v77, v68
	v_mov_b32_e32 v93, s7
	v_sub_co_u32_e32 v86, vcc, v86, v92
	v_lshl_add_u64 v[138:139], v[88:89], 0, 1
	s_nop 0
	v_subb_co_u32_e64 v91, s[18:19], v91, v93, vcc
	v_subrev_co_u32_e64 v92, s[18:19], s6, v86
	v_subb_co_u32_e32 v68, vcc, v77, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[18:19], 0, v91, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v91
	v_cmp_le_u32_e32 vcc, s7, v68
	v_xor_b32_e32 v76, s74, v76
	v_cndmask_b32_e64 v93, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v92
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v86
	v_cndmask_b32_e64 v92, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v91
	v_cndmask_b32_e64 v86, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v68
	v_cndmask_b32_e64 v91, v93, v92, s[18:19]
	v_lshl_add_u64 v[92:93], v[88:89], 0, 2
	v_cndmask_b32_e32 v68, v77, v86, vcc
	v_cmp_ne_u32_e32 vcc, 0, v91
	s_nop 1
	v_cndmask_b32_e32 v77, v138, v92, vcc
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v88, v77, vcc
	v_xor_b32_e32 v68, v68, v76
	v_sub_co_u32_e32 v76, vcc, v68, v76
.LBB0_159:
	s_andn2_saveexec_b64 s[4:5], s[4:5]
	s_cbranch_execz .LBB0_161
	v_rcp_iflag_f32_e32 v68, v96
	s_sub_i32 s6, 0, s70
	v_mul_f32_e32 v68, 0x4f7ffffe, v68
	v_cvt_u32_f32_e32 v68, v68
	v_mul_lo_u32 v76, s6, v68
	v_mul_hi_u32 v76, v68, v76
	v_add_u32_e32 v68, v68, v76
	v_mul_hi_u32 v68, v88, v68
	v_mul_lo_u32 v76, v68, s70
	v_sub_u32_e32 v76, v88, v76
	v_add_u32_e32 v77, 1, v68
	v_subrev_u32_e32 v86, s70, v76
	v_cmp_le_u32_e32 vcc, s70, v76
	s_nop 1
	v_cndmask_b32_e32 v76, v76, v86, vcc
	v_cndmask_b32_e32 v68, v68, v77, vcc
	v_add_u32_e32 v77, 1, v68
	v_cmp_le_u32_e32 vcc, s70, v76
	s_nop 1
	v_cndmask_b32_e32 v76, v68, v77, vcc
.LBB0_161:
	s_or_b64 exec, exec, s[4:5]
	v_lshl_add_u64 v[78:79], v[78:79], 0, s[50:51]
	v_or_b32_e32 v68, s71, v79
	v_cmp_ne_u32_e32 vcc, 0, v68
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[4:5], exec, s[4:5]
	s_cbranch_execz .LBB0_163
	s_add_u32 s6, s70, s74
	s_mov_b32 s75, s74
	s_addc_u32 s7, s71, s74
	s_xor_b64 s[6:7], s[6:7], s[74:75]
	v_cvt_f32_u32_e32 v68, s6
	v_cvt_f32_u32_e32 v77, s7
	s_sub_u32 s12, 0, s6
	s_subb_u32 s13, 0, s7
	v_ashrrev_i32_e32 v92, 31, v79
	v_fmac_f32_e32 v68, 0x4f800000, v77
	v_rcp_f32_e32 v68, v68
	v_mov_b32_e32 v93, v92
	v_lshl_add_u64 v[78:79], v[78:79], 0, v[92:93]
	v_xor_b32_e32 v86, v78, v92
	v_mul_f32_e32 v68, 0x5f7ffffc, v68
	v_mul_f32_e32 v77, 0x2f800000, v68
	v_trunc_f32_e32 v77, v77
	v_fmac_f32_e32 v68, 0xcf800000, v77
	v_cvt_u32_f32_e32 v77, v77
	v_cvt_u32_f32_e32 v68, v68
	v_readfirstlane_b32 s18, v77
	v_readfirstlane_b32 s19, v68
	s_mul_i32 s22, s12, s18
	s_mul_hi_u32 s24, s12, s19
	s_mul_i32 s23, s13, s19
	s_add_i32 s22, s24, s22
	s_add_i32 s22, s22, s23
	s_mul_i32 s25, s12, s19
	s_mul_i32 s24, s19, s22
	s_mul_hi_u32 s26, s19, s25
	s_mul_hi_u32 s23, s19, s22
	s_add_u32 s24, s26, s24
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 s27, s18, s25
	s_mul_i32 s25, s18, s25
	s_add_u32 s24, s24, s25
	s_mul_hi_u32 s26, s18, s22
	s_addc_u32 s23, s23, s27
	s_addc_u32 s24, s26, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s22, s23, s22
	s_addc_u32 s23, 0, s24
	s_add_u32 s19, s19, s22
	s_addc_u32 s18, s18, s23
	s_mul_i32 s22, s12, s18
	s_mul_hi_u32 s23, s12, s19
	s_add_i32 s22, s23, s22
	s_mul_i32 s13, s13, s19
	s_add_i32 s22, s22, s13
	s_mul_i32 s12, s12, s19
	s_mul_hi_u32 s23, s18, s12
	s_mul_i32 s24, s18, s12
	s_mul_i32 s26, s19, s22
	s_mul_hi_u32 s12, s19, s12
	s_mul_hi_u32 s25, s19, s22
	s_add_u32 s12, s12, s26
	s_addc_u32 s25, 0, s25
	s_add_u32 s12, s12, s24
	s_mul_hi_u32 s13, s18, s22
	s_addc_u32 s12, s25, s23
	s_addc_u32 s13, s13, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s12, s12, s22
	s_addc_u32 s13, 0, s13
	s_add_u32 s19, s19, s12
	s_addc_u32 s18, s18, s13
	v_xor_b32_e32 v77, v79, v92
	v_mad_u64_u32 v[78:79], s[12:13], v86, s18, 0
	v_mul_hi_u32 v68, v86, s19
	v_lshl_add_u64 v[78:79], v[68:69], 0, v[78:79]
	v_mad_u64_u32 v[138:139], s[12:13], v77, s19, 0
	v_add_co_u32_e32 v68, vcc, v78, v138
	v_mad_u64_u32 v[88:89], s[12:13], v77, s18, 0
	s_nop 0
	v_addc_co_u32_e32 v78, vcc, v79, v139, vcc
	v_mov_b32_e32 v79, s41
	s_nop 0
	v_addc_co_u32_e32 v89, vcc, 0, v89, vcc
	v_lshl_add_u64 v[78:79], v[78:79], 0, v[88:89]
	v_mul_lo_u32 v68, s7, v78
	v_mul_lo_u32 v88, s6, v79
	v_mad_u64_u32 v[78:79], s[12:13], s6, v78, 0
	v_add3_u32 v68, v79, v88, v68
	v_sub_u32_e32 v79, v77, v68
	v_mov_b32_e32 v88, s7
	v_sub_co_u32_e32 v78, vcc, v86, v78
	s_nop 1
	v_subb_co_u32_e64 v79, s[18:19], v79, v88, vcc
	v_subrev_co_u32_e64 v86, s[18:19], s6, v78
	v_subb_co_u32_e32 v68, vcc, v77, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[22:23], 0, v79, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v89
	v_subb_co_u32_e64 v79, s[18:19], v79, v88, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v86
	v_subrev_co_u32_e64 v88, s[18:19], s6, v86
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v89
	v_subbrev_co_u32_e64 v79, s[18:19], 0, v79, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v91, v91, v93, s[22:23]
	v_cmp_le_u32_e32 vcc, s7, v68
	v_cmp_ne_u32_e64 s[18:19], 0, v91
	s_nop 0
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v78
	v_cndmask_b32_e64 v79, v89, v79, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v89, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v68
	s_nop 1
	v_cndmask_b32_e32 v77, v77, v89, vcc
	v_cmp_ne_u32_e32 vcc, 0, v77
	v_cndmask_b32_e64 v77, v86, v88, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v77, v78, v77, vcc
	v_cndmask_b32_e32 v68, v68, v79, vcc
	v_xor_b32_e32 v77, v77, v92
	v_xor_b32_e32 v68, v68, v92
	v_sub_co_u32_e32 v88, vcc, v77, v92
	s_nop 1
	v_subb_co_u32_e32 v89, vcc, v68, v92, vcc
.LBB0_163:
	s_andn2_saveexec_b64 s[4:5], s[4:5]
	s_cbranch_execz .LBB0_165
	v_rcp_iflag_f32_e32 v68, v96
	s_sub_i32 s6, 0, s70
	v_mul_f32_e32 v68, 0x4f7ffffe, v68
	v_cvt_u32_f32_e32 v68, v68
	v_mul_lo_u32 v77, s6, v68
	v_mul_hi_u32 v77, v68, v77
	v_add_u32_e32 v68, v68, v77
	v_mul_hi_u32 v68, v78, v68
	v_mul_lo_u32 v68, v68, s70
	v_sub_u32_e32 v68, v78, v68
	v_subrev_u32_e32 v77, s70, v68
	v_cmp_le_u32_e32 vcc, s70, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v77, vcc
	v_subrev_u32_e32 v77, s70, v68
	v_cmp_le_u32_e32 vcc, s70, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v77, vcc
	v_mov_b64_e32 v[88:89], v[68:69]
.LBB0_165:
	s_or_b64 exec, exec, s[4:5]
	s_load_dword s4, s[96:97], 0x58
	v_ashrrev_i32_e32 v68, 31, v87
	v_xor_b32_e32 v68, v76, v68
	v_ashrrev_i32_e32 v76, 31, v89
	v_and_b32_e32 v76, s70, v76
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v68, v68, s4
	v_add3_u32 v68, v68, v88, v76
	buffer_load_ubyte v79, v68, s[8:11], 0 offen
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	v_lshl_add_u64 v[76:77], v[84:85], 0, s[0:1]
	v_add_u32_e32 v68, 0x180, v82
	v_cmp_gt_i64_e32 vcc, s[34:35], v[76:77]
	v_add_u32_e32 v78, v68, v34
	s_and_b64 s[18:19], s[2:3], vcc
	s_mov_b32 m0, s45
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[122:125], v[16:19], v[8:11], v121, v35 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e64 v76, v31, v78, s[18:19]
	s_and_b64 vcc, s[14:15], vcc
	buffer_load_dwordx4 v76, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[130:133], v[16:19], v[4:7], v121, v35 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v16, v68, v38
	v_cndmask_b32_e32 v16, v31, v16, vcc
	s_mov_b32 m0, s55
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[134:137], v[20:23], v[8:11], v121, v105 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v16, s[36:39], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[126:129], v[20:23], v[4:7], v121, v105 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 v[130:133], v50
	ds_read_b128 v[126:129], v52 offset:2048
	ds_read_b128 v[122:125], v50 offset:2048
	ds_read_b128 v[134:137], v52
	ds_read_b32 v78, v97 offset:24576
	v_sub_co_u32_e32 v6, vcc, 0xffff9fff, v80
	v_lshl_add_u64 v[4:5], v[80:81], 0, s[16:17]
	s_nop 0
	v_subb_co_u32_e32 v7, vcc, -1, v81, vcc
	v_cmp_gt_i32_e32 vcc, 0, v5
	s_nop 1
	v_cndmask_b32_e32 v17, v5, v7, vcc
	v_cndmask_b32_e32 v16, v4, v6, vcc
	v_or_b32_e32 v6, s35, v17
	v_cmp_ne_u32_e32 vcc, 0, v6
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[4:5], exec, s[4:5]
	s_cbranch_execz .LBB0_167
	s_add_u32 s6, s34, s72
	s_mov_b32 s73, s72
	s_addc_u32 s7, s35, s72
	s_xor_b64 s[6:7], s[6:7], s[72:73]
	v_cvt_f32_u32_e32 v6, s6
	v_cvt_f32_u32_e32 v7, s7
	s_sub_u32 s12, 0, s6
	s_subb_u32 s13, 0, s7
	v_fmac_f32_e32 v6, 0x4f800000, v7
	v_rcp_f32_e32 v6, v6
	s_nop 0
	v_mul_f32_e32 v6, 0x5f7ffffc, v6
	v_mul_f32_e32 v7, 0x2f800000, v6
	v_trunc_f32_e32 v7, v7
	v_fmac_f32_e32 v6, 0xcf800000, v7
	v_cvt_u32_f32_e32 v7, v7
	v_cvt_u32_f32_e32 v6, v6
	v_readfirstlane_b32 s18, v7
	v_readfirstlane_b32 s19, v6
	s_mul_i32 s22, s12, s18
	s_mul_hi_u32 s24, s12, s19
	s_mul_i32 s23, s13, s19
	s_add_i32 s22, s24, s22
	s_add_i32 s22, s22, s23
	s_mul_i32 s25, s12, s19
	s_mul_i32 s24, s19, s22
	s_mul_hi_u32 s26, s19, s25
	s_mul_hi_u32 s23, s19, s22
	s_add_u32 s24, s26, s24
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 s27, s18, s25
	s_mul_i32 s25, s18, s25
	s_add_u32 s24, s24, s25
	s_mul_hi_u32 s26, s18, s22
	s_addc_u32 s23, s23, s27
	s_addc_u32 s24, s26, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s22, s23, s22
	s_addc_u32 s23, 0, s24
	s_add_u32 s19, s19, s22
	s_addc_u32 s18, s18, s23
	s_mul_i32 s22, s12, s18
	s_mul_hi_u32 s23, s12, s19
	s_add_i32 s22, s23, s22
	s_mul_i32 s13, s13, s19
	s_add_i32 s22, s22, s13
	s_mul_i32 s12, s12, s19
	s_mul_hi_u32 s23, s18, s12
	s_mul_i32 s24, s18, s12
	s_mul_i32 s26, s19, s22
	s_mul_hi_u32 s12, s19, s12
	s_mul_hi_u32 s25, s19, s22
	s_add_u32 s12, s12, s26
	s_addc_u32 s25, 0, s25
	s_add_u32 s12, s12, s24
	s_mul_hi_u32 s13, s18, s22
	s_addc_u32 s12, s25, s23
	s_addc_u32 s13, s13, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s12, s12, s22
	v_ashrrev_i32_e32 v6, 31, v17
	s_addc_u32 s13, 0, s13
	v_mov_b32_e32 v7, v6
	s_add_u32 s19, s19, s12
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[6:7]
	s_addc_u32 s18, s18, s13
	v_xor_b32_e32 v35, v16, v6
	v_xor_b32_e32 v7, v17, v6
	v_mad_u64_u32 v[16:17], s[12:13], v35, s18, 0
	v_mul_hi_u32 v68, v35, s19
	v_lshl_add_u64 v[16:17], v[68:69], 0, v[16:17]
	v_mad_u64_u32 v[76:77], s[12:13], v7, s19, 0
	v_add_co_u32_e32 v16, vcc, v16, v76
	v_mad_u64_u32 v[18:19], s[12:13], v7, s18, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v77, vcc
	v_mov_b32_e32 v17, s41
	s_nop 0
	v_addc_co_u32_e32 v19, vcc, 0, v19, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[18:19]
	v_mul_lo_u32 v68, s7, v16
	v_mul_lo_u32 v76, s6, v17
	v_mad_u64_u32 v[18:19], s[12:13], s6, v16, 0
	v_add3_u32 v68, v19, v76, v68
	v_sub_u32_e32 v19, v7, v68
	v_mov_b32_e32 v76, s7
	v_sub_co_u32_e32 v35, vcc, v35, v18
	v_xor_b32_e32 v6, s72, v6
	s_nop 0
	v_subb_co_u32_e64 v18, s[18:19], v19, v76, vcc
	v_subrev_co_u32_e64 v19, s[18:19], s6, v35
	v_subb_co_u32_e32 v7, vcc, v7, v68, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v18, s[18:19], 0, v18, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s7, v18
	v_cmp_le_u32_e32 vcc, s7, v7
	s_nop 0
	v_cndmask_b32_e64 v76, 0, -1, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s6, v19
	s_nop 1
	v_cndmask_b32_e64 v19, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s7, v18
	s_nop 1
	v_cndmask_b32_e64 v80, v76, v19, s[18:19]
	v_lshl_add_u64 v[18:19], v[16:17], 0, 2
	v_lshl_add_u64 v[76:77], v[16:17], 0, 1
	v_cndmask_b32_e64 v17, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v35
	s_nop 1
	v_cndmask_b32_e64 v19, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v7
	s_nop 1
	v_cndmask_b32_e32 v7, v17, v19, vcc
	v_cmp_ne_u32_e32 vcc, 0, v80
	s_nop 1
	v_cndmask_b32_e32 v17, v76, v18, vcc
	v_cmp_ne_u32_e32 vcc, 0, v7
	s_nop 1
	v_cndmask_b32_e32 v7, v16, v17, vcc
	v_xor_b32_e32 v7, v7, v6
	v_sub_co_u32_e32 v6, vcc, v7, v6
.LBB0_167:
	s_andn2_saveexec_b64 s[4:5], s[4:5]
	s_cbranch_execz .LBB0_169
	v_cvt_f32_u32_e32 v6, s34
	s_sub_i32 s6, 0, s34
	v_rcp_iflag_f32_e32 v6, v6
	s_nop 0
	v_mul_f32_e32 v6, 0x4f7ffffe, v6
	v_cvt_u32_f32_e32 v6, v6
	v_mul_lo_u32 v7, s6, v6
	v_mul_hi_u32 v7, v6, v7
	v_add_u32_e32 v6, v6, v7
	v_mul_hi_u32 v6, v16, v6
	v_mul_lo_u32 v7, v6, s34
	v_sub_u32_e32 v7, v16, v7
	v_add_u32_e32 v17, 1, v6
	v_subrev_u32_e32 v16, s34, v7
	v_cmp_le_u32_e32 vcc, s34, v7
	s_nop 1
	v_cndmask_b32_e32 v7, v7, v16, vcc
	v_cndmask_b32_e32 v6, v6, v17, vcc
	v_add_u32_e32 v16, 1, v6
	v_cmp_le_u32_e32 vcc, s34, v7
	s_nop 1
	v_cndmask_b32_e32 v6, v6, v16, vcc
.LBB0_169:
	s_or_b64 exec, exec, s[4:5]
	v_or_b32_e32 v7, s35, v5
	v_cmp_ne_u32_e32 vcc, 0, v7
	v_ashrrev_i32_e32 v16, 31, v5
	s_and_saveexec_b64 s[4:5], vcc
	s_xor_b64 s[4:5], exec, s[4:5]
	s_cbranch_execz .LBB0_171
	s_add_u32 s6, s34, s72
	s_mov_b32 s73, s72
	s_addc_u32 s7, s35, s72
	s_xor_b64 s[6:7], s[6:7], s[72:73]
	v_cvt_f32_u32_e32 v7, s6
	v_cvt_f32_u32_e32 v17, s7
	s_sub_u32 s12, 0, s6
	s_subb_u32 s13, 0, s7
	v_fmac_f32_e32 v7, 0x4f800000, v17
	v_rcp_f32_e32 v7, v7
	s_nop 0
	v_mul_f32_e32 v7, 0x5f7ffffc, v7
	v_mul_f32_e32 v17, 0x2f800000, v7
	v_trunc_f32_e32 v17, v17
	v_fmac_f32_e32 v7, 0xcf800000, v17
	v_cvt_u32_f32_e32 v17, v17
	v_cvt_u32_f32_e32 v7, v7
	v_readfirstlane_b32 s18, v17
	v_readfirstlane_b32 s19, v7
	s_mul_i32 s22, s12, s18
	s_mul_hi_u32 s24, s12, s19
	s_mul_i32 s23, s13, s19
	s_add_i32 s22, s24, s22
	s_add_i32 s22, s22, s23
	s_mul_i32 s25, s12, s19
	s_mul_i32 s24, s19, s22
	s_mul_hi_u32 s26, s19, s25
	s_mul_hi_u32 s23, s19, s22
	s_add_u32 s24, s26, s24
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 s27, s18, s25
	s_mul_i32 s25, s18, s25
	s_add_u32 s24, s24, s25
	s_mul_hi_u32 s26, s18, s22
	s_addc_u32 s23, s23, s27
	s_addc_u32 s24, s26, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s22, s23, s22
	s_addc_u32 s23, 0, s24
	s_add_u32 s19, s19, s22
	s_addc_u32 s18, s18, s23
	s_mul_i32 s22, s12, s18
	s_mul_hi_u32 s23, s12, s19
	s_add_i32 s22, s23, s22
	s_mul_i32 s13, s13, s19
	s_add_i32 s22, s22, s13
	s_mul_i32 s12, s12, s19
	s_mul_hi_u32 s23, s18, s12
	s_mul_i32 s24, s18, s12
	s_mul_i32 s26, s19, s22
	s_mul_hi_u32 s12, s19, s12
	s_mul_hi_u32 s25, s19, s22
	s_add_u32 s12, s12, s26
	s_addc_u32 s25, 0, s25
	s_add_u32 s12, s12, s24
	s_mul_hi_u32 s13, s18, s22
	s_addc_u32 s12, s25, s23
	s_addc_u32 s13, s13, 0
	s_mul_i32 s22, s18, s22
	s_add_u32 s12, s12, s22
	s_addc_u32 s13, 0, s13
	v_mov_b32_e32 v17, v16
	s_add_u32 s19, s19, s12
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[16:17]
	s_addc_u32 s18, s18, s13
	v_xor_b32_e32 v17, v4, v16
	v_xor_b32_e32 v7, v5, v16
	v_mad_u64_u32 v[4:5], s[12:13], v17, s18, 0
	v_mul_hi_u32 v68, v17, s19
	v_lshl_add_u64 v[4:5], v[68:69], 0, v[4:5]
	v_mad_u64_u32 v[76:77], s[12:13], v7, s19, 0
	v_add_co_u32_e32 v4, vcc, v4, v76
	v_mad_u64_u32 v[18:19], s[12:13], v7, s18, 0
	s_nop 0
	v_addc_co_u32_e32 v4, vcc, v5, v77, vcc
	v_mov_b32_e32 v5, s41
	s_nop 0
	v_addc_co_u32_e32 v19, vcc, 0, v19, vcc
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[18:19]
	v_mul_lo_u32 v18, s7, v4
	v_mul_lo_u32 v19, s6, v5
	v_mad_u64_u32 v[4:5], s[12:13], s6, v4, 0
	v_add3_u32 v5, v5, v19, v18
	v_sub_u32_e32 v18, v7, v5
	v_mov_b32_e32 v19, s7
	v_sub_co_u32_e32 v4, vcc, v17, v4
	s_nop 1
	v_subb_co_u32_e64 v17, s[18:19], v18, v19, vcc
	v_subrev_co_u32_e64 v18, s[18:19], s6, v4
	v_subb_co_u32_e32 v5, vcc, v7, v5, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v35, s[22:23], 0, v17, s[18:19]
	v_cmp_le_u32_e64 s[22:23], s7, v35
	v_subb_co_u32_e64 v17, s[18:19], v17, v19, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v68, 0, -1, s[22:23]
	v_cmp_le_u32_e64 s[22:23], s6, v18
	v_subrev_co_u32_e64 v19, s[18:19], s6, v18
	s_nop 0
	v_cndmask_b32_e64 v76, 0, -1, s[22:23]
	v_cmp_eq_u32_e64 s[22:23], s7, v35
	v_subbrev_co_u32_e64 v17, s[18:19], 0, v17, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v68, v68, v76, s[22:23]
	v_cmp_le_u32_e32 vcc, s7, v5
	v_cmp_ne_u32_e64 s[18:19], 0, v68
	s_nop 0
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s6, v4
	v_cndmask_b32_e64 v17, v35, v17, s[18:19]
	s_nop 0
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s7, v5
	s_nop 1
	v_cndmask_b32_e32 v7, v7, v35, vcc
	v_cmp_ne_u32_e32 vcc, 0, v7
	v_cndmask_b32_e64 v7, v18, v19, s[18:19]
	s_nop 0
	v_cndmask_b32_e32 v4, v4, v7, vcc
	v_cndmask_b32_e32 v5, v5, v17, vcc
	v_xor_b32_e32 v4, v4, v16
	v_xor_b32_e32 v5, v5, v16
	v_sub_co_u32_e32 v18, vcc, v4, v16
	s_nop 1
	v_subb_co_u32_e32 v19, vcc, v5, v16, vcc
.LBB0_171:
	s_andn2_saveexec_b64 s[4:5], s[4:5]
	s_cbranch_execz .LBB0_124
	v_cvt_f32_u32_e32 v5, s34
	s_sub_i32 s6, 0, s34
	v_rcp_iflag_f32_e32 v5, v5
	s_nop 0
	v_mul_f32_e32 v5, 0x4f7ffffe, v5
	v_cvt_u32_f32_e32 v5, v5
	v_mul_lo_u32 v7, s6, v5
	v_mul_hi_u32 v7, v5, v7
	v_add_u32_e32 v5, v5, v7
	v_mul_hi_u32 v5, v4, v5
	v_mul_lo_u32 v5, v5, s34
	v_sub_u32_e32 v4, v4, v5
	v_subrev_u32_e32 v5, s34, v4
	v_cmp_le_u32_e32 vcc, s34, v4
	s_nop 1
	v_cndmask_b32_e32 v4, v4, v5, vcc
	v_subrev_u32_e32 v5, s34, v4
	v_cmp_le_u32_e32 vcc, s34, v4
	s_nop 1
	v_cndmask_b32_e32 v68, v4, v5, vcc
	v_mov_b64_e32 v[18:19], v[68:69]
	s_branch .LBB0_124
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_dynamic_gemm_64x64x256
		.amdhsa_group_segment_fixed_size 26112
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
	.size	wave_mxfp4_dynamic_gemm_64x64x256, .Lfunc_end0-wave_mxfp4_dynamic_gemm_64x64x256

	.set wave_mxfp4_dynamic_gemm_64x64x256.num_vgpr, 145
	.set wave_mxfp4_dynamic_gemm_64x64x256.num_agpr, 0
	.set wave_mxfp4_dynamic_gemm_64x64x256.numbered_sgpr, 98
	.set wave_mxfp4_dynamic_gemm_64x64x256.num_named_barrier, 0
	.set wave_mxfp4_dynamic_gemm_64x64x256.private_seg_size, 0
	.set wave_mxfp4_dynamic_gemm_64x64x256.uses_vcc, 1
	.set wave_mxfp4_dynamic_gemm_64x64x256.uses_flat_scratch, 0
	.set wave_mxfp4_dynamic_gemm_64x64x256.has_dyn_sized_stack, 0
	.set wave_mxfp4_dynamic_gemm_64x64x256.has_recursion, 0
	.set wave_mxfp4_dynamic_gemm_64x64x256.has_indirect_call, 0
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
    .group_segment_fixed_size: 26112
    .kernarg_segment_align: 8
    .kernarg_segment_size: 104
    .max_flat_workgroup_size: 256
    .name:           wave_mxfp4_dynamic_gemm_64x64x256
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 64
      - 4
      - 1
    .sgpr_count:     104
    .sgpr_spill_count: 53
    .symbol:         wave_mxfp4_dynamic_gemm_64x64x256.kd
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
