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
	s_load_dword s33, s[0:1], 0x50
	s_mov_b64 s[40:41], s[2:3]
	v_writelane_b32 v202, s17, 0
	v_writelane_b32 v202, s10, 2
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s72, s53, 31
	s_mov_b32 s30, s52
	v_writelane_b32 v202, s11, 3
	s_mov_b32 s31, s53
	s_mov_b32 s73, s72
	v_writelane_b32 v202, s0, 4
	s_xor_b64 s[74:75], s[30:31], s[72:73]
	s_load_dword s54, s[0:1], 0x58
	v_writelane_b32 v202, s1, 5
	s_lshr_b32 s0, s75, 31
	s_add_u32 s0, s74, s0
	s_addc_u32 s1, s75, 0
	s_ashr_i32 s66, s75, 31
	s_ashr_i64 s[0:1], s[0:1], 1
	s_lshr_b32 s2, s66, 27
	s_add_u32 s2, s74, s2
	s_addc_u32 s3, s75, 0
	s_ashr_i64 s[2:3], s[2:3], 5
	s_sub_u32 s18, 0, s52
	s_subb_u32 s19, 0, s53
	s_add_u32 s20, s52, -1
	s_mov_b64 s[36:37], s[6:7]
	v_cmp_lt_i64_e64 s[6:7], s[30:31], 1
	s_addc_u32 s21, s53, -1
	s_and_b64 s[10:11], s[6:7], exec
	s_cselect_b32 s11, s19, s21
	s_cselect_b32 s10, s18, s20
	s_ashr_i32 s18, s11, 31
	s_lshr_b32 s18, s18, 24
	s_add_u32 s10, s10, s18
	s_addc_u32 s11, s11, 0
	s_ashr_i64 s[10:11], s[10:11], 8
	s_sub_u32 s18, 0, s10
	s_subb_u32 s19, 0, s11
	s_add_u32 s10, s10, 1
	s_addc_u32 s11, s11, 0
	v_and_b32_e32 v40, 0x3ff, v0
	v_bfe_u32 v0, v0, 10, 10
	s_and_b64 s[6:7], s[6:7], exec
	s_cselect_b32 s84, s18, s10
	v_lshlrev_b32_e32 v41, 3, v0
	s_mov_b32 s67, s16
	s_mov_b32 s70, s52
	s_mov_b32 s71, 0
	v_mov_b32_e32 v45, 0
	v_readfirstlane_b32 s48, v0
	s_mov_b64 s[16:17], -1
	s_cselect_b32 s85, s19, s11
	v_readfirstlane_b32 s49, v41
	s_cmp_gt_i32 s84, 5
	v_lshrrev_b32_e32 v99, 3, v40
	v_lshrrev_b32_e32 v97, 4, v40
	s_cbranch_scc1 .LBB0_2
	s_ashr_i32 s6, s85, 31
	s_mov_b32 s7, s6
	s_xor_b64 s[10:11], s[6:7], s[84:85]
	s_mul_hi_u32 s16, s10, 0x2aaaaaaa
	s_mul_i32 s17, s10, 0x2aaaaaaa
	s_mul_i32 s19, s11, 0xaaaaaaab
	s_mul_hi_u32 s10, s10, 0xaaaaaaab
	s_mul_hi_u32 s18, s11, 0xaaaaaaab
	s_add_u32 s10, s19, s10
	s_addc_u32 s18, s18, 0
	s_add_u32 s10, s17, s10
	s_addc_u32 s10, s16, 0
	s_add_u32 s10, s18, s10
	s_addc_u32 s16, 0, 0
	s_mul_i32 s18, s11, 0x2aaaaaaa
	s_mul_hi_u32 s17, s11, 0x2aaaaaaa
	s_add_u32 s10, s18, s10
	s_addc_u32 s16, s17, s16
	s_ashr_i32 s11, s11, 31
	s_mul_i32 s17, s11, 0x2aaaaaaa
	s_mul_hi_u32 s18, s11, 0xaaaaaaab
	s_add_i32 s17, s18, s17
	s_mul_i32 s11, s11, 0xaaaaaaab
	s_add_i32 s17, s17, s11
	s_add_u32 s10, s10, s11
	s_addc_u32 s11, s16, s17
	s_lshr_b32 s16, s11, 31
	s_add_u32 s10, s10, s16
	s_addc_u32 s11, s11, 0
	s_xor_b64 s[6:7], s[10:11], s[6:7]
	s_mul_i32 s7, s7, 6
	s_mul_hi_u32 s10, s6, 6
	s_add_i32 s83, s10, s7
	s_mul_i32 s82, s6, 6
	s_sub_u32 s6, 0, s14
	s_subb_u32 s7, 0, s15
	v_writelane_b32 v202, s6, 6
	s_mov_b32 s43, 0x27000
	s_mov_b32 s42, -2
	v_writelane_b32 v202, s7, 7
	s_add_u32 s6, s14, -1
	s_addc_u32 s7, s15, -1
	v_writelane_b32 v202, s6, 8
	s_mov_b64 s[16:17], 0
	v_or_b32_e32 v44, v99, v41
	v_writelane_b32 v202, s7, 9
	s_sub_u32 s6, 0, s12
	s_subb_u32 s7, 0, s13
	v_writelane_b32 v202, s6, 10
	s_mov_b32 s18, s40
	s_mov_b32 s20, s4
	v_writelane_b32 v202, s7, 11
	s_add_u32 s6, s12, -1
	s_addc_u32 s7, s13, -1
	v_writelane_b32 v202, s6, 12
	s_and_b32 s19, s41, 0xffff
	s_and_b32 s21, s5, 0xffff
	v_writelane_b32 v202, s7, 13
	s_ashr_i32 s6, s15, 31
	s_mov_b32 s7, s6
	s_xor_b64 s[10:11], s[14:15], s[6:7]
	s_ashr_i32 s24, s11, 31
	s_lshr_b32 s22, s24, 27
	s_add_u32 s22, s10, s22
	s_addc_u32 s23, s11, 0
	s_ashr_i64 s[22:23], s[22:23], 5
	s_xor_b64 s[22:23], s[22:23], s[6:7]
	s_lshr_b32 s24, s24, 22
	s_add_u32 s10, s10, s24
	s_addc_u32 s11, s11, 0
	s_ashr_i64 s[10:11], s[10:11], 10
	s_xor_b64 s[68:69], s[10:11], s[6:7]
	s_lshl_b64 s[10:11], s[68:69], 5
	s_sub_u32 s6, s22, s10
	v_writelane_b32 v202, s10, 14
	s_subb_u32 s7, s23, s11
	s_mov_b32 s22, s36
	v_writelane_b32 v202, s11, 15
	v_cmp_gt_i64_e64 s[10:11], s[6:7], 1
	s_and_b64 s[10:11], s[10:11], exec
	s_cselect_b32 s7, s7, 0
	s_cselect_b32 s6, s6, 1
	v_writelane_b32 v202, s6, 16
	s_and_b32 s23, s37, 0xffff
	s_and_b32 s25, s9, 0xffff
	v_writelane_b32 v202, s7, 17
	s_and_b32 s6, s14, 0x3ff
	s_mov_b32 s24, s8
	v_mov_b32_e32 v32, s6
	s_mov_b64 s[6:7], s[42:43]
	s_mov_b64 s[38:39], s[42:43]
	s_mov_b64 s[10:11], s[42:43]
	s_branch .LBB0_3
.LBB0_2:
.LBB0_3:
	s_xor_b64 s[44:45], s[0:1], s[72:73]
	s_or_b32 s0, s49, 32
	v_writelane_b32 v202, s0, 18
	s_or_b32 s0, s49, 64
	v_writelane_b32 v202, s0, 20
	s_or_b32 s0, s49, 0x60
	v_writelane_b32 v202, s0, 21
	s_lshl_b32 s0, s48, 8
	v_writelane_b32 v202, s0, 22
	v_writelane_b32 v202, s48, 23
	v_writelane_b32 v202, s67, 24
	s_andn2_b64 vcc, exec, s[16:17]
	v_cmp_lt_i64_e64 s[16:17], s[14:15], 1
	v_writelane_b32 v202, s49, 25
	s_xor_b64 s[46:47], s[2:3], s[72:73]
	v_cmp_lt_i64_e64 s[26:27], s[12:13], 1
	v_lshlrev_b32_e32 v42, 7, v40
	v_writelane_b32 v202, s16, 26
	s_nop 1
	v_writelane_b32 v202, s17, 27
	s_cbranch_vccnz .LBB0_10
	s_sub_u32 s2, 0, s14
	s_subb_u32 s3, 0, s15
	s_add_u32 s6, s14, -1
	s_addc_u32 s7, s15, -1
	s_and_b64 s[0:1], s[16:17], exec
	s_cselect_b32 s0, s2, s6
	v_writelane_b32 v202, s2, 6
	v_or_b32_e32 v44, v99, v41
	v_mov_b32_e32 v45, 0
	v_writelane_b32 v202, s3, 7
	v_writelane_b32 v202, s6, 8
	s_cselect_b32 s1, s3, s7
	s_ashr_i32 s2, s1, 31
	s_lshr_b32 s2, s2, 27
	s_add_u32 s0, s0, s2
	s_addc_u32 s1, s1, 0
	s_ashr_i64 s[0:1], s[0:1], 5
	s_sub_u32 s2, 0, s0
	s_subb_u32 s3, 0, s1
	v_writelane_b32 v202, s7, 9
	s_add_u32 s6, s0, 1
	s_addc_u32 s7, s1, 0
	s_and_b64 s[0:1], s[16:17], exec
	s_cselect_b32 s0, s2, s6
	s_cselect_b32 s1, s3, s7
	s_and_b32 s2, s0, 31
	s_max_u32 s16, s2, 1
	s_cmp_lg_u32 s2, 0
	s_cselect_b64 s[2:3], -1, 0
	s_sub_u32 s10, 0, s12
	s_subb_u32 s11, 0, s13
	s_add_u32 s18, s12, -1
	s_addc_u32 s19, s13, -1
	s_and_b64 s[6:7], s[26:27], exec
	s_cselect_b32 s6, s10, s18
	v_writelane_b32 v202, s10, 10
	v_mov_b32_e32 v53, v45
	s_mov_b32 s43, 0x27000
	v_writelane_b32 v202, s11, 11
	v_writelane_b32 v202, s18, 12
	s_cselect_b32 s7, s11, s19
	s_ashr_i32 s10, s7, 31
	s_lshr_b32 s10, s10, 25
	s_add_u32 s6, s6, s10
	s_addc_u32 s7, s7, 0
	s_ashr_i64 s[6:7], s[6:7], 7
	s_sub_u32 s10, 0, s6
	s_subb_u32 s11, 0, s7
	v_writelane_b32 v202, s19, 13
	s_add_u32 s17, s6, 1
	s_addc_u32 s18, s7, 0
	v_writelane_b32 v202, s26, 28
	s_and_b64 s[6:7], s[26:27], exec
	s_cselect_b32 s29, s11, s18
	v_writelane_b32 v202, s27, 29
	s_cselect_b32 s28, s10, s17
	v_readlane_b32 s11, v202, 0
	s_mul_i32 s6, s29, s11
	s_mul_hi_u32 s7, s28, s11
	s_add_i32 s10, s7, s6
	s_mul_i32 s11, s28, s11
	s_add_u32 s26, s11, s67
	s_addc_u32 s27, s10, 0
	s_ashr_i32 s6, s1, 31
	s_mov_b32 s7, s6
	s_xor_b64 s[0:1], s[6:7], s[0:1]
	s_lshr_b32 s17, s1, 27
	s_add_u32 s0, s0, s17
	s_addc_u32 s1, s1, 0
	s_lshr_b64 s[0:1], s[0:1], 5
	s_xor_b64 s[0:1], s[0:1], s[6:7]
	s_mul_i32 s1, s28, s1
	s_mul_hi_u32 s6, s28, s0
	s_add_i32 s1, s6, s1
	s_mul_i32 s6, s29, s0
	s_add_i32 s1, s1, s6
	s_mul_i32 s0, s28, s0
	s_lshl_b64 s[6:7], s[0:1], 5
	v_mov_b64_e32 v[0:1], s[6:7]
	v_cmp_ge_i64_e32 vcc, s[26:27], v[0:1]
	v_cvt_f32_u32_e32 v0, s16
	v_mov_b32_e32 v1, 0x4f800000
	s_and_b64 s[0:1], s[2:3], vcc
	s_sub_u32 s2, s67, s6
	v_fmac_f32_e32 v0, 0, v1
	v_rcp_f32_e32 v0, v0
	s_subb_u32 s3, 0, s7
	s_add_u32 s6, s2, s11
	s_addc_u32 s7, s3, s10
	v_mul_f32_e32 v0, 0x5f7ffffc, v0
	v_mul_f32_e32 v1, 0x2f800000, v0
	v_trunc_f32_e32 v1, v1
	v_fmac_f32_e32 v0, 0xcf800000, v1
	v_cvt_u32_f32_e32 v2, v1
	v_cvt_u32_f32_e32 v0, v0
	s_ashr_i32 s2, s7, 31
	s_mov_b32 s3, s2
	s_xor_b64 s[6:7], s[2:3], s[6:7]
	s_sub_u32 s10, 0, s16
	v_readfirstlane_b32 s17, v2
	v_readfirstlane_b32 s19, v0
	s_subb_u32 s11, 0, 0
	s_mul_i32 s18, s10, s17
	s_mul_hi_u32 s20, s10, s19
	s_add_i32 s18, s20, s18
	s_mul_i32 s20, s11, s19
	s_add_i32 s18, s18, s20
	s_mul_i32 s22, s10, s19
	s_mul_i32 s21, s19, s18
	s_mul_hi_u32 s23, s19, s22
	s_mul_hi_u32 s20, s19, s18
	s_add_u32 s21, s23, s21
	s_addc_u32 s20, 0, s20
	s_mul_hi_u32 s24, s17, s22
	s_mul_i32 s22, s17, s22
	s_add_u32 s21, s21, s22
	s_mul_hi_u32 s23, s17, s18
	s_addc_u32 s20, s20, s24
	s_addc_u32 s21, s23, 0
	s_mul_i32 s18, s17, s18
	s_add_u32 s18, s20, s18
	s_addc_u32 s20, 0, s21
	s_add_u32 s18, s19, s18
	s_addc_u32 s17, s17, s20
	s_mul_i32 s19, s10, s17
	s_mul_hi_u32 s20, s10, s18
	s_add_i32 s19, s20, s19
	s_mul_i32 s11, s11, s18
	s_add_i32 s19, s19, s11
	s_mul_i32 s10, s10, s18
	s_mul_hi_u32 s20, s17, s10
	s_mul_i32 s21, s17, s10
	s_mul_i32 s23, s18, s19
	s_mul_hi_u32 s10, s18, s10
	s_mul_hi_u32 s22, s18, s19
	s_add_u32 s10, s10, s23
	s_addc_u32 s22, 0, s22
	s_add_u32 s10, s10, s21
	s_mul_hi_u32 s11, s17, s19
	s_addc_u32 s10, s22, s20
	s_addc_u32 s11, s11, 0
	s_mul_i32 s19, s17, s19
	s_add_u32 s10, s10, s19
	s_addc_u32 s11, 0, s11
	s_add_u32 s18, s18, s10
	s_addc_u32 s17, s17, s11
	s_ashr_i32 s10, s7, 31
	s_add_u32 s6, s6, s10
	s_mov_b32 s11, s10
	s_addc_u32 s7, s7, s10
	s_xor_b64 s[6:7], s[6:7], s[10:11]
	s_mul_i32 s20, s6, s17
	s_mul_hi_u32 s21, s6, s18
	s_mul_hi_u32 s19, s6, s17
	s_add_u32 s20, s21, s20
	s_addc_u32 s19, 0, s19
	s_mul_hi_u32 s22, s7, s18
	s_mul_i32 s18, s7, s18
	s_add_u32 s18, s20, s18
	s_mul_hi_u32 s21, s7, s17
	s_addc_u32 s18, s19, s22
	s_addc_u32 s19, s21, 0
	s_mul_i32 s17, s7, s17
	s_add_u32 s17, s18, s17
	s_addc_u32 s18, 0, s19
	s_add_u32 s19, s17, 1
	s_addc_u32 s20, s18, 0
	s_add_u32 s21, s17, 2
	s_mul_i32 s23, s16, s18
	s_mul_hi_u32 s24, s16, s17
	s_addc_u32 s22, s18, 0
	s_add_i32 s24, s24, s23
	s_mul_i32 s23, s16, s17
	s_sub_u32 s6, s6, s23
	s_subb_u32 s7, s7, s24
	s_sub_u32 s23, s6, s16
	s_subb_u32 s24, s7, 0
	s_cmp_ge_u32 s23, s16
	s_cselect_b32 s23, -1, 0
	s_cmp_eq_u32 s24, 0
	s_cselect_b32 s23, s23, -1
	s_cmp_lg_u32 s23, 0
	s_cselect_b32 s19, s21, s19
	s_cselect_b32 s20, s22, s20
	s_cmp_ge_u32 s6, s16
	s_cselect_b32 s6, -1, 0
	s_cmp_eq_u32 s7, 0
	s_cselect_b32 s6, s6, -1
	s_cmp_lg_u32 s6, 0
	s_cselect_b32 s7, s20, s18
	s_cselect_b32 s6, s19, s17
	s_xor_b64 s[6:7], s[6:7], s[10:11]
	s_sub_u32 s6, s6, s10
	s_subb_u32 s7, s7, s10
	s_xor_b64 s[2:3], s[6:7], s[2:3]
	s_ashr_i32 s6, s27, 31
	s_mov_b32 s7, s6
	s_xor_b64 s[10:11], s[6:7], s[26:27]
	s_ashr_i32 s16, s11, 31
	s_lshr_b32 s16, s16, 27
	s_add_u32 s10, s10, s16
	s_addc_u32 s11, s11, 0
	s_ashr_i64 s[10:11], s[10:11], 5
	s_xor_b64 s[38:39], s[10:11], s[6:7]
	s_ashr_i32 s6, s29, 31
	s_add_u32 s10, s28, s6
	s_mov_b32 s7, s6
	s_addc_u32 s11, s29, s6
	s_xor_b64 s[6:7], s[10:11], s[6:7]
	v_cvt_f32_u32_e32 v0, s6
	v_cvt_f32_u32_e32 v1, s7
	s_sub_u32 s10, 0, s6
	s_subb_u32 s11, 0, s7
	v_bitop3_b32 v2, v40, v99, 7 bitop3:0x6c
	v_fmac_f32_e32 v0, 0x4f800000, v1
	v_rcp_f32_e32 v0, v0
	v_lshlrev_b32_e32 v52, 5, v2
	v_lshlrev_b32_e32 v48, 4, v2
	v_cmp_gt_i64_e32 vcc, s[30:31], v[52:53]
	v_mul_f32_e32 v0, 0x5f7ffffc, v0
	v_mul_f32_e32 v1, 0x2f800000, v0
	v_trunc_f32_e32 v1, v1
	v_fmac_f32_e32 v0, 0xcf800000, v1
	v_cvt_u32_f32_e32 v0, v0
	v_cvt_u32_f32_e32 v1, v1
	s_mov_b32 s42, -2
	v_mov_b32_e32 v43, v45
	v_readfirstlane_b32 s16, v0
	v_readfirstlane_b32 s19, v1
	s_mul_hi_u32 s18, s10, s16
	s_mul_i32 s20, s10, s19
	s_mul_i32 s17, s11, s16
	s_add_i32 s18, s18, s20
	s_add_i32 s18, s18, s17
	s_mul_i32 s21, s10, s16
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
	s_mul_i32 s18, s10, s17
	s_mul_hi_u32 s19, s10, s16
	s_add_i32 s18, s19, s18
	s_mul_i32 s11, s11, s16
	s_add_i32 s18, s18, s11
	s_mul_i32 s10, s10, s16
	s_mul_hi_u32 s19, s17, s10
	s_mul_i32 s20, s17, s10
	s_mul_i32 s22, s16, s18
	s_mul_hi_u32 s10, s16, s10
	s_mul_hi_u32 s21, s16, s18
	s_add_u32 s10, s10, s22
	s_addc_u32 s21, 0, s21
	s_add_u32 s10, s10, s20
	s_mul_hi_u32 s11, s17, s18
	s_addc_u32 s10, s21, s19
	s_addc_u32 s11, s11, 0
	s_mul_i32 s18, s17, s18
	s_add_u32 s10, s10, s18
	s_addc_u32 s11, 0, s11
	s_add_u32 s16, s16, s10
	s_addc_u32 s17, s17, s11
	s_ashr_i32 s34, s39, 31
	s_add_u32 s10, s38, s34
	s_mov_b32 s35, s34
	s_addc_u32 s11, s39, s34
	s_xor_b64 s[10:11], s[10:11], s[34:35]
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
	s_add_u32 s16, s16, s17
	s_addc_u32 s17, 0, s18
	s_mul_i32 s17, s6, s17
	s_mul_hi_u32 s18, s6, s16
	s_add_i32 s17, s18, s17
	s_mul_i32 s18, s7, s16
	s_add_i32 s20, s17, s18
	s_sub_i32 s18, s11, s20
	s_mul_i32 s16, s6, s16
	s_sub_u32 s10, s10, s16
	s_cselect_b64 s[16:17], -1, 0
	s_subb_u32 s21, s18, s7
	s_sub_u32 s22, s10, s6
	s_cselect_b64 s[18:19], -1, 0
	s_subb_u32 s23, s21, 0
	s_cmp_ge_u32 s23, s7
	s_cselect_b32 s24, -1, 0
	s_cmp_ge_u32 s22, s6
	s_cselect_b32 s25, -1, 0
	s_cmp_eq_u32 s23, s7
	s_cselect_b32 s24, s25, s24
	s_cmp_lg_u64 s[18:19], 0
	s_subb_u32 s18, s21, s7
	s_sub_u32 s19, s22, s6
	s_subb_u32 s18, s18, 0
	s_cmp_lg_u32 s24, 0
	s_cselect_b32 s19, s19, s22
	s_cselect_b32 s18, s18, s23
	s_cmp_lg_u64 s[16:17], 0
	s_subb_u32 s11, s11, s20
	s_cmp_ge_u32 s11, s7
	s_cselect_b32 s16, -1, 0
	s_cmp_ge_u32 s10, s6
	s_cselect_b32 s6, -1, 0
	s_cmp_eq_u32 s11, s7
	s_cselect_b32 s6, s6, s16
	s_cmp_lg_u32 s6, 0
	s_cselect_b32 s7, s18, s11
	s_cselect_b32 s6, s19, s10
	s_xor_b64 s[6:7], s[6:7], s[34:35]
	s_sub_u32 s10, s6, s34
	s_subb_u32 s11, s7, s34
	s_ashr_i32 s6, s11, 31
	s_mov_b32 s7, s6
	s_and_b64 s[6:7], s[6:7], s[28:29]
	s_add_u32 s6, s6, s10
	s_addc_u32 s7, s7, s11
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s1, s3, s7
	s_cselect_b32 s0, s2, s6
	s_lshl_b64 s[10:11], s[0:1], 7
	v_or_b32_e32 v0, s10, v44
	v_mov_b32_e32 v1, s11
	v_mad_u64_u32 v[50:51], s[2:3], v0, s44, 0
	v_cmp_gt_i64_e64 s[2:3], s[12:13], v[0:1]
	v_add_u32_e32 v1, v48, v50
	s_and_b64 s[6:7], vcc, s[2:3]
	s_lshl_b32 s52, s49, 7
	v_or_b32_e32 v2, 32, v0
	v_mov_b32_e32 v3, s11
	s_and_b32 s41, s41, 0xffff
	v_cndmask_b32_e64 v1, -1, v1, s[6:7]
	s_mov_b32 m0, s52
	v_mad_u64_u32 v[54:55], s[6:7], v2, s44, 0
	v_cmp_gt_i64_e64 s[24:25], s[12:13], v[2:3]
	buffer_load_dwordx4 v1, s[40:43], 0 offen lds
	v_add_u32_e32 v1, v48, v54
	s_and_b64 s[6:7], vcc, s[24:25]
	s_add_i32 s64, s52, 0x1000
	v_or_b32_e32 v2, 64, v0
	v_cndmask_b32_e64 v1, -1, v1, s[6:7]
	s_mov_b32 m0, s64
	v_mad_u64_u32 v[56:57], s[6:7], v2, s44, 0
	v_cmp_gt_i64_e64 s[18:19], s[12:13], v[2:3]
	buffer_load_dwordx4 v1, s[40:43], 0 offen lds
	v_add_u32_e32 v1, v48, v56
	s_and_b64 s[6:7], vcc, s[18:19]
	s_add_i32 s65, s52, 0x2000
	v_cndmask_b32_e64 v1, -1, v1, s[6:7]
	s_mov_b32 m0, s65
	v_or_b32_e32 v0, 0x60, v0
	buffer_load_dwordx4 v1, s[40:43], 0 offen lds
	v_mov_b32_e32 v1, s11
	v_mad_u64_u32 v[58:59], s[6:7], v0, s44, 0
	v_cmp_gt_i64_e64 s[20:21], s[12:13], v[0:1]
	v_add_u32_e32 v0, v48, v58
	s_and_b64 vcc, vcc, s[20:21]
	s_add_i32 s96, s52, 0x3000
	v_cndmask_b32_e32 v0, -1, v0, vcc
	s_mov_b32 m0, s96
	s_lshl_b64 s[0:1], s[0:1], 2
	buffer_load_dwordx4 v0, s[40:43], 0 offen lds
	s_lshl_b32 s7, s30, 5
	s_or_b32 s6, s0, s48
	s_mul_i32 s10, s7, s1
	s_lshr_b64 s[0:1], s[30:31], 27
	s_mul_i32 s0, s0, s6
	v_mov_b32_e32 v0, s6
	s_add_i32 s10, s10, s0
	v_mad_u64_u32 v[60:61], s[0:1], s7, v0, v[42:43]
	v_add_u32_e32 v61, s10, v61
	v_or_b32_e32 v0, s31, v61
	v_cmp_ne_u32_e32 vcc, 0, v0
	v_ashrrev_i32_e32 v0, 31, v61
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	s_cbranch_execz .LBB0_6
	s_ashr_i32 s6, s31, 31
	s_add_u32 s10, s30, s6
	s_mov_b32 s7, s6
	s_addc_u32 s11, s31, s6
	s_xor_b64 s[16:17], s[10:11], s[6:7]
	v_cvt_f32_u32_e32 v1, s16
	v_cvt_f32_u32_e32 v2, s17
	s_sub_u32 s6, 0, s16
	s_subb_u32 s7, 0, s17
	v_mov_b32_e32 v5, v45
	v_fmac_f32_e32 v1, 0x4f800000, v2
	v_rcp_f32_e32 v1, v1
	s_mov_b32 s50, 0
	v_mul_f32_e32 v1, 0x5f7ffffc, v1
	v_mul_f32_e32 v2, 0x2f800000, v1
	v_trunc_f32_e32 v2, v2
	v_fmac_f32_e32 v1, 0xcf800000, v2
	v_cvt_u32_f32_e32 v2, v2
	v_cvt_u32_f32_e32 v1, v1
	v_readfirstlane_b32 s10, v2
	v_readfirstlane_b32 s11, v1
	s_mul_i32 s22, s6, s10
	s_mul_hi_u32 s35, s6, s11
	s_mul_i32 s23, s7, s11
	s_add_i32 s22, s35, s22
	s_add_i32 s22, s22, s23
	s_mul_i32 s48, s6, s11
	s_mul_i32 s35, s11, s22
	s_mul_hi_u32 s49, s11, s48
	s_mul_hi_u32 s23, s11, s22
	s_add_u32 s35, s49, s35
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 s51, s10, s48
	s_mul_i32 s48, s10, s48
	s_add_u32 s35, s35, s48
	s_mul_hi_u32 s49, s10, s22
	s_addc_u32 s23, s23, s51
	s_addc_u32 s35, s49, 0
	s_mul_i32 s22, s10, s22
	s_add_u32 s22, s23, s22
	s_addc_u32 s23, 0, s35
	s_add_u32 s11, s11, s22
	s_addc_u32 s10, s10, s23
	s_mul_i32 s22, s6, s10
	s_mul_hi_u32 s23, s6, s11
	s_add_i32 s22, s23, s22
	s_mul_i32 s7, s7, s11
	s_add_i32 s22, s22, s7
	s_mul_i32 s6, s6, s11
	s_mul_hi_u32 s23, s10, s6
	s_mul_i32 s35, s10, s6
	s_mul_i32 s49, s11, s22
	s_mul_hi_u32 s6, s11, s6
	s_mul_hi_u32 s48, s11, s22
	s_add_u32 s6, s6, s49
	s_addc_u32 s48, 0, s48
	s_add_u32 s6, s6, s35
	s_mul_hi_u32 s7, s10, s22
	s_addc_u32 s6, s48, s23
	s_addc_u32 s7, s7, 0
	s_mul_i32 s22, s10, s22
	s_add_u32 s6, s6, s22
	s_addc_u32 s7, 0, s7
	v_mov_b32_e32 v1, v0
	s_add_u32 s11, s11, s6
	v_lshl_add_u64 v[2:3], v[60:61], 0, v[0:1]
	s_addc_u32 s10, s10, s7
	v_xor_b32_e32 v8, v2, v0
	v_xor_b32_e32 v1, v3, v0
	v_mad_u64_u32 v[2:3], s[6:7], v8, s10, 0
	v_mul_hi_u32 v4, v8, s11
	v_lshl_add_u64 v[2:3], v[4:5], 0, v[2:3]
	v_mad_u64_u32 v[6:7], s[6:7], v1, s11, 0
	v_add_co_u32_e32 v2, vcc, v2, v6
	v_mad_u64_u32 v[4:5], s[6:7], v1, s10, 0
	s_nop 0
	v_addc_co_u32_e32 v2, vcc, v3, v7, vcc
	v_mov_b32_e32 v3, s50
	s_nop 0
	v_addc_co_u32_e32 v5, vcc, 0, v5, vcc
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[4:5]
	v_mul_lo_u32 v4, s17, v2
	v_mul_lo_u32 v5, s16, v3
	v_mad_u64_u32 v[2:3], s[6:7], s16, v2, 0
	v_add3_u32 v3, v3, v5, v4
	v_sub_u32_e32 v4, v1, v3
	v_mov_b32_e32 v5, s17
	v_sub_co_u32_e32 v2, vcc, v8, v2
	s_nop 1
	v_subb_co_u32_e64 v4, s[6:7], v4, v5, vcc
	v_subrev_co_u32_e64 v6, s[6:7], s16, v2
	v_subb_co_u32_e32 v1, vcc, v1, v3, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v7, s[10:11], 0, v4, s[6:7]
	v_cmp_le_u32_e64 s[10:11], s17, v7
	v_subb_co_u32_e64 v4, s[6:7], v4, v5, s[6:7]
	s_nop 0
	v_cndmask_b32_e64 v8, 0, -1, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s16, v6
	v_subrev_co_u32_e64 v5, s[6:7], s16, v6
	s_nop 0
	v_cndmask_b32_e64 v9, 0, -1, s[10:11]
	v_cmp_eq_u32_e64 s[10:11], s17, v7
	v_subbrev_co_u32_e64 v4, s[6:7], 0, v4, s[6:7]
	s_nop 0
	v_cndmask_b32_e64 v8, v8, v9, s[10:11]
	v_cmp_le_u32_e32 vcc, s17, v1
	v_cmp_ne_u32_e64 s[6:7], 0, v8
	s_nop 0
	v_cndmask_b32_e64 v3, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v2
	v_cndmask_b32_e64 v4, v7, v4, s[6:7]
	s_nop 0
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v1
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
	s_andn2_saveexec_b64 s[0:1], s[0:1]
	s_cbranch_execz .LBB0_8
	v_cvt_f32_u32_e32 v1, s30
	s_sub_i32 s6, 0, s30
	v_mov_b32_e32 v3, 0
	v_rcp_iflag_f32_e32 v1, v1
	s_nop 0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_mul_lo_u32 v2, s6, v1
	v_mul_hi_u32 v2, v1, v2
	v_add_u32_e32 v1, v1, v2
	v_mul_hi_u32 v1, v60, v1
	v_mul_lo_u32 v1, v1, s30
	v_sub_u32_e32 v1, v60, v1
	v_subrev_u32_e32 v2, s30, v1
	v_cmp_le_u32_e32 vcc, s30, v1
	s_nop 1
	v_cndmask_b32_e32 v1, v1, v2, vcc
	v_subrev_u32_e32 v2, s30, v1
	v_cmp_le_u32_e32 vcc, s30, v1
	s_nop 1
	v_cndmask_b32_e32 v2, v1, v2, vcc
.LBB0_8:
	s_or_b64 exec, exec, s[0:1]
	s_ashr_i32 s80, s31, 31
	s_add_u32 s0, s30, s80
	s_mov_b32 s81, s80
	s_addc_u32 s1, s31, s80
	s_xor_b64 s[0:1], s[0:1], s[80:81]
	v_cvt_f32_u32_e32 v1, s0
	v_cvt_f32_u32_e32 v4, s1
	s_sub_u32 s6, 0, s0
	s_subb_u32 s7, 0, s1
	v_xor_b32_e32 v5, v0, v61
	v_fmac_f32_e32 v1, 0x4f800000, v4
	v_rcp_f32_e32 v1, v1
	v_xor_b32_e32 v4, v0, v60
	v_mov_b32_e32 v9, 0
	s_mov_b32 s48, 0
	v_mul_f32_e32 v1, 0x5f7ffffc, v1
	v_mul_f32_e32 v6, 0x2f800000, v1
	v_trunc_f32_e32 v6, v6
	v_fmac_f32_e32 v1, 0xcf800000, v6
	v_cvt_u32_f32_e32 v6, v6
	v_cvt_u32_f32_e32 v1, v1
	v_readfirstlane_b32 s10, v6
	v_readfirstlane_b32 s11, v1
	s_mul_hi_u32 s17, s6, s11
	s_mul_i32 s22, s6, s10
	s_mul_i32 s16, s7, s11
	s_add_i32 s17, s17, s22
	s_add_i32 s17, s17, s16
	s_mul_i32 s23, s6, s11
	s_mul_i32 s22, s11, s17
	s_mul_hi_u32 s35, s11, s23
	s_mul_hi_u32 s16, s11, s17
	s_add_u32 s22, s35, s22
	s_addc_u32 s16, 0, s16
	s_mul_hi_u32 s49, s10, s23
	s_mul_i32 s23, s10, s23
	s_add_u32 s22, s22, s23
	s_mul_hi_u32 s35, s10, s17
	s_addc_u32 s16, s16, s49
	s_addc_u32 s22, s35, 0
	s_mul_i32 s17, s10, s17
	s_add_u32 s16, s16, s17
	s_addc_u32 s17, 0, s22
	s_add_u32 s11, s11, s16
	s_addc_u32 s10, s10, s17
	s_mul_i32 s16, s6, s10
	s_mul_hi_u32 s17, s6, s11
	s_add_i32 s16, s17, s16
	s_mul_i32 s7, s7, s11
	s_add_i32 s16, s16, s7
	s_mul_i32 s6, s6, s11
	s_mul_hi_u32 s17, s10, s6
	s_mul_i32 s22, s10, s6
	s_mul_i32 s35, s11, s16
	s_mul_hi_u32 s6, s11, s6
	s_mul_hi_u32 s23, s11, s16
	s_add_u32 s6, s6, s35
	s_addc_u32 s23, 0, s23
	s_add_u32 s6, s6, s22
	s_mul_hi_u32 s7, s10, s16
	s_addc_u32 s6, s23, s17
	s_addc_u32 s7, s7, 0
	s_mul_i32 s16, s10, s16
	s_add_u32 s6, s6, s16
	v_ashrrev_i32_e32 v6, 31, v5
	s_addc_u32 s7, 0, s7
	v_mov_b32_e32 v7, v6
	s_add_u32 s11, s11, s6
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[6:7]
	s_addc_u32 s10, s10, s7
	v_xor_b32_e32 v7, v4, v6
	v_xor_b32_e32 v1, v5, v6
	v_mad_u64_u32 v[4:5], s[6:7], v7, s10, 0
	v_mul_hi_u32 v8, v7, s11
	v_lshl_add_u64 v[4:5], v[8:9], 0, v[4:5]
	v_mad_u64_u32 v[10:11], s[6:7], v1, s11, 0
	v_add_co_u32_e32 v4, vcc, v4, v10
	v_mad_u64_u32 v[8:9], s[6:7], v1, s10, 0
	s_nop 0
	v_addc_co_u32_e32 v4, vcc, v5, v11, vcc
	v_mov_b32_e32 v5, s48
	s_nop 0
	v_addc_co_u32_e32 v9, vcc, 0, v9, vcc
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[8:9]
	v_mul_lo_u32 v10, s1, v4
	v_mul_lo_u32 v5, s0, v5
	v_mad_u64_u32 v[8:9], s[6:7], s0, v4, 0
	v_add3_u32 v5, v9, v5, v10
	v_sub_u32_e32 v9, v1, v5
	v_mov_b32_e32 v10, s1
	v_sub_co_u32_e32 v7, vcc, v7, v8
	s_and_b32 s5, s5, 0xffff
	s_nop 0
	v_subb_co_u32_e64 v8, s[6:7], v9, v10, vcc
	v_subrev_co_u32_e64 v9, s[6:7], s0, v7
	v_subb_co_u32_e32 v1, vcc, v1, v5, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v8, s[6:7], 0, v8, s[6:7]
	v_cmp_le_u32_e64 s[6:7], s1, v8
	v_cmp_le_u32_e32 vcc, s1, v1
	s_nop 0
	v_cndmask_b32_e64 v10, 0, -1, s[6:7]
	v_cmp_le_u32_e64 s[6:7], s0, v9
	v_cndmask_b32_e64 v5, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s0, v7
	v_cndmask_b32_e64 v9, 0, -1, s[6:7]
	v_cmp_eq_u32_e64 s[6:7], s1, v8
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s1, v1
	v_cndmask_b32_e64 v8, v10, v9, s[6:7]
	v_add_u32_e32 v9, 2, v4
	v_add_u32_e32 v10, 1, v4
	v_cmp_ne_u32_e64 s[6:7], 0, v8
	v_cndmask_b32_e32 v1, v5, v7, vcc
	v_cmp_ne_u32_e32 vcc, 0, v1
	v_cndmask_b32_e64 v8, v10, v9, s[6:7]
	v_readlane_b32 s0, v202, 22
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
	v_mov_b32_e32 v3, s48
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[2:3]
	v_alignbit_b32 v0, v1, v0, 5
	v_mul_lo_u32 v1, v4, s46
	s_add_i32 s97, s0, 0xc000
	v_xad_u32 v0, v0, v5, v1
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, -2
	s_mov_b32 m0, s97
	s_ashr_i32 s0, s15, 31
	buffer_load_dword v0, s[4:7], 0 offen lds
	s_mov_b32 s1, s0
	s_xor_b64 s[10:11], s[14:15], s[0:1]
	s_ashr_i32 s15, s11, 31
	s_lshr_b32 s16, s15, 27
	s_add_u32 s16, s10, s16
	s_addc_u32 s17, s11, 0
	s_ashr_i64 s[16:17], s[16:17], 5
	s_xor_b64 s[16:17], s[16:17], s[0:1]
	s_lshr_b32 s15, s15, 22
	s_add_u32 s10, s10, s15
	s_addc_u32 s11, s11, 0
	s_ashr_i64 s[10:11], s[10:11], 10
	s_xor_b64 s[68:69], s[10:11], s[0:1]
	s_lshl_b64 s[10:11], s[68:69], 5
	s_sub_u32 s0, s16, s10
	v_writelane_b32 v202, s10, 14
	s_subb_u32 s1, s17, s11
	s_mul_i32 s16, s68, s28
	v_writelane_b32 v202, s11, 15
	v_cmp_gt_i64_e64 s[10:11], s[0:1], 1
	s_and_b64 s[10:11], s[10:11], exec
	s_cselect_b32 s49, s1, 0
	s_cselect_b32 s48, s0, 1
	s_mul_i32 s0, s68, s29
	s_mul_hi_u32 s1, s68, s28
	s_add_i32 s0, s1, s0
	s_mul_i32 s1, s69, s28
	s_add_i32 s17, s0, s1
	s_lshl_b64 s[10:11], s[16:17], 5
	s_sub_u32 s0, 0, s10
	s_subb_u32 s1, 0, s11
	s_sub_u32 s10, s26, s10
	s_subb_u32 s11, s27, s11
	s_or_b64 s[22:23], s[10:11], s[48:49]
	s_cmp_lg_u32 s23, 0
	v_writelane_b32 v202, s48, 16
	s_nop 1
	v_writelane_b32 v202, s49, 17
	s_cbranch_scc0 .LBB0_11
	s_ashr_i32 s22, s49, 31
	s_add_u32 s48, s48, s22
	s_mov_b32 s23, s22
	s_addc_u32 s49, s49, s22
	s_xor_b64 s[48:49], s[48:49], s[22:23]
	v_cvt_f32_u32_e32 v0, s48
	v_cvt_f32_u32_e32 v1, s49
	s_sub_u32 s15, 0, s48
	s_subb_u32 s35, 0, s49
	s_mov_b64 s[22:23], 0
	v_fmac_f32_e32 v0, 0x4f800000, v1
	v_rcp_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x5f7ffffc, v0
	v_mul_f32_e32 v1, 0x2f800000, v0
	v_trunc_f32_e32 v1, v1
	v_fmac_f32_e32 v0, 0xcf800000, v1
	v_cvt_u32_f32_e32 v1, v1
	v_cvt_u32_f32_e32 v0, v0
	v_readfirstlane_b32 s50, v1
	v_readfirstlane_b32 s51, v0
	s_mul_i32 s55, s15, s50
	s_mul_hi_u32 s57, s15, s51
	s_mul_i32 s56, s35, s51
	s_add_i32 s55, s57, s55
	s_add_i32 s55, s55, s56
	s_mul_i32 s58, s15, s51
	s_mul_i32 s57, s51, s55
	s_mul_hi_u32 s59, s51, s58
	s_mul_hi_u32 s56, s51, s55
	s_add_u32 s57, s59, s57
	s_addc_u32 s56, 0, s56
	s_mul_hi_u32 s60, s50, s58
	s_mul_i32 s58, s50, s58
	s_add_u32 s57, s57, s58
	s_mul_hi_u32 s59, s50, s55
	s_addc_u32 s56, s56, s60
	s_addc_u32 s57, s59, 0
	s_mul_i32 s55, s50, s55
	s_add_u32 s55, s56, s55
	s_addc_u32 s56, 0, s57
	s_add_u32 s51, s51, s55
	s_addc_u32 s50, s50, s56
	s_mul_i32 s55, s15, s50
	s_mul_hi_u32 s56, s15, s51
	s_add_i32 s55, s56, s55
	s_mul_i32 s35, s35, s51
	s_add_i32 s55, s55, s35
	s_mul_i32 s15, s15, s51
	s_mul_hi_u32 s56, s50, s15
	s_mul_i32 s57, s50, s15
	s_mul_i32 s59, s51, s55
	s_mul_hi_u32 s15, s51, s15
	s_mul_hi_u32 s58, s51, s55
	s_add_u32 s15, s15, s59
	s_addc_u32 s58, 0, s58
	s_add_u32 s15, s15, s57
	s_mul_hi_u32 s35, s50, s55
	s_addc_u32 s15, s58, s56
	s_addc_u32 s35, s35, 0
	s_mul_i32 s55, s50, s55
	s_add_u32 s15, s15, s55
	s_addc_u32 s35, 0, s35
	s_add_u32 s15, s51, s15
	s_addc_u32 s35, s50, s35
	s_ashr_i32 s50, s11, 31
	s_add_u32 s56, s10, s50
	s_mov_b32 s51, s50
	s_addc_u32 s57, s11, s50
	s_xor_b64 s[56:57], s[56:57], s[50:51]
	s_mul_i32 s55, s56, s35
	s_mul_hi_u32 s58, s56, s15
	s_mul_hi_u32 s11, s56, s35
	s_add_u32 s55, s58, s55
	s_addc_u32 s11, 0, s11
	s_mul_hi_u32 s59, s57, s15
	s_mul_i32 s15, s57, s15
	s_add_u32 s15, s55, s15
	s_mul_hi_u32 s58, s57, s35
	s_addc_u32 s11, s11, s59
	s_addc_u32 s15, s58, 0
	s_mul_i32 s35, s57, s35
	s_add_u32 s11, s11, s35
	s_addc_u32 s15, 0, s15
	s_mul_i32 s15, s48, s15
	s_mul_hi_u32 s35, s48, s11
	s_add_i32 s15, s35, s15
	s_mul_i32 s35, s49, s11
	s_add_i32 s15, s15, s35
	s_sub_i32 s35, s57, s15
	s_mul_i32 s11, s48, s11
	s_sub_u32 s11, s56, s11
	s_cselect_b64 s[58:59], -1, 0
	s_subb_u32 s35, s35, s49
	s_sub_u32 s55, s11, s48
	s_cselect_b64 s[60:61], -1, 0
	s_subb_u32 s56, s35, 0
	s_cmp_ge_u32 s56, s49
	s_cselect_b32 s62, -1, 0
	s_cmp_ge_u32 s55, s48
	s_cselect_b32 s63, -1, 0
	s_cmp_eq_u32 s56, s49
	s_cselect_b32 s62, s63, s62
	s_cmp_lg_u64 s[60:61], 0
	s_subb_u32 s35, s35, s49
	s_sub_u32 s60, s55, s48
	s_subb_u32 s35, s35, 0
	s_cmp_lg_u32 s62, 0
	s_cselect_b32 s55, s60, s55
	s_cselect_b32 s35, s35, s56
	s_cmp_lg_u64 s[58:59], 0
	s_subb_u32 s15, s57, s15
	s_cmp_ge_u32 s15, s49
	s_cselect_b32 s56, -1, 0
	s_cmp_ge_u32 s11, s48
	s_cselect_b32 s48, -1, 0
	s_cmp_eq_u32 s15, s49
	s_cselect_b32 s48, s48, s56
	s_cmp_lg_u32 s48, 0
	s_cselect_b32 s49, s35, s15
	s_cselect_b32 s48, s55, s11
	s_xor_b64 s[48:49], s[48:49], s[50:51]
	s_sub_u32 s48, s48, s50
	s_subb_u32 s49, s49, s50
	s_branch .LBB0_12
.LBB0_10:
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v1, v0
	v_mov_b64_e32 v[6:7], v[2:3]
	v_mov_b64_e32 v[10:11], v[2:3]
	v_mov_b64_e32 v[14:15], v[2:3]
	v_mov_b64_e32 v[18:19], v[2:3]
	v_mov_b64_e32 v[22:23], v[2:3]
	v_mov_b64_e32 v[26:27], v[2:3]
	v_mov_b64_e32 v[30:31], v[2:3]
	s_mov_b64 s[40:41], s[18:19]
	s_mov_b64 s[4:5], s[20:21]
	s_mov_b64 s[36:37], s[22:23]
	s_mov_b64 s[8:9], s[24:25]
	v_mov_b64_e32 v[4:5], v[0:1]
	v_mov_b64_e32 v[8:9], v[0:1]
	v_mov_b64_e32 v[12:13], v[0:1]
	v_mov_b64_e32 v[16:17], v[0:1]
	v_mov_b64_e32 v[20:21], v[0:1]
	v_mov_b64_e32 v[24:25], v[0:1]
	v_mov_b64_e32 v[28:29], v[0:1]
	s_branch .LBB0_106
.LBB0_11:
	s_mov_b64 s[22:23], -1
.LBB0_12:
	v_writelane_b32 v202, s66, 30
	v_writelane_b32 v202, s70, 31
	s_andn2_b64 vcc, exec, s[22:23]
	s_nop 0
	v_writelane_b32 v202, s71, 32
	s_cbranch_vccnz .LBB0_14
	v_readlane_b32 s22, v202, 16
	s_sub_i32 s11, 0, s22
	s_mov_b32 s49, 0
	v_cvt_f32_u32_e32 v0, s22
	v_readlane_b32 s23, v202, 17
	v_rcp_iflag_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_nop 0
	v_readfirstlane_b32 s15, v0
	s_mul_i32 s11, s11, s15
	s_mul_hi_u32 s11, s15, s11
	s_add_i32 s15, s15, s11
	s_mul_hi_u32 s11, s10, s15
	s_mul_i32 s11, s11, s22
	s_sub_i32 s10, s10, s11
	s_sub_i32 s11, s10, s22
	s_cmp_ge_u32 s10, s22
	s_cselect_b32 s10, s11, s10
	s_sub_i32 s11, s10, s22
	s_cmp_ge_u32 s10, s22
	s_cselect_b32 s48, s11, s10
.LBB0_14:
	s_and_b32 s10, s14, 0x3ff
	v_writelane_b32 v202, s10, 33
	s_cselect_b64 s[22:23], -1, 0
	v_readlane_b32 s50, v202, 14
	v_readlane_b32 s51, v202, 15
	s_mul_i32 s10, s50, s29
	s_mul_hi_u32 s11, s50, s28
	s_add_i32 s10, s11, s10
	s_mul_i32 s11, s51, s28
	s_add_i32 s10, s10, s11
	s_mul_i32 s11, s50, s28
	v_mov_b32_e32 v0, s11
	v_mov_b32_e32 v1, s10
	s_ashr_i32 s14, s49, 31
	v_readlane_b32 s56, v202, 16
	v_cmp_ge_i64_e32 vcc, s[26:27], v[0:1]
	s_mov_b32 s15, s14
	v_readlane_b32 s57, v202, 17
	s_and_b64 s[10:11], s[22:23], vcc
	s_and_b64 s[14:15], s[14:15], s[56:57]
	s_add_u32 s27, s48, s50
	s_addc_u32 s35, s49, s51
	s_add_u32 s27, s27, s14
	s_addc_u32 s55, s35, s15
	s_mov_b32 s35, s34
	s_xor_b64 s[48:49], s[34:35], s[38:39]
	s_ashr_i32 s14, s29, 31
	s_add_u32 s28, s28, s14
	s_mov_b32 s15, s14
	s_addc_u32 s29, s29, s14
	s_xor_b64 s[38:39], s[28:29], s[14:15]
	v_cvt_f32_u32_e32 v0, s38
	v_cvt_f32_u32_e32 v1, s39
	s_sub_u32 s28, 0, s38
	s_subb_u32 s50, 0, s39
	s_mov_b32 s29, 0
	v_fmac_f32_e32 v0, 0x4f800000, v1
	v_rcp_f32_e32 v0, v0
	v_mov_b32_e32 v65, 0
	v_mul_i32_i24_e32 v2, -16, v97
	v_add_u32_e32 v4, v41, v40
	v_mul_f32_e32 v0, 0x5f7ffffc, v0
	v_mul_f32_e32 v1, 0x2f800000, v0
	v_trunc_f32_e32 v1, v1
	v_fmac_f32_e32 v0, 0xcf800000, v1
	v_cvt_u32_f32_e32 v0, v0
	v_cvt_u32_f32_e32 v1, v1
	v_mov_b32_e32 v5, v65
	v_ashrrev_i32_e32 v3, 31, v2
	v_readfirstlane_b32 s51, v0
	v_readfirstlane_b32 s58, v1
	s_mul_hi_u32 s57, s28, s51
	s_mul_i32 s59, s28, s58
	s_mul_i32 s56, s50, s51
	s_add_i32 s57, s57, s59
	s_add_i32 s57, s57, s56
	s_mul_i32 s60, s28, s51
	s_mul_i32 s59, s51, s57
	s_mul_hi_u32 s61, s51, s60
	s_mul_hi_u32 s56, s51, s57
	s_add_u32 s59, s61, s59
	s_addc_u32 s56, 0, s56
	s_mul_hi_u32 s62, s58, s60
	s_mul_i32 s60, s58, s60
	s_add_u32 s59, s59, s60
	s_mul_hi_u32 s61, s58, s57
	s_addc_u32 s56, s56, s62
	s_addc_u32 s59, s61, 0
	s_mul_i32 s57, s58, s57
	s_add_u32 s56, s56, s57
	s_addc_u32 s57, 0, s59
	s_add_u32 s51, s51, s56
	s_addc_u32 s56, s58, s57
	s_mul_i32 s57, s28, s56
	s_mul_hi_u32 s58, s28, s51
	s_add_i32 s57, s58, s57
	s_mul_i32 s50, s50, s51
	s_add_i32 s57, s57, s50
	s_mul_i32 s28, s28, s51
	s_mul_hi_u32 s58, s56, s28
	s_mul_i32 s59, s56, s28
	s_mul_i32 s61, s51, s57
	s_mul_hi_u32 s28, s51, s28
	s_mul_hi_u32 s60, s51, s57
	s_add_u32 s28, s28, s61
	s_addc_u32 s60, 0, s60
	s_add_u32 s28, s28, s59
	s_mul_hi_u32 s50, s56, s57
	s_addc_u32 s28, s60, s58
	s_addc_u32 s50, s50, 0
	s_mul_i32 s57, s56, s57
	s_add_u32 s28, s28, s57
	s_addc_u32 s50, 0, s50
	s_add_u32 s28, s51, s28
	s_addc_u32 s56, s56, s50
	s_ashr_i32 s50, s49, 31
	s_add_u32 s48, s48, s50
	s_mov_b32 s51, s50
	s_addc_u32 s49, s49, s50
	s_xor_b64 s[48:49], s[48:49], s[50:51]
	s_mul_i32 s58, s48, s56
	s_mul_hi_u32 s59, s48, s28
	s_mul_hi_u32 s57, s48, s56
	s_add_u32 s58, s59, s58
	s_addc_u32 s57, 0, s57
	s_mul_hi_u32 s60, s49, s28
	s_mul_i32 s28, s49, s28
	s_add_u32 s28, s58, s28
	s_mul_hi_u32 s59, s49, s56
	s_addc_u32 s28, s57, s60
	s_addc_u32 s57, s59, 0
	s_mul_i32 s56, s49, s56
	s_add_u32 s28, s28, s56
	s_addc_u32 s58, 0, s57
	s_mul_i32 s56, s38, s58
	s_mul_hi_u32 s57, s38, s28
	s_add_i32 s56, s57, s56
	s_mul_i32 s57, s39, s28
	s_add_i32 s59, s56, s57
	s_sub_i32 s60, s49, s59
	s_mul_i32 s56, s38, s28
	s_sub_u32 s48, s48, s56
	s_cselect_b64 s[56:57], -1, 0
	s_subb_u32 s60, s60, s39
	s_sub_u32 s61, s48, s38
	s_subb_u32 s60, s60, 0
	s_cmp_ge_u32 s60, s39
	s_cselect_b32 s62, -1, 0
	s_cmp_ge_u32 s61, s38
	s_cselect_b32 s61, -1, 0
	s_cmp_eq_u32 s60, s39
	s_cselect_b32 s60, s61, s62
	s_add_u32 s61, s28, 1
	s_addc_u32 s62, s58, 0
	s_add_u32 s63, s28, 2
	s_addc_u32 s66, s58, 0
	s_cmp_lg_u32 s60, 0
	s_cselect_b32 s60, s63, s61
	s_cselect_b32 s61, s66, s62
	s_cmp_lg_u64 s[56:57], 0
	s_subb_u32 s49, s49, s59
	s_cmp_ge_u32 s49, s39
	s_cselect_b32 s56, -1, 0
	s_cmp_ge_u32 s48, s38
	s_cselect_b32 s38, -1, 0
	s_cmp_eq_u32 s49, s39
	s_cselect_b32 s38, s38, s56
	s_cmp_lg_u32 s38, 0
	s_cselect_b32 s39, s61, s58
	s_cselect_b32 s38, s60, s28
	s_xor_b64 s[14:15], s[50:51], s[14:15]
	s_xor_b64 s[38:39], s[38:39], s[14:15]
	s_sub_u32 s14, s38, s14
	s_subb_u32 s15, s39, s15
	s_xor_b64 s[14:15], s[14:15], s[34:35]
	s_lshl_b64 s[14:15], s[14:15], 5
	s_and_b32 s28, s26, 31
	s_or_b64 s[14:15], s[14:15], s[28:29]
	s_and_b64 s[10:11], s[10:11], exec
	s_cselect_b32 s11, s55, s15
	s_cselect_b32 s10, s27, s14
	s_lshl_b64 s[14:15], s[10:11], 5
	v_and_b32_e32 v64, 0x70, v4
	v_lshl_add_u64 v[6:7], v[4:5], 0, s[14:15]
	v_lshl_add_u64 v[0:1], v[64:65], 0, v[2:3]
	v_ashrrev_i32_e32 v5, 31, v7
	v_lshl_add_u64 v[46:47], v[0:1], 0, s[14:15]
	v_xor_b32_e32 v1, v5, v7
	v_ashrrev_i32_e32 v8, 31, v1
	v_lshlrev_b32_e32 v43, 4, v40
	v_readlane_b32 s50, v202, 23
	v_xor_b32_e32 v0, v5, v6
	v_lshrrev_b32_e32 v8, 28, v8
	v_mov_b32_e32 v9, s29
	s_lshl_b64 s[10:11], s[10:11], 9
	v_lshl_add_u32 v64, s50, 7, v43
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[8:9]
	v_lshl_add_u64 v[70:71], s[10:11], 0, v[64:65]
	v_lshlrev_b32_e32 v64, 8, v97
	v_ashrrev_i64 v[0:1], 4, v[0:1]
	v_lshl_add_u64 v[74:75], v[70:71], 0, v[64:65]
	v_xor_b32_e32 v0, v0, v5
	s_movk_i32 s49, 0xff00
	v_mad_u64_u32 v[8:9], s[10:11], v0, s49, v[74:75]
	s_ashr_i32 s70, s45, 31
	v_xor_b32_e32 v1, v1, v5
	s_add_u32 s10, s44, s70
	v_lshlrev_b32_e32 v30, 8, v1
	s_mov_b32 s71, s70
	s_addc_u32 s11, s45, s70
	v_sub_u32_e32 v5, v9, v30
	s_xor_b64 s[34:35], s[10:11], s[70:71]
	v_sub_u32_e32 v9, v5, v0
	v_cvt_f32_u32_e32 v5, s34
	v_cvt_f32_u32_e32 v16, s35
	v_ashrrev_i32_e32 v10, 31, v9
	v_xor_b32_e32 v13, v10, v9
	v_ashrrev_i32_e32 v14, 31, v13
	v_fmac_f32_e32 v5, 0x4f800000, v16
	v_rcp_f32_e32 v5, v5
	v_xor_b32_e32 v12, v10, v8
	v_mov_b32_e32 v15, v14
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[14:15]
	v_mul_f32_e32 v5, 0x5f7ffffc, v5
	v_xor_b32_e32 v20, v12, v14
	v_mul_f32_e32 v12, 0x2f800000, v5
	v_trunc_f32_e32 v12, v12
	v_fmac_f32_e32 v5, 0xcf800000, v12
	v_cvt_u32_f32_e32 v5, v5
	v_cvt_u32_f32_e32 v12, v12
	s_sub_u32 s10, 0, s34
	s_subb_u32 s11, 0, s35
	v_readfirstlane_b32 s14, v5
	v_readfirstlane_b32 s27, v12
	s_mul_hi_u32 s26, s10, s14
	s_mul_i32 s28, s10, s27
	s_mul_i32 s15, s11, s14
	s_add_i32 s26, s26, s28
	s_add_i32 s26, s26, s15
	s_mul_i32 s38, s10, s14
	s_mul_i32 s28, s14, s26
	s_mul_hi_u32 s39, s14, s38
	s_mul_hi_u32 s15, s14, s26
	s_add_u32 s28, s39, s28
	s_addc_u32 s15, 0, s15
	s_mul_hi_u32 s48, s27, s38
	s_mul_i32 s38, s27, s38
	s_add_u32 s28, s28, s38
	s_mul_hi_u32 s39, s27, s26
	s_addc_u32 s15, s15, s48
	s_addc_u32 s28, s39, 0
	s_mul_i32 s26, s27, s26
	s_add_u32 s15, s15, s26
	s_addc_u32 s26, 0, s28
	s_add_u32 s14, s14, s15
	s_addc_u32 s15, s27, s26
	s_mul_i32 s26, s10, s15
	s_mul_hi_u32 s27, s10, s14
	s_add_i32 s26, s27, s26
	s_mul_i32 s11, s11, s14
	s_add_i32 s26, s26, s11
	s_mul_i32 s10, s10, s14
	s_mul_hi_u32 s27, s15, s10
	s_mul_i32 s28, s15, s10
	s_mul_i32 s39, s14, s26
	s_mul_hi_u32 s10, s14, s10
	s_mul_hi_u32 s38, s14, s26
	s_add_u32 s10, s10, s39
	s_addc_u32 s38, 0, s38
	s_add_u32 s10, s10, s28
	s_mul_hi_u32 s11, s15, s26
	s_addc_u32 s10, s38, s27
	s_addc_u32 s11, s11, 0
	s_mul_i32 s26, s15, s26
	s_add_u32 s10, s10, s26
	s_addc_u32 s26, 0, s11
	s_add_u32 s48, s14, s10
	s_addc_u32 s28, s15, s26
	v_xor_b32_e32 v15, v13, v14
	v_mad_u64_u32 v[16:17], s[10:11], v20, s28, 0
	v_mul_hi_u32 v18, v20, s48
	v_mov_b32_e32 v19, v65
	v_mad_u64_u32 v[12:13], s[10:11], v15, s48, 0
	v_lshl_add_u64 v[16:17], v[18:19], 0, v[16:17]
	v_add_co_u32_e32 v5, vcc, v16, v12
	v_mad_u64_u32 v[18:19], s[10:11], v15, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v17, v13, vcc
	v_mov_b32_e32 v13, s29
	s_nop 0
	v_addc_co_u32_e32 v19, vcc, 0, v19, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[18:19]
	v_mul_lo_u32 v5, s35, v12
	v_mul_lo_u32 v13, s34, v13
	v_mad_u64_u32 v[16:17], s[10:11], s34, v12, 0
	v_add3_u32 v13, v17, v13, v5
	v_sub_u32_e32 v17, v15, v13
	v_mov_b32_e32 v5, s35
	v_sub_co_u32_e32 v16, vcc, v20, v16
	v_mov_b32_e32 v11, v10
	s_nop 0
	v_subb_co_u32_e64 v17, s[10:11], v17, v5, vcc
	v_subrev_co_u32_e64 v18, s[10:11], s34, v16
	v_subb_co_u32_e32 v13, vcc, v15, v13, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v17, s[10:11], 0, v17, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s35, v17
	v_cmp_le_u32_e32 vcc, s35, v13
	v_mov_b32_e32 v25, v65
	v_cndmask_b32_e64 v19, 0, -1, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s34, v18
	v_cndmask_b32_e64 v15, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v16
	v_cndmask_b32_e64 v18, 0, -1, s[10:11]
	v_cmp_eq_u32_e64 s[10:11], s35, v17
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v13
	v_cndmask_b32_e64 v17, v19, v18, s[10:11]
	v_add_u32_e32 v18, 2, v12
	v_add_u32_e32 v19, 1, v12
	v_cmp_ne_u32_e64 s[10:11], 0, v17
	v_cndmask_b32_e32 v13, v15, v16, vcc
	v_cmp_ne_u32_e32 vcc, 0, v13
	v_cndmask_b32_e64 v17, v19, v18, s[10:11]
	v_xor_b32_e32 v13, s70, v14
	v_cndmask_b32_e32 v12, v12, v17, vcc
	v_xor_b32_e32 v12, v12, v13
	v_sub_u32_e32 v12, v12, v13
	v_xad_u32 v18, v10, v12, v46
	v_lshl_add_u64 v[12:13], v[8:9], 0, v[10:11]
	v_xor_b32_e32 v20, v12, v10
	v_xor_b32_e32 v19, v13, v10
	v_mad_u64_u32 v[14:15], s[10:11], v20, s28, 0
	v_mul_hi_u32 v16, v20, s48
	v_mov_b32_e32 v17, v65
	v_mad_u64_u32 v[12:13], s[10:11], v19, s48, 0
	v_lshl_add_u64 v[14:15], v[16:17], 0, v[14:15]
	v_add_co_u32_e32 v12, vcc, v14, v12
	v_mad_u64_u32 v[16:17], s[10:11], v19, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v15, v13, vcc
	v_mov_b32_e32 v13, s29
	s_nop 0
	v_addc_co_u32_e32 v17, vcc, 0, v17, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[16:17]
	v_mul_lo_u32 v14, s35, v12
	v_mul_lo_u32 v15, s34, v13
	v_mad_u64_u32 v[12:13], s[10:11], s34, v12, 0
	v_add3_u32 v13, v13, v15, v14
	v_sub_u32_e32 v14, v19, v13
	v_sub_co_u32_e32 v12, vcc, v20, v12
	v_mov_b32_e32 v21, v9
	s_nop 0
	v_subb_co_u32_e64 v14, s[10:11], v14, v5, vcc
	v_subrev_co_u32_e64 v15, s[10:11], s34, v12
	v_subb_co_u32_e32 v13, vcc, v19, v13, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v16, s[14:15], 0, v14, s[10:11]
	v_cmp_le_u32_e64 s[14:15], s35, v16
	v_subb_co_u32_e64 v14, s[10:11], v14, v5, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v17, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v15
	v_cmp_le_u32_e32 vcc, s35, v13
	v_mov_b32_e32 v27, v65
	v_cndmask_b32_e64 v20, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v16
	s_mov_b32 s39, 0x27000
	s_mov_b32 s38, -2
	v_cndmask_b32_e64 v17, v17, v20, s[14:15]
	v_subrev_co_u32_e64 v20, s[10:11], s34, v15
	s_and_b32 s37, s37, 0xffff
	s_nop 0
	v_subbrev_co_u32_e64 v14, s[10:11], 0, v14, s[10:11]
	v_cmp_ne_u32_e64 s[10:11], 0, v17
	v_mov_b32_e32 v35, v65
	v_mov_b32_e32 v37, v65
	v_cndmask_b32_e64 v14, v16, v14, s[10:11]
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v12
	v_mov_b32_e32 v63, v65
	v_mov_b32_e32 v67, v65
	v_cndmask_b32_e64 v17, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v13
	v_mov_b32_e32 v69, v65
	v_mov_b32_e32 v73, v65
	v_cndmask_b32_e32 v16, v16, v17, vcc
	v_cmp_ne_u32_e32 vcc, 0, v16
	v_mov_b32_e32 v77, v65
	v_lshrrev_b32_e32 v4, 4, v4
	v_cndmask_b32_e32 v13, v13, v14, vcc
	v_cndmask_b32_e64 v14, v15, v20, s[10:11]
	v_cndmask_b32_e32 v12, v12, v14, vcc
	v_xor_b32_e32 v12, v12, v10
	v_xor_b32_e32 v13, v13, v10
	v_sub_co_u32_e32 v12, vcc, v12, v10
	v_mul_lo_u32 v14, v18, s33
	s_nop 0
	v_subb_co_u32_e32 v13, vcc, v13, v10, vcc
	v_ashrrev_i32_e32 v13, 31, v13
	v_and_b32_e32 v13, s44, v13
	v_add3_u32 v16, v13, v12, v14
	v_xor_b32_e32 v12, -1, v9
	v_cmp_gt_i32_e64 s[10:11], 0, v9
	v_or_b32_e32 v20, 1, v8
	v_xor_b32_e32 v14, -2, v8
	v_cndmask_b32_e64 v13, v9, v12, s[10:11]
	v_cndmask_b32_e64 v12, v20, v14, s[10:11]
	v_ashrrev_i32_e32 v14, 31, v13
	v_mov_b32_e32 v15, v14
	v_lshl_add_u64 v[18:19], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v17, v18, v14
	v_xor_b32_e32 v12, v19, v14
	v_mad_u64_u32 v[22:23], s[14:15], v17, s28, 0
	v_mul_hi_u32 v24, v17, s48
	v_mad_u64_u32 v[18:19], s[14:15], v12, s48, 0
	v_lshl_add_u64 v[22:23], v[24:25], 0, v[22:23]
	v_add_co_u32_e32 v18, vcc, v22, v18
	v_mad_u64_u32 v[24:25], s[14:15], v12, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v18, vcc, v23, v19, vcc
	v_mov_b32_e32 v19, s29
	s_nop 0
	v_addc_co_u32_e32 v25, vcc, 0, v25, vcc
	v_lshl_add_u64 v[18:19], v[18:19], 0, v[24:25]
	v_mul_lo_u32 v24, s35, v18
	v_mul_lo_u32 v19, s34, v19
	v_mad_u64_u32 v[22:23], s[14:15], s34, v18, 0
	v_add3_u32 v19, v23, v19, v24
	v_sub_u32_e32 v23, v12, v19
	v_sub_co_u32_e32 v17, vcc, v17, v22
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[10:11]
	s_nop 0
	v_subb_co_u32_e64 v22, s[14:15], v23, v5, vcc
	v_subrev_co_u32_e64 v23, s[14:15], s34, v17
	v_subb_co_u32_e32 v12, vcc, v12, v19, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v22, s[14:15], 0, v22, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v22
	v_cmp_le_u32_e32 vcc, s35, v12
	v_mov_b32_e32 v25, v65
	v_cndmask_b32_e64 v24, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v23
	v_cndmask_b32_e64 v19, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v17
	v_cndmask_b32_e64 v23, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v22
	v_cndmask_b32_e64 v17, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v12
	v_cndmask_b32_e64 v22, v24, v23, s[14:15]
	v_add_u32_e32 v23, 2, v18
	v_add_u32_e32 v24, 1, v18
	v_cmp_ne_u32_e64 s[14:15], 0, v22
	v_cndmask_b32_e32 v12, v19, v17, vcc
	v_cmp_ne_u32_e32 vcc, 0, v12
	v_cndmask_b32_e64 v22, v24, v23, s[14:15]
	v_xor_b32_e32 v19, v20, v10
	v_cndmask_b32_e32 v12, v18, v22, vcc
	v_xor_b32_e32 v17, v21, v10
	v_mad_u64_u32 v[22:23], s[14:15], v19, s28, 0
	v_mul_hi_u32 v24, v19, s48
	v_mad_u64_u32 v[20:21], s[14:15], v17, s48, 0
	v_lshl_add_u64 v[22:23], v[24:25], 0, v[22:23]
	v_add_co_u32_e32 v20, vcc, v22, v20
	v_mad_u64_u32 v[24:25], s[14:15], v17, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v20, vcc, v23, v21, vcc
	v_mov_b32_e32 v21, s29
	s_nop 0
	v_addc_co_u32_e32 v25, vcc, 0, v25, vcc
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[24:25]
	v_mul_lo_u32 v22, s35, v20
	v_mul_lo_u32 v23, s34, v21
	v_mad_u64_u32 v[20:21], s[14:15], s34, v20, 0
	v_add3_u32 v21, v21, v23, v22
	v_sub_u32_e32 v22, v17, v21
	v_sub_co_u32_e32 v19, vcc, v19, v20
	v_xor_b32_e32 v18, s70, v14
	s_nop 0
	v_subb_co_u32_e64 v20, s[14:15], v22, v5, vcc
	v_subrev_co_u32_e64 v22, s[14:15], s34, v19
	v_subb_co_u32_e32 v17, vcc, v17, v21, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v23, s[26:27], 0, v20, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v23
	v_subb_co_u32_e64 v20, s[14:15], v20, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v24, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v22
	v_cmp_le_u32_e32 vcc, s35, v17
	v_xor_b32_e32 v12, v12, v18
	v_cndmask_b32_e64 v25, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v23
	v_cndmask_b32_e64 v21, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v19
	v_cndmask_b32_e64 v24, v24, v25, s[26:27]
	v_subrev_co_u32_e64 v25, s[14:15], s34, v22
	v_sub_u32_e32 v12, v12, v18
	s_nop 0
	v_subbrev_co_u32_e64 v20, s[14:15], 0, v20, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v24
	v_xad_u32 v12, v10, v12, v46
	v_mul_lo_u32 v12, v12, s33
	v_cndmask_b32_e64 v20, v23, v20, s[14:15]
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v17
	s_nop 1
	v_cndmask_b32_e32 v21, v21, v23, vcc
	v_cmp_ne_u32_e32 vcc, 0, v21
	v_mov_b32_e32 v21, v9
	s_nop 0
	v_cndmask_b32_e32 v17, v17, v20, vcc
	v_cndmask_b32_e64 v20, v22, v25, s[14:15]
	v_cndmask_b32_e32 v19, v19, v20, vcc
	v_xor_b32_e32 v19, v19, v10
	v_xor_b32_e32 v17, v17, v10
	v_sub_co_u32_e32 v19, vcc, v19, v10
	v_or_b32_e32 v20, 2, v8
	s_nop 0
	v_subb_co_u32_e32 v17, vcc, v17, v10, vcc
	v_ashrrev_i32_e32 v17, 31, v17
	v_and_b32_e32 v17, s44, v17
	v_add3_u32 v17, v17, v19, v12
	v_xor_b32_e32 v12, -3, v8
	v_cndmask_b32_e64 v12, v20, v12, s[10:11]
	v_lshl_add_u64 v[22:23], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v19, v22, v14
	v_xor_b32_e32 v12, v23, v14
	v_mad_u64_u32 v[24:25], s[14:15], v19, s28, 0
	v_mul_hi_u32 v26, v19, s48
	v_mad_u64_u32 v[22:23], s[14:15], v12, s48, 0
	v_lshl_add_u64 v[24:25], v[26:27], 0, v[24:25]
	v_add_co_u32_e32 v22, vcc, v24, v22
	v_mad_u64_u32 v[26:27], s[14:15], v12, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v22, vcc, v25, v23, vcc
	v_mov_b32_e32 v23, s29
	s_nop 0
	v_addc_co_u32_e32 v27, vcc, 0, v27, vcc
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[26:27]
	v_mul_lo_u32 v26, s35, v22
	v_mul_lo_u32 v23, s34, v23
	v_mad_u64_u32 v[24:25], s[14:15], s34, v22, 0
	v_add3_u32 v23, v25, v23, v26
	v_sub_u32_e32 v25, v12, v23
	v_sub_co_u32_e32 v19, vcc, v19, v24
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[10:11]
	s_nop 0
	v_subb_co_u32_e64 v24, s[14:15], v25, v5, vcc
	v_subrev_co_u32_e64 v25, s[14:15], s34, v19
	v_subb_co_u32_e32 v12, vcc, v12, v23, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v24, s[14:15], 0, v24, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v24
	v_cmp_le_u32_e32 vcc, s35, v12
	v_mov_b32_e32 v27, v65
	v_cndmask_b32_e64 v26, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v25
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v19
	v_cndmask_b32_e64 v25, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v24
	v_cndmask_b32_e64 v19, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v12
	v_cndmask_b32_e64 v24, v26, v25, s[14:15]
	v_add_u32_e32 v25, 2, v22
	v_add_u32_e32 v26, 1, v22
	v_cmp_ne_u32_e64 s[14:15], 0, v24
	v_cndmask_b32_e32 v12, v23, v19, vcc
	v_cmp_ne_u32_e32 vcc, 0, v12
	v_cndmask_b32_e64 v24, v26, v25, s[14:15]
	v_xor_b32_e32 v26, v20, v10
	v_cndmask_b32_e32 v12, v22, v24, vcc
	v_xor_b32_e32 v19, v21, v10
	v_mad_u64_u32 v[22:23], s[14:15], v26, s28, 0
	v_mul_hi_u32 v24, v26, s48
	v_mov_b32_e32 v25, v65
	v_mad_u64_u32 v[20:21], s[14:15], v19, s48, 0
	v_lshl_add_u64 v[22:23], v[24:25], 0, v[22:23]
	v_add_co_u32_e32 v20, vcc, v22, v20
	v_mad_u64_u32 v[24:25], s[14:15], v19, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v20, vcc, v23, v21, vcc
	v_mov_b32_e32 v21, s29
	s_nop 0
	v_addc_co_u32_e32 v25, vcc, 0, v25, vcc
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[24:25]
	v_mul_lo_u32 v22, s35, v20
	v_mul_lo_u32 v23, s34, v21
	v_mad_u64_u32 v[20:21], s[14:15], s34, v20, 0
	v_add3_u32 v21, v21, v23, v22
	v_sub_u32_e32 v22, v19, v21
	v_sub_co_u32_e32 v20, vcc, v26, v20
	v_xor_b32_e32 v12, v12, v18
	s_nop 0
	v_subb_co_u32_e64 v22, s[14:15], v22, v5, vcc
	v_subrev_co_u32_e64 v23, s[14:15], s34, v20
	v_subb_co_u32_e32 v19, vcc, v19, v21, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v24, s[26:27], 0, v22, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v24
	v_subb_co_u32_e64 v22, s[14:15], v22, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v25, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v23
	v_cmp_le_u32_e32 vcc, s35, v19
	v_sub_u32_e32 v12, v12, v18
	v_cndmask_b32_e64 v26, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v24
	v_cndmask_b32_e64 v21, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v20
	v_cndmask_b32_e64 v25, v25, v26, s[26:27]
	v_subrev_co_u32_e64 v26, s[14:15], s34, v23
	v_xad_u32 v12, v10, v12, v46
	s_nop 0
	v_subbrev_co_u32_e64 v22, s[14:15], 0, v22, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v25
	v_mul_lo_u32 v12, v12, s33
	s_nop 0
	v_cndmask_b32_e64 v22, v24, v22, s[14:15]
	v_cndmask_b32_e64 v24, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v19
	s_nop 1
	v_cndmask_b32_e32 v21, v21, v24, vcc
	v_cmp_ne_u32_e32 vcc, 0, v21
	v_cndmask_b32_e64 v21, v23, v26, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v20, v20, v21, vcc
	v_cndmask_b32_e32 v19, v19, v22, vcc
	v_xor_b32_e32 v20, v20, v10
	v_xor_b32_e32 v19, v19, v10
	v_sub_co_u32_e32 v20, vcc, v20, v10
	v_mov_b32_e32 v21, v9
	s_nop 0
	v_subb_co_u32_e32 v19, vcc, v19, v10, vcc
	v_ashrrev_i32_e32 v19, 31, v19
	v_and_b32_e32 v19, s44, v19
	v_add3_u32 v19, v19, v20, v12
	v_or_b32_e32 v20, 3, v8
	v_xor_b32_e32 v12, -4, v8
	v_cndmask_b32_e64 v12, v20, v12, s[10:11]
	v_lshl_add_u64 v[22:23], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v28, v22, v14
	v_xor_b32_e32 v12, v23, v14
	v_mad_u64_u32 v[24:25], s[14:15], v28, s28, 0
	v_mul_hi_u32 v26, v28, s48
	v_mad_u64_u32 v[22:23], s[14:15], v12, s48, 0
	v_lshl_add_u64 v[24:25], v[26:27], 0, v[24:25]
	v_add_co_u32_e32 v22, vcc, v24, v22
	v_mad_u64_u32 v[26:27], s[14:15], v12, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v22, vcc, v25, v23, vcc
	v_mov_b32_e32 v23, s29
	s_nop 0
	v_addc_co_u32_e32 v27, vcc, 0, v27, vcc
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[26:27]
	v_mul_lo_u32 v26, s35, v22
	v_mul_lo_u32 v23, s34, v23
	v_mad_u64_u32 v[24:25], s[14:15], s34, v22, 0
	v_add3_u32 v23, v25, v23, v26
	v_sub_u32_e32 v25, v12, v23
	v_sub_co_u32_e32 v24, vcc, v28, v24
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[10:11]
	s_nop 0
	v_subb_co_u32_e64 v25, s[14:15], v25, v5, vcc
	v_subrev_co_u32_e64 v26, s[14:15], s34, v24
	v_subb_co_u32_e32 v12, vcc, v12, v23, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v25, s[14:15], 0, v25, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v25
	v_cmp_le_u32_e32 vcc, s35, v12
	s_nop 0
	v_cndmask_b32_e64 v27, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v26
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v24
	v_cndmask_b32_e64 v26, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v25
	v_cndmask_b32_e64 v24, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v12
	v_cndmask_b32_e64 v25, v27, v26, s[14:15]
	v_add_u32_e32 v26, 2, v22
	v_add_u32_e32 v27, 1, v22
	v_cmp_ne_u32_e64 s[14:15], 0, v25
	v_cndmask_b32_e32 v12, v23, v24, vcc
	v_cmp_ne_u32_e32 vcc, 0, v12
	v_cndmask_b32_e64 v25, v27, v26, s[14:15]
	v_xor_b32_e32 v27, v20, v10
	v_cndmask_b32_e32 v12, v22, v25, vcc
	v_xor_b32_e32 v26, v21, v10
	v_mad_u64_u32 v[22:23], s[14:15], v27, s28, 0
	v_mul_hi_u32 v24, v27, s48
	v_mov_b32_e32 v25, v65
	v_mad_u64_u32 v[20:21], s[14:15], v26, s48, 0
	v_lshl_add_u64 v[22:23], v[24:25], 0, v[22:23]
	v_add_co_u32_e32 v20, vcc, v22, v20
	v_mad_u64_u32 v[24:25], s[14:15], v26, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v20, vcc, v23, v21, vcc
	v_mov_b32_e32 v21, s29
	s_nop 0
	v_addc_co_u32_e32 v25, vcc, 0, v25, vcc
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[24:25]
	v_mul_lo_u32 v22, s35, v20
	v_mul_lo_u32 v23, s34, v21
	v_mad_u64_u32 v[20:21], s[14:15], s34, v20, 0
	v_add3_u32 v21, v21, v23, v22
	v_sub_u32_e32 v22, v26, v21
	v_sub_co_u32_e32 v20, vcc, v27, v20
	v_xor_b32_e32 v12, v12, v18
	s_nop 0
	v_subb_co_u32_e64 v22, s[14:15], v22, v5, vcc
	v_subrev_co_u32_e64 v23, s[14:15], s34, v20
	v_subb_co_u32_e32 v21, vcc, v26, v21, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v24, s[26:27], 0, v22, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v24
	v_subb_co_u32_e64 v22, s[14:15], v22, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v25, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v23
	v_cmp_le_u32_e32 vcc, s35, v21
	v_sub_u32_e32 v12, v12, v18
	v_cndmask_b32_e64 v27, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v24
	v_xad_u32 v12, v10, v12, v46
	v_mul_lo_u32 v12, v12, s33
	v_cndmask_b32_e64 v25, v25, v27, s[26:27]
	v_subrev_co_u32_e64 v27, s[14:15], s34, v23
	s_nop 1
	v_subbrev_co_u32_e64 v22, s[14:15], 0, v22, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v25
	s_nop 1
	v_cndmask_b32_e64 v22, v24, v22, s[14:15]
	v_cndmask_b32_e64 v24, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v20
	s_nop 1
	v_cndmask_b32_e64 v25, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v21
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v25, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	s_nop 1
	v_cndmask_b32_e32 v21, v21, v22, vcc
	v_cndmask_b32_e64 v22, v23, v27, s[14:15]
	v_cndmask_b32_e32 v20, v20, v22, vcc
	v_xor_b32_e32 v20, v20, v10
	v_xor_b32_e32 v21, v21, v10
	v_sub_co_u32_e32 v20, vcc, v20, v10
	v_mov_b32_e32 v27, v65
	s_nop 0
	v_subb_co_u32_e32 v21, vcc, v21, v10, vcc
	v_ashrrev_i32_e32 v21, 31, v21
	v_and_b32_e32 v21, s44, v21
	v_add3_u32 v28, v21, v20, v12
	v_or_b32_e32 v20, 4, v8
	v_xor_b32_e32 v12, -5, v8
	v_cndmask_b32_e64 v12, v20, v12, s[10:11]
	v_lshl_add_u64 v[22:23], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v29, v22, v14
	v_xor_b32_e32 v12, v23, v14
	v_mad_u64_u32 v[24:25], s[14:15], v29, s28, 0
	v_mul_hi_u32 v26, v29, s48
	v_mad_u64_u32 v[22:23], s[14:15], v12, s48, 0
	v_lshl_add_u64 v[24:25], v[26:27], 0, v[24:25]
	v_add_co_u32_e32 v22, vcc, v24, v22
	v_mad_u64_u32 v[26:27], s[14:15], v12, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v22, vcc, v25, v23, vcc
	v_mov_b32_e32 v23, s29
	s_nop 0
	v_addc_co_u32_e32 v27, vcc, 0, v27, vcc
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[26:27]
	v_mul_lo_u32 v26, s35, v22
	v_mul_lo_u32 v23, s34, v23
	v_mad_u64_u32 v[24:25], s[14:15], s34, v22, 0
	v_add3_u32 v23, v25, v23, v26
	v_sub_u32_e32 v25, v12, v23
	v_sub_co_u32_e32 v24, vcc, v29, v24
	v_mov_b32_e32 v21, v9
	s_nop 0
	v_subb_co_u32_e64 v25, s[14:15], v25, v5, vcc
	v_subrev_co_u32_e64 v26, s[14:15], s34, v24
	v_subb_co_u32_e32 v12, vcc, v12, v23, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v25, s[14:15], 0, v25, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v25
	v_cmp_le_u32_e32 vcc, s35, v12
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[10:11]
	v_cndmask_b32_e64 v27, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v26
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v24
	v_cndmask_b32_e64 v26, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v25
	v_cndmask_b32_e64 v24, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v12
	v_cndmask_b32_e64 v25, v27, v26, s[14:15]
	v_add_u32_e32 v26, 2, v22
	v_add_u32_e32 v27, 1, v22
	v_cmp_ne_u32_e64 s[14:15], 0, v25
	v_cndmask_b32_e32 v12, v23, v24, vcc
	v_cmp_ne_u32_e32 vcc, 0, v12
	v_cndmask_b32_e64 v25, v27, v26, s[14:15]
	v_xor_b32_e32 v27, v20, v10
	v_cndmask_b32_e32 v12, v22, v25, vcc
	v_xor_b32_e32 v26, v21, v10
	v_mad_u64_u32 v[22:23], s[14:15], v27, s28, 0
	v_mul_hi_u32 v24, v27, s48
	v_mov_b32_e32 v25, v65
	v_mad_u64_u32 v[20:21], s[14:15], v26, s48, 0
	v_lshl_add_u64 v[22:23], v[24:25], 0, v[22:23]
	v_add_co_u32_e32 v20, vcc, v22, v20
	v_mad_u64_u32 v[24:25], s[14:15], v26, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v20, vcc, v23, v21, vcc
	v_mov_b32_e32 v21, s29
	s_nop 0
	v_addc_co_u32_e32 v25, vcc, 0, v25, vcc
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[24:25]
	v_mul_lo_u32 v22, s35, v20
	v_mul_lo_u32 v23, s34, v21
	v_mad_u64_u32 v[20:21], s[14:15], s34, v20, 0
	v_add3_u32 v21, v21, v23, v22
	v_sub_u32_e32 v22, v26, v21
	v_sub_co_u32_e32 v20, vcc, v27, v20
	v_xor_b32_e32 v12, v12, v18
	s_nop 0
	v_subb_co_u32_e64 v22, s[14:15], v22, v5, vcc
	v_subrev_co_u32_e64 v23, s[14:15], s34, v20
	v_subb_co_u32_e32 v21, vcc, v26, v21, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v24, s[26:27], 0, v22, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v24
	v_subb_co_u32_e64 v22, s[14:15], v22, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v25, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v23
	v_cmp_le_u32_e32 vcc, s35, v21
	v_sub_u32_e32 v12, v12, v18
	v_cndmask_b32_e64 v27, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v24
	v_xad_u32 v12, v10, v12, v46
	v_mul_lo_u32 v12, v12, s33
	v_cndmask_b32_e64 v25, v25, v27, s[26:27]
	v_subrev_co_u32_e64 v27, s[14:15], s34, v23
	s_nop 1
	v_subbrev_co_u32_e64 v22, s[14:15], 0, v22, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v25
	s_nop 1
	v_cndmask_b32_e64 v22, v24, v22, s[14:15]
	v_cndmask_b32_e64 v24, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v20
	s_nop 1
	v_cndmask_b32_e64 v25, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v21
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v25, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	s_nop 1
	v_cndmask_b32_e32 v21, v21, v22, vcc
	v_cndmask_b32_e64 v22, v23, v27, s[14:15]
	v_cndmask_b32_e32 v20, v20, v22, vcc
	v_xor_b32_e32 v20, v20, v10
	v_xor_b32_e32 v21, v21, v10
	v_sub_co_u32_e32 v20, vcc, v20, v10
	v_mov_b32_e32 v27, v65
	s_nop 0
	v_subb_co_u32_e32 v21, vcc, v21, v10, vcc
	v_ashrrev_i32_e32 v21, 31, v21
	v_and_b32_e32 v21, s44, v21
	v_add3_u32 v29, v21, v20, v12
	v_or_b32_e32 v20, 5, v8
	v_xor_b32_e32 v12, -6, v8
	v_cndmask_b32_e64 v12, v20, v12, s[10:11]
	v_lshl_add_u64 v[22:23], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v31, v22, v14
	v_xor_b32_e32 v12, v23, v14
	v_mad_u64_u32 v[24:25], s[14:15], v31, s28, 0
	v_mul_hi_u32 v26, v31, s48
	v_mad_u64_u32 v[22:23], s[14:15], v12, s48, 0
	v_lshl_add_u64 v[24:25], v[26:27], 0, v[24:25]
	v_add_co_u32_e32 v22, vcc, v24, v22
	v_mad_u64_u32 v[26:27], s[14:15], v12, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v22, vcc, v25, v23, vcc
	v_mov_b32_e32 v23, s29
	s_nop 0
	v_addc_co_u32_e32 v27, vcc, 0, v27, vcc
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[26:27]
	v_mul_lo_u32 v26, s35, v22
	v_mul_lo_u32 v23, s34, v23
	v_mad_u64_u32 v[24:25], s[14:15], s34, v22, 0
	v_add3_u32 v23, v25, v23, v26
	v_sub_u32_e32 v25, v12, v23
	v_sub_co_u32_e32 v24, vcc, v31, v24
	v_mov_b32_e32 v21, v9
	s_nop 0
	v_subb_co_u32_e64 v25, s[14:15], v25, v5, vcc
	v_subrev_co_u32_e64 v26, s[14:15], s34, v24
	v_subb_co_u32_e32 v12, vcc, v12, v23, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v25, s[14:15], 0, v25, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v25
	v_cmp_le_u32_e32 vcc, s35, v12
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[10:11]
	v_cndmask_b32_e64 v27, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v26
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v24
	v_cndmask_b32_e64 v26, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v25
	v_cndmask_b32_e64 v24, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v12
	v_cndmask_b32_e64 v25, v27, v26, s[14:15]
	v_add_u32_e32 v26, 2, v22
	v_add_u32_e32 v27, 1, v22
	v_cmp_ne_u32_e64 s[14:15], 0, v25
	v_cndmask_b32_e32 v12, v23, v24, vcc
	v_cmp_ne_u32_e32 vcc, 0, v12
	v_cndmask_b32_e64 v25, v27, v26, s[14:15]
	v_xor_b32_e32 v27, v20, v10
	v_cndmask_b32_e32 v12, v22, v25, vcc
	v_xor_b32_e32 v26, v21, v10
	v_mad_u64_u32 v[22:23], s[14:15], v27, s28, 0
	v_mul_hi_u32 v24, v27, s48
	v_mov_b32_e32 v25, v65
	v_mad_u64_u32 v[20:21], s[14:15], v26, s48, 0
	v_lshl_add_u64 v[22:23], v[24:25], 0, v[22:23]
	v_add_co_u32_e32 v20, vcc, v22, v20
	v_mad_u64_u32 v[24:25], s[14:15], v26, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v20, vcc, v23, v21, vcc
	v_mov_b32_e32 v21, s29
	s_nop 0
	v_addc_co_u32_e32 v25, vcc, 0, v25, vcc
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[24:25]
	v_mul_lo_u32 v22, s35, v20
	v_mul_lo_u32 v23, s34, v21
	v_mad_u64_u32 v[20:21], s[14:15], s34, v20, 0
	v_add3_u32 v21, v21, v23, v22
	v_sub_u32_e32 v22, v26, v21
	v_sub_co_u32_e32 v20, vcc, v27, v20
	v_xor_b32_e32 v12, v12, v18
	s_nop 0
	v_subb_co_u32_e64 v22, s[14:15], v22, v5, vcc
	v_subrev_co_u32_e64 v23, s[14:15], s34, v20
	v_subb_co_u32_e32 v21, vcc, v26, v21, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v24, s[26:27], 0, v22, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v24
	v_subb_co_u32_e64 v22, s[14:15], v22, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v25, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v23
	v_cmp_le_u32_e32 vcc, s35, v21
	v_sub_u32_e32 v12, v12, v18
	v_cndmask_b32_e64 v27, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v24
	v_xad_u32 v12, v12, v10, v46
	v_mul_lo_u32 v12, v12, s33
	v_cndmask_b32_e64 v25, v25, v27, s[26:27]
	v_subrev_co_u32_e64 v27, s[14:15], s34, v23
	s_nop 1
	v_subbrev_co_u32_e64 v22, s[14:15], 0, v22, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v25
	s_nop 1
	v_cndmask_b32_e64 v22, v24, v22, s[14:15]
	v_cndmask_b32_e64 v24, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v20
	s_nop 1
	v_cndmask_b32_e64 v25, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v21
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v25, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	s_nop 1
	v_cndmask_b32_e32 v21, v21, v22, vcc
	v_cndmask_b32_e64 v22, v23, v27, s[14:15]
	v_cndmask_b32_e32 v20, v20, v22, vcc
	v_xor_b32_e32 v20, v20, v10
	v_xor_b32_e32 v21, v21, v10
	v_sub_co_u32_e32 v20, vcc, v20, v10
	v_mov_b32_e32 v27, v65
	s_nop 0
	v_subb_co_u32_e32 v21, vcc, v21, v10, vcc
	v_ashrrev_i32_e32 v21, 31, v21
	v_and_b32_e32 v21, s44, v21
	v_add3_u32 v31, v21, v20, v12
	v_or_b32_e32 v20, 6, v8
	v_xor_b32_e32 v12, -7, v8
	v_cndmask_b32_e64 v12, v20, v12, s[10:11]
	v_lshl_add_u64 v[22:23], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v32, v22, v14
	v_xor_b32_e32 v12, v23, v14
	v_mad_u64_u32 v[24:25], s[14:15], v32, s28, 0
	v_mul_hi_u32 v26, v32, s48
	v_mad_u64_u32 v[22:23], s[14:15], v12, s48, 0
	v_lshl_add_u64 v[24:25], v[26:27], 0, v[24:25]
	v_add_co_u32_e32 v22, vcc, v24, v22
	v_mad_u64_u32 v[26:27], s[14:15], v12, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v22, vcc, v25, v23, vcc
	v_mov_b32_e32 v23, s29
	s_nop 0
	v_addc_co_u32_e32 v27, vcc, 0, v27, vcc
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[26:27]
	v_mul_lo_u32 v26, s35, v22
	v_mul_lo_u32 v23, s34, v23
	v_mad_u64_u32 v[24:25], s[14:15], s34, v22, 0
	v_add3_u32 v23, v25, v23, v26
	v_sub_u32_e32 v25, v12, v23
	v_sub_co_u32_e32 v24, vcc, v32, v24
	v_mov_b32_e32 v21, v9
	s_nop 0
	v_subb_co_u32_e64 v25, s[14:15], v25, v5, vcc
	v_subrev_co_u32_e64 v26, s[14:15], s34, v24
	v_subb_co_u32_e32 v12, vcc, v12, v23, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v25, s[14:15], 0, v25, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v25
	v_cmp_le_u32_e32 vcc, s35, v12
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[10:11]
	v_cndmask_b32_e64 v27, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v26
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v24
	v_cndmask_b32_e64 v26, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v25
	v_cndmask_b32_e64 v24, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v12
	v_cndmask_b32_e64 v25, v27, v26, s[14:15]
	v_add_u32_e32 v26, 2, v22
	v_add_u32_e32 v27, 1, v22
	v_cmp_ne_u32_e64 s[14:15], 0, v25
	v_cndmask_b32_e32 v12, v23, v24, vcc
	v_cmp_ne_u32_e32 vcc, 0, v12
	v_cndmask_b32_e64 v25, v27, v26, s[14:15]
	v_xor_b32_e32 v27, v20, v10
	v_cndmask_b32_e32 v12, v22, v25, vcc
	v_xor_b32_e32 v26, v21, v10
	v_mad_u64_u32 v[22:23], s[14:15], v27, s28, 0
	v_mul_hi_u32 v24, v27, s48
	v_mov_b32_e32 v25, v65
	v_mad_u64_u32 v[20:21], s[14:15], v26, s48, 0
	v_lshl_add_u64 v[22:23], v[24:25], 0, v[22:23]
	v_add_co_u32_e32 v20, vcc, v22, v20
	v_mad_u64_u32 v[24:25], s[14:15], v26, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v20, vcc, v23, v21, vcc
	v_mov_b32_e32 v21, s29
	s_nop 0
	v_addc_co_u32_e32 v25, vcc, 0, v25, vcc
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[24:25]
	v_mul_lo_u32 v22, s35, v20
	v_mul_lo_u32 v23, s34, v21
	v_mad_u64_u32 v[20:21], s[14:15], s34, v20, 0
	v_add3_u32 v21, v21, v23, v22
	v_sub_u32_e32 v22, v26, v21
	v_sub_co_u32_e32 v20, vcc, v27, v20
	v_xor_b32_e32 v12, v12, v18
	s_nop 0
	v_subb_co_u32_e64 v22, s[14:15], v22, v5, vcc
	v_subrev_co_u32_e64 v23, s[14:15], s34, v20
	v_subb_co_u32_e32 v21, vcc, v26, v21, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v24, s[26:27], 0, v22, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v24
	v_subb_co_u32_e64 v22, s[14:15], v22, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v25, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v23
	v_cmp_le_u32_e32 vcc, s35, v21
	v_sub_u32_e32 v12, v12, v18
	v_cndmask_b32_e64 v27, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v24
	v_xad_u32 v12, v12, v10, v46
	v_mul_lo_u32 v12, v12, s33
	v_cndmask_b32_e64 v25, v25, v27, s[26:27]
	v_subrev_co_u32_e64 v27, s[14:15], s34, v23
	s_nop 1
	v_subbrev_co_u32_e64 v22, s[14:15], 0, v22, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v25
	s_nop 1
	v_cndmask_b32_e64 v22, v24, v22, s[14:15]
	v_cndmask_b32_e64 v24, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v20
	s_nop 1
	v_cndmask_b32_e64 v25, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v21
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v25, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	s_nop 1
	v_cndmask_b32_e32 v21, v21, v22, vcc
	v_cndmask_b32_e64 v22, v23, v27, s[14:15]
	v_cndmask_b32_e32 v20, v20, v22, vcc
	v_xor_b32_e32 v20, v20, v10
	v_xor_b32_e32 v21, v21, v10
	v_sub_co_u32_e32 v20, vcc, v20, v10
	v_mov_b32_e32 v27, v65
	s_nop 0
	v_subb_co_u32_e32 v21, vcc, v21, v10, vcc
	v_ashrrev_i32_e32 v21, 31, v21
	v_and_b32_e32 v21, s44, v21
	v_add3_u32 v32, v21, v20, v12
	v_or_b32_e32 v20, 7, v8
	v_xor_b32_e32 v12, -8, v8
	v_cndmask_b32_e64 v12, v20, v12, s[10:11]
	v_lshl_add_u64 v[22:23], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v33, v22, v14
	v_xor_b32_e32 v12, v23, v14
	v_mad_u64_u32 v[24:25], s[14:15], v33, s28, 0
	v_mul_hi_u32 v26, v33, s48
	v_mad_u64_u32 v[22:23], s[14:15], v12, s48, 0
	v_lshl_add_u64 v[24:25], v[26:27], 0, v[24:25]
	v_add_co_u32_e32 v22, vcc, v24, v22
	v_mad_u64_u32 v[26:27], s[14:15], v12, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v22, vcc, v25, v23, vcc
	v_mov_b32_e32 v23, s29
	s_nop 0
	v_addc_co_u32_e32 v27, vcc, 0, v27, vcc
	v_lshl_add_u64 v[22:23], v[22:23], 0, v[26:27]
	v_mul_lo_u32 v26, s35, v22
	v_mul_lo_u32 v23, s34, v23
	v_mad_u64_u32 v[24:25], s[14:15], s34, v22, 0
	v_add3_u32 v23, v25, v23, v26
	v_sub_u32_e32 v25, v12, v23
	v_sub_co_u32_e32 v24, vcc, v33, v24
	v_mov_b32_e32 v21, v9
	s_nop 0
	v_subb_co_u32_e64 v25, s[14:15], v25, v5, vcc
	v_subrev_co_u32_e64 v26, s[14:15], s34, v24
	v_subb_co_u32_e32 v12, vcc, v12, v23, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v25, s[14:15], 0, v25, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v25
	v_cmp_le_u32_e32 vcc, s35, v12
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[10:11]
	v_cndmask_b32_e64 v27, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v26
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v24
	v_cndmask_b32_e64 v26, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v25
	v_cndmask_b32_e64 v24, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v12
	v_cndmask_b32_e64 v25, v27, v26, s[14:15]
	v_add_u32_e32 v26, 2, v22
	v_add_u32_e32 v27, 1, v22
	v_cmp_ne_u32_e64 s[14:15], 0, v25
	v_cndmask_b32_e32 v12, v23, v24, vcc
	v_cmp_ne_u32_e32 vcc, 0, v12
	v_cndmask_b32_e64 v25, v27, v26, s[14:15]
	v_xor_b32_e32 v27, v20, v10
	v_cndmask_b32_e32 v12, v22, v25, vcc
	v_xor_b32_e32 v26, v21, v10
	v_mad_u64_u32 v[22:23], s[14:15], v27, s28, 0
	v_mul_hi_u32 v24, v27, s48
	v_mov_b32_e32 v25, v65
	v_mad_u64_u32 v[20:21], s[14:15], v26, s48, 0
	v_lshl_add_u64 v[22:23], v[24:25], 0, v[22:23]
	v_add_co_u32_e32 v20, vcc, v22, v20
	v_mad_u64_u32 v[24:25], s[14:15], v26, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v20, vcc, v23, v21, vcc
	v_mov_b32_e32 v21, s29
	s_nop 0
	v_addc_co_u32_e32 v25, vcc, 0, v25, vcc
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[24:25]
	v_mul_lo_u32 v22, s35, v20
	v_mul_lo_u32 v23, s34, v21
	v_mad_u64_u32 v[20:21], s[14:15], s34, v20, 0
	v_add3_u32 v21, v21, v23, v22
	v_sub_u32_e32 v22, v26, v21
	v_sub_co_u32_e32 v20, vcc, v27, v20
	v_xor_b32_e32 v12, v12, v18
	s_nop 0
	v_subb_co_u32_e64 v22, s[14:15], v22, v5, vcc
	v_subrev_co_u32_e64 v23, s[14:15], s34, v20
	v_subb_co_u32_e32 v21, vcc, v26, v21, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v24, s[26:27], 0, v22, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v24
	v_subb_co_u32_e64 v22, s[14:15], v22, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v25, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v23
	v_cmp_le_u32_e32 vcc, s35, v21
	v_sub_u32_e32 v12, v12, v18
	v_cndmask_b32_e64 v27, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v24
	v_xad_u32 v12, v12, v10, v46
	v_mul_lo_u32 v12, v12, s33
	v_cndmask_b32_e64 v25, v25, v27, s[26:27]
	v_subrev_co_u32_e64 v27, s[14:15], s34, v23
	v_or_b32_e32 v26, 8, v8
	s_nop 0
	v_subbrev_co_u32_e64 v22, s[14:15], 0, v22, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v25
	s_nop 1
	v_cndmask_b32_e64 v22, v24, v22, s[14:15]
	v_cndmask_b32_e64 v24, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v20
	s_nop 1
	v_cndmask_b32_e64 v25, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v21
	s_nop 1
	v_cndmask_b32_e32 v24, v24, v25, vcc
	v_cmp_ne_u32_e32 vcc, 0, v24
	s_nop 1
	v_cndmask_b32_e32 v21, v21, v22, vcc
	v_cndmask_b32_e64 v22, v23, v27, s[14:15]
	v_cndmask_b32_e32 v20, v20, v22, vcc
	v_xor_b32_e32 v20, v20, v10
	v_xor_b32_e32 v21, v21, v10
	v_sub_co_u32_e32 v20, vcc, v20, v10
	v_mov_b32_e32 v27, v9
	s_nop 0
	v_subb_co_u32_e32 v21, vcc, v21, v10, vcc
	v_ashrrev_i32_e32 v21, 31, v21
	v_and_b32_e32 v21, s44, v21
	v_add3_u32 v12, v21, v20, v12
	buffer_load_ubyte v16, v16, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v21, v17, s[36:39], 0 offen
	buffer_load_ubyte v20, v19, s[36:39], 0 offen
	buffer_load_ubyte v22, v28, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v17, v29, s[36:39], 0 offen
	buffer_load_ubyte v24, v31, s[36:39], 0 offen
	buffer_load_ubyte v23, v32, s[36:39], 0 offen
	buffer_load_ubyte v25, v12, s[36:39], 0 offen
	v_xor_b32_e32 v12, -9, v8
	v_cndmask_b32_e64 v12, v26, v12, s[10:11]
	v_lshl_add_u64 v[28:29], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v19, v28, v14
	v_xor_b32_e32 v12, v29, v14
	v_mad_u64_u32 v[32:33], s[14:15], v19, s28, 0
	v_mul_hi_u32 v34, v19, s48
	v_mad_u64_u32 v[28:29], s[14:15], v12, s48, 0
	v_lshl_add_u64 v[32:33], v[34:35], 0, v[32:33]
	v_add_co_u32_e32 v28, vcc, v32, v28
	v_mad_u64_u32 v[34:35], s[14:15], v12, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v28, vcc, v33, v29, vcc
	v_mov_b32_e32 v29, s29
	s_nop 0
	v_addc_co_u32_e32 v35, vcc, 0, v35, vcc
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[34:35]
	v_mul_lo_u32 v31, s35, v28
	v_mul_lo_u32 v29, s34, v29
	v_mad_u64_u32 v[32:33], s[14:15], s34, v28, 0
	v_add3_u32 v29, v33, v29, v31
	v_sub_u32_e32 v31, v12, v29
	v_sub_co_u32_e32 v19, vcc, v19, v32
	v_lshl_add_u64 v[26:27], v[26:27], 0, v[10:11]
	s_nop 0
	v_subb_co_u32_e64 v31, s[14:15], v31, v5, vcc
	v_subrev_co_u32_e64 v32, s[14:15], s34, v19
	v_subb_co_u32_e32 v12, vcc, v12, v29, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v31, s[14:15], 0, v31, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v31
	v_cmp_le_u32_e32 vcc, s35, v12
	v_mov_b32_e32 v35, v65
	v_cndmask_b32_e64 v33, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v32
	v_cndmask_b32_e64 v29, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v19
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v31
	v_cndmask_b32_e64 v19, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v12
	v_cndmask_b32_e64 v31, v33, v32, s[14:15]
	v_add_u32_e32 v32, 2, v28
	v_add_u32_e32 v33, 1, v28
	v_cmp_ne_u32_e64 s[14:15], 0, v31
	v_cndmask_b32_e32 v12, v29, v19, vcc
	v_cmp_ne_u32_e32 vcc, 0, v12
	v_cndmask_b32_e64 v31, v33, v32, s[14:15]
	v_xor_b32_e32 v19, v27, v10
	v_cndmask_b32_e32 v12, v28, v31, vcc
	v_xor_b32_e32 v31, v26, v10
	v_mad_u64_u32 v[28:29], s[14:15], v31, s28, 0
	v_mul_hi_u32 v32, v31, s48
	v_mov_b32_e32 v33, v65
	v_mad_u64_u32 v[26:27], s[14:15], v19, s48, 0
	v_lshl_add_u64 v[28:29], v[32:33], 0, v[28:29]
	v_add_co_u32_e32 v26, vcc, v28, v26
	v_mad_u64_u32 v[32:33], s[14:15], v19, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v26, vcc, v29, v27, vcc
	v_mov_b32_e32 v27, s29
	s_nop 0
	v_addc_co_u32_e32 v33, vcc, 0, v33, vcc
	v_lshl_add_u64 v[26:27], v[26:27], 0, v[32:33]
	v_mul_lo_u32 v28, s35, v26
	v_mul_lo_u32 v29, s34, v27
	v_mad_u64_u32 v[26:27], s[14:15], s34, v26, 0
	v_add3_u32 v27, v27, v29, v28
	v_sub_u32_e32 v28, v19, v27
	v_sub_co_u32_e32 v26, vcc, v31, v26
	v_xor_b32_e32 v12, v12, v18
	s_nop 0
	v_subb_co_u32_e64 v28, s[14:15], v28, v5, vcc
	v_subrev_co_u32_e64 v29, s[14:15], s34, v26
	v_subb_co_u32_e32 v19, vcc, v19, v27, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v31, s[26:27], 0, v28, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v31
	v_subb_co_u32_e64 v28, s[14:15], v28, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v29
	v_cmp_le_u32_e32 vcc, s35, v19
	v_sub_u32_e32 v12, v12, v18
	v_cndmask_b32_e64 v33, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v31
	v_cndmask_b32_e64 v27, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v26
	v_cndmask_b32_e64 v32, v32, v33, s[26:27]
	v_subrev_co_u32_e64 v33, s[14:15], s34, v29
	v_xad_u32 v12, v12, v10, v46
	s_nop 0
	v_subbrev_co_u32_e64 v28, s[14:15], 0, v28, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v32
	v_mul_lo_u32 v12, v12, s33
	s_nop 0
	v_cndmask_b32_e64 v28, v31, v28, s[14:15]
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v19
	s_nop 1
	v_cndmask_b32_e32 v27, v27, v31, vcc
	v_cmp_ne_u32_e32 vcc, 0, v27
	v_cndmask_b32_e64 v27, v29, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v26, v26, v27, vcc
	v_cndmask_b32_e32 v19, v19, v28, vcc
	v_xor_b32_e32 v26, v26, v10
	v_xor_b32_e32 v19, v19, v10
	v_sub_co_u32_e32 v26, vcc, v26, v10
	v_mov_b32_e32 v27, v9
	s_nop 0
	v_subb_co_u32_e32 v19, vcc, v19, v10, vcc
	v_ashrrev_i32_e32 v19, 31, v19
	v_and_b32_e32 v19, s44, v19
	v_add3_u32 v19, v19, v26, v12
	v_or_b32_e32 v26, 9, v8
	v_xor_b32_e32 v12, -10, v8
	v_cndmask_b32_e64 v12, v26, v12, s[10:11]
	v_lshl_add_u64 v[28:29], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v31, v28, v14
	v_xor_b32_e32 v12, v29, v14
	v_mad_u64_u32 v[32:33], s[14:15], v31, s28, 0
	v_mul_hi_u32 v34, v31, s48
	v_mad_u64_u32 v[28:29], s[14:15], v12, s48, 0
	v_lshl_add_u64 v[32:33], v[34:35], 0, v[32:33]
	v_add_co_u32_e32 v28, vcc, v32, v28
	v_mad_u64_u32 v[34:35], s[14:15], v12, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v28, vcc, v33, v29, vcc
	v_mov_b32_e32 v29, s29
	s_nop 0
	v_addc_co_u32_e32 v35, vcc, 0, v35, vcc
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[34:35]
	v_mul_lo_u32 v34, s35, v28
	v_mul_lo_u32 v29, s34, v29
	v_mad_u64_u32 v[32:33], s[14:15], s34, v28, 0
	v_add3_u32 v29, v33, v29, v34
	v_sub_u32_e32 v33, v12, v29
	v_sub_co_u32_e32 v31, vcc, v31, v32
	v_lshl_add_u64 v[26:27], v[26:27], 0, v[10:11]
	s_nop 0
	v_subb_co_u32_e64 v32, s[14:15], v33, v5, vcc
	v_subrev_co_u32_e64 v33, s[14:15], s34, v31
	v_subb_co_u32_e32 v12, vcc, v12, v29, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v32, s[14:15], 0, v32, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v32
	v_cmp_le_u32_e32 vcc, s35, v12
	s_nop 0
	v_cndmask_b32_e64 v34, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v33
	v_cndmask_b32_e64 v29, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v31
	v_cndmask_b32_e64 v33, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v32
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v12
	v_cndmask_b32_e64 v32, v34, v33, s[14:15]
	v_add_u32_e32 v33, 2, v28
	v_add_u32_e32 v34, 1, v28
	v_cmp_ne_u32_e64 s[14:15], 0, v32
	v_cndmask_b32_e32 v12, v29, v31, vcc
	v_cmp_ne_u32_e32 vcc, 0, v12
	v_cndmask_b32_e64 v32, v34, v33, s[14:15]
	v_xor_b32_e32 v34, v26, v10
	v_cndmask_b32_e32 v12, v28, v32, vcc
	v_xor_b32_e32 v31, v27, v10
	v_mad_u64_u32 v[28:29], s[14:15], v34, s28, 0
	v_mul_hi_u32 v32, v34, s48
	v_mov_b32_e32 v33, v65
	v_mad_u64_u32 v[26:27], s[14:15], v31, s48, 0
	v_lshl_add_u64 v[28:29], v[32:33], 0, v[28:29]
	v_add_co_u32_e32 v26, vcc, v28, v26
	v_mad_u64_u32 v[32:33], s[14:15], v31, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v26, vcc, v29, v27, vcc
	v_mov_b32_e32 v27, s29
	s_nop 0
	v_addc_co_u32_e32 v33, vcc, 0, v33, vcc
	v_lshl_add_u64 v[26:27], v[26:27], 0, v[32:33]
	v_mul_lo_u32 v28, s35, v26
	v_mul_lo_u32 v29, s34, v27
	v_mad_u64_u32 v[26:27], s[14:15], s34, v26, 0
	v_add3_u32 v27, v27, v29, v28
	v_sub_u32_e32 v28, v31, v27
	v_sub_co_u32_e32 v26, vcc, v34, v26
	v_xor_b32_e32 v12, v12, v18
	s_nop 0
	v_subb_co_u32_e64 v28, s[14:15], v28, v5, vcc
	v_subrev_co_u32_e64 v29, s[14:15], s34, v26
	v_subb_co_u32_e32 v27, vcc, v31, v27, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v32, s[26:27], 0, v28, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v32
	v_subb_co_u32_e64 v28, s[14:15], v28, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v33, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v29
	v_cmp_le_u32_e32 vcc, s35, v27
	v_sub_u32_e32 v12, v12, v18
	v_cndmask_b32_e64 v34, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v32
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v26
	v_cndmask_b32_e64 v33, v33, v34, s[26:27]
	v_subrev_co_u32_e64 v34, s[14:15], s34, v29
	v_xad_u32 v12, v12, v10, v46
	s_nop 0
	v_subbrev_co_u32_e64 v28, s[14:15], 0, v28, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v33
	v_mul_lo_u32 v12, v12, s33
	s_nop 0
	v_cndmask_b32_e64 v28, v32, v28, s[14:15]
	v_cndmask_b32_e64 v32, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v27
	s_nop 1
	v_cndmask_b32_e32 v31, v31, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v31
	s_nop 1
	v_cndmask_b32_e32 v27, v27, v28, vcc
	v_cndmask_b32_e64 v28, v29, v34, s[14:15]
	v_cndmask_b32_e32 v26, v26, v28, vcc
	v_xor_b32_e32 v26, v26, v10
	v_xor_b32_e32 v27, v27, v10
	v_sub_co_u32_e32 v26, vcc, v26, v10
	v_or_b32_e32 v28, 10, v8
	s_nop 0
	v_subb_co_u32_e32 v27, vcc, v27, v10, vcc
	v_ashrrev_i32_e32 v27, 31, v27
	v_and_b32_e32 v27, s44, v27
	v_add3_u32 v26, v27, v26, v12
	v_xor_b32_e32 v12, -11, v8
	v_cndmask_b32_e64 v12, v28, v12, s[10:11]
	v_lshl_add_u64 v[32:33], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v27, v32, v14
	v_xor_b32_e32 v12, v33, v14
	v_mad_u64_u32 v[34:35], s[14:15], v27, s28, 0
	v_mul_hi_u32 v36, v27, s48
	v_mad_u64_u32 v[32:33], s[14:15], v12, s48, 0
	v_lshl_add_u64 v[34:35], v[36:37], 0, v[34:35]
	v_add_co_u32_e32 v31, vcc, v34, v32
	v_mad_u64_u32 v[36:37], s[14:15], v12, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v35, v33, vcc
	v_mov_b32_e32 v33, s29
	s_nop 0
	v_addc_co_u32_e32 v37, vcc, 0, v37, vcc
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[36:37]
	v_mul_lo_u32 v31, s35, v32
	v_mul_lo_u32 v33, s34, v33
	v_mad_u64_u32 v[34:35], s[14:15], s34, v32, 0
	v_add3_u32 v31, v35, v33, v31
	v_sub_u32_e32 v33, v12, v31
	v_sub_co_u32_e32 v27, vcc, v27, v34
	v_mov_b32_e32 v29, v9
	s_nop 0
	v_subb_co_u32_e64 v33, s[14:15], v33, v5, vcc
	v_subrev_co_u32_e64 v34, s[14:15], s34, v27
	v_subb_co_u32_e32 v12, vcc, v12, v31, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v33, s[14:15], 0, v33, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v33
	v_cmp_le_u32_e32 vcc, s35, v12
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[10:11]
	v_cndmask_b32_e64 v35, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v34
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v27
	v_cndmask_b32_e64 v34, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v33
	v_cndmask_b32_e64 v27, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v12
	v_cndmask_b32_e64 v33, v35, v34, s[14:15]
	v_add_u32_e32 v34, 2, v32
	v_add_u32_e32 v35, 1, v32
	v_cmp_ne_u32_e64 s[14:15], 0, v33
	v_cndmask_b32_e32 v12, v31, v27, vcc
	v_cmp_ne_u32_e32 vcc, 0, v12
	v_cndmask_b32_e64 v33, v35, v34, s[14:15]
	v_xor_b32_e32 v31, v28, v10
	v_cndmask_b32_e32 v12, v32, v33, vcc
	v_xor_b32_e32 v27, v29, v10
	v_mad_u64_u32 v[32:33], s[14:15], v31, s28, 0
	v_mul_hi_u32 v34, v31, s48
	v_mov_b32_e32 v35, v65
	v_mad_u64_u32 v[28:29], s[14:15], v27, s48, 0
	v_lshl_add_u64 v[32:33], v[34:35], 0, v[32:33]
	v_add_co_u32_e32 v28, vcc, v32, v28
	v_mad_u64_u32 v[34:35], s[14:15], v27, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v28, vcc, v33, v29, vcc
	v_mov_b32_e32 v29, s29
	s_nop 0
	v_addc_co_u32_e32 v35, vcc, 0, v35, vcc
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[34:35]
	v_mul_lo_u32 v32, s35, v28
	v_mul_lo_u32 v33, s34, v29
	v_mad_u64_u32 v[28:29], s[14:15], s34, v28, 0
	v_add3_u32 v29, v29, v33, v32
	v_sub_u32_e32 v32, v27, v29
	v_sub_co_u32_e32 v28, vcc, v31, v28
	v_xor_b32_e32 v12, v12, v18
	s_nop 0
	v_subb_co_u32_e64 v31, s[14:15], v32, v5, vcc
	v_subrev_co_u32_e64 v32, s[14:15], s34, v28
	v_subb_co_u32_e32 v27, vcc, v27, v29, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v33, s[26:27], 0, v31, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v33
	v_subb_co_u32_e64 v31, s[14:15], v31, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v34, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v32
	v_cmp_le_u32_e32 vcc, s35, v27
	v_sub_u32_e32 v12, v12, v18
	v_cndmask_b32_e64 v35, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v33
	v_cndmask_b32_e64 v29, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v28
	v_cndmask_b32_e64 v34, v34, v35, s[26:27]
	v_subrev_co_u32_e64 v35, s[14:15], s34, v32
	v_xad_u32 v12, v12, v10, v46
	s_nop 0
	v_subbrev_co_u32_e64 v31, s[14:15], 0, v31, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v34
	v_mul_lo_u32 v12, v12, s33
	v_mov_b32_e32 v37, v65
	v_cndmask_b32_e64 v31, v33, v31, s[14:15]
	v_cndmask_b32_e64 v33, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v27
	s_nop 1
	v_cndmask_b32_e32 v29, v29, v33, vcc
	v_cmp_ne_u32_e32 vcc, 0, v29
	v_cndmask_b32_e64 v29, v32, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v28, v28, v29, vcc
	v_cndmask_b32_e32 v27, v27, v31, vcc
	v_xor_b32_e32 v28, v28, v10
	v_xor_b32_e32 v27, v27, v10
	v_sub_co_u32_e32 v28, vcc, v28, v10
	v_mov_b32_e32 v29, v9
	s_nop 0
	v_subb_co_u32_e32 v27, vcc, v27, v10, vcc
	v_ashrrev_i32_e32 v27, 31, v27
	v_and_b32_e32 v27, s44, v27
	v_add3_u32 v27, v27, v28, v12
	v_or_b32_e32 v28, 11, v8
	v_xor_b32_e32 v12, -12, v8
	v_cndmask_b32_e64 v12, v28, v12, s[10:11]
	v_lshl_add_u64 v[32:33], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v31, v32, v14
	v_xor_b32_e32 v12, v33, v14
	v_mad_u64_u32 v[34:35], s[14:15], v31, s28, 0
	v_mul_hi_u32 v36, v31, s48
	v_mad_u64_u32 v[32:33], s[14:15], v12, s48, 0
	v_lshl_add_u64 v[34:35], v[36:37], 0, v[34:35]
	v_add_co_u32_e32 v32, vcc, v34, v32
	v_mad_u64_u32 v[36:37], s[14:15], v12, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v35, v33, vcc
	v_mov_b32_e32 v33, s29
	s_nop 0
	v_addc_co_u32_e32 v37, vcc, 0, v37, vcc
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[36:37]
	v_mul_lo_u32 v36, s35, v32
	v_mul_lo_u32 v33, s34, v33
	v_mad_u64_u32 v[34:35], s[14:15], s34, v32, 0
	v_add3_u32 v33, v35, v33, v36
	v_sub_u32_e32 v35, v12, v33
	v_sub_co_u32_e32 v31, vcc, v31, v34
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[10:11]
	s_nop 0
	v_subb_co_u32_e64 v34, s[14:15], v35, v5, vcc
	v_subrev_co_u32_e64 v35, s[14:15], s34, v31
	v_subb_co_u32_e32 v12, vcc, v12, v33, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v34, s[14:15], 0, v34, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v34
	v_cmp_le_u32_e32 vcc, s35, v12
	v_mov_b32_e32 v37, v65
	v_cndmask_b32_e64 v36, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v35
	v_cndmask_b32_e64 v33, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v31
	v_cndmask_b32_e64 v35, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v34
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v12
	v_cndmask_b32_e64 v34, v36, v35, s[14:15]
	v_add_u32_e32 v35, 2, v32
	v_add_u32_e32 v36, 1, v32
	v_cmp_ne_u32_e64 s[14:15], 0, v34
	v_cndmask_b32_e32 v12, v33, v31, vcc
	v_cmp_ne_u32_e32 vcc, 0, v12
	v_cndmask_b32_e64 v34, v36, v35, s[14:15]
	v_xor_b32_e32 v36, v28, v10
	v_cndmask_b32_e32 v12, v32, v34, vcc
	v_xor_b32_e32 v31, v29, v10
	v_mad_u64_u32 v[32:33], s[14:15], v36, s28, 0
	v_mul_hi_u32 v34, v36, s48
	v_mov_b32_e32 v35, v65
	v_mad_u64_u32 v[28:29], s[14:15], v31, s48, 0
	v_lshl_add_u64 v[32:33], v[34:35], 0, v[32:33]
	v_add_co_u32_e32 v28, vcc, v32, v28
	v_mad_u64_u32 v[34:35], s[14:15], v31, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v28, vcc, v33, v29, vcc
	v_mov_b32_e32 v29, s29
	s_nop 0
	v_addc_co_u32_e32 v35, vcc, 0, v35, vcc
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[34:35]
	v_mul_lo_u32 v32, s35, v28
	v_mul_lo_u32 v33, s34, v29
	v_mad_u64_u32 v[28:29], s[14:15], s34, v28, 0
	v_add3_u32 v29, v29, v33, v32
	v_sub_u32_e32 v32, v31, v29
	v_sub_co_u32_e32 v28, vcc, v36, v28
	v_xor_b32_e32 v12, v12, v18
	s_nop 0
	v_subb_co_u32_e64 v32, s[14:15], v32, v5, vcc
	v_subrev_co_u32_e64 v33, s[14:15], s34, v28
	v_subb_co_u32_e32 v29, vcc, v31, v29, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v34, s[26:27], 0, v32, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v34
	v_subb_co_u32_e64 v32, s[14:15], v32, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v35, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v33
	v_cmp_le_u32_e32 vcc, s35, v29
	v_sub_u32_e32 v12, v12, v18
	v_cndmask_b32_e64 v36, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v34
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v28
	v_cndmask_b32_e64 v35, v35, v36, s[26:27]
	v_subrev_co_u32_e64 v36, s[14:15], s34, v33
	v_xad_u32 v12, v12, v10, v46
	s_nop 0
	v_subbrev_co_u32_e64 v32, s[14:15], 0, v32, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v35
	v_mul_lo_u32 v12, v12, s33
	s_nop 0
	v_cndmask_b32_e64 v32, v34, v32, s[14:15]
	v_cndmask_b32_e64 v34, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v29
	s_nop 1
	v_cndmask_b32_e32 v31, v31, v34, vcc
	v_cmp_ne_u32_e32 vcc, 0, v31
	v_cndmask_b32_e64 v31, v33, v36, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v28, v28, v31, vcc
	v_cndmask_b32_e32 v29, v29, v32, vcc
	v_xor_b32_e32 v28, v28, v10
	v_xor_b32_e32 v29, v29, v10
	v_sub_co_u32_e32 v28, vcc, v28, v10
	s_nop 1
	v_subb_co_u32_e32 v29, vcc, v29, v10, vcc
	v_ashrrev_i32_e32 v29, 31, v29
	v_and_b32_e32 v29, s44, v29
	v_add3_u32 v31, v29, v28, v12
	v_or_b32_e32 v28, 12, v8
	v_xor_b32_e32 v12, -13, v8
	v_cndmask_b32_e64 v12, v28, v12, s[10:11]
	v_lshl_add_u64 v[32:33], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v38, v32, v14
	v_xor_b32_e32 v12, v33, v14
	v_mad_u64_u32 v[34:35], s[14:15], v38, s28, 0
	v_mul_hi_u32 v36, v38, s48
	v_mad_u64_u32 v[32:33], s[14:15], v12, s48, 0
	v_lshl_add_u64 v[34:35], v[36:37], 0, v[34:35]
	v_add_co_u32_e32 v32, vcc, v34, v32
	v_mad_u64_u32 v[36:37], s[14:15], v12, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v35, v33, vcc
	v_mov_b32_e32 v33, s29
	s_nop 0
	v_addc_co_u32_e32 v37, vcc, 0, v37, vcc
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[36:37]
	v_mul_lo_u32 v36, s35, v32
	v_mul_lo_u32 v33, s34, v33
	v_mad_u64_u32 v[34:35], s[14:15], s34, v32, 0
	v_add3_u32 v33, v35, v33, v36
	v_sub_u32_e32 v35, v12, v33
	v_sub_co_u32_e32 v34, vcc, v38, v34
	v_mov_b32_e32 v29, v9
	s_nop 0
	v_subb_co_u32_e64 v35, s[14:15], v35, v5, vcc
	v_subrev_co_u32_e64 v36, s[14:15], s34, v34
	v_subb_co_u32_e32 v12, vcc, v12, v33, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v35, s[14:15], 0, v35, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v35
	v_cmp_le_u32_e32 vcc, s35, v12
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[10:11]
	v_cndmask_b32_e64 v37, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v36
	v_cndmask_b32_e64 v33, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v34
	v_cndmask_b32_e64 v36, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v35
	v_cndmask_b32_e64 v34, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v12
	v_cndmask_b32_e64 v35, v37, v36, s[14:15]
	v_add_u32_e32 v36, 2, v32
	v_add_u32_e32 v37, 1, v32
	v_cmp_ne_u32_e64 s[14:15], 0, v35
	v_cndmask_b32_e32 v12, v33, v34, vcc
	v_cmp_ne_u32_e32 vcc, 0, v12
	v_cndmask_b32_e64 v35, v37, v36, s[14:15]
	v_xor_b32_e32 v37, v28, v10
	v_cndmask_b32_e32 v12, v32, v35, vcc
	v_xor_b32_e32 v36, v29, v10
	v_mad_u64_u32 v[32:33], s[14:15], v37, s28, 0
	v_mul_hi_u32 v34, v37, s48
	v_mov_b32_e32 v35, v65
	v_mad_u64_u32 v[28:29], s[14:15], v36, s48, 0
	v_lshl_add_u64 v[32:33], v[34:35], 0, v[32:33]
	v_add_co_u32_e32 v28, vcc, v32, v28
	v_mad_u64_u32 v[34:35], s[14:15], v36, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v28, vcc, v33, v29, vcc
	v_mov_b32_e32 v29, s29
	s_nop 0
	v_addc_co_u32_e32 v35, vcc, 0, v35, vcc
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[34:35]
	v_mul_lo_u32 v32, s35, v28
	v_mul_lo_u32 v33, s34, v29
	v_mad_u64_u32 v[28:29], s[14:15], s34, v28, 0
	v_add3_u32 v29, v29, v33, v32
	v_sub_u32_e32 v32, v36, v29
	v_sub_co_u32_e32 v28, vcc, v37, v28
	v_xor_b32_e32 v12, v12, v18
	s_nop 0
	v_subb_co_u32_e64 v32, s[14:15], v32, v5, vcc
	v_subrev_co_u32_e64 v33, s[14:15], s34, v28
	v_subb_co_u32_e32 v29, vcc, v36, v29, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v34, s[26:27], 0, v32, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v34
	v_subb_co_u32_e64 v32, s[14:15], v32, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v35, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v33
	v_cmp_le_u32_e32 vcc, s35, v29
	v_sub_u32_e32 v12, v12, v18
	v_cndmask_b32_e64 v37, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v34
	v_xad_u32 v12, v12, v10, v46
	v_mul_lo_u32 v12, v12, s33
	v_cndmask_b32_e64 v35, v35, v37, s[26:27]
	v_subrev_co_u32_e64 v37, s[14:15], s34, v33
	s_nop 1
	v_subbrev_co_u32_e64 v32, s[14:15], 0, v32, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v35
	s_nop 1
	v_cndmask_b32_e64 v32, v34, v32, s[14:15]
	v_cndmask_b32_e64 v34, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v28
	s_nop 1
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v29
	s_nop 1
	v_cndmask_b32_e32 v34, v34, v35, vcc
	v_cmp_ne_u32_e32 vcc, 0, v34
	s_nop 1
	v_cndmask_b32_e32 v29, v29, v32, vcc
	v_cndmask_b32_e64 v32, v33, v37, s[14:15]
	v_cndmask_b32_e32 v28, v28, v32, vcc
	v_xor_b32_e32 v28, v28, v10
	v_xor_b32_e32 v29, v29, v10
	v_sub_co_u32_e32 v28, vcc, v28, v10
	v_mov_b32_e32 v37, v65
	s_nop 0
	v_subb_co_u32_e32 v29, vcc, v29, v10, vcc
	v_ashrrev_i32_e32 v29, 31, v29
	v_and_b32_e32 v29, s44, v29
	v_add3_u32 v38, v29, v28, v12
	v_or_b32_e32 v28, 13, v8
	v_xor_b32_e32 v12, -14, v8
	v_cndmask_b32_e64 v12, v28, v12, s[10:11]
	v_lshl_add_u64 v[32:33], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v39, v32, v14
	v_xor_b32_e32 v12, v33, v14
	v_mad_u64_u32 v[34:35], s[14:15], v39, s28, 0
	v_mul_hi_u32 v36, v39, s48
	v_mad_u64_u32 v[32:33], s[14:15], v12, s48, 0
	v_lshl_add_u64 v[34:35], v[36:37], 0, v[34:35]
	v_add_co_u32_e32 v32, vcc, v34, v32
	v_mad_u64_u32 v[36:37], s[14:15], v12, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v35, v33, vcc
	v_mov_b32_e32 v33, s29
	s_nop 0
	v_addc_co_u32_e32 v37, vcc, 0, v37, vcc
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[36:37]
	v_mul_lo_u32 v36, s35, v32
	v_mul_lo_u32 v33, s34, v33
	v_mad_u64_u32 v[34:35], s[14:15], s34, v32, 0
	v_add3_u32 v33, v35, v33, v36
	v_sub_u32_e32 v35, v12, v33
	v_sub_co_u32_e32 v34, vcc, v39, v34
	v_mov_b32_e32 v29, v9
	s_nop 0
	v_subb_co_u32_e64 v35, s[14:15], v35, v5, vcc
	v_subrev_co_u32_e64 v36, s[14:15], s34, v34
	v_subb_co_u32_e32 v12, vcc, v12, v33, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v35, s[14:15], 0, v35, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v35
	v_cmp_le_u32_e32 vcc, s35, v12
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[10:11]
	v_cndmask_b32_e64 v37, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v36
	v_cndmask_b32_e64 v33, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v34
	v_cndmask_b32_e64 v36, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v35
	v_cndmask_b32_e64 v34, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v12
	v_cndmask_b32_e64 v35, v37, v36, s[14:15]
	v_add_u32_e32 v36, 2, v32
	v_add_u32_e32 v37, 1, v32
	v_cmp_ne_u32_e64 s[14:15], 0, v35
	v_cndmask_b32_e32 v12, v33, v34, vcc
	v_cmp_ne_u32_e32 vcc, 0, v12
	v_cndmask_b32_e64 v35, v37, v36, s[14:15]
	v_xor_b32_e32 v37, v28, v10
	v_cndmask_b32_e32 v12, v32, v35, vcc
	v_xor_b32_e32 v36, v29, v10
	v_mad_u64_u32 v[32:33], s[14:15], v37, s28, 0
	v_mul_hi_u32 v34, v37, s48
	v_mov_b32_e32 v35, v65
	v_mad_u64_u32 v[28:29], s[14:15], v36, s48, 0
	v_lshl_add_u64 v[32:33], v[34:35], 0, v[32:33]
	v_add_co_u32_e32 v28, vcc, v32, v28
	v_mad_u64_u32 v[34:35], s[14:15], v36, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v28, vcc, v33, v29, vcc
	v_mov_b32_e32 v29, s29
	s_nop 0
	v_addc_co_u32_e32 v35, vcc, 0, v35, vcc
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[34:35]
	v_mul_lo_u32 v32, s35, v28
	v_mul_lo_u32 v33, s34, v29
	v_mad_u64_u32 v[28:29], s[14:15], s34, v28, 0
	v_add3_u32 v29, v29, v33, v32
	v_sub_u32_e32 v32, v36, v29
	v_sub_co_u32_e32 v28, vcc, v37, v28
	v_xor_b32_e32 v12, v12, v18
	s_nop 0
	v_subb_co_u32_e64 v32, s[14:15], v32, v5, vcc
	v_subrev_co_u32_e64 v33, s[14:15], s34, v28
	v_subb_co_u32_e32 v29, vcc, v36, v29, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v34, s[26:27], 0, v32, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v34
	v_subb_co_u32_e64 v32, s[14:15], v32, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v35, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v33
	v_cmp_le_u32_e32 vcc, s35, v29
	v_sub_u32_e32 v12, v12, v18
	v_cndmask_b32_e64 v37, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v34
	v_xad_u32 v12, v12, v10, v46
	v_mul_lo_u32 v12, v12, s33
	v_cndmask_b32_e64 v35, v35, v37, s[26:27]
	v_subrev_co_u32_e64 v37, s[14:15], s34, v33
	s_nop 1
	v_subbrev_co_u32_e64 v32, s[14:15], 0, v32, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v35
	s_nop 1
	v_cndmask_b32_e64 v32, v34, v32, s[14:15]
	v_cndmask_b32_e64 v34, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v28
	s_nop 1
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v29
	s_nop 1
	v_cndmask_b32_e32 v34, v34, v35, vcc
	v_cmp_ne_u32_e32 vcc, 0, v34
	s_nop 1
	v_cndmask_b32_e32 v29, v29, v32, vcc
	v_cndmask_b32_e64 v32, v33, v37, s[14:15]
	v_cndmask_b32_e32 v28, v28, v32, vcc
	v_xor_b32_e32 v28, v28, v10
	v_xor_b32_e32 v29, v29, v10
	v_sub_co_u32_e32 v28, vcc, v28, v10
	v_mov_b32_e32 v37, v65
	s_nop 0
	v_subb_co_u32_e32 v29, vcc, v29, v10, vcc
	v_ashrrev_i32_e32 v29, 31, v29
	v_and_b32_e32 v29, s44, v29
	v_add3_u32 v39, v29, v28, v12
	v_or_b32_e32 v28, 14, v8
	v_xor_b32_e32 v12, -15, v8
	v_cndmask_b32_e64 v12, v28, v12, s[10:11]
	v_lshl_add_u64 v[32:33], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v47, v32, v14
	v_xor_b32_e32 v12, v33, v14
	v_mad_u64_u32 v[34:35], s[14:15], v47, s28, 0
	v_mul_hi_u32 v36, v47, s48
	v_mad_u64_u32 v[32:33], s[14:15], v12, s48, 0
	v_lshl_add_u64 v[34:35], v[36:37], 0, v[34:35]
	v_add_co_u32_e32 v32, vcc, v34, v32
	v_mad_u64_u32 v[36:37], s[14:15], v12, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v35, v33, vcc
	v_mov_b32_e32 v33, s29
	s_nop 0
	v_addc_co_u32_e32 v37, vcc, 0, v37, vcc
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[36:37]
	v_mul_lo_u32 v36, s35, v32
	v_mul_lo_u32 v33, s34, v33
	v_mad_u64_u32 v[34:35], s[14:15], s34, v32, 0
	v_add3_u32 v33, v35, v33, v36
	v_sub_u32_e32 v35, v12, v33
	v_sub_co_u32_e32 v34, vcc, v47, v34
	v_mov_b32_e32 v29, v9
	s_nop 0
	v_subb_co_u32_e64 v35, s[14:15], v35, v5, vcc
	v_subrev_co_u32_e64 v36, s[14:15], s34, v34
	v_subb_co_u32_e32 v12, vcc, v12, v33, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v35, s[14:15], 0, v35, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v35
	v_cmp_le_u32_e32 vcc, s35, v12
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[10:11]
	v_cndmask_b32_e64 v37, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v36
	v_cndmask_b32_e64 v33, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v34
	v_cndmask_b32_e64 v36, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v35
	v_cndmask_b32_e64 v34, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v12
	v_cndmask_b32_e64 v35, v37, v36, s[14:15]
	v_add_u32_e32 v36, 2, v32
	v_add_u32_e32 v37, 1, v32
	v_cmp_ne_u32_e64 s[14:15], 0, v35
	v_cndmask_b32_e32 v12, v33, v34, vcc
	v_cmp_ne_u32_e32 vcc, 0, v12
	v_cndmask_b32_e64 v35, v37, v36, s[14:15]
	v_xor_b32_e32 v37, v28, v10
	v_cndmask_b32_e32 v12, v32, v35, vcc
	v_xor_b32_e32 v36, v29, v10
	v_mad_u64_u32 v[32:33], s[14:15], v37, s28, 0
	v_mul_hi_u32 v34, v37, s48
	v_mov_b32_e32 v35, v65
	v_mad_u64_u32 v[28:29], s[14:15], v36, s48, 0
	v_lshl_add_u64 v[32:33], v[34:35], 0, v[32:33]
	v_add_co_u32_e32 v28, vcc, v32, v28
	v_mad_u64_u32 v[34:35], s[14:15], v36, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v28, vcc, v33, v29, vcc
	v_mov_b32_e32 v29, s29
	s_nop 0
	v_addc_co_u32_e32 v35, vcc, 0, v35, vcc
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[34:35]
	v_mul_lo_u32 v32, s35, v28
	v_mul_lo_u32 v33, s34, v29
	v_mad_u64_u32 v[28:29], s[14:15], s34, v28, 0
	v_add3_u32 v29, v29, v33, v32
	v_sub_u32_e32 v32, v36, v29
	v_sub_co_u32_e32 v28, vcc, v37, v28
	v_xor_b32_e32 v12, v12, v18
	s_nop 0
	v_subb_co_u32_e64 v32, s[14:15], v32, v5, vcc
	v_subrev_co_u32_e64 v33, s[14:15], s34, v28
	v_subb_co_u32_e32 v29, vcc, v36, v29, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v34, s[26:27], 0, v32, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v34
	v_subb_co_u32_e64 v32, s[14:15], v32, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v35, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v33
	v_cmp_le_u32_e32 vcc, s35, v29
	v_sub_u32_e32 v12, v12, v18
	v_cndmask_b32_e64 v37, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v34
	v_xad_u32 v12, v12, v10, v46
	v_mul_lo_u32 v12, v12, s33
	v_cndmask_b32_e64 v35, v35, v37, s[26:27]
	v_subrev_co_u32_e64 v37, s[14:15], s34, v33
	s_nop 1
	v_subbrev_co_u32_e64 v32, s[14:15], 0, v32, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v35
	s_nop 1
	v_cndmask_b32_e64 v32, v34, v32, s[14:15]
	v_cndmask_b32_e64 v34, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v28
	s_nop 1
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v29
	s_nop 1
	v_cndmask_b32_e32 v34, v34, v35, vcc
	v_cmp_ne_u32_e32 vcc, 0, v34
	s_nop 1
	v_cndmask_b32_e32 v29, v29, v32, vcc
	v_cndmask_b32_e64 v32, v33, v37, s[14:15]
	v_cndmask_b32_e32 v28, v28, v32, vcc
	v_xor_b32_e32 v28, v28, v10
	v_xor_b32_e32 v29, v29, v10
	v_sub_co_u32_e32 v28, vcc, v28, v10
	v_mov_b32_e32 v33, v65
	s_nop 0
	v_subb_co_u32_e32 v29, vcc, v29, v10, vcc
	v_ashrrev_i32_e32 v29, 31, v29
	v_and_b32_e32 v29, s44, v29
	v_add3_u32 v34, v29, v28, v12
	v_or_b32_e32 v28, 15, v8
	v_xor_b32_e32 v12, -16, v8
	v_cndmask_b32_e64 v12, v28, v12, s[10:11]
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[14:15]
	v_xor_b32_e32 v36, v12, v14
	v_xor_b32_e32 v35, v13, v14
	v_mad_u64_u32 v[14:15], s[10:11], v36, s28, 0
	v_mul_hi_u32 v32, v36, s48
	v_mad_u64_u32 v[12:13], s[10:11], v35, s48, 0
	v_lshl_add_u64 v[14:15], v[32:33], 0, v[14:15]
	v_add_co_u32_e32 v12, vcc, v14, v12
	v_mad_u64_u32 v[32:33], s[10:11], v35, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v15, v13, vcc
	v_mov_b32_e32 v13, s29
	s_nop 0
	v_addc_co_u32_e32 v33, vcc, 0, v33, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[32:33]
	v_mul_lo_u32 v32, s35, v12
	v_mul_lo_u32 v13, s34, v13
	v_mad_u64_u32 v[14:15], s[10:11], s34, v12, 0
	v_add3_u32 v13, v15, v13, v32
	v_sub_u32_e32 v15, v35, v13
	v_sub_co_u32_e32 v14, vcc, v36, v14
	v_mov_b32_e32 v29, v9
	s_nop 0
	v_subb_co_u32_e64 v15, s[10:11], v15, v5, vcc
	v_subrev_co_u32_e64 v32, s[10:11], s34, v14
	v_subb_co_u32_e32 v13, vcc, v35, v13, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v15, s[10:11], 0, v15, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s35, v15
	v_cmp_le_u32_e32 vcc, s35, v13
	v_mov_b32_e32 v35, v65
	v_cndmask_b32_e64 v33, 0, -1, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s34, v32
	s_nop 1
	v_cndmask_b32_e64 v32, 0, -1, s[10:11]
	v_cmp_eq_u32_e64 s[10:11], s35, v15
	s_nop 1
	v_cndmask_b32_e64 v15, v33, v32, s[10:11]
	v_add_u32_e32 v32, 2, v12
	v_add_u32_e32 v33, 1, v12
	v_cmp_ne_u32_e64 s[10:11], 0, v15
	s_nop 1
	v_cndmask_b32_e64 v15, v33, v32, s[10:11]
	v_cndmask_b32_e64 v32, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v14
	s_nop 1
	v_cndmask_b32_e64 v14, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v13
	s_nop 1
	v_cndmask_b32_e32 v13, v32, v14, vcc
	v_cmp_ne_u32_e32 vcc, 0, v13
	s_nop 1
	v_cndmask_b32_e32 v12, v12, v15, vcc
	v_xor_b32_e32 v12, v12, v18
	v_sub_u32_e32 v12, v12, v18
	v_xad_u32 v18, v12, v10, v46
	v_lshl_add_u64 v[12:13], v[28:29], 0, v[10:11]
	v_xor_b32_e32 v32, v12, v10
	v_xor_b32_e32 v11, v13, v10
	v_mad_u64_u32 v[14:15], s[10:11], v32, s28, 0
	v_mul_hi_u32 v28, v32, s48
	v_mov_b32_e32 v29, v65
	v_mad_u64_u32 v[12:13], s[10:11], v11, s48, 0
	v_lshl_add_u64 v[14:15], v[28:29], 0, v[14:15]
	v_add_co_u32_e32 v12, vcc, v14, v12
	v_mad_u64_u32 v[28:29], s[10:11], v11, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v15, v13, vcc
	v_mov_b32_e32 v13, s29
	s_nop 0
	v_addc_co_u32_e32 v29, vcc, 0, v29, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[28:29]
	v_mul_lo_u32 v14, s35, v12
	v_mul_lo_u32 v15, s34, v13
	v_mad_u64_u32 v[12:13], s[10:11], s34, v12, 0
	v_add3_u32 v13, v13, v15, v14
	v_sub_u32_e32 v14, v11, v13
	v_sub_co_u32_e32 v12, vcc, v32, v12
	s_nop 1
	v_subb_co_u32_e64 v14, s[10:11], v14, v5, vcc
	v_subrev_co_u32_e64 v15, s[10:11], s34, v12
	v_subb_co_u32_e32 v11, vcc, v11, v13, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v28, s[14:15], 0, v14, s[10:11]
	v_cmp_le_u32_e64 s[14:15], s35, v28
	v_subb_co_u32_e64 v14, s[10:11], v14, v5, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v29, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v15
	v_cmp_le_u32_e32 vcc, s35, v11
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v28
	v_cndmask_b32_e64 v13, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v12
	v_cndmask_b32_e64 v29, v29, v32, s[14:15]
	v_subrev_co_u32_e64 v32, s[10:11], s34, v15
	s_nop 1
	v_subbrev_co_u32_e64 v14, s[10:11], 0, v14, s[10:11]
	v_cmp_ne_u32_e64 s[10:11], 0, v29
	s_nop 1
	v_cndmask_b32_e64 v14, v28, v14, s[10:11]
	v_cndmask_b32_e64 v28, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v11
	s_nop 1
	v_cndmask_b32_e32 v13, v13, v28, vcc
	v_cmp_ne_u32_e32 vcc, 0, v13
	v_cndmask_b32_e64 v13, v15, v32, s[10:11]
	s_mov_b64 s[10:11], 0x400
	v_cndmask_b32_e32 v12, v12, v13, vcc
	v_cndmask_b32_e32 v11, v11, v14, vcc
	v_xor_b32_e32 v12, v12, v10
	v_xor_b32_e32 v11, v11, v10
	v_sub_co_u32_e32 v12, vcc, v12, v10
	s_nop 1
	v_subb_co_u32_e32 v10, vcc, v11, v10, vcc
	v_ashrrev_i32_e32 v10, 31, v10
	v_and_b32_e32 v10, s44, v10
	v_mul_lo_u32 v11, v18, s33
	v_add3_u32 v10, v10, v12, v11
	buffer_load_ubyte v14, v19, s[36:39], 0 offen
	buffer_load_ubyte v18, v26, s[36:39], 0 offen
	buffer_load_ubyte v15, v27, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v26, v31, s[36:39], 0 offen
	buffer_load_ubyte v19, v38, s[36:39], 0 offen
	buffer_load_ubyte v28, v39, s[36:39], 0 offen
	buffer_load_ubyte v27, v34, s[36:39], 0 offen
	buffer_load_ubyte v29, v10, s[36:39], 0 offen
	v_lshl_add_u64 v[10:11], v[8:9], 0, s[10:11]
	s_movk_i32 s10, 0xfbff
	v_sub_co_u32_e32 v8, vcc, s10, v8
	v_mov_b32_e32 v39, v65
	s_nop 0
	v_subb_co_u32_e32 v9, vcc, -1, v9, vcc
	v_cmp_gt_i32_e32 vcc, 0, v11
	s_nop 1
	v_cndmask_b32_e32 v9, v11, v9, vcc
	v_ashrrev_i32_e32 v12, 31, v9
	v_cndmask_b32_e32 v8, v10, v8, vcc
	v_mov_b32_e32 v13, v12
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[12:13]
	v_xor_b32_e32 v31, v8, v12
	v_xor_b32_e32 v13, v9, v12
	v_mad_u64_u32 v[32:33], s[10:11], v31, s28, 0
	v_mul_hi_u32 v34, v31, s48
	v_mad_u64_u32 v[8:9], s[10:11], v13, s48, 0
	v_lshl_add_u64 v[32:33], v[34:35], 0, v[32:33]
	v_add_co_u32_e32 v8, vcc, v32, v8
	v_mad_u64_u32 v[34:35], s[10:11], v13, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v8, vcc, v33, v9, vcc
	v_mov_b32_e32 v9, s29
	s_nop 0
	v_addc_co_u32_e32 v35, vcc, 0, v35, vcc
	v_lshl_add_u64 v[8:9], v[8:9], 0, v[34:35]
	v_mul_lo_u32 v34, s35, v8
	v_mul_lo_u32 v9, s34, v9
	v_mad_u64_u32 v[32:33], s[10:11], s34, v8, 0
	v_add3_u32 v9, v33, v9, v34
	v_sub_u32_e32 v33, v13, v9
	v_sub_co_u32_e32 v31, vcc, v31, v32
	s_nop 1
	v_subb_co_u32_e64 v32, s[10:11], v33, v5, vcc
	v_subrev_co_u32_e64 v33, s[10:11], s34, v31
	v_subb_co_u32_e32 v9, vcc, v13, v9, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v32, s[10:11], 0, v32, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s35, v32
	v_cmp_le_u32_e32 vcc, s35, v9
	s_nop 0
	v_cndmask_b32_e64 v34, 0, -1, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s34, v33
	v_cndmask_b32_e64 v13, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v31
	v_cndmask_b32_e64 v33, 0, -1, s[10:11]
	v_cmp_eq_u32_e64 s[10:11], s35, v32
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v9
	v_cndmask_b32_e64 v32, v34, v33, s[10:11]
	v_add_u32_e32 v33, 2, v8
	v_add_u32_e32 v34, 1, v8
	v_cmp_ne_u32_e64 s[10:11], 0, v32
	v_cndmask_b32_e32 v9, v13, v31, vcc
	v_cmp_ne_u32_e32 vcc, 0, v9
	v_cndmask_b32_e64 v32, v34, v33, s[10:11]
	v_xor_b32_e32 v9, s70, v12
	v_cndmask_b32_e32 v8, v8, v32, vcc
	v_xor_b32_e32 v8, v8, v9
	v_sub_u32_e32 v12, v8, v9
	v_ashrrev_i32_e32 v8, 31, v11
	v_mov_b32_e32 v9, v8
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[8:9]
	v_xor_b32_e32 v34, v10, v8
	v_xad_u32 v31, v12, v8, v46
	v_xor_b32_e32 v9, v11, v8
	v_mad_u64_u32 v[12:13], s[10:11], v34, s28, 0
	v_mul_hi_u32 v32, v34, s48
	v_mov_b32_e32 v33, v65
	v_mad_u64_u32 v[10:11], s[10:11], v9, s48, 0
	v_lshl_add_u64 v[12:13], v[32:33], 0, v[12:13]
	v_add_co_u32_e32 v10, vcc, v12, v10
	v_mad_u64_u32 v[32:33], s[10:11], v9, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v13, v11, vcc
	v_mov_b32_e32 v11, s29
	s_nop 0
	v_addc_co_u32_e32 v33, vcc, 0, v33, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[32:33]
	v_mul_lo_u32 v12, s35, v10
	v_mul_lo_u32 v13, s34, v11
	v_mad_u64_u32 v[10:11], s[10:11], s34, v10, 0
	v_add3_u32 v11, v11, v13, v12
	v_sub_u32_e32 v12, v9, v11
	v_sub_co_u32_e32 v10, vcc, v34, v10
	s_nop 1
	v_subb_co_u32_e64 v12, s[10:11], v12, v5, vcc
	v_subrev_co_u32_e64 v13, s[10:11], s34, v10
	v_subb_co_u32_e32 v9, vcc, v9, v11, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v32, s[14:15], 0, v12, s[10:11]
	v_cmp_le_u32_e64 s[14:15], s35, v32
	v_subb_co_u32_e64 v12, s[10:11], v12, v5, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v33, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v13
	v_cmp_le_u32_e32 vcc, s35, v9
	s_nop 0
	v_cndmask_b32_e64 v34, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v32
	v_cndmask_b32_e64 v11, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v10
	v_cndmask_b32_e64 v33, v33, v34, s[14:15]
	v_subrev_co_u32_e64 v34, s[10:11], s34, v13
	s_nop 1
	v_subbrev_co_u32_e64 v12, s[10:11], 0, v12, s[10:11]
	v_cmp_ne_u32_e64 s[10:11], 0, v33
	s_nop 1
	v_cndmask_b32_e64 v12, v32, v12, s[10:11]
	v_cndmask_b32_e64 v32, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v9
	s_nop 1
	v_cndmask_b32_e32 v11, v11, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v11
	v_cndmask_b32_e64 v11, v13, v34, s[10:11]
	s_nop 0
	v_cndmask_b32_e32 v10, v10, v11, vcc
	v_cndmask_b32_e32 v9, v9, v12, vcc
	v_xor_b32_e32 v10, v10, v8
	v_xor_b32_e32 v9, v9, v8
	v_sub_co_u32_e32 v10, vcc, v10, v8
	s_nop 1
	v_subb_co_u32_e32 v9, vcc, v9, v8, vcc
	v_ashrrev_i32_e32 v8, 31, v9
	v_and_b32_e32 v8, s44, v8
	v_mul_lo_u32 v9, v31, s33
	v_add3_u32 v31, v8, v10, v9
	v_lshrrev_b32_e32 v10, 5, v40
	v_mul_i32_i24_e32 v100, 0xfffffe00, v10
	v_ashrrev_i32_e32 v101, 31, v100
	v_mad_u64_u32 v[10:11], s[10:11], v0, s49, v[100:101]
	v_and_b32_e32 v8, 0x200, v43
	v_sub_u32_e32 v11, v11, v30
	v_or_b32_e32 v8, 0x400, v8
	v_mov_b32_e32 v9, v65
	v_sub_u32_e32 v11, v11, v0
	v_lshl_add_u64 v[8:9], v[10:11], 0, v[8:9]
	v_lshl_add_u64 v[12:13], v[8:9], 0, v[74:75]
	v_sub_co_u32_e32 v10, vcc, -2, v12
	v_cmp_gt_i32_e64 s[10:11], 0, v13
	s_nop 0
	v_subb_co_u32_e32 v11, vcc, -1, v13, vcc
	v_cndmask_b32_e64 v11, v13, v11, s[10:11]
	v_or_b32_e32 v32, 1, v12
	v_ashrrev_i32_e32 v34, 31, v11
	v_cndmask_b32_e64 v10, v32, v10, s[10:11]
	v_mov_b32_e32 v35, v34
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[34:35]
	v_xor_b32_e32 v35, v10, v34
	v_xor_b32_e32 v30, v11, v34
	v_mad_u64_u32 v[36:37], s[14:15], v35, s28, 0
	v_mul_hi_u32 v38, v35, s48
	v_mad_u64_u32 v[10:11], s[14:15], v30, s48, 0
	v_lshl_add_u64 v[36:37], v[38:39], 0, v[36:37]
	v_add_co_u32_e32 v10, vcc, v36, v10
	v_mad_u64_u32 v[38:39], s[14:15], v30, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v37, v11, vcc
	v_mov_b32_e32 v11, s29
	s_nop 0
	v_addc_co_u32_e32 v39, vcc, 0, v39, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[38:39]
	v_mul_lo_u32 v38, s35, v10
	v_mul_lo_u32 v11, s34, v11
	v_mad_u64_u32 v[36:37], s[14:15], s34, v10, 0
	v_add3_u32 v11, v37, v11, v38
	v_sub_u32_e32 v37, v30, v11
	v_sub_co_u32_e32 v35, vcc, v35, v36
	v_mov_b32_e32 v33, v13
	s_nop 0
	v_subb_co_u32_e64 v36, s[14:15], v37, v5, vcc
	v_subrev_co_u32_e64 v37, s[14:15], s34, v35
	v_subb_co_u32_e32 v11, vcc, v30, v11, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v36, s[14:15], 0, v36, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v36
	v_cmp_le_u32_e32 vcc, s35, v11
	s_nop 0
	v_cndmask_b32_e64 v38, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v37
	v_cndmask_b32_e64 v30, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v35
	v_cndmask_b32_e64 v37, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v36
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v11
	v_cndmask_b32_e64 v36, v38, v37, s[14:15]
	v_add_u32_e32 v37, 2, v10
	v_add_u32_e32 v38, 1, v10
	v_cmp_ne_u32_e64 s[14:15], 0, v36
	v_cndmask_b32_e32 v11, v30, v35, vcc
	v_cmp_ne_u32_e32 vcc, 0, v11
	v_cndmask_b32_e64 v36, v38, v37, s[14:15]
	v_xor_b32_e32 v11, s70, v34
	v_cndmask_b32_e32 v10, v10, v36, vcc
	v_xor_b32_e32 v10, v10, v11
	v_sub_u32_e32 v30, v10, v11
	v_ashrrev_i32_e32 v10, 31, v13
	v_mov_b32_e32 v11, v10
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[10:11]
	v_xor_b32_e32 v39, v32, v10
	v_xor_b32_e32 v38, v33, v10
	v_mad_u64_u32 v[34:35], s[14:15], v39, s28, 0
	v_mul_hi_u32 v36, v39, s48
	v_mov_b32_e32 v37, v65
	v_mad_u64_u32 v[32:33], s[14:15], v38, s48, 0
	v_lshl_add_u64 v[34:35], v[36:37], 0, v[34:35]
	v_add_co_u32_e32 v32, vcc, v34, v32
	v_mad_u64_u32 v[36:37], s[14:15], v38, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v35, v33, vcc
	v_mov_b32_e32 v33, s29
	s_nop 0
	v_addc_co_u32_e32 v37, vcc, 0, v37, vcc
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[36:37]
	v_mul_lo_u32 v34, s35, v32
	v_mul_lo_u32 v35, s34, v33
	v_mad_u64_u32 v[32:33], s[14:15], s34, v32, 0
	v_add3_u32 v33, v33, v35, v34
	v_sub_u32_e32 v34, v38, v33
	v_sub_co_u32_e32 v32, vcc, v39, v32
	v_xad_u32 v30, v30, v10, v46
	s_nop 0
	v_subb_co_u32_e64 v34, s[14:15], v34, v5, vcc
	v_subrev_co_u32_e64 v35, s[14:15], s34, v32
	v_subb_co_u32_e32 v33, vcc, v38, v33, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v36, s[26:27], 0, v34, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v36
	v_subb_co_u32_e64 v34, s[14:15], v34, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v37, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v35
	v_cmp_le_u32_e32 vcc, s35, v33
	v_mul_lo_u32 v30, v30, s33
	v_cndmask_b32_e64 v39, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v36
	s_nop 1
	v_cndmask_b32_e64 v37, v37, v39, s[26:27]
	v_subrev_co_u32_e64 v39, s[14:15], s34, v35
	s_nop 1
	v_subbrev_co_u32_e64 v34, s[14:15], 0, v34, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v37
	s_nop 1
	v_cndmask_b32_e64 v34, v36, v34, s[14:15]
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v32
	s_nop 1
	v_cndmask_b32_e64 v37, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v33
	s_nop 1
	v_cndmask_b32_e32 v36, v36, v37, vcc
	v_cmp_ne_u32_e32 vcc, 0, v36
	s_nop 1
	v_cndmask_b32_e32 v33, v33, v34, vcc
	v_cndmask_b32_e64 v34, v35, v39, s[14:15]
	v_cndmask_b32_e32 v32, v32, v34, vcc
	v_xor_b32_e32 v32, v32, v10
	v_xor_b32_e32 v33, v33, v10
	v_sub_co_u32_e32 v32, vcc, v32, v10
	s_nop 1
	v_subb_co_u32_e32 v33, vcc, v33, v10, vcc
	v_sub_co_u32_e32 v34, vcc, -3, v12
	v_ashrrev_i32_e32 v33, 31, v33
	s_nop 0
	v_subb_co_u32_e32 v35, vcc, -1, v13, vcc
	v_and_b32_e32 v33, s44, v33
	v_cndmask_b32_e64 v35, v13, v35, s[10:11]
	v_add3_u32 v30, v33, v32, v30
	v_or_b32_e32 v32, 2, v12
	v_ashrrev_i32_e32 v36, 31, v35
	v_cndmask_b32_e64 v34, v32, v34, s[10:11]
	v_mov_b32_e32 v37, v36
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[36:37]
	v_xor_b32_e32 v47, v34, v36
	v_xor_b32_e32 v37, v35, v36
	v_mad_u64_u32 v[38:39], s[14:15], v47, s28, 0
	v_mul_hi_u32 v62, v47, s48
	v_mad_u64_u32 v[34:35], s[14:15], v37, s48, 0
	v_lshl_add_u64 v[38:39], v[62:63], 0, v[38:39]
	v_add_co_u32_e32 v34, vcc, v38, v34
	v_mad_u64_u32 v[62:63], s[14:15], v37, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v34, vcc, v39, v35, vcc
	v_mov_b32_e32 v35, s29
	s_nop 0
	v_addc_co_u32_e32 v63, vcc, 0, v63, vcc
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[62:63]
	v_mul_lo_u32 v49, s35, v34
	v_mul_lo_u32 v35, s34, v35
	v_mad_u64_u32 v[38:39], s[14:15], s34, v34, 0
	v_add3_u32 v35, v39, v35, v49
	v_sub_u32_e32 v39, v37, v35
	v_sub_co_u32_e32 v38, vcc, v47, v38
	v_mov_b32_e32 v33, v13
	s_nop 0
	v_subb_co_u32_e64 v39, s[14:15], v39, v5, vcc
	v_subrev_co_u32_e64 v47, s[14:15], s34, v38
	v_subb_co_u32_e32 v35, vcc, v37, v35, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v39, s[14:15], 0, v39, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v39
	v_cmp_le_u32_e32 vcc, s35, v35
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[10:11]
	v_cndmask_b32_e64 v49, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v47
	v_cndmask_b32_e64 v37, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v38
	v_cndmask_b32_e64 v47, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v39
	v_cndmask_b32_e64 v38, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v35
	v_cndmask_b32_e64 v39, v49, v47, s[14:15]
	v_add_u32_e32 v47, 2, v34
	v_add_u32_e32 v49, 1, v34
	v_cmp_ne_u32_e64 s[14:15], 0, v39
	v_cndmask_b32_e32 v35, v37, v38, vcc
	v_cmp_ne_u32_e32 vcc, 0, v35
	v_cndmask_b32_e64 v39, v49, v47, s[14:15]
	v_xor_b32_e32 v35, s70, v36
	v_cndmask_b32_e32 v34, v34, v39, vcc
	v_xor_b32_e32 v34, v34, v35
	v_sub_u32_e32 v34, v34, v35
	v_xor_b32_e32 v47, v32, v10
	v_xad_u32 v38, v34, v10, v46
	v_xor_b32_e32 v39, v33, v10
	v_mad_u64_u32 v[34:35], s[14:15], v47, s28, 0
	v_mul_hi_u32 v36, v47, s48
	v_mov_b32_e32 v37, v65
	v_mad_u64_u32 v[32:33], s[14:15], v39, s48, 0
	v_lshl_add_u64 v[34:35], v[36:37], 0, v[34:35]
	v_add_co_u32_e32 v32, vcc, v34, v32
	v_mad_u64_u32 v[36:37], s[14:15], v39, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v35, v33, vcc
	v_mov_b32_e32 v33, s29
	s_nop 0
	v_addc_co_u32_e32 v37, vcc, 0, v37, vcc
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[36:37]
	v_mul_lo_u32 v34, s35, v32
	v_mul_lo_u32 v35, s34, v33
	v_mad_u64_u32 v[32:33], s[14:15], s34, v32, 0
	v_add3_u32 v33, v33, v35, v34
	v_sub_u32_e32 v34, v39, v33
	v_sub_co_u32_e32 v32, vcc, v47, v32
	s_nop 1
	v_subb_co_u32_e64 v34, s[14:15], v34, v5, vcc
	v_subrev_co_u32_e64 v35, s[14:15], s34, v32
	v_subb_co_u32_e32 v33, vcc, v39, v33, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v36, s[26:27], 0, v34, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v36
	v_subb_co_u32_e64 v34, s[14:15], v34, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v37, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v35
	v_cmp_le_u32_e32 vcc, s35, v33
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v36
	s_nop 1
	v_cndmask_b32_e64 v37, v37, v47, s[26:27]
	v_subrev_co_u32_e64 v47, s[14:15], s34, v35
	s_nop 1
	v_subbrev_co_u32_e64 v34, s[14:15], 0, v34, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v37
	s_nop 1
	v_cndmask_b32_e64 v34, v36, v34, s[14:15]
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v32
	s_nop 1
	v_cndmask_b32_e64 v37, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v33
	s_nop 1
	v_cndmask_b32_e32 v36, v36, v37, vcc
	v_cmp_ne_u32_e32 vcc, 0, v36
	s_nop 1
	v_cndmask_b32_e32 v33, v33, v34, vcc
	v_cndmask_b32_e64 v34, v35, v47, s[14:15]
	v_cndmask_b32_e32 v32, v32, v34, vcc
	v_xor_b32_e32 v32, v32, v10
	v_xor_b32_e32 v33, v33, v10
	v_sub_co_u32_e32 v32, vcc, v32, v10
	v_mul_lo_u32 v34, v38, s33
	s_nop 0
	v_subb_co_u32_e32 v33, vcc, v33, v10, vcc
	v_ashrrev_i32_e32 v33, 31, v33
	v_and_b32_e32 v33, s44, v33
	v_add3_u32 v33, v33, v32, v34
	v_sub_co_u32_e32 v32, vcc, -4, v12
	v_or_b32_e32 v34, 3, v12
	s_nop 0
	v_subb_co_u32_e32 v36, vcc, -1, v13, vcc
	v_cndmask_b32_e64 v37, v13, v36, s[10:11]
	v_ashrrev_i32_e32 v38, 31, v37
	v_cndmask_b32_e64 v36, v34, v32, s[10:11]
	v_mov_b32_e32 v39, v38
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[38:39]
	v_xor_b32_e32 v39, v36, v38
	v_xor_b32_e32 v32, v37, v38
	v_mad_u64_u32 v[62:63], s[14:15], v39, s28, 0
	v_mul_hi_u32 v66, v39, s48
	v_mad_u64_u32 v[36:37], s[14:15], v32, s48, 0
	v_lshl_add_u64 v[62:63], v[66:67], 0, v[62:63]
	v_add_co_u32_e32 v36, vcc, v62, v36
	v_mad_u64_u32 v[66:67], s[14:15], v32, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v36, vcc, v63, v37, vcc
	v_mov_b32_e32 v37, s29
	s_nop 0
	v_addc_co_u32_e32 v67, vcc, 0, v67, vcc
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[66:67]
	v_mul_lo_u32 v47, s35, v36
	v_mul_lo_u32 v37, s34, v37
	v_mad_u64_u32 v[62:63], s[14:15], s34, v36, 0
	v_add3_u32 v37, v63, v37, v47
	v_sub_u32_e32 v47, v32, v37
	v_sub_co_u32_e32 v39, vcc, v39, v62
	v_mov_b32_e32 v35, v13
	s_nop 0
	v_subb_co_u32_e64 v47, s[14:15], v47, v5, vcc
	v_subrev_co_u32_e64 v49, s[14:15], s34, v39
	v_subb_co_u32_e32 v32, vcc, v32, v37, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v47
	v_cmp_le_u32_e32 vcc, s35, v32
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[10:11]
	v_cndmask_b32_e64 v51, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v49
	v_cndmask_b32_e64 v37, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v39
	v_cndmask_b32_e64 v49, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v47
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v47, v51, v49, s[14:15]
	v_add_u32_e32 v49, 2, v36
	v_add_u32_e32 v51, 1, v36
	v_cmp_ne_u32_e64 s[14:15], 0, v47
	v_cndmask_b32_e32 v32, v37, v39, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v47, v51, v49, s[14:15]
	v_xor_b32_e32 v49, v34, v10
	v_cndmask_b32_e32 v32, v36, v47, vcc
	v_xor_b32_e32 v36, s70, v38
	v_xor_b32_e32 v32, v32, v36
	v_sub_u32_e32 v32, v32, v36
	v_xor_b32_e32 v47, v35, v10
	v_mad_u64_u32 v[36:37], s[14:15], v49, s28, 0
	v_mul_hi_u32 v38, v49, s48
	v_mov_b32_e32 v39, v65
	v_mad_u64_u32 v[34:35], s[14:15], v47, s48, 0
	v_lshl_add_u64 v[36:37], v[38:39], 0, v[36:37]
	v_add_co_u32_e32 v34, vcc, v36, v34
	v_mad_u64_u32 v[38:39], s[14:15], v47, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v34, vcc, v37, v35, vcc
	v_mov_b32_e32 v35, s29
	s_nop 0
	v_addc_co_u32_e32 v39, vcc, 0, v39, vcc
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[38:39]
	v_mul_lo_u32 v36, s35, v34
	v_mul_lo_u32 v37, s34, v35
	v_mad_u64_u32 v[34:35], s[14:15], s34, v34, 0
	v_add3_u32 v35, v35, v37, v36
	v_sub_u32_e32 v36, v47, v35
	v_sub_co_u32_e32 v34, vcc, v49, v34
	v_xad_u32 v32, v32, v10, v46
	s_nop 0
	v_subb_co_u32_e64 v36, s[14:15], v36, v5, vcc
	v_subrev_co_u32_e64 v37, s[14:15], s34, v34
	v_subb_co_u32_e32 v35, vcc, v47, v35, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v38, s[26:27], 0, v36, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v38
	v_subb_co_u32_e64 v36, s[14:15], v36, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v39, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v37
	v_cmp_le_u32_e32 vcc, s35, v35
	v_mul_lo_u32 v32, v32, s33
	v_cndmask_b32_e64 v49, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v38
	s_nop 1
	v_cndmask_b32_e64 v39, v39, v49, s[26:27]
	v_subrev_co_u32_e64 v49, s[14:15], s34, v37
	s_nop 1
	v_subbrev_co_u32_e64 v36, s[14:15], 0, v36, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v39
	s_nop 1
	v_cndmask_b32_e64 v36, v38, v36, s[14:15]
	v_cndmask_b32_e64 v38, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v34
	s_nop 1
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v35
	s_nop 1
	v_cndmask_b32_e32 v38, v38, v39, vcc
	v_cmp_ne_u32_e32 vcc, 0, v38
	s_nop 1
	v_cndmask_b32_e32 v35, v35, v36, vcc
	v_cndmask_b32_e64 v36, v37, v49, s[14:15]
	v_cndmask_b32_e32 v34, v34, v36, vcc
	v_xor_b32_e32 v34, v34, v10
	v_xor_b32_e32 v35, v35, v10
	v_sub_co_u32_e32 v34, vcc, v34, v10
	v_or_b32_e32 v36, 4, v12
	s_nop 0
	v_subb_co_u32_e32 v35, vcc, v35, v10, vcc
	v_ashrrev_i32_e32 v35, 31, v35
	v_and_b32_e32 v35, s44, v35
	v_add3_u32 v35, v35, v34, v32
	v_sub_co_u32_e32 v32, vcc, -5, v12
	v_cndmask_b32_e64 v38, v36, v32, s[10:11]
	s_nop 0
	v_subb_co_u32_e32 v34, vcc, -1, v13, vcc
	v_cndmask_b32_e64 v39, v13, v34, s[10:11]
	v_ashrrev_i32_e32 v62, 31, v39
	v_mov_b32_e32 v63, v62
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[62:63]
	v_xor_b32_e32 v34, v38, v62
	v_xor_b32_e32 v32, v39, v62
	v_mad_u64_u32 v[66:67], s[14:15], v34, s28, 0
	v_mul_hi_u32 v68, v34, s48
	v_mad_u64_u32 v[38:39], s[14:15], v32, s48, 0
	v_lshl_add_u64 v[66:67], v[68:69], 0, v[66:67]
	v_add_co_u32_e32 v38, vcc, v66, v38
	v_mad_u64_u32 v[68:69], s[14:15], v32, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v67, v39, vcc
	v_mov_b32_e32 v39, s29
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[68:69]
	v_mul_lo_u32 v47, s35, v38
	v_mul_lo_u32 v39, s34, v39
	v_mad_u64_u32 v[66:67], s[14:15], s34, v38, 0
	v_add3_u32 v39, v67, v39, v47
	v_sub_u32_e32 v47, v32, v39
	v_sub_co_u32_e32 v34, vcc, v34, v66
	v_mov_b32_e32 v37, v13
	s_nop 0
	v_subb_co_u32_e64 v47, s[14:15], v47, v5, vcc
	v_subrev_co_u32_e64 v49, s[14:15], s34, v34
	v_subb_co_u32_e32 v32, vcc, v32, v39, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[14:15], 0, v47, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v47
	v_cmp_le_u32_e32 vcc, s35, v32
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[10:11]
	v_cndmask_b32_e64 v51, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v49
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v34
	v_cndmask_b32_e64 v49, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v47
	v_cndmask_b32_e64 v34, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v47, v51, v49, s[14:15]
	v_add_u32_e32 v49, 2, v38
	v_add_u32_e32 v51, 1, v38
	v_cmp_ne_u32_e64 s[14:15], 0, v47
	v_cndmask_b32_e32 v32, v39, v34, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v47, v51, v49, s[14:15]
	v_xor_b32_e32 v34, s70, v62
	v_cndmask_b32_e32 v32, v38, v47, vcc
	v_xor_b32_e32 v32, v32, v34
	v_xor_b32_e32 v47, v36, v10
	v_sub_u32_e32 v32, v32, v34
	v_xor_b32_e32 v34, v37, v10
	v_mad_u64_u32 v[38:39], s[14:15], v47, s28, 0
	v_mul_hi_u32 v62, v47, s48
	v_mov_b32_e32 v63, v65
	v_mad_u64_u32 v[36:37], s[14:15], v34, s48, 0
	v_lshl_add_u64 v[38:39], v[62:63], 0, v[38:39]
	v_add_co_u32_e32 v36, vcc, v38, v36
	v_mad_u64_u32 v[62:63], s[14:15], v34, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v36, vcc, v39, v37, vcc
	v_mov_b32_e32 v37, s29
	s_nop 0
	v_addc_co_u32_e32 v63, vcc, 0, v63, vcc
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[62:63]
	v_mul_lo_u32 v38, s35, v36
	v_mul_lo_u32 v39, s34, v37
	v_mad_u64_u32 v[36:37], s[14:15], s34, v36, 0
	v_add3_u32 v37, v37, v39, v38
	v_sub_u32_e32 v38, v34, v37
	v_sub_co_u32_e32 v36, vcc, v47, v36
	v_xad_u32 v32, v32, v10, v46
	s_nop 0
	v_subb_co_u32_e64 v38, s[14:15], v38, v5, vcc
	v_subrev_co_u32_e64 v39, s[14:15], s34, v36
	v_subb_co_u32_e32 v34, vcc, v34, v37, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[26:27], 0, v38, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v47
	v_subb_co_u32_e64 v38, s[14:15], v38, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v39
	v_cmp_le_u32_e32 vcc, s35, v34
	v_mul_lo_u32 v32, v32, s33
	v_cndmask_b32_e64 v51, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v47
	v_cndmask_b32_e64 v37, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v36
	v_cndmask_b32_e64 v49, v49, v51, s[26:27]
	v_subrev_co_u32_e64 v51, s[14:15], s34, v39
	v_mov_b32_e32 v69, v65
	s_nop 0
	v_subbrev_co_u32_e64 v38, s[14:15], 0, v38, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v49
	s_nop 1
	v_cndmask_b32_e64 v38, v47, v38, s[14:15]
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v34
	s_nop 1
	v_cndmask_b32_e32 v37, v37, v47, vcc
	v_cmp_ne_u32_e32 vcc, 0, v37
	v_cndmask_b32_e64 v37, v39, v51, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v36, v36, v37, vcc
	v_cndmask_b32_e32 v34, v34, v38, vcc
	v_xor_b32_e32 v36, v36, v10
	v_xor_b32_e32 v34, v34, v10
	v_sub_co_u32_e32 v36, vcc, v36, v10
	v_mov_b32_e32 v37, v13
	s_nop 0
	v_subb_co_u32_e32 v34, vcc, v34, v10, vcc
	v_ashrrev_i32_e32 v34, 31, v34
	v_and_b32_e32 v34, s44, v34
	v_add3_u32 v47, v34, v36, v32
	v_sub_co_u32_e32 v32, vcc, -6, v12
	v_or_b32_e32 v36, 5, v12
	s_nop 0
	v_subb_co_u32_e32 v34, vcc, -1, v13, vcc
	v_cndmask_b32_e64 v39, v13, v34, s[10:11]
	v_ashrrev_i32_e32 v62, 31, v39
	v_cndmask_b32_e64 v38, v36, v32, s[10:11]
	v_mov_b32_e32 v63, v62
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[62:63]
	v_xor_b32_e32 v34, v38, v62
	v_xor_b32_e32 v32, v39, v62
	v_mad_u64_u32 v[66:67], s[14:15], v34, s28, 0
	v_mul_hi_u32 v68, v34, s48
	v_mad_u64_u32 v[38:39], s[14:15], v32, s48, 0
	v_lshl_add_u64 v[66:67], v[68:69], 0, v[66:67]
	v_add_co_u32_e32 v38, vcc, v66, v38
	v_mad_u64_u32 v[68:69], s[14:15], v32, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v67, v39, vcc
	v_mov_b32_e32 v39, s29
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[68:69]
	v_mul_lo_u32 v49, s35, v38
	v_mul_lo_u32 v39, s34, v39
	v_mad_u64_u32 v[66:67], s[14:15], s34, v38, 0
	v_add3_u32 v39, v67, v39, v49
	v_sub_u32_e32 v49, v32, v39
	v_sub_co_u32_e32 v34, vcc, v34, v66
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[10:11]
	s_nop 0
	v_subb_co_u32_e64 v49, s[14:15], v49, v5, vcc
	v_subrev_co_u32_e64 v51, s[14:15], s34, v34
	v_subb_co_u32_e32 v32, vcc, v32, v39, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[14:15], 0, v49, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v49
	v_cmp_le_u32_e32 vcc, s35, v32
	v_mov_b32_e32 v63, v65
	v_cndmask_b32_e64 v53, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v51
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v34
	v_cndmask_b32_e64 v51, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v49
	v_cndmask_b32_e64 v34, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v49, v53, v51, s[14:15]
	v_add_u32_e32 v51, 2, v38
	v_add_u32_e32 v53, 1, v38
	v_cmp_ne_u32_e64 s[14:15], 0, v49
	v_cndmask_b32_e32 v32, v39, v34, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v49, v53, v51, s[14:15]
	v_xor_b32_e32 v34, s70, v62
	v_cndmask_b32_e32 v32, v38, v49, vcc
	v_xor_b32_e32 v32, v32, v34
	v_xor_b32_e32 v49, v36, v10
	v_sub_u32_e32 v32, v32, v34
	v_xor_b32_e32 v34, v37, v10
	v_mad_u64_u32 v[38:39], s[14:15], v49, s28, 0
	v_mul_hi_u32 v62, v49, s48
	v_mad_u64_u32 v[36:37], s[14:15], v34, s48, 0
	v_lshl_add_u64 v[38:39], v[62:63], 0, v[38:39]
	v_add_co_u32_e32 v36, vcc, v38, v36
	v_mad_u64_u32 v[62:63], s[14:15], v34, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v36, vcc, v39, v37, vcc
	v_mov_b32_e32 v37, s29
	s_nop 0
	v_addc_co_u32_e32 v63, vcc, 0, v63, vcc
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[62:63]
	v_mul_lo_u32 v38, s35, v36
	v_mul_lo_u32 v39, s34, v37
	v_mad_u64_u32 v[36:37], s[14:15], s34, v36, 0
	v_add3_u32 v37, v37, v39, v38
	v_sub_u32_e32 v38, v34, v37
	v_sub_co_u32_e32 v36, vcc, v49, v36
	v_xad_u32 v32, v32, v10, v46
	s_nop 0
	v_subb_co_u32_e64 v38, s[14:15], v38, v5, vcc
	v_subrev_co_u32_e64 v39, s[14:15], s34, v36
	v_subb_co_u32_e32 v34, vcc, v34, v37, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[26:27], 0, v38, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v49
	v_subb_co_u32_e64 v38, s[14:15], v38, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v51, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v39
	v_cmp_le_u32_e32 vcc, s35, v34
	v_mul_lo_u32 v32, v32, s33
	v_cndmask_b32_e64 v53, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v49
	v_cndmask_b32_e64 v37, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v36
	v_cndmask_b32_e64 v51, v51, v53, s[26:27]
	v_subrev_co_u32_e64 v53, s[14:15], s34, v39
	v_mov_b32_e32 v69, v65
	s_nop 0
	v_subbrev_co_u32_e64 v38, s[14:15], 0, v38, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v51
	s_nop 1
	v_cndmask_b32_e64 v38, v49, v38, s[14:15]
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v34
	s_nop 1
	v_cndmask_b32_e32 v37, v37, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v37
	v_cndmask_b32_e64 v37, v39, v53, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v36, v36, v37, vcc
	v_cndmask_b32_e32 v34, v34, v38, vcc
	v_xor_b32_e32 v36, v36, v10
	v_xor_b32_e32 v34, v34, v10
	v_sub_co_u32_e32 v36, vcc, v36, v10
	v_mov_b32_e32 v37, v13
	s_nop 0
	v_subb_co_u32_e32 v34, vcc, v34, v10, vcc
	v_ashrrev_i32_e32 v34, 31, v34
	v_and_b32_e32 v34, s44, v34
	v_add3_u32 v49, v34, v36, v32
	v_sub_co_u32_e32 v32, vcc, -7, v12
	v_or_b32_e32 v36, 6, v12
	s_nop 0
	v_subb_co_u32_e32 v34, vcc, -1, v13, vcc
	v_cndmask_b32_e64 v39, v13, v34, s[10:11]
	v_ashrrev_i32_e32 v62, 31, v39
	v_cndmask_b32_e64 v38, v36, v32, s[10:11]
	v_mov_b32_e32 v63, v62
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[62:63]
	v_xor_b32_e32 v34, v38, v62
	v_xor_b32_e32 v32, v39, v62
	v_mad_u64_u32 v[66:67], s[14:15], v34, s28, 0
	v_mul_hi_u32 v68, v34, s48
	v_mad_u64_u32 v[38:39], s[14:15], v32, s48, 0
	v_lshl_add_u64 v[66:67], v[68:69], 0, v[66:67]
	v_add_co_u32_e32 v38, vcc, v66, v38
	v_mad_u64_u32 v[68:69], s[14:15], v32, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v67, v39, vcc
	v_mov_b32_e32 v39, s29
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[68:69]
	v_mul_lo_u32 v51, s35, v38
	v_mul_lo_u32 v39, s34, v39
	v_mad_u64_u32 v[66:67], s[14:15], s34, v38, 0
	v_add3_u32 v39, v67, v39, v51
	v_sub_u32_e32 v51, v32, v39
	v_sub_co_u32_e32 v34, vcc, v34, v66
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[10:11]
	s_nop 0
	v_subb_co_u32_e64 v51, s[14:15], v51, v5, vcc
	v_subrev_co_u32_e64 v53, s[14:15], s34, v34
	v_subb_co_u32_e32 v32, vcc, v32, v39, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v51, s[14:15], 0, v51, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v51
	v_cmp_le_u32_e32 vcc, s35, v32
	v_mov_b32_e32 v63, v65
	v_cndmask_b32_e64 v55, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v53
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v34
	v_cndmask_b32_e64 v53, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v51
	v_cndmask_b32_e64 v34, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v51, v55, v53, s[14:15]
	v_add_u32_e32 v53, 2, v38
	v_add_u32_e32 v55, 1, v38
	v_cmp_ne_u32_e64 s[14:15], 0, v51
	v_cndmask_b32_e32 v32, v39, v34, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v51, v55, v53, s[14:15]
	v_xor_b32_e32 v34, s70, v62
	v_cndmask_b32_e32 v32, v38, v51, vcc
	v_xor_b32_e32 v32, v32, v34
	v_xor_b32_e32 v51, v36, v10
	v_sub_u32_e32 v32, v32, v34
	v_xor_b32_e32 v34, v37, v10
	v_mad_u64_u32 v[38:39], s[14:15], v51, s28, 0
	v_mul_hi_u32 v62, v51, s48
	v_mad_u64_u32 v[36:37], s[14:15], v34, s48, 0
	v_lshl_add_u64 v[38:39], v[62:63], 0, v[38:39]
	v_add_co_u32_e32 v36, vcc, v38, v36
	v_mad_u64_u32 v[62:63], s[14:15], v34, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v36, vcc, v39, v37, vcc
	v_mov_b32_e32 v37, s29
	s_nop 0
	v_addc_co_u32_e32 v63, vcc, 0, v63, vcc
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[62:63]
	v_mul_lo_u32 v38, s35, v36
	v_mul_lo_u32 v39, s34, v37
	v_mad_u64_u32 v[36:37], s[14:15], s34, v36, 0
	v_add3_u32 v37, v37, v39, v38
	v_sub_u32_e32 v38, v34, v37
	v_sub_co_u32_e32 v36, vcc, v51, v36
	v_xad_u32 v32, v32, v10, v46
	s_nop 0
	v_subb_co_u32_e64 v38, s[14:15], v38, v5, vcc
	v_subrev_co_u32_e64 v39, s[14:15], s34, v36
	v_subb_co_u32_e32 v34, vcc, v34, v37, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v51, s[26:27], 0, v38, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v51
	v_subb_co_u32_e64 v38, s[14:15], v38, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v53, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v39
	v_cmp_le_u32_e32 vcc, s35, v34
	v_mul_lo_u32 v32, v32, s33
	v_cndmask_b32_e64 v55, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v51
	v_cndmask_b32_e64 v37, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v36
	v_cndmask_b32_e64 v53, v53, v55, s[26:27]
	v_subrev_co_u32_e64 v55, s[14:15], s34, v39
	v_mov_b32_e32 v69, v65
	s_nop 0
	v_subbrev_co_u32_e64 v38, s[14:15], 0, v38, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v53
	s_nop 1
	v_cndmask_b32_e64 v38, v51, v38, s[14:15]
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v34
	s_nop 1
	v_cndmask_b32_e32 v37, v37, v51, vcc
	v_cmp_ne_u32_e32 vcc, 0, v37
	v_cndmask_b32_e64 v37, v39, v55, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v36, v36, v37, vcc
	v_cndmask_b32_e32 v34, v34, v38, vcc
	v_xor_b32_e32 v36, v36, v10
	v_xor_b32_e32 v34, v34, v10
	v_sub_co_u32_e32 v36, vcc, v36, v10
	v_mov_b32_e32 v37, v13
	s_nop 0
	v_subb_co_u32_e32 v34, vcc, v34, v10, vcc
	v_ashrrev_i32_e32 v34, 31, v34
	v_and_b32_e32 v34, s44, v34
	v_add3_u32 v51, v34, v36, v32
	v_sub_co_u32_e32 v32, vcc, -8, v12
	v_or_b32_e32 v36, 7, v12
	s_nop 0
	v_subb_co_u32_e32 v34, vcc, -1, v13, vcc
	v_cndmask_b32_e64 v39, v13, v34, s[10:11]
	v_ashrrev_i32_e32 v62, 31, v39
	v_cndmask_b32_e64 v38, v36, v32, s[10:11]
	v_mov_b32_e32 v63, v62
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[62:63]
	v_xor_b32_e32 v34, v38, v62
	v_xor_b32_e32 v32, v39, v62
	v_mad_u64_u32 v[66:67], s[14:15], v34, s28, 0
	v_mul_hi_u32 v68, v34, s48
	v_mad_u64_u32 v[38:39], s[14:15], v32, s48, 0
	v_lshl_add_u64 v[66:67], v[68:69], 0, v[66:67]
	v_add_co_u32_e32 v38, vcc, v66, v38
	v_mad_u64_u32 v[68:69], s[14:15], v32, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v38, vcc, v67, v39, vcc
	v_mov_b32_e32 v39, s29
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[68:69]
	v_mul_lo_u32 v53, s35, v38
	v_mul_lo_u32 v39, s34, v39
	v_mad_u64_u32 v[66:67], s[14:15], s34, v38, 0
	v_add3_u32 v39, v67, v39, v53
	v_sub_u32_e32 v53, v32, v39
	v_sub_co_u32_e32 v34, vcc, v34, v66
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[10:11]
	s_nop 0
	v_subb_co_u32_e64 v53, s[14:15], v53, v5, vcc
	v_subrev_co_u32_e64 v55, s[14:15], s34, v34
	v_subb_co_u32_e32 v32, vcc, v32, v39, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v53, s[14:15], 0, v53, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v53
	v_cmp_le_u32_e32 vcc, s35, v32
	v_mov_b32_e32 v63, v65
	v_cndmask_b32_e64 v57, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v55
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v34
	v_cndmask_b32_e64 v55, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v53
	v_cndmask_b32_e64 v34, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v53, v57, v55, s[14:15]
	v_add_u32_e32 v55, 2, v38
	v_add_u32_e32 v57, 1, v38
	v_cmp_ne_u32_e64 s[14:15], 0, v53
	v_cndmask_b32_e32 v32, v39, v34, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v53, v57, v55, s[14:15]
	v_xor_b32_e32 v34, s70, v62
	v_cndmask_b32_e32 v32, v38, v53, vcc
	v_xor_b32_e32 v32, v32, v34
	v_xor_b32_e32 v53, v36, v10
	v_sub_u32_e32 v32, v32, v34
	v_xor_b32_e32 v34, v37, v10
	v_mad_u64_u32 v[38:39], s[14:15], v53, s28, 0
	v_mul_hi_u32 v62, v53, s48
	v_mad_u64_u32 v[36:37], s[14:15], v34, s48, 0
	v_lshl_add_u64 v[38:39], v[62:63], 0, v[38:39]
	v_add_co_u32_e32 v36, vcc, v38, v36
	v_mad_u64_u32 v[62:63], s[14:15], v34, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v36, vcc, v39, v37, vcc
	v_mov_b32_e32 v37, s29
	s_nop 0
	v_addc_co_u32_e32 v63, vcc, 0, v63, vcc
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[62:63]
	v_mul_lo_u32 v38, s35, v36
	v_mul_lo_u32 v39, s34, v37
	v_mad_u64_u32 v[36:37], s[14:15], s34, v36, 0
	v_add3_u32 v37, v37, v39, v38
	v_sub_u32_e32 v38, v34, v37
	v_sub_co_u32_e32 v36, vcc, v53, v36
	v_xad_u32 v32, v32, v10, v46
	s_nop 0
	v_subb_co_u32_e64 v38, s[14:15], v38, v5, vcc
	v_subrev_co_u32_e64 v39, s[14:15], s34, v36
	v_subb_co_u32_e32 v34, vcc, v34, v37, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v53, s[26:27], 0, v38, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v53
	v_subb_co_u32_e64 v38, s[14:15], v38, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v55, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v39
	v_cmp_le_u32_e32 vcc, s35, v34
	v_mul_lo_u32 v32, v32, s33
	v_cndmask_b32_e64 v57, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v53
	v_cndmask_b32_e64 v37, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v36
	v_cndmask_b32_e64 v55, v55, v57, s[26:27]
	v_subrev_co_u32_e64 v57, s[14:15], s34, v39
	s_nop 1
	v_subbrev_co_u32_e64 v38, s[14:15], 0, v38, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v55
	s_nop 1
	v_cndmask_b32_e64 v38, v53, v38, s[14:15]
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v34
	s_nop 1
	v_cndmask_b32_e32 v37, v37, v53, vcc
	v_cmp_ne_u32_e32 vcc, 0, v37
	v_cndmask_b32_e64 v37, v39, v57, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v36, v36, v37, vcc
	v_cndmask_b32_e32 v34, v34, v38, vcc
	v_xor_b32_e32 v36, v36, v10
	v_xor_b32_e32 v34, v34, v10
	v_sub_co_u32_e32 v36, vcc, v36, v10
	s_nop 1
	v_subb_co_u32_e32 v34, vcc, v34, v10, vcc
	v_ashrrev_i32_e32 v34, 31, v34
	v_and_b32_e32 v34, s44, v34
	v_add3_u32 v39, v34, v36, v32
	buffer_load_ubyte v32, v31, s[36:39], 0 offen
	buffer_load_ubyte v34, v30, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v33, v33, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v35, v35, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v36, v47, s[36:39], 0 offen
	buffer_load_ubyte v38, v49, s[36:39], 0 offen
	buffer_load_ubyte v37, v51, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v39, v39, s[36:39], 0 offen
	v_sub_co_u32_e32 v47, vcc, -9, v12
	v_or_b32_e32 v30, 8, v12
	s_nop 0
	v_subb_co_u32_e32 v49, vcc, -1, v13, vcc
	v_cndmask_b32_e64 v63, v13, v49, s[10:11]
	v_ashrrev_i32_e32 v66, 31, v63
	v_cndmask_b32_e64 v62, v30, v47, s[10:11]
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[62:63], v[62:63], 0, v[66:67]
	v_xor_b32_e32 v49, v62, v66
	v_xor_b32_e32 v47, v63, v66
	v_mad_u64_u32 v[68:69], s[14:15], v49, s28, 0
	v_mul_hi_u32 v72, v49, s48
	v_mad_u64_u32 v[62:63], s[14:15], v47, s48, 0
	v_lshl_add_u64 v[68:69], v[72:73], 0, v[68:69]
	v_add_co_u32_e32 v51, vcc, v68, v62
	v_mad_u64_u32 v[72:73], s[14:15], v47, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v62, vcc, v69, v63, vcc
	v_mov_b32_e32 v63, s29
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[62:63], v[62:63], 0, v[72:73]
	v_mul_lo_u32 v51, s35, v62
	v_mul_lo_u32 v53, s34, v63
	v_mad_u64_u32 v[68:69], s[14:15], s34, v62, 0
	v_add3_u32 v51, v69, v53, v51
	v_sub_u32_e32 v53, v47, v51
	v_sub_co_u32_e32 v49, vcc, v49, v68
	v_mov_b32_e32 v31, v13
	s_nop 0
	v_subb_co_u32_e64 v53, s[14:15], v53, v5, vcc
	v_subrev_co_u32_e64 v55, s[14:15], s34, v49
	v_subb_co_u32_e32 v47, vcc, v47, v51, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v53, s[14:15], 0, v53, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v53
	v_cmp_le_u32_e32 vcc, s35, v47
	v_lshl_add_u64 v[30:31], v[30:31], 0, v[10:11]
	v_cndmask_b32_e64 v57, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v55
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v49
	v_cndmask_b32_e64 v55, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v53
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v47
	v_cndmask_b32_e64 v53, v57, v55, s[14:15]
	v_add_u32_e32 v55, 2, v62
	v_add_u32_e32 v57, 1, v62
	v_cmp_ne_u32_e64 s[14:15], 0, v53
	v_cndmask_b32_e32 v47, v51, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v47
	v_cndmask_b32_e64 v53, v57, v55, s[14:15]
	v_xor_b32_e32 v49, s70, v66
	v_cndmask_b32_e32 v47, v62, v53, vcc
	v_xor_b32_e32 v47, v47, v49
	v_xor_b32_e32 v51, v30, v10
	v_sub_u32_e32 v47, v47, v49
	v_xor_b32_e32 v49, v31, v10
	v_mad_u64_u32 v[62:63], s[14:15], v51, s28, 0
	v_mul_hi_u32 v66, v51, s48
	v_mov_b32_e32 v67, v65
	v_mad_u64_u32 v[30:31], s[14:15], v49, s48, 0
	v_lshl_add_u64 v[62:63], v[66:67], 0, v[62:63]
	v_add_co_u32_e32 v30, vcc, v62, v30
	v_mad_u64_u32 v[66:67], s[14:15], v49, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v30, vcc, v63, v31, vcc
	v_mov_b32_e32 v31, s29
	s_nop 0
	v_addc_co_u32_e32 v67, vcc, 0, v67, vcc
	v_lshl_add_u64 v[30:31], v[30:31], 0, v[66:67]
	v_mul_lo_u32 v53, s35, v30
	v_mul_lo_u32 v55, s34, v31
	v_mad_u64_u32 v[30:31], s[14:15], s34, v30, 0
	v_add3_u32 v31, v31, v55, v53
	v_sub_u32_e32 v53, v49, v31
	v_sub_co_u32_e32 v30, vcc, v51, v30
	v_xad_u32 v47, v47, v10, v46
	s_nop 0
	v_subb_co_u32_e64 v51, s[14:15], v53, v5, vcc
	v_subrev_co_u32_e64 v53, s[14:15], s34, v30
	v_subb_co_u32_e32 v31, vcc, v49, v31, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v55, s[26:27], 0, v51, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v55
	v_subb_co_u32_e64 v51, s[14:15], v51, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v53
	v_cmp_le_u32_e32 vcc, s35, v31
	v_mul_lo_u32 v47, v47, s33
	v_cndmask_b32_e64 v59, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v55
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v30
	v_cndmask_b32_e64 v57, v57, v59, s[26:27]
	v_subrev_co_u32_e64 v59, s[14:15], s34, v53
	v_or_b32_e32 v62, 9, v12
	s_nop 0
	v_subbrev_co_u32_e64 v51, s[14:15], 0, v51, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v57
	v_mov_b32_e32 v63, v13
	s_nop 0
	v_cndmask_b32_e64 v51, v55, v51, s[14:15]
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v31
	s_nop 1
	v_cndmask_b32_e32 v49, v49, v55, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	v_cndmask_b32_e64 v49, v53, v59, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v30, v30, v49, vcc
	v_cndmask_b32_e32 v31, v31, v51, vcc
	v_xor_b32_e32 v30, v30, v10
	v_xor_b32_e32 v31, v31, v10
	v_sub_co_u32_e32 v30, vcc, v30, v10
	s_nop 1
	v_subb_co_u32_e32 v31, vcc, v31, v10, vcc
	v_ashrrev_i32_e32 v31, 31, v31
	v_and_b32_e32 v31, s44, v31
	v_add3_u32 v30, v31, v30, v47
	v_sub_co_u32_e32 v31, vcc, -10, v12
	v_cndmask_b32_e64 v66, v62, v31, s[10:11]
	s_nop 0
	v_subb_co_u32_e32 v47, vcc, -1, v13, vcc
	v_cndmask_b32_e64 v67, v13, v47, s[10:11]
	v_ashrrev_i32_e32 v68, 31, v67
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v47, v66, v68
	v_xor_b32_e32 v31, v67, v68
	v_mad_u64_u32 v[72:73], s[14:15], v47, s28, 0
	v_mul_hi_u32 v76, v47, s48
	v_mad_u64_u32 v[66:67], s[14:15], v31, s48, 0
	v_lshl_add_u64 v[72:73], v[76:77], 0, v[72:73]
	v_add_co_u32_e32 v49, vcc, v72, v66
	v_mad_u64_u32 v[76:77], s[14:15], v31, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v73, v67, vcc
	v_mov_b32_e32 v67, s29
	s_nop 0
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[76:77]
	v_mul_lo_u32 v49, s35, v66
	v_mul_lo_u32 v51, s34, v67
	v_mad_u64_u32 v[72:73], s[14:15], s34, v66, 0
	v_add3_u32 v49, v73, v51, v49
	v_sub_u32_e32 v51, v31, v49
	v_sub_co_u32_e32 v47, vcc, v47, v72
	v_lshl_add_u64 v[62:63], v[62:63], 0, v[10:11]
	s_nop 0
	v_subb_co_u32_e64 v51, s[14:15], v51, v5, vcc
	v_subrev_co_u32_e64 v53, s[14:15], s34, v47
	v_subb_co_u32_e32 v31, vcc, v31, v49, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v51, s[14:15], 0, v51, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v51
	v_cmp_le_u32_e32 vcc, s35, v31
	v_mov_b32_e32 v69, v65
	v_cndmask_b32_e64 v55, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v53
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v47
	v_cndmask_b32_e64 v53, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v51
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v31
	v_cndmask_b32_e64 v51, v55, v53, s[14:15]
	v_add_u32_e32 v53, 2, v66
	v_add_u32_e32 v55, 1, v66
	v_cmp_ne_u32_e64 s[14:15], 0, v51
	v_cndmask_b32_e32 v31, v49, v47, vcc
	v_cmp_ne_u32_e32 vcc, 0, v31
	v_cndmask_b32_e64 v51, v55, v53, s[14:15]
	v_xor_b32_e32 v47, s70, v68
	v_cndmask_b32_e32 v31, v66, v51, vcc
	v_xor_b32_e32 v31, v31, v47
	v_xor_b32_e32 v49, v62, v10
	v_sub_u32_e32 v31, v31, v47
	v_xor_b32_e32 v47, v63, v10
	v_mad_u64_u32 v[66:67], s[14:15], v49, s28, 0
	v_mul_hi_u32 v68, v49, s48
	v_mad_u64_u32 v[62:63], s[14:15], v47, s48, 0
	v_lshl_add_u64 v[66:67], v[68:69], 0, v[66:67]
	v_add_co_u32_e32 v51, vcc, v66, v62
	v_mad_u64_u32 v[68:69], s[14:15], v47, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v62, vcc, v67, v63, vcc
	v_mov_b32_e32 v63, s29
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[62:63], v[62:63], 0, v[68:69]
	v_mul_lo_u32 v51, s35, v62
	v_mul_lo_u32 v53, s34, v63
	v_mad_u64_u32 v[62:63], s[14:15], s34, v62, 0
	v_add3_u32 v51, v63, v53, v51
	v_sub_u32_e32 v53, v47, v51
	v_sub_co_u32_e32 v49, vcc, v49, v62
	v_xad_u32 v31, v31, v10, v46
	s_nop 0
	v_subb_co_u32_e64 v53, s[14:15], v53, v5, vcc
	v_subrev_co_u32_e64 v55, s[14:15], s34, v49
	v_subb_co_u32_e32 v47, vcc, v47, v51, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v57, s[26:27], 0, v53, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v57
	v_subb_co_u32_e64 v53, s[14:15], v53, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v55
	v_cmp_le_u32_e32 vcc, s35, v47
	v_mul_lo_u32 v31, v31, s33
	v_cndmask_b32_e64 v62, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v57
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v49
	v_cndmask_b32_e64 v59, v59, v62, s[26:27]
	v_subrev_co_u32_e64 v62, s[14:15], s34, v55
	v_mov_b32_e32 v77, v65
	s_nop 0
	v_subbrev_co_u32_e64 v53, s[14:15], 0, v53, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v59
	v_mov_b32_e32 v63, v13
	s_nop 0
	v_cndmask_b32_e64 v53, v57, v53, s[14:15]
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v47
	s_nop 1
	v_cndmask_b32_e32 v51, v51, v57, vcc
	v_cmp_ne_u32_e32 vcc, 0, v51
	v_cndmask_b32_e64 v51, v55, v62, s[14:15]
	v_or_b32_e32 v62, 10, v12
	v_cndmask_b32_e32 v49, v49, v51, vcc
	v_cndmask_b32_e32 v47, v47, v53, vcc
	v_xor_b32_e32 v49, v49, v10
	v_xor_b32_e32 v47, v47, v10
	v_sub_co_u32_e32 v49, vcc, v49, v10
	s_nop 1
	v_subb_co_u32_e32 v47, vcc, v47, v10, vcc
	v_ashrrev_i32_e32 v47, 31, v47
	v_and_b32_e32 v47, s44, v47
	v_add3_u32 v31, v47, v49, v31
	v_sub_co_u32_e32 v47, vcc, -11, v12
	v_cndmask_b32_e64 v66, v62, v47, s[10:11]
	s_nop 0
	v_subb_co_u32_e32 v49, vcc, -1, v13, vcc
	v_cndmask_b32_e64 v67, v13, v49, s[10:11]
	v_ashrrev_i32_e32 v68, 31, v67
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v49, v66, v68
	v_xor_b32_e32 v47, v67, v68
	v_mad_u64_u32 v[72:73], s[14:15], v49, s28, 0
	v_mul_hi_u32 v76, v49, s48
	v_mad_u64_u32 v[66:67], s[14:15], v47, s48, 0
	v_lshl_add_u64 v[72:73], v[76:77], 0, v[72:73]
	v_add_co_u32_e32 v51, vcc, v72, v66
	v_mad_u64_u32 v[76:77], s[14:15], v47, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v73, v67, vcc
	v_mov_b32_e32 v67, s29
	s_nop 0
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[76:77]
	v_mul_lo_u32 v51, s35, v66
	v_mul_lo_u32 v53, s34, v67
	v_mad_u64_u32 v[72:73], s[14:15], s34, v66, 0
	v_add3_u32 v51, v73, v53, v51
	v_sub_u32_e32 v53, v47, v51
	v_sub_co_u32_e32 v49, vcc, v49, v72
	v_lshl_add_u64 v[62:63], v[62:63], 0, v[10:11]
	s_nop 0
	v_subb_co_u32_e64 v53, s[14:15], v53, v5, vcc
	v_subrev_co_u32_e64 v55, s[14:15], s34, v49
	v_subb_co_u32_e32 v47, vcc, v47, v51, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v53, s[14:15], 0, v53, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v53
	v_cmp_le_u32_e32 vcc, s35, v47
	v_mov_b32_e32 v69, v65
	v_cndmask_b32_e64 v57, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v55
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v49
	v_cndmask_b32_e64 v55, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v53
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v47
	v_cndmask_b32_e64 v53, v57, v55, s[14:15]
	v_add_u32_e32 v55, 2, v66
	v_add_u32_e32 v57, 1, v66
	v_cmp_ne_u32_e64 s[14:15], 0, v53
	v_cndmask_b32_e32 v47, v51, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v47
	v_cndmask_b32_e64 v53, v57, v55, s[14:15]
	v_xor_b32_e32 v49, s70, v68
	v_cndmask_b32_e32 v47, v66, v53, vcc
	v_xor_b32_e32 v47, v47, v49
	v_xor_b32_e32 v51, v62, v10
	v_sub_u32_e32 v47, v47, v49
	v_xor_b32_e32 v49, v63, v10
	v_mad_u64_u32 v[66:67], s[14:15], v51, s28, 0
	v_mul_hi_u32 v68, v51, s48
	v_mad_u64_u32 v[62:63], s[14:15], v49, s48, 0
	v_lshl_add_u64 v[66:67], v[68:69], 0, v[66:67]
	v_add_co_u32_e32 v53, vcc, v66, v62
	v_mad_u64_u32 v[68:69], s[14:15], v49, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v62, vcc, v67, v63, vcc
	v_mov_b32_e32 v63, s29
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[62:63], v[62:63], 0, v[68:69]
	v_mul_lo_u32 v53, s35, v62
	v_mul_lo_u32 v55, s34, v63
	v_mad_u64_u32 v[62:63], s[14:15], s34, v62, 0
	v_add3_u32 v53, v63, v55, v53
	v_sub_u32_e32 v55, v49, v53
	v_sub_co_u32_e32 v51, vcc, v51, v62
	v_xad_u32 v47, v47, v10, v46
	s_nop 0
	v_subb_co_u32_e64 v55, s[14:15], v55, v5, vcc
	v_subrev_co_u32_e64 v57, s[14:15], s34, v51
	v_subb_co_u32_e32 v49, vcc, v49, v53, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v59, s[26:27], 0, v55, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v59
	v_subb_co_u32_e64 v55, s[14:15], v55, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v62, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v57
	v_cmp_le_u32_e32 vcc, s35, v49
	v_mul_lo_u32 v47, v47, s33
	v_cndmask_b32_e64 v63, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v59
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v51
	v_cndmask_b32_e64 v62, v62, v63, s[26:27]
	v_subrev_co_u32_e64 v63, s[14:15], s34, v57
	v_mov_b32_e32 v77, v65
	s_nop 0
	v_subbrev_co_u32_e64 v55, s[14:15], 0, v55, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v62
	v_or_b32_e32 v62, 11, v12
	s_nop 0
	v_cndmask_b32_e64 v55, v59, v55, s[14:15]
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v49
	s_nop 1
	v_cndmask_b32_e32 v53, v53, v59, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_cndmask_b32_e64 v53, v57, v63, s[14:15]
	v_mov_b32_e32 v63, v13
	v_cndmask_b32_e32 v51, v51, v53, vcc
	v_cndmask_b32_e32 v49, v49, v55, vcc
	v_xor_b32_e32 v51, v51, v10
	v_xor_b32_e32 v49, v49, v10
	v_sub_co_u32_e32 v51, vcc, v51, v10
	s_nop 1
	v_subb_co_u32_e32 v49, vcc, v49, v10, vcc
	v_ashrrev_i32_e32 v49, 31, v49
	v_and_b32_e32 v49, s44, v49
	v_add3_u32 v47, v49, v51, v47
	v_sub_co_u32_e32 v49, vcc, -12, v12
	v_cndmask_b32_e64 v66, v62, v49, s[10:11]
	s_nop 0
	v_subb_co_u32_e32 v51, vcc, -1, v13, vcc
	v_cndmask_b32_e64 v67, v13, v51, s[10:11]
	v_ashrrev_i32_e32 v68, 31, v67
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v51, v66, v68
	v_xor_b32_e32 v49, v67, v68
	v_mad_u64_u32 v[72:73], s[14:15], v51, s28, 0
	v_mul_hi_u32 v76, v51, s48
	v_mad_u64_u32 v[66:67], s[14:15], v49, s48, 0
	v_lshl_add_u64 v[72:73], v[76:77], 0, v[72:73]
	v_add_co_u32_e32 v53, vcc, v72, v66
	v_mad_u64_u32 v[76:77], s[14:15], v49, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v73, v67, vcc
	v_mov_b32_e32 v67, s29
	s_nop 0
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[76:77]
	v_mul_lo_u32 v53, s35, v66
	v_mul_lo_u32 v55, s34, v67
	v_mad_u64_u32 v[72:73], s[14:15], s34, v66, 0
	v_add3_u32 v53, v73, v55, v53
	v_sub_u32_e32 v55, v49, v53
	v_sub_co_u32_e32 v51, vcc, v51, v72
	v_lshl_add_u64 v[62:63], v[62:63], 0, v[10:11]
	s_nop 0
	v_subb_co_u32_e64 v55, s[14:15], v55, v5, vcc
	v_subrev_co_u32_e64 v57, s[14:15], s34, v51
	v_subb_co_u32_e32 v49, vcc, v49, v53, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v55, s[14:15], 0, v55, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v55
	v_cmp_le_u32_e32 vcc, s35, v49
	v_mov_b32_e32 v69, v65
	v_cndmask_b32_e64 v59, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v57
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v51
	v_cndmask_b32_e64 v57, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v55
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v49
	v_cndmask_b32_e64 v55, v59, v57, s[14:15]
	v_add_u32_e32 v57, 2, v66
	v_add_u32_e32 v59, 1, v66
	v_cmp_ne_u32_e64 s[14:15], 0, v55
	v_cndmask_b32_e32 v49, v53, v51, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	v_cndmask_b32_e64 v55, v59, v57, s[14:15]
	v_xor_b32_e32 v51, s70, v68
	v_cndmask_b32_e32 v49, v66, v55, vcc
	v_xor_b32_e32 v49, v49, v51
	v_xor_b32_e32 v53, v62, v10
	v_sub_u32_e32 v49, v49, v51
	v_xor_b32_e32 v51, v63, v10
	v_mad_u64_u32 v[66:67], s[14:15], v53, s28, 0
	v_mul_hi_u32 v68, v53, s48
	v_mad_u64_u32 v[62:63], s[14:15], v51, s48, 0
	v_lshl_add_u64 v[66:67], v[68:69], 0, v[66:67]
	v_add_co_u32_e32 v55, vcc, v66, v62
	v_mad_u64_u32 v[68:69], s[14:15], v51, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v62, vcc, v67, v63, vcc
	v_mov_b32_e32 v63, s29
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[62:63], v[62:63], 0, v[68:69]
	v_mul_lo_u32 v55, s35, v62
	v_mul_lo_u32 v57, s34, v63
	v_mad_u64_u32 v[62:63], s[14:15], s34, v62, 0
	v_add3_u32 v55, v63, v57, v55
	v_sub_u32_e32 v57, v51, v55
	v_sub_co_u32_e32 v53, vcc, v53, v62
	v_xad_u32 v49, v49, v10, v46
	s_nop 0
	v_subb_co_u32_e64 v57, s[14:15], v57, v5, vcc
	v_subrev_co_u32_e64 v59, s[14:15], s34, v53
	v_subb_co_u32_e32 v51, vcc, v51, v55, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v62, s[26:27], 0, v57, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v62
	v_subb_co_u32_e64 v57, s[14:15], v57, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v59
	v_cmp_le_u32_e32 vcc, s35, v51
	v_mul_lo_u32 v49, v49, s33
	v_cndmask_b32_e64 v66, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v62
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v53
	v_cndmask_b32_e64 v63, v63, v66, s[26:27]
	v_subrev_co_u32_e64 v66, s[14:15], s34, v59
	v_mov_b32_e32 v77, v65
	s_nop 0
	v_subbrev_co_u32_e64 v57, s[14:15], 0, v57, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v63
	v_mov_b32_e32 v63, v13
	s_nop 0
	v_cndmask_b32_e64 v57, v62, v57, s[14:15]
	v_cndmask_b32_e64 v62, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v51
	s_nop 1
	v_cndmask_b32_e32 v55, v55, v62, vcc
	v_cmp_ne_u32_e32 vcc, 0, v55
	v_cndmask_b32_e64 v55, v59, v66, s[14:15]
	v_or_b32_e32 v62, 12, v12
	v_cndmask_b32_e32 v53, v53, v55, vcc
	v_cndmask_b32_e32 v51, v51, v57, vcc
	v_xor_b32_e32 v53, v53, v10
	v_xor_b32_e32 v51, v51, v10
	v_sub_co_u32_e32 v53, vcc, v53, v10
	s_nop 1
	v_subb_co_u32_e32 v51, vcc, v51, v10, vcc
	v_ashrrev_i32_e32 v51, 31, v51
	v_and_b32_e32 v51, s44, v51
	v_add3_u32 v49, v51, v53, v49
	v_sub_co_u32_e32 v51, vcc, -13, v12
	v_cndmask_b32_e64 v66, v62, v51, s[10:11]
	s_nop 0
	v_subb_co_u32_e32 v53, vcc, -1, v13, vcc
	v_cndmask_b32_e64 v67, v13, v53, s[10:11]
	v_ashrrev_i32_e32 v68, 31, v67
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v53, v66, v68
	v_xor_b32_e32 v51, v67, v68
	v_mad_u64_u32 v[72:73], s[14:15], v53, s28, 0
	v_mul_hi_u32 v76, v53, s48
	v_mad_u64_u32 v[66:67], s[14:15], v51, s48, 0
	v_lshl_add_u64 v[72:73], v[76:77], 0, v[72:73]
	v_add_co_u32_e32 v55, vcc, v72, v66
	v_mad_u64_u32 v[76:77], s[14:15], v51, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v73, v67, vcc
	v_mov_b32_e32 v67, s29
	s_nop 0
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[76:77]
	v_mul_lo_u32 v55, s35, v66
	v_mul_lo_u32 v57, s34, v67
	v_mad_u64_u32 v[72:73], s[14:15], s34, v66, 0
	v_add3_u32 v55, v73, v57, v55
	v_sub_u32_e32 v57, v51, v55
	v_sub_co_u32_e32 v53, vcc, v53, v72
	v_lshl_add_u64 v[62:63], v[62:63], 0, v[10:11]
	s_nop 0
	v_subb_co_u32_e64 v57, s[14:15], v57, v5, vcc
	v_subrev_co_u32_e64 v59, s[14:15], s34, v53
	v_subb_co_u32_e32 v51, vcc, v51, v55, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v57, s[14:15], 0, v57, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v57
	v_cmp_le_u32_e32 vcc, s35, v51
	v_mov_b32_e32 v69, v65
	v_cndmask_b32_e64 v67, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v59
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v53
	v_cndmask_b32_e64 v59, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v57
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v51
	v_cndmask_b32_e64 v57, v67, v59, s[14:15]
	v_add_u32_e32 v59, 2, v66
	v_add_u32_e32 v67, 1, v66
	v_cmp_ne_u32_e64 s[14:15], 0, v57
	v_cndmask_b32_e32 v51, v55, v53, vcc
	v_cmp_ne_u32_e32 vcc, 0, v51
	v_cndmask_b32_e64 v57, v67, v59, s[14:15]
	v_xor_b32_e32 v53, s70, v68
	v_cndmask_b32_e32 v51, v66, v57, vcc
	v_xor_b32_e32 v51, v51, v53
	v_xor_b32_e32 v55, v62, v10
	v_sub_u32_e32 v51, v51, v53
	v_xor_b32_e32 v53, v63, v10
	v_mad_u64_u32 v[66:67], s[14:15], v55, s28, 0
	v_mul_hi_u32 v68, v55, s48
	v_mad_u64_u32 v[62:63], s[14:15], v53, s48, 0
	v_lshl_add_u64 v[66:67], v[68:69], 0, v[66:67]
	v_add_co_u32_e32 v57, vcc, v66, v62
	v_mad_u64_u32 v[68:69], s[14:15], v53, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v62, vcc, v67, v63, vcc
	v_mov_b32_e32 v63, s29
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[62:63], v[62:63], 0, v[68:69]
	v_mul_lo_u32 v57, s35, v62
	v_mul_lo_u32 v59, s34, v63
	v_mad_u64_u32 v[62:63], s[14:15], s34, v62, 0
	v_add3_u32 v57, v63, v59, v57
	v_sub_u32_e32 v59, v53, v57
	v_sub_co_u32_e32 v55, vcc, v55, v62
	v_xad_u32 v51, v51, v10, v46
	s_nop 0
	v_subb_co_u32_e64 v59, s[14:15], v59, v5, vcc
	v_subrev_co_u32_e64 v62, s[14:15], s34, v55
	v_subb_co_u32_e32 v53, vcc, v53, v57, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v63, s[26:27], 0, v59, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v63
	v_subb_co_u32_e64 v59, s[14:15], v59, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v66, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v62
	v_cmp_le_u32_e32 vcc, s35, v53
	v_mul_lo_u32 v51, v51, s33
	v_cndmask_b32_e64 v67, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v63
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v55
	v_cndmask_b32_e64 v66, v66, v67, s[26:27]
	v_subrev_co_u32_e64 v67, s[14:15], s34, v62
	v_mov_b32_e32 v77, v65
	s_nop 0
	v_subbrev_co_u32_e64 v59, s[14:15], 0, v59, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v66
	s_nop 1
	v_cndmask_b32_e64 v59, v63, v59, s[14:15]
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v53
	s_nop 1
	v_cndmask_b32_e32 v57, v57, v63, vcc
	v_cmp_ne_u32_e32 vcc, 0, v57
	v_cndmask_b32_e64 v57, v62, v67, s[14:15]
	v_or_b32_e32 v62, 13, v12
	v_cndmask_b32_e32 v55, v55, v57, vcc
	v_cndmask_b32_e32 v53, v53, v59, vcc
	v_xor_b32_e32 v55, v55, v10
	v_xor_b32_e32 v53, v53, v10
	v_sub_co_u32_e32 v55, vcc, v55, v10
	v_mov_b32_e32 v63, v13
	s_nop 0
	v_subb_co_u32_e32 v53, vcc, v53, v10, vcc
	v_ashrrev_i32_e32 v53, 31, v53
	v_and_b32_e32 v53, s44, v53
	v_add3_u32 v51, v53, v55, v51
	v_sub_co_u32_e32 v53, vcc, -14, v12
	v_cndmask_b32_e64 v66, v62, v53, s[10:11]
	s_nop 0
	v_subb_co_u32_e32 v55, vcc, -1, v13, vcc
	v_cndmask_b32_e64 v67, v13, v55, s[10:11]
	v_ashrrev_i32_e32 v68, 31, v67
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v55, v66, v68
	v_xor_b32_e32 v53, v67, v68
	v_mad_u64_u32 v[72:73], s[14:15], v55, s28, 0
	v_mul_hi_u32 v76, v55, s48
	v_mad_u64_u32 v[66:67], s[14:15], v53, s48, 0
	v_lshl_add_u64 v[72:73], v[76:77], 0, v[72:73]
	v_add_co_u32_e32 v57, vcc, v72, v66
	v_mad_u64_u32 v[76:77], s[14:15], v53, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v73, v67, vcc
	v_mov_b32_e32 v67, s29
	s_nop 0
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[76:77]
	v_mul_lo_u32 v57, s35, v66
	v_mul_lo_u32 v59, s34, v67
	v_mad_u64_u32 v[72:73], s[14:15], s34, v66, 0
	v_add3_u32 v57, v73, v59, v57
	v_sub_u32_e32 v59, v53, v57
	v_sub_co_u32_e32 v55, vcc, v55, v72
	v_lshl_add_u64 v[62:63], v[62:63], 0, v[10:11]
	s_nop 0
	v_subb_co_u32_e64 v59, s[14:15], v59, v5, vcc
	v_subrev_co_u32_e64 v67, s[14:15], s34, v55
	v_subb_co_u32_e32 v53, vcc, v53, v57, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v59, s[14:15], 0, v59, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v59
	v_cmp_le_u32_e32 vcc, s35, v53
	v_mov_b32_e32 v77, v65
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v67
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v55
	v_cndmask_b32_e64 v67, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v59
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v53
	v_cndmask_b32_e64 v59, v69, v67, s[14:15]
	v_add_u32_e32 v67, 2, v66
	v_add_u32_e32 v69, 1, v66
	v_cmp_ne_u32_e64 s[14:15], 0, v59
	v_cndmask_b32_e32 v53, v57, v55, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_cndmask_b32_e64 v59, v69, v67, s[14:15]
	v_xor_b32_e32 v55, s70, v68
	v_cndmask_b32_e32 v53, v66, v59, vcc
	v_xor_b32_e32 v53, v53, v55
	v_xor_b32_e32 v57, v62, v10
	v_sub_u32_e32 v53, v53, v55
	v_xor_b32_e32 v55, v63, v10
	v_mad_u64_u32 v[66:67], s[14:15], v57, s28, 0
	v_mul_hi_u32 v68, v57, s48
	v_mov_b32_e32 v69, v65
	v_mad_u64_u32 v[62:63], s[14:15], v55, s48, 0
	v_lshl_add_u64 v[66:67], v[68:69], 0, v[66:67]
	v_add_co_u32_e32 v59, vcc, v66, v62
	v_mad_u64_u32 v[68:69], s[14:15], v55, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v62, vcc, v67, v63, vcc
	v_mov_b32_e32 v63, s29
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[62:63], v[62:63], 0, v[68:69]
	v_mul_lo_u32 v59, s35, v62
	v_mul_lo_u32 v66, s34, v63
	v_mad_u64_u32 v[62:63], s[14:15], s34, v62, 0
	v_add3_u32 v59, v63, v66, v59
	v_sub_u32_e32 v63, v55, v59
	v_sub_co_u32_e32 v57, vcc, v57, v62
	v_xad_u32 v53, v53, v10, v46
	s_nop 0
	v_subb_co_u32_e64 v62, s[14:15], v63, v5, vcc
	v_subrev_co_u32_e64 v63, s[14:15], s34, v57
	v_subb_co_u32_e32 v55, vcc, v55, v59, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v66, s[26:27], 0, v62, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v66
	v_subb_co_u32_e64 v62, s[14:15], v62, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v67, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v63
	v_cmp_le_u32_e32 vcc, s35, v55
	v_mul_lo_u32 v53, v53, s33
	v_cndmask_b32_e64 v68, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v66
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v57
	v_cndmask_b32_e64 v67, v67, v68, s[26:27]
	v_subrev_co_u32_e64 v68, s[14:15], s34, v63
	s_nop 1
	v_subbrev_co_u32_e64 v62, s[14:15], 0, v62, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v67
	s_nop 1
	v_cndmask_b32_e64 v62, v66, v62, s[14:15]
	v_cndmask_b32_e64 v66, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v55
	s_nop 1
	v_cndmask_b32_e32 v59, v59, v66, vcc
	v_cmp_ne_u32_e32 vcc, 0, v59
	v_cndmask_b32_e64 v59, v63, v68, s[14:15]
	v_mov_b32_e32 v63, v13
	v_cndmask_b32_e32 v57, v57, v59, vcc
	v_cndmask_b32_e32 v55, v55, v62, vcc
	v_xor_b32_e32 v57, v57, v10
	v_xor_b32_e32 v55, v55, v10
	v_sub_co_u32_e32 v57, vcc, v57, v10
	v_or_b32_e32 v62, 14, v12
	s_nop 0
	v_subb_co_u32_e32 v55, vcc, v55, v10, vcc
	v_ashrrev_i32_e32 v55, 31, v55
	v_and_b32_e32 v55, s44, v55
	v_add3_u32 v53, v55, v57, v53
	v_sub_co_u32_e32 v55, vcc, -15, v12
	v_cndmask_b32_e64 v66, v62, v55, s[10:11]
	s_nop 0
	v_subb_co_u32_e32 v57, vcc, -1, v13, vcc
	v_cndmask_b32_e64 v67, v13, v57, s[10:11]
	v_ashrrev_i32_e32 v68, 31, v67
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v57, v66, v68
	v_xor_b32_e32 v55, v67, v68
	v_mad_u64_u32 v[72:73], s[14:15], v57, s28, 0
	v_mul_hi_u32 v76, v57, s48
	v_mad_u64_u32 v[66:67], s[14:15], v55, s48, 0
	v_lshl_add_u64 v[72:73], v[76:77], 0, v[72:73]
	v_add_co_u32_e32 v59, vcc, v72, v66
	v_mad_u64_u32 v[76:77], s[14:15], v55, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v73, v67, vcc
	v_mov_b32_e32 v67, s29
	s_nop 0
	v_addc_co_u32_e32 v77, vcc, 0, v77, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[76:77]
	v_mul_lo_u32 v59, s35, v66
	v_mul_lo_u32 v67, s34, v67
	v_mad_u64_u32 v[72:73], s[14:15], s34, v66, 0
	v_add3_u32 v59, v73, v67, v59
	v_sub_u32_e32 v67, v55, v59
	v_sub_co_u32_e32 v57, vcc, v57, v72
	v_lshl_add_u64 v[62:63], v[62:63], 0, v[10:11]
	s_nop 0
	v_subb_co_u32_e64 v67, s[14:15], v67, v5, vcc
	v_subrev_co_u32_e64 v69, s[14:15], s34, v57
	v_subb_co_u32_e32 v55, vcc, v55, v59, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v67
	v_cmp_le_u32_e32 vcc, s35, v55
	v_mov_b32_e32 v73, v65
	v_cndmask_b32_e64 v72, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v69
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v57
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v67
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v55
	v_cndmask_b32_e64 v67, v72, v69, s[14:15]
	v_add_u32_e32 v69, 2, v66
	v_add_u32_e32 v72, 1, v66
	v_cmp_ne_u32_e64 s[14:15], 0, v67
	v_cndmask_b32_e32 v55, v59, v57, vcc
	v_cmp_ne_u32_e32 vcc, 0, v55
	v_cndmask_b32_e64 v67, v72, v69, s[14:15]
	v_xor_b32_e32 v57, s70, v68
	v_cndmask_b32_e32 v55, v66, v67, vcc
	v_xor_b32_e32 v55, v55, v57
	v_xor_b32_e32 v59, v62, v10
	v_sub_u32_e32 v55, v55, v57
	v_xor_b32_e32 v57, v63, v10
	v_mad_u64_u32 v[66:67], s[14:15], v59, s28, 0
	v_mul_hi_u32 v68, v59, s48
	v_mov_b32_e32 v69, v65
	v_mad_u64_u32 v[62:63], s[14:15], v57, s48, 0
	v_lshl_add_u64 v[66:67], v[68:69], 0, v[66:67]
	v_add_co_u32_e32 v62, vcc, v66, v62
	v_mad_u64_u32 v[68:69], s[14:15], v57, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v62, vcc, v67, v63, vcc
	v_mov_b32_e32 v63, s29
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[62:63], v[62:63], 0, v[68:69]
	v_mul_lo_u32 v66, s35, v62
	v_mul_lo_u32 v67, s34, v63
	v_mad_u64_u32 v[62:63], s[14:15], s34, v62, 0
	v_add3_u32 v63, v63, v67, v66
	v_sub_u32_e32 v66, v57, v63
	v_sub_co_u32_e32 v59, vcc, v59, v62
	v_xad_u32 v55, v55, v10, v46
	s_nop 0
	v_subb_co_u32_e64 v62, s[14:15], v66, v5, vcc
	v_subrev_co_u32_e64 v66, s[14:15], s34, v59
	v_subb_co_u32_e32 v57, vcc, v57, v63, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[26:27], 0, v62, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v67
	v_subb_co_u32_e64 v62, s[14:15], v62, v5, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v68, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v66
	v_cmp_le_u32_e32 vcc, s35, v57
	v_mul_lo_u32 v55, v55, s33
	v_cndmask_b32_e64 v69, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v67
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v59
	v_cndmask_b32_e64 v68, v68, v69, s[26:27]
	v_subrev_co_u32_e64 v69, s[14:15], s34, v66
	s_nop 1
	v_subbrev_co_u32_e64 v62, s[14:15], 0, v62, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v68
	s_nop 1
	v_cndmask_b32_e64 v62, v67, v62, s[14:15]
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v57
	s_nop 1
	v_cndmask_b32_e32 v63, v63, v67, vcc
	v_cmp_ne_u32_e32 vcc, 0, v63
	v_mov_b32_e32 v63, v13
	s_nop 0
	v_cndmask_b32_e32 v57, v57, v62, vcc
	v_cndmask_b32_e64 v62, v66, v69, s[14:15]
	v_cndmask_b32_e32 v59, v59, v62, vcc
	v_xor_b32_e32 v59, v59, v10
	v_xor_b32_e32 v57, v57, v10
	v_sub_co_u32_e32 v59, vcc, v59, v10
	v_or_b32_e32 v62, 15, v12
	s_nop 0
	v_subb_co_u32_e32 v57, vcc, v57, v10, vcc
	v_ashrrev_i32_e32 v57, 31, v57
	v_and_b32_e32 v57, s44, v57
	v_sub_co_u32_e32 v12, vcc, -16, v12
	v_add3_u32 v59, v57, v59, v55
	s_nop 0
	v_subb_co_u32_e32 v55, vcc, -1, v13, vcc
	v_cndmask_b32_e64 v13, v13, v55, s[10:11]
	v_ashrrev_i32_e32 v66, 31, v13
	v_cndmask_b32_e64 v12, v62, v12, s[10:11]
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[66:67]
	v_xor_b32_e32 v57, v12, v66
	v_xor_b32_e32 v55, v13, v66
	v_mad_u64_u32 v[68:69], s[10:11], v57, s28, 0
	v_mul_hi_u32 v72, v57, s48
	v_mad_u64_u32 v[12:13], s[10:11], v55, s48, 0
	v_lshl_add_u64 v[68:69], v[72:73], 0, v[68:69]
	v_add_co_u32_e32 v12, vcc, v68, v12
	v_mad_u64_u32 v[72:73], s[10:11], v55, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v69, v13, vcc
	v_mov_b32_e32 v13, s29
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[72:73]
	v_mul_lo_u32 v67, s35, v12
	v_mul_lo_u32 v13, s34, v13
	v_mad_u64_u32 v[68:69], s[10:11], s34, v12, 0
	v_add3_u32 v13, v69, v13, v67
	v_sub_u32_e32 v67, v55, v13
	v_sub_co_u32_e32 v57, vcc, v57, v68
	s_nop 1
	v_subb_co_u32_e64 v67, s[10:11], v67, v5, vcc
	v_subrev_co_u32_e64 v68, s[10:11], s34, v57
	v_subb_co_u32_e32 v13, vcc, v55, v13, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[10:11], 0, v67, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s35, v67
	v_cmp_le_u32_e32 vcc, s35, v13
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s34, v68
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v57
	v_cndmask_b32_e64 v68, 0, -1, s[10:11]
	v_cmp_eq_u32_e64 s[10:11], s35, v67
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v13
	v_cndmask_b32_e64 v67, v69, v68, s[10:11]
	v_add_u32_e32 v68, 2, v12
	v_add_u32_e32 v69, 1, v12
	v_cmp_ne_u32_e64 s[10:11], 0, v67
	v_cndmask_b32_e32 v13, v55, v57, vcc
	v_cmp_ne_u32_e32 vcc, 0, v13
	v_cndmask_b32_e64 v67, v69, v68, s[10:11]
	v_xor_b32_e32 v13, s70, v66
	v_cndmask_b32_e32 v12, v12, v67, vcc
	v_xor_b32_e32 v12, v12, v13
	v_sub_u32_e32 v12, v12, v13
	v_xad_u32 v55, v12, v10, v46
	v_lshl_add_u64 v[12:13], v[62:63], 0, v[10:11]
	v_xor_b32_e32 v57, v12, v10
	v_xor_b32_e32 v11, v13, v10
	v_mad_u64_u32 v[62:63], s[10:11], v57, s28, 0
	v_mul_hi_u32 v66, v57, s48
	v_mov_b32_e32 v67, v65
	v_mad_u64_u32 v[12:13], s[10:11], v11, s48, 0
	v_lshl_add_u64 v[62:63], v[66:67], 0, v[62:63]
	v_add_co_u32_e32 v12, vcc, v62, v12
	v_mad_u64_u32 v[66:67], s[10:11], v11, s28, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v63, v13, vcc
	v_mov_b32_e32 v13, s29
	s_nop 0
	v_addc_co_u32_e32 v67, vcc, 0, v67, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[66:67]
	v_mul_lo_u32 v62, s35, v12
	v_mul_lo_u32 v63, s34, v13
	v_mad_u64_u32 v[12:13], s[10:11], s34, v12, 0
	v_add3_u32 v13, v13, v63, v62
	v_sub_u32_e32 v62, v11, v13
	v_sub_co_u32_e32 v12, vcc, v57, v12
	s_nop 1
	v_subb_co_u32_e64 v57, s[10:11], v62, v5, vcc
	v_subrev_co_u32_e64 v62, s[10:11], s34, v12
	v_subb_co_u32_e32 v11, vcc, v11, v13, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v63, s[14:15], 0, v57, s[10:11]
	v_cmp_le_u32_e64 s[14:15], s35, v63
	v_subb_co_u32_e64 v5, s[10:11], v57, v5, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v66, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v62
	v_subrev_co_u32_e64 v57, s[10:11], s34, v62
	s_nop 0
	v_cndmask_b32_e64 v67, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v63
	v_subbrev_co_u32_e64 v5, s[10:11], 0, v5, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v66, v66, v67, s[14:15]
	v_cmp_le_u32_e32 vcc, s35, v11
	v_cmp_ne_u32_e64 s[10:11], 0, v66
	s_nop 0
	v_cndmask_b32_e64 v13, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v12
	v_cndmask_b32_e64 v5, v63, v5, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v11
	s_nop 1
	v_cndmask_b32_e32 v13, v13, v63, vcc
	v_cmp_ne_u32_e32 vcc, 0, v13
	s_nop 1
	v_cndmask_b32_e32 v5, v11, v5, vcc
	v_cndmask_b32_e64 v11, v62, v57, s[10:11]
	v_cndmask_b32_e32 v11, v12, v11, vcc
	v_xor_b32_e32 v11, v11, v10
	v_xor_b32_e32 v5, v5, v10
	v_sub_co_u32_e32 v11, vcc, v11, v10
	s_nop 1
	v_subb_co_u32_e32 v5, vcc, v5, v10, vcc
	v_ashrrev_i32_e32 v5, 31, v5
	v_and_b32_e32 v5, s44, v5
	v_mul_lo_u32 v10, v55, s33
	v_add3_u32 v5, v5, v11, v10
	buffer_load_ubyte v30, v30, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v57, v31, s[36:39], 0 offen
	buffer_load_ubyte v55, v47, s[36:39], 0 offen
	buffer_load_ubyte v76, v49, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v31, v51, s[36:39], 0 offen
	buffer_load_ubyte v78, v53, s[36:39], 0 offen
	buffer_load_ubyte v77, v59, s[36:39], 0 offen
	buffer_load_ubyte v79, v5, s[36:39], 0 offen
	s_add_u32 s14, s46, 7
	s_addc_u32 s15, s47, 0
	s_sub_u32 s26, -8, s46
	s_subb_u32 s27, -1, s47
	v_cmp_lt_i64_e64 s[10:11], s[46:47], -7
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[2:3]
	v_lshlrev_b32_e32 v49, 2, v40
	v_cndmask_b32_e64 v3, 0, -1, s[10:11]
	s_and_b64 s[10:11], s[10:11], exec
	s_cselect_b32 s11, s27, s15
	s_cselect_b32 s10, s26, s14
	s_ashr_i32 s14, s11, 31
	s_lshr_b32 s14, s14, 29
	s_add_u32 s10, s10, s14
	v_lshl_add_u32 v47, s50, 5, v49
	v_bfe_u32 v2, v6, 4, 1
	s_addc_u32 s11, s11, 0
	v_readfirstlane_b32 s14, v3
	v_or_b32_e32 v62, v47, v2
	s_ashr_i64 s[10:11], s[10:11], 3
	s_mov_b32 s15, s14
	v_lshlrev_b32_e32 v66, 6, v97
	v_mul_i32_i24_e32 v68, 0xffffffc0, v4
	s_xor_b64 s[26:27], s[10:11], s[14:15]
	v_add3_u32 v4, v68, v66, v62
	s_lshl_b64 s[76:77], s[26:27], 3
	v_ashrrev_i32_e32 v5, 31, v4
	v_or_b32_e32 v3, s77, v5
	v_cmp_ne_u32_e32 vcc, 0, v3
	s_and_saveexec_b64 s[10:11], vcc
	s_xor_b64 s[34:35], exec, s[10:11]
	s_cbranch_execz .LBB0_16
	s_ashr_i32 s10, s77, 31
	s_add_u32 s14, s76, s10
	s_mov_b32 s11, s10
	s_addc_u32 s15, s77, s10
	s_xor_b64 s[48:49], s[14:15], s[10:11]
	v_cvt_f32_u32_e32 v3, s48
	v_cvt_f32_u32_e32 v10, s49
	s_sub_u32 s10, 0, s48
	s_subb_u32 s11, 0, s49
	v_ashrrev_i32_e32 v12, 31, v5
	v_fmac_f32_e32 v3, 0x4f800000, v10
	v_rcp_f32_e32 v3, v3
	v_mov_b32_e32 v13, v12
	v_mov_b32_e32 v73, v65
	v_mul_f32_e32 v3, 0x5f7ffffc, v3
	v_mul_f32_e32 v10, 0x2f800000, v3
	v_trunc_f32_e32 v10, v10
	v_fmac_f32_e32 v3, 0xcf800000, v10
	v_cvt_u32_f32_e32 v10, v10
	v_cvt_u32_f32_e32 v3, v3
	v_readfirstlane_b32 s14, v10
	v_readfirstlane_b32 s15, v3
	s_mul_i32 s28, s10, s14
	s_mul_hi_u32 s51, s10, s15
	s_mul_i32 s50, s11, s15
	s_add_i32 s28, s51, s28
	s_add_i32 s28, s28, s50
	s_mul_i32 s55, s10, s15
	s_mul_i32 s51, s15, s28
	s_mul_hi_u32 s56, s15, s55
	s_mul_hi_u32 s50, s15, s28
	s_add_u32 s51, s56, s51
	s_addc_u32 s50, 0, s50
	s_mul_hi_u32 s57, s14, s55
	s_mul_i32 s55, s14, s55
	s_add_u32 s51, s51, s55
	s_mul_hi_u32 s56, s14, s28
	s_addc_u32 s50, s50, s57
	s_addc_u32 s51, s56, 0
	s_mul_i32 s28, s14, s28
	s_add_u32 s28, s50, s28
	s_addc_u32 s50, 0, s51
	s_add_u32 s15, s15, s28
	s_addc_u32 s14, s14, s50
	s_mul_i32 s28, s10, s14
	s_mul_hi_u32 s50, s10, s15
	s_add_i32 s28, s50, s28
	s_mul_i32 s11, s11, s15
	s_add_i32 s28, s28, s11
	s_mul_i32 s10, s10, s15
	s_mul_hi_u32 s50, s14, s10
	s_mul_i32 s51, s14, s10
	s_mul_i32 s56, s15, s28
	s_mul_hi_u32 s10, s15, s10
	s_mul_hi_u32 s55, s15, s28
	s_add_u32 s10, s10, s56
	s_addc_u32 s55, 0, s55
	s_add_u32 s10, s10, s51
	s_mul_hi_u32 s11, s14, s28
	s_addc_u32 s10, s55, s50
	s_addc_u32 s11, s11, 0
	s_mul_i32 s28, s14, s28
	s_add_u32 s10, s10, s28
	s_addc_u32 s11, 0, s11
	s_add_u32 s15, s15, s10
	v_lshl_add_u64 v[10:11], v[4:5], 0, v[12:13]
	s_addc_u32 s14, s14, s11
	v_xor_b32_e32 v5, v10, v12
	v_xor_b32_e32 v3, v11, v12
	v_mad_u64_u32 v[10:11], s[10:11], v5, s14, 0
	v_mul_hi_u32 v72, v5, s15
	v_lshl_add_u64 v[10:11], v[72:73], 0, v[10:11]
	v_mad_u64_u32 v[80:81], s[10:11], v3, s15, 0
	v_add_co_u32_e32 v10, vcc, v10, v80
	v_mad_u64_u32 v[72:73], s[10:11], v3, s14, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v81, vcc
	v_mov_b32_e32 v11, s29
	s_nop 0
	v_addc_co_u32_e32 v73, vcc, 0, v73, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[72:73]
	v_mul_lo_u32 v13, s49, v10
	v_mul_lo_u32 v51, s48, v11
	v_mad_u64_u32 v[10:11], s[10:11], s48, v10, 0
	v_add3_u32 v11, v11, v51, v13
	v_sub_u32_e32 v13, v3, v11
	v_mov_b32_e32 v51, s49
	v_sub_co_u32_e32 v5, vcc, v5, v10
	s_nop 1
	v_subb_co_u32_e64 v10, s[10:11], v13, v51, vcc
	v_subrev_co_u32_e64 v13, s[10:11], s48, v5
	v_subb_co_u32_e32 v3, vcc, v3, v11, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v53, s[14:15], 0, v10, s[10:11]
	v_cmp_le_u32_e64 s[14:15], s49, v53
	v_subb_co_u32_e64 v10, s[10:11], v10, v51, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s48, v13
	v_subrev_co_u32_e64 v51, s[10:11], s48, v13
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s49, v53
	v_subbrev_co_u32_e64 v10, s[10:11], 0, v10, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v59, v59, v63, s[14:15]
	v_cmp_le_u32_e32 vcc, s49, v3
	v_cmp_ne_u32_e64 s[10:11], 0, v59
	s_nop 0
	v_cndmask_b32_e64 v11, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s48, v5
	v_cndmask_b32_e64 v10, v53, v10, s[10:11]
	s_nop 0
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s49, v3
	s_nop 1
	v_cndmask_b32_e32 v11, v11, v53, vcc
	v_cmp_ne_u32_e32 vcc, 0, v11
	s_nop 1
	v_cndmask_b32_e32 v3, v3, v10, vcc
	v_cndmask_b32_e64 v10, v13, v51, s[10:11]
	v_cndmask_b32_e32 v5, v5, v10, vcc
	v_xor_b32_e32 v5, v5, v12
	v_xor_b32_e32 v3, v3, v12
	v_sub_co_u32_e32 v10, vcc, v5, v12
	s_nop 1
	v_subb_co_u32_e32 v11, vcc, v3, v12, vcc
.LBB0_16:
	s_or_saveexec_b64 s[10:11], s[34:35]
	v_cvt_f32_u32_e32 v59, s76
	s_xor_b64 exec, exec, s[10:11]
	s_cbranch_execz .LBB0_18
	v_rcp_iflag_f32_e32 v3, v59
	s_sub_i32 s14, 0, s76
	v_mov_b32_e32 v11, 0
	v_mul_f32_e32 v3, 0x4f7ffffe, v3
	v_cvt_u32_f32_e32 v3, v3
	v_mul_lo_u32 v5, s14, v3
	v_mul_hi_u32 v5, v3, v5
	v_add_u32_e32 v3, v3, v5
	v_mul_hi_u32 v3, v4, v3
	v_mul_lo_u32 v3, v3, s76
	v_sub_u32_e32 v3, v4, v3
	v_subrev_u32_e32 v5, s76, v3
	v_cmp_le_u32_e32 vcc, s76, v3
	s_nop 1
	v_cndmask_b32_e32 v3, v3, v5, vcc
	v_subrev_u32_e32 v5, s76, v3
	v_cmp_le_u32_e32 vcc, s76, v3
	s_nop 1
	v_cndmask_b32_e32 v10, v3, v5, vcc
.LBB0_18:
	s_or_b64 exec, exec, s[10:11]
	v_ashrrev_i32_e32 v3, 31, v7
	v_xor_b32_e32 v7, v3, v7
	v_mov_b32_e32 v63, 0
	v_ashrrev_i32_e32 v5, 31, v7
	v_xor_b32_e32 v6, v3, v6
	v_lshrrev_b32_e32 v12, 27, v5
	v_mov_b32_e32 v13, v63
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[12:13]
	v_ashrrev_i64 v[6:7], 5, v[6:7]
	v_xor_b32_e32 v5, v7, v3
	v_xor_b32_e32 v3, v6, v3
	s_lshl_b32 s48, s26, 8
	s_lshr_b64 s[10:11], s[26:27], 24
	v_mul_lo_u32 v51, s10, v3
	v_mad_u64_u32 v[72:73], s[10:11], s48, v3, v[62:63]
	s_ashr_i32 s78, s77, 31
	s_add_u32 s10, s76, s78
	s_mov_b32 s79, s78
	s_addc_u32 s11, s77, s78
	s_xor_b64 s[14:15], s[10:11], s[78:79]
	v_mul_lo_u32 v53, s48, v5
	v_cvt_f32_u32_e32 v5, s14
	v_cvt_f32_u32_e32 v65, s15
	s_sub_u32 s10, 0, s14
	s_subb_u32 s11, 0, s15
	v_add3_u32 v73, v51, v73, v53
	v_fmac_f32_e32 v5, 0x4f800000, v65
	v_rcp_f32_e32 v5, v5
	v_mov_b32_e32 v67, v63
	v_lshl_add_u64 v[6:7], v[72:73], 0, v[66:67]
	v_ashrrev_i32_e32 v69, 31, v68
	v_mul_f32_e32 v5, 0x5f7ffffc, v5
	v_mul_f32_e32 v80, 0x2f800000, v5
	v_trunc_f32_e32 v80, v80
	v_fmac_f32_e32 v5, 0xcf800000, v80
	v_cvt_u32_f32_e32 v5, v5
	v_cvt_u32_f32_e32 v80, v80
	v_lshl_add_u64 v[12:13], v[6:7], 0, v[68:69]
	v_ashrrev_i32_e32 v65, 31, v13
	v_readfirstlane_b32 s26, v5
	v_readfirstlane_b32 s29, v80
	s_mul_hi_u32 s28, s10, s26
	s_mul_i32 s34, s10, s29
	s_mul_i32 s27, s11, s26
	s_add_i32 s28, s28, s34
	s_add_i32 s28, s28, s27
	s_mul_i32 s35, s10, s26
	s_mul_i32 s34, s26, s28
	s_mul_hi_u32 s49, s26, s35
	s_mul_hi_u32 s27, s26, s28
	s_add_u32 s34, s49, s34
	s_addc_u32 s27, 0, s27
	s_mul_hi_u32 s50, s29, s35
	s_mul_i32 s35, s29, s35
	s_add_u32 s34, s34, s35
	s_mul_hi_u32 s49, s29, s28
	s_addc_u32 s27, s27, s50
	s_addc_u32 s34, s49, 0
	s_mul_i32 s28, s29, s28
	s_add_u32 s27, s27, s28
	s_addc_u32 s28, 0, s34
	s_add_u32 s26, s26, s27
	s_addc_u32 s27, s29, s28
	s_mul_i32 s28, s10, s27
	s_mul_hi_u32 s29, s10, s26
	s_add_i32 s28, s29, s28
	s_mul_i32 s11, s11, s26
	s_add_i32 s28, s28, s11
	s_mul_i32 s10, s10, s26
	s_mul_hi_u32 s29, s27, s10
	s_mul_i32 s34, s27, s10
	s_mul_i32 s49, s26, s28
	s_mul_hi_u32 s10, s26, s10
	s_mul_hi_u32 s35, s26, s28
	s_add_u32 s10, s10, s49
	s_addc_u32 s35, 0, s35
	s_add_u32 s10, s10, s34
	s_mul_hi_u32 s11, s27, s28
	s_addc_u32 s10, s35, s29
	v_xor_b32_e32 v7, v65, v13
	s_addc_u32 s11, s11, 0
	s_mul_i32 s28, s27, s28
	s_add_u32 s10, s10, s28
	v_ashrrev_i32_e32 v80, 31, v7
	v_xor_b32_e32 v6, v65, v12
	s_addc_u32 s11, 0, s11
	v_mov_b32_e32 v81, v80
	s_add_u32 s26, s26, s10
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[80:81]
	s_addc_u32 s27, s27, s11
	v_xor_b32_e32 v81, v6, v80
	v_xor_b32_e32 v5, v7, v80
	v_mad_u64_u32 v[6:7], s[10:11], v81, s27, 0
	v_mul_hi_u32 v82, v81, s26
	v_mov_b32_e32 v83, v63
	v_lshl_add_u64 v[6:7], v[82:83], 0, v[6:7]
	v_mad_u64_u32 v[84:85], s[10:11], v5, s26, 0
	v_add_co_u32_e32 v6, vcc, v6, v84
	v_mad_u64_u32 v[82:83], s[10:11], v5, s27, 0
	s_nop 0
	v_addc_co_u32_e32 v6, vcc, v7, v85, vcc
	v_mov_b32_e32 v7, v63
	s_nop 0
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[6:7], v[6:7], 0, v[82:83]
	v_mul_lo_u32 v84, s15, v6
	v_mul_lo_u32 v7, s14, v7
	v_mad_u64_u32 v[82:83], s[10:11], s14, v6, 0
	v_add3_u32 v7, v83, v7, v84
	v_sub_u32_e32 v83, v5, v7
	v_mov_b32_e32 v84, s15
	v_sub_co_u32_e32 v81, vcc, v81, v82
	s_and_b32 s9, s9, 0xffff
	s_nop 0
	v_subb_co_u32_e64 v82, s[10:11], v83, v84, vcc
	v_subrev_co_u32_e64 v83, s[10:11], s14, v81
	v_subb_co_u32_e32 v5, vcc, v5, v7, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v82, s[10:11], 0, v82, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s15, v82
	v_cmp_le_u32_e32 vcc, s15, v5
	s_nop 0
	v_cndmask_b32_e64 v84, 0, -1, s[10:11]
	v_cmp_le_u32_e64 s[10:11], s14, v83
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s14, v81
	v_cndmask_b32_e64 v83, 0, -1, s[10:11]
	v_cmp_eq_u32_e64 s[10:11], s15, v82
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s15, v5
	v_cndmask_b32_e64 v82, v84, v83, s[10:11]
	v_add_u32_e32 v83, 2, v6
	v_add_u32_e32 v84, 1, v6
	v_cmp_ne_u32_e64 s[10:11], 0, v82
	v_cndmask_b32_e32 v5, v7, v81, vcc
	v_cmp_ne_u32_e32 vcc, 0, v5
	v_cndmask_b32_e64 v82, v84, v83, s[10:11]
	s_mov_b32 s11, 0x27000
	v_cndmask_b32_e32 v5, v6, v82, vcc
	v_xor_b32_e32 v6, s78, v80
	v_xor_b32_e32 v5, v5, v6
	v_sub_u32_e32 v5, v5, v6
	v_xor_b32_e32 v5, v5, v65
	v_ashrrev_i32_e32 v6, 31, v11
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v5, v5, s54
	v_and_b32_e32 v6, s76, v6
	v_add3_u32 v5, v5, v10, v6
	s_mov_b32 s10, -2
	buffer_load_ubyte v130, v5, s[8:11], 0 offen
	v_sub_co_u32_e32 v5, vcc, -3, v12
	v_lshl_add_u64 v[6:7], v[12:13], 0, 2
	s_nop 0
	v_subb_co_u32_e32 v10, vcc, -1, v13, vcc
	v_cmp_gt_i32_e32 vcc, 0, v7
	s_mov_b32 s14, 0
	s_nop 0
	v_cndmask_b32_e32 v13, v7, v10, vcc
	v_cndmask_b32_e32 v12, v6, v5, vcc
	v_or_b32_e32 v5, s77, v13
	v_cmp_ne_u32_e32 vcc, 0, v5
	s_and_saveexec_b64 s[26:27], vcc
	s_xor_b64 s[26:27], exec, s[26:27]
	s_cbranch_execz .LBB0_20
	s_add_u32 s28, s76, s78
	s_addc_u32 s29, s77, s78
	s_xor_b64 s[28:29], s[28:29], s[78:79]
	v_cvt_f32_u32_e32 v5, s28
	v_cvt_f32_u32_e32 v6, s29
	s_sub_u32 s15, 0, s28
	s_subb_u32 s34, 0, s29
	v_ashrrev_i32_e32 v10, 31, v13
	v_fmac_f32_e32 v5, 0x4f800000, v6
	v_rcp_f32_e32 v5, v5
	v_mov_b32_e32 v11, v10
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[10:11]
	v_mov_b32_e32 v81, v63
	v_mul_f32_e32 v5, 0x5f7ffffc, v5
	v_mul_f32_e32 v6, 0x2f800000, v5
	v_trunc_f32_e32 v6, v6
	v_fmac_f32_e32 v5, 0xcf800000, v6
	v_cvt_u32_f32_e32 v6, v6
	v_cvt_u32_f32_e32 v5, v5
	v_mov_b32_e32 v65, s29
	v_readfirstlane_b32 s35, v6
	v_readfirstlane_b32 s49, v5
	s_mul_i32 s50, s15, s35
	s_mul_hi_u32 s55, s15, s49
	s_mul_i32 s51, s34, s49
	s_add_i32 s50, s55, s50
	s_mul_i32 s56, s15, s49
	s_add_i32 s50, s50, s51
	s_mul_i32 s55, s49, s50
	s_mul_hi_u32 s57, s49, s56
	s_mul_hi_u32 s51, s49, s50
	s_add_u32 s55, s57, s55
	s_addc_u32 s51, 0, s51
	s_mul_hi_u32 s58, s35, s56
	s_mul_i32 s56, s35, s56
	s_add_u32 s55, s55, s56
	s_mul_hi_u32 s57, s35, s50
	s_addc_u32 s51, s51, s58
	s_addc_u32 s55, s57, 0
	s_mul_i32 s50, s35, s50
	s_add_u32 s50, s51, s50
	s_addc_u32 s51, 0, s55
	s_add_u32 s49, s49, s50
	s_addc_u32 s35, s35, s51
	s_mul_i32 s50, s15, s35
	s_mul_hi_u32 s51, s15, s49
	s_add_i32 s50, s51, s50
	s_mul_i32 s34, s34, s49
	s_add_i32 s50, s50, s34
	s_mul_i32 s15, s15, s49
	s_mul_hi_u32 s51, s35, s15
	s_mul_i32 s55, s35, s15
	s_mul_i32 s57, s49, s50
	s_mul_hi_u32 s15, s49, s15
	s_mul_hi_u32 s56, s49, s50
	s_add_u32 s15, s15, s57
	s_addc_u32 s56, 0, s56
	s_add_u32 s15, s15, s55
	s_mul_hi_u32 s34, s35, s50
	s_addc_u32 s15, s56, s51
	s_addc_u32 s34, s34, 0
	s_mul_i32 s50, s35, s50
	s_add_u32 s15, s15, s50
	s_addc_u32 s34, 0, s34
	s_add_u32 s15, s49, s15
	s_addc_u32 s49, s35, s34
	v_xor_b32_e32 v6, v12, v10
	v_xor_b32_e32 v5, v13, v10
	v_mad_u64_u32 v[12:13], s[34:35], v6, s49, 0
	v_mul_hi_u32 v80, v6, s15
	v_lshl_add_u64 v[12:13], v[80:81], 0, v[12:13]
	v_mad_u64_u32 v[82:83], s[34:35], v5, s15, 0
	v_add_co_u32_e32 v11, vcc, v12, v82
	v_mad_u64_u32 v[80:81], s[34:35], v5, s49, 0
	s_nop 0
	v_addc_co_u32_e32 v12, vcc, v13, v83, vcc
	v_mov_b32_e32 v13, s14
	s_nop 0
	v_addc_co_u32_e32 v81, vcc, 0, v81, vcc
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[80:81]
	v_mul_lo_u32 v11, s29, v12
	v_mul_lo_u32 v63, s28, v13
	v_mad_u64_u32 v[80:81], s[14:15], s28, v12, 0
	v_add3_u32 v11, v81, v63, v11
	v_sub_u32_e32 v63, v5, v11
	v_sub_co_u32_e32 v6, vcc, v6, v80
	v_lshl_add_u64 v[82:83], v[12:13], 0, 1
	s_nop 0
	v_subb_co_u32_e64 v63, s[14:15], v63, v65, vcc
	v_subrev_co_u32_e64 v65, s[14:15], s28, v6
	v_subb_co_u32_e32 v5, vcc, v5, v11, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v63, s[14:15], 0, v63, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s29, v63
	v_cmp_le_u32_e32 vcc, s29, v5
	s_nop 0
	v_cndmask_b32_e64 v80, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v65
	v_cndmask_b32_e64 v11, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v6
	v_cndmask_b32_e64 v65, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v63
	v_cndmask_b32_e64 v6, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v5
	v_cndmask_b32_e64 v63, v80, v65, s[14:15]
	v_lshl_add_u64 v[80:81], v[12:13], 0, 2
	v_cndmask_b32_e32 v5, v11, v6, vcc
	v_cmp_ne_u32_e32 vcc, 0, v63
	s_nop 1
	v_cndmask_b32_e32 v6, v82, v80, vcc
	v_cmp_ne_u32_e32 vcc, 0, v5
	s_nop 1
	v_cndmask_b32_e32 v5, v12, v6, vcc
	v_xor_b32_e32 v6, s78, v10
	v_xor_b32_e32 v5, v5, v6
	v_sub_co_u32_e32 v10, vcc, v5, v6
.LBB0_20:
	s_andn2_saveexec_b64 s[14:15], s[26:27]
	s_cbranch_execz .LBB0_22
	v_rcp_iflag_f32_e32 v5, v59
	s_sub_i32 s26, 0, s76
	v_mul_f32_e32 v5, 0x4f7ffffe, v5
	v_cvt_u32_f32_e32 v5, v5
	v_mul_lo_u32 v6, s26, v5
	v_mul_hi_u32 v6, v5, v6
	v_add_u32_e32 v5, v5, v6
	v_mul_hi_u32 v5, v12, v5
	v_mul_lo_u32 v6, v5, s76
	v_sub_u32_e32 v6, v12, v6
	v_add_u32_e32 v10, 1, v5
	v_subrev_u32_e32 v11, s76, v6
	v_cmp_le_u32_e32 vcc, s76, v6
	s_nop 1
	v_cndmask_b32_e32 v6, v6, v11, vcc
	v_cndmask_b32_e32 v5, v5, v10, vcc
	v_add_u32_e32 v10, 1, v5
	v_cmp_le_u32_e32 vcc, s76, v6
	s_nop 1
	v_cndmask_b32_e32 v10, v5, v10, vcc
.LBB0_22:
	s_or_b64 exec, exec, s[14:15]
	v_or_b32_e32 v4, 2, v4
	v_ashrrev_i32_e32 v5, 31, v4
	v_or_b32_e32 v6, s77, v5
	s_mov_b32 s14, 0
	v_cmp_ne_u32_e32 vcc, 0, v6
	s_and_saveexec_b64 s[26:27], vcc
	s_xor_b64 s[28:29], exec, s[26:27]
	s_cbranch_execz .LBB0_24
	s_add_u32 s26, s76, s78
	s_mov_b32 s79, s78
	s_addc_u32 s27, s77, s78
	s_xor_b64 s[34:35], s[26:27], s[78:79]
	v_cvt_f32_u32_e32 v6, s34
	v_cvt_f32_u32_e32 v11, s35
	s_sub_u32 s15, 0, s34
	s_subb_u32 s26, 0, s35
	v_ashrrev_i32_e32 v80, 31, v5
	v_fmamk_f32 v6, v11, 0x4f800000, v6
	v_rcp_f32_e32 v6, v6
	v_mov_b32_e32 v81, v80
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[80:81]
	v_mov_b32_e32 v13, 0
	v_mul_f32_e32 v6, 0x5f7ffffc, v6
	v_mul_f32_e32 v11, 0x2f800000, v6
	v_trunc_f32_e32 v11, v11
	v_fmamk_f32 v6, v11, 0xcf800000, v6
	v_cvt_u32_f32_e32 v11, v11
	v_cvt_u32_f32_e32 v6, v6
	v_readfirstlane_b32 s27, v11
	v_readfirstlane_b32 s49, v6
	s_mul_i32 s50, s15, s27
	s_mul_hi_u32 s55, s15, s49
	s_mul_i32 s51, s26, s49
	s_add_i32 s50, s55, s50
	s_add_i32 s50, s50, s51
	s_mul_i32 s56, s15, s49
	s_mul_i32 s55, s49, s50
	s_mul_hi_u32 s57, s49, s56
	s_mul_hi_u32 s51, s49, s50
	s_add_u32 s55, s57, s55
	s_addc_u32 s51, 0, s51
	s_mul_hi_u32 s58, s27, s56
	s_mul_i32 s56, s27, s56
	s_add_u32 s55, s55, s56
	s_mul_hi_u32 s57, s27, s50
	s_addc_u32 s51, s51, s58
	s_addc_u32 s55, s57, 0
	s_mul_i32 s50, s27, s50
	s_add_u32 s50, s51, s50
	s_addc_u32 s51, 0, s55
	s_add_u32 s49, s49, s50
	s_addc_u32 s27, s27, s51
	s_mul_i32 s50, s15, s27
	s_mul_hi_u32 s51, s15, s49
	s_add_i32 s50, s51, s50
	s_mul_i32 s26, s26, s49
	s_add_i32 s50, s50, s26
	s_mul_i32 s15, s15, s49
	s_mul_hi_u32 s51, s27, s15
	s_mul_i32 s55, s27, s15
	s_mul_i32 s57, s49, s50
	s_mul_hi_u32 s15, s49, s15
	s_mul_hi_u32 s56, s49, s50
	s_add_u32 s15, s15, s57
	s_addc_u32 s56, 0, s56
	s_add_u32 s15, s15, s55
	s_mul_hi_u32 s26, s27, s50
	s_addc_u32 s15, s56, s51
	s_addc_u32 s26, s26, 0
	s_mul_i32 s50, s27, s50
	s_add_u32 s15, s15, s50
	s_addc_u32 s26, 0, s26
	s_add_u32 s15, s49, s15
	s_addc_u32 s49, s27, s26
	v_xor_b32_e32 v11, v4, v80
	v_xor_b32_e32 v6, v5, v80
	v_mad_u64_u32 v[4:5], s[26:27], v11, s49, 0
	v_mul_hi_u32 v12, v11, s15
	v_lshl_add_u64 v[4:5], v[12:13], 0, v[4:5]
	v_mad_u64_u32 v[82:83], s[26:27], v6, s15, 0
	v_add_co_u32_e32 v4, vcc, v4, v82
	v_mad_u64_u32 v[12:13], s[26:27], v6, s49, 0
	s_nop 0
	v_addc_co_u32_e32 v4, vcc, v5, v83, vcc
	v_mov_b32_e32 v5, s14
	s_nop 0
	v_addc_co_u32_e32 v13, vcc, 0, v13, vcc
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[12:13]
	v_mul_lo_u32 v12, s35, v4
	v_mul_lo_u32 v13, s34, v5
	v_mad_u64_u32 v[4:5], s[14:15], s34, v4, 0
	v_add3_u32 v5, v5, v13, v12
	v_sub_u32_e32 v12, v6, v5
	v_mov_b32_e32 v13, s35
	v_sub_co_u32_e32 v4, vcc, v11, v4
	s_nop 1
	v_subb_co_u32_e64 v11, s[14:15], v12, v13, vcc
	v_subrev_co_u32_e64 v12, s[14:15], s34, v4
	v_subb_co_u32_e32 v5, vcc, v6, v5, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v63, s[26:27], 0, v11, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v63
	v_subb_co_u32_e64 v11, s[14:15], v11, v13, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v12
	v_subrev_co_u32_e64 v13, s[14:15], s34, v12
	s_nop 0
	v_cndmask_b32_e64 v81, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v63
	v_subbrev_co_u32_e64 v11, s[14:15], 0, v11, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v65, v65, v81, s[26:27]
	v_cmp_le_u32_e32 vcc, s35, v5
	v_cmp_ne_u32_e64 s[14:15], 0, v65
	s_nop 0
	v_cndmask_b32_e64 v6, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v4
	v_cndmask_b32_e64 v11, v63, v11, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v5
	s_nop 1
	v_cndmask_b32_e32 v6, v6, v63, vcc
	v_cmp_ne_u32_e32 vcc, 0, v6
	v_cndmask_b32_e64 v6, v12, v13, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v4, v4, v6, vcc
	v_cndmask_b32_e32 v5, v5, v11, vcc
	v_xor_b32_e32 v4, v4, v80
	v_xor_b32_e32 v5, v5, v80
	v_sub_co_u32_e32 v12, vcc, v4, v80
	s_nop 1
	v_subb_co_u32_e32 v13, vcc, v5, v80, vcc
.LBB0_24:
	s_andn2_saveexec_b64 s[14:15], s[28:29]
	s_cbranch_execz .LBB0_26
	v_rcp_iflag_f32_e32 v5, v59
	s_sub_i32 s26, 0, s76
	v_mov_b32_e32 v13, 0
	v_mul_f32_e32 v5, 0x4f7ffffe, v5
	v_cvt_u32_f32_e32 v5, v5
	v_mul_lo_u32 v6, s26, v5
	v_mul_hi_u32 v6, v5, v6
	v_add_u32_e32 v5, v5, v6
	v_mul_hi_u32 v5, v4, v5
	v_mul_lo_u32 v5, v5, s76
	v_sub_u32_e32 v4, v4, v5
	v_subrev_u32_e32 v5, s76, v4
	v_cmp_le_u32_e32 vcc, s76, v4
	s_nop 1
	v_cndmask_b32_e32 v4, v4, v5, vcc
	v_subrev_u32_e32 v5, s76, v4
	v_cmp_le_u32_e32 vcc, s76, v4
	s_nop 1
	v_cndmask_b32_e32 v12, v4, v5, vcc
.LBB0_26:
	s_or_b64 exec, exec, s[14:15]
	v_ashrrev_i32_e32 v4, 31, v7
	v_xor_b32_e32 v4, v10, v4
	v_ashrrev_i32_e32 v5, 31, v13
	v_mul_lo_u32 v4, v4, s54
	v_and_b32_e32 v5, s76, v5
	v_add3_u32 v4, v4, v12, v5
	buffer_load_ubyte v114, v4, s[8:11], 0 offen
	v_readlane_b32 s14, v202, 18
	s_lshl_b32 s61, s14, 7
	v_readlane_b32 s14, v202, 20
	s_lshl_b32 s66, s14, 7
	v_readlane_b32 s14, v202, 21
	v_mov_b32_e32 v11, 0
	s_lshl_b32 s67, s14, 7
	s_mov_b32 s34, 0
	s_waitcnt vmcnt(0)
	s_barrier
	v_or_b32_e32 v10, 0x100, v52
	v_or_b32_e32 v4, 0x80, v48
	v_cmp_gt_i64_e32 vcc, s[30:31], v[10:11]
	v_add_u32_e32 v5, v4, v50
	s_and_b64 s[14:15], vcc, s[2:3]
	s_add_i32 s60, s52, 0x4000
	v_cndmask_b32_e64 v5, -1, v5, s[14:15]
	s_mov_b32 m0, s60
	s_and_b64 s[14:15], vcc, s[24:25]
	buffer_load_dwordx4 v5, s[40:43], 0 offen lds
	v_add_u32_e32 v5, v4, v54
	s_addk_i32 s61, 0x4000
	v_cndmask_b32_e64 v5, -1, v5, s[14:15]
	s_mov_b32 m0, s61
	s_and_b64 s[14:15], vcc, s[18:19]
	buffer_load_dwordx4 v5, s[40:43], 0 offen lds
	v_add_u32_e32 v5, v4, v56
	s_addk_i32 s66, 0x4000
	v_cndmask_b32_e64 v5, -1, v5, s[14:15]
	s_mov_b32 m0, s66
	v_add_u32_e32 v4, v4, v58
	s_and_b64 vcc, vcc, s[20:21]
	s_addk_i32 s67, 0x4000
	buffer_load_dwordx4 v5, s[40:43], 0 offen lds
	v_cndmask_b32_e32 v4, -1, v4, vcc
	s_mov_b32 m0, s67
	s_mov_b64 s[14:15], 0x2000
	buffer_load_dwordx4 v4, s[40:43], 0 offen lds
	v_lshl_add_u64 v[4:5], v[60:61], 0, s[14:15]
	s_movk_i32 s14, 0xdfff
	v_sub_co_u32_e32 v6, vcc, s14, v60
	s_nop 1
	v_subb_co_u32_e32 v7, vcc, -1, v61, vcc
	v_cmp_gt_i32_e32 vcc, 0, v5
	s_nop 1
	v_cndmask_b32_e32 v13, v5, v7, vcc
	v_cndmask_b32_e32 v12, v4, v6, vcc
	v_or_b32_e32 v6, s31, v13
	v_cmp_ne_u32_e32 vcc, 0, v6
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[26:27], exec, s[14:15]
	s_cbranch_execz .LBB0_28
	s_add_u32 s14, s30, s80
	s_mov_b32 s81, s80
	s_addc_u32 s15, s31, s80
	s_xor_b64 s[28:29], s[14:15], s[80:81]
	v_cvt_f32_u32_e32 v6, s28
	v_cvt_f32_u32_e32 v7, s29
	s_sub_u32 s14, 0, s28
	s_subb_u32 s15, 0, s29
	v_fmamk_f32 v6, v7, 0x4f800000, v6
	v_rcp_f32_e32 v6, v6
	s_nop 0
	v_mul_f32_e32 v6, 0x5f7ffffc, v6
	v_mul_f32_e32 v7, 0x2f800000, v6
	v_trunc_f32_e32 v7, v7
	v_fmamk_f32 v6, v7, 0xcf800000, v6
	v_cvt_u32_f32_e32 v7, v7
	v_cvt_u32_f32_e32 v6, v6
	v_readfirstlane_b32 s35, v7
	v_readfirstlane_b32 s49, v6
	s_mul_i32 s50, s14, s35
	s_mul_hi_u32 s55, s14, s49
	s_mul_i32 s51, s15, s49
	s_add_i32 s50, s55, s50
	s_add_i32 s50, s50, s51
	s_mul_i32 s56, s14, s49
	s_mul_i32 s55, s49, s50
	s_mul_hi_u32 s57, s49, s56
	s_mul_hi_u32 s51, s49, s50
	s_add_u32 s55, s57, s55
	s_addc_u32 s51, 0, s51
	s_mul_hi_u32 s58, s35, s56
	s_mul_i32 s56, s35, s56
	s_add_u32 s55, s55, s56
	s_mul_hi_u32 s57, s35, s50
	s_addc_u32 s51, s51, s58
	s_addc_u32 s55, s57, 0
	s_mul_i32 s50, s35, s50
	s_add_u32 s50, s51, s50
	s_addc_u32 s51, 0, s55
	s_add_u32 s49, s49, s50
	s_addc_u32 s35, s35, s51
	s_mul_i32 s50, s14, s35
	s_mul_hi_u32 s51, s14, s49
	s_add_i32 s50, s51, s50
	s_mul_i32 s15, s15, s49
	s_add_i32 s50, s50, s15
	s_mul_i32 s14, s14, s49
	s_mul_hi_u32 s51, s35, s14
	s_mul_i32 s55, s35, s14
	s_mul_i32 s57, s49, s50
	s_mul_hi_u32 s14, s49, s14
	s_mul_hi_u32 s56, s49, s50
	s_add_u32 s14, s14, s57
	s_addc_u32 s56, 0, s56
	s_add_u32 s14, s14, s55
	s_mul_hi_u32 s15, s35, s50
	s_addc_u32 s14, s56, s51
	s_addc_u32 s15, s15, 0
	s_mul_i32 s50, s35, s50
	s_add_u32 s14, s14, s50
	v_ashrrev_i32_e32 v6, 31, v13
	s_addc_u32 s15, 0, s15
	v_mov_b32_e32 v7, v6
	s_add_u32 s49, s49, s14
	v_lshl_add_u64 v[12:13], v[12:13], 0, v[6:7]
	s_addc_u32 s35, s35, s15
	v_xor_b32_e32 v63, v12, v6
	v_xor_b32_e32 v7, v13, v6
	v_mad_u64_u32 v[12:13], s[14:15], v63, s35, 0
	v_mul_hi_u32 v10, v63, s49
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[12:13]
	v_mad_u64_u32 v[80:81], s[14:15], v7, s49, 0
	v_add_co_u32_e32 v10, vcc, v10, v80
	v_mad_u64_u32 v[12:13], s[14:15], v7, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v10, vcc, v11, v81, vcc
	v_mov_b32_e32 v11, s34
	s_nop 0
	v_addc_co_u32_e32 v13, vcc, 0, v13, vcc
	v_lshl_add_u64 v[10:11], v[10:11], 0, v[12:13]
	v_mul_lo_u32 v65, s29, v10
	v_mul_lo_u32 v80, s28, v11
	v_mad_u64_u32 v[12:13], s[14:15], s28, v10, 0
	v_add3_u32 v65, v13, v80, v65
	v_sub_u32_e32 v13, v7, v65
	v_mov_b32_e32 v80, s29
	v_sub_co_u32_e32 v63, vcc, v63, v12
	v_xor_b32_e32 v6, s80, v6
	s_nop 0
	v_subb_co_u32_e64 v12, s[14:15], v13, v80, vcc
	v_subrev_co_u32_e64 v13, s[14:15], s28, v63
	v_subb_co_u32_e32 v7, vcc, v7, v65, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v12, s[14:15], 0, v12, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s29, v12
	v_cmp_le_u32_e32 vcc, s29, v7
	s_nop 0
	v_cndmask_b32_e64 v80, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v13
	s_nop 1
	v_cndmask_b32_e64 v13, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v12
	s_nop 1
	v_cndmask_b32_e64 v82, v80, v13, s[14:15]
	v_lshl_add_u64 v[12:13], v[10:11], 0, 2
	v_lshl_add_u64 v[80:81], v[10:11], 0, 1
	v_cndmask_b32_e64 v11, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v63
	s_nop 1
	v_cndmask_b32_e64 v13, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v7
	s_nop 1
	v_cndmask_b32_e32 v7, v11, v13, vcc
	v_cmp_ne_u32_e32 vcc, 0, v82
	s_nop 1
	v_cndmask_b32_e32 v11, v80, v12, vcc
	v_cmp_ne_u32_e32 vcc, 0, v7
	s_nop 1
	v_cndmask_b32_e32 v7, v10, v11, vcc
	v_xor_b32_e32 v7, v7, v6
	v_sub_co_u32_e32 v6, vcc, v7, v6
.LBB0_28:
	s_andn2_saveexec_b64 s[14:15], s[26:27]
	s_cbranch_execz .LBB0_30
	v_cvt_f32_u32_e32 v6, s30
	s_sub_i32 s26, 0, s30
	v_rcp_iflag_f32_e32 v6, v6
	s_nop 0
	v_mul_f32_e32 v6, 0x4f7ffffe, v6
	v_cvt_u32_f32_e32 v6, v6
	v_mul_lo_u32 v7, s26, v6
	v_mul_hi_u32 v7, v6, v7
	v_add_u32_e32 v6, v6, v7
	v_mul_hi_u32 v6, v12, v6
	v_mul_lo_u32 v7, v6, s30
	v_sub_u32_e32 v7, v12, v7
	v_add_u32_e32 v10, 1, v6
	v_subrev_u32_e32 v11, s30, v7
	v_cmp_le_u32_e32 vcc, s30, v7
	s_nop 1
	v_cndmask_b32_e32 v7, v7, v11, vcc
	v_cndmask_b32_e32 v6, v6, v10, vcc
	v_add_u32_e32 v10, 1, v6
	v_cmp_le_u32_e32 vcc, s30, v7
	s_nop 1
	v_cndmask_b32_e32 v6, v6, v10, vcc
.LBB0_30:
	s_or_b64 exec, exec, s[14:15]
	v_or_b32_e32 v7, s31, v5
	v_lshlrev_b64 v[0:1], 8, v[0:1]
	s_mov_b32 s14, 0
	v_cmp_ne_u32_e32 vcc, 0, v7
	v_ashrrev_i32_e32 v10, 31, v5
	s_and_saveexec_b64 s[26:27], vcc
	s_xor_b64 s[28:29], exec, s[26:27]
	s_cbranch_execz .LBB0_32
	s_add_u32 s26, s30, s80
	s_mov_b32 s81, s80
	s_addc_u32 s27, s31, s80
	s_xor_b64 s[34:35], s[26:27], s[80:81]
	v_cvt_f32_u32_e32 v7, s34
	v_cvt_f32_u32_e32 v11, s35
	s_sub_u32 s15, 0, s34
	s_subb_u32 s26, 0, s35
	v_mov_b32_e32 v13, 0
	v_fmamk_f32 v7, v11, 0x4f800000, v7
	v_rcp_f32_e32 v7, v7
	s_nop 0
	v_mul_f32_e32 v7, 0x5f7ffffc, v7
	v_mul_f32_e32 v11, 0x2f800000, v7
	v_trunc_f32_e32 v11, v11
	v_fmamk_f32 v7, v11, 0xcf800000, v7
	v_cvt_u32_f32_e32 v11, v11
	v_cvt_u32_f32_e32 v7, v7
	v_readfirstlane_b32 s27, v11
	v_readfirstlane_b32 s49, v7
	s_mul_i32 s50, s15, s27
	s_mul_hi_u32 s55, s15, s49
	s_mul_i32 s51, s26, s49
	s_add_i32 s50, s55, s50
	s_add_i32 s50, s50, s51
	s_mul_i32 s56, s15, s49
	s_mul_i32 s55, s49, s50
	s_mul_hi_u32 s57, s49, s56
	s_mul_hi_u32 s51, s49, s50
	s_add_u32 s55, s57, s55
	s_addc_u32 s51, 0, s51
	s_mul_hi_u32 s58, s27, s56
	s_mul_i32 s56, s27, s56
	s_add_u32 s55, s55, s56
	s_mul_hi_u32 s57, s27, s50
	s_addc_u32 s51, s51, s58
	s_addc_u32 s55, s57, 0
	s_mul_i32 s50, s27, s50
	s_add_u32 s50, s51, s50
	s_addc_u32 s51, 0, s55
	s_add_u32 s49, s49, s50
	s_addc_u32 s27, s27, s51
	s_mul_i32 s50, s15, s27
	s_mul_hi_u32 s51, s15, s49
	s_add_i32 s50, s51, s50
	s_mul_i32 s26, s26, s49
	s_add_i32 s50, s50, s26
	s_mul_i32 s15, s15, s49
	s_mul_hi_u32 s51, s27, s15
	s_mul_i32 s55, s27, s15
	s_mul_i32 s57, s49, s50
	s_mul_hi_u32 s15, s49, s15
	s_mul_hi_u32 s56, s49, s50
	s_add_u32 s15, s15, s57
	s_addc_u32 s56, 0, s56
	s_add_u32 s15, s15, s55
	s_mul_hi_u32 s26, s27, s50
	s_addc_u32 s15, s56, s51
	s_addc_u32 s26, s26, 0
	s_mul_i32 s50, s27, s50
	s_add_u32 s15, s15, s50
	s_addc_u32 s26, 0, s26
	v_mov_b32_e32 v11, v10
	s_add_u32 s15, s49, s15
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[10:11]
	s_addc_u32 s49, s27, s26
	v_xor_b32_e32 v11, v4, v10
	v_xor_b32_e32 v7, v5, v10
	v_mad_u64_u32 v[4:5], s[26:27], v11, s49, 0
	v_mul_hi_u32 v12, v11, s15
	v_lshl_add_u64 v[4:5], v[12:13], 0, v[4:5]
	v_mad_u64_u32 v[80:81], s[26:27], v7, s15, 0
	v_add_co_u32_e32 v4, vcc, v4, v80
	v_mad_u64_u32 v[12:13], s[26:27], v7, s49, 0
	s_nop 0
	v_addc_co_u32_e32 v4, vcc, v5, v81, vcc
	v_mov_b32_e32 v5, s14
	s_nop 0
	v_addc_co_u32_e32 v13, vcc, 0, v13, vcc
	v_lshl_add_u64 v[4:5], v[4:5], 0, v[12:13]
	v_mul_lo_u32 v12, s35, v4
	v_mul_lo_u32 v13, s34, v5
	v_mad_u64_u32 v[4:5], s[14:15], s34, v4, 0
	v_add3_u32 v5, v5, v13, v12
	v_sub_u32_e32 v12, v7, v5
	v_mov_b32_e32 v13, s35
	v_sub_co_u32_e32 v4, vcc, v11, v4
	s_nop 1
	v_subb_co_u32_e64 v11, s[14:15], v12, v13, vcc
	v_subrev_co_u32_e64 v12, s[14:15], s34, v4
	v_subb_co_u32_e32 v5, vcc, v7, v5, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v63, s[26:27], 0, v11, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v63
	v_subb_co_u32_e64 v11, s[14:15], v11, v13, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v12
	v_subrev_co_u32_e64 v13, s[14:15], s34, v12
	s_nop 0
	v_cndmask_b32_e64 v80, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v63
	v_subbrev_co_u32_e64 v11, s[14:15], 0, v11, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v65, v65, v80, s[26:27]
	v_cmp_le_u32_e32 vcc, s35, v5
	v_cmp_ne_u32_e64 s[14:15], 0, v65
	s_nop 0
	v_cndmask_b32_e64 v7, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v4
	v_cndmask_b32_e64 v11, v63, v11, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v5
	s_nop 1
	v_cndmask_b32_e32 v7, v7, v63, vcc
	v_cmp_ne_u32_e32 vcc, 0, v7
	v_cndmask_b32_e64 v7, v12, v13, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v4, v4, v7, vcc
	v_cndmask_b32_e32 v5, v5, v11, vcc
	v_xor_b32_e32 v4, v4, v10
	v_xor_b32_e32 v5, v5, v10
	v_sub_co_u32_e32 v12, vcc, v4, v10
	s_nop 1
	v_subb_co_u32_e32 v13, vcc, v5, v10, vcc
.LBB0_32:
	s_or_saveexec_b64 s[14:15], s[28:29]
	v_and_b32_e32 v5, 7, v40
	s_xor_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB0_34
	v_cvt_f32_u32_e32 v7, s30
	s_sub_i32 s26, 0, s30
	v_mov_b32_e32 v13, 0
	v_rcp_iflag_f32_e32 v7, v7
	s_nop 0
	v_mul_f32_e32 v7, 0x4f7ffffe, v7
	v_cvt_u32_f32_e32 v7, v7
	v_mul_lo_u32 v11, s26, v7
	v_mul_hi_u32 v11, v7, v11
	v_add_u32_e32 v7, v7, v11
	v_mul_hi_u32 v7, v4, v7
	v_mul_lo_u32 v7, v7, s30
	v_sub_u32_e32 v4, v4, v7
	v_subrev_u32_e32 v7, s30, v4
	v_cmp_le_u32_e32 vcc, s30, v4
	s_nop 1
	v_cndmask_b32_e32 v4, v4, v7, vcc
	v_subrev_u32_e32 v7, s30, v4
	v_cmp_le_u32_e32 vcc, s30, v4
	s_nop 1
	v_cndmask_b32_e32 v12, v4, v7, vcc
.LBB0_34:
	s_or_b64 exec, exec, s[14:15]
	v_sub_co_u32_e32 v102, vcc, 0, v0
	v_lshlrev_b16_e32 v0, 8, v28
	s_nop 0
	v_subb_co_u32_e32 v103, vcc, 0, v1, vcc
	v_lshlrev_b16_e32 v1, 8, v29
	v_or_b32_e32 v0, v19, v0
	v_or_b32_sdwa v1, v27, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_xor_b32_e32 v4, v6, v10
	v_or_b32_sdwa v19, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v0, 8, v18
	v_lshlrev_b16_e32 v1, 8, v26
	v_or_b32_e32 v0, v14, v0
	v_or_b32_sdwa v1, v15, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_mov_b32 s14, 0
	v_or_b32_sdwa v18, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v0, 8, v24
	v_lshlrev_b16_e32 v1, 8, v25
	v_or_b32_e32 v0, v17, v0
	v_or_b32_sdwa v1, v23, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_mov_b32_e32 v7, s14
	v_or_b32_sdwa v17, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v0, 8, v21
	v_lshlrev_b16_e32 v1, 8, v22
	v_or_b32_e32 v0, v16, v0
	v_or_b32_sdwa v1, v20, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_readlane_b32 s14, v202, 22
	v_or_b32_sdwa v16, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v0, 8, v78
	v_lshlrev_b16_e32 v1, 8, v79
	v_or_b32_e32 v0, v31, v0
	v_or_b32_sdwa v1, v77, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_add_i32 s56, s14, 0xc400
	v_or_b32_sdwa v31, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v0, 8, v57
	v_lshlrev_b16_e32 v1, 8, v76
	v_or_b32_e32 v0, v30, v0
	v_or_b32_sdwa v1, v55, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_cmp_lt_i64_e64 s[14:15], s[16:17], 1
	v_or_b32_sdwa v30, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v0, 8, v38
	v_lshlrev_b16_e32 v1, 8, v39
	v_or_b32_e32 v0, v36, v0
	v_or_b32_sdwa v1, v37, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_and_b64 vcc, s[22:23], s[14:15]
	v_or_b32_sdwa v29, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v0, 8, v34
	v_lshlrev_b16_e32 v1, 8, v35
	v_or_b32_e32 v0, v32, v0
	v_or_b32_sdwa v1, v33, v1 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_cmp_gt_i64_e64 s[14:15], s[16:17], 0
	v_or_b32_sdwa v28, v0, v1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_ashrrev_i32_e32 v0, 31, v13
	v_and_b32_e32 v1, s31, v0
	v_and_b32_e32 v0, s30, v0
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[12:13]
	v_ashrrev_i32_e32 v10, 31, v1
	v_xor_b32_e32 v1, v10, v1
	v_ashrrev_i32_e32 v6, 31, v1
	v_xor_b32_e32 v0, v10, v0
	v_lshrrev_b32_e32 v6, 27, v6
	v_lshl_add_u64 v[0:1], v[0:1], 0, v[6:7]
	v_cndmask_b32_e64 v6, 0, -1, s[14:15]
	v_alignbit_b32 v0, v1, v0, 5
	v_readfirstlane_b32 s14, v6
	s_mov_b32 s15, s14
	s_xor_b64 s[22:23], s[0:1], s[14:15]
	v_readlane_b32 s0, v202, 16
	v_readlane_b32 s1, v202, 17
	s_ashr_i32 s14, s1, 31
	s_add_u32 s0, s0, s14
	s_mov_b32 s15, s14
	s_addc_u32 s1, s1, s14
	s_xor_b64 s[16:17], s[0:1], s[14:15]
	v_cvt_f32_u32_e32 v7, s16
	v_cvt_f32_u32_e32 v11, s17
	v_mul_lo_u32 v1, v4, s46
	v_xad_u32 v0, v0, v10, v1
	s_mov_b32 m0, s56
	v_fmamk_f32 v1, v11, 0x4f800000, v7
	v_rcp_f32_e32 v1, v1
	buffer_load_dword v0, s[4:7], 0 offen lds
	s_sub_u32 s26, 0, s16
	s_subb_u32 s27, 0, s17
	v_mul_f32_e32 v0, 0x5f7ffffc, v1
	v_mul_f32_e32 v1, 0x2f800000, v0
	v_trunc_f32_e32 v1, v1
	v_fmamk_f32 v0, v1, 0xcf800000, v0
	v_cvt_u32_f32_e32 v0, v0
	v_cvt_u32_f32_e32 v1, v1
	v_xor_b32_e32 v4, v97, v5
	s_mov_b64 s[0:1], 0
	v_readfirstlane_b32 s28, v0
	v_readfirstlane_b32 s35, v1
	s_mul_hi_u32 s34, s26, s28
	s_mul_i32 s49, s26, s35
	s_mul_i32 s29, s27, s28
	s_add_i32 s34, s34, s49
	s_add_i32 s34, s34, s29
	s_mul_i32 s50, s26, s28
	s_mul_i32 s49, s28, s34
	s_mul_hi_u32 s51, s28, s50
	s_mul_hi_u32 s29, s28, s34
	s_add_u32 s49, s51, s49
	s_addc_u32 s29, 0, s29
	s_mul_hi_u32 s55, s35, s50
	s_mul_i32 s50, s35, s50
	s_add_u32 s49, s49, s50
	s_mul_hi_u32 s51, s35, s34
	s_addc_u32 s29, s29, s55
	s_addc_u32 s49, s51, 0
	s_mul_i32 s34, s35, s34
	s_add_u32 s29, s29, s34
	s_addc_u32 s34, 0, s49
	s_add_u32 s28, s28, s29
	s_addc_u32 s29, s35, s34
	s_mul_i32 s34, s26, s29
	s_mul_hi_u32 s35, s26, s28
	s_add_i32 s34, s35, s34
	s_mul_i32 s27, s27, s28
	s_add_i32 s34, s34, s27
	s_mul_i32 s26, s26, s28
	s_mul_hi_u32 s35, s29, s26
	s_mul_i32 s49, s29, s26
	s_mul_i32 s51, s28, s34
	s_mul_hi_u32 s26, s28, s26
	s_mul_hi_u32 s50, s28, s34
	s_add_u32 s26, s26, s51
	s_addc_u32 s50, 0, s50
	s_add_u32 s26, s26, s49
	s_mul_hi_u32 s27, s29, s34
	s_addc_u32 s26, s50, s35
	s_addc_u32 s27, s27, 0
	s_mul_i32 s34, s29, s34
	s_add_u32 s26, s26, s34
	s_addc_u32 s27, 0, s27
	s_add_u32 s28, s28, s26
	s_addc_u32 s29, s29, s27
	s_ashr_i32 s26, s23, 31
	s_add_u32 s22, s22, s26
	s_mov_b32 s27, s26
	s_addc_u32 s23, s23, s26
	s_xor_b64 s[22:23], s[22:23], s[26:27]
	s_mul_i32 s35, s22, s29
	s_mul_hi_u32 s49, s22, s28
	s_mul_hi_u32 s34, s22, s29
	s_add_u32 s35, s49, s35
	s_addc_u32 s34, 0, s34
	s_mul_hi_u32 s50, s23, s28
	s_mul_i32 s28, s23, s28
	s_add_u32 s28, s35, s28
	s_mul_hi_u32 s49, s23, s29
	s_addc_u32 s28, s34, s50
	s_addc_u32 s34, s49, 0
	s_mul_i32 s29, s23, s29
	s_add_u32 s35, s28, s29
	s_addc_u32 s34, 0, s34
	s_mul_i32 s28, s16, s34
	s_mul_hi_u32 s29, s16, s35
	s_add_i32 s28, s29, s28
	s_mul_i32 s29, s17, s35
	s_add_i32 s49, s28, s29
	s_sub_i32 s50, s23, s49
	s_mul_i32 s28, s16, s35
	s_sub_u32 s22, s22, s28
	s_cselect_b64 s[28:29], -1, 0
	s_subb_u32 s50, s50, s17
	s_sub_u32 s51, s22, s16
	s_subb_u32 s50, s50, 0
	s_cmp_ge_u32 s50, s17
	s_cselect_b32 s55, -1, 0
	s_cmp_ge_u32 s51, s16
	s_cselect_b32 s51, -1, 0
	s_cmp_eq_u32 s50, s17
	s_cselect_b32 s50, s51, s55
	s_add_u32 s51, s35, 1
	s_addc_u32 s55, s34, 0
	s_add_u32 s57, s35, 2
	s_addc_u32 s58, s34, 0
	s_cmp_lg_u32 s50, 0
	s_cselect_b32 s50, s57, s51
	s_cselect_b32 s51, s58, s55
	s_cmp_lg_u64 s[28:29], 0
	s_subb_u32 s23, s23, s49
	s_cmp_ge_u32 s23, s17
	s_cselect_b32 s28, -1, 0
	s_cmp_ge_u32 s22, s16
	s_cselect_b32 s16, -1, 0
	s_cmp_eq_u32 s23, s17
	s_cselect_b32 s16, s16, s28
	s_cmp_lg_u32 s16, 0
	s_cselect_b32 s17, s51, s34
	s_cselect_b32 s16, s50, s35
	s_xor_b64 s[14:15], s[26:27], s[14:15]
	s_xor_b64 s[16:17], s[16:17], s[14:15]
	s_sub_u32 s14, s16, s14
	s_subb_u32 s15, s17, s15
	v_xor_b32_e32 v0, s14, v6
	s_ashr_i32 s14, s85, 31
	v_xor_b32_e32 v1, s15, v6
	s_mov_b32 s15, s14
	s_xor_b64 s[16:17], s[14:15], s[84:85]
	s_mul_hi_u32 s22, s16, 0x2aaaaaaa
	s_mul_i32 s23, s16, 0x2aaaaaaa
	s_mul_i32 s27, s17, 0xaaaaaaab
	s_mul_hi_u32 s16, s16, 0xaaaaaaab
	s_mul_hi_u32 s26, s17, 0xaaaaaaab
	s_add_u32 s16, s27, s16
	s_addc_u32 s26, s26, 0
	s_add_u32 s16, s23, s16
	s_addc_u32 s16, s22, 0
	s_add_u32 s16, s26, s16
	s_addc_u32 s22, 0, 0
	s_mul_i32 s26, s17, 0x2aaaaaaa
	v_cndmask_b32_e32 v1, 0, v1, vcc
	v_cndmask_b32_e32 v0, 0, v0, vcc
	s_mul_hi_u32 s23, s17, 0x2aaaaaaa
	s_add_u32 s16, s26, s16
	v_lshlrev_b64 v[6:7], 10, v[0:1]
	s_addc_u32 s22, s23, s22
	s_ashr_i32 s17, s17, 31
	v_or_b32_e32 v63, v6, v49
	v_lshlrev_b32_e32 v6, 11, v97
	s_mul_i32 s23, s17, 0x2aaaaaaa
	s_mul_hi_u32 s26, s17, 0xaaaaaaab
	v_sub_u32_e32 v6, v42, v6
	v_mov_b32_e32 v7, 0
	v_lshlrev_b64 v[0:1], 14, v[0:1]
	s_add_i32 s23, s26, s23
	s_mul_i32 s17, s17, 0xaaaaaaab
	v_lshl_add_u64 v[76:77], v[0:1], 0, v[6:7]
	s_add_i32 s23, s23, s17
	v_lshl_or_b32 v78, v4, 4, v76
	v_bitop3_b32 v0, v97, v5, 4 bitop3:0x36
	s_add_u32 s16, s16, s17
	v_lshl_or_b32 v76, v0, 4, v76
	ds_read_b128 v[126:129], v78
	ds_read_b128 v[170:173], v78 offset:2048
	ds_read_b128 v[166:169], v76
	ds_read_b128 v[162:165], v76 offset:2048
	ds_read_b128 v[122:125], v78 offset:4096
	ds_read_b128 v[154:157], v78 offset:6144
	ds_read_b128 v[158:161], v76 offset:4096
	ds_read_b128 v[150:153], v76 offset:6144
	s_addc_u32 s17, s22, s23
	s_lshr_b32 s22, s17, 31
	s_add_u32 s16, s16, s22
	s_addc_u32 s17, s17, 0
	s_xor_b64 s[14:15], s[16:17], s[14:15]
	s_mul_i32 s16, s15, 6
	s_mul_hi_u32 s17, s14, 6
	ds_read2st64_b32 v[116:117], v63 offset0:192 offset1:193
	v_mov_b32_e32 v79, v77
	s_add_i32 s83, s17, s16
	s_mov_b64 s[16:17], 0x2000
	v_lshl_add_u64 v[80:81], v[78:79], 0, s[16:17]
	v_lshl_add_u64 v[82:83], v[76:77], 0, s[16:17]
	s_mov_b64 s[16:17], 0x2800
	v_lshl_add_u64 v[84:85], v[78:79], 0, s[16:17]
	v_lshl_add_u64 v[86:87], v[76:77], 0, s[16:17]
	s_mov_b64 s[16:17], 0x3000
	s_mul_i32 s82, s14, 6
	v_lshl_add_u64 v[88:89], v[78:79], 0, s[16:17]
	v_lshl_add_u64 v[90:91], v[76:77], 0, s[16:17]
	s_mov_b64 s[16:17], 0x3800
	v_cmp_gt_i64_e64 s[14:15], s[14:15], 0
	v_lshl_add_u64 v[92:93], v[78:79], 0, s[16:17]
	v_lshl_add_u64 v[94:95], v[76:77], 0, s[16:17]
	v_or_b32_e32 v96, 0x200, v63
	v_or_b32_e32 v98, 0x300, v63
	s_and_b64 vcc, exec, s[14:15]
	s_cbranch_vccz .LBB0_36
	s_mov_b64 s[0:1], -1
.LBB0_36:
	s_andn2_b64 vcc, exec, s[0:1]
	v_mov_b32_e32 v105, 0
	s_cbranch_vccnz .LBB0_87
	v_writelane_b32 v202, s84, 34
	v_mad_u64_u32 v[0:1], s[0:1], s48, v3, 0
	s_nop 0
	v_writelane_b32 v202, s85, 35
	v_add3_u32 v1, v1, v53, v51
	s_add_u32 s84, s82, -2
	s_mov_b64 s[86:87], 0x400
	v_or_b32_e32 v0, v0, v66
	v_mov_b32_e32 v4, 0
	s_addc_u32 s85, s83, -1
	v_lshl_add_u64 v[106:107], v[8:9], 0, s[86:87]
	v_or_b32_e32 v108, v2, v47
	v_lshl_add_u64 v[110:111], v[0:1], 0, v[68:69]
	v_lshl_add_u64 v[112:113], v[66:67], 0, v[68:69]
	s_mov_b64 s[88:89], 0
	s_mov_b64 s[90:91], 0x800
	s_mov_b32 s57, 0
	s_mov_b32 s92, s70
	s_mov_b32 s93, s70
	s_mov_b64 s[94:95], 0xc00
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v6, v4
	v_mov_b32_e32 v7, v4
	v_mov_b32_e32 v0, v4
	v_mov_b32_e32 v1, v4
	v_mov_b32_e32 v2, v4
	v_mov_b32_e32 v3, v4
	v_mov_b32_e32 v36, v4
	v_mov_b32_e32 v37, v4
	v_mov_b32_e32 v38, v4
	v_mov_b32_e32 v39, v4
	v_mov_b32_e32 v8, v4
	v_mov_b32_e32 v9, v4
	v_mov_b32_e32 v10, v4
	v_mov_b32_e32 v11, v4
	v_mov_b32_e32 v20, v4
	v_mov_b32_e32 v21, v4
	v_mov_b32_e32 v22, v4
	v_mov_b32_e32 v23, v4
	v_mov_b32_e32 v12, v4
	v_mov_b32_e32 v13, v4
	v_mov_b32_e32 v14, v4
	v_mov_b32_e32 v15, v4
	v_mov_b32_e32 v24, v4
	v_mov_b32_e32 v25, v4
	v_mov_b32_e32 v26, v4
	v_mov_b32_e32 v27, v4
	v_mov_b32_e32 v32, v4
	v_mov_b32_e32 v33, v4
	v_mov_b32_e32 v34, v4
	v_mov_b32_e32 v35, v4
	s_mov_b64 s[50:51], 0x100
	s_mov_b64 s[58:59], 0x200
	s_mov_b64 s[0:1], 0x202
	s_mov_b64 s[16:17], 0x300
	s_mov_b64 s[22:23], 0x6000
	s_branch .LBB0_39
.LBB0_38:
	s_or_b64 exec, exec, s[14:15]
	v_ashrrev_i32_e32 v16, 31, v31
	v_and_b32_e32 v17, s31, v16
	v_and_b32_e32 v16, s30, v16
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[30:31]
	v_ashrrev_i32_e32 v19, 31, v17
	v_xor_b32_e32 v17, v19, v17
	v_xor_b32_e32 v18, v18, v28
	v_ashrrev_i32_e32 v28, 31, v17
	v_xor_b32_e32 v16, v19, v16
	v_lshrrev_b32_e32 v104, 27, v28
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[104:105]
	v_alignbit_b32 v16, v17, v16, 5
	v_mul_lo_u32 v17, v18, s46
	v_xad_u32 v16, v16, v19, v17
	s_mov_b32 m0, s56
	s_nop 0
	buffer_load_dword v16, s[4:7], 0 offen lds
	s_waitcnt vmcnt(5) lgkmcnt(0)
	s_barrier
	s_add_u32 s88, s88, 2
	s_addc_u32 s89, s89, 0
	v_mov_b64_e32 v[16:17], s[84:85]
	v_cmp_lt_i64_e32 vcc, s[88:89], v[16:17]
	v_lshlrev_b16_e32 v16, 8, v51
	v_lshlrev_b16_e32 v17, 8, v55
	v_or_b32_e32 v16, v47, v16
	v_or_b32_sdwa v17, v53, v17 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v18, 8, v79
	v_or_b32_sdwa v16, v16, v17 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v17, 8, v65
	v_or_b32_e32 v17, v57, v17
	v_or_b32_sdwa v18, v77, v18 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v19, 8, v89
	v_or_b32_sdwa v17, v17, v18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v18, 8, v85
	v_or_b32_e32 v18, v83, v18
	v_or_b32_sdwa v19, v87, v19 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v28, 8, v109
	v_or_b32_sdwa v18, v18, v19 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v19, 8, v93
	v_or_b32_e32 v19, v91, v19
	v_or_b32_sdwa v28, v95, v28 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v29, 8, v135
	v_or_b32_sdwa v19, v19, v28 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v28, 8, v133
	v_or_b32_e32 v28, v132, v28
	v_or_b32_sdwa v29, v134, v29 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v30, 8, v139
	v_or_b32_sdwa v28, v28, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v29, 8, v137
	v_or_b32_e32 v29, v136, v29
	v_or_b32_sdwa v30, v138, v30 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v31, 8, v143
	v_or_b32_sdwa v29, v29, v30 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v30, 8, v141
	v_or_b32_e32 v30, v140, v30
	v_or_b32_sdwa v31, v142, v31 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v47, 8, v81
	v_or_b32_sdwa v30, v30, v31 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v31, 8, v145
	v_or_b32_e32 v31, v144, v31
	v_or_b32_sdwa v47, v146, v47 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_nop 0
	v_or_b32_sdwa v31, v31, v47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_cbranch_vccz .LBB0_88
.LBB0_39:
	s_waitcnt vmcnt(5) lgkmcnt(0)
	s_barrier
	s_lshl_b64 s[14:15], s[88:89], 11
	v_lshl_add_u64 v[118:119], v[74:75], 0, s[14:15]
	s_add_u32 s14, s44, s70
	s_addc_u32 s15, s45, s70
	s_xor_b64 s[28:29], s[14:15], s[92:93]
	v_cvt_f32_u32_e32 v49, s28
	v_cvt_f32_u32_e32 v51, s29
	v_and_b32_e32 v47, 0xff, v130
	s_movk_i32 s14, 0xf7ff
	v_mov_b32_e32 v81, s29
	v_fmac_f32_e32 v49, 0x4f800000, v51
	v_rcp_f32_e32 v49, v49
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[126:129], v[16:19], v[4:7], v116, v47 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshl_add_u64 v[128:129], v[118:119], 0, v[102:103]
	v_sub_co_u32_e32 v53, vcc, s14, v128
	v_mul_f32_e32 v49, 0x5f7ffffc, v49
	v_mul_f32_e32 v51, 0x2f800000, v49
	v_trunc_f32_e32 v51, v51
	v_fmac_f32_e32 v49, 0xcf800000, v51
	v_cvt_u32_f32_e32 v49, v49
	v_cvt_u32_f32_e32 v51, v51
	s_sub_u32 s14, 0, s28
	s_subb_u32 s15, 0, s29
	v_readfirstlane_b32 s26, v49
	v_readfirstlane_b32 s35, v51
	s_mul_hi_u32 s34, s14, s26
	s_mul_i32 s48, s14, s35
	s_mul_i32 s27, s15, s26
	s_add_i32 s34, s34, s48
	s_add_i32 s34, s34, s27
	s_mul_i32 s49, s14, s26
	s_mul_i32 s48, s26, s34
	s_mul_hi_u32 s55, s26, s49
	s_mul_hi_u32 s27, s26, s34
	s_add_u32 s48, s55, s48
	s_addc_u32 s27, 0, s27
	s_mul_hi_u32 s62, s35, s49
	s_mul_i32 s49, s35, s49
	s_add_u32 s48, s48, s49
	s_mul_hi_u32 s55, s35, s34
	s_addc_u32 s27, s27, s62
	s_addc_u32 s48, s55, 0
	s_mul_i32 s34, s35, s34
	s_add_u32 s27, s27, s34
	s_addc_u32 s34, 0, s48
	s_add_u32 s26, s26, s27
	s_addc_u32 s27, s35, s34
	s_mul_i32 s34, s14, s27
	s_mul_hi_u32 s35, s14, s26
	s_add_i32 s34, s35, s34
	s_mul_i32 s15, s15, s26
	s_add_i32 s34, s34, s15
	s_mul_i32 s14, s14, s26
	s_mul_hi_u32 s35, s27, s14
	s_mul_i32 s48, s27, s14
	s_mul_i32 s55, s26, s34
	s_mul_hi_u32 s14, s26, s14
	s_mul_hi_u32 s49, s26, s34
	s_add_u32 s14, s14, s55
	s_addc_u32 s49, 0, s49
	v_lshl_add_u64 v[120:121], v[128:129], 0, s[90:91]
	v_subb_co_u32_e32 v55, vcc, -1, v129, vcc
	s_add_u32 s14, s14, s48
	v_cmp_gt_i32_e32 vcc, 0, v121
	s_mul_hi_u32 s15, s27, s34
	s_addc_u32 s14, s49, s35
	v_cndmask_b32_e32 v127, v121, v55, vcc
	s_addc_u32 s15, s15, 0
	s_mul_i32 s34, s27, s34
	s_add_u32 s14, s14, s34
	v_ashrrev_i32_e32 v130, 31, v127
	v_cndmask_b32_e32 v126, v120, v53, vcc
	s_addc_u32 s15, 0, s15
	v_mov_b32_e32 v131, v130
	s_add_u32 s34, s26, s14
	v_lshl_add_u64 v[126:127], v[126:127], 0, v[130:131]
	s_addc_u32 s35, s27, s15
	v_xor_b32_e32 v51, v126, v130
	v_xor_b32_e32 v49, v127, v130
	v_mad_u64_u32 v[126:127], s[14:15], v51, s35, 0
	v_mul_hi_u32 v104, v51, s34
	v_lshl_add_u64 v[126:127], v[104:105], 0, v[126:127]
	v_mad_u64_u32 v[134:135], s[14:15], v49, s34, 0
	v_add_co_u32_e32 v53, vcc, v126, v134
	v_mad_u64_u32 v[132:133], s[14:15], v49, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v126, vcc, v127, v135, vcc
	v_mov_b32_e32 v127, s57
	s_nop 0
	v_addc_co_u32_e32 v133, vcc, 0, v133, vcc
	v_lshl_add_u64 v[126:127], v[126:127], 0, v[132:133]
	v_mul_lo_u32 v53, s29, v126
	v_mul_lo_u32 v55, s28, v127
	v_mad_u64_u32 v[132:133], s[14:15], s28, v126, 0
	v_add3_u32 v53, v133, v55, v53
	v_sub_co_u32_e32 v51, vcc, v51, v132
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[170:173], v[16:19], v[0:3], v116, v47 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_subb_co_u32_e64 v55, s[14:15], v49, v53, vcc
	v_sub_u32_e32 v49, v49, v53
	v_cmp_le_u32_e64 s[14:15], s29, v55
	v_subb_co_u32_e32 v49, vcc, v49, v81, vcc
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v51
	v_subrev_co_u32_e32 v51, vcc, s28, v51
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v49, vcc, 0, v49, vcc
	v_cmp_le_u32_e32 vcc, s29, v49
	v_cmp_eq_u32_e64 s[14:15], s29, v55
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[122:125], v[16:19], v[36:39], v117, v47 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v51
	v_cndmask_b32_e64 v55, v57, v65, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v49
	s_nop 1
	v_cndmask_b32_e32 v49, v53, v51, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	v_xor_b32_e32 v51, s70, v130
	s_nop 0
	v_cndmask_b32_e64 v49, 1, 2, vcc
	v_add_u32_e32 v49, v126, v49
	v_cmp_ne_u32_e32 vcc, 0, v55
	s_nop 1
	v_cndmask_b32_e32 v49, v126, v49, vcc
	v_ashrrev_i32_e32 v126, 31, v121
	v_mov_b32_e32 v127, v126
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[126:127]
	v_xor_b32_e32 v49, v49, v51
	v_xor_b32_e32 v53, v120, v126
	v_sub_u32_e32 v49, v49, v51
	v_xor_b32_e32 v51, v121, v126
	v_mad_u64_u32 v[120:121], s[14:15], v53, s35, 0
	v_mul_hi_u32 v104, v53, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[132:133], s[14:15], v51, s34, 0
	v_add_co_u32_e32 v55, vcc, v120, v132
	v_mad_u64_u32 v[130:131], s[14:15], v51, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v133, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_mul_lo_u32 v55, s29, v120
	v_mul_lo_u32 v57, s28, v121
	v_mad_u64_u32 v[120:121], s[14:15], s28, v120, 0
	v_add3_u32 v55, v121, v57, v55
	v_sub_u32_e32 v57, v51, v55
	v_sub_co_u32_e32 v53, vcc, v53, v120
	v_xad_u32 v49, v49, v126, v46
	s_nop 0
	v_subb_co_u32_e64 v57, s[14:15], v57, v81, vcc
	v_subrev_co_u32_e64 v65, s[14:15], s28, v53
	v_subb_co_u32_e32 v51, vcc, v51, v55, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v77, s[26:27], 0, v57, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v77
	v_subb_co_u32_e64 v57, s[14:15], v57, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v65
	v_cmp_le_u32_e32 vcc, s29, v51
	v_mul_lo_u32 v49, v49, s33
	v_cndmask_b32_e64 v83, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v77
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v53
	v_cndmask_b32_e64 v79, v79, v83, s[26:27]
	v_subrev_co_u32_e64 v83, s[14:15], s28, v65
	s_nop 1
	v_subbrev_co_u32_e64 v57, s[14:15], 0, v57, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v79
	s_nop 1
	v_cndmask_b32_e64 v57, v77, v57, s[14:15]
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v51
	s_nop 1
	v_cndmask_b32_e32 v55, v55, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v55
	v_cndmask_b32_e64 v55, v65, v83, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v53, v53, v55, vcc
	v_cndmask_b32_e32 v51, v51, v57, vcc
	v_xor_b32_e32 v53, v53, v126
	v_xor_b32_e32 v51, v51, v126
	v_sub_co_u32_e32 v53, vcc, v53, v126
	s_nop 1
	v_subb_co_u32_e32 v51, vcc, v51, v126, vcc
	v_ashrrev_i32_e32 v51, 31, v51
	v_and_b32_e32 v51, s44, v51
	v_lshl_add_u64 v[126:127], v[106:107], 0, v[118:119]
	v_add3_u32 v49, v51, v53, v49
	v_sub_co_u32_e32 v51, vcc, -2, v126
	v_lshl_add_u64 v[118:119], v[126:127], 0, 1
	s_nop 0
	v_subb_co_u32_e32 v53, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -1, v[126:127]
	s_nop 1
	v_cndmask_b32_e32 v121, v119, v53, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_cndmask_b32_e32 v120, v118, v51, vcc
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v53, v120, v130
	v_xor_b32_e32 v51, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v53, s35, 0
	v_mul_hi_u32 v104, v53, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[134:135], s[14:15], v51, s34, 0
	v_add_co_u32_e32 v55, vcc, v120, v134
	v_mad_u64_u32 v[132:133], s[14:15], v51, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v135, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v133, vcc, 0, v133, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[132:133]
	v_mul_lo_u32 v55, s29, v120
	v_mul_lo_u32 v57, s28, v121
	v_mad_u64_u32 v[132:133], s[14:15], s28, v120, 0
	v_add3_u32 v55, v133, v57, v55
	v_sub_co_u32_e32 v53, vcc, v53, v132
	s_nop 1
	v_subb_co_u32_e64 v57, s[14:15], v51, v55, vcc
	v_sub_u32_e32 v51, v51, v55
	v_cmp_le_u32_e64 s[14:15], s29, v57
	v_subb_co_u32_e32 v51, vcc, v51, v81, vcc
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v53
	v_subrev_co_u32_e32 v53, vcc, s28, v53
	s_nop 0
	v_cndmask_b32_e64 v77, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v51, vcc, 0, v51, vcc
	v_cmp_le_u32_e32 vcc, s29, v51
	v_cmp_eq_u32_e64 s[14:15], s29, v57
	s_nop 0
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v53
	v_cndmask_b32_e64 v57, v65, v77, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v51
	s_nop 1
	v_cndmask_b32_e32 v51, v55, v53, vcc
	v_cmp_ne_u32_e32 vcc, 0, v51
	v_xor_b32_e32 v53, s70, v130
	s_nop 0
	v_cndmask_b32_e64 v51, 1, 2, vcc
	v_add_u32_e32 v51, v120, v51
	v_cmp_ne_u32_e32 vcc, 0, v57
	s_nop 1
	v_cndmask_b32_e32 v51, v120, v51, vcc
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v51, v51, v53
	v_xor_b32_e32 v55, v118, v120
	v_sub_u32_e32 v51, v51, v53
	v_xor_b32_e32 v53, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v55, s35, 0
	v_mul_hi_u32 v104, v55, s34
	v_lshl_add_u64 v[118:119], v[104:105], 0, v[118:119]
	v_mad_u64_u32 v[132:133], s[14:15], v53, s34, 0
	v_add_co_u32_e32 v57, vcc, v118, v132
	v_mad_u64_u32 v[130:131], s[14:15], v53, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v118, vcc, v119, v133, vcc
	v_mov_b32_e32 v119, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[130:131]
	v_mul_lo_u32 v57, s29, v118
	v_mul_lo_u32 v65, s28, v119
	v_mad_u64_u32 v[118:119], s[14:15], s28, v118, 0
	v_add3_u32 v57, v119, v65, v57
	v_sub_u32_e32 v65, v53, v57
	v_sub_co_u32_e32 v55, vcc, v55, v118
	v_xad_u32 v51, v51, v120, v46
	s_nop 0
	v_subb_co_u32_e64 v65, s[14:15], v65, v81, vcc
	v_subrev_co_u32_e64 v77, s[14:15], s28, v55
	v_subb_co_u32_e32 v53, vcc, v53, v57, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v79, s[26:27], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v79
	v_subb_co_u32_e64 v65, s[14:15], v65, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v77
	v_cmp_le_u32_e32 vcc, s29, v53
	v_mul_lo_u32 v51, v51, s33
	v_cndmask_b32_e64 v85, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v79
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v55
	v_cndmask_b32_e64 v83, v83, v85, s[26:27]
	v_subrev_co_u32_e64 v85, s[14:15], s28, v77
	v_lshl_add_u64 v[118:119], v[126:127], 0, 2
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v83
	s_nop 1
	v_cndmask_b32_e64 v65, v79, v65, s[14:15]
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v53
	s_nop 1
	v_cndmask_b32_e32 v57, v57, v79, vcc
	v_cmp_ne_u32_e32 vcc, 0, v57
	v_cndmask_b32_e64 v57, v77, v85, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v55, v55, v57, vcc
	v_cndmask_b32_e32 v53, v53, v65, vcc
	v_xor_b32_e32 v55, v55, v120
	v_xor_b32_e32 v53, v53, v120
	v_sub_co_u32_e32 v55, vcc, v55, v120
	s_nop 1
	v_subb_co_u32_e32 v53, vcc, v53, v120, vcc
	v_ashrrev_i32_e32 v53, 31, v53
	v_and_b32_e32 v53, s44, v53
	v_add3_u32 v51, v53, v55, v51
	v_sub_co_u32_e32 v53, vcc, -3, v126
	s_nop 1
	v_subb_co_u32_e32 v55, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -2, v[126:127]
	s_nop 1
	v_cndmask_b32_e32 v121, v119, v55, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_cndmask_b32_e32 v120, v118, v53, vcc
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v55, v120, v130
	v_xor_b32_e32 v53, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v55, s35, 0
	v_mul_hi_u32 v104, v55, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[134:135], s[14:15], v53, s34, 0
	v_add_co_u32_e32 v57, vcc, v120, v134
	v_mad_u64_u32 v[132:133], s[14:15], v53, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v135, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v133, vcc, 0, v133, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[132:133]
	v_mul_lo_u32 v57, s29, v120
	v_mul_lo_u32 v65, s28, v121
	v_mad_u64_u32 v[132:133], s[14:15], s28, v120, 0
	v_add3_u32 v57, v133, v65, v57
	v_sub_co_u32_e32 v55, vcc, v55, v132
	s_nop 1
	v_subb_co_u32_e64 v65, s[14:15], v53, v57, vcc
	v_sub_u32_e32 v53, v53, v57
	v_cmp_le_u32_e64 s[14:15], s29, v65
	v_subb_co_u32_e32 v53, vcc, v53, v81, vcc
	s_nop 0
	v_cndmask_b32_e64 v77, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v55
	v_subrev_co_u32_e32 v55, vcc, s28, v55
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v53, vcc, 0, v53, vcc
	v_cmp_le_u32_e32 vcc, s29, v53
	v_cmp_eq_u32_e64 s[14:15], s29, v65
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v55
	v_cndmask_b32_e64 v65, v77, v79, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v53
	s_nop 1
	v_cndmask_b32_e32 v53, v57, v55, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_xor_b32_e32 v55, s70, v130
	s_nop 0
	v_cndmask_b32_e64 v53, 1, 2, vcc
	v_add_u32_e32 v53, v120, v53
	v_cmp_ne_u32_e32 vcc, 0, v65
	s_nop 1
	v_cndmask_b32_e32 v53, v120, v53, vcc
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v53, v53, v55
	v_xor_b32_e32 v57, v118, v120
	v_sub_u32_e32 v53, v53, v55
	v_xor_b32_e32 v55, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v57, s35, 0
	v_mul_hi_u32 v104, v57, s34
	v_lshl_add_u64 v[118:119], v[104:105], 0, v[118:119]
	v_mad_u64_u32 v[132:133], s[14:15], v55, s34, 0
	v_add_co_u32_e32 v65, vcc, v118, v132
	v_mad_u64_u32 v[130:131], s[14:15], v55, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v118, vcc, v119, v133, vcc
	v_mov_b32_e32 v119, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[130:131]
	v_mul_lo_u32 v65, s29, v118
	v_mul_lo_u32 v77, s28, v119
	v_mad_u64_u32 v[118:119], s[14:15], s28, v118, 0
	v_add3_u32 v65, v119, v77, v65
	v_sub_u32_e32 v77, v55, v65
	v_sub_co_u32_e32 v57, vcc, v57, v118
	v_xad_u32 v53, v53, v120, v46
	s_nop 0
	v_subb_co_u32_e64 v77, s[14:15], v77, v81, vcc
	v_subrev_co_u32_e64 v79, s[14:15], s28, v57
	v_subb_co_u32_e32 v55, vcc, v55, v65, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v83, s[26:27], 0, v77, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v83
	v_subb_co_u32_e64 v77, s[14:15], v77, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v79
	v_cmp_le_u32_e32 vcc, s29, v55
	v_mul_lo_u32 v53, v53, s33
	v_cndmask_b32_e64 v87, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v83
	v_cndmask_b32_e64 v65, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v57
	v_cndmask_b32_e64 v85, v85, v87, s[26:27]
	v_subrev_co_u32_e64 v87, s[14:15], s28, v79
	v_lshl_add_u64 v[118:119], v[126:127], 0, 3
	s_nop 0
	v_subbrev_co_u32_e64 v77, s[14:15], 0, v77, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v85
	s_nop 1
	v_cndmask_b32_e64 v77, v83, v77, s[14:15]
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v55
	s_nop 1
	v_cndmask_b32_e32 v65, v65, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v65
	v_cndmask_b32_e64 v65, v79, v87, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v57, v57, v65, vcc
	v_cndmask_b32_e32 v55, v55, v77, vcc
	v_xor_b32_e32 v57, v57, v120
	v_xor_b32_e32 v55, v55, v120
	v_sub_co_u32_e32 v57, vcc, v57, v120
	s_nop 1
	v_subb_co_u32_e32 v55, vcc, v55, v120, vcc
	v_ashrrev_i32_e32 v55, 31, v55
	v_and_b32_e32 v55, s44, v55
	v_add3_u32 v55, v55, v57, v53
	v_sub_co_u32_e32 v53, vcc, -4, v126
	s_nop 1
	v_subb_co_u32_e32 v57, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -3, v[126:127]
	s_nop 1
	v_cndmask_b32_e32 v121, v119, v57, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_cndmask_b32_e32 v120, v118, v53, vcc
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v57, v120, v130
	v_xor_b32_e32 v53, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v57, s35, 0
	v_mul_hi_u32 v104, v57, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[134:135], s[14:15], v53, s34, 0
	v_add_co_u32_e32 v65, vcc, v120, v134
	v_mad_u64_u32 v[132:133], s[14:15], v53, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v135, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v133, vcc, 0, v133, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[132:133]
	v_mul_lo_u32 v65, s29, v120
	v_mul_lo_u32 v77, s28, v121
	v_mad_u64_u32 v[132:133], s[14:15], s28, v120, 0
	v_add3_u32 v65, v133, v77, v65
	v_sub_co_u32_e32 v57, vcc, v57, v132
	s_nop 1
	v_subb_co_u32_e64 v77, s[14:15], v53, v65, vcc
	v_sub_u32_e32 v53, v53, v65
	v_cmp_le_u32_e64 s[14:15], s29, v77
	v_subb_co_u32_e32 v53, vcc, v53, v81, vcc
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v57
	v_subrev_co_u32_e32 v57, vcc, s28, v57
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v53, vcc, 0, v53, vcc
	v_cmp_le_u32_e32 vcc, s29, v53
	v_cmp_eq_u32_e64 s[14:15], s29, v77
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v57
	v_cndmask_b32_e64 v77, v79, v83, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v53
	s_nop 1
	v_cndmask_b32_e32 v53, v65, v57, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_xor_b32_e32 v57, s70, v130
	s_nop 0
	v_cndmask_b32_e64 v53, 1, 2, vcc
	v_add_u32_e32 v53, v120, v53
	v_cmp_ne_u32_e32 vcc, 0, v77
	s_nop 1
	v_cndmask_b32_e32 v53, v120, v53, vcc
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v53, v53, v57
	v_xor_b32_e32 v65, v118, v120
	v_sub_u32_e32 v53, v53, v57
	v_xor_b32_e32 v57, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v65, s35, 0
	v_mul_hi_u32 v104, v65, s34
	v_lshl_add_u64 v[118:119], v[104:105], 0, v[118:119]
	v_mad_u64_u32 v[132:133], s[14:15], v57, s34, 0
	v_add_co_u32_e32 v77, vcc, v118, v132
	v_mad_u64_u32 v[130:131], s[14:15], v57, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v118, vcc, v119, v133, vcc
	v_mov_b32_e32 v119, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[130:131]
	v_mul_lo_u32 v77, s29, v118
	v_mul_lo_u32 v79, s28, v119
	v_mad_u64_u32 v[118:119], s[14:15], s28, v118, 0
	v_add3_u32 v77, v119, v79, v77
	v_sub_u32_e32 v79, v57, v77
	v_sub_co_u32_e32 v65, vcc, v65, v118
	v_xad_u32 v53, v53, v120, v46
	s_nop 0
	v_subb_co_u32_e64 v79, s[14:15], v79, v81, vcc
	v_subrev_co_u32_e64 v83, s[14:15], s28, v65
	v_subb_co_u32_e32 v57, vcc, v57, v77, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[26:27], 0, v79, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v85
	v_subb_co_u32_e64 v79, s[14:15], v79, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v83
	v_cmp_le_u32_e32 vcc, s29, v57
	v_mul_lo_u32 v53, v53, s33
	v_cndmask_b32_e64 v89, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v85
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v65
	v_cndmask_b32_e64 v87, v87, v89, s[26:27]
	v_subrev_co_u32_e64 v89, s[14:15], s28, v83
	v_lshl_add_u64 v[118:119], v[126:127], 0, 4
	s_nop 0
	v_subbrev_co_u32_e64 v79, s[14:15], 0, v79, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v87
	s_nop 1
	v_cndmask_b32_e64 v79, v85, v79, s[14:15]
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v57
	s_nop 1
	v_cndmask_b32_e32 v77, v77, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v77
	v_cndmask_b32_e64 v77, v83, v89, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v65, v65, v77, vcc
	v_cndmask_b32_e32 v57, v57, v79, vcc
	v_xor_b32_e32 v65, v65, v120
	v_xor_b32_e32 v57, v57, v120
	v_sub_co_u32_e32 v65, vcc, v65, v120
	s_nop 1
	v_subb_co_u32_e32 v57, vcc, v57, v120, vcc
	v_ashrrev_i32_e32 v57, 31, v57
	v_and_b32_e32 v57, s44, v57
	v_add3_u32 v57, v57, v65, v53
	v_sub_co_u32_e32 v53, vcc, -5, v126
	s_nop 1
	v_subb_co_u32_e32 v65, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -4, v[126:127]
	s_nop 1
	v_cndmask_b32_e32 v121, v119, v65, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_cndmask_b32_e32 v120, v118, v53, vcc
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v65, v120, v130
	v_xor_b32_e32 v53, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v65, s35, 0
	v_mul_hi_u32 v104, v65, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[134:135], s[14:15], v53, s34, 0
	v_add_co_u32_e32 v77, vcc, v120, v134
	v_mad_u64_u32 v[132:133], s[14:15], v53, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v135, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v133, vcc, 0, v133, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[132:133]
	v_mul_lo_u32 v77, s29, v120
	v_mul_lo_u32 v79, s28, v121
	v_mad_u64_u32 v[132:133], s[14:15], s28, v120, 0
	v_add3_u32 v77, v133, v79, v77
	v_sub_co_u32_e32 v65, vcc, v65, v132
	s_nop 1
	v_subb_co_u32_e64 v79, s[14:15], v53, v77, vcc
	v_sub_u32_e32 v53, v53, v77
	v_cmp_le_u32_e64 s[14:15], s29, v79
	v_subb_co_u32_e32 v53, vcc, v53, v81, vcc
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v65
	v_subrev_co_u32_e32 v65, vcc, s28, v65
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v53, vcc, 0, v53, vcc
	v_cmp_le_u32_e32 vcc, s29, v53
	v_cmp_eq_u32_e64 s[14:15], s29, v79
	s_nop 0
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v65
	v_cndmask_b32_e64 v79, v83, v85, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v53
	s_nop 1
	v_cndmask_b32_e32 v53, v77, v65, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_xor_b32_e32 v65, s70, v130
	s_nop 0
	v_cndmask_b32_e64 v53, 1, 2, vcc
	v_add_u32_e32 v53, v120, v53
	v_cmp_ne_u32_e32 vcc, 0, v79
	s_nop 1
	v_cndmask_b32_e32 v53, v120, v53, vcc
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v53, v53, v65
	v_xor_b32_e32 v77, v118, v120
	v_sub_u32_e32 v53, v53, v65
	v_xor_b32_e32 v65, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v77, s35, 0
	v_mul_hi_u32 v104, v77, s34
	v_lshl_add_u64 v[118:119], v[104:105], 0, v[118:119]
	v_mad_u64_u32 v[132:133], s[14:15], v65, s34, 0
	v_add_co_u32_e32 v79, vcc, v118, v132
	v_mad_u64_u32 v[130:131], s[14:15], v65, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v118, vcc, v119, v133, vcc
	v_mov_b32_e32 v119, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[130:131]
	v_mul_lo_u32 v79, s29, v118
	v_mul_lo_u32 v83, s28, v119
	v_mad_u64_u32 v[118:119], s[14:15], s28, v118, 0
	v_add3_u32 v79, v119, v83, v79
	v_sub_u32_e32 v83, v65, v79
	v_sub_co_u32_e32 v77, vcc, v77, v118
	v_xad_u32 v53, v53, v120, v46
	s_nop 0
	v_subb_co_u32_e64 v83, s[14:15], v83, v81, vcc
	v_subrev_co_u32_e64 v85, s[14:15], s28, v77
	v_subb_co_u32_e32 v65, vcc, v65, v79, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v87, s[26:27], 0, v83, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v87
	v_subb_co_u32_e64 v83, s[14:15], v83, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v89, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v85
	v_cmp_le_u32_e32 vcc, s29, v65
	v_mul_lo_u32 v53, v53, s33
	v_cndmask_b32_e64 v91, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v87
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v77
	v_cndmask_b32_e64 v89, v89, v91, s[26:27]
	v_subrev_co_u32_e64 v91, s[14:15], s28, v85
	v_lshl_add_u64 v[118:119], v[126:127], 0, 5
	s_nop 0
	v_subbrev_co_u32_e64 v83, s[14:15], 0, v83, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v89
	s_nop 1
	v_cndmask_b32_e64 v83, v87, v83, s[14:15]
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v65
	s_nop 1
	v_cndmask_b32_e32 v79, v79, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v79
	v_cndmask_b32_e64 v79, v85, v91, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v77, v77, v79, vcc
	v_cndmask_b32_e32 v65, v65, v83, vcc
	v_xor_b32_e32 v77, v77, v120
	v_xor_b32_e32 v65, v65, v120
	v_sub_co_u32_e32 v77, vcc, v77, v120
	s_nop 1
	v_subb_co_u32_e32 v65, vcc, v65, v120, vcc
	v_ashrrev_i32_e32 v65, 31, v65
	v_and_b32_e32 v65, s44, v65
	v_add3_u32 v65, v65, v77, v53
	v_sub_co_u32_e32 v53, vcc, -6, v126
	s_nop 1
	v_subb_co_u32_e32 v77, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -5, v[126:127]
	s_nop 1
	v_cndmask_b32_e32 v121, v119, v77, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_cndmask_b32_e32 v120, v118, v53, vcc
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v77, v120, v130
	v_xor_b32_e32 v53, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v77, s35, 0
	v_mul_hi_u32 v104, v77, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[134:135], s[14:15], v53, s34, 0
	v_add_co_u32_e32 v79, vcc, v120, v134
	v_mad_u64_u32 v[132:133], s[14:15], v53, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v135, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v133, vcc, 0, v133, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[132:133]
	v_mul_lo_u32 v79, s29, v120
	v_mul_lo_u32 v83, s28, v121
	v_mad_u64_u32 v[132:133], s[14:15], s28, v120, 0
	v_add3_u32 v79, v133, v83, v79
	v_sub_co_u32_e32 v77, vcc, v77, v132
	s_nop 1
	v_subb_co_u32_e64 v83, s[14:15], v53, v79, vcc
	v_sub_u32_e32 v53, v53, v79
	v_cmp_le_u32_e64 s[14:15], s29, v83
	v_subb_co_u32_e32 v53, vcc, v53, v81, vcc
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v77
	v_subrev_co_u32_e32 v77, vcc, s28, v77
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v53, vcc, 0, v53, vcc
	v_cmp_le_u32_e32 vcc, s29, v53
	v_cmp_eq_u32_e64 s[14:15], s29, v83
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v77
	v_cndmask_b32_e64 v83, v85, v87, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v53
	s_nop 1
	v_cndmask_b32_e32 v53, v79, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_xor_b32_e32 v77, s70, v130
	s_nop 0
	v_cndmask_b32_e64 v53, 1, 2, vcc
	v_add_u32_e32 v53, v120, v53
	v_cmp_ne_u32_e32 vcc, 0, v83
	s_nop 1
	v_cndmask_b32_e32 v53, v120, v53, vcc
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v53, v53, v77
	v_xor_b32_e32 v79, v118, v120
	v_sub_u32_e32 v53, v53, v77
	v_xor_b32_e32 v77, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v79, s35, 0
	v_mul_hi_u32 v104, v79, s34
	v_lshl_add_u64 v[118:119], v[104:105], 0, v[118:119]
	v_mad_u64_u32 v[132:133], s[14:15], v77, s34, 0
	v_add_co_u32_e32 v83, vcc, v118, v132
	v_mad_u64_u32 v[130:131], s[14:15], v77, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v118, vcc, v119, v133, vcc
	v_mov_b32_e32 v119, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[130:131]
	v_mul_lo_u32 v83, s29, v118
	v_mul_lo_u32 v85, s28, v119
	v_mad_u64_u32 v[118:119], s[14:15], s28, v118, 0
	v_add3_u32 v83, v119, v85, v83
	v_sub_u32_e32 v85, v77, v83
	v_sub_co_u32_e32 v79, vcc, v79, v118
	v_xad_u32 v53, v53, v120, v46
	s_nop 0
	v_subb_co_u32_e64 v85, s[14:15], v85, v81, vcc
	v_subrev_co_u32_e64 v87, s[14:15], s28, v79
	v_subb_co_u32_e32 v77, vcc, v77, v83, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[26:27], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v89
	v_subb_co_u32_e64 v85, s[14:15], v85, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v87
	v_cmp_le_u32_e32 vcc, s29, v77
	v_mul_lo_u32 v53, v53, s33
	v_cndmask_b32_e64 v93, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v89
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v79
	v_cndmask_b32_e64 v91, v91, v93, s[26:27]
	v_subrev_co_u32_e64 v93, s[14:15], s28, v87
	v_lshl_add_u64 v[118:119], v[126:127], 0, 6
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v85, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v91
	s_nop 1
	v_cndmask_b32_e64 v85, v89, v85, s[14:15]
	v_cndmask_b32_e64 v89, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v77
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v89, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	v_cndmask_b32_e64 v83, v87, v93, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v79, v79, v83, vcc
	v_cndmask_b32_e32 v77, v77, v85, vcc
	v_xor_b32_e32 v79, v79, v120
	v_xor_b32_e32 v77, v77, v120
	v_sub_co_u32_e32 v79, vcc, v79, v120
	s_nop 1
	v_subb_co_u32_e32 v77, vcc, v77, v120, vcc
	v_ashrrev_i32_e32 v77, 31, v77
	v_and_b32_e32 v77, s44, v77
	v_add3_u32 v77, v77, v79, v53
	v_sub_co_u32_e32 v53, vcc, -7, v126
	s_nop 1
	v_subb_co_u32_e32 v79, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -6, v[126:127]
	s_nop 1
	v_cndmask_b32_e32 v121, v119, v79, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_cndmask_b32_e32 v120, v118, v53, vcc
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v79, v120, v130
	v_xor_b32_e32 v53, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v79, s35, 0
	v_mul_hi_u32 v104, v79, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[134:135], s[14:15], v53, s34, 0
	v_add_co_u32_e32 v83, vcc, v120, v134
	v_mad_u64_u32 v[132:133], s[14:15], v53, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v135, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v133, vcc, 0, v133, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[132:133]
	v_mul_lo_u32 v83, s29, v120
	v_mul_lo_u32 v85, s28, v121
	v_mad_u64_u32 v[132:133], s[14:15], s28, v120, 0
	v_add3_u32 v83, v133, v85, v83
	v_sub_co_u32_e32 v79, vcc, v79, v132
	s_nop 1
	v_subb_co_u32_e64 v85, s[14:15], v53, v83, vcc
	v_sub_u32_e32 v53, v53, v83
	v_cmp_le_u32_e64 s[14:15], s29, v85
	v_subb_co_u32_e32 v53, vcc, v53, v81, vcc
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v79
	v_subrev_co_u32_e32 v79, vcc, s28, v79
	s_nop 0
	v_cndmask_b32_e64 v89, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v53, vcc, 0, v53, vcc
	v_cmp_le_u32_e32 vcc, s29, v53
	v_cmp_eq_u32_e64 s[14:15], s29, v85
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v79
	v_cndmask_b32_e64 v85, v87, v89, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v53
	s_nop 1
	v_cndmask_b32_e32 v53, v83, v79, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_xor_b32_e32 v79, s70, v130
	s_nop 0
	v_cndmask_b32_e64 v53, 1, 2, vcc
	v_add_u32_e32 v53, v120, v53
	v_cmp_ne_u32_e32 vcc, 0, v85
	s_nop 1
	v_cndmask_b32_e32 v53, v120, v53, vcc
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v53, v53, v79
	v_xor_b32_e32 v83, v118, v120
	v_sub_u32_e32 v53, v53, v79
	v_xor_b32_e32 v79, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v83, s35, 0
	v_mul_hi_u32 v104, v83, s34
	v_lshl_add_u64 v[118:119], v[104:105], 0, v[118:119]
	v_mad_u64_u32 v[132:133], s[14:15], v79, s34, 0
	v_add_co_u32_e32 v85, vcc, v118, v132
	v_mad_u64_u32 v[130:131], s[14:15], v79, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v118, vcc, v119, v133, vcc
	v_mov_b32_e32 v119, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[130:131]
	v_mul_lo_u32 v85, s29, v118
	v_mul_lo_u32 v87, s28, v119
	v_mad_u64_u32 v[118:119], s[14:15], s28, v118, 0
	v_add3_u32 v85, v119, v87, v85
	v_sub_u32_e32 v87, v79, v85
	v_sub_co_u32_e32 v83, vcc, v83, v118
	v_xad_u32 v53, v53, v120, v46
	s_nop 0
	v_subb_co_u32_e64 v87, s[14:15], v87, v81, vcc
	v_subrev_co_u32_e64 v89, s[14:15], s28, v83
	v_subb_co_u32_e32 v79, vcc, v79, v85, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[26:27], 0, v87, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v91
	v_subb_co_u32_e64 v87, s[14:15], v87, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v89
	v_cmp_le_u32_e32 vcc, s29, v79
	v_mul_lo_u32 v53, v53, s33
	v_cndmask_b32_e64 v95, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v91
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v83
	v_cndmask_b32_e64 v93, v93, v95, s[26:27]
	v_subrev_co_u32_e64 v95, s[14:15], s28, v89
	v_lshl_add_u64 v[118:119], v[126:127], 0, 7
	s_nop 0
	v_subbrev_co_u32_e64 v87, s[14:15], 0, v87, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v93
	s_nop 1
	v_cndmask_b32_e64 v87, v91, v87, s[14:15]
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v79
	s_nop 1
	v_cndmask_b32_e32 v85, v85, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	v_cndmask_b32_e64 v85, v89, v95, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v83, v83, v85, vcc
	v_cndmask_b32_e32 v79, v79, v87, vcc
	v_xor_b32_e32 v83, v83, v120
	v_xor_b32_e32 v79, v79, v120
	v_sub_co_u32_e32 v83, vcc, v83, v120
	s_nop 1
	v_subb_co_u32_e32 v79, vcc, v79, v120, vcc
	v_ashrrev_i32_e32 v79, 31, v79
	v_and_b32_e32 v79, s44, v79
	v_add3_u32 v79, v79, v83, v53
	v_sub_co_u32_e32 v53, vcc, -8, v126
	s_nop 1
	v_subb_co_u32_e32 v83, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -7, v[126:127]
	s_nop 1
	v_cndmask_b32_e32 v121, v119, v83, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_cndmask_b32_e32 v120, v118, v53, vcc
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v83, v120, v130
	v_xor_b32_e32 v53, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v83, s35, 0
	v_mul_hi_u32 v104, v83, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[134:135], s[14:15], v53, s34, 0
	v_add_co_u32_e32 v85, vcc, v120, v134
	v_mad_u64_u32 v[132:133], s[14:15], v53, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v135, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v133, vcc, 0, v133, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[132:133]
	v_mul_lo_u32 v85, s29, v120
	v_mul_lo_u32 v87, s28, v121
	v_mad_u64_u32 v[132:133], s[14:15], s28, v120, 0
	v_add3_u32 v85, v133, v87, v85
	v_sub_co_u32_e32 v83, vcc, v83, v132
	s_nop 1
	v_subb_co_u32_e64 v87, s[14:15], v53, v85, vcc
	v_sub_u32_e32 v53, v53, v85
	v_cmp_le_u32_e64 s[14:15], s29, v87
	v_subb_co_u32_e32 v53, vcc, v53, v81, vcc
	s_nop 0
	v_cndmask_b32_e64 v89, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v83
	v_subrev_co_u32_e32 v83, vcc, s28, v83
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v53, vcc, 0, v53, vcc
	v_cmp_le_u32_e32 vcc, s29, v53
	v_cmp_eq_u32_e64 s[14:15], s29, v87
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v83
	v_cndmask_b32_e64 v87, v89, v91, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v53
	s_nop 1
	v_cndmask_b32_e32 v53, v85, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_xor_b32_e32 v83, s70, v130
	s_nop 0
	v_cndmask_b32_e64 v53, 1, 2, vcc
	v_add_u32_e32 v53, v120, v53
	v_cmp_ne_u32_e32 vcc, 0, v87
	s_nop 1
	v_cndmask_b32_e32 v53, v120, v53, vcc
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v53, v53, v83
	v_xor_b32_e32 v85, v118, v120
	v_sub_u32_e32 v53, v53, v83
	v_xor_b32_e32 v83, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v85, s35, 0
	v_mul_hi_u32 v104, v85, s34
	v_lshl_add_u64 v[118:119], v[104:105], 0, v[118:119]
	v_mad_u64_u32 v[132:133], s[14:15], v83, s34, 0
	v_add_co_u32_e32 v87, vcc, v118, v132
	v_mad_u64_u32 v[130:131], s[14:15], v83, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v118, vcc, v119, v133, vcc
	v_mov_b32_e32 v119, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[130:131]
	v_mul_lo_u32 v87, s29, v118
	v_mul_lo_u32 v89, s28, v119
	v_mad_u64_u32 v[118:119], s[14:15], s28, v118, 0
	v_add3_u32 v87, v119, v89, v87
	v_sub_u32_e32 v89, v83, v87
	v_sub_co_u32_e32 v85, vcc, v85, v118
	v_xad_u32 v53, v53, v120, v46
	s_nop 0
	v_subb_co_u32_e64 v89, s[14:15], v89, v81, vcc
	v_subrev_co_u32_e64 v91, s[14:15], s28, v85
	v_subb_co_u32_e32 v83, vcc, v83, v87, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v93, s[26:27], 0, v89, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v93
	v_subb_co_u32_e64 v89, s[14:15], v89, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v91
	v_cmp_le_u32_e32 vcc, s29, v83
	v_mul_lo_u32 v53, v53, s33
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v93
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v85
	v_cndmask_b32_e64 v95, v95, v104, s[26:27]
	v_subrev_co_u32_e64 v104, s[14:15], s28, v91
	v_lshl_add_u64 v[118:119], v[126:127], 0, 8
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[14:15], 0, v89, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v95
	s_nop 1
	v_cndmask_b32_e64 v89, v93, v89, s[14:15]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v83
	s_nop 1
	v_cndmask_b32_e32 v87, v87, v93, vcc
	v_cmp_ne_u32_e32 vcc, 0, v87
	v_cndmask_b32_e64 v87, v91, v104, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v85, v85, v87, vcc
	v_cndmask_b32_e32 v83, v83, v89, vcc
	v_xor_b32_e32 v85, v85, v120
	v_xor_b32_e32 v83, v83, v120
	v_sub_co_u32_e32 v85, vcc, v85, v120
	s_nop 1
	v_subb_co_u32_e32 v83, vcc, v83, v120, vcc
	v_ashrrev_i32_e32 v83, 31, v83
	v_and_b32_e32 v83, s44, v83
	v_add3_u32 v83, v83, v85, v53
	buffer_load_ubyte v49, v49, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v53, v51, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v51, v55, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v55, v57, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v57, v65, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v77, v77, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v65, v79, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v79, v83, s[36:39], 0 offen
	v_sub_co_u32_e32 v83, vcc, -9, v126
	s_nop 1
	v_subb_co_u32_e32 v85, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -8, v[126:127]
	s_nop 1
	v_cndmask_b32_e32 v121, v119, v85, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_cndmask_b32_e32 v120, v118, v83, vcc
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v85, v120, v130
	v_xor_b32_e32 v83, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v85, s35, 0
	v_mul_hi_u32 v104, v85, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[134:135], s[14:15], v83, s34, 0
	v_add_co_u32_e32 v87, vcc, v120, v134
	v_mad_u64_u32 v[132:133], s[14:15], v83, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v135, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v133, vcc, 0, v133, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[132:133]
	v_mul_lo_u32 v87, s29, v120
	v_mul_lo_u32 v89, s28, v121
	v_mad_u64_u32 v[132:133], s[14:15], s28, v120, 0
	v_add3_u32 v87, v133, v89, v87
	v_sub_co_u32_e32 v85, vcc, v85, v132
	s_nop 1
	v_subb_co_u32_e64 v89, s[14:15], v83, v87, vcc
	v_sub_u32_e32 v83, v83, v87
	v_cmp_le_u32_e64 s[14:15], s29, v89
	v_subb_co_u32_e32 v83, vcc, v83, v81, vcc
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v85
	v_subrev_co_u32_e32 v85, vcc, s28, v85
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v83, vcc, 0, v83, vcc
	v_cmp_le_u32_e32 vcc, s29, v83
	v_cmp_eq_u32_e64 s[14:15], s29, v89
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v85
	v_cndmask_b32_e64 v89, v91, v93, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v83
	s_nop 1
	v_cndmask_b32_e32 v83, v87, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	v_xor_b32_e32 v85, s70, v130
	s_nop 0
	v_cndmask_b32_e64 v83, 1, 2, vcc
	v_add_u32_e32 v83, v120, v83
	v_cmp_ne_u32_e32 vcc, 0, v89
	s_nop 1
	v_cndmask_b32_e32 v83, v120, v83, vcc
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v83, v83, v85
	v_xor_b32_e32 v87, v118, v120
	v_sub_u32_e32 v83, v83, v85
	v_xor_b32_e32 v85, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v87, s35, 0
	v_mul_hi_u32 v104, v87, s34
	v_lshl_add_u64 v[118:119], v[104:105], 0, v[118:119]
	v_mad_u64_u32 v[132:133], s[14:15], v85, s34, 0
	v_add_co_u32_e32 v89, vcc, v118, v132
	v_mad_u64_u32 v[130:131], s[14:15], v85, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v118, vcc, v119, v133, vcc
	v_mov_b32_e32 v119, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[130:131]
	v_mul_lo_u32 v89, s29, v118
	v_mul_lo_u32 v91, s28, v119
	v_mad_u64_u32 v[118:119], s[14:15], s28, v118, 0
	v_add3_u32 v89, v119, v91, v89
	v_sub_u32_e32 v91, v85, v89
	v_sub_co_u32_e32 v87, vcc, v87, v118
	v_xad_u32 v83, v83, v120, v46
	s_nop 0
	v_subb_co_u32_e64 v91, s[14:15], v91, v81, vcc
	v_subrev_co_u32_e64 v93, s[14:15], s28, v87
	v_subb_co_u32_e32 v85, vcc, v85, v89, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v95, s[26:27], 0, v91, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v95
	v_subb_co_u32_e64 v91, s[14:15], v91, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v93
	v_cmp_le_u32_e32 vcc, s29, v85
	v_mul_lo_u32 v83, v83, s33
	v_cndmask_b32_e64 v109, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v95
	v_cndmask_b32_e64 v89, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v87
	v_cndmask_b32_e64 v104, v104, v109, s[26:27]
	v_subrev_co_u32_e64 v109, s[14:15], s28, v93
	v_lshl_add_u64 v[118:119], v[126:127], 0, 9
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[14:15], 0, v91, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v104
	s_nop 1
	v_cndmask_b32_e64 v91, v95, v91, s[14:15]
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v85
	s_nop 1
	v_cndmask_b32_e32 v89, v89, v95, vcc
	v_cmp_ne_u32_e32 vcc, 0, v89
	v_cndmask_b32_e64 v89, v93, v109, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v87, v87, v89, vcc
	v_cndmask_b32_e32 v85, v85, v91, vcc
	v_xor_b32_e32 v87, v87, v120
	v_xor_b32_e32 v85, v85, v120
	v_sub_co_u32_e32 v87, vcc, v87, v120
	s_nop 1
	v_subb_co_u32_e32 v85, vcc, v85, v120, vcc
	v_ashrrev_i32_e32 v85, 31, v85
	v_and_b32_e32 v85, s44, v85
	v_add3_u32 v83, v85, v87, v83
	v_sub_co_u32_e32 v85, vcc, -10, v126
	s_nop 1
	v_subb_co_u32_e32 v87, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -9, v[126:127]
	s_nop 1
	v_cndmask_b32_e32 v121, v119, v87, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_cndmask_b32_e32 v120, v118, v85, vcc
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v87, v120, v130
	v_xor_b32_e32 v85, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v87, s35, 0
	v_mul_hi_u32 v104, v87, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[134:135], s[14:15], v85, s34, 0
	v_add_co_u32_e32 v89, vcc, v120, v134
	v_mad_u64_u32 v[132:133], s[14:15], v85, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v135, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v133, vcc, 0, v133, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[132:133]
	v_mul_lo_u32 v89, s29, v120
	v_mul_lo_u32 v91, s28, v121
	v_mad_u64_u32 v[132:133], s[14:15], s28, v120, 0
	v_add3_u32 v89, v133, v91, v89
	v_sub_co_u32_e32 v87, vcc, v87, v132
	s_nop 1
	v_subb_co_u32_e64 v91, s[14:15], v85, v89, vcc
	v_sub_u32_e32 v85, v85, v89
	v_cmp_le_u32_e64 s[14:15], s29, v91
	v_subb_co_u32_e32 v85, vcc, v85, v81, vcc
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v87
	v_subrev_co_u32_e32 v87, vcc, s28, v87
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v85, vcc, 0, v85, vcc
	v_cmp_le_u32_e32 vcc, s29, v85
	v_cmp_eq_u32_e64 s[14:15], s29, v91
	s_nop 0
	v_cndmask_b32_e64 v89, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v87
	v_cndmask_b32_e64 v91, v93, v95, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v85
	s_nop 1
	v_cndmask_b32_e32 v85, v89, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	v_xor_b32_e32 v87, s70, v130
	s_nop 0
	v_cndmask_b32_e64 v85, 1, 2, vcc
	v_add_u32_e32 v85, v120, v85
	v_cmp_ne_u32_e32 vcc, 0, v91
	s_nop 1
	v_cndmask_b32_e32 v85, v120, v85, vcc
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v85, v85, v87
	v_xor_b32_e32 v89, v118, v120
	v_sub_u32_e32 v85, v85, v87
	v_xor_b32_e32 v87, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v89, s35, 0
	v_mul_hi_u32 v104, v89, s34
	v_lshl_add_u64 v[118:119], v[104:105], 0, v[118:119]
	v_mad_u64_u32 v[132:133], s[14:15], v87, s34, 0
	v_add_co_u32_e32 v91, vcc, v118, v132
	v_mad_u64_u32 v[130:131], s[14:15], v87, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v118, vcc, v119, v133, vcc
	v_mov_b32_e32 v119, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[130:131]
	v_mul_lo_u32 v91, s29, v118
	v_mul_lo_u32 v93, s28, v119
	v_mad_u64_u32 v[118:119], s[14:15], s28, v118, 0
	v_add3_u32 v91, v119, v93, v91
	v_sub_u32_e32 v93, v87, v91
	v_sub_co_u32_e32 v89, vcc, v89, v118
	v_xad_u32 v85, v85, v120, v46
	s_nop 0
	v_subb_co_u32_e64 v93, s[14:15], v93, v81, vcc
	v_subrev_co_u32_e64 v95, s[14:15], s28, v89
	v_subb_co_u32_e32 v87, vcc, v87, v91, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v104, s[26:27], 0, v93, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v104
	v_subb_co_u32_e64 v93, s[14:15], v93, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v109, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v95
	v_cmp_le_u32_e32 vcc, s29, v87
	v_mul_lo_u32 v85, v85, s33
	v_cndmask_b32_e64 v115, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v104
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v89
	v_cndmask_b32_e64 v109, v109, v115, s[26:27]
	v_subrev_co_u32_e64 v115, s[14:15], s28, v95
	v_lshl_add_u64 v[118:119], v[126:127], 0, 10
	s_nop 0
	v_subbrev_co_u32_e64 v93, s[14:15], 0, v93, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v109
	s_nop 1
	v_cndmask_b32_e64 v93, v104, v93, s[14:15]
	v_cndmask_b32_e64 v104, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v87
	s_nop 1
	v_cndmask_b32_e32 v91, v91, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v91
	v_cndmask_b32_e64 v91, v95, v115, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v89, v89, v91, vcc
	v_cndmask_b32_e32 v87, v87, v93, vcc
	v_xor_b32_e32 v89, v89, v120
	v_xor_b32_e32 v87, v87, v120
	v_sub_co_u32_e32 v89, vcc, v89, v120
	s_nop 1
	v_subb_co_u32_e32 v87, vcc, v87, v120, vcc
	v_ashrrev_i32_e32 v87, 31, v87
	v_and_b32_e32 v87, s44, v87
	v_add3_u32 v85, v87, v89, v85
	v_sub_co_u32_e32 v87, vcc, -11, v126
	s_nop 1
	v_subb_co_u32_e32 v89, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -10, v[126:127]
	s_nop 1
	v_cndmask_b32_e32 v121, v119, v89, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_cndmask_b32_e32 v120, v118, v87, vcc
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v89, v120, v130
	v_xor_b32_e32 v87, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v89, s35, 0
	v_mul_hi_u32 v104, v89, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[134:135], s[14:15], v87, s34, 0
	v_add_co_u32_e32 v91, vcc, v120, v134
	v_mad_u64_u32 v[132:133], s[14:15], v87, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v135, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v133, vcc, 0, v133, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[132:133]
	v_mul_lo_u32 v91, s29, v120
	v_mul_lo_u32 v93, s28, v121
	v_mad_u64_u32 v[132:133], s[14:15], s28, v120, 0
	v_add3_u32 v91, v133, v93, v91
	v_sub_co_u32_e32 v89, vcc, v89, v132
	s_nop 1
	v_subb_co_u32_e64 v93, s[14:15], v87, v91, vcc
	v_sub_u32_e32 v87, v87, v91
	v_cmp_le_u32_e64 s[14:15], s29, v93
	v_subb_co_u32_e32 v87, vcc, v87, v81, vcc
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v89
	v_subrev_co_u32_e32 v89, vcc, s28, v89
	s_nop 0
	v_cndmask_b32_e64 v104, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v87, vcc, 0, v87, vcc
	v_cmp_le_u32_e32 vcc, s29, v87
	v_cmp_eq_u32_e64 s[14:15], s29, v93
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v89
	v_cndmask_b32_e64 v93, v95, v104, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v89, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v87
	s_nop 1
	v_cndmask_b32_e32 v87, v91, v89, vcc
	v_cmp_ne_u32_e32 vcc, 0, v87
	v_xor_b32_e32 v89, s70, v130
	s_nop 0
	v_cndmask_b32_e64 v87, 1, 2, vcc
	v_add_u32_e32 v87, v120, v87
	v_cmp_ne_u32_e32 vcc, 0, v93
	s_nop 1
	v_cndmask_b32_e32 v87, v120, v87, vcc
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v87, v87, v89
	v_xor_b32_e32 v91, v118, v120
	v_sub_u32_e32 v87, v87, v89
	v_xor_b32_e32 v89, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v91, s35, 0
	v_mul_hi_u32 v104, v91, s34
	v_lshl_add_u64 v[118:119], v[104:105], 0, v[118:119]
	v_mad_u64_u32 v[132:133], s[14:15], v89, s34, 0
	v_add_co_u32_e32 v93, vcc, v118, v132
	v_mad_u64_u32 v[130:131], s[14:15], v89, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v118, vcc, v119, v133, vcc
	v_mov_b32_e32 v119, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[130:131]
	v_mul_lo_u32 v93, s29, v118
	v_mul_lo_u32 v95, s28, v119
	v_mad_u64_u32 v[118:119], s[14:15], s28, v118, 0
	v_add3_u32 v93, v119, v95, v93
	v_sub_u32_e32 v95, v89, v93
	v_sub_co_u32_e32 v91, vcc, v91, v118
	v_xad_u32 v87, v87, v120, v46
	s_nop 0
	v_subb_co_u32_e64 v95, s[14:15], v95, v81, vcc
	v_subrev_co_u32_e64 v104, s[14:15], s28, v91
	v_subb_co_u32_e32 v89, vcc, v89, v93, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v109, s[26:27], 0, v95, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v109
	v_subb_co_u32_e64 v95, s[14:15], v95, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v115, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v104
	v_cmp_le_u32_e32 vcc, s29, v89
	v_mul_lo_u32 v87, v87, s33
	v_cndmask_b32_e64 v118, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v109
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v91
	v_cndmask_b32_e64 v115, v115, v118, s[26:27]
	v_subrev_co_u32_e64 v118, s[14:15], s28, v104
	s_nop 1
	v_subbrev_co_u32_e64 v95, s[14:15], 0, v95, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v115
	s_nop 1
	v_cndmask_b32_e64 v95, v109, v95, s[14:15]
	v_cndmask_b32_e64 v109, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v89
	s_nop 1
	v_cndmask_b32_e32 v93, v93, v109, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	v_cndmask_b32_e64 v93, v104, v118, s[14:15]
	v_lshl_add_u64 v[118:119], v[126:127], 0, 11
	v_cndmask_b32_e32 v91, v91, v93, vcc
	v_cndmask_b32_e32 v89, v89, v95, vcc
	v_xor_b32_e32 v91, v91, v120
	v_xor_b32_e32 v89, v89, v120
	v_sub_co_u32_e32 v91, vcc, v91, v120
	s_nop 1
	v_subb_co_u32_e32 v89, vcc, v89, v120, vcc
	v_ashrrev_i32_e32 v89, 31, v89
	v_and_b32_e32 v89, s44, v89
	v_add3_u32 v89, v89, v91, v87
	v_sub_co_u32_e32 v87, vcc, -12, v126
	s_nop 1
	v_subb_co_u32_e32 v91, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -11, v[126:127]
	s_nop 1
	v_cndmask_b32_e32 v121, v119, v91, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_cndmask_b32_e32 v120, v118, v87, vcc
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v91, v120, v130
	v_xor_b32_e32 v87, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v91, s35, 0
	v_mul_hi_u32 v104, v91, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[134:135], s[14:15], v87, s34, 0
	v_add_co_u32_e32 v93, vcc, v120, v134
	v_mad_u64_u32 v[132:133], s[14:15], v87, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v135, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v133, vcc, 0, v133, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[132:133]
	v_mul_lo_u32 v93, s29, v120
	v_mul_lo_u32 v95, s28, v121
	v_mad_u64_u32 v[132:133], s[14:15], s28, v120, 0
	v_add3_u32 v93, v133, v95, v93
	v_sub_co_u32_e32 v91, vcc, v91, v132
	s_nop 1
	v_subb_co_u32_e64 v95, s[14:15], v87, v93, vcc
	v_sub_u32_e32 v87, v87, v93
	v_cmp_le_u32_e64 s[14:15], s29, v95
	v_subb_co_u32_e32 v87, vcc, v87, v81, vcc
	s_nop 0
	v_cndmask_b32_e64 v104, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v91
	v_subrev_co_u32_e32 v91, vcc, s28, v91
	s_nop 0
	v_cndmask_b32_e64 v109, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v87, vcc, 0, v87, vcc
	v_cmp_le_u32_e32 vcc, s29, v87
	v_cmp_eq_u32_e64 s[14:15], s29, v95
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v91
	v_cndmask_b32_e64 v95, v104, v109, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v87
	s_nop 1
	v_cndmask_b32_e32 v87, v93, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v87
	v_xor_b32_e32 v91, s70, v130
	s_nop 0
	v_cndmask_b32_e64 v87, 1, 2, vcc
	v_add_u32_e32 v87, v120, v87
	v_cmp_ne_u32_e32 vcc, 0, v95
	s_nop 1
	v_cndmask_b32_e32 v87, v120, v87, vcc
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v87, v87, v91
	v_xor_b32_e32 v93, v118, v120
	v_sub_u32_e32 v87, v87, v91
	v_xor_b32_e32 v91, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v93, s35, 0
	v_mul_hi_u32 v104, v93, s34
	v_lshl_add_u64 v[118:119], v[104:105], 0, v[118:119]
	v_mad_u64_u32 v[132:133], s[14:15], v91, s34, 0
	v_add_co_u32_e32 v95, vcc, v118, v132
	v_mad_u64_u32 v[130:131], s[14:15], v91, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v118, vcc, v119, v133, vcc
	v_mov_b32_e32 v119, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[130:131]
	v_mul_lo_u32 v95, s29, v118
	v_mul_lo_u32 v104, s28, v119
	v_mad_u64_u32 v[118:119], s[14:15], s28, v118, 0
	v_add3_u32 v95, v119, v104, v95
	v_sub_u32_e32 v104, v91, v95
	v_sub_co_u32_e32 v93, vcc, v93, v118
	v_xad_u32 v87, v87, v120, v46
	s_nop 0
	v_subb_co_u32_e64 v104, s[14:15], v104, v81, vcc
	v_subrev_co_u32_e64 v109, s[14:15], s28, v93
	v_subb_co_u32_e32 v91, vcc, v91, v95, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v115, s[26:27], 0, v104, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v115
	v_subb_co_u32_e64 v104, s[14:15], v104, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v118, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v109
	v_cmp_le_u32_e32 vcc, s29, v91
	v_mul_lo_u32 v87, v87, s33
	v_cndmask_b32_e64 v119, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v115
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v93
	v_cndmask_b32_e64 v118, v118, v119, s[26:27]
	v_subrev_co_u32_e64 v119, s[14:15], s28, v109
	s_nop 1
	v_subbrev_co_u32_e64 v104, s[14:15], 0, v104, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v118
	s_nop 1
	v_cndmask_b32_e64 v104, v115, v104, s[14:15]
	v_cndmask_b32_e64 v115, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v91
	s_nop 1
	v_cndmask_b32_e32 v95, v95, v115, vcc
	v_cmp_ne_u32_e32 vcc, 0, v95
	v_cndmask_b32_e64 v95, v109, v119, s[14:15]
	v_lshl_add_u64 v[118:119], v[126:127], 0, 12
	v_cndmask_b32_e32 v93, v93, v95, vcc
	v_cndmask_b32_e32 v91, v91, v104, vcc
	v_xor_b32_e32 v93, v93, v120
	v_xor_b32_e32 v91, v91, v120
	v_sub_co_u32_e32 v93, vcc, v93, v120
	s_nop 1
	v_subb_co_u32_e32 v91, vcc, v91, v120, vcc
	v_ashrrev_i32_e32 v91, 31, v91
	v_and_b32_e32 v91, s44, v91
	v_add3_u32 v91, v91, v93, v87
	v_sub_co_u32_e32 v87, vcc, -13, v126
	s_nop 1
	v_subb_co_u32_e32 v93, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -12, v[126:127]
	s_nop 1
	v_cndmask_b32_e32 v121, v119, v93, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_cndmask_b32_e32 v120, v118, v87, vcc
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v93, v120, v130
	v_xor_b32_e32 v87, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v93, s35, 0
	v_mul_hi_u32 v104, v93, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[134:135], s[14:15], v87, s34, 0
	v_add_co_u32_e32 v95, vcc, v120, v134
	v_mad_u64_u32 v[132:133], s[14:15], v87, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v135, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v133, vcc, 0, v133, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[132:133]
	v_mul_lo_u32 v95, s29, v120
	v_mul_lo_u32 v104, s28, v121
	v_mad_u64_u32 v[132:133], s[14:15], s28, v120, 0
	v_add3_u32 v95, v133, v104, v95
	v_sub_co_u32_e32 v93, vcc, v93, v132
	s_nop 1
	v_subb_co_u32_e64 v104, s[14:15], v87, v95, vcc
	v_sub_u32_e32 v87, v87, v95
	v_cmp_le_u32_e64 s[14:15], s29, v104
	v_subb_co_u32_e32 v87, vcc, v87, v81, vcc
	s_nop 0
	v_cndmask_b32_e64 v109, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v93
	v_subrev_co_u32_e32 v93, vcc, s28, v93
	s_nop 0
	v_cndmask_b32_e64 v115, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v87, vcc, 0, v87, vcc
	v_cmp_le_u32_e32 vcc, s29, v87
	v_cmp_eq_u32_e64 s[14:15], s29, v104
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v93
	v_cndmask_b32_e64 v104, v109, v115, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v87
	s_nop 1
	v_cndmask_b32_e32 v87, v95, v93, vcc
	v_cmp_ne_u32_e32 vcc, 0, v87
	v_xor_b32_e32 v93, s70, v130
	s_nop 0
	v_cndmask_b32_e64 v87, 1, 2, vcc
	v_add_u32_e32 v87, v120, v87
	v_cmp_ne_u32_e32 vcc, 0, v104
	s_nop 1
	v_cndmask_b32_e32 v87, v120, v87, vcc
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v87, v87, v93
	v_xor_b32_e32 v95, v118, v120
	v_sub_u32_e32 v87, v87, v93
	v_xor_b32_e32 v93, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v95, s35, 0
	v_mul_hi_u32 v104, v95, s34
	v_lshl_add_u64 v[118:119], v[104:105], 0, v[118:119]
	v_mad_u64_u32 v[132:133], s[14:15], v93, s34, 0
	v_add_co_u32_e32 v104, vcc, v118, v132
	v_mad_u64_u32 v[130:131], s[14:15], v93, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v118, vcc, v119, v133, vcc
	v_mov_b32_e32 v119, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[130:131]
	v_mul_lo_u32 v104, s29, v118
	v_mul_lo_u32 v109, s28, v119
	v_mad_u64_u32 v[118:119], s[14:15], s28, v118, 0
	v_add3_u32 v104, v119, v109, v104
	v_sub_u32_e32 v109, v93, v104
	v_sub_co_u32_e32 v95, vcc, v95, v118
	v_xad_u32 v87, v87, v120, v46
	s_nop 0
	v_subb_co_u32_e64 v109, s[14:15], v109, v81, vcc
	v_subrev_co_u32_e64 v115, s[14:15], s28, v95
	v_subb_co_u32_e32 v93, vcc, v93, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v118, s[26:27], 0, v109, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v118
	v_subb_co_u32_e64 v109, s[14:15], v109, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v119, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v115
	v_cmp_le_u32_e32 vcc, s29, v93
	v_mul_lo_u32 v87, v87, s33
	v_cndmask_b32_e64 v121, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v118
	v_cndmask_b32_e64 v104, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v95
	v_cndmask_b32_e64 v119, v119, v121, s[26:27]
	v_subrev_co_u32_e64 v121, s[14:15], s28, v115
	s_nop 1
	v_subbrev_co_u32_e64 v109, s[14:15], 0, v109, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v119
	s_nop 1
	v_cndmask_b32_e64 v109, v118, v109, s[14:15]
	v_cndmask_b32_e64 v118, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v93
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v118, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_cndmask_b32_e64 v104, v115, v121, s[14:15]
	v_lshl_add_u64 v[118:119], v[126:127], 0, 13
	v_cndmask_b32_e32 v95, v95, v104, vcc
	v_cndmask_b32_e32 v93, v93, v109, vcc
	v_xor_b32_e32 v95, v95, v120
	v_xor_b32_e32 v93, v93, v120
	v_sub_co_u32_e32 v95, vcc, v95, v120
	s_nop 1
	v_subb_co_u32_e32 v93, vcc, v93, v120, vcc
	v_ashrrev_i32_e32 v93, 31, v93
	v_and_b32_e32 v93, s44, v93
	v_add3_u32 v93, v93, v95, v87
	v_sub_co_u32_e32 v87, vcc, -14, v126
	s_nop 1
	v_subb_co_u32_e32 v95, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -13, v[126:127]
	s_nop 1
	v_cndmask_b32_e32 v121, v119, v95, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_cndmask_b32_e32 v120, v118, v87, vcc
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v95, v120, v130
	v_xor_b32_e32 v87, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v95, s35, 0
	v_mul_hi_u32 v104, v95, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[134:135], s[14:15], v87, s34, 0
	v_add_co_u32_e32 v104, vcc, v120, v134
	v_mad_u64_u32 v[132:133], s[14:15], v87, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v135, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v133, vcc, 0, v133, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[132:133]
	v_mul_lo_u32 v104, s29, v120
	v_mul_lo_u32 v109, s28, v121
	v_mad_u64_u32 v[132:133], s[14:15], s28, v120, 0
	v_add3_u32 v104, v133, v109, v104
	v_sub_co_u32_e32 v95, vcc, v95, v132
	s_nop 1
	v_subb_co_u32_e64 v109, s[14:15], v87, v104, vcc
	v_sub_u32_e32 v87, v87, v104
	v_cmp_le_u32_e64 s[14:15], s29, v109
	v_subb_co_u32_e32 v87, vcc, v87, v81, vcc
	s_nop 0
	v_cndmask_b32_e64 v115, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v95
	v_subrev_co_u32_e32 v95, vcc, s28, v95
	s_nop 0
	v_cndmask_b32_e64 v121, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v87, vcc, 0, v87, vcc
	v_cmp_le_u32_e32 vcc, s29, v87
	v_cmp_eq_u32_e64 s[14:15], s29, v109
	s_nop 0
	v_cndmask_b32_e64 v104, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v95
	v_cndmask_b32_e64 v109, v115, v121, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v87
	s_nop 1
	v_cndmask_b32_e32 v87, v104, v95, vcc
	v_cmp_ne_u32_e32 vcc, 0, v87
	v_xor_b32_e32 v95, s70, v130
	s_nop 0
	v_cndmask_b32_e64 v87, 1, 2, vcc
	v_add_u32_e32 v87, v120, v87
	v_cmp_ne_u32_e32 vcc, 0, v109
	s_nop 1
	v_cndmask_b32_e32 v87, v120, v87, vcc
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v87, v87, v95
	v_xor_b32_e32 v109, v118, v120
	v_sub_u32_e32 v87, v87, v95
	v_xor_b32_e32 v95, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v109, s35, 0
	v_mul_hi_u32 v104, v109, s34
	v_lshl_add_u64 v[118:119], v[104:105], 0, v[118:119]
	v_mad_u64_u32 v[132:133], s[14:15], v95, s34, 0
	v_add_co_u32_e32 v104, vcc, v118, v132
	v_mad_u64_u32 v[130:131], s[14:15], v95, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v118, vcc, v119, v133, vcc
	v_mov_b32_e32 v119, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[130:131]
	v_mul_lo_u32 v104, s29, v118
	v_mul_lo_u32 v115, s28, v119
	v_mad_u64_u32 v[118:119], s[14:15], s28, v118, 0
	v_add3_u32 v104, v119, v115, v104
	v_sub_u32_e32 v115, v95, v104
	v_sub_co_u32_e32 v109, vcc, v109, v118
	v_xad_u32 v87, v87, v120, v46
	s_nop 0
	v_subb_co_u32_e64 v115, s[14:15], v115, v81, vcc
	v_subrev_co_u32_e64 v118, s[14:15], s28, v109
	v_subb_co_u32_e32 v95, vcc, v95, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v119, s[26:27], 0, v115, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v119
	v_subb_co_u32_e64 v115, s[14:15], v115, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v121, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v118
	v_cmp_le_u32_e32 vcc, s29, v95
	v_mul_lo_u32 v87, v87, s33
	v_cndmask_b32_e64 v130, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v119
	v_cndmask_b32_e64 v104, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v109
	v_cndmask_b32_e64 v121, v121, v130, s[26:27]
	v_subrev_co_u32_e64 v130, s[14:15], s28, v118
	s_nop 1
	v_subbrev_co_u32_e64 v115, s[14:15], 0, v115, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v121
	s_nop 1
	v_cndmask_b32_e64 v115, v119, v115, s[14:15]
	v_cndmask_b32_e64 v119, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v95
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v119, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_cndmask_b32_e64 v104, v118, v130, s[14:15]
	v_lshl_add_u64 v[118:119], v[126:127], 0, 14
	v_cndmask_b32_e32 v104, v109, v104, vcc
	v_cndmask_b32_e32 v95, v95, v115, vcc
	v_xor_b32_e32 v104, v104, v120
	v_xor_b32_e32 v95, v95, v120
	v_sub_co_u32_e32 v104, vcc, v104, v120
	s_nop 1
	v_subb_co_u32_e32 v95, vcc, v95, v120, vcc
	v_ashrrev_i32_e32 v95, 31, v95
	v_and_b32_e32 v95, s44, v95
	v_add3_u32 v95, v95, v104, v87
	v_sub_co_u32_e32 v87, vcc, -15, v126
	s_nop 1
	v_subb_co_u32_e32 v104, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -14, v[126:127]
	s_nop 1
	v_cndmask_b32_e32 v121, v119, v104, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_cndmask_b32_e32 v120, v118, v87, vcc
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v109, v120, v130
	v_xor_b32_e32 v87, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v109, s35, 0
	v_mul_hi_u32 v104, v109, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[134:135], s[14:15], v87, s34, 0
	v_add_co_u32_e32 v104, vcc, v120, v134
	v_mad_u64_u32 v[132:133], s[14:15], v87, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v135, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v133, vcc, 0, v133, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[132:133]
	v_mul_lo_u32 v104, s29, v120
	v_mul_lo_u32 v115, s28, v121
	v_mad_u64_u32 v[132:133], s[14:15], s28, v120, 0
	v_add3_u32 v104, v133, v115, v104
	v_sub_co_u32_e32 v109, vcc, v109, v132
	s_nop 1
	v_subb_co_u32_e64 v115, s[14:15], v87, v104, vcc
	v_sub_u32_e32 v87, v87, v104
	v_subb_co_u32_e32 v87, vcc, v87, v81, vcc
	v_subrev_co_u32_e32 v104, vcc, s28, v109
	v_cmp_le_u32_e64 s[14:15], s29, v115
	s_nop 0
	v_subbrev_co_u32_e32 v87, vcc, 0, v87, vcc
	v_cmp_le_u32_e32 vcc, s29, v87
	v_cndmask_b32_e64 v121, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v109
	v_cndmask_b32_e64 v109, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v104
	v_cndmask_b32_e64 v131, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v115
	v_cndmask_b32_e64 v104, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v87
	v_cndmask_b32_e64 v115, v121, v131, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v87, v109, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v87
	v_xor_b32_e32 v104, s70, v130
	s_nop 0
	v_cndmask_b32_e64 v87, 1, 2, vcc
	v_add_u32_e32 v87, v120, v87
	v_cmp_ne_u32_e32 vcc, 0, v115
	s_nop 1
	v_cndmask_b32_e32 v87, v120, v87, vcc
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v87, v87, v104
	v_xor_b32_e32 v115, v118, v120
	v_sub_u32_e32 v87, v87, v104
	v_xor_b32_e32 v109, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v115, s35, 0
	v_mul_hi_u32 v104, v115, s34
	v_lshl_add_u64 v[118:119], v[104:105], 0, v[118:119]
	v_mad_u64_u32 v[132:133], s[14:15], v109, s34, 0
	v_add_co_u32_e32 v104, vcc, v118, v132
	v_mad_u64_u32 v[130:131], s[14:15], v109, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v118, vcc, v119, v133, vcc
	v_mov_b32_e32 v119, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[130:131]
	v_mul_lo_u32 v104, s29, v118
	v_mul_lo_u32 v121, s28, v119
	v_mad_u64_u32 v[118:119], s[14:15], s28, v118, 0
	v_add3_u32 v104, v119, v121, v104
	v_sub_u32_e32 v119, v109, v104
	v_sub_co_u32_e32 v115, vcc, v115, v118
	v_xad_u32 v87, v87, v120, v46
	s_nop 0
	v_subb_co_u32_e64 v118, s[14:15], v119, v81, vcc
	v_subrev_co_u32_e64 v119, s[14:15], s28, v115
	v_subb_co_u32_e32 v104, vcc, v109, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v121, s[26:27], 0, v118, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v121
	v_subb_co_u32_e64 v118, s[14:15], v118, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v130, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v119
	v_cmp_le_u32_e32 vcc, s29, v104
	v_mul_lo_u32 v87, v87, s33
	v_cndmask_b32_e64 v131, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v121
	v_cndmask_b32_e64 v109, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v115
	v_cndmask_b32_e64 v130, v130, v131, s[26:27]
	v_subrev_co_u32_e64 v131, s[14:15], s28, v119
	s_nop 1
	v_subbrev_co_u32_e64 v118, s[14:15], 0, v118, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v130
	s_nop 1
	v_cndmask_b32_e64 v118, v121, v118, s[14:15]
	v_cndmask_b32_e64 v121, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v109, v109, v121, vcc
	v_cmp_ne_u32_e32 vcc, 0, v109
	v_cndmask_b32_e64 v109, v119, v131, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v109, v115, v109, vcc
	v_cndmask_b32_e32 v104, v104, v118, vcc
	v_xor_b32_e32 v109, v109, v120
	v_xor_b32_e32 v104, v104, v120
	v_sub_co_u32_e32 v109, vcc, v109, v120
	v_lshl_add_u64 v[118:119], v[126:127], 0, 15
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, v104, v120, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v109, v104, v109, v87
	v_sub_co_u32_e32 v87, vcc, -16, v126
	s_nop 1
	v_subb_co_u32_e32 v104, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -15, v[126:127]
	s_nop 1
	v_cndmask_b32_e32 v121, v119, v104, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_cndmask_b32_e32 v120, v118, v87, vcc
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v115, v120, v130
	v_xor_b32_e32 v87, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v115, s35, 0
	v_mul_hi_u32 v104, v115, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[134:135], s[14:15], v87, s34, 0
	v_add_co_u32_e32 v104, vcc, v120, v134
	v_mad_u64_u32 v[132:133], s[14:15], v87, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v135, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v133, vcc, 0, v133, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[132:133]
	v_mul_lo_u32 v104, s29, v120
	v_mul_lo_u32 v121, s28, v121
	v_mad_u64_u32 v[132:133], s[14:15], s28, v120, 0
	v_add3_u32 v104, v133, v121, v104
	v_sub_co_u32_e32 v115, vcc, v115, v132
	s_nop 1
	v_subb_co_u32_e64 v121, s[14:15], v87, v104, vcc
	v_sub_u32_e32 v87, v87, v104
	v_subb_co_u32_e32 v87, vcc, v87, v81, vcc
	v_subrev_co_u32_e32 v104, vcc, s28, v115
	v_cmp_le_u32_e64 s[14:15], s29, v121
	s_nop 0
	v_subbrev_co_u32_e32 v87, vcc, 0, v87, vcc
	v_cmp_le_u32_e32 vcc, s29, v87
	v_cndmask_b32_e64 v131, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v115
	v_cndmask_b32_e64 v115, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v104
	v_cndmask_b32_e64 v132, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v121
	v_cndmask_b32_e64 v104, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v87
	v_cndmask_b32_e64 v121, v131, v132, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v87, v115, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v87
	v_xor_b32_e32 v104, s70, v130
	s_nop 0
	v_cndmask_b32_e64 v87, 1, 2, vcc
	v_add_u32_e32 v87, v120, v87
	v_cmp_ne_u32_e32 vcc, 0, v121
	s_nop 1
	v_cndmask_b32_e32 v87, v120, v87, vcc
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v87, v87, v104
	v_xor_b32_e32 v121, v118, v120
	v_sub_u32_e32 v87, v87, v104
	v_xor_b32_e32 v115, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v121, s35, 0
	v_mul_hi_u32 v104, v121, s34
	v_lshl_add_u64 v[118:119], v[104:105], 0, v[118:119]
	v_mad_u64_u32 v[132:133], s[14:15], v115, s34, 0
	v_add_co_u32_e32 v104, vcc, v118, v132
	v_mad_u64_u32 v[130:131], s[14:15], v115, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v118, vcc, v119, v133, vcc
	v_mov_b32_e32 v119, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[130:131]
	v_mul_lo_u32 v104, s29, v118
	v_mul_lo_u32 v130, s28, v119
	v_mad_u64_u32 v[118:119], s[14:15], s28, v118, 0
	v_add3_u32 v104, v119, v130, v104
	v_sub_u32_e32 v119, v115, v104
	v_sub_co_u32_e32 v118, vcc, v121, v118
	v_xad_u32 v87, v87, v120, v46
	s_nop 0
	v_subb_co_u32_e64 v119, s[14:15], v119, v81, vcc
	v_subrev_co_u32_e64 v121, s[14:15], s28, v118
	v_subb_co_u32_e32 v104, vcc, v115, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v130, s[26:27], 0, v119, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v130
	v_subb_co_u32_e64 v119, s[14:15], v119, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v131, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v121
	v_cmp_le_u32_e32 vcc, s29, v104
	v_mul_lo_u32 v87, v87, s33
	v_cndmask_b32_e64 v132, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v130
	v_cndmask_b32_e64 v115, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v118
	v_cndmask_b32_e64 v131, v131, v132, s[26:27]
	v_subrev_co_u32_e64 v132, s[14:15], s28, v121
	s_nop 1
	v_subbrev_co_u32_e64 v119, s[14:15], 0, v119, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v131
	s_nop 1
	v_cndmask_b32_e64 v119, v130, v119, s[14:15]
	v_cndmask_b32_e64 v130, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v115, v115, v130, vcc
	v_cmp_ne_u32_e32 vcc, 0, v115
	v_cndmask_b32_e64 v115, v121, v132, s[14:15]
	s_movk_i32 s14, 0xf3ff
	v_cndmask_b32_e32 v115, v118, v115, vcc
	v_cndmask_b32_e32 v104, v104, v119, vcc
	v_xor_b32_e32 v115, v115, v120
	v_xor_b32_e32 v104, v104, v120
	v_sub_co_u32_e32 v115, vcc, v115, v120
	v_and_b32_e32 v132, 0xff, v114
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, v104, v120, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v104, v104, v115, v87
	buffer_load_ubyte v83, v83, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v87, v85, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v85, v89, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v89, v91, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v91, v93, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v95, v95, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v93, v109, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v109, v104, s[36:39], 0 offen
	v_sub_co_u32_e32 v104, vcc, s14, v128
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[166:169], v[28:31], v[4:7], v116, v132 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshl_add_u64 v[114:115], v[128:129], 0, s[94:95]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[162:165], v[28:31], v[0:3], v116, v132 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_subb_co_u32_e32 v116, vcc, -1, v129, vcc
	v_cmp_gt_i32_e32 vcc, 0, v115
	s_nop 1
	v_cndmask_b32_e32 v119, v115, v116, vcc
	v_ashrrev_i32_e32 v120, 31, v119
	v_cndmask_b32_e32 v118, v114, v104, vcc
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v121, v118, v120
	v_xor_b32_e32 v116, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v121, s35, 0
	v_mul_hi_u32 v104, v121, s34
	v_lshl_add_u64 v[118:119], v[104:105], 0, v[118:119]
	v_mad_u64_u32 v[134:135], s[14:15], v116, s34, 0
	v_add_co_u32_e32 v104, vcc, v118, v134
	v_mad_u64_u32 v[130:131], s[14:15], v116, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v118, vcc, v119, v135, vcc
	v_mov_b32_e32 v119, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[130:131]
	v_mul_lo_u32 v104, s29, v118
	v_mul_lo_u32 v119, s28, v119
	v_mad_u64_u32 v[130:131], s[14:15], s28, v118, 0
	v_add3_u32 v104, v131, v119, v104
	v_sub_co_u32_e32 v119, vcc, v121, v130
	s_nop 1
	v_subb_co_u32_e64 v121, s[14:15], v116, v104, vcc
	v_sub_u32_e32 v104, v116, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v116, vcc, s28, v119
	v_cmp_le_u32_e64 s[14:15], s29, v121
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cmp_le_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v130, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v119
	v_cndmask_b32_e64 v119, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v116
	v_cndmask_b32_e64 v131, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v121
	v_cndmask_b32_e64 v116, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v121, v130, v131, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v104, v119, v116, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v116, s70, v120
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v118, v104
	v_cmp_ne_u32_e32 vcc, 0, v121
	s_nop 1
	v_cndmask_b32_e32 v104, v118, v104, vcc
	v_ashrrev_i32_e32 v118, 31, v115
	v_mov_b32_e32 v119, v118
	v_xor_b32_e32 v104, v104, v116
	v_lshl_add_u64 v[114:115], v[114:115], 0, v[118:119]
	v_sub_u32_e32 v104, v104, v116
	v_xor_b32_e32 v133, v114, v118
	v_xad_u32 v116, v104, v118, v46
	v_xor_b32_e32 v119, v115, v118
	v_mad_u64_u32 v[114:115], s[14:15], v133, s35, 0
	v_mul_hi_u32 v104, v133, s34
	v_lshl_add_u64 v[114:115], v[104:105], 0, v[114:115]
	v_mad_u64_u32 v[130:131], s[14:15], v119, s34, 0
	v_add_co_u32_e32 v104, vcc, v114, v130
	v_mad_u64_u32 v[120:121], s[14:15], v119, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v114, vcc, v115, v131, vcc
	v_mov_b32_e32 v115, s57
	s_nop 0
	v_addc_co_u32_e32 v121, vcc, 0, v121, vcc
	v_lshl_add_u64 v[114:115], v[114:115], 0, v[120:121]
	v_mul_lo_u32 v104, s29, v114
	v_mul_lo_u32 v120, s28, v115
	v_mad_u64_u32 v[114:115], s[14:15], s28, v114, 0
	v_add3_u32 v104, v115, v120, v104
	v_sub_u32_e32 v115, v119, v104
	v_sub_co_u32_e32 v114, vcc, v133, v114
	s_nop 1
	v_subb_co_u32_e64 v115, s[14:15], v115, v81, vcc
	v_subrev_co_u32_e64 v120, s[14:15], s28, v114
	v_subb_co_u32_e32 v104, vcc, v119, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v121, s[26:27], 0, v115, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v121
	v_subb_co_u32_e64 v115, s[14:15], v115, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v130, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v120
	v_cmp_le_u32_e32 vcc, s29, v104
	s_nop 0
	v_cndmask_b32_e64 v131, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v121
	v_cndmask_b32_e64 v119, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v114
	v_cndmask_b32_e64 v130, v130, v131, s[26:27]
	v_subrev_co_u32_e64 v131, s[14:15], s28, v120
	s_nop 1
	v_subbrev_co_u32_e64 v115, s[14:15], 0, v115, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v130
	s_nop 1
	v_cndmask_b32_e64 v115, v121, v115, s[14:15]
	v_cndmask_b32_e64 v121, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v119, v119, v121, vcc
	v_cmp_ne_u32_e32 vcc, 0, v119
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v115, vcc
	v_cndmask_b32_e64 v115, v120, v131, s[14:15]
	v_cndmask_b32_e32 v114, v114, v115, vcc
	v_xor_b32_e32 v114, v114, v118
	v_xor_b32_e32 v104, v104, v118
	v_sub_co_u32_e32 v114, vcc, v114, v118
	s_mov_b64 s[14:15], 0x401
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, v104, v118, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_mul_lo_u32 v115, v116, s33
	v_lshl_add_u64 v[118:119], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xfbfe
	v_add3_u32 v116, v104, v114, v115
	v_sub_co_u32_e32 v104, vcc, s14, v126
	v_lshl_add_u64 v[114:115], v[126:127], 0, s[86:87]
	s_nop 0
	v_subb_co_u32_e32 v121, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -1, v[114:115]
	s_nop 1
	v_cndmask_b32_e32 v121, v119, v121, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_cndmask_b32_e32 v120, v118, v104, vcc
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v133, v120, v130
	v_xor_b32_e32 v131, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v133, s35, 0
	v_mul_hi_u32 v104, v133, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[136:137], s[14:15], v131, s34, 0
	v_add_co_u32_e32 v104, vcc, v120, v136
	v_mad_u64_u32 v[134:135], s[14:15], v131, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v137, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v135, vcc, 0, v135, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[134:135]
	v_mul_lo_u32 v104, s29, v120
	v_mul_lo_u32 v121, s28, v121
	v_mad_u64_u32 v[134:135], s[14:15], s28, v120, 0
	v_add3_u32 v104, v135, v121, v104
	v_sub_co_u32_e32 v121, vcc, v133, v134
	s_nop 1
	v_subb_co_u32_e64 v133, s[14:15], v131, v104, vcc
	v_sub_u32_e32 v104, v131, v104
	v_cmp_le_u32_e64 s[14:15], s29, v133
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	s_nop 0
	v_cndmask_b32_e64 v134, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v121
	v_subrev_co_u32_e32 v121, vcc, s28, v121
	s_nop 0
	v_cndmask_b32_e64 v135, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cmp_le_u32_e32 vcc, s29, v104
	v_cmp_eq_u32_e64 s[14:15], s29, v133
	s_nop 0
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v121
	v_cndmask_b32_e64 v133, v134, v135, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v121, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v104, v131, v121, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	s_nop 1
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v120, v104
	v_cmp_ne_u32_e32 vcc, 0, v133
	s_nop 1
	v_cndmask_b32_e32 v104, v120, v104, vcc
	v_xor_b32_e32 v120, s70, v130
	v_xor_b32_e32 v104, v104, v120
	v_sub_u32_e32 v104, v104, v120
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v136, v118, v120
	v_xad_u32 v133, v104, v120, v46
	v_xor_b32_e32 v121, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v136, s35, 0
	v_mul_hi_u32 v104, v136, s34
	v_lshl_add_u64 v[118:119], v[104:105], 0, v[118:119]
	v_mad_u64_u32 v[134:135], s[14:15], v121, s34, 0
	v_add_co_u32_e32 v104, vcc, v118, v134
	v_mad_u64_u32 v[130:131], s[14:15], v121, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v118, vcc, v119, v135, vcc
	v_mov_b32_e32 v119, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[130:131]
	v_mul_lo_u32 v104, s29, v118
	v_mul_lo_u32 v130, s28, v119
	v_mad_u64_u32 v[118:119], s[14:15], s28, v118, 0
	v_add3_u32 v104, v119, v130, v104
	v_sub_u32_e32 v119, v121, v104
	v_sub_co_u32_e32 v118, vcc, v136, v118
	s_nop 1
	v_subb_co_u32_e64 v119, s[14:15], v119, v81, vcc
	v_subrev_co_u32_e64 v130, s[14:15], s28, v118
	v_subb_co_u32_e32 v104, vcc, v121, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[26:27], 0, v119, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v131
	v_subb_co_u32_e64 v119, s[14:15], v119, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v134, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v130
	v_cmp_le_u32_e32 vcc, s29, v104
	s_nop 0
	v_cndmask_b32_e64 v135, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v131
	v_cndmask_b32_e64 v121, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v118
	v_cndmask_b32_e64 v134, v134, v135, s[26:27]
	v_subrev_co_u32_e64 v135, s[14:15], s28, v130
	s_nop 1
	v_subbrev_co_u32_e64 v119, s[14:15], 0, v119, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v134
	s_nop 1
	v_cndmask_b32_e64 v119, v131, v119, s[14:15]
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v121, v121, v131, vcc
	v_cmp_ne_u32_e32 vcc, 0, v121
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v119, vcc
	v_cndmask_b32_e64 v119, v130, v135, s[14:15]
	v_cndmask_b32_e32 v118, v118, v119, vcc
	v_xor_b32_e32 v118, v118, v120
	v_xor_b32_e32 v104, v104, v120
	v_sub_co_u32_e32 v118, vcc, v118, v120
	s_mov_b64 s[14:15], 0x402
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, v104, v120, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_mul_lo_u32 v119, v133, s33
	v_lshl_add_u64 v[120:121], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xfbfd
	v_add3_u32 v118, v104, v118, v119
	v_sub_co_u32_e32 v104, vcc, s14, v126
	s_nop 1
	v_subb_co_u32_e32 v119, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -2, v[114:115]
	s_nop 1
	v_cndmask_b32_e32 v131, v121, v119, vcc
	v_ashrrev_i32_e32 v134, 31, v131
	v_cndmask_b32_e32 v130, v120, v104, vcc
	v_mov_b32_e32 v135, v134
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[134:135]
	v_xor_b32_e32 v133, v130, v134
	v_xor_b32_e32 v119, v131, v134
	v_mad_u64_u32 v[130:131], s[14:15], v133, s35, 0
	v_mul_hi_u32 v104, v133, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[138:139], s[14:15], v119, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v138
	v_mad_u64_u32 v[136:137], s[14:15], v119, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v130, vcc, v131, v139, vcc
	v_mov_b32_e32 v131, s57
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[136:137]
	v_mul_lo_u32 v104, s29, v130
	v_mul_lo_u32 v131, s28, v131
	v_mad_u64_u32 v[136:137], s[14:15], s28, v130, 0
	v_add3_u32 v104, v137, v131, v104
	v_sub_co_u32_e32 v131, vcc, v133, v136
	s_nop 1
	v_subb_co_u32_e64 v133, s[14:15], v119, v104, vcc
	v_sub_u32_e32 v104, v119, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v119, vcc, s28, v131
	v_cmp_le_u32_e64 s[14:15], s29, v133
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cmp_le_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v135, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v131
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v119
	v_cndmask_b32_e64 v136, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v133
	v_cndmask_b32_e64 v119, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v133, v135, v136, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v104, v131, v119, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v119, s70, v134
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v130, v104
	v_cmp_ne_u32_e32 vcc, 0, v133
	s_nop 1
	v_cndmask_b32_e32 v104, v130, v104, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_mov_b32_e32 v131, v130
	v_xor_b32_e32 v104, v104, v119
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_sub_u32_e32 v104, v104, v119
	v_xor_b32_e32 v133, v120, v130
	v_xad_u32 v119, v104, v130, v46
	v_xor_b32_e32 v131, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v133, s35, 0
	v_mul_hi_u32 v104, v133, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[136:137], s[14:15], v131, s34, 0
	v_add_co_u32_e32 v104, vcc, v120, v136
	v_mad_u64_u32 v[134:135], s[14:15], v131, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v137, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v135, vcc, 0, v135, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[134:135]
	v_mul_lo_u32 v104, s29, v120
	v_mul_lo_u32 v134, s28, v121
	v_mad_u64_u32 v[120:121], s[14:15], s28, v120, 0
	v_add3_u32 v104, v121, v134, v104
	v_sub_u32_e32 v121, v131, v104
	v_sub_co_u32_e32 v120, vcc, v133, v120
	v_mul_lo_u32 v119, v119, s33
	s_nop 0
	v_subb_co_u32_e64 v121, s[14:15], v121, v81, vcc
	v_subrev_co_u32_e64 v133, s[14:15], s28, v120
	v_subb_co_u32_e32 v104, vcc, v131, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v134, s[26:27], 0, v121, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v134
	v_subb_co_u32_e64 v121, s[14:15], v121, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v135, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v133
	v_cmp_le_u32_e32 vcc, s29, v104
	s_nop 0
	v_cndmask_b32_e64 v136, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v134
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v120
	v_cndmask_b32_e64 v135, v135, v136, s[26:27]
	v_subrev_co_u32_e64 v136, s[14:15], s28, v133
	s_nop 1
	v_subbrev_co_u32_e64 v121, s[14:15], 0, v121, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v135
	s_nop 1
	v_cndmask_b32_e64 v121, v134, v121, s[14:15]
	v_cndmask_b32_e64 v134, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v131, v131, v134, vcc
	v_cmp_ne_u32_e32 vcc, 0, v131
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v121, vcc
	v_cndmask_b32_e64 v121, v133, v136, s[14:15]
	v_cndmask_b32_e32 v120, v120, v121, vcc
	v_xor_b32_e32 v120, v120, v130
	v_xor_b32_e32 v104, v104, v130
	v_sub_co_u32_e32 v120, vcc, v120, v130
	s_mov_b64 s[14:15], 0x403
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, v104, v130, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v119, v104, v120, v119
	v_lshl_add_u64 v[120:121], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xfbfc
	v_sub_co_u32_e32 v104, vcc, s14, v126
	s_nop 1
	v_subb_co_u32_e32 v130, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -3, v[114:115]
	s_nop 1
	v_cndmask_b32_e32 v131, v121, v130, vcc
	v_ashrrev_i32_e32 v134, 31, v131
	v_cndmask_b32_e32 v130, v120, v104, vcc
	v_mov_b32_e32 v135, v134
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[134:135]
	v_xor_b32_e32 v135, v130, v134
	v_xor_b32_e32 v133, v131, v134
	v_mad_u64_u32 v[130:131], s[14:15], v135, s35, 0
	v_mul_hi_u32 v104, v135, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[138:139], s[14:15], v133, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v138
	v_mad_u64_u32 v[136:137], s[14:15], v133, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v130, vcc, v131, v139, vcc
	v_mov_b32_e32 v131, s57
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[136:137]
	v_mul_lo_u32 v104, s29, v130
	v_mul_lo_u32 v131, s28, v131
	v_mad_u64_u32 v[136:137], s[14:15], s28, v130, 0
	v_add3_u32 v104, v137, v131, v104
	v_sub_co_u32_e32 v131, vcc, v135, v136
	s_nop 1
	v_subb_co_u32_e64 v135, s[14:15], v133, v104, vcc
	v_sub_u32_e32 v104, v133, v104
	v_cmp_le_u32_e64 s[14:15], s29, v135
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	s_nop 0
	v_cndmask_b32_e64 v136, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v131
	v_subrev_co_u32_e32 v131, vcc, s28, v131
	s_nop 0
	v_cndmask_b32_e64 v137, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cmp_le_u32_e32 vcc, s29, v104
	v_cmp_eq_u32_e64 s[14:15], s29, v135
	s_nop 0
	v_cndmask_b32_e64 v133, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v131
	v_cndmask_b32_e64 v135, v136, v137, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v104, v133, v131, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	s_nop 1
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v130, v104
	v_cmp_ne_u32_e32 vcc, 0, v135
	s_nop 1
	v_cndmask_b32_e32 v104, v130, v104, vcc
	v_xor_b32_e32 v130, s70, v134
	v_xor_b32_e32 v104, v104, v130
	v_sub_u32_e32 v104, v104, v130
	v_ashrrev_i32_e32 v130, 31, v121
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v138, v120, v130
	v_xad_u32 v133, v104, v130, v46
	v_xor_b32_e32 v131, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v138, s35, 0
	v_mul_hi_u32 v104, v138, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[136:137], s[14:15], v131, s34, 0
	v_add_co_u32_e32 v104, vcc, v120, v136
	v_mad_u64_u32 v[134:135], s[14:15], v131, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v137, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v135, vcc, 0, v135, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[134:135]
	v_mul_lo_u32 v104, s29, v120
	v_mul_lo_u32 v134, s28, v121
	v_mad_u64_u32 v[120:121], s[14:15], s28, v120, 0
	v_add3_u32 v104, v121, v134, v104
	v_sub_u32_e32 v121, v131, v104
	v_sub_co_u32_e32 v120, vcc, v138, v120
	s_nop 1
	v_subb_co_u32_e64 v121, s[14:15], v121, v81, vcc
	v_subrev_co_u32_e64 v134, s[14:15], s28, v120
	v_subb_co_u32_e32 v104, vcc, v131, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v135, s[26:27], 0, v121, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v135
	v_subb_co_u32_e64 v121, s[14:15], v121, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v136, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v134
	v_cmp_le_u32_e32 vcc, s29, v104
	s_nop 0
	v_cndmask_b32_e64 v137, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v135
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v120
	v_cndmask_b32_e64 v136, v136, v137, s[26:27]
	v_subrev_co_u32_e64 v137, s[14:15], s28, v134
	s_nop 1
	v_subbrev_co_u32_e64 v121, s[14:15], 0, v121, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v136
	s_nop 1
	v_cndmask_b32_e64 v121, v135, v121, s[14:15]
	v_cndmask_b32_e64 v135, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v131, v131, v135, vcc
	v_cmp_ne_u32_e32 vcc, 0, v131
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v121, vcc
	v_cndmask_b32_e64 v121, v134, v137, s[14:15]
	v_cndmask_b32_e32 v120, v120, v121, vcc
	v_xor_b32_e32 v120, v120, v130
	v_xor_b32_e32 v104, v104, v130
	v_sub_co_u32_e32 v120, vcc, v120, v130
	s_mov_b64 s[14:15], 0x404
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, v104, v130, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_mul_lo_u32 v121, v133, s33
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xfbfb
	v_add3_u32 v120, v104, v120, v121
	v_sub_co_u32_e32 v104, vcc, s14, v126
	s_nop 1
	v_subb_co_u32_e32 v121, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -4, v[114:115]
	s_nop 1
	v_cndmask_b32_e32 v135, v131, v121, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v130, v104, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v133, v134, v136
	v_xor_b32_e32 v121, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v133, s35, 0
	v_mul_hi_u32 v104, v133, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v121, s34, 0
	v_add_co_u32_e32 v104, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v121, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v134, vcc, v135, v141, vcc
	v_mov_b32_e32 v135, s57
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[138:139]
	v_mul_lo_u32 v104, s29, v134
	v_mul_lo_u32 v135, s28, v135
	v_mad_u64_u32 v[138:139], s[14:15], s28, v134, 0
	v_add3_u32 v104, v139, v135, v104
	v_sub_co_u32_e32 v133, vcc, v133, v138
	s_nop 1
	v_subb_co_u32_e64 v135, s[14:15], v121, v104, vcc
	v_sub_u32_e32 v104, v121, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v121, vcc, s28, v133
	v_cmp_le_u32_e64 s[14:15], s29, v135
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cmp_le_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v137, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v133
	v_cndmask_b32_e64 v133, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v121
	v_cndmask_b32_e64 v138, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v135
	v_cndmask_b32_e64 v121, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v135, v137, v138, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v104, v133, v121, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v121, s70, v136
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v134, v104
	v_cmp_ne_u32_e32 vcc, 0, v135
	s_nop 1
	v_cndmask_b32_e32 v104, v134, v104, vcc
	v_ashrrev_i32_e32 v134, 31, v131
	v_mov_b32_e32 v135, v134
	v_xor_b32_e32 v104, v104, v121
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[134:135]
	v_sub_u32_e32 v104, v104, v121
	v_xor_b32_e32 v135, v130, v134
	v_xad_u32 v121, v104, v134, v46
	v_xor_b32_e32 v133, v131, v134
	v_mad_u64_u32 v[130:131], s[14:15], v135, s35, 0
	v_mul_hi_u32 v104, v135, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[138:139], s[14:15], v133, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v138
	v_mad_u64_u32 v[136:137], s[14:15], v133, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v130, vcc, v131, v139, vcc
	v_mov_b32_e32 v131, s57
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[136:137]
	v_mul_lo_u32 v104, s29, v130
	v_mul_lo_u32 v136, s28, v131
	v_mad_u64_u32 v[130:131], s[14:15], s28, v130, 0
	v_add3_u32 v104, v131, v136, v104
	v_sub_u32_e32 v131, v133, v104
	v_sub_co_u32_e32 v130, vcc, v135, v130
	v_mul_lo_u32 v121, v121, s33
	s_nop 0
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v135, s[14:15], s28, v130
	v_subb_co_u32_e32 v104, vcc, v133, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v136, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v136
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v137, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v135
	v_cmp_le_u32_e32 vcc, s29, v104
	s_nop 0
	v_cndmask_b32_e64 v138, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v136
	v_cndmask_b32_e64 v133, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v130
	v_cndmask_b32_e64 v137, v137, v138, s[26:27]
	v_subrev_co_u32_e64 v138, s[14:15], s28, v135
	s_nop 1
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v137
	s_nop 1
	v_cndmask_b32_e64 v131, v136, v131, s[14:15]
	v_cndmask_b32_e64 v136, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v133, v133, v136, vcc
	v_cmp_ne_u32_e32 vcc, 0, v133
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v135, v138, s[14:15]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v134
	v_xor_b32_e32 v104, v104, v134
	v_sub_co_u32_e32 v130, vcc, v130, v134
	s_mov_b64 s[14:15], 0x405
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, v104, v134, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v121, v104, v130, v121
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xfbfa
	v_sub_co_u32_e32 v104, vcc, s14, v126
	s_nop 1
	v_subb_co_u32_e32 v133, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -5, v[114:115]
	s_nop 1
	v_cndmask_b32_e32 v135, v131, v133, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v130, v104, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v137, v134, v136
	v_xor_b32_e32 v133, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v137, s35, 0
	v_mul_hi_u32 v104, v137, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v133, s34, 0
	v_add_co_u32_e32 v104, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v133, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v134, vcc, v135, v141, vcc
	v_mov_b32_e32 v135, s57
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[138:139]
	v_mul_lo_u32 v104, s29, v134
	v_mul_lo_u32 v135, s28, v135
	v_mad_u64_u32 v[138:139], s[14:15], s28, v134, 0
	v_add3_u32 v104, v139, v135, v104
	v_sub_co_u32_e32 v135, vcc, v137, v138
	s_nop 1
	v_subb_co_u32_e64 v137, s[14:15], v133, v104, vcc
	v_sub_u32_e32 v104, v133, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v133, vcc, s28, v135
	v_cmp_le_u32_e64 s[14:15], s29, v137
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cmp_le_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v138, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v135
	v_cndmask_b32_e64 v135, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v133
	v_cndmask_b32_e64 v139, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v137
	v_cndmask_b32_e64 v133, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v137, v138, v139, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v104, v135, v133, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v133, s70, v136
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v134, v104
	v_cmp_ne_u32_e32 vcc, 0, v137
	s_nop 1
	v_cndmask_b32_e32 v104, v134, v104, vcc
	v_ashrrev_i32_e32 v134, 31, v131
	v_mov_b32_e32 v135, v134
	v_xor_b32_e32 v104, v104, v133
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[134:135]
	v_sub_u32_e32 v104, v104, v133
	v_xor_b32_e32 v140, v130, v134
	v_xad_u32 v133, v104, v134, v46
	v_xor_b32_e32 v135, v131, v134
	v_mad_u64_u32 v[130:131], s[14:15], v140, s35, 0
	v_mul_hi_u32 v104, v140, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[138:139], s[14:15], v135, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v138
	v_mad_u64_u32 v[136:137], s[14:15], v135, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v130, vcc, v131, v139, vcc
	v_mov_b32_e32 v131, s57
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[136:137]
	v_mul_lo_u32 v104, s29, v130
	v_mul_lo_u32 v136, s28, v131
	v_mad_u64_u32 v[130:131], s[14:15], s28, v130, 0
	v_add3_u32 v104, v131, v136, v104
	v_sub_u32_e32 v131, v135, v104
	v_sub_co_u32_e32 v130, vcc, v140, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v136, s[14:15], s28, v130
	v_subb_co_u32_e32 v104, vcc, v135, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v137, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v137
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v138, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v136
	v_cmp_le_u32_e32 vcc, s29, v104
	s_nop 0
	v_cndmask_b32_e64 v139, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v137
	v_cndmask_b32_e64 v135, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v130
	v_cndmask_b32_e64 v138, v138, v139, s[26:27]
	v_subrev_co_u32_e64 v139, s[14:15], s28, v136
	s_nop 1
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v138
	s_nop 1
	v_cndmask_b32_e64 v131, v137, v131, s[14:15]
	v_cndmask_b32_e64 v137, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v135, v135, v137, vcc
	v_cmp_ne_u32_e32 vcc, 0, v135
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v136, v139, s[14:15]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v134
	v_xor_b32_e32 v104, v104, v134
	v_sub_co_u32_e32 v130, vcc, v130, v134
	v_mul_lo_u32 v131, v133, s33
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, v104, v134, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	s_mov_b64 s[14:15], 0x406
	v_add3_u32 v142, v104, v130, v131
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xfbf9
	v_sub_co_u32_e32 v104, vcc, s14, v126
	s_nop 1
	v_subb_co_u32_e32 v133, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -6, v[114:115]
	s_nop 1
	v_cndmask_b32_e32 v135, v131, v133, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v130, v104, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v137, v134, v136
	v_xor_b32_e32 v133, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v137, s35, 0
	v_mul_hi_u32 v104, v137, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v133, s34, 0
	v_add_co_u32_e32 v104, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v133, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v134, vcc, v135, v141, vcc
	v_mov_b32_e32 v135, s57
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[138:139]
	v_mul_lo_u32 v104, s29, v134
	v_mul_lo_u32 v135, s28, v135
	v_mad_u64_u32 v[138:139], s[14:15], s28, v134, 0
	v_add3_u32 v104, v139, v135, v104
	v_sub_co_u32_e32 v135, vcc, v137, v138
	s_nop 1
	v_subb_co_u32_e64 v137, s[14:15], v133, v104, vcc
	v_sub_u32_e32 v104, v133, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v133, vcc, s28, v135
	v_cmp_le_u32_e64 s[14:15], s29, v137
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cmp_le_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v138, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v135
	v_cndmask_b32_e64 v135, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v133
	v_cndmask_b32_e64 v139, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v137
	v_cndmask_b32_e64 v133, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v137, v138, v139, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v104, v135, v133, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v133, s70, v136
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v134, v104
	v_cmp_ne_u32_e32 vcc, 0, v137
	s_nop 1
	v_cndmask_b32_e32 v104, v134, v104, vcc
	v_ashrrev_i32_e32 v134, 31, v131
	v_mov_b32_e32 v135, v134
	v_xor_b32_e32 v104, v104, v133
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[134:135]
	v_sub_u32_e32 v104, v104, v133
	v_xor_b32_e32 v140, v130, v134
	v_xad_u32 v133, v104, v134, v46
	v_xor_b32_e32 v135, v131, v134
	v_mad_u64_u32 v[130:131], s[14:15], v140, s35, 0
	v_mul_hi_u32 v104, v140, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[138:139], s[14:15], v135, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v138
	v_mad_u64_u32 v[136:137], s[14:15], v135, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v130, vcc, v131, v139, vcc
	v_mov_b32_e32 v131, s57
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[136:137]
	v_mul_lo_u32 v104, s29, v130
	v_mul_lo_u32 v136, s28, v131
	v_mad_u64_u32 v[130:131], s[14:15], s28, v130, 0
	v_add3_u32 v104, v131, v136, v104
	v_sub_u32_e32 v131, v135, v104
	v_sub_co_u32_e32 v130, vcc, v140, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v136, s[14:15], s28, v130
	v_subb_co_u32_e32 v104, vcc, v135, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v137, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v137
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v138, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v136
	v_cmp_le_u32_e32 vcc, s29, v104
	s_nop 0
	v_cndmask_b32_e64 v139, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v137
	v_cndmask_b32_e64 v135, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v130
	v_cndmask_b32_e64 v138, v138, v139, s[26:27]
	v_subrev_co_u32_e64 v139, s[14:15], s28, v136
	s_nop 1
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v138
	s_nop 1
	v_cndmask_b32_e64 v131, v137, v131, s[14:15]
	v_cndmask_b32_e64 v137, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v135, v135, v137, vcc
	v_cmp_ne_u32_e32 vcc, 0, v135
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v136, v139, s[14:15]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v134
	v_xor_b32_e32 v104, v104, v134
	v_sub_co_u32_e32 v130, vcc, v130, v134
	v_mul_lo_u32 v131, v133, s33
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, v104, v134, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	s_mov_b64 s[14:15], 0x407
	v_add3_u32 v143, v104, v130, v131
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xfbf8
	v_sub_co_u32_e32 v104, vcc, s14, v126
	s_nop 1
	v_subb_co_u32_e32 v133, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -7, v[114:115]
	s_nop 1
	v_cndmask_b32_e32 v135, v131, v133, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v130, v104, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v137, v134, v136
	v_xor_b32_e32 v133, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v137, s35, 0
	v_mul_hi_u32 v104, v137, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v133, s34, 0
	v_add_co_u32_e32 v104, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v133, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v134, vcc, v135, v141, vcc
	v_mov_b32_e32 v135, s57
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[138:139]
	v_mul_lo_u32 v104, s29, v134
	v_mul_lo_u32 v135, s28, v135
	v_mad_u64_u32 v[138:139], s[14:15], s28, v134, 0
	v_add3_u32 v104, v139, v135, v104
	v_sub_co_u32_e32 v135, vcc, v137, v138
	s_nop 1
	v_subb_co_u32_e64 v137, s[14:15], v133, v104, vcc
	v_sub_u32_e32 v104, v133, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v133, vcc, s28, v135
	v_cmp_le_u32_e64 s[14:15], s29, v137
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cmp_le_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v138, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v135
	v_cndmask_b32_e64 v135, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v133
	v_cndmask_b32_e64 v139, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v137
	v_cndmask_b32_e64 v133, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v137, v138, v139, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v104, v135, v133, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v133, s70, v136
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v134, v104
	v_cmp_ne_u32_e32 vcc, 0, v137
	s_nop 1
	v_cndmask_b32_e32 v104, v134, v104, vcc
	v_ashrrev_i32_e32 v134, 31, v131
	v_mov_b32_e32 v135, v134
	v_xor_b32_e32 v104, v104, v133
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[134:135]
	v_sub_u32_e32 v104, v104, v133
	v_xor_b32_e32 v140, v130, v134
	v_xad_u32 v133, v104, v134, v46
	v_xor_b32_e32 v135, v131, v134
	v_mad_u64_u32 v[130:131], s[14:15], v140, s35, 0
	v_mul_hi_u32 v104, v140, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[138:139], s[14:15], v135, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v138
	v_mad_u64_u32 v[136:137], s[14:15], v135, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v130, vcc, v131, v139, vcc
	v_mov_b32_e32 v131, s57
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[136:137]
	v_mul_lo_u32 v104, s29, v130
	v_mul_lo_u32 v136, s28, v131
	v_mad_u64_u32 v[130:131], s[14:15], s28, v130, 0
	v_add3_u32 v104, v131, v136, v104
	v_sub_u32_e32 v131, v135, v104
	v_sub_co_u32_e32 v130, vcc, v140, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v136, s[14:15], s28, v130
	v_subb_co_u32_e32 v104, vcc, v135, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v137, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v137
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v138, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v136
	v_cmp_le_u32_e32 vcc, s29, v104
	s_nop 0
	v_cndmask_b32_e64 v139, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v137
	v_cndmask_b32_e64 v135, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v130
	v_cndmask_b32_e64 v138, v138, v139, s[26:27]
	v_subrev_co_u32_e64 v139, s[14:15], s28, v136
	s_nop 1
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v138
	s_nop 1
	v_cndmask_b32_e64 v131, v137, v131, s[14:15]
	v_cndmask_b32_e64 v137, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v135, v135, v137, vcc
	v_cmp_ne_u32_e32 vcc, 0, v135
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v136, v139, s[14:15]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v134
	v_xor_b32_e32 v104, v104, v134
	v_sub_co_u32_e32 v130, vcc, v130, v134
	v_mul_lo_u32 v131, v133, s33
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, v104, v134, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	s_mov_b64 s[14:15], 0x408
	v_add3_u32 v104, v104, v130, v131
	buffer_load_ubyte v133, v116, s[36:39], 0 offen
	buffer_load_ubyte v135, v118, s[36:39], 0 offen
	buffer_load_ubyte v134, v119, s[36:39], 0 offen
	buffer_load_ubyte v136, v120, s[36:39], 0 offen
	buffer_load_ubyte v137, v121, s[36:39], 0 offen
	buffer_load_ubyte v139, v142, s[36:39], 0 offen
	buffer_load_ubyte v138, v143, s[36:39], 0 offen
	buffer_load_ubyte v140, v104, s[36:39], 0 offen
	v_lshl_add_u64 v[118:119], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xfbf7
	v_sub_co_u32_e32 v104, vcc, s14, v126
	s_nop 1
	v_subb_co_u32_e32 v116, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -8, v[114:115]
	s_nop 1
	v_cndmask_b32_e32 v121, v119, v116, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_cndmask_b32_e32 v120, v118, v104, vcc
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v131, v120, v130
	v_xor_b32_e32 v116, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v131, s35, 0
	v_mul_hi_u32 v104, v131, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[144:145], s[14:15], v116, s34, 0
	v_add_co_u32_e32 v104, vcc, v120, v144
	v_mad_u64_u32 v[142:143], s[14:15], v116, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v145, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v143, vcc, 0, v143, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[142:143]
	v_mul_lo_u32 v104, s29, v120
	v_mul_lo_u32 v121, s28, v121
	v_mad_u64_u32 v[142:143], s[14:15], s28, v120, 0
	v_add3_u32 v104, v143, v121, v104
	v_sub_co_u32_e32 v121, vcc, v131, v142
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v116, v104, vcc
	v_sub_u32_e32 v104, v116, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v116, vcc, s28, v121
	v_cmp_le_u32_e64 s[14:15], s29, v131
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cmp_le_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v141, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v121
	v_cndmask_b32_e64 v121, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v116
	v_cndmask_b32_e64 v142, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v131
	v_cndmask_b32_e64 v116, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v131, v141, v142, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v104, v121, v116, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v116, s70, v130
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v120, v104
	v_cmp_ne_u32_e32 vcc, 0, v131
	s_nop 1
	v_cndmask_b32_e32 v104, v120, v104, vcc
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_xor_b32_e32 v104, v104, v116
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_sub_u32_e32 v104, v104, v116
	v_xor_b32_e32 v141, v118, v120
	v_xad_u32 v116, v104, v120, v46
	v_xor_b32_e32 v121, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v141, s35, 0
	v_mul_hi_u32 v104, v141, s34
	v_lshl_add_u64 v[118:119], v[104:105], 0, v[118:119]
	v_mad_u64_u32 v[142:143], s[14:15], v121, s34, 0
	v_add_co_u32_e32 v104, vcc, v118, v142
	v_mad_u64_u32 v[130:131], s[14:15], v121, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v118, vcc, v119, v143, vcc
	v_mov_b32_e32 v119, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[130:131]
	v_mul_lo_u32 v104, s29, v118
	v_mul_lo_u32 v130, s28, v119
	v_mad_u64_u32 v[118:119], s[14:15], s28, v118, 0
	v_add3_u32 v104, v119, v130, v104
	v_sub_u32_e32 v119, v121, v104
	v_sub_co_u32_e32 v118, vcc, v141, v118
	v_mul_lo_u32 v116, v116, s33
	s_nop 0
	v_subb_co_u32_e64 v119, s[14:15], v119, v81, vcc
	v_subrev_co_u32_e64 v130, s[14:15], s28, v118
	v_subb_co_u32_e32 v104, vcc, v121, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[26:27], 0, v119, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v131
	v_subb_co_u32_e64 v119, s[14:15], v119, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v141, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v130
	v_cmp_le_u32_e32 vcc, s29, v104
	s_nop 0
	v_cndmask_b32_e64 v142, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v131
	v_cndmask_b32_e64 v121, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v118
	v_cndmask_b32_e64 v141, v141, v142, s[26:27]
	v_subrev_co_u32_e64 v142, s[14:15], s28, v130
	s_nop 1
	v_subbrev_co_u32_e64 v119, s[14:15], 0, v119, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v141
	s_nop 1
	v_cndmask_b32_e64 v119, v131, v119, s[14:15]
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v121, v121, v131, vcc
	v_cmp_ne_u32_e32 vcc, 0, v121
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v119, vcc
	v_cndmask_b32_e64 v119, v130, v142, s[14:15]
	v_cndmask_b32_e32 v118, v118, v119, vcc
	v_xor_b32_e32 v118, v118, v120
	v_xor_b32_e32 v104, v104, v120
	v_sub_co_u32_e32 v118, vcc, v118, v120
	s_mov_b64 s[14:15], 0x409
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, v104, v120, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v116, v104, v118, v116
	v_lshl_add_u64 v[118:119], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xfbf6
	v_sub_co_u32_e32 v104, vcc, s14, v126
	s_nop 1
	v_subb_co_u32_e32 v120, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -9, v[114:115]
	s_nop 1
	v_cndmask_b32_e32 v121, v119, v120, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_cndmask_b32_e32 v120, v118, v104, vcc
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v141, v120, v130
	v_xor_b32_e32 v131, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v141, s35, 0
	v_mul_hi_u32 v104, v141, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[144:145], s[14:15], v131, s34, 0
	v_add_co_u32_e32 v104, vcc, v120, v144
	v_mad_u64_u32 v[142:143], s[14:15], v131, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v145, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v143, vcc, 0, v143, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[142:143]
	v_mul_lo_u32 v104, s29, v120
	v_mul_lo_u32 v121, s28, v121
	v_mad_u64_u32 v[142:143], s[14:15], s28, v120, 0
	v_add3_u32 v104, v143, v121, v104
	v_sub_co_u32_e32 v121, vcc, v141, v142
	s_nop 1
	v_subb_co_u32_e64 v141, s[14:15], v131, v104, vcc
	v_sub_u32_e32 v104, v131, v104
	v_cmp_le_u32_e64 s[14:15], s29, v141
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	s_nop 0
	v_cndmask_b32_e64 v142, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v121
	v_subrev_co_u32_e32 v121, vcc, s28, v121
	s_nop 0
	v_cndmask_b32_e64 v143, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cmp_le_u32_e32 vcc, s29, v104
	v_cmp_eq_u32_e64 s[14:15], s29, v141
	s_nop 0
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v121
	v_cndmask_b32_e64 v141, v142, v143, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v121, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v104, v131, v121, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	s_nop 1
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v120, v104
	v_cmp_ne_u32_e32 vcc, 0, v141
	s_nop 1
	v_cndmask_b32_e32 v104, v120, v104, vcc
	v_xor_b32_e32 v120, s70, v130
	v_xor_b32_e32 v104, v104, v120
	v_sub_u32_e32 v104, v104, v120
	v_ashrrev_i32_e32 v120, 31, v119
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[120:121]
	v_xor_b32_e32 v144, v118, v120
	v_xad_u32 v141, v104, v120, v46
	v_xor_b32_e32 v121, v119, v120
	v_mad_u64_u32 v[118:119], s[14:15], v144, s35, 0
	v_mul_hi_u32 v104, v144, s34
	v_lshl_add_u64 v[118:119], v[104:105], 0, v[118:119]
	v_mad_u64_u32 v[142:143], s[14:15], v121, s34, 0
	v_add_co_u32_e32 v104, vcc, v118, v142
	v_mad_u64_u32 v[130:131], s[14:15], v121, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v118, vcc, v119, v143, vcc
	v_mov_b32_e32 v119, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[118:119], v[118:119], 0, v[130:131]
	v_mul_lo_u32 v104, s29, v118
	v_mul_lo_u32 v130, s28, v119
	v_mad_u64_u32 v[118:119], s[14:15], s28, v118, 0
	v_add3_u32 v104, v119, v130, v104
	v_sub_u32_e32 v119, v121, v104
	v_sub_co_u32_e32 v118, vcc, v144, v118
	s_nop 1
	v_subb_co_u32_e64 v119, s[14:15], v119, v81, vcc
	v_subrev_co_u32_e64 v130, s[14:15], s28, v118
	v_subb_co_u32_e32 v104, vcc, v121, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[26:27], 0, v119, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v131
	v_subb_co_u32_e64 v119, s[14:15], v119, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v142, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v130
	v_cmp_le_u32_e32 vcc, s29, v104
	s_nop 0
	v_cndmask_b32_e64 v143, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v131
	v_cndmask_b32_e64 v121, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v118
	v_cndmask_b32_e64 v142, v142, v143, s[26:27]
	v_subrev_co_u32_e64 v143, s[14:15], s28, v130
	s_nop 1
	v_subbrev_co_u32_e64 v119, s[14:15], 0, v119, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v142
	s_nop 1
	v_cndmask_b32_e64 v119, v131, v119, s[14:15]
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v121, v121, v131, vcc
	v_cmp_ne_u32_e32 vcc, 0, v121
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v119, vcc
	v_cndmask_b32_e64 v119, v130, v143, s[14:15]
	v_cndmask_b32_e32 v118, v118, v119, vcc
	v_xor_b32_e32 v118, v118, v120
	v_xor_b32_e32 v104, v104, v120
	v_sub_co_u32_e32 v118, vcc, v118, v120
	s_mov_b64 s[14:15], 0x40a
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, v104, v120, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_mul_lo_u32 v119, v141, s33
	v_lshl_add_u64 v[120:121], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xfbf5
	v_add3_u32 v118, v104, v118, v119
	v_sub_co_u32_e32 v104, vcc, s14, v126
	s_nop 1
	v_subb_co_u32_e32 v119, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -10, v[114:115]
	s_nop 1
	v_cndmask_b32_e32 v131, v121, v119, vcc
	v_ashrrev_i32_e32 v142, 31, v131
	v_cndmask_b32_e32 v130, v120, v104, vcc
	v_mov_b32_e32 v143, v142
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[142:143]
	v_xor_b32_e32 v141, v130, v142
	v_xor_b32_e32 v119, v131, v142
	v_mad_u64_u32 v[130:131], s[14:15], v141, s35, 0
	v_mul_hi_u32 v104, v141, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[146:147], s[14:15], v119, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v146
	v_mad_u64_u32 v[144:145], s[14:15], v119, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v130, vcc, v131, v147, vcc
	v_mov_b32_e32 v131, s57
	s_nop 0
	v_addc_co_u32_e32 v145, vcc, 0, v145, vcc
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[144:145]
	v_mul_lo_u32 v104, s29, v130
	v_mul_lo_u32 v131, s28, v131
	v_mad_u64_u32 v[144:145], s[14:15], s28, v130, 0
	v_add3_u32 v104, v145, v131, v104
	v_sub_co_u32_e32 v131, vcc, v141, v144
	s_nop 1
	v_subb_co_u32_e64 v141, s[14:15], v119, v104, vcc
	v_sub_u32_e32 v104, v119, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v119, vcc, s28, v131
	v_cmp_le_u32_e64 s[14:15], s29, v141
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cmp_le_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v143, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v131
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v119
	v_cndmask_b32_e64 v144, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v141
	v_cndmask_b32_e64 v119, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v141, v143, v144, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v104, v131, v119, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v119, s70, v142
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v130, v104
	v_cmp_ne_u32_e32 vcc, 0, v141
	s_nop 1
	v_cndmask_b32_e32 v104, v130, v104, vcc
	v_ashrrev_i32_e32 v130, 31, v121
	v_mov_b32_e32 v131, v130
	v_xor_b32_e32 v104, v104, v119
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_sub_u32_e32 v104, v104, v119
	v_xor_b32_e32 v141, v120, v130
	v_xad_u32 v119, v104, v130, v46
	v_xor_b32_e32 v131, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v141, s35, 0
	v_mul_hi_u32 v104, v141, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[144:145], s[14:15], v131, s34, 0
	v_add_co_u32_e32 v104, vcc, v120, v144
	v_mad_u64_u32 v[142:143], s[14:15], v131, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v145, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v143, vcc, 0, v143, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[142:143]
	v_mul_lo_u32 v104, s29, v120
	v_mul_lo_u32 v142, s28, v121
	v_mad_u64_u32 v[120:121], s[14:15], s28, v120, 0
	v_add3_u32 v104, v121, v142, v104
	v_sub_u32_e32 v121, v131, v104
	v_sub_co_u32_e32 v120, vcc, v141, v120
	v_mul_lo_u32 v119, v119, s33
	s_nop 0
	v_subb_co_u32_e64 v121, s[14:15], v121, v81, vcc
	v_subrev_co_u32_e64 v141, s[14:15], s28, v120
	v_subb_co_u32_e32 v104, vcc, v131, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v142, s[26:27], 0, v121, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v142
	v_subb_co_u32_e64 v121, s[14:15], v121, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v143, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v141
	v_cmp_le_u32_e32 vcc, s29, v104
	s_nop 0
	v_cndmask_b32_e64 v144, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v142
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v120
	v_cndmask_b32_e64 v143, v143, v144, s[26:27]
	v_subrev_co_u32_e64 v144, s[14:15], s28, v141
	s_nop 1
	v_subbrev_co_u32_e64 v121, s[14:15], 0, v121, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v143
	s_nop 1
	v_cndmask_b32_e64 v121, v142, v121, s[14:15]
	v_cndmask_b32_e64 v142, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v131, v131, v142, vcc
	v_cmp_ne_u32_e32 vcc, 0, v131
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v121, vcc
	v_cndmask_b32_e64 v121, v141, v144, s[14:15]
	v_cndmask_b32_e32 v120, v120, v121, vcc
	v_xor_b32_e32 v120, v120, v130
	v_xor_b32_e32 v104, v104, v130
	v_sub_co_u32_e32 v120, vcc, v120, v130
	s_mov_b64 s[14:15], 0x40b
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, v104, v130, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v119, v104, v120, v119
	v_lshl_add_u64 v[120:121], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xfbf4
	v_sub_co_u32_e32 v104, vcc, s14, v126
	s_nop 1
	v_subb_co_u32_e32 v130, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -11, v[114:115]
	s_nop 1
	v_cndmask_b32_e32 v131, v121, v130, vcc
	v_ashrrev_i32_e32 v142, 31, v131
	v_cndmask_b32_e32 v130, v120, v104, vcc
	v_mov_b32_e32 v143, v142
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[142:143]
	v_xor_b32_e32 v143, v130, v142
	v_xor_b32_e32 v141, v131, v142
	v_mad_u64_u32 v[130:131], s[14:15], v143, s35, 0
	v_mul_hi_u32 v104, v143, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[146:147], s[14:15], v141, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v146
	v_mad_u64_u32 v[144:145], s[14:15], v141, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v130, vcc, v131, v147, vcc
	v_mov_b32_e32 v131, s57
	s_nop 0
	v_addc_co_u32_e32 v145, vcc, 0, v145, vcc
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[144:145]
	v_mul_lo_u32 v104, s29, v130
	v_mul_lo_u32 v131, s28, v131
	v_mad_u64_u32 v[144:145], s[14:15], s28, v130, 0
	v_add3_u32 v104, v145, v131, v104
	v_sub_co_u32_e32 v131, vcc, v143, v144
	s_nop 1
	v_subb_co_u32_e64 v143, s[14:15], v141, v104, vcc
	v_sub_u32_e32 v104, v141, v104
	v_cmp_le_u32_e64 s[14:15], s29, v143
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	s_nop 0
	v_cndmask_b32_e64 v144, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v131
	v_subrev_co_u32_e32 v131, vcc, s28, v131
	s_nop 0
	v_cndmask_b32_e64 v145, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cmp_le_u32_e32 vcc, s29, v104
	v_cmp_eq_u32_e64 s[14:15], s29, v143
	s_nop 0
	v_cndmask_b32_e64 v141, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v131
	v_cndmask_b32_e64 v143, v144, v145, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v104, v141, v131, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	s_nop 1
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v130, v104
	v_cmp_ne_u32_e32 vcc, 0, v143
	s_nop 1
	v_cndmask_b32_e32 v104, v130, v104, vcc
	v_xor_b32_e32 v130, s70, v142
	v_xor_b32_e32 v104, v104, v130
	v_sub_u32_e32 v104, v104, v130
	v_ashrrev_i32_e32 v130, 31, v121
	v_mov_b32_e32 v131, v130
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[130:131]
	v_xor_b32_e32 v146, v120, v130
	v_xad_u32 v141, v104, v130, v46
	v_xor_b32_e32 v131, v121, v130
	v_mad_u64_u32 v[120:121], s[14:15], v146, s35, 0
	v_mul_hi_u32 v104, v146, s34
	v_lshl_add_u64 v[120:121], v[104:105], 0, v[120:121]
	v_mad_u64_u32 v[144:145], s[14:15], v131, s34, 0
	v_add_co_u32_e32 v104, vcc, v120, v144
	v_mad_u64_u32 v[142:143], s[14:15], v131, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v120, vcc, v121, v145, vcc
	v_mov_b32_e32 v121, s57
	s_nop 0
	v_addc_co_u32_e32 v143, vcc, 0, v143, vcc
	v_lshl_add_u64 v[120:121], v[120:121], 0, v[142:143]
	v_mul_lo_u32 v104, s29, v120
	v_mul_lo_u32 v142, s28, v121
	v_mad_u64_u32 v[120:121], s[14:15], s28, v120, 0
	v_add3_u32 v104, v121, v142, v104
	v_sub_u32_e32 v121, v131, v104
	v_sub_co_u32_e32 v120, vcc, v146, v120
	s_nop 1
	v_subb_co_u32_e64 v121, s[14:15], v121, v81, vcc
	v_subrev_co_u32_e64 v142, s[14:15], s28, v120
	v_subb_co_u32_e32 v104, vcc, v131, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v143, s[26:27], 0, v121, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v143
	v_subb_co_u32_e64 v121, s[14:15], v121, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v144, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v142
	v_cmp_le_u32_e32 vcc, s29, v104
	s_nop 0
	v_cndmask_b32_e64 v145, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v143
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v120
	v_cndmask_b32_e64 v144, v144, v145, s[26:27]
	v_subrev_co_u32_e64 v145, s[14:15], s28, v142
	s_nop 1
	v_subbrev_co_u32_e64 v121, s[14:15], 0, v121, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v144
	s_nop 1
	v_cndmask_b32_e64 v121, v143, v121, s[14:15]
	v_cndmask_b32_e64 v143, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v131, v131, v143, vcc
	v_cmp_ne_u32_e32 vcc, 0, v131
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v121, vcc
	v_cndmask_b32_e64 v121, v142, v145, s[14:15]
	v_cndmask_b32_e32 v120, v120, v121, vcc
	v_xor_b32_e32 v120, v120, v130
	v_xor_b32_e32 v104, v104, v130
	v_sub_co_u32_e32 v120, vcc, v120, v130
	s_mov_b64 s[14:15], 0x40c
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, v104, v130, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_mul_lo_u32 v121, v141, s33
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xfbf3
	v_add3_u32 v120, v104, v120, v121
	v_sub_co_u32_e32 v104, vcc, s14, v126
	s_nop 1
	v_subb_co_u32_e32 v121, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -12, v[114:115]
	s_nop 1
	v_cndmask_b32_e32 v143, v131, v121, vcc
	v_ashrrev_i32_e32 v144, 31, v143
	v_cndmask_b32_e32 v142, v130, v104, vcc
	v_mov_b32_e32 v145, v144
	v_lshl_add_u64 v[142:143], v[142:143], 0, v[144:145]
	v_xor_b32_e32 v141, v142, v144
	v_xor_b32_e32 v121, v143, v144
	v_mad_u64_u32 v[142:143], s[14:15], v141, s35, 0
	v_mul_hi_u32 v104, v141, s34
	v_lshl_add_u64 v[142:143], v[104:105], 0, v[142:143]
	v_mad_u64_u32 v[148:149], s[14:15], v121, s34, 0
	v_add_co_u32_e32 v104, vcc, v142, v148
	v_mad_u64_u32 v[146:147], s[14:15], v121, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v142, vcc, v143, v149, vcc
	v_mov_b32_e32 v143, s57
	s_nop 0
	v_addc_co_u32_e32 v147, vcc, 0, v147, vcc
	v_lshl_add_u64 v[142:143], v[142:143], 0, v[146:147]
	v_mul_lo_u32 v104, s29, v142
	v_mul_lo_u32 v143, s28, v143
	v_mad_u64_u32 v[146:147], s[14:15], s28, v142, 0
	v_add3_u32 v104, v147, v143, v104
	v_sub_co_u32_e32 v141, vcc, v141, v146
	s_nop 1
	v_subb_co_u32_e64 v143, s[14:15], v121, v104, vcc
	v_sub_u32_e32 v104, v121, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v121, vcc, s28, v141
	v_cmp_le_u32_e64 s[14:15], s29, v143
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cmp_le_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v145, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v141
	v_cndmask_b32_e64 v141, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v121
	v_cndmask_b32_e64 v146, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v143
	v_cndmask_b32_e64 v121, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v143, v145, v146, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v104, v141, v121, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v121, s70, v144
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v142, v104
	v_cmp_ne_u32_e32 vcc, 0, v143
	s_nop 1
	v_cndmask_b32_e32 v104, v142, v104, vcc
	v_ashrrev_i32_e32 v142, 31, v131
	v_mov_b32_e32 v143, v142
	v_xor_b32_e32 v104, v104, v121
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[142:143]
	v_sub_u32_e32 v104, v104, v121
	v_xor_b32_e32 v143, v130, v142
	v_xad_u32 v121, v104, v142, v46
	v_xor_b32_e32 v141, v131, v142
	v_mad_u64_u32 v[130:131], s[14:15], v143, s35, 0
	v_mul_hi_u32 v104, v143, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[146:147], s[14:15], v141, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v146
	v_mad_u64_u32 v[144:145], s[14:15], v141, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v130, vcc, v131, v147, vcc
	v_mov_b32_e32 v131, s57
	s_nop 0
	v_addc_co_u32_e32 v145, vcc, 0, v145, vcc
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[144:145]
	v_mul_lo_u32 v104, s29, v130
	v_mul_lo_u32 v144, s28, v131
	v_mad_u64_u32 v[130:131], s[14:15], s28, v130, 0
	v_add3_u32 v104, v131, v144, v104
	v_sub_u32_e32 v131, v141, v104
	v_sub_co_u32_e32 v130, vcc, v143, v130
	v_mul_lo_u32 v121, v121, s33
	s_nop 0
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v143, s[14:15], s28, v130
	v_subb_co_u32_e32 v104, vcc, v141, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v144, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v144
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v145, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v143
	v_cmp_le_u32_e32 vcc, s29, v104
	s_nop 0
	v_cndmask_b32_e64 v146, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v144
	v_cndmask_b32_e64 v141, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v130
	v_cndmask_b32_e64 v145, v145, v146, s[26:27]
	v_subrev_co_u32_e64 v146, s[14:15], s28, v143
	s_nop 1
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v145
	s_nop 1
	v_cndmask_b32_e64 v131, v144, v131, s[14:15]
	v_cndmask_b32_e64 v144, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v141, v141, v144, vcc
	v_cmp_ne_u32_e32 vcc, 0, v141
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v143, v146, s[14:15]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v142
	v_xor_b32_e32 v104, v104, v142
	v_sub_co_u32_e32 v130, vcc, v130, v142
	s_mov_b64 s[14:15], 0x40d
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, v104, v142, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v121, v104, v130, v121
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xfbf2
	v_sub_co_u32_e32 v104, vcc, s14, v126
	s_nop 1
	v_subb_co_u32_e32 v141, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -13, v[114:115]
	s_nop 1
	v_cndmask_b32_e32 v143, v131, v141, vcc
	v_ashrrev_i32_e32 v144, 31, v143
	v_cndmask_b32_e32 v142, v130, v104, vcc
	v_mov_b32_e32 v145, v144
	v_lshl_add_u64 v[142:143], v[142:143], 0, v[144:145]
	v_xor_b32_e32 v145, v142, v144
	v_xor_b32_e32 v141, v143, v144
	v_mad_u64_u32 v[142:143], s[14:15], v145, s35, 0
	v_mul_hi_u32 v104, v145, s34
	v_lshl_add_u64 v[142:143], v[104:105], 0, v[142:143]
	v_mad_u64_u32 v[148:149], s[14:15], v141, s34, 0
	v_add_co_u32_e32 v104, vcc, v142, v148
	v_mad_u64_u32 v[146:147], s[14:15], v141, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v142, vcc, v143, v149, vcc
	v_mov_b32_e32 v143, s57
	s_nop 0
	v_addc_co_u32_e32 v147, vcc, 0, v147, vcc
	v_lshl_add_u64 v[142:143], v[142:143], 0, v[146:147]
	v_mul_lo_u32 v104, s29, v142
	v_mul_lo_u32 v143, s28, v143
	v_mad_u64_u32 v[146:147], s[14:15], s28, v142, 0
	v_add3_u32 v104, v147, v143, v104
	v_sub_co_u32_e32 v143, vcc, v145, v146
	s_nop 1
	v_subb_co_u32_e64 v145, s[14:15], v141, v104, vcc
	v_sub_u32_e32 v104, v141, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v141, vcc, s28, v143
	v_cmp_le_u32_e64 s[14:15], s29, v145
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cmp_le_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v146, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v143
	v_cndmask_b32_e64 v143, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v141
	v_cndmask_b32_e64 v147, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v145
	v_cndmask_b32_e64 v141, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v145, v146, v147, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v104, v143, v141, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v141, s70, v144
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v142, v104
	v_cmp_ne_u32_e32 vcc, 0, v145
	s_nop 1
	v_cndmask_b32_e32 v104, v142, v104, vcc
	v_ashrrev_i32_e32 v142, 31, v131
	v_mov_b32_e32 v143, v142
	v_xor_b32_e32 v104, v104, v141
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[142:143]
	v_sub_u32_e32 v104, v104, v141
	v_xor_b32_e32 v148, v130, v142
	v_xad_u32 v141, v104, v142, v46
	v_xor_b32_e32 v143, v131, v142
	v_mad_u64_u32 v[130:131], s[14:15], v148, s35, 0
	v_mul_hi_u32 v104, v148, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[146:147], s[14:15], v143, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v146
	v_mad_u64_u32 v[144:145], s[14:15], v143, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v130, vcc, v131, v147, vcc
	v_mov_b32_e32 v131, s57
	s_nop 0
	v_addc_co_u32_e32 v145, vcc, 0, v145, vcc
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[144:145]
	v_mul_lo_u32 v104, s29, v130
	v_mul_lo_u32 v144, s28, v131
	v_mad_u64_u32 v[130:131], s[14:15], s28, v130, 0
	v_add3_u32 v104, v131, v144, v104
	v_sub_u32_e32 v131, v143, v104
	v_sub_co_u32_e32 v130, vcc, v148, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v144, s[14:15], s28, v130
	v_subb_co_u32_e32 v104, vcc, v143, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v145, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v145
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v146, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v144
	v_cmp_le_u32_e32 vcc, s29, v104
	s_nop 0
	v_cndmask_b32_e64 v147, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v145
	v_cndmask_b32_e64 v143, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v130
	v_cndmask_b32_e64 v146, v146, v147, s[26:27]
	v_subrev_co_u32_e64 v147, s[14:15], s28, v144
	s_nop 1
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v146
	s_nop 1
	v_cndmask_b32_e64 v131, v145, v131, s[14:15]
	v_cndmask_b32_e64 v145, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v143, v143, v145, vcc
	v_cmp_ne_u32_e32 vcc, 0, v143
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v144, v147, s[14:15]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v142
	v_xor_b32_e32 v104, v104, v142
	v_sub_co_u32_e32 v130, vcc, v130, v142
	v_mul_lo_u32 v131, v141, s33
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, v104, v142, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	s_mov_b64 s[14:15], 0x40e
	v_add3_u32 v162, v104, v130, v131
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xfbf1
	v_sub_co_u32_e32 v104, vcc, s14, v126
	s_nop 1
	v_subb_co_u32_e32 v141, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -14, v[114:115]
	s_nop 1
	v_cndmask_b32_e32 v143, v131, v141, vcc
	v_ashrrev_i32_e32 v144, 31, v143
	v_cndmask_b32_e32 v142, v130, v104, vcc
	v_mov_b32_e32 v145, v144
	v_lshl_add_u64 v[142:143], v[142:143], 0, v[144:145]
	v_xor_b32_e32 v145, v142, v144
	v_xor_b32_e32 v141, v143, v144
	v_mad_u64_u32 v[142:143], s[14:15], v145, s35, 0
	v_mul_hi_u32 v104, v145, s34
	v_lshl_add_u64 v[142:143], v[104:105], 0, v[142:143]
	v_mad_u64_u32 v[148:149], s[14:15], v141, s34, 0
	v_add_co_u32_e32 v104, vcc, v142, v148
	v_mad_u64_u32 v[146:147], s[14:15], v141, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v142, vcc, v143, v149, vcc
	v_mov_b32_e32 v143, s57
	s_nop 0
	v_addc_co_u32_e32 v147, vcc, 0, v147, vcc
	v_lshl_add_u64 v[142:143], v[142:143], 0, v[146:147]
	v_mul_lo_u32 v104, s29, v142
	v_mul_lo_u32 v143, s28, v143
	v_mad_u64_u32 v[146:147], s[14:15], s28, v142, 0
	v_add3_u32 v104, v147, v143, v104
	v_sub_co_u32_e32 v143, vcc, v145, v146
	s_nop 1
	v_subb_co_u32_e64 v145, s[14:15], v141, v104, vcc
	v_sub_u32_e32 v104, v141, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v141, vcc, s28, v143
	v_cmp_le_u32_e64 s[14:15], s29, v145
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cmp_le_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v146, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v143
	v_cndmask_b32_e64 v143, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v141
	v_cndmask_b32_e64 v147, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v145
	v_cndmask_b32_e64 v141, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v145, v146, v147, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v104, v143, v141, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v141, s70, v144
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v142, v104
	v_cmp_ne_u32_e32 vcc, 0, v145
	s_nop 1
	v_cndmask_b32_e32 v104, v142, v104, vcc
	v_ashrrev_i32_e32 v142, 31, v131
	v_mov_b32_e32 v143, v142
	v_xor_b32_e32 v104, v104, v141
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[142:143]
	v_sub_u32_e32 v104, v104, v141
	v_xor_b32_e32 v148, v130, v142
	v_xad_u32 v141, v104, v142, v46
	v_xor_b32_e32 v143, v131, v142
	v_mad_u64_u32 v[130:131], s[14:15], v148, s35, 0
	v_mul_hi_u32 v104, v148, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[146:147], s[14:15], v143, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v146
	v_mad_u64_u32 v[144:145], s[14:15], v143, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v130, vcc, v131, v147, vcc
	v_mov_b32_e32 v131, s57
	s_nop 0
	v_addc_co_u32_e32 v145, vcc, 0, v145, vcc
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[144:145]
	v_mul_lo_u32 v104, s29, v130
	v_mul_lo_u32 v144, s28, v131
	v_mad_u64_u32 v[130:131], s[14:15], s28, v130, 0
	v_add3_u32 v104, v131, v144, v104
	v_sub_u32_e32 v131, v143, v104
	v_sub_co_u32_e32 v130, vcc, v148, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v144, s[14:15], s28, v130
	v_subb_co_u32_e32 v104, vcc, v143, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v145, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v145
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v146, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v144
	v_cmp_le_u32_e32 vcc, s29, v104
	s_nop 0
	v_cndmask_b32_e64 v147, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v145
	v_cndmask_b32_e64 v143, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v130
	v_cndmask_b32_e64 v146, v146, v147, s[26:27]
	v_subrev_co_u32_e64 v147, s[14:15], s28, v144
	s_nop 1
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v146
	s_nop 1
	v_cndmask_b32_e64 v131, v145, v131, s[14:15]
	v_cndmask_b32_e64 v145, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v143, v143, v145, vcc
	v_cmp_ne_u32_e32 vcc, 0, v143
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v144, v147, s[14:15]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v142
	v_xor_b32_e32 v104, v104, v142
	v_sub_co_u32_e32 v130, vcc, v130, v142
	v_mul_lo_u32 v131, v141, s33
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, v104, v142, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	s_mov_b64 s[14:15], 0x40f
	v_add3_u32 v148, v104, v130, v131
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xfbf0
	v_sub_co_u32_e32 v104, vcc, s14, v126
	s_nop 1
	v_subb_co_u32_e32 v141, vcc, -1, v127, vcc
	v_cmp_gt_i64_e32 vcc, -15, v[114:115]
	s_nop 1
	v_cndmask_b32_e32 v115, v131, v141, vcc
	v_ashrrev_i32_e32 v142, 31, v115
	v_cndmask_b32_e32 v114, v130, v104, vcc
	v_mov_b32_e32 v143, v142
	v_lshl_add_u64 v[114:115], v[114:115], 0, v[142:143]
	v_xor_b32_e32 v143, v114, v142
	v_xor_b32_e32 v141, v115, v142
	v_mad_u64_u32 v[114:115], s[14:15], v143, s35, 0
	v_mul_hi_u32 v104, v143, s34
	v_lshl_add_u64 v[114:115], v[104:105], 0, v[114:115]
	v_mad_u64_u32 v[146:147], s[14:15], v141, s34, 0
	v_add_co_u32_e32 v104, vcc, v114, v146
	v_mad_u64_u32 v[144:145], s[14:15], v141, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v114, vcc, v115, v147, vcc
	v_mov_b32_e32 v115, s57
	s_nop 0
	v_addc_co_u32_e32 v145, vcc, 0, v145, vcc
	v_lshl_add_u64 v[114:115], v[114:115], 0, v[144:145]
	v_mul_lo_u32 v104, s29, v114
	v_mul_lo_u32 v115, s28, v115
	v_mad_u64_u32 v[144:145], s[14:15], s28, v114, 0
	v_add3_u32 v104, v145, v115, v104
	v_sub_co_u32_e32 v115, vcc, v143, v144
	s_nop 1
	v_subb_co_u32_e64 v143, s[14:15], v141, v104, vcc
	v_sub_u32_e32 v104, v141, v104
	v_cmp_le_u32_e64 s[14:15], s29, v143
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	s_nop 0
	v_cndmask_b32_e64 v144, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v115
	v_subrev_co_u32_e32 v115, vcc, s28, v115
	s_nop 0
	v_cndmask_b32_e64 v145, 0, -1, s[14:15]
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cmp_le_u32_e32 vcc, s29, v104
	v_cmp_eq_u32_e64 s[14:15], s29, v143
	s_nop 0
	v_cndmask_b32_e64 v141, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v115
	v_cndmask_b32_e64 v143, v144, v145, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v115, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v104, v141, v115, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	s_nop 1
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v114, v104
	v_cmp_ne_u32_e32 vcc, 0, v143
	s_nop 1
	v_cndmask_b32_e32 v104, v114, v104, vcc
	v_xor_b32_e32 v114, s70, v142
	v_xor_b32_e32 v104, v104, v114
	v_sub_u32_e32 v104, v104, v114
	v_ashrrev_i32_e32 v114, 31, v131
	v_mov_b32_e32 v115, v114
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[114:115]
	v_xor_b32_e32 v146, v130, v114
	v_xad_u32 v141, v104, v114, v46
	v_xor_b32_e32 v115, v131, v114
	v_mad_u64_u32 v[130:131], s[14:15], v146, s35, 0
	v_mul_hi_u32 v104, v146, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[144:145], s[14:15], v115, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v144
	v_mad_u64_u32 v[142:143], s[14:15], v115, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v130, vcc, v131, v145, vcc
	v_mov_b32_e32 v131, s57
	s_nop 0
	v_addc_co_u32_e32 v143, vcc, 0, v143, vcc
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[142:143]
	v_mul_lo_u32 v104, s29, v130
	v_mul_lo_u32 v142, s28, v131
	v_mad_u64_u32 v[130:131], s[14:15], s28, v130, 0
	v_add3_u32 v104, v131, v142, v104
	v_sub_u32_e32 v131, v115, v104
	v_sub_co_u32_e32 v130, vcc, v146, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v142, s[14:15], s28, v130
	v_subb_co_u32_e32 v104, vcc, v115, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v143, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s29, v143
	v_subb_co_u32_e64 v81, s[14:15], v131, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v144, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s28, v142
	v_subrev_co_u32_e64 v131, s[14:15], s28, v142
	s_nop 0
	v_cndmask_b32_e64 v145, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s29, v143
	v_subbrev_co_u32_e64 v81, s[14:15], 0, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v144, v144, v145, s[26:27]
	v_cmp_le_u32_e32 vcc, s29, v104
	v_cmp_ne_u32_e64 s[14:15], 0, v144
	s_nop 0
	v_cndmask_b32_e64 v115, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v130
	v_cndmask_b32_e64 v81, v143, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v143, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_lshl_b64 s[28:29], s[88:89], 8
	s_nop 0
	v_cndmask_b32_e32 v115, v115, v143, vcc
	v_cmp_ne_u32_e32 vcc, 0, v115
	v_mov_b32_e32 v115, s29
	s_nop 0
	v_cndmask_b32_e32 v81, v104, v81, vcc
	v_cndmask_b32_e64 v104, v142, v131, s[14:15]
	v_cndmask_b32_e32 v104, v130, v104, vcc
	v_xor_b32_e32 v104, v104, v114
	v_xor_b32_e32 v81, v81, v114
	v_sub_co_u32_e32 v104, vcc, v104, v114
	s_nop 1
	v_subb_co_u32_e32 v81, vcc, v81, v114, vcc
	v_ashrrev_i32_e32 v81, 31, v81
	v_and_b32_e32 v81, s44, v81
	v_mul_lo_u32 v114, v141, s33
	v_add3_u32 v104, v81, v104, v114
	buffer_load_ubyte v81, v116, s[36:39], 0 offen
	buffer_load_ubyte v142, v118, s[36:39], 0 offen
	buffer_load_ubyte v141, v119, s[36:39], 0 offen
	buffer_load_ubyte v143, v120, s[36:39], 0 offen
	buffer_load_ubyte v144, v121, s[36:39], 0 offen
	buffer_load_ubyte v146, v162, s[36:39], 0 offen
	buffer_load_ubyte v145, v148, s[36:39], 0 offen
	buffer_load_ubyte v147, v104, s[36:39], 0 offen
	v_or_b32_e32 v114, s28, v108
	v_lshl_add_u64 v[124:125], v[114:115], 0, v[110:111]
	v_sub_co_u32_e32 v104, vcc, 0xfffffeff, v124
	ds_read_b128 v[162:165], v78 offset:8192
	v_lshl_add_u64 v[118:119], v[124:125], 0, s[50:51]
	v_subb_co_u32_e32 v116, vcc, -1, v125, vcc
	v_cmp_gt_i32_e32 vcc, 0, v119
	s_nop 1
	v_cndmask_b32_e32 v123, v119, v116, vcc
	v_cndmask_b32_e32 v122, v118, v104, vcc
	v_or_b32_e32 v104, s77, v123
	v_cmp_ne_u32_e32 vcc, 0, v104
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[26:27], exec, s[14:15]
	s_cbranch_execz .LBB0_41
	s_add_u32 s14, s76, s78
	s_mov_b32 s79, s78
	s_addc_u32 s15, s77, s78
	s_xor_b64 s[34:35], s[14:15], s[78:79]
	v_cvt_f32_u32_e32 v104, s34
	v_cvt_f32_u32_e32 v116, s35
	s_sub_u32 s14, 0, s34
	s_subb_u32 s15, 0, s35
	v_ashrrev_i32_e32 v120, 31, v123
	v_fmac_f32_e32 v104, 0x4f800000, v116
	v_rcp_f32_e32 v104, v104
	v_mov_b32_e32 v121, v120
	v_lshl_add_u64 v[122:123], v[122:123], 0, v[120:121]
	v_xor_b32_e32 v121, v122, v120
	v_mul_f32_e32 v104, 0x5f7ffffc, v104
	v_mul_f32_e32 v116, 0x2f800000, v104
	v_trunc_f32_e32 v116, v116
	v_fmac_f32_e32 v104, 0xcf800000, v116
	v_cvt_u32_f32_e32 v116, v116
	v_cvt_u32_f32_e32 v104, v104
	v_readfirstlane_b32 s48, v116
	v_readfirstlane_b32 s49, v104
	s_mul_i32 s55, s14, s48
	s_mul_hi_u32 s63, s14, s49
	s_mul_i32 s62, s15, s49
	s_add_i32 s55, s63, s55
	s_add_i32 s55, s55, s62
	s_mul_i32 s71, s14, s49
	s_mul_i32 s63, s49, s55
	s_mul_hi_u32 s79, s49, s71
	s_mul_hi_u32 s62, s49, s55
	s_add_u32 s63, s79, s63
	s_addc_u32 s62, 0, s62
	s_mul_hi_u32 s81, s48, s71
	s_mul_i32 s71, s48, s71
	s_add_u32 s63, s63, s71
	s_mul_hi_u32 s79, s48, s55
	s_addc_u32 s62, s62, s81
	s_addc_u32 s63, s79, 0
	s_mul_i32 s55, s48, s55
	s_add_u32 s55, s62, s55
	s_addc_u32 s62, 0, s63
	s_add_u32 s49, s49, s55
	s_addc_u32 s48, s48, s62
	s_mul_i32 s55, s14, s48
	s_mul_hi_u32 s62, s14, s49
	s_add_i32 s55, s62, s55
	s_mul_i32 s15, s15, s49
	s_add_i32 s55, s55, s15
	s_mul_i32 s14, s14, s49
	s_mul_hi_u32 s62, s48, s14
	s_mul_i32 s63, s48, s14
	s_mul_i32 s79, s49, s55
	s_mul_hi_u32 s14, s49, s14
	s_mul_hi_u32 s71, s49, s55
	s_add_u32 s14, s14, s79
	s_addc_u32 s71, 0, s71
	s_add_u32 s14, s14, s63
	s_mul_hi_u32 s15, s48, s55
	s_addc_u32 s14, s71, s62
	s_addc_u32 s15, s15, 0
	s_mul_i32 s55, s48, s55
	s_add_u32 s14, s14, s55
	s_addc_u32 s15, 0, s15
	s_add_u32 s49, s49, s14
	s_addc_u32 s48, s48, s15
	v_xor_b32_e32 v116, v123, v120
	v_mad_u64_u32 v[122:123], s[14:15], v121, s48, 0
	v_mul_hi_u32 v104, v121, s49
	v_lshl_add_u64 v[122:123], v[104:105], 0, v[122:123]
	v_mad_u64_u32 v[148:149], s[14:15], v116, s49, 0
	v_add_co_u32_e32 v104, vcc, v122, v148
	v_mad_u64_u32 v[130:131], s[14:15], v116, s48, 0
	s_nop 0
	v_addc_co_u32_e32 v122, vcc, v123, v149, vcc
	v_mov_b32_e32 v123, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[122:123], v[122:123], 0, v[130:131]
	v_mul_lo_u32 v104, s35, v122
	v_mul_lo_u32 v148, s34, v123
	v_mad_u64_u32 v[130:131], s[14:15], s34, v122, 0
	v_add3_u32 v104, v131, v148, v104
	v_sub_u32_e32 v131, v116, v104
	v_mov_b32_e32 v148, s35
	v_sub_co_u32_e32 v121, vcc, v121, v130
	s_nop 1
	v_subb_co_u32_e64 v130, s[14:15], v131, v148, vcc
	v_subrev_co_u32_e64 v131, s[14:15], s34, v121
	v_subb_co_u32_e32 v104, vcc, v116, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v130, s[14:15], 0, v130, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v130
	v_cmp_le_u32_e32 vcc, s35, v104
	s_nop 0
	v_cndmask_b32_e64 v148, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v131
	v_cndmask_b32_e64 v116, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v121
	v_cndmask_b32_e64 v131, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v130
	v_cndmask_b32_e64 v121, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v104
	v_cndmask_b32_e64 v166, v148, v131, s[14:15]
	v_lshl_add_u64 v[130:131], v[122:123], 0, 2
	v_lshl_add_u64 v[148:149], v[122:123], 0, 1
	v_cndmask_b32_e32 v104, v116, v121, vcc
	v_cmp_ne_u32_e32 vcc, 0, v166
	s_nop 1
	v_cndmask_b32_e32 v116, v148, v130, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	s_nop 1
	v_cndmask_b32_e32 v104, v122, v116, vcc
	v_xor_b32_e32 v116, s78, v120
	v_xor_b32_e32 v104, v104, v116
	v_sub_co_u32_e32 v120, vcc, v104, v116
.LBB0_41:
	s_andn2_saveexec_b64 s[14:15], s[26:27]
	s_cbranch_execz .LBB0_43
	v_rcp_iflag_f32_e32 v104, v59
	s_sub_i32 s26, 0, s76
	v_mul_f32_e32 v104, 0x4f7ffffe, v104
	v_cvt_u32_f32_e32 v104, v104
	v_mul_lo_u32 v116, s26, v104
	v_mul_hi_u32 v116, v104, v116
	v_add_u32_e32 v104, v104, v116
	v_mul_hi_u32 v104, v122, v104
	v_mul_lo_u32 v116, v104, s76
	v_sub_u32_e32 v116, v122, v116
	v_add_u32_e32 v120, 1, v104
	v_subrev_u32_e32 v121, s76, v116
	v_cmp_le_u32_e32 vcc, s76, v116
	s_nop 1
	v_cndmask_b32_e32 v116, v116, v121, vcc
	v_cndmask_b32_e32 v104, v104, v120, vcc
	v_add_u32_e32 v120, 1, v104
	v_cmp_le_u32_e32 vcc, s76, v116
	s_nop 1
	v_cndmask_b32_e32 v120, v104, v120, vcc
.LBB0_43:
	s_or_b64 exec, exec, s[14:15]
	v_lshl_add_u64 v[114:115], v[114:115], 0, v[112:113]
	v_lshl_add_u64 v[122:123], v[114:115], 0, s[50:51]
	v_or_b32_e32 v104, s77, v123
	v_cmp_ne_u32_e32 vcc, 0, v104
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[34:35], exec, s[14:15]
	s_cbranch_execz .LBB0_45
	s_add_u32 s14, s76, s78
	s_mov_b32 s79, s78
	s_addc_u32 s15, s77, s78
	s_xor_b64 s[48:49], s[14:15], s[78:79]
	v_cvt_f32_u32_e32 v104, s48
	v_cvt_f32_u32_e32 v116, s49
	s_sub_u32 s14, 0, s48
	s_subb_u32 s15, 0, s49
	v_ashrrev_i32_e32 v148, 31, v123
	v_fmac_f32_e32 v104, 0x4f800000, v116
	v_rcp_f32_e32 v104, v104
	v_mov_b32_e32 v149, v148
	v_lshl_add_u64 v[122:123], v[122:123], 0, v[148:149]
	v_xor_b32_e32 v121, v122, v148
	v_mul_f32_e32 v104, 0x5f7ffffc, v104
	v_mul_f32_e32 v116, 0x2f800000, v104
	v_trunc_f32_e32 v116, v116
	v_fmac_f32_e32 v104, 0xcf800000, v116
	v_cvt_u32_f32_e32 v116, v116
	v_cvt_u32_f32_e32 v104, v104
	v_readfirstlane_b32 s26, v116
	v_readfirstlane_b32 s27, v104
	s_mul_i32 s55, s14, s26
	s_mul_hi_u32 s63, s14, s27
	s_mul_i32 s62, s15, s27
	s_add_i32 s55, s63, s55
	s_add_i32 s55, s55, s62
	s_mul_i32 s71, s14, s27
	s_mul_i32 s63, s27, s55
	s_mul_hi_u32 s79, s27, s71
	s_mul_hi_u32 s62, s27, s55
	s_add_u32 s63, s79, s63
	s_addc_u32 s62, 0, s62
	s_mul_hi_u32 s81, s26, s71
	s_mul_i32 s71, s26, s71
	s_add_u32 s63, s63, s71
	s_mul_hi_u32 s79, s26, s55
	s_addc_u32 s62, s62, s81
	s_addc_u32 s63, s79, 0
	s_mul_i32 s55, s26, s55
	s_add_u32 s55, s62, s55
	s_addc_u32 s62, 0, s63
	s_add_u32 s27, s27, s55
	s_addc_u32 s26, s26, s62
	s_mul_i32 s55, s14, s26
	s_mul_hi_u32 s62, s14, s27
	s_add_i32 s55, s62, s55
	s_mul_i32 s15, s15, s27
	s_add_i32 s55, s55, s15
	s_mul_i32 s14, s14, s27
	s_mul_hi_u32 s62, s26, s14
	s_mul_i32 s63, s26, s14
	s_mul_i32 s79, s27, s55
	s_mul_hi_u32 s14, s27, s14
	s_mul_hi_u32 s71, s27, s55
	s_add_u32 s14, s14, s79
	s_addc_u32 s71, 0, s71
	s_add_u32 s14, s14, s63
	s_mul_hi_u32 s15, s26, s55
	s_addc_u32 s14, s71, s62
	s_addc_u32 s15, s15, 0
	s_mul_i32 s55, s26, s55
	s_add_u32 s14, s14, s55
	s_addc_u32 s15, 0, s15
	s_add_u32 s27, s27, s14
	s_addc_u32 s26, s26, s15
	v_xor_b32_e32 v116, v123, v148
	v_mad_u64_u32 v[122:123], s[14:15], v121, s26, 0
	v_mul_hi_u32 v104, v121, s27
	v_lshl_add_u64 v[122:123], v[104:105], 0, v[122:123]
	v_mad_u64_u32 v[166:167], s[14:15], v116, s27, 0
	v_add_co_u32_e32 v104, vcc, v122, v166
	v_mad_u64_u32 v[130:131], s[14:15], v116, s26, 0
	s_nop 0
	v_addc_co_u32_e32 v122, vcc, v123, v167, vcc
	v_mov_b32_e32 v123, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[122:123], v[122:123], 0, v[130:131]
	v_mul_lo_u32 v104, s49, v122
	v_mul_lo_u32 v130, s48, v123
	v_mad_u64_u32 v[122:123], s[14:15], s48, v122, 0
	v_add3_u32 v104, v123, v130, v104
	v_sub_u32_e32 v123, v116, v104
	v_mov_b32_e32 v130, s49
	v_sub_co_u32_e32 v121, vcc, v121, v122
	s_nop 1
	v_subb_co_u32_e64 v122, s[14:15], v123, v130, vcc
	v_subrev_co_u32_e64 v123, s[14:15], s48, v121
	v_subb_co_u32_e32 v104, vcc, v116, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[26:27], 0, v122, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s49, v131
	v_subb_co_u32_e64 v122, s[14:15], v122, v130, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v149, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s48, v123
	v_subrev_co_u32_e64 v130, s[14:15], s48, v123
	s_nop 0
	v_cndmask_b32_e64 v166, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s49, v131
	v_subbrev_co_u32_e64 v122, s[14:15], 0, v122, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v149, v149, v166, s[26:27]
	v_cmp_le_u32_e32 vcc, s49, v104
	v_cmp_ne_u32_e64 s[14:15], 0, v149
	s_nop 0
	v_cndmask_b32_e64 v116, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s48, v121
	v_cndmask_b32_e64 v122, v131, v122, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s49, v104
	s_nop 1
	v_cndmask_b32_e32 v116, v116, v131, vcc
	v_cmp_ne_u32_e32 vcc, 0, v116
	v_cndmask_b32_e64 v116, v123, v130, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v116, v121, v116, vcc
	v_cndmask_b32_e32 v104, v104, v122, vcc
	v_xor_b32_e32 v116, v116, v148
	v_xor_b32_e32 v104, v104, v148
	v_sub_co_u32_e32 v130, vcc, v116, v148
	s_nop 1
	v_subb_co_u32_e32 v131, vcc, v104, v148, vcc
.LBB0_45:
	s_andn2_saveexec_b64 s[14:15], s[34:35]
	s_cbranch_execz .LBB0_47
	v_rcp_iflag_f32_e32 v104, v59
	s_sub_i32 s26, 0, s76
	v_mul_f32_e32 v104, 0x4f7ffffe, v104
	v_cvt_u32_f32_e32 v104, v104
	v_mul_lo_u32 v116, s26, v104
	v_mul_hi_u32 v116, v104, v116
	v_add_u32_e32 v104, v104, v116
	v_mul_hi_u32 v104, v122, v104
	v_mul_lo_u32 v104, v104, s76
	v_sub_u32_e32 v104, v122, v104
	v_subrev_u32_e32 v116, s76, v104
	v_cmp_le_u32_e32 vcc, s76, v104
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v116, vcc
	v_subrev_u32_e32 v116, s76, v104
	v_cmp_le_u32_e32 vcc, s76, v104
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v116, vcc
	v_mov_b64_e32 v[130:131], v[104:105]
.LBB0_47:
	s_or_b64 exec, exec, s[14:15]
	v_ashrrev_i32_e32 v104, 31, v119
	v_xor_b32_e32 v104, v120, v104
	v_ashrrev_i32_e32 v116, 31, v131
	v_mul_lo_u32 v104, v104, s54
	v_and_b32_e32 v116, s76, v116
	v_add3_u32 v104, v104, v130, v116
	buffer_load_ubyte v148, v104, s[8:11], 0 offen
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[154:157], v[16:19], v[8:11], v117, v47 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_sub_co_u32_e32 v104, vcc, -3, v118
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[158:161], v[28:31], v[36:39], v117, v132 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[150:153], v[28:31], v[8:11], v117, v132 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 v[174:177], v78 offset:10240
	ds_read_b128 v[166:169], v78 offset:12288
	ds_read_b128 v[170:173], v76 offset:10240
	ds_read_b128 v[158:161], v76 offset:12288
	ds_read_b128 v[178:181], v76 offset:8192
	ds_read_b128 v[154:157], v78 offset:14336
	ds_read_b128 v[150:153], v76 offset:14336
	ds_read2st64_b32 v[120:121], v63 offset0:194 offset1:195
	v_lshl_add_u64 v[116:117], v[118:119], 0, 2
	v_subb_co_u32_e32 v118, vcc, -1, v119, vcc
	v_cmp_gt_i32_e32 vcc, 0, v117
	s_nop 1
	v_cndmask_b32_e32 v123, v117, v118, vcc
	v_cndmask_b32_e32 v122, v116, v104, vcc
	v_or_b32_e32 v104, s77, v123
	v_cmp_ne_u32_e32 vcc, 0, v104
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[26:27], exec, s[14:15]
	s_cbranch_execz .LBB0_49
	s_add_u32 s14, s76, s78
	s_mov_b32 s79, s78
	s_addc_u32 s15, s77, s78
	s_xor_b64 s[34:35], s[14:15], s[78:79]
	v_cvt_f32_u32_e32 v104, s34
	v_cvt_f32_u32_e32 v116, s35
	s_sub_u32 s14, 0, s34
	s_subb_u32 s15, 0, s35
	v_ashrrev_i32_e32 v118, 31, v123
	v_fmac_f32_e32 v104, 0x4f800000, v116
	v_rcp_f32_e32 v104, v104
	v_mov_b32_e32 v119, v118
	v_lshl_add_u64 v[122:123], v[122:123], 0, v[118:119]
	v_xor_b32_e32 v119, v122, v118
	v_mul_f32_e32 v104, 0x5f7ffffc, v104
	v_mul_f32_e32 v116, 0x2f800000, v104
	v_trunc_f32_e32 v116, v116
	v_fmac_f32_e32 v104, 0xcf800000, v116
	v_cvt_u32_f32_e32 v116, v116
	v_cvt_u32_f32_e32 v104, v104
	v_readfirstlane_b32 s48, v116
	v_readfirstlane_b32 s49, v104
	s_mul_i32 s55, s14, s48
	s_mul_hi_u32 s63, s14, s49
	s_mul_i32 s62, s15, s49
	s_add_i32 s55, s63, s55
	s_add_i32 s55, s55, s62
	s_mul_i32 s71, s14, s49
	s_mul_i32 s63, s49, s55
	s_mul_hi_u32 s79, s49, s71
	s_mul_hi_u32 s62, s49, s55
	s_add_u32 s63, s79, s63
	s_addc_u32 s62, 0, s62
	s_mul_hi_u32 s81, s48, s71
	s_mul_i32 s71, s48, s71
	s_add_u32 s63, s63, s71
	s_mul_hi_u32 s79, s48, s55
	s_addc_u32 s62, s62, s81
	s_addc_u32 s63, s79, 0
	s_mul_i32 s55, s48, s55
	s_add_u32 s55, s62, s55
	s_addc_u32 s62, 0, s63
	s_add_u32 s49, s49, s55
	s_addc_u32 s48, s48, s62
	s_mul_i32 s55, s14, s48
	s_mul_hi_u32 s62, s14, s49
	s_add_i32 s55, s62, s55
	s_mul_i32 s15, s15, s49
	s_add_i32 s55, s55, s15
	s_mul_i32 s14, s14, s49
	s_mul_hi_u32 s62, s48, s14
	s_mul_i32 s63, s48, s14
	s_mul_i32 s79, s49, s55
	s_mul_hi_u32 s14, s49, s14
	s_mul_hi_u32 s71, s49, s55
	s_add_u32 s14, s14, s79
	s_addc_u32 s71, 0, s71
	s_add_u32 s14, s14, s63
	s_mul_hi_u32 s15, s48, s55
	s_addc_u32 s14, s71, s62
	s_addc_u32 s15, s15, 0
	s_mul_i32 s55, s48, s55
	s_add_u32 s14, s14, s55
	s_addc_u32 s15, 0, s15
	s_add_u32 s49, s49, s14
	s_addc_u32 s48, s48, s15
	v_xor_b32_e32 v116, v123, v118
	v_mad_u64_u32 v[122:123], s[14:15], v119, s48, 0
	v_mul_hi_u32 v104, v119, s49
	v_lshl_add_u64 v[122:123], v[104:105], 0, v[122:123]
	v_mad_u64_u32 v[182:183], s[14:15], v116, s49, 0
	v_add_co_u32_e32 v104, vcc, v122, v182
	v_mad_u64_u32 v[130:131], s[14:15], v116, s48, 0
	s_nop 0
	v_addc_co_u32_e32 v122, vcc, v123, v183, vcc
	v_mov_b32_e32 v123, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[122:123], v[122:123], 0, v[130:131]
	v_mul_lo_u32 v104, s35, v122
	v_mul_lo_u32 v149, s34, v123
	v_mad_u64_u32 v[130:131], s[14:15], s34, v122, 0
	v_add3_u32 v104, v131, v149, v104
	v_sub_u32_e32 v131, v116, v104
	v_mov_b32_e32 v149, s35
	v_sub_co_u32_e32 v119, vcc, v119, v130
	v_lshl_add_u64 v[182:183], v[122:123], 0, 1
	s_nop 0
	v_subb_co_u32_e64 v130, s[14:15], v131, v149, vcc
	v_subrev_co_u32_e64 v131, s[14:15], s34, v119
	v_subb_co_u32_e32 v104, vcc, v116, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v130, s[14:15], 0, v130, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v130
	v_cmp_le_u32_e32 vcc, s35, v104
	s_nop 0
	v_cndmask_b32_e64 v149, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v131
	v_cndmask_b32_e64 v116, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v119
	v_cndmask_b32_e64 v131, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v130
	v_cndmask_b32_e64 v119, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v104
	v_cndmask_b32_e64 v149, v149, v131, s[14:15]
	v_lshl_add_u64 v[130:131], v[122:123], 0, 2
	v_cndmask_b32_e32 v104, v116, v119, vcc
	v_cmp_ne_u32_e32 vcc, 0, v149
	s_nop 1
	v_cndmask_b32_e32 v116, v182, v130, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	s_nop 1
	v_cndmask_b32_e32 v104, v122, v116, vcc
	v_xor_b32_e32 v116, s78, v118
	v_xor_b32_e32 v104, v104, v116
	v_sub_co_u32_e32 v118, vcc, v104, v116
.LBB0_49:
	s_andn2_saveexec_b64 s[14:15], s[26:27]
	s_cbranch_execz .LBB0_51
	v_rcp_iflag_f32_e32 v104, v59
	s_sub_i32 s26, 0, s76
	v_mul_f32_e32 v104, 0x4f7ffffe, v104
	v_cvt_u32_f32_e32 v104, v104
	v_mul_lo_u32 v116, s26, v104
	v_mul_hi_u32 v116, v104, v116
	v_add_u32_e32 v104, v104, v116
	v_mul_hi_u32 v104, v122, v104
	v_mul_lo_u32 v116, v104, s76
	v_sub_u32_e32 v116, v122, v116
	v_add_u32_e32 v118, 1, v104
	v_subrev_u32_e32 v119, s76, v116
	v_cmp_le_u32_e32 vcc, s76, v116
	s_nop 1
	v_cndmask_b32_e32 v116, v116, v119, vcc
	v_cndmask_b32_e32 v104, v104, v118, vcc
	v_add_u32_e32 v118, 1, v104
	v_cmp_le_u32_e32 vcc, s76, v116
	s_nop 1
	v_cndmask_b32_e32 v118, v104, v118, vcc
.LBB0_51:
	s_or_b64 exec, exec, s[14:15]
	s_mov_b64 s[14:15], 0x102
	v_lshl_add_u64 v[122:123], v[114:115], 0, s[14:15]
	v_or_b32_e32 v104, s77, v123
	v_cmp_ne_u32_e32 vcc, 0, v104
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[34:35], exec, s[14:15]
	s_cbranch_execz .LBB0_53
	s_add_u32 s14, s76, s78
	s_mov_b32 s79, s78
	s_addc_u32 s15, s77, s78
	s_xor_b64 s[48:49], s[14:15], s[78:79]
	v_cvt_f32_u32_e32 v104, s48
	v_cvt_f32_u32_e32 v116, s49
	s_sub_u32 s14, 0, s48
	s_subb_u32 s15, 0, s49
	v_ashrrev_i32_e32 v182, 31, v123
	v_fmac_f32_e32 v104, 0x4f800000, v116
	v_rcp_f32_e32 v104, v104
	v_mov_b32_e32 v183, v182
	v_lshl_add_u64 v[122:123], v[122:123], 0, v[182:183]
	v_xor_b32_e32 v119, v122, v182
	v_mul_f32_e32 v104, 0x5f7ffffc, v104
	v_mul_f32_e32 v116, 0x2f800000, v104
	v_trunc_f32_e32 v116, v116
	v_fmac_f32_e32 v104, 0xcf800000, v116
	v_cvt_u32_f32_e32 v116, v116
	v_cvt_u32_f32_e32 v104, v104
	v_readfirstlane_b32 s26, v116
	v_readfirstlane_b32 s27, v104
	s_mul_i32 s55, s14, s26
	s_mul_hi_u32 s63, s14, s27
	s_mul_i32 s62, s15, s27
	s_add_i32 s55, s63, s55
	s_add_i32 s55, s55, s62
	s_mul_i32 s71, s14, s27
	s_mul_i32 s63, s27, s55
	s_mul_hi_u32 s79, s27, s71
	s_mul_hi_u32 s62, s27, s55
	s_add_u32 s63, s79, s63
	s_addc_u32 s62, 0, s62
	s_mul_hi_u32 s81, s26, s71
	s_mul_i32 s71, s26, s71
	s_add_u32 s63, s63, s71
	s_mul_hi_u32 s79, s26, s55
	s_addc_u32 s62, s62, s81
	s_addc_u32 s63, s79, 0
	s_mul_i32 s55, s26, s55
	s_add_u32 s55, s62, s55
	s_addc_u32 s62, 0, s63
	s_add_u32 s27, s27, s55
	s_addc_u32 s26, s26, s62
	s_mul_i32 s55, s14, s26
	s_mul_hi_u32 s62, s14, s27
	s_add_i32 s55, s62, s55
	s_mul_i32 s15, s15, s27
	s_add_i32 s55, s55, s15
	s_mul_i32 s14, s14, s27
	s_mul_hi_u32 s62, s26, s14
	s_mul_i32 s63, s26, s14
	s_mul_i32 s79, s27, s55
	s_mul_hi_u32 s14, s27, s14
	s_mul_hi_u32 s71, s27, s55
	s_add_u32 s14, s14, s79
	s_addc_u32 s71, 0, s71
	s_add_u32 s14, s14, s63
	s_mul_hi_u32 s15, s26, s55
	s_addc_u32 s14, s71, s62
	s_addc_u32 s15, s15, 0
	s_mul_i32 s55, s26, s55
	s_add_u32 s14, s14, s55
	s_addc_u32 s15, 0, s15
	s_add_u32 s27, s27, s14
	s_addc_u32 s26, s26, s15
	v_xor_b32_e32 v116, v123, v182
	v_mad_u64_u32 v[122:123], s[14:15], v119, s26, 0
	v_mul_hi_u32 v104, v119, s27
	v_lshl_add_u64 v[122:123], v[104:105], 0, v[122:123]
	v_mad_u64_u32 v[184:185], s[14:15], v116, s27, 0
	v_add_co_u32_e32 v104, vcc, v122, v184
	v_mad_u64_u32 v[130:131], s[14:15], v116, s26, 0
	s_nop 0
	v_addc_co_u32_e32 v122, vcc, v123, v185, vcc
	v_mov_b32_e32 v123, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[122:123], v[122:123], 0, v[130:131]
	v_mul_lo_u32 v104, s49, v122
	v_mul_lo_u32 v130, s48, v123
	v_mad_u64_u32 v[122:123], s[14:15], s48, v122, 0
	v_add3_u32 v104, v123, v130, v104
	v_sub_u32_e32 v123, v116, v104
	v_mov_b32_e32 v130, s49
	v_sub_co_u32_e32 v119, vcc, v119, v122
	s_nop 1
	v_subb_co_u32_e64 v122, s[14:15], v123, v130, vcc
	v_subrev_co_u32_e64 v123, s[14:15], s48, v119
	v_subb_co_u32_e32 v104, vcc, v116, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[26:27], 0, v122, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s49, v131
	v_subb_co_u32_e64 v122, s[14:15], v122, v130, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v149, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s48, v123
	v_subrev_co_u32_e64 v130, s[14:15], s48, v123
	s_nop 0
	v_cndmask_b32_e64 v183, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s49, v131
	v_subbrev_co_u32_e64 v122, s[14:15], 0, v122, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v149, v149, v183, s[26:27]
	v_cmp_le_u32_e32 vcc, s49, v104
	v_cmp_ne_u32_e64 s[14:15], 0, v149
	s_nop 0
	v_cndmask_b32_e64 v116, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s48, v119
	v_cndmask_b32_e64 v122, v131, v122, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s49, v104
	s_nop 1
	v_cndmask_b32_e32 v116, v116, v131, vcc
	v_cmp_ne_u32_e32 vcc, 0, v116
	v_cndmask_b32_e64 v116, v123, v130, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v116, v119, v116, vcc
	v_cndmask_b32_e32 v104, v104, v122, vcc
	v_xor_b32_e32 v116, v116, v182
	v_xor_b32_e32 v104, v104, v182
	v_sub_co_u32_e32 v130, vcc, v116, v182
	s_nop 1
	v_subb_co_u32_e32 v131, vcc, v104, v182, vcc
.LBB0_53:
	s_andn2_saveexec_b64 s[14:15], s[34:35]
	s_cbranch_execz .LBB0_55
	v_rcp_iflag_f32_e32 v104, v59
	s_sub_i32 s26, 0, s76
	v_mul_f32_e32 v104, 0x4f7ffffe, v104
	v_cvt_u32_f32_e32 v104, v104
	v_mul_lo_u32 v116, s26, v104
	v_mul_hi_u32 v116, v104, v116
	v_add_u32_e32 v104, v104, v116
	v_mul_hi_u32 v104, v122, v104
	v_mul_lo_u32 v104, v104, s76
	v_sub_u32_e32 v104, v122, v104
	v_subrev_u32_e32 v116, s76, v104
	v_cmp_le_u32_e32 vcc, s76, v104
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v116, vcc
	v_subrev_u32_e32 v116, s76, v104
	v_cmp_le_u32_e32 vcc, s76, v104
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v116, vcc
	v_mov_b64_e32 v[130:131], v[104:105]
.LBB0_55:
	s_or_b64 exec, exec, s[14:15]
	v_ashrrev_i32_e32 v104, 31, v117
	v_xor_b32_e32 v104, v118, v104
	v_ashrrev_i32_e32 v116, 31, v131
	v_mul_lo_u32 v104, v104, s54
	v_and_b32_e32 v116, s76, v116
	v_add3_u32 v104, v104, v130, v116
	buffer_load_ubyte v117, v104, s[8:11], 0 offen
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	s_lshl_b64 s[14:15], s[88:89], 7
	v_mov_b32_e32 v119, s29
	v_or_b32_e32 v118, s28, v52
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[162:165], v[16:19], v[20:23], v120, v47 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_or_b32_e32 v116, s14, v48
	v_lshl_add_u64 v[122:123], v[118:119], 0, s[58:59]
	v_add_u32_e32 v104, 0x100, v116
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[174:177], v[16:19], v[12:15], v120, v47 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_gt_i64_e32 vcc, s[30:31], v[122:123]
	v_add_u32_e32 v130, v104, v50
	s_and_b64 s[14:15], s[2:3], vcc
	s_mov_b32 m0, s52
	v_cndmask_b32_e64 v122, -1, v130, s[14:15]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[178:181], v[28:31], v[20:23], v120, v132 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_and_b64 s[14:15], s[24:25], vcc
	buffer_load_dwordx4 v122, s[40:43], 0 offen lds
	s_mov_b32 m0, s64
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[170:173], v[28:31], v[12:15], v120, v132 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v120, v104, v54
	v_cndmask_b32_e64 v120, -1, v120, s[14:15]
	buffer_load_dwordx4 v120, s[40:43], 0 offen lds
	v_add_u32_e32 v120, v104, v56
	s_and_b64 s[14:15], s[18:19], vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[166:169], v[16:19], v[24:27], v121, v47 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e64 v120, -1, v120, s[14:15]
	s_mov_b32 m0, s65
	s_and_b64 vcc, s[20:21], vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[154:157], v[16:19], v[32:35], v121, v47 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v120, s[40:43], 0 offen lds
	s_mov_b32 m0, s96
	s_lshl_b64 s[14:15], s[88:89], 13
	v_add_u32_e32 v32, v104, v58
	v_cndmask_b32_e32 v32, -1, v32, vcc
	buffer_load_dwordx4 v32, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[158:161], v[28:31], v[24:27], v121, v132 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[150:153], v[28:31], v[16:19], v121, v132 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 v[178:181], v78 offset:16384
	ds_read_b128 v[162:165], v78 offset:18432
	ds_read_b128 v[174:177], v76 offset:16384
	ds_read_b128 v[170:173], v76 offset:18432
	ds_read_b128 v[166:169], v78 offset:20480
	ds_read_b128 v[154:157], v78 offset:22528
	ds_read_b128 v[158:161], v76 offset:20480
	ds_read_b128 v[150:153], v76 offset:22528
	ds_read2st64_b32 v[122:123], v63 offset0:196 offset1:197
	v_lshl_add_u64 v[120:121], v[60:61], 0, s[14:15]
	s_mov_b64 s[14:15], 0x4000
	v_sub_co_u32_e32 v30, vcc, 0xffffbfff, v120
	v_lshl_add_u64 v[28:29], v[120:121], 0, s[14:15]
	s_nop 0
	v_subb_co_u32_e32 v31, vcc, -1, v121, vcc
	v_cmp_gt_i32_e32 vcc, 0, v29
	s_nop 1
	v_cndmask_b32_e32 v33, v29, v31, vcc
	v_cndmask_b32_e32 v32, v28, v30, vcc
	v_or_b32_e32 v30, s31, v33
	v_cmp_ne_u32_e32 vcc, 0, v30
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[26:27], exec, s[14:15]
	s_cbranch_execz .LBB0_57
	s_add_u32 s14, s30, s80
	s_mov_b32 s81, s80
	s_addc_u32 s15, s31, s80
	s_xor_b64 s[28:29], s[14:15], s[80:81]
	v_cvt_f32_u32_e32 v30, s28
	v_cvt_f32_u32_e32 v31, s29
	s_sub_u32 s14, 0, s28
	s_subb_u32 s15, 0, s29
	v_fmac_f32_e32 v30, 0x4f800000, v31
	v_rcp_f32_e32 v30, v30
	s_nop 0
	v_mul_f32_e32 v30, 0x5f7ffffc, v30
	v_mul_f32_e32 v31, 0x2f800000, v30
	v_trunc_f32_e32 v31, v31
	v_fmac_f32_e32 v30, 0xcf800000, v31
	v_cvt_u32_f32_e32 v31, v31
	v_cvt_u32_f32_e32 v30, v30
	v_readfirstlane_b32 s34, v31
	v_readfirstlane_b32 s35, v30
	s_mul_i32 s48, s14, s34
	s_mul_hi_u32 s55, s14, s35
	s_mul_i32 s49, s15, s35
	s_add_i32 s48, s55, s48
	s_add_i32 s48, s48, s49
	s_mul_i32 s62, s14, s35
	s_mul_i32 s55, s35, s48
	s_mul_hi_u32 s63, s35, s62
	s_mul_hi_u32 s49, s35, s48
	s_add_u32 s55, s63, s55
	s_addc_u32 s49, 0, s49
	s_mul_hi_u32 s71, s34, s62
	s_mul_i32 s62, s34, s62
	s_add_u32 s55, s55, s62
	s_mul_hi_u32 s63, s34, s48
	s_addc_u32 s49, s49, s71
	s_addc_u32 s55, s63, 0
	s_mul_i32 s48, s34, s48
	s_add_u32 s48, s49, s48
	s_addc_u32 s49, 0, s55
	s_add_u32 s35, s35, s48
	s_addc_u32 s34, s34, s49
	s_mul_i32 s48, s14, s34
	s_mul_hi_u32 s49, s14, s35
	s_add_i32 s48, s49, s48
	s_mul_i32 s15, s15, s35
	s_add_i32 s48, s48, s15
	s_mul_i32 s14, s14, s35
	s_mul_hi_u32 s49, s34, s14
	s_mul_i32 s55, s34, s14
	s_mul_i32 s63, s35, s48
	s_mul_hi_u32 s14, s35, s14
	s_mul_hi_u32 s62, s35, s48
	s_add_u32 s14, s14, s63
	s_addc_u32 s62, 0, s62
	s_add_u32 s14, s14, s55
	s_mul_hi_u32 s15, s34, s48
	s_addc_u32 s14, s62, s49
	s_addc_u32 s15, s15, 0
	s_mul_i32 s48, s34, s48
	s_add_u32 s14, s14, s48
	v_ashrrev_i32_e32 v30, 31, v33
	s_addc_u32 s15, 0, s15
	v_mov_b32_e32 v31, v30
	s_add_u32 s35, s35, s14
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[30:31]
	s_addc_u32 s34, s34, s15
	v_xor_b32_e32 v47, v32, v30
	v_xor_b32_e32 v31, v33, v30
	v_mad_u64_u32 v[32:33], s[14:15], v47, s34, 0
	v_mul_hi_u32 v104, v47, s35
	v_lshl_add_u64 v[32:33], v[104:105], 0, v[32:33]
	v_mad_u64_u32 v[130:131], s[14:15], v31, s35, 0
	v_add_co_u32_e32 v32, vcc, v32, v130
	v_mad_u64_u32 v[34:35], s[14:15], v31, s34, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v33, v131, vcc
	v_mov_b32_e32 v33, s57
	s_nop 0
	v_addc_co_u32_e32 v35, vcc, 0, v35, vcc
	v_lshl_add_u64 v[32:33], v[32:33], 0, v[34:35]
	v_mul_lo_u32 v104, s29, v32
	v_mul_lo_u32 v130, s28, v33
	v_mad_u64_u32 v[34:35], s[14:15], s28, v32, 0
	v_add3_u32 v104, v35, v130, v104
	v_sub_u32_e32 v35, v31, v104
	v_mov_b32_e32 v130, s29
	v_sub_co_u32_e32 v47, vcc, v47, v34
	v_xor_b32_e32 v30, s80, v30
	s_nop 0
	v_subb_co_u32_e64 v34, s[14:15], v35, v130, vcc
	v_subrev_co_u32_e64 v35, s[14:15], s28, v47
	v_subb_co_u32_e32 v31, vcc, v31, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v34, s[14:15], 0, v34, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s29, v34
	v_cmp_le_u32_e32 vcc, s29, v31
	s_nop 0
	v_cndmask_b32_e64 v130, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v35
	s_nop 1
	v_cndmask_b32_e64 v35, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v34
	s_nop 1
	v_cndmask_b32_e64 v132, v130, v35, s[14:15]
	v_lshl_add_u64 v[34:35], v[32:33], 0, 2
	v_lshl_add_u64 v[130:131], v[32:33], 0, 1
	v_cndmask_b32_e64 v33, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v47
	s_nop 1
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v31
	s_nop 1
	v_cndmask_b32_e32 v31, v33, v35, vcc
	v_cmp_ne_u32_e32 vcc, 0, v132
	s_nop 1
	v_cndmask_b32_e32 v33, v130, v34, vcc
	v_cmp_ne_u32_e32 vcc, 0, v31
	s_nop 1
	v_cndmask_b32_e32 v31, v32, v33, vcc
	v_xor_b32_e32 v31, v31, v30
	v_sub_co_u32_e32 v30, vcc, v31, v30
.LBB0_57:
	s_andn2_saveexec_b64 s[14:15], s[26:27]
	s_cbranch_execz .LBB0_59
	v_cvt_f32_u32_e32 v30, s30
	s_sub_i32 s26, 0, s30
	v_rcp_iflag_f32_e32 v30, v30
	s_nop 0
	v_mul_f32_e32 v30, 0x4f7ffffe, v30
	v_cvt_u32_f32_e32 v30, v30
	v_mul_lo_u32 v31, s26, v30
	v_mul_hi_u32 v31, v30, v31
	v_add_u32_e32 v30, v30, v31
	v_mul_hi_u32 v30, v32, v30
	v_mul_lo_u32 v31, v30, s30
	v_sub_u32_e32 v31, v32, v31
	v_add_u32_e32 v33, 1, v30
	v_subrev_u32_e32 v32, s30, v31
	v_cmp_le_u32_e32 vcc, s30, v31
	s_nop 1
	v_cndmask_b32_e32 v31, v31, v32, vcc
	v_cndmask_b32_e32 v30, v30, v33, vcc
	v_add_u32_e32 v32, 1, v30
	v_cmp_le_u32_e32 vcc, s30, v31
	s_nop 1
	v_cndmask_b32_e32 v30, v30, v32, vcc
.LBB0_59:
	s_or_b64 exec, exec, s[14:15]
	v_or_b32_e32 v31, s31, v29
	v_cmp_ne_u32_e32 vcc, 0, v31
	v_ashrrev_i32_e32 v32, 31, v29
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[28:29], exec, s[14:15]
	s_cbranch_execz .LBB0_61
	s_add_u32 s14, s30, s80
	s_mov_b32 s81, s80
	s_addc_u32 s15, s31, s80
	s_xor_b64 s[34:35], s[14:15], s[80:81]
	v_cvt_f32_u32_e32 v31, s34
	v_cvt_f32_u32_e32 v33, s35
	s_sub_u32 s14, 0, s34
	s_subb_u32 s15, 0, s35
	v_fmac_f32_e32 v31, 0x4f800000, v33
	v_rcp_f32_e32 v31, v31
	s_nop 0
	v_mul_f32_e32 v31, 0x5f7ffffc, v31
	v_mul_f32_e32 v33, 0x2f800000, v31
	v_trunc_f32_e32 v33, v33
	v_fmac_f32_e32 v31, 0xcf800000, v33
	v_cvt_u32_f32_e32 v33, v33
	v_cvt_u32_f32_e32 v31, v31
	v_readfirstlane_b32 s26, v33
	v_readfirstlane_b32 s27, v31
	s_mul_i32 s48, s14, s26
	s_mul_hi_u32 s55, s14, s27
	s_mul_i32 s49, s15, s27
	s_add_i32 s48, s55, s48
	s_add_i32 s48, s48, s49
	s_mul_i32 s62, s14, s27
	s_mul_i32 s55, s27, s48
	s_mul_hi_u32 s63, s27, s62
	s_mul_hi_u32 s49, s27, s48
	s_add_u32 s55, s63, s55
	s_addc_u32 s49, 0, s49
	s_mul_hi_u32 s71, s26, s62
	s_mul_i32 s62, s26, s62
	s_add_u32 s55, s55, s62
	s_mul_hi_u32 s63, s26, s48
	s_addc_u32 s49, s49, s71
	s_addc_u32 s55, s63, 0
	s_mul_i32 s48, s26, s48
	s_add_u32 s48, s49, s48
	s_addc_u32 s49, 0, s55
	s_add_u32 s27, s27, s48
	s_addc_u32 s26, s26, s49
	s_mul_i32 s48, s14, s26
	s_mul_hi_u32 s49, s14, s27
	s_add_i32 s48, s49, s48
	s_mul_i32 s15, s15, s27
	s_add_i32 s48, s48, s15
	s_mul_i32 s14, s14, s27
	s_mul_hi_u32 s49, s26, s14
	s_mul_i32 s55, s26, s14
	s_mul_i32 s63, s27, s48
	s_mul_hi_u32 s14, s27, s14
	s_mul_hi_u32 s62, s27, s48
	s_add_u32 s14, s14, s63
	s_addc_u32 s62, 0, s62
	s_add_u32 s14, s14, s55
	s_mul_hi_u32 s15, s26, s48
	s_addc_u32 s14, s62, s49
	s_addc_u32 s15, s15, 0
	s_mul_i32 s48, s26, s48
	s_add_u32 s14, s14, s48
	s_addc_u32 s15, 0, s15
	v_mov_b32_e32 v33, v32
	s_add_u32 s27, s27, s14
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[32:33]
	s_addc_u32 s26, s26, s15
	v_xor_b32_e32 v33, v28, v32
	v_xor_b32_e32 v31, v29, v32
	v_mad_u64_u32 v[28:29], s[14:15], v33, s26, 0
	v_mul_hi_u32 v104, v33, s27
	v_lshl_add_u64 v[28:29], v[104:105], 0, v[28:29]
	v_mad_u64_u32 v[130:131], s[14:15], v31, s27, 0
	v_add_co_u32_e32 v28, vcc, v28, v130
	v_mad_u64_u32 v[34:35], s[14:15], v31, s26, 0
	s_nop 0
	v_addc_co_u32_e32 v28, vcc, v29, v131, vcc
	v_mov_b32_e32 v29, s57
	s_nop 0
	v_addc_co_u32_e32 v35, vcc, 0, v35, vcc
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[34:35]
	v_mul_lo_u32 v34, s35, v28
	v_mul_lo_u32 v35, s34, v29
	v_mad_u64_u32 v[28:29], s[14:15], s34, v28, 0
	v_add3_u32 v29, v29, v35, v34
	v_sub_u32_e32 v34, v31, v29
	v_mov_b32_e32 v35, s35
	v_sub_co_u32_e32 v28, vcc, v33, v28
	s_nop 1
	v_subb_co_u32_e64 v33, s[14:15], v34, v35, vcc
	v_subrev_co_u32_e64 v34, s[14:15], s34, v28
	v_subb_co_u32_e32 v29, vcc, v31, v29, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v47, s[26:27], 0, v33, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v47
	v_subb_co_u32_e64 v33, s[14:15], v33, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v34
	v_subrev_co_u32_e64 v35, s[14:15], s34, v34
	s_nop 0
	v_cndmask_b32_e64 v130, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v47
	v_subbrev_co_u32_e64 v33, s[14:15], 0, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v104, v104, v130, s[26:27]
	v_cmp_le_u32_e32 vcc, s35, v29
	v_cmp_ne_u32_e64 s[14:15], 0, v104
	s_nop 0
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v28
	v_cndmask_b32_e64 v33, v47, v33, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v29
	s_nop 1
	v_cndmask_b32_e32 v31, v31, v47, vcc
	v_cmp_ne_u32_e32 vcc, 0, v31
	v_cndmask_b32_e64 v31, v34, v35, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v28, v28, v31, vcc
	v_cndmask_b32_e32 v29, v29, v33, vcc
	v_xor_b32_e32 v28, v28, v32
	v_xor_b32_e32 v29, v29, v32
	v_sub_co_u32_e32 v34, vcc, v28, v32
	s_nop 1
	v_subb_co_u32_e32 v35, vcc, v29, v32, vcc
.LBB0_61:
	s_andn2_saveexec_b64 s[14:15], s[28:29]
	s_cbranch_execz .LBB0_63
	v_cvt_f32_u32_e32 v29, s30
	s_sub_i32 s26, 0, s30
	v_rcp_iflag_f32_e32 v29, v29
	s_nop 0
	v_mul_f32_e32 v29, 0x4f7ffffe, v29
	v_cvt_u32_f32_e32 v29, v29
	v_mul_lo_u32 v31, s26, v29
	v_mul_hi_u32 v31, v29, v31
	v_add_u32_e32 v29, v29, v31
	v_mul_hi_u32 v29, v28, v29
	v_mul_lo_u32 v29, v29, s30
	v_sub_u32_e32 v28, v28, v29
	v_subrev_u32_e32 v29, s30, v28
	v_cmp_le_u32_e32 vcc, s30, v28
	s_nop 1
	v_cndmask_b32_e32 v28, v28, v29, vcc
	v_subrev_u32_e32 v29, s30, v28
	v_cmp_le_u32_e32 vcc, s30, v28
	s_nop 1
	v_cndmask_b32_e32 v104, v28, v29, vcc
	v_mov_b64_e32 v[34:35], v[104:105]
.LBB0_63:
	s_or_b64 exec, exec, s[14:15]
	v_ashrrev_i32_e32 v28, 31, v35
	v_and_b32_e32 v29, s31, v28
	v_and_b32_e32 v28, s30, v28
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[34:35]
	v_ashrrev_i32_e32 v31, 31, v29
	v_xor_b32_e32 v29, v31, v29
	v_xor_b32_e32 v30, v30, v32
	v_ashrrev_i32_e32 v32, 31, v29
	v_xor_b32_e32 v28, v31, v28
	v_lshrrev_b32_e32 v104, 27, v32
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[104:105]
	v_alignbit_b32 v28, v29, v28, 5
	v_mul_lo_u32 v29, v30, s46
	s_mov_b32 m0, s97
	v_xad_u32 v28, v28, v31, v29
	buffer_load_dword v28, s[4:7], 0 offen lds
	v_lshlrev_b16_e32 v28, 8, v95
	v_lshlrev_b16_e32 v29, 8, v109
	v_or_b32_e32 v28, v91, v28
	v_or_b32_sdwa v29, v93, v29 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v32, 8, v55
	v_or_b32_sdwa v31, v28, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v28, 8, v87
	v_lshlrev_b16_e32 v29, 8, v89
	v_or_b32_e32 v28, v83, v28
	v_or_b32_sdwa v29, v85, v29 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v32, v51, v32 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v30, v28, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v28, 8, v77
	v_lshlrev_b16_e32 v29, 8, v79
	v_or_b32_e32 v28, v57, v28
	v_or_b32_sdwa v29, v65, v29 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt vmcnt(7)
	v_lshlrev_b16_e32 v33, 8, v147
	v_or_b32_sdwa v29, v28, v29 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v28, 8, v53
	v_or_b32_e32 v28, v49, v28
	v_or_b32_sdwa v28, v28, v32 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v32, 8, v146
	v_or_b32_e32 v32, v144, v32
	v_or_b32_sdwa v33, v145, v33 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v47, 8, v136
	v_or_b32_sdwa v35, v32, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v32, 8, v142
	v_lshlrev_b16_e32 v33, 8, v143
	v_or_b32_e32 v32, v81, v32
	v_or_b32_sdwa v33, v141, v33 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v47, v134, v47 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v34, v32, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v32, 8, v139
	v_lshlrev_b16_e32 v33, 8, v140
	v_or_b32_e32 v32, v137, v32
	v_or_b32_sdwa v33, v138, v33 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_nop 0
	v_or_b32_sdwa v33, v32, v33 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v32, 8, v135
	v_or_b32_e32 v32, v133, v32
	v_or_b32_sdwa v32, v32, v47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_waitcnt vmcnt(5) lgkmcnt(0)
	s_barrier
	s_mov_b64 s[14:15], 0x1000
	v_lshl_add_u64 v[130:131], v[128:129], 0, s[14:15]
	s_movk_i32 s14, 0xefff
	v_sub_co_u32_e64 v47, s[14:15], s14, v128
	s_mov_b32 s71, s70
	s_nop 0
	v_subb_co_u32_e64 v51, s[14:15], -1, v129, s[14:15]
	s_add_u32 s14, s44, s70
	s_addc_u32 s15, s45, s70
	v_cmp_gt_i32_e32 vcc, 0, v131
	s_xor_b64 s[62:63], s[14:15], s[70:71]
	s_sub_u32 s14, 0, s62
	v_cndmask_b32_e32 v133, v131, v51, vcc
	v_cndmask_b32_e32 v132, v130, v47, vcc
	v_cvt_f32_u32_e32 v47, s62
	v_cvt_f32_u32_e32 v51, s63
	s_subb_u32 s15, 0, s63
	v_ashrrev_i32_e32 v134, 31, v133
	v_mov_b32_e32 v135, v134
	v_fmac_f32_e32 v47, 0x4f800000, v51
	v_rcp_f32_e32 v47, v47
	v_lshl_add_u64 v[132:133], v[132:133], 0, v[134:135]
	v_mov_b32_e32 v81, s63
	v_and_b32_e32 v49, 0xffff, v148
	v_mul_f32_e32 v47, 0x5f7ffffc, v47
	v_mul_f32_e32 v51, 0x2f800000, v47
	v_trunc_f32_e32 v51, v51
	v_fmac_f32_e32 v47, 0xcf800000, v51
	v_cvt_u32_f32_e32 v47, v47
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[178:181], v[28:31], v[4:7], v122, v49 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_and_b32_e32 v117, 0xffff, v117
	v_readfirstlane_b32 s26, v47
	v_cvt_u32_f32_e32 v47, v51
	s_mul_hi_u32 s28, s14, s26
	s_mul_i32 s27, s15, s26
	s_mul_i32 s35, s14, s26
	v_readfirstlane_b32 s29, v47
	s_mul_i32 s34, s14, s29
	s_add_i32 s28, s28, s34
	s_add_i32 s28, s28, s27
	s_mul_i32 s34, s26, s28
	s_mul_hi_u32 s48, s26, s35
	s_mul_hi_u32 s27, s26, s28
	s_add_u32 s34, s48, s34
	s_addc_u32 s27, 0, s27
	s_mul_hi_u32 s49, s29, s35
	s_mul_i32 s35, s29, s35
	s_add_u32 s34, s34, s35
	s_mul_hi_u32 s48, s29, s28
	s_addc_u32 s27, s27, s49
	s_addc_u32 s34, s48, 0
	s_mul_i32 s28, s29, s28
	s_add_u32 s27, s27, s28
	s_addc_u32 s28, 0, s34
	s_add_u32 s26, s26, s27
	s_addc_u32 s27, s29, s28
	s_mul_i32 s28, s14, s27
	s_mul_hi_u32 s29, s14, s26
	s_add_i32 s28, s29, s28
	s_mul_i32 s15, s15, s26
	s_add_i32 s28, s28, s15
	s_mul_i32 s14, s14, s26
	s_mul_hi_u32 s29, s27, s14
	s_mul_i32 s34, s27, s14
	s_mul_i32 s48, s26, s28
	s_mul_hi_u32 s14, s26, s14
	s_mul_hi_u32 s35, s26, s28
	s_add_u32 s14, s14, s48
	s_addc_u32 s35, 0, s35
	s_add_u32 s14, s14, s34
	s_mul_hi_u32 s15, s27, s28
	s_addc_u32 s14, s35, s29
	s_addc_u32 s15, s15, 0
	s_mul_i32 s28, s27, s28
	s_add_u32 s14, s14, s28
	s_addc_u32 s15, 0, s15
	s_add_u32 s34, s26, s14
	s_addc_u32 s35, s27, s15
	v_xor_b32_e32 v51, v132, v134
	v_xor_b32_e32 v47, v133, v134
	v_mad_u64_u32 v[132:133], s[14:15], v51, s35, 0
	v_mul_hi_u32 v104, v51, s34
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[132:133]
	v_mad_u64_u32 v[138:139], s[14:15], v47, s34, 0
	v_add_co_u32_e32 v53, vcc, v132, v138
	v_mad_u64_u32 v[136:137], s[14:15], v47, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v133, v139, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[162:165], v[28:31], v[0:3], v122, v49 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[136:137]
	v_mul_lo_u32 v53, s63, v132
	v_mul_lo_u32 v55, s62, v133
	v_mad_u64_u32 v[136:137], s[14:15], s62, v132, 0
	v_add3_u32 v53, v137, v55, v53
	v_sub_co_u32_e32 v51, vcc, v51, v136
	v_cmp_le_u32_e64 s[26:27], s62, v51
	s_nop 0
	v_subb_co_u32_e64 v55, s[14:15], v47, v53, vcc
	v_sub_u32_e32 v47, v47, v53
	v_subb_co_u32_e32 v47, vcc, v47, v81, vcc
	v_subrev_co_u32_e32 v51, vcc, s62, v51
	v_cmp_le_u32_e64 s[14:15], s63, v55
	s_nop 0
	v_subbrev_co_u32_e32 v47, vcc, 0, v47, vcc
	v_cndmask_b32_e64 v57, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v55
	v_cndmask_b32_e64 v55, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v47
	v_cmp_le_u32_e64 s[26:27], s62, v51
	v_cndmask_b32_e64 v55, v57, v55, s[14:15]
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v47
	v_cndmask_b32_e64 v47, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v55
	v_cndmask_b32_e32 v47, v53, v47, vcc
	v_cmp_ne_u32_e32 vcc, 0, v47
	v_xor_b32_e32 v51, s70, v134
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[174:177], v[32:35], v[4:7], v122, v117 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e64 v47, 1, 2, vcc
	v_add_u32_e32 v47, v132, v47
	v_cndmask_b32_e64 v47, v132, v47, s[14:15]
	v_ashrrev_i32_e32 v132, 31, v131
	v_mov_b32_e32 v133, v132
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[132:133]
	v_xor_b32_e32 v47, v47, v51
	v_xor_b32_e32 v53, v130, v132
	v_sub_u32_e32 v47, v47, v51
	v_xor_b32_e32 v51, v131, v132
	v_mad_u64_u32 v[130:131], s[14:15], v53, s35, 0
	v_mul_hi_u32 v104, v53, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[136:137], s[14:15], v51, s34, 0
	v_add_co_u32_e32 v55, vcc, v130, v136
	v_mad_u64_u32 v[134:135], s[14:15], v51, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v131, v137, vcc
	v_xad_u32 v47, v47, v132, v46
	s_nop 0
	v_addc_co_u32_e32 v135, vcc, 0, v135, vcc
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[134:135]
	v_mul_lo_u32 v55, s63, v130
	v_mul_lo_u32 v57, s62, v131
	v_mad_u64_u32 v[130:131], s[14:15], s62, v130, 0
	v_add3_u32 v55, v131, v57, v55
	v_sub_u32_e32 v57, v51, v55
	v_sub_co_u32_e32 v53, vcc, v53, v130
	v_mul_lo_u32 v47, v47, s33
	s_nop 0
	v_subb_co_u32_e64 v57, s[14:15], v57, v81, vcc
	v_subrev_co_u32_e64 v65, s[14:15], s62, v53
	v_cmp_le_u32_e64 s[28:29], s62, v65
	s_nop 0
	v_subbrev_co_u32_e64 v77, s[26:27], 0, v57, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v77
	v_cndmask_b32_e64 v83, 0, -1, s[28:29]
	v_subb_co_u32_e64 v57, s[14:15], v57, v81, s[14:15]
	v_cndmask_b32_e64 v79, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v77
	v_subb_co_u32_e32 v51, vcc, v51, v55, vcc
	s_nop 0
	v_cndmask_b32_e64 v79, v79, v83, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v79
	v_subrev_co_u32_e64 v79, s[14:15], s62, v65
	v_cmp_le_u32_e32 vcc, s63, v51
	s_nop 0
	v_subbrev_co_u32_e64 v57, s[14:15], 0, v57, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v53
	v_cndmask_b32_e64 v57, v77, v57, s[26:27]
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v51
	v_cndmask_b32_e64 v77, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x801
	v_cndmask_b32_e32 v55, v55, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v55
	v_cndmask_b32_e64 v55, v65, v79, s[26:27]
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[90:91]
	v_cndmask_b32_e32 v53, v53, v55, vcc
	v_cndmask_b32_e32 v51, v51, v57, vcc
	v_xor_b32_e32 v53, v53, v132
	v_xor_b32_e32 v51, v51, v132
	v_sub_co_u32_e32 v53, vcc, v53, v132
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[170:173], v[32:35], v[0:3], v122, v117 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_subb_co_u32_e32 v51, vcc, v51, v132, vcc
	v_ashrrev_i32_e32 v51, 31, v51
	v_and_b32_e32 v51, s44, v51
	v_lshl_add_u64 v[132:133], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xf7fe
	v_add3_u32 v47, v51, v53, v47
	v_sub_co_u32_e64 v51, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -1, v[130:131]
	s_nop 0
	v_subb_co_u32_e64 v53, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v135, v133, v53, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v132, v51, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v53, v134, v136
	v_xor_b32_e32 v51, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v53, s35, 0
	v_mul_hi_u32 v104, v53, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v51, s34, 0
	v_add_co_u32_e32 v55, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v51, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v135, v141, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[166:169], v[28:31], v[36:39], v123, v49 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[138:139]
	v_mul_lo_u32 v55, s63, v134
	v_mul_lo_u32 v57, s62, v135
	v_mad_u64_u32 v[138:139], s[14:15], s62, v134, 0
	v_add3_u32 v55, v139, v57, v55
	v_sub_co_u32_e32 v53, vcc, v53, v138
	v_cmp_le_u32_e64 s[26:27], s62, v53
	s_nop 0
	v_subb_co_u32_e64 v57, s[14:15], v51, v55, vcc
	v_sub_u32_e32 v51, v51, v55
	v_subb_co_u32_e32 v51, vcc, v51, v81, vcc
	v_subrev_co_u32_e32 v53, vcc, s62, v53
	v_cmp_le_u32_e64 s[14:15], s63, v57
	s_nop 0
	v_subbrev_co_u32_e32 v51, vcc, 0, v51, vcc
	v_cndmask_b32_e64 v65, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v57
	v_cndmask_b32_e64 v57, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v51
	v_cmp_le_u32_e64 s[26:27], s62, v53
	v_cndmask_b32_e64 v57, v65, v57, s[14:15]
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v51
	v_cndmask_b32_e64 v51, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v57
	v_cndmask_b32_e32 v51, v55, v51, vcc
	v_cmp_ne_u32_e32 vcc, 0, v51
	v_xor_b32_e32 v53, s70, v136
	buffer_load_ubyte v47, v47, s[36:39], 0 offen
	v_cndmask_b32_e64 v51, 1, 2, vcc
	v_add_u32_e32 v51, v134, v51
	v_cndmask_b32_e64 v51, v134, v51, s[14:15]
	v_ashrrev_i32_e32 v134, 31, v133
	v_mov_b32_e32 v135, v134
	v_lshl_add_u64 v[132:133], v[132:133], 0, v[134:135]
	v_xor_b32_e32 v51, v51, v53
	v_xor_b32_e32 v55, v132, v134
	v_sub_u32_e32 v51, v51, v53
	v_xor_b32_e32 v53, v133, v134
	v_mad_u64_u32 v[132:133], s[14:15], v55, s35, 0
	v_mul_hi_u32 v104, v55, s34
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[132:133]
	v_mad_u64_u32 v[138:139], s[14:15], v53, s34, 0
	v_add_co_u32_e32 v57, vcc, v132, v138
	v_mad_u64_u32 v[136:137], s[14:15], v53, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v133, v139, vcc
	v_xad_u32 v51, v51, v134, v46
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[136:137]
	v_mul_lo_u32 v57, s63, v132
	v_mul_lo_u32 v65, s62, v133
	v_mad_u64_u32 v[132:133], s[14:15], s62, v132, 0
	v_add3_u32 v57, v133, v65, v57
	v_sub_u32_e32 v65, v53, v57
	v_sub_co_u32_e32 v55, vcc, v55, v132
	v_mul_lo_u32 v51, v51, s33
	s_nop 0
	v_subb_co_u32_e64 v65, s[14:15], v65, v81, vcc
	v_subrev_co_u32_e64 v77, s[14:15], s62, v55
	v_cmp_le_u32_e64 s[28:29], s62, v77
	s_nop 0
	v_subbrev_co_u32_e64 v79, s[26:27], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v79
	v_cndmask_b32_e64 v85, 0, -1, s[28:29]
	v_subb_co_u32_e64 v65, s[14:15], v65, v81, s[14:15]
	v_cndmask_b32_e64 v83, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v79
	v_subb_co_u32_e32 v53, vcc, v53, v57, vcc
	s_nop 0
	v_cndmask_b32_e64 v83, v83, v85, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v83
	v_subrev_co_u32_e64 v83, s[14:15], s62, v77
	v_cmp_le_u32_e32 vcc, s63, v53
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v55
	v_cndmask_b32_e64 v65, v79, v65, s[26:27]
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v53
	v_cndmask_b32_e64 v79, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x802
	v_cndmask_b32_e32 v57, v57, v79, vcc
	v_cmp_ne_u32_e32 vcc, 0, v57
	v_cndmask_b32_e64 v57, v77, v83, s[26:27]
	v_lshl_add_u64 v[132:133], v[126:127], 0, s[14:15]
	v_cndmask_b32_e32 v55, v55, v57, vcc
	v_cndmask_b32_e32 v53, v53, v65, vcc
	v_xor_b32_e32 v55, v55, v134
	v_xor_b32_e32 v53, v53, v134
	v_sub_co_u32_e32 v55, vcc, v55, v134
	s_movk_i32 s14, 0xf7fd
	s_nop 0
	v_subb_co_u32_e32 v53, vcc, v53, v134, vcc
	v_ashrrev_i32_e32 v53, 31, v53
	v_and_b32_e32 v53, s44, v53
	v_add3_u32 v51, v53, v55, v51
	v_sub_co_u32_e64 v53, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -2, v[130:131]
	s_nop 0
	v_subb_co_u32_e64 v55, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v135, v133, v55, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v132, v53, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v55, v134, v136
	v_xor_b32_e32 v53, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v55, s35, 0
	v_mul_hi_u32 v104, v55, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v53, s34, 0
	v_add_co_u32_e32 v57, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v53, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v135, v141, vcc
	buffer_load_ubyte v51, v51, s[36:39], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[138:139]
	v_mul_lo_u32 v57, s63, v134
	v_mul_lo_u32 v65, s62, v135
	v_mad_u64_u32 v[138:139], s[14:15], s62, v134, 0
	v_add3_u32 v57, v139, v65, v57
	v_sub_co_u32_e32 v55, vcc, v55, v138
	v_cmp_le_u32_e64 s[26:27], s62, v55
	s_nop 0
	v_subb_co_u32_e64 v65, s[14:15], v53, v57, vcc
	v_sub_u32_e32 v53, v53, v57
	v_subb_co_u32_e32 v53, vcc, v53, v81, vcc
	v_subrev_co_u32_e32 v55, vcc, s62, v55
	v_cmp_le_u32_e64 s[14:15], s63, v65
	s_nop 0
	v_subbrev_co_u32_e32 v53, vcc, 0, v53, vcc
	v_cndmask_b32_e64 v77, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v65
	v_cndmask_b32_e64 v65, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v53
	v_cmp_le_u32_e64 s[26:27], s62, v55
	v_cndmask_b32_e64 v65, v77, v65, s[14:15]
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v53
	v_cndmask_b32_e64 v53, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v65
	v_cndmask_b32_e32 v53, v57, v53, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_xor_b32_e32 v55, s70, v136
	ds_read_b128 v[162:165], v78 offset:24576
	v_cndmask_b32_e64 v53, 1, 2, vcc
	v_add_u32_e32 v53, v134, v53
	v_cndmask_b32_e64 v53, v134, v53, s[14:15]
	v_ashrrev_i32_e32 v134, 31, v133
	v_mov_b32_e32 v135, v134
	v_lshl_add_u64 v[132:133], v[132:133], 0, v[134:135]
	v_xor_b32_e32 v53, v53, v55
	v_xor_b32_e32 v57, v132, v134
	v_sub_u32_e32 v53, v53, v55
	v_xor_b32_e32 v55, v133, v134
	v_mad_u64_u32 v[132:133], s[14:15], v57, s35, 0
	v_mul_hi_u32 v104, v57, s34
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[132:133]
	v_mad_u64_u32 v[138:139], s[14:15], v55, s34, 0
	v_add_co_u32_e32 v65, vcc, v132, v138
	v_mad_u64_u32 v[136:137], s[14:15], v55, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v133, v139, vcc
	v_xad_u32 v53, v53, v134, v46
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[136:137]
	v_mul_lo_u32 v65, s63, v132
	v_mul_lo_u32 v77, s62, v133
	v_mad_u64_u32 v[132:133], s[14:15], s62, v132, 0
	v_add3_u32 v65, v133, v77, v65
	v_sub_u32_e32 v77, v55, v65
	v_sub_co_u32_e32 v57, vcc, v57, v132
	v_mul_lo_u32 v53, v53, s33
	s_nop 0
	v_subb_co_u32_e64 v77, s[14:15], v77, v81, vcc
	v_subrev_co_u32_e64 v79, s[14:15], s62, v57
	v_cmp_le_u32_e64 s[28:29], s62, v79
	s_nop 0
	v_subbrev_co_u32_e64 v83, s[26:27], 0, v77, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v83
	v_cndmask_b32_e64 v87, 0, -1, s[28:29]
	v_subb_co_u32_e64 v77, s[14:15], v77, v81, s[14:15]
	v_cndmask_b32_e64 v85, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v83
	v_subb_co_u32_e32 v55, vcc, v55, v65, vcc
	s_nop 0
	v_cndmask_b32_e64 v85, v85, v87, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v85
	v_subrev_co_u32_e64 v85, s[14:15], s62, v79
	v_cmp_le_u32_e32 vcc, s63, v55
	s_nop 0
	v_subbrev_co_u32_e64 v77, s[14:15], 0, v77, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v57
	v_cndmask_b32_e64 v77, v83, v77, s[26:27]
	v_cndmask_b32_e64 v65, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v55
	v_cndmask_b32_e64 v83, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x803
	v_cndmask_b32_e32 v65, v65, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v65
	v_cndmask_b32_e64 v65, v79, v85, s[26:27]
	v_lshl_add_u64 v[132:133], v[126:127], 0, s[14:15]
	v_cndmask_b32_e32 v57, v57, v65, vcc
	v_cndmask_b32_e32 v55, v55, v77, vcc
	v_xor_b32_e32 v57, v57, v134
	v_xor_b32_e32 v55, v55, v134
	v_sub_co_u32_e32 v57, vcc, v57, v134
	s_movk_i32 s14, 0xf7fc
	s_nop 0
	v_subb_co_u32_e32 v55, vcc, v55, v134, vcc
	v_ashrrev_i32_e32 v55, 31, v55
	v_and_b32_e32 v55, s44, v55
	v_add3_u32 v53, v55, v57, v53
	v_sub_co_u32_e64 v55, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -3, v[130:131]
	s_nop 0
	v_subb_co_u32_e64 v57, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v135, v133, v57, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v132, v55, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v57, v134, v136
	v_xor_b32_e32 v55, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v57, s35, 0
	v_mul_hi_u32 v104, v57, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v55, s34, 0
	v_add_co_u32_e32 v65, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v55, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v135, v141, vcc
	buffer_load_ubyte v53, v53, s[36:39], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[138:139]
	v_mul_lo_u32 v65, s63, v134
	v_mul_lo_u32 v77, s62, v135
	v_mad_u64_u32 v[138:139], s[14:15], s62, v134, 0
	v_add3_u32 v65, v139, v77, v65
	v_sub_co_u32_e32 v57, vcc, v57, v138
	v_cmp_le_u32_e64 s[26:27], s62, v57
	s_nop 0
	v_subb_co_u32_e64 v77, s[14:15], v55, v65, vcc
	v_sub_u32_e32 v55, v55, v65
	v_subb_co_u32_e32 v55, vcc, v55, v81, vcc
	v_subrev_co_u32_e32 v57, vcc, s62, v57
	v_cmp_le_u32_e64 s[14:15], s63, v77
	s_nop 0
	v_subbrev_co_u32_e32 v55, vcc, 0, v55, vcc
	v_cndmask_b32_e64 v79, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v77
	v_cndmask_b32_e64 v77, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v55
	v_cmp_le_u32_e64 s[26:27], s62, v57
	v_cndmask_b32_e64 v77, v79, v77, s[14:15]
	v_cndmask_b32_e64 v65, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v55
	v_cndmask_b32_e64 v55, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v77
	v_cndmask_b32_e32 v55, v65, v55, vcc
	v_cmp_ne_u32_e32 vcc, 0, v55
	v_xor_b32_e32 v57, s70, v136
	s_nop 0
	v_cndmask_b32_e64 v55, 1, 2, vcc
	v_add_u32_e32 v55, v134, v55
	v_cndmask_b32_e64 v55, v134, v55, s[14:15]
	v_ashrrev_i32_e32 v134, 31, v133
	v_mov_b32_e32 v135, v134
	v_lshl_add_u64 v[132:133], v[132:133], 0, v[134:135]
	v_xor_b32_e32 v55, v55, v57
	v_xor_b32_e32 v65, v132, v134
	v_sub_u32_e32 v55, v55, v57
	v_xor_b32_e32 v57, v133, v134
	v_mad_u64_u32 v[132:133], s[14:15], v65, s35, 0
	v_mul_hi_u32 v104, v65, s34
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[132:133]
	v_mad_u64_u32 v[138:139], s[14:15], v57, s34, 0
	v_add_co_u32_e32 v77, vcc, v132, v138
	v_mad_u64_u32 v[136:137], s[14:15], v57, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v133, v139, vcc
	v_xad_u32 v55, v55, v134, v46
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[136:137]
	v_mul_lo_u32 v77, s63, v132
	v_mul_lo_u32 v79, s62, v133
	v_mad_u64_u32 v[132:133], s[14:15], s62, v132, 0
	v_add3_u32 v77, v133, v79, v77
	v_sub_u32_e32 v79, v57, v77
	v_sub_co_u32_e32 v65, vcc, v65, v132
	v_mul_lo_u32 v55, v55, s33
	s_nop 0
	v_subb_co_u32_e64 v79, s[14:15], v79, v81, vcc
	v_subrev_co_u32_e64 v83, s[14:15], s62, v65
	v_cmp_le_u32_e64 s[28:29], s62, v83
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[26:27], 0, v79, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v85
	v_cndmask_b32_e64 v89, 0, -1, s[28:29]
	v_subb_co_u32_e64 v79, s[14:15], v79, v81, s[14:15]
	v_cndmask_b32_e64 v87, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v85
	v_subb_co_u32_e32 v57, vcc, v57, v77, vcc
	s_nop 0
	v_cndmask_b32_e64 v87, v87, v89, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v87
	v_subrev_co_u32_e64 v87, s[14:15], s62, v83
	v_cmp_le_u32_e32 vcc, s63, v57
	s_nop 0
	v_subbrev_co_u32_e64 v79, s[14:15], 0, v79, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v65
	v_cndmask_b32_e64 v79, v85, v79, s[26:27]
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v57
	v_cndmask_b32_e64 v85, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x804
	v_cndmask_b32_e32 v77, v77, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v77
	v_cndmask_b32_e64 v77, v83, v87, s[26:27]
	v_lshl_add_u64 v[132:133], v[126:127], 0, s[14:15]
	v_cndmask_b32_e32 v65, v65, v77, vcc
	v_cndmask_b32_e32 v57, v57, v79, vcc
	v_xor_b32_e32 v65, v65, v134
	v_xor_b32_e32 v57, v57, v134
	v_sub_co_u32_e32 v65, vcc, v65, v134
	s_movk_i32 s14, 0xf7fb
	s_nop 0
	v_subb_co_u32_e32 v57, vcc, v57, v134, vcc
	v_ashrrev_i32_e32 v57, 31, v57
	v_and_b32_e32 v57, s44, v57
	v_add3_u32 v55, v57, v65, v55
	v_sub_co_u32_e64 v57, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -4, v[130:131]
	s_nop 0
	v_subb_co_u32_e64 v65, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v135, v133, v65, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v132, v57, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v65, v134, v136
	v_xor_b32_e32 v57, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v65, s35, 0
	v_mul_hi_u32 v104, v65, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v57, s34, 0
	v_add_co_u32_e32 v77, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v57, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v135, v141, vcc
	buffer_load_ubyte v55, v55, s[36:39], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[138:139]
	v_mul_lo_u32 v77, s63, v134
	v_mul_lo_u32 v79, s62, v135
	v_mad_u64_u32 v[138:139], s[14:15], s62, v134, 0
	v_add3_u32 v77, v139, v79, v77
	v_sub_co_u32_e32 v65, vcc, v65, v138
	v_cmp_le_u32_e64 s[26:27], s62, v65
	s_nop 0
	v_subb_co_u32_e64 v79, s[14:15], v57, v77, vcc
	v_sub_u32_e32 v57, v57, v77
	v_subb_co_u32_e32 v57, vcc, v57, v81, vcc
	v_subrev_co_u32_e32 v65, vcc, s62, v65
	v_cmp_le_u32_e64 s[14:15], s63, v79
	s_nop 0
	v_subbrev_co_u32_e32 v57, vcc, 0, v57, vcc
	v_cndmask_b32_e64 v83, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v79
	v_cndmask_b32_e64 v79, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v57
	v_cmp_le_u32_e64 s[26:27], s62, v65
	v_cndmask_b32_e64 v79, v83, v79, s[14:15]
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v57
	v_cndmask_b32_e64 v57, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v79
	v_cndmask_b32_e32 v57, v77, v57, vcc
	v_cmp_ne_u32_e32 vcc, 0, v57
	v_xor_b32_e32 v65, s70, v136
	s_nop 0
	v_cndmask_b32_e64 v57, 1, 2, vcc
	v_add_u32_e32 v57, v134, v57
	v_cndmask_b32_e64 v57, v134, v57, s[14:15]
	v_ashrrev_i32_e32 v134, 31, v133
	v_mov_b32_e32 v135, v134
	v_lshl_add_u64 v[132:133], v[132:133], 0, v[134:135]
	v_xor_b32_e32 v57, v57, v65
	v_xor_b32_e32 v77, v132, v134
	v_sub_u32_e32 v57, v57, v65
	v_xor_b32_e32 v65, v133, v134
	v_mad_u64_u32 v[132:133], s[14:15], v77, s35, 0
	v_mul_hi_u32 v104, v77, s34
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[132:133]
	v_mad_u64_u32 v[138:139], s[14:15], v65, s34, 0
	v_add_co_u32_e32 v79, vcc, v132, v138
	v_mad_u64_u32 v[136:137], s[14:15], v65, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v133, v139, vcc
	v_xad_u32 v57, v57, v134, v46
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[136:137]
	v_mul_lo_u32 v79, s63, v132
	v_mul_lo_u32 v83, s62, v133
	v_mad_u64_u32 v[132:133], s[14:15], s62, v132, 0
	v_add3_u32 v79, v133, v83, v79
	v_sub_u32_e32 v83, v65, v79
	v_sub_co_u32_e32 v77, vcc, v77, v132
	v_mul_lo_u32 v57, v57, s33
	s_nop 0
	v_subb_co_u32_e64 v83, s[14:15], v83, v81, vcc
	v_subrev_co_u32_e64 v85, s[14:15], s62, v77
	v_cmp_le_u32_e64 s[28:29], s62, v85
	s_nop 0
	v_subbrev_co_u32_e64 v87, s[26:27], 0, v83, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v87
	v_cndmask_b32_e64 v91, 0, -1, s[28:29]
	v_subb_co_u32_e64 v83, s[14:15], v83, v81, s[14:15]
	v_cndmask_b32_e64 v89, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v87
	v_subb_co_u32_e32 v65, vcc, v65, v79, vcc
	s_nop 0
	v_cndmask_b32_e64 v89, v89, v91, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v89
	v_subrev_co_u32_e64 v89, s[14:15], s62, v85
	v_cmp_le_u32_e32 vcc, s63, v65
	s_nop 0
	v_subbrev_co_u32_e64 v83, s[14:15], 0, v83, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v77
	v_cndmask_b32_e64 v83, v87, v83, s[26:27]
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v65
	v_cndmask_b32_e64 v87, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x805
	v_cndmask_b32_e32 v79, v79, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v79
	v_cndmask_b32_e64 v79, v85, v89, s[26:27]
	v_lshl_add_u64 v[132:133], v[126:127], 0, s[14:15]
	v_cndmask_b32_e32 v77, v77, v79, vcc
	v_cndmask_b32_e32 v65, v65, v83, vcc
	v_xor_b32_e32 v77, v77, v134
	v_xor_b32_e32 v65, v65, v134
	v_sub_co_u32_e32 v77, vcc, v77, v134
	s_movk_i32 s14, 0xf7fa
	s_nop 0
	v_subb_co_u32_e32 v65, vcc, v65, v134, vcc
	v_ashrrev_i32_e32 v65, 31, v65
	v_and_b32_e32 v65, s44, v65
	v_add3_u32 v57, v65, v77, v57
	v_sub_co_u32_e64 v65, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -5, v[130:131]
	s_nop 0
	v_subb_co_u32_e64 v77, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v135, v133, v77, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v132, v65, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v77, v134, v136
	v_xor_b32_e32 v65, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v77, s35, 0
	v_mul_hi_u32 v104, v77, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v65, s34, 0
	v_add_co_u32_e32 v79, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v65, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v135, v141, vcc
	buffer_load_ubyte v57, v57, s[36:39], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[138:139]
	v_mul_lo_u32 v79, s63, v134
	v_mul_lo_u32 v83, s62, v135
	v_mad_u64_u32 v[138:139], s[14:15], s62, v134, 0
	v_add3_u32 v79, v139, v83, v79
	v_sub_co_u32_e32 v77, vcc, v77, v138
	v_cmp_le_u32_e64 s[26:27], s62, v77
	s_nop 0
	v_subb_co_u32_e64 v83, s[14:15], v65, v79, vcc
	v_sub_u32_e32 v65, v65, v79
	v_subb_co_u32_e32 v65, vcc, v65, v81, vcc
	v_subrev_co_u32_e32 v77, vcc, s62, v77
	v_cmp_le_u32_e64 s[14:15], s63, v83
	s_nop 0
	v_subbrev_co_u32_e32 v65, vcc, 0, v65, vcc
	v_cndmask_b32_e64 v85, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v83
	v_cndmask_b32_e64 v83, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v65
	v_cmp_le_u32_e64 s[26:27], s62, v77
	v_cndmask_b32_e64 v83, v85, v83, s[14:15]
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v65
	v_cndmask_b32_e64 v65, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v83
	v_cndmask_b32_e32 v65, v79, v65, vcc
	v_cmp_ne_u32_e32 vcc, 0, v65
	v_xor_b32_e32 v77, s70, v136
	s_nop 0
	v_cndmask_b32_e64 v65, 1, 2, vcc
	v_add_u32_e32 v65, v134, v65
	v_cndmask_b32_e64 v65, v134, v65, s[14:15]
	v_ashrrev_i32_e32 v134, 31, v133
	v_mov_b32_e32 v135, v134
	v_lshl_add_u64 v[132:133], v[132:133], 0, v[134:135]
	v_xor_b32_e32 v65, v65, v77
	v_xor_b32_e32 v79, v132, v134
	v_sub_u32_e32 v65, v65, v77
	v_xor_b32_e32 v77, v133, v134
	v_mad_u64_u32 v[132:133], s[14:15], v79, s35, 0
	v_mul_hi_u32 v104, v79, s34
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[132:133]
	v_mad_u64_u32 v[138:139], s[14:15], v77, s34, 0
	v_add_co_u32_e32 v83, vcc, v132, v138
	v_mad_u64_u32 v[136:137], s[14:15], v77, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v133, v139, vcc
	v_xad_u32 v65, v65, v134, v46
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[136:137]
	v_mul_lo_u32 v83, s63, v132
	v_mul_lo_u32 v85, s62, v133
	v_mad_u64_u32 v[132:133], s[14:15], s62, v132, 0
	v_add3_u32 v83, v133, v85, v83
	v_sub_u32_e32 v85, v77, v83
	v_sub_co_u32_e32 v79, vcc, v79, v132
	v_mul_lo_u32 v65, v65, s33
	s_nop 0
	v_subb_co_u32_e64 v85, s[14:15], v85, v81, vcc
	v_subrev_co_u32_e64 v87, s[14:15], s62, v79
	v_cmp_le_u32_e64 s[28:29], s62, v87
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[26:27], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v89
	v_cndmask_b32_e64 v93, 0, -1, s[28:29]
	v_subb_co_u32_e64 v85, s[14:15], v85, v81, s[14:15]
	v_cndmask_b32_e64 v91, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v89
	v_subb_co_u32_e32 v77, vcc, v77, v83, vcc
	s_nop 0
	v_cndmask_b32_e64 v91, v91, v93, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v91
	v_subrev_co_u32_e64 v91, s[14:15], s62, v87
	v_cmp_le_u32_e32 vcc, s63, v77
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v85, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v79
	v_cndmask_b32_e64 v85, v89, v85, s[26:27]
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v77
	v_cndmask_b32_e64 v89, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x806
	v_cndmask_b32_e32 v83, v83, v89, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	v_cndmask_b32_e64 v83, v87, v91, s[26:27]
	v_lshl_add_u64 v[132:133], v[126:127], 0, s[14:15]
	v_cndmask_b32_e32 v79, v79, v83, vcc
	v_cndmask_b32_e32 v77, v77, v85, vcc
	v_xor_b32_e32 v79, v79, v134
	v_xor_b32_e32 v77, v77, v134
	v_sub_co_u32_e32 v79, vcc, v79, v134
	s_movk_i32 s14, 0xf7f9
	s_nop 0
	v_subb_co_u32_e32 v77, vcc, v77, v134, vcc
	v_ashrrev_i32_e32 v77, 31, v77
	v_and_b32_e32 v77, s44, v77
	v_add3_u32 v65, v77, v79, v65
	v_sub_co_u32_e64 v77, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -6, v[130:131]
	s_nop 0
	v_subb_co_u32_e64 v79, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v135, v133, v79, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v132, v77, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v79, v134, v136
	v_xor_b32_e32 v77, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v79, s35, 0
	v_mul_hi_u32 v104, v79, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v77, s34, 0
	v_add_co_u32_e32 v83, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v77, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v135, v141, vcc
	buffer_load_ubyte v65, v65, s[36:39], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[138:139]
	v_mul_lo_u32 v83, s63, v134
	v_mul_lo_u32 v85, s62, v135
	v_mad_u64_u32 v[138:139], s[14:15], s62, v134, 0
	v_add3_u32 v83, v139, v85, v83
	v_sub_co_u32_e32 v79, vcc, v79, v138
	v_cmp_le_u32_e64 s[26:27], s62, v79
	s_nop 0
	v_subb_co_u32_e64 v85, s[14:15], v77, v83, vcc
	v_sub_u32_e32 v77, v77, v83
	v_subb_co_u32_e32 v77, vcc, v77, v81, vcc
	v_subrev_co_u32_e32 v79, vcc, s62, v79
	v_cmp_le_u32_e64 s[14:15], s63, v85
	s_nop 0
	v_subbrev_co_u32_e32 v77, vcc, 0, v77, vcc
	v_cndmask_b32_e64 v87, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v85
	v_cndmask_b32_e64 v85, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v77
	v_cmp_le_u32_e64 s[26:27], s62, v79
	v_cndmask_b32_e64 v85, v87, v85, s[14:15]
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v77
	v_cndmask_b32_e64 v77, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v85
	v_cndmask_b32_e32 v77, v83, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v77
	v_xor_b32_e32 v79, s70, v136
	s_nop 0
	v_cndmask_b32_e64 v77, 1, 2, vcc
	v_add_u32_e32 v77, v134, v77
	v_cndmask_b32_e64 v77, v134, v77, s[14:15]
	v_ashrrev_i32_e32 v134, 31, v133
	v_mov_b32_e32 v135, v134
	v_lshl_add_u64 v[132:133], v[132:133], 0, v[134:135]
	v_xor_b32_e32 v77, v77, v79
	v_xor_b32_e32 v83, v132, v134
	v_sub_u32_e32 v77, v77, v79
	v_xor_b32_e32 v79, v133, v134
	v_mad_u64_u32 v[132:133], s[14:15], v83, s35, 0
	v_mul_hi_u32 v104, v83, s34
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[132:133]
	v_mad_u64_u32 v[138:139], s[14:15], v79, s34, 0
	v_add_co_u32_e32 v85, vcc, v132, v138
	v_mad_u64_u32 v[136:137], s[14:15], v79, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v133, v139, vcc
	v_xad_u32 v77, v77, v134, v46
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[136:137]
	v_mul_lo_u32 v85, s63, v132
	v_mul_lo_u32 v87, s62, v133
	v_mad_u64_u32 v[132:133], s[14:15], s62, v132, 0
	v_add3_u32 v85, v133, v87, v85
	v_sub_u32_e32 v87, v79, v85
	v_sub_co_u32_e32 v83, vcc, v83, v132
	v_mul_lo_u32 v77, v77, s33
	s_nop 0
	v_subb_co_u32_e64 v87, s[14:15], v87, v81, vcc
	v_subrev_co_u32_e64 v89, s[14:15], s62, v83
	v_cmp_le_u32_e64 s[28:29], s62, v89
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[26:27], 0, v87, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v91
	v_cndmask_b32_e64 v95, 0, -1, s[28:29]
	v_subb_co_u32_e64 v87, s[14:15], v87, v81, s[14:15]
	v_cndmask_b32_e64 v93, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v91
	v_subb_co_u32_e32 v79, vcc, v79, v85, vcc
	s_nop 0
	v_cndmask_b32_e64 v93, v93, v95, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v93
	v_subrev_co_u32_e64 v93, s[14:15], s62, v89
	v_cmp_le_u32_e32 vcc, s63, v79
	s_nop 0
	v_subbrev_co_u32_e64 v87, s[14:15], 0, v87, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v83
	v_cndmask_b32_e64 v87, v91, v87, s[26:27]
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v79
	v_cndmask_b32_e64 v91, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x807
	v_cndmask_b32_e32 v85, v85, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	v_cndmask_b32_e64 v85, v89, v93, s[26:27]
	v_lshl_add_u64 v[132:133], v[126:127], 0, s[14:15]
	v_cndmask_b32_e32 v83, v83, v85, vcc
	v_cndmask_b32_e32 v79, v79, v87, vcc
	v_xor_b32_e32 v83, v83, v134
	v_xor_b32_e32 v79, v79, v134
	v_sub_co_u32_e32 v83, vcc, v83, v134
	s_movk_i32 s14, 0xf7f8
	s_nop 0
	v_subb_co_u32_e32 v79, vcc, v79, v134, vcc
	v_ashrrev_i32_e32 v79, 31, v79
	v_and_b32_e32 v79, s44, v79
	v_add3_u32 v77, v79, v83, v77
	v_sub_co_u32_e64 v79, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -7, v[130:131]
	s_nop 0
	v_subb_co_u32_e64 v83, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v135, v133, v83, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v132, v79, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v83, v134, v136
	v_xor_b32_e32 v79, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v83, s35, 0
	v_mul_hi_u32 v104, v83, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v79, s34, 0
	v_add_co_u32_e32 v85, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v79, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v135, v141, vcc
	buffer_load_ubyte v77, v77, s[36:39], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[138:139]
	v_mul_lo_u32 v85, s63, v134
	v_mul_lo_u32 v87, s62, v135
	v_mad_u64_u32 v[138:139], s[14:15], s62, v134, 0
	v_add3_u32 v85, v139, v87, v85
	v_sub_co_u32_e32 v83, vcc, v83, v138
	v_cmp_le_u32_e64 s[26:27], s62, v83
	s_nop 0
	v_subb_co_u32_e64 v87, s[14:15], v79, v85, vcc
	v_sub_u32_e32 v79, v79, v85
	v_subb_co_u32_e32 v79, vcc, v79, v81, vcc
	v_subrev_co_u32_e32 v83, vcc, s62, v83
	v_cmp_le_u32_e64 s[14:15], s63, v87
	s_nop 0
	v_subbrev_co_u32_e32 v79, vcc, 0, v79, vcc
	v_cndmask_b32_e64 v89, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v87
	v_cndmask_b32_e64 v87, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v79
	v_cmp_le_u32_e64 s[26:27], s62, v83
	v_cndmask_b32_e64 v87, v89, v87, s[14:15]
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v79
	v_cndmask_b32_e64 v79, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v87
	v_cndmask_b32_e32 v79, v85, v79, vcc
	v_cmp_ne_u32_e32 vcc, 0, v79
	v_xor_b32_e32 v83, s70, v136
	s_nop 0
	v_cndmask_b32_e64 v79, 1, 2, vcc
	v_add_u32_e32 v79, v134, v79
	v_cndmask_b32_e64 v79, v134, v79, s[14:15]
	v_ashrrev_i32_e32 v134, 31, v133
	v_mov_b32_e32 v135, v134
	v_lshl_add_u64 v[132:133], v[132:133], 0, v[134:135]
	v_xor_b32_e32 v79, v79, v83
	v_xor_b32_e32 v85, v132, v134
	v_sub_u32_e32 v79, v79, v83
	v_xor_b32_e32 v83, v133, v134
	v_mad_u64_u32 v[132:133], s[14:15], v85, s35, 0
	v_mul_hi_u32 v104, v85, s34
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[132:133]
	v_mad_u64_u32 v[138:139], s[14:15], v83, s34, 0
	v_add_co_u32_e32 v87, vcc, v132, v138
	v_mad_u64_u32 v[136:137], s[14:15], v83, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v133, v139, vcc
	v_xad_u32 v79, v79, v134, v46
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[136:137]
	v_mul_lo_u32 v87, s63, v132
	v_mul_lo_u32 v89, s62, v133
	v_mad_u64_u32 v[132:133], s[14:15], s62, v132, 0
	v_add3_u32 v87, v133, v89, v87
	v_sub_u32_e32 v89, v83, v87
	v_sub_co_u32_e32 v85, vcc, v85, v132
	v_mul_lo_u32 v79, v79, s33
	s_nop 0
	v_subb_co_u32_e64 v89, s[14:15], v89, v81, vcc
	v_subrev_co_u32_e64 v91, s[14:15], s62, v85
	v_cmp_le_u32_e64 s[28:29], s62, v91
	s_nop 0
	v_subbrev_co_u32_e64 v93, s[26:27], 0, v89, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v93
	v_cndmask_b32_e64 v104, 0, -1, s[28:29]
	v_subb_co_u32_e64 v89, s[14:15], v89, v81, s[14:15]
	v_cndmask_b32_e64 v95, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v93
	v_subb_co_u32_e32 v83, vcc, v83, v87, vcc
	s_nop 0
	v_cndmask_b32_e64 v95, v95, v104, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v95
	v_subrev_co_u32_e64 v95, s[14:15], s62, v91
	v_cmp_le_u32_e32 vcc, s63, v83
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[14:15], 0, v89, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v85
	v_cndmask_b32_e64 v89, v93, v89, s[26:27]
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v83
	v_cndmask_b32_e64 v93, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x808
	v_cndmask_b32_e32 v87, v87, v93, vcc
	v_cmp_ne_u32_e32 vcc, 0, v87
	v_cndmask_b32_e64 v87, v91, v95, s[26:27]
	v_lshl_add_u64 v[132:133], v[126:127], 0, s[14:15]
	v_cndmask_b32_e32 v85, v85, v87, vcc
	v_cndmask_b32_e32 v83, v83, v89, vcc
	v_xor_b32_e32 v85, v85, v134
	v_xor_b32_e32 v83, v83, v134
	v_sub_co_u32_e32 v85, vcc, v85, v134
	s_movk_i32 s14, 0xf7f7
	s_nop 0
	v_subb_co_u32_e32 v83, vcc, v83, v134, vcc
	v_ashrrev_i32_e32 v83, 31, v83
	v_and_b32_e32 v83, s44, v83
	v_add3_u32 v79, v83, v85, v79
	v_sub_co_u32_e64 v83, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -8, v[130:131]
	s_nop 0
	v_subb_co_u32_e64 v85, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v135, v133, v85, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v132, v83, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v85, v134, v136
	v_xor_b32_e32 v83, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v85, s35, 0
	v_mul_hi_u32 v104, v85, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v83, s34, 0
	v_add_co_u32_e32 v87, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v83, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v135, v141, vcc
	buffer_load_ubyte v79, v79, s[36:39], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[138:139]
	v_mul_lo_u32 v87, s63, v134
	v_mul_lo_u32 v89, s62, v135
	v_mad_u64_u32 v[138:139], s[14:15], s62, v134, 0
	v_add3_u32 v87, v139, v89, v87
	v_sub_co_u32_e32 v85, vcc, v85, v138
	v_cmp_le_u32_e64 s[26:27], s62, v85
	s_nop 0
	v_subb_co_u32_e64 v89, s[14:15], v83, v87, vcc
	v_sub_u32_e32 v83, v83, v87
	v_subb_co_u32_e32 v83, vcc, v83, v81, vcc
	v_subrev_co_u32_e32 v85, vcc, s62, v85
	v_cmp_le_u32_e64 s[14:15], s63, v89
	s_nop 0
	v_subbrev_co_u32_e32 v83, vcc, 0, v83, vcc
	v_cndmask_b32_e64 v91, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v89
	v_cndmask_b32_e64 v89, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v83
	v_cmp_le_u32_e64 s[26:27], s62, v85
	v_cndmask_b32_e64 v89, v91, v89, s[14:15]
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v83
	v_cndmask_b32_e64 v83, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v89
	v_cndmask_b32_e32 v83, v87, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	v_xor_b32_e32 v85, s70, v136
	s_nop 0
	v_cndmask_b32_e64 v83, 1, 2, vcc
	v_add_u32_e32 v83, v134, v83
	v_cndmask_b32_e64 v83, v134, v83, s[14:15]
	v_ashrrev_i32_e32 v134, 31, v133
	v_mov_b32_e32 v135, v134
	v_lshl_add_u64 v[132:133], v[132:133], 0, v[134:135]
	v_xor_b32_e32 v83, v83, v85
	v_xor_b32_e32 v87, v132, v134
	v_sub_u32_e32 v83, v83, v85
	v_xor_b32_e32 v85, v133, v134
	v_mad_u64_u32 v[132:133], s[14:15], v87, s35, 0
	v_mul_hi_u32 v104, v87, s34
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[132:133]
	v_mad_u64_u32 v[138:139], s[14:15], v85, s34, 0
	v_add_co_u32_e32 v89, vcc, v132, v138
	v_mad_u64_u32 v[136:137], s[14:15], v85, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v133, v139, vcc
	v_xad_u32 v83, v83, v134, v46
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[136:137]
	v_mul_lo_u32 v89, s63, v132
	v_mul_lo_u32 v91, s62, v133
	v_mad_u64_u32 v[132:133], s[14:15], s62, v132, 0
	v_add3_u32 v89, v133, v91, v89
	v_sub_u32_e32 v91, v85, v89
	v_sub_co_u32_e32 v87, vcc, v87, v132
	v_mul_lo_u32 v83, v83, s33
	s_nop 0
	v_subb_co_u32_e64 v91, s[14:15], v91, v81, vcc
	v_subrev_co_u32_e64 v93, s[14:15], s62, v87
	v_cmp_le_u32_e64 s[28:29], s62, v93
	s_nop 0
	v_subbrev_co_u32_e64 v95, s[26:27], 0, v91, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v95
	v_cndmask_b32_e64 v109, 0, -1, s[28:29]
	v_subb_co_u32_e64 v91, s[14:15], v91, v81, s[14:15]
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v95
	v_subb_co_u32_e32 v85, vcc, v85, v89, vcc
	s_nop 0
	v_cndmask_b32_e64 v104, v104, v109, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v104
	v_subrev_co_u32_e64 v104, s[14:15], s62, v93
	v_cmp_le_u32_e32 vcc, s63, v85
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[14:15], 0, v91, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v87
	v_cndmask_b32_e64 v91, v95, v91, s[26:27]
	v_cndmask_b32_e64 v89, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v85
	v_cndmask_b32_e64 v95, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x809
	v_cndmask_b32_e32 v89, v89, v95, vcc
	v_cmp_ne_u32_e32 vcc, 0, v89
	v_cndmask_b32_e64 v89, v93, v104, s[26:27]
	v_lshl_add_u64 v[132:133], v[126:127], 0, s[14:15]
	v_cndmask_b32_e32 v87, v87, v89, vcc
	v_cndmask_b32_e32 v85, v85, v91, vcc
	v_xor_b32_e32 v87, v87, v134
	v_xor_b32_e32 v85, v85, v134
	v_sub_co_u32_e32 v87, vcc, v87, v134
	s_movk_i32 s14, 0xf7f6
	s_nop 0
	v_subb_co_u32_e32 v85, vcc, v85, v134, vcc
	v_ashrrev_i32_e32 v85, 31, v85
	v_and_b32_e32 v85, s44, v85
	v_add3_u32 v83, v85, v87, v83
	v_sub_co_u32_e64 v85, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -9, v[130:131]
	s_nop 0
	v_subb_co_u32_e64 v87, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v135, v133, v87, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v132, v85, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v87, v134, v136
	v_xor_b32_e32 v85, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v87, s35, 0
	v_mul_hi_u32 v104, v87, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v85, s34, 0
	v_add_co_u32_e32 v89, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v85, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v135, v141, vcc
	buffer_load_ubyte v83, v83, s[36:39], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[138:139]
	v_mul_lo_u32 v89, s63, v134
	v_mul_lo_u32 v91, s62, v135
	v_mad_u64_u32 v[138:139], s[14:15], s62, v134, 0
	v_add3_u32 v89, v139, v91, v89
	v_sub_co_u32_e32 v87, vcc, v87, v138
	v_cmp_le_u32_e64 s[26:27], s62, v87
	s_nop 0
	v_subb_co_u32_e64 v91, s[14:15], v85, v89, vcc
	v_sub_u32_e32 v85, v85, v89
	v_subb_co_u32_e32 v85, vcc, v85, v81, vcc
	v_subrev_co_u32_e32 v87, vcc, s62, v87
	v_cmp_le_u32_e64 s[14:15], s63, v91
	s_nop 0
	v_subbrev_co_u32_e32 v85, vcc, 0, v85, vcc
	v_cndmask_b32_e64 v93, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v91
	v_cndmask_b32_e64 v91, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v85
	v_cmp_le_u32_e64 s[26:27], s62, v87
	v_cndmask_b32_e64 v91, v93, v91, s[14:15]
	v_cndmask_b32_e64 v89, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v85
	v_cndmask_b32_e64 v85, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v91
	v_cndmask_b32_e32 v85, v89, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	v_xor_b32_e32 v87, s70, v136
	s_nop 0
	v_cndmask_b32_e64 v85, 1, 2, vcc
	v_add_u32_e32 v85, v134, v85
	v_cndmask_b32_e64 v85, v134, v85, s[14:15]
	v_ashrrev_i32_e32 v134, 31, v133
	v_mov_b32_e32 v135, v134
	v_lshl_add_u64 v[132:133], v[132:133], 0, v[134:135]
	v_xor_b32_e32 v85, v85, v87
	v_xor_b32_e32 v89, v132, v134
	v_sub_u32_e32 v85, v85, v87
	v_xor_b32_e32 v87, v133, v134
	v_mad_u64_u32 v[132:133], s[14:15], v89, s35, 0
	v_mul_hi_u32 v104, v89, s34
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[132:133]
	v_mad_u64_u32 v[138:139], s[14:15], v87, s34, 0
	v_add_co_u32_e32 v91, vcc, v132, v138
	v_mad_u64_u32 v[136:137], s[14:15], v87, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v133, v139, vcc
	v_xad_u32 v85, v85, v134, v46
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[136:137]
	v_mul_lo_u32 v91, s63, v132
	v_mul_lo_u32 v93, s62, v133
	v_mad_u64_u32 v[132:133], s[14:15], s62, v132, 0
	v_add3_u32 v91, v133, v93, v91
	v_sub_u32_e32 v93, v87, v91
	v_sub_co_u32_e32 v89, vcc, v89, v132
	v_mul_lo_u32 v85, v85, s33
	s_nop 0
	v_subb_co_u32_e64 v93, s[14:15], v93, v81, vcc
	v_subrev_co_u32_e64 v95, s[14:15], s62, v89
	v_cmp_le_u32_e64 s[28:29], s62, v95
	s_nop 0
	v_subbrev_co_u32_e64 v104, s[26:27], 0, v93, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v104
	v_cndmask_b32_e64 v132, 0, -1, s[28:29]
	v_subb_co_u32_e64 v93, s[14:15], v93, v81, s[14:15]
	v_cndmask_b32_e64 v109, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v104
	v_subb_co_u32_e32 v87, vcc, v87, v91, vcc
	s_nop 0
	v_cndmask_b32_e64 v109, v109, v132, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v109
	v_subrev_co_u32_e64 v109, s[14:15], s62, v95
	v_cmp_le_u32_e32 vcc, s63, v87
	s_nop 0
	v_subbrev_co_u32_e64 v93, s[14:15], 0, v93, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v89
	v_cndmask_b32_e64 v93, v104, v93, s[26:27]
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v87
	v_cndmask_b32_e64 v104, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x80a
	v_cndmask_b32_e32 v91, v91, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v91
	v_cndmask_b32_e64 v91, v95, v109, s[26:27]
	v_lshl_add_u64 v[132:133], v[126:127], 0, s[14:15]
	v_cndmask_b32_e32 v89, v89, v91, vcc
	v_cndmask_b32_e32 v87, v87, v93, vcc
	v_xor_b32_e32 v89, v89, v134
	v_xor_b32_e32 v87, v87, v134
	v_sub_co_u32_e32 v89, vcc, v89, v134
	s_movk_i32 s14, 0xf7f5
	s_nop 0
	v_subb_co_u32_e32 v87, vcc, v87, v134, vcc
	v_ashrrev_i32_e32 v87, 31, v87
	v_and_b32_e32 v87, s44, v87
	v_add3_u32 v85, v87, v89, v85
	v_sub_co_u32_e64 v87, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -10, v[130:131]
	s_nop 0
	v_subb_co_u32_e64 v89, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v135, v133, v89, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v132, v87, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v89, v134, v136
	v_xor_b32_e32 v87, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v89, s35, 0
	v_mul_hi_u32 v104, v89, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v87, s34, 0
	v_add_co_u32_e32 v91, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v87, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v135, v141, vcc
	buffer_load_ubyte v85, v85, s[36:39], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[138:139]
	v_mul_lo_u32 v91, s63, v134
	v_mul_lo_u32 v93, s62, v135
	v_mad_u64_u32 v[138:139], s[14:15], s62, v134, 0
	v_add3_u32 v91, v139, v93, v91
	v_sub_co_u32_e32 v89, vcc, v89, v138
	v_cmp_le_u32_e64 s[26:27], s62, v89
	s_nop 0
	v_subb_co_u32_e64 v93, s[14:15], v87, v91, vcc
	v_sub_u32_e32 v87, v87, v91
	v_subb_co_u32_e32 v87, vcc, v87, v81, vcc
	v_subrev_co_u32_e32 v89, vcc, s62, v89
	v_cmp_le_u32_e64 s[14:15], s63, v93
	s_nop 0
	v_subbrev_co_u32_e32 v87, vcc, 0, v87, vcc
	v_cndmask_b32_e64 v95, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v93
	v_cndmask_b32_e64 v93, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v87
	v_cmp_le_u32_e64 s[26:27], s62, v89
	v_cndmask_b32_e64 v93, v95, v93, s[14:15]
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v87
	v_cndmask_b32_e64 v87, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v93
	v_cndmask_b32_e32 v87, v91, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v87
	v_xor_b32_e32 v89, s70, v136
	s_nop 0
	v_cndmask_b32_e64 v87, 1, 2, vcc
	v_add_u32_e32 v87, v134, v87
	v_cndmask_b32_e64 v87, v134, v87, s[14:15]
	v_ashrrev_i32_e32 v134, 31, v133
	v_mov_b32_e32 v135, v134
	v_lshl_add_u64 v[132:133], v[132:133], 0, v[134:135]
	v_xor_b32_e32 v87, v87, v89
	v_xor_b32_e32 v91, v132, v134
	v_sub_u32_e32 v87, v87, v89
	v_xor_b32_e32 v89, v133, v134
	v_mad_u64_u32 v[132:133], s[14:15], v91, s35, 0
	v_mul_hi_u32 v104, v91, s34
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[132:133]
	v_mad_u64_u32 v[138:139], s[14:15], v89, s34, 0
	v_add_co_u32_e32 v93, vcc, v132, v138
	v_mad_u64_u32 v[136:137], s[14:15], v89, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v133, v139, vcc
	v_xad_u32 v87, v87, v134, v46
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[136:137]
	v_mul_lo_u32 v93, s63, v132
	v_mul_lo_u32 v95, s62, v133
	v_mad_u64_u32 v[132:133], s[14:15], s62, v132, 0
	v_add3_u32 v93, v133, v95, v93
	v_sub_u32_e32 v95, v89, v93
	v_sub_co_u32_e32 v91, vcc, v91, v132
	v_mul_lo_u32 v87, v87, s33
	s_nop 0
	v_subb_co_u32_e64 v95, s[14:15], v95, v81, vcc
	v_subrev_co_u32_e64 v104, s[14:15], s62, v91
	v_cmp_le_u32_e64 s[28:29], s62, v104
	s_nop 0
	v_subbrev_co_u32_e64 v109, s[26:27], 0, v95, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v109
	v_cndmask_b32_e64 v133, 0, -1, s[28:29]
	v_subb_co_u32_e64 v95, s[14:15], v95, v81, s[14:15]
	v_cndmask_b32_e64 v132, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v109
	v_subb_co_u32_e32 v89, vcc, v89, v93, vcc
	s_nop 0
	v_cndmask_b32_e64 v132, v132, v133, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v132
	v_subrev_co_u32_e64 v132, s[14:15], s62, v104
	v_cmp_le_u32_e32 vcc, s63, v89
	s_nop 0
	v_subbrev_co_u32_e64 v95, s[14:15], 0, v95, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v91
	v_cndmask_b32_e64 v95, v109, v95, s[26:27]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v89
	v_cndmask_b32_e64 v109, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x80b
	v_cndmask_b32_e32 v93, v93, v109, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	v_cndmask_b32_e64 v93, v104, v132, s[26:27]
	v_lshl_add_u64 v[132:133], v[126:127], 0, s[14:15]
	v_cndmask_b32_e32 v91, v91, v93, vcc
	v_cndmask_b32_e32 v89, v89, v95, vcc
	v_xor_b32_e32 v91, v91, v134
	v_xor_b32_e32 v89, v89, v134
	v_sub_co_u32_e32 v91, vcc, v91, v134
	s_movk_i32 s14, 0xf7f4
	s_nop 0
	v_subb_co_u32_e32 v89, vcc, v89, v134, vcc
	v_ashrrev_i32_e32 v89, 31, v89
	v_and_b32_e32 v89, s44, v89
	v_add3_u32 v87, v89, v91, v87
	v_sub_co_u32_e64 v89, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -11, v[130:131]
	s_nop 0
	v_subb_co_u32_e64 v91, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v135, v133, v91, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v132, v89, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v91, v134, v136
	v_xor_b32_e32 v89, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v91, s35, 0
	v_mul_hi_u32 v104, v91, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v89, s34, 0
	v_add_co_u32_e32 v93, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v89, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v135, v141, vcc
	buffer_load_ubyte v87, v87, s[36:39], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[138:139]
	v_mul_lo_u32 v93, s63, v134
	v_mul_lo_u32 v95, s62, v135
	v_mad_u64_u32 v[138:139], s[14:15], s62, v134, 0
	v_add3_u32 v93, v139, v95, v93
	v_sub_co_u32_e32 v91, vcc, v91, v138
	v_cmp_le_u32_e64 s[26:27], s62, v91
	s_nop 0
	v_subb_co_u32_e64 v95, s[14:15], v89, v93, vcc
	v_sub_u32_e32 v89, v89, v93
	v_subb_co_u32_e32 v89, vcc, v89, v81, vcc
	v_subrev_co_u32_e32 v91, vcc, s62, v91
	v_cmp_le_u32_e64 s[14:15], s63, v95
	s_nop 0
	v_subbrev_co_u32_e32 v89, vcc, 0, v89, vcc
	v_cndmask_b32_e64 v104, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v95
	v_cndmask_b32_e64 v95, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v89
	v_cmp_le_u32_e64 s[26:27], s62, v91
	v_cndmask_b32_e64 v95, v104, v95, s[14:15]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v89
	v_cndmask_b32_e64 v89, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v95
	v_cndmask_b32_e32 v89, v93, v89, vcc
	v_cmp_ne_u32_e32 vcc, 0, v89
	v_xor_b32_e32 v91, s70, v136
	s_nop 0
	v_cndmask_b32_e64 v89, 1, 2, vcc
	v_add_u32_e32 v89, v134, v89
	v_cndmask_b32_e64 v89, v134, v89, s[14:15]
	v_ashrrev_i32_e32 v134, 31, v133
	v_mov_b32_e32 v135, v134
	v_lshl_add_u64 v[132:133], v[132:133], 0, v[134:135]
	v_xor_b32_e32 v89, v89, v91
	v_xor_b32_e32 v93, v132, v134
	v_sub_u32_e32 v89, v89, v91
	v_xor_b32_e32 v91, v133, v134
	v_mad_u64_u32 v[132:133], s[14:15], v93, s35, 0
	v_mul_hi_u32 v104, v93, s34
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[132:133]
	v_mad_u64_u32 v[138:139], s[14:15], v91, s34, 0
	v_add_co_u32_e32 v95, vcc, v132, v138
	v_mad_u64_u32 v[136:137], s[14:15], v91, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v133, v139, vcc
	v_xad_u32 v89, v89, v134, v46
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[136:137]
	v_mul_lo_u32 v95, s63, v132
	v_mul_lo_u32 v104, s62, v133
	v_mad_u64_u32 v[132:133], s[14:15], s62, v132, 0
	v_add3_u32 v95, v133, v104, v95
	v_sub_u32_e32 v104, v91, v95
	v_sub_co_u32_e32 v93, vcc, v93, v132
	v_mul_lo_u32 v89, v89, s33
	s_nop 0
	v_subb_co_u32_e64 v104, s[14:15], v104, v81, vcc
	v_subrev_co_u32_e64 v109, s[14:15], s62, v93
	v_cmp_le_u32_e64 s[28:29], s62, v109
	s_nop 0
	v_subbrev_co_u32_e64 v132, s[26:27], 0, v104, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v132
	v_cndmask_b32_e64 v135, 0, -1, s[28:29]
	v_subb_co_u32_e64 v104, s[14:15], v104, v81, s[14:15]
	v_cndmask_b32_e64 v133, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v132
	v_subb_co_u32_e32 v91, vcc, v91, v95, vcc
	s_nop 0
	v_cndmask_b32_e64 v133, v133, v135, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v133
	v_subrev_co_u32_e64 v133, s[14:15], s62, v109
	v_cmp_le_u32_e32 vcc, s63, v91
	s_nop 0
	v_subbrev_co_u32_e64 v104, s[14:15], 0, v104, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v93
	v_cndmask_b32_e64 v104, v132, v104, s[26:27]
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v91
	v_cndmask_b32_e64 v132, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x80c
	v_cndmask_b32_e32 v95, v95, v132, vcc
	v_cmp_ne_u32_e32 vcc, 0, v95
	v_cndmask_b32_e64 v95, v109, v133, s[26:27]
	v_lshl_add_u64 v[132:133], v[126:127], 0, s[14:15]
	v_cndmask_b32_e32 v93, v93, v95, vcc
	v_cndmask_b32_e32 v91, v91, v104, vcc
	v_xor_b32_e32 v93, v93, v134
	v_xor_b32_e32 v91, v91, v134
	v_sub_co_u32_e32 v93, vcc, v93, v134
	s_movk_i32 s14, 0xf7f3
	s_nop 0
	v_subb_co_u32_e32 v91, vcc, v91, v134, vcc
	v_ashrrev_i32_e32 v91, 31, v91
	v_and_b32_e32 v91, s44, v91
	v_add3_u32 v89, v91, v93, v89
	v_sub_co_u32_e64 v91, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -12, v[130:131]
	s_nop 0
	v_subb_co_u32_e64 v93, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v135, v133, v93, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v132, v91, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v93, v134, v136
	v_xor_b32_e32 v91, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v93, s35, 0
	v_mul_hi_u32 v104, v93, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v91, s34, 0
	v_add_co_u32_e32 v95, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v91, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v135, v141, vcc
	buffer_load_ubyte v89, v89, s[36:39], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[138:139]
	v_mul_lo_u32 v95, s63, v134
	v_mul_lo_u32 v104, s62, v135
	v_mad_u64_u32 v[138:139], s[14:15], s62, v134, 0
	v_add3_u32 v95, v139, v104, v95
	v_sub_co_u32_e32 v93, vcc, v93, v138
	v_cmp_le_u32_e64 s[26:27], s62, v93
	s_nop 0
	v_subb_co_u32_e64 v104, s[14:15], v91, v95, vcc
	v_sub_u32_e32 v91, v91, v95
	v_subb_co_u32_e32 v91, vcc, v91, v81, vcc
	v_subrev_co_u32_e32 v93, vcc, s62, v93
	v_cmp_le_u32_e64 s[14:15], s63, v104
	s_nop 0
	v_subbrev_co_u32_e32 v91, vcc, 0, v91, vcc
	v_cndmask_b32_e64 v109, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v104
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v91
	v_cmp_le_u32_e64 s[26:27], s62, v93
	v_cndmask_b32_e64 v104, v109, v104, s[14:15]
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v91
	v_cndmask_b32_e64 v91, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v104
	v_cndmask_b32_e32 v91, v95, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v91
	v_xor_b32_e32 v93, s70, v136
	s_nop 0
	v_cndmask_b32_e64 v91, 1, 2, vcc
	v_add_u32_e32 v91, v134, v91
	v_cndmask_b32_e64 v91, v134, v91, s[14:15]
	v_ashrrev_i32_e32 v134, 31, v133
	v_mov_b32_e32 v135, v134
	v_lshl_add_u64 v[132:133], v[132:133], 0, v[134:135]
	v_xor_b32_e32 v91, v91, v93
	v_xor_b32_e32 v95, v132, v134
	v_sub_u32_e32 v91, v91, v93
	v_xor_b32_e32 v93, v133, v134
	v_mad_u64_u32 v[132:133], s[14:15], v95, s35, 0
	v_mul_hi_u32 v104, v95, s34
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[132:133]
	v_mad_u64_u32 v[138:139], s[14:15], v93, s34, 0
	v_add_co_u32_e32 v104, vcc, v132, v138
	v_mad_u64_u32 v[136:137], s[14:15], v93, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v133, v139, vcc
	v_xad_u32 v91, v91, v134, v46
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[136:137]
	v_mul_lo_u32 v104, s63, v132
	v_mul_lo_u32 v109, s62, v133
	v_mad_u64_u32 v[132:133], s[14:15], s62, v132, 0
	v_add3_u32 v104, v133, v109, v104
	v_sub_u32_e32 v109, v93, v104
	v_sub_co_u32_e32 v95, vcc, v95, v132
	v_mul_lo_u32 v91, v91, s33
	s_nop 0
	v_subb_co_u32_e64 v109, s[14:15], v109, v81, vcc
	v_subrev_co_u32_e64 v132, s[14:15], s62, v95
	v_cmp_le_u32_e64 s[28:29], s62, v132
	s_nop 0
	v_subbrev_co_u32_e64 v133, s[26:27], 0, v109, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v133
	v_cndmask_b32_e64 v136, 0, -1, s[28:29]
	v_subb_co_u32_e64 v109, s[14:15], v109, v81, s[14:15]
	v_cndmask_b32_e64 v135, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v133
	v_subb_co_u32_e32 v93, vcc, v93, v104, vcc
	s_nop 0
	v_cndmask_b32_e64 v135, v135, v136, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v135
	v_subrev_co_u32_e64 v135, s[14:15], s62, v132
	v_cmp_le_u32_e32 vcc, s63, v93
	s_nop 0
	v_subbrev_co_u32_e64 v109, s[14:15], 0, v109, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v95
	v_cndmask_b32_e64 v109, v133, v109, s[26:27]
	v_cndmask_b32_e64 v104, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v93
	v_cndmask_b32_e64 v133, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x80d
	v_cndmask_b32_e32 v104, v104, v133, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_cndmask_b32_e64 v104, v132, v135, s[26:27]
	v_lshl_add_u64 v[132:133], v[126:127], 0, s[14:15]
	v_cndmask_b32_e32 v95, v95, v104, vcc
	v_cndmask_b32_e32 v93, v93, v109, vcc
	v_xor_b32_e32 v95, v95, v134
	v_xor_b32_e32 v93, v93, v134
	v_sub_co_u32_e32 v95, vcc, v95, v134
	s_movk_i32 s14, 0xf7f2
	s_nop 0
	v_subb_co_u32_e32 v93, vcc, v93, v134, vcc
	v_ashrrev_i32_e32 v93, 31, v93
	v_and_b32_e32 v93, s44, v93
	v_add3_u32 v91, v93, v95, v91
	v_sub_co_u32_e64 v93, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -13, v[130:131]
	s_nop 0
	v_subb_co_u32_e64 v95, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v135, v133, v95, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v132, v93, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v95, v134, v136
	v_xor_b32_e32 v93, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v95, s35, 0
	v_mul_hi_u32 v104, v95, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v93, s34, 0
	v_add_co_u32_e32 v104, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v93, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v135, v141, vcc
	buffer_load_ubyte v91, v91, s[36:39], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[138:139]
	v_mul_lo_u32 v104, s63, v134
	v_mul_lo_u32 v109, s62, v135
	v_mad_u64_u32 v[138:139], s[14:15], s62, v134, 0
	v_add3_u32 v104, v139, v109, v104
	v_sub_co_u32_e32 v95, vcc, v95, v138
	v_cmp_le_u32_e64 s[26:27], s62, v95
	s_nop 0
	v_subb_co_u32_e64 v109, s[14:15], v93, v104, vcc
	v_sub_u32_e32 v93, v93, v104
	v_subb_co_u32_e32 v93, vcc, v93, v81, vcc
	v_subrev_co_u32_e32 v95, vcc, s62, v95
	v_cmp_le_u32_e64 s[14:15], s63, v109
	s_nop 0
	v_subbrev_co_u32_e32 v93, vcc, 0, v93, vcc
	v_cndmask_b32_e64 v135, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v109
	v_cndmask_b32_e64 v109, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v93
	v_cmp_le_u32_e64 s[26:27], s62, v95
	v_cndmask_b32_e64 v109, v135, v109, s[14:15]
	v_cndmask_b32_e64 v104, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v93
	v_cndmask_b32_e64 v93, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v109
	v_cndmask_b32_e32 v93, v104, v93, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	v_xor_b32_e32 v95, s70, v136
	s_nop 0
	v_cndmask_b32_e64 v93, 1, 2, vcc
	v_add_u32_e32 v93, v134, v93
	v_cndmask_b32_e64 v93, v134, v93, s[14:15]
	v_ashrrev_i32_e32 v134, 31, v133
	v_mov_b32_e32 v135, v134
	v_lshl_add_u64 v[132:133], v[132:133], 0, v[134:135]
	v_xor_b32_e32 v93, v93, v95
	v_xor_b32_e32 v109, v132, v134
	v_sub_u32_e32 v93, v93, v95
	v_xor_b32_e32 v95, v133, v134
	v_mad_u64_u32 v[132:133], s[14:15], v109, s35, 0
	v_mul_hi_u32 v104, v109, s34
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[132:133]
	v_mad_u64_u32 v[138:139], s[14:15], v95, s34, 0
	v_add_co_u32_e32 v104, vcc, v132, v138
	v_mad_u64_u32 v[136:137], s[14:15], v95, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v133, v139, vcc
	v_xad_u32 v93, v93, v134, v46
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[136:137]
	v_mul_lo_u32 v104, s63, v132
	v_mul_lo_u32 v135, s62, v133
	v_mad_u64_u32 v[132:133], s[14:15], s62, v132, 0
	v_add3_u32 v104, v133, v135, v104
	v_sub_u32_e32 v133, v95, v104
	v_sub_co_u32_e32 v109, vcc, v109, v132
	v_mul_lo_u32 v93, v93, s33
	s_nop 0
	v_subb_co_u32_e64 v132, s[14:15], v133, v81, vcc
	v_subrev_co_u32_e64 v133, s[14:15], s62, v109
	v_cmp_le_u32_e64 s[28:29], s62, v133
	s_nop 0
	v_subbrev_co_u32_e64 v135, s[26:27], 0, v132, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v135
	v_cndmask_b32_e64 v137, 0, -1, s[28:29]
	v_subb_co_u32_e64 v132, s[14:15], v132, v81, s[14:15]
	v_cndmask_b32_e64 v136, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v135
	v_subb_co_u32_e32 v95, vcc, v95, v104, vcc
	s_nop 0
	v_cndmask_b32_e64 v136, v136, v137, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v136
	v_subrev_co_u32_e64 v136, s[14:15], s62, v133
	v_cmp_le_u32_e32 vcc, s63, v95
	s_nop 0
	v_subbrev_co_u32_e64 v132, s[14:15], 0, v132, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v109
	v_cndmask_b32_e64 v132, v135, v132, s[26:27]
	v_cndmask_b32_e64 v104, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v95
	v_cndmask_b32_e64 v135, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x80e
	v_cndmask_b32_e32 v104, v104, v135, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_cndmask_b32_e64 v104, v133, v136, s[26:27]
	s_nop 0
	v_cndmask_b32_e32 v104, v109, v104, vcc
	v_cndmask_b32_e32 v95, v95, v132, vcc
	v_xor_b32_e32 v104, v104, v134
	v_xor_b32_e32 v95, v95, v134
	v_sub_co_u32_e32 v104, vcc, v104, v134
	v_lshl_add_u64 v[132:133], v[126:127], 0, s[14:15]
	s_nop 0
	v_subb_co_u32_e32 v95, vcc, v95, v134, vcc
	v_ashrrev_i32_e32 v95, 31, v95
	v_and_b32_e32 v95, s44, v95
	s_movk_i32 s14, 0xf7f1
	v_add3_u32 v93, v95, v104, v93
	v_sub_co_u32_e64 v95, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -14, v[130:131]
	s_nop 0
	v_subb_co_u32_e64 v104, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v135, v133, v104, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v132, v95, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v109, v134, v136
	v_xor_b32_e32 v95, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v109, s35, 0
	v_mul_hi_u32 v104, v109, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v95, s34, 0
	v_add_co_u32_e32 v104, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v95, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v135, v141, vcc
	buffer_load_ubyte v93, v93, s[36:39], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[138:139]
	v_mul_lo_u32 v104, s63, v134
	v_mul_lo_u32 v135, s62, v135
	v_mad_u64_u32 v[138:139], s[14:15], s62, v134, 0
	v_add3_u32 v104, v139, v135, v104
	v_sub_co_u32_e32 v109, vcc, v109, v138
	v_cmp_le_u32_e64 s[26:27], s62, v109
	s_nop 0
	v_subb_co_u32_e64 v135, s[14:15], v95, v104, vcc
	v_sub_u32_e32 v95, v95, v104
	v_subb_co_u32_e32 v95, vcc, v95, v81, vcc
	v_subrev_co_u32_e32 v104, vcc, s62, v109
	v_cmp_le_u32_e64 s[14:15], s63, v135
	s_nop 0
	v_subbrev_co_u32_e32 v95, vcc, 0, v95, vcc
	v_cndmask_b32_e64 v137, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v135
	v_cndmask_b32_e64 v135, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v95
	v_cmp_le_u32_e64 s[26:27], s62, v104
	v_cndmask_b32_e64 v135, v137, v135, s[14:15]
	v_cndmask_b32_e64 v109, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v95
	v_cndmask_b32_e64 v95, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v135
	v_cndmask_b32_e32 v95, v109, v95, vcc
	v_cmp_ne_u32_e32 vcc, 0, v95
	v_xor_b32_e32 v104, s70, v136
	s_nop 0
	v_cndmask_b32_e64 v95, 1, 2, vcc
	v_add_u32_e32 v95, v134, v95
	v_cndmask_b32_e64 v95, v134, v95, s[14:15]
	v_ashrrev_i32_e32 v134, 31, v133
	v_mov_b32_e32 v135, v134
	v_lshl_add_u64 v[132:133], v[132:133], 0, v[134:135]
	v_xor_b32_e32 v95, v95, v104
	v_xor_b32_e32 v135, v132, v134
	v_sub_u32_e32 v95, v95, v104
	v_xor_b32_e32 v109, v133, v134
	v_mad_u64_u32 v[132:133], s[14:15], v135, s35, 0
	v_mul_hi_u32 v104, v135, s34
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[132:133]
	v_mad_u64_u32 v[138:139], s[14:15], v109, s34, 0
	v_add_co_u32_e32 v104, vcc, v132, v138
	v_mad_u64_u32 v[136:137], s[14:15], v109, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v133, v139, vcc
	v_xad_u32 v95, v95, v134, v46
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[136:137]
	v_mul_lo_u32 v104, s63, v132
	v_mul_lo_u32 v136, s62, v133
	v_mad_u64_u32 v[132:133], s[14:15], s62, v132, 0
	v_add3_u32 v104, v133, v136, v104
	v_sub_u32_e32 v133, v109, v104
	v_sub_co_u32_e32 v132, vcc, v135, v132
	v_mul_lo_u32 v95, v95, s33
	s_nop 0
	v_subb_co_u32_e64 v133, s[14:15], v133, v81, vcc
	v_subrev_co_u32_e64 v135, s[14:15], s62, v132
	v_cmp_le_u32_e64 s[28:29], s62, v135
	s_nop 0
	v_subbrev_co_u32_e64 v136, s[26:27], 0, v133, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v136
	v_cndmask_b32_e64 v138, 0, -1, s[28:29]
	v_subb_co_u32_e64 v133, s[14:15], v133, v81, s[14:15]
	v_cndmask_b32_e64 v137, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v136
	v_subb_co_u32_e32 v104, vcc, v109, v104, vcc
	s_nop 0
	v_cndmask_b32_e64 v137, v137, v138, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v137
	v_subrev_co_u32_e64 v137, s[14:15], s62, v135
	v_cmp_le_u32_e32 vcc, s63, v104
	s_nop 0
	v_subbrev_co_u32_e64 v133, s[14:15], 0, v133, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v132
	v_cndmask_b32_e64 v133, v136, v133, s[26:27]
	v_cndmask_b32_e64 v109, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v136, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x80f
	v_cndmask_b32_e32 v109, v109, v136, vcc
	v_cmp_ne_u32_e32 vcc, 0, v109
	v_cndmask_b32_e64 v109, v135, v137, s[26:27]
	s_nop 0
	v_cndmask_b32_e32 v109, v132, v109, vcc
	v_cndmask_b32_e32 v104, v104, v133, vcc
	v_xor_b32_e32 v109, v109, v134
	v_xor_b32_e32 v104, v104, v134
	v_sub_co_u32_e32 v109, vcc, v109, v134
	v_lshl_add_u64 v[132:133], v[126:127], 0, s[14:15]
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, v104, v134, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	s_movk_i32 s14, 0xf7f0
	v_add3_u32 v95, v104, v109, v95
	v_sub_co_u32_e64 v104, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -15, v[130:131]
	s_nop 0
	v_subb_co_u32_e64 v109, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v131, v133, v109, vcc
	v_ashrrev_i32_e32 v134, 31, v131
	v_cndmask_b32_e32 v130, v132, v104, vcc
	v_mov_b32_e32 v135, v134
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[134:135]
	v_xor_b32_e32 v135, v130, v134
	v_xor_b32_e32 v109, v131, v134
	v_mad_u64_u32 v[130:131], s[14:15], v135, s35, 0
	v_mul_hi_u32 v104, v135, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[138:139], s[14:15], v109, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v138
	v_mad_u64_u32 v[136:137], s[14:15], v109, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v131, v139, vcc
	buffer_load_ubyte v95, v95, s[36:39], 0 offen
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[136:137]
	v_mul_lo_u32 v104, s63, v130
	v_mul_lo_u32 v131, s62, v131
	v_mad_u64_u32 v[136:137], s[14:15], s62, v130, 0
	v_add3_u32 v104, v137, v131, v104
	v_sub_co_u32_e32 v131, vcc, v135, v136
	v_cmp_le_u32_e64 s[26:27], s62, v131
	s_nop 0
	v_subb_co_u32_e64 v135, s[14:15], v109, v104, vcc
	v_sub_u32_e32 v104, v109, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v109, vcc, s62, v131
	v_cmp_le_u32_e64 s[14:15], s63, v135
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cndmask_b32_e64 v136, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v135
	v_cndmask_b32_e64 v135, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v104
	v_cmp_le_u32_e64 s[26:27], s62, v109
	v_cndmask_b32_e64 v135, v136, v135, s[14:15]
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v135
	v_cndmask_b32_e32 v104, v131, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v109, s70, v134
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v130, v104
	v_cndmask_b32_e64 v104, v130, v104, s[14:15]
	v_ashrrev_i32_e32 v130, 31, v133
	v_mov_b32_e32 v131, v130
	v_xor_b32_e32 v104, v104, v109
	v_lshl_add_u64 v[132:133], v[132:133], 0, v[130:131]
	v_sub_u32_e32 v104, v104, v109
	v_xor_b32_e32 v138, v132, v130
	v_xad_u32 v109, v104, v130, v46
	v_xor_b32_e32 v131, v133, v130
	v_mad_u64_u32 v[132:133], s[14:15], v138, s35, 0
	v_mul_hi_u32 v104, v138, s34
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[132:133]
	v_mad_u64_u32 v[136:137], s[14:15], v131, s34, 0
	v_add_co_u32_e32 v104, vcc, v132, v136
	v_mad_u64_u32 v[134:135], s[14:15], v131, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v133, v137, vcc
	v_mul_lo_u32 v109, v109, s33
	s_nop 0
	v_addc_co_u32_e32 v135, vcc, 0, v135, vcc
	v_lshl_add_u64 v[132:133], v[104:105], 0, v[134:135]
	v_mul_lo_u32 v104, s63, v132
	v_mul_lo_u32 v134, s62, v133
	v_mad_u64_u32 v[132:133], s[14:15], s62, v132, 0
	v_add3_u32 v104, v133, v134, v104
	v_sub_u32_e32 v133, v131, v104
	v_sub_co_u32_e32 v132, vcc, v138, v132
	s_nop 1
	v_subb_co_u32_e64 v133, s[14:15], v133, v81, vcc
	v_subrev_co_u32_e64 v134, s[14:15], s62, v132
	v_cmp_le_u32_e64 s[28:29], s62, v134
	s_nop 0
	v_subbrev_co_u32_e64 v135, s[26:27], 0, v133, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v135
	v_cndmask_b32_e64 v137, 0, -1, s[28:29]
	v_subb_co_u32_e64 v133, s[14:15], v133, v81, s[14:15]
	v_cndmask_b32_e64 v136, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v135
	v_subb_co_u32_e32 v104, vcc, v131, v104, vcc
	s_nop 0
	v_cndmask_b32_e64 v136, v136, v137, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v136
	v_subrev_co_u32_e64 v136, s[14:15], s62, v134
	v_cmp_le_u32_e32 vcc, s63, v104
	s_nop 0
	v_subbrev_co_u32_e64 v133, s[14:15], 0, v133, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v132
	v_cndmask_b32_e64 v133, v135, v133, s[26:27]
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v135, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x1400
	v_cndmask_b32_e32 v131, v131, v135, vcc
	v_cmp_ne_u32_e32 vcc, 0, v131
	v_cndmask_b32_e64 v131, v134, v136, s[26:27]
	s_nop 0
	v_cndmask_b32_e32 v131, v132, v131, vcc
	v_cndmask_b32_e32 v104, v104, v133, vcc
	v_xor_b32_e32 v131, v131, v130
	v_xor_b32_e32 v104, v104, v130
	v_sub_co_u32_e32 v131, vcc, v131, v130
	s_nop 1
	v_subb_co_u32_e32 v104, vcc, v104, v130, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v104, v104, v131, v109
	v_lshl_add_u64 v[130:131], v[128:129], 0, s[14:15]
	s_movk_i32 s14, 0xebff
	buffer_load_ubyte v109, v104, s[36:39], 0 offen
	v_sub_co_u32_e64 v104, s[14:15], s14, v128
	v_cmp_gt_i32_e32 vcc, 0, v131
	s_nop 0
	v_subb_co_u32_e64 v122, s[14:15], -1, v129, s[14:15]
	v_cndmask_b32_e32 v129, v131, v122, vcc
	v_ashrrev_i32_e32 v132, 31, v129
	v_cndmask_b32_e32 v128, v130, v104, vcc
	v_mov_b32_e32 v133, v132
	v_lshl_add_u64 v[128:129], v[128:129], 0, v[132:133]
	v_xor_b32_e32 v133, v128, v132
	v_xor_b32_e32 v122, v129, v132
	v_mad_u64_u32 v[128:129], s[14:15], v133, s35, 0
	v_mul_hi_u32 v104, v133, s34
	v_lshl_add_u64 v[128:129], v[104:105], 0, v[128:129]
	v_mad_u64_u32 v[136:137], s[14:15], v122, s34, 0
	v_add_co_u32_e32 v104, vcc, v128, v136
	v_mad_u64_u32 v[134:135], s[14:15], v122, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v129, v137, vcc
	s_nop 1
	v_addc_co_u32_e32 v135, vcc, 0, v135, vcc
	v_lshl_add_u64 v[128:129], v[104:105], 0, v[134:135]
	v_mul_lo_u32 v104, s63, v128
	v_mul_lo_u32 v129, s62, v129
	v_mad_u64_u32 v[134:135], s[14:15], s62, v128, 0
	v_add3_u32 v104, v135, v129, v104
	v_sub_co_u32_e32 v129, vcc, v133, v134
	v_cmp_le_u32_e64 s[26:27], s62, v129
	s_nop 0
	v_subb_co_u32_e64 v133, s[14:15], v122, v104, vcc
	v_sub_u32_e32 v104, v122, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v122, vcc, s62, v129
	v_cmp_le_u32_e64 s[14:15], s63, v133
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cndmask_b32_e64 v134, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v133
	v_cndmask_b32_e64 v133, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v104
	v_cmp_le_u32_e64 s[26:27], s62, v122
	v_cndmask_b32_e64 v133, v134, v133, s[14:15]
	v_cndmask_b32_e64 v129, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v133
	v_cndmask_b32_e32 v104, v129, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v122, s70, v132
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v128, v104
	v_cndmask_b32_e64 v104, v128, v104, s[14:15]
	v_ashrrev_i32_e32 v128, 31, v131
	v_mov_b32_e32 v129, v128
	v_xor_b32_e32 v104, v104, v122
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[128:129]
	v_sub_u32_e32 v104, v104, v122
	v_xor_b32_e32 v136, v130, v128
	v_xad_u32 v122, v104, v128, v46
	v_xor_b32_e32 v129, v131, v128
	v_mad_u64_u32 v[130:131], s[14:15], v136, s35, 0
	v_mul_hi_u32 v104, v136, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[134:135], s[14:15], v129, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v134
	v_mad_u64_u32 v[132:133], s[14:15], v129, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v131, v135, vcc
	v_mul_lo_u32 v122, v122, s33
	s_nop 0
	v_addc_co_u32_e32 v133, vcc, 0, v133, vcc
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[132:133]
	v_mul_lo_u32 v104, s63, v130
	v_mul_lo_u32 v132, s62, v131
	v_mad_u64_u32 v[130:131], s[14:15], s62, v130, 0
	v_add3_u32 v104, v131, v132, v104
	v_sub_u32_e32 v131, v129, v104
	v_sub_co_u32_e32 v130, vcc, v136, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v132, s[14:15], s62, v130
	v_cmp_le_u32_e64 s[28:29], s62, v132
	s_nop 0
	v_subbrev_co_u32_e64 v133, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v133
	v_cndmask_b32_e64 v135, 0, -1, s[28:29]
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	v_cndmask_b32_e64 v134, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v133
	v_subb_co_u32_e32 v104, vcc, v129, v104, vcc
	s_nop 0
	v_cndmask_b32_e64 v134, v134, v135, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v134
	v_subrev_co_u32_e64 v134, s[14:15], s62, v132
	v_cmp_le_u32_e32 vcc, s63, v104
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v130
	v_cndmask_b32_e64 v131, v133, v131, s[26:27]
	v_cndmask_b32_e64 v129, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v133, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0xc01
	v_cndmask_b32_e32 v129, v129, v133, vcc
	v_cmp_ne_u32_e32 vcc, 0, v129
	v_cndmask_b32_e64 v129, v132, v134, s[26:27]
	s_nop 0
	v_cndmask_b32_e32 v129, v130, v129, vcc
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_xor_b32_e32 v129, v129, v128
	v_xor_b32_e32 v104, v104, v128
	v_sub_co_u32_e32 v129, vcc, v129, v128
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, v104, v128, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v104, v104, v129, v122
	s_movk_i32 s14, 0xf3fe
	buffer_load_ubyte v132, v104, s[36:39], 0 offen
	v_lshl_add_u64 v[128:129], v[126:127], 0, s[94:95]
	v_sub_co_u32_e64 v104, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -1, v[128:129]
	s_nop 0
	v_subb_co_u32_e64 v122, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v135, v131, v122, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v130, v104, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v133, v134, v136
	v_xor_b32_e32 v122, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v133, s35, 0
	v_mul_hi_u32 v104, v133, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v122, s34, 0
	v_add_co_u32_e32 v104, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v122, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v135, v141, vcc
	s_nop 1
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[138:139]
	v_mul_lo_u32 v104, s63, v134
	v_mul_lo_u32 v135, s62, v135
	v_mad_u64_u32 v[138:139], s[14:15], s62, v134, 0
	v_add3_u32 v104, v139, v135, v104
	v_sub_co_u32_e32 v133, vcc, v133, v138
	v_cmp_le_u32_e64 s[26:27], s62, v133
	s_nop 0
	v_subb_co_u32_e64 v135, s[14:15], v122, v104, vcc
	v_sub_u32_e32 v104, v122, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v122, vcc, s62, v133
	v_cmp_le_u32_e64 s[14:15], s63, v135
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cndmask_b32_e64 v137, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v135
	v_cndmask_b32_e64 v135, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v104
	v_cmp_le_u32_e64 s[26:27], s62, v122
	v_cndmask_b32_e64 v135, v137, v135, s[14:15]
	v_cndmask_b32_e64 v133, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v135
	v_cndmask_b32_e32 v104, v133, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v122, s70, v136
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v134, v104
	v_cndmask_b32_e64 v104, v134, v104, s[14:15]
	v_ashrrev_i32_e32 v134, 31, v131
	v_mov_b32_e32 v135, v134
	v_xor_b32_e32 v104, v104, v122
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[134:135]
	v_sub_u32_e32 v104, v104, v122
	v_xor_b32_e32 v135, v130, v134
	v_xad_u32 v122, v104, v134, v46
	v_xor_b32_e32 v133, v131, v134
	v_mad_u64_u32 v[130:131], s[14:15], v135, s35, 0
	v_mul_hi_u32 v104, v135, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[138:139], s[14:15], v133, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v138
	v_mad_u64_u32 v[136:137], s[14:15], v133, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v131, v139, vcc
	v_mul_lo_u32 v122, v122, s33
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[136:137]
	v_mul_lo_u32 v104, s63, v130
	v_mul_lo_u32 v136, s62, v131
	v_mad_u64_u32 v[130:131], s[14:15], s62, v130, 0
	v_add3_u32 v104, v131, v136, v104
	v_sub_u32_e32 v131, v133, v104
	v_sub_co_u32_e32 v130, vcc, v135, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v135, s[14:15], s62, v130
	v_cmp_le_u32_e64 s[28:29], s62, v135
	s_nop 0
	v_subbrev_co_u32_e64 v136, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v136
	v_cndmask_b32_e64 v138, 0, -1, s[28:29]
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	v_cndmask_b32_e64 v137, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v136
	v_subb_co_u32_e32 v104, vcc, v133, v104, vcc
	s_nop 0
	v_cndmask_b32_e64 v137, v137, v138, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v137
	v_subrev_co_u32_e64 v137, s[14:15], s62, v135
	v_cmp_le_u32_e32 vcc, s63, v104
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v130
	v_cndmask_b32_e64 v131, v136, v131, s[26:27]
	v_cndmask_b32_e64 v133, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v136, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0xc02
	v_cndmask_b32_e32 v133, v133, v136, vcc
	v_cmp_ne_u32_e32 vcc, 0, v133
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v135, v137, s[26:27]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v134
	v_xor_b32_e32 v104, v104, v134
	v_sub_co_u32_e32 v130, vcc, v130, v134
	s_nop 1
	v_subb_co_u32_e32 v104, vcc, v104, v134, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v104, v104, v130, v122
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xf3fd
	buffer_load_ubyte v133, v104, s[36:39], 0 offen
	v_sub_co_u32_e64 v104, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -2, v[128:129]
	s_nop 0
	v_subb_co_u32_e64 v122, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v135, v131, v122, vcc
	v_ashrrev_i32_e32 v136, 31, v135
	v_cndmask_b32_e32 v134, v130, v104, vcc
	v_mov_b32_e32 v137, v136
	v_lshl_add_u64 v[134:135], v[134:135], 0, v[136:137]
	v_xor_b32_e32 v137, v134, v136
	v_xor_b32_e32 v122, v135, v136
	v_mad_u64_u32 v[134:135], s[14:15], v137, s35, 0
	v_mul_hi_u32 v104, v137, s34
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[134:135]
	v_mad_u64_u32 v[140:141], s[14:15], v122, s34, 0
	v_add_co_u32_e32 v104, vcc, v134, v140
	v_mad_u64_u32 v[138:139], s[14:15], v122, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v135, v141, vcc
	s_nop 1
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[134:135], v[104:105], 0, v[138:139]
	v_mul_lo_u32 v104, s63, v134
	v_mul_lo_u32 v135, s62, v135
	v_mad_u64_u32 v[138:139], s[14:15], s62, v134, 0
	v_add3_u32 v104, v139, v135, v104
	v_sub_co_u32_e32 v135, vcc, v137, v138
	v_cmp_le_u32_e64 s[26:27], s62, v135
	s_nop 0
	v_subb_co_u32_e64 v137, s[14:15], v122, v104, vcc
	v_sub_u32_e32 v104, v122, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v122, vcc, s62, v135
	v_cmp_le_u32_e64 s[14:15], s63, v137
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cndmask_b32_e64 v138, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v137
	v_cndmask_b32_e64 v137, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v104
	v_cmp_le_u32_e64 s[26:27], s62, v122
	v_cndmask_b32_e64 v137, v138, v137, s[14:15]
	v_cndmask_b32_e64 v135, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v137
	v_cndmask_b32_e32 v104, v135, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v122, s70, v136
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v134, v104
	v_cndmask_b32_e64 v104, v134, v104, s[14:15]
	v_ashrrev_i32_e32 v134, 31, v131
	v_mov_b32_e32 v135, v134
	v_xor_b32_e32 v104, v104, v122
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[134:135]
	v_sub_u32_e32 v104, v104, v122
	v_xor_b32_e32 v140, v130, v134
	v_xad_u32 v122, v104, v134, v46
	v_xor_b32_e32 v135, v131, v134
	v_mad_u64_u32 v[130:131], s[14:15], v140, s35, 0
	v_mul_hi_u32 v104, v140, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[138:139], s[14:15], v135, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v138
	v_mad_u64_u32 v[136:137], s[14:15], v135, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v131, v139, vcc
	v_mul_lo_u32 v122, v122, s33
	s_nop 0
	v_addc_co_u32_e32 v137, vcc, 0, v137, vcc
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[136:137]
	v_mul_lo_u32 v104, s63, v130
	v_mul_lo_u32 v136, s62, v131
	v_mad_u64_u32 v[130:131], s[14:15], s62, v130, 0
	v_add3_u32 v104, v131, v136, v104
	v_sub_u32_e32 v131, v135, v104
	v_sub_co_u32_e32 v130, vcc, v140, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v136, s[14:15], s62, v130
	v_cmp_le_u32_e64 s[28:29], s62, v136
	s_nop 0
	v_subbrev_co_u32_e64 v137, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v137
	v_cndmask_b32_e64 v139, 0, -1, s[28:29]
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	v_cndmask_b32_e64 v138, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v137
	v_subb_co_u32_e32 v104, vcc, v135, v104, vcc
	s_nop 0
	v_cndmask_b32_e64 v138, v138, v139, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v138
	v_subrev_co_u32_e64 v138, s[14:15], s62, v136
	v_cmp_le_u32_e32 vcc, s63, v104
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v130
	v_cndmask_b32_e64 v131, v137, v131, s[26:27]
	v_cndmask_b32_e64 v135, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v137, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0xc03
	v_cndmask_b32_e32 v135, v135, v137, vcc
	v_cmp_ne_u32_e32 vcc, 0, v135
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v136, v138, s[26:27]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v134
	v_xor_b32_e32 v104, v104, v134
	v_sub_co_u32_e32 v130, vcc, v130, v134
	s_nop 1
	v_subb_co_u32_e32 v104, vcc, v104, v134, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v104, v104, v130, v122
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xf3fc
	buffer_load_ubyte v134, v104, s[36:39], 0 offen
	v_sub_co_u32_e64 v104, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -3, v[128:129]
	s_nop 0
	v_subb_co_u32_e64 v122, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v137, v131, v122, vcc
	v_ashrrev_i32_e32 v138, 31, v137
	v_cndmask_b32_e32 v136, v130, v104, vcc
	v_mov_b32_e32 v139, v138
	v_lshl_add_u64 v[136:137], v[136:137], 0, v[138:139]
	v_xor_b32_e32 v135, v136, v138
	v_xor_b32_e32 v122, v137, v138
	v_mad_u64_u32 v[136:137], s[14:15], v135, s35, 0
	v_mul_hi_u32 v104, v135, s34
	v_lshl_add_u64 v[136:137], v[104:105], 0, v[136:137]
	v_mad_u64_u32 v[142:143], s[14:15], v122, s34, 0
	v_add_co_u32_e32 v104, vcc, v136, v142
	v_mad_u64_u32 v[140:141], s[14:15], v122, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v137, v143, vcc
	s_nop 1
	v_addc_co_u32_e32 v141, vcc, 0, v141, vcc
	v_lshl_add_u64 v[136:137], v[104:105], 0, v[140:141]
	v_mul_lo_u32 v104, s63, v136
	v_mul_lo_u32 v137, s62, v137
	v_mad_u64_u32 v[140:141], s[14:15], s62, v136, 0
	v_add3_u32 v104, v141, v137, v104
	v_sub_co_u32_e32 v135, vcc, v135, v140
	v_cmp_le_u32_e64 s[26:27], s62, v135
	s_nop 0
	v_subb_co_u32_e64 v137, s[14:15], v122, v104, vcc
	v_sub_u32_e32 v104, v122, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v122, vcc, s62, v135
	v_cmp_le_u32_e64 s[14:15], s63, v137
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cndmask_b32_e64 v139, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v137
	v_cndmask_b32_e64 v137, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v104
	v_cmp_le_u32_e64 s[26:27], s62, v122
	v_cndmask_b32_e64 v137, v139, v137, s[14:15]
	v_cndmask_b32_e64 v135, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v137
	v_cndmask_b32_e32 v104, v135, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v122, s70, v138
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v136, v104
	v_cndmask_b32_e64 v104, v136, v104, s[14:15]
	v_ashrrev_i32_e32 v136, 31, v131
	v_mov_b32_e32 v137, v136
	v_xor_b32_e32 v104, v104, v122
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[136:137]
	v_sub_u32_e32 v104, v104, v122
	v_xor_b32_e32 v137, v130, v136
	v_xad_u32 v122, v104, v136, v46
	v_xor_b32_e32 v135, v131, v136
	v_mad_u64_u32 v[130:131], s[14:15], v137, s35, 0
	v_mul_hi_u32 v104, v137, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[140:141], s[14:15], v135, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v140
	v_mad_u64_u32 v[138:139], s[14:15], v135, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v131, v141, vcc
	v_mul_lo_u32 v122, v122, s33
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[138:139]
	v_mul_lo_u32 v104, s63, v130
	v_mul_lo_u32 v138, s62, v131
	v_mad_u64_u32 v[130:131], s[14:15], s62, v130, 0
	v_add3_u32 v104, v131, v138, v104
	v_sub_u32_e32 v131, v135, v104
	v_sub_co_u32_e32 v130, vcc, v137, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v137, s[14:15], s62, v130
	v_cmp_le_u32_e64 s[28:29], s62, v137
	s_nop 0
	v_subbrev_co_u32_e64 v138, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v138
	v_cndmask_b32_e64 v140, 0, -1, s[28:29]
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	v_cndmask_b32_e64 v139, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v138
	v_subb_co_u32_e32 v104, vcc, v135, v104, vcc
	s_nop 0
	v_cndmask_b32_e64 v139, v139, v140, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v139
	v_subrev_co_u32_e64 v139, s[14:15], s62, v137
	v_cmp_le_u32_e32 vcc, s63, v104
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v130
	v_cndmask_b32_e64 v131, v138, v131, s[26:27]
	v_cndmask_b32_e64 v135, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v138, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0xc04
	v_cndmask_b32_e32 v135, v135, v138, vcc
	v_cmp_ne_u32_e32 vcc, 0, v135
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v137, v139, s[26:27]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v136
	v_xor_b32_e32 v104, v104, v136
	v_sub_co_u32_e32 v130, vcc, v130, v136
	s_nop 1
	v_subb_co_u32_e32 v104, vcc, v104, v136, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v104, v104, v130, v122
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xf3fb
	buffer_load_ubyte v135, v104, s[36:39], 0 offen
	v_sub_co_u32_e64 v104, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -4, v[128:129]
	s_nop 0
	v_subb_co_u32_e64 v122, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v137, v131, v122, vcc
	v_ashrrev_i32_e32 v138, 31, v137
	v_cndmask_b32_e32 v136, v130, v104, vcc
	v_mov_b32_e32 v139, v138
	v_lshl_add_u64 v[136:137], v[136:137], 0, v[138:139]
	v_xor_b32_e32 v139, v136, v138
	v_xor_b32_e32 v122, v137, v138
	v_mad_u64_u32 v[136:137], s[14:15], v139, s35, 0
	v_mul_hi_u32 v104, v139, s34
	v_lshl_add_u64 v[136:137], v[104:105], 0, v[136:137]
	v_mad_u64_u32 v[142:143], s[14:15], v122, s34, 0
	v_add_co_u32_e32 v104, vcc, v136, v142
	v_mad_u64_u32 v[140:141], s[14:15], v122, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v137, v143, vcc
	s_nop 1
	v_addc_co_u32_e32 v141, vcc, 0, v141, vcc
	v_lshl_add_u64 v[136:137], v[104:105], 0, v[140:141]
	v_mul_lo_u32 v104, s63, v136
	v_mul_lo_u32 v137, s62, v137
	v_mad_u64_u32 v[140:141], s[14:15], s62, v136, 0
	v_add3_u32 v104, v141, v137, v104
	v_sub_co_u32_e32 v137, vcc, v139, v140
	v_cmp_le_u32_e64 s[26:27], s62, v137
	s_nop 0
	v_subb_co_u32_e64 v139, s[14:15], v122, v104, vcc
	v_sub_u32_e32 v104, v122, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v122, vcc, s62, v137
	v_cmp_le_u32_e64 s[14:15], s63, v139
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cndmask_b32_e64 v140, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v139
	v_cndmask_b32_e64 v139, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v104
	v_cmp_le_u32_e64 s[26:27], s62, v122
	v_cndmask_b32_e64 v139, v140, v139, s[14:15]
	v_cndmask_b32_e64 v137, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v139
	v_cndmask_b32_e32 v104, v137, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v122, s70, v138
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v136, v104
	v_cndmask_b32_e64 v104, v136, v104, s[14:15]
	v_ashrrev_i32_e32 v136, 31, v131
	v_mov_b32_e32 v137, v136
	v_xor_b32_e32 v104, v104, v122
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[136:137]
	v_sub_u32_e32 v104, v104, v122
	v_xor_b32_e32 v142, v130, v136
	v_xad_u32 v122, v104, v136, v46
	v_xor_b32_e32 v137, v131, v136
	v_mad_u64_u32 v[130:131], s[14:15], v142, s35, 0
	v_mul_hi_u32 v104, v142, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[140:141], s[14:15], v137, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v140
	v_mad_u64_u32 v[138:139], s[14:15], v137, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v131, v141, vcc
	v_mul_lo_u32 v122, v122, s33
	s_nop 0
	v_addc_co_u32_e32 v139, vcc, 0, v139, vcc
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[138:139]
	v_mul_lo_u32 v104, s63, v130
	v_mul_lo_u32 v138, s62, v131
	v_mad_u64_u32 v[130:131], s[14:15], s62, v130, 0
	v_add3_u32 v104, v131, v138, v104
	v_sub_u32_e32 v131, v137, v104
	v_sub_co_u32_e32 v130, vcc, v142, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v138, s[14:15], s62, v130
	v_cmp_le_u32_e64 s[28:29], s62, v138
	s_nop 0
	v_subbrev_co_u32_e64 v139, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v139
	v_cndmask_b32_e64 v141, 0, -1, s[28:29]
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	v_cndmask_b32_e64 v140, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v139
	v_subb_co_u32_e32 v104, vcc, v137, v104, vcc
	s_nop 0
	v_cndmask_b32_e64 v140, v140, v141, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v140
	v_subrev_co_u32_e64 v140, s[14:15], s62, v138
	v_cmp_le_u32_e32 vcc, s63, v104
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v130
	v_cndmask_b32_e64 v131, v139, v131, s[26:27]
	v_cndmask_b32_e64 v137, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v139, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0xc05
	v_cndmask_b32_e32 v137, v137, v139, vcc
	v_cmp_ne_u32_e32 vcc, 0, v137
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v138, v140, s[26:27]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v136
	v_xor_b32_e32 v104, v104, v136
	v_sub_co_u32_e32 v130, vcc, v130, v136
	s_nop 1
	v_subb_co_u32_e32 v104, vcc, v104, v136, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v104, v104, v130, v122
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xf3fa
	buffer_load_ubyte v136, v104, s[36:39], 0 offen
	v_sub_co_u32_e64 v104, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -5, v[128:129]
	s_nop 0
	v_subb_co_u32_e64 v122, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v139, v131, v122, vcc
	v_ashrrev_i32_e32 v140, 31, v139
	v_cndmask_b32_e32 v138, v130, v104, vcc
	v_mov_b32_e32 v141, v140
	v_lshl_add_u64 v[138:139], v[138:139], 0, v[140:141]
	v_xor_b32_e32 v137, v138, v140
	v_xor_b32_e32 v122, v139, v140
	v_mad_u64_u32 v[138:139], s[14:15], v137, s35, 0
	v_mul_hi_u32 v104, v137, s34
	v_lshl_add_u64 v[138:139], v[104:105], 0, v[138:139]
	v_mad_u64_u32 v[144:145], s[14:15], v122, s34, 0
	v_add_co_u32_e32 v104, vcc, v138, v144
	v_mad_u64_u32 v[142:143], s[14:15], v122, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v139, v145, vcc
	s_nop 1
	v_addc_co_u32_e32 v143, vcc, 0, v143, vcc
	v_lshl_add_u64 v[138:139], v[104:105], 0, v[142:143]
	v_mul_lo_u32 v104, s63, v138
	v_mul_lo_u32 v139, s62, v139
	v_mad_u64_u32 v[142:143], s[14:15], s62, v138, 0
	v_add3_u32 v104, v143, v139, v104
	v_sub_co_u32_e32 v137, vcc, v137, v142
	v_cmp_le_u32_e64 s[26:27], s62, v137
	s_nop 0
	v_subb_co_u32_e64 v139, s[14:15], v122, v104, vcc
	v_sub_u32_e32 v104, v122, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v122, vcc, s62, v137
	v_cmp_le_u32_e64 s[14:15], s63, v139
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cndmask_b32_e64 v141, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v139
	v_cndmask_b32_e64 v139, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v104
	v_cmp_le_u32_e64 s[26:27], s62, v122
	v_cndmask_b32_e64 v139, v141, v139, s[14:15]
	v_cndmask_b32_e64 v137, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v139
	v_cndmask_b32_e32 v104, v137, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v122, s70, v140
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v138, v104
	v_cndmask_b32_e64 v104, v138, v104, s[14:15]
	v_ashrrev_i32_e32 v138, 31, v131
	v_mov_b32_e32 v139, v138
	v_xor_b32_e32 v104, v104, v122
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[138:139]
	v_sub_u32_e32 v104, v104, v122
	v_xor_b32_e32 v139, v130, v138
	v_xad_u32 v122, v104, v138, v46
	v_xor_b32_e32 v137, v131, v138
	v_mad_u64_u32 v[130:131], s[14:15], v139, s35, 0
	v_mul_hi_u32 v104, v139, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[142:143], s[14:15], v137, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v142
	v_mad_u64_u32 v[140:141], s[14:15], v137, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v131, v143, vcc
	v_mul_lo_u32 v122, v122, s33
	s_nop 0
	v_addc_co_u32_e32 v141, vcc, 0, v141, vcc
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[140:141]
	v_mul_lo_u32 v104, s63, v130
	v_mul_lo_u32 v140, s62, v131
	v_mad_u64_u32 v[130:131], s[14:15], s62, v130, 0
	v_add3_u32 v104, v131, v140, v104
	v_sub_u32_e32 v131, v137, v104
	v_sub_co_u32_e32 v130, vcc, v139, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v139, s[14:15], s62, v130
	v_cmp_le_u32_e64 s[28:29], s62, v139
	s_nop 0
	v_subbrev_co_u32_e64 v140, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v140
	v_cndmask_b32_e64 v142, 0, -1, s[28:29]
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	v_cndmask_b32_e64 v141, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v140
	v_subb_co_u32_e32 v104, vcc, v137, v104, vcc
	s_nop 0
	v_cndmask_b32_e64 v141, v141, v142, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v141
	v_subrev_co_u32_e64 v141, s[14:15], s62, v139
	v_cmp_le_u32_e32 vcc, s63, v104
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v130
	v_cndmask_b32_e64 v131, v140, v131, s[26:27]
	v_cndmask_b32_e64 v137, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v140, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0xc06
	v_cndmask_b32_e32 v137, v137, v140, vcc
	v_cmp_ne_u32_e32 vcc, 0, v137
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v139, v141, s[26:27]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v138
	v_xor_b32_e32 v104, v104, v138
	v_sub_co_u32_e32 v130, vcc, v130, v138
	s_nop 1
	v_subb_co_u32_e32 v104, vcc, v104, v138, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v104, v104, v130, v122
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xf3f9
	buffer_load_ubyte v137, v104, s[36:39], 0 offen
	v_sub_co_u32_e64 v104, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -6, v[128:129]
	s_nop 0
	v_subb_co_u32_e64 v122, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v139, v131, v122, vcc
	v_ashrrev_i32_e32 v140, 31, v139
	v_cndmask_b32_e32 v138, v130, v104, vcc
	v_mov_b32_e32 v141, v140
	v_lshl_add_u64 v[138:139], v[138:139], 0, v[140:141]
	v_xor_b32_e32 v141, v138, v140
	v_xor_b32_e32 v122, v139, v140
	v_mad_u64_u32 v[138:139], s[14:15], v141, s35, 0
	v_mul_hi_u32 v104, v141, s34
	v_lshl_add_u64 v[138:139], v[104:105], 0, v[138:139]
	v_mad_u64_u32 v[144:145], s[14:15], v122, s34, 0
	v_add_co_u32_e32 v104, vcc, v138, v144
	v_mad_u64_u32 v[142:143], s[14:15], v122, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v139, v145, vcc
	s_nop 1
	v_addc_co_u32_e32 v143, vcc, 0, v143, vcc
	v_lshl_add_u64 v[138:139], v[104:105], 0, v[142:143]
	v_mul_lo_u32 v104, s63, v138
	v_mul_lo_u32 v139, s62, v139
	v_mad_u64_u32 v[142:143], s[14:15], s62, v138, 0
	v_add3_u32 v104, v143, v139, v104
	v_sub_co_u32_e32 v139, vcc, v141, v142
	v_cmp_le_u32_e64 s[26:27], s62, v139
	s_nop 0
	v_subb_co_u32_e64 v141, s[14:15], v122, v104, vcc
	v_sub_u32_e32 v104, v122, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v122, vcc, s62, v139
	v_cmp_le_u32_e64 s[14:15], s63, v141
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cndmask_b32_e64 v142, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v141
	v_cndmask_b32_e64 v141, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v104
	v_cmp_le_u32_e64 s[26:27], s62, v122
	v_cndmask_b32_e64 v141, v142, v141, s[14:15]
	v_cndmask_b32_e64 v139, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v141
	v_cndmask_b32_e32 v104, v139, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v122, s70, v140
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v138, v104
	v_cndmask_b32_e64 v104, v138, v104, s[14:15]
	v_ashrrev_i32_e32 v138, 31, v131
	v_mov_b32_e32 v139, v138
	v_xor_b32_e32 v104, v104, v122
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[138:139]
	v_sub_u32_e32 v104, v104, v122
	v_xor_b32_e32 v144, v130, v138
	v_xad_u32 v122, v104, v138, v46
	v_xor_b32_e32 v139, v131, v138
	v_mad_u64_u32 v[130:131], s[14:15], v144, s35, 0
	v_mul_hi_u32 v104, v144, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[142:143], s[14:15], v139, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v142
	v_mad_u64_u32 v[140:141], s[14:15], v139, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v131, v143, vcc
	v_mul_lo_u32 v122, v122, s33
	s_nop 0
	v_addc_co_u32_e32 v141, vcc, 0, v141, vcc
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[140:141]
	v_mul_lo_u32 v104, s63, v130
	v_mul_lo_u32 v140, s62, v131
	v_mad_u64_u32 v[130:131], s[14:15], s62, v130, 0
	v_add3_u32 v104, v131, v140, v104
	v_sub_u32_e32 v131, v139, v104
	v_sub_co_u32_e32 v130, vcc, v144, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v140, s[14:15], s62, v130
	v_cmp_le_u32_e64 s[28:29], s62, v140
	s_nop 0
	v_subbrev_co_u32_e64 v141, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v141
	v_cndmask_b32_e64 v143, 0, -1, s[28:29]
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	v_cndmask_b32_e64 v142, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v141
	v_subb_co_u32_e32 v104, vcc, v139, v104, vcc
	s_nop 0
	v_cndmask_b32_e64 v142, v142, v143, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v142
	v_subrev_co_u32_e64 v142, s[14:15], s62, v140
	v_cmp_le_u32_e32 vcc, s63, v104
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v130
	v_cndmask_b32_e64 v131, v141, v131, s[26:27]
	v_cndmask_b32_e64 v139, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v141, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0xc07
	v_cndmask_b32_e32 v139, v139, v141, vcc
	v_cmp_ne_u32_e32 vcc, 0, v139
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v140, v142, s[26:27]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v138
	v_xor_b32_e32 v104, v104, v138
	v_sub_co_u32_e32 v130, vcc, v130, v138
	s_nop 1
	v_subb_co_u32_e32 v104, vcc, v104, v138, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v104, v104, v130, v122
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xf3f8
	buffer_load_ubyte v138, v104, s[36:39], 0 offen
	v_sub_co_u32_e64 v104, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -7, v[128:129]
	s_nop 0
	v_subb_co_u32_e64 v122, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v141, v131, v122, vcc
	v_ashrrev_i32_e32 v142, 31, v141
	v_cndmask_b32_e32 v140, v130, v104, vcc
	v_mov_b32_e32 v143, v142
	v_lshl_add_u64 v[140:141], v[140:141], 0, v[142:143]
	v_xor_b32_e32 v139, v140, v142
	v_xor_b32_e32 v122, v141, v142
	v_mad_u64_u32 v[140:141], s[14:15], v139, s35, 0
	v_mul_hi_u32 v104, v139, s34
	v_lshl_add_u64 v[140:141], v[104:105], 0, v[140:141]
	v_mad_u64_u32 v[146:147], s[14:15], v122, s34, 0
	v_add_co_u32_e32 v104, vcc, v140, v146
	v_mad_u64_u32 v[144:145], s[14:15], v122, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v141, v147, vcc
	s_nop 1
	v_addc_co_u32_e32 v145, vcc, 0, v145, vcc
	v_lshl_add_u64 v[140:141], v[104:105], 0, v[144:145]
	v_mul_lo_u32 v104, s63, v140
	v_mul_lo_u32 v141, s62, v141
	v_mad_u64_u32 v[144:145], s[14:15], s62, v140, 0
	v_add3_u32 v104, v145, v141, v104
	v_sub_co_u32_e32 v139, vcc, v139, v144
	v_cmp_le_u32_e64 s[26:27], s62, v139
	s_nop 0
	v_subb_co_u32_e64 v141, s[14:15], v122, v104, vcc
	v_sub_u32_e32 v104, v122, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v122, vcc, s62, v139
	v_cmp_le_u32_e64 s[14:15], s63, v141
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cndmask_b32_e64 v143, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v141
	v_cndmask_b32_e64 v141, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v104
	v_cmp_le_u32_e64 s[26:27], s62, v122
	v_cndmask_b32_e64 v141, v143, v141, s[14:15]
	v_cndmask_b32_e64 v139, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v141
	v_cndmask_b32_e32 v104, v139, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v122, s70, v142
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v140, v104
	v_cndmask_b32_e64 v104, v140, v104, s[14:15]
	v_ashrrev_i32_e32 v140, 31, v131
	v_mov_b32_e32 v141, v140
	v_xor_b32_e32 v104, v104, v122
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[140:141]
	v_sub_u32_e32 v104, v104, v122
	v_xor_b32_e32 v141, v130, v140
	v_xad_u32 v122, v104, v140, v46
	v_xor_b32_e32 v139, v131, v140
	v_mad_u64_u32 v[130:131], s[14:15], v141, s35, 0
	v_mul_hi_u32 v104, v141, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[144:145], s[14:15], v139, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v144
	v_mad_u64_u32 v[142:143], s[14:15], v139, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v131, v145, vcc
	v_mul_lo_u32 v122, v122, s33
	s_nop 0
	v_addc_co_u32_e32 v143, vcc, 0, v143, vcc
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[142:143]
	v_mul_lo_u32 v104, s63, v130
	v_mul_lo_u32 v142, s62, v131
	v_mad_u64_u32 v[130:131], s[14:15], s62, v130, 0
	v_add3_u32 v104, v131, v142, v104
	v_sub_u32_e32 v131, v139, v104
	v_sub_co_u32_e32 v130, vcc, v141, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v141, s[14:15], s62, v130
	v_cmp_le_u32_e64 s[28:29], s62, v141
	s_nop 0
	v_subbrev_co_u32_e64 v142, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v142
	v_cndmask_b32_e64 v144, 0, -1, s[28:29]
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	v_cndmask_b32_e64 v143, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v142
	v_subb_co_u32_e32 v104, vcc, v139, v104, vcc
	s_nop 0
	v_cndmask_b32_e64 v143, v143, v144, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v143
	v_subrev_co_u32_e64 v143, s[14:15], s62, v141
	v_cmp_le_u32_e32 vcc, s63, v104
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v130
	v_cndmask_b32_e64 v131, v142, v131, s[26:27]
	v_cndmask_b32_e64 v139, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v142, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0xc08
	v_cndmask_b32_e32 v139, v139, v142, vcc
	v_cmp_ne_u32_e32 vcc, 0, v139
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v141, v143, s[26:27]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v140
	v_xor_b32_e32 v104, v104, v140
	v_sub_co_u32_e32 v130, vcc, v130, v140
	s_nop 1
	v_subb_co_u32_e32 v104, vcc, v104, v140, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v104, v104, v130, v122
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xf3f7
	buffer_load_ubyte v139, v104, s[36:39], 0 offen
	v_sub_co_u32_e64 v104, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -8, v[128:129]
	s_nop 0
	v_subb_co_u32_e64 v122, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v141, v131, v122, vcc
	v_ashrrev_i32_e32 v142, 31, v141
	v_cndmask_b32_e32 v140, v130, v104, vcc
	v_mov_b32_e32 v143, v142
	v_lshl_add_u64 v[140:141], v[140:141], 0, v[142:143]
	v_xor_b32_e32 v143, v140, v142
	v_xor_b32_e32 v122, v141, v142
	v_mad_u64_u32 v[140:141], s[14:15], v143, s35, 0
	v_mul_hi_u32 v104, v143, s34
	v_lshl_add_u64 v[140:141], v[104:105], 0, v[140:141]
	v_mad_u64_u32 v[146:147], s[14:15], v122, s34, 0
	v_add_co_u32_e32 v104, vcc, v140, v146
	v_mad_u64_u32 v[144:145], s[14:15], v122, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v141, v147, vcc
	s_nop 1
	v_addc_co_u32_e32 v145, vcc, 0, v145, vcc
	v_lshl_add_u64 v[140:141], v[104:105], 0, v[144:145]
	v_mul_lo_u32 v104, s63, v140
	v_mul_lo_u32 v141, s62, v141
	v_mad_u64_u32 v[144:145], s[14:15], s62, v140, 0
	v_add3_u32 v104, v145, v141, v104
	v_sub_co_u32_e32 v141, vcc, v143, v144
	v_cmp_le_u32_e64 s[26:27], s62, v141
	s_nop 0
	v_subb_co_u32_e64 v143, s[14:15], v122, v104, vcc
	v_sub_u32_e32 v104, v122, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v122, vcc, s62, v141
	v_cmp_le_u32_e64 s[14:15], s63, v143
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cndmask_b32_e64 v144, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v143
	v_cndmask_b32_e64 v143, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v104
	v_cmp_le_u32_e64 s[26:27], s62, v122
	v_cndmask_b32_e64 v143, v144, v143, s[14:15]
	v_cndmask_b32_e64 v141, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v143
	v_cndmask_b32_e32 v104, v141, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v122, s70, v142
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v140, v104
	v_cndmask_b32_e64 v104, v140, v104, s[14:15]
	v_ashrrev_i32_e32 v140, 31, v131
	v_mov_b32_e32 v141, v140
	v_xor_b32_e32 v104, v104, v122
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[140:141]
	v_sub_u32_e32 v104, v104, v122
	v_xor_b32_e32 v146, v130, v140
	v_xad_u32 v122, v104, v140, v46
	v_xor_b32_e32 v141, v131, v140
	v_mad_u64_u32 v[130:131], s[14:15], v146, s35, 0
	v_mul_hi_u32 v104, v146, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[144:145], s[14:15], v141, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v144
	v_mad_u64_u32 v[142:143], s[14:15], v141, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v131, v145, vcc
	v_mul_lo_u32 v122, v122, s33
	s_nop 0
	v_addc_co_u32_e32 v143, vcc, 0, v143, vcc
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[142:143]
	v_mul_lo_u32 v104, s63, v130
	v_mul_lo_u32 v142, s62, v131
	v_mad_u64_u32 v[130:131], s[14:15], s62, v130, 0
	v_add3_u32 v104, v131, v142, v104
	v_sub_u32_e32 v131, v141, v104
	v_sub_co_u32_e32 v130, vcc, v146, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v142, s[14:15], s62, v130
	v_cmp_le_u32_e64 s[28:29], s62, v142
	s_nop 0
	v_subbrev_co_u32_e64 v143, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v143
	v_cndmask_b32_e64 v145, 0, -1, s[28:29]
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	v_cndmask_b32_e64 v144, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v143
	v_subb_co_u32_e32 v104, vcc, v141, v104, vcc
	s_nop 0
	v_cndmask_b32_e64 v144, v144, v145, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v144
	v_subrev_co_u32_e64 v144, s[14:15], s62, v142
	v_cmp_le_u32_e32 vcc, s63, v104
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v130
	v_cndmask_b32_e64 v131, v143, v131, s[26:27]
	v_cndmask_b32_e64 v141, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v143, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0xc09
	v_cndmask_b32_e32 v141, v141, v143, vcc
	v_cmp_ne_u32_e32 vcc, 0, v141
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v142, v144, s[26:27]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v140
	v_xor_b32_e32 v104, v104, v140
	v_sub_co_u32_e32 v130, vcc, v130, v140
	s_nop 1
	v_subb_co_u32_e32 v104, vcc, v104, v140, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v104, v104, v130, v122
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xf3f6
	buffer_load_ubyte v140, v104, s[36:39], 0 offen
	v_sub_co_u32_e64 v104, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -9, v[128:129]
	s_nop 0
	v_subb_co_u32_e64 v122, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v143, v131, v122, vcc
	v_ashrrev_i32_e32 v144, 31, v143
	v_cndmask_b32_e32 v142, v130, v104, vcc
	v_mov_b32_e32 v145, v144
	v_lshl_add_u64 v[142:143], v[142:143], 0, v[144:145]
	v_xor_b32_e32 v141, v142, v144
	v_xor_b32_e32 v122, v143, v144
	v_mad_u64_u32 v[142:143], s[14:15], v141, s35, 0
	v_mul_hi_u32 v104, v141, s34
	v_lshl_add_u64 v[142:143], v[104:105], 0, v[142:143]
	v_mad_u64_u32 v[148:149], s[14:15], v122, s34, 0
	v_add_co_u32_e32 v104, vcc, v142, v148
	v_mad_u64_u32 v[146:147], s[14:15], v122, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v143, v149, vcc
	s_nop 1
	v_addc_co_u32_e32 v147, vcc, 0, v147, vcc
	v_lshl_add_u64 v[142:143], v[104:105], 0, v[146:147]
	v_mul_lo_u32 v104, s63, v142
	v_mul_lo_u32 v143, s62, v143
	v_mad_u64_u32 v[146:147], s[14:15], s62, v142, 0
	v_add3_u32 v104, v147, v143, v104
	v_sub_co_u32_e32 v141, vcc, v141, v146
	v_cmp_le_u32_e64 s[26:27], s62, v141
	s_nop 0
	v_subb_co_u32_e64 v143, s[14:15], v122, v104, vcc
	v_sub_u32_e32 v104, v122, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v122, vcc, s62, v141
	v_cmp_le_u32_e64 s[14:15], s63, v143
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cndmask_b32_e64 v145, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v143
	v_cndmask_b32_e64 v143, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v104
	v_cmp_le_u32_e64 s[26:27], s62, v122
	v_cndmask_b32_e64 v143, v145, v143, s[14:15]
	v_cndmask_b32_e64 v141, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v143
	v_cndmask_b32_e32 v104, v141, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v122, s70, v144
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v142, v104
	v_cndmask_b32_e64 v104, v142, v104, s[14:15]
	v_ashrrev_i32_e32 v142, 31, v131
	v_mov_b32_e32 v143, v142
	v_xor_b32_e32 v104, v104, v122
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[142:143]
	v_sub_u32_e32 v104, v104, v122
	v_xor_b32_e32 v143, v130, v142
	v_xad_u32 v122, v104, v142, v46
	v_xor_b32_e32 v141, v131, v142
	v_mad_u64_u32 v[130:131], s[14:15], v143, s35, 0
	v_mul_hi_u32 v104, v143, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[146:147], s[14:15], v141, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v146
	v_mad_u64_u32 v[144:145], s[14:15], v141, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v131, v147, vcc
	v_mul_lo_u32 v122, v122, s33
	s_nop 0
	v_addc_co_u32_e32 v145, vcc, 0, v145, vcc
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[144:145]
	v_mul_lo_u32 v104, s63, v130
	v_mul_lo_u32 v144, s62, v131
	v_mad_u64_u32 v[130:131], s[14:15], s62, v130, 0
	v_add3_u32 v104, v131, v144, v104
	v_sub_u32_e32 v131, v141, v104
	v_sub_co_u32_e32 v130, vcc, v143, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v143, s[14:15], s62, v130
	v_cmp_le_u32_e64 s[28:29], s62, v143
	s_nop 0
	v_subbrev_co_u32_e64 v144, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v144
	v_cndmask_b32_e64 v146, 0, -1, s[28:29]
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	v_cndmask_b32_e64 v145, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v144
	v_subb_co_u32_e32 v104, vcc, v141, v104, vcc
	s_nop 0
	v_cndmask_b32_e64 v145, v145, v146, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v145
	v_subrev_co_u32_e64 v145, s[14:15], s62, v143
	v_cmp_le_u32_e32 vcc, s63, v104
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v130
	v_cndmask_b32_e64 v131, v144, v131, s[26:27]
	v_cndmask_b32_e64 v141, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v144, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0xc0a
	v_cndmask_b32_e32 v141, v141, v144, vcc
	v_cmp_ne_u32_e32 vcc, 0, v141
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v143, v145, s[26:27]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v142
	v_xor_b32_e32 v104, v104, v142
	v_sub_co_u32_e32 v130, vcc, v130, v142
	s_nop 1
	v_subb_co_u32_e32 v104, vcc, v104, v142, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v104, v104, v130, v122
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xf3f5
	buffer_load_ubyte v141, v104, s[36:39], 0 offen
	v_sub_co_u32_e64 v104, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -10, v[128:129]
	s_nop 0
	v_subb_co_u32_e64 v122, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v143, v131, v122, vcc
	v_ashrrev_i32_e32 v144, 31, v143
	v_cndmask_b32_e32 v142, v130, v104, vcc
	v_mov_b32_e32 v145, v144
	v_lshl_add_u64 v[142:143], v[142:143], 0, v[144:145]
	v_xor_b32_e32 v145, v142, v144
	v_xor_b32_e32 v122, v143, v144
	v_mad_u64_u32 v[142:143], s[14:15], v145, s35, 0
	v_mul_hi_u32 v104, v145, s34
	v_lshl_add_u64 v[142:143], v[104:105], 0, v[142:143]
	v_mad_u64_u32 v[148:149], s[14:15], v122, s34, 0
	v_add_co_u32_e32 v104, vcc, v142, v148
	v_mad_u64_u32 v[146:147], s[14:15], v122, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v143, v149, vcc
	s_nop 1
	v_addc_co_u32_e32 v147, vcc, 0, v147, vcc
	v_lshl_add_u64 v[142:143], v[104:105], 0, v[146:147]
	v_mul_lo_u32 v104, s63, v142
	v_mul_lo_u32 v143, s62, v143
	v_mad_u64_u32 v[146:147], s[14:15], s62, v142, 0
	v_add3_u32 v104, v147, v143, v104
	v_sub_co_u32_e32 v143, vcc, v145, v146
	v_cmp_le_u32_e64 s[26:27], s62, v143
	s_nop 0
	v_subb_co_u32_e64 v145, s[14:15], v122, v104, vcc
	v_sub_u32_e32 v104, v122, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v122, vcc, s62, v143
	v_cmp_le_u32_e64 s[14:15], s63, v145
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cndmask_b32_e64 v146, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v145
	v_cndmask_b32_e64 v145, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v104
	v_cmp_le_u32_e64 s[26:27], s62, v122
	v_cndmask_b32_e64 v145, v146, v145, s[14:15]
	v_cndmask_b32_e64 v143, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v145
	v_cndmask_b32_e32 v104, v143, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v122, s70, v144
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v142, v104
	v_cndmask_b32_e64 v104, v142, v104, s[14:15]
	v_ashrrev_i32_e32 v142, 31, v131
	v_mov_b32_e32 v143, v142
	v_xor_b32_e32 v104, v104, v122
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[142:143]
	v_sub_u32_e32 v104, v104, v122
	v_xor_b32_e32 v148, v130, v142
	v_xad_u32 v122, v104, v142, v46
	v_xor_b32_e32 v143, v131, v142
	v_mad_u64_u32 v[130:131], s[14:15], v148, s35, 0
	v_mul_hi_u32 v104, v148, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[146:147], s[14:15], v143, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v146
	v_mad_u64_u32 v[144:145], s[14:15], v143, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v131, v147, vcc
	v_mul_lo_u32 v122, v122, s33
	s_nop 0
	v_addc_co_u32_e32 v145, vcc, 0, v145, vcc
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[144:145]
	v_mul_lo_u32 v104, s63, v130
	v_mul_lo_u32 v144, s62, v131
	v_mad_u64_u32 v[130:131], s[14:15], s62, v130, 0
	v_add3_u32 v104, v131, v144, v104
	v_sub_u32_e32 v131, v143, v104
	v_sub_co_u32_e32 v130, vcc, v148, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v144, s[14:15], s62, v130
	v_cmp_le_u32_e64 s[28:29], s62, v144
	s_nop 0
	v_subbrev_co_u32_e64 v145, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v145
	v_cndmask_b32_e64 v147, 0, -1, s[28:29]
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	v_cndmask_b32_e64 v146, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v145
	v_subb_co_u32_e32 v104, vcc, v143, v104, vcc
	s_nop 0
	v_cndmask_b32_e64 v146, v146, v147, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v146
	v_subrev_co_u32_e64 v146, s[14:15], s62, v144
	v_cmp_le_u32_e32 vcc, s63, v104
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v130
	v_cndmask_b32_e64 v131, v145, v131, s[26:27]
	v_cndmask_b32_e64 v143, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v145, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0xc0b
	v_cndmask_b32_e32 v143, v143, v145, vcc
	v_cmp_ne_u32_e32 vcc, 0, v143
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v144, v146, s[26:27]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v142
	v_xor_b32_e32 v104, v104, v142
	v_sub_co_u32_e32 v130, vcc, v130, v142
	s_nop 1
	v_subb_co_u32_e32 v104, vcc, v104, v142, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v104, v104, v130, v122
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xf3f4
	buffer_load_ubyte v142, v104, s[36:39], 0 offen
	v_sub_co_u32_e64 v104, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -11, v[128:129]
	s_nop 0
	v_subb_co_u32_e64 v122, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v145, v131, v122, vcc
	v_ashrrev_i32_e32 v146, 31, v145
	v_cndmask_b32_e32 v144, v130, v104, vcc
	v_mov_b32_e32 v147, v146
	v_lshl_add_u64 v[144:145], v[144:145], 0, v[146:147]
	v_xor_b32_e32 v143, v144, v146
	v_xor_b32_e32 v122, v145, v146
	v_mad_u64_u32 v[144:145], s[14:15], v143, s35, 0
	v_mul_hi_u32 v104, v143, s34
	v_lshl_add_u64 v[144:145], v[104:105], 0, v[144:145]
	v_mad_u64_u32 v[166:167], s[14:15], v122, s34, 0
	v_add_co_u32_e32 v104, vcc, v144, v166
	v_mad_u64_u32 v[148:149], s[14:15], v122, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v145, v167, vcc
	s_nop 1
	v_addc_co_u32_e32 v149, vcc, 0, v149, vcc
	v_lshl_add_u64 v[144:145], v[104:105], 0, v[148:149]
	v_mul_lo_u32 v104, s63, v144
	v_mul_lo_u32 v145, s62, v145
	v_mad_u64_u32 v[148:149], s[14:15], s62, v144, 0
	v_add3_u32 v104, v149, v145, v104
	v_sub_co_u32_e32 v143, vcc, v143, v148
	v_cmp_le_u32_e64 s[26:27], s62, v143
	s_nop 0
	v_subb_co_u32_e64 v145, s[14:15], v122, v104, vcc
	v_sub_u32_e32 v104, v122, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v122, vcc, s62, v143
	v_cmp_le_u32_e64 s[14:15], s63, v145
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cndmask_b32_e64 v147, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v145
	v_cndmask_b32_e64 v145, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v104
	v_cmp_le_u32_e64 s[26:27], s62, v122
	v_cndmask_b32_e64 v145, v147, v145, s[14:15]
	v_cndmask_b32_e64 v143, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v145
	v_cndmask_b32_e32 v104, v143, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v122, s70, v146
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v144, v104
	v_cndmask_b32_e64 v104, v144, v104, s[14:15]
	v_ashrrev_i32_e32 v144, 31, v131
	v_mov_b32_e32 v145, v144
	v_xor_b32_e32 v104, v104, v122
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[144:145]
	v_sub_u32_e32 v104, v104, v122
	v_xor_b32_e32 v145, v130, v144
	v_xad_u32 v122, v104, v144, v46
	v_xor_b32_e32 v143, v131, v144
	v_mad_u64_u32 v[130:131], s[14:15], v145, s35, 0
	v_mul_hi_u32 v104, v145, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[148:149], s[14:15], v143, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v148
	v_mad_u64_u32 v[146:147], s[14:15], v143, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v131, v149, vcc
	v_mul_lo_u32 v122, v122, s33
	s_nop 0
	v_addc_co_u32_e32 v147, vcc, 0, v147, vcc
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[146:147]
	v_mul_lo_u32 v104, s63, v130
	v_mul_lo_u32 v146, s62, v131
	v_mad_u64_u32 v[130:131], s[14:15], s62, v130, 0
	v_add3_u32 v104, v131, v146, v104
	v_sub_u32_e32 v131, v143, v104
	v_sub_co_u32_e32 v130, vcc, v145, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v145, s[14:15], s62, v130
	v_cmp_le_u32_e64 s[28:29], s62, v145
	s_nop 0
	v_subbrev_co_u32_e64 v146, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v146
	v_cndmask_b32_e64 v148, 0, -1, s[28:29]
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	v_cndmask_b32_e64 v147, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v146
	v_subb_co_u32_e32 v104, vcc, v143, v104, vcc
	s_nop 0
	v_cndmask_b32_e64 v147, v147, v148, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v147
	v_subrev_co_u32_e64 v147, s[14:15], s62, v145
	v_cmp_le_u32_e32 vcc, s63, v104
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v130
	v_cndmask_b32_e64 v131, v146, v131, s[26:27]
	v_cndmask_b32_e64 v143, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v146, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0xc0c
	v_cndmask_b32_e32 v143, v143, v146, vcc
	v_cmp_ne_u32_e32 vcc, 0, v143
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v145, v147, s[26:27]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v144
	v_xor_b32_e32 v104, v104, v144
	v_sub_co_u32_e32 v130, vcc, v130, v144
	s_nop 1
	v_subb_co_u32_e32 v104, vcc, v104, v144, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v104, v104, v130, v122
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xf3f3
	buffer_load_ubyte v143, v104, s[36:39], 0 offen
	v_sub_co_u32_e64 v104, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -12, v[128:129]
	s_nop 0
	v_subb_co_u32_e64 v122, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v145, v131, v122, vcc
	v_ashrrev_i32_e32 v146, 31, v145
	v_cndmask_b32_e32 v144, v130, v104, vcc
	v_mov_b32_e32 v147, v146
	v_lshl_add_u64 v[144:145], v[144:145], 0, v[146:147]
	v_xor_b32_e32 v147, v144, v146
	v_xor_b32_e32 v122, v145, v146
	v_mad_u64_u32 v[144:145], s[14:15], v147, s35, 0
	v_mul_hi_u32 v104, v147, s34
	v_lshl_add_u64 v[144:145], v[104:105], 0, v[144:145]
	v_mad_u64_u32 v[166:167], s[14:15], v122, s34, 0
	v_add_co_u32_e32 v104, vcc, v144, v166
	v_mad_u64_u32 v[148:149], s[14:15], v122, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v145, v167, vcc
	s_nop 1
	v_addc_co_u32_e32 v149, vcc, 0, v149, vcc
	v_lshl_add_u64 v[144:145], v[104:105], 0, v[148:149]
	v_mul_lo_u32 v104, s63, v144
	v_mul_lo_u32 v145, s62, v145
	v_mad_u64_u32 v[148:149], s[14:15], s62, v144, 0
	v_add3_u32 v104, v149, v145, v104
	v_sub_co_u32_e32 v145, vcc, v147, v148
	v_cmp_le_u32_e64 s[26:27], s62, v145
	s_nop 0
	v_subb_co_u32_e64 v147, s[14:15], v122, v104, vcc
	v_sub_u32_e32 v104, v122, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v122, vcc, s62, v145
	v_cmp_le_u32_e64 s[14:15], s63, v147
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cndmask_b32_e64 v148, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v147
	v_cndmask_b32_e64 v147, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v104
	v_cmp_le_u32_e64 s[26:27], s62, v122
	v_cndmask_b32_e64 v147, v148, v147, s[14:15]
	v_cndmask_b32_e64 v145, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v147
	v_cndmask_b32_e32 v104, v145, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v122, s70, v146
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v144, v104
	v_cndmask_b32_e64 v104, v144, v104, s[14:15]
	v_ashrrev_i32_e32 v144, 31, v131
	v_mov_b32_e32 v145, v144
	v_xor_b32_e32 v104, v104, v122
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[144:145]
	v_sub_u32_e32 v104, v104, v122
	v_xor_b32_e32 v166, v130, v144
	v_xad_u32 v122, v104, v144, v46
	v_xor_b32_e32 v145, v131, v144
	v_mad_u64_u32 v[130:131], s[14:15], v166, s35, 0
	v_mul_hi_u32 v104, v166, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[148:149], s[14:15], v145, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v148
	v_mad_u64_u32 v[146:147], s[14:15], v145, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v131, v149, vcc
	v_mul_lo_u32 v122, v122, s33
	s_nop 0
	v_addc_co_u32_e32 v147, vcc, 0, v147, vcc
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[146:147]
	v_mul_lo_u32 v104, s63, v130
	v_mul_lo_u32 v146, s62, v131
	v_mad_u64_u32 v[130:131], s[14:15], s62, v130, 0
	v_add3_u32 v104, v131, v146, v104
	v_sub_u32_e32 v131, v145, v104
	v_sub_co_u32_e32 v130, vcc, v166, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v146, s[14:15], s62, v130
	v_cmp_le_u32_e64 s[28:29], s62, v146
	s_nop 0
	v_subbrev_co_u32_e64 v147, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v147
	v_cndmask_b32_e64 v149, 0, -1, s[28:29]
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	v_cndmask_b32_e64 v148, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v147
	v_subb_co_u32_e32 v104, vcc, v145, v104, vcc
	s_nop 0
	v_cndmask_b32_e64 v148, v148, v149, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v148
	v_subrev_co_u32_e64 v148, s[14:15], s62, v146
	v_cmp_le_u32_e32 vcc, s63, v104
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v130
	v_cndmask_b32_e64 v131, v147, v131, s[26:27]
	v_cndmask_b32_e64 v145, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v147, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0xc0d
	v_cndmask_b32_e32 v145, v145, v147, vcc
	v_cmp_ne_u32_e32 vcc, 0, v145
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v146, v148, s[26:27]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v144
	v_xor_b32_e32 v104, v104, v144
	v_sub_co_u32_e32 v130, vcc, v130, v144
	s_nop 1
	v_subb_co_u32_e32 v104, vcc, v104, v144, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v104, v104, v130, v122
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xf3f2
	buffer_load_ubyte v144, v104, s[36:39], 0 offen
	v_sub_co_u32_e64 v104, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -13, v[128:129]
	s_nop 0
	v_subb_co_u32_e64 v122, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v147, v131, v122, vcc
	v_ashrrev_i32_e32 v148, 31, v147
	v_cndmask_b32_e32 v146, v130, v104, vcc
	v_mov_b32_e32 v149, v148
	v_lshl_add_u64 v[146:147], v[146:147], 0, v[148:149]
	v_xor_b32_e32 v145, v146, v148
	v_xor_b32_e32 v122, v147, v148
	v_mad_u64_u32 v[146:147], s[14:15], v145, s35, 0
	v_mul_hi_u32 v104, v145, s34
	v_lshl_add_u64 v[146:147], v[104:105], 0, v[146:147]
	v_mad_u64_u32 v[168:169], s[14:15], v122, s34, 0
	v_add_co_u32_e32 v104, vcc, v146, v168
	v_mad_u64_u32 v[166:167], s[14:15], v122, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v147, v169, vcc
	s_nop 1
	v_addc_co_u32_e32 v167, vcc, 0, v167, vcc
	v_lshl_add_u64 v[146:147], v[104:105], 0, v[166:167]
	v_mul_lo_u32 v104, s63, v146
	v_mul_lo_u32 v147, s62, v147
	v_mad_u64_u32 v[166:167], s[14:15], s62, v146, 0
	v_add3_u32 v104, v167, v147, v104
	v_sub_co_u32_e32 v145, vcc, v145, v166
	v_cmp_le_u32_e64 s[26:27], s62, v145
	s_nop 0
	v_subb_co_u32_e64 v147, s[14:15], v122, v104, vcc
	v_sub_u32_e32 v104, v122, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v122, vcc, s62, v145
	v_cmp_le_u32_e64 s[14:15], s63, v147
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cndmask_b32_e64 v149, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v147
	v_cndmask_b32_e64 v147, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v104
	v_cmp_le_u32_e64 s[26:27], s62, v122
	v_cndmask_b32_e64 v147, v149, v147, s[14:15]
	v_cndmask_b32_e64 v145, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v147
	v_cndmask_b32_e32 v104, v145, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v122, s70, v148
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v146, v104
	v_cndmask_b32_e64 v104, v146, v104, s[14:15]
	v_ashrrev_i32_e32 v146, 31, v131
	v_mov_b32_e32 v147, v146
	v_xor_b32_e32 v104, v104, v122
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[146:147]
	v_sub_u32_e32 v104, v104, v122
	v_xor_b32_e32 v147, v130, v146
	v_xad_u32 v122, v104, v146, v46
	v_xor_b32_e32 v145, v131, v146
	v_mad_u64_u32 v[130:131], s[14:15], v147, s35, 0
	v_mul_hi_u32 v104, v147, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[166:167], s[14:15], v145, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v166
	v_mad_u64_u32 v[148:149], s[14:15], v145, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v131, v167, vcc
	v_mul_lo_u32 v122, v122, s33
	s_nop 0
	v_addc_co_u32_e32 v149, vcc, 0, v149, vcc
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[148:149]
	v_mul_lo_u32 v104, s63, v130
	v_mul_lo_u32 v148, s62, v131
	v_mad_u64_u32 v[130:131], s[14:15], s62, v130, 0
	v_add3_u32 v104, v131, v148, v104
	v_sub_u32_e32 v131, v145, v104
	v_sub_co_u32_e32 v130, vcc, v147, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v147, s[14:15], s62, v130
	v_cmp_le_u32_e64 s[28:29], s62, v147
	s_nop 0
	v_subbrev_co_u32_e64 v148, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v148
	v_cndmask_b32_e64 v166, 0, -1, s[28:29]
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	v_cndmask_b32_e64 v149, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v148
	v_subb_co_u32_e32 v104, vcc, v145, v104, vcc
	s_nop 0
	v_cndmask_b32_e64 v149, v149, v166, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v149
	v_subrev_co_u32_e64 v149, s[14:15], s62, v147
	v_cmp_le_u32_e32 vcc, s63, v104
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v130
	v_cndmask_b32_e64 v131, v148, v131, s[26:27]
	v_cndmask_b32_e64 v145, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v148, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0xc0e
	v_cndmask_b32_e32 v145, v145, v148, vcc
	v_cmp_ne_u32_e32 vcc, 0, v145
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v147, v149, s[26:27]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v146
	v_xor_b32_e32 v104, v104, v146
	v_sub_co_u32_e32 v130, vcc, v130, v146
	s_nop 1
	v_subb_co_u32_e32 v104, vcc, v104, v146, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v104, v104, v130, v122
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xf3f1
	buffer_load_ubyte v145, v104, s[36:39], 0 offen
	v_sub_co_u32_e64 v104, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -14, v[128:129]
	s_nop 0
	v_subb_co_u32_e64 v122, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v147, v131, v122, vcc
	v_ashrrev_i32_e32 v148, 31, v147
	v_cndmask_b32_e32 v146, v130, v104, vcc
	v_mov_b32_e32 v149, v148
	v_lshl_add_u64 v[146:147], v[146:147], 0, v[148:149]
	v_xor_b32_e32 v149, v146, v148
	v_xor_b32_e32 v122, v147, v148
	v_mad_u64_u32 v[146:147], s[14:15], v149, s35, 0
	v_mul_hi_u32 v104, v149, s34
	v_lshl_add_u64 v[146:147], v[104:105], 0, v[146:147]
	v_mad_u64_u32 v[168:169], s[14:15], v122, s34, 0
	v_add_co_u32_e32 v104, vcc, v146, v168
	v_mad_u64_u32 v[166:167], s[14:15], v122, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v147, v169, vcc
	s_nop 1
	v_addc_co_u32_e32 v167, vcc, 0, v167, vcc
	v_lshl_add_u64 v[146:147], v[104:105], 0, v[166:167]
	v_mul_lo_u32 v104, s63, v146
	v_mul_lo_u32 v147, s62, v147
	v_mad_u64_u32 v[166:167], s[14:15], s62, v146, 0
	v_add3_u32 v104, v167, v147, v104
	v_sub_co_u32_e32 v147, vcc, v149, v166
	v_cmp_le_u32_e64 s[26:27], s62, v147
	s_nop 0
	v_subb_co_u32_e64 v149, s[14:15], v122, v104, vcc
	v_sub_u32_e32 v104, v122, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v122, vcc, s62, v147
	v_cmp_le_u32_e64 s[14:15], s63, v149
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cndmask_b32_e64 v166, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v149
	v_cndmask_b32_e64 v149, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v104
	v_cmp_le_u32_e64 s[26:27], s62, v122
	v_cndmask_b32_e64 v149, v166, v149, s[14:15]
	v_cndmask_b32_e64 v147, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v149
	v_cndmask_b32_e32 v104, v147, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v122, s70, v148
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v146, v104
	v_cndmask_b32_e64 v104, v146, v104, s[14:15]
	v_ashrrev_i32_e32 v146, 31, v131
	v_mov_b32_e32 v147, v146
	v_xor_b32_e32 v104, v104, v122
	v_lshl_add_u64 v[130:131], v[130:131], 0, v[146:147]
	v_sub_u32_e32 v104, v104, v122
	v_xor_b32_e32 v168, v130, v146
	v_xad_u32 v122, v104, v146, v46
	v_xor_b32_e32 v147, v131, v146
	v_mad_u64_u32 v[130:131], s[14:15], v168, s35, 0
	v_mul_hi_u32 v104, v168, s34
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[130:131]
	v_mad_u64_u32 v[166:167], s[14:15], v147, s34, 0
	v_add_co_u32_e32 v104, vcc, v130, v166
	v_mad_u64_u32 v[148:149], s[14:15], v147, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v131, v167, vcc
	v_mul_lo_u32 v122, v122, s33
	s_nop 0
	v_addc_co_u32_e32 v149, vcc, 0, v149, vcc
	v_lshl_add_u64 v[130:131], v[104:105], 0, v[148:149]
	v_mul_lo_u32 v104, s63, v130
	v_mul_lo_u32 v148, s62, v131
	v_mad_u64_u32 v[130:131], s[14:15], s62, v130, 0
	v_add3_u32 v104, v131, v148, v104
	v_sub_u32_e32 v131, v147, v104
	v_sub_co_u32_e32 v130, vcc, v168, v130
	s_nop 1
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, vcc
	v_subrev_co_u32_e64 v148, s[14:15], s62, v130
	v_cmp_le_u32_e64 s[28:29], s62, v148
	s_nop 0
	v_subbrev_co_u32_e64 v149, s[26:27], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v149
	v_cndmask_b32_e64 v167, 0, -1, s[28:29]
	v_subb_co_u32_e64 v131, s[14:15], v131, v81, s[14:15]
	v_cndmask_b32_e64 v166, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v149
	v_subb_co_u32_e32 v104, vcc, v147, v104, vcc
	s_nop 0
	v_cndmask_b32_e64 v166, v166, v167, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v166
	v_subrev_co_u32_e64 v166, s[14:15], s62, v148
	v_cmp_le_u32_e32 vcc, s63, v104
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s62, v130
	v_cndmask_b32_e64 v131, v149, v131, s[26:27]
	v_cndmask_b32_e64 v147, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v149, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0xc0f
	v_cndmask_b32_e32 v147, v147, v149, vcc
	v_cmp_ne_u32_e32 vcc, 0, v147
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v131, vcc
	v_cndmask_b32_e64 v131, v148, v166, s[26:27]
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_xor_b32_e32 v130, v130, v146
	v_xor_b32_e32 v104, v104, v146
	v_sub_co_u32_e32 v130, vcc, v130, v146
	s_nop 1
	v_subb_co_u32_e32 v104, vcc, v104, v146, vcc
	v_ashrrev_i32_e32 v104, 31, v104
	v_and_b32_e32 v104, s44, v104
	v_add3_u32 v104, v104, v130, v122
	v_lshl_add_u64 v[130:131], v[126:127], 0, s[14:15]
	s_movk_i32 s14, 0xf3f0
	buffer_load_ubyte v146, v104, s[36:39], 0 offen
	v_sub_co_u32_e64 v104, s[14:15], s14, v126
	v_cmp_gt_i64_e32 vcc, -15, v[128:129]
	s_nop 0
	v_subb_co_u32_e64 v122, s[14:15], -1, v127, s[14:15]
	v_cndmask_b32_e32 v127, v131, v122, vcc
	v_ashrrev_i32_e32 v128, 31, v127
	v_cndmask_b32_e32 v126, v130, v104, vcc
	v_mov_b32_e32 v129, v128
	v_lshl_add_u64 v[126:127], v[126:127], 0, v[128:129]
	v_xor_b32_e32 v129, v126, v128
	v_xor_b32_e32 v122, v127, v128
	v_mad_u64_u32 v[126:127], s[14:15], v129, s35, 0
	v_mul_hi_u32 v104, v129, s34
	v_lshl_add_u64 v[126:127], v[104:105], 0, v[126:127]
	v_mad_u64_u32 v[166:167], s[14:15], v122, s34, 0
	v_add_co_u32_e32 v104, vcc, v126, v166
	v_mad_u64_u32 v[148:149], s[14:15], v122, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v127, v167, vcc
	s_nop 1
	v_addc_co_u32_e32 v149, vcc, 0, v149, vcc
	v_lshl_add_u64 v[126:127], v[104:105], 0, v[148:149]
	v_mul_lo_u32 v104, s63, v126
	v_mul_lo_u32 v127, s62, v127
	v_mad_u64_u32 v[148:149], s[14:15], s62, v126, 0
	v_add3_u32 v104, v149, v127, v104
	v_sub_co_u32_e32 v127, vcc, v129, v148
	v_cmp_le_u32_e64 s[26:27], s62, v127
	s_nop 0
	v_subb_co_u32_e64 v129, s[14:15], v122, v104, vcc
	v_sub_u32_e32 v104, v122, v104
	v_subb_co_u32_e32 v104, vcc, v104, v81, vcc
	v_subrev_co_u32_e32 v122, vcc, s62, v127
	v_cmp_le_u32_e64 s[14:15], s63, v129
	s_nop 0
	v_subbrev_co_u32_e32 v104, vcc, 0, v104, vcc
	v_cndmask_b32_e64 v147, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s63, v129
	v_cndmask_b32_e64 v129, 0, -1, s[26:27]
	v_cmp_le_u32_e32 vcc, s63, v104
	v_cmp_le_u32_e64 s[26:27], s62, v122
	v_cndmask_b32_e64 v129, v147, v129, s[14:15]
	v_cndmask_b32_e64 v127, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v104
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_ne_u32_e64 s[14:15], 0, v129
	v_cndmask_b32_e32 v104, v127, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_xor_b32_e32 v122, s70, v128
	s_nop 0
	v_cndmask_b32_e64 v104, 1, 2, vcc
	v_add_u32_e32 v104, v126, v104
	v_cndmask_b32_e64 v104, v126, v104, s[14:15]
	v_ashrrev_i32_e32 v126, 31, v131
	v_mov_b32_e32 v127, v126
	v_xor_b32_e32 v104, v104, v122
	v_lshl_add_u64 v[128:129], v[130:131], 0, v[126:127]
	v_sub_u32_e32 v104, v104, v122
	v_xor_b32_e32 v130, v128, v126
	v_xad_u32 v122, v104, v126, v46
	v_xor_b32_e32 v127, v129, v126
	v_mad_u64_u32 v[128:129], s[14:15], v130, s35, 0
	v_mul_hi_u32 v104, v130, s34
	v_lshl_add_u64 v[148:149], v[104:105], 0, v[128:129]
	v_mad_u64_u32 v[166:167], s[14:15], v127, s34, 0
	v_add_co_u32_e32 v104, vcc, v148, v166
	v_mad_u64_u32 v[128:129], s[14:15], v127, s35, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v149, v167, vcc
	v_mul_lo_u32 v122, v122, s33
	s_nop 0
	v_addc_co_u32_e32 v129, vcc, 0, v129, vcc
	v_lshl_add_u64 v[128:129], v[104:105], 0, v[128:129]
	v_mul_lo_u32 v104, s63, v128
	v_mul_lo_u32 v129, s62, v129
	v_mad_u64_u32 v[148:149], s[14:15], s62, v128, 0
	v_add3_u32 v128, v149, v129, v104
	v_sub_u32_e32 v129, v127, v128
	v_sub_co_u32_e32 v104, vcc, v130, v148
	s_nop 1
	v_subb_co_u32_e64 v129, s[14:15], v129, v81, vcc
	v_subrev_co_u32_e64 v130, s[14:15], s62, v104
	v_cmp_le_u32_e64 s[28:29], s62, v130
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[26:27], 0, v129, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s63, v131
	v_subb_co_u32_e64 v81, s[14:15], v129, v81, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v147, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s63, v131
	v_cndmask_b32_e64 v148, 0, -1, s[28:29]
	v_subrev_co_u32_e64 v129, s[14:15], s62, v130
	v_cndmask_b32_e64 v147, v147, v148, s[26:27]
	s_nop 0
	v_subbrev_co_u32_e64 v81, s[14:15], 0, v81, s[14:15]
	v_subb_co_u32_e32 v127, vcc, v127, v128, vcc
	v_cmp_ne_u32_e64 s[26:27], 0, v147
	v_cmp_le_u32_e32 vcc, s63, v127
	v_cmp_le_u32_e64 s[14:15], s62, v104
	v_cndmask_b32_e64 v81, v131, v81, s[26:27]
	v_cndmask_b32_e64 v128, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s63, v127
	v_cndmask_b32_e64 v131, 0, -1, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v128, v128, v131, vcc
	v_cmp_ne_u32_e32 vcc, 0, v128
	s_nop 1
	v_cndmask_b32_e32 v81, v127, v81, vcc
	v_cndmask_b32_e64 v127, v130, v129, s[26:27]
	v_cndmask_b32_e32 v104, v104, v127, vcc
	v_xor_b32_e32 v104, v104, v126
	v_xor_b32_e32 v81, v81, v126
	v_sub_co_u32_e32 v104, vcc, v104, v126
	s_nop 1
	v_subb_co_u32_e32 v81, vcc, v81, v126, vcc
	v_ashrrev_i32_e32 v81, 31, v81
	v_and_b32_e32 v81, s44, v81
	v_add3_u32 v81, v81, v104, v122
	buffer_load_ubyte v81, v81, s[36:39], 0 offen
	v_lshl_add_u64 v[126:127], v[124:125], 0, s[58:59]
	v_sub_co_u32_e32 v104, vcc, 0xfffffdff, v124
	v_cmp_gt_i32_e64 s[14:15], 0, v127
	s_nop 0
	v_subb_co_u32_e32 v122, vcc, -1, v125, vcc
	v_cndmask_b32_e64 v125, v127, v122, s[14:15]
	v_cndmask_b32_e64 v124, v126, v104, s[14:15]
	v_or_b32_e32 v104, s77, v125
	v_cmp_ne_u32_e32 vcc, 0, v104
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[26:27], exec, s[14:15]
	s_cbranch_execz .LBB0_65
	s_add_u32 s14, s76, s78
	s_mov_b32 s79, s78
	s_addc_u32 s15, s77, s78
	s_xor_b64 s[28:29], s[14:15], s[78:79]
	v_cvt_f32_u32_e32 v104, s28
	v_cvt_f32_u32_e32 v122, s29
	s_sub_u32 s14, 0, s28
	s_subb_u32 s15, 0, s29
	v_ashrrev_i32_e32 v128, 31, v125
	v_fmac_f32_e32 v104, 0x4f800000, v122
	v_rcp_f32_e32 v104, v104
	v_mov_b32_e32 v129, v128
	v_lshl_add_u64 v[124:125], v[124:125], 0, v[128:129]
	v_xor_b32_e32 v129, v124, v128
	v_mul_f32_e32 v104, 0x5f7ffffc, v104
	v_mul_f32_e32 v122, 0x2f800000, v104
	v_trunc_f32_e32 v122, v122
	v_fmac_f32_e32 v104, 0xcf800000, v122
	v_cvt_u32_f32_e32 v122, v122
	v_cvt_u32_f32_e32 v104, v104
	v_readfirstlane_b32 s34, v122
	v_readfirstlane_b32 s35, v104
	s_mul_i32 s48, s14, s34
	s_mul_hi_u32 s55, s14, s35
	s_mul_i32 s49, s15, s35
	s_add_i32 s48, s55, s48
	s_add_i32 s48, s48, s49
	s_mul_i32 s62, s14, s35
	s_mul_i32 s55, s35, s48
	s_mul_hi_u32 s63, s35, s62
	s_mul_hi_u32 s49, s35, s48
	s_add_u32 s55, s63, s55
	s_addc_u32 s49, 0, s49
	s_mul_hi_u32 s71, s34, s62
	s_mul_i32 s62, s34, s62
	s_add_u32 s55, s55, s62
	s_mul_hi_u32 s63, s34, s48
	s_addc_u32 s49, s49, s71
	s_addc_u32 s55, s63, 0
	s_mul_i32 s48, s34, s48
	s_add_u32 s48, s49, s48
	s_addc_u32 s49, 0, s55
	s_add_u32 s35, s35, s48
	s_addc_u32 s34, s34, s49
	s_mul_i32 s48, s14, s34
	s_mul_hi_u32 s49, s14, s35
	s_add_i32 s48, s49, s48
	s_mul_i32 s15, s15, s35
	s_add_i32 s48, s48, s15
	s_mul_i32 s14, s14, s35
	s_mul_hi_u32 s49, s34, s14
	s_mul_i32 s55, s34, s14
	s_mul_i32 s63, s35, s48
	s_mul_hi_u32 s14, s35, s14
	s_mul_hi_u32 s62, s35, s48
	s_add_u32 s14, s14, s63
	s_addc_u32 s62, 0, s62
	s_add_u32 s14, s14, s55
	s_mul_hi_u32 s15, s34, s48
	s_addc_u32 s14, s62, s49
	s_addc_u32 s15, s15, 0
	s_mul_i32 s48, s34, s48
	s_add_u32 s14, s14, s48
	s_addc_u32 s15, 0, s15
	s_add_u32 s35, s35, s14
	s_addc_u32 s34, s34, s15
	v_xor_b32_e32 v122, v125, v128
	v_mad_u64_u32 v[124:125], s[14:15], v129, s34, 0
	v_mul_hi_u32 v104, v129, s35
	v_lshl_add_u64 v[124:125], v[104:105], 0, v[124:125]
	v_mad_u64_u32 v[148:149], s[14:15], v122, s35, 0
	v_add_co_u32_e32 v104, vcc, v124, v148
	v_mad_u64_u32 v[130:131], s[14:15], v122, s34, 0
	s_nop 0
	v_addc_co_u32_e32 v124, vcc, v125, v149, vcc
	v_mov_b32_e32 v125, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[124:125], v[124:125], 0, v[130:131]
	v_mul_lo_u32 v104, s29, v124
	v_mul_lo_u32 v147, s28, v125
	v_mad_u64_u32 v[130:131], s[14:15], s28, v124, 0
	v_add3_u32 v104, v131, v147, v104
	v_sub_u32_e32 v131, v122, v104
	v_mov_b32_e32 v147, s29
	v_sub_co_u32_e32 v129, vcc, v129, v130
	v_lshl_add_u64 v[148:149], v[124:125], 0, 1
	s_nop 0
	v_subb_co_u32_e64 v130, s[14:15], v131, v147, vcc
	v_subrev_co_u32_e64 v131, s[14:15], s28, v129
	v_subb_co_u32_e32 v104, vcc, v122, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v130, s[14:15], 0, v130, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s29, v130
	v_cmp_le_u32_e32 vcc, s29, v104
	s_nop 0
	v_cndmask_b32_e64 v147, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v131
	v_cndmask_b32_e64 v122, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v129
	v_cndmask_b32_e64 v131, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v130
	s_nop 1
	v_cndmask_b32_e64 v147, v147, v131, s[14:15]
	v_lshl_add_u64 v[130:131], v[124:125], 0, 2
	v_cndmask_b32_e64 v125, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	s_nop 1
	v_cndmask_b32_e32 v104, v122, v125, vcc
	v_cmp_ne_u32_e32 vcc, 0, v147
	s_nop 1
	v_cndmask_b32_e32 v122, v148, v130, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	s_nop 1
	v_cndmask_b32_e32 v104, v124, v122, vcc
	v_xor_b32_e32 v122, s78, v128
	v_xor_b32_e32 v104, v104, v122
	v_sub_co_u32_e32 v128, vcc, v104, v122
.LBB0_65:
	s_andn2_saveexec_b64 s[14:15], s[26:27]
	s_cbranch_execz .LBB0_67
	v_rcp_iflag_f32_e32 v104, v59
	s_sub_i32 s26, 0, s76
	v_mul_f32_e32 v104, 0x4f7ffffe, v104
	v_cvt_u32_f32_e32 v104, v104
	v_mul_lo_u32 v122, s26, v104
	v_mul_hi_u32 v122, v104, v122
	v_add_u32_e32 v104, v104, v122
	v_mul_hi_u32 v104, v124, v104
	v_mul_lo_u32 v122, v104, s76
	v_sub_u32_e32 v122, v124, v122
	v_add_u32_e32 v125, 1, v104
	v_subrev_u32_e32 v124, s76, v122
	v_cmp_le_u32_e32 vcc, s76, v122
	s_nop 1
	v_cndmask_b32_e32 v122, v122, v124, vcc
	v_cndmask_b32_e32 v104, v104, v125, vcc
	v_add_u32_e32 v124, 1, v104
	v_cmp_le_u32_e32 vcc, s76, v122
	s_nop 1
	v_cndmask_b32_e32 v128, v104, v124, vcc
.LBB0_67:
	s_or_b64 exec, exec, s[14:15]
	v_lshl_add_u64 v[124:125], v[114:115], 0, s[58:59]
	v_or_b32_e32 v104, s77, v125
	v_cmp_ne_u32_e32 vcc, 0, v104
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[28:29], exec, s[14:15]
	s_cbranch_execz .LBB0_69
	s_add_u32 s14, s76, s78
	s_mov_b32 s79, s78
	s_addc_u32 s15, s77, s78
	s_xor_b64 s[34:35], s[14:15], s[78:79]
	v_cvt_f32_u32_e32 v104, s34
	v_cvt_f32_u32_e32 v122, s35
	s_sub_u32 s14, 0, s34
	s_subb_u32 s15, 0, s35
	v_ashrrev_i32_e32 v148, 31, v125
	v_fmac_f32_e32 v104, 0x4f800000, v122
	v_rcp_f32_e32 v104, v104
	v_mov_b32_e32 v149, v148
	v_lshl_add_u64 v[124:125], v[124:125], 0, v[148:149]
	v_xor_b32_e32 v129, v124, v148
	v_mul_f32_e32 v104, 0x5f7ffffc, v104
	v_mul_f32_e32 v122, 0x2f800000, v104
	v_trunc_f32_e32 v122, v122
	v_fmac_f32_e32 v104, 0xcf800000, v122
	v_cvt_u32_f32_e32 v122, v122
	v_cvt_u32_f32_e32 v104, v104
	v_readfirstlane_b32 s26, v122
	v_readfirstlane_b32 s27, v104
	s_mul_i32 s48, s14, s26
	s_mul_hi_u32 s55, s14, s27
	s_mul_i32 s49, s15, s27
	s_add_i32 s48, s55, s48
	s_add_i32 s48, s48, s49
	s_mul_i32 s62, s14, s27
	s_mul_i32 s55, s27, s48
	s_mul_hi_u32 s63, s27, s62
	s_mul_hi_u32 s49, s27, s48
	s_add_u32 s55, s63, s55
	s_addc_u32 s49, 0, s49
	s_mul_hi_u32 s71, s26, s62
	s_mul_i32 s62, s26, s62
	s_add_u32 s55, s55, s62
	s_mul_hi_u32 s63, s26, s48
	s_addc_u32 s49, s49, s71
	s_addc_u32 s55, s63, 0
	s_mul_i32 s48, s26, s48
	s_add_u32 s48, s49, s48
	s_addc_u32 s49, 0, s55
	s_add_u32 s27, s27, s48
	s_addc_u32 s26, s26, s49
	s_mul_i32 s48, s14, s26
	s_mul_hi_u32 s49, s14, s27
	s_add_i32 s48, s49, s48
	s_mul_i32 s15, s15, s27
	s_add_i32 s48, s48, s15
	s_mul_i32 s14, s14, s27
	s_mul_hi_u32 s49, s26, s14
	s_mul_i32 s55, s26, s14
	s_mul_i32 s63, s27, s48
	s_mul_hi_u32 s14, s27, s14
	s_mul_hi_u32 s62, s27, s48
	s_add_u32 s14, s14, s63
	s_addc_u32 s62, 0, s62
	s_add_u32 s14, s14, s55
	s_mul_hi_u32 s15, s26, s48
	s_addc_u32 s14, s62, s49
	s_addc_u32 s15, s15, 0
	s_mul_i32 s48, s26, s48
	s_add_u32 s14, s14, s48
	s_addc_u32 s15, 0, s15
	s_add_u32 s27, s27, s14
	s_addc_u32 s26, s26, s15
	v_xor_b32_e32 v122, v125, v148
	v_mad_u64_u32 v[124:125], s[14:15], v129, s26, 0
	v_mul_hi_u32 v104, v129, s27
	v_lshl_add_u64 v[124:125], v[104:105], 0, v[124:125]
	v_mad_u64_u32 v[166:167], s[14:15], v122, s27, 0
	v_add_co_u32_e32 v104, vcc, v124, v166
	v_mad_u64_u32 v[130:131], s[14:15], v122, s26, 0
	s_nop 0
	v_addc_co_u32_e32 v124, vcc, v125, v167, vcc
	v_mov_b32_e32 v125, s57
	s_nop 0
	v_addc_co_u32_e32 v131, vcc, 0, v131, vcc
	v_lshl_add_u64 v[124:125], v[124:125], 0, v[130:131]
	v_mul_lo_u32 v104, s35, v124
	v_mul_lo_u32 v130, s34, v125
	v_mad_u64_u32 v[124:125], s[14:15], s34, v124, 0
	v_add3_u32 v104, v125, v130, v104
	v_sub_u32_e32 v125, v122, v104
	v_mov_b32_e32 v130, s35
	v_sub_co_u32_e32 v124, vcc, v129, v124
	s_nop 1
	v_subb_co_u32_e64 v125, s[14:15], v125, v130, vcc
	v_subrev_co_u32_e64 v129, s[14:15], s34, v124
	v_subb_co_u32_e32 v104, vcc, v122, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[26:27], 0, v125, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v131
	v_subb_co_u32_e64 v125, s[14:15], v125, v130, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v147, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v129
	v_subrev_co_u32_e64 v130, s[14:15], s34, v129
	s_nop 0
	v_cndmask_b32_e64 v149, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v131
	v_subbrev_co_u32_e64 v125, s[14:15], 0, v125, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v147, v147, v149, s[26:27]
	v_cmp_le_u32_e32 vcc, s35, v104
	v_cmp_ne_u32_e64 s[14:15], 0, v147
	s_nop 0
	v_cndmask_b32_e64 v122, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v124
	v_cndmask_b32_e64 v125, v131, v125, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v104
	s_nop 1
	v_cndmask_b32_e32 v122, v122, v131, vcc
	v_cmp_ne_u32_e32 vcc, 0, v122
	v_cndmask_b32_e64 v122, v129, v130, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v122, v124, v122, vcc
	v_cndmask_b32_e32 v104, v104, v125, vcc
	v_xor_b32_e32 v122, v122, v148
	v_xor_b32_e32 v104, v104, v148
	v_sub_co_u32_e32 v130, vcc, v122, v148
	s_nop 1
	v_subb_co_u32_e32 v131, vcc, v104, v148, vcc
.LBB0_69:
	s_andn2_saveexec_b64 s[14:15], s[28:29]
	s_cbranch_execz .LBB0_71
	v_rcp_iflag_f32_e32 v104, v59
	s_sub_i32 s26, 0, s76
	v_mul_f32_e32 v104, 0x4f7ffffe, v104
	v_cvt_u32_f32_e32 v104, v104
	v_mul_lo_u32 v122, s26, v104
	v_mul_hi_u32 v122, v104, v122
	v_add_u32_e32 v104, v104, v122
	v_mul_hi_u32 v104, v124, v104
	v_mul_lo_u32 v104, v104, s76
	v_sub_u32_e32 v104, v124, v104
	v_subrev_u32_e32 v122, s76, v104
	v_cmp_le_u32_e32 vcc, s76, v104
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v122, vcc
	v_subrev_u32_e32 v122, s76, v104
	v_cmp_le_u32_e32 vcc, s76, v104
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v122, vcc
	v_mov_b64_e32 v[130:131], v[104:105]
.LBB0_71:
	s_or_b64 exec, exec, s[14:15]
	v_ashrrev_i32_e32 v104, 31, v127
	v_xor_b32_e32 v104, v128, v104
	v_ashrrev_i32_e32 v122, 31, v131
	v_mul_lo_u32 v104, v104, s54
	v_and_b32_e32 v122, s76, v122
	v_add3_u32 v104, v104, v130, v122
	buffer_load_ubyte v130, v104, s[8:11], 0 offen
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[154:157], v[28:31], v[8:11], v123, v49 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_sub_co_u32_e32 v104, vcc, -3, v126
	v_lshl_add_u64 v[124:125], v[126:127], 0, 2
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[158:161], v[32:35], v[36:39], v123, v117 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_subb_co_u32_e32 v126, vcc, -1, v127, vcc
	v_cmp_gt_i32_e32 vcc, 0, v125
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[150:153], v[32:35], v[8:11], v123, v117 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 v[170:173], v78 offset:26624
	ds_read_b128 v[166:169], v78 offset:28672
	ds_read_b128 v[174:177], v76 offset:26624
	ds_read_b128 v[156:159], v76 offset:28672
	ds_read_b128 v[178:181], v76 offset:24576
	ds_read_b128 v[152:155], v78 offset:30720
	ds_read_b128 v[148:151], v76 offset:30720
	ds_read2st64_b32 v[122:123], v63 offset0:198 offset1:199
	v_cndmask_b32_e32 v129, v125, v126, vcc
	v_cndmask_b32_e32 v128, v124, v104, vcc
	v_or_b32_e32 v104, s77, v129
	v_cmp_ne_u32_e32 vcc, 0, v104
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[26:27], exec, s[14:15]
	s_cbranch_execz .LBB0_73
	s_add_u32 s14, s76, s78
	s_mov_b32 s79, s78
	s_addc_u32 s15, s77, s78
	s_xor_b64 s[28:29], s[14:15], s[78:79]
	v_cvt_f32_u32_e32 v104, s28
	v_cvt_f32_u32_e32 v124, s29
	s_sub_u32 s14, 0, s28
	s_subb_u32 s15, 0, s29
	v_ashrrev_i32_e32 v126, 31, v129
	v_fmac_f32_e32 v104, 0x4f800000, v124
	v_rcp_f32_e32 v104, v104
	v_mov_b32_e32 v127, v126
	v_lshl_add_u64 v[128:129], v[128:129], 0, v[126:127]
	v_xor_b32_e32 v127, v128, v126
	v_mul_f32_e32 v104, 0x5f7ffffc, v104
	v_mul_f32_e32 v124, 0x2f800000, v104
	v_trunc_f32_e32 v124, v124
	v_fmac_f32_e32 v104, 0xcf800000, v124
	v_cvt_u32_f32_e32 v124, v124
	v_cvt_u32_f32_e32 v104, v104
	v_mov_b32_e32 v147, s29
	v_readfirstlane_b32 s34, v124
	v_readfirstlane_b32 s35, v104
	s_mul_i32 s48, s14, s34
	s_mul_hi_u32 s55, s14, s35
	s_mul_i32 s49, s15, s35
	s_add_i32 s48, s55, s48
	s_add_i32 s48, s48, s49
	s_mul_i32 s62, s14, s35
	s_mul_i32 s55, s35, s48
	s_mul_hi_u32 s63, s35, s62
	s_mul_hi_u32 s49, s35, s48
	s_add_u32 s55, s63, s55
	s_addc_u32 s49, 0, s49
	s_mul_hi_u32 s71, s34, s62
	s_mul_i32 s62, s34, s62
	s_add_u32 s55, s55, s62
	s_mul_hi_u32 s63, s34, s48
	s_addc_u32 s49, s49, s71
	s_addc_u32 s55, s63, 0
	s_mul_i32 s48, s34, s48
	s_add_u32 s48, s49, s48
	s_addc_u32 s49, 0, s55
	s_add_u32 s35, s35, s48
	s_addc_u32 s34, s34, s49
	s_mul_i32 s48, s14, s34
	s_mul_hi_u32 s49, s14, s35
	s_add_i32 s48, s49, s48
	s_mul_i32 s15, s15, s35
	s_add_i32 s48, s48, s15
	s_mul_i32 s14, s14, s35
	s_mul_hi_u32 s49, s34, s14
	s_mul_i32 s55, s34, s14
	s_mul_i32 s63, s35, s48
	s_mul_hi_u32 s14, s35, s14
	s_mul_hi_u32 s62, s35, s48
	s_add_u32 s14, s14, s63
	s_addc_u32 s62, 0, s62
	s_add_u32 s14, s14, s55
	s_mul_hi_u32 s15, s34, s48
	s_addc_u32 s14, s62, s49
	s_addc_u32 s15, s15, 0
	s_mul_i32 s48, s34, s48
	s_add_u32 s14, s14, s48
	s_addc_u32 s15, 0, s15
	s_add_u32 s35, s35, s14
	s_addc_u32 s34, s34, s15
	v_xor_b32_e32 v124, v129, v126
	v_mad_u64_u32 v[128:129], s[14:15], v127, s34, 0
	v_mul_hi_u32 v104, v127, s35
	v_lshl_add_u64 v[128:129], v[104:105], 0, v[128:129]
	v_mad_u64_u32 v[182:183], s[14:15], v124, s35, 0
	v_add_co_u32_e32 v104, vcc, v128, v182
	v_mad_u64_u32 v[160:161], s[14:15], v124, s34, 0
	s_nop 0
	v_addc_co_u32_e32 v128, vcc, v129, v183, vcc
	v_mov_b32_e32 v129, s57
	s_nop 0
	v_addc_co_u32_e32 v161, vcc, 0, v161, vcc
	v_lshl_add_u64 v[128:129], v[128:129], 0, v[160:161]
	v_mul_lo_u32 v104, s29, v128
	v_mul_lo_u32 v131, s28, v129
	v_mad_u64_u32 v[160:161], s[14:15], s28, v128, 0
	v_add3_u32 v104, v161, v131, v104
	v_sub_u32_e32 v131, v124, v104
	v_sub_co_u32_e32 v127, vcc, v127, v160
	v_lshl_add_u64 v[182:183], v[128:129], 0, 1
	s_nop 0
	v_subb_co_u32_e64 v131, s[14:15], v131, v147, vcc
	v_subrev_co_u32_e64 v147, s[14:15], s28, v127
	v_subb_co_u32_e32 v104, vcc, v124, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v131, s[14:15], 0, v131, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s29, v131
	v_cmp_le_u32_e32 vcc, s29, v104
	s_nop 0
	v_cndmask_b32_e64 v160, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v147
	v_cndmask_b32_e64 v124, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v127
	v_cndmask_b32_e64 v147, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v131
	v_cndmask_b32_e64 v127, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v104
	v_cndmask_b32_e64 v131, v160, v147, s[14:15]
	v_lshl_add_u64 v[160:161], v[128:129], 0, 2
	v_cndmask_b32_e32 v104, v124, v127, vcc
	v_cmp_ne_u32_e32 vcc, 0, v131
	s_nop 1
	v_cndmask_b32_e32 v124, v182, v160, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	s_nop 1
	v_cndmask_b32_e32 v104, v128, v124, vcc
	v_xor_b32_e32 v124, s78, v126
	v_xor_b32_e32 v104, v104, v124
	v_sub_co_u32_e32 v126, vcc, v104, v124
.LBB0_73:
	s_andn2_saveexec_b64 s[14:15], s[26:27]
	s_cbranch_execz .LBB0_75
	v_rcp_iflag_f32_e32 v104, v59
	s_sub_i32 s26, 0, s76
	v_mul_f32_e32 v104, 0x4f7ffffe, v104
	v_cvt_u32_f32_e32 v104, v104
	v_mul_lo_u32 v124, s26, v104
	v_mul_hi_u32 v124, v104, v124
	v_add_u32_e32 v104, v104, v124
	v_mul_hi_u32 v104, v128, v104
	v_mul_lo_u32 v124, v104, s76
	v_sub_u32_e32 v124, v128, v124
	v_add_u32_e32 v126, 1, v104
	v_subrev_u32_e32 v127, s76, v124
	v_cmp_le_u32_e32 vcc, s76, v124
	s_nop 1
	v_cndmask_b32_e32 v124, v124, v127, vcc
	v_cndmask_b32_e32 v104, v104, v126, vcc
	v_add_u32_e32 v126, 1, v104
	v_cmp_le_u32_e32 vcc, s76, v124
	s_nop 1
	v_cndmask_b32_e32 v126, v104, v126, vcc
.LBB0_75:
	s_or_b64 exec, exec, s[14:15]
	v_lshl_add_u64 v[114:115], v[114:115], 0, s[0:1]
	v_or_b32_e32 v104, s77, v115
	v_cmp_ne_u32_e32 vcc, 0, v104
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[28:29], exec, s[14:15]
	s_cbranch_execz .LBB0_77
	s_add_u32 s14, s76, s78
	s_mov_b32 s79, s78
	s_addc_u32 s15, s77, s78
	s_xor_b64 s[34:35], s[14:15], s[78:79]
	v_cvt_f32_u32_e32 v104, s34
	v_cvt_f32_u32_e32 v124, s35
	s_sub_u32 s14, 0, s34
	s_subb_u32 s15, 0, s35
	v_ashrrev_i32_e32 v160, 31, v115
	v_fmac_f32_e32 v104, 0x4f800000, v124
	v_rcp_f32_e32 v104, v104
	v_mov_b32_e32 v161, v160
	v_lshl_add_u64 v[114:115], v[114:115], 0, v[160:161]
	v_xor_b32_e32 v127, v114, v160
	v_mul_f32_e32 v104, 0x5f7ffffc, v104
	v_mul_f32_e32 v124, 0x2f800000, v104
	v_trunc_f32_e32 v124, v124
	v_fmac_f32_e32 v104, 0xcf800000, v124
	v_cvt_u32_f32_e32 v124, v124
	v_cvt_u32_f32_e32 v104, v104
	v_readfirstlane_b32 s26, v124
	v_readfirstlane_b32 s27, v104
	s_mul_i32 s48, s14, s26
	s_mul_hi_u32 s55, s14, s27
	s_mul_i32 s49, s15, s27
	s_add_i32 s48, s55, s48
	s_add_i32 s48, s48, s49
	s_mul_i32 s62, s14, s27
	s_mul_i32 s55, s27, s48
	s_mul_hi_u32 s63, s27, s62
	s_mul_hi_u32 s49, s27, s48
	s_add_u32 s55, s63, s55
	s_addc_u32 s49, 0, s49
	s_mul_hi_u32 s71, s26, s62
	s_mul_i32 s62, s26, s62
	s_add_u32 s55, s55, s62
	s_mul_hi_u32 s63, s26, s48
	s_addc_u32 s49, s49, s71
	s_addc_u32 s55, s63, 0
	s_mul_i32 s48, s26, s48
	s_add_u32 s48, s49, s48
	s_addc_u32 s49, 0, s55
	s_add_u32 s27, s27, s48
	s_addc_u32 s26, s26, s49
	s_mul_i32 s48, s14, s26
	s_mul_hi_u32 s49, s14, s27
	s_add_i32 s48, s49, s48
	s_mul_i32 s15, s15, s27
	s_add_i32 s48, s48, s15
	s_mul_i32 s14, s14, s27
	s_mul_hi_u32 s49, s26, s14
	s_mul_i32 s55, s26, s14
	s_mul_i32 s63, s27, s48
	s_mul_hi_u32 s14, s27, s14
	s_mul_hi_u32 s62, s27, s48
	s_add_u32 s14, s14, s63
	s_addc_u32 s62, 0, s62
	s_add_u32 s14, s14, s55
	s_mul_hi_u32 s15, s26, s48
	s_addc_u32 s14, s62, s49
	s_addc_u32 s15, s15, 0
	s_mul_i32 s48, s26, s48
	s_add_u32 s14, s14, s48
	s_addc_u32 s15, 0, s15
	s_add_u32 s27, s27, s14
	s_addc_u32 s26, s26, s15
	v_xor_b32_e32 v124, v115, v160
	v_mad_u64_u32 v[114:115], s[14:15], v127, s26, 0
	v_mul_hi_u32 v104, v127, s27
	v_lshl_add_u64 v[114:115], v[104:105], 0, v[114:115]
	v_mad_u64_u32 v[182:183], s[14:15], v124, s27, 0
	v_add_co_u32_e32 v104, vcc, v114, v182
	v_mad_u64_u32 v[128:129], s[14:15], v124, s26, 0
	s_nop 0
	v_addc_co_u32_e32 v114, vcc, v115, v183, vcc
	v_mov_b32_e32 v115, s57
	s_nop 0
	v_addc_co_u32_e32 v129, vcc, 0, v129, vcc
	v_lshl_add_u64 v[114:115], v[114:115], 0, v[128:129]
	v_mul_lo_u32 v104, s35, v114
	v_mul_lo_u32 v128, s34, v115
	v_mad_u64_u32 v[114:115], s[14:15], s34, v114, 0
	v_add3_u32 v104, v115, v128, v104
	v_sub_u32_e32 v115, v124, v104
	v_mov_b32_e32 v128, s35
	v_sub_co_u32_e32 v114, vcc, v127, v114
	s_nop 1
	v_subb_co_u32_e64 v115, s[14:15], v115, v128, vcc
	v_subrev_co_u32_e64 v127, s[14:15], s34, v114
	v_subb_co_u32_e32 v104, vcc, v124, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v129, s[26:27], 0, v115, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v129
	v_subb_co_u32_e64 v115, s[14:15], v115, v128, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v131, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v127
	v_subrev_co_u32_e64 v128, s[14:15], s34, v127
	s_nop 0
	v_cndmask_b32_e64 v147, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v129
	v_subbrev_co_u32_e64 v115, s[14:15], 0, v115, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v131, v131, v147, s[26:27]
	v_cmp_le_u32_e32 vcc, s35, v104
	v_cmp_ne_u32_e64 s[14:15], 0, v131
	s_nop 0
	v_cndmask_b32_e64 v124, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v114
	v_cndmask_b32_e64 v115, v129, v115, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v129, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v104
	s_nop 1
	v_cndmask_b32_e32 v124, v124, v129, vcc
	v_cmp_ne_u32_e32 vcc, 0, v124
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v115, vcc
	v_cndmask_b32_e64 v115, v127, v128, s[14:15]
	v_cndmask_b32_e32 v114, v114, v115, vcc
	v_xor_b32_e32 v114, v114, v160
	v_xor_b32_e32 v104, v104, v160
	v_sub_co_u32_e32 v128, vcc, v114, v160
	s_nop 1
	v_subb_co_u32_e32 v129, vcc, v104, v160, vcc
.LBB0_77:
	s_andn2_saveexec_b64 s[14:15], s[28:29]
	s_cbranch_execz .LBB0_79
	v_rcp_iflag_f32_e32 v104, v59
	s_sub_i32 s26, 0, s76
	v_mul_f32_e32 v104, 0x4f7ffffe, v104
	v_cvt_u32_f32_e32 v104, v104
	v_mul_lo_u32 v115, s26, v104
	v_mul_hi_u32 v115, v104, v115
	v_add_u32_e32 v104, v104, v115
	v_mul_hi_u32 v104, v114, v104
	v_mul_lo_u32 v104, v104, s76
	v_sub_u32_e32 v104, v114, v104
	v_subrev_u32_e32 v114, s76, v104
	v_cmp_le_u32_e32 vcc, s76, v104
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v114, vcc
	v_subrev_u32_e32 v114, s76, v104
	v_cmp_le_u32_e32 vcc, s76, v104
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v114, vcc
	v_mov_b64_e32 v[128:129], v[104:105]
.LBB0_79:
	s_or_b64 exec, exec, s[14:15]
	v_ashrrev_i32_e32 v104, 31, v125
	v_xor_b32_e32 v104, v126, v104
	v_ashrrev_i32_e32 v114, 31, v129
	v_mul_lo_u32 v104, v104, s54
	v_and_b32_e32 v114, s76, v114
	v_add3_u32 v104, v104, v128, v114
	buffer_load_ubyte v114, v104, s[8:11], 0 offen
	s_waitcnt vmcnt(4) lgkmcnt(0)
	s_barrier
	v_lshl_add_u64 v[118:119], v[118:119], 0, s[16:17]
	v_add_u32_e32 v104, 0x180, v116
	v_cmp_gt_i64_e32 vcc, s[30:31], v[118:119]
	v_add_u32_e32 v115, v104, v50
	s_and_b64 s[14:15], s[2:3], vcc
	s_mov_b32 m0, s60
	v_cndmask_b32_e64 v115, -1, v115, s[14:15]
	v_add_u32_e32 v116, v104, v54
	s_and_b64 s[14:15], s[24:25], vcc
	v_cndmask_b32_e64 v116, -1, v116, s[14:15]
	buffer_load_dwordx4 v115, s[40:43], 0 offen lds
	s_mov_b32 m0, s61
	v_add_u32_e32 v115, v104, v56
	s_and_b64 s[14:15], s[18:19], vcc
	buffer_load_dwordx4 v116, s[40:43], 0 offen lds
	v_cndmask_b32_e64 v115, -1, v115, s[14:15]
	s_mov_b32 m0, s66
	v_add_u32_e32 v104, v104, v58
	s_and_b64 vcc, s[20:21], vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[162:165], v[28:31], v[20:23], v122, v49 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v115, s[40:43], 0 offen lds
	s_mov_b32 m0, s67
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[170:173], v[28:31], v[12:15], v122, v49 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[166:169], v[28:31], v[24:27], v123, v49 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[152:155], v[28:31], v[16:19], v123, v49 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v28, -1, v104, vcc
	buffer_load_dwordx4 v28, s[40:43], 0 offen lds
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[178:181], v[32:35], v[20:23], v122, v117 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[174:177], v[32:35], v[12:15], v122, v117 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[156:159], v[32:35], v[24:27], v123, v117 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[148:151], v[32:35], v[16:19], v123, v117 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 v[126:129], v78
	ds_read_b128 v[170:173], v78 offset:2048
	ds_read_b128 v[166:169], v76
	ds_read_b128 v[162:165], v76 offset:2048
	ds_read_b128 v[122:125], v78 offset:4096
	ds_read_b128 v[154:157], v78 offset:6144
	ds_read_b128 v[158:161], v76 offset:4096
	ds_read_b128 v[150:153], v76 offset:6144
	ds_read2st64_b32 v[116:117], v63 offset0:192 offset1:193
	v_sub_co_u32_e32 v18, vcc, 0xffff9fff, v120
	v_lshl_add_u64 v[16:17], v[120:121], 0, s[22:23]
	s_nop 0
	v_subb_co_u32_e32 v19, vcc, -1, v121, vcc
	v_cmp_gt_i32_e32 vcc, 0, v17
	s_nop 1
	v_cndmask_b32_e32 v29, v17, v19, vcc
	v_cndmask_b32_e32 v28, v16, v18, vcc
	v_or_b32_e32 v18, s31, v29
	v_cmp_ne_u32_e32 vcc, 0, v18
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[26:27], exec, s[14:15]
	s_cbranch_execz .LBB0_81
	s_add_u32 s14, s30, s80
	s_mov_b32 s81, s80
	s_addc_u32 s15, s31, s80
	s_xor_b64 s[28:29], s[14:15], s[80:81]
	v_cvt_f32_u32_e32 v18, s28
	v_cvt_f32_u32_e32 v19, s29
	s_sub_u32 s14, 0, s28
	s_subb_u32 s15, 0, s29
	v_fmac_f32_e32 v18, 0x4f800000, v19
	v_rcp_f32_e32 v18, v18
	s_nop 0
	v_mul_f32_e32 v18, 0x5f7ffffc, v18
	v_mul_f32_e32 v19, 0x2f800000, v18
	v_trunc_f32_e32 v19, v19
	v_fmac_f32_e32 v18, 0xcf800000, v19
	v_cvt_u32_f32_e32 v19, v19
	v_cvt_u32_f32_e32 v18, v18
	v_readfirstlane_b32 s34, v19
	v_readfirstlane_b32 s35, v18
	s_mul_i32 s48, s14, s34
	s_mul_hi_u32 s55, s14, s35
	s_mul_i32 s49, s15, s35
	s_add_i32 s48, s55, s48
	s_add_i32 s48, s48, s49
	s_mul_i32 s62, s14, s35
	s_mul_i32 s55, s35, s48
	s_mul_hi_u32 s63, s35, s62
	s_mul_hi_u32 s49, s35, s48
	s_add_u32 s55, s63, s55
	s_addc_u32 s49, 0, s49
	s_mul_hi_u32 s71, s34, s62
	s_mul_i32 s62, s34, s62
	s_add_u32 s55, s55, s62
	s_mul_hi_u32 s63, s34, s48
	s_addc_u32 s49, s49, s71
	s_addc_u32 s55, s63, 0
	s_mul_i32 s48, s34, s48
	s_add_u32 s48, s49, s48
	s_addc_u32 s49, 0, s55
	s_add_u32 s35, s35, s48
	s_addc_u32 s34, s34, s49
	s_mul_i32 s48, s14, s34
	s_mul_hi_u32 s49, s14, s35
	s_add_i32 s48, s49, s48
	s_mul_i32 s15, s15, s35
	s_add_i32 s48, s48, s15
	s_mul_i32 s14, s14, s35
	s_mul_hi_u32 s49, s34, s14
	s_mul_i32 s55, s34, s14
	s_mul_i32 s63, s35, s48
	s_mul_hi_u32 s14, s35, s14
	s_mul_hi_u32 s62, s35, s48
	s_add_u32 s14, s14, s63
	s_addc_u32 s62, 0, s62
	s_add_u32 s14, s14, s55
	s_mul_hi_u32 s15, s34, s48
	s_addc_u32 s14, s62, s49
	s_addc_u32 s15, s15, 0
	s_mul_i32 s48, s34, s48
	s_add_u32 s14, s14, s48
	v_ashrrev_i32_e32 v18, 31, v29
	s_addc_u32 s15, 0, s15
	v_mov_b32_e32 v19, v18
	s_add_u32 s35, s35, s14
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[18:19]
	s_addc_u32 s34, s34, s15
	v_xor_b32_e32 v49, v28, v18
	v_xor_b32_e32 v19, v29, v18
	v_mad_u64_u32 v[28:29], s[14:15], v49, s34, 0
	v_mul_hi_u32 v104, v49, s35
	v_lshl_add_u64 v[28:29], v[104:105], 0, v[28:29]
	v_mad_u64_u32 v[118:119], s[14:15], v19, s35, 0
	v_add_co_u32_e32 v28, vcc, v28, v118
	v_mad_u64_u32 v[30:31], s[14:15], v19, s34, 0
	s_nop 0
	v_addc_co_u32_e32 v28, vcc, v29, v119, vcc
	v_mov_b32_e32 v29, s57
	s_nop 0
	v_addc_co_u32_e32 v31, vcc, 0, v31, vcc
	v_lshl_add_u64 v[28:29], v[28:29], 0, v[30:31]
	v_mul_lo_u32 v104, s29, v28
	v_mul_lo_u32 v115, s28, v29
	v_mad_u64_u32 v[30:31], s[14:15], s28, v28, 0
	v_add3_u32 v104, v31, v115, v104
	v_sub_u32_e32 v31, v19, v104
	v_mov_b32_e32 v115, s29
	v_sub_co_u32_e32 v49, vcc, v49, v30
	v_lshl_add_u64 v[118:119], v[28:29], 0, 1
	s_nop 0
	v_subb_co_u32_e64 v30, s[14:15], v31, v115, vcc
	v_subrev_co_u32_e64 v31, s[14:15], s28, v49
	v_subb_co_u32_e32 v19, vcc, v19, v104, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v30, s[14:15], 0, v30, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s29, v30
	v_cmp_le_u32_e32 vcc, s29, v19
	v_xor_b32_e32 v18, s80, v18
	v_cndmask_b32_e64 v115, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v31
	s_nop 1
	v_cndmask_b32_e64 v31, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v30
	s_nop 1
	v_cndmask_b32_e64 v115, v115, v31, s[14:15]
	v_lshl_add_u64 v[30:31], v[28:29], 0, 2
	v_cndmask_b32_e64 v29, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v49
	s_nop 1
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v19
	s_nop 1
	v_cndmask_b32_e32 v19, v29, v31, vcc
	v_cmp_ne_u32_e32 vcc, 0, v115
	s_nop 1
	v_cndmask_b32_e32 v29, v118, v30, vcc
	v_cmp_ne_u32_e32 vcc, 0, v19
	s_nop 1
	v_cndmask_b32_e32 v19, v28, v29, vcc
	v_xor_b32_e32 v19, v19, v18
	v_sub_co_u32_e32 v18, vcc, v19, v18
.LBB0_81:
	s_andn2_saveexec_b64 s[14:15], s[26:27]
	s_cbranch_execz .LBB0_83
	v_cvt_f32_u32_e32 v18, s30
	s_sub_i32 s26, 0, s30
	v_rcp_iflag_f32_e32 v18, v18
	s_nop 0
	v_mul_f32_e32 v18, 0x4f7ffffe, v18
	v_cvt_u32_f32_e32 v18, v18
	v_mul_lo_u32 v19, s26, v18
	v_mul_hi_u32 v19, v18, v19
	v_add_u32_e32 v18, v18, v19
	v_mul_hi_u32 v18, v28, v18
	v_mul_lo_u32 v19, v18, s30
	v_sub_u32_e32 v19, v28, v19
	v_add_u32_e32 v29, 1, v18
	v_subrev_u32_e32 v28, s30, v19
	v_cmp_le_u32_e32 vcc, s30, v19
	s_nop 1
	v_cndmask_b32_e32 v19, v19, v28, vcc
	v_cndmask_b32_e32 v18, v18, v29, vcc
	v_add_u32_e32 v28, 1, v18
	v_cmp_le_u32_e32 vcc, s30, v19
	s_nop 1
	v_cndmask_b32_e32 v18, v18, v28, vcc
.LBB0_83:
	s_or_b64 exec, exec, s[14:15]
	v_or_b32_e32 v19, s31, v17
	v_cmp_ne_u32_e32 vcc, 0, v19
	v_ashrrev_i32_e32 v28, 31, v17
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[28:29], exec, s[14:15]
	s_cbranch_execz .LBB0_85
	s_add_u32 s14, s30, s80
	s_mov_b32 s81, s80
	s_addc_u32 s15, s31, s80
	s_xor_b64 s[34:35], s[14:15], s[80:81]
	v_cvt_f32_u32_e32 v19, s34
	v_cvt_f32_u32_e32 v29, s35
	s_sub_u32 s14, 0, s34
	s_subb_u32 s15, 0, s35
	v_fmac_f32_e32 v19, 0x4f800000, v29
	v_rcp_f32_e32 v19, v19
	s_nop 0
	v_mul_f32_e32 v19, 0x5f7ffffc, v19
	v_mul_f32_e32 v29, 0x2f800000, v19
	v_trunc_f32_e32 v29, v29
	v_fmac_f32_e32 v19, 0xcf800000, v29
	v_cvt_u32_f32_e32 v29, v29
	v_cvt_u32_f32_e32 v19, v19
	v_readfirstlane_b32 s26, v29
	v_readfirstlane_b32 s27, v19
	s_mul_i32 s48, s14, s26
	s_mul_hi_u32 s55, s14, s27
	s_mul_i32 s49, s15, s27
	s_add_i32 s48, s55, s48
	s_add_i32 s48, s48, s49
	s_mul_i32 s62, s14, s27
	s_mul_i32 s55, s27, s48
	s_mul_hi_u32 s63, s27, s62
	s_mul_hi_u32 s49, s27, s48
	s_add_u32 s55, s63, s55
	s_addc_u32 s49, 0, s49
	s_mul_hi_u32 s71, s26, s62
	s_mul_i32 s62, s26, s62
	s_add_u32 s55, s55, s62
	s_mul_hi_u32 s63, s26, s48
	s_addc_u32 s49, s49, s71
	s_addc_u32 s55, s63, 0
	s_mul_i32 s48, s26, s48
	s_add_u32 s48, s49, s48
	s_addc_u32 s49, 0, s55
	s_add_u32 s27, s27, s48
	s_addc_u32 s26, s26, s49
	s_mul_i32 s48, s14, s26
	s_mul_hi_u32 s49, s14, s27
	s_add_i32 s48, s49, s48
	s_mul_i32 s15, s15, s27
	s_add_i32 s48, s48, s15
	s_mul_i32 s14, s14, s27
	s_mul_hi_u32 s49, s26, s14
	s_mul_i32 s55, s26, s14
	s_mul_i32 s63, s27, s48
	s_mul_hi_u32 s14, s27, s14
	s_mul_hi_u32 s62, s27, s48
	s_add_u32 s14, s14, s63
	s_addc_u32 s62, 0, s62
	s_add_u32 s14, s14, s55
	s_mul_hi_u32 s15, s26, s48
	s_addc_u32 s14, s62, s49
	s_addc_u32 s15, s15, 0
	s_mul_i32 s48, s26, s48
	s_add_u32 s14, s14, s48
	s_addc_u32 s15, 0, s15
	v_mov_b32_e32 v29, v28
	s_add_u32 s27, s27, s14
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[28:29]
	s_addc_u32 s26, s26, s15
	v_xor_b32_e32 v29, v16, v28
	v_xor_b32_e32 v19, v17, v28
	v_mad_u64_u32 v[16:17], s[14:15], v29, s26, 0
	v_mul_hi_u32 v104, v29, s27
	v_lshl_add_u64 v[16:17], v[104:105], 0, v[16:17]
	v_mad_u64_u32 v[118:119], s[14:15], v19, s27, 0
	v_add_co_u32_e32 v16, vcc, v16, v118
	v_mad_u64_u32 v[30:31], s[14:15], v19, s26, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v119, vcc
	v_mov_b32_e32 v17, s57
	s_nop 0
	v_addc_co_u32_e32 v31, vcc, 0, v31, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[30:31]
	v_mul_lo_u32 v30, s35, v16
	v_mul_lo_u32 v31, s34, v17
	v_mad_u64_u32 v[16:17], s[14:15], s34, v16, 0
	v_add3_u32 v17, v17, v31, v30
	v_sub_u32_e32 v30, v19, v17
	v_mov_b32_e32 v31, s35
	v_sub_co_u32_e32 v16, vcc, v29, v16
	s_nop 1
	v_subb_co_u32_e64 v29, s[14:15], v30, v31, vcc
	v_subrev_co_u32_e64 v30, s[14:15], s34, v16
	v_subb_co_u32_e32 v17, vcc, v19, v17, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v49, s[26:27], 0, v29, s[14:15]
	v_cmp_le_u32_e64 s[26:27], s35, v49
	v_subb_co_u32_e64 v29, s[14:15], v29, v31, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v104, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s34, v30
	v_subrev_co_u32_e64 v31, s[14:15], s34, v30
	s_nop 0
	v_cndmask_b32_e64 v115, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s35, v49
	v_subbrev_co_u32_e64 v29, s[14:15], 0, v29, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v104, v104, v115, s[26:27]
	v_cmp_le_u32_e32 vcc, s35, v17
	v_cmp_ne_u32_e64 s[14:15], 0, v104
	s_nop 0
	v_cndmask_b32_e64 v19, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v16
	v_cndmask_b32_e64 v29, v49, v29, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v17
	s_nop 1
	v_cndmask_b32_e32 v19, v19, v49, vcc
	v_cmp_ne_u32_e32 vcc, 0, v19
	v_cndmask_b32_e64 v19, v30, v31, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v16, v16, v19, vcc
	v_cndmask_b32_e32 v17, v17, v29, vcc
	v_xor_b32_e32 v16, v16, v28
	v_xor_b32_e32 v17, v17, v28
	v_sub_co_u32_e32 v30, vcc, v16, v28
	s_nop 1
	v_subb_co_u32_e32 v31, vcc, v17, v28, vcc
.LBB0_85:
	s_andn2_saveexec_b64 s[14:15], s[28:29]
	s_cbranch_execz .LBB0_38
	v_cvt_f32_u32_e32 v17, s30
	s_sub_i32 s26, 0, s30
	v_rcp_iflag_f32_e32 v17, v17
	s_nop 0
	v_mul_f32_e32 v17, 0x4f7ffffe, v17
	v_cvt_u32_f32_e32 v17, v17
	v_mul_lo_u32 v19, s26, v17
	v_mul_hi_u32 v19, v17, v19
	v_add_u32_e32 v17, v17, v19
	v_mul_hi_u32 v17, v16, v17
	v_mul_lo_u32 v17, v17, s30
	v_sub_u32_e32 v16, v16, v17
	v_subrev_u32_e32 v17, s30, v16
	v_cmp_le_u32_e32 vcc, s30, v16
	s_nop 1
	v_cndmask_b32_e32 v16, v16, v17, vcc
	v_subrev_u32_e32 v17, s30, v16
	v_cmp_le_u32_e32 vcc, s30, v16
	s_nop 1
	v_cndmask_b32_e32 v104, v16, v17, vcc
	v_mov_b64_e32 v[30:31], v[104:105]
	s_branch .LBB0_38
.LBB0_87:
	v_mov_b32_e32 v7, 0
	v_mov_b32_e32 v6, v105
	v_mov_b32_e32 v5, v105
	v_mov_b32_e32 v4, v105
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, v105
	v_mov_b32_e32 v1, v105
	v_mov_b32_e32 v0, v105
	v_mov_b32_e32 v39, 0
	v_mov_b32_e32 v38, v105
	v_mov_b32_e32 v37, v105
	v_mov_b32_e32 v36, v105
	v_mov_b32_e32 v11, 0
	v_mov_b32_e32 v10, v105
	v_mov_b32_e32 v9, v105
	v_mov_b32_e32 v8, v105
	v_mov_b32_e32 v23, 0
	v_mov_b32_e32 v22, v105
	v_mov_b32_e32 v21, v105
	v_mov_b32_e32 v20, v105
	v_mov_b32_e32 v15, 0
	v_mov_b32_e32 v14, v105
	v_mov_b32_e32 v13, v105
	v_mov_b32_e32 v12, v105
	v_mov_b32_e32 v27, 0
	v_mov_b32_e32 v26, v105
	v_mov_b32_e32 v25, v105
	v_mov_b32_e32 v24, v105
	v_mov_b32_e32 v35, 0
	v_mov_b32_e32 v34, v105
	v_mov_b32_e32 v33, v105
	v_mov_b32_e32 v32, v105
	s_branch .LBB0_89
.LBB0_88:
	v_readlane_b32 s84, v202, 34
	v_readlane_b32 s85, v202, 35
.LBB0_89:
	s_mul_i32 s3, s75, 0xaaaaaaab
	s_mul_hi_u32 s14, s74, 0xaaaaaaab
	s_mul_hi_u32 s2, s75, 0xaaaaaaab
	s_add_u32 s3, s3, s14
	s_mul_i32 s1, s74, 0x2aaaaaaa
	s_addc_u32 s2, s2, 0
	s_mul_hi_u32 s0, s74, 0x2aaaaaaa
	s_add_u32 s1, s1, s3
	s_addc_u32 s0, s0, 0
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, 0, 0
	s_mul_i32 s3, s75, 0x2aaaaaaa
	s_mul_hi_u32 s2, s75, 0x2aaaaaaa
	s_add_u32 s0, s3, s0
	v_readlane_b32 s14, v202, 30
	s_addc_u32 s1, s2, s1
	s_mul_i32 s2, s14, 0x2aaaaaaa
	s_mul_hi_u32 s3, s14, 0xaaaaaaab
	s_add_i32 s2, s3, s2
	s_mul_i32 s3, s14, 0xaaaaaaab
	s_add_i32 s2, s2, s3
	s_add_u32 s0, s0, s3
	s_addc_u32 s1, s1, s2
	s_ashr_i64 s[2:3], s[0:1], 8
	s_lshr_b32 s0, s1, 31
	s_add_u32 s0, s2, s0
	s_addc_u32 s1, s3, 0
	s_xor_b64 s[0:1], s[0:1], s[72:73]
	s_mul_i32 s2, s1, 0x3000
	s_mul_hi_u32 s3, s0, 0x3000
	s_add_i32 s3, s3, s2
	s_mul_i32 s2, s0, 0x3000
	v_mov_b32_e32 v49, s3
	v_or_b32_e32 v48, s2, v64
	v_lshl_add_u64 v[50:51], v[48:49], 0, v[70:71]
	s_movk_i32 s2, 0xf800
	v_lshl_add_u64 v[52:53], v[102:103], 0, v[50:51]
	s_mov_b32 s3, -1
	v_lshl_add_u64 v[54:55], v[52:53], 0, s[2:3]
	s_movk_i32 s2, 0x7ff
	v_sub_co_u32_e32 v47, vcc, s2, v52
	s_add_u32 s2, s44, s70
	s_mov_b32 s71, s70
	s_addc_u32 s3, s45, s70
	s_xor_b64 s[16:17], s[2:3], s[70:71]
	v_cvt_f32_u32_e32 v58, s16
	v_cvt_f32_u32_e32 v60, s17
	v_subb_co_u32_e32 v48, vcc, 0, v53, vcc
	v_cmp_gt_i32_e32 vcc, 0, v55
	v_fmac_f32_e32 v58, 0x4f800000, v60
	s_sub_u32 s2, 0, s16
	v_cndmask_b32_e32 v49, v55, v48, vcc
	v_cndmask_b32_e32 v48, v54, v47, vcc
	v_rcp_f32_e32 v47, v58
	v_ashrrev_i32_e32 v56, 31, v49
	v_mov_b32_e32 v57, v56
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[56:57]
	v_mul_f32_e32 v47, 0x5f7ffffc, v47
	v_xor_b32_e32 v58, v48, v56
	v_mul_f32_e32 v48, 0x2f800000, v47
	v_trunc_f32_e32 v48, v48
	v_fmac_f32_e32 v47, 0xcf800000, v48
	v_cvt_u32_f32_e32 v47, v47
	v_cvt_u32_f32_e32 v48, v48
	s_subb_u32 s3, 0, s17
	v_xor_b32_e32 v57, v49, v56
	v_readfirstlane_b32 s14, v47
	v_readfirstlane_b32 s20, v48
	s_mul_hi_u32 s19, s2, s14
	s_mul_i32 s21, s2, s20
	s_mul_i32 s15, s3, s14
	s_add_i32 s19, s19, s21
	s_add_i32 s19, s19, s15
	s_mul_i32 s22, s2, s14
	s_mul_i32 s21, s14, s19
	s_mul_hi_u32 s23, s14, s22
	s_mul_hi_u32 s15, s14, s19
	s_add_u32 s21, s23, s21
	s_addc_u32 s15, 0, s15
	s_mul_hi_u32 s24, s20, s22
	s_mul_i32 s22, s20, s22
	s_add_u32 s21, s21, s22
	s_mul_hi_u32 s23, s20, s19
	s_addc_u32 s15, s15, s24
	s_addc_u32 s21, s23, 0
	s_mul_i32 s19, s20, s19
	s_add_u32 s15, s15, s19
	s_addc_u32 s19, 0, s21
	s_add_u32 s14, s14, s15
	s_addc_u32 s15, s20, s19
	s_mul_i32 s19, s2, s15
	s_mul_hi_u32 s20, s2, s14
	s_add_i32 s19, s20, s19
	s_mul_i32 s3, s3, s14
	s_add_i32 s19, s19, s3
	s_mul_i32 s2, s2, s14
	s_mul_hi_u32 s20, s15, s2
	s_mul_i32 s21, s15, s2
	s_mul_i32 s23, s14, s19
	s_mul_hi_u32 s2, s14, s2
	s_mul_hi_u32 s22, s14, s19
	s_add_u32 s2, s2, s23
	s_addc_u32 s22, 0, s22
	s_add_u32 s2, s2, s21
	s_mul_hi_u32 s3, s15, s19
	s_addc_u32 s2, s22, s20
	s_addc_u32 s3, s3, 0
	s_mul_i32 s19, s15, s19
	s_add_u32 s2, s2, s19
	s_addc_u32 s3, 0, s3
	s_add_u32 s19, s14, s2
	s_addc_u32 s20, s15, s3
	v_mad_u64_u32 v[60:61], s[2:3], v58, s20, 0
	v_mul_hi_u32 v48, v58, s19
	v_mov_b32_e32 v49, 0
	v_lshl_add_u64 v[60:61], v[48:49], 0, v[60:61]
	v_mad_u64_u32 v[74:75], s[2:3], v57, s19, 0
	v_add_co_u32_e32 v47, vcc, v60, v74
	s_mov_b32 s18, 0
	v_mad_u64_u32 v[70:71], s[2:3], v57, s20, 0
	v_addc_co_u32_e32 v60, vcc, v61, v75, vcc
	v_mov_b32_e32 v61, s18
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[60:61], v[60:61], 0, v[70:71]
	v_mul_lo_u32 v47, s17, v60
	v_mul_lo_u32 v48, s16, v61
	v_mad_u64_u32 v[70:71], s[2:3], s16, v60, 0
	v_add3_u32 v48, v71, v48, v47
	v_sub_u32_e32 v61, v57, v48
	v_mov_b32_e32 v47, s17
	v_sub_co_u32_e32 v58, vcc, v58, v70
	v_xor_b32_e32 v56, s70, v56
	s_nop 0
	v_subb_co_u32_e64 v61, s[2:3], v61, v47, vcc
	v_subrev_co_u32_e64 v65, s[2:3], s16, v58
	v_subb_co_u32_e32 v48, vcc, v57, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v61, s[2:3], 0, v61, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v61
	v_cmp_le_u32_e32 vcc, s17, v48
	v_lshl_add_u64 v[50:51], v[50:51], 0, v[100:101]
	v_cndmask_b32_e64 v70, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v65
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v58
	v_cndmask_b32_e64 v65, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v61
	v_cndmask_b32_e64 v58, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v61, v70, v65, s[2:3]
	v_add_u32_e32 v65, 2, v60
	v_add_u32_e32 v70, 1, v60
	v_cmp_ne_u32_e64 s[2:3], 0, v61
	v_cndmask_b32_e32 v48, v57, v58, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v61, v70, v65, s[2:3]
	v_and_b32_e32 v83, 0x200, v64
	v_cndmask_b32_e32 v48, v60, v61, vcc
	v_xor_b32_e32 v48, v48, v56
	v_sub_u32_e32 v48, v48, v56
	v_ashrrev_i32_e32 v56, 31, v55
	v_mov_b32_e32 v57, v56
	v_lshl_add_u64 v[54:55], v[54:55], 0, v[56:57]
	v_xor_b32_e32 v65, v54, v56
	v_xad_u32 v58, v48, v56, v46
	v_xor_b32_e32 v57, v55, v56
	v_mad_u64_u32 v[54:55], s[2:3], v65, s20, 0
	v_mul_hi_u32 v48, v65, s19
	v_lshl_add_u64 v[54:55], v[48:49], 0, v[54:55]
	v_mad_u64_u32 v[70:71], s[2:3], v57, s19, 0
	v_add_co_u32_e32 v48, vcc, v54, v70
	v_mad_u64_u32 v[60:61], s[2:3], v57, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v54, vcc, v55, v71, vcc
	v_mov_b32_e32 v55, s18
	s_nop 0
	v_addc_co_u32_e32 v61, vcc, 0, v61, vcc
	v_lshl_add_u64 v[54:55], v[54:55], 0, v[60:61]
	v_mul_lo_u32 v48, s17, v54
	v_mul_lo_u32 v60, s16, v55
	v_mad_u64_u32 v[54:55], s[2:3], s16, v54, 0
	v_add3_u32 v48, v55, v60, v48
	v_sub_u32_e32 v55, v57, v48
	v_sub_co_u32_e32 v54, vcc, v65, v54
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_nop 0
	v_subb_co_u32_e64 v55, s[2:3], v55, v47, vcc
	v_subrev_co_u32_e64 v60, s[2:3], s16, v54
	v_subb_co_u32_e32 v48, vcc, v57, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v61, s[14:15], 0, v55, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v61
	v_subb_co_u32_e64 v55, s[2:3], v55, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v60
	v_cmp_le_u32_e32 vcc, s17, v48
	s_barrier
	v_cndmask_b32_e64 v70, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v61
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v54
	v_cndmask_b32_e64 v65, v65, v70, s[14:15]
	v_subrev_co_u32_e64 v70, s[2:3], s16, v60
	s_mulk_i32 s1, 0x600
	s_nop 0
	v_subbrev_co_u32_e64 v55, s[2:3], 0, v55, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v65
	s_nop 1
	v_cndmask_b32_e64 v55, v61, v55, s[2:3]
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v57, v57, v61, vcc
	v_cmp_ne_u32_e32 vcc, 0, v57
	s_nop 1
	v_cndmask_b32_e32 v48, v48, v55, vcc
	v_cndmask_b32_e64 v55, v60, v70, s[2:3]
	v_cndmask_b32_e32 v54, v54, v55, vcc
	v_xor_b32_e32 v54, v54, v56
	v_xor_b32_e32 v48, v48, v56
	v_sub_co_u32_e32 v54, vcc, v54, v56
	v_mul_lo_u32 v55, v58, s33
	s_nop 0
	v_subb_co_u32_e32 v48, vcc, v48, v56, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v58, v48, v54, v55
	s_movk_i32 s2, 0xfa00
	v_mov_b32_e32 v48, 0xfffff800
	v_bitop3_b32 v54, v43, s2, v48 bitop3:0xc8
	v_ashrrev_i32_e32 v55, 31, v54
	v_lshl_add_u64 v[54:55], v[50:51], 0, v[54:55]
	v_lshl_add_u64 v[54:55], v[54:55], 0, v[102:103]
	v_sub_co_u32_e32 v48, vcc, -2, v54
	v_lshl_add_u64 v[56:57], v[54:55], 0, 1
	s_nop 0
	v_subb_co_u32_e32 v60, vcc, -1, v55, vcc
	v_cmp_gt_i32_e32 vcc, 0, v57
	s_nop 1
	v_cndmask_b32_e32 v61, v57, v60, vcc
	v_ashrrev_i32_e32 v70, 31, v61
	v_cndmask_b32_e32 v60, v56, v48, vcc
	v_mov_b32_e32 v71, v70
	v_lshl_add_u64 v[60:61], v[60:61], 0, v[70:71]
	v_xor_b32_e32 v71, v60, v70
	v_xor_b32_e32 v65, v61, v70
	v_mad_u64_u32 v[60:61], s[2:3], v71, s20, 0
	v_mul_hi_u32 v48, v71, s19
	v_lshl_add_u64 v[60:61], v[48:49], 0, v[60:61]
	v_mad_u64_u32 v[100:101], s[2:3], v65, s19, 0
	v_add_co_u32_e32 v48, vcc, v60, v100
	v_mad_u64_u32 v[74:75], s[2:3], v65, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v60, vcc, v61, v101, vcc
	v_mov_b32_e32 v61, s18
	s_nop 0
	v_addc_co_u32_e32 v75, vcc, 0, v75, vcc
	v_lshl_add_u64 v[60:61], v[60:61], 0, v[74:75]
	v_mul_lo_u32 v48, s17, v60
	v_mul_lo_u32 v61, s16, v61
	v_mad_u64_u32 v[74:75], s[2:3], s16, v60, 0
	v_add3_u32 v48, v75, v61, v48
	v_sub_u32_e32 v61, v65, v48
	v_sub_co_u32_e32 v71, vcc, v71, v74
	s_nop 1
	v_subb_co_u32_e64 v61, s[2:3], v61, v47, vcc
	v_subrev_co_u32_e64 v74, s[2:3], s16, v71
	v_subb_co_u32_e32 v48, vcc, v65, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v61, s[2:3], 0, v61, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v61
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v75, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v74
	v_cndmask_b32_e64 v65, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v71
	v_cndmask_b32_e64 v74, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v61
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v61, v75, v74, s[2:3]
	v_add_u32_e32 v74, 2, v60
	v_add_u32_e32 v75, 1, v60
	v_cmp_ne_u32_e64 s[2:3], 0, v61
	v_cndmask_b32_e32 v48, v65, v71, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v61, v75, v74, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v48, v60, v61, vcc
	v_xor_b32_e32 v60, s70, v70
	v_xor_b32_e32 v48, v48, v60
	v_sub_u32_e32 v48, v48, v60
	v_ashrrev_i32_e32 v60, 31, v57
	v_mov_b32_e32 v61, v60
	v_lshl_add_u64 v[56:57], v[56:57], 0, v[60:61]
	v_xor_b32_e32 v77, v56, v60
	v_xad_u32 v65, v48, v60, v46
	v_xor_b32_e32 v61, v57, v60
	v_mad_u64_u32 v[56:57], s[2:3], v77, s20, 0
	v_mul_hi_u32 v48, v77, s19
	v_lshl_add_u64 v[56:57], v[48:49], 0, v[56:57]
	v_mad_u64_u32 v[74:75], s[2:3], v61, s19, 0
	v_add_co_u32_e32 v48, vcc, v56, v74
	v_mad_u64_u32 v[70:71], s[2:3], v61, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v56, vcc, v57, v75, vcc
	v_mov_b32_e32 v57, s18
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[56:57], v[56:57], 0, v[70:71]
	v_mul_lo_u32 v48, s17, v56
	v_mul_lo_u32 v70, s16, v57
	v_mad_u64_u32 v[56:57], s[2:3], s16, v56, 0
	v_add3_u32 v48, v57, v70, v48
	v_sub_u32_e32 v57, v61, v48
	v_sub_co_u32_e32 v56, vcc, v77, v56
	s_nop 1
	v_subb_co_u32_e64 v57, s[2:3], v57, v47, vcc
	v_subrev_co_u32_e64 v70, s[2:3], s16, v56
	v_subb_co_u32_e32 v48, vcc, v61, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v71, s[14:15], 0, v57, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v71
	v_subb_co_u32_e64 v57, s[2:3], v57, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v74, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v70
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v75, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v71
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v56
	v_cndmask_b32_e64 v74, v74, v75, s[14:15]
	v_subrev_co_u32_e64 v75, s[2:3], s16, v70
	s_nop 1
	v_subbrev_co_u32_e64 v57, s[2:3], 0, v57, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v74
	s_nop 1
	v_cndmask_b32_e64 v57, v71, v57, s[2:3]
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v61, v61, v71, vcc
	v_cmp_ne_u32_e32 vcc, 0, v61
	s_nop 1
	v_cndmask_b32_e32 v48, v48, v57, vcc
	v_cndmask_b32_e64 v57, v70, v75, s[2:3]
	v_cndmask_b32_e32 v56, v56, v57, vcc
	v_xor_b32_e32 v56, v56, v60
	v_xor_b32_e32 v48, v48, v60
	v_sub_co_u32_e32 v56, vcc, v56, v60
	v_mul_lo_u32 v57, v65, s33
	s_nop 0
	v_subb_co_u32_e32 v48, vcc, v48, v60, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v60, v48, v56, v57
	v_or_b32_e32 v56, 0xfffff800, v83
	v_ashrrev_i32_e32 v57, 31, v56
	v_lshl_add_u64 v[56:57], v[50:51], 0, v[56:57]
	v_lshl_add_u64 v[56:57], v[56:57], 0, v[102:103]
	v_sub_co_u32_e32 v48, vcc, -3, v56
	v_lshl_add_u64 v[64:65], v[56:57], 0, 2
	s_nop 0
	v_subb_co_u32_e32 v61, vcc, -1, v57, vcc
	v_cmp_gt_i32_e32 vcc, 0, v65
	s_nop 1
	v_cndmask_b32_e32 v71, v65, v61, vcc
	v_ashrrev_i32_e32 v74, 31, v71
	v_cndmask_b32_e32 v70, v64, v48, vcc
	v_mov_b32_e32 v75, v74
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[74:75]
	v_xor_b32_e32 v75, v70, v74
	v_xor_b32_e32 v61, v71, v74
	v_mad_u64_u32 v[70:71], s[2:3], v75, s20, 0
	v_mul_hi_u32 v48, v75, s19
	v_lshl_add_u64 v[70:71], v[48:49], 0, v[70:71]
	v_mad_u64_u32 v[104:105], s[2:3], v61, s19, 0
	v_add_co_u32_e32 v48, vcc, v70, v104
	v_mad_u64_u32 v[100:101], s[2:3], v61, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v71, v105, vcc
	v_mov_b32_e32 v71, s18
	s_nop 0
	v_addc_co_u32_e32 v101, vcc, 0, v101, vcc
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[100:101]
	v_mul_lo_u32 v48, s17, v70
	v_mul_lo_u32 v71, s16, v71
	v_mad_u64_u32 v[100:101], s[2:3], s16, v70, 0
	v_add3_u32 v48, v101, v71, v48
	v_sub_u32_e32 v71, v61, v48
	v_sub_co_u32_e32 v75, vcc, v75, v100
	s_nop 1
	v_subb_co_u32_e64 v71, s[2:3], v71, v47, vcc
	v_subrev_co_u32_e64 v77, s[2:3], s16, v75
	v_subb_co_u32_e32 v48, vcc, v61, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v71, s[2:3], 0, v71, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v71
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v77
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v75
	v_cndmask_b32_e64 v77, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v71
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v71, v79, v77, s[2:3]
	v_add_u32_e32 v77, 2, v70
	v_add_u32_e32 v79, 1, v70
	v_cmp_ne_u32_e64 s[2:3], 0, v71
	v_cndmask_b32_e32 v48, v61, v75, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v71, v79, v77, s[2:3]
	v_xor_b32_e32 v61, s70, v74
	v_cndmask_b32_e32 v48, v70, v71, vcc
	v_ashrrev_i32_e32 v70, 31, v65
	v_mov_b32_e32 v71, v70
	v_xor_b32_e32 v48, v48, v61
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[70:71]
	v_sub_u32_e32 v48, v48, v61
	v_xor_b32_e32 v77, v64, v70
	v_xad_u32 v61, v48, v70, v46
	v_xor_b32_e32 v71, v65, v70
	v_mad_u64_u32 v[64:65], s[2:3], v77, s20, 0
	v_mul_hi_u32 v48, v77, s19
	v_lshl_add_u64 v[64:65], v[48:49], 0, v[64:65]
	v_mad_u64_u32 v[100:101], s[2:3], v71, s19, 0
	v_add_co_u32_e32 v48, vcc, v64, v100
	v_mad_u64_u32 v[74:75], s[2:3], v71, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v64, vcc, v65, v101, vcc
	v_mov_b32_e32 v65, s18
	s_nop 0
	v_addc_co_u32_e32 v75, vcc, 0, v75, vcc
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[74:75]
	v_mul_lo_u32 v48, s17, v64
	v_mul_lo_u32 v74, s16, v65
	v_mad_u64_u32 v[64:65], s[2:3], s16, v64, 0
	v_add3_u32 v48, v65, v74, v48
	v_sub_u32_e32 v65, v71, v48
	v_sub_co_u32_e32 v64, vcc, v77, v64
	v_mul_lo_u32 v61, v61, s33
	s_nop 0
	v_subb_co_u32_e64 v65, s[2:3], v65, v47, vcc
	v_subrev_co_u32_e64 v74, s[2:3], s16, v64
	v_subb_co_u32_e32 v48, vcc, v71, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v75, s[14:15], 0, v65, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v75
	v_subb_co_u32_e64 v65, s[2:3], v65, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v77, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v74
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v75
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v64
	v_cndmask_b32_e64 v77, v77, v79, s[14:15]
	v_subrev_co_u32_e64 v79, s[2:3], s16, v74
	s_nop 1
	v_subbrev_co_u32_e64 v65, s[2:3], 0, v65, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v77
	s_nop 1
	v_cndmask_b32_e64 v65, v75, v65, s[2:3]
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v71, v71, v75, vcc
	v_cmp_ne_u32_e32 vcc, 0, v71
	s_nop 1
	v_cndmask_b32_e32 v48, v48, v65, vcc
	v_cndmask_b32_e64 v65, v74, v79, s[2:3]
	v_cndmask_b32_e32 v64, v64, v65, vcc
	v_xor_b32_e32 v64, v64, v70
	v_xor_b32_e32 v48, v48, v70
	v_sub_co_u32_e32 v64, vcc, v64, v70
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v70, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v64, v48, v64, v61
	v_sub_co_u32_e32 v48, vcc, -4, v54
	v_lshl_add_u64 v[70:71], v[54:55], 0, 3
	s_nop 0
	v_subb_co_u32_e32 v61, vcc, -1, v55, vcc
	v_cmp_gt_i32_e32 vcc, 0, v71
	s_nop 1
	v_cndmask_b32_e32 v75, v71, v61, vcc
	v_ashrrev_i32_e32 v100, 31, v75
	v_cndmask_b32_e32 v74, v70, v48, vcc
	v_mov_b32_e32 v101, v100
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[100:101]
	v_xor_b32_e32 v65, v74, v100
	v_xor_b32_e32 v61, v75, v100
	v_mad_u64_u32 v[74:75], s[2:3], v65, s20, 0
	v_mul_hi_u32 v48, v65, s19
	v_lshl_add_u64 v[74:75], v[48:49], 0, v[74:75]
	v_mad_u64_u32 v[106:107], s[2:3], v61, s19, 0
	v_add_co_u32_e32 v48, vcc, v74, v106
	v_mad_u64_u32 v[104:105], s[2:3], v61, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v74, vcc, v75, v107, vcc
	v_mov_b32_e32 v75, s18
	s_nop 0
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[104:105]
	v_mul_lo_u32 v48, s17, v74
	v_mul_lo_u32 v75, s16, v75
	v_mad_u64_u32 v[104:105], s[2:3], s16, v74, 0
	v_add3_u32 v48, v105, v75, v48
	v_sub_u32_e32 v75, v61, v48
	v_sub_co_u32_e32 v65, vcc, v65, v104
	s_nop 1
	v_subb_co_u32_e64 v75, s[2:3], v75, v47, vcc
	v_subrev_co_u32_e64 v77, s[2:3], s16, v65
	v_subb_co_u32_e32 v48, vcc, v61, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v75, s[2:3], 0, v75, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v75
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v77
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v65
	v_cndmask_b32_e64 v77, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v75
	v_cndmask_b32_e64 v65, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v75, v79, v77, s[2:3]
	v_add_u32_e32 v77, 2, v74
	v_add_u32_e32 v79, 1, v74
	v_cmp_ne_u32_e64 s[2:3], 0, v75
	v_cndmask_b32_e32 v48, v61, v65, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v75, v79, v77, s[2:3]
	v_xor_b32_e32 v61, s70, v100
	v_cndmask_b32_e32 v48, v74, v75, vcc
	v_ashrrev_i32_e32 v74, 31, v71
	v_mov_b32_e32 v75, v74
	v_xor_b32_e32 v48, v48, v61
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[74:75]
	v_sub_u32_e32 v48, v48, v61
	v_xor_b32_e32 v75, v70, v74
	v_xad_u32 v61, v48, v74, v46
	v_xor_b32_e32 v65, v71, v74
	v_mad_u64_u32 v[70:71], s[2:3], v75, s20, 0
	v_mul_hi_u32 v48, v75, s19
	v_lshl_add_u64 v[70:71], v[48:49], 0, v[70:71]
	v_mad_u64_u32 v[104:105], s[2:3], v65, s19, 0
	v_add_co_u32_e32 v48, vcc, v70, v104
	v_mad_u64_u32 v[100:101], s[2:3], v65, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v71, v105, vcc
	v_mov_b32_e32 v71, s18
	s_nop 0
	v_addc_co_u32_e32 v101, vcc, 0, v101, vcc
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[100:101]
	v_mul_lo_u32 v48, s17, v70
	v_mul_lo_u32 v77, s16, v71
	v_mad_u64_u32 v[70:71], s[2:3], s16, v70, 0
	v_add3_u32 v48, v71, v77, v48
	v_sub_u32_e32 v71, v65, v48
	v_sub_co_u32_e32 v70, vcc, v75, v70
	v_mul_lo_u32 v61, v61, s33
	s_nop 0
	v_subb_co_u32_e64 v71, s[2:3], v71, v47, vcc
	v_subrev_co_u32_e64 v75, s[2:3], s16, v70
	v_subb_co_u32_e32 v48, vcc, v65, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v77, s[14:15], 0, v71, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v77
	v_subb_co_u32_e64 v71, s[2:3], v71, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v79, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v75
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v81, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v77
	v_cndmask_b32_e64 v65, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v70
	v_cndmask_b32_e64 v79, v79, v81, s[14:15]
	v_subrev_co_u32_e64 v81, s[2:3], s16, v75
	s_nop 1
	v_subbrev_co_u32_e64 v71, s[2:3], 0, v71, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v79
	s_nop 1
	v_cndmask_b32_e64 v71, v77, v71, s[2:3]
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v65, v65, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v65
	v_cndmask_b32_e64 v65, v75, v81, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v65, v70, v65, vcc
	v_cndmask_b32_e32 v48, v48, v71, vcc
	v_xor_b32_e32 v65, v65, v74
	v_xor_b32_e32 v48, v48, v74
	v_sub_co_u32_e32 v65, vcc, v65, v74
	v_lshl_add_u64 v[70:71], v[56:57], 0, 4
	s_nop 0
	v_subb_co_u32_e32 v48, vcc, v48, v74, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v65, v48, v65, v61
	v_sub_co_u32_e32 v48, vcc, -5, v56
	s_nop 1
	v_subb_co_u32_e32 v61, vcc, -1, v57, vcc
	v_cmp_gt_i32_e32 vcc, 0, v71
	s_nop 1
	v_cndmask_b32_e32 v75, v71, v61, vcc
	v_ashrrev_i32_e32 v100, 31, v75
	v_cndmask_b32_e32 v74, v70, v48, vcc
	v_mov_b32_e32 v101, v100
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[100:101]
	v_xor_b32_e32 v77, v74, v100
	v_xor_b32_e32 v61, v75, v100
	v_mad_u64_u32 v[74:75], s[2:3], v77, s20, 0
	v_mul_hi_u32 v48, v77, s19
	v_lshl_add_u64 v[74:75], v[48:49], 0, v[74:75]
	v_mad_u64_u32 v[106:107], s[2:3], v61, s19, 0
	v_add_co_u32_e32 v48, vcc, v74, v106
	v_mad_u64_u32 v[104:105], s[2:3], v61, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v74, vcc, v75, v107, vcc
	v_mov_b32_e32 v75, s18
	s_nop 0
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[104:105]
	v_mul_lo_u32 v48, s17, v74
	v_mul_lo_u32 v75, s16, v75
	v_mad_u64_u32 v[104:105], s[2:3], s16, v74, 0
	v_add3_u32 v48, v105, v75, v48
	v_sub_u32_e32 v75, v61, v48
	v_sub_co_u32_e32 v77, vcc, v77, v104
	s_nop 1
	v_subb_co_u32_e64 v75, s[2:3], v75, v47, vcc
	v_subrev_co_u32_e64 v79, s[2:3], s16, v77
	v_subb_co_u32_e32 v48, vcc, v61, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v75, s[2:3], 0, v75, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v75
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v81, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v79
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v77
	v_cndmask_b32_e64 v79, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v75
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v75, v81, v79, s[2:3]
	v_add_u32_e32 v79, 2, v74
	v_add_u32_e32 v81, 1, v74
	v_cmp_ne_u32_e64 s[2:3], 0, v75
	v_cndmask_b32_e32 v48, v61, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v75, v81, v79, s[2:3]
	v_xor_b32_e32 v61, s70, v100
	v_cndmask_b32_e32 v48, v74, v75, vcc
	v_ashrrev_i32_e32 v74, 31, v71
	v_mov_b32_e32 v75, v74
	v_xor_b32_e32 v48, v48, v61
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[74:75]
	v_sub_u32_e32 v48, v48, v61
	v_xor_b32_e32 v77, v70, v74
	v_xad_u32 v61, v48, v74, v46
	v_xor_b32_e32 v75, v71, v74
	v_mad_u64_u32 v[70:71], s[2:3], v77, s20, 0
	v_mul_hi_u32 v48, v77, s19
	v_lshl_add_u64 v[70:71], v[48:49], 0, v[70:71]
	v_mad_u64_u32 v[104:105], s[2:3], v75, s19, 0
	v_add_co_u32_e32 v48, vcc, v70, v104
	v_mad_u64_u32 v[100:101], s[2:3], v75, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v71, v105, vcc
	v_mov_b32_e32 v71, s18
	s_nop 0
	v_addc_co_u32_e32 v101, vcc, 0, v101, vcc
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[100:101]
	v_mul_lo_u32 v48, s17, v70
	v_mul_lo_u32 v79, s16, v71
	v_mad_u64_u32 v[70:71], s[2:3], s16, v70, 0
	v_add3_u32 v48, v71, v79, v48
	v_sub_u32_e32 v71, v75, v48
	v_sub_co_u32_e32 v70, vcc, v77, v70
	v_mul_lo_u32 v61, v61, s33
	s_nop 0
	v_subb_co_u32_e64 v71, s[2:3], v71, v47, vcc
	v_subrev_co_u32_e64 v77, s[2:3], s16, v70
	v_subb_co_u32_e32 v48, vcc, v75, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v79, s[14:15], 0, v71, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v79
	v_subb_co_u32_e64 v71, s[2:3], v71, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v81, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v77
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v79
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v70
	v_cndmask_b32_e64 v81, v81, v85, s[14:15]
	v_subrev_co_u32_e64 v85, s[2:3], s16, v77
	s_nop 1
	v_subbrev_co_u32_e64 v71, s[2:3], 0, v71, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v81
	s_nop 1
	v_cndmask_b32_e64 v71, v79, v71, s[2:3]
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v75, v75, v79, vcc
	v_cmp_ne_u32_e32 vcc, 0, v75
	s_nop 1
	v_cndmask_b32_e32 v48, v48, v71, vcc
	v_cndmask_b32_e64 v71, v77, v85, s[2:3]
	v_cndmask_b32_e32 v70, v70, v71, vcc
	v_xor_b32_e32 v70, v70, v74
	v_xor_b32_e32 v48, v48, v74
	v_sub_co_u32_e32 v70, vcc, v70, v74
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v74, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v77, v48, v70, v61
	v_sub_co_u32_e32 v48, vcc, -6, v54
	v_lshl_add_u64 v[70:71], v[54:55], 0, 5
	s_nop 0
	v_subb_co_u32_e32 v61, vcc, -1, v55, vcc
	v_cmp_gt_i32_e32 vcc, 0, v71
	s_nop 1
	v_cndmask_b32_e32 v75, v71, v61, vcc
	v_ashrrev_i32_e32 v100, 31, v75
	v_cndmask_b32_e32 v74, v70, v48, vcc
	v_mov_b32_e32 v101, v100
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[100:101]
	v_xor_b32_e32 v79, v74, v100
	v_xor_b32_e32 v61, v75, v100
	v_mad_u64_u32 v[74:75], s[2:3], v79, s20, 0
	v_mul_hi_u32 v48, v79, s19
	v_lshl_add_u64 v[74:75], v[48:49], 0, v[74:75]
	v_mad_u64_u32 v[106:107], s[2:3], v61, s19, 0
	v_add_co_u32_e32 v48, vcc, v74, v106
	v_mad_u64_u32 v[104:105], s[2:3], v61, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v74, vcc, v75, v107, vcc
	v_mov_b32_e32 v75, s18
	s_nop 0
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[104:105]
	v_mul_lo_u32 v48, s17, v74
	v_mul_lo_u32 v75, s16, v75
	v_mad_u64_u32 v[104:105], s[2:3], s16, v74, 0
	v_add3_u32 v48, v105, v75, v48
	v_sub_u32_e32 v75, v61, v48
	v_sub_co_u32_e32 v79, vcc, v79, v104
	s_nop 1
	v_subb_co_u32_e64 v75, s[2:3], v75, v47, vcc
	v_subrev_co_u32_e64 v81, s[2:3], s16, v79
	v_subb_co_u32_e32 v48, vcc, v61, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v75, s[2:3], 0, v75, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v75
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v81
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v79
	v_cndmask_b32_e64 v81, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v75
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v75, v85, v81, s[2:3]
	v_add_u32_e32 v81, 2, v74
	v_add_u32_e32 v85, 1, v74
	v_cmp_ne_u32_e64 s[2:3], 0, v75
	v_cndmask_b32_e32 v48, v61, v79, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v75, v85, v81, s[2:3]
	v_xor_b32_e32 v61, s70, v100
	v_cndmask_b32_e32 v48, v74, v75, vcc
	v_ashrrev_i32_e32 v74, 31, v71
	v_mov_b32_e32 v75, v74
	v_xor_b32_e32 v48, v48, v61
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[74:75]
	v_sub_u32_e32 v48, v48, v61
	v_xor_b32_e32 v79, v70, v74
	v_xad_u32 v61, v48, v74, v46
	v_xor_b32_e32 v75, v71, v74
	v_mad_u64_u32 v[70:71], s[2:3], v79, s20, 0
	v_mul_hi_u32 v48, v79, s19
	v_lshl_add_u64 v[70:71], v[48:49], 0, v[70:71]
	v_mad_u64_u32 v[104:105], s[2:3], v75, s19, 0
	v_add_co_u32_e32 v48, vcc, v70, v104
	v_mad_u64_u32 v[100:101], s[2:3], v75, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v71, v105, vcc
	v_mov_b32_e32 v71, s18
	s_nop 0
	v_addc_co_u32_e32 v101, vcc, 0, v101, vcc
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[100:101]
	v_mul_lo_u32 v48, s17, v70
	v_mul_lo_u32 v81, s16, v71
	v_mad_u64_u32 v[70:71], s[2:3], s16, v70, 0
	v_add3_u32 v48, v71, v81, v48
	v_sub_u32_e32 v71, v75, v48
	v_sub_co_u32_e32 v70, vcc, v79, v70
	v_mul_lo_u32 v61, v61, s33
	s_nop 0
	v_subb_co_u32_e64 v71, s[2:3], v71, v47, vcc
	v_subrev_co_u32_e64 v79, s[2:3], s16, v70
	v_subb_co_u32_e32 v48, vcc, v75, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v81, s[14:15], 0, v71, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v81
	v_subb_co_u32_e64 v71, s[2:3], v71, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v79
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v81
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v70
	v_cndmask_b32_e64 v85, v85, v87, s[14:15]
	v_subrev_co_u32_e64 v87, s[2:3], s16, v79
	s_nop 1
	v_subbrev_co_u32_e64 v71, s[2:3], 0, v71, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v85
	s_nop 1
	v_cndmask_b32_e64 v71, v81, v71, s[2:3]
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v75, v75, v81, vcc
	v_cmp_ne_u32_e32 vcc, 0, v75
	s_nop 1
	v_cndmask_b32_e32 v48, v48, v71, vcc
	v_cndmask_b32_e64 v71, v79, v87, s[2:3]
	v_cndmask_b32_e32 v70, v70, v71, vcc
	v_xor_b32_e32 v70, v70, v74
	v_xor_b32_e32 v48, v48, v74
	v_sub_co_u32_e32 v70, vcc, v70, v74
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v74, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v79, v48, v70, v61
	v_sub_co_u32_e32 v48, vcc, -7, v56
	v_lshl_add_u64 v[70:71], v[56:57], 0, 6
	s_nop 0
	v_subb_co_u32_e32 v61, vcc, -1, v57, vcc
	v_cmp_gt_i32_e32 vcc, 0, v71
	s_nop 1
	v_cndmask_b32_e32 v75, v71, v61, vcc
	v_ashrrev_i32_e32 v100, 31, v75
	v_cndmask_b32_e32 v74, v70, v48, vcc
	v_mov_b32_e32 v101, v100
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[100:101]
	v_xor_b32_e32 v81, v74, v100
	v_xor_b32_e32 v61, v75, v100
	v_mad_u64_u32 v[74:75], s[2:3], v81, s20, 0
	v_mul_hi_u32 v48, v81, s19
	v_lshl_add_u64 v[74:75], v[48:49], 0, v[74:75]
	v_mad_u64_u32 v[106:107], s[2:3], v61, s19, 0
	v_add_co_u32_e32 v48, vcc, v74, v106
	v_mad_u64_u32 v[104:105], s[2:3], v61, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v74, vcc, v75, v107, vcc
	v_mov_b32_e32 v75, s18
	s_nop 0
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[104:105]
	v_mul_lo_u32 v48, s17, v74
	v_mul_lo_u32 v75, s16, v75
	v_mad_u64_u32 v[104:105], s[2:3], s16, v74, 0
	v_add3_u32 v48, v105, v75, v48
	v_sub_u32_e32 v75, v61, v48
	v_sub_co_u32_e32 v81, vcc, v81, v104
	s_nop 1
	v_subb_co_u32_e64 v75, s[2:3], v75, v47, vcc
	v_subrev_co_u32_e64 v85, s[2:3], s16, v81
	v_subb_co_u32_e32 v48, vcc, v61, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v75, s[2:3], 0, v75, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v75
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v85
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v81
	v_cndmask_b32_e64 v85, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v75
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v75, v87, v85, s[2:3]
	v_add_u32_e32 v85, 2, v74
	v_add_u32_e32 v87, 1, v74
	v_cmp_ne_u32_e64 s[2:3], 0, v75
	v_cndmask_b32_e32 v48, v61, v81, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v75, v87, v85, s[2:3]
	v_xor_b32_e32 v61, s70, v100
	v_cndmask_b32_e32 v48, v74, v75, vcc
	v_ashrrev_i32_e32 v74, 31, v71
	v_mov_b32_e32 v75, v74
	v_xor_b32_e32 v48, v48, v61
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[74:75]
	v_sub_u32_e32 v48, v48, v61
	v_xor_b32_e32 v81, v70, v74
	v_xad_u32 v61, v48, v74, v46
	v_xor_b32_e32 v75, v71, v74
	v_mad_u64_u32 v[70:71], s[2:3], v81, s20, 0
	v_mul_hi_u32 v48, v81, s19
	v_lshl_add_u64 v[70:71], v[48:49], 0, v[70:71]
	v_mad_u64_u32 v[104:105], s[2:3], v75, s19, 0
	v_add_co_u32_e32 v48, vcc, v70, v104
	v_mad_u64_u32 v[100:101], s[2:3], v75, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v71, v105, vcc
	v_mov_b32_e32 v71, s18
	s_nop 0
	v_addc_co_u32_e32 v101, vcc, 0, v101, vcc
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[100:101]
	v_mul_lo_u32 v48, s17, v70
	v_mul_lo_u32 v85, s16, v71
	v_mad_u64_u32 v[70:71], s[2:3], s16, v70, 0
	v_add3_u32 v48, v71, v85, v48
	v_sub_u32_e32 v71, v75, v48
	v_sub_co_u32_e32 v70, vcc, v81, v70
	v_mul_lo_u32 v61, v61, s33
	s_nop 0
	v_subb_co_u32_e64 v71, s[2:3], v71, v47, vcc
	v_subrev_co_u32_e64 v81, s[2:3], s16, v70
	v_subb_co_u32_e32 v48, vcc, v75, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[14:15], 0, v71, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v85
	v_subb_co_u32_e64 v71, s[2:3], v71, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v81
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v89, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v85
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v70
	v_cndmask_b32_e64 v87, v87, v89, s[14:15]
	v_subrev_co_u32_e64 v89, s[2:3], s16, v81
	s_nop 1
	v_subbrev_co_u32_e64 v71, s[2:3], 0, v71, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v87
	s_nop 1
	v_cndmask_b32_e64 v71, v85, v71, s[2:3]
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v75, v75, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v75
	s_nop 1
	v_cndmask_b32_e32 v48, v48, v71, vcc
	v_cndmask_b32_e64 v71, v81, v89, s[2:3]
	v_cndmask_b32_e32 v70, v70, v71, vcc
	v_xor_b32_e32 v70, v70, v74
	v_xor_b32_e32 v48, v48, v74
	v_sub_co_u32_e32 v70, vcc, v70, v74
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v74, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v81, v48, v70, v61
	v_sub_co_u32_e32 v48, vcc, -8, v54
	v_lshl_add_u64 v[70:71], v[54:55], 0, 7
	s_nop 0
	v_subb_co_u32_e32 v61, vcc, -1, v55, vcc
	v_cmp_gt_i32_e32 vcc, 0, v71
	s_nop 1
	v_cndmask_b32_e32 v75, v71, v61, vcc
	v_ashrrev_i32_e32 v100, 31, v75
	v_cndmask_b32_e32 v74, v70, v48, vcc
	v_mov_b32_e32 v101, v100
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[100:101]
	v_xor_b32_e32 v85, v74, v100
	v_xor_b32_e32 v61, v75, v100
	v_mad_u64_u32 v[74:75], s[2:3], v85, s20, 0
	v_mul_hi_u32 v48, v85, s19
	v_lshl_add_u64 v[74:75], v[48:49], 0, v[74:75]
	v_mad_u64_u32 v[106:107], s[2:3], v61, s19, 0
	v_add_co_u32_e32 v48, vcc, v74, v106
	v_mad_u64_u32 v[104:105], s[2:3], v61, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v74, vcc, v75, v107, vcc
	v_mov_b32_e32 v75, s18
	s_nop 0
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[74:75], v[74:75], 0, v[104:105]
	v_mul_lo_u32 v48, s17, v74
	v_mul_lo_u32 v75, s16, v75
	v_mad_u64_u32 v[104:105], s[2:3], s16, v74, 0
	v_add3_u32 v48, v105, v75, v48
	v_sub_u32_e32 v75, v61, v48
	v_sub_co_u32_e32 v85, vcc, v85, v104
	s_nop 1
	v_subb_co_u32_e64 v75, s[2:3], v75, v47, vcc
	v_subrev_co_u32_e64 v87, s[2:3], s16, v85
	v_subb_co_u32_e32 v48, vcc, v61, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v75, s[2:3], 0, v75, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v75
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v89, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v87
	v_cndmask_b32_e64 v61, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v85
	v_cndmask_b32_e64 v87, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v75
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v75, v89, v87, s[2:3]
	v_add_u32_e32 v87, 2, v74
	v_add_u32_e32 v89, 1, v74
	v_cmp_ne_u32_e64 s[2:3], 0, v75
	v_cndmask_b32_e32 v48, v61, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v75, v89, v87, s[2:3]
	v_xor_b32_e32 v61, s70, v100
	v_cndmask_b32_e32 v48, v74, v75, vcc
	v_ashrrev_i32_e32 v74, 31, v71
	v_mov_b32_e32 v75, v74
	v_xor_b32_e32 v48, v48, v61
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[74:75]
	v_sub_u32_e32 v48, v48, v61
	v_xor_b32_e32 v85, v70, v74
	v_xad_u32 v61, v48, v74, v46
	v_xor_b32_e32 v75, v71, v74
	v_mad_u64_u32 v[70:71], s[2:3], v85, s20, 0
	v_mul_hi_u32 v48, v85, s19
	v_lshl_add_u64 v[70:71], v[48:49], 0, v[70:71]
	v_mad_u64_u32 v[104:105], s[2:3], v75, s19, 0
	v_add_co_u32_e32 v48, vcc, v70, v104
	v_mad_u64_u32 v[100:101], s[2:3], v75, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v71, v105, vcc
	v_mov_b32_e32 v71, s18
	s_nop 0
	v_addc_co_u32_e32 v101, vcc, 0, v101, vcc
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[100:101]
	v_mul_lo_u32 v48, s17, v70
	v_mul_lo_u32 v87, s16, v71
	v_mad_u64_u32 v[70:71], s[2:3], s16, v70, 0
	v_add3_u32 v48, v71, v87, v48
	v_sub_u32_e32 v71, v75, v48
	v_sub_co_u32_e32 v70, vcc, v85, v70
	v_mul_lo_u32 v61, v61, s33
	s_nop 0
	v_subb_co_u32_e64 v71, s[2:3], v71, v47, vcc
	v_subrev_co_u32_e64 v85, s[2:3], s16, v70
	v_subb_co_u32_e32 v48, vcc, v75, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v87, s[14:15], 0, v71, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v87
	v_subb_co_u32_e64 v71, s[2:3], v71, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v89, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v85
	v_cmp_le_u32_e32 vcc, s17, v48
	v_lshl_add_u64 v[100:101], v[56:57], 0, 8
	v_cndmask_b32_e64 v91, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v87
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v70
	v_cndmask_b32_e64 v89, v89, v91, s[14:15]
	v_subrev_co_u32_e64 v91, s[2:3], s16, v85
	s_nop 1
	v_subbrev_co_u32_e64 v71, s[2:3], 0, v71, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v89
	s_nop 1
	v_cndmask_b32_e64 v71, v87, v71, s[2:3]
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v75, v75, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v75
	s_nop 1
	v_cndmask_b32_e32 v48, v48, v71, vcc
	v_cndmask_b32_e64 v71, v85, v91, s[2:3]
	v_cndmask_b32_e32 v70, v70, v71, vcc
	v_xor_b32_e32 v70, v70, v74
	v_xor_b32_e32 v48, v48, v74
	v_sub_co_u32_e32 v70, vcc, v70, v74
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v74, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v48, v48, v70, v61
	buffer_load_ubyte v58, v58, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v61, v60, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v60, v64, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v64, v65, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v65, v77, s[36:39], 0 offen
	buffer_load_ubyte v71, v79, s[36:39], 0 offen
	buffer_load_ubyte v70, v81, s[36:39], 0 offen
	buffer_load_ubyte v74, v48, s[36:39], 0 offen
	v_sub_co_u32_e32 v48, vcc, -9, v56
	s_nop 1
	v_subb_co_u32_e32 v75, vcc, -1, v57, vcc
	v_cmp_gt_i32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v105, v101, v75, vcc
	v_ashrrev_i32_e32 v106, 31, v105
	v_cndmask_b32_e32 v104, v100, v48, vcc
	v_mov_b32_e32 v107, v106
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[106:107]
	v_xor_b32_e32 v77, v104, v106
	v_xor_b32_e32 v75, v105, v106
	v_mad_u64_u32 v[104:105], s[2:3], v77, s20, 0
	v_mul_hi_u32 v48, v77, s19
	v_lshl_add_u64 v[104:105], v[48:49], 0, v[104:105]
	v_mad_u64_u32 v[110:111], s[2:3], v75, s19, 0
	v_add_co_u32_e32 v48, vcc, v104, v110
	v_mad_u64_u32 v[108:109], s[2:3], v75, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v105, v111, vcc
	v_mov_b32_e32 v105, s18
	s_nop 0
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[108:109]
	v_mul_lo_u32 v48, s17, v104
	v_mul_lo_u32 v79, s16, v105
	v_mad_u64_u32 v[108:109], s[2:3], s16, v104, 0
	v_add3_u32 v48, v109, v79, v48
	v_sub_u32_e32 v79, v75, v48
	v_sub_co_u32_e32 v77, vcc, v77, v108
	s_nop 1
	v_subb_co_u32_e64 v79, s[2:3], v79, v47, vcc
	v_subrev_co_u32_e64 v81, s[2:3], s16, v77
	v_subb_co_u32_e32 v48, vcc, v75, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v79, s[2:3], 0, v79, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v79
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v85, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v81
	v_cndmask_b32_e64 v75, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v77
	v_cndmask_b32_e64 v81, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v79
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v79, v85, v81, s[2:3]
	v_add_u32_e32 v81, 2, v104
	v_add_u32_e32 v85, 1, v104
	v_cmp_ne_u32_e64 s[2:3], 0, v79
	v_cndmask_b32_e32 v48, v75, v77, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v79, v85, v81, s[2:3]
	v_xor_b32_e32 v75, s70, v106
	v_cndmask_b32_e32 v48, v104, v79, vcc
	v_ashrrev_i32_e32 v104, 31, v101
	v_mov_b32_e32 v105, v104
	v_xor_b32_e32 v48, v48, v75
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[104:105]
	v_sub_u32_e32 v48, v48, v75
	v_xor_b32_e32 v79, v100, v104
	v_xad_u32 v75, v48, v104, v46
	v_xor_b32_e32 v77, v101, v104
	v_mad_u64_u32 v[100:101], s[2:3], v79, s20, 0
	v_mul_hi_u32 v48, v79, s19
	v_lshl_add_u64 v[100:101], v[48:49], 0, v[100:101]
	v_mad_u64_u32 v[108:109], s[2:3], v77, s19, 0
	v_add_co_u32_e32 v48, vcc, v100, v108
	v_mad_u64_u32 v[106:107], s[2:3], v77, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v100, vcc, v101, v109, vcc
	v_mov_b32_e32 v101, s18
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[106:107]
	v_mul_lo_u32 v48, s17, v100
	v_mul_lo_u32 v81, s16, v101
	v_mad_u64_u32 v[100:101], s[2:3], s16, v100, 0
	v_add3_u32 v48, v101, v81, v48
	v_sub_u32_e32 v81, v77, v48
	v_sub_co_u32_e32 v79, vcc, v79, v100
	v_mul_lo_u32 v75, v75, s33
	s_nop 0
	v_subb_co_u32_e64 v81, s[2:3], v81, v47, vcc
	v_subrev_co_u32_e64 v85, s[2:3], s16, v79
	v_subb_co_u32_e32 v48, vcc, v77, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v87, s[14:15], 0, v81, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v87
	v_subb_co_u32_e64 v81, s[2:3], v81, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v89, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v85
	v_cmp_le_u32_e32 vcc, s17, v48
	v_lshl_add_u64 v[100:101], v[54:55], 0, 9
	v_cndmask_b32_e64 v91, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v87
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v79
	v_cndmask_b32_e64 v89, v89, v91, s[14:15]
	v_subrev_co_u32_e64 v91, s[2:3], s16, v85
	s_nop 1
	v_subbrev_co_u32_e64 v81, s[2:3], 0, v81, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v89
	s_nop 1
	v_cndmask_b32_e64 v81, v87, v81, s[2:3]
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v77, v77, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v77
	v_cndmask_b32_e64 v77, v85, v91, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v77, v79, v77, vcc
	v_cndmask_b32_e32 v48, v48, v81, vcc
	v_xor_b32_e32 v77, v77, v104
	v_xor_b32_e32 v48, v48, v104
	v_sub_co_u32_e32 v77, vcc, v77, v104
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v104, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v75, v48, v77, v75
	v_sub_co_u32_e32 v48, vcc, -10, v54
	s_nop 1
	v_subb_co_u32_e32 v77, vcc, -1, v55, vcc
	v_cmp_gt_i32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v105, v101, v77, vcc
	v_ashrrev_i32_e32 v106, 31, v105
	v_cndmask_b32_e32 v104, v100, v48, vcc
	v_mov_b32_e32 v107, v106
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[106:107]
	v_xor_b32_e32 v79, v104, v106
	v_xor_b32_e32 v77, v105, v106
	v_mad_u64_u32 v[104:105], s[2:3], v79, s20, 0
	v_mul_hi_u32 v48, v79, s19
	v_lshl_add_u64 v[104:105], v[48:49], 0, v[104:105]
	v_mad_u64_u32 v[110:111], s[2:3], v77, s19, 0
	v_add_co_u32_e32 v48, vcc, v104, v110
	v_mad_u64_u32 v[108:109], s[2:3], v77, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v105, v111, vcc
	v_mov_b32_e32 v105, s18
	s_nop 0
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[108:109]
	v_mul_lo_u32 v48, s17, v104
	v_mul_lo_u32 v81, s16, v105
	v_mad_u64_u32 v[108:109], s[2:3], s16, v104, 0
	v_add3_u32 v48, v109, v81, v48
	v_sub_u32_e32 v81, v77, v48
	v_sub_co_u32_e32 v79, vcc, v79, v108
	s_nop 1
	v_subb_co_u32_e64 v81, s[2:3], v81, v47, vcc
	v_subrev_co_u32_e64 v85, s[2:3], s16, v79
	v_subb_co_u32_e32 v48, vcc, v77, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v81, s[2:3], 0, v81, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v81
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v87, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v85
	v_cndmask_b32_e64 v77, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v79
	v_cndmask_b32_e64 v85, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v81
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v81, v87, v85, s[2:3]
	v_add_u32_e32 v85, 2, v104
	v_add_u32_e32 v87, 1, v104
	v_cmp_ne_u32_e64 s[2:3], 0, v81
	v_cndmask_b32_e32 v48, v77, v79, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v81, v87, v85, s[2:3]
	v_xor_b32_e32 v77, s70, v106
	v_cndmask_b32_e32 v48, v104, v81, vcc
	v_ashrrev_i32_e32 v104, 31, v101
	v_mov_b32_e32 v105, v104
	v_xor_b32_e32 v48, v48, v77
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[104:105]
	v_sub_u32_e32 v48, v48, v77
	v_xor_b32_e32 v81, v100, v104
	v_xad_u32 v77, v48, v104, v46
	v_xor_b32_e32 v79, v101, v104
	v_mad_u64_u32 v[100:101], s[2:3], v81, s20, 0
	v_mul_hi_u32 v48, v81, s19
	v_lshl_add_u64 v[100:101], v[48:49], 0, v[100:101]
	v_mad_u64_u32 v[108:109], s[2:3], v79, s19, 0
	v_add_co_u32_e32 v48, vcc, v100, v108
	v_mad_u64_u32 v[106:107], s[2:3], v79, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v100, vcc, v101, v109, vcc
	v_mov_b32_e32 v101, s18
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[106:107]
	v_mul_lo_u32 v48, s17, v100
	v_mul_lo_u32 v85, s16, v101
	v_mad_u64_u32 v[100:101], s[2:3], s16, v100, 0
	v_add3_u32 v48, v101, v85, v48
	v_sub_u32_e32 v85, v79, v48
	v_sub_co_u32_e32 v81, vcc, v81, v100
	v_mul_lo_u32 v77, v77, s33
	s_nop 0
	v_subb_co_u32_e64 v85, s[2:3], v85, v47, vcc
	v_subrev_co_u32_e64 v87, s[2:3], s16, v81
	v_subb_co_u32_e32 v48, vcc, v79, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[14:15], 0, v85, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v89
	v_subb_co_u32_e64 v85, s[2:3], v85, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v87
	v_cmp_le_u32_e32 vcc, s17, v48
	v_lshl_add_u64 v[100:101], v[56:57], 0, 10
	v_cndmask_b32_e64 v93, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v89
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v81
	v_cndmask_b32_e64 v91, v91, v93, s[14:15]
	v_subrev_co_u32_e64 v93, s[2:3], s16, v87
	s_nop 1
	v_subbrev_co_u32_e64 v85, s[2:3], 0, v85, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v91
	s_nop 1
	v_cndmask_b32_e64 v85, v89, v85, s[2:3]
	v_cndmask_b32_e64 v89, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v79, v79, v89, vcc
	v_cmp_ne_u32_e32 vcc, 0, v79
	v_cndmask_b32_e64 v79, v87, v93, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v79, v81, v79, vcc
	v_cndmask_b32_e32 v48, v48, v85, vcc
	v_xor_b32_e32 v79, v79, v104
	v_xor_b32_e32 v48, v48, v104
	v_sub_co_u32_e32 v79, vcc, v79, v104
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v104, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v77, v48, v79, v77
	v_sub_co_u32_e32 v48, vcc, -11, v56
	s_nop 1
	v_subb_co_u32_e32 v79, vcc, -1, v57, vcc
	v_cmp_gt_i32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v105, v101, v79, vcc
	v_ashrrev_i32_e32 v106, 31, v105
	v_cndmask_b32_e32 v104, v100, v48, vcc
	v_mov_b32_e32 v107, v106
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[106:107]
	v_xor_b32_e32 v81, v104, v106
	v_xor_b32_e32 v79, v105, v106
	v_mad_u64_u32 v[104:105], s[2:3], v81, s20, 0
	v_mul_hi_u32 v48, v81, s19
	v_lshl_add_u64 v[104:105], v[48:49], 0, v[104:105]
	v_mad_u64_u32 v[110:111], s[2:3], v79, s19, 0
	v_add_co_u32_e32 v48, vcc, v104, v110
	v_mad_u64_u32 v[108:109], s[2:3], v79, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v105, v111, vcc
	v_mov_b32_e32 v105, s18
	s_nop 0
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[108:109]
	v_mul_lo_u32 v48, s17, v104
	v_mul_lo_u32 v85, s16, v105
	v_mad_u64_u32 v[108:109], s[2:3], s16, v104, 0
	v_add3_u32 v48, v109, v85, v48
	v_sub_u32_e32 v85, v79, v48
	v_sub_co_u32_e32 v81, vcc, v81, v108
	s_nop 1
	v_subb_co_u32_e64 v85, s[2:3], v85, v47, vcc
	v_subrev_co_u32_e64 v87, s[2:3], s16, v81
	v_subb_co_u32_e32 v48, vcc, v79, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v85, s[2:3], 0, v85, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v85
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v89, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v87
	v_cndmask_b32_e64 v79, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v81
	v_cndmask_b32_e64 v87, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v85
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v85, v89, v87, s[2:3]
	v_add_u32_e32 v87, 2, v104
	v_add_u32_e32 v89, 1, v104
	v_cmp_ne_u32_e64 s[2:3], 0, v85
	v_cndmask_b32_e32 v48, v79, v81, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v85, v89, v87, s[2:3]
	v_xor_b32_e32 v79, s70, v106
	v_cndmask_b32_e32 v48, v104, v85, vcc
	v_ashrrev_i32_e32 v104, 31, v101
	v_mov_b32_e32 v105, v104
	v_xor_b32_e32 v48, v48, v79
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[104:105]
	v_sub_u32_e32 v48, v48, v79
	v_xor_b32_e32 v85, v100, v104
	v_xad_u32 v79, v48, v104, v46
	v_xor_b32_e32 v81, v101, v104
	v_mad_u64_u32 v[100:101], s[2:3], v85, s20, 0
	v_mul_hi_u32 v48, v85, s19
	v_lshl_add_u64 v[100:101], v[48:49], 0, v[100:101]
	v_mad_u64_u32 v[108:109], s[2:3], v81, s19, 0
	v_add_co_u32_e32 v48, vcc, v100, v108
	v_mad_u64_u32 v[106:107], s[2:3], v81, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v100, vcc, v101, v109, vcc
	v_mov_b32_e32 v101, s18
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[106:107]
	v_mul_lo_u32 v48, s17, v100
	v_mul_lo_u32 v87, s16, v101
	v_mad_u64_u32 v[100:101], s[2:3], s16, v100, 0
	v_add3_u32 v48, v101, v87, v48
	v_sub_u32_e32 v87, v81, v48
	v_sub_co_u32_e32 v85, vcc, v85, v100
	v_mul_lo_u32 v79, v79, s33
	s_nop 0
	v_subb_co_u32_e64 v87, s[2:3], v87, v47, vcc
	v_subrev_co_u32_e64 v89, s[2:3], s16, v85
	v_subb_co_u32_e32 v48, vcc, v81, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[14:15], 0, v87, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v91
	v_subb_co_u32_e64 v87, s[2:3], v87, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v89
	v_cmp_le_u32_e32 vcc, s17, v48
	v_lshl_add_u64 v[100:101], v[54:55], 0, 11
	v_cndmask_b32_e64 v95, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v91
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v85
	v_cndmask_b32_e64 v93, v93, v95, s[14:15]
	v_subrev_co_u32_e64 v95, s[2:3], s16, v89
	s_nop 1
	v_subbrev_co_u32_e64 v87, s[2:3], 0, v87, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v93
	s_nop 1
	v_cndmask_b32_e64 v87, v91, v87, s[2:3]
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v81, v81, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v81
	v_cndmask_b32_e64 v81, v89, v95, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v81, v85, v81, vcc
	v_cndmask_b32_e32 v48, v48, v87, vcc
	v_xor_b32_e32 v81, v81, v104
	v_xor_b32_e32 v48, v48, v104
	v_sub_co_u32_e32 v81, vcc, v81, v104
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v104, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v79, v48, v81, v79
	v_sub_co_u32_e32 v48, vcc, -12, v54
	s_nop 1
	v_subb_co_u32_e32 v81, vcc, -1, v55, vcc
	v_cmp_gt_i32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v105, v101, v81, vcc
	v_ashrrev_i32_e32 v106, 31, v105
	v_cndmask_b32_e32 v104, v100, v48, vcc
	v_mov_b32_e32 v107, v106
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[106:107]
	v_xor_b32_e32 v85, v104, v106
	v_xor_b32_e32 v81, v105, v106
	v_mad_u64_u32 v[104:105], s[2:3], v85, s20, 0
	v_mul_hi_u32 v48, v85, s19
	v_lshl_add_u64 v[104:105], v[48:49], 0, v[104:105]
	v_mad_u64_u32 v[110:111], s[2:3], v81, s19, 0
	v_add_co_u32_e32 v48, vcc, v104, v110
	v_mad_u64_u32 v[108:109], s[2:3], v81, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v105, v111, vcc
	v_mov_b32_e32 v105, s18
	s_nop 0
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[108:109]
	v_mul_lo_u32 v48, s17, v104
	v_mul_lo_u32 v87, s16, v105
	v_mad_u64_u32 v[108:109], s[2:3], s16, v104, 0
	v_add3_u32 v48, v109, v87, v48
	v_sub_u32_e32 v87, v81, v48
	v_sub_co_u32_e32 v85, vcc, v85, v108
	s_nop 1
	v_subb_co_u32_e64 v87, s[2:3], v87, v47, vcc
	v_subrev_co_u32_e64 v89, s[2:3], s16, v85
	v_subb_co_u32_e32 v48, vcc, v81, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v87, s[2:3], 0, v87, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v87
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v89
	v_cndmask_b32_e64 v81, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v85
	v_cndmask_b32_e64 v89, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v87
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v87, v91, v89, s[2:3]
	v_add_u32_e32 v89, 2, v104
	v_add_u32_e32 v91, 1, v104
	v_cmp_ne_u32_e64 s[2:3], 0, v87
	v_cndmask_b32_e32 v48, v81, v85, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v87, v91, v89, s[2:3]
	v_xor_b32_e32 v81, s70, v106
	v_cndmask_b32_e32 v48, v104, v87, vcc
	v_ashrrev_i32_e32 v104, 31, v101
	v_mov_b32_e32 v105, v104
	v_xor_b32_e32 v48, v48, v81
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[104:105]
	v_sub_u32_e32 v48, v48, v81
	v_xor_b32_e32 v87, v100, v104
	v_xad_u32 v81, v48, v104, v46
	v_xor_b32_e32 v85, v101, v104
	v_mad_u64_u32 v[100:101], s[2:3], v87, s20, 0
	v_mul_hi_u32 v48, v87, s19
	v_lshl_add_u64 v[100:101], v[48:49], 0, v[100:101]
	v_mad_u64_u32 v[108:109], s[2:3], v85, s19, 0
	v_add_co_u32_e32 v48, vcc, v100, v108
	v_mad_u64_u32 v[106:107], s[2:3], v85, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v100, vcc, v101, v109, vcc
	v_mov_b32_e32 v101, s18
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[106:107]
	v_mul_lo_u32 v48, s17, v100
	v_mul_lo_u32 v89, s16, v101
	v_mad_u64_u32 v[100:101], s[2:3], s16, v100, 0
	v_add3_u32 v48, v101, v89, v48
	v_sub_u32_e32 v89, v85, v48
	v_sub_co_u32_e32 v87, vcc, v87, v100
	v_mul_lo_u32 v81, v81, s33
	s_nop 0
	v_subb_co_u32_e64 v89, s[2:3], v89, v47, vcc
	v_subrev_co_u32_e64 v91, s[2:3], s16, v87
	v_subb_co_u32_e32 v48, vcc, v85, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v93, s[14:15], 0, v89, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v93
	v_subb_co_u32_e64 v89, s[2:3], v89, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v91
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v100, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v93
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v87
	v_cndmask_b32_e64 v95, v95, v100, s[14:15]
	v_subrev_co_u32_e64 v100, s[2:3], s16, v91
	s_nop 1
	v_subbrev_co_u32_e64 v89, s[2:3], 0, v89, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v95
	s_nop 1
	v_cndmask_b32_e64 v89, v93, v89, s[2:3]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v85, v85, v93, vcc
	v_cmp_ne_u32_e32 vcc, 0, v85
	v_cndmask_b32_e64 v85, v91, v100, s[2:3]
	v_lshl_add_u64 v[100:101], v[56:57], 0, 12
	v_cndmask_b32_e32 v85, v87, v85, vcc
	v_cndmask_b32_e32 v48, v48, v89, vcc
	v_xor_b32_e32 v85, v85, v104
	v_xor_b32_e32 v48, v48, v104
	v_sub_co_u32_e32 v85, vcc, v85, v104
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v104, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v81, v48, v85, v81
	v_sub_co_u32_e32 v48, vcc, -13, v56
	s_nop 1
	v_subb_co_u32_e32 v85, vcc, -1, v57, vcc
	v_cmp_gt_i32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v105, v101, v85, vcc
	v_ashrrev_i32_e32 v106, 31, v105
	v_cndmask_b32_e32 v104, v100, v48, vcc
	v_mov_b32_e32 v107, v106
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[106:107]
	v_xor_b32_e32 v87, v104, v106
	v_xor_b32_e32 v85, v105, v106
	v_mad_u64_u32 v[104:105], s[2:3], v87, s20, 0
	v_mul_hi_u32 v48, v87, s19
	v_lshl_add_u64 v[104:105], v[48:49], 0, v[104:105]
	v_mad_u64_u32 v[110:111], s[2:3], v85, s19, 0
	v_add_co_u32_e32 v48, vcc, v104, v110
	v_mad_u64_u32 v[108:109], s[2:3], v85, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v105, v111, vcc
	v_mov_b32_e32 v105, s18
	s_nop 0
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[108:109]
	v_mul_lo_u32 v48, s17, v104
	v_mul_lo_u32 v89, s16, v105
	v_mad_u64_u32 v[108:109], s[2:3], s16, v104, 0
	v_add3_u32 v48, v109, v89, v48
	v_sub_u32_e32 v89, v85, v48
	v_sub_co_u32_e32 v87, vcc, v87, v108
	s_nop 1
	v_subb_co_u32_e64 v89, s[2:3], v89, v47, vcc
	v_subrev_co_u32_e64 v91, s[2:3], s16, v87
	v_subb_co_u32_e32 v48, vcc, v85, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[2:3], 0, v89, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v89
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v91
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v87
	v_cndmask_b32_e64 v91, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v89
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v89, v93, v91, s[2:3]
	v_add_u32_e32 v91, 2, v104
	v_add_u32_e32 v93, 1, v104
	v_cmp_ne_u32_e64 s[2:3], 0, v89
	v_cndmask_b32_e32 v48, v85, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v89, v93, v91, s[2:3]
	v_xor_b32_e32 v85, s70, v106
	v_cndmask_b32_e32 v48, v104, v89, vcc
	v_ashrrev_i32_e32 v104, 31, v101
	v_mov_b32_e32 v105, v104
	v_xor_b32_e32 v48, v48, v85
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[104:105]
	v_sub_u32_e32 v48, v48, v85
	v_xor_b32_e32 v89, v100, v104
	v_xad_u32 v85, v48, v104, v46
	v_xor_b32_e32 v87, v101, v104
	v_mad_u64_u32 v[100:101], s[2:3], v89, s20, 0
	v_mul_hi_u32 v48, v89, s19
	v_lshl_add_u64 v[100:101], v[48:49], 0, v[100:101]
	v_mad_u64_u32 v[108:109], s[2:3], v87, s19, 0
	v_add_co_u32_e32 v48, vcc, v100, v108
	v_mad_u64_u32 v[106:107], s[2:3], v87, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v100, vcc, v101, v109, vcc
	v_mov_b32_e32 v101, s18
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[106:107]
	v_mul_lo_u32 v48, s17, v100
	v_mul_lo_u32 v91, s16, v101
	v_mad_u64_u32 v[100:101], s[2:3], s16, v100, 0
	v_add3_u32 v48, v101, v91, v48
	v_sub_u32_e32 v91, v87, v48
	v_sub_co_u32_e32 v89, vcc, v89, v100
	v_mul_lo_u32 v85, v85, s33
	s_nop 0
	v_subb_co_u32_e64 v91, s[2:3], v91, v47, vcc
	v_subrev_co_u32_e64 v93, s[2:3], s16, v89
	v_subb_co_u32_e32 v48, vcc, v87, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v95, s[14:15], 0, v91, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v95
	v_subb_co_u32_e64 v91, s[2:3], v91, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v100, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v93
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v101, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v95
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v89
	v_cndmask_b32_e64 v100, v100, v101, s[14:15]
	v_subrev_co_u32_e64 v101, s[2:3], s16, v93
	s_nop 1
	v_subbrev_co_u32_e64 v91, s[2:3], 0, v91, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v100
	s_nop 1
	v_cndmask_b32_e64 v91, v95, v91, s[2:3]
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v87, v87, v95, vcc
	v_cmp_ne_u32_e32 vcc, 0, v87
	v_cndmask_b32_e64 v87, v93, v101, s[2:3]
	v_lshl_add_u64 v[100:101], v[54:55], 0, 13
	v_cndmask_b32_e32 v87, v89, v87, vcc
	v_cndmask_b32_e32 v48, v48, v91, vcc
	v_xor_b32_e32 v87, v87, v104
	v_xor_b32_e32 v48, v48, v104
	v_sub_co_u32_e32 v87, vcc, v87, v104
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v104, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v85, v48, v87, v85
	v_sub_co_u32_e32 v48, vcc, -14, v54
	s_nop 1
	v_subb_co_u32_e32 v87, vcc, -1, v55, vcc
	v_cmp_gt_i32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v105, v101, v87, vcc
	v_ashrrev_i32_e32 v106, 31, v105
	v_cndmask_b32_e32 v104, v100, v48, vcc
	v_mov_b32_e32 v107, v106
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[106:107]
	v_xor_b32_e32 v89, v104, v106
	v_xor_b32_e32 v87, v105, v106
	v_mad_u64_u32 v[104:105], s[2:3], v89, s20, 0
	v_mul_hi_u32 v48, v89, s19
	v_lshl_add_u64 v[104:105], v[48:49], 0, v[104:105]
	v_mad_u64_u32 v[110:111], s[2:3], v87, s19, 0
	v_add_co_u32_e32 v48, vcc, v104, v110
	v_mad_u64_u32 v[108:109], s[2:3], v87, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v105, v111, vcc
	v_mov_b32_e32 v105, s18
	s_nop 0
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[108:109]
	v_mul_lo_u32 v48, s17, v104
	v_mul_lo_u32 v91, s16, v105
	v_mad_u64_u32 v[108:109], s[2:3], s16, v104, 0
	v_add3_u32 v48, v109, v91, v48
	v_sub_u32_e32 v91, v87, v48
	v_sub_co_u32_e32 v89, vcc, v89, v108
	s_nop 1
	v_subb_co_u32_e64 v91, s[2:3], v91, v47, vcc
	v_subrev_co_u32_e64 v93, s[2:3], s16, v89
	v_subb_co_u32_e32 v48, vcc, v87, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[2:3], 0, v91, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v91
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v93
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v89
	v_cndmask_b32_e64 v93, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v91
	v_cndmask_b32_e64 v89, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v91, v95, v93, s[2:3]
	v_add_u32_e32 v93, 2, v104
	v_add_u32_e32 v95, 1, v104
	v_cmp_ne_u32_e64 s[2:3], 0, v91
	v_cndmask_b32_e32 v48, v87, v89, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v91, v95, v93, s[2:3]
	v_xor_b32_e32 v87, s70, v106
	v_cndmask_b32_e32 v48, v104, v91, vcc
	v_ashrrev_i32_e32 v104, 31, v101
	v_mov_b32_e32 v105, v104
	v_xor_b32_e32 v48, v48, v87
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[104:105]
	v_sub_u32_e32 v48, v48, v87
	v_xor_b32_e32 v91, v100, v104
	v_xad_u32 v87, v48, v104, v46
	v_xor_b32_e32 v89, v101, v104
	v_mad_u64_u32 v[100:101], s[2:3], v91, s20, 0
	v_mul_hi_u32 v48, v91, s19
	v_lshl_add_u64 v[100:101], v[48:49], 0, v[100:101]
	v_mad_u64_u32 v[108:109], s[2:3], v89, s19, 0
	v_add_co_u32_e32 v48, vcc, v100, v108
	v_mad_u64_u32 v[106:107], s[2:3], v89, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v100, vcc, v101, v109, vcc
	v_mov_b32_e32 v101, s18
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[106:107]
	v_mul_lo_u32 v48, s17, v100
	v_mul_lo_u32 v93, s16, v101
	v_mad_u64_u32 v[100:101], s[2:3], s16, v100, 0
	v_add3_u32 v48, v101, v93, v48
	v_sub_u32_e32 v93, v89, v48
	v_sub_co_u32_e32 v91, vcc, v91, v100
	v_mul_lo_u32 v87, v87, s33
	s_nop 0
	v_subb_co_u32_e64 v93, s[2:3], v93, v47, vcc
	v_subrev_co_u32_e64 v95, s[2:3], s16, v91
	v_subb_co_u32_e32 v48, vcc, v89, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v100, s[14:15], 0, v93, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v100
	v_subb_co_u32_e64 v93, s[2:3], v93, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v101, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v95
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v105, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v100
	v_cndmask_b32_e64 v89, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v91
	v_cndmask_b32_e64 v101, v101, v105, s[14:15]
	v_subrev_co_u32_e64 v105, s[2:3], s16, v95
	s_nop 1
	v_subbrev_co_u32_e64 v93, s[2:3], 0, v93, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v101
	s_nop 1
	v_cndmask_b32_e64 v93, v100, v93, s[2:3]
	v_cndmask_b32_e64 v100, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v89, v89, v100, vcc
	v_cmp_ne_u32_e32 vcc, 0, v89
	v_cndmask_b32_e64 v89, v95, v105, s[2:3]
	v_lshl_add_u64 v[100:101], v[56:57], 0, 14
	v_cndmask_b32_e32 v89, v91, v89, vcc
	v_cndmask_b32_e32 v48, v48, v93, vcc
	v_xor_b32_e32 v89, v89, v104
	v_xor_b32_e32 v48, v48, v104
	v_sub_co_u32_e32 v89, vcc, v89, v104
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v104, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v87, v48, v89, v87
	v_sub_co_u32_e32 v48, vcc, -15, v56
	s_nop 1
	v_subb_co_u32_e32 v56, vcc, -1, v57, vcc
	v_cmp_gt_i32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v57, v101, v56, vcc
	v_ashrrev_i32_e32 v104, 31, v57
	v_cndmask_b32_e32 v56, v100, v48, vcc
	v_mov_b32_e32 v105, v104
	v_lshl_add_u64 v[56:57], v[56:57], 0, v[104:105]
	v_xor_b32_e32 v91, v56, v104
	v_xor_b32_e32 v89, v57, v104
	v_mad_u64_u32 v[56:57], s[2:3], v91, s20, 0
	v_mul_hi_u32 v48, v91, s19
	v_lshl_add_u64 v[56:57], v[48:49], 0, v[56:57]
	v_mad_u64_u32 v[108:109], s[2:3], v89, s19, 0
	v_add_co_u32_e32 v48, vcc, v56, v108
	v_mad_u64_u32 v[106:107], s[2:3], v89, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v56, vcc, v57, v109, vcc
	v_mov_b32_e32 v57, s18
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[56:57], v[56:57], 0, v[106:107]
	v_mul_lo_u32 v48, s17, v56
	v_mul_lo_u32 v57, s16, v57
	v_mad_u64_u32 v[106:107], s[2:3], s16, v56, 0
	v_add3_u32 v48, v107, v57, v48
	v_sub_u32_e32 v57, v89, v48
	v_sub_co_u32_e32 v91, vcc, v91, v106
	s_nop 1
	v_subb_co_u32_e64 v57, s[2:3], v57, v47, vcc
	v_subrev_co_u32_e64 v93, s[2:3], s16, v91
	v_subb_co_u32_e32 v48, vcc, v89, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v57, s[2:3], 0, v57, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v57
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v93
	v_cndmask_b32_e64 v89, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v91
	v_cndmask_b32_e64 v93, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v57
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v57, v95, v93, s[2:3]
	v_add_u32_e32 v93, 2, v56
	v_add_u32_e32 v95, 1, v56
	v_cmp_ne_u32_e64 s[2:3], 0, v57
	v_cndmask_b32_e32 v48, v89, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v57, v95, v93, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v48, v56, v57, vcc
	v_xor_b32_e32 v56, s70, v104
	v_xor_b32_e32 v48, v48, v56
	v_sub_u32_e32 v48, v48, v56
	v_ashrrev_i32_e32 v56, 31, v101
	v_mov_b32_e32 v57, v56
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[56:57]
	v_xor_b32_e32 v91, v100, v56
	v_xad_u32 v89, v48, v56, v46
	v_xor_b32_e32 v57, v101, v56
	v_mad_u64_u32 v[100:101], s[2:3], v91, s20, 0
	v_mul_hi_u32 v48, v91, s19
	v_lshl_add_u64 v[100:101], v[48:49], 0, v[100:101]
	v_mad_u64_u32 v[106:107], s[2:3], v57, s19, 0
	v_add_co_u32_e32 v48, vcc, v100, v106
	v_mad_u64_u32 v[104:105], s[2:3], v57, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v100, vcc, v101, v107, vcc
	v_mov_b32_e32 v101, s18
	s_nop 0
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[104:105]
	v_mul_lo_u32 v48, s17, v100
	v_mul_lo_u32 v93, s16, v101
	v_mad_u64_u32 v[100:101], s[2:3], s16, v100, 0
	v_add3_u32 v48, v101, v93, v48
	v_sub_u32_e32 v93, v57, v48
	v_sub_co_u32_e32 v91, vcc, v91, v100
	s_nop 1
	v_subb_co_u32_e64 v93, s[2:3], v93, v47, vcc
	v_subrev_co_u32_e64 v95, s[2:3], s16, v91
	v_subb_co_u32_e32 v48, vcc, v57, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v100, s[14:15], 0, v93, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v100
	v_subb_co_u32_e64 v93, s[2:3], v93, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v101, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v95
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v104, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v100
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v91
	v_cndmask_b32_e64 v101, v101, v104, s[14:15]
	v_subrev_co_u32_e64 v104, s[2:3], s16, v95
	s_nop 1
	v_subbrev_co_u32_e64 v93, s[2:3], 0, v93, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v101
	s_nop 1
	v_cndmask_b32_e64 v93, v100, v93, s[2:3]
	v_cndmask_b32_e64 v100, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v57, v57, v100, vcc
	v_cmp_ne_u32_e32 vcc, 0, v57
	v_cndmask_b32_e64 v57, v95, v104, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v57, v91, v57, vcc
	v_cndmask_b32_e32 v48, v48, v93, vcc
	v_xor_b32_e32 v57, v57, v56
	v_xor_b32_e32 v48, v48, v56
	v_sub_co_u32_e32 v57, vcc, v57, v56
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v56, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_mul_lo_u32 v56, v89, s33
	v_add3_u32 v89, v48, v57, v56
	v_sub_co_u32_e32 v48, vcc, -16, v54
	v_lshl_add_u64 v[56:57], v[54:55], 0, 15
	s_nop 0
	v_subb_co_u32_e32 v54, vcc, -1, v55, vcc
	v_cmp_gt_i32_e32 vcc, 0, v57
	s_nop 1
	v_cndmask_b32_e32 v55, v57, v54, vcc
	v_ashrrev_i32_e32 v100, 31, v55
	v_cndmask_b32_e32 v54, v56, v48, vcc
	v_mov_b32_e32 v101, v100
	v_lshl_add_u64 v[54:55], v[54:55], 0, v[100:101]
	v_xor_b32_e32 v93, v54, v100
	v_xor_b32_e32 v91, v55, v100
	v_mad_u64_u32 v[54:55], s[2:3], v93, s20, 0
	v_mul_hi_u32 v48, v93, s19
	v_lshl_add_u64 v[54:55], v[48:49], 0, v[54:55]
	v_mad_u64_u32 v[106:107], s[2:3], v91, s19, 0
	v_add_co_u32_e32 v48, vcc, v54, v106
	v_mad_u64_u32 v[104:105], s[2:3], v91, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v54, vcc, v55, v107, vcc
	v_mov_b32_e32 v55, s18
	s_nop 0
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[54:55], v[54:55], 0, v[104:105]
	v_mul_lo_u32 v48, s17, v54
	v_mul_lo_u32 v55, s16, v55
	v_mad_u64_u32 v[104:105], s[2:3], s16, v54, 0
	v_add3_u32 v48, v105, v55, v48
	v_sub_u32_e32 v55, v91, v48
	v_sub_co_u32_e32 v93, vcc, v93, v104
	s_nop 1
	v_subb_co_u32_e64 v55, s[2:3], v55, v47, vcc
	v_subrev_co_u32_e64 v95, s[2:3], s16, v93
	v_subb_co_u32_e32 v48, vcc, v91, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v55, s[2:3], 0, v55, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v55
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v101, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v95
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v93
	v_cndmask_b32_e64 v95, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v55
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v55, v101, v95, s[2:3]
	v_add_u32_e32 v95, 2, v54
	v_add_u32_e32 v101, 1, v54
	v_cmp_ne_u32_e64 s[2:3], 0, v55
	v_cndmask_b32_e32 v48, v91, v93, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v55, v101, v95, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v48, v54, v55, vcc
	v_xor_b32_e32 v54, s70, v100
	v_xor_b32_e32 v48, v48, v54
	v_sub_u32_e32 v48, v48, v54
	v_ashrrev_i32_e32 v54, 31, v57
	v_mov_b32_e32 v55, v54
	v_lshl_add_u64 v[56:57], v[56:57], 0, v[54:55]
	v_xor_b32_e32 v93, v56, v54
	v_xad_u32 v91, v48, v54, v46
	v_xor_b32_e32 v55, v57, v54
	v_mad_u64_u32 v[56:57], s[2:3], v93, s20, 0
	v_mul_hi_u32 v48, v93, s19
	v_lshl_add_u64 v[56:57], v[48:49], 0, v[56:57]
	v_mad_u64_u32 v[104:105], s[2:3], v55, s19, 0
	v_add_co_u32_e32 v48, vcc, v56, v104
	v_mad_u64_u32 v[100:101], s[2:3], v55, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v56, vcc, v57, v105, vcc
	v_mov_b32_e32 v57, s18
	s_nop 0
	v_addc_co_u32_e32 v101, vcc, 0, v101, vcc
	v_lshl_add_u64 v[56:57], v[56:57], 0, v[100:101]
	v_mul_lo_u32 v48, s17, v56
	v_mul_lo_u32 v95, s16, v57
	v_mad_u64_u32 v[56:57], s[2:3], s16, v56, 0
	v_add3_u32 v48, v57, v95, v48
	v_sub_u32_e32 v57, v55, v48
	v_sub_co_u32_e32 v56, vcc, v93, v56
	s_nop 1
	v_subb_co_u32_e64 v57, s[2:3], v57, v47, vcc
	v_subrev_co_u32_e64 v93, s[2:3], s16, v56
	v_subb_co_u32_e32 v48, vcc, v55, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v95, s[14:15], 0, v57, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v95
	v_subb_co_u32_e64 v57, s[2:3], v57, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v100, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v93
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v101, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v95
	v_cndmask_b32_e64 v55, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v56
	v_cndmask_b32_e64 v100, v100, v101, s[14:15]
	v_subrev_co_u32_e64 v101, s[2:3], s16, v93
	s_nop 1
	v_subbrev_co_u32_e64 v57, s[2:3], 0, v57, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v100
	s_nop 1
	v_cndmask_b32_e64 v57, v95, v57, s[2:3]
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v55, v55, v95, vcc
	v_cmp_ne_u32_e32 vcc, 0, v55
	v_cndmask_b32_e64 v55, v93, v101, s[2:3]
	s_movk_i32 s2, 0xfc00
	v_cndmask_b32_e32 v55, v56, v55, vcc
	v_cndmask_b32_e32 v48, v48, v57, vcc
	v_xor_b32_e32 v55, v55, v54
	v_xor_b32_e32 v48, v48, v54
	v_sub_co_u32_e32 v55, vcc, v55, v54
	s_mov_b32 s3, -1
	s_nop 0
	v_subb_co_u32_e32 v48, vcc, v48, v54, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_mul_lo_u32 v54, v91, s33
	v_add3_u32 v48, v48, v55, v54
	v_lshl_add_u64 v[100:101], v[52:53], 0, s[2:3]
	s_movk_i32 s2, 0x3ff
	buffer_load_ubyte v54, v75, s[36:39], 0 offen
	buffer_load_ubyte v56, v77, s[36:39], 0 offen
	buffer_load_ubyte v55, v79, s[36:39], 0 offen
	buffer_load_ubyte v57, v81, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v75, v85, s[36:39], 0 offen
	buffer_load_ubyte v79, v87, s[36:39], 0 offen
	buffer_load_ubyte v77, v89, s[36:39], 0 offen
	buffer_load_ubyte v81, v48, s[36:39], 0 offen
	v_sub_co_u32_e32 v48, vcc, s2, v52
	s_nop 1
	v_subb_co_u32_e32 v53, vcc, 0, v53, vcc
	v_cmp_gt_i32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v53, v101, v53, vcc
	v_ashrrev_i32_e32 v104, 31, v53
	v_cndmask_b32_e32 v52, v100, v48, vcc
	v_mov_b32_e32 v105, v104
	v_lshl_add_u64 v[52:53], v[52:53], 0, v[104:105]
	v_xor_b32_e32 v87, v52, v104
	v_xor_b32_e32 v85, v53, v104
	v_mad_u64_u32 v[52:53], s[2:3], v87, s20, 0
	v_mul_hi_u32 v48, v87, s19
	v_lshl_add_u64 v[52:53], v[48:49], 0, v[52:53]
	v_mad_u64_u32 v[108:109], s[2:3], v85, s19, 0
	v_add_co_u32_e32 v48, vcc, v52, v108
	v_mad_u64_u32 v[106:107], s[2:3], v85, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v52, vcc, v53, v109, vcc
	v_mov_b32_e32 v53, s18
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[52:53], v[52:53], 0, v[106:107]
	v_mul_lo_u32 v48, s17, v52
	v_mul_lo_u32 v53, s16, v53
	v_mad_u64_u32 v[106:107], s[2:3], s16, v52, 0
	v_add3_u32 v48, v107, v53, v48
	v_sub_u32_e32 v53, v85, v48
	v_sub_co_u32_e32 v87, vcc, v87, v106
	s_nop 1
	v_subb_co_u32_e64 v53, s[2:3], v53, v47, vcc
	v_subrev_co_u32_e64 v89, s[2:3], s16, v87
	v_subb_co_u32_e32 v48, vcc, v85, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v53, s[2:3], 0, v53, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v53
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v91, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v89
	v_cndmask_b32_e64 v85, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v87
	v_cndmask_b32_e64 v89, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v53
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v53, v91, v89, s[2:3]
	v_add_u32_e32 v89, 2, v52
	v_add_u32_e32 v91, 1, v52
	v_cmp_ne_u32_e64 s[2:3], 0, v53
	v_cndmask_b32_e32 v48, v85, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v53, v91, v89, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v48, v52, v53, vcc
	v_xor_b32_e32 v52, s70, v104
	v_xor_b32_e32 v48, v48, v52
	v_sub_u32_e32 v48, v48, v52
	v_ashrrev_i32_e32 v52, 31, v101
	v_mov_b32_e32 v53, v52
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[52:53]
	v_xor_b32_e32 v87, v100, v52
	v_xad_u32 v85, v48, v52, v46
	v_xor_b32_e32 v53, v101, v52
	v_mad_u64_u32 v[100:101], s[2:3], v87, s20, 0
	v_mul_hi_u32 v48, v87, s19
	v_lshl_add_u64 v[100:101], v[48:49], 0, v[100:101]
	v_mad_u64_u32 v[106:107], s[2:3], v53, s19, 0
	v_add_co_u32_e32 v48, vcc, v100, v106
	v_mad_u64_u32 v[104:105], s[2:3], v53, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v100, vcc, v101, v107, vcc
	v_mov_b32_e32 v101, s18
	s_nop 0
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[104:105]
	v_mul_lo_u32 v48, s17, v100
	v_mul_lo_u32 v89, s16, v101
	v_mad_u64_u32 v[100:101], s[2:3], s16, v100, 0
	v_add3_u32 v48, v101, v89, v48
	v_sub_u32_e32 v89, v53, v48
	v_sub_co_u32_e32 v87, vcc, v87, v100
	s_nop 1
	v_subb_co_u32_e64 v89, s[2:3], v89, v47, vcc
	v_subrev_co_u32_e64 v91, s[2:3], s16, v87
	v_subb_co_u32_e32 v48, vcc, v53, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v93, s[14:15], 0, v89, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v93
	v_subb_co_u32_e64 v89, s[2:3], v89, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v91
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v100, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v93
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v87
	v_cndmask_b32_e64 v95, v95, v100, s[14:15]
	v_subrev_co_u32_e64 v100, s[2:3], s16, v91
	s_nop 1
	v_subbrev_co_u32_e64 v89, s[2:3], 0, v89, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v95
	s_nop 1
	v_cndmask_b32_e64 v89, v93, v89, s[2:3]
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v53, v53, v93, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_cndmask_b32_e64 v53, v91, v100, s[2:3]
	s_movk_i32 s2, 0xfe00
	v_cndmask_b32_e32 v53, v87, v53, vcc
	v_cndmask_b32_e32 v48, v48, v89, vcc
	v_xor_b32_e32 v53, v53, v52
	v_xor_b32_e32 v48, v48, v52
	v_sub_co_u32_e32 v53, vcc, v53, v52
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v52, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_mul_lo_u32 v52, v85, s33
	v_add3_u32 v85, v48, v53, v52
	v_mov_b32_e32 v48, 0xfffffc00
	v_bitop3_b32 v52, v43, s2, v48 bitop3:0xc8
	v_ashrrev_i32_e32 v53, 31, v52
	v_lshl_add_u64 v[52:53], v[50:51], 0, v[52:53]
	v_lshl_add_u64 v[52:53], v[52:53], 0, v[102:103]
	v_sub_co_u32_e32 v43, vcc, -2, v52
	v_lshl_add_u64 v[100:101], v[52:53], 0, 1
	s_nop 0
	v_subb_co_u32_e32 v48, vcc, -1, v53, vcc
	v_cmp_gt_i32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v105, v101, v48, vcc
	v_ashrrev_i32_e32 v106, 31, v105
	v_cndmask_b32_e32 v104, v100, v43, vcc
	v_mov_b32_e32 v107, v106
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[106:107]
	v_xor_b32_e32 v87, v104, v106
	v_xor_b32_e32 v43, v105, v106
	v_mad_u64_u32 v[104:105], s[2:3], v87, s20, 0
	v_mul_hi_u32 v48, v87, s19
	v_lshl_add_u64 v[104:105], v[48:49], 0, v[104:105]
	v_mad_u64_u32 v[110:111], s[2:3], v43, s19, 0
	v_add_co_u32_e32 v48, vcc, v104, v110
	v_mad_u64_u32 v[108:109], s[2:3], v43, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v105, v111, vcc
	v_mov_b32_e32 v105, s18
	s_nop 0
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[108:109]
	v_mul_lo_u32 v48, s17, v104
	v_mul_lo_u32 v89, s16, v105
	v_mad_u64_u32 v[108:109], s[2:3], s16, v104, 0
	v_add3_u32 v48, v109, v89, v48
	v_sub_u32_e32 v89, v43, v48
	v_sub_co_u32_e32 v87, vcc, v87, v108
	s_nop 1
	v_subb_co_u32_e64 v89, s[2:3], v89, v47, vcc
	v_subrev_co_u32_e64 v91, s[2:3], s16, v87
	v_subb_co_u32_e32 v43, vcc, v43, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[2:3], 0, v89, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v89
	v_cmp_le_u32_e32 vcc, s17, v43
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v91
	v_cndmask_b32_e64 v48, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v87
	v_cndmask_b32_e64 v91, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v89
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v43
	v_cndmask_b32_e64 v89, v93, v91, s[2:3]
	v_add_u32_e32 v91, 2, v104
	v_add_u32_e32 v93, 1, v104
	v_cmp_ne_u32_e64 s[2:3], 0, v89
	v_cndmask_b32_e32 v43, v48, v87, vcc
	v_cmp_ne_u32_e32 vcc, 0, v43
	v_cndmask_b32_e64 v89, v93, v91, s[2:3]
	v_xor_b32_e32 v48, s70, v106
	v_cndmask_b32_e32 v43, v104, v89, vcc
	v_ashrrev_i32_e32 v104, 31, v101
	v_mov_b32_e32 v105, v104
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[104:105]
	v_xor_b32_e32 v43, v43, v48
	v_xor_b32_e32 v89, v100, v104
	v_sub_u32_e32 v43, v43, v48
	v_xor_b32_e32 v87, v101, v104
	v_mad_u64_u32 v[100:101], s[2:3], v89, s20, 0
	v_mul_hi_u32 v48, v89, s19
	v_lshl_add_u64 v[100:101], v[48:49], 0, v[100:101]
	v_mad_u64_u32 v[108:109], s[2:3], v87, s19, 0
	v_add_co_u32_e32 v48, vcc, v100, v108
	v_mad_u64_u32 v[106:107], s[2:3], v87, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v100, vcc, v101, v109, vcc
	v_mov_b32_e32 v101, s18
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[106:107]
	v_mul_lo_u32 v48, s17, v100
	v_mul_lo_u32 v91, s16, v101
	v_mad_u64_u32 v[100:101], s[2:3], s16, v100, 0
	v_add3_u32 v48, v101, v91, v48
	v_sub_u32_e32 v91, v87, v48
	v_sub_co_u32_e32 v89, vcc, v89, v100
	v_xad_u32 v43, v43, v104, v46
	s_nop 0
	v_subb_co_u32_e64 v91, s[2:3], v91, v47, vcc
	v_subrev_co_u32_e64 v93, s[2:3], s16, v89
	v_subb_co_u32_e32 v48, vcc, v87, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v95, s[14:15], 0, v91, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v95
	v_subb_co_u32_e64 v91, s[2:3], v91, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v100, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v93
	v_cmp_le_u32_e32 vcc, s17, v48
	v_mul_lo_u32 v43, v43, s33
	v_cndmask_b32_e64 v101, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v95
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v89
	v_cndmask_b32_e64 v100, v100, v101, s[14:15]
	v_subrev_co_u32_e64 v101, s[2:3], s16, v93
	s_nop 1
	v_subbrev_co_u32_e64 v91, s[2:3], 0, v91, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v100
	v_or_b32_e32 v100, 0xfffffc00, v83
	s_nop 0
	v_cndmask_b32_e64 v91, v95, v91, s[2:3]
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v87, v87, v95, vcc
	v_cmp_ne_u32_e32 vcc, 0, v87
	v_cndmask_b32_e64 v87, v93, v101, s[2:3]
	v_ashrrev_i32_e32 v101, 31, v100
	v_cndmask_b32_e32 v87, v89, v87, vcc
	v_cndmask_b32_e32 v48, v48, v91, vcc
	v_xor_b32_e32 v87, v87, v104
	v_xor_b32_e32 v48, v48, v104
	v_sub_co_u32_e32 v87, vcc, v87, v104
	v_lshl_add_u64 v[50:51], v[50:51], 0, v[100:101]
	s_nop 0
	v_subb_co_u32_e32 v48, vcc, v48, v104, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_lshl_add_u64 v[50:51], v[50:51], 0, v[102:103]
	v_add3_u32 v87, v48, v87, v43
	v_sub_co_u32_e32 v43, vcc, -3, v50
	v_lshl_add_u64 v[100:101], v[50:51], 0, 2
	s_nop 0
	v_subb_co_u32_e32 v48, vcc, -1, v51, vcc
	v_cmp_gt_i32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v103, v101, v48, vcc
	v_ashrrev_i32_e32 v104, 31, v103
	v_cndmask_b32_e32 v102, v100, v43, vcc
	v_mov_b32_e32 v105, v104
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[104:105]
	v_xor_b32_e32 v83, v102, v104
	v_xor_b32_e32 v43, v103, v104
	v_mad_u64_u32 v[102:103], s[2:3], v83, s20, 0
	v_mul_hi_u32 v48, v83, s19
	v_lshl_add_u64 v[102:103], v[48:49], 0, v[102:103]
	v_mad_u64_u32 v[108:109], s[2:3], v43, s19, 0
	v_add_co_u32_e32 v48, vcc, v102, v108
	v_mad_u64_u32 v[106:107], s[2:3], v43, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v102, vcc, v103, v109, vcc
	v_mov_b32_e32 v103, s18
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[106:107]
	v_mul_lo_u32 v48, s17, v102
	v_mul_lo_u32 v89, s16, v103
	v_mad_u64_u32 v[106:107], s[2:3], s16, v102, 0
	v_add3_u32 v48, v107, v89, v48
	v_sub_u32_e32 v89, v43, v48
	v_sub_co_u32_e32 v83, vcc, v83, v106
	s_nop 1
	v_subb_co_u32_e64 v89, s[2:3], v89, v47, vcc
	v_subrev_co_u32_e64 v91, s[2:3], s16, v83
	v_subb_co_u32_e32 v43, vcc, v43, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v89, s[2:3], 0, v89, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v89
	v_cmp_le_u32_e32 vcc, s17, v43
	s_nop 0
	v_cndmask_b32_e64 v93, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v91
	v_cndmask_b32_e64 v48, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v83
	v_cndmask_b32_e64 v91, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v89
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v43
	v_cndmask_b32_e64 v89, v93, v91, s[2:3]
	v_add_u32_e32 v91, 2, v102
	v_add_u32_e32 v93, 1, v102
	v_cmp_ne_u32_e64 s[2:3], 0, v89
	v_cndmask_b32_e32 v43, v48, v83, vcc
	v_cmp_ne_u32_e32 vcc, 0, v43
	v_cndmask_b32_e64 v89, v93, v91, s[2:3]
	v_xor_b32_e32 v48, s70, v104
	v_cndmask_b32_e32 v43, v102, v89, vcc
	v_ashrrev_i32_e32 v102, 31, v101
	v_mov_b32_e32 v103, v102
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[102:103]
	v_xor_b32_e32 v43, v43, v48
	v_xor_b32_e32 v89, v100, v102
	v_sub_u32_e32 v43, v43, v48
	v_xor_b32_e32 v83, v101, v102
	v_mad_u64_u32 v[100:101], s[2:3], v89, s20, 0
	v_mul_hi_u32 v48, v89, s19
	v_lshl_add_u64 v[100:101], v[48:49], 0, v[100:101]
	v_mad_u64_u32 v[106:107], s[2:3], v83, s19, 0
	v_add_co_u32_e32 v48, vcc, v100, v106
	v_mad_u64_u32 v[104:105], s[2:3], v83, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v100, vcc, v101, v107, vcc
	v_mov_b32_e32 v101, s18
	s_nop 0
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[104:105]
	v_mul_lo_u32 v48, s17, v100
	v_mul_lo_u32 v91, s16, v101
	v_mad_u64_u32 v[100:101], s[2:3], s16, v100, 0
	v_add3_u32 v48, v101, v91, v48
	v_sub_u32_e32 v91, v83, v48
	v_sub_co_u32_e32 v89, vcc, v89, v100
	v_xad_u32 v43, v43, v102, v46
	s_nop 0
	v_subb_co_u32_e64 v91, s[2:3], v91, v47, vcc
	v_subrev_co_u32_e64 v93, s[2:3], s16, v89
	v_subb_co_u32_e32 v48, vcc, v83, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v95, s[14:15], 0, v91, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v95
	v_subb_co_u32_e64 v91, s[2:3], v91, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v100, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v93
	v_cmp_le_u32_e32 vcc, s17, v48
	v_mul_lo_u32 v43, v43, s33
	v_cndmask_b32_e64 v101, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v95
	v_cndmask_b32_e64 v83, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v89
	v_cndmask_b32_e64 v100, v100, v101, s[14:15]
	v_subrev_co_u32_e64 v101, s[2:3], s16, v93
	s_nop 1
	v_subbrev_co_u32_e64 v91, s[2:3], 0, v91, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v100
	s_nop 1
	v_cndmask_b32_e64 v91, v95, v91, s[2:3]
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v83, v83, v95, vcc
	v_cmp_ne_u32_e32 vcc, 0, v83
	v_cndmask_b32_e64 v83, v93, v101, s[2:3]
	v_lshl_add_u64 v[100:101], v[52:53], 0, 3
	v_cndmask_b32_e32 v83, v89, v83, vcc
	v_cndmask_b32_e32 v48, v48, v91, vcc
	v_xor_b32_e32 v83, v83, v102
	v_xor_b32_e32 v48, v48, v102
	v_sub_co_u32_e32 v83, vcc, v83, v102
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v102, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v83, v48, v83, v43
	v_sub_co_u32_e32 v43, vcc, -4, v52
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, -1, v53, vcc
	v_cmp_gt_i32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v103, v101, v48, vcc
	v_ashrrev_i32_e32 v104, 31, v103
	v_cndmask_b32_e32 v102, v100, v43, vcc
	v_mov_b32_e32 v105, v104
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[104:105]
	v_xor_b32_e32 v89, v102, v104
	v_xor_b32_e32 v43, v103, v104
	v_mad_u64_u32 v[102:103], s[2:3], v89, s20, 0
	v_mul_hi_u32 v48, v89, s19
	v_lshl_add_u64 v[102:103], v[48:49], 0, v[102:103]
	v_mad_u64_u32 v[108:109], s[2:3], v43, s19, 0
	v_add_co_u32_e32 v48, vcc, v102, v108
	v_mad_u64_u32 v[106:107], s[2:3], v43, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v102, vcc, v103, v109, vcc
	v_mov_b32_e32 v103, s18
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[106:107]
	v_mul_lo_u32 v48, s17, v102
	v_mul_lo_u32 v91, s16, v103
	v_mad_u64_u32 v[106:107], s[2:3], s16, v102, 0
	v_add3_u32 v48, v107, v91, v48
	v_sub_u32_e32 v91, v43, v48
	v_sub_co_u32_e32 v89, vcc, v89, v106
	s_nop 1
	v_subb_co_u32_e64 v91, s[2:3], v91, v47, vcc
	v_subrev_co_u32_e64 v93, s[2:3], s16, v89
	v_subb_co_u32_e32 v43, vcc, v43, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v91, s[2:3], 0, v91, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v91
	v_cmp_le_u32_e32 vcc, s17, v43
	s_nop 0
	v_cndmask_b32_e64 v95, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v93
	v_cndmask_b32_e64 v48, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v89
	v_cndmask_b32_e64 v93, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v91
	v_cndmask_b32_e64 v89, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v43
	v_cndmask_b32_e64 v91, v95, v93, s[2:3]
	v_add_u32_e32 v93, 2, v102
	v_add_u32_e32 v95, 1, v102
	v_cmp_ne_u32_e64 s[2:3], 0, v91
	v_cndmask_b32_e32 v43, v48, v89, vcc
	v_cmp_ne_u32_e32 vcc, 0, v43
	v_cndmask_b32_e64 v91, v95, v93, s[2:3]
	v_xor_b32_e32 v48, s70, v104
	v_cndmask_b32_e32 v43, v102, v91, vcc
	v_ashrrev_i32_e32 v102, 31, v101
	v_mov_b32_e32 v103, v102
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[102:103]
	v_xor_b32_e32 v43, v43, v48
	v_xor_b32_e32 v91, v100, v102
	v_sub_u32_e32 v43, v43, v48
	v_xor_b32_e32 v89, v101, v102
	v_mad_u64_u32 v[100:101], s[2:3], v91, s20, 0
	v_mul_hi_u32 v48, v91, s19
	v_lshl_add_u64 v[100:101], v[48:49], 0, v[100:101]
	v_mad_u64_u32 v[106:107], s[2:3], v89, s19, 0
	v_add_co_u32_e32 v48, vcc, v100, v106
	v_mad_u64_u32 v[104:105], s[2:3], v89, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v100, vcc, v101, v107, vcc
	v_mov_b32_e32 v101, s18
	s_nop 0
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[104:105]
	v_mul_lo_u32 v48, s17, v100
	v_mul_lo_u32 v93, s16, v101
	v_mad_u64_u32 v[100:101], s[2:3], s16, v100, 0
	v_add3_u32 v48, v101, v93, v48
	v_sub_u32_e32 v93, v89, v48
	v_sub_co_u32_e32 v91, vcc, v91, v100
	v_xad_u32 v43, v43, v102, v46
	s_nop 0
	v_subb_co_u32_e64 v93, s[2:3], v93, v47, vcc
	v_subrev_co_u32_e64 v95, s[2:3], s16, v91
	v_subb_co_u32_e32 v48, vcc, v89, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v100, s[14:15], 0, v93, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v100
	v_subb_co_u32_e64 v93, s[2:3], v93, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v101, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v95
	v_cmp_le_u32_e32 vcc, s17, v48
	v_mul_lo_u32 v43, v43, s33
	v_cndmask_b32_e64 v103, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v100
	v_cndmask_b32_e64 v89, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v91
	v_cndmask_b32_e64 v101, v101, v103, s[14:15]
	v_subrev_co_u32_e64 v103, s[2:3], s16, v95
	s_nop 1
	v_subbrev_co_u32_e64 v93, s[2:3], 0, v93, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v101
	s_nop 1
	v_cndmask_b32_e64 v93, v100, v93, s[2:3]
	v_cndmask_b32_e64 v100, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v89, v89, v100, vcc
	v_cmp_ne_u32_e32 vcc, 0, v89
	v_cndmask_b32_e64 v89, v95, v103, s[2:3]
	v_lshl_add_u64 v[100:101], v[50:51], 0, 4
	v_cndmask_b32_e32 v89, v91, v89, vcc
	v_cndmask_b32_e32 v48, v48, v93, vcc
	v_xor_b32_e32 v89, v89, v102
	v_xor_b32_e32 v48, v48, v102
	v_sub_co_u32_e32 v89, vcc, v89, v102
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v102, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v89, v48, v89, v43
	v_sub_co_u32_e32 v43, vcc, -5, v50
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, -1, v51, vcc
	v_cmp_gt_i32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v103, v101, v48, vcc
	v_ashrrev_i32_e32 v104, 31, v103
	v_cndmask_b32_e32 v102, v100, v43, vcc
	v_mov_b32_e32 v105, v104
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[104:105]
	v_xor_b32_e32 v91, v102, v104
	v_xor_b32_e32 v43, v103, v104
	v_mad_u64_u32 v[102:103], s[2:3], v91, s20, 0
	v_mul_hi_u32 v48, v91, s19
	v_lshl_add_u64 v[102:103], v[48:49], 0, v[102:103]
	v_mad_u64_u32 v[108:109], s[2:3], v43, s19, 0
	v_add_co_u32_e32 v48, vcc, v102, v108
	v_mad_u64_u32 v[106:107], s[2:3], v43, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v102, vcc, v103, v109, vcc
	v_mov_b32_e32 v103, s18
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[106:107]
	v_mul_lo_u32 v48, s17, v102
	v_mul_lo_u32 v93, s16, v103
	v_mad_u64_u32 v[106:107], s[2:3], s16, v102, 0
	v_add3_u32 v48, v107, v93, v48
	v_sub_u32_e32 v93, v43, v48
	v_sub_co_u32_e32 v91, vcc, v91, v106
	s_nop 1
	v_subb_co_u32_e64 v93, s[2:3], v93, v47, vcc
	v_subrev_co_u32_e64 v95, s[2:3], s16, v91
	v_subb_co_u32_e32 v43, vcc, v43, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v93, s[2:3], 0, v93, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v93
	v_cmp_le_u32_e32 vcc, s17, v43
	s_nop 0
	v_cndmask_b32_e64 v103, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v95
	v_cndmask_b32_e64 v48, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v91
	v_cndmask_b32_e64 v95, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v93
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v43
	v_cndmask_b32_e64 v93, v103, v95, s[2:3]
	v_add_u32_e32 v95, 2, v102
	v_add_u32_e32 v103, 1, v102
	v_cmp_ne_u32_e64 s[2:3], 0, v93
	v_cndmask_b32_e32 v43, v48, v91, vcc
	v_cmp_ne_u32_e32 vcc, 0, v43
	v_cndmask_b32_e64 v93, v103, v95, s[2:3]
	v_xor_b32_e32 v48, s70, v104
	v_cndmask_b32_e32 v43, v102, v93, vcc
	v_ashrrev_i32_e32 v102, 31, v101
	v_mov_b32_e32 v103, v102
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[102:103]
	v_xor_b32_e32 v43, v43, v48
	v_xor_b32_e32 v93, v100, v102
	v_sub_u32_e32 v43, v43, v48
	v_xor_b32_e32 v91, v101, v102
	v_mad_u64_u32 v[100:101], s[2:3], v93, s20, 0
	v_mul_hi_u32 v48, v93, s19
	v_lshl_add_u64 v[100:101], v[48:49], 0, v[100:101]
	v_mad_u64_u32 v[106:107], s[2:3], v91, s19, 0
	v_add_co_u32_e32 v48, vcc, v100, v106
	v_mad_u64_u32 v[104:105], s[2:3], v91, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v100, vcc, v101, v107, vcc
	v_mov_b32_e32 v101, s18
	s_nop 0
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[104:105]
	v_mul_lo_u32 v48, s17, v100
	v_mul_lo_u32 v95, s16, v101
	v_mad_u64_u32 v[100:101], s[2:3], s16, v100, 0
	v_add3_u32 v48, v101, v95, v48
	v_sub_u32_e32 v95, v91, v48
	v_sub_co_u32_e32 v93, vcc, v93, v100
	v_xad_u32 v43, v43, v102, v46
	s_nop 0
	v_subb_co_u32_e64 v95, s[2:3], v95, v47, vcc
	v_subrev_co_u32_e64 v100, s[2:3], s16, v93
	v_subb_co_u32_e32 v48, vcc, v91, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v101, s[14:15], 0, v95, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v101
	v_subb_co_u32_e64 v95, s[2:3], v95, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v103, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v100
	v_cmp_le_u32_e32 vcc, s17, v48
	v_mul_lo_u32 v43, v43, s33
	v_cndmask_b32_e64 v104, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v101
	v_cndmask_b32_e64 v91, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v93
	v_cndmask_b32_e64 v103, v103, v104, s[14:15]
	v_subrev_co_u32_e64 v104, s[2:3], s16, v100
	s_nop 1
	v_subbrev_co_u32_e64 v95, s[2:3], 0, v95, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v103
	s_nop 1
	v_cndmask_b32_e64 v95, v101, v95, s[2:3]
	v_cndmask_b32_e64 v101, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v91, v91, v101, vcc
	v_cmp_ne_u32_e32 vcc, 0, v91
	v_cndmask_b32_e64 v91, v100, v104, s[2:3]
	v_lshl_add_u64 v[100:101], v[52:53], 0, 5
	v_cndmask_b32_e32 v91, v93, v91, vcc
	v_cndmask_b32_e32 v48, v48, v95, vcc
	v_xor_b32_e32 v91, v91, v102
	v_xor_b32_e32 v48, v48, v102
	v_sub_co_u32_e32 v91, vcc, v91, v102
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v102, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v91, v48, v91, v43
	v_sub_co_u32_e32 v43, vcc, -6, v52
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, -1, v53, vcc
	v_cmp_gt_i32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v103, v101, v48, vcc
	v_ashrrev_i32_e32 v104, 31, v103
	v_cndmask_b32_e32 v102, v100, v43, vcc
	v_mov_b32_e32 v105, v104
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[104:105]
	v_xor_b32_e32 v93, v102, v104
	v_xor_b32_e32 v43, v103, v104
	v_mad_u64_u32 v[102:103], s[2:3], v93, s20, 0
	v_mul_hi_u32 v48, v93, s19
	v_lshl_add_u64 v[102:103], v[48:49], 0, v[102:103]
	v_mad_u64_u32 v[108:109], s[2:3], v43, s19, 0
	v_add_co_u32_e32 v48, vcc, v102, v108
	v_mad_u64_u32 v[106:107], s[2:3], v43, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v102, vcc, v103, v109, vcc
	v_mov_b32_e32 v103, s18
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[106:107]
	v_mul_lo_u32 v48, s17, v102
	v_mul_lo_u32 v95, s16, v103
	v_mad_u64_u32 v[106:107], s[2:3], s16, v102, 0
	v_add3_u32 v48, v107, v95, v48
	v_sub_u32_e32 v95, v43, v48
	v_sub_co_u32_e32 v93, vcc, v93, v106
	s_nop 1
	v_subb_co_u32_e64 v95, s[2:3], v95, v47, vcc
	v_subrev_co_u32_e64 v103, s[2:3], s16, v93
	v_subb_co_u32_e32 v43, vcc, v43, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v95, s[2:3], 0, v95, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v95
	v_cmp_le_u32_e32 vcc, s17, v43
	s_nop 0
	v_cndmask_b32_e64 v105, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v103
	v_cndmask_b32_e64 v48, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v93
	v_cndmask_b32_e64 v103, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v95
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v43
	v_cndmask_b32_e64 v95, v105, v103, s[2:3]
	v_add_u32_e32 v103, 2, v102
	v_add_u32_e32 v105, 1, v102
	v_cmp_ne_u32_e64 s[2:3], 0, v95
	v_cndmask_b32_e32 v43, v48, v93, vcc
	v_cmp_ne_u32_e32 vcc, 0, v43
	v_cndmask_b32_e64 v95, v105, v103, s[2:3]
	v_xor_b32_e32 v48, s70, v104
	v_cndmask_b32_e32 v43, v102, v95, vcc
	v_ashrrev_i32_e32 v102, 31, v101
	v_mov_b32_e32 v103, v102
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[102:103]
	v_xor_b32_e32 v43, v43, v48
	v_xor_b32_e32 v95, v100, v102
	v_sub_u32_e32 v43, v43, v48
	v_xor_b32_e32 v93, v101, v102
	v_mad_u64_u32 v[100:101], s[2:3], v95, s20, 0
	v_mul_hi_u32 v48, v95, s19
	v_lshl_add_u64 v[100:101], v[48:49], 0, v[100:101]
	v_mad_u64_u32 v[106:107], s[2:3], v93, s19, 0
	v_add_co_u32_e32 v48, vcc, v100, v106
	v_mad_u64_u32 v[104:105], s[2:3], v93, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v100, vcc, v101, v107, vcc
	v_mov_b32_e32 v101, s18
	s_nop 0
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[104:105]
	v_mul_lo_u32 v48, s17, v100
	v_mul_lo_u32 v103, s16, v101
	v_mad_u64_u32 v[100:101], s[2:3], s16, v100, 0
	v_add3_u32 v48, v101, v103, v48
	v_sub_u32_e32 v101, v93, v48
	v_sub_co_u32_e32 v95, vcc, v95, v100
	v_xad_u32 v43, v43, v102, v46
	s_nop 0
	v_subb_co_u32_e64 v100, s[2:3], v101, v47, vcc
	v_subrev_co_u32_e64 v101, s[2:3], s16, v95
	v_subb_co_u32_e32 v48, vcc, v93, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v103, s[14:15], 0, v100, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v103
	v_subb_co_u32_e64 v100, s[2:3], v100, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v104, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v101
	v_cmp_le_u32_e32 vcc, s17, v48
	v_mul_lo_u32 v43, v43, s33
	v_cndmask_b32_e64 v105, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v103
	v_cndmask_b32_e64 v93, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v95
	v_cndmask_b32_e64 v104, v104, v105, s[14:15]
	v_subrev_co_u32_e64 v105, s[2:3], s16, v101
	s_nop 1
	v_subbrev_co_u32_e64 v100, s[2:3], 0, v100, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v104
	s_nop 1
	v_cndmask_b32_e64 v100, v103, v100, s[2:3]
	v_cndmask_b32_e64 v103, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v93, v93, v103, vcc
	v_cmp_ne_u32_e32 vcc, 0, v93
	v_cndmask_b32_e64 v93, v101, v105, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v93, v95, v93, vcc
	v_cndmask_b32_e32 v48, v48, v100, vcc
	v_xor_b32_e32 v93, v93, v102
	v_xor_b32_e32 v48, v48, v102
	v_sub_co_u32_e32 v93, vcc, v93, v102
	v_lshl_add_u64 v[100:101], v[50:51], 0, 6
	s_nop 0
	v_subb_co_u32_e32 v48, vcc, v48, v102, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v93, v48, v93, v43
	v_sub_co_u32_e32 v43, vcc, -7, v50
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, -1, v51, vcc
	v_cmp_gt_i32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v103, v101, v48, vcc
	v_ashrrev_i32_e32 v104, 31, v103
	v_cndmask_b32_e32 v102, v100, v43, vcc
	v_mov_b32_e32 v105, v104
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[104:105]
	v_xor_b32_e32 v95, v102, v104
	v_xor_b32_e32 v43, v103, v104
	v_mad_u64_u32 v[102:103], s[2:3], v95, s20, 0
	v_mul_hi_u32 v48, v95, s19
	v_lshl_add_u64 v[102:103], v[48:49], 0, v[102:103]
	v_mad_u64_u32 v[108:109], s[2:3], v43, s19, 0
	v_add_co_u32_e32 v48, vcc, v102, v108
	v_mad_u64_u32 v[106:107], s[2:3], v43, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v102, vcc, v103, v109, vcc
	v_mov_b32_e32 v103, s18
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[106:107]
	v_mul_lo_u32 v48, s17, v102
	v_mul_lo_u32 v103, s16, v103
	v_mad_u64_u32 v[106:107], s[2:3], s16, v102, 0
	v_add3_u32 v48, v107, v103, v48
	v_sub_u32_e32 v103, v43, v48
	v_sub_co_u32_e32 v95, vcc, v95, v106
	s_nop 1
	v_subb_co_u32_e64 v103, s[2:3], v103, v47, vcc
	v_subrev_co_u32_e64 v105, s[2:3], s16, v95
	v_subb_co_u32_e32 v43, vcc, v43, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v103, s[2:3], 0, v103, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v103
	v_cmp_le_u32_e32 vcc, s17, v43
	s_nop 0
	v_cndmask_b32_e64 v106, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v105
	v_cndmask_b32_e64 v48, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v95
	v_cndmask_b32_e64 v105, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v103
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v43
	v_cndmask_b32_e64 v103, v106, v105, s[2:3]
	v_add_u32_e32 v105, 2, v102
	v_add_u32_e32 v106, 1, v102
	v_cmp_ne_u32_e64 s[2:3], 0, v103
	v_cndmask_b32_e32 v43, v48, v95, vcc
	v_cmp_ne_u32_e32 vcc, 0, v43
	v_cndmask_b32_e64 v103, v106, v105, s[2:3]
	v_xor_b32_e32 v48, s70, v104
	v_cndmask_b32_e32 v43, v102, v103, vcc
	v_ashrrev_i32_e32 v102, 31, v101
	v_mov_b32_e32 v103, v102
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[102:103]
	v_xor_b32_e32 v43, v43, v48
	v_xor_b32_e32 v103, v100, v102
	v_sub_u32_e32 v43, v43, v48
	v_xor_b32_e32 v95, v101, v102
	v_mad_u64_u32 v[100:101], s[2:3], v103, s20, 0
	v_mul_hi_u32 v48, v103, s19
	v_lshl_add_u64 v[100:101], v[48:49], 0, v[100:101]
	v_mad_u64_u32 v[106:107], s[2:3], v95, s19, 0
	v_add_co_u32_e32 v48, vcc, v100, v106
	v_mad_u64_u32 v[104:105], s[2:3], v95, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v100, vcc, v101, v107, vcc
	v_mov_b32_e32 v101, s18
	s_nop 0
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[104:105]
	v_mul_lo_u32 v48, s17, v100
	v_mul_lo_u32 v104, s16, v101
	v_mad_u64_u32 v[100:101], s[2:3], s16, v100, 0
	v_add3_u32 v48, v101, v104, v48
	v_sub_u32_e32 v101, v95, v48
	v_sub_co_u32_e32 v100, vcc, v103, v100
	v_xad_u32 v43, v43, v102, v46
	s_nop 0
	v_subb_co_u32_e64 v101, s[2:3], v101, v47, vcc
	v_subrev_co_u32_e64 v103, s[2:3], s16, v100
	v_subb_co_u32_e32 v48, vcc, v95, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v104, s[14:15], 0, v101, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v104
	v_subb_co_u32_e64 v101, s[2:3], v101, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v105, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v103
	v_cmp_le_u32_e32 vcc, s17, v48
	v_mul_lo_u32 v43, v43, s33
	v_cndmask_b32_e64 v106, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v104
	v_cndmask_b32_e64 v95, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v100
	v_cndmask_b32_e64 v105, v105, v106, s[14:15]
	v_subrev_co_u32_e64 v106, s[2:3], s16, v103
	s_nop 1
	v_subbrev_co_u32_e64 v101, s[2:3], 0, v101, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v105
	s_nop 1
	v_cndmask_b32_e64 v101, v104, v101, s[2:3]
	v_cndmask_b32_e64 v104, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v95, v95, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v95
	v_cndmask_b32_e64 v95, v103, v106, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v95, v100, v95, vcc
	v_cndmask_b32_e32 v48, v48, v101, vcc
	v_xor_b32_e32 v95, v95, v102
	v_xor_b32_e32 v48, v48, v102
	v_sub_co_u32_e32 v95, vcc, v95, v102
	v_lshl_add_u64 v[100:101], v[52:53], 0, 7
	s_nop 0
	v_subb_co_u32_e32 v48, vcc, v48, v102, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v95, v48, v95, v43
	v_sub_co_u32_e32 v43, vcc, -8, v52
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, -1, v53, vcc
	v_cmp_gt_i32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v103, v101, v48, vcc
	v_ashrrev_i32_e32 v104, 31, v103
	v_cndmask_b32_e32 v102, v100, v43, vcc
	v_mov_b32_e32 v105, v104
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[104:105]
	v_xor_b32_e32 v105, v102, v104
	v_xor_b32_e32 v43, v103, v104
	v_mad_u64_u32 v[102:103], s[2:3], v105, s20, 0
	v_mul_hi_u32 v48, v105, s19
	v_lshl_add_u64 v[102:103], v[48:49], 0, v[102:103]
	v_mad_u64_u32 v[108:109], s[2:3], v43, s19, 0
	v_add_co_u32_e32 v48, vcc, v102, v108
	v_mad_u64_u32 v[106:107], s[2:3], v43, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v102, vcc, v103, v109, vcc
	v_mov_b32_e32 v103, s18
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[106:107]
	v_mul_lo_u32 v48, s17, v102
	v_mul_lo_u32 v103, s16, v103
	v_mad_u64_u32 v[106:107], s[2:3], s16, v102, 0
	v_add3_u32 v48, v107, v103, v48
	v_sub_u32_e32 v103, v43, v48
	v_sub_co_u32_e32 v105, vcc, v105, v106
	s_nop 1
	v_subb_co_u32_e64 v103, s[2:3], v103, v47, vcc
	v_subrev_co_u32_e64 v106, s[2:3], s16, v105
	v_subb_co_u32_e32 v43, vcc, v43, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v103, s[2:3], 0, v103, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v103
	v_cmp_le_u32_e32 vcc, s17, v43
	s_nop 0
	v_cndmask_b32_e64 v107, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v106
	v_cndmask_b32_e64 v48, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v105
	v_cndmask_b32_e64 v106, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v103
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v43
	v_cndmask_b32_e64 v103, v107, v106, s[2:3]
	v_add_u32_e32 v106, 2, v102
	v_add_u32_e32 v107, 1, v102
	v_cmp_ne_u32_e64 s[2:3], 0, v103
	v_cndmask_b32_e32 v43, v48, v105, vcc
	v_cmp_ne_u32_e32 vcc, 0, v43
	v_cndmask_b32_e64 v103, v107, v106, s[2:3]
	v_xor_b32_e32 v48, s70, v104
	v_cndmask_b32_e32 v43, v102, v103, vcc
	v_ashrrev_i32_e32 v102, 31, v101
	v_mov_b32_e32 v103, v102
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[102:103]
	v_xor_b32_e32 v43, v43, v48
	v_xor_b32_e32 v108, v100, v102
	v_sub_u32_e32 v43, v43, v48
	v_xor_b32_e32 v103, v101, v102
	v_mad_u64_u32 v[100:101], s[2:3], v108, s20, 0
	v_mul_hi_u32 v48, v108, s19
	v_lshl_add_u64 v[100:101], v[48:49], 0, v[100:101]
	v_mad_u64_u32 v[106:107], s[2:3], v103, s19, 0
	v_add_co_u32_e32 v48, vcc, v100, v106
	v_mad_u64_u32 v[104:105], s[2:3], v103, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v100, vcc, v101, v107, vcc
	v_mov_b32_e32 v101, s18
	s_nop 0
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[104:105]
	v_mul_lo_u32 v48, s17, v100
	v_mul_lo_u32 v104, s16, v101
	v_mad_u64_u32 v[100:101], s[2:3], s16, v100, 0
	v_add3_u32 v48, v101, v104, v48
	v_sub_u32_e32 v101, v103, v48
	v_sub_co_u32_e32 v100, vcc, v108, v100
	v_xad_u32 v43, v43, v102, v46
	s_nop 0
	v_subb_co_u32_e64 v101, s[2:3], v101, v47, vcc
	v_subrev_co_u32_e64 v104, s[2:3], s16, v100
	v_subb_co_u32_e32 v48, vcc, v103, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v105, s[14:15], 0, v101, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v105
	v_subb_co_u32_e64 v101, s[2:3], v101, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v106, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v104
	v_cmp_le_u32_e32 vcc, s17, v48
	v_mul_lo_u32 v43, v43, s33
	v_cndmask_b32_e64 v107, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v105
	v_cndmask_b32_e64 v103, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v100
	v_cndmask_b32_e64 v106, v106, v107, s[14:15]
	v_subrev_co_u32_e64 v107, s[2:3], s16, v104
	s_nop 1
	v_subbrev_co_u32_e64 v101, s[2:3], 0, v101, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v106
	s_nop 1
	v_cndmask_b32_e64 v101, v105, v101, s[2:3]
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v103, v103, v105, vcc
	v_cmp_ne_u32_e32 vcc, 0, v103
	s_nop 1
	v_cndmask_b32_e32 v48, v48, v101, vcc
	v_cndmask_b32_e64 v101, v104, v107, s[2:3]
	v_cndmask_b32_e32 v100, v100, v101, vcc
	v_xor_b32_e32 v100, v100, v102
	v_xor_b32_e32 v48, v48, v102
	v_sub_co_u32_e32 v100, vcc, v100, v102
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v102, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v48, v48, v100, v43
	buffer_load_ubyte v43, v85, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v85, v87, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v83, v83, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v87, v89, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v89, v91, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v93, v93, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v91, v95, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v95, v48, s[36:39], 0 offen
	v_sub_co_u32_e32 v48, vcc, -9, v50
	v_lshl_add_u64 v[100:101], v[50:51], 0, 8
	s_nop 0
	v_subb_co_u32_e32 v102, vcc, -1, v51, vcc
	v_cmp_gt_i32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v103, v101, v102, vcc
	v_ashrrev_i32_e32 v104, 31, v103
	v_cndmask_b32_e32 v102, v100, v48, vcc
	v_mov_b32_e32 v105, v104
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[104:105]
	v_xor_b32_e32 v110, v102, v104
	v_xor_b32_e32 v105, v103, v104
	v_mad_u64_u32 v[102:103], s[2:3], v110, s20, 0
	v_mul_hi_u32 v48, v110, s19
	v_lshl_add_u64 v[102:103], v[48:49], 0, v[102:103]
	v_mad_u64_u32 v[108:109], s[2:3], v105, s19, 0
	v_add_co_u32_e32 v48, vcc, v102, v108
	v_mad_u64_u32 v[106:107], s[2:3], v105, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v102, vcc, v103, v109, vcc
	v_mov_b32_e32 v103, s18
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[106:107]
	v_mul_lo_u32 v48, s17, v102
	v_mul_lo_u32 v103, s16, v103
	v_mad_u64_u32 v[106:107], s[2:3], s16, v102, 0
	v_add3_u32 v48, v107, v103, v48
	v_sub_u32_e32 v103, v105, v48
	v_sub_co_u32_e32 v106, vcc, v110, v106
	s_nop 1
	v_subb_co_u32_e64 v103, s[2:3], v103, v47, vcc
	v_subrev_co_u32_e64 v107, s[2:3], s16, v106
	v_subb_co_u32_e32 v48, vcc, v105, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v103, s[2:3], 0, v103, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v103
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v108, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v107
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v106
	v_cndmask_b32_e64 v107, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v103
	v_cndmask_b32_e64 v106, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v103, v108, v107, s[2:3]
	v_add_u32_e32 v107, 2, v102
	v_add_u32_e32 v108, 1, v102
	v_cmp_ne_u32_e64 s[2:3], 0, v103
	v_cndmask_b32_e32 v48, v105, v106, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v103, v108, v107, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v48, v102, v103, vcc
	v_xor_b32_e32 v102, s70, v104
	v_xor_b32_e32 v48, v48, v102
	v_sub_u32_e32 v48, v48, v102
	v_ashrrev_i32_e32 v102, 31, v101
	v_mov_b32_e32 v103, v102
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[102:103]
	v_xor_b32_e32 v109, v100, v102
	v_xad_u32 v108, v48, v102, v46
	v_xor_b32_e32 v103, v101, v102
	v_mad_u64_u32 v[100:101], s[2:3], v109, s20, 0
	v_mul_hi_u32 v48, v109, s19
	v_lshl_add_u64 v[100:101], v[48:49], 0, v[100:101]
	v_mad_u64_u32 v[106:107], s[2:3], v103, s19, 0
	v_add_co_u32_e32 v48, vcc, v100, v106
	v_mad_u64_u32 v[104:105], s[2:3], v103, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v100, vcc, v101, v107, vcc
	v_mov_b32_e32 v101, s18
	s_nop 0
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[104:105]
	v_mul_lo_u32 v48, s17, v100
	v_mul_lo_u32 v104, s16, v101
	v_mad_u64_u32 v[100:101], s[2:3], s16, v100, 0
	v_add3_u32 v48, v101, v104, v48
	v_sub_u32_e32 v101, v103, v48
	v_sub_co_u32_e32 v100, vcc, v109, v100
	s_nop 1
	v_subb_co_u32_e64 v101, s[2:3], v101, v47, vcc
	v_subrev_co_u32_e64 v104, s[2:3], s16, v100
	v_subb_co_u32_e32 v48, vcc, v103, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v105, s[14:15], 0, v101, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v105
	v_subb_co_u32_e64 v101, s[2:3], v101, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v106, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v104
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v107, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v105
	v_cndmask_b32_e64 v103, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v100
	v_cndmask_b32_e64 v106, v106, v107, s[14:15]
	v_subrev_co_u32_e64 v107, s[2:3], s16, v104
	s_nop 1
	v_subbrev_co_u32_e64 v101, s[2:3], 0, v101, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v106
	s_nop 1
	v_cndmask_b32_e64 v101, v105, v101, s[2:3]
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v103, v103, v105, vcc
	v_cmp_ne_u32_e32 vcc, 0, v103
	s_nop 1
	v_cndmask_b32_e32 v48, v48, v101, vcc
	v_cndmask_b32_e64 v101, v104, v107, s[2:3]
	v_cndmask_b32_e32 v100, v100, v101, vcc
	v_xor_b32_e32 v100, v100, v102
	v_xor_b32_e32 v48, v48, v102
	v_sub_co_u32_e32 v100, vcc, v100, v102
	v_mul_lo_u32 v101, v108, s33
	s_nop 0
	v_subb_co_u32_e32 v48, vcc, v48, v102, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v100, v48, v100, v101
	v_sub_co_u32_e32 v48, vcc, -10, v52
	v_lshl_add_u64 v[102:103], v[52:53], 0, 9
	s_nop 0
	v_subb_co_u32_e32 v101, vcc, -1, v53, vcc
	v_cmp_gt_i32_e32 vcc, 0, v103
	s_nop 1
	v_cndmask_b32_e32 v105, v103, v101, vcc
	v_ashrrev_i32_e32 v106, 31, v105
	v_cndmask_b32_e32 v104, v102, v48, vcc
	v_mov_b32_e32 v107, v106
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[106:107]
	v_xor_b32_e32 v107, v104, v106
	v_xor_b32_e32 v101, v105, v106
	v_mad_u64_u32 v[104:105], s[2:3], v107, s20, 0
	v_mul_hi_u32 v48, v107, s19
	v_lshl_add_u64 v[104:105], v[48:49], 0, v[104:105]
	v_mad_u64_u32 v[110:111], s[2:3], v101, s19, 0
	v_add_co_u32_e32 v48, vcc, v104, v110
	v_mad_u64_u32 v[108:109], s[2:3], v101, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v105, v111, vcc
	v_mov_b32_e32 v105, s18
	s_nop 0
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[108:109]
	v_mul_lo_u32 v48, s17, v104
	v_mul_lo_u32 v105, s16, v105
	v_mad_u64_u32 v[108:109], s[2:3], s16, v104, 0
	v_add3_u32 v48, v109, v105, v48
	v_sub_u32_e32 v105, v101, v48
	v_sub_co_u32_e32 v107, vcc, v107, v108
	s_nop 1
	v_subb_co_u32_e64 v105, s[2:3], v105, v47, vcc
	v_subrev_co_u32_e64 v108, s[2:3], s16, v107
	v_subb_co_u32_e32 v48, vcc, v101, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v105, s[2:3], 0, v105, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v105
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v109, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v108
	v_cndmask_b32_e64 v101, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v107
	v_cndmask_b32_e64 v108, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v105
	v_cndmask_b32_e64 v107, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v105, v109, v108, s[2:3]
	v_add_u32_e32 v108, 2, v104
	v_add_u32_e32 v109, 1, v104
	v_cmp_ne_u32_e64 s[2:3], 0, v105
	v_cndmask_b32_e32 v48, v101, v107, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v105, v109, v108, s[2:3]
	v_xor_b32_e32 v101, s70, v106
	v_cndmask_b32_e32 v48, v104, v105, vcc
	v_ashrrev_i32_e32 v104, 31, v103
	v_mov_b32_e32 v105, v104
	v_xor_b32_e32 v48, v48, v101
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[104:105]
	v_sub_u32_e32 v48, v48, v101
	v_xor_b32_e32 v110, v102, v104
	v_xad_u32 v101, v48, v104, v46
	v_xor_b32_e32 v105, v103, v104
	v_mad_u64_u32 v[102:103], s[2:3], v110, s20, 0
	v_mul_hi_u32 v48, v110, s19
	v_lshl_add_u64 v[102:103], v[48:49], 0, v[102:103]
	v_mad_u64_u32 v[108:109], s[2:3], v105, s19, 0
	v_add_co_u32_e32 v48, vcc, v102, v108
	v_mad_u64_u32 v[106:107], s[2:3], v105, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v102, vcc, v103, v109, vcc
	v_mov_b32_e32 v103, s18
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[106:107]
	v_mul_lo_u32 v48, s17, v102
	v_mul_lo_u32 v106, s16, v103
	v_mad_u64_u32 v[102:103], s[2:3], s16, v102, 0
	v_add3_u32 v48, v103, v106, v48
	v_sub_u32_e32 v103, v105, v48
	v_sub_co_u32_e32 v102, vcc, v110, v102
	v_mul_lo_u32 v101, v101, s33
	s_nop 0
	v_subb_co_u32_e64 v103, s[2:3], v103, v47, vcc
	v_subrev_co_u32_e64 v106, s[2:3], s16, v102
	v_subb_co_u32_e32 v48, vcc, v105, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v107, s[14:15], 0, v103, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v107
	v_subb_co_u32_e64 v103, s[2:3], v103, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v108, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v106
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v109, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v107
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v102
	v_cndmask_b32_e64 v108, v108, v109, s[14:15]
	v_subrev_co_u32_e64 v109, s[2:3], s16, v106
	s_nop 1
	v_subbrev_co_u32_e64 v103, s[2:3], 0, v103, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v108
	s_nop 1
	v_cndmask_b32_e64 v103, v107, v103, s[2:3]
	v_cndmask_b32_e64 v107, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v105, v105, v107, vcc
	v_cmp_ne_u32_e32 vcc, 0, v105
	s_nop 1
	v_cndmask_b32_e32 v48, v48, v103, vcc
	v_cndmask_b32_e64 v103, v106, v109, s[2:3]
	v_cndmask_b32_e32 v102, v102, v103, vcc
	v_xor_b32_e32 v102, v102, v104
	v_xor_b32_e32 v48, v48, v104
	v_sub_co_u32_e32 v102, vcc, v102, v104
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v104, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v101, v48, v102, v101
	v_sub_co_u32_e32 v48, vcc, -11, v50
	v_lshl_add_u64 v[102:103], v[50:51], 0, 10
	s_nop 0
	v_subb_co_u32_e32 v104, vcc, -1, v51, vcc
	v_cmp_gt_i32_e32 vcc, 0, v103
	s_nop 1
	v_cndmask_b32_e32 v105, v103, v104, vcc
	v_ashrrev_i32_e32 v106, 31, v105
	v_cndmask_b32_e32 v104, v102, v48, vcc
	v_mov_b32_e32 v107, v106
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[106:107]
	v_xor_b32_e32 v112, v104, v106
	v_xor_b32_e32 v107, v105, v106
	v_mad_u64_u32 v[104:105], s[2:3], v112, s20, 0
	v_mul_hi_u32 v48, v112, s19
	v_lshl_add_u64 v[104:105], v[48:49], 0, v[104:105]
	v_mad_u64_u32 v[110:111], s[2:3], v107, s19, 0
	v_add_co_u32_e32 v48, vcc, v104, v110
	v_mad_u64_u32 v[108:109], s[2:3], v107, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v105, v111, vcc
	v_mov_b32_e32 v105, s18
	s_nop 0
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[108:109]
	v_mul_lo_u32 v48, s17, v104
	v_mul_lo_u32 v105, s16, v105
	v_mad_u64_u32 v[108:109], s[2:3], s16, v104, 0
	v_add3_u32 v48, v109, v105, v48
	v_sub_u32_e32 v105, v107, v48
	v_sub_co_u32_e32 v108, vcc, v112, v108
	s_nop 1
	v_subb_co_u32_e64 v105, s[2:3], v105, v47, vcc
	v_subrev_co_u32_e64 v109, s[2:3], s16, v108
	v_subb_co_u32_e32 v48, vcc, v107, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v105, s[2:3], 0, v105, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v105
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v110, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v109
	v_cndmask_b32_e64 v107, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v108
	v_cndmask_b32_e64 v109, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v105
	v_cndmask_b32_e64 v108, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v105, v110, v109, s[2:3]
	v_add_u32_e32 v109, 2, v104
	v_add_u32_e32 v110, 1, v104
	v_cmp_ne_u32_e64 s[2:3], 0, v105
	v_cndmask_b32_e32 v48, v107, v108, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v105, v110, v109, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v48, v104, v105, vcc
	v_xor_b32_e32 v104, s70, v106
	v_xor_b32_e32 v48, v48, v104
	v_sub_u32_e32 v48, v48, v104
	v_ashrrev_i32_e32 v104, 31, v103
	v_mov_b32_e32 v105, v104
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[104:105]
	v_xor_b32_e32 v111, v102, v104
	v_xad_u32 v110, v48, v104, v46
	v_xor_b32_e32 v105, v103, v104
	v_mad_u64_u32 v[102:103], s[2:3], v111, s20, 0
	v_mul_hi_u32 v48, v111, s19
	v_lshl_add_u64 v[102:103], v[48:49], 0, v[102:103]
	v_mad_u64_u32 v[108:109], s[2:3], v105, s19, 0
	v_add_co_u32_e32 v48, vcc, v102, v108
	v_mad_u64_u32 v[106:107], s[2:3], v105, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v102, vcc, v103, v109, vcc
	v_mov_b32_e32 v103, s18
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[106:107]
	v_mul_lo_u32 v48, s17, v102
	v_mul_lo_u32 v106, s16, v103
	v_mad_u64_u32 v[102:103], s[2:3], s16, v102, 0
	v_add3_u32 v48, v103, v106, v48
	v_sub_u32_e32 v103, v105, v48
	v_sub_co_u32_e32 v102, vcc, v111, v102
	s_nop 1
	v_subb_co_u32_e64 v103, s[2:3], v103, v47, vcc
	v_subrev_co_u32_e64 v106, s[2:3], s16, v102
	v_subb_co_u32_e32 v48, vcc, v105, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v107, s[14:15], 0, v103, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v107
	v_subb_co_u32_e64 v103, s[2:3], v103, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v108, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v106
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v109, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v107
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v102
	v_cndmask_b32_e64 v108, v108, v109, s[14:15]
	v_subrev_co_u32_e64 v109, s[2:3], s16, v106
	s_nop 1
	v_subbrev_co_u32_e64 v103, s[2:3], 0, v103, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v108
	s_nop 1
	v_cndmask_b32_e64 v103, v107, v103, s[2:3]
	v_cndmask_b32_e64 v107, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v105, v105, v107, vcc
	v_cmp_ne_u32_e32 vcc, 0, v105
	s_nop 1
	v_cndmask_b32_e32 v48, v48, v103, vcc
	v_cndmask_b32_e64 v103, v106, v109, s[2:3]
	v_cndmask_b32_e32 v102, v102, v103, vcc
	v_xor_b32_e32 v102, v102, v104
	v_xor_b32_e32 v48, v48, v104
	v_sub_co_u32_e32 v102, vcc, v102, v104
	v_mul_lo_u32 v103, v110, s33
	s_nop 0
	v_subb_co_u32_e32 v48, vcc, v48, v104, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v103, v48, v102, v103
	v_sub_co_u32_e32 v48, vcc, -12, v52
	v_lshl_add_u64 v[104:105], v[52:53], 0, 11
	s_nop 0
	v_subb_co_u32_e32 v102, vcc, -1, v53, vcc
	v_cmp_gt_i32_e32 vcc, 0, v105
	s_nop 1
	v_cndmask_b32_e32 v107, v105, v102, vcc
	v_ashrrev_i32_e32 v108, 31, v107
	v_cndmask_b32_e32 v106, v104, v48, vcc
	v_mov_b32_e32 v109, v108
	v_lshl_add_u64 v[106:107], v[106:107], 0, v[108:109]
	v_xor_b32_e32 v109, v106, v108
	v_xor_b32_e32 v102, v107, v108
	v_mad_u64_u32 v[106:107], s[2:3], v109, s20, 0
	v_mul_hi_u32 v48, v109, s19
	v_lshl_add_u64 v[106:107], v[48:49], 0, v[106:107]
	v_mad_u64_u32 v[112:113], s[2:3], v102, s19, 0
	v_add_co_u32_e32 v48, vcc, v106, v112
	v_mad_u64_u32 v[110:111], s[2:3], v102, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v106, vcc, v107, v113, vcc
	v_mov_b32_e32 v107, s18
	s_nop 0
	v_addc_co_u32_e32 v111, vcc, 0, v111, vcc
	v_lshl_add_u64 v[106:107], v[106:107], 0, v[110:111]
	v_mul_lo_u32 v48, s17, v106
	v_mul_lo_u32 v107, s16, v107
	v_mad_u64_u32 v[110:111], s[2:3], s16, v106, 0
	v_add3_u32 v48, v111, v107, v48
	v_sub_u32_e32 v107, v102, v48
	v_sub_co_u32_e32 v109, vcc, v109, v110
	s_nop 1
	v_subb_co_u32_e64 v107, s[2:3], v107, v47, vcc
	v_subrev_co_u32_e64 v110, s[2:3], s16, v109
	v_subb_co_u32_e32 v48, vcc, v102, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v107, s[2:3], 0, v107, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v107
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v111, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v110
	v_cndmask_b32_e64 v102, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v109
	v_cndmask_b32_e64 v110, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v107
	v_cndmask_b32_e64 v109, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v107, v111, v110, s[2:3]
	v_add_u32_e32 v110, 2, v106
	v_add_u32_e32 v111, 1, v106
	v_cmp_ne_u32_e64 s[2:3], 0, v107
	v_cndmask_b32_e32 v48, v102, v109, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v107, v111, v110, s[2:3]
	v_xor_b32_e32 v102, s70, v108
	v_cndmask_b32_e32 v48, v106, v107, vcc
	v_ashrrev_i32_e32 v106, 31, v105
	v_mov_b32_e32 v107, v106
	v_xor_b32_e32 v48, v48, v102
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[106:107]
	v_sub_u32_e32 v48, v48, v102
	v_xor_b32_e32 v112, v104, v106
	v_xad_u32 v102, v48, v106, v46
	v_xor_b32_e32 v107, v105, v106
	v_mad_u64_u32 v[104:105], s[2:3], v112, s20, 0
	v_mul_hi_u32 v48, v112, s19
	v_lshl_add_u64 v[104:105], v[48:49], 0, v[104:105]
	v_mad_u64_u32 v[110:111], s[2:3], v107, s19, 0
	v_add_co_u32_e32 v48, vcc, v104, v110
	v_mad_u64_u32 v[108:109], s[2:3], v107, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v104, vcc, v105, v111, vcc
	v_mov_b32_e32 v105, s18
	s_nop 0
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[108:109]
	v_mul_lo_u32 v48, s17, v104
	v_mul_lo_u32 v108, s16, v105
	v_mad_u64_u32 v[104:105], s[2:3], s16, v104, 0
	v_add3_u32 v48, v105, v108, v48
	v_sub_u32_e32 v105, v107, v48
	v_sub_co_u32_e32 v104, vcc, v112, v104
	v_mul_lo_u32 v102, v102, s33
	s_nop 0
	v_subb_co_u32_e64 v105, s[2:3], v105, v47, vcc
	v_subrev_co_u32_e64 v108, s[2:3], s16, v104
	v_subb_co_u32_e32 v48, vcc, v107, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v109, s[14:15], 0, v105, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v109
	v_subb_co_u32_e64 v105, s[2:3], v105, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v110, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v108
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v111, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v109
	v_cndmask_b32_e64 v107, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v104
	v_cndmask_b32_e64 v110, v110, v111, s[14:15]
	v_subrev_co_u32_e64 v111, s[2:3], s16, v108
	s_nop 1
	v_subbrev_co_u32_e64 v105, s[2:3], 0, v105, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v110
	s_nop 1
	v_cndmask_b32_e64 v105, v109, v105, s[2:3]
	v_cndmask_b32_e64 v109, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v107, v107, v109, vcc
	v_cmp_ne_u32_e32 vcc, 0, v107
	s_nop 1
	v_cndmask_b32_e32 v48, v48, v105, vcc
	v_cndmask_b32_e64 v105, v108, v111, s[2:3]
	v_cndmask_b32_e32 v104, v104, v105, vcc
	v_xor_b32_e32 v104, v104, v106
	v_xor_b32_e32 v48, v48, v106
	v_sub_co_u32_e32 v104, vcc, v104, v106
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v106, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v105, v48, v104, v102
	v_sub_co_u32_e32 v48, vcc, -13, v50
	v_lshl_add_u64 v[106:107], v[50:51], 0, 12
	s_nop 0
	v_subb_co_u32_e32 v102, vcc, -1, v51, vcc
	v_cmp_gt_i32_e32 vcc, 0, v107
	s_nop 1
	v_cndmask_b32_e32 v109, v107, v102, vcc
	v_ashrrev_i32_e32 v110, 31, v109
	v_cndmask_b32_e32 v108, v106, v48, vcc
	v_mov_b32_e32 v111, v110
	v_lshl_add_u64 v[108:109], v[108:109], 0, v[110:111]
	v_xor_b32_e32 v104, v108, v110
	v_xor_b32_e32 v102, v109, v110
	v_mad_u64_u32 v[108:109], s[2:3], v104, s20, 0
	v_mul_hi_u32 v48, v104, s19
	v_lshl_add_u64 v[108:109], v[48:49], 0, v[108:109]
	v_mad_u64_u32 v[118:119], s[2:3], v102, s19, 0
	v_add_co_u32_e32 v48, vcc, v108, v118
	v_mad_u64_u32 v[112:113], s[2:3], v102, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v108, vcc, v109, v119, vcc
	v_mov_b32_e32 v109, s18
	s_nop 0
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	v_lshl_add_u64 v[108:109], v[108:109], 0, v[112:113]
	v_mul_lo_u32 v48, s17, v108
	v_mul_lo_u32 v109, s16, v109
	v_mad_u64_u32 v[112:113], s[2:3], s16, v108, 0
	v_add3_u32 v48, v113, v109, v48
	v_sub_u32_e32 v109, v102, v48
	v_sub_co_u32_e32 v104, vcc, v104, v112
	s_nop 1
	v_subb_co_u32_e64 v109, s[2:3], v109, v47, vcc
	v_subrev_co_u32_e64 v111, s[2:3], s16, v104
	v_subb_co_u32_e32 v48, vcc, v102, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v109, s[2:3], 0, v109, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v109
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v112, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v111
	v_cndmask_b32_e64 v102, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v104
	v_cndmask_b32_e64 v111, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v109
	v_cndmask_b32_e64 v104, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v109, v112, v111, s[2:3]
	v_add_u32_e32 v111, 2, v108
	v_add_u32_e32 v112, 1, v108
	v_cmp_ne_u32_e64 s[2:3], 0, v109
	v_cndmask_b32_e32 v48, v102, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v109, v112, v111, s[2:3]
	v_xor_b32_e32 v102, s70, v110
	v_cndmask_b32_e32 v48, v108, v109, vcc
	v_ashrrev_i32_e32 v108, 31, v107
	v_mov_b32_e32 v109, v108
	v_xor_b32_e32 v48, v48, v102
	v_lshl_add_u64 v[106:107], v[106:107], 0, v[108:109]
	v_sub_u32_e32 v48, v48, v102
	v_xor_b32_e32 v109, v106, v108
	v_xad_u32 v102, v48, v108, v46
	v_xor_b32_e32 v104, v107, v108
	v_mad_u64_u32 v[106:107], s[2:3], v109, s20, 0
	v_mul_hi_u32 v48, v109, s19
	v_lshl_add_u64 v[106:107], v[48:49], 0, v[106:107]
	v_mad_u64_u32 v[112:113], s[2:3], v104, s19, 0
	v_add_co_u32_e32 v48, vcc, v106, v112
	v_mad_u64_u32 v[110:111], s[2:3], v104, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v106, vcc, v107, v113, vcc
	v_mov_b32_e32 v107, s18
	s_nop 0
	v_addc_co_u32_e32 v111, vcc, 0, v111, vcc
	v_lshl_add_u64 v[106:107], v[106:107], 0, v[110:111]
	v_mul_lo_u32 v48, s17, v106
	v_mul_lo_u32 v110, s16, v107
	v_mad_u64_u32 v[106:107], s[2:3], s16, v106, 0
	v_add3_u32 v48, v107, v110, v48
	v_sub_u32_e32 v107, v104, v48
	v_sub_co_u32_e32 v106, vcc, v109, v106
	v_mul_lo_u32 v102, v102, s33
	s_nop 0
	v_subb_co_u32_e64 v107, s[2:3], v107, v47, vcc
	v_subrev_co_u32_e64 v109, s[2:3], s16, v106
	v_subb_co_u32_e32 v48, vcc, v104, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v110, s[14:15], 0, v107, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v110
	v_subb_co_u32_e64 v107, s[2:3], v107, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v111, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v109
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v112, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v110
	v_cndmask_b32_e64 v104, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v106
	v_cndmask_b32_e64 v111, v111, v112, s[14:15]
	v_subrev_co_u32_e64 v112, s[2:3], s16, v109
	s_nop 1
	v_subbrev_co_u32_e64 v107, s[2:3], 0, v107, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v111
	s_nop 1
	v_cndmask_b32_e64 v107, v110, v107, s[2:3]
	v_cndmask_b32_e64 v110, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v110, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_cndmask_b32_e64 v104, v109, v112, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v104, v106, v104, vcc
	v_cndmask_b32_e32 v48, v48, v107, vcc
	v_xor_b32_e32 v104, v104, v108
	v_xor_b32_e32 v48, v48, v108
	v_sub_co_u32_e32 v104, vcc, v104, v108
	v_lshl_add_u64 v[106:107], v[52:53], 0, 13
	s_nop 0
	v_subb_co_u32_e32 v48, vcc, v48, v108, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v115, v48, v104, v102
	v_sub_co_u32_e32 v48, vcc, -14, v52
	s_nop 1
	v_subb_co_u32_e32 v102, vcc, -1, v53, vcc
	v_cmp_gt_i32_e32 vcc, 0, v107
	s_nop 1
	v_cndmask_b32_e32 v109, v107, v102, vcc
	v_ashrrev_i32_e32 v110, 31, v109
	v_cndmask_b32_e32 v108, v106, v48, vcc
	v_mov_b32_e32 v111, v110
	v_lshl_add_u64 v[108:109], v[108:109], 0, v[110:111]
	v_xor_b32_e32 v104, v108, v110
	v_xor_b32_e32 v102, v109, v110
	v_mad_u64_u32 v[108:109], s[2:3], v104, s20, 0
	v_mul_hi_u32 v48, v104, s19
	v_lshl_add_u64 v[108:109], v[48:49], 0, v[108:109]
	v_mad_u64_u32 v[118:119], s[2:3], v102, s19, 0
	v_add_co_u32_e32 v48, vcc, v108, v118
	v_mad_u64_u32 v[112:113], s[2:3], v102, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v108, vcc, v109, v119, vcc
	v_mov_b32_e32 v109, s18
	s_nop 0
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	v_lshl_add_u64 v[108:109], v[108:109], 0, v[112:113]
	v_mul_lo_u32 v48, s17, v108
	v_mul_lo_u32 v109, s16, v109
	v_mad_u64_u32 v[112:113], s[2:3], s16, v108, 0
	v_add3_u32 v48, v113, v109, v48
	v_sub_u32_e32 v109, v102, v48
	v_sub_co_u32_e32 v104, vcc, v104, v112
	s_nop 1
	v_subb_co_u32_e64 v109, s[2:3], v109, v47, vcc
	v_subrev_co_u32_e64 v111, s[2:3], s16, v104
	v_subb_co_u32_e32 v48, vcc, v102, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v109, s[2:3], 0, v109, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v109
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v112, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v111
	v_cndmask_b32_e64 v102, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v104
	v_cndmask_b32_e64 v111, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v109
	v_cndmask_b32_e64 v104, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v109, v112, v111, s[2:3]
	v_add_u32_e32 v111, 2, v108
	v_add_u32_e32 v112, 1, v108
	v_cmp_ne_u32_e64 s[2:3], 0, v109
	v_cndmask_b32_e32 v48, v102, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v109, v112, v111, s[2:3]
	v_xor_b32_e32 v102, s70, v110
	v_cndmask_b32_e32 v48, v108, v109, vcc
	v_ashrrev_i32_e32 v108, 31, v107
	v_mov_b32_e32 v109, v108
	v_xor_b32_e32 v48, v48, v102
	v_lshl_add_u64 v[106:107], v[106:107], 0, v[108:109]
	v_sub_u32_e32 v48, v48, v102
	v_xor_b32_e32 v109, v106, v108
	v_xad_u32 v102, v48, v108, v46
	v_xor_b32_e32 v104, v107, v108
	v_mad_u64_u32 v[106:107], s[2:3], v109, s20, 0
	v_mul_hi_u32 v48, v109, s19
	v_lshl_add_u64 v[106:107], v[48:49], 0, v[106:107]
	v_mad_u64_u32 v[112:113], s[2:3], v104, s19, 0
	v_add_co_u32_e32 v48, vcc, v106, v112
	v_mad_u64_u32 v[110:111], s[2:3], v104, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v106, vcc, v107, v113, vcc
	v_mov_b32_e32 v107, s18
	s_nop 0
	v_addc_co_u32_e32 v111, vcc, 0, v111, vcc
	v_lshl_add_u64 v[106:107], v[106:107], 0, v[110:111]
	v_mul_lo_u32 v48, s17, v106
	v_mul_lo_u32 v110, s16, v107
	v_mad_u64_u32 v[106:107], s[2:3], s16, v106, 0
	v_add3_u32 v48, v107, v110, v48
	v_sub_u32_e32 v107, v104, v48
	v_sub_co_u32_e32 v106, vcc, v109, v106
	v_mul_lo_u32 v102, v102, s33
	s_nop 0
	v_subb_co_u32_e64 v107, s[2:3], v107, v47, vcc
	v_subrev_co_u32_e64 v109, s[2:3], s16, v106
	v_subb_co_u32_e32 v48, vcc, v104, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v110, s[14:15], 0, v107, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v110
	v_subb_co_u32_e64 v107, s[2:3], v107, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v111, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v109
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v112, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v110
	v_cndmask_b32_e64 v104, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v106
	v_cndmask_b32_e64 v111, v111, v112, s[14:15]
	v_subrev_co_u32_e64 v112, s[2:3], s16, v109
	s_nop 1
	v_subbrev_co_u32_e64 v107, s[2:3], 0, v107, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v111
	s_nop 1
	v_cndmask_b32_e64 v107, v110, v107, s[2:3]
	v_cndmask_b32_e64 v110, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v104, v104, v110, vcc
	v_cmp_ne_u32_e32 vcc, 0, v104
	v_cndmask_b32_e64 v104, v109, v112, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v104, v106, v104, vcc
	v_cndmask_b32_e32 v48, v48, v107, vcc
	v_xor_b32_e32 v104, v104, v108
	v_xor_b32_e32 v48, v48, v108
	v_sub_co_u32_e32 v104, vcc, v104, v108
	v_lshl_add_u64 v[106:107], v[50:51], 0, 14
	s_nop 0
	v_subb_co_u32_e32 v48, vcc, v48, v108, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_add3_u32 v118, v48, v104, v102
	v_sub_co_u32_e32 v48, vcc, -15, v50
	s_nop 1
	v_subb_co_u32_e32 v50, vcc, -1, v51, vcc
	v_cmp_gt_i32_e32 vcc, 0, v107
	s_nop 1
	v_cndmask_b32_e32 v51, v107, v50, vcc
	v_ashrrev_i32_e32 v108, 31, v51
	v_cndmask_b32_e32 v50, v106, v48, vcc
	v_mov_b32_e32 v109, v108
	v_lshl_add_u64 v[50:51], v[50:51], 0, v[108:109]
	v_xor_b32_e32 v104, v50, v108
	v_xor_b32_e32 v102, v51, v108
	v_mad_u64_u32 v[50:51], s[2:3], v104, s20, 0
	v_mul_hi_u32 v48, v104, s19
	v_lshl_add_u64 v[50:51], v[48:49], 0, v[50:51]
	v_mad_u64_u32 v[112:113], s[2:3], v102, s19, 0
	v_add_co_u32_e32 v48, vcc, v50, v112
	v_mad_u64_u32 v[110:111], s[2:3], v102, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v50, vcc, v51, v113, vcc
	v_mov_b32_e32 v51, s18
	s_nop 0
	v_addc_co_u32_e32 v111, vcc, 0, v111, vcc
	v_lshl_add_u64 v[50:51], v[50:51], 0, v[110:111]
	v_mul_lo_u32 v48, s17, v50
	v_mul_lo_u32 v51, s16, v51
	v_mad_u64_u32 v[110:111], s[2:3], s16, v50, 0
	v_add3_u32 v48, v111, v51, v48
	v_sub_u32_e32 v51, v102, v48
	v_sub_co_u32_e32 v104, vcc, v104, v110
	s_nop 1
	v_subb_co_u32_e64 v51, s[2:3], v51, v47, vcc
	v_subrev_co_u32_e64 v109, s[2:3], s16, v104
	v_subb_co_u32_e32 v48, vcc, v102, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v51, s[2:3], 0, v51, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v51
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v110, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v109
	v_cndmask_b32_e64 v102, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v104
	v_cndmask_b32_e64 v109, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v51
	v_cndmask_b32_e64 v104, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v51, v110, v109, s[2:3]
	v_add_u32_e32 v109, 2, v50
	v_add_u32_e32 v110, 1, v50
	v_cmp_ne_u32_e64 s[2:3], 0, v51
	v_cndmask_b32_e32 v48, v102, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v51, v110, v109, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v48, v50, v51, vcc
	v_xor_b32_e32 v50, s70, v108
	v_xor_b32_e32 v48, v48, v50
	v_sub_u32_e32 v48, v48, v50
	v_ashrrev_i32_e32 v50, 31, v107
	v_mov_b32_e32 v51, v50
	v_lshl_add_u64 v[106:107], v[106:107], 0, v[50:51]
	v_xor_b32_e32 v104, v106, v50
	v_xad_u32 v102, v48, v50, v46
	v_xor_b32_e32 v51, v107, v50
	v_mad_u64_u32 v[106:107], s[2:3], v104, s20, 0
	v_mul_hi_u32 v48, v104, s19
	v_lshl_add_u64 v[106:107], v[48:49], 0, v[106:107]
	v_mad_u64_u32 v[110:111], s[2:3], v51, s19, 0
	v_add_co_u32_e32 v48, vcc, v106, v110
	v_mad_u64_u32 v[108:109], s[2:3], v51, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v106, vcc, v107, v111, vcc
	v_mov_b32_e32 v107, s18
	s_nop 0
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[106:107], v[106:107], 0, v[108:109]
	v_mul_lo_u32 v48, s17, v106
	v_mul_lo_u32 v108, s16, v107
	v_mad_u64_u32 v[106:107], s[2:3], s16, v106, 0
	v_add3_u32 v48, v107, v108, v48
	v_sub_u32_e32 v107, v51, v48
	v_sub_co_u32_e32 v104, vcc, v104, v106
	s_nop 1
	v_subb_co_u32_e64 v106, s[2:3], v107, v47, vcc
	v_subrev_co_u32_e64 v107, s[2:3], s16, v104
	v_subb_co_u32_e32 v48, vcc, v51, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v108, s[14:15], 0, v106, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v108
	v_subb_co_u32_e64 v106, s[2:3], v106, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v109, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v107
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v110, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v108
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v104
	v_cndmask_b32_e64 v109, v109, v110, s[14:15]
	v_subrev_co_u32_e64 v110, s[2:3], s16, v107
	s_nop 1
	v_subbrev_co_u32_e64 v106, s[2:3], 0, v106, s[2:3]
	v_cmp_ne_u32_e64 s[2:3], 0, v109
	s_nop 1
	v_cndmask_b32_e64 v106, v108, v106, s[2:3]
	v_cndmask_b32_e64 v108, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v51, v51, v108, vcc
	v_cmp_ne_u32_e32 vcc, 0, v51
	v_cndmask_b32_e64 v51, v107, v110, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v51, v104, v51, vcc
	v_cndmask_b32_e32 v48, v48, v106, vcc
	v_xor_b32_e32 v51, v51, v50
	v_xor_b32_e32 v48, v48, v50
	v_sub_co_u32_e32 v51, vcc, v51, v50
	s_nop 1
	v_subb_co_u32_e32 v48, vcc, v48, v50, vcc
	v_ashrrev_i32_e32 v48, 31, v48
	v_and_b32_e32 v48, s44, v48
	v_mul_lo_u32 v50, v102, s33
	v_add3_u32 v112, v48, v51, v50
	v_sub_co_u32_e32 v48, vcc, -16, v52
	v_lshl_add_u64 v[50:51], v[52:53], 0, 15
	s_nop 0
	v_subb_co_u32_e32 v52, vcc, -1, v53, vcc
	v_cmp_gt_i32_e32 vcc, 0, v51
	s_nop 1
	v_cndmask_b32_e32 v53, v51, v52, vcc
	v_ashrrev_i32_e32 v106, 31, v53
	v_cndmask_b32_e32 v52, v50, v48, vcc
	v_mov_b32_e32 v107, v106
	v_lshl_add_u64 v[52:53], v[52:53], 0, v[106:107]
	v_xor_b32_e32 v104, v52, v106
	v_xor_b32_e32 v102, v53, v106
	v_mad_u64_u32 v[52:53], s[2:3], v104, s20, 0
	v_mul_hi_u32 v48, v104, s19
	v_lshl_add_u64 v[52:53], v[48:49], 0, v[52:53]
	v_mad_u64_u32 v[110:111], s[2:3], v102, s19, 0
	v_add_co_u32_e32 v48, vcc, v52, v110
	v_mad_u64_u32 v[108:109], s[2:3], v102, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v52, vcc, v53, v111, vcc
	v_mov_b32_e32 v53, s18
	s_nop 0
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[52:53], v[52:53], 0, v[108:109]
	v_mul_lo_u32 v48, s17, v52
	v_mul_lo_u32 v53, s16, v53
	v_mad_u64_u32 v[108:109], s[2:3], s16, v52, 0
	v_add3_u32 v48, v109, v53, v48
	v_sub_u32_e32 v53, v102, v48
	v_sub_co_u32_e32 v104, vcc, v104, v108
	s_nop 1
	v_subb_co_u32_e64 v53, s[2:3], v53, v47, vcc
	v_subrev_co_u32_e64 v107, s[2:3], s16, v104
	v_subb_co_u32_e32 v48, vcc, v102, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v53, s[2:3], 0, v53, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s17, v53
	v_cmp_le_u32_e32 vcc, s17, v48
	s_nop 0
	v_cndmask_b32_e64 v108, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s16, v107
	v_cndmask_b32_e64 v102, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v104
	v_cndmask_b32_e64 v107, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s17, v53
	v_cndmask_b32_e64 v104, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	v_cndmask_b32_e64 v53, v108, v107, s[2:3]
	v_add_u32_e32 v107, 2, v52
	v_add_u32_e32 v108, 1, v52
	v_cmp_ne_u32_e64 s[2:3], 0, v53
	v_cndmask_b32_e32 v48, v102, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v48
	v_cndmask_b32_e64 v53, v108, v107, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v48, v52, v53, vcc
	v_xor_b32_e32 v52, s70, v106
	v_xor_b32_e32 v48, v48, v52
	v_sub_u32_e32 v48, v48, v52
	v_ashrrev_i32_e32 v52, 31, v51
	v_mov_b32_e32 v53, v52
	v_lshl_add_u64 v[50:51], v[50:51], 0, v[52:53]
	v_xor_b32_e32 v102, v50, v52
	v_xad_u32 v46, v48, v52, v46
	v_xor_b32_e32 v53, v51, v52
	v_mad_u64_u32 v[50:51], s[2:3], v102, s20, 0
	v_mul_hi_u32 v48, v102, s19
	v_lshl_add_u64 v[50:51], v[48:49], 0, v[50:51]
	v_mad_u64_u32 v[108:109], s[2:3], v53, s19, 0
	v_add_co_u32_e32 v48, vcc, v50, v108
	v_mad_u64_u32 v[106:107], s[2:3], v53, s20, 0
	s_nop 0
	v_addc_co_u32_e32 v50, vcc, v51, v109, vcc
	v_mov_b32_e32 v51, s18
	s_nop 0
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[50:51], v[50:51], 0, v[106:107]
	v_mul_lo_u32 v48, s17, v50
	v_mul_lo_u32 v104, s16, v51
	v_mad_u64_u32 v[50:51], s[2:3], s16, v50, 0
	v_add3_u32 v48, v51, v104, v48
	v_sub_u32_e32 v51, v53, v48
	v_sub_co_u32_e32 v50, vcc, v102, v50
	v_mul_lo_u32 v46, v46, s33
	s_nop 0
	v_subb_co_u32_e64 v51, s[2:3], v51, v47, vcc
	v_subrev_co_u32_e64 v102, s[2:3], s16, v50
	v_subb_co_u32_e32 v48, vcc, v53, v48, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v104, s[14:15], 0, v51, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v104
	v_subb_co_u32_e64 v47, s[2:3], v51, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v106, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v102
	v_subrev_co_u32_e64 v51, s[2:3], s16, v102
	s_nop 0
	v_cndmask_b32_e64 v107, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v104
	v_subbrev_co_u32_e64 v47, s[2:3], 0, v47, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v106, v106, v107, s[14:15]
	v_cmp_le_u32_e32 vcc, s17, v48
	v_cmp_ne_u32_e64 s[2:3], 0, v106
	s_nop 0
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v50
	v_cndmask_b32_e64 v47, v104, v47, s[2:3]
	s_mul_i32 s16, s0, 0x600
	v_cndmask_b32_e64 v104, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v48
	s_nop 1
	v_cndmask_b32_e32 v53, v53, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	s_nop 1
	v_cndmask_b32_e32 v47, v48, v47, vcc
	v_cndmask_b32_e64 v48, v102, v51, s[2:3]
	v_cndmask_b32_e32 v48, v50, v48, vcc
	v_xor_b32_e32 v48, v48, v52
	v_xor_b32_e32 v47, v47, v52
	v_sub_co_u32_e32 v48, vcc, v48, v52
	s_mul_hi_u32 s2, s0, 0x600
	s_nop 0
	v_subb_co_u32_e32 v47, vcc, v47, v52, vcc
	v_ashrrev_i32_e32 v47, 31, v47
	v_and_b32_e32 v47, s44, v47
	v_add3_u32 v46, v47, v48, v46
	buffer_load_ubyte v102, v100, s[36:39], 0 offen
	buffer_load_ubyte v104, v101, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v103, v103, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v105, v105, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v106, v115, s[36:39], 0 offen
	buffer_load_ubyte v108, v118, s[36:39], 0 offen
	buffer_load_ubyte v107, v112, s[36:39], 0 offen
	buffer_load_ubyte v109, v46, s[36:39], 0 offen
	ds_read_b32 v101, v96 offset:49152
	ds_read_b32 v100, v98 offset:49152
	ds_read_b128 v[136:139], v80
	ds_read_b128 v[140:143], v82
	ds_read_b128 v[144:147], v84
	ds_read_b128 v[174:177], v86
	ds_read_b128 v[178:181], v88
	ds_read_b128 v[132:135], v90
	ds_read_b128 v[118:121], v92
	ds_read_b128 v[110:113], v94
	s_add_i32 s17, s2, s1
	v_mov_b32_e32 v47, s17
	v_or_b32_e32 v46, s16, v66
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[72:73]
	s_movk_i32 s0, 0xff00
	v_lshl_add_u64 v[50:51], v[46:47], 0, v[68:69]
	s_mov_b32 s1, -1
	v_lshl_add_u64 v[46:47], v[50:51], 0, s[0:1]
	s_movk_i32 s0, 0xff
	v_sub_co_u32_e32 v48, vcc, s0, v50
	s_nop 1
	v_subb_co_u32_e32 v50, vcc, 0, v51, vcc
	v_cmp_gt_i32_e32 vcc, 0, v47
	s_nop 1
	v_cndmask_b32_e32 v53, v47, v50, vcc
	v_cndmask_b32_e32 v52, v46, v48, vcc
	v_or_b32_e32 v48, s77, v53
	v_cmp_ne_u32_e32 vcc, 0, v48
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	s_cbranch_execz .LBB0_91
	s_add_u32 s2, s76, s78
	s_mov_b32 s79, s78
	s_addc_u32 s3, s77, s78
	s_xor_b64 s[14:15], s[2:3], s[78:79]
	v_cvt_f32_u32_e32 v48, s14
	v_cvt_f32_u32_e32 v50, s15
	s_sub_u32 s2, 0, s14
	s_subb_u32 s3, 0, s15
	v_fmac_f32_e32 v48, 0x4f800000, v50
	v_rcp_f32_e32 v48, v48
	s_nop 0
	v_mul_f32_e32 v48, 0x5f7ffffc, v48
	v_mul_f32_e32 v50, 0x2f800000, v48
	v_trunc_f32_e32 v50, v50
	v_fmac_f32_e32 v48, 0xcf800000, v50
	v_cvt_u32_f32_e32 v50, v50
	v_cvt_u32_f32_e32 v48, v48
	v_readfirstlane_b32 s19, v50
	v_readfirstlane_b32 s20, v48
	s_mul_i32 s21, s2, s19
	s_mul_hi_u32 s23, s2, s20
	s_mul_i32 s22, s3, s20
	s_add_i32 s21, s23, s21
	s_add_i32 s21, s21, s22
	s_mul_i32 s24, s2, s20
	s_mul_i32 s23, s20, s21
	s_mul_hi_u32 s25, s20, s24
	s_mul_hi_u32 s22, s20, s21
	s_add_u32 s23, s25, s23
	s_addc_u32 s22, 0, s22
	s_mul_hi_u32 s26, s19, s24
	s_mul_i32 s24, s19, s24
	s_add_u32 s23, s23, s24
	s_mul_hi_u32 s25, s19, s21
	s_addc_u32 s22, s22, s26
	s_addc_u32 s23, s25, 0
	s_mul_i32 s21, s19, s21
	s_add_u32 s21, s22, s21
	s_addc_u32 s22, 0, s23
	s_add_u32 s20, s20, s21
	s_addc_u32 s19, s19, s22
	s_mul_i32 s21, s2, s19
	s_mul_hi_u32 s22, s2, s20
	s_add_i32 s21, s22, s21
	s_mul_i32 s3, s3, s20
	s_add_i32 s21, s21, s3
	s_mul_i32 s2, s2, s20
	s_mul_hi_u32 s22, s19, s2
	s_mul_i32 s23, s19, s2
	s_mul_i32 s25, s20, s21
	s_mul_hi_u32 s2, s20, s2
	s_mul_hi_u32 s24, s20, s21
	s_add_u32 s2, s2, s25
	s_addc_u32 s24, 0, s24
	s_add_u32 s2, s2, s23
	s_mul_hi_u32 s3, s19, s21
	s_addc_u32 s2, s24, s22
	s_addc_u32 s3, s3, 0
	s_mul_i32 s21, s19, s21
	s_add_u32 s2, s2, s21
	v_ashrrev_i32_e32 v50, 31, v53
	s_addc_u32 s3, 0, s3
	v_mov_b32_e32 v51, v50
	s_add_u32 s20, s20, s2
	v_lshl_add_u64 v[52:53], v[52:53], 0, v[50:51]
	s_addc_u32 s19, s19, s3
	v_xor_b32_e32 v115, v52, v50
	v_xor_b32_e32 v51, v53, v50
	v_mad_u64_u32 v[52:53], s[2:3], v115, s19, 0
	v_mul_hi_u32 v48, v115, s20
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[52:53]
	v_mad_u64_u32 v[72:73], s[2:3], v51, s20, 0
	v_add_co_u32_e32 v48, vcc, v48, v72
	v_mad_u64_u32 v[52:53], s[2:3], v51, s19, 0
	s_nop 0
	v_addc_co_u32_e32 v48, vcc, v49, v73, vcc
	v_mov_b32_e32 v49, s18
	s_nop 0
	v_addc_co_u32_e32 v53, vcc, 0, v53, vcc
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[52:53]
	v_mul_lo_u32 v72, s15, v48
	v_mul_lo_u32 v73, s14, v49
	v_mad_u64_u32 v[52:53], s[2:3], s14, v48, 0
	v_add3_u32 v131, v53, v73, v72
	v_sub_u32_e32 v53, v51, v131
	v_mov_b32_e32 v72, s15
	v_sub_co_u32_e32 v115, vcc, v115, v52
	s_nop 1
	v_subb_co_u32_e64 v52, s[2:3], v53, v72, vcc
	v_subrev_co_u32_e64 v53, s[2:3], s14, v115
	s_nop 1
	v_subbrev_co_u32_e64 v52, s[2:3], 0, v52, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s15, v52
	s_nop 1
	v_cndmask_b32_e64 v72, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s14, v53
	s_nop 1
	v_cndmask_b32_e64 v53, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s15, v52
	s_nop 1
	v_cndmask_b32_e64 v148, v72, v53, s[2:3]
	v_lshl_add_u64 v[52:53], v[48:49], 0, 2
	v_lshl_add_u64 v[72:73], v[48:49], 0, 1
	v_subb_co_u32_e32 v49, vcc, v51, v131, vcc
	v_cmp_le_u32_e32 vcc, s15, v49
	s_nop 1
	v_cndmask_b32_e64 v51, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s14, v115
	s_nop 1
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s15, v49
	s_nop 1
	v_cndmask_b32_e32 v49, v51, v53, vcc
	v_cmp_ne_u32_e32 vcc, 0, v148
	s_nop 1
	v_cndmask_b32_e32 v51, v72, v52, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	v_xor_b32_e32 v49, s78, v50
	s_nop 0
	v_cndmask_b32_e32 v48, v48, v51, vcc
	v_xor_b32_e32 v48, v48, v49
	v_sub_co_u32_e32 v50, vcc, v48, v49
.LBB0_91:
	s_or_saveexec_b64 s[0:1], s[0:1]
	v_readlane_b32 s70, v202, 31
	v_readlane_b32 s26, v202, 28
	v_readlane_b32 s67, v202, 24
	v_readlane_b32 s71, v202, 32
	v_readlane_b32 s27, v202, 29
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execz .LBB0_93
	v_rcp_iflag_f32_e32 v48, v59
	s_sub_i32 s2, 0, s76
	v_mul_f32_e32 v48, 0x4f7ffffe, v48
	v_cvt_u32_f32_e32 v48, v48
	v_mul_lo_u32 v49, s2, v48
	v_mul_hi_u32 v49, v48, v49
	v_add_u32_e32 v48, v48, v49
	v_mul_hi_u32 v48, v52, v48
	v_mul_lo_u32 v49, v48, s76
	v_sub_u32_e32 v49, v52, v49
	v_add_u32_e32 v50, 1, v48
	v_subrev_u32_e32 v51, s76, v49
	v_cmp_le_u32_e32 vcc, s76, v49
	s_nop 1
	v_cndmask_b32_e32 v49, v49, v51, vcc
	v_cndmask_b32_e32 v48, v48, v50, vcc
	v_add_u32_e32 v50, 1, v48
	v_cmp_le_u32_e32 vcc, s76, v49
	s_nop 1
	v_cndmask_b32_e32 v50, v48, v50, vcc
.LBB0_93:
	s_or_b64 exec, exec, s[0:1]
	v_mov_b32_e32 v49, s17
	v_or_b32_e32 v48, s16, v62
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[66:67]
	s_movk_i32 s0, 0xff00
	v_lshl_add_u64 v[48:49], v[68:69], 0, v[48:49]
	s_mov_b32 s1, -1
	v_lshl_add_u64 v[48:49], v[48:49], 0, s[0:1]
	v_or_b32_e32 v51, s77, v49
	s_mov_b32 s2, 0
	v_cmp_ne_u32_e32 vcc, 0, v51
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	s_cbranch_execz .LBB0_95
	s_add_u32 s14, s76, s78
	s_mov_b32 s79, s78
	s_addc_u32 s15, s77, s78
	s_xor_b64 s[16:17], s[14:15], s[78:79]
	v_cvt_f32_u32_e32 v51, s16
	v_cvt_f32_u32_e32 v52, s17
	s_sub_u32 s3, 0, s16
	s_subb_u32 s14, 0, s17
	v_ashrrev_i32_e32 v66, 31, v49
	v_fmamk_f32 v51, v52, 0x4f800000, v51
	v_rcp_f32_e32 v51, v51
	v_mov_b32_e32 v67, v66
	v_mov_b32_e32 v69, 0
	v_mul_f32_e32 v51, 0x5f7ffffc, v51
	v_mul_f32_e32 v52, 0x2f800000, v51
	v_trunc_f32_e32 v52, v52
	v_fmamk_f32 v51, v52, 0xcf800000, v51
	v_cvt_u32_f32_e32 v52, v52
	v_cvt_u32_f32_e32 v51, v51
	v_readfirstlane_b32 s15, v52
	v_readfirstlane_b32 s18, v51
	s_mul_i32 s19, s3, s15
	s_mul_hi_u32 s21, s3, s18
	s_mul_i32 s20, s14, s18
	s_add_i32 s19, s21, s19
	s_add_i32 s19, s19, s20
	s_mul_i32 s22, s3, s18
	s_mul_i32 s21, s18, s19
	s_mul_hi_u32 s23, s18, s22
	s_mul_hi_u32 s20, s18, s19
	s_add_u32 s21, s23, s21
	s_addc_u32 s20, 0, s20
	s_mul_hi_u32 s24, s15, s22
	s_mul_i32 s22, s15, s22
	s_add_u32 s21, s21, s22
	s_mul_hi_u32 s23, s15, s19
	s_addc_u32 s20, s20, s24
	s_addc_u32 s21, s23, 0
	s_mul_i32 s19, s15, s19
	s_add_u32 s19, s20, s19
	s_addc_u32 s20, 0, s21
	s_add_u32 s18, s18, s19
	s_addc_u32 s15, s15, s20
	s_mul_i32 s19, s3, s15
	s_mul_hi_u32 s20, s3, s18
	s_add_i32 s19, s20, s19
	s_mul_i32 s14, s14, s18
	s_add_i32 s19, s19, s14
	s_mul_i32 s3, s3, s18
	s_mul_hi_u32 s20, s15, s3
	s_mul_i32 s21, s15, s3
	s_mul_i32 s23, s18, s19
	s_mul_hi_u32 s3, s18, s3
	s_mul_hi_u32 s22, s18, s19
	s_add_u32 s3, s3, s23
	s_addc_u32 s22, 0, s22
	s_add_u32 s3, s3, s21
	s_mul_hi_u32 s14, s15, s19
	s_addc_u32 s3, s22, s20
	s_addc_u32 s14, s14, 0
	s_mul_i32 s19, s15, s19
	s_add_u32 s3, s3, s19
	s_addc_u32 s14, 0, s14
	s_add_u32 s3, s18, s3
	v_lshl_add_u64 v[52:53], v[48:49], 0, v[66:67]
	s_addc_u32 s18, s15, s14
	v_xor_b32_e32 v62, v52, v66
	v_xor_b32_e32 v51, v53, v66
	v_mad_u64_u32 v[52:53], s[14:15], v62, s18, 0
	v_mul_hi_u32 v68, v62, s3
	v_lshl_add_u64 v[52:53], v[68:69], 0, v[52:53]
	v_mad_u64_u32 v[72:73], s[14:15], v51, s3, 0
	v_add_co_u32_e32 v52, vcc, v52, v72
	v_mad_u64_u32 v[68:69], s[14:15], v51, s18, 0
	s_nop 0
	v_addc_co_u32_e32 v52, vcc, v53, v73, vcc
	v_mov_b32_e32 v53, s2
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[52:53], v[52:53], 0, v[68:69]
	v_mul_lo_u32 v67, s17, v52
	v_mul_lo_u32 v68, s16, v53
	v_mad_u64_u32 v[52:53], s[2:3], s16, v52, 0
	v_add3_u32 v53, v53, v68, v67
	v_sub_u32_e32 v67, v51, v53
	v_mov_b32_e32 v68, s17
	v_sub_co_u32_e32 v52, vcc, v62, v52
	s_nop 1
	v_subb_co_u32_e64 v62, s[2:3], v67, v68, vcc
	v_subrev_co_u32_e64 v67, s[2:3], s16, v52
	v_subb_co_u32_e32 v51, vcc, v51, v53, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v62, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v69
	v_subb_co_u32_e64 v62, s[2:3], v62, v68, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v72, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v67
	v_subrev_co_u32_e64 v68, s[2:3], s16, v67
	s_nop 0
	v_cndmask_b32_e64 v73, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v69
	v_subbrev_co_u32_e64 v62, s[2:3], 0, v62, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v72, v72, v73, s[14:15]
	v_cmp_le_u32_e32 vcc, s17, v51
	v_cmp_ne_u32_e64 s[2:3], 0, v72
	s_nop 0
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v52
	v_cndmask_b32_e64 v62, v69, v62, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v51
	s_nop 1
	v_cndmask_b32_e32 v53, v53, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v53
	v_cndmask_b32_e64 v53, v67, v68, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v52, v52, v53, vcc
	v_cndmask_b32_e32 v51, v51, v62, vcc
	v_xor_b32_e32 v52, v52, v66
	v_xor_b32_e32 v51, v51, v66
	v_sub_co_u32_e32 v52, vcc, v52, v66
	s_nop 1
	v_subb_co_u32_e32 v53, vcc, v51, v66, vcc
.LBB0_95:
	s_andn2_saveexec_b64 s[0:1], s[0:1]
	s_cbranch_execz .LBB0_97
	v_rcp_iflag_f32_e32 v51, v59
	s_sub_i32 s2, 0, s76
	v_mov_b32_e32 v53, 0
	v_mul_f32_e32 v51, 0x4f7ffffe, v51
	v_cvt_u32_f32_e32 v51, v51
	v_mul_lo_u32 v52, s2, v51
	v_mul_hi_u32 v52, v51, v52
	v_add_u32_e32 v51, v51, v52
	v_mul_hi_u32 v51, v48, v51
	v_mul_lo_u32 v51, v51, s76
	v_sub_u32_e32 v51, v48, v51
	v_subrev_u32_e32 v52, s76, v51
	v_cmp_le_u32_e32 vcc, s76, v51
	s_nop 1
	v_cndmask_b32_e32 v51, v51, v52, vcc
	v_subrev_u32_e32 v52, s76, v51
	v_cmp_le_u32_e32 vcc, s76, v51
	s_nop 1
	v_cndmask_b32_e32 v52, v51, v52, vcc
.LBB0_97:
	s_or_b64 exec, exec, s[0:1]
	v_ashrrev_i32_e32 v51, 31, v47
	v_xor_b32_e32 v50, v50, v51
	v_ashrrev_i32_e32 v51, 31, v53
	v_mul_lo_u32 v50, v50, s54
	v_and_b32_e32 v51, s76, v51
	v_add3_u32 v50, v50, v52, v51
	buffer_load_ubyte v62, v50, s[8:11], 0 offen
	v_lshl_add_u64 v[50:51], v[46:47], 0, 2
	v_sub_co_u32_e32 v46, vcc, -3, v46
	s_mov_b32 s2, 0
	s_nop 0
	v_subb_co_u32_e32 v47, vcc, -1, v47, vcc
	v_cmp_gt_i32_e32 vcc, 0, v51
	s_nop 1
	v_cndmask_b32_e32 v53, v51, v47, vcc
	v_cndmask_b32_e32 v52, v50, v46, vcc
	v_or_b32_e32 v46, s77, v53
	v_cmp_ne_u32_e32 vcc, 0, v46
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	s_cbranch_execz .LBB0_99
	s_add_u32 s14, s76, s78
	s_mov_b32 s79, s78
	s_addc_u32 s15, s77, s78
	s_xor_b64 s[14:15], s[14:15], s[78:79]
	v_cvt_f32_u32_e32 v46, s14
	v_cvt_f32_u32_e32 v47, s15
	s_sub_u32 s3, 0, s14
	s_subb_u32 s16, 0, s15
	v_mov_b32_e32 v67, 0
	v_fmamk_f32 v46, v47, 0x4f800000, v46
	v_rcp_f32_e32 v46, v46
	s_nop 0
	v_mul_f32_e32 v46, 0x5f7ffffc, v46
	v_mul_f32_e32 v47, 0x2f800000, v46
	v_trunc_f32_e32 v47, v47
	v_fmamk_f32 v46, v47, 0xcf800000, v46
	v_cvt_u32_f32_e32 v47, v47
	v_cvt_u32_f32_e32 v46, v46
	v_readfirstlane_b32 s17, v47
	v_readfirstlane_b32 s18, v46
	s_mul_i32 s19, s3, s17
	s_mul_hi_u32 s21, s3, s18
	s_mul_i32 s20, s16, s18
	s_add_i32 s19, s21, s19
	s_add_i32 s19, s19, s20
	s_mul_i32 s22, s3, s18
	s_mul_i32 s21, s18, s19
	s_mul_hi_u32 s23, s18, s22
	s_mul_hi_u32 s20, s18, s19
	s_add_u32 s21, s23, s21
	s_addc_u32 s20, 0, s20
	s_mul_hi_u32 s24, s17, s22
	s_mul_i32 s22, s17, s22
	s_add_u32 s21, s21, s22
	s_mul_hi_u32 s23, s17, s19
	s_addc_u32 s20, s20, s24
	s_addc_u32 s21, s23, 0
	s_mul_i32 s19, s17, s19
	s_add_u32 s19, s20, s19
	s_addc_u32 s20, 0, s21
	s_add_u32 s18, s18, s19
	s_addc_u32 s17, s17, s20
	s_mul_i32 s19, s3, s17
	s_mul_hi_u32 s20, s3, s18
	s_add_i32 s19, s20, s19
	s_mul_i32 s16, s16, s18
	s_add_i32 s19, s19, s16
	s_mul_i32 s3, s3, s18
	s_mul_hi_u32 s20, s17, s3
	s_mul_i32 s21, s17, s3
	s_mul_i32 s23, s18, s19
	s_mul_hi_u32 s3, s18, s3
	s_mul_hi_u32 s22, s18, s19
	s_add_u32 s3, s3, s23
	s_addc_u32 s22, 0, s22
	s_add_u32 s3, s3, s21
	s_mul_hi_u32 s16, s17, s19
	s_addc_u32 s3, s22, s20
	s_addc_u32 s16, s16, 0
	s_mul_i32 s19, s17, s19
	s_add_u32 s3, s3, s19
	v_ashrrev_i32_e32 v46, 31, v53
	s_addc_u32 s16, 0, s16
	v_mov_b32_e32 v47, v46
	s_add_u32 s3, s18, s3
	v_lshl_add_u64 v[52:53], v[52:53], 0, v[46:47]
	s_addc_u32 s18, s17, s16
	v_xor_b32_e32 v50, v52, v46
	v_xor_b32_e32 v47, v53, v46
	v_mad_u64_u32 v[52:53], s[16:17], v50, s18, 0
	v_mul_hi_u32 v66, v50, s3
	v_lshl_add_u64 v[52:53], v[66:67], 0, v[52:53]
	v_mad_u64_u32 v[68:69], s[16:17], v47, s3, 0
	v_add_co_u32_e32 v52, vcc, v52, v68
	v_mad_u64_u32 v[66:67], s[16:17], v47, s18, 0
	s_nop 0
	v_addc_co_u32_e32 v52, vcc, v53, v69, vcc
	v_mov_b32_e32 v53, s2
	s_nop 0
	v_addc_co_u32_e32 v67, vcc, 0, v67, vcc
	v_lshl_add_u64 v[52:53], v[52:53], 0, v[66:67]
	v_mul_lo_u32 v68, s15, v52
	v_mul_lo_u32 v69, s14, v53
	v_mad_u64_u32 v[66:67], s[2:3], s14, v52, 0
	v_add3_u32 v72, v67, v69, v68
	v_sub_u32_e32 v67, v47, v72
	v_mov_b32_e32 v68, s15
	v_sub_co_u32_e32 v50, vcc, v50, v66
	v_xor_b32_e32 v46, s78, v46
	s_nop 0
	v_subb_co_u32_e64 v66, s[2:3], v67, v68, vcc
	v_subrev_co_u32_e64 v67, s[2:3], s14, v50
	v_subb_co_u32_e32 v47, vcc, v47, v72, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v66, s[2:3], 0, v66, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s15, v66
	v_cmp_le_u32_e32 vcc, s15, v47
	s_nop 0
	v_cndmask_b32_e64 v68, 0, -1, s[2:3]
	v_cmp_le_u32_e64 s[2:3], s14, v67
	s_nop 1
	v_cndmask_b32_e64 v67, 0, -1, s[2:3]
	v_cmp_eq_u32_e64 s[2:3], s15, v66
	s_nop 1
	v_cndmask_b32_e64 v73, v68, v67, s[2:3]
	v_lshl_add_u64 v[66:67], v[52:53], 0, 2
	v_lshl_add_u64 v[68:69], v[52:53], 0, 1
	v_cndmask_b32_e64 v53, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s14, v50
	s_nop 1
	v_cndmask_b32_e64 v50, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s15, v47
	s_nop 1
	v_cndmask_b32_e32 v47, v53, v50, vcc
	v_cmp_ne_u32_e32 vcc, 0, v73
	s_nop 1
	v_cndmask_b32_e32 v50, v68, v66, vcc
	v_cmp_ne_u32_e32 vcc, 0, v47
	s_nop 1
	v_cndmask_b32_e32 v47, v52, v50, vcc
	v_xor_b32_e32 v47, v47, v46
	v_sub_co_u32_e32 v46, vcc, v47, v46
.LBB0_99:
	s_andn2_saveexec_b64 s[0:1], s[0:1]
	s_cbranch_execz .LBB0_101
	v_rcp_iflag_f32_e32 v46, v59
	s_sub_i32 s2, 0, s76
	v_mul_f32_e32 v46, 0x4f7ffffe, v46
	v_cvt_u32_f32_e32 v46, v46
	v_mul_lo_u32 v47, s2, v46
	v_mul_hi_u32 v47, v46, v47
	v_add_u32_e32 v46, v46, v47
	v_mul_hi_u32 v46, v52, v46
	v_mul_lo_u32 v47, v46, s76
	v_sub_u32_e32 v47, v52, v47
	v_add_u32_e32 v50, 1, v46
	v_subrev_u32_e32 v52, s76, v47
	v_cmp_le_u32_e32 vcc, s76, v47
	s_nop 1
	v_cndmask_b32_e32 v47, v47, v52, vcc
	v_cndmask_b32_e32 v46, v46, v50, vcc
	v_add_u32_e32 v50, 1, v46
	v_cmp_le_u32_e32 vcc, s76, v47
	s_nop 1
	v_cndmask_b32_e32 v46, v46, v50, vcc
.LBB0_101:
	s_or_b64 exec, exec, s[0:1]
	v_lshl_add_u64 v[52:53], v[48:49], 0, 2
	v_or_b32_e32 v47, s77, v53
	s_mov_b32 s2, 0
	v_cmp_ne_u32_e32 vcc, 0, v47
	s_and_saveexec_b64 s[0:1], vcc
	s_xor_b64 s[0:1], exec, s[0:1]
	s_cbranch_execz .LBB0_103
	s_add_u32 s14, s76, s78
	s_mov_b32 s79, s78
	s_addc_u32 s15, s77, s78
	s_xor_b64 s[16:17], s[14:15], s[78:79]
	v_cvt_f32_u32_e32 v47, s16
	v_cvt_f32_u32_e32 v48, s17
	s_sub_u32 s3, 0, s16
	s_subb_u32 s14, 0, s17
	v_ashrrev_i32_e32 v66, 31, v53
	v_fmamk_f32 v47, v48, 0x4f800000, v47
	v_rcp_f32_e32 v47, v47
	v_mov_b32_e32 v67, v66
	v_mul_f32_e32 v47, 0x5f7ffffc, v47
	v_mul_f32_e32 v48, 0x2f800000, v47
	v_trunc_f32_e32 v48, v48
	v_fmamk_f32 v47, v48, 0xcf800000, v47
	v_cvt_u32_f32_e32 v48, v48
	v_cvt_u32_f32_e32 v47, v47
	v_readfirstlane_b32 s15, v48
	v_readfirstlane_b32 s18, v47
	s_mul_i32 s19, s3, s15
	s_mul_hi_u32 s21, s3, s18
	s_mul_i32 s20, s14, s18
	s_add_i32 s19, s21, s19
	s_add_i32 s19, s19, s20
	s_mul_i32 s22, s3, s18
	s_mul_i32 s21, s18, s19
	s_mul_hi_u32 s23, s18, s22
	s_mul_hi_u32 s20, s18, s19
	s_add_u32 s21, s23, s21
	s_addc_u32 s20, 0, s20
	s_mul_hi_u32 s24, s15, s22
	s_mul_i32 s22, s15, s22
	s_add_u32 s21, s21, s22
	s_mul_hi_u32 s23, s15, s19
	s_addc_u32 s20, s20, s24
	s_addc_u32 s21, s23, 0
	s_mul_i32 s19, s15, s19
	s_add_u32 s19, s20, s19
	s_addc_u32 s20, 0, s21
	s_add_u32 s18, s18, s19
	s_addc_u32 s15, s15, s20
	s_mul_i32 s19, s3, s15
	s_mul_hi_u32 s20, s3, s18
	s_add_i32 s19, s20, s19
	s_mul_i32 s14, s14, s18
	s_add_i32 s19, s19, s14
	s_mul_i32 s3, s3, s18
	s_mul_hi_u32 s20, s15, s3
	s_mul_i32 s21, s15, s3
	s_mul_i32 s23, s18, s19
	s_mul_hi_u32 s3, s18, s3
	s_mul_hi_u32 s22, s18, s19
	s_add_u32 s3, s3, s23
	s_addc_u32 s22, 0, s22
	s_add_u32 s3, s3, s21
	s_mul_hi_u32 s14, s15, s19
	s_addc_u32 s3, s22, s20
	s_addc_u32 s14, s14, 0
	s_mul_i32 s19, s15, s19
	s_add_u32 s3, s3, s19
	s_addc_u32 s14, 0, s14
	s_add_u32 s3, s18, s3
	v_lshl_add_u64 v[48:49], v[52:53], 0, v[66:67]
	s_addc_u32 s18, s15, s14
	v_xor_b32_e32 v50, v48, v66
	v_xor_b32_e32 v47, v49, v66
	v_mad_u64_u32 v[48:49], s[14:15], v50, s18, 0
	v_mul_hi_u32 v52, v50, s3
	v_mov_b32_e32 v53, 0
	v_lshl_add_u64 v[48:49], v[52:53], 0, v[48:49]
	v_mad_u64_u32 v[68:69], s[14:15], v47, s3, 0
	v_add_co_u32_e32 v48, vcc, v48, v68
	v_mad_u64_u32 v[52:53], s[14:15], v47, s18, 0
	s_nop 0
	v_addc_co_u32_e32 v48, vcc, v49, v69, vcc
	v_mov_b32_e32 v49, s2
	s_nop 0
	v_addc_co_u32_e32 v53, vcc, 0, v53, vcc
	v_lshl_add_u64 v[48:49], v[48:49], 0, v[52:53]
	v_mul_lo_u32 v52, s17, v48
	v_mul_lo_u32 v53, s16, v49
	v_mad_u64_u32 v[48:49], s[2:3], s16, v48, 0
	v_add3_u32 v49, v49, v53, v52
	v_sub_u32_e32 v52, v47, v49
	v_mov_b32_e32 v53, s17
	v_sub_co_u32_e32 v48, vcc, v50, v48
	s_nop 1
	v_subb_co_u32_e64 v50, s[2:3], v52, v53, vcc
	v_subrev_co_u32_e64 v52, s[2:3], s16, v48
	v_subb_co_u32_e32 v47, vcc, v47, v49, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v59, s[14:15], 0, v50, s[2:3]
	v_cmp_le_u32_e64 s[14:15], s17, v59
	v_subb_co_u32_e64 v50, s[2:3], v50, v53, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v67, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s16, v52
	v_subrev_co_u32_e64 v53, s[2:3], s16, v52
	s_nop 0
	v_cndmask_b32_e64 v68, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s17, v59
	v_subbrev_co_u32_e64 v50, s[2:3], 0, v50, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v67, v67, v68, s[14:15]
	v_cmp_le_u32_e32 vcc, s17, v47
	v_cmp_ne_u32_e64 s[2:3], 0, v67
	s_nop 0
	v_cndmask_b32_e64 v49, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s16, v48
	v_cndmask_b32_e64 v50, v59, v50, s[2:3]
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s17, v47
	s_nop 1
	v_cndmask_b32_e32 v49, v49, v59, vcc
	v_cmp_ne_u32_e32 vcc, 0, v49
	v_cndmask_b32_e64 v49, v52, v53, s[2:3]
	s_nop 0
	v_cndmask_b32_e32 v48, v48, v49, vcc
	v_cndmask_b32_e32 v47, v47, v50, vcc
	v_xor_b32_e32 v48, v48, v66
	v_xor_b32_e32 v47, v47, v66
	v_sub_co_u32_e32 v48, vcc, v48, v66
	s_nop 1
	v_subb_co_u32_e32 v49, vcc, v47, v66, vcc
.LBB0_103:
	s_andn2_saveexec_b64 s[0:1], s[0:1]
	s_cbranch_execz .LBB0_105
	v_rcp_iflag_f32_e32 v47, v59
	s_sub_i32 s2, 0, s76
	v_mov_b32_e32 v49, 0
	v_mul_f32_e32 v47, 0x4f7ffffe, v47
	v_cvt_u32_f32_e32 v47, v47
	v_mul_lo_u32 v48, s2, v47
	v_mul_hi_u32 v48, v47, v48
	v_add_u32_e32 v47, v47, v48
	v_mul_hi_u32 v47, v52, v47
	v_mul_lo_u32 v47, v47, s76
	v_sub_u32_e32 v47, v52, v47
	v_subrev_u32_e32 v48, s76, v47
	v_cmp_le_u32_e32 vcc, s76, v47
	s_nop 1
	v_cndmask_b32_e32 v47, v47, v48, vcc
	v_subrev_u32_e32 v48, s76, v47
	v_cmp_le_u32_e32 vcc, s76, v47
	s_nop 1
	v_cndmask_b32_e32 v48, v47, v48, vcc
.LBB0_105:
	s_or_b64 exec, exec, s[0:1]
	v_ashrrev_i32_e32 v47, 31, v51
	v_xor_b32_e32 v46, v46, v47
	v_ashrrev_i32_e32 v47, 31, v49
	v_mul_lo_u32 v46, v46, s54
	v_and_b32_e32 v47, s76, v47
	v_add3_u32 v46, v46, v48, v47
	buffer_load_ubyte v72, v46, s[8:11], 0 offen
	s_waitcnt vmcnt(20)
	v_lshlrev_b16_e32 v46, 8, v79
	s_waitcnt vmcnt(18)
	v_lshlrev_b16_e32 v47, 8, v81
	v_or_b32_e32 v46, v75, v46
	v_or_b32_sdwa v47, v77, v47 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v50, 8, v64
	v_or_b32_sdwa v49, v46, v47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v46, 8, v56
	v_lshlrev_b16_e32 v47, 8, v57
	v_or_b32_e32 v46, v54, v46
	v_or_b32_sdwa v47, v55, v47 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v50, v60, v50 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v48, v46, v47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v46, 8, v71
	v_lshlrev_b16_e32 v47, 8, v74
	v_or_b32_e32 v46, v65, v46
	v_or_b32_sdwa v47, v70, v47 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt vmcnt(2)
	v_lshlrev_b16_e32 v51, 8, v109
	v_or_b32_sdwa v47, v46, v47 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v46, 8, v61
	v_or_b32_e32 v46, v58, v46
	v_or_b32_sdwa v46, v46, v50 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v50, 8, v108
	v_or_b32_e32 v50, v106, v50
	v_or_b32_sdwa v51, v107, v51 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt vmcnt(0) lgkmcnt(0)
	v_or_b32_sdwa v53, v50, v51 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v50, 8, v104
	v_lshlrev_b16_e32 v51, 8, v105
	v_or_b32_e32 v50, v102, v50
	v_or_b32_sdwa v51, v103, v51 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_barrier
	v_or_b32_sdwa v52, v50, v51 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v50, 8, v93
	v_lshlrev_b16_e32 v51, 8, v95
	v_or_b32_e32 v50, v89, v50
	v_or_b32_sdwa v51, v91, v51 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_nop 0
	v_or_b32_sdwa v51, v50, v51 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v50, 8, v85
	v_or_b32_e32 v43, v43, v50
	v_lshlrev_b16_e32 v50, 8, v87
	v_or_b32_sdwa v50, v83, v50 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_nop 0
	v_or_b32_sdwa v50, v43, v50 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_and_b32_e32 v43, 0xff, v130
	ds_read2st64_b32 v[68:69], v63 offset0:196 offset1:197
	ds_read_b128 v[102:105], v78 offset:16384
	ds_read_b128 v[106:109], v78 offset:18432
	ds_read_b128 v[182:185], v76 offset:16384
	ds_read_b128 v[186:189], v76 offset:18432
	ds_read_b128 v[190:193], v78 offset:20480
	ds_read_b128 v[194:197], v78 offset:22528
	ds_read_b128 v[198:201], v76 offset:20480
	ds_read_b128 v[74:77], v76 offset:22528
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[144:147], v[16:19], v[12:15], v101, v43 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_and_b32_e32 v63, 0xff, v114
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[126:129], v[16:19], v[4:7], v116, v43 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b32 v70, v96 offset:50176
	ds_read_b32 v71, v98 offset:50176
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[170:173], v[16:19], v[0:3], v116, v43 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[122:125], v[16:19], v[36:39], v117, v43 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[154:157], v[16:19], v[8:11], v117, v43 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], v[174:177], v[28:31], v[12:15], v101, v63 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[118:121], v[16:19], v[32:35], v100, v43 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[166:169], v[28:31], v[4:7], v116, v63 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[162:165], v[28:31], v[0:3], v116, v63 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[158:161], v[28:31], v[36:39], v117, v63 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[150:153], v[28:31], v[8:11], v117, v63 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 v[78:81], v80 offset:16384
	ds_read_b128 v[114:117], v82 offset:16384
	ds_read_b128 v[82:85], v84 offset:16384
	ds_read_b128 v[122:125], v86 offset:16384
	ds_read_b128 v[86:89], v88 offset:16384
	ds_read_b128 v[126:129], v90 offset:16384
	ds_read_b128 v[90:93], v92 offset:16384
	ds_read_b128 v[118:121], v94 offset:16384
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[178:181], v[16:19], v[24:27], v100, v43 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[136:139], v[16:19], v[20:23], v101, v43 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[110:113], v[28:31], v[12:15], v100, v63 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], v[140:143], v[28:31], v[20:23], v101, v63 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[132:135], v[28:31], v[24:27], v100, v63 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_and_b32_e32 v43, 0xffff, v62
	v_readlane_b32 s0, v202, 33
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[106:109], v[46:49], v[0:3], v68, v43 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[186:189], v[50:53], v[0:3], v68, v72 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[190:193], v[46:49], v[36:39], v69, v43 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[198:201], v[50:53], v[0:3], v69, v72 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[194:197], v[46:49], v[8:11], v69, v43 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[74:77], v[50:53], v[0:3], v69, v72 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[78:81], v[46:49], v[54:57], v70, v43 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[114:117], v[50:53], v[0:3], v70, v72 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[82:85], v[46:49], v[58:61], v70, v43 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[102:105], v[46:49], v[4:7], v68, v43 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[122:125], v[50:53], v[0:3], v70, v72 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[86:89], v[46:49], v[64:67], v71, v43 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[182:185], v[50:53], v[4:7], v68, v72 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[126:129], v[50:53], v[0:3], v71, v72 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[90:93], v[46:49], v[32:35], v71, v43 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[118:121], v[50:53], v[0:3], v71, v72 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 1
	v_mov_b32_e32 v32, s0
.LBB0_106:
	v_readlane_b32 s0, v202, 4
	v_readlane_b32 s1, v202, 5
	s_load_dwordx2 s[16:17], s[0:1], 0x60
	s_and_b64 s[0:1], s[26:27], exec
	v_readlane_b32 s0, v202, 10
	v_readlane_b32 s2, v202, 12
	v_readlane_b32 s1, v202, 11
	v_readlane_b32 s3, v202, 13
	s_cselect_b32 s1, s1, s3
	s_cselect_b32 s0, s0, s2
	s_ashr_i32 s2, s1, 31
	s_lshr_b32 s2, s2, 25
	s_add_u32 s0, s0, s2
	s_addc_u32 s1, s1, 0
	s_ashr_i64 s[0:1], s[0:1], 7
	s_sub_u32 s2, 0, s0
	s_subb_u32 s3, 0, s1
	s_add_u32 s14, s0, 1
	s_addc_u32 s15, s1, 0
	s_and_b64 s[0:1], s[26:27], exec
	s_cselect_b32 s1, s3, s15
	s_cselect_b32 s0, s2, s14
	v_readlane_b32 s14, v202, 0
	s_mul_i32 s2, s1, s14
	s_mul_hi_u32 s3, s0, s14
	s_add_i32 s72, s3, s2
	s_mul_i32 s73, s0, s14
	s_add_u32 s14, s73, s67
	s_mul_i32 s2, s68, s1
	s_mul_hi_u32 s3, s68, s0
	s_addc_u32 s15, s72, 0
	s_add_i32 s2, s3, s2
	s_mul_i32 s3, s69, s0
	s_add_i32 s3, s2, s3
	s_mul_i32 s2, s68, s0
	s_lshl_b64 s[20:21], s[2:3], 5
	s_sub_u32 s62, 0, s20
	s_subb_u32 s63, 0, s21
	s_sub_u32 s18, s14, s20
	v_writelane_b32 v202, s20, 0
	s_subb_u32 s19, s15, s21
	s_nop 0
	v_writelane_b32 v202, s21, 1
	s_nop 0
	v_readlane_b32 s68, v202, 16
	v_readlane_b32 s69, v202, 17
	s_or_b64 s[20:21], s[18:19], s[68:69]
	s_cmp_lg_u32 s21, 0
	s_cbranch_scc0 .LBB0_197
	s_ashr_i32 s20, s69, 31
	s_add_u32 s22, s68, s20
	s_mov_b32 s21, s20
	s_addc_u32 s23, s69, s20
	s_xor_b64 s[20:21], s[22:23], s[20:21]
	v_cvt_f32_u32_e32 v33, s20
	v_cvt_f32_u32_e32 v34, s21
	s_sub_u32 s24, 0, s20
	s_subb_u32 s25, 0, s21
	v_fmamk_f32 v33, v34, 0x4f800000, v33
	v_rcp_f32_e32 v33, v33
	s_nop 0
	v_mul_f32_e32 v33, 0x5f7ffffc, v33
	v_mul_f32_e32 v34, 0x2f800000, v33
	v_trunc_f32_e32 v34, v34
	v_fmamk_f32 v33, v34, 0xcf800000, v33
	v_cvt_u32_f32_e32 v34, v34
	v_cvt_u32_f32_e32 v33, v33
	v_readfirstlane_b32 s26, v34
	v_readfirstlane_b32 s27, v33
	s_mul_i32 s28, s24, s26
	s_mul_hi_u32 s34, s24, s27
	s_mul_i32 s29, s25, s27
	s_add_i32 s28, s34, s28
	s_add_i32 s28, s28, s29
	s_mul_i32 s35, s24, s27
	s_mul_i32 s34, s27, s28
	s_mul_hi_u32 s48, s27, s35
	s_mul_hi_u32 s29, s27, s28
	s_add_u32 s34, s48, s34
	s_addc_u32 s29, 0, s29
	s_mul_hi_u32 s49, s26, s35
	s_mul_i32 s35, s26, s35
	s_add_u32 s34, s34, s35
	s_mul_hi_u32 s48, s26, s28
	s_addc_u32 s29, s29, s49
	s_addc_u32 s34, s48, 0
	s_mul_i32 s28, s26, s28
	s_add_u32 s28, s29, s28
	s_addc_u32 s29, 0, s34
	s_add_u32 s27, s27, s28
	s_addc_u32 s26, s26, s29
	s_mul_i32 s28, s24, s26
	s_mul_hi_u32 s29, s24, s27
	s_add_i32 s28, s29, s28
	s_mul_i32 s25, s25, s27
	s_add_i32 s28, s28, s25
	s_mul_i32 s24, s24, s27
	s_mul_hi_u32 s29, s26, s24
	s_mul_i32 s34, s26, s24
	s_mul_i32 s48, s27, s28
	s_mul_hi_u32 s24, s27, s24
	s_mul_hi_u32 s35, s27, s28
	s_add_u32 s24, s24, s48
	s_addc_u32 s35, 0, s35
	s_add_u32 s24, s24, s34
	s_mul_hi_u32 s25, s26, s28
	s_addc_u32 s24, s35, s29
	s_addc_u32 s25, s25, 0
	s_mul_i32 s28, s26, s28
	s_add_u32 s24, s24, s28
	s_addc_u32 s25, 0, s25
	s_add_u32 s28, s27, s24
	s_addc_u32 s29, s26, s25
	s_ashr_i32 s24, s19, 31
	s_add_u32 s26, s18, s24
	s_mov_b32 s25, s24
	s_addc_u32 s27, s19, s24
	s_xor_b64 s[26:27], s[26:27], s[24:25]
	s_mul_i32 s34, s26, s29
	s_mul_hi_u32 s35, s26, s28
	s_mul_hi_u32 s19, s26, s29
	s_add_u32 s34, s35, s34
	s_addc_u32 s19, 0, s19
	s_mul_hi_u32 s48, s27, s28
	s_mul_i32 s28, s27, s28
	s_add_u32 s28, s34, s28
	s_mul_hi_u32 s35, s27, s29
	s_addc_u32 s19, s19, s48
	s_addc_u32 s28, s35, 0
	s_mul_i32 s29, s27, s29
	s_add_u32 s19, s19, s29
	s_addc_u32 s28, 0, s28
	s_mul_i32 s28, s20, s28
	s_mul_hi_u32 s29, s20, s19
	s_add_i32 s28, s29, s28
	s_mul_i32 s29, s21, s19
	s_add_i32 s48, s28, s29
	s_sub_i32 s34, s27, s48
	s_mul_i32 s19, s20, s19
	s_sub_u32 s19, s26, s19
	s_cselect_b64 s[28:29], -1, 0
	s_subb_u32 s26, s34, s21
	s_sub_u32 s49, s19, s20
	s_cselect_b64 s[34:35], -1, 0
	s_subb_u32 s50, s26, 0
	s_cmp_ge_u32 s50, s21
	s_cselect_b32 s51, -1, 0
	s_cmp_ge_u32 s49, s20
	s_cselect_b32 s52, -1, 0
	s_cmp_eq_u32 s50, s21
	s_cselect_b32 s51, s52, s51
	s_cmp_lg_u64 s[34:35], 0
	s_subb_u32 s26, s26, s21
	s_sub_u32 s34, s49, s20
	s_subb_u32 s26, s26, 0
	s_cmp_lg_u32 s51, 0
	s_cselect_b32 s34, s34, s49
	s_cselect_b32 s26, s26, s50
	s_cmp_lg_u64 s[28:29], 0
	s_subb_u32 s27, s27, s48
	s_cmp_ge_u32 s27, s21
	s_cselect_b32 s28, -1, 0
	s_cmp_ge_u32 s19, s20
	s_cselect_b32 s20, -1, 0
	s_cmp_eq_u32 s27, s21
	s_cselect_b32 s20, s20, s28
	s_cmp_lg_u32 s20, 0
	s_cselect_b32 s21, s26, s27
	s_cselect_b32 s20, s34, s19
	s_xor_b64 s[20:21], s[20:21], s[24:25]
	s_sub_u32 s20, s20, s24
	s_subb_u32 s21, s21, s24
	v_cvt_f32_u32_e32 v74, s68
	s_cbranch_execnz .LBB0_109
.LBB0_108:
	v_rcp_iflag_f32_e32 v33, v74
	s_sub_i32 s19, 0, s68
	s_mov_b32 s21, 0
	v_mul_f32_e32 v33, 0x4f7ffffe, v33
	v_cvt_u32_f32_e32 v33, v33
	s_nop 0
	v_readfirstlane_b32 s20, v33
	s_mul_i32 s19, s19, s20
	s_mul_hi_u32 s19, s20, s19
	s_add_i32 s20, s20, s19
	s_mul_hi_u32 s19, s18, s20
	s_mul_i32 s19, s19, s68
	s_sub_i32 s18, s18, s19
	s_sub_i32 s19, s18, s68
	s_cmp_ge_u32 s18, s68
	s_cselect_b32 s18, s19, s18
	s_sub_i32 s19, s18, s68
	s_cmp_ge_u32 s18, s68
	s_cselect_b32 s20, s19, s18
.LBB0_109:
	v_cmp_gt_i64_e64 s[18:19], s[84:85], 5
	s_and_b64 s[18:19], s[18:19], exec
	s_cselect_b32 s61, s83, 0
	s_cselect_b32 s60, s82, 0
	s_ashr_i32 s18, s15, 31
	s_mov_b32 s19, s18
	s_xor_b64 s[22:23], s[18:19], s[14:15]
	s_ashr_i32 s24, s23, 31
	s_lshr_b32 s24, s24, 27
	s_add_u32 s22, s22, s24
	s_addc_u32 s23, s23, 0
	s_ashr_i64 s[22:23], s[22:23], 5
	s_xor_b64 s[26:27], s[22:23], s[18:19]
	s_ashr_i32 s18, s27, 31
	s_add_u32 s22, s26, s18
	s_mov_b32 s19, s18
	s_addc_u32 s23, s27, s18
	s_xor_b64 s[28:29], s[22:23], s[18:19]
	s_ashr_i32 s22, s1, 31
	s_add_u32 s24, s0, s22
	s_mov_b32 s23, s22
	s_addc_u32 s25, s1, s22
	s_xor_b64 s[24:25], s[24:25], s[22:23]
	v_cvt_f32_u32_e32 v33, s24
	v_cvt_f32_u32_e32 v34, s25
	s_sub_u32 s34, 0, s24
	s_subb_u32 s35, 0, s25
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_fmac_f32_e32 v33, 0x4f800000, v34
	v_rcp_f32_e32 v33, v33
	v_writelane_b32 v202, s62, 10
	s_mov_b32 s65, 0
	v_mul_f32_e32 v33, 0x5f7ffffc, v33
	v_mul_f32_e32 v34, 0x2f800000, v33
	v_trunc_f32_e32 v34, v34
	v_fmac_f32_e32 v33, 0xcf800000, v34
	v_cvt_u32_f32_e32 v33, v33
	v_cvt_u32_f32_e32 v34, v34
	v_writelane_b32 v202, s63, 11
	v_readfirstlane_b32 s48, v33
	v_readfirstlane_b32 s51, v34
	s_mul_hi_u32 s50, s34, s48
	s_mul_i32 s52, s34, s51
	s_mul_i32 s49, s35, s48
	s_add_i32 s50, s50, s52
	s_add_i32 s50, s50, s49
	s_mul_i32 s55, s34, s48
	s_mul_i32 s52, s48, s50
	s_mul_hi_u32 s56, s48, s55
	s_mul_hi_u32 s49, s48, s50
	s_add_u32 s52, s56, s52
	s_addc_u32 s49, 0, s49
	s_mul_hi_u32 s57, s51, s55
	s_mul_i32 s55, s51, s55
	s_add_u32 s52, s52, s55
	s_mul_hi_u32 s56, s51, s50
	s_addc_u32 s49, s49, s57
	s_addc_u32 s52, s56, 0
	s_mul_i32 s50, s51, s50
	s_add_u32 s49, s49, s50
	s_addc_u32 s50, 0, s52
	s_add_u32 s48, s48, s49
	s_addc_u32 s49, s51, s50
	s_mul_i32 s50, s34, s49
	s_mul_hi_u32 s51, s34, s48
	s_add_i32 s50, s51, s50
	s_mul_i32 s35, s35, s48
	s_add_i32 s50, s50, s35
	s_mul_i32 s34, s34, s48
	s_mul_hi_u32 s51, s49, s34
	s_mul_i32 s52, s49, s34
	s_mul_i32 s56, s48, s50
	s_mul_hi_u32 s34, s48, s34
	s_mul_hi_u32 s55, s48, s50
	s_add_u32 s34, s34, s56
	s_addc_u32 s55, 0, s55
	s_add_u32 s34, s34, s52
	s_mul_hi_u32 s35, s49, s50
	s_addc_u32 s34, s55, s51
	s_addc_u32 s35, s35, 0
	s_mul_i32 s50, s49, s50
	s_add_u32 s34, s34, s50
	s_addc_u32 s50, 0, s35
	s_add_u32 s35, s48, s34
	s_addc_u32 s34, s49, s50
	s_mul_i32 s49, s28, s34
	s_mul_hi_u32 s50, s28, s35
	s_mul_hi_u32 s48, s28, s34
	s_add_u32 s49, s50, s49
	s_addc_u32 s48, 0, s48
	s_mul_i32 s52, s29, s35
	s_mul_hi_u32 s51, s29, s35
	s_add_u32 s49, s49, s52
	s_mul_hi_u32 s50, s29, s34
	s_addc_u32 s48, s48, s51
	s_addc_u32 s49, s50, 0
	s_mul_i32 s50, s29, s34
	s_add_u32 s48, s48, s50
	s_addc_u32 s49, 0, s49
	s_mul_i32 s49, s24, s49
	s_mul_hi_u32 s50, s24, s48
	s_add_i32 s49, s50, s49
	s_mul_i32 s50, s25, s48
	s_add_i32 s52, s49, s50
	s_sub_i32 s50, s29, s52
	s_mul_i32 s48, s24, s48
	s_sub_u32 s28, s28, s48
	s_cselect_b64 s[48:49], -1, 0
	s_subb_u32 s55, s50, s25
	s_sub_u32 s56, s28, s24
	s_cselect_b64 s[50:51], -1, 0
	s_subb_u32 s57, s55, 0
	s_cmp_ge_u32 s57, s25
	s_cselect_b32 s58, -1, 0
	s_cmp_ge_u32 s56, s24
	s_cselect_b32 s59, -1, 0
	s_cmp_eq_u32 s57, s25
	s_cselect_b32 s58, s59, s58
	s_cmp_lg_u64 s[50:51], 0
	s_subb_u32 s50, s55, s25
	s_sub_u32 s51, s56, s24
	s_subb_u32 s50, s50, 0
	s_cmp_lg_u32 s58, 0
	s_cselect_b32 s51, s51, s56
	s_cselect_b32 s50, s50, s57
	s_cmp_lg_u64 s[48:49], 0
	s_subb_u32 s29, s29, s52
	s_cmp_ge_u32 s29, s25
	s_cselect_b32 s48, -1, 0
	s_cmp_ge_u32 s28, s24
	s_cselect_b32 s49, -1, 0
	s_cmp_eq_u32 s29, s25
	s_cselect_b32 s48, s49, s48
	s_cmp_lg_u32 s48, 0
	s_cselect_b32 s29, s50, s29
	s_cselect_b32 s28, s51, s28
	s_xor_b64 s[28:29], s[28:29], s[18:19]
	s_sub_u32 s48, s28, s18
	s_subb_u32 s49, s29, s18
	s_ashr_i32 s28, s49, 31
	s_mov_b32 s29, s28
	s_and_b64 s[28:29], s[28:29], s[0:1]
	s_add_u32 s76, s28, s48
	s_addc_u32 s77, s29, s49
	v_cmp_lt_i64_e64 s[28:29], s[2:3], 1
	v_cmp_gt_i64_e64 s[2:3], s[2:3], 0
	s_and_b64 s[74:75], vcc, s[28:29]
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[2:3]
	s_nop 0
	v_readfirstlane_b32 s2, v32
	s_mov_b32 s3, s2
	s_xor_b64 s[28:29], s[62:63], s[2:3]
	s_ashr_i32 s62, s69, 31
	s_add_u32 s2, s68, s62
	s_mov_b32 s63, s62
	s_addc_u32 s3, s69, s62
	s_xor_b64 s[2:3], s[2:3], s[62:63]
	v_cvt_f32_u32_e32 v33, s2
	v_cvt_f32_u32_e32 v34, s3
	s_sub_u32 s48, 0, s2
	s_subb_u32 s49, 0, s3
	v_fmac_f32_e32 v33, 0x4f800000, v34
	v_rcp_f32_e32 v33, v33
	s_nop 0
	v_mul_f32_e32 v33, 0x5f7ffffc, v33
	v_mul_f32_e32 v34, 0x2f800000, v33
	v_trunc_f32_e32 v34, v34
	v_fmac_f32_e32 v33, 0xcf800000, v34
	v_cvt_u32_f32_e32 v33, v33
	v_cvt_u32_f32_e32 v34, v34
	v_readfirstlane_b32 s50, v33
	v_readfirstlane_b32 s55, v34
	s_mul_hi_u32 s52, s48, s50
	s_mul_i32 s56, s48, s55
	s_mul_i32 s51, s49, s50
	s_add_i32 s52, s52, s56
	s_add_i32 s52, s52, s51
	s_mul_i32 s57, s48, s50
	s_mul_i32 s56, s50, s52
	s_mul_hi_u32 s58, s50, s57
	s_mul_hi_u32 s51, s50, s52
	s_add_u32 s56, s58, s56
	s_addc_u32 s51, 0, s51
	s_mul_hi_u32 s59, s55, s57
	s_mul_i32 s57, s55, s57
	s_add_u32 s56, s56, s57
	s_mul_hi_u32 s58, s55, s52
	s_addc_u32 s51, s51, s59
	s_addc_u32 s56, s58, 0
	s_mul_i32 s52, s55, s52
	s_add_u32 s51, s51, s52
	s_addc_u32 s52, 0, s56
	s_add_u32 s50, s50, s51
	s_addc_u32 s51, s55, s52
	s_mul_i32 s52, s48, s51
	s_mul_hi_u32 s55, s48, s50
	s_add_i32 s52, s55, s52
	s_mul_i32 s49, s49, s50
	s_add_i32 s52, s52, s49
	s_mul_i32 s48, s48, s50
	s_mul_hi_u32 s55, s51, s48
	s_mul_i32 s56, s51, s48
	s_mul_i32 s58, s50, s52
	s_mul_hi_u32 s48, s50, s48
	s_mul_hi_u32 s57, s50, s52
	s_add_u32 s48, s48, s58
	s_addc_u32 s57, 0, s57
	s_add_u32 s48, s48, s56
	s_mul_hi_u32 s49, s51, s52
	s_addc_u32 s48, s57, s55
	s_addc_u32 s49, s49, 0
	s_mul_i32 s52, s51, s52
	s_add_u32 s48, s48, s52
	s_addc_u32 s49, 0, s49
	s_add_u32 s50, s50, s48
	s_addc_u32 s51, s51, s49
	s_ashr_i32 s48, s29, 31
	s_add_u32 s28, s28, s48
	s_mov_b32 s49, s48
	s_addc_u32 s29, s29, s48
	s_xor_b64 s[28:29], s[28:29], s[48:49]
	s_mul_i32 s55, s28, s51
	s_mul_hi_u32 s56, s28, s50
	s_mul_hi_u32 s52, s28, s51
	s_add_u32 s55, s56, s55
	s_addc_u32 s52, 0, s52
	s_mul_hi_u32 s57, s29, s50
	s_mul_i32 s50, s29, s50
	s_add_u32 s50, s55, s50
	s_mul_hi_u32 s56, s29, s51
	s_addc_u32 s50, s52, s57
	s_addc_u32 s52, s56, 0
	s_mul_i32 s51, s29, s51
	s_add_u32 s55, s50, s51
	s_addc_u32 s52, 0, s52
	s_mul_i32 s50, s2, s52
	s_mul_hi_u32 s51, s2, s55
	s_add_i32 s50, s51, s50
	s_mul_i32 s51, s3, s55
	s_add_i32 s56, s50, s51
	s_sub_i32 s57, s29, s56
	s_mul_i32 s50, s2, s55
	s_sub_u32 s28, s28, s50
	s_cselect_b64 s[50:51], -1, 0
	s_subb_u32 s57, s57, s3
	s_sub_u32 s58, s28, s2
	s_subb_u32 s57, s57, 0
	s_cmp_ge_u32 s57, s3
	s_cselect_b32 s59, -1, 0
	s_cmp_ge_u32 s58, s2
	s_cselect_b32 s58, -1, 0
	s_cmp_eq_u32 s57, s3
	s_cselect_b32 s57, s58, s59
	s_add_u32 s58, s55, 1
	s_addc_u32 s59, s52, 0
	s_add_u32 s64, s55, 2
	s_addc_u32 s66, s52, 0
	s_cmp_lg_u32 s57, 0
	s_cselect_b32 s57, s64, s58
	s_cselect_b32 s58, s66, s59
	s_cmp_lg_u64 s[50:51], 0
	s_subb_u32 s29, s29, s56
	s_cmp_ge_u32 s29, s3
	s_cselect_b32 s50, -1, 0
	s_cmp_ge_u32 s28, s2
	s_cselect_b32 s2, -1, 0
	s_cmp_eq_u32 s29, s3
	s_cselect_b32 s2, s2, s50
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s3, s58, s52
	s_cselect_b32 s2, s57, s55
	s_xor_b64 s[28:29], s[48:49], s[62:63]
	s_xor_b64 s[2:3], s[2:3], s[28:29]
	s_sub_u32 s2, s2, s28
	s_subb_u32 s3, s3, s29
	v_xor_b32_e32 v33, s2, v32
	v_xor_b32_e32 v32, s3, v32
	v_cndmask_b32_e64 v73, 0, v32, s[74:75]
	v_writelane_b32 v202, s74, 4
	s_nop 1
	v_writelane_b32 v202, s75, 5
	v_cndmask_b32_e64 v72, 0, v33, s[74:75]
	v_readlane_b32 s28, v202, 14
	v_readlane_b32 s29, v202, 15
	s_mul_i32 s2, s28, s1
	s_mul_hi_u32 s3, s28, s0
	s_add_i32 s2, s3, s2
	s_mul_i32 s3, s29, s0
	s_add_i32 s2, s2, s3
	s_mul_i32 s3, s28, s0
	v_mov_b32_e32 v32, s3
	v_mov_b32_e32 v33, s2
	v_cmp_ge_i64_e64 s[2:3], s[14:15], v[32:33]
	s_and_b64 s[56:57], vcc, s[2:3]
	s_ashr_i32 s2, s21, 31
	s_mov_b32 s3, s2
	s_and_b64 s[2:3], s[2:3], s[68:69]
	s_add_u32 s20, s20, s28
	s_addc_u32 s21, s21, s29
	s_add_u32 s28, s20, s2
	s_addc_u32 s29, s21, s3
	s_xor_b64 s[2:3], s[18:19], s[26:27]
	s_ashr_i32 s20, s3, 31
	s_add_u32 s2, s2, s20
	s_mov_b32 s21, s20
	s_addc_u32 s3, s3, s20
	s_xor_b64 s[2:3], s[2:3], s[20:21]
	s_mul_i32 s27, s2, s34
	s_mul_hi_u32 s48, s2, s35
	s_mul_hi_u32 s26, s2, s34
	s_add_u32 s27, s48, s27
	s_addc_u32 s26, 0, s26
	s_mul_hi_u32 s49, s3, s35
	s_mul_i32 s35, s3, s35
	s_add_u32 s27, s27, s35
	s_mul_hi_u32 s48, s3, s34
	s_addc_u32 s26, s26, s49
	s_addc_u32 s27, s48, 0
	s_mul_i32 s34, s3, s34
	s_add_u32 s34, s26, s34
	s_addc_u32 s35, 0, s27
	s_mul_i32 s26, s24, s35
	s_mul_hi_u32 s27, s24, s34
	s_add_i32 s26, s27, s26
	s_mul_i32 s27, s25, s34
	s_add_i32 s48, s26, s27
	s_sub_i32 s49, s3, s48
	s_mul_i32 s26, s24, s34
	s_sub_u32 s2, s2, s26
	s_cselect_b64 s[26:27], -1, 0
	s_subb_u32 s49, s49, s25
	s_sub_u32 s50, s2, s24
	s_subb_u32 s49, s49, 0
	s_cmp_ge_u32 s49, s25
	s_cselect_b32 s51, -1, 0
	s_cmp_ge_u32 s50, s24
	s_cselect_b32 s50, -1, 0
	s_cmp_eq_u32 s49, s25
	s_cselect_b32 s49, s50, s51
	s_add_u32 s50, s34, 1
	s_addc_u32 s51, s35, 0
	s_add_u32 s52, s34, 2
	s_addc_u32 s55, s35, 0
	s_cmp_lg_u32 s49, 0
	s_cselect_b32 s49, s52, s50
	s_cselect_b32 s50, s55, s51
	s_cmp_lg_u64 s[26:27], 0
	s_subb_u32 s3, s3, s48
	s_cmp_ge_u32 s3, s25
	s_cselect_b32 s26, -1, 0
	s_cmp_ge_u32 s2, s24
	s_cselect_b32 s2, -1, 0
	s_cmp_eq_u32 s3, s25
	s_cselect_b32 s2, s2, s26
	s_cmp_lg_u32 s2, 0
	s_cselect_b32 s3, s50, s35
	s_cselect_b32 s2, s49, s34
	s_xor_b64 s[20:21], s[20:21], s[22:23]
	s_xor_b64 s[2:3], s[2:3], s[20:21]
	s_sub_u32 s2, s2, s20
	s_subb_u32 s3, s3, s21
	s_xor_b64 s[2:3], s[2:3], s[18:19]
	s_lshl_b64 s[2:3], s[2:3], 5
	s_and_b32 s64, s14, 31
	s_or_b64 s[2:3], s[2:3], s[64:65]
	v_mov_b64_e32 v[32:33], s[84:85]
	v_writelane_b32 v202, s56, 12
	s_and_b64 s[18:19], s[56:57], exec
	v_cmp_ge_i64_e32 vcc, s[60:61], v[32:33]
	s_cselect_b32 s35, s29, s3
	s_cselect_b32 s34, s28, s2
	v_writelane_b32 v202, s57, 13
	s_cbranch_vccnz .LBB0_129
	v_readlane_b32 s22, v202, 26
	v_readlane_b32 s23, v202, 27
	s_and_b64 s[2:3], s[22:23], exec
	v_readlane_b32 s2, v202, 6
	v_readlane_b32 s18, v202, 8
	v_readlane_b32 s3, v202, 7
	v_readlane_b32 s19, v202, 9
	s_cselect_b32 s3, s3, s19
	s_cselect_b32 s2, s2, s18
	s_ashr_i32 s18, s3, 31
	s_lshr_b32 s18, s18, 27
	s_add_u32 s2, s2, s18
	s_addc_u32 s3, s3, 0
	s_ashr_i64 s[2:3], s[2:3], 5
	s_sub_u32 s18, 0, s2
	s_subb_u32 s19, 0, s3
	s_add_u32 s20, s2, 1
	s_addc_u32 s21, s3, 0
	s_and_b64 s[2:3], s[22:23], exec
	s_cselect_b32 s2, s18, s20
	s_cselect_b32 s3, s19, s21
	s_and_b32 s18, s2, 31
	s_max_u32 s22, s18, 1
	s_cmp_lg_u32 s18, 0
	s_cselect_b64 s[18:19], -1, 0
	s_ashr_i32 s20, s3, 31
	s_mov_b32 s21, s20
	s_xor_b64 s[2:3], s[20:21], s[2:3]
	s_ashr_i32 s23, s3, 31
	s_lshr_b32 s23, s23, 27
	s_add_u32 s2, s2, s23
	s_addc_u32 s3, s3, 0
	s_ashr_i64 s[2:3], s[2:3], 5
	s_xor_b64 s[2:3], s[2:3], s[20:21]
	s_mul_i32 s3, s0, s3
	s_mul_hi_u32 s20, s0, s2
	s_add_i32 s3, s20, s3
	s_mul_i32 s1, s1, s2
	s_add_i32 s1, s3, s1
	s_mul_i32 s0, s0, s2
	s_lshl_b64 s[2:3], s[0:1], 5
	v_mov_b64_e32 v[32:33], s[2:3]
	v_cmp_ge_i64_e32 vcc, s[14:15], v[32:33]
	v_cvt_f32_u32_e32 v32, s22
	v_mov_b32_e32 v33, 0x4f800000
	s_and_b64 s[0:1], s[18:19], vcc
	s_sub_u32 s2, s67, s2
	v_fmac_f32_e32 v32, 0, v33
	v_rcp_f32_e32 v32, v32
	s_subb_u32 s3, 0, s3
	s_add_u32 s14, s2, s73
	s_addc_u32 s15, s3, s72
	v_mul_f32_e32 v32, 0x5f7ffffc, v32
	v_mul_f32_e32 v33, 0x2f800000, v32
	v_trunc_f32_e32 v33, v33
	v_fmac_f32_e32 v32, 0xcf800000, v33
	v_cvt_u32_f32_e32 v34, v33
	v_cvt_u32_f32_e32 v32, v32
	s_ashr_i32 s2, s15, 31
	s_mov_b32 s3, s2
	s_xor_b64 s[14:15], s[2:3], s[14:15]
	s_sub_u32 s18, 0, s22
	v_readfirstlane_b32 s20, v34
	v_readfirstlane_b32 s23, v32
	s_subb_u32 s19, 0, 0
	s_mul_i32 s21, s18, s20
	s_mul_hi_u32 s24, s18, s23
	s_add_i32 s21, s24, s21
	s_mul_i32 s24, s19, s23
	s_add_i32 s21, s21, s24
	s_mul_i32 s26, s18, s23
	s_mul_i32 s25, s23, s21
	s_mul_hi_u32 s27, s23, s26
	s_mul_hi_u32 s24, s23, s21
	s_add_u32 s25, s27, s25
	s_addc_u32 s24, 0, s24
	s_mul_hi_u32 s28, s20, s26
	s_mul_i32 s26, s20, s26
	s_add_u32 s25, s25, s26
	s_mul_hi_u32 s27, s20, s21
	s_addc_u32 s24, s24, s28
	s_addc_u32 s25, s27, 0
	s_mul_i32 s21, s20, s21
	s_add_u32 s21, s24, s21
	s_addc_u32 s24, 0, s25
	s_add_u32 s21, s23, s21
	s_addc_u32 s20, s20, s24
	s_mul_i32 s23, s18, s20
	s_mul_hi_u32 s24, s18, s21
	s_add_i32 s23, s24, s23
	s_mul_i32 s19, s19, s21
	s_add_i32 s23, s23, s19
	s_mul_i32 s18, s18, s21
	s_mul_hi_u32 s24, s20, s18
	s_mul_i32 s25, s20, s18
	s_mul_i32 s27, s21, s23
	s_mul_hi_u32 s18, s21, s18
	s_mul_hi_u32 s26, s21, s23
	s_add_u32 s18, s18, s27
	s_addc_u32 s26, 0, s26
	s_add_u32 s18, s18, s25
	s_mul_hi_u32 s19, s20, s23
	s_addc_u32 s18, s26, s24
	s_addc_u32 s19, s19, 0
	s_mul_i32 s23, s20, s23
	s_add_u32 s18, s18, s23
	s_addc_u32 s19, 0, s19
	s_add_u32 s21, s21, s18
	s_addc_u32 s20, s20, s19
	s_ashr_i32 s18, s15, 31
	s_add_u32 s14, s14, s18
	s_mov_b32 s19, s18
	s_addc_u32 s15, s15, s18
	s_xor_b64 s[14:15], s[14:15], s[18:19]
	s_mul_i32 s24, s14, s20
	s_mul_hi_u32 s25, s14, s21
	s_mul_hi_u32 s23, s14, s20
	s_add_u32 s24, s25, s24
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 s26, s15, s21
	s_mul_i32 s21, s15, s21
	s_add_u32 s21, s24, s21
	s_mul_hi_u32 s25, s15, s20
	s_addc_u32 s21, s23, s26
	s_addc_u32 s23, s25, 0
	s_mul_i32 s20, s15, s20
	s_add_u32 s20, s21, s20
	s_addc_u32 s21, 0, s23
	s_add_u32 s23, s20, 1
	s_addc_u32 s24, s21, 0
	s_add_u32 s25, s20, 2
	s_mul_i32 s27, s22, s21
	s_mul_hi_u32 s28, s22, s20
	s_addc_u32 s26, s21, 0
	s_add_i32 s28, s28, s27
	s_mul_i32 s27, s22, s20
	s_sub_u32 s14, s14, s27
	s_subb_u32 s15, s15, s28
	s_sub_u32 s27, s14, s22
	s_subb_u32 s28, s15, 0
	s_cmp_ge_u32 s27, s22
	s_cselect_b32 s27, -1, 0
	s_cmp_eq_u32 s28, 0
	s_cselect_b32 s27, s27, -1
	s_cmp_lg_u32 s27, 0
	s_cselect_b32 s23, s25, s23
	s_cselect_b32 s24, s26, s24
	s_cmp_ge_u32 s14, s22
	s_cselect_b32 s14, -1, 0
	s_cmp_eq_u32 s15, 0
	s_cselect_b32 s14, s14, -1
	s_cmp_lg_u32 s14, 0
	s_cselect_b32 s15, s24, s21
	s_cselect_b32 s14, s23, s20
	s_xor_b64 s[14:15], s[14:15], s[18:19]
	s_sub_u32 s14, s14, s18
	s_subb_u32 s15, s15, s18
	s_xor_b64 s[2:3], s[14:15], s[2:3]
	s_and_b64 s[0:1], s[0:1], exec
	v_writelane_b32 v202, s73, 8
	s_cselect_b32 s1, s3, s77
	s_cselect_b32 s0, s2, s76
	v_writelane_b32 v202, s72, 6
	s_lshl_b64 s[2:3], s[0:1], 7
	v_writelane_b32 v202, s77, 31
	v_or_b32_e32 v33, s3, v45
	v_or_b32_e32 v32, s2, v44
	v_writelane_b32 v202, s76, 26
	v_or_b32_e32 v34, 32, v32
	v_mov_b32_e32 v35, v33
	v_cmp_gt_i64_e64 s[2:3], s[12:13], v[32:33]
	v_cmp_gt_i64_e64 s[26:27], s[12:13], v[34:35]
	v_or_b32_e32 v34, 64, v32
	v_or_b32_e32 v32, 0x60, v32
	v_readlane_b32 s14, v202, 25
	v_cmp_gt_i64_e64 s[20:21], s[12:13], v[32:33]
	v_and_b32_e32 v32, 7, v40
	s_lshl_b32 s24, s14, 7
	v_readlane_b32 s14, v202, 18
	v_cmp_gt_i64_e64 s[18:19], s[12:13], v[34:35]
	v_bitop3_b32 v34, v97, v32, 4 bitop3:0x36
	v_lshlrev_b32_e32 v32, 11, v97
	s_lshl_b32 s63, s14, 7
	v_readlane_b32 s14, v202, 20
	v_sub_u32_e32 v36, v42, v32
	s_lshl_b32 s64, s14, 7
	v_readlane_b32 s14, v202, 21
	v_bitop3_b32 v35, v97, v40, 7 bitop3:0x78
	v_mov_b32_e32 v33, 0
	s_lshl_b32 s55, s14, 7
	s_lshl_b64 s[14:15], s[0:1], 2
	v_lshl_add_u32 v65, v72, 14, v36
	s_mov_b32 s1, 0x8000
	v_mul_i32_i24_e32 v36, -16, v97
	v_add_u32_e32 v38, v41, v40
	v_lshl_add_u32 v66, v35, 4, s1
	v_lshl_add_u32 v67, v34, 4, s1
	v_ashrrev_i32_e32 v37, 31, v36
	s_lshl_b64 s[28:29], s[34:35], 5
	v_and_b32_e32 v34, 0x70, v38
	v_mov_b32_e32 v35, v33
	v_writelane_b32 v202, s34, 18
	v_lshl_add_u64 v[34:35], v[34:35], 0, v[36:37]
	v_lshl_add_u64 v[34:35], v[34:35], 0, s[28:29]
	v_writelane_b32 v202, s35, 19
	v_lshlrev_b32_e32 v35, 4, v40
	v_readlane_b32 s48, v202, 23
	s_lshl_b32 s1, s48, 7
	v_lshlrev_b32_e32 v39, 8, v97
	v_add3_u32 v48, s1, v35, v39
	v_mov_b32_e32 v39, v33
	v_lshrrev_b32_e32 v58, 4, v38
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[28:29]
	v_lshlrev_b32_e32 v45, 2, v40
	v_lshl_add_u64 v[54:55], v[38:39], 0, v[36:37]
	v_or_b32_e32 v62, 0xc800, v45
	v_lshl_add_u32 v56, s48, 5, v45
	v_ashrrev_i32_e32 v45, 31, v55
	s_lshl_b64 s[22:23], s[34:35], 9
	v_ashrrev_i32_e32 v35, 31, v39
	v_xor_b32_e32 v37, v45, v55
	v_xor_b32_e32 v46, v35, v38
	v_ashrrev_i32_e32 v38, 31, v37
	s_add_u32 s1, s46, 7
	v_xor_b32_e32 v47, v35, v39
	v_xor_b32_e32 v36, v45, v54
	v_lshrrev_b32_e32 v38, 27, v38
	v_mov_b32_e32 v39, s65
	s_addc_u32 s25, s47, 0
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[38:39]
	s_sub_u32 s34, -8, s46
	v_ashrrev_i64 v[36:37], 5, v[36:37]
	s_subb_u32 s35, -1, s47
	v_cmp_lt_i64_e64 s[28:29], s[46:47], -7
	v_xor_b32_e32 v68, v37, v45
	v_xor_b32_e32 v45, v36, v45
	v_cndmask_b32_e64 v36, 0, -1, s[28:29]
	s_and_b64 s[28:29], s[28:29], exec
	v_ashrrev_i32_e32 v50, 31, v47
	s_cselect_b32 s25, s35, s25
	v_lshrrev_b32_e32 v50, 28, v50
	v_mov_b32_e32 v51, s65
	s_cselect_b32 s1, s34, s1
	s_ashr_i32 s28, s25, 31
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[50:51]
	s_lshr_b32 s28, s28, 29
	v_lshrrev_b64 v[46:47], 4, v[46:47]
	s_add_u32 s28, s1, s28
	v_xor_b32_e32 v47, v47, v35
	v_xor_b32_e32 v46, v46, v35
	v_lshrrev_b32_e32 v35, 5, v40
	s_addc_u32 s29, s25, 0
	v_readfirstlane_b32 s34, v36
	v_mul_i32_i24_e32 v50, 0xfffffe00, v35
	s_ashr_i64 s[28:29], s[28:29], 3
	s_mov_b32 s35, s34
	v_mov_b32_e32 v49, v33
	v_ashrrev_i32_e32 v51, 31, v50
	s_xor_b64 s[28:29], s[28:29], s[34:35]
	s_lshl_b32 s1, s28, 8
	s_lshl_b64 s[66:67], s[28:29], 3
	v_lshl_add_u64 v[38:39], v[50:51], 0, v[48:49]
	s_lshl_b64 s[34:35], s[60:61], 11
	v_lshlrev_b64 v[36:37], 8, v[46:47]
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[22:23]
	v_lshlrev_b32_e32 v46, 9, v35
	v_mov_b32_e32 v47, v33
	s_add_u32 s22, s34, s22
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[46:47]
	s_addc_u32 s23, s35, s23
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[34:35]
	v_lshl_add_u64 v[46:47], v[46:47], 0, v[50:51]
	v_lshl_add_u64 v[50:51], s[22:23], 0, v[48:49]
	s_lshl_b32 s23, s53, 5
	s_mov_b32 s22, s65
	s_lshl_b64 s[34:35], s[70:71], 5
	s_or_b64 s[22:23], s[22:23], s[34:35]
	s_or_b32 s14, s14, s48
	s_mul_i32 s15, s34, s15
	s_mul_hi_u32 s22, s34, s14
	s_add_i32 s15, s22, s15
	s_mul_i32 s22, s23, s14
	s_add_i32 s22, s15, s22
	s_mul_i32 s23, s34, s14
	s_lshl_b64 s[14:15], s[60:61], 13
	s_add_u32 s14, s23, s14
	s_addc_u32 s15, s22, s15
	s_lshl_b64 s[52:53], s[60:61], 8
	v_mov_b32_e32 v43, v33
	v_mul_i32_i24_e32 v58, 0xffffffc0, v58
	v_sub_co_u32_e32 v38, vcc, 0, v38
	v_bfe_u32 v60, v54, 4, 1
	v_mov_b64_e32 v[54:55], s[52:53]
	v_lshlrev_b32_e32 v52, 6, v97
	v_mov_b32_e32 v53, v33
	v_ashrrev_i32_e32 v59, 31, v58
	v_subb_co_u32_e32 v39, vcc, 0, v39, vcc
	v_lshl_add_u64 v[42:43], s[14:15], 0, v[42:43]
	v_mad_u64_u32 v[54:55], s[14:15], s1, v45, v[54:55]
	v_mov_b32_e32 v57, v33
	v_sub_co_u32_e32 v48, vcc, v50, v36
	v_lshl_add_u64 v[58:59], v[52:53], 0, v[58:59]
	s_lshr_b64 s[14:15], s[28:29], 24
	v_lshl_add_u32 v44, s0, 7, v44
	v_bitop3_b32 v32, v40, v99, 7 bitop3:0x6c
	v_subb_co_u32_e32 v49, vcc, v51, v37, vcc
	v_lshl_add_u64 v[52:53], v[58:59], 0, v[56:57]
	v_mul_lo_u32 v35, s1, v68
	v_mul_lo_u32 v68, s14, v45
	v_mad_u64_u32 v[58:59], s[14:15], s1, v45, v[58:59]
	v_add_u32_e32 v45, 64, v44
	v_lshlrev_b32_e32 v63, 4, v32
	v_sub_co_u32_e32 v50, vcc, 0, v50
	v_or_b32_e32 v52, v52, v60
	v_add3_u32 v55, v68, v55, v35
	s_lshl_b32 s0, s60, 7
	v_mul_lo_u32 v45, v45, s44
	v_subb_co_u32_e32 v51, vcc, 0, v51, vcc
	v_lshl_add_u64 v[54:55], v[52:53], 0, v[54:55]
	v_add3_u32 v59, v68, v59, v35
	v_add_u32_e32 v35, 0x60, v44
	v_add3_u32 v75, v45, s0, v63
	v_add_u32_e32 v45, 32, v44
	v_lshlrev_b32_e32 v32, 5, v32
	v_lshlrev_b32_e32 v64, 10, v72
	v_mov_b32_e32 v61, v33
	v_sub_co_u32_e32 v54, vcc, -3, v54
	v_lshl_add_u64 v[56:57], v[58:59], 0, v[56:57]
	v_mul_lo_u32 v35, v35, s44
	v_mul_lo_u32 v45, v45, s44
	v_mul_lo_u32 v44, v44, s44
	s_ashr_i32 s68, s45, 31
	s_movk_i32 s74, 0xfbf4
	s_ashr_i32 s76, s67, 31
	s_movk_i32 s82, 0xfbf3
	s_movk_i32 s86, 0xfbf2
	s_movk_i32 s90, 0xfbf1
	s_movk_i32 s94, 0xfbf0
	s_movk_i32 s96, 0xf800
	s_movk_i32 s58, 0xff00
	v_subb_co_u32_e32 v55, vcc, -1, v55, vcc
	v_lshl_add_u64 v[56:57], v[56:57], 0, v[60:61]
	v_add3_u32 v35, v35, s0, v63
	v_add3_u32 v76, v45, s0, v63
	v_add3_u32 v77, v44, s0, v63
	s_add_i32 s47, s24, 0x8000
	s_add_i32 s63, s63, 0x8000
	s_add_i32 s64, s64, 0x8000
	s_add_i32 s55, s55, 0x8000
	v_add_u32_e32 v78, v62, v64
	v_add_u32_e32 v79, v66, v65
	v_add_u32_e32 v80, v67, v65
	s_mov_b32 s69, s68
	s_mov_b64 s[70:71], 0x400
	s_mov_b32 s75, -1
	s_mov_b32 s78, s76
	s_mov_b32 s79, s76
	v_mov_b64_e32 v[44:45], v[32:33]
	s_mov_b64 s[80:81], 0x40c
	s_mov_b32 s83, -1
	s_mov_b64 s[72:73], 0x40d
	s_mov_b32 s87, -1
	s_mov_b64 s[88:89], 0x40e
	s_mov_b32 s91, -1
	s_mov_b64 s[92:93], 0x40f
	s_mov_b32 s95, -1
	s_mov_b32 s97, -1
	s_mov_b64 s[56:57], 0x800
	s_mov_b64 s[48:49], 0x2000
	s_mov_b64 s[0:1], 0x100
	s_mov_b32 s59, -1
	s_branch .LBB0_112
	.p2align	5
.LBB0_111:
	s_or_b64 exec, exec, s[14:15]
	s_add_u32 s14, s66, s76
	s_addc_u32 s15, s67, s76
	s_xor_b64 s[24:25], s[14:15], s[78:79]
	v_cvt_f32_u32_e32 v32, s24
	v_cvt_f32_u32_e32 v69, s25
	v_ashrrev_i32_e32 v180, 31, v65
	v_xor_b32_e32 v181, v68, v180
	s_sub_u32 s14, 0, s24
	v_fmac_f32_e32 v32, 0x4f800000, v69
	v_rcp_f32_e32 v32, v32
	s_subb_u32 s15, 0, s25
	v_xor_b32_e32 v65, v180, v65
	v_xor_b32_e32 v64, v180, v64
	v_mul_f32_e32 v32, 0x5f7ffffc, v32
	v_mul_f32_e32 v68, 0x2f800000, v32
	v_trunc_f32_e32 v68, v68
	v_fmac_f32_e32 v32, 0xcf800000, v68
	v_cvt_u32_f32_e32 v32, v32
	v_cvt_u32_f32_e32 v68, v68
	v_ashrrev_i32_e32 v63, 31, v63
	v_and_b32_e32 v63, s66, v63
	v_readfirstlane_b32 s22, v32
	v_readfirstlane_b32 s29, v68
	s_mul_hi_u32 s28, s14, s22
	s_mul_i32 s34, s14, s29
	s_mul_i32 s23, s15, s22
	s_add_i32 s28, s28, s34
	s_add_i32 s28, s28, s23
	s_mul_i32 s35, s14, s22
	s_mul_i32 s34, s22, s28
	s_mul_hi_u32 s50, s22, s35
	s_mul_hi_u32 s23, s22, s28
	s_add_u32 s34, s50, s34
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 s51, s29, s35
	s_mul_i32 s35, s29, s35
	s_add_u32 s34, s34, s35
	s_mul_hi_u32 s50, s29, s28
	s_addc_u32 s23, s23, s51
	s_addc_u32 s34, s50, 0
	s_mul_i32 s28, s29, s28
	s_add_u32 s23, s23, s28
	s_addc_u32 s28, 0, s34
	s_add_u32 s22, s22, s23
	s_addc_u32 s23, s29, s28
	s_mul_i32 s28, s14, s23
	s_mul_hi_u32 s29, s14, s22
	s_add_i32 s28, s29, s28
	s_mul_i32 s15, s15, s22
	s_add_i32 s28, s28, s15
	s_mul_i32 s14, s14, s22
	s_mul_hi_u32 s29, s23, s14
	s_mul_i32 s34, s23, s14
	s_mul_i32 s50, s22, s28
	s_mul_hi_u32 s14, s22, s14
	s_mul_hi_u32 s35, s22, s28
	s_add_u32 s14, s14, s50
	s_addc_u32 s35, 0, s35
	s_add_u32 s14, s14, s34
	s_mul_hi_u32 s15, s23, s28
	s_addc_u32 s14, s35, s29
	s_addc_u32 s15, s15, 0
	s_mul_i32 s28, s23, s28
	s_add_u32 s14, s14, s28
	v_ashrrev_i32_e32 v68, 31, v65
	s_addc_u32 s15, 0, s15
	v_mov_b32_e32 v69, v68
	s_add_u32 s22, s22, s14
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[68:69]
	s_addc_u32 s23, s23, s15
	v_xor_b32_e32 v182, v64, v68
	v_xor_b32_e32 v69, v65, v68
	v_mad_u64_u32 v[64:65], s[14:15], v182, s23, 0
	v_mul_hi_u32 v32, v182, s22
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[114:115], s[14:15], v69, s22, 0
	v_add_co_u32_e32 v32, vcc, v64, v114
	v_mad_u64_u32 v[70:71], s[14:15], v69, s23, 0
	s_nop 0
	v_addc_co_u32_e32 v64, vcc, v65, v115, vcc
	v_mov_b32_e32 v65, s65
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[70:71]
	v_mul_lo_u32 v32, s25, v64
	v_mul_lo_u32 v65, s24, v65
	v_mad_u64_u32 v[70:71], s[14:15], s24, v64, 0
	v_add3_u32 v32, v71, v65, v32
	v_sub_u32_e32 v65, v69, v32
	v_mov_b32_e32 v71, s25
	v_sub_co_u32_e32 v70, vcc, v182, v70
	s_add_u32 s60, s60, 1
	s_nop 0
	v_subb_co_u32_e64 v65, s[14:15], v65, v71, vcc
	v_subrev_co_u32_e64 v71, s[14:15], s24, v70
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s25, v65
	v_cmp_le_u32_e32 vcc, s25, v32
	s_addc_u32 s61, s61, 0
	v_cndmask_b32_e64 v114, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s24, v71
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s24, v70
	v_cndmask_b32_e64 v71, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s25, v65
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s25, v32
	v_cndmask_b32_e64 v65, v114, v71, s[14:15]
	v_add_u32_e32 v71, 2, v64
	v_add_u32_e32 v114, 1, v64
	v_cmp_ne_u32_e64 s[14:15], 0, v65
	v_cndmask_b32_e32 v32, v69, v70, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_cndmask_b32_e64 v65, v114, v71, s[14:15]
	s_waitcnt vmcnt(12)
	v_lshlrev_b16_e32 v71, 8, v101
	v_cndmask_b32_e32 v32, v64, v65, vcc
	v_xor_b32_e32 v64, s76, v68
	v_xor_b32_e32 v32, v32, v64
	v_sub_u32_e32 v32, v32, v64
	v_xor_b32_e32 v32, v32, v180
	v_mul_lo_u32 v32, v32, s54
	v_add3_u32 v32, v32, v62, v63
	buffer_load_ubyte v32, v32, s[8:11], 0 offen
	v_ashrrev_i32_e32 v63, 31, v67
	v_mul_lo_u32 v62, v181, s54
	v_and_b32_e32 v63, s66, v63
	v_add3_u32 v62, v62, v66, v63
	buffer_load_ubyte v70, v62, s[8:11], 0 offen
	v_lshlrev_b16_e32 v62, 8, v94
	v_lshlrev_b16_e32 v63, 8, v96
	v_or_b32_e32 v62, v93, v62
	v_or_b32_sdwa v63, v95, v63 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshlrev_b16_e32 v66, 8, v84
	v_or_b32_sdwa v65, v62, v63 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v62, 8, v90
	v_lshlrev_b16_e32 v63, 8, v92
	v_or_b32_e32 v62, v89, v62
	v_or_b32_sdwa v63, v91, v63 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v66, v83, v66 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_or_b32_sdwa v64, v62, v63 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v62, 8, v86
	v_lshlrev_b16_e32 v63, 8, v88
	v_or_b32_e32 v62, v85, v62
	v_or_b32_sdwa v63, v87, v63 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt vmcnt(2)
	v_lshlrev_b16_e32 v67, 8, v113
	v_or_b32_sdwa v63, v62, v63 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v62, 8, v82
	v_or_b32_e32 v62, v81, v62
	v_or_b32_sdwa v62, v62, v66 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v66, 8, v111
	v_or_b32_e32 v66, v110, v66
	v_or_b32_sdwa v67, v112, v67 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt vmcnt(1) lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[176:179], v[62:65], v[28:31], v60, v32 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_or_b32_sdwa v69, v66, v67 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v66, 8, v107
	v_lshlrev_b16_e32 v67, 8, v109
	v_or_b32_e32 v66, v106, v66
	v_or_b32_sdwa v67, v108, v67 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt lgkmcnt(13)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[120:123], v[62:65], v[24:27], v60, v32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_or_b32_sdwa v68, v66, v67 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v66, 8, v103
	v_lshlrev_b16_e32 v67, 8, v105
	v_or_b32_e32 v66, v102, v66
	v_or_b32_sdwa v67, v104, v67 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[128:131], v[62:65], v[20:23], v61, v32 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_or_b32_sdwa v67, v66, v67 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshlrev_b16_e32 v66, 8, v99
	v_or_b32_e32 v66, v98, v66
	s_waitcnt lgkmcnt(9)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[136:139], v[62:65], v[16:19], v61, v32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_or_b32_sdwa v71, v100, v71 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
	v_lshl_add_u64 v[38:39], v[38:39], 0, s[96:97]
	v_or_b32_sdwa v66, v66, v71 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[144:147], v[62:65], v[12:15], v58, v32 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshl_add_u64 v[48:49], v[48:49], 0, s[56:57]
	v_lshl_add_u64 v[50:51], v[50:51], 0, s[96:97]
	v_lshl_add_u64 v[42:43], v[42:43], 0, s[48:49]
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[152:155], v[62:65], v[8:11], v58, v32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshl_add_u64 v[52:53], v[52:53], 0, s[0:1]
	v_lshl_add_u64 v[54:55], v[54:55], 0, s[58:59]
	v_lshl_add_u64 v[56:57], v[56:57], 0, s[0:1]
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[160:163], v[62:65], v[4:7], v59, v32 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshl_add_u64 v[44:45], v[44:45], 0, s[0:1]
	v_add_u32_e32 v35, 0x80, v35
	v_add_u32_e32 v75, 0x80, v75
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[168:171], v[62:65], v[0:3], v59, v32 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v76, 0x80, v76
	v_add_u32_e32 v77, 0x80, v77
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[116:119], v[66:69], v[28:31], v60, v70 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[124:127], v[66:69], v[24:27], v60, v70 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[132:135], v[66:69], v[20:23], v61, v70 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[140:143], v[66:69], v[16:19], v61, v70 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[148:151], v[66:69], v[12:15], v58, v70 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[156:159], v[66:69], v[8:11], v58, v70 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[164:167], v[66:69], v[4:7], v59, v70 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[172:175], v[66:69], v[0:3], v59, v70 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mov_b64_e32 v[58:59], s[84:85]
	v_cmp_lt_i64_e32 vcc, s[60:61], v[58:59]
	s_cbranch_vccz .LBB0_128
.LBB0_112:
	v_lshl_add_u64 v[58:59], s[52:53], 0, v[44:45]
	v_cmp_gt_i64_e32 vcc, s[30:31], v[58:59]
	s_and_b64 s[14:15], s[2:3], vcc
	s_mov_b32 m0, s47
	v_cndmask_b32_e64 v32, -1, v77, s[14:15]
	s_and_b64 s[14:15], s[26:27], vcc
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	buffer_load_dwordx4 v32, s[40:43], 0 offen lds
	v_cndmask_b32_e64 v32, -1, v76, s[14:15]
	s_mov_b32 m0, s63
	s_and_b64 s[14:15], s[18:19], vcc
	buffer_load_dwordx4 v32, s[40:43], 0 offen lds
	v_cndmask_b32_e64 v32, -1, v75, s[14:15]
	s_mov_b32 m0, s64
	s_and_b64 vcc, s[20:21], vcc
	buffer_load_dwordx4 v32, s[40:43], 0 offen lds
	v_cndmask_b32_e32 v32, -1, v35, vcc
	s_mov_b32 m0, s55
	v_ashrrev_i32_e32 v58, 31, v43
	buffer_load_dwordx4 v32, s[40:43], 0 offen lds
	v_or_b32_e32 v32, s31, v43
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[28:29], exec, s[14:15]
	s_cbranch_execz .LBB0_114
	s_ashr_i32 s14, s31, 31
	s_add_u32 s22, s30, s14
	s_mov_b32 s15, s14
	s_addc_u32 s23, s31, s14
	s_xor_b64 s[34:35], s[22:23], s[14:15]
	v_cvt_f32_u32_e32 v32, s34
	v_cvt_f32_u32_e32 v59, s35
	s_sub_u32 s14, 0, s34
	s_subb_u32 s15, 0, s35
	v_fmac_f32_e32 v32, 0x4f800000, v59
	v_rcp_f32_e32 v32, v32
	s_nop 0
	v_mul_f32_e32 v32, 0x5f7ffffc, v32
	v_mul_f32_e32 v59, 0x2f800000, v32
	v_trunc_f32_e32 v59, v59
	v_fmac_f32_e32 v32, 0xcf800000, v59
	v_cvt_u32_f32_e32 v59, v59
	v_cvt_u32_f32_e32 v32, v32
	v_readfirstlane_b32 s22, v59
	v_readfirstlane_b32 s23, v32
	s_mul_i32 s24, s14, s22
	s_mul_hi_u32 s77, s14, s23
	s_mul_i32 s25, s15, s23
	s_add_i32 s24, s77, s24
	s_add_i32 s24, s24, s25
	s_mul_i32 vcc_lo, s14, s23
	s_mul_i32 s77, s23, s24
	s_mul_hi_u32 vcc_hi, s23, vcc_lo
	s_mul_hi_u32 s25, s23, s24
	s_add_u32 s77, vcc_hi, s77
	s_addc_u32 s25, 0, s25
	s_mul_i32 s51, s22, vcc_lo
	s_mul_hi_u32 s50, s22, vcc_lo
	s_add_u32 s51, s77, s51
	s_mul_hi_u32 vcc_hi, s22, s24
	s_addc_u32 s25, s25, s50
	s_addc_u32 s50, vcc_hi, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s24, s25, s24
	s_addc_u32 s25, 0, s50
	s_add_u32 s23, s23, s24
	s_addc_u32 s22, s22, s25
	s_mul_i32 s24, s14, s22
	s_mul_hi_u32 s25, s14, s23
	s_add_i32 s24, s25, s24
	s_mul_i32 s15, s15, s23
	s_add_i32 s24, s24, s15
	s_mul_i32 s14, s14, s23
	s_mul_hi_u32 s25, s22, s14
	s_mul_i32 s50, s22, s14
	s_mul_i32 s77, s23, s24
	s_mul_hi_u32 s14, s23, s14
	s_mul_hi_u32 s51, s23, s24
	s_add_u32 s14, s14, s77
	s_addc_u32 s51, 0, s51
	s_add_u32 s14, s14, s50
	s_mul_hi_u32 s15, s22, s24
	s_addc_u32 s14, s51, s25
	s_addc_u32 s15, s15, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s14, s14, s24
	s_addc_u32 s15, 0, s15
	v_mov_b32_e32 v59, v58
	s_add_u32 s23, s23, s14
	v_lshl_add_u64 v[60:61], v[42:43], 0, v[58:59]
	s_addc_u32 s22, s22, s15
	v_xor_b32_e32 v66, v60, v58
	v_xor_b32_e32 v59, v61, v58
	v_mad_u64_u32 v[60:61], s[14:15], v66, s22, 0
	v_mul_hi_u32 v32, v66, s23
	v_lshl_add_u64 v[60:61], v[32:33], 0, v[60:61]
	v_mad_u64_u32 v[64:65], s[14:15], v59, s23, 0
	v_add_co_u32_e32 v32, vcc, v60, v64
	v_mad_u64_u32 v[62:63], s[14:15], v59, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v60, vcc, v61, v65, vcc
	v_mov_b32_e32 v61, s65
	s_nop 0
	v_addc_co_u32_e32 v63, vcc, 0, v63, vcc
	v_lshl_add_u64 v[60:61], v[60:61], 0, v[62:63]
	v_mul_lo_u32 v32, s35, v60
	v_mul_lo_u32 v62, s34, v61
	v_mad_u64_u32 v[60:61], s[14:15], s34, v60, 0
	v_add3_u32 v32, v61, v62, v32
	v_sub_u32_e32 v61, v59, v32
	v_mov_b32_e32 v62, s35
	v_sub_co_u32_e32 v60, vcc, v66, v60
	s_nop 1
	v_subb_co_u32_e64 v61, s[14:15], v61, v62, vcc
	v_subrev_co_u32_e64 v63, s[14:15], s34, v60
	v_subb_co_u32_e32 v32, vcc, v59, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v64, s[24:25], 0, v61, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v64
	v_subb_co_u32_e64 v61, s[14:15], v61, v62, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s34, v63
	v_subrev_co_u32_e64 v62, s[14:15], s34, v63
	s_nop 0
	v_cndmask_b32_e64 v66, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v64
	v_subbrev_co_u32_e64 v61, s[14:15], 0, v61, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v65, v65, v66, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cmp_ne_u32_e64 s[14:15], 0, v65
	s_nop 0
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v60
	v_cndmask_b32_e64 v61, v64, v61, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v64, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 1
	v_cndmask_b32_e32 v59, v59, v64, vcc
	v_cmp_ne_u32_e32 vcc, 0, v59
	v_cndmask_b32_e64 v59, v63, v62, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v59, v60, v59, vcc
	v_cndmask_b32_e32 v32, v32, v61, vcc
	v_xor_b32_e32 v59, v59, v58
	v_xor_b32_e32 v32, v32, v58
	v_sub_co_u32_e32 v60, vcc, v59, v58
	s_nop 1
	v_subb_co_u32_e32 v61, vcc, v32, v58, vcc
.LBB0_114:
	s_andn2_saveexec_b64 s[14:15], s[28:29]
	s_cbranch_execz .LBB0_116
	v_cvt_f32_u32_e32 v32, s30
	s_sub_i32 s22, 0, s30
	v_rcp_iflag_f32_e32 v32, v32
	s_nop 0
	v_mul_f32_e32 v32, 0x4f7ffffe, v32
	v_cvt_u32_f32_e32 v32, v32
	v_mul_lo_u32 v59, s22, v32
	v_mul_hi_u32 v59, v32, v59
	v_add_u32_e32 v32, v32, v59
	v_mul_hi_u32 v32, v42, v32
	v_mul_lo_u32 v32, v32, s30
	v_sub_u32_e32 v32, v42, v32
	v_subrev_u32_e32 v59, s30, v32
	v_cmp_le_u32_e32 vcc, s30, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v59, vcc
	v_subrev_u32_e32 v59, s30, v32
	v_cmp_le_u32_e32 vcc, s30, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v59, vcc
	v_mov_b64_e32 v[60:61], v[32:33]
.LBB0_116:
	s_or_b64 exec, exec, s[14:15]
	s_ashr_i32 s28, s31, 31
	s_add_u32 s14, s30, s28
	s_mov_b32 s29, s28
	s_addc_u32 s15, s31, s28
	s_xor_b64 s[34:35], s[14:15], s[28:29]
	v_cvt_f32_u32_e32 v32, s34
	v_cvt_f32_u32_e32 v59, s35
	s_sub_u32 s14, 0, s34
	s_subb_u32 s15, 0, s35
	v_xor_b32_e32 v63, v58, v43
	v_fmac_f32_e32 v32, 0x4f800000, v59
	v_rcp_f32_e32 v32, v32
	v_ashrrev_i32_e32 v64, 31, v63
	v_xor_b32_e32 v62, v58, v42
	v_mov_b32_e32 v65, v64
	v_mul_f32_e32 v32, 0x5f7ffffc, v32
	v_mul_f32_e32 v59, 0x2f800000, v32
	v_trunc_f32_e32 v59, v59
	v_fmac_f32_e32 v32, 0xcf800000, v59
	v_cvt_u32_f32_e32 v32, v32
	v_lshl_add_u64 v[62:63], v[62:63], 0, v[64:65]
	v_xor_b32_e32 v65, v62, v64
	v_readfirstlane_b32 s22, v32
	v_cvt_u32_f32_e32 v32, v59
	s_mul_hi_u32 s24, s14, s22
	s_mul_i32 s23, s15, s22
	s_mul_i32 s50, s14, s22
	v_readfirstlane_b32 s25, v32
	s_mul_i32 s29, s14, s25
	s_add_i32 s24, s24, s29
	s_add_i32 s24, s24, s23
	s_mul_i32 s29, s22, s24
	s_mul_hi_u32 s51, s22, s50
	s_mul_hi_u32 s23, s22, s24
	s_add_u32 s29, s51, s29
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 s77, s25, s50
	s_mul_i32 s50, s25, s50
	s_add_u32 s29, s29, s50
	s_mul_hi_u32 s51, s25, s24
	s_addc_u32 s23, s23, s77
	s_addc_u32 s29, s51, 0
	s_mul_i32 s24, s25, s24
	s_add_u32 s23, s23, s24
	s_addc_u32 s24, 0, s29
	s_add_u32 s22, s22, s23
	s_addc_u32 s23, s25, s24
	s_mul_i32 s24, s14, s23
	s_mul_hi_u32 s25, s14, s22
	s_add_i32 s24, s25, s24
	s_mul_i32 s15, s15, s22
	s_add_i32 s24, s24, s15
	s_mul_i32 s14, s14, s22
	s_mul_hi_u32 s25, s23, s14
	s_mul_i32 s29, s23, s14
	s_mul_i32 s51, s22, s24
	s_mul_hi_u32 s14, s22, s14
	s_mul_hi_u32 s50, s22, s24
	s_add_u32 s14, s14, s51
	s_addc_u32 s50, 0, s50
	s_add_u32 s14, s14, s29
	s_mul_hi_u32 s15, s23, s24
	s_addc_u32 s14, s50, s25
	s_addc_u32 s15, s15, 0
	s_mul_i32 s24, s23, s24
	s_add_u32 s14, s14, s24
	s_addc_u32 s15, 0, s15
	s_add_u32 s22, s22, s14
	s_addc_u32 s23, s23, s15
	v_xor_b32_e32 v59, v63, v64
	v_mad_u64_u32 v[62:63], s[14:15], v65, s23, 0
	v_mul_hi_u32 v32, v65, s22
	v_lshl_add_u64 v[62:63], v[32:33], 0, v[62:63]
	v_mad_u64_u32 v[68:69], s[14:15], v59, s22, 0
	v_add_co_u32_e32 v32, vcc, v62, v68
	v_mad_u64_u32 v[66:67], s[14:15], v59, s23, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v63, v69, vcc
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v67, vcc
	v_lshl_add_u64 v[62:63], v[32:33], 0, v[66:67]
	v_mul_lo_u32 v32, s35, v62
	v_mul_lo_u32 v63, s34, v63
	v_mad_u64_u32 v[66:67], s[14:15], s34, v62, 0
	v_add3_u32 v32, v67, v63, v32
	v_sub_u32_e32 v63, v59, v32
	v_sub_co_u32_e32 v65, vcc, v65, v66
	v_mov_b32_e32 v66, s35
	s_nop 0
	v_subb_co_u32_e64 v63, s[14:15], v63, v66, vcc
	v_subrev_co_u32_e64 v66, s[14:15], s34, v65
	v_cmp_le_u32_e64 s[24:25], s34, v66
	s_nop 0
	v_subbrev_co_u32_e64 v63, s[14:15], 0, v63, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v63
	v_add_u32_e32 v66, 1, v62
	v_subb_co_u32_e32 v32, vcc, v59, v32, vcc
	v_cndmask_b32_e64 v67, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v63
	v_cndmask_b32_e64 v63, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v63, v67, v63, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v63
	v_add_u32_e32 v63, 2, v62
	v_cndmask_b32_e64 v59, 0, -1, vcc
	v_cndmask_b32_e64 v63, v66, v63, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v65
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v59, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_xor_b32_e32 v59, s28, v64
	v_readlane_b32 s14, v202, 22
	v_cndmask_b32_e32 v32, v62, v63, vcc
	v_xor_b32_e32 v32, v32, v59
	v_sub_u32_e32 v32, v32, v59
	v_xor_b32_e32 v62, v32, v58
	v_ashrrev_i32_e32 v32, 31, v61
	v_and_b32_e32 v59, s31, v32
	v_and_b32_e32 v58, s30, v32
	v_lshl_add_u64 v[58:59], v[58:59], 0, v[60:61]
	v_ashrrev_i32_e32 v60, 31, v59
	v_xor_b32_e32 v59, v60, v59
	v_ashrrev_i32_e32 v32, 31, v59
	v_xor_b32_e32 v58, v60, v58
	v_lshrrev_b32_e32 v32, 27, v32
	v_lshl_add_u64 v[58:59], v[58:59], 0, v[32:33]
	v_alignbit_b32 v32, v59, v58, 5
	v_mul_lo_u32 v58, v62, s46
	v_ashrrev_i32_e32 v62, 31, v49
	s_add_i32 m0, s14, 0xc800
	v_xor_b32_e32 v67, v62, v49
	v_ashrrev_i32_e32 v64, 31, v67
	s_add_u32 s14, s44, s68
	v_xor_b32_e32 v66, v62, v48
	v_mov_b32_e32 v65, v64
	s_addc_u32 s15, s45, s68
	v_xad_u32 v32, v32, v60, v58
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[64:65]
	s_xor_b64 s[34:35], s[14:15], s[68:69]
	buffer_load_dword v32, s[4:7], 0 offen lds
	v_xor_b32_e32 v81, v66, v64
	v_cvt_f32_u32_e32 v32, s34
	v_cvt_f32_u32_e32 v66, s35
	s_sub_u32 s14, 0, s34
	s_subb_u32 s15, 0, s35
	v_xor_b32_e32 v65, v67, v64
	v_fmac_f32_e32 v32, 0x4f800000, v66
	v_rcp_f32_e32 v32, v32
	v_xor_b32_e32 v64, s68, v64
	v_mov_b32_e32 v63, v62
	s_waitcnt vmcnt(0)
	v_mul_f32_e32 v32, 0x5f7ffffc, v32
	v_mul_f32_e32 v66, 0x2f800000, v32
	v_trunc_f32_e32 v66, v66
	v_fmac_f32_e32 v32, 0xcf800000, v66
	v_cvt_u32_f32_e32 v32, v32
	s_barrier
	v_readfirstlane_b32 s22, v32
	v_cvt_u32_f32_e32 v32, v66
	s_mul_hi_u32 s24, s14, s22
	s_mul_i32 s23, s15, s22
	s_mul_i32 s29, s14, s22
	v_readfirstlane_b32 s25, v32
	s_mul_i32 s28, s14, s25
	s_add_i32 s24, s24, s28
	s_add_i32 s24, s24, s23
	s_mul_i32 s28, s22, s24
	s_mul_hi_u32 s50, s22, s29
	s_mul_hi_u32 s23, s22, s24
	s_add_u32 s28, s50, s28
	s_addc_u32 s23, 0, s23
	s_mul_hi_u32 s51, s25, s29
	s_mul_i32 s29, s25, s29
	s_add_u32 s28, s28, s29
	s_mul_hi_u32 s50, s25, s24
	s_addc_u32 s23, s23, s51
	s_addc_u32 s28, s50, 0
	s_mul_i32 s24, s25, s24
	s_add_u32 s23, s23, s24
	s_addc_u32 s24, 0, s28
	s_add_u32 s22, s22, s23
	s_addc_u32 s23, s25, s24
	s_mul_i32 s24, s14, s23
	s_mul_hi_u32 s25, s14, s22
	s_add_i32 s24, s25, s24
	s_mul_i32 s15, s15, s22
	s_add_i32 s24, s24, s15
	s_mul_i32 s14, s14, s22
	s_mul_hi_u32 s25, s23, s14
	s_mul_i32 s28, s23, s14
	s_mul_i32 s50, s22, s24
	s_mul_hi_u32 s14, s22, s14
	s_mul_hi_u32 s29, s22, s24
	s_add_u32 s14, s14, s50
	s_addc_u32 s29, 0, s29
	s_add_u32 s14, s14, s28
	s_mul_hi_u32 s15, s23, s24
	s_addc_u32 s14, s29, s25
	s_addc_u32 s15, s15, 0
	s_mul_i32 s24, s23, s24
	s_add_u32 s14, s14, s24
	s_addc_u32 s15, 0, s15
	s_add_u32 s77, s22, s14
	s_addc_u32 s22, s23, s15
	v_mad_u64_u32 v[66:67], s[14:15], v81, s22, 0
	v_mul_hi_u32 v32, v81, s77
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[70:71], s[14:15], v65, s77, 0
	v_add_co_u32_e32 v32, vcc, v66, v70
	v_mad_u64_u32 v[68:69], s[14:15], v65, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v67, v71, vcc
	v_mov_b32_e32 v70, s35
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[68:69]
	v_mul_lo_u32 v32, s35, v66
	v_mul_lo_u32 v67, s34, v67
	v_mad_u64_u32 v[68:69], s[14:15], s34, v66, 0
	v_add3_u32 v32, v69, v67, v32
	v_sub_u32_e32 v67, v65, v32
	v_sub_co_u32_e32 v68, vcc, v81, v68
	ds_read2st64_b32 v[60:61], v78 offset1:1
	ds_read2st64_b32 v[58:59], v78 offset0:2 offset1:3
	ds_read_b128 v[176:179], v79
	ds_read_b128 v[116:119], v80
	ds_read_b128 v[120:123], v79 offset:2048
	ds_read_b128 v[124:127], v80 offset:2048
	ds_read_b128 v[128:131], v79 offset:4096
	ds_read_b128 v[132:135], v80 offset:4096
	ds_read_b128 v[136:139], v79 offset:6144
	ds_read_b128 v[140:143], v80 offset:6144
	ds_read_b128 v[144:147], v79 offset:8192
	ds_read_b128 v[148:151], v80 offset:8192
	ds_read_b128 v[152:155], v79 offset:10240
	ds_read_b128 v[156:159], v80 offset:10240
	ds_read_b128 v[160:163], v79 offset:12288
	ds_read_b128 v[164:167], v80 offset:12288
	ds_read_b128 v[168:171], v79 offset:14336
	ds_read_b128 v[172:175], v80 offset:14336
	v_subb_co_u32_e64 v67, s[14:15], v67, v70, vcc
	v_subrev_co_u32_e64 v69, s[14:15], s34, v68
	v_cmp_le_u32_e64 s[24:25], s34, v69
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v67
	v_add_u32_e32 v69, 1, v66
	v_subb_co_u32_e32 v32, vcc, v65, v32, vcc
	v_cndmask_b32_e64 v71, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v67
	v_cndmask_b32_e64 v67, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v67, v71, v67, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v67
	v_add_u32_e32 v67, 2, v66
	v_cndmask_b32_e64 v65, 0, -1, vcc
	v_cndmask_b32_e64 v67, v69, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v68
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v65, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v66, v67, vcc
	v_xor_b32_e32 v32, v32, v64
	v_sub_u32_e32 v32, v32, v64
	v_lshl_add_u64 v[64:65], v[48:49], 0, v[62:63]
	v_xor_b32_e32 v81, v64, v62
	v_xad_u32 v71, v32, v62, v34
	v_xor_b32_e32 v63, v65, v62
	v_mad_u64_u32 v[64:65], s[14:15], v81, s22, 0
	v_mul_hi_u32 v32, v81, s77
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[68:69], s[14:15], v63, s77, 0
	v_add_co_u32_e32 v32, vcc, v64, v68
	v_mad_u64_u32 v[66:67], s[14:15], v63, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v65, v69, vcc
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v67, vcc
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[66:67]
	v_mul_lo_u32 v32, s35, v64
	v_mul_lo_u32 v66, s34, v65
	v_mad_u64_u32 v[64:65], s[14:15], s34, v64, 0
	v_add3_u32 v32, v65, v66, v32
	v_sub_u32_e32 v65, v63, v32
	v_sub_co_u32_e32 v64, vcc, v81, v64
	s_nop 1
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, vcc
	v_subrev_co_u32_e64 v66, s[14:15], s34, v64
	v_cmp_le_u32_e64 s[28:29], s34, v66
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[24:25], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v67
	v_cndmask_b32_e64 v69, 0, -1, s[28:29]
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, s[14:15]
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v67
	v_subb_co_u32_e32 v32, vcc, v63, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v68, v68, v69, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v68
	v_subrev_co_u32_e64 v68, s[14:15], s34, v66
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v64
	v_cndmask_b32_e64 v65, v67, v65, s[24:25]
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v67, 0, -1, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v63, v63, v67, vcc
	v_cmp_ne_u32_e32 vcc, 0, v63
	v_cndmask_b32_e64 v63, v66, v68, s[24:25]
	s_nop 0
	v_cndmask_b32_e32 v63, v64, v63, vcc
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_xor_b32_e32 v63, v63, v62
	v_xor_b32_e32 v32, v32, v62
	v_sub_co_u32_e32 v63, vcc, v63, v62
	v_lshl_add_u64 v[64:65], v[48:49], 0, 1
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v62, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_mul_lo_u32 v62, v71, s33
	v_add3_u32 v32, v32, v63, v62
	v_lshl_add_u64 v[62:63], v[36:37], 0, v[50:51]
	v_cmp_gt_i64_e32 vcc, -1, v[48:49]
	v_lshl_add_u64 v[66:67], v[62:63], 0, -2
	buffer_load_ubyte v81, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v67, v65, v67, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v66, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v71, v66, v68
	v_xor_b32_e32 v69, v67, v68
	v_mad_u64_u32 v[66:67], s[14:15], v71, s22, 0
	v_mul_hi_u32 v32, v71, s77
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[84:85], s[14:15], v69, s77, 0
	v_add_co_u32_e32 v32, vcc, v66, v84
	v_mad_u64_u32 v[82:83], s[14:15], v69, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v67, v85, vcc
	s_nop 1
	v_addc_co_u32_e32 v83, vcc, 0, v83, vcc
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[82:83]
	v_mul_lo_u32 v32, s35, v66
	v_mul_lo_u32 v67, s34, v67
	v_mad_u64_u32 v[82:83], s[14:15], s34, v66, 0
	v_add3_u32 v32, v83, v67, v32
	v_sub_u32_e32 v67, v69, v32
	v_sub_co_u32_e32 v71, vcc, v71, v82
	s_nop 1
	v_subb_co_u32_e64 v67, s[14:15], v67, v70, vcc
	v_subrev_co_u32_e64 v82, s[14:15], s34, v71
	v_cmp_le_u32_e64 s[24:25], s34, v82
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v67
	v_add_u32_e32 v82, 1, v66
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	v_cndmask_b32_e64 v83, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v67
	v_cndmask_b32_e64 v67, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v67, v83, v67, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v67
	v_add_u32_e32 v67, 2, v66
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cndmask_b32_e64 v67, v82, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v71
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v69, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v66, v67, vcc
	v_xor_b32_e32 v66, s68, v68
	v_xor_b32_e32 v32, v32, v66
	v_sub_u32_e32 v32, v32, v66
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_xor_b32_e32 v84, v64, v66
	v_xad_u32 v71, v32, v66, v34
	v_xor_b32_e32 v67, v65, v66
	v_mad_u64_u32 v[64:65], s[14:15], v84, s22, 0
	v_mul_hi_u32 v32, v84, s77
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[82:83], s[14:15], v67, s77, 0
	v_add_co_u32_e32 v32, vcc, v64, v82
	v_mad_u64_u32 v[68:69], s[14:15], v67, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v65, v83, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[68:69]
	v_mul_lo_u32 v32, s35, v64
	v_mul_lo_u32 v68, s34, v65
	v_mad_u64_u32 v[64:65], s[14:15], s34, v64, 0
	v_add3_u32 v32, v65, v68, v32
	v_sub_u32_e32 v65, v67, v32
	v_sub_co_u32_e32 v64, vcc, v84, v64
	s_nop 1
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, vcc
	v_subrev_co_u32_e64 v68, s[14:15], s34, v64
	v_cmp_le_u32_e64 s[28:29], s34, v68
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[24:25], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v69
	v_cndmask_b32_e64 v83, 0, -1, s[28:29]
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, s[14:15]
	v_cndmask_b32_e64 v82, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v69
	v_subb_co_u32_e32 v32, vcc, v67, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v82, v82, v83, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v82
	v_subrev_co_u32_e64 v82, s[14:15], s34, v68
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v64
	v_cndmask_b32_e64 v65, v69, v65, s[24:25]
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v67, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_cndmask_b32_e64 v65, v68, v82, s[24:25]
	v_cndmask_b32_e32 v64, v64, v65, vcc
	v_xor_b32_e32 v64, v64, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v64, vcc, v64, v66
	v_mul_lo_u32 v65, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v64, v65
	v_lshl_add_u64 v[64:65], v[48:49], 0, 2
	v_cmp_gt_i64_e32 vcc, -2, v[48:49]
	v_lshl_add_u64 v[66:67], v[62:63], 0, -3
	buffer_load_ubyte v82, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v67, v65, v67, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v66, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v71, v66, v68
	v_xor_b32_e32 v69, v67, v68
	v_mad_u64_u32 v[66:67], s[14:15], v71, s22, 0
	v_mul_hi_u32 v32, v71, s77
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[86:87], s[14:15], v69, s77, 0
	v_add_co_u32_e32 v32, vcc, v66, v86
	v_mad_u64_u32 v[84:85], s[14:15], v69, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v67, v87, vcc
	s_nop 1
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[84:85]
	v_mul_lo_u32 v32, s35, v66
	v_mul_lo_u32 v67, s34, v67
	v_mad_u64_u32 v[84:85], s[14:15], s34, v66, 0
	v_add3_u32 v32, v85, v67, v32
	v_sub_u32_e32 v67, v69, v32
	v_sub_co_u32_e32 v71, vcc, v71, v84
	s_nop 1
	v_subb_co_u32_e64 v67, s[14:15], v67, v70, vcc
	v_subrev_co_u32_e64 v83, s[14:15], s34, v71
	v_cmp_le_u32_e64 s[24:25], s34, v83
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v67
	v_add_u32_e32 v83, 1, v66
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	v_cndmask_b32_e64 v84, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v67
	v_cndmask_b32_e64 v67, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v67, v84, v67, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v67
	v_add_u32_e32 v67, 2, v66
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cndmask_b32_e64 v67, v83, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v71
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v69, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v66, v67, vcc
	v_xor_b32_e32 v66, s68, v68
	v_xor_b32_e32 v32, v32, v66
	v_sub_u32_e32 v32, v32, v66
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_xor_b32_e32 v83, v64, v66
	v_xad_u32 v71, v32, v66, v34
	v_xor_b32_e32 v67, v65, v66
	v_mad_u64_u32 v[64:65], s[14:15], v83, s22, 0
	v_mul_hi_u32 v32, v83, s77
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[84:85], s[14:15], v67, s77, 0
	v_add_co_u32_e32 v32, vcc, v64, v84
	v_mad_u64_u32 v[68:69], s[14:15], v67, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v65, v85, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[68:69]
	v_mul_lo_u32 v32, s35, v64
	v_mul_lo_u32 v68, s34, v65
	v_mad_u64_u32 v[64:65], s[14:15], s34, v64, 0
	v_add3_u32 v32, v65, v68, v32
	v_sub_u32_e32 v65, v67, v32
	v_sub_co_u32_e32 v64, vcc, v83, v64
	s_nop 1
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, vcc
	v_subrev_co_u32_e64 v68, s[14:15], s34, v64
	v_cmp_le_u32_e64 s[28:29], s34, v68
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[24:25], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v69
	v_cndmask_b32_e64 v84, 0, -1, s[28:29]
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, s[14:15]
	v_cndmask_b32_e64 v83, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v69
	v_subb_co_u32_e32 v32, vcc, v67, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v83, v83, v84, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v83
	v_subrev_co_u32_e64 v83, s[14:15], s34, v68
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v64
	v_cndmask_b32_e64 v65, v69, v65, s[24:25]
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v67, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_cndmask_b32_e64 v65, v68, v83, s[24:25]
	v_cndmask_b32_e32 v64, v64, v65, vcc
	v_xor_b32_e32 v64, v64, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v64, vcc, v64, v66
	v_mul_lo_u32 v65, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v64, v65
	v_lshl_add_u64 v[64:65], v[48:49], 0, 3
	v_cmp_gt_i64_e32 vcc, -3, v[48:49]
	v_lshl_add_u64 v[66:67], v[62:63], 0, -4
	buffer_load_ubyte v83, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v67, v65, v67, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v66, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v71, v66, v68
	v_xor_b32_e32 v69, v67, v68
	v_mad_u64_u32 v[66:67], s[14:15], v71, s22, 0
	v_mul_hi_u32 v32, v71, s77
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[86:87], s[14:15], v69, s77, 0
	v_add_co_u32_e32 v32, vcc, v66, v86
	v_mad_u64_u32 v[84:85], s[14:15], v69, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v67, v87, vcc
	s_nop 1
	v_addc_co_u32_e32 v85, vcc, 0, v85, vcc
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[84:85]
	v_mul_lo_u32 v32, s35, v66
	v_mul_lo_u32 v67, s34, v67
	v_mad_u64_u32 v[84:85], s[14:15], s34, v66, 0
	v_add3_u32 v32, v85, v67, v32
	v_sub_u32_e32 v67, v69, v32
	v_sub_co_u32_e32 v71, vcc, v71, v84
	s_nop 1
	v_subb_co_u32_e64 v67, s[14:15], v67, v70, vcc
	v_subrev_co_u32_e64 v84, s[14:15], s34, v71
	v_cmp_le_u32_e64 s[24:25], s34, v84
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v67
	v_add_u32_e32 v84, 1, v66
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	v_cndmask_b32_e64 v85, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v67
	v_cndmask_b32_e64 v67, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v67, v85, v67, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v67
	v_add_u32_e32 v67, 2, v66
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cndmask_b32_e64 v67, v84, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v71
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v69, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v66, v67, vcc
	v_xor_b32_e32 v66, s68, v68
	v_xor_b32_e32 v32, v32, v66
	v_sub_u32_e32 v32, v32, v66
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_xor_b32_e32 v86, v64, v66
	v_xad_u32 v71, v32, v66, v34
	v_xor_b32_e32 v67, v65, v66
	v_mad_u64_u32 v[64:65], s[14:15], v86, s22, 0
	v_mul_hi_u32 v32, v86, s77
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[84:85], s[14:15], v67, s77, 0
	v_add_co_u32_e32 v32, vcc, v64, v84
	v_mad_u64_u32 v[68:69], s[14:15], v67, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v65, v85, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[68:69]
	v_mul_lo_u32 v32, s35, v64
	v_mul_lo_u32 v68, s34, v65
	v_mad_u64_u32 v[64:65], s[14:15], s34, v64, 0
	v_add3_u32 v32, v65, v68, v32
	v_sub_u32_e32 v65, v67, v32
	v_sub_co_u32_e32 v64, vcc, v86, v64
	s_nop 1
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, vcc
	v_subrev_co_u32_e64 v68, s[14:15], s34, v64
	v_cmp_le_u32_e64 s[28:29], s34, v68
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[24:25], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v69
	v_cndmask_b32_e64 v85, 0, -1, s[28:29]
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, s[14:15]
	v_cndmask_b32_e64 v84, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v69
	v_subb_co_u32_e32 v32, vcc, v67, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v84, v84, v85, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v84
	v_subrev_co_u32_e64 v84, s[14:15], s34, v68
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v64
	v_cndmask_b32_e64 v65, v69, v65, s[24:25]
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v67, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_cndmask_b32_e64 v65, v68, v84, s[24:25]
	v_cndmask_b32_e32 v64, v64, v65, vcc
	v_xor_b32_e32 v64, v64, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v64, vcc, v64, v66
	v_mul_lo_u32 v65, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v64, v65
	v_lshl_add_u64 v[64:65], v[48:49], 0, 4
	v_cmp_gt_i64_e32 vcc, -4, v[48:49]
	v_lshl_add_u64 v[66:67], v[62:63], 0, -5
	buffer_load_ubyte v84, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v67, v65, v67, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v66, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v71, v66, v68
	v_xor_b32_e32 v69, v67, v68
	v_mad_u64_u32 v[66:67], s[14:15], v71, s22, 0
	v_mul_hi_u32 v32, v71, s77
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[88:89], s[14:15], v69, s77, 0
	v_add_co_u32_e32 v32, vcc, v66, v88
	v_mad_u64_u32 v[86:87], s[14:15], v69, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v67, v89, vcc
	s_nop 1
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[86:87]
	v_mul_lo_u32 v32, s35, v66
	v_mul_lo_u32 v67, s34, v67
	v_mad_u64_u32 v[86:87], s[14:15], s34, v66, 0
	v_add3_u32 v32, v87, v67, v32
	v_sub_u32_e32 v67, v69, v32
	v_sub_co_u32_e32 v71, vcc, v71, v86
	s_nop 1
	v_subb_co_u32_e64 v67, s[14:15], v67, v70, vcc
	v_subrev_co_u32_e64 v85, s[14:15], s34, v71
	v_cmp_le_u32_e64 s[24:25], s34, v85
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v67
	v_add_u32_e32 v85, 1, v66
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	v_cndmask_b32_e64 v86, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v67
	v_cndmask_b32_e64 v67, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v67, v86, v67, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v67
	v_add_u32_e32 v67, 2, v66
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cndmask_b32_e64 v67, v85, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v71
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v69, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v66, v67, vcc
	v_xor_b32_e32 v66, s68, v68
	v_xor_b32_e32 v32, v32, v66
	v_sub_u32_e32 v32, v32, v66
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_xor_b32_e32 v85, v64, v66
	v_xad_u32 v71, v32, v66, v34
	v_xor_b32_e32 v67, v65, v66
	v_mad_u64_u32 v[64:65], s[14:15], v85, s22, 0
	v_mul_hi_u32 v32, v85, s77
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[86:87], s[14:15], v67, s77, 0
	v_add_co_u32_e32 v32, vcc, v64, v86
	v_mad_u64_u32 v[68:69], s[14:15], v67, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v65, v87, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[68:69]
	v_mul_lo_u32 v32, s35, v64
	v_mul_lo_u32 v68, s34, v65
	v_mad_u64_u32 v[64:65], s[14:15], s34, v64, 0
	v_add3_u32 v32, v65, v68, v32
	v_sub_u32_e32 v65, v67, v32
	v_sub_co_u32_e32 v64, vcc, v85, v64
	s_nop 1
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, vcc
	v_subrev_co_u32_e64 v68, s[14:15], s34, v64
	v_cmp_le_u32_e64 s[28:29], s34, v68
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[24:25], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v69
	v_cndmask_b32_e64 v86, 0, -1, s[28:29]
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, s[14:15]
	v_cndmask_b32_e64 v85, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v69
	v_subb_co_u32_e32 v32, vcc, v67, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v85, v85, v86, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v85
	v_subrev_co_u32_e64 v85, s[14:15], s34, v68
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v64
	v_cndmask_b32_e64 v65, v69, v65, s[24:25]
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v67, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_cndmask_b32_e64 v65, v68, v85, s[24:25]
	v_cndmask_b32_e32 v64, v64, v65, vcc
	v_xor_b32_e32 v64, v64, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v64, vcc, v64, v66
	v_mul_lo_u32 v65, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v64, v65
	v_lshl_add_u64 v[64:65], v[48:49], 0, 5
	v_cmp_gt_i64_e32 vcc, -5, v[48:49]
	v_lshl_add_u64 v[66:67], v[62:63], 0, -6
	buffer_load_ubyte v85, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v67, v65, v67, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v66, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v71, v66, v68
	v_xor_b32_e32 v69, v67, v68
	v_mad_u64_u32 v[66:67], s[14:15], v71, s22, 0
	v_mul_hi_u32 v32, v71, s77
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[88:89], s[14:15], v69, s77, 0
	v_add_co_u32_e32 v32, vcc, v66, v88
	v_mad_u64_u32 v[86:87], s[14:15], v69, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v67, v89, vcc
	s_nop 1
	v_addc_co_u32_e32 v87, vcc, 0, v87, vcc
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[86:87]
	v_mul_lo_u32 v32, s35, v66
	v_mul_lo_u32 v67, s34, v67
	v_mad_u64_u32 v[86:87], s[14:15], s34, v66, 0
	v_add3_u32 v32, v87, v67, v32
	v_sub_u32_e32 v67, v69, v32
	v_sub_co_u32_e32 v71, vcc, v71, v86
	s_nop 1
	v_subb_co_u32_e64 v67, s[14:15], v67, v70, vcc
	v_subrev_co_u32_e64 v86, s[14:15], s34, v71
	v_cmp_le_u32_e64 s[24:25], s34, v86
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v67
	v_add_u32_e32 v86, 1, v66
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	v_cndmask_b32_e64 v87, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v67
	v_cndmask_b32_e64 v67, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v67, v87, v67, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v67
	v_add_u32_e32 v67, 2, v66
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cndmask_b32_e64 v67, v86, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v71
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v69, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v66, v67, vcc
	v_xor_b32_e32 v66, s68, v68
	v_xor_b32_e32 v32, v32, v66
	v_sub_u32_e32 v32, v32, v66
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_xor_b32_e32 v88, v64, v66
	v_xad_u32 v71, v32, v66, v34
	v_xor_b32_e32 v67, v65, v66
	v_mad_u64_u32 v[64:65], s[14:15], v88, s22, 0
	v_mul_hi_u32 v32, v88, s77
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[86:87], s[14:15], v67, s77, 0
	v_add_co_u32_e32 v32, vcc, v64, v86
	v_mad_u64_u32 v[68:69], s[14:15], v67, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v65, v87, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[68:69]
	v_mul_lo_u32 v32, s35, v64
	v_mul_lo_u32 v68, s34, v65
	v_mad_u64_u32 v[64:65], s[14:15], s34, v64, 0
	v_add3_u32 v32, v65, v68, v32
	v_sub_u32_e32 v65, v67, v32
	v_sub_co_u32_e32 v64, vcc, v88, v64
	s_nop 1
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, vcc
	v_subrev_co_u32_e64 v68, s[14:15], s34, v64
	v_cmp_le_u32_e64 s[28:29], s34, v68
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[24:25], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v69
	v_cndmask_b32_e64 v87, 0, -1, s[28:29]
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, s[14:15]
	v_cndmask_b32_e64 v86, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v69
	v_subb_co_u32_e32 v32, vcc, v67, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v86, v86, v87, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v86
	v_subrev_co_u32_e64 v86, s[14:15], s34, v68
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v64
	v_cndmask_b32_e64 v65, v69, v65, s[24:25]
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v67, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_cndmask_b32_e64 v65, v68, v86, s[24:25]
	v_cndmask_b32_e32 v64, v64, v65, vcc
	v_xor_b32_e32 v64, v64, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v64, vcc, v64, v66
	v_mul_lo_u32 v65, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v64, v65
	v_lshl_add_u64 v[64:65], v[48:49], 0, 6
	v_cmp_gt_i64_e32 vcc, -6, v[48:49]
	v_lshl_add_u64 v[66:67], v[62:63], 0, -7
	buffer_load_ubyte v86, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v67, v65, v67, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v66, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v71, v66, v68
	v_xor_b32_e32 v69, v67, v68
	v_mad_u64_u32 v[66:67], s[14:15], v71, s22, 0
	v_mul_hi_u32 v32, v71, s77
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[90:91], s[14:15], v69, s77, 0
	v_add_co_u32_e32 v32, vcc, v66, v90
	v_mad_u64_u32 v[88:89], s[14:15], v69, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v67, v91, vcc
	s_nop 1
	v_addc_co_u32_e32 v89, vcc, 0, v89, vcc
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[88:89]
	v_mul_lo_u32 v32, s35, v66
	v_mul_lo_u32 v67, s34, v67
	v_mad_u64_u32 v[88:89], s[14:15], s34, v66, 0
	v_add3_u32 v32, v89, v67, v32
	v_sub_u32_e32 v67, v69, v32
	v_sub_co_u32_e32 v71, vcc, v71, v88
	s_nop 1
	v_subb_co_u32_e64 v67, s[14:15], v67, v70, vcc
	v_subrev_co_u32_e64 v87, s[14:15], s34, v71
	v_cmp_le_u32_e64 s[24:25], s34, v87
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v67
	v_add_u32_e32 v87, 1, v66
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	v_cndmask_b32_e64 v88, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v67
	v_cndmask_b32_e64 v67, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v67, v88, v67, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v67
	v_add_u32_e32 v67, 2, v66
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cndmask_b32_e64 v67, v87, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v71
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v69, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v66, v67, vcc
	v_xor_b32_e32 v66, s68, v68
	v_xor_b32_e32 v32, v32, v66
	v_sub_u32_e32 v32, v32, v66
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_xor_b32_e32 v87, v64, v66
	v_xad_u32 v71, v32, v66, v34
	v_xor_b32_e32 v67, v65, v66
	v_mad_u64_u32 v[64:65], s[14:15], v87, s22, 0
	v_mul_hi_u32 v32, v87, s77
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[88:89], s[14:15], v67, s77, 0
	v_add_co_u32_e32 v32, vcc, v64, v88
	v_mad_u64_u32 v[68:69], s[14:15], v67, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v65, v89, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[68:69]
	v_mul_lo_u32 v32, s35, v64
	v_mul_lo_u32 v68, s34, v65
	v_mad_u64_u32 v[64:65], s[14:15], s34, v64, 0
	v_add3_u32 v32, v65, v68, v32
	v_sub_u32_e32 v65, v67, v32
	v_sub_co_u32_e32 v64, vcc, v87, v64
	s_nop 1
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, vcc
	v_subrev_co_u32_e64 v68, s[14:15], s34, v64
	v_cmp_le_u32_e64 s[28:29], s34, v68
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[24:25], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v69
	v_cndmask_b32_e64 v88, 0, -1, s[28:29]
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, s[14:15]
	v_cndmask_b32_e64 v87, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v69
	v_subb_co_u32_e32 v32, vcc, v67, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v87, v87, v88, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v87
	v_subrev_co_u32_e64 v87, s[14:15], s34, v68
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v64
	v_cndmask_b32_e64 v65, v69, v65, s[24:25]
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v67, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_cndmask_b32_e64 v65, v68, v87, s[24:25]
	v_cndmask_b32_e32 v64, v64, v65, vcc
	v_xor_b32_e32 v64, v64, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v64, vcc, v64, v66
	v_mul_lo_u32 v65, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v64, v65
	v_lshl_add_u64 v[64:65], v[48:49], 0, 7
	v_cmp_gt_i64_e32 vcc, -7, v[48:49]
	v_lshl_add_u64 v[66:67], v[62:63], 0, -8
	buffer_load_ubyte v87, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v67, v65, v67, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v66, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v71, v66, v68
	v_xor_b32_e32 v69, v67, v68
	v_mad_u64_u32 v[66:67], s[14:15], v71, s22, 0
	v_mul_hi_u32 v32, v71, s77
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[90:91], s[14:15], v69, s77, 0
	v_add_co_u32_e32 v32, vcc, v66, v90
	v_mad_u64_u32 v[88:89], s[14:15], v69, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v67, v91, vcc
	s_nop 1
	v_addc_co_u32_e32 v89, vcc, 0, v89, vcc
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[88:89]
	v_mul_lo_u32 v32, s35, v66
	v_mul_lo_u32 v67, s34, v67
	v_mad_u64_u32 v[88:89], s[14:15], s34, v66, 0
	v_add3_u32 v32, v89, v67, v32
	v_sub_u32_e32 v67, v69, v32
	v_sub_co_u32_e32 v71, vcc, v71, v88
	s_nop 1
	v_subb_co_u32_e64 v67, s[14:15], v67, v70, vcc
	v_subrev_co_u32_e64 v88, s[14:15], s34, v71
	v_cmp_le_u32_e64 s[24:25], s34, v88
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v67
	v_add_u32_e32 v88, 1, v66
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	v_cndmask_b32_e64 v89, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v67
	v_cndmask_b32_e64 v67, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v67, v89, v67, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v67
	v_add_u32_e32 v67, 2, v66
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cndmask_b32_e64 v67, v88, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v71
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v69, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v66, v67, vcc
	v_xor_b32_e32 v66, s68, v68
	v_xor_b32_e32 v32, v32, v66
	v_sub_u32_e32 v32, v32, v66
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_xor_b32_e32 v90, v64, v66
	v_xad_u32 v71, v32, v66, v34
	v_xor_b32_e32 v67, v65, v66
	v_mad_u64_u32 v[64:65], s[14:15], v90, s22, 0
	v_mul_hi_u32 v32, v90, s77
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[88:89], s[14:15], v67, s77, 0
	v_add_co_u32_e32 v32, vcc, v64, v88
	v_mad_u64_u32 v[68:69], s[14:15], v67, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v65, v89, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[68:69]
	v_mul_lo_u32 v32, s35, v64
	v_mul_lo_u32 v68, s34, v65
	v_mad_u64_u32 v[64:65], s[14:15], s34, v64, 0
	v_add3_u32 v32, v65, v68, v32
	v_sub_u32_e32 v65, v67, v32
	v_sub_co_u32_e32 v64, vcc, v90, v64
	s_nop 1
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, vcc
	v_subrev_co_u32_e64 v68, s[14:15], s34, v64
	v_cmp_le_u32_e64 s[28:29], s34, v68
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[24:25], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v69
	v_cndmask_b32_e64 v89, 0, -1, s[28:29]
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, s[14:15]
	v_cndmask_b32_e64 v88, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v69
	v_subb_co_u32_e32 v32, vcc, v67, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v88, v88, v89, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v88
	v_subrev_co_u32_e64 v88, s[14:15], s34, v68
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v64
	v_cndmask_b32_e64 v65, v69, v65, s[24:25]
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v67, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_cndmask_b32_e64 v65, v68, v88, s[24:25]
	v_cndmask_b32_e32 v64, v64, v65, vcc
	v_xor_b32_e32 v64, v64, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v64, vcc, v64, v66
	v_mul_lo_u32 v65, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v64, v65
	v_lshl_add_u64 v[64:65], v[48:49], 0, 8
	v_cmp_gt_i64_e32 vcc, -8, v[48:49]
	v_lshl_add_u64 v[66:67], v[62:63], 0, -9
	buffer_load_ubyte v88, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v67, v65, v67, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v66, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v71, v66, v68
	v_xor_b32_e32 v69, v67, v68
	v_mad_u64_u32 v[66:67], s[14:15], v71, s22, 0
	v_mul_hi_u32 v32, v71, s77
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[92:93], s[14:15], v69, s77, 0
	v_add_co_u32_e32 v32, vcc, v66, v92
	v_mad_u64_u32 v[90:91], s[14:15], v69, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v67, v93, vcc
	s_nop 1
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[90:91]
	v_mul_lo_u32 v32, s35, v66
	v_mul_lo_u32 v67, s34, v67
	v_mad_u64_u32 v[90:91], s[14:15], s34, v66, 0
	v_add3_u32 v32, v91, v67, v32
	v_sub_u32_e32 v67, v69, v32
	v_sub_co_u32_e32 v71, vcc, v71, v90
	s_nop 1
	v_subb_co_u32_e64 v67, s[14:15], v67, v70, vcc
	v_subrev_co_u32_e64 v89, s[14:15], s34, v71
	v_cmp_le_u32_e64 s[24:25], s34, v89
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v67
	v_add_u32_e32 v89, 1, v66
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	v_cndmask_b32_e64 v90, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v67
	v_cndmask_b32_e64 v67, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v67, v90, v67, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v67
	v_add_u32_e32 v67, 2, v66
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cndmask_b32_e64 v67, v89, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v71
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v69, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v66, v67, vcc
	v_xor_b32_e32 v66, s68, v68
	v_xor_b32_e32 v32, v32, v66
	v_sub_u32_e32 v32, v32, v66
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_xor_b32_e32 v89, v64, v66
	v_xad_u32 v71, v32, v66, v34
	v_xor_b32_e32 v67, v65, v66
	v_mad_u64_u32 v[64:65], s[14:15], v89, s22, 0
	v_mul_hi_u32 v32, v89, s77
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[90:91], s[14:15], v67, s77, 0
	v_add_co_u32_e32 v32, vcc, v64, v90
	v_mad_u64_u32 v[68:69], s[14:15], v67, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v65, v91, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[68:69]
	v_mul_lo_u32 v32, s35, v64
	v_mul_lo_u32 v68, s34, v65
	v_mad_u64_u32 v[64:65], s[14:15], s34, v64, 0
	v_add3_u32 v32, v65, v68, v32
	v_sub_u32_e32 v65, v67, v32
	v_sub_co_u32_e32 v64, vcc, v89, v64
	s_nop 1
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, vcc
	v_subrev_co_u32_e64 v68, s[14:15], s34, v64
	v_cmp_le_u32_e64 s[28:29], s34, v68
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[24:25], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v69
	v_cndmask_b32_e64 v90, 0, -1, s[28:29]
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, s[14:15]
	v_cndmask_b32_e64 v89, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v69
	v_subb_co_u32_e32 v32, vcc, v67, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v89, v89, v90, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v89
	v_subrev_co_u32_e64 v89, s[14:15], s34, v68
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v64
	v_cndmask_b32_e64 v65, v69, v65, s[24:25]
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v67, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_cndmask_b32_e64 v65, v68, v89, s[24:25]
	v_cndmask_b32_e32 v64, v64, v65, vcc
	v_xor_b32_e32 v64, v64, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v64, vcc, v64, v66
	v_mul_lo_u32 v65, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v64, v65
	v_lshl_add_u64 v[64:65], v[48:49], 0, 9
	v_cmp_gt_i64_e32 vcc, -9, v[48:49]
	v_lshl_add_u64 v[66:67], v[62:63], 0, -10
	buffer_load_ubyte v89, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v67, v65, v67, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v66, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v71, v66, v68
	v_xor_b32_e32 v69, v67, v68
	v_mad_u64_u32 v[66:67], s[14:15], v71, s22, 0
	v_mul_hi_u32 v32, v71, s77
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[92:93], s[14:15], v69, s77, 0
	v_add_co_u32_e32 v32, vcc, v66, v92
	v_mad_u64_u32 v[90:91], s[14:15], v69, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v67, v93, vcc
	s_nop 1
	v_addc_co_u32_e32 v91, vcc, 0, v91, vcc
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[90:91]
	v_mul_lo_u32 v32, s35, v66
	v_mul_lo_u32 v67, s34, v67
	v_mad_u64_u32 v[90:91], s[14:15], s34, v66, 0
	v_add3_u32 v32, v91, v67, v32
	v_sub_u32_e32 v67, v69, v32
	v_sub_co_u32_e32 v71, vcc, v71, v90
	s_nop 1
	v_subb_co_u32_e64 v67, s[14:15], v67, v70, vcc
	v_subrev_co_u32_e64 v90, s[14:15], s34, v71
	v_cmp_le_u32_e64 s[24:25], s34, v90
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v67
	v_add_u32_e32 v90, 1, v66
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	v_cndmask_b32_e64 v91, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v67
	v_cndmask_b32_e64 v67, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v67, v91, v67, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v67
	v_add_u32_e32 v67, 2, v66
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cndmask_b32_e64 v67, v90, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v71
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v69, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v66, v67, vcc
	v_xor_b32_e32 v66, s68, v68
	v_xor_b32_e32 v32, v32, v66
	v_sub_u32_e32 v32, v32, v66
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_xor_b32_e32 v92, v64, v66
	v_xad_u32 v71, v32, v66, v34
	v_xor_b32_e32 v67, v65, v66
	v_mad_u64_u32 v[64:65], s[14:15], v92, s22, 0
	v_mul_hi_u32 v32, v92, s77
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[90:91], s[14:15], v67, s77, 0
	v_add_co_u32_e32 v32, vcc, v64, v90
	v_mad_u64_u32 v[68:69], s[14:15], v67, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v65, v91, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[68:69]
	v_mul_lo_u32 v32, s35, v64
	v_mul_lo_u32 v68, s34, v65
	v_mad_u64_u32 v[64:65], s[14:15], s34, v64, 0
	v_add3_u32 v32, v65, v68, v32
	v_sub_u32_e32 v65, v67, v32
	v_sub_co_u32_e32 v64, vcc, v92, v64
	s_nop 1
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, vcc
	v_subrev_co_u32_e64 v68, s[14:15], s34, v64
	v_cmp_le_u32_e64 s[28:29], s34, v68
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[24:25], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v69
	v_cndmask_b32_e64 v91, 0, -1, s[28:29]
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, s[14:15]
	v_cndmask_b32_e64 v90, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v69
	v_subb_co_u32_e32 v32, vcc, v67, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v90, v90, v91, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v90
	v_subrev_co_u32_e64 v90, s[14:15], s34, v68
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v64
	v_cndmask_b32_e64 v65, v69, v65, s[24:25]
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v67, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_cndmask_b32_e64 v65, v68, v90, s[24:25]
	v_cndmask_b32_e32 v64, v64, v65, vcc
	v_xor_b32_e32 v64, v64, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v64, vcc, v64, v66
	v_mul_lo_u32 v65, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v64, v65
	v_lshl_add_u64 v[64:65], v[48:49], 0, 10
	v_cmp_gt_i64_e32 vcc, -10, v[48:49]
	v_lshl_add_u64 v[66:67], v[62:63], 0, -11
	buffer_load_ubyte v90, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v67, v65, v67, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v66, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v71, v66, v68
	v_xor_b32_e32 v69, v67, v68
	v_mad_u64_u32 v[66:67], s[14:15], v71, s22, 0
	v_mul_hi_u32 v32, v71, s77
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[94:95], s[14:15], v69, s77, 0
	v_add_co_u32_e32 v32, vcc, v66, v94
	v_mad_u64_u32 v[92:93], s[14:15], v69, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v67, v95, vcc
	s_nop 1
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[92:93]
	v_mul_lo_u32 v32, s35, v66
	v_mul_lo_u32 v67, s34, v67
	v_mad_u64_u32 v[92:93], s[14:15], s34, v66, 0
	v_add3_u32 v32, v93, v67, v32
	v_sub_u32_e32 v67, v69, v32
	v_sub_co_u32_e32 v71, vcc, v71, v92
	s_nop 1
	v_subb_co_u32_e64 v67, s[14:15], v67, v70, vcc
	v_subrev_co_u32_e64 v91, s[14:15], s34, v71
	v_cmp_le_u32_e64 s[24:25], s34, v91
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v67
	v_add_u32_e32 v91, 1, v66
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	v_cndmask_b32_e64 v92, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v67
	v_cndmask_b32_e64 v67, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v67, v92, v67, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v67
	v_add_u32_e32 v67, 2, v66
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cndmask_b32_e64 v67, v91, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v71
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v69, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v66, v67, vcc
	v_xor_b32_e32 v66, s68, v68
	v_xor_b32_e32 v32, v32, v66
	v_sub_u32_e32 v32, v32, v66
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_xor_b32_e32 v91, v64, v66
	v_xad_u32 v71, v32, v66, v34
	v_xor_b32_e32 v67, v65, v66
	v_mad_u64_u32 v[64:65], s[14:15], v91, s22, 0
	v_mul_hi_u32 v32, v91, s77
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[92:93], s[14:15], v67, s77, 0
	v_add_co_u32_e32 v32, vcc, v64, v92
	v_mad_u64_u32 v[68:69], s[14:15], v67, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v65, v93, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[68:69]
	v_mul_lo_u32 v32, s35, v64
	v_mul_lo_u32 v68, s34, v65
	v_mad_u64_u32 v[64:65], s[14:15], s34, v64, 0
	v_add3_u32 v32, v65, v68, v32
	v_sub_u32_e32 v65, v67, v32
	v_sub_co_u32_e32 v64, vcc, v91, v64
	s_nop 1
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, vcc
	v_subrev_co_u32_e64 v68, s[14:15], s34, v64
	v_cmp_le_u32_e64 s[28:29], s34, v68
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[24:25], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v69
	v_cndmask_b32_e64 v92, 0, -1, s[28:29]
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, s[14:15]
	v_cndmask_b32_e64 v91, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v69
	v_subb_co_u32_e32 v32, vcc, v67, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v91, v91, v92, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v91
	v_subrev_co_u32_e64 v91, s[14:15], s34, v68
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v64
	v_cndmask_b32_e64 v65, v69, v65, s[24:25]
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v67, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_cndmask_b32_e64 v65, v68, v91, s[24:25]
	v_cndmask_b32_e32 v64, v64, v65, vcc
	v_xor_b32_e32 v64, v64, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v64, vcc, v64, v66
	v_mul_lo_u32 v65, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v64, v65
	v_lshl_add_u64 v[64:65], v[48:49], 0, 11
	v_cmp_gt_i64_e32 vcc, -11, v[48:49]
	v_lshl_add_u64 v[66:67], v[62:63], 0, -12
	buffer_load_ubyte v91, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v67, v65, v67, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v66, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v71, v66, v68
	v_xor_b32_e32 v69, v67, v68
	v_mad_u64_u32 v[66:67], s[14:15], v71, s22, 0
	v_mul_hi_u32 v32, v71, s77
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[94:95], s[14:15], v69, s77, 0
	v_add_co_u32_e32 v32, vcc, v66, v94
	v_mad_u64_u32 v[92:93], s[14:15], v69, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v67, v95, vcc
	s_nop 1
	v_addc_co_u32_e32 v93, vcc, 0, v93, vcc
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[92:93]
	v_mul_lo_u32 v32, s35, v66
	v_mul_lo_u32 v67, s34, v67
	v_mad_u64_u32 v[92:93], s[14:15], s34, v66, 0
	v_add3_u32 v32, v93, v67, v32
	v_sub_u32_e32 v67, v69, v32
	v_sub_co_u32_e32 v71, vcc, v71, v92
	s_nop 1
	v_subb_co_u32_e64 v67, s[14:15], v67, v70, vcc
	v_subrev_co_u32_e64 v92, s[14:15], s34, v71
	v_cmp_le_u32_e64 s[24:25], s34, v92
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v67
	v_add_u32_e32 v92, 1, v66
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	v_cndmask_b32_e64 v93, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v67
	v_cndmask_b32_e64 v67, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v67, v93, v67, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v67
	v_add_u32_e32 v67, 2, v66
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cndmask_b32_e64 v67, v92, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v71
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v69, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v66, v67, vcc
	v_xor_b32_e32 v66, s68, v68
	v_xor_b32_e32 v32, v32, v66
	v_sub_u32_e32 v32, v32, v66
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_xor_b32_e32 v94, v64, v66
	v_xad_u32 v71, v32, v66, v34
	v_xor_b32_e32 v67, v65, v66
	v_mad_u64_u32 v[64:65], s[14:15], v94, s22, 0
	v_mul_hi_u32 v32, v94, s77
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[92:93], s[14:15], v67, s77, 0
	v_add_co_u32_e32 v32, vcc, v64, v92
	v_mad_u64_u32 v[68:69], s[14:15], v67, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v65, v93, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[68:69]
	v_mul_lo_u32 v32, s35, v64
	v_mul_lo_u32 v68, s34, v65
	v_mad_u64_u32 v[64:65], s[14:15], s34, v64, 0
	v_add3_u32 v32, v65, v68, v32
	v_sub_u32_e32 v65, v67, v32
	v_sub_co_u32_e32 v64, vcc, v94, v64
	s_nop 1
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, vcc
	v_subrev_co_u32_e64 v68, s[14:15], s34, v64
	v_cmp_le_u32_e64 s[28:29], s34, v68
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[24:25], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v69
	v_cndmask_b32_e64 v93, 0, -1, s[28:29]
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, s[14:15]
	v_cndmask_b32_e64 v92, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v69
	v_subb_co_u32_e32 v32, vcc, v67, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v92, v92, v93, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v92
	v_subrev_co_u32_e64 v92, s[14:15], s34, v68
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v64
	v_cndmask_b32_e64 v65, v69, v65, s[24:25]
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v67, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_cndmask_b32_e64 v65, v68, v92, s[24:25]
	v_cndmask_b32_e32 v64, v64, v65, vcc
	v_xor_b32_e32 v64, v64, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v64, vcc, v64, v66
	v_mul_lo_u32 v65, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v64, v65
	v_lshl_add_u64 v[64:65], v[48:49], 0, 12
	v_cmp_gt_i64_e32 vcc, -12, v[48:49]
	v_lshl_add_u64 v[66:67], v[62:63], 0, -13
	buffer_load_ubyte v92, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v67, v65, v67, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v66, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v71, v66, v68
	v_xor_b32_e32 v69, v67, v68
	v_mad_u64_u32 v[66:67], s[14:15], v71, s22, 0
	v_mul_hi_u32 v32, v71, s77
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[98:99], s[14:15], v69, s77, 0
	v_add_co_u32_e32 v32, vcc, v66, v98
	v_mad_u64_u32 v[94:95], s[14:15], v69, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v67, v99, vcc
	s_nop 1
	v_addc_co_u32_e32 v95, vcc, 0, v95, vcc
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[94:95]
	v_mul_lo_u32 v32, s35, v66
	v_mul_lo_u32 v67, s34, v67
	v_mad_u64_u32 v[94:95], s[14:15], s34, v66, 0
	v_add3_u32 v32, v95, v67, v32
	v_sub_u32_e32 v67, v69, v32
	v_sub_co_u32_e32 v71, vcc, v71, v94
	s_nop 1
	v_subb_co_u32_e64 v67, s[14:15], v67, v70, vcc
	v_subrev_co_u32_e64 v93, s[14:15], s34, v71
	v_cmp_le_u32_e64 s[24:25], s34, v93
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v67
	v_add_u32_e32 v93, 1, v66
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	v_cndmask_b32_e64 v94, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v67
	v_cndmask_b32_e64 v67, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v67, v94, v67, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v67
	v_add_u32_e32 v67, 2, v66
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cndmask_b32_e64 v67, v93, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v71
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v69, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v66, v67, vcc
	v_xor_b32_e32 v66, s68, v68
	v_xor_b32_e32 v32, v32, v66
	v_sub_u32_e32 v32, v32, v66
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_xor_b32_e32 v93, v64, v66
	v_xad_u32 v71, v32, v66, v34
	v_xor_b32_e32 v67, v65, v66
	v_mad_u64_u32 v[64:65], s[14:15], v93, s22, 0
	v_mul_hi_u32 v32, v93, s77
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[94:95], s[14:15], v67, s77, 0
	v_add_co_u32_e32 v32, vcc, v64, v94
	v_mad_u64_u32 v[68:69], s[14:15], v67, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v65, v95, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[68:69]
	v_mul_lo_u32 v32, s35, v64
	v_mul_lo_u32 v68, s34, v65
	v_mad_u64_u32 v[64:65], s[14:15], s34, v64, 0
	v_add3_u32 v32, v65, v68, v32
	v_sub_u32_e32 v65, v67, v32
	v_sub_co_u32_e32 v64, vcc, v93, v64
	s_nop 1
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, vcc
	v_subrev_co_u32_e64 v68, s[14:15], s34, v64
	v_cmp_le_u32_e64 s[28:29], s34, v68
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[24:25], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v69
	v_cndmask_b32_e64 v94, 0, -1, s[28:29]
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, s[14:15]
	v_cndmask_b32_e64 v93, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v69
	v_subb_co_u32_e32 v32, vcc, v67, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v93, v93, v94, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v93
	v_subrev_co_u32_e64 v93, s[14:15], s34, v68
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v64
	v_cndmask_b32_e64 v65, v69, v65, s[24:25]
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v67, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_cndmask_b32_e64 v65, v68, v93, s[24:25]
	v_cndmask_b32_e32 v64, v64, v65, vcc
	v_xor_b32_e32 v64, v64, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v64, vcc, v64, v66
	v_mul_lo_u32 v65, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v64, v65
	v_lshl_add_u64 v[64:65], v[48:49], 0, 13
	v_cmp_gt_i64_e32 vcc, -13, v[48:49]
	v_lshl_add_u64 v[66:67], v[62:63], 0, -14
	buffer_load_ubyte v93, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v67, v65, v67, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v66, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v71, v66, v68
	v_xor_b32_e32 v69, v67, v68
	v_mad_u64_u32 v[66:67], s[14:15], v71, s22, 0
	v_mul_hi_u32 v32, v71, s77
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[98:99], s[14:15], v69, s77, 0
	v_add_co_u32_e32 v32, vcc, v66, v98
	v_mad_u64_u32 v[94:95], s[14:15], v69, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v67, v99, vcc
	s_nop 1
	v_addc_co_u32_e32 v95, vcc, 0, v95, vcc
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[94:95]
	v_mul_lo_u32 v32, s35, v66
	v_mul_lo_u32 v67, s34, v67
	v_mad_u64_u32 v[94:95], s[14:15], s34, v66, 0
	v_add3_u32 v32, v95, v67, v32
	v_sub_u32_e32 v67, v69, v32
	v_sub_co_u32_e32 v71, vcc, v71, v94
	s_nop 1
	v_subb_co_u32_e64 v67, s[14:15], v67, v70, vcc
	v_subrev_co_u32_e64 v94, s[14:15], s34, v71
	v_cmp_le_u32_e64 s[24:25], s34, v94
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v67
	v_add_u32_e32 v94, 1, v66
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	v_cndmask_b32_e64 v95, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v67
	v_cndmask_b32_e64 v67, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v67, v95, v67, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v67
	v_add_u32_e32 v67, 2, v66
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cndmask_b32_e64 v67, v94, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v71
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v69, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v66, v67, vcc
	v_xor_b32_e32 v66, s68, v68
	v_xor_b32_e32 v32, v32, v66
	v_sub_u32_e32 v32, v32, v66
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_xor_b32_e32 v96, v64, v66
	v_xad_u32 v71, v32, v66, v34
	v_xor_b32_e32 v67, v65, v66
	v_mad_u64_u32 v[64:65], s[14:15], v96, s22, 0
	v_mul_hi_u32 v32, v96, s77
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[94:95], s[14:15], v67, s77, 0
	v_add_co_u32_e32 v32, vcc, v64, v94
	v_mad_u64_u32 v[68:69], s[14:15], v67, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v65, v95, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[68:69]
	v_mul_lo_u32 v32, s35, v64
	v_mul_lo_u32 v68, s34, v65
	v_mad_u64_u32 v[64:65], s[14:15], s34, v64, 0
	v_add3_u32 v32, v65, v68, v32
	v_sub_u32_e32 v65, v67, v32
	v_sub_co_u32_e32 v64, vcc, v96, v64
	s_nop 1
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, vcc
	v_subrev_co_u32_e64 v68, s[14:15], s34, v64
	v_cmp_le_u32_e64 s[28:29], s34, v68
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[24:25], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v69
	v_cndmask_b32_e64 v95, 0, -1, s[28:29]
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, s[14:15]
	v_cndmask_b32_e64 v94, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v69
	v_subb_co_u32_e32 v32, vcc, v67, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v94, v94, v95, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v94
	v_subrev_co_u32_e64 v94, s[14:15], s34, v68
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v64
	v_cndmask_b32_e64 v65, v69, v65, s[24:25]
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v67, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_cndmask_b32_e64 v65, v68, v94, s[24:25]
	v_cndmask_b32_e32 v64, v64, v65, vcc
	v_xor_b32_e32 v64, v64, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v64, vcc, v64, v66
	v_mul_lo_u32 v65, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v64, v65
	v_lshl_add_u64 v[64:65], v[48:49], 0, 14
	v_cmp_gt_i64_e32 vcc, -14, v[48:49]
	v_lshl_add_u64 v[66:67], v[62:63], 0, -15
	buffer_load_ubyte v94, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v67, v65, v67, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v66, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v71, v66, v68
	v_xor_b32_e32 v69, v67, v68
	v_mad_u64_u32 v[66:67], s[14:15], v71, s22, 0
	v_mul_hi_u32 v32, v71, s77
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[100:101], s[14:15], v69, s77, 0
	v_add_co_u32_e32 v32, vcc, v66, v100
	v_mad_u64_u32 v[98:99], s[14:15], v69, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v67, v101, vcc
	s_nop 1
	v_addc_co_u32_e32 v99, vcc, 0, v99, vcc
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[98:99]
	v_mul_lo_u32 v32, s35, v66
	v_mul_lo_u32 v67, s34, v67
	v_mad_u64_u32 v[98:99], s[14:15], s34, v66, 0
	v_add3_u32 v32, v99, v67, v32
	v_sub_u32_e32 v67, v69, v32
	v_sub_co_u32_e32 v71, vcc, v71, v98
	s_nop 1
	v_subb_co_u32_e64 v67, s[14:15], v67, v70, vcc
	v_subrev_co_u32_e64 v95, s[14:15], s34, v71
	v_cmp_le_u32_e64 s[24:25], s34, v95
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v67
	v_add_u32_e32 v95, 1, v66
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	v_cndmask_b32_e64 v96, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v67
	v_cndmask_b32_e64 v67, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v67, v96, v67, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v67
	v_add_u32_e32 v67, 2, v66
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cndmask_b32_e64 v67, v95, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v71
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v69, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v66, v67, vcc
	v_xor_b32_e32 v66, s68, v68
	v_xor_b32_e32 v32, v32, v66
	v_sub_u32_e32 v32, v32, v66
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_xor_b32_e32 v95, v64, v66
	v_xad_u32 v71, v32, v66, v34
	v_xor_b32_e32 v67, v65, v66
	v_mad_u64_u32 v[64:65], s[14:15], v95, s22, 0
	v_mul_hi_u32 v32, v95, s77
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[98:99], s[14:15], v67, s77, 0
	v_add_co_u32_e32 v32, vcc, v64, v98
	v_mad_u64_u32 v[68:69], s[14:15], v67, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v65, v99, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[68:69]
	v_mul_lo_u32 v32, s35, v64
	v_mul_lo_u32 v68, s34, v65
	v_mad_u64_u32 v[64:65], s[14:15], s34, v64, 0
	v_add3_u32 v32, v65, v68, v32
	v_sub_u32_e32 v65, v67, v32
	v_sub_co_u32_e32 v64, vcc, v95, v64
	s_nop 1
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, vcc
	v_subrev_co_u32_e64 v68, s[14:15], s34, v64
	v_cmp_le_u32_e64 s[28:29], s34, v68
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[24:25], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v69
	v_cndmask_b32_e64 v96, 0, -1, s[28:29]
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, s[14:15]
	v_cndmask_b32_e64 v95, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v69
	v_subb_co_u32_e32 v32, vcc, v67, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v95, v95, v96, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v95
	v_subrev_co_u32_e64 v95, s[14:15], s34, v68
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v64
	v_cndmask_b32_e64 v65, v69, v65, s[24:25]
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v67, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v67
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_cndmask_b32_e64 v65, v68, v95, s[24:25]
	v_cndmask_b32_e32 v64, v64, v65, vcc
	v_xor_b32_e32 v64, v64, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v64, vcc, v64, v66
	v_mul_lo_u32 v65, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v64, v65
	v_lshl_add_u64 v[64:65], v[48:49], 0, 15
	v_cmp_gt_i64_e32 vcc, -15, v[48:49]
	v_lshl_add_u64 v[66:67], v[62:63], 0, -16
	buffer_load_ubyte v95, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v67, v65, v67, vcc
	v_ashrrev_i32_e32 v68, 31, v67
	v_cndmask_b32_e32 v66, v64, v66, vcc
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_xor_b32_e32 v71, v66, v68
	v_xor_b32_e32 v69, v67, v68
	v_mad_u64_u32 v[66:67], s[14:15], v71, s22, 0
	v_mul_hi_u32 v32, v71, s77
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[100:101], s[14:15], v69, s77, 0
	v_add_co_u32_e32 v32, vcc, v66, v100
	v_mad_u64_u32 v[98:99], s[14:15], v69, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v67, v101, vcc
	s_nop 1
	v_addc_co_u32_e32 v99, vcc, 0, v99, vcc
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[98:99]
	v_mul_lo_u32 v32, s35, v66
	v_mul_lo_u32 v67, s34, v67
	v_mad_u64_u32 v[98:99], s[14:15], s34, v66, 0
	v_add3_u32 v32, v99, v67, v32
	v_sub_u32_e32 v67, v69, v32
	v_sub_co_u32_e32 v71, vcc, v71, v98
	s_nop 1
	v_subb_co_u32_e64 v67, s[14:15], v67, v70, vcc
	v_subrev_co_u32_e64 v96, s[14:15], s34, v71
	v_cmp_le_u32_e64 s[24:25], s34, v96
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v67
	v_add_u32_e32 v96, 1, v66
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	v_cndmask_b32_e64 v98, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v67
	v_cndmask_b32_e64 v67, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v67, v98, v67, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v67
	v_add_u32_e32 v67, 2, v66
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cndmask_b32_e64 v67, v96, v67, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v71
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v69, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v66, v67, vcc
	v_xor_b32_e32 v66, s68, v68
	v_xor_b32_e32 v32, v32, v66
	v_sub_u32_e32 v32, v32, v66
	v_ashrrev_i32_e32 v66, 31, v65
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	v_xor_b32_e32 v96, v64, v66
	v_xad_u32 v71, v32, v66, v34
	v_xor_b32_e32 v67, v65, v66
	v_mad_u64_u32 v[64:65], s[14:15], v96, s22, 0
	v_mul_hi_u32 v32, v96, s77
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[98:99], s[14:15], v67, s77, 0
	v_add_co_u32_e32 v32, vcc, v64, v98
	v_mad_u64_u32 v[68:69], s[14:15], v67, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v65, v99, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[68:69]
	v_mul_lo_u32 v32, s35, v64
	v_mul_lo_u32 v68, s34, v65
	v_mad_u64_u32 v[64:65], s[14:15], s34, v64, 0
	v_add3_u32 v32, v65, v68, v32
	v_sub_u32_e32 v65, v67, v32
	v_sub_co_u32_e32 v64, vcc, v96, v64
	s_nop 1
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, vcc
	v_subrev_co_u32_e64 v68, s[14:15], s34, v64
	v_cmp_le_u32_e64 s[28:29], s34, v68
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[24:25], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v69
	v_cndmask_b32_e64 v98, 0, -1, s[28:29]
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, s[14:15]
	v_cndmask_b32_e64 v96, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v69
	v_subb_co_u32_e32 v32, vcc, v67, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v96, v96, v98, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v96
	v_subrev_co_u32_e64 v96, s[14:15], s34, v68
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v64
	v_cndmask_b32_e64 v65, v69, v65, s[24:25]
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v69, 0, -1, s[14:15]
	s_movk_i32 s14, 0xfbff
	v_cndmask_b32_e32 v67, v67, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v67
	s_mov_b32 s15, -1
	v_lshl_add_u64 v[62:63], v[62:63], 0, s[14:15]
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_cndmask_b32_e64 v65, v68, v96, s[24:25]
	v_cndmask_b32_e32 v64, v64, v65, vcc
	v_xor_b32_e32 v64, v64, v66
	v_xor_b32_e32 v32, v32, v66
	v_sub_co_u32_e32 v64, vcc, v64, v66
	v_mul_lo_u32 v65, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v66, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v64, v65
	v_lshl_add_u64 v[64:65], v[48:49], 0, s[70:71]
	v_cmp_gt_i32_e32 vcc, 0, v65
	buffer_load_ubyte v96, v32, s[36:39], 0 offen
	s_nop 0
	v_cndmask_b32_e32 v63, v65, v63, vcc
	v_ashrrev_i32_e32 v66, 31, v63
	v_cndmask_b32_e32 v62, v64, v62, vcc
	v_mov_b32_e32 v67, v66
	v_lshl_add_u64 v[62:63], v[62:63], 0, v[66:67]
	v_xor_b32_e32 v71, v62, v66
	v_xor_b32_e32 v67, v63, v66
	v_mad_u64_u32 v[62:63], s[14:15], v71, s22, 0
	v_mul_hi_u32 v32, v71, s77
	v_lshl_add_u64 v[62:63], v[32:33], 0, v[62:63]
	v_mad_u64_u32 v[98:99], s[14:15], v67, s77, 0
	v_add_co_u32_e32 v32, vcc, v62, v98
	v_mad_u64_u32 v[68:69], s[14:15], v67, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v63, v99, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[62:63], v[32:33], 0, v[68:69]
	v_mul_lo_u32 v32, s35, v62
	v_mul_lo_u32 v63, s34, v63
	v_mad_u64_u32 v[68:69], s[14:15], s34, v62, 0
	v_add3_u32 v32, v69, v63, v32
	v_sub_u32_e32 v63, v67, v32
	v_sub_co_u32_e32 v68, vcc, v71, v68
	s_nop 1
	v_subb_co_u32_e64 v63, s[14:15], v63, v70, vcc
	v_subrev_co_u32_e64 v69, s[14:15], s34, v68
	v_cmp_le_u32_e64 s[24:25], s34, v69
	s_nop 0
	v_subbrev_co_u32_e64 v63, s[14:15], 0, v63, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v63
	v_add_u32_e32 v69, 1, v62
	v_subb_co_u32_e32 v32, vcc, v67, v32, vcc
	v_cndmask_b32_e64 v71, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v63
	v_cndmask_b32_e64 v63, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v63, v71, v63, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v63
	v_add_u32_e32 v63, 2, v62
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cndmask_b32_e64 v63, v69, v63, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v68
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v67, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v62, v63, vcc
	v_xor_b32_e32 v62, s68, v66
	v_xor_b32_e32 v32, v32, v62
	v_sub_u32_e32 v32, v32, v62
	v_ashrrev_i32_e32 v62, 31, v65
	v_mov_b32_e32 v63, v62
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[62:63]
	v_xor_b32_e32 v98, v64, v62
	v_xad_u32 v71, v32, v62, v34
	v_xor_b32_e32 v63, v65, v62
	v_mad_u64_u32 v[64:65], s[14:15], v98, s22, 0
	v_mul_hi_u32 v32, v98, s77
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[68:69], s[14:15], v63, s77, 0
	v_add_co_u32_e32 v32, vcc, v64, v68
	v_mad_u64_u32 v[66:67], s[14:15], v63, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v65, v69, vcc
	s_nop 1
	v_addc_co_u32_e32 v67, vcc, 0, v67, vcc
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[66:67]
	v_mul_lo_u32 v32, s35, v64
	v_mul_lo_u32 v66, s34, v65
	v_mad_u64_u32 v[64:65], s[14:15], s34, v64, 0
	v_add3_u32 v32, v65, v66, v32
	v_sub_u32_e32 v65, v63, v32
	v_sub_co_u32_e32 v64, vcc, v98, v64
	s_nop 1
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, vcc
	v_subrev_co_u32_e64 v66, s[14:15], s34, v64
	v_cmp_le_u32_e64 s[28:29], s34, v66
	s_nop 0
	v_subbrev_co_u32_e64 v67, s[24:25], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v67
	v_cndmask_b32_e64 v69, 0, -1, s[28:29]
	v_subb_co_u32_e64 v65, s[14:15], v65, v70, s[14:15]
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v67
	v_subb_co_u32_e32 v32, vcc, v63, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v68, v68, v69, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v68
	v_subrev_co_u32_e64 v68, s[14:15], s34, v66
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[14:15], 0, v65, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v64
	v_cndmask_b32_e64 v65, v67, v65, s[24:25]
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v67, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x401
	v_cndmask_b32_e32 v63, v63, v67, vcc
	v_cmp_ne_u32_e32 vcc, 0, v63
	v_cndmask_b32_e64 v63, v66, v68, s[24:25]
	s_nop 0
	v_cndmask_b32_e32 v63, v64, v63, vcc
	v_cndmask_b32_e32 v32, v32, v65, vcc
	v_xor_b32_e32 v63, v63, v62
	v_xor_b32_e32 v32, v32, v62
	v_sub_co_u32_e32 v63, vcc, v63, v62
	v_lshl_add_u64 v[64:65], v[46:47], 0, v[48:49]
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v62, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_mul_lo_u32 v62, v71, s33
	v_lshl_add_u64 v[68:69], v[64:65], 0, s[14:15]
	s_movk_i32 s14, 0xfbfe
	v_add3_u32 v32, v32, v63, v62
	v_lshl_add_u64 v[66:67], v[64:65], 0, s[70:71]
	v_lshl_add_u64 v[62:63], v[36:37], 0, v[38:39]
	s_mov_b32 s15, -1
	v_cmp_gt_i64_e32 vcc, -1, v[66:67]
	v_lshl_add_u64 v[100:101], v[62:63], 0, s[14:15]
	buffer_load_ubyte v98, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v101, v69, v101, vcc
	v_ashrrev_i32_e32 v102, 31, v101
	v_cndmask_b32_e32 v100, v68, v100, vcc
	v_mov_b32_e32 v103, v102
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[102:103]
	v_xor_b32_e32 v99, v100, v102
	v_xor_b32_e32 v71, v101, v102
	v_mad_u64_u32 v[100:101], s[14:15], v99, s22, 0
	v_mul_hi_u32 v32, v99, s77
	v_lshl_add_u64 v[100:101], v[32:33], 0, v[100:101]
	v_mad_u64_u32 v[106:107], s[14:15], v71, s77, 0
	v_add_co_u32_e32 v32, vcc, v100, v106
	v_mad_u64_u32 v[104:105], s[14:15], v71, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v101, v107, vcc
	s_nop 1
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[100:101], v[32:33], 0, v[104:105]
	v_mul_lo_u32 v32, s35, v100
	v_mul_lo_u32 v101, s34, v101
	v_mad_u64_u32 v[104:105], s[14:15], s34, v100, 0
	v_add3_u32 v32, v105, v101, v32
	v_sub_u32_e32 v101, v71, v32
	v_sub_co_u32_e32 v99, vcc, v99, v104
	s_nop 1
	v_subb_co_u32_e64 v101, s[14:15], v101, v70, vcc
	v_subrev_co_u32_e64 v103, s[14:15], s34, v99
	v_cmp_le_u32_e64 s[24:25], s34, v103
	s_nop 0
	v_subbrev_co_u32_e64 v101, s[14:15], 0, v101, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v101
	v_add_u32_e32 v103, 1, v100
	v_subb_co_u32_e32 v32, vcc, v71, v32, vcc
	v_cndmask_b32_e64 v104, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v101
	v_cndmask_b32_e64 v101, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v101, v104, v101, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v101
	v_add_u32_e32 v101, 2, v100
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cndmask_b32_e64 v101, v103, v101, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v99
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v71, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_xor_b32_e32 v71, s68, v102
	s_nop 0
	v_cndmask_b32_e32 v32, v100, v101, vcc
	v_ashrrev_i32_e32 v100, 31, v69
	v_mov_b32_e32 v101, v100
	v_xor_b32_e32 v32, v32, v71
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[100:101]
	v_sub_u32_e32 v32, v32, v71
	v_xor_b32_e32 v101, v68, v100
	v_xad_u32 v71, v32, v100, v34
	v_xor_b32_e32 v99, v69, v100
	v_mad_u64_u32 v[68:69], s[14:15], v101, s22, 0
	v_mul_hi_u32 v32, v101, s77
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[68:69]
	v_mad_u64_u32 v[104:105], s[14:15], v99, s77, 0
	v_add_co_u32_e32 v32, vcc, v68, v104
	v_mad_u64_u32 v[102:103], s[14:15], v99, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v69, v105, vcc
	s_nop 1
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[102:103]
	v_mul_lo_u32 v32, s35, v68
	v_mul_lo_u32 v102, s34, v69
	v_mad_u64_u32 v[68:69], s[14:15], s34, v68, 0
	v_add3_u32 v32, v69, v102, v32
	v_sub_u32_e32 v69, v99, v32
	v_sub_co_u32_e32 v68, vcc, v101, v68
	s_nop 1
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, vcc
	v_subrev_co_u32_e64 v101, s[14:15], s34, v68
	v_cmp_le_u32_e64 s[28:29], s34, v101
	s_nop 0
	v_subbrev_co_u32_e64 v102, s[24:25], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v102
	v_cndmask_b32_e64 v104, 0, -1, s[28:29]
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, s[14:15]
	v_cndmask_b32_e64 v103, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v102
	v_subb_co_u32_e32 v32, vcc, v99, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v103, v103, v104, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v103
	v_subrev_co_u32_e64 v103, s[14:15], s34, v101
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v68
	v_cndmask_b32_e64 v69, v102, v69, s[24:25]
	v_cndmask_b32_e64 v99, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v102, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x402
	v_cndmask_b32_e32 v99, v99, v102, vcc
	v_cmp_ne_u32_e32 vcc, 0, v99
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v69, vcc
	v_cndmask_b32_e64 v69, v101, v103, s[24:25]
	v_cndmask_b32_e32 v68, v68, v69, vcc
	v_xor_b32_e32 v68, v68, v100
	v_xor_b32_e32 v32, v32, v100
	v_sub_co_u32_e32 v68, vcc, v68, v100
	v_mul_lo_u32 v69, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v100, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v68, v69
	v_lshl_add_u64 v[68:69], v[64:65], 0, s[14:15]
	s_movk_i32 s14, 0xfbfd
	s_mov_b32 s15, -1
	v_cmp_gt_i64_e32 vcc, -2, v[66:67]
	v_lshl_add_u64 v[100:101], v[62:63], 0, s[14:15]
	buffer_load_ubyte v99, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v101, v69, v101, vcc
	v_ashrrev_i32_e32 v102, 31, v101
	v_cndmask_b32_e32 v100, v68, v100, vcc
	v_mov_b32_e32 v103, v102
	v_lshl_add_u64 v[100:101], v[100:101], 0, v[102:103]
	v_xor_b32_e32 v103, v100, v102
	v_xor_b32_e32 v71, v101, v102
	v_mad_u64_u32 v[100:101], s[14:15], v103, s22, 0
	v_mul_hi_u32 v32, v103, s77
	v_lshl_add_u64 v[100:101], v[32:33], 0, v[100:101]
	v_mad_u64_u32 v[106:107], s[14:15], v71, s77, 0
	v_add_co_u32_e32 v32, vcc, v100, v106
	v_mad_u64_u32 v[104:105], s[14:15], v71, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v101, v107, vcc
	s_nop 1
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[100:101], v[32:33], 0, v[104:105]
	v_mul_lo_u32 v32, s35, v100
	v_mul_lo_u32 v101, s34, v101
	v_mad_u64_u32 v[104:105], s[14:15], s34, v100, 0
	v_add3_u32 v32, v105, v101, v32
	v_sub_u32_e32 v101, v71, v32
	v_sub_co_u32_e32 v103, vcc, v103, v104
	s_nop 1
	v_subb_co_u32_e64 v101, s[14:15], v101, v70, vcc
	v_subrev_co_u32_e64 v104, s[14:15], s34, v103
	v_cmp_le_u32_e64 s[24:25], s34, v104
	s_nop 0
	v_subbrev_co_u32_e64 v101, s[14:15], 0, v101, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v101
	v_add_u32_e32 v104, 1, v100
	v_subb_co_u32_e32 v32, vcc, v71, v32, vcc
	v_cndmask_b32_e64 v105, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v101
	v_cndmask_b32_e64 v101, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v101, v105, v101, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v101
	v_add_u32_e32 v101, 2, v100
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cndmask_b32_e64 v101, v104, v101, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v103
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v71, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_xor_b32_e32 v71, s68, v102
	s_nop 0
	v_cndmask_b32_e32 v32, v100, v101, vcc
	v_ashrrev_i32_e32 v100, 31, v69
	v_mov_b32_e32 v101, v100
	v_xor_b32_e32 v32, v32, v71
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[100:101]
	v_sub_u32_e32 v32, v32, v71
	v_xor_b32_e32 v106, v68, v100
	v_xad_u32 v71, v32, v100, v34
	v_xor_b32_e32 v101, v69, v100
	v_mad_u64_u32 v[68:69], s[14:15], v106, s22, 0
	v_mul_hi_u32 v32, v106, s77
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[68:69]
	v_mad_u64_u32 v[104:105], s[14:15], v101, s77, 0
	v_add_co_u32_e32 v32, vcc, v68, v104
	v_mad_u64_u32 v[102:103], s[14:15], v101, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v69, v105, vcc
	s_nop 1
	v_addc_co_u32_e32 v103, vcc, 0, v103, vcc
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[102:103]
	v_mul_lo_u32 v32, s35, v68
	v_mul_lo_u32 v102, s34, v69
	v_mad_u64_u32 v[68:69], s[14:15], s34, v68, 0
	v_add3_u32 v32, v69, v102, v32
	v_sub_u32_e32 v69, v101, v32
	v_sub_co_u32_e32 v68, vcc, v106, v68
	s_nop 1
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, vcc
	v_subrev_co_u32_e64 v102, s[14:15], s34, v68
	v_cmp_le_u32_e64 s[28:29], s34, v102
	s_nop 0
	v_subbrev_co_u32_e64 v103, s[24:25], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v103
	v_cndmask_b32_e64 v105, 0, -1, s[28:29]
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, s[14:15]
	v_cndmask_b32_e64 v104, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v103
	v_subb_co_u32_e32 v32, vcc, v101, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v104, v104, v105, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v104
	v_subrev_co_u32_e64 v104, s[14:15], s34, v102
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v68
	v_cndmask_b32_e64 v69, v103, v69, s[24:25]
	v_cndmask_b32_e64 v101, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v103, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x403
	v_cndmask_b32_e32 v101, v101, v103, vcc
	v_cmp_ne_u32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v69, vcc
	v_cndmask_b32_e64 v69, v102, v104, s[24:25]
	v_cndmask_b32_e32 v68, v68, v69, vcc
	v_xor_b32_e32 v68, v68, v100
	v_xor_b32_e32 v32, v32, v100
	v_sub_co_u32_e32 v68, vcc, v68, v100
	v_mul_lo_u32 v69, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v100, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v68, v69
	v_lshl_add_u64 v[68:69], v[64:65], 0, s[14:15]
	s_movk_i32 s14, 0xfbfc
	s_mov_b32 s15, -1
	v_cmp_gt_i64_e32 vcc, -3, v[66:67]
	v_lshl_add_u64 v[102:103], v[62:63], 0, s[14:15]
	buffer_load_ubyte v100, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v103, v69, v103, vcc
	v_ashrrev_i32_e32 v104, 31, v103
	v_cndmask_b32_e32 v102, v68, v102, vcc
	v_mov_b32_e32 v105, v104
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[104:105]
	v_xor_b32_e32 v101, v102, v104
	v_xor_b32_e32 v71, v103, v104
	v_mad_u64_u32 v[102:103], s[14:15], v101, s22, 0
	v_mul_hi_u32 v32, v101, s77
	v_lshl_add_u64 v[102:103], v[32:33], 0, v[102:103]
	v_mad_u64_u32 v[108:109], s[14:15], v71, s77, 0
	v_add_co_u32_e32 v32, vcc, v102, v108
	v_mad_u64_u32 v[106:107], s[14:15], v71, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v103, v109, vcc
	s_nop 1
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[102:103], v[32:33], 0, v[106:107]
	v_mul_lo_u32 v32, s35, v102
	v_mul_lo_u32 v103, s34, v103
	v_mad_u64_u32 v[106:107], s[14:15], s34, v102, 0
	v_add3_u32 v32, v107, v103, v32
	v_sub_u32_e32 v103, v71, v32
	v_sub_co_u32_e32 v101, vcc, v101, v106
	s_nop 1
	v_subb_co_u32_e64 v103, s[14:15], v103, v70, vcc
	v_subrev_co_u32_e64 v105, s[14:15], s34, v101
	v_cmp_le_u32_e64 s[24:25], s34, v105
	s_nop 0
	v_subbrev_co_u32_e64 v103, s[14:15], 0, v103, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v103
	v_add_u32_e32 v105, 1, v102
	v_subb_co_u32_e32 v32, vcc, v71, v32, vcc
	v_cndmask_b32_e64 v106, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v103
	v_cndmask_b32_e64 v103, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v103, v106, v103, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v103
	v_add_u32_e32 v103, 2, v102
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cndmask_b32_e64 v103, v105, v103, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v101
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v71, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_xor_b32_e32 v71, s68, v104
	s_nop 0
	v_cndmask_b32_e32 v32, v102, v103, vcc
	v_ashrrev_i32_e32 v102, 31, v69
	v_mov_b32_e32 v103, v102
	v_xor_b32_e32 v32, v32, v71
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[102:103]
	v_sub_u32_e32 v32, v32, v71
	v_xor_b32_e32 v103, v68, v102
	v_xad_u32 v71, v32, v102, v34
	v_xor_b32_e32 v101, v69, v102
	v_mad_u64_u32 v[68:69], s[14:15], v103, s22, 0
	v_mul_hi_u32 v32, v103, s77
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[68:69]
	v_mad_u64_u32 v[106:107], s[14:15], v101, s77, 0
	v_add_co_u32_e32 v32, vcc, v68, v106
	v_mad_u64_u32 v[104:105], s[14:15], v101, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v69, v107, vcc
	s_nop 1
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[104:105]
	v_mul_lo_u32 v32, s35, v68
	v_mul_lo_u32 v104, s34, v69
	v_mad_u64_u32 v[68:69], s[14:15], s34, v68, 0
	v_add3_u32 v32, v69, v104, v32
	v_sub_u32_e32 v69, v101, v32
	v_sub_co_u32_e32 v68, vcc, v103, v68
	s_nop 1
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, vcc
	v_subrev_co_u32_e64 v103, s[14:15], s34, v68
	v_cmp_le_u32_e64 s[28:29], s34, v103
	s_nop 0
	v_subbrev_co_u32_e64 v104, s[24:25], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v104
	v_cndmask_b32_e64 v106, 0, -1, s[28:29]
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, s[14:15]
	v_cndmask_b32_e64 v105, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v104
	v_subb_co_u32_e32 v32, vcc, v101, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v105, v105, v106, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v105
	v_subrev_co_u32_e64 v105, s[14:15], s34, v103
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v68
	v_cndmask_b32_e64 v69, v104, v69, s[24:25]
	v_cndmask_b32_e64 v101, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v104, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x404
	v_cndmask_b32_e32 v101, v101, v104, vcc
	v_cmp_ne_u32_e32 vcc, 0, v101
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v69, vcc
	v_cndmask_b32_e64 v69, v103, v105, s[24:25]
	v_cndmask_b32_e32 v68, v68, v69, vcc
	v_xor_b32_e32 v68, v68, v102
	v_xor_b32_e32 v32, v32, v102
	v_sub_co_u32_e32 v68, vcc, v68, v102
	v_mul_lo_u32 v69, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v102, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v68, v69
	v_lshl_add_u64 v[68:69], v[64:65], 0, s[14:15]
	s_movk_i32 s14, 0xfbfb
	s_mov_b32 s15, -1
	v_cmp_gt_i64_e32 vcc, -4, v[66:67]
	v_lshl_add_u64 v[102:103], v[62:63], 0, s[14:15]
	buffer_load_ubyte v101, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v103, v69, v103, vcc
	v_ashrrev_i32_e32 v104, 31, v103
	v_cndmask_b32_e32 v102, v68, v102, vcc
	v_mov_b32_e32 v105, v104
	v_lshl_add_u64 v[102:103], v[102:103], 0, v[104:105]
	v_xor_b32_e32 v105, v102, v104
	v_xor_b32_e32 v71, v103, v104
	v_mad_u64_u32 v[102:103], s[14:15], v105, s22, 0
	v_mul_hi_u32 v32, v105, s77
	v_lshl_add_u64 v[102:103], v[32:33], 0, v[102:103]
	v_mad_u64_u32 v[108:109], s[14:15], v71, s77, 0
	v_add_co_u32_e32 v32, vcc, v102, v108
	v_mad_u64_u32 v[106:107], s[14:15], v71, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v103, v109, vcc
	s_nop 1
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[102:103], v[32:33], 0, v[106:107]
	v_mul_lo_u32 v32, s35, v102
	v_mul_lo_u32 v103, s34, v103
	v_mad_u64_u32 v[106:107], s[14:15], s34, v102, 0
	v_add3_u32 v32, v107, v103, v32
	v_sub_u32_e32 v103, v71, v32
	v_sub_co_u32_e32 v105, vcc, v105, v106
	s_nop 1
	v_subb_co_u32_e64 v103, s[14:15], v103, v70, vcc
	v_subrev_co_u32_e64 v106, s[14:15], s34, v105
	v_cmp_le_u32_e64 s[24:25], s34, v106
	s_nop 0
	v_subbrev_co_u32_e64 v103, s[14:15], 0, v103, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v103
	v_add_u32_e32 v106, 1, v102
	v_subb_co_u32_e32 v32, vcc, v71, v32, vcc
	v_cndmask_b32_e64 v107, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v103
	v_cndmask_b32_e64 v103, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v103, v107, v103, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v103
	v_add_u32_e32 v103, 2, v102
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cndmask_b32_e64 v103, v106, v103, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v105
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v71, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_xor_b32_e32 v71, s68, v104
	s_nop 0
	v_cndmask_b32_e32 v32, v102, v103, vcc
	v_ashrrev_i32_e32 v102, 31, v69
	v_mov_b32_e32 v103, v102
	v_xor_b32_e32 v32, v32, v71
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[102:103]
	v_sub_u32_e32 v32, v32, v71
	v_xor_b32_e32 v108, v68, v102
	v_xad_u32 v71, v32, v102, v34
	v_xor_b32_e32 v103, v69, v102
	v_mad_u64_u32 v[68:69], s[14:15], v108, s22, 0
	v_mul_hi_u32 v32, v108, s77
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[68:69]
	v_mad_u64_u32 v[106:107], s[14:15], v103, s77, 0
	v_add_co_u32_e32 v32, vcc, v68, v106
	v_mad_u64_u32 v[104:105], s[14:15], v103, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v69, v107, vcc
	s_nop 1
	v_addc_co_u32_e32 v105, vcc, 0, v105, vcc
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[104:105]
	v_mul_lo_u32 v32, s35, v68
	v_mul_lo_u32 v104, s34, v69
	v_mad_u64_u32 v[68:69], s[14:15], s34, v68, 0
	v_add3_u32 v32, v69, v104, v32
	v_sub_u32_e32 v69, v103, v32
	v_sub_co_u32_e32 v68, vcc, v108, v68
	s_nop 1
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, vcc
	v_subrev_co_u32_e64 v104, s[14:15], s34, v68
	v_cmp_le_u32_e64 s[28:29], s34, v104
	s_nop 0
	v_subbrev_co_u32_e64 v105, s[24:25], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v105
	v_cndmask_b32_e64 v107, 0, -1, s[28:29]
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, s[14:15]
	v_cndmask_b32_e64 v106, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v105
	v_subb_co_u32_e32 v32, vcc, v103, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v106, v106, v107, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v106
	v_subrev_co_u32_e64 v106, s[14:15], s34, v104
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v68
	v_cndmask_b32_e64 v69, v105, v69, s[24:25]
	v_cndmask_b32_e64 v103, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v105, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x405
	v_cndmask_b32_e32 v103, v103, v105, vcc
	v_cmp_ne_u32_e32 vcc, 0, v103
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v69, vcc
	v_cndmask_b32_e64 v69, v104, v106, s[24:25]
	v_cndmask_b32_e32 v68, v68, v69, vcc
	v_xor_b32_e32 v68, v68, v102
	v_xor_b32_e32 v32, v32, v102
	v_sub_co_u32_e32 v68, vcc, v68, v102
	v_mul_lo_u32 v69, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v102, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v68, v69
	v_lshl_add_u64 v[68:69], v[64:65], 0, s[14:15]
	s_movk_i32 s14, 0xfbfa
	s_mov_b32 s15, -1
	v_cmp_gt_i64_e32 vcc, -5, v[66:67]
	v_lshl_add_u64 v[104:105], v[62:63], 0, s[14:15]
	buffer_load_ubyte v102, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v105, v69, v105, vcc
	v_ashrrev_i32_e32 v106, 31, v105
	v_cndmask_b32_e32 v104, v68, v104, vcc
	v_mov_b32_e32 v107, v106
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[106:107]
	v_xor_b32_e32 v103, v104, v106
	v_xor_b32_e32 v71, v105, v106
	v_mad_u64_u32 v[104:105], s[14:15], v103, s22, 0
	v_mul_hi_u32 v32, v103, s77
	v_lshl_add_u64 v[104:105], v[32:33], 0, v[104:105]
	v_mad_u64_u32 v[110:111], s[14:15], v71, s77, 0
	v_add_co_u32_e32 v32, vcc, v104, v110
	v_mad_u64_u32 v[108:109], s[14:15], v71, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v105, v111, vcc
	s_nop 1
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[104:105], v[32:33], 0, v[108:109]
	v_mul_lo_u32 v32, s35, v104
	v_mul_lo_u32 v105, s34, v105
	v_mad_u64_u32 v[108:109], s[14:15], s34, v104, 0
	v_add3_u32 v32, v109, v105, v32
	v_sub_u32_e32 v105, v71, v32
	v_sub_co_u32_e32 v103, vcc, v103, v108
	s_nop 1
	v_subb_co_u32_e64 v105, s[14:15], v105, v70, vcc
	v_subrev_co_u32_e64 v107, s[14:15], s34, v103
	v_cmp_le_u32_e64 s[24:25], s34, v107
	s_nop 0
	v_subbrev_co_u32_e64 v105, s[14:15], 0, v105, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v105
	v_add_u32_e32 v107, 1, v104
	v_subb_co_u32_e32 v32, vcc, v71, v32, vcc
	v_cndmask_b32_e64 v108, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v105
	v_cndmask_b32_e64 v105, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v105, v108, v105, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v105
	v_add_u32_e32 v105, 2, v104
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cndmask_b32_e64 v105, v107, v105, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v103
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v71, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_xor_b32_e32 v71, s68, v106
	s_nop 0
	v_cndmask_b32_e32 v32, v104, v105, vcc
	v_ashrrev_i32_e32 v104, 31, v69
	v_mov_b32_e32 v105, v104
	v_xor_b32_e32 v32, v32, v71
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[104:105]
	v_sub_u32_e32 v32, v32, v71
	v_xor_b32_e32 v105, v68, v104
	v_xad_u32 v71, v32, v104, v34
	v_xor_b32_e32 v103, v69, v104
	v_mad_u64_u32 v[68:69], s[14:15], v105, s22, 0
	v_mul_hi_u32 v32, v105, s77
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[68:69]
	v_mad_u64_u32 v[108:109], s[14:15], v103, s77, 0
	v_add_co_u32_e32 v32, vcc, v68, v108
	v_mad_u64_u32 v[106:107], s[14:15], v103, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v69, v109, vcc
	s_nop 1
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[106:107]
	v_mul_lo_u32 v32, s35, v68
	v_mul_lo_u32 v106, s34, v69
	v_mad_u64_u32 v[68:69], s[14:15], s34, v68, 0
	v_add3_u32 v32, v69, v106, v32
	v_sub_u32_e32 v69, v103, v32
	v_sub_co_u32_e32 v68, vcc, v105, v68
	s_nop 1
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, vcc
	v_subrev_co_u32_e64 v105, s[14:15], s34, v68
	v_cmp_le_u32_e64 s[28:29], s34, v105
	s_nop 0
	v_subbrev_co_u32_e64 v106, s[24:25], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v106
	v_cndmask_b32_e64 v108, 0, -1, s[28:29]
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, s[14:15]
	v_cndmask_b32_e64 v107, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v106
	v_subb_co_u32_e32 v32, vcc, v103, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v107, v107, v108, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v107
	v_subrev_co_u32_e64 v107, s[14:15], s34, v105
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v68
	v_cndmask_b32_e64 v69, v106, v69, s[24:25]
	v_cndmask_b32_e64 v103, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v106, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x406
	v_cndmask_b32_e32 v103, v103, v106, vcc
	v_cmp_ne_u32_e32 vcc, 0, v103
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v69, vcc
	v_cndmask_b32_e64 v69, v105, v107, s[24:25]
	v_cndmask_b32_e32 v68, v68, v69, vcc
	v_xor_b32_e32 v68, v68, v104
	v_xor_b32_e32 v32, v32, v104
	v_sub_co_u32_e32 v68, vcc, v68, v104
	v_mul_lo_u32 v69, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v104, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v68, v69
	v_lshl_add_u64 v[68:69], v[64:65], 0, s[14:15]
	s_movk_i32 s14, 0xfbf9
	s_mov_b32 s15, -1
	v_cmp_gt_i64_e32 vcc, -6, v[66:67]
	v_lshl_add_u64 v[104:105], v[62:63], 0, s[14:15]
	buffer_load_ubyte v103, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v105, v69, v105, vcc
	v_ashrrev_i32_e32 v106, 31, v105
	v_cndmask_b32_e32 v104, v68, v104, vcc
	v_mov_b32_e32 v107, v106
	v_lshl_add_u64 v[104:105], v[104:105], 0, v[106:107]
	v_xor_b32_e32 v107, v104, v106
	v_xor_b32_e32 v71, v105, v106
	v_mad_u64_u32 v[104:105], s[14:15], v107, s22, 0
	v_mul_hi_u32 v32, v107, s77
	v_lshl_add_u64 v[104:105], v[32:33], 0, v[104:105]
	v_mad_u64_u32 v[110:111], s[14:15], v71, s77, 0
	v_add_co_u32_e32 v32, vcc, v104, v110
	v_mad_u64_u32 v[108:109], s[14:15], v71, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v105, v111, vcc
	s_nop 1
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[104:105], v[32:33], 0, v[108:109]
	v_mul_lo_u32 v32, s35, v104
	v_mul_lo_u32 v105, s34, v105
	v_mad_u64_u32 v[108:109], s[14:15], s34, v104, 0
	v_add3_u32 v32, v109, v105, v32
	v_sub_u32_e32 v105, v71, v32
	v_sub_co_u32_e32 v107, vcc, v107, v108
	s_nop 1
	v_subb_co_u32_e64 v105, s[14:15], v105, v70, vcc
	v_subrev_co_u32_e64 v108, s[14:15], s34, v107
	v_cmp_le_u32_e64 s[24:25], s34, v108
	s_nop 0
	v_subbrev_co_u32_e64 v105, s[14:15], 0, v105, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v105
	v_add_u32_e32 v108, 1, v104
	v_subb_co_u32_e32 v32, vcc, v71, v32, vcc
	v_cndmask_b32_e64 v109, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v105
	v_cndmask_b32_e64 v105, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v105, v109, v105, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v105
	v_add_u32_e32 v105, 2, v104
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cndmask_b32_e64 v105, v108, v105, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v107
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v71, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_xor_b32_e32 v71, s68, v106
	s_nop 0
	v_cndmask_b32_e32 v32, v104, v105, vcc
	v_ashrrev_i32_e32 v104, 31, v69
	v_mov_b32_e32 v105, v104
	v_xor_b32_e32 v32, v32, v71
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[104:105]
	v_sub_u32_e32 v32, v32, v71
	v_xor_b32_e32 v110, v68, v104
	v_xad_u32 v71, v32, v104, v34
	v_xor_b32_e32 v105, v69, v104
	v_mad_u64_u32 v[68:69], s[14:15], v110, s22, 0
	v_mul_hi_u32 v32, v110, s77
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[68:69]
	v_mad_u64_u32 v[108:109], s[14:15], v105, s77, 0
	v_add_co_u32_e32 v32, vcc, v68, v108
	v_mad_u64_u32 v[106:107], s[14:15], v105, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v69, v109, vcc
	s_nop 1
	v_addc_co_u32_e32 v107, vcc, 0, v107, vcc
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[106:107]
	v_mul_lo_u32 v32, s35, v68
	v_mul_lo_u32 v106, s34, v69
	v_mad_u64_u32 v[68:69], s[14:15], s34, v68, 0
	v_add3_u32 v32, v69, v106, v32
	v_sub_u32_e32 v69, v105, v32
	v_sub_co_u32_e32 v68, vcc, v110, v68
	s_nop 1
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, vcc
	v_subrev_co_u32_e64 v106, s[14:15], s34, v68
	v_cmp_le_u32_e64 s[28:29], s34, v106
	s_nop 0
	v_subbrev_co_u32_e64 v107, s[24:25], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v107
	v_cndmask_b32_e64 v109, 0, -1, s[28:29]
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, s[14:15]
	v_cndmask_b32_e64 v108, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v107
	v_subb_co_u32_e32 v32, vcc, v105, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v108, v108, v109, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v108
	v_subrev_co_u32_e64 v108, s[14:15], s34, v106
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v68
	v_cndmask_b32_e64 v69, v107, v69, s[24:25]
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v107, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x407
	v_cndmask_b32_e32 v105, v105, v107, vcc
	v_cmp_ne_u32_e32 vcc, 0, v105
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v69, vcc
	v_cndmask_b32_e64 v69, v106, v108, s[24:25]
	v_cndmask_b32_e32 v68, v68, v69, vcc
	v_xor_b32_e32 v68, v68, v104
	v_xor_b32_e32 v32, v32, v104
	v_sub_co_u32_e32 v68, vcc, v68, v104
	v_mul_lo_u32 v69, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v104, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v68, v69
	v_lshl_add_u64 v[68:69], v[64:65], 0, s[14:15]
	s_movk_i32 s14, 0xfbf8
	s_mov_b32 s15, -1
	v_cmp_gt_i64_e32 vcc, -7, v[66:67]
	v_lshl_add_u64 v[106:107], v[62:63], 0, s[14:15]
	buffer_load_ubyte v104, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v107, v69, v107, vcc
	v_ashrrev_i32_e32 v108, 31, v107
	v_cndmask_b32_e32 v106, v68, v106, vcc
	v_mov_b32_e32 v109, v108
	v_lshl_add_u64 v[106:107], v[106:107], 0, v[108:109]
	v_xor_b32_e32 v105, v106, v108
	v_xor_b32_e32 v71, v107, v108
	v_mad_u64_u32 v[106:107], s[14:15], v105, s22, 0
	v_mul_hi_u32 v32, v105, s77
	v_lshl_add_u64 v[106:107], v[32:33], 0, v[106:107]
	v_mad_u64_u32 v[112:113], s[14:15], v71, s77, 0
	v_add_co_u32_e32 v32, vcc, v106, v112
	v_mad_u64_u32 v[110:111], s[14:15], v71, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v107, v113, vcc
	s_nop 1
	v_addc_co_u32_e32 v111, vcc, 0, v111, vcc
	v_lshl_add_u64 v[106:107], v[32:33], 0, v[110:111]
	v_mul_lo_u32 v32, s35, v106
	v_mul_lo_u32 v107, s34, v107
	v_mad_u64_u32 v[110:111], s[14:15], s34, v106, 0
	v_add3_u32 v32, v111, v107, v32
	v_sub_u32_e32 v107, v71, v32
	v_sub_co_u32_e32 v105, vcc, v105, v110
	s_nop 1
	v_subb_co_u32_e64 v107, s[14:15], v107, v70, vcc
	v_subrev_co_u32_e64 v109, s[14:15], s34, v105
	v_cmp_le_u32_e64 s[24:25], s34, v109
	s_nop 0
	v_subbrev_co_u32_e64 v107, s[14:15], 0, v107, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v107
	v_add_u32_e32 v109, 1, v106
	v_subb_co_u32_e32 v32, vcc, v71, v32, vcc
	v_cndmask_b32_e64 v110, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v107
	v_cndmask_b32_e64 v107, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v107, v110, v107, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v107
	v_add_u32_e32 v107, 2, v106
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cndmask_b32_e64 v107, v109, v107, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v105
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v71, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_xor_b32_e32 v71, s68, v108
	s_nop 0
	v_cndmask_b32_e32 v32, v106, v107, vcc
	v_ashrrev_i32_e32 v106, 31, v69
	v_mov_b32_e32 v107, v106
	v_xor_b32_e32 v32, v32, v71
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[106:107]
	v_sub_u32_e32 v32, v32, v71
	v_xor_b32_e32 v107, v68, v106
	v_xad_u32 v71, v32, v106, v34
	v_xor_b32_e32 v105, v69, v106
	v_mad_u64_u32 v[68:69], s[14:15], v107, s22, 0
	v_mul_hi_u32 v32, v107, s77
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[68:69]
	v_mad_u64_u32 v[110:111], s[14:15], v105, s77, 0
	v_add_co_u32_e32 v32, vcc, v68, v110
	v_mad_u64_u32 v[108:109], s[14:15], v105, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v69, v111, vcc
	s_nop 1
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[108:109]
	v_mul_lo_u32 v32, s35, v68
	v_mul_lo_u32 v108, s34, v69
	v_mad_u64_u32 v[68:69], s[14:15], s34, v68, 0
	v_add3_u32 v32, v69, v108, v32
	v_sub_u32_e32 v69, v105, v32
	v_sub_co_u32_e32 v68, vcc, v107, v68
	s_nop 1
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, vcc
	v_subrev_co_u32_e64 v107, s[14:15], s34, v68
	v_cmp_le_u32_e64 s[28:29], s34, v107
	s_nop 0
	v_subbrev_co_u32_e64 v108, s[24:25], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v108
	v_cndmask_b32_e64 v110, 0, -1, s[28:29]
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, s[14:15]
	v_cndmask_b32_e64 v109, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v108
	v_subb_co_u32_e32 v32, vcc, v105, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v109, v109, v110, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v109
	v_subrev_co_u32_e64 v109, s[14:15], s34, v107
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v68
	v_cndmask_b32_e64 v69, v108, v69, s[24:25]
	v_cndmask_b32_e64 v105, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v108, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x408
	v_cndmask_b32_e32 v105, v105, v108, vcc
	v_cmp_ne_u32_e32 vcc, 0, v105
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v69, vcc
	v_cndmask_b32_e64 v69, v107, v109, s[24:25]
	v_cndmask_b32_e32 v68, v68, v69, vcc
	v_xor_b32_e32 v68, v68, v106
	v_xor_b32_e32 v32, v32, v106
	v_sub_co_u32_e32 v68, vcc, v68, v106
	v_mul_lo_u32 v69, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v106, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v68, v69
	v_lshl_add_u64 v[68:69], v[64:65], 0, s[14:15]
	s_movk_i32 s14, 0xfbf7
	s_mov_b32 s15, -1
	v_cmp_gt_i64_e32 vcc, -8, v[66:67]
	v_lshl_add_u64 v[106:107], v[62:63], 0, s[14:15]
	buffer_load_ubyte v105, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v107, v69, v107, vcc
	v_ashrrev_i32_e32 v108, 31, v107
	v_cndmask_b32_e32 v106, v68, v106, vcc
	v_mov_b32_e32 v109, v108
	v_lshl_add_u64 v[106:107], v[106:107], 0, v[108:109]
	v_xor_b32_e32 v109, v106, v108
	v_xor_b32_e32 v71, v107, v108
	v_mad_u64_u32 v[106:107], s[14:15], v109, s22, 0
	v_mul_hi_u32 v32, v109, s77
	v_lshl_add_u64 v[106:107], v[32:33], 0, v[106:107]
	v_mad_u64_u32 v[112:113], s[14:15], v71, s77, 0
	v_add_co_u32_e32 v32, vcc, v106, v112
	v_mad_u64_u32 v[110:111], s[14:15], v71, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v107, v113, vcc
	s_nop 1
	v_addc_co_u32_e32 v111, vcc, 0, v111, vcc
	v_lshl_add_u64 v[106:107], v[32:33], 0, v[110:111]
	v_mul_lo_u32 v32, s35, v106
	v_mul_lo_u32 v107, s34, v107
	v_mad_u64_u32 v[110:111], s[14:15], s34, v106, 0
	v_add3_u32 v32, v111, v107, v32
	v_sub_u32_e32 v107, v71, v32
	v_sub_co_u32_e32 v109, vcc, v109, v110
	s_nop 1
	v_subb_co_u32_e64 v107, s[14:15], v107, v70, vcc
	v_subrev_co_u32_e64 v110, s[14:15], s34, v109
	v_cmp_le_u32_e64 s[24:25], s34, v110
	s_nop 0
	v_subbrev_co_u32_e64 v107, s[14:15], 0, v107, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v107
	v_add_u32_e32 v110, 1, v106
	v_subb_co_u32_e32 v32, vcc, v71, v32, vcc
	v_cndmask_b32_e64 v111, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v107
	v_cndmask_b32_e64 v107, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v107, v111, v107, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v107
	v_add_u32_e32 v107, 2, v106
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cndmask_b32_e64 v107, v110, v107, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v109
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v71, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_xor_b32_e32 v71, s68, v108
	s_nop 0
	v_cndmask_b32_e32 v32, v106, v107, vcc
	v_ashrrev_i32_e32 v106, 31, v69
	v_mov_b32_e32 v107, v106
	v_xor_b32_e32 v32, v32, v71
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[106:107]
	v_sub_u32_e32 v32, v32, v71
	v_xor_b32_e32 v112, v68, v106
	v_xad_u32 v71, v32, v106, v34
	v_xor_b32_e32 v107, v69, v106
	v_mad_u64_u32 v[68:69], s[14:15], v112, s22, 0
	v_mul_hi_u32 v32, v112, s77
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[68:69]
	v_mad_u64_u32 v[110:111], s[14:15], v107, s77, 0
	v_add_co_u32_e32 v32, vcc, v68, v110
	v_mad_u64_u32 v[108:109], s[14:15], v107, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v69, v111, vcc
	s_nop 1
	v_addc_co_u32_e32 v109, vcc, 0, v109, vcc
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[108:109]
	v_mul_lo_u32 v32, s35, v68
	v_mul_lo_u32 v108, s34, v69
	v_mad_u64_u32 v[68:69], s[14:15], s34, v68, 0
	v_add3_u32 v32, v69, v108, v32
	v_sub_u32_e32 v69, v107, v32
	v_sub_co_u32_e32 v68, vcc, v112, v68
	s_nop 1
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, vcc
	v_subrev_co_u32_e64 v108, s[14:15], s34, v68
	v_cmp_le_u32_e64 s[28:29], s34, v108
	s_nop 0
	v_subbrev_co_u32_e64 v109, s[24:25], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v109
	v_cndmask_b32_e64 v111, 0, -1, s[28:29]
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, s[14:15]
	v_cndmask_b32_e64 v110, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v109
	v_subb_co_u32_e32 v32, vcc, v107, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v110, v110, v111, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v110
	v_subrev_co_u32_e64 v110, s[14:15], s34, v108
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v68
	v_cndmask_b32_e64 v69, v109, v69, s[24:25]
	v_cndmask_b32_e64 v107, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v109, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x409
	v_cndmask_b32_e32 v107, v107, v109, vcc
	v_cmp_ne_u32_e32 vcc, 0, v107
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v69, vcc
	v_cndmask_b32_e64 v69, v108, v110, s[24:25]
	v_cndmask_b32_e32 v68, v68, v69, vcc
	v_xor_b32_e32 v68, v68, v106
	v_xor_b32_e32 v32, v32, v106
	v_sub_co_u32_e32 v68, vcc, v68, v106
	v_mul_lo_u32 v69, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v106, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v68, v69
	v_lshl_add_u64 v[68:69], v[64:65], 0, s[14:15]
	s_movk_i32 s14, 0xfbf6
	s_mov_b32 s15, -1
	v_cmp_gt_i64_e32 vcc, -9, v[66:67]
	v_lshl_add_u64 v[108:109], v[62:63], 0, s[14:15]
	buffer_load_ubyte v106, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v109, v69, v109, vcc
	v_ashrrev_i32_e32 v110, 31, v109
	v_cndmask_b32_e32 v108, v68, v108, vcc
	v_mov_b32_e32 v111, v110
	v_lshl_add_u64 v[108:109], v[108:109], 0, v[110:111]
	v_xor_b32_e32 v107, v108, v110
	v_xor_b32_e32 v71, v109, v110
	v_mad_u64_u32 v[108:109], s[14:15], v107, s22, 0
	v_mul_hi_u32 v32, v107, s77
	v_lshl_add_u64 v[108:109], v[32:33], 0, v[108:109]
	v_mad_u64_u32 v[114:115], s[14:15], v71, s77, 0
	v_add_co_u32_e32 v32, vcc, v108, v114
	v_mad_u64_u32 v[112:113], s[14:15], v71, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v109, v115, vcc
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	v_lshl_add_u64 v[108:109], v[32:33], 0, v[112:113]
	v_mul_lo_u32 v32, s35, v108
	v_mul_lo_u32 v109, s34, v109
	v_mad_u64_u32 v[112:113], s[14:15], s34, v108, 0
	v_add3_u32 v32, v113, v109, v32
	v_sub_u32_e32 v109, v71, v32
	v_sub_co_u32_e32 v107, vcc, v107, v112
	s_nop 1
	v_subb_co_u32_e64 v109, s[14:15], v109, v70, vcc
	v_subrev_co_u32_e64 v111, s[14:15], s34, v107
	v_cmp_le_u32_e64 s[24:25], s34, v111
	s_nop 0
	v_subbrev_co_u32_e64 v109, s[14:15], 0, v109, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v109
	v_add_u32_e32 v111, 1, v108
	v_subb_co_u32_e32 v32, vcc, v71, v32, vcc
	v_cndmask_b32_e64 v112, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v109
	v_cndmask_b32_e64 v109, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v109, v112, v109, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v109
	v_add_u32_e32 v109, 2, v108
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cndmask_b32_e64 v109, v111, v109, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v107
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v71, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_xor_b32_e32 v71, s68, v110
	s_nop 0
	v_cndmask_b32_e32 v32, v108, v109, vcc
	v_ashrrev_i32_e32 v108, 31, v69
	v_mov_b32_e32 v109, v108
	v_xor_b32_e32 v32, v32, v71
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[108:109]
	v_sub_u32_e32 v32, v32, v71
	v_xor_b32_e32 v109, v68, v108
	v_xad_u32 v71, v32, v108, v34
	v_xor_b32_e32 v107, v69, v108
	v_mad_u64_u32 v[68:69], s[14:15], v109, s22, 0
	v_mul_hi_u32 v32, v109, s77
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[68:69]
	v_mad_u64_u32 v[112:113], s[14:15], v107, s77, 0
	v_add_co_u32_e32 v32, vcc, v68, v112
	v_mad_u64_u32 v[110:111], s[14:15], v107, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v69, v113, vcc
	s_nop 1
	v_addc_co_u32_e32 v111, vcc, 0, v111, vcc
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[110:111]
	v_mul_lo_u32 v32, s35, v68
	v_mul_lo_u32 v110, s34, v69
	v_mad_u64_u32 v[68:69], s[14:15], s34, v68, 0
	v_add3_u32 v32, v69, v110, v32
	v_sub_u32_e32 v69, v107, v32
	v_sub_co_u32_e32 v68, vcc, v109, v68
	s_nop 1
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, vcc
	v_subrev_co_u32_e64 v109, s[14:15], s34, v68
	v_cmp_le_u32_e64 s[28:29], s34, v109
	s_nop 0
	v_subbrev_co_u32_e64 v110, s[24:25], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v110
	v_cndmask_b32_e64 v112, 0, -1, s[28:29]
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, s[14:15]
	v_cndmask_b32_e64 v111, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v110
	v_subb_co_u32_e32 v32, vcc, v107, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v111, v111, v112, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v111
	v_subrev_co_u32_e64 v111, s[14:15], s34, v109
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v68
	v_cndmask_b32_e64 v69, v110, v69, s[24:25]
	v_cndmask_b32_e64 v107, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v110, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x40a
	v_cndmask_b32_e32 v107, v107, v110, vcc
	v_cmp_ne_u32_e32 vcc, 0, v107
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v69, vcc
	v_cndmask_b32_e64 v69, v109, v111, s[24:25]
	v_cndmask_b32_e32 v68, v68, v69, vcc
	v_xor_b32_e32 v68, v68, v108
	v_xor_b32_e32 v32, v32, v108
	v_sub_co_u32_e32 v68, vcc, v68, v108
	v_mul_lo_u32 v69, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v108, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v68, v69
	v_lshl_add_u64 v[68:69], v[64:65], 0, s[14:15]
	s_movk_i32 s14, 0xfbf5
	s_mov_b32 s15, -1
	v_cmp_gt_i64_e32 vcc, -10, v[66:67]
	v_lshl_add_u64 v[108:109], v[62:63], 0, s[14:15]
	buffer_load_ubyte v107, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v109, v69, v109, vcc
	v_ashrrev_i32_e32 v110, 31, v109
	v_cndmask_b32_e32 v108, v68, v108, vcc
	v_mov_b32_e32 v111, v110
	v_lshl_add_u64 v[108:109], v[108:109], 0, v[110:111]
	v_xor_b32_e32 v111, v108, v110
	v_xor_b32_e32 v71, v109, v110
	v_mad_u64_u32 v[108:109], s[14:15], v111, s22, 0
	v_mul_hi_u32 v32, v111, s77
	v_lshl_add_u64 v[108:109], v[32:33], 0, v[108:109]
	v_mad_u64_u32 v[114:115], s[14:15], v71, s77, 0
	v_add_co_u32_e32 v32, vcc, v108, v114
	v_mad_u64_u32 v[112:113], s[14:15], v71, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v109, v115, vcc
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	v_lshl_add_u64 v[108:109], v[32:33], 0, v[112:113]
	v_mul_lo_u32 v32, s35, v108
	v_mul_lo_u32 v109, s34, v109
	v_mad_u64_u32 v[112:113], s[14:15], s34, v108, 0
	v_add3_u32 v32, v113, v109, v32
	v_sub_u32_e32 v109, v71, v32
	v_sub_co_u32_e32 v111, vcc, v111, v112
	s_nop 1
	v_subb_co_u32_e64 v109, s[14:15], v109, v70, vcc
	v_subrev_co_u32_e64 v112, s[14:15], s34, v111
	v_cmp_le_u32_e64 s[24:25], s34, v112
	s_nop 0
	v_subbrev_co_u32_e64 v109, s[14:15], 0, v109, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v109
	v_add_u32_e32 v112, 1, v108
	v_subb_co_u32_e32 v32, vcc, v71, v32, vcc
	v_cndmask_b32_e64 v113, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v109
	v_cndmask_b32_e64 v109, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v109, v113, v109, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v109
	v_add_u32_e32 v109, 2, v108
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cndmask_b32_e64 v109, v112, v109, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v111
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v71, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_xor_b32_e32 v71, s68, v110
	s_nop 0
	v_cndmask_b32_e32 v32, v108, v109, vcc
	v_ashrrev_i32_e32 v108, 31, v69
	v_mov_b32_e32 v109, v108
	v_xor_b32_e32 v32, v32, v71
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[108:109]
	v_sub_u32_e32 v32, v32, v71
	v_xor_b32_e32 v114, v68, v108
	v_xad_u32 v71, v32, v108, v34
	v_xor_b32_e32 v109, v69, v108
	v_mad_u64_u32 v[68:69], s[14:15], v114, s22, 0
	v_mul_hi_u32 v32, v114, s77
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[68:69]
	v_mad_u64_u32 v[112:113], s[14:15], v109, s77, 0
	v_add_co_u32_e32 v32, vcc, v68, v112
	v_mad_u64_u32 v[110:111], s[14:15], v109, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v69, v113, vcc
	s_nop 1
	v_addc_co_u32_e32 v111, vcc, 0, v111, vcc
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[110:111]
	v_mul_lo_u32 v32, s35, v68
	v_mul_lo_u32 v110, s34, v69
	v_mad_u64_u32 v[68:69], s[14:15], s34, v68, 0
	v_add3_u32 v32, v69, v110, v32
	v_sub_u32_e32 v69, v109, v32
	v_sub_co_u32_e32 v68, vcc, v114, v68
	s_nop 1
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, vcc
	v_subrev_co_u32_e64 v110, s[14:15], s34, v68
	v_cmp_le_u32_e64 s[28:29], s34, v110
	s_nop 0
	v_subbrev_co_u32_e64 v111, s[24:25], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v111
	v_cndmask_b32_e64 v113, 0, -1, s[28:29]
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, s[14:15]
	v_cndmask_b32_e64 v112, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v111
	v_subb_co_u32_e32 v32, vcc, v109, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v112, v112, v113, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v112
	v_subrev_co_u32_e64 v112, s[14:15], s34, v110
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v68
	v_cndmask_b32_e64 v69, v111, v69, s[24:25]
	v_cndmask_b32_e64 v109, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v111, 0, -1, s[14:15]
	s_mov_b64 s[14:15], 0x40b
	v_cndmask_b32_e32 v109, v109, v111, vcc
	v_cmp_ne_u32_e32 vcc, 0, v109
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v69, vcc
	v_cndmask_b32_e64 v69, v110, v112, s[24:25]
	v_cndmask_b32_e32 v68, v68, v69, vcc
	v_xor_b32_e32 v68, v68, v108
	v_xor_b32_e32 v32, v32, v108
	v_sub_co_u32_e32 v68, vcc, v68, v108
	v_mul_lo_u32 v69, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v108, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v68, v69
	v_lshl_add_u64 v[68:69], v[64:65], 0, s[14:15]
	v_cmp_gt_i64_e32 vcc, -11, v[66:67]
	v_lshl_add_u64 v[110:111], v[62:63], 0, s[74:75]
	buffer_load_ubyte v108, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v111, v69, v111, vcc
	v_ashrrev_i32_e32 v112, 31, v111
	v_cndmask_b32_e32 v110, v68, v110, vcc
	v_mov_b32_e32 v113, v112
	v_lshl_add_u64 v[110:111], v[110:111], 0, v[112:113]
	v_xor_b32_e32 v109, v110, v112
	v_xor_b32_e32 v71, v111, v112
	v_mad_u64_u32 v[110:111], s[14:15], v109, s22, 0
	v_mul_hi_u32 v32, v109, s77
	v_lshl_add_u64 v[110:111], v[32:33], 0, v[110:111]
	v_mad_u64_u32 v[180:181], s[14:15], v71, s77, 0
	v_add_co_u32_e32 v32, vcc, v110, v180
	v_mad_u64_u32 v[114:115], s[14:15], v71, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v111, v181, vcc
	s_nop 1
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	v_lshl_add_u64 v[110:111], v[32:33], 0, v[114:115]
	v_mul_lo_u32 v32, s35, v110
	v_mul_lo_u32 v111, s34, v111
	v_mad_u64_u32 v[114:115], s[14:15], s34, v110, 0
	v_add3_u32 v32, v115, v111, v32
	v_sub_u32_e32 v111, v71, v32
	v_sub_co_u32_e32 v109, vcc, v109, v114
	s_nop 1
	v_subb_co_u32_e64 v111, s[14:15], v111, v70, vcc
	v_subrev_co_u32_e64 v113, s[14:15], s34, v109
	v_cmp_le_u32_e64 s[24:25], s34, v113
	s_nop 0
	v_subbrev_co_u32_e64 v111, s[14:15], 0, v111, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v111
	v_add_u32_e32 v113, 1, v110
	v_subb_co_u32_e32 v32, vcc, v71, v32, vcc
	v_cndmask_b32_e64 v114, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v111
	v_cndmask_b32_e64 v111, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v111, v114, v111, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v111
	v_add_u32_e32 v111, 2, v110
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cndmask_b32_e64 v111, v113, v111, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v109
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v71, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_xor_b32_e32 v71, s68, v112
	s_nop 0
	v_cndmask_b32_e32 v32, v110, v111, vcc
	v_ashrrev_i32_e32 v110, 31, v69
	v_mov_b32_e32 v111, v110
	v_xor_b32_e32 v32, v32, v71
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[110:111]
	v_sub_u32_e32 v32, v32, v71
	v_xor_b32_e32 v111, v68, v110
	v_xad_u32 v71, v32, v110, v34
	v_xor_b32_e32 v109, v69, v110
	v_mad_u64_u32 v[68:69], s[14:15], v111, s22, 0
	v_mul_hi_u32 v32, v111, s77
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[68:69]
	v_mad_u64_u32 v[114:115], s[14:15], v109, s77, 0
	v_add_co_u32_e32 v32, vcc, v68, v114
	v_mad_u64_u32 v[112:113], s[14:15], v109, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v69, v115, vcc
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[112:113]
	v_mul_lo_u32 v32, s35, v68
	v_mul_lo_u32 v112, s34, v69
	v_mad_u64_u32 v[68:69], s[14:15], s34, v68, 0
	v_add3_u32 v32, v69, v112, v32
	v_sub_u32_e32 v69, v109, v32
	v_sub_co_u32_e32 v68, vcc, v111, v68
	s_nop 1
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, vcc
	v_subrev_co_u32_e64 v111, s[14:15], s34, v68
	v_cmp_le_u32_e64 s[28:29], s34, v111
	s_nop 0
	v_subbrev_co_u32_e64 v112, s[24:25], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v112
	v_cndmask_b32_e64 v114, 0, -1, s[28:29]
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, s[14:15]
	v_cndmask_b32_e64 v113, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v112
	v_subb_co_u32_e32 v32, vcc, v109, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v113, v113, v114, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v113
	v_subrev_co_u32_e64 v113, s[14:15], s34, v111
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v68
	v_cndmask_b32_e64 v69, v112, v69, s[24:25]
	v_cndmask_b32_e64 v109, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v112, 0, -1, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v109, v109, v112, vcc
	v_cmp_ne_u32_e32 vcc, 0, v109
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v69, vcc
	v_cndmask_b32_e64 v69, v111, v113, s[24:25]
	v_cndmask_b32_e32 v68, v68, v69, vcc
	v_xor_b32_e32 v68, v68, v110
	v_xor_b32_e32 v32, v32, v110
	v_sub_co_u32_e32 v68, vcc, v68, v110
	v_mul_lo_u32 v69, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v110, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v68, v69
	v_lshl_add_u64 v[68:69], v[64:65], 0, s[80:81]
	v_cmp_gt_i64_e32 vcc, -12, v[66:67]
	v_lshl_add_u64 v[110:111], v[62:63], 0, s[82:83]
	buffer_load_ubyte v109, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v111, v69, v111, vcc
	v_ashrrev_i32_e32 v112, 31, v111
	v_cndmask_b32_e32 v110, v68, v110, vcc
	v_mov_b32_e32 v113, v112
	v_lshl_add_u64 v[110:111], v[110:111], 0, v[112:113]
	v_xor_b32_e32 v113, v110, v112
	v_xor_b32_e32 v71, v111, v112
	v_mad_u64_u32 v[110:111], s[14:15], v113, s22, 0
	v_mul_hi_u32 v32, v113, s77
	v_lshl_add_u64 v[110:111], v[32:33], 0, v[110:111]
	v_mad_u64_u32 v[180:181], s[14:15], v71, s77, 0
	v_add_co_u32_e32 v32, vcc, v110, v180
	v_mad_u64_u32 v[114:115], s[14:15], v71, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v111, v181, vcc
	s_nop 1
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	v_lshl_add_u64 v[110:111], v[32:33], 0, v[114:115]
	v_mul_lo_u32 v32, s35, v110
	v_mul_lo_u32 v111, s34, v111
	v_mad_u64_u32 v[114:115], s[14:15], s34, v110, 0
	v_add3_u32 v32, v115, v111, v32
	v_sub_u32_e32 v111, v71, v32
	v_sub_co_u32_e32 v113, vcc, v113, v114
	s_nop 1
	v_subb_co_u32_e64 v111, s[14:15], v111, v70, vcc
	v_subrev_co_u32_e64 v114, s[14:15], s34, v113
	v_cmp_le_u32_e64 s[24:25], s34, v114
	s_nop 0
	v_subbrev_co_u32_e64 v111, s[14:15], 0, v111, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v111
	v_add_u32_e32 v114, 1, v110
	v_subb_co_u32_e32 v32, vcc, v71, v32, vcc
	v_cndmask_b32_e64 v115, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v111
	v_cndmask_b32_e64 v111, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v111, v115, v111, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v111
	v_add_u32_e32 v111, 2, v110
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cndmask_b32_e64 v111, v114, v111, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v113
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v71, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_xor_b32_e32 v71, s68, v112
	s_nop 0
	v_cndmask_b32_e32 v32, v110, v111, vcc
	v_ashrrev_i32_e32 v110, 31, v69
	v_mov_b32_e32 v111, v110
	v_xor_b32_e32 v32, v32, v71
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[110:111]
	v_sub_u32_e32 v32, v32, v71
	v_xor_b32_e32 v180, v68, v110
	v_xad_u32 v71, v32, v110, v34
	v_xor_b32_e32 v111, v69, v110
	v_mad_u64_u32 v[68:69], s[14:15], v180, s22, 0
	v_mul_hi_u32 v32, v180, s77
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[68:69]
	v_mad_u64_u32 v[114:115], s[14:15], v111, s77, 0
	v_add_co_u32_e32 v32, vcc, v68, v114
	v_mad_u64_u32 v[112:113], s[14:15], v111, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v69, v115, vcc
	s_nop 1
	v_addc_co_u32_e32 v113, vcc, 0, v113, vcc
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[112:113]
	v_mul_lo_u32 v32, s35, v68
	v_mul_lo_u32 v112, s34, v69
	v_mad_u64_u32 v[68:69], s[14:15], s34, v68, 0
	v_add3_u32 v32, v69, v112, v32
	v_sub_u32_e32 v69, v111, v32
	v_sub_co_u32_e32 v68, vcc, v180, v68
	s_nop 1
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, vcc
	v_subrev_co_u32_e64 v112, s[14:15], s34, v68
	v_cmp_le_u32_e64 s[28:29], s34, v112
	s_nop 0
	v_subbrev_co_u32_e64 v113, s[24:25], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v113
	v_cndmask_b32_e64 v115, 0, -1, s[28:29]
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, s[14:15]
	v_cndmask_b32_e64 v114, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v113
	v_subb_co_u32_e32 v32, vcc, v111, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v114, v114, v115, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v114
	v_subrev_co_u32_e64 v114, s[14:15], s34, v112
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v68
	v_cndmask_b32_e64 v69, v113, v69, s[24:25]
	v_cndmask_b32_e64 v111, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v113, 0, -1, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v111, v111, v113, vcc
	v_cmp_ne_u32_e32 vcc, 0, v111
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v69, vcc
	v_cndmask_b32_e64 v69, v112, v114, s[24:25]
	v_cndmask_b32_e32 v68, v68, v69, vcc
	v_xor_b32_e32 v68, v68, v110
	v_xor_b32_e32 v32, v32, v110
	v_sub_co_u32_e32 v68, vcc, v68, v110
	v_mul_lo_u32 v69, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v110, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v68, v69
	v_lshl_add_u64 v[68:69], v[64:65], 0, s[72:73]
	v_cmp_gt_i64_e32 vcc, -13, v[66:67]
	v_lshl_add_u64 v[112:113], v[62:63], 0, s[86:87]
	buffer_load_ubyte v110, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v113, v69, v113, vcc
	v_ashrrev_i32_e32 v114, 31, v113
	v_cndmask_b32_e32 v112, v68, v112, vcc
	v_mov_b32_e32 v115, v114
	v_lshl_add_u64 v[112:113], v[112:113], 0, v[114:115]
	v_xor_b32_e32 v111, v112, v114
	v_xor_b32_e32 v71, v113, v114
	v_mad_u64_u32 v[112:113], s[14:15], v111, s22, 0
	v_mul_hi_u32 v32, v111, s77
	v_lshl_add_u64 v[112:113], v[32:33], 0, v[112:113]
	v_mad_u64_u32 v[182:183], s[14:15], v71, s77, 0
	v_add_co_u32_e32 v32, vcc, v112, v182
	v_mad_u64_u32 v[180:181], s[14:15], v71, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v113, v183, vcc
	s_nop 1
	v_addc_co_u32_e32 v181, vcc, 0, v181, vcc
	v_lshl_add_u64 v[112:113], v[32:33], 0, v[180:181]
	v_mul_lo_u32 v32, s35, v112
	v_mul_lo_u32 v113, s34, v113
	v_mad_u64_u32 v[180:181], s[14:15], s34, v112, 0
	v_add3_u32 v32, v181, v113, v32
	v_sub_u32_e32 v113, v71, v32
	v_sub_co_u32_e32 v111, vcc, v111, v180
	s_nop 1
	v_subb_co_u32_e64 v113, s[14:15], v113, v70, vcc
	v_subrev_co_u32_e64 v115, s[14:15], s34, v111
	v_cmp_le_u32_e64 s[24:25], s34, v115
	s_nop 0
	v_subbrev_co_u32_e64 v113, s[14:15], 0, v113, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v113
	v_add_u32_e32 v115, 1, v112
	v_subb_co_u32_e32 v32, vcc, v71, v32, vcc
	v_cndmask_b32_e64 v180, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v113
	v_cndmask_b32_e64 v113, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v113, v180, v113, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v113
	v_add_u32_e32 v113, 2, v112
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cndmask_b32_e64 v113, v115, v113, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v111
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v71, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	v_xor_b32_e32 v71, s68, v114
	s_nop 0
	v_cndmask_b32_e32 v32, v112, v113, vcc
	v_ashrrev_i32_e32 v112, 31, v69
	v_mov_b32_e32 v113, v112
	v_xor_b32_e32 v32, v32, v71
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[112:113]
	v_sub_u32_e32 v32, v32, v71
	v_xor_b32_e32 v113, v68, v112
	v_xad_u32 v71, v32, v112, v34
	v_xor_b32_e32 v111, v69, v112
	v_mad_u64_u32 v[68:69], s[14:15], v113, s22, 0
	v_mul_hi_u32 v32, v113, s77
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[68:69]
	v_mad_u64_u32 v[180:181], s[14:15], v111, s77, 0
	v_add_co_u32_e32 v32, vcc, v68, v180
	v_mad_u64_u32 v[114:115], s[14:15], v111, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v69, v181, vcc
	s_nop 1
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[114:115]
	v_mul_lo_u32 v32, s35, v68
	v_mul_lo_u32 v114, s34, v69
	v_mad_u64_u32 v[68:69], s[14:15], s34, v68, 0
	v_add3_u32 v32, v69, v114, v32
	v_sub_u32_e32 v69, v111, v32
	v_sub_co_u32_e32 v68, vcc, v113, v68
	s_nop 1
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, vcc
	v_subrev_co_u32_e64 v113, s[14:15], s34, v68
	v_cmp_le_u32_e64 s[28:29], s34, v113
	s_nop 0
	v_subbrev_co_u32_e64 v114, s[24:25], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v114
	v_cndmask_b32_e64 v180, 0, -1, s[28:29]
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, s[14:15]
	v_cndmask_b32_e64 v115, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v114
	v_subb_co_u32_e32 v32, vcc, v111, v32, vcc
	s_nop 0
	v_cndmask_b32_e64 v115, v115, v180, s[24:25]
	v_cmp_ne_u32_e64 s[24:25], 0, v115
	v_subrev_co_u32_e64 v115, s[14:15], s34, v113
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v68
	v_cndmask_b32_e64 v69, v114, v69, s[24:25]
	v_cndmask_b32_e64 v111, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v114, 0, -1, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v111, v111, v114, vcc
	v_cmp_ne_u32_e32 vcc, 0, v111
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v69, vcc
	v_cndmask_b32_e64 v69, v113, v115, s[24:25]
	v_cndmask_b32_e32 v68, v68, v69, vcc
	v_xor_b32_e32 v68, v68, v112
	v_xor_b32_e32 v32, v32, v112
	v_sub_co_u32_e32 v68, vcc, v68, v112
	v_mul_lo_u32 v69, v71, s33
	s_nop 0
	v_subb_co_u32_e32 v32, vcc, v32, v112, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_add3_u32 v32, v32, v68, v69
	v_lshl_add_u64 v[112:113], v[64:65], 0, s[88:89]
	v_cmp_gt_i64_e32 vcc, -14, v[66:67]
	v_lshl_add_u64 v[68:69], v[62:63], 0, s[90:91]
	buffer_load_ubyte v111, v32, s[36:39], 0 offen
	v_cndmask_b32_e32 v69, v113, v69, vcc
	v_ashrrev_i32_e32 v114, 31, v69
	v_cndmask_b32_e32 v68, v112, v68, vcc
	v_mov_b32_e32 v115, v114
	v_lshl_add_u64 v[68:69], v[68:69], 0, v[114:115]
	v_xor_b32_e32 v115, v68, v114
	v_xor_b32_e32 v71, v69, v114
	v_mad_u64_u32 v[68:69], s[14:15], v115, s22, 0
	v_mul_hi_u32 v32, v115, s77
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[68:69]
	v_mad_u64_u32 v[182:183], s[14:15], v71, s77, 0
	v_add_co_u32_e32 v32, vcc, v68, v182
	v_mad_u64_u32 v[180:181], s[14:15], v71, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v69, v183, vcc
	v_lshl_add_u64 v[64:65], v[64:65], 0, s[92:93]
	s_nop 0
	v_addc_co_u32_e32 v181, vcc, 0, v181, vcc
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[180:181]
	v_mul_lo_u32 v32, s35, v68
	v_mul_lo_u32 v69, s34, v69
	v_mad_u64_u32 v[180:181], s[14:15], s34, v68, 0
	v_add3_u32 v32, v181, v69, v32
	v_sub_u32_e32 v69, v71, v32
	v_sub_co_u32_e32 v115, vcc, v115, v180
	v_lshl_add_u64 v[62:63], v[62:63], 0, s[94:95]
	s_nop 0
	v_subb_co_u32_e64 v69, s[14:15], v69, v70, vcc
	v_subrev_co_u32_e64 v180, s[14:15], s34, v115
	v_cmp_le_u32_e64 s[24:25], s34, v180
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[14:15], 0, v69, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v69
	v_add_u32_e32 v180, 1, v68
	v_subb_co_u32_e32 v32, vcc, v71, v32, vcc
	v_cndmask_b32_e64 v181, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v69
	v_cndmask_b32_e64 v69, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v69, v181, v69, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v69
	v_add_u32_e32 v69, 2, v68
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cndmask_b32_e64 v69, v180, v69, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v115
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v71, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v68, v69, vcc
	v_xor_b32_e32 v68, s68, v114
	v_xor_b32_e32 v32, v32, v68
	v_sub_u32_e32 v32, v32, v68
	v_ashrrev_i32_e32 v68, 31, v113
	v_mov_b32_e32 v69, v68
	v_lshl_add_u64 v[112:113], v[112:113], 0, v[68:69]
	v_xor_b32_e32 v182, v112, v68
	v_xad_u32 v71, v32, v68, v34
	v_xor_b32_e32 v69, v113, v68
	v_mad_u64_u32 v[112:113], s[14:15], v182, s22, 0
	v_mul_hi_u32 v32, v182, s77
	v_lshl_add_u64 v[112:113], v[32:33], 0, v[112:113]
	v_mad_u64_u32 v[180:181], s[14:15], v69, s77, 0
	v_add_co_u32_e32 v32, vcc, v112, v180
	v_mad_u64_u32 v[114:115], s[14:15], v69, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v113, v181, vcc
	s_nop 1
	v_addc_co_u32_e32 v115, vcc, 0, v115, vcc
	v_lshl_add_u64 v[112:113], v[32:33], 0, v[114:115]
	v_mul_lo_u32 v32, s35, v112
	v_mul_lo_u32 v114, s34, v113
	v_mad_u64_u32 v[112:113], s[14:15], s34, v112, 0
	v_add3_u32 v32, v113, v114, v32
	v_sub_u32_e32 v113, v69, v32
	v_sub_co_u32_e32 v112, vcc, v182, v112
	v_subrev_co_u32_e64 v114, s[24:25], s34, v112
	s_nop 0
	v_subb_co_u32_e64 v113, s[14:15], v113, v70, vcc
	v_subbrev_co_u32_e64 v115, s[14:15], 0, v113, s[24:25]
	v_cmp_le_u32_e64 s[14:15], s35, v115
	v_cmp_le_u32_e64 s[28:29], s34, v114
	v_subb_co_u32_e64 v113, s[24:25], v113, v70, s[24:25]
	v_cndmask_b32_e64 v180, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v115
	v_cndmask_b32_e64 v181, 0, -1, s[28:29]
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	v_cndmask_b32_e64 v180, v180, v181, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v180
	v_subrev_co_u32_e64 v180, s[24:25], s34, v114
	v_cmp_le_u32_e32 vcc, s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v113, s[24:25], 0, v113, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s34, v112
	v_cndmask_b32_e64 v113, v115, v113, s[14:15]
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v115, 0, -1, s[24:25]
	s_nop 0
	v_cndmask_b32_e32 v69, v69, v115, vcc
	v_cmp_ne_u32_e32 vcc, 0, v69
	v_cndmask_b32_e64 v69, v114, v180, s[14:15]
	s_nop 0
	v_cndmask_b32_e32 v69, v112, v69, vcc
	v_cndmask_b32_e32 v32, v32, v113, vcc
	v_xor_b32_e32 v69, v69, v68
	v_xor_b32_e32 v32, v32, v68
	v_sub_co_u32_e32 v69, vcc, v69, v68
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v68, vcc
	v_cmp_gt_i64_e32 vcc, -15, v[66:67]
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_cndmask_b32_e32 v63, v65, v63, vcc
	v_ashrrev_i32_e32 v66, 31, v63
	v_cndmask_b32_e32 v62, v64, v62, vcc
	v_mov_b32_e32 v67, v66
	v_mul_lo_u32 v68, v71, s33
	v_lshl_add_u64 v[62:63], v[62:63], 0, v[66:67]
	v_add3_u32 v32, v32, v69, v68
	v_xor_b32_e32 v71, v62, v66
	buffer_load_ubyte v112, v32, s[36:39], 0 offen
	v_xor_b32_e32 v67, v63, v66
	v_mad_u64_u32 v[62:63], s[14:15], v71, s22, 0
	v_mul_hi_u32 v32, v71, s77
	v_lshl_add_u64 v[62:63], v[32:33], 0, v[62:63]
	v_mad_u64_u32 v[114:115], s[14:15], v67, s77, 0
	v_add_co_u32_e32 v32, vcc, v62, v114
	v_mad_u64_u32 v[68:69], s[14:15], v67, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v63, v115, vcc
	s_nop 1
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[62:63], v[32:33], 0, v[68:69]
	v_mul_lo_u32 v32, s35, v62
	v_mul_lo_u32 v63, s34, v63
	v_mad_u64_u32 v[68:69], s[14:15], s34, v62, 0
	v_add3_u32 v32, v69, v63, v32
	v_sub_u32_e32 v63, v67, v32
	v_sub_co_u32_e32 v68, vcc, v71, v68
	s_nop 1
	v_subb_co_u32_e64 v63, s[14:15], v63, v70, vcc
	v_subrev_co_u32_e64 v69, s[14:15], s34, v68
	v_cmp_le_u32_e64 s[24:25], s34, v69
	s_nop 0
	v_subbrev_co_u32_e64 v63, s[14:15], 0, v63, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s35, v63
	v_add_u32_e32 v69, 1, v62
	v_subb_co_u32_e32 v32, vcc, v67, v32, vcc
	v_cndmask_b32_e64 v71, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v63
	v_cndmask_b32_e64 v63, 0, -1, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cndmask_b32_e64 v63, v71, v63, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v63
	v_add_u32_e32 v63, 2, v62
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cndmask_b32_e64 v63, v69, v63, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s34, v68
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 0
	v_cndmask_b32_e64 v32, 0, -1, s[14:15]
	v_cndmask_b32_e32 v32, v67, v32, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v62, v63, vcc
	v_xor_b32_e32 v62, s68, v66
	v_xor_b32_e32 v32, v32, v62
	v_sub_u32_e32 v32, v32, v62
	v_ashrrev_i32_e32 v62, 31, v65
	v_mov_b32_e32 v63, v62
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[62:63]
	v_xor_b32_e32 v67, v64, v62
	v_xad_u32 v66, v32, v62, v34
	v_xor_b32_e32 v63, v65, v62
	v_mad_u64_u32 v[64:65], s[14:15], v67, s22, 0
	v_mul_hi_u32 v32, v67, s77
	v_lshl_add_u64 v[68:69], v[32:33], 0, v[64:65]
	v_mad_u64_u32 v[114:115], s[14:15], v63, s77, 0
	v_add_co_u32_e32 v32, vcc, v68, v114
	v_mad_u64_u32 v[64:65], s[14:15], v63, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v32, vcc, v69, v115, vcc
	s_nop 1
	v_addc_co_u32_e32 v65, vcc, 0, v65, vcc
	v_lshl_add_u64 v[64:65], v[32:33], 0, v[64:65]
	v_mul_lo_u32 v32, s35, v64
	v_mul_lo_u32 v68, s34, v65
	v_mad_u64_u32 v[64:65], s[14:15], s34, v64, 0
	v_add3_u32 v32, v65, v68, v32
	v_sub_u32_e32 v65, v63, v32
	v_sub_co_u32_e64 v64, s[14:15], v67, v64
	v_subrev_co_u32_e64 v67, s[24:25], s34, v64
	s_nop 0
	v_subb_co_u32_e64 v65, vcc, v65, v70, s[14:15]
	v_subbrev_co_u32_e64 v68, vcc, 0, v65, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v68
	v_cmp_le_u32_e64 s[28:29], s34, v67
	v_subb_co_u32_e64 v65, s[24:25], v65, v70, s[24:25]
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v68
	v_cndmask_b32_e64 v71, 0, -1, s[28:29]
	v_subb_co_u32_e64 v32, s[14:15], v63, v32, s[14:15]
	v_cndmask_b32_e32 v69, v69, v71, vcc
	v_cmp_ne_u32_e32 vcc, 0, v69
	v_subrev_co_u32_e64 v69, s[24:25], s34, v67
	v_cmp_le_u32_e64 s[14:15], s35, v32
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[24:25], 0, v65, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s34, v64
	v_cndmask_b32_e32 v65, v68, v65, vcc
	v_cndmask_b32_e64 v63, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s35, v32
	v_cndmask_b32_e64 v68, 0, -1, s[24:25]
	s_nop 0
	v_cndmask_b32_e64 v63, v63, v68, s[14:15]
	v_cmp_ne_u32_e64 s[14:15], 0, v63
	v_cndmask_b32_e32 v63, v67, v69, vcc
	s_nop 0
	v_cndmask_b32_e64 v63, v64, v63, s[14:15]
	v_cndmask_b32_e64 v32, v32, v65, s[14:15]
	v_xor_b32_e32 v63, v63, v62
	v_xor_b32_e32 v32, v32, v62
	v_sub_co_u32_e32 v63, vcc, v63, v62
	s_nop 1
	v_subb_co_u32_e32 v32, vcc, v32, v62, vcc
	v_ashrrev_i32_e32 v32, 31, v32
	v_and_b32_e32 v32, s44, v32
	v_mul_lo_u32 v62, v66, s33
	v_add3_u32 v32, v32, v63, v62
	buffer_load_ubyte v113, v32, s[36:39], 0 offen
	v_lshl_add_u64 v[66:67], s[52:53], 0, v[52:53]
	v_or_b32_e32 v32, s67, v67
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[28:29], exec, s[14:15]
	s_cbranch_execz .LBB0_118
	s_add_u32 s14, s66, s76
	s_mov_b32 s77, s76
	s_addc_u32 s15, s67, s76
	s_xor_b64 s[34:35], s[14:15], s[76:77]
	v_cvt_f32_u32_e32 v32, s34
	v_cvt_f32_u32_e32 v62, s35
	s_sub_u32 s14, 0, s34
	s_subb_u32 s15, 0, s35
	v_ashrrev_i32_e32 v64, 31, v67
	v_fmac_f32_e32 v32, 0x4f800000, v62
	v_rcp_f32_e32 v32, v32
	v_mov_b32_e32 v65, v64
	v_mul_f32_e32 v32, 0x5f7ffffc, v32
	v_mul_f32_e32 v62, 0x2f800000, v32
	v_trunc_f32_e32 v62, v62
	v_fmac_f32_e32 v32, 0xcf800000, v62
	v_cvt_u32_f32_e32 v62, v62
	v_cvt_u32_f32_e32 v32, v32
	v_readfirstlane_b32 s22, v62
	v_readfirstlane_b32 s23, v32
	s_mul_i32 s24, s14, s22
	s_mul_hi_u32 s50, s14, s23
	s_mul_i32 s25, s15, s23
	s_add_i32 s24, s50, s24
	s_add_i32 s24, s24, s25
	s_mul_i32 s51, s14, s23
	s_mul_i32 s50, s23, s24
	s_mul_hi_u32 s77, s23, s51
	s_mul_hi_u32 s25, s23, s24
	s_add_u32 s50, s77, s50
	s_addc_u32 s25, 0, s25
	s_mul_hi_u32 vcc_lo, s22, s51
	s_mul_i32 s51, s22, s51
	s_add_u32 s50, s50, s51
	s_mul_hi_u32 s77, s22, s24
	s_addc_u32 s25, s25, vcc_lo
	s_addc_u32 s50, s77, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s24, s25, s24
	s_addc_u32 s25, 0, s50
	s_add_u32 s23, s23, s24
	s_addc_u32 s22, s22, s25
	s_mul_i32 s24, s14, s22
	s_mul_hi_u32 s25, s14, s23
	s_add_i32 s24, s25, s24
	s_mul_i32 s15, s15, s23
	s_add_i32 s24, s24, s15
	s_mul_i32 s14, s14, s23
	s_mul_hi_u32 s25, s22, s14
	s_mul_i32 s50, s22, s14
	s_mul_i32 s77, s23, s24
	s_mul_hi_u32 s14, s23, s14
	s_mul_hi_u32 s51, s23, s24
	s_add_u32 s14, s14, s77
	s_addc_u32 s51, 0, s51
	s_add_u32 s14, s14, s50
	s_mul_hi_u32 s15, s22, s24
	s_addc_u32 s14, s51, s25
	s_addc_u32 s15, s15, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s14, s14, s24
	s_addc_u32 s15, 0, s15
	s_add_u32 s23, s23, s14
	v_lshl_add_u64 v[62:63], v[66:67], 0, v[64:65]
	s_addc_u32 s22, s22, s15
	v_xor_b32_e32 v114, v62, v64
	v_xor_b32_e32 v65, v63, v64
	v_mad_u64_u32 v[62:63], s[14:15], v114, s22, 0
	v_mul_hi_u32 v32, v114, s23
	v_lshl_add_u64 v[62:63], v[32:33], 0, v[62:63]
	v_mad_u64_u32 v[70:71], s[14:15], v65, s23, 0
	v_add_co_u32_e32 v32, vcc, v62, v70
	v_mad_u64_u32 v[68:69], s[14:15], v65, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v62, vcc, v63, v71, vcc
	v_mov_b32_e32 v63, s65
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[62:63], v[62:63], 0, v[68:69]
	v_mul_lo_u32 v32, s35, v62
	v_mul_lo_u32 v68, s34, v63
	v_mad_u64_u32 v[62:63], s[14:15], s34, v62, 0
	v_add3_u32 v32, v63, v68, v32
	v_sub_u32_e32 v63, v65, v32
	v_mov_b32_e32 v68, s35
	v_sub_co_u32_e32 v62, vcc, v114, v62
	s_nop 1
	v_subb_co_u32_e64 v63, s[14:15], v63, v68, vcc
	v_subrev_co_u32_e64 v69, s[14:15], s34, v62
	v_subb_co_u32_e32 v32, vcc, v65, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v70, s[24:25], 0, v63, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v70
	v_subb_co_u32_e64 v63, s[14:15], v63, v68, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s34, v69
	v_subrev_co_u32_e64 v68, s[14:15], s34, v69
	s_nop 0
	v_cndmask_b32_e64 v114, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v70
	v_subbrev_co_u32_e64 v63, s[14:15], 0, v63, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v71, v71, v114, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cmp_ne_u32_e64 s[14:15], 0, v71
	s_nop 0
	v_cndmask_b32_e64 v65, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v62
	v_cndmask_b32_e64 v63, v70, v63, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 1
	v_cndmask_b32_e32 v65, v65, v70, vcc
	v_cmp_ne_u32_e32 vcc, 0, v65
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v63, vcc
	v_cndmask_b32_e64 v63, v69, v68, s[14:15]
	v_cndmask_b32_e32 v62, v62, v63, vcc
	v_xor_b32_e32 v62, v62, v64
	v_xor_b32_e32 v32, v32, v64
	v_sub_co_u32_e32 v62, vcc, v62, v64
	s_nop 1
	v_subb_co_u32_e32 v63, vcc, v32, v64, vcc
.LBB0_118:
	s_or_saveexec_b64 s[14:15], s[28:29]
	v_cvt_f32_u32_e32 v114, s66
	s_xor_b64 exec, exec, s[14:15]
	s_cbranch_execz .LBB0_120
	v_rcp_iflag_f32_e32 v32, v114
	s_sub_i32 s22, 0, s66
	v_mul_f32_e32 v32, 0x4f7ffffe, v32
	v_cvt_u32_f32_e32 v32, v32
	v_mul_lo_u32 v62, s22, v32
	v_mul_hi_u32 v62, v32, v62
	v_add_u32_e32 v32, v32, v62
	v_mul_hi_u32 v32, v66, v32
	v_mul_lo_u32 v32, v32, s66
	v_sub_u32_e32 v32, v66, v32
	v_subrev_u32_e32 v62, s66, v32
	v_cmp_le_u32_e32 vcc, s66, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v62, vcc
	v_subrev_u32_e32 v62, s66, v32
	v_cmp_le_u32_e32 vcc, s66, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v62, vcc
	v_mov_b64_e32 v[62:63], v[32:33]
.LBB0_120:
	s_or_b64 exec, exec, s[14:15]
	v_lshl_add_u64 v[64:65], s[52:53], 0, v[56:57]
	v_lshl_add_u64 v[68:69], v[64:65], 0, 2
	v_cmp_gt_i32_e32 vcc, 0, v65
	s_nop 1
	v_cndmask_b32_e32 v71, v69, v55, vcc
	v_or_b32_e32 v32, s67, v71
	v_cndmask_b32_e32 v70, v68, v54, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[24:25], exec, s[14:15]
	s_cbranch_execz .LBB0_122
	s_add_u32 s14, s66, s76
	s_mov_b32 s77, s76
	s_addc_u32 s15, s67, s76
	s_xor_b64 s[28:29], s[14:15], s[76:77]
	v_cvt_f32_u32_e32 v32, s28
	v_cvt_f32_u32_e32 v68, s29
	s_sub_u32 s14, 0, s28
	s_subb_u32 s15, 0, s29
	v_fmac_f32_e32 v32, 0x4f800000, v68
	v_rcp_f32_e32 v32, v32
	s_nop 0
	v_mul_f32_e32 v32, 0x5f7ffffc, v32
	v_mul_f32_e32 v68, 0x2f800000, v32
	v_trunc_f32_e32 v68, v68
	v_fmac_f32_e32 v32, 0xcf800000, v68
	v_cvt_u32_f32_e32 v68, v68
	v_cvt_u32_f32_e32 v32, v32
	v_readfirstlane_b32 s22, v68
	v_readfirstlane_b32 s23, v32
	s_mul_i32 s34, s14, s22
	s_mul_hi_u32 s50, s14, s23
	s_mul_i32 s35, s15, s23
	s_add_i32 s34, s50, s34
	s_add_i32 s34, s34, s35
	s_mul_i32 s51, s14, s23
	s_mul_i32 s50, s23, s34
	s_mul_hi_u32 s77, s23, s51
	s_mul_hi_u32 s35, s23, s34
	s_add_u32 s50, s77, s50
	s_addc_u32 s35, 0, s35
	s_mul_hi_u32 vcc_lo, s22, s51
	s_mul_i32 s51, s22, s51
	s_add_u32 s50, s50, s51
	s_mul_hi_u32 s77, s22, s34
	s_addc_u32 s35, s35, vcc_lo
	s_addc_u32 s50, s77, 0
	s_mul_i32 s34, s22, s34
	s_add_u32 s34, s35, s34
	s_addc_u32 s35, 0, s50
	s_add_u32 s23, s23, s34
	s_addc_u32 s22, s22, s35
	s_mul_i32 s34, s14, s22
	s_mul_hi_u32 s35, s14, s23
	s_add_i32 s34, s35, s34
	s_mul_i32 s15, s15, s23
	s_add_i32 s34, s34, s15
	s_mul_i32 s14, s14, s23
	s_mul_hi_u32 s35, s22, s14
	s_mul_i32 s50, s22, s14
	s_mul_i32 s77, s23, s34
	s_mul_hi_u32 s14, s23, s14
	s_mul_hi_u32 s51, s23, s34
	s_add_u32 s14, s14, s77
	s_addc_u32 s51, 0, s51
	s_add_u32 s14, s14, s50
	s_mul_hi_u32 s15, s22, s34
	s_addc_u32 s14, s51, s35
	s_addc_u32 s15, s15, 0
	s_mul_i32 s34, s22, s34
	s_add_u32 s14, s14, s34
	v_ashrrev_i32_e32 v68, 31, v71
	s_addc_u32 s15, 0, s15
	v_mov_b32_e32 v69, v68
	s_add_u32 s23, s23, s14
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[68:69]
	s_addc_u32 s22, s22, s15
	v_xor_b32_e32 v115, v70, v68
	v_xor_b32_e32 v69, v71, v68
	v_mad_u64_u32 v[70:71], s[14:15], v115, s22, 0
	v_mul_hi_u32 v32, v115, s23
	v_lshl_add_u64 v[70:71], v[32:33], 0, v[70:71]
	v_mad_u64_u32 v[182:183], s[14:15], v69, s23, 0
	v_add_co_u32_e32 v32, vcc, v70, v182
	v_mad_u64_u32 v[180:181], s[14:15], v69, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v70, vcc, v71, v183, vcc
	v_mov_b32_e32 v71, s65
	s_nop 0
	v_addc_co_u32_e32 v181, vcc, 0, v181, vcc
	v_lshl_add_u64 v[70:71], v[70:71], 0, v[180:181]
	v_mul_lo_u32 v32, s29, v70
	v_mul_lo_u32 v182, s28, v71
	v_mad_u64_u32 v[180:181], s[14:15], s28, v70, 0
	v_add3_u32 v32, v181, v182, v32
	v_sub_u32_e32 v181, v69, v32
	v_mov_b32_e32 v182, s29
	v_sub_co_u32_e32 v115, vcc, v115, v180
	v_xor_b32_e32 v68, s76, v68
	s_nop 0
	v_subb_co_u32_e64 v180, s[14:15], v181, v182, vcc
	v_subrev_co_u32_e64 v181, s[14:15], s28, v115
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v180, s[14:15], 0, v180, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s29, v180
	v_cmp_le_u32_e32 vcc, s29, v32
	s_nop 0
	v_cndmask_b32_e64 v182, 0, -1, s[14:15]
	v_cmp_le_u32_e64 s[14:15], s28, v181
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s28, v115
	v_cndmask_b32_e64 v181, 0, -1, s[14:15]
	v_cmp_eq_u32_e64 s[14:15], s29, v180
	s_nop 1
	v_cndmask_b32_e64 v184, v182, v181, s[14:15]
	v_lshl_add_u64 v[180:181], v[70:71], 0, 2
	v_lshl_add_u64 v[182:183], v[70:71], 0, 1
	v_cndmask_b32_e64 v71, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s29, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v69, v71, vcc
	v_cmp_ne_u32_e32 vcc, 0, v184
	s_nop 1
	v_cndmask_b32_e32 v69, v182, v180, vcc
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v70, v69, vcc
	v_xor_b32_e32 v32, v32, v68
	v_sub_co_u32_e32 v68, vcc, v32, v68
.LBB0_122:
	s_andn2_saveexec_b64 s[14:15], s[24:25]
	s_cbranch_execz .LBB0_124
	v_rcp_iflag_f32_e32 v32, v114
	s_sub_i32 s22, 0, s66
	v_mul_f32_e32 v32, 0x4f7ffffe, v32
	v_cvt_u32_f32_e32 v32, v32
	v_mul_lo_u32 v68, s22, v32
	v_mul_hi_u32 v68, v32, v68
	v_add_u32_e32 v32, v32, v68
	v_mul_hi_u32 v32, v70, v32
	v_mul_lo_u32 v68, v32, s66
	v_sub_u32_e32 v68, v70, v68
	v_add_u32_e32 v69, 1, v32
	v_subrev_u32_e32 v70, s66, v68
	v_cmp_le_u32_e32 vcc, s66, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v68, v70, vcc
	v_cndmask_b32_e32 v32, v32, v69, vcc
	v_add_u32_e32 v69, 1, v32
	v_cmp_le_u32_e32 vcc, s66, v68
	s_nop 1
	v_cndmask_b32_e32 v68, v32, v69, vcc
.LBB0_124:
	s_or_b64 exec, exec, s[14:15]
	v_lshl_add_u64 v[70:71], v[66:67], 0, 2
	v_or_b32_e32 v32, s67, v71
	v_cmp_ne_u32_e32 vcc, 0, v32
	s_and_saveexec_b64 s[14:15], vcc
	s_xor_b64 s[28:29], exec, s[14:15]
	s_cbranch_execz .LBB0_126
	s_add_u32 s14, s66, s76
	s_mov_b32 s77, s76
	s_addc_u32 s15, s67, s76
	s_xor_b64 s[34:35], s[14:15], s[76:77]
	v_cvt_f32_u32_e32 v32, s34
	v_cvt_f32_u32_e32 v66, s35
	s_sub_u32 s14, 0, s34
	s_subb_u32 s15, 0, s35
	v_ashrrev_i32_e32 v114, 31, v71
	v_fmac_f32_e32 v32, 0x4f800000, v66
	v_rcp_f32_e32 v32, v32
	v_mov_b32_e32 v115, v114
	v_mul_f32_e32 v32, 0x5f7ffffc, v32
	v_mul_f32_e32 v66, 0x2f800000, v32
	v_trunc_f32_e32 v66, v66
	v_fmac_f32_e32 v32, 0xcf800000, v66
	v_cvt_u32_f32_e32 v66, v66
	v_cvt_u32_f32_e32 v32, v32
	v_readfirstlane_b32 s22, v66
	v_readfirstlane_b32 s23, v32
	s_mul_i32 s24, s14, s22
	s_mul_hi_u32 s50, s14, s23
	s_mul_i32 s25, s15, s23
	s_add_i32 s24, s50, s24
	s_add_i32 s24, s24, s25
	s_mul_i32 s51, s14, s23
	s_mul_i32 s50, s23, s24
	s_mul_hi_u32 s77, s23, s51
	s_mul_hi_u32 s25, s23, s24
	s_add_u32 s50, s77, s50
	s_addc_u32 s25, 0, s25
	s_mul_hi_u32 vcc_lo, s22, s51
	s_mul_i32 s51, s22, s51
	s_add_u32 s50, s50, s51
	s_mul_hi_u32 s77, s22, s24
	s_addc_u32 s25, s25, vcc_lo
	s_addc_u32 s50, s77, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s24, s25, s24
	s_addc_u32 s25, 0, s50
	s_add_u32 s23, s23, s24
	s_addc_u32 s22, s22, s25
	s_mul_i32 s24, s14, s22
	s_mul_hi_u32 s25, s14, s23
	s_add_i32 s24, s25, s24
	s_mul_i32 s15, s15, s23
	s_add_i32 s24, s24, s15
	s_mul_i32 s14, s14, s23
	s_mul_hi_u32 s25, s22, s14
	s_mul_i32 s50, s22, s14
	s_mul_i32 s77, s23, s24
	s_mul_hi_u32 s14, s23, s14
	s_mul_hi_u32 s51, s23, s24
	s_add_u32 s14, s14, s77
	s_addc_u32 s51, 0, s51
	s_add_u32 s14, s14, s50
	s_mul_hi_u32 s15, s22, s24
	s_addc_u32 s14, s51, s25
	s_addc_u32 s15, s15, 0
	s_mul_i32 s24, s22, s24
	s_add_u32 s14, s14, s24
	s_addc_u32 s15, 0, s15
	s_add_u32 s23, s23, s14
	v_lshl_add_u64 v[66:67], v[70:71], 0, v[114:115]
	s_addc_u32 s22, s22, s15
	v_xor_b32_e32 v115, v66, v114
	v_xor_b32_e32 v69, v67, v114
	v_mad_u64_u32 v[66:67], s[14:15], v115, s22, 0
	v_mul_hi_u32 v32, v115, s23
	v_lshl_add_u64 v[66:67], v[32:33], 0, v[66:67]
	v_mad_u64_u32 v[180:181], s[14:15], v69, s23, 0
	v_add_co_u32_e32 v32, vcc, v66, v180
	v_mad_u64_u32 v[70:71], s[14:15], v69, s22, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v181, vcc
	v_mov_b32_e32 v67, s65
	s_nop 0
	v_addc_co_u32_e32 v71, vcc, 0, v71, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[70:71]
	v_mul_lo_u32 v32, s35, v66
	v_mul_lo_u32 v70, s34, v67
	v_mad_u64_u32 v[66:67], s[14:15], s34, v66, 0
	v_add3_u32 v32, v67, v70, v32
	v_sub_u32_e32 v67, v69, v32
	v_mov_b32_e32 v70, s35
	v_sub_co_u32_e32 v66, vcc, v115, v66
	s_nop 1
	v_subb_co_u32_e64 v67, s[14:15], v67, v70, vcc
	v_subrev_co_u32_e64 v71, s[14:15], s34, v66
	v_subb_co_u32_e32 v32, vcc, v69, v32, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v115, s[24:25], 0, v67, s[14:15]
	v_cmp_le_u32_e64 s[24:25], s35, v115
	v_subb_co_u32_e64 v67, s[14:15], v67, v70, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v180, 0, -1, s[24:25]
	v_cmp_le_u32_e64 s[24:25], s34, v71
	v_subrev_co_u32_e64 v70, s[14:15], s34, v71
	s_nop 0
	v_cndmask_b32_e64 v181, 0, -1, s[24:25]
	v_cmp_eq_u32_e64 s[24:25], s35, v115
	v_subbrev_co_u32_e64 v67, s[14:15], 0, v67, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v180, v180, v181, s[24:25]
	v_cmp_le_u32_e32 vcc, s35, v32
	v_cmp_ne_u32_e64 s[14:15], 0, v180
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s34, v66
	v_cndmask_b32_e64 v67, v115, v67, s[14:15]
	s_nop 0
	v_cndmask_b32_e64 v115, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s35, v32
	s_nop 1
	v_cndmask_b32_e32 v69, v69, v115, vcc
	v_cmp_ne_u32_e32 vcc, 0, v69
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v67, vcc
	v_cndmask_b32_e64 v67, v71, v70, s[14:15]
	v_cndmask_b32_e32 v66, v66, v67, vcc
	v_xor_b32_e32 v66, v66, v114
	v_xor_b32_e32 v32, v32, v114
	v_sub_co_u32_e32 v66, vcc, v66, v114
	s_nop 1
	v_subb_co_u32_e32 v67, vcc, v32, v114, vcc
.LBB0_126:
	s_andn2_saveexec_b64 s[14:15], s[28:29]
	s_cbranch_execz .LBB0_111
	v_rcp_iflag_f32_e32 v32, v114
	s_sub_i32 s22, 0, s66
	v_mul_f32_e32 v32, 0x4f7ffffe, v32
	v_cvt_u32_f32_e32 v32, v32
	v_mul_lo_u32 v66, s22, v32
	v_mul_hi_u32 v66, v32, v66
	v_add_u32_e32 v32, v32, v66
	v_mul_hi_u32 v32, v70, v32
	v_mul_lo_u32 v32, v32, s66
	v_sub_u32_e32 v32, v70, v32
	v_subrev_u32_e32 v66, s66, v32
	v_cmp_le_u32_e32 vcc, s66, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v66, vcc
	v_subrev_u32_e32 v66, s66, v32
	v_cmp_le_u32_e32 vcc, s66, v32
	s_nop 1
	v_cndmask_b32_e32 v32, v32, v66, vcc
	v_mov_b64_e32 v[66:67], v[32:33]
	s_branch .LBB0_111
.LBB0_128:
	v_readlane_b32 s34, v202, 18
	v_readlane_b32 s67, v202, 24
	v_readlane_b32 s72, v202, 6
	v_readlane_b32 s73, v202, 8
	v_readlane_b32 s76, v202, 26
	v_readlane_b32 s77, v202, 31
	v_readlane_b32 s35, v202, 19
.LBB0_129:
	v_readlane_b32 s2, v202, 16
	v_readlane_b32 s22, v202, 10
	v_readlane_b32 s3, v202, 17
	v_readlane_b32 s23, v202, 11
	s_or_b64 s[0:1], s[22:23], s[2:3]
	s_cmp_lg_u32 s1, 0
	s_cbranch_scc0 .LBB0_198
	s_add_u32 s0, s2, s62
	s_mov_b32 s63, s62
	s_addc_u32 s1, s3, s62
	s_xor_b64 s[0:1], s[0:1], s[62:63]
	v_cvt_f32_u32_e32 v32, s0
	v_cvt_f32_u32_e32 v33, s1
	s_sub_u32 s4, 0, s0
	s_subb_u32 s5, 0, s1
	v_fmamk_f32 v32, v33, 0x4f800000, v32
	v_rcp_f32_e32 v32, v32
	s_nop 0
	v_mul_f32_e32 v32, 0x5f7ffffc, v32
	v_mul_f32_e32 v33, 0x2f800000, v32
	v_trunc_f32_e32 v33, v33
	v_fmamk_f32 v32, v33, 0xcf800000, v32
	v_cvt_u32_f32_e32 v33, v33
	v_cvt_u32_f32_e32 v32, v32
	v_readfirstlane_b32 s6, v33
	v_readfirstlane_b32 s7, v32
	s_mul_i32 s8, s4, s6
	s_mul_hi_u32 s10, s4, s7
	s_mul_i32 s9, s5, s7
	s_add_i32 s8, s10, s8
	s_add_i32 s8, s8, s9
	s_mul_i32 s11, s4, s7
	s_mul_i32 s10, s7, s8
	s_mul_hi_u32 s14, s7, s11
	s_mul_hi_u32 s9, s7, s8
	s_add_u32 s10, s14, s10
	s_addc_u32 s9, 0, s9
	s_mul_hi_u32 s15, s6, s11
	s_mul_i32 s11, s6, s11
	s_add_u32 s10, s10, s11
	s_mul_hi_u32 s14, s6, s8
	s_addc_u32 s9, s9, s15
	s_addc_u32 s10, s14, 0
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
	s_mul_i32 s14, s7, s8
	s_mul_hi_u32 s4, s7, s4
	s_mul_hi_u32 s11, s7, s8
	s_add_u32 s4, s4, s14
	s_addc_u32 s11, 0, s11
	s_add_u32 s4, s4, s10
	s_mul_hi_u32 s5, s6, s8
	s_addc_u32 s4, s11, s9
	s_addc_u32 s5, s5, 0
	s_mul_i32 s8, s6, s8
	s_add_u32 s4, s4, s8
	s_addc_u32 s5, 0, s5
	s_add_u32 s8, s7, s4
	s_addc_u32 s9, s6, s5
	s_ashr_i32 s4, s23, 31
	s_add_u32 s6, s22, s4
	s_mov_b32 s5, s4
	s_addc_u32 s7, s23, s4
	s_xor_b64 s[6:7], s[6:7], s[4:5]
	s_mul_i32 s11, s6, s9
	s_mul_hi_u32 s14, s6, s8
	s_mul_hi_u32 s10, s6, s9
	s_add_u32 s11, s14, s11
	s_addc_u32 s10, 0, s10
	s_mul_hi_u32 s15, s7, s8
	s_mul_i32 s8, s7, s8
	s_add_u32 s8, s11, s8
	s_mul_hi_u32 s14, s7, s9
	s_addc_u32 s8, s10, s15
	s_addc_u32 s10, s14, 0
	s_mul_i32 s9, s7, s9
	s_add_u32 s8, s8, s9
	s_addc_u32 s9, 0, s10
	s_mul_i32 s9, s0, s9
	s_mul_hi_u32 s10, s0, s8
	s_add_i32 s9, s10, s9
	s_mul_i32 s10, s1, s8
	s_add_i32 s14, s9, s10
	s_sub_i32 s10, s7, s14
	s_mul_i32 s8, s0, s8
	s_sub_u32 s6, s6, s8
	s_cselect_b64 s[8:9], -1, 0
	s_subb_u32 s15, s10, s1
	s_sub_u32 s18, s6, s0
	s_cselect_b64 s[10:11], -1, 0
	s_subb_u32 s19, s15, 0
	s_cmp_ge_u32 s19, s1
	s_cselect_b32 s20, -1, 0
	s_cmp_ge_u32 s18, s0
	s_cselect_b32 s21, -1, 0
	s_cmp_eq_u32 s19, s1
	s_cselect_b32 s20, s21, s20
	s_cmp_lg_u64 s[10:11], 0
	s_subb_u32 s10, s15, s1
	s_sub_u32 s11, s18, s0
	s_subb_u32 s10, s10, 0
	s_cmp_lg_u32 s20, 0
	s_cselect_b32 s11, s11, s18
	s_cselect_b32 s10, s10, s19
	s_cmp_lg_u64 s[8:9], 0
	s_subb_u32 s7, s7, s14
	s_cmp_ge_u32 s7, s1
	s_cselect_b32 s8, -1, 0
	s_cmp_ge_u32 s6, s0
	s_cselect_b32 s0, -1, 0
	s_cmp_eq_u32 s7, s1
	s_cselect_b32 s0, s0, s8
	s_cmp_lg_u32 s0, 0
	s_cselect_b32 s1, s10, s7
	s_cselect_b32 s0, s11, s6
	s_xor_b64 s[0:1], s[0:1], s[4:5]
	s_sub_u32 s0, s0, s4
	s_subb_u32 s1, s1, s4
	s_cbranch_execnz .LBB0_132
.LBB0_131:
	v_rcp_iflag_f32_e32 v32, v74
	v_readlane_b32 s4, v202, 16
	s_sub_i32 s0, 0, s4
	s_mov_b32 s1, 0
	v_mul_f32_e32 v32, 0x4f7ffffe, v32
	v_cvt_u32_f32_e32 v32, v32
	v_readlane_b32 s5, v202, 17
	v_readfirstlane_b32 s2, v32
	s_mul_i32 s0, s0, s2
	s_mul_hi_u32 s0, s2, s0
	s_add_i32 s2, s2, s0
	s_mul_hi_u32 s0, s22, s2
	s_mul_i32 s0, s0, s4
	s_sub_i32 s0, s22, s0
	s_sub_i32 s2, s0, s4
	s_cmp_ge_u32 s0, s4
	s_cselect_b32 s0, s2, s0
	s_sub_i32 s2, s0, s4
	s_cmp_ge_u32 s0, s4
	s_cselect_b32 s0, s2, s0
.LBB0_132:
	v_bfe_u32 v32, v31, 16, 1
	s_movk_i32 s2, 0x7fff
	v_bfe_u32 v33, v30, 16, 1
	v_add3_u32 v32, v31, v32, s2
	v_bfe_u32 v34, v29, 16, 1
	v_add3_u32 v33, v30, v33, s2
	v_lshrrev_b32_e32 v32, 16, v32
	v_mov_b32_e32 v42, 0x7fc0
	v_cmp_o_f32_e32 vcc, v31, v31
	v_add3_u32 v34, v29, v34, s2
	v_lshrrev_b32_e32 v33, 16, v33
	v_cndmask_b32_e32 v36, v42, v32, vcc
	v_cmp_o_f32_e32 vcc, v30, v30
	v_bfe_u32 v35, v28, 16, 1
	v_lshrrev_b32_e32 v34, 16, v34
	v_cndmask_b32_e32 v37, v42, v33, vcc
	v_cmp_o_f32_e32 vcc, v29, v29
	v_add3_u32 v35, v28, v35, s2
	v_lshrrev_b32_e32 v35, 16, v35
	v_cndmask_b32_e32 v38, v42, v34, vcc
	v_cmp_o_f32_e32 vcc, v28, v28
	v_bfe_u32 v28, v27, 16, 1
	v_bfe_u32 v29, v26, 16, 1
	v_add3_u32 v28, v27, v28, s2
	v_cndmask_b32_e32 v39, v42, v35, vcc
	v_bfe_u32 v30, v25, 16, 1
	v_add3_u32 v29, v26, v29, s2
	v_lshrrev_b32_e32 v28, 16, v28
	v_cmp_o_f32_e32 vcc, v27, v27
	v_add3_u32 v30, v25, v30, s2
	v_lshrrev_b32_e32 v29, 16, v29
	v_cndmask_b32_e32 v32, v42, v28, vcc
	v_cmp_o_f32_e32 vcc, v26, v26
	v_bfe_u32 v31, v24, 16, 1
	v_lshrrev_b32_e32 v30, 16, v30
	v_cndmask_b32_e32 v33, v42, v29, vcc
	v_cmp_o_f32_e32 vcc, v25, v25
	v_add3_u32 v31, v24, v31, s2
	v_lshrrev_b32_e32 v31, 16, v31
	v_cndmask_b32_e32 v34, v42, v30, vcc
	v_cmp_o_f32_e32 vcc, v24, v24
	v_bfe_u32 v24, v23, 16, 1
	v_bfe_u32 v25, v22, 16, 1
	v_add3_u32 v24, v23, v24, s2
	v_cndmask_b32_e32 v35, v42, v31, vcc
	v_bfe_u32 v26, v21, 16, 1
	v_add3_u32 v25, v22, v25, s2
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v23, v23
	v_add3_u32 v26, v21, v26, s2
	v_lshrrev_b32_e32 v25, 16, v25
	v_cndmask_b32_e32 v28, v42, v24, vcc
	v_cmp_o_f32_e32 vcc, v22, v22
	v_bfe_u32 v27, v20, 16, 1
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v29, v42, v25, vcc
	v_cmp_o_f32_e32 vcc, v21, v21
	v_add3_u32 v27, v20, v27, s2
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v30, v42, v26, vcc
	v_cmp_o_f32_e32 vcc, v20, v20
	v_bfe_u32 v20, v19, 16, 1
	v_bfe_u32 v21, v18, 16, 1
	v_add3_u32 v20, v19, v20, s2
	v_cndmask_b32_e32 v31, v42, v27, vcc
	v_bfe_u32 v22, v17, 16, 1
	v_add3_u32 v21, v18, v21, s2
	v_lshrrev_b32_e32 v20, 16, v20
	v_cmp_o_f32_e32 vcc, v19, v19
	v_add3_u32 v22, v17, v22, s2
	v_lshrrev_b32_e32 v21, 16, v21
	v_cndmask_b32_e32 v24, v42, v20, vcc
	v_cmp_o_f32_e32 vcc, v18, v18
	v_bfe_u32 v23, v16, 16, 1
	v_lshrrev_b32_e32 v22, 16, v22
	v_cndmask_b32_e32 v25, v42, v21, vcc
	v_cmp_o_f32_e32 vcc, v17, v17
	v_add3_u32 v23, v16, v23, s2
	v_lshrrev_b32_e32 v23, 16, v23
	v_cndmask_b32_e32 v26, v42, v22, vcc
	v_cmp_o_f32_e32 vcc, v16, v16
	v_bfe_u32 v16, v15, 16, 1
	v_bfe_u32 v17, v14, 16, 1
	v_add3_u32 v16, v15, v16, s2
	v_cndmask_b32_e32 v27, v42, v23, vcc
	v_bfe_u32 v18, v13, 16, 1
	v_add3_u32 v17, v14, v17, s2
	v_lshrrev_b32_e32 v16, 16, v16
	v_cmp_o_f32_e32 vcc, v15, v15
	v_add3_u32 v18, v13, v18, s2
	v_lshrrev_b32_e32 v17, 16, v17
	v_cndmask_b32_e32 v20, v42, v16, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_bfe_u32 v19, v12, 16, 1
	v_lshrrev_b32_e32 v18, 16, v18
	v_cndmask_b32_e32 v21, v42, v17, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_add3_u32 v19, v12, v19, s2
	v_lshrrev_b32_e32 v19, 16, v19
	v_cndmask_b32_e32 v22, v42, v18, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v12, v11, 16, 1
	v_bfe_u32 v13, v10, 16, 1
	v_add3_u32 v12, v11, v12, s2
	v_cndmask_b32_e32 v23, v42, v19, vcc
	v_bfe_u32 v14, v9, 16, 1
	v_add3_u32 v13, v10, v13, s2
	v_lshrrev_b32_e32 v12, 16, v12
	v_cmp_o_f32_e32 vcc, v11, v11
	v_add3_u32 v14, v9, v14, s2
	v_lshrrev_b32_e32 v13, 16, v13
	v_cndmask_b32_e32 v16, v42, v12, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_bfe_u32 v15, v8, 16, 1
	v_lshrrev_b32_e32 v14, 16, v14
	v_cndmask_b32_e32 v17, v42, v13, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_add3_u32 v15, v8, v15, s2
	v_lshrrev_b32_e32 v15, 16, v15
	v_cndmask_b32_e32 v18, v42, v14, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v8, v7, 16, 1
	v_bfe_u32 v9, v6, 16, 1
	v_add3_u32 v8, v7, v8, s2
	v_cndmask_b32_e32 v19, v42, v15, vcc
	v_bfe_u32 v10, v5, 16, 1
	v_bfe_u32 v11, v4, 16, 1
	v_add3_u32 v9, v6, v9, s2
	v_lshrrev_b32_e32 v8, 16, v8
	v_cmp_o_f32_e32 vcc, v7, v7
	v_add3_u32 v11, v4, v11, s2
	v_add3_u32 v10, v5, v10, s2
	v_lshrrev_b32_e32 v12, 16, v9
	v_cndmask_b32_e32 v9, v42, v8, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_lshrrev_b32_e32 v10, 16, v10
	v_lshrrev_b32_e32 v14, 16, v11
	v_cndmask_b32_e32 v11, v42, v12, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_bfe_u32 v5, v2, 16, 1
	v_bfe_u32 v6, v1, 16, 1
	v_cndmask_b32_e32 v13, v42, v10, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v4, v3, 16, 1
	v_bfe_u32 v7, v0, 16, 1
	v_readlane_b32 s26, v202, 16
	v_add3_u32 v7, v0, v7, s2
	v_add3_u32 v6, v1, v6, s2
	v_add3_u32 v5, v2, v5, s2
	v_add3_u32 v4, v3, v4, s2
	v_readlane_b32 s2, v202, 0
	v_readlane_b32 s27, v202, 17
	v_readlane_b32 s3, v202, 1
	s_sub_u32 s2, s73, s2
	s_mul_i32 s4, s26, s27
	s_mul_hi_u32 s5, s26, s26
	s_subb_u32 s3, s72, s3
	s_add_i32 s5, s5, s4
	s_mul_i32 s7, s26, s26
	s_add_i32 s6, s5, s4
	s_mul_i32 s3, s7, s3
	s_mul_hi_u32 s5, s7, s2
	s_mul_i32 s4, s6, s2
	s_add_i32 s3, s5, s3
	s_add_i32 s3, s3, s4
	s_mul_i32 s4, s6, s67
	s_mul_hi_u32 s5, s7, s67
	s_mul_i32 s2, s7, s2
	s_add_i32 s5, s5, s4
	s_mul_i32 s4, s7, s67
	s_add_u32 s4, s2, s4
	s_addc_u32 s5, s3, s5
	s_mul_i32 s2, s7, s27
	s_mul_hi_u32 s3, s7, s26
	s_add_i32 s2, s3, s2
	s_mul_i32 s6, s6, s26
	s_add_i32 s10, s2, s6
	s_ashr_i32 s2, s5, 31
	s_mov_b32 s3, s2
	s_mul_i32 s7, s7, s26
	s_xor_b64 s[8:9], s[2:3], s[4:5]
	s_ashr_i32 s4, s10, 31
	s_add_u32 s6, s7, s4
	s_mov_b32 s5, s4
	s_addc_u32 s7, s10, s4
	s_xor_b64 s[6:7], s[6:7], s[4:5]
	v_cndmask_b32_e32 v15, v42, v14, vcc
	v_cvt_f32_u32_e32 v10, s6
	v_cvt_f32_u32_e32 v14, s7
	v_lshrrev_b32_e32 v4, 16, v4
	v_cmp_o_f32_e32 vcc, v3, v3
	v_lshrrev_b32_e32 v12, 16, v7
	v_lshrrev_b32_e32 v5, 16, v5
	v_cndmask_b32_e32 v7, v42, v4, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	v_fmamk_f32 v2, v14, 0x4f800000, v10
	v_rcp_f32_e32 v2, v2
	v_lshrrev_b32_e32 v6, 16, v6
	v_cndmask_b32_e32 v8, v42, v5, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	s_sub_u32 s10, 0, s6
	s_subb_u32 s11, 0, s7
	v_cndmask_b32_e32 v10, v42, v6, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	v_mul_f32_e32 v0, 0x5f7ffffc, v2
	v_mul_f32_e32 v1, 0x2f800000, v0
	v_trunc_f32_e32 v1, v1
	v_fmamk_f32 v0, v1, 0xcf800000, v0
	v_cvt_u32_f32_e32 v0, v0
	v_cvt_u32_f32_e32 v1, v1
	v_cndmask_b32_e32 v12, v42, v12, vcc
	v_readfirstlane_b32 s14, v0
	v_readfirstlane_b32 s19, v1
	s_mul_hi_u32 s18, s10, s14
	s_mul_i32 s20, s10, s19
	s_mul_i32 s15, s11, s14
	s_add_i32 s18, s18, s20
	s_add_i32 s18, s18, s15
	s_mul_i32 s21, s10, s14
	s_mul_i32 s20, s14, s18
	s_mul_hi_u32 s22, s14, s21
	s_mul_hi_u32 s15, s14, s18
	s_add_u32 s20, s22, s20
	s_addc_u32 s15, 0, s15
	s_mul_hi_u32 s23, s19, s21
	s_mul_i32 s21, s19, s21
	s_add_u32 s20, s20, s21
	s_mul_hi_u32 s22, s19, s18
	s_addc_u32 s15, s15, s23
	s_addc_u32 s20, s22, 0
	s_mul_i32 s18, s19, s18
	s_add_u32 s15, s15, s18
	s_addc_u32 s18, 0, s20
	s_add_u32 s14, s14, s15
	s_addc_u32 s15, s19, s18
	s_mul_i32 s18, s10, s15
	s_mul_hi_u32 s19, s10, s14
	s_add_i32 s18, s19, s18
	s_mul_i32 s11, s11, s14
	s_add_i32 s18, s18, s11
	s_mul_i32 s10, s10, s14
	s_mul_hi_u32 s19, s15, s10
	s_mul_i32 s20, s15, s10
	s_mul_i32 s22, s14, s18
	s_mul_hi_u32 s10, s14, s10
	s_mul_hi_u32 s21, s14, s18
	s_add_u32 s10, s10, s22
	s_addc_u32 s21, 0, s21
	s_add_u32 s10, s10, s20
	s_mul_hi_u32 s11, s15, s18
	s_addc_u32 s10, s21, s19
	s_addc_u32 s11, s11, 0
	s_mul_i32 s18, s15, s18
	s_add_u32 s10, s10, s18
	s_addc_u32 s11, 0, s11
	s_add_u32 s14, s14, s10
	s_addc_u32 s15, s15, s11
	s_ashr_i32 s10, s9, 31
	s_add_u32 s8, s8, s10
	s_mov_b32 s11, s10
	s_addc_u32 s9, s9, s10
	s_xor_b64 s[8:9], s[8:9], s[10:11]
	s_mul_i32 s19, s8, s15
	s_mul_hi_u32 s20, s8, s14
	s_mul_hi_u32 s18, s8, s15
	s_add_u32 s19, s20, s19
	s_addc_u32 s18, 0, s18
	s_mul_hi_u32 s21, s9, s14
	s_mul_i32 s14, s9, s14
	s_add_u32 s14, s19, s14
	s_mul_hi_u32 s20, s9, s15
	s_addc_u32 s14, s18, s21
	s_addc_u32 s18, s20, 0
	s_mul_i32 s15, s9, s15
	s_add_u32 s19, s14, s15
	s_addc_u32 s18, 0, s18
	s_mul_i32 s14, s6, s18
	s_mul_hi_u32 s15, s6, s19
	s_add_i32 s14, s15, s14
	s_mul_i32 s15, s7, s19
	s_add_i32 s20, s14, s15
	s_sub_i32 s21, s9, s20
	s_mul_i32 s14, s6, s19
	s_sub_u32 s8, s8, s14
	s_cselect_b64 s[14:15], -1, 0
	s_subb_u32 s21, s21, s7
	s_sub_u32 s22, s8, s6
	s_subb_u32 s21, s21, 0
	s_cmp_ge_u32 s21, s7
	s_cselect_b32 s23, -1, 0
	s_cmp_ge_u32 s22, s6
	s_cselect_b32 s22, -1, 0
	s_cmp_eq_u32 s21, s7
	s_cselect_b32 s21, s22, s23
	s_add_u32 s22, s19, 1
	s_addc_u32 s23, s18, 0
	s_add_u32 s24, s19, 2
	s_addc_u32 s25, s18, 0
	s_cmp_lg_u32 s21, 0
	s_cselect_b32 s21, s24, s22
	s_cselect_b32 s22, s25, s23
	s_cmp_lg_u64 s[14:15], 0
	s_subb_u32 s9, s9, s20
	s_cmp_ge_u32 s9, s7
	s_cselect_b32 s14, -1, 0
	s_cmp_ge_u32 s8, s6
	s_cselect_b32 s6, -1, 0
	s_cmp_eq_u32 s9, s7
	s_cselect_b32 s6, s6, s14
	s_cmp_lg_u32 s6, 0
	s_cselect_b32 s7, s22, s18
	s_cselect_b32 s6, s21, s19
	s_xor_b64 s[4:5], s[10:11], s[4:5]
	s_xor_b64 s[6:7], s[6:7], s[4:5]
	s_sub_u32 s4, s6, s4
	s_subb_u32 s5, s7, s5
	s_xor_b64 s[2:3], s[4:5], s[2:3]
	v_readlane_b32 s4, v202, 12
	v_readlane_b32 s5, v202, 13
	s_and_b64 s[4:5], s[4:5], exec
	s_cselect_b32 s5, s3, s77
	s_cselect_b32 s4, s2, s76
	s_lshl_b64 s[2:3], s[4:5], 7
	v_sub_co_u32_e32 v2, vcc, s4, v72
	s_ashr_i32 s4, s1, 31
	v_mov_b32_e32 v1, s5
	s_mov_b32 s5, s4
	s_and_b64 s[4:5], s[4:5], s[26:27]
	v_readlane_b32 s6, v202, 14
	v_readlane_b32 s7, v202, 15
	s_add_u32 s0, s0, s6
	v_lshlrev_b32_e32 v0, 2, v97
	s_addc_u32 s1, s1, s7
	v_or_b32_e32 v4, s2, v0
	s_add_u32 s2, s0, s4
	s_addc_u32 s4, s1, s5
	v_readlane_b32 s0, v202, 4
	v_readlane_b32 s1, v202, 5
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s2, s2, 0
	s_cselect_b32 s1, s4, 0
	s_sub_u32 s0, s34, s2
	s_subb_u32 s1, s35, s1
	s_lshl_b32 s2, s2, 5
	v_subb_co_u32_e32 v1, vcc, v1, v73, vcc
	v_lshl_or_b32 v14, v72, 7, v0
	v_and_b32_e32 v0, 15, v40
	v_or_b32_e32 v3, s2, v41
	v_add_u32_e32 v6, v3, v0
	s_waitcnt lgkmcnt(0)
	v_mul_lo_u32 v0, s16, v1
	v_mul_hi_u32 v1, s16, v2
	v_add_u32_e32 v0, v1, v0
	v_mul_lo_u32 v1, s17, v2
	v_mad_u64_u32 v[40:41], s[4:5], v14, s16, v[6:7]
	v_add_u32_e32 v1, v0, v1
	v_mul_lo_u32 v0, s16, v2
	v_readlane_b32 s4, v202, 2
	v_mov_b32_e32 v5, s3
	v_lshlrev_b64 v[0:1], 8, v[0:1]
	v_readlane_b32 s5, v202, 3
	s_lshl_b64 s[0:1], s[0:1], 6
	v_bfrev_b32_e32 v41, -2
	v_lshl_add_u64 v[0:1], s[4:5], 0, v[0:1]
	v_cmp_gt_i64_e32 vcc, s[12:13], v[4:5]
	v_lshl_add_u64 v[0:1], v[0:1], 0, s[0:1]
	v_and_b32_e32 v1, 0xffff, v1
	v_cndmask_b32_e32 v5, v41, v40, vcc
	v_mov_b32_e32 v2, -3
	v_mov_b32_e32 v3, 0x27000
	v_lshlrev_b32_e32 v5, 1, v5
	s_mov_b64 s[8:9], exec
.LBB0_133:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v39, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_133
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v40, 1, v4
	v_mov_b32_e32 v41, s3
	v_or_b32_e32 v5, 1, v14
	v_mad_u64_u32 v[42:43], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[40:41]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v42, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_135:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v38, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_135
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v38, 2, v4
	v_mov_b32_e32 v39, s3
	v_or_b32_e32 v5, 2, v14
	v_mad_u64_u32 v[40:41], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[38:39]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v40, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_137:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v37, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_137
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v38, 3, v4
	v_mov_b32_e32 v39, s3
	v_or_b32_e32 v5, 3, v14
	v_mad_u64_u32 v[40:41], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[38:39]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v40, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_139:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v36, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_139
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v36, 16, v4
	v_mov_b32_e32 v37, s3
	v_or_b32_e32 v5, 16, v14
	v_mad_u64_u32 v[38:39], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[36:37]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v38, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_141:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v35, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_141
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v36, 17, v4
	v_mov_b32_e32 v37, s3
	v_or_b32_e32 v5, 17, v14
	v_mad_u64_u32 v[38:39], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[36:37]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v38, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_143:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v34, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_143
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v34, 18, v4
	v_mov_b32_e32 v35, s3
	v_or_b32_e32 v5, 18, v14
	v_mad_u64_u32 v[36:37], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[34:35]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v36, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_145:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v33, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_145
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v34, 19, v4
	v_mov_b32_e32 v35, s3
	v_or_b32_e32 v5, 19, v14
	v_mad_u64_u32 v[36:37], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[34:35]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v36, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_147:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v32, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_147
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v32, 32, v4
	v_mov_b32_e32 v33, s3
	v_or_b32_e32 v5, 32, v14
	v_mad_u64_u32 v[34:35], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[32:33]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v34, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_149:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v31, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_149
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v32, 33, v4
	v_mov_b32_e32 v33, s3
	v_or_b32_e32 v5, 33, v14
	v_mad_u64_u32 v[34:35], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[32:33]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v34, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_151:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v30, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_151
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v30, 34, v4
	v_mov_b32_e32 v31, s3
	v_or_b32_e32 v5, 34, v14
	v_mad_u64_u32 v[32:33], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[30:31]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v32, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_153:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v29, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_153
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v30, 35, v4
	v_mov_b32_e32 v31, s3
	v_or_b32_e32 v5, 35, v14
	v_mad_u64_u32 v[32:33], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[30:31]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v32, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_155:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v28, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_155
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v28, 48, v4
	v_mov_b32_e32 v29, s3
	v_or_b32_e32 v5, 48, v14
	v_mad_u64_u32 v[30:31], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[28:29]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v30, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_157:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v27, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_157
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v28, 49, v4
	v_mov_b32_e32 v29, s3
	v_or_b32_e32 v5, 49, v14
	v_mad_u64_u32 v[30:31], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[28:29]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v30, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_159:
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
	s_cbranch_execnz .LBB0_159
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v26, 50, v4
	v_mov_b32_e32 v27, s3
	v_or_b32_e32 v5, 50, v14
	v_mad_u64_u32 v[28:29], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[26:27]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v28, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_161:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v25, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_161
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v26, 51, v4
	v_mov_b32_e32 v27, s3
	v_or_b32_e32 v5, 51, v14
	v_mad_u64_u32 v[28:29], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[26:27]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v28, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_163:
	v_readfirstlane_b32 s4, v0
	v_readfirstlane_b32 s5, v1
	v_readfirstlane_b32 s6, v2
	v_readfirstlane_b32 s7, v3
	v_cmp_eq_u64_e32 vcc, s[4:5], v[0:1]
	s_nop 0
	v_cmp_eq_u64_e64 s[0:1], s[6:7], v[2:3]
	s_and_b64 s[0:1], vcc, s[0:1]
	s_and_saveexec_b64 s[0:1], s[0:1]
	buffer_store_short v24, v5, s[4:7], 0 offen
	s_xor_b64 exec, exec, s[0:1]
	s_cbranch_execnz .LBB0_163
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v24, 64, v4
	v_mov_b32_e32 v25, s3
	v_or_b32_e32 v5, 64, v14
	v_mad_u64_u32 v[26:27], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[24:25]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v26, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_165:
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
	s_cbranch_execnz .LBB0_165
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v24, 0x41, v4
	v_mov_b32_e32 v25, s3
	v_or_b32_e32 v5, 0x41, v14
	v_mad_u64_u32 v[26:27], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[24:25]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v26, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_167:
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
	s_cbranch_execnz .LBB0_167
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v22, 0x42, v4
	v_mov_b32_e32 v23, s3
	v_or_b32_e32 v5, 0x42, v14
	v_mad_u64_u32 v[24:25], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[22:23]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v24, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_169:
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
	s_cbranch_execnz .LBB0_169
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v22, 0x43, v4
	v_mov_b32_e32 v23, s3
	v_or_b32_e32 v5, 0x43, v14
	v_mad_u64_u32 v[24:25], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[22:23]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v24, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_171:
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
	s_cbranch_execnz .LBB0_171
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v20, 0x50, v4
	v_mov_b32_e32 v21, s3
	v_or_b32_e32 v5, 0x50, v14
	v_mad_u64_u32 v[22:23], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[20:21]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v22, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_173:
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
	s_cbranch_execnz .LBB0_173
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v20, 0x51, v4
	v_mov_b32_e32 v21, s3
	v_or_b32_e32 v5, 0x51, v14
	v_mad_u64_u32 v[22:23], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[20:21]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v22, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_175:
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
	s_cbranch_execnz .LBB0_175
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v18, 0x52, v4
	v_mov_b32_e32 v19, s3
	v_or_b32_e32 v5, 0x52, v14
	v_mad_u64_u32 v[20:21], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[18:19]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v20, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_177:
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
	s_cbranch_execnz .LBB0_177
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v18, 0x53, v4
	v_mov_b32_e32 v19, s3
	v_or_b32_e32 v5, 0x53, v14
	v_mad_u64_u32 v[20:21], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[18:19]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v20, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_179:
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
	s_cbranch_execnz .LBB0_179
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v16, 0x60, v4
	v_mov_b32_e32 v17, s3
	v_or_b32_e32 v5, 0x60, v14
	v_mad_u64_u32 v[18:19], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[16:17]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v18, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_181:
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
	s_cbranch_execnz .LBB0_181
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v16, 0x61, v4
	v_mov_b32_e32 v17, s3
	v_or_b32_e32 v5, 0x61, v14
	v_mad_u64_u32 v[18:19], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[16:17]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v18, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_183:
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
	s_cbranch_execnz .LBB0_183
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v16, 0x62, v4
	v_mov_b32_e32 v17, s3
	v_or_b32_e32 v5, 0x62, v14
	v_mad_u64_u32 v[18:19], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[16:17]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v18, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_185:
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
	s_cbranch_execnz .LBB0_185
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v16, 0x63, v4
	v_mov_b32_e32 v17, s3
	v_or_b32_e32 v5, 0x63, v14
	v_mad_u64_u32 v[18:19], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[16:17]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v18, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_187:
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
	s_cbranch_execnz .LBB0_187
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v16, 0x70, v4
	v_mov_b32_e32 v17, s3
	v_or_b32_e32 v5, 0x70, v14
	v_mad_u64_u32 v[18:19], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[16:17]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v18, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_189:
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
	s_cbranch_execnz .LBB0_189
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v12, 0x71, v4
	v_mov_b32_e32 v13, s3
	v_or_b32_e32 v5, 0x71, v14
	v_mad_u64_u32 v[16:17], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[12:13]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v16, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_191:
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
	s_cbranch_execnz .LBB0_191
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v10, 0x72, v4
	v_mov_b32_e32 v11, s3
	v_or_b32_e32 v5, 0x72, v14
	v_mad_u64_u32 v[12:13], s[0:1], v5, s16, v[6:7]
	v_bfrev_b32_e32 v5, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[10:11]
	s_mov_b64 s[8:9], exec
	s_nop 0
	v_cndmask_b32_e32 v5, v5, v12, vcc
	v_lshlrev_b32_e32 v5, 1, v5
.LBB0_193:
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
	s_cbranch_execnz .LBB0_193
	s_mov_b64 exec, s[8:9]
	v_or_b32_e32 v4, 0x73, v4
	v_mov_b32_e32 v5, s3
	v_or_b32_e32 v8, 0x73, v14
	v_mad_u64_u32 v[8:9], s[0:1], v8, s16, v[6:7]
	v_bfrev_b32_e32 v6, -2
	v_cmp_gt_i64_e32 vcc, s[12:13], v[4:5]
	s_mov_b64 s[0:1], exec
	s_nop 0
	v_cndmask_b32_e32 v4, v6, v8, vcc
	v_lshlrev_b32_e32 v4, 1, v4
.LBB0_195:
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
	s_cbranch_execnz .LBB0_195
	s_endpgm
.LBB0_197:
	v_cvt_f32_u32_e32 v74, s68
	s_branch .LBB0_108
.LBB0_198:
	s_branch .LBB0_131
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
		.amdhsa_next_free_vgpr 203
		.amdhsa_next_free_sgpr 98
		.amdhsa_accum_offset 204
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

	.set wave_mxfp4_dynamic_gemm_128x32x256.num_vgpr, 203
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
      - 64
      - 4
      - 1
    .sgpr_count:     104
    .sgpr_spill_count: 60
    .symbol:         wave_mxfp4_dynamic_gemm_128x32x256.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     203
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
