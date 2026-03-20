; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_dynamic_gemm_256x192x256
	.p2align	8
	.type	wave_mxfp4_dynamic_gemm_256x192x256,@function
wave_mxfp4_dynamic_gemm_256x192x256:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	s_load_dwordx2 s[18:19], s[0:1], 0x38
	s_mov_b64 s[24:25], s[2:3]
	s_mov_b64 s[20:21], s[6:7]
	v_and_b32_e32 v97, 0x3ff, v0
	v_bfe_u32 v3, v0, 10, 10
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s2, s19, 31
	s_mov_b32 s3, s2
	s_xor_b64 s[6:7], s[18:19], s[2:3]
	s_lshr_b32 s14, s7, 31
	s_add_u32 s6, s6, s14
	s_addc_u32 s7, s7, 0
	v_lshrrev_b32_e32 v4, 6, v97
	v_lshlrev_b32_e32 v105, 5, v3
	s_ashr_i64 s[6:7], s[6:7], 1
	s_mov_b32 s12, s17
	s_mov_b32 s17, 0
	v_lshl_or_b32 v0, v4, 3, v105
	s_xor_b64 s[34:35], s[6:7], s[2:3]
	v_lshrrev_b32_e32 v2, 3, v97
	v_readfirstlane_b32 s13, v0
	v_or_b32_e32 v0, v2, v105
	s_lshl_b64 s[6:7], s[16:17], 8
	s_and_b32 s2, s34, 0x3fff
	v_or_b32_e32 v5, s6, v0
	v_bitop3_b32 v1, v2, 7, v97 bitop3:0x48
	s_bitset1_b32 s2, 14
	v_lshlrev_b32_e32 v108, 4, v1
	v_mul_lo_u32 v107, s34, v5
	s_lshl_b32 s28, s2, 16
	s_and_b32 s2, s25, 0xffff
	s_lshl_b32 s35, s13, 7
	s_or_b32 s25, s2, s28
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_add_u32_e32 v6, v108, v107
	s_mov_b32 m0, s35
	s_or_b32 s13, s35, 0x2000
	buffer_load_dwordx4 v6, s[24:27], 0 offen lds
	v_or_b32_e32 v6, 64, v5
	v_mul_lo_u32 v109, s34, v6
	v_add_u32_e32 v6, v108, v109
	s_mov_b32 m0, s13
	s_or_b32 s33, s35, 0x4000
	buffer_load_dwordx4 v6, s[24:27], 0 offen lds
	v_or_b32_e32 v6, 0x80, v5
	v_mul_lo_u32 v110, s34, v6
	v_or_b32_e32 v5, 0xc0, v5
	v_add_u32_e32 v6, v108, v110
	s_mov_b32 m0, s33
	v_mul_lo_u32 v111, s34, v5
	s_or_b32 s48, s35, 0x6000
	buffer_load_dwordx4 v6, s[24:27], 0 offen lds
	v_add_u32_e32 v5, v108, v111
	s_mov_b32 m0, s48
	s_add_u32 s22, s34, -1
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	s_addc_u32 s15, 0, 0
	s_mov_b32 s14, s17
	s_cmp_lg_u64 s[14:15], 0
	s_cbranch_scc0 .LBB0_15
	v_cvt_f32_u32_e32 v5, s34
	v_mov_b32_e32 v6, 0x4f800000
	s_sub_u32 s14, 0, s34
	s_subb_u32 s17, 0, 0
	v_fmac_f32_e32 v5, 0, v6
	v_rcp_f32_e32 v5, v5
	s_nop 0
	v_mul_f32_e32 v5, 0x5f7ffffc, v5
	v_mul_f32_e32 v6, 0x2f800000, v5
	v_trunc_f32_e32 v6, v6
	v_fmamk_f32 v5, v6, 0xcf800000, v5
	v_cvt_u32_f32_e32 v6, v6
	v_cvt_u32_f32_e32 v5, v5
	v_readfirstlane_b32 s23, v6
	v_readfirstlane_b32 s29, v5
	s_mul_i32 s30, s14, s23
	s_mul_hi_u32 s36, s14, s29
	s_mul_i32 s31, s17, s29
	s_add_i32 s30, s36, s30
	s_add_i32 s30, s30, s31
	s_mul_i32 s37, s14, s29
	s_mul_i32 s36, s29, s30
	s_mul_hi_u32 s38, s29, s37
	s_mul_hi_u32 s31, s29, s30
	s_add_u32 s36, s38, s36
	s_addc_u32 s31, 0, s31
	s_mul_hi_u32 s39, s23, s37
	s_mul_i32 s37, s23, s37
	s_add_u32 s36, s36, s37
	s_mul_hi_u32 s38, s23, s30
	s_addc_u32 s31, s31, s39
	s_addc_u32 s36, s38, 0
	s_mul_i32 s30, s23, s30
	s_add_u32 s30, s31, s30
	s_addc_u32 s31, 0, s36
	s_add_u32 s29, s29, s30
	s_addc_u32 s23, s23, s31
	s_mul_i32 s30, s14, s23
	s_mul_hi_u32 s31, s14, s29
	s_add_i32 s30, s31, s30
	s_mul_i32 s17, s17, s29
	s_add_i32 s30, s30, s17
	s_mul_i32 s14, s14, s29
	s_mul_hi_u32 s31, s23, s14
	s_mul_i32 s36, s23, s14
	s_mul_i32 s38, s29, s30
	s_mul_hi_u32 s14, s29, s14
	s_mul_hi_u32 s37, s29, s30
	s_add_u32 s14, s14, s38
	s_addc_u32 s37, 0, s37
	s_add_u32 s14, s14, s36
	s_mul_hi_u32 s17, s23, s30
	s_addc_u32 s14, s37, s31
	s_addc_u32 s17, s17, 0
	s_mul_i32 s30, s23, s30
	s_add_u32 s14, s14, s30
	s_addc_u32 s17, 0, s17
	s_add_u32 s14, s29, s14
	s_addc_u32 s17, s23, s17
	s_mul_i32 s29, s22, s17
	s_mul_hi_u32 s30, s22, s14
	s_mul_hi_u32 s23, s22, s17
	s_add_u32 s29, s30, s29
	s_addc_u32 s23, 0, s23
	s_mul_i32 s14, s15, s14
	s_add_u32 s14, s29, s14
	s_addc_u32 s14, s23, 0
	s_addc_u32 s23, 0, 0
	s_mul_i32 s17, s15, s17
	s_add_u32 s14, s14, s17
	s_addc_u32 s17, 0, s23
	s_mul_i32 s17, s34, s17
	s_mul_hi_u32 s30, s34, s14
	s_add_u32 s23, s14, 1
	s_add_u32 s29, s14, 2
	s_add_i32 s30, s30, s17
	s_mul_i32 s17, s34, s14
	s_sub_u32 s17, s22, s17
	s_subb_u32 s15, s15, s30
	s_sub_u32 s30, s17, s34
	s_subb_u32 s31, s15, 0
	s_cmp_ge_u32 s30, s34
	s_cselect_b32 s30, -1, 0
	s_cmp_eq_u32 s31, 0
	s_cselect_b32 s30, s30, -1
	s_cmp_lg_u32 s30, 0
	s_cselect_b32 s23, s29, s23
	s_cmp_ge_u32 s17, s34
	s_cselect_b32 s17, -1, 0
	s_cmp_eq_u32 s15, 0
	s_cselect_b32 s15, s17, -1
	s_cmp_lg_u32 s15, 0
	s_cselect_b32 s40, s23, s14
	s_cbranch_execnz .LBB0_3
.LBB0_2:
	v_cvt_f32_u32_e32 v5, s34
	s_sub_i32 s2, 0, s34
	v_rcp_iflag_f32_e32 v5, v5
	s_nop 0
	v_mul_f32_e32 v5, 0x4f7ffffe, v5
	v_cvt_u32_f32_e32 v5, v5
	s_nop 0
	v_readfirstlane_b32 s3, v5
	s_mul_i32 s2, s2, s3
	s_mul_hi_u32 s2, s3, s2
	s_add_i32 s3, s3, s2
	s_mul_hi_u32 s2, s22, s3
	s_mul_i32 s14, s2, s34
	s_sub_i32 s14, s22, s14
	s_add_i32 s3, s2, 1
	s_sub_i32 s15, s14, s34
	s_cmp_ge_u32 s14, s34
	s_cselect_b32 s2, s3, s2
	s_cselect_b32 s14, s15, s14
	s_add_i32 s3, s2, 1
	s_cmp_ge_u32 s14, s34
	s_cselect_b32 s40, s3, s2
.LBB0_3:
	v_lshrrev_b32_e32 v5, 7, v97
	v_sub_u32_e32 v1, v1, v5
	v_lshlrev_b32_e32 v122, 8, v1
	v_lshlrev_b32_e32 v123, 4, v2
	v_add_u32_e32 v118, v122, v123
	v_mul_hi_u32 v1, s40, v118
	v_mul_lo_u32 v6, v1, s34
	v_sub_u32_e32 v6, v118, v6
	v_cmp_le_u32_e32 vcc, s34, v6
	s_mul_i32 s14, s12, 0xc0
	v_and_b32_e32 v96, 48, v0
	v_subbrev_co_u32_e64 v1, s[2:3], 0, v1, vcc
	v_mov_b32_e32 v0, s34
	v_add_u32_e32 v1, s14, v1
	v_cndmask_b32_e32 v0, 0, v0, vcc
	v_lshlrev_b32_e32 v121, 4, v5
	v_add_u32_e32 v7, v1, v96
	s_and_b32 s2, s21, 0xffff
	v_add_u32_e32 v0, v6, v0
	v_or_b32_e32 v5, v121, v105
	s_or_b32 s21, s2, s28
	v_mad_u64_u32 v[6:7], s[2:3], v7, s34, v[0:1]
	s_add_i32 s49, s35, 0x10000
	v_add_u32_e32 v1, v5, v1
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	s_mov_b32 m0, s49
	v_add_u32_e32 v5, 64, v1
	s_add_i32 s50, s13, 0x10000
	buffer_load_dwordx4 v6, s[20:23], 0 offen lds
	v_mad_u64_u32 v[6:7], s[2:3], v5, s34, v[0:1]
	s_mov_b32 m0, s50
	v_add_u32_e32 v1, 0x80, v1
	s_add_i32 s51, s33, 0x10000
	buffer_load_dwordx4 v6, s[20:23], 0 offen lds
	v_mad_u64_u32 v[0:1], s[2:3], v1, s34, v[0:1]
	s_mov_b32 m0, s51
	s_add_u32 s17, s18, -1
	buffer_load_dwordx4 v0, s[20:23], 0 offen lds
	s_mov_b32 s28, 0
	s_addc_u32 s29, 0, 0
	s_cmp_lg_u64 s[28:29], 0
	s_mul_hi_u32 s15, s12, 0xc0
	s_cbranch_scc0 .LBB0_16
	v_cvt_f32_u32_e32 v0, s18
	v_mov_b32_e32 v1, 0x4f800000
	s_sub_u32 s28, 0, s18
	s_subb_u32 s30, 0, 0
	v_fmac_f32_e32 v0, 0, v1
	v_rcp_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x5f7ffffc, v0
	v_mul_f32_e32 v1, 0x2f800000, v0
	v_trunc_f32_e32 v1, v1
	v_fmamk_f32 v0, v1, 0xcf800000, v0
	v_cvt_u32_f32_e32 v1, v1
	v_cvt_u32_f32_e32 v0, v0
	v_readfirstlane_b32 s31, v1
	v_readfirstlane_b32 s36, v0
	s_mul_i32 s37, s28, s31
	s_mul_hi_u32 s39, s28, s36
	s_mul_i32 s38, s30, s36
	s_add_i32 s37, s39, s37
	s_add_i32 s37, s37, s38
	s_mul_i32 s41, s28, s36
	s_mul_i32 s39, s36, s37
	s_mul_hi_u32 s42, s36, s41
	s_mul_hi_u32 s38, s36, s37
	s_add_u32 s39, s42, s39
	s_addc_u32 s38, 0, s38
	s_mul_hi_u32 s43, s31, s41
	s_mul_i32 s41, s31, s41
	s_add_u32 s39, s39, s41
	s_mul_hi_u32 s42, s31, s37
	s_addc_u32 s38, s38, s43
	s_addc_u32 s39, s42, 0
	s_mul_i32 s37, s31, s37
	s_add_u32 s37, s38, s37
	s_addc_u32 s38, 0, s39
	s_add_u32 s36, s36, s37
	s_addc_u32 s31, s31, s38
	s_mul_i32 s37, s28, s31
	s_mul_hi_u32 s38, s28, s36
	s_add_i32 s37, s38, s37
	s_mul_i32 s30, s30, s36
	s_add_i32 s37, s37, s30
	s_mul_i32 s28, s28, s36
	s_mul_hi_u32 s38, s31, s28
	s_mul_i32 s39, s31, s28
	s_mul_i32 s42, s36, s37
	s_mul_hi_u32 s28, s36, s28
	s_mul_hi_u32 s41, s36, s37
	s_add_u32 s28, s28, s42
	s_addc_u32 s41, 0, s41
	s_add_u32 s28, s28, s39
	s_mul_hi_u32 s30, s31, s37
	s_addc_u32 s28, s41, s38
	s_addc_u32 s30, s30, 0
	s_mul_i32 s37, s31, s37
	s_add_u32 s28, s28, s37
	s_addc_u32 s30, 0, s30
	s_add_u32 s28, s36, s28
	s_addc_u32 s30, s31, s30
	s_mul_i32 s36, s17, s30
	s_mul_hi_u32 s37, s17, s28
	s_mul_hi_u32 s31, s17, s30
	s_add_u32 s36, s37, s36
	s_addc_u32 s31, 0, s31
	s_mul_i32 s28, s29, s28
	s_add_u32 s28, s36, s28
	s_addc_u32 s28, s31, 0
	s_addc_u32 s31, 0, 0
	s_mul_i32 s30, s29, s30
	s_add_u32 s28, s28, s30
	s_addc_u32 s30, 0, s31
	s_mul_i32 s30, s18, s30
	s_mul_hi_u32 s37, s18, s28
	s_add_u32 s31, s28, 1
	s_add_u32 s36, s28, 2
	s_add_i32 s37, s37, s30
	s_mul_i32 s30, s18, s28
	s_sub_u32 s30, s17, s30
	s_subb_u32 s29, s29, s37
	s_sub_u32 s37, s30, s18
	s_subb_u32 s38, s29, 0
	s_cmp_ge_u32 s37, s18
	s_cselect_b32 s37, -1, 0
	s_cmp_eq_u32 s38, 0
	s_cselect_b32 s37, s37, -1
	s_cmp_lg_u32 s37, 0
	s_cselect_b32 s31, s36, s31
	s_cmp_ge_u32 s30, s18
	s_cselect_b32 s30, -1, 0
	s_cmp_eq_u32 s29, 0
	s_cselect_b32 s29, s30, -1
	s_cmp_lg_u32 s29, 0
	s_cselect_b32 s46, s31, s28
	s_load_dwordx2 s[44:45], s[0:1], 0x48
	s_load_dwordx2 s[42:43], s[0:1], 0x58
	s_cbranch_execnz .LBB0_6
.LBB0_5:
	v_cvt_f32_u32_e32 v0, s18
	s_sub_i32 s2, 0, s18
	v_rcp_iflag_f32_e32 v0, v0
	s_nop 0
	v_mul_f32_e32 v0, 0x4f7ffffe, v0
	v_cvt_u32_f32_e32 v0, v0
	s_nop 0
	v_readfirstlane_b32 s3, v0
	s_mul_i32 s2, s2, s3
	s_mul_hi_u32 s2, s3, s2
	s_add_i32 s3, s3, s2
	s_mul_hi_u32 s2, s17, s3
	s_mul_i32 s28, s2, s18
	s_sub_i32 s17, s17, s28
	s_add_i32 s3, s2, 1
	s_sub_i32 s28, s17, s18
	s_cmp_ge_u32 s17, s18
	s_cselect_b32 s2, s3, s2
	s_cselect_b32 s17, s28, s17
	s_add_i32 s3, s2, 1
	s_cmp_ge_u32 s17, s18
	s_cselect_b32 s46, s3, s2
.LBB0_6:
	v_lshrrev_b32_e32 v7, 4, v97
	s_movk_i32 s2, 0xf800
	v_bfe_u32 v102, v97, 4, 2
	v_lshlrev_b32_e32 v5, 7, v97
	v_lshlrev_b32_e32 v6, 11, v102
	v_mad_i32_i24 v8, v7, s2, v5
	v_add_u32_e32 v124, v8, v6
	v_mul_hi_u32 v0, s46, v124
	v_mul_lo_u32 v1, s18, v0
	v_sub_u32_e32 v1, v124, v1
	v_cmp_le_u32_e32 vcc, s18, v1
	v_and_b32_e32 v103, 0xc0, v97
	s_mov_b32 s31, 0x27000
	v_subbrev_co_u32_e64 v9, s[2:3], 0, v0, vcc
	v_mov_b32_e32 v0, s18
	v_cndmask_b32_e32 v0, 0, v0, vcc
	v_add_u32_e32 v0, v0, v1
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s44, s7
	s_mul_hi_u32 s3, s44, s6
	v_xor_b32_e32 v0, v1, v0
	s_add_i32 s2, s3, s2
	s_mul_i32 s3, s45, s6
	v_ashrrev_i32_e32 v10, 31, v0
	s_add_i32 s2, s2, s3
	s_mul_i32 s3, s44, s6
	v_lshrrev_b32_e32 v10, 27, v10
	s_add_u32 s28, s4, s3
	v_add_u32_e32 v0, v0, v10
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s44, 0x3fff
	v_ashrrev_i32_e32 v0, 5, v0
	s_bitset1_b32 s3, 14
	v_xor_b32_e32 v0, v0, v1
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	v_add_u32_e32 v1, v103, v9
	s_or_b32 s29, s2, s3
	v_mad_u64_u32 v[10:11], s[2:3], v1, s44, v[0:1]
	s_movk_i32 s2, 0x60
	s_nop 0
	v_mul_lo_u32 v104, v3, s2
	s_mul_i32 s2, s42, s15
	s_mul_hi_u32 s3, s42, s14
	s_add_i32 s2, s3, s2
	s_mul_i32 s3, s43, s14
	s_lshl_b32 s17, s44, 5
	s_add_i32 s2, s2, s3
	s_mul_i32 s3, s42, s14
	s_add_u32 s36, s8, s3
	s_addc_u32 s2, s9, s2
	s_and_b32 s3, s42, 0x3fff
	s_mov_b32 s30, 0x7ffffffe
	v_add_u32_e32 v1, s17, v10
	s_bitset1_b32 s3, 14
	buffer_load_dword v115, v10, s[28:31], 0 offen
	buffer_load_dword v106, v1, s[28:31], 0 offen
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	v_add_u32_e32 v1, v104, v9
	s_or_b32 s37, s2, s3
	v_mad_u64_u32 v[0:1], s[2:3], v1, s42, v[0:1]
	s_lshl_b32 s41, s42, 5
	s_mov_b32 s38, s30
	s_mov_b32 s39, s31
	v_add_u32_e32 v1, s41, v0
	v_add_u32_e32 v9, s41, v1
	buffer_load_dword v114, v0, s[36:39], 0 offen
	buffer_load_dword v113, v1, s[36:39], 0 offen
	buffer_load_dword v112, v9, s[36:39], 0 offen
	v_cmp_eq_u32_e64 s[2:3], 0, v3
	v_mov_b32_e32 v1, 0
	s_and_b64 vcc, exec, s[2:3]
	s_waitcnt vmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB0_8
	s_barrier
.LBB0_8:
	s_sub_u32 s38, 0, s18
	s_subb_u32 s39, 0, s19
	s_add_u32 s43, s18, -1
	s_addc_u32 s45, s19, -1
	v_cmp_lt_i64_e64 s[4:5], s[18:19], 1
	v_and_b32_e32 v9, 7, v97
	s_mov_b64 s[8:9], 0
	v_cndmask_b32_e64 v0, 0, -1, s[4:5]
	s_and_b64 s[4:5], s[4:5], exec
	s_cselect_b32 s5, s39, s45
	s_cselect_b32 s4, s38, s43
	s_ashr_i32 s19, s5, 31
	s_lshr_b32 s19, s19, 24
	s_add_u32 s4, s4, s19
	s_addc_u32 s5, s5, 0
	s_ashr_i64 s[4:5], s[4:5], 8
	v_xor_b32_e32 v11, s5, v0
	v_xor_b32_e32 v10, s4, v0
	v_lshrrev_b32_e32 v0, 31, v11
	v_lshl_add_u64 v[98:99], v[10:11], 0, v[0:1]
	s_movk_i32 s4, 0x3000
	v_and_b32_e32 v98, -2, v98
	v_bitop3_b32 v0, v102, v97, 7 bitop3:0x78
	v_mul_lo_u32 v3, v3, s4
	v_sub_co_u32_e32 v100, vcc, v10, v98
	v_lshl_add_u32 v1, v4, 13, v8
	v_lshlrev_b32_e32 v0, 4, v0
	v_add_u32_e32 v3, v8, v3
	v_subb_co_u32_e32 v101, vcc, v11, v99, vcc
	v_or_b32_e32 v116, v1, v0
	v_or_b32_e32 v119, v3, v0
	v_bitop3_b32 v0, v102, v9, 4 bitop3:0x36
	v_lshlrev_b32_e32 v0, 4, v0
	v_cmp_gt_i64_e32 vcc, 1, v[98:99]
	v_or_b32_e32 v117, v0, v1
	v_or_b32_e32 v120, v0, v3
	s_cbranch_vccnz .LBB0_17
	v_mul_i32_i24_e32 v0, 0xfffff800, v7
	v_add_u32_e32 v0, v0, v6
	s_movk_i32 s4, 0x4000
	v_add3_u32 v127, v0, v5, s4
	s_mul_i32 s4, s12, 0xc0
	v_or_b32_e32 v0, s4, v105
	v_and_b32_e32 v1, 16, v2
	v_add_u32_e32 v0, v0, v1
	v_add_u32_e32 v128, 0x80, v0
	v_add_u32_e32 v129, 64, v0
	v_add_u32_e32 v0, v2, v105
	v_and_b32_e32 v1, 0x70, v0
	v_lshl_or_b32 v0, s16, 8, v0
	v_add_u32_e32 v130, s4, v1
	v_add_u32_e32 v1, 64, v0
	v_mul_lo_u32 v131, v1, s34
	v_or_b32_e32 v1, 0x80, v0
	v_mul_lo_u32 v132, v1, s34
	v_add_u32_e32 v1, 0xc0, v0
	v_mov_b32_e32 v56, 0
	s_movk_i32 s19, 0x800
	v_add_u32_e32 v125, 0x800, v118
	v_add_u32_e32 v126, 0x2000, v124
	v_mul_lo_u32 v133, v1, s34
	v_mul_lo_u32 v134, v0, s34
	s_add_i32 s12, s35, 0x8000
	s_add_i32 s16, s13, 0x8000
	s_add_i32 s43, s33, 0x8000
	s_add_i32 s45, s48, 0x8000
	v_mov_b32_e32 v135, s34
	s_add_i32 s47, s35, 0x16000
	s_add_i32 s52, s13, 0x16000
	s_add_i32 s53, s33, 0x16000
	v_add_u32_e32 v136, 0x10000, v119
	v_mov_b32_e32 v137, s18
	s_mov_b32 s38, s30
	s_mov_b32 s39, s31
	v_add_u32_e32 v138, 0x10000, v120
	s_movk_i32 s54, 0x1000
	v_add_u32_e32 v139, 0x16000, v119
	v_add_u32_e32 v140, 0x16000, v120
	v_mov_b32_e32 v57, v56
	v_mov_b32_e32 v58, v56
	v_mov_b32_e32 v59, v56
	v_mov_b32_e32 v92, v56
	v_mov_b32_e32 v93, v56
	v_mov_b32_e32 v94, v56
	v_mov_b32_e32 v95, v56
	v_mov_b32_e32 v88, v56
	v_mov_b32_e32 v89, v56
	v_mov_b32_e32 v90, v56
	v_mov_b32_e32 v91, v56
	v_mov_b32_e32 v84, v56
	v_mov_b32_e32 v85, v56
	v_mov_b32_e32 v86, v56
	v_mov_b32_e32 v87, v56
	v_mov_b32_e32 v80, v56
	v_mov_b32_e32 v81, v56
	v_mov_b32_e32 v82, v56
	v_mov_b32_e32 v83, v56
	v_mov_b32_e32 v76, v56
	v_mov_b32_e32 v77, v56
	v_mov_b32_e32 v78, v56
	v_mov_b32_e32 v79, v56
	v_mov_b32_e32 v72, v56
	v_mov_b32_e32 v73, v56
	v_mov_b32_e32 v74, v56
	v_mov_b32_e32 v75, v56
	v_mov_b32_e32 v68, v56
	v_mov_b32_e32 v69, v56
	v_mov_b32_e32 v70, v56
	v_mov_b32_e32 v71, v56
	v_mov_b32_e32 v64, v56
	v_mov_b32_e32 v65, v56
	v_mov_b32_e32 v66, v56
	v_mov_b32_e32 v67, v56
	v_mov_b32_e32 v60, v56
	v_mov_b32_e32 v61, v56
	v_mov_b32_e32 v62, v56
	v_mov_b32_e32 v63, v56
	v_mov_b32_e32 v52, v56
	v_mov_b32_e32 v53, v56
	v_mov_b32_e32 v54, v56
	v_mov_b32_e32 v55, v56
	v_mov_b32_e32 v48, v56
	v_mov_b32_e32 v49, v56
	v_mov_b32_e32 v50, v56
	v_mov_b32_e32 v51, v56
	v_mov_b32_e32 v44, v56
	v_mov_b32_e32 v45, v56
	v_mov_b32_e32 v46, v56
	v_mov_b32_e32 v47, v56
	v_mov_b32_e32 v40, v56
	v_mov_b32_e32 v41, v56
	v_mov_b32_e32 v42, v56
	v_mov_b32_e32 v43, v56
	v_mov_b32_e32 v36, v56
	v_mov_b32_e32 v37, v56
	v_mov_b32_e32 v38, v56
	v_mov_b32_e32 v39, v56
	v_mov_b32_e32 v20, v56
	v_mov_b32_e32 v21, v56
	v_mov_b32_e32 v22, v56
	v_mov_b32_e32 v23, v56
	v_mov_b32_e32 v12, v56
	v_mov_b32_e32 v13, v56
	v_mov_b32_e32 v14, v56
	v_mov_b32_e32 v15, v56
	v_mov_b32_e32 v16, v56
	v_mov_b32_e32 v17, v56
	v_mov_b32_e32 v18, v56
	v_mov_b32_e32 v19, v56
	v_mov_b32_e32 v24, v56
	v_mov_b32_e32 v25, v56
	v_mov_b32_e32 v26, v56
	v_mov_b32_e32 v27, v56
	v_mov_b32_e32 v28, v56
	v_mov_b32_e32 v29, v56
	v_mov_b32_e32 v30, v56
	v_mov_b32_e32 v31, v56
	v_mov_b32_e32 v32, v56
	v_mov_b32_e32 v33, v56
	v_mov_b32_e32 v34, v56
	v_mov_b32_e32 v35, v56
	v_mov_b32_e32 v8, v56
	v_mov_b32_e32 v9, v56
	v_mov_b32_e32 v10, v56
	v_mov_b32_e32 v11, v56
	v_mov_b32_e32 v4, v56
	v_mov_b32_e32 v5, v56
	v_mov_b32_e32 v6, v56
	v_mov_b32_e32 v7, v56
	v_mov_b32_e32 v0, v56
	v_mov_b32_e32 v1, v56
	v_mov_b32_e32 v2, v56
	v_mov_b32_e32 v3, v56
	v_mov_b32_e32 v141, v122
.LBB0_10:
	v_add_u32_e32 v146, v134, v108
	s_mov_b32 m0, s12
	v_add_u32_e32 v142, 0x80, v146
	v_add_u32_e32 v147, v131, v108
	s_waitcnt vmcnt(5)
	s_barrier
	buffer_load_dwordx4 v142, s[24:27], 0 offen lds
	v_add_u32_e32 v142, 0x80, v147
	s_mov_b32 m0, s16
	v_add_u32_e32 v148, v132, v108
	buffer_load_dwordx4 v142, s[24:27], 0 offen lds
	v_add_u32_e32 v142, 0x80, v148
	s_mov_b32 m0, s43
	v_add_u32_e32 v149, v133, v108
	buffer_load_dwordx4 v142, s[24:27], 0 offen lds
	v_add_u32_e32 v142, 0x80, v149
	s_mov_b32 m0, s45
	v_add_u32_e32 v151, v123, v122
	buffer_load_dwordx4 v142, s[24:27], 0 offen lds
	v_add_u32_e32 v142, 0x800, v151
	v_mul_hi_u32 v142, s40, v142
	v_lshl_add_u32 v150, s8, 11, v125
	v_mul_lo_u32 v142, v142, s34
	v_sub_u32_e32 v142, v150, v142
	v_cmp_le_u32_e32 vcc, s34, v142
	v_add_u32_e32 v152, v141, v123
	s_mov_b32 m0, s47
	v_cndmask_b32_e32 v142, 0, v135, vcc
	v_subbrev_co_u32_e64 v143, s[4:5], 0, v130, vcc
	v_mad_u64_u32 v[144:145], s[4:5], s34, v143, v[142:143]
	v_add3_u32 v143, v152, v144, s19
	buffer_load_dwordx4 v143, s[20:23], 0 offen lds
	v_subbrev_co_u32_e64 v143, s[4:5], 0, v129, vcc
	v_mad_u64_u32 v[144:145], s[4:5], s34, v143, v[142:143]
	v_add3_u32 v143, v152, v144, s19
	s_mov_b32 m0, s52
	s_nop 0
	buffer_load_dwordx4 v143, s[20:23], 0 offen lds
	v_subbrev_co_u32_e32 v143, vcc, 0, v128, vcc
	v_mad_u64_u32 v[142:143], s[4:5], s34, v143, v[142:143]
	v_add3_u32 v142, v152, v142, s19
	s_mov_b32 m0, s53
	s_nop 0
	buffer_load_dwordx4 v142, s[20:23], 0 offen lds
	ds_read_b128 v[142:145], v116
	ds_read_b128 v[154:157], v116 offset:2048
	ds_read_b128 v[158:161], v116 offset:4096
	ds_read_b128 v[162:165], v116 offset:6144
	ds_read_b128 v[166:169], v136
	ds_read_b128 v[170:173], v136 offset:2048
	ds_read_b128 v[174:177], v136 offset:4096
	ds_read_b128 v[178:181], v136 offset:6144
	ds_read_b128 v[182:185], v136 offset:8192
	ds_read_b128 v[186:189], v136 offset:10240
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(9) lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[142:145], v[166:169], v[56:59], v115, v114 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[142:145], v[170:173], v[92:95], v115, v114 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(8) lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[142:145], v[174:177], v[88:91], v115, v113 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[142:145], v[178:181], v[84:87], v115, v113 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(7) lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[142:145], v[182:185], v[80:83], v115, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[142:145], v[186:189], v[76:79], v115, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[154:157], v[166:169], v[72:75], v115, v114 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[154:157], v[170:173], v[68:71], v115, v114 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[154:157], v[174:177], v[64:67], v115, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[154:157], v[178:181], v[60:63], v115, v113 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[154:157], v[182:185], v[52:55], v115, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[154:157], v[186:189], v[48:51], v115, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[158:161], v[166:169], v[44:47], v106, v114 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[158:161], v[170:173], v[40:43], v106, v114 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[158:161], v[174:177], v[36:39], v106, v113 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[158:161], v[178:181], v[20:23], v106, v113 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[158:161], v[182:185], v[12:15], v106, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[158:161], v[186:189], v[16:19], v106, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[162:165], v[166:169], v[24:27], v106, v114 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[162:165], v[170:173], v[28:31], v106, v114 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[162:165], v[174:177], v[32:35], v106, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[162:165], v[178:181], v[8:11], v106, v113 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[162:165], v[182:185], v[4:7], v106, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[162:165], v[186:189], v[0:3], v106, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v142, 0xffffe000, v127
	v_mul_hi_u32 v142, s46, v142
	v_lshl_add_u32 v153, s8, 13, v126
	v_mul_lo_u32 v143, s18, v142
	v_sub_u32_e32 v143, v153, v143
	v_cmp_le_u32_e32 vcc, s18, v143
	ds_read_b128 v[156:159], v117
	ds_read_b128 v[160:163], v117 offset:2048
	ds_read_b128 v[164:167], v117 offset:4096
	ds_read_b128 v[168:171], v117 offset:6144
	ds_read_b128 v[172:175], v138
	ds_read_b128 v[176:179], v138 offset:2048
	ds_read_b128 v[180:183], v138 offset:4096
	ds_read_b128 v[184:187], v138 offset:6144
	ds_read_b128 v[188:191], v138 offset:8192
	ds_read_b128 v[192:195], v138 offset:10240
	v_subbrev_co_u32_e64 v154, s[4:5], 0, v142, vcc
	v_cndmask_b32_e32 v142, 0, v137, vcc
	v_add_u32_e32 v142, v142, v143
	v_ashrrev_i32_e32 v143, 31, v142
	v_xor_b32_e32 v142, v143, v142
	v_ashrrev_i32_e32 v144, 31, v142
	v_lshrrev_b32_e32 v144, 27, v144
	v_add_u32_e32 v142, v142, v144
	v_ashrrev_i32_e32 v142, 5, v142
	v_xor_b32_e32 v142, v142, v143
	v_add_u32_e32 v143, v103, v154
	v_mad_u64_u32 v[144:145], s[4:5], v143, s44, v[142:143]
	v_add_u32_e32 v143, s17, v144
	buffer_load_dword v144, v144, s[28:31], 0 offen
	s_nop 0
	buffer_load_dword v145, v143, s[28:31], 0 offen
	v_add_u32_e32 v143, v104, v154
	v_mad_u64_u32 v[142:143], s[4:5], v143, s42, v[142:143]
	v_add_u32_e32 v143, s41, v142
	v_add_u32_e32 v154, s41, v143
	buffer_load_dword v142, v142, s[36:39], 0 offen
	s_nop 0
	buffer_load_dword v143, v143, s[36:39], 0 offen
	s_nop 0
	buffer_load_dword v154, v154, s[36:39], 0 offen
	s_waitcnt vmcnt(7)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[156:159], v[172:175], v[56:59], v115, v114 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[156:159], v[176:179], v[92:95], v115, v114 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[156:159], v[180:183], v[88:91], v115, v113 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[156:159], v[184:187], v[84:87], v115, v113 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[156:159], v[188:191], v[80:83], v115, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[156:159], v[192:195], v[76:79], v115, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[160:163], v[172:175], v[72:75], v115, v114 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[160:163], v[176:179], v[68:71], v115, v114 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[160:163], v[180:183], v[64:67], v115, v113 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[160:163], v[184:187], v[60:63], v115, v113 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[160:163], v[188:191], v[52:55], v115, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[160:163], v[192:195], v[48:51], v115, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[164:167], v[172:175], v[44:47], v106, v114 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[164:167], v[176:179], v[40:43], v106, v114 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[164:167], v[180:183], v[36:39], v106, v113 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[164:167], v[184:187], v[20:23], v106, v113 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[164:167], v[188:191], v[12:15], v106, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[164:167], v[192:195], v[16:19], v106, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[168:171], v[172:175], v[24:27], v106, v114 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[168:171], v[176:179], v[28:31], v106, v114 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[168:171], v[180:183], v[32:35], v106, v113 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[168:171], v[184:187], v[8:11], v106, v113 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[168:171], v[188:191], v[4:7], v106, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[168:171], v[192:195], v[0:3], v106, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_mov_b32 m0, s35
	v_add_u32_e32 v106, 0x100, v146
	s_waitcnt vmcnt(5)
	s_barrier
	buffer_load_dwordx4 v106, s[24:27], 0 offen lds
	v_add_u32_e32 v106, 0x100, v147
	s_mov_b32 m0, s13
	s_nop 0
	buffer_load_dwordx4 v106, s[24:27], 0 offen lds
	v_add_u32_e32 v106, 0x100, v148
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v106, s[24:27], 0 offen lds
	v_add_u32_e32 v106, 0x100, v149
	s_mov_b32 m0, s48
	s_nop 0
	buffer_load_dwordx4 v106, s[24:27], 0 offen lds
	v_add_u32_e32 v106, 0x1000, v151
	v_mul_hi_u32 v106, s40, v106
	v_mul_lo_u32 v106, v106, s34
	v_sub_u32_e32 v106, v150, v106
	v_add_u32_e32 v106, 0x800, v106
	v_cmp_le_u32_e32 vcc, s34, v106
	s_mov_b32 m0, s49
	s_nop 0
	v_cndmask_b32_e32 v106, 0, v135, vcc
	v_subbrev_co_u32_e64 v112, s[4:5], 0, v130, vcc
	v_mad_u64_u32 v[112:113], s[4:5], s34, v112, v[106:107]
	v_add3_u32 v112, v152, v112, s54
	buffer_load_dwordx4 v112, s[20:23], 0 offen lds
	v_subbrev_co_u32_e64 v112, s[4:5], 0, v129, vcc
	v_mad_u64_u32 v[112:113], s[4:5], s34, v112, v[106:107]
	v_add3_u32 v112, v152, v112, s54
	s_mov_b32 m0, s50
	s_nop 0
	buffer_load_dwordx4 v112, s[20:23], 0 offen lds
	v_subbrev_co_u32_e32 v112, vcc, 0, v128, vcc
	v_mad_u64_u32 v[112:113], s[4:5], s34, v112, v[106:107]
	v_add3_u32 v106, v152, v112, s54
	s_mov_b32 m0, s51
	s_nop 0
	buffer_load_dwordx4 v106, s[20:23], 0 offen lds
	ds_read_b128 v[112:115], v116 offset:32768
	ds_read_b128 v[146:149], v116 offset:34816
	ds_read_b128 v[156:159], v116 offset:36864
	ds_read_b128 v[160:163], v116 offset:38912
	ds_read_b128 v[164:167], v139
	ds_read_b128 v[168:171], v139 offset:2048
	ds_read_b128 v[172:175], v139 offset:4096
	ds_read_b128 v[176:179], v139 offset:6144
	ds_read_b128 v[180:183], v139 offset:8192
	ds_read_b128 v[184:187], v139 offset:10240
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(9) lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[112:115], v[164:167], v[56:59], v144, v142 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[112:115], v[168:171], v[92:95], v144, v142 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(8) lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[112:115], v[172:175], v[88:91], v144, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[112:115], v[176:179], v[84:87], v144, v143 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(7) lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[112:115], v[180:183], v[80:83], v144, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[112:115], v[184:187], v[76:79], v144, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[146:149], v[164:167], v[72:75], v144, v142 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[146:149], v[168:171], v[68:71], v144, v142 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[146:149], v[172:175], v[64:67], v144, v143 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[146:149], v[176:179], v[60:63], v144, v143 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[146:149], v[180:183], v[52:55], v144, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[146:149], v[184:187], v[48:51], v144, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[156:159], v[164:167], v[44:47], v145, v142 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[156:159], v[168:171], v[40:43], v145, v142 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[156:159], v[172:175], v[36:39], v145, v143 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[156:159], v[176:179], v[20:23], v145, v143 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[156:159], v[180:183], v[12:15], v145, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[156:159], v[184:187], v[16:19], v145, v154 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[160:163], v[164:167], v[24:27], v145, v142 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[160:163], v[168:171], v[28:31], v145, v142 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[160:163], v[172:175], v[32:35], v145, v143 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[160:163], v[176:179], v[8:11], v145, v143 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[160:163], v[180:183], v[4:7], v145, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[160:163], v[184:187], v[0:3], v145, v154 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_mul_hi_u32 v106, s46, v127
	v_mul_lo_u32 v112, s18, v106
	v_sub_u32_e32 v112, v153, v112
	v_add_u32_e32 v112, 0x2000, v112
	v_cmp_le_u32_e32 vcc, s18, v112
	s_nop 1
	v_subbrev_co_u32_e64 v113, s[4:5], 0, v106, vcc
	v_cndmask_b32_e32 v106, 0, v137, vcc
	v_add_u32_e32 v106, v106, v112
	v_ashrrev_i32_e32 v112, 31, v106
	v_xor_b32_e32 v106, v112, v106
	v_ashrrev_i32_e32 v114, 31, v106
	v_lshrrev_b32_e32 v114, 27, v114
	v_add_u32_e32 v106, v106, v114
	v_ashrrev_i32_e32 v106, 5, v106
	v_xor_b32_e32 v112, v106, v112
	v_add_u32_e32 v106, v103, v113
	v_mad_u64_u32 v[114:115], s[4:5], v106, s44, v[112:113]
	v_add_u32_e32 v113, v104, v113
	v_mad_u64_u32 v[112:113], s[4:5], v113, s42, v[112:113]
	v_add_u32_e32 v106, s17, v114
	v_add_u32_e32 v113, s41, v112
	buffer_load_dword v115, v114, s[28:31], 0 offen
	s_nop 0
	buffer_load_dword v106, v106, s[28:31], 0 offen
	v_add_u32_e32 v146, s41, v113
	buffer_load_dword v114, v112, s[36:39], 0 offen
	s_nop 0
	buffer_load_dword v113, v113, s[36:39], 0 offen
	s_nop 0
	buffer_load_dword v112, v146, s[36:39], 0 offen
	ds_read_b128 v[146:149], v117 offset:32768
	ds_read_b128 v[150:153], v117 offset:34816
	ds_read_b128 v[156:159], v117 offset:36864
	ds_read_b128 v[160:163], v117 offset:38912
	ds_read_b128 v[164:167], v140
	ds_read_b128 v[168:171], v140 offset:2048
	ds_read_b128 v[172:175], v140 offset:4096
	ds_read_b128 v[176:179], v140 offset:6144
	ds_read_b128 v[180:183], v140 offset:8192
	ds_read_b128 v[184:187], v140 offset:10240
	s_waitcnt vmcnt(7)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[146:149], v[164:167], v[56:59], v144, v142 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[146:149], v[168:171], v[92:95], v144, v142 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[146:149], v[172:175], v[88:91], v144, v143 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[146:149], v[176:179], v[84:87], v144, v143 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[146:149], v[180:183], v[80:83], v144, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[146:149], v[184:187], v[76:79], v144, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[150:153], v[164:167], v[72:75], v144, v142 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[150:153], v[168:171], v[68:71], v144, v142 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[150:153], v[172:175], v[64:67], v144, v143 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[150:153], v[176:179], v[60:63], v144, v143 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[150:153], v[180:183], v[52:55], v144, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[150:153], v[184:187], v[48:51], v144, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[156:159], v[164:167], v[44:47], v145, v142 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[156:159], v[168:171], v[40:43], v145, v142 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[156:159], v[172:175], v[36:39], v145, v143 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[156:159], v[176:179], v[20:23], v145, v143 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[156:159], v[180:183], v[12:15], v145, v154 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[156:159], v[184:187], v[16:19], v145, v154 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[160:163], v[164:167], v[24:27], v145, v142 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[160:163], v[168:171], v[28:31], v145, v142 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[160:163], v[172:175], v[32:35], v145, v143 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[160:163], v[176:179], v[8:11], v145, v143 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[160:163], v[180:183], v[4:7], v145, v154 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[160:163], v[184:187], v[0:3], v145, v154 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_add_u32 s8, s8, 2
	s_addc_u32 s9, s9, 0
	v_cmp_ge_i64_e32 vcc, s[8:9], v[98:99]
	v_add_u32_e32 v127, 0x4000, v127
	v_add_u32_e32 v141, 0x1000, v141
	v_add_u32_e32 v122, 0x1000, v122
	v_add_u32_e32 v131, 0x100, v131
	v_add_u32_e32 v132, 0x100, v132
	v_add_u32_e32 v133, 0x100, v133
	v_add_u32_e32 v134, 0x100, v134
	s_cbranch_vccz .LBB0_10
	s_load_dwordx2 s[4:5], s[0:1], 0x60
	v_cmp_lt_i64_e32 vcc, 0, v[100:101]
	s_cbranch_vccz .LBB0_18
.LBB0_12:
	v_or_b32_e32 v100, s14, v105
	v_lshl_add_u32 v99, v98, 13, v124
	v_or_b32_e32 v100, v100, v121
	v_or_b32_e32 v96, s14, v96
	v_lshl_add_u32 v101, v98, 11, v118
	v_lshl_or_b32 v98, v98, 7, v108
	s_movk_i32 s0, 0x80
	v_add3_u32 v105, v98, v107, s0
	s_add_i32 m0, s35, 0x8000
	s_waitcnt vmcnt(5)
	s_barrier
	buffer_load_dwordx4 v105, s[24:27], 0 offen lds
	v_add3_u32 v105, v98, v109, s0
	s_add_i32 m0, s13, 0x8000
	s_movk_i32 s8, 0x800
	buffer_load_dwordx4 v105, s[24:27], 0 offen lds
	v_add3_u32 v105, v98, v110, s0
	s_add_i32 m0, s33, 0x8000
	v_add3_u32 v98, v98, v111, s0
	buffer_load_dwordx4 v105, s[24:27], 0 offen lds
	s_add_i32 m0, s48, 0x8000
	s_mov_b32 s9, 0x16000
	buffer_load_dwordx4 v98, s[24:27], 0 offen lds
	v_add_u32_e32 v98, 0x800, v101
	v_mul_hi_u32 v101, s40, v98
	v_mul_lo_u32 v105, v101, s34
	v_sub_u32_e32 v98, v98, v105
	v_cmp_le_u32_e32 vcc, s34, v98
	v_mov_b32_e32 v105, s34
	s_add_i32 m0, s35, 0x16000
	v_subbrev_co_u32_e64 v101, s[0:1], 0, v101, vcc
	v_add_u32_e32 v96, v96, v101
	v_cndmask_b32_e32 v105, 0, v105, vcc
	s_movk_i32 s0, 0xf800
	v_mul_lo_u32 v96, v96, s34
	v_add3_u32 v98, v98, v105, s0
	v_add3_u32 v96, v98, v96, s8
	buffer_load_dwordx4 v96, s[20:23], 0 offen lds
	v_add_u32_e32 v96, v101, v100
	v_add_u32_e32 v100, 64, v96
	v_mul_lo_u32 v100, v100, s34
	v_add_u32_e32 v96, 0x80, v96
	v_add3_u32 v100, v98, v100, s8
	s_add_i32 m0, s13, 0x16000
	v_mul_lo_u32 v96, v96, s34
	buffer_load_dwordx4 v100, s[20:23], 0 offen lds
	v_add3_u32 v96, v98, v96, s8
	s_add_i32 m0, s33, 0x16000
	s_mov_b32 s8, 0x8000
	buffer_load_dwordx4 v96, s[20:23], 0 offen lds
	v_add_u32_e32 v96, 0x10000, v119
	ds_read_b128 v[134:137], v96
	ds_read_b128 v[138:141], v96 offset:2048
	ds_read_b128 v[142:145], v96 offset:4096
	ds_read_b128 v[146:149], v96 offset:6144
	ds_read_b128 v[150:153], v96 offset:8192
	ds_read_b128 v[154:157], v96 offset:10240
	ds_read_b128 v[108:111], v116
	ds_read_b128 v[122:125], v116 offset:2048
	ds_read_b128 v[126:129], v116 offset:4096
	ds_read_b128 v[130:133], v116 offset:6144
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(9) lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[108:111], v[134:137], v[56:59], v115, v114 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[108:111], v[138:141], v[92:95], v115, v114 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(8)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[108:111], v[142:145], v[88:91], v115, v113 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[108:111], v[146:149], v[84:87], v115, v113 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[108:111], v[150:153], v[80:83], v115, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[108:111], v[154:157], v[76:79], v115, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[122:125], v[134:137], v[72:75], v115, v114 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[122:125], v[138:141], v[68:71], v115, v114 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[122:125], v[142:145], v[64:67], v115, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[122:125], v[146:149], v[60:63], v115, v113 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[122:125], v[150:153], v[52:55], v115, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[122:125], v[154:157], v[48:51], v115, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[126:129], v[134:137], v[44:47], v106, v114 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[126:129], v[138:141], v[40:43], v106, v114 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[126:129], v[142:145], v[36:39], v106, v113 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[126:129], v[146:149], v[20:23], v106, v113 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[126:129], v[150:153], v[12:15], v106, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[126:129], v[154:157], v[16:19], v106, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[130:133], v[134:137], v[24:27], v106, v114 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[130:133], v[138:141], v[28:31], v106, v114 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[130:133], v[142:145], v[32:35], v106, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[130:133], v[146:149], v[8:11], v106, v113 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[130:133], v[150:153], v[4:7], v106, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[130:133], v[154:157], v[0:3], v106, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v96, 0x2000, v99
	v_mul_hi_u32 v98, s46, v96
	v_mul_lo_u32 v99, s18, v98
	v_sub_u32_e32 v96, v96, v99
	v_cmp_le_u32_e32 vcc, s18, v96
	s_mov_b32 s38, s30
	s_mov_b32 s39, s31
	v_subbrev_co_u32_e64 v100, s[0:1], 0, v98, vcc
	v_mov_b32_e32 v98, s18
	v_cndmask_b32_e32 v98, 0, v98, vcc
	v_add_u32_e32 v96, v98, v96
	v_ashrrev_i32_e32 v98, 31, v96
	v_xor_b32_e32 v96, v98, v96
	v_ashrrev_i32_e32 v99, 31, v96
	v_lshrrev_b32_e32 v99, 27, v99
	v_add_u32_e32 v96, v96, v99
	v_ashrrev_i32_e32 v96, 5, v96
	v_xor_b32_e32 v96, v96, v98
	v_add_u32_e32 v98, v103, v100
	v_mad_u64_u32 v[98:99], s[0:1], v98, s44, v[96:97]
	v_add_u32_e32 v99, s17, v98
	buffer_load_dword v101, v98, s[28:31], 0 offen
	buffer_load_dword v105, v99, s[28:31], 0 offen
	v_add_u32_e32 v98, v104, v100
	v_mad_u64_u32 v[98:99], s[0:1], v98, s42, v[96:97]
	v_add_u32_e32 v96, s41, v98
	v_add_u32_e32 v99, s41, v96
	buffer_load_dword v100, v98, s[36:39], 0 offen
	buffer_load_dword v107, v96, s[36:39], 0 offen
	buffer_load_dword v108, v99, s[36:39], 0 offen
	v_add_u32_e32 v96, 0x10000, v120
	ds_read_b128 v[138:141], v96
	ds_read_b128 v[142:145], v96 offset:2048
	ds_read_b128 v[146:149], v96 offset:4096
	ds_read_b128 v[150:153], v96 offset:6144
	ds_read_b128 v[154:157], v96 offset:8192
	ds_read_b128 v[158:161], v96 offset:10240
	ds_read_b128 v[122:125], v117
	ds_read_b128 v[126:129], v117 offset:2048
	ds_read_b128 v[130:133], v117 offset:4096
	ds_read_b128 v[134:137], v117 offset:6144
	s_waitcnt vmcnt(7)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[122:125], v[138:141], v[56:59], v115, v114 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[122:125], v[142:145], v[92:95], v115, v114 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[122:125], v[146:149], v[88:91], v115, v113 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[122:125], v[150:153], v[84:87], v115, v113 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[122:125], v[154:157], v[80:83], v115, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[122:125], v[158:161], v[76:79], v115, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[126:129], v[138:141], v[72:75], v115, v114 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[126:129], v[142:145], v[68:71], v115, v114 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[126:129], v[146:149], v[64:67], v115, v113 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[126:129], v[150:153], v[60:63], v115, v113 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[126:129], v[154:157], v[52:55], v115, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[126:129], v[158:161], v[48:51], v115, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[130:133], v[138:141], v[44:47], v106, v114 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[130:133], v[142:145], v[40:43], v106, v114 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[130:133], v[146:149], v[36:39], v106, v113 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[130:133], v[150:153], v[20:23], v106, v113 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[130:133], v[154:157], v[12:15], v106, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[130:133], v[158:161], v[16:19], v106, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[134:137], v[138:141], v[24:27], v106, v114 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[134:137], v[142:145], v[28:31], v106, v114 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[134:137], v[146:149], v[32:35], v106, v113 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[134:137], v[150:153], v[8:11], v106, v113 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[134:137], v[154:157], v[4:7], v106, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[134:137], v[158:161], v[0:3], v106, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_waitcnt vmcnt(4)
	v_mov_b32_e32 v115, v101
	s_waitcnt vmcnt(3)
	v_mov_b32_e32 v106, v105
	s_waitcnt vmcnt(2)
	v_mov_b32_e32 v114, v100
	s_waitcnt vmcnt(1)
	v_mov_b32_e32 v113, v107
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v112, v108
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_14
.LBB0_13:
	s_barrier
.LBB0_14:
	v_add_u32_e32 v96, s9, v119
	v_add_u32_e32 v98, s9, v120
	s_waitcnt vmcnt(5) lgkmcnt(0)
	s_barrier
	ds_read_b128 v[166:169], v96
	ds_read_b128 v[170:173], v96 offset:2048
	ds_read_b128 v[174:177], v98
	ds_read_b128 v[178:181], v98 offset:2048
	ds_read_b128 v[182:185], v96 offset:4096
	ds_read_b128 v[134:137], v96 offset:6144
	ds_read_b128 v[186:189], v98 offset:4096
	ds_read_b128 v[126:129], v98 offset:6144
	ds_read_b128 v[122:125], v96 offset:8192
	ds_read_b128 v[108:111], v96 offset:10240
	ds_read_b128 v[118:121], v98 offset:8192
	ds_read_b128 v[98:101], v98 offset:10240
	v_add_u32_e32 v96, s8, v116
	ds_read_b128 v[154:157], v96
	ds_read_b128 v[190:193], v96 offset:2048
	s_waitcnt vmcnt(2) lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[154:157], v[166:169], v[56:59], v115, v114 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v105, s8, v117
	ds_read_b128 v[158:161], v105
	ds_read_b128 v[194:197], v105 offset:2048
	ds_read_b128 v[198:201], v96 offset:4096
	ds_read_b128 v[138:141], v96 offset:6144
	ds_read_b128 v[202:205], v105 offset:4096
	ds_read_b128 v[130:133], v105 offset:6144
	s_movk_i32 s0, 0x7fff
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[158:161], v[174:177], v[56:59], v115, v114 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v96, 0x7fc0
	s_mul_hi_u32 s1, s4, s6
	v_and_b32_e32 v97, 15, v97
	s_waitcnt vmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[154:157], v[182:185], v[88:91], v115, v113 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[158:161], v[186:189], v[88:91], v115, v113 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 1
	v_bfe_u32 v105, v59, 16, 1
	v_bfe_u32 v107, v58, 16, 1
	v_add3_u32 v105, v59, v105, s0
	v_bfe_u32 v116, v57, 16, 1
	v_add3_u32 v107, v58, v107, s0
	v_lshrrev_b32_e32 v105, 16, v105
	v_cmp_o_f32_e32 vcc, v59, v59
	v_add3_u32 v116, v57, v116, s0
	v_lshrrev_b32_e32 v107, 16, v107
	v_cndmask_b32_e32 v88, v96, v105, vcc
	v_cmp_o_f32_e32 vcc, v58, v58
	v_bfe_u32 v117, v56, 16, 1
	v_lshrrev_b32_e32 v89, 16, v116
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[154:157], v[134:137], v[84:87], v115, v113 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v117, v56, v117, s0
	v_lshrrev_b32_e32 v90, 16, v117
	s_nop 0
	v_cndmask_b32_e32 v84, v96, v107, vcc
	v_cmp_o_f32_e32 vcc, v57, v57
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[154:157], v[170:173], v[92:95], v115, v114 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v85, v96, v89, vcc
	v_cmp_o_f32_e32 vcc, v56, v56
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[154:157], v[122:125], v[80:83], v115, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v86, v96, v90, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[158:161], v[118:121], v[56:59], v115, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[154:157], v[108:111], v[76:79], v115, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], v[158:161], v[98:101], v[56:59], v115, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[190:193], v[166:169], v[72:75], v115, v114 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[158:161], v[178:181], v[92:95], v115, v114 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 1
	v_bfe_u32 v72, v145, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[158:161], v[126:129], v[146:149], v115, v113 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], v[194:197], v[174:177], v[56:59], v115, v114 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 1
	v_bfe_u32 v87, v95, 16, 1
	v_bfe_u32 v80, v94, 16, 1
	v_add3_u32 v76, v95, v87, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[190:193], v[170:173], v[68:71], v115, v114 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v81, v93, 16, 1
	v_add3_u32 v80, v94, v80, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], v[194:197], v[178:181], v[56:59], v115, v114 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v95, v95
	v_bfe_u32 v82, v92, 16, 1
	v_add3_u32 v81, v93, v81, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[190:193], v[182:185], v[64:67], v115, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v77, 16, v80
	v_cndmask_b32_e32 v76, v96, v76, vcc
	v_cmp_o_f32_e32 vcc, v94, v94
	v_add3_u32 v82, v92, v82, s0
	v_lshrrev_b32_e32 v78, 16, v81
	v_cndmask_b32_e32 v77, v96, v77, vcc
	v_cmp_o_f32_e32 vcc, v93, v93
	v_lshrrev_b32_e32 v79, 16, v82
	v_bfe_u32 v68, v144, 16, 1
	v_cndmask_b32_e32 v78, v96, v78, vcc
	v_cmp_o_f32_e32 vcc, v92, v92
	v_add3_u32 v64, v145, v72, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[194:197], v[186:189], v[56:59], v115, v113 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v79, v96, v79, vcc
	v_bfe_u32 v69, v143, 16, 1
	v_add3_u32 v68, v144, v68, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[190:193], v[134:137], v[60:63], v115, v113 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v64, 16, v64
	v_cmp_o_f32_e32 vcc, v145, v145
	v_bfe_u32 v70, v142, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[190:193], v[122:125], v[52:55], v115, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v69, v143, v69, s0
	v_lshrrev_b32_e32 v65, 16, v68
	v_cndmask_b32_e32 v80, v96, v64, vcc
	v_cmp_o_f32_e32 vcc, v144, v144
	v_add3_u32 v70, v142, v70, s0
	v_lshrrev_b32_e32 v66, 16, v69
	v_cndmask_b32_e32 v81, v96, v65, vcc
	v_cmp_o_f32_e32 vcc, v143, v143
	v_lshrrev_b32_e32 v67, 16, v70
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[194:197], v[126:129], v[56:59], v115, v113 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v82, v96, v66, vcc
	v_cmp_o_f32_e32 vcc, v142, v142
	s_nop 0
	v_bfe_u32 v56, v149, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[190:193], v[108:111], v[48:51], v115, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v83, v96, v67, vcc
	v_bfe_u32 v57, v148, 16, 1
	v_bfe_u32 v58, v147, 16, 1
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[198:201], v[166:169], v[44:47], v106, v114 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v149, v149
	v_bfe_u32 v59, v146, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[194:197], v[118:121], v[52:55], v115, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 2
	v_add3_u32 v55, v149, v56, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[198:201], v[170:173], v[40:43], v106, v114 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v54, v148, v57, s0
	v_lshrrev_b32_e32 v55, 16, v55
	v_add3_u32 v53, v147, v58, s0
	v_lshrrev_b32_e32 v54, 16, v54
	v_cndmask_b32_e32 v87, v96, v55, vcc
	v_cmp_o_f32_e32 vcc, v148, v148
	v_add3_u32 v52, v146, v59, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[194:197], v[98:101], v[48:51], v115, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v89, v96, v54, vcc
	v_cmp_o_f32_e32 vcc, v147, v147
	s_nop 0
	v_lshrrev_b32_e32 v48, 16, v53
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[202:205], v[174:177], v[44:47], v106, v114 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v49, 16, v52
	v_cndmask_b32_e32 v90, v96, v48, vcc
	v_cmp_o_f32_e32 vcc, v146, v146
	v_bfe_u32 v44, v153, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[198:201], v[122:125], v[12:15], v106, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v45, v152, 16, 1
	v_cndmask_b32_e32 v91, v96, v49, vcc
	v_bfe_u32 v46, v151, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[202:205], v[178:181], v[40:43], v106, v114 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_bfe_u32 v47, v150, 16, 1
	v_cmp_o_f32_e32 vcc, v153, v153
	s_nop 0
	v_add3_u32 v43, v153, v44, s0
	v_add3_u32 v42, v152, v45, s0
	v_lshrrev_b32_e32 v43, 16, v43
	v_add3_u32 v40, v150, v47, s0
	v_add3_u32 v41, v151, v46, s0
	v_lshrrev_b32_e32 v42, 16, v42
	v_cndmask_b32_e32 v92, v96, v43, vcc
	v_cmp_o_f32_e32 vcc, v152, v152
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[198:201], v[182:185], v[36:39], v106, v113 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v94, 16, v41
	v_lshrrev_b32_e32 v95, 16, v40
	v_cndmask_b32_e32 v93, v96, v42, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[202:205], v[118:121], v[12:15], v106, v112 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v151, v151
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[198:201], v[108:111], v[16:19], v106, v112 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[202:205], v[186:189], v[36:39], v106, v113 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[198:201], v[134:137], v[20:23], v106, v113 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[202:205], v[98:101], v[12:15], v106, v112 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[138:141], v[166:169], v[24:27], v106, v114 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_bfe_u32 v26, v157, 16, 1
	v_cndmask_b32_e32 v24, v96, v94, vcc
	v_cmp_o_f32_e32 vcc, v150, v150
	v_bfe_u32 v27, v156, 16, 1
	v_add3_u32 v26, v157, v26, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[202:205], v[126:129], v[20:23], v106, v113 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v25, v96, v95, vcc
	v_add3_u32 v27, v156, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[130:133], v[174:177], v[12:15], v106, v114 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v157, v157
	v_lshrrev_b32_e32 v27, 16, v27
	v_bfe_u32 v94, v163, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[138:141], v[170:173], v[28:31], v106, v114 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v26, v96, v26, vcc
	v_cmp_o_f32_e32 vcc, v156, v156
	v_bfe_u32 v95, v162, 16, 1
	v_bfe_u32 v28, v155, 16, 1
	v_bfe_u32 v29, v154, 16, 1
	v_add3_u32 v28, v155, v28, s0
	v_add3_u32 v29, v154, v29, s0
	v_lshrrev_b32_e32 v28, 16, v28
	v_cndmask_b32_e32 v27, v96, v27, vcc
	v_cmp_o_f32_e32 vcc, v155, v155
	v_bfe_u32 v30, v161, 16, 1
	v_lshrrev_b32_e32 v29, 16, v29
	v_cndmask_b32_e32 v28, v96, v28, vcc
	v_cmp_o_f32_e32 vcc, v154, v154
	v_bfe_u32 v31, v160, 16, 1
	v_add3_u32 v30, v161, v30, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[130:133], v[178:181], v[12:15], v106, v114 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v29, v96, v29, vcc
	v_add3_u32 v31, v160, v31, s0
	v_lshrrev_b32_e32 v30, 16, v30
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[138:141], v[182:185], v[32:35], v106, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v161, v161
	v_lshrrev_b32_e32 v31, 16, v31
	v_add3_u32 v94, v163, v94, s0
	v_bfe_u32 v32, v159, 16, 1
	v_bfe_u32 v33, v158, 16, 1
	v_add3_u32 v32, v159, v32, s0
	v_cndmask_b32_e32 v30, v96, v30, vcc
	v_cmp_o_f32_e32 vcc, v160, v160
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[138:141], v[108:111], v[0:3], v106, v112 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v33, v158, v33, s0
	v_lshrrev_b32_e32 v32, 16, v32
	v_cndmask_b32_e32 v31, v96, v31, vcc
	v_cmp_o_f32_e32 vcc, v159, v159
	v_bfe_u32 v34, v165, 16, 1
	v_lshrrev_b32_e32 v33, 16, v33
	v_cndmask_b32_e32 v32, v96, v32, vcc
	v_cmp_o_f32_e32 vcc, v158, v158
	v_bfe_u32 v35, v164, 16, 1
	v_add3_u32 v34, v165, v34, s0
	v_cndmask_b32_e32 v33, v96, v33, vcc
	v_add3_u32 v35, v164, v35, s0
	v_lshrrev_b32_e32 v34, 16, v34
	v_cmp_o_f32_e32 vcc, v165, v165
	v_lshrrev_b32_e32 v35, 16, v35
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[130:133], v[98:101], v[0:3], v106, v112 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v34, v96, v34, vcc
	v_cmp_o_f32_e32 vcc, v164, v164
	v_add3_u32 v95, v162, v95, s0
	v_lshrrev_b32_e32 v94, 16, v94
	v_cndmask_b32_e32 v35, v96, v35, vcc
	v_cmp_o_f32_e32 vcc, v163, v163
	v_bfe_u32 v98, v75, 16, 1
	v_lshrrev_b32_e32 v95, 16, v95
	v_cndmask_b32_e32 v94, v96, v94, vcc
	v_cmp_o_f32_e32 vcc, v162, v162
	v_bfe_u32 v99, v74, 16, 1
	v_add3_u32 v98, v75, v98, s0
	v_cndmask_b32_e32 v95, v96, v95, vcc
	v_bfe_u32 v100, v73, 16, 1
	v_add3_u32 v99, v74, v99, s0
	v_lshrrev_b32_e32 v98, 16, v98
	v_cmp_o_f32_e32 vcc, v75, v75
	v_bfe_u32 v101, v72, 16, 1
	v_add3_u32 v100, v73, v100, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_cndmask_b32_e32 v75, v96, v98, vcc
	v_cmp_o_f32_e32 vcc, v74, v74
	v_add3_u32 v101, v72, v101, s0
	v_lshrrev_b32_e32 v100, 16, v100
	v_cndmask_b32_e32 v74, v96, v99, vcc
	v_cmp_o_f32_e32 vcc, v73, v73
	v_bfe_u32 v98, v71, 16, 1
	v_lshrrev_b32_e32 v101, 16, v101
	v_cndmask_b32_e32 v73, v96, v100, vcc
	v_cmp_o_f32_e32 vcc, v72, v72
	v_bfe_u32 v99, v70, 16, 1
	v_add3_u32 v98, v71, v98, s0
	v_cndmask_b32_e32 v72, v96, v101, vcc
	v_bfe_u32 v100, v69, 16, 1
	v_add3_u32 v99, v70, v99, s0
	v_lshrrev_b32_e32 v98, 16, v98
	v_cmp_o_f32_e32 vcc, v71, v71
	v_bfe_u32 v101, v68, 16, 1
	v_add3_u32 v100, v69, v100, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_cndmask_b32_e32 v71, v96, v98, vcc
	v_cmp_o_f32_e32 vcc, v70, v70
	v_add3_u32 v101, v68, v101, s0
	v_lshrrev_b32_e32 v100, 16, v100
	v_cndmask_b32_e32 v70, v96, v99, vcc
	v_cmp_o_f32_e32 vcc, v69, v69
	v_bfe_u32 v98, v67, 16, 1
	v_lshrrev_b32_e32 v101, 16, v101
	v_cndmask_b32_e32 v69, v96, v100, vcc
	v_cmp_o_f32_e32 vcc, v68, v68
	v_bfe_u32 v99, v66, 16, 1
	v_add3_u32 v98, v67, v98, s0
	v_cndmask_b32_e32 v68, v96, v101, vcc
	v_bfe_u32 v100, v65, 16, 1
	v_add3_u32 v99, v66, v99, s0
	v_lshrrev_b32_e32 v98, 16, v98
	v_cmp_o_f32_e32 vcc, v67, v67
	v_bfe_u32 v101, v64, 16, 1
	v_add3_u32 v100, v65, v100, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_cndmask_b32_e32 v67, v96, v98, vcc
	v_cmp_o_f32_e32 vcc, v66, v66
	v_add3_u32 v101, v64, v101, s0
	v_lshrrev_b32_e32 v100, 16, v100
	v_cndmask_b32_e32 v66, v96, v99, vcc
	v_cmp_o_f32_e32 vcc, v65, v65
	v_bfe_u32 v98, v63, 16, 1
	v_lshrrev_b32_e32 v101, 16, v101
	v_cndmask_b32_e32 v65, v96, v100, vcc
	v_cmp_o_f32_e32 vcc, v64, v64
	v_bfe_u32 v99, v62, 16, 1
	v_add3_u32 v98, v63, v98, s0
	v_cndmask_b32_e32 v64, v96, v101, vcc
	v_bfe_u32 v100, v61, 16, 1
	v_add3_u32 v99, v62, v99, s0
	v_lshrrev_b32_e32 v98, 16, v98
	v_cmp_o_f32_e32 vcc, v63, v63
	v_bfe_u32 v101, v60, 16, 1
	v_add3_u32 v100, v61, v100, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_cndmask_b32_e32 v63, v96, v98, vcc
	v_cmp_o_f32_e32 vcc, v62, v62
	v_add3_u32 v101, v60, v101, s0
	v_lshrrev_b32_e32 v100, 16, v100
	v_cndmask_b32_e32 v62, v96, v99, vcc
	v_cmp_o_f32_e32 vcc, v61, v61
	v_bfe_u32 v98, v59, 16, 1
	v_lshrrev_b32_e32 v101, 16, v101
	v_cndmask_b32_e32 v61, v96, v100, vcc
	v_cmp_o_f32_e32 vcc, v60, v60
	v_bfe_u32 v99, v58, 16, 1
	v_add3_u32 v98, v59, v98, s0
	v_cndmask_b32_e32 v60, v96, v101, vcc
	v_bfe_u32 v100, v57, 16, 1
	v_add3_u32 v99, v58, v99, s0
	v_lshrrev_b32_e32 v98, 16, v98
	v_cmp_o_f32_e32 vcc, v59, v59
	v_bfe_u32 v101, v56, 16, 1
	v_add3_u32 v100, v57, v100, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_cndmask_b32_e32 v59, v96, v98, vcc
	v_cmp_o_f32_e32 vcc, v58, v58
	v_add3_u32 v101, v56, v101, s0
	v_lshrrev_b32_e32 v100, 16, v100
	v_cndmask_b32_e32 v58, v96, v99, vcc
	v_cmp_o_f32_e32 vcc, v57, v57
	v_bfe_u32 v98, v55, 16, 1
	v_lshrrev_b32_e32 v101, 16, v101
	v_cndmask_b32_e32 v57, v96, v100, vcc
	v_cmp_o_f32_e32 vcc, v56, v56
	v_bfe_u32 v99, v54, 16, 1
	v_add3_u32 v98, v55, v98, s0
	v_cndmask_b32_e32 v56, v96, v101, vcc
	v_bfe_u32 v100, v53, 16, 1
	v_add3_u32 v99, v54, v99, s0
	v_lshrrev_b32_e32 v98, 16, v98
	v_cmp_o_f32_e32 vcc, v55, v55
	v_bfe_u32 v101, v52, 16, 1
	v_add3_u32 v100, v53, v100, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_cndmask_b32_e32 v55, v96, v98, vcc
	v_cmp_o_f32_e32 vcc, v54, v54
	v_add3_u32 v101, v52, v101, s0
	v_lshrrev_b32_e32 v100, 16, v100
	v_cndmask_b32_e32 v54, v96, v99, vcc
	v_cmp_o_f32_e32 vcc, v53, v53
	v_bfe_u32 v98, v51, 16, 1
	v_lshrrev_b32_e32 v101, 16, v101
	v_cndmask_b32_e32 v53, v96, v100, vcc
	v_cmp_o_f32_e32 vcc, v52, v52
	v_bfe_u32 v99, v50, 16, 1
	v_add3_u32 v98, v51, v98, s0
	v_cndmask_b32_e32 v52, v96, v101, vcc
	v_bfe_u32 v100, v49, 16, 1
	v_add3_u32 v99, v50, v99, s0
	v_lshrrev_b32_e32 v98, 16, v98
	v_cmp_o_f32_e32 vcc, v51, v51
	v_bfe_u32 v101, v48, 16, 1
	v_add3_u32 v100, v49, v100, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_cndmask_b32_e32 v51, v96, v98, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_add3_u32 v101, v48, v101, s0
	v_lshrrev_b32_e32 v100, 16, v100
	v_cndmask_b32_e32 v50, v96, v99, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_bfe_u32 v98, v47, 16, 1
	v_lshrrev_b32_e32 v101, 16, v101
	v_cndmask_b32_e32 v49, v96, v100, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_bfe_u32 v99, v46, 16, 1
	v_add3_u32 v98, v47, v98, s0
	v_cndmask_b32_e32 v48, v96, v101, vcc
	v_bfe_u32 v100, v45, 16, 1
	v_add3_u32 v99, v46, v99, s0
	v_lshrrev_b32_e32 v98, 16, v98
	v_cmp_o_f32_e32 vcc, v47, v47
	v_bfe_u32 v101, v44, 16, 1
	v_add3_u32 v100, v45, v100, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_cndmask_b32_e32 v47, v96, v98, vcc
	v_cmp_o_f32_e32 vcc, v46, v46
	v_add3_u32 v101, v44, v101, s0
	v_lshrrev_b32_e32 v100, 16, v100
	v_cndmask_b32_e32 v46, v96, v99, vcc
	v_cmp_o_f32_e32 vcc, v45, v45
	v_bfe_u32 v98, v43, 16, 1
	v_lshrrev_b32_e32 v101, 16, v101
	v_cndmask_b32_e32 v45, v96, v100, vcc
	v_cmp_o_f32_e32 vcc, v44, v44
	v_bfe_u32 v99, v42, 16, 1
	v_add3_u32 v98, v43, v98, s0
	v_cndmask_b32_e32 v44, v96, v101, vcc
	v_bfe_u32 v100, v41, 16, 1
	v_add3_u32 v99, v42, v99, s0
	v_lshrrev_b32_e32 v98, 16, v98
	v_cmp_o_f32_e32 vcc, v43, v43
	v_bfe_u32 v101, v40, 16, 1
	v_add3_u32 v100, v41, v100, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_cndmask_b32_e32 v43, v96, v98, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	v_add3_u32 v101, v40, v101, s0
	v_lshrrev_b32_e32 v100, 16, v100
	v_cndmask_b32_e32 v42, v96, v99, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	v_bfe_u32 v98, v39, 16, 1
	v_lshrrev_b32_e32 v101, 16, v101
	v_cndmask_b32_e32 v41, v96, v100, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_bfe_u32 v99, v38, 16, 1
	v_add3_u32 v98, v39, v98, s0
	v_cndmask_b32_e32 v40, v96, v101, vcc
	v_bfe_u32 v100, v37, 16, 1
	v_add3_u32 v99, v38, v99, s0
	v_lshrrev_b32_e32 v98, 16, v98
	v_cmp_o_f32_e32 vcc, v39, v39
	v_bfe_u32 v101, v36, 16, 1
	v_add3_u32 v100, v37, v100, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_cndmask_b32_e32 v39, v96, v98, vcc
	v_cmp_o_f32_e32 vcc, v38, v38
	v_add3_u32 v101, v36, v101, s0
	v_lshrrev_b32_e32 v100, 16, v100
	v_cndmask_b32_e32 v38, v96, v99, vcc
	v_cmp_o_f32_e32 vcc, v37, v37
	v_bfe_u32 v98, v23, 16, 1
	v_lshrrev_b32_e32 v101, 16, v101
	v_cndmask_b32_e32 v37, v96, v100, vcc
	v_cmp_o_f32_e32 vcc, v36, v36
	v_bfe_u32 v99, v22, 16, 1
	v_add3_u32 v98, v23, v98, s0
	v_cndmask_b32_e32 v36, v96, v101, vcc
	v_bfe_u32 v100, v21, 16, 1
	v_add3_u32 v99, v22, v99, s0
	v_lshrrev_b32_e32 v98, 16, v98
	v_cmp_o_f32_e32 vcc, v23, v23
	v_bfe_u32 v101, v20, 16, 1
	v_add3_u32 v100, v21, v100, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_cndmask_b32_e32 v23, v96, v98, vcc
	v_cmp_o_f32_e32 vcc, v22, v22
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[130:133], v[186:189], v[12:15], v106, v113 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v101, v20, v101, s0
	v_lshrrev_b32_e32 v100, 16, v100
	v_cndmask_b32_e32 v22, v96, v99, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[138:141], v[134:137], v[8:11], v106, v113 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v21, v21
	v_bfe_u32 v98, v19, 16, 1
	v_lshrrev_b32_e32 v101, 16, v101
	v_cndmask_b32_e32 v21, v96, v100, vcc
	v_cmp_o_f32_e32 vcc, v20, v20
	v_bfe_u32 v99, v18, 16, 1
	v_add3_u32 v98, v19, v98, s0
	v_cndmask_b32_e32 v20, v96, v101, vcc
	v_bfe_u32 v100, v17, 16, 1
	v_add3_u32 v99, v18, v99, s0
	v_lshrrev_b32_e32 v98, 16, v98
	v_cmp_o_f32_e32 vcc, v19, v19
	v_bfe_u32 v101, v16, 16, 1
	v_add3_u32 v100, v17, v100, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_cndmask_b32_e32 v19, v96, v98, vcc
	v_cmp_o_f32_e32 vcc, v18, v18
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[130:133], v[126:129], v[8:11], v106, v113 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v101, v16, v101, s0
	v_lshrrev_b32_e32 v100, 16, v100
	v_cndmask_b32_e32 v18, v96, v99, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[138:141], v[122:125], v[4:7], v106, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v17, v17
	v_bfe_u32 v98, v15, 16, 1
	v_lshrrev_b32_e32 v101, 16, v101
	v_cndmask_b32_e32 v17, v96, v100, vcc
	v_cmp_o_f32_e32 vcc, v16, v16
	v_bfe_u32 v99, v14, 16, 1
	v_add3_u32 v98, v15, v98, s0
	v_cndmask_b32_e32 v16, v96, v101, vcc
	v_bfe_u32 v100, v13, 16, 1
	v_add3_u32 v99, v14, v99, s0
	v_lshrrev_b32_e32 v98, 16, v98
	v_cmp_o_f32_e32 vcc, v15, v15
	v_bfe_u32 v101, v12, 16, 1
	v_add3_u32 v100, v13, v100, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_cndmask_b32_e32 v15, v96, v98, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[130:133], v[118:121], v[4:7], v106, v112 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_add3_u32 v101, v12, v101, s0
	v_lshrrev_b32_e32 v100, 16, v100
	v_cndmask_b32_e32 v14, v96, v99, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_bfe_u32 v98, v11, 16, 1
	v_lshrrev_b32_e32 v101, 16, v101
	v_cndmask_b32_e32 v13, v96, v100, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v99, v10, 16, 1
	v_add3_u32 v98, v11, v98, s0
	v_cndmask_b32_e32 v12, v96, v101, vcc
	v_bfe_u32 v100, v9, 16, 1
	v_add3_u32 v99, v10, v99, s0
	v_lshrrev_b32_e32 v98, 16, v98
	v_cmp_o_f32_e32 vcc, v11, v11
	v_bfe_u32 v101, v8, 16, 1
	v_add3_u32 v100, v9, v100, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_cndmask_b32_e32 v11, v96, v98, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_add3_u32 v101, v8, v101, s0
	v_lshrrev_b32_e32 v100, 16, v100
	v_cndmask_b32_e32 v10, v96, v99, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_bfe_u32 v98, v7, 16, 1
	v_lshrrev_b32_e32 v101, 16, v101
	v_cndmask_b32_e32 v9, v96, v100, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v99, v6, 16, 1
	v_add3_u32 v98, v7, v98, s0
	v_cndmask_b32_e32 v8, v96, v101, vcc
	v_bfe_u32 v100, v5, 16, 1
	v_add3_u32 v99, v6, v99, s0
	v_lshrrev_b32_e32 v98, 16, v98
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v101, v4, 16, 1
	v_add3_u32 v100, v5, v100, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_cndmask_b32_e32 v7, v96, v98, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v101, v4, v101, s0
	v_lshrrev_b32_e32 v100, 16, v100
	v_cndmask_b32_e32 v6, v96, v99, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_lshrrev_b32_e32 v101, 16, v101
	v_bfe_u32 v98, v3, 16, 1
	v_cndmask_b32_e32 v5, v96, v100, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v99, v2, 16, 1
	v_bfe_u32 v100, v1, 16, 1
	v_cndmask_b32_e32 v4, v96, v101, vcc
	v_bfe_u32 v101, v0, 16, 1
	v_add3_u32 v101, v0, v101, s0
	v_add3_u32 v100, v1, v100, s0
	v_add3_u32 v99, v2, v99, s0
	v_add3_u32 v98, v3, v98, s0
	s_mul_i32 s0, s4, s7
	s_add_i32 s0, s1, s0
	s_mul_i32 s1, s5, s6
	s_add_i32 s1, s0, s1
	s_mul_i32 s0, s4, s6
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshrrev_b32_e32 v98, 16, v98
	v_cmp_o_f32_e32 vcc, v3, v3
	s_add_u32 s2, s10, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_cndmask_b32_e32 v3, v96, v98, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	s_addc_u32 s3, s11, s1
	s_lshl_b64 s[0:1], s[14:15], 1
	v_lshrrev_b32_e32 v100, 16, v100
	v_cndmask_b32_e32 v2, v96, v99, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	s_add_u32 s0, s2, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cndmask_b32_e32 v1, v96, v100, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	s_addc_u32 s1, s3, s1
	s_and_b32 s2, s4, 0x3fff
	v_cndmask_b32_e32 v0, v96, v101, vcc
	v_lshl_or_b32 v96, v102, 2, v103
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	v_lshlrev_b32_e32 v98, 1, v104
	v_mul_lo_u32 v96, s4, v96
	s_or_b32 s1, s2, s1
	v_lshl_add_u32 v97, v97, 1, v98
	s_or_b32 s1, s1, 2.0
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, 0x7ffffffd
	v_lshl_add_u32 v98, v96, 1, v97
	s_lshl_b32 s5, s4, 1
	buffer_store_short v86, v98, s[0:3], 0 offen
	v_add_u32_e32 v86, s5, v98
	buffer_store_short v85, v86, s[0:3], 0 offen
	v_add_u32_e32 v85, s5, v86
	buffer_store_short v84, v85, s[0:3], 0 offen
	v_add_u32_e32 v84, s5, v85
	s_lshl_b32 s4, s4, 4
	buffer_store_short v88, v84, s[0:3], 0 offen
	buffer_store_short v79, v98, s[0:3], 0 offen offset:32
	buffer_store_short v78, v86, s[0:3], 0 offen offset:32
	buffer_store_short v77, v85, s[0:3], 0 offen offset:32
	buffer_store_short v76, v84, s[0:3], 0 offen offset:32
	buffer_store_short v83, v98, s[0:3], 0 offen offset:64
	buffer_store_short v82, v86, s[0:3], 0 offen offset:64
	buffer_store_short v81, v85, s[0:3], 0 offen offset:64
	buffer_store_short v80, v84, s[0:3], 0 offen offset:64
	buffer_store_short v91, v98, s[0:3], 0 offen offset:96
	buffer_store_short v90, v86, s[0:3], 0 offen offset:96
	buffer_store_short v89, v85, s[0:3], 0 offen offset:96
	buffer_store_short v87, v84, s[0:3], 0 offen offset:96
	buffer_store_short v25, v98, s[0:3], 0 offen offset:128
	buffer_store_short v24, v86, s[0:3], 0 offen offset:128
	buffer_store_short v93, v85, s[0:3], 0 offen offset:128
	buffer_store_short v92, v84, s[0:3], 0 offen offset:128
	buffer_store_short v29, v98, s[0:3], 0 offen offset:160
	buffer_store_short v28, v86, s[0:3], 0 offen offset:160
	buffer_store_short v27, v85, s[0:3], 0 offen offset:160
	buffer_store_short v26, v84, s[0:3], 0 offen offset:160
	v_add_u32_e32 v24, s4, v96
	v_lshl_add_u32 v25, v24, 1, v97
	v_add_u32_e32 v26, s5, v25
	v_add_u32_e32 v27, s5, v26
	v_add_u32_e32 v28, s5, v27
	v_add_u32_e32 v24, s4, v24
	buffer_store_short v33, v25, s[0:3], 0 offen
	buffer_store_short v32, v26, s[0:3], 0 offen
	buffer_store_short v31, v27, s[0:3], 0 offen
	buffer_store_short v30, v28, s[0:3], 0 offen
	buffer_store_short v95, v25, s[0:3], 0 offen offset:32
	buffer_store_short v94, v26, s[0:3], 0 offen offset:32
	buffer_store_short v35, v27, s[0:3], 0 offen offset:32
	buffer_store_short v34, v28, s[0:3], 0 offen offset:32
	buffer_store_short v72, v25, s[0:3], 0 offen offset:64
	buffer_store_short v73, v26, s[0:3], 0 offen offset:64
	buffer_store_short v74, v27, s[0:3], 0 offen offset:64
	buffer_store_short v75, v28, s[0:3], 0 offen offset:64
	buffer_store_short v68, v25, s[0:3], 0 offen offset:96
	buffer_store_short v69, v26, s[0:3], 0 offen offset:96
	buffer_store_short v70, v27, s[0:3], 0 offen offset:96
	buffer_store_short v71, v28, s[0:3], 0 offen offset:96
	buffer_store_short v64, v25, s[0:3], 0 offen offset:128
	buffer_store_short v65, v26, s[0:3], 0 offen offset:128
	buffer_store_short v66, v27, s[0:3], 0 offen offset:128
	buffer_store_short v67, v28, s[0:3], 0 offen offset:128
	buffer_store_short v60, v25, s[0:3], 0 offen offset:160
	buffer_store_short v61, v26, s[0:3], 0 offen offset:160
	buffer_store_short v62, v27, s[0:3], 0 offen offset:160
	buffer_store_short v63, v28, s[0:3], 0 offen offset:160
	v_lshl_add_u32 v25, v24, 1, v97
	v_add_u32_e32 v26, s5, v25
	v_add_u32_e32 v27, s5, v26
	v_add_u32_e32 v24, s4, v24
	v_add_u32_e32 v28, s5, v27
	v_lshl_add_u32 v24, v24, 1, v97
	buffer_store_short v56, v25, s[0:3], 0 offen
	buffer_store_short v57, v26, s[0:3], 0 offen
	buffer_store_short v58, v27, s[0:3], 0 offen
	buffer_store_short v59, v28, s[0:3], 0 offen
	buffer_store_short v52, v25, s[0:3], 0 offen offset:32
	buffer_store_short v53, v26, s[0:3], 0 offen offset:32
	buffer_store_short v54, v27, s[0:3], 0 offen offset:32
	buffer_store_short v55, v28, s[0:3], 0 offen offset:32
	buffer_store_short v48, v25, s[0:3], 0 offen offset:64
	buffer_store_short v49, v26, s[0:3], 0 offen offset:64
	buffer_store_short v50, v27, s[0:3], 0 offen offset:64
	buffer_store_short v51, v28, s[0:3], 0 offen offset:64
	buffer_store_short v44, v25, s[0:3], 0 offen offset:96
	buffer_store_short v45, v26, s[0:3], 0 offen offset:96
	buffer_store_short v46, v27, s[0:3], 0 offen offset:96
	buffer_store_short v47, v28, s[0:3], 0 offen offset:96
	buffer_store_short v40, v25, s[0:3], 0 offen offset:128
	buffer_store_short v41, v26, s[0:3], 0 offen offset:128
	buffer_store_short v42, v27, s[0:3], 0 offen offset:128
	buffer_store_short v43, v28, s[0:3], 0 offen offset:128
	buffer_store_short v36, v25, s[0:3], 0 offen offset:160
	buffer_store_short v37, v26, s[0:3], 0 offen offset:160
	buffer_store_short v38, v27, s[0:3], 0 offen offset:160
	buffer_store_short v39, v28, s[0:3], 0 offen offset:160
	buffer_store_short v20, v24, s[0:3], 0 offen
	v_add_u32_e32 v20, s5, v24
	buffer_store_short v21, v20, s[0:3], 0 offen
	v_add_u32_e32 v21, s5, v20
	buffer_store_short v22, v21, s[0:3], 0 offen
	v_add_u32_e32 v22, s5, v21
	buffer_store_short v23, v22, s[0:3], 0 offen
	buffer_store_short v16, v24, s[0:3], 0 offen offset:32
	buffer_store_short v17, v20, s[0:3], 0 offen offset:32
	buffer_store_short v18, v21, s[0:3], 0 offen offset:32
	buffer_store_short v19, v22, s[0:3], 0 offen offset:32
	buffer_store_short v12, v24, s[0:3], 0 offen offset:64
	buffer_store_short v13, v20, s[0:3], 0 offen offset:64
	buffer_store_short v14, v21, s[0:3], 0 offen offset:64
	buffer_store_short v15, v22, s[0:3], 0 offen offset:64
	buffer_store_short v8, v24, s[0:3], 0 offen offset:96
	buffer_store_short v9, v20, s[0:3], 0 offen offset:96
	buffer_store_short v10, v21, s[0:3], 0 offen offset:96
	buffer_store_short v11, v22, s[0:3], 0 offen offset:96
	buffer_store_short v4, v24, s[0:3], 0 offen offset:128
	buffer_store_short v5, v20, s[0:3], 0 offen offset:128
	buffer_store_short v6, v21, s[0:3], 0 offen offset:128
	buffer_store_short v7, v22, s[0:3], 0 offen offset:128
	buffer_store_short v0, v24, s[0:3], 0 offen offset:160
	buffer_store_short v1, v20, s[0:3], 0 offen offset:160
	buffer_store_short v2, v21, s[0:3], 0 offen offset:160
	buffer_store_short v3, v22, s[0:3], 0 offen offset:160
	s_endpgm
.LBB0_15:
	s_branch .LBB0_2
.LBB0_16:
	s_load_dwordx2 s[44:45], s[0:1], 0x48
	s_load_dwordx2 s[42:43], s[0:1], 0x58
	s_branch .LBB0_5
.LBB0_17:
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v1, v0
	v_mov_b64_e32 v[6:7], v[2:3]
	v_mov_b64_e32 v[10:11], v[2:3]
	v_mov_b64_e32 v[34:35], v[2:3]
	v_mov_b64_e32 v[30:31], v[2:3]
	v_mov_b64_e32 v[26:27], v[2:3]
	v_mov_b64_e32 v[18:19], v[2:3]
	v_mov_b64_e32 v[14:15], v[2:3]
	v_mov_b64_e32 v[22:23], v[2:3]
	v_mov_b64_e32 v[38:39], v[2:3]
	v_mov_b64_e32 v[42:43], v[2:3]
	v_mov_b64_e32 v[46:47], v[2:3]
	v_mov_b64_e32 v[50:51], v[2:3]
	v_mov_b64_e32 v[54:55], v[2:3]
	v_mov_b64_e32 v[62:63], v[2:3]
	v_mov_b64_e32 v[66:67], v[2:3]
	v_mov_b64_e32 v[70:71], v[2:3]
	v_mov_b64_e32 v[74:75], v[2:3]
	v_mov_b64_e32 v[78:79], v[2:3]
	v_mov_b64_e32 v[82:83], v[2:3]
	v_mov_b64_e32 v[86:87], v[2:3]
	v_mov_b64_e32 v[90:91], v[2:3]
	v_mov_b64_e32 v[94:95], v[2:3]
	v_mov_b64_e32 v[58:59], v[2:3]
	v_mov_b64_e32 v[4:5], v[0:1]
	v_mov_b64_e32 v[8:9], v[0:1]
	v_mov_b64_e32 v[32:33], v[0:1]
	v_mov_b64_e32 v[28:29], v[0:1]
	v_mov_b64_e32 v[24:25], v[0:1]
	v_mov_b64_e32 v[16:17], v[0:1]
	v_mov_b64_e32 v[12:13], v[0:1]
	v_mov_b64_e32 v[20:21], v[0:1]
	v_mov_b64_e32 v[36:37], v[0:1]
	v_mov_b64_e32 v[40:41], v[0:1]
	v_mov_b64_e32 v[44:45], v[0:1]
	v_mov_b64_e32 v[48:49], v[0:1]
	v_mov_b64_e32 v[52:53], v[0:1]
	v_mov_b64_e32 v[60:61], v[0:1]
	v_mov_b64_e32 v[64:65], v[0:1]
	v_mov_b64_e32 v[68:69], v[0:1]
	v_mov_b64_e32 v[72:73], v[0:1]
	v_mov_b64_e32 v[76:77], v[0:1]
	v_mov_b64_e32 v[80:81], v[0:1]
	v_mov_b64_e32 v[84:85], v[0:1]
	v_mov_b64_e32 v[88:89], v[0:1]
	v_mov_b64_e32 v[92:93], v[0:1]
	v_mov_b64_e32 v[56:57], v[0:1]
	s_load_dwordx2 s[4:5], s[0:1], 0x60
	v_cmp_lt_i64_e32 vcc, 0, v[100:101]
	s_cbranch_vccnz .LBB0_12
.LBB0_18:
	s_mov_b32 s9, 0x10000
	s_mov_b32 s8, 0
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_14
	s_branch .LBB0_13
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_dynamic_gemm_256x192x256
		.amdhsa_group_segment_fixed_size 114688
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
		.amdhsa_next_free_vgpr 206
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 208
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
	.size	wave_mxfp4_dynamic_gemm_256x192x256, .Lfunc_end0-wave_mxfp4_dynamic_gemm_256x192x256

	.set wave_mxfp4_dynamic_gemm_256x192x256.num_vgpr, 206
	.set wave_mxfp4_dynamic_gemm_256x192x256.num_agpr, 0
	.set wave_mxfp4_dynamic_gemm_256x192x256.numbered_sgpr, 55
	.set wave_mxfp4_dynamic_gemm_256x192x256.num_named_barrier, 0
	.set wave_mxfp4_dynamic_gemm_256x192x256.private_seg_size, 0
	.set wave_mxfp4_dynamic_gemm_256x192x256.uses_vcc, 1
	.set wave_mxfp4_dynamic_gemm_256x192x256.uses_flat_scratch, 0
	.set wave_mxfp4_dynamic_gemm_256x192x256.has_dyn_sized_stack, 0
	.set wave_mxfp4_dynamic_gemm_256x192x256.has_recursion, 0
	.set wave_mxfp4_dynamic_gemm_256x192x256.has_indirect_call, 0
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
    .group_segment_fixed_size: 114688
    .kernarg_segment_align: 8
    .kernarg_segment_size: 104
    .max_flat_workgroup_size: 512
    .name:           wave_mxfp4_dynamic_gemm_256x192x256
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     61
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_dynamic_gemm_256x192x256.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     206
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
