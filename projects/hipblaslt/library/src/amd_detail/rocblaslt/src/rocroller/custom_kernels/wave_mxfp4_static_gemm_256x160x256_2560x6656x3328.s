; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_256x160x256_2560x6656x3328
	.p2align	8
	.type	wave_mxfp4_static_gemm_256x160x256_2560x6656x3328,@function
wave_mxfp4_static_gemm_256x160x256_2560x6656x3328:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v85, 0x3ff, v0
	v_bfe_u32 v3, v0, 10, 10
	v_lshrrev_b32_e32 v5, 6, v85
	v_lshlrev_b32_e32 v0, 5, v3
	v_lshl_or_b32 v1, v5, 3, v0
	v_bfe_u32 v9, v85, 2, 3
	s_mov_b64 s[24:25], s[2:3]
	v_readfirstlane_b32 s2, v1
	v_lshrrev_b32_e32 v1, 3, v85
	s_lshl_b32 s16, s16, 8
	v_lshrrev_b32_e32 v7, 5, v85
	v_lshrrev_b32_e32 v8, 2, v85
	v_and_b32_e32 v10, 31, v85
	v_lshlrev_b32_e32 v9, 2, v9
	v_or3_b32 v2, v1, v0, s16
	v_xor_b32_e32 v1, v1, v85
	v_bitop3_b32 v13, v8, v7, 7 bitop3:0x6c
	v_sub_u32_e32 v9, v10, v9
	v_lshlrev_b32_e32 v1, 4, v1
	v_lshl_add_u32 v9, v13, 2, v9
	v_and_b32_e32 v1, 0x70, v1
	v_mul_u32_u24_e32 v2, 0x680, v2
	s_and_b32 s3, s25, 0xffff
	s_lshl_b32 s35, s2, 7
	v_ashrrev_i32_e32 v10, 31, v9
	s_or_b32 s25, s3, 0x46800000
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_or_b32_e32 v4, v2, v1
	s_mov_b32 m0, s35
	s_or_b32 s40, s35, 0x2000
	v_xor_b32_e32 v9, v10, v9
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_add_u32_e32 v6, 0x1a000, v4
	s_mov_b32 m0, s40
	s_or_b32 s41, s35, 0x4000
	v_ashrrev_i32_e32 v11, 31, v9
	buffer_load_dwordx4 v6, s[24:27], 0 offen lds
	v_add_u32_e32 v6, 0x34000, v4
	s_mov_b32 m0, s41
	s_or_b32 s42, s35, 0x6000
	v_lshrrev_b32_e32 v11, 29, v11
	buffer_load_dwordx4 v6, s[24:27], 0 offen lds
	v_add_u32_e32 v4, 0x4e000, v4
	s_mov_b32 m0, s42
	v_add_u32_e32 v9, v9, v11
	buffer_load_dwordx4 v4, s[24:27], 0 offen lds
	v_lshlrev_b32_e32 v4, 3, v3
	v_ashrrev_i32_e32 v9, 3, v9
	v_lshrrev_b32_e32 v16, 1, v13
	v_lshl_or_b32 v4, v5, 1, v4
	v_xor_b32_e32 v15, v9, v10
	v_and_b32_e32 v10, 0xfc, v85
	v_lshlrev_b32_e32 v11, 7, v16
	v_readfirstlane_b32 s2, v4
	v_lshlrev_b32_e32 v4, 6, v13
	v_lshlrev_b32_e32 v6, 2, v7
	v_add_u32_e32 v12, v10, v11
	v_add3_u32 v14, v0, v85, v4
	v_lshlrev_b32_e32 v9, 7, v15
	v_sub_u32_e32 v17, v6, v12
	v_add3_u32 v14, v17, v14, v9
	v_ashrrev_i32_e32 v17, 31, v14
	v_xor_b32_e32 v14, v17, v14
	s_mov_b32 s43, 0x4ec4ec4f
	v_mul_hi_i32 v14, v14, s43
	v_lshrrev_b32_e32 v18, 31, v14
	v_ashrrev_i32_e32 v14, 7, v14
	v_add_u32_e32 v14, v14, v18
	v_xor_b32_e32 v17, v14, v17
	v_sub_u32_e32 v14, v7, v8
	v_lshlrev_b32_e32 v19, 4, v14
	v_lshlrev_b32_e32 v86, 2, v85
	v_lshlrev_b32_e32 v14, 9, v15
	v_lshlrev_b32_e32 v15, 8, v13
	s_mul_i32 s33, s17, 0xa0
	v_add3_u32 v19, v19, v86, v15
	v_lshlrev_b32_e32 v13, 9, v16
	s_mov_b64 s[20:21], s[6:7]
	s_movk_i32 s34, 0x680
	v_add_u32_e32 v18, s33, v17
	v_sub_u32_e32 v19, v19, v13
	v_lshlrev_b32_e32 v16, 7, v3
	v_mul_i32_i24_e32 v17, 0xfffff980, v17
	v_add3_u32 v19, v19, v14, v16
	v_mul_lo_u32 v18, v18, s34
	s_and_b32 s3, s21, 0xffff
	s_lshl_b32 s44, s2, 7
	s_or_b32 s21, s3, 0x46800000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_add3_u32 v17, v19, v17, v18
	s_add_i32 m0, s44, 0x10000
	s_or_b32 s45, s44, 0x800
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x6800, v17
	s_add_i32 m0, s45, 0x10000
	s_or_b32 s46, s44, 0x1000
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0xd000, v17
	s_add_i32 m0, s46, 0x10000
	s_or_b32 s47, s44, 0x1800
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x13800, v17
	s_add_i32 m0, s47, 0x10000
	s_or_b32 s48, s44, 0x2000
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x1a000, v17
	s_add_i32 m0, s48, 0x10000
	s_or_b32 s49, s44, 0x2800
	buffer_load_dword v18, s[20:23], 0 offen lds
	s_add_i32 m0, s49, 0x10000
	v_add_u32_e32 v18, 0x20800, v17
	s_cmp_lt_u32 s17, 41
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x27000, v17
	v_bfrev_b32_e32 v87, -2
	s_cselect_b64 s[6:7], -1, 0
	s_or_b32 s50, s44, 0x3000
	v_cndmask_b32_e64 v18, v87, v18, s[6:7]
	s_add_i32 m0, s50, 0x10000
	s_or_b32 s51, s44, 0x3800
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x2d800, v17
	v_cndmask_b32_e64 v18, v87, v18, s[6:7]
	s_add_i32 m0, s51, 0x10000
	s_or_b32 s52, s44, 0x4000
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x34000, v17
	v_cndmask_b32_e64 v18, v87, v18, s[6:7]
	s_add_i32 m0, s52, 0x10000
	v_add_u32_e32 v17, 0x3a800, v17
	s_or_b32 s53, s44, 0x4800
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_cndmask_b32_e64 v17, v87, v17, s[6:7]
	s_add_i32 m0, s53, 0x10000
	v_lshrrev_b32_e32 v20, 4, v85
	v_bfe_u32 v83, v85, 4, 2
	s_mul_i32 s15, s15, s16
	s_mul_hi_u32 s2, s14, s16
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v17, 4, v83
	v_mad_i32_i24 v21, v20, -16, v85
	s_add_i32 s2, s2, s15
	s_mul_i32 s3, s14, s16
	v_add_u32_e32 v18, v21, v17
	s_add_u32 s28, s4, s3
	v_ashrrev_i32_e32 v19, 31, v18
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s14, 0x3fff
	v_xor_b32_e32 v18, v19, v18
	s_bitset1_b32 s3, 14
	v_mul_hi_i32 v18, v18, s43
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	v_lshrrev_b32_e32 v22, 31, v18
	v_ashrrev_i32_e32 v18, 3, v18
	s_or_b32 s29, s2, s3
	s_movk_i32 s2, 0x50
	v_add_u32_e32 v18, v18, v22
	v_and_b32_e32 v84, 0xc0, v85
	v_mad_u32_u24 v80, v3, s2, v21
	v_xad_u32 v22, v18, v19, v84
	v_and_b32_e32 v18, 62, v85
	v_ashrrev_i16_e32 v21, 15, v80
	v_mul_lo_u16_e32 v18, 10, v18
	s_movk_i32 s12, 0xff98
	v_lshrrev_b16_e32 v21, 11, v21
	v_mul_i32_i24_sdwa v23, v18, s12 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:DWORD
	v_lshlrev_b32_e32 v18, 6, v83
	v_mul_i32_i24_e32 v19, 0xffffffc0, v20
	v_add_u16_e32 v21, v80, v21
	v_add3_u32 v24, v19, v86, v18
	v_mul_lo_u32 v22, v22, s14
	v_and_b32_e32 v21, 0xffffffe0, v21
	s_mov_b32 s30, s26
	s_mov_b32 s31, s27
	v_add3_u32 v22, v24, v23, v22
	v_sub_u16_e32 v21, v80, v21
	v_lshl_add_u32 v23, s14, 5, v22
	buffer_load_dword v91, v22, s[28:31], 0 offen
	buffer_load_dword v90, v23, s[28:31], 0 offen
	v_bfe_i32 v22, v21, 0, 16
	v_ashrrev_i32_e32 v23, 31, v22
	v_add_u16_e32 v25, 32, v21
	v_cmp_gt_i16_e32 vcc, 0, v21
	s_load_dwordx2 s[18:19], s[0:1], 0x40
	s_movk_i32 s5, 0x68
	v_cndmask_b32_e32 v21, v22, v25, vcc
	v_cndmask_b32_e64 v22, v23, 0, vcc
	v_xor_b32_e32 v21, v22, v21
	v_lshrrev_b32_e32 v23, 28, v21
	v_add_u32_e32 v21, v21, v23
	v_ashrrev_i32_e32 v21, 4, v21
	v_xor_b32_e32 v21, v21, v22
	v_add_u32_e32 v23, v24, v21
	v_ashrrev_i32_e32 v22, 31, v23
	v_xor_b32_e32 v24, v22, v23
	v_mul_hi_i32 v24, v24, s43
	v_lshrrev_b32_e32 v25, 31, v24
	v_ashrrev_i32_e32 v24, 5, v24
	v_add_u32_e32 v24, v24, v25
	v_ashrrev_i32_e32 v25, 31, v80
	v_xor_b32_e32 v26, v25, v80
	v_ashrrev_i32_e32 v27, 31, v26
	v_lshrrev_b32_e32 v27, 27, v27
	v_add_u32_e32 v26, v26, v27
	v_lshrrev_b32_e32 v26, 5, v26
	v_xor_b32_e32 v25, v26, v25
	v_lshlrev_b32_e32 v88, 5, v25
	v_xad_u32 v24, v24, v22, v88
	v_mul_hi_i32 v22, v23, s43
	v_lshrrev_b32_e32 v25, 31, v22
	v_ashrrev_i32_e32 v22, 5, v22
	v_add_u32_e32 v22, v22, v25
	v_mul_lo_u32 v22, v22, s5
	v_sub_u32_e32 v22, v23, v22
	v_add_u32_e32 v25, 0x68, v22
	v_cmp_gt_i32_e32 vcc, 0, v22
	v_add_u32_e32 v27, s33, v24
	s_movk_i32 s4, 0x1a00
	v_cndmask_b32_e32 v22, v22, v25, vcc
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s19, s33
	s_mul_hi_u32 s3, s18, s33
	s_add_i32 s13, s3, s2
	v_mad_u64_u32 v[24:25], s[2:3], s18, v24, v[22:23]
	v_cmp_gt_i32_e32 vcc, s4, v27
	v_add_u32_e32 v22, 2, v23
	v_sub_u32_e32 v25, -3, v23
	v_cndmask_b32_e32 v32, v87, v24, vcc
	v_cmp_gt_i32_e32 vcc, -2, v23
	v_add_u32_e32 v81, 16, v80
	s_movk_i32 s54, 0xffc0
	v_cndmask_b32_e32 v23, v22, v25, vcc
	v_mul_hi_i32 v23, v23, s43
	v_lshrrev_b32_e32 v25, 31, v23
	v_ashrrev_i32_e32 v23, 5, v23
	v_add_u32_e32 v23, v23, v25
	v_cndmask_b32_e64 v25, 0, -1, vcc
	v_xor_b32_e32 v23, v23, v25
	v_add_u32_e32 v25, v23, v88
	v_add_u32_e32 v33, s33, v25
	v_mul_lo_u32 v23, v23, s12
	v_mul_lo_u32 v25, s18, v25
	v_add3_u32 v25, v22, v23, v25
	v_cmp_gt_i32_e32 vcc, s4, v33
	v_sub_u32_e32 v22, 0xffef, v80
	v_mad_u32_u24 v26, v3, 5, v86
	v_cndmask_b32_e32 v34, v87, v25, vcc
	v_cmp_gt_i32_e32 vcc, -16, v80
	s_mul_i32 s15, s18, s33
	s_add_u32 s36, s8, s15
	v_cndmask_b32_e32 v22, v81, v22, vcc
	v_ashrrev_i16_e32 v23, 15, v22
	v_lshrrev_b16_e32 v23, 11, v23
	v_add_u16_e32 v22, v22, v23
	v_ashrrev_i16_e32 v22, 5, v22
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_xor_b32_e32 v28, v22, v23
	v_mov_b32_e32 v22, -2
	v_mul_i32_i24_sdwa v23, sext(v28), v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_mad_i32_i24 v22, v20, s54, v18
	v_add3_u32 v35, v22, v26, v23
	v_add_u32_e32 v26, 1, v35
	v_sub_u32_e32 v29, -2, v35
	v_cmp_gt_i32_e32 vcc, -1, v35
	s_addc_u32 s2, s9, s13
	s_and_b32 s3, s18, 0x3fff
	v_cndmask_b32_e32 v29, v26, v29, vcc
	v_mul_hi_i32 v29, v29, s43
	v_lshrrev_b32_e32 v30, 31, v29
	v_ashrrev_i32_e32 v29, 5, v29
	v_add_u32_e32 v29, v29, v30
	v_cndmask_b32_e64 v30, 0, -1, vcc
	v_mov_b32_e32 v37, 5
	s_bitset1_b32 s3, 14
	v_xor_b32_e32 v36, v29, v30
	v_lshlrev_b32_sdwa v89, v37, sext(v28) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	v_add_u32_e32 v30, v36, v89
	s_or_b32 s37, s2, s3
	v_add_u32_e32 v38, s33, v30
	v_mad_u64_u32 v[28:29], s[2:3], v36, s12, v[26:27]
	v_mad_u64_u32 v[30:31], s[2:3], v30, s18, v[28:29]
	v_cmp_gt_i32_e32 vcc, s4, v38
	v_sub_u32_e32 v31, -4, v35
	s_movk_i32 s8, 0x19e0
	v_cndmask_b32_e32 v29, v87, v30, vcc
	v_add_u32_e32 v30, 3, v35
	v_cmp_gt_i32_e32 vcc, -3, v35
	s_movk_i32 s9, 0xffd0
	v_add_u32_e32 v82, 48, v80
	v_cndmask_b32_e32 v30, v30, v31, vcc
	v_mul_hi_i32 v30, v30, s43
	v_lshrrev_b32_e32 v31, 31, v30
	v_ashrrev_i32_e32 v30, 5, v30
	v_add_u32_e32 v30, v30, v31
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_xor_b32_e32 v35, v30, v31
	v_add_u32_e32 v30, v35, v89
	v_mul_lo_u32 v31, v35, s12
	v_add_u32_e32 v38, s33, v30
	v_add3_u32 v26, v26, v31, 2
	v_mad_u64_u32 v[30:31], s[2:3], v30, s18, v[26:27]
	v_cmp_gt_i32_e32 vcc, s4, v38
	s_lshl_b32 s12, s18, 5
	v_add_u32_e32 v31, s12, v24
	v_cndmask_b32_e32 v30, v87, v30, vcc
	v_cmp_gt_i32_e32 vcc, s8, v27
	v_add_u32_e32 v39, s12, v25
	v_sub_u32_e32 v24, 0xffcf, v80
	v_cndmask_b32_e32 v38, v87, v31, vcc
	v_cmp_gt_i32_e32 vcc, s8, v33
	s_mov_b32 s38, s26
	s_mov_b32 s39, s27
	v_cndmask_b32_e32 v40, v87, v39, vcc
	v_cmp_gt_i32_e32 vcc, s9, v80
	s_movk_i32 s55, 0x19c0
	s_mov_b32 s15, 0
	v_cndmask_b32_e32 v24, v82, v24, vcc
	v_ashrrev_i16_e32 v25, 15, v24
	v_lshrrev_b16_e32 v25, 11, v25
	v_add_u16_e32 v24, v24, v25
	v_ashrrev_i16_e32 v24, 5, v24
	v_cndmask_b32_e64 v25, 0, -1, vcc
	v_xor_b32_e32 v24, v24, v25
	v_lshlrev_b32_sdwa v92, v37, sext(v24) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_e32 v24, v36, v92
	v_add_u32_e32 v36, s33, v24
	v_mad_u64_u32 v[24:25], s[2:3], v24, s18, v[28:29]
	v_cmp_gt_i32_e32 vcc, s4, v36
	s_movk_i32 s56, 0xf980
	s_mov_b32 s19, 0x10000
	v_cndmask_b32_e32 v28, v87, v24, vcc
	v_add_u32_e32 v24, v35, v92
	v_add_u32_e32 v35, s33, v24
	v_mad_u64_u32 v[24:25], s[2:3], v24, s18, v[26:27]
	v_cmp_gt_i32_e32 vcc, s4, v35
	v_add_u32_e32 v25, s12, v39
	v_cmp_eq_u32_e64 s[2:3], 0, v3
	v_cndmask_b32_e32 v24, v87, v24, vcc
	buffer_load_ubyte v124, v32, s[36:39], 0 offen
	buffer_load_ubyte v119, v34, s[36:39], 0 offen
	buffer_load_ubyte v123, v29, s[36:39], 0 offen
	buffer_load_ubyte v118, v30, s[36:39], 0 offen
	buffer_load_ubyte v122, v38, s[36:39], 0 offen
	buffer_load_ubyte v117, v40, s[36:39], 0 offen
	buffer_load_ubyte v121, v28, s[36:39], 0 offen
	buffer_load_ubyte v116, v24, s[36:39], 0 offen
	v_add_u32_e32 v24, s12, v31
	v_cmp_gt_i32_e32 vcc, s55, v27
	s_movk_i32 s22, 0x2800
	s_nop 0
	v_cndmask_b32_e32 v24, v87, v24, vcc
	v_cmp_gt_i32_e32 vcc, s55, v33
	s_nop 1
	v_cndmask_b32_e32 v25, v87, v25, vcc
	buffer_load_ubyte v125, v24, s[36:39], 0 offen
	buffer_load_ubyte v120, v25, s[36:39], 0 offen
	v_mul_i32_i24_e32 v24, -16, v20
	v_mul_u32_u24_e32 v25, 5, v3
	s_and_b64 vcc, exec, s[2:3]
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	v_lshlrev_b32_e32 v28, 7, v85
	v_lshlrev_b32_e32 v20, 11, v20
	v_and_b32_e32 v26, 7, v85
	v_sub_u32_e32 v20, v28, v20
	v_mul_lo_u32 v3, v3, s22
	v_bitop3_b32 v27, v83, v85, 7 bitop3:0x78
	v_lshl_add_u32 v5, v5, 13, v20
	v_add_u32_e32 v3, v20, v3
	v_bitop3_b32 v20, v83, v26, 4 bitop3:0x36
	v_lshlrev_b32_e32 v27, 4, v27
	v_lshlrev_b32_e32 v20, 4, v20
	v_or_b32_e32 v95, v3, v27
	v_or_b32_e32 v96, v20, v3
	v_add_u32_e32 v3, s33, v89
	v_sub_u32_e32 v98, 0x1a00, v3
	v_add_u32_e32 v3, v23, v19
	v_add3_u32 v100, v3, v18, v25
	v_add_u32_e32 v3, v3, v86
	v_add3_u32 v3, v3, v25, v18
	v_or_b32_e32 v93, v5, v27
	v_or_b32_e32 v94, v20, v5
	v_add_u32_e32 v5, s33, v92
	v_sub_u32_e32 v101, 0, v3
	v_sub_u32_e32 v3, 0, v19
	v_sub_u32_e32 v99, 0x1a00, v5
	v_sub_u32_e32 v5, v3, v23
	v_sub_u32_e32 v3, v3, v21
	v_sub_u32_e32 v3, v3, v86
	s_mul_i32 s17, s17, 0x41000
	v_sub_u32_e32 v5, v5, v86
	v_sub_u32_e32 v103, v3, v18
	v_add_u32_e32 v3, s17, v14
	v_sub_u32_e32 v5, v5, v25
	v_add3_u32 v3, v3, v15, v16
	v_sub_u32_e32 v102, v5, v18
	v_lshl_add_u32 v3, v7, 4, v3
	v_lshlrev_b32_e32 v5, 4, v8
	v_sub_u32_e32 v3, v3, v5
	v_sub_u32_e32 v112, v3, v13
	v_sub_u32_e32 v3, v12, v6
	v_sub_u32_e32 v3, v3, v0
	s_load_dwordx2 s[12:13], s[0:1], 0x48
	v_sub_u32_e32 v3, v3, v4
	v_sub_u32_e32 v113, v3, v9
	v_add_u32_e32 v3, v9, v4
	v_add3_u32 v0, v3, v0, v6
	v_add_u32_e32 v110, v24, v17
	v_sub_u32_e32 v0, v0, v10
	s_mov_b32 s0, 0x4e080
	v_mov_b32_e32 v12, 0
	v_add_u32_e32 v97, s33, v88
	v_add3_u32 v104, v19, v21, v18
	v_add_u32_e32 v105, 32, v88
	v_add_u32_e32 v106, 64, v88
	v_add_u32_e32 v107, 32, v84
	v_add_u32_e32 v108, 48, v22
	v_sub_u32_e32 v109, 0, v85
	v_sub_u32_e32 v111, 0, v110
	v_sub_u32_e32 v114, v0, v11
	v_add3_u32 v115, v2, v1, s0
	s_mov_b32 s61, 0x8000
	s_mov_b32 s62, 0x15000
	s_mov_b32 s0, 0
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	s_mov_b32 s30, s26
	s_mov_b32 s31, s27
	s_movk_i32 s1, 0xff00
	s_movk_i32 s17, 0xfeff
	s_mov_b32 s38, s26
	s_mov_b32 s39, s27
	s_movk_i32 s57, 0xfefd
	s_movk_i32 s58, 0xfefe
	s_movk_i32 s59, 0xffcf
	v_mov_b32_e32 v13, v12
	v_mov_b32_e32 v14, v12
	v_mov_b32_e32 v15, v12
	v_mov_b32_e32 v76, v12
	v_mov_b32_e32 v77, v12
	v_mov_b32_e32 v78, v12
	v_mov_b32_e32 v79, v12
	v_mov_b32_e32 v72, v12
	v_mov_b32_e32 v73, v12
	v_mov_b32_e32 v74, v12
	v_mov_b32_e32 v75, v12
	v_mov_b32_e32 v68, v12
	v_mov_b32_e32 v69, v12
	v_mov_b32_e32 v70, v12
	v_mov_b32_e32 v71, v12
	v_mov_b32_e32 v64, v12
	v_mov_b32_e32 v65, v12
	v_mov_b32_e32 v66, v12
	v_mov_b32_e32 v67, v12
	v_mov_b32_e32 v60, v12
	v_mov_b32_e32 v61, v12
	v_mov_b32_e32 v62, v12
	v_mov_b32_e32 v63, v12
	v_mov_b32_e32 v56, v12
	v_mov_b32_e32 v57, v12
	v_mov_b32_e32 v58, v12
	v_mov_b32_e32 v59, v12
	v_mov_b32_e32 v52, v12
	v_mov_b32_e32 v53, v12
	v_mov_b32_e32 v54, v12
	v_mov_b32_e32 v55, v12
	v_mov_b32_e32 v48, v12
	v_mov_b32_e32 v49, v12
	v_mov_b32_e32 v50, v12
	v_mov_b32_e32 v51, v12
	v_mov_b32_e32 v44, v12
	v_mov_b32_e32 v45, v12
	v_mov_b32_e32 v46, v12
	v_mov_b32_e32 v47, v12
	v_mov_b32_e32 v40, v12
	v_mov_b32_e32 v41, v12
	v_mov_b32_e32 v42, v12
	v_mov_b32_e32 v43, v12
	v_mov_b32_e32 v28, v12
	v_mov_b32_e32 v29, v12
	v_mov_b32_e32 v30, v12
	v_mov_b32_e32 v31, v12
	v_mov_b32_e32 v16, v12
	v_mov_b32_e32 v17, v12
	v_mov_b32_e32 v18, v12
	v_mov_b32_e32 v19, v12
	v_mov_b32_e32 v20, v12
	v_mov_b32_e32 v21, v12
	v_mov_b32_e32 v22, v12
	v_mov_b32_e32 v23, v12
	v_mov_b32_e32 v24, v12
	v_mov_b32_e32 v25, v12
	v_mov_b32_e32 v26, v12
	v_mov_b32_e32 v27, v12
	v_mov_b32_e32 v32, v12
	v_mov_b32_e32 v33, v12
	v_mov_b32_e32 v34, v12
	v_mov_b32_e32 v35, v12
	v_mov_b32_e32 v36, v12
	v_mov_b32_e32 v37, v12
	v_mov_b32_e32 v38, v12
	v_mov_b32_e32 v39, v12
	v_mov_b32_e32 v8, v12
	v_mov_b32_e32 v9, v12
	v_mov_b32_e32 v10, v12
	v_mov_b32_e32 v11, v12
	v_mov_b32_e32 v4, v12
	v_mov_b32_e32 v5, v12
	v_mov_b32_e32 v6, v12
	v_mov_b32_e32 v7, v12
	v_mov_b32_e32 v0, v12
	v_mov_b32_e32 v1, v12
	v_mov_b32_e32 v2, v12
	v_mov_b32_e32 v3, v12
	v_mov_b32_e32 v126, v112
.LBB0_3:
	s_mov_b32 s60, s15
	s_mov_b32 s15, s61
	s_mov_b32 s61, s19
	s_mov_b32 s19, s62
	v_add_u32_e32 v127, 0xfffb2000, v115
	s_add_i32 m0, s15, s35
	s_barrier
	buffer_load_dwordx4 v127, s[24:27], 0 offen lds
	v_add_u32_e32 v127, 0xfffcc000, v115
	s_add_i32 m0, s15, s40
	v_add_u32_e32 v129, v109, v113
	buffer_load_dwordx4 v127, s[24:27], 0 offen lds
	v_add_u32_e32 v127, 0xfffe6000, v115
	s_add_i32 m0, s15, s41
	v_add_u32_e32 v130, 0xfffffdff, v129
	buffer_load_dwordx4 v127, s[24:27], 0 offen lds
	v_add_u32_e32 v127, v85, v114
	v_add_u32_e32 v128, 0x200, v127
	v_cmp_gt_i32_e32 vcc, 0, v128
	v_add_u32_e32 v127, 0x60, v127
	v_add_u32_e32 v129, 0xffffff9f, v129
	v_cndmask_b32_e32 v130, v128, v130, vcc
	v_mul_hi_i32 v130, v130, s43
	v_cmp_gt_i32_e32 vcc, 0, v127
	v_lshrrev_b32_e32 v131, 31, v130
	v_lshrrev_b32_e32 v130, 7, v130
	v_cndmask_b32_e32 v129, v127, v129, vcc
	v_add_u32_e32 v130, v130, v131
	v_ashrrev_i32_e32 v128, 31, v128
	v_mul_hi_i32 v129, v129, s43
	v_xor_b32_e32 v128, v130, v128
	v_lshrrev_b32_e32 v130, 31, v129
	v_lshrrev_b32_e32 v129, 7, v129
	v_add_u32_e32 v129, v129, v130
	v_ashrrev_i32_e32 v127, 31, v127
	v_xor_b32_e32 v127, v129, v127
	v_mul_lo_u32 v128, v128, s34
	v_mad_i32_i24 v127, v127, s56, v128
	s_add_i32 m0, s15, s42
	v_add3_u32 v128, v126, v86, v127
	buffer_load_dwordx4 v115, s[24:27], 0 offen lds
	v_add_u32_e32 v129, 0x180, v128
	s_add_i32 m0, s62, s44
	v_add3_u32 v127, v86, v112, v127
	buffer_load_dword v129, s[20:23], 0 offen lds
	v_add_u32_e32 v129, 0x6980, v128
	s_add_i32 m0, s62, s45
	s_nop 0
	buffer_load_dword v129, s[20:23], 0 offen lds
	v_add_u32_e32 v129, 0xd180, v128
	s_add_i32 m0, s62, s46
	s_nop 0
	buffer_load_dword v129, s[20:23], 0 offen lds
	v_add_u32_e32 v129, 0x13980, v128
	s_add_i32 m0, s62, s47
	s_nop 0
	buffer_load_dword v129, s[20:23], 0 offen lds
	v_add_u32_e32 v129, 0x1a180, v128
	s_add_i32 m0, s62, s48
	v_add_u32_e32 v128, 0x20980, v128
	buffer_load_dword v129, s[20:23], 0 offen lds
	s_add_i32 m0, s62, s49
	s_nop 0
	buffer_load_dword v128, s[20:23], 0 offen lds
	v_add_u32_e32 v128, 0x27180, v127
	v_cndmask_b32_e64 v128, v87, v128, s[6:7]
	s_add_i32 m0, s62, s50
	s_nop 0
	buffer_load_dword v128, s[20:23], 0 offen lds
	v_add_u32_e32 v128, 0x2d980, v127
	v_cndmask_b32_e64 v128, v87, v128, s[6:7]
	s_add_i32 m0, s62, s51
	s_nop 0
	buffer_load_dword v128, s[20:23], 0 offen lds
	v_add_u32_e32 v128, 0x34180, v127
	v_cndmask_b32_e64 v128, v87, v128, s[6:7]
	s_add_i32 m0, s62, s52
	v_add_u32_e32 v127, 0x3a980, v127
	buffer_load_dword v128, s[20:23], 0 offen lds
	v_cndmask_b32_e64 v127, v87, v127, s[6:7]
	s_add_i32 m0, s62, s53
	s_nop 0
	buffer_load_dword v127, s[20:23], 0 offen lds
	v_add_u32_e32 v127, v85, v110
	v_add_u32_e32 v129, v109, v111
	v_add_u32_e32 v128, 64, v127
	v_add_u32_e32 v130, 0xffffffbf, v129
	v_cmp_gt_i32_e32 vcc, s54, v127
	v_add_u32_e32 v131, 12, v127
	v_add_u32_e32 v129, -13, v129
	v_cndmask_b32_e32 v128, v128, v130, vcc
	v_mul_hi_i32 v128, v128, s43
	v_lshrrev_b32_e32 v130, 31, v128
	v_ashrrev_i32_e32 v128, 3, v128
	v_add_u32_e32 v128, v128, v130
	v_cndmask_b32_e64 v130, 0, -1, vcc
	v_cmp_gt_i32_e32 vcc, -12, v127
	v_xor_b32_e32 v128, v128, v130
	v_add_u32_e32 v130, v128, v84
	v_cndmask_b32_e32 v127, v131, v129, vcc
	v_mul_hi_i32 v127, v127, s43
	v_lshrrev_b32_e32 v129, 31, v127
	v_lshrrev_b32_e32 v127, 3, v127
	v_add_u32_e32 v127, v127, v129
	v_cndmask_b32_e64 v129, 0, -1, vcc
	v_xor_b32_e32 v127, v127, v129
	v_add_u32_e32 v128, v107, v128
	v_mul_lo_u32 v129, v130, s14
	v_mul_lo_u32 v127, v127, s5
	v_mul_lo_u32 v128, s14, v128
	v_sub_u32_e32 v129, v129, v127
	v_sub_u32_e32 v127, v128, v127
	v_add3_u32 v129, v86, v129, v108
	v_add3_u32 v127, v86, v127, v108
	buffer_load_dword v128, v129, s[28:31], 0 offen
	s_nop 0
	buffer_load_dword v127, v127, s[28:31], 0 offen
	v_add_u32_e32 v129, v86, v104
	v_add_u32_e32 v131, s0, v103
	v_add_u32_e32 v130, 0x100, v129
	v_add_u32_e32 v132, 0xfffffeff, v131
	v_cmp_gt_i32_e32 vcc, s1, v129
	v_add_u32_e32 v139, 48, v129
	v_subrev_u32_e32 v135, 49, v131
	v_cndmask_b32_e32 v132, v130, v132, vcc
	v_mul_hi_i32 v132, v132, s43
	v_lshrrev_b32_e32 v133, 31, v132
	v_ashrrev_i32_e32 v132, 5, v132
	v_add_u32_e32 v132, v132, v133
	v_cndmask_b32_e64 v133, 0, -1, vcc
	v_cmp_gt_i32_e32 vcc, s9, v129
	v_xor_b32_e32 v132, v132, v133
	v_add_u32_e32 v133, v132, v88
	v_cndmask_b32_e32 v135, v139, v135, vcc
	v_mul_hi_i32 v135, v135, s43
	v_lshrrev_b32_e32 v136, 31, v135
	v_ashrrev_i32_e32 v135, 5, v135
	v_add_u32_e32 v135, v135, v136
	v_cndmask_b32_e64 v136, 0, -1, vcc
	v_xor_b32_e32 v135, v135, v136
	v_mul_lo_u32 v133, v133, s18
	v_mul_lo_u32 v140, v135, s5
	v_add_u32_e32 v134, v132, v97
	v_sub_u32_e32 v133, v133, v140
	v_add_u32_e32 v133, v139, v133
	v_cmp_gt_i32_e32 vcc, s4, v134
	v_add_u32_e32 v141, 0x102, v129
	v_add_u32_e32 v131, 0xfffffefd, v131
	v_cndmask_b32_e32 v133, v87, v133, vcc
	v_cmp_gt_i32_e32 vcc, -2, v130
	v_add_u32_e32 v137, s0, v102
	v_subrev_u32_e32 v138, 50, v137
	v_cndmask_b32_e32 v130, v141, v131, vcc
	v_cmp_gt_i32_e32 vcc, s58, v129
	v_mul_hi_i32 v130, v130, s43
	v_lshrrev_b32_e32 v135, 31, v130
	v_cndmask_b32_e32 v129, v141, v131, vcc
	v_ashrrev_i32_e32 v130, 5, v130
	v_mul_hi_i32 v129, v129, s43
	v_add_u32_e32 v130, v130, v135
	v_ashrrev_i32_e32 v135, 31, v141
	v_lshrrev_b32_e32 v131, 31, v129
	v_ashrrev_i32_e32 v129, 5, v129
	v_xor_b32_e32 v142, v130, v135
	v_add_u32_e32 v129, v129, v131
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_add_u32_e32 v130, v142, v88
	v_xor_b32_e32 v129, v129, v131
	v_mul_lo_u32 v130, v130, s18
	v_mul_lo_u32 v144, v129, s5
	v_add_u32_e32 v143, v142, v97
	v_sub_u32_e32 v129, v130, v144
	v_add_u32_e32 v129, v141, v129
	v_cmp_gt_i32_e32 vcc, s4, v143
	v_add_u32_e32 v130, v86, v100
	v_add_u32_e32 v135, s0, v101
	v_cndmask_b32_e32 v129, v87, v129, vcc
	v_add_u32_e32 v131, 0x101, v130
	v_add_u32_e32 v135, 0xfffffefe, v135
	v_cmp_gt_i32_e32 vcc, s17, v130
	v_add_u32_e32 v137, 0xfffffefc, v137
	v_add_u32_e32 v150, v105, v142
	v_cndmask_b32_e32 v135, v131, v135, vcc
	v_mul_hi_i32 v135, v135, s43
	v_lshrrev_b32_e32 v136, 31, v135
	v_ashrrev_i32_e32 v135, 5, v135
	v_add_u32_e32 v135, v135, v136
	v_add_u32_e32 v136, 49, v130
	v_cmp_gt_i32_e32 vcc, s59, v130
	v_ashrrev_i32_e32 v131, 31, v131
	v_xor_b32_e32 v131, v135, v131
	v_cndmask_b32_e32 v138, v136, v138, vcc
	v_mul_hi_i32 v138, v138, s43
	v_lshrrev_b32_e32 v145, 31, v138
	v_ashrrev_i32_e32 v138, 5, v138
	v_add_u32_e32 v138, v138, v145
	v_ashrrev_i32_e32 v145, 31, v136
	v_add_u32_e32 v135, v131, v89
	v_xor_b32_e32 v138, v138, v145
	v_mul_lo_u32 v135, v135, s18
	v_mul_lo_u32 v138, v138, s5
	v_sub_u32_e32 v135, v135, v138
	v_add_u32_e32 v135, v136, v135
	v_cmp_lt_i32_e32 vcc, v131, v98
	v_add_u32_e32 v145, 0x103, v130
	v_mul_lo_u32 v150, s18, v150
	v_cndmask_b32_e32 v135, v87, v135, vcc
	v_cmp_gt_i32_e32 vcc, s57, v130
	v_add_u32_e32 v151, v131, v92
	v_sub_u32_e32 v150, v150, v144
	v_cndmask_b32_e32 v146, v145, v137, vcc
	v_mul_hi_i32 v146, v146, s43
	v_lshrrev_b32_e32 v147, 31, v146
	v_ashrrev_i32_e32 v146, 5, v146
	v_add_u32_e32 v146, v146, v147
	v_ashrrev_i32_e32 v147, 31, v145
	v_xor_b32_e32 v146, v146, v147
	v_add_u32_e32 v148, v146, v89
	v_mul_lo_u32 v148, v148, s18
	v_mul_lo_u32 v149, v146, s5
	v_sub_u32_e32 v148, v148, v149
	v_add_u32_e32 v148, v145, v148
	v_cmp_lt_i32_e32 vcc, v146, v98
	v_mul_lo_u32 v151, v151, s18
	v_add_u32_e32 v150, v141, v150
	v_cndmask_b32_e32 v146, v87, v148, vcc
	v_add_u32_e32 v148, v105, v132
	v_mul_lo_u32 v148, s18, v148
	v_sub_u32_e32 v148, v148, v140
	v_add_u32_e32 v148, v139, v148
	v_cmp_gt_i32_e32 vcc, s8, v134
	v_sub_u32_e32 v138, v151, v138
	v_add_u32_e32 v136, v136, v138
	v_cndmask_b32_e32 v148, v87, v148, vcc
	v_cmp_gt_i32_e32 vcc, s8, v143
	v_add_u32_e32 v130, 0x100, v130
	v_add_u32_e32 v132, v106, v132
	v_cndmask_b32_e32 v150, v87, v150, vcc
	v_cmp_lt_i32_e32 vcc, v131, v99
	v_mul_lo_u32 v132, s18, v132
	v_sub_u32_e32 v132, v132, v140
	v_cndmask_b32_e32 v151, v87, v136, vcc
	v_cmp_gt_i32_e32 vcc, -3, v130
	v_add_u32_e32 v132, v139, v132
	s_nop 0
	v_cndmask_b32_e32 v130, v145, v137, vcc
	v_mul_hi_i32 v130, v130, s43
	v_lshrrev_b32_e32 v131, 31, v130
	v_ashrrev_i32_e32 v130, 5, v130
	v_add_u32_e32 v130, v130, v131
	v_xor_b32_e32 v130, v130, v147
	v_add_u32_e32 v131, v130, v92
	v_mul_lo_u32 v131, v131, s18
	v_sub_u32_e32 v131, v131, v149
	v_add_u32_e32 v131, v145, v131
	v_cmp_lt_i32_e32 vcc, v130, v99
	s_nop 1
	v_cndmask_b32_e32 v145, v87, v131, vcc
	buffer_load_ubyte v138, v133, s[36:39], 0 offen
	buffer_load_ubyte v137, v129, s[36:39], 0 offen
	buffer_load_ubyte v136, v135, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v135, v146, s[36:39], 0 offen
	buffer_load_ubyte v133, v148, s[36:39], 0 offen
	buffer_load_ubyte v131, v150, s[36:39], 0 offen
	buffer_load_ubyte v130, v151, s[36:39], 0 offen
	buffer_load_ubyte v129, v145, s[36:39], 0 offen
	v_cmp_gt_i32_e32 vcc, s55, v134
	v_add_u32_e32 v134, v106, v142
	v_mul_lo_u32 v134, s18, v134
	v_sub_u32_e32 v134, v134, v144
	v_cndmask_b32_e32 v132, v87, v132, vcc
	v_add_u32_e32 v134, v141, v134
	v_cmp_gt_i32_e32 vcc, s55, v143
	s_nop 1
	v_cndmask_b32_e32 v139, v87, v134, vcc
	buffer_load_ubyte v134, v132, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v132, v139, s[36:39], 0 offen
	v_add_u32_e32 v139, s60, v93
	ds_read_b128 v[140:143], v139
	ds_read_b128 v[144:147], v139 offset:2048
	ds_read_b128 v[148:151], v139 offset:4096
	ds_read_b128 v[152:155], v139 offset:6144
	v_add_u32_e32 v139, s61, v95
	ds_read_b128 v[156:159], v139
	ds_read_b128 v[160:163], v139 offset:2048
	ds_read_b128 v[164:167], v139 offset:4096
	ds_read_b128 v[168:171], v139 offset:6144
	ds_read_b128 v[172:175], v139 offset:8192
	s_barrier
	s_setprio 1
	v_and_b32_e32 v124, 0xff, v124
	v_and_b32_e32 v123, 0xff, v123
	v_and_b32_e32 v122, 0xff, v122
	v_and_b32_e32 v121, 0xff, v121
	v_and_b32_e32 v125, 0xff, v125
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[140:143], v[156:159], v[12:15], v91, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[140:143], v[160:163], v[76:79], v91, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[140:143], v[164:167], v[72:75], v91, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[140:143], v[168:171], v[68:71], v91, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[140:143], v[172:175], v[64:67], v91, v125 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[144:147], v[156:159], v[60:63], v91, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[144:147], v[160:163], v[56:59], v91, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[144:147], v[164:167], v[52:55], v91, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[144:147], v[168:171], v[48:51], v91, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[144:147], v[172:175], v[44:47], v91, v125 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[148:151], v[156:159], v[40:43], v90, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[148:151], v[160:163], v[28:31], v90, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[148:151], v[164:167], v[16:19], v90, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[148:151], v[168:171], v[20:23], v90, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[148:151], v[172:175], v[24:27], v90, v125 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[152:155], v[156:159], v[32:35], v90, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[152:155], v[160:163], v[36:39], v90, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[152:155], v[164:167], v[8:11], v90, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[152:155], v[168:171], v[4:7], v90, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[152:155], v[172:175], v[0:3], v90, v125 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v121, s60, v94
	ds_read_b128 v[122:125], v121
	ds_read_b128 v[140:143], v121 offset:2048
	ds_read_b128 v[144:147], v121 offset:4096
	ds_read_b128 v[148:151], v121 offset:6144
	v_add_u32_e32 v121, s61, v96
	ds_read_b128 v[152:155], v121
	ds_read_b128 v[156:159], v121 offset:2048
	ds_read_b128 v[160:163], v121 offset:4096
	ds_read_b128 v[164:167], v121 offset:6144
	ds_read_b128 v[168:171], v121 offset:8192
	s_waitcnt vmcnt(12)
	s_barrier
	s_setprio 1
	v_and_b32_e32 v119, 0xff, v119
	v_and_b32_e32 v118, 0xff, v118
	v_and_b32_e32 v117, 0xff, v117
	v_and_b32_e32 v116, 0xff, v116
	v_and_b32_e32 v120, 0xff, v120
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[122:125], v[152:155], v[12:15], v91, v119 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[122:125], v[156:159], v[76:79], v91, v118 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[122:125], v[160:163], v[72:75], v91, v117 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[122:125], v[164:167], v[68:71], v91, v116 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[122:125], v[168:171], v[64:67], v91, v120 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[140:143], v[152:155], v[60:63], v91, v119 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[140:143], v[156:159], v[56:59], v91, v118 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[140:143], v[160:163], v[52:55], v91, v117 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[140:143], v[164:167], v[48:51], v91, v116 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[140:143], v[168:171], v[44:47], v91, v120 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[144:147], v[152:155], v[40:43], v90, v119 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[144:147], v[156:159], v[28:31], v90, v118 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[144:147], v[160:163], v[16:19], v90, v117 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[144:147], v[164:167], v[20:23], v90, v116 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[144:147], v[168:171], v[24:27], v90, v120 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[148:151], v[152:155], v[32:35], v90, v119 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[148:151], v[156:159], v[36:39], v90, v118 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[148:151], v[160:163], v[8:11], v90, v117 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[148:151], v[164:167], v[4:7], v90, v116 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[148:151], v[168:171], v[0:3], v90, v120 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_addk_i32 s0, 0xff00
	v_add_u32_e32 v100, 0x100, v100
	v_add_u32_e32 v104, 0x100, v104
	v_add_u32_e32 v108, 0x100, v108
	v_subrev_u32_e32 v111, 64, v111
	v_add_u32_e32 v110, 64, v110
	v_add_u32_e32 v126, 0x800, v126
	v_add_u32_e32 v112, 0x800, v112
	v_add_u32_e32 v113, 0xfffffe00, v113
	v_add_u32_e32 v114, 0x200, v114
	v_add_u32_e32 v115, 0x80, v115
	s_cmpk_lg_i32 s0, 0xf400
	s_mov_b32 s62, s61
	s_mov_b32 s61, s60
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v120, v132
	v_mov_b32_e32 v125, v134
	v_mov_b32_e32 v116, v129
	v_mov_b32_e32 v121, v130
	v_mov_b32_e32 v117, v131
	v_mov_b32_e32 v122, v133
	v_mov_b32_e32 v118, v135
	v_mov_b32_e32 v123, v136
	v_mov_b32_e32 v119, v137
	v_mov_b32_e32 v124, v138
	v_mov_b32_e32 v90, v127
	v_mov_b32_e32 v91, v128
	s_cbranch_scc1 .LBB0_3
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	v_add_u32_e32 v85, s19, v95
	v_add_u32_e32 v86, s19, v96
	s_barrier
	ds_read_b128 v[110:113], v85
	ds_read_b128 v[114:117], v85 offset:2048
	ds_read_b128 v[118:121], v86
	ds_read_b128 v[122:125], v85 offset:8192
	ds_read_b128 v[140:143], v86 offset:2048
	ds_read_b128 v[144:147], v86 offset:4096
	ds_read_b128 v[148:151], v85 offset:4096
	ds_read_b128 v[152:155], v85 offset:6144
	ds_read_b128 v[156:159], v86 offset:6144
	ds_read_b128 v[160:163], v86 offset:8192
	v_add_u32_e32 v85, s15, v93
	ds_read_b128 v[96:99], v85
	ds_read_b128 v[106:109], v85 offset:2048
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[96:99], v[110:113], v[12:15], v128, v138 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v86, s15, v94
	ds_read_b128 v[100:103], v86
	ds_read_b128 v[164:167], v86 offset:2048
	ds_read_b128 v[168:171], v85 offset:4096
	ds_read_b128 v[172:175], v85 offset:6144
	ds_read_b128 v[176:179], v86 offset:4096
	ds_read_b128 v[180:183], v86 offset:6144
	s_movk_i32 s0, 0x7fff
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[100:103], v[118:121], v[12:15], v128, v137 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mov_b32_e32 v85, 0x7fc0
	s_mul_hi_u32 s1, s12, s16
	v_lshl_or_b32 v83, v83, 2, v84
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[96:99], v[148:151], v[72:75], v128, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_movk_i32 s2, 0x1a00
	s_nop 2
	v_bfe_u32 v86, v15, 16, 1
	v_bfe_u32 v87, v14, 16, 1
	v_add3_u32 v86, v15, v86, s0
	v_bfe_u32 v88, v13, 16, 1
	v_bfe_u32 v89, v12, 16, 1
	v_add3_u32 v87, v14, v87, s0
	v_lshrrev_b32_e32 v92, 16, v86
	v_cmp_o_f32_e32 vcc, v15, v15
	v_add3_u32 v90, v12, v89, s0
	v_add3_u32 v91, v13, v88, s0
	v_lshrrev_b32_e32 v94, 16, v87
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[100:103], v[144:147], v[72:75], v128, v131 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mul_lo_u32 v84, s12, v83
	s_movk_i32 s4, 0x19d0
	s_movk_i32 s6, 0x19c0
	v_cndmask_b32_e32 v72, v85, v92, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_lshrrev_b32_e32 v73, 16, v91
	v_lshrrev_b32_e32 v74, 16, v90
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[96:99], v[152:155], v[68:71], v128, v130 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_cndmask_b32_e32 v68, v85, v94, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[96:99], v[114:117], v[76:79], v128, v136 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v69, v85, v73, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[96:99], v[122:125], v[64:67], v128, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v70, v85, v74, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[100:103], v[160:163], v[12:15], v128, v132 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[106:109], v[110:113], v[60:63], v128, v138 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[100:103], v[140:143], v[76:79], v128, v135 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[164:167], v[118:121], v[12:15], v128, v137 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[106:109], v[114:117], v[56:59], v128, v136 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 4
	v_bfe_u32 v71, v79, 16, 1
	v_bfe_u32 v73, v78, 16, 1
	v_add3_u32 v60, v79, v71, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[100:103], v[156:159], v[90:93], v128, v129 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v74, v77, 16, 1
	v_add3_u32 v73, v78, v73, s0
	v_lshrrev_b32_e32 v60, 16, v60
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[164:167], v[140:143], v[12:15], v128, v135 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v79, v79
	v_bfe_u32 v75, v76, 16, 1
	v_add3_u32 v74, v77, v74, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[106:109], v[148:151], v[52:55], v128, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v61, 16, v73
	v_cndmask_b32_e32 v60, v85, v60, vcc
	v_cmp_o_f32_e32 vcc, v78, v78
	v_add3_u32 v75, v76, v75, s0
	v_lshrrev_b32_e32 v62, 16, v74
	v_cndmask_b32_e32 v56, v85, v61, vcc
	v_cmp_o_f32_e32 vcc, v77, v77
	v_lshrrev_b32_e32 v63, 16, v75
	v_bfe_u32 v59, v89, 16, 1
	v_cndmask_b32_e32 v57, v85, v62, vcc
	v_cmp_o_f32_e32 vcc, v76, v76
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[164:167], v[144:147], v[12:15], v128, v131 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v52, v88, 16, 1
	v_cndmask_b32_e32 v58, v85, v63, vcc
	v_bfe_u32 v53, v87, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[106:109], v[152:155], v[48:51], v128, v130 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v52, v88, v52, s0
	v_cmp_o_f32_e32 vcc, v89, v89
	v_bfe_u32 v54, v86, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[164:167], v[156:159], v[12:15], v128, v129 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v48, v89, v59, s0
	v_lshrrev_b32_e32 v48, 16, v48
	v_add3_u32 v53, v87, v53, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[106:109], v[122:125], v[44:47], v128, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v49, 16, v52
	v_cndmask_b32_e32 v52, v85, v48, vcc
	v_cmp_o_f32_e32 vcc, v88, v88
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[164:167], v[160:163], v[12:15], v128, v132 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v54, v86, v54, s0
	v_lshrrev_b32_e32 v50, 16, v53
	v_cndmask_b32_e32 v53, v85, v49, vcc
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[110:113], v[40:43], v127, v138 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v87, v87
	v_lshrrev_b32_e32 v51, 16, v54
	v_bfe_u32 v44, v93, 16, 1
	v_cndmask_b32_e32 v55, v85, v50, vcc
	v_cmp_o_f32_e32 vcc, v86, v86
	v_bfe_u32 v40, v92, 16, 1
	v_bfe_u32 v41, v91, 16, 1
	v_cndmask_b32_e32 v59, v85, v51, vcc
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[176:179], v[118:121], v[12:15], v127, v137 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v42, v90, 16, 1
	v_add3_u32 v42, v90, v42, s0
	v_add3_u32 v41, v91, v41, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[114:117], v[28:31], v127, v136 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v40, v92, v40, s0
	v_lshrrev_b32_e32 v61, 16, v40
	v_lshrrev_b32_e32 v62, 16, v41
	v_add3_u32 v28, v93, v44, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[176:179], v[140:143], v[12:15], v127, v135 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v63, 16, v42
	v_lshrrev_b32_e32 v28, 16, v28
	v_cmp_o_f32_e32 vcc, v93, v93
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[148:151], v[16:19], v127, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v71, v105, 16, 1
	v_cndmask_b32_e32 v54, v85, v28, vcc
	v_cmp_o_f32_e32 vcc, v92, v92
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[176:179], v[144:147], v[12:15], v127, v131 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v73, v104, 16, 1
	v_add3_u32 v71, v105, v71, s0
	v_add3_u32 v73, v104, v73, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[152:155], v[20:23], v127, v130 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v71, 16, v71
	v_lshrrev_b32_e32 v73, 16, v73
	v_bfe_u32 v78, v107, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[176:179], v[156:159], v[12:15], v127, v129 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v79, v106, 16, 1
	v_add3_u32 v78, v107, v78, s0
	v_add3_u32 v79, v106, v79, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[122:125], v[24:27], v127, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v78, 16, v78
	v_bfe_u32 v86, v51, 16, 1
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v24, v85, v61, vcc
	v_cmp_o_f32_e32 vcc, v91, v91
	v_bfe_u32 v27, v67, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[176:179], v[160:163], v[12:15], v127, v132 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v25, v85, v62, vcc
	v_cmp_o_f32_e32 vcc, v90, v90
	v_add3_u32 v27, v67, v27, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[172:175], v[110:113], v[32:35], v127, v138 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v26, v85, v63, vcc
	v_lshrrev_b32_e32 v27, 16, v27
	v_cmp_o_f32_e32 vcc, v67, v67
	v_bfe_u32 v32, v66, 16, 1
	v_bfe_u32 v33, v65, 16, 1
	v_add3_u32 v32, v66, v32, s0
	v_bfe_u32 v34, v64, 16, 1
	v_add3_u32 v33, v65, v33, s0
	v_lshrrev_b32_e32 v32, 16, v32
	v_cndmask_b32_e32 v27, v85, v27, vcc
	v_cmp_o_f32_e32 vcc, v66, v66
	v_add3_u32 v34, v64, v34, s0
	v_lshrrev_b32_e32 v33, 16, v33
	v_cndmask_b32_e32 v32, v85, v32, vcc
	v_cmp_o_f32_e32 vcc, v65, v65
	v_bfe_u32 v35, v97, 16, 1
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[180:183], v[118:121], v[12:15], v127, v137 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v34, 16, v34
	v_cndmask_b32_e32 v33, v85, v33, vcc
	v_cmp_o_f32_e32 vcc, v64, v64
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[172:175], v[114:117], v[36:39], v127, v136 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v35, v97, v35, s0
	v_cndmask_b32_e32 v34, v85, v34, vcc
	v_lshrrev_b32_e32 v35, 16, v35
	v_bfe_u32 v36, v96, 16, 1
	v_bfe_u32 v37, v95, 16, 1
	v_add3_u32 v36, v96, v36, s0
	v_cmp_o_f32_e32 vcc, v97, v97
	v_bfe_u32 v38, v94, 16, 1
	v_add3_u32 v37, v95, v37, s0
	v_lshrrev_b32_e32 v36, 16, v36
	v_cndmask_b32_e32 v35, v85, v35, vcc
	v_cmp_o_f32_e32 vcc, v96, v96
	v_add3_u32 v38, v94, v38, s0
	v_lshrrev_b32_e32 v37, 16, v37
	v_cndmask_b32_e32 v36, v85, v36, vcc
	v_cmp_o_f32_e32 vcc, v95, v95
	v_bfe_u32 v39, v101, 16, 1
	v_lshrrev_b32_e32 v38, 16, v38
	v_cndmask_b32_e32 v37, v85, v37, vcc
	v_cmp_o_f32_e32 vcc, v94, v94
	v_bfe_u32 v61, v100, 16, 1
	v_add3_u32 v39, v101, v39, s0
	v_cndmask_b32_e32 v38, v85, v38, vcc
	v_bfe_u32 v62, v99, 16, 1
	v_add3_u32 v61, v100, v61, s0
	v_lshrrev_b32_e32 v39, 16, v39
	v_cmp_o_f32_e32 vcc, v101, v101
	v_bfe_u32 v63, v98, 16, 1
	v_add3_u32 v62, v99, v62, s0
	v_lshrrev_b32_e32 v61, 16, v61
	v_cndmask_b32_e32 v39, v85, v39, vcc
	v_cmp_o_f32_e32 vcc, v100, v100
	v_add3_u32 v63, v98, v63, s0
	v_lshrrev_b32_e32 v62, 16, v62
	v_cndmask_b32_e32 v61, v85, v61, vcc
	v_cmp_o_f32_e32 vcc, v99, v99
	v_bfe_u32 v64, v77, 16, 1
	v_lshrrev_b32_e32 v63, 16, v63
	v_cndmask_b32_e32 v62, v85, v62, vcc
	v_cmp_o_f32_e32 vcc, v98, v98
	v_bfe_u32 v65, v76, 16, 1
	v_add3_u32 v64, v77, v64, s0
	v_cndmask_b32_e32 v63, v85, v63, vcc
	v_bfe_u32 v66, v75, 16, 1
	v_add3_u32 v65, v76, v65, s0
	v_lshrrev_b32_e32 v64, 16, v64
	v_cmp_o_f32_e32 vcc, v77, v77
	v_bfe_u32 v67, v74, 16, 1
	v_add3_u32 v66, v75, v66, s0
	v_lshrrev_b32_e32 v65, 16, v65
	v_cndmask_b32_e32 v64, v85, v64, vcc
	v_cmp_o_f32_e32 vcc, v76, v76
	v_add3_u32 v67, v74, v67, s0
	v_lshrrev_b32_e32 v66, 16, v66
	v_cndmask_b32_e32 v65, v85, v65, vcc
	v_cmp_o_f32_e32 vcc, v75, v75
	v_lshrrev_b32_e32 v67, 16, v67
	v_bfe_u32 v75, v102, 16, 1
	v_cndmask_b32_e32 v66, v85, v66, vcc
	v_cmp_o_f32_e32 vcc, v74, v74
	v_bfe_u32 v74, v103, 16, 1
	v_add3_u32 v74, v103, v74, s0
	v_cndmask_b32_e32 v67, v85, v67, vcc
	v_cmp_o_f32_e32 vcc, v105, v105
	v_add3_u32 v75, v102, v75, s0
	v_lshrrev_b32_e32 v74, 16, v74
	v_cndmask_b32_e32 v71, v85, v71, vcc
	v_cmp_o_f32_e32 vcc, v104, v104
	v_bfe_u32 v76, v109, 16, 1
	v_lshrrev_b32_e32 v75, 16, v75
	v_cndmask_b32_e32 v73, v85, v73, vcc
	v_cmp_o_f32_e32 vcc, v103, v103
	v_bfe_u32 v77, v108, 16, 1
	v_add3_u32 v76, v109, v76, s0
	v_cndmask_b32_e32 v74, v85, v74, vcc
	v_cmp_o_f32_e32 vcc, v102, v102
	v_add3_u32 v77, v108, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cndmask_b32_e32 v75, v85, v75, vcc
	v_cmp_o_f32_e32 vcc, v109, v109
	v_lshrrev_b32_e32 v77, 16, v77
	v_bfe_u32 v87, v50, 16, 1
	v_cndmask_b32_e32 v76, v85, v76, vcc
	v_cmp_o_f32_e32 vcc, v108, v108
	v_add3_u32 v86, v51, v86, s0
	v_bfe_u32 v88, v49, 16, 1
	v_cndmask_b32_e32 v77, v85, v77, vcc
	v_cmp_o_f32_e32 vcc, v107, v107
	v_add3_u32 v87, v50, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v78, v85, v78, vcc
	v_cmp_o_f32_e32 vcc, v106, v106
	v_bfe_u32 v89, v48, 16, 1
	v_add3_u32 v88, v49, v88, s0
	v_cndmask_b32_e32 v79, v85, v79, vcc
	v_cmp_o_f32_e32 vcc, v51, v51
	v_lshrrev_b32_e32 v87, 16, v87
	v_add3_u32 v89, v48, v89, s0
	v_cndmask_b32_e32 v51, v85, v86, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_lshrrev_b32_e32 v88, 16, v88
	v_bfe_u32 v86, v47, 16, 1
	v_cndmask_b32_e32 v50, v85, v87, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_lshrrev_b32_e32 v89, 16, v89
	v_bfe_u32 v87, v46, 16, 1
	v_cndmask_b32_e32 v49, v85, v88, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_add3_u32 v86, v47, v86, s0
	v_bfe_u32 v88, v45, 16, 1
	v_cndmask_b32_e32 v48, v85, v89, vcc
	v_add3_u32 v87, v46, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cmp_o_f32_e32 vcc, v47, v47
	v_bfe_u32 v89, v44, 16, 1
	v_add3_u32 v88, v45, v88, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v47, v85, v86, vcc
	v_cmp_o_f32_e32 vcc, v46, v46
	v_add3_u32 v89, v44, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cndmask_b32_e32 v46, v85, v87, vcc
	v_cmp_o_f32_e32 vcc, v45, v45
	v_bfe_u32 v86, v43, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v45, v85, v88, vcc
	v_cmp_o_f32_e32 vcc, v44, v44
	v_bfe_u32 v87, v42, 16, 1
	v_add3_u32 v86, v43, v86, s0
	v_cndmask_b32_e32 v44, v85, v89, vcc
	v_bfe_u32 v88, v41, 16, 1
	v_add3_u32 v87, v42, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cmp_o_f32_e32 vcc, v43, v43
	v_bfe_u32 v89, v40, 16, 1
	v_add3_u32 v88, v41, v88, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v43, v85, v86, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	v_add3_u32 v89, v40, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cndmask_b32_e32 v42, v85, v87, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	v_bfe_u32 v86, v31, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v41, v85, v88, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_bfe_u32 v87, v30, 16, 1
	v_add3_u32 v86, v31, v86, s0
	v_cndmask_b32_e32 v40, v85, v89, vcc
	v_bfe_u32 v88, v29, 16, 1
	v_add3_u32 v87, v30, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cmp_o_f32_e32 vcc, v31, v31
	v_bfe_u32 v89, v28, 16, 1
	v_add3_u32 v88, v29, v88, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v31, v85, v86, vcc
	v_cmp_o_f32_e32 vcc, v30, v30
	v_add3_u32 v89, v28, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cndmask_b32_e32 v30, v85, v87, vcc
	v_cmp_o_f32_e32 vcc, v29, v29
	v_bfe_u32 v86, v23, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v29, v85, v88, vcc
	v_cmp_o_f32_e32 vcc, v28, v28
	v_bfe_u32 v87, v22, 16, 1
	v_add3_u32 v86, v23, v86, s0
	v_cndmask_b32_e32 v28, v85, v89, vcc
	v_bfe_u32 v88, v21, 16, 1
	v_add3_u32 v87, v22, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cmp_o_f32_e32 vcc, v23, v23
	v_bfe_u32 v89, v20, 16, 1
	v_add3_u32 v88, v21, v88, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v23, v85, v86, vcc
	v_cmp_o_f32_e32 vcc, v22, v22
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[180:183], v[140:143], v[12:15], v127, v135 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v89, v20, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cndmask_b32_e32 v22, v85, v87, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[172:175], v[148:151], v[8:11], v127, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v21, v21
	v_bfe_u32 v86, v19, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v21, v85, v88, vcc
	v_cmp_o_f32_e32 vcc, v20, v20
	v_bfe_u32 v87, v18, 16, 1
	v_add3_u32 v86, v19, v86, s0
	v_cndmask_b32_e32 v20, v85, v89, vcc
	v_bfe_u32 v88, v17, 16, 1
	v_add3_u32 v87, v18, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cmp_o_f32_e32 vcc, v19, v19
	v_bfe_u32 v89, v16, 16, 1
	v_add3_u32 v88, v17, v88, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v19, v85, v86, vcc
	v_cmp_o_f32_e32 vcc, v18, v18
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[180:183], v[144:147], v[8:11], v127, v131 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v89, v16, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cndmask_b32_e32 v18, v85, v87, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[172:175], v[152:155], v[4:7], v127, v130 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v17, v17
	v_bfe_u32 v86, v15, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v17, v85, v88, vcc
	v_cmp_o_f32_e32 vcc, v16, v16
	v_bfe_u32 v87, v14, 16, 1
	v_add3_u32 v86, v15, v86, s0
	v_cndmask_b32_e32 v16, v85, v89, vcc
	v_bfe_u32 v88, v13, 16, 1
	v_add3_u32 v87, v14, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cmp_o_f32_e32 vcc, v15, v15
	v_bfe_u32 v89, v12, 16, 1
	v_add3_u32 v88, v13, v88, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v15, v85, v86, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[180:183], v[156:159], v[4:7], v127, v129 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v89, v12, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cndmask_b32_e32 v14, v85, v87, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_bfe_u32 v86, v11, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v13, v85, v88, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v87, v10, 16, 1
	v_add3_u32 v86, v11, v86, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[172:175], v[122:125], v[0:3], v127, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v12, v85, v89, vcc
	v_bfe_u32 v88, v9, 16, 1
	v_add3_u32 v87, v10, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cmp_o_f32_e32 vcc, v11, v11
	v_bfe_u32 v89, v8, 16, 1
	v_add3_u32 v88, v9, v88, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v11, v85, v86, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_add3_u32 v89, v8, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cndmask_b32_e32 v10, v85, v87, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_bfe_u32 v86, v7, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v9, v85, v88, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v87, v6, 16, 1
	v_add3_u32 v86, v7, v86, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[180:183], v[160:163], v[0:3], v127, v132 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v8, v85, v89, vcc
	v_bfe_u32 v88, v5, 16, 1
	v_add3_u32 v87, v6, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v89, v4, 16, 1
	v_add3_u32 v88, v5, v88, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v7, v85, v86, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v89, v4, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cndmask_b32_e32 v6, v85, v87, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_lshrrev_b32_e32 v89, 16, v89
	v_bfe_u32 v86, v3, 16, 1
	v_cndmask_b32_e32 v5, v85, v88, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v87, v2, 16, 1
	v_bfe_u32 v88, v1, 16, 1
	v_cndmask_b32_e32 v4, v85, v89, vcc
	v_bfe_u32 v89, v0, 16, 1
	v_add3_u32 v89, v0, v89, s0
	v_add3_u32 v88, v1, v88, s0
	v_add3_u32 v87, v2, v87, s0
	v_add3_u32 v86, v3, v86, s0
	s_mul_i32 s0, s13, s16
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s12, s16
	v_lshrrev_b32_e32 v86, 16, v86
	v_cmp_o_f32_e32 vcc, v3, v3
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v3, v85, v86, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	s_add_u32 s0, s10, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cndmask_b32_e32 v2, v85, v87, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	s_addc_u32 s1, s11, s1
	s_lshl_b32 s3, s33, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v1, v85, v88, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	s_add_u32 s8, s0, s3
	s_addc_u32 s0, s1, 0
	v_cndmask_b32_e32 v0, v85, v89, vcc
	v_add_u32_e32 v85, s33, v80
	s_and_b32 s1, s12, 0x3fff
	s_lshl_b32 s1, s1, 16
	s_and_b32 s0, s0, 0xffff
	v_add_u32_e32 v86, v84, v80
	v_not_b32_e32 v87, -2.0
	v_cmp_gt_i32_e32 vcc, s2, v85
	s_or_b32 s0, s1, s0
	s_or_b32 s9, s0, 2.0
	v_cndmask_b32_e32 v86, v87, v86, vcc
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, 0x7ffffffd
	v_lshlrev_b32_e32 v86, 1, v86
	buffer_store_short v70, v86, s[8:11], 0 offen
	v_or_b32_e32 v70, 1, v83
	v_mul_lo_u32 v70, s12, v70
	v_add_u32_e32 v86, v70, v80
	v_cndmask_b32_e32 v86, v87, v86, vcc
	v_lshlrev_b32_e32 v86, 1, v86
	buffer_store_short v69, v86, s[8:11], 0 offen
	v_or_b32_e32 v69, 2, v83
	v_mul_lo_u32 v69, s12, v69
	v_add_u32_e32 v86, v69, v80
	v_cndmask_b32_e32 v86, v87, v86, vcc
	v_lshlrev_b32_e32 v86, 1, v86
	buffer_store_short v68, v86, s[8:11], 0 offen
	v_or_b32_e32 v68, 3, v83
	v_mul_lo_u32 v68, s12, v68
	v_add_u32_e32 v86, v68, v80
	v_cndmask_b32_e32 v86, v87, v86, vcc
	v_lshlrev_b32_e32 v86, 1, v86
	s_movk_i32 s0, 0x19f0
	buffer_store_short v72, v86, s[8:11], 0 offen
	v_add_u32_e32 v72, v84, v81
	v_cmp_gt_i32_e64 s[0:1], s0, v85
	s_movk_i32 s2, 0x19e0
	v_cmp_gt_i32_e64 s[2:3], s2, v85
	v_cndmask_b32_e64 v72, v87, v72, s[0:1]
	v_lshlrev_b32_e32 v72, 1, v72
	buffer_store_short v58, v72, s[8:11], 0 offen
	v_add_u32_e32 v58, v70, v81
	v_cndmask_b32_e64 v58, v87, v58, s[0:1]
	v_lshlrev_b32_e32 v58, 1, v58
	buffer_store_short v57, v58, s[8:11], 0 offen
	v_add_u32_e32 v57, v69, v81
	v_cndmask_b32_e64 v57, v87, v57, s[0:1]
	v_lshlrev_b32_e32 v57, 1, v57
	buffer_store_short v56, v57, s[8:11], 0 offen
	v_add_u32_e32 v56, v68, v81
	v_cndmask_b32_e64 v56, v87, v56, s[0:1]
	v_lshlrev_b32_e32 v56, 1, v56
	buffer_store_short v60, v56, s[8:11], 0 offen
	v_add_u32_e32 v56, 32, v80
	v_add_u32_e32 v57, v84, v56
	v_cndmask_b32_e64 v57, v87, v57, s[2:3]
	v_lshlrev_b32_e32 v57, 1, v57
	buffer_store_short v59, v57, s[8:11], 0 offen
	v_add_u32_e32 v57, v70, v56
	v_cndmask_b32_e64 v57, v87, v57, s[2:3]
	v_lshlrev_b32_e32 v57, 1, v57
	buffer_store_short v55, v57, s[8:11], 0 offen
	v_add_u32_e32 v55, v69, v56
	v_cndmask_b32_e64 v55, v87, v55, s[2:3]
	v_lshlrev_b32_e32 v55, 1, v55
	buffer_store_short v53, v55, s[8:11], 0 offen
	v_add_u32_e32 v53, v68, v56
	v_cndmask_b32_e64 v53, v87, v53, s[2:3]
	v_lshlrev_b32_e32 v53, 1, v53
	buffer_store_short v52, v53, s[8:11], 0 offen
	v_add_u32_e32 v52, v84, v82
	v_cmp_gt_i32_e64 s[4:5], s4, v85
	v_cmp_gt_i32_e64 s[6:7], s6, v85
	s_nop 0
	v_cndmask_b32_e64 v52, v87, v52, s[4:5]
	v_lshlrev_b32_e32 v52, 1, v52
	buffer_store_short v26, v52, s[8:11], 0 offen
	v_add_u32_e32 v26, v70, v82
	v_cndmask_b32_e64 v26, v87, v26, s[4:5]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v25, v26, s[8:11], 0 offen
	v_add_u32_e32 v25, v69, v82
	v_cndmask_b32_e64 v25, v87, v25, s[4:5]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v24, v25, s[8:11], 0 offen
	v_add_u32_e32 v24, v68, v82
	v_cndmask_b32_e64 v24, v87, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v54, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, 64, v80
	v_add_u32_e32 v25, v84, v24
	v_cndmask_b32_e64 v25, v87, v25, s[6:7]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v34, v25, s[8:11], 0 offen
	v_add_u32_e32 v25, v70, v24
	v_cndmask_b32_e64 v25, v87, v25, s[6:7]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v33, v25, s[8:11], 0 offen
	v_add_u32_e32 v25, v69, v24
	v_cndmask_b32_e64 v25, v87, v25, s[6:7]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v32, v25, s[8:11], 0 offen
	v_add_u32_e32 v25, v68, v24
	v_cndmask_b32_e64 v25, v87, v25, s[6:7]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v27, v25, s[8:11], 0 offen
	v_or_b32_e32 v25, 16, v83
	v_mul_lo_u32 v25, s12, v25
	v_add_u32_e32 v26, v25, v80
	v_cndmask_b32_e32 v26, v87, v26, vcc
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v38, v26, s[8:11], 0 offen
	v_or_b32_e32 v26, 17, v83
	v_mul_lo_u32 v26, s12, v26
	v_add_u32_e32 v27, v26, v80
	v_cndmask_b32_e32 v27, v87, v27, vcc
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v37, v27, s[8:11], 0 offen
	v_or_b32_e32 v27, 18, v83
	v_mul_lo_u32 v27, s12, v27
	v_add_u32_e32 v32, v27, v80
	v_cndmask_b32_e32 v32, v87, v32, vcc
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v36, v32, s[8:11], 0 offen
	v_or_b32_e32 v32, 19, v83
	v_mul_lo_u32 v32, s12, v32
	v_add_u32_e32 v33, v32, v80
	v_cndmask_b32_e32 v33, v87, v33, vcc
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v35, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v25, v81
	v_cndmask_b32_e64 v33, v87, v33, s[0:1]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v63, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v26, v81
	v_cndmask_b32_e64 v33, v87, v33, s[0:1]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v62, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v27, v81
	v_cndmask_b32_e64 v33, v87, v33, s[0:1]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v61, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v32, v81
	v_cndmask_b32_e64 v33, v87, v33, s[0:1]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v39, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v25, v56
	v_cndmask_b32_e64 v33, v87, v33, s[2:3]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v67, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v26, v56
	v_cndmask_b32_e64 v33, v87, v33, s[2:3]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v66, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v27, v56
	v_cndmask_b32_e64 v33, v87, v33, s[2:3]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v65, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v32, v56
	v_cndmask_b32_e64 v33, v87, v33, s[2:3]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v64, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v25, v82
	v_cndmask_b32_e64 v33, v87, v33, s[4:5]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v75, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v26, v82
	v_cndmask_b32_e64 v33, v87, v33, s[4:5]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v74, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v27, v82
	v_cndmask_b32_e64 v33, v87, v33, s[4:5]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v73, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v32, v82
	v_add_u32_e32 v25, v25, v24
	v_cndmask_b32_e64 v33, v87, v33, s[4:5]
	v_cndmask_b32_e64 v25, v87, v25, s[6:7]
	v_lshlrev_b32_e32 v33, 1, v33
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v71, v33, s[8:11], 0 offen
	buffer_store_short v79, v25, s[8:11], 0 offen
	v_add_u32_e32 v25, v26, v24
	v_cndmask_b32_e64 v25, v87, v25, s[6:7]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v78, v25, s[8:11], 0 offen
	v_add_u32_e32 v25, v27, v24
	v_cndmask_b32_e64 v25, v87, v25, s[6:7]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v77, v25, s[8:11], 0 offen
	v_add_u32_e32 v25, v32, v24
	v_cndmask_b32_e64 v25, v87, v25, s[6:7]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v76, v25, s[8:11], 0 offen
	v_or_b32_e32 v25, 32, v83
	v_mul_lo_u32 v25, s12, v25
	v_add_u32_e32 v26, v25, v80
	v_cndmask_b32_e32 v26, v87, v26, vcc
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v48, v26, s[8:11], 0 offen
	v_or_b32_e32 v26, 33, v83
	v_mul_lo_u32 v26, s12, v26
	v_add_u32_e32 v27, v26, v80
	v_cndmask_b32_e32 v27, v87, v27, vcc
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v49, v27, s[8:11], 0 offen
	v_or_b32_e32 v27, 34, v83
	v_mul_lo_u32 v27, s12, v27
	v_add_u32_e32 v32, v27, v80
	v_cndmask_b32_e32 v32, v87, v32, vcc
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v50, v32, s[8:11], 0 offen
	v_or_b32_e32 v32, 35, v83
	v_mul_lo_u32 v32, s12, v32
	v_add_u32_e32 v33, v32, v80
	v_cndmask_b32_e32 v33, v87, v33, vcc
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v51, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v25, v81
	v_cndmask_b32_e64 v33, v87, v33, s[0:1]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v44, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v26, v81
	v_cndmask_b32_e64 v33, v87, v33, s[0:1]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v45, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v27, v81
	v_cndmask_b32_e64 v33, v87, v33, s[0:1]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v46, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v32, v81
	v_cndmask_b32_e64 v33, v87, v33, s[0:1]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v47, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v25, v56
	v_cndmask_b32_e64 v33, v87, v33, s[2:3]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v40, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v26, v56
	v_cndmask_b32_e64 v33, v87, v33, s[2:3]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v41, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v27, v56
	v_cndmask_b32_e64 v33, v87, v33, s[2:3]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v42, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v32, v56
	v_cndmask_b32_e64 v33, v87, v33, s[2:3]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v43, v33, s[8:11], 0 offen
	v_add_u32_e32 v33, v25, v82
	v_cndmask_b32_e64 v33, v87, v33, s[4:5]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v28, v33, s[8:11], 0 offen
	v_add_u32_e32 v28, v26, v82
	v_cndmask_b32_e64 v28, v87, v28, s[4:5]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v29, v28, s[8:11], 0 offen
	v_add_u32_e32 v28, v27, v82
	v_cndmask_b32_e64 v28, v87, v28, s[4:5]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v30, v28, s[8:11], 0 offen
	v_add_u32_e32 v28, v32, v82
	v_add_u32_e32 v25, v25, v24
	v_cndmask_b32_e64 v28, v87, v28, s[4:5]
	v_cndmask_b32_e64 v25, v87, v25, s[6:7]
	v_lshlrev_b32_e32 v28, 1, v28
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v31, v28, s[8:11], 0 offen
	buffer_store_short v20, v25, s[8:11], 0 offen
	v_add_u32_e32 v20, v26, v24
	v_cndmask_b32_e64 v20, v87, v20, s[6:7]
	v_lshlrev_b32_e32 v20, 1, v20
	buffer_store_short v21, v20, s[8:11], 0 offen
	v_add_u32_e32 v20, v27, v24
	v_cndmask_b32_e64 v20, v87, v20, s[6:7]
	v_lshlrev_b32_e32 v20, 1, v20
	buffer_store_short v22, v20, s[8:11], 0 offen
	v_add_u32_e32 v20, v32, v24
	v_cndmask_b32_e64 v20, v87, v20, s[6:7]
	v_lshlrev_b32_e32 v20, 1, v20
	buffer_store_short v23, v20, s[8:11], 0 offen
	v_or_b32_e32 v20, 48, v83
	v_mul_lo_u32 v20, s12, v20
	v_add_u32_e32 v21, v20, v80
	v_cndmask_b32_e32 v21, v87, v21, vcc
	v_lshlrev_b32_e32 v21, 1, v21
	buffer_store_short v16, v21, s[8:11], 0 offen
	v_or_b32_e32 v16, 49, v83
	v_mul_lo_u32 v16, s12, v16
	v_add_u32_e32 v21, v16, v80
	v_cndmask_b32_e32 v21, v87, v21, vcc
	v_lshlrev_b32_e32 v21, 1, v21
	buffer_store_short v17, v21, s[8:11], 0 offen
	v_or_b32_e32 v17, 50, v83
	v_mul_lo_u32 v17, s12, v17
	v_add_u32_e32 v21, v17, v80
	v_cndmask_b32_e32 v21, v87, v21, vcc
	v_lshlrev_b32_e32 v21, 1, v21
	buffer_store_short v18, v21, s[8:11], 0 offen
	v_or_b32_e32 v18, 51, v83
	v_mul_lo_u32 v18, s12, v18
	v_add_u32_e32 v21, v18, v80
	v_cndmask_b32_e32 v21, v87, v21, vcc
	v_lshlrev_b32_e32 v21, 1, v21
	buffer_store_short v19, v21, s[8:11], 0 offen
	v_add_u32_e32 v19, v20, v81
	v_cndmask_b32_e64 v19, v87, v19, s[0:1]
	v_lshlrev_b32_e32 v19, 1, v19
	buffer_store_short v12, v19, s[8:11], 0 offen
	v_add_u32_e32 v12, v16, v81
	v_cndmask_b32_e64 v12, v87, v12, s[0:1]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v13, v12, s[8:11], 0 offen
	v_add_u32_e32 v12, v17, v81
	v_cndmask_b32_e64 v12, v87, v12, s[0:1]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v14, v12, s[8:11], 0 offen
	v_add_u32_e32 v12, v18, v81
	v_cndmask_b32_e64 v12, v87, v12, s[0:1]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v15, v12, s[8:11], 0 offen
	v_add_u32_e32 v12, v20, v56
	v_cndmask_b32_e64 v12, v87, v12, s[2:3]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v8, v12, s[8:11], 0 offen
	v_add_u32_e32 v8, v16, v56
	v_cndmask_b32_e64 v8, v87, v8, s[2:3]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v9, v8, s[8:11], 0 offen
	v_add_u32_e32 v8, v17, v56
	v_cndmask_b32_e64 v8, v87, v8, s[2:3]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v10, v8, s[8:11], 0 offen
	v_add_u32_e32 v8, v18, v56
	v_cndmask_b32_e64 v8, v87, v8, s[2:3]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v11, v8, s[8:11], 0 offen
	v_add_u32_e32 v8, v20, v82
	v_cndmask_b32_e64 v8, v87, v8, s[4:5]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v4, v8, s[8:11], 0 offen
	v_add_u32_e32 v4, v16, v82
	v_cndmask_b32_e64 v4, v87, v4, s[4:5]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v5, v4, s[8:11], 0 offen
	v_add_u32_e32 v4, v17, v82
	v_cndmask_b32_e64 v4, v87, v4, s[4:5]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v6, v4, s[8:11], 0 offen
	v_add_u32_e32 v4, v18, v82
	v_cndmask_b32_e64 v4, v87, v4, s[4:5]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v7, v4, s[8:11], 0 offen
	v_add_u32_e32 v4, v20, v24
	v_cndmask_b32_e64 v4, v87, v4, s[6:7]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v0, v4, s[8:11], 0 offen
	v_add_u32_e32 v0, v16, v24
	v_cndmask_b32_e64 v0, v87, v0, s[6:7]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v1, v0, s[8:11], 0 offen
	v_add_u32_e32 v0, v17, v24
	v_cndmask_b32_e64 v0, v87, v0, s[6:7]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v2, v0, s[8:11], 0 offen
	v_add_u32_e32 v0, v18, v24
	v_cndmask_b32_e64 v0, v87, v0, s[6:7]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v3, v0, s[8:11], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_256x160x256_2560x6656x3328
		.amdhsa_group_segment_fixed_size 106496
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 80
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
		.amdhsa_next_free_vgpr 184
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 184
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
	.size	wave_mxfp4_static_gemm_256x160x256_2560x6656x3328, .Lfunc_end0-wave_mxfp4_static_gemm_256x160x256_2560x6656x3328

	.set wave_mxfp4_static_gemm_256x160x256_2560x6656x3328.num_vgpr, 184
	.set wave_mxfp4_static_gemm_256x160x256_2560x6656x3328.num_agpr, 0
	.set wave_mxfp4_static_gemm_256x160x256_2560x6656x3328.numbered_sgpr, 63
	.set wave_mxfp4_static_gemm_256x160x256_2560x6656x3328.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_256x160x256_2560x6656x3328.private_seg_size, 0
	.set wave_mxfp4_static_gemm_256x160x256_2560x6656x3328.uses_vcc, 1
	.set wave_mxfp4_static_gemm_256x160x256_2560x6656x3328.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_256x160x256_2560x6656x3328.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_256x160x256_2560x6656x3328.has_recursion, 0
	.set wave_mxfp4_static_gemm_256x160x256_2560x6656x3328.has_indirect_call, 0
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
    .group_segment_fixed_size: 106496
    .kernarg_segment_align: 8
    .kernarg_segment_size: 80
    .max_flat_workgroup_size: 512
    .name:           wave_mxfp4_static_gemm_256x160x256_2560x6656x3328
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     69
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_256x160x256_2560x6656x3328.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     184
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
