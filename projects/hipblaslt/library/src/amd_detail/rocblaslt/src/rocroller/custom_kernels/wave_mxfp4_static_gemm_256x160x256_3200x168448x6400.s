; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_256x160x256_3200x168448x6400
	.p2align	8
	.type	wave_mxfp4_static_gemm_256x160x256_3200x168448x6400,@function
wave_mxfp4_static_gemm_256x160x256_3200x168448x6400:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v84, 0x3ff, v0
	v_bfe_u32 v3, v0, 10, 10
	v_lshrrev_b32_e32 v4, 6, v84
	v_lshlrev_b32_e32 v0, 5, v3
	v_lshl_or_b32 v1, v4, 3, v0
	s_mov_b64 s[24:25], s[2:3]
	v_readfirstlane_b32 s2, v1
	v_lshrrev_b32_e32 v1, 3, v84
	s_lshl_b32 s33, s16, 8
	v_or3_b32 v2, v1, v0, s33
	v_xor_b32_e32 v1, v1, v84
	v_lshlrev_b32_e32 v1, 4, v1
	v_and_b32_e32 v85, 0x70, v1
	v_mul_u32_u24_e32 v86, 0xc80, v2
	s_and_b32 s3, s25, 0xffff
	s_lshl_b32 s43, s2, 7
	s_or_b32 s25, s3, 0x4c800000
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_or_b32_e32 v1, v86, v85
	s_mov_b32 m0, s43
	s_or_b32 s44, s43, 0x2000
	v_bfe_u32 v9, v84, 2, 3
	buffer_load_dwordx4 v1, s[24:27], 0 offen lds
	v_add_u32_e32 v1, 0x32000, v1
	s_mov_b32 m0, s44
	v_or_b32_e32 v5, 0x80, v2
	v_lshrrev_b32_e32 v7, 5, v84
	v_lshrrev_b32_e32 v8, 2, v84
	v_and_b32_e32 v10, 31, v84
	v_lshlrev_b32_e32 v9, 2, v9
	s_movk_i32 s42, 0xc80
	buffer_load_dwordx4 v1, s[24:27], 0 offen lds
	v_mul_u32_u24_e32 v1, 0xc80, v5
	v_bitop3_b32 v13, v8, v7, 7 bitop3:0x6c
	v_sub_u32_e32 v9, v10, v9
	v_or_b32_e32 v6, v1, v85
	v_bfrev_b32_e32 v87, -2
	v_cmp_gt_u32_e64 s[12:13], s42, v5
	s_or_b32 s45, s43, 0x4000
	v_lshl_add_u32 v9, v13, 2, v9
	v_cndmask_b32_e64 v5, v87, v6, s[12:13]
	s_mov_b32 m0, s45
	v_ashrrev_i32_e32 v10, 31, v9
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_or_b32_e32 v5, 0xc0, v2
	v_xor_b32_e32 v9, v10, v9
	v_mul_u32_u24_e32 v2, 0xc80, v5
	v_ashrrev_i32_e32 v11, 31, v9
	v_or_b32_e32 v6, v2, v85
	v_cmp_gt_u32_e64 s[2:3], s42, v5
	s_or_b32 s46, s43, 0x6000
	v_lshrrev_b32_e32 v11, 29, v11
	v_cndmask_b32_e64 v5, v87, v6, s[2:3]
	s_mov_b32 m0, s46
	v_add_u32_e32 v9, v9, v11
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_lshlrev_b32_e32 v5, 3, v3
	v_ashrrev_i32_e32 v9, 3, v9
	v_lshrrev_b32_e32 v16, 1, v13
	v_lshl_or_b32 v5, v4, 1, v5
	v_xor_b32_e32 v15, v9, v10
	v_and_b32_e32 v10, 0xfc, v84
	v_lshlrev_b32_e32 v11, 7, v16
	s_mov_b64 s[20:21], s[6:7]
	v_readfirstlane_b32 s6, v5
	v_lshlrev_b32_e32 v5, 6, v13
	v_lshlrev_b32_e32 v6, 2, v7
	v_add_u32_e32 v12, v10, v11
	v_add3_u32 v14, v0, v84, v5
	v_lshlrev_b32_e32 v9, 7, v15
	v_sub_u32_e32 v17, v6, v12
	v_add3_u32 v14, v17, v14, v9
	v_ashrrev_i32_e32 v17, 31, v14
	v_xor_b32_e32 v14, v17, v14
	s_mov_b32 s47, 0x51eb851f
	v_mul_hi_i32 v14, v14, s47
	v_lshrrev_b32_e32 v18, 31, v14
	v_ashrrev_i32_e32 v14, 8, v14
	v_add_u32_e32 v14, v14, v18
	v_xor_b32_e32 v17, v14, v17
	v_sub_u32_e32 v14, v7, v8
	v_lshlrev_b32_e32 v19, 4, v14
	v_lshlrev_b32_e32 v88, 2, v84
	v_lshlrev_b32_e32 v14, 9, v15
	v_lshlrev_b32_e32 v15, 8, v13
	v_add3_u32 v19, v19, v88, v15
	v_lshlrev_b32_e32 v13, 9, v16
	s_mul_i32 s41, s17, 0xa0
	v_sub_u32_e32 v16, v19, v13
	v_add_u32_e32 v18, s41, v17
	v_add_u32_e32 v19, v16, v14
	v_mul_i32_i24_e32 v17, 0xfffff380, v17
	v_lshlrev_b32_e32 v16, 7, v3
	v_add3_u32 v17, v19, v16, v17
	s_and_b32 s7, s21, 0xffff
	s_lshl_b32 s48, s6, 7
	s_or_b32 s21, s7, 0x4c800000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_mad_i32_i24 v17, v18, s42, v17
	s_add_i32 m0, s48, 0x10000
	s_or_b32 s49, s48, 0x800
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0xc800, v17
	s_add_i32 m0, s49, 0x10000
	s_or_b32 s50, s48, 0x1000
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x19000, v17
	s_add_i32 m0, s50, 0x10000
	s_or_b32 s51, s48, 0x1800
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x25800, v17
	s_add_i32 m0, s51, 0x10000
	s_or_b32 s52, s48, 0x2000
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x32000, v17
	s_add_i32 m0, s52, 0x10000
	s_or_b32 s53, s48, 0x2800
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x3e800, v17
	s_add_i32 m0, s53, 0x10000
	s_or_b32 s54, s48, 0x3000
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x4b000, v17
	s_add_i32 m0, s54, 0x10000
	s_or_b32 s55, s48, 0x3800
	buffer_load_dword v18, s[20:23], 0 offen lds
	s_add_i32 m0, s55, 0x10000
	v_add_u32_e32 v18, 0x57800, v17
	s_cmpk_lt_u32 s17, 0x41c
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x64000, v17
	s_cselect_b64 s[16:17], -1, 0
	s_or_b32 s56, s48, 0x4000
	v_cndmask_b32_e64 v18, v87, v18, s[16:17]
	s_add_i32 m0, s56, 0x10000
	v_add_u32_e32 v17, 0x70800, v17
	s_or_b32 s57, s48, 0x4800
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_cndmask_b32_e64 v17, v87, v17, s[16:17]
	s_add_i32 m0, s57, 0x10000
	v_lshrrev_b32_e32 v19, 4, v84
	v_bfe_u32 v83, v84, 4, 2
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v17, 4, v83
	v_mad_i32_i24 v21, v19, -16, v84
	v_add_u32_e32 v18, v21, v17
	s_mul_i32 s15, s15, s33
	s_mul_hi_u32 s6, s14, s33
	v_ashrrev_i32_e32 v20, 31, v18
	s_add_i32 s6, s6, s15
	s_mul_i32 s7, s14, s33
	v_xor_b32_e32 v18, v20, v18
	s_add_u32 s28, s4, s7
	v_mul_hi_i32 v18, v18, s47
	s_addc_u32 s4, s5, s6
	s_and_b32 s5, s14, 0x3fff
	v_lshrrev_b32_e32 v22, 31, v18
	v_ashrrev_i32_e32 v18, 4, v18
	s_bitset1_b32 s5, 14
	v_add_u32_e32 v18, v18, v22
	v_and_b32_e32 v27, 0xc0, v84
	s_and_b32 s4, s4, 0xffff
	s_lshl_b32 s5, s5, 16
	v_xad_u32 v22, v18, v20, v27
	v_and_b32_e32 v18, 62, v84
	s_or_b32 s29, s4, s5
	s_movk_i32 s4, 0x50
	v_mov_b32_e32 v20, 0xffffff38
	v_cmp_lt_u32_e32 vcc, 49, v18
	v_mad_u32_u24 v80, v3, s4, v21
	v_lshlrev_b32_e32 v18, 6, v83
	v_cndmask_b32_e32 v24, 0, v20, vcc
	v_mul_i32_i24_e32 v20, 0xffffffc0, v19
	v_ashrrev_i16_e32 v21, 15, v80
	v_add_u32_e32 v23, s33, v22
	v_add3_u32 v26, v20, v88, v18
	v_mul_lo_u32 v22, v22, s14
	v_lshrrev_b16_e32 v21, 11, v21
	v_add3_u32 v22, v26, v24, v22
	v_cmp_gt_i32_e32 vcc, s42, v23
	s_movk_i32 s5, 0xc60
	v_add_u16_e32 v21, v80, v21
	v_cndmask_b32_e32 v24, v87, v22, vcc
	v_lshl_add_u32 v22, s14, 5, v22
	v_cmp_gt_i32_e32 vcc, s5, v23
	v_and_b32_e32 v21, 0xffffffe0, v21
	s_mov_b32 s30, s26
	s_mov_b32 s31, s27
	v_cndmask_b32_e32 v22, v87, v22, vcc
	v_sub_u16_e32 v21, v80, v21
	buffer_load_dword v92, v24, s[28:31], 0 offen
	buffer_load_dword v91, v22, s[28:31], 0 offen
	v_bfe_i32 v22, v21, 0, 16
	v_ashrrev_i32_e32 v23, 31, v22
	v_add_u16_e32 v24, 32, v21
	v_cmp_gt_i16_e32 vcc, 0, v21
	s_load_dwordx2 s[18:19], s[0:1], 0x40
	s_movk_i32 s58, 0xc8
	v_cndmask_b32_e32 v21, v22, v24, vcc
	v_cndmask_b32_e64 v22, v23, 0, vcc
	v_xor_b32_e32 v21, v22, v21
	v_lshrrev_b32_e32 v23, 28, v21
	v_add_u32_e32 v21, v21, v23
	v_ashrrev_i32_e32 v21, 4, v21
	v_xor_b32_e32 v21, v21, v22
	v_add_u32_e32 v23, v26, v21
	v_ashrrev_i32_e32 v22, 31, v23
	v_xor_b32_e32 v24, v22, v23
	v_mul_hi_i32 v24, v24, s47
	v_lshrrev_b32_e32 v25, 31, v24
	v_ashrrev_i32_e32 v24, 6, v24
	v_add_u32_e32 v24, v24, v25
	v_ashrrev_i32_e32 v25, 31, v80
	v_xor_b32_e32 v28, v25, v80
	v_ashrrev_i32_e32 v29, 31, v28
	v_lshrrev_b32_e32 v29, 27, v29
	v_add_u32_e32 v28, v28, v29
	v_lshrrev_b32_e32 v28, 5, v28
	v_xor_b32_e32 v25, v28, v25
	v_lshlrev_b32_e32 v89, 5, v25
	v_xad_u32 v24, v24, v22, v89
	v_mul_hi_i32 v22, v23, s47
	v_lshrrev_b32_e32 v25, 31, v22
	v_ashrrev_i32_e32 v22, 6, v22
	v_add_u32_e32 v22, v22, v25
	v_mul_lo_u32 v22, v22, s58
	v_sub_u32_e32 v22, v23, v22
	v_add_u32_e32 v25, 0xc8, v22
	v_cmp_gt_i32_e32 vcc, 0, v22
	v_add_u32_e32 v32, s41, v24
	s_mov_b32 s15, 0x29200
	v_cndmask_b32_e32 v22, v22, v25, vcc
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s4, s19, s41
	s_mul_hi_u32 s6, s18, s41
	s_add_i32 s4, s6, s4
	v_mad_u64_u32 v[24:25], s[6:7], s18, v24, v[22:23]
	v_cmp_gt_i32_e32 vcc, s15, v32
	v_add_u32_e32 v22, 2, v23
	v_sub_u32_e32 v25, -3, v23
	v_cndmask_b32_e32 v33, v87, v24, vcc
	v_cmp_gt_i32_e32 vcc, -2, v23
	s_movk_i32 s22, 0xff38
	v_add_u32_e32 v81, 16, v80
	v_cndmask_b32_e32 v23, v22, v25, vcc
	v_mul_hi_i32 v23, v23, s47
	v_lshrrev_b32_e32 v25, 31, v23
	v_ashrrev_i32_e32 v23, 6, v23
	v_add_u32_e32 v23, v23, v25
	v_cndmask_b32_e64 v25, 0, -1, vcc
	v_xor_b32_e32 v23, v23, v25
	v_add_u32_e32 v25, v23, v89
	v_add_u32_e32 v34, s41, v25
	v_mul_lo_u32 v23, v23, s22
	v_mul_lo_u32 v25, s18, v25
	v_add3_u32 v25, v22, v23, v25
	v_cmp_gt_i32_e32 vcc, s15, v34
	v_sub_u32_e32 v23, 0xffef, v80
	v_mad_u32_u24 v28, v3, 5, v88
	v_cndmask_b32_e32 v35, v87, v25, vcc
	v_cmp_gt_i32_e32 vcc, -16, v80
	s_mul_i32 s19, s18, s41
	s_add_u32 s36, s8, s19
	v_cndmask_b32_e32 v23, v81, v23, vcc
	v_ashrrev_i16_e32 v29, 15, v23
	v_lshrrev_b16_e32 v29, 11, v29
	v_add_u16_e32 v23, v23, v29
	v_ashrrev_i16_e32 v23, 5, v23
	v_cndmask_b32_e64 v29, 0, -1, vcc
	v_xor_b32_e32 v23, v23, v29
	v_bfe_i32 v36, v23, 0, 16
	v_mad_i32_i24 v23, v36, -2, v20
	v_add3_u32 v37, v23, v28, v18
	v_add_u32_e32 v28, 1, v37
	v_sub_u32_e32 v29, -2, v37
	v_cmp_gt_i32_e32 vcc, -1, v37
	s_addc_u32 s4, s9, s4
	s_and_b32 s6, s18, 0x3fff
	v_cndmask_b32_e32 v28, v28, v29, vcc
	v_mul_hi_i32 v28, v28, s47
	v_lshrrev_b32_e32 v29, 31, v28
	v_ashrrev_i32_e32 v28, 6, v28
	v_mul_u32_u24_e32 v22, 5, v3
	v_add_u32_e32 v28, v28, v29
	v_cndmask_b32_e64 v29, 0, -1, vcc
	s_bitset1_b32 s6, 14
	v_xor_b32_e32 v38, v28, v29
	v_lshlrev_b32_e32 v90, 5, v36
	v_mad_i32_i24 v28, v36, -2, v22
	s_and_b32 s4, s4, 0xffff
	s_lshl_b32 s6, s6, 16
	v_add_u32_e32 v30, v38, v90
	v_add3_u32 v26, v28, v26, 1
	s_or_b32 s37, s4, s6
	v_add_u32_e32 v39, s41, v30
	v_mad_u64_u32 v[28:29], s[6:7], v38, s22, v[26:27]
	v_mad_u64_u32 v[30:31], s[6:7], v30, s18, v[28:29]
	v_cmp_gt_i32_e32 vcc, s15, v39
	v_sub_u32_e32 v31, -4, v37
	s_mov_b32 s8, 0x291e0
	v_cndmask_b32_e32 v29, v87, v30, vcc
	v_add_u32_e32 v30, 3, v37
	v_cmp_gt_i32_e32 vcc, -3, v37
	s_lshl_b32 s4, s18, 5
	v_add_u32_e32 v40, s4, v25
	v_cndmask_b32_e32 v30, v30, v31, vcc
	v_mul_hi_i32 v30, v30, s47
	v_lshrrev_b32_e32 v31, 31, v30
	v_ashrrev_i32_e32 v30, 6, v30
	v_add_u32_e32 v30, v30, v31
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_xor_b32_e32 v37, v30, v31
	v_add_u32_e32 v30, v37, v90
	v_mul_lo_u32 v31, v37, s22
	v_add_u32_e32 v39, s41, v30
	v_add3_u32 v26, v26, v31, 2
	v_mad_u64_u32 v[30:31], s[6:7], v30, s18, v[26:27]
	v_cmp_gt_i32_e32 vcc, s15, v39
	v_add_u32_e32 v31, s4, v24
	s_movk_i32 s6, 0xffd0
	v_cndmask_b32_e32 v30, v87, v30, vcc
	v_cmp_gt_i32_e32 vcc, s8, v32
	v_add_u32_e32 v82, 48, v80
	v_sub_u32_e32 v24, 0xffcf, v80
	v_cndmask_b32_e32 v39, v87, v31, vcc
	v_cmp_gt_i32_e32 vcc, s8, v34
	s_mov_b32 s38, s26
	s_mov_b32 s39, s27
	v_cndmask_b32_e32 v41, v87, v40, vcc
	v_cmp_gt_i32_e32 vcc, s6, v80
	s_mov_b32 s59, 0x291c0
	s_mov_b32 s9, 0
	v_cndmask_b32_e32 v24, v82, v24, vcc
	v_ashrrev_i16_e32 v25, 15, v24
	v_lshrrev_b16_e32 v25, 11, v25
	v_add_u16_e32 v24, v24, v25
	v_ashrrev_i16_e32 v24, 5, v24
	v_cndmask_b32_e64 v25, 0, -1, vcc
	v_xor_b32_e32 v24, v24, v25
	v_mov_b32_e32 v25, 5
	v_lshlrev_b32_sdwa v93, v25, sext(v24) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_e32 v24, v93, v38
	v_add_u32_e32 v38, s41, v24
	v_mad_u64_u32 v[24:25], s[6:7], s18, v24, v[28:29]
	v_cmp_gt_i32_e32 vcc, s15, v38
	s_movk_i32 s60, 0xf380
	s_mov_b32 s19, 0x10000
	v_cndmask_b32_e32 v28, v87, v24, vcc
	v_add_u32_e32 v24, v93, v37
	v_add_u32_e32 v37, s41, v24
	v_mad_u64_u32 v[24:25], s[6:7], s18, v24, v[26:27]
	v_cmp_gt_i32_e32 vcc, s15, v37
	v_add_u32_e32 v25, s4, v40
	v_cmp_eq_u32_e64 s[6:7], 0, v3
	v_cndmask_b32_e32 v24, v87, v24, vcc
	buffer_load_ubyte v125, v33, s[36:39], 0 offen
	buffer_load_ubyte v120, v35, s[36:39], 0 offen
	buffer_load_ubyte v124, v29, s[36:39], 0 offen
	buffer_load_ubyte v119, v30, s[36:39], 0 offen
	buffer_load_ubyte v123, v39, s[36:39], 0 offen
	buffer_load_ubyte v118, v41, s[36:39], 0 offen
	buffer_load_ubyte v122, v28, s[36:39], 0 offen
	buffer_load_ubyte v117, v24, s[36:39], 0 offen
	v_add_u32_e32 v24, s4, v31
	v_cmp_gt_i32_e32 vcc, s59, v32
	s_movk_i32 s22, 0x2800
	v_readfirstlane_b32 s4, v27
	v_cndmask_b32_e32 v24, v87, v24, vcc
	v_cmp_gt_i32_e32 vcc, s59, v34
	s_movk_i32 s61, 0xffc0
	s_nop 0
	v_cndmask_b32_e32 v25, v87, v25, vcc
	buffer_load_ubyte v126, v24, s[36:39], 0 offen
	buffer_load_ubyte v121, v25, s[36:39], 0 offen
	v_mul_i32_i24_e32 v24, -16, v19
	v_mul_i32_i24_e32 v25, -2, v36
	s_and_b64 vcc, exec, s[6:7]
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	v_lshlrev_b32_e32 v28, 7, v84
	v_lshlrev_b32_e32 v19, 11, v19
	v_and_b32_e32 v26, 7, v84
	v_sub_u32_e32 v19, v28, v19
	v_mul_lo_u32 v3, v3, s22
	v_bitop3_b32 v27, v83, v84, 7 bitop3:0x78
	v_lshl_add_u32 v4, v4, 13, v19
	v_add_u32_e32 v3, v19, v3
	v_bitop3_b32 v19, v83, v26, 4 bitop3:0x36
	v_lshlrev_b32_e32 v27, 4, v27
	v_lshlrev_b32_e32 v19, 4, v19
	v_or_b32_e32 v96, v3, v27
	v_or_b32_e32 v97, v19, v3
	v_add_u32_e32 v3, s41, v90
	v_or_b32_e32 v94, v4, v27
	v_or_b32_e32 v95, v19, v4
	v_add_u32_e32 v4, s41, v93
	v_sub_u32_e32 v99, 0x29200, v3
	v_sub_u32_e32 v3, 0, v20
	v_sub_u32_e32 v100, 0x29200, v4
	v_sub_u32_e32 v4, v3, v21
	v_sub_u32_e32 v3, v3, v25
	v_sub_u32_e32 v3, v3, v88
	v_sub_u32_e32 v4, v4, v88
	v_sub_u32_e32 v3, v3, v22
	v_sub_u32_e32 v101, v4, v18
	v_add_u32_e32 v4, v20, v18
	v_sub_u32_e32 v105, v3, v18
	v_add3_u32 v3, v14, v15, v16
	v_add_u32_e32 v102, v4, v21
	v_add_u32_e32 v108, 56, v4
	v_lshl_add_u32 v3, v7, 4, v3
	v_lshlrev_b32_e32 v4, 4, v8
	v_sub_u32_e32 v3, v3, v4
	v_sub_u32_e32 v112, v3, v13
	v_sub_u32_e32 v3, v12, v6
	v_sub_u32_e32 v3, v3, v0
	v_sub_u32_e32 v3, v3, v5
	s_load_dwordx2 s[34:35], s[0:1], 0x48
	v_sub_u32_e32 v113, v3, v9
	v_add_u32_e32 v3, v9, v5
	v_add3_u32 v0, v3, v0, v6
	v_add3_u32 v19, v25, v20, v88
	v_sub_u32_e32 v0, v0, v10
	v_add3_u32 v19, v19, v22, v18
	v_add_u32_e32 v110, v24, v17
	v_sub_u32_e32 v0, v0, v11
	v_mov_b32_e32 v12, 0
	s_or_b32 s40, s4, s33
	v_add_u32_e32 v98, s41, v89
	v_add3_u32 v103, v23, v18, v22
	v_sub_u32_e32 v104, 0, v19
	v_add_u32_e32 v106, 32, v89
	v_add_u32_e32 v107, 64, v89
	s_add_i32 s0, s4, 32
	v_sub_u32_e32 v109, 0, v84
	v_sub_u32_e32 v111, 0, v110
	v_add_u32_e32 v114, 0x200, v0
	v_add_u32_e32 v115, 0x80, v2
	v_add_u32_e32 v116, 0x80, v1
	s_mov_b32 s70, 0x8000
	s_mov_b32 s71, 0x15000
	s_mov_b32 s1, 0
	s_movk_i32 s62, 0xfdff
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	s_mov_b32 s30, s26
	s_mov_b32 s31, s27
	s_movk_i32 s63, 0xff00
	s_movk_i32 s64, 0xfeff
	s_movk_i32 s65, 0xffc8
	s_mov_b32 s38, s26
	s_mov_b32 s39, s27
	s_movk_i32 s66, 0xfefd
	s_movk_i32 s67, 0xfefe
	s_movk_i32 s68, 0xffc7
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
	v_mov_b32_e32 v127, v112
.LBB0_3:
	s_mov_b32 s69, s9
	s_mov_b32 s9, s70
	s_mov_b32 s70, s19
	s_mov_b32 s19, s71
	v_add_u32_e32 v128, v86, v85
	v_add_u32_e32 v129, 0x80, v128
	s_add_i32 m0, s9, s43
	s_barrier
	buffer_load_dwordx4 v129, s[24:27], 0 offen lds
	v_add_u32_e32 v128, 0x32080, v128
	s_add_i32 m0, s9, s44
	v_add3_u32 v129, v109, v113, s62
	buffer_load_dwordx4 v128, s[24:27], 0 offen lds
	v_add_u32_e32 v128, v85, v116
	v_cndmask_b32_e64 v128, v87, v128, s[12:13]
	s_add_i32 m0, s9, s45
	s_nop 0
	buffer_load_dwordx4 v128, s[24:27], 0 offen lds
	v_add_u32_e32 v128, v85, v115
	v_cndmask_b32_e64 v128, v87, v128, s[2:3]
	s_add_i32 m0, s9, s46
	s_nop 0
	buffer_load_dwordx4 v128, s[24:27], 0 offen lds
	v_add_u32_e32 v128, v84, v114
	v_cmp_gt_i32_e32 vcc, 0, v128
	s_add_i32 m0, s71, s48
	s_nop 0
	v_cndmask_b32_e32 v129, v128, v129, vcc
	v_mul_hi_i32 v129, v129, s47
	v_lshrrev_b32_e32 v130, 31, v129
	v_ashrrev_i32_e32 v129, 8, v129
	v_add_u32_e32 v129, v129, v130
	v_ashrrev_i32_e32 v128, 31, v128
	v_xor_b32_e32 v128, v129, v128
	v_add_u32_e32 v129, s41, v128
	v_mul_i32_i24_e32 v129, 0xc80, v129
	v_mad_i32_i24 v128, v128, s60, v129
	v_add3_u32 v129, v127, v88, v128
	v_add_u32_e32 v130, 0x800, v129
	buffer_load_dword v130, s[20:23], 0 offen lds
	v_add_u32_e32 v130, 0xd000, v129
	s_add_i32 m0, s71, s49
	v_add3_u32 v128, v88, v112, v128
	buffer_load_dword v130, s[20:23], 0 offen lds
	v_add_u32_e32 v130, 0x19800, v129
	s_add_i32 m0, s71, s50
	s_nop 0
	buffer_load_dword v130, s[20:23], 0 offen lds
	v_add_u32_e32 v130, 0x26000, v129
	s_add_i32 m0, s71, s51
	s_nop 0
	buffer_load_dword v130, s[20:23], 0 offen lds
	v_add_u32_e32 v130, 0x32800, v129
	s_add_i32 m0, s71, s52
	s_nop 0
	buffer_load_dword v130, s[20:23], 0 offen lds
	v_add_u32_e32 v130, 0x3f000, v129
	s_add_i32 m0, s71, s53
	s_nop 0
	buffer_load_dword v130, s[20:23], 0 offen lds
	v_add_u32_e32 v130, 0x4b800, v129
	s_add_i32 m0, s71, s54
	v_add_u32_e32 v129, 0x58000, v129
	buffer_load_dword v130, s[20:23], 0 offen lds
	s_add_i32 m0, s71, s55
	s_nop 0
	buffer_load_dword v129, s[20:23], 0 offen lds
	v_add_u32_e32 v129, 0x64800, v128
	v_cndmask_b32_e64 v129, v87, v129, s[16:17]
	s_add_i32 m0, s71, s56
	v_add_u32_e32 v128, 0x71000, v128
	buffer_load_dword v129, s[20:23], 0 offen lds
	v_cndmask_b32_e64 v128, v87, v128, s[16:17]
	s_add_i32 m0, s71, s57
	s_nop 0
	buffer_load_dword v128, s[20:23], 0 offen lds
	v_add_u32_e32 v128, v84, v110
	v_add_u32_e32 v130, v109, v111
	v_add_u32_e32 v129, 64, v128
	v_add_u32_e32 v131, 0xffffffbf, v130
	v_cmp_gt_i32_e32 vcc, s61, v128
	v_add_u32_e32 v133, 14, v128
	v_add_u32_e32 v130, -15, v130
	v_cndmask_b32_e32 v129, v129, v131, vcc
	v_mul_hi_i32 v129, v129, s47
	v_lshrrev_b32_e32 v131, 31, v129
	v_ashrrev_i32_e32 v129, 4, v129
	v_add_u32_e32 v129, v129, v131
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_cmp_gt_i32_e32 vcc, -14, v128
	v_xor_b32_e32 v129, v129, v131
	v_add_u32_e32 v132, s4, v129
	v_cndmask_b32_e32 v128, v133, v130, vcc
	v_mul_hi_i32 v128, v128, s47
	v_lshrrev_b32_e32 v130, 31, v128
	v_ashrrev_i32_e32 v128, 4, v128
	v_add_u32_e32 v128, v128, v130
	v_cndmask_b32_e64 v130, 0, -1, vcc
	v_xor_b32_e32 v128, v128, v130
	v_add_u32_e32 v131, s40, v129
	v_mul_lo_u32 v130, v132, s14
	v_mul_lo_u32 v128, v128, s58
	v_add_u32_e32 v129, s0, v129
	v_sub_u32_e32 v130, v130, v128
	v_add_u32_e32 v132, v88, v108
	v_mul_lo_u32 v129, s14, v129
	v_add_u32_e32 v130, v132, v130
	v_cmp_gt_i32_e32 vcc, s42, v131
	v_sub_u32_e32 v128, v129, v128
	v_add_u32_e32 v128, v132, v128
	v_cndmask_b32_e32 v130, v87, v130, vcc
	v_cmp_gt_i32_e32 vcc, s5, v131
	v_add_u32_e32 v132, s1, v101
	v_add_u32_e32 v133, 0xfffffeff, v132
	v_cndmask_b32_e32 v128, v87, v128, vcc
	buffer_load_dword v129, v130, s[28:31], 0 offen
	s_nop 0
	buffer_load_dword v128, v128, s[28:31], 0 offen
	v_add_u32_e32 v130, v88, v102
	v_add_u32_e32 v131, 0x100, v130
	v_cmp_gt_i32_e32 vcc, s63, v130
	v_add_u32_e32 v140, 56, v130
	v_subrev_u32_e32 v136, 57, v132
	v_cndmask_b32_e32 v133, v131, v133, vcc
	v_mul_hi_i32 v133, v133, s47
	v_lshrrev_b32_e32 v134, 31, v133
	v_ashrrev_i32_e32 v133, 6, v133
	v_add_u32_e32 v133, v133, v134
	v_cndmask_b32_e64 v134, 0, -1, vcc
	v_cmp_gt_i32_e32 vcc, s65, v130
	v_xor_b32_e32 v133, v133, v134
	v_add_u32_e32 v134, v133, v89
	v_cndmask_b32_e32 v136, v140, v136, vcc
	v_mul_hi_i32 v136, v136, s47
	v_lshrrev_b32_e32 v137, 31, v136
	v_ashrrev_i32_e32 v136, 6, v136
	v_add_u32_e32 v136, v136, v137
	v_cndmask_b32_e64 v137, 0, -1, vcc
	v_xor_b32_e32 v136, v136, v137
	v_mul_lo_u32 v134, v134, s18
	v_mul_lo_u32 v141, v136, s58
	v_add_u32_e32 v135, v133, v98
	v_sub_u32_e32 v134, v134, v141
	v_add_u32_e32 v134, v140, v134
	v_cmp_gt_i32_e32 vcc, s15, v135
	v_add_u32_e32 v142, 0x102, v130
	v_add_u32_e32 v132, 0xfffffefd, v132
	v_cndmask_b32_e32 v134, v87, v134, vcc
	v_cmp_gt_i32_e32 vcc, -2, v131
	v_add_u32_e32 v138, s1, v105
	v_subrev_u32_e32 v139, 58, v138
	v_cndmask_b32_e32 v131, v142, v132, vcc
	v_cmp_gt_i32_e32 vcc, s67, v130
	v_mul_hi_i32 v131, v131, s47
	v_lshrrev_b32_e32 v136, 31, v131
	v_cndmask_b32_e32 v130, v142, v132, vcc
	v_ashrrev_i32_e32 v131, 6, v131
	v_mul_hi_i32 v130, v130, s47
	v_add_u32_e32 v131, v131, v136
	v_ashrrev_i32_e32 v136, 31, v142
	v_lshrrev_b32_e32 v132, 31, v130
	v_ashrrev_i32_e32 v130, 6, v130
	v_xor_b32_e32 v143, v131, v136
	v_add_u32_e32 v130, v130, v132
	v_cndmask_b32_e64 v132, 0, -1, vcc
	v_add_u32_e32 v131, v143, v89
	v_xor_b32_e32 v130, v130, v132
	v_mul_lo_u32 v131, v131, s18
	v_mul_lo_u32 v145, v130, s58
	v_add_u32_e32 v144, v143, v98
	v_sub_u32_e32 v130, v131, v145
	v_add_u32_e32 v130, v142, v130
	v_cmp_gt_i32_e32 vcc, s15, v144
	v_add_u32_e32 v131, v88, v103
	v_add_u32_e32 v136, s1, v104
	v_cndmask_b32_e32 v130, v87, v130, vcc
	v_add_u32_e32 v132, 0x101, v131
	v_add_u32_e32 v136, 0xfffffefe, v136
	v_cmp_gt_i32_e32 vcc, s64, v131
	v_add_u32_e32 v138, 0xfffffefc, v138
	v_add_u32_e32 v151, v106, v143
	v_cndmask_b32_e32 v136, v132, v136, vcc
	v_mul_hi_i32 v136, v136, s47
	v_lshrrev_b32_e32 v137, 31, v136
	v_ashrrev_i32_e32 v136, 6, v136
	v_add_u32_e32 v136, v136, v137
	v_add_u32_e32 v137, 57, v131
	v_cmp_gt_i32_e32 vcc, s68, v131
	v_ashrrev_i32_e32 v132, 31, v132
	v_xor_b32_e32 v132, v136, v132
	v_cndmask_b32_e32 v139, v137, v139, vcc
	v_mul_hi_i32 v139, v139, s47
	v_lshrrev_b32_e32 v146, 31, v139
	v_ashrrev_i32_e32 v139, 6, v139
	v_add_u32_e32 v139, v139, v146
	v_ashrrev_i32_e32 v146, 31, v137
	v_add_u32_e32 v136, v132, v90
	v_xor_b32_e32 v139, v139, v146
	v_mul_lo_u32 v136, v136, s18
	v_mul_lo_u32 v139, v139, s58
	v_sub_u32_e32 v136, v136, v139
	v_add_u32_e32 v136, v137, v136
	v_cmp_lt_i32_e32 vcc, v132, v99
	v_add_u32_e32 v146, 0x103, v131
	v_mul_lo_u32 v151, s18, v151
	v_cndmask_b32_e32 v136, v87, v136, vcc
	v_cmp_gt_i32_e32 vcc, s66, v131
	v_add_u32_e32 v152, v132, v93
	v_sub_u32_e32 v151, v151, v145
	v_cndmask_b32_e32 v147, v146, v138, vcc
	v_mul_hi_i32 v147, v147, s47
	v_lshrrev_b32_e32 v148, 31, v147
	v_ashrrev_i32_e32 v147, 6, v147
	v_add_u32_e32 v147, v147, v148
	v_ashrrev_i32_e32 v148, 31, v146
	v_xor_b32_e32 v147, v147, v148
	v_add_u32_e32 v149, v147, v90
	v_mul_lo_u32 v149, v149, s18
	v_mul_lo_u32 v150, v147, s58
	v_sub_u32_e32 v149, v149, v150
	v_add_u32_e32 v149, v146, v149
	v_cmp_lt_i32_e32 vcc, v147, v99
	v_mul_lo_u32 v152, v152, s18
	v_add_u32_e32 v151, v142, v151
	v_cndmask_b32_e32 v147, v87, v149, vcc
	v_add_u32_e32 v149, v106, v133
	v_mul_lo_u32 v149, s18, v149
	v_sub_u32_e32 v149, v149, v141
	v_add_u32_e32 v149, v140, v149
	v_cmp_gt_i32_e32 vcc, s8, v135
	v_sub_u32_e32 v139, v152, v139
	v_add_u32_e32 v137, v137, v139
	v_cndmask_b32_e32 v149, v87, v149, vcc
	v_cmp_gt_i32_e32 vcc, s8, v144
	v_add_u32_e32 v131, 0x100, v131
	v_add_u32_e32 v133, v107, v133
	v_cndmask_b32_e32 v151, v87, v151, vcc
	v_cmp_lt_i32_e32 vcc, v132, v100
	v_mul_lo_u32 v133, s18, v133
	v_sub_u32_e32 v133, v133, v141
	v_cndmask_b32_e32 v152, v87, v137, vcc
	v_cmp_gt_i32_e32 vcc, -3, v131
	v_add_u32_e32 v133, v140, v133
	s_nop 0
	v_cndmask_b32_e32 v131, v146, v138, vcc
	v_mul_hi_i32 v131, v131, s47
	v_lshrrev_b32_e32 v132, 31, v131
	v_ashrrev_i32_e32 v131, 6, v131
	v_add_u32_e32 v131, v131, v132
	v_xor_b32_e32 v131, v131, v148
	v_add_u32_e32 v132, v131, v93
	v_mul_lo_u32 v132, v132, s18
	v_sub_u32_e32 v132, v132, v150
	v_add_u32_e32 v132, v146, v132
	v_cmp_lt_i32_e32 vcc, v131, v100
	s_nop 1
	v_cndmask_b32_e32 v146, v87, v132, vcc
	buffer_load_ubyte v139, v134, s[36:39], 0 offen
	buffer_load_ubyte v138, v130, s[36:39], 0 offen
	buffer_load_ubyte v137, v136, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v136, v147, s[36:39], 0 offen
	buffer_load_ubyte v134, v149, s[36:39], 0 offen
	buffer_load_ubyte v132, v151, s[36:39], 0 offen
	buffer_load_ubyte v131, v152, s[36:39], 0 offen
	buffer_load_ubyte v130, v146, s[36:39], 0 offen
	v_cmp_gt_i32_e32 vcc, s59, v135
	v_add_u32_e32 v135, v107, v143
	v_mul_lo_u32 v135, s18, v135
	v_sub_u32_e32 v135, v135, v145
	v_cndmask_b32_e32 v133, v87, v133, vcc
	v_add_u32_e32 v135, v142, v135
	v_cmp_gt_i32_e32 vcc, s59, v144
	s_nop 1
	v_cndmask_b32_e32 v140, v87, v135, vcc
	buffer_load_ubyte v135, v133, s[36:39], 0 offen
	s_nop 0
	buffer_load_ubyte v133, v140, s[36:39], 0 offen
	v_add_u32_e32 v140, s69, v94
	ds_read_b128 v[142:145], v140
	ds_read_b128 v[146:149], v140 offset:2048
	ds_read_b128 v[150:153], v140 offset:4096
	ds_read_b128 v[154:157], v140 offset:6144
	v_add_u32_e32 v140, s70, v96
	ds_read_b128 v[158:161], v140
	ds_read_b128 v[162:165], v140 offset:2048
	ds_read_b128 v[166:169], v140 offset:4096
	ds_read_b128 v[170:173], v140 offset:6144
	ds_read_b128 v[174:177], v140 offset:8192
	s_barrier
	s_setprio 1
	v_and_b32_e32 v125, 0xff, v125
	v_and_b32_e32 v124, 0xff, v124
	v_and_b32_e32 v123, 0xff, v123
	v_and_b32_e32 v122, 0xff, v122
	v_and_b32_e32 v126, 0xff, v126
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[142:145], v[158:161], v[12:15], v92, v125 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[142:145], v[162:165], v[76:79], v92, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[142:145], v[166:169], v[72:75], v92, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[142:145], v[170:173], v[68:71], v92, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[142:145], v[174:177], v[64:67], v92, v126 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[146:149], v[158:161], v[60:63], v92, v125 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[146:149], v[162:165], v[56:59], v92, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[146:149], v[166:169], v[52:55], v92, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[146:149], v[170:173], v[48:51], v92, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[146:149], v[174:177], v[44:47], v92, v126 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[150:153], v[158:161], v[40:43], v91, v125 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[150:153], v[162:165], v[28:31], v91, v124 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[150:153], v[166:169], v[16:19], v91, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[150:153], v[170:173], v[20:23], v91, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[150:153], v[174:177], v[24:27], v91, v126 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[154:157], v[158:161], v[32:35], v91, v125 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[154:157], v[162:165], v[36:39], v91, v124 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[154:157], v[166:169], v[8:11], v91, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[154:157], v[170:173], v[4:7], v91, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[154:157], v[174:177], v[0:3], v91, v126 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v122, s69, v95
	ds_read_b128 v[140:143], v122
	ds_read_b128 v[144:147], v122 offset:2048
	ds_read_b128 v[148:151], v122 offset:4096
	ds_read_b128 v[152:155], v122 offset:6144
	v_add_u32_e32 v122, s70, v97
	ds_read_b128 v[156:159], v122
	ds_read_b128 v[160:163], v122 offset:2048
	ds_read_b128 v[164:167], v122 offset:4096
	ds_read_b128 v[168:171], v122 offset:6144
	ds_read_b128 v[122:125], v122 offset:8192
	s_waitcnt vmcnt(12)
	s_barrier
	s_setprio 1
	v_and_b32_e32 v120, 0xff, v120
	v_and_b32_e32 v119, 0xff, v119
	v_and_b32_e32 v118, 0xff, v118
	v_and_b32_e32 v117, 0xff, v117
	v_and_b32_e32 v121, 0xff, v121
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[140:143], v[156:159], v[12:15], v92, v120 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[140:143], v[160:163], v[76:79], v92, v119 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[140:143], v[164:167], v[72:75], v92, v118 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[140:143], v[168:171], v[68:71], v92, v117 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[140:143], v[122:125], v[64:67], v92, v121 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[144:147], v[156:159], v[60:63], v92, v120 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[144:147], v[160:163], v[56:59], v92, v119 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[144:147], v[164:167], v[52:55], v92, v118 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[144:147], v[168:171], v[48:51], v92, v117 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[144:147], v[122:125], v[44:47], v92, v121 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[148:151], v[156:159], v[40:43], v91, v120 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[148:151], v[160:163], v[28:31], v91, v119 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[148:151], v[164:167], v[16:19], v91, v118 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[148:151], v[168:171], v[20:23], v91, v117 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[148:151], v[122:125], v[24:27], v91, v121 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[152:155], v[156:159], v[32:35], v91, v120 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[152:155], v[160:163], v[36:39], v91, v119 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[152:155], v[164:167], v[8:11], v91, v118 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[152:155], v[168:171], v[4:7], v91, v117 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[152:155], v[122:125], v[0:3], v91, v121 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_addk_i32 s1, 0xff00
	v_add_u32_e32 v102, 0x100, v102
	v_add_u32_e32 v103, 0x100, v103
	v_add_u32_e32 v108, 0x100, v108
	v_subrev_u32_e32 v111, 64, v111
	v_add_u32_e32 v110, 64, v110
	v_add_u32_e32 v127, 0x800, v127
	v_add_u32_e32 v112, 0x800, v112
	v_add_u32_e32 v113, 0xfffffe00, v113
	v_add_u32_e32 v114, 0x200, v114
	v_add_u32_e32 v86, 0x80, v86
	v_add_u32_e32 v115, 0x80, v115
	v_add_u32_e32 v116, 0x80, v116
	s_cmpk_lg_i32 s1, 0xe800
	s_mov_b32 s71, s70
	s_mov_b32 s70, s69
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v121, v133
	v_mov_b32_e32 v126, v135
	v_mov_b32_e32 v117, v130
	v_mov_b32_e32 v122, v131
	v_mov_b32_e32 v118, v132
	v_mov_b32_e32 v123, v134
	v_mov_b32_e32 v119, v136
	v_mov_b32_e32 v124, v137
	v_mov_b32_e32 v120, v138
	v_mov_b32_e32 v125, v139
	v_mov_b32_e32 v91, v128
	v_mov_b32_e32 v92, v129
	s_cbranch_scc1 .LBB0_3
	s_andn2_b64 vcc, exec, s[6:7]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	v_add_u32_e32 v84, s19, v96
	v_add_u32_e32 v85, s19, v97
	s_barrier
	ds_read_b128 v[110:113], v84
	ds_read_b128 v[114:117], v84 offset:2048
	ds_read_b128 v[118:121], v85
	ds_read_b128 v[122:125], v84 offset:8192
	ds_read_b128 v[140:143], v85 offset:2048
	ds_read_b128 v[144:147], v85 offset:4096
	ds_read_b128 v[148:151], v84 offset:4096
	ds_read_b128 v[152:155], v84 offset:6144
	ds_read_b128 v[156:159], v85 offset:6144
	ds_read_b128 v[160:163], v85 offset:8192
	v_add_u32_e32 v84, s9, v94
	ds_read_b128 v[96:99], v84
	ds_read_b128 v[106:109], v84 offset:2048
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[96:99], v[110:113], v[12:15], v129, v139 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v85, s9, v95
	ds_read_b128 v[100:103], v85
	ds_read_b128 v[164:167], v85 offset:2048
	ds_read_b128 v[168:171], v84 offset:4096
	ds_read_b128 v[172:175], v84 offset:6144
	ds_read_b128 v[176:179], v85 offset:4096
	ds_read_b128 v[180:183], v85 offset:6144
	s_movk_i32 s0, 0x7fff
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[100:103], v[118:121], v[12:15], v129, v138 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mov_b32_e32 v84, 0x7fc0
	s_cmpk_lt_u32 s40, 0xc80
	s_mul_i32 s2, s35, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[96:99], v[148:151], v[72:75], v129, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mul_hi_u32 s3, s34, s33
	s_nop 2
	v_bfe_u32 v85, v15, 16, 1
	v_bfe_u32 v86, v14, 16, 1
	v_add3_u32 v85, v15, v85, s0
	v_bfe_u32 v87, v13, 16, 1
	v_bfe_u32 v88, v12, 16, 1
	v_add3_u32 v86, v14, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cmp_o_f32_e32 vcc, v15, v15
	v_add3_u32 v90, v12, v88, s0
	v_add3_u32 v91, v13, v87, s0
	v_lshrrev_b32_e32 v94, 16, v86
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], v[100:103], v[144:147], v[72:75], v129, v132 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_cselect_b64 s[14:15], -1, 0
	s_add_i32 s3, s3, s2
	s_mul_i32 s2, s34, s33
	v_cndmask_b32_e32 v72, v84, v85, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_lshrrev_b32_e32 v73, 16, v91
	v_lshrrev_b32_e32 v74, 16, v90
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[96:99], v[152:155], v[68:71], v129, v131 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_lshl_b64 s[2:3], s[2:3], 1
	s_mov_b32 s19, 0x27000
	s_mov_b32 s18, 0x7ffffffd
	v_cndmask_b32_e32 v68, v84, v94, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[96:99], v[114:117], v[76:79], v129, v137 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v69, v84, v73, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[96:99], v[122:125], v[64:67], v129, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v70, v84, v74, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[100:103], v[160:163], v[12:15], v129, v133 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[106:109], v[110:113], v[60:63], v129, v139 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[100:103], v[140:143], v[76:79], v129, v136 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], v[164:167], v[118:121], v[12:15], v129, v138 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[106:109], v[114:117], v[56:59], v129, v137 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 4
	v_bfe_u32 v71, v79, 16, 1
	v_bfe_u32 v73, v78, 16, 1
	v_add3_u32 v60, v79, v71, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], v[100:103], v[156:159], v[90:93], v129, v130 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v74, v77, 16, 1
	v_add3_u32 v73, v78, v73, s0
	v_lshrrev_b32_e32 v60, 16, v60
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], v[164:167], v[140:143], v[12:15], v129, v136 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v79, v79
	v_bfe_u32 v75, v76, 16, 1
	v_add3_u32 v74, v77, v74, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[106:109], v[148:151], v[52:55], v129, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v61, 16, v73
	v_cndmask_b32_e32 v60, v84, v60, vcc
	v_cmp_o_f32_e32 vcc, v78, v78
	v_add3_u32 v75, v76, v75, s0
	v_lshrrev_b32_e32 v62, 16, v74
	v_cndmask_b32_e32 v56, v84, v61, vcc
	v_cmp_o_f32_e32 vcc, v77, v77
	v_lshrrev_b32_e32 v63, 16, v75
	v_bfe_u32 v58, v89, 16, 1
	v_cndmask_b32_e32 v57, v84, v62, vcc
	v_cmp_o_f32_e32 vcc, v76, v76
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[164:167], v[144:147], v[12:15], v129, v132 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v52, v88, 16, 1
	v_cndmask_b32_e32 v59, v84, v63, vcc
	v_bfe_u32 v53, v87, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[106:109], v[152:155], v[48:51], v129, v131 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v54, v86, 16, 1
	v_add3_u32 v52, v88, v52, s0
	v_cmp_o_f32_e32 vcc, v89, v89
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], v[164:167], v[156:159], v[12:15], v129, v130 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v48, v89, v58, s0
	v_lshrrev_b32_e32 v48, 16, v48
	v_add3_u32 v54, v86, v54, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[106:109], v[122:125], v[44:47], v129, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v53, v87, v53, s0
	v_lshrrev_b32_e32 v49, 16, v52
	v_cndmask_b32_e32 v52, v84, v48, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[164:167], v[160:163], v[12:15], v129, v133 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v88, v88
	v_lshrrev_b32_e32 v50, 16, v53
	v_lshrrev_b32_e32 v51, 16, v54
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[110:113], v[40:43], v128, v139 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v54, v84, v49, vcc
	v_cmp_o_f32_e32 vcc, v87, v87
	v_bfe_u32 v44, v93, 16, 1
	v_bfe_u32 v40, v92, 16, 1
	v_cndmask_b32_e32 v55, v84, v50, vcc
	v_cmp_o_f32_e32 vcc, v86, v86
	v_bfe_u32 v41, v91, 16, 1
	v_bfe_u32 v42, v90, 16, 1
	v_cndmask_b32_e32 v58, v84, v51, vcc
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[176:179], v[118:121], v[12:15], v128, v138 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v42, v90, v42, s0
	v_add3_u32 v41, v91, v41, s0
	v_add3_u32 v40, v92, v40, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[114:117], v[28:31], v128, v137 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v61, 16, v40
	v_lshrrev_b32_e32 v62, 16, v41
	v_lshrrev_b32_e32 v63, 16, v42
	v_add3_u32 v28, v93, v44, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[176:179], v[140:143], v[12:15], v128, v136 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v28, 16, v28
	v_cmp_o_f32_e32 vcc, v93, v93
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[148:151], v[16:19], v128, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v53, v84, v28, vcc
	v_cmp_o_f32_e32 vcc, v92, v92
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[176:179], v[144:147], v[12:15], v128, v132 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v78, v84, v61, vcc
	v_cmp_o_f32_e32 vcc, v91, v91
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[152:155], v[20:23], v128, v131 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v79, v84, v62, vcc
	v_cmp_o_f32_e32 vcc, v90, v90
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[176:179], v[156:159], v[12:15], v128, v130 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v85, v84, v63, vcc
	v_cmp_o_f32_e32 vcc, v67, v67
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[122:125], v[24:27], v128, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_bfe_u32 v24, v67, 16, 1
	v_bfe_u32 v25, v66, 16, 1
	v_add3_u32 v24, v67, v24, s0
	v_bfe_u32 v26, v65, 16, 1
	v_add3_u32 v25, v66, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_bfe_u32 v27, v64, 16, 1
	v_add3_u32 v26, v65, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_cndmask_b32_e32 v86, v84, v24, vcc
	v_cmp_o_f32_e32 vcc, v66, v66
	v_add3_u32 v27, v64, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v87, v84, v25, vcc
	v_cmp_o_f32_e32 vcc, v65, v65
	v_bfe_u32 v24, v97, 16, 1
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v88, v84, v26, vcc
	v_cmp_o_f32_e32 vcc, v64, v64
	v_bfe_u32 v25, v96, 16, 1
	v_add3_u32 v24, v97, v24, s0
	v_cndmask_b32_e32 v89, v84, v27, vcc
	v_bfe_u32 v26, v95, 16, 1
	v_add3_u32 v25, v96, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v97, v97
	v_bfe_u32 v27, v94, 16, 1
	v_add3_u32 v26, v95, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_cndmask_b32_e32 v66, v84, v24, vcc
	v_cmp_o_f32_e32 vcc, v96, v96
	v_add3_u32 v27, v94, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v90, v84, v25, vcc
	v_cmp_o_f32_e32 vcc, v95, v95
	v_bfe_u32 v24, v101, 16, 1
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v91, v84, v26, vcc
	v_cmp_o_f32_e32 vcc, v94, v94
	v_bfe_u32 v25, v100, 16, 1
	v_add3_u32 v24, v101, v24, s0
	v_cndmask_b32_e32 v92, v84, v27, vcc
	v_bfe_u32 v26, v99, 16, 1
	v_add3_u32 v25, v100, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v101, v101
	v_bfe_u32 v27, v98, 16, 1
	v_add3_u32 v26, v99, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_cndmask_b32_e32 v64, v84, v24, vcc
	v_cmp_o_f32_e32 vcc, v100, v100
	v_add3_u32 v27, v98, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v73, v84, v25, vcc
	v_cmp_o_f32_e32 vcc, v99, v99
	v_bfe_u32 v24, v77, 16, 1
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v93, v84, v26, vcc
	v_cmp_o_f32_e32 vcc, v98, v98
	v_bfe_u32 v25, v76, 16, 1
	v_add3_u32 v24, v77, v24, s0
	v_cndmask_b32_e32 v94, v84, v27, vcc
	v_bfe_u32 v26, v75, 16, 1
	v_add3_u32 v25, v76, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v77, v77
	v_bfe_u32 v27, v74, 16, 1
	v_add3_u32 v26, v75, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_cndmask_b32_e32 v62, v84, v24, vcc
	v_cmp_o_f32_e32 vcc, v76, v76
	v_add3_u32 v27, v74, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v67, v84, v25, vcc
	v_cmp_o_f32_e32 vcc, v75, v75
	v_bfe_u32 v24, v105, 16, 1
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v75, v84, v26, vcc
	v_cmp_o_f32_e32 vcc, v74, v74
	v_bfe_u32 v25, v104, 16, 1
	v_add3_u32 v24, v105, v24, s0
	v_cndmask_b32_e32 v76, v84, v27, vcc
	v_bfe_u32 v26, v103, 16, 1
	v_add3_u32 v25, v104, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v105, v105
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[176:179], v[160:163], v[12:15], v128, v133 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v27, v102, 16, 1
	v_add3_u32 v26, v103, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[172:175], v[110:113], v[32:35], v128, v139 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v61, v84, v24, vcc
	v_cmp_o_f32_e32 vcc, v104, v104
	v_add3_u32 v27, v102, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v65, v84, v25, vcc
	v_cmp_o_f32_e32 vcc, v103, v103
	v_bfe_u32 v24, v109, 16, 1
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v74, v84, v26, vcc
	v_cmp_o_f32_e32 vcc, v102, v102
	v_bfe_u32 v25, v108, 16, 1
	v_add3_u32 v24, v109, v24, s0
	v_cndmask_b32_e32 v77, v84, v27, vcc
	v_bfe_u32 v26, v107, 16, 1
	v_add3_u32 v25, v108, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v109, v109
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[180:183], v[118:121], v[12:15], v128, v138 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v27, v106, 16, 1
	v_add3_u32 v26, v107, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[172:175], v[114:117], v[36:39], v128, v137 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v27, v106, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v39, v84, v24, vcc
	v_cmp_o_f32_e32 vcc, v108, v108
	v_bfe_u32 v24, v51, 16, 1
	v_add3_u32 v24, v51, v24, s0
	v_cndmask_b32_e32 v63, v84, v25, vcc
	v_cmp_o_f32_e32 vcc, v107, v107
	v_bfe_u32 v25, v50, 16, 1
	v_add3_u32 v25, v50, v25, s0
	v_cndmask_b32_e32 v71, v84, v26, vcc
	v_cmp_o_f32_e32 vcc, v106, v106
	v_bfe_u32 v26, v49, 16, 1
	v_lshrrev_b32_e32 v24, 16, v24
	v_cndmask_b32_e32 v95, v84, v27, vcc
	v_cmp_o_f32_e32 vcc, v51, v51
	v_bfe_u32 v27, v48, 16, 1
	v_add3_u32 v26, v49, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_cndmask_b32_e32 v32, v84, v24, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_add3_u32 v27, v48, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v36, v84, v25, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_bfe_u32 v24, v47, 16, 1
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v49, v84, v26, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_bfe_u32 v25, v46, 16, 1
	v_bfe_u32 v26, v45, 16, 1
	v_add3_u32 v24, v47, v24, s0
	v_cndmask_b32_e32 v48, v84, v27, vcc
	v_add3_u32 v26, v45, v26, s0
	v_add3_u32 v25, v46, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v47, v47
	v_bfe_u32 v27, v44, 16, 1
	v_lshrrev_b32_e32 v25, 16, v25
	v_lshrrev_b32_e32 v33, 16, v26
	v_cndmask_b32_e32 v26, v84, v24, vcc
	v_cmp_o_f32_e32 vcc, v46, v46
	v_add3_u32 v27, v44, v27, s0
	v_bfe_u32 v24, v43, 16, 1
	v_cndmask_b32_e32 v34, v84, v25, vcc
	v_cmp_o_f32_e32 vcc, v45, v45
	v_lshrrev_b32_e32 v27, 16, v27
	v_bfe_u32 v25, v42, 16, 1
	v_cndmask_b32_e32 v37, v84, v33, vcc
	v_cmp_o_f32_e32 vcc, v44, v44
	v_add3_u32 v24, v43, v24, s0
	v_bfe_u32 v33, v40, 16, 1
	v_cndmask_b32_e32 v44, v84, v27, vcc
	v_bfe_u32 v27, v41, 16, 1
	v_add3_u32 v25, v42, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v43, v43
	v_add3_u32 v33, v40, v33, s0
	v_add3_u32 v27, v41, v27, s0
	v_lshrrev_b32_e32 v35, 16, v25
	v_cndmask_b32_e32 v25, v84, v24, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	v_lshrrev_b32_e32 v27, 16, v27
	v_lshrrev_b32_e32 v38, 16, v33
	v_cndmask_b32_e32 v33, v84, v35, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	v_bfe_u32 v24, v31, 16, 1
	v_add3_u32 v24, v31, v24, s0
	v_cndmask_b32_e32 v35, v84, v27, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_bfe_u32 v27, v30, 16, 1
	v_bfe_u32 v40, v29, 16, 1
	v_cndmask_b32_e32 v38, v84, v38, vcc
	v_add3_u32 v27, v30, v27, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v31, v31
	v_add3_u32 v40, v29, v40, s0
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v24, v84, v24, vcc
	v_cmp_o_f32_e32 vcc, v30, v30
	v_bfe_u32 v41, v28, 16, 1
	v_lshrrev_b32_e32 v40, 16, v40
	v_cndmask_b32_e32 v27, v84, v27, vcc
	v_cmp_o_f32_e32 vcc, v29, v29
	v_add3_u32 v41, v28, v41, s0
	v_lshrrev_b32_e32 v41, 16, v41
	v_cndmask_b32_e32 v29, v84, v40, vcc
	v_cmp_o_f32_e32 vcc, v28, v28
	v_bfe_u32 v28, v23, 16, 1
	v_bfe_u32 v31, v22, 16, 1
	v_add3_u32 v28, v23, v28, s0
	v_cndmask_b32_e32 v30, v84, v41, vcc
	v_bfe_u32 v40, v21, 16, 1
	v_add3_u32 v31, v22, v31, s0
	v_lshrrev_b32_e32 v28, 16, v28
	v_cmp_o_f32_e32 vcc, v23, v23
	v_add3_u32 v40, v21, v40, s0
	v_lshrrev_b32_e32 v31, 16, v31
	v_cndmask_b32_e32 v23, v84, v28, vcc
	v_cmp_o_f32_e32 vcc, v22, v22
	v_bfe_u32 v41, v20, 16, 1
	v_lshrrev_b32_e32 v40, 16, v40
	v_cndmask_b32_e32 v22, v84, v31, vcc
	v_cmp_o_f32_e32 vcc, v21, v21
	v_add3_u32 v41, v20, v41, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[180:183], v[140:143], v[12:15], v128, v136 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v21, v84, v40, vcc
	v_cmp_o_f32_e32 vcc, v20, v20
	v_bfe_u32 v20, v19, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[172:175], v[148:151], v[8:11], v128, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v41, 16, v41
	v_bfe_u32 v31, v18, 16, 1
	v_add3_u32 v20, v19, v20, s0
	v_cndmask_b32_e32 v28, v84, v41, vcc
	v_bfe_u32 v40, v17, 16, 1
	v_add3_u32 v31, v18, v31, s0
	v_lshrrev_b32_e32 v20, 16, v20
	v_cmp_o_f32_e32 vcc, v19, v19
	v_add3_u32 v40, v17, v40, s0
	v_lshrrev_b32_e32 v31, 16, v31
	v_cndmask_b32_e32 v19, v84, v20, vcc
	v_cmp_o_f32_e32 vcc, v18, v18
	v_bfe_u32 v41, v16, 16, 1
	v_lshrrev_b32_e32 v40, 16, v40
	v_cndmask_b32_e32 v18, v84, v31, vcc
	v_cmp_o_f32_e32 vcc, v17, v17
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[180:183], v[144:147], v[8:11], v128, v132 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v41, v16, v41, s0
	v_cndmask_b32_e32 v17, v84, v40, vcc
	v_cmp_o_f32_e32 vcc, v16, v16
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[172:175], v[152:155], v[4:7], v128, v131 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v16, v15, 16, 1
	v_lshrrev_b32_e32 v41, 16, v41
	v_bfe_u32 v31, v14, 16, 1
	v_add3_u32 v16, v15, v16, s0
	v_cndmask_b32_e32 v20, v84, v41, vcc
	v_bfe_u32 v40, v13, 16, 1
	v_add3_u32 v31, v14, v31, s0
	v_lshrrev_b32_e32 v16, 16, v16
	v_cmp_o_f32_e32 vcc, v15, v15
	v_bfe_u32 v41, v12, 16, 1
	v_add3_u32 v40, v13, v40, s0
	v_lshrrev_b32_e32 v31, 16, v31
	v_cndmask_b32_e32 v15, v84, v16, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[180:183], v[156:159], v[4:7], v128, v130 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v41, v12, v41, s0
	v_lshrrev_b32_e32 v40, 16, v40
	v_cndmask_b32_e32 v14, v84, v31, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[172:175], v[122:125], v[0:3], v128, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v13, v13
	v_bfe_u32 v16, v11, 16, 1
	v_lshrrev_b32_e32 v41, 16, v41
	v_cndmask_b32_e32 v13, v84, v40, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v31, v10, 16, 1
	v_add3_u32 v16, v11, v16, s0
	v_cndmask_b32_e32 v12, v84, v41, vcc
	v_bfe_u32 v40, v9, 16, 1
	v_add3_u32 v31, v10, v31, s0
	v_lshrrev_b32_e32 v16, 16, v16
	v_cmp_o_f32_e32 vcc, v11, v11
	v_bfe_u32 v41, v8, 16, 1
	v_add3_u32 v40, v9, v40, s0
	v_lshrrev_b32_e32 v31, 16, v31
	v_cndmask_b32_e32 v11, v84, v16, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[180:183], v[160:163], v[0:3], v128, v133 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v41, v8, v41, s0
	v_lshrrev_b32_e32 v40, 16, v40
	v_cndmask_b32_e32 v10, v84, v31, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_bfe_u32 v16, v7, 16, 1
	v_lshrrev_b32_e32 v41, 16, v41
	v_cndmask_b32_e32 v9, v84, v40, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v31, v6, 16, 1
	v_add3_u32 v16, v7, v16, s0
	v_cndmask_b32_e32 v8, v84, v41, vcc
	v_bfe_u32 v40, v5, 16, 1
	v_add3_u32 v31, v6, v31, s0
	v_lshrrev_b32_e32 v16, 16, v16
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v41, v4, 16, 1
	v_add3_u32 v40, v5, v40, s0
	v_lshrrev_b32_e32 v31, 16, v31
	v_cndmask_b32_e32 v7, v84, v16, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v41, v4, v41, s0
	v_lshrrev_b32_e32 v40, 16, v40
	v_cndmask_b32_e32 v6, v84, v31, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_bfe_u32 v16, v3, 16, 1
	v_lshrrev_b32_e32 v41, 16, v41
	v_cndmask_b32_e32 v5, v84, v40, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v31, v2, 16, 1
	v_add3_u32 v16, v3, v16, s0
	v_cndmask_b32_e32 v4, v84, v41, vcc
	v_bfe_u32 v40, v1, 16, 1
	v_add3_u32 v31, v2, v31, s0
	v_lshrrev_b32_e32 v16, 16, v16
	v_cmp_o_f32_e32 vcc, v3, v3
	v_bfe_u32 v41, v0, 16, 1
	v_add3_u32 v40, v1, v40, s0
	v_lshrrev_b32_e32 v31, 16, v31
	v_cndmask_b32_e32 v3, v84, v16, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	v_add3_u32 v41, v0, v41, s0
	v_lshrrev_b32_e32 v40, 16, v40
	v_cndmask_b32_e32 v2, v84, v31, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	v_lshrrev_b32_e32 v41, 16, v41
	v_add_u32_e32 v31, s41, v80
	v_cndmask_b32_e32 v1, v84, v40, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	s_mov_b32 s0, 0x29200
	v_lshlrev_b32_e32 v16, 2, v83
	v_cndmask_b32_e32 v0, v84, v41, vcc
	v_cmp_gt_i32_e32 vcc, s0, v31
	s_and_b64 s[0:1], vcc, s[14:15]
	s_add_u32 s2, s10, s2
	v_or_b32_e32 v41, s4, v16
	s_addc_u32 s3, s11, s3
	s_lshl_b32 s4, s41, 1
	s_add_u32 s16, s2, s4
	v_mul_lo_u32 v42, s34, v41
	s_addc_u32 s2, s3, 0
	s_and_b32 s3, s34, 0x3fff
	v_or_b32_e32 v40, s40, v16
	v_add_u32_e32 v43, v42, v80
	v_not_b32_e32 v16, -2.0
	s_lshl_b32 s3, s3, 16
	s_and_b32 s2, s2, 0xffff
	s_or_b32 s2, s3, s2
	v_cndmask_b32_e64 v43, v16, v43, s[0:1]
	s_or_b32 s17, s2, 2.0
	v_lshlrev_b32_e32 v43, 1, v43
	buffer_store_short v70, v43, s[16:19], 0 offen
	s_movk_i32 s0, 0xc7f
	v_or_b32_e32 v43, 1, v41
	v_cmp_gt_u32_e64 s[8:9], s0, v40
	v_mul_lo_u32 v43, s34, v43
	v_add_u32_e32 v45, v43, v80
	s_and_b64 s[0:1], vcc, s[8:9]
	v_cndmask_b32_e64 v45, v16, v45, s[0:1]
	v_lshlrev_b32_e32 v45, 1, v45
	buffer_store_short v69, v45, s[16:19], 0 offen
	s_movk_i32 s0, 0xc7e
	v_or_b32_e32 v45, 2, v41
	v_cmp_gt_u32_e64 s[10:11], s0, v40
	v_mul_lo_u32 v45, s34, v45
	v_add_u32_e32 v46, v45, v80
	s_and_b64 s[0:1], vcc, s[10:11]
	v_cndmask_b32_e64 v46, v16, v46, s[0:1]
	v_lshlrev_b32_e32 v46, 1, v46
	buffer_store_short v68, v46, s[16:19], 0 offen
	s_movk_i32 s0, 0xc7d
	v_or_b32_e32 v46, 3, v41
	v_cmp_gt_u32_e64 s[12:13], s0, v40
	v_mul_lo_u32 v46, s34, v46
	v_add_u32_e32 v47, v46, v80
	s_and_b64 s[0:1], vcc, s[12:13]
	v_cndmask_b32_e64 v47, v16, v47, s[0:1]
	s_mov_b32 s0, 0x291f0
	v_lshlrev_b32_e32 v47, 1, v47
	v_cmp_gt_i32_e64 s[0:1], s0, v31
	buffer_store_short v72, v47, s[16:19], 0 offen
	v_add_u32_e32 v47, v42, v81
	s_and_b64 s[2:3], s[0:1], s[14:15]
	v_cndmask_b32_e64 v47, v16, v47, s[2:3]
	v_lshlrev_b32_e32 v47, 1, v47
	buffer_store_short v59, v47, s[16:19], 0 offen
	v_add_u32_e32 v47, v43, v81
	s_and_b64 s[2:3], s[0:1], s[8:9]
	v_cndmask_b32_e64 v47, v16, v47, s[2:3]
	v_lshlrev_b32_e32 v47, 1, v47
	buffer_store_short v57, v47, s[16:19], 0 offen
	v_add_u32_e32 v47, v45, v81
	s_and_b64 s[2:3], s[0:1], s[10:11]
	v_cndmask_b32_e64 v47, v16, v47, s[2:3]
	v_lshlrev_b32_e32 v47, 1, v47
	buffer_store_short v56, v47, s[16:19], 0 offen
	v_add_u32_e32 v47, v46, v81
	s_and_b64 s[2:3], s[0:1], s[12:13]
	v_cndmask_b32_e64 v47, v16, v47, s[2:3]
	v_lshlrev_b32_e32 v47, 1, v47
	s_mov_b32 s2, 0x291e0
	buffer_store_short v60, v47, s[16:19], 0 offen
	v_cmp_gt_i32_e64 s[2:3], s2, v31
	v_add_u32_e32 v47, 32, v80
	v_add_u32_e32 v50, v42, v47
	s_and_b64 s[4:5], s[2:3], s[14:15]
	v_cndmask_b32_e64 v50, v16, v50, s[4:5]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v58, v50, s[16:19], 0 offen
	v_add_u32_e32 v50, v43, v47
	s_and_b64 s[4:5], s[2:3], s[8:9]
	v_cndmask_b32_e64 v50, v16, v50, s[4:5]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v55, v50, s[16:19], 0 offen
	v_add_u32_e32 v50, v45, v47
	s_and_b64 s[4:5], s[2:3], s[10:11]
	v_cndmask_b32_e64 v50, v16, v50, s[4:5]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v54, v50, s[16:19], 0 offen
	v_add_u32_e32 v50, v46, v47
	s_and_b64 s[4:5], s[2:3], s[12:13]
	v_cndmask_b32_e64 v50, v16, v50, s[4:5]
	s_mov_b32 s4, 0x291d0
	v_lshlrev_b32_e32 v50, 1, v50
	v_cmp_gt_i32_e64 s[4:5], s4, v31
	buffer_store_short v52, v50, s[16:19], 0 offen
	v_add_u32_e32 v50, v42, v82
	s_and_b64 s[6:7], s[4:5], s[14:15]
	v_cndmask_b32_e64 v50, v16, v50, s[6:7]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v85, v50, s[16:19], 0 offen
	v_add_u32_e32 v50, v43, v82
	s_and_b64 s[6:7], s[4:5], s[8:9]
	v_cndmask_b32_e64 v50, v16, v50, s[6:7]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v79, v50, s[16:19], 0 offen
	v_add_u32_e32 v50, v45, v82
	s_and_b64 s[6:7], s[4:5], s[10:11]
	v_cndmask_b32_e64 v50, v16, v50, s[6:7]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v78, v50, s[16:19], 0 offen
	v_add_u32_e32 v50, v46, v82
	s_and_b64 s[6:7], s[4:5], s[12:13]
	v_cndmask_b32_e64 v50, v16, v50, s[6:7]
	s_mov_b32 s6, 0x291c0
	v_cmp_gt_i32_e64 s[6:7], s6, v31
	v_add_u32_e32 v31, 64, v80
	v_add_u32_e32 v42, v42, v31
	s_and_b64 s[14:15], s[6:7], s[14:15]
	v_cndmask_b32_e64 v42, v16, v42, s[14:15]
	v_lshlrev_b32_e32 v50, 1, v50
	v_lshlrev_b32_e32 v42, 1, v42
	buffer_store_short v53, v50, s[16:19], 0 offen
	buffer_store_short v89, v42, s[16:19], 0 offen
	v_add_u32_e32 v42, v43, v31
	s_and_b64 s[8:9], s[6:7], s[8:9]
	v_cndmask_b32_e64 v42, v16, v42, s[8:9]
	v_lshlrev_b32_e32 v42, 1, v42
	buffer_store_short v88, v42, s[16:19], 0 offen
	v_add_u32_e32 v42, v45, v31
	s_and_b64 s[8:9], s[6:7], s[10:11]
	v_cndmask_b32_e64 v42, v16, v42, s[8:9]
	v_lshlrev_b32_e32 v42, 1, v42
	buffer_store_short v87, v42, s[16:19], 0 offen
	v_add_u32_e32 v42, v46, v31
	s_and_b64 s[8:9], s[6:7], s[12:13]
	v_cndmask_b32_e64 v42, v16, v42, s[8:9]
	v_lshlrev_b32_e32 v42, 1, v42
	buffer_store_short v86, v42, s[16:19], 0 offen
	s_cmpk_lt_u32 s40, 0xc70
	v_or_b32_e32 v42, 16, v41
	s_cselect_b64 s[20:21], -1, 0
	v_mul_lo_u32 v42, s34, v42
	v_add_u32_e32 v43, v42, v80
	s_and_b64 s[8:9], vcc, s[20:21]
	v_cndmask_b32_e64 v43, v16, v43, s[8:9]
	v_lshlrev_b32_e32 v43, 1, v43
	buffer_store_short v92, v43, s[16:19], 0 offen
	s_movk_i32 s8, 0xc6f
	v_or_b32_e32 v43, 17, v41
	v_cmp_gt_u32_e64 s[8:9], s8, v40
	v_mul_lo_u32 v43, s34, v43
	v_add_u32_e32 v45, v43, v80
	s_and_b64 s[10:11], vcc, s[8:9]
	v_cndmask_b32_e64 v45, v16, v45, s[10:11]
	v_lshlrev_b32_e32 v45, 1, v45
	buffer_store_short v91, v45, s[16:19], 0 offen
	s_movk_i32 s10, 0xc6e
	v_or_b32_e32 v45, 18, v41
	v_cmp_gt_u32_e64 s[10:11], s10, v40
	v_mul_lo_u32 v45, s34, v45
	v_add_u32_e32 v46, v45, v80
	s_and_b64 s[12:13], vcc, s[10:11]
	v_cndmask_b32_e64 v46, v16, v46, s[12:13]
	v_lshlrev_b32_e32 v46, 1, v46
	buffer_store_short v90, v46, s[16:19], 0 offen
	s_movk_i32 s12, 0xc6d
	v_or_b32_e32 v46, 19, v41
	v_cmp_gt_u32_e64 s[12:13], s12, v40
	v_mul_lo_u32 v46, s34, v46
	v_add_u32_e32 v50, v46, v80
	s_and_b64 s[14:15], vcc, s[12:13]
	v_cndmask_b32_e64 v50, v16, v50, s[14:15]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v66, v50, s[16:19], 0 offen
	v_add_u32_e32 v50, v42, v81
	s_and_b64 s[14:15], s[0:1], s[20:21]
	v_cndmask_b32_e64 v50, v16, v50, s[14:15]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v94, v50, s[16:19], 0 offen
	v_add_u32_e32 v50, v43, v81
	s_and_b64 s[14:15], s[0:1], s[8:9]
	v_cndmask_b32_e64 v50, v16, v50, s[14:15]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v93, v50, s[16:19], 0 offen
	v_add_u32_e32 v50, v45, v81
	s_and_b64 s[14:15], s[0:1], s[10:11]
	v_cndmask_b32_e64 v50, v16, v50, s[14:15]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v73, v50, s[16:19], 0 offen
	v_add_u32_e32 v50, v46, v81
	s_and_b64 s[14:15], s[0:1], s[12:13]
	v_cndmask_b32_e64 v50, v16, v50, s[14:15]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v64, v50, s[16:19], 0 offen
	v_add_u32_e32 v50, v42, v47
	s_and_b64 s[14:15], s[2:3], s[20:21]
	v_cndmask_b32_e64 v50, v16, v50, s[14:15]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v76, v50, s[16:19], 0 offen
	v_add_u32_e32 v50, v43, v47
	s_and_b64 s[14:15], s[2:3], s[8:9]
	v_cndmask_b32_e64 v50, v16, v50, s[14:15]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v75, v50, s[16:19], 0 offen
	v_add_u32_e32 v50, v45, v47
	s_and_b64 s[14:15], s[2:3], s[10:11]
	v_cndmask_b32_e64 v50, v16, v50, s[14:15]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v67, v50, s[16:19], 0 offen
	v_add_u32_e32 v50, v46, v47
	s_and_b64 s[14:15], s[2:3], s[12:13]
	v_cndmask_b32_e64 v50, v16, v50, s[14:15]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v62, v50, s[16:19], 0 offen
	v_add_u32_e32 v50, v42, v82
	s_and_b64 s[14:15], s[4:5], s[20:21]
	v_cndmask_b32_e64 v50, v16, v50, s[14:15]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v77, v50, s[16:19], 0 offen
	v_add_u32_e32 v50, v43, v82
	s_and_b64 s[14:15], s[4:5], s[8:9]
	v_cndmask_b32_e64 v50, v16, v50, s[14:15]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v74, v50, s[16:19], 0 offen
	v_add_u32_e32 v50, v45, v82
	s_and_b64 s[14:15], s[4:5], s[10:11]
	v_cndmask_b32_e64 v50, v16, v50, s[14:15]
	v_lshlrev_b32_e32 v50, 1, v50
	buffer_store_short v65, v50, s[16:19], 0 offen
	v_add_u32_e32 v50, v46, v82
	s_and_b64 s[14:15], s[4:5], s[12:13]
	v_cndmask_b32_e64 v50, v16, v50, s[14:15]
	v_add_u32_e32 v42, v42, v31
	s_and_b64 s[14:15], s[6:7], s[20:21]
	v_cndmask_b32_e64 v42, v16, v42, s[14:15]
	v_lshlrev_b32_e32 v50, 1, v50
	v_lshlrev_b32_e32 v42, 1, v42
	buffer_store_short v61, v50, s[16:19], 0 offen
	buffer_store_short v95, v42, s[16:19], 0 offen
	v_add_u32_e32 v42, v43, v31
	s_and_b64 s[8:9], s[6:7], s[8:9]
	v_cndmask_b32_e64 v42, v16, v42, s[8:9]
	v_lshlrev_b32_e32 v42, 1, v42
	buffer_store_short v71, v42, s[16:19], 0 offen
	v_add_u32_e32 v42, v45, v31
	s_and_b64 s[8:9], s[6:7], s[10:11]
	v_cndmask_b32_e64 v42, v16, v42, s[8:9]
	v_lshlrev_b32_e32 v42, 1, v42
	buffer_store_short v63, v42, s[16:19], 0 offen
	v_add_u32_e32 v42, v46, v31
	s_and_b64 s[8:9], s[6:7], s[12:13]
	v_cndmask_b32_e64 v42, v16, v42, s[8:9]
	v_lshlrev_b32_e32 v42, 1, v42
	buffer_store_short v39, v42, s[16:19], 0 offen
	s_cmpk_lt_u32 s40, 0xc60
	v_or_b32_e32 v39, 32, v41
	s_cselect_b64 s[20:21], -1, 0
	v_mul_lo_u32 v39, s34, v39
	v_add_u32_e32 v42, v39, v80
	s_and_b64 s[8:9], vcc, s[20:21]
	v_cndmask_b32_e64 v42, v16, v42, s[8:9]
	v_lshlrev_b32_e32 v42, 1, v42
	buffer_store_short v48, v42, s[16:19], 0 offen
	s_movk_i32 s8, 0xc5f
	v_or_b32_e32 v42, 33, v41
	v_cmp_gt_u32_e64 s[8:9], s8, v40
	v_mul_lo_u32 v42, s34, v42
	v_add_u32_e32 v43, v42, v80
	s_and_b64 s[10:11], vcc, s[8:9]
	v_cndmask_b32_e64 v43, v16, v43, s[10:11]
	v_lshlrev_b32_e32 v43, 1, v43
	buffer_store_short v49, v43, s[16:19], 0 offen
	s_movk_i32 s10, 0xc5e
	v_or_b32_e32 v43, 34, v41
	v_cmp_gt_u32_e64 s[10:11], s10, v40
	v_mul_lo_u32 v43, s34, v43
	v_add_u32_e32 v45, v43, v80
	s_and_b64 s[12:13], vcc, s[10:11]
	v_cndmask_b32_e64 v45, v16, v45, s[12:13]
	v_lshlrev_b32_e32 v45, 1, v45
	buffer_store_short v36, v45, s[16:19], 0 offen
	s_movk_i32 s12, 0xc5d
	v_or_b32_e32 v36, 35, v41
	v_cmp_gt_u32_e64 s[12:13], s12, v40
	v_mul_lo_u32 v36, s34, v36
	v_add_u32_e32 v45, v36, v80
	s_and_b64 s[14:15], vcc, s[12:13]
	v_cndmask_b32_e64 v45, v16, v45, s[14:15]
	v_lshlrev_b32_e32 v45, 1, v45
	buffer_store_short v32, v45, s[16:19], 0 offen
	v_add_u32_e32 v32, v39, v81
	s_and_b64 s[14:15], s[0:1], s[20:21]
	v_cndmask_b32_e64 v32, v16, v32, s[14:15]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v44, v32, s[16:19], 0 offen
	v_add_u32_e32 v32, v42, v81
	s_and_b64 s[14:15], s[0:1], s[8:9]
	v_cndmask_b32_e64 v32, v16, v32, s[14:15]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v37, v32, s[16:19], 0 offen
	v_add_u32_e32 v32, v43, v81
	s_and_b64 s[14:15], s[0:1], s[10:11]
	v_cndmask_b32_e64 v32, v16, v32, s[14:15]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v34, v32, s[16:19], 0 offen
	v_add_u32_e32 v32, v36, v81
	s_and_b64 s[14:15], s[0:1], s[12:13]
	v_cndmask_b32_e64 v32, v16, v32, s[14:15]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v26, v32, s[16:19], 0 offen
	v_add_u32_e32 v26, v39, v47
	s_and_b64 s[14:15], s[2:3], s[20:21]
	v_cndmask_b32_e64 v26, v16, v26, s[14:15]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v38, v26, s[16:19], 0 offen
	v_add_u32_e32 v26, v42, v47
	s_and_b64 s[14:15], s[2:3], s[8:9]
	v_cndmask_b32_e64 v26, v16, v26, s[14:15]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v35, v26, s[16:19], 0 offen
	v_add_u32_e32 v26, v43, v47
	s_and_b64 s[14:15], s[2:3], s[10:11]
	v_cndmask_b32_e64 v26, v16, v26, s[14:15]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v33, v26, s[16:19], 0 offen
	v_add_u32_e32 v26, v36, v47
	s_and_b64 s[14:15], s[2:3], s[12:13]
	v_cndmask_b32_e64 v26, v16, v26, s[14:15]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v25, v26, s[16:19], 0 offen
	v_add_u32_e32 v25, v39, v82
	s_and_b64 s[14:15], s[4:5], s[20:21]
	v_cndmask_b32_e64 v25, v16, v25, s[14:15]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v30, v25, s[16:19], 0 offen
	v_add_u32_e32 v25, v42, v82
	s_and_b64 s[14:15], s[4:5], s[8:9]
	v_cndmask_b32_e64 v25, v16, v25, s[14:15]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v29, v25, s[16:19], 0 offen
	v_add_u32_e32 v25, v43, v82
	s_and_b64 s[14:15], s[4:5], s[10:11]
	v_cndmask_b32_e64 v25, v16, v25, s[14:15]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v27, v25, s[16:19], 0 offen
	v_add_u32_e32 v25, v36, v82
	s_and_b64 s[14:15], s[4:5], s[12:13]
	v_cndmask_b32_e64 v25, v16, v25, s[14:15]
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v24, v25, s[16:19], 0 offen
	v_add_u32_e32 v24, v39, v31
	s_and_b64 s[14:15], s[6:7], s[20:21]
	v_cndmask_b32_e64 v24, v16, v24, s[14:15]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v28, v24, s[16:19], 0 offen
	v_add_u32_e32 v24, v42, v31
	s_and_b64 s[8:9], s[6:7], s[8:9]
	v_cndmask_b32_e64 v24, v16, v24, s[8:9]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v21, v24, s[16:19], 0 offen
	v_add_u32_e32 v21, v43, v31
	s_and_b64 s[8:9], s[6:7], s[10:11]
	v_cndmask_b32_e64 v21, v16, v21, s[8:9]
	v_lshlrev_b32_e32 v21, 1, v21
	buffer_store_short v22, v21, s[16:19], 0 offen
	v_add_u32_e32 v21, v36, v31
	s_and_b64 s[8:9], s[6:7], s[12:13]
	v_cndmask_b32_e64 v21, v16, v21, s[8:9]
	v_lshlrev_b32_e32 v21, 1, v21
	buffer_store_short v23, v21, s[16:19], 0 offen
	s_cmpk_lt_u32 s40, 0xc50
	v_or_b32_e32 v21, 48, v41
	s_cselect_b64 s[14:15], -1, 0
	v_mul_lo_u32 v21, s34, v21
	v_add_u32_e32 v22, v21, v80
	s_and_b64 s[8:9], vcc, s[14:15]
	v_cndmask_b32_e64 v22, v16, v22, s[8:9]
	v_lshlrev_b32_e32 v22, 1, v22
	buffer_store_short v20, v22, s[16:19], 0 offen
	s_movk_i32 s8, 0xc4f
	v_or_b32_e32 v20, 49, v41
	v_cmp_gt_u32_e64 s[8:9], s8, v40
	v_mul_lo_u32 v20, s34, v20
	v_add_u32_e32 v22, v20, v80
	s_and_b64 s[10:11], vcc, s[8:9]
	v_cndmask_b32_e64 v22, v16, v22, s[10:11]
	v_lshlrev_b32_e32 v22, 1, v22
	buffer_store_short v17, v22, s[16:19], 0 offen
	s_movk_i32 s10, 0xc4e
	v_or_b32_e32 v17, 50, v41
	v_cmp_gt_u32_e64 s[10:11], s10, v40
	v_mul_lo_u32 v17, s34, v17
	v_add_u32_e32 v22, v17, v80
	s_and_b64 s[12:13], vcc, s[10:11]
	v_cndmask_b32_e64 v22, v16, v22, s[12:13]
	v_lshlrev_b32_e32 v22, 1, v22
	buffer_store_short v18, v22, s[16:19], 0 offen
	s_movk_i32 s12, 0xc4d
	v_or_b32_e32 v18, 51, v41
	v_cmp_gt_u32_e64 s[12:13], s12, v40
	v_mul_lo_u32 v18, s34, v18
	v_add_u32_e32 v22, v18, v80
	s_and_b64 vcc, vcc, s[12:13]
	v_cndmask_b32_e32 v22, v16, v22, vcc
	v_lshlrev_b32_e32 v22, 1, v22
	buffer_store_short v19, v22, s[16:19], 0 offen
	v_add_u32_e32 v19, v21, v81
	s_and_b64 vcc, s[0:1], s[14:15]
	v_cndmask_b32_e32 v19, v16, v19, vcc
	v_lshlrev_b32_e32 v19, 1, v19
	buffer_store_short v12, v19, s[16:19], 0 offen
	v_add_u32_e32 v12, v20, v81
	s_and_b64 vcc, s[0:1], s[8:9]
	v_cndmask_b32_e32 v12, v16, v12, vcc
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v13, v12, s[16:19], 0 offen
	v_add_u32_e32 v12, v17, v81
	s_and_b64 vcc, s[0:1], s[10:11]
	v_cndmask_b32_e32 v12, v16, v12, vcc
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v14, v12, s[16:19], 0 offen
	v_add_u32_e32 v12, v18, v81
	s_and_b64 vcc, s[0:1], s[12:13]
	v_cndmask_b32_e32 v12, v16, v12, vcc
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v15, v12, s[16:19], 0 offen
	v_add_u32_e32 v12, v21, v47
	s_and_b64 vcc, s[2:3], s[14:15]
	v_cndmask_b32_e32 v12, v16, v12, vcc
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v8, v12, s[16:19], 0 offen
	v_add_u32_e32 v8, v20, v47
	s_and_b64 vcc, s[2:3], s[8:9]
	v_cndmask_b32_e32 v8, v16, v8, vcc
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v9, v8, s[16:19], 0 offen
	v_add_u32_e32 v8, v17, v47
	s_and_b64 vcc, s[2:3], s[10:11]
	v_cndmask_b32_e32 v8, v16, v8, vcc
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v10, v8, s[16:19], 0 offen
	v_add_u32_e32 v8, v18, v47
	s_and_b64 vcc, s[2:3], s[12:13]
	v_cndmask_b32_e32 v8, v16, v8, vcc
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v11, v8, s[16:19], 0 offen
	v_add_u32_e32 v8, v21, v82
	s_and_b64 vcc, s[4:5], s[14:15]
	v_cndmask_b32_e32 v8, v16, v8, vcc
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v4, v8, s[16:19], 0 offen
	v_add_u32_e32 v4, v20, v82
	s_and_b64 vcc, s[4:5], s[8:9]
	v_cndmask_b32_e32 v4, v16, v4, vcc
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v5, v4, s[16:19], 0 offen
	v_add_u32_e32 v4, v17, v82
	s_and_b64 vcc, s[4:5], s[10:11]
	v_cndmask_b32_e32 v4, v16, v4, vcc
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v6, v4, s[16:19], 0 offen
	v_add_u32_e32 v4, v18, v82
	s_and_b64 vcc, s[4:5], s[12:13]
	v_cndmask_b32_e32 v4, v16, v4, vcc
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v7, v4, s[16:19], 0 offen
	v_add_u32_e32 v4, v21, v31
	s_and_b64 vcc, s[6:7], s[14:15]
	v_cndmask_b32_e32 v4, v16, v4, vcc
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v0, v4, s[16:19], 0 offen
	v_add_u32_e32 v0, v20, v31
	s_and_b64 vcc, s[6:7], s[8:9]
	v_cndmask_b32_e32 v0, v16, v0, vcc
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v1, v0, s[16:19], 0 offen
	v_add_u32_e32 v0, v17, v31
	s_and_b64 vcc, s[6:7], s[10:11]
	v_cndmask_b32_e32 v0, v16, v0, vcc
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v2, v0, s[16:19], 0 offen
	v_add_u32_e32 v0, v18, v31
	s_and_b64 vcc, s[6:7], s[12:13]
	v_cndmask_b32_e32 v0, v16, v0, vcc
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v3, v0, s[16:19], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_256x160x256_3200x168448x6400
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
	.size	wave_mxfp4_static_gemm_256x160x256_3200x168448x6400, .Lfunc_end0-wave_mxfp4_static_gemm_256x160x256_3200x168448x6400

	.set wave_mxfp4_static_gemm_256x160x256_3200x168448x6400.num_vgpr, 184
	.set wave_mxfp4_static_gemm_256x160x256_3200x168448x6400.num_agpr, 0
	.set wave_mxfp4_static_gemm_256x160x256_3200x168448x6400.numbered_sgpr, 72
	.set wave_mxfp4_static_gemm_256x160x256_3200x168448x6400.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_256x160x256_3200x168448x6400.private_seg_size, 0
	.set wave_mxfp4_static_gemm_256x160x256_3200x168448x6400.uses_vcc, 1
	.set wave_mxfp4_static_gemm_256x160x256_3200x168448x6400.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_256x160x256_3200x168448x6400.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_256x160x256_3200x168448x6400.has_recursion, 0
	.set wave_mxfp4_static_gemm_256x160x256_3200x168448x6400.has_indirect_call, 0
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
    .name:           wave_mxfp4_static_gemm_256x160x256_3200x168448x6400
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     78
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_256x160x256_3200x168448x6400.kd
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
