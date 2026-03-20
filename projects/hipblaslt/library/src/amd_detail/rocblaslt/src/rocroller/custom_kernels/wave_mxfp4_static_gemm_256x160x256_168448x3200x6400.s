; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_256x160x256_168448x3200x6400
	.p2align	8
	.type	wave_mxfp4_static_gemm_256x160x256_168448x3200x6400,@function
wave_mxfp4_static_gemm_256x160x256_168448x3200x6400:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v83, 0x3ff, v0
	v_bfe_u32 v3, v0, 10, 10
	v_lshrrev_b32_e32 v4, 6, v83
	v_lshlrev_b32_e32 v0, 5, v3
	v_lshl_or_b32 v1, v4, 3, v0
	v_bfe_u32 v9, v83, 2, 3
	s_mov_b64 s[24:25], s[2:3]
	v_readfirstlane_b32 s2, v1
	v_lshrrev_b32_e32 v1, 3, v83
	s_lshl_b32 s28, s16, 8
	v_lshrrev_b32_e32 v7, 5, v83
	v_lshrrev_b32_e32 v8, 2, v83
	v_and_b32_e32 v10, 31, v83
	v_lshlrev_b32_e32 v9, 2, v9
	v_or3_b32 v2, v1, v0, s28
	v_xor_b32_e32 v1, v1, v83
	v_bitop3_b32 v13, v8, v7, 7 bitop3:0x6c
	v_sub_u32_e32 v9, v10, v9
	v_lshlrev_b32_e32 v1, 4, v1
	v_lshl_add_u32 v9, v13, 2, v9
	s_mov_b64 s[20:21], s[6:7]
	v_and_b32_e32 v1, 0x70, v1
	v_mul_u32_u24_e32 v2, 0xc80, v2
	s_and_b32 s6, s25, 0xffff
	s_lshl_b32 s30, s2, 7
	v_ashrrev_i32_e32 v10, 31, v9
	s_or_b32 s25, s6, 0x4c800000
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_or_b32_e32 v5, v2, v1
	s_mov_b32 m0, s30
	s_or_b32 s31, s30, 0x2000
	v_xor_b32_e32 v9, v10, v9
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_add_u32_e32 v6, 0x32000, v5
	s_mov_b32 m0, s31
	s_or_b32 s33, s30, 0x4000
	v_ashrrev_i32_e32 v11, 31, v9
	buffer_load_dwordx4 v6, s[24:27], 0 offen lds
	v_add_u32_e32 v6, 0x64000, v5
	s_mov_b32 m0, s33
	s_or_b32 s34, s30, 0x6000
	v_lshrrev_b32_e32 v11, 29, v11
	buffer_load_dwordx4 v6, s[24:27], 0 offen lds
	v_add_u32_e32 v5, 0x96000, v5
	s_mov_b32 m0, s34
	v_add_u32_e32 v9, v9, v11
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_lshlrev_b32_e32 v5, 3, v3
	v_ashrrev_i32_e32 v9, 3, v9
	v_lshrrev_b32_e32 v16, 1, v13
	v_lshl_or_b32 v5, v4, 1, v5
	v_xor_b32_e32 v15, v9, v10
	v_and_b32_e32 v10, 0xfc, v83
	v_lshlrev_b32_e32 v11, 7, v16
	v_readfirstlane_b32 s2, v5
	v_lshlrev_b32_e32 v5, 6, v13
	v_lshlrev_b32_e32 v6, 2, v7
	v_add_u32_e32 v12, v10, v11
	v_add3_u32 v14, v0, v83, v5
	v_lshlrev_b32_e32 v9, 7, v15
	v_sub_u32_e32 v17, v6, v12
	v_add3_u32 v14, v17, v14, v9
	v_ashrrev_i32_e32 v17, 31, v14
	v_xor_b32_e32 v14, v17, v14
	s_mov_b32 s35, 0x51eb851f
	v_mul_hi_i32 v14, v14, s35
	v_lshrrev_b32_e32 v18, 31, v14
	v_ashrrev_i32_e32 v14, 8, v14
	v_add_u32_e32 v14, v14, v18
	v_xor_b32_e32 v17, v14, v17
	v_sub_u32_e32 v14, v7, v8
	v_lshlrev_b32_e32 v19, 4, v14
	v_lshlrev_b32_e32 v85, 2, v83
	v_lshlrev_b32_e32 v14, 9, v15
	v_lshlrev_b32_e32 v15, 8, v13
	v_add3_u32 v19, v19, v85, v15
	v_lshlrev_b32_e32 v13, 9, v16
	s_mul_i32 s29, s17, 0xa0
	v_sub_u32_e32 v16, v19, v13
	v_add_u32_e32 v18, s29, v17
	v_add_u32_e32 v19, v16, v14
	v_mul_i32_i24_e32 v17, 0xfffff380, v17
	v_lshlrev_b32_e32 v16, 7, v3
	s_lshl_b32 s50, s2, 7
	s_movk_i32 s3, 0xc80
	v_add3_u32 v17, v19, v16, v17
	s_and_b32 s6, s21, 0xffff
	s_add_i32 s36, s50, 0x10000
	s_or_b32 s51, s50, 0x800
	s_or_b32 s21, s6, 0x4c800000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_mad_i32_i24 v17, v18, s3, v17
	s_mov_b32 m0, s36
	s_add_i32 s37, s51, 0x10000
	s_or_b32 s52, s50, 0x1000
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0xc800, v17
	s_mov_b32 m0, s37
	s_add_i32 s38, s52, 0x10000
	s_or_b32 s53, s50, 0x1800
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x19000, v17
	s_mov_b32 m0, s38
	s_add_i32 s39, s53, 0x10000
	s_or_b32 s54, s50, 0x2000
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x25800, v17
	s_mov_b32 m0, s39
	s_add_i32 s40, s54, 0x10000
	s_or_b32 s55, s50, 0x2800
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x32000, v17
	s_mov_b32 m0, s40
	s_add_i32 s41, s55, 0x10000
	s_or_b32 s56, s50, 0x3000
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x3e800, v17
	s_mov_b32 m0, s41
	s_add_i32 s42, s56, 0x10000
	s_or_b32 s57, s50, 0x3800
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x4b000, v17
	s_mov_b32 m0, s42
	s_add_i32 s43, s57, 0x10000
	s_or_b32 s58, s50, 0x4000
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x57800, v17
	s_mov_b32 m0, s43
	s_add_i32 s44, s58, 0x10000
	s_or_b32 s59, s50, 0x4800
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x64000, v17
	s_mov_b32 m0, s44
	s_add_i32 s45, s59, 0x10000
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0x70800, v17
	s_mov_b32 m0, s45
	v_bfe_u32 v80, v83, 4, 2
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_lshrrev_b32_e32 v21, 4, v83
	v_lshlrev_b32_e32 v17, 4, v80
	v_mad_i32_i24 v86, v21, -16, v17
	v_add_u32_e32 v19, v86, v83
	v_ashrrev_i32_e32 v20, 31, v19
	v_xor_b32_e32 v19, v20, v19
	v_mul_hi_i32 v19, v19, s35
	s_mul_i32 s15, s15, s28
	s_mul_hi_u32 s2, s14, s28
	v_lshrrev_b32_e32 v22, 31, v19
	v_ashrrev_i32_e32 v19, 4, v19
	s_add_i32 s2, s2, s15
	s_mul_i32 s3, s14, s28
	v_add_u32_e32 v19, v19, v22
	v_and_b32_e32 v81, 0xc0, v83
	s_add_u32 s4, s4, s3
	v_xad_u32 v22, v19, v20, v81
	v_and_b32_e32 v19, 62, v83
	s_movk_i32 s47, 0xffc0
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s14, 0x3fff
	v_mov_b32_e32 v20, 0xffffff38
	v_cmp_lt_u32_e32 vcc, 49, v19
	v_lshlrev_b32_e32 v19, 6, v80
	v_mad_i32_i24 v25, v21, s47, v85
	s_bitset1_b32 s3, 14
	v_cndmask_b32_e32 v23, 0, v20, vcc
	v_add_u32_e32 v28, v25, v19
	v_mul_lo_u32 v22, v22, s14
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	s_or_b32 s5, s2, s3
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	v_add3_u32 v22, v28, v23, v22
	v_mul_i32_i24_e32 v18, -16, v21
	v_lshl_add_u32 v23, s14, 5, v22
	buffer_load_dword v87, v22, s[4:7], 0 offen
	buffer_load_dword v84, v23, s[4:7], 0 offen
	v_mul_u32_u24_e32 v22, 0x50, v3
	v_add3_u32 v82, v18, v83, v22
	v_ashrrev_i16_e32 v22, 15, v82
	v_lshrrev_b16_e32 v22, 11, v22
	v_add_u16_e32 v22, v82, v22
	v_and_b32_e32 v22, 0xffffffe0, v22
	v_sub_u16_e32 v22, v82, v22
	v_bfe_i32 v23, v22, 0, 16
	v_ashrrev_i32_e32 v24, 31, v23
	v_add_u16_e32 v26, 32, v22
	v_cmp_gt_i16_e32 vcc, 0, v22
	s_load_dwordx2 s[12:13], s[0:1], 0x40
	s_movk_i32 s15, 0xc8
	v_cndmask_b32_e32 v22, v23, v26, vcc
	v_cndmask_b32_e64 v23, v24, 0, vcc
	v_xor_b32_e32 v22, v23, v22
	v_lshrrev_b32_e32 v24, 28, v22
	v_add_u32_e32 v22, v22, v24
	v_ashrrev_i32_e32 v22, 4, v22
	v_xor_b32_e32 v22, v22, v23
	v_add_u32_e32 v23, v28, v22
	v_ashrrev_i32_e32 v24, 31, v23
	v_xor_b32_e32 v26, v24, v23
	v_mul_hi_i32 v26, v26, s35
	v_lshrrev_b32_e32 v27, 31, v26
	v_ashrrev_i32_e32 v26, 6, v26
	v_add_u32_e32 v26, v26, v27
	v_ashrrev_i32_e32 v27, 31, v82
	v_xor_b32_e32 v29, v27, v82
	v_ashrrev_i32_e32 v30, 31, v29
	v_lshrrev_b32_e32 v30, 27, v30
	v_add_u32_e32 v29, v29, v30
	v_lshrrev_b32_e32 v29, 5, v29
	v_xor_b32_e32 v27, v29, v27
	v_lshlrev_b32_e32 v88, 5, v27
	v_xad_u32 v26, v26, v24, v88
	v_mul_hi_i32 v24, v23, s35
	v_lshrrev_b32_e32 v27, 31, v24
	v_ashrrev_i32_e32 v24, 6, v24
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s13, s29
	s_mul_hi_u32 s3, s12, s29
	v_add_u32_e32 v24, v24, v27
	s_add_i32 s3, s3, s2
	s_mul_i32 s2, s12, s29
	v_mul_lo_u32 v24, v24, s15
	s_add_u32 s16, s8, s2
	v_sub_u32_e32 v24, v23, v24
	s_addc_u32 s2, s9, s3
	s_and_b32 s3, s12, 0x3fff
	v_add_u32_e32 v27, 0xc8, v24
	v_cmp_gt_i32_e32 vcc, 0, v24
	s_bitset1_b32 s3, 14
	s_and_b32 s2, s2, 0xffff
	v_cndmask_b32_e32 v24, v24, v27, vcc
	s_lshl_b32 s3, s3, 16
	s_or_b32 s17, s2, s3
	v_mad_u64_u32 v[26:27], s[2:3], s12, v26, v[24:25]
	v_add_u32_e32 v24, 2, v23
	v_sub_u32_e32 v27, -3, v23
	v_cmp_gt_i32_e32 vcc, -2, v23
	s_movk_i32 s46, 0xff38
	v_add3_u32 v25, v22, v19, v25
	v_cndmask_b32_e32 v23, v24, v27, vcc
	v_mul_hi_i32 v23, v23, s35
	v_lshrrev_b32_e32 v24, 31, v23
	v_ashrrev_i32_e32 v23, 6, v23
	v_add_u32_e32 v23, v23, v24
	v_cndmask_b32_e64 v24, 0, -1, vcc
	v_xor_b32_e32 v23, v23, v24
	v_add_u32_e32 v24, v23, v88
	v_mul_lo_u32 v23, v23, s46
	v_mul_lo_u32 v24, s12, v24
	v_add3_u32 v27, v25, v23, v24
	v_add_u32_e32 v24, 16, v82
	v_sub_u32_e32 v25, 0xffef, v82
	v_cmp_gt_i32_e32 vcc, -16, v82
	v_mul_i32_i24_e32 v20, 0xffffffc0, v21
	v_mad_u32_u24 v29, v3, 5, v85
	v_cndmask_b32_e32 v24, v24, v25, vcc
	v_ashrrev_i16_e32 v25, 15, v24
	v_lshrrev_b16_e32 v25, 11, v25
	v_add_u16_e32 v24, v24, v25
	v_ashrrev_i16_e32 v24, 5, v24
	v_cndmask_b32_e64 v25, 0, -1, vcc
	v_xor_b32_e32 v24, v24, v25
	v_bfe_i32 v30, v24, 0, 16
	v_mad_i32_i24 v25, v30, -2, v20
	v_add3_u32 v29, v25, v29, v19
	v_add_u32_e32 v31, 1, v29
	v_sub_u32_e32 v32, -2, v29
	v_cmp_gt_i32_e32 vcc, -1, v29
	v_mul_u32_u24_e32 v23, 5, v3
	v_mul_i32_i24_e32 v24, -2, v30
	v_cndmask_b32_e32 v31, v31, v32, vcc
	v_mul_hi_i32 v31, v31, s35
	v_lshrrev_b32_e32 v32, 31, v31
	v_ashrrev_i32_e32 v31, 6, v31
	v_add_u32_e32 v31, v31, v32
	v_cndmask_b32_e64 v32, 0, -1, vcc
	v_xor_b32_e32 v36, v31, v32
	v_lshlrev_b32_e32 v89, 5, v30
	v_add3_u32 v28, v24, v23, v28
	v_add_u32_e32 v32, v36, v89
	v_mad_u64_u32 v[30:31], s[2:3], v36, s46, v[28:29]
	v_mad_u64_u32 v[32:33], s[2:3], v32, s12, v[30:31]
	v_add_u32_e32 v31, 3, v29
	v_sub_u32_e32 v33, -4, v29
	v_cmp_gt_i32_e32 vcc, -3, v29
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	v_cndmask_b32_e32 v29, v31, v33, vcc
	v_mul_hi_i32 v29, v29, s35
	v_lshrrev_b32_e32 v31, 31, v29
	v_ashrrev_i32_e32 v29, 6, v29
	v_add_u32_e32 v29, v29, v31
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_xor_b32_e32 v33, v29, v31
	v_add_u32_e32 v31, v33, v89
	v_mad_u64_u32 v[28:29], s[2:3], v33, s46, v[28:29]
	v_mad_u64_u32 v[34:35], s[2:3], v31, s12, v[28:29]
	s_movk_i32 s2, 0xffd0
	v_add_u32_e32 v29, 48, v82
	v_sub_u32_e32 v31, 0xffcf, v82
	v_cmp_gt_i32_e32 vcc, s2, v82
	s_lshl_b32 s6, s12, 5
	v_add_u32_e32 v35, s6, v26
	v_cndmask_b32_e32 v29, v29, v31, vcc
	v_ashrrev_i16_e32 v31, 15, v29
	v_lshrrev_b16_e32 v31, 11, v31
	v_add_u16_e32 v29, v29, v31
	v_ashrrev_i16_e32 v29, 5, v29
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_xor_b32_e32 v29, v29, v31
	v_mov_b32_e32 v31, 5
	v_lshlrev_b32_sdwa v90, v31, sext(v29) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_e32 v29, v90, v36
	v_mad_u64_u32 v[30:31], s[2:3], s12, v29, v[30:31]
	v_add_u32_e32 v29, v90, v33
	v_add_u32_e32 v37, s6, v27
	v_mad_u64_u32 v[28:29], s[2:3], s12, v29, v[28:29]
	buffer_load_ubyte v123, v26, s[16:19], 0 offen
	buffer_load_ubyte v117, v27, s[16:19], 0 offen offset:2
	buffer_load_ubyte v122, v32, s[16:19], 0 offen offset:1
	buffer_load_ubyte v116, v34, s[16:19], 0 offen offset:3
	buffer_load_ubyte v105, v35, s[16:19], 0 offen
	buffer_load_ubyte v94, v37, s[16:19], 0 offen offset:2
	buffer_load_ubyte v101, v30, s[16:19], 0 offen offset:1
	buffer_load_ubyte v93, v28, s[16:19], 0 offen offset:3
	v_add_u32_e32 v26, s6, v35
	v_add_u32_e32 v27, s6, v37
	buffer_load_ubyte v110, v26, s[16:19], 0 offen
	buffer_load_ubyte v95, v27, s[16:19], 0 offen offset:2
	v_cmp_eq_u32_e64 s[2:3], 0, v3
	s_movk_i32 s6, 0x2800
	s_mov_b32 s13, -2
	s_and_b64 vcc, exec, s[2:3]
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	v_lshlrev_b32_e32 v28, 7, v83
	v_lshlrev_b32_e32 v21, 11, v21
	v_and_b32_e32 v26, 7, v83
	v_sub_u32_e32 v21, v28, v21
	v_mul_lo_u32 v3, v3, s6
	v_bitop3_b32 v27, v80, v83, 7 bitop3:0x78
	v_lshl_add_u32 v4, v4, 13, v21
	v_add_u32_e32 v3, v21, v3
	v_bitop3_b32 v21, v80, v26, 4 bitop3:0x36
	v_lshlrev_b32_e32 v27, 4, v27
	v_lshlrev_b32_e32 v21, 4, v21
	v_or_b32_e32 v96, v3, v27
	v_or_b32_e32 v97, v21, v3
	v_add3_u32 v3, v14, v15, v16
	v_or_b32_e32 v91, v4, v27
	v_or_b32_e32 v92, v21, v4
	v_lshl_add_u32 v3, v7, 4, v3
	v_lshlrev_b32_e32 v4, 4, v8
	v_sub_u32_e32 v3, v3, v4
	v_sub_u32_e32 v98, v3, v13
	v_sub_u32_e32 v3, v12, v6
	v_sub_u32_e32 v3, v3, v0
	v_sub_u32_e32 v3, v3, v5
	v_sub_u32_e32 v100, v3, v9
	v_add_u32_e32 v3, v9, v5
	v_add3_u32 v0, v3, v0, v6
	v_sub_u32_e32 v0, v0, v10
	s_load_dwordx2 s[8:9], s[0:1], 0x48
	v_sub_u32_e32 v102, v0, v11
	v_add3_u32 v0, v20, v22, v19
	v_sub_u32_e32 v106, 0, v0
	v_add_u32_e32 v0, v24, v20
	v_add_u32_e32 v107, v20, v19
	v_add3_u32 v0, v0, v23, v19
	s_mov_b32 s0, 0x96100
	v_add_u32_e32 v108, v107, v22
	v_add3_u32 v112, v25, v19, v23
	v_sub_u32_e32 v113, 0, v0
	v_add_u32_e32 v0, v18, v17
	v_mov_b32_e32 v12, 0
	v_sub_u32_e32 v99, 0, v83
	v_add3_u32 v103, v2, v1, s0
	v_sub_u32_e32 v104, 0, v85
	v_add_u32_e32 v109, 32, v88
	v_add_u32_e32 v111, 64, v88
	v_add_u32_e32 v114, 32, v81
	v_sub_u32_e32 v115, 0, v0
	s_add_i32 s0, s30, 0x8000
	s_add_i32 s1, s31, 0x8000
	s_add_i32 s48, s33, 0x8000
	s_add_i32 s49, s34, 0x8000
	s_add_i32 s50, s50, 0x15000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	s_add_i32 s51, s51, 0x15000
	s_add_i32 s52, s52, 0x15000
	s_add_i32 s53, s53, 0x15000
	s_add_i32 s54, s54, 0x15000
	s_add_i32 s55, s55, 0x15000
	s_add_i32 s56, s56, 0x15000
	s_add_i32 s57, s57, 0x15000
	s_add_i32 s58, s58, 0x15000
	s_add_i32 s59, s59, 0x15000
	v_add_u32_e32 v118, 0x10000, v96
	s_movk_i32 s60, 0x147b
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	s_movk_i32 s61, 0xff00
	s_movk_i32 s62, 0xfeff
	s_movk_i32 s63, 0xffc8
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	s_movk_i32 s64, 0xfefd
	s_movk_i32 s65, 0xfefe
	s_movk_i32 s66, 0xffc7
	v_add_u32_e32 v119, 0x10000, v97
	v_add_u32_e32 v120, 0x15000, v96
	v_add_u32_e32 v121, 0x15000, v97
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
	v_mov_b32_e32 v32, v12
	v_mov_b32_e32 v33, v12
	v_mov_b32_e32 v34, v12
	v_mov_b32_e32 v35, v12
	v_mov_b32_e32 v20, v12
	v_mov_b32_e32 v21, v12
	v_mov_b32_e32 v22, v12
	v_mov_b32_e32 v23, v12
	v_mov_b32_e32 v16, v12
	v_mov_b32_e32 v17, v12
	v_mov_b32_e32 v18, v12
	v_mov_b32_e32 v19, v12
	v_mov_b32_e32 v24, v12
	v_mov_b32_e32 v25, v12
	v_mov_b32_e32 v26, v12
	v_mov_b32_e32 v27, v12
	v_mov_b32_e32 v28, v12
	v_mov_b32_e32 v29, v12
	v_mov_b32_e32 v30, v12
	v_mov_b32_e32 v31, v12
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
	v_mov_b32_e32 v124, v112
	v_mov_b32_e32 v125, v108
.LBB0_3:
	s_mov_b32 m0, s0
	v_add_u32_e32 v126, 0xfff69f80, v103
	s_waitcnt vmcnt(0)
	s_barrier
	buffer_load_dwordx4 v126, s[24:27], 0 offen lds
	v_add_u32_e32 v126, 0xfff9bf80, v103
	s_mov_b32 m0, s1
	v_add_u32_e32 v164, v83, v102
	buffer_load_dwordx4 v126, s[24:27], 0 offen lds
	v_add_u32_e32 v126, 0xfffcdf80, v103
	s_mov_b32 m0, s48
	v_add_u32_e32 v165, v99, v100
	buffer_load_dwordx4 v126, s[24:27], 0 offen lds
	v_add_u32_e32 v126, 0xffffff80, v103
	s_mov_b32 m0, s49
	v_add_u32_e32 v127, 0xfffffdff, v165
	buffer_load_dwordx4 v126, s[24:27], 0 offen lds
	v_add_u32_e32 v126, 0x200, v164
	v_cmp_gt_i32_e32 vcc, 0, v126
	v_add_u32_e32 v166, v98, v85
	s_mov_b32 m0, s50
	v_cndmask_b32_e32 v127, v126, v127, vcc
	v_mul_hi_i32 v127, v127, s35
	v_lshrrev_b32_e32 v128, 31, v127
	v_ashrrev_i32_e32 v127, 8, v127
	v_add_u32_e32 v127, v127, v128
	v_ashrrev_i32_e32 v126, 31, v126
	v_xor_b32_e32 v126, v127, v126
	v_add_u32_e32 v127, s29, v126
	v_mul_i32_i24_e32 v126, 0xfffff380, v126
	v_mul_i32_i24_e32 v127, 0xc80, v127
	v_add3_u32 v126, v126, v127, v166
	v_add_u32_e32 v127, 0x800, v126
	buffer_load_dword v127, s[20:23], 0 offen lds
	v_add_u32_e32 v127, 0xd000, v126
	s_mov_b32 m0, s51
	s_nop 0
	buffer_load_dword v127, s[20:23], 0 offen lds
	v_add_u32_e32 v127, 0x19800, v126
	s_mov_b32 m0, s52
	s_nop 0
	buffer_load_dword v127, s[20:23], 0 offen lds
	v_add_u32_e32 v127, 0x26000, v126
	s_mov_b32 m0, s53
	s_nop 0
	buffer_load_dword v127, s[20:23], 0 offen lds
	v_add_u32_e32 v127, 0x32800, v126
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dword v127, s[20:23], 0 offen lds
	v_add_u32_e32 v127, 0x3f000, v126
	s_mov_b32 m0, s55
	s_nop 0
	buffer_load_dword v127, s[20:23], 0 offen lds
	v_add_u32_e32 v127, 0x4b800, v126
	s_mov_b32 m0, s56
	s_nop 0
	buffer_load_dword v127, s[20:23], 0 offen lds
	v_add_u32_e32 v127, 0x58000, v126
	s_mov_b32 m0, s57
	s_nop 0
	buffer_load_dword v127, s[20:23], 0 offen lds
	v_add_u32_e32 v127, 0x64800, v126
	s_mov_b32 m0, s58
	v_add_u32_e32 v126, 0x71000, v126
	buffer_load_dword v127, s[20:23], 0 offen lds
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dword v126, s[20:23], 0 offen lds
	v_add_u32_e32 v167, v83, v86
	v_add_u32_e32 v169, v99, v115
	v_add_u32_e32 v168, 64, v167
	v_add_u32_e32 v126, 0xffbf, v169
	v_cmp_gt_i32_e32 vcc, s47, v167
	v_add_u32_e32 v128, 14, v167
	v_add_u32_e32 v129, 0xfff1, v169
	v_cndmask_b32_e32 v126, v168, v126, vcc
	v_mul_i32_i24_sdwa v126, sext(v126), s60 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshrrev_b32_e32 v127, 31, v126
	v_ashrrev_i32_e32 v126, 18, v126
	v_add_u16_e32 v126, v126, v127
	v_cndmask_b32_e64 v127, 0, -1, vcc
	v_cmp_gt_i32_e32 vcc, -14, v167
	v_xor_b32_e32 v126, v126, v127
	v_add_u32_sdwa v127, v81, sext(v126) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_cndmask_b32_e32 v128, v128, v129, vcc
	v_mul_i32_i24_sdwa v128, sext(v128), s60 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshrrev_b32_e32 v129, 31, v128
	v_ashrrev_i32_e32 v128, 18, v128
	v_add_u16_e32 v128, v128, v129
	v_cndmask_b32_e64 v129, 0, -1, vcc
	v_xor_b32_e32 v128, v128, v129
	v_add_u32_sdwa v126, v114, sext(v126) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_bfe_i32 v128, v128, 0, 16
	v_mul_lo_u32 v127, v127, s14
	v_mul_lo_u32 v126, s14, v126
	v_add_u32_e32 v170, v85, v108
	v_add_u32_e32 v172, v104, v106
	v_mad_i32_i24 v127, v128, s46, v127
	v_mad_i32_i24 v126, v128, s46, v126
	v_add_u32_e32 v171, 0x100, v170
	v_add_u32_e32 v128, 0xfffffeff, v172
	v_cmp_gt_i32_e32 vcc, s61, v170
	v_add_u32_e32 v130, 56, v170
	v_subrev_u32_e32 v131, 57, v172
	v_cndmask_b32_e32 v128, v171, v128, vcc
	v_mul_hi_i32 v128, v128, s35
	v_lshrrev_b32_e32 v129, 31, v128
	v_ashrrev_i32_e32 v128, 6, v128
	v_add_u32_e32 v128, v128, v129
	v_cndmask_b32_e64 v129, 0, -1, vcc
	v_cmp_gt_i32_e32 vcc, s63, v170
	v_add_u32_e32 v132, 0xfffffefd, v172
	v_add_u32_e32 v174, v85, v112
	v_cndmask_b32_e32 v130, v130, v131, vcc
	v_mul_hi_i32 v130, v130, s35
	v_lshrrev_b32_e32 v131, 31, v130
	v_ashrrev_i32_e32 v130, 6, v130
	v_add_u32_e32 v130, v130, v131
	v_cndmask_b32_e64 v131, 0, -1, vcc
	v_xor_b32_e32 v130, v130, v131
	v_add_u32_e32 v131, 0x102, v170
	v_cmp_gt_i32_e32 vcc, -2, v171
	v_add_u32_e32 v175, v104, v113
	v_add_u32_e32 v135, 0xfffffefe, v175
	v_cndmask_b32_e32 v133, v131, v132, vcc
	v_mul_hi_i32 v133, v133, s35
	v_lshrrev_b32_e32 v134, 31, v133
	v_ashrrev_i32_e32 v133, 6, v133
	v_cmp_gt_i32_e32 vcc, s65, v170
	v_add_u32_e32 v133, v133, v134
	v_ashrrev_i32_e32 v134, 31, v131
	v_cndmask_b32_e32 v131, v131, v132, vcc
	v_mul_hi_i32 v131, v131, s35
	v_xor_b32_e32 v133, v133, v134
	v_lshrrev_b32_e32 v132, 31, v131
	v_ashrrev_i32_e32 v131, 6, v131
	v_add_u32_e32 v134, v133, v88
	v_add_u32_e32 v131, v131, v132
	v_cndmask_b32_e64 v132, 0, -1, vcc
	v_xor_b32_e32 v131, v131, v132
	v_mul_lo_u32 v132, v134, s12
	v_add_u32_e32 v134, 0x101, v174
	v_cmp_gt_i32_e32 vcc, s62, v174
	v_subrev_u32_e32 v137, 58, v175
	v_add_u32_e32 v176, 0x100, v174
	v_cndmask_b32_e32 v135, v134, v135, vcc
	v_mul_hi_i32 v135, v135, s35
	v_lshrrev_b32_e32 v136, 31, v135
	v_ashrrev_i32_e32 v135, 6, v135
	v_add_u32_e32 v135, v135, v136
	v_add_u32_e32 v136, 57, v174
	v_cmp_gt_i32_e32 vcc, s66, v174
	v_ashrrev_i32_e32 v134, 31, v134
	v_xor_b32_e32 v134, v135, v134
	v_cndmask_b32_e32 v137, v136, v137, vcc
	v_mul_hi_i32 v137, v137, s35
	v_lshrrev_b32_e32 v138, 31, v137
	v_ashrrev_i32_e32 v137, 6, v137
	v_add_u32_e32 v137, v137, v138
	v_ashrrev_i32_e32 v136, 31, v136
	v_add_u32_e32 v135, v134, v89
	v_xor_b32_e32 v136, v137, v136
	v_add_u32_e32 v137, 0x103, v174
	v_add_u32_e32 v138, 0xfffffefc, v175
	v_cmp_gt_i32_e32 vcc, s64, v174
	v_add_u32_e32 v134, v134, v90
	v_mul_lo_u32 v135, v135, s12
	v_mul_lo_u32 v136, v136, s15
	v_cndmask_b32_e32 v139, v137, v138, vcc
	v_mul_lo_u32 v134, v134, s12
	v_cmp_gt_i32_e32 vcc, -3, v176
	v_sub_u32_e32 v135, v135, v136
	v_mul_hi_i32 v139, v139, s35
	v_sub_u32_e32 v134, v134, v136
	v_cndmask_b32_e32 v136, v137, v138, vcc
	v_lshrrev_b32_e32 v140, 31, v139
	v_ashrrev_i32_e32 v139, 6, v139
	v_mul_hi_i32 v136, v136, s35
	v_xor_b32_e32 v128, v128, v129
	v_add_u32_e32 v139, v139, v140
	v_ashrrev_i32_e32 v140, 31, v137
	v_lshrrev_b32_e32 v137, 31, v136
	v_ashrrev_i32_e32 v136, 6, v136
	v_add_u32_e32 v129, v128, v88
	v_xor_b32_e32 v139, v139, v140
	v_add_u32_e32 v136, v136, v137
	v_mul_lo_u32 v129, v129, s12
	v_mul_lo_u32 v130, v130, s15
	v_add_u32_e32 v141, v139, v89
	v_add_u32_e32 v142, v109, v128
	v_add_u32_e32 v143, v109, v133
	v_xad_u32 v136, v136, v140, v90
	v_sub_u32_e32 v129, v129, v130
	v_add_u32_e32 v173, v125, v85
	v_mul_lo_u32 v131, v131, s15
	v_mul_lo_u32 v141, v141, s12
	v_mul_lo_u32 v139, v139, s15
	v_mul_lo_u32 v142, s12, v142
	v_mul_lo_u32 v143, s12, v143
	v_mul_lo_u32 v136, v136, s12
	v_add_u32_e32 v129, v173, v129
	v_sub_u32_e32 v132, v132, v131
	v_sub_u32_e32 v141, v141, v139
	v_sub_u32_e32 v142, v142, v130
	v_sub_u32_e32 v143, v143, v131
	v_sub_u32_e32 v136, v136, v139
	v_add3_u32 v132, v85, v132, v125
	v_add3_u32 v135, v85, v135, v124
	v_add3_u32 v141, v85, v141, v124
	v_add_u32_e32 v142, v173, v142
	v_add3_u32 v143, v85, v143, v125
	v_add3_u32 v134, v85, v134, v124
	v_add3_u32 v136, v85, v136, v124
	buffer_load_ubyte v177, v129, s[16:19], 0 offen offset:56
	buffer_load_ubyte v178, v132, s[16:19], 0 offen offset:258
	buffer_load_ubyte v179, v135, s[16:19], 0 offen offset:57
	buffer_load_ubyte v180, v141, s[16:19], 0 offen offset:259
	buffer_load_ubyte v181, v142, s[16:19], 0 offen offset:56
	buffer_load_ubyte v182, v143, s[16:19], 0 offen offset:258
	buffer_load_ubyte v183, v134, s[16:19], 0 offen offset:57
	buffer_load_ubyte v184, v136, s[16:19], 0 offen offset:259
	v_add_u32_e32 v128, v111, v128
	v_mul_lo_u32 v128, s12, v128
	v_add_u32_e32 v129, v111, v133
	v_sub_u32_e32 v128, v128, v130
	v_mul_lo_u32 v129, s12, v129
	v_add3_u32 v127, v85, v127, v107
	v_add3_u32 v126, v85, v126, v107
	v_add_u32_e32 v128, v173, v128
	v_sub_u32_e32 v129, v129, v131
	buffer_load_dword v127, v127, s[4:7], 0 offen offset:56
	s_nop 0
	buffer_load_dword v126, v126, s[4:7], 0 offen offset:56
	v_add3_u32 v129, v85, v129, v125
	buffer_load_ubyte v185, v128, s[16:19], 0 offen offset:56
	buffer_load_ubyte v186, v129, s[16:19], 0 offen offset:258
	ds_read_b128 v[128:131], v91
	ds_read_b128 v[132:135], v91 offset:2048
	ds_read_b128 v[140:143], v91 offset:4096
	ds_read_b128 v[160:163], v91 offset:6144
	ds_read_b128 v[144:147], v118
	ds_read_b128 v[148:151], v118 offset:2048
	ds_read_b128 v[152:155], v118 offset:4096
	ds_read_b128 v[156:159], v118 offset:6144
	ds_read_b128 v[188:191], v118 offset:8192
	s_barrier
	s_setprio 1
	v_and_b32_e32 v123, 0xff, v123
	v_and_b32_e32 v122, 0xff, v122
	v_and_b32_e32 v105, 0xff, v105
	v_and_b32_e32 v101, 0xff, v101
	v_and_b32_e32 v110, 0xff, v110
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[128:131], v[144:147], v[12:15], v87, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[128:131], v[148:151], v[76:79], v87, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[128:131], v[152:155], v[72:75], v87, v105 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[128:131], v[156:159], v[68:71], v87, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[128:131], v[188:191], v[64:67], v87, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[132:135], v[144:147], v[60:63], v87, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[132:135], v[148:151], v[56:59], v87, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[132:135], v[152:155], v[52:55], v87, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[132:135], v[156:159], v[48:51], v87, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[132:135], v[188:191], v[44:47], v87, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[140:143], v[144:147], v[40:43], v84, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[128:131], v[140:143], v[148:151], v[32:35], v84, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[132:135], v[140:143], v[152:155], v[20:23], v84, v105 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[136:139], v[140:143], v[156:159], v[16:19], v84, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[140:143], v[140:143], v[188:191], v[24:27], v84, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], v[160:163], v[144:147], v[28:31], v84, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], v[160:163], v[148:151], v[36:39], v84, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], v[160:163], v[152:155], v[8:11], v84, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[156:159], v[160:163], v[156:159], v[4:7], v84, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[160:163], v[160:163], v[188:191], v[0:3], v84, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	ds_read_b128 v[16:19], v92
	ds_read_b128 v[36:39], v92 offset:2048
	ds_read_b128 v[188:191], v92 offset:4096
	ds_read_b128 v[192:195], v92 offset:6144
	ds_read_b128 v[196:199], v119
	ds_read_b128 v[200:203], v119 offset:2048
	ds_read_b128 v[204:207], v119 offset:4096
	ds_read_b128 v[208:211], v119 offset:6144
	ds_read_b128 v[212:215], v119 offset:8192
	s_waitcnt vmcnt(12)
	s_barrier
	s_setprio 1
	v_and_b32_e32 v105, 0xff, v116
	v_and_b32_e32 v101, 0xff, v117
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[16:19], v[200:203], v[76:79], v87, v105 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 2
	v_and_b32_e32 v76, 0xff, v94
	v_and_b32_e32 v77, 0xff, v95
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[16:19], v[196:199], v[12:15], v87, v101 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[16:19], v[204:207], v[72:75], v87, v76 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 2
	v_and_b32_e32 v72, 0xff, v93
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[36:39], v[196:199], v[60:63], v87, v101 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[16:19], v[208:211], v[68:71], v87, v72 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[16:19], v[212:215], v[64:67], v87, v77 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[36:39], v[200:203], v[56:59], v87, v105 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[36:39], v[204:207], v[52:55], v87, v76 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[36:39], v[208:211], v[48:51], v87, v72 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[36:39], v[212:215], v[44:47], v87, v77 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[188:191], v[196:199], v[40:43], v84, v101 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[188:191], v[200:203], v[128:131], v84, v105 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[188:191], v[204:207], v[132:135], v84, v76 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[188:191], v[208:211], v[136:139], v84, v72 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[188:191], v[212:215], v[140:143], v84, v77 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[192:195], v[196:199], v[144:147], v84, v101 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[192:195], v[200:203], v[148:151], v84, v105 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[192:195], v[204:207], v[152:155], v84, v76 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[192:195], v[208:211], v[156:159], v84, v72 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[192:195], v[212:215], v[160:163], v84, v77 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_mov_b32 m0, s30
	v_add_u32_e32 v84, 0xfff6a000, v103
	s_waitcnt vmcnt(0)
	s_barrier
	buffer_load_dwordx4 v84, s[24:27], 0 offen lds
	v_add_u32_e32 v84, 0xfff9c000, v103
	s_mov_b32 m0, s31
	v_add_u32_e32 v87, 0xfffffbff, v165
	buffer_load_dwordx4 v84, s[24:27], 0 offen lds
	v_add_u32_e32 v84, 0xfffce000, v103
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v84, s[24:27], 0 offen lds
	v_add_u32_e32 v84, 0x400, v164
	v_cmp_gt_i32_e32 vcc, 0, v84
	s_mov_b32 m0, s34
	s_nop 0
	v_cndmask_b32_e32 v87, v84, v87, vcc
	v_mul_hi_i32 v87, v87, s35
	v_lshrrev_b32_e32 v93, 31, v87
	v_ashrrev_i32_e32 v87, 8, v87
	v_add_u32_e32 v87, v87, v93
	v_ashrrev_i32_e32 v84, 31, v84
	v_xor_b32_e32 v84, v87, v84
	v_add_u32_e32 v87, s29, v84
	v_mul_i32_i24_e32 v84, 0xfffff380, v84
	v_mul_i32_i24_e32 v87, 0xc80, v87
	v_add3_u32 v84, v84, v87, v166
	buffer_load_dwordx4 v103, s[24:27], 0 offen lds
	v_add_u32_e32 v87, 0x1000, v84
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dword v87, s[20:23], 0 offen lds
	v_add_u32_e32 v87, 0xd800, v84
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dword v87, s[20:23], 0 offen lds
	v_add_u32_e32 v87, 0x1a000, v84
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dword v87, s[20:23], 0 offen lds
	v_add_u32_e32 v87, 0x26800, v84
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dword v87, s[20:23], 0 offen lds
	v_add_u32_e32 v87, 0x33000, v84
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dword v87, s[20:23], 0 offen lds
	v_add_u32_e32 v87, 0x3f800, v84
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dword v87, s[20:23], 0 offen lds
	v_add_u32_e32 v87, 0x4c000, v84
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dword v87, s[20:23], 0 offen lds
	v_add_u32_e32 v87, 0x58800, v84
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dword v87, s[20:23], 0 offen lds
	v_add_u32_e32 v87, 0x65000, v84
	s_mov_b32 m0, s44
	v_add_u32_e32 v84, 0x71800, v84
	buffer_load_dword v87, s[20:23], 0 offen lds
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dword v84, s[20:23], 0 offen lds
	v_add_u32_e32 v84, 0x80, v167
	v_add_u32_e32 v87, 0xff7f, v169
	v_cmp_gt_i32_e32 vcc, s47, v168
	v_add_u32_e32 v93, 0x4e, v167
	v_add_u32_e32 v94, 0xffb1, v169
	v_cndmask_b32_e32 v84, v84, v87, vcc
	v_mul_i32_i24_sdwa v84, sext(v84), s60 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshrrev_b32_e32 v87, 31, v84
	v_ashrrev_i32_e32 v84, 18, v84
	v_add_u16_e32 v84, v84, v87
	v_cndmask_b32_e64 v87, 0, -1, vcc
	v_cmp_gt_i32_e32 vcc, -14, v168
	v_xor_b32_e32 v84, v84, v87
	v_add_u32_sdwa v87, v81, sext(v84) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_cndmask_b32_e32 v93, v93, v94, vcc
	v_mul_i32_i24_sdwa v93, sext(v93), s60 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshrrev_b32_e32 v94, 31, v93
	v_ashrrev_i32_e32 v93, 18, v93
	v_add_u16_e32 v93, v93, v94
	v_cndmask_b32_e64 v94, 0, -1, vcc
	v_xor_b32_e32 v93, v93, v94
	v_add_u32_sdwa v84, v114, sext(v84) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_bfe_i32 v93, v93, 0, 16
	v_mul_lo_u32 v87, v87, s14
	v_mul_lo_u32 v84, s14, v84
	v_mad_i32_i24 v87, v93, s46, v87
	v_mad_i32_i24 v84, v93, s46, v84
	v_add_u32_e32 v93, 0x200, v170
	v_add_u32_e32 v94, 0xfffffdff, v172
	v_cmp_gt_i32_e32 vcc, s61, v171
	v_add_u32_e32 v101, 0x138, v170
	v_add_u32_e32 v105, 0xfffffec7, v172
	v_cndmask_b32_e32 v94, v93, v94, vcc
	v_cmp_gt_i32_e32 vcc, s63, v171
	v_mul_hi_i32 v94, v94, s35
	v_lshrrev_b32_e32 v95, 31, v94
	v_cndmask_b32_e32 v105, v101, v105, vcc
	v_mul_hi_i32 v105, v105, s35
	v_lshrrev_b32_e32 v110, 31, v105
	v_ashrrev_i32_e32 v105, 6, v105
	v_add_u32_e32 v105, v105, v110
	v_ashrrev_i32_e32 v101, 31, v101
	v_xor_b32_e32 v101, v105, v101
	v_ashrrev_i32_e32 v94, 6, v94
	v_mul_lo_u32 v110, v101, s15
	v_add_u32_e32 v101, 0x202, v170
	v_add_u32_e32 v105, 0xfffffdfd, v172
	v_cmp_gt_i32_e32 vcc, -2, v93
	v_add_u32_e32 v94, v94, v95
	v_ashrrev_i32_e32 v95, 31, v93
	v_cndmask_b32_e32 v93, v101, v105, vcc
	v_mul_hi_i32 v93, v93, s35
	v_lshrrev_b32_e32 v116, 31, v93
	v_ashrrev_i32_e32 v93, 6, v93
	v_cmp_gt_i32_e32 vcc, s65, v171
	v_add_u32_e32 v93, v93, v116
	v_ashrrev_i32_e32 v116, 31, v101
	v_cndmask_b32_e32 v101, v101, v105, vcc
	v_mul_hi_i32 v101, v101, s35
	v_lshrrev_b32_e32 v105, 31, v101
	v_ashrrev_i32_e32 v101, 6, v101
	v_add_u32_e32 v101, v101, v105
	v_xor_b32_e32 v101, v101, v116
	v_mul_lo_u32 v129, v101, s15
	v_add_u32_e32 v101, 0x201, v174
	v_add_u32_e32 v105, 0xfffffdfe, v175
	v_cmp_gt_i32_e32 vcc, s62, v176
	v_xor_b32_e32 v128, v93, v116
	v_add_u32_e32 v117, 0xfffffec6, v175
	v_cndmask_b32_e32 v105, v101, v105, vcc
	v_mul_hi_i32 v105, v105, s35
	v_lshrrev_b32_e32 v116, 31, v105
	v_ashrrev_i32_e32 v105, 6, v105
	v_add_u32_e32 v105, v105, v116
	v_add_u32_e32 v116, 0x139, v174
	v_cmp_gt_i32_e32 vcc, s66, v176
	v_ashrrev_i32_e32 v101, 31, v101
	v_xor_b32_e32 v101, v105, v101
	v_cndmask_b32_e32 v117, v116, v117, vcc
	v_mul_hi_i32 v117, v117, s35
	v_lshrrev_b32_e32 v122, 31, v117
	v_ashrrev_i32_e32 v117, 6, v117
	v_add_u32_e32 v117, v117, v122
	v_ashrrev_i32_e32 v116, 31, v116
	v_add_u32_e32 v105, v101, v89
	v_xor_b32_e32 v116, v117, v116
	v_add_u32_e32 v101, v101, v90
	v_mul_lo_u32 v105, v105, s12
	v_mul_lo_u32 v116, v116, s15
	v_mul_lo_u32 v101, v101, s12
	v_sub_u32_e32 v105, v105, v116
	v_add_u32_e32 v117, 0x203, v174
	v_add_u32_e32 v122, 0xfffffdfc, v175
	v_cmp_gt_i32_e32 vcc, s64, v176
	v_sub_u32_e32 v101, v101, v116
	v_add_u32_e32 v116, 0x200, v174
	v_cndmask_b32_e32 v123, v117, v122, vcc
	v_cmp_gt_i32_e32 vcc, -3, v116
	v_mul_hi_i32 v123, v123, s35
	v_lshrrev_b32_e32 v130, 31, v123
	v_cndmask_b32_e32 v116, v117, v122, vcc
	v_ashrrev_i32_e32 v123, 6, v123
	v_mul_hi_i32 v116, v116, s35
	v_xor_b32_e32 v95, v94, v95
	v_add_u32_e32 v123, v123, v130
	v_ashrrev_i32_e32 v130, 31, v117
	v_lshrrev_b32_e32 v117, 31, v116
	v_ashrrev_i32_e32 v116, 6, v116
	v_add_u32_e32 v94, v95, v88
	v_add_u32_e32 v93, v128, v88
	v_xor_b32_e32 v123, v123, v130
	v_add_u32_e32 v116, v116, v117
	v_mul_lo_u32 v94, v94, s12
	v_mul_lo_u32 v93, v93, s12
	v_add_u32_e32 v131, v123, v89
	v_add_u32_e32 v132, v109, v95
	v_add_u32_e32 v133, v109, v128
	v_xad_u32 v116, v116, v130, v90
	v_sub_u32_e32 v94, v94, v110
	v_sub_u32_e32 v93, v93, v129
	v_mul_lo_u32 v131, v131, s12
	v_mul_lo_u32 v123, v123, s15
	v_mul_lo_u32 v132, s12, v132
	v_mul_lo_u32 v133, s12, v133
	v_mul_lo_u32 v116, v116, s12
	v_add_u32_e32 v94, v173, v94
	v_add3_u32 v93, v85, v93, v125
	v_add3_u32 v105, v85, v105, v124
	v_sub_u32_e32 v131, v131, v123
	v_sub_u32_e32 v132, v132, v110
	v_sub_u32_e32 v133, v133, v129
	v_add3_u32 v101, v85, v101, v124
	v_sub_u32_e32 v116, v116, v123
	v_add3_u32 v131, v85, v131, v124
	v_add_u32_e32 v132, v173, v132
	v_add3_u32 v133, v85, v133, v125
	v_add3_u32 v130, v85, v116, v124
	buffer_load_ubyte v123, v94, s[16:19], 0 offen offset:312
	buffer_load_ubyte v117, v93, s[16:19], 0 offen offset:514
	buffer_load_ubyte v122, v105, s[16:19], 0 offen offset:313
	buffer_load_ubyte v116, v131, s[16:19], 0 offen offset:515
	s_nop 0
	buffer_load_ubyte v105, v132, s[16:19], 0 offen offset:312
	buffer_load_ubyte v94, v133, s[16:19], 0 offen offset:514
	s_nop 0
	buffer_load_ubyte v101, v101, s[16:19], 0 offen offset:313
	s_nop 0
	buffer_load_ubyte v93, v130, s[16:19], 0 offen offset:515
	v_add_u32_e32 v95, v111, v95
	v_mul_lo_u32 v95, s12, v95
	v_sub_u32_e32 v95, v95, v110
	v_add_u32_e32 v110, v111, v128
	v_mul_lo_u32 v110, s12, v110
	v_add3_u32 v87, v85, v87, v107
	v_add3_u32 v84, v85, v84, v107
	v_add_u32_e32 v95, v173, v95
	v_sub_u32_e32 v110, v110, v129
	buffer_load_dword v87, v87, s[4:7], 0 offen offset:312
	s_nop 0
	buffer_load_dword v84, v84, s[4:7], 0 offen offset:312
	v_add3_u32 v128, v85, v110, v125
	buffer_load_ubyte v110, v95, s[16:19], 0 offen offset:312
	s_nop 0
	buffer_load_ubyte v95, v128, s[16:19], 0 offen offset:514
	ds_read_b128 v[132:135], v91 offset:32768
	ds_read_b128 v[136:139], v91 offset:34816
	ds_read_b128 v[144:147], v91 offset:36864
	ds_read_b128 v[164:167], v91 offset:38912
	ds_read_b128 v[148:151], v120
	ds_read_b128 v[152:155], v120 offset:2048
	ds_read_b128 v[156:159], v120 offset:4096
	ds_read_b128 v[160:163], v120 offset:6144
	ds_read_b128 v[168:171], v120 offset:8192
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[132:135], v[148:151], v[0:3], v127, v177 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[132:135], v[152:155], v[4:7], v127, v179 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[132:135], v[156:159], v[8:11], v127, v181 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[132:135], v[168:171], v[16:19], v127, v185 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[136:139], v[148:151], v[20:23], v127, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[136:139], v[152:155], v[24:27], v127, v179 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[136:139], v[156:159], v[28:31], v127, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[136:139], v[160:163], v[32:35], v127, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[136:139], v[168:171], v[36:39], v127, v185 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[144:147], v[148:151], v[40:43], v126, v177 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[128:131], v[132:135], v[160:163], v[12:15], v127, v183 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[132:135], v[144:147], v[152:155], v[44:47], v126, v179 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[136:139], v[144:147], v[156:159], v[48:51], v126, v181 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[140:143], v[144:147], v[160:163], v[52:55], v126, v183 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], v[144:147], v[168:171], v[56:59], v126, v185 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], v[164:167], v[148:151], v[60:63], v126, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], v[164:167], v[152:155], v[64:67], v126, v179 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[156:159], v[164:167], v[156:159], v[68:71], v126, v181 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[160:163], v[164:167], v[160:163], v[72:75], v126, v183 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[164:167], v[164:167], v[168:171], v[76:79], v126, v185 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	ds_read_b128 v[44:47], v92 offset:32768
	ds_read_b128 v[168:171], v92 offset:34816
	ds_read_b128 v[172:175], v92 offset:36864
	ds_read_b128 v[188:191], v92 offset:38912
	ds_read_b128 v[192:195], v121
	ds_read_b128 v[196:199], v121 offset:2048
	ds_read_b128 v[200:203], v121 offset:4096
	ds_read_b128 v[204:207], v121 offset:6144
	ds_read_b128 v[208:211], v121 offset:8192
	s_waitcnt vmcnt(12)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[44:47], v[192:195], v[0:3], v127, v178 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[44:47], v[196:199], v[4:7], v127, v180 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[44:47], v[200:203], v[8:11], v127, v182 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[44:47], v[204:207], v[128:131], v127, v184 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[44:47], v[208:211], v[16:19], v127, v186 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[168:171], v[192:195], v[20:23], v127, v178 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[168:171], v[196:199], v[24:27], v127, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[168:171], v[200:203], v[28:31], v127, v182 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[168:171], v[204:207], v[32:35], v127, v184 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[168:171], v[208:211], v[36:39], v127, v186 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[172:175], v[192:195], v[40:43], v126, v178 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[172:175], v[196:199], v[132:135], v126, v180 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[172:175], v[200:203], v[136:139], v126, v182 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[172:175], v[204:207], v[140:143], v126, v184 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[172:175], v[208:211], v[144:147], v126, v186 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[188:191], v[192:195], v[148:151], v126, v178 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[188:191], v[196:199], v[152:155], v126, v180 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[188:191], v[200:203], v[156:159], v126, v182 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[188:191], v[204:207], v[160:163], v126, v184 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[188:191], v[208:211], v[164:167], v126, v186 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_add_i32 s13, s13, 2
	v_add_u32_e32 v98, 0x1000, v98
	v_add_u32_e32 v100, 0xfffffc00, v100
	v_add_u32_e32 v102, 0x400, v102
	v_add_u32_e32 v103, 0x100, v103
	v_add_u32_e32 v106, 0xfffffe00, v106
	v_add_u32_e32 v108, 0x200, v108
	v_add_u32_e32 v125, 0x200, v125
	v_add_u32_e32 v112, 0x200, v112
	v_add_u32_e32 v113, 0xfffffe00, v113
	v_add_u32_e32 v124, 0x200, v124
	v_add_u32_e32 v107, 0x200, v107
	v_add_u32_e32 v115, 0xffffff80, v115
	s_cmp_lt_u32 s13, 22
	v_add_u32_e32 v86, 0x80, v86
	s_cbranch_scc1 .LBB0_3
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	v_add_u32_e32 v83, 0x10000, v96
	s_barrier
	ds_read_b128 v[124:127], v83
	ds_read_b128 v[128:131], v83 offset:2048
	v_add_u32_e32 v85, 0x10000, v97
	ds_read_b128 v[132:135], v85
	ds_read_b128 v[136:139], v83 offset:8192
	ds_read_b128 v[140:143], v85 offset:2048
	ds_read_b128 v[144:147], v85 offset:4096
	ds_read_b128 v[148:151], v83 offset:4096
	ds_read_b128 v[152:155], v83 offset:6144
	ds_read_b128 v[156:159], v85 offset:6144
	ds_read_b128 v[160:163], v85 offset:8192
	ds_read_b128 v[106:109], v91
	ds_read_b128 v[118:121], v91 offset:2048
	ds_read_b128 v[112:115], v92
	ds_read_b128 v[164:167], v92 offset:2048
	ds_read_b128 v[168:171], v91 offset:4096
	ds_read_b128 v[172:175], v91 offset:6144
	ds_read_b128 v[176:179], v92 offset:4096
	ds_read_b128 v[180:183], v92 offset:6144
	s_waitcnt vmcnt(3) lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[106:109], v[124:127], v[12:15], v87, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_movk_i32 s0, 0x7fff
	v_mov_b32_e32 v83, 0x7fc0
	s_mul_hi_u32 s1, s8, s28
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[112:115], v[132:135], v[12:15], v87, v117 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshl_or_b32 v80, v80, 2, v81
	v_mul_lo_u32 v80, s8, v80
	v_lshlrev_b32_e32 v81, 1, v82
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[106:109], v[148:151], v[72:75], v87, v105 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 s3, 0x27000
	s_nop 2
	v_bfe_u32 v85, v15, 16, 1
	v_bfe_u32 v86, v14, 16, 1
	v_add3_u32 v85, v15, v85, s0
	v_bfe_u32 v88, v13, 16, 1
	v_bfe_u32 v89, v12, 16, 1
	v_add3_u32 v86, v14, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cmp_o_f32_e32 vcc, v15, v15
	v_add3_u32 v92, v12, v89, s0
	v_add3_u32 v96, v13, v88, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[112:115], v[144:147], v[72:75], v87, v94 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshl_add_u32 v82, v80, 1, v81
	s_nop 1
	v_cndmask_b32_e32 v72, v83, v85, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_lshrrev_b32_e32 v73, 16, v96
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[106:109], v[152:155], v[68:71], v87, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v74, 16, v92
	s_nop 1
	v_cndmask_b32_e32 v68, v83, v86, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[106:109], v[128:131], v[76:79], v87, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v69, v83, v73, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	s_waitcnt vmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[106:109], v[136:139], v[64:67], v87, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v70, v83, v74, vcc
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[112:115], v[160:163], v[12:15], v87, v95 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[118:121], v[124:127], v[60:63], v87, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[164:167], v[132:135], v[12:15], v87, v117 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[118:121], v[128:131], v[56:59], v87, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[112:115], v[140:143], v[76:79], v87, v116 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 1
	v_bfe_u32 v59, v91, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[112:115], v[156:159], v[96:99], v87, v93 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[112:115], v[164:167], v[140:143], v[12:15], v87, v116 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 2
	v_bfe_u32 v71, v79, 16, 1
	v_bfe_u32 v73, v78, 16, 1
	v_add3_u32 v60, v79, v71, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[118:121], v[148:151], v[52:55], v87, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v74, v77, 16, 1
	v_add3_u32 v73, v78, v73, s0
	v_lshrrev_b32_e32 v60, 16, v60
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[164:167], v[144:147], v[12:15], v87, v94 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v79, v79
	v_bfe_u32 v75, v76, 16, 1
	v_add3_u32 v74, v77, v74, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[118:121], v[152:155], v[48:51], v87, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v61, 16, v73
	v_cndmask_b32_e32 v60, v83, v60, vcc
	v_cmp_o_f32_e32 vcc, v78, v78
	v_add3_u32 v75, v76, v75, s0
	v_lshrrev_b32_e32 v62, 16, v74
	v_cndmask_b32_e32 v56, v83, v61, vcc
	v_cmp_o_f32_e32 vcc, v77, v77
	v_lshrrev_b32_e32 v63, 16, v75
	v_bfe_u32 v61, v90, 16, 1
	v_cndmask_b32_e32 v57, v83, v62, vcc
	v_cmp_o_f32_e32 vcc, v76, v76
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[164:167], v[156:159], v[12:15], v87, v93 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v48, v91, v59, s0
	v_cndmask_b32_e32 v58, v83, v63, vcc
	v_bfe_u32 v62, v89, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[118:121], v[136:139], v[44:47], v87, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v61, v90, v61, s0
	v_lshrrev_b32_e32 v48, 16, v48
	v_cmp_o_f32_e32 vcc, v91, v91
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[164:167], v[160:163], v[12:15], v87, v95 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v63, v88, 16, 1
	v_add3_u32 v62, v89, v62, s0
	v_lshrrev_b32_e32 v49, 16, v61
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[124:127], v[40:43], v84, v123 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v59, v83, v48, vcc
	v_cmp_o_f32_e32 vcc, v90, v90
	v_add3_u32 v63, v88, v63, s0
	v_lshrrev_b32_e32 v50, 16, v62
	v_cndmask_b32_e32 v61, v83, v49, vcc
	v_cmp_o_f32_e32 vcc, v89, v89
	v_lshrrev_b32_e32 v51, 16, v63
	v_bfe_u32 v44, v99, 16, 1
	v_cndmask_b32_e32 v62, v83, v50, vcc
	v_cmp_o_f32_e32 vcc, v88, v88
	v_bfe_u32 v40, v98, 16, 1
	v_bfe_u32 v41, v97, 16, 1
	v_cndmask_b32_e32 v63, v83, v51, vcc
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[176:179], v[132:135], v[12:15], v84, v117 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v42, v96, 16, 1
	v_add3_u32 v42, v96, v42, s0
	v_add3_u32 v41, v97, v41, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[128:131], v[32:35], v84, v122 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v40, v98, v40, s0
	v_lshrrev_b32_e32 v71, 16, v41
	v_lshrrev_b32_e32 v73, 16, v42
	v_add3_u32 v32, v99, v44, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[176:179], v[140:143], v[12:15], v84, v116 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v33, 16, v40
	v_lshrrev_b32_e32 v32, 16, v32
	v_cmp_o_f32_e32 vcc, v99, v99
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[148:151], v[20:23], v84, v105 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v85, v50, 16, 1
	v_cndmask_b32_e32 v78, v83, v32, vcc
	v_cmp_o_f32_e32 vcc, v98, v98
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[176:179], v[144:147], v[12:15], v84, v94 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v86, v49, 16, 1
	v_cndmask_b32_e32 v79, v83, v33, vcc
	v_cmp_o_f32_e32 vcc, v97, v97
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[152:155], v[16:19], v84, v101 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v85, v50, v85, s0
	v_bfe_u32 v87, v48, 16, 1
	v_add3_u32 v86, v49, v86, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[176:179], v[156:159], v[12:15], v84, v93 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v85, 16, v85
	v_add3_u32 v87, v48, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[136:139], v[24:27], v84, v110 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v87, 16, v87
	s_nop 1
	v_bfe_u32 v26, v67, 16, 1
	v_cndmask_b32_e32 v24, v83, v71, vcc
	v_cmp_o_f32_e32 vcc, v96, v96
	v_bfe_u32 v27, v66, 16, 1
	v_add3_u32 v26, v67, v26, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[176:179], v[160:163], v[12:15], v84, v95 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v25, v83, v73, vcc
	v_add3_u32 v27, v66, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[172:175], v[124:127], v[28:31], v84, v123 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v67, v67
	v_lshrrev_b32_e32 v27, 16, v27
	v_bfe_u32 v67, v54, 16, 1
	v_bfe_u32 v28, v65, 16, 1
	v_bfe_u32 v29, v64, 16, 1
	v_add3_u32 v28, v65, v28, s0
	v_cndmask_b32_e32 v26, v83, v26, vcc
	v_cmp_o_f32_e32 vcc, v66, v66
	v_add3_u32 v29, v64, v29, s0
	v_lshrrev_b32_e32 v28, 16, v28
	v_cndmask_b32_e32 v27, v83, v27, vcc
	v_cmp_o_f32_e32 vcc, v65, v65
	v_bfe_u32 v30, v109, 16, 1
	v_lshrrev_b32_e32 v29, 16, v29
	v_cndmask_b32_e32 v28, v83, v28, vcc
	v_cmp_o_f32_e32 vcc, v64, v64
	v_bfe_u32 v31, v108, 16, 1
	v_add3_u32 v30, v109, v30, s0
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[180:183], v[132:135], v[12:15], v84, v117 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v29, v83, v29, vcc
	v_add3_u32 v31, v108, v31, s0
	v_lshrrev_b32_e32 v30, 16, v30
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[172:175], v[128:131], v[36:39], v84, v122 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v109, v109
	v_lshrrev_b32_e32 v31, 16, v31
	v_bfe_u32 v64, v113, 16, 1
	v_bfe_u32 v36, v107, 16, 1
	v_bfe_u32 v37, v106, 16, 1
	v_add3_u32 v36, v107, v36, s0
	v_cndmask_b32_e32 v30, v83, v30, vcc
	v_cmp_o_f32_e32 vcc, v108, v108
	v_add3_u32 v37, v106, v37, s0
	v_lshrrev_b32_e32 v36, 16, v36
	v_cndmask_b32_e32 v31, v83, v31, vcc
	v_cmp_o_f32_e32 vcc, v107, v107
	v_bfe_u32 v38, v115, 16, 1
	v_lshrrev_b32_e32 v37, 16, v37
	v_cndmask_b32_e32 v36, v83, v36, vcc
	v_cmp_o_f32_e32 vcc, v106, v106
	v_bfe_u32 v39, v114, 16, 1
	v_add3_u32 v38, v115, v38, s0
	v_cndmask_b32_e32 v37, v83, v37, vcc
	v_add3_u32 v39, v114, v39, s0
	v_lshrrev_b32_e32 v38, 16, v38
	v_cmp_o_f32_e32 vcc, v115, v115
	v_bfe_u32 v65, v112, 16, 1
	v_add3_u32 v64, v113, v64, s0
	v_lshrrev_b32_e32 v39, 16, v39
	v_cndmask_b32_e32 v38, v83, v38, vcc
	v_cmp_o_f32_e32 vcc, v114, v114
	v_add3_u32 v65, v112, v65, s0
	v_lshrrev_b32_e32 v64, 16, v64
	v_cndmask_b32_e32 v39, v83, v39, vcc
	v_cmp_o_f32_e32 vcc, v113, v113
	v_bfe_u32 v66, v55, 16, 1
	v_lshrrev_b32_e32 v65, 16, v65
	v_cndmask_b32_e32 v64, v83, v64, vcc
	v_cmp_o_f32_e32 vcc, v112, v112
	v_add3_u32 v66, v55, v66, s0
	v_bfe_u32 v71, v53, 16, 1
	v_cndmask_b32_e32 v65, v83, v65, vcc
	v_add3_u32 v67, v54, v67, s0
	v_lshrrev_b32_e32 v66, 16, v66
	v_cmp_o_f32_e32 vcc, v55, v55
	v_bfe_u32 v73, v52, 16, 1
	v_add3_u32 v71, v53, v71, s0
	v_lshrrev_b32_e32 v67, 16, v67
	v_cndmask_b32_e32 v55, v83, v66, vcc
	v_cmp_o_f32_e32 vcc, v54, v54
	v_add3_u32 v73, v52, v73, s0
	v_lshrrev_b32_e32 v71, 16, v71
	v_cndmask_b32_e32 v54, v83, v67, vcc
	v_cmp_o_f32_e32 vcc, v53, v53
	v_bfe_u32 v66, v77, 16, 1
	v_lshrrev_b32_e32 v73, 16, v73
	v_cndmask_b32_e32 v53, v83, v71, vcc
	v_cmp_o_f32_e32 vcc, v52, v52
	v_bfe_u32 v67, v76, 16, 1
	v_add3_u32 v66, v77, v66, s0
	v_cndmask_b32_e32 v52, v83, v73, vcc
	v_bfe_u32 v71, v75, 16, 1
	v_add3_u32 v67, v76, v67, s0
	v_lshrrev_b32_e32 v66, 16, v66
	v_cmp_o_f32_e32 vcc, v77, v77
	v_add3_u32 v71, v75, v71, s0
	v_lshrrev_b32_e32 v67, 16, v67
	v_cndmask_b32_e32 v66, v83, v66, vcc
	v_cmp_o_f32_e32 vcc, v76, v76
	v_bfe_u32 v73, v74, 16, 1
	v_lshrrev_b32_e32 v71, 16, v71
	v_cndmask_b32_e32 v67, v83, v67, vcc
	v_cmp_o_f32_e32 vcc, v75, v75
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[172:175], v[148:151], v[8:11], v84, v105 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v73, v74, v73, s0
	v_cndmask_b32_e32 v71, v83, v71, vcc
	v_cmp_o_f32_e32 vcc, v74, v74
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[172:175], v[152:155], v[4:7], v84, v101 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v74, v121, 16, 1
	v_lshrrev_b32_e32 v73, 16, v73
	v_bfe_u32 v75, v120, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[172:175], v[136:139], v[0:3], v84, v110 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v74, v121, v74, s0
	v_cndmask_b32_e32 v73, v83, v73, vcc
	v_bfe_u32 v76, v119, 16, 1
	v_add3_u32 v75, v120, v75, s0
	v_lshrrev_b32_e32 v74, 16, v74
	v_cmp_o_f32_e32 vcc, v121, v121
	v_bfe_u32 v77, v118, 16, 1
	v_add3_u32 v76, v119, v76, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cndmask_b32_e32 v74, v83, v74, vcc
	v_cmp_o_f32_e32 vcc, v120, v120
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[180:183], v[140:143], v[12:15], v84, v116 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v77, v118, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cndmask_b32_e32 v75, v83, v75, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[180:183], v[144:147], v[8:11], v84, v94 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v119, v119
	v_lshrrev_b32_e32 v77, 16, v77
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[180:183], v[156:159], v[4:7], v84, v93 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v76, v83, v76, vcc
	v_cmp_o_f32_e32 vcc, v118, v118
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[180:183], v[160:163], v[0:3], v84, v95 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v84, v51, 16, 1
	v_add3_u32 v84, v51, v84, s0
	v_cndmask_b32_e32 v77, v83, v77, vcc
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v51, v51
	s_nop 1
	v_cndmask_b32_e32 v51, v83, v84, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_bfe_u32 v84, v47, 16, 1
	v_add3_u32 v84, v47, v84, s0
	v_cndmask_b32_e32 v50, v83, v85, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_bfe_u32 v85, v46, 16, 1
	v_add3_u32 v85, v46, v85, s0
	v_cndmask_b32_e32 v49, v83, v86, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_bfe_u32 v86, v45, 16, 1
	v_lshrrev_b32_e32 v84, 16, v84
	v_cndmask_b32_e32 v48, v83, v87, vcc
	v_cmp_o_f32_e32 vcc, v47, v47
	v_bfe_u32 v87, v44, 16, 1
	v_add3_u32 v86, v45, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v47, v83, v84, vcc
	v_cmp_o_f32_e32 vcc, v46, v46
	v_add3_u32 v87, v44, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v46, v83, v85, vcc
	v_cmp_o_f32_e32 vcc, v45, v45
	v_bfe_u32 v84, v43, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v45, v83, v86, vcc
	v_cmp_o_f32_e32 vcc, v44, v44
	v_bfe_u32 v85, v42, 16, 1
	v_add3_u32 v84, v43, v84, s0
	v_cndmask_b32_e32 v44, v83, v87, vcc
	v_bfe_u32 v86, v41, 16, 1
	v_add3_u32 v85, v42, v85, s0
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v43, v43
	v_bfe_u32 v87, v40, 16, 1
	v_add3_u32 v86, v41, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v43, v83, v84, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	v_add3_u32 v87, v40, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v42, v83, v85, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	v_bfe_u32 v84, v35, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v41, v83, v86, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_bfe_u32 v85, v34, 16, 1
	v_add3_u32 v84, v35, v84, s0
	v_cndmask_b32_e32 v40, v83, v87, vcc
	v_bfe_u32 v86, v33, 16, 1
	v_add3_u32 v85, v34, v85, s0
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v35, v35
	v_bfe_u32 v87, v32, 16, 1
	v_add3_u32 v86, v33, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v35, v83, v84, vcc
	v_cmp_o_f32_e32 vcc, v34, v34
	v_add3_u32 v87, v32, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v34, v83, v85, vcc
	v_cmp_o_f32_e32 vcc, v33, v33
	v_bfe_u32 v84, v23, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v33, v83, v86, vcc
	v_cmp_o_f32_e32 vcc, v32, v32
	v_bfe_u32 v85, v22, 16, 1
	v_add3_u32 v84, v23, v84, s0
	v_cndmask_b32_e32 v32, v83, v87, vcc
	v_bfe_u32 v86, v21, 16, 1
	v_add3_u32 v85, v22, v85, s0
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v23, v23
	v_bfe_u32 v87, v20, 16, 1
	v_add3_u32 v86, v21, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v23, v83, v84, vcc
	v_cmp_o_f32_e32 vcc, v22, v22
	v_add3_u32 v87, v20, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v22, v83, v85, vcc
	v_cmp_o_f32_e32 vcc, v21, v21
	v_bfe_u32 v84, v19, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v21, v83, v86, vcc
	v_cmp_o_f32_e32 vcc, v20, v20
	v_bfe_u32 v85, v18, 16, 1
	v_add3_u32 v84, v19, v84, s0
	v_cndmask_b32_e32 v20, v83, v87, vcc
	v_bfe_u32 v86, v17, 16, 1
	v_add3_u32 v85, v18, v85, s0
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v19, v19
	v_bfe_u32 v87, v16, 16, 1
	v_add3_u32 v86, v17, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v19, v83, v84, vcc
	v_cmp_o_f32_e32 vcc, v18, v18
	v_add3_u32 v87, v16, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v18, v83, v85, vcc
	v_cmp_o_f32_e32 vcc, v17, v17
	v_bfe_u32 v84, v15, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v17, v83, v86, vcc
	v_cmp_o_f32_e32 vcc, v16, v16
	v_bfe_u32 v85, v14, 16, 1
	v_add3_u32 v84, v15, v84, s0
	v_cndmask_b32_e32 v16, v83, v87, vcc
	v_bfe_u32 v86, v13, 16, 1
	v_add3_u32 v85, v14, v85, s0
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v15, v15
	v_bfe_u32 v87, v12, 16, 1
	v_add3_u32 v86, v13, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v15, v83, v84, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_add3_u32 v87, v12, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v14, v83, v85, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_bfe_u32 v84, v11, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v13, v83, v86, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v85, v10, 16, 1
	v_add3_u32 v84, v11, v84, s0
	v_cndmask_b32_e32 v12, v83, v87, vcc
	v_bfe_u32 v86, v9, 16, 1
	v_add3_u32 v85, v10, v85, s0
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v11, v11
	v_bfe_u32 v87, v8, 16, 1
	v_add3_u32 v86, v9, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v11, v83, v84, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_add3_u32 v87, v8, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v10, v83, v85, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_bfe_u32 v84, v7, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v9, v83, v86, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v85, v6, 16, 1
	v_add3_u32 v84, v7, v84, s0
	v_cndmask_b32_e32 v8, v83, v87, vcc
	v_bfe_u32 v86, v5, 16, 1
	v_add3_u32 v85, v6, v85, s0
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v87, v4, 16, 1
	v_add3_u32 v86, v5, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v7, v83, v84, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v87, v4, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v6, v83, v85, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_lshrrev_b32_e32 v87, 16, v87
	v_bfe_u32 v84, v3, 16, 1
	v_cndmask_b32_e32 v5, v83, v86, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v85, v2, 16, 1
	v_bfe_u32 v86, v1, 16, 1
	v_cndmask_b32_e32 v4, v83, v87, vcc
	v_bfe_u32 v87, v0, 16, 1
	v_add3_u32 v87, v0, v87, s0
	v_add3_u32 v86, v1, v86, s0
	v_add3_u32 v85, v2, v85, s0
	v_add3_u32 v84, v3, v84, s0
	s_mul_i32 s0, s9, s28
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s8, s28
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s10, s0
	s_addc_u32 s1, s11, s1
	s_lshl_b32 s2, s29, 1
	s_add_u32 s0, s0, s2
	s_addc_u32 s1, s1, 0
	s_and_b32 s2, s8, 0x3fff
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	s_or_b32 s1, s2, s1
	s_or_b32 s1, s1, 2.0
	s_mov_b32 s2, 0x7ffffffd
	s_lshl_b32 s4, s8, 1
	buffer_store_short v70, v82, s[0:3], 0 offen
	v_add_u32_e32 v70, s4, v82
	buffer_store_short v69, v70, s[0:3], 0 offen
	v_add_u32_e32 v69, s4, v70
	buffer_store_short v68, v69, s[0:3], 0 offen
	v_add_u32_e32 v68, s4, v69
	s_lshl_b32 s5, s8, 4
	buffer_store_short v72, v68, s[0:3], 0 offen
	buffer_store_short v58, v82, s[0:3], 0 offen offset:32
	buffer_store_short v57, v70, s[0:3], 0 offen offset:32
	buffer_store_short v56, v69, s[0:3], 0 offen offset:32
	buffer_store_short v60, v68, s[0:3], 0 offen offset:32
	buffer_store_short v63, v82, s[0:3], 0 offen offset:64
	buffer_store_short v62, v70, s[0:3], 0 offen offset:64
	buffer_store_short v61, v69, s[0:3], 0 offen offset:64
	buffer_store_short v59, v68, s[0:3], 0 offen offset:64
	buffer_store_short v25, v82, s[0:3], 0 offen offset:96
	buffer_store_short v24, v70, s[0:3], 0 offen offset:96
	buffer_store_short v79, v69, s[0:3], 0 offen offset:96
	buffer_store_short v78, v68, s[0:3], 0 offen offset:96
	buffer_store_short v29, v82, s[0:3], 0 offen offset:128
	buffer_store_short v28, v70, s[0:3], 0 offen offset:128
	buffer_store_short v27, v69, s[0:3], 0 offen offset:128
	buffer_store_short v26, v68, s[0:3], 0 offen offset:128
	v_add_u32_e32 v24, s5, v80
	v_lshl_add_u32 v25, v24, 1, v81
	v_add_u32_e32 v26, s4, v25
	v_add_u32_e32 v27, s4, v26
	v_add_u32_e32 v28, s4, v27
	v_add_u32_e32 v24, s5, v24
	buffer_store_short v37, v25, s[0:3], 0 offen
	buffer_store_short v36, v26, s[0:3], 0 offen
	buffer_store_short v31, v27, s[0:3], 0 offen
	buffer_store_short v30, v28, s[0:3], 0 offen
	buffer_store_short v65, v25, s[0:3], 0 offen offset:32
	buffer_store_short v64, v26, s[0:3], 0 offen offset:32
	buffer_store_short v39, v27, s[0:3], 0 offen offset:32
	buffer_store_short v38, v28, s[0:3], 0 offen offset:32
	buffer_store_short v52, v25, s[0:3], 0 offen offset:64
	buffer_store_short v53, v26, s[0:3], 0 offen offset:64
	buffer_store_short v54, v27, s[0:3], 0 offen offset:64
	buffer_store_short v55, v28, s[0:3], 0 offen offset:64
	buffer_store_short v73, v25, s[0:3], 0 offen offset:96
	buffer_store_short v71, v26, s[0:3], 0 offen offset:96
	buffer_store_short v67, v27, s[0:3], 0 offen offset:96
	buffer_store_short v66, v28, s[0:3], 0 offen offset:96
	buffer_store_short v77, v25, s[0:3], 0 offen offset:128
	buffer_store_short v76, v26, s[0:3], 0 offen offset:128
	buffer_store_short v75, v27, s[0:3], 0 offen offset:128
	buffer_store_short v74, v28, s[0:3], 0 offen offset:128
	v_lshl_add_u32 v25, v24, 1, v81
	v_add_u32_e32 v26, s4, v25
	v_add_u32_e32 v27, s4, v26
	v_add_u32_e32 v28, s4, v27
	buffer_store_short v48, v25, s[0:3], 0 offen
	buffer_store_short v49, v26, s[0:3], 0 offen
	buffer_store_short v50, v27, s[0:3], 0 offen
	buffer_store_short v51, v28, s[0:3], 0 offen
	buffer_store_short v44, v25, s[0:3], 0 offen offset:32
	buffer_store_short v45, v26, s[0:3], 0 offen offset:32
	buffer_store_short v46, v27, s[0:3], 0 offen offset:32
	buffer_store_short v47, v28, s[0:3], 0 offen offset:32
	buffer_store_short v40, v25, s[0:3], 0 offen offset:64
	buffer_store_short v41, v26, s[0:3], 0 offen offset:64
	buffer_store_short v42, v27, s[0:3], 0 offen offset:64
	buffer_store_short v43, v28, s[0:3], 0 offen offset:64
	buffer_store_short v32, v25, s[0:3], 0 offen offset:96
	buffer_store_short v33, v26, s[0:3], 0 offen offset:96
	buffer_store_short v34, v27, s[0:3], 0 offen offset:96
	buffer_store_short v35, v28, s[0:3], 0 offen offset:96
	buffer_store_short v20, v25, s[0:3], 0 offen offset:128
	buffer_store_short v21, v26, s[0:3], 0 offen offset:128
	buffer_store_short v22, v27, s[0:3], 0 offen offset:128
	buffer_store_short v23, v28, s[0:3], 0 offen offset:128
	v_add_u32_e32 v20, s5, v24
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v3, v3
	v_lshl_add_u32 v20, v20, 1, v81
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v3, v83, v84, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	buffer_store_short v16, v20, s[0:3], 0 offen
	v_add_u32_e32 v16, s4, v20
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v2, v83, v85, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	buffer_store_short v17, v16, s[0:3], 0 offen
	v_add_u32_e32 v17, s4, v16
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v1, v83, v86, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	buffer_store_short v18, v17, s[0:3], 0 offen
	v_add_u32_e32 v18, s4, v17
	v_cndmask_b32_e32 v0, v83, v87, vcc
	buffer_store_short v19, v18, s[0:3], 0 offen
	buffer_store_short v12, v20, s[0:3], 0 offen offset:32
	buffer_store_short v13, v16, s[0:3], 0 offen offset:32
	buffer_store_short v14, v17, s[0:3], 0 offen offset:32
	buffer_store_short v15, v18, s[0:3], 0 offen offset:32
	buffer_store_short v8, v20, s[0:3], 0 offen offset:64
	buffer_store_short v9, v16, s[0:3], 0 offen offset:64
	buffer_store_short v10, v17, s[0:3], 0 offen offset:64
	buffer_store_short v11, v18, s[0:3], 0 offen offset:64
	buffer_store_short v4, v20, s[0:3], 0 offen offset:96
	buffer_store_short v5, v16, s[0:3], 0 offen offset:96
	buffer_store_short v6, v17, s[0:3], 0 offen offset:96
	buffer_store_short v7, v18, s[0:3], 0 offen offset:96
	buffer_store_short v0, v20, s[0:3], 0 offen offset:128
	buffer_store_short v1, v16, s[0:3], 0 offen offset:128
	buffer_store_short v2, v17, s[0:3], 0 offen offset:128
	buffer_store_short v3, v18, s[0:3], 0 offen offset:128
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_256x160x256_168448x3200x6400
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
		.amdhsa_next_free_vgpr 216
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 216
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
	.size	wave_mxfp4_static_gemm_256x160x256_168448x3200x6400, .Lfunc_end0-wave_mxfp4_static_gemm_256x160x256_168448x3200x6400

	.set wave_mxfp4_static_gemm_256x160x256_168448x3200x6400.num_vgpr, 216
	.set wave_mxfp4_static_gemm_256x160x256_168448x3200x6400.num_agpr, 0
	.set wave_mxfp4_static_gemm_256x160x256_168448x3200x6400.numbered_sgpr, 67
	.set wave_mxfp4_static_gemm_256x160x256_168448x3200x6400.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_256x160x256_168448x3200x6400.private_seg_size, 0
	.set wave_mxfp4_static_gemm_256x160x256_168448x3200x6400.uses_vcc, 1
	.set wave_mxfp4_static_gemm_256x160x256_168448x3200x6400.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_256x160x256_168448x3200x6400.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_256x160x256_168448x3200x6400.has_recursion, 0
	.set wave_mxfp4_static_gemm_256x160x256_168448x3200x6400.has_indirect_call, 0
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
    .name:           wave_mxfp4_static_gemm_256x160x256_168448x3200x6400
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     73
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_256x160x256_168448x3200x6400.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     216
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
