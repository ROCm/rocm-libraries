; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_256x160x256_1536x3200x10496
	.p2align	8
	.type	wave_mxfp4_static_gemm_256x160x256_1536x3200x10496,@function
wave_mxfp4_static_gemm_256x160x256_1536x3200x10496:
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
	v_mul_u32_u24_e32 v2, 0x1480, v2
	s_and_b32 s6, s25, 0xffff
	s_lshl_b32 s30, s2, 7
	v_ashrrev_i32_e32 v10, 31, v9
	s_or_b32 s25, s6, 0x54800000
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_or_b32_e32 v5, v2, v1
	s_mov_b32 m0, s30
	s_or_b32 s31, s30, 0x2000
	v_xor_b32_e32 v9, v10, v9
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_add_u32_e32 v6, 0x52000, v5
	s_mov_b32 m0, s31
	s_or_b32 s33, s30, 0x4000
	v_ashrrev_i32_e32 v11, 31, v9
	buffer_load_dwordx4 v6, s[24:27], 0 offen lds
	v_add_u32_e32 v6, 0xa4000, v5
	s_mov_b32 m0, s33
	s_or_b32 s34, s30, 0x6000
	v_lshrrev_b32_e32 v11, 29, v11
	buffer_load_dwordx4 v6, s[24:27], 0 offen lds
	v_add_u32_e32 v5, 0xf6000, v5
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
	v_add_u32_e32 v22, v0, v83
	v_lshlrev_b32_e32 v6, 2, v7
	v_add_u32_e32 v12, v10, v11
	v_add_u32_e32 v14, v22, v5
	v_lshlrev_b32_e32 v9, 7, v15
	v_sub_u32_e32 v17, v6, v12
	v_add3_u32 v14, v17, v14, v9
	v_ashrrev_i32_e32 v17, 31, v14
	v_xor_b32_e32 v14, v17, v14
	s_mov_b32 s35, 0x63e7063f
	v_mul_hi_i32 v14, v14, s35
	v_lshrrev_b32_e32 v18, 31, v14
	v_ashrrev_i32_e32 v14, 9, v14
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
	v_mul_i32_i24_e32 v17, 0xffffeb80, v17
	v_lshlrev_b32_e32 v16, 7, v3
	s_lshl_b32 s49, s2, 7
	s_movk_i32 s3, 0x1480
	v_add3_u32 v17, v19, v16, v17
	s_and_b32 s6, s21, 0xffff
	s_add_i32 s36, s49, 0x10000
	s_or_b32 s50, s49, 0x800
	s_or_b32 s21, s6, 0x54800000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_mad_i32_i24 v17, v18, s3, v17
	s_mov_b32 m0, s36
	s_add_i32 s37, s50, 0x10000
	s_or_b32 s51, s49, 0x1000
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x14800, v17
	s_mov_b32 m0, s37
	s_add_i32 s38, s51, 0x10000
	s_or_b32 s52, s49, 0x1800
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x29000, v17
	s_mov_b32 m0, s38
	s_add_i32 s39, s52, 0x10000
	s_or_b32 s53, s49, 0x2000
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x3d800, v17
	s_mov_b32 m0, s39
	s_add_i32 s40, s53, 0x10000
	s_or_b32 s54, s49, 0x2800
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x52000, v17
	s_mov_b32 m0, s40
	s_add_i32 s41, s54, 0x10000
	s_or_b32 s55, s49, 0x3000
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x66800, v17
	s_mov_b32 m0, s41
	s_add_i32 s42, s55, 0x10000
	s_or_b32 s56, s49, 0x3800
	s_or_b32 s57, s49, 0x4000
	s_or_b32 s58, s49, 0x4800
	s_mul_i32 s15, s15, s28
	s_mul_hi_u32 s2, s14, s28
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x7b000, v17
	s_mov_b32 m0, s42
	s_add_i32 s43, s56, 0x10000
	s_add_i32 s44, s57, 0x10000
	s_add_i32 s45, s58, 0x10000
	s_add_i32 s2, s2, s15
	s_mul_i32 s3, s14, s28
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x8f800, v17
	s_mov_b32 m0, s43
	s_add_u32 s4, s4, s3
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0xa4000, v17
	s_mov_b32 m0, s44
	v_lshrrev_b32_e32 v19, 4, v83
	s_movk_i32 s46, 0xffc0
	v_bfe_u32 v81, v83, 4, 2
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s14, 0x3fff
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v18, 6, v81
	v_mad_i32_i24 v23, v19, s46, v85
	s_bitset1_b32 s3, 14
	v_and_b32_e32 v80, 0xc0, v83
	v_add_u32_e32 v26, v23, v18
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	s_or_b32 s5, s2, s3
	v_mad_u64_u32 v[20:21], s[2:3], s14, v80, v[26:27]
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	v_lshl_add_u32 v21, s14, 5, v20
	buffer_load_dword v86, v20, s[4:7], 0 offen
	buffer_load_dword v84, v21, s[4:7], 0 offen
	v_mul_u32_u24_e32 v20, 48, v3
	v_mul_i32_i24_e32 v21, -16, v19
	v_add3_u32 v82, v22, v20, v21
	v_ashrrev_i16_e32 v20, 15, v82
	v_lshrrev_b16_e32 v20, 11, v20
	v_add_u16_e32 v20, v82, v20
	v_and_b32_e32 v20, 0xffffffe0, v20
	v_sub_u16_e32 v20, v82, v20
	v_bfe_i32 v22, v20, 0, 16
	v_ashrrev_i32_e32 v24, 31, v22
	v_add_u16_e32 v25, 32, v20
	v_cmp_gt_i16_e32 vcc, 0, v20
	s_load_dwordx2 s[12:13], s[0:1], 0x40
	s_movk_i32 s2, 0x148
	v_cndmask_b32_e32 v20, v22, v25, vcc
	v_cndmask_b32_e64 v22, v24, 0, vcc
	v_xor_b32_e32 v20, v22, v20
	v_lshrrev_b32_e32 v24, 28, v20
	v_add_u32_e32 v20, v20, v24
	v_ashrrev_i32_e32 v20, 4, v20
	v_xor_b32_e32 v20, v20, v22
	v_add_u32_e32 v24, v26, v20
	v_ashrrev_i32_e32 v22, 31, v24
	v_xor_b32_e32 v25, v22, v24
	v_mul_hi_i32 v25, v25, s35
	v_lshrrev_b32_e32 v27, 31, v25
	v_ashrrev_i32_e32 v25, 7, v25
	v_add_u32_e32 v25, v25, v27
	v_ashrrev_i32_e32 v27, 31, v82
	v_xor_b32_e32 v28, v27, v82
	v_ashrrev_i32_e32 v29, 31, v28
	v_lshrrev_b32_e32 v29, 27, v29
	v_add_u32_e32 v28, v28, v29
	v_lshrrev_b32_e32 v28, 5, v28
	v_xor_b32_e32 v27, v28, v27
	v_lshlrev_b32_e32 v87, 5, v27
	v_xad_u32 v25, v25, v22, v87
	v_mul_hi_i32 v22, v24, s35
	v_lshrrev_b32_e32 v27, 31, v22
	v_ashrrev_i32_e32 v22, 7, v22
	v_add_u32_e32 v22, v22, v27
	v_mul_lo_u32 v22, v22, s2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s13, s29
	s_mul_hi_u32 s3, s12, s29
	s_add_i32 s3, s3, s2
	s_mul_i32 s2, s12, s29
	s_add_u32 s16, s8, s2
	v_sub_u32_e32 v22, v24, v22
	s_addc_u32 s2, s9, s3
	s_and_b32 s3, s12, 0x3fff
	v_add_u32_e32 v27, 0x148, v22
	v_cmp_gt_i32_e32 vcc, 0, v22
	s_bitset1_b32 s3, 14
	s_and_b32 s2, s2, 0xffff
	v_cndmask_b32_e32 v22, v22, v27, vcc
	s_lshl_b32 s3, s3, 16
	s_or_b32 s17, s2, s3
	v_mad_u64_u32 v[28:29], s[2:3], s12, v25, v[22:23]
	v_add_u32_e32 v22, 2, v24
	v_sub_u32_e32 v25, -3, v24
	v_cmp_gt_i32_e32 vcc, -2, v24
	v_add3_u32 v23, v20, v18, v23
	v_add_u32_e32 v17, 0xb8800, v17
	v_cndmask_b32_e32 v22, v22, v25, vcc
	v_mul_hi_i32 v22, v22, s35
	v_lshrrev_b32_e32 v24, 31, v22
	v_ashrrev_i32_e32 v22, 7, v22
	v_add_u32_e32 v22, v22, v24
	v_cndmask_b32_e64 v24, 0, -1, vcc
	v_xor_b32_e32 v22, v22, v24
	v_add_u32_e32 v24, v22, v87
	v_mul_i32_i24_e32 v22, 0xfffffeb8, v22
	v_mul_lo_u32 v24, s12, v24
	v_add3_u32 v25, v23, v22, v24
	v_add_u32_e32 v23, 16, v82
	v_sub_u32_e32 v24, 0xffef, v82
	v_cmp_gt_i32_e32 vcc, -16, v82
	s_mov_b32 m0, s45
	v_mad_u32_u24 v27, v3, 5, v85
	v_cndmask_b32_e32 v23, v23, v24, vcc
	v_ashrrev_i16_e32 v24, 15, v23
	v_lshrrev_b16_e32 v24, 11, v24
	v_add_u16_e32 v23, v23, v24
	v_ashrrev_i16_e32 v23, 5, v23
	v_cndmask_b32_e64 v24, 0, -1, vcc
	v_xor_b32_e32 v23, v23, v24
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_mul_i32_i24_e32 v17, 0xffffffc0, v19
	v_bfe_i32 v29, v23, 0, 16
	v_mad_i32_i24 v24, v29, -2, v17
	v_add3_u32 v27, v24, v27, v18
	v_add_u32_e32 v30, 1, v27
	v_sub_u32_e32 v31, -2, v27
	v_cmp_gt_i32_e32 vcc, -1, v27
	v_mul_u32_u24_e32 v22, 5, v3
	v_mul_i32_i24_e32 v23, -2, v29
	v_cndmask_b32_e32 v30, v30, v31, vcc
	v_mul_hi_i32 v30, v30, s35
	v_lshrrev_b32_e32 v31, 31, v30
	v_ashrrev_i32_e32 v30, 7, v30
	v_add_u32_e32 v30, v30, v31
	v_cndmask_b32_e64 v31, 0, -1, vcc
	s_movk_i32 s13, 0xfeb8
	v_xor_b32_e32 v33, v30, v31
	v_lshlrev_b32_e32 v88, 5, v29
	v_add3_u32 v32, v23, v22, v26
	v_add_u32_e32 v29, v33, v88
	v_mad_i32_i24 v26, v33, s13, v32
	v_mad_u64_u32 v[30:31], s[2:3], v29, s12, v[26:27]
	v_add_u32_e32 v29, 3, v27
	v_sub_u32_e32 v31, -4, v27
	v_cmp_gt_i32_e32 vcc, -3, v27
	v_sub_u32_e32 v36, 0xffcf, v82
	s_mov_b32 s18, s26
	v_cndmask_b32_e32 v27, v29, v31, vcc
	v_mul_hi_i32 v27, v27, s35
	v_lshrrev_b32_e32 v29, 31, v27
	v_ashrrev_i32_e32 v27, 7, v27
	v_add_u32_e32 v27, v27, v29
	v_cndmask_b32_e64 v29, 0, -1, vcc
	v_xor_b32_e32 v29, v27, v29
	v_add_u32_e32 v27, v29, v88
	v_mad_i32_i24 v32, v29, s13, v32
	v_mad_u64_u32 v[34:35], s[2:3], v27, s12, v[32:33]
	s_movk_i32 s2, 0xffd0
	v_add_u32_e32 v27, 48, v82
	v_cmp_gt_i32_e32 vcc, s2, v82
	s_mov_b32 s19, s27
	s_lshl_b32 s6, s12, 5
	v_cndmask_b32_e32 v27, v27, v36, vcc
	v_ashrrev_i16_e32 v36, 15, v27
	v_lshrrev_b16_e32 v36, 11, v36
	v_add_u16_e32 v27, v27, v36
	v_ashrrev_i16_e32 v27, 5, v27
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_xor_b32_e32 v27, v27, v36
	v_mov_b32_e32 v36, 5
	v_lshlrev_b32_sdwa v92, v36, sext(v27) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_e32 v27, v92, v33
	v_mad_u64_u32 v[26:27], s[2:3], s12, v27, v[26:27]
	v_add_u32_e32 v27, v92, v29
	v_add_u32_e32 v31, s6, v28
	v_add_u32_e32 v35, s6, v25
	v_mad_u64_u32 v[32:33], s[2:3], s12, v27, v[32:33]
	buffer_load_ubyte v119, v28, s[16:19], 0 offen
	buffer_load_ubyte v113, v25, s[16:19], 0 offen offset:2
	buffer_load_ubyte v118, v30, s[16:19], 0 offen offset:1
	buffer_load_ubyte v112, v34, s[16:19], 0 offen offset:3
	buffer_load_ubyte v96, v31, s[16:19], 0 offen
	buffer_load_ubyte v90, v35, s[16:19], 0 offen offset:2
	buffer_load_ubyte v95, v26, s[16:19], 0 offen offset:1
	buffer_load_ubyte v89, v32, s[16:19], 0 offen offset:3
	v_add_u32_e32 v25, s6, v31
	v_add_u32_e32 v26, s6, v35
	buffer_load_ubyte v99, v25, s[16:19], 0 offen
	buffer_load_ubyte v91, v26, s[16:19], 0 offen offset:2
	v_cmp_eq_u32_e64 s[2:3], 0, v3
	s_movk_i32 s6, 0x2800
	s_mov_b32 s15, -2
	s_and_b64 vcc, exec, s[2:3]
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	v_lshlrev_b32_e32 v27, 7, v83
	v_lshlrev_b32_e32 v19, 11, v19
	v_and_b32_e32 v25, 7, v83
	v_sub_u32_e32 v19, v27, v19
	v_mul_lo_u32 v3, v3, s6
	v_bitop3_b32 v26, v81, v83, 7 bitop3:0x78
	v_lshl_add_u32 v4, v4, 13, v19
	v_add_u32_e32 v3, v19, v3
	v_bitop3_b32 v19, v81, v25, 4 bitop3:0x36
	v_lshlrev_b32_e32 v26, 4, v26
	v_lshlrev_b32_e32 v19, 4, v19
	v_or_b32_e32 v97, v3, v26
	v_or_b32_e32 v98, v19, v3
	v_add3_u32 v3, v14, v15, v16
	v_or_b32_e32 v93, v4, v26
	v_or_b32_e32 v94, v19, v4
	v_lshl_add_u32 v3, v7, 4, v3
	v_lshlrev_b32_e32 v4, 4, v8
	v_sub_u32_e32 v3, v3, v4
	v_sub_u32_e32 v101, v3, v13
	v_sub_u32_e32 v3, v12, v6
	v_sub_u32_e32 v3, v3, v0
	v_sub_u32_e32 v3, v3, v5
	v_sub_u32_e32 v103, v3, v9
	v_add_u32_e32 v3, v9, v5
	v_add3_u32 v0, v3, v0, v6
	s_load_dwordx2 s[8:9], s[0:1], 0x48
	v_sub_u32_e32 v0, v0, v10
	v_sub_u32_e32 v104, v0, v11
	v_add3_u32 v0, v17, v20, v18
	v_sub_u32_e32 v107, 0, v0
	v_add_u32_e32 v108, v17, v18
	v_add_u32_e32 v0, v23, v17
	v_lshl_add_u32 v100, v81, 4, v21
	s_mov_b32 s0, 0xf6100
	v_add_u32_e32 v109, v108, v20
	v_add3_u32 v114, v24, v18, v22
	v_add3_u32 v0, v0, v22, v18
	v_mov_b32_e32 v12, 0
	v_sub_u32_e32 v102, 0, v83
	v_add3_u32 v105, v2, v1, s0
	v_sub_u32_e32 v106, 0, v85
	v_add_u32_e32 v110, 32, v87
	v_add_u32_e32 v111, 64, v87
	v_sub_u32_e32 v115, 0, v0
	v_add_u32_e32 v116, 32, v80
	v_sub_u32_e32 v117, 0, v100
	s_add_i32 s0, s30, 0x8000
	s_add_i32 s1, s31, 0x8000
	s_add_i32 s47, s33, 0x8000
	s_add_i32 s48, s34, 0x8000
	s_add_i32 s49, s49, 0x15000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	s_add_i32 s50, s50, 0x15000
	s_add_i32 s51, s51, 0x15000
	s_add_i32 s52, s52, 0x15000
	s_add_i32 s53, s53, 0x15000
	s_add_i32 s54, s54, 0x15000
	s_add_i32 s55, s55, 0x15000
	s_add_i32 s56, s56, 0x15000
	s_add_i32 s57, s57, 0x15000
	s_add_i32 s58, s58, 0x15000
	v_add_u32_e32 v120, 0x10000, v97
	s_movk_i32 s59, 0xc7cf
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	s_movk_i32 s60, 0xff00
	s_movk_i32 s61, 0xfeff
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	s_movk_i32 s62, 0xfefd
	s_movk_i32 s63, 0xfefe
	v_add_u32_e32 v121, 0x10000, v98
	v_add_u32_e32 v122, 0x15000, v97
	v_add_u32_e32 v123, 0x15000, v98
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
	v_mov_b32_e32 v124, v114
	v_mov_b32_e32 v125, v109
.LBB0_3:
	s_mov_b32 m0, s0
	v_add_u32_e32 v126, 0xfff09f80, v105
	s_waitcnt vmcnt(0)
	s_barrier
	buffer_load_dwordx4 v126, s[24:27], 0 offen lds
	v_add_u32_e32 v126, 0xfff5bf80, v105
	s_mov_b32 m0, s1
	v_add_u32_e32 v128, v102, v103
	buffer_load_dwordx4 v126, s[24:27], 0 offen lds
	v_add_u32_e32 v126, 0xfffadf80, v105
	s_mov_b32 m0, s47
	v_add_u32_e32 v129, 0xfffffdff, v128
	buffer_load_dwordx4 v126, s[24:27], 0 offen lds
	v_add_u32_e32 v126, 0xffffff80, v105
	s_mov_b32 m0, s48
	s_nop 0
	buffer_load_dwordx4 v126, s[24:27], 0 offen lds
	v_add_u32_e32 v126, v83, v104
	v_add_u32_e32 v127, 0x200, v126
	v_cmp_gt_i32_e32 vcc, 0, v127
	s_mov_b32 m0, s49
	s_nop 0
	v_cndmask_b32_e32 v129, v127, v129, vcc
	v_mul_hi_i32 v129, v129, s35
	v_lshrrev_b32_e32 v130, 31, v129
	v_ashrrev_i32_e32 v129, 9, v129
	v_add_u32_e32 v129, v129, v130
	v_ashrrev_i32_e32 v127, 31, v127
	v_xor_b32_e32 v127, v129, v127
	v_add_u32_e32 v129, s29, v127
	v_mul_i32_i24_e32 v127, 0xffffeb80, v127
	v_mul_i32_i24_e32 v129, 0x1480, v129
	v_add_u32_e32 v130, v101, v85
	v_add3_u32 v127, v127, v129, v130
	v_add_u32_e32 v129, 0x800, v127
	buffer_load_dword v129, s[20:23], 0 offen lds
	v_add_u32_e32 v129, 0x15000, v127
	s_mov_b32 m0, s50
	s_nop 0
	buffer_load_dword v129, s[20:23], 0 offen lds
	v_add_u32_e32 v129, 0x29800, v127
	s_mov_b32 m0, s51
	s_nop 0
	buffer_load_dword v129, s[20:23], 0 offen lds
	v_add_u32_e32 v129, 0x3e000, v127
	s_mov_b32 m0, s52
	s_nop 0
	buffer_load_dword v129, s[20:23], 0 offen lds
	v_add_u32_e32 v129, 0x52800, v127
	s_mov_b32 m0, s53
	s_nop 0
	buffer_load_dword v129, s[20:23], 0 offen lds
	v_add_u32_e32 v129, 0x67000, v127
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dword v129, s[20:23], 0 offen lds
	v_add_u32_e32 v129, 0x7b800, v127
	s_mov_b32 m0, s55
	s_nop 0
	buffer_load_dword v129, s[20:23], 0 offen lds
	v_add_u32_e32 v129, 0x90000, v127
	s_mov_b32 m0, s56
	s_nop 0
	buffer_load_dword v129, s[20:23], 0 offen lds
	v_add_u32_e32 v129, 0xa4800, v127
	s_mov_b32 m0, s57
	v_add_u32_e32 v127, 0xb9000, v127
	buffer_load_dword v129, s[20:23], 0 offen lds
	s_mov_b32 m0, s58
	s_nop 0
	buffer_load_dword v127, s[20:23], 0 offen lds
	v_add_u32_e32 v127, v83, v100
	v_add_u32_e32 v131, v102, v117
	v_add_u32_e32 v129, 64, v127
	v_add_u32_e32 v132, 0xffbf, v131
	v_cmp_gt_i32_e32 vcc, s46, v127
	v_add_u32_e32 v136, v106, v107
	v_add_u32_e32 v137, 0xfffffeff, v136
	v_cndmask_b32_e32 v132, v129, v132, vcc
	v_mul_i32_i24_sdwa v133, sext(v132), s59 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_add_u16_sdwa v132, v133, v132 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_lshrrev_b16_e32 v133, 15, v132
	v_ashrrev_i16_e32 v132, 6, v132
	v_add_u16_e32 v132, v132, v133
	v_cndmask_b32_e64 v133, 0, -1, vcc
	v_xor_b32_e32 v132, v132, v133
	v_bfe_i32 v132, v132, 0, 16
	v_add_u32_e32 v133, v80, v132
	v_add_u32_e32 v134, v116, v132
	v_mul_lo_u32 v133, v133, s14
	v_mul_lo_u32 v134, s14, v134
	v_mad_i32_i24 v133, v132, s13, v133
	v_mad_i32_i24 v132, v132, s13, v134
	v_add_u32_e32 v134, v85, v109
	v_add_u32_e32 v135, 0x100, v134
	v_cmp_gt_i32_e32 vcc, s60, v134
	v_add_u32_e32 v141, 0x102, v134
	v_add_u32_e32 v142, 0xfffffefd, v136
	v_cndmask_b32_e32 v137, v135, v137, vcc
	v_mul_hi_i32 v137, v137, s35
	v_lshrrev_b32_e32 v138, 31, v137
	v_ashrrev_i32_e32 v137, 7, v137
	v_add_u32_e32 v137, v137, v138
	v_cndmask_b32_e64 v138, 0, -1, vcc
	v_cmp_gt_i32_e32 vcc, -2, v135
	v_add_u32_e32 v146, v106, v115
	v_add_u32_e32 v147, 0xfffffefe, v146
	v_cndmask_b32_e32 v143, v141, v142, vcc
	v_mul_hi_i32 v143, v143, s35
	v_lshrrev_b32_e32 v144, 31, v143
	v_ashrrev_i32_e32 v143, 7, v143
	v_cmp_gt_i32_e32 vcc, s63, v134
	v_add_u32_e32 v143, v143, v144
	v_ashrrev_i32_e32 v144, 31, v141
	v_cndmask_b32_e32 v141, v141, v142, vcc
	v_mul_hi_i32 v141, v141, s35
	v_xor_b32_e32 v143, v143, v144
	v_lshrrev_b32_e32 v142, 31, v141
	v_lshrrev_b32_e32 v141, 7, v141
	v_add_u32_e32 v144, v143, v87
	v_add_u32_e32 v141, v141, v142
	v_cndmask_b32_e64 v142, 0, -1, vcc
	v_xor_b32_e32 v141, v141, v142
	v_mul_lo_u32 v142, v144, s12
	v_add_u32_e32 v144, v85, v114
	v_add_u32_e32 v145, 0x101, v144
	v_cmp_gt_i32_e32 vcc, s61, v144
	v_add_u32_e32 v149, 0xfffffefc, v146
	v_xor_b32_e32 v137, v137, v138
	v_cndmask_b32_e32 v147, v145, v147, vcc
	v_mul_hi_i32 v147, v147, s35
	v_lshrrev_b32_e32 v148, 31, v147
	v_ashrrev_i32_e32 v147, 7, v147
	v_add_u32_e32 v147, v147, v148
	v_ashrrev_i32_e32 v145, 31, v145
	v_xor_b32_e32 v145, v147, v145
	v_add_u32_e32 v147, v145, v88
	v_add_u32_e32 v148, 0x103, v144
	v_cmp_gt_i32_e32 vcc, s62, v144
	v_add_u32_e32 v155, v145, v92
	v_mul_lo_u32 v147, v147, s12
	v_cndmask_b32_e32 v150, v148, v149, vcc
	v_mul_lo_u32 v155, v155, s12
	v_mad_i32_i24 v147, v145, s13, v147
	v_mul_hi_i32 v150, v150, s35
	v_mad_i32_i24 v145, v145, s13, v155
	v_add_u32_e32 v155, 0x100, v144
	v_lshrrev_b32_e32 v151, 31, v150
	v_ashrrev_i32_e32 v150, 7, v150
	v_cmp_gt_i32_e32 vcc, -3, v155
	v_add_u32_e32 v150, v150, v151
	v_ashrrev_i32_e32 v151, 31, v148
	v_cndmask_b32_e32 v148, v148, v149, vcc
	v_mul_hi_i32 v148, v148, s35
	v_lshrrev_b32_e32 v149, 31, v148
	v_ashrrev_i32_e32 v148, 7, v148
	v_add_u32_e32 v148, v148, v149
	v_xor_b32_e32 v150, v150, v151
	v_xad_u32 v148, v148, v151, v92
	v_add_u32_e32 v138, v137, v87
	v_add_u32_e32 v152, v150, v88
	v_add_u32_e32 v154, v110, v143
	v_mul_lo_u32 v148, v148, s12
	v_mul_i32_i24_e32 v139, 0xfffffeb8, v137
	v_mul_lo_u32 v138, v138, s12
	v_add_u32_e32 v140, v125, v85
	v_mad_i32_i24 v142, v141, s13, v142
	v_mul_lo_u32 v152, v152, s12
	v_add_u32_e32 v153, v110, v137
	v_mul_lo_u32 v154, s12, v154
	v_mad_i32_i24 v148, v150, s13, v148
	v_add3_u32 v138, v139, v138, v140
	v_add3_u32 v142, v85, v142, v125
	v_add3_u32 v147, v85, v147, v124
	v_mad_i32_i24 v152, v150, s13, v152
	v_mul_lo_u32 v153, s12, v153
	v_mad_i32_i24 v154, v141, s13, v154
	v_add3_u32 v145, v85, v145, v124
	v_add3_u32 v148, v85, v148, v124
	v_add3_u32 v152, v85, v152, v124
	v_add3_u32 v153, v139, v153, v140
	v_add3_u32 v154, v85, v154, v125
	buffer_load_ubyte v138, v138, s[16:19], 0 offen offset:256
	s_nop 0
	buffer_load_ubyte v142, v142, s[16:19], 0 offen offset:258
	s_nop 0
	buffer_load_ubyte v147, v147, s[16:19], 0 offen offset:257
	s_nop 0
	buffer_load_ubyte v149, v152, s[16:19], 0 offen offset:259
	buffer_load_ubyte v150, v153, s[16:19], 0 offen offset:256
	buffer_load_ubyte v151, v154, s[16:19], 0 offen offset:258
	s_nop 0
	buffer_load_ubyte v145, v145, s[16:19], 0 offen offset:257
	s_nop 0
	buffer_load_ubyte v148, v148, s[16:19], 0 offen offset:259
	v_add_u32_e32 v137, v111, v137
	v_mul_lo_u32 v137, s12, v137
	v_add3_u32 v137, v139, v137, v140
	v_add_u32_e32 v139, v111, v143
	v_mul_lo_u32 v139, s12, v139
	v_mad_i32_i24 v139, v141, s13, v139
	v_add3_u32 v133, v85, v133, v108
	v_add3_u32 v132, v85, v132, v108
	v_add3_u32 v139, v85, v139, v125
	buffer_load_dword v133, v133, s[4:7], 0 offen offset:256
	s_nop 0
	buffer_load_dword v132, v132, s[4:7], 0 offen offset:256
	s_nop 0
	buffer_load_ubyte v137, v137, s[16:19], 0 offen offset:256
	s_nop 0
	buffer_load_ubyte v139, v139, s[16:19], 0 offen offset:258
	ds_read_b128 v[156:159], v93
	ds_read_b128 v[160:163], v93 offset:2048
	ds_read_b128 v[164:167], v93 offset:4096
	ds_read_b128 v[168:171], v93 offset:6144
	ds_read_b128 v[172:175], v120
	ds_read_b128 v[176:179], v120 offset:2048
	ds_read_b128 v[180:183], v120 offset:4096
	ds_read_b128 v[184:187], v120 offset:6144
	ds_read_b128 v[188:191], v120 offset:8192
	s_barrier
	s_setprio 1
	v_and_b32_e32 v119, 0xff, v119
	v_and_b32_e32 v118, 0xff, v118
	v_and_b32_e32 v96, 0xff, v96
	v_and_b32_e32 v95, 0xff, v95
	v_and_b32_e32 v99, 0xff, v99
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[156:159], v[172:175], v[12:15], v86, v119 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[156:159], v[176:179], v[76:79], v86, v118 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[156:159], v[180:183], v[72:75], v86, v96 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[156:159], v[184:187], v[68:71], v86, v95 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[156:159], v[188:191], v[64:67], v86, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[160:163], v[172:175], v[60:63], v86, v119 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[160:163], v[176:179], v[56:59], v86, v118 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[160:163], v[180:183], v[52:55], v86, v96 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[160:163], v[184:187], v[48:51], v86, v95 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[160:163], v[188:191], v[44:47], v86, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[164:167], v[172:175], v[40:43], v84, v119 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[164:167], v[176:179], v[32:35], v84, v118 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[164:167], v[180:183], v[20:23], v84, v96 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[164:167], v[184:187], v[16:19], v84, v95 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[164:167], v[188:191], v[24:27], v84, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[168:171], v[172:175], v[28:31], v84, v119 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[168:171], v[176:179], v[36:39], v84, v118 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[168:171], v[180:183], v[8:11], v84, v96 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[168:171], v[184:187], v[4:7], v84, v95 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[168:171], v[188:191], v[0:3], v84, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	ds_read_b128 v[156:159], v94
	ds_read_b128 v[160:163], v94 offset:2048
	ds_read_b128 v[164:167], v94 offset:4096
	ds_read_b128 v[168:171], v94 offset:6144
	ds_read_b128 v[172:175], v121
	ds_read_b128 v[176:179], v121 offset:2048
	ds_read_b128 v[180:183], v121 offset:4096
	ds_read_b128 v[184:187], v121 offset:6144
	ds_read_b128 v[188:191], v121 offset:8192
	s_waitcnt vmcnt(12)
	s_barrier
	s_setprio 1
	v_and_b32_e32 v95, 0xff, v113
	v_and_b32_e32 v96, 0xff, v112
	v_and_b32_e32 v90, 0xff, v90
	v_and_b32_e32 v89, 0xff, v89
	v_and_b32_e32 v91, 0xff, v91
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[156:159], v[172:175], v[12:15], v86, v95 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[156:159], v[176:179], v[76:79], v86, v96 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[156:159], v[180:183], v[72:75], v86, v90 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[156:159], v[184:187], v[68:71], v86, v89 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[156:159], v[188:191], v[64:67], v86, v91 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[160:163], v[172:175], v[60:63], v86, v95 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[160:163], v[176:179], v[56:59], v86, v96 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[160:163], v[180:183], v[52:55], v86, v90 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[160:163], v[184:187], v[48:51], v86, v89 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[160:163], v[188:191], v[44:47], v86, v91 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[164:167], v[172:175], v[40:43], v84, v95 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[164:167], v[176:179], v[32:35], v84, v96 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[164:167], v[180:183], v[20:23], v84, v90 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[164:167], v[184:187], v[16:19], v84, v89 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[164:167], v[188:191], v[24:27], v84, v91 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[168:171], v[172:175], v[28:31], v84, v95 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[168:171], v[176:179], v[36:39], v84, v96 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[168:171], v[180:183], v[8:11], v84, v90 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[168:171], v[184:187], v[4:7], v84, v89 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[168:171], v[188:191], v[0:3], v84, v91 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_mov_b32 m0, s30
	v_add_u32_e32 v84, 0xfff0a000, v105
	s_waitcnt vmcnt(0)
	s_barrier
	buffer_load_dwordx4 v84, s[24:27], 0 offen lds
	v_add_u32_e32 v84, 0xfff5c000, v105
	s_mov_b32 m0, s31
	v_add_u32_e32 v86, 0xfffffbff, v128
	buffer_load_dwordx4 v84, s[24:27], 0 offen lds
	v_add_u32_e32 v84, 0xfffae000, v105
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v84, s[24:27], 0 offen lds
	v_add_u32_e32 v84, 0x400, v126
	v_cmp_gt_i32_e32 vcc, 0, v84
	s_mov_b32 m0, s34
	s_nop 0
	v_cndmask_b32_e32 v86, v84, v86, vcc
	v_mul_hi_i32 v86, v86, s35
	v_lshrrev_b32_e32 v89, 31, v86
	v_ashrrev_i32_e32 v86, 9, v86
	v_add_u32_e32 v86, v86, v89
	v_ashrrev_i32_e32 v84, 31, v84
	v_xor_b32_e32 v84, v86, v84
	v_add_u32_e32 v86, s29, v84
	v_mul_i32_i24_e32 v84, 0xffffeb80, v84
	v_mul_i32_i24_e32 v86, 0x1480, v86
	v_add3_u32 v84, v84, v86, v130
	buffer_load_dwordx4 v105, s[24:27], 0 offen lds
	v_add_u32_e32 v86, 0x1000, v84
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dword v86, s[20:23], 0 offen lds
	v_add_u32_e32 v86, 0x15800, v84
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dword v86, s[20:23], 0 offen lds
	v_add_u32_e32 v86, 0x2a000, v84
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dword v86, s[20:23], 0 offen lds
	v_add_u32_e32 v86, 0x3e800, v84
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dword v86, s[20:23], 0 offen lds
	v_add_u32_e32 v86, 0x53000, v84
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dword v86, s[20:23], 0 offen lds
	v_add_u32_e32 v86, 0x67800, v84
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dword v86, s[20:23], 0 offen lds
	v_add_u32_e32 v86, 0x7c000, v84
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dword v86, s[20:23], 0 offen lds
	v_add_u32_e32 v86, 0x90800, v84
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dword v86, s[20:23], 0 offen lds
	v_add_u32_e32 v86, 0xa5000, v84
	s_mov_b32 m0, s44
	v_add_u32_e32 v84, 0xb9800, v84
	buffer_load_dword v86, s[20:23], 0 offen lds
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dword v84, s[20:23], 0 offen lds
	v_add_u32_e32 v84, 0x80, v127
	v_add_u32_e32 v86, 0xff7f, v131
	v_cmp_gt_i32_e32 vcc, s46, v129
	v_add_u32_e32 v90, 0xfffffdff, v136
	v_add_u32_e32 v95, 0x202, v134
	v_cndmask_b32_e32 v84, v84, v86, vcc
	v_mul_i32_i24_sdwa v86, sext(v84), s59 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_add_u16_sdwa v84, v86, v84 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_lshrrev_b16_e32 v86, 15, v84
	v_ashrrev_i16_e32 v84, 6, v84
	v_add_u16_e32 v84, v84, v86
	v_cndmask_b32_e64 v86, 0, -1, vcc
	v_xor_b32_e32 v84, v84, v86
	v_bfe_i32 v84, v84, 0, 16
	v_add_u32_e32 v86, v80, v84
	v_add_u32_e32 v89, v116, v84
	v_mul_lo_u32 v86, v86, s14
	v_mul_lo_u32 v89, s14, v89
	v_mad_i32_i24 v86, v84, s13, v86
	v_mad_i32_i24 v84, v84, s13, v89
	v_add_u32_e32 v89, 0x200, v134
	v_cmp_gt_i32_e32 vcc, s60, v135
	v_add_u32_e32 v96, 0xfffffdfd, v136
	v_add_u32_e32 v113, 0xfffffdfc, v146
	v_cndmask_b32_e32 v90, v89, v90, vcc
	v_mul_hi_i32 v90, v90, s35
	v_lshrrev_b32_e32 v91, 31, v90
	v_ashrrev_i32_e32 v90, 7, v90
	v_cmp_gt_i32_e32 vcc, -2, v89
	v_add_u32_e32 v90, v90, v91
	v_ashrrev_i32_e32 v91, 31, v89
	v_cndmask_b32_e32 v89, v95, v96, vcc
	v_mul_hi_i32 v89, v89, s35
	v_lshrrev_b32_e32 v112, 31, v89
	v_ashrrev_i32_e32 v89, 7, v89
	v_cmp_gt_i32_e32 vcc, s63, v135
	v_add_u32_e32 v89, v89, v112
	v_ashrrev_i32_e32 v112, 31, v95
	v_cndmask_b32_e32 v95, v95, v96, vcc
	v_mul_hi_i32 v95, v95, s35
	v_lshrrev_b32_e32 v96, 31, v95
	v_lshrrev_b32_e32 v95, 7, v95
	v_add_u32_e32 v95, v95, v96
	v_xor_b32_e32 v127, v95, v112
	v_add_u32_e32 v95, 0x201, v144
	v_add_u32_e32 v96, 0xfffffdfe, v146
	v_cmp_gt_i32_e32 vcc, s61, v155
	v_xor_b32_e32 v126, v89, v112
	v_xor_b32_e32 v91, v90, v91
	v_cndmask_b32_e32 v96, v95, v96, vcc
	v_mul_hi_i32 v96, v96, s35
	v_lshrrev_b32_e32 v112, 31, v96
	v_ashrrev_i32_e32 v96, 7, v96
	v_add_u32_e32 v96, v96, v112
	v_ashrrev_i32_e32 v95, 31, v95
	v_xor_b32_e32 v95, v96, v95
	v_add_u32_e32 v96, v95, v88
	v_add_u32_e32 v112, 0x203, v144
	v_cmp_gt_i32_e32 vcc, s62, v155
	v_add_u32_e32 v131, v95, v92
	v_mul_lo_u32 v96, v96, s12
	v_cndmask_b32_e32 v118, v112, v113, vcc
	v_mul_lo_u32 v131, v131, s12
	v_mad_i32_i24 v96, v95, s13, v96
	v_mul_hi_i32 v118, v118, s35
	v_mad_i32_i24 v95, v95, s13, v131
	v_add_u32_e32 v131, 0x200, v144
	v_lshrrev_b32_e32 v119, 31, v118
	v_ashrrev_i32_e32 v118, 7, v118
	v_cmp_gt_i32_e32 vcc, -3, v131
	v_add_u32_e32 v118, v118, v119
	v_ashrrev_i32_e32 v119, 31, v112
	v_cndmask_b32_e32 v112, v112, v113, vcc
	v_mul_hi_i32 v112, v112, s35
	v_lshrrev_b32_e32 v113, 31, v112
	v_ashrrev_i32_e32 v112, 7, v112
	v_add_u32_e32 v89, v126, v87
	v_xor_b32_e32 v118, v118, v119
	v_add_u32_e32 v112, v112, v113
	v_add_u32_e32 v90, v91, v87
	v_mul_lo_u32 v89, v89, s12
	v_add_u32_e32 v128, v118, v88
	v_add_u32_e32 v130, v110, v126
	v_xad_u32 v112, v112, v119, v92
	v_mul_i32_i24_e32 v99, 0xfffffeb8, v91
	v_mul_lo_u32 v90, v90, s12
	v_mad_i32_i24 v89, v127, s13, v89
	v_mul_lo_u32 v128, v128, s12
	v_add_u32_e32 v129, v110, v91
	v_mul_lo_u32 v130, s12, v130
	v_mul_lo_u32 v112, v112, s12
	v_add3_u32 v90, v99, v90, v140
	v_add3_u32 v89, v85, v89, v125
	v_add3_u32 v96, v85, v96, v124
	v_mad_i32_i24 v128, v118, s13, v128
	v_mul_lo_u32 v129, s12, v129
	v_mad_i32_i24 v130, v127, s13, v130
	v_add3_u32 v95, v85, v95, v124
	v_mad_i32_i24 v112, v118, s13, v112
	v_add3_u32 v128, v85, v128, v124
	v_add3_u32 v129, v99, v129, v140
	v_add3_u32 v130, v85, v130, v125
	v_add3_u32 v131, v85, v112, v124
	buffer_load_ubyte v119, v90, s[16:19], 0 offen offset:512
	buffer_load_ubyte v113, v89, s[16:19], 0 offen offset:514
	buffer_load_ubyte v118, v96, s[16:19], 0 offen offset:513
	buffer_load_ubyte v112, v128, s[16:19], 0 offen offset:515
	s_nop 0
	buffer_load_ubyte v96, v129, s[16:19], 0 offen offset:512
	buffer_load_ubyte v90, v130, s[16:19], 0 offen offset:514
	s_nop 0
	buffer_load_ubyte v95, v95, s[16:19], 0 offen offset:513
	s_nop 0
	buffer_load_ubyte v89, v131, s[16:19], 0 offen offset:515
	v_add_u32_e32 v91, v111, v91
	v_mul_lo_u32 v91, s12, v91
	v_add3_u32 v91, v99, v91, v140
	v_add_u32_e32 v99, v111, v126
	v_mul_lo_u32 v99, s12, v99
	v_add3_u32 v86, v85, v86, v108
	v_add3_u32 v84, v85, v84, v108
	v_mad_i32_i24 v99, v127, s13, v99
	buffer_load_dword v86, v86, s[4:7], 0 offen offset:512
	s_nop 0
	buffer_load_dword v84, v84, s[4:7], 0 offen offset:512
	v_add3_u32 v126, v85, v99, v125
	buffer_load_ubyte v99, v91, s[16:19], 0 offen offset:512
	s_nop 0
	buffer_load_ubyte v91, v126, s[16:19], 0 offen offset:514
	ds_read_b128 v[126:129], v93 offset:32768
	ds_read_b128 v[152:155], v93 offset:34816
	ds_read_b128 v[156:159], v93 offset:36864
	ds_read_b128 v[160:163], v93 offset:38912
	ds_read_b128 v[164:167], v122
	ds_read_b128 v[168:171], v122 offset:2048
	ds_read_b128 v[172:175], v122 offset:4096
	ds_read_b128 v[176:179], v122 offset:6144
	ds_read_b128 v[180:183], v122 offset:8192
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[126:129], v[164:167], v[12:15], v133, v138 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[126:129], v[168:171], v[76:79], v133, v147 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[126:129], v[172:175], v[72:75], v133, v150 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[126:129], v[176:179], v[68:71], v133, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[126:129], v[180:183], v[64:67], v133, v137 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[152:155], v[164:167], v[60:63], v133, v138 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[152:155], v[168:171], v[56:59], v133, v147 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[152:155], v[172:175], v[52:55], v133, v150 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[152:155], v[176:179], v[48:51], v133, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[152:155], v[180:183], v[44:47], v133, v137 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[156:159], v[164:167], v[40:43], v132, v138 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[156:159], v[168:171], v[32:35], v132, v147 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[156:159], v[172:175], v[20:23], v132, v150 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[156:159], v[176:179], v[16:19], v132, v145 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[156:159], v[180:183], v[24:27], v132, v137 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[160:163], v[164:167], v[28:31], v132, v138 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[160:163], v[168:171], v[36:39], v132, v147 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[160:163], v[172:175], v[8:11], v132, v150 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[160:163], v[176:179], v[4:7], v132, v145 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[160:163], v[180:183], v[0:3], v132, v137 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	ds_read_b128 v[126:129], v94 offset:32768
	ds_read_b128 v[134:137], v94 offset:34816
	ds_read_b128 v[144:147], v94 offset:36864
	ds_read_b128 v[152:155], v94 offset:38912
	ds_read_b128 v[156:159], v123
	ds_read_b128 v[160:163], v123 offset:2048
	ds_read_b128 v[164:167], v123 offset:4096
	ds_read_b128 v[168:171], v123 offset:6144
	ds_read_b128 v[172:175], v123 offset:8192
	s_waitcnt vmcnt(12)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[126:129], v[156:159], v[12:15], v133, v142 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[126:129], v[160:163], v[76:79], v133, v149 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[126:129], v[164:167], v[72:75], v133, v151 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[126:129], v[168:171], v[68:71], v133, v148 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[126:129], v[172:175], v[64:67], v133, v139 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[134:137], v[156:159], v[60:63], v133, v142 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[134:137], v[160:163], v[56:59], v133, v149 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[134:137], v[164:167], v[52:55], v133, v151 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[134:137], v[168:171], v[48:51], v133, v148 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[134:137], v[172:175], v[44:47], v133, v139 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[144:147], v[156:159], v[40:43], v132, v142 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[144:147], v[160:163], v[32:35], v132, v149 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[144:147], v[164:167], v[20:23], v132, v151 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[144:147], v[168:171], v[16:19], v132, v148 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[144:147], v[172:175], v[24:27], v132, v139 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[152:155], v[156:159], v[28:31], v132, v142 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[152:155], v[160:163], v[36:39], v132, v149 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[152:155], v[164:167], v[8:11], v132, v151 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[152:155], v[168:171], v[4:7], v132, v148 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[152:155], v[172:175], v[0:3], v132, v139 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_add_i32 s15, s15, 2
	v_add_u32_e32 v101, 0x1000, v101
	v_add_u32_e32 v103, 0xfffffc00, v103
	v_add_u32_e32 v104, 0x400, v104
	v_add_u32_e32 v105, 0x100, v105
	v_add_u32_e32 v107, 0xfffffe00, v107
	v_add_u32_e32 v109, 0x200, v109
	v_add_u32_e32 v125, 0x200, v125
	v_add_u32_e32 v114, 0x200, v114
	v_add_u32_e32 v115, 0xfffffe00, v115
	v_add_u32_e32 v124, 0x200, v124
	v_add_u32_e32 v108, 0x200, v108
	v_add_u32_e32 v117, 0xffffff80, v117
	s_cmp_lt_u32 s15, 38
	v_add_u32_e32 v100, 0x80, v100
	s_cbranch_scc1 .LBB0_3
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	v_add_u32_e32 v83, 0x10000, v97
	s_barrier
	ds_read_b128 v[124:127], v83
	ds_read_b128 v[128:131], v83 offset:2048
	v_add_u32_e32 v85, 0x10000, v98
	ds_read_b128 v[132:135], v85
	ds_read_b128 v[136:139], v83 offset:8192
	ds_read_b128 v[140:143], v85 offset:2048
	ds_read_b128 v[144:147], v85 offset:4096
	ds_read_b128 v[148:151], v83 offset:4096
	ds_read_b128 v[152:155], v83 offset:6144
	ds_read_b128 v[156:159], v85 offset:6144
	ds_read_b128 v[160:163], v85 offset:8192
	ds_read_b128 v[108:111], v93
	ds_read_b128 v[120:123], v93 offset:2048
	ds_read_b128 v[114:117], v94
	ds_read_b128 v[164:167], v94 offset:2048
	ds_read_b128 v[168:171], v93 offset:4096
	ds_read_b128 v[172:175], v93 offset:6144
	ds_read_b128 v[176:179], v94 offset:4096
	ds_read_b128 v[180:183], v94 offset:6144
	s_waitcnt vmcnt(3) lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[108:111], v[124:127], v[12:15], v86, v119 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_movk_i32 s0, 0x7fff
	v_mov_b32_e32 v83, 0x7fc0
	s_mul_hi_u32 s1, s8, s28
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[114:117], v[132:135], v[12:15], v86, v113 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshl_or_b32 v80, v81, 2, v80
	v_mul_lo_u32 v80, s8, v80
	v_lshlrev_b32_e32 v81, 1, v82
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[108:111], v[148:151], v[72:75], v86, v96 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 s3, 0x27000
	s_nop 2
	v_bfe_u32 v85, v15, 16, 1
	v_bfe_u32 v87, v14, 16, 1
	v_add3_u32 v85, v15, v85, s0
	v_bfe_u32 v88, v13, 16, 1
	v_add3_u32 v87, v14, v87, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cmp_o_f32_e32 vcc, v15, v15
	v_add3_u32 v88, v13, v88, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[114:117], v[144:147], v[72:75], v86, v90 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v92, v12, 16, 1
	v_add3_u32 v92, v12, v92, s0
	v_lshl_add_u32 v82, v80, 1, v81
	v_cndmask_b32_e32 v72, v83, v85, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_lshrrev_b32_e32 v73, 16, v88
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[108:111], v[152:155], v[68:71], v86, v95 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v74, 16, v92
	s_nop 1
	v_cndmask_b32_e32 v68, v83, v87, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[108:111], v[128:131], v[76:79], v86, v118 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v69, v83, v73, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	s_waitcnt vmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[108:111], v[136:139], v[64:67], v86, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v70, v83, v74, vcc
	s_waitcnt vmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[114:117], v[160:163], v[12:15], v86, v91 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[120:123], v[124:127], v[60:63], v86, v119 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[164:167], v[132:135], v[12:15], v86, v113 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[120:123], v[128:131], v[56:59], v86, v118 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[114:117], v[140:143], v[76:79], v86, v112 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 1
	v_bfe_u32 v59, v103, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[114:117], v[156:159], v[104:107], v86, v89 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], v[164:167], v[140:143], v[12:15], v86, v112 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 2
	v_bfe_u32 v71, v79, 16, 1
	v_bfe_u32 v73, v78, 16, 1
	v_add3_u32 v60, v79, v71, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[120:123], v[148:151], v[52:55], v86, v96 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v74, v77, 16, 1
	v_add3_u32 v73, v78, v73, s0
	v_lshrrev_b32_e32 v60, 16, v60
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[164:167], v[144:147], v[12:15], v86, v90 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v79, v79
	v_bfe_u32 v75, v76, 16, 1
	v_add3_u32 v74, v77, v74, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[120:123], v[152:155], v[48:51], v86, v95 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v61, 16, v73
	v_cndmask_b32_e32 v60, v83, v60, vcc
	v_cmp_o_f32_e32 vcc, v78, v78
	v_add3_u32 v75, v76, v75, s0
	v_lshrrev_b32_e32 v62, 16, v74
	v_cndmask_b32_e32 v56, v83, v61, vcc
	v_cmp_o_f32_e32 vcc, v77, v77
	v_lshrrev_b32_e32 v63, 16, v75
	v_bfe_u32 v61, v102, 16, 1
	v_cndmask_b32_e32 v57, v83, v62, vcc
	v_cmp_o_f32_e32 vcc, v76, v76
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], v[164:167], v[156:159], v[12:15], v86, v89 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v48, v103, v59, s0
	v_cndmask_b32_e32 v58, v83, v63, vcc
	v_bfe_u32 v62, v101, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[120:123], v[136:139], v[44:47], v86, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v61, v102, v61, s0
	v_lshrrev_b32_e32 v48, 16, v48
	v_cmp_o_f32_e32 vcc, v103, v103
	v_mfma_scale_f32_16x16x128_f8f6f4 v[120:123], v[164:167], v[160:163], v[12:15], v86, v91 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v63, v100, 16, 1
	v_add3_u32 v62, v101, v62, s0
	v_lshrrev_b32_e32 v49, 16, v61
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[124:127], v[40:43], v84, v119 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v59, v83, v48, vcc
	v_cmp_o_f32_e32 vcc, v102, v102
	v_add3_u32 v63, v100, v63, s0
	v_lshrrev_b32_e32 v50, 16, v62
	v_cndmask_b32_e32 v61, v83, v49, vcc
	v_cmp_o_f32_e32 vcc, v101, v101
	v_lshrrev_b32_e32 v51, 16, v63
	v_bfe_u32 v44, v107, 16, 1
	v_cndmask_b32_e32 v62, v83, v50, vcc
	v_cmp_o_f32_e32 vcc, v100, v100
	v_bfe_u32 v40, v106, 16, 1
	v_bfe_u32 v41, v105, 16, 1
	v_cndmask_b32_e32 v63, v83, v51, vcc
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[176:179], v[132:135], v[12:15], v84, v113 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v42, v104, 16, 1
	v_add3_u32 v42, v104, v42, s0
	v_add3_u32 v41, v105, v41, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[128:131], v[32:35], v84, v118 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v40, v106, v40, s0
	v_lshrrev_b32_e32 v71, 16, v41
	v_lshrrev_b32_e32 v73, 16, v42
	v_add3_u32 v32, v107, v44, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[176:179], v[140:143], v[12:15], v84, v112 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v33, 16, v40
	v_lshrrev_b32_e32 v32, 16, v32
	v_cmp_o_f32_e32 vcc, v107, v107
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[148:151], v[20:23], v84, v96 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v85, v50, 16, 1
	v_cndmask_b32_e32 v78, v83, v32, vcc
	v_cmp_o_f32_e32 vcc, v106, v106
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[176:179], v[144:147], v[12:15], v84, v90 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v86, v49, 16, 1
	v_cndmask_b32_e32 v79, v83, v33, vcc
	v_cmp_o_f32_e32 vcc, v105, v105
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[152:155], v[16:19], v84, v95 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v85, v50, v85, s0
	v_bfe_u32 v87, v48, 16, 1
	v_add3_u32 v86, v49, v86, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[176:179], v[156:159], v[12:15], v84, v89 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v85, 16, v85
	v_add3_u32 v87, v48, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[168:171], v[136:139], v[24:27], v84, v99 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v87, 16, v87
	s_nop 1
	v_bfe_u32 v26, v67, 16, 1
	v_cndmask_b32_e32 v24, v83, v71, vcc
	v_cmp_o_f32_e32 vcc, v104, v104
	v_bfe_u32 v27, v66, 16, 1
	v_add3_u32 v26, v67, v26, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[176:179], v[160:163], v[12:15], v84, v91 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v25, v83, v73, vcc
	v_add3_u32 v27, v66, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[172:175], v[124:127], v[28:31], v84, v119 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
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
	v_bfe_u32 v30, v111, 16, 1
	v_lshrrev_b32_e32 v29, 16, v29
	v_cndmask_b32_e32 v28, v83, v28, vcc
	v_cmp_o_f32_e32 vcc, v64, v64
	v_bfe_u32 v31, v110, 16, 1
	v_add3_u32 v30, v111, v30, s0
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[180:183], v[132:135], v[12:15], v84, v113 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v29, v83, v29, vcc
	v_add3_u32 v31, v110, v31, s0
	v_lshrrev_b32_e32 v30, 16, v30
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[172:175], v[128:131], v[36:39], v84, v118 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v111, v111
	v_lshrrev_b32_e32 v31, 16, v31
	v_bfe_u32 v64, v115, 16, 1
	v_bfe_u32 v36, v109, 16, 1
	v_bfe_u32 v37, v108, 16, 1
	v_add3_u32 v36, v109, v36, s0
	v_cndmask_b32_e32 v30, v83, v30, vcc
	v_cmp_o_f32_e32 vcc, v110, v110
	v_add3_u32 v37, v108, v37, s0
	v_lshrrev_b32_e32 v36, 16, v36
	v_cndmask_b32_e32 v31, v83, v31, vcc
	v_cmp_o_f32_e32 vcc, v109, v109
	v_bfe_u32 v38, v117, 16, 1
	v_lshrrev_b32_e32 v37, 16, v37
	v_cndmask_b32_e32 v36, v83, v36, vcc
	v_cmp_o_f32_e32 vcc, v108, v108
	v_bfe_u32 v39, v116, 16, 1
	v_add3_u32 v38, v117, v38, s0
	v_cndmask_b32_e32 v37, v83, v37, vcc
	v_add3_u32 v39, v116, v39, s0
	v_lshrrev_b32_e32 v38, 16, v38
	v_cmp_o_f32_e32 vcc, v117, v117
	v_bfe_u32 v65, v114, 16, 1
	v_add3_u32 v64, v115, v64, s0
	v_lshrrev_b32_e32 v39, 16, v39
	v_cndmask_b32_e32 v38, v83, v38, vcc
	v_cmp_o_f32_e32 vcc, v116, v116
	v_add3_u32 v65, v114, v65, s0
	v_lshrrev_b32_e32 v64, 16, v64
	v_cndmask_b32_e32 v39, v83, v39, vcc
	v_cmp_o_f32_e32 vcc, v115, v115
	v_bfe_u32 v66, v55, 16, 1
	v_lshrrev_b32_e32 v65, 16, v65
	v_cndmask_b32_e32 v64, v83, v64, vcc
	v_cmp_o_f32_e32 vcc, v114, v114
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
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[172:175], v[148:151], v[8:11], v84, v96 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v73, v74, v73, s0
	v_cndmask_b32_e32 v71, v83, v71, vcc
	v_cmp_o_f32_e32 vcc, v74, v74
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[172:175], v[152:155], v[4:7], v84, v95 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v74, v123, 16, 1
	v_lshrrev_b32_e32 v73, 16, v73
	v_bfe_u32 v75, v122, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[172:175], v[136:139], v[0:3], v84, v99 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v74, v123, v74, s0
	v_cndmask_b32_e32 v73, v83, v73, vcc
	v_bfe_u32 v76, v121, 16, 1
	v_add3_u32 v75, v122, v75, s0
	v_lshrrev_b32_e32 v74, 16, v74
	v_cmp_o_f32_e32 vcc, v123, v123
	v_bfe_u32 v77, v120, 16, 1
	v_add3_u32 v76, v121, v76, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cndmask_b32_e32 v74, v83, v74, vcc
	v_cmp_o_f32_e32 vcc, v122, v122
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[180:183], v[140:143], v[12:15], v84, v112 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v77, v120, v77, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cndmask_b32_e32 v75, v83, v75, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[180:183], v[144:147], v[8:11], v84, v90 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v121, v121
	v_lshrrev_b32_e32 v77, 16, v77
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[180:183], v[156:159], v[4:7], v84, v89 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v76, v83, v76, vcc
	v_cmp_o_f32_e32 vcc, v120, v120
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[180:183], v[160:163], v[0:3], v84, v91 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
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
	.amdhsa_kernel wave_mxfp4_static_gemm_256x160x256_1536x3200x10496
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
		.amdhsa_next_free_vgpr 192
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 192
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
	.size	wave_mxfp4_static_gemm_256x160x256_1536x3200x10496, .Lfunc_end0-wave_mxfp4_static_gemm_256x160x256_1536x3200x10496

	.set wave_mxfp4_static_gemm_256x160x256_1536x3200x10496.num_vgpr, 192
	.set wave_mxfp4_static_gemm_256x160x256_1536x3200x10496.num_agpr, 0
	.set wave_mxfp4_static_gemm_256x160x256_1536x3200x10496.numbered_sgpr, 64
	.set wave_mxfp4_static_gemm_256x160x256_1536x3200x10496.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_256x160x256_1536x3200x10496.private_seg_size, 0
	.set wave_mxfp4_static_gemm_256x160x256_1536x3200x10496.uses_vcc, 1
	.set wave_mxfp4_static_gemm_256x160x256_1536x3200x10496.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_256x160x256_1536x3200x10496.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_256x160x256_1536x3200x10496.has_recursion, 0
	.set wave_mxfp4_static_gemm_256x160x256_1536x3200x10496.has_indirect_call, 0
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
    .name:           wave_mxfp4_static_gemm_256x160x256_1536x3200x10496
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     70
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_256x160x256_1536x3200x10496.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     192
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
