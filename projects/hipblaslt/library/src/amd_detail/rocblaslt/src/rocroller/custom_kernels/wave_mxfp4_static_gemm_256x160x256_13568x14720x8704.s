; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_256x160x256_13568x14720x8704
	.p2align	8
	.type	wave_mxfp4_static_gemm_256x160x256_13568x14720x8704,@function
wave_mxfp4_static_gemm_256x160x256_13568x14720x8704:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v93, 0x3ff, v0
	v_bfe_u32 v1, v0, 10, 10
	v_lshrrev_b32_e32 v2, 6, v93
	v_lshlrev_b32_e32 v0, 5, v1
	v_lshl_or_b32 v3, v2, 3, v0
	s_mov_b64 s[24:25], s[2:3]
	v_readfirstlane_b32 s2, v3
	v_lshrrev_b32_e32 v3, 3, v93
	s_lshl_b32 s28, s16, 8
	v_or3_b32 v4, v3, v0, s28
	v_xor_b32_e32 v3, v3, v93
	v_lshlrev_b32_e32 v3, 4, v3
	s_mov_b64 s[20:21], s[6:7]
	v_and_b32_e32 v84, 0x70, v3
	v_mul_u32_u24_e32 v85, 0x1100, v4
	s_and_b32 s6, s25, 0xffff
	s_lshl_b32 s30, s2, 7
	s_or_b32 s25, s6, 0x51000000
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_or_b32_e32 v3, v85, v84
	s_mov_b32 m0, s30
	s_movk_i32 s3, 0x1100
	buffer_load_dwordx4 v3, s[24:27], 0 offen lds
	v_mov_b32_e32 v3, 0x44000
	v_bfe_u32 v7, v93, 2, 3
	v_mad_u32_u24 v86, v4, s3, v3
	s_or_b32 s31, s30, 0x2000
	v_lshrrev_b32_e32 v5, 5, v93
	v_lshrrev_b32_e32 v6, 2, v93
	v_and_b32_e32 v8, 31, v93
	v_lshlrev_b32_e32 v7, 2, v7
	v_or_b32_e32 v3, v86, v84
	s_mov_b32 m0, s31
	v_bitop3_b32 v11, v6, v5, 7 bitop3:0x6c
	v_sub_u32_e32 v7, v8, v7
	buffer_load_dwordx4 v3, s[24:27], 0 offen lds
	v_mov_b32_e32 v3, 0x88000
	v_lshl_add_u32 v7, v11, 2, v7
	v_mad_u32_u24 v87, v4, s3, v3
	s_or_b32 s33, s30, 0x4000
	v_ashrrev_i32_e32 v8, 31, v7
	v_or_b32_e32 v3, v87, v84
	s_mov_b32 m0, s33
	v_xor_b32_e32 v7, v8, v7
	buffer_load_dwordx4 v3, s[24:27], 0 offen lds
	v_mov_b32_e32 v3, 0xcc000
	v_ashrrev_i32_e32 v9, 31, v7
	v_mad_u32_u24 v88, v4, s3, v3
	s_or_b32 s34, s30, 0x6000
	v_lshrrev_b32_e32 v9, 29, v9
	v_or_b32_e32 v3, v88, v84
	s_mov_b32 m0, s34
	v_add_u32_e32 v7, v7, v9
	buffer_load_dwordx4 v3, s[24:27], 0 offen lds
	v_lshlrev_b32_e32 v3, 3, v1
	v_ashrrev_i32_e32 v7, 3, v7
	v_lshrrev_b32_e32 v14, 1, v11
	v_lshl_or_b32 v3, v2, 1, v3
	v_xor_b32_e32 v13, v7, v8
	v_and_b32_e32 v8, 0xfc, v93
	v_lshlrev_b32_e32 v9, 7, v14
	v_readfirstlane_b32 s2, v3
	v_lshlrev_b32_e32 v3, 6, v11
	v_lshlrev_b32_e32 v4, 2, v5
	v_add_u32_e32 v10, v8, v9
	v_add3_u32 v12, v0, v93, v3
	v_lshlrev_b32_e32 v7, 7, v13
	v_sub_u32_e32 v15, v4, v10
	v_add3_u32 v89, v15, v12, v7
	v_ashrrev_i32_e32 v12, 31, v89
	v_xor_b32_e32 v15, v12, v89
	s_mov_b32 s35, 0x78787879
	v_mul_hi_i32 v15, v15, s35
	v_lshrrev_b32_e32 v16, 31, v15
	v_ashrrev_i32_e32 v15, 9, v15
	v_add_u32_e32 v15, v15, v16
	v_xor_b32_e32 v15, v15, v12
	v_sub_u32_e32 v12, v5, v6
	v_lshlrev_b32_e32 v17, 4, v12
	v_lshlrev_b32_e32 v94, 2, v93
	v_lshlrev_b32_e32 v12, 9, v13
	v_lshlrev_b32_e32 v13, 8, v11
	v_add3_u32 v17, v17, v94, v13
	v_lshlrev_b32_e32 v11, 9, v14
	v_sub_u32_e32 v17, v17, v11
	v_lshlrev_b32_e32 v14, 7, v1
	s_mul_i32 s29, s17, 0xa0
	v_add3_u32 v90, v17, v12, v14
	s_movk_i32 s6, 0xef00
	s_lshl_b32 s52, s2, 7
	v_add_u32_e32 v16, s29, v15
	v_mad_i32_i24 v15, v15, s6, v90
	s_and_b32 s6, s21, 0xffff
	s_add_i32 s36, s52, 0x10000
	s_or_b32 s54, s52, 0x800
	s_or_b32 s21, s6, 0x51000000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_mad_i32_i24 v15, v16, s3, v15
	s_mov_b32 m0, s36
	s_add_i32 s37, s54, 0x10000
	s_or_b32 s53, s52, 0x1000
	buffer_load_dword v15, s[20:23], 0 offen lds
	v_add_u32_e32 v16, 0x11000, v15
	s_mov_b32 m0, s37
	s_add_i32 s38, s53, 0x10000
	s_or_b32 s51, s52, 0x1800
	buffer_load_dword v16, s[20:23], 0 offen lds
	v_add_u32_e32 v16, 0x22000, v15
	s_mov_b32 m0, s38
	s_add_i32 s39, s51, 0x10000
	s_or_b32 s50, s52, 0x2000
	buffer_load_dword v16, s[20:23], 0 offen lds
	v_add_u32_e32 v16, 0x33000, v15
	s_mov_b32 m0, s39
	s_add_i32 s40, s50, 0x10000
	s_or_b32 s49, s52, 0x2800
	buffer_load_dword v16, s[20:23], 0 offen lds
	v_add_u32_e32 v16, 0x44000, v15
	s_mov_b32 m0, s40
	s_add_i32 s41, s49, 0x10000
	s_or_b32 s48, s52, 0x3000
	buffer_load_dword v16, s[20:23], 0 offen lds
	v_add_u32_e32 v16, 0x55000, v15
	s_mov_b32 m0, s41
	s_add_i32 s42, s48, 0x10000
	s_or_b32 s47, s52, 0x3800
	buffer_load_dword v16, s[20:23], 0 offen lds
	v_add_u32_e32 v16, 0x66000, v15
	s_mov_b32 m0, s42
	s_add_i32 s43, s47, 0x10000
	s_or_b32 s68, s52, 0x4000
	s_or_b32 s69, s52, 0x4800
	s_mul_i32 s15, s15, s28
	s_mul_hi_u32 s2, s14, s28
	buffer_load_dword v16, s[20:23], 0 offen lds
	v_add_u32_e32 v16, 0x77000, v15
	s_mov_b32 m0, s43
	s_add_i32 s44, s68, 0x10000
	s_add_i32 s45, s69, 0x10000
	s_add_i32 s2, s2, s15
	s_mul_i32 s3, s14, s28
	buffer_load_dword v16, s[20:23], 0 offen lds
	v_add_u32_e32 v16, 0x88000, v15
	s_mov_b32 m0, s44
	s_add_u32 s4, s4, s3
	buffer_load_dword v16, s[20:23], 0 offen lds
	v_add_u32_e32 v15, 0x99000, v15
	s_mov_b32 m0, s45
	v_lshrrev_b32_e32 v17, 4, v93
	v_bfe_u32 v82, v93, 4, 2
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s14, 0x3fff
	buffer_load_dword v15, s[20:23], 0 offen lds
	v_mul_i32_i24_e32 v15, 0xffffffc0, v17
	v_lshlrev_b32_e32 v16, 6, v82
	s_bitset1_b32 s3, 14
	v_and_b32_e32 v81, 0xc0, v93
	v_add3_u32 v80, v15, v94, v16
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	s_or_b32 s5, s2, s3
	v_mad_u64_u32 v[18:19], s[2:3], s14, v81, v[80:81]
	v_mad_i32_i24 v101, v17, -16, v93
	s_movk_i32 s2, 0x50
	s_mov_b32 s16, s4
	s_mov_b32 s17, s5
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	s_lshl_b32 s15, s14, 5
	v_mad_u32_u24 v83, v1, s2, v101
	v_add_u32_e32 v19, s15, v18
	buffer_load_dword v92, v18, s[16:19], 0 offen
	buffer_load_dword v91, v19, s[16:19], 0 offen
	v_ashrrev_i16_e32 v18, 15, v83
	v_lshrrev_b16_e32 v18, 11, v18
	v_add_u16_e32 v18, v83, v18
	v_and_b32_e32 v18, 0xffffffe0, v18
	v_sub_u16_e32 v18, v83, v18
	v_bfe_i32 v19, v18, 0, 16
	v_ashrrev_i32_e32 v20, 31, v19
	v_add_u16_e32 v21, 32, v18
	v_cmp_gt_i16_e32 vcc, 0, v18
	s_load_dwordx2 s[12:13], s[0:1], 0x40
	s_movk_i32 s2, 0x110
	v_cndmask_b32_e32 v18, v19, v21, vcc
	v_cndmask_b32_e64 v19, v20, 0, vcc
	v_xor_b32_e32 v18, v19, v18
	v_lshrrev_b32_e32 v20, 28, v18
	v_add_u32_e32 v18, v18, v20
	v_ashrrev_i32_e32 v18, 4, v18
	v_xor_b32_e32 v18, v18, v19
	v_add_u32_e32 v95, v18, v80
	v_ashrrev_i32_e32 v19, 31, v95
	v_xor_b32_e32 v20, v19, v95
	v_mul_hi_i32 v20, v20, s35
	v_lshrrev_b32_e32 v21, 31, v20
	v_ashrrev_i32_e32 v20, 7, v20
	v_add_u32_e32 v20, v20, v21
	v_ashrrev_i32_e32 v21, 31, v83
	v_xor_b32_e32 v22, v21, v83
	v_ashrrev_i32_e32 v23, 31, v22
	v_lshrrev_b32_e32 v23, 27, v23
	v_add_u32_e32 v22, v22, v23
	v_lshrrev_b32_e32 v22, 5, v22
	v_xor_b32_e32 v21, v22, v21
	v_lshlrev_b32_e32 v98, 5, v21
	v_xad_u32 v19, v20, v19, v98
	v_mul_hi_i32 v20, v95, s35
	v_lshrrev_b32_e32 v21, 31, v20
	v_ashrrev_i32_e32 v20, 7, v20
	v_add_u32_e32 v20, v20, v21
	v_mul_lo_u32 v20, v20, s2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s13, s29
	s_mul_hi_u32 s3, s12, s29
	s_add_i32 s3, s3, s2
	s_mul_i32 s2, s12, s29
	s_add_u32 s16, s8, s2
	v_sub_u32_e32 v20, v95, v20
	s_addc_u32 s2, s9, s3
	s_and_b32 s3, s12, 0x3fff
	v_add_u32_e32 v21, 0x110, v20
	v_cmp_gt_i32_e32 vcc, 0, v20
	s_bitset1_b32 s3, 14
	s_and_b32 s2, s2, 0xffff
	v_cndmask_b32_e32 v20, v20, v21, vcc
	s_lshl_b32 s3, s3, 16
	s_or_b32 s17, s2, s3
	v_mad_u64_u32 v[22:23], s[2:3], v19, s12, v[20:21]
	v_add_u32_e32 v19, 2, v95
	v_sub_u32_e32 v20, -3, v95
	v_cmp_gt_i32_e32 vcc, -2, v95
	v_sub_u32_e32 v21, 0xffef, v83
	v_mad_u32_u24 v24, v1, 5, v94
	v_cndmask_b32_e32 v19, v19, v20, vcc
	v_mul_hi_i32 v19, v19, s35
	v_lshrrev_b32_e32 v20, 31, v19
	v_ashrrev_i32_e32 v19, 7, v19
	v_add_u32_e32 v19, v19, v20
	v_cndmask_b32_e64 v20, 0, -1, vcc
	v_xor_b32_e32 v19, v19, v20
	v_add_u32_e32 v20, v19, v98
	v_mul_i32_i24_e32 v19, 0xfffffef0, v19
	v_mul_lo_u32 v20, v20, s12
	v_add3_u32 v23, v95, v19, v20
	v_add_u32_e32 v20, 16, v83
	v_cmp_gt_i32_e32 vcc, -16, v83
	v_mul_u32_u24_e32 v19, 5, v1
	s_movk_i32 s46, 0xfef0
	v_cndmask_b32_e32 v20, v20, v21, vcc
	v_ashrrev_i16_e32 v21, 15, v20
	v_lshrrev_b16_e32 v21, 11, v21
	v_add_u16_e32 v20, v20, v21
	v_ashrrev_i16_e32 v20, 5, v20
	v_cndmask_b32_e64 v21, 0, -1, vcc
	v_xor_b32_e32 v20, v20, v21
	v_bfe_i32 v25, v20, 0, 16
	v_mad_i32_i24 v21, v25, -2, v15
	v_add3_u32 v102, v21, v24, v16
	v_add_u32_e32 v24, 1, v102
	v_sub_u32_e32 v26, -2, v102
	v_cmp_gt_i32_e32 vcc, -1, v102
	v_mul_i32_i24_e32 v20, -2, v25
	v_lshlrev_b32_e32 v103, 5, v25
	v_cndmask_b32_e32 v24, v24, v26, vcc
	v_mul_hi_i32 v24, v24, s35
	v_lshrrev_b32_e32 v26, 31, v24
	v_ashrrev_i32_e32 v24, 7, v24
	v_add_u32_e32 v24, v24, v26
	v_cndmask_b32_e64 v26, 0, -1, vcc
	v_xor_b32_e32 v29, v24, v26
	v_add3_u32 v104, v20, v19, v80
	v_add_u32_e32 v25, v29, v103
	v_mad_i32_i24 v24, v29, s46, v104
	v_mad_u64_u32 v[26:27], s[2:3], v25, s12, v[24:25]
	v_add_u32_e32 v25, 3, v102
	v_sub_u32_e32 v27, -4, v102
	v_cmp_gt_i32_e32 vcc, -3, v102
	v_sub_u32_e32 v33, 0xffcf, v83
	s_lshl_b32 s13, s12, 5
	v_cndmask_b32_e32 v25, v25, v27, vcc
	v_mul_hi_i32 v25, v25, s35
	v_lshrrev_b32_e32 v27, 31, v25
	v_ashrrev_i32_e32 v25, 7, v25
	v_add_u32_e32 v25, v25, v27
	v_cndmask_b32_e64 v27, 0, -1, vcc
	v_xor_b32_e32 v27, v25, v27
	v_add_u32_e32 v25, v27, v103
	v_mad_i32_i24 v28, v27, s46, v104
	v_mad_u64_u32 v[30:31], s[2:3], v25, s12, v[28:29]
	s_movk_i32 s2, 0xffd0
	v_add_u32_e32 v25, 48, v83
	v_cmp_gt_i32_e32 vcc, s2, v83
	v_add_u32_e32 v31, s13, v22
	v_add_u32_e32 v32, s13, v23
	v_cndmask_b32_e32 v25, v25, v33, vcc
	v_ashrrev_i16_e32 v33, 15, v25
	v_lshrrev_b16_e32 v33, 11, v33
	v_add_u16_e32 v25, v25, v33
	v_ashrrev_i16_e32 v25, 5, v25
	v_cndmask_b32_e64 v33, 0, -1, vcc
	v_xor_b32_e32 v25, v25, v33
	v_mov_b32_e32 v33, 5
	v_lshlrev_b32_sdwa v109, v33, sext(v25) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_e32 v25, v109, v29
	v_mad_u64_u32 v[24:25], s[2:3], s12, v25, v[24:25]
	v_add_u32_e32 v25, v109, v27
	v_mad_u64_u32 v[28:29], s[2:3], s12, v25, v[28:29]
	buffer_load_ubyte v111, v22, s[16:19], 0 offen
	buffer_load_ubyte v105, v23, s[16:19], 0 offen offset:2
	buffer_load_ubyte v112, v26, s[16:19], 0 offen offset:1
	buffer_load_ubyte v106, v30, s[16:19], 0 offen offset:3
	buffer_load_ubyte v113, v31, s[16:19], 0 offen
	buffer_load_ubyte v107, v32, s[16:19], 0 offen offset:2
	buffer_load_ubyte v114, v24, s[16:19], 0 offen offset:1
	buffer_load_ubyte v108, v28, s[16:19], 0 offen offset:3
	v_add_u32_e32 v22, s13, v31
	v_add_u32_e32 v23, s13, v32
	buffer_load_ubyte v115, v22, s[16:19], 0 offen
	buffer_load_ubyte v110, v23, s[16:19], 0 offen offset:2
	v_cmp_eq_u32_e64 s[2:3], 0, v1
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	s_movk_i32 s18, 0x2800
	s_movk_i32 s58, 0xffc0
	v_mul_i32_i24_e32 v22, -16, v17
	s_mov_b32 s59, -2
	s_and_b64 vcc, exec, s[2:3]
	s_waitcnt vmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	v_lshlrev_b32_e32 v25, 7, v93
	v_lshlrev_b32_e32 v17, 11, v17
	v_and_b32_e32 v23, 7, v93
	v_sub_u32_e32 v17, v25, v17
	v_mul_lo_u32 v1, v1, s18
	v_bitop3_b32 v24, v82, v93, 7 bitop3:0x78
	v_lshl_add_u32 v2, v2, 13, v17
	v_add_u32_e32 v1, v17, v1
	v_bitop3_b32 v17, v82, v23, 4 bitop3:0x36
	v_lshlrev_b32_e32 v24, 4, v24
	v_lshlrev_b32_e32 v17, 4, v17
	v_or_b32_e32 v99, v1, v24
	v_or_b32_e32 v100, v17, v1
	v_add3_u32 v1, v12, v13, v14
	v_or_b32_e32 v96, v2, v24
	v_or_b32_e32 v97, v17, v2
	v_lshl_add_u32 v1, v5, 4, v1
	v_lshlrev_b32_e32 v2, 4, v6
	v_sub_u32_e32 v1, v1, v2
	v_sub_u32_e32 v117, v1, v11
	v_sub_u32_e32 v1, v10, v4
	v_sub_u32_e32 v1, v1, v0
	v_sub_u32_e32 v1, v1, v3
	v_sub_u32_e32 v119, v1, v7
	v_add_u32_e32 v1, v7, v3
	v_add3_u32 v0, v1, v0, v4
	v_sub_u32_e32 v0, v0, v8
	v_sub_u32_e32 v120, v0, v9
	v_add_u32_e32 v0, v20, v15
	s_load_dwordx2 s[8:9], s[0:1], 0x48
	v_add3_u32 v0, v0, v19, v16
	v_add_u32_e32 v127, v15, v16
	v_sub_u32_e32 v124, 0xfffffdfc, v0
	v_sub_u32_e32 v125, 0, v0
	v_add3_u32 v0, v15, v18, v16
	v_add_u32_e32 v128, v127, v18
	v_lshlrev_b32_e32 v116, 4, v82
	v_sub_u32_e32 v126, 0, v0
	v_add_u32_e32 v0, v128, v94
	s_mov_b32 s0, 0xcc100
	v_add3_u32 v122, v21, v16, v19
	v_add_u32_e32 v129, 0x2400, v0
	v_add_u32_e32 v130, 0x4600, v0
	v_add_u32_e32 v131, 0x2300, v0
	v_add_u32_e32 v132, 0x4500, v0
	v_add_u32_e32 v136, v22, v116
	v_mov_b32_e32 v0, 0
	v_sub_u32_e32 v118, 0, v93
	v_add3_u32 v121, v85, v84, s0
	v_sub_u32_e32 v123, 0, v94
	v_add_u32_e32 v133, 32, v98
	v_add_u32_e32 v134, 64, v98
	v_add_u32_e32 v135, 32, v81
	v_sub_u32_e32 v137, 0, v136
	s_mov_b32 s60, 0xf0f0f0f1
	s_movk_i32 s61, 0x7879
	s_movk_i32 s62, 0xff00
	s_movk_i32 s63, 0xfeff
	s_mov_b32 s18, s6
	s_mov_b32 s19, s7
	s_movk_i32 s64, 0x102
	s_movk_i32 s66, 0xfefd
	s_movk_i32 s67, 0xfefe
	s_add_i32 s65, s30, 0x8000
	s_mov_b32 s26, s6
	s_mov_b32 s27, s7
	s_add_i32 s57, s31, 0x8000
	s_add_i32 s56, s33, 0x8000
	s_add_i32 s55, s34, 0x8000
	s_add_i32 s52, s52, 0x15000
	s_mov_b32 s22, s6
	s_mov_b32 s23, s7
	s_add_i32 s54, s54, 0x15000
	s_add_i32 s53, s53, 0x15000
	s_add_i32 s51, s51, 0x15000
	s_add_i32 s50, s50, 0x15000
	s_add_i32 s49, s49, 0x15000
	s_add_i32 s48, s48, 0x15000
	s_add_i32 s47, s47, 0x15000
	s_add_i32 s1, s68, 0x15000
	s_add_i32 s0, s69, 0x15000
	v_add_u32_e32 v138, 0x10000, v99
	v_add_u32_e32 v139, 0x10000, v100
	v_add_u32_e32 v140, 0x15000, v99
	v_add_u32_e32 v141, 0x15000, v100
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v7, v0
	v_mov_b32_e32 v8, v0
	v_mov_b32_e32 v9, v0
	v_mov_b32_e32 v10, v0
	v_mov_b32_e32 v11, v0
	v_mov_b32_e32 v12, v0
	v_mov_b32_e32 v13, v0
	v_mov_b32_e32 v14, v0
	v_mov_b32_e32 v15, v0
	v_mov_b32_e32 v16, v0
	v_mov_b32_e32 v17, v0
	v_mov_b32_e32 v18, v0
	v_mov_b32_e32 v19, v0
	v_mov_b32_e32 v20, v0
	v_mov_b32_e32 v21, v0
	v_mov_b32_e32 v22, v0
	v_mov_b32_e32 v23, v0
	v_mov_b32_e32 v24, v0
	v_mov_b32_e32 v25, v0
	v_mov_b32_e32 v26, v0
	v_mov_b32_e32 v27, v0
	v_mov_b32_e32 v28, v0
	v_mov_b32_e32 v29, v0
	v_mov_b32_e32 v30, v0
	v_mov_b32_e32 v31, v0
	v_mov_b32_e32 v32, v0
	v_mov_b32_e32 v33, v0
	v_mov_b32_e32 v34, v0
	v_mov_b32_e32 v35, v0
	v_mov_b32_e32 v36, v0
	v_mov_b32_e32 v37, v0
	v_mov_b32_e32 v38, v0
	v_mov_b32_e32 v39, v0
	v_mov_b32_e32 v40, v0
	v_mov_b32_e32 v41, v0
	v_mov_b32_e32 v42, v0
	v_mov_b32_e32 v43, v0
	v_mov_b32_e32 v44, v0
	v_mov_b32_e32 v45, v0
	v_mov_b32_e32 v46, v0
	v_mov_b32_e32 v47, v0
	v_mov_b32_e32 v48, v0
	v_mov_b32_e32 v49, v0
	v_mov_b32_e32 v50, v0
	v_mov_b32_e32 v51, v0
	v_mov_b32_e32 v52, v0
	v_mov_b32_e32 v53, v0
	v_mov_b32_e32 v54, v0
	v_mov_b32_e32 v55, v0
	v_mov_b32_e32 v56, v0
	v_mov_b32_e32 v57, v0
	v_mov_b32_e32 v58, v0
	v_mov_b32_e32 v59, v0
	v_mov_b32_e32 v60, v0
	v_mov_b32_e32 v61, v0
	v_mov_b32_e32 v62, v0
	v_mov_b32_e32 v63, v0
	v_mov_b32_e32 v64, v0
	v_mov_b32_e32 v65, v0
	v_mov_b32_e32 v66, v0
	v_mov_b32_e32 v67, v0
	v_mov_b32_e32 v68, v0
	v_mov_b32_e32 v69, v0
	v_mov_b32_e32 v70, v0
	v_mov_b32_e32 v71, v0
	v_mov_b32_e32 v72, v0
	v_mov_b32_e32 v73, v0
	v_mov_b32_e32 v74, v0
	v_mov_b32_e32 v75, v0
	v_mov_b32_e32 v76, v0
	v_mov_b32_e32 v77, v0
	v_mov_b32_e32 v78, v0
	v_mov_b32_e32 v79, v0
	v_mov_b32_e32 v142, v128
	v_mov_b32_e32 v143, v122
.LBB0_3:
	v_mul_hi_u32 v144, v129, s60
	v_lshrrev_b32_e32 v144, 8, v144
	v_add_u32_e32 v144, v98, v144
	v_mul_lo_u32 v146, s12, v144
	v_mul_hi_u32 v144, v130, s60
	v_lshrrev_b32_e32 v144, 8, v144
	v_add_u32_e32 v144, v98, v144
	v_mul_lo_u32 v147, s12, v144
	v_mul_hi_u32 v144, v131, s60
	v_lshrrev_b32_e32 v144, 8, v144
	v_add_u32_e32 v144, v98, v144
	v_mul_lo_u32 v148, s12, v144
	v_mul_hi_u32 v144, v132, s60
	v_lshrrev_b32_e32 v144, 8, v144
	v_add_u32_e32 v144, v98, v144
	v_mul_lo_u32 v149, s12, v144
	v_add_u32_e32 v150, v93, v136
	v_add_u32_e32 v152, v118, v137
	v_add_u32_e32 v151, 64, v150
	v_add_u32_e32 v144, 0xffbf, v152
	v_cmp_gt_i32_e32 vcc, s58, v150
	v_add_u32_e32 v155, v123, v126
	v_add_u32_e32 v156, 0xfffffeff, v155
	v_cndmask_b32_e32 v144, v151, v144, vcc
	v_mul_i32_i24_sdwa v144, sext(v144), s61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshrrev_b32_e32 v145, 31, v144
	v_ashrrev_i32_e32 v144, 21, v144
	v_add_u16_e32 v144, v144, v145
	v_cndmask_b32_e64 v145, 0, -1, vcc
	v_xor_b32_e32 v144, v144, v145
	v_bfe_i32 v144, v144, 0, 16
	v_add_u32_e32 v145, v81, v144
	v_add_u32_e32 v153, v135, v144
	v_mul_lo_u32 v145, v145, s14
	v_mul_lo_u32 v153, s14, v153
	v_mad_i32_i24 v145, v144, s46, v145
	v_mad_i32_i24 v144, v144, s46, v153
	v_add_u32_e32 v153, v94, v128
	v_add_u32_e32 v154, 0x100, v153
	v_cmp_gt_i32_e32 vcc, s62, v153
	v_add_u32_e32 v159, v142, v94
	v_add_u32_e32 v160, 0xfffffefd, v155
	v_cndmask_b32_e32 v156, v154, v156, vcc
	v_mul_hi_i32 v156, v156, s35
	v_lshrrev_b32_e32 v157, 31, v156
	v_ashrrev_i32_e32 v156, 7, v156
	v_add_u32_e32 v156, v156, v157
	v_cndmask_b32_e64 v157, 0, -1, vcc
	v_xor_b32_e32 v156, v156, v157
	v_add_u32_e32 v157, v156, v98
	v_mul_i32_i24_e32 v158, 0xfffffef0, v156
	v_mul_lo_u32 v157, v157, s12
	v_add3_u32 v157, v158, v157, v159
	v_add_u32_e32 v158, 0x102, v153
	v_cmp_gt_i32_e32 vcc, -2, v154
	v_add3_u32 v164, v123, v124, s64
	v_add_u32_e32 v166, v123, v125
	v_cndmask_b32_e32 v161, v158, v160, vcc
	v_mul_hi_i32 v161, v161, s35
	v_lshrrev_b32_e32 v162, 31, v161
	v_ashrrev_i32_e32 v161, 7, v161
	v_cmp_gt_i32_e32 vcc, s67, v153
	v_add_u32_e32 v161, v161, v162
	v_ashrrev_i32_e32 v162, 31, v158
	v_cndmask_b32_e32 v158, v158, v160, vcc
	v_mul_hi_i32 v158, v158, s35
	v_xor_b32_e32 v161, v161, v162
	v_lshrrev_b32_e32 v160, 31, v158
	v_lshrrev_b32_e32 v158, 7, v158
	v_add_u32_e32 v162, v161, v98
	v_add_u32_e32 v158, v158, v160
	v_cndmask_b32_e64 v160, 0, -1, vcc
	v_xor_b32_e32 v158, v158, v160
	v_mul_lo_u32 v160, v162, s12
	v_add_u32_e32 v162, v94, v122
	v_add_u32_e32 v163, 0x101, v162
	v_cmp_gt_i32_e32 vcc, s63, v162
	v_add_u32_e32 v167, 0xfffffefc, v166
	v_mad_i32_i24 v156, v156, s46, v128
	v_cndmask_b32_e32 v164, v163, v164, vcc
	v_mul_hi_i32 v164, v164, s35
	v_lshrrev_b32_e32 v165, 31, v164
	v_ashrrev_i32_e32 v164, 7, v164
	v_add_u32_e32 v164, v164, v165
	v_ashrrev_i32_e32 v163, 31, v163
	v_xor_b32_e32 v163, v164, v163
	v_add_u32_e32 v164, v163, v103
	v_add_u32_e32 v165, 0x103, v162
	v_cmp_gt_i32_e32 vcc, s66, v162
	v_add_u32_e32 v172, v163, v109
	v_mul_lo_u32 v164, v164, s12
	v_cndmask_b32_e32 v168, v165, v167, vcc
	v_mul_lo_u32 v172, v172, s12
	v_mad_i32_i24 v164, v163, s46, v164
	v_mul_hi_i32 v168, v168, s35
	v_mad_i32_i24 v163, v163, s46, v172
	v_add_u32_e32 v172, 0x100, v162
	v_lshrrev_b32_e32 v169, 31, v168
	v_ashrrev_i32_e32 v168, 7, v168
	v_cmp_gt_i32_e32 vcc, -3, v172
	v_add_u32_e32 v168, v168, v169
	v_ashrrev_i32_e32 v169, 31, v165
	v_cndmask_b32_e32 v165, v165, v167, vcc
	v_mul_hi_i32 v165, v165, s35
	v_lshrrev_b32_e32 v167, 31, v165
	v_ashrrev_i32_e32 v165, 7, v165
	v_add_u32_e32 v165, v165, v167
	v_xor_b32_e32 v168, v168, v169
	v_add3_u32 v148, v156, v148, v94
	v_xad_u32 v165, v165, v169, v109
	v_add3_u32 v149, v156, v149, v94
	v_add_u32_e32 v156, v134, v161
	v_add_u32_e32 v170, v168, v103
	v_add_u32_e32 v171, v133, v161
	v_mul_lo_u32 v165, v165, s12
	v_mul_lo_u32 v156, s12, v156
	v_mad_i32_i24 v160, v158, s46, v160
	v_mul_lo_u32 v170, v170, s12
	v_mul_lo_u32 v171, s12, v171
	v_mad_i32_i24 v165, v168, s46, v165
	v_mad_i32_i24 v156, v158, s46, v156
	s_mov_b32 m0, s65
	v_add3_u32 v145, v94, v145, v127
	v_add3_u32 v144, v94, v144, v127
	v_add3_u32 v160, v94, v160, v142
	v_add3_u32 v164, v94, v164, v143
	v_mad_i32_i24 v170, v168, s46, v170
	v_mad_i32_i24 v171, v158, s46, v171
	v_add3_u32 v163, v94, v163, v143
	v_add3_u32 v165, v94, v165, v143
	v_add3_u32 v156, v94, v156, v142
	v_add_u32_e32 v158, 0xfff33f80, v121
	buffer_load_dword v145, v145, s[4:7], 0 offen offset:256
	s_nop 0
	buffer_load_dword v144, v144, s[4:7], 0 offen offset:256
	v_add3_u32 v170, v94, v170, v143
	v_add3_u32 v171, v94, v171, v142
	buffer_load_ubyte v157, v157, s[16:19], 0 offen offset:256
	s_nop 0
	buffer_load_ubyte v160, v160, s[16:19], 0 offen offset:258
	s_nop 0
	buffer_load_ubyte v164, v164, s[16:19], 0 offen offset:257
	s_nop 0
	buffer_load_ubyte v167, v170, s[16:19], 0 offen offset:259
	s_nop 0
	buffer_load_ubyte v148, v148, s[16:19], 0 offen offset:256
	s_nop 0
	buffer_load_ubyte v168, v171, s[16:19], 0 offen offset:258
	s_nop 0
	buffer_load_ubyte v163, v163, s[16:19], 0 offen offset:257
	s_nop 0
	buffer_load_ubyte v165, v165, s[16:19], 0 offen offset:259
	s_nop 0
	buffer_load_ubyte v149, v149, s[16:19], 0 offen offset:256
	s_nop 0
	buffer_load_ubyte v156, v156, s[16:19], 0 offen offset:258
	s_waitcnt vmcnt(12)
	s_barrier
	buffer_load_dwordx4 v158, s[24:27], 0 offen lds
	v_add_u32_e32 v158, 0xfff77f80, v121
	s_mov_b32 m0, s57
	v_add_u32_e32 v169, v118, v119
	buffer_load_dwordx4 v158, s[24:27], 0 offen lds
	v_add_u32_e32 v158, 0xfffbbf80, v121
	s_mov_b32 m0, s56
	v_add_u32_e32 v170, 0xfffffdff, v169
	buffer_load_dwordx4 v158, s[24:27], 0 offen lds
	v_add_u32_e32 v158, 0xffffff80, v121
	s_mov_b32 m0, s55
	s_nop 0
	buffer_load_dwordx4 v158, s[24:27], 0 offen lds
	v_add_u32_e32 v158, v93, v120
	v_add_u32_e32 v161, 0x200, v158
	v_cmp_gt_i32_e32 vcc, 0, v161
	s_mov_b32 m0, s52
	s_nop 0
	v_cndmask_b32_e32 v170, v161, v170, vcc
	v_mul_hi_i32 v170, v170, s35
	v_lshrrev_b32_e32 v171, 31, v170
	v_ashrrev_i32_e32 v170, 9, v170
	v_add_u32_e32 v170, v170, v171
	v_ashrrev_i32_e32 v161, 31, v161
	v_xor_b32_e32 v161, v170, v161
	v_add_u32_e32 v170, s29, v161
	v_mul_i32_i24_e32 v161, 0xffffef00, v161
	v_mul_i32_i24_e32 v170, 0x1100, v170
	v_add_u32_e32 v171, v117, v94
	v_add3_u32 v161, v161, v170, v171
	v_add_u32_e32 v170, 0x800, v161
	buffer_load_dword v170, s[20:23], 0 offen lds
	v_add_u32_e32 v170, 0x11800, v161
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dword v170, s[20:23], 0 offen lds
	v_add_u32_e32 v170, 0x22800, v161
	s_mov_b32 m0, s53
	s_nop 0
	buffer_load_dword v170, s[20:23], 0 offen lds
	v_add_u32_e32 v170, 0x33800, v161
	s_mov_b32 m0, s51
	s_nop 0
	buffer_load_dword v170, s[20:23], 0 offen lds
	v_add_u32_e32 v170, 0x44800, v161
	s_mov_b32 m0, s50
	s_nop 0
	buffer_load_dword v170, s[20:23], 0 offen lds
	v_add_u32_e32 v170, 0x55800, v161
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dword v170, s[20:23], 0 offen lds
	v_add_u32_e32 v170, 0x66800, v161
	s_mov_b32 m0, s48
	s_nop 0
	buffer_load_dword v170, s[20:23], 0 offen lds
	v_add_u32_e32 v170, 0x77800, v161
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dword v170, s[20:23], 0 offen lds
	v_add_u32_e32 v170, 0x88800, v161
	s_mov_b32 m0, s1
	v_add_u32_e32 v161, 0x99800, v161
	buffer_load_dword v170, s[20:23], 0 offen lds
	s_mov_b32 m0, s0
	s_nop 0
	buffer_load_dword v161, s[20:23], 0 offen lds
	ds_read_b128 v[174:177], v96
	ds_read_b128 v[178:181], v96 offset:2048
	ds_read_b128 v[182:185], v96 offset:4096
	ds_read_b128 v[186:189], v96 offset:6144
	ds_read_b128 v[190:193], v138
	ds_read_b128 v[194:197], v138 offset:2048
	ds_read_b128 v[198:201], v138 offset:4096
	ds_read_b128 v[202:205], v138 offset:6144
	ds_read_b128 v[206:209], v97
	ds_read_b128 v[210:213], v97 offset:2048
	ds_read_b128 v[214:217], v97 offset:4096
	ds_read_b128 v[218:221], v97 offset:6144
	ds_read_b128 v[222:225], v138 offset:8192
	ds_read_b128 v[226:229], v139
	ds_read_b128 v[230:233], v139 offset:2048
	ds_read_b128 v[234:237], v139 offset:4096
	ds_read_b128 v[238:241], v139 offset:6144
	ds_read_b128 v[242:245], v139 offset:8192
	s_barrier
	s_setprio 1
	v_and_b32_e32 v111, 0xff, v111
	v_and_b32_e32 v112, 0xff, v112
	v_and_b32_e32 v113, 0xff, v113
	v_and_b32_e32 v114, 0xff, v114
	v_and_b32_e32 v115, 0xff, v115
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[174:177], v[190:193], v[0:3], v92, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[174:177], v[194:197], v[4:7], v92, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[174:177], v[198:201], v[8:11], v92, v113 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[174:177], v[202:205], v[12:15], v92, v114 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[174:177], v[222:225], v[16:19], v92, v115 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[178:181], v[190:193], v[20:23], v92, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[178:181], v[194:197], v[24:27], v92, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[178:181], v[198:201], v[28:31], v92, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[178:181], v[202:205], v[32:35], v92, v114 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[178:181], v[222:225], v[36:39], v92, v115 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[182:185], v[190:193], v[40:43], v91, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[182:185], v[194:197], v[44:47], v91, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[182:185], v[198:201], v[48:51], v91, v113 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[182:185], v[202:205], v[52:55], v91, v114 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[182:185], v[222:225], v[56:59], v91, v115 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[186:189], v[190:193], v[60:63], v91, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[186:189], v[194:197], v[64:67], v91, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[186:189], v[198:201], v[68:71], v91, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[186:189], v[202:205], v[72:75], v91, v114 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[186:189], v[222:225], v[76:79], v91, v115 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	s_waitcnt vmcnt(17)
	s_barrier
	s_setprio 1
	v_and_b32_e32 v105, 0xff, v105
	v_and_b32_e32 v106, 0xff, v106
	v_and_b32_e32 v107, 0xff, v107
	v_and_b32_e32 v108, 0xff, v108
	v_and_b32_e32 v110, 0xff, v110
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[206:209], v[226:229], v[0:3], v92, v105 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[206:209], v[230:233], v[4:7], v92, v106 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[206:209], v[234:237], v[8:11], v92, v107 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[206:209], v[238:241], v[12:15], v92, v108 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[206:209], v[242:245], v[16:19], v92, v110 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[210:213], v[226:229], v[20:23], v92, v105 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[210:213], v[230:233], v[24:27], v92, v106 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[210:213], v[234:237], v[28:31], v92, v107 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[210:213], v[238:241], v[32:35], v92, v108 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[210:213], v[242:245], v[36:39], v92, v110 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[214:217], v[226:229], v[40:43], v91, v105 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[214:217], v[230:233], v[44:47], v91, v106 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[214:217], v[234:237], v[48:51], v91, v107 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[214:217], v[238:241], v[52:55], v91, v108 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[214:217], v[242:245], v[56:59], v91, v110 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[218:221], v[226:229], v[60:63], v91, v105 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[218:221], v[230:233], v[64:67], v91, v106 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[218:221], v[234:237], v[68:71], v91, v107 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[218:221], v[238:241], v[72:75], v91, v108 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[218:221], v[242:245], v[76:79], v91, v110 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_setprio 0
	v_add_u32_e32 v91, 0x80, v150
	v_add_u32_e32 v92, 0xff7f, v152
	v_cmp_gt_i32_e32 vcc, s58, v151
	v_add_u32_e32 v106, 0xfffffdff, v155
	v_add_u32_e32 v110, 0xfffffdfd, v155
	v_cndmask_b32_e32 v91, v91, v92, vcc
	v_mul_i32_i24_sdwa v91, sext(v91), s61 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshrrev_b32_e32 v92, 31, v91
	v_ashrrev_i32_e32 v91, 21, v91
	v_add_u16_e32 v91, v91, v92
	v_cndmask_b32_e64 v92, 0, -1, vcc
	v_xor_b32_e32 v91, v91, v92
	v_bfe_i32 v91, v91, 0, 16
	v_add_u32_e32 v92, v81, v91
	v_add_u32_e32 v105, v135, v91
	v_mul_lo_u32 v92, v92, s14
	v_mul_lo_u32 v105, s14, v105
	v_mad_i32_i24 v92, v91, s46, v92
	v_mad_i32_i24 v91, v91, s46, v105
	v_add_u32_e32 v105, 0x200, v153
	v_cmp_gt_i32_e32 vcc, s62, v154
	v_add_u32_e32 v113, 0xfffffdfc, v166
	s_mov_b32 m0, s30
	v_cndmask_b32_e32 v106, v105, v106, vcc
	v_mul_hi_i32 v106, v106, s35
	v_lshrrev_b32_e32 v107, 31, v106
	v_ashrrev_i32_e32 v106, 7, v106
	v_add_u32_e32 v106, v106, v107
	v_ashrrev_i32_e32 v107, 31, v105
	v_xor_b32_e32 v106, v106, v107
	v_add_u32_e32 v107, v106, v98
	v_mul_i32_i24_e32 v108, 0xfffffef0, v106
	v_mul_lo_u32 v107, v107, s12
	v_add3_u32 v107, v108, v107, v159
	v_add_u32_e32 v108, 0x202, v153
	v_cmp_gt_i32_e32 vcc, -2, v105
	v_mad_i32_i24 v152, v106, s46, v128
	v_add3_u32 v146, v152, v146, v94
	v_cndmask_b32_e32 v105, v108, v110, vcc
	v_mul_hi_i32 v105, v105, s35
	v_lshrrev_b32_e32 v111, 31, v105
	v_ashrrev_i32_e32 v105, 7, v105
	v_cmp_gt_i32_e32 vcc, s67, v154
	v_add_u32_e32 v105, v105, v111
	v_ashrrev_i32_e32 v111, 31, v108
	v_cndmask_b32_e32 v108, v108, v110, vcc
	v_mul_hi_i32 v108, v108, s35
	v_lshrrev_b32_e32 v110, 31, v108
	v_lshrrev_b32_e32 v108, 7, v108
	v_add_u32_e32 v108, v108, v110
	v_xor_b32_e32 v115, v105, v111
	v_xor_b32_e32 v110, v108, v111
	v_add_u32_e32 v108, 0x201, v162
	v_add_u32_e32 v111, 0xfffffdfe, v166
	v_cmp_gt_i32_e32 vcc, s63, v172
	v_add_u32_e32 v106, v133, v115
	v_mul_lo_u32 v106, s12, v106
	v_cndmask_b32_e32 v111, v108, v111, vcc
	v_mul_hi_i32 v111, v111, s35
	v_lshrrev_b32_e32 v112, 31, v111
	v_ashrrev_i32_e32 v111, 7, v111
	v_add_u32_e32 v111, v111, v112
	v_ashrrev_i32_e32 v108, 31, v108
	v_xor_b32_e32 v108, v111, v108
	v_mad_i32_i24 v106, v110, s46, v106
	v_add_u32_e32 v111, v108, v103
	v_add3_u32 v153, v94, v106, v142
	v_add_u32_e32 v106, v108, v109
	v_mul_lo_u32 v111, v111, s12
	v_mul_lo_u32 v106, v106, s12
	v_mad_i32_i24 v111, v108, s46, v111
	v_mad_i32_i24 v106, v108, s46, v106
	v_add3_u32 v112, v94, v111, v143
	v_add_u32_e32 v111, 0x203, v162
	v_cmp_gt_i32_e32 vcc, s66, v172
	v_add3_u32 v108, v94, v106, v143
	v_add_u32_e32 v106, 0x200, v162
	v_cndmask_b32_e32 v114, v111, v113, vcc
	v_cmp_gt_i32_e32 vcc, -3, v106
	v_mul_hi_i32 v114, v114, s35
	v_lshrrev_b32_e32 v150, 31, v114
	v_cndmask_b32_e32 v106, v111, v113, vcc
	v_ashrrev_i32_e32 v114, 7, v114
	v_mul_hi_i32 v106, v106, s35
	v_add_u32_e32 v114, v114, v150
	v_ashrrev_i32_e32 v150, 31, v111
	v_lshrrev_b32_e32 v111, 31, v106
	v_ashrrev_i32_e32 v106, 7, v106
	v_add_u32_e32 v105, v115, v98
	v_xor_b32_e32 v114, v114, v150
	v_add_u32_e32 v106, v106, v111
	v_mul_lo_u32 v105, v105, s12
	v_add_u32_e32 v151, v114, v103
	v_xad_u32 v106, v106, v150, v109
	v_add_u32_e32 v115, v134, v115
	v_mad_i32_i24 v105, v110, s46, v105
	v_mul_lo_u32 v151, v151, s12
	v_mul_lo_u32 v106, v106, s12
	v_mul_lo_u32 v115, s12, v115
	v_add3_u32 v105, v94, v105, v142
	v_mad_i32_i24 v151, v114, s46, v151
	v_mad_i32_i24 v106, v114, s46, v106
	v_mad_i32_i24 v110, v110, s46, v115
	v_add3_u32 v151, v94, v151, v143
	v_add3_u32 v150, v94, v106, v143
	buffer_load_ubyte v111, v107, s[16:19], 0 offen offset:512
	s_nop 0
	buffer_load_ubyte v105, v105, s[16:19], 0 offen offset:514
	s_nop 0
	buffer_load_ubyte v112, v112, s[16:19], 0 offen offset:513
	s_nop 0
	buffer_load_ubyte v106, v151, s[16:19], 0 offen offset:515
	buffer_load_ubyte v113, v146, s[16:19], 0 offen offset:512
	buffer_load_ubyte v107, v153, s[16:19], 0 offen offset:514
	buffer_load_ubyte v114, v108, s[16:19], 0 offen offset:513
	s_nop 0
	buffer_load_ubyte v108, v150, s[16:19], 0 offen offset:515
	v_add3_u32 v146, v152, v147, v94
	v_add3_u32 v110, v94, v110, v142
	v_add3_u32 v92, v94, v92, v127
	v_add3_u32 v91, v94, v91, v127
	buffer_load_ubyte v115, v146, s[16:19], 0 offen offset:512
	s_nop 0
	buffer_load_ubyte v110, v110, s[16:19], 0 offen offset:514
	v_add_u32_e32 v146, 0xfff34000, v121
	buffer_load_dword v92, v92, s[4:7], 0 offen offset:512
	s_nop 0
	buffer_load_dword v91, v91, s[4:7], 0 offen offset:512
	s_waitcnt vmcnt(12)
	s_barrier
	buffer_load_dwordx4 v146, s[24:27], 0 offen lds
	v_add_u32_e32 v146, 0xfff78000, v121
	s_mov_b32 m0, s31
	v_add_u32_e32 v147, 0xfffffbff, v169
	buffer_load_dwordx4 v146, s[24:27], 0 offen lds
	v_add_u32_e32 v146, 0xfffbc000, v121
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v146, s[24:27], 0 offen lds
	v_add_u32_e32 v146, 0x400, v158
	v_cmp_gt_i32_e32 vcc, 0, v146
	s_mov_b32 m0, s34
	s_nop 0
	v_cndmask_b32_e32 v147, v146, v147, vcc
	v_mul_hi_i32 v147, v147, s35
	v_lshrrev_b32_e32 v150, 31, v147
	v_ashrrev_i32_e32 v147, 9, v147
	v_add_u32_e32 v147, v147, v150
	v_ashrrev_i32_e32 v146, 31, v146
	v_xor_b32_e32 v146, v147, v146
	v_add_u32_e32 v147, s29, v146
	v_mul_i32_i24_e32 v146, 0xffffef00, v146
	v_mul_i32_i24_e32 v147, 0x1100, v147
	v_add3_u32 v146, v146, v147, v171
	buffer_load_dwordx4 v121, s[24:27], 0 offen lds
	v_add_u32_e32 v147, 0x1000, v146
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dword v147, s[20:23], 0 offen lds
	v_add_u32_e32 v147, 0x12000, v146
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dword v147, s[20:23], 0 offen lds
	v_add_u32_e32 v147, 0x23000, v146
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dword v147, s[20:23], 0 offen lds
	v_add_u32_e32 v147, 0x34000, v146
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dword v147, s[20:23], 0 offen lds
	v_add_u32_e32 v147, 0x45000, v146
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dword v147, s[20:23], 0 offen lds
	v_add_u32_e32 v147, 0x56000, v146
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dword v147, s[20:23], 0 offen lds
	v_add_u32_e32 v147, 0x67000, v146
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dword v147, s[20:23], 0 offen lds
	v_add_u32_e32 v147, 0x78000, v146
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dword v147, s[20:23], 0 offen lds
	v_add_u32_e32 v147, 0x89000, v146
	s_mov_b32 m0, s44
	v_add_u32_e32 v146, 0x9a000, v146
	buffer_load_dword v147, s[20:23], 0 offen lds
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dword v146, s[20:23], 0 offen lds
	ds_read_b128 v[150:153], v96 offset:32768
	ds_read_b128 v[170:173], v96 offset:34816
	ds_read_b128 v[174:177], v96 offset:36864
	ds_read_b128 v[178:181], v96 offset:38912
	ds_read_b128 v[182:185], v140
	ds_read_b128 v[186:189], v140 offset:2048
	ds_read_b128 v[190:193], v140 offset:4096
	ds_read_b128 v[194:197], v140 offset:6144
	ds_read_b128 v[198:201], v97 offset:32768
	ds_read_b128 v[202:205], v97 offset:34816
	ds_read_b128 v[206:209], v97 offset:36864
	ds_read_b128 v[210:213], v97 offset:38912
	ds_read_b128 v[214:217], v140 offset:8192
	ds_read_b128 v[218:221], v141
	ds_read_b128 v[222:225], v141 offset:2048
	ds_read_b128 v[226:229], v141 offset:4096
	ds_read_b128 v[230:233], v141 offset:6144
	ds_read_b128 v[234:237], v141 offset:8192
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(13)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[150:153], v[182:185], v[0:3], v145, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(12)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[150:153], v[186:189], v[4:7], v145, v164 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[150:153], v[190:193], v[8:11], v145, v148 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(10)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[150:153], v[194:197], v[12:15], v145, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[150:153], v[214:217], v[16:19], v145, v149 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[170:173], v[182:185], v[20:23], v145, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[170:173], v[186:189], v[24:27], v145, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[170:173], v[190:193], v[28:31], v145, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[170:173], v[194:197], v[32:35], v145, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[170:173], v[214:217], v[36:39], v145, v149 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[174:177], v[182:185], v[40:43], v144, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[174:177], v[186:189], v[44:47], v144, v164 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[174:177], v[190:193], v[48:51], v144, v148 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[174:177], v[194:197], v[52:55], v144, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[174:177], v[214:217], v[56:59], v144, v149 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[178:181], v[182:185], v[60:63], v144, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[178:181], v[186:189], v[64:67], v144, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[178:181], v[190:193], v[68:71], v144, v148 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[178:181], v[194:197], v[72:75], v144, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[178:181], v[214:217], v[76:79], v144, v149 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	s_waitcnt vmcnt(17)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[198:201], v[218:221], v[0:3], v145, v160 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[198:201], v[222:225], v[4:7], v145, v167 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[198:201], v[226:229], v[8:11], v145, v168 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[198:201], v[230:233], v[12:15], v145, v165 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[198:201], v[234:237], v[16:19], v145, v156 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[202:205], v[218:221], v[20:23], v145, v160 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[202:205], v[222:225], v[24:27], v145, v167 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[202:205], v[226:229], v[28:31], v145, v168 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[202:205], v[230:233], v[32:35], v145, v165 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[202:205], v[234:237], v[36:39], v145, v156 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[206:209], v[218:221], v[40:43], v144, v160 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[206:209], v[222:225], v[44:47], v144, v167 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[206:209], v[226:229], v[48:51], v144, v168 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[206:209], v[230:233], v[52:55], v144, v165 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[206:209], v[234:237], v[56:59], v144, v156 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[210:213], v[218:221], v[60:63], v144, v160 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[210:213], v[222:225], v[64:67], v144, v167 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[210:213], v[226:229], v[68:71], v144, v168 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[210:213], v[230:233], v[72:75], v144, v165 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[210:213], v[234:237], v[76:79], v144, v156 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_add_i32 s59, s59, 2
	v_add_u32_e32 v117, 0x1000, v117
	v_add_u32_e32 v119, 0xfffffc00, v119
	v_add_u32_e32 v120, 0x400, v120
	v_add_u32_e32 v121, 0x100, v121
	v_add_u32_e32 v122, 0x200, v122
	v_add_u32_e32 v124, 0xfffffe00, v124
	v_add_u32_e32 v125, 0xfffffe00, v125
	v_add_u32_e32 v143, 0x200, v143
	v_add_u32_e32 v126, 0xfffffe00, v126
	v_add_u32_e32 v128, 0x200, v128
	v_add_u32_e32 v129, 0x200, v129
	v_add_u32_e32 v130, 0x200, v130
	v_add_u32_e32 v142, 0x200, v142
	v_add_u32_e32 v131, 0x200, v131
	v_add_u32_e32 v132, 0x200, v132
	v_add_u32_e32 v127, 0x200, v127
	v_add_u32_e32 v137, 0xffffff80, v137
	s_cmp_lt_u32 s59, 30
	v_add_u32_e32 v136, 0x80, v136
	s_cbranch_scc1 .LBB0_3
	s_movk_i32 s18, 0x840
	v_add3_u32 v93, v101, v116, s18
	s_mov_b32 s18, 0xf0f1
	v_mul_u32_u24_sdwa v93, v93, s18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshrrev_b32_e32 v93, 22, v93
	v_add_u32_e32 v93, v81, v93
	v_mad_u64_u32 v[116:117], s[18:19], s14, v93, v[80:81]
	v_add_u32_e32 v94, s15, v116
	buffer_load_dword v93, v116, s[4:7], 0 offen offset:16
	buffer_load_dword v80, v94, s[4:7], 0 offen offset:16
	v_add_u32_e32 v94, 0x2100, v95
	s_mov_b32 s14, 0xf0f0f0f1
	v_mul_hi_u32 v101, v94, s14
	v_lshrrev_b32_e32 v101, 8, v101
	v_add_u32_e32 v116, v101, v98
	v_mul_u32_u24_e32 v101, 0x110, v101
	v_sub_u32_e32 v118, v94, v101
	v_add_u32_e32 v94, 0x2102, v95
	v_mul_hi_u32 v101, v94, s14
	v_lshrrev_b32_e32 v101, 8, v101
	v_mad_u64_u32 v[120:121], s[4:5], v116, s12, v[118:119]
	v_add_u32_e32 v116, v101, v98
	v_mul_u32_u24_e32 v101, 0x110, v101
	v_sub_u32_e32 v94, v94, v101
	v_mad_u64_u32 v[122:123], s[4:5], v116, s12, v[94:95]
	v_add_u32_e32 v94, 0x2101, v102
	v_mul_hi_u32 v94, v94, s14
	v_lshrrev_b32_e32 v116, 8, v94
	s_movk_i32 s15, 0xfef0
	s_movk_i32 s4, 0x2101
	v_mul_lo_u32 v94, v116, s15
	v_add_u32_e32 v102, 0x2103, v102
	v_add_u32_e32 v117, v116, v103
	v_add3_u32 v94, v104, v94, s4
	v_mul_hi_u32 v102, v102, s14
	v_mad_u64_u32 v[124:125], s[4:5], v117, s12, v[94:95]
	v_lshrrev_b32_e32 v117, 8, v102
	v_add_u32_e32 v101, s13, v122
	s_movk_i32 s4, 0x2103
	v_mul_lo_u32 v102, v117, s15
	v_add_u32_e32 v103, v117, v103
	v_add3_u32 v102, v104, v102, s4
	v_add_u32_e32 v121, 0xffffff00, v101
	v_add_u32_e32 v101, v109, v116
	v_mad_u64_u32 v[126:127], s[4:5], v103, s12, v[102:103]
	v_add_u32_e32 v103, 0x4300, v95
	v_mad_u64_u32 v[130:131], s[4:5], s12, v101, v[94:95]
	v_add_u32_e32 v95, 0x6500, v95
	v_mul_hi_u32 v103, v103, s14
	v_mul_hi_u32 v95, v95, s14
	v_lshrrev_b32_e32 v103, 8, v103
	v_lshrrev_b32_e32 v95, 8, v95
	v_add_u32_e32 v103, v103, v98
	v_add_u32_e32 v94, v109, v117
	v_add_u32_e32 v95, v95, v98
	v_mad_u64_u32 v[128:129], s[4:5], v103, s12, v[118:119]
	v_mad_u64_u32 v[132:133], s[4:5], s12, v94, v[102:103]
	v_mad_u64_u32 v[118:119], s[4:5], v95, s12, v[118:119]
	s_movk_i32 s4, 0x1080
	s_mov_b32 m0, s65
	s_mov_b32 s18, s6
	s_mov_b32 s19, s7
	v_add3_u32 v85, v84, v85, s4
	s_mov_b32 s26, s6
	s_mov_b32 s27, s7
	buffer_load_ubyte v117, v120, s[16:19], 0 offen
	buffer_load_ubyte v104, v122, s[16:19], 0 offen
	buffer_load_ubyte v109, v124, s[16:19], 0 offen
	buffer_load_ubyte v116, v126, s[16:19], 0 offen
	buffer_load_ubyte v101, v128, s[16:19], 0 offen
	buffer_load_ubyte v102, v121, s[16:19], 0 offen offset:256
	buffer_load_ubyte v103, v130, s[16:19], 0 offen
	buffer_load_ubyte v94, v132, s[16:19], 0 offen
	v_add_u32_e32 v119, s13, v121
	buffer_load_ubyte v95, v118, s[16:19], 0 offen
	buffer_load_ubyte v98, v119, s[16:19], 0 offen offset:256
	s_waitcnt vmcnt(12)
	s_barrier
	buffer_load_dwordx4 v85, s[24:27], 0 offen lds
	v_add3_u32 v85, v84, v86, s4
	s_mov_b32 m0, s57
	s_mov_b32 s22, s6
	buffer_load_dwordx4 v85, s[24:27], 0 offen lds
	v_add3_u32 v85, v84, v87, s4
	s_mov_b32 m0, s56
	v_add3_u32 v84, v84, v88, s4
	buffer_load_dwordx4 v85, s[24:27], 0 offen lds
	s_mov_b32 m0, s55
	s_mov_b32 s23, s7
	buffer_load_dwordx4 v84, s[24:27], 0 offen lds
	v_add_u32_e32 v84, 0x4200, v89
	v_mul_hi_u32 v84, v84, s14
	v_lshrrev_b32_e32 v84, 10, v84
	v_add_u32_e32 v85, s29, v84
	v_mul_i32_i24_e32 v84, 0xffffef00, v84
	v_mul_u32_u24_e32 v85, 0x1100, v85
	v_add3_u32 v84, v90, v84, v85
	v_add_u32_e32 v85, 0x10800, v84
	s_mov_b32 m0, s52
	s_nop 0
	buffer_load_dword v85, s[20:23], 0 offen lds
	v_add_u32_e32 v85, 0x21800, v84
	s_mov_b32 m0, s54
	s_nop 0
	buffer_load_dword v85, s[20:23], 0 offen lds
	v_add_u32_e32 v85, 0x32800, v84
	s_mov_b32 m0, s53
	s_nop 0
	buffer_load_dword v85, s[20:23], 0 offen lds
	v_add_u32_e32 v85, 0x43800, v84
	s_mov_b32 m0, s51
	s_nop 0
	buffer_load_dword v85, s[20:23], 0 offen lds
	v_add_u32_e32 v85, 0x54800, v84
	s_mov_b32 m0, s50
	s_nop 0
	buffer_load_dword v85, s[20:23], 0 offen lds
	v_add_u32_e32 v85, 0x65800, v84
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dword v85, s[20:23], 0 offen lds
	v_add_u32_e32 v85, 0x76800, v84
	s_mov_b32 m0, s48
	s_nop 0
	buffer_load_dword v85, s[20:23], 0 offen lds
	v_add_u32_e32 v85, 0x87800, v84
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dword v85, s[20:23], 0 offen lds
	v_add_u32_e32 v85, 0x98800, v84
	s_mov_b32 m0, s1
	v_add_u32_e32 v84, 0xa9800, v84
	buffer_load_dword v85, s[20:23], 0 offen lds
	s_mov_b32 m0, s0
	s_nop 0
	buffer_load_dword v84, s[20:23], 0 offen lds
	v_add_u32_e32 v84, 0x10000, v99
	ds_read_b128 v[134:137], v84
	ds_read_b128 v[138:141], v84 offset:2048
	ds_read_b128 v[142:145], v84 offset:4096
	ds_read_b128 v[146:149], v84 offset:6144
	ds_read_b128 v[154:157], v97
	ds_read_b128 v[158:161], v97 offset:2048
	ds_read_b128 v[162:165], v97 offset:4096
	ds_read_b128 v[166:169], v97 offset:6144
	v_add_u32_e32 v85, 0x10000, v100
	ds_read_b128 v[170:173], v84 offset:8192
	ds_read_b128 v[174:177], v85
	ds_read_b128 v[178:181], v85 offset:2048
	ds_read_b128 v[182:185], v85 offset:4096
	ds_read_b128 v[186:189], v85 offset:6144
	ds_read_b128 v[190:193], v85 offset:8192
	ds_read_b128 v[86:89], v96
	ds_read_b128 v[118:121], v96 offset:2048
	ds_read_b128 v[130:133], v96 offset:4096
	ds_read_b128 v[150:153], v96 offset:6144
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[86:89], v[134:137], v[0:3], v92, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[86:89], v[138:141], v[4:7], v92, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[86:89], v[142:145], v[8:11], v92, v113 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[86:89], v[146:149], v[12:15], v92, v114 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[86:89], v[170:173], v[16:19], v92, v115 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[118:121], v[134:137], v[20:23], v92, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[118:121], v[138:141], v[24:27], v92, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[118:121], v[142:145], v[28:31], v92, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[118:121], v[146:149], v[32:35], v92, v114 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[118:121], v[170:173], v[36:39], v92, v115 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[130:133], v[134:137], v[40:43], v91, v111 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[130:133], v[138:141], v[44:47], v91, v112 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[130:133], v[142:145], v[48:51], v91, v113 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[130:133], v[146:149], v[52:55], v91, v114 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], v[130:133], v[170:173], v[56:59], v91, v115 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], v[150:153], v[134:137], v[60:63], v91, v111 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], v[150:153], v[138:141], v[64:67], v91, v112 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[150:153], v[142:145], v[68:71], v91, v113 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[150:153], v[146:149], v[72:75], v91, v114 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[112:115], v[150:153], v[170:173], v[76:79], v91, v115 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	s_waitcnt vmcnt(17)
	s_barrier
	s_setprio 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[154:157], v[174:177], v[0:3], v92, v105 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[154:157], v[178:181], v[4:7], v92, v106 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[154:157], v[182:185], v[8:11], v92, v107 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[154:157], v[186:189], v[12:15], v92, v108 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[154:157], v[190:193], v[16:19], v92, v110 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[158:161], v[174:177], v[20:23], v92, v105 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[158:161], v[178:181], v[24:27], v92, v106 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[158:161], v[182:185], v[28:31], v92, v107 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[158:161], v[186:189], v[32:35], v92, v108 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[158:161], v[190:193], v[36:39], v92, v110 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[162:165], v[174:177], v[84:87], v91, v105 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[162:165], v[178:181], v[118:121], v91, v106 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[162:165], v[182:185], v[122:125], v91, v107 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[162:165], v[186:189], v[126:129], v91, v108 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[162:165], v[190:193], v[130:133], v91, v110 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[166:169], v[174:177], v[134:137], v91, v105 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[166:169], v[178:181], v[138:141], v91, v106 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[166:169], v[182:185], v[142:145], v91, v107 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[166:169], v[186:189], v[146:149], v91, v108 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[166:169], v[190:193], v[112:115], v91, v110 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	v_add_u32_e32 v84, 0x15000, v99
	s_waitcnt vmcnt(0)
	s_barrier
	ds_read_b128 v[110:113], v84
	ds_read_b128 v[118:121], v84 offset:2048
	v_add_u32_e32 v85, 0x15000, v100
	v_and_b32_e32 v92, 0xffff, v117
	ds_read_b128 v[122:125], v85
	ds_read_b128 v[126:129], v84 offset:8192
	ds_read_b128 v[130:133], v85 offset:2048
	ds_read_b128 v[134:137], v85 offset:4096
	ds_read_b128 v[138:141], v84 offset:4096
	ds_read_b128 v[142:145], v84 offset:6144
	ds_read_b128 v[146:149], v85 offset:6144
	ds_read_b128 v[150:153], v85 offset:8192
	ds_read_b128 v[154:157], v96 offset:32768
	ds_read_b128 v[158:161], v96 offset:34816
	ds_read_b128 v[162:165], v97 offset:32768
	ds_read_b128 v[166:169], v97 offset:34816
	ds_read_b128 v[170:173], v96 offset:36864
	ds_read_b128 v[174:177], v96 offset:38912
	ds_read_b128 v[178:181], v97 offset:36864
	ds_read_b128 v[182:185], v97 offset:38912
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[154:157], v[110:113], v[76:79], v93, v92 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_and_b32_e32 v96, 0xffff, v104
	v_and_b32_e32 v97, 0xffff, v109
	v_and_b32_e32 v100, 0xffff, v101
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[162:165], v[122:125], v[76:79], v93, v96 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_and_b32_e32 v101, 0xffff, v102
	v_and_b32_e32 v102, 0xffff, v103
	v_and_b32_e32 v99, 0xffff, v116
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[154:157], v[118:121], v[72:75], v93, v97 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_movk_i32 s0, 0x7fff
	s_nop 2
	v_bfe_u32 v84, v79, 16, 1
	v_bfe_u32 v85, v78, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[154:157], v[142:145], v[64:67], v93, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v86, v77, 16, 1
	v_and_b32_e32 v94, 0xffff, v94
	v_bfe_u32 v87, v76, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[162:165], v[130:133], v[72:75], v93, v99 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v86, v77, v86, s0
	v_add3_u32 v85, v78, v85, s0
	v_add3_u32 v84, v79, v84, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[154:157], v[138:141], v[68:71], v93, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v88, v76, v87, s0
	v_lshrrev_b32_e32 v89, 16, v84
	v_lshrrev_b32_e32 v103, 16, v85
	v_lshrrev_b32_e32 v104, 16, v86
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[162:165], v[146:149], v[64:67], v93, v94 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v79, v79
	v_and_b32_e32 v95, 0xffff, v95
	v_and_b32_e32 v98, 0xffff, v98
	v_mov_b32_e32 v66, 0x7fc0
	v_cndmask_b32_e32 v64, v66, v89, vcc
	v_cmp_o_f32_e32 vcc, v78, v78
	v_lshrrev_b32_e32 v65, 16, v88
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[154:157], v[126:129], v[60:63], v93, v95 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v67, v73, 16, 1
	v_add3_u32 v67, v73, v67, s0
	v_lshrrev_b32_e32 v67, 16, v67
	v_cndmask_b32_e32 v60, v66, v103, vcc
	v_cmp_o_f32_e32 vcc, v77, v77
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[162:165], v[134:137], v[68:71], v93, v101 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v63, v75, 16, 1
	v_cndmask_b32_e32 v61, v66, v104, vcc
	v_cmp_o_f32_e32 vcc, v76, v76
	v_add3_u32 v63, v75, v63, s0
	v_lshrrev_b32_e32 v63, 16, v63
	v_cndmask_b32_e32 v62, v66, v65, vcc
	v_bfe_u32 v65, v74, 16, 1
	v_add3_u32 v65, v74, v65, s0
	v_cmp_o_f32_e32 vcc, v75, v75
	v_bfe_u32 v76, v72, 16, 1
	v_lshrrev_b32_e32 v65, 16, v65
	v_cndmask_b32_e32 v63, v66, v63, vcc
	v_cmp_o_f32_e32 vcc, v74, v74
	v_add3_u32 v76, v72, v76, s0
	v_lshrrev_b32_e32 v76, 16, v76
	v_cndmask_b32_e32 v65, v66, v65, vcc
	v_cmp_o_f32_e32 vcc, v73, v73
	v_bfe_u32 v73, v71, 16, 1
	v_bfe_u32 v74, v70, 16, 1
	v_cndmask_b32_e32 v67, v66, v67, vcc
	v_cmp_o_f32_e32 vcc, v72, v72
	v_add3_u32 v73, v71, v73, s0
	v_bfe_u32 v75, v69, 16, 1
	v_cndmask_b32_e32 v72, v66, v76, vcc
	v_add3_u32 v74, v70, v74, s0
	v_lshrrev_b32_e32 v73, 16, v73
	v_cmp_o_f32_e32 vcc, v71, v71
	v_bfe_u32 v76, v68, 16, 1
	v_add3_u32 v75, v69, v75, s0
	v_lshrrev_b32_e32 v74, 16, v74
	v_cndmask_b32_e32 v71, v66, v73, vcc
	v_cmp_o_f32_e32 vcc, v70, v70
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[162:165], v[150:153], v[88:91], v93, v98 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v76, v68, v76, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cndmask_b32_e32 v70, v66, v74, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[158:161], v[110:113], v[56:59], v93, v92 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v69, v69
	v_bfe_u32 v73, v87, 16, 1
	v_lshrrev_b32_e32 v76, 16, v76
	v_cndmask_b32_e32 v69, v66, v75, vcc
	v_cmp_o_f32_e32 vcc, v68, v68
	v_bfe_u32 v74, v86, 16, 1
	v_add3_u32 v73, v87, v73, s0
	v_cndmask_b32_e32 v68, v66, v76, vcc
	v_bfe_u32 v75, v85, 16, 1
	v_add3_u32 v74, v86, v74, s0
	v_lshrrev_b32_e32 v73, 16, v73
	v_cmp_o_f32_e32 vcc, v87, v87
	v_bfe_u32 v76, v84, 16, 1
	v_add3_u32 v75, v85, v75, s0
	v_lshrrev_b32_e32 v74, 16, v74
	v_cndmask_b32_e32 v73, v66, v73, vcc
	v_cmp_o_f32_e32 vcc, v86, v86
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[166:169], v[122:125], v[56:59], v93, v96 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v76, v84, v76, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_cndmask_b32_e32 v74, v66, v74, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[158:161], v[118:121], v[52:55], v93, v97 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v85, v85
	v_bfe_u32 v77, v91, 16, 1
	v_lshrrev_b32_e32 v76, 16, v76
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[170:173], v[110:113], v[36:39], v80, v92 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v75, v66, v75, vcc
	v_cmp_o_f32_e32 vcc, v84, v84
	v_bfe_u32 v78, v90, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[170:173], v[118:121], v[32:35], v80, v97 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v77, v91, v77, s0
	v_cndmask_b32_e32 v76, v66, v76, vcc
	v_bfe_u32 v79, v89, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[170:173], v[138:141], v[28:31], v80, v100 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v78, v90, v78, s0
	v_lshrrev_b32_e32 v77, 16, v77
	v_cmp_o_f32_e32 vcc, v91, v91
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[170:173], v[142:145], v[24:27], v80, v102 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v84, v88, 16, 1
	v_add3_u32 v79, v89, v79, s0
	v_lshrrev_b32_e32 v78, 16, v78
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[170:173], v[126:129], v[12:15], v80, v95 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v77, v66, v77, vcc
	v_cmp_o_f32_e32 vcc, v90, v90
	v_lshrrev_b32_e32 v79, 16, v79
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[174:177], v[110:113], v[8:11], v80, v92 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v78, v66, v78, vcc
	v_cmp_o_f32_e32 vcc, v89, v89
	v_bfe_u32 v85, v58, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[174:177], v[118:121], v[4:7], v80, v97 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v79, v66, v79, vcc
	v_cmp_o_f32_e32 vcc, v88, v88
	v_bfe_u32 v86, v57, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[174:177], v[138:141], v[0:3], v80, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v85, v58, v85, s0
	v_bfe_u32 v87, v56, 16, 1
	v_add3_u32 v86, v57, v86, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[174:177], v[142:145], v[20:23], v80, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v85, 16, v85
	v_add3_u32 v87, v56, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[174:177], v[126:129], v[16:19], v80, v95 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v87, 16, v87
	s_mul_hi_u32 s1, s8, s28
	s_mov_b32 s3, 0x27000
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[166:169], v[130:133], v[52:55], v93, v99 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[158:161], v[138:141], v[48:51], v93, v100 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[178:181], v[122:125], v[36:39], v80, v96 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[178:181], v[130:133], v[32:35], v80, v99 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[178:181], v[134:137], v[28:31], v80, v101 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[178:181], v[146:149], v[24:27], v80, v94 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[178:181], v[150:153], v[12:15], v80, v98 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[182:185], v[122:125], v[8:11], v80, v96 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[182:185], v[130:133], v[4:7], v80, v99 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[182:185], v[134:137], v[0:3], v80, v101 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[182:185], v[146:149], v[20:23], v80, v94 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[182:185], v[150:153], v[16:19], v80, v98 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v80, v88, v84, s0
	v_bfe_u32 v84, v59, 16, 1
	v_lshrrev_b32_e32 v80, 16, v80
	v_add3_u32 v84, v59, v84, s0
	v_cndmask_b32_e32 v80, v66, v80, vcc
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v59, v59
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[166:169], v[134:137], v[48:51], v93, v101 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v59, v66, v84, vcc
	v_cmp_o_f32_e32 vcc, v58, v58
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[158:161], v[142:145], v[44:47], v93, v102 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v84, v55, 16, 1
	v_cndmask_b32_e32 v58, v66, v85, vcc
	v_cmp_o_f32_e32 vcc, v57, v57
	v_bfe_u32 v85, v54, 16, 1
	v_add3_u32 v84, v55, v84, s0
	v_cndmask_b32_e32 v57, v66, v86, vcc
	v_cmp_o_f32_e32 vcc, v56, v56
	v_bfe_u32 v86, v53, 16, 1
	v_add3_u32 v85, v54, v85, s0
	v_cndmask_b32_e32 v56, v66, v87, vcc
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v55, v55
	v_bfe_u32 v87, v52, 16, 1
	v_add3_u32 v86, v53, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v55, v66, v84, vcc
	v_cmp_o_f32_e32 vcc, v54, v54
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[166:169], v[146:149], v[44:47], v93, v94 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v87, v52, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v54, v66, v85, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[158:161], v[126:129], v[40:43], v93, v95 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v53, v53
	v_bfe_u32 v84, v51, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v53, v66, v86, vcc
	v_cmp_o_f32_e32 vcc, v52, v52
	v_bfe_u32 v85, v50, 16, 1
	v_add3_u32 v84, v51, v84, s0
	v_cndmask_b32_e32 v52, v66, v87, vcc
	v_bfe_u32 v86, v49, 16, 1
	v_add3_u32 v85, v50, v85, s0
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v51, v51
	v_bfe_u32 v87, v48, 16, 1
	v_add3_u32 v86, v49, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v51, v66, v84, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[166:169], v[150:153], v[40:43], v93, v98 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v87, v48, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v50, v66, v85, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_bfe_u32 v84, v47, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v49, v66, v86, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_bfe_u32 v85, v46, 16, 1
	v_add3_u32 v84, v47, v84, s0
	v_cndmask_b32_e32 v48, v66, v87, vcc
	v_bfe_u32 v86, v45, 16, 1
	v_add3_u32 v85, v46, v85, s0
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v47, v47
	v_bfe_u32 v87, v44, 16, 1
	v_add3_u32 v86, v45, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v47, v66, v84, vcc
	v_cmp_o_f32_e32 vcc, v46, v46
	v_add3_u32 v87, v44, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v46, v66, v85, vcc
	v_cmp_o_f32_e32 vcc, v45, v45
	v_bfe_u32 v84, v43, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v45, v66, v86, vcc
	v_cmp_o_f32_e32 vcc, v44, v44
	v_bfe_u32 v85, v42, 16, 1
	v_add3_u32 v84, v43, v84, s0
	v_cndmask_b32_e32 v44, v66, v87, vcc
	v_bfe_u32 v86, v41, 16, 1
	v_add3_u32 v85, v42, v85, s0
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v43, v43
	v_bfe_u32 v87, v40, 16, 1
	v_add3_u32 v86, v41, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v43, v66, v84, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	v_add3_u32 v87, v40, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v42, v66, v85, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	v_bfe_u32 v84, v39, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v41, v66, v86, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_bfe_u32 v85, v38, 16, 1
	v_add3_u32 v84, v39, v84, s0
	v_cndmask_b32_e32 v40, v66, v87, vcc
	v_bfe_u32 v86, v37, 16, 1
	v_add3_u32 v85, v38, v85, s0
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v39, v39
	v_bfe_u32 v87, v36, 16, 1
	v_add3_u32 v86, v37, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v39, v66, v84, vcc
	v_cmp_o_f32_e32 vcc, v38, v38
	v_add3_u32 v87, v36, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v38, v66, v85, vcc
	v_cmp_o_f32_e32 vcc, v37, v37
	v_bfe_u32 v84, v35, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v37, v66, v86, vcc
	v_cmp_o_f32_e32 vcc, v36, v36
	v_bfe_u32 v85, v34, 16, 1
	v_add3_u32 v84, v35, v84, s0
	v_cndmask_b32_e32 v36, v66, v87, vcc
	v_bfe_u32 v86, v33, 16, 1
	v_add3_u32 v85, v34, v85, s0
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v35, v35
	v_bfe_u32 v87, v32, 16, 1
	v_add3_u32 v86, v33, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v35, v66, v84, vcc
	v_cmp_o_f32_e32 vcc, v34, v34
	v_add3_u32 v87, v32, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v34, v66, v85, vcc
	v_cmp_o_f32_e32 vcc, v33, v33
	v_bfe_u32 v84, v31, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v33, v66, v86, vcc
	v_cmp_o_f32_e32 vcc, v32, v32
	v_bfe_u32 v85, v30, 16, 1
	v_add3_u32 v84, v31, v84, s0
	v_cndmask_b32_e32 v32, v66, v87, vcc
	v_bfe_u32 v86, v29, 16, 1
	v_add3_u32 v85, v30, v85, s0
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v31, v31
	v_bfe_u32 v87, v28, 16, 1
	v_add3_u32 v86, v29, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v31, v66, v84, vcc
	v_cmp_o_f32_e32 vcc, v30, v30
	v_add3_u32 v87, v28, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v30, v66, v85, vcc
	v_cmp_o_f32_e32 vcc, v29, v29
	v_bfe_u32 v84, v27, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v29, v66, v86, vcc
	v_cmp_o_f32_e32 vcc, v28, v28
	v_bfe_u32 v85, v26, 16, 1
	v_add3_u32 v84, v27, v84, s0
	v_cndmask_b32_e32 v28, v66, v87, vcc
	v_bfe_u32 v86, v25, 16, 1
	v_add3_u32 v85, v26, v85, s0
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v27, v27
	v_bfe_u32 v87, v24, 16, 1
	v_add3_u32 v86, v25, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v27, v66, v84, vcc
	v_cmp_o_f32_e32 vcc, v26, v26
	v_add3_u32 v87, v24, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v26, v66, v85, vcc
	v_cmp_o_f32_e32 vcc, v25, v25
	v_bfe_u32 v84, v15, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v25, v66, v86, vcc
	v_cmp_o_f32_e32 vcc, v24, v24
	v_bfe_u32 v85, v14, 16, 1
	v_add3_u32 v84, v15, v84, s0
	v_cndmask_b32_e32 v24, v66, v87, vcc
	v_bfe_u32 v86, v13, 16, 1
	v_add3_u32 v85, v14, v85, s0
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v15, v15
	v_bfe_u32 v87, v12, 16, 1
	v_add3_u32 v86, v13, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v15, v66, v84, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_add3_u32 v87, v12, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v14, v66, v85, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_bfe_u32 v84, v11, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v13, v66, v86, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v85, v10, 16, 1
	v_add3_u32 v84, v11, v84, s0
	v_cndmask_b32_e32 v12, v66, v87, vcc
	v_bfe_u32 v86, v9, 16, 1
	v_add3_u32 v85, v10, v85, s0
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v11, v11
	v_bfe_u32 v87, v8, 16, 1
	v_add3_u32 v86, v9, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v11, v66, v84, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_add3_u32 v87, v8, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v10, v66, v85, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_bfe_u32 v84, v7, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v9, v66, v86, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v85, v6, 16, 1
	v_add3_u32 v84, v7, v84, s0
	v_cndmask_b32_e32 v8, v66, v87, vcc
	v_bfe_u32 v86, v5, 16, 1
	v_add3_u32 v85, v6, v85, s0
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v87, v4, 16, 1
	v_add3_u32 v86, v5, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v7, v66, v84, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v87, v4, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v6, v66, v85, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_bfe_u32 v84, v3, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v5, v66, v86, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v85, v2, 16, 1
	v_add3_u32 v84, v3, v84, s0
	v_cndmask_b32_e32 v4, v66, v87, vcc
	v_bfe_u32 v86, v1, 16, 1
	v_add3_u32 v85, v2, v85, s0
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v3, v3
	v_bfe_u32 v87, v0, 16, 1
	v_add3_u32 v86, v1, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v3, v66, v84, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	v_add3_u32 v87, v0, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v2, v66, v85, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	v_bfe_u32 v84, v23, 16, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v1, v66, v86, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	v_bfe_u32 v85, v22, 16, 1
	v_add3_u32 v84, v23, v84, s0
	v_cndmask_b32_e32 v0, v66, v87, vcc
	v_bfe_u32 v86, v21, 16, 1
	v_add3_u32 v85, v22, v85, s0
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v23, v23
	v_bfe_u32 v87, v20, 16, 1
	v_add3_u32 v86, v21, v86, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v23, v66, v84, vcc
	v_cmp_o_f32_e32 vcc, v22, v22
	v_add3_u32 v87, v20, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v22, v66, v85, vcc
	v_cmp_o_f32_e32 vcc, v21, v21
	v_lshrrev_b32_e32 v87, 16, v87
	v_bfe_u32 v84, v19, 16, 1
	v_cndmask_b32_e32 v21, v66, v86, vcc
	v_cmp_o_f32_e32 vcc, v20, v20
	v_bfe_u32 v85, v18, 16, 1
	v_bfe_u32 v86, v17, 16, 1
	v_cndmask_b32_e32 v20, v66, v87, vcc
	v_bfe_u32 v87, v16, 16, 1
	v_add3_u32 v87, v16, v87, s0
	v_add3_u32 v86, v17, v86, s0
	v_add3_u32 v85, v18, v85, s0
	v_add3_u32 v84, v19, v84, s0
	s_mul_i32 s0, s9, s28
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s8, s28
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshrrev_b32_e32 v84, 16, v84
	v_cmp_o_f32_e32 vcc, v19, v19
	s_add_u32 s0, s10, s0
	v_lshrrev_b32_e32 v85, 16, v85
	v_cndmask_b32_e32 v19, v66, v84, vcc
	v_cmp_o_f32_e32 vcc, v18, v18
	s_addc_u32 s1, s11, s1
	s_lshl_b32 s2, s29, 1
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v18, v66, v85, vcc
	v_cmp_o_f32_e32 vcc, v17, v17
	s_add_u32 s0, s0, s2
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v17, v66, v86, vcc
	v_cmp_o_f32_e32 vcc, v16, v16
	s_addc_u32 s1, s1, 0
	s_and_b32 s2, s8, 0x3fff
	v_cndmask_b32_e32 v16, v66, v87, vcc
	v_lshl_or_b32 v66, v82, 2, v81
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	v_mul_lo_u32 v66, s8, v66
	s_or_b32 s1, s2, s1
	v_lshlrev_b32_e32 v81, 1, v83
	s_or_b32 s1, s1, 2.0
	s_mov_b32 s2, 0x7ffffffd
	v_lshl_add_u32 v82, v66, 1, v81
	s_lshl_b32 s4, s8, 1
	buffer_store_short v62, v82, s[0:3], 0 offen
	v_add_u32_e32 v62, s4, v82
	buffer_store_short v61, v62, s[0:3], 0 offen
	v_add_u32_e32 v61, s4, v62
	buffer_store_short v60, v61, s[0:3], 0 offen
	v_add_u32_e32 v60, s4, v61
	s_lshl_b32 s5, s8, 4
	buffer_store_short v64, v60, s[0:3], 0 offen
	buffer_store_short v72, v82, s[0:3], 0 offen offset:32
	buffer_store_short v67, v62, s[0:3], 0 offen offset:32
	buffer_store_short v65, v61, s[0:3], 0 offen offset:32
	buffer_store_short v63, v60, s[0:3], 0 offen offset:32
	buffer_store_short v68, v82, s[0:3], 0 offen offset:64
	buffer_store_short v69, v62, s[0:3], 0 offen offset:64
	buffer_store_short v70, v61, s[0:3], 0 offen offset:64
	buffer_store_short v71, v60, s[0:3], 0 offen offset:64
	buffer_store_short v76, v82, s[0:3], 0 offen offset:96
	buffer_store_short v75, v62, s[0:3], 0 offen offset:96
	buffer_store_short v74, v61, s[0:3], 0 offen offset:96
	buffer_store_short v73, v60, s[0:3], 0 offen offset:96
	buffer_store_short v80, v82, s[0:3], 0 offen offset:128
	buffer_store_short v79, v62, s[0:3], 0 offen offset:128
	buffer_store_short v78, v61, s[0:3], 0 offen offset:128
	buffer_store_short v77, v60, s[0:3], 0 offen offset:128
	v_add_u32_e32 v60, s5, v66
	v_lshl_add_u32 v61, v60, 1, v81
	buffer_store_short v56, v61, s[0:3], 0 offen
	v_add_u32_e32 v56, s4, v61
	buffer_store_short v57, v56, s[0:3], 0 offen
	v_add_u32_e32 v57, s4, v56
	buffer_store_short v58, v57, s[0:3], 0 offen
	v_add_u32_e32 v58, s4, v57
	buffer_store_short v59, v58, s[0:3], 0 offen
	buffer_store_short v52, v61, s[0:3], 0 offen offset:32
	buffer_store_short v53, v56, s[0:3], 0 offen offset:32
	buffer_store_short v54, v57, s[0:3], 0 offen offset:32
	buffer_store_short v55, v58, s[0:3], 0 offen offset:32
	buffer_store_short v48, v61, s[0:3], 0 offen offset:64
	buffer_store_short v49, v56, s[0:3], 0 offen offset:64
	buffer_store_short v50, v57, s[0:3], 0 offen offset:64
	buffer_store_short v51, v58, s[0:3], 0 offen offset:64
	buffer_store_short v44, v61, s[0:3], 0 offen offset:96
	buffer_store_short v45, v56, s[0:3], 0 offen offset:96
	buffer_store_short v46, v57, s[0:3], 0 offen offset:96
	buffer_store_short v47, v58, s[0:3], 0 offen offset:96
	buffer_store_short v40, v61, s[0:3], 0 offen offset:128
	buffer_store_short v41, v56, s[0:3], 0 offen offset:128
	buffer_store_short v42, v57, s[0:3], 0 offen offset:128
	buffer_store_short v43, v58, s[0:3], 0 offen offset:128
	v_add_u32_e32 v40, s5, v60
	v_lshl_add_u32 v41, v40, 1, v81
	buffer_store_short v36, v41, s[0:3], 0 offen
	v_add_u32_e32 v36, s4, v41
	buffer_store_short v37, v36, s[0:3], 0 offen
	v_add_u32_e32 v37, s4, v36
	buffer_store_short v38, v37, s[0:3], 0 offen
	v_add_u32_e32 v38, s4, v37
	buffer_store_short v39, v38, s[0:3], 0 offen
	buffer_store_short v32, v41, s[0:3], 0 offen offset:32
	buffer_store_short v33, v36, s[0:3], 0 offen offset:32
	buffer_store_short v34, v37, s[0:3], 0 offen offset:32
	buffer_store_short v35, v38, s[0:3], 0 offen offset:32
	buffer_store_short v28, v41, s[0:3], 0 offen offset:64
	buffer_store_short v29, v36, s[0:3], 0 offen offset:64
	buffer_store_short v30, v37, s[0:3], 0 offen offset:64
	buffer_store_short v31, v38, s[0:3], 0 offen offset:64
	buffer_store_short v24, v41, s[0:3], 0 offen offset:96
	buffer_store_short v25, v36, s[0:3], 0 offen offset:96
	buffer_store_short v26, v37, s[0:3], 0 offen offset:96
	buffer_store_short v27, v38, s[0:3], 0 offen offset:96
	buffer_store_short v12, v41, s[0:3], 0 offen offset:128
	buffer_store_short v13, v36, s[0:3], 0 offen offset:128
	buffer_store_short v14, v37, s[0:3], 0 offen offset:128
	buffer_store_short v15, v38, s[0:3], 0 offen offset:128
	v_add_u32_e32 v12, s5, v40
	v_lshl_add_u32 v12, v12, 1, v81
	buffer_store_short v8, v12, s[0:3], 0 offen
	v_add_u32_e32 v8, s4, v12
	buffer_store_short v9, v8, s[0:3], 0 offen
	v_add_u32_e32 v9, s4, v8
	buffer_store_short v10, v9, s[0:3], 0 offen
	v_add_u32_e32 v10, s4, v9
	buffer_store_short v11, v10, s[0:3], 0 offen
	buffer_store_short v4, v12, s[0:3], 0 offen offset:32
	buffer_store_short v5, v8, s[0:3], 0 offen offset:32
	buffer_store_short v6, v9, s[0:3], 0 offen offset:32
	buffer_store_short v7, v10, s[0:3], 0 offen offset:32
	buffer_store_short v0, v12, s[0:3], 0 offen offset:64
	buffer_store_short v1, v8, s[0:3], 0 offen offset:64
	buffer_store_short v2, v9, s[0:3], 0 offen offset:64
	buffer_store_short v3, v10, s[0:3], 0 offen offset:64
	buffer_store_short v20, v12, s[0:3], 0 offen offset:96
	buffer_store_short v21, v8, s[0:3], 0 offen offset:96
	buffer_store_short v22, v9, s[0:3], 0 offen offset:96
	buffer_store_short v23, v10, s[0:3], 0 offen offset:96
	buffer_store_short v16, v12, s[0:3], 0 offen offset:128
	buffer_store_short v17, v8, s[0:3], 0 offen offset:128
	buffer_store_short v18, v9, s[0:3], 0 offen offset:128
	buffer_store_short v19, v10, s[0:3], 0 offen offset:128
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_256x160x256_13568x14720x8704
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
		.amdhsa_next_free_vgpr 246
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 248
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
	.size	wave_mxfp4_static_gemm_256x160x256_13568x14720x8704, .Lfunc_end0-wave_mxfp4_static_gemm_256x160x256_13568x14720x8704

	.set wave_mxfp4_static_gemm_256x160x256_13568x14720x8704.num_vgpr, 246
	.set wave_mxfp4_static_gemm_256x160x256_13568x14720x8704.num_agpr, 0
	.set wave_mxfp4_static_gemm_256x160x256_13568x14720x8704.numbered_sgpr, 70
	.set wave_mxfp4_static_gemm_256x160x256_13568x14720x8704.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_256x160x256_13568x14720x8704.private_seg_size, 0
	.set wave_mxfp4_static_gemm_256x160x256_13568x14720x8704.uses_vcc, 1
	.set wave_mxfp4_static_gemm_256x160x256_13568x14720x8704.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_256x160x256_13568x14720x8704.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_256x160x256_13568x14720x8704.has_recursion, 0
	.set wave_mxfp4_static_gemm_256x160x256_13568x14720x8704.has_indirect_call, 0
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
    .name:           wave_mxfp4_static_gemm_256x160x256_13568x14720x8704
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     76
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_256x160x256_13568x14720x8704.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     246
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
