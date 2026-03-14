; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_256x224x256_7168x33280x10752
	.p2align	8
	.type	wave_mxfp4_static_gemm_256x224x256_7168x33280x10752,@function
wave_mxfp4_static_gemm_256x224x256_7168x33280x10752:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v118, 0x3ff, v0
	v_bfe_u32 v3, v0, 10, 10
	v_lshrrev_b32_e32 v4, 6, v118
	v_lshlrev_b32_e32 v0, 5, v3
	v_lshl_or_b32 v1, v4, 3, v0
	s_mov_b64 s[24:25], s[2:3]
	v_readfirstlane_b32 s2, v1
	v_lshrrev_b32_e32 v1, 3, v118
	s_lshl_b32 s33, s16, 8
	v_or3_b32 v2, v1, v0, s33
	v_xor_b32_e32 v1, v1, v118
	v_lshlrev_b32_e32 v1, 4, v1
	s_mov_b64 s[20:21], s[6:7]
	v_and_b32_e32 v1, 0x70, v1
	v_mul_u32_u24_e32 v2, 0x1500, v2
	s_and_b32 s6, s25, 0xffff
	s_lshl_b32 s37, s2, 7
	s_or_b32 s25, s6, 0x55000000
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_or_b32_e32 v5, v2, v1
	s_mov_b32 m0, s37
	s_or_b32 s38, s37, 0x2000
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_add_u32_e32 v6, 0x54000, v5
	s_mov_b32 m0, s38
	s_or_b32 s39, s37, 0x4000
	buffer_load_dwordx4 v6, s[24:27], 0 offen lds
	v_add_u32_e32 v6, 0xa8000, v5
	s_mov_b32 m0, s39
	s_or_b32 s40, s37, 0x6000
	buffer_load_dwordx4 v6, s[24:27], 0 offen lds
	v_add_u32_e32 v5, 0xfc000, v5
	s_mov_b32 m0, s40
	v_bfe_u32 v9, v118, 2, 3
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_lshlrev_b32_e32 v5, 3, v3
	v_lshl_or_b32 v5, v4, 1, v5
	v_lshrrev_b32_e32 v6, 2, v118
	v_readfirstlane_b32 s2, v5
	v_lshrrev_b32_e32 v5, 5, v118
	v_and_b32_e32 v10, 31, v118
	v_lshlrev_b32_e32 v9, 2, v9
	v_bitop3_b32 v12, v6, v5, 7 bitop3:0x6c
	v_sub_u32_e32 v9, v10, v9
	v_lshl_add_u32 v9, v12, 2, v9
	v_ashrrev_i32_e32 v10, 31, v9
	v_xor_b32_e32 v9, v10, v9
	v_ashrrev_i32_e32 v11, 31, v9
	v_lshrrev_b32_e32 v11, 29, v11
	v_add_u32_e32 v9, v9, v11
	v_ashrrev_i32_e32 v9, 3, v9
	v_lshrrev_b32_e32 v15, 1, v12
	v_xor_b32_e32 v14, v9, v10
	v_and_b32_e32 v9, 0xfc, v118
	v_lshlrev_b32_e32 v10, 7, v15
	v_lshlrev_b32_e32 v7, 6, v12
	v_lshlrev_b32_e32 v8, 2, v5
	v_add_u32_e32 v16, v9, v10
	v_add3_u32 v13, v0, v118, v7
	v_lshlrev_b32_e32 v11, 7, v14
	v_sub_u32_e32 v16, v8, v16
	v_add3_u32 v13, v16, v13, v11
	v_ashrrev_i32_e32 v16, 31, v13
	v_xor_b32_e32 v13, v16, v13
	s_mov_b32 s41, 0x30c30c31
	v_mul_hi_i32 v13, v13, s41
	v_lshrrev_b32_e32 v17, 31, v13
	v_ashrrev_i32_e32 v13, 8, v13
	v_add_u32_e32 v13, v13, v17
	v_xor_b32_e32 v16, v13, v16
	v_sub_u32_e32 v13, v5, v6
	v_lshlrev_b32_e32 v18, 4, v13
	v_lshlrev_b32_e32 v119, 2, v118
	v_lshlrev_b32_e32 v13, 9, v14
	v_lshlrev_b32_e32 v14, 8, v12
	v_add3_u32 v18, v18, v119, v14
	v_lshlrev_b32_e32 v12, 9, v15
	s_mul_i32 s36, s17, 0xe0
	v_sub_u32_e32 v15, v18, v12
	v_add_u32_e32 v17, s36, v16
	v_add_u32_e32 v18, v15, v13
	v_mul_i32_i24_e32 v16, 0xffffeb00, v16
	v_lshlrev_b32_e32 v15, 7, v3
	s_movk_i32 s3, 0x1500
	v_add3_u32 v16, v18, v15, v16
	s_and_b32 s6, s21, 0xffff
	s_lshl_b32 s42, s2, 7
	s_or_b32 s21, s6, 0x55000000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_mad_i32_i24 v16, v17, s3, v16
	s_add_i32 m0, s42, 0x10000
	s_or_b32 s43, s42, 0x800
	buffer_load_dword v16, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0x15000, v16
	s_add_i32 m0, s43, 0x10000
	s_or_b32 s44, s42, 0x1000
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0x2a000, v16
	s_add_i32 m0, s44, 0x10000
	s_or_b32 s45, s42, 0x1800
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0x3f000, v16
	s_add_i32 m0, s45, 0x10000
	s_or_b32 s46, s42, 0x2000
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0x54000, v16
	s_add_i32 m0, s46, 0x10000
	s_or_b32 s47, s42, 0x2800
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0x69000, v16
	s_add_i32 m0, s47, 0x10000
	s_or_b32 s48, s42, 0x3000
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0x7e000, v16
	s_add_i32 m0, s48, 0x10000
	s_or_b32 s49, s42, 0x3800
	buffer_load_dword v17, s[20:23], 0 offen lds
	s_add_i32 m0, s49, 0x10000
	v_add_u32_e32 v17, 0x93000, v16
	s_cmpk_lt_u32 s17, 0x94
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0xa8000, v16
	v_bfrev_b32_e32 v120, -2
	s_cselect_b64 s[6:7], -1, 0
	s_or_b32 s50, s42, 0x4000
	v_cndmask_b32_e64 v17, v120, v17, s[6:7]
	s_add_i32 m0, s50, 0x10000
	s_or_b32 s51, s42, 0x4800
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0xbd000, v16
	v_cndmask_b32_e64 v17, v120, v17, s[6:7]
	s_add_i32 m0, s51, 0x10000
	s_or_b32 s52, s42, 0x5000
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0xd2000, v16
	v_cndmask_b32_e64 v17, v120, v17, s[6:7]
	s_add_i32 m0, s52, 0x10000
	s_or_b32 s53, s42, 0x5800
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0xe7000, v16
	v_cndmask_b32_e64 v17, v120, v17, s[6:7]
	s_add_i32 m0, s53, 0x10000
	s_or_b32 s54, s42, 0x6000
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0xfc000, v16
	v_cndmask_b32_e64 v17, v120, v17, s[6:7]
	s_add_i32 m0, s54, 0x10000
	s_or_b32 s55, s42, 0x6800
	s_mul_i32 s15, s15, s33
	s_mul_hi_u32 s2, s14, s33
	buffer_load_dword v17, s[20:23], 0 offen lds
	s_add_i32 m0, s55, 0x10000
	s_add_i32 s2, s2, s15
	s_mul_i32 s3, s14, s33
	v_add_u32_e32 v16, 0x111000, v16
	s_add_u32 s16, s4, s3
	v_cndmask_b32_e64 v16, v120, v16, s[6:7]
	v_lshrrev_b32_e32 v18, 4, v118
	v_bfe_u32 v117, v118, 4, 2
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s14, 0x3fff
	buffer_load_dword v16, s[20:23], 0 offen lds
	v_mul_i32_i24_e32 v17, 0xffffffc0, v18
	v_lshlrev_b32_e32 v16, 6, v117
	s_bitset1_b32 s3, 14
	v_and_b32_e32 v116, 0xc0, v118
	v_add3_u32 v22, v17, v119, v16
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	s_or_b32 s17, s2, s3
	v_mad_u64_u32 v[20:21], s[2:3], s14, v116, v[22:23]
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	v_lshl_add_u32 v19, s14, 5, v20
	s_movk_i32 s28, 0x70
	buffer_load_dword v125, v20, s[16:19], 0 offen
	buffer_load_dword v124, v19, s[16:19], 0 offen
	v_mad_i32_i24 v19, v18, -16, v118
	v_mad_u32_u24 v112, v3, s28, v19
	v_ashrrev_i16_e32 v20, 15, v112
	v_lshrrev_b16_e32 v20, 11, v20
	v_add_u16_e32 v20, v112, v20
	v_and_b32_e32 v20, 0xffffffe0, v20
	v_sub_u16_e32 v20, v112, v20
	v_bfe_i32 v21, v20, 0, 16
	v_ashrrev_i32_e32 v23, 31, v21
	v_add_u16_e32 v24, 32, v20
	v_cmp_gt_i16_e32 vcc, 0, v20
	s_load_dwordx2 s[12:13], s[0:1], 0x40
	s_movk_i32 s5, 0x150
	v_cndmask_b32_e32 v20, v21, v24, vcc
	v_cndmask_b32_e64 v21, v23, 0, vcc
	v_xor_b32_e32 v20, v21, v20
	v_lshrrev_b32_e32 v23, 28, v20
	v_add_u32_e32 v20, v20, v23
	v_ashrrev_i32_e32 v20, 4, v20
	v_xor_b32_e32 v20, v20, v21
	v_add_u32_e32 v21, v22, v20
	v_ashrrev_i32_e32 v22, 31, v21
	v_xor_b32_e32 v23, v22, v21
	v_mul_hi_i32 v23, v23, s41
	v_lshrrev_b32_e32 v24, 31, v23
	v_ashrrev_i32_e32 v23, 6, v23
	v_add_u32_e32 v23, v23, v24
	v_ashrrev_i32_e32 v24, 31, v112
	v_xor_b32_e32 v25, v24, v112
	v_ashrrev_i32_e32 v26, 31, v25
	v_lshrrev_b32_e32 v26, 27, v26
	v_add_u32_e32 v25, v25, v26
	v_lshrrev_b32_e32 v25, 5, v25
	v_xor_b32_e32 v24, v25, v24
	v_lshlrev_b32_e32 v121, 5, v24
	v_xad_u32 v23, v23, v22, v121
	v_mul_hi_i32 v22, v21, s41
	v_lshrrev_b32_e32 v24, 31, v22
	v_ashrrev_i32_e32 v22, 6, v22
	v_add_u32_e32 v22, v22, v24
	v_mul_lo_u32 v22, v22, s5
	v_sub_u32_e32 v22, v21, v22
	v_add_u32_e32 v24, 0x150, v22
	v_cmp_gt_i32_e32 vcc, 0, v22
	v_add_u32_e32 v25, s36, v23
	s_mov_b32 s4, 0x8200
	v_cndmask_b32_e32 v22, v22, v24, vcc
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s13, s36
	s_mul_hi_u32 s3, s12, s36
	s_add_i32 s13, s3, s2
	v_mad_u64_u32 v[22:23], s[2:3], s12, v23, v[22:23]
	v_cmp_gt_i32_e32 vcc, s4, v25
	v_add_u32_e32 v23, 2, v21
	v_sub_u32_e32 v24, -3, v21
	v_cndmask_b32_e32 v30, v120, v22, vcc
	v_cmp_gt_i32_e32 vcc, -2, v21
	v_add_u32_e32 v113, 16, v112
	s_mul_i32 s15, s12, s36
	v_cndmask_b32_e32 v21, v23, v24, vcc
	v_mul_hi_i32 v21, v21, s41
	v_lshrrev_b32_e32 v24, 31, v21
	v_ashrrev_i32_e32 v21, 6, v21
	v_add_u32_e32 v21, v21, v24
	v_cndmask_b32_e64 v24, 0, -1, vcc
	v_xor_b32_e32 v21, v21, v24
	v_add_u32_e32 v24, v21, v121
	v_add_u32_e32 v31, s36, v24
	v_mul_i32_i24_e32 v21, 0xfffffeb0, v21
	v_mul_lo_u32 v24, s12, v24
	v_add3_u32 v23, v23, v21, v24
	v_cmp_gt_i32_e32 vcc, s4, v31
	v_sub_u32_e32 v21, 0xffef, v112
	v_mad_u32_u24 v24, v3, 7, v119
	v_cndmask_b32_e32 v32, v120, v23, vcc
	v_cmp_gt_i32_e32 vcc, -16, v112
	s_add_u32 s28, s8, s15
	s_addc_u32 s2, s9, s13
	v_cndmask_b32_e32 v21, v113, v21, vcc
	v_ashrrev_i16_e32 v26, 15, v21
	v_lshrrev_b16_e32 v26, 11, v26
	v_add_u16_e32 v21, v21, v26
	v_ashrrev_i16_e32 v21, 5, v21
	v_cndmask_b32_e64 v26, 0, -1, vcc
	v_xor_b32_e32 v21, v21, v26
	v_bfe_i32 v33, v21, 0, 16
	v_mad_i32_i24 v21, v33, -2, v17
	v_add3_u32 v28, v21, v24, v16
	v_add_u32_e32 v29, 1, v28
	v_sub_u32_e32 v24, -2, v28
	v_cmp_gt_i32_e32 vcc, -1, v28
	s_and_b32 s3, s12, 0x3fff
	s_bitset1_b32 s3, 14
	v_cndmask_b32_e32 v24, v29, v24, vcc
	v_mul_hi_i32 v24, v24, s41
	v_lshrrev_b32_e32 v26, 31, v24
	v_ashrrev_i32_e32 v24, 6, v24
	v_add_u32_e32 v24, v24, v26
	v_cndmask_b32_e64 v26, 0, -1, vcc
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	v_xor_b32_e32 v34, v24, v26
	v_lshlrev_b32_e32 v122, 5, v33
	s_or_b32 s29, s2, s3
	s_movk_i32 s2, 0xfeb0
	v_add_u32_e32 v26, v34, v122
	v_add_u32_e32 v35, s36, v26
	v_mad_i32_i24 v24, v34, s2, v29
	v_mad_u64_u32 v[26:27], s[2:3], v26, s12, v[24:25]
	v_cmp_gt_i32_e32 vcc, s4, v35
	v_sub_u32_e32 v35, -4, v28
	s_mov_b32 s8, 0x81e0
	v_cndmask_b32_e32 v27, v120, v26, vcc
	v_add_u32_e32 v26, 3, v28
	v_cmp_gt_i32_e32 vcc, -3, v28
	s_lshl_b32 s13, s12, 5
	v_add_u32_e32 v37, s13, v23
	v_cndmask_b32_e32 v26, v26, v35, vcc
	v_mul_hi_i32 v26, v26, s41
	v_lshrrev_b32_e32 v28, 31, v26
	v_ashrrev_i32_e32 v26, 6, v26
	v_add_u32_e32 v26, v26, v28
	v_cndmask_b32_e64 v28, 0, -1, vcc
	v_xor_b32_e32 v35, v26, v28
	v_add_u32_e32 v28, v35, v122
	v_mul_i32_i24_e32 v26, 0xfffffeb0, v35
	v_add_u32_e32 v36, s36, v28
	v_add3_u32 v26, v29, v26, 2
	v_mad_u64_u32 v[28:29], s[2:3], v28, s12, v[26:27]
	v_cmp_gt_i32_e32 vcc, s4, v36
	v_add_u32_e32 v29, s13, v22
	s_movk_i32 s2, 0xffd0
	v_cndmask_b32_e32 v28, v120, v28, vcc
	v_cmp_gt_i32_e32 vcc, s8, v25
	v_add_u32_e32 v114, 48, v112
	v_sub_u32_e32 v22, 0xffcf, v112
	v_cndmask_b32_e32 v36, v120, v29, vcc
	v_cmp_gt_i32_e32 vcc, s8, v31
	v_mov_b32_e32 v39, 5
	s_mov_b32 s30, s26
	v_cndmask_b32_e32 v38, v120, v37, vcc
	v_cmp_gt_i32_e32 vcc, s2, v112
	s_mov_b32 s31, s27
	s_mov_b32 s9, 0x81c0
	v_cndmask_b32_e32 v22, v114, v22, vcc
	v_ashrrev_i16_e32 v23, 15, v22
	v_lshrrev_b16_e32 v23, 11, v23
	v_add_u16_e32 v22, v22, v23
	v_ashrrev_i16_e32 v22, 5, v22
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_xor_b32_e32 v22, v22, v23
	v_lshlrev_b32_sdwa v123, v39, sext(v22) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_e32 v22, v123, v34
	v_add_u32_e32 v40, s36, v22
	v_mad_u64_u32 v[22:23], s[2:3], s12, v22, v[24:25]
	v_cmp_gt_i32_e32 vcc, s4, v40
	v_add_u32_e32 v115, 0x50, v112
	s_mov_b32 s15, 0x81a0
	v_cndmask_b32_e32 v40, v120, v22, vcc
	v_add_u32_e32 v22, v123, v35
	v_add_u32_e32 v41, s36, v22
	v_mad_u64_u32 v[22:23], s[2:3], s12, v22, v[26:27]
	v_cmp_gt_i32_e32 vcc, s4, v41
	s_movk_i32 s2, 0xffb0
	s_movk_i32 s57, 0xeb00
	v_cndmask_b32_e32 v22, v120, v22, vcc
	buffer_load_ubyte v160, v30, s[28:31], 0 offen
	buffer_load_ubyte v153, v32, s[28:31], 0 offen
	buffer_load_ubyte v159, v27, s[28:31], 0 offen
	buffer_load_ubyte v152, v28, s[28:31], 0 offen
	buffer_load_ubyte v158, v36, s[28:31], 0 offen
	buffer_load_ubyte v151, v38, s[28:31], 0 offen
	buffer_load_ubyte v157, v40, s[28:31], 0 offen
	buffer_load_ubyte v150, v22, s[28:31], 0 offen
	v_add_u32_e32 v27, s13, v29
	v_cmp_gt_i32_e32 vcc, s9, v25
	v_add_u32_e32 v29, s13, v37
	v_sub_u32_e32 v22, 0xffaf, v112
	v_cndmask_b32_e32 v28, v120, v27, vcc
	v_cmp_gt_i32_e32 vcc, s9, v31
	s_mov_b32 s56, 0x10000
	s_movk_i32 s18, 0x3800
	v_cndmask_b32_e32 v30, v120, v29, vcc
	v_cmp_gt_i32_e32 vcc, s2, v112
	s_movk_i32 s58, 0xffc0
	s_nop 0
	v_cndmask_b32_e32 v22, v115, v22, vcc
	v_ashrrev_i16_e32 v23, 15, v22
	v_lshrrev_b16_e32 v23, 11, v23
	v_add_u16_e32 v22, v22, v23
	v_ashrrev_i16_e32 v22, 5, v22
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_xor_b32_e32 v22, v22, v23
	v_lshlrev_b32_sdwa v126, v39, sext(v22) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_e32 v22, v126, v34
	v_add_u32_e32 v32, s36, v22
	v_mad_u64_u32 v[22:23], s[2:3], s12, v22, v[24:25]
	v_cmp_gt_i32_e32 vcc, s4, v32
	s_nop 1
	v_cndmask_b32_e32 v24, v120, v22, vcc
	v_add_u32_e32 v22, v126, v35
	v_add_u32_e32 v32, s36, v22
	v_mad_u64_u32 v[22:23], s[2:3], s12, v22, v[26:27]
	v_cmp_gt_i32_e32 vcc, s4, v32
	v_add_u32_e32 v23, s13, v27
	v_cmp_eq_u32_e64 s[2:3], 0, v3
	v_cndmask_b32_e32 v22, v120, v22, vcc
	v_cmp_gt_i32_e32 vcc, s15, v25
	v_add_u32_e32 v25, s13, v29
	s_mov_b32 s13, 0
	v_cndmask_b32_e32 v23, v120, v23, vcc
	v_cmp_gt_i32_e32 vcc, s15, v31
	s_nop 1
	v_cndmask_b32_e32 v25, v120, v25, vcc
	buffer_load_ubyte v163, v28, s[28:31], 0 offen
	buffer_load_ubyte v156, v30, s[28:31], 0 offen
	buffer_load_ubyte v162, v24, s[28:31], 0 offen
	buffer_load_ubyte v155, v22, s[28:31], 0 offen
	buffer_load_ubyte v161, v23, s[28:31], 0 offen
	buffer_load_ubyte v154, v25, s[28:31], 0 offen
	v_mul_i32_i24_e32 v22, -16, v18
	v_mul_u32_u24_e32 v23, 7, v3
	v_mul_i32_i24_e32 v24, -2, v33
	s_and_b64 vcc, exec, s[2:3]
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	v_lshlrev_b32_e32 v27, 7, v118
	v_lshlrev_b32_e32 v18, 11, v18
	v_and_b32_e32 v25, 7, v118
	v_sub_u32_e32 v18, v27, v18
	v_mul_lo_u32 v3, v3, s18
	v_bitop3_b32 v26, v117, v118, 7 bitop3:0x78
	v_lshl_add_u32 v4, v4, 13, v18
	v_add_u32_e32 v3, v18, v3
	v_bitop3_b32 v18, v117, v25, 4 bitop3:0x36
	v_lshlrev_b32_e32 v26, 4, v26
	v_lshlrev_b32_e32 v18, 4, v18
	v_or_b32_e32 v127, v4, v26
	v_or_b32_e32 v128, v18, v4
	v_add_u32_e32 v4, s36, v123
	v_or_b32_e32 v130, v18, v3
	v_add_u32_e32 v18, s36, v126
	v_sub_u32_e32 v133, 0x8200, v4
	v_sub_u32_e32 v4, 0, v17
	v_or_b32_e32 v129, v3, v26
	v_add_u32_e32 v3, s36, v122
	v_lshlrev_b32_e32 v25, 4, v117
	v_sub_u32_e32 v134, 0x8200, v18
	v_sub_u32_e32 v18, v4, v20
	v_sub_u32_e32 v4, v4, v24
	v_sub_u32_e32 v132, 0x8200, v3
	v_add_u32_e32 v3, v19, v25
	v_sub_u32_e32 v4, v4, v119
	v_sub_u32_e32 v4, v4, v23
	v_sub_u32_e32 v144, 0xffffffbf, v3
	v_add3_u32 v3, v13, v14, v15
	v_sub_u32_e32 v141, v4, v16
	v_lshl_add_u32 v3, v5, 4, v3
	v_lshlrev_b32_e32 v4, 4, v6
	v_sub_u32_e32 v3, v3, v4
	v_sub_u32_e32 v146, v3, v12
	v_add_u32_e32 v3, v11, v7
	s_load_dwordx2 s[34:35], s[0:1], 0x48
	v_add3_u32 v0, v3, v0, v8
	v_add_u32_e32 v3, v0, v118
	v_sub_u32_e32 v3, v9, v3
	s_movk_i32 s0, 0xfdff
	v_sub_u32_e32 v0, v0, v9
	v_sub_u32_e32 v18, v18, v119
	v_add_u32_e32 v17, v17, v16
	v_add3_u32 v147, v10, v3, s0
	v_sub_u32_e32 v0, v0, v10
	s_mov_b32 s0, 0xfc080
	v_mov_b32_e32 v40, 0
	v_add_u32_e32 v131, s36, v121
	v_sub_u32_e32 v135, v18, v16
	v_add_u32_e32 v136, v17, v20
	v_add_u32_e32 v137, 32, v121
	v_add_u32_e32 v138, 64, v121
	v_add_u32_e32 v139, 0x60, v121
	v_add3_u32 v140, v21, v16, v23
	v_add_u32_e32 v142, 32, v116
	v_add_u32_e32 v143, 0x100, v17
	v_add_u32_e32 v145, v22, v25
	v_add_u32_e32 v148, 0x200, v0
	v_add3_u32 v149, v2, v1, s0
	s_mov_b32 s63, 0x8000
	s_mov_b32 s64, 0x17000
	s_mov_b32 s0, 0
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	s_movk_i32 s1, 0xff00
	s_movk_i32 s59, 0xfeff
	s_mov_b32 s30, s26
	s_mov_b32 s31, s27
	s_movk_i32 s60, 0xfefd
	s_movk_i32 s61, 0xfefe
	v_mov_b32_e32 v41, v40
	v_mov_b32_e32 v42, v40
	v_mov_b32_e32 v43, v40
	v_mov_b32_e32 v108, v40
	v_mov_b32_e32 v109, v40
	v_mov_b32_e32 v110, v40
	v_mov_b32_e32 v111, v40
	v_mov_b32_e32 v104, v40
	v_mov_b32_e32 v105, v40
	v_mov_b32_e32 v106, v40
	v_mov_b32_e32 v107, v40
	v_mov_b32_e32 v100, v40
	v_mov_b32_e32 v101, v40
	v_mov_b32_e32 v102, v40
	v_mov_b32_e32 v103, v40
	v_mov_b32_e32 v96, v40
	v_mov_b32_e32 v97, v40
	v_mov_b32_e32 v98, v40
	v_mov_b32_e32 v99, v40
	v_mov_b32_e32 v92, v40
	v_mov_b32_e32 v93, v40
	v_mov_b32_e32 v94, v40
	v_mov_b32_e32 v95, v40
	v_mov_b32_e32 v88, v40
	v_mov_b32_e32 v89, v40
	v_mov_b32_e32 v90, v40
	v_mov_b32_e32 v91, v40
	v_mov_b32_e32 v84, v40
	v_mov_b32_e32 v85, v40
	v_mov_b32_e32 v86, v40
	v_mov_b32_e32 v87, v40
	v_mov_b32_e32 v80, v40
	v_mov_b32_e32 v81, v40
	v_mov_b32_e32 v82, v40
	v_mov_b32_e32 v83, v40
	v_mov_b32_e32 v76, v40
	v_mov_b32_e32 v77, v40
	v_mov_b32_e32 v78, v40
	v_mov_b32_e32 v79, v40
	v_mov_b32_e32 v72, v40
	v_mov_b32_e32 v73, v40
	v_mov_b32_e32 v74, v40
	v_mov_b32_e32 v75, v40
	v_mov_b32_e32 v68, v40
	v_mov_b32_e32 v69, v40
	v_mov_b32_e32 v70, v40
	v_mov_b32_e32 v71, v40
	v_mov_b32_e32 v64, v40
	v_mov_b32_e32 v65, v40
	v_mov_b32_e32 v66, v40
	v_mov_b32_e32 v67, v40
	v_mov_b32_e32 v60, v40
	v_mov_b32_e32 v61, v40
	v_mov_b32_e32 v62, v40
	v_mov_b32_e32 v63, v40
	v_mov_b32_e32 v56, v40
	v_mov_b32_e32 v57, v40
	v_mov_b32_e32 v58, v40
	v_mov_b32_e32 v59, v40
	v_mov_b32_e32 v52, v40
	v_mov_b32_e32 v53, v40
	v_mov_b32_e32 v54, v40
	v_mov_b32_e32 v55, v40
	v_mov_b32_e32 v48, v40
	v_mov_b32_e32 v49, v40
	v_mov_b32_e32 v50, v40
	v_mov_b32_e32 v51, v40
	v_mov_b32_e32 v44, v40
	v_mov_b32_e32 v45, v40
	v_mov_b32_e32 v46, v40
	v_mov_b32_e32 v47, v40
	v_mov_b32_e32 v36, v40
	v_mov_b32_e32 v37, v40
	v_mov_b32_e32 v38, v40
	v_mov_b32_e32 v39, v40
	v_mov_b32_e32 v20, v40
	v_mov_b32_e32 v21, v40
	v_mov_b32_e32 v22, v40
	v_mov_b32_e32 v23, v40
	v_mov_b32_e32 v8, v40
	v_mov_b32_e32 v9, v40
	v_mov_b32_e32 v10, v40
	v_mov_b32_e32 v11, v40
	v_mov_b32_e32 v4, v40
	v_mov_b32_e32 v5, v40
	v_mov_b32_e32 v6, v40
	v_mov_b32_e32 v7, v40
	v_mov_b32_e32 v12, v40
	v_mov_b32_e32 v13, v40
	v_mov_b32_e32 v14, v40
	v_mov_b32_e32 v15, v40
	v_mov_b32_e32 v16, v40
	v_mov_b32_e32 v17, v40
	v_mov_b32_e32 v18, v40
	v_mov_b32_e32 v19, v40
	v_mov_b32_e32 v24, v40
	v_mov_b32_e32 v25, v40
	v_mov_b32_e32 v26, v40
	v_mov_b32_e32 v27, v40
	v_mov_b32_e32 v28, v40
	v_mov_b32_e32 v29, v40
	v_mov_b32_e32 v30, v40
	v_mov_b32_e32 v31, v40
	v_mov_b32_e32 v32, v40
	v_mov_b32_e32 v33, v40
	v_mov_b32_e32 v34, v40
	v_mov_b32_e32 v35, v40
	v_mov_b32_e32 v0, v40
	v_mov_b32_e32 v1, v40
	v_mov_b32_e32 v2, v40
	v_mov_b32_e32 v3, v40
	v_mov_b32_e32 v164, v146
.LBB0_3:
	s_mov_b32 s62, s13
	s_mov_b32 s13, s63
	s_mov_b32 s63, s56
	s_mov_b32 s56, s64
	v_add_u32_e32 v165, 0xfff04000, v149
	s_add_i32 m0, s13, s37
	s_barrier
	buffer_load_dwordx4 v165, s[24:27], 0 offen lds
	v_add_u32_e32 v165, 0xfff58000, v149
	s_add_i32 m0, s13, s38
	s_nop 0
	buffer_load_dwordx4 v165, s[24:27], 0 offen lds
	v_add_u32_e32 v165, 0xfffac000, v149
	s_add_i32 m0, s13, s39
	s_nop 0
	buffer_load_dwordx4 v165, s[24:27], 0 offen lds
	v_add_u32_e32 v165, v118, v148
	v_cmp_gt_i32_e32 vcc, 0, v165
	s_add_i32 m0, s13, s40
	s_nop 0
	v_cndmask_b32_e32 v166, v165, v147, vcc
	v_mul_hi_i32 v166, v166, s41
	v_lshrrev_b32_e32 v167, 31, v166
	v_ashrrev_i32_e32 v166, 8, v166
	v_add_u32_e32 v166, v166, v167
	v_ashrrev_i32_e32 v165, 31, v165
	v_xor_b32_e32 v165, v166, v165
	v_add_u32_e32 v166, s36, v165
	v_mul_i32_i24_e32 v166, 0x1500, v166
	v_mad_i32_i24 v165, v165, s57, v166
	v_add3_u32 v166, v164, v119, v165
	buffer_load_dwordx4 v149, s[24:27], 0 offen lds
	v_add_u32_e32 v167, 0x800, v166
	s_add_i32 m0, s64, s42
	v_add3_u32 v165, v119, v146, v165
	buffer_load_dword v167, s[20:23], 0 offen lds
	v_add_u32_e32 v167, 0x15800, v166
	s_add_i32 m0, s64, s43
	s_nop 0
	buffer_load_dword v167, s[20:23], 0 offen lds
	v_add_u32_e32 v167, 0x2a800, v166
	s_add_i32 m0, s64, s44
	s_nop 0
	buffer_load_dword v167, s[20:23], 0 offen lds
	v_add_u32_e32 v167, 0x3f800, v166
	s_add_i32 m0, s64, s45
	s_nop 0
	buffer_load_dword v167, s[20:23], 0 offen lds
	v_add_u32_e32 v167, 0x54800, v166
	s_add_i32 m0, s64, s46
	s_nop 0
	buffer_load_dword v167, s[20:23], 0 offen lds
	v_add_u32_e32 v167, 0x69800, v166
	s_add_i32 m0, s64, s47
	s_nop 0
	buffer_load_dword v167, s[20:23], 0 offen lds
	v_add_u32_e32 v167, 0x7e800, v166
	s_add_i32 m0, s64, s48
	v_add_u32_e32 v166, 0x93800, v166
	buffer_load_dword v167, s[20:23], 0 offen lds
	s_add_i32 m0, s64, s49
	s_nop 0
	buffer_load_dword v166, s[20:23], 0 offen lds
	v_add_u32_e32 v166, 0xa8800, v165
	v_cndmask_b32_e64 v166, v120, v166, s[6:7]
	s_add_i32 m0, s64, s50
	s_nop 0
	buffer_load_dword v166, s[20:23], 0 offen lds
	v_add_u32_e32 v166, 0xbd800, v165
	v_cndmask_b32_e64 v166, v120, v166, s[6:7]
	s_add_i32 m0, s64, s51
	s_nop 0
	buffer_load_dword v166, s[20:23], 0 offen lds
	v_add_u32_e32 v166, 0xd2800, v165
	v_cndmask_b32_e64 v166, v120, v166, s[6:7]
	s_add_i32 m0, s64, s52
	s_nop 0
	buffer_load_dword v166, s[20:23], 0 offen lds
	v_add_u32_e32 v166, 0xe7800, v165
	v_cndmask_b32_e64 v166, v120, v166, s[6:7]
	s_add_i32 m0, s64, s53
	s_nop 0
	buffer_load_dword v166, s[20:23], 0 offen lds
	v_add_u32_e32 v166, 0xfc800, v165
	v_cndmask_b32_e64 v166, v120, v166, s[6:7]
	s_add_i32 m0, s64, s54
	v_add_u32_e32 v165, 0x111800, v165
	buffer_load_dword v166, s[20:23], 0 offen lds
	v_cndmask_b32_e64 v165, v120, v165, s[6:7]
	s_add_i32 m0, s64, s55
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, v118, v145
	v_add_u32_e32 v166, 64, v165
	v_cmp_gt_i32_e32 vcc, s58, v165
	v_add_u32_e32 v169, s0, v135
	v_add_u32_e32 v170, 0xfffffeff, v169
	v_cndmask_b32_e32 v165, v166, v144, vcc
	v_mul_hi_i32 v165, v165, s41
	v_lshrrev_b32_e32 v166, 31, v165
	v_ashrrev_i32_e32 v165, 4, v165
	v_add_u32_e32 v165, v165, v166
	v_cndmask_b32_e64 v166, 0, -1, vcc
	v_xor_b32_e32 v165, v165, v166
	v_add_u32_e32 v166, v165, v116
	v_mul_lo_u32 v167, v165, s5
	v_add_u32_e32 v165, v142, v165
	v_mul_lo_u32 v166, v166, s14
	v_mul_lo_u32 v165, s14, v165
	v_sub_u32_e32 v166, v166, v167
	v_sub_u32_e32 v165, v165, v167
	v_add3_u32 v166, v119, v166, v143
	v_add3_u32 v165, v119, v165, v143
	buffer_load_dword v168, v166, s[16:19], 0 offen
	s_nop 0
	buffer_load_dword v165, v165, s[16:19], 0 offen
	v_add_u32_e32 v166, v119, v136
	v_add_u32_e32 v167, 0x100, v166
	v_cmp_gt_i32_e32 vcc, s1, v166
	v_add_u32_e32 v180, 0x102, v166
	v_add_u32_e32 v169, 0xfffffefd, v169
	v_cndmask_b32_e32 v170, v167, v170, vcc
	v_mul_hi_i32 v170, v170, s41
	v_lshrrev_b32_e32 v171, 31, v170
	v_ashrrev_i32_e32 v170, 6, v170
	v_add_u32_e32 v170, v170, v171
	v_cndmask_b32_e64 v171, 0, -1, vcc
	v_xor_b32_e32 v177, v170, v171
	v_add_u32_e32 v170, v177, v121
	v_add_u32_e32 v178, v177, v131
	v_mul_i32_i24_e32 v179, 0xfffffeb0, v177
	v_mul_lo_u32 v170, v170, s12
	v_add3_u32 v170, v179, v170, v167
	v_cmp_gt_i32_e32 vcc, s4, v178
	s_nop 1
	v_cndmask_b32_e32 v170, v120, v170, vcc
	v_cmp_gt_i32_e32 vcc, -2, v167
	s_nop 1
	v_cndmask_b32_e32 v171, v180, v169, vcc
	v_cmp_gt_i32_e32 vcc, s61, v166
	v_mul_hi_i32 v171, v171, s41
	v_lshrrev_b32_e32 v172, 31, v171
	v_cndmask_b32_e32 v166, v180, v169, vcc
	v_ashrrev_i32_e32 v171, 6, v171
	v_mul_hi_i32 v166, v166, s41
	v_add_u32_e32 v171, v171, v172
	v_ashrrev_i32_e32 v172, 31, v180
	v_lshrrev_b32_e32 v169, 31, v166
	v_lshrrev_b32_e32 v166, 6, v166
	v_xor_b32_e32 v181, v171, v172
	v_add_u32_e32 v166, v166, v169
	v_cndmask_b32_e64 v169, 0, -1, vcc
	v_add_u32_e32 v171, v181, v121
	v_xor_b32_e32 v166, v166, v169
	v_add_u32_e32 v182, v181, v131
	v_mul_i32_i24_e32 v166, 0xfffffeb0, v166
	v_mul_lo_u32 v169, v171, s12
	v_add3_u32 v169, v166, v169, v180
	v_cmp_gt_i32_e32 vcc, s4, v182
	v_add_u32_e32 v171, v119, v140
	v_add_u32_e32 v172, s0, v141
	v_cndmask_b32_e32 v169, v120, v169, vcc
	v_add_u32_e32 v183, 0x101, v171
	v_add_u32_e32 v173, 0xfffffefe, v172
	v_cmp_gt_i32_e32 vcc, s59, v171
	v_add_u32_e32 v186, 0x103, v171
	v_add_u32_e32 v172, 0xfffffefc, v172
	v_cndmask_b32_e32 v173, v183, v173, vcc
	v_mul_hi_i32 v173, v173, s41
	v_lshrrev_b32_e32 v174, 31, v173
	v_ashrrev_i32_e32 v173, 6, v173
	v_add_u32_e32 v173, v173, v174
	v_ashrrev_i32_e32 v174, 31, v183
	v_xor_b32_e32 v184, v173, v174
	v_add_u32_e32 v173, v184, v122
	v_mul_i32_i24_e32 v185, 0xfffffeb0, v184
	v_mul_lo_u32 v173, v173, s12
	v_add3_u32 v173, v185, v173, v183
	v_cmp_lt_i32_e32 vcc, v184, v132
	s_nop 1
	v_cndmask_b32_e32 v173, v120, v173, vcc
	v_cmp_gt_i32_e32 vcc, s60, v171
	v_add_u32_e32 v171, 0x100, v171
	s_nop 0
	v_cndmask_b32_e32 v174, v186, v172, vcc
	v_mul_hi_i32 v174, v174, s41
	v_lshrrev_b32_e32 v175, 31, v174
	v_ashrrev_i32_e32 v174, 6, v174
	v_add_u32_e32 v174, v174, v175
	v_ashrrev_i32_e32 v175, 31, v186
	v_xor_b32_e32 v174, v174, v175
	v_add_u32_e32 v176, v174, v122
	v_mul_i32_i24_e32 v187, 0xfffffeb0, v174
	v_mul_lo_u32 v176, v176, s12
	v_cmp_lt_i32_e32 vcc, v174, v132
	v_add_u32_e32 v174, v137, v177
	v_add3_u32 v176, v187, v176, v186
	v_mul_lo_u32 v174, s12, v174
	v_cndmask_b32_e32 v188, v120, v176, vcc
	v_add3_u32 v174, v179, v174, v167
	v_cmp_gt_i32_e32 vcc, s8, v178
	s_nop 1
	v_cndmask_b32_e32 v189, v120, v174, vcc
	v_add_u32_e32 v174, v137, v181
	v_mul_lo_u32 v174, s12, v174
	v_add3_u32 v174, v166, v174, v180
	v_cmp_gt_i32_e32 vcc, s8, v182
	s_nop 1
	v_cndmask_b32_e32 v190, v120, v174, vcc
	v_add_u32_e32 v174, v184, v123
	v_mul_lo_u32 v174, v174, s12
	v_add3_u32 v174, v185, v174, v183
	v_cmp_lt_i32_e32 vcc, v184, v133
	s_nop 1
	v_cndmask_b32_e32 v191, v120, v174, vcc
	v_cmp_gt_i32_e32 vcc, -3, v171
	s_nop 1
	v_cndmask_b32_e32 v171, v186, v172, vcc
	v_mul_hi_i32 v171, v171, s41
	v_lshrrev_b32_e32 v172, 31, v171
	v_ashrrev_i32_e32 v171, 6, v171
	v_add_u32_e32 v171, v171, v172
	v_xor_b32_e32 v192, v171, v175
	v_add_u32_e32 v171, v192, v123
	v_mul_lo_u32 v171, v171, s12
	v_add3_u32 v171, v187, v171, v186
	v_cmp_lt_i32_e32 vcc, v192, v133
	s_nop 1
	v_cndmask_b32_e32 v193, v120, v171, vcc
	buffer_load_ubyte v176, v170, s[28:31], 0 offen
	buffer_load_ubyte v175, v169, s[28:31], 0 offen
	buffer_load_ubyte v174, v173, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v173, v188, s[28:31], 0 offen
	buffer_load_ubyte v172, v189, s[28:31], 0 offen
	buffer_load_ubyte v171, v190, s[28:31], 0 offen
	buffer_load_ubyte v170, v191, s[28:31], 0 offen
	buffer_load_ubyte v169, v193, s[28:31], 0 offen
	v_add_u32_e32 v188, v138, v177
	v_mul_lo_u32 v188, s12, v188
	v_add_u32_e32 v189, v138, v181
	v_add3_u32 v188, v179, v188, v167
	v_cmp_gt_i32_e32 vcc, s9, v178
	v_mul_lo_u32 v189, s12, v189
	v_add3_u32 v189, v166, v189, v180
	v_cndmask_b32_e32 v188, v120, v188, vcc
	v_cmp_gt_i32_e32 vcc, s9, v182
	v_add_u32_e32 v190, v184, v126
	v_mul_lo_u32 v190, v190, s12
	v_cndmask_b32_e32 v189, v120, v189, vcc
	v_cmp_lt_i32_e32 vcc, v184, v134
	v_add_u32_e32 v184, v192, v126
	v_add_u32_e32 v177, v139, v177
	v_add3_u32 v183, v185, v190, v183
	v_mul_lo_u32 v184, v184, s12
	v_mul_lo_u32 v177, s12, v177
	v_cndmask_b32_e32 v183, v120, v183, vcc
	v_add3_u32 v184, v187, v184, v186
	v_cmp_lt_i32_e32 vcc, v192, v134
	v_add3_u32 v167, v179, v177, v167
	v_add_u32_e32 v177, v139, v181
	v_cndmask_b32_e32 v184, v120, v184, vcc
	v_cmp_gt_i32_e32 vcc, s15, v178
	v_mul_lo_u32 v177, s12, v177
	v_add3_u32 v166, v166, v177, v180
	v_cndmask_b32_e32 v167, v120, v167, vcc
	v_cmp_gt_i32_e32 vcc, s15, v182
	v_add_u32_e32 v181, s62, v127
	s_nop 0
	v_cndmask_b32_e32 v166, v120, v166, vcc
	buffer_load_ubyte v180, v188, s[28:31], 0 offen
	buffer_load_ubyte v179, v189, s[28:31], 0 offen
	buffer_load_ubyte v178, v183, s[28:31], 0 offen
	buffer_load_ubyte v177, v184, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v167, v167, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v166, v166, s[28:31], 0 offen
	ds_read_b128 v[182:185], v181
	ds_read_b128 v[186:189], v181 offset:2048
	ds_read_b128 v[190:193], v181 offset:4096
	ds_read_b128 v[194:197], v181 offset:6144
	v_add_u32_e32 v181, s63, v129
	ds_read_b128 v[198:201], v181
	ds_read_b128 v[202:205], v181 offset:2048
	ds_read_b128 v[206:209], v181 offset:4096
	ds_read_b128 v[210:213], v181 offset:6144
	ds_read_b128 v[214:217], v181 offset:8192
	ds_read_b128 v[218:221], v181 offset:10240
	ds_read_b128 v[222:225], v181 offset:12288
	s_barrier
	s_setprio 1
	v_and_b32_e32 v160, 0xff, v160
	v_and_b32_e32 v159, 0xff, v159
	v_and_b32_e32 v158, 0xff, v158
	v_and_b32_e32 v157, 0xff, v157
	v_and_b32_e32 v163, 0xff, v163
	v_and_b32_e32 v162, 0xff, v162
	v_and_b32_e32 v161, 0xff, v161
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[182:185], v[198:201], v[40:43], v125, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[182:185], v[202:205], v[108:111], v125, v159 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[182:185], v[206:209], v[104:107], v125, v158 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[182:185], v[210:213], v[100:103], v125, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[182:185], v[214:217], v[96:99], v125, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[182:185], v[218:221], v[92:95], v125, v162 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[182:185], v[222:225], v[88:91], v125, v161 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[186:189], v[198:201], v[84:87], v125, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[186:189], v[202:205], v[80:83], v125, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[186:189], v[206:209], v[76:79], v125, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[186:189], v[210:213], v[72:75], v125, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[186:189], v[214:217], v[68:71], v125, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[186:189], v[218:221], v[64:67], v125, v162 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[186:189], v[222:225], v[60:63], v125, v161 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[190:193], v[198:201], v[56:59], v124, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[190:193], v[202:205], v[52:55], v124, v159 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[190:193], v[206:209], v[48:51], v124, v158 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[190:193], v[210:213], v[44:47], v124, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[190:193], v[214:217], v[36:39], v124, v163 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[190:193], v[218:221], v[20:23], v124, v162 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[190:193], v[222:225], v[8:11], v124, v161 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[194:197], v[198:201], v[4:7], v124, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[194:197], v[202:205], v[12:15], v124, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[194:197], v[206:209], v[16:19], v124, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[194:197], v[210:213], v[24:27], v124, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[194:197], v[214:217], v[28:31], v124, v163 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[194:197], v[218:221], v[32:35], v124, v162 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[194:197], v[222:225], v[0:3], v124, v161 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v157, s62, v128
	ds_read_b128 v[158:161], v157
	ds_read_b128 v[182:185], v157 offset:2048
	ds_read_b128 v[186:189], v157 offset:4096
	ds_read_b128 v[190:193], v157 offset:6144
	v_add_u32_e32 v157, s63, v130
	ds_read_b128 v[194:197], v157
	ds_read_b128 v[198:201], v157 offset:2048
	ds_read_b128 v[202:205], v157 offset:4096
	ds_read_b128 v[206:209], v157 offset:6144
	ds_read_b128 v[210:213], v157 offset:8192
	ds_read_b128 v[214:217], v157 offset:10240
	ds_read_b128 v[218:221], v157 offset:12288
	s_waitcnt vmcnt(16)
	s_barrier
	s_setprio 1
	v_and_b32_e32 v153, 0xff, v153
	v_and_b32_e32 v152, 0xff, v152
	v_and_b32_e32 v151, 0xff, v151
	v_and_b32_e32 v150, 0xff, v150
	v_and_b32_e32 v156, 0xff, v156
	v_and_b32_e32 v155, 0xff, v155
	v_and_b32_e32 v154, 0xff, v154
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[158:161], v[194:197], v[40:43], v125, v153 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[158:161], v[198:201], v[108:111], v125, v152 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[158:161], v[202:205], v[104:107], v125, v151 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[158:161], v[206:209], v[100:103], v125, v150 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[158:161], v[210:213], v[96:99], v125, v156 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[158:161], v[214:217], v[92:95], v125, v155 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[158:161], v[218:221], v[88:91], v125, v154 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[182:185], v[194:197], v[84:87], v125, v153 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[182:185], v[198:201], v[80:83], v125, v152 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[182:185], v[202:205], v[76:79], v125, v151 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[182:185], v[206:209], v[72:75], v125, v150 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[182:185], v[210:213], v[68:71], v125, v156 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[182:185], v[214:217], v[64:67], v125, v155 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[182:185], v[218:221], v[60:63], v125, v154 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[186:189], v[194:197], v[56:59], v124, v153 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[186:189], v[198:201], v[52:55], v124, v152 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[186:189], v[202:205], v[48:51], v124, v151 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[186:189], v[206:209], v[44:47], v124, v150 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[186:189], v[210:213], v[36:39], v124, v156 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[186:189], v[214:217], v[20:23], v124, v155 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[186:189], v[218:221], v[8:11], v124, v154 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[190:193], v[194:197], v[4:7], v124, v153 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[190:193], v[198:201], v[12:15], v124, v152 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[190:193], v[202:205], v[16:19], v124, v151 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[190:193], v[206:209], v[24:27], v124, v150 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[190:193], v[210:213], v[28:31], v124, v156 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[190:193], v[214:217], v[32:35], v124, v155 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[190:193], v[218:221], v[0:3], v124, v154 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_addk_i32 s0, 0xff00
	v_add_u32_e32 v136, 0x100, v136
	v_add_u32_e32 v140, 0x100, v140
	v_add_u32_e32 v143, 0x100, v143
	v_subrev_u32_e32 v144, 64, v144
	v_add_u32_e32 v145, 64, v145
	v_add_u32_e32 v164, 0x800, v164
	v_add_u32_e32 v146, 0x800, v146
	v_add_u32_e32 v147, 0xfffffe00, v147
	v_add_u32_e32 v148, 0x200, v148
	v_add_u32_e32 v149, 0x80, v149
	s_cmpk_lg_i32 s0, 0xd700
	s_mov_b32 s64, s63
	s_mov_b32 s63, s62
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v154, v166
	v_mov_b32_e32 v161, v167
	v_mov_b32_e32 v155, v177
	v_mov_b32_e32 v162, v178
	v_mov_b32_e32 v156, v179
	v_mov_b32_e32 v163, v180
	v_mov_b32_e32 v150, v169
	v_mov_b32_e32 v157, v170
	v_mov_b32_e32 v151, v171
	v_mov_b32_e32 v158, v172
	v_mov_b32_e32 v152, v173
	v_mov_b32_e32 v159, v174
	v_mov_b32_e32 v153, v175
	v_mov_b32_e32 v160, v176
	v_mov_b32_e32 v124, v165
	v_mov_b32_e32 v125, v168
	s_cbranch_scc1 .LBB0_3
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	v_add_u32_e32 v118, s56, v129
	v_add_u32_e32 v119, s56, v130
	s_barrier
	ds_read_b128 v[156:159], v118
	ds_read_b128 v[160:163], v118 offset:2048
	ds_read_b128 v[182:185], v119
	ds_read_b128 v[132:135], v118 offset:12288
	ds_read_b128 v[186:189], v119 offset:2048
	ds_read_b128 v[190:193], v119 offset:4096
	ds_read_b128 v[194:197], v118 offset:4096
	ds_read_b128 v[198:201], v118 offset:6144
	ds_read_b128 v[202:205], v119 offset:6144
	ds_read_b128 v[206:209], v119 offset:8192
	ds_read_b128 v[210:213], v118 offset:8192
	ds_read_b128 v[214:217], v118 offset:10240
	ds_read_b128 v[218:221], v119 offset:10240
	ds_read_b128 v[136:139], v119 offset:12288
	v_add_u32_e32 v118, s13, v127
	ds_read_b128 v[222:225], v118
	ds_read_b128 v[226:229], v118 offset:2048
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[222:225], v[156:159], v[40:43], v168, v176 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v119, s13, v128
	ds_read_b128 v[230:233], v119
	ds_read_b128 v[234:237], v119 offset:2048
	ds_read_b128 v[238:241], v118 offset:4096
	ds_read_b128 v[144:147], v118 offset:6144
	ds_read_b128 v[242:245], v119 offset:4096
	ds_read_b128 v[140:143], v119 offset:6144
	s_movk_i32 s0, 0x7fff
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[230:233], v[182:185], v[40:43], v168, v175 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mov_b32_e32 v120, 0x7fc0
	s_mul_hi_u32 s1, s34, s33
	s_mov_b32 s2, 0x8200
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[222:225], v[160:163], v[108:111], v168, v174 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 s15, 0x27000
	s_nop 2
	v_bfe_u32 v118, v43, 16, 1
	v_bfe_u32 v121, v41, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[222:225], v[194:197], v[104:107], v168, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v126, v40, 16, 1
	v_bfe_u32 v119, v42, 16, 1
	v_cmp_o_f32_e32 vcc, v43, v43
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[230:233], v[186:189], v[108:111], v168, v173 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_mov_b32 s14, 0x7ffffffd
	s_mov_b32 s4, 0x81d0
	s_mov_b32 s6, 0x81c0
	v_add3_u32 v108, v40, v126, s0
	v_add3_u32 v109, v41, v121, s0
	v_add3_u32 v111, v43, v118, s0
	v_add3_u32 v110, v42, v119, s0
	v_lshrrev_b32_e32 v111, 16, v111
	v_mfma_scale_f32_16x16x128_f8f6f4 v[126:129], v[230:233], v[190:193], v[104:107], v168, v171 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v118, 16, v108
	v_lshrrev_b32_e32 v110, 16, v110
	s_mov_b32 s8, 0x81b0
	v_lshrrev_b32_e32 v105, 16, v109
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[222:225], v[198:201], v[100:103], v168, v170 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v104, v120, v111, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	s_nop 0
	v_bfe_u32 v100, v125, 16, 1
	v_cndmask_b32_e32 v101, v120, v110, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[222:225], v[214:217], v[92:95], v168, v178 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v105, v120, v105, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[222:225], v[210:213], v[96:99], v168, v180 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_bfe_u32 v96, v124, 16, 1
	v_add3_u32 v99, v125, v100, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[222:225], v[132:135], v[88:91], v168, v167 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v97, v123, 16, 1
	v_add3_u32 v96, v124, v96, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], v[230:233], v[202:205], v[106:109], v168, v169 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v98, v122, 16, 1
	v_add3_u32 v97, v123, v97, s0
	v_lshrrev_b32_e32 v96, 16, v96
	v_cndmask_b32_e32 v109, v120, v118, vcc
	v_cmp_o_f32_e32 vcc, v125, v125
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[226:229], v[156:159], v[84:87], v168, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v98, v122, v98, s0
	v_cndmask_b32_e32 v100, v120, v99, vcc
	v_cmp_o_f32_e32 vcc, v124, v124
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], v[230:233], v[218:221], v[92:95], v168, v177 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v103, v120, v96, vcc
	v_cmp_o_f32_e32 vcc, v123, v123
	v_lshrrev_b32_e32 v92, 16, v97
	v_lshrrev_b32_e32 v93, 16, v98
	v_cndmask_b32_e32 v106, v120, v92, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[230:233], v[136:139], v[88:91], v168, v166 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v122, v122
	s_nop 1
	v_bfe_u32 v88, v129, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[226:229], v[160:163], v[80:83], v168, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v118, v120, v93, vcc
	v_bfe_u32 v89, v128, 16, 1
	v_bfe_u32 v90, v127, 16, 1
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[234:237], v[182:185], v[84:87], v168, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v129, v129
	v_bfe_u32 v91, v126, 16, 1
	s_nop 0
	v_add3_u32 v87, v129, v88, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[226:229], v[194:197], v[76:79], v168, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v86, v128, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_add3_u32 v85, v127, v90, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[226:229], v[198:201], v[72:75], v168, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v102, v120, v87, vcc
	v_cmp_o_f32_e32 vcc, v128, v128
	v_add3_u32 v84, v126, v91, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[234:237], v[186:189], v[80:83], v168, v173 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v107, v120, v86, vcc
	v_cmp_o_f32_e32 vcc, v127, v127
	s_nop 0
	v_lshrrev_b32_e32 v80, 16, v85
	v_lshrrev_b32_e32 v81, 16, v84
	v_cndmask_b32_e32 v110, v120, v80, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[234:237], v[190:193], v[76:79], v168, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v126, v126
	s_nop 1
	v_bfe_u32 v76, v151, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[226:229], v[210:213], v[68:71], v168, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v121, v120, v81, vcc
	v_bfe_u32 v77, v150, 16, 1
	v_bfe_u32 v78, v149, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[230:233], v[206:209], v[40:43], v168, v179 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v151, v151
	v_bfe_u32 v79, v148, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[234:237], v[202:205], v[72:75], v168, v169 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 2
	v_add3_u32 v75, v151, v76, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[226:229], v[214:217], v[64:67], v168, v178 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v74, v150, v77, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_add3_u32 v73, v149, v78, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[226:229], v[132:135], v[60:63], v168, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v74, 16, v74
	v_cndmask_b32_e32 v108, v120, v75, vcc
	v_cmp_o_f32_e32 vcc, v150, v150
	v_add3_u32 v72, v148, v79, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[234:237], v[206:209], v[68:71], v168, v179 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v119, v120, v74, vcc
	v_cmp_o_f32_e32 vcc, v149, v149
	s_nop 0
	v_lshrrev_b32_e32 v68, 16, v73
	v_lshrrev_b32_e32 v69, 16, v72
	v_cndmask_b32_e32 v122, v120, v68, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[234:237], v[218:221], v[64:67], v168, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v148, v148
	s_nop 1
	v_bfe_u32 v64, v43, 16, 1
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[238:241], v[156:159], v[56:59], v165, v176 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v126, v120, v69, vcc
	v_bfe_u32 v65, v42, 16, 1
	v_bfe_u32 v66, v41, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[234:237], v[136:139], v[60:63], v168, v166 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v43, v43
	v_bfe_u32 v67, v40, 16, 1
	s_nop 0
	v_add3_u32 v63, v43, v64, s0
	v_add3_u32 v62, v42, v65, s0
	v_lshrrev_b32_e32 v63, 16, v63
	v_add3_u32 v61, v41, v66, s0
	v_lshrrev_b32_e32 v62, 16, v62
	v_cndmask_b32_e32 v111, v120, v63, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	v_add3_u32 v60, v40, v67, s0
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[242:245], v[182:185], v[56:59], v165, v175 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v123, v120, v62, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	s_nop 0
	v_lshrrev_b32_e32 v56, 16, v61
	v_cndmask_b32_e32 v125, v120, v56, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[238:241], v[194:197], v[48:51], v165, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v57, 16, v60
	v_cndmask_b32_e32 v128, v120, v57, vcc
	v_cmp_o_f32_e32 vcc, v155, v155
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[238:241], v[160:163], v[52:55], v165, v174 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v48, v154, 16, 1
	v_bfe_u32 v49, v153, 16, 1
	v_bfe_u32 v50, v152, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[242:245], v[190:193], v[40:43], v165, v171 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v50, v152, v50, s0
	v_add3_u32 v49, v153, v49, s0
	v_add3_u32 v48, v154, v48, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[238:241], v[198:201], v[44:47], v165, v170 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[238:241], v[210:213], v[36:39], v165, v180 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 1
	v_lshrrev_b32_e32 v45, 16, v48
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[242:245], v[186:189], v[52:55], v165, v173 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 2
	v_bfe_u32 v52, v155, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[238:241], v[214:217], v[20:23], v165, v178 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v44, v155, v52, s0
	v_lshrrev_b32_e32 v44, 16, v44
	v_cndmask_b32_e32 v124, v120, v44, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[144:147], v[156:159], v[4:7], v165, v176 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v154, v154
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[242:245], v[202:205], v[40:43], v165, v169 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v127, v120, v45, vcc
	v_cmp_o_f32_e32 vcc, v153, v153
	v_lshrrev_b32_e32 v40, 16, v49
	v_lshrrev_b32_e32 v41, 16, v50
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[242:245], v[206:209], v[36:39], v165, v179 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v129, v120, v40, vcc
	v_cmp_o_f32_e32 vcc, v152, v152
	s_nop 0
	v_bfe_u32 v36, v99, 16, 1
	v_bfe_u32 v37, v98, 16, 1
	v_bfe_u32 v38, v97, 16, 1
	v_bfe_u32 v39, v96, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[242:245], v[218:221], v[20:23], v165, v177 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v131, v96, v39, s0
	v_add3_u32 v148, v97, v38, s0
	v_cndmask_b32_e32 v130, v120, v41, vcc
	v_add3_u32 v20, v98, v37, s0
	v_add3_u32 v21, v99, v36, s0
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[140:143], v[182:185], v[4:7], v165, v175 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v149, 16, v21
	v_lshrrev_b32_e32 v150, 16, v20
	v_cmp_o_f32_e32 vcc, v99, v99
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[144:147], v[160:163], v[12:15], v165, v174 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[140:143], v[186:189], v[4:7], v165, v173 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[144:147], v[194:197], v[16:19], v165, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[140:143], v[190:193], v[4:7], v165, v171 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[144:147], v[198:201], v[24:27], v165, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[238:241], v[132:135], v[8:11], v165, v167 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 1
	v_cndmask_b32_e32 v25, v120, v149, vcc
	v_cmp_o_f32_e32 vcc, v98, v98
	v_lshrrev_b32_e32 v24, 16, v148
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[140:143], v[202:205], v[4:7], v165, v169 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v26, 16, v131
	v_bfe_u32 v27, v93, 16, 1
	v_add3_u32 v27, v93, v27, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[144:147], v[210:213], v[28:31], v165, v180 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_cndmask_b32_e32 v29, v120, v150, vcc
	v_cmp_o_f32_e32 vcc, v97, v97
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[242:245], v[136:139], v[8:11], v165, v166 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v28, v92, 16, 1
	v_add3_u32 v28, v92, v28, s0
	v_lshrrev_b32_e32 v30, 16, v27
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[140:143], v[206:209], v[4:7], v165, v179 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v28, 16, v28
	v_bfe_u32 v31, v88, 16, 1
	v_add3_u32 v31, v88, v31, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[144:147], v[214:217], v[32:35], v165, v178 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v31, 16, v31
	s_nop 1
	v_cndmask_b32_e32 v33, v120, v24, vcc
	v_cmp_o_f32_e32 vcc, v96, v96
	v_bfe_u32 v24, v95, 16, 1
	v_add3_u32 v24, v95, v24, s0
	v_cndmask_b32_e32 v96, v120, v26, vcc
	v_bfe_u32 v26, v94, 16, 1
	v_add3_u32 v26, v94, v26, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v95, v95
	v_lshrrev_b32_e32 v26, 16, v26
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[140:143], v[218:221], v[4:7], v165, v177 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v24, v120, v24, vcc
	v_cmp_o_f32_e32 vcc, v94, v94
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[144:147], v[132:135], v[0:3], v165, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v27, v120, v26, vcc
	v_cmp_o_f32_e32 vcc, v93, v93
	v_bfe_u32 v26, v91, 16, 1
	v_add3_u32 v26, v91, v26, s0
	v_cndmask_b32_e32 v34, v120, v30, vcc
	v_cmp_o_f32_e32 vcc, v92, v92
	v_bfe_u32 v30, v89, 16, 1
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v92, v120, v28, vcc
	v_bfe_u32 v28, v90, 16, 1
	v_add3_u32 v28, v90, v28, s0
	v_cmp_o_f32_e32 vcc, v91, v91
	v_add3_u32 v30, v89, v30, s0
	v_lshrrev_b32_e32 v28, 16, v28
	v_cndmask_b32_e32 v26, v120, v26, vcc
	v_cmp_o_f32_e32 vcc, v90, v90
	v_lshrrev_b32_e32 v32, 16, v30
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[140:143], v[136:139], v[0:3], v165, v166 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v30, v120, v28, vcc
	v_cmp_o_f32_e32 vcc, v89, v89
	v_bfe_u32 v28, v87, 16, 1
	v_add3_u32 v28, v87, v28, s0
	v_cndmask_b32_e32 v35, v120, v32, vcc
	v_cmp_o_f32_e32 vcc, v88, v88
	v_bfe_u32 v32, v85, 16, 1
	v_lshrrev_b32_e32 v28, 16, v28
	v_cndmask_b32_e32 v88, v120, v31, vcc
	v_bfe_u32 v31, v86, 16, 1
	v_add3_u32 v31, v86, v31, s0
	v_cmp_o_f32_e32 vcc, v87, v87
	v_bfe_u32 v89, v84, 16, 1
	v_add3_u32 v32, v85, v32, s0
	v_lshrrev_b32_e32 v31, 16, v31
	v_cndmask_b32_e32 v28, v120, v28, vcc
	v_cmp_o_f32_e32 vcc, v86, v86
	v_add3_u32 v89, v84, v89, s0
	v_lshrrev_b32_e32 v90, 16, v32
	v_cndmask_b32_e32 v32, v120, v31, vcc
	v_cmp_o_f32_e32 vcc, v85, v85
	v_bfe_u32 v31, v83, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v85, v120, v90, vcc
	v_cmp_o_f32_e32 vcc, v84, v84
	v_bfe_u32 v86, v82, 16, 1
	v_add3_u32 v31, v83, v31, s0
	v_cndmask_b32_e32 v84, v120, v89, vcc
	v_bfe_u32 v87, v81, 16, 1
	v_add3_u32 v86, v82, v86, s0
	v_lshrrev_b32_e32 v31, 16, v31
	v_cmp_o_f32_e32 vcc, v83, v83
	v_bfe_u32 v89, v80, 16, 1
	v_add3_u32 v87, v81, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v31, v120, v31, vcc
	v_cmp_o_f32_e32 vcc, v82, v82
	v_add3_u32 v89, v80, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v82, v120, v86, vcc
	v_cmp_o_f32_e32 vcc, v81, v81
	v_bfe_u32 v83, v79, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v81, v120, v87, vcc
	v_cmp_o_f32_e32 vcc, v80, v80
	v_bfe_u32 v86, v78, 16, 1
	v_add3_u32 v83, v79, v83, s0
	v_cndmask_b32_e32 v80, v120, v89, vcc
	v_bfe_u32 v87, v77, 16, 1
	v_add3_u32 v86, v78, v86, s0
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v79, v79
	v_bfe_u32 v89, v76, 16, 1
	v_add3_u32 v87, v77, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v79, v120, v83, vcc
	v_cmp_o_f32_e32 vcc, v78, v78
	v_add3_u32 v89, v76, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v78, v120, v86, vcc
	v_cmp_o_f32_e32 vcc, v77, v77
	v_bfe_u32 v83, v75, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v77, v120, v87, vcc
	v_cmp_o_f32_e32 vcc, v76, v76
	v_bfe_u32 v86, v74, 16, 1
	v_add3_u32 v83, v75, v83, s0
	v_cndmask_b32_e32 v76, v120, v89, vcc
	v_bfe_u32 v87, v73, 16, 1
	v_add3_u32 v86, v74, v86, s0
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v75, v75
	v_bfe_u32 v89, v72, 16, 1
	v_add3_u32 v87, v73, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v75, v120, v83, vcc
	v_cmp_o_f32_e32 vcc, v74, v74
	v_add3_u32 v89, v72, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v74, v120, v86, vcc
	v_cmp_o_f32_e32 vcc, v73, v73
	v_bfe_u32 v83, v71, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v73, v120, v87, vcc
	v_cmp_o_f32_e32 vcc, v72, v72
	v_bfe_u32 v86, v70, 16, 1
	v_add3_u32 v83, v71, v83, s0
	v_cndmask_b32_e32 v72, v120, v89, vcc
	v_bfe_u32 v87, v69, 16, 1
	v_add3_u32 v86, v70, v86, s0
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v71, v71
	v_bfe_u32 v89, v68, 16, 1
	v_add3_u32 v87, v69, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v71, v120, v83, vcc
	v_cmp_o_f32_e32 vcc, v70, v70
	v_add3_u32 v89, v68, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v70, v120, v86, vcc
	v_cmp_o_f32_e32 vcc, v69, v69
	v_bfe_u32 v83, v67, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v69, v120, v87, vcc
	v_cmp_o_f32_e32 vcc, v68, v68
	v_bfe_u32 v86, v66, 16, 1
	v_add3_u32 v83, v67, v83, s0
	v_cndmask_b32_e32 v68, v120, v89, vcc
	v_bfe_u32 v87, v65, 16, 1
	v_add3_u32 v86, v66, v86, s0
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v67, v67
	v_bfe_u32 v89, v64, 16, 1
	v_add3_u32 v87, v65, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v67, v120, v83, vcc
	v_cmp_o_f32_e32 vcc, v66, v66
	v_add3_u32 v89, v64, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v66, v120, v86, vcc
	v_cmp_o_f32_e32 vcc, v65, v65
	v_bfe_u32 v83, v63, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v65, v120, v87, vcc
	v_cmp_o_f32_e32 vcc, v64, v64
	v_bfe_u32 v86, v62, 16, 1
	v_add3_u32 v83, v63, v83, s0
	v_cndmask_b32_e32 v64, v120, v89, vcc
	v_bfe_u32 v87, v61, 16, 1
	v_add3_u32 v86, v62, v86, s0
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v63, v63
	v_bfe_u32 v89, v60, 16, 1
	v_add3_u32 v87, v61, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v63, v120, v83, vcc
	v_cmp_o_f32_e32 vcc, v62, v62
	v_add3_u32 v89, v60, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v62, v120, v86, vcc
	v_cmp_o_f32_e32 vcc, v61, v61
	v_bfe_u32 v83, v59, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v61, v120, v87, vcc
	v_cmp_o_f32_e32 vcc, v60, v60
	v_bfe_u32 v86, v58, 16, 1
	v_add3_u32 v83, v59, v83, s0
	v_cndmask_b32_e32 v60, v120, v89, vcc
	v_bfe_u32 v87, v57, 16, 1
	v_add3_u32 v86, v58, v86, s0
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v59, v59
	v_bfe_u32 v89, v56, 16, 1
	v_add3_u32 v87, v57, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v59, v120, v83, vcc
	v_cmp_o_f32_e32 vcc, v58, v58
	v_add3_u32 v89, v56, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v58, v120, v86, vcc
	v_cmp_o_f32_e32 vcc, v57, v57
	v_bfe_u32 v83, v55, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v57, v120, v87, vcc
	v_cmp_o_f32_e32 vcc, v56, v56
	v_bfe_u32 v86, v54, 16, 1
	v_add3_u32 v83, v55, v83, s0
	v_cndmask_b32_e32 v56, v120, v89, vcc
	v_bfe_u32 v87, v53, 16, 1
	v_add3_u32 v86, v54, v86, s0
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v55, v55
	v_bfe_u32 v89, v52, 16, 1
	v_add3_u32 v87, v53, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v55, v120, v83, vcc
	v_cmp_o_f32_e32 vcc, v54, v54
	v_add3_u32 v89, v52, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v54, v120, v86, vcc
	v_cmp_o_f32_e32 vcc, v53, v53
	v_bfe_u32 v83, v51, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v53, v120, v87, vcc
	v_cmp_o_f32_e32 vcc, v52, v52
	v_bfe_u32 v86, v50, 16, 1
	v_add3_u32 v83, v51, v83, s0
	v_cndmask_b32_e32 v52, v120, v89, vcc
	v_bfe_u32 v87, v49, 16, 1
	v_add3_u32 v86, v50, v86, s0
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v51, v51
	v_bfe_u32 v89, v48, 16, 1
	v_add3_u32 v87, v49, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v51, v120, v83, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_add3_u32 v89, v48, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v50, v120, v86, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_bfe_u32 v83, v47, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v49, v120, v87, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_bfe_u32 v86, v46, 16, 1
	v_add3_u32 v83, v47, v83, s0
	v_cndmask_b32_e32 v48, v120, v89, vcc
	v_bfe_u32 v87, v45, 16, 1
	v_add3_u32 v86, v46, v86, s0
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v47, v47
	v_bfe_u32 v89, v44, 16, 1
	v_add3_u32 v87, v45, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v47, v120, v83, vcc
	v_cmp_o_f32_e32 vcc, v46, v46
	v_add3_u32 v89, v44, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v46, v120, v86, vcc
	v_cmp_o_f32_e32 vcc, v45, v45
	v_bfe_u32 v83, v43, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v45, v120, v87, vcc
	v_cmp_o_f32_e32 vcc, v44, v44
	v_bfe_u32 v86, v42, 16, 1
	v_add3_u32 v83, v43, v83, s0
	v_cndmask_b32_e32 v44, v120, v89, vcc
	v_bfe_u32 v87, v41, 16, 1
	v_add3_u32 v86, v42, v86, s0
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v43, v43
	v_bfe_u32 v89, v40, 16, 1
	v_add3_u32 v87, v41, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v43, v120, v83, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	v_add3_u32 v89, v40, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v42, v120, v86, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	v_bfe_u32 v83, v39, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v41, v120, v87, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_bfe_u32 v86, v38, 16, 1
	v_add3_u32 v83, v39, v83, s0
	v_cndmask_b32_e32 v40, v120, v89, vcc
	v_bfe_u32 v87, v37, 16, 1
	v_add3_u32 v86, v38, v86, s0
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v39, v39
	v_bfe_u32 v89, v36, 16, 1
	v_add3_u32 v87, v37, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v39, v120, v83, vcc
	v_cmp_o_f32_e32 vcc, v38, v38
	v_add3_u32 v89, v36, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v38, v120, v86, vcc
	v_cmp_o_f32_e32 vcc, v37, v37
	v_bfe_u32 v83, v23, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v37, v120, v87, vcc
	v_cmp_o_f32_e32 vcc, v36, v36
	v_bfe_u32 v86, v22, 16, 1
	v_add3_u32 v83, v23, v83, s0
	v_cndmask_b32_e32 v36, v120, v89, vcc
	v_bfe_u32 v87, v21, 16, 1
	v_add3_u32 v86, v22, v86, s0
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v23, v23
	v_bfe_u32 v89, v20, 16, 1
	v_add3_u32 v87, v21, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v23, v120, v83, vcc
	v_cmp_o_f32_e32 vcc, v22, v22
	v_add3_u32 v89, v20, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v22, v120, v86, vcc
	v_cmp_o_f32_e32 vcc, v21, v21
	v_bfe_u32 v83, v19, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v21, v120, v87, vcc
	v_cmp_o_f32_e32 vcc, v20, v20
	v_bfe_u32 v86, v18, 16, 1
	v_add3_u32 v83, v19, v83, s0
	v_cndmask_b32_e32 v20, v120, v89, vcc
	v_bfe_u32 v87, v17, 16, 1
	v_add3_u32 v86, v18, v86, s0
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v19, v19
	v_bfe_u32 v89, v16, 16, 1
	v_add3_u32 v87, v17, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v19, v120, v83, vcc
	v_cmp_o_f32_e32 vcc, v18, v18
	v_add3_u32 v89, v16, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v18, v120, v86, vcc
	v_cmp_o_f32_e32 vcc, v17, v17
	v_bfe_u32 v83, v15, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v17, v120, v87, vcc
	v_cmp_o_f32_e32 vcc, v16, v16
	v_bfe_u32 v86, v14, 16, 1
	v_add3_u32 v83, v15, v83, s0
	v_cndmask_b32_e32 v16, v120, v89, vcc
	v_bfe_u32 v87, v13, 16, 1
	v_add3_u32 v86, v14, v86, s0
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v15, v15
	v_bfe_u32 v89, v12, 16, 1
	v_add3_u32 v87, v13, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v15, v120, v83, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_add3_u32 v89, v12, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v14, v120, v86, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_bfe_u32 v83, v11, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v13, v120, v87, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v86, v10, 16, 1
	v_add3_u32 v83, v11, v83, s0
	v_cndmask_b32_e32 v12, v120, v89, vcc
	v_bfe_u32 v87, v9, 16, 1
	v_add3_u32 v86, v10, v86, s0
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v11, v11
	v_bfe_u32 v89, v8, 16, 1
	v_add3_u32 v87, v9, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v11, v120, v83, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_add3_u32 v89, v8, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v10, v120, v86, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_bfe_u32 v83, v7, 16, 1
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v9, v120, v87, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v86, v6, 16, 1
	v_add3_u32 v83, v7, v83, s0
	v_cndmask_b32_e32 v8, v120, v89, vcc
	v_bfe_u32 v87, v5, 16, 1
	v_add3_u32 v86, v6, v86, s0
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v89, v4, 16, 1
	v_add3_u32 v87, v5, v87, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v7, v120, v83, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v89, v4, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v6, v120, v86, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_lshrrev_b32_e32 v89, 16, v89
	v_bfe_u32 v83, v3, 16, 1
	v_cndmask_b32_e32 v5, v120, v87, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v86, v2, 16, 1
	v_bfe_u32 v87, v1, 16, 1
	v_cndmask_b32_e32 v4, v120, v89, vcc
	v_bfe_u32 v89, v0, 16, 1
	v_add3_u32 v89, v0, v89, s0
	v_add3_u32 v87, v1, v87, s0
	v_add3_u32 v86, v2, v86, s0
	v_add3_u32 v83, v3, v83, s0
	s_mul_i32 s0, s35, s33
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s34, s33
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshrrev_b32_e32 v83, 16, v83
	v_cmp_o_f32_e32 vcc, v3, v3
	s_add_u32 s0, s10, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v3, v120, v83, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	s_addc_u32 s1, s11, s1
	s_lshl_b32 s3, s36, 1
	v_lshrrev_b32_e32 v87, 16, v87
	v_cndmask_b32_e32 v2, v120, v86, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	v_lshl_or_b32 v86, v117, 2, v116
	s_add_u32 s12, s0, s3
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v1, v120, v87, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	v_add_u32_e32 v83, s36, v112
	v_mul_lo_u32 v87, s34, v86
	s_addc_u32 s0, s1, 0
	s_and_b32 s1, s34, 0x3fff
	v_cndmask_b32_e32 v0, v120, v89, vcc
	s_lshl_b32 s1, s1, 16
	s_and_b32 s0, s0, 0xffff
	v_add_u32_e32 v89, v87, v112
	v_not_b32_e32 v90, -2.0
	v_cmp_gt_i32_e32 vcc, s2, v83
	s_or_b32 s0, s1, s0
	s_or_b32 s13, s0, 2.0
	v_cndmask_b32_e32 v89, v90, v89, vcc
	v_lshlrev_b32_e32 v89, 1, v89
	buffer_store_short v109, v89, s[12:15], 0 offen
	v_or_b32_e32 v89, 1, v86
	v_mul_lo_u32 v89, s34, v89
	v_add_u32_e32 v91, v89, v112
	v_cndmask_b32_e32 v91, v90, v91, vcc
	v_lshlrev_b32_e32 v91, 1, v91
	buffer_store_short v105, v91, s[12:15], 0 offen
	v_or_b32_e32 v91, 2, v86
	v_mul_lo_u32 v91, s34, v91
	v_add_u32_e32 v93, v91, v112
	v_cndmask_b32_e32 v93, v90, v93, vcc
	v_lshlrev_b32_e32 v93, 1, v93
	buffer_store_short v101, v93, s[12:15], 0 offen
	v_or_b32_e32 v93, 3, v86
	v_mul_lo_u32 v93, s34, v93
	v_add_u32_e32 v94, v93, v112
	v_cndmask_b32_e32 v94, v90, v94, vcc
	v_lshlrev_b32_e32 v94, 1, v94
	s_mov_b32 s0, 0x81f0
	buffer_store_short v104, v94, s[12:15], 0 offen
	v_add_u32_e32 v94, v87, v113
	v_cmp_gt_i32_e64 s[0:1], s0, v83
	s_mov_b32 s2, 0x81e0
	v_cmp_gt_i32_e64 s[2:3], s2, v83
	v_cndmask_b32_e64 v94, v90, v94, s[0:1]
	v_lshlrev_b32_e32 v94, 1, v94
	buffer_store_short v118, v94, s[12:15], 0 offen
	v_add_u32_e32 v94, v89, v113
	v_cndmask_b32_e64 v94, v90, v94, s[0:1]
	v_lshlrev_b32_e32 v94, 1, v94
	buffer_store_short v106, v94, s[12:15], 0 offen
	v_add_u32_e32 v94, v91, v113
	v_cndmask_b32_e64 v94, v90, v94, s[0:1]
	v_lshlrev_b32_e32 v94, 1, v94
	buffer_store_short v103, v94, s[12:15], 0 offen
	v_add_u32_e32 v94, v93, v113
	v_cndmask_b32_e64 v94, v90, v94, s[0:1]
	v_lshlrev_b32_e32 v94, 1, v94
	buffer_store_short v100, v94, s[12:15], 0 offen
	v_add_u32_e32 v94, 32, v112
	v_add_u32_e32 v95, v87, v94
	v_cndmask_b32_e64 v95, v90, v95, s[2:3]
	v_lshlrev_b32_e32 v95, 1, v95
	buffer_store_short v121, v95, s[12:15], 0 offen
	v_add_u32_e32 v95, v89, v94
	v_cndmask_b32_e64 v95, v90, v95, s[2:3]
	v_lshlrev_b32_e32 v95, 1, v95
	buffer_store_short v110, v95, s[12:15], 0 offen
	v_add_u32_e32 v95, v91, v94
	v_cndmask_b32_e64 v95, v90, v95, s[2:3]
	v_lshlrev_b32_e32 v95, 1, v95
	buffer_store_short v107, v95, s[12:15], 0 offen
	v_add_u32_e32 v95, v93, v94
	v_cndmask_b32_e64 v95, v90, v95, s[2:3]
	v_lshlrev_b32_e32 v95, 1, v95
	buffer_store_short v102, v95, s[12:15], 0 offen
	v_add_u32_e32 v95, v87, v114
	v_cmp_gt_i32_e64 s[4:5], s4, v83
	v_cmp_gt_i32_e64 s[6:7], s6, v83
	v_cmp_gt_i32_e64 s[8:9], s8, v83
	v_cndmask_b32_e64 v95, v90, v95, s[4:5]
	v_lshlrev_b32_e32 v95, 1, v95
	buffer_store_short v126, v95, s[12:15], 0 offen
	v_add_u32_e32 v95, v89, v114
	v_cndmask_b32_e64 v95, v90, v95, s[4:5]
	v_lshlrev_b32_e32 v95, 1, v95
	buffer_store_short v122, v95, s[12:15], 0 offen
	v_add_u32_e32 v95, v91, v114
	v_cndmask_b32_e64 v95, v90, v95, s[4:5]
	v_lshlrev_b32_e32 v95, 1, v95
	buffer_store_short v119, v95, s[12:15], 0 offen
	v_add_u32_e32 v95, v93, v114
	v_cndmask_b32_e64 v95, v90, v95, s[4:5]
	v_lshlrev_b32_e32 v95, 1, v95
	buffer_store_short v108, v95, s[12:15], 0 offen
	v_add_u32_e32 v95, 64, v112
	v_add_u32_e32 v97, v87, v95
	v_cndmask_b32_e64 v97, v90, v97, s[6:7]
	v_lshlrev_b32_e32 v97, 1, v97
	buffer_store_short v128, v97, s[12:15], 0 offen
	v_add_u32_e32 v97, v89, v95
	v_cndmask_b32_e64 v97, v90, v97, s[6:7]
	v_lshlrev_b32_e32 v97, 1, v97
	buffer_store_short v125, v97, s[12:15], 0 offen
	v_add_u32_e32 v97, v91, v95
	v_cndmask_b32_e64 v97, v90, v97, s[6:7]
	v_lshlrev_b32_e32 v97, 1, v97
	buffer_store_short v123, v97, s[12:15], 0 offen
	v_add_u32_e32 v97, v93, v95
	v_cndmask_b32_e64 v97, v90, v97, s[6:7]
	v_lshlrev_b32_e32 v97, 1, v97
	buffer_store_short v111, v97, s[12:15], 0 offen
	v_add_u32_e32 v97, v87, v115
	v_cndmask_b32_e64 v97, v90, v97, s[8:9]
	v_lshlrev_b32_e32 v97, 1, v97
	buffer_store_short v130, v97, s[12:15], 0 offen
	v_add_u32_e32 v97, v89, v115
	v_cndmask_b32_e64 v97, v90, v97, s[8:9]
	v_lshlrev_b32_e32 v97, 1, v97
	buffer_store_short v129, v97, s[12:15], 0 offen
	v_add_u32_e32 v97, v91, v115
	v_cndmask_b32_e64 v97, v90, v97, s[8:9]
	v_lshlrev_b32_e32 v97, 1, v97
	buffer_store_short v127, v97, s[12:15], 0 offen
	v_add_u32_e32 v97, v93, v115
	v_cndmask_b32_e64 v97, v90, v97, s[8:9]
	v_lshlrev_b32_e32 v97, 1, v97
	buffer_store_short v124, v97, s[12:15], 0 offen
	s_mov_b32 s10, 0x81a0
	v_add_u32_e32 v97, 0x60, v112
	v_add_u32_e32 v87, v87, v97
	v_cmp_gt_i32_e64 s[10:11], s10, v83
	s_nop 1
	v_cndmask_b32_e64 v83, v90, v87, s[10:11]
	v_lshlrev_b32_e32 v83, 1, v83
	buffer_store_short v96, v83, s[12:15], 0 offen
	v_add_u32_e32 v83, v89, v97
	v_cndmask_b32_e64 v83, v90, v83, s[10:11]
	v_lshlrev_b32_e32 v83, 1, v83
	buffer_store_short v33, v83, s[12:15], 0 offen
	v_add_u32_e32 v33, v91, v97
	v_cndmask_b32_e64 v33, v90, v33, s[10:11]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v29, v33, s[12:15], 0 offen
	v_add_u32_e32 v29, v93, v97
	v_cndmask_b32_e64 v29, v90, v29, s[10:11]
	v_lshlrev_b32_e32 v29, 1, v29
	buffer_store_short v25, v29, s[12:15], 0 offen
	v_or_b32_e32 v25, 16, v86
	v_mul_lo_u32 v25, s34, v25
	v_add_u32_e32 v29, v25, v112
	v_cndmask_b32_e32 v29, v90, v29, vcc
	v_lshlrev_b32_e32 v29, 1, v29
	buffer_store_short v92, v29, s[12:15], 0 offen
	v_or_b32_e32 v29, 17, v86
	v_mul_lo_u32 v29, s34, v29
	v_add_u32_e32 v33, v29, v112
	v_cndmask_b32_e32 v33, v90, v33, vcc
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v34, v33, s[12:15], 0 offen
	v_or_b32_e32 v33, 18, v86
	v_mul_lo_u32 v33, s34, v33
	v_add_u32_e32 v34, v33, v112
	v_cndmask_b32_e32 v34, v90, v34, vcc
	v_lshlrev_b32_e32 v34, 1, v34
	buffer_store_short v27, v34, s[12:15], 0 offen
	v_or_b32_e32 v27, 19, v86
	v_mul_lo_u32 v27, s34, v27
	v_add_u32_e32 v34, v27, v112
	v_cndmask_b32_e32 v34, v90, v34, vcc
	v_lshlrev_b32_e32 v34, 1, v34
	buffer_store_short v24, v34, s[12:15], 0 offen
	v_add_u32_e32 v24, v25, v113
	v_cndmask_b32_e64 v24, v90, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v88, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v29, v113
	v_cndmask_b32_e64 v24, v90, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v35, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v33, v113
	v_cndmask_b32_e64 v24, v90, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v30, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v27, v113
	v_cndmask_b32_e64 v24, v90, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v26, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v25, v94
	v_cndmask_b32_e64 v24, v90, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v84, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v29, v94
	v_cndmask_b32_e64 v24, v90, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v85, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v33, v94
	v_cndmask_b32_e64 v24, v90, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v32, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v27, v94
	v_cndmask_b32_e64 v24, v90, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v28, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v25, v114
	v_cndmask_b32_e64 v24, v90, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v80, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v29, v114
	v_cndmask_b32_e64 v24, v90, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v81, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v33, v114
	v_cndmask_b32_e64 v24, v90, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v82, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v27, v114
	v_cndmask_b32_e64 v24, v90, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v31, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v25, v95
	v_cndmask_b32_e64 v24, v90, v24, s[6:7]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v76, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v29, v95
	v_cndmask_b32_e64 v24, v90, v24, s[6:7]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v77, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v33, v95
	v_cndmask_b32_e64 v24, v90, v24, s[6:7]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v78, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v27, v95
	v_cndmask_b32_e64 v24, v90, v24, s[6:7]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v79, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v25, v115
	v_cndmask_b32_e64 v24, v90, v24, s[8:9]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v72, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v29, v115
	v_cndmask_b32_e64 v24, v90, v24, s[8:9]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v73, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v33, v115
	v_cndmask_b32_e64 v24, v90, v24, s[8:9]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v74, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v27, v115
	v_cndmask_b32_e64 v24, v90, v24, s[8:9]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v75, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v25, v97
	v_cndmask_b32_e64 v24, v90, v24, s[10:11]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v68, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v29, v97
	v_cndmask_b32_e64 v24, v90, v24, s[10:11]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v69, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v33, v97
	v_cndmask_b32_e64 v24, v90, v24, s[10:11]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v70, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v27, v97
	v_cndmask_b32_e64 v24, v90, v24, s[10:11]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v71, v24, s[12:15], 0 offen
	v_or_b32_e32 v24, 32, v86
	v_mul_lo_u32 v24, s34, v24
	v_add_u32_e32 v25, v24, v112
	v_cndmask_b32_e32 v25, v90, v25, vcc
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v64, v25, s[12:15], 0 offen
	v_or_b32_e32 v25, 33, v86
	v_mul_lo_u32 v25, s34, v25
	v_add_u32_e32 v26, v25, v112
	v_cndmask_b32_e32 v26, v90, v26, vcc
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v65, v26, s[12:15], 0 offen
	v_or_b32_e32 v26, 34, v86
	v_mul_lo_u32 v26, s34, v26
	v_add_u32_e32 v27, v26, v112
	v_cndmask_b32_e32 v27, v90, v27, vcc
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v66, v27, s[12:15], 0 offen
	v_or_b32_e32 v27, 35, v86
	v_mul_lo_u32 v27, s34, v27
	v_add_u32_e32 v28, v27, v112
	v_cndmask_b32_e32 v28, v90, v28, vcc
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v67, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v24, v113
	v_cndmask_b32_e64 v28, v90, v28, s[0:1]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v60, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v25, v113
	v_cndmask_b32_e64 v28, v90, v28, s[0:1]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v61, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v26, v113
	v_cndmask_b32_e64 v28, v90, v28, s[0:1]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v62, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v27, v113
	v_cndmask_b32_e64 v28, v90, v28, s[0:1]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v63, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v24, v94
	v_cndmask_b32_e64 v28, v90, v28, s[2:3]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v56, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v25, v94
	v_cndmask_b32_e64 v28, v90, v28, s[2:3]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v57, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v26, v94
	v_cndmask_b32_e64 v28, v90, v28, s[2:3]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v58, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v27, v94
	v_cndmask_b32_e64 v28, v90, v28, s[2:3]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v59, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v24, v114
	v_cndmask_b32_e64 v28, v90, v28, s[4:5]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v52, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v25, v114
	v_cndmask_b32_e64 v28, v90, v28, s[4:5]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v53, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v26, v114
	v_cndmask_b32_e64 v28, v90, v28, s[4:5]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v54, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v27, v114
	v_cndmask_b32_e64 v28, v90, v28, s[4:5]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v55, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v24, v95
	v_cndmask_b32_e64 v28, v90, v28, s[6:7]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v48, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v25, v95
	v_cndmask_b32_e64 v28, v90, v28, s[6:7]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v49, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v26, v95
	v_cndmask_b32_e64 v28, v90, v28, s[6:7]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v50, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v27, v95
	v_cndmask_b32_e64 v28, v90, v28, s[6:7]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v51, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v24, v115
	v_cndmask_b32_e64 v28, v90, v28, s[8:9]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v44, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v25, v115
	v_cndmask_b32_e64 v28, v90, v28, s[8:9]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v45, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v26, v115
	v_cndmask_b32_e64 v28, v90, v28, s[8:9]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v46, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v27, v115
	v_add_u32_e32 v24, v24, v97
	v_cndmask_b32_e64 v28, v90, v28, s[8:9]
	v_cndmask_b32_e64 v24, v90, v24, s[10:11]
	v_lshlrev_b32_e32 v28, 1, v28
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v47, v28, s[12:15], 0 offen
	buffer_store_short v40, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v25, v97
	v_cndmask_b32_e64 v24, v90, v24, s[10:11]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v41, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v26, v97
	v_cndmask_b32_e64 v24, v90, v24, s[10:11]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v42, v24, s[12:15], 0 offen
	v_add_u32_e32 v24, v27, v97
	v_cndmask_b32_e64 v24, v90, v24, s[10:11]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v43, v24, s[12:15], 0 offen
	v_or_b32_e32 v24, 48, v86
	v_mul_lo_u32 v24, s34, v24
	v_add_u32_e32 v25, v24, v112
	v_cndmask_b32_e32 v25, v90, v25, vcc
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v36, v25, s[12:15], 0 offen
	v_or_b32_e32 v25, 49, v86
	v_mul_lo_u32 v25, s34, v25
	v_add_u32_e32 v26, v25, v112
	v_cndmask_b32_e32 v26, v90, v26, vcc
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v37, v26, s[12:15], 0 offen
	v_or_b32_e32 v26, 50, v86
	v_mul_lo_u32 v26, s34, v26
	v_add_u32_e32 v27, v26, v112
	v_cndmask_b32_e32 v27, v90, v27, vcc
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v38, v27, s[12:15], 0 offen
	v_or_b32_e32 v27, 51, v86
	v_mul_lo_u32 v27, s34, v27
	v_add_u32_e32 v28, v27, v112
	v_cndmask_b32_e32 v28, v90, v28, vcc
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v39, v28, s[12:15], 0 offen
	v_add_u32_e32 v28, v24, v113
	v_cndmask_b32_e64 v28, v90, v28, s[0:1]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v20, v28, s[12:15], 0 offen
	v_add_u32_e32 v20, v25, v113
	v_cndmask_b32_e64 v20, v90, v20, s[0:1]
	v_lshlrev_b32_e32 v20, 1, v20
	buffer_store_short v21, v20, s[12:15], 0 offen
	v_add_u32_e32 v20, v26, v113
	v_cndmask_b32_e64 v20, v90, v20, s[0:1]
	v_lshlrev_b32_e32 v20, 1, v20
	buffer_store_short v22, v20, s[12:15], 0 offen
	v_add_u32_e32 v20, v27, v113
	v_cndmask_b32_e64 v20, v90, v20, s[0:1]
	v_lshlrev_b32_e32 v20, 1, v20
	buffer_store_short v23, v20, s[12:15], 0 offen
	v_add_u32_e32 v20, v24, v94
	v_cndmask_b32_e64 v20, v90, v20, s[2:3]
	v_lshlrev_b32_e32 v20, 1, v20
	buffer_store_short v16, v20, s[12:15], 0 offen
	v_add_u32_e32 v16, v25, v94
	v_cndmask_b32_e64 v16, v90, v16, s[2:3]
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v17, v16, s[12:15], 0 offen
	v_add_u32_e32 v16, v26, v94
	v_cndmask_b32_e64 v16, v90, v16, s[2:3]
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v18, v16, s[12:15], 0 offen
	v_add_u32_e32 v16, v27, v94
	v_cndmask_b32_e64 v16, v90, v16, s[2:3]
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v19, v16, s[12:15], 0 offen
	v_add_u32_e32 v16, v24, v114
	v_cndmask_b32_e64 v16, v90, v16, s[4:5]
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v12, v16, s[12:15], 0 offen
	v_add_u32_e32 v12, v25, v114
	v_cndmask_b32_e64 v12, v90, v12, s[4:5]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v13, v12, s[12:15], 0 offen
	v_add_u32_e32 v12, v26, v114
	v_cndmask_b32_e64 v12, v90, v12, s[4:5]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v14, v12, s[12:15], 0 offen
	v_add_u32_e32 v12, v27, v114
	v_cndmask_b32_e64 v12, v90, v12, s[4:5]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v15, v12, s[12:15], 0 offen
	v_add_u32_e32 v12, v24, v95
	v_cndmask_b32_e64 v12, v90, v12, s[6:7]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v8, v12, s[12:15], 0 offen
	v_add_u32_e32 v8, v25, v95
	v_cndmask_b32_e64 v8, v90, v8, s[6:7]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v9, v8, s[12:15], 0 offen
	v_add_u32_e32 v8, v26, v95
	v_cndmask_b32_e64 v8, v90, v8, s[6:7]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v10, v8, s[12:15], 0 offen
	v_add_u32_e32 v8, v27, v95
	v_cndmask_b32_e64 v8, v90, v8, s[6:7]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v11, v8, s[12:15], 0 offen
	v_add_u32_e32 v8, v24, v115
	v_cndmask_b32_e64 v8, v90, v8, s[8:9]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v4, v8, s[12:15], 0 offen
	v_add_u32_e32 v4, v25, v115
	v_cndmask_b32_e64 v4, v90, v4, s[8:9]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v5, v4, s[12:15], 0 offen
	v_add_u32_e32 v4, v26, v115
	v_cndmask_b32_e64 v4, v90, v4, s[8:9]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v6, v4, s[12:15], 0 offen
	v_add_u32_e32 v4, v27, v115
	v_cndmask_b32_e64 v4, v90, v4, s[8:9]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v7, v4, s[12:15], 0 offen
	v_add_u32_e32 v4, v24, v97
	v_cndmask_b32_e64 v4, v90, v4, s[10:11]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v0, v4, s[12:15], 0 offen
	v_add_u32_e32 v0, v25, v97
	v_cndmask_b32_e64 v0, v90, v0, s[10:11]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v1, v0, s[12:15], 0 offen
	v_add_u32_e32 v0, v26, v97
	v_cndmask_b32_e64 v0, v90, v0, s[10:11]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v2, v0, s[12:15], 0 offen
	v_add_u32_e32 v0, v27, v97
	v_cndmask_b32_e64 v0, v90, v0, s[10:11]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v3, v0, s[12:15], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_256x224x256_7168x33280x10752
		.amdhsa_group_segment_fixed_size 122880
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
	.size	wave_mxfp4_static_gemm_256x224x256_7168x33280x10752, .Lfunc_end0-wave_mxfp4_static_gemm_256x224x256_7168x33280x10752

	.set wave_mxfp4_static_gemm_256x224x256_7168x33280x10752.num_vgpr, 246
	.set wave_mxfp4_static_gemm_256x224x256_7168x33280x10752.num_agpr, 0
	.set wave_mxfp4_static_gemm_256x224x256_7168x33280x10752.numbered_sgpr, 65
	.set wave_mxfp4_static_gemm_256x224x256_7168x33280x10752.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_256x224x256_7168x33280x10752.private_seg_size, 0
	.set wave_mxfp4_static_gemm_256x224x256_7168x33280x10752.uses_vcc, 1
	.set wave_mxfp4_static_gemm_256x224x256_7168x33280x10752.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_256x224x256_7168x33280x10752.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_256x224x256_7168x33280x10752.has_recursion, 0
	.set wave_mxfp4_static_gemm_256x224x256_7168x33280x10752.has_indirect_call, 0
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
    .group_segment_fixed_size: 122880
    .kernarg_segment_align: 8
    .kernarg_segment_size: 80
    .max_flat_workgroup_size: 512
    .name:           wave_mxfp4_static_gemm_256x224x256_7168x33280x10752
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     71
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_256x224x256_7168x33280x10752.kd
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
