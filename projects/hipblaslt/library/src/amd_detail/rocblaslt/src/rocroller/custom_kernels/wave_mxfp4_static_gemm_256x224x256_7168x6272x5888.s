; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_256x224x256_7168x6272x5888
	.p2align	8
	.type	wave_mxfp4_static_gemm_256x224x256_7168x6272x5888,@function
wave_mxfp4_static_gemm_256x224x256_7168x6272x5888:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v114, 0x3ff, v0
	v_bfe_u32 v3, v0, 10, 10
	v_lshrrev_b32_e32 v4, 6, v114
	v_lshlrev_b32_e32 v0, 5, v3
	v_lshl_or_b32 v1, v4, 3, v0
	v_bfe_u32 v9, v114, 2, 3
	s_mov_b64 s[24:25], s[2:3]
	v_readfirstlane_b32 s2, v1
	v_lshrrev_b32_e32 v1, 3, v114
	s_lshl_b32 s28, s16, 8
	v_lshrrev_b32_e32 v7, 5, v114
	v_lshrrev_b32_e32 v8, 2, v114
	v_and_b32_e32 v10, 31, v114
	v_lshlrev_b32_e32 v9, 2, v9
	v_or3_b32 v2, v1, v0, s28
	v_xor_b32_e32 v1, v1, v114
	v_bitop3_b32 v13, v8, v7, 7 bitop3:0x6c
	v_sub_u32_e32 v9, v10, v9
	v_lshlrev_b32_e32 v1, 4, v1
	v_lshl_add_u32 v9, v13, 2, v9
	s_mov_b64 s[20:21], s[6:7]
	v_and_b32_e32 v1, 0x70, v1
	v_mul_u32_u24_e32 v2, 0xb80, v2
	s_and_b32 s6, s25, 0xffff
	s_lshl_b32 s30, s2, 7
	v_ashrrev_i32_e32 v10, 31, v9
	s_or_b32 s25, s6, 0x4b800000
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_or_b32_e32 v5, v2, v1
	s_mov_b32 m0, s30
	s_or_b32 s31, s30, 0x2000
	v_xor_b32_e32 v9, v10, v9
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_add_u32_e32 v6, 0x2e000, v5
	s_mov_b32 m0, s31
	s_or_b32 s33, s30, 0x4000
	v_ashrrev_i32_e32 v11, 31, v9
	buffer_load_dwordx4 v6, s[24:27], 0 offen lds
	v_add_u32_e32 v6, 0x5c000, v5
	s_mov_b32 m0, s33
	s_or_b32 s34, s30, 0x6000
	v_lshrrev_b32_e32 v11, 29, v11
	buffer_load_dwordx4 v6, s[24:27], 0 offen lds
	v_add_u32_e32 v5, 0x8a000, v5
	s_mov_b32 m0, s34
	v_add_u32_e32 v9, v9, v11
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_lshlrev_b32_e32 v5, 3, v3
	v_ashrrev_i32_e32 v9, 3, v9
	v_lshrrev_b32_e32 v16, 1, v13
	v_lshl_or_b32 v5, v4, 1, v5
	v_xor_b32_e32 v15, v9, v10
	v_and_b32_e32 v10, 0xfc, v114
	v_lshlrev_b32_e32 v11, 7, v16
	v_readfirstlane_b32 s2, v5
	v_lshlrev_b32_e32 v5, 6, v13
	v_lshlrev_b32_e32 v6, 2, v7
	v_add_u32_e32 v12, v10, v11
	v_add3_u32 v14, v0, v114, v5
	v_lshlrev_b32_e32 v9, 7, v15
	v_sub_u32_e32 v17, v6, v12
	v_add3_u32 v14, v17, v14, v9
	v_ashrrev_i32_e32 v17, 31, v14
	v_xor_b32_e32 v14, v17, v14
	s_mov_b32 s35, 0xb21642c9
	v_mul_hi_i32 v18, v14, s35
	v_add_u32_e32 v14, v18, v14
	v_lshrrev_b32_e32 v18, 31, v14
	v_ashrrev_i32_e32 v14, 9, v14
	v_add_u32_e32 v14, v14, v18
	v_xor_b32_e32 v17, v14, v17
	v_sub_u32_e32 v14, v7, v8
	v_lshlrev_b32_e32 v19, 4, v14
	v_lshlrev_b32_e32 v117, 2, v114
	v_lshlrev_b32_e32 v14, 9, v15
	v_lshlrev_b32_e32 v15, 8, v13
	v_add3_u32 v19, v19, v117, v15
	v_lshlrev_b32_e32 v13, 9, v16
	s_mul_i32 s29, s17, 0xe0
	v_sub_u32_e32 v16, v19, v13
	v_add_u32_e32 v18, s29, v17
	v_add_u32_e32 v19, v16, v14
	v_mul_i32_i24_e32 v17, 0xfffff480, v17
	v_lshlrev_b32_e32 v16, 7, v3
	s_lshl_b32 s54, s2, 7
	s_movk_i32 s3, 0xb80
	v_add3_u32 v17, v19, v16, v17
	s_and_b32 s6, s21, 0xffff
	s_add_i32 s36, s54, 0x10000
	s_or_b32 s55, s54, 0x800
	s_or_b32 s21, s6, 0x4b800000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_mad_i32_i24 v17, v18, s3, v17
	s_mov_b32 m0, s36
	s_add_i32 s37, s55, 0x10000
	s_or_b32 s56, s54, 0x1000
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0xb800, v17
	s_mov_b32 m0, s37
	s_add_i32 s38, s56, 0x10000
	s_or_b32 s57, s54, 0x1800
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x17000, v17
	s_mov_b32 m0, s38
	s_add_i32 s39, s57, 0x10000
	s_or_b32 s58, s54, 0x2000
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x22800, v17
	s_mov_b32 m0, s39
	s_add_i32 s40, s58, 0x10000
	s_or_b32 s59, s54, 0x2800
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x2e000, v17
	s_mov_b32 m0, s40
	s_add_i32 s41, s59, 0x10000
	s_or_b32 s60, s54, 0x3000
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x39800, v17
	s_mov_b32 m0, s41
	s_add_i32 s42, s60, 0x10000
	s_or_b32 s61, s54, 0x3800
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x45000, v17
	s_mov_b32 m0, s42
	s_add_i32 s43, s61, 0x10000
	s_or_b32 s62, s54, 0x4000
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x50800, v17
	s_mov_b32 m0, s43
	s_add_i32 s44, s62, 0x10000
	s_or_b32 s63, s54, 0x4800
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x5c000, v17
	s_mov_b32 m0, s44
	s_add_i32 s45, s63, 0x10000
	s_or_b32 s64, s54, 0x5000
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x67800, v17
	s_mov_b32 m0, s45
	s_add_i32 s46, s64, 0x10000
	s_or_b32 s65, s54, 0x5800
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x73000, v17
	s_mov_b32 m0, s46
	s_add_i32 s47, s65, 0x10000
	s_or_b32 s66, s54, 0x6000
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x7e800, v17
	s_mov_b32 m0, s47
	s_add_i32 s48, s66, 0x10000
	s_or_b32 s67, s54, 0x6800
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v18, 0x8a000, v17
	s_mov_b32 m0, s48
	s_add_i32 s49, s67, 0x10000
	buffer_load_dword v18, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0x95800, v17
	s_mov_b32 m0, s49
	v_bfe_u32 v42, v114, 4, 2
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_lshrrev_b32_e32 v21, 4, v114
	v_lshlrev_b32_e32 v17, 4, v42
	v_mad_i32_i24 v118, v21, -16, v17
	v_add_u32_e32 v19, v118, v114
	v_ashrrev_i32_e32 v20, 31, v19
	v_xor_b32_e32 v19, v20, v19
	v_mul_hi_i32 v22, v19, s35
	v_add_u32_e32 v19, v22, v19
	s_mul_i32 s15, s15, s28
	s_mul_hi_u32 s2, s14, s28
	v_lshrrev_b32_e32 v22, 31, v19
	v_ashrrev_i32_e32 v19, 5, v19
	s_add_i32 s2, s2, s15
	s_mul_i32 s3, s14, s28
	v_add_u32_e32 v19, v19, v22
	v_and_b32_e32 v250, 0xc0, v114
	s_add_u32 s4, s4, s3
	v_xad_u32 v22, v19, v20, v250
	v_and_b32_e32 v19, 62, v114
	s_movk_i32 s51, 0xffc0
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s14, 0x3fff
	v_mov_b32_e32 v20, 0xffffff48
	v_cmp_lt_u32_e32 vcc, 45, v19
	v_lshlrev_b32_e32 v19, 6, v42
	v_mad_i32_i24 v25, v21, s51, v117
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
	buffer_load_dword v119, v22, s[4:7], 0 offen
	buffer_load_dword v115, v23, s[4:7], 0 offen
	v_mul_u32_u24_e32 v22, 0x70, v3
	v_add3_u32 v43, v18, v114, v22
	v_ashrrev_i16_e32 v22, 15, v43
	v_lshrrev_b16_e32 v22, 11, v22
	v_add_u16_e32 v22, v43, v22
	v_and_b32_e32 v22, 0xffffffe0, v22
	v_sub_u16_e32 v22, v43, v22
	v_bfe_i32 v23, v22, 0, 16
	v_ashrrev_i32_e32 v24, 31, v23
	v_add_u16_e32 v26, 32, v22
	v_cmp_gt_i16_e32 vcc, 0, v22
	s_load_dwordx2 s[12:13], s[0:1], 0x40
	s_movk_i32 s15, 0xb8
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
	v_mul_hi_i32 v27, v26, s35
	v_add_u32_e32 v26, v27, v26
	v_lshrrev_b32_e32 v27, 31, v26
	v_ashrrev_i32_e32 v26, 7, v26
	v_add_u32_e32 v26, v26, v27
	v_ashrrev_i32_e32 v27, 31, v43
	v_xor_b32_e32 v29, v27, v43
	v_ashrrev_i32_e32 v30, 31, v29
	v_lshrrev_b32_e32 v30, 27, v30
	v_add_u32_e32 v29, v29, v30
	v_lshrrev_b32_e32 v29, 5, v29
	v_xor_b32_e32 v27, v29, v27
	v_lshlrev_b32_e32 v125, 5, v27
	v_xad_u32 v26, v26, v24, v125
	v_mul_hi_i32 v24, v23, s35
	v_add_u32_e32 v24, v24, v23
	v_lshrrev_b32_e32 v27, 31, v24
	v_ashrrev_i32_e32 v24, 7, v24
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
	v_add_u32_e32 v27, 0xb8, v24
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
	s_movk_i32 s50, 0xff48
	v_add3_u32 v25, v22, v19, v25
	v_cndmask_b32_e32 v23, v24, v27, vcc
	v_mul_hi_i32 v24, v23, s35
	v_add_u32_e32 v23, v24, v23
	v_lshrrev_b32_e32 v24, 31, v23
	v_ashrrev_i32_e32 v23, 7, v23
	v_add_u32_e32 v23, v23, v24
	v_cndmask_b32_e64 v24, 0, -1, vcc
	v_xor_b32_e32 v23, v23, v24
	v_add_u32_e32 v24, v23, v125
	v_mul_lo_u32 v23, v23, s50
	v_mul_lo_u32 v24, s12, v24
	v_add3_u32 v27, v25, v23, v24
	v_add_u32_e32 v24, 16, v43
	v_sub_u32_e32 v25, 0xffef, v43
	v_cmp_gt_i32_e32 vcc, -16, v43
	v_mul_i32_i24_e32 v20, 0xffffffc0, v21
	v_mad_u32_u24 v29, v3, 7, v117
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
	v_mul_u32_u24_e32 v23, 7, v3
	v_mul_i32_i24_e32 v24, -2, v30
	v_cndmask_b32_e32 v31, v31, v32, vcc
	v_mul_hi_i32 v32, v31, s35
	v_add_u32_e32 v31, v32, v31
	v_lshrrev_b32_e32 v32, 31, v31
	v_ashrrev_i32_e32 v31, 7, v31
	v_add_u32_e32 v31, v31, v32
	v_cndmask_b32_e64 v32, 0, -1, vcc
	v_xor_b32_e32 v40, v31, v32
	v_lshlrev_b32_e32 v255, 5, v30
	v_add3_u32 v28, v24, v23, v28
	v_add_u32_e32 v32, v40, v255
	v_mad_u64_u32 v[30:31], s[2:3], v40, s50, v[28:29]
	v_mad_u64_u32 v[32:33], s[2:3], v32, s12, v[30:31]
	v_add_u32_e32 v31, 3, v29
	v_sub_u32_e32 v33, -4, v29
	v_cmp_gt_i32_e32 vcc, -3, v29
	v_sub_u32_e32 v36, 0xffcf, v43
	v_mov_b32_e32 v41, 5
	v_cndmask_b32_e32 v29, v31, v33, vcc
	v_mul_hi_i32 v31, v29, s35
	v_add_u32_e32 v29, v31, v29
	v_lshrrev_b32_e32 v31, 31, v29
	v_ashrrev_i32_e32 v29, 7, v29
	v_add_u32_e32 v29, v29, v31
	v_cndmask_b32_e64 v31, 0, -1, vcc
	v_xor_b32_e32 v31, v29, v31
	v_add_u32_e32 v33, v31, v255
	v_mad_u64_u32 v[28:29], s[2:3], v31, s50, v[28:29]
	v_mad_u64_u32 v[34:35], s[2:3], v33, s12, v[28:29]
	s_movk_i32 s2, 0xffd0
	v_add_u32_e32 v35, 48, v43
	v_cmp_gt_i32_e32 vcc, s2, v43
	s_lshl_b32 s6, s12, 5
	v_add_u32_e32 v29, s6, v26
	v_cndmask_b32_e32 v35, v35, v36, vcc
	v_ashrrev_i16_e32 v36, 15, v35
	v_lshrrev_b16_e32 v36, 11, v36
	v_add_u16_e32 v35, v35, v36
	v_ashrrev_i16_e32 v35, 5, v35
	v_cndmask_b32_e64 v36, 0, -1, vcc
	v_xor_b32_e32 v35, v35, v36
	v_lshlrev_b32_sdwa v252, v41, sext(v35) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_e32 v35, v252, v40
	v_mad_u64_u32 v[36:37], s[2:3], s12, v35, v[30:31]
	v_add_u32_e32 v35, v252, v31
	v_mad_u64_u32 v[38:39], s[2:3], s12, v35, v[28:29]
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	s_movk_i32 s2, 0xffb0
	v_add_u32_e32 v33, s6, v27
	buffer_load_ubyte v155, v26, s[16:19], 0 offen
	buffer_load_ubyte v140, v27, s[16:19], 0 offen offset:2
	buffer_load_ubyte v154, v32, s[16:19], 0 offen offset:1
	buffer_load_ubyte v133, v34, s[16:19], 0 offen offset:3
	buffer_load_ubyte v153, v29, s[16:19], 0 offen
	buffer_load_ubyte v253, v33, s[16:19], 0 offen offset:2
	buffer_load_ubyte v152, v36, s[16:19], 0 offen offset:1
	buffer_load_ubyte v132, v38, s[16:19], 0 offen offset:3
	v_add_u32_e32 v26, 0x50, v43
	v_sub_u32_e32 v27, 0xffaf, v43
	v_cmp_gt_i32_e32 vcc, s2, v43
	v_add_u32_e32 v32, s6, v29
	v_add_u32_e32 v33, s6, v33
	v_cndmask_b32_e32 v26, v26, v27, vcc
	v_ashrrev_i16_e32 v27, 15, v26
	v_lshrrev_b16_e32 v27, 11, v27
	v_add_u16_e32 v26, v26, v27
	v_ashrrev_i16_e32 v26, 5, v26
	v_cndmask_b32_e64 v27, 0, -1, vcc
	v_xor_b32_e32 v26, v26, v27
	v_lshlrev_b32_sdwa v113, v41, sext(v26) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_e32 v26, v113, v40
	v_mad_u64_u32 v[26:27], s[2:3], s12, v26, v[30:31]
	v_add_u32_e32 v27, v113, v31
	v_mad_u64_u32 v[28:29], s[2:3], s12, v27, v[28:29]
	v_add_u32_e32 v27, s6, v32
	v_add_u32_e32 v29, s6, v33
	buffer_load_ubyte v161, v32, s[16:19], 0 offen
	buffer_load_ubyte v254, v33, s[16:19], 0 offen offset:2
	buffer_load_ubyte v160, v26, s[16:19], 0 offen offset:1
	buffer_load_ubyte v147, v28, s[16:19], 0 offen offset:3
	buffer_load_ubyte v127, v27, s[16:19], 0 offen
	buffer_load_ubyte v112, v29, s[16:19], 0 offen offset:2
	v_cmp_eq_u32_e64 s[2:3], 0, v3
	s_movk_i32 s6, 0x3800
	s_mov_b32 s13, -2
	s_and_b64 vcc, exec, s[2:3]
	scratch_store_dword off, v43, off offset:68
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	v_lshlrev_b32_e32 v28, 7, v114
	v_lshlrev_b32_e32 v21, 11, v21
	v_and_b32_e32 v26, 7, v114
	v_sub_u32_e32 v21, v28, v21
	v_mul_lo_u32 v3, v3, s6
	v_bitop3_b32 v27, v42, v114, 7 bitop3:0x78
	v_lshl_add_u32 v4, v4, 13, v21
	v_add_u32_e32 v3, v21, v3
	v_bitop3_b32 v21, v42, v26, 4 bitop3:0x36
	v_lshlrev_b32_e32 v27, 4, v27
	v_lshlrev_b32_e32 v21, 4, v21
	v_or_b32_e32 v146, v4, v27
	v_or_b32_e32 v27, v3, v27
	v_or_b32_e32 v126, v21, v4
	v_or_b32_e32 v21, v21, v3
	v_add3_u32 v3, v14, v15, v16
	v_lshl_add_u32 v3, v7, 4, v3
	v_lshlrev_b32_e32 v4, 4, v8
	v_sub_u32_e32 v3, v3, v4
	v_sub_u32_e32 v241, v3, v13
	v_sub_u32_e32 v3, 0, v114
	scratch_store_dword off, v3, off offset:28
	v_sub_u32_e32 v3, v12, v6
	v_sub_u32_e32 v3, v3, v0
	v_sub_u32_e32 v3, v3, v5
	v_sub_u32_e32 v120, v3, v9
	v_add_u32_e32 v3, v9, v5
	v_add3_u32 v0, v3, v0, v6
	v_sub_u32_e32 v0, v0, v10
	v_sub_u32_e32 v8, v0, v11
	v_add3_u32 v0, v20, v22, v19
	v_sub_u32_e32 v9, 0, v0
	v_add_u32_e32 v0, 32, v125
	scratch_store_dword off, v0, off offset:32
	v_add_u32_e32 v0, 64, v125
	scratch_store_dword off, v0, off offset:36
	v_add_u32_e32 v0, 0x60, v125
	scratch_store_dword off, v0, off offset:40
	scratch_load_dword v139, off, off offset:40
	v_add_u32_e32 v0, v24, v20
	v_add3_u32 v0, v0, v23, v19
	v_sub_u32_e32 v121, 0, v0
	v_add_u32_e32 v0, 32, v250
	scratch_store_dword off, v0, off offset:44
	v_add_u32_e32 v0, v18, v17
	s_load_dwordx2 s[8:9], s[0:1], 0x48
	v_sub_u32_e32 v122, 0, v0
	v_add_u32_e32 v0, 0x10000, v27
	scratch_store_dword off, v0, off offset:48
	v_add_u32_e32 v0, 0x10000, v21
	v_add_u32_e32 v141, v20, v19
	scratch_store_dword off, v0, off offset:52
	scratch_store_dword off, v27, off offset:72
	v_add_u32_e32 v0, 0x17000, v27
	s_mov_b32 s0, 0x8a100
	v_add_u32_e32 v10, v141, v22
	v_add3_u32 v11, v25, v19, v23
	v_mov_b32_e32 v4, 0
	scratch_store_dword off, v0, off offset:56
	scratch_store_dword off, v21, off offset:76
	v_add_u32_e32 v0, 0x17000, v21
	scratch_store_dword off, v42, off offset:64
	v_add3_u32 v138, v2, v1, s0
	s_add_i32 s0, s30, 0x8000
	s_add_i32 s1, s31, 0x8000
	s_add_i32 s52, s33, 0x8000
	s_add_i32 s53, s34, 0x8000
	s_add_i32 s54, s54, 0x17000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	s_add_i32 s55, s55, 0x17000
	s_add_i32 s56, s56, 0x17000
	s_add_i32 s57, s57, 0x17000
	s_add_i32 s58, s58, 0x17000
	s_add_i32 s59, s59, 0x17000
	s_add_i32 s60, s60, 0x17000
	s_add_i32 s61, s61, 0x17000
	s_add_i32 s62, s62, 0x17000
	s_add_i32 s63, s63, 0x17000
	s_add_i32 s64, s64, 0x17000
	s_add_i32 s65, s65, 0x17000
	s_add_i32 s66, s66, 0x17000
	s_add_i32 s67, s67, 0x17000
	s_movk_i32 s68, 0xb217
	s_movk_i32 s69, 0xffee
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	s_movk_i32 s70, 0xff00
	s_movk_i32 s71, 0xfeff
	s_movk_i32 s72, 0xffb8
	s_movk_i32 s73, 0xffb7
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	s_movk_i32 s74, 0xfefd
	s_movk_i32 s75, 0xfefe
	scratch_store_dword off, v0, off offset:60
	v_mov_b32_e32 v5, v4
	v_mov_b32_e32 v6, v4
	v_mov_b32_e32 v7, v4
	v_mov_b32_e32 v108, v4
	v_mov_b32_e32 v109, v4
	v_mov_b32_e32 v110, v4
	v_mov_b32_e32 v111, v4
	v_mov_b32_e32 v104, v4
	v_mov_b32_e32 v105, v4
	v_mov_b32_e32 v106, v4
	v_mov_b32_e32 v107, v4
	v_mov_b32_e32 v100, v4
	v_mov_b32_e32 v101, v4
	v_mov_b32_e32 v102, v4
	v_mov_b32_e32 v103, v4
	v_mov_b32_e32 v96, v4
	v_mov_b32_e32 v97, v4
	v_mov_b32_e32 v98, v4
	v_mov_b32_e32 v99, v4
	v_mov_b32_e32 v92, v4
	v_mov_b32_e32 v93, v4
	v_mov_b32_e32 v94, v4
	v_mov_b32_e32 v95, v4
	v_mov_b32_e32 v88, v4
	v_mov_b32_e32 v89, v4
	v_mov_b32_e32 v90, v4
	v_mov_b32_e32 v91, v4
	v_mov_b32_e32 v84, v4
	v_mov_b32_e32 v85, v4
	v_mov_b32_e32 v86, v4
	v_mov_b32_e32 v87, v4
	v_mov_b32_e32 v80, v4
	v_mov_b32_e32 v81, v4
	v_mov_b32_e32 v82, v4
	v_mov_b32_e32 v83, v4
	v_mov_b32_e32 v76, v4
	v_mov_b32_e32 v77, v4
	v_mov_b32_e32 v78, v4
	v_mov_b32_e32 v79, v4
	v_mov_b32_e32 v72, v4
	v_mov_b32_e32 v73, v4
	v_mov_b32_e32 v74, v4
	v_mov_b32_e32 v75, v4
	v_mov_b32_e32 v68, v4
	v_mov_b32_e32 v69, v4
	v_mov_b32_e32 v70, v4
	v_mov_b32_e32 v71, v4
	v_mov_b32_e32 v64, v4
	v_mov_b32_e32 v65, v4
	v_mov_b32_e32 v66, v4
	v_mov_b32_e32 v67, v4
	v_mov_b32_e32 v60, v4
	v_mov_b32_e32 v61, v4
	v_mov_b32_e32 v62, v4
	v_mov_b32_e32 v63, v4
	v_mov_b32_e32 v56, v4
	v_mov_b32_e32 v57, v4
	v_mov_b32_e32 v58, v4
	v_mov_b32_e32 v59, v4
	v_mov_b32_e32 v52, v4
	v_mov_b32_e32 v53, v4
	v_mov_b32_e32 v54, v4
	v_mov_b32_e32 v55, v4
	v_mov_b32_e32 v48, v4
	v_mov_b32_e32 v49, v4
	v_mov_b32_e32 v50, v4
	v_mov_b32_e32 v51, v4
	v_mov_b32_e32 v44, v4
	v_mov_b32_e32 v45, v4
	v_mov_b32_e32 v46, v4
	v_mov_b32_e32 v47, v4
	v_mov_b32_e32 v40, v4
	v_mov_b32_e32 v41, v4
	v_mov_b32_e32 v42, v4
	v_mov_b32_e32 v43, v4
	v_mov_b32_e32 v28, v4
	v_mov_b32_e32 v29, v4
	v_mov_b32_e32 v30, v4
	v_mov_b32_e32 v31, v4
	v_mov_b32_e32 v16, v4
	v_mov_b32_e32 v17, v4
	v_mov_b32_e32 v18, v4
	v_mov_b32_e32 v19, v4
	v_mov_b32_e32 v142, v4
	v_mov_b32_e32 v143, v4
	v_mov_b32_e32 v144, v4
	v_mov_b32_e32 v145, v4
	v_mov_b32_e32 v12, v4
	v_mov_b32_e32 v13, v4
	v_mov_b32_e32 v14, v4
	v_mov_b32_e32 v15, v4
	v_mov_b32_e32 v148, v4
	v_mov_b32_e32 v149, v4
	v_mov_b32_e32 v150, v4
	v_mov_b32_e32 v151, v4
	v_mov_b32_e32 v24, v4
	v_mov_b32_e32 v25, v4
	v_mov_b32_e32 v26, v4
	v_mov_b32_e32 v27, v4
	v_mov_b32_e32 v32, v4
	v_mov_b32_e32 v33, v4
	v_mov_b32_e32 v34, v4
	v_mov_b32_e32 v35, v4
	v_mov_b32_e32 v36, v4
	v_mov_b32_e32 v37, v4
	v_mov_b32_e32 v38, v4
	v_mov_b32_e32 v39, v4
	v_mov_b32_e32 v0, v4
	v_mov_b32_e32 v1, v4
	v_mov_b32_e32 v2, v4
	v_mov_b32_e32 v3, v4
	v_mov_b32_e32 v162, v11
	v_mov_b32_e32 v163, v10
.LBB0_3:
	scratch_store_dword off, v122, off offset:24
	scratch_store_dword off, v11, off offset:20
	scratch_store_dword off, v121, off offset:16
	scratch_store_dword off, v10, off offset:12
	scratch_store_dword off, v9, off offset:8
	scratch_store_dword off, v8, off offset:4
	scratch_store_dword off, v120, off
	s_waitcnt vmcnt(0)
	s_barrier
	v_add_u32_e32 v171, v114, v8
	scratch_load_dword v8, off, off offset:28
	s_mov_b32 m0, s0
	v_add_u32_e32 v164, 0xfff75f80, v138
	buffer_load_dwordx4 v164, s[24:27], 0 offen lds
	v_add_u32_e32 v164, 0xfffa3f80, v138
	s_mov_b32 m0, s1
	v_add_u32_e32 v221, v241, v117
	buffer_load_dwordx4 v164, s[24:27], 0 offen lds
	v_add_u32_e32 v164, 0xfffd1f80, v138
	s_mov_b32 m0, s52
	s_waitcnt vmcnt(2)
	v_add_u32_e32 v220, v8, v120
	buffer_load_dwordx4 v164, s[24:27], 0 offen lds
	v_add_u32_e32 v164, 0xffffff80, v138
	s_mov_b32 m0, s53
	v_add_u32_e32 v165, 0xfffffdff, v220
	buffer_load_dwordx4 v164, s[24:27], 0 offen lds
	v_add_u32_e32 v164, 0x200, v171
	v_cmp_gt_i32_e32 vcc, 0, v164
	s_mov_b32 m0, s54
	s_nop 0
	v_cndmask_b32_e32 v165, v164, v165, vcc
	v_mul_hi_i32 v166, v165, s35
	v_add_u32_e32 v165, v166, v165
	v_lshrrev_b32_e32 v166, 31, v165
	v_ashrrev_i32_e32 v165, 9, v165
	v_add_u32_e32 v165, v165, v166
	v_ashrrev_i32_e32 v164, 31, v164
	v_xor_b32_e32 v164, v165, v164
	v_add_u32_e32 v165, s29, v164
	v_mul_i32_i24_e32 v164, 0xfffff480, v164
	v_mul_i32_i24_e32 v165, 0xb80, v165
	v_add3_u32 v164, v164, v165, v221
	v_add_u32_e32 v165, 0x800, v164
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0xc000, v164
	s_mov_b32 m0, s55
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x17800, v164
	s_mov_b32 m0, s56
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x23000, v164
	s_mov_b32 m0, s57
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x2e800, v164
	s_mov_b32 m0, s58
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x3a000, v164
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x45800, v164
	s_mov_b32 m0, s60
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x51000, v164
	s_mov_b32 m0, s61
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x5c800, v164
	s_mov_b32 m0, s62
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x68000, v164
	s_mov_b32 m0, s63
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x73800, v164
	s_mov_b32 m0, s64
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x7f000, v164
	s_mov_b32 m0, s65
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x8a800, v164
	s_mov_b32 m0, s66
	v_add_u32_e32 v164, 0x96000, v164
	buffer_load_dword v165, s[20:23], 0 offen lds
	s_mov_b32 m0, s67
	s_nop 0
	buffer_load_dword v164, s[20:23], 0 offen lds
	scratch_load_dword v124, off, off offset:44
	v_mov_b32_e32 v20, v147
	scratch_load_dword v147, off, off offset:32
	v_add_u32_e32 v225, v117, v10
	v_mov_b32_e32 v21, v254
	scratch_load_dword v10, off, off offset:48
	scratch_load_dword v254, off, off offset:36
	v_add_u32_e32 v222, v114, v118
	v_add_u32_e32 v224, v8, v122
	v_add_u32_e32 v223, 64, v222
	v_add_u32_e32 v164, 0xffbf, v224
	v_cmp_gt_i32_e32 vcc, s51, v222
	v_add_u32_e32 v166, 18, v222
	v_add_u32_e32 v167, 0xffed, v224
	v_cndmask_b32_e32 v164, v223, v164, vcc
	v_mul_i32_i24_sdwa v165, sext(v164), s68 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_add_u16_sdwa v164, v165, v164 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_lshrrev_b16_e32 v165, 15, v164
	v_ashrrev_i16_e32 v164, 5, v164
	v_add_u16_e32 v164, v164, v165
	v_cndmask_b32_e64 v165, 0, -1, vcc
	v_cmp_gt_i32_e32 vcc, s69, v222
	v_xor_b32_e32 v164, v164, v165
	v_add_u32_sdwa v165, v250, sext(v164) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_cndmask_b32_e32 v166, v166, v167, vcc
	v_mul_i32_i24_sdwa v167, sext(v166), s68 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_add_u16_sdwa v166, v167, v166 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_lshrrev_b16_e32 v167, 15, v166
	v_ashrrev_i16_e32 v166, 5, v166
	v_add_u16_e32 v166, v166, v167
	v_cndmask_b32_e64 v167, 0, -1, vcc
	v_xor_b32_e32 v166, v166, v167
	v_sub_u32_e32 v8, 0, v117
	v_bfe_i32 v166, v166, 0, 16
	v_mul_lo_u32 v165, v165, s14
	v_add_u32_e32 v227, v8, v9
	v_mad_i32_i24 v165, v166, s50, v165
	v_add_u32_e32 v226, 0x100, v225
	v_cmp_gt_i32_e32 vcc, s70, v225
	v_add_u32_e32 v168, 0xffffffb7, v227
	v_add_u32_e32 v169, 0xfffffefd, v227
	v_add_u32_e32 v228, v117, v11
	v_add_u32_e32 v229, v8, v121
	v_add_u32_e32 v177, 0xffffffb6, v229
	v_add_u32_e32 v230, 0x100, v228
	v_add3_u32 v165, v117, v165, v141
	v_mov_b32_e32 v9, v140
	v_mov_b32_e32 v140, v125
	v_mov_b32_e32 v116, v113
	v_mov_b32_e32 v8, v253
	s_waitcnt vmcnt(1)
	ds_read_b128 v[206:209], v10
	ds_read_b128 v[210:213], v10 offset:2048
	ds_read_b128 v[242:245], v10 offset:4096
	ds_read_b128 v[246:249], v10 offset:6144
	v_add_u32_sdwa v164, v124, sext(v164) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_mul_lo_u32 v164, s14, v164
	v_mad_i32_i24 v164, v166, s50, v164
	v_add_u32_e32 v166, 0xfffffeff, v227
	v_cndmask_b32_e32 v166, v226, v166, vcc
	v_mul_hi_i32 v167, v166, s35
	v_add_u32_e32 v166, v167, v166
	v_lshrrev_b32_e32 v167, 31, v166
	v_ashrrev_i32_e32 v166, 7, v166
	v_add_u32_e32 v166, v166, v167
	v_cndmask_b32_e64 v167, 0, -1, vcc
	v_xor_b32_e32 v172, v166, v167
	v_add_u32_e32 v167, 0x48, v225
	v_cmp_gt_i32_e32 vcc, s72, v225
	v_add_u32_e32 v166, v172, v125
	v_mul_lo_u32 v166, v166, s12
	v_cndmask_b32_e32 v167, v167, v168, vcc
	v_mul_hi_i32 v168, v167, s35
	v_add_u32_e32 v167, v168, v167
	v_lshrrev_b32_e32 v168, 31, v167
	v_ashrrev_i32_e32 v167, 7, v167
	v_add_u32_e32 v167, v167, v168
	v_cndmask_b32_e64 v168, 0, -1, vcc
	v_xor_b32_e32 v167, v167, v168
	v_add_u32_e32 v168, 0x102, v225
	v_cmp_gt_i32_e32 vcc, -2, v226
	v_mul_lo_u32 v173, v167, s15
	v_add_u32_e32 v182, v147, v172
	v_cndmask_b32_e32 v170, v168, v169, vcc
	v_mul_hi_i32 v174, v170, s35
	v_add_u32_e32 v170, v174, v170
	v_lshrrev_b32_e32 v174, 31, v170
	v_ashrrev_i32_e32 v170, 7, v170
	v_cmp_gt_i32_e32 vcc, s75, v225
	v_add_u32_e32 v170, v170, v174
	v_ashrrev_i32_e32 v174, 31, v168
	v_cndmask_b32_e32 v168, v168, v169, vcc
	v_mul_hi_i32 v169, v168, s35
	v_add_u32_e32 v168, v169, v168
	v_lshrrev_b32_e32 v169, 31, v168
	v_ashrrev_i32_e32 v168, 7, v168
	v_xor_b32_e32 v174, v170, v174
	v_add_u32_e32 v168, v168, v169
	v_cndmask_b32_e64 v169, 0, -1, vcc
	v_add_u32_e32 v170, v174, v125
	v_xor_b32_e32 v168, v168, v169
	v_mul_lo_u32 v169, v170, s12
	v_mul_lo_u32 v175, v168, s15
	v_sub_u32_e32 v168, v169, v175
	v_add_u32_e32 v169, 0x101, v228
	v_add_u32_e32 v170, 0xfffffefe, v229
	v_cmp_gt_i32_e32 vcc, s71, v228
	v_add_u32_e32 v183, v147, v174
	v_sub_u32_e32 v167, v166, v173
	v_cndmask_b32_e32 v170, v169, v170, vcc
	v_mul_hi_i32 v176, v170, s35
	v_add_u32_e32 v170, v176, v170
	v_lshrrev_b32_e32 v176, 31, v170
	v_ashrrev_i32_e32 v170, 7, v170
	v_add_u32_e32 v170, v170, v176
	v_ashrrev_i32_e32 v169, 31, v169
	v_xor_b32_e32 v176, v170, v169
	v_add_u32_e32 v170, 0x49, v228
	v_cmp_gt_i32_e32 vcc, s73, v228
	v_add_u32_e32 v169, v176, v255
	v_mul_lo_u32 v169, v169, s12
	v_cndmask_b32_e32 v177, v170, v177, vcc
	v_mul_hi_i32 v178, v177, s35
	v_add_u32_e32 v177, v178, v177
	v_lshrrev_b32_e32 v178, 31, v177
	v_ashrrev_i32_e32 v177, 7, v177
	v_add_u32_e32 v177, v177, v178
	v_ashrrev_i32_e32 v170, 31, v170
	v_xor_b32_e32 v170, v177, v170
	v_mul_lo_u32 v177, v170, s15
	v_add_u32_e32 v170, 0x103, v228
	v_add_u32_e32 v178, 0xfffffefc, v229
	v_cmp_gt_i32_e32 vcc, s74, v228
	v_add_u32_e32 v184, v176, v252
	v_add_u32_e32 v166, v163, v117
	v_cndmask_b32_e32 v179, v170, v178, vcc
	v_mul_hi_i32 v180, v179, s35
	v_add_u32_e32 v179, v180, v179
	v_lshrrev_b32_e32 v180, 31, v179
	v_ashrrev_i32_e32 v179, 7, v179
	v_cmp_gt_i32_e32 vcc, -3, v230
	v_add_u32_e32 v179, v179, v180
	v_ashrrev_i32_e32 v180, 31, v170
	v_cndmask_b32_e32 v170, v170, v178, vcc
	v_mul_hi_i32 v178, v170, s35
	v_add_u32_e32 v170, v178, v170
	v_lshrrev_b32_e32 v178, 31, v170
	v_ashrrev_i32_e32 v170, 7, v170
	v_add_u32_e32 v170, v170, v178
	v_xor_b32_e32 v179, v179, v180
	v_xor_b32_e32 v178, v170, v180
	v_add_u32_e32 v181, v179, v255
	v_add_u32_e32 v170, v178, v252
	v_mul_lo_u32 v181, v181, s12
	v_mul_lo_u32 v179, v179, s15
	v_mul_lo_u32 v170, v170, s12
	v_sub_u32_e32 v169, v169, v177
	v_sub_u32_e32 v181, v181, v179
	v_mul_lo_u32 v182, s12, v182
	v_mul_lo_u32 v183, s12, v183
	v_mul_lo_u32 v184, v184, s12
	v_sub_u32_e32 v170, v170, v179
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v180, v254, v172
	v_add_u32_e32 v176, v176, v113
	v_add_u32_e32 v172, v139, v172
	v_add_u32_e32 v167, v166, v167
	v_add3_u32 v168, v117, v168, v163
	v_add3_u32 v169, v117, v169, v162
	v_add3_u32 v181, v117, v181, v162
	v_sub_u32_e32 v182, v182, v173
	v_sub_u32_e32 v183, v183, v175
	v_sub_u32_e32 v184, v184, v177
	v_add3_u32 v170, v117, v170, v162
	v_mul_lo_u32 v180, s12, v180
	v_mul_lo_u32 v176, v176, s12
	v_mul_lo_u32 v172, s12, v172
	v_add_u32_e32 v182, v166, v182
	v_add3_u32 v183, v117, v183, v163
	v_add3_u32 v184, v117, v184, v162
	buffer_load_ubyte v231, v167, s[16:19], 0 offen offset:72
	s_nop 0
	buffer_load_ubyte v167, v168, s[16:19], 0 offen offset:258
	buffer_load_ubyte v232, v169, s[16:19], 0 offen offset:73
	s_nop 0
	buffer_load_ubyte v168, v181, s[16:19], 0 offen offset:259
	buffer_load_ubyte v233, v182, s[16:19], 0 offen offset:72
	buffer_load_ubyte v169, v183, s[16:19], 0 offen offset:258
	buffer_load_ubyte v234, v184, s[16:19], 0 offen offset:73
	s_nop 0
	buffer_load_ubyte v170, v170, s[16:19], 0 offen offset:259
	v_sub_u32_e32 v180, v180, v173
	v_add_u32_e32 v181, v254, v174
	v_sub_u32_e32 v176, v176, v177
	v_add_u32_e32 v177, v178, v113
	v_sub_u32_e32 v172, v172, v173
	v_add_u32_e32 v173, v139, v174
	v_mul_lo_u32 v181, s12, v181
	v_mul_lo_u32 v177, v177, s12
	v_mul_lo_u32 v173, s12, v173
	v_add_u32_e32 v180, v166, v180
	v_sub_u32_e32 v181, v181, v175
	v_sub_u32_e32 v177, v177, v179
	v_sub_u32_e32 v173, v173, v175
	v_add3_u32 v181, v117, v181, v163
	v_add3_u32 v176, v117, v176, v162
	v_add3_u32 v177, v117, v177, v162
	v_add_u32_e32 v172, v166, v172
	v_add3_u32 v173, v117, v173, v163
	buffer_load_ubyte v235, v180, s[16:19], 0 offen offset:72
	buffer_load_ubyte v236, v181, s[16:19], 0 offen offset:258
	buffer_load_ubyte v237, v176, s[16:19], 0 offen offset:73
	buffer_load_ubyte v238, v177, s[16:19], 0 offen offset:259
	buffer_load_ubyte v239, v172, s[16:19], 0 offen offset:72
	buffer_load_ubyte v240, v173, s[16:19], 0 offen offset:258
	v_add3_u32 v164, v117, v164, v141
	buffer_load_dword v165, v165, s[4:7], 0 offen offset:72
	s_nop 0
	buffer_load_dword v164, v164, s[4:7], 0 offen offset:72
	ds_read_b128 v[172:175], v146
	ds_read_b128 v[176:179], v146 offset:2048
	ds_read_b128 v[188:191], v146 offset:4096
	ds_read_b128 v[216:219], v146 offset:6144
	v_mov_b32_e32 v125, v114
	v_mov_b32_e32 v114, v250
	v_mov_b32_e32 v113, v118
	v_mov_b32_e32 v118, v252
	ds_read_b128 v[250:253], v10 offset:8192
	ds_read_b128 v[156:159], v10 offset:10240
	ds_read_b128 v[120:123], v10 offset:12288
	s_barrier
	s_setprio 1
	v_and_b32_e32 v192, 0xff, v155
	v_and_b32_e32 v196, 0xff, v154
	v_and_b32_e32 v200, 0xff, v153
	v_and_b32_e32 v204, 0xff, v152
	v_and_b32_e32 v161, 0xff, v161
	v_and_b32_e32 v160, 0xff, v160
	v_and_b32_e32 v127, 0xff, v127
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[172:175], v[206:209], v[4:7], v119, v192 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[172:175], v[210:213], v[108:111], v119, v196 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[172:175], v[242:245], v[104:107], v119, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[172:175], v[246:249], v[100:103], v119, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[172:175], v[250:253], v[96:99], v119, v161 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[172:175], v[156:159], v[92:95], v119, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[172:175], v[120:123], v[88:91], v119, v127 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[176:179], v[206:209], v[84:87], v119, v192 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[176:179], v[210:213], v[80:83], v119, v196 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[176:179], v[242:245], v[76:79], v119, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[176:179], v[246:249], v[72:75], v119, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[176:179], v[250:253], v[68:71], v119, v161 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[176:179], v[156:159], v[64:67], v119, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[176:179], v[120:123], v[60:63], v119, v127 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[188:191], v[206:209], v[56:59], v115, v192 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], v[188:191], v[210:213], v[52:55], v115, v196 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[172:175], v[188:191], v[242:245], v[48:51], v115, v200 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[176:179], v[188:191], v[246:249], v[44:47], v115, v204 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[180:183], v[188:191], v[250:253], v[40:43], v115, v161 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[184:187], v[188:191], v[156:159], v[28:31], v115, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[188:191], v[188:191], v[120:123], v[16:19], v115, v127 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[192:195], v[216:219], v[206:209], v[142:145], v115, v192 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[196:199], v[216:219], v[210:213], v[12:15], v115, v196 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[200:203], v[216:219], v[242:245], v[148:151], v115, v200 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[204:207], v[216:219], v[246:249], v[24:27], v115, v204 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[208:211], v[216:219], v[250:253], v[32:35], v115, v161 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[212:215], v[216:219], v[156:159], v[36:39], v115, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[216:219], v[216:219], v[120:123], v[0:3], v115, v127 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	s_nop 0
	scratch_load_dword v0, off, off offset:52
	ds_read_b128 v[24:27], v126
	ds_read_b128 v[52:55], v126 offset:2048
	ds_read_b128 v[120:123], v126 offset:4096
	ds_read_b128 v[156:159], v126 offset:6144
	s_waitcnt vmcnt(0)
	ds_read_b128 v[242:245], v0
	ds_read_b128 v[246:249], v0 offset:2048
	ds_read_b128 v[250:253], v0 offset:4096
	ds_read_b128 v[128:131], v0 offset:6144
	ds_read_b128 v[134:137], v0 offset:8192
	ds_read_b128 v[148:151], v0 offset:10240
	ds_read_b128 v[142:145], v0 offset:12288
	s_waitcnt vmcnt(16)
	s_barrier
	s_setprio 1
	v_and_b32_e32 v127, 0xff, v9
	v_and_b32_e32 v133, 0xff, v133
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[24:27], v[242:245], v[4:7], v119, v127 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[24:27], v[246:249], v[108:111], v119, v133 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 2
	v_and_b32_e32 v108, 0xff, v8
	v_and_b32_e32 v109, 0xff, v112
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[52:55], v[242:245], v[84:87], v119, v127 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[24:27], v[250:253], v[104:107], v119, v108 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 2
	v_and_b32_e32 v104, 0xff, v132
	v_and_b32_e32 v105, 0xff, v20
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[52:55], v[246:249], v[80:83], v119, v133 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[24:27], v[128:131], v[100:103], v119, v104 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 2
	v_and_b32_e32 v100, 0xff, v21
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[24:27], v[148:151], v[92:95], v119, v105 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[24:27], v[134:137], v[96:99], v119, v100 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[24:27], v[142:145], v[88:91], v119, v109 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[52:55], v[250:253], v[76:79], v119, v108 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[52:55], v[128:131], v[72:75], v119, v104 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[52:55], v[134:137], v[68:71], v119, v100 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[52:55], v[148:151], v[64:67], v119, v105 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[52:55], v[142:145], v[60:63], v119, v109 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[120:123], v[242:245], v[56:59], v115, v127 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[120:123], v[246:249], v[152:155], v115, v133 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[120:123], v[250:253], v[172:175], v115, v108 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[120:123], v[128:131], v[176:179], v115, v104 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[120:123], v[134:137], v[180:183], v115, v100 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[120:123], v[148:151], v[184:187], v115, v105 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[120:123], v[142:145], v[188:191], v115, v109 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[156:159], v[242:245], v[192:195], v115, v127 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[156:159], v[246:249], v[196:199], v115, v133 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[156:159], v[250:253], v[200:203], v115, v108 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mov_b32_e32 v252, v118
	v_mov_b32_e32 v118, v113
	v_mov_b32_e32 v113, v116
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[156:159], v[128:131], v[204:207], v115, v104 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mov_b32_e32 v250, v114
	v_mov_b32_e32 v114, v125
	v_mov_b32_e32 v125, v140
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[156:159], v[134:137], v[208:211], v115, v100 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[156:159], v[148:151], v[212:215], v115, v105 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[156:159], v[142:145], v[216:219], v115, v109 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_mov_b32 m0, s30
	v_add_u32_e32 v115, 0xfff76000, v138
	s_waitcnt vmcnt(0)
	s_barrier
	buffer_load_dwordx4 v115, s[24:27], 0 offen lds
	v_add_u32_e32 v115, 0xfffa4000, v138
	s_mov_b32 m0, s31
	v_add_u32_e32 v119, 0xfffffbff, v220
	buffer_load_dwordx4 v115, s[24:27], 0 offen lds
	v_add_u32_e32 v115, 0xfffd2000, v138
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v115, s[24:27], 0 offen lds
	v_add_u32_e32 v115, 0x400, v171
	v_cmp_gt_i32_e32 vcc, 0, v115
	s_mov_b32 m0, s34
	s_nop 0
	v_cndmask_b32_e32 v119, v115, v119, vcc
	v_mul_hi_i32 v122, v119, s35
	v_add_u32_e32 v119, v122, v119
	v_lshrrev_b32_e32 v122, 31, v119
	v_ashrrev_i32_e32 v119, 9, v119
	v_add_u32_e32 v119, v119, v122
	v_ashrrev_i32_e32 v115, 31, v115
	v_xor_b32_e32 v115, v119, v115
	v_add_u32_e32 v119, s29, v115
	v_mul_i32_i24_e32 v115, 0xfffff480, v115
	v_mul_i32_i24_e32 v119, 0xb80, v119
	v_add3_u32 v115, v115, v119, v221
	buffer_load_dwordx4 v138, s[24:27], 0 offen lds
	v_add_u32_e32 v119, 0x1000, v115
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dword v119, s[20:23], 0 offen lds
	v_add_u32_e32 v119, 0xc800, v115
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dword v119, s[20:23], 0 offen lds
	v_add_u32_e32 v119, 0x18000, v115
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dword v119, s[20:23], 0 offen lds
	v_add_u32_e32 v119, 0x23800, v115
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dword v119, s[20:23], 0 offen lds
	v_add_u32_e32 v119, 0x2f000, v115
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dword v119, s[20:23], 0 offen lds
	v_add_u32_e32 v119, 0x3a800, v115
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dword v119, s[20:23], 0 offen lds
	v_add_u32_e32 v119, 0x46000, v115
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dword v119, s[20:23], 0 offen lds
	v_add_u32_e32 v119, 0x51800, v115
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dword v119, s[20:23], 0 offen lds
	v_add_u32_e32 v119, 0x5d000, v115
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dword v119, s[20:23], 0 offen lds
	v_add_u32_e32 v119, 0x68800, v115
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dword v119, s[20:23], 0 offen lds
	v_add_u32_e32 v119, 0x74000, v115
	s_mov_b32 m0, s46
	s_nop 0
	buffer_load_dword v119, s[20:23], 0 offen lds
	v_add_u32_e32 v119, 0x7f800, v115
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dword v119, s[20:23], 0 offen lds
	v_add_u32_e32 v119, 0x8b000, v115
	s_mov_b32 m0, s48
	v_add_u32_e32 v115, 0x96800, v115
	buffer_load_dword v119, s[20:23], 0 offen lds
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dword v115, s[20:23], 0 offen lds
	v_add_u32_e32 v115, 0x80, v222
	v_add_u32_e32 v119, 0xff7f, v224
	v_cmp_gt_i32_e32 vcc, s51, v223
	v_add_u32_e32 v122, 0x52, v222
	v_add_u32_e32 v127, 0xffad, v224
	v_cndmask_b32_e32 v115, v115, v119, vcc
	v_mul_i32_i24_sdwa v119, sext(v115), s68 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_add_u16_sdwa v115, v119, v115 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_lshrrev_b16_e32 v119, 15, v115
	v_ashrrev_i16_e32 v115, 5, v115
	v_add_u16_e32 v115, v115, v119
	v_cndmask_b32_e64 v119, 0, -1, vcc
	v_cmp_gt_i32_e32 vcc, s69, v223
	v_xor_b32_e32 v115, v115, v119
	v_add_u32_sdwa v119, v250, sext(v115) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_cndmask_b32_e32 v122, v122, v127, vcc
	v_mul_i32_i24_sdwa v127, sext(v122), s68 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_add_u16_sdwa v122, v127, v122 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_lshrrev_b16_e32 v127, 15, v122
	v_ashrrev_i16_e32 v122, 5, v122
	v_add_u16_e32 v122, v122, v127
	v_cndmask_b32_e64 v127, 0, -1, vcc
	v_xor_b32_e32 v122, v122, v127
	v_add_u32_sdwa v115, v124, sext(v115) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_bfe_i32 v122, v122, 0, 16
	v_mul_lo_u32 v119, v119, s14
	v_mul_lo_u32 v115, s14, v115
	v_mad_i32_i24 v119, v122, s50, v119
	v_mad_i32_i24 v115, v122, s50, v115
	v_add_u32_e32 v122, 0x200, v225
	v_add_u32_e32 v127, 0xfffffdff, v227
	v_cmp_gt_i32_e32 vcc, s70, v226
	v_add_u32_e32 v132, 0x148, v225
	v_add_u32_e32 v133, 0xfffffeb7, v227
	v_cndmask_b32_e32 v127, v122, v127, vcc
	v_cmp_gt_i32_e32 vcc, s72, v226
	v_mul_hi_i32 v131, v127, s35
	v_add_u32_e32 v127, v131, v127
	v_cndmask_b32_e32 v133, v132, v133, vcc
	v_mul_hi_i32 v134, v133, s35
	v_add_u32_e32 v133, v134, v133
	v_lshrrev_b32_e32 v134, 31, v133
	v_ashrrev_i32_e32 v133, 7, v133
	v_add_u32_e32 v133, v133, v134
	v_ashrrev_i32_e32 v132, 31, v132
	v_xor_b32_e32 v132, v133, v132
	v_lshrrev_b32_e32 v131, 31, v127
	v_ashrrev_i32_e32 v127, 7, v127
	v_mul_lo_u32 v150, v132, s15
	v_add_u32_e32 v132, 0x202, v225
	v_add_u32_e32 v133, 0xfffffdfd, v227
	v_cmp_gt_i32_e32 vcc, -2, v122
	v_add_u32_e32 v127, v127, v131
	v_ashrrev_i32_e32 v131, 31, v122
	v_cndmask_b32_e32 v122, v132, v133, vcc
	v_mul_hi_i32 v134, v122, s35
	v_add_u32_e32 v122, v134, v122
	v_lshrrev_b32_e32 v134, 31, v122
	v_ashrrev_i32_e32 v122, 7, v122
	v_cmp_gt_i32_e32 vcc, s75, v226
	v_add_u32_e32 v122, v122, v134
	v_ashrrev_i32_e32 v134, 31, v132
	v_cndmask_b32_e32 v132, v132, v133, vcc
	v_mul_hi_i32 v133, v132, s35
	v_add_u32_e32 v132, v133, v132
	v_lshrrev_b32_e32 v133, 31, v132
	v_ashrrev_i32_e32 v132, 7, v132
	v_xor_b32_e32 v122, v122, v134
	v_add_u32_e32 v132, v132, v133
	v_add_u32_e32 v151, v122, v125
	v_xor_b32_e32 v132, v132, v134
	v_mul_lo_u32 v133, v151, s12
	v_mul_lo_u32 v151, v132, s15
	v_sub_u32_e32 v132, v133, v151
	v_add_u32_e32 v133, 0x201, v228
	v_add_u32_e32 v134, 0xfffffdfe, v229
	v_cmp_gt_i32_e32 vcc, s71, v230
	v_xor_b32_e32 v127, v127, v131
	v_add_u32_e32 v131, v127, v125
	v_cndmask_b32_e32 v134, v133, v134, vcc
	v_mul_hi_i32 v152, v134, s35
	v_add_u32_e32 v134, v152, v134
	v_lshrrev_b32_e32 v152, 31, v134
	v_ashrrev_i32_e32 v134, 7, v134
	v_add_u32_e32 v134, v134, v152
	v_ashrrev_i32_e32 v133, 31, v133
	v_xor_b32_e32 v160, v134, v133
	v_add_u32_e32 v134, 0x149, v228
	v_add_u32_e32 v152, 0xfffffeb6, v229
	v_cmp_gt_i32_e32 vcc, s73, v230
	v_add_u32_e32 v133, v160, v255
	v_mul_lo_u32 v131, v131, s12
	v_cndmask_b32_e32 v152, v134, v152, vcc
	v_mul_hi_i32 v153, v152, s35
	v_add_u32_e32 v152, v153, v152
	v_lshrrev_b32_e32 v153, 31, v152
	v_ashrrev_i32_e32 v152, 7, v152
	v_add_u32_e32 v152, v152, v153
	v_ashrrev_i32_e32 v134, 31, v134
	v_xor_b32_e32 v134, v152, v134
	v_mul_lo_u32 v161, v134, s15
	v_add_u32_e32 v134, 0x203, v228
	v_add_u32_e32 v152, 0xfffffdfc, v229
	v_cmp_gt_i32_e32 vcc, s74, v230
	v_mul_lo_u32 v133, v133, s12
	v_sub_u32_e32 v131, v131, v150
	v_cndmask_b32_e32 v153, v134, v152, vcc
	v_mul_hi_i32 v154, v153, s35
	v_add_u32_e32 v153, v154, v153
	v_lshrrev_b32_e32 v154, 31, v153
	v_ashrrev_i32_e32 v153, 7, v153
	v_add_u32_e32 v153, v153, v154
	v_ashrrev_i32_e32 v154, 31, v134
	v_xor_b32_e32 v153, v153, v154
	v_add_u32_e32 v155, v153, v255
	v_mul_lo_u32 v155, v155, s12
	v_mul_lo_u32 v171, v153, s15
	v_sub_u32_e32 v153, v155, v171
	v_add_u32_e32 v155, v147, v127
	v_mul_lo_u32 v155, s12, v155
	v_sub_u32_e32 v155, v155, v150
	v_add_u32_e32 v172, v166, v155
	v_add_u32_e32 v155, v147, v122
	v_mul_lo_u32 v155, s12, v155
	v_sub_u32_e32 v155, v155, v151
	v_add3_u32 v173, v117, v155, v163
	v_add_u32_e32 v155, v160, v252
	v_mul_lo_u32 v155, v155, s12
	v_sub_u32_e32 v155, v155, v161
	v_add3_u32 v174, v117, v155, v162
	v_add_u32_e32 v155, 0x200, v228
	v_cmp_gt_i32_e32 vcc, -3, v155
	v_sub_u32_e32 v133, v133, v161
	scratch_load_dword v124, off, off offset:56
	v_cndmask_b32_e32 v134, v134, v152, vcc
	v_mul_hi_i32 v152, v134, s35
	v_add_u32_e32 v134, v152, v134
	v_lshrrev_b32_e32 v152, 31, v134
	v_ashrrev_i32_e32 v134, 7, v134
	v_add_u32_e32 v134, v134, v152
	v_xor_b32_e32 v175, v134, v154
	v_add_u32_e32 v134, v175, v252
	v_mul_lo_u32 v134, v134, s12
	v_add_u32_e32 v131, v166, v131
	v_add3_u32 v132, v117, v132, v163
	v_add3_u32 v133, v117, v133, v162
	v_add3_u32 v153, v117, v153, v162
	v_sub_u32_e32 v134, v134, v171
	v_add_u32_e32 v160, v160, v113
	v_add3_u32 v176, v117, v134, v162
	buffer_load_ubyte v155, v131, s[16:19], 0 offen offset:328
	buffer_load_ubyte v140, v132, s[16:19], 0 offen offset:514
	buffer_load_ubyte v154, v133, s[16:19], 0 offen offset:329
	s_nop 0
	buffer_load_ubyte v133, v153, s[16:19], 0 offen offset:515
	s_nop 0
	buffer_load_ubyte v153, v172, s[16:19], 0 offen offset:328
	buffer_load_ubyte v253, v173, s[16:19], 0 offen offset:514
	buffer_load_ubyte v152, v174, s[16:19], 0 offen offset:329
	buffer_load_ubyte v132, v176, s[16:19], 0 offen offset:515
	v_add_u32_e32 v172, v254, v127
	v_mul_lo_u32 v160, v160, s12
	v_add_u32_e32 v127, v139, v127
	v_mul_lo_u32 v172, s12, v172
	v_add_u32_e32 v173, v254, v122
	v_sub_u32_e32 v160, v160, v161
	v_add_u32_e32 v161, v175, v113
	v_mul_lo_u32 v127, s12, v127
	v_add_u32_e32 v122, v139, v122
	v_sub_u32_e32 v172, v172, v150
	v_mul_lo_u32 v173, s12, v173
	v_mul_lo_u32 v161, v161, s12
	v_sub_u32_e32 v127, v127, v150
	v_mul_lo_u32 v122, s12, v122
	v_add_u32_e32 v172, v166, v172
	v_sub_u32_e32 v173, v173, v151
	v_add3_u32 v160, v117, v160, v162
	v_sub_u32_e32 v161, v161, v171
	v_add_u32_e32 v127, v166, v127
	v_sub_u32_e32 v122, v122, v151
	v_add3_u32 v173, v117, v173, v163
	v_add3_u32 v171, v117, v161, v162
	v_add3_u32 v122, v117, v122, v163
	buffer_load_ubyte v161, v172, s[16:19], 0 offen offset:328
	buffer_load_ubyte v254, v173, s[16:19], 0 offen offset:514
	s_nop 0
	buffer_load_ubyte v160, v160, s[16:19], 0 offen offset:329
	s_nop 0
	buffer_load_ubyte v147, v171, s[16:19], 0 offen offset:515
	s_nop 0
	buffer_load_ubyte v127, v127, s[16:19], 0 offen offset:328
	s_nop 0
	buffer_load_ubyte v112, v122, s[16:19], 0 offen offset:514
	v_add3_u32 v119, v117, v119, v141
	v_add3_u32 v115, v117, v115, v141
	buffer_load_dword v119, v119, s[4:7], 0 offen offset:328
	s_nop 0
	buffer_load_dword v115, v115, s[4:7], 0 offen offset:328
	s_waitcnt vmcnt(16)
	ds_read_b128 v[148:151], v124
	ds_read_b128 v[156:159], v124 offset:2048
	ds_read_b128 v[208:211], v124 offset:4096
	ds_read_b128 v[212:215], v124 offset:6144
	ds_read_b128 v[216:219], v124 offset:8192
	ds_read_b128 v[220:223], v124 offset:10240
	ds_read_b128 v[224:227], v124 offset:12288
	ds_read_b128 v[120:123], v146 offset:32768
	ds_read_b128 v[128:131], v146 offset:34816
	ds_read_b128 v[134:137], v146 offset:36864
	ds_read_b128 v[142:145], v146 offset:38912
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[120:123], v[148:151], v[0:3], v165, v231 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[120:123], v[208:211], v[8:11], v165, v233 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[120:123], v[212:215], v[12:15], v165, v234 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[120:123], v[216:219], v[16:19], v165, v235 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[120:123], v[224:227], v[24:27], v165, v239 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[128:131], v[148:151], v[28:31], v165, v231 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[128:131], v[156:159], v[32:35], v165, v232 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[128:131], v[208:211], v[36:39], v165, v233 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[128:131], v[212:215], v[40:43], v165, v234 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[128:131], v[216:219], v[44:47], v165, v235 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[128:131], v[220:223], v[48:51], v165, v237 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[128:131], v[224:227], v[52:55], v165, v239 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[134:137], v[148:151], v[56:59], v164, v231 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[172:175], v[120:123], v[156:159], v[4:7], v165, v232 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[120:123], v[220:223], v[20:23], v165, v237 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[176:179], v[134:137], v[156:159], v[60:63], v164, v232 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[180:183], v[134:137], v[208:211], v[64:67], v164, v233 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[184:187], v[134:137], v[212:215], v[68:71], v164, v234 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[188:191], v[134:137], v[216:219], v[72:75], v164, v235 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[192:195], v[134:137], v[220:223], v[76:79], v164, v237 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[196:199], v[134:137], v[224:227], v[80:83], v164, v239 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[200:203], v[142:145], v[148:151], v[84:87], v164, v231 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[204:207], v[142:145], v[156:159], v[88:91], v164, v232 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[208:211], v[142:145], v[208:211], v[92:95], v164, v233 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[212:215], v[142:145], v[212:215], v[96:99], v164, v234 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[216:219], v[142:145], v[216:219], v[100:103], v164, v235 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[220:223], v[142:145], v[220:223], v[104:107], v164, v237 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[224:227], v[142:145], v[224:227], v[108:111], v164, v239 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	scratch_load_dword v4, off, off offset:60
	ds_read_b128 v[60:63], v126 offset:32768
	ds_read_b128 v[120:123], v126 offset:34816
	ds_read_b128 v[128:131], v126 offset:36864
	ds_read_b128 v[134:137], v126 offset:38912
	s_waitcnt vmcnt(0)
	ds_read_b128 v[142:145], v4
	ds_read_b128 v[148:151], v4 offset:2048
	ds_read_b128 v[156:159], v4 offset:4096
	ds_read_b128 v[228:231], v4 offset:6144
	ds_read_b128 v[232:235], v4 offset:8192
	ds_read_b128 v[242:245], v4 offset:10240
	ds_read_b128 v[246:249], v4 offset:12288
	s_waitcnt vmcnt(16)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[60:63], v[156:159], v[8:11], v165, v169 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 2
	scratch_load_dword v11, off, off offset:20
	scratch_load_dword v10, off, off offset:12
	scratch_load_dword v9, off, off offset:8
	scratch_load_dword v8, off, off offset:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[60:63], v[142:145], v[0:3], v165, v167 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[60:63], v[148:151], v[172:175], v165, v168 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[60:63], v[228:231], v[12:15], v165, v170 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[60:63], v[232:235], v[16:19], v165, v236 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[60:63], v[242:245], v[20:23], v165, v238 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[60:63], v[246:249], v[24:27], v165, v240 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[120:123], v[142:145], v[28:31], v165, v167 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[120:123], v[148:151], v[32:35], v165, v168 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[120:123], v[156:159], v[36:39], v165, v169 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[120:123], v[228:231], v[40:43], v165, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[120:123], v[232:235], v[44:47], v165, v236 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[120:123], v[242:245], v[48:51], v165, v238 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[120:123], v[246:249], v[52:55], v165, v240 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	scratch_load_dword v122, off, off offset:24
	scratch_load_dword v121, off, off offset:16
	scratch_load_dword v120, off, off
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[128:131], v[142:145], v[56:59], v164, v167 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[128:131], v[148:151], v[176:179], v164, v168 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[128:131], v[156:159], v[180:183], v164, v169 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[128:131], v[228:231], v[184:187], v164, v170 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[128:131], v[232:235], v[188:191], v164, v236 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[128:131], v[242:245], v[192:195], v164, v238 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[128:131], v[246:249], v[196:199], v164, v240 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], v[134:137], v[142:145], v[200:203], v164, v167 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[134:137], v[148:151], v[204:207], v164, v168 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], v[134:137], v[156:159], v[208:211], v164, v169 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[134:137], v[228:231], v[212:215], v164, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[134:137], v[232:235], v[216:219], v164, v236 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[134:137], v[242:245], v[220:223], v164, v238 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[134:137], v[246:249], v[224:227], v164, v240 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_add_i32 s13, s13, 2
	v_add_u32_e32 v241, 0x1000, v241
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v120, 0xfffffc00, v120
	v_add_u32_e32 v8, 0x400, v8
	v_add_u32_e32 v138, 0x100, v138
	v_add_u32_e32 v9, 0xfffffe00, v9
	v_add_u32_e32 v10, 0x200, v10
	v_add_u32_e32 v163, 0x200, v163
	v_add_u32_e32 v11, 0x200, v11
	v_add_u32_e32 v121, 0xfffffe00, v121
	v_add_u32_e32 v162, 0x200, v162
	v_add_u32_e32 v141, 0x200, v141
	v_add_u32_e32 v122, 0xffffff80, v122
	s_cmp_lt_u32 s13, 20
	v_add_u32_e32 v118, 0x80, v118
	s_cbranch_scc1 .LBB0_3
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	s_barrier
	scratch_load_dword v113, off, off offset:72
	s_movk_i32 s0, 0x7fff
	s_mul_hi_u32 s1, s8, s28
	s_mov_b32 s3, 0x27000
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v116, 0x10000, v113
	scratch_load_dword v113, off, off offset:76
	ds_read_b128 v[8:11], v116
	ds_read_b128 v[242:245], v116 offset:2048
	s_waitcnt vmcnt(0)
	v_add_u32_e32 v117, 0x10000, v113
	ds_read_b128 v[186:189], v117
	ds_read_b128 v[178:181], v116 offset:12288
	ds_read_b128 v[190:193], v117 offset:2048
	ds_read_b128 v[194:197], v117 offset:4096
	ds_read_b128 v[198:201], v116 offset:4096
	ds_read_b128 v[202:205], v116 offset:6144
	ds_read_b128 v[206:209], v117 offset:6144
	ds_read_b128 v[210:213], v117 offset:8192
	ds_read_b128 v[214:217], v116 offset:8192
	ds_read_b128 v[218:221], v116 offset:10240
	ds_read_b128 v[222:225], v117 offset:10240
	ds_read_b128 v[182:185], v117 offset:12288
	ds_read_b128 v[128:131], v146
	ds_read_b128 v[20:23], v146 offset:2048
	ds_read_b128 v[226:229], v126
	ds_read_b128 v[230:233], v126 offset:2048
	ds_read_b128 v[234:237], v146 offset:4096
	ds_read_b128 v[162:165], v146 offset:6144
	ds_read_b128 v[238:241], v126 offset:4096
	ds_read_b128 v[156:159], v126 offset:6144
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[128:131], v[8:11], v[4:7], v119, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mov_b32_e32 v116, 0x7fc0
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[226:229], v[186:189], v[4:7], v119, v140 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[128:131], v[242:245], v[108:111], v119, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[128:131], v[198:201], v[104:107], v119, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 5
	v_bfe_u32 v117, v7, 16, 1
	v_bfe_u32 v118, v6, 16, 1
	v_bfe_u32 v120, v5, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[136:139], v[226:229], v[190:193], v[108:111], v119, v133 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v121, v4, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[128:131], v[202:205], v[100:103], v119, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v111, v7, v117, s0
	v_add3_u32 v110, v6, v118, s0
	v_lshrrev_b32_e32 v111, 16, v111
	v_add3_u32 v109, v5, v120, s0
	v_lshrrev_b32_e32 v110, 16, v110
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], v[226:229], v[194:197], v[104:107], v119, v253 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v108, v4, v121, s0
	s_nop 1
	v_cndmask_b32_e32 v104, v116, v111, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_lshrrev_b32_e32 v106, 16, v109
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], v[226:229], v[206:209], v[100:103], v119, v132 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v105, v116, v110, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_lshrrev_b32_e32 v107, 16, v108
	v_bfe_u32 v100, v139, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[128:131], v[218:221], v[92:95], v119, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v106, v116, v106, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[128:131], v[178:181], v[88:91], v119, v127 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v107, v116, v107, vcc
	v_cmp_o_f32_e32 vcc, v139, v139
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[128:131], v[214:217], v[96:99], v119, v161 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_bfe_u32 v96, v138, 16, 1
	v_add3_u32 v99, v139, v100, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[20:23], v[8:11], v[84:87], v119, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v97, v137, 16, 1
	v_add3_u32 v96, v138, v96, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_bfe_u32 v98, v136, 16, 1
	v_add3_u32 v97, v137, v97, s0
	v_lshrrev_b32_e32 v96, 16, v96
	v_cndmask_b32_e32 v108, v116, v99, vcc
	v_cmp_o_f32_e32 vcc, v138, v138
	v_add3_u32 v98, v136, v98, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[226:229], v[222:225], v[92:95], v119, v147 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v109, v116, v96, vcc
	v_cmp_o_f32_e32 vcc, v137, v137
	s_nop 0
	v_lshrrev_b32_e32 v92, 16, v97
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[226:229], v[182:185], v[88:91], v119, v112 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v93, 16, v98
	v_cndmask_b32_e32 v110, v116, v92, vcc
	v_cmp_o_f32_e32 vcc, v136, v136
	v_bfe_u32 v88, v169, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[20:23], v[242:245], v[80:83], v119, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v89, v168, 16, 1
	v_cndmask_b32_e32 v111, v116, v93, vcc
	v_bfe_u32 v90, v167, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[20:23], v[198:201], v[76:79], v119, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v169, v169
	v_bfe_u32 v91, v166, 16, 1
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[230:233], v[186:189], v[84:87], v119, v140 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 2
	v_add3_u32 v87, v169, v88, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[20:23], v[202:205], v[72:75], v119, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v86, v168, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_add3_u32 v85, v167, v90, s0
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v117, v116, v87, vcc
	v_cmp_o_f32_e32 vcc, v168, v168
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[226:229], v[210:213], v[4:7], v119, v254 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v84, v166, v91, s0
	v_cndmask_b32_e32 v118, v116, v86, vcc
	v_cmp_o_f32_e32 vcc, v167, v167
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[230:233], v[190:193], v[80:83], v119, v133 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 2
	v_lshrrev_b32_e32 v80, 16, v85
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[230:233], v[194:197], v[76:79], v119, v253 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v81, 16, v84
	v_cndmask_b32_e32 v120, v116, v80, vcc
	v_cmp_o_f32_e32 vcc, v166, v166
	v_bfe_u32 v76, v173, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[20:23], v[214:217], v[68:71], v119, v161 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v77, v172, 16, 1
	v_cndmask_b32_e32 v121, v116, v81, vcc
	v_bfe_u32 v78, v171, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[20:23], v[218:221], v[64:67], v119, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v173, v173
	v_bfe_u32 v79, v170, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[230:233], v[206:209], v[72:75], v119, v132 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 2
	v_add3_u32 v75, v173, v76, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[20:23], v[178:181], v[60:63], v119, v127 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v74, v172, v77, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_add3_u32 v73, v171, v78, s0
	v_lshrrev_b32_e32 v74, 16, v74
	v_cndmask_b32_e32 v123, v116, v75, vcc
	v_cmp_o_f32_e32 vcc, v172, v172
	v_add3_u32 v72, v170, v79, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[230:233], v[210:213], v[68:71], v119, v254 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v124, v116, v74, vcc
	v_cmp_o_f32_e32 vcc, v171, v171
	s_nop 0
	v_lshrrev_b32_e32 v68, 16, v73
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[230:233], v[222:225], v[64:67], v119, v147 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v69, 16, v72
	v_cndmask_b32_e32 v113, v116, v68, vcc
	v_cmp_o_f32_e32 vcc, v170, v170
	v_bfe_u32 v64, v7, 16, 1
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[234:237], v[8:11], v[56:59], v115, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v65, v6, 16, 1
	v_cndmask_b32_e32 v126, v116, v69, vcc
	v_bfe_u32 v66, v5, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[230:233], v[182:185], v[60:63], v119, v112 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v67, v4, 16, 1
	s_nop 0
	v_add3_u32 v63, v7, v64, s0
	v_add3_u32 v62, v6, v65, s0
	v_lshrrev_b32_e32 v63, 16, v63
	v_add3_u32 v61, v5, v66, s0
	v_lshrrev_b32_e32 v62, 16, v62
	v_cndmask_b32_e32 v119, v116, v63, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[238:241], v[186:189], v[56:59], v115, v140 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v60, v4, v67, s0
	v_cndmask_b32_e32 v128, v116, v62, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_lshrrev_b32_e32 v56, 16, v61
	v_lshrrev_b32_e32 v57, 16, v60
	v_cndmask_b32_e32 v129, v116, v56, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[234:237], v[198:201], v[48:51], v115, v153 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v130, v116, v57, vcc
	v_cmp_o_f32_e32 vcc, v177, v177
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[238:241], v[194:197], v[4:7], v115, v253 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v48, v176, 16, 1
	v_bfe_u32 v49, v175, 16, 1
	v_bfe_u32 v50, v174, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[234:237], v[202:205], v[44:47], v115, v152 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v50, v174, v50, s0
	v_add3_u32 v49, v175, v49, s0
	v_add3_u32 v48, v176, v48, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[234:237], v[242:245], v[52:55], v115, v154 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v45, 16, v48
	v_lshrrev_b32_e32 v46, 16, v49
	v_lshrrev_b32_e32 v47, 16, v50
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[238:241], v[206:209], v[4:7], v115, v132 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[234:237], v[214:217], v[40:43], v115, v161 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[238:241], v[190:193], v[52:55], v115, v133 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 1
	v_bfe_u32 v40, v103, 16, 1
	v_bfe_u32 v52, v177, 16, 1
	v_add3_u32 v44, v177, v52, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[238:241], v[210:213], v[4:7], v115, v254 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v44, 16, v44
	v_cndmask_b32_e32 v135, v116, v44, vcc
	v_cmp_o_f32_e32 vcc, v176, v176
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[234:237], v[218:221], v[28:31], v115, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v136, v116, v45, vcc
	v_cmp_o_f32_e32 vcc, v175, v175
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[238:241], v[222:225], v[4:7], v115, v147 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v28, v102, 16, 1
	v_cndmask_b32_e32 v137, v116, v46, vcc
	v_cmp_o_f32_e32 vcc, v174, v174
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[234:237], v[178:181], v[16:19], v115, v127 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v29, v101, 16, 1
	v_cndmask_b32_e32 v138, v116, v47, vcc
	v_bfe_u32 v30, v100, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[238:241], v[182:185], v[4:7], v115, v112 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v16, v103, v40, s0
	v_add3_u32 v30, v100, v30, s0
	v_add3_u32 v29, v101, v29, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[162:165], v[8:11], v[142:145], v115, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v28, v102, v28, s0
	v_lshrrev_b32_e32 v21, 16, v28
	v_lshrrev_b32_e32 v22, 16, v29
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[156:159], v[186:189], v[4:7], v115, v140 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v166, 16, v30
	v_lshrrev_b32_e32 v139, 16, v16
	v_cmp_o_f32_e32 vcc, v103, v103
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[162:165], v[242:245], v[12:15], v115, v154 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v20, v116, v139, vcc
	v_cmp_o_f32_e32 vcc, v102, v102
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[156:159], v[190:193], v[4:7], v115, v133 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v21, v116, v21, vcc
	v_cmp_o_f32_e32 vcc, v101, v101
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[162:165], v[198:201], v[148:151], v115, v153 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v22, v116, v22, vcc
	v_cmp_o_f32_e32 vcc, v100, v100
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[156:159], v[194:197], v[4:7], v115, v253 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v23, v116, v166, vcc
	v_cmp_o_f32_e32 vcc, v99, v99
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[162:165], v[202:205], v[24:27], v115, v152 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_bfe_u32 v24, v99, 16, 1
	v_bfe_u32 v25, v98, 16, 1
	v_add3_u32 v24, v99, v24, s0
	v_bfe_u32 v26, v97, 16, 1
	v_add3_u32 v25, v98, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_bfe_u32 v27, v96, 16, 1
	v_add3_u32 v26, v97, v26, s0
	v_lshrrev_b32_e32 v25, 16, v25
	v_cndmask_b32_e32 v24, v116, v24, vcc
	v_cmp_o_f32_e32 vcc, v98, v98
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[156:159], v[206:209], v[4:7], v115, v132 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v27, v96, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v25, v116, v25, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[162:165], v[214:217], v[32:35], v115, v161 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v97, v97
	v_lshrrev_b32_e32 v27, 16, v27
	s_nop 0
	v_bfe_u32 v32, v95, 16, 1
	v_cndmask_b32_e32 v26, v116, v26, vcc
	v_cmp_o_f32_e32 vcc, v96, v96
	v_bfe_u32 v33, v94, 16, 1
	v_add3_u32 v32, v95, v32, s0
	v_cndmask_b32_e32 v27, v116, v27, vcc
	v_bfe_u32 v34, v93, 16, 1
	v_add3_u32 v33, v94, v33, s0
	v_lshrrev_b32_e32 v32, 16, v32
	v_cmp_o_f32_e32 vcc, v95, v95
	v_bfe_u32 v35, v92, 16, 1
	v_add3_u32 v34, v93, v34, s0
	v_lshrrev_b32_e32 v33, 16, v33
	v_cndmask_b32_e32 v32, v116, v32, vcc
	v_cmp_o_f32_e32 vcc, v94, v94
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[156:159], v[210:213], v[4:7], v115, v254 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v35, v92, v35, s0
	v_lshrrev_b32_e32 v34, 16, v34
	v_cndmask_b32_e32 v33, v116, v33, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[162:165], v[218:221], v[36:39], v115, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v93, v93
	v_lshrrev_b32_e32 v35, 16, v35
	s_nop 0
	v_bfe_u32 v36, v91, 16, 1
	v_cndmask_b32_e32 v34, v116, v34, vcc
	v_cmp_o_f32_e32 vcc, v92, v92
	v_bfe_u32 v37, v90, 16, 1
	v_add3_u32 v36, v91, v36, s0
	v_cndmask_b32_e32 v35, v116, v35, vcc
	v_bfe_u32 v38, v89, 16, 1
	v_add3_u32 v37, v90, v37, s0
	v_lshrrev_b32_e32 v36, 16, v36
	v_cmp_o_f32_e32 vcc, v91, v91
	v_add3_u32 v38, v89, v38, s0
	v_lshrrev_b32_e32 v37, 16, v37
	v_cndmask_b32_e32 v36, v116, v36, vcc
	v_cmp_o_f32_e32 vcc, v90, v90
	v_bfe_u32 v39, v88, 16, 1
	v_lshrrev_b32_e32 v38, 16, v38
	v_cndmask_b32_e32 v37, v116, v37, vcc
	v_cmp_o_f32_e32 vcc, v89, v89
	v_add3_u32 v39, v88, v39, s0
	v_lshrrev_b32_e32 v39, 16, v39
	v_cndmask_b32_e32 v38, v116, v38, vcc
	v_cmp_o_f32_e32 vcc, v88, v88
	v_bfe_u32 v88, v87, 16, 1
	v_bfe_u32 v89, v86, 16, 1
	v_add3_u32 v88, v87, v88, s0
	v_cndmask_b32_e32 v39, v116, v39, vcc
	v_bfe_u32 v90, v85, 16, 1
	v_add3_u32 v89, v86, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v87, v87
	v_bfe_u32 v91, v84, 16, 1
	v_add3_u32 v90, v85, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v87, v116, v88, vcc
	v_cmp_o_f32_e32 vcc, v86, v86
	v_add3_u32 v91, v84, v91, s0
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v86, v116, v89, vcc
	v_cmp_o_f32_e32 vcc, v85, v85
	v_bfe_u32 v88, v83, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v85, v116, v90, vcc
	v_cmp_o_f32_e32 vcc, v84, v84
	v_bfe_u32 v89, v82, 16, 1
	v_add3_u32 v88, v83, v88, s0
	v_cndmask_b32_e32 v84, v116, v91, vcc
	v_bfe_u32 v90, v81, 16, 1
	v_add3_u32 v89, v82, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v83, v83
	v_bfe_u32 v91, v80, 16, 1
	v_add3_u32 v90, v81, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v83, v116, v88, vcc
	v_cmp_o_f32_e32 vcc, v82, v82
	v_add3_u32 v91, v80, v91, s0
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v82, v116, v89, vcc
	v_cmp_o_f32_e32 vcc, v81, v81
	v_bfe_u32 v88, v79, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v81, v116, v90, vcc
	v_cmp_o_f32_e32 vcc, v80, v80
	v_bfe_u32 v89, v78, 16, 1
	v_add3_u32 v88, v79, v88, s0
	v_cndmask_b32_e32 v80, v116, v91, vcc
	v_bfe_u32 v90, v77, 16, 1
	v_add3_u32 v89, v78, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v79, v79
	v_bfe_u32 v91, v76, 16, 1
	v_add3_u32 v90, v77, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v79, v116, v88, vcc
	v_cmp_o_f32_e32 vcc, v78, v78
	v_add3_u32 v91, v76, v91, s0
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v78, v116, v89, vcc
	v_cmp_o_f32_e32 vcc, v77, v77
	v_bfe_u32 v88, v75, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v77, v116, v90, vcc
	v_cmp_o_f32_e32 vcc, v76, v76
	v_bfe_u32 v89, v74, 16, 1
	v_add3_u32 v88, v75, v88, s0
	v_cndmask_b32_e32 v76, v116, v91, vcc
	v_bfe_u32 v90, v73, 16, 1
	v_add3_u32 v89, v74, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v75, v75
	v_bfe_u32 v91, v72, 16, 1
	v_add3_u32 v90, v73, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v75, v116, v88, vcc
	v_cmp_o_f32_e32 vcc, v74, v74
	v_add3_u32 v91, v72, v91, s0
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v74, v116, v89, vcc
	v_cmp_o_f32_e32 vcc, v73, v73
	v_bfe_u32 v88, v71, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v73, v116, v90, vcc
	v_cmp_o_f32_e32 vcc, v72, v72
	v_bfe_u32 v89, v70, 16, 1
	v_add3_u32 v88, v71, v88, s0
	v_cndmask_b32_e32 v72, v116, v91, vcc
	v_bfe_u32 v90, v69, 16, 1
	v_add3_u32 v89, v70, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v71, v71
	v_bfe_u32 v91, v68, 16, 1
	v_add3_u32 v90, v69, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v71, v116, v88, vcc
	v_cmp_o_f32_e32 vcc, v70, v70
	v_add3_u32 v91, v68, v91, s0
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v70, v116, v89, vcc
	v_cmp_o_f32_e32 vcc, v69, v69
	v_bfe_u32 v88, v67, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v69, v116, v90, vcc
	v_cmp_o_f32_e32 vcc, v68, v68
	v_bfe_u32 v89, v66, 16, 1
	v_add3_u32 v88, v67, v88, s0
	v_cndmask_b32_e32 v68, v116, v91, vcc
	v_bfe_u32 v90, v65, 16, 1
	v_add3_u32 v89, v66, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v67, v67
	v_bfe_u32 v91, v64, 16, 1
	v_add3_u32 v90, v65, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v67, v116, v88, vcc
	v_cmp_o_f32_e32 vcc, v66, v66
	v_add3_u32 v91, v64, v91, s0
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v66, v116, v89, vcc
	v_cmp_o_f32_e32 vcc, v65, v65
	v_bfe_u32 v88, v63, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v65, v116, v90, vcc
	v_cmp_o_f32_e32 vcc, v64, v64
	v_bfe_u32 v89, v62, 16, 1
	v_add3_u32 v88, v63, v88, s0
	v_cndmask_b32_e32 v64, v116, v91, vcc
	v_bfe_u32 v90, v61, 16, 1
	v_add3_u32 v89, v62, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v63, v63
	v_bfe_u32 v91, v60, 16, 1
	v_add3_u32 v90, v61, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v63, v116, v88, vcc
	v_cmp_o_f32_e32 vcc, v62, v62
	v_add3_u32 v91, v60, v91, s0
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v62, v116, v89, vcc
	v_cmp_o_f32_e32 vcc, v61, v61
	v_bfe_u32 v88, v59, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v61, v116, v90, vcc
	v_cmp_o_f32_e32 vcc, v60, v60
	v_bfe_u32 v89, v58, 16, 1
	v_add3_u32 v88, v59, v88, s0
	v_cndmask_b32_e32 v60, v116, v91, vcc
	v_bfe_u32 v90, v57, 16, 1
	v_add3_u32 v89, v58, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v59, v59
	v_bfe_u32 v91, v56, 16, 1
	v_add3_u32 v90, v57, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v59, v116, v88, vcc
	v_cmp_o_f32_e32 vcc, v58, v58
	v_add3_u32 v91, v56, v91, s0
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v58, v116, v89, vcc
	v_cmp_o_f32_e32 vcc, v57, v57
	v_bfe_u32 v88, v55, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v57, v116, v90, vcc
	v_cmp_o_f32_e32 vcc, v56, v56
	v_bfe_u32 v89, v54, 16, 1
	v_add3_u32 v88, v55, v88, s0
	v_cndmask_b32_e32 v56, v116, v91, vcc
	v_bfe_u32 v90, v53, 16, 1
	v_add3_u32 v89, v54, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v55, v55
	v_bfe_u32 v91, v52, 16, 1
	v_add3_u32 v90, v53, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v55, v116, v88, vcc
	v_cmp_o_f32_e32 vcc, v54, v54
	v_add3_u32 v91, v52, v91, s0
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v54, v116, v89, vcc
	v_cmp_o_f32_e32 vcc, v53, v53
	v_bfe_u32 v88, v51, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v53, v116, v90, vcc
	v_cmp_o_f32_e32 vcc, v52, v52
	v_bfe_u32 v89, v50, 16, 1
	v_add3_u32 v88, v51, v88, s0
	v_cndmask_b32_e32 v52, v116, v91, vcc
	v_bfe_u32 v90, v49, 16, 1
	v_add3_u32 v89, v50, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v51, v51
	v_bfe_u32 v91, v48, 16, 1
	v_add3_u32 v90, v49, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v51, v116, v88, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_add3_u32 v91, v48, v91, s0
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v50, v116, v89, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_bfe_u32 v88, v47, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v49, v116, v90, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_bfe_u32 v89, v46, 16, 1
	v_add3_u32 v88, v47, v88, s0
	v_cndmask_b32_e32 v48, v116, v91, vcc
	v_bfe_u32 v90, v45, 16, 1
	v_add3_u32 v89, v46, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v47, v47
	v_bfe_u32 v91, v44, 16, 1
	v_add3_u32 v90, v45, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v47, v116, v88, vcc
	v_cmp_o_f32_e32 vcc, v46, v46
	v_add3_u32 v91, v44, v91, s0
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v46, v116, v89, vcc
	v_cmp_o_f32_e32 vcc, v45, v45
	v_bfe_u32 v88, v43, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v45, v116, v90, vcc
	v_cmp_o_f32_e32 vcc, v44, v44
	v_bfe_u32 v89, v42, 16, 1
	v_add3_u32 v88, v43, v88, s0
	v_cndmask_b32_e32 v44, v116, v91, vcc
	v_bfe_u32 v90, v41, 16, 1
	v_add3_u32 v89, v42, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v43, v43
	v_bfe_u32 v91, v40, 16, 1
	v_add3_u32 v90, v41, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v43, v116, v88, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	v_add3_u32 v91, v40, v91, s0
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v42, v116, v89, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	v_bfe_u32 v88, v31, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v41, v116, v90, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_bfe_u32 v89, v30, 16, 1
	v_add3_u32 v88, v31, v88, s0
	v_cndmask_b32_e32 v40, v116, v91, vcc
	v_bfe_u32 v90, v29, 16, 1
	v_add3_u32 v89, v30, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v31, v31
	v_bfe_u32 v91, v28, 16, 1
	v_add3_u32 v90, v29, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v31, v116, v88, vcc
	v_cmp_o_f32_e32 vcc, v30, v30
	v_add3_u32 v91, v28, v91, s0
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v30, v116, v89, vcc
	v_cmp_o_f32_e32 vcc, v29, v29
	v_bfe_u32 v88, v19, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v29, v116, v90, vcc
	v_cmp_o_f32_e32 vcc, v28, v28
	v_bfe_u32 v89, v18, 16, 1
	v_add3_u32 v88, v19, v88, s0
	v_cndmask_b32_e32 v28, v116, v91, vcc
	v_bfe_u32 v90, v17, 16, 1
	v_add3_u32 v89, v18, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v19, v19
	v_bfe_u32 v91, v16, 16, 1
	v_add3_u32 v90, v17, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v19, v116, v88, vcc
	v_cmp_o_f32_e32 vcc, v18, v18
	v_add3_u32 v91, v16, v91, s0
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v18, v116, v89, vcc
	v_cmp_o_f32_e32 vcc, v17, v17
	v_bfe_u32 v88, v15, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v17, v116, v90, vcc
	v_cmp_o_f32_e32 vcc, v16, v16
	v_bfe_u32 v89, v14, 16, 1
	v_add3_u32 v88, v15, v88, s0
	v_cndmask_b32_e32 v16, v116, v91, vcc
	v_bfe_u32 v90, v13, 16, 1
	v_add3_u32 v89, v14, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v15, v15
	v_bfe_u32 v91, v12, 16, 1
	v_add3_u32 v90, v13, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v15, v116, v88, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[156:159], v[222:225], v[4:7], v115, v147 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v91, v12, v91, s0
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v14, v116, v89, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[162:165], v[178:181], v[0:3], v115, v127 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v13, v13
	v_bfe_u32 v88, v11, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v13, v116, v90, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v89, v10, 16, 1
	v_add3_u32 v88, v11, v88, s0
	v_cndmask_b32_e32 v12, v116, v91, vcc
	v_bfe_u32 v90, v9, 16, 1
	v_add3_u32 v89, v10, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v11, v11
	v_bfe_u32 v91, v8, 16, 1
	v_add3_u32 v90, v9, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v11, v116, v88, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[156:159], v[182:185], v[0:3], v115, v112 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v91, v8, v91, s0
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v10, v116, v89, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_bfe_u32 v88, v7, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v9, v116, v90, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v89, v6, 16, 1
	v_add3_u32 v88, v7, v88, s0
	v_cndmask_b32_e32 v8, v116, v91, vcc
	v_bfe_u32 v90, v5, 16, 1
	v_add3_u32 v89, v6, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v91, v4, 16, 1
	v_add3_u32 v90, v5, v90, s0
	v_lshrrev_b32_e32 v89, 16, v89
	v_cndmask_b32_e32 v7, v116, v88, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v91, v4, v91, s0
	v_lshrrev_b32_e32 v90, 16, v90
	v_cndmask_b32_e32 v6, v116, v89, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_bfe_u32 v88, v3, 16, 1
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v5, v116, v90, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v89, v2, 16, 1
	v_add3_u32 v88, v3, v88, s0
	v_cndmask_b32_e32 v4, v116, v91, vcc
	v_add3_u32 v89, v2, v89, s0
	v_lshrrev_b32_e32 v88, 16, v88
	v_cmp_o_f32_e32 vcc, v3, v3
	v_lshrrev_b32_e32 v89, 16, v89
	v_bfe_u32 v90, v1, 16, 1
	v_cndmask_b32_e32 v3, v116, v88, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	scratch_load_dword v88, off, off offset:64
	v_bfe_u32 v91, v0, 16, 1
	v_cndmask_b32_e32 v2, v116, v89, vcc
	scratch_load_dword v89, off, off offset:68
	v_add3_u32 v91, v0, v91, s0
	v_add3_u32 v90, v1, v90, s0
	s_mul_i32 s0, s9, s28
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s8, s28
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s10, s0
	s_addc_u32 s1, s11, s1
	s_lshl_b32 s2, s29, 1
	v_lshrrev_b32_e32 v90, 16, v90
	v_cmp_o_f32_e32 vcc, v1, v1
	s_add_u32 s0, s0, s2
	v_lshrrev_b32_e32 v91, 16, v91
	v_cndmask_b32_e32 v1, v116, v90, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	s_addc_u32 s1, s1, 0
	s_and_b32 s2, s8, 0x3fff
	s_lshl_b32 s4, s8, 1
	v_cndmask_b32_e32 v0, v116, v91, vcc
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	s_or_b32 s1, s2, s1
	s_or_b32 s1, s1, 2.0
	s_mov_b32 s2, 0x7ffffffd
	s_lshl_b32 s5, s8, 4
	s_waitcnt vmcnt(1)
	v_lshl_or_b32 v88, v88, 2, v250
	v_mul_lo_u32 v88, s8, v88
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v89, 1, v89
	v_lshl_add_u32 v90, v88, 1, v89
	v_add_u32_e32 v91, s4, v90
	v_add_u32_e32 v92, s4, v91
	v_add_u32_e32 v93, s4, v92
	buffer_store_short v107, v90, s[0:3], 0 offen
	buffer_store_short v106, v91, s[0:3], 0 offen
	buffer_store_short v105, v92, s[0:3], 0 offen
	buffer_store_short v104, v93, s[0:3], 0 offen
	buffer_store_short v111, v90, s[0:3], 0 offen offset:32
	buffer_store_short v110, v91, s[0:3], 0 offen offset:32
	buffer_store_short v109, v92, s[0:3], 0 offen offset:32
	buffer_store_short v108, v93, s[0:3], 0 offen offset:32
	buffer_store_short v121, v90, s[0:3], 0 offen offset:64
	buffer_store_short v120, v91, s[0:3], 0 offen offset:64
	buffer_store_short v118, v92, s[0:3], 0 offen offset:64
	buffer_store_short v117, v93, s[0:3], 0 offen offset:64
	buffer_store_short v126, v90, s[0:3], 0 offen offset:96
	buffer_store_short v113, v91, s[0:3], 0 offen offset:96
	buffer_store_short v124, v92, s[0:3], 0 offen offset:96
	buffer_store_short v123, v93, s[0:3], 0 offen offset:96
	buffer_store_short v130, v90, s[0:3], 0 offen offset:128
	buffer_store_short v129, v91, s[0:3], 0 offen offset:128
	buffer_store_short v128, v92, s[0:3], 0 offen offset:128
	buffer_store_short v119, v93, s[0:3], 0 offen offset:128
	buffer_store_short v138, v90, s[0:3], 0 offen offset:160
	buffer_store_short v137, v91, s[0:3], 0 offen offset:160
	buffer_store_short v136, v92, s[0:3], 0 offen offset:160
	buffer_store_short v135, v93, s[0:3], 0 offen offset:160
	buffer_store_short v23, v90, s[0:3], 0 offen offset:192
	buffer_store_short v22, v91, s[0:3], 0 offen offset:192
	buffer_store_short v21, v92, s[0:3], 0 offen offset:192
	buffer_store_short v20, v93, s[0:3], 0 offen offset:192
	v_add_u32_e32 v20, s5, v88
	v_lshl_add_u32 v21, v20, 1, v89
	v_add_u32_e32 v22, s4, v21
	v_add_u32_e32 v23, s4, v22
	buffer_store_short v27, v21, s[0:3], 0 offen
	buffer_store_short v26, v22, s[0:3], 0 offen
	buffer_store_short v25, v23, s[0:3], 0 offen
	v_add_u32_e32 v25, s4, v23
	v_add_u32_e32 v20, s5, v20
	buffer_store_short v24, v25, s[0:3], 0 offen
	buffer_store_short v35, v21, s[0:3], 0 offen offset:32
	buffer_store_short v34, v22, s[0:3], 0 offen offset:32
	buffer_store_short v33, v23, s[0:3], 0 offen offset:32
	buffer_store_short v32, v25, s[0:3], 0 offen offset:32
	buffer_store_short v39, v21, s[0:3], 0 offen offset:64
	buffer_store_short v38, v22, s[0:3], 0 offen offset:64
	buffer_store_short v37, v23, s[0:3], 0 offen offset:64
	buffer_store_short v36, v25, s[0:3], 0 offen offset:64
	buffer_store_short v84, v21, s[0:3], 0 offen offset:96
	buffer_store_short v85, v22, s[0:3], 0 offen offset:96
	buffer_store_short v86, v23, s[0:3], 0 offen offset:96
	buffer_store_short v87, v25, s[0:3], 0 offen offset:96
	buffer_store_short v80, v21, s[0:3], 0 offen offset:128
	buffer_store_short v81, v22, s[0:3], 0 offen offset:128
	buffer_store_short v82, v23, s[0:3], 0 offen offset:128
	buffer_store_short v83, v25, s[0:3], 0 offen offset:128
	buffer_store_short v76, v21, s[0:3], 0 offen offset:160
	buffer_store_short v77, v22, s[0:3], 0 offen offset:160
	buffer_store_short v78, v23, s[0:3], 0 offen offset:160
	buffer_store_short v79, v25, s[0:3], 0 offen offset:160
	buffer_store_short v72, v21, s[0:3], 0 offen offset:192
	buffer_store_short v73, v22, s[0:3], 0 offen offset:192
	buffer_store_short v74, v23, s[0:3], 0 offen offset:192
	buffer_store_short v75, v25, s[0:3], 0 offen offset:192
	v_lshl_add_u32 v21, v20, 1, v89
	v_add_u32_e32 v22, s4, v21
	v_add_u32_e32 v23, s4, v22
	v_add_u32_e32 v20, s5, v20
	v_add_u32_e32 v24, s4, v23
	v_lshl_add_u32 v20, v20, 1, v89
	buffer_store_short v68, v21, s[0:3], 0 offen
	buffer_store_short v69, v22, s[0:3], 0 offen
	buffer_store_short v70, v23, s[0:3], 0 offen
	buffer_store_short v71, v24, s[0:3], 0 offen
	buffer_store_short v64, v21, s[0:3], 0 offen offset:32
	buffer_store_short v65, v22, s[0:3], 0 offen offset:32
	buffer_store_short v66, v23, s[0:3], 0 offen offset:32
	buffer_store_short v67, v24, s[0:3], 0 offen offset:32
	buffer_store_short v60, v21, s[0:3], 0 offen offset:64
	buffer_store_short v61, v22, s[0:3], 0 offen offset:64
	buffer_store_short v62, v23, s[0:3], 0 offen offset:64
	buffer_store_short v63, v24, s[0:3], 0 offen offset:64
	buffer_store_short v56, v21, s[0:3], 0 offen offset:96
	buffer_store_short v57, v22, s[0:3], 0 offen offset:96
	buffer_store_short v58, v23, s[0:3], 0 offen offset:96
	buffer_store_short v59, v24, s[0:3], 0 offen offset:96
	buffer_store_short v52, v21, s[0:3], 0 offen offset:128
	buffer_store_short v53, v22, s[0:3], 0 offen offset:128
	buffer_store_short v54, v23, s[0:3], 0 offen offset:128
	buffer_store_short v55, v24, s[0:3], 0 offen offset:128
	buffer_store_short v48, v21, s[0:3], 0 offen offset:160
	buffer_store_short v49, v22, s[0:3], 0 offen offset:160
	buffer_store_short v50, v23, s[0:3], 0 offen offset:160
	buffer_store_short v51, v24, s[0:3], 0 offen offset:160
	buffer_store_short v44, v21, s[0:3], 0 offen offset:192
	buffer_store_short v45, v22, s[0:3], 0 offen offset:192
	buffer_store_short v46, v23, s[0:3], 0 offen offset:192
	buffer_store_short v47, v24, s[0:3], 0 offen offset:192
	v_add_u32_e32 v21, s4, v20
	v_add_u32_e32 v22, s4, v21
	v_add_u32_e32 v23, s4, v22
	buffer_store_short v40, v20, s[0:3], 0 offen
	buffer_store_short v41, v21, s[0:3], 0 offen
	buffer_store_short v42, v22, s[0:3], 0 offen
	buffer_store_short v43, v23, s[0:3], 0 offen
	buffer_store_short v28, v20, s[0:3], 0 offen offset:32
	buffer_store_short v29, v21, s[0:3], 0 offen offset:32
	buffer_store_short v30, v22, s[0:3], 0 offen offset:32
	buffer_store_short v31, v23, s[0:3], 0 offen offset:32
	buffer_store_short v16, v20, s[0:3], 0 offen offset:64
	buffer_store_short v17, v21, s[0:3], 0 offen offset:64
	buffer_store_short v18, v22, s[0:3], 0 offen offset:64
	buffer_store_short v19, v23, s[0:3], 0 offen offset:64
	buffer_store_short v12, v20, s[0:3], 0 offen offset:96
	buffer_store_short v13, v21, s[0:3], 0 offen offset:96
	buffer_store_short v14, v22, s[0:3], 0 offen offset:96
	buffer_store_short v15, v23, s[0:3], 0 offen offset:96
	buffer_store_short v8, v20, s[0:3], 0 offen offset:128
	buffer_store_short v9, v21, s[0:3], 0 offen offset:128
	buffer_store_short v10, v22, s[0:3], 0 offen offset:128
	buffer_store_short v11, v23, s[0:3], 0 offen offset:128
	buffer_store_short v4, v20, s[0:3], 0 offen offset:160
	buffer_store_short v5, v21, s[0:3], 0 offen offset:160
	buffer_store_short v6, v22, s[0:3], 0 offen offset:160
	buffer_store_short v7, v23, s[0:3], 0 offen offset:160
	buffer_store_short v0, v20, s[0:3], 0 offen offset:192
	buffer_store_short v1, v21, s[0:3], 0 offen offset:192
	buffer_store_short v2, v22, s[0:3], 0 offen offset:192
	buffer_store_short v3, v23, s[0:3], 0 offen offset:192
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_256x224x256_7168x6272x5888
		.amdhsa_group_segment_fixed_size 122880
		.amdhsa_private_segment_fixed_size 84
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
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 256
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 256
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
	.size	wave_mxfp4_static_gemm_256x224x256_7168x6272x5888, .Lfunc_end0-wave_mxfp4_static_gemm_256x224x256_7168x6272x5888

	.set wave_mxfp4_static_gemm_256x224x256_7168x6272x5888.num_vgpr, 256
	.set wave_mxfp4_static_gemm_256x224x256_7168x6272x5888.num_agpr, 0
	.set wave_mxfp4_static_gemm_256x224x256_7168x6272x5888.numbered_sgpr, 76
	.set wave_mxfp4_static_gemm_256x224x256_7168x6272x5888.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_256x224x256_7168x6272x5888.private_seg_size, 84
	.set wave_mxfp4_static_gemm_256x224x256_7168x6272x5888.uses_vcc, 1
	.set wave_mxfp4_static_gemm_256x224x256_7168x6272x5888.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_256x224x256_7168x6272x5888.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_256x224x256_7168x6272x5888.has_recursion, 0
	.set wave_mxfp4_static_gemm_256x224x256_7168x6272x5888.has_indirect_call, 0
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
    .name:           wave_mxfp4_static_gemm_256x224x256_7168x6272x5888
    .private_segment_fixed_size: 84
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     82
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_256x224x256_7168x6272x5888.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 20
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
