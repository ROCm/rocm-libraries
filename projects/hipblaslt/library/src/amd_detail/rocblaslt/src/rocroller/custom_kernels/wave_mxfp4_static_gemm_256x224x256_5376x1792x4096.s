; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_256x224x256_5376x1792x4096
	.p2align	8
	.type	wave_mxfp4_static_gemm_256x224x256_5376x1792x4096,@function
wave_mxfp4_static_gemm_256x224x256_5376x1792x4096:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v9, 0x3ff, v0
	v_bfe_u32 v5, v0, 10, 10
	v_lshrrev_b32_e32 v8, 6, v9
	v_lshlrev_b32_e32 v3, 5, v5
	v_lshrrev_b32_e32 v4, 3, v9
	v_lshl_or_b32 v0, v8, 3, v3
	s_lshl_b32 s16, s16, 8
	v_xor_b32_e32 v1, v4, v9
	s_mov_b64 s[24:25], s[2:3]
	v_readfirstlane_b32 s2, v0
	v_or3_b32 v0, v4, v3, s16
	v_lshlrev_b32_e32 v1, 4, v1
	v_and_b32_e32 v144, 0x70, v1
	v_lshlrev_b32_e32 v145, 11, v0
	s_and_b32 s3, s25, 0xffff
	s_lshl_b32 s19, s2, 7
	s_or_b32 s25, s3, 0x48000000
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_or_b32_e32 v0, v145, v144
	s_mov_b32 m0, s19
	v_or_b32_e32 v146, 0x20000, v145
	s_or_b32 s33, s19, 0x2000
	buffer_load_dwordx4 v0, s[24:27], 0 offen lds
	v_or_b32_e32 v0, v146, v144
	s_mov_b32 m0, s33
	v_or_b32_e32 v147, 0x40000, v145
	s_or_b32 s34, s19, 0x4000
	buffer_load_dwordx4 v0, s[24:27], 0 offen lds
	v_or_b32_e32 v0, v147, v144
	s_mov_b32 m0, s34
	v_or_b32_e32 v148, 0x60000, v145
	s_or_b32 s35, s19, 0x6000
	buffer_load_dwordx4 v0, s[24:27], 0 offen lds
	v_or_b32_e32 v0, v148, v144
	s_mov_b32 m0, s35
	v_lshrrev_b32_e32 v6, 5, v9
	buffer_load_dwordx4 v0, s[24:27], 0 offen lds
	v_lshlrev_b32_e32 v0, 3, v5
	v_lshl_or_b32 v0, v8, 1, v0
	v_lshrrev_b32_e32 v7, 2, v9
	v_readfirstlane_b32 s2, v0
	v_bfe_u32 v0, v9, 2, 3
	v_and_b32_e32 v14, 31, v9
	v_lshlrev_b32_e32 v0, 2, v0
	v_bitop3_b32 v1, v7, v6, 7 bitop3:0x6c
	v_sub_u32_e32 v0, v14, v0
	v_lshl_add_u32 v0, v1, 2, v0
	v_ashrrev_i32_e32 v14, 31, v0
	v_xor_b32_e32 v0, v14, v0
	v_ashrrev_i32_e32 v15, 31, v0
	v_lshrrev_b32_e32 v15, 29, v15
	v_add_u32_e32 v0, v0, v15
	v_lshlrev_b32_e32 v11, 2, v6
	v_ashrrev_i32_e32 v0, 3, v0
	v_and_b32_e32 v15, 0xfc, v9
	v_add_u32_e32 v2, v3, v9
	v_lshlrev_b32_e32 v10, 6, v1
	v_lshrrev_b32_e32 v13, 1, v1
	v_xor_b32_e32 v0, v0, v14
	v_sub_u32_e32 v11, v11, v15
	v_mul_i32_i24_e32 v12, 0xffffff80, v13
	v_lshlrev_b32_e32 v14, 7, v0
	v_add3_u32 v10, v2, v10, v11
	v_add3_u32 v10, v10, v12, v14
	v_ashrrev_i32_e32 v11, 31, v10
	v_xor_b32_e32 v10, v11, v10
	v_ashrrev_i32_e32 v12, 31, v10
	v_lshrrev_b32_e32 v12, 23, v12
	v_add_u32_e32 v10, v10, v12
	v_lshlrev_b32_e32 v152, 2, v9
	v_lshlrev_b32_e32 v14, 7, v5
	v_ashrrev_i32_e32 v10, 9, v10
	v_sub_u32_e32 v16, v6, v7
	v_lshlrev_b32_e32 v12, 8, v1
	v_add_u32_e32 v1, v14, v152
	v_xor_b32_e32 v10, v10, v11
	s_mul_i32 s18, s17, 0xe0
	v_lshlrev_b32_e32 v11, 9, v0
	v_lshl_add_u32 v0, v16, 4, v1
	v_lshlrev_b32_e32 v13, 9, v13
	s_mov_b64 s[20:21], s[6:7]
	v_add_u32_e32 v15, s18, v10
	v_add3_u32 v0, v0, v12, v11
	v_lshl_or_b32 v10, v10, 11, v13
	s_lshl_b32 s63, s2, 7
	v_sub_u32_e32 v0, v0, v10
	s_and_b32 s3, s21, 0xffff
	s_add_i32 s36, s63, 0x10000
	s_or_b32 s65, s63, 0x800
	s_or_b32 s21, s3, 0x48000000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_lshl_add_u32 v149, v15, 11, v0
	s_mov_b32 m0, s36
	s_add_i32 s37, s65, 0x10000
	s_or_b32 s64, s63, 0x1000
	buffer_load_dword v149, s[20:23], 0 offen lds
	v_add_u32_e32 v0, 0x8000, v149
	s_mov_b32 m0, s37
	s_add_i32 s38, s64, 0x10000
	s_or_b32 s62, s63, 0x1800
	buffer_load_dword v0, s[20:23], 0 offen lds
	v_add_u32_e32 v0, 0x10000, v149
	s_mov_b32 m0, s38
	s_add_i32 s39, s62, 0x10000
	s_or_b32 s61, s63, 0x2000
	buffer_load_dword v0, s[20:23], 0 offen lds
	v_add_u32_e32 v0, 0x18000, v149
	s_mov_b32 m0, s39
	s_add_i32 s40, s61, 0x10000
	s_or_b32 s60, s63, 0x2800
	buffer_load_dword v0, s[20:23], 0 offen lds
	v_add_u32_e32 v0, 0x20000, v149
	s_mov_b32 m0, s40
	s_add_i32 s41, s60, 0x10000
	s_or_b32 s59, s63, 0x3000
	buffer_load_dword v0, s[20:23], 0 offen lds
	v_add_u32_e32 v0, 0x28000, v149
	s_mov_b32 m0, s41
	s_add_i32 s42, s59, 0x10000
	s_or_b32 s58, s63, 0x3800
	buffer_load_dword v0, s[20:23], 0 offen lds
	v_add_u32_e32 v0, 0x30000, v149
	s_mov_b32 m0, s42
	s_add_i32 s43, s58, 0x10000
	s_or_b32 s57, s63, 0x4000
	buffer_load_dword v0, s[20:23], 0 offen lds
	v_add_u32_e32 v0, 0x38000, v149
	s_mov_b32 m0, s43
	s_add_i32 s44, s57, 0x10000
	s_or_b32 s70, s63, 0x4800
	buffer_load_dword v0, s[20:23], 0 offen lds
	v_add_u32_e32 v0, 0x40000, v149
	s_mov_b32 m0, s44
	s_add_i32 s45, s70, 0x10000
	s_or_b32 s46, s63, 0x5000
	buffer_load_dword v0, s[20:23], 0 offen lds
	v_add_u32_e32 v0, 0x48000, v149
	s_mov_b32 m0, s45
	s_add_i32 s47, s46, 0x10000
	s_or_b32 s48, s63, 0x5800
	s_or_b32 s50, s63, 0x6000
	s_or_b32 s52, s63, 0x6800
	s_mul_i32 s15, s15, s16
	s_mul_hi_u32 s2, s14, s16
	buffer_load_dword v0, s[20:23], 0 offen lds
	v_add_u32_e32 v0, 0x50000, v149
	s_mov_b32 m0, s47
	s_add_i32 s49, s48, 0x10000
	s_add_i32 s51, s50, 0x10000
	s_add_i32 s53, s52, 0x10000
	s_add_i32 s2, s2, s15
	s_mul_i32 s3, s14, s16
	buffer_load_dword v0, s[20:23], 0 offen lds
	v_add_u32_e32 v0, 0x58000, v149
	s_mov_b32 m0, s49
	s_add_u32 s4, s4, s3
	buffer_load_dword v0, s[20:23], 0 offen lds
	v_add_u32_e32 v0, 0x60000, v149
	s_mov_b32 m0, s51
	v_lshrrev_b32_e32 v21, 4, v9
	v_bfe_u32 v18, v9, 5, 1
	s_movk_i32 s6, 0xffc0
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s14, 0x3fff
	buffer_load_dword v0, s[20:23], 0 offen lds
	v_add_u32_e32 v0, 0x68000, v149
	s_mov_b32 m0, s53
	v_bfe_u32 v115, v9, 4, 2
	v_and_b32_e32 v125, 0xc0, v9
	v_mad_i32_i24 v17, v21, s6, v152
	v_lshlrev_b32_e32 v15, 7, v18
	s_lshl_b32 s3, s3, 16
	s_and_b32 s2, s2, 0xffff
	buffer_load_dword v0, s[20:23], 0 offen lds
	v_or_b32_e32 v0, v18, v125
	v_lshlrev_b32_e32 v10, 6, v115
	v_sub_u32_e32 v16, v17, v15
	s_or_b32 s2, s3, s2
	v_add_u32_e32 v153, v16, v10
	v_mul_lo_u32 v154, s14, v0
	s_or_b32 s5, s2, 2.0
	s_mov_b32 s28, s4
	s_mov_b32 s29, s5
	s_mov_b32 s30, s26
	s_mov_b32 s31, s27
	v_add_u32_e32 v0, v154, v153
	s_lshl_b32 s15, s14, 5
	s_movk_i32 s2, 0x50
	v_add_u32_e32 v16, s15, v0
	buffer_load_dword v151, v0, s[28:31], 0 offen
	buffer_load_dword v150, v16, s[28:31], 0 offen
	v_mad_u32_u24 v0, v5, s2, v2
	v_and_b32_e32 v2, 0xf0, v9
	v_sub_u32_e32 v127, v0, v2
	v_ashrrev_i16_e32 v0, 15, v127
	v_lshrrev_b16_e32 v0, 11, v0
	v_add_u16_e32 v0, v127, v0
	v_and_b32_e32 v0, 0xffffffe0, v0
	v_sub_u16_e32 v0, v127, v0
	v_bfe_i32 v2, v0, 0, 16
	v_ashrrev_i32_e32 v16, 31, v2
	v_add_u16_e32 v19, 32, v0
	v_cmp_gt_i16_e32 vcc, 0, v0
	s_load_dwordx2 s[12:13], s[0:1], 0x40
	v_add_u32_e32 v23, 16, v127
	v_cndmask_b32_e32 v0, v2, v19, vcc
	v_cndmask_b32_e64 v2, v16, 0, vcc
	v_xor_b32_e32 v0, v2, v0
	v_lshrrev_b32_e32 v16, 28, v0
	v_add_u32_e32 v0, v0, v16
	v_ashrrev_i32_e32 v0, 4, v0
	v_xor_b32_e32 v16, v0, v2
	v_add3_u32 v2, v10, v17, v16
	v_ashrrev_i32_e32 v0, 31, v2
	v_xor_b32_e32 v17, v0, v2
	v_ashrrev_i32_e32 v19, 31, v17
	v_lshrrev_b32_e32 v19, 25, v19
	v_add_u32_e32 v17, v17, v19
	v_ashrrev_i32_e32 v19, 31, v127
	v_xor_b32_e32 v20, v19, v127
	v_ashrrev_i32_e32 v22, 31, v20
	v_lshrrev_b32_e32 v22, 27, v22
	v_add_u32_e32 v20, v20, v22
	v_lshrrev_b32_e32 v20, 5, v20
	v_xor_b32_e32 v19, v20, v19
	v_ashrrev_i32_e32 v17, 7, v17
	v_lshlrev_b32_e32 v19, 5, v19
	v_xad_u32 v20, v17, v0, v19
	v_add_u32_e32 v17, 2, v2
	v_sub_u32_e32 v22, -3, v2
	v_cmp_gt_i32_e32 vcc, -2, v2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s13, s18
	s_mul_hi_u32 s3, s12, s18
	v_cndmask_b32_e32 v17, v17, v22, vcc
	v_ashrrev_i32_e32 v22, 31, v17
	v_lshrrev_b32_e32 v22, 25, v22
	s_add_i32 s3, s3, s2
	s_mul_i32 s2, s12, s18
	v_add_u32_e32 v17, v17, v22
	s_add_u32 s28, s8, s2
	v_ashrrev_i32_e32 v17, 7, v17
	v_cndmask_b32_e64 v22, 0, -1, vcc
	s_addc_u32 s2, s9, s3
	s_and_b32 s3, s12, 0x3fff
	v_xor_b32_e32 v17, v17, v22
	s_lshl_b32 s3, s3, 16
	s_and_b32 s2, s2, 0xffff
	v_add_u32_e32 v19, v17, v19
	v_lshlrev_b32_e32 v17, 7, v17
	v_and_b32_e32 v0, 0x7f, v2
	s_or_b32 s2, s3, s2
	v_sub_u32_e32 v112, v2, v17
	s_or_b32 s29, s2, 2.0
	v_mad_u64_u32 v[28:29], s[2:3], v20, s12, v[0:1]
	v_mad_u64_u32 v[30:31], s[2:3], v19, s12, v[112:113]
	s_movk_i32 s2, 0xff87
	s_nop 0
	v_mad_i32_i24 v1, v5, s2, v1
	s_movk_i32 s2, 0xffd0
	v_sub_u32_e32 v25, 0xffef, v127
	v_cmp_gt_i32_e32 vcc, -16, v127
	v_add_u32_e32 v22, 48, v127
	v_sub_u32_e32 v24, 0xffcf, v127
	v_cndmask_b32_e32 v23, v23, v25, vcc
	v_cmp_gt_i32_e64 s[2:3], s2, v127
	v_mad_i32_i24 v2, v21, s6, v10
	s_mov_b32 s6, 0x5040100
	v_cndmask_b32_e64 v22, v22, v24, s[2:3]
	v_ashrrev_i16_e32 v24, 15, v23
	v_lshrrev_b16_e32 v24, 11, v24
	v_add_u16_e32 v23, v23, v24
	v_ashrrev_i16_e32 v24, 15, v22
	v_lshrrev_b16_e32 v24, 11, v24
	v_add_u16_e32 v22, v22, v24
	v_ashrrev_i16_e32 v23, 5, v23
	v_ashrrev_i16_e32 v22, 5, v22
	v_perm_b32 v22, v22, v23, s6
	v_cndmask_b32_e64 v23, 0, -1, vcc
	v_cndmask_b32_e64 v24, 0, -1, s[2:3]
	v_perm_b32 v23, v24, v23, s6
	v_xor_b32_e32 v23, v22, v23
	v_mov_b32_e32 v22, -2
	v_mul_i32_i24_sdwa v26, sext(v23), v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_add3_u32 v25, v2, v1, v26
	v_add_u32_e32 v1, 1, v25
	v_sub_u32_e32 v2, -2, v25
	v_cmp_gt_i32_e32 vcc, -1, v25
	v_mov_b32_e32 v34, 5
	v_lshlrev_b32_sdwa v24, v34, sext(v23) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_cndmask_b32_e32 v1, v1, v2, vcc
	v_ashrrev_i32_e32 v2, 31, v1
	v_lshrrev_b32_e32 v2, 25, v2
	v_add_u32_e32 v1, v1, v2
	v_cndmask_b32_e64 v2, 0, -1, vcc
	v_lshlrev_b32_sdwa v32, v34, sext(v23) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_add_u32_e32 v23, 3, v25
	v_sub_u32_e32 v33, -4, v25
	v_cmp_gt_i32_e32 vcc, -3, v25
	v_ashrrev_i32_e32 v1, 7, v1
	v_xor_b32_e32 v31, v1, v2
	v_cndmask_b32_e32 v23, v23, v33, vcc
	v_ashrrev_i32_e32 v33, 31, v23
	v_lshrrev_b32_e32 v33, 25, v33
	v_add_u32_e32 v23, v23, v33
	v_ashrrev_i32_e32 v23, 7, v23
	v_cndmask_b32_e64 v33, 0, -1, vcc
	v_xor_b32_e32 v35, v23, v33
	v_lshlrev_b32_e32 v22, 7, v31
	v_lshlrev_b32_e32 v23, 7, v35
	v_add_u32_e32 v1, v31, v32
	v_add_u32_e32 v2, v31, v24
	v_sub_u32_e32 v124, v25, v22
	v_add_u32_e32 v24, v35, v24
	v_sub_u32_e32 v114, v25, v23
	v_add_u32_e32 v25, v35, v32
	s_lshl_b32 s13, s12, 5
	v_mad_u64_u32 v[116:117], s[2:3], v2, s12, v[124:125]
	v_mad_u64_u32 v[120:121], s[2:3], v24, s12, v[114:115]
	v_mad_u64_u32 v[118:119], s[2:3], v1, s12, v[124:125]
	v_mad_u64_u32 v[32:33], s[2:3], v25, s12, v[114:115]
	v_add_u32_e32 v29, s13, v30
	s_movk_i32 s2, 0xffb0
	v_add_u32_e32 v27, s13, v28
	buffer_load_ubyte v166, v28, s[28:31], 0 offen
	buffer_load_ubyte v156, v30, s[28:31], 0 offen offset:2
	buffer_load_ubyte v167, v27, s[28:31], 0 offen
	buffer_load_ubyte v155, v29, s[28:31], 0 offen offset:2
	buffer_load_ubyte v168, v116, s[28:31], 0 offen offset:1
	buffer_load_ubyte v157, v120, s[28:31], 0 offen offset:3
	buffer_load_ubyte v169, v118, s[28:31], 0 offen offset:1
	buffer_load_ubyte v158, v32, s[28:31], 0 offen offset:3
	v_add_u32_e32 v32, s13, v29
	v_add_u32_e32 v28, 0x50, v127
	v_sub_u32_e32 v29, 0xffaf, v127
	v_cmp_gt_i32_e32 vcc, s2, v127
	v_add_u32_e32 v27, s13, v27
	v_add_u32_e32 v159, s13, v27
	v_cndmask_b32_e32 v28, v28, v29, vcc
	v_ashrrev_i16_e32 v29, 15, v28
	v_lshrrev_b16_e32 v29, 11, v29
	v_add_u16_e32 v28, v28, v29
	v_ashrrev_i16_e32 v28, 5, v28
	v_cndmask_b32_e64 v29, 0, -1, vcc
	v_xor_b32_e32 v28, v28, v29
	v_lshlrev_b32_sdwa v29, v34, sext(v28) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_e32 v28, v31, v29
	v_add_u32_e32 v29, v35, v29
	v_mad_u64_u32 v[30:31], s[2:3], v29, s12, v[114:115]
	v_mad_u64_u32 v[122:123], s[2:3], v28, s12, v[124:125]
	v_add_u32_e32 v31, s13, v32
	buffer_load_ubyte v171, v27, s[28:31], 0 offen
	buffer_load_ubyte v160, v32, s[28:31], 0 offen offset:2
	buffer_load_ubyte v172, v122, s[28:31], 0 offen offset:1
	buffer_load_ubyte v162, v30, s[28:31], 0 offen offset:3
	buffer_load_ubyte v170, v159, s[28:31], 0 offen
	buffer_load_ubyte v161, v31, s[28:31], 0 offen offset:2
	v_cmp_eq_u32_e64 s[2:3], 0, v5
	s_mov_b32 s54, 0
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	s_movk_i32 s22, 0x3800
	v_mul_i32_i24_e32 v27, 0xffffffc0, v21
	s_mov_b32 s55, -2
	s_and_b64 vcc, exec, s[2:3]
	s_waitcnt vmcnt(0)
	s_barrier
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	v_and_b32_e32 v30, 7, v9
	v_bitop3_b32 v31, v115, v9, 7 bitop3:0x78
	v_lshlrev_b32_e32 v9, 7, v9
	v_lshlrev_b32_e32 v21, 11, v21
	v_sub_u32_e32 v9, v9, v21
	v_lshlrev_b32_e32 v21, 4, v31
	v_mul_lo_u32 v31, v5, s22
	v_lshl_add_u32 v8, v8, 13, v9
	v_add_u32_e32 v9, v9, v31
	v_or_b32_e32 v117, v8, v21
	v_or_b32_e32 v121, v9, v21
	v_bitop3_b32 v21, v115, v30, 4 bitop3:0x36
	v_lshlrev_b32_e32 v21, 4, v21
	v_or_b32_e32 v119, v21, v8
	v_or_b32_e32 v123, v21, v9
	v_add_u32_e32 v9, 2, v24
	v_add_u32_e32 v113, 2, v1
	v_add_u32_e32 v8, v26, v27
	v_add_u32_e32 v1, 4, v1
	v_mad_u64_u32 v[30:31], s[22:23], s12, v1, v[8:9]
	v_mul_lo_u32 v5, v5, 7
	v_add3_u32 v1, v30, v10, v5
	v_sub_u32_e32 v173, v1, v22
	v_add_u32_e32 v1, 4, v2
	v_mad_u64_u32 v[30:31], s[22:23], s12, v1, v[8:9]
	v_add3_u32 v1, v30, v10, v5
	v_sub_u32_e32 v174, v1, v22
	v_add_u32_e32 v1, v125, v18
	v_add_u32_e32 v126, 2, v2
	v_add_u32_e32 v2, 36, v1
	v_mul_lo_u32 v2, s14, v2
	v_add3_u32 v2, v27, v2, v10
	v_sub_u32_e32 v175, v2, v15
	v_add_u32_e32 v2, 4, v1
	v_mul_lo_u32 v2, s14, v2
	v_add3_u32 v2, v27, v2, v10
	s_mul_i32 s17, s17, 0x70000
	v_sub_u32_e32 v176, v2, v15
	v_add_u32_e32 v2, s17, v11
	v_add3_u32 v2, v2, v12, v14
	v_lshl_add_u32 v2, v6, 4, v2
	v_lshlrev_b32_e32 v6, 4, v7
	v_sub_u32_e32 v2, v2, v6
	v_sub_u32_e32 v177, v2, v13
	v_add_u32_e32 v2, s16, v3
	v_add_u32_e32 v2, v2, v4
	v_lshl_or_b32 v2, v2, 11, v144
	v_add_u32_e32 v178, 0x60100, v2
	v_add_u32_e32 v2, 0x62, v20
	v_mad_u64_u32 v[128:129], s[22:23], s12, v2, v[0:1]
	v_add_u32_e32 v2, 0x42, v20
	v_mad_u64_u32 v[130:131], s[22:23], s12, v2, v[0:1]
	v_add_u32_e32 v2, 34, v20
	v_mad_u64_u32 v[132:133], s[22:23], s12, v2, v[0:1]
	v_add_u32_e32 v2, 0x64, v20
	v_mad_u64_u32 v[136:137], s[22:23], s12, v2, v[0:1]
	v_add_u32_e32 v2, 0x44, v20
	v_mad_u64_u32 v[138:139], s[22:23], s12, v2, v[0:1]
	v_add_u32_e32 v2, 36, v20
	v_add_u32_e32 v32, 2, v20
	v_mad_u64_u32 v[140:141], s[22:23], s12, v2, v[0:1]
	v_add_u32_e32 v2, 4, v20
	v_mad_u64_u32 v[134:135], s[22:23], s12, v32, v[0:1]
	v_mad_u64_u32 v[142:143], s[22:23], s12, v2, v[0:1]
	v_add_u32_e32 v0, 34, v1
	v_mul_lo_u32 v0, s14, v0
	v_add3_u32 v0, v27, v0, v10
	v_sub_u32_e32 v129, v0, v15
	v_add_u32_e32 v0, 2, v1
	v_mul_lo_u32 v0, s14, v0
	v_add3_u32 v0, v27, v0, v10
	v_sub_u32_e32 v131, v0, v15
	v_add_u32_e32 v0, 4, v29
	v_mad_u64_u32 v[0:1], s[22:23], s12, v0, v[8:9]
	v_add_u32_e32 v163, 2, v29
	v_add3_u32 v0, v0, v10, v5
	v_sub_u32_e32 v133, v0, v23
	v_mad_u64_u32 v[0:1], s[22:23], s12, v163, v[8:9]
	v_add3_u32 v0, v0, v10, v5
	v_sub_u32_e32 v135, v0, v23
	v_add_u32_e32 v0, 4, v28
	v_mad_u64_u32 v[0:1], s[22:23], s12, v0, v[8:9]
	v_add_u32_e32 v21, 2, v28
	v_add3_u32 v0, v0, v10, v5
	v_sub_u32_e32 v137, v0, v22
	v_mad_u64_u32 v[0:1], s[22:23], s12, v21, v[8:9]
	v_add3_u32 v0, v0, v10, v5
	v_sub_u32_e32 v139, v0, v22
	v_add_u32_e32 v0, 4, v25
	v_mad_u64_u32 v[0:1], s[22:23], s12, v0, v[8:9]
	v_add_u32_e32 v164, 2, v25
	v_add3_u32 v0, v0, v10, v5
	v_sub_u32_e32 v141, v0, v23
	v_mad_u64_u32 v[0:1], s[22:23], s12, v164, v[8:9]
	v_add3_u32 v0, v0, v10, v5
	v_sub_u32_e32 v143, v0, v23
	v_add_u32_e32 v0, 4, v24
	v_mad_u64_u32 v[0:1], s[22:23], s12, v0, v[8:9]
	v_add3_u32 v0, v0, v10, v5
	v_sub_u32_e32 v179, v0, v23
	v_mad_u64_u32 v[0:1], s[22:23], s12, v9, v[8:9]
	v_add3_u32 v0, v0, v10, v5
	v_add_u32_e32 v1, 0x64, v19
	v_sub_u32_e32 v180, v0, v23
	v_add_u32_e32 v0, v27, v16
	v_mul_lo_u32 v1, s12, v1
	v_add3_u32 v1, v0, v1, v10
	v_sub_u32_e32 v181, v1, v17
	v_add_u32_e32 v1, 0x44, v19
	v_mul_lo_u32 v1, s12, v1
	v_add3_u32 v1, v0, v1, v10
	v_sub_u32_e32 v182, v1, v17
	v_add_u32_e32 v1, 36, v19
	v_mul_lo_u32 v1, s12, v1
	v_add3_u32 v1, v0, v1, v10
	v_sub_u32_e32 v183, v1, v17
	v_add_u32_e32 v1, 4, v19
	v_mul_lo_u32 v1, s12, v1
	v_add3_u32 v1, v0, v1, v10
	v_sub_u32_e32 v184, v1, v17
	v_add_u32_e32 v1, 0x62, v19
	v_mul_lo_u32 v1, s12, v1
	v_add3_u32 v1, v0, v1, v10
	v_sub_u32_e32 v185, v1, v17
	v_add_u32_e32 v1, 0x42, v19
	v_mul_lo_u32 v1, s12, v1
	v_add3_u32 v1, v0, v1, v10
	v_sub_u32_e32 v186, v1, v17
	v_add_u32_e32 v1, 34, v19
	v_mul_lo_u32 v1, s12, v1
	s_load_dwordx2 s[8:9], s[0:1], 0x48
	v_add_u32_e32 v165, 2, v19
	v_add3_u32 v1, v0, v1, v10
	v_sub_u32_e32 v187, v1, v17
	v_mul_lo_u32 v1, s12, v165
	v_add3_u32 v0, v0, v1, v10
	v_sub_u32_e32 v188, v0, v17
	v_mov_b32_e32 v0, 0
	s_mov_b32 s0, s12
	s_lshl_b32 s1, s12, 2
	s_lshl_b32 s56, s14, 2
	s_add_i32 s69, s19, 0x8000
	s_mov_b32 s26, s6
	s_mov_b32 s27, s7
	s_add_i32 s68, s33, 0x8000
	s_add_i32 s67, s34, 0x8000
	s_add_i32 s66, s35, 0x8000
	s_add_i32 s63, s63, 0x17000
	s_mov_b32 s22, s6
	s_mov_b32 s23, s7
	s_add_i32 s65, s65, 0x17000
	s_add_i32 s64, s64, 0x17000
	s_add_i32 s62, s62, 0x17000
	s_add_i32 s61, s61, 0x17000
	s_add_i32 s60, s60, 0x17000
	s_add_i32 s59, s59, 0x17000
	s_add_i32 s58, s58, 0x17000
	s_add_i32 s57, s57, 0x17000
	s_add_i32 s17, s70, 0x17000
	v_add_u32_e32 v189, 0x10000, v121
	v_add_u32_e32 v190, 0x10000, v123
	v_add_u32_e32 v191, 0x17000, v121
	v_add_u32_e32 v192, 0x17000, v123
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
	v_mov_b32_e32 v80, v0
	v_mov_b32_e32 v81, v0
	v_mov_b32_e32 v82, v0
	v_mov_b32_e32 v83, v0
	v_mov_b32_e32 v84, v0
	v_mov_b32_e32 v85, v0
	v_mov_b32_e32 v86, v0
	v_mov_b32_e32 v87, v0
	v_mov_b32_e32 v88, v0
	v_mov_b32_e32 v89, v0
	v_mov_b32_e32 v90, v0
	v_mov_b32_e32 v91, v0
	v_mov_b32_e32 v92, v0
	v_mov_b32_e32 v93, v0
	v_mov_b32_e32 v94, v0
	v_mov_b32_e32 v95, v0
	v_mov_b32_e32 v96, v0
	v_mov_b32_e32 v97, v0
	v_mov_b32_e32 v98, v0
	v_mov_b32_e32 v99, v0
	v_mov_b32_e32 v100, v0
	v_mov_b32_e32 v101, v0
	v_mov_b32_e32 v102, v0
	v_mov_b32_e32 v103, v0
	v_mov_b32_e32 v104, v0
	v_mov_b32_e32 v105, v0
	v_mov_b32_e32 v106, v0
	v_mov_b32_e32 v107, v0
	v_mov_b32_e32 v108, v0
	v_mov_b32_e32 v109, v0
	v_mov_b32_e32 v110, v0
	v_mov_b32_e32 v111, v0
.LBB0_3:
	s_mov_b32 m0, s69
	v_add_u32_e32 v193, 0xfff9ff80, v178
	s_barrier
	buffer_load_dwordx4 v193, s[24:27], 0 offen lds
	v_add_u32_e32 v193, 0xfffbff80, v178
	s_mov_b32 m0, s68
	s_add_i32 s70, s46, 0x17000
	buffer_load_dwordx4 v193, s[24:27], 0 offen lds
	v_add_u32_e32 v193, 0xfffdff80, v178
	s_mov_b32 m0, s67
	s_add_i32 s71, s48, 0x17000
	buffer_load_dwordx4 v193, s[24:27], 0 offen lds
	v_add_u32_e32 v193, 0xffffff80, v178
	s_mov_b32 m0, s66
	s_add_i32 s72, s50, 0x17000
	buffer_load_dwordx4 v193, s[24:27], 0 offen lds
	v_add_u32_e32 v193, v177, v152
	v_add_u32_e32 v194, 0x800, v193
	s_mov_b32 m0, s63
	s_add_i32 s73, s52, 0x17000
	buffer_load_dword v194, s[20:23], 0 offen lds
	v_add_u32_e32 v194, 0x8800, v193
	s_mov_b32 m0, s65
	s_nop 0
	buffer_load_dword v194, s[20:23], 0 offen lds
	v_add_u32_e32 v194, 0x10800, v193
	s_mov_b32 m0, s64
	s_nop 0
	buffer_load_dword v194, s[20:23], 0 offen lds
	v_add_u32_e32 v194, 0x18800, v193
	s_mov_b32 m0, s62
	s_nop 0
	buffer_load_dword v194, s[20:23], 0 offen lds
	v_add_u32_e32 v194, 0x20800, v193
	s_mov_b32 m0, s61
	s_nop 0
	buffer_load_dword v194, s[20:23], 0 offen lds
	v_add_u32_e32 v194, 0x28800, v193
	s_mov_b32 m0, s60
	s_nop 0
	buffer_load_dword v194, s[20:23], 0 offen lds
	v_add_u32_e32 v194, 0x30800, v193
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dword v194, s[20:23], 0 offen lds
	v_add_u32_e32 v194, 0x38800, v193
	s_mov_b32 m0, s58
	s_nop 0
	buffer_load_dword v194, s[20:23], 0 offen lds
	v_add_u32_e32 v194, 0x40800, v193
	s_mov_b32 m0, s57
	s_nop 0
	buffer_load_dword v194, s[20:23], 0 offen lds
	v_add_u32_e32 v194, 0x48800, v193
	s_mov_b32 m0, s17
	s_nop 0
	buffer_load_dword v194, s[20:23], 0 offen lds
	v_add_u32_e32 v194, 0x50800, v193
	s_mov_b32 m0, s70
	s_nop 0
	buffer_load_dword v194, s[20:23], 0 offen lds
	v_add_u32_e32 v194, 0x58800, v193
	s_mov_b32 m0, s71
	s_nop 0
	buffer_load_dword v194, s[20:23], 0 offen lds
	v_add_u32_e32 v194, 0x60800, v193
	s_mov_b32 m0, s72
	s_nop 0
	buffer_load_dword v194, s[20:23], 0 offen lds
	v_add_u32_e32 v194, 0x68800, v193
	s_mov_b32 m0, s73
	s_nop 0
	buffer_load_dword v194, s[20:23], 0 offen lds
	v_add_u32_e32 v194, v131, v152
	v_add_u32_e32 v195, v129, v152
	buffer_load_dword v198, v194, s[4:7], 0 offen
	buffer_load_dword v199, v195, s[4:7], 0 offen
	v_add_u32_e32 v194, s54, v126
	s_mov_b32 s30, s6
	s_mov_b32 s31, s7
	v_add_u32_e32 v200, v188, v152
	v_add_u32_e32 v201, v180, v152
	v_add_u32_e32 v202, v187, v152
	v_add_u32_e32 v196, s54, v113
	v_mad_u64_u32 v[194:195], s[74:75], v194, s12, v[124:125]
	v_mad_u64_u32 v[196:197], s[74:75], v196, s0, v[124:125]
	buffer_load_ubyte v205, v134, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v200, v200, s[28:31], 0 offen offset:2
	s_nop 0
	buffer_load_ubyte v201, v201, s[28:31], 0 offen offset:3
	s_nop 0
	buffer_load_ubyte v202, v202, s[28:31], 0 offen offset:2
	s_nop 0
	buffer_load_ubyte v194, v194, s[28:31], 0 offen offset:1
	s_nop 0
	buffer_load_ubyte v206, v132, s[28:31], 0 offen
	buffer_load_ubyte v207, v130, s[28:31], 0 offen
	buffer_load_ubyte v208, v128, s[28:31], 0 offen
	v_add_u32_e32 v195, v143, v152
	v_add_u32_e32 v197, v186, v152
	v_add_u32_e32 v203, v139, v152
	v_add_u32_e32 v204, v135, v152
	v_add_u32_e32 v209, v185, v152
	buffer_load_ubyte v196, v196, s[28:31], 0 offen offset:1
	s_nop 0
	buffer_load_ubyte v195, v195, s[28:31], 0 offen offset:3
	s_nop 0
	buffer_load_ubyte v197, v197, s[28:31], 0 offen offset:2
	s_nop 0
	buffer_load_ubyte v203, v203, s[28:31], 0 offen offset:1
	s_nop 0
	buffer_load_ubyte v204, v204, s[28:31], 0 offen offset:3
	s_nop 0
	buffer_load_ubyte v209, v209, s[28:31], 0 offen offset:2
	ds_read_b128 v[210:213], v117
	ds_read_b128 v[214:217], v117 offset:2048
	ds_read_b128 v[218:221], v117 offset:4096
	ds_read_b128 v[222:225], v117 offset:6144
	ds_read_b128 v[226:229], v189
	ds_read_b128 v[230:233], v189 offset:2048
	ds_read_b128 v[234:237], v189 offset:4096
	ds_read_b128 v[238:241], v189 offset:6144
	ds_read_b128 v[242:245], v189 offset:8192
	ds_read_b128 v[246:249], v189 offset:10240
	ds_read_b128 v[250:253], v189 offset:12288
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(47)
	v_and_b32_e32 v166, 0xff, v166
	s_waitcnt vmcnt(45)
	v_and_b32_e32 v168, 0xff, v168
	s_waitcnt vmcnt(42)
	v_and_b32_e32 v167, 0xff, v167
	s_waitcnt vmcnt(39)
	v_and_b32_e32 v169, 0xff, v169
	v_and_b32_e32 v171, 0xff, v171
	s_waitcnt vmcnt(36)
	v_and_b32_e32 v172, 0xff, v172
	v_and_b32_e32 v170, 0xff, v170
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[210:213], v[226:229], v[0:3], v151, v166 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[210:213], v[230:233], v[4:7], v151, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[210:213], v[234:237], v[8:11], v151, v167 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[210:213], v[238:241], v[12:15], v151, v169 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[210:213], v[242:245], v[16:19], v151, v171 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[210:213], v[246:249], v[20:23], v151, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[210:213], v[250:253], v[24:27], v151, v170 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[214:217], v[226:229], v[28:31], v151, v166 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[214:217], v[230:233], v[32:35], v151, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[214:217], v[234:237], v[36:39], v151, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[214:217], v[238:241], v[40:43], v151, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[214:217], v[242:245], v[44:47], v151, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[214:217], v[246:249], v[48:51], v151, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[214:217], v[250:253], v[52:55], v151, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[218:221], v[226:229], v[56:59], v150, v166 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[218:221], v[230:233], v[60:63], v150, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[218:221], v[234:237], v[64:67], v150, v167 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[218:221], v[238:241], v[68:71], v150, v169 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[218:221], v[242:245], v[72:75], v150, v171 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[218:221], v[246:249], v[76:79], v150, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[218:221], v[250:253], v[80:83], v150, v170 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[222:225], v[226:229], v[84:87], v150, v166 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[222:225], v[230:233], v[88:91], v150, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[222:225], v[234:237], v[92:95], v150, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[222:225], v[238:241], v[96:99], v150, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[222:225], v[242:245], v[100:103], v150, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[222:225], v[246:249], v[104:107], v150, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[222:225], v[250:253], v[108:111], v150, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	ds_read_b128 v[166:169], v119
	ds_read_b128 v[210:213], v119 offset:2048
	ds_read_b128 v[214:217], v119 offset:4096
	ds_read_b128 v[218:221], v119 offset:6144
	ds_read_b128 v[222:225], v190
	ds_read_b128 v[226:229], v190 offset:2048
	ds_read_b128 v[230:233], v190 offset:4096
	ds_read_b128 v[234:237], v190 offset:6144
	ds_read_b128 v[238:241], v190 offset:8192
	ds_read_b128 v[242:245], v190 offset:10240
	ds_read_b128 v[246:249], v190 offset:12288
	s_waitcnt vmcnt(16)
	s_barrier
	s_setprio 1
	v_and_b32_e32 v156, 0xff, v156
	v_and_b32_e32 v157, 0xff, v157
	v_and_b32_e32 v155, 0xff, v155
	v_and_b32_e32 v158, 0xff, v158
	v_and_b32_e32 v160, 0xff, v160
	v_and_b32_e32 v162, 0xff, v162
	v_and_b32_e32 v161, 0xff, v161
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[166:169], v[222:225], v[0:3], v151, v156 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[166:169], v[226:229], v[4:7], v151, v157 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[166:169], v[230:233], v[8:11], v151, v155 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[166:169], v[234:237], v[12:15], v151, v158 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[166:169], v[238:241], v[16:19], v151, v160 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[166:169], v[242:245], v[20:23], v151, v162 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[166:169], v[246:249], v[24:27], v151, v161 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[210:213], v[222:225], v[28:31], v151, v156 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[210:213], v[226:229], v[32:35], v151, v157 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[210:213], v[230:233], v[36:39], v151, v155 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[210:213], v[234:237], v[40:43], v151, v158 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[210:213], v[238:241], v[44:47], v151, v160 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[210:213], v[242:245], v[48:51], v151, v162 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[210:213], v[246:249], v[52:55], v151, v161 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[214:217], v[222:225], v[56:59], v150, v156 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[214:217], v[226:229], v[60:63], v150, v157 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[214:217], v[230:233], v[64:67], v150, v155 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[214:217], v[234:237], v[68:71], v150, v158 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[214:217], v[238:241], v[72:75], v150, v160 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[214:217], v[242:245], v[76:79], v150, v162 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[214:217], v[246:249], v[80:83], v150, v161 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[218:221], v[222:225], v[84:87], v150, v156 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[218:221], v[226:229], v[88:91], v150, v157 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[218:221], v[230:233], v[92:95], v150, v155 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[218:221], v[234:237], v[96:99], v150, v158 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[218:221], v[238:241], v[100:103], v150, v160 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[218:221], v[242:245], v[104:107], v150, v162 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[218:221], v[246:249], v[108:111], v150, v161 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_mov_b32 m0, s19
	v_add_u32_e32 v150, 0xfffa0000, v178
	s_barrier
	buffer_load_dwordx4 v150, s[24:27], 0 offen lds
	v_add_u32_e32 v150, 0xfffc0000, v178
	s_mov_b32 m0, s33
	s_nop 0
	buffer_load_dwordx4 v150, s[24:27], 0 offen lds
	v_add_u32_e32 v150, 0xfffe0000, v178
	s_mov_b32 m0, s34
	s_nop 0
	buffer_load_dwordx4 v150, s[24:27], 0 offen lds
	s_mov_b32 m0, s35
	v_add_u32_e32 v150, 0x1000, v193
	buffer_load_dwordx4 v178, s[24:27], 0 offen lds
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dword v150, s[20:23], 0 offen lds
	v_add_u32_e32 v150, 0x9000, v193
	s_mov_b32 m0, s37
	s_nop 0
	buffer_load_dword v150, s[20:23], 0 offen lds
	v_add_u32_e32 v150, 0x11000, v193
	s_mov_b32 m0, s38
	s_nop 0
	buffer_load_dword v150, s[20:23], 0 offen lds
	v_add_u32_e32 v150, 0x19000, v193
	s_mov_b32 m0, s39
	s_nop 0
	buffer_load_dword v150, s[20:23], 0 offen lds
	v_add_u32_e32 v150, 0x21000, v193
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dword v150, s[20:23], 0 offen lds
	v_add_u32_e32 v150, 0x29000, v193
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dword v150, s[20:23], 0 offen lds
	v_add_u32_e32 v150, 0x31000, v193
	s_mov_b32 m0, s42
	s_nop 0
	buffer_load_dword v150, s[20:23], 0 offen lds
	v_add_u32_e32 v150, 0x39000, v193
	s_mov_b32 m0, s43
	s_nop 0
	buffer_load_dword v150, s[20:23], 0 offen lds
	v_add_u32_e32 v150, 0x41000, v193
	s_mov_b32 m0, s44
	s_nop 0
	buffer_load_dword v150, s[20:23], 0 offen lds
	v_add_u32_e32 v150, 0x49000, v193
	s_mov_b32 m0, s45
	s_nop 0
	buffer_load_dword v150, s[20:23], 0 offen lds
	v_add_u32_e32 v150, 0x51000, v193
	s_mov_b32 m0, s47
	s_nop 0
	buffer_load_dword v150, s[20:23], 0 offen lds
	v_add_u32_e32 v150, 0x59000, v193
	s_mov_b32 m0, s49
	s_nop 0
	buffer_load_dword v150, s[20:23], 0 offen lds
	v_add_u32_e32 v150, 0x61000, v193
	s_mov_b32 m0, s51
	s_nop 0
	buffer_load_dword v150, s[20:23], 0 offen lds
	v_add_u32_e32 v150, 0x69000, v193
	s_mov_b32 m0, s53
	s_nop 0
	buffer_load_dword v150, s[20:23], 0 offen lds
	v_add_u32_e32 v150, v176, v152
	v_add_u32_e32 v155, v175, v152
	buffer_load_dword v151, v150, s[4:7], 0 offen
	s_nop 0
	buffer_load_dword v150, v155, s[4:7], 0 offen
	v_add_u32_e32 v155, v184, v152
	v_add_u32_e32 v157, v174, v152
	v_add_u32_e32 v158, v179, v152
	v_add_u32_e32 v160, v183, v152
	buffer_load_ubyte v166, v142, s[28:31], 0 offen
	buffer_load_ubyte v156, v155, s[28:31], 0 offen offset:2
	buffer_load_ubyte v168, v157, s[28:31], 0 offen offset:1
	s_nop 0
	buffer_load_ubyte v157, v158, s[28:31], 0 offen offset:3
	buffer_load_ubyte v155, v160, s[28:31], 0 offen offset:2
	buffer_load_ubyte v167, v140, s[28:31], 0 offen
	buffer_load_ubyte v171, v138, s[28:31], 0 offen
	buffer_load_ubyte v170, v136, s[28:31], 0 offen
	v_add_u32_e32 v161, v173, v152
	v_add_u32_e32 v162, v141, v152
	v_add_u32_e32 v172, v182, v152
	v_add_u32_e32 v193, v137, v152
	v_add_u32_e32 v210, v133, v152
	v_add_u32_e32 v211, v181, v152
	buffer_load_ubyte v169, v161, s[28:31], 0 offen offset:1
	buffer_load_ubyte v158, v162, s[28:31], 0 offen offset:3
	buffer_load_ubyte v160, v172, s[28:31], 0 offen offset:2
	s_nop 0
	buffer_load_ubyte v172, v193, s[28:31], 0 offen offset:1
	buffer_load_ubyte v162, v210, s[28:31], 0 offen offset:3
	buffer_load_ubyte v161, v211, s[28:31], 0 offen offset:2
	ds_read_b128 v[210:213], v117 offset:32768
	ds_read_b128 v[214:217], v117 offset:34816
	ds_read_b128 v[218:221], v117 offset:36864
	ds_read_b128 v[222:225], v117 offset:38912
	ds_read_b128 v[226:229], v191
	ds_read_b128 v[230:233], v191 offset:2048
	ds_read_b128 v[234:237], v191 offset:4096
	ds_read_b128 v[238:241], v191 offset:6144
	ds_read_b128 v[242:245], v191 offset:8192
	ds_read_b128 v[246:249], v191 offset:10240
	ds_read_b128 v[250:253], v191 offset:12288
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(47) lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[210:213], v[226:229], v[0:3], v198, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(43) lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[210:213], v[230:233], v[4:7], v198, v194 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(42) lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[210:213], v[234:237], v[8:11], v198, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(39) lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[210:213], v[238:241], v[12:15], v198, v196 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[210:213], v[242:245], v[16:19], v198, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(36) lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[210:213], v[246:249], v[20:23], v198, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[210:213], v[250:253], v[24:27], v198, v208 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[214:217], v[226:229], v[28:31], v198, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[214:217], v[230:233], v[32:35], v198, v194 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[214:217], v[234:237], v[36:39], v198, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[214:217], v[238:241], v[40:43], v198, v196 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[214:217], v[242:245], v[44:47], v198, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[214:217], v[246:249], v[48:51], v198, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[214:217], v[250:253], v[52:55], v198, v208 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[218:221], v[226:229], v[56:59], v199, v205 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[218:221], v[230:233], v[60:63], v199, v194 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[218:221], v[234:237], v[64:67], v199, v206 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[218:221], v[238:241], v[68:71], v199, v196 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[218:221], v[242:245], v[72:75], v199, v207 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[218:221], v[246:249], v[76:79], v199, v203 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[218:221], v[250:253], v[80:83], v199, v208 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[222:225], v[226:229], v[84:87], v199, v205 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[222:225], v[230:233], v[88:91], v199, v194 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[222:225], v[234:237], v[92:95], v199, v206 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[222:225], v[238:241], v[96:99], v199, v196 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[222:225], v[242:245], v[100:103], v199, v207 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[222:225], v[246:249], v[104:107], v199, v203 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[222:225], v[250:253], v[108:111], v199, v208 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	ds_read_b128 v[210:213], v119 offset:32768
	ds_read_b128 v[214:217], v119 offset:34816
	ds_read_b128 v[218:221], v119 offset:36864
	ds_read_b128 v[222:225], v119 offset:38912
	ds_read_b128 v[226:229], v192
	ds_read_b128 v[230:233], v192 offset:2048
	ds_read_b128 v[234:237], v192 offset:4096
	ds_read_b128 v[238:241], v192 offset:6144
	ds_read_b128 v[242:245], v192 offset:8192
	ds_read_b128 v[246:249], v192 offset:10240
	ds_read_b128 v[250:253], v192 offset:12288
	s_waitcnt vmcnt(16)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[210:213], v[226:229], v[0:3], v198, v200 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[210:213], v[230:233], v[4:7], v198, v201 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[210:213], v[234:237], v[8:11], v198, v202 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[210:213], v[238:241], v[12:15], v198, v195 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[210:213], v[242:245], v[16:19], v198, v197 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[210:213], v[246:249], v[20:23], v198, v204 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[210:213], v[250:253], v[24:27], v198, v209 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[214:217], v[226:229], v[28:31], v198, v200 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[214:217], v[230:233], v[32:35], v198, v201 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[214:217], v[234:237], v[36:39], v198, v202 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[214:217], v[238:241], v[40:43], v198, v195 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[214:217], v[242:245], v[44:47], v198, v197 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[214:217], v[246:249], v[48:51], v198, v204 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[214:217], v[250:253], v[52:55], v198, v209 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[218:221], v[226:229], v[56:59], v199, v200 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[218:221], v[230:233], v[60:63], v199, v201 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[218:221], v[234:237], v[64:67], v199, v202 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[218:221], v[238:241], v[68:71], v199, v195 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[218:221], v[242:245], v[72:75], v199, v197 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[218:221], v[246:249], v[76:79], v199, v204 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[218:221], v[250:253], v[80:83], v199, v209 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[222:225], v[226:229], v[84:87], v199, v200 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[222:225], v[230:233], v[88:91], v199, v201 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[222:225], v[234:237], v[92:95], v199, v202 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[222:225], v[238:241], v[96:99], v199, v195 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[222:225], v[242:245], v[100:103], v199, v197 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[222:225], v[246:249], v[104:107], v199, v204 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[222:225], v[250:253], v[108:111], v199, v209 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_add_i32 s55, s55, 2
	s_add_i32 s54, s54, 4
	v_add_u32_e32 v173, s1, v173
	v_add_u32_e32 v174, s1, v174
	v_add_u32_e32 v175, s56, v175
	v_add_u32_e32 v176, s56, v176
	v_add_u32_e32 v177, 0x1000, v177
	v_add_u32_e32 v178, 0x100, v178
	v_add_u32_e32 v128, s1, v128
	v_add_u32_e32 v130, s1, v130
	v_add_u32_e32 v132, s1, v132
	v_add_u32_e32 v134, s1, v134
	v_add_u32_e32 v136, s1, v136
	v_add_u32_e32 v138, s1, v138
	v_add_u32_e32 v140, s1, v140
	v_add_u32_e32 v142, s1, v142
	v_add_u32_e32 v129, s56, v129
	v_add_u32_e32 v131, s56, v131
	v_add_u32_e32 v133, s1, v133
	v_add_u32_e32 v135, s1, v135
	v_add_u32_e32 v137, s1, v137
	v_add_u32_e32 v139, s1, v139
	v_add_u32_e32 v141, s1, v141
	v_add_u32_e32 v143, s1, v143
	v_add_u32_e32 v179, s1, v179
	v_add_u32_e32 v180, s1, v180
	v_add_u32_e32 v181, s1, v181
	v_add_u32_e32 v182, s1, v182
	v_add_u32_e32 v183, s1, v183
	v_add_u32_e32 v184, s1, v184
	v_add_u32_e32 v185, s1, v185
	v_add_u32_e32 v186, s1, v186
	v_add_u32_e32 v187, s1, v187
	s_cmp_lt_u32 s55, 12
	v_add_u32_e32 v188, s1, v188
	s_cbranch_scc1 .LBB0_3
	v_add_u32_e32 v113, s15, v154
	s_movk_i32 s0, 0x780
	s_mov_b32 m0, s69
	v_add3_u32 v124, v144, v145, s0
	s_mov_b32 s26, s6
	s_mov_b32 s27, s7
	s_barrier
	buffer_load_dwordx4 v124, s[24:27], 0 offen lds
	v_add3_u32 v124, v144, v146, s0
	s_mov_b32 m0, s68
	s_mov_b32 s22, s6
	buffer_load_dwordx4 v124, s[24:27], 0 offen lds
	v_add3_u32 v124, v144, v147, s0
	s_mov_b32 m0, s67
	s_mov_b32 s23, s7
	buffer_load_dwordx4 v124, s[24:27], 0 offen lds
	v_add3_u32 v124, v144, v148, s0
	s_mov_b32 m0, s66
	s_nop 0
	buffer_load_dwordx4 v124, s[24:27], 0 offen lds
	v_add_u32_e32 v124, 0x7800, v149
	s_mov_b32 m0, s63
	s_nop 0
	buffer_load_dword v124, s[20:23], 0 offen lds
	v_add_u32_e32 v124, 0xf800, v149
	s_mov_b32 m0, s65
	s_nop 0
	buffer_load_dword v124, s[20:23], 0 offen lds
	v_add_u32_e32 v124, 0x17800, v149
	s_mov_b32 m0, s64
	s_nop 0
	buffer_load_dword v124, s[20:23], 0 offen lds
	v_add_u32_e32 v124, 0x1f800, v149
	s_mov_b32 m0, s62
	s_nop 0
	buffer_load_dword v124, s[20:23], 0 offen lds
	v_add_u32_e32 v124, 0x27800, v149
	s_mov_b32 m0, s61
	s_nop 0
	buffer_load_dword v124, s[20:23], 0 offen lds
	v_add_u32_e32 v124, 0x2f800, v149
	s_mov_b32 m0, s60
	s_nop 0
	buffer_load_dword v124, s[20:23], 0 offen lds
	v_add_u32_e32 v124, 0x37800, v149
	s_mov_b32 m0, s59
	s_nop 0
	buffer_load_dword v124, s[20:23], 0 offen lds
	v_add_u32_e32 v124, 0x3f800, v149
	s_mov_b32 m0, s58
	s_nop 0
	buffer_load_dword v124, s[20:23], 0 offen lds
	v_add_u32_e32 v124, 0x47800, v149
	s_mov_b32 m0, s57
	s_nop 0
	buffer_load_dword v124, s[20:23], 0 offen lds
	v_add_u32_e32 v124, 0x4f800, v149
	s_mov_b32 m0, s17
	s_nop 0
	buffer_load_dword v124, s[20:23], 0 offen lds
	v_add_u32_e32 v124, 0x57800, v149
	s_mov_b32 m0, s70
	s_nop 0
	buffer_load_dword v124, s[20:23], 0 offen lds
	v_add_u32_e32 v124, 0x5f800, v149
	s_mov_b32 m0, s71
	s_nop 0
	buffer_load_dword v124, s[20:23], 0 offen lds
	v_add_u32_e32 v124, 0x67800, v149
	s_mov_b32 m0, s72
	s_nop 0
	buffer_load_dword v124, s[20:23], 0 offen lds
	v_add_u32_e32 v124, 0x6f800, v149
	s_mov_b32 m0, s73
	s_nop 0
	buffer_load_dword v124, s[20:23], 0 offen lds
	s_lshl_b32 s0, s14, 1
	v_subrev_u32_e32 v113, s0, v113
	v_add_u32_e32 v113, v153, v113
	v_add_u32_e32 v126, s15, v113
	buffer_load_dword v124, v113, s[4:7], 0 offen
	s_nop 0
	buffer_load_dword v113, v126, s[4:7], 0 offen
	s_mul_i32 s0, s12, 0xffffffbe
	v_add_u32_e32 v128, 28, v165
	s_mul_i32 s4, s12, 30
	v_add_u32_e32 v126, s0, v159
	v_add_u32_e32 v131, s13, v126
	v_add_u32_e32 v122, s4, v122
	ds_read_b128 v[136:139], v117
	ds_read_b128 v[140:143], v117 offset:2048
	ds_read_b128 v[186:189], v117 offset:4096
	ds_read_b128 v[210:213], v117 offset:6144
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[128:129], s[0:1], v128, s12, v[112:113]
	v_add_u32_e32 v112, s4, v116
	v_add_u32_e32 v116, s4, v120
	v_add_u32_e32 v120, s4, v118
	v_add_u32_e32 v118, 28, v164
	v_mad_u64_u32 v[132:133], s[0:1], v118, s12, v[114:115]
	v_add_u32_e32 v134, s13, v128
	buffer_load_ubyte v130, v126, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v126, v128, s[28:31], 0 offen offset:2
	s_nop 0
	buffer_load_ubyte v128, v112, s[28:31], 0 offen offset:1
	buffer_load_ubyte v129, v116, s[28:31], 0 offen offset:3
	s_nop 0
	buffer_load_ubyte v116, v131, s[28:31], 0 offen
	buffer_load_ubyte v118, v134, s[28:31], 0 offen offset:2
	s_nop 0
	buffer_load_ubyte v120, v120, s[28:31], 0 offen offset:1
	s_nop 0
	buffer_load_ubyte v112, v132, s[28:31], 0 offen offset:3
	v_add_u32_e32 v131, s13, v131
	v_add_u32_e32 v132, 28, v163
	v_add_u32_e32 v134, s13, v134
	v_mad_u64_u32 v[132:133], s[0:1], v132, s12, v[114:115]
	v_add_u32_e32 v114, s13, v131
	v_add_u32_e32 v135, s13, v134
	buffer_load_ubyte v133, v131, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v134, v134, s[28:31], 0 offen offset:2
	s_nop 0
	buffer_load_ubyte v122, v122, s[28:31], 0 offen offset:1
	s_nop 0
	buffer_load_ubyte v131, v132, s[28:31], 0 offen offset:3
	s_nop 0
	buffer_load_ubyte v132, v114, s[28:31], 0 offen
	s_nop 0
	buffer_load_ubyte v114, v135, s[28:31], 0 offen offset:2
	v_add_u32_e32 v135, 0x10000, v121
	ds_read_b128 v[190:193], v135
	ds_read_b128 v[194:197], v135 offset:2048
	ds_read_b128 v[198:201], v135 offset:4096
	ds_read_b128 v[202:205], v135 offset:6144
	ds_read_b128 v[206:209], v135 offset:8192
	ds_read_b128 v[214:217], v135 offset:10240
	ds_read_b128 v[218:221], v135 offset:12288
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[136:139], v[190:193], v[0:3], v151, v166 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[136:139], v[194:197], v[4:7], v151, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[136:139], v[198:201], v[8:11], v151, v167 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[136:139], v[202:205], v[12:15], v151, v169 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[136:139], v[206:209], v[16:19], v151, v171 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[136:139], v[214:217], v[20:23], v151, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[136:139], v[218:221], v[24:27], v151, v170 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[140:143], v[190:193], v[28:31], v151, v166 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[140:143], v[194:197], v[32:35], v151, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[140:143], v[198:201], v[36:39], v151, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[140:143], v[202:205], v[40:43], v151, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[140:143], v[206:209], v[44:47], v151, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[140:143], v[214:217], v[48:51], v151, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[140:143], v[218:221], v[52:55], v151, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[136:139], v[186:189], v[190:193], v[56:59], v150, v166 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[140:143], v[186:189], v[194:197], v[60:63], v150, v168 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], v[186:189], v[198:201], v[64:67], v150, v167 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], v[186:189], v[202:205], v[68:71], v150, v169 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], v[186:189], v[206:209], v[72:75], v150, v171 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[182:185], v[186:189], v[214:217], v[76:79], v150, v172 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[186:189], v[186:189], v[218:221], v[80:83], v150, v170 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[190:193], v[210:213], v[190:193], v[84:87], v150, v166 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[194:197], v[210:213], v[194:197], v[88:91], v150, v168 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[164:167], v[210:213], v[198:201], v[92:95], v150, v167 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[198:201], v[210:213], v[202:205], v[96:99], v150, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[202:205], v[210:213], v[206:209], v[100:103], v150, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[206:209], v[210:213], v[214:217], v[104:107], v150, v172 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[168:171], v[210:213], v[218:221], v[108:111], v150, v170 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v56, 0x10000, v123
	ds_read_b128 v[222:225], v56
	ds_read_b128 v[226:229], v56 offset:2048
	ds_read_b128 v[230:233], v56 offset:4096
	ds_read_b128 v[234:237], v56 offset:6144
	ds_read_b128 v[238:241], v56 offset:8192
	ds_read_b128 v[242:245], v56 offset:10240
	ds_read_b128 v[246:249], v56 offset:12288
	ds_read_b128 v[58:61], v119
	ds_read_b128 v[210:213], v119 offset:2048
	ds_read_b128 v[214:217], v119 offset:4096
	ds_read_b128 v[218:221], v119 offset:6144
	s_waitcnt vmcnt(16)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[58:61], v[222:225], v[0:3], v151, v156 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[58:61], v[226:229], v[4:7], v151, v157 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[58:61], v[230:233], v[8:11], v151, v155 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[58:61], v[234:237], v[12:15], v151, v158 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[58:61], v[238:241], v[16:19], v151, v160 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[58:61], v[242:245], v[20:23], v151, v162 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[58:61], v[246:249], v[24:27], v151, v161 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[210:213], v[222:225], v[28:31], v151, v156 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[210:213], v[226:229], v[32:35], v151, v157 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[210:213], v[230:233], v[36:39], v151, v155 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[210:213], v[234:237], v[40:43], v151, v158 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[210:213], v[238:241], v[44:47], v151, v160 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[210:213], v[242:245], v[48:51], v151, v162 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[210:213], v[246:249], v[52:55], v151, v161 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[214:217], v[222:225], v[136:139], v150, v156 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[214:217], v[226:229], v[140:143], v150, v157 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[214:217], v[230:233], v[144:147], v150, v155 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[214:217], v[234:237], v[174:177], v150, v158 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[214:217], v[238:241], v[178:181], v150, v160 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[214:217], v[242:245], v[182:185], v150, v162 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[214:217], v[246:249], v[186:189], v150, v161 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[218:221], v[222:225], v[190:193], v150, v156 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[218:221], v[226:229], v[194:197], v150, v157 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[218:221], v[230:233], v[164:167], v150, v155 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[218:221], v[234:237], v[198:201], v150, v158 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[218:221], v[238:241], v[202:205], v150, v160 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[218:221], v[242:245], v[206:209], v150, v162 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[218:221], v[246:249], v[168:171], v150, v161 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	v_add_u32_e32 v121, 0x17000, v121
	s_barrier
	ds_read_b128 v[168:171], v121
	ds_read_b128 v[172:175], v121 offset:2048
	v_add_u32_e32 v123, 0x17000, v123
	s_waitcnt vmcnt(13)
	v_and_b32_e32 v135, 0xffff, v130
	ds_read_b128 v[176:179], v123
	ds_read_b128 v[136:139], v121 offset:12288
	ds_read_b128 v[180:183], v123 offset:2048
	ds_read_b128 v[184:187], v123 offset:4096
	ds_read_b128 v[188:191], v121 offset:4096
	ds_read_b128 v[192:195], v121 offset:6144
	ds_read_b128 v[196:199], v123 offset:6144
	ds_read_b128 v[200:203], v123 offset:8192
	ds_read_b128 v[204:207], v121 offset:8192
	ds_read_b128 v[148:151], v121 offset:10240
	ds_read_b128 v[144:147], v123 offset:10240
	ds_read_b128 v[140:143], v123 offset:12288
	ds_read_b128 v[208:211], v117 offset:32768
	ds_read_b128 v[212:215], v117 offset:34816
	ds_read_b128 v[216:219], v119 offset:32768
	ds_read_b128 v[220:223], v119 offset:34816
	ds_read_b128 v[224:227], v117 offset:36864
	ds_read_b128 v[156:159], v117 offset:38912
	ds_read_b128 v[228:231], v119 offset:36864
	ds_read_b128 v[152:155], v119 offset:38912
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[208:211], v[168:171], v[108:111], v124, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(12)
	v_and_b32_e32 v126, 0xffff, v126
	s_waitcnt vmcnt(11)
	v_and_b32_e32 v160, 0xffff, v128
	s_waitcnt vmcnt(9)
	v_and_b32_e32 v162, 0xffff, v116
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[216:219], v[176:179], v[108:111], v124, v126 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(7)
	v_and_b32_e32 v164, 0xffff, v120
	s_waitcnt vmcnt(5)
	v_and_b32_e32 v133, 0xffff, v133
	v_and_b32_e32 v161, 0xffff, v129
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[208:211], v[172:175], v[104:107], v124, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_and_b32_e32 v163, 0xffff, v118
	v_and_b32_e32 v165, 0xffff, v112
	v_bfe_u32 v112, v111, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[208:211], v[188:191], v[100:103], v124, v162 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v120, v110, 16, 1
	s_movk_i32 s0, 0x7fff
	s_waitcnt vmcnt(3)
	v_and_b32_e32 v166, 0xffff, v122
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[208:211], v[192:195], v[96:99], v124, v164 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_and_b32_e32 v134, 0xffff, v134
	v_cmp_o_f32_e32 vcc, v111, v111
	s_waitcnt vmcnt(2)
	v_and_b32_e32 v167, 0xffff, v131
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[208:211], v[204:207], v[92:95], v124, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v132, 0xffff, v132
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v114, 0xffff, v114
	s_mul_hi_u32 s1, s8, s16
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[216:219], v[180:183], v[104:107], v124, v161 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_mov_b32 s3, 0x27000
	v_mfma_scale_f32_16x16x128_f8f6f4 v[116:119], v[216:219], v[184:187], v[100:103], v124, v163 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 2
	v_add3_u32 v102, v110, v120, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[120:123], v[216:219], v[196:199], v[96:99], v124, v165 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v100, v109, 16, 1
	v_bfe_u32 v101, v108, 16, 1
	v_add3_u32 v100, v109, v100, s0
	v_add3_u32 v96, v111, v112, s0
	v_lshrrev_b32_e32 v96, 16, v96
	v_mov_b32_e32 v98, 0x7fc0
	v_lshrrev_b32_e32 v97, 16, v102
	v_mfma_scale_f32_16x16x128_f8f6f4 v[128:131], v[216:219], v[200:203], v[92:95], v124, v134 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v101, v108, v101, s0
	v_lshrrev_b32_e32 v99, 16, v100
	v_lshrrev_b32_e32 v100, 16, v101
	v_cndmask_b32_e32 v92, v98, v96, vcc
	v_cmp_o_f32_e32 vcc, v110, v110
	v_bfe_u32 v96, v107, 16, 1
	v_add3_u32 v96, v107, v96, s0
	v_cndmask_b32_e32 v93, v98, v97, vcc
	v_cmp_o_f32_e32 vcc, v109, v109
	v_bfe_u32 v97, v106, 16, 1
	v_add3_u32 v97, v106, v97, s0
	v_cndmask_b32_e32 v94, v98, v99, vcc
	v_cmp_o_f32_e32 vcc, v108, v108
	v_bfe_u32 v99, v105, 16, 1
	v_lshrrev_b32_e32 v96, 16, v96
	v_cndmask_b32_e32 v95, v98, v100, vcc
	v_cmp_o_f32_e32 vcc, v107, v107
	v_bfe_u32 v100, v104, 16, 1
	v_add3_u32 v99, v105, v99, s0
	v_lshrrev_b32_e32 v97, 16, v97
	v_cndmask_b32_e32 v96, v98, v96, vcc
	v_cmp_o_f32_e32 vcc, v106, v106
	v_add3_u32 v100, v104, v100, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_cndmask_b32_e32 v97, v98, v97, vcc
	v_cmp_o_f32_e32 vcc, v105, v105
	v_bfe_u32 v101, v119, 16, 1
	v_lshrrev_b32_e32 v100, 16, v100
	v_cndmask_b32_e32 v99, v98, v99, vcc
	v_cmp_o_f32_e32 vcc, v104, v104
	v_bfe_u32 v102, v118, 16, 1
	v_add3_u32 v101, v119, v101, s0
	v_cndmask_b32_e32 v100, v98, v100, vcc
	v_bfe_u32 v103, v117, 16, 1
	v_add3_u32 v102, v118, v102, s0
	v_lshrrev_b32_e32 v101, 16, v101
	v_cmp_o_f32_e32 vcc, v119, v119
	v_bfe_u32 v104, v116, 16, 1
	v_add3_u32 v103, v117, v103, s0
	v_lshrrev_b32_e32 v102, 16, v102
	v_cndmask_b32_e32 v101, v98, v101, vcc
	v_cmp_o_f32_e32 vcc, v118, v118
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[208:211], v[148:151], v[88:91], v124, v166 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v104, v116, v104, s0
	v_lshrrev_b32_e32 v103, 16, v103
	v_cndmask_b32_e32 v102, v98, v102, vcc
	v_cmp_o_f32_e32 vcc, v117, v117
	v_bfe_u32 v105, v123, 16, 1
	v_lshrrev_b32_e32 v104, 16, v104
	v_cndmask_b32_e32 v103, v98, v103, vcc
	v_cmp_o_f32_e32 vcc, v116, v116
	v_bfe_u32 v106, v122, 16, 1
	v_add3_u32 v105, v123, v105, s0
	v_cndmask_b32_e32 v104, v98, v104, vcc
	v_bfe_u32 v107, v121, 16, 1
	v_add3_u32 v106, v122, v106, s0
	v_lshrrev_b32_e32 v105, 16, v105
	v_cmp_o_f32_e32 vcc, v123, v123
	v_bfe_u32 v108, v120, 16, 1
	v_add3_u32 v107, v121, v107, s0
	v_lshrrev_b32_e32 v106, 16, v106
	v_cndmask_b32_e32 v105, v98, v105, vcc
	v_cmp_o_f32_e32 vcc, v122, v122
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[216:219], v[144:147], v[88:91], v124, v167 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v108, v120, v108, s0
	v_lshrrev_b32_e32 v107, 16, v107
	v_cndmask_b32_e32 v106, v98, v106, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[208:211], v[136:139], v[84:87], v124, v132 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v121, v121
	v_bfe_u32 v109, v131, 16, 1
	v_lshrrev_b32_e32 v108, 16, v108
	v_cndmask_b32_e32 v107, v98, v107, vcc
	v_cmp_o_f32_e32 vcc, v120, v120
	v_bfe_u32 v110, v130, 16, 1
	v_add3_u32 v109, v131, v109, s0
	v_cndmask_b32_e32 v108, v98, v108, vcc
	v_bfe_u32 v111, v129, 16, 1
	v_add3_u32 v110, v130, v110, s0
	v_lshrrev_b32_e32 v109, 16, v109
	v_cmp_o_f32_e32 vcc, v131, v131
	v_bfe_u32 v112, v128, 16, 1
	v_add3_u32 v111, v129, v111, s0
	v_lshrrev_b32_e32 v110, 16, v110
	v_cndmask_b32_e32 v109, v98, v109, vcc
	v_cmp_o_f32_e32 vcc, v130, v130
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[216:219], v[140:143], v[84:87], v124, v114 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v112, v128, v112, s0
	v_lshrrev_b32_e32 v111, 16, v111
	v_cndmask_b32_e32 v110, v98, v110, vcc
	v_cmp_o_f32_e32 vcc, v129, v129
	v_bfe_u32 v116, v91, 16, 1
	v_lshrrev_b32_e32 v112, 16, v112
	v_cndmask_b32_e32 v111, v98, v111, vcc
	v_cmp_o_f32_e32 vcc, v128, v128
	v_bfe_u32 v117, v90, 16, 1
	v_add3_u32 v116, v91, v116, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[212:215], v[168:171], v[80:83], v124, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v112, v98, v112, vcc
	v_bfe_u32 v118, v89, 16, 1
	v_add3_u32 v117, v90, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cmp_o_f32_e32 vcc, v91, v91
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[212:215], v[136:139], v[56:59], v124, v132 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v119, v88, 16, 1
	v_add3_u32 v118, v89, v118, s0
	v_lshrrev_b32_e32 v117, 16, v117
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[224:227], v[136:139], v[20:23], v113, v132 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v91, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v90, v90
	v_add3_u32 v119, v88, v119, s0
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[156:159], v[136:139], v[24:27], v113, v132 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v118, 16, v118
	v_cndmask_b32_e32 v90, v98, v117, vcc
	v_cmp_o_f32_e32 vcc, v89, v89
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[224:227], v[168:171], v[52:55], v113, v135 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v116, v87, 16, 1
	v_lshrrev_b32_e32 v119, 16, v119
	v_cndmask_b32_e32 v89, v98, v118, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[224:227], v[172:175], v[48:51], v113, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v88, v88
	v_bfe_u32 v117, v86, 16, 1
	v_add3_u32 v116, v87, v116, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[224:227], v[188:191], v[44:47], v113, v162 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v88, v98, v119, vcc
	v_bfe_u32 v118, v85, 16, 1
	v_add3_u32 v117, v86, v117, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[224:227], v[192:195], v[40:43], v113, v164 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v116, 16, v116
	v_cmp_o_f32_e32 vcc, v87, v87
	v_bfe_u32 v119, v84, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[224:227], v[204:207], v[36:39], v113, v133 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v87, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v86, v86
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[224:227], v[148:151], v[32:35], v113, v166 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v86, v98, v117, vcc
	v_cmp_o_f32_e32 vcc, v85, v85
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[156:159], v[168:171], v[16:19], v113, v135 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[156:159], v[172:175], v[12:15], v113, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[156:159], v[188:191], v[8:11], v113, v162 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[156:159], v[192:195], v[4:7], v113, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[156:159], v[204:207], v[0:3], v113, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[156:159], v[148:151], v[28:31], v113, v166 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[220:223], v[176:179], v[80:83], v124, v126 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[212:215], v[172:175], v[76:79], v124, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[220:223], v[140:143], v[56:59], v124, v114 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 5
	v_bfe_u32 v116, v81, 16, 1
	v_bfe_u32 v117, v80, 16, 1
	v_add3_u32 v116, v81, v116, s0
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[228:231], v[140:143], v[20:23], v113, v114 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v117, v80, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_lshrrev_b32_e32 v117, 16, v117
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[152:155], v[140:143], v[24:27], v113, v114 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v114, v85, v118, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v85, v98, v114, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[228:231], v[176:179], v[52:55], v113, v126 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v84, v84
	v_bfe_u32 v114, v82, 16, 1
	v_add3_u32 v114, v82, v114, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[228:231], v[180:183], v[48:51], v113, v161 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v114, 16, v114
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[228:231], v[184:187], v[44:47], v113, v163 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[228:231], v[196:199], v[40:43], v113, v165 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[228:231], v[200:203], v[36:39], v113, v134 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[228:231], v[144:147], v[32:35], v113, v167 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[152:155], v[176:179], v[16:19], v113, v126 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[152:155], v[180:183], v[12:15], v113, v161 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[152:155], v[184:187], v[8:11], v113, v163 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[152:155], v[196:199], v[4:7], v113, v165 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[152:155], v[200:203], v[0:3], v113, v134 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[152:155], v[144:147], v[28:31], v113, v167 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v113, v84, v119, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cndmask_b32_e32 v84, v98, v113, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[220:223], v[180:183], v[76:79], v124, v161 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v113, v83, 16, 1
	v_add3_u32 v113, v83, v113, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[212:215], v[188:191], v[72:75], v124, v162 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v83, v83
	s_nop 1
	v_cndmask_b32_e32 v83, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v82, v82
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[220:223], v[184:187], v[72:75], v124, v163 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v113, v79, 16, 1
	v_cndmask_b32_e32 v82, v98, v114, vcc
	v_cmp_o_f32_e32 vcc, v81, v81
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[212:215], v[192:195], v[68:71], v124, v164 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v114, v78, 16, 1
	v_cndmask_b32_e32 v81, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v80, v80
	v_add3_u32 v113, v79, v113, s0
	v_bfe_u32 v116, v77, 16, 1
	v_cndmask_b32_e32 v80, v98, v117, vcc
	v_add3_u32 v114, v78, v114, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v79, v79
	v_bfe_u32 v117, v76, 16, 1
	v_add3_u32 v116, v77, v116, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v79, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v78, v78
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[220:223], v[196:199], v[68:71], v124, v165 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v117, v76, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v78, v98, v114, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[212:215], v[204:207], v[64:67], v124, v133 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v77, v77
	v_bfe_u32 v113, v75, 16, 1
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v77, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v76, v76
	v_bfe_u32 v114, v74, 16, 1
	v_add3_u32 v113, v75, v113, s0
	v_cndmask_b32_e32 v76, v98, v117, vcc
	v_bfe_u32 v116, v73, 16, 1
	v_add3_u32 v114, v74, v114, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v75, v75
	v_bfe_u32 v117, v72, 16, 1
	v_add3_u32 v116, v73, v116, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v75, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v74, v74
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[220:223], v[200:203], v[64:67], v124, v134 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v117, v72, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v74, v98, v114, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[212:215], v[148:151], v[60:63], v124, v166 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v73, v73
	v_bfe_u32 v113, v71, 16, 1
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v73, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v72, v72
	v_bfe_u32 v114, v70, 16, 1
	v_add3_u32 v113, v71, v113, s0
	v_cndmask_b32_e32 v72, v98, v117, vcc
	v_bfe_u32 v116, v69, 16, 1
	v_add3_u32 v114, v70, v114, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v71, v71
	v_bfe_u32 v117, v68, 16, 1
	v_add3_u32 v116, v69, v116, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v71, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v70, v70
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[220:223], v[144:147], v[60:63], v124, v167 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v117, v68, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v70, v98, v114, vcc
	v_cmp_o_f32_e32 vcc, v69, v69
	v_bfe_u32 v113, v67, 16, 1
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v69, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v68, v68
	v_bfe_u32 v114, v66, 16, 1
	v_add3_u32 v113, v67, v113, s0
	v_cndmask_b32_e32 v68, v98, v117, vcc
	v_bfe_u32 v116, v65, 16, 1
	v_add3_u32 v114, v66, v114, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v67, v67
	v_bfe_u32 v117, v64, 16, 1
	v_add3_u32 v116, v65, v116, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v67, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v66, v66
	v_add3_u32 v117, v64, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v66, v98, v114, vcc
	v_cmp_o_f32_e32 vcc, v65, v65
	v_bfe_u32 v113, v63, 16, 1
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v65, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v64, v64
	v_bfe_u32 v114, v62, 16, 1
	v_add3_u32 v113, v63, v113, s0
	v_cndmask_b32_e32 v64, v98, v117, vcc
	v_bfe_u32 v116, v61, 16, 1
	v_add3_u32 v114, v62, v114, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v63, v63
	v_bfe_u32 v117, v60, 16, 1
	v_add3_u32 v116, v61, v116, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v63, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v62, v62
	v_add3_u32 v117, v60, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v62, v98, v114, vcc
	v_cmp_o_f32_e32 vcc, v61, v61
	v_bfe_u32 v113, v59, 16, 1
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v61, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v60, v60
	v_bfe_u32 v114, v58, 16, 1
	v_add3_u32 v113, v59, v113, s0
	v_cndmask_b32_e32 v60, v98, v117, vcc
	v_bfe_u32 v116, v57, 16, 1
	v_add3_u32 v114, v58, v114, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v59, v59
	v_bfe_u32 v117, v56, 16, 1
	v_add3_u32 v116, v57, v116, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v59, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v58, v58
	v_add3_u32 v117, v56, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v58, v98, v114, vcc
	v_cmp_o_f32_e32 vcc, v57, v57
	v_bfe_u32 v113, v55, 16, 1
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v57, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v56, v56
	v_bfe_u32 v114, v54, 16, 1
	v_add3_u32 v113, v55, v113, s0
	v_cndmask_b32_e32 v56, v98, v117, vcc
	v_bfe_u32 v116, v53, 16, 1
	v_add3_u32 v114, v54, v114, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v55, v55
	v_bfe_u32 v117, v52, 16, 1
	v_add3_u32 v116, v53, v116, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v55, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v54, v54
	v_add3_u32 v117, v52, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v54, v98, v114, vcc
	v_cmp_o_f32_e32 vcc, v53, v53
	v_bfe_u32 v113, v51, 16, 1
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v53, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v52, v52
	v_bfe_u32 v114, v50, 16, 1
	v_add3_u32 v113, v51, v113, s0
	v_cndmask_b32_e32 v52, v98, v117, vcc
	v_bfe_u32 v116, v49, 16, 1
	v_add3_u32 v114, v50, v114, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v51, v51
	v_bfe_u32 v117, v48, 16, 1
	v_add3_u32 v116, v49, v116, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v51, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_add3_u32 v117, v48, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v50, v98, v114, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_bfe_u32 v113, v47, 16, 1
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v49, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_bfe_u32 v114, v46, 16, 1
	v_add3_u32 v113, v47, v113, s0
	v_cndmask_b32_e32 v48, v98, v117, vcc
	v_bfe_u32 v116, v45, 16, 1
	v_add3_u32 v114, v46, v114, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v47, v47
	v_bfe_u32 v117, v44, 16, 1
	v_add3_u32 v116, v45, v116, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v47, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v46, v46
	v_add3_u32 v117, v44, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v46, v98, v114, vcc
	v_cmp_o_f32_e32 vcc, v45, v45
	v_bfe_u32 v113, v43, 16, 1
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v45, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v44, v44
	v_bfe_u32 v114, v42, 16, 1
	v_add3_u32 v113, v43, v113, s0
	v_cndmask_b32_e32 v44, v98, v117, vcc
	v_bfe_u32 v116, v41, 16, 1
	v_add3_u32 v114, v42, v114, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v43, v43
	v_bfe_u32 v117, v40, 16, 1
	v_add3_u32 v116, v41, v116, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v43, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	v_add3_u32 v117, v40, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v42, v98, v114, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	v_bfe_u32 v113, v39, 16, 1
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v41, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_bfe_u32 v114, v38, 16, 1
	v_add3_u32 v113, v39, v113, s0
	v_cndmask_b32_e32 v40, v98, v117, vcc
	v_bfe_u32 v116, v37, 16, 1
	v_add3_u32 v114, v38, v114, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v39, v39
	v_bfe_u32 v117, v36, 16, 1
	v_add3_u32 v116, v37, v116, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v39, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v38, v38
	v_add3_u32 v117, v36, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v38, v98, v114, vcc
	v_cmp_o_f32_e32 vcc, v37, v37
	v_bfe_u32 v113, v35, 16, 1
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v37, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v36, v36
	v_bfe_u32 v114, v34, 16, 1
	v_add3_u32 v113, v35, v113, s0
	v_cndmask_b32_e32 v36, v98, v117, vcc
	v_bfe_u32 v116, v33, 16, 1
	v_add3_u32 v114, v34, v114, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v35, v35
	v_bfe_u32 v117, v32, 16, 1
	v_add3_u32 v116, v33, v116, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v35, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v34, v34
	v_add3_u32 v117, v32, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v34, v98, v114, vcc
	v_cmp_o_f32_e32 vcc, v33, v33
	v_bfe_u32 v113, v23, 16, 1
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v33, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v32, v32
	v_bfe_u32 v114, v22, 16, 1
	v_add3_u32 v113, v23, v113, s0
	v_cndmask_b32_e32 v32, v98, v117, vcc
	v_bfe_u32 v116, v21, 16, 1
	v_add3_u32 v114, v22, v114, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v23, v23
	v_bfe_u32 v117, v20, 16, 1
	v_add3_u32 v116, v21, v116, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v23, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v22, v22
	v_add3_u32 v117, v20, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v22, v98, v114, vcc
	v_cmp_o_f32_e32 vcc, v21, v21
	v_bfe_u32 v113, v19, 16, 1
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v21, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v20, v20
	v_bfe_u32 v114, v18, 16, 1
	v_add3_u32 v113, v19, v113, s0
	v_cndmask_b32_e32 v20, v98, v117, vcc
	v_bfe_u32 v116, v17, 16, 1
	v_add3_u32 v114, v18, v114, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v19, v19
	v_bfe_u32 v117, v16, 16, 1
	v_add3_u32 v116, v17, v116, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v19, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v18, v18
	v_add3_u32 v117, v16, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v18, v98, v114, vcc
	v_cmp_o_f32_e32 vcc, v17, v17
	v_bfe_u32 v113, v15, 16, 1
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v17, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v16, v16
	v_bfe_u32 v114, v14, 16, 1
	v_add3_u32 v113, v15, v113, s0
	v_cndmask_b32_e32 v16, v98, v117, vcc
	v_bfe_u32 v116, v13, 16, 1
	v_add3_u32 v114, v14, v114, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v15, v15
	v_bfe_u32 v117, v12, 16, 1
	v_add3_u32 v116, v13, v116, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v15, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_add3_u32 v117, v12, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v14, v98, v114, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_bfe_u32 v113, v11, 16, 1
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v13, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v114, v10, 16, 1
	v_add3_u32 v113, v11, v113, s0
	v_cndmask_b32_e32 v12, v98, v117, vcc
	v_bfe_u32 v116, v9, 16, 1
	v_add3_u32 v114, v10, v114, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v11, v11
	v_bfe_u32 v117, v8, 16, 1
	v_add3_u32 v116, v9, v116, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v11, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_add3_u32 v117, v8, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v10, v98, v114, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_bfe_u32 v113, v7, 16, 1
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v9, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v114, v6, 16, 1
	v_add3_u32 v113, v7, v113, s0
	v_cndmask_b32_e32 v8, v98, v117, vcc
	v_bfe_u32 v116, v5, 16, 1
	v_add3_u32 v114, v6, v114, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v117, v4, 16, 1
	v_add3_u32 v116, v5, v116, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v7, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v117, v4, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v6, v98, v114, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_bfe_u32 v113, v3, 16, 1
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v5, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v114, v2, 16, 1
	v_add3_u32 v113, v3, v113, s0
	v_cndmask_b32_e32 v4, v98, v117, vcc
	v_bfe_u32 v116, v1, 16, 1
	v_add3_u32 v114, v2, v114, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v3, v3
	v_bfe_u32 v117, v0, 16, 1
	v_add3_u32 v116, v1, v116, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v3, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	v_add3_u32 v117, v0, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v2, v98, v114, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	v_bfe_u32 v113, v31, 16, 1
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v1, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	v_bfe_u32 v114, v30, 16, 1
	v_add3_u32 v113, v31, v113, s0
	v_cndmask_b32_e32 v0, v98, v117, vcc
	v_bfe_u32 v116, v29, 16, 1
	v_add3_u32 v114, v30, v114, s0
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v31, v31
	v_bfe_u32 v117, v28, 16, 1
	v_add3_u32 v116, v29, v116, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v31, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v30, v30
	v_add3_u32 v117, v28, v117, s0
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v30, v98, v114, vcc
	v_cmp_o_f32_e32 vcc, v29, v29
	v_lshrrev_b32_e32 v117, 16, v117
	v_bfe_u32 v113, v27, 16, 1
	v_cndmask_b32_e32 v29, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v28, v28
	v_bfe_u32 v114, v26, 16, 1
	v_bfe_u32 v116, v25, 16, 1
	v_cndmask_b32_e32 v28, v98, v117, vcc
	v_bfe_u32 v117, v24, 16, 1
	v_add3_u32 v117, v24, v117, s0
	v_add3_u32 v116, v25, v116, s0
	v_add3_u32 v114, v26, v114, s0
	v_add3_u32 v113, v27, v113, s0
	s_mul_i32 s0, s9, s16
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s8, s16
	s_lshl_b64 s[0:1], s[0:1], 1
	v_lshrrev_b32_e32 v113, 16, v113
	v_cmp_o_f32_e32 vcc, v27, v27
	s_add_u32 s0, s10, s0
	v_lshrrev_b32_e32 v114, 16, v114
	v_cndmask_b32_e32 v27, v98, v113, vcc
	v_cmp_o_f32_e32 vcc, v26, v26
	s_addc_u32 s1, s11, s1
	s_lshl_b32 s2, s18, 1
	v_lshrrev_b32_e32 v116, 16, v116
	v_cndmask_b32_e32 v26, v98, v114, vcc
	v_cmp_o_f32_e32 vcc, v25, v25
	s_add_u32 s0, s0, s2
	v_lshrrev_b32_e32 v117, 16, v117
	v_cndmask_b32_e32 v25, v98, v116, vcc
	v_cmp_o_f32_e32 vcc, v24, v24
	s_addc_u32 s1, s1, 0
	s_and_b32 s2, s8, 0x3fff
	v_cndmask_b32_e32 v24, v98, v117, vcc
	v_lshl_or_b32 v98, v115, 2, v125
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	v_mul_lo_u32 v98, s8, v98
	s_or_b32 s1, s2, s1
	v_lshlrev_b32_e32 v113, 1, v127
	s_or_b32 s1, s1, 2.0
	s_mov_b32 s2, 0x7ffffffd
	v_lshl_add_u32 v114, v98, 1, v113
	s_lshl_b32 s4, s8, 1
	buffer_store_short v95, v114, s[0:3], 0 offen
	v_add_u32_e32 v95, s4, v114
	buffer_store_short v94, v95, s[0:3], 0 offen
	v_add_u32_e32 v94, s4, v95
	buffer_store_short v93, v94, s[0:3], 0 offen
	v_add_u32_e32 v93, s4, v94
	s_lshl_b32 s5, s8, 4
	buffer_store_short v92, v93, s[0:3], 0 offen
	buffer_store_short v100, v114, s[0:3], 0 offen offset:32
	buffer_store_short v99, v95, s[0:3], 0 offen offset:32
	buffer_store_short v97, v94, s[0:3], 0 offen offset:32
	buffer_store_short v96, v93, s[0:3], 0 offen offset:32
	buffer_store_short v104, v114, s[0:3], 0 offen offset:64
	buffer_store_short v103, v95, s[0:3], 0 offen offset:64
	buffer_store_short v102, v94, s[0:3], 0 offen offset:64
	buffer_store_short v101, v93, s[0:3], 0 offen offset:64
	buffer_store_short v108, v114, s[0:3], 0 offen offset:96
	buffer_store_short v107, v95, s[0:3], 0 offen offset:96
	buffer_store_short v106, v94, s[0:3], 0 offen offset:96
	buffer_store_short v105, v93, s[0:3], 0 offen offset:96
	buffer_store_short v112, v114, s[0:3], 0 offen offset:128
	buffer_store_short v111, v95, s[0:3], 0 offen offset:128
	buffer_store_short v110, v94, s[0:3], 0 offen offset:128
	buffer_store_short v109, v93, s[0:3], 0 offen offset:128
	buffer_store_short v88, v114, s[0:3], 0 offen offset:160
	buffer_store_short v89, v95, s[0:3], 0 offen offset:160
	buffer_store_short v90, v94, s[0:3], 0 offen offset:160
	buffer_store_short v91, v93, s[0:3], 0 offen offset:160
	buffer_store_short v84, v114, s[0:3], 0 offen offset:192
	buffer_store_short v85, v95, s[0:3], 0 offen offset:192
	buffer_store_short v86, v94, s[0:3], 0 offen offset:192
	buffer_store_short v87, v93, s[0:3], 0 offen offset:192
	v_add_u32_e32 v84, s5, v98
	v_lshl_add_u32 v85, v84, 1, v113
	buffer_store_short v80, v85, s[0:3], 0 offen
	v_add_u32_e32 v80, s4, v85
	buffer_store_short v81, v80, s[0:3], 0 offen
	v_add_u32_e32 v81, s4, v80
	buffer_store_short v82, v81, s[0:3], 0 offen
	v_add_u32_e32 v82, s4, v81
	buffer_store_short v83, v82, s[0:3], 0 offen
	buffer_store_short v76, v85, s[0:3], 0 offen offset:32
	buffer_store_short v77, v80, s[0:3], 0 offen offset:32
	buffer_store_short v78, v81, s[0:3], 0 offen offset:32
	buffer_store_short v79, v82, s[0:3], 0 offen offset:32
	buffer_store_short v72, v85, s[0:3], 0 offen offset:64
	buffer_store_short v73, v80, s[0:3], 0 offen offset:64
	buffer_store_short v74, v81, s[0:3], 0 offen offset:64
	buffer_store_short v75, v82, s[0:3], 0 offen offset:64
	buffer_store_short v68, v85, s[0:3], 0 offen offset:96
	buffer_store_short v69, v80, s[0:3], 0 offen offset:96
	buffer_store_short v70, v81, s[0:3], 0 offen offset:96
	buffer_store_short v71, v82, s[0:3], 0 offen offset:96
	buffer_store_short v64, v85, s[0:3], 0 offen offset:128
	buffer_store_short v65, v80, s[0:3], 0 offen offset:128
	buffer_store_short v66, v81, s[0:3], 0 offen offset:128
	buffer_store_short v67, v82, s[0:3], 0 offen offset:128
	buffer_store_short v60, v85, s[0:3], 0 offen offset:160
	buffer_store_short v61, v80, s[0:3], 0 offen offset:160
	buffer_store_short v62, v81, s[0:3], 0 offen offset:160
	buffer_store_short v63, v82, s[0:3], 0 offen offset:160
	buffer_store_short v56, v85, s[0:3], 0 offen offset:192
	buffer_store_short v57, v80, s[0:3], 0 offen offset:192
	buffer_store_short v58, v81, s[0:3], 0 offen offset:192
	buffer_store_short v59, v82, s[0:3], 0 offen offset:192
	v_add_u32_e32 v56, s5, v84
	v_lshl_add_u32 v57, v56, 1, v113
	buffer_store_short v52, v57, s[0:3], 0 offen
	v_add_u32_e32 v52, s4, v57
	buffer_store_short v53, v52, s[0:3], 0 offen
	v_add_u32_e32 v53, s4, v52
	buffer_store_short v54, v53, s[0:3], 0 offen
	v_add_u32_e32 v54, s4, v53
	buffer_store_short v55, v54, s[0:3], 0 offen
	buffer_store_short v48, v57, s[0:3], 0 offen offset:32
	buffer_store_short v49, v52, s[0:3], 0 offen offset:32
	buffer_store_short v50, v53, s[0:3], 0 offen offset:32
	buffer_store_short v51, v54, s[0:3], 0 offen offset:32
	buffer_store_short v44, v57, s[0:3], 0 offen offset:64
	buffer_store_short v45, v52, s[0:3], 0 offen offset:64
	buffer_store_short v46, v53, s[0:3], 0 offen offset:64
	buffer_store_short v47, v54, s[0:3], 0 offen offset:64
	buffer_store_short v40, v57, s[0:3], 0 offen offset:96
	buffer_store_short v41, v52, s[0:3], 0 offen offset:96
	buffer_store_short v42, v53, s[0:3], 0 offen offset:96
	buffer_store_short v43, v54, s[0:3], 0 offen offset:96
	buffer_store_short v36, v57, s[0:3], 0 offen offset:128
	buffer_store_short v37, v52, s[0:3], 0 offen offset:128
	buffer_store_short v38, v53, s[0:3], 0 offen offset:128
	buffer_store_short v39, v54, s[0:3], 0 offen offset:128
	buffer_store_short v32, v57, s[0:3], 0 offen offset:160
	buffer_store_short v33, v52, s[0:3], 0 offen offset:160
	buffer_store_short v34, v53, s[0:3], 0 offen offset:160
	buffer_store_short v35, v54, s[0:3], 0 offen offset:160
	buffer_store_short v20, v57, s[0:3], 0 offen offset:192
	buffer_store_short v21, v52, s[0:3], 0 offen offset:192
	buffer_store_short v22, v53, s[0:3], 0 offen offset:192
	buffer_store_short v23, v54, s[0:3], 0 offen offset:192
	v_add_u32_e32 v20, s5, v56
	v_lshl_add_u32 v20, v20, 1, v113
	buffer_store_short v16, v20, s[0:3], 0 offen
	v_add_u32_e32 v16, s4, v20
	buffer_store_short v17, v16, s[0:3], 0 offen
	v_add_u32_e32 v17, s4, v16
	buffer_store_short v18, v17, s[0:3], 0 offen
	v_add_u32_e32 v18, s4, v17
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
	buffer_store_short v28, v20, s[0:3], 0 offen offset:160
	buffer_store_short v29, v16, s[0:3], 0 offen offset:160
	buffer_store_short v30, v17, s[0:3], 0 offen offset:160
	buffer_store_short v31, v18, s[0:3], 0 offen offset:160
	buffer_store_short v24, v20, s[0:3], 0 offen offset:192
	buffer_store_short v25, v16, s[0:3], 0 offen offset:192
	buffer_store_short v26, v17, s[0:3], 0 offen offset:192
	buffer_store_short v27, v18, s[0:3], 0 offen offset:192
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_256x224x256_5376x1792x4096
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
		.amdhsa_next_free_vgpr 254
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
	.size	wave_mxfp4_static_gemm_256x224x256_5376x1792x4096, .Lfunc_end0-wave_mxfp4_static_gemm_256x224x256_5376x1792x4096

	.set wave_mxfp4_static_gemm_256x224x256_5376x1792x4096.num_vgpr, 254
	.set wave_mxfp4_static_gemm_256x224x256_5376x1792x4096.num_agpr, 0
	.set wave_mxfp4_static_gemm_256x224x256_5376x1792x4096.numbered_sgpr, 76
	.set wave_mxfp4_static_gemm_256x224x256_5376x1792x4096.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_256x224x256_5376x1792x4096.private_seg_size, 0
	.set wave_mxfp4_static_gemm_256x224x256_5376x1792x4096.uses_vcc, 1
	.set wave_mxfp4_static_gemm_256x224x256_5376x1792x4096.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_256x224x256_5376x1792x4096.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_256x224x256_5376x1792x4096.has_recursion, 0
	.set wave_mxfp4_static_gemm_256x224x256_5376x1792x4096.has_indirect_call, 0
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
    .name:           wave_mxfp4_static_gemm_256x224x256_5376x1792x4096
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     82
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_256x224x256_5376x1792x4096.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     254
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
