; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_256x224x256_6272x7168x5888
	.p2align	8
	.type	wave_mxfp4_static_gemm_256x224x256_6272x7168x5888,@function
wave_mxfp4_static_gemm_256x224x256_6272x7168x5888:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v117, 0x3ff, v0
	v_bfe_u32 v2, v0, 10, 10
	v_lshrrev_b32_e32 v4, 6, v117
	v_lshlrev_b32_e32 v0, 5, v2
	v_lshl_or_b32 v1, v4, 3, v0
	s_mov_b64 s[24:25], s[2:3]
	v_readfirstlane_b32 s2, v1
	v_lshrrev_b32_e32 v1, 3, v117
	s_lshl_b32 s33, s16, 8
	v_or3_b32 v3, v1, v0, s33
	v_xor_b32_e32 v1, v1, v117
	v_lshlrev_b32_e32 v1, 4, v1
	v_and_b32_e32 v118, 0x70, v1
	v_mul_u32_u24_e32 v119, 0xb80, v3
	s_and_b32 s3, s25, 0xffff
	s_lshl_b32 s37, s2, 7
	s_or_b32 s25, s3, 0x4b800000
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_or_b32_e32 v1, v119, v118
	s_mov_b32 m0, s37
	s_or_b32 s38, s37, 0x2000
	buffer_load_dwordx4 v1, s[24:27], 0 offen lds
	v_add_u32_e32 v1, 0x2e000, v1
	s_mov_b32 m0, s38
	v_or_b32_e32 v5, 0x80, v3
	buffer_load_dwordx4 v1, s[24:27], 0 offen lds
	v_mul_u32_u24_e32 v1, 0xb80, v5
	s_movk_i32 s39, 0x1880
	s_mov_b64 s[20:21], s[6:7]
	v_or_b32_e32 v6, v1, v118
	v_bfrev_b32_e32 v120, -2
	v_cmp_gt_u32_e64 s[6:7], s39, v5
	s_or_b32 s40, s37, 0x4000
	s_mov_b32 m0, s40
	v_cndmask_b32_e64 v5, v120, v6, s[6:7]
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_or_b32_e32 v5, 0xc0, v3
	v_mul_u32_u24_e32 v3, 0xb80, v5
	v_or_b32_e32 v6, v3, v118
	v_cmp_gt_u32_e64 s[2:3], s39, v5
	s_or_b32 s41, s37, 0x6000
	s_mov_b32 m0, s41
	v_cndmask_b32_e64 v5, v120, v6, s[2:3]
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_lshlrev_b32_e32 v5, 3, v2
	v_lshl_or_b32 v5, v4, 1, v5
	v_bfe_u32 v9, v117, 2, 3
	v_readfirstlane_b32 s16, v5
	v_lshrrev_b32_e32 v5, 5, v117
	v_lshrrev_b32_e32 v6, 2, v117
	v_and_b32_e32 v10, 31, v117
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
	v_and_b32_e32 v9, 0xfc, v117
	v_lshlrev_b32_e32 v10, 7, v15
	v_lshlrev_b32_e32 v7, 6, v12
	v_lshlrev_b32_e32 v8, 2, v5
	v_add_u32_e32 v16, v9, v10
	v_add3_u32 v13, v0, v117, v7
	v_lshlrev_b32_e32 v11, 7, v14
	v_sub_u32_e32 v16, v8, v16
	v_add3_u32 v13, v16, v13, v11
	v_ashrrev_i32_e32 v16, 31, v13
	v_xor_b32_e32 v13, v16, v13
	s_mov_b32 s42, 0xb21642c9
	v_mul_hi_i32 v17, v13, s42
	v_add_u32_e32 v13, v17, v13
	v_lshrrev_b32_e32 v17, 31, v13
	v_ashrrev_i32_e32 v13, 9, v13
	v_add_u32_e32 v13, v13, v17
	v_xor_b32_e32 v16, v13, v16
	v_sub_u32_e32 v13, v5, v6
	v_lshlrev_b32_e32 v18, 4, v13
	v_lshlrev_b32_e32 v121, 2, v117
	v_lshlrev_b32_e32 v13, 9, v14
	v_lshlrev_b32_e32 v14, 8, v12
	v_add3_u32 v18, v18, v121, v14
	v_lshlrev_b32_e32 v12, 9, v15
	s_mul_i32 s36, s17, 0xe0
	v_sub_u32_e32 v15, v18, v12
	v_add_u32_e32 v17, s36, v16
	v_add_u32_e32 v18, v15, v13
	v_mul_i32_i24_e32 v16, 0xfffff480, v16
	v_lshlrev_b32_e32 v15, 7, v2
	s_movk_i32 s13, 0xb80
	v_add3_u32 v16, v18, v15, v16
	s_and_b32 s17, s21, 0xffff
	s_lshl_b32 s43, s16, 7
	s_or_b32 s21, s17, 0x4b800000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_mad_i32_i24 v16, v17, s13, v16
	s_add_i32 m0, s43, 0x10000
	s_or_b32 s44, s43, 0x800
	buffer_load_dword v16, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0xb800, v16
	s_add_i32 m0, s44, 0x10000
	s_or_b32 s45, s43, 0x1000
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0x17000, v16
	s_add_i32 m0, s45, 0x10000
	s_or_b32 s46, s43, 0x1800
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0x22800, v16
	s_add_i32 m0, s46, 0x10000
	s_or_b32 s47, s43, 0x2000
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0x2e000, v16
	s_add_i32 m0, s47, 0x10000
	s_or_b32 s48, s43, 0x2800
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0x39800, v16
	s_add_i32 m0, s48, 0x10000
	s_or_b32 s49, s43, 0x3000
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0x45000, v16
	s_add_i32 m0, s49, 0x10000
	s_or_b32 s50, s43, 0x3800
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0x50800, v16
	s_add_i32 m0, s50, 0x10000
	s_or_b32 s51, s43, 0x4000
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0x5c000, v16
	s_add_i32 m0, s51, 0x10000
	s_or_b32 s52, s43, 0x4800
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0x67800, v16
	s_add_i32 m0, s52, 0x10000
	s_or_b32 s53, s43, 0x5000
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0x73000, v16
	s_add_i32 m0, s53, 0x10000
	s_or_b32 s54, s43, 0x5800
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0x7e800, v16
	s_add_i32 m0, s54, 0x10000
	s_or_b32 s55, s43, 0x6000
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v17, 0x8a000, v16
	s_add_i32 m0, s55, 0x10000
	s_or_b32 s56, s43, 0x6800
	buffer_load_dword v17, s[20:23], 0 offen lds
	v_add_u32_e32 v16, 0x95800, v16
	s_add_i32 m0, s56, 0x10000
	v_lshrrev_b32_e32 v19, 4, v117
	v_bfe_u32 v116, v117, 4, 2
	buffer_load_dword v16, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v16, 4, v116
	v_mad_i32_i24 v20, v19, -16, v117
	v_add_u32_e32 v17, v20, v16
	v_ashrrev_i32_e32 v18, 31, v17
	v_xor_b32_e32 v21, v18, v17
	v_mul_hi_i32 v22, v21, s42
	v_add_u32_e32 v21, v22, v21
	v_lshrrev_b32_e32 v22, 31, v21
	v_ashrrev_i32_e32 v21, 5, v21
	s_movk_i32 s12, 0x70
	v_add_u32_e32 v21, v21, v22
	v_and_b32_e32 v38, 0xc0, v117
	s_mul_i32 s15, s15, s33
	s_mul_hi_u32 s16, s14, s33
	v_xad_u32 v21, v21, v18, v38
	v_and_b32_e32 v18, 62, v117
	s_movk_i32 s57, 0xffc0
	s_add_i32 s15, s16, s15
	s_mul_i32 s16, s14, s33
	v_mad_u32_u24 v112, v2, s12, v20
	v_mov_b32_e32 v23, 0xffffff48
	v_cmp_lt_u32_e32 vcc, 45, v18
	v_lshlrev_b32_e32 v18, 6, v116
	v_mad_i32_i24 v26, v19, s57, v121
	s_add_u32 s16, s4, s16
	v_ashrrev_i16_e32 v20, 15, v112
	v_add_u32_e32 v22, s33, v21
	v_cndmask_b32_e32 v23, 0, v23, vcc
	v_add_u32_e32 v24, v26, v18
	v_mul_lo_u32 v21, v21, s14
	s_addc_u32 s4, s5, s15
	s_and_b32 s5, s14, 0x3fff
	v_lshrrev_b16_e32 v20, 11, v20
	v_add3_u32 v21, v24, v23, v21
	s_bitset1_b32 s5, 14
	v_cmp_gt_i32_e32 vcc, s39, v22
	s_movk_i32 s58, 0x1860
	v_add_u16_e32 v20, v112, v20
	s_and_b32 s4, s4, 0xffff
	s_lshl_b32 s5, s5, 16
	v_cndmask_b32_e32 v23, v120, v21, vcc
	v_lshl_add_u32 v21, s14, 5, v21
	v_cmp_gt_i32_e32 vcc, s58, v22
	v_and_b32_e32 v20, 0xffffffe0, v20
	s_or_b32 s17, s4, s5
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	v_cndmask_b32_e32 v21, v120, v21, vcc
	v_sub_u16_e32 v20, v112, v20
	buffer_load_dword v127, v23, s[16:19], 0 offen
	buffer_load_dword v125, v21, s[16:19], 0 offen
	v_bfe_i32 v21, v20, 0, 16
	v_ashrrev_i32_e32 v22, 31, v21
	v_add_u16_e32 v23, 32, v20
	v_cmp_gt_i16_e32 vcc, 0, v20
	s_load_dwordx2 s[34:35], s[0:1], 0x40
	s_movk_i32 s59, 0xb8
	v_cndmask_b32_e32 v20, v21, v23, vcc
	v_cndmask_b32_e64 v21, v22, 0, vcc
	v_xor_b32_e32 v20, v21, v20
	v_lshrrev_b32_e32 v22, 28, v20
	v_add_u32_e32 v20, v20, v22
	v_ashrrev_i32_e32 v20, 4, v20
	v_xor_b32_e32 v20, v20, v21
	v_add_u32_e32 v21, v24, v20
	v_ashrrev_i32_e32 v22, 31, v21
	v_xor_b32_e32 v23, v22, v21
	v_mul_hi_i32 v24, v23, s42
	v_add_u32_e32 v23, v24, v23
	v_lshrrev_b32_e32 v24, 31, v23
	v_ashrrev_i32_e32 v23, 7, v23
	v_add_u32_e32 v23, v23, v24
	v_ashrrev_i32_e32 v24, 31, v112
	v_xor_b32_e32 v25, v24, v112
	v_ashrrev_i32_e32 v27, 31, v25
	v_lshrrev_b32_e32 v27, 27, v27
	v_add_u32_e32 v25, v25, v27
	v_lshrrev_b32_e32 v25, 5, v25
	v_xor_b32_e32 v24, v25, v24
	v_lshlrev_b32_e32 v122, 5, v24
	v_xad_u32 v23, v23, v22, v122
	v_mul_hi_i32 v22, v21, s42
	v_add_u32_e32 v22, v22, v21
	v_lshrrev_b32_e32 v24, 31, v22
	v_ashrrev_i32_e32 v22, 7, v22
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s4, s35, s36
	s_mul_hi_u32 s5, s34, s36
	v_add_u32_e32 v22, v22, v24
	s_add_i32 s5, s5, s4
	s_mul_i32 s4, s34, s36
	v_mul_lo_u32 v22, v22, s59
	s_add_u32 s28, s8, s4
	v_sub_u32_e32 v22, v21, v22
	s_addc_u32 s4, s9, s5
	s_and_b32 s5, s34, 0x3fff
	v_add_u32_e32 v24, 0xb8, v22
	v_cmp_gt_i32_e32 vcc, 0, v22
	s_bitset1_b32 s5, 14
	s_and_b32 s4, s4, 0xffff
	v_cndmask_b32_e32 v22, v22, v24, vcc
	s_lshl_b32 s5, s5, 16
	s_or_b32 s29, s4, s5
	v_mad_u64_u32 v[24:25], s[4:5], s34, v23, v[22:23]
	v_add_u32_e32 v22, 2, v21
	v_sub_u32_e32 v23, -3, v21
	v_cmp_gt_i32_e32 vcc, -2, v21
	s_movk_i32 s13, 0xff48
	v_add_u32_e32 v113, 16, v112
	v_cndmask_b32_e32 v21, v22, v23, vcc
	v_mul_hi_i32 v22, v21, s42
	v_add_u32_e32 v21, v22, v21
	v_lshrrev_b32_e32 v22, 31, v21
	v_ashrrev_i32_e32 v21, 7, v21
	v_add_u32_e32 v21, v21, v22
	v_cndmask_b32_e64 v22, 0, -1, vcc
	v_xor_b32_e32 v21, v21, v22
	v_add_u32_e32 v22, v21, v122
	v_mul_lo_u32 v21, v21, s13
	v_mul_lo_u32 v22, s34, v22
	v_add3_u32 v23, v20, v18, v26
	v_add3_u32 v23, v23, v21, v22
	v_sub_u32_e32 v21, 0xffef, v112
	v_cmp_gt_i32_e32 vcc, -16, v112
	v_mad_u32_u24 v25, v2, 7, v121
	v_mov_b32_e32 v39, 5
	v_cndmask_b32_e32 v21, v113, v21, vcc
	v_ashrrev_i16_e32 v22, 15, v21
	v_lshrrev_b16_e32 v22, 11, v22
	v_add_u16_e32 v21, v21, v22
	v_ashrrev_i16_e32 v21, 5, v21
	v_cndmask_b32_e64 v22, 0, -1, vcc
	v_xor_b32_e32 v27, v21, v22
	v_mov_b32_e32 v21, -2
	v_mul_i32_i24_sdwa v22, sext(v27), v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_mad_i32_i24 v21, v19, s57, v18
	v_add3_u32 v26, v21, v25, v22
	v_add_u32_e32 v25, 1, v26
	v_sub_u32_e32 v28, -2, v26
	v_cmp_gt_i32_e32 vcc, -1, v26
	v_lshlrev_b32_sdwa v123, v39, sext(v27) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_e32 v114, 48, v112
	v_cndmask_b32_e32 v25, v25, v28, vcc
	v_mul_hi_i32 v28, v25, s42
	v_add_u32_e32 v25, v28, v25
	v_lshrrev_b32_e32 v28, 31, v25
	v_ashrrev_i32_e32 v25, 7, v25
	v_add_u32_e32 v25, v25, v28
	v_cndmask_b32_e64 v28, 0, -1, vcc
	v_xor_b32_e32 v25, v25, v28
	v_add_u32_e32 v27, v25, v123
	v_mad_u64_u32 v[28:29], s[4:5], v25, s13, v[26:27]
	v_mad_u64_u32 v[30:31], s[4:5], v27, s34, v[28:29]
	v_add_u32_e32 v27, 3, v26
	v_sub_u32_e32 v29, -4, v26
	v_cmp_gt_i32_e32 vcc, -3, v26
	s_lshl_b32 s8, s34, 5
	s_mov_b32 s30, s26
	v_cndmask_b32_e32 v27, v27, v29, vcc
	v_mul_hi_i32 v29, v27, s42
	v_add_u32_e32 v27, v29, v27
	v_lshrrev_b32_e32 v29, 31, v27
	v_ashrrev_i32_e32 v27, 7, v27
	v_add_u32_e32 v27, v27, v29
	v_cndmask_b32_e64 v29, 0, -1, vcc
	v_xor_b32_e32 v29, v27, v29
	v_add_u32_e32 v31, v29, v123
	v_mad_u64_u32 v[26:27], s[4:5], v29, s13, v[26:27]
	v_mad_u64_u32 v[32:33], s[4:5], v31, s34, v[26:27]
	s_movk_i32 s4, 0xffd0
	v_sub_u32_e32 v33, 0xffcf, v112
	v_cmp_gt_i32_e32 vcc, s4, v112
	v_add_u32_e32 v27, s8, v24
	s_mov_b32 s31, s27
	v_cndmask_b32_e32 v33, v114, v33, vcc
	v_ashrrev_i16_e32 v34, 15, v33
	v_lshrrev_b16_e32 v34, 11, v34
	v_add_u16_e32 v33, v33, v34
	v_ashrrev_i16_e32 v33, 5, v33
	v_cndmask_b32_e64 v34, 0, -1, vcc
	v_xor_b32_e32 v33, v33, v34
	v_lshlrev_b32_sdwa v124, v39, sext(v33) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_e32 v33, v25, v124
	v_mad_u64_u32 v[34:35], s[4:5], v33, s34, v[28:29]
	v_add_u32_e32 v33, v29, v124
	v_mad_u64_u32 v[36:37], s[4:5], v33, s34, v[26:27]
	s_movk_i32 s4, 0xffb0
	v_add_u32_e32 v31, s8, v23
	buffer_load_ubyte v158, v24, s[28:31], 0 offen
	buffer_load_ubyte v151, v23, s[28:31], 0 offen offset:2
	buffer_load_ubyte v157, v30, s[28:31], 0 offen offset:1
	buffer_load_ubyte v150, v32, s[28:31], 0 offen offset:3
	buffer_load_ubyte v156, v27, s[28:31], 0 offen
	buffer_load_ubyte v149, v31, s[28:31], 0 offen offset:2
	buffer_load_ubyte v155, v34, s[28:31], 0 offen offset:1
	buffer_load_ubyte v148, v36, s[28:31], 0 offen offset:3
	v_add_u32_e32 v115, 0x50, v112
	v_sub_u32_e32 v24, 0xffaf, v112
	v_cmp_gt_i32_e32 vcc, s4, v112
	v_add_u32_e32 v23, s8, v27
	v_add_u32_e32 v30, s8, v31
	v_cndmask_b32_e32 v24, v115, v24, vcc
	v_ashrrev_i16_e32 v27, 15, v24
	v_lshrrev_b16_e32 v27, 11, v27
	v_add_u16_e32 v24, v24, v27
	v_ashrrev_i16_e32 v24, 5, v24
	v_cndmask_b32_e64 v27, 0, -1, vcc
	v_xor_b32_e32 v24, v24, v27
	v_lshlrev_b32_sdwa v126, v39, sext(v24) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_e32 v24, v25, v126
	v_mad_u64_u32 v[24:25], s[4:5], v24, s34, v[28:29]
	v_add_u32_e32 v25, v29, v126
	v_mad_u64_u32 v[26:27], s[4:5], v25, s34, v[26:27]
	v_add_u32_e32 v25, s8, v23
	v_add_u32_e32 v27, s8, v30
	buffer_load_ubyte v161, v23, s[28:31], 0 offen
	buffer_load_ubyte v154, v30, s[28:31], 0 offen offset:2
	buffer_load_ubyte v160, v24, s[28:31], 0 offen offset:1
	buffer_load_ubyte v153, v26, s[28:31], 0 offen offset:3
	buffer_load_ubyte v159, v25, s[28:31], 0 offen
	buffer_load_ubyte v152, v27, s[28:31], 0 offen offset:2
	v_cmp_eq_u32_e64 s[4:5], 0, v2
	s_mov_b32 s9, 0
	s_movk_i32 s60, 0xf480
	s_mov_b32 s35, 0x10000
	s_mov_b32 s70, 0x17000
	s_movk_i32 s15, 0x3800
	v_mul_i32_i24_e32 v23, -16, v19
	v_readfirstlane_b32 s8, v38
	v_mul_i32_i24_e32 v24, 0xffffffc0, v19
	v_mul_u32_u24_e32 v25, 7, v2
	s_and_b64 vcc, exec, s[4:5]
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	v_lshlrev_b32_e32 v28, 7, v117
	v_lshlrev_b32_e32 v19, 11, v19
	v_and_b32_e32 v26, 7, v117
	v_sub_u32_e32 v19, v28, v19
	v_mul_lo_u32 v2, v2, s15
	v_bitop3_b32 v27, v116, v117, 7 bitop3:0x78
	v_lshl_add_u32 v4, v4, 13, v19
	v_add_u32_e32 v2, v19, v2
	v_bitop3_b32 v19, v116, v26, 4 bitop3:0x36
	v_lshlrev_b32_e32 v27, 4, v27
	v_lshlrev_b32_e32 v19, 4, v19
	v_or_b32_e32 v130, v2, v27
	v_or_b32_e32 v131, v19, v2
	v_add_u32_e32 v2, v22, v24
	v_add3_u32 v132, v2, v18, v25
	v_add_u32_e32 v2, v2, v121
	v_add3_u32 v2, v2, v25, v18
	v_sub_u32_e32 v133, 0, v2
	v_sub_u32_e32 v2, 0, v24
	v_or_b32_e32 v128, v4, v27
	v_or_b32_e32 v129, v19, v4
	v_sub_u32_e32 v4, v2, v22
	v_sub_u32_e32 v2, v2, v20
	v_sub_u32_e32 v4, v4, v121
	v_sub_u32_e32 v2, v2, v121
	v_sub_u32_e32 v4, v4, v25
	v_sub_u32_e32 v135, v2, v18
	v_add3_u32 v2, v13, v14, v15
	v_sub_u32_e32 v134, v4, v18
	v_lshl_add_u32 v2, v5, 4, v2
	v_lshlrev_b32_e32 v4, 4, v6
	v_sub_u32_e32 v2, v2, v4
	s_load_dwordx2 s[12:13], s[0:1], 0x48
	v_sub_u32_e32 v143, v2, v12
	v_add_u32_e32 v2, v11, v7
	v_add3_u32 v0, v2, v0, v8
	v_add_u32_e32 v2, v0, v117
	v_sub_u32_e32 v0, v0, v9
	v_add3_u32 v136, v24, v20, v18
	v_sub_u32_e32 v2, v9, v2
	s_movk_i32 s1, 0xfdff
	v_sub_u32_e32 v0, v0, v10
	v_mov_b32_e32 v28, 0
	s_or_b32 s15, s8, s33
	v_add_u32_e32 v137, 32, v122
	v_add_u32_e32 v138, 64, v122
	v_add_u32_e32 v139, 0x60, v122
	s_add_i32 s0, s8, 32
	v_add_u32_e32 v140, 0x48, v21
	v_sub_u32_e32 v141, 0xffffffed, v17
	v_add_u32_e32 v142, v23, v16
	v_add3_u32 v144, v10, v2, s1
	v_add_u32_e32 v145, 0x200, v0
	v_add_u32_e32 v146, 0x80, v3
	v_add_u32_e32 v147, 0x80, v1
	s_mov_b32 s69, 0x8000
	s_mov_b32 s1, 0
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	s_movk_i32 s61, 0xffee
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	s_movk_i32 s62, 0xff00
	s_movk_i32 s63, 0xfeff
	s_movk_i32 s64, 0xffb8
	s_movk_i32 s65, 0xffb7
	s_mov_b32 s30, s26
	s_mov_b32 s31, s27
	s_movk_i32 s66, 0xfefd
	s_movk_i32 s67, 0xfefe
	v_mov_b32_e32 v29, v28
	v_mov_b32_e32 v30, v28
	v_mov_b32_e32 v31, v28
	v_mov_b32_e32 v108, v28
	v_mov_b32_e32 v109, v28
	v_mov_b32_e32 v110, v28
	v_mov_b32_e32 v111, v28
	v_mov_b32_e32 v104, v28
	v_mov_b32_e32 v105, v28
	v_mov_b32_e32 v106, v28
	v_mov_b32_e32 v107, v28
	v_mov_b32_e32 v100, v28
	v_mov_b32_e32 v101, v28
	v_mov_b32_e32 v102, v28
	v_mov_b32_e32 v103, v28
	v_mov_b32_e32 v96, v28
	v_mov_b32_e32 v97, v28
	v_mov_b32_e32 v98, v28
	v_mov_b32_e32 v99, v28
	v_mov_b32_e32 v92, v28
	v_mov_b32_e32 v93, v28
	v_mov_b32_e32 v94, v28
	v_mov_b32_e32 v95, v28
	v_mov_b32_e32 v88, v28
	v_mov_b32_e32 v89, v28
	v_mov_b32_e32 v90, v28
	v_mov_b32_e32 v91, v28
	v_mov_b32_e32 v84, v28
	v_mov_b32_e32 v85, v28
	v_mov_b32_e32 v86, v28
	v_mov_b32_e32 v87, v28
	v_mov_b32_e32 v80, v28
	v_mov_b32_e32 v81, v28
	v_mov_b32_e32 v82, v28
	v_mov_b32_e32 v83, v28
	v_mov_b32_e32 v76, v28
	v_mov_b32_e32 v77, v28
	v_mov_b32_e32 v78, v28
	v_mov_b32_e32 v79, v28
	v_mov_b32_e32 v72, v28
	v_mov_b32_e32 v73, v28
	v_mov_b32_e32 v74, v28
	v_mov_b32_e32 v75, v28
	v_mov_b32_e32 v68, v28
	v_mov_b32_e32 v69, v28
	v_mov_b32_e32 v70, v28
	v_mov_b32_e32 v71, v28
	v_mov_b32_e32 v64, v28
	v_mov_b32_e32 v65, v28
	v_mov_b32_e32 v66, v28
	v_mov_b32_e32 v67, v28
	v_mov_b32_e32 v60, v28
	v_mov_b32_e32 v61, v28
	v_mov_b32_e32 v62, v28
	v_mov_b32_e32 v63, v28
	v_mov_b32_e32 v56, v28
	v_mov_b32_e32 v57, v28
	v_mov_b32_e32 v58, v28
	v_mov_b32_e32 v59, v28
	v_mov_b32_e32 v52, v28
	v_mov_b32_e32 v53, v28
	v_mov_b32_e32 v54, v28
	v_mov_b32_e32 v55, v28
	v_mov_b32_e32 v48, v28
	v_mov_b32_e32 v49, v28
	v_mov_b32_e32 v50, v28
	v_mov_b32_e32 v51, v28
	v_mov_b32_e32 v44, v28
	v_mov_b32_e32 v45, v28
	v_mov_b32_e32 v46, v28
	v_mov_b32_e32 v47, v28
	v_mov_b32_e32 v40, v28
	v_mov_b32_e32 v41, v28
	v_mov_b32_e32 v42, v28
	v_mov_b32_e32 v43, v28
	v_mov_b32_e32 v20, v28
	v_mov_b32_e32 v21, v28
	v_mov_b32_e32 v22, v28
	v_mov_b32_e32 v23, v28
	v_mov_b32_e32 v8, v28
	v_mov_b32_e32 v9, v28
	v_mov_b32_e32 v10, v28
	v_mov_b32_e32 v11, v28
	v_mov_b32_e32 v4, v28
	v_mov_b32_e32 v5, v28
	v_mov_b32_e32 v6, v28
	v_mov_b32_e32 v7, v28
	v_mov_b32_e32 v12, v28
	v_mov_b32_e32 v13, v28
	v_mov_b32_e32 v14, v28
	v_mov_b32_e32 v15, v28
	v_mov_b32_e32 v16, v28
	v_mov_b32_e32 v17, v28
	v_mov_b32_e32 v18, v28
	v_mov_b32_e32 v19, v28
	v_mov_b32_e32 v24, v28
	v_mov_b32_e32 v25, v28
	v_mov_b32_e32 v26, v28
	v_mov_b32_e32 v27, v28
	v_mov_b32_e32 v32, v28
	v_mov_b32_e32 v33, v28
	v_mov_b32_e32 v34, v28
	v_mov_b32_e32 v35, v28
	v_mov_b32_e32 v36, v28
	v_mov_b32_e32 v37, v28
	v_mov_b32_e32 v38, v28
	v_mov_b32_e32 v39, v28
	v_mov_b32_e32 v0, v28
	v_mov_b32_e32 v1, v28
	v_mov_b32_e32 v2, v28
	v_mov_b32_e32 v3, v28
	v_mov_b32_e32 v162, v136
	v_mov_b32_e32 v163, v132
.LBB0_3:
	s_mov_b32 s68, s9
	s_mov_b32 s9, s69
	s_mov_b32 s69, s35
	s_mov_b32 s35, s70
	v_add_u32_e32 v164, v119, v118
	v_add_u32_e32 v165, 0x80, v164
	s_add_i32 m0, s9, s37
	s_barrier
	buffer_load_dwordx4 v165, s[24:27], 0 offen lds
	v_add_u32_e32 v164, 0x2e080, v164
	s_add_i32 m0, s9, s38
	s_nop 0
	buffer_load_dwordx4 v164, s[24:27], 0 offen lds
	v_add_u32_e32 v164, v118, v147
	v_cndmask_b32_e64 v164, v120, v164, s[6:7]
	s_add_i32 m0, s9, s40
	s_nop 0
	buffer_load_dwordx4 v164, s[24:27], 0 offen lds
	v_add_u32_e32 v164, v118, v146
	v_cndmask_b32_e64 v164, v120, v164, s[2:3]
	s_add_i32 m0, s9, s41
	s_nop 0
	buffer_load_dwordx4 v164, s[24:27], 0 offen lds
	v_add_u32_e32 v164, v117, v145
	v_cmp_gt_i32_e32 vcc, 0, v164
	s_add_i32 m0, s70, s43
	s_nop 0
	v_cndmask_b32_e32 v165, v164, v144, vcc
	v_mul_hi_i32 v166, v165, s42
	v_add_u32_e32 v165, v166, v165
	v_lshrrev_b32_e32 v166, 31, v165
	v_ashrrev_i32_e32 v165, 9, v165
	v_add_u32_e32 v165, v165, v166
	v_ashrrev_i32_e32 v164, 31, v164
	v_xor_b32_e32 v164, v165, v164
	v_add_u32_e32 v165, s36, v164
	v_mul_i32_i24_e32 v165, 0xb80, v165
	v_mad_i32_i24 v164, v164, s60, v165
	v_add3_u32 v164, v143, v121, v164
	v_add_u32_e32 v165, 0x800, v164
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0xc000, v164
	s_add_i32 m0, s70, s44
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x17800, v164
	s_add_i32 m0, s70, s45
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x23000, v164
	s_add_i32 m0, s70, s46
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x2e800, v164
	s_add_i32 m0, s70, s47
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x3a000, v164
	s_add_i32 m0, s70, s48
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x45800, v164
	s_add_i32 m0, s70, s49
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x51000, v164
	s_add_i32 m0, s70, s50
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x5c800, v164
	s_add_i32 m0, s70, s51
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x68000, v164
	s_add_i32 m0, s70, s52
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x73800, v164
	s_add_i32 m0, s70, s53
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x7f000, v164
	s_add_i32 m0, s70, s54
	s_nop 0
	buffer_load_dword v165, s[20:23], 0 offen lds
	v_add_u32_e32 v165, 0x8a800, v164
	s_add_i32 m0, s70, s55
	v_add_u32_e32 v164, 0x96000, v164
	buffer_load_dword v165, s[20:23], 0 offen lds
	s_add_i32 m0, s70, s56
	s_nop 0
	buffer_load_dword v164, s[20:23], 0 offen lds
	v_add_u32_e32 v164, v117, v142
	v_add_u32_e32 v165, 64, v164
	v_subrev_u32_e32 v166, 46, v141
	v_cmp_gt_i32_e32 vcc, s57, v164
	v_add_u32_e32 v168, 18, v164
	v_add_u32_e32 v172, s1, v133
	v_cndmask_b32_e32 v165, v165, v166, vcc
	v_mul_hi_i32 v166, v165, s42
	v_add_u32_e32 v165, v166, v165
	v_lshrrev_b32_e32 v166, 31, v165
	v_ashrrev_i32_e32 v165, 5, v165
	v_add_u32_e32 v165, v165, v166
	v_cndmask_b32_e64 v166, 0, -1, vcc
	v_cmp_gt_i32_e32 vcc, s61, v164
	v_xor_b32_e32 v165, v165, v166
	v_add_u32_e32 v167, s8, v165
	v_cndmask_b32_e32 v164, v168, v141, vcc
	v_mul_hi_i32 v168, v164, s42
	v_add_u32_e32 v164, v168, v164
	v_lshrrev_b32_e32 v168, 31, v164
	v_ashrrev_i32_e32 v164, 5, v164
	v_add_u32_e32 v164, v164, v168
	v_cndmask_b32_e64 v168, 0, -1, vcc
	v_xor_b32_e32 v164, v164, v168
	v_add_u32_e32 v166, s15, v165
	v_mul_lo_u32 v167, v167, s14
	v_mul_lo_u32 v164, v164, s59
	v_add_u32_e32 v165, s0, v165
	v_sub_u32_e32 v167, v167, v164
	v_add_u32_e32 v168, v121, v140
	v_mul_lo_u32 v165, s14, v165
	v_add_u32_e32 v167, v168, v167
	v_cmp_gt_i32_e32 vcc, s39, v166
	v_sub_u32_e32 v164, v165, v164
	v_add_u32_e32 v164, v168, v164
	v_cndmask_b32_e32 v167, v120, v167, vcc
	v_cmp_gt_i32_e32 vcc, s58, v166
	v_add_u32_e32 v165, v121, v136
	v_add_u32_e32 v168, s1, v135
	v_cndmask_b32_e32 v164, v120, v164, vcc
	v_add_u32_e32 v166, 0x100, v165
	v_add_u32_e32 v169, 0xfffffeff, v168
	v_cmp_gt_i32_e32 vcc, s62, v165
	v_add_u32_e32 v171, 0xffffffb7, v168
	v_add_u32_e32 v168, 0xfffffefd, v168
	v_cndmask_b32_e32 v169, v166, v169, vcc
	v_mul_hi_i32 v170, v169, s42
	v_add_u32_e32 v169, v170, v169
	v_lshrrev_b32_e32 v170, 31, v169
	v_ashrrev_i32_e32 v169, 7, v169
	v_add_u32_e32 v169, v169, v170
	v_cndmask_b32_e64 v170, 0, -1, vcc
	v_xor_b32_e32 v176, v169, v170
	v_add_u32_e32 v170, 0x48, v165
	v_cmp_gt_i32_e32 vcc, s64, v165
	v_add_u32_e32 v172, 0xfffffefe, v172
	v_add_u32_e32 v178, v162, v121
	v_cndmask_b32_e32 v170, v170, v171, vcc
	v_mul_hi_i32 v171, v170, s42
	v_add_u32_e32 v170, v171, v170
	v_lshrrev_b32_e32 v171, 31, v170
	v_ashrrev_i32_e32 v170, 7, v170
	v_add_u32_e32 v170, v170, v171
	v_cndmask_b32_e64 v171, 0, -1, vcc
	v_xor_b32_e32 v170, v170, v171
	v_mul_lo_u32 v177, v170, s59
	v_add_u32_e32 v170, 0x102, v165
	v_cmp_gt_i32_e32 vcc, -2, v166
	v_add_u32_e32 v169, v176, v122
	v_mul_lo_u32 v169, v169, s34
	v_cndmask_b32_e32 v166, v170, v168, vcc
	v_mul_hi_i32 v171, v166, s42
	v_cmp_gt_i32_e32 vcc, s67, v165
	v_add_u32_e32 v166, v171, v166
	v_lshrrev_b32_e32 v171, 31, v166
	v_cndmask_b32_e32 v165, v170, v168, vcc
	v_ashrrev_i32_e32 v166, 7, v166
	v_mul_hi_i32 v168, v165, s42
	v_add_u32_e32 v166, v166, v171
	v_ashrrev_i32_e32 v171, 31, v170
	v_add_u32_e32 v165, v168, v165
	v_xor_b32_e32 v166, v166, v171
	v_lshrrev_b32_e32 v168, 31, v165
	v_ashrrev_i32_e32 v165, 7, v165
	v_add_u32_e32 v171, v166, v122
	v_add_u32_e32 v165, v165, v168
	v_cndmask_b32_e64 v168, 0, -1, vcc
	v_add_u32_e32 v170, v121, v132
	v_xor_b32_e32 v165, v165, v168
	v_mul_lo_u32 v168, v171, s34
	v_add_u32_e32 v171, 0x101, v170
	v_cmp_gt_i32_e32 vcc, s63, v170
	v_mul_lo_u32 v165, v165, s59
	v_sub_u32_e32 v169, v169, v177
	v_cndmask_b32_e32 v172, v171, v172, vcc
	v_mul_hi_i32 v173, v172, s42
	v_add_u32_e32 v172, v173, v172
	v_lshrrev_b32_e32 v173, 31, v172
	v_ashrrev_i32_e32 v172, 7, v172
	v_add_u32_e32 v172, v172, v173
	v_ashrrev_i32_e32 v171, 31, v171
	v_add_u32_e32 v173, s1, v134
	v_xor_b32_e32 v179, v172, v171
	v_add_u32_e32 v172, 0x49, v170
	v_add_u32_e32 v174, 0xffffffb6, v173
	v_cmp_gt_i32_e32 vcc, s65, v170
	v_add_u32_e32 v173, 0xfffffefc, v173
	v_add_u32_e32 v171, v179, v123
	v_cndmask_b32_e32 v174, v172, v174, vcc
	v_mul_hi_i32 v175, v174, s42
	v_add_u32_e32 v174, v175, v174
	v_lshrrev_b32_e32 v175, 31, v174
	v_ashrrev_i32_e32 v174, 7, v174
	v_add_u32_e32 v174, v174, v175
	v_ashrrev_i32_e32 v172, 31, v172
	v_xor_b32_e32 v172, v174, v172
	v_mul_lo_u32 v180, v172, s59
	v_add_u32_e32 v172, 0x103, v170
	v_cmp_gt_i32_e32 vcc, s66, v170
	v_add_u32_e32 v170, 0x100, v170
	v_mul_lo_u32 v171, v171, s34
	v_cndmask_b32_e32 v174, v172, v173, vcc
	v_mul_hi_i32 v175, v174, s42
	v_add_u32_e32 v174, v175, v174
	v_lshrrev_b32_e32 v175, 31, v174
	v_ashrrev_i32_e32 v174, 7, v174
	v_add_u32_e32 v174, v174, v175
	v_ashrrev_i32_e32 v175, 31, v172
	v_xor_b32_e32 v174, v174, v175
	v_add_u32_e32 v181, v174, v123
	v_mul_lo_u32 v181, v181, s34
	v_mul_lo_u32 v182, v174, s59
	v_sub_u32_e32 v174, v181, v182
	v_cmp_gt_i32_e32 vcc, -3, v170
	v_add3_u32 v181, v121, v174, v163
	v_add_u32_e32 v174, v137, v176
	v_cndmask_b32_e32 v170, v172, v173, vcc
	v_mul_lo_u32 v174, s34, v174
	v_mul_hi_i32 v172, v170, s42
	v_sub_u32_e32 v174, v174, v177
	v_add_u32_e32 v170, v172, v170
	v_add_u32_e32 v183, v178, v174
	v_add_u32_e32 v174, v137, v166
	v_lshrrev_b32_e32 v172, 31, v170
	v_ashrrev_i32_e32 v170, 7, v170
	v_mul_lo_u32 v174, s34, v174
	v_add_u32_e32 v170, v170, v172
	v_sub_u32_e32 v174, v174, v165
	v_xor_b32_e32 v186, v170, v175
	v_add3_u32 v184, v121, v174, v162
	v_add_u32_e32 v174, v179, v124
	v_add_u32_e32 v170, v186, v124
	v_sub_u32_e32 v168, v168, v165
	v_sub_u32_e32 v171, v171, v180
	v_mul_lo_u32 v174, v174, s34
	v_mul_lo_u32 v170, v170, s34
	v_add_u32_e32 v179, v179, v126
	v_add_u32_e32 v169, v178, v169
	v_add3_u32 v168, v121, v168, v162
	v_add3_u32 v171, v121, v171, v163
	v_sub_u32_e32 v174, v174, v180
	v_sub_u32_e32 v170, v170, v182
	v_mul_lo_u32 v179, v179, s34
	v_add3_u32 v185, v121, v174, v163
	v_add3_u32 v187, v121, v170, v163
	buffer_load_ubyte v175, v169, s[28:31], 0 offen offset:72
	buffer_load_ubyte v174, v168, s[28:31], 0 offen offset:258
	buffer_load_ubyte v173, v171, s[28:31], 0 offen offset:73
	buffer_load_ubyte v172, v181, s[28:31], 0 offen offset:259
	s_nop 0
	buffer_load_ubyte v171, v183, s[28:31], 0 offen offset:72
	buffer_load_ubyte v170, v184, s[28:31], 0 offen offset:258
	buffer_load_ubyte v169, v185, s[28:31], 0 offen offset:73
	buffer_load_ubyte v168, v187, s[28:31], 0 offen offset:259
	v_add_u32_e32 v181, v138, v176
	v_add_u32_e32 v183, v138, v166
	v_sub_u32_e32 v179, v179, v180
	v_add_u32_e32 v166, v139, v166
	v_mul_lo_u32 v181, s34, v181
	v_mul_lo_u32 v183, s34, v183
	v_add3_u32 v180, v121, v179, v163
	v_add_u32_e32 v179, v186, v126
	v_add_u32_e32 v176, v139, v176
	v_mul_lo_u32 v166, s34, v166
	v_sub_u32_e32 v181, v181, v177
	v_sub_u32_e32 v183, v183, v165
	v_mul_lo_u32 v179, v179, s34
	v_mul_lo_u32 v176, s34, v176
	v_sub_u32_e32 v165, v166, v165
	v_add_u32_e32 v181, v178, v181
	v_sub_u32_e32 v179, v179, v182
	v_sub_u32_e32 v176, v176, v177
	v_add3_u32 v165, v121, v165, v162
	v_add3_u32 v183, v121, v183, v162
	v_add3_u32 v182, v121, v179, v163
	v_add_u32_e32 v184, v178, v176
	buffer_load_ubyte v179, v181, s[28:31], 0 offen offset:72
	buffer_load_ubyte v178, v183, s[28:31], 0 offen offset:258
	buffer_load_ubyte v177, v180, s[28:31], 0 offen offset:73
	buffer_load_ubyte v176, v182, s[28:31], 0 offen offset:259
	buffer_load_ubyte v166, v184, s[28:31], 0 offen offset:72
	s_nop 0
	buffer_load_ubyte v165, v165, s[28:31], 0 offen offset:258
	s_nop 0
	buffer_load_dword v167, v167, s[16:19], 0 offen
	s_nop 0
	buffer_load_dword v164, v164, s[16:19], 0 offen
	v_add_u32_e32 v180, s68, v128
	ds_read_b128 v[182:185], v180
	ds_read_b128 v[186:189], v180 offset:2048
	ds_read_b128 v[190:193], v180 offset:4096
	ds_read_b128 v[194:197], v180 offset:6144
	v_add_u32_e32 v180, s69, v130
	ds_read_b128 v[198:201], v180
	ds_read_b128 v[202:205], v180 offset:2048
	ds_read_b128 v[206:209], v180 offset:4096
	ds_read_b128 v[210:213], v180 offset:6144
	ds_read_b128 v[214:217], v180 offset:8192
	ds_read_b128 v[218:221], v180 offset:10240
	ds_read_b128 v[222:225], v180 offset:12288
	s_barrier
	s_setprio 1
	v_and_b32_e32 v158, 0xff, v158
	v_and_b32_e32 v157, 0xff, v157
	v_and_b32_e32 v156, 0xff, v156
	v_and_b32_e32 v155, 0xff, v155
	v_and_b32_e32 v161, 0xff, v161
	v_and_b32_e32 v160, 0xff, v160
	v_and_b32_e32 v159, 0xff, v159
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[182:185], v[198:201], v[28:31], v127, v158 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[182:185], v[202:205], v[108:111], v127, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[182:185], v[206:209], v[104:107], v127, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[182:185], v[210:213], v[100:103], v127, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[182:185], v[214:217], v[96:99], v127, v161 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[182:185], v[218:221], v[92:95], v127, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[182:185], v[222:225], v[88:91], v127, v159 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[186:189], v[198:201], v[84:87], v127, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[186:189], v[202:205], v[80:83], v127, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[186:189], v[206:209], v[76:79], v127, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[186:189], v[210:213], v[72:75], v127, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[186:189], v[214:217], v[68:71], v127, v161 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[186:189], v[218:221], v[64:67], v127, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[186:189], v[222:225], v[60:63], v127, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[190:193], v[198:201], v[56:59], v125, v158 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[190:193], v[202:205], v[52:55], v125, v157 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[190:193], v[206:209], v[48:51], v125, v156 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[190:193], v[210:213], v[44:47], v125, v155 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[190:193], v[214:217], v[40:43], v125, v161 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[190:193], v[218:221], v[20:23], v125, v160 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[190:193], v[222:225], v[8:11], v125, v159 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[194:197], v[198:201], v[4:7], v125, v158 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[194:197], v[202:205], v[12:15], v125, v157 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[194:197], v[206:209], v[16:19], v125, v156 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[194:197], v[210:213], v[24:27], v125, v155 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[194:197], v[214:217], v[32:35], v125, v161 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[194:197], v[218:221], v[36:39], v125, v160 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[194:197], v[222:225], v[0:3], v125, v159 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v155, s68, v129
	ds_read_b128 v[156:159], v155
	ds_read_b128 v[180:183], v155 offset:2048
	ds_read_b128 v[184:187], v155 offset:4096
	ds_read_b128 v[188:191], v155 offset:6144
	v_add_u32_e32 v155, s69, v131
	ds_read_b128 v[192:195], v155
	ds_read_b128 v[196:199], v155 offset:2048
	ds_read_b128 v[200:203], v155 offset:4096
	ds_read_b128 v[204:207], v155 offset:6144
	ds_read_b128 v[208:211], v155 offset:8192
	ds_read_b128 v[212:215], v155 offset:10240
	ds_read_b128 v[216:219], v155 offset:12288
	s_waitcnt vmcnt(16)
	s_barrier
	s_setprio 1
	v_and_b32_e32 v151, 0xff, v151
	v_and_b32_e32 v150, 0xff, v150
	v_and_b32_e32 v149, 0xff, v149
	v_and_b32_e32 v148, 0xff, v148
	v_and_b32_e32 v154, 0xff, v154
	v_and_b32_e32 v153, 0xff, v153
	v_and_b32_e32 v152, 0xff, v152
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[156:159], v[192:195], v[28:31], v127, v151 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[156:159], v[196:199], v[108:111], v127, v150 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[156:159], v[200:203], v[104:107], v127, v149 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[100:103], v[156:159], v[204:207], v[100:103], v127, v148 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[156:159], v[208:211], v[96:99], v127, v154 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[156:159], v[212:215], v[92:95], v127, v153 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[156:159], v[216:219], v[88:91], v127, v152 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[180:183], v[192:195], v[84:87], v127, v151 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[180:183], v[196:199], v[80:83], v127, v150 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[180:183], v[200:203], v[76:79], v127, v149 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[180:183], v[204:207], v[72:75], v127, v148 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[180:183], v[208:211], v[68:71], v127, v154 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[180:183], v[212:215], v[64:67], v127, v153 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[180:183], v[216:219], v[60:63], v127, v152 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[184:187], v[192:195], v[56:59], v125, v151 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[184:187], v[196:199], v[52:55], v125, v150 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[184:187], v[200:203], v[48:51], v125, v149 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[184:187], v[204:207], v[44:47], v125, v148 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[184:187], v[208:211], v[40:43], v125, v154 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[184:187], v[212:215], v[20:23], v125, v153 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[184:187], v[216:219], v[8:11], v125, v152 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[188:191], v[192:195], v[4:7], v125, v151 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[188:191], v[196:199], v[12:15], v125, v150 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[188:191], v[200:203], v[16:19], v125, v149 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[188:191], v[204:207], v[24:27], v125, v148 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[32:35], v[188:191], v[208:211], v[32:35], v125, v154 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[36:39], v[188:191], v[212:215], v[36:39], v125, v153 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[188:191], v[216:219], v[0:3], v125, v152 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_addk_i32 s1, 0xff00
	v_add_u32_e32 v132, 0x100, v132
	v_add_u32_e32 v163, 0x100, v163
	v_add_u32_e32 v136, 0x100, v136
	v_add_u32_e32 v162, 0x100, v162
	v_add_u32_e32 v140, 0x100, v140
	v_subrev_u32_e32 v141, 64, v141
	v_add_u32_e32 v142, 64, v142
	v_add_u32_e32 v143, 0x800, v143
	v_add_u32_e32 v144, 0xfffffe00, v144
	v_add_u32_e32 v145, 0x200, v145
	v_add_u32_e32 v119, 0x80, v119
	v_add_u32_e32 v146, 0x80, v146
	v_add_u32_e32 v147, 0x80, v147
	s_cmpk_lg_i32 s1, 0xea00
	s_mov_b32 s70, s69
	s_mov_b32 s69, s68
	s_waitcnt vmcnt(2)
	v_mov_b32_e32 v152, v165
	v_mov_b32_e32 v159, v166
	v_mov_b32_e32 v153, v176
	v_mov_b32_e32 v160, v177
	v_mov_b32_e32 v154, v178
	v_mov_b32_e32 v161, v179
	v_mov_b32_e32 v148, v168
	v_mov_b32_e32 v155, v169
	v_mov_b32_e32 v149, v170
	v_mov_b32_e32 v156, v171
	v_mov_b32_e32 v150, v172
	v_mov_b32_e32 v157, v173
	v_mov_b32_e32 v151, v174
	v_mov_b32_e32 v158, v175
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v125, v164
	v_mov_b32_e32 v127, v167
	s_cbranch_scc1 .LBB0_3
	s_andn2_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	v_add_u32_e32 v117, s35, v130
	v_add_u32_e32 v118, s35, v131
	s_barrier
	ds_read_b128 v[154:157], v117
	ds_read_b128 v[158:161], v117 offset:2048
	ds_read_b128 v[180:183], v118
	ds_read_b128 v[130:133], v117 offset:12288
	ds_read_b128 v[184:187], v118 offset:2048
	ds_read_b128 v[188:191], v118 offset:4096
	ds_read_b128 v[192:195], v117 offset:4096
	ds_read_b128 v[196:199], v117 offset:6144
	ds_read_b128 v[200:203], v118 offset:6144
	ds_read_b128 v[204:207], v118 offset:8192
	ds_read_b128 v[208:211], v117 offset:8192
	ds_read_b128 v[212:215], v117 offset:10240
	ds_read_b128 v[216:219], v118 offset:10240
	ds_read_b128 v[134:137], v118 offset:12288
	v_add_u32_e32 v117, s9, v128
	ds_read_b128 v[220:223], v117
	ds_read_b128 v[224:227], v117 offset:2048
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[220:223], v[154:157], v[28:31], v167, v175 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v118, s9, v129
	ds_read_b128 v[228:231], v118
	ds_read_b128 v[232:235], v118 offset:2048
	ds_read_b128 v[236:239], v117 offset:4096
	ds_read_b128 v[142:145], v117 offset:6144
	ds_read_b128 v[240:243], v118 offset:4096
	ds_read_b128 v[138:141], v118 offset:6144
	s_movk_i32 s0, 0x7fff
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[228:231], v[180:183], v[28:31], v167, v174 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mov_b32_e32 v127, 0x7fc0
	s_cmpk_lt_u32 s15, 0x1880
	s_mul_hi_u32 s1, s12, s33
	v_mfma_scale_f32_16x16x128_f8f6f4 v[108:111], v[220:223], v[158:161], v[108:111], v167, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_movk_i32 s4, 0x187d
	s_nop 2
	v_bfe_u32 v117, v31, 16, 1
	v_bfe_u32 v122, v30, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[104:107], v[220:223], v[192:195], v[104:107], v167, v171 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v123, v29, 16, 1
	v_bfe_u32 v124, v28, 16, 1
	v_cmp_o_f32_e32 vcc, v31, v31
	v_mfma_scale_f32_16x16x128_f8f6f4 v[118:121], v[228:231], v[184:187], v[108:111], v167, v172 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 2
	v_add3_u32 v111, v31, v117, s0
	v_add3_u32 v110, v30, v122, s0
	v_lshrrev_b32_e32 v111, 16, v111
	v_add3_u32 v108, v28, v124, s0
	v_add3_u32 v109, v29, v123, s0
	v_lshrrev_b32_e32 v110, 16, v110
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], v[228:231], v[188:191], v[104:107], v167, v170 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v117, 16, v108
	s_nop 1
	v_cndmask_b32_e32 v104, v127, v111, vcc
	v_cmp_o_f32_e32 vcc, v30, v30
	v_lshrrev_b32_e32 v105, 16, v109
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], v[220:223], v[196:199], v[100:103], v167, v169 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_cndmask_b32_e32 v101, v127, v110, vcc
	v_cmp_o_f32_e32 vcc, v29, v29
	v_bfe_u32 v100, v121, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[220:223], v[212:215], v[92:95], v167, v177 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v105, v127, v105, vcc
	v_cmp_o_f32_e32 vcc, v28, v28
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[220:223], v[208:211], v[96:99], v167, v179 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v111, v127, v117, vcc
	v_cmp_o_f32_e32 vcc, v121, v121
	v_bfe_u32 v96, v120, 16, 1
	v_add3_u32 v99, v121, v100, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[220:223], v[130:133], v[88:91], v167, v166 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v97, v119, 16, 1
	v_add3_u32 v96, v120, v96, s0
	v_lshrrev_b32_e32 v99, 16, v99
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[224:227], v[154:157], v[84:87], v167, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v98, v118, 16, 1
	v_add3_u32 v97, v119, v97, s0
	v_lshrrev_b32_e32 v96, 16, v96
	v_cndmask_b32_e32 v100, v127, v99, vcc
	v_cmp_o_f32_e32 vcc, v120, v120
	v_add3_u32 v98, v118, v98, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], v[228:231], v[216:219], v[92:95], v167, v176 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v102, v127, v96, vcc
	v_cmp_o_f32_e32 vcc, v119, v119
	s_nop 0
	v_lshrrev_b32_e32 v92, 16, v97
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], v[228:231], v[200:203], v[106:109], v167, v168 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v93, 16, v98
	s_nop 1
	v_cndmask_b32_e32 v106, v127, v92, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], v[228:231], v[134:137], v[88:91], v167, v165 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v118, v118
	s_nop 1
	v_bfe_u32 v88, v125, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[224:227], v[158:161], v[80:83], v167, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v109, v127, v93, vcc
	v_bfe_u32 v89, v124, 16, 1
	v_bfe_u32 v90, v123, 16, 1
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[92:95], v[232:235], v[180:183], v[84:87], v167, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v125, v125
	v_bfe_u32 v91, v122, 16, 1
	s_nop 0
	v_add3_u32 v87, v125, v88, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[224:227], v[192:195], v[76:79], v167, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v86, v124, v89, s0
	v_lshrrev_b32_e32 v87, 16, v87
	v_add3_u32 v85, v123, v90, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[224:227], v[196:199], v[72:75], v167, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v86, 16, v86
	v_cndmask_b32_e32 v103, v127, v87, vcc
	v_cmp_o_f32_e32 vcc, v124, v124
	v_add3_u32 v84, v122, v91, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[88:91], v[232:235], v[184:187], v[80:83], v167, v172 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v107, v127, v86, vcc
	v_cmp_o_f32_e32 vcc, v123, v123
	s_nop 0
	v_lshrrev_b32_e32 v80, 16, v85
	v_lshrrev_b32_e32 v81, 16, v84
	v_cndmask_b32_e32 v110, v127, v80, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[84:87], v[232:235], v[188:191], v[76:79], v167, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v122, v122
	s_nop 1
	v_bfe_u32 v76, v149, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[224:227], v[208:211], v[68:71], v167, v179 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v119, v127, v81, vcc
	v_bfe_u32 v77, v148, 16, 1
	v_bfe_u32 v78, v147, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[228:231], v[204:207], v[28:31], v167, v178 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v149, v149
	v_bfe_u32 v79, v146, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[80:83], v[232:235], v[200:203], v[72:75], v167, v168 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 2
	v_add3_u32 v75, v149, v76, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[224:227], v[212:215], v[64:67], v167, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add3_u32 v74, v148, v77, s0
	v_lshrrev_b32_e32 v75, 16, v75
	v_add3_u32 v73, v147, v78, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[224:227], v[130:133], v[60:63], v167, v166 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v74, 16, v74
	v_cndmask_b32_e32 v108, v127, v75, vcc
	v_cmp_o_f32_e32 vcc, v148, v148
	v_add3_u32 v72, v146, v79, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[76:79], v[232:235], v[204:207], v[68:71], v167, v178 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v117, v127, v74, vcc
	v_cmp_o_f32_e32 vcc, v147, v147
	s_nop 0
	v_lshrrev_b32_e32 v68, 16, v73
	v_lshrrev_b32_e32 v69, 16, v72
	v_cndmask_b32_e32 v120, v127, v68, vcc
	v_mfma_scale_f32_16x16x128_f8f6f4 v[72:75], v[232:235], v[216:219], v[64:67], v167, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v146, v146
	s_nop 1
	v_bfe_u32 v64, v31, 16, 1
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[236:239], v[154:157], v[56:59], v164, v175 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v123, v127, v69, vcc
	v_bfe_u32 v65, v30, 16, 1
	v_bfe_u32 v66, v29, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[68:71], v[232:235], v[134:137], v[60:63], v167, v165 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cmp_o_f32_e32 vcc, v31, v31
	v_bfe_u32 v67, v28, 16, 1
	s_nop 0
	v_add3_u32 v63, v31, v64, s0
	v_add3_u32 v62, v30, v65, s0
	v_lshrrev_b32_e32 v63, 16, v63
	v_add3_u32 v61, v29, v66, s0
	v_lshrrev_b32_e32 v62, 16, v62
	v_cndmask_b32_e32 v118, v127, v63, vcc
	v_cmp_o_f32_e32 vcc, v30, v30
	v_add3_u32 v60, v28, v67, s0
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[64:67], v[240:243], v[180:183], v[56:59], v164, v174 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v121, v127, v62, vcc
	v_cmp_o_f32_e32 vcc, v29, v29
	s_nop 0
	v_lshrrev_b32_e32 v56, 16, v61
	v_cndmask_b32_e32 v124, v127, v56, vcc
	v_cmp_o_f32_e32 vcc, v28, v28
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[236:239], v[192:195], v[48:51], v164, v171 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v57, 16, v60
	v_cndmask_b32_e32 v126, v127, v57, vcc
	v_cmp_o_f32_e32 vcc, v153, v153
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[236:239], v[158:161], v[52:55], v164, v173 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_bfe_u32 v48, v152, 16, 1
	v_bfe_u32 v49, v151, 16, 1
	v_add3_u32 v48, v152, v48, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[56:59], v[240:243], v[188:191], v[28:31], v164, v170 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v50, v150, 16, 1
	v_add3_u32 v49, v151, v49, s0
	v_add3_u32 v50, v150, v50, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[236:239], v[196:199], v[44:47], v164, v169 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[60:63], v[240:243], v[184:187], v[52:55], v164, v172 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 1
	v_lshrrev_b32_e32 v45, 16, v48
	v_lshrrev_b32_e32 v46, 16, v49
	v_lshrrev_b32_e32 v47, 16, v50
	v_bfe_u32 v52, v153, 16, 1
	v_add3_u32 v44, v153, v52, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[52:55], v[240:243], v[200:203], v[28:31], v164, v168 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v44, 16, v44
	v_cndmask_b32_e32 v122, v127, v44, vcc
	v_cmp_o_f32_e32 vcc, v152, v152
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[236:239], v[208:211], v[40:43], v164, v179 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v125, v127, v45, vcc
	v_cmp_o_f32_e32 vcc, v151, v151
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[236:239], v[212:215], v[20:23], v164, v177 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v128, v127, v46, vcc
	v_cmp_o_f32_e32 vcc, v150, v150
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[142:145], v[154:157], v[4:7], v164, v175 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v129, v127, v47, vcc
	v_cmp_o_f32_e32 vcc, v99, v99
	v_mfma_scale_f32_16x16x128_f8f6f4 v[48:51], v[240:243], v[204:207], v[28:31], v164, v178 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 2
	v_bfe_u32 v28, v99, 16, 1
	v_bfe_u32 v29, v98, 16, 1
	v_bfe_u32 v30, v97, 16, 1
	v_bfe_u32 v31, v96, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[44:47], v[240:243], v[216:219], v[20:23], v164, v176 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_add3_u32 v146, v96, v31, s0
	v_add3_u32 v147, v97, v30, s0
	s_nop 0
	v_add3_u32 v20, v98, v29, s0
	v_add3_u32 v21, v99, v28, s0
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[138:141], v[180:183], v[4:7], v164, v174 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v148, 16, v21
	v_lshrrev_b32_e32 v149, 16, v20
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[142:145], v[158:161], v[12:15], v164, v173 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[138:141], v[184:187], v[4:7], v164, v172 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[142:145], v[192:195], v[16:19], v164, v171 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[138:141], v[188:191], v[4:7], v164, v170 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[142:145], v[196:199], v[24:27], v164, v169 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[236:239], v[130:133], v[8:11], v164, v166 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 1
	v_lshrrev_b32_e32 v24, 16, v147
	v_lshrrev_b32_e32 v25, 16, v146
	v_bfe_u32 v26, v93, 16, 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[12:15], v[138:141], v[200:203], v[4:7], v164, v168 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v27, v92, 16, 1
	v_add3_u32 v26, v93, v26, s0
	v_add3_u32 v27, v92, v27, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[142:145], v[208:211], v[32:35], v164, v179 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshrrev_b32_e32 v26, 16, v26
	v_lshrrev_b32_e32 v27, 16, v27
	v_mfma_scale_f32_16x16x128_f8f6f4 v[40:43], v[240:243], v[134:137], v[8:11], v164, v165 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v32, v88, 16, 1
	v_add3_u32 v32, v88, v32, s0
	v_lshrrev_b32_e32 v34, 16, v32
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[138:141], v[204:207], v[4:7], v164, v178 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_bfe_u32 v35, v84, 16, 1
	v_add3_u32 v35, v84, v35, s0
	v_lshrrev_b32_e32 v35, 16, v35
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[142:145], v[212:215], v[36:39], v164, v177 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 2
	v_cndmask_b32_e32 v37, v127, v148, vcc
	v_cmp_o_f32_e32 vcc, v98, v98
	v_bfe_u32 v39, v80, 16, 1
	v_add3_u32 v39, v80, v39, s0
	v_cndmask_b32_e32 v98, v127, v149, vcc
	v_cmp_o_f32_e32 vcc, v97, v97
	v_lshrrev_b32_e32 v39, 16, v39
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[138:141], v[216:219], v[4:7], v164, v176 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v97, v127, v24, vcc
	v_cmp_o_f32_e32 vcc, v96, v96
	v_bfe_u32 v24, v95, 16, 1
	v_add3_u32 v24, v95, v24, s0
	v_cndmask_b32_e32 v96, v127, v25, vcc
	v_bfe_u32 v25, v94, 16, 1
	v_add3_u32 v25, v94, v25, s0
	v_lshrrev_b32_e32 v24, 16, v24
	v_cmp_o_f32_e32 vcc, v95, v95
	v_lshrrev_b32_e32 v25, 16, v25
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[142:145], v[130:133], v[0:3], v164, v166 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_cndmask_b32_e32 v24, v127, v24, vcc
	v_cmp_o_f32_e32 vcc, v94, v94
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[138:141], v[134:137], v[0:3], v164, v165 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_cndmask_b32_e32 v33, v127, v25, vcc
	v_cmp_o_f32_e32 vcc, v93, v93
	v_bfe_u32 v25, v91, 16, 1
	v_add3_u32 v25, v91, v25, s0
	v_cndmask_b32_e32 v93, v127, v26, vcc
	v_cmp_o_f32_e32 vcc, v92, v92
	v_bfe_u32 v26, v90, 16, 1
	v_add3_u32 v26, v90, v26, s0
	v_cndmask_b32_e32 v92, v127, v27, vcc
	v_bfe_u32 v27, v89, 16, 1
	v_lshrrev_b32_e32 v25, 16, v25
	v_cmp_o_f32_e32 vcc, v91, v91
	v_add3_u32 v27, v89, v27, s0
	v_lshrrev_b32_e32 v26, 16, v26
	v_cndmask_b32_e32 v25, v127, v25, vcc
	v_cmp_o_f32_e32 vcc, v90, v90
	v_lshrrev_b32_e32 v27, 16, v27
	s_nop 0
	v_cndmask_b32_e32 v32, v127, v26, vcc
	v_cmp_o_f32_e32 vcc, v89, v89
	v_bfe_u32 v26, v87, 16, 1
	v_add3_u32 v26, v87, v26, s0
	v_cndmask_b32_e32 v38, v127, v27, vcc
	v_cmp_o_f32_e32 vcc, v88, v88
	v_bfe_u32 v27, v86, 16, 1
	v_add3_u32 v27, v86, v27, s0
	v_cndmask_b32_e32 v88, v127, v34, vcc
	v_bfe_u32 v34, v85, 16, 1
	v_lshrrev_b32_e32 v26, 16, v26
	v_cmp_o_f32_e32 vcc, v87, v87
	v_add3_u32 v34, v85, v34, s0
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v26, v127, v26, vcc
	v_cmp_o_f32_e32 vcc, v86, v86
	v_lshrrev_b32_e32 v34, 16, v34
	s_nop 0
	v_cndmask_b32_e32 v36, v127, v27, vcc
	v_cmp_o_f32_e32 vcc, v85, v85
	v_bfe_u32 v27, v83, 16, 1
	v_add3_u32 v27, v83, v27, s0
	v_cndmask_b32_e32 v85, v127, v34, vcc
	v_cmp_o_f32_e32 vcc, v84, v84
	v_bfe_u32 v34, v82, 16, 1
	v_add3_u32 v34, v82, v34, s0
	v_cndmask_b32_e32 v84, v127, v35, vcc
	v_bfe_u32 v35, v81, 16, 1
	v_lshrrev_b32_e32 v27, 16, v27
	v_cmp_o_f32_e32 vcc, v83, v83
	v_add3_u32 v35, v81, v35, s0
	v_lshrrev_b32_e32 v86, 16, v34
	v_cndmask_b32_e32 v34, v127, v27, vcc
	v_cmp_o_f32_e32 vcc, v82, v82
	v_lshrrev_b32_e32 v35, 16, v35
	v_bfe_u32 v27, v79, 16, 1
	v_cndmask_b32_e32 v82, v127, v86, vcc
	v_cmp_o_f32_e32 vcc, v81, v81
	v_add3_u32 v27, v79, v27, s0
	v_lshrrev_b32_e32 v27, 16, v27
	v_cndmask_b32_e32 v81, v127, v35, vcc
	v_cmp_o_f32_e32 vcc, v80, v80
	v_bfe_u32 v35, v78, 16, 1
	v_add3_u32 v35, v78, v35, s0
	v_cndmask_b32_e32 v80, v127, v39, vcc
	v_bfe_u32 v39, v77, 16, 1
	v_add3_u32 v39, v77, v39, s0
	v_cmp_o_f32_e32 vcc, v79, v79
	v_bfe_u32 v83, v76, 16, 1
	v_lshrrev_b32_e32 v35, 16, v35
	v_lshrrev_b32_e32 v86, 16, v39
	v_cndmask_b32_e32 v39, v127, v27, vcc
	v_cmp_o_f32_e32 vcc, v78, v78
	v_add3_u32 v83, v76, v83, s0
	v_bfe_u32 v27, v75, 16, 1
	v_cndmask_b32_e32 v78, v127, v35, vcc
	v_cmp_o_f32_e32 vcc, v77, v77
	v_lshrrev_b32_e32 v83, 16, v83
	v_bfe_u32 v35, v74, 16, 1
	v_cndmask_b32_e32 v77, v127, v86, vcc
	v_cmp_o_f32_e32 vcc, v76, v76
	v_add3_u32 v27, v75, v27, s0
	v_bfe_u32 v79, v73, 16, 1
	v_cndmask_b32_e32 v76, v127, v83, vcc
	v_add3_u32 v35, v74, v35, s0
	v_lshrrev_b32_e32 v27, 16, v27
	v_cmp_o_f32_e32 vcc, v75, v75
	v_bfe_u32 v83, v72, 16, 1
	v_add3_u32 v79, v73, v79, s0
	v_lshrrev_b32_e32 v35, 16, v35
	v_cndmask_b32_e32 v75, v127, v27, vcc
	v_cmp_o_f32_e32 vcc, v74, v74
	v_add3_u32 v83, v72, v83, s0
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v74, v127, v35, vcc
	v_cmp_o_f32_e32 vcc, v73, v73
	v_bfe_u32 v27, v71, 16, 1
	v_lshrrev_b32_e32 v83, 16, v83
	v_cndmask_b32_e32 v73, v127, v79, vcc
	v_cmp_o_f32_e32 vcc, v72, v72
	v_bfe_u32 v35, v70, 16, 1
	v_add3_u32 v27, v71, v27, s0
	v_cndmask_b32_e32 v72, v127, v83, vcc
	v_bfe_u32 v79, v69, 16, 1
	v_add3_u32 v35, v70, v35, s0
	v_lshrrev_b32_e32 v27, 16, v27
	v_cmp_o_f32_e32 vcc, v71, v71
	v_bfe_u32 v83, v68, 16, 1
	v_add3_u32 v79, v69, v79, s0
	v_lshrrev_b32_e32 v35, 16, v35
	v_cndmask_b32_e32 v71, v127, v27, vcc
	v_cmp_o_f32_e32 vcc, v70, v70
	v_add3_u32 v83, v68, v83, s0
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v70, v127, v35, vcc
	v_cmp_o_f32_e32 vcc, v69, v69
	v_bfe_u32 v27, v67, 16, 1
	v_lshrrev_b32_e32 v83, 16, v83
	v_cndmask_b32_e32 v69, v127, v79, vcc
	v_cmp_o_f32_e32 vcc, v68, v68
	v_bfe_u32 v35, v66, 16, 1
	v_add3_u32 v27, v67, v27, s0
	v_cndmask_b32_e32 v68, v127, v83, vcc
	v_bfe_u32 v79, v65, 16, 1
	v_add3_u32 v35, v66, v35, s0
	v_lshrrev_b32_e32 v27, 16, v27
	v_cmp_o_f32_e32 vcc, v67, v67
	v_bfe_u32 v83, v64, 16, 1
	v_add3_u32 v79, v65, v79, s0
	v_lshrrev_b32_e32 v35, 16, v35
	v_cndmask_b32_e32 v27, v127, v27, vcc
	v_cmp_o_f32_e32 vcc, v66, v66
	v_add3_u32 v83, v64, v83, s0
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v66, v127, v35, vcc
	v_cmp_o_f32_e32 vcc, v65, v65
	v_bfe_u32 v35, v63, 16, 1
	v_lshrrev_b32_e32 v83, 16, v83
	v_cndmask_b32_e32 v65, v127, v79, vcc
	v_cmp_o_f32_e32 vcc, v64, v64
	v_bfe_u32 v67, v62, 16, 1
	v_add3_u32 v35, v63, v35, s0
	v_cndmask_b32_e32 v64, v127, v83, vcc
	v_bfe_u32 v79, v61, 16, 1
	v_add3_u32 v67, v62, v67, s0
	v_lshrrev_b32_e32 v35, 16, v35
	v_cmp_o_f32_e32 vcc, v63, v63
	v_bfe_u32 v83, v60, 16, 1
	v_add3_u32 v79, v61, v79, s0
	v_lshrrev_b32_e32 v67, 16, v67
	v_cndmask_b32_e32 v35, v127, v35, vcc
	v_cmp_o_f32_e32 vcc, v62, v62
	v_add3_u32 v83, v60, v83, s0
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v62, v127, v67, vcc
	v_cmp_o_f32_e32 vcc, v61, v61
	v_bfe_u32 v63, v59, 16, 1
	v_lshrrev_b32_e32 v83, 16, v83
	v_cndmask_b32_e32 v61, v127, v79, vcc
	v_cmp_o_f32_e32 vcc, v60, v60
	v_bfe_u32 v67, v58, 16, 1
	v_add3_u32 v63, v59, v63, s0
	v_cndmask_b32_e32 v60, v127, v83, vcc
	v_bfe_u32 v79, v57, 16, 1
	v_add3_u32 v67, v58, v67, s0
	v_lshrrev_b32_e32 v63, 16, v63
	v_cmp_o_f32_e32 vcc, v59, v59
	v_bfe_u32 v83, v56, 16, 1
	v_add3_u32 v79, v57, v79, s0
	v_lshrrev_b32_e32 v67, 16, v67
	v_cndmask_b32_e32 v59, v127, v63, vcc
	v_cmp_o_f32_e32 vcc, v58, v58
	v_add3_u32 v83, v56, v83, s0
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v58, v127, v67, vcc
	v_cmp_o_f32_e32 vcc, v57, v57
	v_bfe_u32 v63, v55, 16, 1
	v_lshrrev_b32_e32 v83, 16, v83
	v_cndmask_b32_e32 v57, v127, v79, vcc
	v_cmp_o_f32_e32 vcc, v56, v56
	v_bfe_u32 v67, v54, 16, 1
	v_add3_u32 v63, v55, v63, s0
	v_cndmask_b32_e32 v56, v127, v83, vcc
	v_bfe_u32 v79, v53, 16, 1
	v_add3_u32 v67, v54, v67, s0
	v_lshrrev_b32_e32 v63, 16, v63
	v_cmp_o_f32_e32 vcc, v55, v55
	v_bfe_u32 v83, v52, 16, 1
	v_add3_u32 v79, v53, v79, s0
	v_lshrrev_b32_e32 v67, 16, v67
	v_cndmask_b32_e32 v55, v127, v63, vcc
	v_cmp_o_f32_e32 vcc, v54, v54
	v_add3_u32 v83, v52, v83, s0
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v54, v127, v67, vcc
	v_cmp_o_f32_e32 vcc, v53, v53
	v_bfe_u32 v63, v51, 16, 1
	v_lshrrev_b32_e32 v83, 16, v83
	v_cndmask_b32_e32 v53, v127, v79, vcc
	v_cmp_o_f32_e32 vcc, v52, v52
	v_bfe_u32 v67, v50, 16, 1
	v_add3_u32 v63, v51, v63, s0
	v_cndmask_b32_e32 v52, v127, v83, vcc
	v_bfe_u32 v79, v49, 16, 1
	v_add3_u32 v67, v50, v67, s0
	v_lshrrev_b32_e32 v63, 16, v63
	v_cmp_o_f32_e32 vcc, v51, v51
	v_bfe_u32 v83, v48, 16, 1
	v_add3_u32 v79, v49, v79, s0
	v_lshrrev_b32_e32 v67, 16, v67
	v_cndmask_b32_e32 v51, v127, v63, vcc
	v_cmp_o_f32_e32 vcc, v50, v50
	v_add3_u32 v83, v48, v83, s0
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v50, v127, v67, vcc
	v_cmp_o_f32_e32 vcc, v49, v49
	v_bfe_u32 v63, v47, 16, 1
	v_lshrrev_b32_e32 v83, 16, v83
	v_cndmask_b32_e32 v49, v127, v79, vcc
	v_cmp_o_f32_e32 vcc, v48, v48
	v_bfe_u32 v67, v46, 16, 1
	v_add3_u32 v63, v47, v63, s0
	v_cndmask_b32_e32 v48, v127, v83, vcc
	v_bfe_u32 v79, v45, 16, 1
	v_add3_u32 v67, v46, v67, s0
	v_lshrrev_b32_e32 v63, 16, v63
	v_cmp_o_f32_e32 vcc, v47, v47
	v_bfe_u32 v83, v44, 16, 1
	v_add3_u32 v79, v45, v79, s0
	v_lshrrev_b32_e32 v67, 16, v67
	v_cndmask_b32_e32 v47, v127, v63, vcc
	v_cmp_o_f32_e32 vcc, v46, v46
	v_add3_u32 v83, v44, v83, s0
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v46, v127, v67, vcc
	v_cmp_o_f32_e32 vcc, v45, v45
	v_bfe_u32 v63, v43, 16, 1
	v_lshrrev_b32_e32 v83, 16, v83
	v_cndmask_b32_e32 v45, v127, v79, vcc
	v_cmp_o_f32_e32 vcc, v44, v44
	v_bfe_u32 v67, v42, 16, 1
	v_add3_u32 v63, v43, v63, s0
	v_cndmask_b32_e32 v44, v127, v83, vcc
	v_bfe_u32 v79, v41, 16, 1
	v_add3_u32 v67, v42, v67, s0
	v_lshrrev_b32_e32 v63, 16, v63
	v_cmp_o_f32_e32 vcc, v43, v43
	v_bfe_u32 v83, v40, 16, 1
	v_add3_u32 v79, v41, v79, s0
	v_lshrrev_b32_e32 v67, 16, v67
	v_cndmask_b32_e32 v43, v127, v63, vcc
	v_cmp_o_f32_e32 vcc, v42, v42
	v_add3_u32 v83, v40, v83, s0
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v42, v127, v67, vcc
	v_cmp_o_f32_e32 vcc, v41, v41
	v_bfe_u32 v63, v31, 16, 1
	v_lshrrev_b32_e32 v83, 16, v83
	v_cndmask_b32_e32 v41, v127, v79, vcc
	v_cmp_o_f32_e32 vcc, v40, v40
	v_bfe_u32 v67, v30, 16, 1
	v_add3_u32 v63, v31, v63, s0
	v_cndmask_b32_e32 v40, v127, v83, vcc
	v_bfe_u32 v79, v29, 16, 1
	v_add3_u32 v67, v30, v67, s0
	v_lshrrev_b32_e32 v63, 16, v63
	v_cmp_o_f32_e32 vcc, v31, v31
	v_bfe_u32 v83, v28, 16, 1
	v_add3_u32 v79, v29, v79, s0
	v_lshrrev_b32_e32 v67, 16, v67
	v_cndmask_b32_e32 v31, v127, v63, vcc
	v_cmp_o_f32_e32 vcc, v30, v30
	v_add3_u32 v83, v28, v83, s0
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v30, v127, v67, vcc
	v_cmp_o_f32_e32 vcc, v29, v29
	v_bfe_u32 v63, v23, 16, 1
	v_lshrrev_b32_e32 v83, 16, v83
	v_cndmask_b32_e32 v29, v127, v79, vcc
	v_cmp_o_f32_e32 vcc, v28, v28
	v_bfe_u32 v67, v22, 16, 1
	v_add3_u32 v63, v23, v63, s0
	v_cndmask_b32_e32 v28, v127, v83, vcc
	v_bfe_u32 v79, v21, 16, 1
	v_add3_u32 v67, v22, v67, s0
	v_lshrrev_b32_e32 v63, 16, v63
	v_cmp_o_f32_e32 vcc, v23, v23
	v_bfe_u32 v83, v20, 16, 1
	v_add3_u32 v79, v21, v79, s0
	v_lshrrev_b32_e32 v67, 16, v67
	v_cndmask_b32_e32 v23, v127, v63, vcc
	v_cmp_o_f32_e32 vcc, v22, v22
	v_add3_u32 v83, v20, v83, s0
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v22, v127, v67, vcc
	v_cmp_o_f32_e32 vcc, v21, v21
	v_bfe_u32 v63, v19, 16, 1
	v_lshrrev_b32_e32 v83, 16, v83
	v_cndmask_b32_e32 v21, v127, v79, vcc
	v_cmp_o_f32_e32 vcc, v20, v20
	v_bfe_u32 v67, v18, 16, 1
	v_add3_u32 v63, v19, v63, s0
	v_cndmask_b32_e32 v20, v127, v83, vcc
	v_bfe_u32 v79, v17, 16, 1
	v_add3_u32 v67, v18, v67, s0
	v_lshrrev_b32_e32 v63, 16, v63
	v_cmp_o_f32_e32 vcc, v19, v19
	v_bfe_u32 v83, v16, 16, 1
	v_add3_u32 v79, v17, v79, s0
	v_lshrrev_b32_e32 v67, 16, v67
	v_cndmask_b32_e32 v19, v127, v63, vcc
	v_cmp_o_f32_e32 vcc, v18, v18
	v_add3_u32 v83, v16, v83, s0
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v18, v127, v67, vcc
	v_cmp_o_f32_e32 vcc, v17, v17
	v_bfe_u32 v63, v15, 16, 1
	v_lshrrev_b32_e32 v83, 16, v83
	v_cndmask_b32_e32 v17, v127, v79, vcc
	v_cmp_o_f32_e32 vcc, v16, v16
	v_bfe_u32 v67, v14, 16, 1
	v_add3_u32 v63, v15, v63, s0
	v_cndmask_b32_e32 v16, v127, v83, vcc
	v_bfe_u32 v79, v13, 16, 1
	v_add3_u32 v67, v14, v67, s0
	v_lshrrev_b32_e32 v63, 16, v63
	v_cmp_o_f32_e32 vcc, v15, v15
	v_bfe_u32 v83, v12, 16, 1
	v_add3_u32 v79, v13, v79, s0
	v_lshrrev_b32_e32 v67, 16, v67
	v_cndmask_b32_e32 v15, v127, v63, vcc
	v_cmp_o_f32_e32 vcc, v14, v14
	v_add3_u32 v83, v12, v83, s0
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v14, v127, v67, vcc
	v_cmp_o_f32_e32 vcc, v13, v13
	v_bfe_u32 v63, v11, 16, 1
	v_lshrrev_b32_e32 v83, 16, v83
	v_cndmask_b32_e32 v13, v127, v79, vcc
	v_cmp_o_f32_e32 vcc, v12, v12
	v_bfe_u32 v67, v10, 16, 1
	v_add3_u32 v63, v11, v63, s0
	v_cndmask_b32_e32 v12, v127, v83, vcc
	v_bfe_u32 v79, v9, 16, 1
	v_add3_u32 v67, v10, v67, s0
	v_lshrrev_b32_e32 v63, 16, v63
	v_cmp_o_f32_e32 vcc, v11, v11
	v_bfe_u32 v83, v8, 16, 1
	v_add3_u32 v79, v9, v79, s0
	v_lshrrev_b32_e32 v67, 16, v67
	v_cndmask_b32_e32 v11, v127, v63, vcc
	v_cmp_o_f32_e32 vcc, v10, v10
	v_add3_u32 v83, v8, v83, s0
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v10, v127, v67, vcc
	v_cmp_o_f32_e32 vcc, v9, v9
	v_bfe_u32 v63, v7, 16, 1
	v_lshrrev_b32_e32 v83, 16, v83
	v_cndmask_b32_e32 v9, v127, v79, vcc
	v_cmp_o_f32_e32 vcc, v8, v8
	v_bfe_u32 v67, v6, 16, 1
	v_add3_u32 v63, v7, v63, s0
	v_cndmask_b32_e32 v8, v127, v83, vcc
	v_bfe_u32 v79, v5, 16, 1
	v_add3_u32 v67, v6, v67, s0
	v_lshrrev_b32_e32 v63, 16, v63
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v83, v4, 16, 1
	v_add3_u32 v79, v5, v79, s0
	v_lshrrev_b32_e32 v67, 16, v67
	v_cndmask_b32_e32 v7, v127, v63, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v83, v4, v83, s0
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v6, v127, v67, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_bfe_u32 v63, v3, 16, 1
	v_lshrrev_b32_e32 v83, 16, v83
	v_cndmask_b32_e32 v5, v127, v79, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v67, v2, 16, 1
	v_add3_u32 v63, v3, v63, s0
	v_cndmask_b32_e32 v4, v127, v83, vcc
	v_bfe_u32 v79, v1, 16, 1
	v_add3_u32 v67, v2, v67, s0
	v_lshrrev_b32_e32 v63, 16, v63
	v_cmp_o_f32_e32 vcc, v3, v3
	v_bfe_u32 v83, v0, 16, 1
	v_add3_u32 v79, v1, v79, s0
	v_lshrrev_b32_e32 v67, 16, v67
	v_cndmask_b32_e32 v3, v127, v63, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	v_add3_u32 v83, v0, v83, s0
	v_lshrrev_b32_e32 v79, 16, v79
	v_cndmask_b32_e32 v2, v127, v67, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	v_lshrrev_b32_e32 v83, 16, v83
	s_mul_i32 s0, s13, s33
	v_cndmask_b32_e32 v1, v127, v79, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	v_lshlrev_b32_e32 v63, 2, v116
	v_or_b32_e32 v79, s8, v63
	v_cndmask_b32_e32 v0, v127, v83, vcc
	s_cselect_b64 vcc, -1, 0
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s12, s33
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s10, s0
	s_addc_u32 s1, s11, s1
	s_lshl_b32 s2, s36, 1
	s_add_u32 s8, s0, s2
	v_mul_lo_u32 v83, s12, v79
	s_addc_u32 s0, s1, 0
	s_and_b32 s1, s12, 0x3fff
	v_or_b32_e32 v67, s15, v63
	v_add_u32_e32 v86, v83, v112
	v_not_b32_e32 v63, -2.0
	s_lshl_b32 s1, s1, 16
	s_and_b32 s0, s0, 0xffff
	s_or_b32 s0, s1, s0
	v_cndmask_b32_e32 v86, v63, v86, vcc
	s_or_b32 s9, s0, 2.0
	s_mov_b32 s11, 0x27000
	s_mov_b32 s10, 0x7ffffffd
	v_lshlrev_b32_e32 v86, 1, v86
	buffer_store_short v111, v86, s[8:11], 0 offen
	v_or_b32_e32 v86, 1, v79
	s_movk_i32 s0, 0x187f
	v_mul_lo_u32 v86, s12, v86
	v_add_u32_e32 v87, v86, v112
	v_cmp_gt_u32_e64 s[0:1], s0, v67
	s_movk_i32 s2, 0x187e
	v_cmp_gt_u32_e64 s[2:3], s2, v67
	v_cndmask_b32_e64 v87, v63, v87, s[0:1]
	v_lshlrev_b32_e32 v87, 1, v87
	buffer_store_short v105, v87, s[8:11], 0 offen
	v_or_b32_e32 v87, 2, v79
	v_mul_lo_u32 v87, s12, v87
	v_add_u32_e32 v89, v87, v112
	v_cndmask_b32_e64 v89, v63, v89, s[2:3]
	v_lshlrev_b32_e32 v89, 1, v89
	buffer_store_short v101, v89, s[8:11], 0 offen
	v_or_b32_e32 v89, 3, v79
	v_mul_lo_u32 v89, s12, v89
	v_add_u32_e32 v90, v89, v112
	v_cmp_gt_u32_e64 s[4:5], s4, v67
	s_cmpk_lt_u32 s15, 0x1870
	s_nop 0
	v_cndmask_b32_e64 v90, v63, v90, s[4:5]
	v_lshlrev_b32_e32 v90, 1, v90
	buffer_store_short v104, v90, s[8:11], 0 offen
	v_add_u32_e32 v90, v83, v113
	v_cndmask_b32_e32 v90, v63, v90, vcc
	v_lshlrev_b32_e32 v90, 1, v90
	buffer_store_short v109, v90, s[8:11], 0 offen
	v_add_u32_e32 v90, v86, v113
	v_cndmask_b32_e64 v90, v63, v90, s[0:1]
	v_lshlrev_b32_e32 v90, 1, v90
	buffer_store_short v106, v90, s[8:11], 0 offen
	v_add_u32_e32 v90, v87, v113
	v_cndmask_b32_e64 v90, v63, v90, s[2:3]
	v_lshlrev_b32_e32 v90, 1, v90
	buffer_store_short v102, v90, s[8:11], 0 offen
	v_add_u32_e32 v90, v89, v113
	v_cndmask_b32_e64 v90, v63, v90, s[4:5]
	v_lshlrev_b32_e32 v90, 1, v90
	buffer_store_short v100, v90, s[8:11], 0 offen
	v_add_u32_e32 v90, 32, v112
	v_add_u32_e32 v91, v83, v90
	v_cndmask_b32_e32 v91, v63, v91, vcc
	v_lshlrev_b32_e32 v91, 1, v91
	buffer_store_short v119, v91, s[8:11], 0 offen
	v_add_u32_e32 v91, v86, v90
	v_cndmask_b32_e64 v91, v63, v91, s[0:1]
	v_lshlrev_b32_e32 v91, 1, v91
	buffer_store_short v110, v91, s[8:11], 0 offen
	v_add_u32_e32 v91, v87, v90
	v_cndmask_b32_e64 v91, v63, v91, s[2:3]
	v_lshlrev_b32_e32 v91, 1, v91
	buffer_store_short v107, v91, s[8:11], 0 offen
	v_add_u32_e32 v91, v89, v90
	v_cndmask_b32_e64 v91, v63, v91, s[4:5]
	v_lshlrev_b32_e32 v91, 1, v91
	buffer_store_short v103, v91, s[8:11], 0 offen
	v_add_u32_e32 v91, v83, v114
	v_cndmask_b32_e32 v91, v63, v91, vcc
	v_lshlrev_b32_e32 v91, 1, v91
	buffer_store_short v123, v91, s[8:11], 0 offen
	v_add_u32_e32 v91, v86, v114
	v_cndmask_b32_e64 v91, v63, v91, s[0:1]
	v_lshlrev_b32_e32 v91, 1, v91
	buffer_store_short v120, v91, s[8:11], 0 offen
	v_add_u32_e32 v91, v87, v114
	v_cndmask_b32_e64 v91, v63, v91, s[2:3]
	v_lshlrev_b32_e32 v91, 1, v91
	buffer_store_short v117, v91, s[8:11], 0 offen
	v_add_u32_e32 v91, v89, v114
	v_cndmask_b32_e64 v91, v63, v91, s[4:5]
	v_lshlrev_b32_e32 v91, 1, v91
	buffer_store_short v108, v91, s[8:11], 0 offen
	v_add_u32_e32 v91, 64, v112
	v_add_u32_e32 v94, v83, v91
	v_cndmask_b32_e32 v94, v63, v94, vcc
	v_lshlrev_b32_e32 v94, 1, v94
	buffer_store_short v126, v94, s[8:11], 0 offen
	v_add_u32_e32 v94, v86, v91
	v_cndmask_b32_e64 v94, v63, v94, s[0:1]
	v_lshlrev_b32_e32 v94, 1, v94
	buffer_store_short v124, v94, s[8:11], 0 offen
	v_add_u32_e32 v94, v87, v91
	v_cndmask_b32_e64 v94, v63, v94, s[2:3]
	v_lshlrev_b32_e32 v94, 1, v94
	buffer_store_short v121, v94, s[8:11], 0 offen
	v_add_u32_e32 v94, v89, v91
	v_cndmask_b32_e64 v94, v63, v94, s[4:5]
	v_lshlrev_b32_e32 v94, 1, v94
	buffer_store_short v118, v94, s[8:11], 0 offen
	v_add_u32_e32 v94, v83, v115
	v_cndmask_b32_e32 v94, v63, v94, vcc
	v_lshlrev_b32_e32 v94, 1, v94
	buffer_store_short v129, v94, s[8:11], 0 offen
	v_add_u32_e32 v94, v86, v115
	v_cndmask_b32_e64 v94, v63, v94, s[0:1]
	v_lshlrev_b32_e32 v94, 1, v94
	buffer_store_short v128, v94, s[8:11], 0 offen
	v_add_u32_e32 v94, v87, v115
	v_cndmask_b32_e64 v94, v63, v94, s[2:3]
	v_lshlrev_b32_e32 v94, 1, v94
	buffer_store_short v125, v94, s[8:11], 0 offen
	v_add_u32_e32 v94, v89, v115
	v_cndmask_b32_e64 v94, v63, v94, s[4:5]
	v_lshlrev_b32_e32 v94, 1, v94
	buffer_store_short v122, v94, s[8:11], 0 offen
	v_add_u32_e32 v94, 0x60, v112
	v_add_u32_e32 v83, v83, v94
	v_cndmask_b32_e32 v83, v63, v83, vcc
	v_lshlrev_b32_e32 v83, 1, v83
	buffer_store_short v96, v83, s[8:11], 0 offen
	v_add_u32_e32 v83, v86, v94
	v_cndmask_b32_e64 v83, v63, v83, s[0:1]
	v_lshlrev_b32_e32 v83, 1, v83
	buffer_store_short v97, v83, s[8:11], 0 offen
	v_add_u32_e32 v83, v87, v94
	v_cndmask_b32_e64 v83, v63, v83, s[2:3]
	v_lshlrev_b32_e32 v83, 1, v83
	buffer_store_short v98, v83, s[8:11], 0 offen
	v_add_u32_e32 v83, v89, v94
	v_cndmask_b32_e64 v83, v63, v83, s[4:5]
	v_lshlrev_b32_e32 v83, 1, v83
	buffer_store_short v37, v83, s[8:11], 0 offen
	v_or_b32_e32 v37, 16, v79
	v_mul_lo_u32 v37, s12, v37
	v_add_u32_e32 v83, v37, v112
	s_cselect_b64 vcc, -1, 0
	v_cndmask_b32_e32 v83, v63, v83, vcc
	v_lshlrev_b32_e32 v83, 1, v83
	buffer_store_short v92, v83, s[8:11], 0 offen
	v_or_b32_e32 v83, 17, v79
	s_movk_i32 s0, 0x186f
	v_mul_lo_u32 v83, s12, v83
	v_add_u32_e32 v86, v83, v112
	v_cmp_gt_u32_e64 s[0:1], s0, v67
	s_movk_i32 s2, 0x186e
	v_cmp_gt_u32_e64 s[2:3], s2, v67
	v_cndmask_b32_e64 v86, v63, v86, s[0:1]
	v_lshlrev_b32_e32 v86, 1, v86
	buffer_store_short v93, v86, s[8:11], 0 offen
	v_or_b32_e32 v86, 18, v79
	v_mul_lo_u32 v86, s12, v86
	v_add_u32_e32 v87, v86, v112
	v_cndmask_b32_e64 v87, v63, v87, s[2:3]
	v_lshlrev_b32_e32 v87, 1, v87
	buffer_store_short v33, v87, s[8:11], 0 offen
	v_or_b32_e32 v33, 19, v79
	s_movk_i32 s4, 0x186d
	v_mul_lo_u32 v33, s12, v33
	v_add_u32_e32 v87, v33, v112
	v_cmp_gt_u32_e64 s[4:5], s4, v67
	s_cmpk_lt_u32 s15, 0x1860
	s_nop 0
	v_cndmask_b32_e64 v87, v63, v87, s[4:5]
	v_lshlrev_b32_e32 v87, 1, v87
	buffer_store_short v24, v87, s[8:11], 0 offen
	v_add_u32_e32 v24, v37, v113
	v_cndmask_b32_e32 v24, v63, v24, vcc
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v88, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v83, v113
	v_cndmask_b32_e64 v24, v63, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v38, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v86, v113
	v_cndmask_b32_e64 v24, v63, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v32, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v33, v113
	v_cndmask_b32_e64 v24, v63, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v25, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v37, v90
	v_cndmask_b32_e32 v24, v63, v24, vcc
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v84, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v83, v90
	v_cndmask_b32_e64 v24, v63, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v85, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v86, v90
	v_cndmask_b32_e64 v24, v63, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v36, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v33, v90
	v_cndmask_b32_e64 v24, v63, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v26, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v37, v114
	v_cndmask_b32_e32 v24, v63, v24, vcc
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v80, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v83, v114
	v_cndmask_b32_e64 v24, v63, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v81, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v86, v114
	v_cndmask_b32_e64 v24, v63, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v82, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v33, v114
	v_cndmask_b32_e64 v24, v63, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v34, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v37, v91
	v_cndmask_b32_e32 v24, v63, v24, vcc
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v76, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v83, v91
	v_cndmask_b32_e64 v24, v63, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v77, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v86, v91
	v_cndmask_b32_e64 v24, v63, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v78, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v33, v91
	v_cndmask_b32_e64 v24, v63, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v39, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v37, v115
	v_cndmask_b32_e32 v24, v63, v24, vcc
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v72, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v83, v115
	v_cndmask_b32_e64 v24, v63, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v73, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v86, v115
	v_cndmask_b32_e64 v24, v63, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v74, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v33, v115
	v_cndmask_b32_e64 v24, v63, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v75, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v37, v94
	v_cndmask_b32_e32 v24, v63, v24, vcc
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v68, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v83, v94
	v_cndmask_b32_e64 v24, v63, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v69, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v86, v94
	v_cndmask_b32_e64 v24, v63, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v70, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v33, v94
	v_cndmask_b32_e64 v24, v63, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v71, v24, s[8:11], 0 offen
	v_or_b32_e32 v24, 32, v79
	v_mul_lo_u32 v24, s12, v24
	v_add_u32_e32 v25, v24, v112
	s_cselect_b64 vcc, -1, 0
	v_cndmask_b32_e32 v25, v63, v25, vcc
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v64, v25, s[8:11], 0 offen
	v_or_b32_e32 v25, 33, v79
	s_movk_i32 s0, 0x185f
	v_mul_lo_u32 v25, s12, v25
	v_add_u32_e32 v26, v25, v112
	v_cmp_gt_u32_e64 s[0:1], s0, v67
	s_movk_i32 s2, 0x185e
	v_cmp_gt_u32_e64 s[2:3], s2, v67
	v_cndmask_b32_e64 v26, v63, v26, s[0:1]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v65, v26, s[8:11], 0 offen
	v_or_b32_e32 v26, 34, v79
	v_mul_lo_u32 v26, s12, v26
	v_add_u32_e32 v32, v26, v112
	v_cndmask_b32_e64 v32, v63, v32, s[2:3]
	v_lshlrev_b32_e32 v32, 1, v32
	buffer_store_short v66, v32, s[8:11], 0 offen
	v_or_b32_e32 v32, 35, v79
	s_movk_i32 s4, 0x185d
	v_mul_lo_u32 v32, s12, v32
	v_add_u32_e32 v33, v32, v112
	v_cmp_gt_u32_e64 s[4:5], s4, v67
	s_cmpk_lt_u32 s15, 0x1850
	s_nop 0
	v_cndmask_b32_e64 v33, v63, v33, s[4:5]
	v_lshlrev_b32_e32 v33, 1, v33
	buffer_store_short v27, v33, s[8:11], 0 offen
	v_add_u32_e32 v27, v24, v113
	v_cndmask_b32_e32 v27, v63, v27, vcc
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v60, v27, s[8:11], 0 offen
	v_add_u32_e32 v27, v25, v113
	v_cndmask_b32_e64 v27, v63, v27, s[0:1]
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v61, v27, s[8:11], 0 offen
	v_add_u32_e32 v27, v26, v113
	v_cndmask_b32_e64 v27, v63, v27, s[2:3]
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v62, v27, s[8:11], 0 offen
	v_add_u32_e32 v27, v32, v113
	v_cndmask_b32_e64 v27, v63, v27, s[4:5]
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v35, v27, s[8:11], 0 offen
	v_add_u32_e32 v27, v24, v90
	v_cndmask_b32_e32 v27, v63, v27, vcc
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v56, v27, s[8:11], 0 offen
	v_add_u32_e32 v27, v25, v90
	v_cndmask_b32_e64 v27, v63, v27, s[0:1]
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v57, v27, s[8:11], 0 offen
	v_add_u32_e32 v27, v26, v90
	v_cndmask_b32_e64 v27, v63, v27, s[2:3]
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v58, v27, s[8:11], 0 offen
	v_add_u32_e32 v27, v32, v90
	v_cndmask_b32_e64 v27, v63, v27, s[4:5]
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v59, v27, s[8:11], 0 offen
	v_add_u32_e32 v27, v24, v114
	v_cndmask_b32_e32 v27, v63, v27, vcc
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v52, v27, s[8:11], 0 offen
	v_add_u32_e32 v27, v25, v114
	v_cndmask_b32_e64 v27, v63, v27, s[0:1]
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v53, v27, s[8:11], 0 offen
	v_add_u32_e32 v27, v26, v114
	v_cndmask_b32_e64 v27, v63, v27, s[2:3]
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v54, v27, s[8:11], 0 offen
	v_add_u32_e32 v27, v32, v114
	v_cndmask_b32_e64 v27, v63, v27, s[4:5]
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v55, v27, s[8:11], 0 offen
	v_add_u32_e32 v27, v24, v91
	v_cndmask_b32_e32 v27, v63, v27, vcc
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v48, v27, s[8:11], 0 offen
	v_add_u32_e32 v27, v25, v91
	v_cndmask_b32_e64 v27, v63, v27, s[0:1]
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v49, v27, s[8:11], 0 offen
	v_add_u32_e32 v27, v26, v91
	v_cndmask_b32_e64 v27, v63, v27, s[2:3]
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v50, v27, s[8:11], 0 offen
	v_add_u32_e32 v27, v32, v91
	v_cndmask_b32_e64 v27, v63, v27, s[4:5]
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v51, v27, s[8:11], 0 offen
	v_add_u32_e32 v27, v24, v115
	v_cndmask_b32_e32 v27, v63, v27, vcc
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v44, v27, s[8:11], 0 offen
	v_add_u32_e32 v27, v25, v115
	v_cndmask_b32_e64 v27, v63, v27, s[0:1]
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v45, v27, s[8:11], 0 offen
	v_add_u32_e32 v27, v26, v115
	v_cndmask_b32_e64 v27, v63, v27, s[2:3]
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v46, v27, s[8:11], 0 offen
	v_add_u32_e32 v27, v32, v115
	v_add_u32_e32 v24, v24, v94
	v_cndmask_b32_e64 v27, v63, v27, s[4:5]
	v_cndmask_b32_e32 v24, v63, v24, vcc
	v_lshlrev_b32_e32 v27, 1, v27
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v47, v27, s[8:11], 0 offen
	buffer_store_short v40, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v25, v94
	v_cndmask_b32_e64 v24, v63, v24, s[0:1]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v41, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v26, v94
	v_cndmask_b32_e64 v24, v63, v24, s[2:3]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v42, v24, s[8:11], 0 offen
	v_add_u32_e32 v24, v32, v94
	v_cndmask_b32_e64 v24, v63, v24, s[4:5]
	v_lshlrev_b32_e32 v24, 1, v24
	buffer_store_short v43, v24, s[8:11], 0 offen
	v_or_b32_e32 v24, 48, v79
	v_mul_lo_u32 v24, s12, v24
	v_add_u32_e32 v25, v24, v112
	s_cselect_b64 vcc, -1, 0
	v_cndmask_b32_e32 v25, v63, v25, vcc
	v_lshlrev_b32_e32 v25, 1, v25
	buffer_store_short v28, v25, s[8:11], 0 offen
	v_or_b32_e32 v25, 49, v79
	s_movk_i32 s0, 0x184f
	v_mul_lo_u32 v25, s12, v25
	v_add_u32_e32 v26, v25, v112
	v_cmp_gt_u32_e64 s[0:1], s0, v67
	s_movk_i32 s2, 0x184e
	v_cmp_gt_u32_e64 s[2:3], s2, v67
	v_cndmask_b32_e64 v26, v63, v26, s[0:1]
	v_lshlrev_b32_e32 v26, 1, v26
	buffer_store_short v29, v26, s[8:11], 0 offen
	v_or_b32_e32 v26, 50, v79
	v_mul_lo_u32 v26, s12, v26
	v_add_u32_e32 v27, v26, v112
	v_cndmask_b32_e64 v27, v63, v27, s[2:3]
	v_lshlrev_b32_e32 v27, 1, v27
	buffer_store_short v30, v27, s[8:11], 0 offen
	v_or_b32_e32 v27, 51, v79
	s_movk_i32 s4, 0x184d
	v_mul_lo_u32 v27, s12, v27
	v_add_u32_e32 v28, v27, v112
	v_cmp_gt_u32_e64 s[4:5], s4, v67
	s_nop 1
	v_cndmask_b32_e64 v28, v63, v28, s[4:5]
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v31, v28, s[8:11], 0 offen
	v_add_u32_e32 v28, v24, v113
	v_cndmask_b32_e32 v28, v63, v28, vcc
	v_lshlrev_b32_e32 v28, 1, v28
	buffer_store_short v20, v28, s[8:11], 0 offen
	v_add_u32_e32 v20, v25, v113
	v_cndmask_b32_e64 v20, v63, v20, s[0:1]
	v_lshlrev_b32_e32 v20, 1, v20
	buffer_store_short v21, v20, s[8:11], 0 offen
	v_add_u32_e32 v20, v26, v113
	v_cndmask_b32_e64 v20, v63, v20, s[2:3]
	v_lshlrev_b32_e32 v20, 1, v20
	buffer_store_short v22, v20, s[8:11], 0 offen
	v_add_u32_e32 v20, v27, v113
	v_cndmask_b32_e64 v20, v63, v20, s[4:5]
	v_lshlrev_b32_e32 v20, 1, v20
	buffer_store_short v23, v20, s[8:11], 0 offen
	v_add_u32_e32 v20, v24, v90
	v_cndmask_b32_e32 v20, v63, v20, vcc
	v_lshlrev_b32_e32 v20, 1, v20
	buffer_store_short v16, v20, s[8:11], 0 offen
	v_add_u32_e32 v16, v25, v90
	v_cndmask_b32_e64 v16, v63, v16, s[0:1]
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v17, v16, s[8:11], 0 offen
	v_add_u32_e32 v16, v26, v90
	v_cndmask_b32_e64 v16, v63, v16, s[2:3]
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v18, v16, s[8:11], 0 offen
	v_add_u32_e32 v16, v27, v90
	v_cndmask_b32_e64 v16, v63, v16, s[4:5]
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v19, v16, s[8:11], 0 offen
	v_add_u32_e32 v16, v24, v114
	v_cndmask_b32_e32 v16, v63, v16, vcc
	v_lshlrev_b32_e32 v16, 1, v16
	buffer_store_short v12, v16, s[8:11], 0 offen
	v_add_u32_e32 v12, v25, v114
	v_cndmask_b32_e64 v12, v63, v12, s[0:1]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v13, v12, s[8:11], 0 offen
	v_add_u32_e32 v12, v26, v114
	v_cndmask_b32_e64 v12, v63, v12, s[2:3]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v14, v12, s[8:11], 0 offen
	v_add_u32_e32 v12, v27, v114
	v_cndmask_b32_e64 v12, v63, v12, s[4:5]
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v15, v12, s[8:11], 0 offen
	v_add_u32_e32 v12, v24, v91
	v_cndmask_b32_e32 v12, v63, v12, vcc
	v_lshlrev_b32_e32 v12, 1, v12
	buffer_store_short v8, v12, s[8:11], 0 offen
	v_add_u32_e32 v8, v25, v91
	v_cndmask_b32_e64 v8, v63, v8, s[0:1]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v9, v8, s[8:11], 0 offen
	v_add_u32_e32 v8, v26, v91
	v_cndmask_b32_e64 v8, v63, v8, s[2:3]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v10, v8, s[8:11], 0 offen
	v_add_u32_e32 v8, v27, v91
	v_cndmask_b32_e64 v8, v63, v8, s[4:5]
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v11, v8, s[8:11], 0 offen
	v_add_u32_e32 v8, v24, v115
	v_cndmask_b32_e32 v8, v63, v8, vcc
	v_lshlrev_b32_e32 v8, 1, v8
	buffer_store_short v4, v8, s[8:11], 0 offen
	v_add_u32_e32 v4, v25, v115
	v_cndmask_b32_e64 v4, v63, v4, s[0:1]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v5, v4, s[8:11], 0 offen
	v_add_u32_e32 v4, v26, v115
	v_cndmask_b32_e64 v4, v63, v4, s[2:3]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v6, v4, s[8:11], 0 offen
	v_add_u32_e32 v4, v27, v115
	v_cndmask_b32_e64 v4, v63, v4, s[4:5]
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v7, v4, s[8:11], 0 offen
	v_add_u32_e32 v4, v24, v94
	v_cndmask_b32_e32 v4, v63, v4, vcc
	v_lshlrev_b32_e32 v4, 1, v4
	buffer_store_short v0, v4, s[8:11], 0 offen
	v_add_u32_e32 v0, v25, v94
	v_cndmask_b32_e64 v0, v63, v0, s[0:1]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v1, v0, s[8:11], 0 offen
	v_add_u32_e32 v0, v26, v94
	v_cndmask_b32_e64 v0, v63, v0, s[2:3]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v2, v0, s[8:11], 0 offen
	v_add_u32_e32 v0, v27, v94
	v_cndmask_b32_e64 v0, v63, v0, s[4:5]
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v3, v0, s[8:11], 0 offen
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_256x224x256_6272x7168x5888
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
		.amdhsa_next_free_vgpr 244
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 244
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
	.size	wave_mxfp4_static_gemm_256x224x256_6272x7168x5888, .Lfunc_end0-wave_mxfp4_static_gemm_256x224x256_6272x7168x5888

	.set wave_mxfp4_static_gemm_256x224x256_6272x7168x5888.num_vgpr, 244
	.set wave_mxfp4_static_gemm_256x224x256_6272x7168x5888.num_agpr, 0
	.set wave_mxfp4_static_gemm_256x224x256_6272x7168x5888.numbered_sgpr, 71
	.set wave_mxfp4_static_gemm_256x224x256_6272x7168x5888.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_256x224x256_6272x7168x5888.private_seg_size, 0
	.set wave_mxfp4_static_gemm_256x224x256_6272x7168x5888.uses_vcc, 1
	.set wave_mxfp4_static_gemm_256x224x256_6272x7168x5888.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_256x224x256_6272x7168x5888.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_256x224x256_6272x7168x5888.has_recursion, 0
	.set wave_mxfp4_static_gemm_256x224x256_6272x7168x5888.has_indirect_call, 0
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
    .name:           wave_mxfp4_static_gemm_256x224x256_6272x7168x5888
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     77
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_256x224x256_6272x7168x5888.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     244
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
