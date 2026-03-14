; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_64x64x256_1792x512x7424
	.p2align	8
	.type	wave_mxfp4_static_gemm_64x64x256_1792x512x7424,@function
wave_mxfp4_static_gemm_64x64x256_1792x512x7424:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v6, 0x3ff, v0
	v_bfe_u32 v3, v0, 10, 10
	v_lshrrev_b32_e32 v4, 6, v6
	v_lshlrev_b32_e32 v8, 5, v3
	v_lshrrev_b32_e32 v2, 3, v6
	v_lshl_or_b32 v0, v4, 3, v8
	v_or_b32_e32 v5, v2, v8
	s_lshl_b32 s16, s16, 6
	s_mov_b64 s[24:25], s[2:3]
	v_readfirstlane_b32 s2, v0
	v_or_b32_e32 v1, s16, v5
	v_bitop3_b32 v9, v2, 7, v6 bitop3:0x48
	s_mov_b64 s[20:21], s[6:7]
	v_lshlrev_b32_e32 v0, 4, v9
	v_mul_u32_u24_e32 v1, 0xe80, v1
	s_and_b32 s6, s25, 0xffff
	s_lshl_b32 s19, s2, 7
	s_or_b32 s25, s6, 0x4e800000
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_or_b32_e32 v7, v1, v0
	s_mov_b32 m0, s19
	s_lshl_b32 s18, s17, 6
	buffer_load_dwordx4 v7, s[24:27], 0 offen lds
	v_lshlrev_b32_e32 v7, 1, v6
	v_and_or_b32 v10, v5, 48, s18
	v_lshlrev_b32_e32 v5, 4, v2
	v_and_b32_e32 v7, 0x100, v7
	v_sub_u32_e32 v11, v5, v7
	v_lshlrev_b32_e32 v29, 8, v9
	s_movk_i32 s3, 0xe80
	v_add_u32_e32 v9, v29, v11
	s_and_b32 s2, s21, 0xffff
	s_or_b32 s21, s2, 0x4e800000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_mad_u32_u24 v9, v10, s3, v9
	s_add_i32 m0, s19, 0x4000
	v_lshrrev_b32_e32 v13, 4, v6
	buffer_load_dwordx4 v9, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v11, 4, v4
	v_mad_i32_i24 v9, v13, -16, v6
	v_add_u32_e32 v12, v9, v11
	v_ashrrev_i16_e32 v14, 15, v12
	v_lshrrev_b16_e32 v14, 11, v14
	v_add_u16_e32 v14, v12, v14
	v_and_b32_e32 v14, 0xffffffe0, v14
	v_sub_u16_e32 v14, v12, v14
	v_bfe_i32 v15, v14, 0, 16
	v_ashrrev_i32_e32 v17, 31, v15
	v_add_u16_e32 v18, 32, v14
	v_cmp_gt_i16_e32 vcc, 0, v14
	v_lshlrev_b32_e32 v24, 2, v6
	s_movk_i32 s33, 0xffc0
	v_cndmask_b32_e32 v14, v15, v18, vcc
	v_cndmask_b32_e64 v15, v17, 0, vcc
	v_xor_b32_e32 v14, v15, v14
	v_lshrrev_b32_e32 v17, 28, v14
	v_add_u32_e32 v14, v14, v17
	v_bfe_u32 v10, v6, 4, 2
	v_ashrrev_i32_e32 v14, 4, v14
	v_mad_i32_i24 v16, v13, s33, v24
	v_lshlrev_b32_e32 v22, 6, v10
	v_xor_b32_e32 v25, v14, v15
	v_add3_u32 v17, v16, v22, v25
	v_ashrrev_i32_e32 v14, 31, v17
	v_xor_b32_e32 v15, v14, v17
	s_mov_b32 s34, 0x8d3dcb09
	v_mul_hi_i32 v18, v15, s34
	v_add_u32_e32 v15, v18, v15
	v_lshrrev_b32_e32 v18, 31, v15
	v_ashrrev_i32_e32 v15, 7, v15
	v_add_u32_e32 v15, v15, v18
	v_ashrrev_i32_e32 v18, 31, v12
	v_xor_b32_e32 v12, v18, v12
	v_ashrrev_i32_e32 v19, 31, v12
	v_lshrrev_b32_e32 v19, 27, v19
	v_add_u32_e32 v12, v12, v19
	v_lshrrev_b32_e32 v12, 5, v12
	v_xor_b32_e32 v12, v12, v18
	v_lshlrev_b32_e32 v12, 5, v12
	v_bfe_u32 v27, v6, 6, 1
	v_xad_u32 v15, v15, v14, v12
	v_or_b32_e32 v14, v16, v27
	v_add_u32_e32 v18, v14, v22
	v_mul_hi_i32 v14, v18, s34
	v_add_u32_e32 v14, v14, v18
	v_lshrrev_b32_e32 v19, 31, v14
	v_ashrrev_i32_e32 v14, 7, v14
	s_mul_i32 s15, s15, s16
	s_mul_hi_u32 s2, s14, s16
	v_add_u32_e32 v14, v14, v19
	s_movk_i32 s35, 0xe8
	s_add_i32 s2, s2, s15
	s_mul_i32 s3, s14, s16
	v_mul_lo_u32 v14, v14, s35
	s_add_u32 s4, s4, s3
	v_sub_u32_e32 v14, v18, v14
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s14, 0x3fff
	v_add_u32_e32 v19, 0xe8, v14
	v_cmp_gt_i32_e32 vcc, 0, v14
	s_bitset1_b32 s3, 14
	s_and_b32 s2, s2, 0xffff
	v_cndmask_b32_e32 v14, v14, v19, vcc
	s_lshl_b32 s3, s3, 16
	s_or_b32 s5, s2, s3
	v_mad_u64_u32 v[14:15], s[2:3], v15, s14, v[14:15]
	v_add_u32_e32 v15, 2, v17
	v_sub_u32_e32 v19, -3, v17
	v_cmp_gt_i32_e32 vcc, -2, v17
	s_movk_i32 s2, 0xff18
	s_load_dwordx2 s[12:13], s[0:1], 0x40
	v_cndmask_b32_e32 v15, v15, v19, vcc
	v_mul_hi_i32 v17, v15, s34
	v_add_u32_e32 v15, v17, v15
	v_lshrrev_b32_e32 v17, 31, v15
	v_ashrrev_i32_e32 v15, 7, v15
	v_add_u32_e32 v15, v15, v17
	v_cndmask_b32_e64 v17, 0, -1, vcc
	v_xad_u32 v15, v15, v17, v12
	v_or_b32_e32 v17, 2, v18
	v_xor_b32_e32 v19, -3, v18
	v_cmp_gt_i32_e32 vcc, 0, v18
	v_ashrrev_i32_e32 v18, 31, v18
	v_mul_lo_u32 v15, v15, s14
	v_cndmask_b32_e32 v17, v17, v19, vcc
	v_mul_hi_i32 v19, v17, s34
	v_add_u32_e32 v17, v19, v17
	v_lshrrev_b32_e32 v19, 31, v17
	v_ashrrev_i32_e32 v17, 7, v17
	v_add_u32_e32 v17, v17, v19
	v_xor_b32_e32 v17, v17, v18
	v_mul_lo_u32 v17, v17, s2
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	v_add3_u32 v15, v15, v17, v22
	v_lshl_add_u32 v26, v10, 4, v9
	v_add3_u32 v15, v15, v27, v16
	buffer_load_ubyte v18, v14, s[4:7], 0 offen
	buffer_load_ubyte v17, v15, s[4:7], 0 offen offset:2
	v_ashrrev_i32_e32 v14, 31, v26
	v_xor_b32_e32 v15, v14, v26
	v_mul_hi_i32 v19, v15, s34
	v_add_u32_e32 v15, v19, v15
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s13, s18
	s_mul_hi_u32 s3, s12, s18
	v_lshrrev_b32_e32 v19, 31, v15
	v_ashrrev_i32_e32 v15, 5, v15
	s_add_i32 s3, s3, s2
	s_mul_i32 s2, s12, s18
	v_add_u32_e32 v15, v15, v19
	s_add_u32 s28, s8, s2
	v_xad_u32 v15, v15, v14, v8
	v_and_b32_e32 v14, 62, v6
	s_addc_u32 s2, s9, s3
	s_and_b32 s3, s12, 0x3fff
	v_mov_b32_e32 v19, 0xffffff18
	v_cmp_lt_u32_e32 vcc, 57, v14
	s_bitset1_b32 s3, 14
	s_and_b32 s2, s2, 0xffff
	v_cndmask_b32_e32 v14, 0, v19, vcc
	s_lshl_b32 s3, s3, 16
	s_or_b32 s29, s2, s3
	v_mad_u64_u32 v[14:15], s[2:3], s12, v15, v[14:15]
	s_mov_b32 s30, s26
	s_mov_b32 s31, s27
	v_add3_u32 v14, v14, v22, v16
	buffer_load_dword v23, v14, s[28:31], 0 offen
	v_cmp_eq_u32_e64 s[2:3], 0, v3
	s_mov_b32 s9, 0
	s_movk_i32 s8, 0x4000
	v_mul_i32_i24_e32 v28, 0xffffffc0, v13
	s_and_b64 vcc, exec, s[2:3]
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	v_and_b32_e32 v15, 7, v6
	v_bitop3_b32 v14, v10, v6, 7 bitop3:0x78
	v_lshlrev_b32_e32 v6, 7, v6
	v_lshlrev_b32_e32 v13, 11, v13
	v_sub_u32_e32 v6, v6, v13
	v_lshl_add_u32 v4, v4, 11, v6
	v_lshl_add_u32 v3, v3, 12, v6
	v_bitop3_b32 v6, v10, v15, 4 bitop3:0x36
	v_lshlrev_b32_e32 v14, 4, v14
	v_lshlrev_b32_e32 v6, 4, v6
	v_or_b32_e32 v13, v4, v14
	v_or_b32_e32 v14, v3, v14
	v_or_b32_e32 v15, v6, v3
	v_add_u32_e32 v3, v22, v27
	v_add3_u32 v19, v3, v28, v24
	v_add_u32_e32 v3, v28, v22
	v_or_b32_e32 v16, v6, v4
	v_add_u32_e32 v4, v3, v25
	v_sub_u32_e32 v22, 0xfffffefd, v4
	v_add_u32_e32 v4, v3, v24
	v_add_u32_e32 v3, v3, v27
	s_mul_i32 s6, s17, 0x3a000
	v_add_u32_e32 v2, v2, v8
	v_sub_u32_e32 v21, 0, v24
	v_add_u32_e32 v24, v4, v25
	v_add_u32_e32 v25, v4, v27
	v_sub_u32_e32 v27, 0, v3
	v_lshrrev_b32_e32 v2, 4, v2
	s_mov_b32 s7, 0xe800
	v_mov_b32_e32 v3, s6
	s_load_dwordx2 s[0:1], s[0:1], 0x48
	v_mad_u32_u24 v2, v2, s7, v3
	v_or_b32_e32 v2, v2, v29
	v_add_u32_e32 v2, v2, v5
	s_movk_i32 s6, 0x80
	v_sub_u32_e32 v2, v2, v7
	v_add3_u32 v30, v1, v0, s6
	v_mov_b32_e32 v0, 0
	v_sub_u32_e32 v20, -7, v26
	v_add_u32_e32 v28, 24, v4
	v_add_u32_e32 v29, 0x800, v2
	s_movk_i32 s38, 0x2000
	s_movk_i32 s37, 0x6000
	s_mov_b32 s13, 0
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	s_movk_i32 s15, 0xff00
	s_movk_i32 s17, 0xffe8
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	s_movk_i32 s36, 0xfefe
	s_mov_b32 s30, s26
	s_mov_b32 s31, s27
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v7, v0
.LBB0_3:
	s_mov_b32 s39, s9
	s_mov_b32 s9, s38
	s_mov_b32 s38, s8
	s_mov_b32 s8, s37
	s_add_i32 m0, s9, s19
	s_barrier
	buffer_load_dwordx4 v30, s[24:27], 0 offen lds
	s_add_i32 m0, s37, s19
	s_nop 0
	buffer_load_dwordx4 v29, s[20:23], 0 offen lds
	v_add_u32_e32 v31, s13, v24
	v_add_u32_e32 v33, v21, v22
	v_add_u32_e32 v32, 0x100, v31
	v_add_u32_e32 v34, 2, v33
	v_cmp_gt_i32_e32 vcc, s15, v31
	v_add_u32_e32 v37, v21, v27
	v_subrev_u32_e32 v38, 25, v37
	v_cndmask_b32_e32 v34, v32, v34, vcc
	v_mul_hi_i32 v35, v34, s34
	v_add_u32_e32 v34, v35, v34
	v_lshrrev_b32_e32 v35, 31, v34
	v_ashrrev_i32_e32 v34, 7, v34
	v_add_u32_e32 v34, v34, v35
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_xad_u32 v34, v34, v35, v12
	v_add_u32_e32 v35, s13, v25
	v_add_u32_e32 v36, 24, v35
	v_cmp_gt_i32_e32 vcc, s17, v35
	v_add_u32_e32 v31, 0x102, v31
	v_mul_lo_u32 v34, v34, s14
	v_cndmask_b32_e32 v36, v36, v38, vcc
	v_mul_hi_i32 v38, v36, s34
	v_add_u32_e32 v36, v38, v36
	v_lshrrev_b32_e32 v38, 31, v36
	v_ashrrev_i32_e32 v36, 7, v36
	v_add_u32_e32 v36, v36, v38
	v_cndmask_b32_e64 v38, 0, -1, vcc
	v_cmp_gt_i32_e32 vcc, -2, v32
	v_xor_b32_e32 v36, v36, v38
	v_mul_lo_u32 v36, v36, s35
	v_cndmask_b32_e32 v32, v31, v33, vcc
	v_mul_hi_i32 v33, v32, s34
	v_add_u32_e32 v32, v33, v32
	v_lshrrev_b32_e32 v33, 31, v32
	v_ashrrev_i32_e32 v32, 7, v32
	v_add_u32_e32 v32, v32, v33
	v_ashrrev_i32_e32 v31, 31, v31
	v_xad_u32 v31, v32, v31, v12
	v_add_u32_e32 v32, 0x102, v35
	v_add_u32_e32 v33, 0xfffffefd, v37
	v_cmp_gt_i32_e32 vcc, s36, v35
	v_mul_lo_u32 v31, v31, s14
	v_sub_u32_e32 v34, v34, v36
	v_cndmask_b32_e32 v32, v32, v33, vcc
	v_mul_hi_i32 v33, v32, s34
	v_add_u32_e32 v32, v33, v32
	v_lshrrev_b32_e32 v33, 31, v32
	v_ashrrev_i32_e32 v32, 7, v32
	v_add_u32_e32 v32, v32, v33
	v_cndmask_b32_e64 v33, 0, -1, vcc
	v_xor_b32_e32 v32, v32, v33
	v_mul_lo_u32 v32, v32, s35
	v_sub_u32_e32 v31, v31, v32
	v_add3_u32 v34, s13, v34, v19
	v_add3_u32 v33, v19, v31, s13
	buffer_load_ubyte v32, v34, s[4:7], 0 offen offset:24
	buffer_load_ubyte v31, v33, s[4:7], 0 offen offset:258
	v_add_u32_e32 v34, 64, v26
	v_subrev_u32_e32 v33, 58, v20
	v_cmp_gt_i32_e32 vcc, s33, v26
	s_nop 1
	v_cndmask_b32_e32 v33, v34, v33, vcc
	v_mul_hi_i32 v35, v33, s34
	v_add_u32_e32 v33, v35, v33
	v_lshrrev_b32_e32 v35, 31, v33
	v_ashrrev_i32_e32 v33, 5, v33
	v_add_u32_e32 v33, v33, v35
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_xad_u32 v33, v33, v35, v8
	v_add_u32_e32 v35, 6, v26
	v_cmp_gt_i32_e32 vcc, -6, v26
	v_mul_lo_u32 v33, v33, s12
	s_nop 0
	v_cndmask_b32_e32 v26, v35, v20, vcc
	v_mul_hi_i32 v35, v26, s34
	v_add_u32_e32 v26, v35, v26
	v_lshrrev_b32_e32 v35, 31, v26
	v_ashrrev_i32_e32 v26, 5, v26
	v_add_u32_e32 v26, v26, v35
	v_cndmask_b32_e64 v35, 0, -1, vcc
	v_xor_b32_e32 v26, v26, v35
	v_mul_lo_u32 v26, v26, s35
	v_sub_u32_e32 v26, v33, v26
	v_add3_u32 v26, v28, v26, s13
	buffer_load_dword v33, v26, s[28:31], 0 offen
	v_add_u32_e32 v26, s39, v13
	ds_read_b128 v[36:39], v26
	v_add_u32_e32 v26, s38, v14
	ds_read_b128 v[40:43], v26
	ds_read_b128 v[44:47], v26 offset:2048
	s_barrier
	s_setprio 1
	v_and_b32_e32 v18, 0xff, v18
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[36:39], v[40:43], v[0:3], v18, v23 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[36:39], v[44:47], v[4:7], v18, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v18, s39, v16
	ds_read_b128 v[36:39], v18
	v_add_u32_e32 v18, s38, v15
	ds_read_b128 v[40:43], v18
	ds_read_b128 v[44:47], v18 offset:2048
	s_waitcnt vmcnt(3)
	s_barrier
	s_setprio 1
	v_and_b32_e32 v17, 0xff, v17
	s_waitcnt lgkmcnt(1)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[36:39], v[40:43], v[0:3], v17, v23 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[36:39], v[44:47], v[4:7], v17, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_addk_i32 s13, 0x100
	v_subrev_u32_e32 v20, 64, v20
	v_add_u32_e32 v22, 0xffffff00, v22
	v_add_u32_e32 v27, 0xffffff00, v27
	v_add_u32_e32 v29, 0x800, v29
	v_add_u32_e32 v30, 0x80, v30
	s_cmpk_lg_i32 s13, 0x1c00
	v_mov_b32_e32 v26, v34
	s_mov_b32 s37, s38
	s_mov_b32 s38, s39
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v23, v33
	v_mov_b32_e32 v17, v31
	v_mov_b32_e32 v18, v32
	s_cbranch_scc1 .LBB0_3
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	v_add_u32_e32 v12, s9, v13
	s_barrier
	ds_read_b128 v[18:21], v12
	v_add_u32_e32 v13, s9, v16
	ds_read_b128 v[26:29], v13
	v_add_u32_e32 v12, s8, v14
	ds_read_b128 v[22:25], v12
	v_add_u32_e32 v13, s8, v15
	ds_read_b128 v[14:17], v12 offset:2048
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[18:21], v[22:25], v[0:3], v32, v33 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[22:25], v13
	ds_read_b128 v[34:37], v13 offset:2048
	s_movk_i32 s2, 0x7fff
	s_mul_i32 s1, s1, s16
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[26:29], v[22:25], v[0:3], v31, v33 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_lshl_or_b32 v10, v10, 2, v11
	v_lshlrev_b32_e32 v8, 1, v8
	v_mul_lo_u32 v10, s0, v10
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[18:21], v[14:17], v[4:7], v32, v33 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 3
	v_bfe_u32 v12, v3, 16, 1
	v_bfe_u32 v13, v2, 16, 1
	v_add3_u32 v12, v3, v12, s2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[26:29], v[34:37], v[4:7], v31, v33 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_bfe_u32 v14, v1, 16, 1
	v_add3_u32 v13, v2, v13, s2
	v_lshrrev_b32_e32 v12, 16, v12
	v_mov_b32_e32 v16, 0x7fc0
	v_cmp_o_f32_e32 vcc, v3, v3
	v_bfe_u32 v15, v0, 16, 1
	v_add3_u32 v14, v1, v14, s2
	v_lshrrev_b32_e32 v13, 16, v13
	v_cndmask_b32_e32 v3, v16, v12, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	v_add3_u32 v15, v0, v15, s2
	v_lshrrev_b32_e32 v14, 16, v14
	v_cndmask_b32_e32 v2, v16, v13, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	v_lshrrev_b32_e32 v15, 16, v15
	v_bfe_u32 v12, v7, 16, 1
	v_cndmask_b32_e32 v1, v16, v14, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	v_bfe_u32 v13, v6, 16, 1
	v_bfe_u32 v14, v5, 16, 1
	v_cndmask_b32_e32 v0, v16, v15, vcc
	v_bfe_u32 v15, v4, 16, 1
	v_add3_u32 v15, v4, v15, s2
	v_add3_u32 v14, v5, v14, s2
	v_add3_u32 v13, v6, v13, s2
	v_add3_u32 v12, v7, v12, s2
	s_mul_hi_u32 s2, s0, s16
	s_add_i32 s3, s2, s1
	s_mul_i32 s2, s0, s16
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s1, s10, s2
	s_addc_u32 s2, s11, s3
	s_lshl_b32 s3, s18, 1
	s_add_u32 s4, s1, s3
	s_addc_u32 s1, s2, 0
	s_and_b32 s2, s0, 0x3fff
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	s_or_b32 s1, s2, s1
	v_lshl_add_u32 v8, v9, 1, v8
	v_lshrrev_b32_e32 v12, 16, v12
	v_cmp_o_f32_e32 vcc, v7, v7
	s_or_b32 s5, s1, 2.0
	s_mov_b32 s7, 0x27000
	s_mov_b32 s6, 0x7ffffffd
	v_lshl_add_u32 v8, v10, 1, v8
	s_lshl_b32 s0, s0, 1
	v_lshrrev_b32_e32 v13, 16, v13
	v_cndmask_b32_e32 v7, v16, v12, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	buffer_store_short v0, v8, s[4:7], 0 offen
	v_add_u32_e32 v0, s0, v8
	v_lshrrev_b32_e32 v14, 16, v14
	v_cndmask_b32_e32 v6, v16, v13, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	buffer_store_short v1, v0, s[4:7], 0 offen
	v_add_u32_e32 v1, s0, v0
	v_lshrrev_b32_e32 v15, 16, v15
	v_cndmask_b32_e32 v5, v16, v14, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	buffer_store_short v2, v1, s[4:7], 0 offen
	v_add_u32_e32 v2, s0, v1
	v_cndmask_b32_e32 v4, v16, v15, vcc
	buffer_store_short v3, v2, s[4:7], 0 offen
	buffer_store_short v4, v8, s[4:7], 0 offen offset:32
	buffer_store_short v5, v0, s[4:7], 0 offen offset:32
	buffer_store_short v6, v1, s[4:7], 0 offen offset:32
	buffer_store_short v7, v2, s[4:7], 0 offen offset:32
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_64x64x256_1792x512x7424
		.amdhsa_group_segment_fixed_size 32768
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
		.amdhsa_next_free_vgpr 48
		.amdhsa_next_free_sgpr 40
		.amdhsa_accum_offset 48
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
	.size	wave_mxfp4_static_gemm_64x64x256_1792x512x7424, .Lfunc_end0-wave_mxfp4_static_gemm_64x64x256_1792x512x7424

	.set wave_mxfp4_static_gemm_64x64x256_1792x512x7424.num_vgpr, 48
	.set wave_mxfp4_static_gemm_64x64x256_1792x512x7424.num_agpr, 0
	.set wave_mxfp4_static_gemm_64x64x256_1792x512x7424.numbered_sgpr, 40
	.set wave_mxfp4_static_gemm_64x64x256_1792x512x7424.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_64x64x256_1792x512x7424.private_seg_size, 0
	.set wave_mxfp4_static_gemm_64x64x256_1792x512x7424.uses_vcc, 1
	.set wave_mxfp4_static_gemm_64x64x256_1792x512x7424.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_64x64x256_1792x512x7424.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_64x64x256_1792x512x7424.has_recursion, 0
	.set wave_mxfp4_static_gemm_64x64x256_1792x512x7424.has_indirect_call, 0
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
    .group_segment_fixed_size: 32768
    .kernarg_segment_align: 8
    .kernarg_segment_size: 80
    .max_flat_workgroup_size: 512
    .name:           wave_mxfp4_static_gemm_64x64x256_1792x512x7424
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     46
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_64x64x256_1792x512x7424.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     48
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
