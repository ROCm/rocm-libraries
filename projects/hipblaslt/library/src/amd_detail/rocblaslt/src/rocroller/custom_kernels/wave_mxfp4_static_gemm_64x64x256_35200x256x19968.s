; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_64x64x256_35200x256x19968
	.p2align	8
	.type	wave_mxfp4_static_gemm_64x64x256_35200x256x19968,@function
wave_mxfp4_static_gemm_64x64x256_35200x256x19968:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v1, 0x3ff, v0
	v_bfe_u32 v0, v0, 10, 10
	v_lshrrev_b32_e32 v2, 6, v1
	v_lshlrev_b32_e32 v11, 5, v0
	v_lshl_or_b32 v3, v2, 3, v11
	s_mov_b64 s[20:21], s[2:3]
	v_readfirstlane_b32 s2, v3
	v_lshrrev_b32_e32 v3, 3, v1
	v_or_b32_e32 v16, v3, v11
	s_lshl_b32 s28, s16, 6
	v_or_b32_e32 v4, s28, v16
	v_bitop3_b32 v5, v3, 7, v1 bitop3:0x48
	v_lshlrev_b32_e32 v17, 4, v5
	v_mul_u32_u24_e32 v18, 0x2700, v4
	s_lshl_b32 s30, s2, 7
	s_and_b32 s21, s21, 0xffff
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	v_or_b32_e32 v4, v18, v17
	s_mov_b32 m0, s30
	v_lshlrev_b32_e32 v3, 4, v3
	buffer_load_dwordx4 v4, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v4, 1, v1
	v_and_b32_e32 v4, 0x100, v4
	s_lshl_b32 s29, s17, 6
	v_sub_u32_e32 v20, v3, v4
	v_lshlrev_b32_e32 v19, 8, v5
	s_mov_b64 s[24:25], s[6:7]
	s_movk_i32 s3, 0x2700
	v_and_or_b32 v23, v16, 48, s29
	v_add_u32_e32 v5, v19, v20
	s_add_i32 s31, s30, 0x4000
	s_and_b32 s25, s25, 0xffff
	s_mov_b32 s26, s22
	s_mov_b32 s27, s23
	v_mad_u32_u24 v5, v23, s3, v5
	s_mov_b32 m0, s31
	v_lshlrev_b32_e32 v14, 4, v2
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_lshrrev_b32_e32 v5, 4, v1
	v_mad_i32_i24 v12, v5, -16, v1
	v_add_u32_e32 v9, v12, v14
	v_ashrrev_i16_e32 v6, 15, v9
	v_lshrrev_b16_e32 v6, 11, v6
	v_add_u16_e32 v6, v9, v6
	v_and_b32_e32 v6, 0xffffffe0, v6
	v_sub_u16_e32 v6, v9, v6
	v_bfe_i32 v21, v6, 0, 16
	v_ashrrev_i32_e32 v22, 31, v21
	v_add_u16_e32 v24, 32, v6
	v_cmp_gt_i16_e32 vcc, 0, v6
	v_lshlrev_b32_e32 v15, 2, v1
	s_movk_i32 s33, 0xffc0
	v_cndmask_b32_e32 v6, v21, v24, vcc
	v_cndmask_b32_e64 v21, v22, 0, vcc
	v_xor_b32_e32 v6, v21, v6
	v_lshrrev_b32_e32 v22, 28, v6
	v_bfe_u32 v13, v1, 4, 2
	v_add_u32_e32 v6, v6, v22
	v_mad_i32_i24 v7, v5, s33, v15
	v_lshlrev_b32_e32 v8, 6, v13
	v_ashrrev_i32_e32 v6, 4, v6
	v_add_u32_e32 v10, v7, v8
	v_xor_b32_e32 v6, v6, v21
	v_add_u32_e32 v21, v6, v10
	v_ashrrev_i32_e32 v22, 31, v21
	v_xor_b32_e32 v24, v22, v21
	s_mov_b32 s34, 0xd20d20d3
	v_mul_hi_i32 v25, v24, s34
	v_add_u32_e32 v24, v25, v24
	v_lshrrev_b32_e32 v25, 31, v24
	v_ashrrev_i32_e32 v24, 9, v24
	v_add_u32_e32 v25, v24, v25
	v_ashrrev_i32_e32 v24, 31, v9
	v_xor_b32_e32 v9, v24, v9
	v_ashrrev_i32_e32 v26, 31, v9
	v_lshrrev_b32_e32 v26, 27, v26
	v_add_u32_e32 v9, v9, v26
	v_lshrrev_b32_e32 v9, 5, v9
	v_xor_b32_e32 v9, v9, v24
	v_lshlrev_b32_e32 v24, 5, v9
	v_xad_u32 v9, v25, v22, v24
	v_bfe_u32 v22, v1, 6, 1
	v_or_b32_e32 v25, v7, v22
	v_add_u32_e32 v25, v25, v8
	v_mul_hi_i32 v26, v25, s34
	v_add_u32_e32 v26, v26, v25
	v_lshrrev_b32_e32 v27, 31, v26
	v_ashrrev_i32_e32 v26, 9, v26
	s_mul_i32 s15, s15, s28
	s_mul_hi_u32 s2, s14, s28
	v_add_u32_e32 v26, v26, v27
	s_add_i32 s2, s2, s15
	s_mul_i32 s3, s14, s28
	v_mul_i32_i24_e32 v26, 0x270, v26
	s_add_u32 s4, s4, s3
	v_sub_u32_e32 v26, v25, v26
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s14, 0x3fff
	v_add_u32_e32 v27, 0x270, v26
	v_cmp_gt_i32_e32 vcc, 0, v26
	s_bitset1_b32 s3, 14
	s_and_b32 s2, s2, 0xffff
	v_cndmask_b32_e32 v26, v26, v27, vcc
	s_lshl_b32 s3, s3, 16
	s_or_b32 s5, s2, s3
	v_mad_u64_u32 v[26:27], s[2:3], v9, s14, v[26:27]
	v_add_u32_e32 v9, 2, v21
	v_sub_u32_e32 v27, -3, v21
	v_cmp_gt_i32_e32 vcc, -2, v21
	s_load_dwordx2 s[12:13], s[0:1], 0x40
	v_xor_b32_e32 v28, -3, v25
	v_cndmask_b32_e32 v9, v9, v27, vcc
	v_mul_hi_i32 v27, v9, s34
	v_add_u32_e32 v9, v27, v9
	v_lshrrev_b32_e32 v27, 31, v9
	v_ashrrev_i32_e32 v9, 9, v9
	v_add_u32_e32 v9, v9, v27
	v_cndmask_b32_e64 v27, 0, -1, vcc
	v_xad_u32 v9, v9, v27, v24
	v_or_b32_e32 v27, 2, v25
	v_cmp_gt_i32_e32 vcc, 0, v25
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s13, s29
	s_mul_hi_u32 s3, s12, s29
	v_cndmask_b32_e32 v27, v27, v28, vcc
	v_mul_hi_i32 v28, v27, s34
	v_add_u32_e32 v27, v28, v27
	v_lshrrev_b32_e32 v28, 31, v27
	v_ashrrev_i32_e32 v27, 9, v27
	s_add_i32 s3, s3, s2
	s_mul_i32 s2, s12, s29
	v_add_u32_e32 v27, v27, v28
	v_ashrrev_i32_e32 v28, 31, v25
	s_add_u32 s16, s8, s2
	v_xor_b32_e32 v27, v27, v28
	s_addc_u32 s2, s9, s3
	s_and_b32 s3, s12, 0x3fff
	v_mul_i32_i24_e32 v27, 0xfffffd90, v27
	v_mul_lo_u32 v9, v9, s14
	s_bitset1_b32 s3, 14
	s_mov_b32 s6, s22
	s_mov_b32 s7, s23
	v_add3_u32 v9, v9, v27, v8
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	v_add3_u32 v7, v9, v22, v7
	buffer_load_ubyte v33, v26, s[4:7], 0 offen
	buffer_load_ubyte v31, v7, s[4:7], 0 offen offset:2
	s_or_b32 s17, s2, s3
	s_mov_b32 s18, s22
	s_mov_b32 s19, s23
	v_mad_u64_u32 v[26:27], s[2:3], s12, v11, v[10:11]
	buffer_load_dword v34, v26, s[16:19], 0 offen
	v_cmp_eq_u32_e64 s[2:3], 0, v0
	s_mov_b32 s13, 0
	v_mul_i32_i24_e32 v30, 0xffffffc0, v5
	v_mul_i32_i24_e32 v7, -16, v5
	s_mov_b32 s46, -2
	s_movk_i32 s15, 0xfd90
	s_and_b64 vcc, exec, s[2:3]
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	v_lshlrev_b32_e32 v26, 7, v1
	v_lshlrev_b32_e32 v5, 11, v5
	v_and_b32_e32 v9, 7, v1
	v_sub_u32_e32 v5, v26, v5
	v_bitop3_b32 v10, v13, v1, 7 bitop3:0x78
	v_lshl_add_u32 v2, v2, 11, v5
	v_lshl_add_u32 v0, v0, 12, v5
	v_bitop3_b32 v5, v13, v9, 4 bitop3:0x36
	v_lshlrev_b32_e32 v10, 4, v10
	v_lshlrev_b32_e32 v5, 4, v5
	v_or_b32_e32 v29, v0, v10
	v_or_b32_e32 v26, v5, v0
	v_add_u32_e32 v0, v19, v3
	v_sub_u32_e32 v0, v0, v4
	s_load_dwordx2 s[8:9], s[0:1], 0x48
	v_add_u32_e32 v37, 0x1000, v0
	v_add3_u32 v0, v30, v6, v8
	v_lshlrev_b32_e32 v32, 4, v13
	v_sub_u32_e32 v39, 0xfffffdfd, v0
	v_sub_u32_e32 v0, 0, v30
	v_or_b32_e32 v28, v2, v10
	v_add3_u32 v10, v32, v7, v1
	v_add_u32_e32 v35, v30, v8
	v_sub_u32_e32 v1, v0, v6
	v_sub_u32_e32 v0, v0, v22
	v_add_u32_e32 v36, v35, v22
	v_sub_u32_e32 v42, v0, v8
	v_mov_b32_e32 v0, 0
	v_or_b32_e32 v27, v5, v2
	v_mov_b32_e32 v9, v10
	s_mov_b32 s35, s12
	v_sub_u32_e32 v38, 0, v15
	v_add_u32_e32 v40, v35, v6
	v_sub_u32_e32 v41, v1, v8
	s_add_i32 s37, s30, 0x2000
	s_movk_i32 s38, 0xd900
	s_movk_i32 s39, 0xf800
	s_add_i32 s36, s30, 0x6000
	s_mov_b32 s26, s22
	s_mov_b32 s27, s23
	s_movk_i32 s40, 0xff00
	s_movk_i32 s41, 0x102
	s_mov_b32 s6, s22
	s_mov_b32 s7, s23
	s_movk_i32 s42, 0xfefe
	s_movk_i32 s43, 0x6907
	s_mov_b32 s44, 0x5040100
	s_mov_b32 s18, s22
	s_mov_b32 s19, s23
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v7, v0
	v_mov_b32_e32 v43, v18
	v_mov_b32_e32 v44, v36
.LBB0_3:
	s_add_i32 s45, s46, 2
	v_add_u32_e32 v45, v43, v17
	s_mov_b32 m0, s37
	v_add_u32_e32 v46, 0x80, v45
	s_waitcnt vmcnt(0)
	s_barrier
	buffer_load_dwordx4 v46, s[20:23], 0 offen lds
	v_add_u32_e32 v46, s13, v17
	v_add_u16_e32 v47, 0x80, v46
	v_lshrrev_b16_e32 v47, 4, v47
	v_mul_u32_u24_e32 v47, 0xd21, v47
	v_lshrrev_b32_e32 v47, 17, v47
	v_add_u32_e32 v48, v23, v47
	v_mul_u32_u24_e32 v48, 0x2700, v48
	v_mad_i32_i24 v47, v47, s38, v48
	v_add3_u32 v47, v37, v47, s39
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v47, s[24:27], 0 offen lds
	v_add_u32_e32 v47, v15, v40
	v_add_u32_e32 v48, 0x100, v47
	v_add3_u32 v49, v38, v39, s41
	v_cmp_gt_i32_e32 vcc, s40, v47
	v_add_u32_e32 v52, v38, v42
	v_add_u32_e32 v53, 0xfffffeff, v52
	v_cndmask_b32_e32 v49, v48, v49, vcc
	v_mul_hi_i32 v50, v49, s34
	v_add_u32_e32 v49, v50, v49
	v_lshrrev_b32_e32 v50, 31, v49
	v_ashrrev_i32_e32 v49, 9, v49
	v_add_u32_e32 v49, v49, v50
	v_cndmask_b32_e64 v50, 0, -1, vcc
	v_xad_u32 v49, v49, v50, v24
	v_add_u32_e32 v50, v15, v36
	v_add_u32_e32 v51, 0x100, v50
	v_cmp_gt_i32_e32 vcc, s40, v50
	v_mul_lo_u32 v49, v49, s14
	v_add_u32_e32 v55, v38, v41
	v_cndmask_b32_e32 v53, v51, v53, vcc
	v_mul_hi_i32 v54, v53, s34
	v_add_u32_e32 v53, v54, v53
	v_lshrrev_b32_e32 v54, 31, v53
	v_ashrrev_i32_e32 v53, 9, v53
	v_add_u32_e32 v53, v53, v54
	v_cndmask_b32_e64 v54, 0, -1, vcc
	v_xor_b32_e32 v53, v53, v54
	v_mul_i32_i24_e32 v53, 0xfffffd90, v53
	v_add_u32_e32 v54, v44, v15
	v_add3_u32 v49, v53, v49, v54
	v_add_u32_e32 v53, 0x102, v47
	v_add_u32_e32 v56, 0xfffffefd, v55
	v_cmp_gt_i32_e32 vcc, -2, v48
	ds_read_b128 v[58:61], v28
	ds_read_b128 v[62:65], v29 offset:16384
	ds_read_b128 v[66:69], v29 offset:18432
	v_cndmask_b32_e32 v56, v53, v56, vcc
	v_mul_hi_i32 v57, v56, s34
	v_add_u32_e32 v56, v57, v56
	v_lshrrev_b32_e32 v57, 31, v56
	v_ashrrev_i32_e32 v56, 9, v56
	v_add_u32_e32 v56, v56, v57
	v_ashrrev_i32_e32 v53, 31, v53
	v_xad_u32 v53, v56, v53, v24
	v_add_u32_e32 v56, 0x102, v50
	v_add_u32_e32 v57, 0xfffffefd, v52
	v_cmp_gt_i32_e32 vcc, s42, v50
	v_mul_lo_u32 v53, v53, s14
	s_nop 0
	v_cndmask_b32_e32 v56, v56, v57, vcc
	v_mul_hi_i32 v57, v56, s34
	v_add_u32_e32 v56, v57, v56
	v_lshrrev_b32_e32 v57, 31, v56
	v_ashrrev_i32_e32 v56, 9, v56
	v_add_u32_e32 v56, v56, v57
	v_cndmask_b32_e64 v57, 0, -1, vcc
	v_xor_b32_e32 v56, v56, v57
	v_mad_i32_i24 v53, v56, s15, v53
	v_add3_u32 v53, v15, v53, v44
	buffer_load_ubyte v56, v49, s[4:7], 0 offen offset:256
	buffer_load_ubyte v57, v53, s[4:7], 0 offen offset:258
	s_barrier
	s_setprio 1
	v_and_b32_e32 v33, 0xff, v33
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[58:61], v[62:65], v[0:3], v33, v34 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[58:61], v[66:69], v[4:7], v33, v34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	ds_read_b128 v[58:61], v27
	ds_read_b128 v[62:65], v26 offset:16384
	ds_read_b128 v[66:69], v26 offset:18432
	s_waitcnt vmcnt(3)
	s_barrier
	s_setprio 1
	v_and_b32_e32 v31, 0xff, v31
	s_waitcnt lgkmcnt(1)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[58:61], v[62:65], v[0:3], v31, v34 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[58:61], v[66:69], v[4:7], v31, v34 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_lshl_b32 s0, s46, 6
	s_mov_b32 m0, s30
	v_add_u32_e32 v31, 0x100, v45
	s_waitcnt vmcnt(0)
	s_barrier
	buffer_load_dwordx4 v31, s[20:23], 0 offen lds
	v_add_u16_e32 v31, 0x100, v46
	v_lshrrev_b16_e32 v31, 4, v31
	v_mul_u32_u24_e32 v31, 0xd21, v31
	v_lshrrev_b32_e32 v31, 17, v31
	v_add_u32_e32 v33, v23, v31
	v_mul_i32_i24_e32 v31, 0xffffd900, v31
	v_mul_u32_u24_e32 v33, 0x2700, v33
	v_add3_u32 v31, v31, v33, v37
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v31, s[24:27], 0 offen lds
	v_add_u32_e32 v31, 0x200, v47
	v_add_u32_e32 v33, 0xfffffdff, v55
	v_cmp_gt_i32_e32 vcc, s40, v48
	v_add_u32_e32 v45, 0xfffffdff, v52
	s_addk_i32 s0, 0xc0
	v_cndmask_b32_e32 v33, v31, v33, vcc
	v_mul_hi_i32 v34, v33, s34
	v_add_u32_e32 v33, v34, v33
	v_lshrrev_b32_e32 v34, 31, v33
	v_ashrrev_i32_e32 v33, 9, v33
	v_add_u32_e32 v33, v33, v34
	v_ashrrev_i32_e32 v34, 31, v31
	v_xad_u32 v33, v33, v34, v24
	v_add_u32_e32 v34, 0x200, v50
	v_cmp_gt_i32_e32 vcc, s40, v51
	v_mul_lo_u32 v33, v33, s14
	s_nop 0
	v_cndmask_b32_e32 v45, v34, v45, vcc
	v_mul_hi_i32 v46, v45, s34
	v_add_u32_e32 v45, v46, v45
	v_lshrrev_b32_e32 v46, 31, v45
	v_ashrrev_i32_e32 v45, 9, v45
	v_add_u32_e32 v45, v45, v46
	v_ashrrev_i32_e32 v34, 31, v34
	v_xor_b32_e32 v34, v45, v34
	v_mul_i32_i24_e32 v34, 0xfffffd90, v34
	v_add3_u32 v34, v34, v33, v54
	v_add_u32_e32 v33, 0x202, v47
	v_add_u32_e32 v45, 0xfffffdfd, v55
	v_cmp_gt_i32_e32 vcc, -2, v31
	s_nop 1
	v_cndmask_b32_e32 v31, v33, v45, vcc
	v_mul_hi_i32 v45, v31, s34
	v_add_u32_e32 v31, v45, v31
	v_lshrrev_b32_e32 v45, 31, v31
	v_ashrrev_i32_e32 v31, 9, v31
	v_add_u32_e32 v31, v31, v45
	v_ashrrev_i32_e32 v33, 31, v33
	v_xad_u32 v31, v31, v33, v24
	v_add_u32_e32 v33, 0x202, v50
	v_add_u32_e32 v45, 0xfffffdfd, v52
	v_cmp_gt_i32_e32 vcc, s42, v51
	v_mul_lo_u32 v31, v31, s14
	s_nop 0
	v_cndmask_b32_e32 v45, v33, v45, vcc
	v_mul_hi_i32 v46, v45, s34
	v_add_u32_e32 v45, v46, v45
	v_lshrrev_b32_e32 v46, 31, v45
	v_ashrrev_i32_e32 v45, 9, v45
	v_add_u32_e32 v45, v45, v46
	v_ashrrev_i32_e32 v33, 31, v33
	v_xor_b32_e32 v33, v45, v33
	v_mad_i32_i24 v31, v33, s15, v31
	v_add3_u32 v45, v15, v31, v44
	buffer_load_ubyte v33, v34, s[4:7], 0 offen offset:512
	buffer_load_ubyte v31, v45, s[4:7], 0 offen offset:514
	v_lshl_add_u32 v45, s45, 6, v10
	v_add_u32_e32 v34, s0, v9
	v_add_u32_e32 v46, 64, v45
	v_sub_u32_e32 v48, 0xffbf, v45
	v_cmp_gt_i32_e64 s[0:1], s33, v45
	v_add_u32_e32 v47, 64, v34
	v_sub_u32_e32 v49, 0xffbf, v34
	v_cmp_gt_i32_e32 vcc, s33, v34
	v_cndmask_b32_e64 v45, v46, v48, s[0:1]
	v_mul_i32_i24_sdwa v45, sext(v45), s43 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e32 v34, v47, v49, vcc
	v_lshrrev_b32_e32 v46, 31, v45
	v_ashrrev_i32_e32 v45, 22, v45
	v_mul_i32_i24_sdwa v34, sext(v34), s43 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_add_u16_e32 v45, v45, v46
	v_lshrrev_b32_e32 v46, 31, v34
	v_ashrrev_i32_e32 v34, 22, v34
	v_add_u16_e32 v34, v34, v46
	v_perm_b32 v34, v34, v45, s44
	v_cndmask_b32_e64 v45, 0, -1, s[0:1]
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_perm_b32 v45, v46, v45, s44
	v_xor_b32_e32 v34, v34, v45
	v_ashrrev_i32_e32 v45, 16, v34
	v_bfe_i32 v34, v34, 0, 16
	v_add_u32_e32 v46, v11, v34
	v_add_u32_e32 v47, v11, v45
	v_mul_lo_u32 v46, v46, s12
	v_mul_lo_u32 v47, v47, s35
	v_mad_i32_i24 v34, v34, s15, v46
	v_add3_u32 v46, v34, v15, v35
	v_mad_i32_i24 v34, v45, s15, v47
	v_add3_u32 v45, v34, v15, v35
	buffer_load_dword v47, v46, s[16:19], 0 offen offset:256
	buffer_load_dword v34, v45, s[16:19], 0 offen offset:512
	ds_read_b128 v[48:51], v28 offset:8192
	ds_read_b128 v[52:55], v29 offset:24576
	ds_read_b128 v[58:61], v29 offset:26624
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(1) lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[48:51], v[52:55], v[0:3], v56, v47 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[48:51], v[58:61], v[4:7], v56, v47 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	ds_read_b128 v[48:51], v27 offset:8192
	ds_read_b128 v[52:55], v26 offset:24576
	ds_read_b128 v[58:61], v26 offset:26624
	s_waitcnt vmcnt(3)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[48:51], v[52:55], v[0:3], v57, v47 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[48:51], v[58:61], v[4:7], v57, v47 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_addk_i32 s13, 0x100
	v_add_u32_e32 v44, 0x200, v44
	v_add_u32_e32 v35, 0x200, v35
	v_add_u32_e32 v37, 0x1000, v37
	v_add_u32_e32 v43, 0x100, v43
	v_add_u32_e32 v39, 0xfffffe00, v39
	v_add_u32_e32 v40, 0x200, v40
	v_add_u32_e32 v41, 0xfffffe00, v41
	v_add_u32_e32 v42, 0xfffffe00, v42
	v_add_u32_e32 v36, 0x200, v36
	s_cmpk_lt_u32 s45, 0x4a
	s_mov_b32 s46, s45
	s_cbranch_scc1 .LBB0_3
	s_movk_i32 s0, 0x2680
	s_mov_b32 m0, s37
	v_add3_u32 v9, v17, v18, s0
	s_waitcnt vmcnt(0)
	s_barrier
	buffer_load_dwordx4 v9, s[20:23], 0 offen lds
	v_or3_b32 v9, s29, v16, 15
	s_movk_i32 s0, 0x2700
	v_mad_u32_u24 v9, v9, s0, v20
	s_movk_i32 s0, 0x1f00
	v_add3_u32 v9, v9, v19, s0
	s_mov_b32 s26, s22
	s_mov_b32 s27, s23
	s_mov_b32 m0, s36
	s_nop 0
	buffer_load_dwordx4 v9, s[24:27], 0 offen lds
	v_add_u32_e32 v10, 0x4d00, v25
	v_add_u32_e32 v9, 0x4d00, v21
	s_mov_b32 s0, 0x1a41a41b
	v_lshrrev_b32_e32 v16, 4, v10
	v_lshrrev_b32_e32 v9, 4, v9
	v_mul_hi_u32 v16, v16, s0
	v_mul_hi_u32 v9, v9, s0
	v_lshrrev_b32_e32 v16, 2, v16
	v_lshrrev_b32_e32 v9, 2, v9
	v_mul_u32_u24_e32 v16, 0x270, v16
	v_add_u32_e32 v9, v9, v24
	v_sub_u32_e32 v16, v10, v16
	v_sub_u32_e32 v10, v16, v10
	v_mul_lo_u32 v9, v9, s14
	v_add3_u32 v9, v9, v10, v8
	v_add3_u32 v9, v9, v22, v30
	s_movk_i32 s1, 0x4000
	v_add_u32_e32 v10, 0x4d02, v25
	v_add3_u32 v16, v9, v15, s1
	v_add_u32_e32 v9, 0x4d02, v21
	v_lshrrev_b32_e32 v17, 4, v10
	v_lshrrev_b32_e32 v9, 4, v9
	v_mul_hi_u32 v17, v17, s0
	v_mul_hi_u32 v9, v9, s0
	v_lshrrev_b32_e32 v17, 2, v17
	v_lshrrev_b32_e32 v9, 2, v9
	v_mul_u32_u24_e32 v17, 0x270, v17
	v_add_u32_e32 v9, v9, v24
	v_sub_u32_e32 v17, v10, v17
	v_sub_u32_e32 v10, v17, v10
	v_mul_lo_u32 v9, v9, s14
	v_add3_u32 v9, v9, v10, v8
	s_mov_b32 s6, s22
	s_mov_b32 s7, s23
	v_add3_u32 v9, v9, v22, v30
	v_add3_u32 v17, v9, v15, s1
	buffer_load_ubyte v10, v16, s[4:7], 0 offen offset:3328
	buffer_load_ubyte v9, v17, s[4:7], 0 offen offset:3330
	v_add_u32_e32 v16, v12, v32
	v_add_u16_e32 v16, 0x1340, v16
	v_lshrrev_b16_e32 v16, 2, v16
	v_mul_u32_u24_e32 v16, 0xd21, v16
	v_lshrrev_b32_e32 v16, 17, v16
	v_add_u32_e32 v16, v11, v16
	s_mov_b32 s18, s22
	s_mov_b32 s19, s23
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[16:17], s[0:1], s12, v16, v[8:9]
	v_add3_u32 v8, v16, v30, v15
	buffer_load_dword v8, v8, s[16:19], 0 offen offset:368
	ds_read_b128 v[16:19], v28
	ds_read_b128 v[20:23], v29 offset:16384
	ds_read_b128 v[36:39], v29 offset:18432
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[16:19], v[20:23], v[0:3], v33, v34 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[16:19], v[16:19], v[36:39], v[4:7], v33, v34 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	ds_read_b128 v[20:23], v27
	ds_read_b128 v[4:7], v26 offset:16384
	ds_read_b128 v[36:39], v26 offset:18432
	s_waitcnt vmcnt(3)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[20:23], v[4:7], v[0:3], v31, v34 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[20:23], v[36:39], v[16:19], v31, v34 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	s_barrier
	ds_read_b128 v[16:19], v28 offset:8192
	ds_read_b128 v[20:23], v29 offset:24576
	ds_read_b128 v[28:31], v29 offset:26624
	v_and_b32_e32 v10, 0xffff, v10
	v_and_b32_e32 v9, 0xffff, v9
	s_movk_i32 s0, 0x7fff
	s_waitcnt vmcnt(0) lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[16:19], v[20:23], v[4:7], v10, v8 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v27 offset:8192
	s_mul_hi_u32 s1, s8, s28
	s_mov_b32 s3, 0x27000
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[16:19], v[28:31], v[0:3], v10, v8 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[16:19], v26 offset:24576
	ds_read_b128 v[24:27], v26 offset:26624
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[20:23], v[16:19], v[4:7], v9, v8 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v16, 0x7fc0
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[20:23], v[24:27], v[0:3], v9, v8 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_nop 4
	v_bfe_u32 v8, v7, 16, 1
	v_bfe_u32 v9, v6, 16, 1
	v_add3_u32 v8, v7, v8, s0
	v_bfe_u32 v10, v5, 16, 1
	v_add3_u32 v9, v6, v9, s0
	v_lshrrev_b32_e32 v8, 16, v8
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v15, v4, 16, 1
	v_add3_u32 v10, v5, v10, s0
	v_lshrrev_b32_e32 v9, 16, v9
	v_cndmask_b32_e32 v7, v16, v8, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v15, v4, v15, s0
	v_lshrrev_b32_e32 v10, 16, v10
	v_cndmask_b32_e32 v6, v16, v9, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_lshrrev_b32_e32 v15, 16, v15
	v_bfe_u32 v8, v3, 16, 1
	v_cndmask_b32_e32 v5, v16, v10, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v9, v2, 16, 1
	v_bfe_u32 v10, v1, 16, 1
	v_cndmask_b32_e32 v4, v16, v15, vcc
	v_bfe_u32 v15, v0, 16, 1
	v_add3_u32 v15, v0, v15, s0
	v_add3_u32 v10, v1, v10, s0
	v_add3_u32 v9, v2, v9, s0
	v_add3_u32 v8, v3, v8, s0
	s_mul_i32 s0, s9, s28
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s8, s28
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s10, s0
	s_addc_u32 s1, s11, s1
	s_lshl_b32 s2, s29, 1
	v_lshrrev_b32_e32 v8, 16, v8
	v_cmp_o_f32_e32 vcc, v3, v3
	s_add_u32 s0, s0, s2
	v_lshrrev_b32_e32 v9, 16, v9
	v_cndmask_b32_e32 v3, v16, v8, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	s_addc_u32 s1, s1, 0
	s_and_b32 s2, s8, 0x3fff
	v_cndmask_b32_e32 v2, v16, v9, vcc
	v_lshl_or_b32 v8, v13, 2, v14
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	v_lshlrev_b32_e32 v9, 1, v11
	v_mul_lo_u32 v8, s8, v8
	s_or_b32 s1, s2, s1
	v_lshl_add_u32 v9, v12, 1, v9
	s_or_b32 s1, s1, 2.0
	s_mov_b32 s2, 0x7ffffffd
	v_lshl_add_u32 v8, v8, 1, v9
	s_lshl_b32 s4, s8, 1
	buffer_store_short v4, v8, s[0:3], 0 offen
	v_add_u32_e32 v4, s4, v8
	v_lshrrev_b32_e32 v10, 16, v10
	v_cmp_o_f32_e32 vcc, v1, v1
	buffer_store_short v5, v4, s[0:3], 0 offen
	v_add_u32_e32 v5, s4, v4
	v_lshrrev_b32_e32 v15, 16, v15
	v_cndmask_b32_e32 v1, v16, v10, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	buffer_store_short v6, v5, s[0:3], 0 offen
	v_add_u32_e32 v6, s4, v5
	v_cndmask_b32_e32 v0, v16, v15, vcc
	buffer_store_short v7, v6, s[0:3], 0 offen
	buffer_store_short v0, v8, s[0:3], 0 offen offset:32
	buffer_store_short v1, v4, s[0:3], 0 offen offset:32
	buffer_store_short v2, v5, s[0:3], 0 offen offset:32
	buffer_store_short v3, v6, s[0:3], 0 offen offset:32
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_64x64x256_35200x256x19968
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
		.amdhsa_next_free_vgpr 70
		.amdhsa_next_free_sgpr 47
		.amdhsa_accum_offset 72
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
	.size	wave_mxfp4_static_gemm_64x64x256_35200x256x19968, .Lfunc_end0-wave_mxfp4_static_gemm_64x64x256_35200x256x19968

	.set wave_mxfp4_static_gemm_64x64x256_35200x256x19968.num_vgpr, 70
	.set wave_mxfp4_static_gemm_64x64x256_35200x256x19968.num_agpr, 0
	.set wave_mxfp4_static_gemm_64x64x256_35200x256x19968.numbered_sgpr, 47
	.set wave_mxfp4_static_gemm_64x64x256_35200x256x19968.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_64x64x256_35200x256x19968.private_seg_size, 0
	.set wave_mxfp4_static_gemm_64x64x256_35200x256x19968.uses_vcc, 1
	.set wave_mxfp4_static_gemm_64x64x256_35200x256x19968.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_64x64x256_35200x256x19968.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_64x64x256_35200x256x19968.has_recursion, 0
	.set wave_mxfp4_static_gemm_64x64x256_35200x256x19968.has_indirect_call, 0
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
    .name:           wave_mxfp4_static_gemm_64x64x256_35200x256x19968
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     53
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_64x64x256_35200x256x19968.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     70
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
