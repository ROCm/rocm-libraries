; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_64x64x256_896x7040x1280
	.p2align	8
	.type	wave_mxfp4_static_gemm_64x64x256_896x7040x1280,@function
wave_mxfp4_static_gemm_64x64x256_896x7040x1280:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v5, 0x3ff, v0
	v_bfe_u32 v2, v0, 10, 10
	v_lshrrev_b32_e32 v4, 6, v5
	v_lshlrev_b32_e32 v8, 5, v2
	v_lshl_or_b32 v0, v4, 3, v8
	s_mov_b64 s[24:25], s[2:3]
	v_readfirstlane_b32 s2, v0
	v_lshrrev_b32_e32 v0, 3, v5
	v_or_b32_e32 v1, v0, v8
	s_lshl_b32 s28, s16, 6
	v_or_b32_e32 v1, s28, v1
	v_bitop3_b32 v9, v0, 7, v5 bitop3:0x48
	v_lshlrev_b32_e32 v12, 4, v9
	v_mul_u32_u24_e32 v1, 0x280, v1
	s_and_b32 s3, s25, 0xffff
	s_lshl_b32 s31, s2, 7
	s_or_b32 s25, s3, 0x42800000
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_or_b32_e32 v3, v1, v12
	s_mov_b32 m0, s31
	s_lshl_b32 s29, s17, 6
	buffer_load_dwordx4 v3, s[24:27], 0 offen lds
	v_bfe_u32 v3, v5, 3, 4
	v_or_b32_e32 v6, v12, v3
	v_mul_lo_u16_e32 v6, 0x67, v6
	v_lshrrev_b16_e32 v10, 12, v6
	v_bitop3_b32 v6, v0, 48, v8 bitop3:0xc8
	v_lshlrev_b32_e32 v7, 1, v5
	v_or3_b32 v11, s29, v10, v6
	v_lshlrev_b32_e32 v6, 4, v0
	v_and_b32_e32 v7, 0x100, v7
	s_mov_b64 s[20:21], s[6:7]
	s_movk_i32 s30, 0x280
	v_sub_u32_e32 v13, v6, v7
	v_mul_i32_i24_e32 v10, 0xfffffd80, v10
	v_lshlrev_b32_e32 v29, 8, v9
	s_and_b32 s2, s21, 0xffff
	v_mad_u32_u24 v9, v11, s30, v13
	s_or_b32 s21, s2, 0x42800000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_add3_u32 v9, v9, v10, v29
	s_add_i32 m0, s31, 0x4000
	v_lshrrev_b32_e32 v14, 4, v5
	buffer_load_dwordx4 v9, s[20:23], 0 offen lds
	v_lshlrev_b32_e32 v11, 4, v4
	v_mad_i32_i24 v9, v14, -16, v5
	v_add_u32_e32 v13, v9, v11
	v_ashrrev_i16_e32 v16, 15, v13
	v_lshrrev_b16_e32 v16, 11, v16
	v_add_u16_e32 v16, v13, v16
	v_and_b32_e32 v16, 0xffffffe0, v16
	v_sub_u16_e32 v16, v13, v16
	v_bfe_i32 v17, v16, 0, 16
	v_ashrrev_i32_e32 v18, 31, v17
	v_add_u16_e32 v19, 32, v16
	v_cmp_gt_i16_e32 vcc, 0, v16
	v_lshlrev_b32_e32 v24, 2, v5
	s_movk_i32 s33, 0xffc0
	v_cndmask_b32_e32 v16, v17, v19, vcc
	v_cndmask_b32_e64 v17, v18, 0, vcc
	v_xor_b32_e32 v16, v17, v16
	v_lshrrev_b32_e32 v18, 28, v16
	v_add_u32_e32 v16, v16, v18
	v_bfe_u32 v10, v5, 4, 2
	v_ashrrev_i32_e32 v16, 4, v16
	v_mad_i32_i24 v15, v14, s33, v24
	v_lshlrev_b32_e32 v23, 6, v10
	v_xor_b32_e32 v25, v16, v17
	v_add3_u32 v18, v15, v23, v25
	v_ashrrev_i32_e32 v16, 31, v18
	v_xor_b32_e32 v17, v16, v18
	s_mov_b32 s34, 0x66666667
	v_mul_hi_i32 v17, v17, s34
	v_lshrrev_b32_e32 v19, 31, v17
	v_ashrrev_i32_e32 v17, 4, v17
	v_add_u32_e32 v17, v17, v19
	v_ashrrev_i32_e32 v19, 31, v13
	v_xor_b32_e32 v13, v19, v13
	v_ashrrev_i32_e32 v20, 31, v13
	v_lshrrev_b32_e32 v20, 27, v20
	v_add_u32_e32 v13, v13, v20
	v_lshrrev_b32_e32 v13, 5, v13
	v_xor_b32_e32 v13, v13, v19
	v_lshlrev_b32_e32 v13, 5, v13
	v_bfe_u32 v26, v5, 6, 1
	v_xad_u32 v17, v17, v16, v13
	v_or_b32_e32 v16, v15, v26
	v_add_u32_e32 v19, v16, v23
	v_mul_hi_i32 v16, v19, s34
	v_lshrrev_b32_e32 v20, 31, v16
	v_ashrrev_i32_e32 v16, 4, v16
	s_mul_i32 s15, s15, s28
	s_mul_hi_u32 s2, s14, s28
	v_add_u32_e32 v16, v16, v20
	s_add_i32 s2, s2, s15
	s_mul_i32 s3, s14, s28
	v_mul_lo_u32 v16, v16, 40
	s_add_u32 s4, s4, s3
	v_sub_u32_e32 v16, v19, v16
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s14, 0x3fff
	v_add_u32_e32 v20, 40, v16
	v_cmp_gt_i32_e32 vcc, 0, v16
	s_bitset1_b32 s3, 14
	s_and_b32 s2, s2, 0xffff
	v_cndmask_b32_e32 v16, v16, v20, vcc
	s_lshl_b32 s3, s3, 16
	s_or_b32 s5, s2, s3
	v_mad_u64_u32 v[16:17], s[2:3], v17, s14, v[16:17]
	v_add_u32_e32 v17, 2, v18
	v_sub_u32_e32 v20, -3, v18
	v_cmp_gt_i32_e32 vcc, -2, v18
	s_movk_i32 s2, 0xffd8
	s_load_dwordx2 s[12:13], s[0:1], 0x40
	v_cndmask_b32_e32 v17, v17, v20, vcc
	v_mul_hi_i32 v17, v17, s34
	v_lshrrev_b32_e32 v18, 31, v17
	v_ashrrev_i32_e32 v17, 4, v17
	v_add_u32_e32 v17, v17, v18
	v_cndmask_b32_e64 v18, 0, -1, vcc
	v_xad_u32 v17, v17, v18, v13
	v_or_b32_e32 v18, 2, v19
	v_xor_b32_e32 v20, -3, v19
	v_cmp_gt_i32_e32 vcc, 0, v19
	v_ashrrev_i32_e32 v19, 31, v19
	v_mul_lo_u32 v17, v17, s14
	v_cndmask_b32_e32 v18, v18, v20, vcc
	v_mul_hi_i32 v18, v18, s34
	v_lshrrev_b32_e32 v20, 31, v18
	v_ashrrev_i32_e32 v18, 4, v18
	v_add_u32_e32 v18, v18, v20
	v_xor_b32_e32 v18, v18, v19
	v_mul_lo_u32 v18, v18, s2
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	v_add3_u32 v17, v17, v18, v23
	v_lshl_add_u32 v31, v10, 4, v9
	v_add3_u32 v17, v17, v26, v15
	buffer_load_ubyte v19, v16, s[4:7], 0 offen
	buffer_load_ubyte v18, v17, s[4:7], 0 offen offset:2
	v_ashrrev_i32_e32 v16, 31, v31
	v_xor_b32_e32 v17, v16, v31
	v_mul_hi_i32 v17, v17, s34
	v_lshrrev_b32_e32 v20, 31, v17
	v_ashrrev_i32_e32 v17, 2, v17
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s3, s13, s29
	s_mul_hi_u32 s6, s12, s29
	v_add_u32_e32 v17, v17, v20
	s_add_i32 s6, s6, s3
	s_mul_i32 s3, s12, s29
	v_xad_u32 v16, v17, v16, v8
	v_and_b32_e32 v17, 62, v5
	s_add_u32 s16, s8, s3
	v_mul_lo_u16_e32 v17, 26, v17
	s_addc_u32 s3, s9, s6
	s_and_b32 s6, s12, 0x3fff
	v_lshrrev_b16_e32 v17, 8, v17
	v_mul_lo_u32 v16, s12, v16
	s_bitset1_b32 s6, 14
	s_and_b32 s3, s3, 0xffff
	s_lshl_b32 s6, s6, 16
	v_mad_i32_i24 v16, v17, s2, v16
	s_or_b32 s17, s3, s6
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	v_add3_u32 v15, v16, v23, v15
	buffer_load_dword v28, v15, s[16:19], 0 offen
	v_cmp_eq_u32_e64 s[2:3], 0, v2
	s_mov_b32 s9, 0
	s_movk_i32 s8, 0x4000
	v_mul_i32_i24_e32 v27, 0xffffffc0, v14
	s_and_b64 vcc, exec, s[2:3]
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	v_and_b32_e32 v16, 7, v5
	v_bitop3_b32 v15, v10, v5, 7 bitop3:0x78
	v_lshlrev_b32_e32 v5, 7, v5
	v_lshlrev_b32_e32 v14, 11, v14
	v_sub_u32_e32 v5, v5, v14
	v_lshl_add_u32 v4, v4, 11, v5
	v_lshl_add_u32 v2, v2, 12, v5
	v_bitop3_b32 v5, v10, v16, 4 bitop3:0x36
	v_lshlrev_b32_e32 v15, 4, v15
	v_lshlrev_b32_e32 v5, 4, v5
	v_or_b32_e32 v14, v4, v15
	v_or_b32_e32 v15, v2, v15
	v_or_b32_e32 v16, v5, v2
	v_add_u32_e32 v2, v23, v26
	v_add3_u32 v20, v2, v27, v24
	v_add_u32_e32 v2, v27, v23
	s_load_dwordx2 s[0:1], s[0:1], 0x48
	v_or_b32_e32 v17, v5, v4
	v_add_u32_e32 v4, v2, v25
	v_add_u32_e32 v0, v0, v8
	v_sub_u32_e32 v23, 0xfffffefd, v4
	v_add_u32_e32 v4, v2, v24
	v_add_u32_e32 v2, v2, v26
	v_and_b32_e32 v0, 0x70, v0
	v_sub_u32_e32 v22, 0, v24
	v_add_u32_e32 v24, v4, v25
	v_add_u32_e32 v25, v4, v26
	v_sub_u32_e32 v26, 0, v2
	v_add_u32_e32 v2, v29, v6
	v_add_u32_e32 v30, s29, v0
	v_add_u32_e32 v0, v12, v3
	v_sub_u32_e32 v2, v2, v7
	v_add_u32_e32 v32, 0x80, v0
	v_add_u32_e32 v33, 8, v0
	v_mov_b32_e32 v0, 0
	v_sub_u32_e32 v21, -5, v31
	v_add_u32_e32 v27, 16, v4
	v_add_u32_e32 v29, 0x80, v2
	v_add_u32_e32 v34, 0x80, v1
	s_movk_i32 s38, 0x2000
	s_movk_i32 s37, 0x6000
	s_mov_b32 s13, 0
	s_mov_b32 s15, 0xcccccccd
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	s_movk_i32 s35, 0xff00
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	s_movk_i32 s36, 0xfefe
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v7, v0
.LBB0_3:
	v_mul_hi_u32 v35, v32, s15
	v_mul_hi_u32 v36, v33, s15
	v_lshrrev_b32_e32 v35, 5, v35
	v_lshrrev_b32_e32 v36, 5, v36
	v_add_u32_e32 v35, v30, v35
	v_mul_lo_u32 v36, v36, s30
	v_mul_lo_u32 v35, v35, s30
	v_sub_u32_e32 v35, v35, v36
	s_mov_b32 s39, s9
	s_mov_b32 s9, s38
	s_mov_b32 s38, s8
	s_mov_b32 s8, s37
	v_add_u32_e32 v35, v29, v35
	v_add_u32_e32 v36, v34, v12
	s_add_i32 m0, s9, s31
	s_barrier
	buffer_load_dwordx4 v36, s[24:27], 0 offen lds
	s_add_i32 m0, s37, s31
	s_nop 0
	buffer_load_dwordx4 v35, s[20:23], 0 offen lds
	v_add_u32_e32 v35, s13, v24
	v_add_u32_e32 v37, v22, v23
	v_add_u32_e32 v36, 0x100, v35
	v_add_u32_e32 v38, 2, v37
	v_cmp_gt_i32_e32 vcc, s35, v35
	v_add_u32_e32 v41, v22, v26
	v_subrev_u32_e32 v42, 17, v41
	v_cndmask_b32_e32 v38, v36, v38, vcc
	v_mul_hi_i32 v38, v38, s34
	v_lshrrev_b32_e32 v39, 31, v38
	v_ashrrev_i32_e32 v38, 4, v38
	v_add_u32_e32 v38, v38, v39
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_xad_u32 v38, v38, v39, v13
	v_add_u32_e32 v39, s13, v25
	v_add_u32_e32 v40, 16, v39
	v_cmp_gt_i32_e32 vcc, -16, v39
	v_add_u32_e32 v35, 0x102, v35
	v_mul_lo_u32 v38, v38, s14
	v_cndmask_b32_e32 v40, v40, v42, vcc
	v_mul_hi_i32 v40, v40, s34
	v_lshrrev_b32_e32 v42, 31, v40
	v_ashrrev_i32_e32 v40, 4, v40
	v_add_u32_e32 v40, v40, v42
	v_cndmask_b32_e64 v42, 0, -1, vcc
	v_cmp_gt_i32_e32 vcc, -2, v36
	v_xor_b32_e32 v40, v40, v42
	v_mul_lo_u32 v40, v40, 40
	v_cndmask_b32_e32 v36, v35, v37, vcc
	v_mul_hi_i32 v36, v36, s34
	v_lshrrev_b32_e32 v37, 31, v36
	v_ashrrev_i32_e32 v36, 4, v36
	v_add_u32_e32 v36, v36, v37
	v_ashrrev_i32_e32 v35, 31, v35
	v_xad_u32 v35, v36, v35, v13
	v_add_u32_e32 v36, 0x102, v39
	v_add_u32_e32 v37, 0xfffffefd, v41
	v_cmp_gt_i32_e32 vcc, s36, v39
	v_mul_lo_u32 v35, v35, s14
	v_sub_u32_e32 v38, v38, v40
	v_cndmask_b32_e32 v36, v36, v37, vcc
	v_mul_hi_i32 v36, v36, s34
	v_lshrrev_b32_e32 v37, 31, v36
	v_ashrrev_i32_e32 v36, 4, v36
	v_add_u32_e32 v36, v36, v37
	v_cndmask_b32_e64 v37, 0, -1, vcc
	v_xor_b32_e32 v36, v36, v37
	v_mul_lo_u32 v36, v36, 40
	v_sub_u32_e32 v35, v35, v36
	v_add3_u32 v38, s13, v38, v20
	v_add3_u32 v37, v20, v35, s13
	buffer_load_ubyte v36, v38, s[4:7], 0 offen offset:16
	buffer_load_ubyte v35, v37, s[4:7], 0 offen offset:258
	v_add_u32_e32 v38, 64, v31
	v_subrev_u32_e32 v37, 60, v21
	v_cmp_gt_i32_e32 vcc, s33, v31
	s_nop 1
	v_cndmask_b32_e32 v37, v38, v37, vcc
	v_mul_hi_i32 v37, v37, s34
	v_lshrrev_b32_e32 v39, 31, v37
	v_ashrrev_i32_e32 v37, 2, v37
	v_add_u32_e32 v37, v37, v39
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_xad_u32 v37, v37, v39, v8
	v_add_u32_e32 v39, 4, v31
	v_cmp_gt_i32_e32 vcc, -4, v31
	v_mul_lo_u32 v37, v37, s12
	s_nop 0
	v_cndmask_b32_e32 v31, v39, v21, vcc
	v_mul_hi_i32 v31, v31, s34
	v_lshrrev_b32_e32 v39, 31, v31
	v_lshrrev_b32_e32 v31, 2, v31
	v_add_u32_e32 v31, v31, v39
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_xor_b32_e32 v31, v31, v39
	v_mul_lo_u32 v31, v31, 40
	v_sub_u32_e32 v31, v37, v31
	v_add3_u32 v31, v27, v31, s13
	buffer_load_dword v37, v31, s[16:19], 0 offen
	v_add_u32_e32 v31, s39, v14
	ds_read_b128 v[40:43], v31
	v_add_u32_e32 v31, s38, v15
	ds_read_b128 v[44:47], v31
	ds_read_b128 v[48:51], v31 offset:2048
	s_barrier
	s_setprio 1
	v_and_b32_e32 v19, 0xff, v19
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[40:43], v[44:47], v[0:3], v19, v28 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[40:43], v[48:51], v[4:7], v19, v28 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v19, s39, v17
	ds_read_b128 v[40:43], v19
	v_add_u32_e32 v19, s38, v16
	ds_read_b128 v[44:47], v19
	ds_read_b128 v[48:51], v19 offset:2048
	s_waitcnt vmcnt(3)
	s_barrier
	s_setprio 1
	v_and_b32_e32 v18, 0xff, v18
	s_waitcnt lgkmcnt(1)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[40:43], v[44:47], v[0:3], v18, v28 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[40:43], v[48:51], v[4:7], v18, v28 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_addk_i32 s13, 0x100
	v_subrev_u32_e32 v21, 64, v21
	v_add_u32_e32 v23, 0xffffff00, v23
	v_add_u32_e32 v26, 0xffffff00, v26
	v_add_u32_e32 v29, 0x800, v29
	v_add_u32_e32 v32, 0x80, v32
	v_add_u32_e32 v33, 0x80, v33
	v_add_u32_e32 v34, 0x80, v34
	s_cmpk_lg_i32 s13, 0x400
	v_mov_b32_e32 v31, v38
	s_mov_b32 s37, s38
	s_mov_b32 s38, s39
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v28, v37
	v_mov_b32_e32 v18, v35
	v_mov_b32_e32 v19, v36
	s_cbranch_scc1 .LBB0_3
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	v_add_u32_e32 v12, s9, v14
	s_barrier
	ds_read_b128 v[18:21], v12
	v_add_u32_e32 v13, s9, v17
	ds_read_b128 v[26:29], v13
	v_add_u32_e32 v12, s8, v15
	ds_read_b128 v[22:25], v12
	v_add_u32_e32 v13, s8, v16
	ds_read_b128 v[14:17], v12 offset:2048
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[18:21], v[22:25], v[0:3], v36, v37 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[22:25], v13
	ds_read_b128 v[30:33], v13 offset:2048
	s_movk_i32 s2, 0x7fff
	s_mul_i32 s1, s1, s28
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[26:29], v[22:25], v[0:3], v35, v37 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_lshl_or_b32 v10, v10, 2, v11
	v_lshlrev_b32_e32 v8, 1, v8
	v_mul_lo_u32 v10, s0, v10
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[18:21], v[14:17], v[4:7], v36, v37 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 3
	v_bfe_u32 v12, v3, 16, 1
	v_bfe_u32 v13, v2, 16, 1
	v_add3_u32 v12, v3, v12, s2
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[26:29], v[30:33], v[4:7], v35, v37 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
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
	s_mul_hi_u32 s2, s0, s28
	s_add_i32 s3, s2, s1
	s_mul_i32 s2, s0, s28
	s_lshl_b64 s[2:3], s[2:3], 1
	s_add_u32 s1, s10, s2
	s_addc_u32 s2, s11, s3
	s_lshl_b32 s3, s29, 1
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
	.amdhsa_kernel wave_mxfp4_static_gemm_64x64x256_896x7040x1280
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
		.amdhsa_next_free_vgpr 52
		.amdhsa_next_free_sgpr 40
		.amdhsa_accum_offset 52
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
	.size	wave_mxfp4_static_gemm_64x64x256_896x7040x1280, .Lfunc_end0-wave_mxfp4_static_gemm_64x64x256_896x7040x1280

	.set wave_mxfp4_static_gemm_64x64x256_896x7040x1280.num_vgpr, 52
	.set wave_mxfp4_static_gemm_64x64x256_896x7040x1280.num_agpr, 0
	.set wave_mxfp4_static_gemm_64x64x256_896x7040x1280.numbered_sgpr, 40
	.set wave_mxfp4_static_gemm_64x64x256_896x7040x1280.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_64x64x256_896x7040x1280.private_seg_size, 0
	.set wave_mxfp4_static_gemm_64x64x256_896x7040x1280.uses_vcc, 1
	.set wave_mxfp4_static_gemm_64x64x256_896x7040x1280.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_64x64x256_896x7040x1280.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_64x64x256_896x7040x1280.has_recursion, 0
	.set wave_mxfp4_static_gemm_64x64x256_896x7040x1280.has_indirect_call, 0
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
    .name:           wave_mxfp4_static_gemm_64x64x256_896x7040x1280
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     46
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_64x64x256_896x7040x1280.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     52
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
