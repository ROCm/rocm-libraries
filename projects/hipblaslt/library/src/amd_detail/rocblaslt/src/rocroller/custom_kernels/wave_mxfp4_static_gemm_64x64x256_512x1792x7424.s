; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_64x64x256_512x1792x7424
	.p2align	8
	.type	wave_mxfp4_static_gemm_64x64x256_512x1792x7424,@function
wave_mxfp4_static_gemm_64x64x256_512x1792x7424:
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
	v_lshlrev_b32_e32 v10, 5, v0
	v_lshl_or_b32 v3, v2, 3, v10
	s_mov_b64 s[24:25], s[2:3]
	v_readfirstlane_b32 s2, v3
	v_lshrrev_b32_e32 v3, 3, v1
	v_or_b32_e32 v4, v3, v10
	s_lshl_b32 s28, s16, 6
	v_or_b32_e32 v5, s28, v4
	v_bitop3_b32 v6, v3, 7, v1 bitop3:0x48
	s_lshl_b32 s29, s17, 6
	s_mov_b64 s[20:21], s[6:7]
	v_lshlrev_b32_e32 v14, 4, v6
	v_mul_u32_u24_e32 v15, 0xe80, v5
	s_and_b32 s6, s25, 0xffff
	s_lshl_b32 s30, s2, 7
	v_and_or_b32 v16, v4, 48, s29
	v_lshlrev_b32_e32 v4, 1, v1
	s_or_b32 s25, s6, 0x4e800000
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	v_or_b32_e32 v5, v15, v14
	s_mov_b32 m0, s30
	v_lshlrev_b32_e32 v3, 4, v3
	v_and_b32_e32 v4, 0x100, v4
	buffer_load_dwordx4 v5, s[24:27], 0 offen lds
	v_sub_u32_e32 v7, v3, v4
	v_lshlrev_b32_e32 v5, 8, v6
	v_add_u32_e32 v6, v5, v7
	v_lshrrev_b32_e32 v7, 4, v1
	v_lshlrev_b32_e32 v13, 4, v2
	v_mad_i32_i24 v11, v7, -16, v1
	v_add_u32_e32 v8, v11, v13
	v_ashrrev_i16_e32 v9, 15, v8
	v_lshrrev_b16_e32 v9, 11, v9
	v_add_u16_e32 v9, v8, v9
	v_and_b32_e32 v9, 0xffffffe0, v9
	v_sub_u16_e32 v9, v8, v9
	v_bfe_i32 v18, v9, 0, 16
	v_ashrrev_i32_e32 v21, 31, v18
	v_add_u16_e32 v22, 32, v9
	v_cmp_gt_i16_e32 vcc, 0, v9
	s_movk_i32 s3, 0xe80
	s_and_b32 s2, s21, 0xffff
	v_cndmask_b32_e32 v9, v18, v22, vcc
	v_cndmask_b32_e64 v18, v21, 0, vcc
	v_xor_b32_e32 v9, v18, v9
	v_lshrrev_b32_e32 v21, 28, v9
	s_add_i32 s31, s30, 0x4000
	v_add_u32_e32 v9, v9, v21
	s_or_b32 s21, s2, 0x4e800000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	v_mad_u32_u24 v6, v16, s3, v6
	s_mov_b32 m0, s31
	v_lshlrev_b32_e32 v17, 2, v1
	s_movk_i32 s33, 0xffc0
	v_bfe_u32 v12, v1, 4, 2
	v_ashrrev_i32_e32 v9, 4, v9
	buffer_load_dwordx4 v6, s[20:23], 0 offen lds
	v_mad_i32_i24 v19, v7, s33, v17
	v_lshlrev_b32_e32 v6, 6, v12
	v_xor_b32_e32 v32, v9, v18
	v_add3_u32 v21, v19, v6, v32
	v_ashrrev_i32_e32 v9, 31, v21
	v_xor_b32_e32 v18, v9, v21
	s_mov_b32 s34, 0x8d3dcb09
	v_mul_hi_i32 v22, v18, s34
	v_add_u32_e32 v18, v22, v18
	v_lshrrev_b32_e32 v22, 31, v18
	v_ashrrev_i32_e32 v18, 7, v18
	v_add_u32_e32 v22, v18, v22
	v_ashrrev_i32_e32 v18, 31, v8
	v_xor_b32_e32 v8, v18, v8
	v_ashrrev_i32_e32 v23, 31, v8
	v_lshrrev_b32_e32 v23, 27, v23
	v_add_u32_e32 v8, v8, v23
	v_lshrrev_b32_e32 v8, 5, v8
	v_xor_b32_e32 v8, v8, v18
	v_bfe_u32 v33, v1, 6, 1
	v_lshlrev_b32_e32 v18, 5, v8
	v_or_b32_e32 v8, v19, v33
	v_xad_u32 v9, v22, v9, v18
	v_add_u32_e32 v22, v8, v6
	v_mul_hi_i32 v8, v22, s34
	v_add_u32_e32 v8, v8, v22
	v_lshrrev_b32_e32 v23, 31, v8
	v_ashrrev_i32_e32 v8, 7, v8
	s_mul_i32 s15, s15, s28
	s_mul_hi_u32 s2, s14, s28
	v_add_u32_e32 v8, v8, v23
	s_movk_i32 s35, 0xe8
	s_add_i32 s2, s2, s15
	s_mul_i32 s3, s14, s28
	v_mul_lo_u32 v8, v8, s35
	s_add_u32 s4, s4, s3
	v_sub_u32_e32 v8, v22, v8
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s14, 0x3fff
	v_add_u32_e32 v23, 0xe8, v8
	v_cmp_gt_i32_e32 vcc, 0, v8
	s_bitset1_b32 s3, 14
	s_and_b32 s2, s2, 0xffff
	v_cndmask_b32_e32 v8, v8, v23, vcc
	s_lshl_b32 s3, s3, 16
	s_or_b32 s5, s2, s3
	v_mad_u64_u32 v[8:9], s[2:3], v9, s14, v[8:9]
	v_add_u32_e32 v9, 2, v21
	v_sub_u32_e32 v23, -3, v21
	v_cmp_gt_i32_e32 vcc, -2, v21
	s_movk_i32 s15, 0xff18
	v_mul_i32_i24_e32 v20, -16, v7
	v_cndmask_b32_e32 v9, v9, v23, vcc
	v_mul_hi_i32 v21, v9, s34
	v_add_u32_e32 v9, v21, v9
	v_lshrrev_b32_e32 v21, 31, v9
	v_ashrrev_i32_e32 v9, 7, v9
	v_add_u32_e32 v9, v9, v21
	v_cndmask_b32_e64 v21, 0, -1, vcc
	v_xad_u32 v9, v9, v21, v18
	v_or_b32_e32 v21, 2, v22
	v_xor_b32_e32 v23, -3, v22
	v_cmp_gt_i32_e32 vcc, 0, v22
	v_ashrrev_i32_e32 v22, 31, v22
	v_mul_lo_u32 v9, v9, s14
	v_cndmask_b32_e32 v21, v21, v23, vcc
	v_mul_hi_i32 v23, v21, s34
	v_add_u32_e32 v21, v23, v21
	v_lshrrev_b32_e32 v23, 31, v21
	v_ashrrev_i32_e32 v21, 7, v21
	v_add_u32_e32 v21, v21, v23
	v_xor_b32_e32 v21, v21, v22
	v_mul_lo_u32 v21, v21, s15
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	v_add3_u32 v9, v9, v21, v6
	s_load_dwordx2 s[12:13], s[0:1], 0x40
	v_add3_u32 v9, v9, v33, v19
	buffer_load_ubyte v29, v8, s[4:7], 0 offen
	buffer_load_ubyte v23, v9, s[4:7], 0 offen offset:2
	v_lshl_add_u32 v8, v12, 4, v20
	v_add_u32_e32 v8, v8, v1
	v_ashrrev_i32_e32 v9, 31, v8
	v_xor_b32_e32 v20, v9, v8
	v_mul_hi_i32 v21, v20, s34
	v_add_u32_e32 v20, v21, v20
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s13, s29
	s_mul_hi_u32 s3, s12, s29
	v_lshrrev_b32_e32 v21, 31, v20
	v_ashrrev_i32_e32 v20, 5, v20
	s_add_i32 s3, s3, s2
	s_mul_i32 s2, s12, s29
	v_add_u32_e32 v20, v20, v21
	s_add_u32 s16, s8, s2
	v_xad_u32 v9, v20, v9, v10
	v_and_b32_e32 v20, 62, v1
	s_addc_u32 s2, s9, s3
	s_and_b32 s3, s12, 0x3fff
	v_mov_b32_e32 v21, 0xffffff18
	v_cmp_lt_u32_e32 vcc, 57, v20
	s_bitset1_b32 s3, 14
	s_and_b32 s2, s2, 0xffff
	v_cndmask_b32_e32 v20, 0, v21, vcc
	s_lshl_b32 s3, s3, 16
	s_or_b32 s17, s2, s3
	v_mad_u64_u32 v[20:21], s[2:3], s12, v9, v[20:21]
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	v_add3_u32 v9, v20, v6, v19
	buffer_load_dword v35, v9, s[16:19], 0 offen
	v_cmp_eq_u32_e64 s[2:3], 0, v0
	v_mul_i32_i24_e32 v34, 0xffffffc0, v7
	s_mov_b32 s47, -2
	s_and_b64 vcc, exec, s[2:3]
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	v_lshlrev_b32_e32 v20, 7, v1
	v_lshlrev_b32_e32 v7, 11, v7
	v_and_b32_e32 v9, 7, v1
	v_sub_u32_e32 v7, v20, v7
	v_bitop3_b32 v19, v12, v1, 7 bitop3:0x78
	v_lshl_add_u32 v2, v2, 11, v7
	v_lshl_add_u32 v0, v0, 12, v7
	v_bitop3_b32 v7, v12, v9, 4 bitop3:0x36
	v_lshlrev_b32_e32 v19, 4, v19
	v_lshlrev_b32_e32 v7, 4, v7
	v_or_b32_e32 v20, v2, v19
	v_or_b32_e32 v21, v0, v19
	v_or_b32_e32 v19, v7, v0
	v_add_u32_e32 v0, v5, v3
	v_sub_u32_e32 v0, v0, v4
	s_load_dwordx2 s[8:9], s[0:1], 0x48
	v_add_u32_e32 v27, 0x1000, v0
	v_add3_u32 v0, v34, v32, v6
	v_sub_u32_e32 v30, 0xfffffdfd, v0
	v_sub_u32_e32 v0, 0, v34
	v_bfe_u32 v24, v1, 3, 4
	v_add_u32_e32 v25, v34, v6
	v_sub_u32_e32 v1, v0, v32
	v_sub_u32_e32 v0, v0, v33
	v_add_u32_e32 v26, v25, v33
	v_sub_u32_e32 v33, v0, v6
	v_mov_b32_e32 v0, 0
	v_or_b32_e32 v22, v7, v2
	v_mov_b32_e32 v9, v8
	s_mov_b32 s13, s12
	v_sub_u32_e32 v28, 0, v17
	v_add_u32_e32 v31, v25, v32
	v_sub_u32_e32 v32, v1, v6
	s_add_i32 s36, s30, 0x2000
	s_movk_i32 s37, 0x469f
	s_movk_i32 s38, 0xf180
	s_movk_i32 s39, 0xf800
	s_add_i32 s40, s30, 0x6000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	s_movk_i32 s41, 0xff00
	s_movk_i32 s42, 0x102
	s_movk_i32 s43, 0xffe8
	s_mov_b32 s6, s26
	s_mov_b32 s7, s27
	s_movk_i32 s44, 0xfefe
	s_mov_b32 s45, 0x5040100
	s_mov_b32 s18, s26
	s_mov_b32 s19, s27
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v4, v0
	v_mov_b32_e32 v5, v0
	v_mov_b32_e32 v6, v0
	v_mov_b32_e32 v7, v0
	v_mov_b32_e32 v34, v26
.LBB0_3:
	s_add_i32 s46, s47, 2
	v_add_u32_e32 v36, v15, v14
	s_mov_b32 m0, s36
	v_add_u32_e32 v37, 0x80, v36
	s_waitcnt vmcnt(0)
	s_barrier
	buffer_load_dwordx4 v37, s[24:27], 0 offen lds
	v_add_u32_e32 v37, v14, v24
	v_add_u32_e32 v38, 0x80, v37
	v_mul_u32_u24_sdwa v38, v38, s37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshrrev_b32_e32 v38, 22, v38
	v_add_u32_e32 v39, v16, v38
	v_mul_u32_u24_e32 v39, 0xe80, v39
	v_mad_i32_i24 v38, v38, s38, v39
	v_add3_u32 v38, v27, v38, s39
	s_mov_b32 m0, s40
	s_nop 0
	buffer_load_dwordx4 v38, s[20:23], 0 offen lds
	v_add_u32_e32 v38, v17, v31
	v_add_u32_e32 v39, 0x100, v38
	v_add3_u32 v40, v28, v30, s42
	v_cmp_gt_i32_e32 vcc, s41, v38
	v_add_u32_e32 v43, v28, v33
	v_subrev_u32_e32 v44, 25, v43
	v_cndmask_b32_e32 v40, v39, v40, vcc
	v_mul_hi_i32 v41, v40, s34
	v_add_u32_e32 v40, v41, v40
	v_lshrrev_b32_e32 v41, 31, v40
	v_ashrrev_i32_e32 v40, 7, v40
	v_add_u32_e32 v40, v40, v41
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_xad_u32 v40, v40, v41, v18
	v_add_u32_e32 v41, v17, v26
	v_add_u32_e32 v42, 24, v41
	v_cmp_gt_i32_e32 vcc, s43, v41
	v_add_u32_e32 v45, v28, v32
	v_add_u32_e32 v46, 0xfffffefd, v45
	v_cndmask_b32_e32 v42, v42, v44, vcc
	v_mul_hi_i32 v44, v42, s34
	v_add_u32_e32 v42, v44, v42
	v_lshrrev_b32_e32 v44, 31, v42
	v_ashrrev_i32_e32 v42, 7, v42
	v_add_u32_e32 v42, v42, v44
	v_cndmask_b32_e64 v44, 0, -1, vcc
	v_xor_b32_e32 v42, v42, v44
	v_add_u32_e32 v44, 0x102, v38
	v_cmp_gt_i32_e32 vcc, -2, v39
	v_mul_lo_u32 v40, v40, s14
	v_mul_lo_u32 v42, v42, s35
	v_cndmask_b32_e32 v46, v44, v46, vcc
	v_mul_hi_i32 v47, v46, s34
	v_add_u32_e32 v46, v47, v46
	v_lshrrev_b32_e32 v47, 31, v46
	v_ashrrev_i32_e32 v46, 7, v46
	v_add_u32_e32 v46, v46, v47
	v_ashrrev_i32_e32 v44, 31, v44
	v_xad_u32 v44, v46, v44, v18
	v_add_u32_e32 v46, 0x102, v41
	v_add_u32_e32 v47, 0xfffffefd, v43
	v_cmp_gt_i32_e32 vcc, s44, v41
	v_sub_u32_e32 v40, v40, v42
	v_add_u32_e32 v42, v34, v17
	v_cndmask_b32_e32 v46, v46, v47, vcc
	v_mul_hi_i32 v47, v46, s34
	v_add_u32_e32 v46, v47, v46
	v_lshrrev_b32_e32 v47, 31, v46
	v_ashrrev_i32_e32 v46, 7, v46
	v_add_u32_e32 v46, v46, v47
	v_cndmask_b32_e64 v47, 0, -1, vcc
	v_xor_b32_e32 v46, v46, v47
	v_mul_lo_u32 v44, v44, s14
	v_mul_lo_u32 v46, v46, s35
	v_add_u32_e32 v40, v42, v40
	v_sub_u32_e32 v44, v44, v46
	v_add3_u32 v44, v17, v44, v34
	buffer_load_ubyte v46, v40, s[4:7], 0 offen offset:24
	buffer_load_ubyte v47, v44, s[4:7], 0 offen offset:258
	ds_read_b128 v[48:51], v20
	ds_read_b128 v[52:55], v21 offset:16384
	ds_read_b128 v[56:59], v21 offset:18432
	s_barrier
	s_setprio 1
	v_and_b32_e32 v29, 0xff, v29
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[48:51], v[52:55], v[0:3], v29, v35 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[48:51], v[56:59], v[4:7], v29, v35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	ds_read_b128 v[48:51], v22
	ds_read_b128 v[52:55], v19 offset:16384
	ds_read_b128 v[56:59], v19 offset:18432
	s_waitcnt vmcnt(3)
	s_barrier
	s_setprio 1
	v_and_b32_e32 v23, 0xff, v23
	s_waitcnt lgkmcnt(1)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[48:51], v[52:55], v[0:3], v23, v35 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[48:51], v[56:59], v[4:7], v23, v35 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_lshl_b32 s0, s47, 6
	s_mov_b32 m0, s30
	v_add_u32_e32 v23, 0x100, v36
	s_waitcnt vmcnt(0)
	s_barrier
	buffer_load_dwordx4 v23, s[24:27], 0 offen lds
	v_add_u32_e32 v23, 0x100, v37
	v_mul_u32_u24_sdwa v23, v23, s37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshrrev_b32_e32 v23, 22, v23
	v_add_u32_e32 v29, v16, v23
	v_mul_i32_i24_e32 v23, 0xfffff180, v23
	v_mul_u32_u24_e32 v29, 0xe80, v29
	v_add3_u32 v23, v23, v29, v27
	s_mov_b32 m0, s31
	s_nop 0
	buffer_load_dwordx4 v23, s[20:23], 0 offen lds
	v_add_u32_e32 v23, 0x200, v38
	v_add_u32_e32 v29, 0xfffffdff, v45
	v_cmp_gt_i32_e32 vcc, s41, v39
	v_add_u32_e32 v36, 0x118, v41
	v_add_u32_e32 v37, 0xfffffee7, v43
	v_cndmask_b32_e32 v29, v23, v29, vcc
	v_mul_hi_i32 v35, v29, s34
	v_add_u32_e32 v29, v35, v29
	v_lshrrev_b32_e32 v35, 31, v29
	v_ashrrev_i32_e32 v29, 7, v29
	v_add_u32_e32 v29, v29, v35
	v_ashrrev_i32_e32 v35, 31, v23
	v_xad_u32 v29, v29, v35, v18
	v_add_u32_e32 v35, 0x100, v41
	v_cmp_gt_i32_e32 vcc, s43, v35
	v_mul_lo_u32 v29, v29, s14
	s_addk_i32 s0, 0xc0
	v_cndmask_b32_e32 v37, v36, v37, vcc
	v_mul_hi_i32 v39, v37, s34
	v_add_u32_e32 v37, v39, v37
	v_lshrrev_b32_e32 v39, 31, v37
	v_ashrrev_i32_e32 v37, 7, v37
	v_add_u32_e32 v37, v37, v39
	v_ashrrev_i32_e32 v36, 31, v36
	v_xor_b32_e32 v36, v37, v36
	v_mul_lo_u32 v36, v36, s35
	v_sub_u32_e32 v29, v29, v36
	v_add_u32_e32 v36, v42, v29
	v_add_u32_e32 v29, 0x202, v38
	v_add_u32_e32 v37, 0xfffffdfd, v45
	v_cmp_gt_i32_e32 vcc, -2, v23
	s_nop 1
	v_cndmask_b32_e32 v23, v29, v37, vcc
	v_mul_hi_i32 v37, v23, s34
	v_add_u32_e32 v23, v37, v23
	v_lshrrev_b32_e32 v37, 31, v23
	v_ashrrev_i32_e32 v23, 7, v23
	v_add_u32_e32 v23, v23, v37
	v_ashrrev_i32_e32 v29, 31, v29
	v_xad_u32 v23, v23, v29, v18
	v_add_u32_e32 v29, 0x202, v41
	v_add_u32_e32 v37, 0xfffffdfd, v43
	v_cmp_gt_i32_e32 vcc, s44, v35
	v_mul_lo_u32 v23, v23, s14
	s_nop 0
	v_cndmask_b32_e32 v35, v29, v37, vcc
	v_mul_hi_i32 v37, v35, s34
	v_add_u32_e32 v35, v37, v35
	v_lshrrev_b32_e32 v37, 31, v35
	v_ashrrev_i32_e32 v35, 7, v35
	v_add_u32_e32 v35, v35, v37
	v_ashrrev_i32_e32 v29, 31, v29
	v_xor_b32_e32 v29, v35, v29
	v_mul_lo_u32 v29, v29, s35
	v_sub_u32_e32 v23, v23, v29
	v_add3_u32 v35, v17, v23, v34
	buffer_load_ubyte v29, v36, s[4:7], 0 offen offset:280
	buffer_load_ubyte v23, v35, s[4:7], 0 offen offset:514
	v_lshl_add_u32 v36, s46, 6, v8
	v_add_u32_e32 v37, 6, v36
	v_sub_u32_e32 v38, 0xfff9, v36
	v_cmp_gt_i32_e32 vcc, -6, v36
	v_add_u32_e32 v35, s0, v9
	v_sub_u32_e32 v40, 0xffbf, v36
	v_cndmask_b32_e32 v37, v37, v38, vcc
	v_mul_i32_i24_sdwa v37, sext(v37), s37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshrrev_b32_e32 v38, 31, v37
	v_ashrrev_i32_e32 v37, 20, v37
	v_add_u16_e32 v37, v37, v38
	v_cndmask_b32_e64 v38, 0, -1, vcc
	v_xor_b32_e32 v37, v37, v38
	v_add_u32_e32 v38, 64, v36
	v_cmp_gt_i32_e64 s[0:1], s33, v36
	v_add_u32_e32 v39, 64, v35
	v_sub_u32_e32 v41, 0xffbf, v35
	v_cndmask_b32_e64 v36, v38, v40, s[0:1]
	v_cmp_gt_i32_e32 vcc, s33, v35
	v_mul_i32_i24_sdwa v36, sext(v36), s37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshrrev_b32_e32 v38, 31, v36
	v_cndmask_b32_e32 v39, v39, v41, vcc
	v_ashrrev_i32_e32 v36, 20, v36
	v_add_u16_e32 v36, v36, v38
	v_mul_i32_i24_sdwa v38, sext(v39), s37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshrrev_b32_e32 v39, 31, v38
	v_ashrrev_i32_e32 v38, 20, v38
	v_add_u16_e32 v38, v38, v39
	v_perm_b32 v36, v38, v36, s45
	v_cndmask_b32_e64 v38, 0, -1, s[0:1]
	v_cndmask_b32_e64 v39, 0, -1, vcc
	v_perm_b32 v38, v39, v38, s45
	v_xor_b32_e32 v36, v36, v38
	v_add_u32_sdwa v38, v10, sext(v36) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_bfe_i32 v37, v37, 0, 16
	v_mul_lo_u32 v38, v38, s12
	v_mad_i32_i24 v37, v37, s15, v38
	v_add_u32_e32 v38, 6, v35
	v_sub_u32_e32 v39, 0xfff9, v35
	v_cmp_gt_i32_e32 vcc, -6, v35
	v_add_u32_sdwa v36, v10, sext(v36) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mul_lo_u32 v36, v36, s13
	v_cndmask_b32_e32 v35, v38, v39, vcc
	v_mul_i32_i24_sdwa v35, sext(v35), s37 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshrrev_b32_e32 v38, 31, v35
	v_ashrrev_i32_e32 v35, 20, v35
	v_add_u16_e32 v35, v35, v38
	v_cndmask_b32_e64 v38, 0, -1, vcc
	v_xor_b32_e32 v35, v35, v38
	v_bfe_i32 v35, v35, 0, 16
	v_add3_u32 v37, v37, v17, v25
	v_mad_i32_i24 v35, v35, s15, v36
	v_add3_u32 v36, v35, v17, v25
	buffer_load_dword v38, v37, s[16:19], 0 offen offset:24
	buffer_load_dword v35, v36, s[16:19], 0 offen offset:280
	ds_read_b128 v[40:43], v20 offset:8192
	ds_read_b128 v[48:51], v21 offset:24576
	ds_read_b128 v[52:55], v21 offset:26624
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(1) lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[40:43], v[48:51], v[0:3], v46, v38 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[40:43], v[52:55], v[4:7], v46, v38 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	ds_read_b128 v[40:43], v22 offset:8192
	ds_read_b128 v[48:51], v19 offset:24576
	ds_read_b128 v[52:55], v19 offset:26624
	s_waitcnt vmcnt(3)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[40:43], v[48:51], v[0:3], v47, v38 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[40:43], v[52:55], v[4:7], v47, v38 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_setprio 0
	v_add_u32_e32 v34, 0x200, v34
	v_add_u32_e32 v25, 0x200, v25
	v_add_u32_e32 v27, 0x1000, v27
	v_add_u32_e32 v15, 0x100, v15
	v_add_u32_e32 v24, 0x100, v24
	v_add_u32_e32 v30, 0xfffffe00, v30
	v_add_u32_e32 v31, 0x200, v31
	v_add_u32_e32 v32, 0xfffffe00, v32
	v_add_u32_e32 v26, 0x200, v26
	v_add_u32_e32 v33, 0xfffffe00, v33
	s_cmp_lt_u32 s46, 26
	s_mov_b32 s47, s46
	s_cbranch_scc1 .LBB0_3
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	s_barrier
	ds_read_b128 v[14:17], v20
	ds_read_b128 v[24:27], v21 offset:16384
	ds_read_b128 v[30:33], v21 offset:18432
	s_waitcnt vmcnt(0) lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[14:17], v[24:27], v[0:3], v29, v35 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v22
	s_movk_i32 s0, 0x7fff
	s_mul_hi_u32 s1, s8, s28
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[14:17], v[30:33], v[4:7], v29, v35 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[14:17], v19 offset:16384
	ds_read_b128 v[18:21], v19 offset:18432
	s_mov_b32 s3, 0x27000
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[24:27], v[14:17], v[0:3], v23, v35 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v16, 0x7fc0
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[24:27], v[18:21], v[4:7], v23, v35 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_nop 4
	v_bfe_u32 v8, v3, 16, 1
	v_bfe_u32 v9, v2, 16, 1
	v_add3_u32 v8, v3, v8, s0
	v_bfe_u32 v14, v1, 16, 1
	v_add3_u32 v9, v2, v9, s0
	v_lshrrev_b32_e32 v8, 16, v8
	v_cmp_o_f32_e32 vcc, v3, v3
	v_bfe_u32 v15, v0, 16, 1
	v_add3_u32 v14, v1, v14, s0
	v_lshrrev_b32_e32 v9, 16, v9
	v_cndmask_b32_e32 v3, v16, v8, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	v_add3_u32 v15, v0, v15, s0
	v_lshrrev_b32_e32 v14, 16, v14
	v_cndmask_b32_e32 v2, v16, v9, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	v_lshrrev_b32_e32 v15, 16, v15
	v_bfe_u32 v8, v7, 16, 1
	v_cndmask_b32_e32 v1, v16, v14, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	v_bfe_u32 v9, v6, 16, 1
	v_bfe_u32 v14, v5, 16, 1
	v_cndmask_b32_e32 v0, v16, v15, vcc
	v_bfe_u32 v15, v4, 16, 1
	v_add3_u32 v15, v4, v15, s0
	v_add3_u32 v14, v5, v14, s0
	v_add3_u32 v9, v6, v9, s0
	v_add3_u32 v8, v7, v8, s0
	s_mul_i32 s0, s9, s28
	s_add_i32 s1, s1, s0
	s_mul_i32 s0, s8, s28
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s0, s10, s0
	s_addc_u32 s1, s11, s1
	s_lshl_b32 s2, s29, 1
	v_lshrrev_b32_e32 v8, 16, v8
	v_cmp_o_f32_e32 vcc, v7, v7
	s_add_u32 s0, s0, s2
	v_lshrrev_b32_e32 v9, 16, v9
	v_cndmask_b32_e32 v7, v16, v8, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	s_addc_u32 s1, s1, 0
	s_and_b32 s2, s8, 0x3fff
	v_cndmask_b32_e32 v6, v16, v9, vcc
	v_lshl_or_b32 v8, v12, 2, v13
	s_lshl_b32 s2, s2, 16
	s_and_b32 s1, s1, 0xffff
	v_lshlrev_b32_e32 v9, 1, v10
	v_mul_lo_u32 v8, s8, v8
	s_or_b32 s1, s2, s1
	v_lshl_add_u32 v9, v11, 1, v9
	s_or_b32 s1, s1, 2.0
	s_mov_b32 s2, 0x7ffffffd
	v_lshl_add_u32 v8, v8, 1, v9
	s_lshl_b32 s4, s8, 1
	buffer_store_short v0, v8, s[0:3], 0 offen
	v_add_u32_e32 v0, s4, v8
	v_lshrrev_b32_e32 v14, 16, v14
	v_cmp_o_f32_e32 vcc, v5, v5
	buffer_store_short v1, v0, s[0:3], 0 offen
	v_add_u32_e32 v1, s4, v0
	v_lshrrev_b32_e32 v15, 16, v15
	v_cndmask_b32_e32 v5, v16, v14, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	buffer_store_short v2, v1, s[0:3], 0 offen
	v_add_u32_e32 v2, s4, v1
	v_cndmask_b32_e32 v4, v16, v15, vcc
	buffer_store_short v3, v2, s[0:3], 0 offen
	buffer_store_short v4, v8, s[0:3], 0 offen offset:32
	buffer_store_short v5, v0, s[0:3], 0 offen offset:32
	buffer_store_short v6, v1, s[0:3], 0 offen offset:32
	buffer_store_short v7, v2, s[0:3], 0 offen offset:32
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_static_gemm_64x64x256_512x1792x7424
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
		.amdhsa_next_free_vgpr 60
		.amdhsa_next_free_sgpr 48
		.amdhsa_accum_offset 60
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
	.size	wave_mxfp4_static_gemm_64x64x256_512x1792x7424, .Lfunc_end0-wave_mxfp4_static_gemm_64x64x256_512x1792x7424

	.set wave_mxfp4_static_gemm_64x64x256_512x1792x7424.num_vgpr, 60
	.set wave_mxfp4_static_gemm_64x64x256_512x1792x7424.num_agpr, 0
	.set wave_mxfp4_static_gemm_64x64x256_512x1792x7424.numbered_sgpr, 48
	.set wave_mxfp4_static_gemm_64x64x256_512x1792x7424.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_64x64x256_512x1792x7424.private_seg_size, 0
	.set wave_mxfp4_static_gemm_64x64x256_512x1792x7424.uses_vcc, 1
	.set wave_mxfp4_static_gemm_64x64x256_512x1792x7424.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_64x64x256_512x1792x7424.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_64x64x256_512x1792x7424.has_recursion, 0
	.set wave_mxfp4_static_gemm_64x64x256_512x1792x7424.has_indirect_call, 0
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
    .name:           wave_mxfp4_static_gemm_64x64x256_512x1792x7424
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     54
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_64x64x256_512x1792x7424.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     60
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
