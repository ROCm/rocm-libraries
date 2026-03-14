; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_static_gemm_64x64x256_256x640x768
	.p2align	8
	.type	wave_mxfp4_static_gemm_64x64x256_256x640x768,@function
wave_mxfp4_static_gemm_64x64x256_256x640x768:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	v_and_b32_e32 v1, 0x3ff, v0
	v_bfe_u32 v12, v0, 10, 10
	v_lshrrev_b32_e32 v13, 6, v1
	v_lshlrev_b32_e32 v8, 5, v12
	v_lshl_or_b32 v0, v13, 3, v8
	v_lshrrev_b32_e32 v3, 3, v1
	s_mov_b64 s[20:21], s[2:3]
	v_readfirstlane_b32 s2, v0
	v_or_b32_e32 v0, v3, v8
	s_lshl_b32 s28, s16, 6
	v_or_b32_e32 v0, s28, v0
	v_bitop3_b32 v4, v3, 7, v1 bitop3:0x48
	v_lshlrev_b32_e32 v5, 4, v4
	v_mul_u32_u24_e32 v0, 0x180, v0
	v_or_b32_e32 v2, v0, v5
	v_and_or_b32 v0, v3, 15, v5
	v_mul_lo_u16_e32 v5, 43, v0
	v_lshlrev_b32_e32 v7, 1, v1
	v_lshrrev_b16_e32 v5, 10, v5
	s_lshl_b32 s29, s17, 6
	v_bitop3_b32 v16, v3, 48, v8 bitop3:0xc8
	v_lshlrev_b32_e32 v3, 4, v3
	v_and_b32_e32 v7, 0x100, v7
	s_mov_b64 s[24:25], s[6:7]
	s_movk_i32 s31, 0x180
	s_and_b32 s3, s21, 0xffff
	s_lshl_b32 s34, s2, 7
	v_or3_b32 v6, s29, v5, v16
	v_sub_u32_e32 v17, v3, v7
	v_lshrrev_b32_e32 v14, 4, v1
	s_or_b32 s21, s3, 0x41800000
	s_mov_b32 s23, 0x27000
	s_mov_b32 s22, 0x7ffffffe
	s_mov_b32 m0, s34
	v_mul_i32_i24_e32 v3, 0xfffffe80, v5
	v_lshlrev_b32_e32 v19, 8, v4
	s_and_b32 s2, s25, 0xffff
	v_mad_u32_u24 v4, v6, s31, v17
	s_add_i32 s35, s34, 0x4000
	v_lshlrev_b32_e32 v11, 4, v13
	v_mad_i32_i24 v9, v14, -16, v1
	buffer_load_dwordx4 v2, s[20:23], 0 offen lds
	s_or_b32 s25, s2, 0x41800000
	s_mov_b32 s26, s22
	s_mov_b32 s27, s23
	v_add3_u32 v3, v4, v3, v19
	s_mov_b32 m0, s35
	v_add_u32_e32 v4, v9, v11
	buffer_load_dwordx4 v3, s[24:27], 0 offen lds
	v_ashrrev_i16_e32 v3, 15, v4
	v_lshrrev_b16_e32 v3, 11, v3
	v_add_u16_e32 v3, v4, v3
	v_and_b32_e32 v3, 0xffffffe0, v3
	v_sub_u16_e32 v3, v4, v3
	v_bfe_i32 v5, v3, 0, 16
	v_ashrrev_i32_e32 v6, 31, v5
	v_add_u16_e32 v7, 32, v3
	v_cmp_gt_i16_e32 vcc, 0, v3
	v_lshlrev_b32_e32 v20, 2, v1
	s_movk_i32 s30, 0xffc0
	v_cndmask_b32_e32 v3, v5, v7, vcc
	v_cndmask_b32_e64 v5, v6, 0, vcc
	v_xor_b32_e32 v3, v5, v3
	v_lshrrev_b32_e32 v6, 28, v3
	v_add_u32_e32 v3, v3, v6
	v_bfe_u32 v10, v1, 4, 2
	v_ashrrev_i32_e32 v3, 4, v3
	v_mad_i32_i24 v15, v14, s30, v20
	v_lshlrev_b32_e32 v22, 6, v10
	v_xor_b32_e32 v3, v3, v5
	v_add3_u32 v3, v15, v22, v3
	v_ashrrev_i32_e32 v5, 31, v3
	v_xor_b32_e32 v6, v5, v3
	s_mov_b32 s33, 0x2aaaaaab
	v_mul_hi_i32 v6, v6, s33
	v_lshrrev_b32_e32 v7, 31, v6
	v_ashrrev_i32_e32 v6, 2, v6
	v_add_u32_e32 v6, v6, v7
	v_ashrrev_i32_e32 v7, 31, v4
	v_xor_b32_e32 v4, v7, v4
	v_ashrrev_i32_e32 v18, 31, v4
	v_lshrrev_b32_e32 v18, 27, v18
	v_add_u32_e32 v4, v4, v18
	v_lshrrev_b32_e32 v4, 5, v4
	v_xor_b32_e32 v4, v4, v7
	v_lshlrev_b32_e32 v4, 5, v4
	v_bfe_u32 v24, v1, 6, 1
	v_xad_u32 v7, v6, v5, v4
	v_or_b32_e32 v5, v15, v24
	v_add_u32_e32 v5, v5, v22
	v_mul_hi_i32 v6, v5, s33
	v_lshrrev_b32_e32 v18, 31, v6
	v_lshrrev_b32_e32 v6, 2, v6
	s_mul_i32 s15, s15, s28
	s_mul_hi_u32 s2, s14, s28
	v_add_u32_e32 v6, v6, v18
	s_add_i32 s2, s2, s15
	s_mul_i32 s3, s14, s28
	v_mul_lo_u32 v6, v6, 24
	s_add_u32 s4, s4, s3
	v_sub_u32_e32 v6, v5, v6
	s_addc_u32 s2, s5, s2
	s_and_b32 s3, s14, 0x3fff
	v_add_u32_e32 v18, 24, v6
	v_cmp_gt_i32_e32 vcc, 0, v6
	s_bitset1_b32 s3, 14
	s_and_b32 s2, s2, 0xffff
	v_cndmask_b32_e32 v6, v6, v18, vcc
	s_lshl_b32 s3, s3, 16
	s_or_b32 s5, s2, s3
	v_mad_u64_u32 v[6:7], s[2:3], v7, s14, v[6:7]
	v_add_u32_e32 v7, 2, v3
	v_sub_u32_e32 v18, -3, v3
	v_cmp_gt_i32_e32 vcc, -2, v3
	v_xor_b32_e32 v21, -3, v5
	s_movk_i32 s15, 0xffe8
	v_cndmask_b32_e32 v7, v7, v18, vcc
	v_mul_hi_i32 v7, v7, s33
	v_lshrrev_b32_e32 v18, 31, v7
	v_ashrrev_i32_e32 v7, 2, v7
	v_add_u32_e32 v7, v7, v18
	v_cndmask_b32_e64 v18, 0, -1, vcc
	v_xad_u32 v7, v7, v18, v4
	v_or_b32_e32 v18, 2, v5
	v_cmp_gt_i32_e32 vcc, 0, v5
	v_mul_lo_u32 v7, v7, s14
	s_mov_b32 s6, s22
	v_cndmask_b32_e32 v18, v18, v21, vcc
	v_mul_hi_i32 v18, v18, s33
	v_lshrrev_b32_e32 v21, 31, v18
	v_lshrrev_b32_e32 v18, 2, v18
	v_add_u32_e32 v18, v18, v21
	v_ashrrev_i32_e32 v21, 31, v5
	v_xor_b32_e32 v18, v18, v21
	v_mul_lo_u32 v18, v18, s15
	s_mov_b32 s7, s23
	v_add3_u32 v7, v7, v18, v22
	s_load_dwordx2 s[12:13], s[0:1], 0x40
	v_add3_u32 v7, v7, v24, v15
	buffer_load_ubyte v21, v6, s[4:7], 0 offen
	buffer_load_ubyte v18, v7, s[4:7], 0 offen offset:2
	v_lshlrev_b32_e32 v6, 4, v10
	v_add_u32_e32 v7, v9, v6
	v_ashrrev_i32_e32 v23, 31, v7
	v_xor_b32_e32 v7, v23, v7
	v_mul_hi_i32 v7, v7, s33
	v_lshrrev_b32_e32 v25, 31, v7
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s13, s29
	s_mul_hi_u32 s3, s12, s29
	v_add_u32_e32 v7, v7, v25
	s_add_i32 s3, s3, s2
	s_mul_i32 s2, s12, s29
	v_xad_u32 v7, v7, v23, v8
	v_and_b32_e32 v23, 62, v1
	s_add_u32 s16, s8, s2
	v_mul_lo_u16_e32 v23, 43, v23
	s_addc_u32 s2, s9, s3
	s_and_b32 s3, s12, 0x3fff
	v_lshrrev_b16_e32 v23, 8, v23
	v_mul_lo_u32 v7, s12, v7
	s_bitset1_b32 s3, 14
	s_and_b32 s2, s2, 0xffff
	s_lshl_b32 s3, s3, 16
	v_mad_i32_i24 v7, v23, s15, v7
	s_or_b32 s17, s2, s3
	s_mov_b32 s18, s22
	s_mov_b32 s19, s23
	v_add3_u32 v7, v7, v22, v15
	buffer_load_dword v23, v7, s[16:19], 0 offen
	v_cmp_eq_u32_e64 s[2:3], 0, v12
	s_movk_i32 s13, 0xfe80
	v_mul_i32_i24_e32 v25, 0xffffffc0, v14
	v_mul_i32_i24_e32 v7, -16, v14
	s_and_b64 vcc, exec, s[2:3]
	s_barrier
	s_waitcnt vmcnt(0)
	s_cbranch_vccnz .LBB0_2
	s_barrier
.LBB0_2:
	s_load_dwordx2 s[8:9], s[0:1], 0x48
	v_and_b32_e32 v26, 7, v1
	v_lshlrev_b32_e32 v27, 7, v1
	v_lshlrev_b32_e32 v14, 11, v14
	v_bitop3_b32 v15, v10, v1, 7 bitop3:0x78
	v_sub_u32_e32 v27, v27, v14
	v_bitop3_b32 v26, v10, v26, 4 bitop3:0x36
	v_lshl_add_u32 v13, v13, 11, v27
	v_lshlrev_b32_e32 v15, 4, v15
	v_lshl_add_u32 v12, v12, 12, v27
	v_lshlrev_b32_e32 v26, 4, v26
	v_or_b32_e32 v14, v13, v15
	v_or_b32_e32 v15, v12, v15
	v_or_b32_e32 v13, v26, v13
	v_or_b32_e32 v12, v26, v12
	v_or_b32_e32 v16, s29, v16
	v_add_u32_e32 v17, v17, v19
	v_add3_u32 v19, v22, v25, v20
	s_movk_i32 s0, 0x80
	v_add_u32_e32 v26, 0x80, v2
	v_add_u32_e32 v32, v19, v24
	v_or_b32_e32 v20, 0x80, v0
	s_movk_i32 s1, 0xab
	v_mul_lo_u16_e32 v20, 0xab, v20
	v_mov_b32_e32 v22, 0x558
	v_lshrrev_b16_e32 v20, 12, v20
	v_mad_legacy_u16 v22, v0, s1, v22
	v_or_b32_e32 v20, v16, v20
	v_lshrrev_b16_e32 v22, 12, v22
	s_add_i32 m0, s34, 0x2000
	v_mul_i32_i24_e32 v22, 0xfffffe80, v22
	v_mad_u32_u24 v20, v20, s31, v17
	s_barrier
	buffer_load_dwordx4 v26, s[20:23], 0 offen lds
	v_add3_u32 v20, v20, v22, s0
	s_add_i32 m0, s34, 0x6000
	s_mov_b32 s26, s22
	s_mov_b32 s27, s23
	buffer_load_dwordx4 v20, s[24:27], 0 offen lds
	s_movk_i32 s0, 0xff00
	v_add_u32_e32 v33, 0x100, v3
	v_sub_u32_e32 v20, 0xfffffeff, v3
	v_cmp_gt_i32_e32 vcc, s0, v3
	v_sub_u32_e32 v24, 0xffffffef, v5
	s_movk_i32 s1, 0xfefe
	v_cndmask_b32_e32 v20, v33, v20, vcc
	v_mul_hi_i32 v20, v20, s33
	v_lshrrev_b32_e32 v22, 31, v20
	v_ashrrev_i32_e32 v20, 2, v20
	v_add_u32_e32 v20, v20, v22
	v_cndmask_b32_e64 v22, 0, -1, vcc
	v_xad_u32 v20, v20, v22, v4
	v_add_u32_e32 v22, 16, v5
	v_cmp_gt_i32_e32 vcc, -16, v5
	v_mul_lo_u32 v20, v20, s14
	v_sub_u32_e32 v25, 0xfffffefd, v5
	v_cndmask_b32_e32 v22, v22, v24, vcc
	v_mul_hi_i32 v22, v22, s33
	v_lshrrev_b32_e32 v24, 31, v22
	v_lshrrev_b32_e32 v22, 2, v22
	v_add_u32_e32 v22, v22, v24
	v_cndmask_b32_e64 v24, 0, -1, vcc
	v_xor_b32_e32 v22, v22, v24
	v_mul_lo_u32 v22, v22, s15
	v_add3_u32 v20, v32, v20, v22
	v_add_u32_e32 v22, 0x102, v3
	v_sub_u32_e32 v24, 0xfffffefd, v3
	v_cmp_gt_i32_e32 vcc, -2, v33
	s_mov_b32 s6, s22
	s_mov_b32 s7, s23
	v_cndmask_b32_e32 v22, v22, v24, vcc
	v_mul_hi_i32 v22, v22, s33
	v_lshrrev_b32_e32 v24, 31, v22
	v_ashrrev_i32_e32 v22, 2, v22
	v_add_u32_e32 v22, v22, v24
	v_cndmask_b32_e64 v24, 0, -1, vcc
	v_xad_u32 v22, v22, v24, v4
	v_add_u32_e32 v24, 0x102, v5
	v_cmp_gt_i32_e32 vcc, s1, v5
	v_mul_lo_u32 v22, v22, s14
	s_nop 0
	v_cndmask_b32_e32 v24, v24, v25, vcc
	v_mul_hi_i32 v24, v24, s33
	v_lshrrev_b32_e32 v25, 31, v24
	v_lshrrev_b32_e32 v24, 2, v24
	v_add_u32_e32 v24, v24, v25
	v_cndmask_b32_e64 v25, 0, -1, vcc
	v_xor_b32_e32 v24, v24, v25
	v_mul_lo_u32 v24, v24, s15
	v_add3_u32 v22, v32, v22, v24
	buffer_load_ubyte v34, v20, s[4:7], 0 offen offset:16
	buffer_load_ubyte v35, v22, s[4:7], 0 offen offset:258
	ds_read_b128 v[28:31], v14
	ds_read_b128 v[24:27], v15 offset:16384
	ds_read_b128 v[36:39], v15 offset:18432
	s_barrier
	s_setprio 1
	v_and_b32_e32 v20, 0xffff, v21
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[28:31], v[24:27], 0, v20, v23 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[28:31], v[28:31], v[36:39], 0, v20, v23 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	ds_read_b128 v[36:39], v13
	ds_read_b128 v[40:43], v12 offset:16384
	ds_read_b128 v[44:47], v12 offset:18432
	s_waitcnt vmcnt(3)
	s_barrier
	s_setprio 1
	v_and_b32_e32 v18, 0xffff, v18
	s_waitcnt lgkmcnt(1)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[24:27], v[36:39], v[40:43], v[24:27], v18, v23 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[36:39], v[44:47], v[28:31], v18, v23 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_mov_b32 m0, s34
	v_add_u32_e32 v2, 0x100, v2
	s_barrier
	buffer_load_dwordx4 v2, s[20:23], 0 offen lds
	v_or_b32_e32 v2, 0x100, v0
	s_movk_i32 s18, 0xaab
	v_mul_u32_u24_sdwa v2, v2, s18 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_add_u16_e32 v0, 0x88, v0
	v_or_b32_sdwa v2, v16, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mul_u32_u24_e32 v0, 0xaab, v0
	v_mul_i32_i24_sdwa v0, v0, s13 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_mad_u32_u24 v2, v2, s31, v17
	s_movk_i32 s13, 0x880
	v_add3_u32 v0, v2, v0, s13
	s_mov_b32 m0, s35
	s_nop 0
	buffer_load_dwordx4 v0, s[24:27], 0 offen lds
	v_add_u32_e32 v0, 0x200, v3
	v_sub_u32_e32 v2, 0xfffffdff, v3
	v_cmp_gt_i32_e32 vcc, s0, v33
	v_add_u32_e32 v17, 0x110, v5
	v_sub_u32_e32 v18, 0xfffffeef, v5
	v_cndmask_b32_e32 v2, v0, v2, vcc
	v_mul_hi_i32 v2, v2, s33
	v_lshrrev_b32_e32 v16, 31, v2
	v_ashrrev_i32_e32 v2, 2, v2
	v_add_u32_e32 v2, v2, v16
	v_cndmask_b32_e64 v16, 0, -1, vcc
	v_xad_u32 v2, v2, v16, v4
	v_add_u32_e32 v16, 0x100, v5
	v_cmp_gt_i32_e32 vcc, -16, v16
	v_mul_lo_u32 v2, v2, s14
	s_mov_b32 s18, s22
	v_cndmask_b32_e32 v17, v17, v18, vcc
	v_mul_hi_i32 v17, v17, s33
	v_lshrrev_b32_e32 v18, 31, v17
	v_lshrrev_b32_e32 v17, 2, v17
	v_add_u32_e32 v17, v17, v18
	v_cndmask_b32_e64 v18, 0, -1, vcc
	v_xor_b32_e32 v17, v17, v18
	v_mul_lo_u32 v17, v17, s15
	v_add3_u32 v2, v32, v2, v17
	v_add_u32_e32 v17, 0x202, v3
	v_sub_u32_e32 v3, 0xfffffdfd, v3
	v_cmp_gt_i32_e32 vcc, -2, v0
	s_mov_b32 s19, s23
	s_nop 0
	v_cndmask_b32_e32 v0, v17, v3, vcc
	v_mul_hi_i32 v0, v0, s33
	v_lshrrev_b32_e32 v3, 31, v0
	v_ashrrev_i32_e32 v0, 2, v0
	v_add_u32_e32 v0, v0, v3
	v_cndmask_b32_e64 v3, 0, -1, vcc
	v_xad_u32 v0, v0, v3, v4
	v_add_u32_e32 v3, 0x202, v5
	v_sub_u32_e32 v4, 0xfffffdfd, v5
	v_cmp_gt_i32_e32 vcc, s1, v16
	v_mul_lo_u32 v0, v0, s14
	s_nop 0
	v_cndmask_b32_e32 v3, v3, v4, vcc
	v_mul_hi_i32 v3, v3, s33
	v_lshrrev_b32_e32 v4, 31, v3
	v_lshrrev_b32_e32 v3, 2, v3
	v_add_u32_e32 v3, v3, v4
	v_cndmask_b32_e64 v4, 0, -1, vcc
	v_xor_b32_e32 v3, v3, v4
	v_mul_lo_u32 v3, v3, s15
	v_add3_u32 v0, v32, v0, v3
	buffer_load_ubyte v17, v2, s[4:7], 0 offen offset:272
	buffer_load_ubyte v16, v0, s[4:7], 0 offen offset:514
	v_add3_u32 v0, v6, v7, v1
	v_add_u32_e32 v1, 4, v0
	v_sub_u32_e32 v2, 0xfffb, v0
	v_cmp_gt_i32_e32 vcc, -4, v0
	s_movk_i32 s4, 0x2aab
	v_add_u32_e32 v3, 64, v0
	v_cndmask_b32_e32 v1, v1, v2, vcc
	v_mul_i32_i24_sdwa v1, sext(v1), s4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshrrev_b32_e32 v2, 31, v1
	v_add_u16_sdwa v1, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e64 v2, 0, -1, vcc
	v_sub_u32_e32 v4, 0xffbf, v0
	v_cmp_gt_i32_e64 s[0:1], s30, v0
	v_xor_b32_e32 v1, v1, v2
	v_add_u32_e32 v2, 0x80, v0
	v_sub_u32_e32 v5, 0xff7f, v0
	v_cmp_gt_i32_e32 vcc, s30, v3
	v_cndmask_b32_e64 v4, v3, v4, s[0:1]
	v_mul_i32_i24_sdwa v4, sext(v4), s4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_cndmask_b32_e32 v2, v2, v5, vcc
	v_lshrrev_b32_e32 v5, 31, v4
	v_mul_i32_i24_sdwa v2, sext(v2), s4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_add_u16_sdwa v4, v4, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_lshrrev_b32_e32 v5, 31, v2
	v_add_u16_sdwa v2, v2, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	s_mov_b32 s5, 0x5040100
	v_perm_b32 v2, v2, v4, s5
	v_cndmask_b32_e64 v4, 0, -1, s[0:1]
	v_cndmask_b32_e64 v5, 0, -1, vcc
	v_perm_b32 v4, v5, v4, s5
	v_xor_b32_e32 v2, v2, v4
	v_add_u32_sdwa v4, v8, sext(v2) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_mul_i32_i24_sdwa v1, sext(v1), s15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_mul_lo_u32 v4, v4, s12
	v_add3_u32 v1, v19, v4, v1
	v_add_u32_e32 v4, 0x44, v0
	v_sub_u32_e32 v0, 0xffbb, v0
	v_cmp_gt_i32_e32 vcc, -4, v3
	v_add_u32_sdwa v2, v8, sext(v2) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_1
	v_mul_lo_u32 v2, v2, s12
	v_cndmask_b32_e32 v0, v4, v0, vcc
	v_mul_i32_i24_sdwa v0, sext(v0), s4 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_lshrrev_b32_e32 v3, 31, v0
	v_add_u16_sdwa v0, v0, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
	v_cndmask_b32_e64 v3, 0, -1, vcc
	v_xor_b32_e32 v0, v0, v3
	v_mul_i32_i24_sdwa v0, sext(v0), s15 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:DWORD
	v_add3_u32 v0, v19, v2, v0
	buffer_load_dword v19, v1, s[16:19], 0 offen offset:16
	buffer_load_dword v18, v0, s[16:19], 0 offen offset:272
	ds_read_b128 v[4:7], v14 offset:8192
	s_waitcnt vmcnt(8)
	ds_read_b128 v[0:3], v15 offset:24576
	ds_read_b128 v[28:31], v15 offset:26624
	s_barrier
	s_setprio 1
	s_waitcnt vmcnt(1) lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[4:7], v[0:3], v[24:27], v34, v19 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[20:23], v[4:7], v[28:31], v[20:23], v34, v19 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	ds_read_b128 v[24:27], v13 offset:8192
	ds_read_b128 v[4:7], v12 offset:24576
	ds_read_b128 v[28:31], v12 offset:26624
	s_waitcnt vmcnt(3)
	s_barrier
	s_setprio 1
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[24:27], v[4:7], v[0:3], v35, v19 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[24:27], v[28:31], v[20:23], v35, v19 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_setprio 0
	s_andn2_b64 vcc, exec, s[2:3]
	s_cbranch_vccnz .LBB0_4
	s_barrier
.LBB0_4:
	s_barrier
	ds_read_b128 v[20:23], v14
	ds_read_b128 v[24:27], v15 offset:16384
	ds_read_b128 v[28:31], v15 offset:18432
	v_and_b32_e32 v14, 0xffff, v17
	s_movk_i32 s0, 0x7fff
	s_mul_hi_u32 s1, s8, s28
	s_waitcnt vmcnt(0) lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[20:23], v[24:27], v[4:7], v14, v18 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[24:27], v13
	v_lshl_or_b32 v10, v10, 2, v11
	v_lshlrev_b32_e32 v8, 1, v8
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[20:23], v[28:31], v[0:3], v14, v18 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[20:23], v12 offset:16384
	ds_read_b128 v[28:31], v12 offset:18432
	v_and_b32_e32 v12, 0xffff, v16
	v_mov_b32_e32 v16, 0x7fc0
	v_mul_lo_u32 v10, s8, v10
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[24:27], v[20:23], v[4:7], v12, v18 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_lshl_add_u32 v8, v9, 1, v8
	s_mov_b32 s3, 0x27000
	v_lshl_add_u32 v8, v10, 1, v8
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[24:27], v[28:31], v[0:3], v12, v18 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_nop 2
	v_bfe_u32 v12, v7, 16, 1
	v_bfe_u32 v13, v6, 16, 1
	v_add3_u32 v12, v7, v12, s0
	v_bfe_u32 v14, v5, 16, 1
	v_add3_u32 v13, v6, v13, s0
	v_lshrrev_b32_e32 v12, 16, v12
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v15, v4, 16, 1
	v_add3_u32 v14, v5, v14, s0
	v_lshrrev_b32_e32 v13, 16, v13
	v_cndmask_b32_e32 v7, v16, v12, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v15, v4, v15, s0
	v_lshrrev_b32_e32 v14, 16, v14
	v_cndmask_b32_e32 v6, v16, v13, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_lshrrev_b32_e32 v15, 16, v15
	v_bfe_u32 v12, v3, 16, 1
	v_cndmask_b32_e32 v5, v16, v14, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v13, v2, 16, 1
	v_bfe_u32 v14, v1, 16, 1
	v_cndmask_b32_e32 v4, v16, v15, vcc
	v_bfe_u32 v15, v0, 16, 1
	v_add3_u32 v15, v0, v15, s0
	v_add3_u32 v14, v1, v14, s0
	v_add3_u32 v13, v2, v13, s0
	v_add3_u32 v12, v3, v12, s0
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
	v_lshrrev_b32_e32 v12, 16, v12
	v_cmp_o_f32_e32 vcc, v3, v3
	s_or_b32 s1, s1, 2.0
	s_mov_b32 s2, 0x7ffffffd
	s_lshl_b32 s4, s8, 1
	v_lshrrev_b32_e32 v13, 16, v13
	v_cndmask_b32_e32 v3, v16, v12, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	buffer_store_short v4, v8, s[0:3], 0 offen
	v_add_u32_e32 v4, s4, v8
	v_lshrrev_b32_e32 v14, 16, v14
	v_cndmask_b32_e32 v2, v16, v13, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	buffer_store_short v5, v4, s[0:3], 0 offen
	v_add_u32_e32 v5, s4, v4
	v_lshrrev_b32_e32 v15, 16, v15
	v_cndmask_b32_e32 v1, v16, v14, vcc
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
	.amdhsa_kernel wave_mxfp4_static_gemm_64x64x256_256x640x768
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
		.amdhsa_next_free_sgpr 36
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
	.size	wave_mxfp4_static_gemm_64x64x256_256x640x768, .Lfunc_end0-wave_mxfp4_static_gemm_64x64x256_256x640x768

	.set wave_mxfp4_static_gemm_64x64x256_256x640x768.num_vgpr, 48
	.set wave_mxfp4_static_gemm_64x64x256_256x640x768.num_agpr, 0
	.set wave_mxfp4_static_gemm_64x64x256_256x640x768.numbered_sgpr, 36
	.set wave_mxfp4_static_gemm_64x64x256_256x640x768.num_named_barrier, 0
	.set wave_mxfp4_static_gemm_64x64x256_256x640x768.private_seg_size, 0
	.set wave_mxfp4_static_gemm_64x64x256_256x640x768.uses_vcc, 1
	.set wave_mxfp4_static_gemm_64x64x256_256x640x768.uses_flat_scratch, 0
	.set wave_mxfp4_static_gemm_64x64x256_256x640x768.has_dyn_sized_stack, 0
	.set wave_mxfp4_static_gemm_64x64x256_256x640x768.has_recursion, 0
	.set wave_mxfp4_static_gemm_64x64x256_256x640x768.has_indirect_call, 0
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
    .name:           wave_mxfp4_static_gemm_64x64x256_256x640x768
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     42
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_static_gemm_64x64x256_256x640x768.kd
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
