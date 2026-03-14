; To reproduce the .rocmasm from .optimized.ll, run:
; llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -mattr='-fma-mix-insts' -O3 <.optimized.ll> -o <out.rocmasm>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.text
	.globl	wave_mxfp4_dynamic_gemm_64x64x256
	.p2align	8
	.type	wave_mxfp4_dynamic_gemm_64x64x256,@function
wave_mxfp4_dynamic_gemm_64x64x256:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x8
	s_load_dwordx4 s[12:15], s[0:1], 0x28
	s_waitcnt lgkmcnt(0)
	s_branch .LBB0_0
	.p2align	8
.LBB0_0:
	s_load_dwordx2 s[34:35], s[0:1], 0x38
	s_load_dwordx2 s[52:53], s[0:1], 0x48
	s_load_dwordx2 s[54:55], s[0:1], 0x58
	s_mov_b64 s[40:41], s[2:3]
	s_mov_b64 s[36:37], s[6:7]
	s_waitcnt lgkmcnt(0)
	s_sub_u32 s20, 0, s34
	s_subb_u32 s21, 0, s35
	s_add_u32 s22, s34, -1
	v_cmp_lt_i64_e64 s[2:3], s[34:35], 1
	s_addc_u32 s23, s35, -1
	s_and_b64 s[6:7], s[2:3], exec
	s_cselect_b32 s7, s21, s23
	s_cselect_b32 s6, s20, s22
	s_ashr_i32 s20, s7, 31
	s_lshr_b32 s20, s20, 24
	s_add_u32 s6, s6, s20
	s_addc_u32 s7, s7, 0
	s_ashr_i64 s[6:7], s[6:7], 8
	s_sub_u32 s20, 0, s6
	s_subb_u32 s21, 0, s7
	s_add_u32 s6, s6, 1
	v_and_b32_e32 v35, 0x3ff, v0
	v_bfe_u32 v15, v0, 10, 10
	s_addc_u32 s7, s7, 0
	s_and_b64 s[2:3], s[2:3], exec
	v_lshrrev_b32_e32 v40, 6, v35
	v_lshlrev_b32_e32 v0, 5, v15
	s_mov_b32 s28, s17
	s_mov_b32 s17, 0
	s_cselect_b32 s60, s20, s6
	v_readfirstlane_b32 s33, v0
	v_lshl_or_b32 v0, v40, 3, v0
	v_bfe_u32 v33, v35, 4, 2
	s_mov_b32 s29, s17
	s_mov_b64 s[18:19], -1
	s_cselect_b32 s61, s21, s7
	s_cmp_gt_i32 s60, 1
	v_readfirstlane_b32 s82, v0
	v_lshrrev_b32_e32 v39, 3, v35
	v_lshrrev_b32_e32 v37, 4, v35
	v_lshlrev_b32_e32 v34, 4, v40
	v_and_b32_e32 v32, 15, v35
	v_lshlrev_b32_e32 v1, 2, v35
	v_bfe_u32 v38, v35, 6, 1
	v_lshlrev_b32_e32 v36, 6, v33
	s_cbranch_scc1 .LBB0_2
	s_ashr_i32 s2, s61, 31
	s_mov_b32 s3, s2
	s_xor_b64 s[6:7], s[2:3], s[60:61]
	s_lshr_b32 s18, s7, 31
	s_add_u32 s6, s6, s18
	s_addc_u32 s7, s7, 0
	s_lshr_b64 s[6:7], s[6:7], 1
	s_xor_b64 s[2:3], s[6:7], s[2:3]
	s_lshl_b64 s[66:67], s[2:3], 1
	s_ashr_i32 s2, s35, 31
	s_mov_b32 s3, s2
	s_xor_b64 s[6:7], s[34:35], s[2:3]
	s_lshr_b32 s18, s7, 31
	s_add_u32 s6, s6, s18
	s_addc_u32 s7, s7, 0
	s_lshl_b64 s[30:31], s[16:17], 6
	s_lshr_b32 s18, s16, 26
	v_lshlrev_b32_e32 v8, 4, v40
	s_mul_i32 s18, s52, s18
	s_mul_hi_u32 s19, s52, s30
	v_or_b32_e32 v0, v32, v8
	s_add_i32 s18, s19, s18
	s_mul_i32 s19, s53, s30
	s_ashr_i64 s[6:7], s[6:7], 1
	v_or_b32_e32 v18, s30, v0
	v_lshlrev_b32_e32 v0, 6, v37
	s_add_i32 s63, s18, s19
	s_and_b32 s18, s52, 0x3fff
	s_xor_b64 s[56:57], s[6:7], s[2:3]
	s_lshl_b64 s[58:59], s[28:29], 6
	v_sub_u32_e32 v44, v1, v0
	v_bfe_u32 v9, v35, 6, 1
	s_or_b32 s83, s18, 0x4000
	s_lshr_b32 s18, s28, 26
	s_and_b32 s2, s56, 0x3fff
	v_or_b32_e32 v0, v44, v9
	s_mul_i32 s18, s54, s18
	s_mul_hi_u32 s19, s54, s58
	s_bitset1_b32 s2, 14
	v_add_u32_e32 v45, v0, v36
	v_or_b32_e32 v0, s33, v32
	s_add_i32 s18, s19, s18
	s_mul_i32 s19, s55, s58
	v_or_b32_e32 v41, s33, v39
	s_lshl_b32 s6, s2, 16
	s_and_b32 s2, s41, 0xffff
	s_mov_b32 s43, 0x27000
	s_mov_b32 s42, 0x7ffffffe
	s_and_b32 s7, s37, 0xffff
	v_or_b32_e32 v20, s58, v0
	v_add_u32_e32 v46, v44, v36
	s_add_i32 s65, s18, s19
	s_and_b32 s18, s54, 0x3fff
	v_or_b32_e32 v12, s30, v41
	v_mov_b32_e32 v13, s31
	s_or_b32 s3, s2, s6
	s_mov_b32 s2, s40
	s_or_b32 s7, s7, s6
	s_mov_b32 s6, s36
	v_or_b32_e32 v16, s58, v41
	v_mov_b32_e32 v17, s59
	v_mov_b32_e32 v19, s31
	s_mul_i32 s62, s52, s30
	v_mov_b32_e32 v21, s59
	s_mul_i32 s64, s54, s58
	s_or_b32 s84, s18, 0x4000
	v_or_b32_e32 v24, 2, v46
	v_or_b32_e32 v22, 16, v20
	v_mov_b32_e32 v23, s59
	v_and_b32_e32 v14, 0xfd, v45
	s_mov_b64 s[38:39], s[42:43]
	s_mov_b64 s[18:19], 0
	s_branch .LBB0_3
.LBB0_2:
.LBB0_3:
	s_mov_b32 s20, 0
	s_andn2_b64 vcc, exec, s[18:19]
	v_lshlrev_b32_e32 v43, 1, v35
	v_lshrrev_b32_e32 v42, 2, v35
	s_cbranch_vccnz .LBB0_12
	s_ashr_i32 s2, s35, 31
	s_mov_b32 s3, s2
	s_xor_b64 s[6:7], s[34:35], s[2:3]
	s_lshr_b32 s18, s7, 31
	s_add_u32 s6, s6, s18
	s_addc_u32 s7, s7, 0
	s_ashr_i64 s[6:7], s[6:7], 1
	s_xor_b64 s[56:57], s[6:7], s[2:3]
	s_and_b32 s2, s56, 0x3fff
	v_or_b32_e32 v41, s33, v39
	s_lshl_b64 s[30:31], s[16:17], 6
	v_bitop3_b32 v6, v39, 7, v35 bitop3:0x48
	s_bitset1_b32 s2, 14
	v_or_b32_e32 v12, s30, v41
	v_mov_b32_e32 v13, s31
	s_lshl_b32 s17, s2, 16
	s_and_b32 s2, s41, 0xffff
	v_lshlrev_b32_e32 v4, 5, v6
	v_mov_b32_e32 v5, 0
	v_lshlrev_b32_e32 v0, 4, v6
	s_or_b32 s41, s2, s17
	v_cmp_gt_i64_e64 s[2:3], s[12:13], v[12:13]
	v_cmp_gt_i64_e64 s[18:19], s[34:35], v[4:5]
	v_mad_u64_u32 v[2:3], s[6:7], s56, v12, v[0:1]
	s_and_b64 vcc, s[2:3], s[18:19]
	s_lshl_b32 s85, s82, 7
	s_ashr_i32 s68, s57, 31
	v_bfrev_b32_e32 v9, -2
	s_add_u32 s6, s56, s68
	s_mov_b32 s43, 0x27000
	s_mov_b32 s42, 0x7ffffffe
	v_cndmask_b32_e32 v2, v9, v2, vcc
	s_mov_b32 m0, s85
	v_lshlrev_b32_e32 v8, 4, v39
	v_and_b32_e32 v3, 0x100, v43
	s_mov_b32 s69, s68
	s_addc_u32 s7, s57, s68
	buffer_load_dwordx4 v2, s[40:43], 0 offen lds
	v_lshlrev_b32_e32 v2, 8, v6
	v_sub_u32_e32 v6, v8, v3
	s_xor_b64 s[22:23], s[6:7], s[68:69]
	v_add_u32_e32 v10, v6, v2
	v_cvt_f32_u32_e32 v6, s22
	v_cvt_f32_u32_e32 v7, s23
	v_not_b32_e32 v14, v10
	v_cmp_gt_i32_e32 vcc, 0, v10
	v_ashrrev_i32_e32 v11, 31, v10
	v_fmamk_f32 v6, v7, 0x4f800000, v6
	v_rcp_f32_e32 v17, v6
	v_ashrrev_i32_e32 v16, 31, v14
	v_cndmask_b32_e32 v6, v10, v14, vcc
	v_cndmask_b32_e32 v7, v11, v16, vcc
	v_mul_f32_e32 v14, 0x5f7ffffc, v17
	v_mul_f32_e32 v16, 0x2f800000, v14
	v_trunc_f32_e32 v16, v16
	v_fmamk_f32 v14, v16, 0xcf800000, v14
	v_cvt_u32_f32_e32 v14, v14
	v_cvt_u32_f32_e32 v16, v16
	s_sub_u32 s6, 0, s22
	s_subb_u32 s7, 0, s23
	v_readfirstlane_b32 s21, v14
	v_readfirstlane_b32 s26, v16
	s_mul_hi_u32 s25, s6, s21
	s_mul_i32 s27, s6, s26
	s_mul_i32 s24, s7, s21
	s_add_i32 s25, s25, s27
	s_add_i32 s25, s25, s24
	s_mul_i32 s38, s6, s21
	s_mul_i32 s27, s21, s25
	s_mul_hi_u32 s39, s21, s38
	s_mul_hi_u32 s24, s21, s25
	s_add_u32 s27, s39, s27
	s_addc_u32 s24, 0, s24
	s_mul_hi_u32 s44, s26, s38
	s_mul_i32 s38, s26, s38
	s_add_u32 s27, s27, s38
	s_mul_hi_u32 s39, s26, s25
	s_addc_u32 s24, s24, s44
	s_addc_u32 s27, s39, 0
	s_mul_i32 s25, s26, s25
	s_add_u32 s24, s24, s25
	s_addc_u32 s25, 0, s27
	s_add_u32 s21, s21, s24
	s_addc_u32 s24, s26, s25
	s_mul_i32 s25, s6, s24
	s_mul_hi_u32 s26, s6, s21
	s_add_i32 s25, s26, s25
	s_mul_i32 s7, s7, s21
	s_add_i32 s25, s25, s7
	s_mul_i32 s6, s6, s21
	s_mul_hi_u32 s26, s24, s6
	s_mul_i32 s27, s24, s6
	s_mul_i32 s39, s21, s25
	s_mul_hi_u32 s6, s21, s6
	s_mul_hi_u32 s38, s21, s25
	s_add_u32 s6, s6, s39
	s_addc_u32 s38, 0, s38
	s_add_u32 s6, s6, s27
	s_mul_hi_u32 s7, s24, s25
	s_addc_u32 s6, s38, s26
	s_addc_u32 s7, s7, 0
	s_mul_i32 s25, s24, s25
	s_add_u32 s6, s6, s25
	s_addc_u32 s7, 0, s7
	v_mov_b32_e32 v16, v7
	v_mov_b32_e32 v17, v7
	s_add_u32 s21, s21, s6
	v_lshl_add_u64 v[16:17], v[6:7], 0, v[16:17]
	s_addc_u32 s24, s24, s7
	v_xor_b32_e32 v14, v16, v7
	v_xor_b32_e32 v6, v17, v7
	v_mad_u64_u32 v[16:17], s[6:7], v14, s24, 0
	v_mul_hi_u32 v18, v14, s21
	v_mov_b32_e32 v19, v5
	v_lshl_add_u64 v[16:17], v[18:19], 0, v[16:17]
	v_mad_u64_u32 v[20:21], s[6:7], v6, s21, 0
	v_add_co_u32_e32 v16, vcc, v16, v20
	v_mad_u64_u32 v[18:19], s[6:7], v6, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v21, vcc
	v_mov_b32_e32 v17, s20
	s_nop 0
	v_addc_co_u32_e32 v19, vcc, 0, v19, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[18:19]
	v_mul_lo_u32 v20, s23, v16
	v_mul_lo_u32 v17, s22, v17
	v_mad_u64_u32 v[18:19], s[6:7], s22, v16, 0
	v_add3_u32 v17, v19, v17, v20
	v_sub_u32_e32 v19, v6, v17
	v_mov_b32_e32 v22, s23
	v_sub_co_u32_e32 v14, vcc, v14, v18
	v_xor_b32_e32 v7, s68, v7
	s_nop 0
	v_subb_co_u32_e64 v18, s[6:7], v19, v22, vcc
	v_subrev_co_u32_e64 v19, s[6:7], s22, v14
	v_subb_co_u32_e32 v6, vcc, v6, v17, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v18, s[6:7], 0, v18, s[6:7]
	v_cmp_le_u32_e64 s[6:7], s23, v18
	v_cmp_le_u32_e32 vcc, s23, v6
	s_lshl_b64 s[58:59], s[28:29], 6
	v_cndmask_b32_e64 v20, 0, -1, s[6:7]
	v_cmp_le_u32_e64 s[6:7], s22, v19
	v_cndmask_b32_e64 v17, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s22, v14
	v_cndmask_b32_e64 v19, 0, -1, s[6:7]
	v_cmp_eq_u32_e64 s[6:7], s23, v18
	v_cndmask_b32_e64 v14, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s23, v6
	v_cndmask_b32_e64 v18, v20, v19, s[6:7]
	v_add_u32_e32 v19, 2, v16
	v_add_u32_e32 v20, 1, v16
	v_cmp_ne_u32_e64 s[6:7], 0, v18
	v_cndmask_b32_e32 v6, v17, v14, vcc
	v_cmp_ne_u32_e32 vcc, 0, v6
	v_cndmask_b32_e64 v18, v20, v19, s[6:7]
	v_mov_b32_e32 v17, v11
	v_cndmask_b32_e32 v6, v16, v18, vcc
	v_mov_b32_e32 v16, v11
	v_lshl_add_u64 v[16:17], v[10:11], 0, v[16:17]
	v_xor_b32_e32 v14, v16, v11
	v_xor_b32_e32 v10, v17, v11
	v_mad_u64_u32 v[16:17], s[6:7], v14, s24, 0
	v_mul_hi_u32 v18, v14, s21
	v_mov_b32_e32 v19, v5
	v_lshl_add_u64 v[16:17], v[18:19], 0, v[16:17]
	v_mad_u64_u32 v[20:21], s[6:7], v10, s21, 0
	v_add_co_u32_e32 v16, vcc, v16, v20
	v_mad_u64_u32 v[18:19], s[6:7], v10, s24, 0
	s_nop 0
	v_addc_co_u32_e32 v16, vcc, v17, v21, vcc
	v_mov_b32_e32 v17, s20
	s_nop 0
	v_addc_co_u32_e32 v19, vcc, 0, v19, vcc
	v_lshl_add_u64 v[16:17], v[16:17], 0, v[18:19]
	v_mul_lo_u32 v18, s23, v16
	v_mul_lo_u32 v19, s22, v17
	v_mad_u64_u32 v[16:17], s[6:7], s22, v16, 0
	v_add3_u32 v17, v17, v19, v18
	v_sub_u32_e32 v18, v10, v17
	v_sub_co_u32_e32 v14, vcc, v14, v16
	v_xor_b32_e32 v6, v6, v7
	s_nop 0
	v_subb_co_u32_e64 v16, s[6:7], v18, v22, vcc
	v_subrev_co_u32_e64 v18, s[6:7], s22, v14
	v_subb_co_u32_e32 v10, vcc, v10, v17, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v19, s[20:21], 0, v16, s[6:7]
	v_cmp_le_u32_e64 s[20:21], s23, v19
	v_subb_co_u32_e64 v16, s[6:7], v16, v22, s[6:7]
	s_nop 0
	v_cndmask_b32_e64 v20, 0, -1, s[20:21]
	v_cmp_le_u32_e64 s[20:21], s22, v18
	v_cmp_le_u32_e32 vcc, s23, v10
	v_sub_u32_e32 v7, v6, v7
	v_cndmask_b32_e64 v21, 0, -1, s[20:21]
	v_cmp_eq_u32_e64 s[20:21], s23, v19
	v_cndmask_b32_e64 v17, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s22, v14
	v_cndmask_b32_e64 v20, v20, v21, s[20:21]
	v_subrev_co_u32_e64 v21, s[6:7], s22, v18
	v_and_or_b32 v6, v41, 48, s58
	s_nop 0
	v_subbrev_co_u32_e64 v16, s[6:7], 0, v16, s[6:7]
	v_cmp_ne_u32_e64 s[6:7], 0, v20
	v_xad_u32 v7, v7, v11, v6
	v_mul_lo_u32 v7, v7, s56
	v_cndmask_b32_e64 v16, v19, v16, s[6:7]
	v_cndmask_b32_e64 v19, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s23, v10
	s_mov_b32 s38, s42
	s_mov_b32 s39, s43
	v_cndmask_b32_e32 v17, v17, v19, vcc
	v_cmp_ne_u32_e32 vcc, 0, v17
	v_mov_b32_e32 v17, s59
	s_add_i32 m0, s85, 0x6000
	v_cndmask_b32_e32 v10, v10, v16, vcc
	v_cndmask_b32_e64 v16, v18, v21, s[6:7]
	v_cndmask_b32_e32 v14, v14, v16, vcc
	v_xor_b32_e32 v14, v14, v11
	v_xor_b32_e32 v10, v10, v11
	v_sub_co_u32_e32 v14, vcc, v14, v11
	s_and_b32 s6, s37, 0xffff
	s_nop 0
	v_subb_co_u32_e32 v10, vcc, v10, v11, vcc
	v_ashrrev_i32_e32 v10, 31, v10
	v_or_b32_e32 v16, s58, v41
	v_and_b32_e32 v10, s56, v10
	s_or_b32 s37, s6, s17
	v_cmp_gt_i64_e64 s[6:7], s[14:15], v[16:17]
	v_add3_u32 v7, v10, v14, v7
	s_and_b64 vcc, s[6:7], s[18:19]
	v_cndmask_b32_e32 v7, v9, v7, vcc
	v_lshlrev_b32_e32 v10, 6, v37
	buffer_load_dwordx4 v7, s[36:39], 0 offen lds
	v_or_b32_e32 v7, v32, v34
	v_sub_u32_e32 v44, v1, v10
	v_or_b32_e32 v18, s30, v7
	v_or_b32_e32 v7, v44, v38
	v_add_u32_e32 v45, v7, v36
	v_ashrrev_i32_e32 v7, 31, v45
	v_xor_b32_e32 v11, v7, v45
	v_ashrrev_i32_e32 v14, 31, v11
	v_add_u32_sdwa v11, v11, v14 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_ashrrev_i32_e32 v11, 8, v11
	v_and_b32_e32 v26, 32, v42
	s_lshr_b32 s17, s16, 26
	v_lshlrev_b32_e32 v24, 5, v33
	v_mov_b32_e32 v25, v5
	v_mov_b32_e32 v19, s31
	v_xad_u32 v11, v11, v7, v26
	v_and_b32_e32 v14, 0xfd, v45
	s_mul_i32 s17, s52, s17
	s_mul_hi_u32 s22, s52, s30
	v_cmp_gt_i64_e32 vcc, s[34:35], v[24:25]
	v_cmp_gt_i64_e64 s[18:19], s[12:13], v[18:19]
	v_mad_u64_u32 v[20:21], s[20:21], s52, v11, v[14:15]
	s_add_i32 s17, s22, s17
	s_mul_i32 s22, s53, s30
	s_and_b64 s[20:21], s[18:19], vcc
	s_add_i32 s63, s17, s22
	s_mul_i32 s62, s52, s30
	s_add_u32 s44, s4, s62
	s_addc_u32 s17, s5, s63
	s_and_b32 s22, s52, 0x3fff
	s_or_b32 s83, s22, 0x4000
	s_and_b32 s17, s17, 0xffff
	s_lshl_b32 s22, s83, 16
	v_cndmask_b32_e64 v22, v9, v20, s[20:21]
	v_or_b32_e32 v20, 0x80, v24
	v_mov_b32_e32 v21, v5
	v_add_u32_e32 v46, v44, v36
	s_or_b32 s45, s17, s22
	v_cmp_gt_i64_e64 s[22:23], s[34:35], v[20:21]
	v_or_b32_e32 v20, v46, v38
	v_or_b32_e32 v20, 2, v20
	v_bitop3_b32 v21, v46, -3, v38 bitop3:0x36
	v_cmp_gt_i32_e64 s[26:27], 0, v46
	v_ashrrev_i32_e32 v48, 31, v46
	v_xor_b32_e32 v23, -3, v45
	v_cndmask_b32_e64 v20, v20, v21, s[26:27]
	v_ashrrev_i32_e32 v21, 31, v20
	v_add_u32_sdwa v20, v20, v21 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_ashrrev_i32_e32 v20, 8, v20
	v_xad_u32 v21, v20, v48, v26
	v_or_b32_e32 v20, 2, v45
	v_cmp_gt_i32_e64 s[20:21], 0, v45
	v_or3_b32 v27, v44, v40, 2
	v_lshlrev_b32_e32 v30, 5, v37
	v_cndmask_b32_e64 v20, v20, v23, s[20:21]
	v_ashrrev_i32_e32 v23, 31, v20
	v_add_u32_sdwa v20, v20, v23 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_ashrrev_i32_e32 v20, 8, v20
	v_xor_b32_e32 v28, v20, v7
	v_add_u32_e32 v7, v27, v36
	v_lshlrev_b32_e32 v29, 8, v28
	v_sub_u32_e32 v20, v7, v29
	v_mad_u64_u32 v[20:21], s[20:21], s52, v21, v[20:21]
	v_sub_u32_e32 v30, v43, v30
	s_and_b64 s[20:21], s[18:19], s[22:23]
	v_add_u32_e32 v30, v30, v24
	s_mov_b32 s46, s42
	s_mov_b32 s47, s43
	v_cndmask_b32_e64 v7, v9, v20, s[20:21]
	s_mul_i32 s17, s54, s33
	v_ashrrev_i32_e32 v31, 31, v30
	buffer_load_ubyte v47, v22, s[44:47], 0 offen
	s_nop 0
	buffer_load_ubyte v7, v7, s[44:47], 0 offen
	v_or_b32_e32 v20, s33, v32
	v_add_u32_e32 v22, s17, v46
	s_lshr_b32 s17, s28, 26
	v_xor_b32_e32 v30, v31, v30
	v_or_b32_e32 v20, s58, v20
	v_mov_b32_e32 v21, s59
	s_mul_i32 s17, s54, s17
	s_mul_hi_u32 s28, s54, s58
	v_ashrrev_i32_e32 v51, 31, v30
	v_or_b32_e32 v52, 3, v46
	v_xor_b32_e32 v53, -4, v46
	v_cmp_gt_i64_e64 s[20:21], s[14:15], v[20:21]
	s_add_i32 s17, s28, s17
	s_mul_i32 s28, s55, s58
	v_lshrrev_b32_e32 v51, 25, v51
	v_cndmask_b32_e64 v52, v52, v53, s[26:27]
	s_and_b64 s[24:25], s[20:21], vcc
	s_add_i32 s65, s17, s28
	s_mul_i32 s64, s54, s58
	v_add_u32_e32 v30, v30, v51
	v_ashrrev_i32_e32 v53, 31, v52
	s_add_u32 s48, s8, s64
	v_ashrrev_i32_e32 v30, 7, v30
	v_add_u32_sdwa v52, v52, v53 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	s_addc_u32 s17, s9, s65
	v_cndmask_b32_e64 v49, v9, v22, s[24:25]
	v_add_u32_e32 v22, 2, v22
	s_and_b64 s[24:25], s[20:21], s[22:23]
	v_xor_b32_e32 v30, v30, v31
	v_ashrrev_i32_e32 v52, 8, v52
	s_and_b32 s28, s54, 0x3fff
	v_cndmask_b32_e64 v50, v9, v22, s[24:25]
	v_or_b32_e32 v22, 16, v20
	v_mov_b32_e32 v23, s59
	v_add_u32_e32 v31, s33, v30
	v_lshlrev_b32_e32 v30, 8, v30
	v_xor_b32_e32 v48, v52, v48
	s_or_b32 s84, s28, 0x4000
	v_cmp_gt_i64_e64 s[24:25], s[14:15], v[22:23]
	v_sub_u32_e32 v51, v46, v30
	v_mul_lo_u32 v31, s54, v31
	v_add_u32_e32 v52, s33, v48
	v_lshlrev_b32_e32 v56, 8, v48
	s_and_b32 s17, s17, 0xffff
	s_lshl_b32 s28, s84, 16
	v_add3_u32 v51, v31, v51, 1
	s_and_b64 vcc, s[24:25], vcc
	v_sub_u32_e32 v48, v46, v56
	v_mul_lo_u32 v57, s54, v52
	s_or_b32 s49, s17, s28
	s_mov_b32 s50, s42
	s_mov_b32 s51, s43
	v_cndmask_b32_e32 v51, v9, v51, vcc
	v_add3_u32 v48, v57, v48, 3
	s_and_b64 vcc, s[24:25], s[22:23]
	v_cndmask_b32_e32 v9, v9, v48, vcc
	buffer_load_ubyte v55, v49, s[48:51], 0 offen
	buffer_load_ubyte v53, v50, s[48:51], 0 offen
	buffer_load_ubyte v54, v51, s[48:51], 0 offen
	buffer_load_ubyte v52, v9, s[48:51], 0 offen
	v_cmp_eq_u32_e64 s[22:23], 0, v15
	s_movk_i32 s53, 0x6000
	s_and_b64 vcc, exec, s[22:23]
	s_barrier
	s_cbranch_vccnz .LBB0_6
	s_barrier
.LBB0_6:
	s_ashr_i32 s26, s61, 31
	s_mov_b32 s27, s26
	s_xor_b64 s[28:29], s[26:27], s[60:61]
	s_lshr_b32 s17, s29, 31
	s_add_u32 s28, s28, s17
	s_addc_u32 s29, s29, 0
	v_and_b32_e32 v9, 7, v35
	s_ashr_i64 s[28:29], s[28:29], 1
	v_lshlrev_b32_e32 v49, 7, v35
	v_lshlrev_b32_e32 v50, 11, v37
	s_xor_b64 s[26:27], s[28:29], s[26:27]
	v_bitop3_b32 v48, v33, v35, 7 bitop3:0x78
	v_sub_u32_e32 v49, v49, v50
	v_bitop3_b32 v9, v33, v9, 4 bitop3:0x36
	s_lshl_b64 s[66:67], s[26:27], 1
	v_lshl_add_u32 v50, v40, 11, v49
	v_lshlrev_b32_e32 v51, 4, v48
	v_lshl_add_u32 v58, v15, 12, v49
	v_lshlrev_b32_e32 v9, 4, v9
	v_cmp_lt_i64_e64 s[26:27], s[26:27], 1
	s_mov_b32 s55, 0
	v_or_b32_e32 v48, v50, v51
	v_or_b32_e32 v51, v58, v51
	v_or_b32_e32 v49, v9, v50
	v_or_b32_e32 v50, v9, v58
	s_and_b64 vcc, exec, s[26:27]
	s_cbranch_vccnz .LBB0_17
	s_add_u32 s70, s66, -1
	s_addc_u32 s71, s67, -1
	s_add_i32 s17, s33, 1
	s_mul_i32 s17, s54, s17
	v_add3_u32 v57, v57, s54, v1
	v_add3_u32 v31, v31, s54, v1
	v_add_u32_e32 v1, s17, v1
	v_sub_u32_e32 v58, v1, v10
	v_add_u32_e32 v1, v28, v26
	v_mul_lo_u32 v1, s52, v1
	v_add3_u32 v1, v1, s52, v27
	v_sub_u32_e32 v57, v57, v10
	v_sub_u32_e32 v31, v31, v10
	v_sub_u32_e32 v59, v1, v29
	v_mul_lo_u32 v1, s52, v11
	v_lshrrev_b16_e32 v10, 6, v35
	v_lshlrev_b16_e32 v11, 2, v35
	v_or_b32_e32 v10, v11, v10
	s_lshl_b32 s16, s16, 6
	v_sub_u32_e32 v62, 0, v3
	v_mov_b32_e32 v3, v5
	v_mov_b32_e32 v9, v5
	v_and_b32_e32 v10, 0xfd, v10
	s_or_b32 s16, s33, s16
	v_ashrrev_i32_e32 v63, 31, v62
	v_add3_u32 v60, v1, s52, v10
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[8:9]
	v_add_u32_e32 v1, s16, v39
	v_lshl_add_u64 v[2:3], v[2:3], 0, v[62:63]
	s_movk_i32 s17, 0xf7ff
	v_mul_lo_u32 v1, v1, s56
	s_movk_i32 s16, 0x80
	v_sub_u32_e32 v56, v57, v56
	v_sub_u32_e32 v30, v31, v30
	v_sub_co_u32_e32 v26, vcc, s17, v2
	s_mov_b64 s[72:73], 0x800
	v_or_b32_e32 v4, 0x100, v4
	v_add3_u32 v61, v1, v0, s16
	v_mov_b32_e32 v0, 0
	s_movk_i32 s74, 0xff80
	s_movk_i32 s76, 0xf800
	v_add_u32_e32 v56, 3, v56
	v_add_u32_e32 v57, 1, v30
	v_subb_co_u32_e32 v27, vcc, -1, v3, vcc
	v_lshl_add_u64 v[28:29], v[2:3], 0, s[72:73]
	v_or_b32_e32 v24, 0x180, v24
	s_mov_b32 s86, 0
	s_movk_i32 s17, 0x2000
	s_mov_b32 s16, 0x8000
	v_bfrev_b32_e32 v62, -2
	s_mov_b32 s69, s68
	s_mov_b32 s38, s42
	s_mov_b32 s39, s43
	s_mov_b32 s75, -1
	s_mov_b32 s46, s42
	s_mov_b32 s47, s43
	s_mov_b32 s50, s42
	s_mov_b32 s51, s43
	s_mov_b32 s77, -1
	s_mov_b64 s[78:79], 0x100
	v_mov_b64_e32 v[30:31], v[4:5]
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v8, v0
	v_mov_b32_e32 v9, v0
	v_mov_b32_e32 v10, v0
	v_mov_b32_e32 v11, v0
.LBB0_8:
	s_mov_b32 s87, s55
	s_mov_b32 s55, s17
	s_mov_b32 s88, s53
	s_mov_b32 s53, s16
	v_cmp_gt_i64_e64 s[16:17], s[34:35], v[30:31]
	s_and_b64 vcc, s[2:3], s[16:17]
	s_add_i32 m0, s55, s85
	s_add_u32 s26, s56, s68
	s_addc_u32 s27, s57, s68
	s_xor_b64 s[80:81], s[26:27], s[68:69]
	v_cvt_f32_u32_e32 v4, s80
	v_cvt_f32_u32_e32 v63, s81
	s_sub_u32 s26, 0, s80
	s_subb_u32 s27, 0, s81
	v_cndmask_b32_e32 v64, v62, v61, vcc
	v_fmac_f32_e32 v4, 0x4f800000, v63
	v_rcp_f32_e32 v4, v4
	v_cmp_gt_i32_e32 vcc, 0, v29
	s_waitcnt vmcnt(0)
	s_barrier
	v_mul_f32_e32 v4, 0x5f7ffffc, v4
	v_mul_f32_e32 v63, 0x2f800000, v4
	v_trunc_f32_e32 v63, v63
	v_fmac_f32_e32 v4, 0xcf800000, v63
	v_cvt_u32_f32_e32 v4, v4
	v_cvt_u32_f32_e32 v63, v63
	v_cndmask_b32_e32 v65, v29, v27, vcc
	v_ashrrev_i32_e32 v66, 31, v65
	v_readfirstlane_b32 s28, v4
	v_readfirstlane_b32 s90, v63
	s_mul_hi_u32 s89, s26, s28
	s_mul_i32 s91, s26, s90
	s_mul_i32 s29, s27, s28
	s_add_i32 s89, s89, s91
	s_add_i32 s89, s89, s29
	s_mul_i32 s92, s26, s28
	s_mul_i32 s91, s28, s89
	s_mul_hi_u32 s93, s28, s92
	s_mul_hi_u32 s29, s28, s89
	s_add_u32 s91, s93, s91
	s_addc_u32 s29, 0, s29
	s_mul_hi_u32 s94, s90, s92
	s_mul_i32 s92, s90, s92
	s_add_u32 s91, s91, s92
	s_mul_hi_u32 s93, s90, s89
	s_addc_u32 s29, s29, s94
	s_addc_u32 s91, s93, 0
	s_mul_i32 s89, s90, s89
	s_add_u32 s29, s29, s89
	s_addc_u32 s89, 0, s91
	s_add_u32 s28, s28, s29
	s_addc_u32 s29, s90, s89
	s_mul_i32 s89, s26, s29
	s_mul_hi_u32 s90, s26, s28
	s_add_i32 s89, s90, s89
	s_mul_i32 s27, s27, s28
	s_add_i32 s89, s89, s27
	s_mul_i32 s26, s26, s28
	s_mul_hi_u32 s90, s29, s26
	s_mul_i32 s91, s29, s26
	s_mul_i32 s93, s28, s89
	s_mul_hi_u32 s26, s28, s26
	s_mul_hi_u32 s92, s28, s89
	s_add_u32 s26, s26, s93
	s_addc_u32 s92, 0, s92
	s_add_u32 s26, s26, s91
	s_mul_hi_u32 s27, s29, s89
	s_addc_u32 s26, s92, s90
	s_addc_u32 s27, s27, 0
	s_mul_i32 s89, s29, s89
	s_add_u32 s26, s26, s89
	s_barrier
	buffer_load_dwordx4 v64, s[40:43], 0 offen lds
	v_cndmask_b32_e32 v64, v28, v26, vcc
	s_addc_u32 s27, 0, s27
	v_mov_b32_e32 v67, v66
	s_add_u32 s28, s28, s26
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[66:67]
	s_addc_u32 s29, s29, s27
	v_xor_b32_e32 v67, v64, v66
	v_xor_b32_e32 v63, v65, v66
	v_mad_u64_u32 v[64:65], s[26:27], v67, s29, 0
	v_mul_hi_u32 v4, v67, s28
	v_lshl_add_u64 v[64:65], v[4:5], 0, v[64:65]
	v_mad_u64_u32 v[70:71], s[26:27], v63, s28, 0
	v_add_co_u32_e32 v4, vcc, v64, v70
	v_mad_u64_u32 v[68:69], s[26:27], v63, s29, 0
	s_nop 0
	v_addc_co_u32_e32 v64, vcc, v65, v71, vcc
	v_mov_b32_e32 v65, s86
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[64:65], v[64:65], 0, v[68:69]
	v_mul_lo_u32 v4, s81, v64
	v_mul_lo_u32 v65, s80, v65
	v_mad_u64_u32 v[68:69], s[26:27], s80, v64, 0
	v_add3_u32 v4, v69, v65, v4
	v_sub_u32_e32 v65, v63, v4
	v_mov_b32_e32 v72, s81
	v_sub_co_u32_e32 v67, vcc, v67, v68
	s_add_i32 m0, s53, s85
	s_nop 0
	v_subb_co_u32_e64 v65, s[26:27], v65, v72, vcc
	v_subrev_co_u32_e64 v68, s[26:27], s80, v67
	v_subb_co_u32_e32 v4, vcc, v63, v4, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v65, s[26:27], 0, v65, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s81, v65
	v_cmp_le_u32_e32 vcc, s81, v4
	s_nop 0
	v_cndmask_b32_e64 v69, 0, -1, s[26:27]
	v_cmp_le_u32_e64 s[26:27], s80, v68
	v_cndmask_b32_e64 v63, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s80, v67
	v_cndmask_b32_e64 v68, 0, -1, s[26:27]
	v_cmp_eq_u32_e64 s[26:27], s81, v65
	v_cndmask_b32_e64 v67, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s81, v4
	v_cndmask_b32_e64 v65, v69, v68, s[26:27]
	v_add_u32_e32 v68, 2, v64
	v_add_u32_e32 v69, 1, v64
	v_cmp_ne_u32_e64 s[26:27], 0, v65
	v_cndmask_b32_e32 v4, v63, v67, vcc
	v_cmp_ne_u32_e32 vcc, 0, v4
	v_cndmask_b32_e64 v65, v69, v68, s[26:27]
	v_xor_b32_e32 v63, s68, v66
	v_cndmask_b32_e32 v4, v64, v65, vcc
	v_ashrrev_i32_e32 v64, 31, v29
	v_mov_b32_e32 v65, v64
	v_xor_b32_e32 v4, v4, v63
	v_lshl_add_u64 v[66:67], v[28:29], 0, v[64:65]
	v_sub_u32_e32 v4, v4, v63
	v_xor_b32_e32 v73, v66, v64
	v_xad_u32 v63, v4, v64, v6
	v_xor_b32_e32 v65, v67, v64
	v_mad_u64_u32 v[66:67], s[26:27], v73, s29, 0
	v_mul_hi_u32 v4, v73, s28
	v_lshl_add_u64 v[66:67], v[4:5], 0, v[66:67]
	v_mad_u64_u32 v[70:71], s[26:27], v65, s28, 0
	v_add_co_u32_e32 v4, vcc, v66, v70
	v_mad_u64_u32 v[68:69], s[26:27], v65, s29, 0
	s_nop 0
	v_addc_co_u32_e32 v66, vcc, v67, v71, vcc
	v_mov_b32_e32 v67, s86
	s_nop 0
	v_addc_co_u32_e32 v69, vcc, 0, v69, vcc
	v_lshl_add_u64 v[66:67], v[66:67], 0, v[68:69]
	v_mul_lo_u32 v4, s81, v66
	v_mul_lo_u32 v68, s80, v67
	v_mad_u64_u32 v[66:67], s[26:27], s80, v66, 0
	v_add3_u32 v4, v67, v68, v4
	v_sub_u32_e32 v67, v65, v4
	v_sub_co_u32_e32 v66, vcc, v73, v66
	v_mul_lo_u32 v63, v63, s56
	s_nop 0
	v_subb_co_u32_e64 v67, s[26:27], v67, v72, vcc
	v_subrev_co_u32_e64 v68, s[26:27], s80, v66
	v_subb_co_u32_e32 v4, vcc, v65, v4, vcc
	s_nop 0
	v_subbrev_co_u32_e64 v69, s[28:29], 0, v67, s[26:27]
	v_cmp_le_u32_e64 s[28:29], s81, v69
	v_subb_co_u32_e64 v67, s[26:27], v67, v72, s[26:27]
	s_nop 0
	v_cndmask_b32_e64 v70, 0, -1, s[28:29]
	v_cmp_le_u32_e64 s[28:29], s80, v68
	v_cmp_le_u32_e32 vcc, s81, v4
	s_nop 0
	v_cndmask_b32_e64 v71, 0, -1, s[28:29]
	v_cmp_eq_u32_e64 s[28:29], s81, v69
	v_cndmask_b32_e64 v65, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s80, v66
	v_cndmask_b32_e64 v70, v70, v71, s[28:29]
	v_subrev_co_u32_e64 v71, s[26:27], s80, v68
	s_nop 1
	v_subbrev_co_u32_e64 v67, s[26:27], 0, v67, s[26:27]
	v_cmp_ne_u32_e64 s[26:27], 0, v70
	s_nop 1
	v_cndmask_b32_e64 v67, v69, v67, s[26:27]
	v_cndmask_b32_e64 v69, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s81, v4
	s_nop 1
	v_cndmask_b32_e32 v65, v65, v69, vcc
	v_cmp_ne_u32_e32 vcc, 0, v65
	v_cndmask_b32_e64 v65, v68, v71, s[26:27]
	s_nop 0
	v_cndmask_b32_e32 v65, v66, v65, vcc
	v_cndmask_b32_e32 v4, v4, v67, vcc
	v_xor_b32_e32 v65, v65, v64
	v_xor_b32_e32 v4, v4, v64
	v_sub_co_u32_e32 v65, vcc, v65, v64
	s_nop 1
	v_subb_co_u32_e32 v4, vcc, v4, v64, vcc
	v_ashrrev_i32_e32 v4, 31, v4
	v_and_b32_e32 v4, s56, v4
	v_add3_u32 v4, v4, v65, v63
	s_and_b64 vcc, s[6:7], s[16:17]
	v_cndmask_b32_e32 v4, v62, v4, vcc
	buffer_load_dwordx4 v4, s[36:39], 0 offen lds
	v_lshl_add_u64 v[64:65], v[24:25], 0, s[74:75]
	v_cmp_gt_i64_e32 vcc, s[34:35], v[64:65]
	s_and_b64 s[16:17], s[18:19], vcc
	v_cndmask_b32_e64 v4, v62, v60, s[16:17]
	v_cmp_gt_i64_e64 s[16:17], s[34:35], v[24:25]
	v_add_u32_e32 v63, v36, v59
	s_and_b64 s[26:27], s[18:19], s[16:17]
	v_cndmask_b32_e64 v63, v62, v63, s[26:27]
	v_add_u32_e32 v64, v36, v58
	s_and_b64 s[26:27], s[20:21], vcc
	v_cndmask_b32_e64 v65, v62, v64, s[26:27]
	v_add_u32_e32 v64, 2, v64
	s_and_b64 s[26:27], s[20:21], s[16:17]
	v_cndmask_b32_e64 v66, v62, v64, s[26:27]
	v_add_u32_e32 v64, v36, v57
	s_and_b64 vcc, s[24:25], vcc
	v_cndmask_b32_e32 v67, v62, v64, vcc
	v_add_u32_e32 v64, v36, v56
	s_and_b64 vcc, s[24:25], s[16:17]
	v_cndmask_b32_e32 v68, v62, v64, vcc
	buffer_load_ubyte v64, v65, s[48:51], 0 offen
	s_nop 0
	buffer_load_ubyte v65, v66, s[48:51], 0 offen
	s_nop 0
	buffer_load_ubyte v66, v67, s[48:51], 0 offen
	s_nop 0
	buffer_load_ubyte v67, v68, s[48:51], 0 offen
	s_nop 0
	buffer_load_ubyte v4, v4, s[44:47], 0 offen
	s_nop 0
	buffer_load_ubyte v63, v63, s[44:47], 0 offen
	v_add_u32_e32 v68, s87, v48
	ds_read_b128 v[70:73], v68
	v_add_u32_e32 v68, s88, v51
	ds_read_b128 v[74:77], v68
	ds_read_b128 v[78:81], v68 offset:2048
	s_barrier
	s_setprio 1
	v_and_b32_e32 v47, 0xff, v47
	v_and_b32_e32 v55, 0xff, v55
	v_and_b32_e32 v54, 0xff, v54
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[70:73], v[74:77], v[0:3], v47, v55 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[70:73], v[78:81], v[8:11], v47, v54 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_barrier
	v_add_u32_e32 v47, s87, v49
	ds_read_b128 v[68:71], v47
	v_add_u32_e32 v47, s88, v50
	ds_read_b128 v[72:75], v47
	ds_read_b128 v[76:79], v47 offset:2048
	s_barrier
	s_setprio 1
	v_and_b32_e32 v7, 0xff, v7
	v_and_b32_e32 v47, 0xff, v53
	s_waitcnt lgkmcnt(1)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[68:71], v[72:75], v[0:3], v7, v47 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_and_b32_e32 v47, 0xff, v52
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[8:11], v[68:71], v[76:79], v[8:11], v7, v47 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_setprio 0
	s_add_u32 s70, s70, -1
	s_addc_u32 s71, s71, -1
	v_add_u32_e32 v56, s54, v56
	v_add_u32_e32 v57, s54, v57
	v_add_u32_e32 v58, s54, v58
	v_add_u32_e32 v59, s52, v59
	v_add_u32_e32 v60, s52, v60
	v_lshl_add_u64 v[26:27], v[26:27], 0, s[76:77]
	v_lshl_add_u64 v[28:29], v[28:29], 0, s[72:73]
	v_lshl_add_u64 v[24:25], v[24:25], 0, s[78:79]
	v_lshl_add_u64 v[30:31], v[30:31], 0, s[78:79]
	v_add_u32_e32 v61, 0x80, v61
	s_cmp_lg_u64 s[70:71], 0
	s_mov_b32 s16, s88
	s_mov_b32 s17, s87
	s_waitcnt vmcnt(2)
	v_mov_b32_e32 v52, v67
	v_mov_b32_e32 v54, v66
	v_mov_b32_e32 v53, v65
	v_mov_b32_e32 v55, v64
	s_waitcnt vmcnt(0)
	v_mov_b32_e32 v7, v63
	v_mov_b32_e32 v47, v4
	s_cbranch_scc1 .LBB0_8
	v_mov_b32_e32 v52, v67
	v_mov_b32_e32 v54, v66
	v_mov_b32_e32 v53, v65
	v_mov_b32_e32 v55, v64
	v_mov_b32_e32 v7, v63
	v_mov_b32_e32 v47, v4
	s_andn2_b64 vcc, exec, s[22:23]
	s_cbranch_vccnz .LBB0_11
.LBB0_10:
	s_barrier
.LBB0_11:
	v_add_u32_e32 v5, s55, v48
	s_waitcnt vmcnt(6)
	s_barrier
	v_add_u32_e32 v4, s53, v51
	ds_read_b128 v[26:29], v5
	ds_read_b128 v[56:59], v4
	v_add_u32_e32 v6, s55, v49
	v_add_u32_e32 v5, s53, v50
	ds_read_b128 v[48:51], v6
	ds_read_b128 v[60:63], v5
	ds_read_b128 v[64:67], v4 offset:2048
	s_waitcnt vmcnt(5)
	v_and_b32_e32 v24, 0xff, v47
	s_waitcnt vmcnt(3)
	v_and_b32_e32 v4, 0xff, v55
	v_and_b32_e32 v25, 0xff, v7
	s_mov_b64 s[38:39], s[42:43]
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[26:29], v[56:59], v[0:3], v24, v4 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(2)
	v_and_b32_e32 v4, 0xff, v53
	ds_read_b128 v[56:59], v5 offset:2048
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[48:51], v[60:63], v[0:3], v25, v4 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt vmcnt(1)
	s_nop 1
	v_and_b32_e32 v0, 0xff, v54
	s_waitcnt lgkmcnt(1)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[26:29], v[64:67], v[8:11], v24, v0 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_or_b32_e32 v24, 2, v46
	s_waitcnt vmcnt(0)
	s_nop 0
	v_and_b32_e32 v8, 0xff, v52
	s_waitcnt lgkmcnt(0)
	s_nop 0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[48:51], v[56:59], v[0:3], v25, v8 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_branch .LBB0_13
.LBB0_12:
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v2, v0
	v_mov_b32_e32 v3, v0
	v_mov_b32_e32 v1, v0
	v_mov_b64_e32 v[6:7], v[2:3]
	s_mov_b64 s[40:41], s[2:3]
	s_mov_b64 s[36:37], s[6:7]
	v_mov_b64_e32 v[4:5], v[0:1]
	v_mov_b32_e32 v34, v8
	v_mov_b32_e32 v38, v9
.LBB0_13:
	s_load_dwordx2 s[28:29], s[0:1], 0x60
	v_cmp_gt_i64_e64 s[0:1], s[60:61], 1
	s_and_b64 s[0:1], s[0:1], exec
	s_cselect_b32 s45, s67, 0
	s_cselect_b32 s44, s66, 0
	v_mov_b64_e32 v[8:9], s[60:61]
	v_cmp_ge_i64_e32 vcc, s[44:45], v[8:9]
	s_mov_b32 s53, 0
	v_cmp_gt_i64_e64 s[2:3], s[14:15], v[20:21]
	v_cmp_gt_i64_e64 s[0:1], s[14:15], v[22:23]
	s_cbranch_vccnz .LBB0_16
	v_sub_u32_e32 v8, -4, v46
	v_or_b32_e32 v9, 3, v46
	v_cmp_gt_i32_e32 vcc, 0, v46
	v_cmp_gt_i64_e64 s[6:7], s[12:13], v[12:13]
	v_cmp_gt_i64_e64 s[16:17], s[12:13], v[18:19]
	v_cndmask_b32_e32 v8, v9, v8, vcc
	v_ashrrev_i32_e32 v9, 31, v8
	v_add_u32_sdwa v8, v8, v9 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_ashrrev_i32_e32 v8, 8, v8
	v_ashrrev_i32_e32 v9, 31, v46
	v_xor_b32_e32 v13, v8, v9
	v_lshlrev_b32_e32 v8, 5, v37
	v_sub_u32_e32 v8, v43, v8
	v_lshlrev_b32_e32 v18, 5, v33
	s_add_u32 s20, s4, s62
	v_add_u32_e32 v8, v8, v18
	s_addc_u32 s4, s5, s63
	v_ashrrev_i32_e32 v10, 31, v8
	s_and_b32 s4, s4, 0xffff
	s_lshl_b32 s5, s83, 16
	v_xor_b32_e32 v8, v10, v8
	s_or_b32 s21, s4, s5
	v_ashrrev_i32_e32 v11, 31, v8
	s_add_u32 s24, s8, s64
	v_lshrrev_b32_e32 v11, 25, v11
	s_addc_u32 s4, s9, s65
	v_add_u32_e32 v8, v8, v11
	s_and_b32 s4, s4, 0xffff
	s_lshl_b32 s5, s84, 16
	v_ashrrev_i32_e32 v8, 7, v8
	s_or_b32 s25, s4, s5
	v_xor_b32_e32 v21, v8, v10
	v_sub_u32_e32 v8, -3, v45
	v_or_b32_e32 v10, 2, v45
	v_cmp_gt_i32_e64 s[4:5], 0, v45
	v_lshlrev_b32_e32 v22, 7, v35
	v_lshlrev_b32_e32 v19, 11, v40
	v_cndmask_b32_e64 v8, v10, v8, s[4:5]
	v_ashrrev_i32_e32 v10, 31, v8
	s_movk_i32 s5, 0xf800
	v_mul_i32_i24_e32 v23, 0xfffff800, v37
	v_lshl_add_u32 v15, v15, 12, v22
	s_mov_b32 s8, 0xa000
	v_add_u32_sdwa v8, v8, v10 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_ashrrev_i32_e32 v20, 31, v45
	v_bitop3_b32 v30, v39, 7, v35 bitop3:0x48
	v_lshlrev_b32_e32 v10, 4, v39
	v_add3_u32 v39, v15, v23, s8
	v_mad_i32_i24 v15, v37, s5, v19
	s_movk_i32 s5, 0x4000
	v_add3_u32 v37, v15, v22, s5
	v_xor_b32_e32 v15, v20, v45
	v_ashrrev_i32_e32 v22, 31, v15
	v_add_u32_sdwa v15, v15, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_lshrrev_b32_e32 v8, 8, v8
	v_ashrrev_i32_e32 v15, 8, v15
	v_xor_b32_e32 v8, v8, v20
	v_xor_b32_e32 v15, v15, v20
	v_or_b32_e32 v20, v46, v38
	v_or_b32_e32 v22, 2, v20
	v_sub_u32_e32 v20, -3, v20
	v_cndmask_b32_e32 v20, v22, v20, vcc
	v_ashrrev_i32_e32 v22, 31, v20
	v_add_u32_sdwa v20, v20, v22 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_3
	v_ashrrev_i32_e32 v20, 8, v20
	s_movk_i32 s4, 0xff00
	v_xor_b32_e32 v29, v20, v9
	v_add_u32_e32 v20, v36, v44
	s_add_i32 s8, s33, s44
	v_mad_i32_i24 v9, v13, s4, v20
	v_add_u32_e32 v13, s8, v13
	v_mul_i32_i24_e32 v25, 0xffffff00, v8
	v_and_b32_e32 v8, 7, v35
	v_mul_lo_u32 v13, s54, v13
	v_bitop3_b32 v8, v33, v8, 4 bitop3:0x36
	v_add3_u32 v9, v9, v13, 3
	v_add_u32_e32 v13, s8, v21
	v_lshlrev_b32_e32 v40, 4, v8
	v_bitop3_b32 v8, v33, v35, 7 bitop3:0x78
	v_mad_u64_u32 v[22:23], s[4:5], s54, v13, v[20:21]
	v_lshlrev_b32_e32 v13, 8, v21
	v_lshlrev_b32_e32 v31, 4, v8
	v_and_or_b32 v8, v41, 48, s58
	v_and_b32_e32 v41, 32, v42
	v_sub_u32_e32 v13, v22, v13
	v_add_u32_e32 v26, 1, v13
	v_add3_u32 v13, v29, v41, s44
	v_mul_lo_u32 v13, s52, v13
	v_add3_u32 v13, v20, v25, v13
	s_mul_i32 s4, s54, s8
	v_add3_u32 v29, v13, v38, 2
	v_add3_u32 v13, v15, v41, s44
	v_add_u32_e32 v27, s4, v24
	v_add_u32_e32 v28, s4, v46
	v_mad_u64_u32 v[14:15], s[4:5], s52, v13, v[14:15]
	v_cmp_gt_i64_e64 s[14:15], s[14:15], v[16:17]
	v_and_b32_e32 v16, 0x100, v43
	s_lshl_b64 s[4:5], s[44:45], 11
	v_lshlrev_b32_e32 v35, 4, v30
	v_mov_b32_e32 v11, 0
	v_sub_u32_e32 v16, 0, v16
	v_lshl_or_b32 v20, v30, 8, s4
	v_mov_b32_e32 v21, s5
	v_ashrrev_i32_e32 v17, 31, v16
	v_mov_b32_e32 v19, v11
	v_lshl_add_u64 v[20:21], v[20:21], 0, v[10:11]
	s_lshl_b64 s[4:5], s[44:45], 8
	v_lshl_or_b32 v10, s44, 7, v35
	s_mov_b32 s27, 0x27000
	s_mov_b32 s26, 0x7ffffffe
	s_lshl_b32 s64, s82, 7
	v_lshl_add_u64 v[16:17], v[20:21], 0, v[16:17]
	v_lshl_add_u64 v[18:19], s[4:5], 0, v[18:19]
	v_lshl_or_b32 v20, v30, 5, s4
	v_mov_b32_e32 v21, s5
	v_mad_u64_u32 v[12:13], s[4:5], s56, v12, v[10:11]
	s_ashr_i32 s46, s57, 31
	v_bfrev_b32_e32 v13, -2
	s_add_i32 s55, s64, 0x4000
	s_mov_b32 s47, s46
	v_add_u32_e32 v15, v39, v31
	v_add_u32_e32 v30, v39, v40
	v_add_u32_e32 v31, v37, v31
	v_add_u32_e32 v35, v37, v40
	s_mov_b64 s[48:49], 0x80
	s_mov_b64 s[50:51], 0x800
	s_mov_b64 s[62:63], 0x100
	v_mov_b64_e32 v[22:23], s[60:61]
	s_add_i32 s64, s64, 0xa000
	s_mov_b32 s22, s26
	s_mov_b32 s23, s27
	s_mov_b32 s65, 0
	s_mov_b32 s66, 0
.LBB0_15:
	v_cmp_gt_i64_e64 s[4:5], s[34:35], v[20:21]
	v_ashrrev_i32_e32 v24, 31, v17
	s_and_b64 s[8:9], s[6:7], s[4:5]
	v_lshl_add_u64 v[40:41], v[18:19], 0, s[48:49]
	v_xor_b32_e32 v43, v24, v17
	s_add_u32 s18, s56, s46
	v_cndmask_b32_e64 v10, v13, v12, s[8:9]
	v_cmp_gt_i64_e64 s[8:9], s[34:35], v[40:41]
	v_ashrrev_i32_e32 v40, 31, v43
	s_addc_u32 s19, s57, s46
	s_mov_b32 m0, s55
	v_xor_b32_e32 v42, v24, v16
	v_mov_b32_e32 v41, v40
	s_xor_b64 s[60:61], s[18:19], s[46:47]
	buffer_load_dwordx4 v10, s[40:43], 0 offen lds
	v_lshl_add_u64 v[42:43], v[42:43], 0, v[40:41]
	v_cvt_f32_u32_e32 v10, s60
	v_cvt_f32_u32_e32 v41, s61
	v_mov_b32_e32 v25, v24
	v_lshl_add_u64 v[44:45], v[16:17], 0, v[24:25]
	v_xor_b32_e32 v25, s46, v40
	v_fmac_f32_e32 v10, 0x4f800000, v41
	v_rcp_f32_e32 v10, v10
	v_xor_b32_e32 v54, v43, v40
	v_xor_b32_e32 v55, v42, v40
	s_sub_u32 s18, 0, s60
	v_mul_f32_e32 v10, 0x5f7ffffc, v10
	v_mul_f32_e32 v40, 0x2f800000, v10
	v_trunc_f32_e32 v40, v40
	v_fmac_f32_e32 v10, 0xcf800000, v40
	v_cvt_u32_f32_e32 v40, v40
	v_cvt_u32_f32_e32 v10, v10
	s_subb_u32 s19, 0, s61
	v_xor_b32_e32 v52, v45, v24
	v_readfirstlane_b32 s67, v40
	v_readfirstlane_b32 s68, v10
	s_mul_i32 s69, s18, s67
	s_mul_hi_u32 s71, s18, s68
	s_mul_i32 s70, s19, s68
	s_add_i32 s69, s71, s69
	s_mul_i32 s72, s18, s68
	s_add_i32 s69, s69, s70
	s_mul_hi_u32 s71, s68, s72
	s_mul_i32 s74, s68, s69
	s_mul_hi_u32 s70, s68, s69
	s_add_u32 s71, s71, s74
	s_mul_hi_u32 s73, s67, s72
	s_mul_i32 s72, s67, s72
	s_addc_u32 s70, 0, s70
	s_add_u32 s71, s71, s72
	s_mul_hi_u32 s75, s67, s69
	s_addc_u32 s70, s70, s73
	s_mul_i32 s69, s67, s69
	s_addc_u32 s71, s75, 0
	s_add_u32 s69, s70, s69
	s_addc_u32 s70, 0, s71
	s_add_u32 s68, s68, s69
	s_addc_u32 s67, s67, s70
	s_mul_hi_u32 s69, s18, s68
	s_mul_i32 s70, s18, s68
	s_mul_i32 s18, s18, s67
	s_mul_i32 s19, s19, s68
	s_add_i32 s18, s69, s18
	s_add_i32 s18, s18, s19
	s_mul_hi_u32 s71, s67, s70
	s_mul_i32 s72, s67, s70
	s_mul_hi_u32 s70, s68, s70
	s_mul_i32 s73, s68, s18
	s_mul_hi_u32 s69, s68, s18
	s_add_u32 s70, s70, s73
	s_addc_u32 s69, 0, s69
	s_add_u32 s70, s70, s72
	s_mul_hi_u32 s19, s67, s18
	s_addc_u32 s69, s69, s71
	s_mul_i32 s18, s67, s18
	s_addc_u32 s19, s19, 0
	s_add_u32 s18, s69, s18
	s_addc_u32 s19, 0, s19
	s_add_u32 s68, s68, s18
	v_cmp_gt_i64_e32 vcc, s[34:35], v[18:19]
	s_addc_u32 s67, s67, s19
	v_mad_u64_u32 v[40:41], s[18:19], v54, s68, 0
	v_mad_u64_u32 v[42:43], s[18:19], v52, s68, 0
	v_add_u32_e32 v36, s65, v14
	s_and_b64 s[18:19], s[16:17], vcc
	v_add_u32_e32 v38, s65, v29
	v_cndmask_b32_e64 v36, v13, v36, s[18:19]
	s_and_b64 s[18:19], s[16:17], s[8:9]
	v_add_u32_e32 v46, s66, v28
	v_add_u32_e32 v48, s66, v26
	v_cndmask_b32_e64 v38, v13, v38, s[18:19]
	s_and_b64 s[18:19], s[2:3], vcc
	s_and_b64 vcc, s[0:1], vcc
	v_add_u32_e32 v47, s66, v27
	v_add_u32_e32 v49, s66, v9
	v_xor_b32_e32 v53, v44, v24
	v_mul_hi_u32 v10, v55, s68
	v_cndmask_b32_e64 v57, v13, v46, s[18:19]
	s_and_b64 s[18:19], s[2:3], s[8:9]
	v_cndmask_b32_e32 v59, v13, v48, vcc
	s_and_b64 vcc, s[0:1], s[8:9]
	v_mad_u64_u32 v[44:45], s[8:9], v55, s67, 0
	v_cndmask_b32_e64 v58, v13, v47, s[18:19]
	v_cndmask_b32_e32 v60, v13, v49, vcc
	v_lshl_add_u64 v[44:45], v[10:11], 0, v[44:45]
	buffer_load_ubyte v61, v36, s[20:23], 0 offen
	buffer_load_ubyte v62, v38, s[20:23], 0 offen
	s_nop 0
	buffer_load_ubyte v57, v57, s[24:27], 0 offen
	s_nop 0
	buffer_load_ubyte v58, v58, s[24:27], 0 offen
	s_nop 0
	buffer_load_ubyte v59, v59, s[24:27], 0 offen
	s_nop 0
	buffer_load_ubyte v60, v60, s[24:27], 0 offen
	v_add_co_u32_e32 v36, vcc, v44, v40
	v_mad_u64_u32 v[46:47], s[8:9], v54, s67, 0
	v_mad_u64_u32 v[48:49], s[8:9], v53, s67, 0
	v_mul_hi_u32 v10, v53, s68
	v_addc_co_u32_e32 v36, vcc, v45, v41, vcc
	v_mov_b32_e32 v37, s53
	v_lshl_add_u64 v[48:49], v[10:11], 0, v[48:49]
	v_addc_co_u32_e32 v47, vcc, 0, v47, vcc
	v_lshl_add_u64 v[36:37], v[36:37], 0, v[46:47]
	v_add_co_u32_e32 v10, vcc, v48, v42
	v_mad_u64_u32 v[50:51], s[8:9], v52, s67, 0
	v_mul_lo_u32 v10, s61, v36
	v_mul_lo_u32 v37, s60, v37
	v_mad_u64_u32 v[40:41], s[8:9], s60, v36, 0
	v_addc_co_u32_e32 v38, vcc, v49, v43, vcc
	v_add3_u32 v10, v41, v37, v10
	s_nop 0
	v_addc_co_u32_e32 v51, vcc, 0, v51, vcc
	v_mov_b32_e32 v56, s61
	v_sub_u32_e32 v37, v54, v10
	v_sub_co_u32_e32 v40, vcc, v55, v40
	v_mov_b32_e32 v39, s53
	s_nop 0
	v_subb_co_u32_e64 v37, s[8:9], v37, v56, vcc
	v_subb_co_u32_e32 v10, vcc, v54, v10, vcc
	v_cmp_le_u32_e32 vcc, s60, v40
	v_lshl_add_u64 v[38:39], v[38:39], 0, v[50:51]
	v_mul_lo_u32 v43, s61, v38
	v_cndmask_b32_e64 v41, 0, -1, vcc
	v_subrev_co_u32_e32 v40, vcc, s60, v40
	v_mul_lo_u32 v45, s60, v39
	s_nop 0
	v_subbrev_co_u32_e32 v37, vcc, 0, v37, vcc
	v_cmp_le_u32_e32 vcc, s60, v40
	v_mad_u64_u32 v[38:39], s[8:9], s60, v38, 0
	s_nop 0
	v_cndmask_b32_e64 v40, 0, -1, vcc
	v_cmp_le_u32_e32 vcc, s61, v10
	v_cmp_le_u32_e64 s[8:9], s61, v37
	v_add_u32_e32 v42, 2, v36
	v_cndmask_b32_e64 v46, 0, -1, vcc
	v_cmp_eq_u32_e32 vcc, s61, v10
	v_add3_u32 v10, v39, v45, v43
	v_cndmask_b32_e64 v39, 0, -1, s[8:9]
	v_cndmask_b32_e32 v41, v46, v41, vcc
	v_sub_u32_e32 v43, v52, v10
	v_sub_co_u32_e32 v38, vcc, v53, v38
	v_cmp_eq_u32_e64 s[8:9], s61, v37
	v_add_u32_e32 v44, 1, v36
	v_cmp_ne_u32_e64 s[18:19], 0, v41
	v_cndmask_b32_e64 v37, v39, v40, s[8:9]
	v_subb_co_u32_e64 v39, s[8:9], v43, v56, vcc
	v_subb_co_u32_e32 v10, vcc, v52, v10, vcc
	v_cmp_le_u32_e32 vcc, s60, v38
	v_subrev_co_u32_e64 v40, s[8:9], s60, v38
	s_nop 0
	v_cndmask_b32_e64 v43, 0, -1, vcc
	v_cmp_ne_u32_e32 vcc, 0, v37
	s_mov_b32 m0, s64
	v_lshl_add_u64 v[18:19], v[18:19], 0, s[62:63]
	v_cndmask_b32_e32 v37, v44, v42, vcc
	v_subbrev_co_u32_e64 v42, vcc, 0, v39, s[8:9]
	v_cmp_le_u32_e32 vcc, s60, v40
	v_cndmask_b32_e64 v36, v36, v37, s[18:19]
	v_cmp_le_u32_e64 s[18:19], s61, v42
	v_cndmask_b32_e64 v44, 0, -1, vcc
	v_subb_co_u32_e64 v39, vcc, v39, v56, s[8:9]
	v_subrev_co_u32_e32 v45, vcc, s60, v40
	v_cmp_le_u32_e64 s[8:9], s61, v10
	v_cndmask_b32_e64 v37, 0, -1, s[18:19]
	v_cmp_eq_u32_e64 s[18:19], s61, v42
	v_cndmask_b32_e64 v46, 0, -1, s[8:9]
	v_cmp_eq_u32_e64 s[8:9], s61, v10
	v_subbrev_co_u32_e32 v39, vcc, 0, v39, vcc
	v_cndmask_b32_e64 v37, v37, v44, s[18:19]
	v_cndmask_b32_e64 v41, v46, v43, s[8:9]
	v_xor_b32_e32 v36, v36, v25
	v_cmp_ne_u32_e32 vcc, 0, v37
	v_sub_u32_e32 v25, v36, v25
	v_cmp_ne_u32_e64 s[8:9], 0, v41
	v_cndmask_b32_e32 v36, v42, v39, vcc
	v_xad_u32 v25, v25, v24, v8
	v_cndmask_b32_e64 v10, v10, v36, s[8:9]
	v_cndmask_b32_e32 v36, v40, v45, vcc
	v_cndmask_b32_e64 v36, v38, v36, s[8:9]
	v_xor_b32_e32 v36, v36, v24
	v_xor_b32_e32 v10, v10, v24
	v_sub_co_u32_e32 v36, vcc, v36, v24
	v_mul_lo_u32 v25, v25, s56
	s_nop 0
	v_subb_co_u32_e32 v10, vcc, v10, v24, vcc
	v_ashrrev_i32_e32 v10, 31, v10
	v_and_b32_e32 v10, s56, v10
	v_add3_u32 v10, v10, v36, v25
	s_and_b64 vcc, s[14:15], s[4:5]
	v_cndmask_b32_e32 v10, v13, v10, vcc
	buffer_load_dwordx4 v10, s[36:39], 0 offen lds
	s_waitcnt vmcnt(7)
	ds_read_b128 v[36:39], v31
	ds_read_b128 v[40:43], v35
	s_waitcnt vmcnt(0)
	ds_read_b128 v[44:47], v15
	ds_read_b128 v[48:51], v15 offset:2048
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[36:39], v[44:47], v[4:7], v61, v57 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s44, s44, 1
	s_addc_u32 s45, s45, 0
	v_cmp_lt_i64_e32 vcc, s[44:45], v[22:23]
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[36:39], v[48:51], v[0:3], v61, v59 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 v[36:39], v30
	ds_read_b128 v[44:47], v30 offset:2048
	v_lshl_add_u64 v[20:21], v[20:21], 0, s[62:63]
	v_add_u32_e32 v12, 0x80, v12
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[4:7], v[40:43], v[36:39], v[4:7], v62, v58 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_lshl_add_u64 v[16:17], v[16:17], 0, s[50:51]
	s_add_i32 s66, s66, s54
	s_add_i32 s65, s65, s52
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], v[40:43], v[44:47], v[0:3], v62, v60 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_cbranch_vccnz .LBB0_15
.LBB0_16:
	s_nop 1
	v_bfe_u32 v8, v7, 16, 1
	s_movk_i32 s4, 0x7fff
	v_bfe_u32 v9, v6, 16, 1
	v_add3_u32 v8, v7, v8, s4
	v_bfe_u32 v10, v5, 16, 1
	v_add3_u32 v9, v6, v9, s4
	v_lshrrev_b32_e32 v8, 16, v8
	v_mov_b32_e32 v12, 0x7fc0
	v_cmp_o_f32_e32 vcc, v7, v7
	v_bfe_u32 v11, v4, 16, 1
	v_add3_u32 v10, v5, v10, s4
	v_lshrrev_b32_e32 v9, 16, v9
	v_cndmask_b32_e32 v7, v12, v8, vcc
	v_cmp_o_f32_e32 vcc, v6, v6
	v_add3_u32 v11, v4, v11, s4
	v_lshrrev_b32_e32 v10, 16, v10
	v_cndmask_b32_e32 v6, v12, v9, vcc
	v_cmp_o_f32_e32 vcc, v5, v5
	v_bfe_u32 v5, v3, 16, 1
	v_lshrrev_b32_e32 v11, 16, v11
	v_cndmask_b32_e32 v8, v12, v10, vcc
	v_cmp_o_f32_e32 vcc, v4, v4
	v_bfe_u32 v9, v2, 16, 1
	v_add3_u32 v5, v3, v5, s4
	v_cndmask_b32_e32 v4, v12, v11, vcc
	v_bfe_u32 v10, v1, 16, 1
	v_add3_u32 v9, v2, v9, s4
	v_lshrrev_b32_e32 v5, 16, v5
	v_cmp_o_f32_e32 vcc, v3, v3
	v_bfe_u32 v11, v0, 16, 1
	v_add3_u32 v10, v1, v10, s4
	v_lshrrev_b32_e32 v9, 16, v9
	v_cndmask_b32_e32 v3, v12, v5, vcc
	v_cmp_o_f32_e32 vcc, v2, v2
	v_add3_u32 v11, v0, v11, s4
	v_lshrrev_b32_e32 v10, 16, v10
	v_cndmask_b32_e32 v9, v12, v9, vcc
	v_cmp_o_f32_e32 vcc, v1, v1
	v_lshl_or_b32 v2, v33, 2, v34
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s6, s28, s31
	s_mul_hi_u32 s7, s28, s30
	v_lshrrev_b32_e32 v11, 16, v11
	v_cndmask_b32_e32 v10, v12, v10, vcc
	v_cmp_o_f32_e32 vcc, v0, v0
	v_or_b32_e32 v0, s30, v2
	v_mov_b32_e32 v1, s31
	s_add_i32 s6, s7, s6
	s_mul_i32 s7, s29, s30
	v_cndmask_b32_e32 v11, v12, v11, vcc
	v_cmp_gt_i64_e32 vcc, s[12:13], v[0:1]
	s_add_i32 s7, s6, s7
	s_mul_i32 s6, s28, s30
	s_and_b64 s[4:5], vcc, s[2:3]
	s_lshl_b64 s[6:7], s[6:7], 1
	s_add_u32 s8, s10, s6
	s_addc_u32 s9, s11, s7
	s_lshl_b64 s[6:7], s[58:59], 1
	s_add_u32 s16, s8, s6
	v_or_b32_e32 v12, s33, v32
	v_mul_lo_u32 v2, s28, v2
	s_addc_u32 s6, s9, s7
	s_and_b32 s7, s28, 0x3fff
	v_add_u32_e32 v1, v2, v12
	v_not_b32_e32 v13, -2.0
	s_lshl_b32 s7, s7, 16
	s_and_b32 s6, s6, 0xffff
	s_or_b32 s6, s7, s6
	v_cndmask_b32_e64 v1, v13, v1, s[4:5]
	s_or_b32 s17, s6, 2.0
	s_mov_b32 s19, 0x27000
	s_mov_b32 s18, 0x7ffffffd
	v_lshlrev_b32_e32 v1, 1, v1
	buffer_store_short v4, v1, s[16:19], 0 offen
	v_or_b32_e32 v4, 1, v0
	v_mov_b32_e32 v5, s31
	v_cmp_gt_i64_e64 s[4:5], s[12:13], v[4:5]
	v_add_u32_e32 v14, s28, v2
	v_add_u32_e32 v1, v14, v12
	s_and_b64 s[6:7], s[4:5], s[2:3]
	v_cndmask_b32_e64 v1, v13, v1, s[6:7]
	v_or_b32_e32 v4, 2, v0
	v_lshlrev_b32_e32 v1, 1, v1
	v_cmp_gt_i64_e64 s[6:7], s[12:13], v[4:5]
	v_lshl_add_u32 v4, s28, 1, v2
	buffer_store_short v8, v1, s[16:19], 0 offen
	v_add_u32_e32 v1, v4, v12
	s_and_b64 s[8:9], s[6:7], s[2:3]
	v_cndmask_b32_e64 v1, v13, v1, s[8:9]
	v_lshlrev_b32_e32 v1, 1, v1
	buffer_store_short v6, v1, s[16:19], 0 offen
	v_or_b32_e32 v0, 3, v0
	v_mov_b32_e32 v1, s31
	v_cmp_gt_i64_e64 s[8:9], s[12:13], v[0:1]
	v_mad_u64_u32 v[0:1], s[10:11], s28, 3, v[2:3]
	v_add_u32_e32 v1, v0, v12
	s_and_b64 s[2:3], s[8:9], s[2:3]
	v_cndmask_b32_e64 v1, v13, v1, s[2:3]
	v_lshlrev_b32_e32 v1, 1, v1
	buffer_store_short v7, v1, s[16:19], 0 offen
	v_or_b32_e32 v1, 16, v12
	v_add_u32_e32 v2, v2, v1
	s_and_b64 vcc, vcc, s[0:1]
	v_cndmask_b32_e32 v2, v13, v2, vcc
	v_lshlrev_b32_e32 v2, 1, v2
	buffer_store_short v11, v2, s[16:19], 0 offen
	v_add_u32_e32 v2, v14, v1
	s_and_b64 vcc, s[4:5], s[0:1]
	v_cndmask_b32_e32 v2, v13, v2, vcc
	v_lshlrev_b32_e32 v2, 1, v2
	buffer_store_short v10, v2, s[16:19], 0 offen
	v_add_u32_e32 v2, v4, v1
	s_and_b64 vcc, s[6:7], s[0:1]
	v_cndmask_b32_e32 v2, v13, v2, vcc
	v_add_u32_e32 v0, v0, v1
	s_and_b64 vcc, s[8:9], s[0:1]
	v_cndmask_b32_e32 v0, v13, v0, vcc
	v_lshlrev_b32_e32 v2, 1, v2
	v_lshlrev_b32_e32 v0, 1, v0
	buffer_store_short v9, v2, s[16:19], 0 offen
	buffer_store_short v3, v0, s[16:19], 0 offen
	s_endpgm
.LBB0_17:
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v2, v3
	v_mov_b32_e32 v1, v3
	v_mov_b32_e32 v0, v3
	v_mov_b32_e32 v11, v3
	v_mov_b32_e32 v10, v3
	v_mov_b32_e32 v9, v3
	v_mov_b32_e32 v8, v3
	s_andn2_b64 vcc, exec, s[22:23]
	s_cbranch_vccnz .LBB0_11
	s_branch .LBB0_10
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mxfp4_dynamic_gemm_64x64x256
		.amdhsa_group_segment_fixed_size 49152
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 104
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
		.amdhsa_next_free_vgpr 82
		.amdhsa_next_free_sgpr 95
		.amdhsa_accum_offset 84
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
	.size	wave_mxfp4_dynamic_gemm_64x64x256, .Lfunc_end0-wave_mxfp4_dynamic_gemm_64x64x256

	.set wave_mxfp4_dynamic_gemm_64x64x256.num_vgpr, 82
	.set wave_mxfp4_dynamic_gemm_64x64x256.num_agpr, 0
	.set wave_mxfp4_dynamic_gemm_64x64x256.numbered_sgpr, 95
	.set wave_mxfp4_dynamic_gemm_64x64x256.num_named_barrier, 0
	.set wave_mxfp4_dynamic_gemm_64x64x256.private_seg_size, 0
	.set wave_mxfp4_dynamic_gemm_64x64x256.uses_vcc, 1
	.set wave_mxfp4_dynamic_gemm_64x64x256.uses_flat_scratch, 0
	.set wave_mxfp4_dynamic_gemm_64x64x256.has_dyn_sized_stack, 0
	.set wave_mxfp4_dynamic_gemm_64x64x256.has_recursion, 0
	.set wave_mxfp4_dynamic_gemm_64x64x256.has_indirect_call, 0
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
      - .offset:         80
        .size:           4
        .value_kind:     by_value
      - .offset:         84
        .size:           4
        .value_kind:     by_value
      - .offset:         88
        .size:           4
        .value_kind:     by_value
      - .offset:         92
        .size:           4
        .value_kind:     by_value
      - .offset:         96
        .size:           4
        .value_kind:     by_value
      - .offset:         100
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 49152
    .kernarg_segment_align: 8
    .kernarg_segment_size: 104
    .max_flat_workgroup_size: 512
    .name:           wave_mxfp4_dynamic_gemm_64x64x256
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 2
      - 1
    .sgpr_count:     101
    .sgpr_spill_count: 0
    .symbol:         wave_mxfp4_dynamic_gemm_64x64x256.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     82
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 0
...

	.end_amdgpu_metadata
