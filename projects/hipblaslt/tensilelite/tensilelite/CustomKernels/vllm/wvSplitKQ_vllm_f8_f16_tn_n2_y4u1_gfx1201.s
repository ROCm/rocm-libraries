	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201 ; -- Begin function wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201
	.globl	wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201
	.p2align	8
	.type	wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201,@function
wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201: ; @wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201
; %bb.0:
	s_clause 0x1
	s_load_b64 s[12:13], s[0:1], 0x0
	s_load_b64 s[20:21], s[0:1], 0x10
	v_and_b32_e32 v34, 0x3ff, v0
	v_bfe_u32 v2, v0, 10, 10
	s_mov_b32 s23, 0
	s_mov_b32 s2, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v58, 4, v34
	v_lshl_add_u32 v3, v2, 9, v58
	s_wait_kmcnt 0x0
	s_cmp_lt_u32 s12, 0x10000
	s_mov_b32 s22, s12
	s_cselect_b32 s29, s12, 0x10000
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cmpx_gt_u32_e64 s29, v3
	s_cbranch_execz .LBB0_9
; %bb.1:
	global_load_b128 v[4:7], v3, s[20:21]
	v_add_nc_u32_e32 v0, 0x2000, v3
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7]
	v_cmpx_gt_u32_e64 s29, v0
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB0_9
; %bb.2:
	v_add_co_u32 v0, s3, s20, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v1, null, s21, 0, s3
	v_add_nc_u32_e32 v8, 0x4000, v3
	s_mov_b32 s3, exec_lo
	global_load_b128 v[4:7], v[0:1], off offset:8192
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:8192
	v_cmpx_gt_u32_e64 s29, v8
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB0_9
; %bb.3:
	global_load_b128 v[4:7], v[0:1], off offset:16384
	v_add_nc_u32_e32 v8, 0x6000, v3
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:16384
	v_cmpx_gt_u32_e64 s29, v8
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB0_9
; %bb.4:
	global_load_b128 v[4:7], v[0:1], off offset:24576
	v_add_nc_u32_e32 v8, 0x8000, v3
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:24576
	v_cmpx_gt_u32_e64 s29, v8
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB0_9
; %bb.5:
	global_load_b128 v[4:7], v[0:1], off offset:32768
	v_add_nc_u32_e32 v8, 0xa000, v3
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:32768
	v_cmpx_gt_u32_e64 s29, v8
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB0_9
; %bb.6:
	global_load_b128 v[4:7], v[0:1], off offset:40960
	v_add_nc_u32_e32 v8, 0xc000, v3
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:40960
	v_cmpx_gt_u32_e64 s29, v8
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB0_9
; %bb.7:
	global_load_b128 v[4:7], v[0:1], off offset:49152
	v_add_nc_u32_e32 v8, 0xe000, v3
	s_mov_b32 s3, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:49152
	v_cmpx_gt_u32_e64 s29, v8
	s_wait_alu 0xfffe
	s_xor_b32 s3, exec_lo, s3
	s_cbranch_execz .LBB0_9
; %bb.8:
	global_load_b128 v[4:7], v[0:1], off offset:57344
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:57344
.LBB0_9:
	s_or_b32 exec_lo, exec_lo, s2
	s_clause 0x3
	s_load_b64 s[14:15], s[0:1], 0x8
	s_load_b64 s[16:17], s[0:1], 0x20
	s_load_b64 s[18:19], s[0:1], 0x58
	s_load_b128 s[4:7], s[0:1], 0x48
	s_sub_nc_u64 s[2:3], 0x10000, s[22:23]
	s_mov_b32 s8, exec_lo
	s_wait_alu 0xfffe
	v_cmp_lt_u64_e64 s3, s[2:3], s[22:23]
	s_and_b32 s3, s3, exec_lo
	s_cselect_b32 s2, s2, s22
	s_cmp_lt_u32 s12, 0x10000
	s_wait_alu 0xfffe
	s_cselect_b32 s30, s2, 0
	s_wait_kmcnt 0x0
	s_mov_b32 s22, s17
	v_cmpx_gt_u32_e64 s30, v3
	s_cbranch_execz .LBB0_12
; %bb.10:
	s_wait_alu 0xfffe
	s_add_nc_u64 s[2:3], s[20:21], s[22:23]
	s_mov_b32 s9, 0
.LBB0_11:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[4:7], v3, s[2:3]
	v_add_nc_u32_e32 v0, s12, v3
	v_add_nc_u32_e32 v3, 0x2000, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s30, v3
	s_or_b32 s9, vcc_lo, s9
	s_wait_loadcnt 0x0
	ds_store_b128 v0, v[4:7]
	s_and_not1_b32 exec_lo, exec_lo, s9
	s_cbranch_execnz .LBB0_11
.LBB0_12:
	s_or_b32 exec_lo, exec_lo, s8
	s_load_b32 s2, s[0:1], 0x18
	s_add_co_i32 s17, s13, -1
	s_wait_kmcnt 0x0
	s_lshl_b32 s8, s2, 6
	s_delay_alu instid0(SALU_CYCLE_1)
	v_add_co_u32 v0, s3, s8, s17
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_mov_b32 s3, 4
	v_readfirstlane_b32 s9, v0
	s_cbranch_vccz .LBB0_14
; %bb.13:
	s_cvt_f32_u32 s3, s8
	s_sub_co_i32 s10, 0, s8
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v0, s3
	v_readfirstlane_b32 s3, v0
	s_mul_f32 s3, s3, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s3, s3
	s_wait_alu 0xfffe
	s_mul_i32 s10, s10, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s10, s3, s10
	s_add_co_i32 s3, s3, s10
	s_wait_alu 0xfffe
	s_mul_hi_u32 s3, s9, s3
	s_wait_alu 0xfffe
	s_mul_i32 s10, s3, s8
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s9, s9, s10
	s_add_co_i32 s10, s3, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s11, s9, s8
	s_cmp_ge_u32 s9, s8
	s_cselect_b32 s3, s10, s3
	s_cselect_b32 s9, s11, s9
	s_wait_alu 0xfffe
	s_add_co_i32 s10, s3, 1
	s_cmp_ge_u32 s9, s8
	s_cselect_b32 s3, s10, s3
	s_wait_alu 0xfffe
	s_mul_i32 s3, s2, s3
	s_wait_alu 0xfffe
	s_lshl_b32 s3, s3, 2
	s_wait_alu 0xfffe
	s_cvt_f32_u32 s8, s3
	s_sub_co_i32 s9, 0, s3
	s_add_co_i32 s10, s3, s17
	s_wait_alu 0xfffe
	v_rcp_iflag_f32_e32 v0, s8
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_2)
	v_readfirstlane_b32 s8, v0
	s_mul_f32 s8, s8, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_cvt_u32_f32 s8, s8
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2)
	s_mul_i32 s9, s9, s8
	s_wait_alu 0xfffe
	s_mul_hi_u32 s9, s8, s9
	s_wait_alu 0xfffe
	s_add_co_i32 s8, s8, s9
	s_wait_alu 0xfffe
	s_mul_hi_u32 s8, s10, s8
	s_wait_alu 0xfffe
	s_mul_i32 s9, s8, s3
	s_wait_alu 0xfffe
	s_sub_co_i32 s9, s10, s9
	s_add_co_i32 s10, s8, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s11, s9, s3
	s_cmp_ge_u32 s9, s3
	s_cselect_b32 s8, s10, s8
	s_cselect_b32 s9, s11, s9
	s_wait_alu 0xfffe
	s_add_co_i32 s10, s8, 1
	s_cmp_ge_u32 s9, s3
	s_cselect_b32 s3, s10, s8
	s_wait_alu 0xfffe
	s_max_u32 s3, s3, 4
.LBB0_14:
	s_mov_b32 s23, 1.0
	s_cmp_eq_u64 s[4:5], 0
	s_mov_b32 s24, 1.0
	s_cbranch_scc1 .LBB0_16
; %bb.15:
	s_load_b32 s24, s[4:5], 0x0
.LBB0_16:
	s_cmp_eq_u64 s[6:7], 0
	s_cbranch_scc1 .LBB0_18
; %bb.17:
	s_load_b32 s23, s[6:7], 0x0
.LBB0_18:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_mov_b32 s4, exec_lo
	s_wait_alu 0xfffe
	v_cmpx_gt_u32_e64 s3, v2
	s_cbranch_execz .LBB0_109
; %bb.19:
	s_load_b256 s[4:11], s[0:1], 0x28
	s_cmp_eq_u32 s30, s12
	s_mul_i32 s2, s2, s3
	s_cselect_b32 s0, -1, 0
	s_cmp_eq_u32 s12, s29
	s_cselect_b32 s1, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_1)
	s_and_b32 s1, s1, s0
	s_mul_i32 s0, s3, ttmp9
	s_and_not1_b32 vcc_lo, exec_lo, s1
	v_add_lshl_u32 v48, s0, v2, 2
	s_mov_b32 s1, -1
	v_cmp_gt_u32_e64 s0, s13, v48
	s_wait_kmcnt 0x0
	s_mul_f32 s3, s8, s24
	s_wait_alu 0xfffe
	s_lshl_b32 s8, s2, 2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mul_f32 s28, s3, s23
	s_cbranch_vccz .LBB0_65
; %bb.20:
	s_and_saveexec_b32 s23, s0
	s_cbranch_execz .LBB0_64
; %bb.21:
	s_cmp_eq_u64 s[18:19], 0
	v_mbcnt_lo_u32_b32 v35, -1, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[18:19], 0
	s_mov_b32 s3, 0
	s_cselect_b32 s31, -1, 0
	s_cmp_lg_u32 s12, 0
	v_cmp_eq_u32_e64 s1, 31, v34
	s_cselect_b32 s33, -1, 0
	s_add_co_i32 s34, s12, -16
	s_cmp_neq_f32 s9, 0
	v_dual_mov_b32 v25, 0 :: v_dual_add_nc_u32 v36, s12, v58
	v_xor_b32_e32 v37, 16, v35
	v_mov_b32_e32 v24, v48
	s_mov_b32 s2, s10
	s_mov_b32 s26, s11
	s_wait_alu 0xfffe
	s_mov_b32 s27, s3
	s_cselect_b32 s35, -1, 0
	s_lshl_b64 s[24:25], s[2:3], 1
	s_lshl_b64 s[26:27], s[26:27], 1
	s_branch .LBB0_24
.LBB0_22:                               ;   in Loop: Header=BB0_24 Depth=1
	v_cvt_f32_f16_e32 v0, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v4, v0
	v_cndmask_b32_e64 v0, v0, v4, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v0, v0
	global_store_b16 v[2:3], v0, off offset:6
.LBB0_23:                               ;   in Loop: Header=BB0_24 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s36
	v_add_nc_u32_e32 v24, s8, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s13, v24
	s_or_b32 s3, vcc_lo, s3
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s3
	s_cbranch_execz .LBB0_64
.LBB0_24:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_30 Depth 2
	v_or_b32_e32 v41, 1, v24
	v_min_u32_e32 v0, s17, v24
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v39, 0
	v_mov_b32_e32 v40, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_min_u32_e32 v2, s17, v41
	v_mov_b32_e32 v38, 0
	s_and_not1_b32 vcc_lo, exec_lo, s31
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_26
; %bb.25:                               ;   in Loop: Header=BB0_24 Depth=1
	s_wait_dscnt 0x7
	v_mov_b32_e32 v1, v25
	v_mov_b32_e32 v3, v25
	v_or_b32_e32 v6, 2, v24
	s_wait_dscnt 0x1
	v_mov_b32_e32 v7, v25
	v_mov_b32_e32 v11, v25
	s_wait_dscnt 0x0
	v_lshlrev_b64_e32 v[4:5], 1, v[0:1]
	v_or_b32_e32 v1, 3, v24
	v_lshlrev_b64_e32 v[8:9], 1, v[2:3]
	v_min_u32_e32 v6, s17, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_min_u32_e32 v10, s17, v1
	v_add_co_u32 v3, vcc_lo, s18, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s19, v5, vcc_lo
	v_lshlrev_b64_e32 v[5:6], 1, v[6:7]
	v_add_co_u32 v7, vcc_lo, s18, v8
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, s19, v9, vcc_lo
	v_lshlrev_b64_e32 v[9:10], 1, v[10:11]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_co_u32 v5, vcc_lo, s18, v5
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, s19, v6, vcc_lo
	v_add_co_u32 v9, vcc_lo, s18, v9
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v10, null, s19, v10, vcc_lo
	s_clause 0x3
	global_load_u16 v42, v[3:4], off
	global_load_u16 v40, v[7:8], off
	global_load_u16 v39, v[5:6], off
	global_load_u16 v38, v[9:10], off
.LBB0_26:                               ;   in Loop: Header=BB0_24 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s33
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_39
; %bb.27:                               ;   in Loop: Header=BB0_24 Depth=1
	s_wait_dscnt 0x7
	v_or_b32_e32 v1, 2, v24
	v_or_b32_e32 v3, 3, v24
	v_mad_co_u64_u32 v[26:27], null, v0, s16, 0
	v_mad_co_u64_u32 v[28:29], null, v2, s16, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_min_u32_e32 v0, s17, v1
	v_min_u32_e32 v1, s17, v3
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v43, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mad_co_u64_u32 v[30:31], null, v0, s16, 0
	v_mad_co_u64_u32 v[32:33], null, v1, s16, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v50, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v51, 0
	s_mov_b32 s2, 0
	s_branch .LBB0_30
.LBB0_28:                               ;   in Loop: Header=BB0_30 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s37
.LBB0_29:                               ;   in Loop: Header=BB0_30 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s36
	s_wait_loadcnt_dscnt 0x0
	v_dot4_f32_fp8_fp8 v51, v16, v0, v51
	v_dot4_f32_fp8_fp8 v50, v16, v4, v50
	v_dot4_f32_fp8_fp8 v49, v16, v8, v49
	v_dot4_f32_fp8_fp8 v16, v16, v12, v47
	v_dot4_f32_fp8_fp8 v0, v20, v0, v46
	v_dot4_f32_fp8_fp8 v4, v20, v4, v45
	v_dot4_f32_fp8_fp8 v8, v20, v8, v44
	v_dot4_f32_fp8_fp8 v12, v20, v12, v43
	v_dot4_f32_fp8_fp8 v51, v17, v1, v51
	v_dot4_f32_fp8_fp8 v50, v17, v5, v50
	v_dot4_f32_fp8_fp8 v49, v17, v9, v49
	v_dot4_f32_fp8_fp8 v16, v17, v13, v16
	v_dot4_f32_fp8_fp8 v0, v21, v1, v0
	v_dot4_f32_fp8_fp8 v1, v21, v5, v4
	v_dot4_f32_fp8_fp8 v4, v21, v9, v8
	v_dot4_f32_fp8_fp8 v5, v21, v13, v12
	v_dot4_f32_fp8_fp8 v46, v18, v2, v51
	v_dot4_f32_fp8_fp8 v47, v18, v6, v50
	v_dot4_f32_fp8_fp8 v49, v18, v10, v49
	v_dot4_f32_fp8_fp8 v8, v18, v14, v16
	v_dot4_f32_fp8_fp8 v0, v22, v2, v0
	v_dot4_f32_fp8_fp8 v1, v22, v6, v1
	v_dot4_f32_fp8_fp8 v2, v22, v10, v4
	v_dot4_f32_fp8_fp8 v4, v22, v14, v5
	v_dot4_f32_fp8_fp8 v51, v19, v3, v46
	v_dot4_f32_fp8_fp8 v50, v19, v7, v47
	v_dot4_f32_fp8_fp8 v49, v19, v11, v49
	v_dot4_f32_fp8_fp8 v47, v19, v15, v8
	v_dot4_f32_fp8_fp8 v46, v23, v3, v0
	v_dot4_f32_fp8_fp8 v45, v23, v7, v1
	v_dot4_f32_fp8_fp8 v44, v23, v11, v2
	v_dot4_f32_fp8_fp8 v43, v23, v15, v4
	s_addk_co_i32 s2, 0x200
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s2, s12
	s_cbranch_scc1 .LBB0_40
.LBB0_30:                               ;   Parent Loop BB0_24 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_dual_mov_b32 v23, 0 :: v_dual_add_nc_u32 v52, s2, v58
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v19, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_min_u32_e32 v0, s34, v52
	s_wait_dscnt 0x6
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v17, 0
	s_wait_dscnt 0x5
	v_mov_b32_e32 v16, 0
	v_add_co_u32 v2, s36, s14, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v3, null, s15, 0, s36
	s_mov_b32 s36, exec_lo
	v_add_co_u32 v0, vcc_lo, v2, v26
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v3, v27, vcc_lo
	v_add_co_u32 v4, vcc_lo, v2, v28
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v3, v29, vcc_lo
	v_add_co_u32 v8, vcc_lo, v2, v30
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, v3, v31, vcc_lo
	v_add_co_u32 v12, vcc_lo, v2, v32
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v13, null, v3, v33, vcc_lo
	s_clause 0x3
	global_load_b128 v[0:3], v[0:1], off
	global_load_b128 v[4:7], v[4:5], off
	global_load_b128 v[8:11], v[8:9], off
	global_load_b128 v[12:15], v[12:13], off
	v_cmpx_gt_u32_e64 s12, v52
	s_cbranch_execz .LBB0_29
; %bb.31:                               ;   in Loop: Header=BB0_30 Depth=2
	v_add_co_u32 v20, s37, s20, v52
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v21, null, s21, 0, s37
	s_mov_b32 s37, exec_lo
                                        ; implicit-def: $vgpr19
	v_cmpx_le_u32_e64 s29, v52
	s_wait_alu 0xfffe
	s_xor_b32 s37, exec_lo, s37
	s_cbranch_execz .LBB0_33
; %bb.32:                               ;   in Loop: Header=BB0_30 Depth=2
	global_load_b128 v[16:19], v[20:21], off
.LBB0_33:                               ;   in Loop: Header=BB0_30 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s37, s37
	s_cbranch_execz .LBB0_35
; %bb.34:                               ;   in Loop: Header=BB0_30 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[16:19], v52
.LBB0_35:                               ;   in Loop: Header=BB0_30 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s37
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s37, exec_lo
                                        ; implicit-def: $vgpr23
	v_cmpx_le_u32_e64 s30, v52
	s_wait_alu 0xfffe
	s_xor_b32 s37, exec_lo, s37
	s_cbranch_execz .LBB0_37
; %bb.36:                               ;   in Loop: Header=BB0_30 Depth=2
	v_add_co_u32 v20, vcc_lo, v20, s22
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v21, null, 0, v21, vcc_lo
	global_load_b128 v[20:23], v[20:21], off
.LBB0_37:                               ;   in Loop: Header=BB0_30 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s37, s37
	s_cbranch_execz .LBB0_28
; %bb.38:                               ;   in Loop: Header=BB0_30 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v20, s2, v36
	ds_load_b128 v[20:23], v20
	s_branch .LBB0_28
.LBB0_39:                               ;   in Loop: Header=BB0_24 Depth=1
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v46, 0
	v_mov_b32_e32 v43, 0
	v_mov_b32_e32 v45, 0
.LBB0_40:                               ;   in Loop: Header=BB0_24 Depth=1
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_dpp v2, v49, v49 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v51, v51 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x7
	v_add_f32_dpp v1, v50, v50 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v47, v47 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x0
	v_add_f32_dpp v5, v45, v45 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v44, v44 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v43, v43 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v37
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v46, v46 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v35, v37, vcc_lo
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshlrev_b32_e32 v3, 2, v3
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v14, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v7, v7 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v1, v3, v0
	ds_bpermute_b32 v18, v3, v17
	ds_bpermute_b32 v16, v3, v12
	ds_bpermute_b32 v15, v3, v14
	ds_bpermute_b32 v11, v3, v10
	ds_bpermute_b32 v9, v3, v8
	ds_bpermute_b32 v7, v3, v6
	ds_bpermute_b32 v5, v3, v4
	s_wait_loadcnt 0x3
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x2
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s36, s1
	s_cbranch_execz .LBB0_23
; %bb.41:                               ;   in Loop: Header=BB0_24 Depth=1
	v_lshlrev_b64_e32 v[2:3], 1, v[24:25]
	s_wait_dscnt 0x7
	v_add_f32_e32 v13, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v19, s28, v13
	v_add_co_u32 v0, vcc_lo, s4, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s5, v3, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s35
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_43
; %bb.42:                               ;   in Loop: Header=BB0_24 Depth=1
	global_load_u16 v13, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v19, s9, v13, v19 op_sel_hi:[0,1,0]
.LBB0_43:                               ;   in Loop: Header=BB0_24 Depth=1
	v_cvt_f32_f16_e32 v13, v42
	v_add_co_u32 v2, vcc_lo, s6, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v20, v19, v13
	v_cmp_gt_u32_e64 s2, s13, v41
	v_cndmask_b32_e64 v19, v20, v19, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v19, v19
	global_store_b16 v[2:3], v19, off
	s_and_saveexec_b32 s37, s2
	s_cbranch_execz .LBB0_53
; %bb.44:                               ;   in Loop: Header=BB0_24 Depth=1
	s_wait_dscnt 0x6
	v_add_f32_e32 v17, v17, v18
	s_and_not1_b32 vcc_lo, exec_lo, s35
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v17, s28, v17
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_46
; %bb.45:                               ;   in Loop: Header=BB0_24 Depth=1
	global_load_u16 v18, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v17, s9, v18, v17 op_sel_hi:[0,1,0]
.LBB0_46:                               ;   in Loop: Header=BB0_24 Depth=1
	v_cvt_f32_f16_e32 v18, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v18, v17, v18
	v_cndmask_b32_e64 v17, v18, v17, s0
	v_or_b32_e32 v18, 2, v24
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v17, v17
	v_cmp_gt_u32_e32 vcc_lo, s13, v18
	global_store_b16 v[2:3], v17, off offset:2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_53
; %bb.47:                               ;   in Loop: Header=BB0_24 Depth=1
	s_wait_dscnt 0x5
	v_add_f32_e32 v12, v12, v16
	s_and_not1_b32 vcc_lo, exec_lo, s35
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v12, s28, v12
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_49
; %bb.48:                               ;   in Loop: Header=BB0_24 Depth=1
	global_load_u16 v16, v[0:1], off offset:4
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v12, s9, v16, v12 op_sel_hi:[0,1,0]
.LBB0_49:                               ;   in Loop: Header=BB0_24 Depth=1
	v_cvt_f32_f16_e32 v16, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v16, v12, v16
	v_cndmask_b32_e64 v12, v16, v12, s0
	v_or_b32_e32 v16, 3, v24
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v12, v12
	v_cmp_gt_u32_e32 vcc_lo, s13, v16
	global_store_b16 v[2:3], v12, off offset:4
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_53
; %bb.50:                               ;   in Loop: Header=BB0_24 Depth=1
	s_wait_dscnt 0x4
	v_add_f32_e32 v12, v14, v15
	s_and_not1_b32 vcc_lo, exec_lo, s35
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v12, s28, v12
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_52
; %bb.51:                               ;   in Loop: Header=BB0_24 Depth=1
	global_load_u16 v14, v[0:1], off offset:6
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v12, s9, v14, v12 op_sel_hi:[0,1,0]
.LBB0_52:                               ;   in Loop: Header=BB0_24 Depth=1
	v_cvt_f32_f16_e32 v14, v38
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v14, v12, v14
	v_cndmask_b32_e64 v12, v14, v12, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v12, v12
	global_store_b16 v[2:3], v12, off offset:6
.LBB0_53:                               ;   in Loop: Header=BB0_24 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s37
	s_wait_dscnt 0x3
	v_add_f32_e32 v10, v10, v11
	v_add_co_u32 v0, vcc_lo, v0, s24
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s25, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v10, s28, v10
	s_and_not1_b32 vcc_lo, exec_lo, s35
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_55
; %bb.54:                               ;   in Loop: Header=BB0_24 Depth=1
	global_load_u16 v11, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v10, s9, v11, v10 op_sel_hi:[0,1,0]
.LBB0_55:                               ;   in Loop: Header=BB0_24 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v11, v10, v13
	v_add_co_u32 v2, vcc_lo, v2, s26
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s27, v3, vcc_lo
	v_cndmask_b32_e64 v10, v11, v10, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v10, v10
	global_store_b16 v[2:3], v10, off
	s_and_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB0_23
; %bb.56:                               ;   in Loop: Header=BB0_24 Depth=1
	s_wait_dscnt 0x2
	v_add_f32_e32 v8, v8, v9
	s_and_not1_b32 vcc_lo, exec_lo, s35
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v8, s28, v8
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_58
; %bb.57:                               ;   in Loop: Header=BB0_24 Depth=1
	global_load_u16 v9, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v8, s9, v9, v8 op_sel_hi:[0,1,0]
.LBB0_58:                               ;   in Loop: Header=BB0_24 Depth=1
	v_cvt_f32_f16_e32 v9, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v8, v9
	v_cndmask_b32_e64 v8, v9, v8, s0
	v_or_b32_e32 v9, 2, v24
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v8, v8
	v_cmp_gt_u32_e32 vcc_lo, s13, v9
	global_store_b16 v[2:3], v8, off offset:2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_23
; %bb.59:                               ;   in Loop: Header=BB0_24 Depth=1
	s_wait_dscnt 0x1
	v_add_f32_e32 v6, v6, v7
	s_and_not1_b32 vcc_lo, exec_lo, s35
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v6, s28, v6
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_61
; %bb.60:                               ;   in Loop: Header=BB0_24 Depth=1
	global_load_u16 v7, v[0:1], off offset:4
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v6, s9, v7, v6 op_sel_hi:[0,1,0]
.LBB0_61:                               ;   in Loop: Header=BB0_24 Depth=1
	v_cvt_f32_f16_e32 v7, v39
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v7, v6, v7
	v_cndmask_b32_e64 v6, v7, v6, s0
	v_or_b32_e32 v7, 3, v24
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v6, v6
	v_cmp_gt_u32_e32 vcc_lo, s13, v7
	global_store_b16 v[2:3], v6, off offset:4
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_23
; %bb.62:                               ;   in Loop: Header=BB0_24 Depth=1
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	s_and_not1_b32 vcc_lo, exec_lo, s35
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v4, s28, v4
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_22
; %bb.63:                               ;   in Loop: Header=BB0_24 Depth=1
	global_load_u16 v0, v[0:1], off offset:6
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v4, s9, v0, v4 op_sel_hi:[0,1,0]
	s_branch .LBB0_22
.LBB0_64:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s23
	s_mov_b32 s1, 0
.LBB0_65:
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_109
; %bb.66:
	v_cmp_gt_u32_e32 vcc_lo, s13, v48
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_109
; %bb.67:
	v_sub_co_u32 v0, s1, s12, 1
	s_cmp_eq_u64 s[18:19], 0
	v_mbcnt_lo_u32_b32 v60, -1, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[18:19], 0
	v_readfirstlane_b32 s2, v0
	s_cselect_b32 s24, -1, 0
	s_xor_b32 s25, s1, -1
	s_add_co_i32 s26, s12, -16
	s_cmp_neq_f32 s9, 0
	v_dual_mov_b32 v49, 0 :: v_dual_and_b32 v0, 0x200, v0
	s_mov_b32 s21, 0
	s_cselect_b32 s27, -1, 0
	s_lshr_b32 s2, s2, 9
	v_cmp_eq_u32_e64 s1, 31, v34
	s_wait_alu 0xfffe
	s_add_co_i32 s3, s2, 1
	v_add_nc_u32_e32 v59, s12, v58
	v_cmp_eq_u32_e64 s2, 0, v0
	v_xor_b32_e32 v61, 16, v60
	s_mov_b32 s20, s10
	s_mov_b32 s22, s11
	s_mov_b32 s23, s21
	s_cmp_gt_u32 s12, 0x200
	s_cselect_b32 s29, -1, 0
	s_wait_alu 0xfffe
	s_and_b32 s30, s3, 0xfffffe
	s_lshl_b64 s[10:11], s[20:21], 1
	s_lshl_b64 s[22:23], s[22:23], 1
	s_branch .LBB0_70
.LBB0_68:                               ;   in Loop: Header=BB0_70 Depth=1
	v_cvt_f32_f16_e32 v0, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v4, v0
	v_cndmask_b32_e64 v0, v0, v4, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v0, v0
	global_store_b16 v[2:3], v0, off offset:6
.LBB0_69:                               ;   in Loop: Header=BB0_70 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s20
	v_add_nc_u32_e32 v48, s8, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s13, v48
	s_or_b32 s21, vcc_lo, s21
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s21
	s_cbranch_execz .LBB0_109
.LBB0_70:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_76 Depth 2
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v64, 0
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v62, 0
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_72
; %bb.71:                               ;   in Loop: Header=BB0_70 Depth=1
	v_or_b32_e32 v2, 1, v48
	v_or_b32_e32 v4, 2, v48
	v_min_u32_e32 v0, s17, v48
	s_wait_dscnt 0x7
	v_mov_b32_e32 v1, v49
	v_or_b32_e32 v6, 3, v48
	v_mov_b32_e32 v3, v49
	v_min_u32_e32 v2, s17, v2
	s_wait_dscnt 0x0
	v_mov_b32_e32 v5, v49
	v_min_u32_e32 v4, s17, v4
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]
	v_min_u32_e32 v6, s17, v6
	v_mov_b32_e32 v7, v49
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]
	v_add_co_u32 v0, vcc_lo, s18, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b64_e32 v[6:7], 1, v[6:7]
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s19, v1, vcc_lo
	v_add_co_u32 v2, vcc_lo, s18, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s19, v3, vcc_lo
	v_add_co_u32 v4, vcc_lo, s18, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, s19, v5, vcc_lo
	v_add_co_u32 v6, vcc_lo, s18, v6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v7, null, s19, v7, vcc_lo
	s_clause 0x3
	global_load_u16 v65, v[0:1], off
	global_load_u16 v64, v[2:3], off
	global_load_u16 v63, v[4:5], off
	global_load_u16 v62, v[6:7], off
.LBB0_72:                               ;   in Loop: Header=BB0_70 Depth=1
	v_dual_mov_b32 v72, 0 :: v_dual_mov_b32 v69, 0
	v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v73, 0
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v67, 0
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v71, 0
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_85
; %bb.73:                               ;   in Loop: Header=BB0_70 Depth=1
	v_or_b32_e32 v0, 1, v48
	s_wait_dscnt 0x7
	v_min_u32_e32 v1, s17, v48
	v_or_b32_e32 v2, 2, v48
	v_or_b32_e32 v3, 3, v48
	s_and_not1_b32 vcc_lo, exec_lo, s29
	v_min_u32_e32 v0, s17, v0
	v_mad_co_u64_u32 v[56:57], null, v1, s16, 0
	v_min_u32_e32 v1, s17, v2
	v_min_u32_e32 v2, s17, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[54:55], null, v0, s16, 0
	v_mad_co_u64_u32 v[52:53], null, v1, s16, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[50:51], null, v2, s16, 0
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_80
; %bb.74:                               ;   in Loop: Header=BB0_70 Depth=1
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v72, 0
	s_mov_b32 s3, 0
	s_mov_b32 s20, s30
	s_branch .LBB0_76
.LBB0_75:                               ;   in Loop: Header=BB0_76 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s31
	s_wait_loadcnt_dscnt 0x701
	v_dot4_f32_fp8_fp8 v72, v36, v12, v72
	s_wait_loadcnt 0x6
	v_dot4_f32_fp8_fp8 v70, v36, v16, v70
	s_wait_loadcnt 0x5
	v_dot4_f32_fp8_fp8 v69, v36, v0, v69
	s_wait_loadcnt 0x4
	v_dot4_f32_fp8_fp8 v36, v36, v4, v73
	s_wait_dscnt 0x0
	v_dot4_f32_fp8_fp8 v12, v8, v12, v71
	v_dot4_f32_fp8_fp8 v71, v37, v13, v72
	v_dot4_f32_fp8_fp8 v16, v8, v16, v68
	v_dot4_f32_fp8_fp8 v68, v37, v17, v70
	v_dot4_f32_fp8_fp8 v69, v37, v1, v69
	v_dot4_f32_fp8_fp8 v36, v37, v5, v36
	v_dot4_f32_fp8_fp8 v37, v38, v14, v71
	v_dot4_f32_fp8_fp8 v12, v9, v13, v12
	v_dot4_f32_fp8_fp8 v13, v9, v17, v16
	v_dot4_f32_fp8_fp8 v16, v38, v18, v68
	v_dot4_f32_fp8_fp8 v0, v8, v0, v67
	v_dot4_f32_fp8_fp8 v37, v39, v15, v37
	v_dot4_f32_fp8_fp8 v12, v10, v14, v12
	v_dot4_f32_fp8_fp8 v4, v8, v4, v66
	v_dot4_f32_fp8_fp8 v14, v39, v19, v16
	v_dot4_f32_fp8_fp8 v0, v9, v1, v0
	s_wait_loadcnt 0x3
	v_dot4_f32_fp8_fp8 v8, v40, v20, v37
	v_dot4_f32_fp8_fp8 v17, v38, v2, v69
	v_dot4_f32_fp8_fp8 v1, v9, v5, v4
	s_wait_loadcnt 0x2
	v_dot4_f32_fp8_fp8 v14, v40, v24, v14
	v_dot4_f32_fp8_fp8 v36, v38, v6, v36
	v_dot4_f32_fp8_fp8 v13, v10, v18, v13
	v_dot4_f32_fp8_fp8 v4, v41, v21, v8
	v_dot4_f32_fp8_fp8 v0, v10, v2, v0
	v_dot4_f32_fp8_fp8 v5, v41, v25, v14
	v_dot4_f32_fp8_fp8 v1, v10, v6, v1
	v_dot4_f32_fp8_fp8 v16, v39, v3, v17
	v_dot4_f32_fp8_fp8 v17, v39, v7, v36
	v_dot4_f32_fp8_fp8 v12, v11, v15, v12
	v_dot4_f32_fp8_fp8 v13, v11, v19, v13
	v_dot4_f32_fp8_fp8 v2, v42, v22, v4
	v_dot4_f32_fp8_fp8 v4, v42, v26, v5
	v_dot4_f32_fp8_fp8 v0, v11, v3, v0
	v_dot4_f32_fp8_fp8 v1, v11, v7, v1
	s_wait_loadcnt 0x1
	v_dot4_f32_fp8_fp8 v15, v40, v28, v16
	v_dot4_f32_fp8_fp8 v72, v43, v23, v2
	v_dot4_f32_fp8_fp8 v70, v43, v27, v4
	s_wait_loadcnt 0x0
	v_dot4_f32_fp8_fp8 v2, v40, v32, v17
	v_dot4_f32_fp8_fp8 v3, v44, v20, v12
	v_dot4_f32_fp8_fp8 v4, v44, v24, v13
	v_dot4_f32_fp8_fp8 v0, v44, v28, v0
	v_dot4_f32_fp8_fp8 v1, v44, v32, v1
	v_dot4_f32_fp8_fp8 v8, v41, v29, v15
	v_dot4_f32_fp8_fp8 v2, v41, v33, v2
	v_dot4_f32_fp8_fp8 v3, v45, v21, v3
	v_dot4_f32_fp8_fp8 v4, v45, v25, v4
	v_dot4_f32_fp8_fp8 v0, v45, v29, v0
	v_dot4_f32_fp8_fp8 v1, v45, v33, v1
	v_dot4_f32_fp8_fp8 v5, v42, v30, v8
	v_dot4_f32_fp8_fp8 v2, v42, v34, v2
	v_dot4_f32_fp8_fp8 v3, v46, v22, v3
	v_dot4_f32_fp8_fp8 v4, v46, v26, v4
	v_dot4_f32_fp8_fp8 v0, v46, v30, v0
	v_dot4_f32_fp8_fp8 v1, v46, v34, v1
	v_dot4_f32_fp8_fp8 v69, v43, v31, v5
	v_dot4_f32_fp8_fp8 v73, v43, v35, v2
	v_dot4_f32_fp8_fp8 v71, v47, v23, v3
	v_dot4_f32_fp8_fp8 v68, v47, v27, v4
	v_dot4_f32_fp8_fp8 v67, v47, v31, v0
	v_dot4_f32_fp8_fp8 v66, v47, v35, v1
	s_add_co_i32 s20, s20, -2
	s_addk_co_i32 s3, 0x400
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s20, 0
	s_cbranch_scc1 .LBB0_81
.LBB0_76:                               ;   Parent Loop BB0_70 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_dual_mov_b32 v37, 0 :: v_dual_add_nc_u32 v74, s3, v58
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v39, 0
	s_wait_dscnt 0x2
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v9, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v0, s26, v74
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v11, 0
	v_mov_b32_e32 v10, 0
	v_add_co_u32 v6, s31, s14, v0
	s_wait_dscnt 0x1
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v7, null, s15, 0, s31
	s_mov_b32 s31, exec_lo
	v_add_co_u32 v0, vcc_lo, v6, v56
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v7, v57, vcc_lo
	v_add_co_u32 v2, vcc_lo, v6, v54
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, v7, v55, vcc_lo
	v_add_co_u32 v4, vcc_lo, v6, v52
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v7, v53, vcc_lo
	v_add_co_u32 v6, vcc_lo, v6, v50
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v7, null, v7, v51, vcc_lo
	s_clause 0x3
	global_load_b128 v[12:15], v[0:1], off
	global_load_b128 v[16:19], v[2:3], off
	global_load_b128 v[0:3], v[4:5], off
	global_load_b128 v[4:7], v[6:7], off
	v_cmpx_gt_u32_e64 s12, v74
	s_cbranch_execz .LBB0_78
; %bb.77:                               ;   in Loop: Header=BB0_76 Depth=2
	v_add_nc_u32_e32 v8, s3, v59
	ds_load_b128 v[36:39], v74
	ds_load_b128 v[8:11], v8
.LBB0_78:                               ;   in Loop: Header=BB0_76 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s31
	v_dual_mov_b32 v41, 0 :: v_dual_add_nc_u32 v40, 0x200, v74
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v47, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_min_u32_e32 v20, s26, v40
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v43, 0
	v_add_co_u32 v22, s31, s14, v20
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v23, null, s15, 0, s31
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v20, vcc_lo, v22, v56
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v21, null, v23, v57, vcc_lo
	v_add_co_u32 v24, vcc_lo, v22, v54
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v25, null, v23, v55, vcc_lo
	v_add_co_u32 v28, vcc_lo, v22, v52
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v29, null, v23, v53, vcc_lo
	v_add_co_u32 v32, vcc_lo, v22, v50
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v33, null, v23, v51, vcc_lo
	s_clause 0x3
	global_load_b128 v[20:23], v[20:21], off
	global_load_b128 v[24:27], v[24:25], off
	global_load_b128 v[28:31], v[28:29], off
	global_load_b128 v[32:35], v[32:33], off
	v_cmp_gt_u32_e32 vcc_lo, s12, v40
	v_mov_b32_e32 v40, 0
	s_and_saveexec_b32 s31, vcc_lo
	s_cbranch_execz .LBB0_75
; %bb.79:                               ;   in Loop: Header=BB0_76 Depth=2
	v_add_nc_u32_e32 v44, s3, v59
	ds_load_b128 v[40:43], v74 offset:512
	ds_load_b128 v[44:47], v44 offset:512
	s_branch .LBB0_75
.LBB0_80:                               ;   in Loop: Header=BB0_70 Depth=1
	v_dual_mov_b32 v72, 0 :: v_dual_mov_b32 v69, 0
	v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v73, 0
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v67, 0
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v71, 0
	s_mov_b32 s3, 0
.LBB0_81:                               ;   in Loop: Header=BB0_70 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_85
; %bb.82:                               ;   in Loop: Header=BB0_70 Depth=1
	v_dual_mov_b32 v17, 0 :: v_dual_add_nc_u32 v24, s3, v58
	s_wait_dscnt 0x5
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v21, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v0, s26, v24
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v23, 0
	v_mov_b32_e32 v22, 0
	v_add_co_u32 v2, s3, s14, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v3, null, s15, 0, s3
	s_mov_b32 s3, exec_lo
	v_add_co_u32 v0, vcc_lo, v2, v56
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v3, v57, vcc_lo
	v_add_co_u32 v4, vcc_lo, v2, v54
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v3, v55, vcc_lo
	v_add_co_u32 v8, vcc_lo, v2, v52
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, v3, v53, vcc_lo
	v_add_co_u32 v12, vcc_lo, v2, v50
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v13, null, v3, v51, vcc_lo
	s_clause 0x3
	global_load_b128 v[0:3], v[0:1], off
	global_load_b128 v[4:7], v[4:5], off
	global_load_b128 v[8:11], v[8:9], off
	global_load_b128 v[12:15], v[12:13], off
	v_cmpx_gt_u32_e64 s12, v24
	s_cbranch_execz .LBB0_84
; %bb.83:                               ;   in Loop: Header=BB0_70 Depth=1
	v_add_nc_u32_e32 v20, s12, v24
	ds_load_b128 v[16:19], v24
	ds_load_b128 v[20:23], v20
.LBB0_84:                               ;   in Loop: Header=BB0_70 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_wait_loadcnt_dscnt 0x301
	v_dot4_f32_fp8_fp8 v24, v16, v0, v72
	s_wait_loadcnt 0x2
	v_dot4_f32_fp8_fp8 v25, v16, v4, v70
	s_wait_loadcnt 0x1
	v_dot4_f32_fp8_fp8 v26, v16, v8, v69
	s_wait_loadcnt 0x0
	v_dot4_f32_fp8_fp8 v16, v16, v12, v73
	s_wait_dscnt 0x0
	v_dot4_f32_fp8_fp8 v0, v20, v0, v71
	v_dot4_f32_fp8_fp8 v4, v20, v4, v68
	v_dot4_f32_fp8_fp8 v8, v20, v8, v67
	v_dot4_f32_fp8_fp8 v12, v20, v12, v66
	v_dot4_f32_fp8_fp8 v24, v17, v1, v24
	v_dot4_f32_fp8_fp8 v25, v17, v5, v25
	v_dot4_f32_fp8_fp8 v26, v17, v9, v26
	v_dot4_f32_fp8_fp8 v16, v17, v13, v16
	v_dot4_f32_fp8_fp8 v0, v21, v1, v0
	v_dot4_f32_fp8_fp8 v1, v21, v5, v4
	v_dot4_f32_fp8_fp8 v4, v21, v9, v8
	v_dot4_f32_fp8_fp8 v5, v21, v13, v12
	v_dot4_f32_fp8_fp8 v24, v18, v2, v24
	v_dot4_f32_fp8_fp8 v25, v18, v6, v25
	v_dot4_f32_fp8_fp8 v26, v18, v10, v26
	v_dot4_f32_fp8_fp8 v8, v18, v14, v16
	v_dot4_f32_fp8_fp8 v0, v22, v2, v0
	v_dot4_f32_fp8_fp8 v1, v22, v6, v1
	v_dot4_f32_fp8_fp8 v2, v22, v10, v4
	v_dot4_f32_fp8_fp8 v4, v22, v14, v5
	v_dot4_f32_fp8_fp8 v72, v19, v3, v24
	v_dot4_f32_fp8_fp8 v70, v19, v7, v25
	v_dot4_f32_fp8_fp8 v69, v19, v11, v26
	v_dot4_f32_fp8_fp8 v73, v19, v15, v8
	v_dot4_f32_fp8_fp8 v71, v23, v3, v0
	v_dot4_f32_fp8_fp8 v68, v23, v7, v1
	v_dot4_f32_fp8_fp8 v67, v23, v11, v2
	v_dot4_f32_fp8_fp8 v66, v23, v15, v4
.LBB0_85:                               ;   in Loop: Header=BB0_70 Depth=1
	v_add_f32_dpp v2, v69, v69 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v72, v72 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x7
	v_add_f32_dpp v1, v70, v70 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v73, v73 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x0
	v_add_f32_dpp v5, v68, v68 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v67, v67 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v66, v66 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v61
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v71, v71 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v60, v61, vcc_lo
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshlrev_b32_e32 v3, 2, v3
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v7, v7 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v1, v3, v0
	ds_bpermute_b32 v18, v3, v17
	ds_bpermute_b32 v16, v3, v12
	ds_bpermute_b32 v15, v3, v13
	ds_bpermute_b32 v11, v3, v10
	ds_bpermute_b32 v9, v3, v8
	ds_bpermute_b32 v7, v3, v6
	ds_bpermute_b32 v5, v3, v4
	s_wait_loadcnt 0x3
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x2
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s20, s1
	s_cbranch_execz .LBB0_69
; %bb.86:                               ;   in Loop: Header=BB0_70 Depth=1
	v_lshlrev_b64_e32 v[2:3], 1, v[48:49]
	s_wait_dscnt 0x7
	v_add_f32_e32 v14, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v19, s28, v14
	v_add_co_u32 v0, vcc_lo, s4, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s5, v3, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s27
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_88
; %bb.87:                               ;   in Loop: Header=BB0_70 Depth=1
	global_load_u16 v14, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v19, s9, v14, v19 op_sel_hi:[0,1,0]
.LBB0_88:                               ;   in Loop: Header=BB0_70 Depth=1
	v_cvt_f32_f16_e32 v14, v65
	v_add_co_u32 v2, vcc_lo, s6, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v20, v19, v14
	v_cndmask_b32_e64 v19, v20, v19, s0
	v_or_b32_e32 v20, 1, v48
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v19, v19
	v_cmp_gt_u32_e64 s3, s13, v20
	global_store_b16 v[2:3], v19, off
	s_and_saveexec_b32 s31, s3
	s_cbranch_execz .LBB0_98
; %bb.89:                               ;   in Loop: Header=BB0_70 Depth=1
	s_wait_dscnt 0x6
	v_add_f32_e32 v17, v17, v18
	s_and_not1_b32 vcc_lo, exec_lo, s27
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v17, s28, v17
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_91
; %bb.90:                               ;   in Loop: Header=BB0_70 Depth=1
	global_load_u16 v18, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v17, s9, v18, v17 op_sel_hi:[0,1,0]
.LBB0_91:                               ;   in Loop: Header=BB0_70 Depth=1
	v_cvt_f32_f16_e32 v18, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v18, v17, v18
	v_cndmask_b32_e64 v17, v18, v17, s0
	v_or_b32_e32 v18, 2, v48
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v17, v17
	v_cmp_gt_u32_e32 vcc_lo, s13, v18
	global_store_b16 v[2:3], v17, off offset:2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_98
; %bb.92:                               ;   in Loop: Header=BB0_70 Depth=1
	s_wait_dscnt 0x5
	v_add_f32_e32 v12, v12, v16
	s_and_not1_b32 vcc_lo, exec_lo, s27
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v12, s28, v12
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_94
; %bb.93:                               ;   in Loop: Header=BB0_70 Depth=1
	global_load_u16 v16, v[0:1], off offset:4
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v12, s9, v16, v12 op_sel_hi:[0,1,0]
.LBB0_94:                               ;   in Loop: Header=BB0_70 Depth=1
	v_cvt_f32_f16_e32 v16, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v16, v12, v16
	v_cndmask_b32_e64 v12, v16, v12, s0
	v_or_b32_e32 v16, 3, v48
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v12, v12
	v_cmp_gt_u32_e32 vcc_lo, s13, v16
	global_store_b16 v[2:3], v12, off offset:4
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_98
; %bb.95:                               ;   in Loop: Header=BB0_70 Depth=1
	s_wait_dscnt 0x4
	v_add_f32_e32 v12, v13, v15
	s_and_not1_b32 vcc_lo, exec_lo, s27
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v12, s28, v12
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_97
; %bb.96:                               ;   in Loop: Header=BB0_70 Depth=1
	global_load_u16 v13, v[0:1], off offset:6
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v12, s9, v13, v12 op_sel_hi:[0,1,0]
.LBB0_97:                               ;   in Loop: Header=BB0_70 Depth=1
	v_cvt_f32_f16_e32 v13, v62
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v13, v12, v13
	v_cndmask_b32_e64 v12, v13, v12, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v12, v12
	global_store_b16 v[2:3], v12, off offset:6
.LBB0_98:                               ;   in Loop: Header=BB0_70 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s31
	s_wait_dscnt 0x3
	v_add_f32_e32 v10, v10, v11
	v_add_co_u32 v0, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v10, s28, v10
	s_and_not1_b32 vcc_lo, exec_lo, s27
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_100
; %bb.99:                               ;   in Loop: Header=BB0_70 Depth=1
	global_load_u16 v11, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v10, s9, v11, v10 op_sel_hi:[0,1,0]
.LBB0_100:                              ;   in Loop: Header=BB0_70 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v11, v10, v14
	v_add_co_u32 v2, vcc_lo, v2, s22
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s23, v3, vcc_lo
	v_cndmask_b32_e64 v10, v11, v10, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v10, v10
	global_store_b16 v[2:3], v10, off
	s_and_b32 exec_lo, exec_lo, s3
	s_cbranch_execz .LBB0_69
; %bb.101:                              ;   in Loop: Header=BB0_70 Depth=1
	s_wait_dscnt 0x2
	v_add_f32_e32 v8, v8, v9
	s_and_not1_b32 vcc_lo, exec_lo, s27
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v8, s28, v8
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_103
; %bb.102:                              ;   in Loop: Header=BB0_70 Depth=1
	global_load_u16 v9, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v8, s9, v9, v8 op_sel_hi:[0,1,0]
.LBB0_103:                              ;   in Loop: Header=BB0_70 Depth=1
	v_cvt_f32_f16_e32 v9, v64
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v8, v9
	v_cndmask_b32_e64 v8, v9, v8, s0
	v_or_b32_e32 v9, 2, v48
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v8, v8
	v_cmp_gt_u32_e32 vcc_lo, s13, v9
	global_store_b16 v[2:3], v8, off offset:2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_69
; %bb.104:                              ;   in Loop: Header=BB0_70 Depth=1
	s_wait_dscnt 0x1
	v_add_f32_e32 v6, v6, v7
	s_and_not1_b32 vcc_lo, exec_lo, s27
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v6, s28, v6
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_106
; %bb.105:                              ;   in Loop: Header=BB0_70 Depth=1
	global_load_u16 v7, v[0:1], off offset:4
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v6, s9, v7, v6 op_sel_hi:[0,1,0]
.LBB0_106:                              ;   in Loop: Header=BB0_70 Depth=1
	v_cvt_f32_f16_e32 v7, v63
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v7, v6, v7
	v_cndmask_b32_e64 v6, v7, v6, s0
	v_or_b32_e32 v7, 3, v48
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v6, v6
	v_cmp_gt_u32_e32 vcc_lo, s13, v7
	global_store_b16 v[2:3], v6, off offset:4
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_69
; %bb.107:                              ;   in Loop: Header=BB0_70 Depth=1
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	s_and_not1_b32 vcc_lo, exec_lo, s27
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v4, s28, v4
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_68
; %bb.108:                              ;   in Loop: Header=BB0_70 Depth=1
	global_load_u16 v0, v[0:1], off offset:6
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v4, s9, v0, v4 op_sel_hi:[0,1,0]
	s_branch .LBB0_68
.LBB0_109:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201
		.amdhsa_group_segment_fixed_size 65536
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 96
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 169
		.amdhsa_next_free_sgpr 38
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 58
		.amdhsa_round_robin_scheduling 0
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
	.size	wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201, .Lfunc_end0-wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201
                                        ; -- End function
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201.num_vgpr, 75
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201.num_agpr, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201.numbered_sgpr, 38
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201.num_named_barrier, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201.private_seg_size, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201.uses_vcc, 1
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201.uses_flat_scratch, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201.has_dyn_sized_stack, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201.has_recursion, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 7316
; TotalNumSgprs: 40
; NumVgprs: 75
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 21
; NumSGPRsForWavesPerEU: 40
; NumVGPRsForWavesPerEU: 169
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 1
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.type	__hip_cuid_33946b7e1b96c04e,@object ; @__hip_cuid_33946b7e1b96c04e
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_33946b7e1b96c04e
__hip_cuid_33946b7e1b96c04e:
	.byte	0                               ; 0x0
	.size	__hip_cuid_33946b7e1b96c04e, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_33946b7e1b96c04e
	.amdgpu_metadata
---
custom.config:
  Source:
    Origin: vllm
    Repository: "https://github.com/vllm-project/vllm"
  Version: 392ed873c7fed4c41fafaad022f83f7d0fe9f57a
  Features:
    SupportsUserArgs: false
    SupportsBias: true
    SupportsActivation: false
    SupportsScaleAlpha: false
    SupportsGSU: false
  InternalSupportParams:
    KernArgsVersion: 0
  ProblemType:
    OperationType: GEMM
    DataType: F8
    DestDataType: h
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: False
    UseBeta: True
    Batched: True
    UseScaleAB: Scalar
    UseBias: 1
    BiasSrc: D
    BiasDataTypeList: ['h']
    Activation: False
    UseScaleAlphaVec: 0
  CustomKernel:
    args: [ { type: int32, semantic: SizeSum },
            { type: int32, semantic: SizeFree0 },
            { type: address, semantic: AddressA },
            { type: address, semantic: AddressB },
            { type: int32, semantic: ComputeUnits },
            { type: int32, semantic: SizeFree1 },
            { type: int32, semantic: StrideA0 },
            { type: int32, semantic: StrideB0 },
            { type: address, semantic: AddressC },
            { type: address, semantic: AddressD },
            { type: float32, semantic: Alpha },
            { type: float32, semantic: Beta },
            { type: int32, semantic: StrideC0 },
            { type: int32, semantic: StrideD0 },
            { type: address, semantic: AddressScaleA },
            { type: address, semantic: AddressScaleB },
            { type: address, semantic: AddressBias } ]
    macrotile: [32, 16, 16]
    threads: [32, 16, 1]
    grid: [ComputeUnits, One, One]
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 16, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  AssertSummationElementMultiple: 16
  AssertSizeEqual: { 1: 2, 2: 1 }
  AssertStrideAEqual: { 0: 1 }
  AssertStrideBEqual: { 0: 1 }
  AssertStrideCEqual: { 0: 1 }
  AssertStrideDEqual: { 0: 1 }
  StaggerU: 0
  WavefrontSize: 32
amdhsa.kernels:
  - .args:
      - .offset:         0
        .size:           4
        .value_kind:     by_value
      - .offset:         4
        .size:           4
        .value_kind:     by_value
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         36
        .size:           4
        .value_kind:     by_value
      - .address_space:  global
        .offset:         40
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         48
        .size:           8
        .value_kind:     global_buffer
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
      - .actual_access:  read_only
        .address_space:  global
        .offset:         72
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         80
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         88
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     40
    .sgpr_spill_count: 0
    .symbol:         wvSplitKQ_vllm_f8_f16_tn_n2_y4u1_gfx1201.kd
    .vgpr_count:     75
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
