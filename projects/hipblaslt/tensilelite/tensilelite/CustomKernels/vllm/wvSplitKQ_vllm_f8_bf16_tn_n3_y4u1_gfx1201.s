	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201 ; -- Begin function wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201
	.globl	wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201
	.p2align	8
	.type	wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201,@function
wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201: ; @wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201
; %bb.0:
	s_clause 0x1
	s_load_b64 s[12:13], s[0:1], 0x0
	s_load_b64 s[20:21], s[0:1], 0x10
	v_and_b32_e32 v40, 0x3ff, v0
	v_bfe_u32 v2, v0, 10, 10
	s_mov_b32 s23, 0
	s_mov_b32 s4, exec_lo
	s_mov_b32 s3, s23
	v_lshlrev_b32_e32 v39, 4, v40
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	v_lshl_add_u32 v3, v2, 9, v39
	s_wait_kmcnt 0x0
	s_cmp_lt_u32 s12, 0x10000
	s_mov_b32 s2, s12
	s_cselect_b32 s31, s12, 0x10000
	v_cmpx_gt_u32_e64 s31, v3
	s_cbranch_execz .LBB0_9
; %bb.1:
	global_load_b128 v[4:7], v3, s[20:21]
	v_add_nc_u32_e32 v0, 0x2000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7]
	v_cmpx_gt_u32_e64 s31, v0
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.2:
	v_add_co_u32 v0, s5, s20, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v1, null, s21, 0, s5
	v_add_nc_u32_e32 v8, 0x4000, v3
	s_mov_b32 s5, exec_lo
	global_load_b128 v[4:7], v[0:1], off offset:8192
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:8192
	v_cmpx_gt_u32_e64 s31, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.3:
	global_load_b128 v[4:7], v[0:1], off offset:16384
	v_add_nc_u32_e32 v8, 0x6000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:16384
	v_cmpx_gt_u32_e64 s31, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.4:
	global_load_b128 v[4:7], v[0:1], off offset:24576
	v_add_nc_u32_e32 v8, 0x8000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:24576
	v_cmpx_gt_u32_e64 s31, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.5:
	global_load_b128 v[4:7], v[0:1], off offset:32768
	v_add_nc_u32_e32 v8, 0xa000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:32768
	v_cmpx_gt_u32_e64 s31, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.6:
	global_load_b128 v[4:7], v[0:1], off offset:40960
	v_add_nc_u32_e32 v8, 0xc000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:40960
	v_cmpx_gt_u32_e64 s31, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.7:
	global_load_b128 v[4:7], v[0:1], off offset:49152
	v_add_nc_u32_e32 v8, 0xe000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:49152
	v_cmpx_gt_u32_e64 s31, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.8:
	global_load_b128 v[4:7], v[0:1], off offset:57344
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:57344
.LBB0_9:
	s_or_b32 exec_lo, exec_lo, s4
	s_clause 0x3
	s_load_b64 s[14:15], s[0:1], 0x20
	s_load_b64 s[16:17], s[0:1], 0x8
	s_load_b64 s[18:19], s[0:1], 0x58
	s_load_b128 s[4:7], s[0:1], 0x48
	s_sub_nc_u64 s[8:9], 0x10000, s[2:3]
	s_mov_b32 s10, exec_lo
	v_cmp_lt_u64_e64 s9, s[8:9], s[2:3]
	s_and_b32 s9, s9, exec_lo
	s_cselect_b32 s8, s8, s2
	s_cmp_lt_u32 s12, 0x10000
	s_wait_alu 0xfffe
	s_cselect_b32 s33, s8, 0
	s_wait_kmcnt 0x0
	s_mov_b32 s22, s15
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[8:9], s[20:21], s[22:23]
	v_cmpx_gt_u32_e64 s33, v3
	s_cbranch_execz .LBB0_12
; %bb.10:
	v_mov_b32_e32 v0, v3
	s_mov_b32 s11, 0
.LBB0_11:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[4:7], v0, s[8:9]
	v_add_nc_u32_e32 v1, s12, v0
	v_add_nc_u32_e32 v0, 0x2000, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s33, v0
	s_or_b32 s11, vcc_lo, s11
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[4:7]
	s_and_not1_b32 exec_lo, exec_lo, s11
	s_cbranch_execnz .LBB0_11
.LBB0_12:
	s_or_b32 exec_lo, exec_lo, s10
	s_lshl_b64 s[10:11], s[2:3], 1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_sub_nc_u64 s[10:11], 0x10000, s[10:11]
	v_cmp_lt_u64_e64 s3, s[10:11], s[2:3]
	s_mov_b32 s11, exec_lo
	s_and_b32 s3, s3, exec_lo
	s_cselect_b32 s2, s10, s2
	s_cmp_lt_u32 s12, 0x8000
	s_mov_b32 s10, 0
	s_wait_alu 0xfffe
	s_cselect_b32 s35, s2, 0
	s_lshl_b32 s34, s12, 1
	v_cmpx_gt_u32_e64 s35, v3
	s_cbranch_execz .LBB0_15
; %bb.13:
	s_add_nc_u64 s[2:3], s[8:9], s[22:23]
.LBB0_14:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[4:7], v3, s[2:3]
	v_add_nc_u32_e32 v0, s34, v3
	v_add_nc_u32_e32 v3, 0x2000, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s35, v3
	s_or_b32 s10, vcc_lo, s10
	s_wait_loadcnt 0x0
	ds_store_b128 v0, v[4:7]
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s10
	s_cbranch_execnz .LBB0_14
.LBB0_15:
	s_or_b32 exec_lo, exec_lo, s11
	s_load_b32 s2, s[0:1], 0x18
	s_add_co_i32 s15, s13, -1
	s_wait_kmcnt 0x0
	s_lshl_b32 s8, s2, 6
	s_wait_alu 0xfffe
	v_add_co_u32 v0, s3, s8, s15
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_mov_b32 s3, 4
	v_readfirstlane_b32 s9, v0
	s_cbranch_vccz .LBB0_17
; %bb.16:
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
	s_wait_alu 0xfffe
	s_mul_hi_u32 s10, s3, s10
	s_wait_alu 0xfffe
	s_add_co_i32 s3, s3, s10
	s_wait_alu 0xfffe
	s_mul_hi_u32 s3, s9, s3
	s_wait_alu 0xfffe
	s_mul_i32 s10, s3, s8
	s_wait_alu 0xfffe
	s_sub_co_i32 s9, s9, s10
	s_add_co_i32 s10, s3, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s11, s9, s8
	s_cmp_ge_u32 s9, s8
	s_cselect_b32 s3, s10, s3
	s_wait_alu 0xfffe
	s_cselect_b32 s9, s11, s9
	s_add_co_i32 s10, s3, 1
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s9, s8
	s_cselect_b32 s3, s10, s3
	s_wait_alu 0xfffe
	s_mul_i32 s3, s2, s3
	s_wait_alu 0xfffe
	s_lshl_b32 s3, s3, 2
	s_wait_alu 0xfffe
	s_cvt_f32_u32 s8, s3
	s_sub_co_i32 s9, 0, s3
	s_add_co_i32 s10, s3, s15
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
	s_wait_alu 0xfffe
	s_cselect_b32 s9, s11, s9
	s_add_co_i32 s10, s8, 1
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s9, s3
	s_cselect_b32 s3, s10, s8
	s_wait_alu 0xfffe
	s_max_u32 s3, s3, 4
.LBB0_17:
	s_mov_b32 s24, 1.0
	s_cmp_eq_u64 s[4:5], 0
	s_mov_b32 s25, 1.0
	s_cbranch_scc1 .LBB0_19
; %bb.18:
	s_load_b32 s25, s[4:5], 0x0
.LBB0_19:
	s_cmp_eq_u64 s[6:7], 0
	s_cbranch_scc1 .LBB0_21
; %bb.20:
	s_load_b32 s24, s[6:7], 0x0
.LBB0_21:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_mov_b32 s4, exec_lo
	s_wait_alu 0xfffe
	v_cmpx_gt_u32_e64 s3, v2
	s_cbranch_execz .LBB0_133
; %bb.22:
	s_load_b256 s[4:11], s[0:1], 0x28
	s_cmp_eq_u32 s33, s12
	s_mul_i32 s2, s2, s3
	s_cselect_b32 s0, -1, 0
	s_cmp_eq_u32 s35, s12
	s_cselect_b32 s1, -1, 0
	s_cmp_eq_u32 s12, s31
	s_cselect_b32 s26, -1, 0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_and_b32 s0, s26, s0
	s_and_b32 s1, s0, s1
	s_mul_i32 s0, s3, ttmp9
	s_and_not1_b32 vcc_lo, exec_lo, s1
	v_add_lshl_u32 v28, s0, v2, 2
	s_mov_b32 s1, -1
	s_wait_kmcnt 0x0
	s_mul_f32 s3, s8, s25
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s0, s13, v28
	s_wait_alu 0xfffe
	s_lshl_b32 s8, s2, 2
	s_mul_f32 s30, s3, s24
	s_cbranch_vccz .LBB0_84
; %bb.23:
	s_and_saveexec_b32 s36, s0
	s_cbranch_execz .LBB0_83
; %bb.24:
	s_lshl_b64 s[24:25], s[22:23], 1
	s_cmp_eq_u64 s[18:19], 0
	v_mbcnt_lo_u32_b32 v42, -1, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[18:19], 0
	s_mov_b32 s3, 0
	s_cselect_b32 s23, -1, 0
	s_cmp_lg_u32 s12, 0
	v_cmp_eq_u32_e64 s1, 31, v40
	s_cselect_b32 s37, -1, 0
	s_add_co_i32 s38, s12, -16
	s_cmp_neq_f32 s9, 0
	v_dual_mov_b32 v30, 0 :: v_dual_add_nc_u32 v41, s12, v39
	v_add_nc_u32_e32 v43, s34, v39
	v_xor_b32_e32 v44, 16, v42
	v_mov_b32_e32 v29, v28
	s_mov_b32 s2, s10
	s_mov_b32 s28, s11
	s_wait_alu 0xfffe
	s_mov_b32 s29, s3
	s_cselect_b32 s39, -1, 0
	s_lshl_b64 s[26:27], s[2:3], 1
	s_lshl_b64 s[28:29], s[28:29], 1
	s_branch .LBB0_27
.LBB0_25:                               ;   in Loop: Header=BB0_27 Depth=1
	v_lshlrev_b32_e32 v0, 16, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v4, v0
	v_cndmask_b32_e64 v0, v0, v4, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v1, v0, 16, 1
	v_or_b32_e32 v4, 0x400000, v0
	v_cmp_u_f32_e32 vcc_lo, v0, v0
	v_add3_u32 v1, v1, v0, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v0, v1, v4, vcc_lo
	global_store_d16_hi_b16 v[2:3], v0, off offset:6
.LBB0_26:                               ;   in Loop: Header=BB0_27 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s40
	v_add_nc_u32_e32 v29, s8, v29
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s13, v29
	s_or_b32 s3, vcc_lo, s3
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s3
	s_cbranch_execz .LBB0_83
.LBB0_27:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_33 Depth 2
	v_or_b32_e32 v48, 1, v29
	v_min_u32_e32 v0, s15, v29
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v46, 0
	v_mov_b32_e32 v47, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_min_u32_e32 v2, s15, v48
	v_mov_b32_e32 v45, 0
	s_and_not1_b32 vcc_lo, exec_lo, s23
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_29
; %bb.28:                               ;   in Loop: Header=BB0_27 Depth=1
	s_wait_dscnt 0xb
	v_mov_b32_e32 v1, v30
	v_mov_b32_e32 v3, v30
	v_or_b32_e32 v6, 2, v29
	s_wait_dscnt 0x1
	v_mov_b32_e32 v7, v30
	v_mov_b32_e32 v11, v30
	s_wait_dscnt 0x0
	v_lshlrev_b64_e32 v[4:5], 1, v[0:1]
	v_or_b32_e32 v1, 3, v29
	v_lshlrev_b64_e32 v[8:9], 1, v[2:3]
	v_min_u32_e32 v6, s15, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_min_u32_e32 v10, s15, v1
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
	global_load_u16 v49, v[3:4], off
	global_load_u16 v47, v[7:8], off
	global_load_u16 v46, v[5:6], off
	global_load_u16 v45, v[9:10], off
.LBB0_29:                               ;   in Loop: Header=BB0_27 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s37
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_46
; %bb.30:                               ;   in Loop: Header=BB0_27 Depth=1
	s_wait_dscnt 0xb
	v_or_b32_e32 v1, 2, v29
	v_or_b32_e32 v3, 3, v29
	v_mad_co_u64_u32 v[31:32], null, v0, s14, 0
	v_mad_co_u64_u32 v[33:34], null, v2, s14, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_min_u32_e32 v0, s15, v1
	v_min_u32_e32 v1, s15, v3
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v54, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mad_co_u64_u32 v[35:36], null, v0, s14, 0
	v_mad_co_u64_u32 v[37:38], null, v1, s14, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v59, 0
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v61, 0
	s_mov_b32 s2, 0
	s_branch .LBB0_33
.LBB0_31:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s41
.LBB0_32:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s40
	s_wait_loadcnt_dscnt 0x0
	v_dot4_f32_fp8_fp8 v61, v16, v0, v61
	v_dot4_f32_fp8_fp8 v60, v16, v4, v60
	v_dot4_f32_fp8_fp8 v59, v16, v8, v59
	v_dot4_f32_fp8_fp8 v16, v16, v12, v58
	v_dot4_f32_fp8_fp8 v57, v20, v0, v57
	v_dot4_f32_fp8_fp8 v58, v17, v1, v61
	v_dot4_f32_fp8_fp8 v56, v20, v4, v56
	v_dot4_f32_fp8_fp8 v55, v20, v8, v55
	v_dot4_f32_fp8_fp8 v16, v17, v13, v16
	v_dot4_f32_fp8_fp8 v0, v24, v0, v53
	v_dot4_f32_fp8_fp8 v58, v18, v2, v58
	v_dot4_f32_fp8_fp8 v4, v24, v4, v52
	v_dot4_f32_fp8_fp8 v8, v24, v8, v51
	v_dot4_f32_fp8_fp8 v16, v18, v14, v16
	v_dot4_f32_fp8_fp8 v60, v17, v5, v60
	v_dot4_f32_fp8_fp8 v61, v19, v3, v58
	v_dot4_f32_fp8_fp8 v59, v17, v9, v59
	v_dot4_f32_fp8_fp8 v17, v21, v1, v57
	v_dot4_f32_fp8_fp8 v58, v19, v15, v16
	v_dot4_f32_fp8_fp8 v16, v20, v12, v54
	v_dot4_f32_fp8_fp8 v12, v24, v12, v50
	v_dot4_f32_fp8_fp8 v56, v21, v5, v56
	v_dot4_f32_fp8_fp8 v55, v21, v9, v55
	v_dot4_f32_fp8_fp8 v0, v25, v1, v0
	v_dot4_f32_fp8_fp8 v16, v21, v13, v16
	v_dot4_f32_fp8_fp8 v1, v25, v5, v4
	v_dot4_f32_fp8_fp8 v4, v25, v9, v8
	v_dot4_f32_fp8_fp8 v5, v25, v13, v12
	v_dot4_f32_fp8_fp8 v60, v18, v6, v60
	v_dot4_f32_fp8_fp8 v57, v18, v10, v59
	v_dot4_f32_fp8_fp8 v17, v22, v2, v17
	v_dot4_f32_fp8_fp8 v18, v22, v6, v56
	v_dot4_f32_fp8_fp8 v55, v22, v10, v55
	v_dot4_f32_fp8_fp8 v8, v22, v14, v16
	v_dot4_f32_fp8_fp8 v0, v26, v2, v0
	v_dot4_f32_fp8_fp8 v1, v26, v6, v1
	v_dot4_f32_fp8_fp8 v2, v26, v10, v4
	v_dot4_f32_fp8_fp8 v4, v26, v14, v5
	v_dot4_f32_fp8_fp8 v60, v19, v7, v60
	v_dot4_f32_fp8_fp8 v59, v19, v11, v57
	v_dot4_f32_fp8_fp8 v57, v23, v3, v17
	v_dot4_f32_fp8_fp8 v56, v23, v7, v18
	v_dot4_f32_fp8_fp8 v55, v23, v11, v55
	v_dot4_f32_fp8_fp8 v54, v23, v15, v8
	v_dot4_f32_fp8_fp8 v53, v27, v3, v0
	v_dot4_f32_fp8_fp8 v52, v27, v7, v1
	v_dot4_f32_fp8_fp8 v51, v27, v11, v2
	v_dot4_f32_fp8_fp8 v50, v27, v15, v4
	s_addk_co_i32 s2, 0x200
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s2, s12
	s_cbranch_scc1 .LBB0_47
.LBB0_33:                               ;   Parent Loop BB0_27 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_dual_mov_b32 v27, 0 :: v_dual_add_nc_u32 v62, s2, v39
	s_wait_dscnt 0xa
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v25, 0
	s_wait_dscnt 0x9
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v23, 0
	v_min_u32_e32 v0, s38, v62
	s_wait_dscnt 0x8
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v21, 0
	s_wait_dscnt 0x7
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v19, 0
	v_add_co_u32 v2, s40, s16, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v3, null, s17, 0, s40
	s_wait_dscnt 0x6
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v17, 0
	v_add_co_u32 v0, vcc_lo, v2, v31
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v3, v32, vcc_lo
	v_add_co_u32 v4, vcc_lo, v2, v33
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v3, v34, vcc_lo
	v_add_co_u32 v8, vcc_lo, v2, v35
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, v3, v36, vcc_lo
	v_add_co_u32 v12, vcc_lo, v2, v37
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v13, null, v3, v38, vcc_lo
	s_clause 0x3
	global_load_b128 v[0:3], v[0:1], off
	global_load_b128 v[4:7], v[4:5], off
	global_load_b128 v[8:11], v[8:9], off
	global_load_b128 v[12:15], v[12:13], off
	v_mov_b32_e32 v16, 0
	s_mov_b32 s40, exec_lo
	v_cmpx_gt_u32_e64 s12, v62
	s_cbranch_execz .LBB0_32
; %bb.34:                               ;   in Loop: Header=BB0_33 Depth=2
	v_add_co_u32 v24, s41, s20, v62
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v25, null, s21, 0, s41
	s_mov_b32 s41, exec_lo
                                        ; implicit-def: $vgpr19
	v_cmpx_le_u32_e64 s31, v62
	s_wait_alu 0xfffe
	s_xor_b32 s41, exec_lo, s41
	s_cbranch_execz .LBB0_36
; %bb.35:                               ;   in Loop: Header=BB0_33 Depth=2
	global_load_b128 v[16:19], v[24:25], off
.LBB0_36:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s41, s41
	s_cbranch_execz .LBB0_38
; %bb.37:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[16:19], v62
.LBB0_38:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s41
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s41, exec_lo
                                        ; implicit-def: $vgpr23
	v_cmpx_le_u32_e64 s33, v62
	s_wait_alu 0xfffe
	s_xor_b32 s41, exec_lo, s41
	s_cbranch_execz .LBB0_40
; %bb.39:                               ;   in Loop: Header=BB0_33 Depth=2
	v_add_co_u32 v20, vcc_lo, v24, s22
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v21, null, 0, v25, vcc_lo
	global_load_b128 v[20:23], v[20:21], off
.LBB0_40:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s41, s41
	s_cbranch_execz .LBB0_42
; %bb.41:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v20, s2, v41
	ds_load_b128 v[20:23], v20
.LBB0_42:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s41
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s41, exec_lo
                                        ; implicit-def: $vgpr27
	v_cmpx_le_u32_e64 s35, v62
	s_wait_alu 0xfffe
	s_xor_b32 s41, exec_lo, s41
	s_cbranch_execz .LBB0_44
; %bb.43:                               ;   in Loop: Header=BB0_33 Depth=2
	v_add_co_u32 v24, vcc_lo, v24, s24
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v25, null, s25, v25, vcc_lo
	global_load_b128 v[24:27], v[24:25], off
.LBB0_44:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s41, s41
	s_cbranch_execz .LBB0_31
; %bb.45:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v24, s2, v43
	ds_load_b128 v[24:27], v24
	s_branch .LBB0_31
.LBB0_46:                               ;   in Loop: Header=BB0_27 Depth=1
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v60, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v51, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v57, 0
.LBB0_47:                               ;   in Loop: Header=BB0_27 Depth=1
	v_add_f32_dpp v3, v59, v59 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v58, v58 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3)
	v_add_f32_dpp v5, v57, v57 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v56, v56 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v55, v55 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v61, v61 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v60, v60 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v23, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v21, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v19, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v14, v7, v7 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v54, v54 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v53, v53 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v52, v52 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v51, v51 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v50, v50 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v44
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v2, v42, v44, vcc_lo
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshlrev_b32_e32 v2, 2, v2
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v25, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v7, v7 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v1, v2, v0
	ds_bpermute_b32 v26, v2, v25
	ds_bpermute_b32 v24, v2, v23
	ds_bpermute_b32 v22, v2, v21
	ds_bpermute_b32 v20, v2, v19
	ds_bpermute_b32 v18, v2, v17
	ds_bpermute_b32 v16, v2, v14
	ds_bpermute_b32 v15, v2, v13
	ds_bpermute_b32 v11, v2, v10
	ds_bpermute_b32 v9, v2, v8
	ds_bpermute_b32 v7, v2, v6
	ds_bpermute_b32 v5, v2, v4
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
	s_and_saveexec_b32 s40, s1
	s_cbranch_execz .LBB0_26
; %bb.48:                               ;   in Loop: Header=BB0_27 Depth=1
	v_lshlrev_b64_e32 v[2:3], 1, v[29:30]
	s_wait_dscnt 0xb
	v_add_f32_e32 v12, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v27, s30, v12
	v_add_co_u32 v0, vcc_lo, s4, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s5, v3, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_50
; %bb.49:                               ;   in Loop: Header=BB0_27 Depth=1
	global_load_u16 v12, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v12, 16, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, s9, v12
.LBB0_50:                               ;   in Loop: Header=BB0_27 Depth=1
	v_lshlrev_b32_e32 v12, 16, v49
	v_add_co_u32 v2, vcc_lo, s6, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v31, v27, v12
	v_cmp_gt_u32_e64 s2, s13, v48
	v_cndmask_b32_e64 v27, v31, v27, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v31, v27, 16, 1
	v_or_b32_e32 v32, 0x400000, v27
	v_cmp_u_f32_e32 vcc_lo, v27, v27
	v_add3_u32 v31, v31, v27, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v27, v31, v32, vcc_lo
	global_store_d16_hi_b16 v[2:3], v27, off
	s_and_saveexec_b32 s41, s2
	s_cbranch_execz .LBB0_60
; %bb.51:                               ;   in Loop: Header=BB0_27 Depth=1
	s_wait_dscnt 0xa
	v_add_f32_e32 v25, v25, v26
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v25, s30, v25
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_53
; %bb.52:                               ;   in Loop: Header=BB0_27 Depth=1
	global_load_u16 v26, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v26, 16, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, s9, v26
.LBB0_53:                               ;   in Loop: Header=BB0_27 Depth=1
	v_lshlrev_b32_e32 v26, 16, v47
	v_or_b32_e32 v31, 2, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v26, v25, v26
	v_cndmask_b32_e64 v25, v26, v25, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v26, v25, 16, 1
	v_or_b32_e32 v27, 0x400000, v25
	v_cmp_u_f32_e32 vcc_lo, v25, v25
	v_add3_u32 v26, v26, v25, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v25, v26, v27, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, s13, v31
	global_store_d16_hi_b16 v[2:3], v25, off offset:2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_60
; %bb.54:                               ;   in Loop: Header=BB0_27 Depth=1
	s_wait_dscnt 0x9
	v_add_f32_e32 v23, v23, v24
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v23, s30, v23
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_27 Depth=1
	global_load_u16 v24, v[0:1], off offset:4
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v24, 16, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v23, s9, v24
.LBB0_56:                               ;   in Loop: Header=BB0_27 Depth=1
	v_lshlrev_b32_e32 v24, 16, v46
	v_or_b32_e32 v26, 3, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v24, v23, v24
	v_cndmask_b32_e64 v23, v24, v23, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v24, v23, 16, 1
	v_or_b32_e32 v25, 0x400000, v23
	v_cmp_u_f32_e32 vcc_lo, v23, v23
	v_add3_u32 v24, v24, v23, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v23, v24, v25, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, s13, v26
	global_store_d16_hi_b16 v[2:3], v23, off offset:4
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_60
; %bb.57:                               ;   in Loop: Header=BB0_27 Depth=1
	s_wait_dscnt 0x8
	v_add_f32_e32 v21, v21, v22
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v21, s30, v21
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_59
; %bb.58:                               ;   in Loop: Header=BB0_27 Depth=1
	global_load_u16 v22, v[0:1], off offset:6
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v22, 16, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v21, s9, v22
.LBB0_59:                               ;   in Loop: Header=BB0_27 Depth=1
	v_lshlrev_b32_e32 v22, 16, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v22, v21, v22
	v_cndmask_b32_e64 v21, v22, v21, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v22, v21, 16, 1
	v_or_b32_e32 v23, 0x400000, v21
	v_cmp_u_f32_e32 vcc_lo, v21, v21
	v_add3_u32 v22, v22, v21, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v21, v22, v23, vcc_lo
	global_store_d16_hi_b16 v[2:3], v21, off offset:6
.LBB0_60:                               ;   in Loop: Header=BB0_27 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s41
	s_wait_dscnt 0x7
	v_add_f32_e32 v19, v19, v20
	v_add_co_u32 v0, vcc_lo, v0, s26
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s27, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v19, s30, v19
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_62
; %bb.61:                               ;   in Loop: Header=BB0_27 Depth=1
	global_load_u16 v20, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v20, 16, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v19, s9, v20
.LBB0_62:                               ;   in Loop: Header=BB0_27 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v20, v19, v12
	v_add_co_u32 v2, vcc_lo, v2, s28
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s29, v3, vcc_lo
	v_cndmask_b32_e64 v19, v20, v19, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v20, v19, 16, 1
	v_or_b32_e32 v21, 0x400000, v19
	v_cmp_u_f32_e32 vcc_lo, v19, v19
	v_add3_u32 v20, v20, v19, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v19, v20, v21, vcc_lo
	global_store_d16_hi_b16 v[2:3], v19, off
	s_and_saveexec_b32 s41, s2
	s_cbranch_execz .LBB0_72
; %bb.63:                               ;   in Loop: Header=BB0_27 Depth=1
	s_wait_dscnt 0x6
	v_add_f32_e32 v17, v17, v18
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v17, s30, v17
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_65
; %bb.64:                               ;   in Loop: Header=BB0_27 Depth=1
	global_load_u16 v18, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v18, 16, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, s9, v18
.LBB0_65:                               ;   in Loop: Header=BB0_27 Depth=1
	v_lshlrev_b32_e32 v18, 16, v47
	v_or_b32_e32 v20, 2, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v18, v17, v18
	v_cndmask_b32_e64 v17, v18, v17, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v18, v17, 16, 1
	v_or_b32_e32 v19, 0x400000, v17
	v_cmp_u_f32_e32 vcc_lo, v17, v17
	v_add3_u32 v18, v18, v17, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v17, v18, v19, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, s13, v20
	global_store_d16_hi_b16 v[2:3], v17, off offset:2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_72
; %bb.66:                               ;   in Loop: Header=BB0_27 Depth=1
	s_wait_dscnt 0x5
	v_add_f32_e32 v14, v14, v16
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v14, s30, v14
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_68
; %bb.67:                               ;   in Loop: Header=BB0_27 Depth=1
	global_load_u16 v16, v[0:1], off offset:4
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, s9, v16
.LBB0_68:                               ;   in Loop: Header=BB0_27 Depth=1
	v_lshlrev_b32_e32 v16, 16, v46
	v_or_b32_e32 v18, 3, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v16, v14, v16
	v_cndmask_b32_e64 v14, v16, v14, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v16, v14, 16, 1
	v_or_b32_e32 v17, 0x400000, v14
	v_cmp_u_f32_e32 vcc_lo, v14, v14
	v_add3_u32 v16, v16, v14, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v14, v16, v17, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, s13, v18
	global_store_d16_hi_b16 v[2:3], v14, off offset:4
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_72
; %bb.69:                               ;   in Loop: Header=BB0_27 Depth=1
	s_wait_dscnt 0x4
	v_add_f32_e32 v13, v13, v15
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v13, s30, v13
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_71
; %bb.70:                               ;   in Loop: Header=BB0_27 Depth=1
	global_load_u16 v14, v[0:1], off offset:6
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, s9, v14
.LBB0_71:                               ;   in Loop: Header=BB0_27 Depth=1
	v_lshlrev_b32_e32 v14, 16, v45
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v14, v13, v14
	v_cndmask_b32_e64 v13, v14, v13, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v14, v13, 16, 1
	v_or_b32_e32 v15, 0x400000, v13
	v_cmp_u_f32_e32 vcc_lo, v13, v13
	v_add3_u32 v14, v14, v13, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v13, v14, v15, vcc_lo
	global_store_d16_hi_b16 v[2:3], v13, off offset:6
.LBB0_72:                               ;   in Loop: Header=BB0_27 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s41
	s_wait_dscnt 0x3
	v_add_f32_e32 v10, v10, v11
	v_add_co_u32 v0, vcc_lo, v0, s26
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s27, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v10, s30, v10
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_74
; %bb.73:                               ;   in Loop: Header=BB0_27 Depth=1
	global_load_u16 v11, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v11, 16, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v10, s9, v11
.LBB0_74:                               ;   in Loop: Header=BB0_27 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v11, v10, v12
	v_add_co_u32 v2, vcc_lo, v2, s28
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s29, v3, vcc_lo
	v_cndmask_b32_e64 v10, v11, v10, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v11, v10, 16, 1
	v_or_b32_e32 v12, 0x400000, v10
	v_cmp_u_f32_e32 vcc_lo, v10, v10
	v_add3_u32 v11, v11, v10, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v10, v11, v12, vcc_lo
	global_store_d16_hi_b16 v[2:3], v10, off
	s_and_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB0_26
; %bb.75:                               ;   in Loop: Header=BB0_27 Depth=1
	s_wait_dscnt 0x2
	v_add_f32_e32 v8, v8, v9
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v8, s30, v8
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_77
; %bb.76:                               ;   in Loop: Header=BB0_27 Depth=1
	global_load_u16 v9, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v9, 16, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, s9, v9
.LBB0_77:                               ;   in Loop: Header=BB0_27 Depth=1
	v_lshlrev_b32_e32 v9, 16, v47
	v_or_b32_e32 v11, 2, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v8, v9
	v_cndmask_b32_e64 v8, v9, v8, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v9, v8, 16, 1
	v_or_b32_e32 v10, 0x400000, v8
	v_cmp_u_f32_e32 vcc_lo, v8, v8
	v_add3_u32 v9, v9, v8, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v8, v9, v10, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, s13, v11
	global_store_d16_hi_b16 v[2:3], v8, off offset:2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_26
; %bb.78:                               ;   in Loop: Header=BB0_27 Depth=1
	s_wait_dscnt 0x1
	v_add_f32_e32 v6, v6, v7
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v6, s30, v6
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_80
; %bb.79:                               ;   in Loop: Header=BB0_27 Depth=1
	global_load_u16 v7, v[0:1], off offset:4
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v7, 16, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v6, s9, v7
.LBB0_80:                               ;   in Loop: Header=BB0_27 Depth=1
	v_lshlrev_b32_e32 v7, 16, v46
	v_or_b32_e32 v9, 3, v29
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v7, v6, v7
	v_cndmask_b32_e64 v6, v7, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v7, v6, 16, 1
	v_or_b32_e32 v8, 0x400000, v6
	v_cmp_u_f32_e32 vcc_lo, v6, v6
	v_add3_u32 v7, v7, v6, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v6, v7, v8, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, s13, v9
	global_store_d16_hi_b16 v[2:3], v6, off offset:4
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_26
; %bb.81:                               ;   in Loop: Header=BB0_27 Depth=1
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v4, s30, v4
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_25
; %bb.82:                               ;   in Loop: Header=BB0_27 Depth=1
	global_load_u16 v0, v[0:1], off offset:6
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v0, 16, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v4, s9, v0
	s_branch .LBB0_25
.LBB0_83:
	s_or_b32 exec_lo, exec_lo, s36
	s_mov_b32 s1, 0
.LBB0_84:
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_133
; %bb.85:
	v_cmp_gt_u32_e32 vcc_lo, s13, v28
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_133
; %bb.86:
	s_cmp_eq_u64 s[18:19], 0
	v_cmp_eq_u32_e64 s1, 31, v40
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[18:19], 0
	v_mbcnt_lo_u32_b32 v40, -1, 0
	s_cselect_b32 s22, -1, 0
	s_cmp_lg_u32 s12, 0
	s_mov_b32 s3, 0
	s_cselect_b32 s23, -1, 0
	s_add_co_i32 s24, s12, -16
	s_cmp_neq_f32 s9, 0
	v_dual_mov_b32 v29, 0 :: v_dual_add_nc_u32 v38, s34, v39
	v_add_nc_u32_e32 v41, s12, v39
	v_xor_b32_e32 v42, 16, v40
	s_mov_b32 s2, s10
	s_mov_b32 s20, s11
	s_wait_alu 0xfffe
	s_mov_b32 s21, s3
	s_cselect_b32 s25, -1, 0
	s_lshl_b64 s[10:11], s[2:3], 1
	s_wait_alu 0xfffe
	s_lshl_b64 s[20:21], s[20:21], 1
	s_branch .LBB0_89
.LBB0_87:                               ;   in Loop: Header=BB0_89 Depth=1
	v_lshlrev_b32_e32 v0, 16, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v4, v0
	v_cndmask_b32_e64 v0, v0, v4, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v1, v0, 16, 1
	v_or_b32_e32 v4, 0x400000, v0
	v_cmp_u_f32_e32 vcc_lo, v0, v0
	v_add3_u32 v1, v1, v0, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v0, v1, v4, vcc_lo
	global_store_d16_hi_b16 v[2:3], v0, off offset:6
.LBB0_88:                               ;   in Loop: Header=BB0_89 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s26
	v_add_nc_u32_e32 v28, s8, v28
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s13, v28
	s_or_b32 s3, vcc_lo, s3
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s3
	s_cbranch_execz .LBB0_133
.LBB0_89:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_94 Depth 2
	v_or_b32_e32 v46, 1, v28
	v_min_u32_e32 v0, s15, v28
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v44, 0
	v_mov_b32_e32 v45, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_min_u32_e32 v2, s15, v46
	v_mov_b32_e32 v43, 0
	s_and_not1_b32 vcc_lo, exec_lo, s22
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_91
; %bb.90:                               ;   in Loop: Header=BB0_89 Depth=1
	s_wait_dscnt 0xb
	v_mov_b32_e32 v1, v29
	v_mov_b32_e32 v3, v29
	v_or_b32_e32 v6, 2, v28
	s_wait_dscnt 0x1
	v_mov_b32_e32 v7, v29
	v_mov_b32_e32 v11, v29
	s_wait_dscnt 0x0
	v_lshlrev_b64_e32 v[4:5], 1, v[0:1]
	v_or_b32_e32 v1, 3, v28
	v_lshlrev_b64_e32 v[8:9], 1, v[2:3]
	v_min_u32_e32 v6, s15, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_min_u32_e32 v10, s15, v1
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
	global_load_u16 v47, v[3:4], off
	global_load_u16 v45, v[7:8], off
	global_load_u16 v44, v[5:6], off
	global_load_u16 v43, v[9:10], off
.LBB0_91:                               ;   in Loop: Header=BB0_89 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s23
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_96
; %bb.92:                               ;   in Loop: Header=BB0_89 Depth=1
	s_wait_dscnt 0xb
	v_or_b32_e32 v1, 2, v28
	v_or_b32_e32 v3, 3, v28
	v_mad_co_u64_u32 v[30:31], null, v0, s14, 0
	v_mad_co_u64_u32 v[32:33], null, v2, s14, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_min_u32_e32 v0, s15, v1
	v_min_u32_e32 v1, s15, v3
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v52, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mad_co_u64_u32 v[34:35], null, v0, s14, 0
	v_mad_co_u64_u32 v[36:37], null, v1, s14, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v57, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v59, 0
	s_mov_b32 s2, 0
	s_branch .LBB0_94
.LBB0_93:                               ;   in Loop: Header=BB0_94 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s26
	s_wait_loadcnt_dscnt 0x302
	v_dot4_f32_fp8_fp8 v59, v24, v0, v59
	s_wait_loadcnt 0x2
	v_dot4_f32_fp8_fp8 v58, v24, v4, v58
	s_wait_loadcnt 0x1
	v_dot4_f32_fp8_fp8 v57, v24, v8, v57
	s_wait_loadcnt 0x0
	v_dot4_f32_fp8_fp8 v24, v24, v12, v56
	s_wait_dscnt 0x1
	v_dot4_f32_fp8_fp8 v55, v16, v0, v55
	v_dot4_f32_fp8_fp8 v54, v16, v4, v54
	v_dot4_f32_fp8_fp8 v53, v16, v8, v53
	v_dot4_f32_fp8_fp8 v16, v16, v12, v52
	s_wait_dscnt 0x0
	v_dot4_f32_fp8_fp8 v0, v20, v0, v51
	v_dot4_f32_fp8_fp8 v4, v20, v4, v50
	v_dot4_f32_fp8_fp8 v8, v20, v8, v49
	v_dot4_f32_fp8_fp8 v12, v20, v12, v48
	v_dot4_f32_fp8_fp8 v56, v25, v1, v59
	v_dot4_f32_fp8_fp8 v58, v25, v5, v58
	v_dot4_f32_fp8_fp8 v57, v25, v9, v57
	v_dot4_f32_fp8_fp8 v24, v25, v13, v24
	v_dot4_f32_fp8_fp8 v25, v17, v1, v55
	v_dot4_f32_fp8_fp8 v54, v17, v5, v54
	v_dot4_f32_fp8_fp8 v53, v17, v9, v53
	v_dot4_f32_fp8_fp8 v16, v17, v13, v16
	v_dot4_f32_fp8_fp8 v0, v21, v1, v0
	v_dot4_f32_fp8_fp8 v1, v21, v5, v4
	v_dot4_f32_fp8_fp8 v4, v21, v9, v8
	v_dot4_f32_fp8_fp8 v5, v21, v13, v12
	v_dot4_f32_fp8_fp8 v56, v26, v2, v56
	v_dot4_f32_fp8_fp8 v58, v26, v6, v58
	v_dot4_f32_fp8_fp8 v55, v26, v10, v57
	v_dot4_f32_fp8_fp8 v24, v26, v14, v24
	v_dot4_f32_fp8_fp8 v25, v18, v2, v25
	v_dot4_f32_fp8_fp8 v26, v18, v6, v54
	v_dot4_f32_fp8_fp8 v53, v18, v10, v53
	v_dot4_f32_fp8_fp8 v8, v18, v14, v16
	v_dot4_f32_fp8_fp8 v0, v22, v2, v0
	v_dot4_f32_fp8_fp8 v1, v22, v6, v1
	v_dot4_f32_fp8_fp8 v2, v22, v10, v4
	v_dot4_f32_fp8_fp8 v4, v22, v14, v5
	v_dot4_f32_fp8_fp8 v59, v27, v3, v56
	v_dot4_f32_fp8_fp8 v58, v27, v7, v58
	v_dot4_f32_fp8_fp8 v57, v27, v11, v55
	v_dot4_f32_fp8_fp8 v56, v27, v15, v24
	v_dot4_f32_fp8_fp8 v55, v19, v3, v25
	v_dot4_f32_fp8_fp8 v54, v19, v7, v26
	v_dot4_f32_fp8_fp8 v53, v19, v11, v53
	v_dot4_f32_fp8_fp8 v52, v19, v15, v8
	v_dot4_f32_fp8_fp8 v51, v23, v3, v0
	v_dot4_f32_fp8_fp8 v50, v23, v7, v1
	v_dot4_f32_fp8_fp8 v49, v23, v11, v2
	v_dot4_f32_fp8_fp8 v48, v23, v15, v4
	s_addk_co_i32 s2, 0x200
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s2, s12
	s_cbranch_scc1 .LBB0_97
.LBB0_94:                               ;   Parent Loop BB0_89 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_dual_mov_b32 v25, 0 :: v_dual_add_nc_u32 v60, s2, v39
	s_wait_dscnt 0x9
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v17, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_min_u32_e32 v0, s24, v60
	s_wait_dscnt 0x5
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v21, 0
	v_add_co_u32 v2, s26, s16, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v3, null, s17, 0, s26
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v23, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, v2, v30
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v3, v31, vcc_lo
	v_add_co_u32 v4, vcc_lo, v2, v32
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v3, v33, vcc_lo
	v_add_co_u32 v8, vcc_lo, v2, v34
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, v3, v35, vcc_lo
	v_add_co_u32 v12, vcc_lo, v2, v36
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v13, null, v3, v37, vcc_lo
	s_clause 0x3
	global_load_b128 v[0:3], v[0:1], off
	global_load_b128 v[4:7], v[4:5], off
	global_load_b128 v[8:11], v[8:9], off
	global_load_b128 v[12:15], v[12:13], off
	v_mov_b32_e32 v22, 0
	s_mov_b32 s26, exec_lo
	v_cmpx_gt_u32_e64 s12, v60
	s_cbranch_execz .LBB0_93
; %bb.95:                               ;   in Loop: Header=BB0_94 Depth=2
	v_add_nc_u32_e32 v16, s2, v41
	v_add_nc_u32_e32 v20, s2, v38
	ds_load_b128 v[24:27], v60
	ds_load_b128 v[16:19], v16
	ds_load_b128 v[20:23], v20
	s_branch .LBB0_93
.LBB0_96:                               ;   in Loop: Header=BB0_89 Depth=1
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v51, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v55, 0
.LBB0_97:                               ;   in Loop: Header=BB0_89 Depth=1
	v_add_f32_dpp v3, v57, v57 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v56, v56 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3)
	v_add_f32_dpp v5, v55, v55 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v54, v54 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v53, v53 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v59, v59 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v58, v58 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v23, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v21, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v19, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v14, v7, v7 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v52, v52 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v51, v51 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v50, v50 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v49, v49 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v48, v48 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v42
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v2, v40, v42, vcc_lo
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshlrev_b32_e32 v2, 2, v2
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v25, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v7, v7 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v1, v2, v0
	ds_bpermute_b32 v26, v2, v25
	ds_bpermute_b32 v24, v2, v23
	ds_bpermute_b32 v22, v2, v21
	ds_bpermute_b32 v20, v2, v19
	ds_bpermute_b32 v18, v2, v17
	ds_bpermute_b32 v16, v2, v14
	ds_bpermute_b32 v15, v2, v13
	ds_bpermute_b32 v11, v2, v10
	ds_bpermute_b32 v9, v2, v8
	ds_bpermute_b32 v7, v2, v6
	ds_bpermute_b32 v5, v2, v4
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
	s_and_saveexec_b32 s26, s1
	s_cbranch_execz .LBB0_88
; %bb.98:                               ;   in Loop: Header=BB0_89 Depth=1
	v_lshlrev_b64_e32 v[2:3], 1, v[28:29]
	s_wait_dscnt 0xb
	v_add_f32_e32 v12, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v27, s30, v12
	v_add_co_u32 v0, vcc_lo, s4, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s5, v3, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_100
; %bb.99:                               ;   in Loop: Header=BB0_89 Depth=1
	global_load_u16 v12, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v12, 16, v12
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v27, s9, v12
.LBB0_100:                              ;   in Loop: Header=BB0_89 Depth=1
	v_lshlrev_b32_e32 v12, 16, v47
	v_add_co_u32 v2, vcc_lo, s6, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v30, v27, v12
	v_cmp_gt_u32_e64 s2, s13, v46
	v_cndmask_b32_e64 v27, v30, v27, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v30, v27, 16, 1
	v_or_b32_e32 v31, 0x400000, v27
	v_cmp_u_f32_e32 vcc_lo, v27, v27
	v_add3_u32 v30, v30, v27, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v27, v30, v31, vcc_lo
	global_store_d16_hi_b16 v[2:3], v27, off
	s_and_saveexec_b32 s27, s2
	s_cbranch_execz .LBB0_110
; %bb.101:                              ;   in Loop: Header=BB0_89 Depth=1
	s_wait_dscnt 0xa
	v_add_f32_e32 v25, v25, v26
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v25, s30, v25
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_103
; %bb.102:                              ;   in Loop: Header=BB0_89 Depth=1
	global_load_u16 v26, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v26, 16, v26
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v25, s9, v26
.LBB0_103:                              ;   in Loop: Header=BB0_89 Depth=1
	v_lshlrev_b32_e32 v26, 16, v45
	v_or_b32_e32 v30, 2, v28
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v26, v25, v26
	v_cndmask_b32_e64 v25, v26, v25, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v26, v25, 16, 1
	v_or_b32_e32 v27, 0x400000, v25
	v_cmp_u_f32_e32 vcc_lo, v25, v25
	v_add3_u32 v26, v26, v25, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v25, v26, v27, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, s13, v30
	global_store_d16_hi_b16 v[2:3], v25, off offset:2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_110
; %bb.104:                              ;   in Loop: Header=BB0_89 Depth=1
	s_wait_dscnt 0x9
	v_add_f32_e32 v23, v23, v24
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v23, s30, v23
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_106
; %bb.105:                              ;   in Loop: Header=BB0_89 Depth=1
	global_load_u16 v24, v[0:1], off offset:4
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v24, 16, v24
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v23, s9, v24
.LBB0_106:                              ;   in Loop: Header=BB0_89 Depth=1
	v_lshlrev_b32_e32 v24, 16, v44
	v_or_b32_e32 v26, 3, v28
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v24, v23, v24
	v_cndmask_b32_e64 v23, v24, v23, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v24, v23, 16, 1
	v_or_b32_e32 v25, 0x400000, v23
	v_cmp_u_f32_e32 vcc_lo, v23, v23
	v_add3_u32 v24, v24, v23, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v23, v24, v25, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, s13, v26
	global_store_d16_hi_b16 v[2:3], v23, off offset:4
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_110
; %bb.107:                              ;   in Loop: Header=BB0_89 Depth=1
	s_wait_dscnt 0x8
	v_add_f32_e32 v21, v21, v22
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v21, s30, v21
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_109
; %bb.108:                              ;   in Loop: Header=BB0_89 Depth=1
	global_load_u16 v22, v[0:1], off offset:6
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v22, 16, v22
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v21, s9, v22
.LBB0_109:                              ;   in Loop: Header=BB0_89 Depth=1
	v_lshlrev_b32_e32 v22, 16, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v22, v21, v22
	v_cndmask_b32_e64 v21, v22, v21, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v22, v21, 16, 1
	v_or_b32_e32 v23, 0x400000, v21
	v_cmp_u_f32_e32 vcc_lo, v21, v21
	v_add3_u32 v22, v22, v21, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v21, v22, v23, vcc_lo
	global_store_d16_hi_b16 v[2:3], v21, off offset:6
.LBB0_110:                              ;   in Loop: Header=BB0_89 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s27
	s_wait_dscnt 0x7
	v_add_f32_e32 v19, v19, v20
	v_add_co_u32 v0, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v19, s30, v19
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_112
; %bb.111:                              ;   in Loop: Header=BB0_89 Depth=1
	global_load_u16 v20, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v20, 16, v20
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v19, s9, v20
.LBB0_112:                              ;   in Loop: Header=BB0_89 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v20, v19, v12
	v_add_co_u32 v2, vcc_lo, v2, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	v_cndmask_b32_e64 v19, v20, v19, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v20, v19, 16, 1
	v_or_b32_e32 v21, 0x400000, v19
	v_cmp_u_f32_e32 vcc_lo, v19, v19
	v_add3_u32 v20, v20, v19, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v19, v20, v21, vcc_lo
	global_store_d16_hi_b16 v[2:3], v19, off
	s_and_saveexec_b32 s27, s2
	s_cbranch_execz .LBB0_122
; %bb.113:                              ;   in Loop: Header=BB0_89 Depth=1
	s_wait_dscnt 0x6
	v_add_f32_e32 v17, v17, v18
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v17, s30, v17
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_115
; %bb.114:                              ;   in Loop: Header=BB0_89 Depth=1
	global_load_u16 v18, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v18, 16, v18
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v17, s9, v18
.LBB0_115:                              ;   in Loop: Header=BB0_89 Depth=1
	v_lshlrev_b32_e32 v18, 16, v45
	v_or_b32_e32 v20, 2, v28
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v18, v17, v18
	v_cndmask_b32_e64 v17, v18, v17, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v18, v17, 16, 1
	v_or_b32_e32 v19, 0x400000, v17
	v_cmp_u_f32_e32 vcc_lo, v17, v17
	v_add3_u32 v18, v18, v17, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v17, v18, v19, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, s13, v20
	global_store_d16_hi_b16 v[2:3], v17, off offset:2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_122
; %bb.116:                              ;   in Loop: Header=BB0_89 Depth=1
	s_wait_dscnt 0x5
	v_add_f32_e32 v14, v14, v16
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v14, s30, v14
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_118
; %bb.117:                              ;   in Loop: Header=BB0_89 Depth=1
	global_load_u16 v16, v[0:1], off offset:4
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v16, 16, v16
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v14, s9, v16
.LBB0_118:                              ;   in Loop: Header=BB0_89 Depth=1
	v_lshlrev_b32_e32 v16, 16, v44
	v_or_b32_e32 v18, 3, v28
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v16, v14, v16
	v_cndmask_b32_e64 v14, v16, v14, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v16, v14, 16, 1
	v_or_b32_e32 v17, 0x400000, v14
	v_cmp_u_f32_e32 vcc_lo, v14, v14
	v_add3_u32 v16, v16, v14, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v14, v16, v17, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, s13, v18
	global_store_d16_hi_b16 v[2:3], v14, off offset:4
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_122
; %bb.119:                              ;   in Loop: Header=BB0_89 Depth=1
	s_wait_dscnt 0x4
	v_add_f32_e32 v13, v13, v15
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v13, s30, v13
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_121
; %bb.120:                              ;   in Loop: Header=BB0_89 Depth=1
	global_load_u16 v14, v[0:1], off offset:6
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v14, 16, v14
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v13, s9, v14
.LBB0_121:                              ;   in Loop: Header=BB0_89 Depth=1
	v_lshlrev_b32_e32 v14, 16, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v14, v13, v14
	v_cndmask_b32_e64 v13, v14, v13, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v14, v13, 16, 1
	v_or_b32_e32 v15, 0x400000, v13
	v_cmp_u_f32_e32 vcc_lo, v13, v13
	v_add3_u32 v14, v14, v13, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v13, v14, v15, vcc_lo
	global_store_d16_hi_b16 v[2:3], v13, off offset:6
.LBB0_122:                              ;   in Loop: Header=BB0_89 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s27
	s_wait_dscnt 0x3
	v_add_f32_e32 v10, v10, v11
	v_add_co_u32 v0, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v10, s30, v10
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_124
; %bb.123:                              ;   in Loop: Header=BB0_89 Depth=1
	global_load_u16 v11, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v11, 16, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v10, s9, v11
.LBB0_124:                              ;   in Loop: Header=BB0_89 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v11, v10, v12
	v_add_co_u32 v2, vcc_lo, v2, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	v_cndmask_b32_e64 v10, v11, v10, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v11, v10, 16, 1
	v_or_b32_e32 v12, 0x400000, v10
	v_cmp_u_f32_e32 vcc_lo, v10, v10
	v_add3_u32 v11, v11, v10, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v10, v11, v12, vcc_lo
	global_store_d16_hi_b16 v[2:3], v10, off
	s_and_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB0_88
; %bb.125:                              ;   in Loop: Header=BB0_89 Depth=1
	s_wait_dscnt 0x2
	v_add_f32_e32 v8, v8, v9
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v8, s30, v8
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_127
; %bb.126:                              ;   in Loop: Header=BB0_89 Depth=1
	global_load_u16 v9, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v9, 16, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, s9, v9
.LBB0_127:                              ;   in Loop: Header=BB0_89 Depth=1
	v_lshlrev_b32_e32 v9, 16, v45
	v_or_b32_e32 v11, 2, v28
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v8, v9
	v_cndmask_b32_e64 v8, v9, v8, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v9, v8, 16, 1
	v_or_b32_e32 v10, 0x400000, v8
	v_cmp_u_f32_e32 vcc_lo, v8, v8
	v_add3_u32 v9, v9, v8, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v8, v9, v10, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, s13, v11
	global_store_d16_hi_b16 v[2:3], v8, off offset:2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_88
; %bb.128:                              ;   in Loop: Header=BB0_89 Depth=1
	s_wait_dscnt 0x1
	v_add_f32_e32 v6, v6, v7
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v6, s30, v6
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_130
; %bb.129:                              ;   in Loop: Header=BB0_89 Depth=1
	global_load_u16 v7, v[0:1], off offset:4
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v7, 16, v7
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v6, s9, v7
.LBB0_130:                              ;   in Loop: Header=BB0_89 Depth=1
	v_lshlrev_b32_e32 v7, 16, v44
	v_or_b32_e32 v9, 3, v28
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v7, v6, v7
	v_cndmask_b32_e64 v6, v7, v6, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v7, v6, 16, 1
	v_or_b32_e32 v8, 0x400000, v6
	v_cmp_u_f32_e32 vcc_lo, v6, v6
	v_add3_u32 v7, v7, v6, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v6, v7, v8, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, s13, v9
	global_store_d16_hi_b16 v[2:3], v6, off offset:4
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_88
; %bb.131:                              ;   in Loop: Header=BB0_89 Depth=1
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v4, s30, v4
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_87
; %bb.132:                              ;   in Loop: Header=BB0_89 Depth=1
	global_load_u16 v0, v[0:1], off offset:6
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v0, 16, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v4, s9, v0
	s_branch .LBB0_87
.LBB0_133:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201
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
		.amdhsa_next_free_sgpr 42
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 71
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
	.size	wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201, .Lfunc_end0-wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201
                                        ; -- End function
	.set wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201.num_vgpr, 63
	.set wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201.num_agpr, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201.numbered_sgpr, 42
	.set wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201.num_named_barrier, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201.private_seg_size, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201.uses_vcc, 1
	.set wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201.uses_flat_scratch, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201.has_dyn_sized_stack, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201.has_recursion, 0
	.set wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9076
; TotalNumSgprs: 44
; NumVgprs: 63
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 21
; NumSGPRsForWavesPerEU: 44
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
	.type	__hip_cuid_6f51406bf1035b69,@object ; @__hip_cuid_6f51406bf1035b69
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_6f51406bf1035b69
__hip_cuid_6f51406bf1035b69:
	.byte	0                               ; 0x0
	.size	__hip_cuid_6f51406bf1035b69, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_6f51406bf1035b69
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
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: False
    UseBeta: True
    Batched: True
    UseScaleAB: Scalar
    UseBias: 1
    BiasSrc: D
    BiasDataTypeList: ['b']
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
  AssertSizeEqual: { 1: 3, 2: 1 }
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
    .name:           wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     44
    .sgpr_spill_count: 0
    .symbol:         wvSplitKQ_vllm_f8_bf16_tn_n3_y4u1_gfx1201.kd
    .vgpr_count:     63
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
