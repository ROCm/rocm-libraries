	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201 ; -- Begin function wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201
	.globl	wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201
	.p2align	8
	.type	wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201,@function
wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201: ; @wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201
; %bb.0:
	s_clause 0x1
	s_load_b64 s[2:3], s[0:1], 0x0
	s_load_b64 s[14:15], s[0:1], 0x10
	v_and_b32_e32 v52, 0x3ff, v0
	v_bfe_u32 v2, v0, 10, 10
	s_mov_b32 s18, 4
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v51, 4, v52
	v_lshl_add_u32 v3, v2, 9, v51
	s_wait_kmcnt 0x0
	s_min_u32 s5, s2, 0x10000
	s_delay_alu instid0(VALU_DEP_1) | instid1(SALU_CYCLE_1)
	v_cmpx_gt_u32_e64 s5, v3
	s_cbranch_execz .LBB0_9
; %bb.1:
	global_load_b128 v[4:7], v3, s[14:15]
	v_add_nc_u32_e32 v0, 0x2000, v3
	s_mov_b32 s6, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7]
	v_cmpx_gt_u32_e64 s5, v0
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_9
; %bb.2:
	v_add_co_u32 v0, s6, s14, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v1, null, s15, 0, s6
	v_add_nc_u32_e32 v8, 0x4000, v3
	s_mov_b32 s6, exec_lo
	global_load_b128 v[4:7], v[0:1], off offset:8192
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:8192
	v_cmpx_gt_u32_e64 s5, v8
	s_wait_alu 0xfffe
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_9
; %bb.3:
	global_load_b128 v[4:7], v[0:1], off offset:16384
	v_add_nc_u32_e32 v8, 0x6000, v3
	s_mov_b32 s6, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:16384
	v_cmpx_gt_u32_e64 s5, v8
	s_wait_alu 0xfffe
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_9
; %bb.4:
	global_load_b128 v[4:7], v[0:1], off offset:24576
	v_add_nc_u32_e32 v8, 0x8000, v3
	s_mov_b32 s6, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:24576
	v_cmpx_gt_u32_e64 s5, v8
	s_wait_alu 0xfffe
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_9
; %bb.5:
	global_load_b128 v[4:7], v[0:1], off offset:32768
	v_add_nc_u32_e32 v8, 0xa000, v3
	s_mov_b32 s6, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:32768
	v_cmpx_gt_u32_e64 s5, v8
	s_wait_alu 0xfffe
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_9
; %bb.6:
	global_load_b128 v[4:7], v[0:1], off offset:40960
	v_add_nc_u32_e32 v8, 0xc000, v3
	s_mov_b32 s6, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:40960
	v_cmpx_gt_u32_e64 s5, v8
	s_wait_alu 0xfffe
	s_xor_b32 s6, exec_lo, s6
	s_cbranch_execz .LBB0_9
; %bb.7:
	global_load_b128 v[4:7], v[0:1], off offset:49152
	v_add_nc_u32_e32 v8, 0xe000, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, s5, v8
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:49152
	s_and_saveexec_b32 s5, vcc_lo
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.8:
	global_load_b128 v[4:7], v[0:1], off offset:57344
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:57344
.LBB0_9:
	s_or_b32 exec_lo, exec_lo, s4
	s_clause 0x1
	s_load_b32 s19, s[0:1], 0x18
	s_load_b64 s[4:5], s[0:1], 0x48
	s_add_co_i32 s16, s3, -1
	s_wait_kmcnt 0x0
	s_lshl_b32 s6, s19, 6
	s_wait_alu 0xfffe
	v_add_co_u32 v0, s8, s6, s16
	s_and_not1_b32 vcc_lo, exec_lo, s8
	v_readfirstlane_b32 s7, v0
	s_cbranch_vccz .LBB0_11
; %bb.10:
	s_cvt_f32_u32 s8, s6
	s_sub_co_i32 s9, 0, s6
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v0, s8
	v_readfirstlane_b32 s8, v0
	s_mul_f32 s8, s8, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s8, s8
	s_wait_alu 0xfffe
	s_mul_i32 s9, s9, s8
	s_wait_alu 0xfffe
	s_mul_hi_u32 s9, s8, s9
	s_wait_alu 0xfffe
	s_add_co_i32 s8, s8, s9
	s_wait_alu 0xfffe
	s_mul_hi_u32 s8, s7, s8
	s_wait_alu 0xfffe
	s_mul_i32 s9, s8, s6
	s_wait_alu 0xfffe
	s_sub_co_i32 s7, s7, s9
	s_add_co_i32 s9, s8, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s10, s7, s6
	s_cmp_ge_u32 s7, s6
	s_cselect_b32 s8, s9, s8
	s_cselect_b32 s7, s10, s7
	s_wait_alu 0xfffe
	s_add_co_i32 s9, s8, 1
	s_cmp_ge_u32 s7, s6
	s_wait_alu 0xfffe
	s_cselect_b32 s6, s9, s8
	s_wait_alu 0xfffe
	s_mul_i32 s6, s19, s6
	s_wait_alu 0xfffe
	s_lshl_b32 s6, s6, 2
	s_wait_alu 0xfffe
	s_cvt_f32_u32 s7, s6
	s_sub_co_i32 s8, 0, s6
	s_add_co_i32 s9, s6, s16
	s_wait_alu 0xfffe
	v_rcp_iflag_f32_e32 v0, s7
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_2)
	v_readfirstlane_b32 s7, v0
	s_mul_f32 s7, s7, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_cvt_u32_f32 s7, s7
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2)
	s_mul_i32 s8, s8, s7
	s_wait_alu 0xfffe
	s_mul_hi_u32 s8, s7, s8
	s_wait_alu 0xfffe
	s_add_co_i32 s7, s7, s8
	s_wait_alu 0xfffe
	s_mul_hi_u32 s7, s9, s7
	s_wait_alu 0xfffe
	s_mul_i32 s8, s7, s6
	s_wait_alu 0xfffe
	s_sub_co_i32 s8, s9, s8
	s_add_co_i32 s9, s7, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s10, s8, s6
	s_cmp_ge_u32 s8, s6
	s_cselect_b32 s7, s9, s7
	s_cselect_b32 s8, s10, s8
	s_wait_alu 0xfffe
	s_add_co_i32 s9, s7, 1
	s_cmp_ge_u32 s8, s6
	s_wait_alu 0xfffe
	s_cselect_b32 s6, s9, s7
	s_wait_alu 0xfffe
	s_max_u32 s18, s6, 4
.LBB0_11:
	s_load_b64 s[6:7], s[0:1], 0x50
	s_mov_b32 s20, 1.0
	s_cmp_eq_u64 s[4:5], 0
	s_mov_b32 s21, 1.0
	s_cbranch_scc1 .LBB0_13
; %bb.12:
	s_load_b32 s21, s[4:5], 0x0
.LBB0_13:
	s_wait_kmcnt 0x0
	s_cmp_eq_u64 s[6:7], 0
	s_cbranch_scc1 .LBB0_15
; %bb.14:
	s_load_b32 s20, s[6:7], 0x0
.LBB0_15:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_mov_b32 s4, exec_lo
	v_cmpx_gt_u32_e64 s18, v2
	s_cbranch_execz .LBB0_93
; %bb.16:
	s_clause 0x4
	s_load_b64 s[8:9], s[0:1], 0x38
	s_load_b128 s[4:7], s[0:1], 0x28
	s_load_b64 s[10:11], s[0:1], 0x58
	s_load_b64 s[12:13], s[0:1], 0x8
	s_load_b32 s17, s[0:1], 0x20
	s_mul_i32 s0, s18, ttmp9
	s_mul_i32 s19, s19, s18
	v_add_lshl_u32 v40, s0, v2, 2
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, s3, v40
	s_wait_kmcnt 0x0
	s_mul_f32 s0, s8, s21
	s_wait_alu 0xfffe
	s_lshl_b32 s8, s19, 2
	s_cmp_gt_u32 s2, 0x10000
	s_mul_f32 s18, s0, s20
	s_mov_b32 s0, -1
	s_cbranch_scc0 .LBB0_60
; %bb.17:
	s_and_saveexec_b32 s19, vcc_lo
	s_cbranch_execz .LBB0_59
; %bb.18:
	s_cmp_eq_u64 s[10:11], 0
	v_cmp_eq_u32_e64 s1, 31, v52
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[10:11], 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v43, v40
	s_cselect_b32 s20, -1, 0
	s_add_co_i32 s21, s2, -16
	s_cmp_neq_f32 s9, 0
	v_mbcnt_lo_u32_b32 v53, -1, 0
	s_mov_b32 s25, 0
	s_cselect_b32 s22, -1, 0
	s_add_co_i32 s24, s2, -1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_lshr_b32 s23, s24, 9
	s_add_co_i32 s23, s23, 1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 s23, s23, 0xfffffe
	s_bitcmp0_b32 s24, 9
	s_cselect_b32 s24, -1, 0
	s_branch .LBB0_21
.LBB0_19:                               ;   in Loop: Header=BB0_21 Depth=1
	v_cvt_f32_f16_e32 v0, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v4, v0
	v_cndmask_b32_e64 v0, v0, v4, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v0, v0
	global_store_b16 v[2:3], v0, off offset:6
.LBB0_20:                               ;   in Loop: Header=BB0_21 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s26
	v_add_nc_u32_e32 v43, s8, v43
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)
	v_cmp_le_u32_e32 vcc_lo, s3, v43
	s_or_b32 s25, vcc_lo, s25
	s_and_not1_b32 exec_lo, exec_lo, s25
	s_cbranch_execz .LBB0_59
.LBB0_21:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_28 Depth 2
	v_min_u32_e32 v41, s16, v43
	v_or_b32_e32 v6, 1, v43
	s_wait_dscnt 0x1
	v_or_b32_e32 v7, 2, v43
	v_or_b32_e32 v8, 3, v43
	s_and_b32 vcc_lo, exec_lo, s20
	s_mov_b32 s26, -1
                                        ; implicit-def: $vgpr54
                                        ; implicit-def: $vgpr57
                                        ; implicit-def: $vgpr59
                                        ; implicit-def: $vgpr60
                                        ; implicit-def: $vgpr58
                                        ; implicit-def: $vgpr0_vgpr1
                                        ; implicit-def: $vgpr56
                                        ; implicit-def: $vgpr2_vgpr3
                                        ; implicit-def: $vgpr55
                                        ; implicit-def: $vgpr4_vgpr5
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB0_23
; %bb.22:                               ;   in Loop: Header=BB0_21 Depth=1
	v_or_b32_e32 v58, 1, v43
	s_wait_dscnt 0x0
	v_lshlrev_b64_e32 v[4:5], 1, v[41:42]
	v_or_b32_e32 v56, 2, v43
	v_or_b32_e32 v55, 3, v43
	v_mov_b32_e32 v1, v42
	v_min_u32_e32 v0, s16, v58
	v_mov_b32_e32 v3, v42
	v_min_u32_e32 v2, s16, v56
	v_add_co_u32 v11, vcc_lo, s10, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v12, null, s11, v5, vcc_lo
	v_min_u32_e32 v4, s16, v55
	v_mov_b32_e32 v5, v42
	v_lshlrev_b64_e32 v[9:10], 1, v[0:1]
	v_lshlrev_b64_e32 v[13:14], 1, v[2:3]
	s_mov_b32 s26, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[15:16], 1, v[4:5]
	v_add_co_u32 v9, vcc_lo, s10, v9
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v10, null, s11, v10, vcc_lo
	v_add_co_u32 v13, vcc_lo, s10, v13
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v14, null, s11, v14, vcc_lo
	v_add_co_u32 v15, vcc_lo, s10, v15
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v16, null, s11, v16, vcc_lo
	s_clause 0x3
	global_load_u16 v60, v[11:12], off
	global_load_u16 v59, v[9:10], off
	global_load_u16 v57, v[13:14], off
	global_load_u16 v54, v[15:16], off
.LBB0_23:                               ;   in Loop: Header=BB0_21 Depth=1
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_25
; %bb.24:                               ;   in Loop: Header=BB0_21 Depth=1
	v_min_u32_e32 v0, s16, v6
	s_wait_loadcnt 0x3
	v_dual_mov_b32 v1, v42 :: v_dual_mov_b32 v60, 0
	v_min_u32_e32 v2, s16, v7
	v_dual_mov_b32 v3, v42 :: v_dual_mov_b32 v56, v7
	v_min_u32_e32 v4, s16, v8
	s_wait_dscnt 0x0
	v_dual_mov_b32 v5, v42 :: v_dual_mov_b32 v58, v6
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v55, v8 :: v_dual_mov_b32 v54, 0
	v_mov_b32_e32 v59, 0
	v_mov_b32_e32 v57, 0
.LBB0_25:                               ;   in Loop: Header=BB0_21 Depth=1
	v_mad_co_u64_u32 v[48:49], null, v0, s17, 0
	v_mad_co_u64_u32 v[46:47], null, v2, s17, 0
	v_mad_co_u64_u32 v[44:45], null, v4, s17, 0
	s_mov_b32 s26, 0
	s_mov_b32 s27, s23
	v_dual_mov_b32 v0, v49 :: v_dual_mov_b32 v61, 0
	v_dual_mov_b32 v2, v47 :: v_dual_mov_b32 v47, 0
	v_mad_co_u64_u32 v[49:50], null, v41, s17, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mad_co_u64_u32 v[0:1], null, v1, s17, v[0:1]
	v_mad_co_u64_u32 v[1:2], null, v3, s17, v[2:3]
	v_dual_mov_b32 v4, v45 :: v_dual_mov_b32 v45, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v64, v0
	v_mov_b32_e32 v63, v1
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[2:3], null, v5, s17, v[4:5]
	v_mov_b32_e32 v62, v2
	s_branch .LBB0_28
.LBB0_26:                               ;   in Loop: Header=BB0_28 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s29
.LBB0_27:                               ;   in Loop: Header=BB0_28 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s28
	s_wait_loadcnt_dscnt 0x400
	v_dot4_f32_fp8_fp8 v0, v16, v0, v61
	v_dot4_f32_fp8_fp8 v4, v16, v4, v47
	v_dot4_f32_fp8_fp8 v8, v16, v8, v45
	v_dot4_f32_fp8_fp8 v12, v16, v12, v41
	s_add_co_i32 s27, s27, -2
	v_dot4_f32_fp8_fp8 v0, v17, v1, v0
	v_dot4_f32_fp8_fp8 v1, v17, v5, v4
	v_dot4_f32_fp8_fp8 v4, v17, v9, v8
	v_dot4_f32_fp8_fp8 v5, v17, v13, v12
	s_addk_co_i32 s26, 0x400
	v_dot4_f32_fp8_fp8 v0, v18, v2, v0
	v_dot4_f32_fp8_fp8 v1, v18, v6, v1
	v_dot4_f32_fp8_fp8 v2, v18, v10, v4
	v_dot4_f32_fp8_fp8 v4, v18, v14, v5
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s27, 0
	v_dot4_f32_fp8_fp8 v0, v19, v3, v0
	v_dot4_f32_fp8_fp8 v1, v19, v7, v1
	v_dot4_f32_fp8_fp8 v2, v19, v11, v2
	v_dot4_f32_fp8_fp8 v3, v19, v15, v4
	s_wait_loadcnt 0x0
	v_dot4_f32_fp8_fp8 v0, v36, v20, v0
	v_dot4_f32_fp8_fp8 v1, v36, v24, v1
	v_dot4_f32_fp8_fp8 v2, v36, v28, v2
	v_dot4_f32_fp8_fp8 v3, v36, v32, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v0, v37, v21, v0
	v_dot4_f32_fp8_fp8 v1, v37, v25, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v2, v37, v29, v2
	v_dot4_f32_fp8_fp8 v3, v37, v33, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v0, v38, v22, v0
	v_dot4_f32_fp8_fp8 v1, v38, v26, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v2, v38, v30, v2
	v_dot4_f32_fp8_fp8 v3, v38, v34, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v61, v39, v23, v0
	v_dot4_f32_fp8_fp8 v47, v39, v27, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v45, v39, v31, v2
	v_dot4_f32_fp8_fp8 v41, v39, v35, v3
	s_cbranch_scc1 .LBB0_39
.LBB0_28:                               ;   Parent Loop BB0_21 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_dual_mov_b32 v18, 0 :: v_dual_add_nc_u32 v65, s26, v51
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v16, 0
	v_mov_b32_e32 v17, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v0, s21, v65
	v_add_co_u32 v2, s28, s12, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v3, null, s13, 0, s28
	s_mov_b32 s28, exec_lo
	v_add_co_u32 v0, vcc_lo, v2, v49
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v3, v50, vcc_lo
	v_add_co_u32 v4, vcc_lo, v2, v48
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v3, v64, vcc_lo
	v_add_co_u32 v8, vcc_lo, v2, v46
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, v3, v63, vcc_lo
	v_add_co_u32 v12, vcc_lo, v2, v44
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v13, null, v3, v62, vcc_lo
	s_clause 0x3
	global_load_b128 v[0:3], v[0:1], off
	global_load_b128 v[4:7], v[4:5], off
	global_load_b128 v[8:11], v[8:9], off
	global_load_b128 v[12:15], v[12:13], off
	v_cmpx_gt_u32_e64 s2, v65
	s_cbranch_execz .LBB0_34
; %bb.29:                               ;   in Loop: Header=BB0_28 Depth=2
	s_mov_b32 s29, exec_lo
                                        ; implicit-def: $vgpr16
	v_cmpx_lt_u32_e32 0xffff, v65
	s_wait_alu 0xfffe
	s_xor_b32 s29, exec_lo, s29
	s_cbranch_execz .LBB0_31
; %bb.30:                               ;   in Loop: Header=BB0_28 Depth=2
	global_load_b128 v[16:19], v65, s[14:15]
.LBB0_31:                               ;   in Loop: Header=BB0_28 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s29, s29
	s_cbranch_execz .LBB0_33
; %bb.32:                               ;   in Loop: Header=BB0_28 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[16:19], v65
.LBB0_33:                               ;   in Loop: Header=BB0_28 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s29
.LBB0_34:                               ;   in Loop: Header=BB0_28 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s28
	v_dual_mov_b32 v39, 0 :: v_dual_add_nc_u32 v66, 0x200, v65
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v37, 0
	v_mov_b32_e32 v36, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v20, s21, v66
	v_add_co_u32 v22, s28, s12, v20
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v23, null, s13, 0, s28
	s_mov_b32 s28, exec_lo
	v_add_co_u32 v20, vcc_lo, v22, v49
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v21, null, v23, v50, vcc_lo
	v_add_co_u32 v24, vcc_lo, v22, v48
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v25, null, v23, v64, vcc_lo
	v_add_co_u32 v28, vcc_lo, v22, v46
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v29, null, v23, v63, vcc_lo
	v_add_co_u32 v32, vcc_lo, v22, v44
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v33, null, v23, v62, vcc_lo
	s_clause 0x3
	global_load_b128 v[20:23], v[20:21], off
	global_load_b128 v[24:27], v[24:25], off
	global_load_b128 v[28:31], v[28:29], off
	global_load_b128 v[32:35], v[32:33], off
	v_cmpx_gt_u32_e64 s2, v66
	s_cbranch_execz .LBB0_27
; %bb.35:                               ;   in Loop: Header=BB0_28 Depth=2
	s_mov_b32 s29, exec_lo
                                        ; implicit-def: $vgpr36
	v_cmpx_lt_u32_e32 0xffff, v66
	s_wait_alu 0xfffe
	s_xor_b32 s29, exec_lo, s29
	s_cbranch_execz .LBB0_37
; %bb.36:                               ;   in Loop: Header=BB0_28 Depth=2
	global_load_b128 v[36:39], v66, s[14:15]
                                        ; implicit-def: $vgpr65
.LBB0_37:                               ;   in Loop: Header=BB0_28 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s29, s29
	s_cbranch_execz .LBB0_26
; %bb.38:                               ;   in Loop: Header=BB0_28 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[36:39], v65 offset:512
	s_branch .LBB0_26
.LBB0_39:                               ;   in Loop: Header=BB0_21 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_47
; %bb.40:                               ;   in Loop: Header=BB0_21 Depth=1
	v_dual_mov_b32 v19, 0 :: v_dual_add_nc_u32 v20, s26, v51
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v17, 0
	v_mov_b32_e32 v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v0, s21, v20
	v_add_co_u32 v2, s26, s12, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v3, null, s13, 0, s26
	s_mov_b32 s26, exec_lo
	v_add_co_u32 v0, vcc_lo, v2, v49
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v3, v50, vcc_lo
	v_add_co_u32 v4, vcc_lo, v2, v48
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v3, v64, vcc_lo
	v_add_co_u32 v8, vcc_lo, v2, v46
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, v3, v63, vcc_lo
	v_add_co_u32 v12, vcc_lo, v2, v44
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v13, null, v3, v62, vcc_lo
	s_clause 0x3
	global_load_b128 v[0:3], v[0:1], off
	global_load_b128 v[4:7], v[4:5], off
	global_load_b128 v[8:11], v[8:9], off
	global_load_b128 v[12:15], v[12:13], off
	v_cmpx_gt_u32_e64 s2, v20
	s_cbranch_execz .LBB0_46
; %bb.41:                               ;   in Loop: Header=BB0_21 Depth=1
	s_mov_b32 s27, exec_lo
                                        ; implicit-def: $vgpr16
	v_cmpx_lt_u32_e32 0xffff, v20
	s_wait_alu 0xfffe
	s_xor_b32 s27, exec_lo, s27
	s_cbranch_execz .LBB0_43
; %bb.42:                               ;   in Loop: Header=BB0_21 Depth=1
	global_load_b128 v[16:19], v20, s[14:15]
                                        ; implicit-def: $vgpr20
.LBB0_43:                               ;   in Loop: Header=BB0_21 Depth=1
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s27, s27
	s_cbranch_execz .LBB0_45
; %bb.44:                               ;   in Loop: Header=BB0_21 Depth=1
	s_wait_loadcnt 0x0
	ds_load_b128 v[16:19], v20
.LBB0_45:                               ;   in Loop: Header=BB0_21 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s27
.LBB0_46:                               ;   in Loop: Header=BB0_21 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s26
	s_wait_loadcnt_dscnt 0x0
	v_dot4_f32_fp8_fp8 v0, v16, v0, v61
	v_dot4_f32_fp8_fp8 v4, v16, v4, v47
	v_dot4_f32_fp8_fp8 v8, v16, v8, v45
	v_dot4_f32_fp8_fp8 v12, v16, v12, v41
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v0, v17, v1, v0
	v_dot4_f32_fp8_fp8 v1, v17, v5, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v4, v17, v9, v8
	v_dot4_f32_fp8_fp8 v5, v17, v13, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v0, v18, v2, v0
	v_dot4_f32_fp8_fp8 v1, v18, v6, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v2, v18, v10, v4
	v_dot4_f32_fp8_fp8 v4, v18, v14, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v61, v19, v3, v0
	v_dot4_f32_fp8_fp8 v47, v19, v7, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v45, v19, v11, v2
	v_dot4_f32_fp8_fp8 v41, v19, v15, v4
.LBB0_47:                               ;   in Loop: Header=BB0_21 Depth=1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_dpp v0, v61, v61 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v47, v47 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_dpp v2, v45, v45 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v41, v41 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_xor_b32_e32 v4, 16, v53
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v4
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v4, v53, v4, vcc_lo
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v5, 2, v4
	v_add_f32_dpp v4, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	;;#ASMSTART
	;;#ASMEND
	ds_bpermute_b32 v1, v5, v0
	ds_bpermute_b32 v9, v5, v8
	ds_bpermute_b32 v7, v5, v6
	ds_bpermute_b32 v5, v5, v4
	s_and_saveexec_b32 s26, s1
	s_cbranch_execz .LBB0_20
; %bb.48:                               ;   in Loop: Header=BB0_21 Depth=1
	v_mov_b32_e32 v44, v42
	s_wait_dscnt 0x3
	v_add_f32_e32 v10, v0, v1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 1, v[43:44]
	v_mul_f32_e32 v10, s18, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s4, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s5, v3, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s22
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_50
; %bb.49:                               ;   in Loop: Header=BB0_21 Depth=1
	global_load_u16 v11, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v10, s9, v11, v10 op_sel_hi:[0,1,0]
.LBB0_50:                               ;   in Loop: Header=BB0_21 Depth=1
	v_cvt_f32_f16_e32 v11, v60
	v_add_co_u32 v2, vcc_lo, s6, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v11, v10, v11
	v_cmp_gt_u32_e32 vcc_lo, s3, v58
	v_cndmask_b32_e64 v10, v11, v10, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v10, v10
	global_store_b16 v[2:3], v10, off
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_20
; %bb.51:                               ;   in Loop: Header=BB0_21 Depth=1
	s_wait_dscnt 0x2
	v_add_f32_e32 v8, v8, v9
	s_and_not1_b32 vcc_lo, exec_lo, s22
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v8, s18, v8
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_53
; %bb.52:                               ;   in Loop: Header=BB0_21 Depth=1
	global_load_u16 v9, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v8, s9, v9, v8 op_sel_hi:[0,1,0]
.LBB0_53:                               ;   in Loop: Header=BB0_21 Depth=1
	v_cvt_f32_f16_e32 v9, v59
	v_cmp_gt_u32_e32 vcc_lo, s3, v56
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v8, v9
	v_cndmask_b32_e64 v8, v9, v8, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v8, v8
	global_store_b16 v[2:3], v8, off offset:2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_20
; %bb.54:                               ;   in Loop: Header=BB0_21 Depth=1
	s_wait_dscnt 0x1
	v_add_f32_e32 v6, v6, v7
	s_and_not1_b32 vcc_lo, exec_lo, s22
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v6, s18, v6
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_21 Depth=1
	global_load_u16 v7, v[0:1], off offset:4
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v6, s9, v7, v6 op_sel_hi:[0,1,0]
.LBB0_56:                               ;   in Loop: Header=BB0_21 Depth=1
	v_cvt_f32_f16_e32 v7, v57
	v_cmp_gt_u32_e32 vcc_lo, s3, v55
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v7, v6, v7
	v_cndmask_b32_e64 v6, v7, v6, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v6, v6
	global_store_b16 v[2:3], v6, off offset:4
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_20
; %bb.57:                               ;   in Loop: Header=BB0_21 Depth=1
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	s_and_not1_b32 vcc_lo, exec_lo, s22
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v4, s18, v4
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_19
; %bb.58:                               ;   in Loop: Header=BB0_21 Depth=1
	global_load_u16 v0, v[0:1], off offset:6
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v4, s9, v0, v4 op_sel_hi:[0,1,0]
	s_branch .LBB0_19
.LBB0_59:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s19
	s_mov_b32 s0, 0
.LBB0_60:
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_93
; %bb.61:
	v_cmp_gt_u32_e32 vcc_lo, s3, v40
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_93
; %bb.62:
	s_cmp_eq_u64 s[10:11], 0
	v_mbcnt_lo_u32_b32 v50, -1, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[10:11], 0
	v_cmp_eq_u32_e64 s1, 31, v52
	s_cselect_b32 s14, -1, 0
	s_cmp_lg_u32 s2, 0
	v_mov_b32_e32 v41, 0
	s_cselect_b32 s15, -1, 0
	s_add_co_i32 s19, s2, -16
	s_cmp_neq_f32 s9, 0
	v_xor_b32_e32 v52, 16, v50
	s_mov_b32 s24, 0
	s_cselect_b32 s20, -1, 0
	s_add_co_i32 s23, s2, -1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_lshr_b32 s21, s23, 9
	s_wait_alu 0xfffe
	s_add_co_i32 s22, s21, 1
	s_cmp_gt_u32 s2, 0x200
	s_cselect_b32 s21, -1, 0
	s_and_b32 s22, s22, 0xfffffe
	s_bitcmp0_b32 s23, 9
	s_cselect_b32 s23, -1, 0
	s_branch .LBB0_65
.LBB0_63:                               ;   in Loop: Header=BB0_65 Depth=1
	v_cvt_f32_f16_e32 v0, v53
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v4, v0
	v_cndmask_b32_e64 v0, v0, v4, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v0, v0
	global_store_b16 v[2:3], v0, off offset:6
.LBB0_64:                               ;   in Loop: Header=BB0_65 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s25
	v_add_nc_u32_e32 v40, s8, v40
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s3, v40
	s_or_b32 s24, vcc_lo, s24
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s24
	s_cbranch_execz .LBB0_93
.LBB0_65:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_71 Depth 2
	v_or_b32_e32 v55, 1, v40
	v_min_u32_e32 v0, s16, v40
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v53, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_min_u32_e32 v2, s16, v55
	s_and_not1_b32 vcc_lo, exec_lo, s14
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_67
; %bb.66:                               ;   in Loop: Header=BB0_65 Depth=1
	s_wait_dscnt 0x3
	v_mov_b32_e32 v1, v41
	v_mov_b32_e32 v3, v41
	v_or_b32_e32 v6, 2, v40
	s_wait_dscnt 0x1
	v_mov_b32_e32 v7, v41
	v_mov_b32_e32 v11, v41
	s_wait_dscnt 0x0
	v_lshlrev_b64_e32 v[4:5], 1, v[0:1]
	v_or_b32_e32 v1, 3, v40
	v_lshlrev_b64_e32 v[8:9], 1, v[2:3]
	v_min_u32_e32 v6, s16, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_min_u32_e32 v10, s16, v1
	v_add_co_u32 v3, vcc_lo, s10, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s11, v5, vcc_lo
	v_lshlrev_b64_e32 v[5:6], 1, v[6:7]
	v_add_co_u32 v7, vcc_lo, s10, v8
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, s11, v9, vcc_lo
	v_lshlrev_b64_e32 v[9:10], 1, v[10:11]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_co_u32 v5, vcc_lo, s10, v5
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, s11, v6, vcc_lo
	v_add_co_u32 v9, vcc_lo, s10, v9
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v10, null, s11, v10, vcc_lo
	s_clause 0x3
	global_load_u16 v57, v[3:4], off
	global_load_u16 v56, v[7:8], off
	global_load_u16 v54, v[5:6], off
	global_load_u16 v53, v[9:10], off
.LBB0_67:                               ;   in Loop: Header=BB0_65 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s15
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_75
; %bb.68:                               ;   in Loop: Header=BB0_65 Depth=1
	s_wait_dscnt 0x3
	v_or_b32_e32 v1, 2, v40
	v_or_b32_e32 v3, 3, v40
	v_mad_co_u64_u32 v[48:49], null, v0, s17, 0
	v_mad_co_u64_u32 v[46:47], null, v2, s17, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_min_u32_e32 v0, s16, v1
	v_min_u32_e32 v1, s16, v3
	s_and_not1_b32 vcc_lo, exec_lo, s21
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[44:45], null, v0, s17, 0
	v_mad_co_u64_u32 v[42:43], null, v1, s17, 0
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_76
; %bb.69:                               ;   in Loop: Header=BB0_65 Depth=1
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v60, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v58, 0
	s_mov_b32 s25, 0
	s_mov_b32 s26, s22
	s_branch .LBB0_71
.LBB0_70:                               ;   in Loop: Header=BB0_71 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s27
	s_wait_loadcnt_dscnt 0x700
	v_dot4_f32_fp8_fp8 v0, v16, v0, v61
	s_wait_loadcnt 0x6
	v_dot4_f32_fp8_fp8 v4, v16, v4, v60
	s_wait_loadcnt 0x5
	v_dot4_f32_fp8_fp8 v8, v16, v8, v59
	s_wait_loadcnt 0x4
	v_dot4_f32_fp8_fp8 v12, v16, v12, v58
	s_add_co_i32 s26, s26, -2
	v_dot4_f32_fp8_fp8 v0, v17, v1, v0
	v_dot4_f32_fp8_fp8 v1, v17, v5, v4
	v_dot4_f32_fp8_fp8 v4, v17, v9, v8
	v_dot4_f32_fp8_fp8 v5, v17, v13, v12
	s_addk_co_i32 s25, 0x400
	v_dot4_f32_fp8_fp8 v0, v18, v2, v0
	v_dot4_f32_fp8_fp8 v1, v18, v6, v1
	v_dot4_f32_fp8_fp8 v2, v18, v10, v4
	v_dot4_f32_fp8_fp8 v4, v18, v14, v5
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s26, 0
	v_dot4_f32_fp8_fp8 v0, v19, v3, v0
	v_dot4_f32_fp8_fp8 v1, v19, v7, v1
	v_dot4_f32_fp8_fp8 v2, v19, v11, v2
	v_dot4_f32_fp8_fp8 v3, v19, v15, v4
	s_wait_loadcnt 0x3
	v_dot4_f32_fp8_fp8 v0, v36, v20, v0
	s_wait_loadcnt 0x2
	v_dot4_f32_fp8_fp8 v1, v36, v24, v1
	s_wait_loadcnt 0x1
	v_dot4_f32_fp8_fp8 v2, v36, v28, v2
	s_wait_loadcnt 0x0
	v_dot4_f32_fp8_fp8 v3, v36, v32, v3
	v_dot4_f32_fp8_fp8 v0, v37, v21, v0
	v_dot4_f32_fp8_fp8 v1, v37, v25, v1
	v_dot4_f32_fp8_fp8 v2, v37, v29, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v3, v37, v33, v3
	v_dot4_f32_fp8_fp8 v0, v38, v22, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v1, v38, v26, v1
	v_dot4_f32_fp8_fp8 v2, v38, v30, v2
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v3, v38, v34, v3
	v_dot4_f32_fp8_fp8 v61, v39, v23, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v60, v39, v27, v1
	v_dot4_f32_fp8_fp8 v59, v39, v31, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v58, v39, v35, v3
	s_cbranch_scc1 .LBB0_77
.LBB0_71:                               ;   Parent Loop BB0_65 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_dual_mov_b32 v19, 0 :: v_dual_add_nc_u32 v62, s25, v51
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v17, 0
	v_mov_b32_e32 v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v0, s19, v62
	v_add_co_u32 v2, s27, s12, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v3, null, s13, 0, s27
	s_mov_b32 s27, exec_lo
	v_add_co_u32 v0, vcc_lo, v2, v48
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v3, v49, vcc_lo
	v_add_co_u32 v4, vcc_lo, v2, v46
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v3, v47, vcc_lo
	v_add_co_u32 v8, vcc_lo, v2, v44
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, v3, v45, vcc_lo
	v_add_co_u32 v12, vcc_lo, v2, v42
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v13, null, v3, v43, vcc_lo
	s_clause 0x3
	global_load_b128 v[0:3], v[0:1], off
	global_load_b128 v[4:7], v[4:5], off
	global_load_b128 v[8:11], v[8:9], off
	global_load_b128 v[12:15], v[12:13], off
	v_cmpx_gt_u32_e64 s2, v62
; %bb.72:                               ;   in Loop: Header=BB0_71 Depth=2
	ds_load_b128 v[16:19], v62
; %bb.73:                               ;   in Loop: Header=BB0_71 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s27
	v_dual_mov_b32 v39, 0 :: v_dual_add_nc_u32 v36, 0x200, v62
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v37, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v20, s19, v36
	v_add_co_u32 v22, s27, s12, v20
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v23, null, s13, 0, s27
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v20, vcc_lo, v22, v48
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v21, null, v23, v49, vcc_lo
	v_add_co_u32 v24, vcc_lo, v22, v46
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v25, null, v23, v47, vcc_lo
	v_add_co_u32 v28, vcc_lo, v22, v44
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v29, null, v23, v45, vcc_lo
	v_add_co_u32 v32, vcc_lo, v22, v42
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v33, null, v23, v43, vcc_lo
	s_clause 0x3
	global_load_b128 v[20:23], v[20:21], off
	global_load_b128 v[24:27], v[24:25], off
	global_load_b128 v[28:31], v[28:29], off
	global_load_b128 v[32:35], v[32:33], off
	v_cmp_gt_u32_e32 vcc_lo, s2, v36
	v_mov_b32_e32 v36, 0
	s_and_saveexec_b32 s27, vcc_lo
	s_cbranch_execz .LBB0_70
; %bb.74:                               ;   in Loop: Header=BB0_71 Depth=2
	ds_load_b128 v[36:39], v62 offset:512
	s_branch .LBB0_70
.LBB0_75:                               ;   in Loop: Header=BB0_65 Depth=1
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v59, 0
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v61, 0
	s_branch .LBB0_81
.LBB0_76:                               ;   in Loop: Header=BB0_65 Depth=1
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v59, 0
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v61, 0
	s_mov_b32 s25, 0
.LBB0_77:                               ;   in Loop: Header=BB0_65 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s23
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_81
; %bb.78:                               ;   in Loop: Header=BB0_65 Depth=1
	v_dual_mov_b32 v19, 0 :: v_dual_add_nc_u32 v20, s25, v51
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v17, 0
	v_mov_b32_e32 v16, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_min_u32_e32 v0, s19, v20
	v_add_co_u32 v2, s25, s12, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v3, null, s13, 0, s25
	s_mov_b32 s25, exec_lo
	v_add_co_u32 v0, vcc_lo, v2, v48
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v3, v49, vcc_lo
	v_add_co_u32 v4, vcc_lo, v2, v46
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v3, v47, vcc_lo
	v_add_co_u32 v8, vcc_lo, v2, v44
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, v3, v45, vcc_lo
	v_add_co_u32 v12, vcc_lo, v2, v42
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v13, null, v3, v43, vcc_lo
	s_clause 0x3
	global_load_b128 v[0:3], v[0:1], off
	global_load_b128 v[4:7], v[4:5], off
	global_load_b128 v[8:11], v[8:9], off
	global_load_b128 v[12:15], v[12:13], off
	v_cmpx_gt_u32_e64 s2, v20
; %bb.79:                               ;   in Loop: Header=BB0_65 Depth=1
	ds_load_b128 v[16:19], v20
; %bb.80:                               ;   in Loop: Header=BB0_65 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s25
	s_wait_loadcnt_dscnt 0x300
	v_dot4_f32_fp8_fp8 v0, v16, v0, v61
	s_wait_loadcnt 0x2
	v_dot4_f32_fp8_fp8 v4, v16, v4, v60
	s_wait_loadcnt 0x1
	v_dot4_f32_fp8_fp8 v8, v16, v8, v59
	s_wait_loadcnt 0x0
	v_dot4_f32_fp8_fp8 v12, v16, v12, v58
	v_dot4_f32_fp8_fp8 v0, v17, v1, v0
	v_dot4_f32_fp8_fp8 v1, v17, v5, v4
	v_dot4_f32_fp8_fp8 v4, v17, v9, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v5, v17, v13, v12
	v_dot4_f32_fp8_fp8 v0, v18, v2, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v1, v18, v6, v1
	v_dot4_f32_fp8_fp8 v2, v18, v10, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v4, v18, v14, v5
	v_dot4_f32_fp8_fp8 v61, v19, v3, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v60, v19, v7, v1
	v_dot4_f32_fp8_fp8 v59, v19, v11, v2
	s_delay_alu instid0(VALU_DEP_4)
	v_dot4_f32_fp8_fp8 v58, v19, v15, v4
.LBB0_81:                               ;   in Loop: Header=BB0_65 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_dpp v0, v61, v61 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x3
	v_add_f32_dpp v1, v60, v60 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v59, v59 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v58, v58 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v52
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v4, v50, v52, vcc_lo
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x0
	v_lshlrev_b32_e32 v5, 2, v4
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_loadcnt 0x3
	;;#ASMSTART
	;;#ASMEND
	ds_bpermute_b32 v1, v5, v0
	ds_bpermute_b32 v9, v5, v8
	ds_bpermute_b32 v7, v5, v6
	ds_bpermute_b32 v5, v5, v4
	s_wait_loadcnt 0x2
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s25, s1
	s_cbranch_execz .LBB0_64
; %bb.82:                               ;   in Loop: Header=BB0_65 Depth=1
	v_lshlrev_b64_e32 v[2:3], 1, v[40:41]
	s_wait_dscnt 0x3
	v_add_f32_e32 v10, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v10, s18, v10
	v_add_co_u32 v0, vcc_lo, s4, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s5, v3, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s20
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_84
; %bb.83:                               ;   in Loop: Header=BB0_65 Depth=1
	global_load_u16 v11, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v10, s9, v11, v10 op_sel_hi:[0,1,0]
.LBB0_84:                               ;   in Loop: Header=BB0_65 Depth=1
	v_cvt_f32_f16_e32 v11, v57
	v_add_co_u32 v2, vcc_lo, s6, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v11, v10, v11
	v_cmp_gt_u32_e32 vcc_lo, s3, v55
	v_cndmask_b32_e64 v10, v11, v10, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v10, v10
	global_store_b16 v[2:3], v10, off
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_64
; %bb.85:                               ;   in Loop: Header=BB0_65 Depth=1
	s_wait_dscnt 0x2
	v_add_f32_e32 v8, v8, v9
	s_and_not1_b32 vcc_lo, exec_lo, s20
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v8, s18, v8
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_87
; %bb.86:                               ;   in Loop: Header=BB0_65 Depth=1
	global_load_u16 v9, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v8, s9, v9, v8 op_sel_hi:[0,1,0]
.LBB0_87:                               ;   in Loop: Header=BB0_65 Depth=1
	v_cvt_f32_f16_e32 v9, v56
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v8, v9
	v_cndmask_b32_e64 v8, v9, v8, s0
	v_or_b32_e32 v9, 2, v40
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v8, v8
	v_cmp_gt_u32_e32 vcc_lo, s3, v9
	global_store_b16 v[2:3], v8, off offset:2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_64
; %bb.88:                               ;   in Loop: Header=BB0_65 Depth=1
	s_wait_dscnt 0x1
	v_add_f32_e32 v6, v6, v7
	s_and_not1_b32 vcc_lo, exec_lo, s20
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v6, s18, v6
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_90
; %bb.89:                               ;   in Loop: Header=BB0_65 Depth=1
	global_load_u16 v7, v[0:1], off offset:4
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v6, s9, v7, v6 op_sel_hi:[0,1,0]
.LBB0_90:                               ;   in Loop: Header=BB0_65 Depth=1
	v_cvt_f32_f16_e32 v7, v54
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v7, v6, v7
	v_cndmask_b32_e64 v6, v7, v6, s0
	v_or_b32_e32 v7, 3, v40
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v6, v6
	v_cmp_gt_u32_e32 vcc_lo, s3, v7
	global_store_b16 v[2:3], v6, off offset:4
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_64
; %bb.91:                               ;   in Loop: Header=BB0_65 Depth=1
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	s_and_not1_b32 vcc_lo, exec_lo, s20
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v4, s18, v4
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_63
; %bb.92:                               ;   in Loop: Header=BB0_65 Depth=1
	global_load_u16 v0, v[0:1], off offset:6
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v4, s9, v0, v4 op_sel_hi:[0,1,0]
	s_branch .LBB0_63
.LBB0_93:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201
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
		.amdhsa_next_free_sgpr 30
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 48
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
	.size	wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201, .Lfunc_end0-wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201
                                        ; -- End function
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201.num_vgpr, 67
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201.num_agpr, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201.numbered_sgpr, 30
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201.num_named_barrier, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201.private_seg_size, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201.uses_vcc, 1
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201.uses_flat_scratch, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201.has_dyn_sized_stack, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201.has_recursion, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6024
; TotalNumSgprs: 32
; NumVgprs: 67
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 21
; NumSGPRsForWavesPerEU: 32
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
	.type	__hip_cuid_6353a70e97553aa2,@object ; @__hip_cuid_6353a70e97553aa2
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_6353a70e97553aa2
__hip_cuid_6353a70e97553aa2:
	.byte	0                               ; 0x0
	.size	__hip_cuid_6353a70e97553aa2, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_6353a70e97553aa2
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
  AssertSizeEqual: { 1: 1, 2: 1 }
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
    .name:           wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     32
    .sgpr_spill_count: 0
    .symbol:         wvSplitKQ_vllm_f8_f16_tn_n1_y4u1_gfx1201.kd
    .vgpr_count:     67
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
