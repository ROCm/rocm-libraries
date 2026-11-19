	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201 ; -- Begin function wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201
	.globl	wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201
	.p2align	8
	.type	wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201,@function
wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201: ; @wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201
; %bb.0:
	s_clause 0x1
	s_load_b64 s[2:3], s[0:1], 0x0
	s_load_b64 s[18:19], s[0:1], 0x10
	v_and_b32_e32 v54, 0x3ff, v0
	v_bfe_u32 v0, v0, 10, 10
	s_mov_b32 s21, 0
	s_mov_b32 s4, exec_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v53, 4, v54
	v_lshl_add_u32 v3, v0, 9, v53
	s_wait_kmcnt 0x0
	s_cmp_lt_u32 s2, 0x10000
	s_mov_b32 s20, s2
	s_cselect_b32 s28, s2, 0x10000
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cmpx_gt_u32_e64 s28, v3
	s_cbranch_execz .LBB0_9
; %bb.1:
	global_load_b128 v[4:7], v3, s[18:19]
	v_add_nc_u32_e32 v1, 0x2000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7]
	v_cmpx_gt_u32_e64 s28, v1
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.2:
	v_add_co_u32 v1, s5, s18, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_ci_u32_e64 v2, null, s19, 0, s5
	v_add_nc_u32_e32 v8, 0x4000, v3
	s_mov_b32 s5, exec_lo
	global_load_b128 v[4:7], v[1:2], off offset:8192
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:8192
	v_cmpx_gt_u32_e64 s28, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.3:
	global_load_b128 v[4:7], v[1:2], off offset:16384
	v_add_nc_u32_e32 v8, 0x6000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:16384
	v_cmpx_gt_u32_e64 s28, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.4:
	global_load_b128 v[4:7], v[1:2], off offset:24576
	v_add_nc_u32_e32 v8, 0x8000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:24576
	v_cmpx_gt_u32_e64 s28, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.5:
	global_load_b128 v[4:7], v[1:2], off offset:32768
	v_add_nc_u32_e32 v8, 0xa000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:32768
	v_cmpx_gt_u32_e64 s28, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.6:
	global_load_b128 v[4:7], v[1:2], off offset:40960
	v_add_nc_u32_e32 v8, 0xc000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:40960
	v_cmpx_gt_u32_e64 s28, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.7:
	global_load_b128 v[4:7], v[1:2], off offset:49152
	v_add_nc_u32_e32 v8, 0xe000, v3
	s_mov_b32 s5, exec_lo
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:49152
	v_cmpx_gt_u32_e64 s28, v8
	s_wait_alu 0xfffe
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .LBB0_9
; %bb.8:
	global_load_b128 v[4:7], v[1:2], off offset:57344
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[4:7] offset:57344
.LBB0_9:
	s_or_b32 exec_lo, exec_lo, s4
	s_clause 0x2
	s_load_b64 s[12:13], s[0:1], 0x20
	s_load_b128 s[4:7], s[0:1], 0x48
	s_load_b64 s[14:15], s[0:1], 0x58
	s_sub_nc_u64 s[8:9], 0x10000, s[20:21]
	s_mov_b32 s10, exec_lo
	v_cmp_lt_u64_e64 s9, s[8:9], s[20:21]
	s_and_b32 s9, s9, exec_lo
	s_cselect_b32 s8, s8, s20
	s_cmp_lt_u32 s2, 0x10000
	s_wait_alu 0xfffe
	s_cselect_b32 s29, s8, 0
	s_wait_kmcnt 0x0
	s_mov_b32 s20, s13
	s_wait_alu 0xfffe
	v_cmpx_gt_u32_e64 s29, v3
	s_cbranch_execz .LBB0_12
; %bb.10:
	s_add_nc_u64 s[8:9], s[18:19], s[20:21]
	s_mov_b32 s11, 0
.LBB0_11:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[4:7], v3, s[8:9]
	v_add_nc_u32_e32 v1, s2, v3
	v_add_nc_u32_e32 v3, 0x2000, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s29, v3
	s_or_b32 s11, vcc_lo, s11
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[4:7]
	s_and_not1_b32 exec_lo, exec_lo, s11
	s_cbranch_execnz .LBB0_11
.LBB0_12:
	s_or_b32 exec_lo, exec_lo, s10
	s_clause 0x1
	s_load_b64 s[16:17], s[0:1], 0x8
	s_load_b32 s13, s[0:1], 0x18
	s_add_co_i32 s8, s3, -1
	s_mov_b32 s21, 4
	s_wait_kmcnt 0x0
	s_lshl_b32 s9, s13, 4
	s_wait_alu 0xfffe
	v_add_co_u32 v1, s11, s9, s8
	s_and_not1_b32 vcc_lo, exec_lo, s11
	v_readfirstlane_b32 s10, v1
	s_cbranch_vccz .LBB0_14
; %bb.13:
	s_cvt_f32_u32 s11, s9
	s_sub_co_i32 s21, 0, s9
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v1, s11
	v_readfirstlane_b32 s11, v1
	s_mul_f32 s11, s11, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s11, s11
	s_wait_alu 0xfffe
	s_mul_i32 s21, s21, s11
	s_wait_alu 0xfffe
	s_mul_hi_u32 s21, s11, s21
	s_wait_alu 0xfffe
	s_add_co_i32 s11, s11, s21
	s_wait_alu 0xfffe
	s_mul_hi_u32 s11, s10, s11
	s_wait_alu 0xfffe
	s_mul_i32 s21, s11, s9
	s_wait_alu 0xfffe
	s_sub_co_i32 s10, s10, s21
	s_add_co_i32 s21, s11, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s22, s10, s9
	s_cmp_ge_u32 s10, s9
	s_cselect_b32 s11, s21, s11
	s_cselect_b32 s10, s22, s10
	s_wait_alu 0xfffe
	s_add_co_i32 s21, s11, 1
	s_cmp_ge_u32 s10, s9
	s_wait_alu 0xfffe
	s_cselect_b32 s9, s21, s11
	s_wait_alu 0xfffe
	s_mul_i32 s9, s9, s13
	s_wait_alu 0xfffe
	s_cvt_f32_u32 s10, s9
	s_sub_co_i32 s11, 0, s9
	s_add_co_i32 s8, s9, s8
	s_wait_alu 0xfffe
	v_rcp_iflag_f32_e32 v1, s10
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_2)
	v_readfirstlane_b32 s10, v1
	s_mul_f32 s10, s10, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_cvt_u32_f32 s10, s10
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2)
	s_mul_i32 s11, s11, s10
	s_wait_alu 0xfffe
	s_mul_hi_u32 s11, s10, s11
	s_wait_alu 0xfffe
	s_add_co_i32 s10, s10, s11
	s_wait_alu 0xfffe
	s_mul_hi_u32 s10, s8, s10
	s_wait_alu 0xfffe
	s_mul_i32 s11, s10, s9
	s_wait_alu 0xfffe
	s_sub_co_i32 s8, s8, s11
	s_add_co_i32 s11, s10, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s21, s8, s9
	s_cmp_ge_u32 s8, s9
	s_cselect_b32 s10, s11, s10
	s_wait_alu 0xfffe
	s_cselect_b32 s8, s21, s8
	s_add_co_i32 s11, s10, 1
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s8, s9
	s_cselect_b32 s8, s11, s10
	s_wait_alu 0xfffe
	s_max_u32 s21, s8, 4
.LBB0_14:
	s_mov_b32 s22, 1.0
	s_cmp_eq_u64 s[4:5], 0
	s_mov_b32 s23, 1.0
	s_cbranch_scc1 .LBB0_16
; %bb.15:
	s_load_b32 s23, s[4:5], 0x0
.LBB0_16:
	s_cmp_eq_u64 s[6:7], 0
	s_cbranch_scc1 .LBB0_18
; %bb.17:
	s_load_b32 s22, s[6:7], 0x0
.LBB0_18:
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_mov_b32 s4, exec_lo
	s_wait_alu 0xfffe
	v_cmpx_gt_u32_e64 s21, v0
	s_cbranch_execz .LBB0_100
; %bb.19:
	s_load_b256 s[4:11], s[0:1], 0x28
	v_mad_co_u64_u32 v[48:49], null, s21, ttmp9, v[0:1]
	s_cmp_eq_u32 s29, s2
	s_mul_i32 s13, s21, s13
	s_cselect_b32 s0, -1, 0
	s_cmp_eq_u32 s2, s28
	s_cselect_b32 s1, -1, 0
	s_wait_alu 0xfffe
	s_and_b32 s1, s1, s0
	v_cmp_gt_u32_e64 s0, s3, v48
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_mov_b32 s1, -1
	s_wait_kmcnt 0x0
	s_mul_f32 s8, s8, s23
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2)
	s_mul_f32 s8, s8, s22
	s_cbranch_vccz .LBB0_76
; %bb.20:
	s_and_saveexec_b32 s21, s0
	s_cbranch_execz .LBB0_75
; %bb.21:
	s_cmp_eq_u64 s[14:15], 0
	s_mov_b32 s23, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[14:15], 0
	v_cmp_eq_u32_e64 s1, 31, v54
	s_cselect_b32 s30, -1, 0
	s_cmp_lg_u32 s2, 0
	v_dual_mov_b32 v50, 0 :: v_dual_add_nc_u32 v55, s2, v53
	s_cselect_b32 s31, -1, 0
	s_add_co_i32 s33, s2, -16
	s_cmp_neq_f32 s9, 0
	v_mbcnt_lo_u32_b32 v56, -1, 0
	v_mov_b32_e32 v49, v48
	s_mov_b32 s24, s10
	s_mov_b32 s25, s23
	s_mov_b32 s22, s11
	s_cselect_b32 s34, -1, 0
	s_lshl_b64 s[24:25], s[24:25], 1
	s_lshl_b64 s[26:27], s[22:23], 1
	s_branch .LBB0_24
.LBB0_22:                               ;   in Loop: Header=BB0_24 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v4, v6
	v_cndmask_b32_e64 v4, v0, v4, s0
	v_add_co_u32 v0, vcc_lo, v2, s26
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s27, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f16_f32_e32 v2, v4
	global_store_b16 v[0:1], v2, off
.LBB0_23:                               ;   in Loop: Header=BB0_24 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	v_add_nc_u32_e32 v49, s13, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s3, v49
	s_or_b32 s23, vcc_lo, s23
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s23
	s_cbranch_execz .LBB0_75
.LBB0_24:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_33 Depth 2
	v_mov_b32_e32 v57, 0
	s_and_not1_b32 vcc_lo, exec_lo, s30
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_26
; %bb.25:                               ;   in Loop: Header=BB0_24 Depth=1
	s_wait_dscnt 0x1
	v_lshlrev_b64_e32 v[0:1], 1, v[49:50]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s14, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s15, v1, vcc_lo
	global_load_u16 v57, v[0:1], off
.LBB0_26:                               ;   in Loop: Header=BB0_24 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s31
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_69
; %bb.27:                               ;   in Loop: Header=BB0_24 Depth=1
	v_mad_co_u64_u32 v[51:52], null, v49, s12, s[16:17]
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v58, 0
	s_mov_b32 s22, 0
	s_branch .LBB0_33
.LBB0_28:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s39
.LBB0_29:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s38
.LBB0_30:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s37
.LBB0_31:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s36
.LBB0_32:                               ;   in Loop: Header=BB0_33 Depth=2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s35
	s_wait_loadcnt_dscnt 0x0
	v_dot4_f32_fp8_fp8 v16, v16, v12, v59
	v_dot4_f32_fp8_fp8 v12, v20, v12, v58
	s_addk_co_i32 s22, 0x800
	s_wait_alu 0xfffe
	s_cmp_lt_u32 s22, s2
	v_dot4_f32_fp8_fp8 v16, v17, v13, v16
	v_dot4_f32_fp8_fp8 v12, v21, v13, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v13, v18, v14, v16
	v_dot4_f32_fp8_fp8 v12, v22, v14, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v13, v19, v15, v13
	v_dot4_f32_fp8_fp8 v12, v23, v15, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v13, v24, v8, v13
	v_dot4_f32_fp8_fp8 v8, v28, v8, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v12, v25, v9, v13
	v_dot4_f32_fp8_fp8 v8, v29, v9, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v9, v26, v10, v12
	v_dot4_f32_fp8_fp8 v8, v30, v10, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v9, v27, v11, v9
	v_dot4_f32_fp8_fp8 v8, v31, v11, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v9, v32, v4, v9
	v_dot4_f32_fp8_fp8 v4, v36, v4, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v8, v33, v5, v9
	v_dot4_f32_fp8_fp8 v4, v37, v5, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v5, v34, v6, v8
	v_dot4_f32_fp8_fp8 v4, v38, v6, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v5, v35, v7, v5
	v_dot4_f32_fp8_fp8 v4, v39, v7, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v5, v40, v0, v5
	v_dot4_f32_fp8_fp8 v0, v44, v0, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v4, v41, v1, v5
	v_dot4_f32_fp8_fp8 v0, v45, v1, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v1, v42, v2, v4
	v_dot4_f32_fp8_fp8 v0, v46, v2, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v59, v43, v3, v1
	v_dot4_f32_fp8_fp8 v58, v47, v3, v0
	s_cbranch_scc0 .LBB0_70
.LBB0_33:                               ;   Parent Loop BB0_24 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_add_nc_u32_e32 v61, s22, v53
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v44, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v18, 0 :: v_dual_add_nc_u32 v63, 0x200, v61
	v_add_nc_u32_e32 v62, 0x400, v61
	v_min_u32_e32 v0, s33, v61
	v_dual_mov_b32 v19, 0 :: v_dual_add_nc_u32 v60, 0x600, v61
	v_min_u32_e32 v2, s33, v63
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_min_u32_e32 v4, s33, v62
	v_add_co_u32 v0, vcc_lo, v51, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_min_u32_e32 v6, s33, v60
	s_wait_dscnt 0x1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, 0, v52, vcc_lo
	v_add_co_u32 v2, vcc_lo, v51, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, 0, v52, vcc_lo
	v_add_co_u32 v4, vcc_lo, v51, v4
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, 0, v52, vcc_lo
	v_add_co_u32 v16, vcc_lo, v51, v6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v17, null, 0, v52, vcc_lo
	s_clause 0x3
	global_load_b128 v[12:15], v[0:1], off
	global_load_b128 v[8:11], v[2:3], off
	global_load_b128 v[4:7], v[4:5], off
	global_load_b128 v[0:3], v[16:17], off
	v_mov_b32_e32 v16, 0
	v_cmp_gt_u32_e32 vcc_lo, s2, v61
	v_mov_b32_e32 v17, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v20, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v32, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v24, 0
	s_and_saveexec_b32 s35, vcc_lo
	s_cbranch_execz .LBB0_32
; %bb.34:                               ;   in Loop: Header=BB0_33 Depth=2
	v_add_co_u32 v20, s36, s18, v61
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v21, null, s19, 0, s36
	s_mov_b32 s36, exec_lo
                                        ; implicit-def: $vgpr16
	v_cmpx_le_u32_e64 s28, v61
	s_wait_alu 0xfffe
	s_xor_b32 s36, exec_lo, s36
	s_cbranch_execz .LBB0_36
; %bb.35:                               ;   in Loop: Header=BB0_33 Depth=2
	global_load_b128 v[16:19], v[20:21], off
.LBB0_36:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s36, s36
	s_cbranch_execz .LBB0_38
; %bb.37:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[16:19], v61
.LBB0_38:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s36
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s36, exec_lo
                                        ; implicit-def: $vgpr23
	v_cmpx_le_u32_e64 s29, v61
	s_wait_alu 0xfffe
	s_xor_b32 s36, exec_lo, s36
	s_cbranch_execz .LBB0_40
; %bb.39:                               ;   in Loop: Header=BB0_33 Depth=2
	v_add_co_u32 v20, vcc_lo, v20, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v21, null, 0, v21, vcc_lo
	global_load_b128 v[20:23], v[20:21], off
.LBB0_40:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s36, s36
	s_cbranch_execz .LBB0_42
; %bb.41:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v20, s22, v55
	ds_load_b128 v[20:23], v20
.LBB0_42:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s36
	v_cmp_gt_u32_e32 vcc_lo, s2, v63
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v32, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v24, 0
	s_and_saveexec_b32 s36, vcc_lo
	s_cbranch_execz .LBB0_31
; %bb.43:                               ;   in Loop: Header=BB0_33 Depth=2
	v_add_co_u32 v28, s37, s18, v63
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v29, null, s19, 0, s37
	s_mov_b32 s37, exec_lo
                                        ; implicit-def: $vgpr27
	v_cmpx_le_u32_e64 s28, v63
	s_wait_alu 0xfffe
	s_xor_b32 s37, exec_lo, s37
	s_cbranch_execz .LBB0_45
; %bb.44:                               ;   in Loop: Header=BB0_33 Depth=2
	global_load_b128 v[24:27], v[28:29], off
.LBB0_45:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s37, s37
	s_cbranch_execz .LBB0_47
; %bb.46:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[24:27], v61 offset:512
.LBB0_47:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s37
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s37, exec_lo
                                        ; implicit-def: $vgpr31
	v_cmpx_le_u32_e64 s29, v63
	s_wait_alu 0xfffe
	s_xor_b32 s37, exec_lo, s37
	s_cbranch_execz .LBB0_49
; %bb.48:                               ;   in Loop: Header=BB0_33 Depth=2
	v_add_co_u32 v28, vcc_lo, v28, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v29, null, 0, v29, vcc_lo
	global_load_b128 v[28:31], v[28:29], off
.LBB0_49:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s37, s37
	s_cbranch_execz .LBB0_51
; %bb.50:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v28, s22, v55
	ds_load_b128 v[28:31], v28 offset:512
.LBB0_51:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s37
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v32, 0
	s_mov_b32 s37, exec_lo
	v_cmpx_gt_u32_e64 s2, v62
	s_cbranch_execz .LBB0_30
; %bb.52:                               ;   in Loop: Header=BB0_33 Depth=2
	v_add_co_u32 v36, s38, s18, v62
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v37, null, s19, 0, s38
	s_mov_b32 s38, exec_lo
                                        ; implicit-def: $vgpr35
	v_cmpx_le_u32_e64 s28, v62
	s_wait_alu 0xfffe
	s_xor_b32 s38, exec_lo, s38
	s_cbranch_execz .LBB0_54
; %bb.53:                               ;   in Loop: Header=BB0_33 Depth=2
	global_load_b128 v[32:35], v[36:37], off
.LBB0_54:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s38, s38
	s_cbranch_execz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[32:35], v61 offset:1024
.LBB0_56:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s38
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s38, exec_lo
                                        ; implicit-def: $vgpr39
	v_cmpx_le_u32_e64 s29, v62
	s_wait_alu 0xfffe
	s_xor_b32 s38, exec_lo, s38
	s_cbranch_execz .LBB0_58
; %bb.57:                               ;   in Loop: Header=BB0_33 Depth=2
	v_add_co_u32 v36, vcc_lo, v36, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v37, null, 0, v37, vcc_lo
	global_load_b128 v[36:39], v[36:37], off
.LBB0_58:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s38, s38
	s_cbranch_execz .LBB0_60
; %bb.59:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v36, s22, v55
	ds_load_b128 v[36:39], v36 offset:1024
.LBB0_60:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s38
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v40, 0
	s_mov_b32 s38, exec_lo
	v_cmpx_gt_u32_e64 s2, v60
	s_cbranch_execz .LBB0_29
; %bb.61:                               ;   in Loop: Header=BB0_33 Depth=2
	v_add_co_u32 v44, s39, s18, v60
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v45, null, s19, 0, s39
	s_mov_b32 s39, exec_lo
                                        ; implicit-def: $vgpr43
	v_cmpx_le_u32_e64 s28, v60
	s_wait_alu 0xfffe
	s_xor_b32 s39, exec_lo, s39
	s_cbranch_execz .LBB0_63
; %bb.62:                               ;   in Loop: Header=BB0_33 Depth=2
	global_load_b128 v[40:43], v[44:45], off
                                        ; implicit-def: $vgpr61
.LBB0_63:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s39, s39
	s_cbranch_execz .LBB0_65
; %bb.64:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_loadcnt 0x0
	ds_load_b128 v[40:43], v61 offset:1536
.LBB0_65:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s39
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s39, exec_lo
                                        ; implicit-def: $vgpr47
	v_cmpx_le_u32_e64 s29, v60
	s_wait_alu 0xfffe
	s_xor_b32 s39, exec_lo, s39
	s_cbranch_execz .LBB0_67
; %bb.66:                               ;   in Loop: Header=BB0_33 Depth=2
	v_add_co_u32 v44, vcc_lo, v44, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v45, null, 0, v45, vcc_lo
	global_load_b128 v[44:47], v[44:45], off
.LBB0_67:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_alu 0xfffe
	s_and_not1_saveexec_b32 s39, s39
	s_cbranch_execz .LBB0_28
; %bb.68:                               ;   in Loop: Header=BB0_33 Depth=2
	s_wait_loadcnt 0x0
	v_add_nc_u32_e32 v44, s22, v55
	ds_load_b128 v[44:47], v44 offset:1536
	s_branch .LBB0_28
.LBB0_69:                               ;   in Loop: Header=BB0_24 Depth=1
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v59, 0
.LBB0_70:                               ;   in Loop: Header=BB0_24 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_dpp v0, v59, v59 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x1
	v_add_f32_dpp v1, v58, v58 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_xor_b32_e32 v2, 16, v56
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v2, v56, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b32_e32 v2, 2, v2
	ds_bpermute_b32 v1, v2, v0
	s_wait_dscnt 0x1
	ds_bpermute_b32 v5, v2, v4
	s_and_saveexec_b32 s22, s1
	s_cbranch_execz .LBB0_23
; %bb.71:                               ;   in Loop: Header=BB0_24 Depth=1
	v_lshlrev_b64_e32 v[2:3], 1, v[49:50]
	s_wait_dscnt 0x1
	v_add_f32_e32 v6, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v7, s8, v6
	v_add_co_u32 v0, vcc_lo, s4, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s5, v3, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s34
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_73
; %bb.72:                               ;   in Loop: Header=BB0_24 Depth=1
	global_load_u16 v6, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v7, s9, v6, v7 op_sel_hi:[0,1,0]
.LBB0_73:                               ;   in Loop: Header=BB0_24 Depth=1
	v_cvt_f32_f16_e32 v6, v57
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	v_add_co_u32 v2, vcc_lo, s6, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	v_add_f32_e32 v8, v7, v6
	v_mul_f32_e32 v4, s8, v4
	s_and_not1_b32 vcc_lo, exec_lo, s34
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v5, v8, v7, s0
	v_cvt_f16_f32_e32 v5, v5
	global_store_b16 v[2:3], v5, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_22
; %bb.74:                               ;   in Loop: Header=BB0_24 Depth=1
	v_add_co_u32 v0, vcc_lo, v0, s24
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s25, v1, vcc_lo
	global_load_u16 v0, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v4, s9, v0, v4 op_sel_hi:[0,1,0]
	s_branch .LBB0_22
.LBB0_75:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s21
	s_mov_b32 s1, 0
.LBB0_76:
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s1
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_100
; %bb.77:
	v_cmp_gt_u32_e32 vcc_lo, s3, v48
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_100
; %bb.78:
	s_cmp_eq_u64 s[14:15], 0
	v_mbcnt_lo_u32_b32 v52, -1, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[14:15], 0
	s_mov_b32 s19, 0
	s_cselect_b32 s22, -1, 0
	s_cmp_lg_u32 s2, 0
	v_cmp_eq_u32_e64 s1, 31, v54
	s_cselect_b32 s23, -1, 0
	s_add_co_i32 s24, s2, -16
	s_cmp_neq_f32 s9, 0
	v_dual_mov_b32 v49, 0 :: v_dual_add_nc_u32 v54, s2, v53
	v_xor_b32_e32 v55, 16, v52
	s_mov_b32 s18, s11
	s_wait_alu 0xfffe
	s_mov_b32 s11, s19
	s_cselect_b32 s25, -1, 0
	s_wait_alu 0xfffe
	s_lshl_b64 s[10:11], s[10:11], 1
	s_lshl_b64 s[20:21], s[18:19], 1
	s_branch .LBB0_81
.LBB0_79:                               ;   in Loop: Header=BB0_81 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v4, v6
	v_cndmask_b32_e64 v4, v0, v4, s0
	v_add_co_u32 v0, vcc_lo, v2, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s21, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cvt_f16_f32_e32 v2, v4
	global_store_b16 v[0:1], v2, off
.LBB0_80:                               ;   in Loop: Header=BB0_81 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s18
	v_add_nc_u32_e32 v48, s13, v48
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s3, v48
	s_or_b32 s19, vcc_lo, s19
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s19
	s_cbranch_execz .LBB0_100
.LBB0_81:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_89 Depth 2
	v_mov_b32_e32 v56, 0
	s_and_not1_b32 vcc_lo, exec_lo, s22
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_83
; %bb.82:                               ;   in Loop: Header=BB0_81 Depth=1
	s_wait_dscnt 0x1
	v_lshlrev_b64_e32 v[0:1], 1, v[48:49]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s14, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s15, v1, vcc_lo
	global_load_u16 v56, v[0:1], off
.LBB0_83:                               ;   in Loop: Header=BB0_81 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s23
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_94
; %bb.84:                               ;   in Loop: Header=BB0_81 Depth=1
	v_mad_co_u64_u32 v[50:51], null, v48, s12, s[16:17]
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v57, 0
	s_mov_b32 s18, 0
	s_branch .LBB0_89
.LBB0_85:                               ;   in Loop: Header=BB0_89 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s29
.LBB0_86:                               ;   in Loop: Header=BB0_89 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s28
.LBB0_87:                               ;   in Loop: Header=BB0_89 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s27
.LBB0_88:                               ;   in Loop: Header=BB0_89 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s26
	s_wait_loadcnt_dscnt 0x301
	v_dot4_f32_fp8_fp8 v44, v44, v12, v58
	s_wait_dscnt 0x0
	v_dot4_f32_fp8_fp8 v12, v28, v12, v57
	s_addk_co_i32 s18, 0x800
	s_wait_alu 0xfffe
	s_cmp_lt_u32 s18, s2
	v_dot4_f32_fp8_fp8 v28, v45, v13, v44
	v_dot4_f32_fp8_fp8 v12, v29, v13, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v13, v46, v14, v28
	v_dot4_f32_fp8_fp8 v12, v30, v14, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v13, v47, v15, v13
	v_dot4_f32_fp8_fp8 v12, v31, v15, v12
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v13, v24, v8, v13
	v_dot4_f32_fp8_fp8 v8, v36, v8, v12
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v12, v25, v9, v13
	v_dot4_f32_fp8_fp8 v8, v37, v9, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v9, v26, v10, v12
	v_dot4_f32_fp8_fp8 v8, v38, v10, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v9, v27, v11, v9
	v_dot4_f32_fp8_fp8 v8, v39, v11, v8
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v9, v40, v4, v9
	v_dot4_f32_fp8_fp8 v4, v32, v4, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v8, v41, v5, v9
	v_dot4_f32_fp8_fp8 v4, v33, v5, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v5, v42, v6, v8
	v_dot4_f32_fp8_fp8 v4, v34, v6, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v5, v43, v7, v5
	v_dot4_f32_fp8_fp8 v4, v35, v7, v4
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v5, v20, v0, v5
	v_dot4_f32_fp8_fp8 v0, v16, v0, v4
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v4, v21, v1, v5
	v_dot4_f32_fp8_fp8 v0, v17, v1, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v1, v22, v2, v4
	v_dot4_f32_fp8_fp8 v0, v18, v2, v0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dot4_f32_fp8_fp8 v58, v23, v3, v1
	v_dot4_f32_fp8_fp8 v57, v19, v3, v0
	s_cbranch_scc0 .LBB0_95
.LBB0_89:                               ;   Parent Loop BB0_81 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_dual_mov_b32 v24, 0 :: v_dual_add_nc_u32 v59, s18, v53
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v20, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v26, 0 :: v_dual_add_nc_u32 v63, 0x200, v59
	v_dual_mov_b32 v40, 0 :: v_dual_add_nc_u32 v61, 0x400, v59
	v_min_u32_e32 v0, s24, v59
	v_dual_mov_b32 v25, 0 :: v_dual_add_nc_u32 v60, 0x600, v59
	v_min_u32_e32 v2, s24, v63
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_min_u32_e32 v4, s24, v61
	v_add_co_u32 v0, vcc_lo, v50, v0
	s_delay_alu instid0(VALU_DEP_4)
	v_min_u32_e32 v6, s24, v60
	s_wait_dscnt 0x1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, 0, v51, vcc_lo
	v_add_co_u32 v2, vcc_lo, v50, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, 0, v51, vcc_lo
	v_add_co_u32 v4, vcc_lo, v50, v4
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, 0, v51, vcc_lo
	v_add_co_u32 v16, vcc_lo, v50, v6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v17, null, 0, v51, vcc_lo
	s_clause 0x3
	global_load_b128 v[12:15], v[0:1], off
	global_load_b128 v[8:11], v[2:3], off
	global_load_b128 v[4:7], v[4:5], off
	global_load_b128 v[0:3], v[16:17], off
	v_cmp_gt_u32_e32 vcc_lo, s2, v59
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v32, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v16, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v44, 0
	v_mov_b32_e32 v47, 0
	v_mov_b32_e32 v45, 0
	s_and_saveexec_b32 s26, vcc_lo
	s_cbranch_execz .LBB0_88
; %bb.90:                               ;   in Loop: Header=BB0_89 Depth=2
	v_dual_mov_b32 v19, 0 :: v_dual_add_nc_u32 v62, s18, v54
	v_cmp_gt_u32_e32 vcc_lo, s2, v63
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v17, 0
	ds_load_b128 v[44:47], v59
	ds_load_b128 v[28:31], v62
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v39, 0
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v25, 0
	v_mov_b32_e32 v24, 0
	s_and_saveexec_b32 s27, vcc_lo
	s_cbranch_execz .LBB0_87
; %bb.91:                               ;   in Loop: Header=BB0_89 Depth=2
	ds_load_b128 v[24:27], v59 offset:512
	ds_load_b128 v[36:39], v62 offset:512
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v16, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v32, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v20, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v40, 0
	s_mov_b32 s28, exec_lo
	v_cmpx_gt_u32_e64 s2, v61
	s_cbranch_execz .LBB0_86
; %bb.92:                               ;   in Loop: Header=BB0_89 Depth=2
	ds_load_b128 v[40:43], v59 offset:1024
	ds_load_b128 v[32:35], v62 offset:1024
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v16, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v20, 0
	s_mov_b32 s29, exec_lo
	v_cmpx_gt_u32_e64 s2, v60
	s_cbranch_execz .LBB0_85
; %bb.93:                               ;   in Loop: Header=BB0_89 Depth=2
	ds_load_b128 v[20:23], v59 offset:1536
	ds_load_b128 v[16:19], v62 offset:1536
	s_branch .LBB0_85
.LBB0_94:                               ;   in Loop: Header=BB0_81 Depth=1
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v58, 0
.LBB0_95:                               ;   in Loop: Header=BB0_81 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_dpp v0, v58, v58 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x1
	v_add_f32_dpp v1, v57, v57 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v55
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v2, v52, v55, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b32_e32 v2, 2, v2
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_3)
	v_add_f32_dpp v4, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v1, v2, v0
	s_wait_dscnt 0x1
	ds_bpermute_b32 v5, v2, v4
	s_and_saveexec_b32 s18, s1
	s_cbranch_execz .LBB0_80
; %bb.96:                               ;   in Loop: Header=BB0_81 Depth=1
	v_lshlrev_b64_e32 v[2:3], 1, v[48:49]
	s_wait_dscnt 0x1
	v_add_f32_e32 v6, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v7, s8, v6
	v_add_co_u32 v0, vcc_lo, s4, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s5, v3, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_98
; %bb.97:                               ;   in Loop: Header=BB0_81 Depth=1
	global_load_u16 v6, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v7, s9, v6, v7 op_sel_hi:[0,1,0]
.LBB0_98:                               ;   in Loop: Header=BB0_81 Depth=1
	v_cvt_f32_f16_e32 v6, v56
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	v_add_co_u32 v2, vcc_lo, s6, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	v_add_f32_e32 v8, v7, v6
	v_mul_f32_e32 v4, s8, v4
	s_and_not1_b32 vcc_lo, exec_lo, s25
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v5, v8, v7, s0
	v_cvt_f16_f32_e32 v5, v5
	global_store_b16 v[2:3], v5, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_79
; %bb.99:                               ;   in Loop: Header=BB0_81 Depth=1
	v_add_co_u32 v0, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	global_load_u16 v0, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v4, s9, v0, v4 op_sel_hi:[0,1,0]
	s_branch .LBB0_79
.LBB0_100:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201
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
		.amdhsa_next_free_sgpr 40
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 41
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
	.size	wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201, .Lfunc_end0-wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201
                                        ; -- End function
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201.num_vgpr, 64
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201.num_agpr, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201.numbered_sgpr, 40
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201.num_named_barrier, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201.private_seg_size, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201.uses_vcc, 1
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201.uses_flat_scratch, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201.has_dyn_sized_stack, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201.has_recursion, 0
	.set wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 5240
; TotalNumSgprs: 42
; NumVgprs: 64
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 21
; NumSGPRsForWavesPerEU: 42
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
	.type	__hip_cuid_96f65188a6d0e07e,@object ; @__hip_cuid_96f65188a6d0e07e
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_96f65188a6d0e07e
__hip_cuid_96f65188a6d0e07e:
	.byte	0                               ; 0x0
	.size	__hip_cuid_96f65188a6d0e07e, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_96f65188a6d0e07e
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
    .name:           wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     42
    .sgpr_spill_count: 0
    .symbol:         wvSplitKQ_vllm_f8_f16_tn_n2_y1u4_gfx1201.kd
    .vgpr_count:     64
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
