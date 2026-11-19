	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201 ; -- Begin function wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201
	.globl	wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201
	.p2align	8
	.type	wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201,@function
wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201:  ; @wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201
; %bb.0:
	s_clause 0x2
	s_load_b32 s2, s[0:1], 0x18
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b64 s[12:13], s[0:1], 0x48
	s_mov_b32 s24, 4
	s_mov_b32 s18, -1
	s_wait_kmcnt 0x0
	s_lshl_b32 s8, s2, 4
	s_add_co_i32 s3, s5, -1
	s_delay_alu instid0(SALU_CYCLE_1)
	v_add_co_u32 v1, s10, s8, s3
	s_and_b32 vcc_lo, exec_lo, s10
	v_readfirstlane_b32 s9, v1
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cvt_f32_u32 s10, s8
	s_sub_co_i32 s11, 0, s8
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v1, s10
	v_readfirstlane_b32 s10, v1
	s_mul_f32 s10, s10, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s10, s10
	s_wait_alu 0xfffe
	s_mul_i32 s11, s11, s10
	s_wait_alu 0xfffe
	s_mul_hi_u32 s11, s10, s11
	s_wait_alu 0xfffe
	s_add_co_i32 s10, s10, s11
	s_wait_alu 0xfffe
	s_mul_hi_u32 s10, s9, s10
	s_wait_alu 0xfffe
	s_mul_i32 s11, s10, s8
	s_wait_alu 0xfffe
	s_sub_co_i32 s9, s9, s11
	s_add_co_i32 s11, s10, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s14, s9, s8
	s_cmp_ge_u32 s9, s8
	s_cselect_b32 s10, s11, s10
	s_cselect_b32 s9, s14, s9
	s_wait_alu 0xfffe
	s_add_co_i32 s11, s10, 1
	s_cmp_ge_u32 s9, s8
	s_wait_alu 0xfffe
	s_cselect_b32 s8, s11, s10
	s_wait_alu 0xfffe
	s_mul_i32 s8, s8, s2
	s_wait_alu 0xfffe
	s_cvt_f32_u32 s9, s8
	s_sub_co_i32 s10, 0, s8
	s_add_co_i32 s11, s8, s3
	s_wait_alu 0xfffe
	v_rcp_iflag_f32_e32 v1, s9
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_2)
	v_readfirstlane_b32 s9, v1
	s_mul_f32 s9, s9, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_cvt_u32_f32 s9, s9
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2)
	s_mul_i32 s10, s10, s9
	s_wait_alu 0xfffe
	s_mul_hi_u32 s10, s9, s10
	s_wait_alu 0xfffe
	s_add_co_i32 s9, s9, s10
	s_wait_alu 0xfffe
	s_mul_hi_u32 s9, s11, s9
	s_wait_alu 0xfffe
	s_mul_i32 s10, s9, s8
	s_wait_alu 0xfffe
	s_sub_co_i32 s10, s11, s10
	s_add_co_i32 s11, s9, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s14, s10, s8
	s_cmp_ge_u32 s10, s8
	s_cselect_b32 s9, s11, s9
	s_cselect_b32 s10, s14, s10
	s_wait_alu 0xfffe
	s_add_co_i32 s11, s9, 1
	s_cmp_ge_u32 s10, s8
	s_wait_alu 0xfffe
	s_cselect_b32 s8, s11, s9
	s_wait_alu 0xfffe
	s_max_u32 s24, s8, 4
.LBB0_2:
	s_clause 0x3
	s_load_b64 s[16:17], s[0:1], 0x10
	s_load_b32 s22, s[0:1], 0x20
	s_load_b128 s[8:11], s[0:1], 0x28
	s_load_b64 s[14:15], s[0:1], 0x38
	v_bfe_u32 v35, v0, 10, 10
	s_cmp_gt_u32 s4, 0x8000
	s_mul_i32 s23, s24, s2
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[33:34], null, s24, ttmp9, v[35:36]
	v_cmp_gt_u32_e32 vcc_lo, s24, v35
	s_cbranch_scc0 .LBB0_32
; %bb.3:
	s_and_saveexec_b32 s25, vcc_lo
	s_cbranch_execz .LBB0_31
; %bb.4:
	s_cvt_f32_u32 s0, s24
	s_sub_co_i32 s1, 0, s24
	s_add_co_i32 s2, s24, s3
	s_mov_b32 s19, 0
	s_wait_alu 0xfffe
	v_rcp_iflag_f32_e32 v1, s0
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_2)
	v_readfirstlane_b32 s0, v1
	s_mul_f32 s0, s0, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_cvt_u32_f32 s0, s0
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2)
	s_mul_i32 s1, s1, s0
	s_wait_alu 0xfffe
	s_mul_hi_u32 s1, s0, s1
	s_wait_alu 0xfffe
	s_add_co_i32 s0, s0, s1
	s_wait_alu 0xfffe
	s_mul_hi_u32 s0, s2, s0
	s_wait_alu 0xfffe
	s_mul_i32 s0, s0, s24
	s_wait_alu 0xfffe
	s_sub_co_i32 s0, s2, s0
	s_wait_alu 0xfffe
	s_sub_co_i32 s1, s0, s24
	s_cmp_ge_u32 s0, s24
	s_wait_alu 0xfffe
	s_cselect_b32 s0, s1, s0
	s_wait_alu 0xfffe
	s_sub_co_i32 s1, s0, s24
	s_cmp_ge_u32 s0, s24
	s_wait_alu 0xfffe
	s_cselect_b32 s0, s1, s0
	s_wait_alu 0xfffe
	s_sub_co_i32 s26, s2, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cmp_gt_u32_e32 vcc_lo, s26, v33
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_31
; %bb.5:
	v_dual_mov_b32 v38, v33 :: v_dual_and_b32 v1, 0x3ff, v0
	v_mbcnt_lo_u32_b32 v34, -1, 0
	s_cmp_lg_u64 s[12:13], 0
	s_mov_b32 s31, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v37, 0 :: v_dual_lshlrev_b32 v40, 3, v1
	v_lshlrev_b32_e32 v2, 4, v1
	s_cselect_b32 s0, -1, 0
	s_lshl_b32 s27, s24, 8
	s_add_co_i32 s28, s4, -8
	s_wait_kmcnt 0x0
	s_cmp_neq_f32 s15, 0
	v_cmp_eq_u32_e64 s1, 31, v1
	v_lshl_add_u32 v41, v35, 8, v40
	v_lshl_add_u32 v42, v35, 9, v2
	v_xor_b32_e32 v43, 16, v34
	s_cselect_b32 s29, -1, 0
	s_lshl_b32 s30, s24, 9
	s_branch .LBB0_8
.LBB0_6:                                ;   in Loop: Header=BB0_8 Depth=1
	v_lshlrev_b32_e32 v4, 16, v44
	v_add_co_u32 v1, vcc_lo, s10, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s11, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v4, v3, v4
	v_cndmask_b32_e64 v3, v3, v4, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v4, v3, 16, 1
	v_or_b32_e32 v5, 0x400000, v3
	v_cmp_u_f32_e32 vcc_lo, v3, v3
	v_add3_u32 v4, v4, v3, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v3, v4, v5, vcc_lo
	global_store_d16_hi_b16 v[1:2], v3, off
.LBB0_7:                                ;   in Loop: Header=BB0_8 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s18
	v_add_nc_u32_e32 v38, s23, v38
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s26, v38
	s_or_b32 s31, vcc_lo, s31
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s31
	s_cbranch_execz .LBB0_31
.LBB0_8:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_16 Depth 2
                                        ;       Child Loop BB0_20 Depth 3
	v_cmp_gt_u32_e64 s2, s5, v38
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v39, v37 :: v_dual_mov_b32 v44, v37
	s_wait_alu 0xfffe
	s_and_b32 s20, s0, s2
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s18, s20
	s_cbranch_execz .LBB0_10
; %bb.9:                                ;   in Loop: Header=BB0_8 Depth=1
	s_wait_dscnt 0x0
	v_lshlrev_b64_e32 v[1:2], 1, v[38:39]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s12, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s13, v2, vcc_lo
	global_load_u16 v44, v[1:2], off
.LBB0_10:                               ;   in Loop: Header=BB0_8 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s18
	s_wait_dscnt 0x0
	v_mad_co_u64_u32 v[1:2], null, v38, s22, 0
	v_mov_b32_e32 v47, 0
	s_mov_b32 s18, 0
	s_wait_alu 0xfffe
	s_mov_b32 s33, s18
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 1, v[1:2]
	v_add_co_u32 v45, vcc_lo, s6, v1
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v46, null, s7, v2, vcc_lo
	s_branch .LBB0_16
.LBB0_11:                               ;   in Loop: Header=BB0_16 Depth=2
	s_or_b32 exec_lo, exec_lo, s36
.LBB0_12:                               ;   in Loop: Header=BB0_16 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s35
.LBB0_13:                               ;   in Loop: Header=BB0_16 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s34
.LBB0_14:                               ;   in Loop: Header=BB0_16 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s21
	s_wait_loadcnt_dscnt 0x300
	v_dot2_f32_bf16 v13, v29, v13, v47
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot2_f32_bf16 v13, v30, v14, v13
	v_dot2_f32_bf16 v13, v31, v15, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dot2_f32_bf16 v13, v32, v16, v13
	s_wait_loadcnt 0x2
	v_dot2_f32_bf16 v9, v17, v9, v13
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot2_f32_bf16 v9, v18, v10, v9
	v_dot2_f32_bf16 v9, v19, v11, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dot2_f32_bf16 v9, v20, v12, v9
	s_wait_loadcnt 0x1
	v_dot2_f32_bf16 v5, v21, v5, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot2_f32_bf16 v5, v22, v6, v5
	v_dot2_f32_bf16 v5, v23, v7, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_dot2_f32_bf16 v5, v24, v8, v5
	s_wait_loadcnt 0x0
	v_dot2_f32_bf16 v1, v25, v1, v5
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot2_f32_bf16 v1, v26, v2, v1
	v_dot2_f32_bf16 v1, v27, v3, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_dot2_f32_bf16 v47, v28, v4, v1
.LBB0_15:                               ;   in Loop: Header=BB0_16 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s20
	s_addk_co_i32 s33, 0x400
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s33, s4
	s_cbranch_scc1 .LBB0_27
.LBB0_16:                               ;   Parent Loop BB0_8 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_20 Depth 3
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s33, 0
	s_cselect_b32 s21, -1, 0
	s_add_co_i32 s20, s18, 0x8000
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s33, s20
	s_cselect_b32 s34, -1, 0
	s_wait_alu 0xfffe
	s_or_b32 s34, s21, s34
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s34
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB0_18
; %bb.17:                               ;   in Loop: Header=BB0_16 Depth=2
	s_and_saveexec_b32 s20, s2
	s_cbranch_execz .LBB0_15
	s_branch .LBB0_22
.LBB0_18:                               ;   in Loop: Header=BB0_16 Depth=2
	s_and_b32 s21, s21, exec_lo
	s_cselect_b32 s18, s18, s20
	s_mov_b32 s35, exec_lo
	s_wait_alu 0xfffe
	s_sub_co_i32 s20, s4, s18
	s_wait_loadcnt 0x0
	s_wait_alu 0xfffe
	s_min_u32 s34, s20, 0x8000
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu 0xfffe
	v_cmpx_gt_u32_e64 s34, v41
	s_cbranch_execz .LBB0_21
; %bb.19:                               ;   in Loop: Header=BB0_16 Depth=2
	v_dual_mov_b32 v1, v42 :: v_dual_mov_b32 v36, v41
	s_lshl_b64 s[20:21], s[18:19], 1
	s_mov_b32 s36, 0
	s_wait_alu 0xfffe
	s_add_nc_u64 s[20:21], s[16:17], s[20:21]
.LBB0_20:                               ;   Parent Loop BB0_8 Depth=1
                                        ;     Parent Loop BB0_16 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_lshlrev_b64_e32 v[2:3], 1, v[36:37]
	v_add_nc_u32_e32 v36, s27, v36
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v2, vcc_lo, s20, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s34, v36
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s36, vcc_lo, s36
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s30, v1
	s_and_not1_b32 exec_lo, exec_lo, s36
	s_cbranch_execnz .LBB0_20
.LBB0_21:                               ;   in Loop: Header=BB0_16 Depth=2
	s_or_b32 exec_lo, exec_lo, s35
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s20, s2
	s_cbranch_execz .LBB0_15
.LBB0_22:                               ;   in Loop: Header=BB0_16 Depth=2
	v_dual_mov_b32 v27, 0 :: v_dual_add_nc_u32 v50, s33, v40
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v21, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v36, s28, v50
	v_dual_mov_b32 v28, 0 :: v_dual_add_nc_u32 v51, 0x100, v50
	v_dual_mov_b32 v26, 0 :: v_dual_add_nc_u32 v49, 0x200, v50
	v_lshlrev_b64_e32 v[1:2], 1, v[36:37]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v36, s28, v51
	v_dual_mov_b32 v25, 0 :: v_dual_add_nc_u32 v48, 0x300, v50
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v19, 0
	v_lshlrev_b64_e32 v[3:4], 1, v[36:37]
	v_min_u32_e32 v36, s28, v49
	v_add_co_u32 v1, vcc_lo, v45, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, v46, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 1, v[36:37]
	v_min_u32_e32 v36, s28, v48
	v_add_co_u32 v3, vcc_lo, v45, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, v46, v4, vcc_lo
	v_lshlrev_b64_e32 v[7:8], 1, v[36:37]
	v_add_co_u32 v5, vcc_lo, v45, v5
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, v46, v6, vcc_lo
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v31, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v17, vcc_lo, v45, v7
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v18, null, v46, v8, vcc_lo
	s_clause 0x3
	global_load_b128 v[13:16], v[1:2], off
	global_load_b128 v[9:12], v[3:4], off
	global_load_b128 v[5:8], v[5:6], off
	global_load_b128 v[1:4], v[17:18], off
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v29, 0
	s_mov_b32 s21, exec_lo
	v_cmpx_gt_u32_e64 s4, v50
	s_cbranch_execz .LBB0_14
; %bb.23:                               ;   in Loop: Header=BB0_16 Depth=2
	v_subrev_nc_u32_e32 v17, s18, v50
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v21, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v17, 0 :: v_dual_lshlrev_b32 v36, 1, v17
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v25, 0
	ds_load_b128 v[29:32], v36
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v27, 0
	v_mov_b32_e32 v28, 0
	s_mov_b32 s34, exec_lo
	v_cmpx_gt_u32_e64 s4, v51
	s_cbranch_execz .LBB0_13
; %bb.24:                               ;   in Loop: Header=BB0_16 Depth=2
	ds_load_b128 v[17:20], v36 offset:512
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v28, 0
	s_mov_b32 s35, exec_lo
	v_cmpx_gt_u32_e64 s4, v49
	s_cbranch_execz .LBB0_12
; %bb.25:                               ;   in Loop: Header=BB0_16 Depth=2
	ds_load_b128 v[21:24], v36 offset:1024
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v28, 0
	s_mov_b32 s36, exec_lo
	v_cmpx_gt_u32_e64 s4, v48
	s_cbranch_execz .LBB0_11
; %bb.26:                               ;   in Loop: Header=BB0_16 Depth=2
	ds_load_b128 v[25:28], v36 offset:1536
	s_branch .LBB0_11
.LBB0_27:                               ;   in Loop: Header=BB0_8 Depth=1
	s_and_saveexec_b32 s18, s2
	s_cbranch_execz .LBB0_7
; %bb.28:                               ;   in Loop: Header=BB0_8 Depth=1
	v_add_f32_dpp v1, v47, v47 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v43
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v2, v34, v43, vcc_lo
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b32_e32 v2, 2, v2
	v_add_f32_dpp v1, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v2, v2, v1
	s_and_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_7
; %bb.29:                               ;   in Loop: Header=BB0_8 Depth=1
	s_wait_dscnt 0x0
	v_add_f32_e32 v3, v1, v2
	v_lshlrev_b64_e32 v[1:2], 1, v[38:39]
	s_and_not1_b32 vcc_lo, exec_lo, s29
	s_delay_alu instid0(VALU_DEP_2)
	v_mul_f32_e32 v3, s14, v3
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_6
; %bb.30:                               ;   in Loop: Header=BB0_8 Depth=1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_u32 v4, vcc_lo, s8, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, s9, v2, vcc_lo
	global_load_u16 v4, v[4:5], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v4, 16, v4
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v3, s15, v4
	s_branch .LBB0_6
.LBB0_31:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s25
	s_mov_b32 s18, 0
.LBB0_32:
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s18
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_58
; %bb.33:
	s_wait_dscnt 0x0
	v_and_b32_e32 v2, 0x3ff, v0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v36, 3, v2
	v_lshl_add_u32 v3, v35, 8, v36
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s4, v3
	s_cbranch_execz .LBB0_36
; %bb.34:
	v_lshlrev_b32_e32 v0, 9, v35
	v_lshlrev_b32_e32 v1, 4, v2
	s_mov_b32 s2, 0
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v5, s0, v0, v1
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v6, null, 0, 0, s0
	v_add_nc_u32_e32 v4, v0, v1
	s_wait_kmcnt 0x0
	v_add_co_u32 v0, vcc_lo, s16, v5
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s17, v6, vcc_lo
.LBB0_35:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[5:8], v[0:1], off
	v_add_nc_u32_e32 v3, 0x1000, v3
	v_add_co_u32 v0, vcc_lo, 0x2000, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s0, s4, v3
	s_wait_alu 0xfffe
	s_or_b32 s2, s0, s2
	s_wait_loadcnt 0x0
	ds_store_b128 v4, v[5:8]
	v_add_nc_u32_e32 v4, 0x2000, v4
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_35
.LBB0_36:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_u32_e64 s24, v35
	s_cbranch_execz .LBB0_58
; %bb.37:
	v_cmp_gt_u32_e32 vcc_lo, s5, v33
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_58
; %bb.38:
	s_cmp_eq_u64 s[12:13], 0
	v_mbcnt_lo_u32_b32 v35, -1, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[12:13], 0
	v_cmp_eq_u32_e64 s1, 31, v2
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s4, 0
	v_dual_mov_b32 v32, 0 :: v_dual_lshlrev_b32 v37, 4, v2
	s_wait_kmcnt 0x0
	s_cselect_b32 s16, -1, 0
	s_add_co_i32 s17, s4, -8
	s_cmp_neq_f32 s15, 0
	v_xor_b32_e32 v38, 16, v35
	s_mov_b32 s19, 0
	s_cselect_b32 s18, -1, 0
	s_branch .LBB0_41
.LBB0_39:                               ;   in Loop: Header=BB0_41 Depth=1
	v_lshlrev_b32_e32 v3, 16, v39
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_co_u32 v0, vcc_lo, s10, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	v_add_f32_e32 v3, v2, v3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_cndmask_b32_e64 v2, v3, v2, s0
	v_bfe_u32 v3, v2, 16, 1
	v_or_b32_e32 v4, 0x400000, v2
	v_cmp_u_f32_e32 vcc_lo, v2, v2
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add3_u32 v3, v3, v2, 0x7fff
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v2, v3, v4, vcc_lo
	global_store_d16_hi_b16 v[0:1], v2, off
.LBB0_40:                               ;   in Loop: Header=BB0_41 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s20
	v_add_nc_u32_e32 v33, s23, v33
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s5, v33
	s_or_b32 s19, vcc_lo, s19
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s19
	s_cbranch_execz .LBB0_58
.LBB0_41:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_49 Depth 2
	v_mov_b32_e32 v39, v32
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_43
; %bb.42:                               ;   in Loop: Header=BB0_41 Depth=1
	v_mov_b32_e32 v34, v32
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 1, v[33:34]
	v_add_co_u32 v0, vcc_lo, s12, v0
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, s13, v1, vcc_lo
	global_load_u16 v39, v[0:1], off
.LBB0_43:                               ;   in Loop: Header=BB0_41 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s16
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_54
; %bb.44:                               ;   in Loop: Header=BB0_41 Depth=1
	v_min_u32_e32 v0, s3, v33
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v42, v37
	s_mov_b32 s20, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_mad_co_u64_u32 v[0:1], null, v0, s22, 0
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v34, vcc_lo, s6, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v40, null, s7, v1, vcc_lo
	s_branch .LBB0_49
.LBB0_45:                               ;   in Loop: Header=BB0_49 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s26
.LBB0_46:                               ;   in Loop: Header=BB0_49 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s25
.LBB0_47:                               ;   in Loop: Header=BB0_49 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s24
.LBB0_48:                               ;   in Loop: Header=BB0_49 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s21
	s_wait_loadcnt_dscnt 0x300
	v_dot2_f32_bf16 v12, v28, v12, v41
	v_add_nc_u32_e32 v42, 0x800, v42
	s_addk_co_i32 s20, 0x400
	s_wait_alu 0xfffe
	s_cmp_lt_u32 s20, s4
	v_dot2_f32_bf16 v12, v29, v13, v12
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot2_f32_bf16 v12, v30, v14, v12
	v_dot2_f32_bf16 v12, v31, v15, v12
	s_wait_loadcnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot2_f32_bf16 v8, v16, v8, v12
	v_dot2_f32_bf16 v8, v17, v9, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot2_f32_bf16 v8, v18, v10, v8
	v_dot2_f32_bf16 v8, v19, v11, v8
	s_wait_loadcnt 0x1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot2_f32_bf16 v4, v20, v4, v8
	v_dot2_f32_bf16 v4, v21, v5, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot2_f32_bf16 v4, v22, v6, v4
	v_dot2_f32_bf16 v4, v23, v7, v4
	s_wait_loadcnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot2_f32_bf16 v0, v24, v0, v4
	v_dot2_f32_bf16 v0, v25, v1, v0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dot2_f32_bf16 v0, v26, v2, v0
	v_dot2_f32_bf16 v41, v27, v3, v0
	s_cbranch_scc0 .LBB0_55
.LBB0_49:                               ;   Parent Loop BB0_41 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_dual_mov_b32 v27, 0 :: v_dual_add_nc_u32 v18, s20, v36
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v20, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v31, s17, v18
	v_dual_mov_b32 v26, 0 :: v_dual_add_nc_u32 v45, 0x100, v18
	v_dual_mov_b32 v25, 0 :: v_dual_add_nc_u32 v44, 0x200, v18
	v_lshlrev_b64_e32 v[0:1], 1, v[31:32]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_min_u32_e32 v31, s17, v45
	v_dual_mov_b32 v24, 0 :: v_dual_add_nc_u32 v43, 0x300, v18
	v_mov_b32_e32 v19, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0
	v_lshlrev_b64_e32 v[2:3], 1, v[31:32]
	v_min_u32_e32 v31, s17, v44
	v_add_co_u32 v0, vcc_lo, v34, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v40, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[4:5], 1, v[31:32]
	v_min_u32_e32 v31, s17, v43
	v_add_co_u32 v2, vcc_lo, v34, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, v40, v3, vcc_lo
	v_lshlrev_b64_e32 v[6:7], 1, v[31:32]
	v_add_co_u32 v4, vcc_lo, v34, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v40, v5, vcc_lo
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v16, vcc_lo, v34, v6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v17, null, v40, v7, vcc_lo
	s_clause 0x3
	global_load_b128 v[12:15], v[0:1], off
	global_load_b128 v[8:11], v[2:3], off
	global_load_b128 v[4:7], v[4:5], off
	global_load_b128 v[0:3], v[16:17], off
	v_cmp_gt_u32_e32 vcc_lo, s4, v18
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v17, 0
	v_mov_b32_e32 v16, 0
	s_and_saveexec_b32 s21, vcc_lo
	s_cbranch_execz .LBB0_48
; %bb.50:                               ;   in Loop: Header=BB0_49 Depth=2
	ds_load_b128 v[28:31], v42
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v27, 0
	s_mov_b32 s24, exec_lo
	v_cmpx_gt_u32_e64 s4, v45
	s_cbranch_execz .LBB0_47
; %bb.51:                               ;   in Loop: Header=BB0_49 Depth=2
	ds_load_b128 v[16:19], v42 offset:512
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v27, 0
	s_mov_b32 s25, exec_lo
	v_cmpx_gt_u32_e64 s4, v44
	s_cbranch_execz .LBB0_46
; %bb.52:                               ;   in Loop: Header=BB0_49 Depth=2
	ds_load_b128 v[20:23], v42 offset:1024
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v27, 0
	s_mov_b32 s26, exec_lo
	v_cmpx_gt_u32_e64 s4, v43
	s_cbranch_execz .LBB0_45
; %bb.53:                               ;   in Loop: Header=BB0_49 Depth=2
	ds_load_b128 v[24:27], v42 offset:1536
	s_branch .LBB0_45
.LBB0_54:                               ;   in Loop: Header=BB0_41 Depth=1
	v_mov_b32_e32 v41, 0
.LBB0_55:                               ;   in Loop: Header=BB0_41 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)
	v_add_f32_dpp v0, v41, v41 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v38
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v1, v35, v38, vcc_lo
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshlrev_b32_e32 v1, 2, v1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v1, v1, v0
	s_and_saveexec_b32 s20, s1
	s_cbranch_execz .LBB0_40
; %bb.56:                               ;   in Loop: Header=BB0_41 Depth=1
	v_mov_b32_e32 v34, v32
	s_wait_dscnt 0x0
	v_add_f32_e32 v2, v0, v1
	s_and_not1_b32 vcc_lo, exec_lo, s18
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[0:1], 1, v[33:34]
	v_mul_f32_e32 v2, s14, v2
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_39
; %bb.57:                               ;   in Loop: Header=BB0_41 Depth=1
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_u32 v3, vcc_lo, s8, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s9, v1, vcc_lo
	global_load_u16 v3, v[3:4], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v3, 16, v3
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v2, s15, v3
	s_branch .LBB0_39
.LBB0_58:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201
		.amdhsa_group_segment_fixed_size 65536
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 80
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
		.amdhsa_next_free_sgpr 37
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 29
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
	.size	wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201, .Lfunc_end0-wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201
                                        ; -- End function
	.set wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201.num_vgpr, 52
	.set wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201.num_agpr, 0
	.set wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201.numbered_sgpr, 37
	.set wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201.num_named_barrier, 0
	.set wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201.private_seg_size, 0
	.set wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201.uses_vcc, 1
	.set wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201.uses_flat_scratch, 0
	.set wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201.has_dyn_sized_stack, 0
	.set wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201.has_recursion, 0
	.set wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 3708
; TotalNumSgprs: 39
; NumVgprs: 52
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 21
; NumSGPRsForWavesPerEU: 39
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
	.type	__hip_cuid_7c0f5989b3c13cac,@object ; @__hip_cuid_7c0f5989b3c13cac
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_7c0f5989b3c13cac
__hip_cuid_7c0f5989b3c13cac:
	.byte	0                               ; 0x0
	.size	__hip_cuid_7c0f5989b3c13cac, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_7c0f5989b3c13cac
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
    DataType: b
    DestDataType: b
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: False
    UseBeta: True
    Batched: True
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
            { type: address, semantic: AddressBias } ]
    macrotile: [32, 16, 8]
    threads: [32, 16, 1]
    grid: [ComputeUnits, One, One]
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 16, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  AssertSummationElementMultiple: 8
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
    .group_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 80
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     39
    .sgpr_spill_count: 0
    .symbol:         wvSplitK_vllm_bf16_tn_n1_y1u4_gfx1201.kd
    .vgpr_count:     52
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
