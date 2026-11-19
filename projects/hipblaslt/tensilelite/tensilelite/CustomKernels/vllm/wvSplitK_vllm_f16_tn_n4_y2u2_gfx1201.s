	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201 ; -- Begin function wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201
	.globl	wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201
	.p2align	8
	.type	wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201,@function
wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201:   ; @wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201
; %bb.0:
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x18
	s_load_b64 s[12:13], s[0:1], 0x0
	s_mov_b32 s6, -1
                                        ; implicit-def: $sgpr35
	s_wait_kmcnt 0x0
	s_lshl_b32 s4, s2, 5
	s_add_co_i32 s33, s13, -1
	s_delay_alu instid0(SALU_CYCLE_1)
	v_add_co_u32 v1, s3, s4, s33
	s_and_not1_b32 vcc_lo, exec_lo, s3
                                        ; implicit-def: $sgpr3
	v_readfirstlane_b32 s5, v1
	s_cbranch_vccz .LBB0_2
; %bb.1:
	s_cvt_f32_u32 s3, s4
	s_sub_co_i32 s6, 0, s4
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v1, s3
	v_readfirstlane_b32 s3, v1
	s_mul_f32 s3, s3, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s3, s3
	s_wait_alu 0xfffe
	s_mul_i32 s6, s6, s3
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_hi_u32 s6, s3, s6
	s_add_co_i32 s3, s3, s6
	s_wait_alu 0xfffe
	s_mul_hi_u32 s3, s5, s3
	s_wait_alu 0xfffe
	s_mul_i32 s6, s3, s4
	s_delay_alu instid0(SALU_CYCLE_1)
	s_sub_co_i32 s5, s5, s6
	s_add_co_i32 s6, s3, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s7, s5, s4
	s_cmp_ge_u32 s5, s4
	s_cselect_b32 s3, s6, s3
	s_cselect_b32 s5, s7, s5
	s_wait_alu 0xfffe
	s_add_co_i32 s6, s3, 1
	s_cmp_ge_u32 s5, s4
	s_cselect_b32 s4, s6, s3
	s_lshl_b32 s3, s2, 1
	s_wait_alu 0xfffe
	s_mul_i32 s4, s4, s3
	s_wait_alu 0xfffe
	s_cvt_f32_u32 s5, s4
	s_sub_co_i32 s6, 0, s4
	s_add_co_i32 s7, s4, s33
	s_wait_alu 0xfffe
	v_rcp_iflag_f32_e32 v1, s5
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_2)
	v_readfirstlane_b32 s5, v1
	s_mul_f32 s5, s5, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_cvt_u32_f32 s5, s5
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(SALU_CYCLE_1)
	s_mul_i32 s6, s6, s5
	s_mul_hi_u32 s6, s5, s6
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s5, s5, s6
	s_wait_alu 0xfffe
	s_mul_hi_u32 s5, s7, s5
	s_wait_alu 0xfffe
	s_mul_i32 s6, s5, s4
	s_sub_co_i32 s6, s7, s6
	s_add_co_i32 s7, s5, 1
	s_sub_co_i32 s8, s6, s4
	s_cmp_ge_u32 s6, s4
	s_cselect_b32 s5, s7, s5
	s_cselect_b32 s6, s8, s6
	s_wait_alu 0xfffe
	s_add_co_i32 s7, s5, 1
	s_cmp_ge_u32 s6, s4
	s_mov_b32 s6, 0
	s_cselect_b32 s4, s7, s5
	s_wait_alu 0xfffe
	s_max_u32 s35, s4, 4
.LBB0_2:
	s_clause 0x1
	s_load_b64 s[14:15], s[0:1], 0x48
	s_load_b64 s[16:17], s[0:1], 0x8
	s_and_not1_b32 vcc_lo, exec_lo, s6
	s_cbranch_vccnz .LBB0_4
; %bb.3:
	s_lshl_b32 s3, s2, 1
	s_mov_b32 s35, 4
.LBB0_4:
	s_clause 0x2
	s_load_b64 s[20:21], s[0:1], 0x10
	s_load_b64 s[18:19], s[0:1], 0x20
	s_load_b256 s[4:11], s[0:1], 0x28
	v_bfe_u32 v57, v0, 10, 10
	s_mul_i32 s0, s35, ttmp9
	s_mul_i32 s34, s35, s3
	s_cmp_gt_u32 s12, 0x2000
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v49, s0, v57, 1
	s_mov_b32 s0, -1
	v_cmp_gt_u32_e32 vcc_lo, s35, v57
	s_cbranch_scc0 .LBB0_59
; %bb.5:
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB0_58
; %bb.6:
	s_lshl_b32 s0, s35, 1
	s_mov_b32 s23, 0
	s_wait_alu 0xfffe
	s_cvt_f32_u32 s1, s0
	s_sub_co_i32 s2, 0, s0
	s_add_co_i32 s22, s0, s33
	s_wait_alu 0xfffe
	v_rcp_iflag_f32_e32 v1, s1
	s_delay_alu instid0(TRANS32_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_2)
	v_readfirstlane_b32 s1, v1
	s_mul_f32 s1, s1, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_cvt_u32_f32 s1, s1
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2)
	s_mul_i32 s2, s2, s1
	s_wait_alu 0xfffe
	s_mul_hi_u32 s2, s1, s2
	s_wait_alu 0xfffe
	s_add_co_i32 s1, s1, s2
	s_wait_alu 0xfffe
	s_mul_hi_u32 s1, s22, s1
	s_wait_alu 0xfffe
	s_mul_i32 s1, s1, s0
	s_wait_alu 0xfffe
	s_sub_co_i32 s1, s22, s1
	s_wait_alu 0xfffe
	s_sub_co_i32 s2, s1, s0
	s_cmp_ge_u32 s1, s0
	s_wait_alu 0xfffe
	s_cselect_b32 s1, s2, s1
	s_wait_alu 0xfffe
	s_sub_co_i32 s2, s1, s0
	s_cmp_ge_u32 s1, s0
	s_wait_alu 0xfffe
	s_cselect_b32 s0, s2, s1
	s_wait_alu 0xfffe
	s_sub_co_i32 s36, s22, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cmp_gt_u32_e32 vcc_lo, s36, v49
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_58
; %bb.7:
	v_dual_mov_b32 v52, v49 :: v_dual_and_b32 v1, 0x3ff, v0
	v_mbcnt_lo_u32_b32 v61, -1, 0
	s_wait_kmcnt 0x0
	s_cmp_lg_u64 s[14:15], 0
	s_mov_b32 s22, s19
	v_dual_mov_b32 v51, 0 :: v_dual_lshlrev_b32 v58, 3, v1
	v_cmp_eq_u32_e64 s1, 31, v1
	v_lshlrev_b32_e32 v1, 4, v1
	s_cselect_b32 s0, -1, 0
	s_lshl_b32 s37, s35, 8
	s_add_co_i32 s38, s12, -8
	s_cmp_neq_f32 s9, 0
	v_lshl_add_u32 v60, v57, 9, v1
	v_lshl_add_u32 v59, v57, 8, v58
	v_xor_b32_e32 v65, 16, v61
	s_mov_b32 s26, s10
	s_mov_b32 s27, s23
	v_add_nc_u32_e32 v62, 0x4000, v60
	v_add_nc_u32_e32 v63, 0x8000, v60
	v_add_nc_u32_e32 v64, 0xc000, v60
	s_mov_b32 s28, s11
	s_mov_b32 s29, s23
	s_cselect_b32 s39, -1, 0
	s_lshl_b64 s[24:25], s[22:23], 1
	s_lshl_b64 s[26:27], s[26:27], 1
	s_lshl_b64 s[28:29], s[28:29], 1
	s_lshl_b32 s40, s35, 9
	s_mov_b32 s41, 0
	s_branch .LBB0_10
.LBB0_8:                                ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e32 v1, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v5, v1
	v_cndmask_b32_e64 v1, v5, v1, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v1, v1
	global_store_b16 v[3:4], v1, off offset:2
.LBB0_9:                                ;   in Loop: Header=BB0_10 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	v_add_nc_u32_e32 v52, s34, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s36, v52
	s_or_b32 s41, vcc_lo, s41
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s41
	s_cbranch_execz .LBB0_58
.LBB0_10:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_18 Depth 2
                                        ;       Child Loop BB0_22 Depth 3
                                        ;       Child Loop BB0_24 Depth 3
                                        ;       Child Loop BB0_26 Depth 3
                                        ;       Child Loop BB0_28 Depth 3
	v_cmp_gt_u32_e64 s2, s13, v52
	v_or_b32_e32 v67, 1, v52
	s_wait_alu 0xfffe
	s_and_b32 s22, s0, s2
	s_wait_alu 0xfffe
	s_xor_b32 s22, s22, -1
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s30, s22
	s_wait_alu 0xfffe
	s_xor_b32 s22, exec_lo, s30
; %bb.11:                               ;   in Loop: Header=BB0_10 Depth=1
	v_or_b32_e32 v67, 1, v52
	s_delay_alu instid0(VALU_DEP_1)
	v_min_u32_e32 v50, s33, v67
; %bb.12:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_alu 0xfffe
	s_or_saveexec_b32 s22, s22
	s_wait_loadcnt 0x1
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v68, 0
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v66, 0
	s_wait_alu 0xfffe
	s_xor_b32 exec_lo, exec_lo, s22
	s_cbranch_execz .LBB0_14
; %bb.13:                               ;   in Loop: Header=BB0_10 Depth=1
	v_mov_b32_e32 v53, v51
	v_min_u32_e32 v50, s33, v67
	s_wait_dscnt 0x7
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[1:2], 1, v[52:53]
	v_lshlrev_b64_e32 v[3:4], 1, v[50:51]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v1, vcc_lo, s14, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s15, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v3, vcc_lo, s14, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s15, v4, vcc_lo
	s_clause 0x1
	global_load_u16 v68, v[1:2], off
	global_load_u16 v66, v[3:4], off
.LBB0_14:                               ;   in Loop: Header=BB0_10 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	s_wait_dscnt 0x7
	v_mad_co_u64_u32 v[1:2], null, v50, s18, 0
	v_min_u32_e32 v4, s33, v52
	v_dual_mov_b32 v72, 0 :: v_dual_mov_b32 v71, 0
	v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v75, 0
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v74, 0
	v_mad_co_u64_u32 v[2:3], null, v51, s18, v[2:3]
	v_mad_co_u64_u32 v[3:4], null, v4, s18, 0
	v_mov_b32_e32 v76, 0
	s_mov_b32 s22, 0
	s_wait_alu 0xfffe
	s_mov_b32 s42, s22
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[55:56], 1, v[1:2]
	v_lshlrev_b64_e32 v[53:54], 1, v[3:4]
	s_branch .LBB0_18
.LBB0_15:                               ;   in Loop: Header=BB0_18 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s43
.LBB0_16:                               ;   in Loop: Header=BB0_18 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s31
	s_wait_loadcnt_dscnt 0x303
	v_dot2_f32_f16 v50, v45, v9, v76
	s_wait_loadcnt 0x2
	v_dot2_f32_f16 v45, v45, v13, v74
	s_wait_dscnt 0x2
	v_dot2_f32_f16 v73, v41, v9, v73
	v_dot2_f32_f16 v41, v41, v13, v75
	s_wait_dscnt 0x1
	v_dot2_f32_f16 v69, v37, v9, v69
	v_dot2_f32_f16 v50, v46, v10, v50
	v_dot2_f32_f16 v37, v37, v13, v72
	v_dot2_f32_f16 v45, v46, v14, v45
	v_dot2_f32_f16 v46, v42, v10, v73
	v_dot2_f32_f16 v41, v42, v14, v41
	v_dot2_f32_f16 v42, v47, v11, v50
	v_dot2_f32_f16 v50, v38, v10, v69
	v_dot2_f32_f16 v37, v38, v14, v37
	v_dot2_f32_f16 v38, v47, v15, v45
	s_wait_dscnt 0x0
	v_dot2_f32_f16 v9, v29, v9, v71
	v_dot2_f32_f16 v42, v48, v12, v42
	v_dot2_f32_f16 v13, v29, v13, v70
	v_dot2_f32_f16 v45, v43, v11, v46
	v_dot2_f32_f16 v38, v48, v16, v38
	v_dot2_f32_f16 v9, v30, v10, v9
	s_wait_loadcnt 0x1
	v_dot2_f32_f16 v29, v33, v1, v42
	v_dot2_f32_f16 v10, v30, v14, v13
	v_dot2_f32_f16 v41, v43, v15, v41
	s_wait_loadcnt 0x0
	v_dot2_f32_f16 v33, v33, v5, v38
	v_dot2_f32_f16 v43, v39, v11, v50
	v_dot2_f32_f16 v37, v39, v15, v37
	v_dot2_f32_f16 v13, v34, v2, v29
	v_dot2_f32_f16 v9, v31, v11, v9
	v_dot2_f32_f16 v14, v34, v6, v33
	v_dot2_f32_f16 v10, v31, v15, v10
	v_dot2_f32_f16 v39, v44, v12, v45
	v_dot2_f32_f16 v41, v44, v16, v41
	v_dot2_f32_f16 v43, v40, v12, v43
	v_dot2_f32_f16 v37, v40, v16, v37
	v_dot2_f32_f16 v11, v35, v3, v13
	v_dot2_f32_f16 v13, v35, v7, v14
	v_dot2_f32_f16 v9, v32, v12, v9
	v_dot2_f32_f16 v10, v32, v16, v10
	v_dot2_f32_f16 v38, v25, v1, v39
	v_dot2_f32_f16 v76, v36, v4, v11
	v_dot2_f32_f16 v74, v36, v8, v13
	v_dot2_f32_f16 v11, v25, v5, v41
	v_dot2_f32_f16 v12, v21, v1, v43
	v_dot2_f32_f16 v13, v21, v5, v37
	v_dot2_f32_f16 v1, v17, v1, v9
	v_dot2_f32_f16 v5, v17, v5, v10
	v_dot2_f32_f16 v29, v26, v2, v38
	v_dot2_f32_f16 v9, v26, v6, v11
	v_dot2_f32_f16 v10, v22, v2, v12
	v_dot2_f32_f16 v11, v22, v6, v13
	v_dot2_f32_f16 v1, v18, v2, v1
	v_dot2_f32_f16 v2, v18, v6, v5
	v_dot2_f32_f16 v14, v27, v3, v29
	v_dot2_f32_f16 v5, v27, v7, v9
	v_dot2_f32_f16 v6, v23, v3, v10
	v_dot2_f32_f16 v9, v23, v7, v11
	v_dot2_f32_f16 v1, v19, v3, v1
	v_dot2_f32_f16 v2, v19, v7, v2
	v_dot2_f32_f16 v73, v28, v4, v14
	v_dot2_f32_f16 v75, v28, v8, v5
	v_dot2_f32_f16 v69, v24, v4, v6
	v_dot2_f32_f16 v72, v24, v8, v9
	v_dot2_f32_f16 v71, v20, v4, v1
	v_dot2_f32_f16 v70, v20, v8, v2
.LBB0_17:                               ;   in Loop: Header=BB0_18 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s30
	s_addk_co_i32 s42, 0x200
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s42, s12
	s_cbranch_scc1 .LBB0_33
.LBB0_18:                               ;   Parent Loop BB0_10 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_22 Depth 3
                                        ;       Child Loop BB0_24 Depth 3
                                        ;       Child Loop BB0_26 Depth 3
                                        ;       Child Loop BB0_28 Depth 3
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s42, 0
	s_cselect_b32 s31, -1, 0
	s_add_co_i32 s30, s22, 0x2000
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s42, s30
	s_cselect_b32 s43, -1, 0
	s_wait_alu 0xfffe
	s_or_b32 s43, s31, s43
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s43
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB0_20
; %bb.19:                               ;   in Loop: Header=BB0_18 Depth=2
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB0_17
	s_branch .LBB0_30
.LBB0_20:                               ;   in Loop: Header=BB0_18 Depth=2
	s_and_b32 s31, s31, exec_lo
	s_cselect_b32 s22, s22, s30
	s_mov_b32 s44, exec_lo
	s_wait_alu 0xfffe
	s_sub_co_i32 s30, s12, s22
	s_wait_loadcnt 0x0
	s_wait_alu 0xfffe
	s_min_u32 s43, s30, 0x2000
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu 0xfffe
	v_cmpx_gt_u32_e64 s43, v59
	s_cbranch_execz .LBB0_29
; %bb.21:                               ;   in Loop: Header=BB0_18 Depth=2
	v_dual_mov_b32 v1, v60 :: v_dual_mov_b32 v50, v59
	s_lshl_b64 s[30:31], s[22:23], 1
	s_mov_b32 s45, 0
	s_wait_alu 0xfffe
	s_add_nc_u64 s[30:31], s[20:21], s[30:31]
.LBB0_22:                               ;   Parent Loop BB0_10 Depth=1
                                        ;     Parent Loop BB0_18 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_lshlrev_b64_e32 v[2:3], 1, v[50:51]
	v_add_nc_u32_e32 v50, s37, v50
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v2, vcc_lo, s30, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s31, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s43, v50
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s45, vcc_lo, s45
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s40, v1
	s_and_not1_b32 exec_lo, exec_lo, s45
	s_cbranch_execnz .LBB0_22
; %bb.23:                               ;   in Loop: Header=BB0_18 Depth=2
	s_or_b32 exec_lo, exec_lo, s45
	v_dual_mov_b32 v1, v62 :: v_dual_mov_b32 v50, v59
	s_add_nc_u64 s[30:31], s[30:31], s[24:25]
	s_mov_b32 s45, 0
.LBB0_24:                               ;   Parent Loop BB0_10 Depth=1
                                        ;     Parent Loop BB0_18 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 1, v[50:51]
	v_add_nc_u32_e32 v50, s37, v50
	s_wait_alu 0xfffe
	v_add_co_u32 v2, vcc_lo, s30, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, s31, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s43, v50
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s45, vcc_lo, s45
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s40, v1
	s_and_not1_b32 exec_lo, exec_lo, s45
	s_cbranch_execnz .LBB0_24
; %bb.25:                               ;   in Loop: Header=BB0_18 Depth=2
	s_or_b32 exec_lo, exec_lo, s45
	v_dual_mov_b32 v1, v63 :: v_dual_mov_b32 v50, v59
	s_add_nc_u64 s[30:31], s[30:31], s[24:25]
	s_mov_b32 s45, 0
.LBB0_26:                               ;   Parent Loop BB0_10 Depth=1
                                        ;     Parent Loop BB0_18 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 1, v[50:51]
	v_add_nc_u32_e32 v50, s37, v50
	s_wait_alu 0xfffe
	v_add_co_u32 v2, vcc_lo, s30, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, s31, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s43, v50
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s45, vcc_lo, s45
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s40, v1
	s_and_not1_b32 exec_lo, exec_lo, s45
	s_cbranch_execnz .LBB0_26
; %bb.27:                               ;   in Loop: Header=BB0_18 Depth=2
	s_or_b32 exec_lo, exec_lo, s45
	v_dual_mov_b32 v1, v64 :: v_dual_mov_b32 v50, v59
	s_add_nc_u64 s[30:31], s[30:31], s[24:25]
	s_mov_b32 s45, 0
.LBB0_28:                               ;   Parent Loop BB0_10 Depth=1
                                        ;     Parent Loop BB0_18 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 1, v[50:51]
	v_add_nc_u32_e32 v50, s37, v50
	s_wait_alu 0xfffe
	v_add_co_u32 v2, vcc_lo, s30, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, s31, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s43, v50
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s45, vcc_lo, s45
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s40, v1
	s_and_not1_b32 exec_lo, exec_lo, s45
	s_cbranch_execnz .LBB0_28
.LBB0_29:                               ;   in Loop: Header=BB0_18 Depth=2
	s_or_b32 exec_lo, exec_lo, s44
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB0_17
.LBB0_30:                               ;   in Loop: Header=BB0_18 Depth=2
	v_dual_mov_b32 v20, 0 :: v_dual_add_nc_u32 v77, s42, v58
	s_wait_dscnt 0x5
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v31, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v50, s38, v77
	v_dual_mov_b32 v19, 0 :: v_dual_add_nc_u32 v78, 0x100, v77
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v29, 0
	v_lshlrev_b64_e32 v[1:2], 1, v[50:51]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_min_u32_e32 v50, s38, v78
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v21, 0
	v_add_co_u32 v5, vcc_lo, s16, v1
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, s17, v2, vcc_lo
	v_lshlrev_b64_e32 v[1:2], 1, v[50:51]
	v_add_co_u32 v3, vcc_lo, v5, v53
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_ci_u32_e64 v4, null, v6, v54, vcc_lo
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v39, 0
	v_add_co_u32 v7, vcc_lo, s16, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, s17, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v5, v55
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, v6, v56, vcc_lo
	v_add_co_u32 v5, vcc_lo, v7, v53
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, v8, v54, vcc_lo
	v_add_co_u32 v7, vcc_lo, v7, v55
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, v8, v56, vcc_lo
	s_clause 0x3
	global_load_b128 v[9:12], v[3:4], off
	global_load_b128 v[13:16], v[1:2], off
	global_load_b128 v[1:4], v[5:6], off
	global_load_b128 v[5:8], v[7:8], off
	v_cmp_gt_u32_e32 vcc_lo, s12, v77
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v45, 0
	s_and_saveexec_b32 s31, vcc_lo
	s_cbranch_execz .LBB0_16
; %bb.31:                               ;   in Loop: Header=BB0_18 Depth=2
	v_subrev_nc_u32_e32 v17, s22, v77
	v_cmp_gt_u32_e32 vcc_lo, s12, v78
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v25, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v33, 0 :: v_dual_lshlrev_b32 v50, 1, v17
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v27, 0
	ds_load_b128 v[45:48], v50
	ds_load_b128 v[41:44], v50 offset:16384
	ds_load_b128 v[37:40], v50 offset:32768
	ds_load_b128 v[29:32], v50 offset:49152
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v19, 0
	v_mov_b32_e32 v20, 0
	s_and_saveexec_b32 s43, vcc_lo
	s_cbranch_execz .LBB0_15
; %bb.32:                               ;   in Loop: Header=BB0_18 Depth=2
	ds_load_b128 v[33:36], v50 offset:512
	ds_load_b128 v[25:28], v50 offset:16896
	ds_load_b128 v[21:24], v50 offset:33280
	ds_load_b128 v[17:20], v50 offset:49664
	s_branch .LBB0_15
.LBB0_33:                               ;   in Loop: Header=BB0_10 Depth=1
	s_and_saveexec_b32 s22, s2
	s_cbranch_execz .LBB0_9
; %bb.34:                               ;   in Loop: Header=BB0_10 Depth=1
	v_add_f32_dpp v3, v73, v73 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v76, v76 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v74, v74 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v75, v75 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x0
	v_add_f32_dpp v6, v72, v72 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v71, v71 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v70, v70 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v65
	v_add_f32_dpp v5, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v15, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v69, v69 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v4, v61, v65, vcc_lo
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshlrev_b32_e32 v4, 2, v4
	v_add_f32_dpp v1, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v18, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v14, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v8, v8 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v2, v4, v1
	ds_bpermute_b32 v19, v4, v18
	ds_bpermute_b32 v17, v4, v15
	ds_bpermute_b32 v16, v4, v14
	ds_bpermute_b32 v13, v4, v12
	ds_bpermute_b32 v11, v4, v10
	ds_bpermute_b32 v8, v4, v7
	ds_bpermute_b32 v6, v4, v5
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	s_and_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_9
; %bb.35:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x7
	v_add_f32_e32 v9, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v53, v51 :: v_dual_mul_f32 v20, s8, v9
	v_lshlrev_b64_e32 v[3:4], 1, v[52:53]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s4, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s5, v4, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_37
; %bb.36:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v9, v[1:2], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v20, s9, v9, v20 op_sel_hi:[0,1,0]
.LBB0_37:                               ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e32 v9, v68
	v_add_co_u32 v3, vcc_lo, s6, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s7, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v21, v20, v9
	v_cmp_gt_u32_e64 s2, s13, v67
	v_cndmask_b32_e64 v20, v20, v21, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v20, v20
	global_store_b16 v[3:4], v20, off
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB0_41
; %bb.38:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x6
	v_add_f32_e32 v18, v18, v19
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v18, s8, v18
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_40
; %bb.39:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v19, v[1:2], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v18, s9, v19, v18 op_sel_hi:[0,1,0]
.LBB0_40:                               ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e32 v19, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v19, v18, v19
	v_cndmask_b32_e64 v18, v18, v19, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18, v18
	global_store_b16 v[3:4], v18, off offset:2
.LBB0_41:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s30
	s_wait_dscnt 0x5
	v_add_f32_e32 v15, v15, v17
	v_add_co_u32 v1, vcc_lo, v1, s26
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s27, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v15, s8, v15
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_43
; %bb.42:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v17, v[1:2], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v15, s9, v17, v15 op_sel_hi:[0,1,0]
.LBB0_43:                               ;   in Loop: Header=BB0_10 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v17, v15, v9
	v_add_co_u32 v3, vcc_lo, v3, s28
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s29, v4, vcc_lo
	v_cndmask_b32_e64 v15, v15, v17, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v15, v15
	global_store_b16 v[3:4], v15, off
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB0_47
; %bb.44:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x4
	v_add_f32_e32 v14, v14, v16
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v14, s8, v14
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_46
; %bb.45:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v15, v[1:2], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v14, s9, v15, v14 op_sel_hi:[0,1,0]
.LBB0_46:                               ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e32 v15, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, v14, v15
	v_cndmask_b32_e64 v14, v14, v15, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v14, v14
	global_store_b16 v[3:4], v14, off offset:2
.LBB0_47:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s30
	s_wait_dscnt 0x3
	v_add_f32_e32 v12, v12, v13
	v_add_co_u32 v1, vcc_lo, v1, s26
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s27, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v12, s8, v12
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_49
; %bb.48:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v13, v[1:2], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v12, s9, v13, v12 op_sel_hi:[0,1,0]
.LBB0_49:                               ;   in Loop: Header=BB0_10 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v13, v12, v9
	v_add_co_u32 v3, vcc_lo, v3, s28
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s29, v4, vcc_lo
	v_cndmask_b32_e64 v12, v12, v13, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v12, v12
	global_store_b16 v[3:4], v12, off
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB0_53
; %bb.50:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x2
	v_add_f32_e32 v10, v10, v11
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v10, s8, v10
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_52
; %bb.51:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v11, v[1:2], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v10, s9, v11, v10 op_sel_hi:[0,1,0]
.LBB0_52:                               ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e32 v11, v66
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v11, v10, v11
	v_cndmask_b32_e64 v10, v10, v11, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v10, v10
	global_store_b16 v[3:4], v10, off offset:2
.LBB0_53:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s30
	s_wait_dscnt 0x1
	v_add_f32_e32 v7, v7, v8
	v_add_co_u32 v1, vcc_lo, v1, s26
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s27, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v7, s8, v7
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_55
; %bb.54:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v8, v[1:2], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v7, s9, v8, v7 op_sel_hi:[0,1,0]
.LBB0_55:                               ;   in Loop: Header=BB0_10 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v8, v7, v9
	v_add_co_u32 v3, vcc_lo, v3, s28
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s29, v4, vcc_lo
	v_cndmask_b32_e64 v7, v7, v8, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v7, v7
	global_store_b16 v[3:4], v7, off
	s_and_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB0_9
; %bb.56:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v5, s8, v5
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_8
; %bb.57:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v1, v[1:2], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v5, s9, v1, v5 op_sel_hi:[0,1,0]
	s_branch .LBB0_8
.LBB0_58:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_mov_b32 s0, 0
.LBB0_59:
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_109
; %bb.60:
	s_wait_dscnt 0x7
	v_and_b32_e32 v2, 0x3ff, v0
	v_lshlrev_b32_e32 v4, 9, v57
	s_mov_b32 s3, 0
	s_wait_kmcnt 0x0
	s_mov_b32 s2, s19
	v_lshlrev_b32_e32 v54, 3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v3, v57, 8, v54
	v_cmp_gt_u32_e32 vcc_lo, s12, v3
	s_and_saveexec_b32 s19, vcc_lo
	s_cbranch_execz .LBB0_67
; %bb.61:
	v_dual_mov_b32 v9, v3 :: v_dual_lshlrev_b32 v0, 4, v2
	s_wait_alu 0xfffe
	s_mov_b32 s22, s3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v5, s0, v4, v0
	s_wait_dscnt 0x0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v6, null, 0, 0, s0
	v_add_nc_u32_e32 v7, v4, v0
	v_add_co_u32 v0, s0, s20, v5
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v1, null, s21, v6, s0
	v_mov_b32_e32 v8, v7
.LBB0_62:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[10:13], v[0:1], off
	v_add_nc_u32_e32 v9, 0x1000, v9
	v_add_co_u32 v0, s0, 0x2000, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v1, null, 0, v1, s0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s1, s12, v9
	s_wait_alu 0xfffe
	s_or_b32 s22, s1, s22
	s_wait_loadcnt 0x0
	ds_store_b128 v8, v[10:13]
	v_add_nc_u32_e32 v8, 0x2000, v8
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s22
	s_cbranch_execnz .LBB0_62
; %bb.63:
	s_or_b32 exec_lo, exec_lo, s22
	s_lshl_b64 s[0:1], s[2:3], 1
	v_lshl_add_u32 v8, s12, 1, v7
	s_wait_alu 0xfffe
	s_add_nc_u64 s[0:1], s[20:21], s[0:1]
	v_mov_b32_e32 v9, v3
	s_wait_alu 0xfffe
	v_add_co_u32 v0, s0, s0, v5
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v1, null, s1, v6, s0
	s_mov_b32 s22, 0
.LBB0_64:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[10:13], v[0:1], off
	v_add_nc_u32_e32 v9, 0x1000, v9
	v_add_co_u32 v0, s0, 0x2000, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v1, null, 0, v1, s0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s1, s12, v9
	s_wait_alu 0xfffe
	s_or_b32 s22, s1, s22
	s_wait_loadcnt 0x0
	ds_store_b128 v8, v[10:13]
	v_add_nc_u32_e32 v8, 0x2000, v8
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s22
	s_cbranch_execnz .LBB0_64
; %bb.65:
	s_or_b32 exec_lo, exec_lo, s22
	s_lshl_b64 s[0:1], s[2:3], 2
	v_lshl_add_u32 v7, s12, 2, v7
	s_wait_alu 0xfffe
	s_add_nc_u64 s[0:1], s[20:21], s[0:1]
	s_mov_b32 s22, 0
	s_wait_alu 0xfffe
	v_add_co_u32 v0, s0, s0, v5
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v1, null, s1, v6, s0
	v_mov_b32_e32 v5, v3
.LBB0_66:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[8:11], v[0:1], off
	v_add_nc_u32_e32 v5, 0x1000, v5
	v_add_co_u32 v0, s0, 0x2000, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v1, null, 0, v1, s0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s1, s12, v5
	s_or_b32 s22, s1, s22
	s_wait_loadcnt 0x0
	ds_store_b128 v7, v[8:11]
	v_add_nc_u32_e32 v7, 0x2000, v7
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s22
	s_cbranch_execnz .LBB0_66
.LBB0_67:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s19
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_70
; %bb.68:
	v_lshlrev_b32_e32 v0, 4, v2
	s_mul_i32 s0, s12, 6
	s_mul_u64 s[2:3], s[2:3], 6
	s_wait_alu 0xfffe
	s_add_nc_u64 s[2:3], s[20:21], s[2:3]
	v_add3_u32 v5, v4, v0, s0
	v_add_co_u32 v0, s0, v4, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v1, null, 0, 0, s0
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s2, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s3, v1, vcc_lo
	s_mov_b32 s2, 0
.LBB0_69:                               ; =>This Inner Loop Header: Depth=1
	s_wait_dscnt 0x0
	global_load_b128 v[6:9], v[0:1], off
	v_add_nc_u32_e32 v3, 0x1000, v3
	v_add_co_u32 v0, vcc_lo, 0x2000, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s0, s12, v3
	s_wait_alu 0xfffe
	s_or_b32 s2, s0, s2
	s_wait_loadcnt 0x0
	ds_store_b128 v5, v[6:9]
	v_add_nc_u32_e32 v5, 0x2000, v5
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_69
.LBB0_70:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_u32_e64 s35, v57
	s_cbranch_execz .LBB0_109
; %bb.71:
	v_cmp_gt_u32_e32 vcc_lo, s13, v49
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_109
; %bb.72:
	s_cmp_eq_u64 s[14:15], 0
	v_mbcnt_lo_u32_b32 v55, -1, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[14:15], 0
	s_mov_b32 s3, 0
	s_cselect_b32 s19, -1, 0
	s_cmp_lg_u32 s12, 0
	v_cmp_eq_u32_e64 s1, 31, v2
	s_cselect_b32 s22, -1, 0
	s_add_co_i32 s23, s12, -8
	s_cmp_neq_f32 s9, 0
	v_lshlrev_b32_e32 v56, 4, v2
	v_mov_b32_e32 v48, 0
	v_xor_b32_e32 v57, 16, v55
	s_mov_b32 s2, s10
	s_mov_b32 s20, s11
	s_wait_alu 0xfffe
	s_mov_b32 s21, s3
	s_cselect_b32 s24, -1, 0
	s_lshl_b32 s25, s12, 1
	s_lshl_b32 s26, s12, 2
	s_lshl_b64 s[10:11], s[2:3], 1
	s_wait_alu 0xfffe
	s_lshl_b64 s[20:21], s[20:21], 1
	s_mul_i32 s27, s12, 6
	s_branch .LBB0_75
.LBB0_73:                               ;   in Loop: Header=BB0_75 Depth=1
	v_cvt_f32_f16_e32 v0, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v4, v0
	v_cndmask_b32_e64 v0, v0, v4, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v0, v0
	global_store_b16 v[2:3], v0, off offset:2
.LBB0_74:                               ;   in Loop: Header=BB0_75 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s28
	v_add_nc_u32_e32 v49, s34, v49
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s13, v49
	s_or_b32 s3, vcc_lo, s3
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s3
	s_cbranch_execz .LBB0_109
.LBB0_75:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_81 Depth 2
	v_or_b32_e32 v59, 1, v49
	v_mov_b32_e32 v60, v48
	v_mov_b32_e32 v58, v48
	s_and_not1_b32 vcc_lo, exec_lo, s19
	s_delay_alu instid0(VALU_DEP_3)
	v_min_u32_e32 v47, s33, v59
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_77
; %bb.76:                               ;   in Loop: Header=BB0_75 Depth=1
	v_mov_b32_e32 v50, v48
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 1, v[47:48]
	s_wait_dscnt 0x7
	v_lshlrev_b64_e32 v[0:1], 1, v[49:50]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s14, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s15, v1, vcc_lo
	v_add_co_u32 v2, vcc_lo, s14, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s15, v3, vcc_lo
	s_clause 0x1
	global_load_u16 v60, v[0:1], off
	global_load_u16 v58, v[2:3], off
.LBB0_77:                               ;   in Loop: Header=BB0_75 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s22
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_84
; %bb.78:                               ;   in Loop: Header=BB0_75 Depth=1
	v_min_u32_e32 v2, s33, v49
	s_wait_dscnt 0x7
	v_mad_co_u64_u32 v[0:1], null, v47, s18, 0
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v64, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mad_co_u64_u32 v[2:3], null, v2, s18, 0
	v_dual_mov_b32 v63, v56 :: v_dual_mov_b32 v62, 0
	v_lshlrev_b64_e32 v[50:51], 1, v[0:1]
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v67, 0
	v_lshlrev_b64_e32 v[52:53], 1, v[2:3]
	v_mov_b32_e32 v69, 0
	s_mov_b32 s2, 0
	s_branch .LBB0_81
.LBB0_79:                               ;   in Loop: Header=BB0_81 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s29
.LBB0_80:                               ;   in Loop: Header=BB0_81 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s28
	s_wait_loadcnt_dscnt 0x301
	v_dot2_f32_f16 v69, v44, v8, v69
	s_wait_loadcnt 0x2
	v_dot2_f32_f16 v44, v44, v12, v67
	v_dot2_f32_f16 v66, v40, v8, v66
	v_dot2_f32_f16 v40, v40, v12, v68
	v_dot2_f32_f16 v65, v36, v8, v65
	v_dot2_f32_f16 v67, v45, v9, v69
	v_dot2_f32_f16 v36, v36, v12, v64
	v_dot2_f32_f16 v44, v45, v13, v44
	v_dot2_f32_f16 v45, v41, v9, v66
	v_dot2_f32_f16 v40, v41, v13, v40
	v_dot2_f32_f16 v41, v46, v10, v67
	v_dot2_f32_f16 v64, v37, v9, v65
	v_dot2_f32_f16 v36, v37, v13, v36
	v_dot2_f32_f16 v37, v46, v14, v44
	s_wait_dscnt 0x0
	v_dot2_f32_f16 v8, v28, v8, v62
	v_dot2_f32_f16 v41, v47, v11, v41
	v_dot2_f32_f16 v12, v28, v12, v61
	v_dot2_f32_f16 v44, v42, v10, v45
	v_dot2_f32_f16 v37, v47, v15, v37
	v_dot2_f32_f16 v8, v29, v9, v8
	s_wait_loadcnt 0x1
	v_dot2_f32_f16 v28, v32, v0, v41
	v_dot2_f32_f16 v9, v29, v13, v12
	v_dot2_f32_f16 v40, v42, v14, v40
	s_wait_loadcnt 0x0
	v_dot2_f32_f16 v32, v32, v4, v37
	v_dot2_f32_f16 v42, v38, v10, v64
	v_dot2_f32_f16 v36, v38, v14, v36
	v_dot2_f32_f16 v12, v33, v1, v28
	v_dot2_f32_f16 v8, v30, v10, v8
	v_dot2_f32_f16 v13, v33, v5, v32
	v_dot2_f32_f16 v9, v30, v14, v9
	v_dot2_f32_f16 v38, v43, v11, v44
	v_dot2_f32_f16 v40, v43, v15, v40
	v_dot2_f32_f16 v42, v39, v11, v42
	v_dot2_f32_f16 v36, v39, v15, v36
	v_dot2_f32_f16 v10, v34, v2, v12
	v_dot2_f32_f16 v12, v34, v6, v13
	v_dot2_f32_f16 v8, v31, v11, v8
	v_dot2_f32_f16 v9, v31, v15, v9
	v_dot2_f32_f16 v37, v24, v0, v38
	v_dot2_f32_f16 v69, v35, v3, v10
	v_dot2_f32_f16 v67, v35, v7, v12
	v_dot2_f32_f16 v10, v24, v4, v40
	v_dot2_f32_f16 v11, v16, v0, v42
	v_dot2_f32_f16 v12, v16, v4, v36
	v_dot2_f32_f16 v0, v20, v0, v8
	v_dot2_f32_f16 v4, v20, v4, v9
	v_dot2_f32_f16 v28, v25, v1, v37
	v_dot2_f32_f16 v8, v25, v5, v10
	v_dot2_f32_f16 v9, v17, v1, v11
	v_dot2_f32_f16 v10, v17, v5, v12
	v_dot2_f32_f16 v0, v21, v1, v0
	v_dot2_f32_f16 v1, v21, v5, v4
	v_dot2_f32_f16 v13, v26, v2, v28
	v_dot2_f32_f16 v4, v26, v6, v8
	v_dot2_f32_f16 v5, v18, v2, v9
	v_dot2_f32_f16 v8, v18, v6, v10
	v_dot2_f32_f16 v0, v22, v2, v0
	v_dot2_f32_f16 v1, v22, v6, v1
	v_dot2_f32_f16 v66, v27, v3, v13
	v_dot2_f32_f16 v68, v27, v7, v4
	v_dot2_f32_f16 v65, v19, v3, v5
	v_dot2_f32_f16 v64, v19, v7, v8
	v_dot2_f32_f16 v62, v23, v3, v0
	v_dot2_f32_f16 v61, v23, v7, v1
	v_add_nc_u32_e32 v63, 0x400, v63
	s_addk_co_i32 s2, 0x200
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s2, s12
	s_cbranch_scc1 .LBB0_85
.LBB0_81:                               ;   Parent Loop BB0_75 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_dscnt 0x5
	s_wait_alu 0xfffe
	v_dual_mov_b32 v23, 0 :: v_dual_add_nc_u32 v16, s2, v54
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v31, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v29, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v47, s23, v16
	v_dual_mov_b32 v21, 0 :: v_dual_add_nc_u32 v70, 0x100, v16
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v19, 0
	v_lshlrev_b64_e32 v[0:1], 1, v[47:48]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_min_u32_e32 v47, s23, v70
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v39, 0
	v_add_co_u32 v4, vcc_lo, s16, v0
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo
	v_lshlrev_b64_e32 v[0:1], 1, v[47:48]
	v_add_co_u32 v2, vcc_lo, v4, v52
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_add_co_ci_u32_e64 v3, null, v5, v53, vcc_lo
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v27, 0
	v_add_co_u32 v6, vcc_lo, s16, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v7, null, s17, v1, vcc_lo
	v_add_co_u32 v0, vcc_lo, v4, v50
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v5, v51, vcc_lo
	v_add_co_u32 v4, vcc_lo, v6, v52
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v7, v53, vcc_lo
	v_add_co_u32 v6, vcc_lo, v6, v50
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v7, null, v7, v51, vcc_lo
	s_clause 0x3
	global_load_b128 v[8:11], v[2:3], off
	global_load_b128 v[12:15], v[0:1], off
	global_load_b128 v[0:3], v[4:5], off
	global_load_b128 v[4:7], v[6:7], off
	v_cmp_gt_u32_e32 vcc_lo, s12, v16
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v45, 0
	v_mov_b32_e32 v46, 0
	v_mov_b32_e32 v44, 0
	s_and_saveexec_b32 s28, vcc_lo
	s_cbranch_execz .LBB0_80
; %bb.82:                               ;   in Loop: Header=BB0_81 Depth=2
	v_dual_mov_b32 v32, 0 :: v_dual_add_nc_u32 v73, s25, v63
	v_dual_mov_b32 v33, 0 :: v_dual_add_nc_u32 v72, s26, v63
	v_dual_mov_b32 v34, 0 :: v_dual_add_nc_u32 v71, s27, v63
	ds_load_b128 v[40:43], v73
	ds_load_b128 v[36:39], v72
	ds_load_b128 v[44:47], v63
	ds_load_b128 v[28:31], v71
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v16, 0
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v20, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v22, 0
	v_mov_b32_e32 v23, 0
	s_mov_b32 s29, exec_lo
	v_cmpx_gt_u32_e64 s12, v70
	s_cbranch_execz .LBB0_79
; %bb.83:                               ;   in Loop: Header=BB0_81 Depth=2
	ds_load_b128 v[24:27], v73 offset:512
	ds_load_b128 v[16:19], v72 offset:512
	ds_load_b128 v[32:35], v63 offset:512
	ds_load_b128 v[20:23], v71 offset:512
	s_branch .LBB0_79
.LBB0_84:                               ;   in Loop: Header=BB0_75 Depth=1
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v64, 0 :: v_dual_mov_b32 v65, 0
.LBB0_85:                               ;   in Loop: Header=BB0_75 Depth=1
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_dpp v2, v66, v66 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v69, v69 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x7
	v_add_f32_dpp v1, v67, v67 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v68, v68 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x0
	v_add_f32_dpp v5, v64, v64 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v62, v62 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v61, v61 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v57
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v14, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v65, v65 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v55, v57, vcc_lo
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
	v_add_f32_dpp v11, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v9, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v7, v7 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v1, v3, v0
	ds_bpermute_b32 v18, v3, v17
	ds_bpermute_b32 v16, v3, v14
	ds_bpermute_b32 v15, v3, v13
	ds_bpermute_b32 v12, v3, v11
	ds_bpermute_b32 v10, v3, v9
	ds_bpermute_b32 v7, v3, v6
	ds_bpermute_b32 v5, v3, v4
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	s_and_saveexec_b32 s28, s1
	s_cbranch_execz .LBB0_74
; %bb.86:                               ;   in Loop: Header=BB0_75 Depth=1
	s_wait_dscnt 0x7
	v_add_f32_e32 v8, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v50, v48 :: v_dual_mul_f32 v19, s8, v8
	v_lshlrev_b64_e32 v[2:3], 1, v[49:50]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s4, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s5, v3, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_88
; %bb.87:                               ;   in Loop: Header=BB0_75 Depth=1
	global_load_u16 v8, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v19, s9, v8, v19 op_sel_hi:[0,1,0]
.LBB0_88:                               ;   in Loop: Header=BB0_75 Depth=1
	v_cvt_f32_f16_e32 v8, v60
	v_add_co_u32 v2, vcc_lo, s6, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v20, v19, v8
	v_cmp_gt_u32_e64 s2, s13, v59
	v_cndmask_b32_e64 v19, v20, v19, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v19, v19
	global_store_b16 v[2:3], v19, off
	s_and_saveexec_b32 s29, s2
	s_cbranch_execz .LBB0_92
; %bb.89:                               ;   in Loop: Header=BB0_75 Depth=1
	s_wait_dscnt 0x6
	v_add_f32_e32 v17, v17, v18
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v17, s8, v17
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_91
; %bb.90:                               ;   in Loop: Header=BB0_75 Depth=1
	global_load_u16 v18, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v17, s9, v18, v17 op_sel_hi:[0,1,0]
.LBB0_91:                               ;   in Loop: Header=BB0_75 Depth=1
	v_cvt_f32_f16_e32 v18, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v18, v17, v18
	v_cndmask_b32_e64 v17, v18, v17, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v17, v17
	global_store_b16 v[2:3], v17, off offset:2
.LBB0_92:                               ;   in Loop: Header=BB0_75 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s29
	s_wait_dscnt 0x5
	v_add_f32_e32 v14, v14, v16
	v_add_co_u32 v0, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v14, s8, v14
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_94
; %bb.93:                               ;   in Loop: Header=BB0_75 Depth=1
	global_load_u16 v16, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v14, s9, v16, v14 op_sel_hi:[0,1,0]
.LBB0_94:                               ;   in Loop: Header=BB0_75 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v16, v14, v8
	v_add_co_u32 v2, vcc_lo, v2, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	v_cndmask_b32_e64 v14, v16, v14, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v14, v14
	global_store_b16 v[2:3], v14, off
	s_and_saveexec_b32 s29, s2
	s_cbranch_execz .LBB0_98
; %bb.95:                               ;   in Loop: Header=BB0_75 Depth=1
	s_wait_dscnt 0x4
	v_add_f32_e32 v13, v13, v15
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v13, s8, v13
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_97
; %bb.96:                               ;   in Loop: Header=BB0_75 Depth=1
	global_load_u16 v14, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v13, s9, v14, v13 op_sel_hi:[0,1,0]
.LBB0_97:                               ;   in Loop: Header=BB0_75 Depth=1
	v_cvt_f32_f16_e32 v14, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v14, v13, v14
	v_cndmask_b32_e64 v13, v14, v13, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v13, v13
	global_store_b16 v[2:3], v13, off offset:2
.LBB0_98:                               ;   in Loop: Header=BB0_75 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s29
	s_wait_dscnt 0x3
	v_add_f32_e32 v11, v11, v12
	v_add_co_u32 v0, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v11, s8, v11
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_100
; %bb.99:                               ;   in Loop: Header=BB0_75 Depth=1
	global_load_u16 v12, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v11, s9, v12, v11 op_sel_hi:[0,1,0]
.LBB0_100:                              ;   in Loop: Header=BB0_75 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v12, v11, v8
	v_add_co_u32 v2, vcc_lo, v2, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	v_cndmask_b32_e64 v11, v12, v11, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v11, v11
	global_store_b16 v[2:3], v11, off
	s_and_saveexec_b32 s29, s2
	s_cbranch_execz .LBB0_104
; %bb.101:                              ;   in Loop: Header=BB0_75 Depth=1
	s_wait_dscnt 0x2
	v_add_f32_e32 v9, v9, v10
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, s8, v9
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_103
; %bb.102:                              ;   in Loop: Header=BB0_75 Depth=1
	global_load_u16 v10, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v9, s9, v10, v9 op_sel_hi:[0,1,0]
.LBB0_103:                              ;   in Loop: Header=BB0_75 Depth=1
	v_cvt_f32_f16_e32 v10, v58
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v10, v9, v10
	v_cndmask_b32_e64 v9, v10, v9, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9, v9
	global_store_b16 v[2:3], v9, off offset:2
.LBB0_104:                              ;   in Loop: Header=BB0_75 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s29
	s_wait_dscnt 0x1
	v_add_f32_e32 v6, v6, v7
	v_add_co_u32 v0, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v6, s8, v6
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_106
; %bb.105:                              ;   in Loop: Header=BB0_75 Depth=1
	global_load_u16 v7, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v6, s9, v7, v6 op_sel_hi:[0,1,0]
.LBB0_106:                              ;   in Loop: Header=BB0_75 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v7, v6, v8
	v_add_co_u32 v2, vcc_lo, v2, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	v_cndmask_b32_e64 v6, v7, v6, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v6, v6
	global_store_b16 v[2:3], v6, off
	s_and_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB0_74
; %bb.107:                              ;   in Loop: Header=BB0_75 Depth=1
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v4, s8, v4
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_73
; %bb.108:                              ;   in Loop: Header=BB0_75 Depth=1
	global_load_u16 v0, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v4, s9, v0, v4 op_sel_hi:[0,1,0]
	s_branch .LBB0_73
.LBB0_109:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201
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
		.amdhsa_next_free_sgpr 46
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 61
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
	.size	wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201, .Lfunc_end0-wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201
                                        ; -- End function
	.set wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201.num_vgpr, 79
	.set wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201.num_agpr, 0
	.set wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201.numbered_sgpr, 46
	.set wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201.num_named_barrier, 0
	.set wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201.private_seg_size, 0
	.set wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201.uses_vcc, 1
	.set wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201.uses_flat_scratch, 0
	.set wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201.has_dyn_sized_stack, 0
	.set wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201.has_recursion, 0
	.set wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 7764
; TotalNumSgprs: 48
; NumVgprs: 79
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 21
; NumSGPRsForWavesPerEU: 48
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
	.type	__hip_cuid_7586aa0064efd35e,@object ; @__hip_cuid_7586aa0064efd35e
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_7586aa0064efd35e
__hip_cuid_7586aa0064efd35e:
	.byte	0                               ; 0x0
	.size	__hip_cuid_7586aa0064efd35e, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_7586aa0064efd35e
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
    DataType: h
    DestDataType: h
    ComputeDataType: s
    HighPrecisionAccumulate: True
    TransposeA: True
    TransposeB: False
    UseBeta: True
    Batched: True
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
            { type: address, semantic: AddressBias } ]
    macrotile: [32, 16, 8]
    threads: [32, 16, 1]
    grid: [ComputeUnits, One, One]
    uniformSummationOrder: true
  MatrixInstruction: [16, 16, 16, 1]
  EnableMatrixInstruction: True
  MIWaveTile: [1, 1]
  AssertSummationElementMultiple: 8
  AssertSizeEqual: { 1: 4, 2: 1 }
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
    .name:           wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     48
    .sgpr_spill_count: 0
    .symbol:         wvSplitK_vllm_f16_tn_n4_y2u2_gfx1201.kd
    .vgpr_count:     79
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
