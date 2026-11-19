	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201 ; -- Begin function wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201
	.globl	wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201
	.p2align	8
	.type	wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201,@function
wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201:  ; @wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201
; %bb.0:
	s_clause 0x2
	s_load_b32 s2, s[0:1], 0x18
	s_load_b128 s[12:15], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x48
	s_mov_b32 s37, 4
	s_wait_kmcnt 0x0
	s_lshl_b32 s4, s2, 4
	s_add_co_i32 s33, s13, -1
	s_delay_alu instid0(SALU_CYCLE_1)
	v_add_co_u32 v1, s3, s4, s33
	s_and_b32 vcc_lo, exec_lo, s3
	s_mov_b32 s3, -1
	v_readfirstlane_b32 s5, v1
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_cvt_f32_u32 s6, s4
	s_sub_co_i32 s7, 0, s4
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(NEXT) | instid1(TRANS32_DEP_1)
	v_rcp_iflag_f32_e32 v1, s6
	v_readfirstlane_b32 s6, v1
	s_mul_f32 s6, s6, 0x4f7ffffe
	s_wait_alu 0xfffe
	s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(SALU_CYCLE_2)
	s_cvt_u32_f32 s6, s6
	s_wait_alu 0xfffe
	s_mul_i32 s7, s7, s6
	s_wait_alu 0xfffe
	s_mul_hi_u32 s7, s6, s7
	s_wait_alu 0xfffe
	s_add_co_i32 s6, s6, s7
	s_wait_alu 0xfffe
	s_mul_hi_u32 s6, s5, s6
	s_wait_alu 0xfffe
	s_mul_i32 s7, s6, s4
	s_wait_alu 0xfffe
	s_sub_co_i32 s5, s5, s7
	s_add_co_i32 s7, s6, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s8, s5, s4
	s_cmp_ge_u32 s5, s4
	s_cselect_b32 s6, s7, s6
	s_cselect_b32 s5, s8, s5
	s_wait_alu 0xfffe
	s_add_co_i32 s7, s6, 1
	s_cmp_ge_u32 s5, s4
	s_wait_alu 0xfffe
	s_cselect_b32 s4, s7, s6
	s_wait_alu 0xfffe
	s_mul_i32 s4, s4, s2
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
	s_delay_alu instid0(SALU_CYCLE_2)
	s_mul_i32 s6, s6, s5
	s_wait_alu 0xfffe
	s_mul_hi_u32 s6, s5, s6
	s_wait_alu 0xfffe
	s_add_co_i32 s5, s5, s6
	s_wait_alu 0xfffe
	s_mul_hi_u32 s5, s7, s5
	s_wait_alu 0xfffe
	s_mul_i32 s6, s5, s4
	s_wait_alu 0xfffe
	s_sub_co_i32 s6, s7, s6
	s_add_co_i32 s7, s5, 1
	s_wait_alu 0xfffe
	s_sub_co_i32 s8, s6, s4
	s_cmp_ge_u32 s6, s4
	s_cselect_b32 s5, s7, s5
	s_cselect_b32 s6, s8, s6
	s_wait_alu 0xfffe
	s_add_co_i32 s7, s5, 1
	s_cmp_ge_u32 s6, s4
	s_wait_alu 0xfffe
	s_cselect_b32 s4, s7, s5
	s_wait_alu 0xfffe
	s_max_u32 s37, s4, 4
.LBB0_2:
	s_clause 0x2
	s_load_b64 s[20:21], s[0:1], 0x10
	s_load_b64 s[18:19], s[0:1], 0x20
	s_load_b256 s[4:11], s[0:1], 0x28
	v_bfe_u32 v67, v0, 10, 10
	s_cmp_gt_u32 s12, 0x2aaa
	s_mul_i32 s36, s37, s2
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[65:66], null, s37, ttmp9, v[67:68]
	v_cmp_gt_u32_e32 vcc_lo, s37, v67
	s_cbranch_scc0 .LBB0_40
; %bb.3:
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB0_39
; %bb.4:
	s_cvt_f32_u32 s0, s37
	s_sub_co_i32 s1, 0, s37
	s_add_co_i32 s2, s37, s33
	s_mov_b32 s23, 0
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
	s_mul_i32 s0, s0, s37
	s_wait_alu 0xfffe
	s_sub_co_i32 s0, s2, s0
	s_wait_alu 0xfffe
	s_sub_co_i32 s1, s0, s37
	s_cmp_ge_u32 s0, s37
	s_wait_alu 0xfffe
	s_cselect_b32 s0, s1, s0
	s_wait_alu 0xfffe
	s_sub_co_i32 s1, s0, s37
	s_cmp_ge_u32 s0, s37
	s_wait_alu 0xfffe
	s_cselect_b32 s0, s1, s0
	s_wait_alu 0xfffe
	s_sub_co_i32 s38, s2, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cmp_gt_u32_e32 vcc_lo, s38, v65
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_39
; %bb.5:
	v_dual_mov_b32 v70, v65 :: v_dual_and_b32 v1, 0x3ff, v0
	s_cmp_lg_u64 s[16:17], 0
	v_mbcnt_lo_u32_b32 v76, -1, 0
	s_cselect_b32 s0, -1, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v69, 0 :: v_dual_lshlrev_b32 v66, 3, v1
	v_cmp_eq_u32_e64 s1, 31, v1
	v_lshlrev_b32_e32 v1, 4, v1
	s_lshl_b32 s39, s37, 8
	s_add_co_i32 s40, s12, -8
	s_wait_kmcnt 0x0
	s_cmp_neq_f32 s9, 0
	v_lshl_add_u32 v72, v67, 8, v66
	v_lshl_add_u32 v73, v67, 9, v1
	s_mov_b32 s22, s19
	s_mov_b32 s30, s10
	s_mov_b32 s31, s23
	s_mov_b32 s28, s11
	v_add_nc_u32_e32 v74, 0x5000, v73
	v_add_nc_u32_e32 v75, 0xa000, v73
	s_mov_b32 s29, s23
	s_cselect_b32 s41, -1, 0
	s_lshl_b64 s[24:25], s[30:31], 2
	s_lshl_b64 s[26:27], s[22:23], 1
	s_lshl_b64 s[28:29], s[28:29], 1
	s_lshl_b32 s42, s37, 9
	s_lshl_b64 s[30:31], s[30:31], 1
	s_mov_b32 s43, 0
	s_branch .LBB0_8
.LBB0_6:                                ;   in Loop: Header=BB0_8 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v5, v6
	v_cndmask_b32_e64 v5, v5, v1, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_bfe_u32 v1, v5, 16, 1
	v_or_b32_e32 v6, 0x400000, v5
	v_add3_u32 v7, v1, v5, 0x7fff
	v_add_co_u32 v1, vcc_lo, v3, s28
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s29, v4, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v5, v5
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v7, v6, vcc_lo
	global_store_d16_hi_b16 v[1:2], v3, off
.LBB0_7:                                ;   in Loop: Header=BB0_8 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	v_add_nc_u32_e32 v70, s36, v70
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s38, v70
	s_or_b32 s43, vcc_lo, s43
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s43
	s_cbranch_execz .LBB0_39
.LBB0_8:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_16 Depth 2
                                        ;       Child Loop BB0_20 Depth 3
                                        ;       Child Loop BB0_22 Depth 3
                                        ;       Child Loop BB0_24 Depth 3
	v_cmp_gt_u32_e64 s2, s13, v70
	v_mov_b32_e32 v71, v69
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v77, v69
	s_wait_alu 0xfffe
	s_and_b32 s34, s0, s2
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s22, s34
	s_cbranch_execz .LBB0_10
; %bb.9:                                ;   in Loop: Header=BB0_8 Depth=1
	s_wait_dscnt 0x2
	v_lshlrev_b64_e32 v[1:2], 1, v[70:71]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s16, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s17, v2, vcc_lo
	global_load_u16 v77, v[1:2], off
.LBB0_10:                               ;   in Loop: Header=BB0_8 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	s_wait_dscnt 0x2
	v_mad_co_u64_u32 v[1:2], null, v70, s18, 0
	v_dual_mov_b32 v82, 0 :: v_dual_mov_b32 v81, 0
	v_mov_b32_e32 v80, 0
	s_mov_b32 s22, 0
	s_wait_alu 0xfffe
	s_mov_b32 s44, s22
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 1, v[1:2]
	v_add_co_u32 v78, vcc_lo, s14, v1
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v79, null, s15, v2, vcc_lo
	s_branch .LBB0_16
.LBB0_11:                               ;   in Loop: Header=BB0_16 Depth=2
	s_or_b32 exec_lo, exec_lo, s47
.LBB0_12:                               ;   in Loop: Header=BB0_16 Depth=2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s46
.LBB0_13:                               ;   in Loop: Header=BB0_16 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s45
.LBB0_14:                               ;   in Loop: Header=BB0_16 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s35
	s_wait_loadcnt_dscnt 0x301
	v_dot2_f32_bf16 v61, v61, v13, v82
	v_dot2_f32_bf16 v53, v53, v13, v81
	s_wait_dscnt 0x0
	v_dot2_f32_bf16 v13, v25, v13, v80
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v25, v62, v14, v61
	v_dot2_f32_bf16 v53, v54, v14, v53
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v13, v26, v14, v13
	v_dot2_f32_bf16 v14, v63, v15, v25
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v25, v55, v15, v53
	v_dot2_f32_bf16 v13, v27, v15, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v14, v64, v16, v14
	v_dot2_f32_bf16 v15, v56, v16, v25
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v13, v28, v16, v13
	s_wait_loadcnt 0x2
	v_dot2_f32_bf16 v14, v45, v9, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v15, v41, v9, v15
	v_dot2_f32_bf16 v9, v29, v9, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v13, v46, v10, v14
	v_dot2_f32_bf16 v14, v42, v10, v15
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v9, v30, v10, v9
	v_dot2_f32_bf16 v10, v47, v11, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v13, v43, v11, v14
	v_dot2_f32_bf16 v9, v31, v11, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v10, v48, v12, v10
	v_dot2_f32_bf16 v11, v44, v12, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v9, v32, v12, v9
	s_wait_loadcnt 0x1
	v_dot2_f32_bf16 v10, v17, v5, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v11, v33, v5, v11
	v_dot2_f32_bf16 v5, v49, v5, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v9, v18, v6, v10
	v_dot2_f32_bf16 v10, v34, v6, v11
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v5, v50, v6, v5
	v_dot2_f32_bf16 v6, v19, v7, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v9, v35, v7, v10
	v_dot2_f32_bf16 v5, v51, v7, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v6, v20, v8, v6
	v_dot2_f32_bf16 v7, v36, v8, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v5, v52, v8, v5
	s_wait_loadcnt 0x0
	v_dot2_f32_bf16 v6, v21, v1, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v7, v37, v1, v7
	v_dot2_f32_bf16 v1, v57, v1, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v5, v22, v2, v6
	v_dot2_f32_bf16 v6, v38, v2, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v1, v58, v2, v1
	v_dot2_f32_bf16 v2, v23, v3, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v5, v39, v3, v6
	v_dot2_f32_bf16 v1, v59, v3, v1
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v82, v24, v4, v2
	v_dot2_f32_bf16 v81, v40, v4, v5
	s_delay_alu instid0(VALU_DEP_3)
	v_dot2_f32_bf16 v80, v60, v4, v1
.LBB0_15:                               ;   in Loop: Header=BB0_16 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s34
	s_addk_co_i32 s44, 0x400
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s44, s12
	s_cbranch_scc1 .LBB0_31
.LBB0_16:                               ;   Parent Loop BB0_8 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_20 Depth 3
                                        ;       Child Loop BB0_22 Depth 3
                                        ;       Child Loop BB0_24 Depth 3
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s44, 0
	s_cselect_b32 s35, -1, 0
	s_add_co_i32 s34, s22, 0x2800
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s44, s34
	s_cselect_b32 s45, -1, 0
	s_wait_alu 0xfffe
	s_or_b32 s45, s35, s45
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s45
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB0_18
; %bb.17:                               ;   in Loop: Header=BB0_16 Depth=2
	s_and_saveexec_b32 s34, s2
	s_cbranch_execz .LBB0_15
	s_branch .LBB0_26
.LBB0_18:                               ;   in Loop: Header=BB0_16 Depth=2
	s_and_b32 s35, s35, exec_lo
	s_cselect_b32 s22, s22, s34
	s_mov_b32 s46, exec_lo
	s_wait_alu 0xfffe
	s_sub_co_i32 s34, s12, s22
	s_wait_loadcnt 0x0
	s_wait_alu 0xfffe
	s_min_u32 s45, s34, 0x2800
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu 0xfffe
	v_cmpx_gt_u32_e64 s45, v72
	s_cbranch_execz .LBB0_25
; %bb.19:                               ;   in Loop: Header=BB0_16 Depth=2
	v_dual_mov_b32 v1, v73 :: v_dual_mov_b32 v68, v72
	s_lshl_b64 s[34:35], s[22:23], 1
	s_mov_b32 s47, 0
	s_wait_alu 0xfffe
	s_add_nc_u64 s[34:35], s[20:21], s[34:35]
.LBB0_20:                               ;   Parent Loop BB0_8 Depth=1
                                        ;     Parent Loop BB0_16 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_lshlrev_b64_e32 v[2:3], 1, v[68:69]
	v_add_nc_u32_e32 v68, s39, v68
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v2, vcc_lo, s34, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s35, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s45, v68
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s47, vcc_lo, s47
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s42, v1
	s_and_not1_b32 exec_lo, exec_lo, s47
	s_cbranch_execnz .LBB0_20
; %bb.21:                               ;   in Loop: Header=BB0_16 Depth=2
	s_or_b32 exec_lo, exec_lo, s47
	v_dual_mov_b32 v1, v74 :: v_dual_mov_b32 v68, v72
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]
	s_mov_b32 s47, 0
.LBB0_22:                               ;   Parent Loop BB0_8 Depth=1
                                        ;     Parent Loop BB0_16 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 1, v[68:69]
	v_add_nc_u32_e32 v68, s39, v68
	s_wait_alu 0xfffe
	v_add_co_u32 v2, vcc_lo, s34, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, s35, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s45, v68
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s47, vcc_lo, s47
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s42, v1
	s_and_not1_b32 exec_lo, exec_lo, s47
	s_cbranch_execnz .LBB0_22
; %bb.23:                               ;   in Loop: Header=BB0_16 Depth=2
	s_or_b32 exec_lo, exec_lo, s47
	v_dual_mov_b32 v1, v75 :: v_dual_mov_b32 v68, v72
	s_add_nc_u64 s[34:35], s[34:35], s[26:27]
	s_mov_b32 s47, 0
.LBB0_24:                               ;   Parent Loop BB0_8 Depth=1
                                        ;     Parent Loop BB0_16 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 1, v[68:69]
	v_add_nc_u32_e32 v68, s39, v68
	s_wait_alu 0xfffe
	v_add_co_u32 v2, vcc_lo, s34, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, s35, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s45, v68
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s47, vcc_lo, s47
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s42, v1
	s_and_not1_b32 exec_lo, exec_lo, s47
	s_cbranch_execnz .LBB0_24
.LBB0_25:                               ;   in Loop: Header=BB0_16 Depth=2
	s_or_b32 exec_lo, exec_lo, s46
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s34, s2
	s_cbranch_execz .LBB0_15
.LBB0_26:                               ;   in Loop: Header=BB0_16 Depth=2
	v_dual_mov_b32 v60, 0 :: v_dual_add_nc_u32 v85, s44, v66
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v51, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v49, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v68, s40, v85
	v_dual_mov_b32 v59, 0 :: v_dual_add_nc_u32 v86, 0x100, v85
	v_dual_mov_b32 v57, 0 :: v_dual_add_nc_u32 v84, 0x200, v85
	v_lshlrev_b64_e32 v[1:2], 1, v[68:69]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v68, s40, v86
	v_dual_mov_b32 v58, 0 :: v_dual_add_nc_u32 v83, 0x300, v85
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v31, 0
	v_lshlrev_b64_e32 v[3:4], 1, v[68:69]
	v_min_u32_e32 v68, s40, v84
	v_add_co_u32 v1, vcc_lo, v78, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, v79, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 1, v[68:69]
	v_min_u32_e32 v68, s40, v83
	v_add_co_u32 v3, vcc_lo, v78, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, v79, v4, vcc_lo
	s_wait_dscnt 0x0
	v_lshlrev_b64_e32 v[7:8], 1, v[68:69]
	v_add_co_u32 v5, vcc_lo, v78, v5
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, v79, v6, vcc_lo
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v29, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v17, vcc_lo, v78, v7
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v18, null, v79, v8, vcc_lo
	s_clause 0x3
	global_load_b128 v[13:16], v[1:2], off
	global_load_b128 v[9:12], v[3:4], off
	global_load_b128 v[5:8], v[5:6], off
	global_load_b128 v[1:4], v[17:18], off
	v_cmp_gt_u32_e32 vcc_lo, s12, v85
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v39, 0
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v64, 0 :: v_dual_mov_b32 v63, 0
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v61, 0
	s_and_saveexec_b32 s35, vcc_lo
	s_cbranch_execz .LBB0_14
; %bb.27:                               ;   in Loop: Header=BB0_16 Depth=2
	v_subrev_nc_u32_e32 v17, s22, v85
	v_mov_b32_e32 v45, 0
	v_cmp_gt_u32_e32 vcc_lo, s12, v86
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v47, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b32_e32 v68, 1, v17
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v19, 0
	ds_load_b128 v[53:56], v68 offset:20480
	ds_load_b128 v[61:64], v68
	ds_load_b128 v[25:28], v68 offset:40960
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v39, 0
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v31, 0
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v51, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v57, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v59, 0
	v_mov_b32_e32 v60, 0
	s_and_saveexec_b32 s45, vcc_lo
	s_cbranch_execz .LBB0_13
; %bb.28:                               ;   in Loop: Header=BB0_16 Depth=2
	ds_load_b128 v[41:44], v68 offset:20992
	ds_load_b128 v[45:48], v68 offset:512
	ds_load_b128 v[29:32], v68 offset:41472
	v_cmp_gt_u32_e32 vcc_lo, s12, v84
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v20, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v60, 0
	s_and_saveexec_b32 s46, vcc_lo
	s_cbranch_execz .LBB0_12
; %bb.29:                               ;   in Loop: Header=BB0_16 Depth=2
	ds_load_b128 v[33:36], v68 offset:21504
	ds_load_b128 v[17:20], v68 offset:1024
	ds_load_b128 v[49:52], v68 offset:41984
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v60, 0
	s_mov_b32 s47, exec_lo
	v_cmpx_gt_u32_e64 s12, v83
	s_cbranch_execz .LBB0_11
; %bb.30:                               ;   in Loop: Header=BB0_16 Depth=2
	ds_load_b128 v[37:40], v68 offset:22016
	ds_load_b128 v[21:24], v68 offset:1536
	ds_load_b128 v[57:60], v68 offset:42496
	s_branch .LBB0_11
.LBB0_31:                               ;   in Loop: Header=BB0_8 Depth=1
	s_and_saveexec_b32 s22, s2
	s_cbranch_execz .LBB0_7
; %bb.32:                               ;   in Loop: Header=BB0_8 Depth=1
	v_add_f32_dpp v1, v82, v82 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v81, v81 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v80, v80 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_xor_b32_e32 v4, 16, v76
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_4)
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v4, v76, v4, vcc_lo
	v_add_f32_dpp v1, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_4)
	v_lshlrev_b32_e32 v4, 2, v4
	ds_bpermute_b32 v2, v4, v1
	s_wait_dscnt 0x2
	ds_bpermute_b32 v9, v4, v8
	s_wait_dscnt 0x2
	ds_bpermute_b32 v7, v4, v5
	s_and_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_7
; %bb.33:                               ;   in Loop: Header=BB0_8 Depth=1
	v_lshlrev_b64_e32 v[3:4], 1, v[70:71]
	s_wait_dscnt 0x2
	v_add_f32_e32 v6, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v10, s8, v6
	v_add_co_u32 v1, vcc_lo, s4, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s5, v4, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s41
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_35
; %bb.34:                               ;   in Loop: Header=BB0_8 Depth=1
	global_load_u16 v6, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v6, 16, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v10, s9, v6
.LBB0_35:                               ;   in Loop: Header=BB0_8 Depth=1
	v_lshlrev_b32_e32 v6, 16, v77
	v_add_co_u32 v3, vcc_lo, s6, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s7, v4, vcc_lo
	s_wait_dscnt 0x1
	v_dual_add_f32 v11, v10, v6 :: v_dual_add_f32 v8, v8, v9
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v10, v10, v11, s0
	v_mul_f32_e32 v8, s8, v8
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v11, v10, 16, 1
	v_or_b32_e32 v9, 0x400000, v10
	v_cmp_u_f32_e32 vcc_lo, v10, v10
	v_add3_u32 v11, v11, v10, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v9, v11, v9, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s41
	global_store_d16_hi_b16 v[3:4], v9, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_37
; %bb.36:                               ;   in Loop: Header=BB0_8 Depth=1
	v_add_co_u32 v9, vcc_lo, v1, s30
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v10, null, s31, v2, vcc_lo
	global_load_u16 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v9, 16, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, s9, v9
.LBB0_37:                               ;   in Loop: Header=BB0_8 Depth=1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v8, v6
	v_add_co_u32 v3, vcc_lo, v3, s28
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v7
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s29, v4, vcc_lo
	v_cndmask_b32_e64 v8, v8, v9, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v5, s8, v5
	v_bfe_u32 v9, v8, 16, 1
	v_or_b32_e32 v7, 0x400000, v8
	v_cmp_u_f32_e32 vcc_lo, v8, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add3_u32 v9, v9, v8, 0x7fff
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v7, v9, v7, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s41
	global_store_d16_hi_b16 v[3:4], v7, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_6
; %bb.38:                               ;   in Loop: Header=BB0_8 Depth=1
	v_add_co_u32 v1, vcc_lo, v1, s24
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s25, v2, vcc_lo
	global_load_u16 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v1, 16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v5, s9, v1
	s_branch .LBB0_6
.LBB0_39:
	s_or_b32 exec_lo, exec_lo, s3
	s_mov_b32 s3, 0
.LBB0_40:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_74
; %bb.41:
	s_wait_dscnt 0x2
	v_and_b32_e32 v2, 0x3ff, v0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v68, 3, v2
	v_lshl_add_u32 v3, v67, 8, v68
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s12, v3
	s_cbranch_execz .LBB0_48
; %bb.42:
	v_lshlrev_b32_e32 v0, 9, v67
	v_dual_mov_b32 v8, v3 :: v_dual_lshlrev_b32 v1, 4, v2
	s_mov_b32 s2, 0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v6, v0, v1
	s_wait_dscnt 0x0
	v_mov_b32_e32 v7, v6
	v_add_co_u32 v4, s0, v0, v1
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v5, null, 0, 0, s0
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s20, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s21, v5, vcc_lo
.LBB0_43:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[9:12], v[0:1], off
	v_add_nc_u32_e32 v8, 0x1000, v8
	v_add_co_u32 v0, vcc_lo, 0x2000, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s0, s12, v8
	s_or_b32 s2, s0, s2
	s_wait_loadcnt 0x0
	ds_store_b128 v7, v[9:12]
	v_add_nc_u32_e32 v7, 0x2000, v7
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_43
; %bb.44:
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s3, 0
	s_mov_b32 s2, s19
	v_lshl_add_u32 v7, s12, 1, v6
	s_lshl_b64 s[22:23], s[2:3], 1
	v_mov_b32_e32 v8, v3
	s_wait_alu 0xfffe
	s_add_nc_u64 s[22:23], s[20:21], s[22:23]
	s_mov_b32 s19, s3
	s_wait_alu 0xfffe
	v_add_co_u32 v0, vcc_lo, s22, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s23, v5, vcc_lo
.LBB0_45:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[9:12], v[0:1], off
	v_add_nc_u32_e32 v8, 0x1000, v8
	v_add_co_u32 v0, vcc_lo, 0x2000, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s0, s12, v8
	s_or_b32 s19, s0, s19
	s_wait_loadcnt 0x0
	ds_store_b128 v7, v[9:12]
	v_add_nc_u32_e32 v7, 0x2000, v7
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s19
	s_cbranch_execnz .LBB0_45
; %bb.46:
	s_or_b32 exec_lo, exec_lo, s19
	s_lshl_b64 s[2:3], s[2:3], 2
	v_lshl_add_u32 v6, s12, 2, v6
	s_add_nc_u64 s[2:3], s[20:21], s[2:3]
	s_delay_alu instid0(SALU_CYCLE_1)
	v_add_co_u32 v0, vcc_lo, s2, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s3, v5, vcc_lo
	s_mov_b32 s2, 0
.LBB0_47:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[7:10], v[0:1], off
	v_add_nc_u32_e32 v3, 0x1000, v3
	v_add_co_u32 v0, vcc_lo, 0x2000, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s0, s12, v3
	s_wait_alu 0xfffe
	s_or_b32 s2, s0, s2
	s_wait_loadcnt 0x0
	ds_store_b128 v6, v[7:10]
	v_add_nc_u32_e32 v6, 0x2000, v6
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_47
.LBB0_48:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_u32_e64 s37, v67
	s_cbranch_execz .LBB0_74
; %bb.49:
	v_cmp_gt_u32_e32 vcc_lo, s13, v65
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_74
; %bb.50:
	s_cmp_eq_u64 s[16:17], 0
	v_mbcnt_lo_u32_b32 v67, -1, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[16:17], 0
	s_mov_b32 s3, 0
	s_wait_kmcnt 0x0
	s_cselect_b32 s19, -1, 0
	s_cmp_lg_u32 s12, 0
	v_cmp_eq_u32_e64 s1, 31, v2
	s_cselect_b32 s24, -1, 0
	s_add_co_i32 s25, s12, -8
	s_cmp_neq_f32 s9, 0
	v_dual_mov_b32 v64, 0 :: v_dual_lshlrev_b32 v69, 4, v2
	v_xor_b32_e32 v70, 16, v67
	s_mov_b32 s2, s10
	s_mov_b32 s20, s11
	s_wait_alu 0xfffe
	s_mov_b32 s21, s3
	s_cselect_b32 s26, -1, 0
	s_lshl_b64 s[10:11], s[2:3], 2
	s_lshl_b32 s27, s12, 1
	s_wait_alu 0xfffe
	s_lshl_b64 s[20:21], s[20:21], 1
	s_lshl_b32 s28, s12, 2
	s_lshl_b64 s[22:23], s[2:3], 1
	s_branch .LBB0_53
.LBB0_51:                               ;   in Loop: Header=BB0_53 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v4, v5
	v_cndmask_b32_e64 v4, v0, v4, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_bfe_u32 v0, v4, 16, 1
	v_or_b32_e32 v5, 0x400000, v4
	v_add3_u32 v6, v0, v4, 0x7fff
	v_add_co_u32 v0, vcc_lo, v2, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s21, v3, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v4, v4
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v2, v6, v5, vcc_lo
	global_store_d16_hi_b16 v[0:1], v2, off
.LBB0_52:                               ;   in Loop: Header=BB0_53 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s2
	v_add_nc_u32_e32 v65, s36, v65
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s13, v65
	s_or_b32 s3, vcc_lo, s3
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s3
	s_cbranch_execz .LBB0_74
.LBB0_53:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_61 Depth 2
	v_mov_b32_e32 v71, v64
	s_and_not1_b32 vcc_lo, exec_lo, s19
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_55
; %bb.54:                               ;   in Loop: Header=BB0_53 Depth=1
	v_mov_b32_e32 v66, v64
	s_wait_dscnt 0x2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 1, v[65:66]
	v_add_co_u32 v0, vcc_lo, s16, v0
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, s17, v1, vcc_lo
	global_load_u16 v71, v[0:1], off
.LBB0_55:                               ;   in Loop: Header=BB0_53 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_66
; %bb.56:                               ;   in Loop: Header=BB0_53 Depth=1
	v_min_u32_e32 v0, s33, v65
	v_dual_mov_b32 v76, 0 :: v_dual_mov_b32 v75, 0
	v_mov_b32_e32 v66, v69
	v_mov_b32_e32 v74, 0
	s_wait_dscnt 0x2
	v_mad_co_u64_u32 v[0:1], null, v0, s18, 0
	s_mov_b32 s2, 0
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v72, vcc_lo, s14, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v73, null, s15, v1, vcc_lo
	s_branch .LBB0_61
.LBB0_57:                               ;   in Loop: Header=BB0_61 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s34
.LBB0_58:                               ;   in Loop: Header=BB0_61 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s31
.LBB0_59:                               ;   in Loop: Header=BB0_61 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s30
.LBB0_60:                               ;   in Loop: Header=BB0_61 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s29
	s_wait_loadcnt_dscnt 0x302
	v_dot2_f32_bf16 v60, v60, v12, v76
	s_wait_dscnt 0x1
	v_dot2_f32_bf16 v52, v52, v12, v75
	s_wait_dscnt 0x0
	v_dot2_f32_bf16 v12, v20, v12, v74
	v_add_nc_u32_e32 v66, 0x800, v66
	s_addk_co_i32 s2, 0x400
	v_dot2_f32_bf16 v20, v61, v13, v60
	v_dot2_f32_bf16 v52, v53, v13, v52
	v_dot2_f32_bf16 v12, v21, v13, v12
	s_wait_alu 0xfffe
	s_cmp_lt_u32 s2, s12
	v_dot2_f32_bf16 v13, v62, v14, v20
	v_dot2_f32_bf16 v20, v54, v14, v52
	v_dot2_f32_bf16 v12, v22, v14, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v13, v63, v15, v13
	v_dot2_f32_bf16 v14, v55, v15, v20
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v12, v23, v15, v12
	s_wait_loadcnt 0x2
	v_dot2_f32_bf16 v13, v48, v8, v13
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v14, v40, v8, v14
	v_dot2_f32_bf16 v8, v32, v8, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v12, v49, v9, v13
	v_dot2_f32_bf16 v13, v41, v9, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v8, v33, v9, v8
	v_dot2_f32_bf16 v9, v50, v10, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v12, v42, v10, v13
	v_dot2_f32_bf16 v8, v34, v10, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v9, v51, v11, v9
	v_dot2_f32_bf16 v10, v43, v11, v12
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v8, v35, v11, v8
	s_wait_loadcnt 0x1
	v_dot2_f32_bf16 v9, v24, v4, v9
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v10, v36, v4, v10
	v_dot2_f32_bf16 v4, v44, v4, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v8, v25, v5, v9
	v_dot2_f32_bf16 v9, v37, v5, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v4, v45, v5, v4
	v_dot2_f32_bf16 v5, v26, v6, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v8, v38, v6, v9
	v_dot2_f32_bf16 v4, v46, v6, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v5, v27, v7, v5
	v_dot2_f32_bf16 v6, v39, v7, v8
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v4, v47, v7, v4
	s_wait_loadcnt 0x0
	v_dot2_f32_bf16 v5, v16, v0, v5
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v6, v28, v0, v6
	v_dot2_f32_bf16 v0, v56, v0, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v4, v17, v1, v5
	v_dot2_f32_bf16 v5, v29, v1, v6
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v0, v57, v1, v0
	v_dot2_f32_bf16 v1, v18, v2, v4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v4, v30, v2, v5
	v_dot2_f32_bf16 v0, v58, v2, v0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_dot2_f32_bf16 v76, v19, v3, v1
	v_dot2_f32_bf16 v75, v31, v3, v4
	s_delay_alu instid0(VALU_DEP_3)
	v_dot2_f32_bf16 v74, v59, v3, v0
	s_cbranch_scc0 .LBB0_67
.LBB0_61:                               ;   Parent Loop BB0_53 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_dual_mov_b32 v59, 0 :: v_dual_add_nc_u32 v18, s2, v68
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v44, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v63, s25, v18
	v_dual_mov_b32 v58, 0 :: v_dual_add_nc_u32 v81, 0x100, v18
	v_dual_mov_b32 v57, 0 :: v_dual_add_nc_u32 v78, 0x200, v18
	v_lshlrev_b64_e32 v[0:1], 1, v[63:64]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v63, s25, v81
	v_dual_mov_b32 v56, 0 :: v_dual_add_nc_u32 v77, 0x300, v18
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v34, 0
	v_lshlrev_b64_e32 v[2:3], 1, v[63:64]
	v_min_u32_e32 v63, s25, v78
	v_add_co_u32 v0, vcc_lo, v72, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v73, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[4:5], 1, v[63:64]
	v_min_u32_e32 v63, s25, v77
	v_add_co_u32 v2, vcc_lo, v72, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, v73, v3, vcc_lo
	s_wait_dscnt 0x0
	v_lshlrev_b64_e32 v[6:7], 1, v[63:64]
	v_add_co_u32 v4, vcc_lo, v72, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v73, v5, vcc_lo
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v32, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v16, vcc_lo, v72, v6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v17, null, v73, v7, vcc_lo
	s_clause 0x3
	global_load_b128 v[12:15], v[0:1], off
	global_load_b128 v[8:11], v[2:3], off
	global_load_b128 v[4:7], v[4:5], off
	global_load_b128 v[0:3], v[16:17], off
	v_cmp_gt_u32_e32 vcc_lo, s12, v18
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v20, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v16, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v60, 0
	s_and_saveexec_b32 s29, vcc_lo
	s_cbranch_execz .LBB0_60
; %bb.62:                               ;   in Loop: Header=BB0_61 Depth=2
	v_dual_mov_b32 v49, 0 :: v_dual_add_nc_u32 v80, s27, v66
	v_dual_mov_b32 v48, 0 :: v_dual_add_nc_u32 v79, s28, v66
	ds_load_b128 v[60:63], v66
	ds_load_b128 v[52:55], v80
	ds_load_b128 v[20:23], v79
	v_cmp_gt_u32_e32 vcc_lo, s12, v81
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v51, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v39, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v31, 0
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v57, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v59, 0
	s_and_saveexec_b32 s30, vcc_lo
	s_cbranch_execz .LBB0_59
; %bb.63:                               ;   in Loop: Header=BB0_61 Depth=2
	ds_load_b128 v[48:51], v66 offset:512
	ds_load_b128 v[40:43], v80 offset:512
	ds_load_b128 v[32:35], v79 offset:512
	v_cmp_gt_u32_e32 vcc_lo, s12, v78
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v39, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v31, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v57, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v59, 0
	s_and_saveexec_b32 s31, vcc_lo
	s_cbranch_execz .LBB0_58
; %bb.64:                               ;   in Loop: Header=BB0_61 Depth=2
	ds_load_b128 v[24:27], v66 offset:1024
	ds_load_b128 v[36:39], v80 offset:1024
	ds_load_b128 v[44:47], v79 offset:1024
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v31, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v57, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v59, 0
	s_mov_b32 s34, exec_lo
	v_cmpx_gt_u32_e64 s12, v77
	s_cbranch_execz .LBB0_57
; %bb.65:                               ;   in Loop: Header=BB0_61 Depth=2
	ds_load_b128 v[16:19], v66 offset:1536
	ds_load_b128 v[28:31], v80 offset:1536
	ds_load_b128 v[56:59], v79 offset:1536
	s_branch .LBB0_57
.LBB0_66:                               ;   in Loop: Header=BB0_53 Depth=1
	v_dual_mov_b32 v74, 0 :: v_dual_mov_b32 v75, 0
	v_mov_b32_e32 v76, 0
.LBB0_67:                               ;   in Loop: Header=BB0_53 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_f32_dpp v0, v76, v76 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x2
	v_add_f32_dpp v1, v75, v75 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v74, v74 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v70
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v67, v70, vcc_lo
	v_add_f32_dpp v0, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshlrev_b32_e32 v3, 2, v3
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add_f32_dpp v7, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v1, v3, v0
	s_wait_dscnt 0x2
	ds_bpermute_b32 v8, v3, v7
	s_wait_dscnt 0x2
	ds_bpermute_b32 v6, v3, v4
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_52
; %bb.68:                               ;   in Loop: Header=BB0_53 Depth=1
	s_wait_dscnt 0x2
	v_add_f32_e32 v5, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v66, v64 :: v_dual_mul_f32 v9, s8, v5
	v_lshlrev_b64_e32 v[2:3], 1, v[65:66]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s4, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s5, v3, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_70
; %bb.69:                               ;   in Loop: Header=BB0_53 Depth=1
	global_load_u16 v5, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v5, 16, v5
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, s9, v5
.LBB0_70:                               ;   in Loop: Header=BB0_53 Depth=1
	v_lshlrev_b32_e32 v5, 16, v71
	v_add_co_u32 v2, vcc_lo, s6, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	s_wait_dscnt 0x1
	v_dual_add_f32 v10, v9, v5 :: v_dual_add_f32 v7, v7, v8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v9, v10, v9, s0
	v_mul_f32_e32 v7, s8, v7
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v10, v9, 16, 1
	v_or_b32_e32 v8, 0x400000, v9
	v_cmp_u_f32_e32 vcc_lo, v9, v9
	v_add3_u32 v10, v10, v9, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v8, v10, v8, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s26
	global_store_d16_hi_b16 v[2:3], v8, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_72
; %bb.71:                               ;   in Loop: Header=BB0_53 Depth=1
	v_add_co_u32 v8, vcc_lo, v0, s22
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, s23, v1, vcc_lo
	global_load_u16 v8, v[8:9], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v8, 16, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, s9, v8
.LBB0_72:                               ;   in Loop: Header=BB0_53 Depth=1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v7, v5
	v_add_co_u32 v2, vcc_lo, v2, s20
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	v_cndmask_b32_e64 v7, v8, v7, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v4, s8, v4
	v_bfe_u32 v8, v7, 16, 1
	v_or_b32_e32 v6, 0x400000, v7
	v_cmp_u_f32_e32 vcc_lo, v7, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add3_u32 v8, v8, v7, 0x7fff
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v6, v8, v6, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s26
	global_store_d16_hi_b16 v[2:3], v6, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_51
; %bb.73:                               ;   in Loop: Header=BB0_53 Depth=1
	v_add_co_u32 v0, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	global_load_u16 v0, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v0, 16, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v4, s9, v0
	s_branch .LBB0_51
.LBB0_74:
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201
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
		.amdhsa_next_free_sgpr 48
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 52
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
	.size	wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201, .Lfunc_end0-wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201
                                        ; -- End function
	.set wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201.num_vgpr, 87
	.set wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201.num_agpr, 0
	.set wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201.numbered_sgpr, 48
	.set wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201.num_named_barrier, 0
	.set wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201.private_seg_size, 0
	.set wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201.uses_vcc, 1
	.set wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201.uses_flat_scratch, 0
	.set wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201.has_dyn_sized_stack, 0
	.set wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201.has_recursion, 0
	.set wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 6560
; TotalNumSgprs: 50
; NumVgprs: 87
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 21
; NumSGPRsForWavesPerEU: 50
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
	.type	__hip_cuid_f031d78c797bcccc,@object ; @__hip_cuid_f031d78c797bcccc
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_f031d78c797bcccc
__hip_cuid_f031d78c797bcccc:
	.byte	0                               ; 0x0
	.size	__hip_cuid_f031d78c797bcccc, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_f031d78c797bcccc
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
    .group_segment_fixed_size: 65536
    .kernarg_segment_align: 8
    .kernarg_segment_size: 80
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     50
    .sgpr_spill_count: 0
    .symbol:         wvSplitK_vllm_bf16_tn_n3_y1u4_gfx1201.kd
    .vgpr_count:     87
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
