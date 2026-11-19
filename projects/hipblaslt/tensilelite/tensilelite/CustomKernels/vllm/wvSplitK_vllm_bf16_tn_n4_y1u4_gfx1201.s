	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201 ; -- Begin function wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201
	.globl	wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201
	.p2align	8
	.type	wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201,@function
wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201:  ; @wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201
; %bb.0:
	s_clause 0x2
	s_load_b32 s2, s[0:1], 0x18
	s_load_b128 s[12:15], s[0:1], 0x0
	s_load_b64 s[16:17], s[0:1], 0x48
	s_mov_b32 s39, 4
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
	s_max_u32 s39, s4, 4
.LBB0_2:
	s_clause 0x2
	s_load_b64 s[20:21], s[0:1], 0x10
	s_load_b64 s[18:19], s[0:1], 0x20
	s_load_b256 s[4:11], s[0:1], 0x28
	v_bfe_u32 v83, v0, 10, 10
	s_cmp_gt_u32 s12, 0x2000
	s_mul_i32 s38, s39, s2
	s_delay_alu instid0(VALU_DEP_1)
	v_mad_co_u64_u32 v[81:82], null, s39, ttmp9, v[83:84]
	v_cmp_gt_u32_e32 vcc_lo, s39, v83
	s_cbranch_scc0 .LBB0_44
; %bb.3:
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB0_43
; %bb.4:
	s_cvt_f32_u32 s0, s39
	s_sub_co_i32 s1, 0, s39
	s_add_co_i32 s2, s39, s33
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
	s_mul_i32 s0, s0, s39
	s_wait_alu 0xfffe
	s_sub_co_i32 s0, s2, s0
	s_wait_alu 0xfffe
	s_sub_co_i32 s1, s0, s39
	s_cmp_ge_u32 s0, s39
	s_wait_alu 0xfffe
	s_cselect_b32 s0, s1, s0
	s_wait_alu 0xfffe
	s_sub_co_i32 s1, s0, s39
	s_cmp_ge_u32 s0, s39
	s_wait_alu 0xfffe
	s_cselect_b32 s0, s1, s0
	s_wait_alu 0xfffe
	s_sub_co_i32 s40, s2, s0
	s_delay_alu instid0(SALU_CYCLE_1)
	v_cmp_gt_u32_e32 vcc_lo, s40, v81
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_43
; %bb.5:
	v_dual_mov_b32 v86, v81 :: v_dual_and_b32 v1, 0x3ff, v0
	s_cmp_lg_u64 s[16:17], 0
	v_mbcnt_lo_u32_b32 v93, -1, 0
	s_cselect_b32 s0, -1, 0
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v85, 0 :: v_dual_lshlrev_b32 v82, 3, v1
	v_cmp_eq_u32_e64 s1, 31, v1
	v_lshlrev_b32_e32 v1, 4, v1
	s_lshl_b32 s41, s39, 8
	s_add_co_i32 s42, s12, -8
	s_wait_kmcnt 0x0
	s_cmp_neq_f32 s9, 0
	v_lshl_add_u32 v88, v83, 8, v82
	v_lshl_add_u32 v89, v83, 9, v1
	s_mov_b32 s22, s19
	s_mov_b32 s34, s10
	s_mov_b32 s35, s23
	s_mov_b32 s30, s11
	v_add_nc_u32_e32 v90, 0x4000, v89
	v_add_nc_u32_e32 v91, 0x8000, v89
	v_add_nc_u32_e32 v92, 0xc000, v89
	s_mov_b32 s31, s23
	s_cselect_b32 s43, -1, 0
	s_lshl_b64 s[24:25], s[34:35], 2
	s_mul_u64 s[26:27], s[34:35], 6
	s_lshl_b64 s[28:29], s[22:23], 1
	s_lshl_b64 s[30:31], s[30:31], 1
	s_lshl_b32 s44, s39, 9
	s_lshl_b64 s[34:35], s[34:35], 1
	s_mov_b32 s45, 0
	s_branch .LBB0_8
.LBB0_6:                                ;   in Loop: Header=BB0_8 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v5, v6
	v_cndmask_b32_e64 v5, v5, v1, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_bfe_u32 v1, v5, 16, 1
	v_or_b32_e32 v6, 0x400000, v5
	v_add3_u32 v7, v1, v5, 0x7fff
	v_add_co_u32 v1, vcc_lo, v3, s30
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s31, v4, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v5, v5
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v7, v6, vcc_lo
	global_store_d16_hi_b16 v[1:2], v3, off
.LBB0_7:                                ;   in Loop: Header=BB0_8 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	v_add_nc_u32_e32 v86, s38, v86
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s40, v86
	s_or_b32 s45, vcc_lo, s45
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s45
	s_cbranch_execz .LBB0_43
.LBB0_8:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_16 Depth 2
                                        ;       Child Loop BB0_20 Depth 3
                                        ;       Child Loop BB0_22 Depth 3
                                        ;       Child Loop BB0_24 Depth 3
                                        ;       Child Loop BB0_26 Depth 3
	v_cmp_gt_u32_e64 s2, s13, v86
	s_wait_loadcnt 0x0
	v_dual_mov_b32 v87, v85 :: v_dual_mov_b32 v94, v85
	s_wait_alu 0xfffe
	s_and_b32 s36, s0, s2
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s22, s36
	s_cbranch_execz .LBB0_10
; %bb.9:                                ;   in Loop: Header=BB0_8 Depth=1
	s_wait_dscnt 0x3
	v_lshlrev_b64_e32 v[1:2], 1, v[86:87]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s16, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s17, v2, vcc_lo
	global_load_u16 v94, v[1:2], off
.LBB0_10:                               ;   in Loop: Header=BB0_8 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	s_wait_dscnt 0x3
	v_mad_co_u64_u32 v[1:2], null, v86, s18, 0
	v_dual_mov_b32 v100, 0 :: v_dual_mov_b32 v99, 0
	v_dual_mov_b32 v98, 0 :: v_dual_mov_b32 v97, 0
	s_mov_b32 s22, 0
	s_wait_alu 0xfffe
	s_mov_b32 s46, s22
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 1, v[1:2]
	v_add_co_u32 v95, vcc_lo, s14, v1
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v96, null, s15, v2, vcc_lo
	s_branch .LBB0_16
.LBB0_11:                               ;   in Loop: Header=BB0_16 Depth=2
	s_or_b32 exec_lo, exec_lo, s49
.LBB0_12:                               ;   in Loop: Header=BB0_16 Depth=2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s48
.LBB0_13:                               ;   in Loop: Header=BB0_16 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s47
.LBB0_14:                               ;   in Loop: Header=BB0_16 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s37
	s_wait_loadcnt_dscnt 0x303
	v_dot2_f32_bf16 v77, v77, v13, v100
	s_wait_dscnt 0x2
	v_dot2_f32_bf16 v73, v73, v13, v99
	s_wait_dscnt 0x1
	v_dot2_f32_bf16 v61, v61, v13, v98
	s_wait_dscnt 0x0
	v_dot2_f32_bf16 v13, v21, v13, v97
	v_dot2_f32_bf16 v21, v78, v14, v77
	v_dot2_f32_bf16 v73, v74, v14, v73
	v_dot2_f32_bf16 v61, v62, v14, v61
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v13, v22, v14, v13
	v_dot2_f32_bf16 v14, v79, v15, v21
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v21, v75, v15, v73
	v_dot2_f32_bf16 v22, v63, v15, v61
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v13, v23, v15, v13
	v_dot2_f32_bf16 v14, v80, v16, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v15, v76, v16, v21
	v_dot2_f32_bf16 v21, v64, v16, v22
	s_delay_alu instid0(VALU_DEP_4)
	v_dot2_f32_bf16 v13, v24, v16, v13
	s_wait_loadcnt 0x2
	v_dot2_f32_bf16 v14, v57, v9, v14
	v_dot2_f32_bf16 v15, v65, v9, v15
	v_dot2_f32_bf16 v16, v45, v9, v21
	v_dot2_f32_bf16 v9, v37, v9, v13
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v13, v58, v10, v14
	v_dot2_f32_bf16 v14, v66, v10, v15
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v15, v46, v10, v16
	v_dot2_f32_bf16 v9, v38, v10, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v10, v59, v11, v13
	v_dot2_f32_bf16 v13, v67, v11, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v14, v47, v11, v15
	v_dot2_f32_bf16 v9, v39, v11, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v10, v60, v12, v10
	v_dot2_f32_bf16 v11, v68, v12, v13
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v13, v48, v12, v14
	v_dot2_f32_bf16 v9, v40, v12, v9
	s_wait_loadcnt 0x1
	v_dot2_f32_bf16 v10, v29, v5, v10
	v_dot2_f32_bf16 v11, v33, v5, v11
	v_dot2_f32_bf16 v12, v41, v5, v13
	v_dot2_f32_bf16 v5, v53, v5, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v9, v30, v6, v10
	v_dot2_f32_bf16 v10, v34, v6, v11
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v11, v42, v6, v12
	v_dot2_f32_bf16 v5, v54, v6, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v6, v31, v7, v9
	v_dot2_f32_bf16 v9, v35, v7, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v10, v43, v7, v11
	v_dot2_f32_bf16 v5, v55, v7, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v6, v32, v8, v6
	v_dot2_f32_bf16 v7, v36, v8, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v9, v44, v8, v10
	v_dot2_f32_bf16 v5, v56, v8, v5
	s_wait_loadcnt 0x0
	v_dot2_f32_bf16 v6, v17, v1, v6
	v_dot2_f32_bf16 v7, v25, v1, v7
	v_dot2_f32_bf16 v8, v49, v1, v9
	v_dot2_f32_bf16 v1, v69, v1, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v5, v18, v2, v6
	v_dot2_f32_bf16 v6, v26, v2, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v7, v50, v2, v8
	v_dot2_f32_bf16 v1, v70, v2, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v2, v19, v3, v5
	v_dot2_f32_bf16 v5, v27, v3, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v6, v51, v3, v7
	v_dot2_f32_bf16 v1, v71, v3, v1
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v100, v20, v4, v2
	v_dot2_f32_bf16 v99, v28, v4, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v98, v52, v4, v6
	v_dot2_f32_bf16 v97, v72, v4, v1
.LBB0_15:                               ;   in Loop: Header=BB0_16 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s36
	s_addk_co_i32 s46, 0x400
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s46, s12
	s_cbranch_scc1 .LBB0_33
.LBB0_16:                               ;   Parent Loop BB0_8 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_20 Depth 3
                                        ;       Child Loop BB0_22 Depth 3
                                        ;       Child Loop BB0_24 Depth 3
                                        ;       Child Loop BB0_26 Depth 3
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s46, 0
	s_cselect_b32 s37, -1, 0
	s_add_co_i32 s36, s22, 0x2000
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s46, s36
	s_cselect_b32 s47, -1, 0
	s_wait_alu 0xfffe
	s_or_b32 s47, s37, s47
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s47
	s_wait_alu 0xfffe
	s_cbranch_vccz .LBB0_18
; %bb.17:                               ;   in Loop: Header=BB0_16 Depth=2
	s_and_saveexec_b32 s36, s2
	s_cbranch_execz .LBB0_15
	s_branch .LBB0_28
.LBB0_18:                               ;   in Loop: Header=BB0_16 Depth=2
	s_and_b32 s37, s37, exec_lo
	s_cselect_b32 s22, s22, s36
	s_mov_b32 s48, exec_lo
	s_wait_alu 0xfffe
	s_sub_co_i32 s36, s12, s22
	s_wait_loadcnt 0x0
	s_wait_alu 0xfffe
	s_min_u32 s47, s36, 0x2000
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu 0xfffe
	v_cmpx_gt_u32_e64 s47, v88
	s_cbranch_execz .LBB0_27
; %bb.19:                               ;   in Loop: Header=BB0_16 Depth=2
	v_dual_mov_b32 v1, v89 :: v_dual_mov_b32 v84, v88
	s_lshl_b64 s[36:37], s[22:23], 1
	s_mov_b32 s49, 0
	s_wait_alu 0xfffe
	s_add_nc_u64 s[36:37], s[20:21], s[36:37]
.LBB0_20:                               ;   Parent Loop BB0_8 Depth=1
                                        ;     Parent Loop BB0_16 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_lshlrev_b64_e32 v[2:3], 1, v[84:85]
	v_add_nc_u32_e32 v84, s41, v84
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v2, vcc_lo, s36, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s37, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s47, v84
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s49, vcc_lo, s49
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s44, v1
	s_and_not1_b32 exec_lo, exec_lo, s49
	s_cbranch_execnz .LBB0_20
; %bb.21:                               ;   in Loop: Header=BB0_16 Depth=2
	s_or_b32 exec_lo, exec_lo, s49
	v_dual_mov_b32 v1, v90 :: v_dual_mov_b32 v84, v88
	s_add_nc_u64 s[36:37], s[36:37], s[28:29]
	s_mov_b32 s49, 0
.LBB0_22:                               ;   Parent Loop BB0_8 Depth=1
                                        ;     Parent Loop BB0_16 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 1, v[84:85]
	v_add_nc_u32_e32 v84, s41, v84
	s_wait_alu 0xfffe
	v_add_co_u32 v2, vcc_lo, s36, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, s37, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s47, v84
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s49, vcc_lo, s49
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s44, v1
	s_and_not1_b32 exec_lo, exec_lo, s49
	s_cbranch_execnz .LBB0_22
; %bb.23:                               ;   in Loop: Header=BB0_16 Depth=2
	s_or_b32 exec_lo, exec_lo, s49
	v_dual_mov_b32 v1, v91 :: v_dual_mov_b32 v84, v88
	s_add_nc_u64 s[36:37], s[36:37], s[28:29]
	s_mov_b32 s49, 0
.LBB0_24:                               ;   Parent Loop BB0_8 Depth=1
                                        ;     Parent Loop BB0_16 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 1, v[84:85]
	v_add_nc_u32_e32 v84, s41, v84
	s_wait_alu 0xfffe
	v_add_co_u32 v2, vcc_lo, s36, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, s37, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s47, v84
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s49, vcc_lo, s49
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s44, v1
	s_and_not1_b32 exec_lo, exec_lo, s49
	s_cbranch_execnz .LBB0_24
; %bb.25:                               ;   in Loop: Header=BB0_16 Depth=2
	s_or_b32 exec_lo, exec_lo, s49
	v_dual_mov_b32 v1, v92 :: v_dual_mov_b32 v84, v88
	s_add_nc_u64 s[36:37], s[36:37], s[28:29]
	s_mov_b32 s49, 0
.LBB0_26:                               ;   Parent Loop BB0_8 Depth=1
                                        ;     Parent Loop BB0_16 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 1, v[84:85]
	v_add_nc_u32_e32 v84, s41, v84
	s_wait_alu 0xfffe
	v_add_co_u32 v2, vcc_lo, s36, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, s37, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s47, v84
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s49, vcc_lo, s49
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s44, v1
	s_and_not1_b32 exec_lo, exec_lo, s49
	s_cbranch_execnz .LBB0_26
.LBB0_27:                               ;   in Loop: Header=BB0_16 Depth=2
	s_or_b32 exec_lo, exec_lo, s48
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s36, s2
	s_cbranch_execz .LBB0_15
.LBB0_28:                               ;   in Loop: Header=BB0_16 Depth=2
	v_dual_mov_b32 v72, 0 :: v_dual_add_nc_u32 v103, s46, v82
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v53, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v84, s42, v103
	v_dual_mov_b32 v71, 0 :: v_dual_add_nc_u32 v104, 0x100, v103
	v_dual_mov_b32 v69, 0 :: v_dual_add_nc_u32 v102, 0x200, v103
	v_lshlrev_b64_e32 v[1:2], 1, v[84:85]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v84, s42, v104
	v_dual_mov_b32 v70, 0 :: v_dual_add_nc_u32 v101, 0x300, v103
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v39, 0
	v_lshlrev_b64_e32 v[3:4], 1, v[84:85]
	v_min_u32_e32 v84, s42, v102
	v_add_co_u32 v1, vcc_lo, v95, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, v96, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[5:6], 1, v[84:85]
	v_min_u32_e32 v84, s42, v101
	v_add_co_u32 v3, vcc_lo, v95, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, v96, v4, vcc_lo
	s_wait_dscnt 0x0
	v_lshlrev_b64_e32 v[7:8], 1, v[84:85]
	v_add_co_u32 v5, vcc_lo, v95, v5
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, v96, v6, vcc_lo
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v37, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v17, vcc_lo, v95, v7
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v18, null, v96, v8, vcc_lo
	s_clause 0x3
	global_load_b128 v[13:16], v[1:2], off
	global_load_b128 v[9:12], v[3:4], off
	global_load_b128 v[5:8], v[5:6], off
	global_load_b128 v[1:4], v[17:18], off
	v_cmp_gt_u32_e32 vcc_lo, s12, v103
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v51, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v64, 0 :: v_dual_mov_b32 v63, 0
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v61, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v67, 0
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v65, 0
	v_dual_mov_b32 v76, 0 :: v_dual_mov_b32 v75, 0
	v_dual_mov_b32 v74, 0 :: v_dual_mov_b32 v73, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v31, 0
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v59, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v57, 0
	v_dual_mov_b32 v80, 0 :: v_dual_mov_b32 v79, 0
	v_dual_mov_b32 v78, 0 :: v_dual_mov_b32 v77, 0
	s_and_saveexec_b32 s37, vcc_lo
	s_cbranch_execz .LBB0_14
; %bb.29:                               ;   in Loop: Header=BB0_16 Depth=2
	v_subrev_nc_u32_e32 v17, s22, v103
	v_cmp_gt_u32_e32 vcc_lo, s12, v104
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v59, 0
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v29, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v57, 0 :: v_dual_lshlrev_b32 v84, 1, v17
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v31, 0
	ds_load_b128 v[77:80], v84
	ds_load_b128 v[73:76], v84 offset:16384
	ds_load_b128 v[61:64], v84 offset:32768
	ds_load_b128 v[21:24], v84 offset:49152
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v19, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v65, 0
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v67, 0
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v51, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v39, 0
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v69, 0
	v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v71, 0
	v_mov_b32_e32 v72, 0
	s_and_saveexec_b32 s47, vcc_lo
	s_cbranch_execz .LBB0_13
; %bb.30:                               ;   in Loop: Header=BB0_16 Depth=2
	ds_load_b128 v[57:60], v84 offset:512
	ds_load_b128 v[65:68], v84 offset:16896
	ds_load_b128 v[45:48], v84 offset:33280
	ds_load_b128 v[37:40], v84 offset:49664
	v_cmp_gt_u32_e32 vcc_lo, s12, v102
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v32, 0
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v20, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v72, 0
	s_and_saveexec_b32 s48, vcc_lo
	s_cbranch_execz .LBB0_12
; %bb.31:                               ;   in Loop: Header=BB0_16 Depth=2
	ds_load_b128 v[29:32], v84 offset:1024
	ds_load_b128 v[33:36], v84 offset:17408
	ds_load_b128 v[41:44], v84 offset:33792
	ds_load_b128 v[53:56], v84 offset:50176
	v_dual_mov_b32 v17, 0 :: v_dual_mov_b32 v18, 0
	v_dual_mov_b32 v19, 0 :: v_dual_mov_b32 v20, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v72, 0
	s_mov_b32 s49, exec_lo
	v_cmpx_gt_u32_e64 s12, v101
	s_cbranch_execz .LBB0_11
; %bb.32:                               ;   in Loop: Header=BB0_16 Depth=2
	ds_load_b128 v[17:20], v84 offset:1536
	ds_load_b128 v[25:28], v84 offset:17920
	ds_load_b128 v[49:52], v84 offset:34304
	ds_load_b128 v[69:72], v84 offset:50688
	s_branch .LBB0_11
.LBB0_33:                               ;   in Loop: Header=BB0_8 Depth=1
	s_and_saveexec_b32 s22, s2
	s_cbranch_execz .LBB0_7
; %bb.34:                               ;   in Loop: Header=BB0_8 Depth=1
	v_add_f32_dpp v1, v100, v100 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v99, v99 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v98, v98 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v97, v97 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_xor_b32_e32 v5, 16, v93
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v5
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v5, v93, v5, vcc_lo
	v_add_f32_dpp v1, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v6, 2, v5
	v_add_f32_dpp v5, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v2, v6, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v11, v6, v10
	s_wait_dscnt 0x3
	ds_bpermute_b32 v9, v6, v8
	s_wait_dscnt 0x3
	ds_bpermute_b32 v7, v6, v5
	s_and_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_7
; %bb.35:                               ;   in Loop: Header=BB0_8 Depth=1
	v_lshlrev_b64_e32 v[3:4], 1, v[86:87]
	s_wait_dscnt 0x3
	v_add_f32_e32 v6, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_mul_f32_e32 v12, s8, v6
	v_add_co_u32 v1, vcc_lo, s4, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s5, v4, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s43
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_37
; %bb.36:                               ;   in Loop: Header=BB0_8 Depth=1
	global_load_u16 v6, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v6, 16, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v12, s9, v6
.LBB0_37:                               ;   in Loop: Header=BB0_8 Depth=1
	v_lshlrev_b32_e32 v6, 16, v94
	v_add_co_u32 v3, vcc_lo, s6, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s7, v4, vcc_lo
	s_wait_dscnt 0x2
	v_dual_add_f32 v13, v12, v6 :: v_dual_add_f32 v10, v10, v11
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cndmask_b32_e64 v12, v12, v13, s0
	v_mul_f32_e32 v10, s8, v10
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v13, v12, 16, 1
	v_or_b32_e32 v11, 0x400000, v12
	v_cmp_u_f32_e32 vcc_lo, v12, v12
	v_add3_u32 v13, v13, v12, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v11, v13, v11, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s43
	global_store_d16_hi_b16 v[3:4], v11, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_39
; %bb.38:                               ;   in Loop: Header=BB0_8 Depth=1
	v_add_co_u32 v11, vcc_lo, v1, s34
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v12, null, s35, v2, vcc_lo
	global_load_u16 v11, v[11:12], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v11, 16, v11
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v10, s9, v11
.LBB0_39:                               ;   in Loop: Header=BB0_8 Depth=1
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_add_f32 v11, v10, v6 :: v_dual_add_f32 v8, v8, v9
	v_add_co_u32 v3, vcc_lo, v3, s30
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s31, v4, vcc_lo
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
	s_and_not1_b32 vcc_lo, exec_lo, s43
	global_store_d16_hi_b16 v[3:4], v9, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_41
; %bb.40:                               ;   in Loop: Header=BB0_8 Depth=1
	v_add_co_u32 v9, vcc_lo, v1, s24
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v10, null, s25, v2, vcc_lo
	global_load_u16 v9, v[9:10], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v9, 16, v9
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v8, s9, v9
.LBB0_41:                               ;   in Loop: Header=BB0_8 Depth=1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v9, v8, v6
	v_add_co_u32 v3, vcc_lo, v3, s30
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v7
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s31, v4, vcc_lo
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
	s_and_not1_b32 vcc_lo, exec_lo, s43
	global_store_d16_hi_b16 v[3:4], v7, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_6
; %bb.42:                               ;   in Loop: Header=BB0_8 Depth=1
	v_add_co_u32 v1, vcc_lo, v1, s26
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s27, v2, vcc_lo
	global_load_u16 v1, v[1:2], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v1, 16, v1
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v5, s9, v1
	s_branch .LBB0_6
.LBB0_43:
	s_or_b32 exec_lo, exec_lo, s3
	s_mov_b32 s3, 0
.LBB0_44:
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_not1_b32 vcc_lo, exec_lo, s3
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_83
; %bb.45:
	s_wait_dscnt 0x3
	v_and_b32_e32 v2, 0x3ff, v0
	v_lshlrev_b32_e32 v4, 9, v83
	s_mov_b32 s3, 0
	s_wait_kmcnt 0x0
	s_mov_b32 s2, s19
	v_lshlrev_b32_e32 v84, 3, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v3, v83, 8, v84
	v_cmp_gt_u32_e32 vcc_lo, s12, v3
	s_and_saveexec_b32 s19, vcc_lo
	s_cbranch_execz .LBB0_52
; %bb.46:
	s_wait_dscnt 0x1
	v_dual_mov_b32 v9, v3 :: v_dual_lshlrev_b32 v0, 4, v2
	s_mov_b32 s22, s3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v5, s0, v4, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v6, null, 0, 0, s0
	s_wait_dscnt 0x0
	v_add_nc_u32_e32 v7, v4, v0
	v_add_co_u32 v0, s0, s20, v5
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v1, null, s21, v6, s0
	s_delay_alu instid0(VALU_DEP_3)
	v_mov_b32_e32 v8, v7
.LBB0_47:                               ; =>This Inner Loop Header: Depth=1
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
	s_cbranch_execnz .LBB0_47
; %bb.48:
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
.LBB0_49:                               ; =>This Inner Loop Header: Depth=1
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
	s_cbranch_execnz .LBB0_49
; %bb.50:
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
.LBB0_51:                               ; =>This Inner Loop Header: Depth=1
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
	s_cbranch_execnz .LBB0_51
.LBB0_52:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s19
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_55
; %bb.53:
	v_lshlrev_b32_e32 v0, 4, v2
	s_mul_i32 s0, s12, 6
	s_mul_u64 s[2:3], s[2:3], 6
	s_delay_alu instid0(SALU_CYCLE_1)
	s_add_nc_u64 s[2:3], s[20:21], s[2:3]
	s_wait_alu 0xfffe
	v_add3_u32 v5, v4, v0, s0
	v_add_co_u32 v0, s0, v4, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v1, null, 0, 0, s0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s2, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s3, v1, vcc_lo
	s_mov_b32 s2, 0
.LBB0_54:                               ; =>This Inner Loop Header: Depth=1
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
	s_cbranch_execnz .LBB0_54
.LBB0_55:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_u32_e64 s39, v83
	s_cbranch_execz .LBB0_83
; %bb.56:
	v_cmp_gt_u32_e32 vcc_lo, s13, v81
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_83
; %bb.57:
	s_cmp_eq_u64 s[16:17], 0
	s_mov_b32 s3, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[16:17], 0
	v_cmp_eq_u32_e64 s1, 31, v2
	s_cselect_b32 s19, -1, 0
	s_cmp_lg_u32 s12, 0
	v_dual_mov_b32 v80, 0 :: v_dual_lshlrev_b32 v83, 4, v2
	s_cselect_b32 s26, -1, 0
	s_add_co_i32 s27, s12, -8
	s_cmp_neq_f32 s9, 0
	v_mbcnt_lo_u32_b32 v85, -1, 0
	s_mov_b32 s2, s10
	s_mov_b32 s22, s11
	s_wait_alu 0xfffe
	s_mov_b32 s23, s3
	s_cselect_b32 s28, -1, 0
	s_lshl_b64 s[10:11], s[2:3], 2
	s_mul_u64 s[20:21], s[2:3], 6
	s_lshl_b32 s29, s12, 1
	s_lshl_b32 s30, s12, 2
	s_wait_alu 0xfffe
	s_lshl_b64 s[22:23], s[22:23], 1
	s_mul_i32 s31, s12, 6
	s_lshl_b64 s[24:25], s[2:3], 1
	s_branch .LBB0_60
.LBB0_58:                               ;   in Loop: Header=BB0_60 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v4, v6
	v_cndmask_b32_e64 v4, v0, v4, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_bfe_u32 v0, v4, 16, 1
	v_or_b32_e32 v5, 0x400000, v4
	v_add3_u32 v6, v0, v4, 0x7fff
	v_add_co_u32 v0, vcc_lo, v2, s22
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s23, v3, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v4, v4
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v2, v6, v5, vcc_lo
	global_store_d16_hi_b16 v[0:1], v2, off
.LBB0_59:                               ;   in Loop: Header=BB0_60 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s2
	v_add_nc_u32_e32 v81, s38, v81
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s13, v81
	s_or_b32 s3, vcc_lo, s3
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s3
	s_cbranch_execz .LBB0_83
.LBB0_60:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_68 Depth 2
	v_mov_b32_e32 v86, v80
	s_and_not1_b32 vcc_lo, exec_lo, s19
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_62
; %bb.61:                               ;   in Loop: Header=BB0_60 Depth=1
	v_mov_b32_e32 v82, v80
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[0:1], 1, v[81:82]
	v_add_co_u32 v0, vcc_lo, s16, v0
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v1, null, s17, v1, vcc_lo
	global_load_u16 v86, v[0:1], off
.LBB0_62:                               ;   in Loop: Header=BB0_60 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_73
; %bb.63:                               ;   in Loop: Header=BB0_60 Depth=1
	v_min_u32_e32 v0, s33, v81
	v_dual_mov_b32 v92, 0 :: v_dual_mov_b32 v91, 0
	v_dual_mov_b32 v82, v83 :: v_dual_mov_b32 v89, 0
	s_wait_dscnt 0x3
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_mad_co_u64_u32 v[0:1], null, v0, s18, 0
	v_mov_b32_e32 v90, 0
	s_mov_b32 s2, 0
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v87, vcc_lo, s14, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v88, null, s15, v1, vcc_lo
	s_branch .LBB0_68
.LBB0_64:                               ;   in Loop: Header=BB0_68 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s37
.LBB0_65:                               ;   in Loop: Header=BB0_68 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s36
.LBB0_66:                               ;   in Loop: Header=BB0_68 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s35
.LBB0_67:                               ;   in Loop: Header=BB0_68 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s34
	s_wait_loadcnt_dscnt 0x301
	v_dot2_f32_bf16 v76, v76, v12, v92
	v_dot2_f32_bf16 v68, v68, v12, v91
	v_dot2_f32_bf16 v36, v36, v12, v90
	s_wait_dscnt 0x0
	v_dot2_f32_bf16 v12, v16, v12, v89
	v_add_nc_u32_e32 v82, 0x800, v82
	v_dot2_f32_bf16 v16, v77, v13, v76
	v_dot2_f32_bf16 v68, v69, v13, v68
	v_dot2_f32_bf16 v36, v37, v13, v36
	v_dot2_f32_bf16 v12, v17, v13, v12
	s_addk_co_i32 s2, 0x400
	v_dot2_f32_bf16 v13, v78, v14, v16
	v_dot2_f32_bf16 v16, v70, v14, v68
	v_dot2_f32_bf16 v17, v38, v14, v36
	v_dot2_f32_bf16 v12, v18, v14, v12
	s_wait_alu 0xfffe
	s_cmp_lt_u32 s2, s12
	v_dot2_f32_bf16 v13, v79, v15, v13
	v_dot2_f32_bf16 v14, v71, v15, v16
	v_dot2_f32_bf16 v16, v39, v15, v17
	v_dot2_f32_bf16 v12, v19, v15, v12
	s_wait_loadcnt 0x2
	v_dot2_f32_bf16 v13, v64, v8, v13
	v_dot2_f32_bf16 v14, v56, v8, v14
	v_dot2_f32_bf16 v15, v48, v8, v16
	v_dot2_f32_bf16 v8, v40, v8, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v12, v65, v9, v13
	v_dot2_f32_bf16 v13, v57, v9, v14
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v14, v49, v9, v15
	v_dot2_f32_bf16 v8, v41, v9, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v9, v66, v10, v12
	v_dot2_f32_bf16 v12, v58, v10, v13
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v13, v50, v10, v14
	v_dot2_f32_bf16 v8, v42, v10, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v9, v67, v11, v9
	v_dot2_f32_bf16 v10, v59, v11, v12
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v12, v51, v11, v13
	v_dot2_f32_bf16 v8, v43, v11, v8
	s_wait_loadcnt 0x1
	v_dot2_f32_bf16 v9, v28, v4, v9
	v_dot2_f32_bf16 v10, v32, v4, v10
	v_dot2_f32_bf16 v11, v44, v4, v12
	v_dot2_f32_bf16 v4, v60, v4, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v8, v29, v5, v9
	v_dot2_f32_bf16 v9, v33, v5, v10
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v10, v45, v5, v11
	v_dot2_f32_bf16 v4, v61, v5, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v5, v30, v6, v8
	v_dot2_f32_bf16 v8, v34, v6, v9
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v9, v46, v6, v10
	v_dot2_f32_bf16 v4, v62, v6, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v5, v31, v7, v5
	v_dot2_f32_bf16 v6, v35, v7, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v8, v47, v7, v9
	v_dot2_f32_bf16 v4, v63, v7, v4
	s_wait_loadcnt 0x0
	v_dot2_f32_bf16 v5, v20, v0, v5
	v_dot2_f32_bf16 v6, v24, v0, v6
	v_dot2_f32_bf16 v7, v52, v0, v8
	v_dot2_f32_bf16 v0, v72, v0, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v4, v21, v1, v5
	v_dot2_f32_bf16 v5, v25, v1, v6
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v6, v53, v1, v7
	v_dot2_f32_bf16 v0, v73, v1, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v1, v22, v2, v4
	v_dot2_f32_bf16 v4, v26, v2, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v5, v54, v2, v6
	v_dot2_f32_bf16 v0, v74, v2, v0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v92, v23, v3, v1
	v_dot2_f32_bf16 v91, v27, v3, v4
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dot2_f32_bf16 v90, v55, v3, v5
	v_dot2_f32_bf16 v89, v75, v3, v0
	s_cbranch_scc0 .LBB0_74
.LBB0_68:                               ;   Parent Loop BB0_60 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_dual_mov_b32 v75, 0 :: v_dual_add_nc_u32 v18, s2, v84
	v_dual_mov_b32 v72, 0 :: v_dual_mov_b32 v61, 0
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v43, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v79, s27, v18
	v_dual_mov_b32 v73, 0 :: v_dual_add_nc_u32 v98, 0x100, v18
	v_dual_mov_b32 v63, 0 :: v_dual_add_nc_u32 v94, 0x200, v18
	v_lshlrev_b64_e32 v[0:1], 1, v[79:80]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v79, s27, v98
	v_dual_mov_b32 v74, 0 :: v_dual_add_nc_u32 v93, 0x300, v18
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v41, 0
	v_lshlrev_b64_e32 v[2:3], 1, v[79:80]
	v_min_u32_e32 v79, s27, v94
	v_add_co_u32 v0, vcc_lo, v87, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v88, v1, vcc_lo
	s_wait_dscnt 0x0
	v_lshlrev_b64_e32 v[4:5], 1, v[79:80]
	v_min_u32_e32 v79, s27, v93
	v_add_co_u32 v2, vcc_lo, v87, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, v88, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[6:7], 1, v[79:80]
	v_add_co_u32 v4, vcc_lo, v87, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v88, v5, vcc_lo
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v19, 0
	v_add_co_u32 v16, vcc_lo, v87, v6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v17, null, v88, v7, vcc_lo
	s_clause 0x3
	global_load_b128 v[12:15], v[0:1], off
	global_load_b128 v[8:11], v[2:3], off
	global_load_b128 v[4:7], v[4:5], off
	global_load_b128 v[0:3], v[16:17], off
	v_cmp_gt_u32_e32 vcc_lo, s12, v18
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v51, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v39, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v59, 0
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v57, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v71, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v69, 0
	v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v31, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v67, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v65, 0
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v79, 0
	v_dual_mov_b32 v64, 0 :: v_dual_mov_b32 v77, 0
	v_mov_b32_e32 v78, 0
	v_mov_b32_e32 v76, 0
	s_and_saveexec_b32 s34, vcc_lo
	s_cbranch_execz .LBB0_67
; %bb.69:                               ;   in Loop: Header=BB0_68 Depth=2
	v_dual_mov_b32 v64, 0 :: v_dual_add_nc_u32 v97, s29, v82
	v_dual_mov_b32 v65, 0 :: v_dual_add_nc_u32 v96, s30, v82
	v_dual_mov_b32 v66, 0 :: v_dual_add_nc_u32 v95, s31, v82
	ds_load_b128 v[68:71], v97
	ds_load_b128 v[36:39], v96
	ds_load_b128 v[76:79], v82
	ds_load_b128 v[16:19], v95
	v_cmp_gt_u32_e32 vcc_lo, s12, v98
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v29, 0 :: v_dual_mov_b32 v30, 0
	v_dual_mov_b32 v31, 0 :: v_dual_mov_b32 v20, 0
	v_dual_mov_b32 v21, 0 :: v_dual_mov_b32 v22, 0
	v_dual_mov_b32 v23, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v32, 0
	v_dual_mov_b32 v33, 0 :: v_dual_mov_b32 v34, 0
	v_dual_mov_b32 v35, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v60, 0
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v72, 0
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v74, 0
	v_mov_b32_e32 v75, 0
	s_and_saveexec_b32 s35, vcc_lo
	s_cbranch_execz .LBB0_66
; %bb.70:                               ;   in Loop: Header=BB0_68 Depth=2
	ds_load_b128 v[56:59], v97 offset:512
	ds_load_b128 v[48:51], v96 offset:512
	ds_load_b128 v[64:67], v82 offset:512
	ds_load_b128 v[40:43], v95 offset:512
	v_cmp_gt_u32_e32 vcc_lo, s12, v94
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v29, 0
	v_dual_mov_b32 v30, 0 :: v_dual_mov_b32 v31, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v32, 0 :: v_dual_mov_b32 v33, 0
	v_dual_mov_b32 v34, 0 :: v_dual_mov_b32 v35, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v61, 0
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v63, 0
	v_dual_mov_b32 v72, 0 :: v_dual_mov_b32 v73, 0
	v_dual_mov_b32 v74, 0 :: v_dual_mov_b32 v75, 0
	s_and_saveexec_b32 s36, vcc_lo
	s_cbranch_execz .LBB0_65
; %bb.71:                               ;   in Loop: Header=BB0_68 Depth=2
	ds_load_b128 v[32:35], v97 offset:1024
	ds_load_b128 v[44:47], v96 offset:1024
	ds_load_b128 v[28:31], v82 offset:1024
	ds_load_b128 v[60:63], v95 offset:1024
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v72, 0 :: v_dual_mov_b32 v73, 0
	v_dual_mov_b32 v74, 0 :: v_dual_mov_b32 v75, 0
	s_mov_b32 s37, exec_lo
	v_cmpx_gt_u32_e64 s12, v93
	s_cbranch_execz .LBB0_64
; %bb.72:                               ;   in Loop: Header=BB0_68 Depth=2
	ds_load_b128 v[24:27], v97 offset:1536
	ds_load_b128 v[52:55], v96 offset:1536
	ds_load_b128 v[20:23], v82 offset:1536
	ds_load_b128 v[72:75], v95 offset:1536
	s_branch .LBB0_64
.LBB0_73:                               ;   in Loop: Header=BB0_60 Depth=1
	v_dual_mov_b32 v89, 0 :: v_dual_mov_b32 v90, 0
	v_dual_mov_b32 v91, 0 :: v_dual_mov_b32 v92, 0
.LBB0_74:                               ;   in Loop: Header=BB0_60 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_dpp v0, v92, v92 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x3
	v_add_f32_dpp v1, v91, v91 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v90, v90 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v89, v89 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_xor_b32_e32 v4, 16, v85
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
	v_cndmask_b32_e32 v4, v85, v4, vcc_lo
	v_add_f32_dpp v0, v0, v0 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v9, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	s_wait_dscnt 0x0
	v_lshlrev_b32_e32 v5, 2, v4
	v_add_f32_dpp v4, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v1, v5, v0
	ds_bpermute_b32 v10, v5, v9
	ds_bpermute_b32 v8, v5, v7
	ds_bpermute_b32 v5, v5, v4
	s_and_saveexec_b32 s2, s1
	s_cbranch_execz .LBB0_59
; %bb.75:                               ;   in Loop: Header=BB0_60 Depth=1
	s_wait_dscnt 0x3
	v_add_f32_e32 v6, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v82, v80 :: v_dual_mul_f32 v11, s8, v6
	v_lshlrev_b64_e32 v[2:3], 1, v[81:82]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s4, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s5, v3, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s28
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_77
; %bb.76:                               ;   in Loop: Header=BB0_60 Depth=1
	global_load_u16 v6, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v6, 16, v6
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v11, s9, v6
.LBB0_77:                               ;   in Loop: Header=BB0_60 Depth=1
	s_wait_dscnt 0x2
	v_add_f32_e32 v9, v9, v10
	v_lshlrev_b32_e32 v6, 16, v86
	v_add_co_u32 v2, vcc_lo, s6, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mul_f32 v9, s8, v9 :: v_dual_add_f32 v12, v11, v6
	v_cndmask_b32_e64 v11, v12, v11, s0
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_bfe_u32 v12, v11, 16, 1
	v_or_b32_e32 v10, 0x400000, v11
	v_cmp_u_f32_e32 vcc_lo, v11, v11
	v_add3_u32 v12, v12, v11, 0x7fff
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_1)
	v_cndmask_b32_e32 v10, v12, v10, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s28
	global_store_d16_hi_b16 v[2:3], v10, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_79
; %bb.78:                               ;   in Loop: Header=BB0_60 Depth=1
	v_add_co_u32 v10, vcc_lo, v0, s24
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v11, null, s25, v1, vcc_lo
	global_load_u16 v10, v[10:11], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v10, 16, v10
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v9, s9, v10
.LBB0_79:                               ;   in Loop: Header=BB0_60 Depth=1
	s_wait_dscnt 0x1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_dual_add_f32 v10, v9, v6 :: v_dual_add_f32 v7, v7, v8
	v_add_co_u32 v2, vcc_lo, v2, s22
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s23, v3, vcc_lo
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
	s_and_not1_b32 vcc_lo, exec_lo, s28
	global_store_d16_hi_b16 v[2:3], v8, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_81
; %bb.80:                               ;   in Loop: Header=BB0_60 Depth=1
	v_add_co_u32 v8, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, s11, v1, vcc_lo
	global_load_u16 v8, v[8:9], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v8, 16, v8
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v7, s9, v8
.LBB0_81:                               ;   in Loop: Header=BB0_60 Depth=1
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v8, v7, v6
	v_add_co_u32 v2, vcc_lo, v2, s22
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s23, v3, vcc_lo
	v_cndmask_b32_e64 v7, v8, v7, s0
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_mul_f32_e32 v4, s8, v4
	v_bfe_u32 v8, v7, 16, 1
	v_or_b32_e32 v5, 0x400000, v7
	v_cmp_u_f32_e32 vcc_lo, v7, v7
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_add3_u32 v8, v8, v7, 0x7fff
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v5, v8, v5, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s28
	global_store_d16_hi_b16 v[2:3], v5, off
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_58
; %bb.82:                               ;   in Loop: Header=BB0_60 Depth=1
	v_add_co_u32 v0, vcc_lo, v0, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s21, v1, vcc_lo
	global_load_u16 v0, v[0:1], off
	s_wait_loadcnt 0x0
	v_lshlrev_b32_e32 v0, 16, v0
	s_delay_alu instid0(VALU_DEP_1)
	v_fmac_f32_e32 v4, s9, v0
	s_branch .LBB0_58
.LBB0_83:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201
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
		.amdhsa_next_free_sgpr 50
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 1
		.amdhsa_inst_pref_size 63
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
	.size	wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201, .Lfunc_end0-wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201
                                        ; -- End function
	.set wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201.num_vgpr, 105
	.set wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201.num_agpr, 0
	.set wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201.numbered_sgpr, 50
	.set wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201.num_named_barrier, 0
	.set wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201.private_seg_size, 0
	.set wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201.uses_vcc, 1
	.set wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201.uses_flat_scratch, 0
	.set wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201.has_dyn_sized_stack, 0
	.set wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201.has_recursion, 0
	.set wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 7944
; TotalNumSgprs: 52
; NumVgprs: 105
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 65536 bytes/workgroup (compile time only)
; SGPRBlocks: 0
; VGPRBlocks: 21
; NumSGPRsForWavesPerEU: 52
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
	.type	__hip_cuid_c3380bd2c72e91c9,@object ; @__hip_cuid_c3380bd2c72e91c9
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_c3380bd2c72e91c9
__hip_cuid_c3380bd2c72e91c9:
	.byte	0                               ; 0x0
	.size	__hip_cuid_c3380bd2c72e91c9, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_c3380bd2c72e91c9
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
    .name:           wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     52
    .sgpr_spill_count: 0
    .symbol:         wvSplitK_vllm_bf16_tn_n4_y1u4_gfx1201.kd
    .vgpr_count:     105
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
