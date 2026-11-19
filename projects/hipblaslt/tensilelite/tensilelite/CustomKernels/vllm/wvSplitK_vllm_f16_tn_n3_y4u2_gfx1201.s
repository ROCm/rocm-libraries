	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201 ; -- Begin function wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201
	.globl	wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201
	.p2align	8
	.type	wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201,@function
wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201:   ; @wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201
; %bb.0:
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x18
	s_load_b64 s[12:13], s[0:1], 0x0
	s_mov_b32 s6, -1
                                        ; implicit-def: $sgpr35
	s_wait_kmcnt 0x0
	s_lshl_b32 s4, s2, 6
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
	s_lshl_b32 s3, s2, 2
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
	s_lshl_b32 s3, s2, 2
	s_mov_b32 s35, 4
.LBB0_4:
	s_clause 0x2
	s_load_b64 s[20:21], s[0:1], 0x10
	s_load_b64 s[18:19], s[0:1], 0x20
	s_load_b256 s[4:11], s[0:1], 0x28
	v_bfe_u32 v69, v0, 10, 10
	s_mul_i32 s0, s35, ttmp9
	s_mul_i32 s34, s35, s3
	s_cmp_gt_u32 s12, 0x2aaa
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v57, s0, v69, 2
	s_mov_b32 s0, -1
	v_cmp_gt_u32_e32 vcc_lo, s35, v69
	s_cbranch_scc0 .LBB0_69
; %bb.5:
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB0_68
; %bb.6:
	s_lshl_b32 s0, s35, 2
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
	v_cmp_gt_u32_e32 vcc_lo, s36, v57
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_68
; %bb.7:
	v_dual_mov_b32 v60, v57 :: v_dual_and_b32 v1, 0x3ff, v0
	s_wait_kmcnt 0x0
	s_cmp_lg_u64 s[14:15], 0
	v_mbcnt_lo_u32_b32 v75, -1, 0
	s_cselect_b32 s0, -1, 0
	v_dual_mov_b32 v59, 0 :: v_dual_lshlrev_b32 v70, 3, v1
	v_cmp_eq_u32_e64 s1, 31, v1
	v_lshlrev_b32_e32 v1, 4, v1
	s_lshl_b32 s37, s35, 8
	s_add_co_i32 s38, s12, -8
	s_cmp_neq_f32 s9, 0
	v_lshl_add_u32 v71, v69, 8, v70
	v_lshl_add_u32 v72, v69, 9, v1
	s_mov_b32 s22, s19
	s_mov_b32 s26, s10
	s_mov_b32 s27, s23
	s_mov_b32 s28, s11
	v_add_nc_u32_e32 v73, 0x5400, v72
	v_add_nc_u32_e32 v74, 0xa800, v72
	s_mov_b32 s29, s23
	s_cselect_b32 s39, -1, 0
	s_lshl_b64 s[24:25], s[22:23], 1
	s_lshl_b64 s[26:27], s[26:27], 1
	s_lshl_b64 s[28:29], s[28:29], 1
	s_lshl_b32 s40, s35, 9
	s_mov_b32 s41, 0
	s_branch .LBB0_10
.LBB0_8:                                ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e32 v1, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v5, v1
	v_cndmask_b32_e64 v1, v5, v1, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v1, v1
	global_store_b16 v[3:4], v1, off offset:6
.LBB0_9:                                ;   in Loop: Header=BB0_10 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	v_add_nc_u32_e32 v60, s34, v60
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s36, v60
	s_or_b32 s41, vcc_lo, s41
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s41
	s_cbranch_execz .LBB0_68
.LBB0_10:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_18 Depth 2
                                        ;       Child Loop BB0_22 Depth 3
                                        ;       Child Loop BB0_24 Depth 3
                                        ;       Child Loop BB0_26 Depth 3
	v_cmp_gt_u32_e64 s2, s13, v60
	v_or_b32_e32 v81, 1, v60
	v_or_b32_e32 v77, 2, v60
	v_or_b32_e32 v76, 3, v60
                                        ; implicit-def: $vgpr1_vgpr2
                                        ; implicit-def: $vgpr3_vgpr4
	s_wait_alu 0xfffe
	s_and_b32 s22, s0, s2
	s_wait_alu 0xfffe
	s_xor_b32 s22, s22, -1
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s30, s22
	s_wait_alu 0xfffe
	s_xor_b32 s22, exec_lo, s30
	s_cbranch_execz .LBB0_12
; %bb.11:                               ;   in Loop: Header=BB0_10 Depth=1
	v_or_b32_e32 v81, 1, v60
	v_or_b32_e32 v77, 2, v60
	v_or_b32_e32 v76, 3, v60
	s_wait_dscnt 0xb
	v_mov_b32_e32 v2, v59
	v_mov_b32_e32 v4, v59
	v_min_u32_e32 v58, s33, v81
	v_min_u32_e32 v1, s33, v77
	v_min_u32_e32 v3, s33, v76
.LBB0_12:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_alu 0xfffe
	s_or_saveexec_b32 s22, s22
	s_wait_loadcnt 0x3
	v_dual_mov_b32 v83, 0 :: v_dual_mov_b32 v82, 0
	s_wait_loadcnt 0x1
	v_dual_mov_b32 v80, 0 :: v_dual_mov_b32 v79, 0
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v78, 0
	s_wait_alu 0xfffe
	s_xor_b32 exec_lo, exec_lo, s22
	s_cbranch_execz .LBB0_14
; %bb.13:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0xb
	v_dual_mov_b32 v61, v59 :: v_dual_mov_b32 v2, v59
	v_min_u32_e32 v58, s33, v81
	v_min_u32_e32 v1, s33, v77
	v_min_u32_e32 v3, s33, v76
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[4:5], 1, v[60:61]
	s_wait_dscnt 0x0
	v_lshlrev_b64_e32 v[6:7], 1, v[58:59]
	v_lshlrev_b64_e32 v[8:9], 1, v[1:2]
	v_add_co_u32 v10, vcc_lo, s14, v4
	v_mov_b32_e32 v4, v59
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v11, null, s15, v5, vcc_lo
	v_add_co_u32 v5, vcc_lo, s14, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[12:13], 1, v[3:4]
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, s15, v7, vcc_lo
	v_add_co_u32 v7, vcc_lo, s14, v8
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, s15, v9, vcc_lo
	v_add_co_u32 v12, vcc_lo, s14, v12
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v13, null, s15, v13, vcc_lo
	s_clause 0x3
	global_load_u16 v82, v[10:11], off
	global_load_u16 v80, v[5:6], off
	global_load_u16 v79, v[7:8], off
	global_load_u16 v78, v[12:13], off
.LBB0_14:                               ;   in Loop: Header=BB0_10 Depth=1
	s_or_b32 exec_lo, exec_lo, s22
	s_wait_dscnt 0x0
	v_mad_co_u64_u32 v[5:6], null, v58, s18, 0
	v_mad_co_u64_u32 v[7:8], null, v1, s18, 0
	v_mad_co_u64_u32 v[9:10], null, v3, s18, 0
	v_min_u32_e32 v12, s33, v60
	v_dual_mov_b32 v84, 0 :: v_dual_mov_b32 v91, 0
	v_mov_b32_e32 v1, v6
	v_dual_mov_b32 v3, v8 :: v_dual_mov_b32 v86, 0
	v_mov_b32_e32 v6, v10
	v_dual_mov_b32 v88, 0 :: v_dual_mov_b32 v93, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mad_co_u64_u32 v[10:11], null, v59, s18, v[1:2]
	v_mad_co_u64_u32 v[1:2], null, v2, s18, v[3:4]
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mad_co_u64_u32 v[2:3], null, v4, s18, v[6:7]
	v_mad_co_u64_u32 v[3:4], null, v12, s18, 0
	v_dual_mov_b32 v6, v10 :: v_dual_mov_b32 v85, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_dual_mov_b32 v8, v1 :: v_dual_mov_b32 v87, 0
	v_dual_mov_b32 v10, v2 :: v_dual_mov_b32 v89, 0
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[61:62], 1, v[3:4]
	v_lshlrev_b64_e32 v[63:64], 1, v[5:6]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[65:66], 1, v[7:8]
	v_lshlrev_b64_e32 v[67:68], 1, v[9:10]
	v_mov_b32_e32 v90, 0
	v_mov_b32_e32 v92, 0
	v_mov_b32_e32 v94, 0
	s_mov_b32 s22, 0
	s_wait_alu 0xfffe
	s_mov_b32 s42, s22
	s_branch .LBB0_18
.LBB0_15:                               ;   in Loop: Header=BB0_18 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s43
.LBB0_16:                               ;   in Loop: Header=BB0_18 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s31
	s_wait_loadcnt_dscnt 0x701
	v_dot2_f32_f16 v58, v53, v37, v93
	s_wait_loadcnt 0x6
	v_dot2_f32_f16 v91, v53, v25, v91
	s_wait_loadcnt 0x5
	v_dot2_f32_f16 v93, v53, v29, v94
	s_wait_loadcnt 0x4
	v_dot2_f32_f16 v53, v53, v33, v92
	v_dot2_f32_f16 v89, v49, v25, v89
	v_dot2_f32_f16 v58, v54, v38, v58
	v_dot2_f32_f16 v91, v54, v26, v91
	v_dot2_f32_f16 v92, v54, v30, v93
	v_dot2_f32_f16 v53, v54, v34, v53
	v_dot2_f32_f16 v88, v49, v29, v88
	v_dot2_f32_f16 v54, v55, v39, v58
	v_dot2_f32_f16 v58, v55, v27, v91
	v_dot2_f32_f16 v91, v55, v31, v92
	v_dot2_f32_f16 v53, v55, v35, v53
	s_wait_dscnt 0x0
	v_dot2_f32_f16 v25, v45, v25, v86
	v_dot2_f32_f16 v54, v56, v40, v54
	v_dot2_f32_f16 v55, v56, v28, v58
	v_dot2_f32_f16 v58, v56, v32, v91
	v_dot2_f32_f16 v53, v56, v36, v53
	v_dot2_f32_f16 v56, v49, v37, v90
	v_dot2_f32_f16 v37, v45, v37, v83
	v_dot2_f32_f16 v49, v49, v33, v87
	v_dot2_f32_f16 v29, v45, v29, v85
	v_dot2_f32_f16 v33, v45, v33, v84
	v_dot2_f32_f16 v56, v50, v38, v56
	v_dot2_f32_f16 v37, v46, v38, v37
	v_dot2_f32_f16 v83, v50, v26, v89
	v_dot2_f32_f16 v87, v50, v30, v88
	v_dot2_f32_f16 v49, v50, v34, v49
	v_dot2_f32_f16 v38, v51, v39, v56
	v_dot2_f32_f16 v37, v47, v39, v37
	s_wait_loadcnt 0x2
	v_dot2_f32_f16 v45, v41, v5, v55
	v_dot2_f32_f16 v25, v46, v26, v25
	v_dot2_f32_f16 v26, v46, v30, v29
	v_dot2_f32_f16 v38, v52, v40, v38
	v_dot2_f32_f16 v37, v48, v40, v37
	v_dot2_f32_f16 v40, v41, v1, v54
	v_dot2_f32_f16 v29, v46, v34, v33
	v_dot2_f32_f16 v50, v51, v27, v83
	v_dot2_f32_f16 v56, v51, v31, v87
	v_dot2_f32_f16 v49, v51, v35, v49
	v_dot2_f32_f16 v30, v42, v2, v40
	v_dot2_f32_f16 v33, v42, v6, v45
	v_dot2_f32_f16 v25, v47, v27, v25
	v_dot2_f32_f16 v26, v47, v31, v26
	v_dot2_f32_f16 v27, v47, v35, v29
	v_dot2_f32_f16 v39, v52, v28, v50
	v_dot2_f32_f16 v50, v52, v32, v56
	v_dot2_f32_f16 v49, v52, v36, v49
	v_dot2_f32_f16 v29, v43, v3, v30
	v_dot2_f32_f16 v30, v43, v7, v33
	v_dot2_f32_f16 v25, v48, v28, v25
	v_dot2_f32_f16 v26, v48, v32, v26
	v_dot2_f32_f16 v27, v48, v36, v27
	v_dot2_f32_f16 v93, v44, v4, v29
	v_dot2_f32_f16 v91, v44, v8, v30
	s_wait_loadcnt 0x1
	v_dot2_f32_f16 v28, v41, v9, v58
	s_wait_loadcnt 0x0
	v_dot2_f32_f16 v29, v41, v13, v53
	v_dot2_f32_f16 v30, v21, v1, v38
	v_dot2_f32_f16 v31, v21, v5, v39
	v_dot2_f32_f16 v32, v21, v9, v50
	v_dot2_f32_f16 v21, v21, v13, v49
	v_dot2_f32_f16 v1, v17, v1, v37
	v_dot2_f32_f16 v5, v17, v5, v25
	v_dot2_f32_f16 v9, v17, v9, v26
	v_dot2_f32_f16 v13, v17, v13, v27
	v_dot2_f32_f16 v28, v42, v10, v28
	v_dot2_f32_f16 v29, v42, v14, v29
	v_dot2_f32_f16 v30, v22, v2, v30
	v_dot2_f32_f16 v31, v22, v6, v31
	v_dot2_f32_f16 v32, v22, v10, v32
	v_dot2_f32_f16 v17, v22, v14, v21
	v_dot2_f32_f16 v1, v18, v2, v1
	v_dot2_f32_f16 v2, v18, v6, v5
	v_dot2_f32_f16 v5, v18, v10, v9
	v_dot2_f32_f16 v6, v18, v14, v13
	v_dot2_f32_f16 v28, v43, v11, v28
	v_dot2_f32_f16 v29, v43, v15, v29
	v_dot2_f32_f16 v30, v23, v3, v30
	v_dot2_f32_f16 v31, v23, v7, v31
	v_dot2_f32_f16 v32, v23, v11, v32
	v_dot2_f32_f16 v9, v23, v15, v17
	v_dot2_f32_f16 v1, v19, v3, v1
	v_dot2_f32_f16 v2, v19, v7, v2
	v_dot2_f32_f16 v3, v19, v11, v5
	v_dot2_f32_f16 v5, v19, v15, v6
	v_dot2_f32_f16 v94, v44, v12, v28
	v_dot2_f32_f16 v92, v44, v16, v29
	v_dot2_f32_f16 v90, v24, v4, v30
	v_dot2_f32_f16 v89, v24, v8, v31
	v_dot2_f32_f16 v88, v24, v12, v32
	v_dot2_f32_f16 v87, v24, v16, v9
	v_dot2_f32_f16 v83, v20, v4, v1
	v_dot2_f32_f16 v86, v20, v8, v2
	v_dot2_f32_f16 v85, v20, v12, v3
	v_dot2_f32_f16 v84, v20, v16, v5
.LBB0_17:                               ;   in Loop: Header=BB0_18 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s30
	s_addk_co_i32 s42, 0x200
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s42, s12
	s_cbranch_scc1 .LBB0_31
.LBB0_18:                               ;   Parent Loop BB0_10 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_22 Depth 3
                                        ;       Child Loop BB0_24 Depth 3
                                        ;       Child Loop BB0_26 Depth 3
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s42, 0
	s_cselect_b32 s31, -1, 0
	s_add_co_i32 s30, s22, 0x2a00
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
	s_branch .LBB0_28
.LBB0_20:                               ;   in Loop: Header=BB0_18 Depth=2
	s_and_b32 s31, s31, exec_lo
	s_cselect_b32 s22, s22, s30
	s_mov_b32 s44, exec_lo
	s_wait_alu 0xfffe
	s_sub_co_i32 s30, s12, s22
	s_wait_loadcnt 0x0
	s_wait_alu 0xfffe
	s_min_u32 s43, s30, 0x2a00
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu 0xfffe
	v_cmpx_gt_u32_e64 s43, v71
	s_cbranch_execz .LBB0_27
; %bb.21:                               ;   in Loop: Header=BB0_18 Depth=2
	v_dual_mov_b32 v1, v72 :: v_dual_mov_b32 v58, v71
	s_lshl_b64 s[30:31], s[22:23], 1
	s_mov_b32 s45, 0
	s_wait_alu 0xfffe
	s_add_nc_u64 s[30:31], s[20:21], s[30:31]
.LBB0_22:                               ;   Parent Loop BB0_10 Depth=1
                                        ;     Parent Loop BB0_18 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_lshlrev_b64_e32 v[2:3], 1, v[58:59]
	v_add_nc_u32_e32 v58, s37, v58
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v2, vcc_lo, s30, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s31, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s43, v58
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s45, vcc_lo, s45
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s40, v1
	s_and_not1_b32 exec_lo, exec_lo, s45
	s_cbranch_execnz .LBB0_22
; %bb.23:                               ;   in Loop: Header=BB0_18 Depth=2
	s_or_b32 exec_lo, exec_lo, s45
	v_dual_mov_b32 v1, v73 :: v_dual_mov_b32 v58, v71
	s_add_nc_u64 s[30:31], s[30:31], s[24:25]
	s_mov_b32 s45, 0
.LBB0_24:                               ;   Parent Loop BB0_10 Depth=1
                                        ;     Parent Loop BB0_18 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 1, v[58:59]
	v_add_nc_u32_e32 v58, s37, v58
	s_wait_alu 0xfffe
	v_add_co_u32 v2, vcc_lo, s30, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, s31, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s43, v58
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s45, vcc_lo, s45
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s40, v1
	s_and_not1_b32 exec_lo, exec_lo, s45
	s_cbranch_execnz .LBB0_24
; %bb.25:                               ;   in Loop: Header=BB0_18 Depth=2
	s_or_b32 exec_lo, exec_lo, s45
	v_dual_mov_b32 v1, v74 :: v_dual_mov_b32 v58, v71
	s_add_nc_u64 s[30:31], s[30:31], s[24:25]
	s_mov_b32 s45, 0
.LBB0_26:                               ;   Parent Loop BB0_10 Depth=1
                                        ;     Parent Loop BB0_18 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 1, v[58:59]
	v_add_nc_u32_e32 v58, s37, v58
	s_wait_alu 0xfffe
	v_add_co_u32 v2, vcc_lo, s30, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, s31, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s43, v58
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s45, vcc_lo, s45
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s40, v1
	s_and_not1_b32 exec_lo, exec_lo, s45
	s_cbranch_execnz .LBB0_26
.LBB0_27:                               ;   in Loop: Header=BB0_18 Depth=2
	s_or_b32 exec_lo, exec_lo, s44
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB0_17
.LBB0_28:                               ;   in Loop: Header=BB0_18 Depth=2
	v_dual_mov_b32 v20, 0 :: v_dual_add_nc_u32 v95, s42, v70
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v47, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v58, s38, v95
	v_dual_mov_b32 v19, 0 :: v_dual_add_nc_u32 v96, 0x100, v95
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v45, 0
	v_lshlrev_b64_e32 v[1:2], 1, v[58:59]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_min_u32_e32 v58, s38, v96
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v21, 0
	v_add_co_u32 v9, vcc_lo, s16, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v10, null, s17, v2, vcc_lo
	v_lshlrev_b64_e32 v[1:2], 1, v[58:59]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v3, vcc_lo, v9, v61
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, v10, v62, vcc_lo
	v_add_co_u32 v5, vcc_lo, v9, v63
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, v10, v64, vcc_lo
	v_add_co_u32 v7, vcc_lo, v9, v65
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, v10, v66, vcc_lo
	v_add_co_u32 v11, vcc_lo, s16, v1
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v12, null, s17, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, v9, v67
	s_clause 0x1
	global_load_b128 v[37:40], v[3:4], off
	global_load_b128 v[25:28], v[5:6], off
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, v10, v68, vcc_lo
	v_add_co_u32 v3, vcc_lo, v11, v61
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, v12, v62, vcc_lo
	v_add_co_u32 v5, vcc_lo, v11, v63
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, v12, v64, vcc_lo
	v_add_co_u32 v9, vcc_lo, v11, v65
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v10, null, v12, v66, vcc_lo
	v_add_co_u32 v13, vcc_lo, v11, v67
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v14, null, v12, v68, vcc_lo
	s_clause 0x5
	global_load_b128 v[29:32], v[7:8], off
	global_load_b128 v[33:36], v[1:2], off
	global_load_b128 v[1:4], v[3:4], off
	global_load_b128 v[5:8], v[5:6], off
	global_load_b128 v[9:12], v[9:10], off
	global_load_b128 v[13:16], v[13:14], off
	v_cmp_gt_u32_e32 vcc_lo, s12, v95
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v51, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v53, 0
	s_and_saveexec_b32 s31, vcc_lo
	s_cbranch_execz .LBB0_16
; %bb.29:                               ;   in Loop: Header=BB0_18 Depth=2
	v_subrev_nc_u32_e32 v17, s22, v95
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v21, 0
	s_delay_alu instid0(VALU_DEP_3)
	v_dual_mov_b32 v41, 0 :: v_dual_lshlrev_b32 v58, 1, v17
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v17, 0
	ds_load_b128 v[49:52], v58 offset:21504
	ds_load_b128 v[53:56], v58
	ds_load_b128 v[45:48], v58 offset:43008
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v19, 0
	v_mov_b32_e32 v20, 0
	s_mov_b32 s43, exec_lo
	v_cmpx_gt_u32_e64 s12, v96
	s_cbranch_execz .LBB0_15
; %bb.30:                               ;   in Loop: Header=BB0_18 Depth=2
	ds_load_b128 v[21:24], v58 offset:22016
	ds_load_b128 v[41:44], v58 offset:512
	ds_load_b128 v[17:20], v58 offset:43520
	s_branch .LBB0_15
.LBB0_31:                               ;   in Loop: Header=BB0_10 Depth=1
	s_and_saveexec_b32 s22, s2
	s_cbranch_execz .LBB0_9
; %bb.32:                               ;   in Loop: Header=BB0_10 Depth=1
	v_add_f32_dpp v4, v94, v94 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v92, v92 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v90, v90 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v89, v89 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v88, v88 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v93, v93 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v91, v91 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_xor_b32_e32 v3, 16, v75
	v_add_f32_dpp v24, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v22, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v18, v7, v7 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v8, v8 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v87, v87 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v83, v83 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v86, v86 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v85, v85 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v84, v84 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v3
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v3, v75, v3, vcc_lo
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v8, v8 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_lshlrev_b32_e32 v3, 2, v3
	v_add_f32_dpp v1, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v26, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v15, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v11, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v9, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v8, v8 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v2, v3, v1
	ds_bpermute_b32 v27, v3, v26
	ds_bpermute_b32 v25, v3, v24
	ds_bpermute_b32 v23, v3, v22
	ds_bpermute_b32 v21, v3, v20
	ds_bpermute_b32 v19, v3, v18
	ds_bpermute_b32 v14, v3, v13
	ds_bpermute_b32 v17, v3, v15
	ds_bpermute_b32 v12, v3, v11
	ds_bpermute_b32 v10, v3, v9
	ds_bpermute_b32 v8, v3, v7
	ds_bpermute_b32 v6, v3, v5
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
	s_and_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_9
; %bb.33:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0xb
	v_dual_mov_b32 v61, v59 :: v_dual_add_f32 v16, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[3:4], 1, v[60:61]
	v_mul_f32_e32 v28, s8, v16
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v1, vcc_lo, s4, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s5, v4, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_35
; %bb.34:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v16, v[1:2], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v28, s9, v16, v28 op_sel_hi:[0,1,0]
.LBB0_35:                               ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e32 v16, v82
	v_add_co_u32 v3, vcc_lo, s6, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s7, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v29, v28, v16
	v_cmp_gt_u32_e64 s2, s13, v81
	v_cndmask_b32_e64 v28, v28, v29, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v28, v28
	global_store_b16 v[3:4], v28, off
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB0_45
; %bb.36:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0xa
	v_add_f32_e32 v26, v26, v27
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v26, s8, v26
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_38
; %bb.37:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v27, v[1:2], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v26, s9, v27, v26 op_sel_hi:[0,1,0]
.LBB0_38:                               ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e32 v27, v80
	v_cmp_gt_u32_e32 vcc_lo, s13, v77
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v27, v26, v27
	v_cndmask_b32_e64 v26, v26, v27, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v26, v26
	global_store_b16 v[3:4], v26, off offset:2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_45
; %bb.39:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x9
	v_add_f32_e32 v24, v24, v25
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v24, s8, v24
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_41
; %bb.40:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v25, v[1:2], off offset:4
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v24, s9, v25, v24 op_sel_hi:[0,1,0]
.LBB0_41:                               ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e32 v25, v79
	v_cmp_gt_u32_e32 vcc_lo, s13, v76
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v25, v24, v25
	v_cndmask_b32_e64 v24, v24, v25, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v24, v24
	global_store_b16 v[3:4], v24, off offset:4
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_45
; %bb.42:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x8
	v_add_f32_e32 v22, v22, v23
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v22, s8, v22
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_44
; %bb.43:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v23, v[1:2], off offset:6
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v22, s9, v23, v22 op_sel_hi:[0,1,0]
.LBB0_44:                               ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e32 v23, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v23, v22, v23
	v_cndmask_b32_e64 v22, v22, v23, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v22, v22
	global_store_b16 v[3:4], v22, off offset:6
.LBB0_45:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s30
	s_wait_dscnt 0x7
	v_add_f32_e32 v20, v20, v21
	v_add_co_u32 v1, vcc_lo, v1, s26
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s27, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v20, s8, v20
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_47
; %bb.46:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v21, v[1:2], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v20, s9, v21, v20 op_sel_hi:[0,1,0]
.LBB0_47:                               ;   in Loop: Header=BB0_10 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v21, v20, v16
	v_add_co_u32 v3, vcc_lo, v3, s28
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s29, v4, vcc_lo
	v_cndmask_b32_e64 v20, v20, v21, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v20, v20
	global_store_b16 v[3:4], v20, off
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB0_57
; %bb.48:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x6
	v_add_f32_e32 v18, v18, v19
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v18, s8, v18
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_50
; %bb.49:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v19, v[1:2], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v18, s9, v19, v18 op_sel_hi:[0,1,0]
.LBB0_50:                               ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e32 v19, v80
	v_cmp_gt_u32_e32 vcc_lo, s13, v77
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v19, v18, v19
	v_cndmask_b32_e64 v18, v18, v19, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18, v18
	global_store_b16 v[3:4], v18, off offset:2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_57
; %bb.51:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x5
	v_add_f32_e32 v13, v13, v14
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v13, s8, v13
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_53
; %bb.52:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v14, v[1:2], off offset:4
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v13, s9, v14, v13 op_sel_hi:[0,1,0]
.LBB0_53:                               ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e32 v14, v79
	v_cmp_gt_u32_e32 vcc_lo, s13, v76
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v14, v13, v14
	v_cndmask_b32_e64 v13, v13, v14, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v13, v13
	global_store_b16 v[3:4], v13, off offset:4
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_57
; %bb.54:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x4
	v_add_f32_e32 v13, v15, v17
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v13, s8, v13
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v14, v[1:2], off offset:6
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v13, s9, v14, v13 op_sel_hi:[0,1,0]
.LBB0_56:                               ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e32 v14, v78
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v14, v13, v14
	v_cndmask_b32_e64 v13, v13, v14, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v13, v13
	global_store_b16 v[3:4], v13, off offset:6
.LBB0_57:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s30
	s_wait_dscnt 0x3
	v_add_f32_e32 v11, v11, v12
	v_add_co_u32 v1, vcc_lo, v1, s26
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s27, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v11, s8, v11
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_59
; %bb.58:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v12, v[1:2], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v11, s9, v12, v11 op_sel_hi:[0,1,0]
.LBB0_59:                               ;   in Loop: Header=BB0_10 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v12, v11, v16
	v_add_co_u32 v3, vcc_lo, v3, s28
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s29, v4, vcc_lo
	v_cndmask_b32_e64 v11, v11, v12, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v11, v11
	global_store_b16 v[3:4], v11, off
	s_and_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB0_9
; %bb.60:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x2
	v_add_f32_e32 v9, v9, v10
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, s8, v9
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_62
; %bb.61:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v10, v[1:2], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v9, s9, v10, v9 op_sel_hi:[0,1,0]
.LBB0_62:                               ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e32 v10, v80
	v_cmp_gt_u32_e32 vcc_lo, s13, v77
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v10, v9, v10
	v_cndmask_b32_e64 v9, v9, v10, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9, v9
	global_store_b16 v[3:4], v9, off offset:2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_9
; %bb.63:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x1
	v_add_f32_e32 v7, v7, v8
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v7, s8, v7
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_65
; %bb.64:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v8, v[1:2], off offset:4
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v7, s9, v8, v7 op_sel_hi:[0,1,0]
.LBB0_65:                               ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e32 v8, v79
	v_cmp_gt_u32_e32 vcc_lo, s13, v76
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v8, v7, v8
	v_cndmask_b32_e64 v7, v7, v8, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v7, v7
	global_store_b16 v[3:4], v7, off offset:4
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_9
; %bb.66:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v5, s8, v5
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_8
; %bb.67:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v1, v[1:2], off offset:6
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v5, s9, v1, v5 op_sel_hi:[0,1,0]
	s_branch .LBB0_8
.LBB0_68:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s3
	s_mov_b32 s0, 0
.LBB0_69:
	s_wait_alu 0xfffe
	s_and_not1_b32 vcc_lo, exec_lo, s0
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_128
; %bb.70:
	s_wait_dscnt 0xb
	v_and_b32_e32 v2, 0x3ff, v0
	s_mov_b32 s1, exec_lo
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b32_e32 v66, 3, v2
	v_lshl_add_u32 v3, v69, 8, v66
	s_delay_alu instid0(VALU_DEP_1)
	v_cmpx_gt_u32_e64 s12, v3
	s_cbranch_execz .LBB0_77
; %bb.71:
	v_lshlrev_b32_e32 v0, 9, v69
	s_wait_dscnt 0x1
	v_dual_mov_b32 v8, v3 :: v_dual_lshlrev_b32 v1, 4, v2
	s_mov_b32 s2, 0
	s_wait_dscnt 0x0
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v6, v0, v1
	v_mov_b32_e32 v7, v6
	v_add_co_u32 v4, s0, v0, v1
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v5, null, 0, 0, s0
	s_wait_kmcnt 0x0
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s20, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s21, v5, vcc_lo
.LBB0_72:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[9:12], v[0:1], off
	v_add_nc_u32_e32 v8, 0x1000, v8
	v_add_co_u32 v0, vcc_lo, 0x2000, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s0, s12, v8
	s_wait_alu 0xfffe
	s_or_b32 s2, s0, s2
	s_wait_loadcnt 0x0
	ds_store_b128 v7, v[9:12]
	v_add_nc_u32_e32 v7, 0x2000, v7
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_72
; %bb.73:
	s_or_b32 exec_lo, exec_lo, s2
	s_mov_b32 s3, 0
	s_mov_b32 s2, s19
	v_lshl_add_u32 v7, s12, 1, v6
	s_wait_alu 0xfffe
	s_lshl_b64 s[22:23], s[2:3], 1
	v_mov_b32_e32 v8, v3
	s_wait_alu 0xfffe
	s_add_nc_u64 s[22:23], s[20:21], s[22:23]
	s_mov_b32 s19, s3
	s_wait_alu 0xfffe
	v_add_co_u32 v0, vcc_lo, s22, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s23, v5, vcc_lo
.LBB0_74:                               ; =>This Inner Loop Header: Depth=1
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
	s_cbranch_execnz .LBB0_74
; %bb.75:
	s_or_b32 exec_lo, exec_lo, s19
	s_lshl_b64 s[2:3], s[2:3], 2
	v_lshl_add_u32 v6, s12, 2, v6
	s_wait_alu 0xfffe
	s_add_nc_u64 s[2:3], s[20:21], s[2:3]
	s_wait_alu 0xfffe
	v_add_co_u32 v0, vcc_lo, s2, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s3, v5, vcc_lo
	s_mov_b32 s2, 0
.LBB0_76:                               ; =>This Inner Loop Header: Depth=1
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
	s_cbranch_execnz .LBB0_76
.LBB0_77:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_u32_e64 s35, v69
	s_cbranch_execz .LBB0_128
; %bb.78:
	v_cmp_gt_u32_e32 vcc_lo, s13, v57
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_128
; %bb.79:
	s_wait_kmcnt 0x0
	s_cmp_eq_u64 s[14:15], 0
	v_mbcnt_lo_u32_b32 v67, -1, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[14:15], 0
	s_mov_b32 s3, 0
	s_cselect_b32 s19, -1, 0
	s_cmp_lg_u32 s12, 0
	v_cmp_eq_u32_e64 s1, 31, v2
	s_cselect_b32 s22, -1, 0
	s_add_co_i32 s23, s12, -8
	s_cmp_neq_f32 s9, 0
	v_lshlrev_b32_e32 v68, 4, v2
	v_mov_b32_e32 v56, 0
	v_xor_b32_e32 v69, 16, v67
	s_mov_b32 s2, s10
	s_mov_b32 s20, s11
	s_wait_alu 0xfffe
	s_mov_b32 s21, s3
	s_cselect_b32 s24, -1, 0
	s_lshl_b32 s25, s12, 1
	s_lshl_b64 s[10:11], s[2:3], 1
	s_wait_alu 0xfffe
	s_lshl_b64 s[20:21], s[20:21], 1
	s_lshl_b32 s26, s12, 2
	s_branch .LBB0_82
.LBB0_80:                               ;   in Loop: Header=BB0_82 Depth=1
	v_cvt_f32_f16_e32 v0, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v4, v0
	v_cndmask_b32_e64 v0, v0, v4, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v0, v0
	global_store_b16 v[2:3], v0, off offset:6
.LBB0_81:                               ;   in Loop: Header=BB0_82 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s27
	v_add_nc_u32_e32 v57, s34, v57
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s13, v57
	s_or_b32 s3, vcc_lo, s3
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s3
	s_cbranch_execz .LBB0_128
.LBB0_82:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_88 Depth 2
	v_or_b32_e32 v73, 1, v57
	v_dual_mov_b32 v74, v56 :: v_dual_mov_b32 v71, v56
	v_mov_b32_e32 v72, v56
	v_mov_b32_e32 v70, v56
	s_delay_alu instid0(VALU_DEP_4)
	v_min_u32_e32 v55, s33, v73
	s_and_not1_b32 vcc_lo, exec_lo, s19
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_84
; %bb.83:                               ;   in Loop: Header=BB0_82 Depth=1
	v_or_b32_e32 v0, 2, v57
	s_wait_dscnt 0xb
	v_dual_mov_b32 v58, v56 :: v_dual_mov_b32 v1, v56
	v_or_b32_e32 v6, 3, v57
	s_wait_dscnt 0x1
	v_mov_b32_e32 v7, v56
	v_min_u32_e32 v0, s33, v0
	v_lshlrev_b64_e32 v[2:3], 1, v[57:58]
	s_wait_dscnt 0x0
	v_lshlrev_b64_e32 v[4:5], 1, v[55:56]
	v_min_u32_e32 v6, s33, v6
	v_lshlrev_b64_e32 v[0:1], 1, v[0:1]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_u32 v2, vcc_lo, s14, v2
	v_lshlrev_b64_e32 v[6:7], 1, v[6:7]
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s15, v3, vcc_lo
	v_add_co_u32 v4, vcc_lo, s14, v4
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, s15, v5, vcc_lo
	v_add_co_u32 v0, vcc_lo, s14, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s15, v1, vcc_lo
	v_add_co_u32 v6, vcc_lo, s14, v6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v7, null, s15, v7, vcc_lo
	s_clause 0x3
	global_load_u16 v74, v[2:3], off
	global_load_u16 v72, v[4:5], off
	global_load_u16 v71, v[0:1], off
	global_load_u16 v70, v[6:7], off
.LBB0_84:                               ;   in Loop: Header=BB0_82 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s22
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_91
; %bb.85:                               ;   in Loop: Header=BB0_82 Depth=1
	v_or_b32_e32 v2, 2, v57
	v_or_b32_e32 v3, 3, v57
	v_min_u32_e32 v4, s33, v57
	s_wait_dscnt 0xb
	v_mad_co_u64_u32 v[0:1], null, v55, s18, 0
	s_wait_dscnt 0x0
	v_min_u32_e32 v5, s33, v2
	v_min_u32_e32 v6, s33, v3
	v_mad_co_u64_u32 v[2:3], null, v4, s18, 0
	v_dual_mov_b32 v78, 0 :: v_dual_mov_b32 v79, v68
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mad_co_u64_u32 v[4:5], null, v5, s18, 0
	v_mad_co_u64_u32 v[6:7], null, v6, s18, 0
	v_lshlrev_b64_e32 v[58:59], 1, v[0:1]
	v_lshlrev_b64_e32 v[60:61], 1, v[2:3]
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v76, 0
	v_lshlrev_b64_e32 v[62:63], 1, v[4:5]
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v80, 0
	v_lshlrev_b64_e32 v[64:65], 1, v[6:7]
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v82, 0
	v_dual_mov_b32 v83, 0 :: v_dual_mov_b32 v86, 0
	v_dual_mov_b32 v87, 0 :: v_dual_mov_b32 v84, 0
	v_mov_b32_e32 v85, 0
	s_mov_b32 s2, 0
	s_branch .LBB0_88
.LBB0_86:                               ;   in Loop: Header=BB0_88 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s28
.LBB0_87:                               ;   in Loop: Header=BB0_88 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s27
	s_wait_loadcnt_dscnt 0x702
	v_dot2_f32_f16 v85, v52, v40, v85
	s_wait_loadcnt 0x6
	v_dot2_f32_f16 v84, v52, v24, v84
	s_wait_loadcnt 0x5
	v_dot2_f32_f16 v87, v52, v28, v87
	s_wait_loadcnt 0x4
	v_dot2_f32_f16 v52, v52, v32, v86
	s_wait_dscnt 0x1
	v_dot2_f32_f16 v82, v48, v24, v82
	v_dot2_f32_f16 v85, v53, v41, v85
	v_dot2_f32_f16 v84, v53, v25, v84
	v_dot2_f32_f16 v86, v53, v29, v87
	v_dot2_f32_f16 v52, v53, v33, v52
	v_dot2_f32_f16 v81, v48, v28, v81
	v_dot2_f32_f16 v53, v54, v42, v85
	v_dot2_f32_f16 v84, v54, v26, v84
	v_dot2_f32_f16 v85, v54, v30, v86
	v_dot2_f32_f16 v52, v54, v34, v52
	s_wait_dscnt 0x0
	v_dot2_f32_f16 v24, v44, v24, v77
	v_dot2_f32_f16 v53, v55, v43, v53
	v_dot2_f32_f16 v54, v55, v27, v84
	v_dot2_f32_f16 v86, v55, v31, v85
	v_dot2_f32_f16 v52, v55, v35, v52
	v_dot2_f32_f16 v55, v48, v40, v83
	v_dot2_f32_f16 v40, v44, v40, v78
	v_dot2_f32_f16 v48, v48, v32, v80
	v_dot2_f32_f16 v28, v44, v28, v76
	v_dot2_f32_f16 v32, v44, v32, v75
	v_dot2_f32_f16 v55, v49, v41, v55
	v_dot2_f32_f16 v40, v45, v41, v40
	v_dot2_f32_f16 v78, v49, v25, v82
	v_dot2_f32_f16 v80, v49, v29, v81
	v_dot2_f32_f16 v48, v49, v33, v48
	v_dot2_f32_f16 v41, v50, v42, v55
	v_dot2_f32_f16 v40, v46, v42, v40
	s_wait_loadcnt 0x2
	v_dot2_f32_f16 v44, v36, v4, v54
	v_dot2_f32_f16 v24, v45, v25, v24
	v_dot2_f32_f16 v25, v45, v29, v28
	v_dot2_f32_f16 v41, v51, v43, v41
	v_dot2_f32_f16 v40, v47, v43, v40
	v_dot2_f32_f16 v43, v36, v0, v53
	v_dot2_f32_f16 v28, v45, v33, v32
	v_dot2_f32_f16 v49, v50, v26, v78
	v_dot2_f32_f16 v55, v50, v30, v80
	v_dot2_f32_f16 v48, v50, v34, v48
	v_dot2_f32_f16 v29, v37, v1, v43
	v_dot2_f32_f16 v32, v37, v5, v44
	v_dot2_f32_f16 v24, v46, v26, v24
	v_dot2_f32_f16 v25, v46, v30, v25
	v_dot2_f32_f16 v26, v46, v34, v28
	v_dot2_f32_f16 v42, v51, v27, v49
	v_dot2_f32_f16 v49, v51, v31, v55
	v_dot2_f32_f16 v48, v51, v35, v48
	v_dot2_f32_f16 v28, v38, v2, v29
	v_dot2_f32_f16 v29, v38, v6, v32
	v_dot2_f32_f16 v24, v47, v27, v24
	v_dot2_f32_f16 v25, v47, v31, v25
	v_dot2_f32_f16 v26, v47, v35, v26
	v_dot2_f32_f16 v85, v39, v3, v28
	v_dot2_f32_f16 v84, v39, v7, v29
	s_wait_loadcnt 0x1
	v_dot2_f32_f16 v27, v36, v8, v86
	s_wait_loadcnt 0x0
	v_dot2_f32_f16 v28, v36, v12, v52
	v_dot2_f32_f16 v29, v20, v0, v41
	v_dot2_f32_f16 v30, v20, v4, v42
	v_dot2_f32_f16 v31, v20, v8, v49
	v_dot2_f32_f16 v20, v20, v12, v48
	v_dot2_f32_f16 v0, v16, v0, v40
	v_dot2_f32_f16 v4, v16, v4, v24
	v_dot2_f32_f16 v8, v16, v8, v25
	v_dot2_f32_f16 v12, v16, v12, v26
	v_dot2_f32_f16 v27, v37, v9, v27
	v_dot2_f32_f16 v28, v37, v13, v28
	v_dot2_f32_f16 v29, v21, v1, v29
	v_dot2_f32_f16 v30, v21, v5, v30
	v_dot2_f32_f16 v31, v21, v9, v31
	v_dot2_f32_f16 v16, v21, v13, v20
	v_dot2_f32_f16 v0, v17, v1, v0
	v_dot2_f32_f16 v1, v17, v5, v4
	v_dot2_f32_f16 v4, v17, v9, v8
	v_dot2_f32_f16 v5, v17, v13, v12
	v_dot2_f32_f16 v27, v38, v10, v27
	v_dot2_f32_f16 v28, v38, v14, v28
	v_dot2_f32_f16 v29, v22, v2, v29
	v_dot2_f32_f16 v30, v22, v6, v30
	v_dot2_f32_f16 v31, v22, v10, v31
	v_dot2_f32_f16 v8, v22, v14, v16
	v_dot2_f32_f16 v0, v18, v2, v0
	v_dot2_f32_f16 v1, v18, v6, v1
	v_dot2_f32_f16 v2, v18, v10, v4
	v_dot2_f32_f16 v4, v18, v14, v5
	v_dot2_f32_f16 v87, v39, v11, v27
	v_dot2_f32_f16 v86, v39, v15, v28
	v_dot2_f32_f16 v83, v23, v3, v29
	v_dot2_f32_f16 v82, v23, v7, v30
	v_dot2_f32_f16 v81, v23, v11, v31
	v_dot2_f32_f16 v80, v23, v15, v8
	v_dot2_f32_f16 v78, v19, v3, v0
	v_dot2_f32_f16 v77, v19, v7, v1
	v_dot2_f32_f16 v76, v19, v11, v2
	v_dot2_f32_f16 v75, v19, v15, v4
	v_add_nc_u32_e32 v79, 0x400, v79
	s_addk_co_i32 s2, 0x200
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s2, s12
	s_cbranch_scc1 .LBB0_92
.LBB0_88:                               ;   Parent Loop BB0_82 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_dual_mov_b32 v19, 0 :: v_dual_add_nc_u32 v16, s2, v66
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v23, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v55, s23, v16
	v_dual_mov_b32 v17, 0 :: v_dual_add_nc_u32 v88, 0x100, v16
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v21, 0
	v_lshlrev_b64_e32 v[0:1], 1, v[55:56]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_min_u32_e32 v55, s23, v88
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v51, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v49, 0
	v_add_co_u32 v8, vcc_lo, s16, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, s17, v1, vcc_lo
	v_lshlrev_b64_e32 v[0:1], 1, v[55:56]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v2, vcc_lo, v8, v60
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, v9, v61, vcc_lo
	v_add_co_u32 v4, vcc_lo, v8, v58
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v9, v59, vcc_lo
	v_add_co_u32 v6, vcc_lo, v8, v62
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v7, null, v9, v63, vcc_lo
	v_add_co_u32 v10, vcc_lo, s16, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v11, null, s17, v1, vcc_lo
	v_add_co_u32 v0, vcc_lo, v8, v64
	s_clause 0x1
	global_load_b128 v[40:43], v[2:3], off
	global_load_b128 v[24:27], v[4:5], off
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v9, v65, vcc_lo
	v_add_co_u32 v2, vcc_lo, v10, v60
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, v11, v61, vcc_lo
	v_add_co_u32 v4, vcc_lo, v10, v58
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v11, v59, vcc_lo
	v_add_co_u32 v8, vcc_lo, v10, v62
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, v11, v63, vcc_lo
	v_add_co_u32 v12, vcc_lo, v10, v64
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v13, null, v11, v65, vcc_lo
	s_clause 0x5
	global_load_b128 v[28:31], v[6:7], off
	global_load_b128 v[32:35], v[0:1], off
	global_load_b128 v[0:3], v[2:3], off
	global_load_b128 v[4:7], v[4:5], off
	global_load_b128 v[8:11], v[8:9], off
	global_load_b128 v[12:15], v[12:13], off
	v_cmp_gt_u32_e32 vcc_lo, s12, v16
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v39, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v53, 0
	v_mov_b32_e32 v54, 0
	v_mov_b32_e32 v52, 0
	s_and_saveexec_b32 s27, vcc_lo
	s_cbranch_execz .LBB0_87
; %bb.89:                               ;   in Loop: Header=BB0_88 Depth=2
	v_dual_mov_b32 v37, 0 :: v_dual_add_nc_u32 v90, s25, v79
	v_dual_mov_b32 v36, 0 :: v_dual_add_nc_u32 v89, s26, v79
	ds_load_b128 v[52:55], v79
	ds_load_b128 v[48:51], v90
	ds_load_b128 v[44:47], v89
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v39, 0
	v_dual_mov_b32 v20, 0 :: v_dual_mov_b32 v21, 0
	v_dual_mov_b32 v22, 0 :: v_dual_mov_b32 v23, 0
	v_dual_mov_b32 v16, 0 :: v_dual_mov_b32 v17, 0
	v_dual_mov_b32 v18, 0 :: v_dual_mov_b32 v19, 0
	s_mov_b32 s28, exec_lo
	v_cmpx_gt_u32_e64 s12, v88
	s_cbranch_execz .LBB0_86
; %bb.90:                               ;   in Loop: Header=BB0_88 Depth=2
	ds_load_b128 v[36:39], v79 offset:512
	ds_load_b128 v[20:23], v90 offset:512
	ds_load_b128 v[16:19], v89 offset:512
	s_branch .LBB0_86
.LBB0_91:                               ;   in Loop: Header=BB0_82 Depth=1
	v_dual_mov_b32 v85, 0 :: v_dual_mov_b32 v84, 0
	v_dual_mov_b32 v87, 0 :: v_dual_mov_b32 v86, 0
	v_dual_mov_b32 v83, 0 :: v_dual_mov_b32 v82, 0
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v80, 0
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v76, 0
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v78, 0
.LBB0_92:                               ;   in Loop: Header=BB0_82 Depth=1
	v_add_f32_dpp v3, v87, v87 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v86, v86 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x0
	v_add_f32_dpp v5, v83, v83 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v82, v82 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v81, v81 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
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
	v_add_f32_dpp v0, v85, v85 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v84, v84 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v23, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v21, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v19, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v12, v7, v7 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v80, v80 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v78, v78 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v77, v77 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v76, v76 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v75, v75 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v69
	v_add_f32_dpp v0, v0, v0 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v2, v67, v69, vcc_lo
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
	v_add_f32_dpp v14, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
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
	ds_bpermute_b32 v13, v2, v12
	ds_bpermute_b32 v16, v2, v14
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
	s_and_saveexec_b32 s27, s1
	s_cbranch_execz .LBB0_81
; %bb.93:                               ;   in Loop: Header=BB0_82 Depth=1
	s_wait_dscnt 0xb
	v_add_f32_e32 v15, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v58, v56 :: v_dual_mul_f32 v27, s8, v15
	v_lshlrev_b64_e32 v[2:3], 1, v[57:58]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s4, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s5, v3, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_95
; %bb.94:                               ;   in Loop: Header=BB0_82 Depth=1
	global_load_u16 v15, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v27, s9, v15, v27 op_sel_hi:[0,1,0]
.LBB0_95:                               ;   in Loop: Header=BB0_82 Depth=1
	v_cvt_f32_f16_e32 v15, v74
	v_add_co_u32 v2, vcc_lo, s6, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v28, v27, v15
	v_cmp_gt_u32_e64 s2, s13, v73
	v_cndmask_b32_e64 v27, v28, v27, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v27, v27
	global_store_b16 v[2:3], v27, off
	s_and_saveexec_b32 s28, s2
	s_cbranch_execz .LBB0_105
; %bb.96:                               ;   in Loop: Header=BB0_82 Depth=1
	s_wait_dscnt 0xa
	v_add_f32_e32 v25, v25, v26
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v25, s8, v25
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_98
; %bb.97:                               ;   in Loop: Header=BB0_82 Depth=1
	global_load_u16 v26, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v25, s9, v26, v25 op_sel_hi:[0,1,0]
.LBB0_98:                               ;   in Loop: Header=BB0_82 Depth=1
	v_cvt_f32_f16_e32 v26, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v26, v25, v26
	v_cndmask_b32_e64 v25, v26, v25, s0
	v_or_b32_e32 v26, 2, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v25, v25
	v_cmp_gt_u32_e32 vcc_lo, s13, v26
	global_store_b16 v[2:3], v25, off offset:2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_105
; %bb.99:                               ;   in Loop: Header=BB0_82 Depth=1
	s_wait_dscnt 0x9
	v_add_f32_e32 v23, v23, v24
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v23, s8, v23
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_101
; %bb.100:                              ;   in Loop: Header=BB0_82 Depth=1
	global_load_u16 v24, v[0:1], off offset:4
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v23, s9, v24, v23 op_sel_hi:[0,1,0]
.LBB0_101:                              ;   in Loop: Header=BB0_82 Depth=1
	v_cvt_f32_f16_e32 v24, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v24, v23, v24
	v_cndmask_b32_e64 v23, v24, v23, s0
	v_or_b32_e32 v24, 3, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v23, v23
	v_cmp_gt_u32_e32 vcc_lo, s13, v24
	global_store_b16 v[2:3], v23, off offset:4
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_105
; %bb.102:                              ;   in Loop: Header=BB0_82 Depth=1
	s_wait_dscnt 0x8
	v_add_f32_e32 v21, v21, v22
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v21, s8, v21
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_104
; %bb.103:                              ;   in Loop: Header=BB0_82 Depth=1
	global_load_u16 v22, v[0:1], off offset:6
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v21, s9, v22, v21 op_sel_hi:[0,1,0]
.LBB0_104:                              ;   in Loop: Header=BB0_82 Depth=1
	v_cvt_f32_f16_e32 v22, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v22, v21, v22
	v_cndmask_b32_e64 v21, v22, v21, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v21, v21
	global_store_b16 v[2:3], v21, off offset:6
.LBB0_105:                              ;   in Loop: Header=BB0_82 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s28
	s_wait_dscnt 0x7
	v_add_f32_e32 v19, v19, v20
	v_add_co_u32 v0, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v19, s8, v19
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_107
; %bb.106:                              ;   in Loop: Header=BB0_82 Depth=1
	global_load_u16 v20, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v19, s9, v20, v19 op_sel_hi:[0,1,0]
.LBB0_107:                              ;   in Loop: Header=BB0_82 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v20, v19, v15
	v_add_co_u32 v2, vcc_lo, v2, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	v_cndmask_b32_e64 v19, v20, v19, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v19, v19
	global_store_b16 v[2:3], v19, off
	s_and_saveexec_b32 s28, s2
	s_cbranch_execz .LBB0_117
; %bb.108:                              ;   in Loop: Header=BB0_82 Depth=1
	s_wait_dscnt 0x6
	v_add_f32_e32 v17, v17, v18
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v17, s8, v17
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_110
; %bb.109:                              ;   in Loop: Header=BB0_82 Depth=1
	global_load_u16 v18, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v17, s9, v18, v17 op_sel_hi:[0,1,0]
.LBB0_110:                              ;   in Loop: Header=BB0_82 Depth=1
	v_cvt_f32_f16_e32 v18, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v18, v17, v18
	v_cndmask_b32_e64 v17, v18, v17, s0
	v_or_b32_e32 v18, 2, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v17, v17
	v_cmp_gt_u32_e32 vcc_lo, s13, v18
	global_store_b16 v[2:3], v17, off offset:2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_117
; %bb.111:                              ;   in Loop: Header=BB0_82 Depth=1
	s_wait_dscnt 0x5
	v_add_f32_e32 v12, v12, v13
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v12, s8, v12
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_113
; %bb.112:                              ;   in Loop: Header=BB0_82 Depth=1
	global_load_u16 v13, v[0:1], off offset:4
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v12, s9, v13, v12 op_sel_hi:[0,1,0]
.LBB0_113:                              ;   in Loop: Header=BB0_82 Depth=1
	v_cvt_f32_f16_e32 v13, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v13, v12, v13
	v_cndmask_b32_e64 v12, v13, v12, s0
	v_or_b32_e32 v13, 3, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v12, v12
	v_cmp_gt_u32_e32 vcc_lo, s13, v13
	global_store_b16 v[2:3], v12, off offset:4
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_117
; %bb.114:                              ;   in Loop: Header=BB0_82 Depth=1
	s_wait_dscnt 0x4
	v_add_f32_e32 v12, v14, v16
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v12, s8, v12
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_116
; %bb.115:                              ;   in Loop: Header=BB0_82 Depth=1
	global_load_u16 v13, v[0:1], off offset:6
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v12, s9, v13, v12 op_sel_hi:[0,1,0]
.LBB0_116:                              ;   in Loop: Header=BB0_82 Depth=1
	v_cvt_f32_f16_e32 v13, v70
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v13, v12, v13
	v_cndmask_b32_e64 v12, v13, v12, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v12, v12
	global_store_b16 v[2:3], v12, off offset:6
.LBB0_117:                              ;   in Loop: Header=BB0_82 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s28
	s_wait_dscnt 0x3
	v_add_f32_e32 v10, v10, v11
	v_add_co_u32 v0, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v10, s8, v10
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_119
; %bb.118:                              ;   in Loop: Header=BB0_82 Depth=1
	global_load_u16 v11, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v10, s9, v11, v10 op_sel_hi:[0,1,0]
.LBB0_119:                              ;   in Loop: Header=BB0_82 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v11, v10, v15
	v_add_co_u32 v2, vcc_lo, v2, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	v_cndmask_b32_e64 v10, v11, v10, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v10, v10
	global_store_b16 v[2:3], v10, off
	s_and_b32 exec_lo, exec_lo, s2
	s_cbranch_execz .LBB0_81
; %bb.120:                              ;   in Loop: Header=BB0_82 Depth=1
	s_wait_dscnt 0x2
	v_add_f32_e32 v8, v8, v9
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v8, s8, v8
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_122
; %bb.121:                              ;   in Loop: Header=BB0_82 Depth=1
	global_load_u16 v9, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v8, s9, v9, v8 op_sel_hi:[0,1,0]
.LBB0_122:                              ;   in Loop: Header=BB0_82 Depth=1
	v_cvt_f32_f16_e32 v9, v72
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v9, v8, v9
	v_cndmask_b32_e64 v8, v9, v8, s0
	v_or_b32_e32 v9, 2, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v8, v8
	v_cmp_gt_u32_e32 vcc_lo, s13, v9
	global_store_b16 v[2:3], v8, off offset:2
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_81
; %bb.123:                              ;   in Loop: Header=BB0_82 Depth=1
	s_wait_dscnt 0x1
	v_add_f32_e32 v6, v6, v7
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v6, s8, v6
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_125
; %bb.124:                              ;   in Loop: Header=BB0_82 Depth=1
	global_load_u16 v7, v[0:1], off offset:4
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v6, s9, v7, v6 op_sel_hi:[0,1,0]
.LBB0_125:                              ;   in Loop: Header=BB0_82 Depth=1
	v_cvt_f32_f16_e32 v7, v71
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v7, v6, v7
	v_cndmask_b32_e64 v6, v7, v6, s0
	v_or_b32_e32 v7, 3, v57
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_cvt_f16_f32_e32 v6, v6
	v_cmp_gt_u32_e32 vcc_lo, s13, v7
	global_store_b16 v[2:3], v6, off offset:4
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_81
; %bb.126:                              ;   in Loop: Header=BB0_82 Depth=1
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v4, s8, v4
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_80
; %bb.127:                              ;   in Loop: Header=BB0_82 Depth=1
	global_load_u16 v0, v[0:1], off offset:6
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v4, s9, v0, v4 op_sel_hi:[0,1,0]
	s_branch .LBB0_80
.LBB0_128:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201
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
		.amdhsa_inst_pref_size 76
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
	.size	wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201, .Lfunc_end0-wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201
                                        ; -- End function
	.set wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201.num_vgpr, 97
	.set wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201.num_agpr, 0
	.set wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201.numbered_sgpr, 46
	.set wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201.num_named_barrier, 0
	.set wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201.private_seg_size, 0
	.set wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201.uses_vcc, 1
	.set wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201.uses_flat_scratch, 0
	.set wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201.has_dyn_sized_stack, 0
	.set wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201.has_recursion, 0
	.set wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 9664
; TotalNumSgprs: 48
; NumVgprs: 97
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
	.type	__hip_cuid_5161f6583aa5b6a1,@object ; @__hip_cuid_5161f6583aa5b6a1
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_5161f6583aa5b6a1
__hip_cuid_5161f6583aa5b6a1:
	.byte	0                               ; 0x0
	.size	__hip_cuid_5161f6583aa5b6a1, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_5161f6583aa5b6a1
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
    .name:           wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     48
    .sgpr_spill_count: 0
    .symbol:         wvSplitK_vllm_f16_tn_n3_y4u2_gfx1201.kd
    .vgpr_count:     97
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
