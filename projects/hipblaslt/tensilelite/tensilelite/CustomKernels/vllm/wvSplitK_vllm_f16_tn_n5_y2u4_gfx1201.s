	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	.amdhsa_code_object_version 4
	.text
	.protected	wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201 ; -- Begin function wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201
	.globl	wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201
	.p2align	8
	.type	wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201,@function
wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201:   ; @wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201
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
	v_bfe_u32 v121, v0, 10, 10
	s_mul_i32 s0, s35, ttmp9
	s_mul_i32 s34, s35, s3
	s_cmp_gt_u32 s12, 0x1999
	s_delay_alu instid0(VALU_DEP_1)
	v_add_lshl_u32 v113, s0, v121, 1
	s_mov_b32 s0, -1
	v_cmp_gt_u32_e32 vcc_lo, s35, v121
	s_cbranch_scc0 .LBB0_69
; %bb.5:
	s_and_saveexec_b32 s3, vcc_lo
	s_cbranch_execz .LBB0_68
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
	v_cmp_gt_u32_e32 vcc_lo, s36, v113
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_68
; %bb.7:
	v_dual_mov_b32 v116, v113 :: v_dual_and_b32 v1, 0x3ff, v0
	s_wait_kmcnt 0x0
	s_cmp_lg_u64 s[14:15], 0
	v_mbcnt_lo_u32_b32 v129, -1, 0
	s_cselect_b32 s0, -1, 0
	v_dual_mov_b32 v115, 0 :: v_dual_lshlrev_b32 v122, 3, v1
	v_cmp_eq_u32_e64 s1, 31, v1
	v_lshlrev_b32_e32 v1, 4, v1
	s_lshl_b32 s37, s35, 8
	s_add_co_i32 s38, s12, -8
	s_cmp_neq_f32 s9, 0
	v_lshl_add_u32 v123, v121, 8, v122
	v_lshl_add_u32 v124, v121, 9, v1
	s_mov_b32 s22, s19
	s_mov_b32 s26, s10
	s_mov_b32 s27, s23
	s_mov_b32 s28, s11
	v_add_nc_u32_e32 v125, 0x3000, v124
	v_add_nc_u32_e32 v126, 0x6000, v124
	v_add_nc_u32_e32 v127, 0x9000, v124
	v_add_nc_u32_e32 v128, 0xc000, v124
	s_mov_b32 s29, s23
	s_cselect_b32 s39, -1, 0
	s_lshl_b64 s[24:25], s[22:23], 1
	s_lshl_b64 s[26:27], s[26:27], 1
	s_lshl_b64 s[28:29], s[28:29], 1
	s_lshl_b32 s40, s35, 9
	s_mov_b32 s41, 0
	s_branch .LBB0_10
.LBB0_8:                                ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e64 v1, v130
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v1, v5, v1
	v_cndmask_b32_e64 v1, v5, v1, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v1, v1
	global_store_b16 v[3:4], v1, off offset:2
.LBB0_9:                                ;   in Loop: Header=BB0_10 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	v_add_nc_u32_e32 v116, s34, v116
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s36, v116
	s_or_b32 s41, vcc_lo, s41
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s41
	s_cbranch_execz .LBB0_68
.LBB0_10:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_18 Depth 2
                                        ;       Child Loop BB0_22 Depth 3
                                        ;       Child Loop BB0_24 Depth 3
                                        ;       Child Loop BB0_26 Depth 3
                                        ;       Child Loop BB0_28 Depth 3
                                        ;       Child Loop BB0_30 Depth 3
	v_or_b32_e32 v131, 1, v116
	v_cmp_gt_u32_e64 s2, s13, v116
	s_wait_loadcnt 0x1
	v_mov_b32_e32 v132, v115
	s_wait_loadcnt 0x0
	v_mov_b32_e32 v130, v115
	v_min_u32_e32 v114, s33, v131
	s_wait_alu 0xfffe
	s_and_b32 s30, s0, s2
	s_wait_alu 0xfffe
	s_and_saveexec_b32 s22, s30
	s_cbranch_execz .LBB0_12
; %bb.11:                               ;   in Loop: Header=BB0_10 Depth=1
	v_mov_b32_e32 v117, v115
	v_lshlrev_b64_e32 v[3:4], 1, v[114:115]
	s_wait_dscnt 0x9
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[1:2], 1, v[116:117]
	v_add_co_u32 v1, vcc_lo, s14, v1
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v2, null, s15, v2, vcc_lo
	v_add_co_u32 v3, vcc_lo, s14, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s15, v4, vcc_lo
	s_clause 0x1
	global_load_u16 v132, v[1:2], off
	global_load_u16 v130, v[3:4], off
.LBB0_12:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	v_min_u32_e32 v3, s33, v116
	s_wait_dscnt 0x9
	v_mad_co_u64_u32 v[1:2], null, v114, s18, 0
	v_dual_mov_b32 v136, 0 :: v_dual_mov_b32 v135, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mad_co_u64_u32 v[3:4], null, v3, s18, 0
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v133, 0
	v_lshlrev_b64_e32 v[117:118], 1, v[1:2]
	v_dual_mov_b32 v137, 0 :: v_dual_mov_b32 v138, 0
	v_dual_mov_b32 v139, 0 :: v_dual_mov_b32 v140, 0
	v_lshlrev_b64_e32 v[119:120], 1, v[3:4]
	v_dual_mov_b32 v141, 0 :: v_dual_mov_b32 v142, 0
	s_mov_b32 s22, 0
	s_wait_alu 0xfffe
	s_mov_b32 s42, s22
	s_branch .LBB0_18
.LBB0_13:                               ;   in Loop: Header=BB0_18 Depth=2
	s_or_b32 exec_lo, exec_lo, s45
.LBB0_14:                               ;   in Loop: Header=BB0_18 Depth=2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_or_b32 exec_lo, exec_lo, s44
.LBB0_15:                               ;   in Loop: Header=BB0_18 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s43
.LBB0_16:                               ;   in Loop: Header=BB0_18 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s31
	s_wait_loadcnt_dscnt 0x704
	v_dot2_f32_f16 v114, v109, v29, v142
	s_wait_loadcnt 0x6
	v_dot2_f32_f16 v109, v109, v33, v141
	s_wait_dscnt 0x3
	v_dot2_f32_f16 v140, v105, v29, v140
	v_dot2_f32_f16 v105, v105, v33, v139
	s_wait_dscnt 0x2
	v_dot2_f32_f16 v138, v101, v29, v138
	v_dot2_f32_f16 v114, v110, v30, v114
	v_dot2_f32_f16 v109, v110, v34, v109
	v_dot2_f32_f16 v110, v106, v30, v140
	v_dot2_f32_f16 v105, v106, v34, v105
	v_dot2_f32_f16 v101, v101, v33, v137
	v_dot2_f32_f16 v106, v102, v30, v138
	v_dot2_f32_f16 v114, v111, v31, v114
	v_dot2_f32_f16 v110, v107, v31, v110
	v_dot2_f32_f16 v105, v107, v35, v105
	v_dot2_f32_f16 v109, v111, v35, v109
	v_dot2_f32_f16 v106, v103, v31, v106
	v_dot2_f32_f16 v107, v112, v32, v114
	v_dot2_f32_f16 v110, v108, v32, v110
	v_dot2_f32_f16 v105, v108, v36, v105
	s_wait_dscnt 0x1
	v_dot2_f32_f16 v108, v89, v29, v136
	s_wait_dscnt 0x0
	v_dot2_f32_f16 v29, v85, v29, v134
	v_dot2_f32_f16 v89, v89, v33, v135
	v_dot2_f32_f16 v33, v85, v33, v133
	v_dot2_f32_f16 v85, v102, v34, v101
	v_dot2_f32_f16 v101, v90, v30, v108
	v_dot2_f32_f16 v29, v86, v30, v29
	v_dot2_f32_f16 v89, v90, v34, v89
	v_dot2_f32_f16 v30, v86, v34, v33
	v_dot2_f32_f16 v33, v103, v35, v85
	v_dot2_f32_f16 v34, v91, v31, v101
	v_dot2_f32_f16 v29, v87, v31, v29
	v_dot2_f32_f16 v106, v104, v32, v106
	v_dot2_f32_f16 v85, v91, v35, v89
	v_dot2_f32_f16 v30, v87, v35, v30
	v_dot2_f32_f16 v31, v104, v36, v33
	v_dot2_f32_f16 v33, v92, v32, v34
	v_dot2_f32_f16 v29, v88, v32, v29
	s_wait_loadcnt 0x5
	v_dot2_f32_f16 v32, v93, v17, v107
	v_dot2_f32_f16 v109, v112, v36, v109
	v_dot2_f32_f16 v34, v92, v36, v85
	v_dot2_f32_f16 v30, v88, v36, v30
	v_dot2_f32_f16 v36, v97, v17, v110
	v_dot2_f32_f16 v86, v81, v17, v106
	v_dot2_f32_f16 v32, v94, v18, v32
	s_wait_loadcnt 0x4
	v_dot2_f32_f16 v31, v81, v21, v31
	v_dot2_f32_f16 v33, v77, v17, v33
	v_dot2_f32_f16 v17, v73, v17, v29
	v_dot2_f32_f16 v35, v93, v21, v109
	v_dot2_f32_f16 v85, v97, v21, v105
	v_dot2_f32_f16 v36, v98, v18, v36
	v_dot2_f32_f16 v86, v82, v18, v86
	v_dot2_f32_f16 v32, v95, v19, v32
	v_dot2_f32_f16 v34, v77, v21, v34
	v_dot2_f32_f16 v21, v73, v21, v30
	v_dot2_f32_f16 v29, v82, v22, v31
	v_dot2_f32_f16 v30, v78, v18, v33
	v_dot2_f32_f16 v17, v74, v18, v17
	v_dot2_f32_f16 v35, v94, v22, v35
	v_dot2_f32_f16 v85, v98, v22, v85
	v_dot2_f32_f16 v36, v99, v19, v36
	v_dot2_f32_f16 v86, v83, v19, v86
	v_dot2_f32_f16 v32, v96, v20, v32
	v_dot2_f32_f16 v31, v78, v22, v34
	v_dot2_f32_f16 v18, v74, v22, v21
	v_dot2_f32_f16 v21, v83, v23, v29
	v_dot2_f32_f16 v22, v79, v19, v30
	v_dot2_f32_f16 v17, v75, v19, v17
	v_dot2_f32_f16 v35, v95, v23, v35
	v_dot2_f32_f16 v85, v99, v23, v85
	v_dot2_f32_f16 v36, v100, v20, v36
	v_dot2_f32_f16 v86, v84, v20, v86
	v_dot2_f32_f16 v29, v79, v23, v31
	v_dot2_f32_f16 v18, v75, v23, v18
	v_dot2_f32_f16 v19, v84, v24, v21
	v_dot2_f32_f16 v21, v80, v20, v22
	v_dot2_f32_f16 v17, v76, v20, v17
	s_wait_loadcnt 0x3
	v_dot2_f32_f16 v20, v65, v9, v32
	v_dot2_f32_f16 v35, v96, v24, v35
	v_dot2_f32_f16 v85, v100, v24, v85
	v_dot2_f32_f16 v22, v80, v24, v29
	v_dot2_f32_f16 v18, v76, v24, v18
	v_dot2_f32_f16 v24, v69, v9, v36
	v_dot2_f32_f16 v30, v53, v9, v86
	v_dot2_f32_f16 v20, v66, v10, v20
	s_wait_loadcnt 0x2
	v_dot2_f32_f16 v19, v53, v13, v19
	v_dot2_f32_f16 v21, v57, v9, v21
	v_dot2_f32_f16 v9, v61, v9, v17
	v_dot2_f32_f16 v23, v65, v13, v35
	v_dot2_f32_f16 v29, v69, v13, v85
	v_dot2_f32_f16 v24, v70, v10, v24
	v_dot2_f32_f16 v30, v54, v10, v30
	v_dot2_f32_f16 v20, v67, v11, v20
	v_dot2_f32_f16 v22, v57, v13, v22
	v_dot2_f32_f16 v13, v61, v13, v18
	v_dot2_f32_f16 v17, v54, v14, v19
	v_dot2_f32_f16 v18, v58, v10, v21
	v_dot2_f32_f16 v9, v62, v10, v9
	v_dot2_f32_f16 v23, v66, v14, v23
	v_dot2_f32_f16 v29, v70, v14, v29
	v_dot2_f32_f16 v24, v71, v11, v24
	v_dot2_f32_f16 v30, v55, v11, v30
	v_dot2_f32_f16 v20, v68, v12, v20
	v_dot2_f32_f16 v19, v58, v14, v22
	v_dot2_f32_f16 v10, v62, v14, v13
	v_dot2_f32_f16 v13, v55, v15, v17
	v_dot2_f32_f16 v14, v59, v11, v18
	v_dot2_f32_f16 v9, v63, v11, v9
	v_dot2_f32_f16 v24, v72, v12, v24
	v_dot2_f32_f16 v30, v56, v12, v30
	v_dot2_f32_f16 v11, v56, v16, v13
	v_dot2_f32_f16 v13, v60, v12, v14
	v_dot2_f32_f16 v9, v64, v12, v9
	s_wait_loadcnt 0x1
	v_dot2_f32_f16 v12, v37, v1, v20
	v_dot2_f32_f16 v23, v67, v15, v23
	v_dot2_f32_f16 v29, v71, v15, v29
	v_dot2_f32_f16 v17, v59, v15, v19
	v_dot2_f32_f16 v10, v63, v15, v10
	v_dot2_f32_f16 v12, v38, v2, v12
	v_dot2_f32_f16 v23, v68, v16, v23
	v_dot2_f32_f16 v29, v72, v16, v29
	v_dot2_f32_f16 v14, v60, v16, v17
	v_dot2_f32_f16 v10, v64, v16, v10
	v_dot2_f32_f16 v12, v39, v3, v12
	s_wait_loadcnt 0x0
	v_dot2_f32_f16 v15, v37, v5, v23
	v_dot2_f32_f16 v16, v45, v1, v24
	v_dot2_f32_f16 v17, v45, v5, v29
	v_dot2_f32_f16 v18, v25, v1, v30
	v_dot2_f32_f16 v142, v40, v4, v12
	v_dot2_f32_f16 v11, v25, v5, v11
	v_dot2_f32_f16 v12, v41, v1, v13
	v_dot2_f32_f16 v13, v41, v5, v14
	v_dot2_f32_f16 v1, v49, v1, v9
	v_dot2_f32_f16 v5, v49, v5, v10
	v_dot2_f32_f16 v15, v38, v6, v15
	v_dot2_f32_f16 v16, v46, v2, v16
	v_dot2_f32_f16 v17, v46, v6, v17
	v_dot2_f32_f16 v18, v26, v2, v18
	v_dot2_f32_f16 v9, v26, v6, v11
	v_dot2_f32_f16 v10, v42, v2, v12
	v_dot2_f32_f16 v11, v42, v6, v13
	v_dot2_f32_f16 v1, v50, v2, v1
	v_dot2_f32_f16 v2, v50, v6, v5
	v_dot2_f32_f16 v15, v39, v7, v15
	v_dot2_f32_f16 v16, v47, v3, v16
	v_dot2_f32_f16 v17, v47, v7, v17
	v_dot2_f32_f16 v18, v27, v3, v18
	v_dot2_f32_f16 v5, v27, v7, v9
	v_dot2_f32_f16 v6, v43, v3, v10
	v_dot2_f32_f16 v9, v43, v7, v11
	v_dot2_f32_f16 v1, v51, v3, v1
	v_dot2_f32_f16 v2, v51, v7, v2
	v_dot2_f32_f16 v141, v40, v8, v15
	v_dot2_f32_f16 v140, v48, v4, v16
	v_dot2_f32_f16 v139, v48, v8, v17
	v_dot2_f32_f16 v138, v28, v4, v18
	v_dot2_f32_f16 v137, v28, v8, v5
	v_dot2_f32_f16 v136, v44, v4, v6
	v_dot2_f32_f16 v135, v44, v8, v9
	v_dot2_f32_f16 v134, v52, v4, v1
	v_dot2_f32_f16 v133, v52, v8, v2
.LBB0_17:                               ;   in Loop: Header=BB0_18 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s30
	s_addk_co_i32 s42, 0x400
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s42, s12
	s_cbranch_scc1 .LBB0_37
.LBB0_18:                               ;   Parent Loop BB0_10 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_22 Depth 3
                                        ;       Child Loop BB0_24 Depth 3
                                        ;       Child Loop BB0_26 Depth 3
                                        ;       Child Loop BB0_28 Depth 3
                                        ;       Child Loop BB0_30 Depth 3
	s_wait_alu 0xfffe
	s_cmp_eq_u32 s42, 0
	s_cselect_b32 s31, -1, 0
	s_add_co_i32 s30, s22, 0x1800
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
	s_branch .LBB0_32
.LBB0_20:                               ;   in Loop: Header=BB0_18 Depth=2
	s_and_b32 s31, s31, exec_lo
	s_cselect_b32 s22, s22, s30
	s_mov_b32 s44, exec_lo
	s_wait_alu 0xfffe
	s_sub_co_i32 s30, s12, s22
	s_wait_loadcnt 0x0
	s_wait_alu 0xfffe
	s_min_u32 s43, s30, 0x1800
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_wait_alu 0xfffe
	v_cmpx_gt_u32_e64 s43, v123
	s_cbranch_execz .LBB0_31
; %bb.21:                               ;   in Loop: Header=BB0_18 Depth=2
	v_dual_mov_b32 v1, v124 :: v_dual_mov_b32 v114, v123
	s_lshl_b64 s[30:31], s[22:23], 1
	s_mov_b32 s45, 0
	s_wait_alu 0xfffe
	s_add_nc_u64 s[30:31], s[20:21], s[30:31]
.LBB0_22:                               ;   Parent Loop BB0_10 Depth=1
                                        ;     Parent Loop BB0_18 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	v_lshlrev_b64_e32 v[2:3], 1, v[114:115]
	v_add_nc_u32_e32 v114, s37, v114
	s_wait_alu 0xfffe
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v2, vcc_lo, s30, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s31, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s43, v114
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s45, vcc_lo, s45
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s40, v1
	s_and_not1_b32 exec_lo, exec_lo, s45
	s_cbranch_execnz .LBB0_22
; %bb.23:                               ;   in Loop: Header=BB0_18 Depth=2
	s_or_b32 exec_lo, exec_lo, s45
	v_dual_mov_b32 v1, v125 :: v_dual_mov_b32 v114, v123
	s_add_nc_u64 s[30:31], s[30:31], s[24:25]
	s_mov_b32 s45, 0
.LBB0_24:                               ;   Parent Loop BB0_10 Depth=1
                                        ;     Parent Loop BB0_18 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 1, v[114:115]
	v_add_nc_u32_e32 v114, s37, v114
	s_wait_alu 0xfffe
	v_add_co_u32 v2, vcc_lo, s30, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, s31, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s43, v114
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s45, vcc_lo, s45
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s40, v1
	s_and_not1_b32 exec_lo, exec_lo, s45
	s_cbranch_execnz .LBB0_24
; %bb.25:                               ;   in Loop: Header=BB0_18 Depth=2
	s_or_b32 exec_lo, exec_lo, s45
	v_dual_mov_b32 v1, v126 :: v_dual_mov_b32 v114, v123
	s_add_nc_u64 s[30:31], s[30:31], s[24:25]
	s_mov_b32 s45, 0
.LBB0_26:                               ;   Parent Loop BB0_10 Depth=1
                                        ;     Parent Loop BB0_18 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 1, v[114:115]
	v_add_nc_u32_e32 v114, s37, v114
	s_wait_alu 0xfffe
	v_add_co_u32 v2, vcc_lo, s30, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, s31, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s43, v114
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s45, vcc_lo, s45
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s40, v1
	s_and_not1_b32 exec_lo, exec_lo, s45
	s_cbranch_execnz .LBB0_26
; %bb.27:                               ;   in Loop: Header=BB0_18 Depth=2
	s_or_b32 exec_lo, exec_lo, s45
	v_dual_mov_b32 v1, v127 :: v_dual_mov_b32 v114, v123
	s_add_nc_u64 s[30:31], s[30:31], s[24:25]
	s_mov_b32 s45, 0
.LBB0_28:                               ;   Parent Loop BB0_10 Depth=1
                                        ;     Parent Loop BB0_18 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 1, v[114:115]
	v_add_nc_u32_e32 v114, s37, v114
	s_wait_alu 0xfffe
	v_add_co_u32 v2, vcc_lo, s30, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, s31, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s43, v114
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s45, vcc_lo, s45
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s40, v1
	s_and_not1_b32 exec_lo, exec_lo, s45
	s_cbranch_execnz .LBB0_28
; %bb.29:                               ;   in Loop: Header=BB0_18 Depth=2
	s_or_b32 exec_lo, exec_lo, s45
	v_dual_mov_b32 v1, v128 :: v_dual_mov_b32 v114, v123
	s_add_nc_u64 s[30:31], s[30:31], s[24:25]
	s_mov_b32 s45, 0
.LBB0_30:                               ;   Parent Loop BB0_10 Depth=1
                                        ;     Parent Loop BB0_18 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 1, v[114:115]
	v_add_nc_u32_e32 v114, s37, v114
	s_wait_alu 0xfffe
	v_add_co_u32 v2, vcc_lo, s30, v2
	s_wait_alu 0xfffd
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v3, null, s31, v3, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s43, v114
	global_load_b128 v[2:5], v[2:3], off
	s_or_b32 s45, vcc_lo, s45
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[2:5]
	v_add_nc_u32_e32 v1, s40, v1
	s_and_not1_b32 exec_lo, exec_lo, s45
	s_cbranch_execnz .LBB0_30
.LBB0_31:                               ;   in Loop: Header=BB0_18 Depth=2
	s_or_b32 exec_lo, exec_lo, s44
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB0_17
.LBB0_32:                               ;   in Loop: Header=BB0_18 Depth=2
	v_dual_mov_b32 v52, 0 :: v_dual_add_nc_u32 v145, s42, v122
	v_dual_mov_b32 v64, 0 :: v_dual_mov_b32 v63, 0
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v61, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v114, s38, v145
	v_dual_mov_b32 v51, 0 :: v_dual_add_nc_u32 v146, 0x100, v145
	v_dual_mov_b32 v49, 0 :: v_dual_add_nc_u32 v144, 0x200, v145
	v_lshlrev_b64_e32 v[1:2], 1, v[114:115]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_min_u32_e32 v114, s38, v146
	v_dual_mov_b32 v50, 0 :: v_dual_add_nc_u32 v143, 0x300, v145
	v_dual_mov_b32 v76, 0 :: v_dual_mov_b32 v75, 0
	v_add_co_u32 v5, vcc_lo, s16, v1
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, s17, v2, vcc_lo
	v_lshlrev_b64_e32 v[3:4], 1, v[114:115]
	v_add_co_u32 v1, vcc_lo, v5, v119
	v_min_u32_e32 v114, s38, v144
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, v6, v120, vcc_lo
	v_add_co_u32 v5, vcc_lo, v5, v117
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, v6, v118, vcc_lo
	v_add_co_u32 v7, vcc_lo, s16, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, s17, v4, vcc_lo
	v_lshlrev_b64_e32 v[3:4], 1, v[114:115]
	s_clause 0x1
	global_load_b128 v[29:32], v[1:2], off
	global_load_b128 v[33:36], v[5:6], off
	v_add_co_u32 v1, vcc_lo, v7, v119
	v_min_u32_e32 v114, s38, v143
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, v8, v120, vcc_lo
	v_add_co_u32 v5, vcc_lo, v7, v117
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v6, null, v8, v118, vcc_lo
	v_add_co_u32 v9, vcc_lo, s16, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v10, null, s17, v4, vcc_lo
	v_lshlrev_b64_e32 v[3:4], 1, v[114:115]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v7, vcc_lo, v9, v119
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v8, null, v10, v120, vcc_lo
	v_dual_mov_b32 v74, 0 :: v_dual_mov_b32 v73, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v11, vcc_lo, s16, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v12, null, s17, v4, vcc_lo
	v_add_co_u32 v3, vcc_lo, v9, v117
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, v10, v118, vcc_lo
	v_add_co_u32 v25, vcc_lo, v11, v119
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v26, null, v12, v120, vcc_lo
	v_add_co_u32 v27, vcc_lo, v11, v117
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v28, null, v12, v118, vcc_lo
	s_clause 0x5
	global_load_b128 v[17:20], v[1:2], off
	global_load_b128 v[21:24], v[5:6], off
	global_load_b128 v[9:12], v[7:8], off
	global_load_b128 v[13:16], v[3:4], off
	global_load_b128 v[1:4], v[25:26], off
	global_load_b128 v[5:8], v[27:28], off
	v_cmp_gt_u32_e32 vcc_lo, s12, v145
	v_dual_mov_b32 v88, 0 :: v_dual_mov_b32 v87, 0
	v_dual_mov_b32 v86, 0 :: v_dual_mov_b32 v85, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v59, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v57, 0
	v_dual_mov_b32 v80, 0 :: v_dual_mov_b32 v79, 0
	v_dual_mov_b32 v78, 0 :: v_dual_mov_b32 v77, 0
	v_dual_mov_b32 v92, 0 :: v_dual_mov_b32 v91, 0
	v_dual_mov_b32 v90, 0 :: v_dual_mov_b32 v89, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v84, 0 :: v_dual_mov_b32 v83, 0
	v_dual_mov_b32 v82, 0 :: v_dual_mov_b32 v81, 0
	v_dual_mov_b32 v104, 0 :: v_dual_mov_b32 v103, 0
	v_dual_mov_b32 v102, 0 :: v_dual_mov_b32 v101, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v72, 0 :: v_dual_mov_b32 v71, 0
	v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v69, 0
	v_dual_mov_b32 v100, 0 :: v_dual_mov_b32 v99, 0
	v_dual_mov_b32 v98, 0 :: v_dual_mov_b32 v97, 0
	v_dual_mov_b32 v108, 0 :: v_dual_mov_b32 v107, 0
	v_dual_mov_b32 v106, 0 :: v_dual_mov_b32 v105, 0
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v39, 0
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v67, 0
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v65, 0
	v_dual_mov_b32 v96, 0 :: v_dual_mov_b32 v95, 0
	v_dual_mov_b32 v94, 0 :: v_dual_mov_b32 v93, 0
	v_dual_mov_b32 v112, 0 :: v_dual_mov_b32 v111, 0
	v_dual_mov_b32 v110, 0 :: v_dual_mov_b32 v109, 0
	s_and_saveexec_b32 s31, vcc_lo
	s_cbranch_execz .LBB0_16
; %bb.33:                               ;   in Loop: Header=BB0_18 Depth=2
	v_subrev_nc_u32_e32 v25, s22, v145
	v_cmp_gt_u32_e32 vcc_lo, s12, v146
	v_dual_mov_b32 v94, 0 :: v_dual_mov_b32 v95, 0
	v_dual_mov_b32 v96, 0 :: v_dual_mov_b32 v65, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_dual_mov_b32 v93, 0 :: v_dual_lshlrev_b32 v114, 1, v25
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v67, 0
	ds_load_b128 v[109:112], v114
	ds_load_b128 v[105:108], v114 offset:12288
	ds_load_b128 v[101:104], v114 offset:24576
	ds_load_b128 v[89:92], v114 offset:36864
	ds_load_b128 v[85:88], v114 offset:49152
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v39, 0
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v97, 0
	v_dual_mov_b32 v98, 0 :: v_dual_mov_b32 v99, 0
	v_dual_mov_b32 v100, 0 :: v_dual_mov_b32 v69, 0
	v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v71, 0
	v_dual_mov_b32 v72, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v81, 0
	v_dual_mov_b32 v82, 0 :: v_dual_mov_b32 v83, 0
	v_dual_mov_b32 v84, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v28, 0 :: v_dual_mov_b32 v77, 0
	v_dual_mov_b32 v78, 0 :: v_dual_mov_b32 v79, 0
	v_dual_mov_b32 v80, 0 :: v_dual_mov_b32 v57, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v59, 0
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v73, 0
	v_dual_mov_b32 v74, 0 :: v_dual_mov_b32 v75, 0
	v_dual_mov_b32 v76, 0 :: v_dual_mov_b32 v61, 0
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v63, 0
	v_dual_mov_b32 v64, 0 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v51, 0
	v_mov_b32_e32 v52, 0
	s_and_saveexec_b32 s43, vcc_lo
	s_cbranch_execz .LBB0_15
; %bb.34:                               ;   in Loop: Header=BB0_18 Depth=2
	ds_load_b128 v[93:96], v114 offset:512
	ds_load_b128 v[97:100], v114 offset:12800
	ds_load_b128 v[81:84], v114 offset:25088
	ds_load_b128 v[77:80], v114 offset:37376
	ds_load_b128 v[73:76], v114 offset:49664
	v_cmp_gt_u32_e32 vcc_lo, s12, v144
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v72, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v60, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v64, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v52, 0
	s_and_saveexec_b32 s44, vcc_lo
	s_cbranch_execz .LBB0_14
; %bb.35:                               ;   in Loop: Header=BB0_18 Depth=2
	ds_load_b128 v[65:68], v114 offset:1024
	ds_load_b128 v[69:72], v114 offset:13312
	ds_load_b128 v[53:56], v114 offset:25600
	ds_load_b128 v[57:60], v114 offset:37888
	ds_load_b128 v[61:64], v114 offset:50176
	v_cmp_gt_u32_e32 vcc_lo, s12, v143
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v28, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v49, 0 :: v_dual_mov_b32 v50, 0
	v_dual_mov_b32 v51, 0 :: v_dual_mov_b32 v52, 0
	s_and_saveexec_b32 s45, vcc_lo
	s_cbranch_execz .LBB0_13
; %bb.36:                               ;   in Loop: Header=BB0_18 Depth=2
	ds_load_b128 v[37:40], v114 offset:1536
	ds_load_b128 v[45:48], v114 offset:13824
	ds_load_b128 v[25:28], v114 offset:26112
	ds_load_b128 v[41:44], v114 offset:38400
	ds_load_b128 v[49:52], v114 offset:50688
	s_branch .LBB0_13
.LBB0_37:                               ;   in Loop: Header=BB0_10 Depth=1
	s_and_saveexec_b32 s22, s2
	s_cbranch_execz .LBB0_9
; %bb.38:                               ;   in Loop: Header=BB0_10 Depth=1
	v_xor_b32_e32 v1, 16, v129
	v_add_f32_dpp v3, v141, v141 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v140, v140 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v139, v139 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x0
	v_add_f32_dpp v6, v138, v138 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v142, v142 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v133, v133 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v1, v129, v1, vcc_lo
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v7, v7 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v9, 2, v1
	v_add_f32_dpp v1, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v22, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v137, v137 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v20, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v18, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v14, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v136, v136 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v135, v135 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v134, v134 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v8, v7, v7 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v15, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v2, v9, v1
	ds_bpermute_b32 v23, v9, v22
	v_add_f32_dpp v12, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v10, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v6, v6 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v8, v8 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v21, v9, v20
	ds_bpermute_b32 v19, v9, v18
	ds_bpermute_b32 v16, v9, v14
	ds_bpermute_b32 v17, v9, v15
	ds_bpermute_b32 v13, v9, v12
	ds_bpermute_b32 v11, v9, v10
	ds_bpermute_b32 v8, v9, v7
	ds_bpermute_b32 v6, v9, v5
	s_and_b32 exec_lo, exec_lo, s1
	s_cbranch_execz .LBB0_9
; %bb.39:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x9
	v_add_f32_e32 v9, v1, v2
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v117, v115 :: v_dual_mul_f32 v24, s8, v9
	v_lshlrev_b64_e32 v[3:4], 1, v[116:117]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v1, vcc_lo, s4, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s5, v4, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_41
; %bb.40:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v9, v[1:2], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v24, s9, v9, v24 op_sel_hi:[0,1,0]
.LBB0_41:                               ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e64 v9, v132
	v_add_co_u32 v3, vcc_lo, s6, v3
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s7, v4, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v25, v24, v9
	v_cmp_gt_u32_e64 s2, s13, v131
	v_cndmask_b32_e64 v24, v24, v25, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v24, v24
	global_store_b16 v[3:4], v24, off
	s_and_saveexec_b32 s30, s2
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
	global_load_u16 v23, v[1:2], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v22, s9, v23, v22 op_sel_hi:[0,1,0]
.LBB0_44:                               ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e64 v23, v130
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v23, v22, v23
	v_cndmask_b32_e64 v22, v22, v23, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v22, v22
	global_store_b16 v[3:4], v22, off offset:2
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
	v_add_f32_e32 v21, v20, v9
	v_add_co_u32 v3, vcc_lo, v3, s28
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s29, v4, vcc_lo
	v_cndmask_b32_e64 v20, v20, v21, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v20, v20
	global_store_b16 v[3:4], v20, off
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB0_51
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
	v_cvt_f32_f16_e64 v19, v130
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v19, v18, v19
	v_cndmask_b32_e64 v18, v18, v19, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v18, v18
	global_store_b16 v[3:4], v18, off offset:2
.LBB0_51:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s30
	s_wait_dscnt 0x5
	v_add_f32_e32 v14, v14, v16
	v_add_co_u32 v1, vcc_lo, v1, s26
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v2, null, s27, v2, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v14, s8, v14
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_53
; %bb.52:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v16, v[1:2], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v14, s9, v16, v14 op_sel_hi:[0,1,0]
.LBB0_53:                               ;   in Loop: Header=BB0_10 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v16, v14, v9
	v_add_co_u32 v3, vcc_lo, v3, s28
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v4, null, s29, v4, vcc_lo
	v_cndmask_b32_e64 v14, v14, v16, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v14, v14
	global_store_b16 v[3:4], v14, off
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB0_57
; %bb.54:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x4
	v_add_f32_e32 v14, v15, v17
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v14, s8, v14
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_56
; %bb.55:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v15, v[1:2], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v14, s9, v15, v14 op_sel_hi:[0,1,0]
.LBB0_56:                               ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e64 v15, v130
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v15, v14, v15
	v_cndmask_b32_e64 v14, v14, v15, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v14, v14
	global_store_b16 v[3:4], v14, off offset:2
.LBB0_57:                               ;   in Loop: Header=BB0_10 Depth=1
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
	s_cbranch_vccnz .LBB0_59
; %bb.58:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v13, v[1:2], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v12, s9, v13, v12 op_sel_hi:[0,1,0]
.LBB0_59:                               ;   in Loop: Header=BB0_10 Depth=1
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
	s_cbranch_execz .LBB0_63
; %bb.60:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x2
	v_add_f32_e32 v10, v10, v11
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v10, s8, v10
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_62
; %bb.61:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v11, v[1:2], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v10, s9, v11, v10 op_sel_hi:[0,1,0]
.LBB0_62:                               ;   in Loop: Header=BB0_10 Depth=1
	v_cvt_f32_f16_e64 v11, v130
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v11, v10, v11
	v_cndmask_b32_e64 v10, v10, v11, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v10, v10
	global_store_b16 v[3:4], v10, off offset:2
.LBB0_63:                               ;   in Loop: Header=BB0_10 Depth=1
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
	s_cbranch_vccnz .LBB0_65
; %bb.64:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v8, v[1:2], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v7, s9, v8, v7 op_sel_hi:[0,1,0]
.LBB0_65:                               ;   in Loop: Header=BB0_10 Depth=1
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
; %bb.66:                               ;   in Loop: Header=BB0_10 Depth=1
	s_wait_dscnt 0x0
	v_add_f32_e32 v5, v5, v6
	s_and_not1_b32 vcc_lo, exec_lo, s39
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v5, s8, v5
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_8
; %bb.67:                               ;   in Loop: Header=BB0_10 Depth=1
	global_load_u16 v1, v[1:2], off offset:2
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
	s_cbranch_vccnz .LBB0_132
; %bb.70:
	v_and_b32_e32 v4, 0x3ff, v0
	s_wait_dscnt 0x9
	v_lshlrev_b32_e32 v2, 9, v121
	s_mov_b32 s3, 0
	s_wait_kmcnt 0x0
	s_mov_b32 s2, s19
	v_lshlrev_b32_e32 v118, 3, v4
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshl_add_u32 v5, v121, 8, v118
	v_cmp_gt_u32_e32 vcc_lo, s12, v5
	s_and_saveexec_b32 s19, vcc_lo
	s_cbranch_execz .LBB0_75
; %bb.71:
	v_dual_mov_b32 v9, v5 :: v_dual_lshlrev_b32 v0, 4, v4
	s_wait_alu 0xfffe
	s_mov_b32 s22, s3
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v3, s0, v2, v0
	s_wait_dscnt 0x0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v6, null, 0, 0, s0
	v_add_nc_u32_e32 v7, v2, v0
	v_add_co_u32 v0, s0, s20, v3
	s_wait_alu 0xf1ff
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_add_co_ci_u32_e64 v1, null, s21, v6, s0
	v_mov_b32_e32 v8, v7
.LBB0_72:                               ; =>This Inner Loop Header: Depth=1
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
	s_cbranch_execnz .LBB0_72
; %bb.73:
	s_or_b32 exec_lo, exec_lo, s22
	s_lshl_b64 s[0:1], s[2:3], 1
	v_lshl_add_u32 v7, s12, 1, v7
	s_wait_alu 0xfffe
	s_add_nc_u64 s[0:1], s[20:21], s[0:1]
	s_mov_b32 s22, 0
	s_wait_alu 0xfffe
	v_add_co_u32 v0, s0, s0, v3
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v1, null, s1, v6, s0
	v_mov_b32_e32 v3, v5
.LBB0_74:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[8:11], v[0:1], off
	v_add_nc_u32_e32 v3, 0x1000, v3
	v_add_co_u32 v0, s0, 0x2000, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v1, null, 0, v1, s0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s1, s12, v3
	s_or_b32 s22, s1, s22
	s_wait_loadcnt 0x0
	ds_store_b128 v7, v[8:11]
	v_add_nc_u32_e32 v7, 0x2000, v7
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s22
	s_cbranch_execnz .LBB0_74
.LBB0_75:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s19
	s_lshl_b32 s19, s12, 2
	s_and_saveexec_b32 s22, vcc_lo
	s_cbranch_execz .LBB0_78
; %bb.76:
	v_lshlrev_b32_e32 v0, 4, v4
	s_lshl_b64 s[0:1], s[2:3], 2
	s_wait_alu 0xfffe
	s_add_nc_u64 s[0:1], s[20:21], s[0:1]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_co_u32 v1, s23, v2, v0
	s_wait_dscnt 0x0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v6, null, 0, 0, s23
	v_add3_u32 v3, v2, v0, s19
	s_wait_alu 0xfffe
	v_add_co_u32 v0, s0, s0, v1
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v1, null, s1, v6, s0
	v_mov_b32_e32 v6, v5
	s_mov_b32 s23, 0
.LBB0_77:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[7:10], v[0:1], off
	v_add_nc_u32_e32 v6, 0x1000, v6
	v_add_co_u32 v0, s0, 0x2000, v0
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v1, null, 0, v1, s0
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s1, s12, v6
	s_wait_alu 0xfffe
	s_or_b32 s23, s1, s23
	s_wait_loadcnt 0x0
	ds_store_b128 v3, v[7:10]
	v_add_nc_u32_e32 v3, 0x2000, v3
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s23
	s_cbranch_execnz .LBB0_77
.LBB0_78:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s22
	s_and_saveexec_b32 s1, vcc_lo
	s_cbranch_execz .LBB0_83
; %bb.79:
	s_wait_dscnt 0x1
	v_dual_mov_b32 v8, v5 :: v_dual_lshlrev_b32 v1, 4, v4
	s_mul_u64 s[22:23], s[2:3], 6
	s_wait_alu 0xfffe
	s_add_nc_u64 s[22:23], s[20:21], s[22:23]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_nc_u32_e32 v0, v2, v1
	s_wait_dscnt 0x0
	v_add_co_u32 v6, s0, v2, v1
	s_wait_alu 0xf1ff
	v_add_co_ci_u32_e64 v7, null, 0, 0, s0
	v_mad_co_u64_u32 v[1:2], null, s12, 6, v[0:1]
	s_wait_alu 0xfffe
	v_add_co_u32 v2, vcc_lo, s22, v6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s23, v7, vcc_lo
	s_mov_b32 s22, 0
.LBB0_80:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[9:12], v[2:3], off
	v_add_nc_u32_e32 v8, 0x1000, v8
	v_add_co_u32 v2, vcc_lo, 0x2000, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, 0, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s0, s12, v8
	s_wait_alu 0xfffe
	s_or_b32 s22, s0, s22
	s_wait_loadcnt 0x0
	ds_store_b128 v1, v[9:12]
	v_add_nc_u32_e32 v1, 0x2000, v1
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s22
	s_cbranch_execnz .LBB0_80
; %bb.81:
	s_or_b32 exec_lo, exec_lo, s22
	s_lshl_b64 s[2:3], s[2:3], 3
	v_lshl_add_u32 v2, s12, 3, v0
	s_wait_alu 0xfffe
	s_add_nc_u64 s[2:3], s[20:21], s[2:3]
	s_wait_alu 0xfffe
	v_add_co_u32 v0, vcc_lo, s2, v6
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s3, v7, vcc_lo
	s_mov_b32 s2, 0
.LBB0_82:                               ; =>This Inner Loop Header: Depth=1
	global_load_b128 v[6:9], v[0:1], off
	v_add_nc_u32_e32 v5, 0x1000, v5
	v_add_co_u32 v0, vcc_lo, 0x2000, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, 0, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s0, s12, v5
	s_wait_alu 0xfffe
	s_or_b32 s2, s0, s2
	s_wait_loadcnt 0x0
	ds_store_b128 v2, v[6:9]
	v_add_nc_u32_e32 v2, 0x2000, v2
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s2
	s_cbranch_execnz .LBB0_82
.LBB0_83:
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_storecnt 0x0
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_u32_e64 s35, v121
	s_cbranch_execz .LBB0_132
; %bb.84:
	v_cmp_gt_u32_e32 vcc_lo, s13, v113
	s_and_b32 exec_lo, exec_lo, vcc_lo
	s_cbranch_execz .LBB0_132
; %bb.85:
	s_lshl_b32 s22, s12, 1
	s_cmp_eq_u64 s[14:15], 0
	s_mov_b32 s3, 0
	s_cselect_b32 s0, -1, 0
	s_cmp_lg_u64 s[14:15], 0
	v_cmp_eq_u32_e64 s1, 31, v4
	s_cselect_b32 s23, -1, 0
	s_cmp_lg_u32 s12, 0
	v_dual_mov_b32 v112, 0 :: v_dual_lshlrev_b32 v119, 4, v4
	s_cselect_b32 s24, -1, 0
	s_add_co_i32 s25, s12, -8
	s_cmp_neq_f32 s9, 0
	v_mbcnt_lo_u32_b32 v120, -1, 0
	s_mov_b32 s2, s10
	s_mov_b32 s20, s11
	s_wait_alu 0xfffe
	s_mov_b32 s21, s3
	s_cselect_b32 s26, -1, 0
	s_lshl_b64 s[10:11], s[2:3], 1
	s_wait_alu 0xfffe
	s_lshl_b64 s[20:21], s[20:21], 1
	s_mul_i32 s27, s12, 6
	s_lshl_b32 s28, s12, 3
	s_branch .LBB0_88
.LBB0_86:                               ;   in Loop: Header=BB0_88 Depth=1
	v_cvt_f32_f16_e32 v0, v121
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v0, v4, v0
	v_cndmask_b32_e64 v0, v0, v4, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v0, v0
	global_store_b16 v[2:3], v0, off offset:2
.LBB0_87:                               ;   in Loop: Header=BB0_88 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s29
	v_add_nc_u32_e32 v113, s34, v113
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_le_u32_e32 vcc_lo, s13, v113
	s_or_b32 s3, vcc_lo, s3
	s_wait_alu 0xfffe
	s_and_not1_b32 exec_lo, exec_lo, s3
	s_cbranch_execz .LBB0_132
.LBB0_88:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_96 Depth 2
	v_or_b32_e32 v122, 1, v113
	v_mov_b32_e32 v123, v112
	v_mov_b32_e32 v121, v112
	s_and_not1_b32 vcc_lo, exec_lo, s23
	s_delay_alu instid0(VALU_DEP_3)
	v_min_u32_e32 v111, s33, v122
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_90
; %bb.89:                               ;   in Loop: Header=BB0_88 Depth=1
	v_mov_b32_e32 v114, v112
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_lshlrev_b64_e32 v[2:3], 1, v[111:112]
	s_wait_dscnt 0x9
	v_lshlrev_b64_e32 v[0:1], 1, v[113:114]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s14, v0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s15, v1, vcc_lo
	v_add_co_u32 v2, vcc_lo, s14, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s15, v3, vcc_lo
	s_clause 0x1
	global_load_u16 v123, v[0:1], off
	global_load_u16 v121, v[2:3], off
.LBB0_90:                               ;   in Loop: Header=BB0_88 Depth=1
	s_and_not1_b32 vcc_lo, exec_lo, s24
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_101
; %bb.91:                               ;   in Loop: Header=BB0_88 Depth=1
	v_min_u32_e32 v2, s33, v113
	s_wait_dscnt 0x9
	v_mad_co_u64_u32 v[0:1], null, v111, s18, 0
	v_dual_mov_b32 v128, 0 :: v_dual_mov_b32 v127, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_mad_co_u64_u32 v[2:3], null, v2, s18, 0
	v_dual_mov_b32 v124, v119 :: v_dual_mov_b32 v125, 0
	v_lshlrev_b64_e32 v[114:115], 1, v[0:1]
	v_dual_mov_b32 v126, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v131, 0
	v_lshlrev_b64_e32 v[116:117], 1, v[2:3]
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v133, 0
	v_mov_b32_e32 v134, 0
	s_mov_b32 s2, 0
	s_branch .LBB0_96
.LBB0_92:                               ;   in Loop: Header=BB0_96 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s35
.LBB0_93:                               ;   in Loop: Header=BB0_96 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s31
.LBB0_94:                               ;   in Loop: Header=BB0_96 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s30
.LBB0_95:                               ;   in Loop: Header=BB0_96 Depth=2
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s29
	s_wait_loadcnt_dscnt 0x704
	v_dot2_f32_f16 v134, v108, v28, v134
	s_wait_loadcnt 0x6
	v_dot2_f32_f16 v108, v108, v32, v133
	s_wait_dscnt 0x3
	v_dot2_f32_f16 v132, v104, v28, v132
	v_dot2_f32_f16 v104, v104, v32, v131
	s_wait_dscnt 0x2
	v_dot2_f32_f16 v130, v96, v28, v130
	v_dot2_f32_f16 v131, v109, v29, v134
	v_dot2_f32_f16 v108, v109, v33, v108
	v_dot2_f32_f16 v109, v105, v29, v132
	v_dot2_f32_f16 v104, v105, v33, v104
	v_dot2_f32_f16 v96, v96, v32, v129
	v_dot2_f32_f16 v105, v97, v29, v130
	v_dot2_f32_f16 v130, v110, v30, v131
	v_dot2_f32_f16 v109, v106, v30, v109
	v_dot2_f32_f16 v104, v106, v34, v104
	v_dot2_f32_f16 v108, v110, v34, v108
	v_dot2_f32_f16 v105, v98, v30, v105
	v_dot2_f32_f16 v106, v111, v31, v130
	v_dot2_f32_f16 v109, v107, v31, v109
	v_dot2_f32_f16 v104, v107, v35, v104
	s_wait_dscnt 0x1
	v_dot2_f32_f16 v107, v88, v28, v128
	s_wait_dscnt 0x0
	v_dot2_f32_f16 v28, v84, v28, v126
	v_dot2_f32_f16 v88, v88, v32, v127
	v_dot2_f32_f16 v32, v84, v32, v125
	v_dot2_f32_f16 v84, v97, v33, v96
	v_dot2_f32_f16 v96, v89, v29, v107
	v_dot2_f32_f16 v28, v85, v29, v28
	v_dot2_f32_f16 v88, v89, v33, v88
	v_dot2_f32_f16 v29, v85, v33, v32
	v_dot2_f32_f16 v32, v98, v34, v84
	v_dot2_f32_f16 v33, v90, v30, v96
	v_dot2_f32_f16 v28, v86, v30, v28
	v_dot2_f32_f16 v105, v99, v31, v105
	v_dot2_f32_f16 v84, v90, v34, v88
	v_dot2_f32_f16 v29, v86, v34, v29
	v_dot2_f32_f16 v30, v99, v35, v32
	v_dot2_f32_f16 v32, v91, v31, v33
	v_dot2_f32_f16 v28, v87, v31, v28
	s_wait_loadcnt 0x5
	v_dot2_f32_f16 v31, v100, v16, v106
	v_dot2_f32_f16 v108, v111, v35, v108
	v_dot2_f32_f16 v33, v91, v35, v84
	v_dot2_f32_f16 v29, v87, v35, v29
	v_dot2_f32_f16 v35, v92, v16, v109
	v_dot2_f32_f16 v85, v80, v16, v105
	v_dot2_f32_f16 v31, v101, v17, v31
	s_wait_loadcnt 0x4
	v_dot2_f32_f16 v30, v80, v20, v30
	v_dot2_f32_f16 v32, v76, v16, v32
	v_dot2_f32_f16 v16, v72, v16, v28
	v_dot2_f32_f16 v34, v100, v20, v108
	v_dot2_f32_f16 v84, v92, v20, v104
	v_dot2_f32_f16 v35, v93, v17, v35
	v_dot2_f32_f16 v85, v81, v17, v85
	v_dot2_f32_f16 v31, v102, v18, v31
	v_dot2_f32_f16 v33, v76, v20, v33
	v_dot2_f32_f16 v20, v72, v20, v29
	v_dot2_f32_f16 v28, v81, v21, v30
	v_dot2_f32_f16 v29, v77, v17, v32
	v_dot2_f32_f16 v16, v73, v17, v16
	v_dot2_f32_f16 v34, v101, v21, v34
	v_dot2_f32_f16 v84, v93, v21, v84
	v_dot2_f32_f16 v35, v94, v18, v35
	v_dot2_f32_f16 v85, v82, v18, v85
	v_dot2_f32_f16 v31, v103, v19, v31
	v_dot2_f32_f16 v30, v77, v21, v33
	v_dot2_f32_f16 v17, v73, v21, v20
	v_dot2_f32_f16 v20, v82, v22, v28
	v_dot2_f32_f16 v21, v78, v18, v29
	v_dot2_f32_f16 v16, v74, v18, v16
	v_dot2_f32_f16 v34, v102, v22, v34
	v_dot2_f32_f16 v84, v94, v22, v84
	v_dot2_f32_f16 v35, v95, v19, v35
	v_dot2_f32_f16 v85, v83, v19, v85
	v_dot2_f32_f16 v28, v78, v22, v30
	v_dot2_f32_f16 v17, v74, v22, v17
	v_dot2_f32_f16 v18, v83, v23, v20
	v_dot2_f32_f16 v20, v79, v19, v21
	v_dot2_f32_f16 v16, v75, v19, v16
	s_wait_loadcnt 0x3
	v_dot2_f32_f16 v19, v64, v8, v31
	v_dot2_f32_f16 v34, v103, v23, v34
	v_dot2_f32_f16 v84, v95, v23, v84
	v_dot2_f32_f16 v21, v79, v23, v28
	v_dot2_f32_f16 v17, v75, v23, v17
	v_dot2_f32_f16 v23, v68, v8, v35
	v_dot2_f32_f16 v29, v52, v8, v85
	v_dot2_f32_f16 v19, v65, v9, v19
	s_wait_loadcnt 0x2
	v_dot2_f32_f16 v18, v52, v12, v18
	v_dot2_f32_f16 v20, v56, v8, v20
	v_dot2_f32_f16 v8, v60, v8, v16
	v_dot2_f32_f16 v22, v64, v12, v34
	v_dot2_f32_f16 v28, v68, v12, v84
	v_dot2_f32_f16 v23, v69, v9, v23
	v_dot2_f32_f16 v29, v53, v9, v29
	v_dot2_f32_f16 v19, v66, v10, v19
	v_dot2_f32_f16 v21, v56, v12, v21
	v_dot2_f32_f16 v12, v60, v12, v17
	v_dot2_f32_f16 v16, v53, v13, v18
	v_dot2_f32_f16 v17, v57, v9, v20
	v_dot2_f32_f16 v8, v61, v9, v8
	v_dot2_f32_f16 v22, v65, v13, v22
	v_dot2_f32_f16 v28, v69, v13, v28
	v_dot2_f32_f16 v23, v70, v10, v23
	v_dot2_f32_f16 v29, v54, v10, v29
	v_dot2_f32_f16 v19, v67, v11, v19
	v_dot2_f32_f16 v18, v57, v13, v21
	v_dot2_f32_f16 v9, v61, v13, v12
	v_dot2_f32_f16 v12, v54, v14, v16
	v_dot2_f32_f16 v13, v58, v10, v17
	v_dot2_f32_f16 v8, v62, v10, v8
	v_dot2_f32_f16 v23, v71, v11, v23
	v_dot2_f32_f16 v29, v55, v11, v29
	v_dot2_f32_f16 v10, v55, v15, v12
	v_dot2_f32_f16 v12, v59, v11, v13
	v_dot2_f32_f16 v8, v63, v11, v8
	s_wait_loadcnt 0x1
	v_dot2_f32_f16 v11, v36, v0, v19
	v_dot2_f32_f16 v22, v66, v14, v22
	v_dot2_f32_f16 v28, v70, v14, v28
	v_dot2_f32_f16 v16, v58, v14, v18
	v_dot2_f32_f16 v9, v62, v14, v9
	v_dot2_f32_f16 v11, v37, v1, v11
	v_dot2_f32_f16 v22, v67, v15, v22
	v_dot2_f32_f16 v28, v71, v15, v28
	v_dot2_f32_f16 v13, v59, v15, v16
	v_dot2_f32_f16 v9, v63, v15, v9
	v_dot2_f32_f16 v11, v38, v2, v11
	s_wait_loadcnt 0x0
	v_dot2_f32_f16 v14, v36, v4, v22
	v_dot2_f32_f16 v15, v44, v0, v23
	v_dot2_f32_f16 v16, v44, v4, v28
	v_dot2_f32_f16 v17, v24, v0, v29
	v_dot2_f32_f16 v134, v39, v3, v11
	v_dot2_f32_f16 v10, v24, v4, v10
	v_dot2_f32_f16 v11, v40, v0, v12
	v_dot2_f32_f16 v12, v40, v4, v13
	v_dot2_f32_f16 v0, v48, v0, v8
	v_dot2_f32_f16 v4, v48, v4, v9
	v_dot2_f32_f16 v14, v37, v5, v14
	v_dot2_f32_f16 v15, v45, v1, v15
	v_dot2_f32_f16 v16, v45, v5, v16
	v_dot2_f32_f16 v17, v25, v1, v17
	v_dot2_f32_f16 v8, v25, v5, v10
	v_dot2_f32_f16 v9, v41, v1, v11
	v_dot2_f32_f16 v10, v41, v5, v12
	v_dot2_f32_f16 v0, v49, v1, v0
	v_dot2_f32_f16 v1, v49, v5, v4
	v_dot2_f32_f16 v14, v38, v6, v14
	v_dot2_f32_f16 v15, v46, v2, v15
	v_dot2_f32_f16 v16, v46, v6, v16
	v_dot2_f32_f16 v17, v26, v2, v17
	v_dot2_f32_f16 v4, v26, v6, v8
	v_dot2_f32_f16 v5, v42, v2, v9
	v_dot2_f32_f16 v8, v42, v6, v10
	v_dot2_f32_f16 v0, v50, v2, v0
	v_dot2_f32_f16 v1, v50, v6, v1
	v_dot2_f32_f16 v133, v39, v7, v14
	v_dot2_f32_f16 v132, v47, v3, v15
	v_dot2_f32_f16 v131, v47, v7, v16
	v_dot2_f32_f16 v130, v27, v3, v17
	v_dot2_f32_f16 v129, v27, v7, v4
	v_dot2_f32_f16 v128, v43, v3, v5
	v_dot2_f32_f16 v127, v43, v7, v8
	v_dot2_f32_f16 v126, v51, v3, v0
	v_dot2_f32_f16 v125, v51, v7, v1
	v_add_nc_u32_e32 v124, 0x800, v124
	s_addk_co_i32 s2, 0x400
	s_wait_alu 0xfffe
	s_cmp_ge_u32 s2, s12
	s_cbranch_scc1 .LBB0_102
.LBB0_96:                               ;   Parent Loop BB0_88 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	s_wait_alu 0xfffe
	v_dual_mov_b32 v51, 0 :: v_dual_add_nc_u32 v36, s2, v118
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v61, 0
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v75, 0
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v111, s25, v36
	v_dual_mov_b32 v49, 0 :: v_dual_add_nc_u32 v138, 0x100, v36
	v_dual_mov_b32 v63, 0 :: v_dual_add_nc_u32 v136, 0x200, v36
	v_lshlrev_b64_e32 v[0:1], 1, v[111:112]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)
	v_min_u32_e32 v111, s25, v138
	v_dual_mov_b32 v50, 0 :: v_dual_add_nc_u32 v135, 0x300, v36
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v73, 0
	v_add_co_u32 v4, vcc_lo, s16, v0
	s_wait_dscnt 0x0
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, s17, v1, vcc_lo
	v_lshlrev_b64_e32 v[2:3], 1, v[111:112]
	v_add_co_u32 v0, vcc_lo, v4, v116
	v_min_u32_e32 v111, s25, v136
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v5, v117, vcc_lo
	v_add_co_u32 v4, vcc_lo, v4, v114
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v5, v115, vcc_lo
	v_add_co_u32 v6, vcc_lo, s16, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v7, null, s17, v3, vcc_lo
	v_lshlrev_b64_e32 v[2:3], 1, v[111:112]
	s_clause 0x1
	global_load_b128 v[28:31], v[0:1], off
	global_load_b128 v[32:35], v[4:5], off
	v_add_co_u32 v0, vcc_lo, v6, v116
	v_min_u32_e32 v111, s25, v135
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, v7, v117, vcc_lo
	v_add_co_u32 v4, vcc_lo, v6, v114
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v5, null, v7, v115, vcc_lo
	v_add_co_u32 v8, vcc_lo, s16, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v9, null, s17, v3, vcc_lo
	v_lshlrev_b64_e32 v[2:3], 1, v[111:112]
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v6, vcc_lo, v8, v116
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v7, null, v9, v117, vcc_lo
	v_dual_mov_b32 v74, 0 :: v_dual_mov_b32 v87, 0
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v10, vcc_lo, s16, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v11, null, s17, v3, vcc_lo
	v_add_co_u32 v2, vcc_lo, v8, v114
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, v9, v115, vcc_lo
	v_add_co_u32 v24, vcc_lo, v10, v116
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v25, null, v11, v117, vcc_lo
	v_add_co_u32 v26, vcc_lo, v10, v114
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v27, null, v11, v115, vcc_lo
	s_clause 0x5
	global_load_b128 v[16:19], v[0:1], off
	global_load_b128 v[20:23], v[4:5], off
	global_load_b128 v[8:11], v[6:7], off
	global_load_b128 v[12:15], v[2:3], off
	global_load_b128 v[0:3], v[24:25], off
	global_load_b128 v[4:7], v[26:27], off
	v_cmp_gt_u32_e32 vcc_lo, s12, v36
	v_dual_mov_b32 v72, 0 :: v_dual_mov_b32 v85, 0
	v_dual_mov_b32 v86, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v84, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v59, 0
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v57, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v79, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v77, 0
	v_dual_mov_b32 v78, 0 :: v_dual_mov_b32 v91, 0
	v_dual_mov_b32 v76, 0 :: v_dual_mov_b32 v89, 0
	v_dual_mov_b32 v90, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v88, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v83, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v81, 0
	v_dual_mov_b32 v82, 0 :: v_dual_mov_b32 v99, 0
	v_dual_mov_b32 v80, 0 :: v_dual_mov_b32 v97, 0
	v_dual_mov_b32 v98, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v96, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v71, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v69, 0
	v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v95, 0
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v93, 0
	v_dual_mov_b32 v94, 0 :: v_dual_mov_b32 v107, 0
	v_dual_mov_b32 v92, 0 :: v_dual_mov_b32 v105, 0
	v_dual_mov_b32 v106, 0 :: v_dual_mov_b32 v39, 0
	v_dual_mov_b32 v104, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v67, 0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v65, 0
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v103, 0
	v_dual_mov_b32 v64, 0 :: v_dual_mov_b32 v101, 0
	v_dual_mov_b32 v102, 0 :: v_dual_mov_b32 v111, 0
	v_dual_mov_b32 v100, 0 :: v_dual_mov_b32 v109, 0
	v_mov_b32_e32 v110, 0
	v_mov_b32_e32 v108, 0
	s_and_saveexec_b32 s29, vcc_lo
	s_cbranch_execz .LBB0_95
; %bb.97:                               ;   in Loop: Header=BB0_96 Depth=2
	v_dual_mov_b32 v100, 0 :: v_dual_add_nc_u32 v141, s22, v124
	v_dual_mov_b32 v101, 0 :: v_dual_add_nc_u32 v140, s19, v124
	v_dual_mov_b32 v102, 0 :: v_dual_add_nc_u32 v139, s27, v124
	v_dual_mov_b32 v64, 0 :: v_dual_add_nc_u32 v137, s28, v124
	ds_load_b128 v[108:111], v124
	ds_load_b128 v[104:107], v141
	ds_load_b128 v[96:99], v140
	ds_load_b128 v[88:91], v139
	ds_load_b128 v[84:87], v137
	v_cmp_gt_u32_e32 vcc_lo, s12, v138
	v_dual_mov_b32 v103, 0 :: v_dual_mov_b32 v66, 0
	v_dual_mov_b32 v65, 0 :: v_dual_mov_b32 v36, 0
	v_dual_mov_b32 v67, 0 :: v_dual_mov_b32 v38, 0
	v_dual_mov_b32 v37, 0 :: v_dual_mov_b32 v92, 0
	v_dual_mov_b32 v39, 0 :: v_dual_mov_b32 v94, 0
	v_dual_mov_b32 v93, 0 :: v_dual_mov_b32 v68, 0
	v_dual_mov_b32 v95, 0 :: v_dual_mov_b32 v70, 0
	v_dual_mov_b32 v69, 0 :: v_dual_mov_b32 v44, 0
	v_dual_mov_b32 v71, 0 :: v_dual_mov_b32 v46, 0
	v_dual_mov_b32 v45, 0 :: v_dual_mov_b32 v80, 0
	v_dual_mov_b32 v47, 0 :: v_dual_mov_b32 v82, 0
	v_dual_mov_b32 v81, 0 :: v_dual_mov_b32 v52, 0
	v_dual_mov_b32 v83, 0 :: v_dual_mov_b32 v54, 0
	v_dual_mov_b32 v53, 0 :: v_dual_mov_b32 v24, 0
	v_dual_mov_b32 v55, 0 :: v_dual_mov_b32 v26, 0
	v_dual_mov_b32 v25, 0 :: v_dual_mov_b32 v76, 0
	v_dual_mov_b32 v27, 0 :: v_dual_mov_b32 v78, 0
	v_dual_mov_b32 v77, 0 :: v_dual_mov_b32 v56, 0
	v_dual_mov_b32 v79, 0 :: v_dual_mov_b32 v58, 0
	v_dual_mov_b32 v57, 0 :: v_dual_mov_b32 v40, 0
	v_dual_mov_b32 v59, 0 :: v_dual_mov_b32 v42, 0
	v_dual_mov_b32 v41, 0 :: v_dual_mov_b32 v72, 0
	v_dual_mov_b32 v43, 0 :: v_dual_mov_b32 v74, 0
	v_dual_mov_b32 v73, 0 :: v_dual_mov_b32 v60, 0
	v_dual_mov_b32 v75, 0 :: v_dual_mov_b32 v62, 0
	v_dual_mov_b32 v61, 0 :: v_dual_mov_b32 v48, 0
	v_dual_mov_b32 v63, 0 :: v_dual_mov_b32 v50, 0
	v_mov_b32_e32 v49, 0
	v_mov_b32_e32 v51, 0
	s_and_saveexec_b32 s30, vcc_lo
	s_cbranch_execz .LBB0_94
; %bb.98:                               ;   in Loop: Header=BB0_96 Depth=2
	ds_load_b128 v[100:103], v124 offset:512
	ds_load_b128 v[92:95], v141 offset:512
	ds_load_b128 v[80:83], v140 offset:512
	ds_load_b128 v[76:79], v139 offset:512
	ds_load_b128 v[72:75], v137 offset:512
	v_cmp_gt_u32_e32 vcc_lo, s12, v136
	v_dual_mov_b32 v64, 0 :: v_dual_mov_b32 v65, 0
	v_dual_mov_b32 v66, 0 :: v_dual_mov_b32 v67, 0
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v39, 0
	v_dual_mov_b32 v68, 0 :: v_dual_mov_b32 v69, 0
	v_dual_mov_b32 v70, 0 :: v_dual_mov_b32 v71, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v52, 0 :: v_dual_mov_b32 v53, 0
	v_dual_mov_b32 v54, 0 :: v_dual_mov_b32 v55, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v56, 0 :: v_dual_mov_b32 v57, 0
	v_dual_mov_b32 v58, 0 :: v_dual_mov_b32 v59, 0
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v60, 0 :: v_dual_mov_b32 v61, 0
	v_dual_mov_b32 v62, 0 :: v_dual_mov_b32 v63, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v51, 0
	s_and_saveexec_b32 s31, vcc_lo
	s_cbranch_execz .LBB0_93
; %bb.99:                               ;   in Loop: Header=BB0_96 Depth=2
	ds_load_b128 v[64:67], v124 offset:1024
	ds_load_b128 v[68:71], v141 offset:1024
	ds_load_b128 v[52:55], v140 offset:1024
	ds_load_b128 v[56:59], v139 offset:1024
	ds_load_b128 v[60:63], v137 offset:1024
	v_cmp_gt_u32_e32 vcc_lo, s12, v135
	v_dual_mov_b32 v36, 0 :: v_dual_mov_b32 v37, 0
	v_dual_mov_b32 v38, 0 :: v_dual_mov_b32 v39, 0
	v_dual_mov_b32 v44, 0 :: v_dual_mov_b32 v45, 0
	v_dual_mov_b32 v46, 0 :: v_dual_mov_b32 v47, 0
	v_dual_mov_b32 v24, 0 :: v_dual_mov_b32 v25, 0
	v_dual_mov_b32 v26, 0 :: v_dual_mov_b32 v27, 0
	v_dual_mov_b32 v40, 0 :: v_dual_mov_b32 v41, 0
	v_dual_mov_b32 v42, 0 :: v_dual_mov_b32 v43, 0
	v_dual_mov_b32 v48, 0 :: v_dual_mov_b32 v49, 0
	v_dual_mov_b32 v50, 0 :: v_dual_mov_b32 v51, 0
	s_and_saveexec_b32 s35, vcc_lo
	s_cbranch_execz .LBB0_92
; %bb.100:                              ;   in Loop: Header=BB0_96 Depth=2
	ds_load_b128 v[36:39], v124 offset:1536
	ds_load_b128 v[44:47], v141 offset:1536
	ds_load_b128 v[24:27], v140 offset:1536
	ds_load_b128 v[40:43], v139 offset:1536
	ds_load_b128 v[48:51], v137 offset:1536
	s_branch .LBB0_92
.LBB0_101:                              ;   in Loop: Header=BB0_88 Depth=1
	v_dual_mov_b32 v134, 0 :: v_dual_mov_b32 v133, 0
	v_dual_mov_b32 v132, 0 :: v_dual_mov_b32 v131, 0
	v_dual_mov_b32 v130, 0 :: v_dual_mov_b32 v129, 0
	v_dual_mov_b32 v125, 0 :: v_dual_mov_b32 v126, 0
	v_dual_mov_b32 v127, 0 :: v_dual_mov_b32 v128, 0
.LBB0_102:                              ;   in Loop: Header=BB0_88 Depth=1
	v_xor_b32_e32 v0, 16, v120
	v_add_f32_dpp v2, v133, v133 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v132, v132 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v131, v131 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_dscnt 0x0
	v_add_f32_dpp v5, v130, v130 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_cmp_gt_i32_e32 vcc_lo, 32, v0
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v134, v134 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v125, v125 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_loadcnt 0x1
	;;#ASMSTART
	;;#ASMEND
	s_wait_alu 0xfffd
	v_cndmask_b32_e32 v0, v120, v0, vcc_lo
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v6, v6 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	s_wait_loadcnt 0x0
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v8, 2, v0
	v_add_f32_dpp v0, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v21, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v129, v129 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v0, v0 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v1, v1, v1 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v19, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v17, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v13, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v128, v128 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v127, v127 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v126, v126 row_shr:8 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v2, v2, v2 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v7, v6, v6 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v0, v1, v1 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v14, v2, v2 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v3, v3, v3 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v4, v4 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v5, v5, v5 row_shr:2 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v1, v8, v0
	ds_bpermute_b32 v22, v8, v21
	v_add_f32_dpp v11, v3, v3 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v9, v4, v4 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v6, v5, v5 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_add_f32_dpp v4, v7, v7 row_shr:1 row_mask:0xf bank_mask:0xf bound_ctrl:1
	ds_bpermute_b32 v20, v8, v19
	ds_bpermute_b32 v18, v8, v17
	ds_bpermute_b32 v15, v8, v13
	ds_bpermute_b32 v16, v8, v14
	ds_bpermute_b32 v12, v8, v11
	ds_bpermute_b32 v10, v8, v9
	ds_bpermute_b32 v7, v8, v6
	ds_bpermute_b32 v5, v8, v4
	s_and_saveexec_b32 s29, s1
	s_cbranch_execz .LBB0_87
; %bb.103:                              ;   in Loop: Header=BB0_88 Depth=1
	s_wait_dscnt 0x9
	v_add_f32_e32 v8, v0, v1
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_dual_mov_b32 v114, v112 :: v_dual_mul_f32 v23, s8, v8
	v_lshlrev_b64_e32 v[2:3], 1, v[113:114]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v0, vcc_lo, s4, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s5, v3, vcc_lo
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_105
; %bb.104:                              ;   in Loop: Header=BB0_88 Depth=1
	global_load_u16 v8, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v23, s9, v8, v23 op_sel_hi:[0,1,0]
.LBB0_105:                              ;   in Loop: Header=BB0_88 Depth=1
	v_cvt_f32_f16_e32 v8, v123
	v_add_co_u32 v2, vcc_lo, s6, v2
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s7, v3, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_f32_e32 v24, v23, v8
	v_cmp_gt_u32_e64 s2, s13, v122
	v_cndmask_b32_e64 v23, v24, v23, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v23, v23
	global_store_b16 v[2:3], v23, off
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB0_109
; %bb.106:                              ;   in Loop: Header=BB0_88 Depth=1
	s_wait_dscnt 0x8
	v_add_f32_e32 v21, v21, v22
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v21, s8, v21
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_108
; %bb.107:                              ;   in Loop: Header=BB0_88 Depth=1
	global_load_u16 v22, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v21, s9, v22, v21 op_sel_hi:[0,1,0]
.LBB0_108:                              ;   in Loop: Header=BB0_88 Depth=1
	v_cvt_f32_f16_e32 v22, v121
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v22, v21, v22
	v_cndmask_b32_e64 v21, v22, v21, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v21, v21
	global_store_b16 v[2:3], v21, off offset:2
.LBB0_109:                              ;   in Loop: Header=BB0_88 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s30
	s_wait_dscnt 0x7
	v_add_f32_e32 v19, v19, v20
	v_add_co_u32 v0, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v19, s8, v19
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_111
; %bb.110:                              ;   in Loop: Header=BB0_88 Depth=1
	global_load_u16 v20, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v19, s9, v20, v19 op_sel_hi:[0,1,0]
.LBB0_111:                              ;   in Loop: Header=BB0_88 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v20, v19, v8
	v_add_co_u32 v2, vcc_lo, v2, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	v_cndmask_b32_e64 v19, v20, v19, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v19, v19
	global_store_b16 v[2:3], v19, off
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB0_115
; %bb.112:                              ;   in Loop: Header=BB0_88 Depth=1
	s_wait_dscnt 0x6
	v_add_f32_e32 v17, v17, v18
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v17, s8, v17
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_114
; %bb.113:                              ;   in Loop: Header=BB0_88 Depth=1
	global_load_u16 v18, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v17, s9, v18, v17 op_sel_hi:[0,1,0]
.LBB0_114:                              ;   in Loop: Header=BB0_88 Depth=1
	v_cvt_f32_f16_e32 v18, v121
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v18, v17, v18
	v_cndmask_b32_e64 v17, v18, v17, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v17, v17
	global_store_b16 v[2:3], v17, off offset:2
.LBB0_115:                              ;   in Loop: Header=BB0_88 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s30
	s_wait_dscnt 0x5
	v_add_f32_e32 v13, v13, v15
	v_add_co_u32 v0, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v13, s8, v13
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_117
; %bb.116:                              ;   in Loop: Header=BB0_88 Depth=1
	global_load_u16 v15, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v13, s9, v15, v13 op_sel_hi:[0,1,0]
.LBB0_117:                              ;   in Loop: Header=BB0_88 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v15, v13, v8
	v_add_co_u32 v2, vcc_lo, v2, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	v_cndmask_b32_e64 v13, v15, v13, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v13, v13
	global_store_b16 v[2:3], v13, off
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB0_121
; %bb.118:                              ;   in Loop: Header=BB0_88 Depth=1
	s_wait_dscnt 0x4
	v_add_f32_e32 v13, v14, v16
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v13, s8, v13
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_120
; %bb.119:                              ;   in Loop: Header=BB0_88 Depth=1
	global_load_u16 v14, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v13, s9, v14, v13 op_sel_hi:[0,1,0]
.LBB0_120:                              ;   in Loop: Header=BB0_88 Depth=1
	v_cvt_f32_f16_e32 v14, v121
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v14, v13, v14
	v_cndmask_b32_e64 v13, v14, v13, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v13, v13
	global_store_b16 v[2:3], v13, off offset:2
.LBB0_121:                              ;   in Loop: Header=BB0_88 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s30
	s_wait_dscnt 0x3
	v_add_f32_e32 v11, v11, v12
	v_add_co_u32 v0, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v11, s8, v11
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_123
; %bb.122:                              ;   in Loop: Header=BB0_88 Depth=1
	global_load_u16 v12, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v11, s9, v12, v11 op_sel_hi:[0,1,0]
.LBB0_123:                              ;   in Loop: Header=BB0_88 Depth=1
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_add_f32_e32 v12, v11, v8
	v_add_co_u32 v2, vcc_lo, v2, s20
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v3, null, s21, v3, vcc_lo
	v_cndmask_b32_e64 v11, v12, v11, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v11, v11
	global_store_b16 v[2:3], v11, off
	s_and_saveexec_b32 s30, s2
	s_cbranch_execz .LBB0_127
; %bb.124:                              ;   in Loop: Header=BB0_88 Depth=1
	s_wait_dscnt 0x2
	v_add_f32_e32 v9, v9, v10
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v9, s8, v9
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_126
; %bb.125:                              ;   in Loop: Header=BB0_88 Depth=1
	global_load_u16 v10, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v9, s9, v10, v9 op_sel_hi:[0,1,0]
.LBB0_126:                              ;   in Loop: Header=BB0_88 Depth=1
	v_cvt_f32_f16_e32 v10, v121
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_f32_e32 v10, v9, v10
	v_cndmask_b32_e64 v9, v10, v9, s0
	s_delay_alu instid0(VALU_DEP_1)
	v_cvt_f16_f32_e32 v9, v9
	global_store_b16 v[2:3], v9, off offset:2
.LBB0_127:                              ;   in Loop: Header=BB0_88 Depth=1
	s_wait_alu 0xfffe
	s_or_b32 exec_lo, exec_lo, s30
	s_wait_dscnt 0x1
	v_add_f32_e32 v6, v6, v7
	v_add_co_u32 v0, vcc_lo, v0, s10
	s_wait_alu 0xfffd
	v_add_co_ci_u32_e64 v1, null, s11, v1, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_mul_f32_e32 v6, s8, v6
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_129
; %bb.128:                              ;   in Loop: Header=BB0_88 Depth=1
	global_load_u16 v7, v[0:1], off
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v6, s9, v7, v6 op_sel_hi:[0,1,0]
.LBB0_129:                              ;   in Loop: Header=BB0_88 Depth=1
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
	s_cbranch_execz .LBB0_87
; %bb.130:                              ;   in Loop: Header=BB0_88 Depth=1
	s_wait_dscnt 0x0
	v_add_f32_e32 v4, v4, v5
	s_and_not1_b32 vcc_lo, exec_lo, s26
	s_delay_alu instid0(VALU_DEP_1)
	v_mul_f32_e32 v4, s8, v4
	s_wait_alu 0xfffe
	s_cbranch_vccnz .LBB0_86
; %bb.131:                              ;   in Loop: Header=BB0_88 Depth=1
	global_load_u16 v0, v[0:1], off offset:2
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v4, s9, v0, v4 op_sel_hi:[0,1,0]
	s_branch .LBB0_86
.LBB0_132:
	s_nop 0
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201
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
		.amdhsa_inst_pref_size 96
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
	.size	wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201, .Lfunc_end0-wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201
                                        ; -- End function
	.set wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201.num_vgpr, 147
	.set wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201.num_agpr, 0
	.set wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201.numbered_sgpr, 46
	.set wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201.num_named_barrier, 0
	.set wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201.private_seg_size, 0
	.set wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201.uses_vcc, 1
	.set wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201.uses_flat_scratch, 0
	.set wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201.has_dyn_sized_stack, 0
	.set wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201.has_recursion, 0
	.set wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 12184
; TotalNumSgprs: 48
; NumVgprs: 147
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
	.type	__hip_cuid_e97a15d6a835d584,@object ; @__hip_cuid_e97a15d6a835d584
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_e97a15d6a835d584
__hip_cuid_e97a15d6a835d584:
	.byte	0                               ; 0x0
	.size	__hip_cuid_e97a15d6a835d584, 1

	.ident	"AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.0 26014 7b800a19466229b8479a78de19143dc33c3ab9b5)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_e97a15d6a835d584
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
  AssertSizeEqual: { 1: 5, 2: 1 }
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
    .name:           wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201
    .private_segment_fixed_size: 0
    .sgpr_count:     48
    .sgpr_spill_count: 0
    .symbol:         wvSplitK_vllm_f16_tn_n5_y2u4_gfx1201.kd
    .vgpr_count:     147
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1201
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
